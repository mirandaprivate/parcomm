#!/usr/bin/env python3
"""Create an isolated benchmark workspace; never edit the original implementation."""
import hashlib
import json
import shutil
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / 'source'
if OUT.exists():
    raise SystemExit(f'Refusing to overwrite {OUT}')
OUT.mkdir()
for name in ['atomic_proof', 'batch', 'fs', 'mat', 'poly-commit']:
    src = ROOT / name
    for item in src.rglob('*'):
        if item.is_file() and item.suffix in {'.rs', '.toml', '.md'} and not {'target', '.git'}.intersection(item.parts):
            dest = OUT / item.relative_to(ROOT)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)
for name in ['Cargo.toml', 'Cargo.lock']:
    shutil.copy2(ROOT / name, OUT / name)

# Resolve only dependency snapshots against the repository, not the generated
# workspace. Keep crate member paths relative to that workspace.
manifest=OUT/'Cargo.toml'
manifest.write_text(re.sub(r'path = "((?:third_party|vendor)/[^"]+)"',
    lambda match:'path = "'+str(ROOT/match[1])+'"',manifest.read_text()))
patch=(ROOT/'.cargo/jolt-patch.toml').read_text()
patch=re.sub(r'path = "([^"]+)"',lambda match:'path = "'+str(ROOT/match[1])+'"',patch)
(OUT/'local-patch.toml').write_text(patch)
if (ROOT/'dependencies-jolt').is_dir():
    (OUT/'local-vendor.toml').write_text('[source.crates-io]\nreplace-with = "vendored-sources"\n[source.vendored-sources]\ndirectory = "'+str(ROOT/'dependencies-jolt')+'"\n')

bucket = OUT / 'batch/src/bucketed_ip.rs'
text = bucket.read_text()
assert 'bit_width == 0 || bit_width > 32' in text
text = text.replace('bit_width == 0 || bit_width > 32', 'bit_width == 0 || bit_width > F::MODULUS_BIT_SIZE as usize', 1)
text = text.replace('if ratio < 2 {', 'if ratio < 1 {', 1)
text += '\n// Isolated implicit-eq experiment; no bucket construction.\ninclude!("packed_eo_aware.inc.rs");\ninclude!("packed_eo_eq.inc.rs");\n'
bucket.write_text(text)
shutil.copy2(HERE / 'packed_eo_aware.inc.rs', bucket.with_name('packed_eo_aware.inc.rs'))
shutil.copy2(HERE / 'packed_eo_cross.inc.rs', bucket.with_name('packed_eo_cross.inc.rs'))
shutil.copy2(HERE / 'packed_eo_bytewise.inc.rs', bucket.with_name('packed_eo_bytewise.inc.rs'))
shutil.copy2(HERE / 'packed_eo_small.inc.rs', bucket.with_name('packed_eo_small.inc.rs'))
shutil.copy2(HERE / 'packed_eo_eq.inc.rs', bucket.with_name('packed_eo_eq.inc.rs'))

hidden = OUT / 'batch/src/hidden_packed_eo.rs'
text = hidden.read_text()
start = text.index('    let mut driver = HidingChallengeDriver::new(', text.index('pub fn prove_hidden_packed_shortint_reduction'))
end = text.index('\n/// Open the registered dense BVA', start)
body = text[start:end]
fold_start = body.index('    let (private_fold, counters, profile) =')
fold_end = body.index('    let output = driver', fold_start)
body = body[:fold_start] + '    let (private_fold, counters, profile) = reduction(&mut driver)\n        .map_err(HiddenPackedEoProverError::Streaming)?;\n' + body[fold_end:]
signature = '''
/// Isolated benchmark adapter: preserves the production commitment transcript
/// and BlindFold wrapper, replacing only the arithmetic reduction callback.
pub fn prove_hidden_bitwidth_reduction<E>(
    parameters: &DoryProverParameters,
    logical_len: usize,
    claimed: Fr,
    bit_width: usize,
    matrix_commitment: &DoryCommitment,
    outer_binding: &[u8],
    rng: &mut impl RngCore,
    reduction: impl FnOnce(&mut dyn PackedEoChallengeTranscript<Fr>) -> Result<
        (PackedEoLiteBulletProof<Fr>, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<E>>,
) -> Result<(HiddenPackedEoProof, HiddenPackedEoProverState, PaddingWorkCounters, StreamingEoProfile), HiddenPackedEoProverError<E>> {
    let packing_stride = packing_stride_for_bit_width::<Fr>(logical_len, bit_width)
        .map_err(|_| HiddenPackedEoProverError::Protocol(HiddenPackedEoError::Shape))?;
'''
text += signature + body
text += '''
/// Checks endpoint linkage outside the measured component, without PCS proofs.
pub fn audit_bitwidth_endpoints(state: &HiddenPackedEoProverState, values: &[Fr], dense_basis: &[Fr], stride: usize) -> bool {
    let mut witness = values.to_vec();
    let mut basis = dense_basis.to_vec();
    let challenge = state.challenges[0];
    let inverse = challenge.inverse().unwrap();
    let mut powers = vec![Fr::one(); stride];
    let mut inverses = powers.clone();
    for i in 1..stride { powers[i] = powers[i-1] * challenge; inverses[i] = inverses[i-1] * inverse; }
    witness = witness.chunks(stride).map(|row| row.iter().zip(&inverses).map(|(v,p)| *v * p).sum()).collect();
    basis = basis.chunks(stride).map(|row| row.iter().zip(&powers).map(|(v,p)| *v * p).sum()).collect();
    for challenge in &state.challenges[stride.ilog2() as usize..] {
        let inverse = challenge.inverse().unwrap();
        witness = witness.chunks_exact(2).map(|v| v[0] + inverse * v[1]).collect();
        basis = basis.chunks_exact(2).map(|v| v[0] + *challenge * v[1]).collect();
    }
    witness == [state.a_reduced.value] && basis == [state.b_reduced.value]
}
'''
hidden.write_text(text)
shutil.copy2(HERE / 'packed_eo_bitwidth_v6.rs', OUT / 'batch/examples/packed_eo_bitwidth_v6.rs')
manifest = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(OUT.rglob('*')) if p.is_file()}
(HERE / 'source-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(OUT)
