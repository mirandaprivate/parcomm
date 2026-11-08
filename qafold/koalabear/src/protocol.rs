use crate::{
    Result, check,
    eo::{self, EoProof},
    field::*,
    layout::{Layout, Shape},
    tbv::{self, Claim},
    transcript::Transcript,
};
use bincode::Options;
use mikros_qafold::{
    packed_koalabear::{PackedCommitWorkspace, PackedData, PackedPcs},
    qafold_short_pc::ShortQaFoldCommitment,
};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::time::Instant;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Config {
    pub chunk_log: usize,
    pub packed_log: usize,
    pub query_rf_bits: usize,
}
impl Default for Config {
    fn default() -> Self {
        Self {
            chunk_log: 30,
            packed_log: 4,
            query_rf_bits: 100,
        }
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Commitment {
    pub version: u32,
    pub config: Config,
    pub shapes: Vec<Shape>,
    pub parameter_digest: [u8; 32],
    pub roots: Vec<[u8; 32]>,
}
pub struct Parameters {
    pub layout: Layout,
    pub config: Config,
    pcs: PackedPcs,
    pub block_len: usize,
}
pub struct Committed {
    pub statement: Commitment,
    values: Vec<i16>,
    states: Vec<PackedData>,
}
impl Committed {
    pub fn values(&self) -> &[i16] {
        &self.values
    }
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BlockOpening {
    pub value: Scalar,
    pub proof: Vec<u8>,
}
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Proof {
    pub eo: EoProof,
    pub openings: Vec<BlockOpening>,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ProverProfile {
    pub native_openings: usize,
    pub terminal_batch_folding_seconds: f64,
    pub terminal_batch_binding_seconds: f64,
    pub conversion_seconds: f64,
    pub packed_eo_seconds: f64,
    pub pcs_open_seconds: f64,
    pub eo: eo::EoTimings,
    pub pcs_projection_seconds: f64,
    pub pcs_reconstruction_seconds: f64,
    pub pcs_sumcheck_seconds: f64,
    pub pcs_native_opening_seconds: f64,
    pub pcs_decide_seconds: f64,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct VerifierProfile {
    pub native_openings: usize,
    pub authenticated_roots: usize,
    pub conversion_eo_seconds: f64,
    pub pcs_seconds: f64,
}
const MAX_PROOF_BYTES: u64 = 256 * 1024 * 1024;
impl Proof {
    pub fn to_bytes(&self) -> Result<Vec<u8>> {
        Ok(bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_limit(MAX_PROOF_BYTES)
            .serialize(self)?)
    }
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        check(
            bytes.len() as u64 <= MAX_PROOF_BYTES,
            "oversize ParComm proof",
        )?;
        Ok(bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .with_limit(MAX_PROOF_BYTES)
            .reject_trailing_bytes()
            .deserialize(bytes)?)
    }
}
impl Parameters {
    pub fn new(shapes: &[Shape], config: Config) -> Result<Self> {
        check(
            (10..=30).contains(&config.chunk_log)
                && (1..=6).contains(&config.packed_log)
                && (32..=120).contains(&config.query_rf_bits),
            "invalid ParComm config",
        )?;
        let layout = Layout::new(shapes)?;
        let chunk_log = config.chunk_log.min(layout.logical.ilog2() as usize);
        let block_len = 1usize << chunk_log;
        check(
            layout.logical / block_len <= 128,
            "at most 128 PCS blocks are supported",
        )?;
        let pcs = PackedPcs::new(chunk_log, 32, config.query_rf_bits, [0x51; 32])?;
        Ok(Self {
            layout,
            config,
            pcs,
            block_len,
        })
    }
    pub fn commit(&self, values: Vec<i16>) -> Result<Committed> {
        self.layout.validate_values(&values)?;
        self.commit_unchecked(values)
    }
    fn commit_unchecked(&self, values: Vec<i16>) -> Result<Committed> {
        let mut states = Vec::new();
        let mut roots = Vec::new();
        for block in 0..self.layout.logical / self.block_len {
            let block_start = Instant::now();
            let source = |index| small(values.get(block * self.block_len + index).copied().unwrap_or(0));
            let state = if self.layout.logical > self.block_len {
                self.pcs.commit_from_fn(source)?
            } else {
                self.pcs.commit((0..self.block_len).into_par_iter().map(source).collect())?
            };
            roots.push(state.commitment().root);
            states.push(state);
            eprintln!(
                "stage=commit_block block={block} seconds={}",
                block_start.elapsed().as_secs_f64()
            );
        }
        let statement = Commitment {
            version: 2,
            config: self.config.clone(),
            shapes: self.layout.shapes.clone(),
            parameter_digest: self.pcs.parameter_digest(),
            roots,
        };
        Ok(Committed {
            statement,
            values,
            states,
        })
    }
    /// Keep roots and compact input only. All later reconstruction is online proving.
    pub fn commit_recomputed(&self, values: Vec<i16>) -> Result<Committed> {
        self.layout.validate_values(&values)?;
        if self.layout.logical <= self.block_len { return self.commit_unchecked(values); }
        let mut roots = Vec::new();
        let mut zero_root = None;
        let mut workspace = PackedCommitWorkspace::default();
        for block in 0..self.layout.logical / self.block_len {
            if block * self.block_len >= values.len() {
                if let Some(root) = zero_root { roots.push(root); continue; }
            }
            let state = self.pcs.commit_from_fn_reusing(|index|
                small(values.get(block * self.block_len + index).copied().unwrap_or(0)), &mut workspace)?;
            let root = state.commitment().root;
            if block * self.block_len >= values.len() { zero_root = Some(root); }
            roots.push(root);
            state.recycle(&mut workspace);
        }
        // EO must not overlap the reusable original-block allocation.
        drop(workspace);
        Ok(Committed { statement: Commitment {
            version: 2, config: self.config.clone(), shapes: self.layout.shapes.clone(),
            parameter_digest: self.pcs.parameter_digest(), roots,
        }, values, states: Vec::new() })
    }
    fn transcript(&self, statement: &Commitment, claims: &[Claim]) -> Result<Transcript> {
        check(
            statement.version == 2
                && statement.config == self.config
                && statement.shapes == self.layout.shapes
                && statement.parameter_digest == self.pcs.parameter_digest()
                && statement.roots.len() == self.layout.logical / self.block_len,
            "commitment/parameter mismatch",
        )?;
        tbv::validate_claims(&self.layout, claims)?;
        let mut transcript = Transcript::new();
        transcript.object(b"collection", statement)?;
        transcript.object(b"claims", &claims)?;
        Ok(transcript)
    }
    pub fn prove(&self, committed: Committed, claims: &[Claim]) -> Result<(Proof, ProverProfile)> {
        let mut profile = ProverProfile::default();
        let start = Instant::now();
        let mut transcript = self.transcript(&committed.statement, claims)?;
        let query = tbv::compile(&self.layout, claims, &mut transcript)?;
        profile.conversion_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let (eo, terminal, timings) = eo::prove(
            &committed.values,
            self.layout.logical,
            1 << self.config.packed_log,
            &query,
            &mut transcript,
        )?;
        profile.packed_eo_seconds = start.elapsed().as_secs_f64();
        profile.eo = timings;
        let start = Instant::now();
        let context = opening_context(transcript.state(), 0);
        let native_values = committed.values;
        let (actual, wire, timings, folding_ms, binding_ms) = if committed.statement.roots.len() > 1 {
            let source = |block, index| small(native_values.get(block * self.block_len + index).copied().unwrap_or(0));
            let opened = if committed.states.is_empty() {
                let roots = committed.statement.roots.iter().map(|&root| ShortQaFoldCommitment { root }).collect();
                self.pcs.open_batch_recomputed_from_fn(roots, &terminal.point_msb, &context, source)?
            } else {
                self.pcs.open_batch_from_fn(committed.states, &terminal.point_msb, &context, source)?
            };
            let wire = opened.to_bytes()?;
            (
                opened.claimed_value,
                wire,
                opened.timings,
                opened.folding_ms,
                opened.binding_ms,
            )
        } else {
            drop(native_values);
            let opened = self.pcs.open(
                committed.states.into_iter().next().unwrap(),
                &terminal.point_msb,
                &context,
            )?;
            let wire = opened.to_bytes()?;
            (opened.claimed_value, wire, opened.timings, 0.0, 0.0)
        };
        profile.pcs_projection_seconds = timings.projection_ms / 1000.0;
        profile.pcs_reconstruction_seconds = timings.reconstruction_ms / 1000.0;
        profile.pcs_sumcheck_seconds = timings.sumcheck_ms / 1000.0;
        profile.pcs_native_opening_seconds = timings.native_opening_ms / 1000.0;
        profile.pcs_decide_seconds = timings.decide_ms / 1000.0;
        profile.terminal_batch_folding_seconds = folding_ms / 1000.0;
        profile.terminal_batch_binding_seconds = binding_ms / 1000.0;
        profile.native_openings = 1;
        let openings = vec![BlockOpening {
            value: actual.into(),
            proof: wire,
        }];
        check(
            actual * terminal.scale == terminal.value,
            "QAFold/EO terminal mismatch",
        )?;
        profile.pcs_open_seconds = start.elapsed().as_secs_f64();
        Ok((Proof { eo, openings }, profile))
    }
    /// Witness-free verifier: only parameters, persistent public commitment,
    /// caller-supplied evaluation claims, and the serialized proof are needed.
    pub fn verify(
        &self,
        statement: &Commitment,
        claims: &[Claim],
        proof: &Proof,
    ) -> Result<VerifierProfile> {
        let start = Instant::now();
        let mut transcript = self.transcript(statement, claims)?;
        check(proof.openings.len() == 1, "PCS opening count")?;
        let query = tbv::compile(&self.layout, claims, &mut transcript)?;
        let terminal = eo::verify(
            &proof.eo,
            self.layout.logical,
            1 << self.config.packed_log,
            &query,
            &mut transcript,
        )?;
        let conversion_eo_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let opening = &proof.openings[0];
        let actual = opening.value.field()?;
        let context = opening_context(transcript.state(), 0);
        let roots = statement
            .roots
            .iter()
            .map(|&root| ShortQaFoldCommitment { root })
            .collect::<Vec<_>>();
        if roots.len() > 1 {
            self.pcs.verify_batch(
                &roots,
                &terminal.point_msb,
                actual,
                &opening.proof,
                &context,
            )?;
        } else {
            self.pcs.verify(
                &roots[0],
                &terminal.point_msb,
                actual,
                &opening.proof,
                &context,
            )?;
        }
        check(
            actual * terminal.scale == terminal.value,
            "PCS terminal aggregate rejected",
        )?;
        Ok(VerifierProfile {
            native_openings: 1,
            authenticated_roots: roots.len(),
            conversion_eo_seconds,
            pcs_seconds: start.elapsed().as_secs_f64(),
        })
    }
}
fn opening_context(state: [u8; 32], block: usize) -> Vec<u8> {
    let mut bytes = b"parcomm/koalabear-i16/terminal/v2".to_vec();
    bytes.extend_from_slice(&state);
    bytes.extend_from_slice(&(block as u64).to_le_bytes());
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn padding_is_a_proved_predicate_not_only_a_local_check() {
        let params = Parameters::new(&[Shape { rows: 31, cols: 31 }], Config::default()).unwrap();
        let values = params.layout.encode(&[vec![1; 31 * 31]]).unwrap();
        let point = (vec![F::from_u64(3); 5], vec![F::from_u64(7); 5]);
        let claims = tbv::evaluate_claims(&params.layout, &values, &[point]).unwrap();
        let mut corrupt = values;
        let bucket = params
            .layout
            .buckets
            .iter()
            .find(|bucket| bucket.rows > bucket.physical_rows || bucket.cols > bucket.physical_cols)
            .unwrap();
        let index = if bucket.cols > bucket.physical_cols {
            bucket.offset + bucket.physical_cols
        } else {
            bucket.offset + bucket.physical_rows * bucket.cols
        };
        corrupt[index] = 1;
        assert!(params.commit(corrupt.clone()).is_err());
        let committed = params.commit_unchecked(corrupt).unwrap();
        assert!(params.prove(committed, &claims).is_err());
    }
    #[test]
    fn implicit_suffix_is_checked_by_the_same_relation() {
        let params = Parameters::new(&[Shape { rows: 3, cols: 5 }], Config::default()).unwrap();
        let mut values = params.layout.encode(&[vec![2; 15]]).unwrap();
        let claims = tbv::evaluate_claims(
            &params.layout,
            &values,
            &[(vec![F::from_u64(3); 2], vec![F::from_u64(7); 3])],
        )
        .unwrap();
        values.resize(params.layout.logical, 0);
        values[params.layout.logical - 1] = 1;
        let committed = params.commit_unchecked(values).unwrap();
        assert!(params.prove(committed, &claims).is_err());
    }
}
