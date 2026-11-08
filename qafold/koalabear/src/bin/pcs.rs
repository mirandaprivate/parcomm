use blake2::{Blake2s256, Digest};
use mikros_qafold::{
    ShortBaseField, ShortFieldElement,
    koalabear::{KoalaBear as B, KoalaBearExt4 as F},
    packed_koalabear::PackedPcs,
};
use rand::{SeedableRng, rngs::StdRng};
use rayon::prelude::*;
use serde_json::json;
use std::{
    io::{self, Read},
    time::Instant,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 5 {
        return Err(
            "usage: pcs CHUNK_LOG LOGICAL_LOG ACTIVE_ENTRIES THREADS < signed-i16-le-bva".into(),
        );
    }
    let chunk_log: usize = args[1].parse()?;
    let logical_log: usize = args[2].parse()?;
    let active: usize = args[3].parse()?;
    let threads: usize = args[4].parse()?;
    if !(4..=30).contains(&chunk_log)
        || logical_log < chunk_log
        || logical_log >= 63
        || active > (1usize << logical_log)
    {
        return Err("invalid dimensions".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let verifier = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let start = Instant::now();
    let pcs = PackedPcs::new(chunk_log, 32, 100, [0x51; 32])?;
    println!(
        "{}",
        json!({"stage":"setup","seconds":start.elapsed().as_secs_f64(),"chunk_log":chunk_log,"logical_log":logical_log,"active_entries":active,"input_bytes":2*active,"field":"KoalaBear/X^4-3","packing_lanes":4,"threads":threads,"columns":32,"query_rf_target_bits":100,"parameter_digest":format!("{:x?}",pcs.parameter_digest())})
    );
    let mut input = io::stdin().lock();
    let mut hash = Blake2s256::new();
    let mut rng = StdRng::seed_from_u64(0x504353313642);
    let point: Vec<F> = (0..logical_log)
        .map(|_| F::sample_uniform(&mut rng))
        .collect();
    let split = logical_log - chunk_log;
    let mut global = F::zero();
    let mut commit_total = 0.0;
    let mut open_total = 0.0;
    let mut verify_total = 0.0;
    let mut proof_total = 0usize;
    let mut roots = Vec::new();
    for block in 0..(1usize << split) {
        let start = Instant::now();
        let count = active
            .saturating_sub(block << chunk_log)
            .min(1usize << chunk_log);
        let mut raw = vec![0u8; 2usize << chunk_log];
        input.read_exact(&mut raw[..2 * count])?;
        hash.update(&raw[..2 * count]);
        let values: Vec<B> = raw
            .par_chunks_exact(2)
            .map(|v| {
                let signed = i16::from_le_bytes([v[0], v[1]]) as i64;
                B::from_u64(if signed < 0 {
                    (2130706433 + signed) as u64
                } else {
                    signed as u64
                })
            })
            .collect();
        drop(raw);
        let input_seconds = start.elapsed().as_secs_f64();
        println!(
            "{}",
            json!({"stage":"commit_start","block":block,"input_seconds":input_seconds,"active_entries":count})
        );
        let start = Instant::now();
        let data = pcs.commit(values)?;
        let commit_seconds = start.elapsed().as_secs_f64();
        let commit_profile = data.commit_timings();
        let root = data.commitment();
        roots.push(format!("{:x?}", root.root));
        println!(
            "{}",
            json!({"stage":"open_start","block":block,"commit_seconds":commit_seconds,"packing_ms":commit_profile.packing_ms,"mobius_ms":commit_profile.mobius_ms,"encoding_ms":commit_profile.encoding_ms,"merkle_ms":commit_profile.merkle_ms})
        );
        let context =
            format!("parcomm/qafold-koalabear-i16-pcs-only/v1/{logical_log}/{active}/{block}");
        let start = Instant::now();
        let proof = pcs.open(data, &point[split..], context.as_bytes())?;
        let bytes = proof.to_bytes()?;
        let opening_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        verifier.install(|| {
            pcs.verify(
                &root,
                &point[split..],
                proof.claimed_value,
                &bytes,
                context.as_bytes(),
            )
        })?;
        let verify_seconds = start.elapsed().as_secs_f64();
        let weight = point[..split]
            .iter()
            .enumerate()
            .fold(F::one(), |w, (coord, &p)| {
                w * if (block >> (split - 1 - coord)) & 1 == 1 {
                    p
                } else {
                    F::one() - p
                }
            });
        global += weight * proof.claimed_value;
        commit_total += commit_seconds;
        open_total += opening_seconds;
        verify_total += verify_seconds;
        proof_total += bytes.len();
        println!(
            "{}",
            json!({"stage":"block_verified","block":block,"commit_seconds":commit_seconds,"opening_decide_serialization_seconds":opening_seconds,"verify_seconds":verify_seconds,"proof_bytes":bytes.len(),"proof_blake2s":format!("{:x}",Blake2s256::digest(&bytes)),"projection_ms":proof.timings.projection_ms,"sumcheck_ms":proof.timings.sumcheck_ms,"native_opening_ms":proof.timings.native_opening_ms,"decide_ms":proof.timings.decide_ms})
        );
    }
    let mut extra = [0u8; 1];
    if input.read(&mut extra)? != 0 {
        return Err("input exceeds declared active bytes".into());
    }
    println!(
        "{}",
        json!({"stage":"complete","verified":true,"scope":"PCS only; sequential chunk commitments, fixed benchmark point, no ParComm/ZK claim","chunks":roots.len(),"roots":roots,"global_evaluation":format!("{:?}",global),"input_blake2s":format!("{:x}",hash.finalize()),"commit_seconds":commit_total,"opening_seconds":open_total,"prover_seconds":commit_total+open_total,"verify_seconds":verify_total,"proof_bytes":proof_total,"commitment_bytes":32*(1usize<<split)})
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn all_i16_bit_patterns_embed_injectively() {
        let mut canonical = std::collections::HashSet::new();
        for bits in 0u32..=65535 {
            let signed = i16::from_le_bytes((bits as u16).to_le_bytes()) as i64;
            let encoded = B::from_u64(signed.rem_euclid(2130706433) as u64).to_canonical_u64();
            assert!(canonical.insert(encoded));
            assert_eq!(
                if encoded > 32767 {
                    encoded as i64 - 2130706433
                } else {
                    encoded as i64
                },
                signed
            );
        }
    }
    #[test]
    fn signed_input_chunk_openings_match_full_mle_and_reject_wrong_claim() {
        let pcs = PackedPcs::new(10, 32, 100, [0x51; 32]).unwrap();
        let mut rng = StdRng::seed_from_u64(99);
        let point: Vec<F> = (0..12).map(|_| F::sample_uniform(&mut rng)).collect();
        let base: Vec<B> = (0..4096)
            .map(|i| {
                let signed = if i < 3500 {
                    (i * 17) as u16 as i16 as i64
                } else {
                    0
                };
                B::from_u64(signed.rem_euclid(2130706433) as u64)
            })
            .collect();
        let mut reference: Vec<F> = base
            .iter()
            .map(|&v| F::from_coefficients([v, B::zero(), B::zero(), B::zero()]))
            .collect();
        for &p in point.iter().rev() {
            reference = reference
                .chunks_exact(2)
                .map(|v| v[0] * (F::one() - p) + v[1] * p)
                .collect();
        }
        let mut aggregate = F::zero();
        for (block, values) in base.chunks_exact(1024).enumerate() {
            let data = pcs.commit(values.to_vec()).unwrap();
            let root = data.commitment();
            let context = block.to_le_bytes();
            let proof = pcs.open(data, &point[2..], &context).unwrap();
            let bytes = proof.to_bytes().unwrap();
            pcs.verify(&root, &point[2..], proof.claimed_value, &bytes, &context)
                .unwrap();
            assert!(
                pcs.verify(
                    &root,
                    &point[2..],
                    proof.claimed_value + F::one(),
                    &bytes,
                    &context
                )
                .is_err()
            );
            let weight = point[..2]
                .iter()
                .enumerate()
                .fold(F::one(), |w, (coord, &p)| {
                    w * if (block >> (1 - coord)) & 1 == 1 {
                        p
                    } else {
                        F::one() - p
                    }
                });
            aggregate += weight * proof.claimed_value;
        }
        assert_eq!(aggregate, reference[0]);
    }
}
