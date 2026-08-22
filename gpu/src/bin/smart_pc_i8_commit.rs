use std::env;
use std::slice;
use std::time::Instant;

use ark_bls12_381::{Bls12_381, Fr, G1Projective};
use ark_ff::Zero;
use poly_commit::smart_pc::{utils::msm_g1, SmartPC, UniversalParams};
use ark_serialize::CanonicalSerialize;
use rand::{RngCore, SeedableRng};
use rand_chacha::ChaCha20Rng;
use sha2::{Digest, Sha256};
use parcomm_gpu::{commit_i8_four_gpu, initialize_cuda_backend, FiatShamir};

fn parse_usize(flag: &str, default: usize) -> usize {
    let arguments: Vec<String> = env::args().collect();
    arguments
        .windows(2)
        .find(|pair| pair[0] == flag)
        .and_then(|pair| pair[1].parse().ok())
        .unwrap_or(default)
}

fn as_bytes(values: &[i8]) -> &[u8] {
    // Every bit pattern is valid for both i8 and u8, and both types have alignment one.
    unsafe { slice::from_raw_parts(values.as_ptr().cast(), values.len()) }
}

fn as_bytes_mut(values: &mut [i8]) -> &mut [u8] {
    // Every bit pattern is valid for both i8 and u8, and both types have alignment one.
    unsafe { slice::from_raw_parts_mut(values.as_mut_ptr().cast(), values.len()) }
}

fn hex_string(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn reference_column(
    pp: &UniversalParams<Bls12_381>,
    matrix: &[i8],
    rows: usize,
    column: usize,
) -> G1Projective {
    let scalars = matrix[column * rows..(column + 1) * rows]
        .iter()
        .map(|value| {
            if *value >= 0 {
                Fr::from(*value as u64)
            } else {
                -Fr::from(value.unsigned_abs() as u64)
            }
        })
        .collect::<Vec<_>>();
    msm_g1::<Bls12_381>(&pp.vec_g[..rows], &scalars)
}

fn main() {
    let log_elements = parse_usize("--log-elements", 30);
    let chunk_columns = parse_usize("--chunk-columns", 256);
    let row_log = log_elements / 2;
    let column_log = log_elements - row_log;
    let rows = 1_usize
        .checked_shl(row_log as u32)
        .expect("row count does not fit usize");
    let columns = 1_usize
        .checked_shl(column_log as u32)
        .expect("column count does not fit usize");
    let elements = rows
        .checked_mul(columns)
        .expect("matrix element count overflow");
    assert!(columns % 4 == 0, "columns must be divisible by four");

    println!("backend=ICICLE-v4-GPU-i8-MSM-plus-Arkworks-CPU-pairing");
    println!("curve=BLS12-381");
    println!("gpu_count=4");
    println!("log_elements={log_elements}");
    println!("rows={rows}");
    println!("columns={columns}");
    println!("elements={elements}");
    println!("input_type=signed_i8");
    println!("msm_bitsize=8");
    println!("signed_strategy=positive_MSM_minus_negative_MSM");
    println!("chunk_columns={chunk_columns}");

    let setup_log = row_log.max(column_log);
    let mut setup_rng = ChaCha20Rng::seed_from_u64(0x4938_534d_4152_5450);
    let setup_start = Instant::now();
    let pp = SmartPC::<Bls12_381>::setup(setup_log, &mut setup_rng).expect("SmartPC setup failed");
    println!("setup_seconds={:.6}", setup_start.elapsed().as_secs_f64());

    let generation_start = Instant::now();
    let mut transcript = FiatShamir::new();
    transcript.push(&(log_elements as u64));
    transcript.push(&(rows as u64));
    transcript.push(&(columns as u64));
    let data_seed = transcript.state();
    let hiding_factor = transcript.challenge::<Fr>();
    let mut matrix = vec![0_i8; elements];
    let mut data_rng = ChaCha20Rng::from_seed(data_seed);
    data_rng.fill_bytes(as_bytes_mut(&mut matrix));
    let input_digest: [u8; 32] = Sha256::digest(as_bytes(&matrix)).into();
    println!(
        "input_generation_seconds={:.6}",
        generation_start.elapsed().as_secs_f64()
    );
    println!("input_resident_bytes={}", matrix.len());
    println!("raw_input_bytes_per_gpu={}", matrix.len() / 4);
    println!("input_sha256={}", hex_string(&input_digest));
    println!("hiding_factor_nonzero={}", !hiding_factor.is_zero());

    initialize_cuda_backend().expect("failed to initialize the ICICLE CUDA backend");

    let warmup_columns = 4;
    let warmup_start = Instant::now();
    let warmup = commit_i8_four_gpu(
        &pp,
        &matrix[..warmup_columns * rows],
        rows,
        warmup_columns,
        hiding_factor,
        [0, 1, 2, 3],
        1,
    )
    .expect("four-GPU i8 warmup failed");
    println!(
        "warmup_wall_seconds={:.6}",
        warmup_start.elapsed().as_secs_f64()
    );
    for column in 0..warmup_columns {
        let reference = reference_column(&pp, &matrix, rows, column);
        let matches = reference == warmup.tier_one[column];
        println!("warmup_column={column} match={matches}");
        assert!(matches, "warmup tier-one column differs from Arkworks");
    }

    let commit_start = Instant::now();
    let commitment = commit_i8_four_gpu(
        &pp,
        &matrix,
        rows,
        columns,
        hiding_factor,
        [0, 1, 2, 3],
        chunk_columns,
    )
    .expect("four-GPU i8 SmartPC commit failed");
    let experiment_wall_seconds = commit_start.elapsed().as_secs_f64();
    println!("experiment_wall_seconds={experiment_wall_seconds:.6}");
    println!(
        "commit_total_seconds={:.6}",
        commitment.profile.total_seconds
    );
    println!(
        "parallel_stage_seconds={:.6}",
        commitment.profile.parallel_stage_seconds
    );
    println!(
        "g1_conversion_seconds={:.6}",
        commitment.profile.g1_conversion_seconds
    );
    println!(
        "constant_merge_seconds={:.6}",
        commitment.profile.constant_merge_seconds
    );
    for shard in &commitment.profile.shards {
        println!(
            "gpu={} columns={}..{} raw_input_bytes={} scalar_scratch_bytes={} raw_upload_seconds={:.6} scalar_expansion_seconds={:.6} positive_msm_seconds={:.6} negative_msm_seconds={:.6} download_seconds={:.6} cpu_pairing_seconds={:.6} total_seconds={:.6}",
            shard.device_id,
            shard.column_start,
            shard.column_end,
            shard.raw_input_bytes,
            shard.scalar_scratch_bytes,
            shard.raw_upload_seconds,
            shard.scalar_expansion_seconds,
            shard.positive_msm_seconds,
            shard.negative_msm_seconds,
            shard.result_download_seconds,
            shard.cpu_pairing_seconds,
            shard.total_seconds,
        );
    }

    let sample_start = Instant::now();
    let mut sample_columns = vec![0, columns / 4, columns / 2, 3 * columns / 4, columns - 1];
    sample_columns.sort_unstable();
    sample_columns.dedup();
    for column in sample_columns {
        let reference = reference_column(&pp, &matrix, rows, column);
        let matches = reference == commitment.tier_one[column];
        println!("sample_column={column} match={matches}");
        assert!(matches, "sampled tier-one column differs from Arkworks");
    }
    println!(
        "sample_verification_seconds={:.6}",
        sample_start.elapsed().as_secs_f64()
    );

    let mut commitment_bytes = Vec::new();
    commitment
        .commitment
        .serialize_compressed(&mut commitment_bytes)
        .expect("commitment serialization failed");
    let commitment_digest: [u8; 32] = Sha256::digest(&commitment_bytes).into();
    let mut tier_one_hasher = Sha256::new();
    for point in &commitment.tier_one {
        let mut point_bytes = Vec::new();
        point
            .serialize_compressed(&mut point_bytes)
            .expect("tier-one serialization failed");
        tier_one_hasher.update(point_bytes);
    }
    let tier_one_digest: [u8; 32] = tier_one_hasher.finalize().into();
    println!("commitment_sha256={}", hex_string(&commitment_digest));
    println!("tier_one_sha256={}", hex_string(&tier_one_digest));
    println!("experiment_complete=true");
}
