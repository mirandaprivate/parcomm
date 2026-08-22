use std::env;
use std::time::Instant;

use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::Zero;
use poly_commit::smart_pc::SmartPC;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use parcomm_gpu::{commit_full_four_gpu, initialize_cuda_backend, FiatShamir};

fn parse_usize(flag: &str, default: usize) -> usize {
    let arguments: Vec<String> = env::args().collect();
    arguments
        .windows(2)
        .find(|pair| pair[0] == flag)
        .and_then(|pair| pair[1].parse().ok())
        .unwrap_or(default)
}

fn main() {
    let log_q = parse_usize("--log-q", 8);
    let repeats = parse_usize("--repeats", 3);
    let q = 1_usize << log_q;
    assert!(q % 4 == 0, "q must be divisible by four");

    println!("backend=ICICLE-v4-GPU-MSM-plus-Arkworks-CPU-pairing");
    println!("curve=BLS12-381");
    println!("gpu_count=4");
    println!("rows={q}");
    println!("columns={q}");
    println!("second_tier=CPU_pairing_four_parallel_shards");

    let mut rng = ChaCha20Rng::seed_from_u64(0x5a4d_4152_5450_4301);
    let setup_start = Instant::now();
    let pp = SmartPC::<Bls12_381>::setup(log_q, &mut rng).expect("SmartPC setup failed");
    println!("setup_seconds={:.6}", setup_start.elapsed().as_secs_f64());

    let fiat_shamir_start = Instant::now();
    let mut transcript = FiatShamir::new();
    transcript.push(&0x4750_5553_4d41_5254_u64);
    let mut matrix = Vec::with_capacity(q);
    for _ in 0..q {
        let mut column = Vec::with_capacity(q);
        for _ in 0..q {
            column.push(transcript.challenge::<Fr>());
        }
        matrix.push(column);
    }
    let hiding_factor = transcript.challenge::<Fr>();
    let transcript_state = transcript.state();
    println!(
        "fiat_shamir_seconds={:.6}",
        fiat_shamir_start.elapsed().as_secs_f64()
    );
    println!("fiat_shamir_state={}", hex_string(&transcript_state));
    println!(
        "matrix_resident_bytes={}",
        q * q * std::mem::size_of::<Fr>()
    );
    println!(
        "scalar_bytes_per_gpu={}",
        q * (q / 4) * std::mem::size_of::<Fr>()
    );

    let cpu_start = Instant::now();
    let (cpu_commitment, cpu_tier_one) =
        SmartPC::<Bls12_381>::commit_full(&pp, &matrix, hiding_factor)
            .expect("CPU SmartPC commit failed");
    println!(
        "cpu_commit_seconds={:.6}",
        cpu_start.elapsed().as_secs_f64()
    );

    initialize_cuda_backend().expect("failed to initialize the ICICLE CUDA backend");
    let warmup = commit_full_four_gpu(&pp, &matrix, hiding_factor, [0, 1, 2, 3])
        .expect("four-GPU SmartPC warmup failed");
    let warmup_tier_one_match = warmup.tier_one == cpu_tier_one;
    let warmup_commitment_match = warmup.commitment == cpu_commitment;
    println!(
        "warmup_hybrid_seconds={:.6}",
        warmup.profile.hybrid_commit_seconds
    );
    println!("warmup_tier_one_match={warmup_tier_one_match}");
    println!("warmup_commitment_match={warmup_commitment_match}");
    assert!(
        warmup_tier_one_match,
        "GPU warmup tier-one commitments differ from Arkworks"
    );
    assert!(
        warmup_commitment_match,
        "GPU warmup final commitment differs from Arkworks"
    );

    let mut hybrid_times = Vec::with_capacity(repeats);
    for repeat in 0..repeats {
        let gpu = commit_full_four_gpu(&pp, &matrix, hiding_factor, [0, 1, 2, 3])
            .expect("four-GPU SmartPC commit failed");
        let tier_one_match = gpu.tier_one == cpu_tier_one;
        let commitment_match = gpu.commitment == cpu_commitment;
        println!("repeat={repeat}");
        println!(
            "hybrid_commit_seconds={:.6}",
            gpu.profile.hybrid_commit_seconds
        );
        println!(
            "parallel_stage_seconds={:.6}",
            gpu.profile.parallel_stage_seconds
        );
        println!(
            "gpu_g1_conversion_seconds={:.6}",
            gpu.profile.g1_conversion_seconds
        );
        println!(
            "cpu_constant_merge_seconds={:.6}",
            gpu.profile.constant_merge_seconds
        );
        for shard in &gpu.profile.shards {
            println!(
                "gpu={} columns={}..{} scalar_bytes={} upload_seconds={:.6} tier_one_msm_seconds={:.6} download_seconds={:.6} cpu_pairing_seconds={:.6} total_seconds={:.6}",
                shard.device_id,
                shard.column_start,
                shard.column_end,
                shard.scalar_bytes,
                shard.upload_seconds,
                shard.tier_one_msm_seconds,
                shard.tier_one_download_seconds,
                shard.cpu_pairing_seconds,
                shard.total_seconds,
            );
        }
        println!("tier_one_match={tier_one_match}");
        println!("commitment_match={commitment_match}");
        assert!(
            tier_one_match,
            "GPU tier-one commitments differ from Arkworks"
        );
        assert!(
            commitment_match,
            "GPU final commitment differs from Arkworks"
        );
        hybrid_times.push(gpu.profile.hybrid_commit_seconds);
    }

    hybrid_times.sort_by(f64::total_cmp);
    let median = hybrid_times[hybrid_times.len() / 2];
    println!("hybrid_median_seconds={median:.6}");
    println!("hiding_factor_nonzero={}", !hiding_factor.is_zero());
    println!("experiment_complete=true");
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
