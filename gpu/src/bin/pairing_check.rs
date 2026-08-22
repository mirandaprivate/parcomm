//! Checks the device multi-Miller loop against arkworks and times both.

use std::env;

use ark_bls12_381::{G1Projective, G2Projective};
use ark_ec::CurveGroup;
use ark_ff::UniformRand;
use ark_std::rand::SeedableRng;
use parcomm_gpu::{benchmark_against_host, initialize_cuda_backend};

fn main() {
    initialize_cuda_backend().expect("load CUDA backend");
    let device_id = env::args()
        .nth(1)
        .and_then(|value| value.parse().ok())
        .unwrap_or(0_i32);

    let mut rng = ark_std::rand::rngs::StdRng::seed_from_u64(11);
    for log_count in [0_usize, 1, 4, 10, 13, 15] {
        let count = 1_usize << log_count;
        let left = (0..count)
            .map(|_| G1Projective::rand(&mut rng).into_affine())
            .collect::<Vec<_>>();
        let right = (0..count)
            .map(|_| G2Projective::rand(&mut rng).into_affine())
            .collect::<Vec<_>>();

        let (device_seconds, host_seconds, matches) =
            benchmark_against_host(device_id, &left, &right).expect("device multi-Miller loop");
        println!(
            "count={count} matches={matches} device_seconds={device_seconds:.6} \
             host_seconds={host_seconds:.6} speedup={:.2}",
            host_seconds / device_seconds.max(f64::MIN_POSITIVE)
        );
        assert!(matches, "device multi-Miller loop disagreed at {count} pairs");
    }
}
