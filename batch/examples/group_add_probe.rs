//! Measures the group-addition throughput that bounds the streaming commitment.

use std::time::Instant;

use ark_bls12_381::{Bls12_381, G1Affine, G1Projective};
use ark_ec::CurveGroup;
use ark_ff::UniformRand;
use ark_std::Zero;
use poly_commit::smart_pc::SmartPC;
use rayon::prelude::*;

fn main() {
    let threads = rayon::current_num_threads();
    let mut rng = ark_std::rand::thread_rng();
    let pp = SmartPC::<Bls12_381>::setup(10, &mut rng).expect("setup");
    let bases = G1Projective::normalize_batch(&pp.vec_g);
    let rounds = 1_usize << 21;

    let started = Instant::now();
    let mixed = (0..threads)
        .into_par_iter()
        .map(|_| {
            let mut accumulator = G1Projective::rand(&mut ark_std::rand::thread_rng());
            for round in 0..rounds {
                accumulator += bases[round % bases.len()];
            }
            accumulator
        })
        .reduce(G1Projective::zero, |left, right| left + right);
    let mixed_seconds = started.elapsed().as_secs_f64();

    let projective = pp.vec_g.clone();
    let started = Instant::now();
    let full = (0..threads)
        .into_par_iter()
        .map(|_| {
            let mut accumulator = G1Projective::rand(&mut ark_std::rand::thread_rng());
            for round in 0..rounds {
                accumulator += projective[round % projective.len()];
            }
            accumulator
        })
        .reduce(G1Projective::zero, |left, right| left + right);
    let projective_seconds = started.elapsed().as_secs_f64();

    assert!(!mixed.is_zero() || full.is_zero());
    println!("threads={threads}");
    println!("adds_per_thread={rounds}");
    println!(
        "mixed_add_nanoseconds={:.1}",
        mixed_seconds * 1e9 / rounds as f64
    );
    println!(
        "projective_add_nanoseconds={:.1}",
        projective_seconds * 1e9 / rounds as f64
    );
    println!("mixed_add_size={}", size_of::<G1Affine>());
}
