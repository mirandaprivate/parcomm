#![allow(warnings)]

extern crate rand;
extern crate criterion;

use criterion::{criterion_group, criterion_main, Criterion};

use ark_std::{test_rng, UniformRand};
use rand_chacha::{
    rand_core::{RngCore, SeedableRng},
    ChaCha20Rng,
};


use ark_bls12_381::Bls12_381;
use ark_ec::pairing::Pairing;

use ark_std::ops::{Add, Mul};


// #[ignore = "only for testing bls12_381"]
fn bench_scalar_mul_scalar(b: &mut criterion::Bencher){
    let rng = &mut ChaCha20Rng::from_rng(test_rng()).unwrap();
    let l_scalar =  <Bls12_381 as Pairing>::ScalarField::rand(rng);
    let r_scalar =  <Bls12_381 as Pairing>::ScalarField::rand(rng);
    b.iter(|| l_scalar.mul(&r_scalar) );
}

fn bench_scalar_add_scalar(b: &mut criterion::Bencher){
    let rng = &mut ChaCha20Rng::from_rng(test_rng()).unwrap();
    let l_scalar =  <Bls12_381 as Pairing>::ScalarField::rand(rng);
    let r_scalar =  <Bls12_381 as Pairing>::ScalarField::rand(rng);
    b.iter(|| l_scalar.add(&r_scalar) );
}

pub fn criterion_benchmark(c: &mut Criterion) {
    c.bench_function("scalar_mul_scalar", |b| bench_scalar_mul_scalar(b));
    c.bench_function("scalar_add_scalar", |b| bench_scalar_add_scalar(b));
}

// Define the benchmark group and main function
criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);


