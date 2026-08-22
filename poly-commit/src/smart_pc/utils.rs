#![deny(trivial_numeric_casts, variant_size_differences)]
/// This module contains utility functions for the smart polynomial commitment scheme.
use ark_ec::{
    pairing::{Pairing, PairingOutput},
    scalar_mul::{BatchMulPreprocessing, ScalarMul},
    AffineRepr, CurveGroup, VariableBaseMSM, PrimeGroup, AdditiveGroup,
};
use ark_ff::{Field, UniformRand, PrimeField};
use ark_std::{Zero, One,
    ops::{Add, Mul},
    vec::Vec,
};

use rayon::iter::IntoParallelIterator;

/// Number of consecutive entries one thread fills when a sequential recurrence
/// is split into independent runs.
const SEQUENTIAL_RUN_LEN: usize = 1 << 12;

/// Computes `[1, base, ..., base^(len-1)]` without a fully sequential pass.
///
/// Each run of `SEQUENTIAL_RUN_LEN` entries starts from an independently
/// computed power, so the recurrence only costs `log(len)` sequential steps.
pub fn powers_of_scalar<F: Field>(base: F, len: usize) -> Vec<F> {
    if len == 0 {
        return Vec::new();
    }
    let run_len = SEQUENTIAL_RUN_LEN.min(len);
    let run_step = base.pow([run_len as u64]);
    let run_count = len.div_ceil(run_len);

    let mut run_starts = Vec::with_capacity(run_count);
    let mut start = F::one();
    for _ in 0..run_count {
        run_starts.push(start);
        start *= run_step;
    }

    let mut powers = vec![F::zero(); len];
    powers
        .par_chunks_mut(run_len)
        .zip(run_starts)
        .for_each(|(chunk, run_start)| {
            let mut current = run_start;
            for slot in chunk.iter_mut() {
                *slot = current;
                current *= base;
            }
        });
    powers
}

/// Multiplies every scalar by one field element in parallel.
pub fn scale_scalars<F: Field>(scalars: &[F], factor: F) -> Vec<F> {
    scalars.par_iter().map(|scalar| *scalar * factor).collect()
}

/// Builds a windowed fixed-base table for `base`.
pub fn fixed_base_table<G: ScalarMul>(base: G, num_scalars: usize) -> BatchMulPreprocessing<G> {
    BatchMulPreprocessing::new(base, num_scalars)
}

/// Multiplies one fixed base by many scalars using a windowed table.
pub fn fixed_base_mul<G: ScalarMul>(table: &BatchMulPreprocessing<G>, scalars: &[G::ScalarField]) -> Vec<G> {
    table
        .batch_mul(scalars)
        .into_par_iter()
        .map(Into::into)
        .collect()
}


use crate::MyInt;

/// Convert a MyInt (i8) to a prime field element, preserving the sign.
#[inline]
pub fn myint_to_field<F: PrimeField>(x: MyInt) -> F {
    if x >= 0 {
        F::from(x as u64)
    } else {
        -F::from((-(x as i32)) as u64)
    }
}

//     borrow::Borrow,
//     fmt::{Debug, Display, Formatter, Result as FmtResult},
//     io::{Read, Write},
//     ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
///  Convert a myint matrix to scalar field matrix
pub fn convert_myint_to_scalar_mat<E>(a: &Vec<Vec<MyInt>>) -> Vec<Vec<E::ScalarField>>
where
    E: Pairing,
{
    #[cfg(feature = "parallel")]
    {
        a.iter().map(|x| {
            x.par_iter().map(|&y| myint_to_field::<E::ScalarField>(y)).collect::<Vec<E::ScalarField>>()
        }).collect::<Vec<Vec<E::ScalarField>>>()
    }
    #[cfg(not(feature = "parallel"))]
    {
        a.iter().map(|x| {
            x.iter().map(|&y| myint_to_field::<E::ScalarField>(y)).collect::<Vec<E::ScalarField>>()
        }).collect::<Vec<Vec<E::ScalarField>>>()
    }
}

/// Convert a boolean mat to scalar mat
pub fn convert_boolean_to_scalar_mat<E>(a: &Vec<Vec<bool>>) -> Vec<Vec<E::ScalarField>>
where
    E: Pairing,
{
    #[cfg(feature = "parallel")]
    {
        a.iter().map(|x| {
            x.par_iter().map(|&y| if y { E::ScalarField::one() } else { E::ScalarField::zero() }).collect::<Vec<E::ScalarField>>()
        }).collect::<Vec<Vec<E::ScalarField>>>()
    }
    #[cfg(not(feature = "parallel"))]
    {
        a.iter().map(|x| {
            x.iter().map(|&y| if y { E::ScalarField::one() } else { E::ScalarField::zero() }).collect::<Vec<E::ScalarField>>()
        }).collect::<Vec<Vec<E::ScalarField>>>()
    }
}

#[cfg(feature = "parallel")]
use rayon::prelude::*;


/// Compute the pairing of two group elements.
pub fn single_pairing<E: Pairing>(
    a: &E::G1,
    b: &E::G2,
) -> PairingOutput<E> {
    E::pairing(a, b)
}

/// Compute the inner pairing product of two group vectors.
/// This is already parallelized using Rayon.
/// export RAYON_NUM_THREADS = 64 to change the number of threads
pub fn inner_pairing_product<E: Pairing>(
    a: &[E::G1],
    b: &[E::G2],
) ->  PairingOutput<E> {
    let len = std::cmp::min(a.len(), b.len());
    let a = &a[0..len];
    let b = &b[0..len];

    #[cfg(feature = "parallel")]
    {
        const MIN_PARALLEL_LEN: usize = 16;
        if len < MIN_PARALLEL_LEN {
            return E::multi_pairing(a, b);
        }
        // One batch inversion per side replaces one inversion per point, and the
        // chunk count follows the thread count so each chunk pays for exactly one
        // final exponentiation.
        let (a_affine, b_affine) = rayon::join(
            || E::G1::normalize_batch(a),
            || E::G2::normalize_batch(b),
        );
        let chunk = len.div_ceil(rayon::current_num_threads()).max(MIN_PARALLEL_LEN);
        return a_affine
            .par_chunks(chunk)
            .zip(b_affine.par_chunks(chunk))
            .map(|(aa, bb)| E::multi_pairing(aa, bb))
            .reduce(|| PairingOutput::<E>::zero(), |acc, x| acc + x);
    }

    #[cfg(not(feature = "parallel"))]
    E::multi_pairing(a, b)
}

/// Compute the inner product of two field vectors in parallel.
pub fn inner_product<E:Pairing>(
    a: &[E::ScalarField],
    b: &[E::ScalarField]
) -> E::ScalarField {
    let len = std::cmp::min(a.len(), b.len());
    let a = &a[0..len];
    let b = &b[0..len];

    #[cfg(feature = "parallel")]
    let result = a.par_iter().zip(b.par_iter())
    .map(|(a, b)| *a * b).sum();
    #[cfg(not(feature = "parallel"))]
    let result = a.iter().zip(b.iter())
    .map(|(a, b)| *a * b).sum();
    result
}

/// Compute MSM in G1
pub fn msm_g1<E: Pairing>(
    a: &[E::G1],
    b: &[E::ScalarField],
) -> E::G1
where
    E::G1: VariableBaseMSM + PrimeGroup,
{
    let len = std::cmp::min(a.len(), b.len());
    let a = &a[0..len];
    let b = &b[0..len];
    // A single Pippenger call over batch-normalized bases beats splitting the
    // MSM into fixed-size chunks: the window grows with `len` and the bases need
    // one batch inversion instead of one inversion each.
    let a_affine = E::G1::normalize_batch(a);
    <E::G1 as VariableBaseMSM>::msm(&a_affine, b).unwrap()
}

/// Compute MSM in G1 with MyInt coefficients (i8), converting scalars on the fly
pub fn msm_g1_myint<E: Pairing>(
    a: &[E::G1],
    b: &[MyInt],
) -> E::G1
where
    E::G1: VariableBaseMSM + PrimeGroup,
{
    let len = std::cmp::min(a.len(), b.len());
    let a = &a[0..len];
    let b = &b[0..len];
    #[cfg(feature = "parallel")]
    {
        const CHUNK: usize = 64;
        if len <= CHUNK {
            let a_affine = a.par_iter()
                .map(|x| x.into_affine())
                .collect::<Vec<E::G1Affine>>();
            let scalars = b.par_iter()
                .map(|&x| myint_to_field::<E::ScalarField>(x))
                .collect::<Vec<E::ScalarField>>();
            return <E::G1 as VariableBaseMSM>::msm(&a_affine, &scalars).unwrap();
        }
        return a
            .par_chunks(CHUNK)
            .zip(b.par_chunks(CHUNK))
            .map(|(aa, bb)| {
                let aa_affine = aa.iter().map(|x| x.into_affine()).collect::<Vec<E::G1Affine>>();
                let bb_scalar = bb.iter().map(|&x| myint_to_field::<E::ScalarField>(x)).collect::<Vec<E::ScalarField>>();
                <E::G1 as VariableBaseMSM>::msm(&aa_affine, &bb_scalar).unwrap()
            })
            .reduce(|| E::G1::zero(), |acc, x| acc + x);
    }
    #[cfg(not(feature = "parallel"))]
    {
        let a_affine = a.iter()
            .map(|x| x.into_affine())
            .collect::<Vec<E::G1Affine>>();
        let scalars = b.iter()
            .map(|&x| myint_to_field::<E::ScalarField>(x))
            .collect::<Vec<E::ScalarField>>();

        <E::G1 as VariableBaseMSM>::msm(&a_affine, &scalars).unwrap()
    }
}

/// Compute MSM in G1 with MyInt coefficients (i8), assuming bases are already affine
/// Non-parallel scalar conversion; used as fallback when parallel feature is off
pub fn msm_g1_myint_affine_nonparallel<E: Pairing>(
    a_affine: &[E::G1Affine],
    b: &[MyInt],
) -> E::G1
where
    E::G1: VariableBaseMSM + PrimeGroup,
{
    let len = std::cmp::min(a_affine.len(), b.len());
    let a_affine = &a_affine[0..len];
    let b = &b[0..len];

    let scalars = b.iter()
        .map(|&x| myint_to_field::<E::ScalarField>(x))
        .collect::<Vec<E::ScalarField>>();

    <E::G1 as VariableBaseMSM>::msm(a_affine, &scalars).unwrap()
}

/// Compute MSM in G1
pub fn msm_g2<E: Pairing>(
    a: &[E::G2],
    b: &[E::ScalarField],
) -> E::G2
where
    E::G2: VariableBaseMSM + PrimeGroup,
{
    let len = std::cmp::min(a.len(), b.len());
    let a = &a[0..len];
    let b = &b[0..len];
    let a_affine = E::G2::normalize_batch(a);
    <E::G2 as VariableBaseMSM>::msm(&a_affine, b).unwrap()
}

/// Sparsity-aware MSM in G1 that skips zero coefficients
/// This implementation is sensitive to sparsity and should show speedup for sparse vectors
pub fn msm_g1_sparse_aware<E: Pairing>(
    a: &[E::G1],
    b: &[E::ScalarField],
) -> E::G1
where
    E::G1: VariableBaseMSM + PrimeGroup,
{
    let len = std::cmp::min(a.len(), b.len());
    let a = &a[0..len];
    let b = &b[0..len];

    // // Parallel filtering of zero coefficients
    // #[cfg(feature = "parallel")]
    // let non_zero_pairs: Vec<(E::G1, E::ScalarField)> = a.par_iter()
    //     .zip(b.par_iter())
    //     .filter(|(_, scalar)| !scalar.is_zero())
    //     .map(|(point, scalar)| (*point, *scalar))
    //     .collect();

    // #[cfg(not(feature = "parallel"))]
    let non_zero_pairs: Vec<(E::G1, E::ScalarField)> = a.iter()
        .zip(b.iter())
        .filter(|(_, scalar)| !scalar.is_zero())
        .map(|(point, scalar)| (*point, *scalar))
        .collect();

    if non_zero_pairs.is_empty() {
        return E::G1::zero();
    }

    let (filtered_points, filtered_scalars): (Vec<E::G1>, Vec<E::ScalarField>) =
        non_zero_pairs.into_iter().unzip();

    #[cfg(feature = "parallel")]
    let filtered_affine = filtered_points.par_iter()
        .map(|x| x.into_affine()).collect::<Vec<E::G1Affine>>();
    #[cfg(not(feature = "parallel"))]
    let filtered_affine = filtered_points.iter()
        .map(|x| x.into_affine()).collect::<Vec<E::G1Affine>>();

    <E::G1 as VariableBaseMSM>::msm(
        &filtered_affine,
        &filtered_scalars,
    ).unwrap()
}

/// Compute MSM in G1 with i64
pub fn msm_g1_short_i64_naive<E: Pairing>(
    a: &[E::G1],
    b: &[i64],
) -> E::G1
where
    E::G1: VariableBaseMSM + PrimeGroup,
{
    let len = std::cmp::min(a.len(), b.len());
    let a = &a[0..len];
    let b = &b[0..len];

    #[cfg(feature = "parallel")]
    let a_affine =
        a.par_iter().map(|x| x.into_affine())
        .collect::<Vec<E::G1Affine>>();
    #[cfg(not(feature = "parallel"))]
    let a_affine =
        a.iter().map(|x| x.into_affine())
        .collect::<Vec<E::G1Affine>>();
    let b_scalar: Vec<E::ScalarField> =
        b.iter().map(|&x| {
            if x >= 0 {
                E::ScalarField::from(x as u64)
            } else {
                -E::ScalarField::from((-(x as i32)) as u64)
            }
        }).collect();
    <E::G1 as VariableBaseMSM>::msm(
        &a_affine,
        &b_scalar,
    ).unwrap()
}

/// Compute the MSM in G1 with a boolean vector
pub fn boolean_msm_g1<E: Pairing>(
    a: &[E::G1],
    b: &[bool],
) -> E::G1
{
    let len = std::cmp::min(a.len(), b.len());
    let a = &a[0..len];
    let b = &b[0..len];

    #[cfg(feature = "parallel")]
    let result = a.par_iter().zip(b.par_iter()).map(|(a, b)| {
        if *b {
            *a
        } else {
            E::G1::zero()
        }
    }).reduce(|| E::G1::zero(), |acc, x| acc.add(&x));
    #[cfg(not(feature = "parallel"))]
    let result = a.iter().zip(b.iter()).map(|(a, b)| {
        if *b {
            *a
        } else {
            E::G1::zero()
        }
    }).fold(E::G1::zero(), |acc, x| acc.add(&x));
    result
}

/// Compute the MSM in G2 with a boolean vector
pub fn boolean_msm_g2<E: Pairing>(
    a: &[E::G2],
    b: &[bool],
) -> E::G2
{
    let len = std::cmp::min(a.len(), b.len());
    let a = &a[0..len];
    let b = &b[0..len];

    #[cfg(feature = "parallel")]
    let result = a.par_iter().zip(b.par_iter()).map(|(a, b)| {
        if *b {
            *a
        } else {
            E::G2::zero()
        }
    }).reduce(|| E::G2::zero(), |acc, x| acc.add(&x));
    #[cfg(not(feature = "parallel"))]
    let result = a.iter().zip(b.iter()).map(|(a, b)| {
        if *b {
            *a
        } else {
            E::G2::zero()
        }
    }).fold(E::G2::zero(), |acc, x| acc.add(&x));
    result
}

/// Add two G1 vectors in parallel
pub fn add_vec_g1<E: Pairing>(
    a: &[E::G1],
    b: &[E::G1],
) -> Vec<E::G1>
{
    let len = std::cmp::min(a.len(), b.len());

    #[cfg(feature = "parallel")]
    let mut result: Vec<E::G1> =
    a[0..len].par_iter().zip(b[0..len].par_iter()).map(|(a, b)| {
        a.add(b)
    }).collect();
    #[cfg(not(feature = "parallel"))]
    let mut result: Vec<E::G1> =
    a[0..len].iter().zip(b[0..len].iter()).map(|(a, b)| {
        a.add(b)
    }).collect();

    if len < a.len() {
        let rest = a[len..].to_vec();
        result = [result, rest].concat();
    } else if len < b.len() {
        let rest = b[len..].to_vec();
        result = [result, rest].concat();
    }

    result
}

/// Add two G2 vectors in parallel
pub fn add_vec_g2<E: Pairing>(
    a: &[E::G2],
    b: &[E::G2],
) -> Vec<E::G2>
{
    #[cfg(feature = "parallel")]
    let result = a.par_iter().zip(b.par_iter()).map(|(a, b)| {
        a.add(b)
    }).collect();
    #[cfg(not(feature = "parallel"))]
    let result = a.iter().zip(b.iter()).map(|(a, b)| {
        a.add(b)
    }).collect();
    result
}

/// Add two Zp vectors in parallel
pub fn add_vec_zp<E: Pairing>(
    a: &[E::ScalarField],
    b: &[E::ScalarField],
) -> Vec<E::ScalarField>
{
    #[cfg(feature = "parallel")]
    let result = a.par_iter().zip(b.par_iter()).map(|(a, b)| {
        a.add(b)
    }).collect();
    #[cfg(not(feature = "parallel"))]
    let result = a.iter().zip(b.iter()).map(|(a, b)| {
        a.add(b)
    }).collect();
    result
}

/// Scalar mult zp vectors in parallel
pub fn scalar_mul_vec_zp<E: Pairing>(
    a: &[E::ScalarField],
    x: &E::ScalarField,
) -> Vec<E::ScalarField>
{
    #[cfg(feature = "parallel")]
    let result = a.par_iter().map(|a| {
        a.mul(x)
    }).collect();
    #[cfg(not(feature = "parallel"))]
    let result = a.iter().map(|a| {
        a.mul(x)
    }).collect();
    result
}

/// Scalar mult G1 vectors in parallel
///
/// Normalizing once lets every ladder step use a mixed addition instead of a
/// projective one.
pub fn scalar_mul_vec_g1<E: Pairing>(
    a: &[E::G1],
    x: &E::ScalarField,
) -> Vec<E::G1>
{
    let scalar = x.into_bigint();
    E::G1::normalize_batch(a)
        .par_iter()
        .map(|base| base.mul_bigint(scalar))
        .collect()
}

/// Scalar mult G2 vectors in parallel
pub fn scalar_mul_vec_g2<E: Pairing>(
    a: &[E::G2],
    x: &E::ScalarField,
) -> Vec<E::G2>
{
    let scalar = x.into_bigint();
    E::G2::normalize_batch(a)
        .par_iter()
        .map(|base| base.mul_bigint(scalar))
        .collect()
}


/// Double a G1 vector in parallel
pub fn double_vec_g1<E: Pairing>(
    a: &[E::G1],
) -> Vec<E::G1>
{
    #[cfg(feature = "parallel")]
    let result = a.par_iter().map(|x| {
        x.double()
    }).collect();
    #[cfg(not(feature = "parallel"))]
    let result = a.iter().map(|x| {
        x.double()
    }).collect();
    result
}

/// Double a G2 vector in parallel
pub fn double_vec_g2<E: Pairing>(
    a: &[E::G2],
) -> Vec<E::G2>
{
    a.par_iter().map(|x| {
        x.double()
    }).collect()
}

/// Prepare the G1 base for short scalar multiplication
pub fn prepare_base_short_g1<E: Pairing>(
    a: &[E::G1],
    k: usize,
) -> Vec<Vec<E::G1>>
{
    let mut result = Vec::new();
    let mut base = a.to_vec();
    result.push(base.clone());
    for _ in 0..(k - 1) {
        base = double_vec_g1::<E>(base.as_slice());
        result.push(base.clone());
    }
    // assert_eq!(result[1][1] + result[1][1], result[2][1]);
    result
}

/// Compute MSM in G1 with i64
pub fn msm_g1_short_i64<E: Pairing>(
    a_prepare: &Vec<Vec<E::G1>>,
    b: &[i64],
    k: usize,
) -> E::G1
{
    let b_plus = b.par_iter()
    .map(|&x| if x > 0 { x as u64 } else { 0 })
    .collect::<Vec<u64>>();
    let b_minus = b.par_iter()
    .map(|&x| if x < 0 { - x as u64 } else { 0 })
    .collect::<Vec<u64>>();
    let mut result_plus = E::G1::zero();
    let mut result_minus = E::G1::zero();


    for i in 0..k {
        let b_bit_plus =
            b_plus.par_iter().map(|x| x & (1<<i) == 1<<i).collect::<Vec<bool>>();
        let b_bit_minus =
            b_minus.par_iter().map(|x| x & (1<<i) == 1<<i).collect::<Vec<bool>>();
        let base_i = a_prepare[i].as_slice();
        result_plus += boolean_msm_g1::<E>(base_i, b_bit_plus.as_slice());
        result_minus += boolean_msm_g1::<E>(base_i, b_bit_minus.as_slice());
    }

    result_plus - result_minus
}

/// Compute MSM in G1 with i64
pub fn msm_g1_short_myint<E: Pairing>(
    a_prepare: &Vec<Vec<E::G1>>,
    b: &[MyInt],
    k: usize,
) -> E::G1
{
    let b_plus = b.par_iter()
    .map(|&x| if x > 0 { x as u32 } else { 0 })
    .collect::<Vec<u32>>();
    let b_minus = b.par_iter()
    .map(|&x| if x < 0 { - x as u32 } else { 0 })
    .collect::<Vec<u32>>();
    let mut result_plus = E::G1::zero();
    let mut result_minus = E::G1::zero();


    for i in 0..k {
        let b_bit_plus =
            b_plus.par_iter().map(|x| x & (1<<i) == 1<<i).collect::<Vec<bool>>();
        let b_bit_minus =
            b_minus.par_iter().map(|x| x & (1<<i) == 1<<i).collect::<Vec<bool>>();
        let base_i = a_prepare[i].as_slice();
        result_plus += boolean_msm_g1::<E>(base_i, b_bit_plus.as_slice());
        result_minus += boolean_msm_g1::<E>(base_i, b_bit_minus.as_slice());
    }

    result_plus - result_minus
}


/// Compute the inner pairing product by parallelizing inner product
/// Slower than tha native version
pub fn inner_pairing_product_slower<E: Pairing>(
    a: &[E::G1],
    b: &[E::G2],
) -> PairingOutput<E> {
    let len = std::cmp::min(a.len(), b.len());
    let a = &a[0..len];
    let b = &b[0..len];
    type Gt<E> = PairingOutput<E>; // Introduce a local generic parameter for the type Gt
    a.par_iter()
        .zip(b.par_iter())
        .map(|(a, b)| single_pairing::<E>(a, b))
        .reduce(|| Gt::<E>::zero(), Gt::<E>::add) // Wrap PairingOutput<E> in a closure with no arguments
}


/// Test the utils functions
pub fn test_utils<E:Pairing>() {

    // let rng = &mut ark_std::test_rng();

    let n = 2_usize.pow(15);

    #[cfg(feature = "parallel")]
    let a_vec = (0..n).into_par_iter().map(|_|{
        let rng_par = &mut ark_std::test_rng();
        E::G1::rand(rng_par)
        }
    ).collect::<Vec<E::G1>>();

    #[cfg(not(feature = "parallel"))]
    let a_vec = (0..n).into_iter().map(|_|{
        let rng_par = &mut ark_std::test_rng();
        E::G1::rand(rng_par)
        }
    ).collect::<Vec<E::G1>>();

    #[cfg(feature = "parallel")]
    let b_vec = (0..n).into_par_iter().map(|_|{
        let rng_par = &mut ark_std::test_rng();
        E::G2::rand(rng_par)
        }
    ).collect::<Vec<E::G2>>();

    #[cfg(not(feature = "parallel"))]
    let b_vec = (0..n).into_iter().map(|_|{
        let rng_par = &mut ark_std::test_rng();
        E::G2::rand(rng_par)
        }
    ).collect::<Vec<E::G2>>();

    #[cfg(feature = "parallel")]
    let c_vec = (0..n).into_par_iter().map(|_|{
        let rng_par = &mut ark_std::test_rng();
        E::ScalarField::rand(rng_par)
        }
    ).collect::<Vec<E::ScalarField>>();

    #[cfg(not(feature = "parallel"))]
    let c_vec = (0..n).into_iter().map(|_|{
        let rng_par = &mut ark_std::test_rng();
        E::ScalarField::rand(rng_par)
        }
    ).collect::<Vec<E::ScalarField>>();

    #[cfg(feature = "parallel")]
    let d_vec = (0..n).into_par_iter().map(|_|{
        let rng_par = &mut ark_std::test_rng();
        E::ScalarField::rand(rng_par)
        }
    ).collect::<Vec<E::ScalarField>>();

    #[cfg(not(feature = "parallel"))]
    let d_vec = (0..n).into_iter().map(|_|{
        let rng_par = &mut ark_std::test_rng();
        E::ScalarField::rand(rng_par)
        }
    ).collect::<Vec<E::ScalarField>>();

    #[cfg(feature = "parallel")]
    let myint_vec: Vec<MyInt> = (0..n).into_par_iter()
    .map(|_|{
        use ark_std::{rand::Rng, test_rng};
        let mut rng = test_rng();
        rng.gen_range(-127..127)
    }).collect();

    #[cfg(not(feature = "parallel"))]
    let myint_vec: Vec<MyInt> = (0..n).into_iter()
    .map(|_|{
        use ark_std::{rand::Rng, test_rng};
        let mut rng = test_rng();
        rng.gen_range(-127..127)
    }).collect();

    #[cfg(feature = "parallel")]
    let boolean_vec: Vec<bool> = (0..n).into_par_iter()
    .map(|x| x % 2 == 0).collect();

    #[cfg(not(feature = "parallel"))]
    let boolean_vec: Vec<bool> = (0..n).into_iter()
    .map(|x| x % 2 == 0).collect();

    #[cfg(feature = "parallel")]
    let i64_vec: Vec<i64> = myint_vec.clone()
    .into_par_iter().map(|x| i64::from(x)).collect();

    #[cfg(not(feature = "parallel"))]
    let i64_vec: Vec<i64> = myint_vec.clone()
    .into_iter().map(|x| i64::from(x)).collect();

    let start = std::time::Instant::now();
    let _ = inner_pairing_product::<E>(&a_vec, &b_vec);
    // println!("Inner pairing product: {:?}", c);
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("*** Time elapsed in inner pairing product is: {:?}ms", elapsed);

    let start = std::time::Instant::now();
    let _ = inner_product::<E>(&c_vec, &d_vec);
    // println!("Inner pairing product: {:?}", c);
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("*** Time elapsed in field inner product is: {:?}ms", elapsed);

    let start = std::time::Instant::now();
    let _ = msm_g1::<E>(&a_vec, &c_vec);
    // println!("Inner pairing product: {:?}", c);
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("*** Time elapsed in msm_g1 is: {:?}ms", elapsed);

    let start = std::time::Instant::now();
    let _ = msm_g2::<E>(&b_vec, &d_vec);
    // println!("Inner pairing product: {:?}", c);
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("*** Time elapsed in msm_g2 is: {:?}ms", elapsed);

    let start = std::time::Instant::now();
    let msm_result_1 = msm_g1_short_i64_naive::<E>(&a_vec, &i64_vec);
    // println!("Inner pairing product: {:?}", c);
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("*** Time elapsed in msm_g1_naive is: {:?}ms", elapsed);

    let prepare_base = prepare_base_short_g1::<E>(&a_vec, 8);
    let start = std::time::Instant::now();
    let msm_result_2 =
        msm_g1_short_myint::<E>(&prepare_base, &myint_vec, 8);
    // println!("Inner pairing product: {:?}", c);
    assert_eq!(msm_result_1, msm_result_2);
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("*** Time elapsed in msm_g1_optimized is: {:?}ms", elapsed);

    let start = std::time::Instant::now();
    let _ = boolean_msm_g1::<E>(&a_vec, &boolean_vec);
    // println!("Inner pairing product: {:?}", c);
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("*** Time elapsed in boolean_msm_g1 is: {:?}ms", elapsed);

    let start = std::time::Instant::now();
    let _ = boolean_msm_g2::<E>(&b_vec, &boolean_vec);
    // println!("Inner pairing product: {:?}", c);
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;
    println!("*** Time elapsed in boolean_msm_g2 is: {:?}ms", elapsed);
}

/// Projection of a col major matrix to the left vector
/// Col Major
pub fn proj_left<E:Pairing> (
    mat: &Vec<Vec<E::ScalarField>>,
    l_vec: &Vec<E::ScalarField>,
) -> Vec<E::ScalarField> {
    let n = mat.len();
    let _ = mat[0].len();

    let mut result = Vec::new();
    for i in 0..n {
        let col = &mat[i];
        let ip_col = inner_product::<E>(
            &l_vec, &col);
        result.push(ip_col);
    }
    result
}


/// Projection of a col major matrix to the left vector
pub fn proj_right<E:Pairing> (
    mat: &Vec<Vec<E::ScalarField>>,
    r_vec: &Vec<E::ScalarField>,
) -> Vec<E::ScalarField> {
    let _ = mat.len();
    let m = mat[0].len();

    let mut result = Vec::new();
    for i in 0..m {
        #[cfg(feature = "parallel")]
        let row = mat.par_iter()
        .map(|x| x[i])
        .collect::<Vec<E::ScalarField>>();

        #[cfg(not(feature = "parallel"))]
        let row = mat.iter()
        .map(|x| x[i])
        .collect::<Vec<E::ScalarField>>();

        let ip_row = inner_product::<E>(
            &r_vec, &row);
        result.push(ip_row);
    }
    result
}

/// Check a vector is a zero vector
pub fn is_zero_vec(vec: &Vec<MyInt>) -> bool {
    vec.par_iter().all(|&x| x == 0)
}


/// Projection of a col major matrix to the left vector
/// Col Major
pub fn proj_left_myint<E:Pairing> (
    mat: &Vec<Vec<MyInt>>,
    l_vec: &Vec<E::ScalarField>,
) -> Vec<E::ScalarField> {
    let n = mat.len();
    let _ = mat[0].len();

    let mut result = Vec::new();
    for i in 0..n {
        let col = &mat[i];
        let ip_col = col.par_iter()
        .zip(l_vec.par_iter())
        .map(|(c, l)| myint_to_field::<E::ScalarField>(*c) * l)
        .sum();
        result.push(ip_col);
    }
    result
}

/// Memory-optimized projection of a column-major flat vector to the left vector
/// flat is stored column-major: column j = flat[j*m..(j+1)*m]
pub fn proj_left_myint_flat<E:Pairing> (
    flat: &[MyInt],
    l_vec: &[E::ScalarField],
    m: usize,  // number of rows
    n: usize,  // number of columns
) -> Vec<E::ScalarField> {
    (0..n).into_par_iter().map(|i| {
        let start = i * m;
        let end = start + m;
        flat[start..end].par_iter()
            .zip(l_vec.par_iter())
            .map(|(c, l)| myint_to_field::<E::ScalarField>(*c) * l)
            .sum()
    }).collect()
}


/// Projection of a col major matrix to the left vector
pub fn proj_right_myint<E:Pairing> (
    mat: &Vec<Vec<MyInt>>,
    r_vec: &Vec<E::ScalarField>,
) -> Vec<E::ScalarField> {
    let _ = mat.len();
    let m = mat[0].len();

    let mut result = Vec::new();
    for i in 0..m {
        let row = mat.par_iter()
        .map(|x| x[i])
        .collect::<Vec<MyInt>>();

        let ip_row = row.par_iter()
        .zip(r_vec.par_iter())
        .map(|(x, r)| myint_to_field::<E::ScalarField>(*x) * r)
        .sum();
        result.push(ip_row);
    }
    result
}

/// Memory-optimized projection of a column-major flat vector to the right vector
/// flat is stored column-major: column j = flat[j*m..(j+1)*m]

pub fn proj_right_myint_flat<E:Pairing> (
    flat: &[MyInt],
    r_vec: &[E::ScalarField],
    m: usize,  // number of rows
    n: usize,  // number of columns
) -> Vec<E::ScalarField> {
    // Parallel computation of inner product for each row, avoiding temporary Vec allocation
    (0..m).into_par_iter().map(|i| {
        (0..n).map(|j| {
            let idx = j * m + i;
            myint_to_field::<E::ScalarField>(flat[idx]) * r_vec[j]
        }).sum()
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Bls12_381;
    use ark_ec::pairing::Pairing;
    use ark_ff::UniformRand;
    use ark_std::test_rng;

    type E = Bls12_381;

    #[test]
    fn test_msm_sparse_vs_dense() {
        let n = 2_usize.pow(15); // 32K elements for meaningful timing
        let mut rng = test_rng();

        println!("\n=== MSM Sparse vs Dense Performance Test ===");
        println!("Vector size: {} elements", n);

        // Generate random G1 points
        let points: Vec<<E as Pairing>::G1> = (0..n)
            .map(|_| <E as Pairing>::G1::rand(&mut rng))
            .collect();

        // Dense vector (all random scalars)
        let dense_scalars: Vec<<E as Pairing>::ScalarField> = (0..n)
            .map(|_| <E as Pairing>::ScalarField::rand(&mut rng))
            .collect();

        // Sparse vector (90% zeros)
        let sparse_90_scalars: Vec<<E as Pairing>::ScalarField> = (0..n)
            .map(|i| {
                if i % 10 == 0 {
                    <E as Pairing>::ScalarField::rand(&mut rng)
                } else {
                    <E as Pairing>::ScalarField::zero()
                }
            })
            .collect();

        // Very sparse vector (99% zeros)
        let sparse_99_scalars: Vec<<E as Pairing>::ScalarField> = (0..n)
            .map(|i| {
                if i % 100 == 0 {
                    <E as Pairing>::ScalarField::rand(&mut rng)
                } else {
                    <E as Pairing>::ScalarField::zero()
                }
            })
            .collect();

        // All zeros vector
        let zero_scalars = vec![<E as Pairing>::ScalarField::zero(); n];

        println!("\n--- Standard MSM ---");

        // Test dense vector
        let start = std::time::Instant::now();
        let result_dense = msm_g1::<E>(&points, &dense_scalars);
        let time_dense = start.elapsed().as_secs_f64() * 1000.0;
        println!("Dense vector (0% zeros): {:.2}ms", time_dense);

        // Test 90% sparse vector
        let start = std::time::Instant::now();
        let result_sparse_90 = msm_g1::<E>(&points, &sparse_90_scalars);
        let time_sparse_90 = start.elapsed().as_secs_f64() * 1000.0;
        println!("Sparse vector (90% zeros): {:.2}ms", time_sparse_90);

        // Test 99% sparse vector
        let start = std::time::Instant::now();
        let _result_sparse_99 = msm_g1::<E>(&points, &sparse_99_scalars);
        let time_sparse_99 = start.elapsed().as_secs_f64() * 1000.0;
        println!("Sparse vector (99% zeros): {:.2}ms", time_sparse_99);

        // Test all zeros vector
        let start = std::time::Instant::now();
        let result_zeros = msm_g1::<E>(&points, &zero_scalars);
        let time_zeros = start.elapsed().as_secs_f64() * 1000.0;
        println!("All zeros vector (100% zeros): {:.2}ms", time_zeros);

        println!("\n--- Sparse-Aware MSM ---");

        // Test sparse-aware implementations
        let start = std::time::Instant::now();
        let result_sparse_aware_dense = msm_g1_sparse_aware::<E>(&points, &dense_scalars);
        let time_sparse_aware_dense = start.elapsed().as_secs_f64() * 1000.0;
        println!("Sparse-aware dense (0% zeros): {:.2}ms", time_sparse_aware_dense);

        let start = std::time::Instant::now();
        let result_sparse_aware_90 = msm_g1_sparse_aware::<E>(&points, &sparse_90_scalars);
        let time_sparse_aware_90 = start.elapsed().as_secs_f64() * 1000.0;
        println!("Sparse-aware (90% zeros): {:.2}ms", time_sparse_aware_90);

    let start = std::time::Instant::now();
    let _result_sparse_aware_99 = msm_g1_sparse_aware::<E>(&points, &sparse_99_scalars);
        let time_sparse_aware_99 = start.elapsed().as_secs_f64() * 1000.0;
        println!("Sparse-aware (99% zeros): {:.2}ms", time_sparse_aware_99);

        let start = std::time::Instant::now();
        let result_sparse_aware_zeros = msm_g1_sparse_aware::<E>(&points, &zero_scalars);
        let time_sparse_aware_zeros = start.elapsed().as_secs_f64() * 1000.0;
        println!("Sparse-aware all zeros: {:.2}ms", time_sparse_aware_zeros);

        println!("\n=== Speedup Analysis ===");
        println!("Standard MSM:");
        println!("  90% sparse speedup: {:.2}x", time_dense / time_sparse_90);
        println!("  99% sparse speedup: {:.2}x", time_dense / time_sparse_99);
        println!("  All zeros speedup: {:.2}x", time_dense / time_zeros);

        println!("Sparse-aware MSM:");
        println!("  90% sparse speedup: {:.2}x", time_sparse_aware_dense / time_sparse_aware_90);
        println!("  99% sparse speedup: {:.2}x", time_sparse_aware_dense / time_sparse_aware_99);
        println!("  All zeros speedup: {:.2}x", time_sparse_aware_dense / time_sparse_aware_zeros);

        // Verify correctness
        assert_eq!(result_zeros, <E as Pairing>::G1::zero());
        assert_eq!(result_sparse_aware_zeros, <E as Pairing>::G1::zero());
        assert_eq!(result_dense, result_sparse_aware_dense);
        assert_eq!(result_sparse_90, result_sparse_aware_90);
        assert_ne!(result_dense, result_sparse_90);
        assert_ne!(result_dense, result_zeros);

        println!("✓ All correctness tests passed");
    }
}