/// This module contains utility functions for the smart polynomial commitment scheme.
use ark_ec::{pairing::{Pairing, PairingOutput},
    CurveGroup, VariableBaseMSM, PrimeGroup, AdditiveGroup
};
use ark_ff::UniformRand;
use ark_std::{Zero, One,
    ops::{Add, Mul}
};

//     borrow::Borrow,
//     fmt::{Debug, Display, Formatter, Result as FmtResult},
//     io::{Read, Write},
//     ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
//     rand::{
//         distributions::{Distribution, Standard},
//         Rng,
//     },
//     vec::*,
//     UniformRand, Zero,
// };


use rayon::prelude::*;

// use super::{FiatShamir, UniversalParams};
// Add this line

// type Gt = PairingOutput<Pairing>;


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
    
    a.par_iter().zip(b.par_iter())
    .map(|(a, b)| *a * b).sum()
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
    let a_affine = a.par_iter()
    .map(|x| x.into_affine()).collect::<Vec<E::G1Affine>>();
    <E::G1 as VariableBaseMSM>::msm(
        &a_affine,
        &b,
    ).unwrap()
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
    let a_affine = a.par_iter()
    .map(|x| x.into_affine()).collect::<Vec<E::G2Affine>>();
    <E::G2 as VariableBaseMSM>::msm(
        &a_affine,
        &b,
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

    let a_affine =
        a.par_iter().map(|x| x.into_affine())
        .collect::<Vec<E::G1Affine>>();
    let b_scalar: Vec<E::ScalarField> =
        b.iter().map(|x| E::ScalarField::from(*x)).collect();
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

    a.par_iter().zip(b.par_iter()).map(|(a, b)| {
        if *b {
            *a
        } else {
            E::G1::zero()
        }
    }).reduce(|| E::G1::zero(), |acc, x| acc.add(&x))
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

    a.par_iter().zip(b.par_iter()).map(|(a, b)| {
        if *b {
            *a
        } else {
            E::G2::zero()
        }
    }).reduce(|| E::G2::zero(), |acc, x| acc.add(&x))
}

/// Add two G1 vectors in parallel 
pub fn add_vec_g1<E: Pairing>(
    a: &[E::G1],
    b: &[E::G1],
) -> Vec<E::G1>
{
    let len = std::cmp::min(a.len(), b.len());

    let mut result: Vec<E::G1> =
    a[0..len].par_iter().zip(b[0..len].par_iter()).map(|(a, b)| {
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
    a.par_iter().zip(b.par_iter()).map(|(a, b)| {
        a.add(b)
    }).collect()
}

/// Add two Zp vectors in parallel
pub fn add_vec_zp<E: Pairing>(
    a: &[E::ScalarField],
    b: &[E::ScalarField],
) -> Vec<E::ScalarField>
{
    a.par_iter().zip(b.par_iter()).map(|(a, b)| {
        a.add(b)
    }).collect()
}

/// Scalar mult zp vectors in parallel
pub fn scalar_mul_vec_zp<E: Pairing>(
    a: &[E::ScalarField],
    x: &E::ScalarField,
) -> Vec<E::ScalarField>
{
    a.par_iter().map(|a| {
        a.mul(x)
    }).collect()
}

/// Scalar mult zp vectors in parallel
pub fn scalar_mul_vec_g1<E: Pairing>(
    a: &[E::G1],
    x: &E::ScalarField,
) -> Vec<E::G1>
{
    a.par_iter().map(|a| {
        a.mul(x)
    }).collect()
}

/// Scalar mult zp vectors in parallel
pub fn scalar_mul_vec_g2<E: Pairing>(
    a: &[E::G2],
    x: &E::ScalarField,
) -> Vec<E::G2>
{
    a.par_iter().map(|a| {
        a.mul(x)
    }).collect()
}


/// Double a G1 vector in parallel
pub fn double_vec_g1<E: Pairing>(
    a: &[E::G1],
) -> Vec<E::G1>
{
    a.par_iter().map(|x| {
        x.double()
    }).collect()
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
pub fn msm_g1_short_i32<E: Pairing>(
    a_prepare: &Vec<Vec<E::G1>>,
    b: &[i32],
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

    use rand::Rng;

    // let rng = &mut ark_std::test_rng();

    let n = 2_usize.pow(15);

    let a_vec = (0..n).into_par_iter().map(|_|{
        let rng_par = &mut ark_std::test_rng();
        E::G1::rand(rng_par)
        }
    ).collect::<Vec<E::G1>>();
    
    let b_vec = (0..n).into_par_iter().map(|_|{
        let rng_par = &mut ark_std::test_rng();
        E::G2::rand(rng_par)
        }
    ).collect::<Vec<E::G2>>();

    let c_vec = (0..n).into_par_iter().map(|_|{
        let rng_par = &mut ark_std::test_rng();
        E::ScalarField::rand(rng_par)
        }
    ).collect::<Vec<E::ScalarField>>();

    let d_vec = (0..n).into_par_iter().map(|_|{
        let rng_par = &mut ark_std::test_rng();
        E::ScalarField::rand(rng_par)
        }
    ).collect::<Vec<E::ScalarField>>();

    let i32_vec: Vec<i32> = (0..n).into_par_iter()
    .map(|_|{
        let mut rng = rand::thread_rng();
        rng.gen_range(-127..127)
    }).collect();
    let boolean_vec: Vec<bool> = (0..n).into_par_iter()
    .map(|x| x % 2 == 0).collect();

    let i64_vec: Vec<i64> = i32_vec.clone()
    .into_par_iter().map(|x| x as i64).collect();

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
        msm_g1_short_i32::<E>(&prepare_base, &i32_vec, 8);
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


/// Convert a i32 mat to scalar mat
pub fn convert_i32_to_scalar_mat<E>(a: &Vec<Vec<i32>>) -> Vec<Vec<E::ScalarField>>
where
    E: Pairing,
{
    a.iter().map(|x| {
        x.par_iter().map(|&y| E::ScalarField::from(y as i64)).collect::<Vec<E::ScalarField>>()
    }).collect::<Vec<Vec<E::ScalarField>>>()
}

/// Convert a boolean mat to scalar mat
pub fn convert_boolean_to_scalar_mat<E>(a: &Vec<Vec<bool>>) -> Vec<Vec<E::ScalarField>>
where
    E: Pairing,
{
    a.iter().map(|x| {
        x.par_iter().map(|&y| if y { E::ScalarField::one() } else { E::ScalarField::zero() }).collect::<Vec<E::ScalarField>>()
    }).collect::<Vec<Vec<E::ScalarField>>>()
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
        let row =
        mat.par_iter()
        .map(|x| x[i])
        .collect::<Vec<E::ScalarField>>();
        
        let ip_row = inner_product::<E>(
            &r_vec, &row);
        result.push(ip_row);
    }
    result
}