//! Implement linear algebra utilities
//! on dense vectors and matrices
//! over prime fields
//!
//! parallelized using rayon
//!
use ark_ec::PrimeGroup;
use ark_ff::PrimeField;

use super::matdef::ShortInt;

use rayon::prelude::*;

use crate::utils::xi;

use crate::MyInt;

// Compute the inner product of two vectors
// parallelized
//
pub fn inner_product<F:PrimeField>(
    a: &Vec<F>,
    b: &Vec<F>
) -> F {
    let len = std::cmp::min(a.len(), b.len());
    let a = &a[0..len];
    let b = &b[0..len];

    a.par_iter().zip(b.par_iter())
    .map(|(a, b)| *a * b).sum()
}


// Compute the inner product of two vectors
// parallelized
//
pub fn inner_product_slice<F:PrimeField>(
    a: &[F],
    b: &[F]
) -> F {

    a.par_iter().zip(b.par_iter())
    .map(|(a, b)| *a * b).sum()
}

// Compute the inner product of two vectors
// parallelized
//
pub fn inner_product_mixed_slice<F:PrimeField>(
    a: &[MyInt],
    b: &[F]
) -> F {

    a.par_iter().zip(b.par_iter())
    .map(|(a, b)| F::from(*a) * b).sum()
}


// Add two Zp vectors in parallel
pub fn vec_addition<F: PrimeField>(
    a: &Vec<F>,
    b: &Vec<F>,
) -> Vec<F>
{
    let len1 = a.len();
    let len2 = b.len();
    let len = std::cmp::min(len1, len2);

    if len1 == len && len2 == len {
        a.par_iter().zip(b.par_iter()).map(|(a, b)| {
            a.add(b)
        }).collect()
    } else if len < len1 {
        let a_cut = &a[0..len].to_vec();
        let part1: Vec<F> = a_cut.par_iter().zip(b.par_iter()).map(|(a, b)| {
            a.add(b)
        }).collect();
        let part2 = &a[len..len1];
        part1.iter().chain(part2.iter()).cloned().collect()
    } else {
        let b_cut = &b[0..len].to_vec();
        let part1: Vec<F> = a.par_iter().zip(b_cut.par_iter()).map(|(a, b)| {
            a.add(b)
        }).collect();
        let part2 = &b[len..len2];
        part1.iter().chain(part2.iter()).cloned().collect()
    }
}

// Scalar mult zp vectors in parallel
pub fn vec_scalar_mul<F: PrimeField>(
    a: &Vec<F>,
    x: &F,
) -> Vec<F>
{
    a.par_iter().map(|a| {
        a.mul(x)
    }).collect()
}

pub fn vec_element_wise_mul<F: PrimeField>(
    a: &Vec<F>,
    b: &Vec<F>,
) -> Vec<F>
{
    let len1 = a.len();
    let len2 = b.len();
    let len = std::cmp::min(len1, len2);

    if len1 == len && len2 == len {
        a.par_iter().zip(b.par_iter()).map(|(a, b)| {
            a.mul(b)
        }).collect()
    } else if len < len1 {
        let a_cut = &a[0..len].to_vec();
        a_cut.par_iter().zip(b.par_iter()).map(|(a, b)| {
            a.mul(b)
        }).collect()
    } else {
        let b_cut = &b[0..len].to_vec();
        a.par_iter().zip(b_cut.par_iter()).map(|(a, b)| {
            a.mul(b)
        }).collect()
    }
}

pub fn vec_element_wise_mul_slice<F: PrimeField>(
    a: &[F],
    b: &[F],
) -> Vec<F>
{
    let len1 = a.len();
    let len2 = b.len();
    let len = std::cmp::min(len1, len2);

    let a_prefix = &a[..len];
    let b_prefix = &b[..len];

    a_prefix
        .par_iter()
        .zip(b_prefix.par_iter())
        .map(|(val_a, val_b)| val_a.mul(val_b))
        .collect()
}


// Reshape a col major matrix into an (almost) square col major matrix
// such that the flattened vec keep unchanged
//
pub fn reshape_mat_cm<F: PrimeField>(
    a: &Vec<Vec<F>>,
) -> Vec<Vec<F>>
{
    let n = a.len();
    let m = a[0].len();

    let log_n_new = ((m * n).ilog2()/2) as usize;

    let n_new = 1usize << log_n_new;
    let m_new = m * n / n_new;

    (0..n_new).into_iter().map(|j| {
        (0..m_new).into_par_iter().map(|i| {
            let k = j * m_new + i;
            // Find the element at the same flattened position 'k' in the old matrix (column-major)
            let old_col = k / m;
            let old_row = k % m;
            a[old_col][old_row]
        }).collect()
    }).collect()

    // (0..n_new).into_iter().map(|j| {       // j = new col
    //     (0..m_new).into_par_iter().map(|i| { // i = new row
    //         // Flattened index 'k' based on new matrix position (column-major)
    //         let k = j * m_new + i;
    //         // Find the element at the same flattened position 'k' in the old matrix (column-major)
    //         let old_col = k / m;
    //         let old_row = k % m;
    //         a[old_col][old_row]
    //     }).collect()
    // }).collect()
}

// Reshape a col major matrix into a (almost) square col major matrix
// such that the flattened vec keep unchanged
//
pub fn reshape_mat_cm_short<I: ShortInt>(
    a: &Vec<Vec<I>>,
) -> Vec<Vec<I>>
where
    Vec<I>: IntoParallelIterator + FromParallelIterator<I>,
{
    let n = a.len();
    let m = a[0].len();

    let log_n_new = ((m * n).ilog2()/2) as usize;

    let n_new = 1usize << log_n_new;
    let m_new = m * n / n_new;

    (0..n_new).into_iter().map(|j| {
        (0..m_new).into_par_iter().map(|i| {
            let k = j * m_new + i;
            // Find the element at the same flattened position 'k' in the old matrix (column-major)
            let old_col = k / m;
            let old_row = k % m;
            a[old_col][old_row]
        }).collect()
    }).collect()

    // (0..n_new).into_iter().map(|j| {       // j = new col
    //     (0..m_new).into_par_iter().map(|i| { // i = new row
    //         // Flattened index 'k' based on new matrix position (column-major)
    //         let k = j * m_new + i;
    //         // Find the element at the same flattened position 'k' in the old matrix (column-major)
    //         let old_col = k / m;
    //         let old_row = k % m;
    //         a[old_col][old_row]
    //     }).collect()
    // }).collect()
}


// Reshape a col major matrix into a (almost) square col major matrix
// such that the flattened vec keep unchanged
//
pub fn reshape_mat_cm_to_myint_short<I: ShortInt>(
    a: &Vec<Vec<I>>,
) -> Vec<Vec<MyInt>>
where
    Vec<I>: IntoParallelIterator + FromParallelIterator<I>,
{
    let n = a.len();
    let m = a[0].len();

    let log_n_new = ((m * n).ilog2()/2) as usize;

    let n_new = 1usize << log_n_new;
    let m_new = m * n / n_new;

    (0..n_new).into_iter().map(|j| {
        (0..m_new).into_par_iter().map(|i| {
            // let index = i * n_new + j;
            // let row = index / n;
            // let col = index % n;
            let k = j * m_new + i;
            // Find the element at the same flattened position 'k' in the old matrix (column-major)
            let old_col = k / m;
            let old_row = k % m;
            a[old_col][old_row].to_myint()
        }).collect()
    }).collect()
}

pub fn reshape_points<F: PrimeField>(
    point: &(Vec<F>, Vec<F>),
) -> (Vec<F>, Vec<F>)
{
    let log_m  = point.0.len();
    let log_n = point.1.len();

    let log_n_new = ((log_m + log_n)/2) as usize;
    // let log_m_new = log_m + log_n - log_n_new;

    let xx =
    [point.1.clone(), point.0.clone()].concat();

    let xl_new = xx[log_n_new..].to_vec();
    let xr_new = xx[0..log_n_new].to_vec();

    (xl_new, xr_new)
}

// Reshape a col major matrix into an vector
// such that the flattened vec keep unchanged
//
pub fn reshape_mat_cm_to_vec_short<I: ShortInt>(
    a: &Vec<Vec<I>>,
) -> Vec<I>
where
    Vec<I>: IntoParallelIterator + FromParallelIterator<I>,
{
    let n = a.len();
    let m = a[0].len();

    let len = m * n;

    let result = (0..len).into_par_iter().map(|index| {
            a[index / m][index % m]
    }).collect();

    result
}


// Reshape a col major matrix into an vector
// such that the evaluation keep unchanged
//
pub fn reshape_mat_cm_to_field_vec_short<I, F>(
    a: &Vec<Vec<I>>,
) -> Vec<F>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    Vec<I>: IntoParallelIterator + FromParallelIterator<I>,
{
    let n = a.len();
    let m = a[0].len();

    let len = m * n;

    (0..len).into_par_iter().map(|index| {
            F::from(a[index / m][index % m])
    }).collect()
}


// Projection of a col major matrix to the left vector
// Col Major
pub fn proj_left_cm<F:PrimeField> (
    mat: &Vec<Vec<F>>,
    l_vec: &Vec<F>,
) -> Vec<F> {
    let n = mat.len();
    let m = mat[0].len();

    if m > l_vec.len() {
        panic!("The length of the left vector is not enough");
    }

    let result: Vec<F> = (0..n).into_par_iter().map(|i| {
        let col = &mat[i];
        inner_product::<F>(&l_vec, &col)
    }).collect();

    result
}


// Projection of a col major matrix to the left vector
pub fn proj_right_cm<F:PrimeField> (
    mat: &Vec<Vec<F>>,
    r_vec: &Vec<F>,
) -> Vec<F> {
    let n = mat.len();
    let m = mat[0].len();

    if n > r_vec.len() {
        panic!("The length of the right vector is not enough");
    }

    let result: Vec<F> = (0..m).into_par_iter().map(|i| {
        let row = mat.iter().map(|x| x[i]).collect::<Vec<F>>();
        inner_product::<F>(&r_vec, &row)
    }).collect();

    result
}

pub fn proj_cm<F:PrimeField> (
    mat: &Vec<Vec<F>>,
    l_vec: &Vec<F>,
    r_vec: &Vec<F>,
) -> F {
    let n = mat.len();
    let m = mat[0].len();

    if m > l_vec.len() {
        panic!("The length of the left vector is not enough");
    }

    if n > r_vec.len() {
        panic!("The length of the right vector is not enough");
    }

    let la = proj_left_cm(mat, l_vec);
    let result = inner_product(&la, r_vec);
    result
}

pub fn proj_cm_on_tensor<F:PrimeField> (
    mat: &Vec<Vec<F>>,
    xl: &Vec<F>,
    xr: &Vec<F>,
) -> F {

    let n = mat.len();

    if n == 0 {
        panic!("!!! The matrix is empty");
    }

    let l_vec = xi::xi_from_challenges::<F>(&xl);

    if n == 1 {
        return inner_product(&mat[0], &l_vec);
    }

    let r_vec = xi::xi_from_challenges::<F>(&xr);

    let m = mat[0].len();

    if m > (1usize << xl.len()) {
        panic!("The length of the left challenge is not enough");
    }

    if n > (1usize << xr.len()) {
        println!("m: {}, xl.len(): {}", m, xl.len());
        println!("n: {}, xr.len(): {}", n, xr.len());
        panic!("The length of the right challenge is not enough");
    }

    proj_cm(mat, &l_vec, &r_vec)
}

// Compute the msm
// parallelized
//
pub fn msm<F, G>(
    a: &Vec<G>,
    b: &Vec<F>
) -> G
where
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    let len = std::cmp::min(a.len(), b.len());
    let a = &a[0..len];
    let b = &b[0..len];

    a.par_iter().zip(b.par_iter())
    .map(|(a, b)| a.mul(b)).sum()
}


pub fn vec_kronecker<F:PrimeField> (
    a: &Vec<F>,
    b: &Vec<F>,
) -> Vec<F> {
    let len1 = a.len();
    let len2 = b.len();

    let result = (0..len1).map(|i| {
        (0..len2).into_par_iter().map(|j| {
            a[i].mul(&b[j])
        }).collect::<Vec<F>>()
    }).collect::<Vec<Vec<F>>>().concat();

    result
}

// Compute the multiplication of two col_major myint dense matrices
//
pub fn mat_mul_shortint<I>(
    a: &Vec<Vec<I>>, b: &Vec<Vec<I>>
) -> Vec<Vec<MyInt>>
where I: ShortInt
{
    let a_rows = a[0].len();
    let a_cols = a.len();
    let b_cols = b.len();

    let mut result =
        vec![vec![0 as MyInt; a_rows]; b_cols];

    result.par_iter_mut().enumerate().for_each(|(j, col)| {
        for i in 0..a_rows {
            let mut acc: i32 = 0;
            for k in 0..a_cols {
                acc += a[k][i].to_myint() as i32 * b[j][k].to_myint() as i32;
            }
            let clamped = acc.clamp(MyInt::MIN as i32, MyInt::MAX as i32);
            col[i] = clamped as MyInt;
        }
    });

    result
}

// Split a vector into parts
//
pub fn split_into_part(
    a: &Vec<MyInt>,
    k_l: usize,
    k_u: usize,
) -> (Vec<MyInt>, Vec<bool>, Vec<MyInt>, Vec<MyInt>, Vec<MyInt>)
{
    let n = a.len();
    let sign = (0..n).into_par_iter().map(
        |i| {
        a[i] >= 0
    }).collect::<Vec<bool>>();

    let abs = a.par_iter().map(|x| {
        x.abs()
    }).collect::<Vec<MyInt>>();

    let low_th = (1 as MyInt) << k_l;
    let up_th = (1 as MyInt) << k_u;

    let lower = abs.par_iter().map(|x| {
        x % low_th
    }).collect::<Vec<MyInt>>();

    let middle = abs.par_iter().map(|x| {
        (x % up_th - x % low_th)/low_th
    }).collect::<Vec<MyInt>>();

    let upper = abs.par_iter().map(|x| {
        x / up_th
    }).collect::<Vec<MyInt>>();

    let middle_signed =
    middle.par_iter().zip(sign.par_iter()).map(|(x, s)| {
        if *s {
            *x
        } else {
            -x
        }
    }).collect::<Vec<MyInt>>();

    (middle_signed, sign, lower, middle, upper)
}

// Split a k-bit positive vector into boolean vectors
//
pub fn split_into_boolean(
    a: &Vec<MyInt>,
    k: usize,
) -> (Vec<bool>, Vec<MyInt>, Vec<Vec<bool>>)
{
    let mut booleans = Vec::new();

    let sign = a.par_iter().map(|x| {
        *x >= 0
    }).collect::<Vec<bool>>();

    let abs = a.par_iter().map(|x| {
        x.abs()
    }).collect::<Vec<MyInt>>();


    for kappa in 0..k {

        let cur = abs.par_iter().map(|x| {
            *x & ( (1 as MyInt) << kappa) != 0
        }).collect::<Vec<bool>>();
        booleans.push(cur);
    }

    (sign, abs, booleans)
}


pub fn split_into_boolean_direct(
    a: &Vec<MyInt>,
    k: usize,
    max_bit: usize,
) -> (Vec<bool>, Vec<Vec<bool>>, Vec<MyInt>, Vec<Vec<bool>> )
{
    let n = a.len();

    let mut booleans_up = Vec::new();
    let mut booleans_low = Vec::new();


    let mut upper = vec![0; n];
    let mut lower = vec![0; n];

    let sign = (0..n).into_par_iter().map(|i| {
        a[i] >= 0
    }).collect::<Vec<bool>>();

    for kappa in 0..k {
        let cur = a.par_iter().map(|x| {
            (x.abs() as MyInt) & ( (1 as MyInt) << kappa) != 0
        }).collect::<Vec<bool>>();
        booleans_low.push(cur);

        let factor = (1 as MyInt) << kappa;
        let cur_low = a.par_iter().map(|x| {
            if (x.abs() as MyInt) & ( (1 as MyInt) << kappa) != 0 {
                factor
            } else {
                0
            }
        }).collect::<Vec<MyInt>>();

        lower = lower.par_iter().zip(cur_low
            .par_iter()).map(|(a, b)| {
            a + b
        }).collect::<Vec<MyInt>>();
    }

    for kappa in (2*k)..max_bit {
        let cur = a.par_iter()
        .map(|x| {
            (x.abs() as u32) & (1u32 << kappa) != 0
        }).collect::<Vec<bool>>();

        booleans_up.push(cur);

        let factor = (1 as MyInt) << kappa;
        let cur_up = a.par_iter()
        .map(|x| {
            if (x.abs() as MyInt) & ( (1 as MyInt) << kappa) != 0 {
                factor
            } else {
                0
            }
        }).collect::<Vec<MyInt>>();


        upper = upper.par_iter()
        .zip(cur_up.par_iter())
        .map(|(a, b)| {
            a + b
        }).collect::<Vec<MyInt>>();

    }

    let abs = a.par_iter().map(|x| {
        x.abs()
    }).collect::<Vec<MyInt>>();

    let middle_positive =
    abs.par_iter().zip(upper.par_iter())
    .zip(lower.par_iter())
    .map(|((a, b), c)| {
        (a - b - c)/ ( (1 as MyInt) << k)
    }).collect::<Vec<MyInt>>();

    let middle_signed =
    middle_positive.par_iter().zip(sign.par_iter())
    .map(|(x, s)| {
        if *s {
            *x
        } else {
            -*x
        }
    }).collect::<Vec<MyInt>>();

    (sign, booleans_low, middle_signed, booleans_up)
}

// mat_a + scalar * mat_b
pub fn mat_scalar_addition<I, F>(
    a: &Vec<Vec<F>>,
    b: &Vec<Vec<I>>,
    scalar: F,
) -> Vec<Vec<F>>
where
    I: ShortInt,
    F: PrimeField + From<I>,
{
    let n1 = a.len();
    let m1 = a[0].len();

    let n2 = b.len();
    let m2 = b[0].len();

    let n = std::cmp::max(n1, n2);
    let m = std::cmp::max(m1, m2);


    let result = (0..n).into_par_iter().map(|j| {

        let mut col_cur = vec![F::zero(); m];

        if j < n1 {
            (0..m1).into_iter().for_each(|i| {
                col_cur[i] = a[j][i];
            });
        }

        if j < n2 {
            (0..m2).into_iter().for_each(|i| {
                col_cur[i] = col_cur[i].add(&F::from(b[j][i]).mul(&scalar));
            });
        }

        col_cur

    }).collect::<Vec<Vec<F>>>();

    // (0..n1).into_iter().for_each(|j| {
    //     let mut col_cur = result[j].clone();
    //     (0..m1).into_iter().for_each(|i| {
    //         col_cur[j][i] = a[j][i];
    //     });
    //     result[j] = col_cur;
    // });

    // (0..n2).into_iter().for_each(|j| {
    //     (0..m2).into_iter().for_each(|i| {
    //         result[j][i] = result[j][i].add(&F::from(b[j][i]).mul(&scalar));
    //     });
    // });

    result
}
