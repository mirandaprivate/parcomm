use crate::MyInt;
use ark_ff::PrimeField;
use rayon::prelude::*;

/// In-place fold: a_even <- a_even + z * a_odd.
/// Returns the shortened mutable slice (first ceil(len/2) elements).
pub fn eo_fold_in_place<F: PrimeField>(a: &mut [F], z: F) -> &mut [F] {
    eo_fold_in_place_with_par(a, z, true)
}

pub fn eo_fold_in_place_with_par<F: PrimeField>(a: &mut [F], z: F, parallel: bool) -> &mut [F] {
    let len = a.len();
    let half = (len + 1) / 2;
    // Parallel or sequential computation of folded results into temporary buffer, then write back to first half segment, avoiding mutable aliasing issues
    let tmp: Vec<F> = if parallel {
        (0..half)
            .into_par_iter()
            .map(|i| {
                let even = a[2 * i];
                let odd = a.get(2 * i + 1).copied().unwrap_or_else(F::zero);
                even + z * odd
            })
            .collect()
    } else {
        (0..half)
            .map(|i| {
                let even = a[2 * i];
                let odd = a.get(2 * i + 1).copied().unwrap_or_else(F::zero);
                even + z * odd
            })
            .collect()
    };
    // Copy back sequentially to ensure deterministic behavior
    a[..half].copy_from_slice(&tmp);
    &mut a[..half]
}

pub fn eo_fold_myint_to_field<F: PrimeField>(a: &[MyInt], z: F, out: &mut [F]) {
    eo_fold_myint_to_field_with_par(a, z, out, true);
}

pub fn eo_fold_myint_to_field_with_par<F: PrimeField>(
    a: &[MyInt],
    z: F,
    out: &mut [F],
    parallel: bool,
) {
    let len = a.len();
    let half = (len + 1) / 2;
    assert!(out.len() >= half, "output buffer too small");

    let tmp: Vec<F> = if parallel {
        (0..half)
            .into_par_iter()
            .map(|i| {
                let even = myint_to_field::<F>(a.get(2 * i).copied().unwrap_or(0));
                let odd = myint_to_field::<F>(a.get(2 * i + 1).copied().unwrap_or(0));
                even + z * odd
            })
            .collect()
    } else {
        (0..half)
            .map(|i| {
                let even = myint_to_field::<F>(a.get(2 * i).copied().unwrap_or(0));
                let odd = myint_to_field::<F>(a.get(2 * i + 1).copied().unwrap_or(0));
                even + z * odd
            })
            .collect()
    };
    out[..half].copy_from_slice(&tmp);
}

/// Iteratively fold a MyInt slice into a scalor using logarithmic numbers of z
/// The first round use eo_fold_myint_to_field to fold the myint slice into a field slice,
/// Then iteratively use eo_fold_in_place to fold the field slice into a field scalar.
pub fn eo_fold_iterative_myint_to_field<F: PrimeField>(a: &[MyInt], x_vec: Vec<F>) -> F {
    let len = a.len();
    let half = (len + 1) / 2;
    let log_len = len.next_power_of_two().ilog2() as usize;

    if log_len == 0 {
        assert!(x_vec.is_empty(), "x_vec must be empty when len == 1");
        return myint_to_field::<F>(a.get(0).copied().unwrap_or(0));
    }

    if x_vec.len() != log_len {
        panic!("x_vec length must be equal to log_len");
    }

    // Allocate once; contents will be overwritten before read.
    let mut buffer: Vec<F> = Vec::with_capacity(half);
    // SAFETY: We immediately set len to half and fill every slot before reading.
    unsafe {
        buffer.set_len(half);
    }
    let mut out: &mut [F] = &mut buffer[..];
    eo_fold_myint_to_field(a, x_vec[0], out);

    for i in 1..log_len {
        out = eo_fold_in_place(out, x_vec[i]);
    }
    out[0]
}

/// Convert a int k to its Least endian representation in total total_bits bits
/// Padding with zeros if necessary
pub fn int_to_least_endian_bits(k: usize, total_bits: usize) -> Vec<bool> {
    let mut bits = Vec::with_capacity(total_bits);
    for i in 0..total_bits {
        bits.push(((k >> i) & 1) == 1);
    }
    bits
}

#[inline]
fn myint_to_field<F: PrimeField>(v: MyInt) -> F {
    if v >= 0 {
        F::from(v as u64)
    } else {
        -F::from((-(v as i32)) as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::Zero;
    use rayon::ThreadPoolBuilder;

    #[test]
    fn test_eo_fold_in_place() {
        let z = Fr::from(4u64);

        // len = 1 (odd, zero-padded partner)
        let mut a1 = vec![Fr::from(9u64)];
        let f1 = eo_fold_in_place(&mut a1, z);
        assert_eq!(f1, &[Fr::from(9u64)]);

        // len = 2 (single pair)
        let mut a2 = vec![Fr::from(1u64), Fr::from(2u64)];
        let f2 = eo_fold_in_place(&mut a2, z);
        assert_eq!(f2, &[Fr::from(1u64) + z * Fr::from(2u64)]);

        // len = 3 (odd, last partner treated as 0)
        let mut a3 = vec![Fr::from(5u64), Fr::from(6u64), Fr::from(7u64)];
        let f3 = eo_fold_in_place(&mut a3, z);
        assert_eq!(f3.len(), 2);
        assert_eq!(f3[0], Fr::from(5u64) + z * Fr::from(6u64));
        assert_eq!(f3[1], Fr::from(7u64)); // paired with 0

        // len = 6, num_threads = 4 (pairs_per_round=4)
        // half=3 -> parallel handles first 3? Actually pairs_per_round=4 > half,
        // so parallel_pairs=0, tail handled sequentially. Should still be correct.
        let z = Fr::from(3u64);
        let mut a = vec![
            Fr::from(2u64),
            Fr::from(4u64),
            Fr::from(6u64),
            Fr::from(8u64),
            Fr::from(10u64),
            Fr::from(12u64),
        ];

        // Expected sequential result
        let mut expected = a.clone();
        {
            let half = (expected.len() + 1) / 2;
            for i in 0..half {
                let even_idx = 2 * i;
                let odd_idx = even_idx + 1;
                let even = expected.get(even_idx).copied().unwrap_or_else(Fr::zero);
                let odd = expected.get(odd_idx).copied().unwrap_or_else(Fr::zero);
                expected[i] = even + z * odd;
            }
            expected.truncate((expected.len() + 1) / 2);
        }

        ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(|| {
                let folded = eo_fold_in_place(&mut a, z);
                assert_eq!(folded, expected.as_slice());
            });
    }

    #[test]
    fn test_eo_fold_myint_to_field_basic() {
        let z = Fr::from(2u64);
        let a: Vec<MyInt> = vec![1, 3, 5, 7, 9];
        let mut out = vec![Fr::zero(); 3];
        eo_fold_myint_to_field(&a, z, &mut out);
        assert_eq!(out.len(), 3);
        assert_eq!(out[0], Fr::from(1) + z * Fr::from(3));
        assert_eq!(out[1], Fr::from(5) + z * Fr::from(7));
        // last odd partner missing -> treated as 0
        assert_eq!(out[2], Fr::from(9));
    }

    #[test]
    fn test_eo_fold_iterative_myint_to_field() {
        // a = [1,3,5,7], half=2, log_len=2
        // x0 folds to:
        //   out[0] = 1 + 2*3 = 7
        //   out[1] = 5 + 2*7 = 19
        // then fold again with x1=3:
        //   out = 7 + 3*19 = 64
        let a: Vec<MyInt> = vec![1, 3, 5, 7];
        let x_vec = vec![Fr::from(2u64), Fr::from(3u64)];
        let res = eo_fold_iterative_myint_to_field(&a, x_vec);
        assert_eq!(res, Fr::from(64u64));
    }

    #[test]
    fn test_eo_fold_iterative_edge_cases() {
        // len=1 (log_len=0): x_vec must be empty
        let a1: Vec<MyInt> = vec![5];
        let x_vec1: Vec<Fr> = vec![];
        let res1 = eo_fold_iterative_myint_to_field(&a1, x_vec1);
        assert_eq!(res1, Fr::from(5u64));

        // len=2 (single pair, log_len=1)
        let a2: Vec<MyInt> = vec![2, -3];
        let x_vec2 = vec![Fr::from(4u64)];
        let res2 = eo_fold_iterative_myint_to_field(&a2, x_vec2);
        // 2 + 4*(-3) = -10
        assert_eq!(res2, -Fr::from(10u64));

        // len=3 (log_len=2), last odd treated as zero in first fold
        let a3: Vec<MyInt> = vec![1, 2, 3];
        // x0: out=[1+5*2, 3] -> [11, 3]; x1: 11 + 7*3 = 32
        let x_vec3 = vec![Fr::from(5u64), Fr::from(7u64)];
        let res3 = eo_fold_iterative_myint_to_field(&a3, x_vec3);
        assert_eq!(res3, Fr::from(32u64));
    }
}
