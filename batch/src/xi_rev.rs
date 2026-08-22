//! Compute the n-vector xi by using log_2 n random challenges.
//!
//! Compute the n-vector psi by using the n-vector xi and a random s.
//!
//!
use crate::MyInt;
use ark_ff::PrimeField;

use rayon::prelude::*;

/// Compute the n-vector xi by using log_2 n random challenges.
///
pub fn xi_rev_from_challenges<F: PrimeField>(challenges: &Vec<F>) -> Vec<F> {
    let log_n = challenges.len();

    let mut xi = vec![F::one()];

    let mut xi_right: Vec<F>;

    for j in 0..log_n {
        xi_right = xi
            .par_iter()
            .map(|a| *a * challenges[j])
            .collect::<Vec<F>>();
        xi.append(&mut xi_right);
    }

    xi
}

/// Compute the inner product of two tensor-structured-vectors.
///
pub fn xi_ip_from_challenges<F: PrimeField>(challenges1: &Vec<F>, challenges2: &Vec<F>) -> F {
    let len1 = challenges1.len();
    let len2 = challenges2.len();
    let len = std::cmp::min(len1, len2);
    let challenges1 = &challenges1[len1 - len..len1].to_vec();
    let challenges2 = &challenges2[len2 - len..len2].to_vec();

    let mut result = F::one();

    for i in 0..len {
        let cur = challenges1[i] * challenges2[i];
        // println!("cur: {}", cur);
        result = result.mul(F::one().add(cur));
    }

    result
}

/// Sequential version of xi_rev_ip_from_challenges_and_vector to avoid nested parallelism overhead.
pub fn xi_rev_ip_from_challenges_and_vector_serial<F: PrimeField>(
    vector: &[F],
    challenges: &[F],
) -> F {
    let n = vector.len();
    if n == 0 {
        return F::zero();
    }
    if n == 1 {
        return vector[0];
    }

    let mut iter = challenges.iter().rev();
    // First round: fold directly from the immutable input without cloning it up front.
    let Some(&challenge_first) = iter.next() else {
        return vector[0];
    };

    let expected_len0 = 2usize.pow(challenges.len() as u32);
    let half0 = expected_len0 / 2;
    let mut acc: Vec<F> = (0..half0)
        .map(|i| {
            let l_val = vector.get(i).copied().unwrap_or_else(F::zero);
            let r_val = vector.get(i + half0).copied().unwrap_or_else(F::zero);
            l_val + r_val * challenge_first
        })
        .collect();

    for &challenge in iter {
        let len = acc.len();
        if len <= 1 {
            break;
        }
        let half = len / 2;
        acc = (0..half)
            .map(|i| {
                let l_val = acc[i];
                let r_val = acc[i + half];
                l_val + r_val * challenge
            })
            .collect();
    }
    acc.get(0).copied().unwrap_or_else(F::zero)
}

/// Sequential version of xi_rev_ip_myint_from_challenges_and_vector.
pub fn xi_rev_ip_myint_from_challenges_and_vector_serial<F: PrimeField>(
    vector: &[MyInt],
    challenges: &[F],
) -> F {
    let to_f = |x: MyInt| {
        if x >= 0 {
            F::from(x as u64)
        } else {
            -F::from((-(x as i32)) as u64)
        }
    };
    let n = vector.len();
    if n == 0 {
        return F::zero();
    }
    if n == 1 {
        return to_f(vector[0]);
    }

    let mut iter = challenges.iter().rev();
    // First round: fold directly from the immutable input without cloning it up front.
    let Some(&challenge_first) = iter.next() else {
        return to_f(vector[0]);
    };

    let expected_len0 = 2usize.pow(challenges.len() as u32);
    let half0 = expected_len0 / 2;
    let mut acc: Vec<F> = (0..half0)
        .map(|i| {
            let l_val = vector.get(i).map(|&x| to_f(x)).unwrap_or_else(F::zero);
            let r_val = vector
                .get(i + half0)
                .map(|&x| to_f(x))
                .unwrap_or_else(F::zero);
            l_val + r_val * challenge_first
        })
        .collect();

    for &challenge in iter {
        let len = acc.len();
        if len <= 1 {
            break;
        }
        let half = len / 2;
        acc = (0..half)
            .map(|i| {
                let l_val = acc[i];
                let r_val = acc[i + half];
                l_val + r_val * challenge
            })
            .collect();
    }
    acc.get(0).copied().unwrap_or_else(F::zero)
}

/// Compute the inner product of a vector and a tensor-structured-vector.
///
/// Reverse challenges then pair left/right halves (MSB-style).
pub fn xi_rev_ip_from_challenges_and_vector<F: PrimeField>(vector: &[F], challenges: &[F]) -> F {
    xi_rev_ip_from_challenges_and_vector_with_par(vector, challenges, true)
}

pub fn xi_rev_ip_from_challenges_and_vector_with_par<F: PrimeField>(
    vector: &[F],
    challenges: &[F],
    parallel: bool,
) -> F {
    if !parallel {
        return xi_rev_ip_from_challenges_and_vector_serial(vector, challenges);
    }
    let mut iter = challenges.iter().rev();
    // First round: fold directly from the immutable input without cloning it up front.
    let Some(&challenge_first) = iter.next() else {
        return vector.get(0).copied().unwrap_or_else(F::zero);
    };

    let len0 = vector.len();
    if len0 <= 1 {
        return vector.get(0).copied().unwrap_or_else(F::zero);
    }
    let expected_len0 = 2usize.pow(challenges.len() as u32);
    let half0 = expected_len0 / 2;
    let mut acc: Vec<F> = (0..half0)
        .into_par_iter()
        .map(|i| {
            let l_val = vector.get(i).copied().unwrap_or_else(F::zero);
            let r_val = vector.get(i + half0).copied().unwrap_or_else(F::zero);
            l_val + r_val * challenge_first
        })
        .collect();

    for &challenge in iter {
        let len = acc.len();
        if len <= 1 {
            break;
        }
        let expected_len = len.next_power_of_two();
        let half = expected_len / 2;
        acc = (0..half)
            .into_par_iter()
            .map(|i| {
                let l_val = acc.get(i).copied().unwrap_or_else(F::zero);
                let r_val = acc.get(i + half).copied().unwrap_or_else(F::zero);
                l_val + r_val * challenge
            })
            .collect();
    }
    acc.get(0).copied().unwrap_or_else(F::zero)
}

/// Compute the inner product of a vector and a tensor-structured-vector.
///
/// Reverse challenges then pair left/right halves (MSB-style).
pub fn xi_rev_ip_myint_from_challenges_and_vector<F: PrimeField>(
    vector: &[MyInt],
    challenges: &[F],
) -> F {
    xi_rev_ip_myint_from_challenges_and_vector_with_par(vector, challenges, true)
}

pub fn xi_rev_ip_myint_from_challenges_and_vector_with_par<F: PrimeField>(
    vector: &[MyInt],
    challenges: &[F],
    parallel: bool,
) -> F {
    if !parallel {
        return xi_rev_ip_myint_from_challenges_and_vector_serial(vector, challenges);
    }
    let to_f = |x: MyInt| {
        if x >= 0 {
            F::from(x as u64)
        } else {
            -F::from((-(x as i32)) as u64)
        }
    };

    let mut iter = challenges.iter().rev();
    // First round: fold directly from the immutable input without cloning it up front.
    let Some(&challenge_first) = iter.next() else {
        return to_f(vector.get(0).copied().unwrap_or(0));
    };

    let len0 = vector.len();
    if len0 <= 1 {
        return to_f(vector.get(0).copied().unwrap_or(0));
    }
    let expected_len0 = 2usize.pow(challenges.len() as u32);
    let half0 = expected_len0 / 2;
    let mut acc: Vec<F> = (0..half0)
        .into_par_iter()
        .map(|i| {
            let l_val = to_f(vector.get(i).copied().unwrap_or(0));
            let r_val = to_f(vector.get(i + half0).copied().unwrap_or(0));
            l_val + r_val * challenge_first
        })
        .collect();

    for &challenge in iter {
        let len = acc.len();
        if len <= 1 {
            break;
        }
        let expected_len = len.next_power_of_two();
        let half = expected_len / 2;
        acc = (0..half)
            .into_par_iter()
            .map(|i| {
                let l_val = acc.get(i).copied().unwrap_or_else(F::zero);
                let r_val = acc.get(i + half).copied().unwrap_or_else(F::zero);
                l_val + r_val * challenge
            })
            .collect();
    }
    acc.get(0).copied().unwrap_or_else(F::zero)
}

/// Compute the inner product of the odd part of a vector and a tensor-structured-vector.
pub fn ip_oe_myint_with_xi_rev<F: PrimeField>(a: &Vec<MyInt>, challenges: &[F]) -> F {
    ip_oe_myint_with_xi_rev_with_par(a, challenges, true)
}

pub fn xi_ip_oe_myint_with_xi_rev_par<F: PrimeField>(a: &Vec<MyInt>, challenges: &[F]) -> F {
    ip_oe_myint_with_xi_rev_with_par(a, challenges, true)
}

pub fn ip_oe_myint_with_xi_rev_with_par<F: PrimeField>(
    a: &Vec<MyInt>,
    challenges: &[F],
    parallel: bool,
) -> F {
    if challenges.is_empty() {
        return F::zero();
    }

    // Compute directly with odd elements using step access
    if parallel {
        xi_rev_ip_myint_from_step_vector(a, 1, &challenges[1..]) // start=1, step=2
    } else {
        xi_rev_ip_myint_from_step_vector_serial(a, 1, &challenges[1..])
    }
}

/// Compute the inner product of the even part of a vector and a tensor-structured-vector.
pub fn ip_eo_myint_with_xi_rev<F: PrimeField>(a: &Vec<MyInt>, challenges: &[F]) -> F {
    ip_eo_myint_with_xi_rev_with_par(a, challenges, true)
}

pub fn xi_ip_eo_myint_with_xi_rev_par<F: PrimeField>(a: &Vec<MyInt>, challenges: &[F]) -> F {
    ip_eo_myint_with_xi_rev_with_par(a, challenges, true)
}

pub fn ip_eo_myint_with_xi_rev_with_par<F: PrimeField>(
    a: &Vec<MyInt>,
    challenges: &[F],
    parallel: bool,
) -> F {
    if challenges.is_empty() {
        return F::zero();
    }

    // Compute directly with even elements using step access
    if parallel {
        xi_rev_ip_myint_from_step_vector(a, 0, &challenges[1..]) * challenges[0]
    // start=0, step=2
    } else {
        xi_rev_ip_myint_from_step_vector_serial(a, 0, &challenges[1..]) * challenges[0]
    }
}

/// Compute the inner product of the odd part of a vector and a tensor-structured-vector.
pub fn ip_oe_with_xi_rev<F: PrimeField>(a: &Vec<F>, challenges: &[F]) -> F {
    ip_oe_with_xi_rev_with_par(a, challenges, true)
}

pub fn xi_ip_oe_with_xi_rev_par<F: PrimeField>(a: &Vec<F>, challenges: &[F]) -> F {
    ip_oe_with_xi_rev_with_par(a, challenges, true)
}

pub fn ip_oe_with_xi_rev_with_par<F: PrimeField>(
    a: &Vec<F>,
    challenges: &[F],
    parallel: bool,
) -> F {
    if challenges.is_empty() {
        return F::zero();
    }

    // Use step access (odd indices start=1, step=2)
    if parallel {
        xi_rev_ip_from_step_vector(a, 1, &challenges[1..])
    } else {
        xi_rev_ip_from_step_vector_serial(a, 1, &challenges[1..])
    }
}

/// Helper function to compute xi_rev IP with step access for Field vectors to avoid extra allocation
fn xi_rev_ip_from_step_vector<F: PrimeField>(vector: &Vec<F>, start: usize, challenges: &[F]) -> F {
    let mut iter = challenges.iter().rev();
    // First round: fold directly from the immutable input without cloning it up front.
    let Some(&challenge_first) = iter.next() else {
        let first_idx = start;
        return vector.get(first_idx).copied().unwrap_or_else(F::zero);
    };

    // Calculate the length of the stepped sequence
    let step = 2;
    let len0 = (vector.len() - start + step - 1) / step;
    if len0 <= 1 {
        let first_idx = start;
        return vector.get(first_idx).copied().unwrap_or_else(F::zero);
    }

    let expected_len0 = 2usize.pow(challenges.len() as u32);
    let half0 = expected_len0 / 2;
    let mut acc: Vec<F> = (0..half0)
        .into_par_iter()
        .map(|i| {
            let l_idx = start + i * step;
            let r_idx = start + (i + half0) * step;
            let l_val = vector.get(l_idx).copied().unwrap_or_else(F::zero);
            let r_val = vector.get(r_idx).copied().unwrap_or_else(F::zero);
            l_val + r_val * challenge_first
        })
        .collect();

    for &challenge in iter {
        let len = acc.len();
        if len <= 1 {
            break;
        }
        let expected_len = len.next_power_of_two();
        let half = expected_len / 2;
        acc = (0..half)
            .into_par_iter()
            .map(|i| {
                let l_val = acc.get(i).copied().unwrap_or_else(F::zero);
                let r_val = acc.get(i + half).copied().unwrap_or_else(F::zero);
                l_val + r_val * challenge
            })
            .collect();
    }

    acc.get(0).copied().unwrap_or_else(F::zero)
}

fn xi_rev_ip_from_step_vector_serial<F: PrimeField>(
    vector: &Vec<F>,
    start: usize,
    challenges: &[F],
) -> F {
    let mut iter = challenges.iter().rev();
    let Some(&challenge_first) = iter.next() else {
        return vector.get(start).copied().unwrap_or_else(F::zero);
    };

    let step = 2;
    let len0 = (vector.len() - start + step - 1) / step;
    if len0 <= 1 {
        return vector.get(start).copied().unwrap_or_else(F::zero);
    }

    let expected_len0 = 2usize.pow(challenges.len() as u32);
    let half0 = expected_len0 / 2;
    let mut acc: Vec<F> = (0..half0)
        .map(|i| {
            let l_idx = start + i * step;
            let r_idx = start + (i + half0) * step;
            let l_val = vector.get(l_idx).copied().unwrap_or_else(F::zero);
            let r_val = vector.get(r_idx).copied().unwrap_or_else(F::zero);
            l_val + r_val * challenge_first
        })
        .collect();

    for &challenge in iter {
        let len = acc.len();
        if len <= 1 {
            break;
        }
        let expected_len = len.next_power_of_two();
        let half = expected_len / 2;
        acc = (0..half)
            .map(|i| {
                let l_val = acc.get(i).copied().unwrap_or_else(F::zero);
                let r_val = acc.get(i + half).copied().unwrap_or_else(F::zero);
                l_val + r_val * challenge
            })
            .collect();
    }
    acc.get(0).copied().unwrap_or_else(F::zero)
}

/// Helper function to compute xi_rev IP with step access for MyInt to avoid any allocations
fn xi_rev_ip_myint_from_step_vector<F: PrimeField>(
    vector: &Vec<MyInt>,
    start: usize,
    challenges: &[F],
) -> F {
    use rayon::prelude::*;

    let to_f = |x: MyInt| {
        if x >= 0 {
            F::from(x as u64)
        } else {
            -F::from((-(x as i32)) as u64)
        }
    };

    let mut iter = challenges.iter().rev();
    // First round: fold directly from the immutable input without cloning it up front.
    let Some(&challenge_first) = iter.next() else {
        let first_idx = start;
        return vector
            .get(first_idx)
            .map(|&x| to_f(x))
            .unwrap_or_else(F::zero);
    };

    // Calculate the length of the stepped sequence
    let step = 2;
    let len0 = (vector.len() - start + step - 1) / step;
    if len0 <= 1 {
        let first_idx = start;
        return vector
            .get(first_idx)
            .map(|&x| to_f(x))
            .unwrap_or_else(F::zero);
    }

    let expected_len0 = 2usize.pow(challenges.len() as u32);
    let half0 = expected_len0 / 2;
    let mut acc: Vec<F> = (0..half0)
        .into_par_iter()
        .map(|i| {
            let l_idx = start + i * step;
            let r_idx = start + (i + half0) * step;
            let l_val = vector.get(l_idx).map(|&x| to_f(x)).unwrap_or_else(F::zero);
            let r_val = vector.get(r_idx).map(|&x| to_f(x)).unwrap_or_else(F::zero);
            l_val + r_val * challenge_first
        })
        .collect();

    for &challenge in iter {
        let len = acc.len();
        if len <= 1 {
            break;
        }
        let expected_len = len.next_power_of_two();
        let half = expected_len / 2;
        acc = (0..half)
            .into_par_iter()
            .map(|i| {
                let l_val = acc.get(i).copied().unwrap_or_else(F::zero);
                let r_val = acc.get(i + half).copied().unwrap_or_else(F::zero);
                l_val + r_val * challenge
            })
            .collect();
    }

    acc.get(0).copied().unwrap_or_else(F::zero)
}

fn xi_rev_ip_myint_from_step_vector_serial<F: PrimeField>(
    vector: &Vec<MyInt>,
    start: usize,
    challenges: &[F],
) -> F {
    let to_f = |x: MyInt| {
        if x >= 0 {
            F::from(x as u64)
        } else {
            -F::from((-(x as i32)) as u64)
        }
    };

    let mut iter = challenges.iter().rev();
    let Some(&challenge_first) = iter.next() else {
        return vector.get(start).map(|&x| to_f(x)).unwrap_or_else(F::zero);
    };

    let step = 2;
    let len0 = (vector.len() - start + step - 1) / step;
    if len0 <= 1 {
        return vector.get(start).map(|&x| to_f(x)).unwrap_or_else(F::zero);
    }

    let expected_len0 = 2usize.pow(challenges.len() as u32);
    let half0 = expected_len0 / 2;
    let mut acc: Vec<F> = (0..half0)
        .map(|i| {
            let l_idx = start + i * step;
            let r_idx = start + (i + half0) * step;
            let l_val = vector.get(l_idx).map(|&x| to_f(x)).unwrap_or_else(F::zero);
            let r_val = vector.get(r_idx).map(|&x| to_f(x)).unwrap_or_else(F::zero);
            l_val + r_val * challenge_first
        })
        .collect();

    for &challenge in iter {
        let len = acc.len();
        if len <= 1 {
            break;
        }
        let expected_len = len.next_power_of_two();
        let half = expected_len / 2;
        acc = (0..half)
            .map(|i| {
                let l_val = acc.get(i).copied().unwrap_or_else(F::zero);
                let r_val = acc.get(i + half).copied().unwrap_or_else(F::zero);
                l_val + r_val * challenge
            })
            .collect();
    }
    acc.get(0).copied().unwrap_or_else(F::zero)
}

/// Compute the inner product of the even part of a vector and a tensor-structured-vector.
pub fn ip_eo_with_xi_rev<F: PrimeField>(a: &Vec<F>, challenges: &[F]) -> F {
    ip_eo_with_xi_rev_with_par(a, challenges, true)
}

pub fn xi_ip_eo_with_xi_rev_par<F: PrimeField>(a: &Vec<F>, challenges: &[F]) -> F {
    ip_eo_with_xi_rev_with_par(a, challenges, true)
}

pub fn ip_eo_with_xi_rev_with_par<F: PrimeField>(
    a: &Vec<F>,
    challenges: &[F],
    parallel: bool,
) -> F {
    if challenges.is_empty() {
        return F::zero();
    }

    // Even part: start=0, step=2
    if parallel {
        xi_rev_ip_from_step_vector(a, 0, &challenges[1..]) * challenges[0]
    } else {
        xi_rev_ip_from_step_vector_serial(a, 0, &challenges[1..]) * challenges[0]
    }
}

/// Compute the projection of an all-ones vector of length `len` onto the xi
/// basis defined by `challenges`.
pub fn vec_one_ip_with_challenges<F: PrimeField>(len: usize, challenges: &[F]) -> F {
    if len == 0 {
        return F::zero();
    }

    let max_len = 1usize << challenges.len();
    assert!(
        len <= max_len,
        "vector length {} exceeds 2^|challenges| {}",
        len,
        max_len
    );

    fn helper<F: PrimeField>(len: usize, challenges: &[F], total_log: usize) -> F {
        if len.is_power_of_two() {
            let mut result = F::one();
            let log_len = len.trailing_zeros() as usize;
            for i in total_log - log_len..total_log {
                result *= F::one() + challenges[i];
            }
            return result;
        }

        let log_len = len.ilog2() as usize;
        let prefix_len = 1usize << log_len;
        let part_1 = helper::<F>(prefix_len, challenges, total_log);
        let coeff = challenges[total_log - log_len - 1];
        let part_2 = helper::<F>(len - prefix_len, challenges, total_log);
        part_1 + part_2 * coeff
    }

    helper(len, challenges, challenges.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{utils::eo_fold_iterative_myint_to_field, MyInt};
    use ark_bls12_381::Fr;

    #[test]
    fn test_xi_ip_eo_myint_consistency() {
        // Here ip_eo_myint_with_xi_rev is (Even BVA * Odd TBV).
        // Odd TBV indices have bit 0 = 1, so they all have factor challenges[0].
        let a: Vec<MyInt> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let challenges = vec![Fr::from(10u64), Fr::from(20u64), Fr::from(30u64)];

        let res = ip_eo_myint_with_xi_rev::<Fr>(&a, &challenges);

        let to_f = |x: MyInt| {
            if x >= 0 {
                Fr::from(x as u64)
            } else {
                -Fr::from((-(x as i32)) as u64)
            }
        };
        let a_even: Vec<Fr> = a.iter().step_by(2).map(|&x| to_f(x)).collect(); // [a0, a2, a4, a6]

        // Reference: bit 0 factor (challenges[0]) is applied to ALL odd TBV elements.
        let sub_challenges: Vec<Fr> = challenges[1..].to_vec();
        let xi_sub = xi_rev_from_challenges(&sub_challenges);

        let expected = xi_sub
            .iter()
            .zip(a_even.iter())
            .fold(Fr::from(0u64), |acc, (&xi, &v)| acc + xi * v)
            * challenges[0]; // Multiply by bit 0 factor

        assert_eq!(
            res, expected,
            "Manual dot product should match ip_eo_myint_with_xi_rev"
        );
    }

    #[test]
    fn test_xi_ip_oe_myint_consistency() {
        // Here ip_oe_myint_with_xi_rev is (Odd BVA * Even TBV).
        // Even TBV indices have bit 0 = 0, so factor is 1.
        let a: Vec<MyInt> = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let challenges = vec![Fr::from(10u64), Fr::from(20u64), Fr::from(30u64)];

        let res = ip_oe_myint_with_xi_rev::<Fr>(&a, &challenges);

        let to_f = |x: MyInt| {
            if x >= 0 {
                Fr::from(x as u64)
            } else {
                -Fr::from((-(x as i32)) as u64)
            }
        };
        let a_odd: Vec<Fr> = a.iter().skip(1).step_by(2).map(|&x| to_f(x)).collect(); // [a1, a3, a5, a7]

        let sub_challenges: Vec<Fr> = challenges[1..].to_vec();
        let xi_sub = xi_rev_from_challenges(&sub_challenges);

        let expected = a_odd
            .iter()
            .zip(xi_sub.iter())
            .fold(Fr::from(0u64), |acc, (&v, &xi)| acc + v * xi);

        assert_eq!(
            res, expected,
            "Manual dot product should match ip_oe_myint_with_xi_rev"
        );
    }

    #[test]
    fn test_xi_ip_and_xi_myint_ip_match_expanded_dot() {
        // Full-length case: xi_ip (field) == xi_myint_ip (myint) == expanded-xi dot product.
        let challenges = vec![Fr::from(3u64), Fr::from(5u64), Fr::from(7u64)];
        let n = 1usize << challenges.len();

        let vector_myint: Vec<MyInt> = vec![1, -2, 3, -4, 5, -6, 7, -8];
        assert_eq!(vector_myint.len(), n);

        let to_f = |x: MyInt| {
            if x >= 0 {
                Fr::from(x as u64)
            } else {
                -Fr::from((-(x as i32)) as u64)
            }
        };
        let vector_field: Vec<Fr> = vector_myint.iter().copied().map(to_f).collect();

        let xi = xi_rev_from_challenges(&challenges);
        let expanded = xi
            .iter()
            .zip(vector_field.iter())
            .fold(Fr::from(0u64), |acc, (&w, &v)| acc + w * v);

        let res_field = xi_rev_ip_from_challenges_and_vector(&vector_field, &challenges);
        let res_myint = xi_rev_ip_myint_from_challenges_and_vector(&vector_myint, &challenges);

        assert_eq!(expanded, res_field);
        assert_eq!(expanded, res_myint);
    }

    #[test]
    fn test_xi_ip_matches_iterative_fold() {
        // Same setup as above but compare with utils' iterative fold over MyInt
        let vector_myint: Vec<MyInt> = vec![1, 2, 3, 4];
        let v_field: Vec<Fr> = vector_myint
            .iter()
            .map(|&x| {
                if x >= 0 {
                    Fr::from(x as u64)
                } else {
                    -Fr::from((-(x as i32)) as u64)
                }
            })
            .collect();
        let challenges = vec![Fr::from(5u64), Fr::from(7u64)];

        let _xi = xi_rev_from_challenges(&challenges);

        // xi in-place fold
        let res_xi = xi_rev_ip_from_challenges_and_vector(&v_field, &challenges);

        // iterative fold over MyInt, same challenges order
        let res_iter = eo_fold_iterative_myint_to_field(&vector_myint, challenges.clone());

        assert_eq!(res_xi.clone(), res_iter);
    }
}
