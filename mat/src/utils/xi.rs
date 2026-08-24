//! Compute the n-vector xi by using log_2 n random challenges.
//!
//! Compute the n-vector psi by using the n-vector xi and a random s.
//!
//!
use ark_ff::PrimeField;

use rayon::prelude::*;

/// Compute the n-vector xi by using log_2 n random challenges.
///
pub fn xi_from_challenges<F: PrimeField>(challenges: &Vec<F>) -> Vec<F> {
    let log_n = challenges.len();

    let mut xi = vec![F::one()];

    let mut xi_right: Vec<F>;

    for j in 0..log_n {
        xi_right = xi
            .par_iter()
            .map(|a| *a * challenges[log_n - j - 1])
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

/// Compute the inner product of a vector and a tensor-structured-vector.
///
/// a_{j+1} = a_jL + x_j * a_jR
/// ip = a_{log_n}
///
pub fn xi_ip_from_challenges_and_vector<F: PrimeField>(vector: &Vec<F>, challenges: &Vec<F>) -> F {
    let expected_len = 1usize << challenges.len();
    assert!(
        vector.len() == expected_len,
        "vector length {} does not match 2^|challenges| {}",
        vector.len(),
        expected_len
    );

    let mut acc: Vec<F> = vector.to_vec();

    for &challenge in challenges {
        let half = acc.len() / 2;
        let (left, right) = acc.split_at_mut(half);
        left.par_iter_mut()
            .zip(right.par_iter())
            .for_each(|(l, r)| {
                *l += *r * challenge;
            });
        acc.truncate(half);
    }

    assert_eq!(acc.len(), 1, "folding did not converge to a single value");
    acc[0]
}

/// Compute the inner product of a vector and a tensor-structured-vector.
///
/// a_{j+1} = a_jL + x_j * a_jR
/// ip = a_{log_n}
///
pub fn xi_ip_from_challenges_and_vector_non_power_of_two<F: PrimeField>(
    vector: &Vec<F>,
    challenges: &Vec<F>,
) -> F {
    let expected_len = 1usize << challenges.len();

    // Pad to power-of-two length with zeros so folding is well-defined
    let mut acc: Vec<F> = vector.clone();
    if acc.len() < expected_len {
        acc.resize(expected_len, F::zero());
    }

    xi_ip_from_challenges_and_vector(&acc, challenges)
}

pub fn xi_get_at_position<F: PrimeField>(index: usize, challenges: &[F]) -> F {
    let log_n = challenges.len();
    let max_len = 1usize << log_n;
    assert!(
        index < max_len,
        "index {} exceeds 2^|challenges| {}",
        index,
        max_len
    );

    let mut result = F::one();
    for i in 0..log_n {
        if (index >> i) & 1 == 1 {
            result *= challenges[log_n - i - 1];
        }
    }
    result
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

pub fn xi_power_from_challenges_power_of_two<F: PrimeField>(base: F, challenges: &[F]) -> F {
    let log_n = challenges.len();
    let mut base_power = std::iter::successors(Some(base), |x| Some(*x * *x))
        .take(log_n)
        .collect::<Vec<F>>();
    base_power.reverse();
    xi_ip_from_challenges(&challenges.to_vec(), &base_power)
}

/// Handle when size is not power of two
pub fn xi_power_ip_with_challenges<F: PrimeField>(len: usize, base: F, challenges: &[F]) -> F {
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

    // Build vector [1, base, base^2, ...] of length len, pad with zero to 2^{|challenges|}
    let mut vec = vec![F::zero(); max_len];
    let mut pow = F::one();
    for i in 0..len {
        vec[i] = pow;
        pow *= base;
    }

    xi_ip_from_challenges_and_vector(&vec, &challenges.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::Zero;

    #[test]
    fn test_xi_generation() {
        // Test basic xi generation
        let challenges = vec![Fr::from(2u64), Fr::from(3u64)];
        let xi = xi_from_challenges(&challenges);

        // Should have 2^2 = 4 elements
        assert_eq!(xi.len(), 4);

        // Note: xi_from_challenges uses reverse order (log_n - j - 1)
        // So for [2, 3], it expands as: 1 -> [1, 1*3] -> [1, 3, 2, 6]
        assert_eq!(xi[0], Fr::from(1u64));
        assert_eq!(xi[1], Fr::from(3u64));
        assert_eq!(xi[2], Fr::from(2u64));
        assert_eq!(xi[3], Fr::from(6u64));
    }

    #[test]
    fn test_xi_ip_basic() {
        let challenges1 = vec![Fr::from(2u64), Fr::from(3u64)];
        let challenges2 = vec![Fr::from(5u64), Fr::from(7u64)];

        let result = xi_ip_from_challenges(&challenges1, &challenges2);

        // Should be (1 + 2*5) * (1 + 3*7) = 11 * 22 = 242
        assert_eq!(result, Fr::from(242u64));
    }

    #[test]
    fn test_vec_one_ip_power_of_two() {
        let challenges = vec![Fr::from(2u64), Fr::from(3u64)];

        // For len=4 (power of 2), should be (1+2)*(1+3) = 12
        let result = vec_one_ip_with_challenges(4, &challenges);
        assert_eq!(result, Fr::from(12u64));
    }

    #[test]
    fn test_vec_one_ip_non_power_of_two() {
        let challenges = vec![Fr::from(2u64), Fr::from(3u64)];

        // For len=3, should compute correctly
        let result = vec_one_ip_with_challenges(3, &challenges);

        // len=3 decomposes as: 2 (prefix) + 1 (remainder)
        // prefix (len=2): (1+3) = 4
        // remainder (len=1): 1
        // result = prefix + remainder * challenges[1] = 4 + 1*2 = 6
        assert_eq!(result, Fr::from(6u64));
    }

    #[test]
    fn test_xi_power_ip_with_challenges_power_of_two() {
        let challenges = vec![Fr::from(3u64), Fr::from(5u64)]; // |challenges| = 2 => max_len = 4
        let base = Fr::from(2u64);
        let len = 4usize;

        let vec = vec![
            Fr::from(1u64),
            Fr::from(2u64),
            Fr::from(4u64),
            Fr::from(8u64),
        ];
        let expected = xi_ip_from_challenges_and_vector(&vec, &challenges);

        let got = xi_power_ip_with_challenges(len, base, &challenges);
        assert_eq!(got, expected);
    }

    #[test]
    fn test_xi_power_ip_with_challenges_non_power_of_two() {
        let challenges = vec![Fr::from(3u64), Fr::from(5u64)]; // |challenges| = 2 => max_len = 4
        let base = Fr::from(2u64);
        let len = 3usize; // not a power of two

        let vec = vec![Fr::from(1u64), Fr::from(2u64), Fr::from(4u64), Fr::zero()];
        let expected = xi_ip_from_challenges_and_vector(&vec, &challenges);

        let got = xi_power_ip_with_challenges(len, base, &challenges);
        assert_eq!(got, expected);
    }
}
