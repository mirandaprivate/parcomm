use crate::MyInt;
use ark_ff::PrimeField;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use rayon::prelude::*;

/// Generate deterministic vectors with lengths 2^max_log, ..., 2, 1.
///
/// Repeating one byte per vector makes generation bandwidth-bound and keeps
/// benchmark setup separate from prover arithmetic.
pub fn generate_deterministic_vecs_decreasing_sizes(max_log: usize) -> Vec<Vec<MyInt>> {
    let vector_count = max_log.checked_add(1).expect("vector count overflow");
    (0..vector_count)
        .into_par_iter()
        .map(|offset| {
            let log_size = max_log - offset;
            let magnitude = (log_size % 31 + 1) as MyInt;
            let value = if log_size & 1 == 0 {
                magnitude
            } else {
                -magnitude
            };
            vec![value; 1usize << log_size]
        })
        .collect()
}

/// Generate deterministic vectors with a fixed size.
pub fn generate_deterministic_vecs_uniform_sizes(
    log_single: usize,
    log_n: usize,
) -> Vec<Vec<MyInt>> {
    let num = 1usize << (log_n - log_single);
    let size = 1usize << log_single;

    (0..num)
        .into_par_iter()
        .map(|vector_index| {
            let magnitude = (vector_index % 31 + 1) as MyInt;
            let value = if vector_index & 1 == 0 {
                magnitude
            } else {
                -magnitude
            };
            vec![value; size]
        })
        .collect()
}

/// Generate random challenges of a given length.
///
/// Each index draws from its own seeded stream, so the vector stays
/// reproducible and the elements stay distinct when generated in parallel.
pub fn generate_random_challenges<F: PrimeField>(len: usize) -> Vec<F> {
    (0..len)
        .into_par_iter()
        .map(|index| {
            let mut seed = [0_u8; 32];
            seed[..8].copy_from_slice(&(index as u64).to_le_bytes());
            let mut rng = StdRng::from_seed(seed);
            F::rand(&mut rng)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;

    #[test]
    fn test_generate_deterministic_vecs_decreasing_sizes() {
        let vecs = generate_deterministic_vecs_decreasing_sizes(4);
        let lengths = vecs.iter().map(Vec::len).collect::<Vec<_>>();
        assert_eq!(lengths, vec![16, 8, 4, 2, 1]);
        assert_eq!(lengths.iter().sum::<usize>(), 31);
    }

    #[test]
    fn test_generate_deterministic_vecs_uniform_sizes() {
        let logn = 10;
        let log_single = 7;
        let vecs = generate_deterministic_vecs_uniform_sizes(log_single, logn);

        assert_eq!(vecs.len(), 8);
        for v in vecs {
            assert_eq!(v.len(), 128);
        }
    }

    #[test]
    fn test_generate_random_challenges() {
        let len = 10;
        let challenges = generate_random_challenges::<Fr>(len);
        assert_eq!(challenges.len(), len);
        let distinct = challenges.iter().collect::<std::collections::BTreeSet<_>>();
        assert_eq!(distinct.len(), len);
        assert_eq!(challenges, generate_random_challenges::<Fr>(len));
    }
}
