use ark_crypto_primitives::sponge::Absorb;
/// Batch prove projections of the same matrix at multiple points (xl, xr)
/// Unblinded, transcript-based (no commitments).
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::UniformRand;

use fs::helper_trans::Transcript;
use mat::utils::matdef::MatOps;
use mat::utils::xi;
use mat::DenseMatFieldCM;
use rayon::prelude::*;

use super::litebullet::LiteBullet;

/// Prove multiple projections of a single matrix at given points.
pub struct BatchPoints<F>
where
    F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
{
    pub hats: Vec<F>,
    pub points: Vec<(Vec<F>, Vec<F>)>,
    pub shape: (usize, usize),
}

impl<F> BatchPoints<F>
where
    F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
{
    pub fn new(hats: Vec<F>, points: Vec<(Vec<F>, Vec<F>)>, shape: (usize, usize)) -> Self {
        Self {
            hats,
            points,
            shape,
        }
    }

    /// Prover: batch prove projections of the same matrix at multiple points.
    pub fn reduce_prover(
        &self,
        trans_seq: &mut Transcript<F>,
        mat: &DenseMatFieldCM<F>,
    ) -> (F, (Vec<F>, Vec<F>)) {
        let (m, n) = self.shape;
        let num = self.points.len();

        // Pad dimensions to next power of two for correct column-major folding
        let m_padded = m.next_power_of_two();
        let n_padded = n.next_power_of_two();
        let len_padded = m_padded * n_padded;

        // Generate challenge 'c' for batching
        let challenge = trans_seq.gen_challenge();

        let mut powers = Vec::with_capacity(num);
        let mut cur = F::one();
        for _ in 0..num {
            powers.push(cur);
            cur *= challenge;
        }

        // batch_hat = sum proj_lr(mat, point_i) * c^i
        let mut batch_hat = F::zero();
        for (i, (xl, xr)) in self.points.iter().enumerate() {
            let hat = mat.proj_lr(xl, xr);
            batch_hat += hat * powers[i];
        }

        // batch_xi_flatten = sum xi(point_i) * c^i
        let mut batch_xi_flatten: Vec<F> = vec![F::zero(); len_padded];
        for (i, (xl, xr)) in self.points.iter().enumerate() {
            let mut combined = xr.clone();
            combined.extend_from_slice(xl);
            let xi_vec = xi::xi_from_challenges(&combined);
            // xi_vec already has length len_padded
            for j in 0..len_padded {
                batch_xi_flatten[j] += xi_vec[j] * powers[i];
            }
        }

        // Flatten the input matrix 'mat' with zero padding to (m_padded, n_padded)
        let mat_flatten: Vec<F> = (0..len_padded)
            .into_par_iter()
            .map(|idx| {
                let row = idx % m_padded;
                let col = idx / m_padded;
                if row < m && col < n {
                    mat.data[col][row]
                } else {
                    F::zero()
                }
            })
            .collect();

        let lb = LiteBullet::new(batch_hat, len_padded);
        let (hat_a, _hat_b, point_a, _point_b) =
            lb.reduce_prover(trans_seq, &mat_flatten, &batch_xi_flatten);

        let _log_m = m_padded.trailing_zeros() as usize;
        let log_n = n_padded.trailing_zeros() as usize;

        // For column-major, first log_n are columns (xr), next log_m are rows (xl)
        let xr_prime = point_a[..log_n].to_vec();
        let xl_prime = point_a[log_n..].to_vec();

        (hat_a, (xl_prime, xr_prime))
    }

    /// Verifier: read challenge and batch_hat, then run LiteBullet verification.
    pub fn verify_as_subprotocol(
        &self,
        trans_seq: &mut Transcript<F>,
    ) -> (bool, F, (Vec<F>, Vec<F>)) {
        let (m, n) = self.shape;
        let num = self.points.len();

        let m_padded = m.next_power_of_two();
        let n_padded = n.next_power_of_two();
        let len_padded = m_padded * n_padded;

        // 1. Read the batch challenge 'c' (prover already generated and recorded)
        let challenge = trans_seq.get_at_position(trans_seq.pointer);
        trans_seq.pointer += 1;

        let mut powers = Vec::with_capacity(num);
        let mut cur = F::one();
        for _ in 0..num {
            powers.push(cur);
            cur *= challenge;
        }

        let batch_hat: F = self
            .hats
            .iter()
            .zip(powers.iter())
            .map(|(hat, power)| *hat * *power)
            .sum();

        let lb = LiteBullet::new(batch_hat, len_padded);
        let (ok1, hat_a, hat_b, point_a_raw, point_b_raw) = lb.verify_as_subprotocol(trans_seq);

        // 3. Verify that hat_b matches the expected combined xi projection
        let mut hat_b_check = F::zero();
        for i in 0..num {
            let (xl, xr) = &self.points[i];
            let mut combined = xr.clone();
            combined.extend_from_slice(xl);
            let cur_b_hat = xi::xi_ip_from_challenges(&combined, &point_b_raw);
            hat_b_check += cur_b_hat * powers[i];
        }

        let ok2 = hat_b == hat_b_check;

        let _log_m = m_padded.trailing_zeros() as usize;
        let log_n = n_padded.trailing_zeros() as usize;
        let xr_prime = point_a_raw[..log_n].to_vec();
        let xl_prime = point_a_raw[log_n..].to_vec();

        (ok1 && ok2, hat_a, (xl_prime, xr_prime))
    }

    /// Full verify: recompute projections and compare.
    pub fn verify(&self, trans_seq: &mut Transcript<F>, mat: &DenseMatFieldCM<F>) -> bool {
        let (ok1, hat_a, point_a) = self.verify_as_subprotocol(trans_seq);
        if !ok1 {
            return false;
        }

        let ok2 = hat_a == mat.proj_lr(&point_a.0, &point_a.1);
        let ok3 = trans_seq.fs.verify_fs();

        ok1 && ok2 && ok3
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::Zero;

    #[test]
    fn test_batchpoints_reduce_and_verify() {
        // 2x2 matrix in column-major format
        let mat = DenseMatFieldCM::<Fr>::from_data(vec![
            vec![Fr::from(1u64), Fr::from(3u64)], // col 0
            vec![Fr::from(2u64), Fr::from(4u64)], // col 1
        ]);

        // two points
        let points = vec![
            (vec![Fr::from(2u64)], vec![Fr::from(5u64)]),
            (vec![Fr::from(7u64)], vec![Fr::from(11u64)]),
        ];

        // Compute hats for each point
        let hats: Vec<Fr> = points.iter().map(|(xl, xr)| mat.proj_lr(xl, xr)).collect();

        let protocol = BatchPoints::<Fr>::new(hats, points.clone(), (2, 2));

        let mut transcript = Transcript::new(Fr::zero());
        let _ = protocol.reduce_prover(&mut transcript, &mat);

        let mut trans_v = transcript.clone();
        trans_v.reset_pointer();

        let ok = protocol.verify(&mut trans_v, &mat);
        assert!(
            ok,
            "BatchPoints verify should return true for matching projections"
        );
    }

    #[test]
    fn test_batchpoints_non_power_of_two() {
        // 3x2 matrix (m=3, n=2) to trigger padding to 4x2 internally
        let mat = DenseMatFieldCM::<Fr>::from_data(vec![
            vec![Fr::from(1u64), Fr::from(4u64), Fr::from(7u64)], // col 0
            vec![Fr::from(2u64), Fr::from(5u64), Fr::from(8u64)], // col 1
        ]);

        // log_m = 2 (padding m from 3 to 4), log_n = 1
        let points = vec![
            (vec![Fr::from(2u64), Fr::from(3u64)], vec![Fr::from(5u64)]),
            (vec![Fr::from(7u64), Fr::from(11u64)], vec![Fr::from(13u64)]),
        ];

        // hats for each point using the original (unpadded) matrix projection
        let hats: Vec<Fr> = points.iter().map(|(xl, xr)| mat.proj_lr(xl, xr)).collect();

        let protocol = BatchPoints::<Fr>::new(hats, points.clone(), (3, 2));

        let mut transcript = Transcript::new(Fr::zero());
        let _ = protocol.reduce_prover(&mut transcript, &mat);

        let mut trans_v = transcript.clone();
        trans_v.reset_pointer();

        let ok = protocol.verify(&mut trans_v, &mat);
        assert!(
            ok,
            "BatchPoints verify should handle non-power-of-two shapes"
        );
    }
}
