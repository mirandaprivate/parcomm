/// Hadamard protocol (unblinded, Transcript-based, field only)
///
/// Minimal changes to align with `litebullet.rs`: no SRS/commitments,
/// uses `Transcript` for challenges/responses, and folds flattened inputs.
use rayon::prelude::*;

use ark_crypto_primitives::sponge::Absorb;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::UniformRand;

use fs::helper_trans::Transcript;
use mat::utils::xi;
use mat::DenseMatFieldCM;

use crate::atomic_protocol::EvalWithPointMat;

use super::litebullet::LiteBullet;

/// Matrix MLE claims produced by a Hadamard-product reduction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HadamardReduction<F> {
    pub left: EvalWithPointMat<F>,
    pub right: EvalWithPointMat<F>,
}

pub struct Hadamard<F>
where
    F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
{
    pub c_hat: F,
    pub point: (Vec<F>, Vec<F>), // (xl_init, xr_init)
    pub shape: (usize, usize),   // (m, n)
}

impl<F> Hadamard<F>
where
    F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
{
    pub fn new(c_hat_value: F, shape_value: (usize, usize), point_value: (Vec<F>, Vec<F>)) -> Self {
        Self {
            c_hat: c_hat_value,
            point: point_value,
            shape: shape_value,
        }
    }

    /// Prover: flatten Hadamard inputs and run litebullet folding.
    /// Returns (a_reduce, b_reduce, point_a, point_b)
    pub fn reduce_prover(
        &self,
        trans_seq: &mut Transcript<F>,
        mat_a: &DenseMatFieldCM<F>,
        mat_b: &DenseMatFieldCM<F>,
    ) -> (F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) {
        let (m, n) = mat_a.shape;
        let (xl_init, xr_init) = (&self.point.0, &self.point.1);

        // Pad dimensions to next power of two for correct column-major folding
        let m_padded = m.next_power_of_two();
        let n_padded = n.next_power_of_two();

        let xi_l = xi::xi_from_challenges::<F>(xl_init);
        let xi_r = xi::xi_from_challenges::<F>(xr_init);

        // Flatten mat_a with initial projection factors (Column-Major: idx = col * m_padded + row)
        // Pad with zeros for indices beyond original matrix dimensions
        let vec_a_prime: Vec<F> = (0..m_padded * n_padded)
            .into_par_iter()
            .map(|idx| {
                let row = idx % m_padded;
                let col = idx / m_padded;
                if row < m && col < n {
                    // DenseMatFieldCM is column-major: data[col][row]
                    xi_l[row] * xi_r[col] * mat_a.data[col][row]
                } else {
                    F::zero()
                }
            })
            .collect();

        // Flatten mat_b directly with zero padding
        let vec_b: Vec<F> = (0..m_padded * n_padded)
            .into_par_iter()
            .map(|idx| {
                let row = idx % m_padded;
                let col = idx / m_padded;
                if row < m && col < n {
                    mat_b.data[col][row]
                } else {
                    F::zero()
                }
            })
            .collect();

        let lb = LiteBullet::<F>::new(self.c_hat, m_padded * n_padded);
        let (a_reduce, b_reduce, challenges_inv, challenges) =
            lb.reduce_prover(trans_seq, &vec_a_prime, &vec_b);

        let _log_m = xl_init.len();
        let log_n = xr_init.len();

        // LiteBullet folds high bits first. In Column-Major, high bits are Col (xr).
        let xr_inv = &challenges_inv[..log_n];
        let xl_inv = &challenges_inv[log_n..];

        let xr = &challenges[..log_n];
        let xl = &challenges[log_n..];

        // point_a = point_init * folding_challenges_inv (element-wise)
        let new_xl_a: Vec<F> = xl_init
            .iter()
            .zip(xl_inv.iter())
            .map(|(&a, &inv)| a * inv)
            .collect();
        let new_xr_a: Vec<F> = xr_init
            .iter()
            .zip(xr_inv.iter())
            .map(|(&a, &inv)| a * inv)
            .collect();

        (
            a_reduce,
            b_reduce,
            (new_xl_a, new_xr_a),
            (xl.to_vec(), xr.to_vec()),
        )
    }

    /// Reduce the Hadamard claim to structured claims for both input matrices.
    pub fn reduce_prover_claims(
        &self,
        trans_seq: &mut Transcript<F>,
        mat_a: &DenseMatFieldCM<F>,
        mat_b: &DenseMatFieldCM<F>,
    ) -> HadamardReduction<F> {
        let (left_value, right_value, left_point, right_point) =
            self.reduce_prover(trans_seq, mat_a, mat_b);
        HadamardReduction {
            left: EvalWithPointMat {
                value: left_value,
                point: left_point,
            },
            right: EvalWithPointMat {
                value: right_value,
                point: right_point,
            },
        }
    }

    pub fn verify_as_subprotocol(
        &self,
        trans_seq: &mut Transcript<F>,
    ) -> (bool, F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) {
        let (m, n) = self.shape;
        let (xl_init, xr_init) = (&self.point.0, &self.point.1);

        // Pad dimensions to next power of two to match prover
        let m_padded = m.next_power_of_two();
        let n_padded = n.next_power_of_two();

        let lb = LiteBullet::<F>::new(self.c_hat, m_padded * n_padded);
        let (flag, a_hat, b_hat, challenges_inv, challenges) = lb.verify_as_subprotocol(trans_seq);

        let _log_m = xl_init.len();
        let log_n = xr_init.len();

        let xr_inv = &challenges_inv[..log_n];
        let xl_inv = &challenges_inv[log_n..];
        let xr = &challenges[..log_n];
        let xl = &challenges[log_n..];

        let new_xl_a: Vec<F> = xl_init
            .iter()
            .zip(xl_inv.iter())
            .map(|(&a, &inv)| a * inv)
            .collect();
        let new_xr_a: Vec<F> = xr_init
            .iter()
            .zip(xr_inv.iter())
            .map(|(&a, &inv)| a * inv)
            .collect();

        (
            flag,
            a_hat,
            b_hat,
            (new_xl_a, new_xr_a),
            (xl.to_vec(), xr.to_vec()),
        )
    }

    /// Verify the reduction and return commitment-independent input claims.
    pub fn verify_as_subprotocol_claims(
        &self,
        trans_seq: &mut Transcript<F>,
    ) -> (bool, HadamardReduction<F>) {
        let (ok, left_value, right_value, left_point, right_point) =
            self.verify_as_subprotocol(trans_seq);
        (
            ok,
            HadamardReduction {
                left: EvalWithPointMat {
                    value: left_value,
                    point: left_point,
                },
                right: EvalWithPointMat {
                    value: right_value,
                    point: right_point,
                },
            },
        )
    }

    pub fn verify(
        &self,
        trans_seq: &mut Transcript<F>,
        mat_a: &DenseMatFieldCM<F>,
        mat_b: &DenseMatFieldCM<F>,
    ) -> bool {
        let (flag, a_hat, b_hat, point_a, point_b) = self.verify_as_subprotocol(trans_seq);

        // Manual verification of reduced values using updated points
        let xi_l_a = xi::xi_from_challenges::<F>(&point_a.0);
        let xi_r_a = xi::xi_from_challenges::<F>(&point_a.1);
        let mut a_check_val = F::zero();
        for i in 0..mat_a.shape.0 {
            for j in 0..mat_a.shape.1 {
                a_check_val += mat_a.data[j][i] * xi_l_a[i] * xi_r_a[j];
            }
        }

        let xi_l_b = xi::xi_from_challenges::<F>(&point_b.0);
        let xi_r_b = xi::xi_from_challenges::<F>(&point_b.1);
        let mut b_check_val = F::zero();
        for i in 0..mat_b.shape.0 {
            for j in 0..mat_b.shape.1 {
                b_check_val += mat_b.data[j][i] * xi_l_b[i] * xi_r_b[j];
            }
        }

        flag && a_hat == a_check_val && b_hat == b_check_val && trans_seq.fs.verify_fs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::Zero;
    use mat::utils::matdef::DenseMatFieldCM;
    use mat::utils::matdef::MatOps;

    #[test]
    fn test_hadamard_reduce_and_verify() {
        let m = 2;
        let n = 2;
        // column-major storage: each inner vec is a column
        let mat_a = DenseMatFieldCM::<Fr>::from_data(vec![
            vec![Fr::from(1u64), Fr::from(3u64)], // col 0
            vec![Fr::from(2u64), Fr::from(4u64)], // col 1
        ]);
        let mat_b = DenseMatFieldCM::<Fr>::from_data(vec![
            vec![Fr::from(5u64), Fr::from(7u64)], // col 0
            vec![Fr::from(6u64), Fr::from(8u64)], // col 1
        ]);

        let xl = vec![Fr::from(2u64)];
        let xr = vec![Fr::from(3u64)];

        let xi_l = xi::xi_from_challenges::<Fr>(&xl);
        let xi_r = xi::xi_from_challenges::<Fr>(&xr);
        let mut c_hat = Fr::zero();
        for i in 0..m {
            for j in 0..n {
                c_hat += mat_a.data[j][i] * mat_b.data[j][i] * xi_l[i] * xi_r[j];
            }
        }

        let had = Hadamard::<Fr>::new(c_hat, (m, n), (xl, xr));

        let mut transcript = Transcript::new(Fr::zero());
        let prover_reduction = had.reduce_prover_claims(&mut transcript, &mat_a, &mat_b);

        let mut trans_v = transcript.clone();
        trans_v.reset_pointer();

        let (ok, verifier_reduction) = had.verify_as_subprotocol_claims(&mut trans_v);
        assert!(ok, "Hadamard reduction should verify");
        assert_eq!(prover_reduction, verifier_reduction);
        assert_eq!(
            prover_reduction.left.value,
            mat_a.proj_lr(
                &prover_reduction.left.point.0,
                &prover_reduction.left.point.1,
            )
        );
        assert_eq!(
            prover_reduction.right.value,
            mat_b.proj_lr(
                &prover_reduction.right.point.0,
                &prover_reduction.right.point.1,
            )
        );
    }

    #[test]
    fn test_hadamard_non_power_of_two() {
        // Test with non-power-of-two dimensions: m=3, n=2
        let m = 3;
        let n = 2;
        // column-major storage
        let mat_a = DenseMatFieldCM::<Fr>::from_data(vec![
            vec![Fr::from(1u64), Fr::from(3u64), Fr::from(5u64)], // col 0
            vec![Fr::from(2u64), Fr::from(4u64), Fr::from(6u64)], // col 1
        ]);
        let mat_b = DenseMatFieldCM::<Fr>::from_data(vec![
            vec![Fr::from(7u64), Fr::from(9u64), Fr::from(11u64)], // col 0
            vec![Fr::from(8u64), Fr::from(10u64), Fr::from(12u64)], // col 1
        ]);

        // log_m = ceil(log2(3)) = 2, so xl needs 2 challenges
        // log_n = ceil(log2(2)) = 1, so xr needs 1 challenge
        let xl = vec![Fr::from(2u64), Fr::from(3u64)];
        let xr = vec![Fr::from(5u64)];

        let xi_l = xi::xi_from_challenges::<Fr>(&xl);
        let xi_r = xi::xi_from_challenges::<Fr>(&xr);
        let mut c_hat = Fr::zero();
        for i in 0..m {
            for j in 0..n {
                c_hat += mat_a.data[j][i] * mat_b.data[j][i] * xi_l[i] * xi_r[j];
            }
        }

        let had = Hadamard::<Fr>::new(c_hat, (m, n), (xl, xr));

        let mut transcript = Transcript::new(Fr::zero());
        let prover_reduction = had.reduce_prover_claims(&mut transcript, &mat_a, &mat_b);

        let mut trans_v = transcript.clone();
        trans_v.reset_pointer();

        let (ok, verifier_reduction) = had.verify_as_subprotocol_claims(&mut trans_v);
        assert!(
            ok,
            "Hadamard reduction should verify for non-power-of-two dimensions"
        );
        assert_eq!(prover_reduction, verifier_reduction);
        assert_eq!(
            prover_reduction.left.value,
            mat_a.proj_lr(
                &prover_reduction.left.point.0,
                &prover_reduction.left.point.1,
            )
        );
        assert_eq!(
            prover_reduction.right.value,
            mat_b.proj_lr(
                &prover_reduction.right.point.0,
                &prover_reduction.right.point.1,
            )
        );
    }
}
