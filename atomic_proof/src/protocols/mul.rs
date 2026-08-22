use ark_crypto_primitives::sponge::Absorb;
/// Implement the matmul protocol
///
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::UniformRand;

use fs::helper_trans::Transcript;
use mat::utils::matdef::MatOps;
use mat::utils::xi;
use mat::DenseMatFieldCM;

use crate::atomic_protocol::EvalWithPointMat;

use super::litebullet::LiteBullet;

/// Matrix MLE claims produced by a matrix-multiplication reduction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatMulReduction<F> {
    pub left: EvalWithPointMat<F>,
    pub right: EvalWithPointMat<F>,
}
pub struct MatMul<F>
where
    F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
{
    pub c_hat: F,
    pub point: (Vec<F>, Vec<F>),
    pub shape_c: (usize, usize),
    pub shape_a: (usize, usize),
    pub shape_b: (usize, usize),
}

impl<F> MatMul<F>
where
    F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
{
    pub fn new(
        c_hat_value: F,
        point_value: (Vec<F>, Vec<F>),
        shape_c_value: (usize, usize),
        shape_a_value: (usize, usize),
        shape_b_value: (usize, usize),
    ) -> Self {
        Self {
            c_hat: c_hat_value,
            point: point_value,
            shape_c: shape_c_value,
            shape_a: shape_a_value,
            shape_b: shape_b_value,
        }
    }

    pub fn reduce_prover(
        &self,
        trans: &mut Transcript<F>,
        mat_a: &DenseMatFieldCM<F>,
        mat_b: &DenseMatFieldCM<F>,
    ) -> (F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) {
        let (log_m, log_n) = (self.point.0.len(), self.point.1.len());
        let m_padded = 1 << log_m;
        let n_padded = 1 << log_n;

        let (ma, na) = mat_a.shape;
        let (mb, nb) = mat_b.shape;

        if ma > m_padded || nb > n_padded || na != mb {
            panic!("Matrix dimension mismatch");
        }

        // Expand challenges to xi vectors
        let xi_l = xi::xi_from_challenges::<F>(&self.point.0);
        let xi_r = xi::xi_from_challenges::<F>(&self.point.1);

        // Compute la = xi_l * A and br = B * xi_r
        // Note: proj_left on CM matrix computes v * M
        let la = mat_a.proj_left(&xi_l);
        let br = mat_b.proj_right(&xi_r);

        // Current len might not be power of 2, LiteBullet will handle padding
        let len = na;
        let lb = LiteBullet::new(self.c_hat, len);

        let (a_hat, b_hat, xl_inv, xl_challenges) = lb.reduce_prover(trans, &la, &br);

        (
            a_hat,
            b_hat,
            (self.point.0.clone(), xl_inv),
            (xl_challenges, self.point.1.clone()),
        )
    }

    /// Reduce the multiplication claim to structured claims for both input matrices.
    pub fn reduce_prover_claims(
        &self,
        trans: &mut Transcript<F>,
        mat_a: &DenseMatFieldCM<F>,
        mat_b: &DenseMatFieldCM<F>,
    ) -> MatMulReduction<F> {
        let (left_value, right_value, left_point, right_point) =
            self.reduce_prover(trans, mat_a, mat_b);
        MatMulReduction {
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
        let len = self.shape_a.1;
        let lb = LiteBullet::new(self.c_hat, len);

        let (flag, a_hat, b_hat, xl_inv, xl_challenges) = lb.verify_as_subprotocol(trans_seq);

        (
            flag,
            a_hat,
            b_hat,
            (self.point.0.clone(), xl_inv),
            (xl_challenges, self.point.1.clone()),
        )
    }

    /// Verify the reduction and return commitment-independent input claims.
    pub fn verify_as_subprotocol_claims(
        &self,
        trans_seq: &mut Transcript<F>,
    ) -> (bool, MatMulReduction<F>) {
        let (ok, left_value, right_value, left_point, right_point) =
            self.verify_as_subprotocol(trans_seq);
        (
            ok,
            MatMulReduction {
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

        // a_hat = proj_lr(mat_a, xl_init, xl_inv)
        let a_hat_check = mat_a.proj_lr(&point_a.0, &point_a.1);
        // b_hat = proj_lr(mat_b, xl_challenges, xr_init)
        let b_hat_check = mat_b.proj_lr(&point_b.0, &point_b.1);

        flag && a_hat == a_hat_check && b_hat == b_hat_check && trans_seq.fs.verify_fs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::Zero;

    // Helper function to compute matrix multiplication C = A * B
    // A is m x len (column-major), B is len x n (column-major)
    // Returns C as m x n (column-major)
    fn mat_mul_field(a_data: &Vec<Vec<Fr>>, b_data: &Vec<Vec<Fr>>) -> Vec<Vec<Fr>> {
        let len = a_data.len();
        let m = a_data[0].len();
        let n = b_data.len();

        let mut c_data = vec![vec![Fr::zero(); m]; n];

        // C[i,j] = sum_k A[i,k] * B[k,j]
        for j in 0..n {
            for i in 0..m {
                let mut sum = Fr::zero();
                for k in 0..len {
                    sum += a_data[k][i] * b_data[j][k];
                }
                c_data[j][i] = sum;
            }
        }

        c_data
    }

    #[test]
    fn test_matmul_reduce_and_verify() {
        // Test C = A * B where A is 2x4 and B is 4x2
        let m = 2;
        let n = 2;
        let len = 4;

        // Matrix A: 2x4 in column-major format
        let mat_a = DenseMatFieldCM::<Fr>::from_data(vec![
            vec![Fr::from(1u64), Fr::from(5u64)], // col 0
            vec![Fr::from(2u64), Fr::from(6u64)], // col 1
            vec![Fr::from(3u64), Fr::from(7u64)], // col 2
            vec![Fr::from(4u64), Fr::from(8u64)], // col 3
        ]);

        // Matrix B: 4x2 in column-major format
        let mat_b = DenseMatFieldCM::<Fr>::from_data(vec![
            vec![
                Fr::from(1u64),
                Fr::from(2u64),
                Fr::from(3u64),
                Fr::from(4u64),
            ], // col 0
            vec![
                Fr::from(5u64),
                Fr::from(6u64),
                Fr::from(7u64),
                Fr::from(8u64),
            ], // col 1
        ]);

        // Compute C = A * B manually for verification
        let c_data = mat_mul_field(&mat_a.data, &mat_b.data);
        let mat_c = DenseMatFieldCM::<Fr>::from_data(c_data);

        // Random challenges for projection
        let xl = vec![Fr::from(2u64)]; // log_m = 1
        let xr = vec![Fr::from(3u64)]; // log_n = 1

        // Compute c_hat = proj_lr(C, xl, xr)
        let c_hat = mat_c.proj_lr(&xl, &xr);

        let protocol =
            MatMul::<Fr>::new(c_hat, (xl.clone(), xr.clone()), (m, n), (m, len), (len, n));

        let mut transcript = Transcript::new(Fr::zero());
        let prover_reduction = protocol.reduce_prover_claims(&mut transcript, &mat_a, &mat_b);

        let mut trans_v = transcript.clone();
        trans_v.reset_pointer();

        let (ok, verifier_reduction) = protocol.verify_as_subprotocol_claims(&mut trans_v);
        assert!(ok, "MatMul reduction should verify");
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
    fn test_matmul_non_power_of_two() {
        // Test C = A * B where A is 3x5 and B is 5x2
        let m = 3;
        let n = 2;
        let len = 5;

        // Matrix A: 3x5 in column-major format
        let mat_a = DenseMatFieldCM::<Fr>::from_data(vec![
            vec![Fr::from(1u64), Fr::from(4u64), Fr::from(7u64)], // col 0
            vec![Fr::from(2u64), Fr::from(5u64), Fr::from(8u64)], // col 1
            vec![Fr::from(3u64), Fr::from(6u64), Fr::from(9u64)], // col 2
            vec![Fr::from(1u64), Fr::from(2u64), Fr::from(3u64)], // col 3
            vec![Fr::from(4u64), Fr::from(5u64), Fr::from(6u64)], // col 4
        ]);

        // Matrix B: 5x2 in column-major format
        let mat_b = DenseMatFieldCM::<Fr>::from_data(vec![
            vec![
                Fr::from(1u64),
                Fr::from(2u64),
                Fr::from(3u64),
                Fr::from(4u64),
                Fr::from(5u64),
            ], // col 0
            vec![
                Fr::from(6u64),
                Fr::from(7u64),
                Fr::from(8u64),
                Fr::from(9u64),
                Fr::from(10u64),
            ], // col 1
        ]);

        // Compute C = A * B
        let c_data = mat_mul_field(&mat_a.data, &mat_b.data);
        let mat_c = DenseMatFieldCM::<Fr>::from_data(c_data);

        // Challenges: log_m = ceil(log2(3)) = 2, log_n = ceil(log2(2)) = 1
        let xl = vec![Fr::from(2u64), Fr::from(3u64)];
        let xr = vec![Fr::from(5u64)];

        let c_hat = mat_c.proj_lr(&xl, &xr);

        let protocol =
            MatMul::<Fr>::new(c_hat, (xl.clone(), xr.clone()), (m, n), (m, len), (len, n));

        let mut transcript = Transcript::new(Fr::zero());
        let prover_reduction = protocol.reduce_prover_claims(&mut transcript, &mat_a, &mat_b);

        let mut trans_v = transcript.clone();
        trans_v.reset_pointer();

        let (ok, verifier_reduction) = protocol.verify_as_subprotocol_claims(&mut trans_v);
        assert!(
            ok,
            "MatMul reduction should verify for non-power-of-two dimensions"
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
