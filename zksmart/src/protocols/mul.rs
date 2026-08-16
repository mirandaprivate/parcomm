/// Implement the matmul protocol
///
use ark_ec::PrimeGroup;
use ark_ff::PrimeField;
use ark_std::marker::PhantomData;

use crate::data_structures::ZkSRS;

use crate::utils::sparse::{ProjectionMatrix, ShortInt};
use crate::utils::zktr::{TranSeq, ZkTranSeq};

use super::litebullet::ZkLiteBullet;

pub struct ZkMatMul<I, F, G, MA, MB>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
    MA: ProjectionMatrix<I, F>,
    MB: ProjectionMatrix<I, F>,
{
    pub v_c_com: G,
    pub point: (Vec<F>, Vec<F>),
    pub len: usize,
    _marker: PhantomData<(I, MA, MB)>,
}

impl<I, F, G, MA, MB> ZkMatMul<I, F, G, MA, MB>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
    MA: ProjectionMatrix<I, F>,
    MB: ProjectionMatrix<I, F>,
{
    pub fn new(v_c_com_value: G, point_value: (Vec<F>, Vec<F>), len_value: usize) -> Self {
        Self {
            v_c_com: v_c_com_value,
            point: point_value,
            len: len_value,
            _marker: PhantomData,
        }
    }

    pub fn reduce_prover(
        &self,
        zksrs: &ZkSRS<F, G>,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        mat_a: &MA,
        mat_b: &MB,
        tilde_c: F,
    ) -> (G, G, F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) {
        let log_m = self.point.0.len();
        let log_n = self.point.1.len();

        let m = 1 << log_m;
        let n = 1 << log_n;

        let len = self.len;

        let ma = mat_a.projection_shape().0;
        let na = mat_a.projection_shape().1;
        let mb = mat_b.projection_shape().0;
        let nb = mat_b.projection_shape().1;

        if ma != m || nb != n || na != len || mb != len {
            panic!("Matrix dimension mismatch");
        }

        let xl = self.point.0.clone();
        let xr = self.point.1.clone();

        let mut la = mat_a.project_left(&xl);
        let mut br = mat_b.project_right(&xr);

        let litebullet = ZkLiteBullet::new(self.v_c_com, len);

        let (a_hat_com, b_hat_com, a_hat_tilde, b_hat_tilde, xr_prime, xl_prime) =
            litebullet.reduce_prover(zksrs, zk_trans_seq, &mut la, &mut br, tilde_c);

        // println!("xl: {:?}", xl);
        // println!("xr: {:?}", xr);
        // println!("xl_prime: {:?}", xl_prime);
        // println!("xr_prime: {:?}", xr_prime);

        (
            a_hat_com,
            b_hat_com,
            a_hat_tilde,
            b_hat_tilde,
            (xl, xr_prime),
            (xl_prime, xr),
        )
    }

    pub fn verify_as_subprotocol(
        &self,
        zksrs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
    ) -> (bool, G, G, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) {
        let xl = self.point.0.clone();
        let xr = self.point.1.clone();

        let litebullet = ZkLiteBullet::new(self.v_c_com, self.len);

        let (flag, a_hat_com, b_hat_com, xr_prime, xl_prime) =
            litebullet.verify_as_subprotocol(zksrs, trans_seq);

        (flag, a_hat_com, b_hat_com, (xl, xr_prime), (xl_prime, xr))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::heterogeneous::{
        DiagonalHeteroView, HeteroBlockMat, MatrixBlock, MatrixDims, PaddedDomain,
        StackedHeteroView,
    };
    use crate::utils::linear;
    use crate::utils::sparse::{DenseMatCM, MatOps};

    #[test]
    fn test_mul() {
        use ark_bls12_381::Fr;
        use ark_ec::pairing::Pairing;
        use ark_std::UniformRand;

        type E = ark_bls12_381::Bls12_381;
        type F = <E as Pairing>::ScalarField;
        type G = <E as Pairing>::G1;

        let rng = &mut ark_std::rand::thread_rng();

        let log_m = 3;
        let log_n = 4;
        let log_len = 5;

        let m = 1 << log_m;
        let n = 1 << log_n;
        let len = 1 << log_len;

        let mut mat_a = DenseMatCM::<i32, Fr>::new(m, len);
        let mut mat_b = DenseMatCM::<i32, Fr>::new(len, n);
        let mut mat_c = DenseMatCM::<i32, Fr>::new(m, n);

        mat_a.gen_rand(8);
        mat_b.gen_rand(8);
        let c_data = linear::mat_mul_shortint::<i32>(&mat_a.data, &mat_b.data);
        mat_c.set_data(c_data);

        let xl = vec![Fr::rand(rng); log_m];
        let xr = vec![Fr::rand(rng); log_n];

        let v_c = mat_c.proj_lr(&xl, &xr);

        let v_c_tilde = Fr::rand(rng);

        let zksrs = ZkSRS::<F, G>::new(G::rand(rng), G::rand(rng));

        let v_c_com = zksrs.commit(v_c, v_c_tilde);

        let protocol = ZkMatMul::<i32, F, G, DenseMatCM<i32, F>, DenseMatCM<i32, F>>::new(
            v_c_com,
            (xl, xr),
            len,
        );

        let zk_trans_seq = &mut ZkTranSeq::new(&zksrs);

        let (_, _, hat_a_tilde, hat_b_tilde, _, _) =
            protocol.reduce_prover(&zksrs, zk_trans_seq, &mat_a, &mat_b, v_c_tilde);

        let trans_seq = &mut zk_trans_seq.publish_trans();

        let (flag, a_hat_com, b_hat_com, (xl_a, xr_a), (xl_b, xr_b)) =
            protocol.verify_as_subprotocol(&zksrs, trans_seq);

        let hat_a_check = mat_a.proj_lr(&xl_a, &xr_a);

        let hat_b_check = mat_b.proj_lr(&xl_b, &xr_b);

        let a_hat_com_check = zksrs.commit(hat_a_check, hat_a_tilde);
        let b_hat_com_check = zksrs.commit(hat_b_check, hat_b_tilde);

        assert_eq!(a_hat_com, a_hat_com_check);
        assert_eq!(b_hat_com, b_hat_com_check);
        assert_eq!(flag, true);
    }

    #[test]
    fn heterogeneous_batch_matmul_reduces_without_padding_materialization() {
        use ark_bls12_381::Fr;
        use ark_ec::pairing::Pairing;
        use ark_std::UniformRand;

        type E = ark_bls12_381::Bls12_381;
        type F = <E as Pairing>::ScalarField;
        type G = <E as Pairing>::G1;

        let left = HeteroBlockMat::new(
            PaddedDomain::new(2, 2, 4).unwrap(),
            vec![
                Some(
                    MatrixBlock::from_columns(
                        MatrixDims::new(2, 3).unwrap(),
                        vec![vec![1_i32, 2], vec![3, 4], vec![5, 6]],
                    )
                    .unwrap(),
                ),
                Some(
                    MatrixBlock::from_columns(MatrixDims::new(1, 1).unwrap(), vec![vec![2_i32]])
                        .unwrap(),
                ),
            ],
        )
        .unwrap();
        let right = HeteroBlockMat::new(
            PaddedDomain::new(2, 4, 2).unwrap(),
            vec![
                Some(
                    MatrixBlock::from_columns(
                        MatrixDims::new(3, 2).unwrap(),
                        vec![vec![1_i32, 0, 1], vec![0, 1, 1]],
                    )
                    .unwrap(),
                ),
                Some(
                    MatrixBlock::from_columns(
                        MatrixDims::new(1, 2).unwrap(),
                        vec![vec![3_i32], vec![4]],
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();
        let output = HeteroBlockMat::new(
            PaddedDomain::new(2, 2, 2).unwrap(),
            vec![
                Some(
                    MatrixBlock::from_columns(
                        MatrixDims::new(2, 2).unwrap(),
                        vec![vec![6_i32, 8], vec![8, 10]],
                    )
                    .unwrap(),
                ),
                Some(
                    MatrixBlock::from_columns(
                        MatrixDims::new(1, 2).unwrap(),
                        vec![vec![6_i32], vec![8]],
                    )
                    .unwrap(),
                ),
            ],
        )
        .unwrap();

        let left_view = DiagonalHeteroView::new(&left);
        let right_view = StackedHeteroView::new(&right);
        let xl = vec![Fr::from(2_u64), Fr::from(3_u64)];
        let xr = vec![Fr::from(5_u64)];
        let output_value = output.proj_stacked_lr(&xl, &xr).unwrap();
        let rng = &mut ark_std::test_rng();
        let zksrs = ZkSRS::<F, G>::new(G::rand(rng), G::rand(rng));
        let output_hiding = Fr::rand(rng);
        let output_commitment = zksrs.commit(output_value, output_hiding);
        let protocol =
            ZkMatMul::<i32, F, G, DiagonalHeteroView<'_, i32>, StackedHeteroView<'_, i32>>::new(
                output_commitment,
                (xl, xr),
                8,
            );
        let transcript = &mut ZkTranSeq::new(&zksrs);

        let (_, _, left_hiding, right_hiding, _, _) =
            protocol.reduce_prover(&zksrs, transcript, &left_view, &right_view, output_hiding);
        let verifier_transcript = &mut transcript.publish_trans();
        let (ok, left_commitment, right_commitment, left_point, right_point) =
            protocol.verify_as_subprotocol(&zksrs, verifier_transcript);

        assert!(ok);
        assert_eq!(
            left_commitment,
            zksrs.commit(
                left.proj_diagonal_lr(&left_point.0, &left_point.1).unwrap(),
                left_hiding
            ),
        );
        assert_eq!(
            right_commitment,
            zksrs.commit(
                right
                    .proj_stacked_lr(&right_point.0, &right_point.1)
                    .unwrap(),
                right_hiding
            ),
        );
    }
}
