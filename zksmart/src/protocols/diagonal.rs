/// Implement the matmul protocol
///
use ark_ec::PrimeGroup;
use ark_ff::PrimeField;
use ark_std::marker::PhantomData;

use crate::data_structures::ZkSRS;

use crate::utils::linear;
use crate::utils::sparse::{ProjectionMatrix, ShortInt};
use crate::utils::zktr::{TranSeq, ZkTranSeq};

pub struct ZkDiagonal<I, F, G, MA>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
    MA: ProjectionMatrix<I, F>,
{
    pub v_c_com: G,
    pub point: (Vec<F>, Vec<F>),
    pub num: usize,
    _marker: PhantomData<(I, MA)>,
}

impl<I, F, G, MA> ZkDiagonal<I, F, G, MA>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
    MA: ProjectionMatrix<I, F>,
{
    pub fn new(v_c_com_value: G, point_value: (Vec<F>, Vec<F>), num_value: usize) -> Self {
        Self {
            v_c_com: v_c_com_value,
            point: point_value,
            num: num_value,
            _marker: PhantomData,
        }
    }

    pub fn reduce_prover(
        &self,
        _zksrs: &ZkSRS<F, G>,
        _zk_trans_seq: &mut ZkTranSeq<F, G>,
        mat_a: &MA,
        tilde_c: F,
    ) -> (G, F, (Vec<F>, Vec<F>)) {
        let log_m = self.point.0.len();
        let log_n = self.point.1.len();

        let m = 1 << log_m;
        let n = 1 << log_n;

        let num = self.num;
        let log_num = num.ilog2() as usize;

        let ma = m;
        let na = n / num;

        if (ma, na) != mat_a.projection_shape() {
            println!("ma, na: {:?}", (ma, na));
            println!("shape: {:?}", mat_a.projection_shape());
            panic!("Matrix dimension mismatch");
        }

        let xl = self.point.0.clone();
        let xr = self.point.1.clone();

        let xl_1 = xl[..log_num].to_vec();
        let xl_2 = xl[log_num..].to_vec();

        let xr_1 = xr[..log_num].to_vec();
        let xr_2 = xr[log_num..].to_vec();

        let xl1_prime = linear::vec_element_wise_mul(&xl_1, &xr_1);

        let xl_prime = [xl1_prime, xl_2].concat();

        (self.v_c_com, tilde_c, (xl_prime, xr_2))
    }

    pub fn verify_as_subprotocol(
        &self,
        _zksrs: &ZkSRS<F, G>,
        _trans_seq: &mut TranSeq<F, G>,
    ) -> (bool, G, (Vec<F>, Vec<F>)) {
        let num = self.num;
        let log_num = num.ilog2() as usize;

        let xl = self.point.0.clone();
        let xr = self.point.1.clone();

        let xl_1 = xl[..log_num].to_vec();
        let xl_2 = xl[log_num..].to_vec();

        let xr_1 = xr[..log_num].to_vec();
        let xr_2 = xr[log_num..].to_vec();

        let xl1_prime = linear::vec_element_wise_mul(&xl_1, &xr_1);

        let xl_prime = [xl1_prime, xl_2].concat();

        (true, self.v_c_com, (xl_prime, xr_2))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::sparse::{DenseBlockMat, DiagBlockMat, MatOps};

    #[test]
    fn test_diagonal() {
        use ark_bls12_381::Fr;
        use ark_ec::pairing::Pairing;
        use ark_std::UniformRand;

        type E = ark_bls12_381::Bls12_381;
        type F = <E as Pairing>::ScalarField;
        type G = <E as Pairing>::G1;

        let rng = &mut ark_std::rand::thread_rng();

        let log_ma = 3;
        let log_na = 2;
        let log_num = 1;

        let ma = 1 << log_ma;
        let na = 1 << log_na;
        let num = 1 << log_num;

        let mut mat_a = DenseBlockMat::<i32, Fr>::new(num, (ma, na));

        mat_a.gen_rand(8);

        let mat_c_diag = DiagBlockMat::<i32, Fr>::from_dense_block(&mat_a);
        let mat_c = mat_c_diag.to_dense();

        let log_m = log_ma + log_num;
        let log_n = log_na + log_num;

        let xl = vec![Fr::rand(rng); log_m];
        let xr = vec![Fr::rand(rng); log_n];

        let v_c = mat_c.proj_lr(&xl, &xr);

        let v_c_tilde = Fr::rand(rng);

        let zksrs = ZkSRS::<F, G>::new(G::rand(rng), G::rand(rng));

        let v_c_com = zksrs.commit(v_c, v_c_tilde);

        let protocol = ZkDiagonal::<i32, F, G, DenseBlockMat<i32, F>>::new(v_c_com, (xl, xr), num);

        let zk_trans_seq = &mut ZkTranSeq::new(&zksrs);

        let (_, hat_a_tilde, _) = protocol.reduce_prover(&zksrs, zk_trans_seq, &mat_a, v_c_tilde);

        let trans_seq = &mut zk_trans_seq.publish_trans();

        let (flag, a_hat_com, (xl_a, xr_a)) = protocol.verify_as_subprotocol(&zksrs, trans_seq);

        let hat_a_check = mat_a.proj_lr(&xl_a, &xr_a);

        let a_hat_com_check = zksrs.commit(hat_a_check, hat_a_tilde);

        assert_eq!(a_hat_com, a_hat_com_check);
        assert_eq!(flag, true);
    }
}
