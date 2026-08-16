/// Check a matrix is a boolean matrix
/// Realized using Hadamard protocol
/// a \circ (1-a) = 0
///
use ark_ec::PrimeGroup;
use ark_ff::PrimeField;
use ark_std::marker::PhantomData;

use crate::data_structures::ZkSRS;

use crate::utils::sparse::{DenseMatCM, MatOps, ShortInt};
use crate::utils::zktr::{TranElem, TranSeq, ZkTranSeq};

use crate::utils::xi;

use super::batchpcs::BatchPoints;
use super::hadamard::ZkHadamard;

pub struct ZkBoolCheck<I, F, G, MA>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
    MA: MatOps<I, F>,
{
    shape: (usize, usize),
    _marker: PhantomData<(I, F, G, MA)>,
}

impl<I, F, G, MA> ZkBoolCheck<I, F, G, MA>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
    MA: MatOps<I, F>,
{
    pub fn new(m: usize, n: usize) -> Self {
        Self {
            shape: (m, n),
            _marker: PhantomData,
        }
    }

    pub fn reduce_prover(
        &self,
        zksrs: &ZkSRS<F, G>,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        mat_a: &MA,
    ) -> (G, F, (Vec<F>, Vec<F>)) {
        let bool_neg = mat_a.to_bool_neg();

        let m = self.shape.0;
        let n = self.shape.1;

        let log_m = m.ilog2() as usize;
        let log_n = n.ilog2() as usize;

        let mut xl = Vec::new();
        let mut xr = Vec::new();

        for _ in 0..log_m {
            xl.push(zk_trans_seq.gen_challenge());
        }

        for _ in 0..log_n {
            xr.push(zk_trans_seq.gen_challenge());
        }

        let hadamard = ZkHadamard::new(G::zero(), (xl.clone(), xr.clone()));

        let (
            hat_a_com,
            hat_b_com_prior,
            hat_a_tilde,
            hat_b_tilde_prior,
            (xl_a, xr_a),
            (xl_b, xr_b),
        ) = hadamard.reduce_prover(zksrs, zk_trans_seq, mat_a, &bool_neg, F::zero());

        let ones = vec![F::one(); log_m + log_n];
        let ones_x_b = xi::xi_ip_from_challenges(&[xl_b.clone(), xr_b.clone()].concat(), &ones);
        let hat_b_com = zksrs.com_base.mul(&ones_x_b) - hat_b_com_prior;

        let hat_b_tilde = -hat_b_tilde_prior;

        let mut batch = BatchPoints::new();

        batch.push_point(hat_a_com, (xl_a, xr_a));

        batch.push_point(hat_b_com, (xl_b, xr_b));

        let (v_a_com, v_a_tilde, point) =
            batch.reduce_prover(zksrs, zk_trans_seq, mat_a, vec![hat_a_tilde, hat_b_tilde]);

        (v_a_com, v_a_tilde, point)
    }

    pub fn verify_as_subprotocol(
        &self,
        zksrs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
    ) -> (bool, G, (Vec<F>, Vec<F>)) {
        let m = self.shape.0;
        let n = self.shape.1;

        let log_m = m.ilog2() as usize;
        let log_n = n.ilog2() as usize;

        let mut xl = Vec::new();
        let mut xr = Vec::new();

        let mut flag = true;
        let mut pointer_old = trans_seq.pointer;

        for _ in 0..log_m {
            if let TranElem::Coin(x) = trans_seq.data[pointer_old].clone() {
                pointer_old += 1;
                xl.push(x);
            } else {
                flag = false;
                println!("Type mismatch");
            }
        }

        for _ in 0..log_n {
            if let TranElem::Coin(x) = trans_seq.data[pointer_old].clone() {
                pointer_old += 1;
                xr.push(x);
            } else {
                flag = false;
                println!("Type mismatch");
            }
        }

        trans_seq.pointer = pointer_old;

        let hadamard = ZkHadamard::<I, I, F, G, MA, DenseMatCM<I, F>>::new(
            G::zero(),
            (xl.clone(), xr.clone()),
        );

        let (flag_hadamard, hat_a_com, hat_b_com_prior, (xl_a, xr_a), (xl_b, xr_b)) =
            hadamard.verify_as_subprotocol(zksrs, trans_seq);

        let ones = vec![F::one(); log_m + log_n];
        let ones_x_b = xi::xi_ip_from_challenges(&[xl_b.clone(), xr_b.clone()].concat(), &ones);
        let hat_b_com = zksrs.com_base.mul(&ones_x_b) - hat_b_com_prior;

        let mut batch = BatchPoints::<I, F, G>::new();

        batch.push_point(hat_a_com, (xl_a, xr_a));

        batch.push_point(hat_b_com, (xl_b, xr_b));

        let (flag_batch, v_a_com, point) = batch.verify_as_subprotocol(zksrs, trans_seq);

        (flag && flag_hadamard && flag_batch, v_a_com, point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::sparse::DenseBlockMat;

    #[test]
    fn test_boolcheck() {
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

        let m = num * ma;
        let n = na;

        let mut mat_a = DenseBlockMat::<bool, Fr>::new(num, (ma, na));

        mat_a.gen_rand(8);

        let zksrs = ZkSRS::<F, G>::new(G::rand(rng), G::rand(rng));

        let protocol = ZkBoolCheck::<bool, F, G, DenseBlockMat<bool, F>>::new(m, n);

        let zk_trans_seq = &mut ZkTranSeq::new(&zksrs);

        let (_, hat_a_tilde, _) = protocol.reduce_prover(&zksrs, zk_trans_seq, &mat_a);

        let trans_seq = &mut zk_trans_seq.publish_trans();

        let (flag, a_hat_com, (xl_a, xr_a)) = protocol.verify_as_subprotocol(&zksrs, trans_seq);

        let hat_a_check = mat_a.proj_lr(&xl_a, &xr_a);

        let a_hat_com_check = zksrs.commit(hat_a_check, hat_a_tilde);

        assert_eq!(a_hat_com, a_hat_com_check);
        assert_eq!(flag, true);
    }
}
