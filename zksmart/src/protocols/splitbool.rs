/// Split into boolean matrices
/// including sign
///
use ark_ec::PrimeGroup;
use ark_ff::PrimeField;
use ark_std::marker::PhantomData;

use crate::data_structures::ZkSRS;

use crate::utils::sparse::{DenseMatCM, MatOps, ShortInt};
use crate::utils::xi;
use crate::utils::zktr::{TranSeq, ZkTranSeq};

use super::batchpcs::BatchPoints;
use super::boolcheck::ZkBoolCheck;
use super::hadamard::ZkHadamard;
use super::linearcomb::ZkSerialComb;

pub struct ZkSplit<I, F, G, MA, MB>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
    MA: MatOps<I, F>,
    MB: MatOps<bool, F>,
{
    pub v_c_com: G,
    pub point: (Vec<F>, Vec<F>),
    pub k: usize,
    _marker: PhantomData<(I, MA, MB)>,
}

impl<I, F, G, MA, MB> ZkSplit<I, F, G, MA, MB>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
    MA: MatOps<I, F>,
    MB: MatOps<bool, F>,
{
    pub fn new(v_c_com_value: G, point_value: (Vec<F>, Vec<F>), k_value: usize) -> Self {
        Self {
            v_c_com: v_c_com_value,
            point: point_value,
            k: k_value,
            _marker: PhantomData,
        }
    }

    pub fn reduce_prover(
        &self,
        zksrs: &ZkSRS<F, G>,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        mat_abs: &DenseMatCM<I, F>,
        signmat: &MB,
        booleans: &Vec<MB>,
        tilde_c: F,
    ) -> (
        G,
        Vec<G>,
        F,
        Vec<F>,
        (Vec<F>, Vec<F>),
        Vec<(Vec<F>, Vec<F>)>,
    ) {
        println!("-- Hadamard in split");

        let signmat_i32 = DenseMatCM::<i32, F>::from_data(
            signmat
                .to_dense()
                .data
                .iter()
                .map(|column| {
                    column
                        .iter()
                        .map(|is_positive| if *is_positive { 1 } else { -1 })
                        .collect()
                })
                .collect(),
        );

        let hadamard = ZkHadamard::<i32, I, F, G, DenseMatCM<i32, F>, DenseMatCM<I, F>>::new(
            self.v_c_com,
            self.point.clone(),
        );

        let (v_sign_com, v_abs_com, v_sign_tilde, v_abs_tilde, point_sign, point_abs) =
            hadamard.reduce_prover(zksrs, zk_trans_seq, &signmat_i32, &mat_abs, tilde_c);

        // signmat_i32.clear();

        let point_sign_concat = [point_sign.0.clone(), point_sign.1.clone()].concat();
        let ones = vec![F::one(); point_sign_concat.len()];
        let ip_ones = xi::xi_ip_from_challenges(&point_sign_concat, &ones);

        let two_inv = F::from(2).inverse().unwrap();

        let v_sign_com1 = (v_sign_com + zksrs.com_base.mul(&ip_ones)) * two_inv;

        let v_sign_tilde1 = v_sign_tilde * two_inv;

        let boolcheck_sign =
            ZkBoolCheck::<bool, F, G, MB>::new(signmat.get_shape().0, signmat.get_shape().1);

        println!("-- Sign boolcheck in split");

        let (v_sign_com_2, v_sign_tilde_2, sign_point_2) =
            boolcheck_sign.reduce_prover(zksrs, zk_trans_seq, &signmat);

        println!("-- Batch sign in split");

        let mut batch = BatchPoints::<bool, F, G>::new();
        batch.push_point(v_sign_com1, point_sign);
        batch.push_point(v_sign_com_2, sign_point_2);

        let (v_sign_com_prime, v_sign_tilde_prime, sign_point_prime) = batch.reduce_prover(
            zksrs,
            zk_trans_seq,
            &signmat.to_dense(),
            vec![v_sign_tilde1, v_sign_tilde_2],
        );

        let mut coeffs = Vec::new();
        let mut factor = F::one();

        for _ in 0..self.k {
            coeffs.push(factor);
            factor = factor.double();
        }

        println!("-- Linear comb in split");

        let linear_comb =
            ZkSerialComb::<bool, F, G, MB>::new(v_abs_com.clone(), point_abs.clone(), coeffs);

        let (v_b_com_vec, v_b_tilde_vec, point_vec) =
            linear_comb.reduce_prover(zksrs, zk_trans_seq, booleans, v_abs_tilde);

        let mut v_b_com_vec_prime = Vec::new();
        let mut v_b_tilde_vec_prime = Vec::new();
        let mut point_vec_prime = Vec::new();

        for kappa in 0..self.k {
            let v_b_com_1 = v_b_com_vec[kappa].clone();
            let v_b_tilde_1 = v_b_tilde_vec[kappa].clone();
            let b_point_1 = point_vec[kappa].clone();

            let log_m = b_point_1.0.len();
            let log_n = b_point_1.1.len();
            let m = 1 << log_m;
            let n = 1 << log_n;

            let boolcheck = ZkBoolCheck::<bool, F, G, MB>::new(m, n);

            let (v_b_com_2, v_b_tilde_2, b_point_2) =
                boolcheck.reduce_prover(zksrs, zk_trans_seq, &booleans[kappa]);

            let mut batch = BatchPoints::<bool, F, G>::new();
            batch.push_point(v_b_com_1, b_point_1);
            batch.push_point(v_b_com_2, b_point_2);

            let (v_b_com_prime, v_b_tilde_prime, b_point_prime) = batch.reduce_prover(
                zksrs,
                zk_trans_seq,
                &booleans[kappa],
                vec![v_b_tilde_1, v_b_tilde_2],
            );

            v_b_com_vec_prime.push(v_b_com_prime);
            v_b_tilde_vec_prime.push(v_b_tilde_prime);
            point_vec_prime.push(b_point_prime);
        }

        (
            v_sign_com_prime,
            v_b_com_vec_prime,
            v_sign_tilde_prime,
            v_b_tilde_vec_prime,
            sign_point_prime,
            point_vec_prime,
        )
    }

    pub fn verify_as_subprotocol(
        &self,
        zksrs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
    ) -> (bool, G, Vec<G>, (Vec<F>, Vec<F>), Vec<(Vec<F>, Vec<F>)>) {
        let hadamard = ZkHadamard::<i32, I, F, G, DenseMatCM<i32, F>, DenseMatCM<I, F>>::new(
            self.v_c_com,
            self.point.clone(),
        );

        let (flag_hadamard, v_sign_com, v_abs_com, point_sign, point_abs) =
            hadamard.verify_as_subprotocol(zksrs, trans_seq);

        let point_sign_concat = [point_sign.0.clone(), point_sign.1.clone()].concat();
        let ones = vec![F::one(); point_sign_concat.len()];
        let ip_ones = xi::xi_ip_from_challenges(&point_sign_concat, &ones);

        let two_inv = F::from(2).inverse().unwrap();

        let v_sign_com1 = (v_sign_com + zksrs.com_base.mul(&ip_ones)) * two_inv;

        let log_m = self.point.0.len();
        let log_n = self.point.1.len();
        let m = 1 << log_m;
        let n = 1 << log_n;

        let boolcheck_sign = ZkBoolCheck::<bool, F, G, MB>::new(m, n);

        let (flag_sign, v_sign_com_2, sign_point_2) =
            boolcheck_sign.verify_as_subprotocol(zksrs, trans_seq);

        let mut batch = BatchPoints::<bool, F, G>::new();
        batch.push_point(v_sign_com1, point_sign);
        batch.push_point(v_sign_com_2, sign_point_2);

        let (flag_sign2, v_sign_com_prime, sign_point_prime) =
            batch.verify_as_subprotocol(zksrs, trans_seq);

        let mut flag = flag_hadamard && flag_sign && flag_sign2;

        let mut coeffs = Vec::new();
        let mut factor = F::one();

        for _ in 0..self.k {
            coeffs.push(factor);
            factor = factor.double();
        }

        let linear_comb =
            ZkSerialComb::<bool, F, G, MB>::new(v_abs_com.clone(), point_abs.clone(), coeffs);

        let (flag_linear, v_b_com_vec, point_vec) =
            linear_comb.verify_as_subprotocol(zksrs, trans_seq);

        flag = flag && flag_linear;

        let mut v_b_com_vec_prime = Vec::new();
        let mut point_vec_prime = Vec::new();

        for kappa in 0..self.k {
            let v_b_com_1 = v_b_com_vec[kappa].clone();
            let b_point_1 = point_vec[kappa].clone();

            let log_m = b_point_1.0.len();
            let log_n = b_point_1.1.len();
            let m = 1 << log_m;
            let n = 1 << log_n;

            let boolcheck = ZkBoolCheck::<bool, F, G, MB>::new(m, n);

            let (flag_cur, v_b_com_2, b_point_2) =
                boolcheck.verify_as_subprotocol(zksrs, trans_seq);

            flag = flag && flag_cur;

            let mut batch = BatchPoints::<bool, F, G>::new();
            batch.push_point(v_b_com_1, b_point_1);
            batch.push_point(v_b_com_2, b_point_2);

            let (flag_cur, v_b_com_prime, b_point_prime) =
                batch.verify_as_subprotocol(zksrs, trans_seq);

            flag = flag && flag_cur;

            v_b_com_vec_prime.push(v_b_com_prime);
            point_vec_prime.push(b_point_prime);
        }

        (
            flag,
            v_sign_com_prime,
            v_b_com_vec_prime,
            sign_point_prime,
            point_vec_prime,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::sparse::{DenseBlockMat, DenseMatCM};
    // use rayon::iter::{IntoParallelIterator, ParallelIterator};

    #[test]
    fn test_splitbool() {
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

        let k = 8;
        mat_a.gen_rand(k);

        let (sign_mat, abs_mat, boolean_mats) = mat_a.bit_decomposition(k);

        let log_m = log_ma + log_num;
        let log_n = log_na;

        let xl = vec![Fr::rand(rng); log_m];
        let xr = vec![Fr::rand(rng); log_n];

        let v_a = mat_a.proj_lr(&xl, &xr);

        let v_a_tilde = Fr::rand(rng);

        let zksrs = ZkSRS::<F, G>::new(G::rand(rng), G::rand(rng));

        let v_a_com = zksrs.commit(v_a, v_a_tilde);

        let protocol = ZkSplit::<i32, F, G, DenseBlockMat<i32, F>, DenseMatCM<bool, F>>::new(
            v_a_com,
            (xl, xr),
            k,
        );

        let zk_trans_seq = &mut ZkTranSeq::new(&zksrs);

        let (_, _, sign_tilde, b_tilde_vec, _, _) = protocol.reduce_prover(
            &zksrs,
            zk_trans_seq,
            &abs_mat,
            &sign_mat,
            &boolean_mats,
            v_a_tilde,
        );

        let trans_seq = &mut zk_trans_seq.publish_trans();

        let (flag, sign_com, v_b_com_vec, sign_point, point_vec) =
            protocol.verify_as_subprotocol(&zksrs, trans_seq);

        let hat_sign_check = sign_mat.proj_lr(&sign_point.0, &sign_point.1);

        let sign_hat_com_check = zksrs.commit(hat_sign_check, sign_tilde);

        assert_eq!(flag, true);
        assert_eq!(sign_com, sign_hat_com_check);

        let v_b_1_check = boolean_mats[1].proj_lr(&point_vec[1].0, &point_vec[1].1);
        let v_b_1_com_check = zksrs.commit(v_b_1_check, b_tilde_vec[1]);

        assert_eq!(v_b_com_vec[1], v_b_1_com_check);
    }
}
