/// Public linear combination of matrices
///
use ark_ec::PrimeGroup;
use ark_ff::PrimeField;
use ark_std::marker::PhantomData;

use crate::data_structures::ZkSRS;

use crate::utils::sparse::{MatOps, ShortInt};
use crate::utils::zktr::{TranElem, TranSeq, ZkTranSeq};

use super::scalars::ZkSchnorr;

pub struct ZkSerialComb<I, F, G, M>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
    M: MatOps<I, F>,
{
    pub v_c_com: G,
    pub point: (Vec<F>, Vec<F>),
    pub coeffs: Vec<F>,
    _marker: PhantomData<(I, G, M)>,
}

impl<I, F, G, M> ZkSerialComb<I, F, G, M>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
    M: MatOps<I, F>,
{
    pub fn new(v_c_com_value: G, point_value: (Vec<F>, Vec<F>), coeffs_value: Vec<F>) -> Self {
        Self {
            v_c_com: v_c_com_value,
            point: point_value,
            coeffs: coeffs_value,
            _marker: PhantomData,
        }
    }

    pub fn reduce_prover(
        &self,
        zksrs: &ZkSRS<F, G>,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        mats: &Vec<M>,
        tilde_c: F,
    ) -> (Vec<G>, Vec<F>, Vec<(Vec<F>, Vec<F>)>) {
        let log_m = self.point.0.len();
        let log_n = self.point.1.len();

        let m = 1 << log_m;
        let n = 1 << log_n;

        let len = self.coeffs.len();

        for kappa in 0..len {
            if mats[kappa].get_shape() != (m, n) {
                panic!("Matrix shape mismatch");
            }
        }

        let xl = self.point.0.clone();
        let xr = self.point.1.clone();

        let mut hat_a_vec = Vec::new();
        let mut hat_a_tilde_vec = Vec::new();
        let mut point_vec = Vec::new();
        let mut hat_a_com_vec = Vec::new();

        let mut eq_tilde = tilde_c.clone();
        let mut eq_com = self.v_c_com.clone();

        for kappa in 0..len {
            let hat_a = mats[kappa].proj_lr(&xl, &xr);
            let (hat_a_com, hat_a_tilde) = zk_trans_seq.push_gen_blinding(hat_a);

            hat_a_vec.push(hat_a);
            hat_a_tilde_vec.push(hat_a_tilde);
            point_vec.push((xl.clone(), xr.clone()));
            hat_a_com_vec.push(hat_a_com);

            eq_tilde = eq_tilde - self.coeffs[kappa] * hat_a_tilde;

            eq_com = eq_com - hat_a_com.mul(self.coeffs[kappa]);
        }

        assert_eq!(eq_com, zksrs.blind_base.mul(eq_tilde.clone()));

        let schnorr = ZkSchnorr::new(zksrs, eq_com.clone());

        schnorr.prove(zk_trans_seq, eq_tilde);

        (hat_a_com_vec, hat_a_tilde_vec, point_vec)
    }

    pub fn verify_as_subprotocol(
        &self,
        zksrs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
    ) -> (bool, Vec<G>, Vec<(Vec<F>, Vec<F>)>) {
        let xl = self.point.0.clone();
        let xr = self.point.1.clone();

        let mut pointer_old = trans_seq.pointer;

        let mut flag = true;

        let mut point_vec = Vec::new();
        let mut hat_a_com_vec = Vec::new();

        let mut eq_com = self.v_c_com.clone();

        for kappa in 0..self.coeffs.len() {
            if let (TranElem::Group(hat_a_com),) = (trans_seq.data[pointer_old].clone(),) {
                pointer_old += 1;

                eq_com = eq_com - hat_a_com.mul(self.coeffs[kappa]);

                hat_a_com_vec.push(hat_a_com);
                point_vec.push((xl.clone(), xr.clone()));
            } else {
                // flag = false;
                panic!("Type mismatch in LinearComb");
            }
        }

        trans_seq.pointer = pointer_old;

        let schnorr = ZkSchnorr::new(zksrs, eq_com.clone());

        let flag_schnorr = schnorr.verify_as_subprotocol(trans_seq);

        flag = flag && flag_schnorr;

        (flag, hat_a_com_vec, point_vec)
    }
}

pub struct ZkLinearComb<I, F, G, MA, MB>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
    MA: MatOps<I, F>,
    MB: MatOps<I, F>,
{
    pub v_c_com: G,
    pub point: (Vec<F>, Vec<F>),
    pub coeffs: (F, F),
    _marker: PhantomData<(I, MA, MB)>,
}

impl<I, F, G, MA, MB> ZkLinearComb<I, F, G, MA, MB>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
    MA: MatOps<I, F>,
    MB: MatOps<I, F>,
{
    pub fn new(v_c_com_value: G, point_value: (Vec<F>, Vec<F>), coeffs_value: (F, F)) -> Self {
        Self {
            v_c_com: v_c_com_value,
            point: point_value,
            coeffs: coeffs_value,
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

        if mat_a.get_shape() != (m, n) || mat_b.get_shape() != (m, n) {
            panic!("Matrix shape mismatch");
        }

        let xl = self.point.0.clone();
        let xr = self.point.1.clone();

        let hat_a = mat_a.proj_lr(&xl, &xr);
        let hat_b = mat_b.proj_lr(&xl, &xr);

        let (hat_a_com, hat_a_tilde) = zk_trans_seq.push_gen_blinding(hat_a);

        let (hat_b_com, hat_b_tilde) = zk_trans_seq.push_gen_blinding(hat_b);

        let eq_tilde = tilde_c.clone()
            - self.coeffs.0.clone() * hat_a_tilde
            - self.coeffs.1.clone() * hat_b_tilde;

        let eq_com =
            self.v_c_com.clone() - hat_a_com.mul(self.coeffs.0) - hat_b_com.mul(self.coeffs.1);

        assert_eq!(eq_com, zksrs.blind_base.mul(eq_tilde.clone()));

        let schnorr = ZkSchnorr::new(zksrs, eq_com.clone());

        schnorr.prove(zk_trans_seq, eq_tilde);

        (
            hat_a_com,
            hat_b_com,
            hat_a_tilde,
            hat_b_tilde,
            (xl.clone(), xr.clone()),
            (xl.clone(), xr.clone()),
        )
    }

    pub fn verify_as_subprotocol(
        &self,
        zksrs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
    ) -> (bool, G, G, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) {
        let xl = self.point.0.clone();
        let xr = self.point.1.clone();

        let pointer_old = trans_seq.pointer;

        let hat_a_com;
        let hat_b_com;

        let mut flag = true;

        if let (TranElem::Group(hat_a_com_value), TranElem::Group(hat_b_com_value)) = (
            trans_seq.data[pointer_old].clone(),
            trans_seq.data[pointer_old + 1].clone(),
        ) {
            trans_seq.pointer += 2;
            hat_a_com = hat_a_com_value;
            hat_b_com = hat_b_com_value;
        } else {
            // flag = false;
            panic!("Type mismatch in LinearComb");
        }

        let eq_com =
            self.v_c_com.clone() - hat_a_com.mul(self.coeffs.0) - hat_b_com.mul(self.coeffs.1);

        let schnorr = ZkSchnorr::new(zksrs, eq_com.clone());

        let flag_schnorr = schnorr.verify_as_subprotocol(trans_seq);

        flag = flag && flag_schnorr;

        (
            flag,
            hat_a_com,
            hat_b_com,
            (xl.clone(), xr.clone()),
            (xl.clone(), xr.clone()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::sparse::DenseMatCM;
    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

    #[test]
    fn test_linearcomb() {
        use ark_bls12_381::Fr;
        use ark_ec::pairing::Pairing;
        use ark_std::UniformRand;

        type E = ark_bls12_381::Bls12_381;
        type F = <E as Pairing>::ScalarField;
        type G = <E as Pairing>::G1;

        let rng = &mut ark_std::rand::thread_rng();

        let log_m = 3;
        let log_n = 4;

        let m = 1 << log_m;
        let n = 1 << log_n;

        let mut mat_a = DenseMatCM::<i32, Fr>::new(m, n);
        let mut mat_b = DenseMatCM::<i32, Fr>::new(m, n);
        let mut mat_c = DenseMatCM::<i32, Fr>::new(m, n);

        let alpha = 15 as i32;
        let beta = 16 as i32;

        mat_a.gen_rand(8);
        mat_b.gen_rand(8);
        let c_data = mat_a
            .data
            .clone()
            .into_iter()
            .zip(mat_b.data.clone().into_iter())
            .map(|(a, b)| {
                a.into_par_iter()
                    .zip(b.into_par_iter())
                    .map(|(a, b)| a * alpha + b * beta)
                    .collect()
            })
            .collect();
        mat_c.set_data(c_data);

        let xl = vec![Fr::rand(rng); log_m];
        let xr = vec![Fr::rand(rng); log_n];

        let v_c = mat_c.proj_lr(&xl, &xr);

        let v_c_tilde = Fr::rand(rng);

        let zksrs = ZkSRS::<F, G>::new(G::rand(rng), G::rand(rng));

        let v_c_com = zksrs.commit(v_c, v_c_tilde);

        let protocol = ZkLinearComb::<i32, F, G, DenseMatCM<i32, F>, DenseMatCM<i32, F>>::new(
            v_c_com,
            (xl.clone(), xr.clone()),
            (alpha.into(), beta.into()),
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

        let protocol2 = ZkSerialComb::<i32, F, G, DenseMatCM<i32, F>>::new(
            v_c_com,
            (xl, xr),
            vec![alpha.into(), beta.into()],
        );

        let zk_trans_seq = &mut ZkTranSeq::new(&zksrs);

        let (_, hat_a_tilde_vec, _) = protocol2.reduce_prover(
            &zksrs,
            zk_trans_seq,
            &vec![mat_a.clone(), mat_b.clone()],
            v_c_tilde,
        );

        let hat_a_tilde = hat_a_tilde_vec[0].clone();
        let hat_b_tilde = hat_a_tilde_vec[1].clone();

        let trans_seq = &mut zk_trans_seq.publish_trans();

        let (flag, a_hat_com_vec, point_vec) = protocol2.verify_as_subprotocol(&zksrs, trans_seq);

        let a_hat_com = a_hat_com_vec[0].clone();
        let b_hat_com = a_hat_com_vec[1].clone();

        let xl_a = point_vec[0].0.clone();
        let xr_a = point_vec[0].1.clone();
        let xl_b = point_vec[1].0.clone();
        let xr_b = point_vec[1].1.clone();

        let hat_a_check = mat_a.proj_lr(&xl_a, &xr_a);

        let hat_b_check = mat_b.proj_lr(&xl_b, &xr_b);

        let a_hat_com_check = zksrs.commit(hat_a_check, hat_a_tilde);
        let b_hat_com_check = zksrs.commit(hat_b_check, hat_b_tilde);

        assert_eq!(a_hat_com, a_hat_com_check);
        assert_eq!(b_hat_com, b_hat_com_check);
        assert_eq!(flag, true);
    }
}
