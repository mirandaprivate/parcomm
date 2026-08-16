/// Verify the projection of a rotation matrix
/// by recursively invoking Hadamard protocols
///
/// We avoid using k full-bitwidth inputs
/// Instead, we compute the input to the Hadamard protocol
/// on run time
///
use ark_ec::PrimeGroup;
use ark_ff::PrimeField;
use ark_std::marker::PhantomData;
// use rayon::iter::IntoParallelIterator;

use crate::data_structures::ZkSRS;
use crate::utils::sparse::{DenseMatCM, MatOps, RotationMatIndexFormat, ShortInt};
use crate::utils::zktr::{TranElem, TranSeq, ZkTranSeq};
use crate::utils::{linear, xi};

use super::hadamard::ZkHadamardField;

pub struct ZkRotationProj<I, F, G>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
{
    pub v_c_com: G,
    pub point: (Vec<F>, Vec<F>),
    _marker: PhantomData<I>,
}

impl<I, F, G> ZkRotationProj<I, F, G>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
{
    pub fn new(v_c_com_value: G, point_value: (Vec<F>, Vec<F>)) -> Self {
        Self {
            v_c_com: v_c_com_value,
            point: point_value,
            _marker: PhantomData,
        }
    }

    pub fn reduce_prover(
        &self,
        zksrs: &ZkSRS<F, G>,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        mat_rotation: &RotationMatIndexFormat<I, F>,
        if_invalid: &Vec<bool>,
        tilde_c: F,
    ) -> (G, Vec<G>, F, Vec<F>, Vec<F>, Vec<Vec<F>>) {
        // let mut mat_rotation_positive =
        // RotationMatIndexFormat::<I, F>::new(
        //     mat_rotation.get_shape().0,
        //     mat_rotation.get_shape().1,
        // );

        // let new_rotation_data =
        // mat_rotation.data.par_iter().map(|&val| {
        //     if val < 0 {
        //         0
        //     } else {
        //         val
        //     }
        // }).collect::<Vec<i32>>();
        // mat_rotation_positive.set_data(new_rotation_data);

        let log_m = self.point.0.len();
        let log_n = self.point.1.len();

        let m = 1 << log_m;
        // let n = 1 << log_n;

        // if (m,n) != (mat_rotation_positive.get_shape()) {
        //     panic!("Matrix shape mismatch");
        // }

        let mut xx_cur = self.point.0.clone();

        let mut mat_if_invalid = DenseMatCM::<bool, F>::new(m, 1);
        mat_if_invalid.from_vec(if_invalid);
        let v_if_invalid = mat_if_invalid.proj_lr(&self.point.0.clone(), &Vec::new());

        let (v_if_invalid_com, v_if_invalide_tilde) = zk_trans_seq.push_gen_blinding(v_if_invalid);

        let mut c_com = self.v_c_com.clone() + v_if_invalid_com;
        let mut c_tilde = tilde_c.clone() + v_if_invalide_tilde;

        let mut v_b_com_vec = Vec::new();
        let mut v_b_tilde_vec = Vec::new();
        let mut point_vec = Vec::new();

        let xr = self.point.1.clone();

        for kappa in 0..log_n - 1 {
            println!("-- {:?}-th iteration in rotation proj", kappa);
            let cur_x = xr[kappa];

            let mut a_vec_gen = vec![F::one(); log_n];

            a_vec_gen[kappa] = cur_x;

            let b_vec_gen_1 = vec![F::one(); kappa + 1].to_vec();

            let b_vec_gen_2 = xr[kappa + 1..].to_vec();

            let b_vec_gen = [b_vec_gen_1, b_vec_gen_2].concat();

            // let a_vec_cur =
            // mat_rotation_positive.proj_right_challenges(&a_vec_gen);

            let a_vec_cur = linear::vec_addition(
                &mat_rotation.proj_right_challenges(&a_vec_gen),
                &mat_if_invalid.proj_right_challenges(&a_vec_gen),
            );

            // let b_vec_cur =
            // mat_rotation_positive.proj_right_challenges(&b_vec_gen);

            let b_vec_cur = linear::vec_addition(
                &mat_rotation.proj_right_challenges(&b_vec_gen),
                &mat_if_invalid.proj_right_challenges(&b_vec_gen),
            );

            let hadamard = ZkHadamardField::new(c_com, (xx_cur, Vec::new()));

            let (hat_a_com, hat_b_com, hat_a_tilde, hat_b_tilde, (xa_prime, _), (xb_prime, _)) =
                hadamard.reduce_prover(
                    zksrs,
                    zk_trans_seq,
                    &vec![a_vec_cur],
                    &vec![b_vec_cur],
                    c_tilde,
                );

            let ones = vec![F::one(); log_m];
            let ip_xa_prime_ones = xi::xi_ip_from_challenges::<F>(&xa_prime, &ones);

            let cur_x_inv = (cur_x - F::one()).inverse().unwrap();

            let hat_a_com_prime =
                (hat_a_com - zksrs.com_base.mul(&ip_xa_prime_ones)).mul(&cur_x_inv);

            let hat_a_tilde_prime = hat_a_tilde * cur_x_inv;

            v_b_com_vec.push(hat_a_com_prime);
            v_b_tilde_vec.push(hat_a_tilde_prime);
            point_vec.push(xa_prime);

            xx_cur = xb_prime;
            c_com = hat_b_com;
            c_tilde = hat_b_tilde;
        }

        let ones = vec![F::one(); log_m];
        let ip_xa_prime_ones = xi::xi_ip_from_challenges::<F>(&xx_cur, &ones);

        let cur_x_inv = (xr[log_n - 1] - F::one()).inverse().unwrap();

        let hat_a_com_prime = (c_com - zksrs.com_base.mul(&ip_xa_prime_ones)).mul(&cur_x_inv);

        let hat_a_tilde_prime = c_tilde * cur_x_inv;

        v_b_com_vec.push(hat_a_com_prime);
        v_b_tilde_vec.push(hat_a_tilde_prime);
        point_vec.push(xx_cur);

        (
            v_if_invalid_com,
            v_b_com_vec,
            v_if_invalide_tilde,
            v_b_tilde_vec,
            self.point.0.clone(),
            point_vec,
        )
    }

    pub fn verify_as_subprotocol(
        &self,
        zksrs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
    ) -> (bool, G, Vec<G>, Vec<F>, Vec<Vec<F>>) {
        let log_m = self.point.0.len();
        let log_n = self.point.1.len();

        let mut xx_cur = self.point.0.clone();

        let v_if_invalid_com;
        if let TranElem::Group(v_if_invalid_com_value) = trans_seq.data[trans_seq.pointer].clone() {
            trans_seq.pointer += 1;
            v_if_invalid_com = v_if_invalid_com_value;
        } else {
            panic!("Invalid input");
        }

        let mut c_com = self.v_c_com.clone() + v_if_invalid_com;

        let mut v_b_com_vec = Vec::new();
        let mut point_vec = Vec::new();

        let mut flag = true;

        let xr = self.point.1.clone();

        for kappa in 0..log_n - 1 {
            let cur_x = xr[kappa];

            let hadamard = ZkHadamardField::new(c_com, (xx_cur, Vec::new()));

            let (flag_cur, hat_a_com, hat_b_com, (xa_prime, _), (xb_prime, _)) =
                hadamard.verify_as_subprotocol(zksrs, trans_seq);

            flag = flag && flag_cur;

            let ones = vec![F::one(); log_m];
            let ip_xa_prime_ones = xi::xi_ip_from_challenges::<F>(&xa_prime, &ones);

            let cur_x_inv = (cur_x - F::one()).inverse().unwrap();

            let hat_a_com_prime =
                (hat_a_com - zksrs.com_base.mul(&ip_xa_prime_ones)).mul(&cur_x_inv);

            v_b_com_vec.push(hat_a_com_prime);
            point_vec.push(xa_prime);

            xx_cur = xb_prime;
            c_com = hat_b_com;
        }

        let ones = vec![F::one(); log_m];
        let ip_xa_prime_ones = xi::xi_ip_from_challenges::<F>(&xx_cur, &ones);

        let cur_x_inv = (xr[log_n - 1] - F::one()).inverse().unwrap();

        let hat_a_com_prime = (c_com - zksrs.com_base.mul(&ip_xa_prime_ones)).mul(&cur_x_inv);

        v_b_com_vec.push(hat_a_com_prime);
        point_vec.push(xx_cur);

        (
            flag,
            v_if_invalid_com,
            v_b_com_vec,
            self.point.0.clone(),
            point_vec,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::linear;
    use crate::utils::xi;

    use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

    use ark_std::{One, Zero};

    #[test]
    fn test_rotation() {
        use ark_bls12_381::Fr;
        use ark_ec::pairing::Pairing;
        use ark_std::UniformRand;

        type E = ark_bls12_381::Bls12_381;
        type F = <E as Pairing>::ScalarField;
        type G = <E as Pairing>::G1;

        let rng = &mut ark_std::rand::thread_rng();

        let log_m = 2 as usize;
        let log_n = 5 as usize;

        let m = (1 << log_m) as usize;
        let n = (1 << log_n) as usize;

        let mut mat_c = RotationMatIndexFormat::<i32, Fr>::new(m, n);

        mat_c.gen_rand(8);

        let xl = vec![Fr::rand(rng); log_m];
        let xr = vec![Fr::rand(rng); log_n];

        let v_c = mat_c.proj_lr(&xl.clone(), &xr.clone());

        let v_c_tilde = Fr::rand(rng);

        let zksrs = ZkSRS::<F, G>::new(G::rand(rng), G::rand(rng));

        let v_c_com = zksrs.commit(v_c, v_c_tilde);

        let protocol = ZkRotationProj::<i32, F, G>::new(v_c_com, (xl.clone(), xr.clone()));

        let zk_trans_seq = &mut ZkTranSeq::new(&zksrs);

        let (if_invalid, boolean_vecs) = mat_c.split_into_boolean();

        let (_, _, _, hat_b_tilde_vec, _, _) =
            protocol.reduce_prover(&zksrs, zk_trans_seq, &mat_c, &if_invalid, v_c_tilde);

        let trans_seq = &mut zk_trans_seq.publish_trans();

        let (flag, _, b_hat_com_vec, _, b_point_vec) =
            protocol.verify_as_subprotocol(&zksrs, trans_seq);

        assert_eq!(flag, true);

        for kappa in 0..log_n {
            let b_hat_com = b_hat_com_vec[kappa].clone();
            let b_point = b_point_vec[kappa].clone();

            let b_vec = boolean_vecs[kappa].clone();

            let b_vec_field = b_vec
                .par_iter()
                .map(|x| if *x { F::one() } else { F::zero() })
                .collect::<Vec<F>>();

            let xi_cur = xi::xi_from_challenges(&b_point);

            let hat_b_check = linear::inner_product::<F>(&b_vec_field, &xi_cur);

            let b_hat_com_check = zksrs.commit(hat_b_check, hat_b_tilde_vec[kappa]);

            assert_eq!(b_hat_com, b_hat_com_check);
        }

        println!("Rotation test passed");
    }
}
