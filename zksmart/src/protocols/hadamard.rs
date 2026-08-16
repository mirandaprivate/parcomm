/// Implement the Hadamard protocol
///
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use ark_ec::PrimeGroup;
use ark_ff::PrimeField;
use ark_std::marker::PhantomData;

use crate::data_structures::ZkSRS;

use crate::utils::linear;
use crate::utils::sparse::{MatOps, ShortInt};
use crate::utils::xi;
use crate::utils::zktr::{TranSeq, ZkTranSeq};

use super::litebullet::ZkLiteBullet;

pub struct ZkHadamard<IA, IB, F, G, MA, MB>
where
    IA: ShortInt,
    IB: ShortInt,
    F: PrimeField + From<IA> + From<IB>,
    G: PrimeGroup<ScalarField = F>,
    MA: MatOps<IA, F>,
    MB: MatOps<IB, F>,
{
    pub v_c_com: G,
    pub point: (Vec<F>, Vec<F>),
    _marker: PhantomData<(IA, IB, MA, MB)>,
}

impl<IA, IB, F, G, MA, MB> ZkHadamard<IA, IB, F, G, MA, MB>
where
    IA: ShortInt,
    IB: ShortInt,
    F: PrimeField + From<IA> + From<IB>,
    G: PrimeGroup<ScalarField = F>,
    MA: MatOps<IA, F>,
    MB: MatOps<IB, F>,
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
        mat_a: &MA,
        mat_b: &MB,
        tilde_c: F,
    ) -> (G, G, F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) {
        let log_m = self.point.0.len();
        let log_n = self.point.1.len();

        let m = 1 << log_m;
        let n = 1 << log_n;

        if (m, n) != mat_a.get_shape() || (m, n) != mat_b.get_shape() {
            println!("m, n: {:?}", (m, n));
            println!("shape: {:?}", mat_a.get_shape());
            println!("shape: {:?}", mat_b.get_shape());

            panic!("Matrix dimension mismatch");
        }

        // do not keep projection
        // let vec_a = mat_a.to_vec();

        let dense_a = mat_a.to_dense();

        let xl = self.point.0.clone();
        let xr = self.point.1.clone();

        // // swap xl and xr
        let x = [xr.clone(), xl.clone()].concat();

        // let xi =
        // xi::xi_from_challenges::<F>(&x);

        // size m
        let xi_l = xi::xi_from_challenges::<F>(&xl);
        // size n
        let xi_r = xi::xi_from_challenges::<F>(&xr);

        let len = m * n;

        let vec_a_prime = (0..(m * n))
            .into_par_iter()
            .map(|idx| {
                let row = idx % m;
                let col = idx / m;
                xi_l[row] * xi_r[col] * F::from(dense_a.data[col][row])
            })
            .collect();

        std::mem::drop(xi_l);
        std::mem::drop(xi_r);

        let vec_b = mat_b.to_field_vec();

        let litebullet = ZkLiteBullet::new(self.v_c_com, len);

        let (a_hat_com, b_hat_com, a_hat_tilde, b_hat_tilde, challenges_inv, challenges) =
            litebullet.reduce_prover(zksrs, zk_trans_seq, &vec_a_prime, &vec_b, tilde_c);

        let x_prime = linear::vec_element_wise_mul(&x, &challenges_inv);

        let xl_a = x_prime[log_n..].to_vec();
        let xr_a = x_prime[..log_n].to_vec();

        let xl_b = challenges[log_n..].to_vec();
        let xr_b = challenges[..log_n].to_vec();

        (
            a_hat_com,
            b_hat_com,
            a_hat_tilde,
            b_hat_tilde,
            (xl_a, xr_a),
            (xl_b, xr_b),
        )
    }

    pub fn verify_as_subprotocol(
        &self,
        zksrs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
    ) -> (bool, G, G, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) {
        let log_m = self.point.0.len();
        let log_n = self.point.1.len();

        let m = 1 << log_m;
        let n = 1 << log_n;

        let xl = self.point.0.clone();
        let xr = self.point.1.clone();

        // swap xl and xr
        let x = [xr, xl].concat();

        let len = m * n;

        let litebullet = ZkLiteBullet::new(self.v_c_com, len);

        let (flag, a_hat_com, b_hat_com, challenges_inv, challenges) =
            litebullet.verify_as_subprotocol(zksrs, trans_seq);

        let x_prime = linear::vec_element_wise_mul(&x, &challenges_inv);

        let xl_a = x_prime[log_n..].to_vec();
        let xr_a = x_prime[..log_n].to_vec();

        let xl_b = challenges[log_n..].to_vec();
        let xr_b = challenges[..log_n].to_vec();

        (flag, a_hat_com, b_hat_com, (xl_a, xr_a), (xl_b, xr_b))
    }
}

pub struct ZkHadamardField<F, G>
where
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    pub v_c_com: G,
    pub point: (Vec<F>, Vec<F>),
}

impl<F, G> ZkHadamardField<F, G>
where
    F: PrimeField,
    G: PrimeGroup<ScalarField = F>,
{
    pub fn new(v_c_com_value: G, point_value: (Vec<F>, Vec<F>)) -> Self {
        Self {
            v_c_com: v_c_com_value,
            point: point_value,
        }
    }

    pub fn reduce_prover(
        &self,
        zksrs: &ZkSRS<F, G>,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        mat_a: &Vec<Vec<F>>,
        mat_b: &Vec<Vec<F>>,
        tilde_c: F,
    ) -> (G, G, F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) {
        let log_m = self.point.0.len();
        let log_n = self.point.1.len();

        let m = 1 << log_m;
        let n = 1 << log_n;

        if (m, n) != (mat_a[0].len(), mat_a.len()) || (m, n) != (mat_b[0].len(), mat_b.len()) {
            println!("m, n: {:?}", (m, n));
            println!("shape: {:?}", (mat_a[0].len(), mat_a.len()));
            println!("shape: {:?}", (mat_b[0].len(), mat_b.len()));
            panic!("Matrix dimension mismatch");
        }

        let xl = self.point.0.clone();
        let xr = self.point.1.clone();

        // swap xl and xr
        let x = [xr, xl].concat();

        let xi = xi::xi_from_challenges::<F>(&x);

        let len = xi.len();

        // let mut vec_a_prime =
        // linear::vec_element_wise_mul(&vec_a, &xi);
        // do not keep projection
        let vec_a_prime: Vec<F> = (0..m * n)
            .into_par_iter()
            .map(|idx| {
                // let row = idx % n;
                // let col = idx / n;
                mat_a[idx % n][idx / n].clone() * xi[idx]
            })
            .collect();

        let vec_b: Vec<F> = (0..m * n)
            .into_par_iter()
            .map(|idx| {
                // let row = idx % n;
                // let col = idx / n;
                mat_b[idx % n][idx / n].clone()
            })
            .collect();

        std::mem::drop(xi);

        let litebullet = ZkLiteBullet::new(self.v_c_com, len);

        let (a_hat_com, b_hat_com, a_hat_tilde, b_hat_tilde, challenges_inv, challenges) =
            litebullet.reduce_prover(zksrs, zk_trans_seq, &vec_a_prime, &vec_b, tilde_c);

        let x_prime = linear::vec_element_wise_mul(&x, &challenges_inv);

        let xl_a = x_prime[log_n..].to_vec();
        let xr_a = x_prime[..log_n].to_vec();

        let xl_b = challenges[log_n..].to_vec();
        let xr_b = challenges[..log_n].to_vec();

        (
            a_hat_com,
            b_hat_com,
            a_hat_tilde,
            b_hat_tilde,
            (xl_a, xr_a),
            (xl_b, xr_b),
        )
    }

    pub fn verify_as_subprotocol(
        &self,
        zksrs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
    ) -> (bool, G, G, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) {
        let log_m = self.point.0.len();
        let log_n = self.point.1.len();

        let m = 1 << log_m;
        let n = 1 << log_n;

        let xl = self.point.0.clone();
        let xr = self.point.1.clone();

        // swap xl and xr
        let x = [xr, xl].concat();

        let len = m * n;

        let litebullet = ZkLiteBullet::new(self.v_c_com, len);

        let (flag, a_hat_com, b_hat_com, challenges_inv, challenges) =
            litebullet.verify_as_subprotocol(zksrs, trans_seq);

        let x_prime = linear::vec_element_wise_mul(&x, &challenges_inv);

        let xl_a = x_prime[log_n..].to_vec();
        let xr_a = x_prime[..log_n].to_vec();

        let xl_b = challenges[log_n..].to_vec();
        let xr_b = challenges[..log_n].to_vec();

        (flag, a_hat_com, b_hat_com, (xl_a, xr_a), (xl_b, xr_b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::sparse::DenseMatCM;
    use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

    #[test]
    fn test_hadamard() {
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
                    .map(|(a, b)| a * b)
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

        let protocol = ZkHadamard::<i32, i32, F, G, DenseMatCM<i32, F>, DenseMatCM<i32, F>>::new(
            v_c_com,
            (xl, xr),
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
}
