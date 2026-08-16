/// Clamp a long-bitwidth integer to a short-bitwidth integer
/// proof for
/// mat_a = clamp (mat_c)
///
/// Achieved through proving
/// mat_c = sign \circ [ res_trailing + 2^k * (mat_a + res) ]
///
use ark_ec::PrimeGroup;
use ark_ff::PrimeField;
use ark_std::marker::PhantomData;

use rayon::prelude::*;

use crate::data_structures::ZkSRS;

use crate::utils::sparse::{DenseMatCM, MatOps, ShortInt};
use crate::utils::zktr::{TranSeq, ZkTranSeq};

use super::batchpcs::BatchPoints;
use super::linearcomb::ZkSerialComb;
use super::splitbool::ZkSplit;

pub struct ZkClamp<I, F, G, MA, MB>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
    MA: MatOps<I, F>,
    MB: MatOps<bool, F>,
{
    pub v_long_com: G,
    pub point: (Vec<F>, Vec<F>),
    pub k: usize,
    pub max_bit: usize,
    _marker: PhantomData<(I, MA, MB)>,
}

impl<I, F, G, MA, MB> ZkClamp<I, F, G, MA, MB>
where
    I: ShortInt,
    F: PrimeField + From<I>,
    G: PrimeGroup<ScalarField = F>,
    MA: MatOps<I, F>,
    MB: MatOps<bool, F>,
{
    pub fn new(
        v_long_com_value: G,
        point_value: (Vec<F>, Vec<F>),
        k_value: usize,
        max_bit_value: usize,
    ) -> Self {
        Self {
            v_long_com: v_long_com_value,
            point: point_value,
            k: k_value,
            max_bit: max_bit_value,
            _marker: PhantomData,
        }
    }

    pub fn reduce_prover(
        &self,
        zksrs: &ZkSRS<F, G>,
        zk_trans_seq: &mut ZkTranSeq<F, G>,
        sign_mat: &MB,
        abs_clamp: &mut MA,
        abs_res: &mut MA,
        abs_trailing: &mut MA,
        booleans_clamp: &Vec<MB>,
        booleans_res: &Vec<MB>,
        booleans_trailing: &Vec<MB>,
        tilde_c: F,
    ) -> (
        G,
        F,
        (Vec<F>, Vec<F>),
        G,
        F,
        (Vec<F>, Vec<F>),
        Vec<G>,
        Vec<F>,
        Vec<(Vec<F>, Vec<F>)>,
        Vec<G>,
        Vec<F>,
        Vec<(Vec<F>, Vec<F>)>,
        Vec<G>,
        Vec<F>,
        Vec<(Vec<F>, Vec<F>)>,
    ) {
        let log_m = self.point.0.len();
        let log_n = self.point.1.len();

        let m = 1 << log_m;
        let n = 1 << log_n;

        let k = self.k;
        let max_bit = self.max_bit;

        if (m, n) != sign_mat.get_shape()
            || (m, n) != abs_clamp.get_shape()
            || (m, n) != abs_res.get_shape()
            || (m, n) != abs_trailing.get_shape()
            || (m, n) != booleans_clamp[0].get_shape()
            || (m, n) != booleans_res[0].get_shape()
            || (m, n) != booleans_trailing[0].get_shape()
        {
            panic!("Matrix shape mismatch in clamp");
        }

        if booleans_clamp.len() != k
            || booleans_trailing.len() != k
            || booleans_res.len() != max_bit
        {
            panic!("Boolean vector length mismatch in clamp");
        }

        println!("- Proving linear combination in clamp");

        // prove
        // mat_c = (sign \circ res_trailing)
        // + 2^k * (sign \circ mat_a + sign \circ res) ]
        let linearcomb = ZkSerialComb::<I, F, G, DenseMatCM<I, F>>::new(
            self.v_long_com.clone(),
            self.point.clone(),
            vec![F::one(), F::from((1 << k) as i32), F::from((1 << k) as i32)],
        );

        // println!("trailing {:?}", abs_trailing.to_dense().data[0]);
        // println!("sign {:?}", sign_mat.to_dense().data[0]);
        // println!("clamp {:?}", abs_clamp.to_dense().data[0]);
        // println!("res {:?}", abs_res.to_dense().data[0]);

        let mut mat_vec = Vec::new();

        let mut part1_mat = DenseMatCM::<I, F>::new(m, n);
        let part_1_data: Vec<Vec<I>> = abs_trailing
            .to_dense()
            .data
            .iter()
            .zip(sign_mat.to_dense().data.iter())
            .map(|(vec_a, vec_b)| {
                vec_a
                    .par_iter()
                    .zip(vec_b.par_iter())
                    .map(|(a, b)| if *b { *a } else { I::from_i32(-1).mul_mix(*a) })
                    .collect::<Vec<I>>()
            })
            .collect();
        // println!("part1 trailing {:?}", part_1_data[0]);
        part1_mat.set_data(part_1_data);

        mat_vec.push(part1_mat);

        let mut part2_mat = DenseMatCM::<I, F>::new(m, n);
        let part_2_data: Vec<Vec<I>> = abs_clamp
            .to_dense()
            .data
            .iter()
            .zip(sign_mat.to_dense().data.iter())
            .map(|(vec_a, vec_b)| {
                vec_a
                    .par_iter()
                    .zip(vec_b.par_iter())
                    .map(|(a, b)| if *b { *a } else { I::from_i32(-1).mul_mix(*a) })
                    .collect::<Vec<I>>()
            })
            .collect();
        // println!("part2 clamp {:?}", part_2_data[0]);
        part2_mat.set_data(part_2_data);

        mat_vec.push(part2_mat);

        let mut part3_mat = DenseMatCM::<I, F>::new(m, n);
        let part_3_data: Vec<Vec<I>> = abs_res
            .to_dense()
            .data
            .iter()
            .zip(sign_mat.to_dense().data.iter())
            .map(|(vec_a, vec_b)| {
                vec_a
                    .par_iter()
                    .zip(vec_b.par_iter())
                    .map(|(a, b)| if *b { *a } else { I::from_i32(-1).mul_mix(*a) })
                    .collect::<Vec<I>>()
            })
            .collect();
        // println!("part3 res {:?}", part_3_data[0]);
        part3_mat.set_data(part_3_data);

        mat_vec.push(part3_mat);

        let (linear_coms, linear_tildes, linear_points) =
            linearcomb.reduce_prover(zksrs, zk_trans_seq, &mat_vec, tilde_c);

        mat_vec.clear();

        let trailing_com = linear_coms[0].clone();
        let clamp_com = linear_coms[1].clone();
        let res_com = linear_coms[2].clone();

        let trailing_tilde = linear_tildes[0];
        let clamp_tilde = linear_tildes[1];
        let res_tilde = linear_tildes[2];

        let trailing_point = linear_points[0].clone();
        let clamp_point = linear_points[1].clone();
        let res_point = linear_points[2].clone();

        // Split the mat
        println!("- Splitting the trailing matrices in clamp");

        let split_trailing =
            ZkSplit::<I, F, G, DenseMatCM<I, F>, MB>::new(trailing_com, trailing_point, k);

        let (
            v_sign_com_1,
            v_boolean_com_vec_trailing,
            v_sign_tilde_1,
            v_boolean_tilde_vec_trailing,
            sign_point_1,
            point_boolean_vec_trailing,
        ) = split_trailing.reduce_prover(
            zksrs,
            zk_trans_seq,
            &abs_trailing.to_dense(),
            &sign_mat,
            booleans_trailing,
            trailing_tilde,
        );

        println!("Splitting the clamp matrices in clamp");

        let split_clamp =
            ZkSplit::<I, F, G, DenseMatCM<I, F>, MB>::new(clamp_com, clamp_point.clone(), k);

        let (
            v_sign_com_2,
            v_boolean_com_vec_clamp,
            v_sign_tilde_2,
            v_boolean_tilde_vec_clamp,
            sign_point_2,
            point_boolean_vec_clamp,
        ) = split_clamp.reduce_prover(
            zksrs,
            zk_trans_seq,
            &abs_clamp.to_dense(),
            &sign_mat,
            booleans_clamp,
            clamp_tilde,
        );

        abs_clamp.clear();

        println!("Splitting the res matrices in clamp");

        let split_res = ZkSplit::<I, F, G, DenseMatCM<I, F>, MB>::new(res_com, res_point, max_bit);

        let (
            v_sign_com_3,
            v_boolean_com_vec_res,
            v_sign_tilde_3,
            v_boolean_tilde_vec_res,
            sign_point_3,
            point_boolean_vec_res,
        ) = split_res.reduce_prover(
            zksrs,
            zk_trans_seq,
            &abs_res.to_dense(),
            &sign_mat,
            booleans_res,
            res_tilde,
        );

        abs_res.clear();

        // Combine the sign mat evaluations
        let mut batch = BatchPoints::<bool, F, G>::new();
        batch.push_point(v_sign_com_1, sign_point_1);
        batch.push_point(v_sign_com_2, sign_point_2);
        batch.push_point(v_sign_com_3, sign_point_3);

        let (v_sign_com, v_sign_tilde, sign_point) = batch.reduce_prover(
            zksrs,
            zk_trans_seq,
            sign_mat,
            vec![v_sign_tilde_1, v_sign_tilde_2, v_sign_tilde_3],
        );

        abs_trailing.clear();

        (
            clamp_com,
            clamp_tilde,
            clamp_point,
            v_sign_com,
            v_sign_tilde,
            sign_point,
            v_boolean_com_vec_clamp,
            v_boolean_tilde_vec_clamp,
            point_boolean_vec_clamp,
            v_boolean_com_vec_res,
            v_boolean_tilde_vec_res,
            point_boolean_vec_res,
            v_boolean_com_vec_trailing,
            v_boolean_tilde_vec_trailing,
            point_boolean_vec_trailing,
        )
    }

    pub fn verify_as_subprotocol(
        &self,
        zksrs: &ZkSRS<F, G>,
        trans_seq: &mut TranSeq<F, G>,
    ) -> (
        bool,
        G,
        (Vec<F>, Vec<F>),
        G,
        (Vec<F>, Vec<F>),
        Vec<G>,
        Vec<(Vec<F>, Vec<F>)>,
        Vec<G>,
        Vec<(Vec<F>, Vec<F>)>,
        Vec<G>,
        Vec<(Vec<F>, Vec<F>)>,
    ) {
        let k = self.k;
        let max_bit = self.max_bit;

        // prove
        // mat_c = (sign \circ res_trailing)
        // + 2^k * (sign \circ mat_a + sign \circ res) ]
        let linearcomb = ZkSerialComb::<I, F, G, DenseMatCM<I, F>>::new(
            self.v_long_com.clone(),
            self.point.clone(),
            vec![F::one(), F::from((1 << k) as i32), F::from((1 << k) as i32)],
        );

        let (flag1, linear_coms, linear_points) =
            linearcomb.verify_as_subprotocol(zksrs, trans_seq);

        let trailing_com = linear_coms[0].clone();
        let clamp_com = linear_coms[1].clone();
        let res_com = linear_coms[2].clone();

        let trailing_point = linear_points[0].clone();
        let clamp_point = linear_points[1].clone();
        let res_point = linear_points[2].clone();

        // Split the mat
        let split_trailing =
            ZkSplit::<I, F, G, DenseMatCM<I, F>, MB>::new(trailing_com, trailing_point, k);

        let (
            flag2,
            v_sign_com_1,
            v_boolean_com_vec_trailing,
            sign_point_1,
            point_boolean_vec_trailing,
        ) = split_trailing.verify_as_subprotocol(zksrs, trans_seq);

        let split_clamp =
            ZkSplit::<I, F, G, DenseMatCM<I, F>, MB>::new(clamp_com, clamp_point.clone(), k);

        let (flag3, v_sign_com_2, v_boolean_com_vec_clamp, sign_point_2, point_boolean_vec_clamp) =
            split_clamp.verify_as_subprotocol(zksrs, trans_seq);

        let split_res = ZkSplit::<I, F, G, DenseMatCM<I, F>, MB>::new(res_com, res_point, max_bit);

        let (flag4, v_sign_com_3, v_boolean_com_vec_res, sign_point_3, point_boolean_vec_res) =
            split_res.verify_as_subprotocol(zksrs, trans_seq);

        // Combine the sign mat evaluations
        let mut batch = BatchPoints::<bool, F, G>::new();
        batch.push_point(v_sign_com_1, sign_point_1);
        batch.push_point(v_sign_com_2, sign_point_2);
        batch.push_point(v_sign_com_3, sign_point_3);

        let (flag5, v_sign_com, sign_point) = batch.verify_as_subprotocol(zksrs, trans_seq);

        (
            flag1 && flag2 && flag3 && flag4 && flag5,
            clamp_com,
            clamp_point,
            v_sign_com,
            sign_point,
            v_boolean_com_vec_clamp,
            point_boolean_vec_clamp,
            v_boolean_com_vec_res,
            point_boolean_vec_res,
            v_boolean_com_vec_trailing,
            point_boolean_vec_trailing,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::mockmat::MockMultiMuls;
    use crate::utils::sparse::DenseBlockMat;

    #[test]
    fn test_clamp() {
        use ark_ec::pairing::Pairing;
        use ark_std::UniformRand;

        type E = ark_bls12_381::Bls12_381;
        type F = <E as Pairing>::ScalarField;
        type G = <E as Pairing>::G1;

        let k = 5;
        let mut mock = MockMultiMuls::<F>::new(4, 8, k);
        mock.gen_mock();

        println!(
            "*** full mat abs {:?}",
            mock.block_mat_o_full.to_dense().data[0]
        );
        println!("*** res mat abs {:?}", mock.abs_o_res.to_dense().data[0]);
        println!("*** o_mat {:?}", mock.block_mat_o.to_dense().data[0]);
        println!(
            "*** trailing mat {:?}",
            mock.abs_o_trailing.to_dense().data[0]
        );
        println!("Mock data generated");

        let long_mat = mock.block_mat_o_full.clone();

        let (m, n) = long_mat.get_shape();
        let log_m = m.ilog2() as usize;
        let log_n = n.ilog2() as usize;
        let max_bit = k + log_n;

        let rng = &mut ark_std::rand::thread_rng();

        let zksrs = ZkSRS::<F, G>::new(G::rand(rng), G::rand(rng));

        let xl = vec![F::rand(rng); log_m];
        let xr = vec![F::rand(rng); log_n];

        let v_long = long_mat.proj_lr(&xl, &xr);
        let v_long_tilde = F::rand(rng);

        let v_long_com = zksrs.commit(v_long, v_long_tilde);

        let protocol = ZkClamp::<i32, F, G, DenseBlockMat<i32, F>, DenseBlockMat<bool, F>>::new(
            v_long_com,
            (xl.clone(), xr.clone()),
            k,
            max_bit,
        );

        let zk_trans_seq = &mut ZkTranSeq::new(&zksrs);

        let (
            _,
            _,
            _,
            _,
            v_sign_tilde,
            _,
            _,
            v_boolean_tilde_vec_clamp,
            _,
            _,
            v_boolean_tilde_vec_res,
            _,
            _,
            v_boolean_tilde_vec_trailing,
            _,
        ) = protocol.reduce_prover(
            &zksrs,
            zk_trans_seq,
            &mock.sign_o,
            &mut mock.abs_o,
            &mut mock.abs_o_res,
            &mut mock.abs_o_trailing,
            &mock.booleans_o,
            &mock.booleans_o_res,
            &mock.booleans_o_trailing,
            v_long_tilde,
        );

        let trans_seq = &mut zk_trans_seq.publish_trans();

        let (
            flag,
            _,
            _,
            v_sign_com,
            sign_point,
            v_boolean_com_vec_clamp,
            point_boolean_vec_clamp,
            v_boolean_com_vec_res,
            point_boolean_vec_res,
            v_boolean_com_vec_trailing,
            point_boolean_vec_trailing,
        ) = protocol.verify_as_subprotocol(&zksrs, trans_seq);

        let hat_sign_check = mock.sign_o.proj_lr(&sign_point.0, &sign_point.1);

        let sign_hat_com_check = zksrs.commit(hat_sign_check, v_sign_tilde);

        assert_eq!(flag, true);
        assert_eq!(v_sign_com, sign_hat_com_check);

        let v_b_1_check_clamp = mock.booleans_o[1]
            .proj_lr(&point_boolean_vec_clamp[1].0, &point_boolean_vec_clamp[1].1);
        let v_b_1_com_check_clamp = zksrs.commit(v_b_1_check_clamp, v_boolean_tilde_vec_clamp[1]);

        assert_eq!(v_boolean_com_vec_clamp[1], v_b_1_com_check_clamp);

        let v_b_1_check_res = mock.booleans_o_res[1]
            .proj_lr(&point_boolean_vec_res[1].0, &point_boolean_vec_res[1].1);
        let v_b_1_com_check_res = zksrs.commit(v_b_1_check_res, v_boolean_tilde_vec_res[1]);

        assert_eq!(v_boolean_com_vec_res[1], v_b_1_com_check_res);

        let v_b_1_check_trailing = mock.booleans_o_trailing[1].proj_lr(
            &point_boolean_vec_trailing[1].0,
            &point_boolean_vec_trailing[1].1,
        );
        let v_b_1_com_check_trailing =
            zksrs.commit(v_b_1_check_trailing, v_boolean_tilde_vec_trailing[1]);

        assert_eq!(v_boolean_com_vec_trailing[1], v_b_1_com_check_trailing);
    }
}
