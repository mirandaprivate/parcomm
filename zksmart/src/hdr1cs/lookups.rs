/// HDR1CS for multiple matrix multiplications
///
use ark_ec::pairing::Pairing;
use ark_serialize::CanonicalSerialize;
use ark_std::Zero;

use ark_poly_commit::smart_pc::data_structures::Trans as PcsTrans;

use crate::data_structures::SRS;

use crate::utils::sparse::{
    DenseBlockMat, DenseMatCM, DiagBlockMat, MatOps, RotationMatIndexFormat, SparseKronecker,
};
use crate::utils::xi;
use crate::utils::zktr::{TranElem, TranSeq, ZkTranSeq};

use crate::utils::mockmat::MockMultiLookups;

use crate::protocols::batchpcs::{
    self, BatchPoints, BatchPointsRotation, ComContainer, EvalContainer, MatContainer,
    PointsContainer,
};
use crate::protocols::{kronecker::ZkKronecker, mul::ZkMatMul, rotationproj::ZkRotationProj};

pub struct HdR1csLookup<E: Pairing> {
    pub num_mat: usize,
    pub n: usize,
    pub k: usize,
    pub large_dim: usize,
    pub large_large_dim: usize,
    pub max_dim_for_q: usize,
    pub witness: MockMultiLookups<E::ScalarField>,
    pub mat_container: MatContainer<E::ScalarField>,
    pub com_container: ComContainer<E>,
    pub eval_container: EvalContainer<E::ScalarField, E::G1>,
    pub points_container: PointsContainer<E::ScalarField>,
}

impl<E: Pairing> HdR1csLookup<E> {
    pub fn new(num_mat: usize, n: usize, k: usize) -> Self {
        let mock = MockMultiLookups::new(num_mat, n, k);

        let log_n = n.ilog2() as usize;

        if n & (n - 1) != 0 {
            panic!("n not a power of 2");
        }

        if num_mat & (num_mat - 1) != 0 {
            panic!("num not a power of 2");
        }

        let log_num = num_mat.ilog2() as usize;
        let mut half_log_num = log_num / 2;

        if log_num != 2 * half_log_num {
            half_log_num += 1;
            // panic!("num not a square number, not supported");
        }

        let log_q = half_log_num + log_n;

        let max_dim_for_q = 1 << log_q;
        let large_dim = n * num_mat;
        let large_large_dim = n * n * num_mat;

        Self {
            num_mat: num_mat,
            n: n,
            k: k,
            large_dim: large_dim,
            large_large_dim: large_large_dim,
            max_dim_for_q: max_dim_for_q,
            witness: mock,
            mat_container: MatContainer::new(k),
            com_container: ComContainer::new(),
            eval_container: EvalContainer::new(),
            points_container: PointsContainer::new(),
        }
    }

    pub fn mat_container_from_witness(&mut self) -> MatContainer<E::ScalarField> {
        let mut mat_container = MatContainer::new(self.k);

        let mut key_mat = DenseMatCM::<i32, E::ScalarField>::new(1 << self.k, 1);
        key_mat.set_data(vec![self.witness.key_vec.clone()]);
        mat_container.dense_i32.push(key_mat);

        let mut value_mat = DenseMatCM::<i32, E::ScalarField>::new(1 << self.k, 1);
        value_mat.set_data(vec![self.witness.value_vec.clone()]);
        mat_container.dense_i32.push(value_mat);

        mat_container
            .dense_bool
            .push(self.witness.ao_mat.to_dense_bool());
        mat_container
            .dense_bool
            .push(self.witness.al_mat.to_dense_bool());

        mat_container
            .dense_block_i32
            .push(self.witness.block_w.clone());

        mat_container
            .dense_block_bool
            .push(self.witness.if_invalid_rotation.clone());

        for i in 0..self.k {
            mat_container
                .dense_block_bool
                .push(self.witness.booleans_rotation[i].clone());
        }

        mat_container
    }

    pub fn gen_mock(num_mat: usize, n: usize, k: usize) -> Self {
        let mut mock = MockMultiLookups::new(num_mat, n, k);

        let timer = std::time::Instant::now();

        mock.gen_mock();

        let duration = timer.elapsed().as_secs_f64();

        println!("Gen Mat Time: {:?} s", duration);

        let log_n = n.ilog2() as usize;

        if n & (n - 1) != 0 {
            panic!("n not a power of 2");
        }

        if num_mat & (num_mat - 1) != 0 {
            panic!("num not a power of 2");
        }

        let log_num = num_mat.ilog2() as usize;
        let mut half_log_num = log_num / 2;

        if log_num != 2 * half_log_num {
            half_log_num += 1;
            // panic!("num not a square number, not supported");
        }

        let mut log_q = half_log_num + log_n;
        if log_q < log_num {
            log_q = log_num;
        }
        if log_q < k / 2 {
            log_q = k / 2;
        }

        let max_dim_for_q = 1 << log_q;
        let large_dim = n * num_mat;
        let large_large_dim = n * n * num_mat;

        // println!("mock.booleans {:?}", mock.booleans_rotation.len());

        Self {
            num_mat: num_mat,
            n: n,
            k: k,
            large_dim: large_dim,
            large_large_dim: large_large_dim,
            max_dim_for_q: max_dim_for_q,
            witness: mock,
            mat_container: MatContainer::new(k),
            com_container: ComContainer::new(),
            eval_container: EvalContainer::new(),
            points_container: PointsContainer::new(),
        }
    }

    pub fn setup(&self) -> SRS<E> {
        let timer = std::time::Instant::now();

        let log_q = self.max_dim_for_q.ilog2() as usize;

        let srs = SRS::<E>::setup(log_q);

        let duration = timer.elapsed().as_secs_f64();

        println!("Setup time: {:?} s", duration);

        let mut writer = Vec::new();
        srs.serialize_compressed(&mut writer).unwrap();
        let srs_size = writer.len();

        println!("SRS size: {:?} B", srs_size);
        println!("\n ********************************\n");

        srs
    }

    pub fn commit_mats(&mut self, srs: &SRS<E>) {
        self.mat_container = self.mat_container_from_witness();

        println!("Committing matrices");

        batchpcs::commit_mats(&srs.pc_pp, &self.mat_container, &mut self.com_container);

        self.mat_container = MatContainer::new(self.k);
    }

    pub fn open_pc_mats(
        &mut self,
        srs: &SRS<E>,
        zk_trans_seq: &mut ZkTranSeq<E::ScalarField, E::G1>,
    ) -> PcsTrans<E> {
        self.mat_container = self.mat_container_from_witness();

        self.witness = MockMultiLookups::new(self.num_mat, self.n, self.k);

        let pc_proof = batchpcs::open_pc_mats(
            &srs.zksrs,
            &srs.pc_pp,
            zk_trans_seq,
            &mut self.mat_container,
            &mut self.com_container,
            &mut self.eval_container,
            &mut self.points_container,
        );

        self.mat_container = MatContainer::new(self.k);

        pc_proof
    }

    pub fn verify_pc_mats(
        &mut self,
        srs: &SRS<E>,
        trans_seq: &mut TranSeq<E::ScalarField, E::G1>,
        pc_proof: &PcsTrans<E>,
    ) -> bool {
        batchpcs::verify_pc_mats(
            &srs.zksrs,
            &srs.pc_pp,
            trans_seq,
            pc_proof,
            &mut self.com_container,
            &mut self.eval_container,
            &mut self.points_container,
            3,
        )
    }

    pub fn prove(
        &mut self,
        srs: &SRS<E>,
        zk_trans_seq: &mut ZkTranSeq<E::ScalarField, E::G1>,
    ) -> (ComContainer<E>, PcsTrans<E>) {
        let start_timer = std::time::Instant::now();

        let commit_timer = std::time::Instant::now();

        self.commit_mats(srs);
        let com_container = self.com_container.clone();

        let commit_duration = commit_timer.elapsed().as_secs_f64();

        let reduce_timer = std::time::Instant::now();

        self.reduce_prover(srs, zk_trans_seq);

        let reduce_duration = reduce_timer.elapsed().as_secs_f64();

        let open_timer = std::time::Instant::now();

        let pc_proof = self.open_pc_mats(srs, zk_trans_seq);

        let open_duration = open_timer.elapsed().as_secs_f64();

        let duration = start_timer.elapsed().as_secs_f64();
        println!("\n ********************************\n");
        println!("Prover time: {:?} s", duration);
        println!("*** Commit time: {:?} s", commit_duration);
        println!("*** Reduce time: {:?} s", reduce_duration);
        println!("*** Open time: {:?} s", open_duration);
        println!("\n ********************************\n");

        (com_container, pc_proof)
    }

    pub fn verify(
        &mut self,
        srs: &SRS<E>,
        trans_seq: &mut TranSeq<E::ScalarField, E::G1>,
        com_container: &ComContainer<E>,
        pc_proof: &PcsTrans<E>,
    ) -> bool {
        let start_timer = std::time::Instant::now();
        self.com_container = com_container.clone();

        let reduce_ok = self.verify_reduce(srs, trans_seq);

        let pcs_ok = self.verify_pc_mats(srs, trans_seq, pc_proof);

        let fiat_shamir_ok = trans_seq.check_fiat_shamir();
        let transcript_consumed = trans_seq.is_fully_consumed();

        let duration = start_timer.elapsed().as_secs_f64();

        println!("\n ********************************\n");
        println!("Verifier time: {:?} ms\n", duration * 1000.0);

        let mut com_writer = Vec::new();
        com_container.flattened[0]
            .0
            .serialize_compressed(&mut com_writer)
            .unwrap();
        let com_size = com_writer.len() * 2;
        println!("Commitment size: {:?} B\n", com_size);

        let mut writer = Vec::new();
        for com in com_container.flattened.iter() {
            com.0.serialize_compressed(&mut writer).unwrap();
        }
        pc_proof.serialize_compressed(&mut writer).unwrap();
        trans_seq.serialize_compressed(&mut writer).unwrap();
        let proof_size = writer.len();

        println!("Proof size: {:?} B", proof_size);
        println!("\n ********************************\n");

        reduce_ok && pcs_ok && fiat_shamir_ok && transcript_consumed
    }

    pub fn reduce_prover(
        &mut self,
        srs: &SRS<E>,
        zk_trans_seq: &mut ZkTranSeq<E::ScalarField, E::G1>,
    ) {
        let n = self.n;
        let k = self.k;
        let large_dim = self.large_dim;

        let log_large_dim = large_dim.ilog2() as usize;
        let log_n = n.ilog2() as usize;

        // ****************************************************
        // Outside projection

        println!("Proving outside projection");

        let mut xl1 = Vec::new();
        let mut xr1 = Vec::new();

        for _ in 0..log_large_dim {
            let cur_x = zk_trans_seq.gen_challenge();
            xl1.push(cur_x);
        }

        for _ in 0..log_n {
            let cur_x = zk_trans_seq.gen_challenge();
            xr1.push(cur_x);
        }

        let xx1 = [xr1.clone(), xl1.clone()].concat();

        let proj_o = self.witness.block_mat_o.proj_lr(&xl1.clone(), &xr1.clone());

        let (proj_o_com, proj_o_tilde) = zk_trans_seq.push_gen_blinding(proj_o);

        let proj_l = self.witness.block_mat_l.proj_lr(&xl1.clone(), &xr1.clone());

        let (proj_l_com, proj_l_tilde) = zk_trans_seq.push_gen_blinding(proj_l);

        // ****************************************************
        // Mul with rotation matrix on both side
        // xx^T (b k) = block_l
        // xx^T (b v) = block_o

        println!("Proving matmul with rotation matrix");

        let mut value_mat = DenseMatCM::<i32, E::ScalarField>::new(1 << k, 1);
        value_mat.set_data(vec![self.witness.value_vec.clone()]);

        let mul_o = ZkMatMul::<
            i32,
            E::ScalarField,
            E::G1,
            RotationMatIndexFormat<i32, E::ScalarField>,
            DenseMatCM<i32, E::ScalarField>,
        >::new(proj_o_com, (xx1.clone(), Vec::new()), 1 << k);

        println!("- Proving O Lookup");

        let (
            v_rotation_com_o,
            v_value_com,
            v_rotation_tilde_o,
            v_value_tilde,
            (xl2, xr2),
            (xx_value, _),
        ) = mul_o.reduce_prover(
            &srs.zksrs,
            zk_trans_seq,
            &self.witness.rotation_mat,
            &value_mat,
            proj_o_tilde,
        );

        let mut key_mat = DenseMatCM::<i32, E::ScalarField>::new(1 << k, 1);
        key_mat.set_data(vec![self.witness.key_vec.clone()]);

        println!("- Proving L Lookup");

        let mul_l = ZkMatMul::<
            i32,
            E::ScalarField,
            E::G1,
            RotationMatIndexFormat<i32, E::ScalarField>,
            DenseMatCM<i32, E::ScalarField>,
        >::new(proj_l_com, (xx1.clone(), Vec::new()), 1 << k);

        let (v_rotation_com_l, v_key_com, v_rotation_tilde_l, v_key_tilde, (xl3, xr3), (xx_key, _)) =
            mul_l.reduce_prover(
                &srs.zksrs,
                zk_trans_seq,
                &self.witness.rotation_mat,
                &key_mat,
                proj_l_tilde,
            );

        // ****************************************************
        // Batch points for rotation matrix
        //

        println!("Proving batch points for rotation matrix");

        let mut batch_points_rotation = BatchPointsRotation::new();
        batch_points_rotation.push_point(v_rotation_com_o, (xl2, xr2));
        batch_points_rotation.push_point(v_rotation_com_l, (xl3, xr3));

        let (v_rotation_com, v_rotation_tilde, (xl_rotation, xr_rotation)) = batch_points_rotation
            .reduce_prover(
                &srs.zksrs,
                zk_trans_seq,
                &self.witness.rotation_mat,
                vec![v_rotation_tilde_o, v_rotation_tilde_l],
            );

        // ****************************************************
        // Prove matmul for a_L w, a_O w

        println!("Proving matmul for a_L w, a_O w");

        let mut identity_mat_data = vec![vec![0; self.n]; self.n];
        for i in 0..self.n {
            identity_mat_data[i][i] = 1;
        }

        let mut identity_mat = DenseMatCM::<i32, E::ScalarField>::new(self.n, self.n);
        identity_mat.set_data(identity_mat_data.clone());

        let mut al_kronecker = SparseKronecker::<i32, E::ScalarField>::new(
            self.witness.al_mat.get_shape(),
            (self.n, self.n),
        );
        al_kronecker.set_data((
            self.witness.al_mat.to_dense().data,
            identity_mat_data.clone(),
        ));

        let mut ao_kronecker = SparseKronecker::<i32, E::ScalarField>::new(
            self.witness.ao_mat.get_shape(),
            (self.n, self.n),
        );
        ao_kronecker.set_data((
            self.witness.ao_mat.to_dense().data,
            identity_mat_data.clone(),
        ));

        let matmul_o = ZkMatMul::new(proj_o_com, (xl1.clone(), xr1.clone()), large_dim);

        let (
            v_ao_kronecker_com,
            v_w_com_1,
            v_ao_kronecker_tilde,
            v_w_tilde_1,
            ao_kronecker_point,
            w_point_1,
        ) = matmul_o.reduce_prover(
            &srs.zksrs,
            zk_trans_seq,
            &ao_kronecker,
            &self.witness.block_w,
            proj_o_tilde,
        );

        let matmul_l = ZkMatMul::new(proj_l_com, (xl1.clone(), xr1.clone()), large_dim);

        let (
            v_al_kronecker_com,
            v_w_com_2,
            v_al_kronecker_tilde,
            v_w_tilde_2,
            al_kronecker_point,
            w_point2,
        ) = matmul_l.reduce_prover(
            &srs.zksrs,
            zk_trans_seq,
            &al_kronecker,
            &self.witness.block_w,
            proj_l_tilde,
        );

        // ****************************************************
        // Reduce the two Kronecker products

        println!("Reducing the two Kronecker products");

        let kronecker_o = ZkKronecker::new(
            v_ao_kronecker_com,
            ao_kronecker_point,
            (self.num_mat, self.num_mat),
        );

        let (v_ao_com, _, v_ao_tilde, v_identity_tilde1, (xl7, xr7), _) = kronecker_o
            .reduce_prover(
                &srs.zksrs,
                zk_trans_seq,
                &self.witness.ao_mat,
                &identity_mat,
                v_ao_kronecker_tilde,
            );

        zk_trans_seq.push_without_blinding(v_identity_tilde1);

        let kronecker_l = ZkKronecker::new(
            v_al_kronecker_com,
            al_kronecker_point,
            (self.num_mat, self.num_mat),
        );

        let (v_al_com, _, v_al_tilde, v_identity_tilde2, (xl8, xr8), _) = kronecker_l
            .reduce_prover(
                &srs.zksrs,
                zk_trans_seq,
                &self.witness.al_mat,
                &identity_mat,
                v_al_kronecker_tilde,
            );

        zk_trans_seq.push_without_blinding(v_identity_tilde2);

        // ****************************************************
        // Batch points for block_w

        println!("Proving batch points for block_w");

        let mut batch_points_w = BatchPoints::new();

        batch_points_w.push_point(v_w_com_1, w_point_1);
        batch_points_w.push_point(v_w_com_2, w_point2);

        let (v_w_com, v_w_tilde, (xl10, xr10)) = batch_points_w.reduce_prover(
            &srs.zksrs,
            zk_trans_seq,
            &self.witness.block_w,
            vec![v_w_tilde_1, v_w_tilde_2],
        );

        // ****************************************************
        // Rotation proj

        println!("Proving rotation proj");

        let rotation_proj = ZkRotationProj::new(v_rotation_com, (xl_rotation, xr_rotation));

        let (
            v_if_invalid_com,
            v_booleans_com_vec,
            v_if_invalid_tilde,
            v_booleans_tilde_vec,
            point_if_invalid,
            point_booleans_vec,
        ) = rotation_proj.reduce_prover(
            &srs.zksrs,
            zk_trans_seq,
            &self.witness.rotation_mat,
            &self.witness.if_invalid_rotation.to_vec(),
            v_rotation_tilde,
        );

        // println!("Rotation mat shape {:?}", self.witness.rotation_mat.get_shape());

        // ****************************************************
        // Add evals and points into the containers

        println!("Adding evals and points into the containers");

        self.eval_container.dense_i32.push((v_key_com, v_key_tilde));

        self.eval_container
            .dense_i32
            .push((v_value_com, v_value_tilde));

        self.eval_container.dense_bool.push((v_ao_com, v_ao_tilde));
        self.eval_container.dense_bool.push((v_al_com, v_al_tilde));

        self.eval_container
            .dense_block_i32
            .push((v_w_com, v_w_tilde));

        self.eval_container
            .dense_block_bool
            .push((v_if_invalid_com, v_if_invalid_tilde));

        for i in 0..k {
            self.eval_container.dense_block_bool.push((
                v_booleans_com_vec[i].clone(),
                v_booleans_tilde_vec[i].clone(),
            ));
        }

        self.points_container
            .dense_i32
            .push((xx_key.clone(), Vec::new()));

        // assert_eq!(
        //     srs.zksrs.commit(
        //         key_mat.proj_lr(&xx_key, &Vec::new()),
        //         v_key_tilde,
        //     ),
        //     v_key_com
        // );

        self.points_container
            .dense_i32
            .push((xx_value.clone(), Vec::new()));

        // assert_eq!(
        //     srs.zksrs.commit(
        //         value_mat.proj_lr(&xx_value, &Vec::new()),
        //         v_value_tilde,
        //     ),
        //     v_value_com
        // );

        self.points_container
            .dense_bool
            .push((xl7.clone(), xr7.clone()));

        // assert_eq!(
        //     srs.zksrs.commit(
        //         self.witness.ao_mat.proj_lr(&xl7, &xr7),
        //         v_ao_tilde,
        //     ),
        //     v_ao_com
        // );

        self.points_container
            .dense_bool
            .push((xl8.clone(), xr8.clone()));

        // assert_eq!(
        //     srs.zksrs.commit(
        //         self.witness.al_mat.proj_lr(&xl8, &xr8),
        //         v_al_tilde,
        //     ),
        //     v_al_com
        // );

        self.points_container
            .dense_block_i32
            .push((xl10.clone(), xr10.clone()));

        // assert_eq!(
        //     srs.zksrs.commit(
        //         self.witness.block_w.proj_lr(&xl10, &xr10),
        //         v_w_tilde,
        //     ),
        //     v_w_com
        // );

        self.points_container.dense_block_bool.push((
            point_if_invalid[log_n..].to_vec(),
            point_if_invalid[..log_n].to_vec(),
        ));

        // assert_eq!(
        //     srs.zksrs.commit(
        //         self.witness.if_invalid_rotation
        //         .proj_lr(
        //             &point_if_invalid[log_n..].to_vec(),
        //             &point_if_invalid[..log_n].to_vec(),
        //         ),
        //         v_if_invalid_tilde,
        //     ),
        //     v_if_invalid_com
        // );

        for i in 0..k {
            let xx_cur = point_booleans_vec[i].clone();
            let xr_cur = xx_cur[..log_n].to_vec();
            let xl_cur = xx_cur[log_n..].to_vec();

            self.points_container
                .dense_block_bool
                .push((xl_cur.clone(), xr_cur.clone()));

            // assert_eq!(
            //     srs.zksrs.commit(
            //         self.witness.booleans_rotation[i]
            //         .proj_lr(&xl_cur, &xr_cur),
            //     v_booleans_tilde_vec[i],
            //     ),
            //     v_booleans_com_vec[i]
            // );
        }
    }

    pub fn verify_reduce(
        &mut self,
        srs: &SRS<E>,
        trans_seq: &mut TranSeq<E::ScalarField, E::G1>,
    ) -> bool {
        let mut flag = true;

        let n = self.n;
        let k = self.k;
        let large_dim = self.large_dim;

        let log_large_dim = large_dim.ilog2() as usize;
        let log_n = n.ilog2() as usize;

        // ****************************************************
        // Outside projection
        let mut xl1 = Vec::new();
        let mut xr1 = Vec::new();

        for _ in 0..log_large_dim {
            if let TranElem::Coin(cur_x) = trans_seq.data[trans_seq.pointer].clone() {
                xl1.push(cur_x);
                trans_seq.pointer += 1;
            } else {
                panic!("Expecting a coin in HDR1CS");
            }
        }

        for _ in 0..log_n {
            if let TranElem::Coin(cur_x) = trans_seq.data[trans_seq.pointer].clone() {
                xr1.push(cur_x);
                trans_seq.pointer += 1;
            } else {
                panic!("Expecting a coin in HDR1CS");
            }
        }

        let xx1 = [xr1.clone(), xl1.clone()].concat();

        let proj_o_com;
        if let TranElem::Group(proj_o_com_value) = trans_seq.data[trans_seq.pointer].clone() {
            trans_seq.pointer += 1;
            proj_o_com = proj_o_com_value;
        } else {
            panic!("Expecting a group element in HDR1CS");
        }

        let proj_l_com;
        if let TranElem::Group(proj_l_com_value) = trans_seq.data[trans_seq.pointer].clone() {
            trans_seq.pointer += 1;
            proj_l_com = proj_l_com_value;
        } else {
            panic!("Expecting a group element in HDR1CS");
        }

        // ****************************************************
        // Mul with rotation matrix on both side
        // xx^T (b k) = block_l
        // xx^T (b v) = block_o

        let mul_o = ZkMatMul::<
            i32,
            E::ScalarField,
            E::G1,
            RotationMatIndexFormat<i32, E::ScalarField>,
            DenseMatCM<i32, E::ScalarField>,
        >::new(proj_o_com, (xx1.clone(), Vec::new()), 1 << k);

        let (flag_cur, v_rotation_com_o, v_value_com, (xl2, xr2), (xx_value, _)) =
            mul_o.verify_as_subprotocol(&srs.zksrs, trans_seq);

        println!("O Lookup: {}", flag_cur);
        flag = flag && flag_cur;

        let mul_l = ZkMatMul::<
            i32,
            E::ScalarField,
            E::G1,
            RotationMatIndexFormat<i32, E::ScalarField>,
            DenseMatCM<i32, E::ScalarField>,
        >::new(proj_l_com, (xx1.clone(), Vec::new()), 1 << k);

        let (flag_cur, v_rotation_com_l, v_key_com, (xl3, xr3), (xx_key, _)) =
            mul_l.verify_as_subprotocol(&srs.zksrs, trans_seq);

        println!("L Lookup: {}", flag_cur);
        flag = flag && flag_cur;

        // ****************************************************
        // Batch points for rotation matrix
        //

        let mut batch_points_rotation = BatchPointsRotation::<i32, E::ScalarField, E::G1>::new();
        batch_points_rotation.push_point(v_rotation_com_o, (xl2, xr2));
        batch_points_rotation.push_point(v_rotation_com_l, (xl3, xr3));

        let (flag_cur, v_rotation_com, (xl_rotation, xr_rotation)) =
            batch_points_rotation.verify_as_subprotocol(&srs.zksrs, trans_seq);

        println!("Batch Rotation proj: {}", flag_cur);
        flag = flag && flag_cur;

        // ****************************************************
        // Prove matmul for a_L w, a_R w, a_O w

        let matmul_o = ZkMatMul::<
            i32,
            E::ScalarField,
            E::G1,
            DiagBlockMat<i32, E::ScalarField>,
            DenseBlockMat<i32, E::ScalarField>,
        >::new(proj_o_com, (xl1.clone(), xr1.clone()), large_dim);

        let (flag_cur, v_ao_kronecker_com, v_w_com_1, ao_kronecker_point, w_point_1) =
            matmul_o.verify_as_subprotocol(&srs.zksrs, trans_seq);

        println!("O Inner Matmul protocol: {}", flag_cur);
        flag = flag && flag_cur;

        let matmul_l = ZkMatMul::<
            i32,
            E::ScalarField,
            E::G1,
            DiagBlockMat<i32, E::ScalarField>,
            DenseBlockMat<i32, E::ScalarField>,
        >::new(proj_l_com, (xl1.clone(), xr1.clone()), large_dim);

        let (flag_cur, v_al_kronecker_com, v_w_com_2, al_kronecker_point, w_point2) =
            matmul_l.verify_as_subprotocol(&srs.zksrs, trans_seq);

        println!("L Inner Matmul protocol: {}", flag_cur);
        flag = flag && flag_cur;

        // ****************************************************
        // Reduce the two Kronecker products

        let kronecker_o = ZkKronecker::<
            i32,
            E::ScalarField,
            E::G1,
            RotationMatIndexFormat<i32, E::ScalarField>,
            DenseMatCM<i32, E::ScalarField>,
        >::new(
            v_ao_kronecker_com,
            ao_kronecker_point,
            (self.num_mat, self.num_mat),
        );

        let (flag_cur, v_ao_com, v_identity_com1, (xl7, xr7), (xl_identity1, xr_identity1)) =
            kronecker_o.verify_as_subprotocol(&srs.zksrs, trans_seq);

        flag = flag && flag_cur;

        if let TranElem::Field(v_identity_tilde1) = trans_seq.data[trans_seq.pointer].clone() {
            trans_seq.pointer += 1;

            let v_identity_check1 = xi::xi_ip_from_challenges(&xl_identity1, &xr_identity1);

            let v_com_check1 = srs.zksrs.commit(v_identity_check1, v_identity_tilde1);

            let flag_check = v_com_check1 == v_identity_com1;

            flag = flag && flag_check;

            println!("O Kronecker protocol: {}", flag_cur && flag_check);
        } else {
            println!(
                "O Kronecker protocol: {:?}",
                trans_seq.data[trans_seq.pointer]
            );
            panic!("Expecting a field element in HDR1CS");
        };

        let kronecker_l = ZkKronecker::<
            i32,
            E::ScalarField,
            E::G1,
            RotationMatIndexFormat<i32, E::ScalarField>,
            DenseMatCM<i32, E::ScalarField>,
        >::new(
            v_al_kronecker_com,
            al_kronecker_point,
            (self.num_mat, self.num_mat),
        );

        let (flag_cur, v_al_com, v_identity_com2, (xl8, xr8), (xl_identity2, xr_identity2)) =
            kronecker_l.verify_as_subprotocol(&srs.zksrs, trans_seq);

        flag = flag && flag_cur;

        if let TranElem::Field(v_identity_tilde2) = trans_seq.data[trans_seq.pointer].clone() {
            trans_seq.pointer += 1;

            let v_identity_check2 = xi::xi_ip_from_challenges(&xl_identity2, &xr_identity2);

            let v_com_check2 = srs.zksrs.commit(v_identity_check2, v_identity_tilde2);

            let flag_check = v_com_check2 == v_identity_com2;

            flag = flag && flag_check;

            println!("L Kronecker protocol: {}", flag_cur && flag_check);
        } else {
            panic!("Expecting a field element in HDR1CS");
        };

        // ****************************************************
        // Batch points for block_w

        let mut batch_points_w = BatchPoints::<i32, E::ScalarField, E::G1>::new();

        batch_points_w.push_point(v_w_com_1, w_point_1);
        batch_points_w.push_point(v_w_com_2, w_point2);

        let (flag_cur, v_w_com, (xl10, xr10)) =
            batch_points_w.verify_as_subprotocol(&srs.zksrs, trans_seq);

        println!("Batch w: {}", flag_cur);
        flag = flag && flag_cur;

        // ****************************************************
        // Rotation proj

        let rotation_proj = ZkRotationProj::<i32, E::ScalarField, E::G1>::new(
            v_rotation_com,
            (xl_rotation, xr_rotation),
        );

        let (flag_cur, v_if_invalid_com, v_booleans_com_vec, point_if_invalid, point_booleans_vec) =
            rotation_proj.verify_as_subprotocol(&srs.zksrs, trans_seq);

        println!("Rotation proj: {}", flag_cur);
        flag = flag && flag_cur;

        // ****************************************************
        // Add evals and points into the containers

        self.eval_container
            .dense_i32
            .push((v_key_com, E::ScalarField::zero()));

        self.eval_container
            .dense_i32
            .push((v_value_com, E::ScalarField::zero()));

        self.eval_container
            .dense_bool
            .push((v_ao_com, E::ScalarField::zero()));
        self.eval_container
            .dense_bool
            .push((v_al_com, E::ScalarField::zero()));

        self.eval_container
            .dense_block_i32
            .push((v_w_com, E::ScalarField::zero()));

        self.eval_container
            .dense_block_bool
            .push((v_if_invalid_com, E::ScalarField::zero()));

        for i in 0..k {
            self.eval_container
                .dense_block_bool
                .push((v_booleans_com_vec[i].clone(), E::ScalarField::zero()));
        }

        self.points_container.dense_i32.push((xx_key, Vec::new()));

        self.points_container.dense_i32.push((xx_value, Vec::new()));

        self.points_container
            .dense_bool
            .push((xl7.clone(), xr7.clone()));

        self.points_container
            .dense_bool
            .push((xl8.clone(), xr8.clone()));

        self.points_container
            .dense_block_i32
            .push((xl10.clone(), xr10.clone()));

        self.points_container.dense_block_bool.push((
            point_if_invalid[log_n..].to_vec(),
            point_if_invalid[..log_n].to_vec(),
        ));

        for i in 0..k {
            let xx_cur = point_booleans_vec[i].clone();
            let xr_cur = xx_cur[..log_n].to_vec();
            let xl_cur = xx_cur[log_n..].to_vec();

            self.points_container
                .dense_block_bool
                .push((xl_cur.clone(), xr_cur.clone()));
        }

        flag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdr1cs_lookup() {
        use ark_ec::pairing::Pairing;

        type E = ark_bls12_381::Bls12_381;
        type F = <E as Pairing>::ScalarField;
        type G = <E as Pairing>::G1;

        let num_mat = 4;
        let n = 2;
        let k = 8;
        let mut protocol_prover = HdR1csLookup::<E>::gen_mock(num_mat, n, k);

        let srs = &protocol_prover.setup();

        let zk_trans_seq = &mut ZkTranSeq::<F, G>::new(&srs.zksrs);

        let (com_container, pc_proof) = protocol_prover.prove(srs, zk_trans_seq);

        let trans_seq = &mut zk_trans_seq.publish_trans();

        let mut protocol_verifier = HdR1csLookup::<E>::new(num_mat, n, k);

        let check = protocol_verifier.verify(srs, trans_seq, &com_container, &pc_proof);

        let mut point_container_v = protocol_verifier.points_container.clone();
        point_container_v.flatten();

        let mut eval_container_v = protocol_verifier.eval_container.clone();
        eval_container_v.flatten();

        assert_eq!(
            protocol_prover.com_container,
            protocol_verifier.com_container
        );
        assert_eq!(protocol_prover.points_container, point_container_v);

        assert_eq!(
            protocol_prover.eval_container.v_coms,
            eval_container_v.v_coms
        );
        assert_eq!(check, true);
    }
}
