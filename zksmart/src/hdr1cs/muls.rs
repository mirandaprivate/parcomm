/// HDR1CS for multiple matrix multiplications
///
use ark_ec::pairing::Pairing;
use ark_serialize::CanonicalSerialize;
use ark_std::Zero;

use ark_poly_commit::smart_pc::data_structures::Trans as PcsTrans;

use crate::data_structures::SRS;

use crate::protocols::mul::ZkMatMul;
use crate::utils::sparse::{
    DenseBlockMat, DenseMatCM, DiagBlockMat, MatOps, RotationMatIndexFormat, SparseKronecker,
};
use crate::utils::xi;
use crate::utils::zktr::{TranElem, TranSeq, ZkTranSeq};

use crate::utils::mockmat::MockMultiMuls;

use crate::protocols::batchpcs::{
    self, BatchPoints, ComContainer, EvalContainer, MatContainer, PointsContainer,
};
use crate::protocols::{clamp::ZkClamp, diagonal::ZkDiagonal, kronecker::ZkKronecker};

pub struct HdR1csMuls<E: Pairing> {
    pub num_mat: usize,
    pub n: usize,
    pub k: usize,
    pub max_bit: usize,
    pub large_dim: usize,
    pub max_dim_for_q: usize,
    pub witness: MockMultiMuls<E::ScalarField>,
    pub mat_container: MatContainer<E::ScalarField>,
    pub com_container: ComContainer<E>,
    pub eval_container: EvalContainer<E::ScalarField, E::G1>,
    pub points_container: PointsContainer<E::ScalarField>,
}

impl<E: Pairing> HdR1csMuls<E> {
    pub fn new(num_mat: usize, n: usize, k: usize) -> Self {
        let mock = MockMultiMuls::new(num_mat, n, k);

        let log_n = n.ilog2() as usize;
        let max_bit = k + log_n;

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

        Self {
            num_mat: num_mat,
            n: n,
            k: k,
            max_bit: max_bit,
            large_dim: large_dim,
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

        mat_container
            .dense_bool
            .push(self.witness.ao_mat.to_dense_bool());
        mat_container
            .dense_bool
            .push(self.witness.al_mat.to_dense_bool());
        mat_container
            .dense_bool
            .push(self.witness.ar_mat.to_dense_bool());

        mat_container
            .dense_block_i32
            .push(self.witness.block_w.clone());

        mat_container
            .dense_block_bool
            .push(self.witness.sign_o.clone());

        for i in 0..self.k {
            mat_container
                .dense_block_bool
                .push(self.witness.booleans_o[i].clone());
            mat_container
                .dense_block_bool
                .push(self.witness.booleans_o_trailing[i].clone());
        }

        for i in 0..self.max_bit {
            mat_container
                .dense_block_bool
                .push(self.witness.booleans_o_res[i].clone());
        }

        mat_container
    }

    pub fn gen_mock(num_mat: usize, n: usize, k: usize) -> Self {
        let mut mock = MockMultiMuls::new(num_mat, n, k);

        let timer = std::time::Instant::now();

        mock.gen_mock();

        let duration = timer.elapsed().as_secs_f64();

        println!("Setup time: {:?} s", duration);

        let log_n = n.ilog2() as usize;
        let max_bit = k + log_n;

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

        let max_dim_for_q = 1 << log_q;
        let large_dim = n * num_mat;

        Self {
            num_mat: num_mat,
            n: n,
            k: k,
            max_bit: max_bit,
            large_dim: large_dim,
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
        println!("Committing matrices");

        self.mat_container = self.mat_container_from_witness();

        batchpcs::commit_mats(&srs.pc_pp, &self.mat_container, &mut self.com_container);

        self.mat_container = MatContainer::new(self.k);
    }

    pub fn open_pc_mats(
        &mut self,
        srs: &SRS<E>,
        zk_trans_seq: &mut ZkTranSeq<E::ScalarField, E::G1>,
    ) -> PcsTrans<E> {
        self.mat_container = self.mat_container_from_witness();

        self.witness = MockMultiMuls::new(self.num_mat, self.n, self.k);

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
            1,
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
        let com_size = com_writer.len() * 3;
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
        let num_mat = self.num_mat;
        let k = self.k;
        let max_bit = self.max_bit;
        let large_dim = self.large_dim;

        let log_large_dim = large_dim.ilog2() as usize;

        // ****************************************************
        // Outside projection
        println!("Reducing outside projection");

        let mut xl1 = Vec::new();
        let mut xr1 = Vec::new();

        for _ in 0..log_large_dim {
            let cur_x = zk_trans_seq.gen_challenge();
            xl1.push(cur_x);
        }

        for _ in 0..log_large_dim {
            let cur_x = zk_trans_seq.gen_challenge();
            xr1.push(cur_x);
        }

        let mut large_diag_o_full = DiagBlockMat::<i32, E::ScalarField>::new(num_mat, (n, n));
        large_diag_o_full.set_data(self.witness.block_mat_o_full.data.clone());

        let v_o_large_full = large_diag_o_full.proj_lr(&xl1, &xr1);

        let (v_o_large_full_com, v_o_large_full_tilde) =
            zk_trans_seq.push_gen_blinding(v_o_large_full);

        // ****************************************************
        // Diagnonal matrix multiplication on LHS

        println!("Diagonal matrix multiplication on LHS");

        let mut large_diag_l = DiagBlockMat::<i32, E::ScalarField>::new(num_mat, (n, n));
        large_diag_l.set_data(self.witness.block_mat_l.data.clone());

        let mut large_diag_r = DiagBlockMat::<i32, E::ScalarField>::new(num_mat, (n, n));
        large_diag_r.set_data(self.witness.block_mat_r.data.clone());

        let mul_protocol = ZkMatMul::<
            i32,
            E::ScalarField,
            E::G1,
            DiagBlockMat<i32, E::ScalarField>,
            DiagBlockMat<i32, E::ScalarField>,
        >::new(v_o_large_full_com, (xl1.clone(), xr1.clone()), large_dim);

        let (
            v_l_large_com,
            v_r_large_com,
            v_l_large_tilde,
            v_r_large_tilde,
            (xl2, xr2),
            (xl3, xr3),
        ) = mul_protocol.reduce_prover(
            &srs.zksrs,
            zk_trans_seq,
            &large_diag_l,
            &large_diag_r,
            v_o_large_full_tilde,
        );

        large_diag_l.data = Vec::new();
        large_diag_r.data = Vec::new();

        // ****************************************************
        // Diagnoal block matrix from dense block matrices on both side
        //
        println!("Reducing diagonal block matrices");

        let diag_o_protocol = ZkDiagonal::<
            i32,
            E::ScalarField,
            E::G1,
            DenseBlockMat<i32, E::ScalarField>,
        >::new(v_o_large_full_com, (xl1, xr1), num_mat);

        let (v_o_full_block_com, v_o_full_block_tilde, (xl4, xr4)) = diag_o_protocol.reduce_prover(
            &srs.zksrs,
            zk_trans_seq,
            &self.witness.block_mat_o_full,
            v_o_large_full_tilde,
        );

        let diag_l_protocol = ZkDiagonal::<
            i32,
            E::ScalarField,
            E::G1,
            DenseBlockMat<i32, E::ScalarField>,
        >::new(v_l_large_com, (xl2, xr2), num_mat);

        let (v_l_block_com, v_l_block_tilde, (xl5, xr5)) = diag_l_protocol.reduce_prover(
            &srs.zksrs,
            zk_trans_seq,
            &self.witness.block_mat_l,
            v_l_large_tilde,
        );

        let diag_r_protocol = ZkDiagonal::<
            i32,
            E::ScalarField,
            E::G1,
            DenseBlockMat<i32, E::ScalarField>,
        >::new(v_r_large_com, (xl3, xr3), num_mat);

        let (v_r_block_com, v_r_block_tilde, (xl6, xr6)) = diag_r_protocol.reduce_prover(
            &srs.zksrs,
            zk_trans_seq,
            &self.witness.block_mat_r,
            v_r_large_tilde,
        );

        // ****************************************************
        // Prove clamp on the RHS

        println!("Proving clamp on the RHS");

        let clamp_protocol = ZkClamp::<
            i32,
            E::ScalarField,
            E::G1,
            DenseBlockMat<i32, E::ScalarField>,
            DenseBlockMat<bool, E::ScalarField>,
        >::new(v_o_full_block_com, (xl4, xr4), self.k, self.max_bit);

        let (
            v_o_clamp_com,
            v_o_clamp_tilde,
            o_clamp_point,
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
        ) = clamp_protocol.reduce_prover(
            &srs.zksrs,
            zk_trans_seq,
            &self.witness.sign_o,
            &mut self.witness.abs_o,
            &mut self.witness.abs_o_res,
            &mut self.witness.abs_o_trailing,
            &self.witness.booleans_o,
            &self.witness.booleans_o_res,
            &self.witness.booleans_o_trailing,
            v_o_full_block_tilde,
        );

        // ****************************************************
        // Prove matmul for a_L w, a_R w, a_O w

        println!("Proving matmul for a_L w, a_R w, a_O w");

        let mut identity_mat_data = vec![vec![0; self.n]; self.n];
        for i in 0..self.n {
            identity_mat_data[i][i] = 1;
        }

        let mut identity_mat = DenseMatCM::<i32, E::ScalarField>::new(self.n, self.n);
        identity_mat.set_data(identity_mat_data.clone());

        let mut ao_kronecker = SparseKronecker::<i32, E::ScalarField>::new(
            self.witness.ao_mat.get_shape(),
            (self.n, self.n),
        );
        ao_kronecker.set_data((
            self.witness.ao_mat.to_dense().data,
            identity_mat_data.clone(),
        ));

        let matmul_o = ZkMatMul::new(v_o_clamp_com, o_clamp_point, large_dim);

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
            v_o_clamp_tilde,
        );

        ao_kronecker.clear();

        let mut al_kronecker = SparseKronecker::<i32, E::ScalarField>::new(
            self.witness.al_mat.get_shape(),
            (self.n, self.n),
        );
        al_kronecker.set_data((
            self.witness.al_mat.to_dense().data,
            identity_mat_data.clone(),
        ));

        let matmul_l = ZkMatMul::new(v_l_block_com, (xl5, xr5), large_dim);

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
            v_l_block_tilde,
        );

        al_kronecker.clear();

        let mut ar_kronecker = SparseKronecker::<i32, E::ScalarField>::new(
            self.witness.ar_mat.get_shape(),
            (self.n, self.n),
        );
        ar_kronecker.set_data((
            self.witness.ar_mat.to_dense().data,
            identity_mat_data.clone(),
        ));

        let matmul_r = ZkMatMul::new(v_r_block_com, (xl6, xr6), large_dim);

        let (
            v_ar_kronecker_com,
            v_w_com_3,
            v_ar_kronecker_tilde,
            v_w_tilde_3,
            ar_kronecker_point,
            w_point3,
        ) = matmul_r.reduce_prover(
            &srs.zksrs,
            zk_trans_seq,
            &ar_kronecker,
            &self.witness.block_w,
            v_r_block_tilde,
        );

        ar_kronecker.clear();

        // ****************************************************
        // Reduce the three Kronecker products

        println!("Reducing the three Kronecker products");

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

        let kronecker_r = ZkKronecker::new(
            v_ar_kronecker_com,
            ar_kronecker_point,
            (self.num_mat, self.num_mat),
        );

        let (v_ar_com, _, v_ar_tilde, v_identity_tilde3, (xl9, xr9), _) = kronecker_r
            .reduce_prover(
                &srs.zksrs,
                zk_trans_seq,
                &self.witness.ar_mat,
                &identity_mat,
                v_ar_kronecker_tilde,
            );

        zk_trans_seq.push_without_blinding(v_identity_tilde3);

        // ****************************************************
        // Batch points for block_w

        println!("Batching points for block_w");

        let mut batch_points_w = BatchPoints::new();

        batch_points_w.push_point(v_w_com_1, w_point_1);
        batch_points_w.push_point(v_w_com_2, w_point2);
        batch_points_w.push_point(v_w_com_3, w_point3);

        let (v_w_com, v_w_tilde, (xl10, xr10)) = batch_points_w.reduce_prover(
            &srs.zksrs,
            zk_trans_seq,
            &self.witness.block_w,
            vec![v_w_tilde_1, v_w_tilde_2, v_w_tilde_3],
        );

        // ****************************************************
        // Add evals and points into the containers

        println!("Adding evals and points into the containers");

        self.eval_container.dense_bool.push((v_ao_com, v_ao_tilde));
        self.eval_container.dense_bool.push((v_al_com, v_al_tilde));
        self.eval_container.dense_bool.push((v_ar_com, v_ar_tilde));

        self.eval_container
            .dense_block_i32
            .push((v_w_com, v_w_tilde));

        self.eval_container
            .dense_block_bool
            .push((v_sign_com, v_sign_tilde));

        for i in 0..k {
            self.eval_container.dense_block_bool.push((
                v_boolean_com_vec_clamp[i].clone(),
                v_boolean_tilde_vec_clamp[i].clone(),
            ));
            self.eval_container.dense_block_bool.push((
                v_boolean_com_vec_trailing[i].clone(),
                v_boolean_tilde_vec_trailing[i].clone(),
            ));
        }

        for i in 0..max_bit {
            self.eval_container.dense_block_bool.push((
                v_boolean_com_vec_res[i].clone(),
                v_boolean_tilde_vec_res[i].clone(),
            ));
        }

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
            .dense_bool
            .push((xl9.clone(), xr9.clone()));

        // assert_eq!(
        //     srs.zksrs.commit(
        //         self.witness.ar_mat.proj_lr(&xl9, &xr9),
        //         v_ar_tilde,
        //     ),
        //     v_ar_com
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

        self.points_container
            .dense_block_bool
            .push(sign_point.clone());

        // assert_eq!(
        //     srs.zksrs.commit(
        //         self.witness.sign_o.proj_lr(&sign_point.0, &sign_point.1),
        //         v_sign_tilde,
        //     ),
        //     v_sign_com
        // );

        for i in 0..k {
            self.points_container
                .dense_block_bool
                .push(point_boolean_vec_clamp[i].clone());

            // assert_eq!(
            //     srs.zksrs.commit(
            //         self.witness.booleans_o[i].proj_lr(&point_boolean_vec_clamp[i].0, &point_boolean_vec_clamp[i].1),
            //     v_boolean_tilde_vec_clamp[i],
            //     ),
            //     v_boolean_com_vec_clamp[i]
            // );

            self.points_container
                .dense_block_bool
                .push(point_boolean_vec_trailing[i].clone());

            // assert_eq!(
            //     srs.zksrs.commit(
            //         self.witness.booleans_o_trailing[i].proj_lr(&point_boolean_vec_trailing[i].0, &point_boolean_vec_trailing[i].1),
            //     v_boolean_tilde_vec_trailing[i],
            //     ),
            //     v_boolean_com_vec_trailing[i]
            // );
        }

        for i in 0..max_bit {
            self.points_container
                .dense_block_bool
                .push(point_boolean_vec_res[i].clone());

            // assert_eq!(
            //     srs.zksrs.commit(
            //         self.witness.booleans_o_res[i].proj_lr(&point_boolean_vec_res[i].0, &point_boolean_vec_res[i].1),
            //     v_boolean_tilde_vec_res[i],
            //     ),
            //     v_boolean_com_vec_res[i]
            // );
        }
    }

    pub fn verify_reduce(
        &mut self,
        srs: &SRS<E>,
        trans_seq: &mut TranSeq<E::ScalarField, E::G1>,
    ) -> bool {
        let mut flag = true;

        let num_mat = self.num_mat;
        let k = self.k;
        let max_bit = self.max_bit;
        let large_dim = self.large_dim;

        let log_large_dim = large_dim.ilog2() as usize;

        // ****************************************************
        // Outside projection
        let mut xl1 = Vec::new();
        let mut xr1 = Vec::new();

        let mut pointer = trans_seq.pointer;

        for _ in 0..log_large_dim {
            if let TranElem::Coin(cur_x) = trans_seq.data[pointer].clone() {
                xl1.push(cur_x);
                pointer += 1;
            } else {
                panic!("Expecting a coin in HDR1CS");
            }
        }

        for _ in 0..log_large_dim {
            if let TranElem::Coin(cur_x) = trans_seq.data[pointer].clone() {
                xr1.push(cur_x);
                pointer += 1;
            } else {
                panic!("Expecting a coin in HDR1CS");
            }
        }

        let v_o_large_full_com;
        if let TranElem::Group(v_o_large_full_com_value) = trans_seq.data[pointer].clone() {
            v_o_large_full_com = v_o_large_full_com_value;
            pointer += 1;
        } else {
            panic!("Expecting a group element in HDR1CS");
        };

        trans_seq.pointer = pointer;

        // ****************************************************
        // Diagnonal matrix multiplication on LHS

        let mul_protocol = ZkMatMul::<
            i32,
            E::ScalarField,
            E::G1,
            DiagBlockMat<i32, E::ScalarField>,
            DiagBlockMat<i32, E::ScalarField>,
        >::new(v_o_large_full_com, (xl1.clone(), xr1.clone()), large_dim);

        let (flag_cur, v_l_large_com, v_r_large_com, (xl2, xr2), (xl3, xr3)) =
            mul_protocol.verify_as_subprotocol(&srs.zksrs, trans_seq);

        // println!("xl2: {:?}", xl2);
        // println!("xr2: {:?}", xr2);
        // println!("xl3: {:?}", xl3);
        // println!("xr3: {:?}", xr3);

        println!("Outer Mul protocol: {}", flag_cur);
        flag = flag && flag_cur;

        // ****************************************************
        // Diagnoal block matrix from dense block matrices on both side
        //
        let diag_o_protocol = ZkDiagonal::<
            i32,
            E::ScalarField,
            E::G1,
            DenseBlockMat<i32, E::ScalarField>,
        >::new(v_o_large_full_com, (xl1, xr1), num_mat);

        let (flag_cur, v_o_full_block_com, (xl4, xr4)) =
            diag_o_protocol.verify_as_subprotocol(&srs.zksrs, trans_seq);

        println!("O Diag protocol: {}", flag_cur);
        flag = flag && flag_cur;

        let diag_l_protocol = ZkDiagonal::<
            i32,
            E::ScalarField,
            E::G1,
            DenseBlockMat<i32, E::ScalarField>,
        >::new(v_l_large_com, (xl2, xr2), num_mat);

        let (flag_cur, v_l_block_com, (xl5, xr5)) =
            diag_l_protocol.verify_as_subprotocol(&srs.zksrs, trans_seq);

        println!("L Diag protocol: {}", flag_cur);
        flag = flag && flag_cur;

        let diag_r_protocol = ZkDiagonal::<
            i32,
            E::ScalarField,
            E::G1,
            DenseBlockMat<i32, E::ScalarField>,
        >::new(v_r_large_com, (xl3, xr3), num_mat);

        let (flag_cur, v_r_block_com, (xl6, xr6)) =
            diag_r_protocol.verify_as_subprotocol(&srs.zksrs, trans_seq);

        println!("R Diag protocol: {}", flag_cur);
        flag = flag && flag_cur;

        // ****************************************************
        // Prove clamp on the RHS

        let clamp_protocol = ZkClamp::<
            i32,
            E::ScalarField,
            E::G1,
            DenseBlockMat<i32, E::ScalarField>,
            DenseBlockMat<bool, E::ScalarField>,
        >::new(v_o_full_block_com, (xl4, xr4), self.k, self.max_bit);

        let (
            flag_cur,
            v_o_clamp_com,
            o_clamp_point,
            v_sign_com,
            sign_point,
            v_boolean_com_vec_clamp,
            point_boolean_vec_clamp,
            v_boolean_com_vec_res,
            point_boolean_vec_res,
            v_boolean_com_vec_trailing,
            point_boolean_vec_trailing,
        ) = clamp_protocol.verify_as_subprotocol(&srs.zksrs, trans_seq);

        println!("Clamp protocol: {}", flag_cur);
        flag = flag && flag_cur;

        // ****************************************************
        // Prove matmul for a_L w, a_R w, a_O w

        let matmul_o = ZkMatMul::<
            i32,
            E::ScalarField,
            E::G1,
            DiagBlockMat<i32, E::ScalarField>,
            DenseBlockMat<i32, E::ScalarField>,
        >::new(v_o_clamp_com, o_clamp_point, large_dim);

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
        >::new(v_l_block_com, (xl5, xr5), large_dim);

        let (flag_cur, v_al_kronecker_com, v_w_com_2, al_kronecker_point, w_point2) =
            matmul_l.verify_as_subprotocol(&srs.zksrs, trans_seq);

        println!("L Inner Matmul protocol: {}", flag_cur);
        flag = flag && flag_cur;

        let matmul_r = ZkMatMul::<
            i32,
            E::ScalarField,
            E::G1,
            DiagBlockMat<i32, E::ScalarField>,
            DenseBlockMat<i32, E::ScalarField>,
        >::new(v_r_block_com, (xl6, xr6), large_dim);

        let (flag_cur, v_ar_kronecker_com, v_w_com_3, ar_kronecker_point, w_point3) =
            matmul_r.verify_as_subprotocol(&srs.zksrs, trans_seq);

        println!("R Inner Matmul protocol: {}", flag_cur);
        flag = flag && flag_cur;

        // ****************************************************
        // Reduce the three Kronecker products

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
            println!("O Kronecker protocol: {:?}", trans_seq.data[pointer]);
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

        let kronecker_r = ZkKronecker::<
            i32,
            E::ScalarField,
            E::G1,
            RotationMatIndexFormat<i32, E::ScalarField>,
            DenseMatCM<i32, E::ScalarField>,
        >::new(
            v_ar_kronecker_com,
            ar_kronecker_point,
            (self.num_mat, self.num_mat),
        );

        let (flag_cur, v_ar_com, v_identity_com3, (xl9, xr9), (xl_identity3, xr_identity3)) =
            kronecker_r.verify_as_subprotocol(&srs.zksrs, trans_seq);

        flag = flag && flag_cur;

        if let TranElem::Field(v_identity_tilde3) = trans_seq.data[trans_seq.pointer].clone() {
            trans_seq.pointer += 1;

            let v_identity_check3 = xi::xi_ip_from_challenges(&xl_identity3, &xr_identity3);

            let v_com_check3 = srs.zksrs.commit(v_identity_check3, v_identity_tilde3);

            let flag_check = v_com_check3 == v_identity_com3;

            flag = flag && flag_check;

            println!("R Kronecker protocol: {}", flag_cur && flag_check);
        } else {
            panic!("Expecting a field element in HDR1CS");
        };

        // ****************************************************
        // Batch points for block_w

        let mut batch_points_w = BatchPoints::<i32, E::ScalarField, E::G1>::new();

        batch_points_w.push_point(v_w_com_1, w_point_1);
        batch_points_w.push_point(v_w_com_2, w_point2);
        batch_points_w.push_point(v_w_com_3, w_point3);

        let (flag_cur, v_w_com, (xl10, xr10)) =
            batch_points_w.verify_as_subprotocol(&srs.zksrs, trans_seq);

        println!("Batch w: {}", flag_cur);
        flag = flag && flag_cur;

        // ****************************************************
        // Add evals and points into the containers

        self.eval_container
            .dense_bool
            .push((v_ao_com, E::ScalarField::zero()));
        self.eval_container
            .dense_bool
            .push((v_al_com, E::ScalarField::zero()));
        self.eval_container
            .dense_bool
            .push((v_ar_com, E::ScalarField::zero()));

        self.eval_container
            .dense_block_i32
            .push((v_w_com, E::ScalarField::zero()));

        self.eval_container
            .dense_block_bool
            .push((v_sign_com, E::ScalarField::zero()));

        for i in 0..k {
            self.eval_container
                .dense_block_bool
                .push((v_boolean_com_vec_clamp[i].clone(), E::ScalarField::zero()));
            self.eval_container.dense_block_bool.push((
                v_boolean_com_vec_trailing[i].clone(),
                E::ScalarField::zero(),
            ));
        }

        for i in 0..max_bit {
            self.eval_container
                .dense_block_bool
                .push((v_boolean_com_vec_res[i].clone(), E::ScalarField::zero()));
        }

        self.points_container.dense_bool.push((xl7, xr7));
        self.points_container.dense_bool.push((xl8, xr8));
        self.points_container.dense_bool.push((xl9, xr9));

        self.points_container.dense_block_i32.push((xl10, xr10));

        self.points_container.dense_block_bool.push(sign_point);

        for i in 0..k {
            self.points_container
                .dense_block_bool
                .push(point_boolean_vec_clamp[i].clone());
            self.points_container
                .dense_block_bool
                .push(point_boolean_vec_trailing[i].clone());
        }

        for i in 0..max_bit {
            self.points_container
                .dense_block_bool
                .push(point_boolean_vec_res[i].clone());
        }

        flag
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hdr1cs_mul() {
        use ark_ec::pairing::Pairing;

        type E = ark_bls12_381::Bls12_381;
        type F = <E as Pairing>::ScalarField;
        type G = <E as Pairing>::G1;

        let num_mat = 64;
        let n = 64;
        let k = 8;
        let mut protocol_prover = HdR1csMuls::<E>::gen_mock(num_mat, n, k);

        let srs = &protocol_prover.setup();

        let zk_trans_seq = &mut ZkTranSeq::<F, G>::new(&srs.zksrs);

        let (com_container, pc_proof) = protocol_prover.prove(srs, zk_trans_seq);

        let trans_seq = &mut zk_trans_seq.publish_trans();

        let mut protocol_verifier = HdR1csMuls::<E>::new(num_mat, n, k);

        let check = protocol_verifier.verify(srs, trans_seq, &com_container, &pc_proof);

        assert_eq!(check, true);

        let mut transcript_with_trailing_data = zk_trans_seq.publish_trans();
        transcript_with_trailing_data
            .data
            .push(TranElem::Field(F::from(123_u64)));
        let mut protocol_verifier = HdR1csMuls::<E>::new(num_mat, n, k);

        let check = protocol_verifier.verify(
            srs,
            &mut transcript_with_trailing_data,
            &com_container,
            &pc_proof,
        );

        assert_eq!(check, false);
    }
}
