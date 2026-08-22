use ark_crypto_primitives::sponge::Absorb;
use ark_ec::pairing::{Pairing, PairingOutput};
/// Implement the BatchPCS
use ark_ff::{One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress};
use ark_std::UniformRand;
use std::ops::Mul;
use std::time::Instant;

use fs::helper_trans::Transcript;

use poly_commit::smart_pc::{SmartPC as PC, Trans as PCTrans, UniversalParams as PP};
use poly_commit::MyInt;

use super::bva::{BvaLayout, BVA};
use super::litebullet_eo::LiteBulletEO;
use super::tbv::{Bucket, TBV};
use super::tbvproof::{FieldMatCommitter, TbvReduceProof};

#[derive(Clone)]
pub struct BatchPCSTrans<E: Pairing>
where
    E::ScalarField: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
    E::G1: CanonicalSerialize + CanonicalDeserialize,
    E::G2: CanonicalSerialize + CanonicalDeserialize,
    PairingOutput<E>: CanonicalSerialize + CanonicalDeserialize,
{
    pub trans_reduce: Transcript<<E as Pairing>::ScalarField>,
    pub trans_pc: PCTrans<E>,
    pub trans_pc_i: PCTrans<E>,
    pub trans_pc_z: PCTrans<E>,
    pub trans_pc_w: PCTrans<E>,

    pub mat_w_com: PairingOutput<E>,
}

impl<E: Pairing> BatchPCSTrans<E>
where
    E::ScalarField: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
    E::G1: CanonicalSerialize + CanonicalDeserialize,
    E::G2: CanonicalSerialize + CanonicalDeserialize,
    PairingOutput<E>: CanonicalSerialize + CanonicalDeserialize,
{
    pub fn new() -> Self {
        let mut rng = ark_std::test_rng();
        let trans_reduce = Transcript::new(E::ScalarField::rand(&mut rng));
        let trans_pc = PCTrans::new();
        let trans_pc_i = PCTrans::new();
        let trans_pc_z = PCTrans::new();
        let trans_pc_w = PCTrans::new();

        Self {
            trans_reduce,
            trans_pc,
            trans_pc_i,
            trans_pc_z,
            trans_pc_w,

            mat_w_com: PairingOutput::<E>::zero(),
        }
    }

    pub fn reset_pointer(&mut self) {
        self.trans_reduce.reset_pointer();
    }

    /// Return serialized size of all transcripts/commits in bytes (uncompressed).
    pub fn get_proof_size(&self) -> usize {
        let trans_bytes = self.trans_reduce.get_proof_size_in_bytes();
        let pc_bytes = self.trans_pc.serialized_size(Compress::No);
        let pc_i_bytes = self.trans_pc_i.serialized_size(Compress::No);
        let pc_z_bytes = self.trans_pc_z.serialized_size(Compress::No);
        let pc_w_bytes = self.trans_pc_w.serialized_size(Compress::No);
        let mat_w_bytes = self.mat_w_com.serialized_size(Compress::No);

        println!("trans_bytes={}, pc_bytes={}, pc_i_bytes={}, pc_z_bytes={}, pc_w_bytes={}, mat_w_bytes={}", trans_bytes, pc_bytes, pc_i_bytes, pc_z_bytes, pc_w_bytes, mat_w_bytes);

        trans_bytes + pc_bytes + pc_i_bytes + pc_z_bytes + pc_w_bytes + mat_w_bytes
    }
}

pub struct BatchPCS<E: Pairing>
where
    E::ScalarField: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
    E::G1: CanonicalSerialize + CanonicalDeserialize,
    E::G2: CanonicalSerialize + CanonicalDeserialize,
    PairingOutput<E>: CanonicalSerialize + CanonicalDeserialize,
{
    pub pc: PC<E>,
    pub pp: PP<E>,
    pub prover_input_bva: BVA,
}

impl<E: Pairing> BatchPCS<E>
where
    E::ScalarField: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
    E::G1: CanonicalSerialize + CanonicalDeserialize,
    E::G2: CanonicalSerialize + CanonicalDeserialize,
    PairingOutput<E>: CanonicalSerialize + CanonicalDeserialize,
{
    pub fn new(pc: PC<E>, pp: PP<E>) -> Self {
        Self {
            pc,
            pp,
            // Initialize placeholder to avoid empty shapes triggering BVA assertion; must call set_prover_input before actual use
            prover_input_bva: BVA::new(vec![vec![0 as MyInt]]),
        }
    }

    pub fn set_prover_input(&mut self, vecs: Vec<Vec<MyInt>>) {
        self.prover_input_bva = BVA::new(vecs);
    }

    pub fn setup(log_max_n: usize) -> Self {
        let mut rng = ark_std::test_rng();
        // qlog is log2 of one side of the square matrix
        let qlog = (log_max_n + 2) / 2;
        let pp = PC::<E>::setup(qlog, &mut rng).unwrap();
        let pc = PC::<E>::new();
        Self::new(pc, pp)
    }

    pub fn commit(
        &self,
    ) -> (
        PairingOutput<E>,
        PairingOutput<E>,
        Vec<<E as Pairing>::G1>,
        Vec<<E as Pairing>::G1>,
    ) {
        let timer_commit_a = Instant::now();
        let layout = self.prover_input_bva.layout.clone();

        // ===========================================
        let (com_a, tier1_a) = {
            let expand = self.prover_input_bva.expand_to_myint_vec();
            let hiding_factor = <E as Pairing>::ScalarField::zero();
            let res = PC::<E>::commit_square_myint_flat(&self.pp, &expand, hiding_factor).unwrap();
            res
        }; // expand is dropped here

        let commit_a_time = timer_commit_a.elapsed().as_secs_f64();
        println!("===> Commit a time: {:.3}ms", commit_a_time * 1000.0);

        let timer_commit_i = Instant::now();
        let committer = FieldMatCommitter::<E>::new(self.pp.clone());

        let mat_i = layout.compute_mat_i();

        let (com_i, tier1_i) = committer.commit_mat(&mat_i);

        // ========================================
        // let com_a = com_i.clone(); let tier1_a = tier1_i.clone();

        let commit_i_time = timer_commit_i.elapsed().as_secs_f64();
        println!("===> Commit i time: {:.3}ms", commit_i_time * 1000.0);

        (com_a, com_i, tier1_a, tier1_i)
    }

    pub fn commit_points(
        &self,
        points: &Vec<Vec<E::ScalarField>>,
    ) -> (PairingOutput<E>, Vec<<E as Pairing>::G1>) {
        let timer_commit_z = Instant::now();
        let committer = FieldMatCommitter::<E>::new(self.pp.clone());
        let mat_z = self.prover_input_bva.layout.compute_mat_z(points);
        let (com_z, tier1_z) = committer.commit_mat(&mat_z);
        let commit_z_time = timer_commit_z.elapsed().as_secs_f64();
        println!("===> Commit z time: {:.3}ms", commit_z_time * 1000.0);
        (com_z, tier1_z)
    }

    pub fn gen_tbv_from_values_and_points(
        &self,
        values: &[E::ScalarField],
        points: &[Vec<E::ScalarField>],
        omega: E::ScalarField,
        layout: &BvaLayout,
    ) -> (E::ScalarField, TBV<E::ScalarField>) {
        let num_points = points.len();
        let powers = std::iter::successors(Some(E::ScalarField::one()), |&x| Some(x * omega))
            .take(num_points)
            .collect::<Vec<_>>();

        let batch_values = values
            .iter()
            .zip(powers.iter())
            .map(|(v, p)| *v * *p)
            .sum::<E::ScalarField>();

        let mut buckets = Vec::new();
        for (i, &k) in layout.bucket_sizes.iter().enumerate() {
            if let Some(orig_idx) = layout.phi_inv[i] {
                let mut z = points[orig_idx].clone();
                z.reverse();
                buckets.push(Bucket {
                    k,
                    eta: powers[orig_idx],
                    z,
                });
            } else {
                // Padding bucket
                buckets.push(Bucket {
                    k,
                    eta: E::ScalarField::zero(),
                    z: vec![E::ScalarField::zero(); k.ilog2() as usize],
                });
            }
        }

        (batch_values, TBV::new(layout.k_total, buckets))
    }

    pub fn open(
        &self,
        trans: &mut BatchPCSTrans<E>,
        values: &[E::ScalarField],
        points: &[Vec<E::ScalarField>],
        a_com: PairingOutput<E>,
        i_com: PairingOutput<E>,
        z_com: PairingOutput<E>,
        a_tier1: Vec<E::G1>,
        i_tier1: Vec<E::G1>,
        z_tier1: Vec<E::G1>,
    ) {
        let timer_open_total = Instant::now();
        let points_vec = points.to_vec();

        let layout = self.prover_input_bva.layout.clone();

        let timer_reduce_litebullet = Instant::now();
        // 2. Perform reduction and PC proof
        let omega = trans.trans_reduce.gen_challenge();
        let (batch_values, tbv) =
            self.gen_tbv_from_values_and_points(values, points, omega, &layout);
        let litebulleteo = LiteBulletEO::new(batch_values, tbv.k_total);

        // Execute reduction, first step eo_fold(&self) produces new FieldBuff without destroying original witness
        let (a_hat, b_hat, a_point, y_vec) =
            litebulleteo.reduce_prover(&mut trans.trans_reduce, &self.prover_input_bva, tbv);
        let reduce_litebullet_time = timer_reduce_litebullet.elapsed().as_secs_f64();
        println!(
            "===> Reduce litebullet time: {:.3}ms",
            reduce_litebullet_time * 1000.0
        );

        let timer_compute_mat_izw = Instant::now();
        let mat_i = layout.compute_mat_i();
        let mat_z = layout.compute_mat_z(&points_vec);
        let mat_x = layout.compute_mat_x(&y_vec, &mat_i, &mat_z);
        let mat_w = layout.compute_mat_w(&mat_x);
        let compute_mat_izw_time = timer_compute_mat_izw.elapsed().as_secs_f64();
        println!(
            "===> Compute mat izw time: {:.3}ms",
            compute_mat_izw_time * 1000.0
        );

        let timer_commit_w = Instant::now();
        let committer = FieldMatCommitter::<E>::new(self.pp.clone());
        let (w_com, tier1_w) = committer.commit_mat(&mat_w);
        trans.mat_w_com = w_com.clone();
        let commit_w_time = timer_commit_w.elapsed().as_secs_f64();
        println!("===> Commit w time: {:.3}ms", commit_w_time * 1000.0);

        let timer_reduce_izw = Instant::now();
        let (i_hat, z_hat, w_hat, i_points, z_points, w_points) = layout.reduce_prover(
            &mut trans.trans_reduce,
            b_hat.clone(),
            &points_vec,
            omega,
            &y_vec,
        );
        let reduce_izw_time = timer_reduce_izw.elapsed().as_secs_f64();
        println!("===> Reduce TBV time: {:.3}ms", reduce_izw_time * 1000.0);

        // Use original copy for PC proof
        let timer_open_pc = Instant::now();
        self.open_pc(
            trans,
            a_hat,
            a_point.clone(),
            &layout,
            &self.prover_input_bva.vecs,
            a_com,
            a_tier1.clone(),
        );
        let open_pc_time = timer_open_pc.elapsed().as_secs_f64();
        println!("===> Open pc time: {:.3}ms", open_pc_time * 1000.0);

        let timer_open_izw = Instant::now();
        trans.trans_pc_i = committer.open_mat(i_com, &i_tier1, &mat_i, i_hat, i_points);
        trans.trans_pc_z = committer.open_mat(z_com, &z_tier1, &mat_z, z_hat, z_points);
        trans.trans_pc_w = committer.open_mat(w_com, &tier1_w, &mat_w, w_hat, w_points);
        let open_izw_time = timer_open_izw.elapsed().as_secs_f64();
        println!("===> Open i, z, w time: {:.3}ms", open_izw_time * 1000.0);

        let proof_tbv_reduction_time =
            compute_mat_izw_time + commit_w_time + reduce_izw_time + open_izw_time;
        println!(
            "===> Proof of TBV reduction time: {:.3}ms",
            proof_tbv_reduction_time * 1000.0
        );

        println!(
            "===> Full Prover Open time: {:.3}ms",
            timer_open_total.elapsed().as_secs_f64() * 1000.0
        );
    }

    pub fn open_pc(
        &self,
        trans: &mut BatchPCSTrans<E>,
        a_hat: E::ScalarField,
        a_points: Vec<E::ScalarField>,
        layout: &BvaLayout,
        vecs: &Vec<Vec<MyInt>>,
        a_com: PairingOutput<E>,
        a_tier1: Vec<E::G1>,
    ) {
        let mut expanded = vec![0 as MyInt; layout.k_total];
        let mut current_offset = 0;
        for (bucket_idx, &src_idx_opt) in layout.phi_inv.iter().enumerate() {
            let k = layout.bucket_sizes[bucket_idx];
            if let Some(src_idx) = src_idx_opt {
                let v = &vecs[src_idx];
                let len = v.len().min(k);
                expanded[current_offset..current_offset + len].copy_from_slice(&v[..len]);
            }
            current_offset += k;
        }

        let mut a_points_rev = a_points.clone();
        a_points_rev.reverse();
        let xl = a_points_rev;
        let xr = Vec::new();

        let v_com = self.pp.u.mul(&a_hat);

        trans.trans_pc = PC::<E>::open_square_myint_flat(
            &self.pp,
            &expanded,
            &xl,
            &xr,
            v_com,
            a_com,
            &a_tier1,
            E::ScalarField::zero(),
            E::ScalarField::zero(),
        )
        .unwrap();
    }

    pub fn verify(
        &self,
        trans: &mut BatchPCSTrans<E>,
        values: &[E::ScalarField],
        points: &[Vec<E::ScalarField>],
        a_com: PairingOutput<E>,
        i_com: PairingOutput<E>,
        z_com: PairingOutput<E>,
    ) -> bool {
        let timer_verify_reduce = Instant::now();
        let (ok_reduce, a_hat, b_hat, a_points, y_vec, omega) =
            self.verify_reduce(&mut trans.trans_reduce, values, points);
        let verify_reduce_time = timer_verify_reduce.elapsed().as_secs_f64();
        println!(
            "===> Verify reduce time: {:.3}ms",
            verify_reduce_time * 1000.0
        );

        let w_com = trans.mat_w_com.clone();
        let timer_verify_tbv_reduce = Instant::now();
        let (ok_sub, i_hat, z_hat, w_hat, i_points, z_points, w_points) = self
            .prover_input_bva
            .layout
            .verify_as_subprotocol(&mut trans.trans_reduce, b_hat, omega, &y_vec);
        let verify_tbv_reduce_time = timer_verify_tbv_reduce.elapsed().as_secs_f64();
        println!(
            "===> Verify tbv reduce time: {:.3}ms",
            verify_tbv_reduce_time * 1000.0
        );

        let timer_verify_pc = Instant::now();
        let committer = FieldMatCommitter::<E>::new(self.pp.clone());
        let ok_pc = self.verify_pc(trans, a_hat, a_points, a_com);
        let verify_pc_time = timer_verify_pc.elapsed().as_secs_f64();
        println!("===> Verify pc time: {:.3}ms", verify_pc_time * 1000.0);

        let timer_verify_pc_izw = Instant::now();
        let ok_i = committer.verify_mat(i_com, i_hat, i_points, &trans.trans_pc_i);
        let ok_z = committer.verify_mat(z_com, z_hat, z_points, &trans.trans_pc_z);
        let ok_w = committer.verify_mat(w_com, w_hat, w_points, &trans.trans_pc_w);
        let verify_pc_izw_time = timer_verify_pc_izw.elapsed().as_secs_f64();
        println!(
            "===> Verify pc izw time: {:.3}ms",
            verify_pc_izw_time * 1000.0
        );

        let timer_verify_fs = Instant::now();
        let ok_fs = trans.trans_reduce.fs.verify_fs();
        let verify_fs_time = timer_verify_fs.elapsed().as_secs_f64();
        println!("===> Verify fs time: {:.3}ms", verify_fs_time * 1000.0);

        let timer_verify_total = Instant::now();
        let verify_total_time = timer_verify_total.elapsed().as_secs_f64();
        println!(
            "===> Verify total time: {:.3}ms",
            verify_total_time * 1000.0
        );

        ok_reduce && ok_sub && ok_pc && ok_i && ok_z && ok_w && ok_fs
    }

    pub fn verify_reduce(
        &self,
        trans: &mut Transcript<<E as Pairing>::ScalarField>,
        values: &[E::ScalarField],
        points: &[Vec<E::ScalarField>],
    ) -> (
        bool,
        E::ScalarField,
        E::ScalarField,
        Vec<E::ScalarField>,
        Vec<E::ScalarField>,
        E::ScalarField,
    ) {
        let omega = trans.get_at_position(trans.pointer);
        trans.pointer += 1;

        let (batch_values, tbv) = self.gen_tbv_from_values_and_points(
            values,
            points,
            omega,
            &self.prover_input_bva.layout,
        );

        #[cfg(test)]
        {
            println!(
                "[debug] omega(verify)={:?}, batch_values={:?}",
                omega, batch_values
            );
        }

        let litebulleteo = LiteBulletEO::new(batch_values, tbv.k_total);

        let (ok, a_hat, b_hat, challenges_inv, challenges) =
            litebulleteo.verify_as_subprotocol(trans);

        (ok, a_hat, b_hat, challenges_inv, challenges, omega)
    }

    pub fn verify_pc(
        &self,
        trans: &mut BatchPCSTrans<E>,
        a_hat: E::ScalarField,
        a_points: Vec<E::ScalarField>,
        a_com: PairingOutput<E>,
    ) -> bool {
        let v_com = self.pp.u.mul(&a_hat);

        let mut a_points_rev = a_points.clone();
        a_points_rev.reverse();
        let xl = a_points_rev;
        let xr = Vec::new();

        PC::<E>::verify_square(&self.pp, a_com, v_com, &xl, &xr, &trans.trans_pc).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_data::{
        generate_deterministic_vecs_decreasing_sizes, generate_random_challenges,
    };
    use ark_bls12_381::{Bls12_381, Fr};

    #[test]
    fn test_batch_pcs_full_flow() {
        let log_max_n = 10;
        let mut batch_pcs = BatchPCS::<Bls12_381>::setup(log_max_n);

        // 1. Generate random data
        let vecs = generate_deterministic_vecs_decreasing_sizes(log_max_n - 1);
        batch_pcs.set_prover_input(vecs.clone());
        let num_vecs = vecs.len();
        let mut points = Vec::new();
        let mut values = Vec::new();

        for i in 0..num_vecs {
            let log_k = vecs[i].len().ilog2() as usize;
            let point = generate_random_challenges::<Fr>(log_k);

            // Calculate ground truth value for verification comparison
            let val = PC::<Bls12_381>::eval_my_int_flat_output_field(&vecs[i], &point, &Vec::new());

            points.push(point);
            values.push(val);
        }

        // 2. Commit
        let (a_com, i_com, a_tier1, i_tier1) = batch_pcs.commit();
        let (z_com, z_tier1) = batch_pcs.commit_points(&points);

        // 3. Open (reduction + PC)
        let mut trans_prover = BatchPCSTrans::<Bls12_381>::new();

        batch_pcs.open(
            &mut trans_prover,
            &values,
            &points,
            a_com,
            i_com,
            z_com,
            a_tier1,
            i_tier1,
            z_tier1,
        );

        // 4. Verify
        let mut trans_verifier = trans_prover.clone();
        trans_verifier.reset_pointer();
        let ok = batch_pcs.verify(&mut trans_verifier, &values, &points, a_com, i_com, z_com);
        assert!(ok, "BatchPCS full flow verification failed");
    }
}
