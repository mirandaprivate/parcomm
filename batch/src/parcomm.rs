//! ParComm: collection commitment and succinct collective opening for a
//! collection of matrix evaluation claims (paper Section 6, Algorithms
//! "Collection Commitment and Succinct Collective Opening" and "Proof of TBV
//! Evaluation").
//!
//! The front end ([`crate::mat_to_bva`]) recursively splits each matrix into
//! aligned power-of-two submatrices and compiles the collection into one BVA
//! `a` plus a TBV `zeta` with `<a, zeta> = sum_j omega^{j-1} v_j`. This module
//! proves that inner product succinctly:
//!
//! * `Commit` commits the omega-independent matrices: the flattened BVA `C_a`,
//!   the offset lookup matrix `C_I`, and the query-point matrix `C_Z`.
//! * `Open` samples `omega`, builds `zeta`, runs Packed EO-LiteBullet on
//!   `<a, zeta>`, opens `C_a` with the base PCS, then proves the TBV evaluation
//!   `bar_b = Eval(zeta, y)` via two row-wise grand products:
//!     - `Pi_i = GrandProdRow(X)_i`, `X = 1 - I + (Z + I) o (1 y)`   (offsets + local points), and
//!     - `eta_i = GrandProdRow(Y)_i`                                  (scale factors, `Y` committed here),
//!   combined as `bar_b = <eta, Pi>`.
//! * `Verify` replays the transcript and checks every reduction and opening.
//!
//! Because the layout is decreasing-size, the flattened BVA is contiguous with
//! no interior zeros (only an implicit power-of-two suffix), so `C_a` commits a
//! single dense vector.
//!
//! `Y` encodes `eta_i = omega^{mat_index}·F_l·F_r`, so it depends on the
//! opening challenge `omega` and is committed at the start of `Open` (as in the
//! paper), together with the grand-product witnesses `W_X`, `W_Y`.

use ark_crypto_primitives::sponge::Absorb;
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::{One, PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress};
use ark_std::UniformRand;
use std::ops::Mul;

use atomic_proof::{BatchPoints, LiteBullet, MatMul};
use fs::helper_trans::Transcript;
use mat::{xi, DenseMatFieldCM};

use poly_commit::smart_pc::{SmartPC as PC, Trans as PCTrans, UniversalParams as PP};
use poly_commit::MyInt;

use crate::bva::BvaLayout;
use crate::litebullet_eo::LiteBulletEO;
use crate::mat_to_bva::MatToBva;
use crate::tbvproof::{FieldMatCommitter, TbvReduceProof};

/// Reduced leaf claims of the TBV-evaluation proof: each `(value, point)` is an
/// MLE opening claim on the correspondingly committed matrix.
#[derive(Clone)]
struct TbvEvalClaims<F: PrimeField> {
    i: (F, (Vec<F>, Vec<F>)),
    z: (F, (Vec<F>, Vec<F>)),
    y: (F, (Vec<F>, Vec<F>)),
    wx: (F, (Vec<F>, Vec<F>)),
    wy: (F, (Vec<F>, Vec<F>)),
}

/// Transcript and commitments produced by a ParComm opening.
#[derive(Clone)]
pub struct ParCommTrans<E: Pairing>
where
    E::ScalarField: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
    E::G1: CanonicalSerialize + CanonicalDeserialize,
    E::G2: CanonicalSerialize + CanonicalDeserialize,
    PairingOutput<E>: CanonicalSerialize + CanonicalDeserialize,
{
    pub trans_reduce: Transcript<<E as Pairing>::ScalarField>,
    pub trans_pc_a: PCTrans<E>,
    pub trans_pc_i: PCTrans<E>,
    pub trans_pc_z: PCTrans<E>,
    pub trans_pc_y: PCTrans<E>,
    pub trans_pc_wx: PCTrans<E>,
    pub trans_pc_wy: PCTrans<E>,
    /// Opening-time commitments (Y and the two grand-product witnesses).
    pub com_y: PairingOutput<E>,
    pub com_wx: PairingOutput<E>,
    pub com_wy: PairingOutput<E>,
}

impl<E: Pairing> ParCommTrans<E>
where
    E::ScalarField: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
    E::G1: CanonicalSerialize + CanonicalDeserialize,
    E::G2: CanonicalSerialize + CanonicalDeserialize,
    PairingOutput<E>: CanonicalSerialize + CanonicalDeserialize,
{
    pub fn new() -> Self {
        let mut rng = ark_std::test_rng();
        Self {
            trans_reduce: Transcript::new(E::ScalarField::rand(&mut rng)),
            trans_pc_a: PCTrans::new(),
            trans_pc_i: PCTrans::new(),
            trans_pc_z: PCTrans::new(),
            trans_pc_y: PCTrans::new(),
            trans_pc_wx: PCTrans::new(),
            trans_pc_wy: PCTrans::new(),
            com_y: PairingOutput::<E>::zero(),
            com_wx: PairingOutput::<E>::zero(),
            com_wy: PairingOutput::<E>::zero(),
        }
    }

    pub fn reset_pointer(&mut self) {
        self.trans_reduce.reset_pointer();
    }

    /// Total serialized proof size in bytes (uncompressed).
    pub fn get_proof_size(&self) -> usize {
        let trans = self.trans_reduce.get_proof_size_in_bytes();
        let pcs = [
            &self.trans_pc_a,
            &self.trans_pc_i,
            &self.trans_pc_z,
            &self.trans_pc_y,
            &self.trans_pc_wx,
            &self.trans_pc_wy,
        ]
        .iter()
        .map(|t| t.serialized_size(Compress::No))
        .sum::<usize>();
        let coms = self.com_y.serialized_size(Compress::No)
            + self.com_wx.serialized_size(Compress::No)
            + self.com_wy.serialized_size(Compress::No);
        trans + pcs + coms
    }
}

/// Registration commitments returned by [`ParComm::commit`].
pub struct ParCommCommitment<E: Pairing>
where
    E::ScalarField: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
    E::G1: CanonicalSerialize + CanonicalDeserialize,
    E::G2: CanonicalSerialize + CanonicalDeserialize,
    PairingOutput<E>: CanonicalSerialize + CanonicalDeserialize,
{
    pub com_a: PairingOutput<E>,
    pub com_i: PairingOutput<E>,
    pub com_z: PairingOutput<E>,
    pub tier1_a: Vec<E::G1>,
    pub tier1_i: Vec<E::G1>,
    pub tier1_z: Vec<E::G1>,
}

pub struct ParComm<E: Pairing>
where
    E::ScalarField: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
    E::G1: CanonicalSerialize + CanonicalDeserialize,
    E::G2: CanonicalSerialize + CanonicalDeserialize,
    PairingOutput<E>: CanonicalSerialize + CanonicalDeserialize,
{
    pub pp: PP<E>,
}

impl<E: Pairing> ParComm<E>
where
    E::ScalarField:
        PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize + From<MyInt>,
    E::G1: CanonicalSerialize + CanonicalDeserialize,
    E::G2: CanonicalSerialize + CanonicalDeserialize,
    PairingOutput<E>: CanonicalSerialize + CanonicalDeserialize,
{
    pub fn new(pp: PP<E>) -> Self {
        Self { pp }
    }

    /// Set up public parameters sized for a logical BVA length `K = 2^log_k`.
    pub fn setup(log_k: usize) -> Self {
        let mut rng = ark_std::test_rng();
        // The base PCS commits square matrices; qlog is log2 of one side.
        let qlog = (log_k + 2) / 2;
        let pp = PC::<E>::setup(qlog, &mut rng).unwrap();
        Self { pp }
    }

    // -- Matrix builders -----------------------------------------------------

    /// Query-point matrix `Z in F^{t x log2 K}`: row `s` is the local TBV point
    /// `z_i` of the bucket that source `s` maps to, zero-padded to `log2 K`.
    fn compute_mat_z(
        &self,
        compiled: &MatToBva<E::ScalarField>,
    ) -> DenseMatFieldCM<E::ScalarField> {
        let layout = &compiled.bva.layout;
        let log_k = layout.k_total.ilog2() as usize;
        let metadata = compiled.bucket_metadata();

        // One row per source (bucket) in source order, then transpose to the
        // column-major t x log_k matrix the reductions expect.
        let mut rows: Vec<Vec<E::ScalarField>> = Vec::with_capacity(metadata.len());
        for (_, _, z) in &metadata {
            let mut row = z.clone();
            row.resize(log_k, E::ScalarField::zero());
            rows.push(row);
        }
        DenseMatFieldCM::from_data(rows).par_transpose()
    }

    /// Scale matrix `Y in F^{t x log2 K}`: row `s` has product `eta_s =
    /// omega^{mat_index} * eta_base`, so the row-wise grand product yields the
    /// TBV scales. Two meaningful factors (`omega` power and `F_l F_r`) are
    /// placed in the leading columns; the rest are ones.
    fn compute_mat_y(
        &self,
        compiled: &MatToBva<E::ScalarField>,
        omega: E::ScalarField,
    ) -> DenseMatFieldCM<E::ScalarField> {
        let layout = &compiled.bva.layout;
        let log_k = layout.k_total.ilog2() as usize;
        let metadata = compiled.bucket_metadata();
        let powers = omega_powers(omega, compiled.num_mats);

        let mut rows: Vec<Vec<E::ScalarField>> = Vec::with_capacity(metadata.len());
        for (mat_index, eta_base, _) in &metadata {
            let mut row = vec![E::ScalarField::one(); log_k];
            let power = powers[*mat_index];
            if log_k == 1 {
                row[0] = power * *eta_base;
            } else {
                row[0] = power;
                row[1] = *eta_base;
            }
            rows.push(row);
        }
        DenseMatFieldCM::from_data(rows).par_transpose()
    }

    /// The scale vector `eta` in source order (`eta_s = omega^{mat_index} eta_base`).
    fn eta_vector(
        &self,
        compiled: &MatToBva<E::ScalarField>,
        omega: E::ScalarField,
    ) -> Vec<E::ScalarField> {
        let powers = omega_powers(omega, compiled.num_mats);
        compiled
            .bucket_metadata()
            .iter()
            .map(|(mat_index, eta_base, _)| powers[*mat_index] * *eta_base)
            .collect()
    }

    // -- Commit --------------------------------------------------------------

    /// Commit the omega-independent registration matrices: the flattened BVA
    /// `C_a`, the offset lookup matrix `C_I`, and the query-point matrix `C_Z`.
    pub fn commit(&self, compiled: &MatToBva<E::ScalarField>) -> ParCommCommitment<E> {
        let layout = &compiled.bva.layout;

        // C_a: contiguous flattened BVA (decreasing layout => no interior zeros).
        let expand = compiled.bva.expand_to_myint_vec();
        let (com_a, tier1_a) =
            PC::<E>::commit_square_myint_flat(&self.pp, &expand, E::ScalarField::zero()).unwrap();

        let committer = FieldMatCommitter::<E>::new(self.pp.clone());
        let mat_i = layout.compute_mat_i();
        let (com_i, tier1_i) = committer.commit_mat(&mat_i);

        let mat_z = self.compute_mat_z(compiled);
        let (com_z, tier1_z) = committer.commit_mat(&mat_z);

        ParCommCommitment {
            com_a,
            com_i,
            com_z,
            tier1_a,
            tier1_i,
            tier1_z,
        }
    }

    // -- Open ----------------------------------------------------------------

    /// Open the collection at the compiled claims, producing a full transcript.
    pub fn open(
        &self,
        trans: &mut ParCommTrans<E>,
        compiled: &MatToBva<E::ScalarField>,
        commitment: &ParCommCommitment<E>,
    ) {
        let layout = compiled.bva.layout.clone();

        // Verifier sends the batching challenge omega.
        let omega = trans.trans_reduce.gen_challenge();
        let tbv = compiled.gen_tbv(omega);

        // Packed EO-LiteBullet on c = <a, zeta>.
        let litebullet = LiteBulletEO::new(compiled.batched_value(omega), tbv.k_total);
        let (a_hat, bar_b, a_point, y_vec) =
            litebullet.reduce_prover(&mut trans.trans_reduce, &compiled.bva, tbv);

        // Open C_a with the base PCS.
        self.open_a(
            trans,
            &compiled.bva,
            a_hat,
            a_point,
            &commitment.com_a,
            &commitment.tier1_a,
        );

        // Build the TBV-evaluation matrices (omega/y dependent).
        let mat_i = layout.compute_mat_i();
        let mat_z = self.compute_mat_z(compiled);
        let mat_y = self.compute_mat_y(compiled, omega);
        let mat_x = layout.compute_mat_x(&y_vec, &mat_i, &mat_z);
        let mat_wx = layout.compute_mat_w(&mat_x);
        let mat_wpx = layout.compute_mat_w_prime(&mat_x, &mat_wx);
        let mat_wy = layout.compute_mat_w(&mat_y);
        let mat_wpy = layout.compute_mat_w_prime(&mat_y, &mat_wy);

        // Commit Y and the two grand-product witnesses (all omega dependent).
        let committer = FieldMatCommitter::<E>::new(self.pp.clone());
        let (com_y, tier1_y) = committer.commit_mat(&mat_y);
        let (com_wx, tier1_wx) = committer.commit_mat(&mat_wx);
        let (com_wy, tier1_wy) = committer.commit_mat(&mat_wy);
        trans.com_y = com_y;
        trans.com_wx = com_wx;
        trans.com_wy = com_wy;

        let eta_vec = self.eta_vector(compiled, omega);
        let claims = self.reduce_tbv_eval_prover(
            &mut trans.trans_reduce,
            &layout,
            bar_b,
            &eta_vec,
            &mat_i,
            &mat_z,
            &mat_y,
            &mat_x,
            &mat_wx,
            &mat_wpx,
            &mat_wy,
            &mat_wpy,
            &y_vec,
        );

        // Open every leaf claim against its commitment.
        trans.trans_pc_i = committer.open_mat(
            commitment.com_i,
            &commitment.tier1_i,
            &mat_i,
            claims.i.0,
            claims.i.1,
        );
        trans.trans_pc_z = committer.open_mat(
            commitment.com_z,
            &commitment.tier1_z,
            &mat_z,
            claims.z.0,
            claims.z.1,
        );
        trans.trans_pc_y = committer.open_mat(com_y, &tier1_y, &mat_y, claims.y.0, claims.y.1);
        trans.trans_pc_wx =
            committer.open_mat(com_wx, &tier1_wx, &mat_wx, claims.wx.0, claims.wx.1);
        trans.trans_pc_wy =
            committer.open_mat(com_wy, &tier1_wy, &mat_wy, claims.wy.0, claims.wy.1);
    }

    /// Open the BVA commitment `C_a` at the Packed EO-LiteBullet point.
    fn open_a(
        &self,
        trans: &mut ParCommTrans<E>,
        bva: &crate::bva::BVA,
        a_hat: E::ScalarField,
        a_point: Vec<E::ScalarField>,
        com_a: &PairingOutput<E>,
        tier1_a: &Vec<E::G1>,
    ) {
        let expand = bva.expand_to_myint_vec();
        let mut xl = a_point;
        xl.reverse();
        let v_com = self.pp.u.mul(&a_hat);
        trans.trans_pc_a = PC::<E>::open_square_myint_flat(
            &self.pp,
            &expand,
            &xl,
            &Vec::new(),
            v_com,
            *com_a,
            tier1_a,
            E::ScalarField::zero(),
            E::ScalarField::zero(),
        )
        .unwrap();
    }

    // -- TBV evaluation reduction (prover) -----------------------------------

    /// Reduce `bar_b = <eta, Pi> = Eval(zeta, y)` to leaf openings of
    /// `I, Z, Y, W_X, W_Y`, where `Pi = GrandProdRow(X)` and
    /// `eta = GrandProdRow(Y)`.
    fn reduce_tbv_eval_prover(
        &self,
        trans: &mut Transcript<E::ScalarField>,
        layout: &BvaLayout,
        bar_b: E::ScalarField,
        eta_vec: &[E::ScalarField],
        mat_i: &DenseMatFieldCM<E::ScalarField>,
        mat_z: &DenseMatFieldCM<E::ScalarField>,
        mat_y: &DenseMatFieldCM<E::ScalarField>,
        mat_x: &DenseMatFieldCM<E::ScalarField>,
        mat_wx: &DenseMatFieldCM<E::ScalarField>,
        mat_wpx: &DenseMatFieldCM<E::ScalarField>,
        mat_wy: &DenseMatFieldCM<E::ScalarField>,
        mat_wpy: &DenseMatFieldCM<E::ScalarField>,
        y_vec: &Vec<E::ScalarField>,
    ) -> TbvEvalClaims<E::ScalarField> {
        let t = layout.phi.len();
        let log_k = layout.k_total.ilog2() as usize;
        let epsilon_mat = epsilon_matrix::<E::ScalarField>(log_k);

        // Pi and eta are the last columns of the two grand-product matrices.
        let pi_vec = mat_wpx.data[log_k - 1].clone();
        let eta_last = mat_wpy.data[log_k - 1].clone();
        debug_assert_eq!(&eta_last[..eta_vec.len()], eta_vec);

        // Combine: bar_b = <eta, Pi>.
        let lb = LiteBullet::<E::ScalarField>::new(bar_b, t);
        let (eta_red, pi_red, chal_inv, chal) = lb.reduce_prover(trans, &eta_last, &pi_vec);

        // pi_red = <lastcol(W_X'), xi(chal)>  ->  W_X' leaf via MatMul with epsilon.
        let matmul_x = MatMul::<E::ScalarField>::new(
            pi_red,
            (chal.clone(), Vec::new()),
            (t, 1),
            mat_wpx.shape,
            epsilon_mat.shape,
        );
        let (wpx_red_a, _eps_red_x, wpx_pt_a, _eps_pt_x) =
            matmul_x.reduce_prover(trans, mat_wpx, &epsilon_mat);

        // eta_red = <lastcol(W_Y'), xi(chal_inv)>  ->  W_Y' leaf via MatMul with epsilon.
        let matmul_y = MatMul::<E::ScalarField>::new(
            eta_red,
            (chal_inv.clone(), Vec::new()),
            (t, 1),
            mat_wpy.shape,
            epsilon_mat.shape,
        );
        let (wpy_red_a, _eps_red_y, wpy_pt_a, _eps_pt_y) =
            matmul_y.reduce_prover(trans, mat_wpy, &epsilon_mat);

        // --- X grand product: Pi = GrandProdRow(X), X = 1 - I + (Z + I) o (1 y).
        let theta_x = trans.gen_challenge();
        let (wx_red1, wpx_red2, wx_pt1, wpx_pt2) =
            layout.reduce_prover_eq2(trans, theta_x, mat_wx, mat_wpx);
        let bp_wpx = BatchPoints::<E::ScalarField>::new(
            vec![wpx_red_a, wpx_red2],
            vec![wpx_pt_a, wpx_pt2],
            mat_wpx.shape,
        );
        let (wpx_red, wpx_pt) = bp_wpx.reduce_prover(trans, mat_wpx);
        let (x_red, wx_red2, x_pt, wx_pt2) =
            layout.reduce_prover_eq3(trans, wpx_red, wpx_pt, mat_x, mat_wx);
        let bp_wx = BatchPoints::<E::ScalarField>::new(
            vec![wx_red1, wx_red2],
            vec![wx_pt1, wx_pt2],
            mat_wx.shape,
        );
        let (wx_red, wx_pt) = bp_wx.reduce_prover(trans, mat_wx);
        let (i_red, z_red, i_pt, z_pt) =
            layout.reduce_prover_eq4(trans, x_red, x_pt, mat_i, mat_z, y_vec);

        // --- Y grand product: eta = GrandProdRow(Y), Y committed directly.
        let theta_y = trans.gen_challenge();
        let (wy_red1, wpy_red2, wy_pt1, wpy_pt2) =
            layout.reduce_prover_eq2(trans, theta_y, mat_wy, mat_wpy);
        let bp_wpy = BatchPoints::<E::ScalarField>::new(
            vec![wpy_red_a, wpy_red2],
            vec![wpy_pt_a, wpy_pt2],
            mat_wpy.shape,
        );
        let (wpy_red, wpy_pt) = bp_wpy.reduce_prover(trans, mat_wpy);
        let (y_red, wy_red2, y_pt, wy_pt2) =
            layout.reduce_prover_eq3(trans, wpy_red, wpy_pt, mat_y, mat_wy);
        let bp_wy = BatchPoints::<E::ScalarField>::new(
            vec![wy_red1, wy_red2],
            vec![wy_pt1, wy_pt2],
            mat_wy.shape,
        );
        let (wy_red, wy_pt) = bp_wy.reduce_prover(trans, mat_wy);

        TbvEvalClaims {
            i: (i_red, i_pt),
            z: (z_red, z_pt),
            y: (y_red, y_pt),
            wx: (wx_red, wx_pt),
            wy: (wy_red, wy_pt),
        }
    }

    // -- Verify --------------------------------------------------------------

    /// Verify a ParComm opening against the registration commitments.
    pub fn verify(
        &self,
        trans: &mut ParCommTrans<E>,
        compiled: &MatToBva<E::ScalarField>,
        commitment: &ParCommCommitment<E>,
    ) -> bool {
        let layout = compiled.bva.layout.clone();

        // Recompute omega and re-run Packed EO-LiteBullet verification.
        let omega = trans
            .trans_reduce
            .get_at_position(trans.trans_reduce.pointer);
        trans.trans_reduce.pointer += 1;
        let tbv = compiled.gen_tbv(omega);
        let litebullet = LiteBulletEO::new(compiled.batched_value(omega), tbv.k_total);
        let (ok_ip, a_hat, bar_b, a_challenges_inv, y_vec) =
            litebullet.verify_as_subprotocol(&mut trans.trans_reduce);

        // Verify C_a opening at the reduced point.
        let mut xl = a_challenges_inv;
        xl.reverse();
        let v_com = self.pp.u.mul(&a_hat);
        let ok_a = PC::<E>::verify_square(
            &self.pp,
            commitment.com_a,
            v_com,
            &xl,
            &Vec::new(),
            &trans.trans_pc_a,
        )
        .unwrap_or(false);

        // Verify the TBV-evaluation reduction.
        let (ok_reduce, claims) =
            self.verify_tbv_eval(&mut trans.trans_reduce, &layout, bar_b, &y_vec);

        // Verify every leaf opening.
        let committer = FieldMatCommitter::<E>::new(self.pp.clone());
        let ok_i =
            committer.verify_mat(commitment.com_i, claims.i.0, claims.i.1, &trans.trans_pc_i);
        let ok_z =
            committer.verify_mat(commitment.com_z, claims.z.0, claims.z.1, &trans.trans_pc_z);
        let ok_y = committer.verify_mat(trans.com_y, claims.y.0, claims.y.1, &trans.trans_pc_y);
        let ok_wx =
            committer.verify_mat(trans.com_wx, claims.wx.0, claims.wx.1, &trans.trans_pc_wx);
        let ok_wy =
            committer.verify_mat(trans.com_wy, claims.wy.0, claims.wy.1, &trans.trans_pc_wy);

        let ok_fs = trans.trans_reduce.fs.verify_fs();

        let ok = ok_ip && ok_a && ok_reduce && ok_i && ok_z && ok_y && ok_wx && ok_wy && ok_fs;
        if !ok {
            println!(
                "[parcomm-verify] ip={ok_ip} a={ok_a} reduce={ok_reduce} i={ok_i} z={ok_z} y={ok_y} wx={ok_wx} wy={ok_wy} fs={ok_fs}"
            );
        }
        ok
    }

    /// Verifier side of [`Self::reduce_tbv_eval_prover`].
    fn verify_tbv_eval(
        &self,
        trans: &mut Transcript<E::ScalarField>,
        layout: &BvaLayout,
        bar_b: E::ScalarField,
        y_vec: &Vec<E::ScalarField>,
    ) -> (bool, TbvEvalClaims<E::ScalarField>) {
        let t = layout.phi.len();
        let log_k = layout.k_total.ilog2() as usize;

        // Combine: bar_b = <eta, Pi>.
        let lb = LiteBullet::<E::ScalarField>::new(bar_b, t);
        let (ok0, eta_red, pi_red, chal_inv, chal) = lb.verify_as_subprotocol(trans);

        let matmul_x = MatMul::<E::ScalarField>::new(
            pi_red,
            (chal.clone(), Vec::new()),
            (t, 1),
            (t, log_k),
            (log_k, 1),
        );
        let (okx_mm, wpx_red_a, eps_red_x, wpx_pt_a, eps_pt_x) =
            matmul_x.verify_as_subprotocol(trans);
        let okx_eps = eps_red_x == xi::xi_get_at_position(log_k - 1, &eps_pt_x.0);

        let matmul_y = MatMul::<E::ScalarField>::new(
            eta_red,
            (chal_inv.clone(), Vec::new()),
            (t, 1),
            (t, log_k),
            (log_k, 1),
        );
        let (oky_mm, wpy_red_a, eps_red_y, wpy_pt_a, eps_pt_y) =
            matmul_y.verify_as_subprotocol(trans);
        let oky_eps = eps_red_y == xi::xi_get_at_position(log_k - 1, &eps_pt_y.0);

        // --- X grand product.
        let theta_x = trans.get_at_position(trans.pointer);
        trans.pointer += 1;
        let (okx2, wx_red1, wpx_red2, wx_pt1, wpx_pt2) = layout.verify_eq2(trans, theta_x);
        let bp_wpx = BatchPoints::<E::ScalarField>::new(
            vec![wpx_red_a, wpx_red2],
            vec![wpx_pt_a, wpx_pt2],
            (t, log_k),
        );
        let (okx3, wpx_red, wpx_pt) = bp_wpx.verify_as_subprotocol(trans);
        let (okx4, x_red, wx_red2, x_pt, wx_pt2) = layout.verify_eq3(trans, wpx_red, wpx_pt);
        let bp_wx = BatchPoints::<E::ScalarField>::new(
            vec![wx_red1, wx_red2],
            vec![wx_pt1, wx_pt2],
            (t, log_k),
        );
        let (okx5, wx_red, wx_pt) = bp_wx.verify_as_subprotocol(trans);
        // y_vec is the public Packed EO-LiteBullet point; eq4 rebuilds the
        // broadcast matrix from it.
        let (okx6, i_red, z_red, i_pt, z_pt) = layout.verify_eq4(trans, x_red, x_pt, y_vec);

        // --- Y grand product.
        let theta_y = trans.get_at_position(trans.pointer);
        trans.pointer += 1;
        let (oky2, wy_red1, wpy_red2, wy_pt1, wpy_pt2) = layout.verify_eq2(trans, theta_y);
        let bp_wpy = BatchPoints::<E::ScalarField>::new(
            vec![wpy_red_a, wpy_red2],
            vec![wpy_pt_a, wpy_pt2],
            (t, log_k),
        );
        let (oky3, wpy_red, wpy_pt) = bp_wpy.verify_as_subprotocol(trans);
        let (oky4, y_red, wy_red2, y_pt, wy_pt2) = layout.verify_eq3(trans, wpy_red, wpy_pt);
        let bp_wy = BatchPoints::<E::ScalarField>::new(
            vec![wy_red1, wy_red2],
            vec![wy_pt1, wy_pt2],
            (t, log_k),
        );
        let (oky5, wy_red, wy_pt) = bp_wy.verify_as_subprotocol(trans);

        let ok = ok0
            && okx_mm
            && okx_eps
            && oky_mm
            && oky_eps
            && okx2
            && okx3
            && okx4
            && okx5
            && okx6
            && oky2
            && oky3
            && oky4
            && oky5;

        (
            ok,
            TbvEvalClaims {
                i: (i_red, i_pt),
                z: (z_red, z_pt),
                y: (y_red, y_pt),
                wx: (wx_red, wx_pt),
                wy: (wy_red, wy_pt),
            },
        )
    }
}

/// Powers `[1, omega, ..., omega^{count-1}]`.
fn omega_powers<F: PrimeField>(omega: F, count: usize) -> Vec<F> {
    std::iter::successors(Some(F::one()), |&x| Some(x * omega))
        .take(count)
        .collect()
}

/// The selector matrix `epsilon in F^{log_k x 1}` picking the last column.
fn epsilon_matrix<F: PrimeField>(log_k: usize) -> DenseMatFieldCM<F> {
    let mut col = vec![F::zero(); log_k];
    if let Some(last) = col.last_mut() {
        *last = F::one();
    }
    DenseMatFieldCM::from_data(vec![col])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mat_to_bva::{dense_matrix_eval, MatEvalClaim};
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_std::rand::{rngs::StdRng, Rng, SeedableRng};

    fn ceil_log2(n: usize) -> usize {
        n.next_power_of_two().trailing_zeros() as usize
    }

    fn rand_matrix(rng: &mut StdRng, mu: usize, nu: usize) -> Vec<Vec<MyInt>> {
        (0..nu)
            .map(|_| (0..mu).map(|_| rng.gen_range(-8..8) as MyInt).collect())
            .collect()
    }

    fn make_claim(rng: &mut StdRng, mu: usize, nu: usize) -> MatEvalClaim<Fr> {
        let cols = rand_matrix(rng, mu, nu);
        let x_l: Vec<Fr> = (0..ceil_log2(mu))
            .map(|_| Fr::from(rng.gen::<u64>()))
            .collect();
        let x_r: Vec<Fr> = (0..ceil_log2(nu))
            .map(|_| Fr::from(rng.gen::<u64>()))
            .collect();
        MatEvalClaim::new(cols, (mu, nu), x_l, x_r)
    }

    #[test]
    fn parcomm_full_flow() {
        let mut rng = StdRng::seed_from_u64(42);
        // Mixed power-of-two and non-power-of-two matrices; the (3,3) matrix
        // splits into several buckets.
        let shapes = [(4usize, 4usize), (8, 2), (2, 2), (3, 3)];
        let claims: Vec<MatEvalClaim<Fr>> = shapes
            .iter()
            .map(|&(mu, nu)| make_claim(&mut rng, mu, nu))
            .collect();

        // Independent ground-truth batched value.
        let compiled = MatToBva::build(&claims);
        let log_k = compiled.bva.layout.k_total.ilog2() as usize;

        let parcomm = ParComm::<Bls12_381>::setup(log_k);
        let commitment = parcomm.commit(&compiled);

        let mut trans = ParCommTrans::<Bls12_381>::new();
        parcomm.open(&mut trans, &compiled, &commitment);

        // Sanity: the batched value equals sum_j omega^{j-1} v_j from the dense
        // projections (omega is the first reduce challenge).
        let mut trans_check = trans.clone();
        trans_check.reset_pointer();
        let omega = trans_check
            .trans_reduce
            .get_at_position(trans_check.trans_reduce.pointer);
        let dense_values: Vec<Fr> = claims
            .iter()
            .map(|c| dense_matrix_eval(&c.cols, c.shape, &c.x_l, &c.x_r))
            .collect();
        let powers: Vec<Fr> = std::iter::successors(Some(Fr::one()), |&x| Some(x * omega))
            .take(claims.len())
            .collect();
        let expected_c: Fr = dense_values
            .iter()
            .zip(powers.iter())
            .map(|(v, p)| *v * *p)
            .sum();
        assert_eq!(expected_c, compiled.batched_value(omega));

        let mut trans_v = trans.clone();
        trans_v.reset_pointer();
        let ok = parcomm.verify(&mut trans_v, &compiled, &commitment);
        assert!(ok, "ParComm full flow verification failed");
    }
}
