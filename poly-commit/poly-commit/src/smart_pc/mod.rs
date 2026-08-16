use std::time::Instant;

// use core::num;

use crate::Error;
use ark_ec::pairing::{Pairing, PairingOutput};
    // scalar_mul::ScalarMul, AffineRepr, CurveGroup, VariableBaseMSM
// };
use ark_ff::{
    One,
    Field,
    UniformRand,
    Zero,
};
// use ark_poly::DenseUVPolynomial;
use ark_std::{
    // format,
    marker::PhantomData,
    rand::RngCore,
    ops::Mul,
};

use ark_serialize::{CanonicalSerialize, Compress};

use rand::thread_rng;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Data structures for the SMART PC
pub mod data_structures;
pub use data_structures::*;

/// utilitiy functions
pub mod utils;
pub use utils::*;

pub mod fiat_shamir;
pub use fiat_shamir::*;
use xi::{xi_from_challenges, xi_ip_from_challenges};

/// subprotocols for the SMART PC
pub mod sub_protocols;
// use sub_protocols::*;

pub mod xi;
/// SMART_PC is an implementation of modified Dory
pub struct SmartPC<E: Pairing> {
    _engine: PhantomData<E>,
}

impl<E> SmartPC<E>
where
    E: Pairing,
    E::ScalarField: CanonicalSerialize,
    E::G1: CanonicalSerialize,
    E::G2: CanonicalSerialize,
    PairingOutput<E>: CanonicalSerialize,
{
    /// Constructs public parameters when given as input the maximum degree `degree`
    /// for the polynomial commitment scheme.
    ///
    pub fn setup<R: RngCore>(
        qlog: usize,
        rng: &mut R,
    ) -> Result<UniversalParams<E>, Error> {
        if qlog < 1 {
            return Err(Error::DegreeIsZero);
        }

        let setup_time =
        start_timer!(||
            format!("Smart-PC::Setup with dimension 2^{}, 2^{} coefficients",
            qlog,
            2 * qlog)
        );

        let q = 1 << qlog;
        
        let mut nu = E::ScalarField::rand(rng);
        while nu.is_zero() {
            nu = E::ScalarField::rand(rng);
        }
        let mut hat_s = E::ScalarField::rand(rng);
        while hat_s.is_zero() {
            hat_s = E::ScalarField::rand(rng);
        }
        let mut g_hat = E::G1::rand(rng);
        while g_hat.is_zero() {
            g_hat = E::G1::rand(rng);
        }
        let mut h_hat = E::G2::rand(rng);
        while h_hat.is_zero() {
            h_hat = E::G2::rand(rng);
        }


        // powers_of_s = [1, s, ..., s^(q-1)], len = q
        let mut powers_of_s = vec![E::ScalarField::one()];
        let mut cur = hat_s;
        for _ in 0..(q-1) {
            powers_of_s.push(cur);
            cur *= &hat_s;
        }

        let hat_sq = cur;

        assert_eq!(hat_sq, hat_s.pow([q as u64]));

        let mut powers_of_sq = vec![E::ScalarField::one()];
        cur = hat_sq;
        for _ in 0..(q-1) {
            powers_of_sq.push(cur);
            cur *= &hat_sq;
        }

        let nu_powers_of_s = powers_of_s.clone()
        .into_par_iter().map(|x| x * &nu)
        .collect::<Vec<_>>();
        let nu_powers_of_sq = powers_of_sq.clone()
        .into_par_iter().map(|x| x * &nu)
        .collect::<Vec<_>>();
        
        let s_h = h_hat.mul(hat_s);
        let sq_h = h_hat.mul(hat_sq);
        let nu_g = g_hat.mul(nu);
        let nu_h = h_hat.mul(nu);
        let hat_s_inv = hat_s.inverse().unwrap();
        let g_0 = g_hat.mul(hat_s_inv);
        let u = E::pairing(g_0, h_hat);
        let tilde_g =
            g_hat.mul(hat_s_inv *  hat_s_inv);
        let tilde_h = h_hat.mul(hat_s_inv);
        let tilde_u = E::pairing(
            tilde_g, h_hat);
        debug_assert_eq!(E::pairing(g_0, tilde_h), tilde_u);
        let vec_g = powers_of_s
        .into_par_iter()
        .map(|x| g_hat.mul(x))
        .collect::<Vec<_>>();
        let vec_h = powers_of_sq.clone()
        .into_par_iter()
        .map(|x| h_hat.mul(x))
        .collect::<Vec<_>>();
        let vec_g_prime = powers_of_sq
        .into_par_iter()
        .map(|x| g_hat.mul(x))
        .collect::<Vec<_>>();
        let vec_nu_g = nu_powers_of_s
        .into_par_iter()
        .map(|x| g_hat.mul(x))
        .collect::<Vec<_>>();
        let vec_nu_h = nu_powers_of_sq
        .into_par_iter()
        .map(|x| h_hat.mul(x))
        .collect::<Vec<_>>();
        
        let pp = UniversalParams {
            q,
            num_vars: q,
            g_hat,
            h_hat,
            s_h,
            sq_h,
            nu_g,
            nu_h,
            g_0,
            tilde_g,
            tilde_h,
            u,
            tilde_u,
            vec_g,
            vec_h,
            vec_g_prime,
            vec_nu_g,
            vec_nu_h,
        };
        end_timer!(setup_time);
        Ok(pp)
    }


    /// Outputs a commitment to a matrix
    /// and intermediate tier-one commitment
    pub fn commit_full(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<E::ScalarField>>,
        hiding_factor: E::ScalarField,
    ) -> Result<(PairingOutput<E>, Vec<E::G1>), Error> {

        let n = mat.len();
        let m = mat[0].len();

        let commit_time = start_timer!(|| format!(
            "Committing to {:?}-vector:",
            n * m
        ));

        let vec_g = pp.vec_g[0..m].to_vec();
        let vec_h = pp.vec_h[0..n].to_vec();

        let timer = Instant::now();
        let mut tier_one_vec = Vec::new();
        for i in 0..n {
            let col = &mat[i];
            let commit_col = msm_g1::<E>(
                &vec_g, &col);
            tier_one_vec.push(commit_col);
        }
        let tier1_time = timer.elapsed().as_secs_f64();
        println!("Tier 1 time: {:?}s", tier1_time);

        let result = inner_pairing_product(
            &tier_one_vec, &vec_h)
            + pp.tilde_u.mul(hiding_factor);

        end_timer!(commit_time);
        Ok((result,tier_one_vec))
    }

    /// Outputs a commitment to a matrix
    /// and intermediate tier-one commitment
    pub fn commit_short(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<i32>>,
        hiding_factor: E::ScalarField,
        k: usize,
    ) -> Result<(PairingOutput<E>, Vec<E::G1>), Error> {

        let n = mat.len();
        let m = mat[0].len();

        let commit_time = start_timer!(|| format!(
            "Committing to {:?}-vector:",
            n * m
        ));

        let vec_g = pp.vec_g[0..m].to_vec();
        let vec_h = pp.vec_h[0..n].to_vec();

        let timer = Instant::now();
        let prepare_g =
            prepare_base_short_g1::<E>(&vec_g, k);
        let mut tier_one_vec = Vec::new();
        for i in 0..n {
            let col = &mat[i];
            let commit_col = msm_g1_short_i32::<E>(
                &prepare_g, &col, k);
            tier_one_vec.push(commit_col);
        }
        let tier1_time = timer.elapsed().as_secs_f64();
        println!("Tier 1 time in short bit: {:?}s", tier1_time);

        let result = inner_pairing_product(
            &tier_one_vec, &vec_h)
            + pp.tilde_u.mul(hiding_factor);

        end_timer!(commit_time);
        Ok((result,tier_one_vec))
    }

    /// Outputs a commitment to a boolean matrix
    /// and intermediate tier-one commitment
    pub fn commit_boolean(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<bool>>,
        hiding_factor: E::ScalarField,
    ) -> Result<(PairingOutput<E>, Vec<E::G1>), Error> {

        let n = mat.len();
        let m = mat[0].len();

        let commit_time = start_timer!(|| format!(
            "Committing to boolean vector with {:?} coefficients:",
            m * n
        ));

        let vec_g = pp.vec_g[0..m].to_vec();
        let vec_h = pp.vec_h[0..n].to_vec();

        let timer = Instant::now();
        let mut tier_one_vec = Vec::new();
        for i in 0..n {
            let col = &mat[i];
            let commit_col = boolean_msm_g1::<E>(
                &vec_g, &col);
            tier_one_vec.push(commit_col);
        }
        let tier1_time = timer.elapsed().as_secs_f64();
        println!("Tier 1 time in boolean: {:?}s", tier1_time);

        let result = inner_pairing_product(
            &tier_one_vec, &vec_h)
            + pp.tilde_u.mul(hiding_factor);

        end_timer!(commit_time);

        end_timer!(commit_time);
        Ok((result,tier_one_vec))
    }

    /// On input a polynomial `p` and a `point`, outputs a [`Proof`] for the same.
    pub fn eval(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<E::ScalarField>>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
    ) -> (PairingOutput<E>, E::ScalarField) {
        
        let rng = &mut thread_rng();
        
        let v_tilde = E::ScalarField::rand(rng);

        let l_vec = xi_from_challenges::<E>(&xl);
        let r_vec = xi_from_challenges::<E>(&xr);

        let la = proj_left::<E>(mat, &l_vec);
        let v = inner_product::<E>(&la, &r_vec);

        let com_v = pp.u.mul(&v) + pp.tilde_u.mul(&v_tilde);
        
        (com_v, v_tilde)
    }

    /// On input a polynomial `p` and a `point`, outputs a [`Proof`] for the same.
    pub fn open (
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<E::ScalarField>>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
        v_com: PairingOutput<E>,
        mat_com: PairingOutput<E>,
        tier1: &Vec<E::G1>,
        v_tilde: E::ScalarField,
        mat_tilde: E::ScalarField,
    ) -> Result<Trans<E>, Error> {
        if mat.is_empty() || mat[0].is_empty() {
            return Err(Error::IncorrectInputLength(
                "SmartPC matrix must be non-empty".to_string(),
            ));
        }
        let n = mat.len();
        let m = mat[0].len();
        if !m.is_power_of_two()
            || !n.is_power_of_two()
            || mat.iter().any(|column| column.len() != m)
            || tier1.len() < n
            || pp.vec_g.len() < m
            || pp.vec_h.len() < n
            || xl.len() != m.ilog2() as usize
            || xr.len() != n.ilog2() as usize
        {
            return Err(Error::IncorrectInputLength(
                "SmartPC opening inputs have inconsistent dimensions".to_string(),
            ));
        }

        let mut fs = FiatShamir::new();
        fs.push(&mat_com);
        fs.push(&v_com);
        fs.push(xl);
        fs.push(xr);
        let x = fs.gen_challenge::<E>();
        let log_m = m.ilog2() as usize;
        let log_n = n.ilog2() as usize;
        let rng = &mut thread_rng();
        let u_0 = E::pairing(pp.g_0, pp.h_hat);
        let l_vec = xi_from_challenges::<E>(xl);
        let r_vec = xi_from_challenges::<E>(xr);
        let mut accumulated_hiding = v_tilde + x * mat_tilde;

        let mut vec_l_gt = Vec::with_capacity(log_n);
        let mut vec_r_gt = Vec::with_capacity(log_n);
        let mut capital_a_current = tier1[..n].to_vec();
        let mut h_vec_current = pp.vec_h[..n].to_vec();
        let mut r_current = r_vec[..n].to_vec();
        let mut v_current = proj_left::<E>(mat, &l_vec);
        let mut challenges_n = Vec::with_capacity(log_n);
        let mut challenges_inv_n = Vec::with_capacity(log_n);

        for round in 0..log_n {
            let current_len = n >> round;
            let midpoint = current_len / 2;
            let v_left = v_current[..midpoint].to_vec();
            let v_right = v_current[midpoint..current_len].to_vec();
            let capital_a_left = capital_a_current[..midpoint].to_vec();
            let capital_a_right = capital_a_current[midpoint..current_len].to_vec();
            let r_left = r_current[..midpoint].to_vec();
            let r_right = r_current[midpoint..current_len].to_vec();
            let h_left = h_vec_current[..midpoint].to_vec();
            let h_right = h_vec_current[midpoint..current_len].to_vec();

            let left_hiding = E::ScalarField::rand(rng);
            let right_hiding = E::ScalarField::rand(rng);
            let left = inner_pairing_product(&capital_a_left, &h_right).mul(&x)
                + u_0.mul(&inner_product::<E>(&v_left, &r_right))
                + pp.tilde_u.mul(left_hiding);
            let right = inner_pairing_product(&capital_a_right, &h_left).mul(&x)
                + u_0.mul(&inner_product::<E>(&v_right, &r_left))
                + pp.tilde_u.mul(right_hiding);
            vec_l_gt.push(left);
            vec_r_gt.push(right);
            fs.push(&left);
            fs.push(&right);

            let challenge = fs.gen_nonzero_challenge::<E>();
            let challenge_inv = challenge.inverse().unwrap();
            accumulated_hiding +=
                left_hiding * challenge + right_hiding * challenge_inv;
            challenges_n.push(challenge);
            challenges_inv_n.push(challenge_inv);
            v_current = add_vec_zp::<E>(
                &v_left,
                &scalar_mul_vec_zp::<E>(&v_right, &challenge_inv),
            );
            capital_a_current = add_vec_g1::<E>(
                &capital_a_left,
                &scalar_mul_vec_g1::<E>(&capital_a_right, &challenge_inv),
            );
            h_vec_current = add_vec_g2::<E>(
                &h_left,
                &scalar_mul_vec_g2::<E>(&h_right, &challenge),
            );
            r_current = add_vec_zp::<E>(
                &r_left,
                &scalar_mul_vec_zp::<E>(&r_right, &challenge),
            );
        }

        let h_reduce = h_vec_current[0];
        let r_reduce = r_current[0];
        let xi_n_inv = xi_from_challenges::<E>(&challenges_inv_n);
        let mut a_current = proj_right::<E>(mat, &xi_n_inv);
        let mut g_vec_current = pp.vec_g[..m].to_vec();
        let mut l_current = l_vec[..m].to_vec();
        let mut challenges_m = Vec::with_capacity(log_m);
        let mut challenges_inv_m = Vec::with_capacity(log_m);
        let mut vec_l_native = Vec::with_capacity(log_m);
        let mut vec_r_native = Vec::with_capacity(log_m);

        for round in 0..log_m {
            let current_len = m >> round;
            let midpoint = current_len / 2;
            let a_left = a_current[..midpoint].to_vec();
            let a_right = a_current[midpoint..current_len].to_vec();
            let l_left = l_current[..midpoint].to_vec();
            let l_right = l_current[midpoint..current_len].to_vec();
            let g_left = g_vec_current[..midpoint].to_vec();
            let g_right = g_vec_current[midpoint..current_len].to_vec();

            let left_cancel = E::ScalarField::rand(rng);
            let left_hiding = E::ScalarField::rand(rng);
            let right_cancel = E::ScalarField::rand(rng);
            let right_hiding = E::ScalarField::rand(rng);
            let left = NativeFoldMessage::<E> {
                g1: msm_g1::<E>(&g_right, &a_left).mul(x) + pp.g_0.mul(left_cancel),
                g2: pp.h_hat.mul(
                    r_reduce * inner_product::<E>(&a_left, &l_right),
                ) - h_reduce.mul(left_cancel)
                    + pp.tilde_h.mul(left_hiding),
            };
            let right = NativeFoldMessage::<E> {
                g1: msm_g1::<E>(&g_left, &a_right).mul(x) + pp.g_0.mul(right_cancel),
                g2: pp.h_hat.mul(
                    r_reduce * inner_product::<E>(&a_right, &l_left),
                ) - h_reduce.mul(right_cancel)
                    + pp.tilde_h.mul(right_hiding),
            };
            vec_l_native.push(left);
            vec_r_native.push(right);
            fs.push(&left);
            fs.push(&right);

            let challenge = fs.gen_nonzero_challenge::<E>();
            let challenge_inv = challenge.inverse().unwrap();
            accumulated_hiding +=
                left_hiding * challenge + right_hiding * challenge_inv;
            challenges_m.push(challenge);
            challenges_inv_m.push(challenge_inv);
            a_current = add_vec_zp::<E>(
                &a_left,
                &scalar_mul_vec_zp::<E>(&a_right, &challenge_inv),
            );
            g_vec_current = add_vec_g1::<E>(
                &g_left,
                &scalar_mul_vec_g1::<E>(&g_right, &challenge),
            );
            l_current = add_vec_zp::<E>(
                &l_left,
                &scalar_mul_vec_zp::<E>(&l_right, &challenge),
            );
        }

        let a_reduce = a_current[0];
        let g_reduce = g_vec_current[0];
        let xi_l = xi_ip_from_challenges::<E>(xl, &challenges_m);
        let xi_r = xi_ip_from_challenges::<E>(xr, &challenges_n);
        let base_rhs = u_0.mul(xi_l * xi_r)
            + E::pairing(g_reduce.mul(x), h_reduce);

        fs.push(&g_reduce);
        fs.push(&h_reduce);
        let pip_g1_challenge = fs.gen_challenge::<E>();
        let (v_g_prime, w_g) = sub_protocols::pip_g1_prove_with_challenge::<E>(
            pp,
            &challenges_m,
            pip_g1_challenge,
        );
        fs.push(&v_g_prime);
        fs.push(&w_g);
        let pip_g2_challenge = fs.gen_challenge::<E>();
        let (v_h_prime, w_h) = sub_protocols::pip_g2_prove_with_challenge::<E>(
            pp,
            &challenges_n,
            pip_g2_challenge,
        );
        fs.push(&v_h_prime);
        fs.push(&w_h);
        let (linear_f, linear_z_a, linear_z_t) = sub_protocols::linear_prove::<E, _>(
            base_rhs,
            pp.tilde_u,
            a_reduce,
            accumulated_hiding,
            &mut fs,
            rng,
        );

        Ok(Trans::<E> {
            vec_l_gt,
            vec_r_gt,
            vec_l_native,
            vec_r_native,
            v_g: g_reduce,
            v_h: h_reduce,
            v_g_prime,
            v_h_prime,
            w_g,
            w_h,
            linear_f,
            linear_z_a,
            linear_z_t,
        })
    }

    /// Reduces SmartPC verification to one deferred linear pairing equation.
    ///
    /// This path records sparse source-group coefficients only. Callers may
    /// aggregate several returned equations before performing any pairing.
    pub fn verify_as_subprotocol(
        pp: &UniversalParams<E>,
        mat_com: PairingOutput<E>,
        v_com: PairingOutput<E>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
        proof: &Trans<E>,
    ) -> Result<SmartPcVerificationEquation<E>, Error> {
        let log_m = xl.len();
        let log_n = xr.len();
        if !proof.has_rounds(log_m, log_n)
            || !pp.q.is_power_of_two()
            || log_m > pp.q.ilog2() as usize
            || log_n > pp.q.ilog2() as usize
        {
            return Err(Error::IncorrectInputLength(
                "SmartPC proof has inconsistent folding dimensions".to_string(),
            ));
        }

        let mut fs = FiatShamir::new();
        fs.push(&mat_com);
        fs.push(&v_com);
        fs.push(xl);
        fs.push(xr);
        let x = fs.gen_challenge::<E>();
        let mut challenges_n = Vec::with_capacity(log_n);
        let mut challenges_inv_n = Vec::with_capacity(log_n);
        for (left, right) in proof.vec_l_gt.iter().zip(&proof.vec_r_gt) {
            fs.push(left);
            fs.push(right);
            let challenge = fs.gen_nonzero_challenge::<E>();
            challenges_n.push(challenge);
            challenges_inv_n.push(challenge.inverse().unwrap());
        }

        let mut challenges_m = Vec::with_capacity(log_m);
        let mut challenges_inv_m = Vec::with_capacity(log_m);
        for (left, right) in proof.vec_l_native.iter().zip(&proof.vec_r_native) {
            fs.push(left);
            fs.push(right);
            let challenge = fs.gen_nonzero_challenge::<E>();
            challenges_m.push(challenge);
            challenges_inv_m.push(challenge.inverse().unwrap());
        }

        let xi_l = xi_ip_from_challenges::<E>(xl, &challenges_m);
        let xi_r = xi_ip_from_challenges::<E>(xr, &challenges_n);
        fs.push(&proof.v_g);
        fs.push(&proof.v_h);
        let pip_g1_challenge = fs.gen_challenge::<E>();
        fs.push(&proof.v_g_prime);
        fs.push(&proof.w_g);
        let pip_g2_challenge = fs.gen_challenge::<E>();
        fs.push(&proof.v_h_prime);
        fs.push(&proof.w_h);
        fs.push(&proof.linear_f);
        let linear_challenge = fs.gen_challenge::<E>();
        fs.push(&proof.linear_z_a);
        fs.push(&proof.linear_z_t);

        // Fiat-Shamir randomization is sampled only after every proof element.
        let batching_challenge = fs.gen_nonzero_challenge::<E>();
        let batching_challenge_2 = batching_challenge * batching_challenge;
        let batching_challenge_3 = batching_challenge_2 * batching_challenge;
        let batching_challenge_4 = batching_challenge_3 * batching_challenge;
        let phi_g = xi::phi_s::<E>(pip_g1_challenge, &challenges_m);
        let phi_h = xi::phi_s::<E>(pip_g2_challenge, &challenges_n);

        let mut equation = SmartPcVerificationEquation::new();
        equation.gt_terms.push((-E::ScalarField::one(), proof.linear_f));
        equation.gt_terms.push((-linear_challenge, v_com));
        equation
            .gt_terms
            .push((-(linear_challenge * x), mat_com));
        for ((left, right), (challenge, challenge_inv)) in proof
            .vec_l_gt
            .iter()
            .zip(&proof.vec_r_gt)
            .zip(challenges_n.iter().zip(&challenges_inv_n))
        {
            equation
                .gt_terms
                .push((-(linear_challenge * challenge), *left));
            equation
                .gt_terms
                .push((-(linear_challenge * challenge_inv), *right));
        }

        let mut reduced_g2_terms = vec![
            (xi_l * xi_r * proof.linear_z_a, pp.h_hat),
            (proof.linear_z_t, pp.tilde_h),
        ];
        let mut reduced_g1_terms = vec![
            (x * proof.linear_z_a, proof.v_g),
            (batching_challenge_3, pp.nu_g),
        ];
        for ((left, right), (challenge, challenge_inv)) in proof
            .vec_l_native
            .iter()
            .zip(&proof.vec_r_native)
            .zip(challenges_m.iter().zip(&challenges_inv_m))
        {
            reduced_g1_terms.push((-(linear_challenge * challenge), left.g1));
            reduced_g1_terms.push((-(linear_challenge * challenge_inv), right.g1));
            reduced_g2_terms.push((-(linear_challenge * challenge), left.g2));
            reduced_g2_terms.push((-(linear_challenge * challenge_inv), right.g2));
        }
        equation.pairing_rows.push(SparsePairingRow {
            g1_terms: vec![(E::ScalarField::one(), pp.g_0)],
            g2_terms: reduced_g2_terms,
        });
        equation.pairing_rows.push(SparsePairingRow {
            g1_terms: reduced_g1_terms,
            g2_terms: vec![(E::ScalarField::one(), proof.v_h)],
        });

        equation.pairing_rows.push(SparsePairingRow {
            g1_terms: vec![(batching_challenge, proof.v_g)],
            g2_terms: vec![(E::ScalarField::one(), pp.nu_h)],
        });
        equation.pairing_rows.push(SparsePairingRow {
            g1_terms: vec![
                (-batching_challenge, proof.v_g_prime),
                (batching_challenge_2, proof.v_g),
                (-(batching_challenge_2 * phi_g), pp.g_hat),
            ],
            g2_terms: vec![(E::ScalarField::one(), pp.h_hat)],
        });
        equation.pairing_rows.push(SparsePairingRow {
            g1_terms: vec![(batching_challenge_2, proof.w_g)],
            g2_terms: vec![
                (pip_g1_challenge, pp.h_hat),
                (-E::ScalarField::one(), pp.s_h),
            ],
        });
        equation.pairing_rows.push(SparsePairingRow {
            g1_terms: vec![(E::ScalarField::one(), pp.g_hat)],
            g2_terms: vec![
                (-batching_challenge_3, proof.v_h_prime),
                (batching_challenge_4, proof.v_h),
                (-(batching_challenge_4 * phi_h), pp.h_hat),
            ],
        });
        equation.pairing_rows.push(SparsePairingRow {
            g1_terms: vec![(batching_challenge_4, proof.w_h)],
            g2_terms: vec![
                (pip_g2_challenge, pp.h_hat),
                (-E::ScalarField::one(), pp.sq_h),
            ],
        });

        Ok(equation)
    }

    /// Verifies that `v_com` is the committed matrix evaluation at `(xl, xr)`.
    pub fn verify(
        pp: &UniversalParams<E>,
        mat_com: PairingOutput<E>,
        v_com: PairingOutput<E>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
        proof: &Trans<E>,
    ) -> Result<bool, Error> {
        let check_time = start_timer!(|| "Checking evaluation");
        let result = Self::verify_as_subprotocol(pp, mat_com, v_com, xl, xr, proof)?
            .evaluate();
        end_timer!(check_time, || format!("Result: {}", result));
        Ok(result)
    }

}




/// Experiment for SMART PC
pub fn experiment_smart_template<E>(num_vars: usize)
where
    E: Pairing,
    E::ScalarField: CanonicalSerialize,
    E::G1: CanonicalSerialize,
    E::G2: CanonicalSerialize,
    PairingOutput<E>: CanonicalSerialize,
{

    let rng = &mut ark_std::test_rng();

    let start_setup = Instant::now();
    let pp = SmartPC::<E>::setup(num_vars/2, rng).unwrap();

    let srs_size = pp.serialized_size(Compress::Yes);

    println!("SRS size: {:?} B", srs_size);

    let setup_time = start_setup.elapsed().as_secs_f64();
    println!("Setup time: {:?} s", setup_time);

    let n: usize = 1 << (num_vars/2);
    let k = 8;
    let mat: Vec<Vec<i32>> = (0..n).into_par_iter()
    .map(|_|{
        use rand::Rng;
        let mut rng = thread_rng();
        (0..n).into_iter().map(|_|{
            rng.gen_range(-128..128)
        }).collect()
    }).collect();

    let mat_scalar =
    convert_i32_to_scalar_mat::<E>(&mat);


    let hiding_factor = E::ScalarField::rand(rng);

    let start_commit = Instant::now();
    let comm_full =
        SmartPC::<E>::commit_short(&pp, &mat, hiding_factor, k).unwrap();
    let comm = comm_full.0;
    let commit_time = start_commit.elapsed().as_secs_f64();
    println!("Commit time: {:?} s", commit_time);


    let mut commit_writer = Vec::new();
    comm.serialize_compressed(&mut commit_writer).unwrap();
    let commit_size = commit_writer.len();
    println!("Commit size: {:?} B", commit_size);


    let xl =
    (0..num_vars/2).map(|_| E::ScalarField::rand(rng)).collect();
    let xr =
    (0..num_vars/2).map(|_| E::ScalarField::rand(rng)).collect();
    let (v_com, v_tilde) =
    SmartPC::<E>::eval(&pp, &mat_scalar, &xl, &xr);

    let start_open = Instant::now();
    let proof = SmartPC::<E>::open(
        &pp, &mat_scalar, &xl, &xr,
        v_com, comm_full.0, &comm_full.1, v_tilde, hiding_factor,
    );
    let proof = proof.unwrap();
    let open_time = start_open.elapsed().as_secs_f64();
    println!("Open time: {:?} s", open_time);

    let verify_time = Instant::now();
    let check = SmartPC::<E>::verify(
        &pp,
        comm,
        v_com,
        &xl,
        &xr,
        &proof,
    );
    let check = check.unwrap();
    let verify_time = verify_time.elapsed().as_secs_f64()*1000.0;
    // println!("Check: {:?}", check);
    println!("Verify time: {:?} ms", verify_time);

    let mut writer = Vec::new();
    proof.serialize_compressed(&mut writer).unwrap();
    let proof_size = writer.len();
    println!("Proof size: {:?} B", proof_size);

    println!(
        "SMART-PC  verified: {:?}",
        check
    );

    // Verify boolean matrix
    let boolean_mat = mat.iter().map(|row|{
        row.iter().map(|x| *x > 0).collect()
    }).collect();
    let boolean_mat_scalar =
    convert_boolean_to_scalar_mat::<E>(&boolean_mat);
    let start_commit_boolean = Instant::now();
    let comm_boolean =
        SmartPC::<E>::commit_boolean(&pp, &boolean_mat, hiding_factor).unwrap();
    let commit_time_boolean = start_commit_boolean.elapsed().as_secs_f64();
    println!("Commit time (boolean): {:?} s", commit_time_boolean);
    let (v_com, v_tilde) =
    SmartPC::<E>::eval(&pp, &boolean_mat_scalar, &xl, &xr);

    // let i32_mat_from_boolean: Vec<Vec<i32>> =
    // boolean_mat.iter().map(|row|{
    //     row.iter().map(|x| if *x {1} else {0}).collect()
    // }).collect();
    // let comm_check =
    // SmartPC::<E>::commit_short(&pp, &i32_mat_from_boolean, hiding_factor, k).unwrap();
    // assert_eq!(comm_boolean, comm_check);

    let proof = SmartPC::<E>::open(
        &pp, &boolean_mat_scalar, &xl, &xr,
        v_com, comm_boolean.0, &comm_boolean.1, v_tilde, hiding_factor,
    );
    let proof = proof.unwrap();

    let check = SmartPC::<E>::verify(
        &pp,
        comm_boolean.0,
        v_com,
        &xl,
        &xr,
        &proof,
    );
    let check = check.unwrap();
    println!(
        "SMART-PC (boolean)  verified: {:?}",
        check
    );
}

/// Test existing utils function
pub fn test_smart<E:Pairing>() {
    test_utils::<E>();
    // let mut rng = ark_std::test_rng();
    // let degree = 10;
    // let num_vars = 5;
    // let num_samples = 1;
    // let num_queries = 1;
    // let num_coefficients
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_serialize::CanonicalDeserialize;

    #[test]
    fn native_hiding_round_trip_has_expected_compressed_size() {
        let rng = &mut ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup(3, rng).unwrap();
        let matrix = (0..8)
            .map(|column| {
                (0..8)
                    .map(|row| Fr::from((column * 8 + row + 1) as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let matrix_hiding = Fr::from(9_u64);
        let (matrix_commitment, tier_one) =
            SmartPC::<Bls12_381>::commit_full(&pp, &matrix, matrix_hiding).unwrap();
        let xl = vec![Fr::from(2_u64), Fr::from(3_u64), Fr::from(4_u64)];
        let xr = vec![Fr::from(5_u64), Fr::from(6_u64), Fr::from(7_u64)];
        let (value_commitment, value_hiding) =
            SmartPC::<Bls12_381>::eval(&pp, &matrix, &xl, &xr);
        let proof = SmartPC::<Bls12_381>::open(
            &pp,
            &matrix,
            &xl,
            &xr,
            value_commitment,
            matrix_commitment,
            &tier_one,
            value_hiding,
            matrix_hiding,
        )
        .unwrap();

        assert!(SmartPC::<Bls12_381>::verify(
            &pp,
            matrix_commitment,
            value_commitment,
            &xl,
            &xr,
            &proof,
        )
        .unwrap());

        let mut alternate_xl = xl.clone();
        alternate_xl[0] += Fr::one();
        assert!(!SmartPC::<Bls12_381>::verify(
            &pp,
            matrix_commitment,
            value_commitment,
            &alternate_xl,
            &xr,
            &proof,
        )
        .unwrap());

        let deferred = SmartPC::<Bls12_381>::verify_as_subprotocol(
            &pp,
            matrix_commitment,
            value_commitment,
            &xl,
            &xr,
            &proof,
        )
        .unwrap();
        assert_eq!(deferred.gt_terms.len(), 3 + 2 * xr.len());
        assert_eq!(deferred.pairing_rows.len(), 7);
        assert!(deferred.evaluate());

        let mut encoded = Vec::new();
        proof.serialize_compressed(&mut encoded).unwrap();
        assert_eq!(proof.serialized_size(Compress::Yes), encoded.len());
        assert_eq!(encoded.len(), 5_376);

        let mut encoded_pp = Vec::new();
        pp.serialize_compressed(&mut encoded_pp).unwrap();
        let decoded_pp = UniversalParams::<Bls12_381>::deserialize_compressed(
            encoded_pp.as_slice(),
        )
        .unwrap();
        assert_eq!(decoded_pp, pp);
        assert_eq!(
            Bls12_381::pairing(decoded_pp.g_0, decoded_pp.tilde_h),
            decoded_pp.tilde_u,
        );
        let mut malformed_pp = pp.clone();
        malformed_pp.tilde_h += pp.h_hat;
        let mut malformed_pp_bytes = Vec::new();
        malformed_pp
            .serialize_compressed(&mut malformed_pp_bytes)
            .unwrap();
        assert!(UniversalParams::<Bls12_381>::deserialize_compressed(
            malformed_pp_bytes.as_slice(),
        )
        .is_err());

        let mut tampered_g1 = proof.clone();
        tampered_g1.vec_l_native[0].g1 += pp.g_hat;
        assert!(!SmartPC::<Bls12_381>::verify(
            &pp,
            matrix_commitment,
            value_commitment,
            &xl,
            &xr,
            &tampered_g1,
        )
        .unwrap());

        let mut tampered_g2 = proof.clone();
        tampered_g2.vec_r_native[0].g2 += pp.h_hat;
        assert!(!SmartPC::<Bls12_381>::verify(
            &pp,
            matrix_commitment,
            value_commitment,
            &xl,
            &xr,
            &tampered_g2,
        )
        .unwrap());

        let mut pairing_preserving_tamper = proof.clone();
        let delta = Fr::from(17_u64);
        pairing_preserving_tamper.vec_l_native[0].g1 += pp.g_0.mul(delta);
        pairing_preserving_tamper.vec_l_native[0].g2 -= proof.v_h.mul(delta);
        assert!(!SmartPC::<Bls12_381>::verify(
            &pp,
            matrix_commitment,
            value_commitment,
            &xl,
            &xr,
            &pairing_preserving_tamper,
        )
        .unwrap());

        let mut tampered_linear_response = proof.clone();
        tampered_linear_response.linear_z_t += Fr::one();
        assert!(!SmartPC::<Bls12_381>::verify(
            &pp,
            matrix_commitment,
            value_commitment,
            &xl,
            &xr,
            &tampered_linear_response,
        )
        .unwrap());

        let mut malformed = proof.clone();
        malformed.vec_l_native.pop();
        assert!(SmartPC::<Bls12_381>::verify_as_subprotocol(
            &pp,
            matrix_commitment,
            value_commitment,
            &xl,
            &xr,
            &malformed,
        )
        .is_err());
    }
}
