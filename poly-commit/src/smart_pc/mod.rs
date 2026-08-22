use std::time::Instant;

// use core::num;

use crate::Error;
use ark_ec::{pairing::{Pairing, PairingOutput}, CurveGroup};
    // scalar_mul::ScalarMul, AffineRepr, CurveGroup, VariableBaseMSM
// };
use ark_ff::{
    Zero,
    One,
    Field,
    UniformRand,
};
// use ark_poly::DenseUVPolynomial;
use ark_std::{
    // format,
    marker::PhantomData,
    rand::{thread_rng, RngCore},
    ops::Mul,
    vec::Vec,
};

use crate::MyInt;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use ark_serialize::{CanonicalSerialize, Compress};

/// Data structures for the SMART PC
pub mod data_structures;
pub use data_structures::*;

/// utilitiy functions
pub mod utils;

pub mod fiat_shamir;
pub use fiat_shamir::*;
use xi::{xi_from_challenges, xi_ip_from_challenges};
use utils::{
    inner_product, msm_g1, msm_g1_sparse_aware, inner_pairing_product, prepare_base_short_g1, msm_g1_short_myint,
    boolean_msm_g1, proj_left, proj_right, scalar_mul_vec_zp, add_vec_zp, scalar_mul_vec_g1,
    add_vec_g1, scalar_mul_vec_g2, add_vec_g2, convert_myint_to_scalar_mat,
    convert_boolean_to_scalar_mat, test_utils, is_zero_vec,
    proj_left_myint, proj_right_myint, proj_left_myint_flat, proj_right_myint_flat,
    msm_g1_myint_affine_nonparallel, myint_to_field,
};

/// subprotocols for the SMART PC
pub mod sub_protocols;
// use sub_protocols::*;

pub mod xi;

/// SmartPC for MyInt, with square matrix optimization encoded
pub struct SmartPCMyInt<E:Pairing> {
    _engine: PhantomData<E>,
}

impl<E> SmartPCMyInt<E>
where
    E: Pairing,
    E::ScalarField: CanonicalSerialize,
    E::G1: CanonicalSerialize,
    E::G2: CanonicalSerialize,
    PairingOutput<E>: CanonicalSerialize,
{
    /// Constructs public parameters when given as input the maximum degree `degree`
    pub fn setup<R: RngCore>(
        qlog: usize,
        rng: &mut R,
    ) -> Result<UniversalParams<E>, Error> {
        SmartPC::<E>::setup(qlog, rng)
    }

    /// Outputs a commitment to a matrix, convert to a square matrix first
    pub fn commit(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<MyInt>>,
    ) -> (PairingOutput<E>, Vec<E::G1>, E::ScalarField) {
        let hiding_factor = E::ScalarField::zero();
        let (com, tier1) = SmartPC::<E>::commit_square_myint(pp, mat, hiding_factor).unwrap();
        (com, tier1, hiding_factor)
    }

    /// Compute the
    pub fn eval(
        mat: &Vec<Vec<MyInt>>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
    ) -> E::ScalarField {
        let l_vec = xi_from_challenges::<E>(&xl);
        let r_vec = xi_from_challenges::<E>(&xr);

        let la = proj_left_myint::<E>(mat, &l_vec);
        let v = inner_product::<E>(&la, &r_vec);
        v
    }

    /// On input a polynomial `p` and a `point`, outputs a [`Proof`] for the same.
    pub fn open(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<MyInt>>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
        v: E::ScalarField,
        mat_com: PairingOutput<E>,
        tier1: &Vec<E::G1>,
        mat_tilde: E::ScalarField,
    ) -> Trans<E> {
        let com_v = pp.u.mul(&v);
        SmartPC::<E>::open_square_myint(pp, mat, xl, xr, com_v, mat_com, tier1, E::ScalarField::zero(), mat_tilde).unwrap()
    }

    /// On input a polynomial `p` and a `point`, verifies the [`Proof`] for the same.
    pub fn verify(
        pp: &UniversalParams<E>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
        v: E::ScalarField,
        mat_com: PairingOutput<E>,
        proof: &Trans<E>,
    ) -> bool {
        // NOTE: verify_square expects (mat_com, v_com) order
        let v_com = pp.u.mul(&v);
        SmartPC::<E>::verify_square(pp, mat_com, v_com, xl, xr, proof).unwrap()
    }
}


/// SMART_PC is an implementation of modified Dory
pub struct SmartPC<E: Pairing> {
    _engine: PhantomData<E>,
}

impl<E: Pairing> SmartPC<E> {
    /// Create a new SmartPC instance
    pub fn new() -> Self {
        Self {
            _engine: PhantomData,
        }
    }
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

        let q = 1usize << qlog;

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
        let powers_of_s = utils::powers_of_scalar::<E::ScalarField>(hat_s, q);

        let hat_sq = powers_of_s[q - 1] * hat_s;

        debug_assert_eq!(hat_sq, hat_s.pow([q as u64]));

        let powers_of_sq = utils::powers_of_scalar::<E::ScalarField>(hat_sq, q);

        let nu_powers_of_s = utils::scale_scalars::<E::ScalarField>(&powers_of_s, nu);
        let nu_powers_of_sq = utils::scale_scalars::<E::ScalarField>(&powers_of_sq, nu);

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

        // One windowed fixed-base table per generator replaces 5 * q independent
        // double-and-add ladders.
        let g_table = utils::fixed_base_table::<E::G1>(g_hat, q);
        let h_table = utils::fixed_base_table::<E::G2>(h_hat, q);

        let vec_g = utils::fixed_base_mul::<E::G1>(&g_table, &powers_of_s);
        let vec_h = utils::fixed_base_mul::<E::G2>(&h_table, &powers_of_sq);
        let vec_g_prime = utils::fixed_base_mul::<E::G1>(&g_table, &powers_of_sq);
        let vec_nu_g = utils::fixed_base_mul::<E::G1>(&g_table, &nu_powers_of_s);
        let vec_nu_h = utils::fixed_base_mul::<E::G2>(&h_table, &nu_powers_of_sq);

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
        // DEBUG: print generator pool sizes
        println!("[SmartPC::commit_full] vec_g_len={}, vec_h_len={}, input n={}, m={}", pp.vec_g.len(), pp.vec_h.len(), mat.len(), mat[0].len());
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
        let _tier1_time = timer.elapsed().as_secs_f64();
        // println!("Tier 1 time: {:?}s", tier1_time);

        let result = inner_pairing_product(
            &tier_one_vec, &vec_h)
            + pp.tilde_u.mul(hiding_factor);

        end_timer!(commit_time);
        Ok((result,tier_one_vec))
    }


    /// Outputs a commitment to a matrix
    /// and intermediate tier-one commitment
    pub fn commit_myint_via_full(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<MyInt>>,
        hiding_factor: E::ScalarField,
    ) -> Result<(PairingOutput<E>, Vec<E::G1>), Error> {
        // DEBUG: print generator pool sizes
        // println!("[SmartPC::commit_myint_via_full] vec_g_len=2^{}, vec_h_len=2^{}, input n=2^{}, m=2^{}", pp.vec_g.len().ilog2(), pp.vec_h.len().ilog2(), mat.len().ilog2(), mat[0].len().ilog2());
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
            if is_zero_vec(&mat[i]) {
                tier_one_vec.push(E::G1::zero());
                continue;
            } else {
                let col = &mat[i].clone().into_iter()
                    .map(|x| myint_to_field::<E::ScalarField>(x))
                    .collect::<Vec<_>>();
                let commit_col = msm_g1_sparse_aware::<E>(
                    &vec_g, &col);
                tier_one_vec.push(commit_col);
            }
        }
        let _tier1_time = timer.elapsed().as_secs_f64();
        // println!("Tier 1 time: {:?}s", tier1_time);

        let result = inner_pairing_product(
            &tier_one_vec, &vec_h)
            + pp.tilde_u.mul(hiding_factor);

        end_timer!(commit_time);
        Ok((result,tier_one_vec))
    }




    /// Commits after transforming into a square matrix
    pub fn commit_square_full(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<E::ScalarField>>,
        hiding_factor: E::ScalarField,
    ) -> Result<(PairingOutput<E>, Vec<E::G1>), Error> {
        let square = convert_to_square::<E>(mat);
        Self::commit_full(pp, &square, hiding_factor)
    }

    /// Commits after transforming into a square matrix
    pub fn commit_square_myint(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<MyInt>>,
        hiding_factor: E::ScalarField,
    ) -> Result<(PairingOutput<E>, Vec<E::G1>), Error> {
        let square = convert_to_square_myint(mat);
        println!("[SmartPC::commit_square_myint] vec_g_len=2^{}, vec_h_len=2^{}, input n=2^{}, m=2^{}", pp.vec_g.len().ilog2(), pp.vec_h.len().ilog2(), square.len().ilog2(), square[0].len().ilog2());
        Self::commit_myint_via_full(pp, &square, hiding_factor)
    }

    /// Commit to a huge 1×N vector without materializing the (almost) square matrix.
    /// Expects `flat` to be the flattened column-major vector (same as `into_flattened_vec`).
    pub fn commit_square_myint_flat(
        pp: &UniversalParams<E>,
        flat: &Vec<MyInt>,
        hiding_factor: E::ScalarField,
    ) -> Result<(PairingOutput<E>, Vec<E::G1>), Error> {
        let total = flat.len();
        if total == 0 { return Err(Error::DegreeIsZero); }

        // Match verify_square/convert_to_square: use floor-based split of log2(total)
        // If total is not perfectly divisible, pad with zeros to avoid out-of-bounds.
        let log_total = (total as u64).ilog2() as usize;
        let log_n_new = log_total / 2;
        let n_new = 1usize << log_n_new;
        let m_new = (total + n_new - 1) / n_new; // ceil division
        let total_padded = n_new * m_new;

        let mut flat_pad = flat.clone();
        flat_pad.resize(total_padded, 0);

        // Ensure SRS has enough generators
        if m_new > pp.vec_g.len() || n_new > pp.vec_h.len() {
            return Err(Error::IncorrectInputLength(
                "commit_square_myint_flat: SRS too small for padded matrix".to_string(),
            ));
        }

        // Generators trimmed to new square shape
        let vec_g = pp.vec_g[0..m_new].to_vec();
        let vec_h = pp.vec_h[0..n_new].to_vec();

        // Precompute affine bases once
        #[cfg(feature = "parallel")]
        let vec_g_affine = vec_g.par_iter()
            .map(|p| p.into_affine())
            .collect::<Vec<_>>();
        #[cfg(not(feature = "parallel"))]
        let vec_g_affine = vec_g.iter()
            .map(|p| p.into_affine())
            .collect::<Vec<_>>();

        // Tier-1 commitments per column (column-major storage: column j = flat[j*m_new..(j+1)*m_new])
        #[cfg(feature = "parallel")]
        let tier_one_vec: Vec<E::G1> = (0..n_new).into_par_iter().map(|j| {
            let start = j * m_new;
            let end = start + m_new;
            let col_myint: &[MyInt] = &flat_pad[start..end];
            msm_g1_myint_affine_nonparallel::<E>(&vec_g_affine, col_myint)
        }).collect();
        #[cfg(not(feature = "parallel"))]
        let tier_one_vec: Vec<E::G1> = (0..n_new).map(|j| {
            let start = j * m_new;
            let end = start + m_new;
            let col_myint: &[MyInt] = &flat_pad[start..end];
            msm_g1_myint_affine_nonparallel::<E>(&vec_g_affine, col_myint)
        }).collect();

    let result = inner_pairing_product(&tier_one_vec, &vec_h)
            + pp.tilde_u.mul(hiding_factor);
        Ok((result, tier_one_vec))
    }


    /// Outputs a commitment to a matrix
    /// and intermediate tier-one commitment
    pub fn commit_short(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<MyInt>>,
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
            let commit_col = msm_g1_short_myint::<E>(
                &prepare_g, &col, k);
            tier_one_vec.push(commit_col);
        }
        let _tier1_time = timer.elapsed().as_secs_f64();
        // println!("Tier 1 time in short bit: {:?}s", tier1_time);

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
        let _tier1_time = timer.elapsed().as_secs_f64();
        // println!("Tier 1 time in boolean: {:?}s", tier1_time);

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

        let rng = &mut ark_std::test_rng();

        let v_tilde = E::ScalarField::rand(rng);

        let l_vec = xi_from_challenges::<E>(&xl);
        let r_vec = xi_from_challenges::<E>(&xr);

        let la = proj_left::<E>(mat, &l_vec);
        let v = inner_product::<E>(&la, &r_vec);

        let com_v = pp.u.mul(&v) + pp.tilde_u.mul(&v_tilde);

        (com_v, v_tilde)
    }

    /// Evaluate a flat MyInt vector by reshaping into (almost) square and reordering challenges
    pub fn eval_my_int_flat(
        pp: &UniversalParams<E>,
        flat: &Vec<MyInt>,
        xl_raw: &Vec<E::ScalarField>,
        xr_raw: &Vec<E::ScalarField>,
    ) -> (PairingOutput<E>, E::ScalarField) {
        let total = flat.len();
        assert!(total > 0, "flat input should not be empty");

        // Derive square shape consistent with commit_square_myint_flat
        let log_total = (total as u64).ilog2() as usize;
        let log_n = log_total / 2;
        let n = 1usize << log_n;
        let m = total / n;

        // Reorder challenges as verify_square/open_square expect
        let mut xxxx = Vec::with_capacity(xr_raw.len() + xl_raw.len());
        xxxx.extend_from_slice(xr_raw);
        xxxx.extend_from_slice(xl_raw);
        let xl_new = xxxx[log_n..].to_vec();
        let xr_new = xxxx[..log_n].to_vec();

        let l_vec = xi_from_challenges::<E>(&xl_new);
        let r_vec = xi_from_challenges::<E>(&xr_new);

        // Memory-optimized projection directly on flat vector; no matrix reconstruction
        let la: Vec<E::ScalarField> = proj_left_myint_flat::<E>(flat, &l_vec, m, n);
        let v = inner_product::<E>(&la, &r_vec);

        let rng = &mut ark_std::test_rng();
        let v_tilde = E::ScalarField::rand(rng);
        let com_v = pp.u.mul(&v) + pp.tilde_u.mul(&v_tilde);

        (com_v, v_tilde)
    }


    /// Evaluate a flat MyInt vector by reshaping into (almost) square and reordering challenges
    pub fn eval_my_int_flat_output_field(
        flat: &Vec<MyInt>,
        xl_raw: &Vec<E::ScalarField>,
        xr_raw: &Vec<E::ScalarField>,
    ) -> E::ScalarField {
        let total = flat.len();
        assert!(total > 0, "flat input should not be empty");

        // Derive square shape consistent with commit_square_myint_flat
        let log_total = (total as u64).ilog2() as usize;
        let log_n = log_total / 2;
        let n = 1usize << log_n;
        let m = total / n;

        // Reorder challenges as verify_square/open_square expect
        let mut xxxx = Vec::with_capacity(xr_raw.len() + xl_raw.len());
        xxxx.extend_from_slice(xr_raw);
        xxxx.extend_from_slice(xl_raw);
        let xl_new = xxxx[log_n..].to_vec();
        let xr_new = xxxx[..log_n].to_vec();

        let l_vec = xi_from_challenges::<E>(&xl_new);
        let r_vec = xi_from_challenges::<E>(&xr_new);

        // Memory-optimized projection directly on flat vector; no matrix reconstruction
        let la: Vec<E::ScalarField> = proj_left_myint_flat::<E>(flat, &l_vec, m, n);
        let v = inner_product::<E>(&la, &r_vec);

        v
    }

    /// On input a polynomial `p` and a `point`, outputs a [`Proof`] for the same.
    pub fn open_square (
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
        let square = convert_to_square::<E>(mat);
        let (_m, n) = shape_to_square_shape((mat[0].len(), mat.len()));
        let xxxx = [xr.as_slice(), xl.as_slice()].concat();
        let log_n_new = (n as u64).ilog2() as usize;

        let xl_new = xxxx[log_n_new..].to_vec();
        let xr_new = xxxx[..log_n_new].to_vec();

        Self::open(pp, &square, &xl_new, &xr_new, v_com, mat_com, tier1, v_tilde, mat_tilde)
    }

        /// On input a polynomial `p` and a `point`, outputs a [`Proof`] for the same.
    pub fn open_square_myint (
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<MyInt>>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
        v_com: PairingOutput<E>,
        mat_com: PairingOutput<E>,
        tier1: &Vec<E::G1>,
        v_tilde: E::ScalarField,
        mat_tilde: E::ScalarField,
    ) -> Result<Trans<E>, Error> {
        let square = convert_to_square_myint(mat);
        let (_m, n) = shape_to_square_shape((mat[0].len(), mat.len()));
        let xxxx = [xr.as_slice(), xl.as_slice()].concat();
        let log_n_new = (n as u64).ilog2() as usize;

        let xl_new = xxxx[log_n_new..].to_vec();
        let xr_new = xxxx[..log_n_new].to_vec();

        Self::open_myint(pp, &square, &xl_new, &xr_new, v_com, mat_com, tier1, v_tilde, mat_tilde)
    }

    fn open_projected<F>(
        pp: &UniversalParams<E>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
        v_com: PairingOutput<E>,
        mat_com: PairingOutput<E>,
        tier1: &Vec<E::G1>,
        v_tilde: E::ScalarField,
        mat_tilde: E::ScalarField,
        m: usize,
        n: usize,
        l_vec: Vec<E::ScalarField>,
        r_vec: Vec<E::ScalarField>,
        mut v_current: Vec<E::ScalarField>,
        project_right: F,
    ) -> Result<Trans<E>, Error>
    where
        F: FnOnce(&Vec<E::ScalarField>) -> Vec<E::ScalarField>,
    {
        if !m.is_power_of_two()
            || !n.is_power_of_two()
            || tier1.len() < n
            || pp.vec_g.len() < m
            || pp.vec_h.len() < n
            || xl.len() != m.ilog2() as usize
            || xr.len() != n.ilog2() as usize
            || l_vec.len() != m
            || r_vec.len() != n
            || v_current.len() != n
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
        let mut accumulated_hiding = v_tilde + x * mat_tilde;

        let mut vec_l_gt = Vec::with_capacity(log_n);
        let mut vec_r_gt = Vec::with_capacity(log_n);
        let mut capital_a_current = tier1[..n].to_vec();
        let mut h_vec_current = pp.vec_h[..n].to_vec();
        let mut r_current = r_vec;
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
        let mut a_current = project_right(&xi_n_inv);
        if a_current.len() != m {
            return Err(Error::IncorrectInputLength(
                "SmartPC right projection has an inconsistent dimension".to_string(),
            ));
        }
        let mut g_vec_current = pp.vec_g[..m].to_vec();
        let mut l_current = l_vec;
        let mut challenges_m = Vec::with_capacity(log_m);
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
                g2: pp.h_hat.mul(r_reduce * inner_product::<E>(&a_left, &l_right))
                    - h_reduce.mul(left_cancel)
                    + pp.tilde_h.mul(left_hiding),
            };
            let right = NativeFoldMessage::<E> {
                g1: msm_g1::<E>(&g_left, &a_right).mul(x) + pp.g_0.mul(right_cancel),
                g2: pp.h_hat.mul(r_reduce * inner_product::<E>(&a_right, &l_left))
                    - h_reduce.mul(right_cancel)
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
        let base_rhs = u_0.mul(xi_l * xi_r) + E::pairing(g_reduce.mul(x), h_reduce);

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

    /// Opens a scalar-field matrix commitment.
    pub fn open(
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
        if mat.iter().any(|column| column.len() != m) {
            return Err(Error::IncorrectInputLength(
                "SmartPC matrix must be rectangular".to_string(),
            ));
        }
        let l_vec = xi_from_challenges::<E>(xl);
        let r_vec = xi_from_challenges::<E>(xr);
        let left = proj_left::<E>(mat, &l_vec);
        Self::open_projected(
            pp, xl, xr, v_com, mat_com, tier1, v_tilde, mat_tilde, m, n,
            l_vec, r_vec, left, |weights| proj_right::<E>(mat, weights),
        )
    }

    /// Opens a `MyInt` matrix commitment without converting the witness matrix.
    pub fn open_myint(
        pp: &UniversalParams<E>,
        mat: &Vec<Vec<MyInt>>,
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
        if mat.iter().any(|column| column.len() != m) {
            return Err(Error::IncorrectInputLength(
                "SmartPC matrix must be rectangular".to_string(),
            ));
        }
        let l_vec = xi_from_challenges::<E>(xl);
        let r_vec = xi_from_challenges::<E>(xr);
        let left = proj_left_myint::<E>(mat, &l_vec);
        Self::open_projected(
            pp, xl, xr, v_com, mat_com, tier1, v_tilde, mat_tilde, m, n,
            l_vec, r_vec, left, |weights| proj_right_myint::<E>(mat, weights),
        )
    }

    /// Opens a flat `MyInt` commitment without materializing its square matrix.
    pub fn open_square_myint_flat(
        pp: &UniversalParams<E>,
        flat: &Vec<MyInt>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
        v_com: PairingOutput<E>,
        mat_com: PairingOutput<E>,
        tier1: &Vec<E::G1>,
        v_tilde: E::ScalarField,
        mat_tilde: E::ScalarField,
    ) -> Result<Trans<E>, Error> {
        let total = flat.len();
        if total == 0 || !total.is_power_of_two() {
            return Err(Error::IncorrectInputLength(
                "SmartPC flat input must have a non-zero power-of-two length".to_string(),
            ));
        }
        let log_total = total.ilog2() as usize;
        let log_n = log_total / 2;
        let n = 1usize << log_n;
        let m = total / n;
        let mut challenges = Vec::with_capacity(xr.len() + xl.len());
        challenges.extend_from_slice(xr);
        challenges.extend_from_slice(xl);
        if challenges.len() != log_total {
            return Err(Error::IncorrectInputLength(
                "SmartPC flat opening point has an inconsistent dimension".to_string(),
            ));
        }
        let xl_new = challenges[log_n..].to_vec();
        let xr_new = challenges[..log_n].to_vec();
        let l_vec = xi_from_challenges::<E>(&xl_new);
        let r_vec = xi_from_challenges::<E>(&xr_new);
        let left = proj_left_myint_flat::<E>(flat, &l_vec, m, n);
        Self::open_projected(
            pp, &xl_new, &xr_new, v_com, mat_com, tier1, v_tilde, mat_tilde,
            m, n, l_vec, r_vec, left,
            |weights| proj_right_myint_flat::<E>(flat, weights, m, n),
        )
    }

    /// Verify as a square matrix
    pub fn verify_square(
        pp: &UniversalParams<E>,
        mat_com: PairingOutput<E>,
        v_com: PairingOutput<E>,
        xl: &Vec<E::ScalarField>,
        xr: &Vec<E::ScalarField>,
        proof: &Trans<E>,
    ) -> Result<bool, Error> {
        let m = 1usize << xl.len();
        let n = 1usize << xr.len();

        let (_m_new, n_new) = shape_to_square_shape((m, n));
        let log_n_new = (n_new as u64).ilog2() as usize;

        // Reconstruct the original challenge vector from xl and xr
        let mut xxxx = Vec::new();
        xxxx.extend_from_slice(xr); // right challenges first
        xxxx.extend_from_slice(xl); // then left challenges

        // Reorganize for the new square shape
        let xl_new = xxxx[log_n_new..].to_vec();
        let xr_new = xxxx[..log_n_new].to_vec();

        Self::verify(pp, mat_com, v_com, &xl_new, &xr_new, proof)
    }

    /// Reduces SmartPC verification to one deferred sparse pairing equation.
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

    /// Verifies a SmartPC opening by evaluating its deferred sparse equation.
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
    #[cfg(feature = "parallel")]
    let mat: Vec<Vec<MyInt>> = (0..n).into_par_iter()
    .map(|_|{
        use ark_std::{rand::Rng, test_rng};
        let mut rng = test_rng();
        (0..n).into_iter().map(|_|{
            rng.gen_range(-128..=127)
        }).collect()
    }).collect();
    #[cfg(not(feature = "parallel"))]
    let mat: Vec<Vec<MyInt>> = (0..n).into_iter()
    .map(|_|{
        use ark_std::{rand::Rng, test_rng};
        let mut rng = test_rng();
        (0..n).into_iter().map(|_|{
            rng.gen_range(-128..=127)
        }).collect()
    }).collect();

    let mat_scalar =
    convert_myint_to_scalar_mat::<E>(&mat);


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

    // let myint_mat_from_boolean: Vec<Vec<MyInt>> =
    // boolean_mat.iter().map(|row|{
    //     row.iter().map(|x| if *x {1} else {0}).collect()
    // }).collect();
    // let comm_check =
    // SmartPC::<E>::commit_short(&pp, &myint_mat_from_boolean, hiding_factor, k).unwrap();
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


/// Reshape a col major matrix into an (almost) square col major matrix
/// such that the projection keep unchanged
///
pub fn convert_to_square<E: Pairing>(
    a: &Vec<Vec<E::ScalarField>>,
) -> Vec<Vec<E::ScalarField>>
{
    let n = a.len();
    let m = a[0].len();

    let log_n_new = ((m * n).ilog2()/2) as usize;

    let n_new = 1usize << log_n_new;
    let m_new = m * n / n_new;

    // (0..n_new).into_iter().map(|j| {
    //     (0..m_new).into_par_iter().map(|i| {
    //         a[(i * n_new + j) % n][(i * n_new + j) / n]
    //     }).collect()
    // }).collect()

    (0..n_new).into_iter().map(|j| {       // j = new col
        (0..m_new).into_par_iter().map(|i| { // i = new row
            // Flattened index 'k' based on new matrix position (column-major)
            let k = j * m_new + i;
            // Find the element at the same flattened position 'k' in the old matrix (column-major)
            let old_col = k / m;
            let old_row = k % m;
            a[old_col][old_row]
        }).collect()
    }).collect()
}

/// Reshape a col major matrix into a (almost) square col major matrix
/// such that the projection keep unchanged
///
pub fn convert_to_square_myint(
    a: &Vec<Vec<MyInt>>,
) -> Vec<Vec<MyInt>>
{
    let n = a.len();
    let m = a[0].len();

    let log_n_new = ((m * n).ilog2()/2) as usize;

    let n_new = 1usize << log_n_new;
    let m_new = m * n / n_new;

    (0..n_new).into_iter().map(|j| {       // j = new col
        (0..m_new).into_par_iter().map(|i| { // i = new row
            // Flattened index 'k' based on new matrix position (column-major)
            let k = j * m_new + i;
            // Find the element at the same flattened position 'k' in the old matrix (column-major)
            let old_col = k / m;
            let old_row = k % m;
            a[old_col][old_row]
        }).collect()
    }).collect()
}

/// Compute the new shape after converting into a square matrix
pub fn shape_to_square_shape((m, n): (usize, usize)) -> (usize, usize) {
    let log_n_new = ((m * n).ilog2()/2) as usize;
    let n_new = 1usize << log_n_new;
    let m_new = m * n / n_new;
    (m_new, n_new)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::{Bls12_381, Fr as BlsFr};
    use ark_ff::UniformRand;
    use ark_serialize::CanonicalDeserialize;
    use ark_std::test_rng;

    #[test]
    fn native_hiding_round_trip_has_expected_compressed_size() {
        let rng = &mut test_rng();
        let pp = SmartPC::<Bls12_381>::setup(3, rng).unwrap();
        let matrix = (0..8)
            .map(|column| {
                (0..8)
                    .map(|row| BlsFr::from((column * 8 + row + 1) as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let matrix_hiding = BlsFr::from(9_u64);
        let (matrix_commitment, tier_one) =
            SmartPC::<Bls12_381>::commit_full(&pp, &matrix, matrix_hiding).unwrap();
        let xl = vec![
            BlsFr::from(2_u64),
            BlsFr::from(3_u64),
            BlsFr::from(4_u64),
        ];
        let xr = vec![
            BlsFr::from(5_u64),
            BlsFr::from(6_u64),
            BlsFr::from(7_u64),
        ];
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
        let decoded_pp =
            UniversalParams::<Bls12_381>::deserialize_compressed(encoded_pp.as_slice()).unwrap();
        assert_eq!(decoded_pp, pp);
        assert_eq!(
            Bls12_381::pairing(decoded_pp.g_0, decoded_pp.tilde_h),
            decoded_pp.tilde_u,
        );

        let mut alternate_xl = xl.clone();
        alternate_xl[0] += BlsFr::one();
        assert!(!SmartPC::<Bls12_381>::verify(
            &pp,
            matrix_commitment,
            value_commitment,
            &alternate_xl,
            &xr,
            &proof,
        )
        .unwrap());

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

    #[test]
    fn test_pc_commit_square_matrix_workflow() {
        let mut rng = test_rng();

        // Test parameters
        let qlog = 12; // q = 4096
        let pp = SmartPC::<Bls12_381>::setup(qlog, &mut rng).unwrap();

        // Create a test matrix as a long vector (simulating concatenated matrices)
        let test_data = vec![BlsFr::from(1u64), BlsFr::from(2u64), BlsFr::from(3u64), BlsFr::from(4u64)];
        let test_matrix = vec![test_data]; // 1×4 matrix

        println!("Original matrix shape: {}×{}", test_matrix.len(), test_matrix[0].len());


        // Generate random hiding factors
        let hiding_factor = BlsFr::rand(&mut rng);

        // Test commit_square_full
        let (commitment, tier1) = SmartPC::<Bls12_381>::commit_square_full(&pp, &test_matrix, hiding_factor).unwrap();
        println!("✓ Commitment succeeded");

        // Raw challenges sized for original (m_raw=1, n_raw=4) layout: we treat columns-length as rows after conversion
        // Original logical shape before square conversion: m_raw = test_matrix.len() = 1 (columns), n_raw = test_matrix[0].len() = 4 (rows)
        let m_raw = test_matrix[0].len(); // we stored data as single column-major row vector
        let n_raw = test_matrix.len();
        let log_rows = (m_raw as u64).ilog2() as usize; // will become part of left
        let log_cols = (n_raw as u64).ilog2() as usize; // zero here
        let xl_raw: Vec<BlsFr> = (0..log_rows).map(|_| BlsFr::rand(&mut rng)).collect();
        let xr_raw: Vec<BlsFr> = (0..log_cols).map(|_| BlsFr::rand(&mut rng)).collect();
        println!("Raw challenge vector lengths: xl_raw={}, xr_raw={}", xl_raw.len(), xr_raw.len());

        // Reproduce open_square transformation locally so evaluation aligns with commitment/open
        // 1. Convert to square
        let square = convert_to_square::<Bls12_381>(&test_matrix);
        // 2. Determine new right dimension bits (same formula as open_square)
        let (_m_new, n_new) = shape_to_square_shape((test_matrix[0].len(), test_matrix.len()));
        let log_n_new = (n_new as u64).ilog2() as usize;
        // 3. Concatenate original xr_raw || xl_raw then split
        let mut xxxx = Vec::new();
        xxxx.extend_from_slice(&xr_raw);
        xxxx.extend_from_slice(&xl_raw);
        let xl_new = xxxx[log_n_new..].to_vec();
        let xr_new = xxxx[..log_n_new].to_vec();
        println!("Transformed challenge lengths: xl_new={}, xr_new={}", xl_new.len(), xr_new.len());

        // 4. Evaluate square matrix at transformed point using existing eval
        let (v_com, v_tilde) = SmartPC::<Bls12_381>::eval(&pp, &test_matrix, &xl_raw, &xr_raw);
        let (v_com_new, v_tilde_new) = SmartPC::<Bls12_381>::eval(&pp, &square, &xl_new, &xr_new);

        assert_eq!(v_com - v_com_new, pp.tilde_u.mul(v_tilde - v_tilde_new), "Evaluation mismatch after square conversion");

        // Test open_square - this should work with the original test_matrix
        let open_result = SmartPC::<Bls12_381>::open_square(
            &pp,
            &test_matrix, // original non-square (open_square converts internally)
            &xl_raw,
            &xr_raw,
            v_com,
            commitment,
            &tier1,
            v_tilde,
            hiding_factor,
        );

        match open_result {
            Ok(proof) => {
                println!("✓ Opening succeeded");

                // Test verify_square
                let verify_result = SmartPC::<Bls12_381>::verify_square(
                    &pp,
                    commitment,
                    v_com,
                    &xl_raw,
                    &xr_raw,
                    &proof,
                );

                match verify_result {
                    Ok(true) => println!("✓ Verification succeeded"),
                    Ok(false) => panic!("✗ Verification failed"),
                    Err(e) => panic!("✗ Verification error: {:?}", e),
                }
            },
            Err(e) => panic!("✗ Opening failed: {:?}", e),
        }
    }

    #[test]
    fn test_pc_commit_square_vector_matrix_workflow() {
        let mut rng = test_rng();

        // Test parameters
        let qlog = 12; // q = 4096
        let pp = SmartPC::<Bls12_381>::setup(qlog, &mut rng).unwrap();

        // Create a test matrix as a long vector (simulating concatenated matrices)
        let test_data = vec![BlsFr::from(1u64), BlsFr::from(2u64), BlsFr::from(3u64), BlsFr::from(4u64), BlsFr::from(5u64), BlsFr::from(6u64), BlsFr::from(7u64), BlsFr::from(8u64)];
        let test_matrix = vec![test_data]; // 1×4 matrix

        println!("Original matrix shape: {}×{}", test_matrix.len(), test_matrix[0].len());


        // Generate random hiding factors
        let hiding_factor = BlsFr::rand(&mut rng);

        // Test commit_square_full
        let (commitment, tier1) = SmartPC::<Bls12_381>::commit_square_full(&pp, &test_matrix, hiding_factor).unwrap();
        println!("✓ Commitment succeeded");

        // Raw challenges sized for original (m_raw=1, n_raw=4) layout: we treat columns-length as rows after conversion
        // Original logical shape before square conversion: m_raw = test_matrix.len() = 1 (columns), n_raw = test_matrix[0].len() = 4 (rows)
        let m_raw = test_matrix[0].len(); // we stored data as single column-major row vector
        let n_raw = test_matrix.len();
        let log_rows = (m_raw as u64).ilog2() as usize; // will become part of left
        let log_cols = (n_raw as u64).ilog2() as usize; // zero here
        let xl_raw: Vec<BlsFr> = (0..log_rows).map(|_| BlsFr::rand(&mut rng)).collect();
        let xr_raw: Vec<BlsFr> = (0..log_cols).map(|_| BlsFr::rand(&mut rng)).collect();
        println!("Raw challenge vector lengths: xl_raw={}, xr_raw={}", xl_raw.len(), xr_raw.len());

        // Reproduce open_square transformation locally so evaluation aligns with commitment/open
        // 1. Convert to square
        let square = convert_to_square::<Bls12_381>(&test_matrix);
        // 2. Determine new right dimension bits (same formula as open_square)
        let (_m_new, n_new) = shape_to_square_shape((test_matrix[0].len(), test_matrix.len()));
        let log_n_new = (n_new as u64).ilog2() as usize;
        // 3. Concatenate original xr_raw || xl_raw then split
        let mut xxxx = Vec::new();
        xxxx.extend_from_slice(&xr_raw);
        xxxx.extend_from_slice(&xl_raw);
        let xl_new = xxxx[log_n_new..].to_vec();
        let xr_new = xxxx[..log_n_new].to_vec();
        println!("Transformed challenge lengths: xl_new={}, xr_new={}", xl_new.len(), xr_new.len());

        // 4. Evaluate square matrix at transformed point using existing eval
        let (v_com, v_tilde) = SmartPC::<Bls12_381>::eval(&pp, &test_matrix, &xl_raw, &xr_raw);
        let (v_com_new, v_tilde_new) = SmartPC::<Bls12_381>::eval(&pp, &square, &xl_new, &xr_new);

        assert_eq!(v_com - v_com_new, pp.tilde_u.mul(v_tilde - v_tilde_new), "Evaluation mismatch after square conversion");

        // Test open_square - this should work with the original test_matrix
        let open_result = SmartPC::<Bls12_381>::open_square(
            &pp,
            &test_matrix, // original non-square (open_square converts internally)
            &xl_raw,
            &xr_raw,
            v_com,
            commitment,
            &tier1,
            v_tilde,
            hiding_factor,
        );

        match open_result {
            Ok(proof) => {
                println!("✓ Opening succeeded");

                // Test verify_square
                let verify_result = SmartPC::<Bls12_381>::verify_square(
                    &pp,
                    commitment,
                    v_com,
                    &xl_raw,
                    &xr_raw,
                    &proof,
                );

                match verify_result {
                    Ok(true) => println!("✓ Verification succeeded"),
                    Ok(false) => panic!("✗ Verification failed"),
                    Err(e) => panic!("✗ Verification error: {:?}", e),
                }
            },
            Err(e) => panic!("✗ Opening failed: {:?}", e),
        }
    }

    #[test]
    fn test_pc_commit_square_challenge_vector_reorganization() {
        // Test the challenge vector reorganization logic in verify_square
        let original_xl = vec![BlsFr::from(1u64), BlsFr::from(2u64), BlsFr::from(3u64)]; // 3 bits -> 8 rows
        let original_xr = vec![]; // 0 bits -> 1 column

        let m = 1 << original_xl.len(); // 8
        let n = 1 << original_xr.len(); // 1
        println!("Original shape: {}×{}", m, n);

        let (m_new, n_new) = shape_to_square_shape((m, n));
        println!("New shape: {}×{}", m_new, n_new);

        // Should be: 8×1 -> total=8, log=3, so 4×2 (log_right=1, log_left=2)
        assert_eq!(m_new, 4);
        assert_eq!(n_new, 2);

        let log_n_new = (n_new as u64).ilog2() as usize;
        assert_eq!(log_n_new, 1);

        // Reconstruct challenge vector
        let mut xxxx = Vec::new();
        xxxx.extend_from_slice(&original_xr); // empty
        xxxx.extend_from_slice(&original_xl); // [1, 2, 3]

        let xl_new = xxxx[log_n_new..].to_vec(); // should be [2, 3]
        let xr_new = xxxx[..log_n_new].to_vec(); // should be [1]

        println!("Original: xl={:?}, xr={:?}", original_xl, original_xr);
        println!("Reorganized: xl_new={:?}, xr_new={:?}", xl_new, xr_new);

        assert_eq!(xl_new.len(), 2);
        assert_eq!(xr_new.len(), 1);
        assert_eq!(xl_new, vec![BlsFr::from(2u64), BlsFr::from(3u64)]);
        assert_eq!(xr_new, vec![BlsFr::from(1u64)]);
    }

    #[test]
    fn test_pc_commit_square_convert_to_square() {
        // Test the convert_to_square function
        let test_matrix = vec![vec![1, 2, 3, 4, 5, 6, 7, 8]]; // 1×8 matrix

        let square = convert_to_square_myint(&test_matrix);

        println!("Original: 1×8 -> Square: {}×{}", square.len(), square[0].len());

        // 8 total elements -> 4×2 or 2×4, should choose more balanced
        // log_total = 3, log_right = 3/2 = 1, so 4×2
        assert_eq!(square.len(), 2); // columns
        assert_eq!(square[0].len(), 4); // rows

        println!("Square matrix:");
        for i in 0..square[0].len() {
            for j in 0..square.len() {
                print!("{} ", square[j][i]);
            }
            println!();
        }
    }

    #[test]
    fn test_pc_commit_flat_equals_square_commit_myint() {
        // Compare commit_square_myint (with convert_to_square_myint) vs commit_square_myint_flat
        // on the same flattened 1×N vector
        let mut rng = test_rng();
        let qlog = 8; // q = 256, ample for tiny shapes
        let pp = SmartPC::<Bls12_381>::setup(qlog, &mut rng).unwrap();

        // Deterministic small vector, length 8
    let flat: Vec<MyInt> = (1..=8).collect();

        // As a column-major 1×8 matrix (one column, 8 rows)
        let mat = vec![flat.clone()];

        let hide = BlsFr::rand(&mut rng);
        let (com_sq, tier1_sq) = SmartPC::<Bls12_381>::commit_square_myint(&pp, &mat, hide).unwrap();
        let (com_flat, tier1_flat) = SmartPC::<Bls12_381>::commit_square_myint_flat(&pp, &flat, hide).unwrap();

        assert_eq!(com_sq, com_flat, "Commitments must match between square and flat commits");
        assert_eq!(tier1_sq.len(), tier1_flat.len(), "Tier1 lengths must match");
        for (a, b) in tier1_sq.iter().zip(tier1_flat.iter()) {
            assert_eq!(a, b, "Tier1 per-column commitments should be identical");
        }
    }

    #[test]
    fn test_pc_open_flat_and_square_verify_true() {
        // Ensure both open_square_myint and open_square_myint_flat produce proofs that verify
        let mut rng = test_rng();
        let qlog = 10; // q = 1024 for a bit more room
        let pp = SmartPC::<Bls12_381>::setup(qlog, &mut rng).unwrap();

        // Build a small deterministic vector of length 16
        let flat: Vec<MyInt> = (1..=16).collect();
        let mat = vec![flat.clone()]; // 1×16

        // Random challenges sized for original (m_raw=16 -> log_m=4, n_raw=1 -> log_n=0)
        let xl_raw: Vec<BlsFr> = (0..4).map(|_| BlsFr::rand(&mut rng)).collect();
        let xr_raw: Vec<BlsFr> = vec![];

        let hide = BlsFr::rand(&mut rng);
        let (com_sq, tier1_sq) = SmartPC::<Bls12_381>::commit_square_myint(&pp, &mat, hide).unwrap();
        let (com_flat, tier1_flat) = SmartPC::<Bls12_381>::commit_square_myint_flat(&pp, &flat, hide).unwrap();
        assert_eq!(com_sq, com_flat);

        // Compute the actual evaluation v at the reorganized (xl_new, xr_new) to build a correct v_com
        // Reproduce the same challenge reorganization as open_square_myint/verify_square
        let (_m_new, n_new) = shape_to_square_shape((mat[0].len(), mat.len()));
        let log_n_new = (n_new as u64).ilog2() as usize;
        let mut xxxx = Vec::new();
        xxxx.extend_from_slice(&xr_raw);
        xxxx.extend_from_slice(&xl_raw);
        let xl_new = xxxx[log_n_new..].to_vec();
        let xr_new = xxxx[..log_n_new].to_vec();
        let square = convert_to_square_myint(&mat);
        let l_vec = xi_from_challenges::<Bls12_381>(&xl_new);
        let r_vec = xi_from_challenges::<Bls12_381>(&xr_new);
        let la = proj_left_myint::<Bls12_381>(&square, &l_vec);
        let v_hat = inner_product::<Bls12_381>(&la, &r_vec);
        let v_com = pp.u.mul(&v_hat);

        // Open via square path
        let proof_sq = SmartPC::<Bls12_381>::open_square_myint(
            &pp,
            &mat,
            &xl_raw,
            &xr_raw,
            v_com,
            com_sq,
            &tier1_sq,
            BlsFr::zero(),
            hide,
        ).unwrap();
        let ok_sq = SmartPC::<Bls12_381>::verify_square(&pp, com_sq, v_com, &xl_raw, &xr_raw, &proof_sq).unwrap();
        assert!(ok_sq, "verify_square should succeed for square open");

        // Open via flat path
        let proof_flat = SmartPC::<Bls12_381>::open_square_myint_flat(
            &pp,
            &flat,
            &xl_raw,
            &xr_raw,
            v_com,
            com_flat,
            &tier1_flat,
            BlsFr::zero(),
            hide,
        ).unwrap();
        let ok_flat = SmartPC::<Bls12_381>::verify_square(&pp, com_flat, v_com, &xl_raw, &xr_raw, &proof_flat).unwrap();
        assert!(ok_flat, "verify_square should succeed for flat open");
    }

    #[test]
    fn test_pc_column_major_layout_verification() {
        use ark_bls12_381::{Bls12_381, Fr as BlsFr};
        use ark_std::test_rng;

        let mut rng = test_rng();
        let qlog = 8;
        let pp = SmartPC::<Bls12_381>::setup(qlog, &mut rng).unwrap();

        // Create a small test case with known values
        let test_matrix = vec![
            vec![1, 2, 3, 4],      // Column 0
            vec![5, 6, 7, 8],      // Column 1
            vec![9, 10, 11, 12],   // Column 2
            vec![13, 14, 15, 16],  // Column 3
        ];

        // Expected flat vector (column-major): [1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16]
        let expected_flat = vec![1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16];

        // Create actual flat vector
        let mut actual_flat: Vec<MyInt> = Vec::new();
        for col in &test_matrix {
            actual_flat.extend_from_slice(col);
        }

        assert_eq!(actual_flat, expected_flat, "Flattening should be column-major");

        let hiding_factor = BlsFr::rand(&mut rng);

        // Test commitment
        let (com_standard, _) = SmartPC::<Bls12_381>::commit_square_myint(&pp, &test_matrix, hiding_factor).unwrap();
        let (com_flat, _) = SmartPC::<Bls12_381>::commit_square_myint_flat(&pp, &actual_flat, hiding_factor).unwrap();

        assert_eq!(com_standard, com_flat, "Column-major layout should produce identical commitments");

        println!("✓ Column-major layout verification passed!");
        println!("  - Original matrix: 4x4");
        println!("  - Flat representation: {:?}", actual_flat);
        println!("  - Commitment consistency: ✓");
    }

    #[test]
    fn test_pc_myint_commit_eval_open_verify_various_shapes() {
        // End-to-end validation across various column-major matrix shapes: commit -> eval -> open -> verify
        let mut rng = test_rng();
        let qlog = 12; // 2^12 generators; sufficient for the small shapes below
        let pp = SmartPCMyInt::<Bls12_381>::setup(qlog, &mut rng).expect("setup failed");

        // Matrix shapes in column-major order as (n_cols, n_rows)
        // Prefer powers of two to avoid truncation/rounding edge cases
        let shapes: &[(usize, usize)] = &[
            (1, 1),   // 1x1
            (1, 8),   // 1x8 vector
            (8, 1),   // 8x1 vector
            (4, 4),   // square
            (2, 8),   // non-square
            (8, 2),   // non-square
        ];

        for &(n_cols, n_rows) in shapes {
            // Build a column-major matrix with increasing entries 1..=N
            let total = n_cols * n_rows;
            let data: Vec<MyInt> = (1..=total as MyInt).collect();
            let mut mat: Vec<Vec<MyInt>> = Vec::with_capacity(n_cols);
            for j in 0..n_cols {
                let start = j * n_rows;
                let end = start + n_rows;
                mat.push(data[start..end].to_vec());
            }

            // 1) Commit (internally converts to a square shape and returns mat_tilde)
            let (mat_com, tier1, mat_tilde) = SmartPCMyInt::<Bls12_381>::commit(&pp, &mat);

            // 2) Eval: compute v directly on the original matrix (no square conversion)
            let log_m = if n_rows > 0 { (n_rows as u64).ilog2() as usize } else { 0 };
            let log_n = if n_cols > 0 { (n_cols as u64).ilog2() as usize } else { 0 };
            let xl_raw: Vec<BlsFr> = (0..log_m).map(|_| BlsFr::rand(&mut rng)).collect();
            let xr_raw: Vec<BlsFr> = (0..log_n).map(|_| BlsFr::rand(&mut rng)).collect();
            let v = SmartPCMyInt::<Bls12_381>::eval(&mat, &xl_raw, &xr_raw);

            // 3) Open: use mat_tilde from commit together with v from eval
            let proof = SmartPCMyInt::<Bls12_381>::open(
                &pp,
                &mat,
                &xl_raw,
                &xr_raw,
                v,
                mat_com,
                &tier1,
                mat_tilde,
            );

            // 4) Verify: use raw challenges (internal logic handles challenge reorganization/square processing)
            let ok = SmartPCMyInt::<Bls12_381>::verify(
                &pp,
                &xl_raw,
                &xr_raw,
                v,
                mat_com,
                &proof,
            );
            assert!(ok, "SmartPCMyInt verify failed for shape {}x{}", n_cols, n_rows);
        }
    }
}
