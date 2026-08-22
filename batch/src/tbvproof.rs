use ark_crypto_primitives::sponge::Absorb;
/// Implement the Proof for TBV reduction
///
use ark_ff::{PrimeField, Zero};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::{test_rng, UniformRand};
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use atomic_proof::{BatchPoints, Hadamard, LinComb, LiteBullet, MatMul};

use ark_ec::pairing::{Pairing, PairingOutput};
use poly_commit::smart_pc::{SmartPC as PC, Trans as PCTrans, UniversalParams as PP};

use mat::{xi, DenseMatFieldCM, MyInt};

use fs::helper_trans::Transcript;

use super::bva::BvaLayout;
use super::tbv::Bucket;
use super::tbv::TBV;
use super::utils::int_to_least_endian_bits;

pub trait TbvReduceProof<
    F: PrimeField + Absorb + UniformRand + CanonicalSerialize + CanonicalDeserialize + From<MyInt>,
>
{
    fn compute_mat_i(&self) -> DenseMatFieldCM<F>;

    fn compute_mat_z(&self, points: &Vec<Vec<F>>) -> DenseMatFieldCM<F>;

    fn compute_mat_x(
        &self,
        vec_y: &Vec<F>,
        mat_i: &DenseMatFieldCM<F>,
        mat_z: &DenseMatFieldCM<F>,
    ) -> DenseMatFieldCM<F>;

    fn compute_mat_w(&self, mat_x: &DenseMatFieldCM<F>) -> DenseMatFieldCM<F>;

    fn compute_mat_w_prime(
        &self,
        mat_x: &DenseMatFieldCM<F>,
        mat_w: &DenseMatFieldCM<F>,
    ) -> DenseMatFieldCM<F>;

    fn compute_vec_pi(&self, mat_x: &DenseMatFieldCM<F>) -> Vec<F>;

    fn compute_bar_b(&self, points: &Vec<Vec<F>>, omega: F, vec_y: &Vec<F>) -> F;

    fn check_relation(
        &self,
        mat_x: &DenseMatFieldCM<F>,
        mat_w: &DenseMatFieldCM<F>,
        mat_wp: &DenseMatFieldCM<F>,
        b_bar: F,
        omega: F,
    ) -> bool;

    fn check_directly(&self, points: &Vec<Vec<F>>, omega: F, vec_y: &Vec<F>) -> bool;

    fn reduce_prover_eq1(
        &self,
        trans_seq: &mut Transcript<F>,
        b_bar: F,
        omega: F,
        mat_wp: &DenseMatFieldCM<F>,
    ) -> (F, (Vec<F>, Vec<F>));

    fn reduce_prover_eq2(
        &self,
        trans_seq: &mut Transcript<F>,
        theta: F,
        mat_w: &DenseMatFieldCM<F>,
        mat_wp: &DenseMatFieldCM<F>,
    ) -> (F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>));

    fn reduce_prover_eq3(
        &self,
        trans_seq: &mut Transcript<F>,
        w_prime_value: F,
        w_prime_point: (Vec<F>, Vec<F>),
        mat_x: &DenseMatFieldCM<F>,
        mat_w: &DenseMatFieldCM<F>,
    ) -> (F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>));

    fn reduce_prover_eq4(
        &self,
        trans_seq: &mut Transcript<F>,
        mat_x_value: F,
        mat_x_point: (Vec<F>, Vec<F>),
        mat_i: &DenseMatFieldCM<F>,
        mat_z: &DenseMatFieldCM<F>,
        vec_y: &Vec<F>,
    ) -> (F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>));

    fn reduce_prover(
        &self,
        trans_seq: &mut Transcript<F>,
        b_bar: F,
        z_points: &Vec<Vec<F>>,
        omega: F,
        y_vec: &Vec<F>,
    ) -> (
        F,
        F,
        F,
        (Vec<F>, Vec<F>),
        (Vec<F>, Vec<F>),
        (Vec<F>, Vec<F>),
    );

    fn verify_eq1(
        &self,
        trans_seq: &mut Transcript<F>,
        b_bar: F,
        omega: F,
    ) -> (bool, F, (Vec<F>, Vec<F>));

    fn verify_eq2(
        &self,
        trans_seq: &mut Transcript<F>,
        theta: F,
    ) -> (bool, F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>));

    fn verify_eq3(
        &self,
        trans_seq: &mut Transcript<F>,
        w_prime_value: F,
        w_prime_point: (Vec<F>, Vec<F>),
    ) -> (bool, F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>));

    fn verify_eq4(
        &self,
        trans_seq: &mut Transcript<F>,
        mat_x_value: F,
        mat_x_point: (Vec<F>, Vec<F>),
        vec_y: &Vec<F>,
    ) -> (bool, F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>));

    fn verify_as_subprotocol(
        &self,
        trans_seq: &mut Transcript<F>,
        b_bar: F,
        omega: F,
        y_vec: &Vec<F>,
    ) -> (
        bool,
        F,
        F,
        F,
        (Vec<F>, Vec<F>),
        (Vec<F>, Vec<F>),
        (Vec<F>, Vec<F>),
    );
}

impl<
        F: PrimeField + Absorb + UniformRand + CanonicalSerialize + CanonicalDeserialize + From<MyInt>,
    > TbvReduceProof<F> for BvaLayout
{
    /// Matrix of indicator bits (padded).
    fn compute_mat_i(&self) -> DenseMatFieldCM<F> {
        let k_total = self.k_total;
        if !k_total.is_power_of_two() {
            panic!("invalid k_total: {}", k_total);
        }

        let log_k_total = k_total.next_power_of_two().ilog2() as usize;
        let num_vec = self.phi.len();

        let rows: Vec<Vec<F>> = (0..num_vec)
            .into_par_iter()
            .map(|i| {
                let k_i = self.bucket_sizes[self.phi[i]];
                let offset_i = self.offsets[i];

                if !k_i.is_power_of_two() {
                    panic!("invalid k_i: {}, offset_i: {}", k_i, offset_i);
                }
                if offset_i % k_i != 0 {
                    panic!("invalid k_i: {}, offset_i: {}", k_i, offset_i);
                }

                let log_k_i = k_i.ilog2() as usize;
                let mut bits_bool = int_to_least_endian_bits(offset_i, log_k_total);
                for j in 0..log_k_i {
                    bits_bool[j] = false;
                }

                bits_bool
                    .into_iter()
                    .map(|b| if b { F::one() } else { F::zero() })
                    .collect()
            })
            .collect();

        let mat_i_transpose = DenseMatFieldCM::<F>::from_data(rows);
        let mat_i = mat_i_transpose.par_transpose();

        mat_i
    }

    /// Matrix of padded z values.
    fn compute_mat_z(&self, points: &Vec<Vec<F>>) -> DenseMatFieldCM<F> {
        let k_total = self.k_total;
        let log_k_total = k_total.ilog2() as usize;

        let num_vec = self.phi.len();
        if points.len() != num_vec {
            panic!("invalid points: {}", points.len());
        }
        let mut data = Vec::<Vec<F>>::new();

        for i in 0..num_vec {
            // Align with TBV construction (gen_tbv_from_values_and_points) which reverses each z vector
            let mut cur_z_vec = points[i].clone();
            cur_z_vec.reverse();
            cur_z_vec.extend(vec![F::zero(); log_k_total - cur_z_vec.len()]);
            data.push(cur_z_vec);
        }

        let mat_z_transpose = DenseMatFieldCM::<F>::from_data(data);
        let mat_z = mat_z_transpose.par_transpose();
        mat_z
    }

    /// X = 1 - I + (Z + I) * y  (scalar y for simplicity)
    fn compute_mat_x(
        &self,
        vec_y: &Vec<F>,
        mat_i: &DenseMatFieldCM<F>,
        mat_z: &DenseMatFieldCM<F>,
    ) -> DenseMatFieldCM<F> {
        let (rows, cols) = mat_i.shape;
        if vec_y.len() != cols {
            panic!("invalid vec_y: {}", vec_y.len());
        }

        let mut data = vec![vec![F::zero(); rows]; cols];

        // Parallel computation for each column, avoiding intermediate large matrix
        use rayon::prelude::*;
        data.par_iter_mut().enumerate().for_each(|(c, col)| {
            let y_c = vec_y[c];
            for r in 0..rows {
                let i_val = mat_i.data[c][r];
                let z_val = mat_z.data[c][r];
                // x[r,c] = (1 - i[r,c]) + (z[r,c] + i[r,c]) * y[c]
                col[r] = (F::one() - i_val) + (z_val + i_val) * y_c;
            }
        });

        DenseMatFieldCM::from_data(data)
    }

    /// Compute the row proudct of the matrix X
    fn compute_vec_pi(&self, mat_x: &DenseMatFieldCM<F>) -> Vec<F> {
        let (rows, cols) = mat_x.shape;
        let mut pi = vec![F::one(); rows];
        for r in 0..rows {
            let mut acc = F::one();
            for c in 0..cols {
                acc *= mat_x.data[c][r];
            }
            pi[r] = acc;
        }
        pi
    }

    /// Compute \bar{b} via TBV inner product with xi_rev(challenges = y).
    fn compute_bar_b(&self, points: &Vec<Vec<F>>, omega: F, vec_y: &Vec<F>) -> F {
        // Build TBV the same way as BatchPCS::gen_tbv_from_values_and_points (values all zero; eta from omega powers)
        let num_points = points.len();
        let powers = std::iter::successors(Some(F::one()), |&x| Some(x * omega))
            .take(num_points)
            .collect::<Vec<_>>();

        let mut buckets = Vec::new();
        for (i, &k) in self.bucket_sizes.iter().enumerate() {
            if let Some(orig_idx) = self.phi_inv[i] {
                let mut z = points[orig_idx].clone();
                z.reverse(); // match gen_tbv_from_values_and_points behavior
                buckets.push(Bucket {
                    k,
                    eta: powers[orig_idx],
                    z,
                });
            } else {
                // padding bucket
                buckets.push(Bucket {
                    k,
                    eta: F::zero(),
                    z: vec![F::zero(); k.ilog2() as usize],
                });
            }
        }
        let tbv = TBV::new(self.k_total, buckets);
        tbv.ip_with_xi_rev_from_challenges(vec_y)
    }

    /// W_{i,0} = 1; W_{i,j} = prod_{r=0}^{j-1} X_{i,r}
    fn compute_mat_w(&self, mat_x: &DenseMatFieldCM<F>) -> DenseMatFieldCM<F> {
        let (rows, cols) = mat_x.shape;
        let mut data = vec![vec![F::one(); rows]; cols];
        for r in 0..rows {
            for c in 1..cols {
                data[c][r] = mat_x.data[c - 1][r] * data[c - 1][r];
            }
        }
        DenseMatFieldCM::<F>::from_data(data)
    }

    /// W' = X ∘ W
    fn compute_mat_w_prime(
        &self,
        mat_x: &DenseMatFieldCM<F>,
        mat_w: &DenseMatFieldCM<F>,
    ) -> DenseMatFieldCM<F> {
        mat_x.par_hadamard(mat_w)
    }

    /// Check the relation
    fn check_relation(
        &self,
        mat_x: &DenseMatFieldCM<F>,
        mat_w: &DenseMatFieldCM<F>,
        mat_wp: &DenseMatFieldCM<F>,
        b_bar: F,
        omega: F,
    ) -> bool {
        let mut rng = test_rng();
        let theta = F::rand(&mut rng);
        let (rows, cols) = mat_x.shape;
        let num_vec = rows;

        // 1) W' = X ∘ W  (Hadamard)
        let wp_expected = mat_x.par_hadamard(mat_w);
        if wp_expected.data != mat_wp.data {
            println!("[debug-proof] Condition 1 failed: W' != X ∘ W");
            return false;
        }

        // Precompute theta powers
        let theta_vec: Vec<F> = std::iter::successors(Some(F::one()), |x| Some(*x * theta))
            .take(cols)
            .collect();
        let mut theta_prime_vec = theta_vec.clone();
        if let Some(last) = theta_prime_vec.last_mut() {
            *last = F::zero();
        }

        let theta_mat = DenseMatFieldCM::<F>::from_data(vec![theta_vec]);
        let theta_mat_prime = DenseMatFieldCM::<F>::from_data(vec![theta_prime_vec]);

        // vec_one_mat: rows x 1 of ones
        let vec_one_mat = DenseMatFieldCM::<F>::from_data(vec![vec![F::one(); num_vec]]);

        // 2) W*theta - 1 = theta * W' * theta'
        let mat_w_mul_theta = mat_w.par_mul(&theta_mat); // (rows x 1)
        let mat_wp_mul_theta_prime = mat_wp.par_mul(&theta_mat_prime); // (rows x 1)

        let mat_w_mul_theta_minus_vec_one = mat_w_mul_theta.par_sub(&vec_one_mat);
        let theta_mult_mat_wp_mul_theta_prime = mat_wp_mul_theta_prime.par_scalar_mul(theta);

        let check_3_mat = mat_w_mul_theta_minus_vec_one.par_sub(&theta_mult_mat_wp_mul_theta_prime);
        // all-zero check
        let ok_3 = check_3_mat
            .data
            .iter()
            .all(|col| col.iter().all(|x| *x == F::zero()));
        if !ok_3 {
            println!("[debug-proof] Condition 2 failed: W*theta - 1 != theta * W' * theta'");
        }

        // 3) bar_b = omega^T * W' * epsilon (last column)
        if cols == 0 {
            return false;
        }
        let last = cols - 1;
        let mut omega_pow = F::one();
        let mut b_bar_check = F::zero();
        for r in 0..rows {
            b_bar_check += omega_pow * mat_wp.data[last][r];
            omega_pow *= omega;
        }

        let ok_1 = b_bar_check == b_bar;
        if !ok_1 {
            println!(
                "[debug-proof] Condition 3 failed: bar_b mismatch. Expected {:?}, got {:?}",
                b_bar, b_bar_check
            );
        }

        ok_1 && ok_3
    }

    /// Check \bar{b} = omega^T * W' * epsilon (last column)
    fn reduce_prover_eq1(
        &self,
        trans_seq: &mut Transcript<F>,
        b_bar: F,
        omega: F,
        mat_wp: &DenseMatFieldCM<F>,
    ) -> (F, (Vec<F>, Vec<F>)) {
        let num_vec = mat_wp.shape.0;
        let log_k_total = mat_wp.shape.1;

        let last_col_wp = mat_wp.data[log_k_total - 1].clone();

        let omega_vec = std::iter::successors(Some(F::one()), |x| Some(*x * omega))
            .take(num_vec)
            .collect::<Vec<_>>();

        let litebullet = LiteBullet::<F>::new(b_bar, num_vec);
        let (a_reduce, b_reduce, challenges_inv, challenges) =
            litebullet.reduce_prover(trans_seq, &omega_vec, &last_col_wp);

        assert_eq!(
            a_reduce,
            xi::xi_power_ip_with_challenges(num_vec, omega, &challenges_inv)
        );

        let epsilon_vec = vec![F::zero(); log_k_total - 1];
        let mut epsilon_vec = epsilon_vec;
        epsilon_vec.push(F::one());
        let epsilon_mat = DenseMatFieldCM::<F>::from_data(vec![epsilon_vec]);

        let mat_matmul = MatMul::<F>::new(
            b_reduce,
            (challenges.clone(), Vec::new()),
            (num_vec, 1),
            mat_wp.shape,
            epsilon_mat.shape,
        );
        let (wp_reduce, epsilon_reduce, wp_point, epsilon_point) =
            mat_matmul.reduce_prover(trans_seq, &mat_wp, &epsilon_mat);

        assert_eq!(
            epsilon_reduce,
            xi::xi_get_at_position(log_k_total - 1, &epsilon_point.0)
        );

        (wp_reduce, wp_point)
    }

    /// Verify \bar{b} = omega^T * W' * epsilon (last column)
    fn verify_eq1(
        &self,
        trans_seq: &mut Transcript<F>,
        b_bar: F,
        omega: F,
    ) -> (bool, F, (Vec<F>, Vec<F>)) {
        let num_vec = self.phi.len();
        let log_k_total = self.k_total.next_power_of_two().ilog2() as usize;

        let litebullet = LiteBullet::<F>::new(b_bar, num_vec);
        let (ok1, a_reduce, b_reduce, challenges_inv, challenges) =
            litebullet.verify_as_subprotocol(trans_seq);

        let ok2 = a_reduce == xi::xi_power_ip_with_challenges(num_vec, omega, &challenges_inv);

        let mat_matmul = MatMul::<F>::new(
            b_reduce,
            (challenges.clone(), Vec::new()),
            (num_vec, 1),
            (num_vec, log_k_total),
            (log_k_total, 1),
        );
        let (ok3, wp_reduce, epsilon_reduce, wp_point, epsilon_point) =
            mat_matmul.verify_as_subprotocol(trans_seq);
        // println!("[verify-eq1-debug] MatMul: {:?}", ok3);

        let ok4 = epsilon_reduce == xi::xi_get_at_position(log_k_total - 1, &epsilon_point.0);
        (ok1 && ok2 && ok3 && ok4, wp_reduce, wp_point)
    }

    /// Check W*vec_theta - 1 = theta * W' * vec_theta_prime
    fn reduce_prover_eq2(
        &self,
        trans_seq: &mut Transcript<F>,
        theta: F,
        mat_w: &DenseMatFieldCM<F>,
        mat_wp: &DenseMatFieldCM<F>,
    ) -> (F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) {
        let num_vec = mat_wp.shape.0;
        let log_k_total = mat_wp.shape.1;

        let theta_vec: Vec<F> = std::iter::successors(Some(F::one()), |x| Some(*x * theta))
            .take(log_k_total)
            .collect::<Vec<_>>();

        let mut theta_prime_vec = theta_vec.clone();
        if let Some(last) = theta_prime_vec.last_mut() {
            *last = F::zero();
        }

        let theta_mat = DenseMatFieldCM::<F>::from_data(vec![theta_vec]);
        let theta_mat_prime = DenseMatFieldCM::<F>::from_data(vec![theta_prime_vec]);

        let mat_w_mul_theta = mat_w.par_mul(&theta_mat);
        let mat_wp_mul_theta_prime = mat_wp.par_mul(&theta_mat_prime);

        let log_num = num_vec.next_power_of_two().ilog2() as usize;

        let mut challenges = Vec::with_capacity(log_num);

        for _ in 0..log_num {
            challenges.push(trans_seq.gen_challenge());
        }

        let lincomb = LinComb::<F>::new(
            F::zero(),
            (challenges.clone(), Vec::new()),
            vec![F::one(), -F::one(), -theta],
        );

        let mat_one = DenseMatFieldCM::<F>::from_data(vec![vec![F::one(); num_vec]]);

        let (reduce_vec, point_vec) = lincomb.reduce_prover(
            trans_seq,
            &vec![mat_w_mul_theta, mat_one, mat_wp_mul_theta_prime],
        );

        assert_eq!(
            reduce_vec[1],
            xi::xi_power_ip_with_challenges(num_vec, F::one(), &challenges)
        );

        let w_theta_reduce = reduce_vec[0].clone();
        let w_theta_point = point_vec[0].clone();

        let w_theta_prime_reduce = reduce_vec[2].clone();
        let w_theta_prime_point = point_vec[2].clone();

        let matmul_w = MatMul::<F>::new(
            w_theta_reduce,
            w_theta_point.clone(),
            (num_vec, 1),
            (num_vec, log_k_total),
            (log_k_total, 1),
        );
        let (w_reduce, theta_reduce, w_point, theta_point) =
            matmul_w.reduce_prover(trans_seq, &mat_w, &theta_mat);

        let matmul_wp = MatMul::<F>::new(
            w_theta_prime_reduce,
            w_theta_prime_point.clone(),
            (num_vec, 1),
            (num_vec, log_k_total),
            (log_k_total, 1),
        );
        let (wp_reduce, theta_prime_reduce, wp_point, theta_prime_point) =
            matmul_wp.reduce_prover(trans_seq, &mat_wp, &theta_mat_prime);

        assert_eq!(
            theta_reduce,
            xi::xi_power_ip_with_challenges(log_k_total, theta.clone(), &theta_point.0)
        );
        assert_eq!(
            theta_prime_reduce,
            xi::xi_power_ip_with_challenges(log_k_total - 1, theta.clone(), &theta_prime_point.0)
        );

        (w_reduce, wp_reduce, w_point, wp_point)
    }

    /// Verify W*vec_theta - 1 = theta * W' * vec_theta_prime
    fn verify_eq2(
        &self,
        trans_seq: &mut Transcript<F>,
        theta: F,
    ) -> (bool, F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) {
        let num_vec = self.phi.len();
        let log_k_total = self.k_total.next_power_of_two().ilog2() as usize;

        let log_num = num_vec.next_power_of_two().ilog2() as usize;

        let mut challenges = Vec::with_capacity(log_num);

        for _ in 0..log_num {
            challenges.push(trans_seq.get_at_position(trans_seq.pointer));
            trans_seq.pointer += 1;
        }

        let lincomb = LinComb::<F>::new(
            F::zero(),
            (challenges.clone(), Vec::new()),
            vec![F::one(), -F::one(), -theta],
        );

        let (ok1, reduce_vec, point_vec) = lincomb.verify_as_subprotocol(trans_seq);

        let ok2 = reduce_vec[1] == xi::xi_power_ip_with_challenges(num_vec, F::one(), &challenges);

        let w_theta_reduce = reduce_vec[0].clone();
        let w_theta_point = point_vec[0].clone();

        let w_theta_prime_reduce = reduce_vec[2].clone();
        let w_theta_prime_point = point_vec[2].clone();

        let matmul_w = MatMul::<F>::new(
            w_theta_reduce,
            w_theta_point.clone(),
            (num_vec, 1),
            (num_vec, log_k_total),
            (log_k_total, 1),
        );
        let (ok3, w_reduce, theta_reduce, w_point, theta_point) =
            matmul_w.verify_as_subprotocol(trans_seq);

        let matmul_wp = MatMul::<F>::new(
            w_theta_prime_reduce,
            w_theta_prime_point.clone(),
            (num_vec, 1),
            (num_vec, log_k_total),
            (log_k_total, 1),
        );
        let (ok4, wp_reduce, theta_prime_reduce, wp_point, theta_prime_point) =
            matmul_wp.verify_as_subprotocol(trans_seq);

        let ok5 = theta_reduce
            == xi::xi_power_ip_with_challenges(log_k_total, theta.clone(), &theta_point.0);
        let ok6 = theta_prime_reduce
            == xi::xi_power_ip_with_challenges(
                log_k_total - 1,
                theta.clone(),
                &theta_prime_point.0,
            );

        (
            ok1 && ok2 && ok3 && ok4 && ok5 && ok6,
            w_reduce,
            wp_reduce,
            w_point,
            wp_point,
        )
    }

    /// Check W' = X ∘ W
    fn reduce_prover_eq3(
        &self,
        trans_seq: &mut Transcript<F>,
        w_prime_value: F,
        w_prime_point: (Vec<F>, Vec<F>),
        mat_x: &DenseMatFieldCM<F>,
        mat_w: &DenseMatFieldCM<F>,
    ) -> (F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) {
        let hadamard = Hadamard::<F>::new(w_prime_value, mat_x.shape, w_prime_point);
        hadamard.reduce_prover(trans_seq, mat_x, mat_w)
    }

    /// Verify W' = X ∘ W
    fn verify_eq3(
        &self,
        trans_seq: &mut Transcript<F>,
        w_prime_value: F,
        w_prime_point: (Vec<F>, Vec<F>),
    ) -> (bool, F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) {
        let num_vec = self.phi.len();
        let log_k_total = self.k_total.next_power_of_two().ilog2() as usize;

        let hadamard = Hadamard::<F>::new(w_prime_value, (num_vec, log_k_total), w_prime_point);
        hadamard.verify_as_subprotocol(trans_seq)
    }

    /// Check X = 1 - I + (Z + I) * y
    fn reduce_prover_eq4(
        &self,
        trans_seq: &mut Transcript<F>,
        mat_x_value: F,
        mat_x_point: (Vec<F>, Vec<F>),
        mat_i: &DenseMatFieldCM<F>,
        mat_z: &DenseMatFieldCM<F>,
        vec_y: &Vec<F>,
    ) -> (F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) {
        let num_vec = mat_i.shape.0;
        let log_k_total = mat_i.shape.1;

        let (rows, cols) = mat_i.shape;

        let mat_y_data = vec![vec_y.clone(); rows];
        let mat_y_transpose = DenseMatFieldCM::<F>::from_data(mat_y_data);
        let mat_y = mat_y_transpose.par_transpose();

        let mat_z_plus_i = mat_z.par_add(&mat_i);
        let mat_z_plus_i_hadamard_mat_y = mat_z_plus_i.par_hadamard(&mat_y);

        let mat_one = DenseMatFieldCM::<F>::from_data(vec![vec![F::one(); rows]; cols]);

        let lincomb = LinComb::<F>::new(
            mat_x_value.clone(),
            (mat_x_point.0.clone(), mat_x_point.1.clone()),
            vec![F::one(), -F::one(), F::one()],
        );

        let (reduce_vec, point_vec) = lincomb.reduce_prover(
            trans_seq,
            &vec![mat_one, mat_i.clone(), mat_z_plus_i_hadamard_mat_y],
        );

        let _log_num = num_vec.next_power_of_two().ilog2() as usize;
        let _log_log_k_total = log_k_total.next_power_of_two().ilog2() as usize;

        let one_reduce = reduce_vec[0].clone();
        let one_point = point_vec[0].clone();
        assert_eq!(
            one_reduce,
            xi::xi_power_ip_with_challenges(num_vec, F::one(), &one_point.0)
                * xi::xi_power_ip_with_challenges(log_k_total, F::one(), &one_point.1)
        );

        let i_reduce_1 = reduce_vec[1].clone();
        let i_point_1 = point_vec[1].clone();

        let z_plus_i_hadamard_mat_y_reduce = reduce_vec[2].clone();
        let z_plus_i_hadamard_mat_y_point = point_vec[2].clone();

        let hadamard = Hadamard::<F>::new(
            z_plus_i_hadamard_mat_y_reduce,
            mat_z.shape.clone(),
            z_plus_i_hadamard_mat_y_point,
        );
        let (z_plus_i_reduce, mat_y_reduce, z_plus_one_point, mat_y_point) =
            hadamard.reduce_prover(trans_seq, &mat_z_plus_i, &mat_y);

        let vec_one_reduce = xi::xi_power_ip_with_challenges(num_vec, F::one(), &mat_y_point.0);
        let vec_y_reduce =
            xi::xi_ip_from_challenges_and_vector_non_power_of_two(&vec_y, &mat_y_point.1);

        assert_eq!(mat_y_reduce, vec_one_reduce * vec_y_reduce);

        let lincomb =
            LinComb::<F>::new(z_plus_i_reduce, z_plus_one_point, vec![F::one(), F::one()]);

        let (reduce_vec, point_vec) =
            lincomb.reduce_prover(trans_seq, &vec![mat_i.clone(), mat_z.clone()]);

        let i_reduce_2 = reduce_vec[0].clone();
        let i_point_2 = point_vec[0].clone();

        let z_reduce = reduce_vec[1].clone();
        let z_point = point_vec[1].clone();

        let bp_i = BatchPoints::<F>::new(
            vec![i_reduce_1, i_reduce_2],
            vec![i_point_1, i_point_2],
            mat_i.shape,
        );

        let (i_reduce, i_point) = bp_i.reduce_prover(trans_seq, &mat_i);

        (i_reduce, z_reduce, i_point, z_point)
    }

    /// Verify X = 1 - I + (Z + I) * y
    fn verify_eq4(
        &self,
        trans_seq: &mut Transcript<F>,
        mat_x_value: F,
        mat_x_point: (Vec<F>, Vec<F>),
        vec_y: &Vec<F>,
    ) -> (bool, F, F, (Vec<F>, Vec<F>), (Vec<F>, Vec<F>)) {
        let num_vec = self.phi.len();
        let log_k_total = self.k_total.next_power_of_two().ilog2() as usize;

        let lincomb = LinComb::<F>::new(
            mat_x_value.clone(),
            (mat_x_point.0.clone(), mat_x_point.1.clone()),
            vec![F::one(), -F::one(), F::one()],
        );

        let (ok1, reduce_vec, point_vec) = lincomb.verify_as_subprotocol(trans_seq);

        let one_reduce = reduce_vec[0].clone();
        let one_point = point_vec[0].clone();
        let ok2 = one_reduce
            == xi::xi_power_ip_with_challenges(num_vec, F::one(), &one_point.0)
                * xi::xi_power_ip_with_challenges(log_k_total, F::one(), &one_point.1);

        let i_reduce_1 = reduce_vec[1].clone();
        let i_point_1 = point_vec[1].clone();

        let z_plus_i_hadamard_mat_y_reduce = reduce_vec[2].clone();
        let z_plus_i_hadamard_mat_y_point = point_vec[2].clone();

        let hadamard = Hadamard::<F>::new(
            z_plus_i_hadamard_mat_y_reduce,
            (num_vec, log_k_total),
            z_plus_i_hadamard_mat_y_point,
        );
        let (ok3, z_plus_i_reduce, mat_y_reduce, z_plus_one_point, mat_y_point) =
            hadamard.verify_as_subprotocol(trans_seq);

        let vec_one_reduce = xi::xi_power_ip_with_challenges(num_vec, F::one(), &mat_y_point.0);
        let vec_y_reduce =
            xi::xi_ip_from_challenges_and_vector_non_power_of_two(&vec_y, &mat_y_point.1);
        // Note: mat_y_reduce is the projection of mat_y at (mat_y_point.0, mat_y_point.1)
        // mat_y is a matrix where each row is vec_y.
        // Its projection at (xl, xr) is (1^T * xl) * (vec_y * xr)
        let ok4 = mat_y_reduce == vec_one_reduce * vec_y_reduce;

        let lincomb2 =
            LinComb::<F>::new(z_plus_i_reduce, z_plus_one_point, vec![F::one(), F::one()]);

        let (ok5, reduce_vec2, point_vec2) = lincomb2.verify_as_subprotocol(trans_seq);

        let i_reduce_2 = reduce_vec2[0].clone();
        let i_point_2 = point_vec2[0].clone();

        let z_reduce = reduce_vec2[1].clone();
        let z_point = point_vec2[1].clone();

        let bp_i = BatchPoints::<F>::new(
            vec![i_reduce_1, i_reduce_2],
            vec![i_point_1, i_point_2],
            (num_vec, log_k_total),
        );

        let (ok6, i_reduce, i_point) = bp_i.verify_as_subprotocol(trans_seq);

        (
            ok1 && ok2 && ok3 && ok4 && ok5 && ok6,
            i_reduce,
            z_reduce,
            i_point,
            z_point,
        )
    }

    /// Reduce Eq. 1-4 to checking projections of I, Z, W
    fn reduce_prover(
        &self,
        trans_seq: &mut Transcript<F>,
        b_bar: F,
        z_points: &Vec<Vec<F>>,
        omega: F,
        y_vec: &Vec<F>,
    ) -> (
        F,
        F,
        F,
        (Vec<F>, Vec<F>),
        (Vec<F>, Vec<F>),
        (Vec<F>, Vec<F>),
    ) {
        let mat_i = self.compute_mat_i();
        let mat_z = self.compute_mat_z(z_points);
        let mat_x = self.compute_mat_x(y_vec, &mat_i, &mat_z);
        let mat_w = self.compute_mat_w(&mat_x);
        let mat_wp = self.compute_mat_w_prime(&mat_x, &mat_w);

        // Eq 1: bar_b = omega^T * W' * epsilon
        let (wp_red1, wp_pt1) = self.reduce_prover_eq1(trans_seq, b_bar, omega, &mat_wp);

        // Eq 2: W*theta - 1 = theta * W' * theta'
        let theta = trans_seq.gen_challenge();
        let (w_red1, wp_red2, w_pt1, wp_pt2) =
            self.reduce_prover_eq2(trans_seq, theta, &mat_w, &mat_wp);

        // Batch open W' at two points
        let bp_wp =
            BatchPoints::<F>::new(vec![wp_red1, wp_red2], vec![wp_pt1, wp_pt2], mat_wp.shape);
        let (wp_red, wp_pt) = bp_wp.reduce_prover(trans_seq, &mat_wp);

        // Eq 3: W' = X ∘ W
        let (x_red, w_red2, x_pt, w_pt2) =
            self.reduce_prover_eq3(trans_seq, wp_red, wp_pt, &mat_x, &mat_w);

        // Batch open W at two points
        let bp_w = BatchPoints::<F>::new(vec![w_red1, w_red2], vec![w_pt1, w_pt2], mat_w.shape);
        let (w_red, w_pt) = bp_w.reduce_prover(trans_seq, &mat_w);

        // Eq 4: X = 1 - I + (Z+I) * y
        let (i_red, z_red, i_pt, z_pt) =
            self.reduce_prover_eq4(trans_seq, x_red, x_pt, &mat_i, &mat_z, y_vec);

        (i_red, z_red, w_red, i_pt, z_pt, w_pt)
    }

    /// Verifier side of the reduction protocol
    fn verify_as_subprotocol(
        &self,
        trans_seq: &mut Transcript<F>,
        b_bar: F,
        omega: F,
        y_vec: &Vec<F>,
    ) -> (
        bool,
        F,
        F,
        F,
        (Vec<F>, Vec<F>),
        (Vec<F>, Vec<F>),
        (Vec<F>, Vec<F>),
    ) {
        let num_vec = self.phi.len();
        let log_k_total = self.k_total.next_power_of_two().ilog2() as usize;

        // Eq 1: bar_b = omega^T * W' * epsilon
        let (ok1, wp_red1, wp_pt1) = self.verify_eq1(trans_seq, b_bar, omega);

        // Eq 2: W*theta - 1 = theta * W' * theta'  (replay prover challenge)
        let theta = trans_seq.get_at_position(trans_seq.pointer);
        trans_seq.pointer += 1;
        let (ok2, w_red1, wp_red2, w_pt1, wp_pt2) = self.verify_eq2(trans_seq, theta);

        // Batch open W' at two points
        let bp_wp = BatchPoints::<F>::new(
            vec![wp_red1, wp_red2],
            vec![wp_pt1, wp_pt2],
            (num_vec, log_k_total),
        );
        let (ok3, wp_red, wp_pt) = bp_wp.verify_as_subprotocol(trans_seq);

        // Eq 3: W' = X ∘ W
        let (ok4, x_red, w_red2, x_pt, w_pt2) = self.verify_eq3(trans_seq, wp_red, wp_pt);

        // Batch open W at two points
        let bp_w = BatchPoints::<F>::new(
            vec![w_red1, w_red2],
            vec![w_pt1, w_pt2],
            (num_vec, log_k_total),
        );
        let (ok5, w_red, w_pt) = bp_w.verify_as_subprotocol(trans_seq);

        // Eq 4: X = 1 - I + (Z+I) * y
        let (ok6, i_red, z_red, i_pt, z_pt) = self.verify_eq4(trans_seq, x_red, x_pt, y_vec);

        let all_ok = ok1 && ok2 && ok3 && ok4 && ok5 && ok6;
        if !all_ok {
            println!(
                "[verify-debug] ok1={}, ok2={}, ok3={}, ok4={}, ok5={}, ok6={}",
                ok1, ok2, ok3, ok4, ok5, ok6
            );
        }

        (all_ok, i_red, z_red, w_red, i_pt, z_pt, w_pt)
    }

    /// Convenience: build matrices from points/y and check relation end-to-end.
    fn check_directly(&self, points: &Vec<Vec<F>>, omega: F, vec_y: &Vec<F>) -> bool {
        let mat_i = self.compute_mat_i();
        let mat_z = self.compute_mat_z(points);
        let mat_x = self.compute_mat_x(vec_y, &mat_i, &mat_z);
        let mat_w = self.compute_mat_w(&mat_x);
        let mat_wp = self.compute_mat_w_prime(&mat_x, &mat_w);
        let b_bar = self.compute_bar_b(points, omega, vec_y);

        self.check_relation(&mat_x, &mat_w, &mat_wp, b_bar, omega)
    }
}

pub struct FieldMatCommitter<E: Pairing>
where
    E::ScalarField:
        PrimeField + Absorb + UniformRand + CanonicalSerialize + CanonicalDeserialize + From<MyInt>,
    E::G1: CanonicalSerialize + CanonicalDeserialize,
    E::G2: CanonicalSerialize + CanonicalDeserialize,
    PairingOutput<E>: CanonicalSerialize + CanonicalDeserialize,
{
    pub pp: PP<E>,
}

impl<E: Pairing> FieldMatCommitter<E>
where
    E::ScalarField:
        PrimeField + Absorb + UniformRand + CanonicalSerialize + CanonicalDeserialize + From<MyInt>,
    E::G1: CanonicalSerialize + CanonicalDeserialize,
    E::G2: CanonicalSerialize + CanonicalDeserialize,
    PairingOutput<E>: CanonicalSerialize + CanonicalDeserialize,
{
    pub fn new(pp: PP<E>) -> Self {
        Self { pp }
    }

    fn pad_to_power_of_two(
        &self,
        mat: &DenseMatFieldCM<E::ScalarField>,
    ) -> DenseMatFieldCM<E::ScalarField> {
        let (m, n) = mat.shape;
        let m_pad = m.next_power_of_two();
        let n_pad = n.next_power_of_two();

        let mut data = Vec::with_capacity(n_pad);
        for col in mat.data.iter() {
            let mut c = col.clone();
            if c.len() < m_pad {
                c.resize(m_pad, E::ScalarField::zero());
            }
            data.push(c);
        }
        // add zero columns if needed
        while data.len() < n_pad {
            data.push(vec![E::ScalarField::zero(); m_pad]);
        }

        DenseMatFieldCM::<E::ScalarField>::from_data(data)
    }

    pub fn commit_mat(
        &self,
        mat_i: &DenseMatFieldCM<E::ScalarField>,
    ) -> (PairingOutput<E>, Vec<E::G1>) {
        let padded = self.pad_to_power_of_two(mat_i);
        let (com, tier1) =
            PC::commit_square_full(&self.pp, &padded.data, E::ScalarField::zero()).unwrap();
        (com, tier1)
    }

    pub fn open_mat(
        &self,
        mat_com: PairingOutput<E>,
        mat_tier1: &Vec<E::G1>,
        mat: &DenseMatFieldCM<E::ScalarField>,
        v: E::ScalarField,
        point: (Vec<E::ScalarField>, Vec<E::ScalarField>),
    ) -> PCTrans<E> {
        let padded = self.pad_to_power_of_two(mat);
        let v_com = self.pp.u * v;
        PC::open_square(
            &self.pp,
            &padded.data,
            &point.0,
            &point.1,
            v_com,
            mat_com,
            mat_tier1,
            E::ScalarField::zero(),
            E::ScalarField::zero(),
        )
        .unwrap()
    }

    pub fn verify_mat(
        &self,
        mat_com: PairingOutput<E>,
        v: E::ScalarField,
        point: (Vec<E::ScalarField>, Vec<E::ScalarField>),
        proof: &PCTrans<E>,
    ) -> bool {
        let v_com = self.pp.u * v;
        PC::verify_square(&self.pp, mat_com, v_com, &point.0, &point.1, proof).unwrap_or(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::batchpcs::BatchPCS;
    use crate::bva::BVA;
    use crate::test_data::generate_deterministic_vecs_decreasing_sizes;
    use crate::xi_rev::xi_rev_from_challenges;
    use ark_bls12_381::Bls12_381;
    use ark_bls12_381::Fr;
    use ark_ff::One;

    #[test]
    fn test_tbvproof_check_directly_matches_tbv_ip() {
        // Generate deterministic MyInt vectors with decreasing sizes.
        let log_n = 8usize; // produces sizes: 1,1,1,1,2,2,4,4 (k_total = 16)
        let vecs_myint = generate_deterministic_vecs_decreasing_sizes(log_n - 1);
        let bva = BVA::new(vecs_myint.clone());
        let layout = bva.layout.clone();

        // Build points = challenges for each original vector (log2 bucket size)
        let mut points: Vec<Vec<Fr>> = Vec::new();
        for v in vecs_myint.iter() {
            let log_k = v.len().next_power_of_two().ilog2() as usize;
            // random challenges for this bucket
            let challenges = (0..log_k)
                .map(|i| Fr::from((i as u64) + 2u64)) // deterministic small non-zero
                .collect::<Vec<_>>();
            points.push(challenges);
        }

        // omega and vec_y
        let omega = Fr::from(7u64);
        let vec_y = vec![Fr::from(3u64); bva.layout.k_total.ilog2() as usize];

        // Build TBV via batchpcs::gen_tbv_from_values_and_points to match production path
        let mut batchpcs = BatchPCS::<Bls12_381>::setup(bva.layout.k_total.ilog2() as usize);
        batchpcs.set_prover_input(vecs_myint);
        let values: Vec<Fr> = vec![Fr::zero(); points.len()];
        let (_batch_values, tbv) =
            batchpcs.gen_tbv_from_values_and_points(&values, &points, omega, &layout);

        // xi from challenges
        let xi_vec = xi_rev_from_challenges(&vec_y);
        let b_bar_tbv = tbv.inner_product(&xi_vec);

        // run matrix-side checker
        let ok = bva.layout.check_directly(&points, omega, &vec_y);
        assert!(ok, "matrix-side check_directly failed");

        // compute bar_b from matrices and compare with TBV ip
        let mat_i = bva.layout.compute_mat_i();
        let mat_z = bva.layout.compute_mat_z(&points);
        let mat_x = bva.layout.compute_mat_x(&vec_y, &mat_i, &mat_z);
        let vec_pi = bva.layout.compute_vec_pi(&mat_x);
        let b_bar_mat = {
            let mut omega_pow = Fr::one();
            let mut acc = Fr::zero();
            for val in vec_pi.iter() {
                acc += omega_pow * *val;
                omega_pow *= omega;
            }
            acc
        };

        assert_eq!(
            b_bar_mat, b_bar_tbv,
            "bar_b mismatch between matrix formulation and TBV IP"
        );
    }

    #[test]
    fn test_tbvproof_flow() {
        let log_n = 8usize;
        let vecs_myint = generate_deterministic_vecs_decreasing_sizes(log_n - 1);
        let bva = BVA::new(vecs_myint.clone());
        let layout = bva.layout;

        let mut points: Vec<Vec<Fr>> = Vec::new();
        for v in vecs_myint.iter() {
            let log_k = v.len().ilog2() as usize;
            points.push((0..log_k).map(|i| Fr::from((i as u64) + 2)).collect());
        }

        let omega = Fr::from(7u64);
        let y_vec = (0..layout.k_total.next_power_of_two().ilog2())
            .map(|i| Fr::from(i as u64 + 1))
            .collect::<Vec<_>>();
        // Build TBV via batchpcs::gen_tbv_from_values_and_points to match production path
        let mut batchpcs = BatchPCS::<Bls12_381>::setup(layout.k_total.ilog2() as usize);
        batchpcs.set_prover_input(vecs_myint);
        let values: Vec<Fr> = vec![Fr::zero(); points.len()];
        let (_batch_values, tbv) =
            batchpcs.gen_tbv_from_values_and_points(&values, &points, omega, &layout);

        // xi from challenges
        let xi_vec = xi_rev_from_challenges(&y_vec);
        let b_bar = tbv.inner_product(&xi_vec);

        // First check Eq1 directly: whether bar_b equals omega^T * W' * epsilon (last column)
        let mat_i = layout.compute_mat_i();
        let mat_z = layout.compute_mat_z(&points);
        let mat_x = layout.compute_mat_x(&y_vec, &mat_i, &mat_z);
        let mat_w = layout.compute_mat_w(&mat_x);
        let mat_wp = layout.compute_mat_w_prime(&mat_x, &mat_w);

        // PC: commit I, Z, W
        let mut rng = test_rng();
        let qlog = layout.k_total.ilog2() as usize;
        let pp = PC::<Bls12_381>::setup(qlog, &mut rng).unwrap();
        let committer = FieldMatCommitter::<Bls12_381>::new(pp);
        let (com_i, tier_i) = committer.commit_mat(&mat_i);
        let (com_z, tier_z) = committer.commit_mat(&mat_z);
        let (com_w, tier_w) = committer.commit_mat(&mat_w);
        let last_col = mat_wp.data[mat_wp.shape.1 - 1].clone();
        let mut omega_pow = Fr::one();
        let mut b_bar_mat = Fr::zero();
        for v in last_col.iter() {
            b_bar_mat += omega_pow * *v;
            omega_pow *= omega;
        }
        assert_eq!(
            b_bar_mat, b_bar,
            "Eq1 check failed: bar_b != omega^T * W' * epsilon"
        );

        let mut trans = Transcript::<Fr>::new(Fr::zero());

        // Perform reduction
        let (i_red, z_red, w_red, i_pt, z_pt, w_pt) =
            layout.reduce_prover(&mut trans, b_bar, &points, omega, &y_vec);

        // --- Verifier side ---
        let mut trans_v = trans.clone();
        trans_v.reset_pointer();
        let (ok_v, i_red_v, z_red_v, w_red_v, i_pt_v, z_pt_v, w_pt_v) =
            layout.verify_as_subprotocol(&mut trans_v, b_bar, omega, &y_vec);

        assert!(ok_v, "Verifier verify_as_subprotocol failed");
        assert_eq!(i_red, i_red_v);
        assert_eq!(z_red, z_red_v);
        assert_eq!(w_red, w_red_v);
        assert_eq!(i_pt, i_pt_v);
        assert_eq!(z_pt, z_pt_v);
        assert_eq!(w_pt, w_pt_v);

        let proof_i = committer.open_mat(com_i.clone(), &tier_i, &mat_i, i_red, i_pt.clone());
        let proof_z = committer.open_mat(com_z.clone(), &tier_z, &mat_z, z_red, z_pt.clone());
        let proof_w = committer.open_mat(com_w.clone(), &tier_w, &mat_w, w_red, w_pt.clone());

        assert!(committer.verify_mat(com_i, i_red, i_pt.clone(), &proof_i));
        assert!(committer.verify_mat(com_z, z_red, z_pt.clone(), &proof_z));
        assert!(committer.verify_mat(com_w, w_red, w_pt.clone(), &proof_w));

        // Verify that the reduced point indeed corresponds to the matrix projection
        let mat_i = layout.compute_mat_i();
        let mat_z = layout.compute_mat_z(&points);
        let mat_x = layout.compute_mat_x(&y_vec, &mat_i, &mat_z);
        let mat_w = layout.compute_mat_w(&mat_x);

        assert_eq!(i_red, mat_i.proj_lr_challenges(&i_pt.0, &i_pt.1));
        assert_eq!(z_red, mat_z.proj_lr_challenges(&z_pt.0, &z_pt.1));
        assert_eq!(w_red, mat_w.proj_lr_challenges(&w_pt.0, &w_pt.1));

        println!("✅ TBV proof reduce_prover flow passed!");
    }
}
