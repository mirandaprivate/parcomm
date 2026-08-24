/// Implement the litebullet protocol
///
/// Use transcript-based Fiat-Shamir for challenges.
/// No commitments or blinding logic.
use rayon::prelude::*;

use ark_crypto_primitives::sponge::Absorb;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::UniformRand;

use fs::helper_trans::Transcript;
use mat::utils::{linear, xi};

pub struct LiteBullet<F>
where
    F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
{
    pub c: F,
    pub length: usize,
}

impl<F> LiteBullet<F>
where
    F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
{
    pub fn new(c_value: F, length_value: usize) -> Self {
        Self {
            c: c_value,
            length: length_value,
        }
    }

    pub fn reduce_prover(
        &self,
        trans_seq: &mut Transcript<F>,
        vec_a: &Vec<F>,
        vec_b: &Vec<F>,
    ) -> (F, F, Vec<F>, Vec<F>) {
        let n = self.length.next_power_of_two();
        let log_n = n.trailing_zeros() as usize;

        let mut a_padded;
        let mut vec_a_current = vec_a;
        if vec_a.len() < n {
            a_padded = vec_a.clone();
            a_padded.resize(n, F::zero());
            vec_a_current = &a_padded;
        }

        let mut b_padded;
        let mut vec_b_current = vec_b;
        if vec_b.len() < n {
            b_padded = vec_b.clone();
            b_padded.resize(n, F::zero());
            vec_b_current = &b_padded;
        }

        let mut challenges: Vec<F> = Vec::new();
        let mut challenges_inv: Vec<F> = Vec::new();

        let mut vec_a_value;
        let mut vec_b_value;
        for j in 0..log_n {
            let current_len = n / 2usize.pow(j as u32);

            let (a_left, a_right) = vec_a_current.as_slice().split_at(current_len / 2);

            let (b_left, b_right) = vec_b_current.as_slice().split_at(current_len / 2);

            let l_tr = linear::inner_product(&a_left.to_vec(), &b_right.to_vec());
            let r_tr = linear::inner_product(&a_right.to_vec(), &b_left.to_vec());

            // record inner products as transcript responses
            trans_seq.push_response(l_tr);
            trans_seq.push_response(r_tr);

            // public challenge (Fiat-Shamir)
            let x_j = trans_seq.gen_challenge();
            let x_j_inv = x_j.inverse().unwrap();

            challenges.push(x_j);
            challenges_inv.push(x_j_inv);

            vec_a_value = (0..current_len / 2)
                .into_par_iter()
                .map(|i| a_left[i] + a_right[i] * x_j_inv)
                .collect();

            vec_a_current = &vec_a_value;

            vec_b_value = (0..current_len / 2)
                .into_par_iter()
                .map(|i| b_left[i] + b_right[i] * x_j)
                .collect();

            vec_b_current = &vec_b_value;
        }

        let a_reduce = vec_a_current[0];
        let b_reduce = vec_b_current[0];

        // record final responses
        trans_seq.push_response(a_reduce);
        trans_seq.push_response(b_reduce);

        (a_reduce, b_reduce, challenges_inv, challenges)
    }

    pub fn verify_as_subprotocol(
        &self,
        trans_seq: &mut Transcript<F>,
    ) -> (bool, F, F, Vec<F>, Vec<F>) {
        let mut challenges: Vec<F> = Vec::new();
        let mut challenges_inv: Vec<F> = Vec::new();

        // reset pointer to start of transcript
        let mut ptr = trans_seq.pointer;

        let n = self.length.next_power_of_two();
        let log_n = n.trailing_zeros() as usize;

        let mut c_expected = self.c.clone();

        for _ in 0..log_n {
            // read l_tr, r_tr (responses) and x_j (challenge)
            let l_tr = trans_seq.get_at_position(ptr);
            ptr += 1;
            let r_tr = trans_seq.get_at_position(ptr);
            ptr += 1;
            let x_j = trans_seq.get_at_position(ptr);
            ptr += 1;

            let x_j_inv = x_j.inverse().unwrap();
            challenges.push(x_j);
            challenges_inv.push(x_j_inv);

            // Update expected inner product under folding relation:
            // c' = c + x * <a_L,b_R> + x^{-1} * <a_R,b_L>
            c_expected = c_expected + x_j * l_tr + x_j_inv * r_tr;
        }

        // read final a_reduce and b_reduce
        let a_reduce = trans_seq.get_at_position(ptr);
        ptr += 1;
        let b_reduce = trans_seq.get_at_position(ptr);
        ptr += 1;

        trans_seq.pointer = ptr;

        let ok = c_expected == a_reduce * b_reduce;

        (ok, a_reduce, b_reduce, challenges_inv, challenges)
    }

    pub fn verify(&self, trans_seq: &mut Transcript<F>, vec_a: &Vec<F>, vec_b: &Vec<F>) -> bool {
        let n = self.length.next_power_of_two();
        let (ok, a_reduce, b_reduce, challenges_inv, challenges) =
            self.verify_as_subprotocol(trans_seq);

        let mut a_padded;
        let mut vec_a_ptr = vec_a;
        if vec_a.len() < n {
            a_padded = vec_a.clone();
            a_padded.resize(n, F::zero());
            vec_a_ptr = &a_padded;
        }

        let mut b_padded;
        let mut vec_b_ptr = vec_b;
        if vec_b.len() < n {
            b_padded = vec_b.clone();
            b_padded.resize(n, F::zero());
            vec_b_ptr = &b_padded;
        }

        // a was folded with x^{-1}, b with x
        let a_check = a_reduce == xi::xi_ip_from_challenges_and_vector(vec_a_ptr, &challenges_inv);
        let b_check = b_reduce == xi::xi_ip_from_challenges_and_vector(vec_b_ptr, &challenges);

        ok && a_check && b_check && trans_seq.fs.verify_fs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::Zero;

    #[test]
    fn test_litebullet_reduce_and_verify() {
        // Deterministic small vectors
        let vec_a: Vec<Fr> = vec![1u64, 2, 3, 4].into_iter().map(Fr::from).collect();
        let vec_b: Vec<Fr> = vec![5u64, 6, 7, 8].into_iter().map(Fr::from).collect();
        let length = vec_a.len();

        // c = <a, b>
        let c = linear::inner_product(&vec_a, &vec_b);

        let mut transcript = Transcript::new(Fr::zero());
        let lb = LiteBullet::<Fr>::new(c, length);

        let _ = lb.reduce_prover(&mut transcript, &vec_a, &vec_b);

        // Reset pointer to beginning for verifier to read
        let mut transcript_verifier = transcript.clone();
        transcript_verifier.reset_pointer();

        let ok = lb.verify(&mut transcript_verifier, &vec_a, &vec_b);
        assert!(
            ok,
            "LiteBullet verify should return true for matching c = <a,b>"
        );
    }

    #[test]
    fn test_litebullet_non_power_of_two() {
        // Length 3 is not a power of 2
        let vec_a: Vec<Fr> = vec![1u64, 2, 3].into_iter().map(Fr::from).collect();
        let vec_b: Vec<Fr> = vec![4u64, 5, 6].into_iter().map(Fr::from).collect();
        let length = vec_a.len();

        let c = linear::inner_product(&vec_a, &vec_b);

        let mut transcript = Transcript::new(Fr::zero());
        let lb = LiteBullet::<Fr>::new(c, length);

        let _ = lb.reduce_prover(&mut transcript, &vec_a, &vec_b);

        // Reset pointer to beginning for verifier to read
        let mut transcript_verifier = transcript.clone();
        transcript_verifier.reset_pointer();

        let ok = lb.verify(&mut transcript_verifier, &vec_a, &vec_b);
        assert!(
            ok,
            "LiteBullet verify should work for non-power-of-two length"
        );
    }
}
