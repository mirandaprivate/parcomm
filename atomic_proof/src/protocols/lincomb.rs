/// Public linear combination of matrices
///
use ark_ff::PrimeField;
use ark_std::UniformRand;
use ark_crypto_primitives::sponge::Absorb;
use ark_serialize::{CanonicalSerialize, CanonicalDeserialize};

use fs::helper_trans::Transcript;
use mat::DenseMatFieldCM;
use mat::utils::matdef::MatOps;

pub struct LinComb<F>
where
    F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
{
    pub c_hat: F,
    pub point: (Vec<F>, Vec<F>),
    pub coeffs: Vec<F>,
}

impl<F> LinComb<F>
where
    F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
{
    pub fn new(
        c_hat_value: F,
        point_value: (Vec<F>, Vec<F>),
        coeffs_value: Vec<F>,
    ) -> Self {
        Self {
            c_hat: c_hat_value,
            point: point_value,
            coeffs: coeffs_value,
        }
    }

    /// Prover: provide reduced values for each matrix in the linear combination.
    /// c_hat = sum(coeffs[i] * proj_lr(mats[i], xl, xr))
    pub fn reduce_prover(
        &self,
        trans_seq: &mut Transcript<F>,
        mats: &Vec<DenseMatFieldCM<F>>,
    ) -> (Vec<F>, Vec<(Vec<F>, Vec<F>)>) {
        let (xl, xr) = (&self.point.0, &self.point.1);
        let len = self.coeffs.len();

        let mut hat_a_vec = Vec::new();
        let mut point_vec = Vec::new();

        for i in 0..len {
            let hat_a = mats[i].proj_lr(xl, xr);
            // push reduced values to transcript
            trans_seq.push_response(hat_a);

            hat_a_vec.push(hat_a);
            point_vec.push((xl.clone(), xr.clone()));
        }

        (hat_a_vec, point_vec)
    }

    pub fn verify_as_subprotocol(
        &self,
        trans_seq: &mut Transcript<F>,
    ) -> (bool, Vec<F>, Vec<(Vec<F>, Vec<F>)>) {
        let (xl, xr) = (&self.point.0, &self.point.1);
        let len = self.coeffs.len();

        let mut hat_a_vec = Vec::new();
        let mut point_vec = Vec::new();
        let mut sum = F::zero();

        let mut ptr = trans_seq.pointer;

        for i in 0..len {
            let hat_a = trans_seq.get_at_position(ptr);
            ptr += 1;

            sum += self.coeffs[i] * hat_a;
            hat_a_vec.push(hat_a);
            point_vec.push((xl.clone(), xr.clone()));
        }

        trans_seq.pointer = ptr;

        let flag = sum == self.c_hat;

        (flag, hat_a_vec, point_vec)
    }

    pub fn verify(
        &self,
        trans_seq: &mut Transcript<F>,
        mats: &Vec<DenseMatFieldCM<F>>,
    ) -> bool {
        let (flag, hat_a_vec, point_vec) = self.verify_as_subprotocol(trans_seq);
        if !flag {
            return false;
        }

        for i in 0..self.coeffs.len() {
            let hat_a_check = mats[i].proj_lr(&point_vec[i].0, &point_vec[i].1);
            if hat_a_vec[i] != hat_a_check {
                return false;
            }
        }

        trans_seq.fs.verify_fs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_bls12_381::Fr;
    use ark_ff::Zero;

    #[test]
    fn test_lincomb_reduce_and_verify() {
        let _m = 2;
        let _n = 2;

        let mat_a = DenseMatFieldCM::<Fr>::from_data(vec![
            vec![Fr::from(1u64), Fr::from(3u64)],
            vec![Fr::from(2u64), Fr::from(4u64)],
        ]);
        let mat_b = DenseMatFieldCM::<Fr>::from_data(vec![
            vec![Fr::from(5u64), Fr::from(7u64)],
            vec![Fr::from(6u64), Fr::from(8u64)],
        ]);

        let xl = vec![Fr::from(2u64)];
        let xr = vec![Fr::from(3u64)];
        let coeffs = vec![Fr::from(10u64), Fr::from(20u64)];

        // c = 10 * A + 20 * B
        // c_hat = proj_lr(c, xl, xr) = 10 * proj_lr(A, xl, xr) + 20 * proj_lr(B, xl, xr)
        let hat_a = mat_a.proj_lr(&xl, &xr);
        let hat_b = mat_b.proj_lr(&xl, &xr);
        let c_hat = coeffs[0] * hat_a + coeffs[1] * hat_b;

        let protocol = LinComb::<Fr>::new(c_hat, (xl, xr), coeffs);

        let mut transcript = Transcript::new(Fr::zero());
        let _ = protocol.reduce_prover(&mut transcript, &vec![mat_a.clone(), mat_b.clone()]);

        let mut trans_v = transcript.clone();
        trans_v.reset_pointer();

        let ok = protocol.verify(&mut trans_v, &vec![mat_a, mat_b]);
        assert!(ok, "LinComb verify should return true for matching inputs");
    }
}
