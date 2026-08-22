use ark_crypto_primitives::sponge::Absorb;
/// Implement the LiteBulletEO protocol
///
/// Use transcript-based Fiat-Shamir for challenges.
/// No commitments or blinding logic.
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use ark_std::UniformRand;

use fs::helper_trans::Transcript;

use super::bva::BvaOps;
use super::bva::BVA;
use super::tbv::TBV;

pub struct LiteBulletEO<F>
where
    F: PrimeField + UniformRand + Absorb + CanonicalSerialize + CanonicalDeserialize,
{
    pub c: F,
    pub length: usize,
}

impl<F> LiteBulletEO<F>
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
        vec_a: &BVA,
        vec_b: TBV<F>,
    ) -> (F, F, Vec<F>, Vec<F>) {
        if !vec_a.check_consistency_with_tbv(&vec_b) {
            panic!("BVA and TBV are not consistent");
        }

        let n = vec_a.layout.k_total;
        if n == 0 {
            return (F::zero(), F::zero(), Vec::new(), Vec::new());
        }

        let log_n = n.ilog2() as usize;
        if log_n == 0 {
            let src_idx = vec_a.layout.phi_inv[0].expect("BVA should have one bucket");
            let a0 = vec_a.vecs[src_idx][0];
            let a0_f: F = if a0 >= 0 {
                F::from(a0 as u64)
            } else {
                -F::from((-(a0 as i32)) as u64)
            };
            let b0_f = vec_b.buckets[0].eta.clone();
            trans_seq.push_response(a0_f);
            trans_seq.push_response(b0_f);
            return (a0_f, b0_f, Vec::new(), Vec::new());
        }

        let mut challenges: Vec<F> = Vec::new();
        let mut challenges_inv: Vec<F> = Vec::new();

        let timer_total = std::time::Instant::now();

        let timer_ip0 = std::time::Instant::now();
        let e_tr = vec_a.ip_eo_with_tbv(&vec_b);
        let o_tr = vec_a.ip_oe_with_tbv(&vec_b);
        let ip0_time = timer_ip0.elapsed();

        // record inner products as transcript responses before first challenge
        trans_seq.push_response(e_tr);
        trans_seq.push_response(o_tr);

        // public challenge (Fiat-Shamir)
        let x_j = trans_seq.gen_challenge();

        #[cfg(test)]
        {
            println!(
                "[debug-litebullet-prover] round=0, e_tr={:?}, o_tr={:?}, x_j={:?}",
                e_tr, o_tr, x_j
            );
        }

        let x_j_inv = x_j
            .inverse()
            .expect("Challenge should not be zero; check transcript initialization");

        challenges.push(x_j);
        challenges_inv.push(x_j_inv);

        let timer_fold0 = std::time::Instant::now();
        let mut vec_a_buff = vec_a.eo_fold(x_j_inv);
        let mut vec_b_buff = vec_b.eo_fold(x_j);
        let fold0_time = timer_fold0.elapsed();

        println!(
            "        [Debug] Round 0: ip={:?}, fold={:?}",
            ip0_time, fold0_time
        );

        #[cfg(test)]
        {
            let ip_folded = vec_a_buff.ip_with_tbv(&vec_b_buff);
            println!(
                "[debug-litebullet-prover] round=0, ip_folded={:?}",
                ip_folded
            );
            // Note: Dense consistency check removed to reduce memory footprint
            // In production, only vec_a_buff (BVAFieldBufs) is kept, which stores ~2^{LOG_TOTAL/2} field elements
        }

        for _j in 1..log_n {
            let timer_round = std::time::Instant::now();
            let timer_ip = std::time::Instant::now();
            let e_tr = vec_a_buff.ip_eo_with_tbv(&vec_b_buff);
            let o_tr = vec_a_buff.ip_oe_with_tbv(&vec_b_buff);
            let ip_time = timer_ip.elapsed();

            // record inner products as transcript responses
            trans_seq.push_response(e_tr);
            trans_seq.push_response(o_tr);

            // public challenge (Fiat-Shamir)
            let x_j = trans_seq.gen_challenge();
            let x_j_inv = x_j.inverse().unwrap();

            #[cfg(test)]
            {
                println!(
                    "[debug-litebullet-prover] round={}, e_tr={:?}, o_tr={:?}, x_j={:?}",
                    _j, e_tr, o_tr, x_j
                );
            }

            challenges.push(x_j);
            challenges_inv.push(x_j_inv);

            let timer_fold = std::time::Instant::now();
            vec_a_buff.eo_fold_in_place(x_j_inv);
            vec_b_buff.eo_fold_in_place(x_j);
            let fold_time = timer_fold.elapsed();

            println!(
                "        [Debug] Round {}: ip={:?}, fold={:?}, total={:?}",
                _j,
                ip_time,
                fold_time,
                timer_round.elapsed()
            );

            #[cfg(test)]
            {
                let ip_folded = vec_a_buff.ip_with_tbv(&vec_b_buff);
                println!(
                    "[debug-litebullet-prover] round={}, ip_folded={:?}",
                    _j, ip_folded
                );
                // Note: Dense consistency check removed to reduce memory footprint
            }
        }
        println!(
            "      [Debug] LiteBulletEO total loop time: {:?}",
            timer_total.elapsed()
        );

        let src_idx = vec_a_buff.layout.phi_inv[0].expect("Reduced BVA should have one bucket");
        let a_reduce = vec_a_buff.vecs[src_idx][0].clone();
        let b_reduce = vec_b_buff.buckets[0].eta.clone();

        // record final responses
        trans_seq.push_response(a_reduce);
        trans_seq.push_response(b_reduce);

        (a_reduce, b_reduce, challenges_inv, challenges)
    }

    pub fn verify_as_subprotocol(
        &self,
        trans_seq: &mut Transcript<F>,
    ) -> (bool, F, F, Vec<F>, Vec<F>) {
        let mut ptr = trans_seq.pointer;

        let mut challenges: Vec<F> = Vec::new();
        let mut challenges_inv: Vec<F> = Vec::new();

        let n = self.length.next_power_of_two();
        let log_n = n.ilog2() as usize;

        let mut c_expected = self.c.clone();

        for _i in 0..log_n {
            // read l_tr, r_tr (responses) and x_j (challenge)
            let e_tr = trans_seq.get_at_position(ptr);
            ptr += 1;
            let o_tr = trans_seq.get_at_position(ptr);
            ptr += 1;
            let x_j = trans_seq.get_at_position(ptr);
            ptr += 1;

            let x_j_inv = x_j.inverse().unwrap();
            challenges.push(x_j);
            challenges_inv.push(x_j_inv);

            // Update expected inner product under folding relation:
            // c' = c + x * <a_E,b_E> + x^{-1} * <a_O,b_O>
            // c' = c + x * <a_E,b_E> + x^{-1} * <a_O,b_O>
            c_expected = c_expected + x_j * e_tr + x_j_inv * o_tr;
        }

        // read final a_reduce and b_reduce
        let a_reduce = trans_seq.get_at_position(ptr);
        ptr += 1;
        let b_reduce = trans_seq.get_at_position(ptr);
        ptr += 1;

        trans_seq.pointer = ptr;

        let ok = c_expected == a_reduce * b_reduce;

        #[cfg(test)]
        {
            println!(
                "[debug-litebullet] log_n={}, c_expected={}, a_reduce={}, b_reduce={}, prod={}",
                log_n,
                c_expected,
                a_reduce,
                b_reduce,
                a_reduce * b_reduce
            );
        }

        (ok, a_reduce, b_reduce, challenges_inv, challenges)
    }

    pub fn verify(&self, trans_seq: &mut Transcript<F>, vec_a: &BVA, vec_b: &TBV<F>) -> bool {
        let (ok, a_reduce, b_reduce, challenges_inv, challenges) =
            self.verify_as_subprotocol(trans_seq);

        let a_reduce_check = vec_a.ip_with_xi_rev_from_challenges(&challenges_inv);
        let b_reduce_check = vec_b.ip_with_xi_rev_from_challenges(&challenges);

        let a_check = a_reduce == a_reduce_check;
        let b_check = b_reduce == b_reduce_check;

        ok && a_check && b_check && trans_seq.fs.verify_fs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bva::{BvaOps, BVA};
    use crate::tbv::{Bucket, TBV};
    use crate::MyInt;
    use ark_bls12_381::Fr;
    use ark_ff::Zero;

    /// Helper: build a simple BVA/TBV pair of length 4 (power of two).
    fn make_sample_bva_tbv() -> (BVA, TBV<Fr>, Fr) {
        // BVA with one bucket of length 4
        let vec_a = vec![1i8, 2, 3, 4]
            .into_iter()
            .map(|x| MyInt::from(x))
            .collect::<Vec<MyInt>>();
        let bva = BVA::new(vec![vec_a]);

        // TBV with one bucket k=4, eta=1, z=[2,3]
        let bucket = Bucket {
            k: 4,
            eta: Fr::from(1u64),
            z: vec![Fr::from(2u64), Fr::from(3u64)], // log2(4)=2 entries
        };
        let tbv = TBV::new(4, vec![bucket]);

        // c = dense inner product of expanded vectors
        let a_dense: Vec<Fr> = bva.expand_to_field_vec();
        let b_dense: Vec<Fr> = tbv.expand();
        let c: Fr = a_dense
            .iter()
            .zip(b_dense.iter())
            .map(|(a, b)| -> Fr { *a * *b })
            .sum();

        (bva, tbv, c)
    }

    /// Helper: build a heterogeneous BVA/TBV pair of total length 8.
    fn make_heterogeneous_bva_tbv() -> (BVA, TBV<Fr>, Fr) {
        // Total length = 1 + 1 + 2 + 4 = 8
        let vec_a1 = vec![MyInt::from(2i8)];
        let vec_a2 = vec![MyInt::from(3i8)];
        let vec_a3 = vec![MyInt::from(5i8), MyInt::from(7i8)];
        let vec_a4 = vec![
            MyInt::from(11i8),
            MyInt::from(13i8),
            MyInt::from(17i8),
            MyInt::from(19i8),
        ];
        let bva = BVA::new(vec![vec_a1, vec_a2, vec_a3, vec_a4]);

        let buckets = vec![
            Bucket {
                k: 4,
                eta: Fr::from(11u64),
                z: vec![Fr::from(13u64), Fr::from(17u64)],
            },
            Bucket {
                k: 2,
                eta: Fr::from(5u64),
                z: vec![Fr::from(7u64)],
            },
            Bucket {
                k: 1,
                eta: Fr::from(2u64),
                z: vec![],
            },
            Bucket {
                k: 1,
                eta: Fr::from(3u64),
                z: vec![],
            },
        ];
        let tbv = TBV::new(8, buckets);

        let a_dense: Vec<Fr> = bva.expand_to_field_vec();
        let b_dense: Vec<Fr> = tbv.expand();
        let c: Fr = a_dense
            .iter()
            .zip(b_dense.iter())
            .map(|(a, b)| -> Fr { *a * *b })
            .sum();

        (bva, tbv, c)
    }

    #[test]
    fn test_litebulleteo_reduce_and_verify() {
        let (bva, tbv, c) = make_sample_bva_tbv();
        let length = bva.layout.k_total;

        let mut transcript = Transcript::new(Fr::zero());
        let lb = LiteBulletEO::<Fr>::new(c, length);

        let _ = lb.reduce_prover(&mut transcript, &bva, tbv.clone());

        // Reset pointer to beginning for verifier to read
        let mut transcript_verifier = transcript.clone();
        transcript_verifier.reset_pointer();

        let ok = lb.verify(&mut transcript_verifier, &bva, &tbv);
        assert!(
            ok,
            "LiteBulletEO verify should return true for matching inputs"
        );
    }

    #[test]
    fn test_litebulleteo_heterogeneous() {
        let (bva, tbv, c) = make_heterogeneous_bva_tbv();
        let length = bva.layout.k_total;

        let mut transcript = Transcript::new(Fr::zero());
        let lb = LiteBulletEO::<Fr>::new(c, length);

        let _ = lb.reduce_prover(&mut transcript, &bva, tbv.clone());

        let mut transcript_verifier = transcript.clone();
        transcript_verifier.reset_pointer();

        let ok = lb.verify(&mut transcript_verifier, &bva, &tbv);
        assert!(
            ok,
            "LiteBulletEO verify should work for heterogeneous buckets"
        );
    }
}
