//! Fiat-Shamir transcript used by the standalone packed EO protocol.

use ark_crypto_primitives::sponge::Absorb;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;

/// A compact Poseidon transcript with nonzero field challenges.
#[derive(Clone, Debug)]
pub struct FiatShamir<F: PrimeField + Absorb> {
    inner: fs::FiatShamir<F>,
}

impl<F: PrimeField + Absorb> Default for FiatShamir<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: PrimeField + Absorb> FiatShamir<F> {
    pub fn new() -> Self {
        let domain = F::from(0x7a6b_736d_6172_7401_u64);
        let inner = fs::FiatShamir::new_with_state(domain)
            .expect("recommended Poseidon parameters must be available");
        Self { inner }
    }

    /// Absorbs a canonical encoding injectively in sub-modulus byte chunks.
    pub fn push<T: CanonicalSerialize>(&mut self, value: &T) {
        let mut bytes = Vec::new();
        value
            .serialize_compressed(&mut bytes)
            .expect("serialization into a byte vector cannot fail");
        self.push_bytes(&bytes);
    }

    /// Absorbs an already serialized byte string with an injective length prefix.
    pub fn push_bytes(&mut self, bytes: &[u8]) {
        self.inner.push(F::from(bytes.len() as u64));
        let chunk_bytes = ((F::MODULUS_BIT_SIZE as usize).saturating_sub(1) / 8).max(1);
        for chunk in bytes.chunks(chunk_bytes) {
            self.inner.push(F::from_le_bytes_mod_order(chunk));
        }
    }

    pub fn gen_challenge(&mut self) -> F {
        loop {
            let challenge = self.inner.gen_challenge();
            if !challenge.is_zero() {
                return challenge;
            }
            self.inner.push(F::one());
        }
    }

    pub fn state(&self) -> F {
        self.inner.get_current_state()
    }
}

#[cfg(test)]
mod tests {
    use super::FiatShamir;
    use ark_bls12_381::Fr;

    #[test]
    fn transcript_is_deterministic_and_domain_sensitive() {
        let mut left = FiatShamir::<Fr>::new();
        let mut right = FiatShamir::<Fr>::new();
        left.push(&7_u64);
        right.push(&7_u64);
        assert_eq!(left.gen_challenge(), right.gen_challenge());

        let mut different = FiatShamir::<Fr>::new();
        different.push(&8_u64);
        assert_ne!(left.state(), different.state());
    }
}
