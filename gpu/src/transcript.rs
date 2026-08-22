use ark_ff::field_hashers::{DefaultFieldHasher, HashToField};
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use sha2::{Digest, Sha256};

#[derive(Clone, Debug)]
pub struct FiatShamir {
    hasher: Sha256,
}

impl Default for FiatShamir {
    fn default() -> Self {
        Self::new()
    }
}

impl FiatShamir {
    pub fn new() -> Self {
        let mut hasher = Sha256::new();
        hasher.update(b"zk-smart");
        Self { hasher }
    }

    pub fn push<T: CanonicalSerialize>(&mut self, data: &T) {
        let mut bytes = Vec::new();
        data.serialize_compressed(&mut bytes).unwrap();
        self.hasher.update(bytes);
    }

    pub fn challenge<F: PrimeField>(&mut self) -> F {
        let hash_to_field = <DefaultFieldHasher<Sha256> as HashToField<F>>::new(b"zk-smart");
        loop {
            let digest: [u8; 32] = self.hasher.clone().finalize().into();
            let challenge: F = hash_to_field.hash_to_field::<1>(&digest)[0];
            if !challenge.is_zero() {
                self.push(&challenge);
                return challenge;
            }
            self.hasher.update(b"zk-smart");
        }
    }

    pub fn state(&self) -> [u8; 32] {
        self.hasher.clone().finalize().into()
    }
}
