use crate::{Result, field::*};
use blake2::{Blake2s256, Digest};
use serde::Serialize;

pub fn digest(bytes: &[u8]) -> [u8; 32] {
    Blake2s256::digest(bytes).into()
}
pub struct Transcript {
    state: [u8; 32],
    counter: u64,
}
impl Transcript {
    pub fn new() -> Self {
        Self {
            state: digest(b"parcomm-koalabear-i16/public/v1"),
            counter: 0,
        }
    }
    pub fn bytes(&mut self, label: &[u8], bytes: &[u8]) {
        let mut h = Blake2s256::new();
        h.update(self.state);
        h.update((label.len() as u64).to_le_bytes());
        h.update(label);
        h.update((bytes.len() as u64).to_le_bytes());
        h.update(bytes);
        self.state = h.finalize().into();
    }
    pub fn object(&mut self, label: &[u8], object: &impl Serialize) -> Result<()> {
        self.bytes(label, &serde_json::to_vec(object)?);
        Ok(())
    }
    pub fn scalar(&mut self, label: &[u8], value: F) {
        self.bytes(label, &Scalar::from(value).0);
    }
    pub fn challenge(&mut self, label: &[u8]) -> F {
        loop {
            let count = self.counter;
            self.counter += 1;
            self.bytes(label, &count.to_le_bytes());
            if let Some(value) = F::from_uniform_bytes(&self.state[..16]) {
                return value;
            }
        }
    }
    pub fn nonzero(&mut self, label: &[u8]) -> F {
        loop {
            let value = self.challenge(label);
            if value != F::zero() {
                return value;
            }
        }
    }
    pub fn fold_challenge(&mut self, packed_log: usize) -> F {
        loop {
            let value = self.nonzero(b"eo-fold");
            let mut power = value;
            let mut admissible = true;
            for _ in 0..packed_log.max(1) {
                admissible &= power != -F::one();
                power = power * power;
            }
            if admissible {
                return value;
            }
        }
    }
    pub fn state(&self) -> [u8; 32] {
        self.state
    }
}
impl Default for Transcript {
    fn default() -> Self {
        Self::new()
    }
}
