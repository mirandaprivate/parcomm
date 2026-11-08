use crate::Result;
pub use mikros_qafold::{
    ShortBaseField, ShortFieldElement,
    koalabear::{KoalaBear as B, KoalaBearExt4 as F},
};
use serde::{Deserialize, Serialize};

/// Canonical quartic-field wire encoding; noncanonical coefficients are rejected.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Scalar(pub [u8; 16]);
impl From<F> for Scalar {
    fn from(value: F) -> Self {
        let mut bytes = [0; 16];
        value
            .write_canonical_bytes(&mut bytes)
            .expect("fixed field encoding");
        Self(bytes)
    }
}
impl Scalar {
    pub fn field(self) -> Result<F> {
        F::from_canonical_bytes(&self.0).ok_or_else(|| "noncanonical field element".into())
    }
}
pub fn small(value: i16) -> B {
    B::from_u64((value as i64).rem_euclid(2130706433) as u64)
}
pub fn lift(value: i16) -> F {
    F::from_u64(small(value).to_canonical_u64())
}
pub fn inverse(value: F) -> Result<F> {
    value.try_inverse().ok_or_else(|| "zero inverse".into())
}
pub fn dot(left: &[F], right: &[F]) -> F {
    use rayon::prelude::*;
    assert_eq!(left.len(), right.len());
    left.par_iter()
        .zip(right)
        .map(|(&x, &y)| x * y)
        .reduce(F::zero, |x, y| x + y)
}
pub fn monomial_weight(index: usize, point_low: &[F]) -> F {
    point_low
        .iter()
        .enumerate()
        .fold(F::one(), |weight, (bit, &p)| {
            if (index >> bit) & 1 == 1 {
                weight * p
            } else {
                weight
            }
        })
}
pub fn equality_weight(index: usize, point_msb: &[F]) -> F {
    point_msb
        .iter()
        .enumerate()
        .fold(F::one(), |weight, (coord, &p)| {
            weight
                * if (index >> (point_msb.len() - 1 - coord)) & 1 == 1 {
                    p
                } else {
                    F::one() - p
                }
        })
}
