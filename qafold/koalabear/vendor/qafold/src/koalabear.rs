//! KoalaBear adapter reusing the scalar field arithmetic from gruen/server64.
//! Polynomial coordinates are [1, X, X^2, X^3], with X^4 = 3.
use crate::qafold_short::ShortQaDecideSetupMode;
use crate::{Error, ShortBaseField, ShortExtensionField, ShortFieldElement};
#[path = "koalabear_simd.rs"]
mod simd;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};
#[allow(dead_code)]
#[path = "gruen_field.rs"]
mod arithmetic;
use arithmetic::{Field, Prime, Quad};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(transparent)]
pub struct KoalaBear(arithmetic::KoalaBear);

impl Default for KoalaBear {
    fn default() -> Self {
        Self(arithmetic::KoalaBear::ZERO)
    }
}
impl KoalaBear {
    pub fn pow(self, exponent: u128) -> Self {
        Self(self.0.pow(exponent))
    }
    pub fn from_wide(value: u128) -> Self {
        Self::from_u64((value % 2130706433) as u64)
    }
}
macro_rules! ops {
    ($ty:ty, $add:expr, $sub:expr, $mul:expr, $neg:expr) => {
        impl Add for $ty {
            type Output = Self;
            #[inline]
            fn add(self, other: Self) -> Self {
                $add(self, other)
            }
        }
        impl Sub for $ty {
            type Output = Self;
            #[inline]
            fn sub(self, other: Self) -> Self {
                $sub(self, other)
            }
        }
        impl Mul for $ty {
            type Output = Self;
            #[inline]
            fn mul(self, other: Self) -> Self {
                $mul(self, other)
            }
        }
        impl Neg for $ty {
            type Output = Self;
            #[inline]
            fn neg(self) -> Self {
                $neg(self)
            }
        }
        impl AddAssign for $ty {
            #[inline]
            fn add_assign(&mut self, other: Self) {
                *self = *self + other;
            }
        }
        impl SubAssign for $ty {
            #[inline]
            fn sub_assign(&mut self, other: Self) {
                *self = *self - other;
            }
        }
        impl MulAssign for $ty {
            #[inline]
            fn mul_assign(&mut self, other: Self) {
                *self = *self * other;
            }
        }
    };
}
ops!(
    KoalaBear,
    |left: Self, right: Self| Self(left.0 + right.0),
    |left: Self, right: Self| Self(left.0 - right.0),
    |left: Self, right: Self| Self(left.0 * right.0),
    |value: Self| Self(-value.0)
);
impl ShortFieldElement for KoalaBear {
    fn zero() -> Self {
        Self(arithmetic::KoalaBear::ZERO)
    }
    fn one() -> Self {
        Self(arithmetic::KoalaBear::ONE)
    }
}
impl ShortBaseField for KoalaBear {
    const NAME: &'static str = "KoalaBear/scalar/v1";
    const MODULUS: u128 = 2130706433;
    const CANONICAL_BYTES: usize = 4;
    const EXTENSION_DEGREE: usize = 4;
    const DECIDE_SETUP_MODE: ShortQaDecideSetupMode = ShortQaDecideSetupMode::DirectInterleavedV1;
    type Extension = KoalaBearExt4;
    #[inline]
    fn from_u64(value: u64) -> Self {
        Self(arithmetic::KoalaBear::from_u64(value))
    }
    #[inline]
    fn to_canonical_u64(self) -> u64 {
        self.0.value()
    }
    fn from_canonical_u64_checked(value: u64) -> Option<Self> {
        (value < 2130706433).then(|| Self::from_u64(value))
    }
    fn try_inverse(self) -> Option<Self> {
        self.0.inverse().map(Self)
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct KoalaBearExt4([KoalaBear; 4]);
impl KoalaBearExt4 {
    #[inline]
    pub fn from_coefficients(value: [KoalaBear; 4]) -> Self {
        Self(value)
    }
    #[inline]
    pub fn coefficients(&self) -> &[KoalaBear; 4] {
        &self.0
    }
    #[inline]
    fn tower(self) -> arithmetic::EKoalaBear {
        Quad(
            Quad(self.0[0].0, self.0[2].0),
            Quad(self.0[1].0, self.0[3].0),
        )
    }
    #[inline]
    fn from_tower(value: arithmetic::EKoalaBear) -> Self {
        Self([
            KoalaBear(value.0.0),
            KoalaBear(value.1.0),
            KoalaBear(value.0.1),
            KoalaBear(value.1.1),
        ])
    }
    #[inline]
    pub fn mul_prime(self, scalar: KoalaBear) -> Self {
        Self(self.0.map(|value| value * scalar))
    }
    #[inline]
    pub fn mul_add(self, other: Self, addend: Self) -> Self {
        self * other + addend
    }
    pub fn frobenius(self, power: usize) -> Self {
        const PHASES: [u64; 4] = [1, 2113994754, 2130706432, 16711679];
        let phase = KoalaBear::from_u64(PHASES[power % 4]);
        let mut scale = KoalaBear::one();
        Self(std::array::from_fn(|index| {
            let value = self.0[index] * scale;
            scale *= phase;
            value
        }))
    }
}
ops!(
    KoalaBearExt4,
    |left: Self, right: Self| Self(std::array::from_fn(|index| left.0[index] + right.0[index])),
    |left: Self, right: Self| Self(std::array::from_fn(|index| left.0[index] - right.0[index])),
    |left: Self, right: Self| Self::from_tower(left.tower() * right.tower()),
    |value: Self| Self(value.0.map(|entry| -entry))
);
impl ShortFieldElement for KoalaBearExt4 {
    fn prepare_encoding_multipliers(values: &[Self]) -> Option<crate::qafold_short::PreparedEncodingMultipliers<Self>> {
        Some(crate::qafold_short::PreparedEncodingMultipliers {
            words: values.iter().flat_map(|value| value.0.iter().map(|coefficient|
                ((coefficient.to_canonical_u64() << 32) % 2130706433) as u32)).collect(),
            field: core::marker::PhantomData,
        })
    }
    fn mul_assign_prepared_slice(left: &mut [Self], right: &[Self],
        prepared: &crate::qafold_short::PreparedEncodingMultipliers<Self>, offset: usize)
    {
        simd::multiply_prepared(left, right, &prepared.words[4 * offset..4 * (offset + right.len())]);
    }
    fn butterfly_slice(left: &mut [Self], right: &mut [Self]) { simd::butterfly(left, right); }
    fn mul_assign_slice(left: &mut [Self], right: &[Self]) { simd::multiply(left, right); }
    fn zero() -> Self {
        Self([KoalaBear::zero(); 4])
    }
    fn one() -> Self {
        Self::from_u64(1)
    }
}
impl ShortBaseField for KoalaBearExt4 {
    const NAME: &'static str = "FullKoalaBearExt4/X^4-3/scalar/v1";
    const MODULUS: u128 = 2130706433;
    const CARDINALITY: u128 = 2130706433u128.pow(4);
    const CANONICAL_BYTES: usize = 16;
    const UNIFORM_BYTES: usize = 16;
    const EXTENSION_DEGREE: usize = 1;
    const DECIDE_SETUP_MODE: ShortQaDecideSetupMode = ShortQaDecideSetupMode::DirectInterleavedV1;
    const FINITE_LENGTH_PROXIMITY: bool = true;
    type Extension = Self;
    fn from_u64(value: u64) -> Self {
        Self([
            KoalaBear::from_u64(value),
            KoalaBear::zero(),
            KoalaBear::zero(),
            KoalaBear::zero(),
        ])
    }
    fn to_canonical_u64(self) -> u64 {
        u64::try_from(self.to_canonical_u128()).expect("quartic encoding exceeds u64")
    }
    fn to_canonical_u128(self) -> u128 {
        self.0
            .iter()
            .enumerate()
            .fold(0, |encoded, (index, value)| {
                encoded | ((value.to_canonical_u64() as u128) << (32 * index))
            })
    }
    fn from_canonical_u64_checked(value: u64) -> Option<Self> {
        Self::from_canonical_u128_checked(value as u128)
    }
    fn from_canonical_u128_checked(value: u128) -> Option<Self> {
        <Self as ShortBaseField>::from_canonical_bytes(&value.to_le_bytes())
    }
    fn write_canonical_bytes(self, output: &mut [u8]) -> Result<(), Error> {
        if output.len() != 16 {
            return Err(Error::EncodingError);
        }
        output.copy_from_slice(&self.to_canonical_u128().to_le_bytes());
        Ok(())
    }
    fn from_canonical_bytes(input: &[u8]) -> Option<Self> {
        if input.len() != 16 {
            return None;
        }
        let mut coefficients = [KoalaBear::zero(); 4];
        for (value, bytes) in coefficients.iter_mut().zip(input.chunks_exact(4)) {
            *value = KoalaBear::from_canonical_u64_checked(u32::from_le_bytes(
                bytes.try_into().ok()?,
            ) as u64)?;
        }
        Some(Self(coefficients))
    }
    fn from_uniform_bytes(input: &[u8]) -> Option<Self> {
        // Discard one unbiased bit per coefficient, then reject the tail >= p.
        let mut bytes: [u8; 16] = input.get(..16)?.try_into().ok()?;
        for index in [3, 7, 11, 15] {
            bytes[index] &= 0x7f;
        }
        <Self as ShortBaseField>::from_canonical_bytes(&bytes)
    }
    fn try_inverse(self) -> Option<Self> {
        self.tower().inverse().map(Self::from_tower)
    }
}
impl ShortExtensionField<KoalaBear> for KoalaBearExt4 {
    fn from_base(value: KoalaBear) -> Self {
        Self([
            value,
            KoalaBear::zero(),
            KoalaBear::zero(),
            KoalaBear::zero(),
        ])
    }
    fn from_basis_coefficients_fn<T: FnMut(usize) -> KoalaBear>(coefficient: T) -> Self {
        Self(std::array::from_fn(coefficient))
    }
    fn basis_coefficients(&self) -> &[KoalaBear] {
        &self.0
    }
    fn mul_base(self, scalar: KoalaBear) -> Self {
        self.mul_prime(scalar)
    }
    fn try_inverse(self) -> Option<Self> {
        <Self as ShortBaseField>::try_inverse(self)
    }
}
impl ShortExtensionField<KoalaBearExt4> for KoalaBearExt4 {
    fn from_base(value: Self) -> Self {
        value
    }
    fn from_basis_coefficients_fn<T: FnMut(usize) -> Self>(mut coefficient: T) -> Self {
        coefficient(0)
    }
    fn basis_coefficients(&self) -> &[Self] {
        core::slice::from_ref(self)
    }
    fn mul_base(self, scalar: Self) -> Self {
        self * scalar
    }
    fn try_inverse(self) -> Option<Self> {
        <Self as ShortBaseField>::try_inverse(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{SeedableRng, rngs::StdRng};
    #[test]
    fn field_matches_polynomial_product_and_frobenius() {
        arithmetic::EKoalaBear::validate();
        let mut rng = StdRng::seed_from_u64(1122);
        for _ in 0..100 {
            let left = KoalaBearExt4::sample_uniform(&mut rng);
            let right = KoalaBearExt4::sample_uniform(&mut rng);
            let mut product = [KoalaBear::zero(); 7];
            for row in 0..4 {
                for col in 0..4 {
                    product[row + col] += left.0[row] * right.0[col];
                }
            }
            for degree in (4..7).rev() {
                product[degree - 4] += product[degree] * KoalaBear::from_u64(3);
            }
            assert_eq!(
                left * right,
                KoalaBearExt4(product[..4].try_into().unwrap())
            );
            assert_eq!(
                left.frobenius(1),
                KoalaBearExt4::from_tower(left.tower().pow(2130706433))
            );
            assert_eq!(left.frobenius(4), left);
            assert_eq!(
                left * <KoalaBearExt4 as ShortBaseField>::try_inverse(left).unwrap(),
                KoalaBearExt4::one()
            );
            let mut bytes = [0; 16];
            <KoalaBearExt4 as ShortBaseField>::write_canonical_bytes(left, &mut bytes).unwrap();
            assert_eq!(
                <KoalaBearExt4 as ShortBaseField>::from_canonical_bytes(&bytes),
                Some(left)
            );
        }
        assert!(<KoalaBearExt4 as ShortBaseField>::from_canonical_bytes(&[255; 16]).is_none());
    }
}
