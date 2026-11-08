//! QAFold whose message, generator, and challenge fields are all F_(65521^8).
//!
//! This adapter uses the polynomial basis of `F_65521[X] / (X^8 - 17)`.
//! Its canonical representation is eight little-endian base-field coefficients,
//! not an integer modulo `65521^8`. The degree-one interface means that no
//! base-field packing reduction is implicit in this PCS.

use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::qafold_short::{
    Prime65521, Prime65521Ext8, ShortBaseField, ShortExtensionField, ShortFieldElement,
    ShortQaDecideSetupMode,
};
use crate::{Error, ShortQaFoldPC};

#[path = "full_extension_simd.rs"]
mod simd;

/// The full octic field, used directly as the QA message field.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct FullPrime65521Ext8(pub Prime65521Ext8);

/// QAFold over full octic-field messages and independently sampled generators.
pub type FullQaFoldPC = ShortQaFoldPC<FullPrime65521Ext8>;

impl FullPrime65521Ext8 {
    /// Creates an element from eight polynomial-basis coefficients.
    #[inline]
    pub const fn new(coefficients: [Prime65521; 8]) -> Self {
        Self(Prime65521Ext8::new(coefficients))
    }

    /// Creates an element from eight polynomial-basis coefficients.
    #[inline]
    pub const fn from_coefficients(coefficients: [Prime65521; 8]) -> Self {
        Self::new(coefficients)
    }

    /// Returns the eight coefficients, in increasing polynomial degree.
    #[inline]
    pub const fn coefficients(&self) -> &[Prime65521; 8] {
        self.0.coefficients_ref()
    }

    /// Applies the automorphism `x -> x^(65521^power)` (period eight).
    #[inline]
    pub fn frobenius(self, power: usize) -> Self {
        Self(self.0.frobenius_power(power))
    }

    /// Multiplies all polynomial coefficients by a prime-field scalar.
    #[inline]
    pub fn mul_prime(self, scalar: Prime65521) -> Self {
        Self(self.0.mul_base(scalar))
    }

    /// Computes `self * rhs + addend` with one final reduction per coefficient.
    #[inline]
    pub fn mul_add(self, rhs: Self, addend: Self) -> Self {
        Self(self.0.mul_add(rhs.0, addend.0))
    }
}

impl From<Prime65521Ext8> for FullPrime65521Ext8 {
    fn from(value: Prime65521Ext8) -> Self {
        Self(value)
    }
}

impl From<FullPrime65521Ext8> for Prime65521Ext8 {
    fn from(value: FullPrime65521Ext8) -> Self {
        value.0
    }
}

macro_rules! impl_binary_op {
    ($trait:ident, $method:ident, $assign_trait:ident, $assign_method:ident, $op:tt) => {
        impl $trait for FullPrime65521Ext8 {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self { Self(self.0 $op rhs.0) }
        }
        impl $assign_trait for FullPrime65521Ext8 {
            #[inline]
            fn $assign_method(&mut self, rhs: Self) { *self = *self $op rhs; }
        }
    };
}

impl_binary_op!(Add, add, AddAssign, add_assign, +);
impl_binary_op!(Sub, sub, SubAssign, sub_assign, -);
impl_binary_op!(Mul, mul, MulAssign, mul_assign, *);

impl Neg for FullPrime65521Ext8 {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl ShortFieldElement for FullPrime65521Ext8 {
    #[inline]
    fn zero() -> Self {
        Self(Prime65521Ext8::zero())
    }
    #[inline]
    fn one() -> Self {
        Self(Prime65521Ext8::one())
    }

    #[inline]
    fn butterfly_slice(left: &mut [Self], right: &mut [Self]) {
        simd::butterfly(left, right);
    }
}

impl ShortBaseField for FullPrime65521Ext8 {
    // An independent identity also binds the polynomial basis and full-field
    // generator policy in the existing parameter and setup hash frames.
    const NAME: &'static str = "FullPrime65521Ext8/X^8-17/v1";
    const MODULUS: u128 = 65_521;
    const CARDINALITY: u128 = 65_521u128.pow(8);
    const CANONICAL_BYTES: usize = 16;
    const UNIFORM_BYTES: usize = 16;
    const EXTENSION_DEGREE: usize = 1;
    const DECIDE_SETUP_MODE: ShortQaDecideSetupMode = ShortQaDecideSetupMode::DirectInterleavedV1;
    const FINITE_LENGTH_PROXIMITY: bool = true;
    type Extension = Self;

    /// Integer embedding reduces modulo the characteristic in coefficient zero.
    #[inline]
    fn from_u64(value: u64) -> Self {
        Self(Prime65521Ext8::from_base(Prime65521::from_u64(value)))
    }

    /// Returns the low four packed coefficients only when the upper four vanish.
    fn to_canonical_u64(self) -> u64 {
        u64::try_from(self.to_canonical_u128())
            .expect("full octic-field canonical encoding exceeds 64 bits")
    }

    fn from_canonical_u64_checked(value: u64) -> Option<Self> {
        Self::from_canonical_u128_checked(u128::from(value))
    }

    /// Returns the eight u16 coefficient encodings packed into a u128.
    fn to_canonical_u128(self) -> u128 {
        self.coefficients()
            .iter()
            .enumerate()
            .fold(0, |packed, (i, value)| {
                packed | (u128::from(value.to_canonical_u64()) << (16 * i))
            })
    }

    fn from_canonical_u128_checked(value: u128) -> Option<Self> {
        <Self as ShortBaseField>::from_canonical_bytes(&value.to_le_bytes())
    }

    fn write_canonical_bytes(self, output: &mut [u8]) -> Result<(), Error> {
        <Prime65521Ext8 as ShortExtensionField<Prime65521>>::write_canonical_bytes(self.0, output)
    }

    fn from_canonical_bytes(input: &[u8]) -> Option<Self> {
        <Prime65521Ext8 as ShortExtensionField<Prime65521>>::from_canonical_bytes(input).map(Self)
    }

    /// Uniform rejection sampling of all eight coefficient limbs at once.
    /// Every accepted 16-byte string represents one field element uniquely.
    fn from_uniform_bytes(input: &[u8]) -> Option<Self> {
        <Self as ShortBaseField>::from_canonical_bytes(input.get(..16)?)
    }

    fn try_inverse(self) -> Option<Self> {
        <Prime65521Ext8 as ShortExtensionField<Prime65521>>::try_inverse(self.0).map(Self)
    }
}

impl ShortExtensionField<FullPrime65521Ext8> for FullPrime65521Ext8 {
    #[inline]
    fn from_base(value: Self) -> Self {
        value
    }

    #[inline]
    fn from_basis_coefficients_fn<F: FnMut(usize) -> Self>(mut coefficient: F) -> Self {
        coefficient(0)
    }

    #[inline]
    fn basis_coefficients(&self) -> &[Self] {
        core::slice::from_ref(self)
    }

    #[inline]
    fn mul_base(self, scalar: Self) -> Self {
        self * scalar
    }

    #[inline]
    fn try_inverse(self) -> Option<Self> {
        <Self as ShortBaseField>::try_inverse(self)
    }
}
