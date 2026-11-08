//! Bit-width-aware QAFold encoding over small prime fields.
//!
//! The top QA code is evaluated over the base field. Every group of extension
//! degree many encoded columns is packed into one extension-field column during
//! the first interleaved reduction. All lower QA levels then use the lifted
//! generator over the extension field.

use core::{
    array,
    fmt::Debug,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use blake2::Blake2s256;
use digest::Digest;
use rand::RngCore;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::Error;
use crate::{
    brakefold::{
        queries_for_relative_distance, rf_extended_relative_distance, rf_layers_for_switch,
    },
    qafold::{
        QAFOLD_RECURSIVE_INDEX, QAFOLD_TOP_INDEX, QaDistanceBound, QaRfDistanceBound,
        qa_distance_bound_with_modulus_log2, qa_entropy_distance_bound_with_modulus_log2,
    },
};

#[path = "qafold_short_distance.rs"]
mod distance;
pub use distance::{
    NONZERO_QA_SECURITY_SLACK_BITS, qa_nonzero_distance_bound, qa_nonzero_log2_failure_bound,
};

#[path = "qafold_short/encoding_profile.rs"]
mod encoding_profile;
#[cfg(feature = "profile-encoding")]
pub use encoding_profile::QaEncodeProfile;
use encoding_profile::{EncodeRecorder, EncodeStage};

fn qafold_short_index_for_level_with_recursive_index<B: ShortBaseField>(
    level: usize,
    levels: usize,
    recursive_index: usize,
) -> Result<usize, Error> {
    if levels == 0 || level >= levels || !matches!(recursive_index, 2 | 4) {
        return Err(Error::EncodingError);
    }
    Ok(if levels - level <= B::TOP_HALF_RATE_LEVELS {
        QAFOLD_TOP_INDEX
    } else {
        recursive_index
    })
}

/// Fixed Decide setup strategy selected by the short base field.
///
/// This is deliberately a closed policy surface for the current short-field
/// implementations. It prevents a caller from selecting a commitment layout
/// that is inconsistent with the field-specific benchmark configuration.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShortQaDecideSetupMode {
    /// The original random-foldable `T_j || H_j` setup commitment.
    RfV1,
    /// A direct lower-QA-code commitment to interleaved `H_j` columns.
    DirectInterleavedV1,
}

impl ShortQaDecideSetupMode {
    /// Stable domain-separation label for parameter and commitment digests.
    pub const fn domain_label(self) -> &'static [u8] {
        match self {
            Self::RfV1 => b"rf-v1",
            Self::DirectInterleavedV1 => b"direct-interleaved-v1",
        }
    }
}

/// Encoding-only prepared constants, kept separate from canonical field values.
#[derive(Clone, Debug)]
pub struct PreparedEncodingMultipliers<F> {
    pub(crate) words: Vec<u32>,
    pub(crate) field: core::marker::PhantomData<F>,
}

/// Operations required by the QA WHT and pointwise products.
pub trait ShortFieldElement:
    Copy
    + Clone
    + Default
    + Debug
    + Eq
    + Send
    + Sync
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Neg<Output = Self>
    + 'static
{
    /// Additive identity.
    fn zero() -> Self;

    /// Multiplicative identity.
    fn one() -> Self;

    /// Pointwise multiplication; field backends can batch canonical coefficients.
    fn mul_assign_slice(left: &mut [Self], right: &[Self]) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter_mut().zip(right) { *left *= *right; }
    }

    /// Optional public-parameter preparation; never hashes prepared words.
    fn prepare_encoding_multipliers(_values: &[Self]) -> Option<PreparedEncodingMultipliers<Self>> {
        None
    }

    /// Canonical output from an encoding-specific prepared constant table.
    fn mul_assign_prepared_slice(
        left: &mut [Self], right: &[Self], _prepared: &PreparedEncodingMultipliers<Self>,
        _offset: usize,
    ) {
        Self::mul_assign_slice(left, right);
    }

    /// Applies `(a, b) -> (a + b, a - b)` to equally sized, disjoint slices.
    /// Field backends may specialize the contiguous coefficient kernel.
    #[inline]
    fn butterfly_slice(left: &mut [Self], right: &mut [Self]) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter_mut().zip(right) {
            let a = *left;
            let b = *right;
            *left = a + b;
            *right = a - b;
        }
    }
}

/// A message field together with the field used by the folding reductions.
pub trait ShortBaseField: ShortFieldElement {
    /// Human-readable backend name.
    const NAME: &'static str;
    /// Prime characteristic (not the cardinality of an extension field).
    const MODULUS: u128;
    /// Number of field elements, used by distance and soundness estimates.
    const CARDINALITY: u128 = Self::MODULUS;
    /// Bytes consumed by one rejection-sampling candidate.
    /// Non-prime message fields must also override `from_uniform_bytes`.
    const UNIFORM_BYTES: usize = if Self::MODULUS > u64::MAX as u128 {
        16
    } else {
        8
    };
    /// Width of one canonical little-endian field encoding.
    const CANONICAL_BYTES: usize;
    /// Extension degree used by short QAFold.
    const EXTENSION_DEGREE: usize;
    /// Field-selected fixed setup strategy for recursive Decide.
    const DECIDE_SETUP_MODE: ShortQaDecideSetupMode;
    /// Uses the nonzero-frequency ensemble and finite-length bound of Appendix B.8.
    const NONZERO_QA_SAMPLING: bool = false;
    /// Number of largest levels forced to inverse rate two in an explicit mixed schedule.
    const TOP_HALF_RATE_LEVELS: usize = 1;
    /// Uses the finite-length QA guarantee for the RF proximity bound.
    const FINITE_LENGTH_PROXIMITY: bool = false;
    /// Extension field used after the first interleaved reduction.
    type Extension: ShortExtensionField<Self>;

    /// Reduces an integer into the base field.
    fn from_u64(value: u64) -> Self;

    /// Returns the canonical representative when it fits in 64 bits.
    /// Wide backends must reject out-of-range conversions instead of truncating.
    fn to_canonical_u64(self) -> u64;

    /// Constructs an element only when the integer is canonical.
    fn from_canonical_u64_checked(value: u64) -> Option<Self>;

    /// Returns the full canonical integer representative.
    fn to_canonical_u128(self) -> u128 {
        self.to_canonical_u64() as u128
    }

    /// Constructs an element from a full-width canonical representative.
    fn from_canonical_u128_checked(value: u128) -> Option<Self> {
        Self::from_canonical_u64_checked(u64::try_from(value).ok()?)
    }

    /// Encodes the modulus without changing historical short-field hash frames.
    fn canonical_modulus_bytes() -> Vec<u8> {
        if Self::MODULUS <= u64::MAX as u128 {
            (Self::MODULUS as u64).to_le_bytes().to_vec()
        } else {
            Self::MODULUS.to_le_bytes().to_vec()
        }
    }

    /// Samples uniformly from a hash block, rejecting a biased tail if needed.
    ///
    /// The short-field branch retains its original eight-byte rejection rule.
    fn from_uniform_bytes(input: &[u8]) -> Option<Self> {
        if Self::MODULUS <= u64::MAX as u128 {
            let candidate = u64::from_le_bytes(input.get(..8)?.try_into().ok()?) as u128;
            let range = 1u128 << 64;
            let acceptance_limit = range - range % Self::MODULUS;
            if candidate < acceptance_limit {
                Self::from_canonical_u128_checked(candidate % Self::MODULUS)
            } else {
                None
            }
        } else {
            let candidate = u128::from_le_bytes(input.get(..16)?.try_into().ok()?);
            let mask = u128::MAX >> Self::MODULUS.leading_zeros();
            Self::from_canonical_u128_checked(candidate & mask)
        }
    }

    /// Samples a uniformly distributed field element from a full-width source.
    fn sample_uniform<R: RngCore>(rng: &mut R) -> Self {
        loop {
            let mut bytes = [0u8; 16];
            bytes[..8].copy_from_slice(&rng.next_u64().to_le_bytes());
            if Self::UNIFORM_BYTES > 8 {
                bytes[8..].copy_from_slice(&rng.next_u64().to_le_bytes());
            }
            if let Some(value) = Self::from_uniform_bytes(&bytes) {
                return value;
            }
        }
    }

    /// Writes the fixed-width canonical little-endian encoding.
    fn write_canonical_bytes(self, output: &mut [u8]) -> Result<(), Error> {
        if output.len() != Self::CANONICAL_BYTES || Self::CANONICAL_BYTES > 8 {
            return Err(Error::EncodingError);
        }
        let encoded = self.to_canonical_u64().to_le_bytes();
        output.copy_from_slice(&encoded[..Self::CANONICAL_BYTES]);
        Ok(())
    }

    /// Decodes a fixed-width canonical little-endian encoding.
    fn from_canonical_bytes(input: &[u8]) -> Option<Self> {
        if input.len() != Self::CANONICAL_BYTES || Self::CANONICAL_BYTES > 8 {
            return None;
        }
        let mut encoded = [0u8; 8];
        encoded[..Self::CANONICAL_BYTES].copy_from_slice(input);
        Self::from_canonical_u64_checked(u64::from_le_bytes(encoded))
    }

    /// Returns the multiplicative inverse, or `None` for zero.
    fn try_inverse(self) -> Option<Self>;
}

/// Extension operations needed by the lifted QA encoder.
pub trait ShortExtensionField<B: ShortBaseField>: ShortFieldElement {
    /// Embeds one base-field element.
    fn from_base(value: B) -> Self;

    /// Constructs one extension element in the fixed polynomial/tower basis.
    fn from_basis_coefficients_fn<F>(coefficient: F) -> Self
    where
        F: FnMut(usize) -> B;

    /// Returns coefficients in the canonical polynomial or tower basis.
    fn basis_coefficients(&self) -> &[B];

    /// Returns the fixed width of one canonical extension-field encoding.
    fn canonical_byte_len() -> usize {
        B::CANONICAL_BYTES * B::EXTENSION_DEGREE
    }

    /// Writes basis coefficients as fixed-width canonical base-field encodings.
    fn write_canonical_bytes(self, output: &mut [u8]) -> Result<(), Error> {
        if output.len() != Self::canonical_byte_len()
            || self.basis_coefficients().len() != B::EXTENSION_DEGREE
        {
            return Err(Error::EncodingError);
        }
        for (coefficient, bytes) in self
            .basis_coefficients()
            .iter()
            .zip(output.chunks_exact_mut(B::CANONICAL_BYTES))
        {
            coefficient.write_canonical_bytes(bytes)?;
        }
        Ok(())
    }

    /// Decodes canonical basis coefficients into one extension element.
    fn from_canonical_bytes(input: &[u8]) -> Option<Self> {
        if input.len() != Self::canonical_byte_len() {
            return None;
        }
        let coefficients = input
            .chunks_exact(B::CANONICAL_BYTES)
            .map(B::from_canonical_bytes)
            .collect::<Option<Vec<_>>>()?;
        Some(Self::from_basis_coefficients_fn(|index| {
            coefficients[index]
        }))
    }

    /// Multiplies by a scalar from the base field.
    fn mul_base(self, scalar: B) -> Self;

    /// Returns the multiplicative inverse, or `None` for zero.
    fn try_inverse(self) -> Option<Self>;
}

/// The 16-bit prime field with modulus `2^16 - 15`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct Prime65521(u16);

impl Prime65521 {
    const MODULUS_U32: u32 = 65_521;

    /// Creates a canonical field element.
    #[inline]
    pub const fn new(value: u32) -> Self {
        Self((value % Self::MODULUS_U32) as u16)
    }

    /// Returns the canonical integer representative.
    #[inline]
    pub const fn as_u16(self) -> u16 {
        self.0
    }

    #[inline]
    const fn reduce_u32(value: u32) -> u16 {
        let folded = (value & 0xffff) + 15 * (value >> 16);
        let folded = (folded & 0xffff) + 15 * (folded >> 16);
        if folded >= Self::MODULUS_U32 {
            (folded - Self::MODULUS_U32) as u16
        } else {
            folded as u16
        }
    }

    #[inline]
    fn pow_u64(mut self, mut exponent: u64) -> Self {
        let mut result = Self::one();
        while exponent != 0 {
            if exponent & 1 == 1 {
                result *= self;
            }
            self *= self;
            exponent >>= 1;
        }
        result
    }
}

impl Add for Prime65521 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        let sum = u32::from(self.0) + u32::from(rhs.0);
        Self(if sum >= Self::MODULUS_U32 {
            (sum - Self::MODULUS_U32) as u16
        } else {
            sum as u16
        })
    }
}

impl AddAssign for Prime65521 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Prime65521 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(if self.0 >= rhs.0 {
            self.0 - rhs.0
        } else {
            (u32::from(self.0) + Self::MODULUS_U32 - u32::from(rhs.0)) as u16
        })
    }
}

impl SubAssign for Prime65521 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for Prime65521 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        let product = u32::from(self.0) * u32::from(rhs.0);
        Self(Self::reduce_u32(product))
    }
}

impl MulAssign for Prime65521 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Neg for Prime65521 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        if self.0 == 0 {
            self
        } else {
            Self((Self::MODULUS_U32 - u32::from(self.0)) as u16)
        }
    }
}

impl ShortFieldElement for Prime65521 {
    #[inline]
    fn zero() -> Self {
        Self(0)
    }

    #[inline]
    fn one() -> Self {
        Self(1)
    }
}

impl ShortBaseField for Prime65521 {
    const NAME: &'static str = "Prime65521";
    const MODULUS: u128 = 65_521;
    const CANONICAL_BYTES: usize = 2;
    const EXTENSION_DEGREE: usize = 8;
    const DECIDE_SETUP_MODE: ShortQaDecideSetupMode = ShortQaDecideSetupMode::RfV1;
    type Extension = Prime65521Ext8;

    #[inline]
    fn from_u64(value: u64) -> Self {
        Self((value % Self::MODULUS as u64) as u16)
    }

    #[inline]
    fn to_canonical_u64(self) -> u64 {
        u64::from(self.0)
    }

    #[inline]
    fn from_canonical_u64_checked(value: u64) -> Option<Self> {
        ((value as u128) < Self::MODULUS).then(|| Self(value as u16))
    }

    #[inline]
    fn try_inverse(self) -> Option<Self> {
        (self != Self::zero()).then(|| self.pow_u64(Self::MODULUS as u64 - 2))
    }
}

/// The octic extension `F_65521[X] / (X^8 - 17)`.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[repr(transparent)]
pub struct Prime65521Ext8([Prime65521; 8]);

impl Prime65521Ext8 {
    // Powers of 17^((65521 - 1) / 8), in the fixed polynomial basis.
    const FROBENIUS_ROOT_POWERS: [Prime65521; 8] = {
        let mut powers = [Prime65521(1); 8];
        let mut i = 1;
        while i < 8 {
            powers[i] = Prime65521(Prime65521::reduce_u32(powers[i - 1].0 as u32 * 57_852));
            i += 1;
        }
        powers
    };

    /// Creates an extension element from polynomial-basis coefficients.
    #[inline]
    pub const fn new(coefficients: [Prime65521; 8]) -> Self {
        Self(coefficients)
    }

    /// Returns the polynomial-basis coefficients.
    #[inline]
    pub const fn coefficients(self) -> [Prime65521; 8] {
        self.0
    }

    /// Borrows the polynomial-basis coefficients.
    #[inline]
    pub const fn coefficients_ref(&self) -> &[Prime65521; 8] {
        &self.0
    }

    /// Applies the prime-field Frobenius `power` times.
    pub fn frobenius_power(self, power: usize) -> Self {
        let power = power % 8;
        if power == 0 {
            return self;
        }
        Self(array::from_fn(|i| match (i * power) % 8 {
            0 => self.0[i],
            4 => -self.0[i],
            index => self.0[i] * Self::FROBENIUS_ROOT_POWERS[index],
        }))
    }

    #[inline]
    fn frobenius(self) -> Self {
        self.frobenius_power(1)
    }

    /// Multiplies and adds before the final eight coefficient reductions.
    #[inline]
    pub fn mul_add(self, rhs: Self, addend: Self) -> Self {
        let product = karatsuba_8_unreduced(self.0, rhs.0);
        Self(array::from_fn(|i| {
            let high = if i < 7 { product[i + 8] } else { 0 };
            reduce_octic_coefficient(product[i] + 17 * high + u64::from(addend.0[i].0))
        }))
    }

    #[inline]
    fn is_zero(self) -> bool {
        self.0.iter().all(|value| *value == Prime65521::zero())
    }
}

impl Add for Prime65521Ext8 {
    type Output = Self;

    #[inline]
    fn add(self, rhs: Self) -> Self::Output {
        Self(array::from_fn(|index| self.0[index] + rhs.0[index]))
    }
}

impl AddAssign for Prime65521Ext8 {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        for (left, right) in self.0.iter_mut().zip(rhs.0) {
            *left += right;
        }
    }
}

impl Sub for Prime65521Ext8 {
    type Output = Self;

    #[inline]
    fn sub(self, rhs: Self) -> Self::Output {
        Self(array::from_fn(|index| self.0[index] - rhs.0[index]))
    }
}

impl SubAssign for Prime65521Ext8 {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        for (left, right) in self.0.iter_mut().zip(rhs.0) {
            *left -= right;
        }
    }
}

impl Mul for Prime65521Ext8 {
    type Output = Self;

    #[inline]
    fn mul(self, rhs: Self) -> Self::Output {
        self.mul_add(rhs, Self::zero())
    }
}

impl MulAssign for Prime65521Ext8 {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Neg for Prime65521Ext8 {
    type Output = Self;

    #[inline]
    fn neg(self) -> Self::Output {
        Self(self.0.map(Neg::neg))
    }
}

impl ShortFieldElement for Prime65521Ext8 {
    #[inline]
    fn zero() -> Self {
        Self([Prime65521::zero(); 8])
    }

    #[inline]
    fn one() -> Self {
        let mut coefficients = [Prime65521::zero(); 8];
        coefficients[0] = Prime65521::one();
        Self(coefficients)
    }
}

impl ShortExtensionField<Prime65521> for Prime65521Ext8 {
    #[inline]
    fn from_base(value: Prime65521) -> Self {
        let mut coefficients = [Prime65521::zero(); 8];
        coefficients[0] = value;
        Self(coefficients)
    }

    #[inline]
    fn from_basis_coefficients_fn<F>(mut coefficient: F) -> Self
    where
        F: FnMut(usize) -> Prime65521,
    {
        Self(array::from_fn(&mut coefficient))
    }

    #[inline]
    fn basis_coefficients(&self) -> &[Prime65521] {
        &self.0
    }

    #[inline]
    fn mul_base(self, scalar: Prime65521) -> Self {
        Self(self.0.map(|value| value * scalar))
    }

    fn try_inverse(self) -> Option<Self> {
        if self.is_zero() {
            return None;
        }
        let mut product_of_conjugates = self.frobenius();
        for _ in 2..8 {
            product_of_conjugates = (product_of_conjugates * self).frobenius();
        }
        let norm = (self * product_of_conjugates).0[0];
        Some(product_of_conjugates.mul_base(norm.try_inverse()?))
    }
}

/// A QA generator stored over the small base field and lifted on demand.
#[derive(Clone, Debug)]
pub struct ShortQaCode<B: ShortBaseField> {
    message_len: usize,
    index: usize,
    alphas: Vec<Vec<B>>,
    // Public setup cache only. Canonical alphas and all digests remain unchanged.
    prepared_parity: Option<PreparedEncodingMultipliers<B>>,
    generator_digest: [u8; 32],
    nonzero_frequency_sampling: bool,
    // Derived only from immutable alpha[0]. This optimization hint is not a
    // parameter and must not change the generator or transcript digest.
    systematic_first: bool,
}

/// An MLE claim on a matrix over a short field extension.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortQaMatrixClaim<E> {
    /// Claimed MLE value.
    pub value: E,
    /// Row and column MLE points.
    pub point: (Vec<E>, Vec<E>),
}

/// Deterministic reduction of a QA generator claim to its fixed alpha matrix.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortQaGeneratorReduction<E> {
    /// Whether the prover-side value matches the materialized QA generator.
    pub ok: bool,
    /// Product of both WHT normalizations and the tensor-point merge.
    pub normalization: E,
    /// Remaining MLE claim on the fixed alpha matrix `H_j`.
    pub alpha_claim: ShortQaMatrixClaim<E>,
}

impl<B: ShortBaseField> ShortQaCode<B> {
    /// Samples a deterministic systematic QA generator.
    pub fn sample_systematic(
        message_len: usize,
        index: usize,
        seed: [u8; 32],
    ) -> Result<Self, Error> {
        if message_len == 0
            || !message_len.is_power_of_two()
            || index <= 1
            || !index.is_power_of_two()
        {
            return Err(Error::EncodingError);
        }
        if B::NONZERO_QA_SAMPLING {
            let inverse_len = B::from_u64(message_len as u64)
                .try_inverse()
                .ok_or(Error::EncodingError)?;
            // Normalizing a tuple of independent invertible generators by its
            // first generator preserves the code and makes the first block systematic.
            let mut alphas = Vec::with_capacity(index);
            alphas.push(vec![inverse_len; message_len]);
            for block in 1..index {
                alphas.push(
                    (0..message_len)
                        .map(|row| {
                            sample_generator_element::<B>(
                                &seed,
                                message_len,
                                index,
                                block,
                                row,
                                true,
                            )
                            .map(|value| value * inverse_len)
                        })
                        .collect::<Result<Vec<_>, Error>>()?,
                );
            }
            let generator_digest = qa_generator_digest::<B>(&alphas)?;
            let prepared_parity = if index == 2 { B::prepare_encoding_multipliers(&alphas[1]) } else { None };
            return Ok(Self {
                message_len,
                index,
                alphas,
                prepared_parity,
                generator_digest,
                nonzero_frequency_sampling: true,
                systematic_first: true,
            });
        }
        let mut coefficient_columns = Vec::with_capacity(index);
        let mut identity = vec![B::zero(); message_len];
        identity[0] = B::one();
        coefficient_columns.push(identity);
        for block in 1..index {
            coefficient_columns.push(
                (0..message_len)
                    .map(|row| {
                        sample_generator_element::<B>(&seed, message_len, index, block, row, false)
                    })
                    .collect::<Result<Vec<_>, Error>>()?,
            );
        }
        Self::from_group_algebra(coefficient_columns)
    }

    /// Builds a code from group-algebra coefficient columns.
    pub fn from_group_algebra(mut coefficient_columns: Vec<Vec<B>>) -> Result<Self, Error> {
        if coefficient_columns.len() <= 1 || !coefficient_columns.len().is_power_of_two() {
            return Err(Error::EncodingError);
        }
        let message_len = coefficient_columns[0].len();
        if message_len == 0
            || !message_len.is_power_of_two()
            || coefficient_columns
                .iter()
                .any(|column| column.len() != message_len)
        {
            return Err(Error::EncodingError);
        }
        let inverse_len = B::from_u64(message_len as u64)
            .try_inverse()
            .ok_or(Error::EncodingError)?;
        for alpha in &mut coefficient_columns {
            short_wht_in_place(alpha)?;
            alpha.iter_mut().for_each(|value| *value *= inverse_len);
        }
        let systematic_first = coefficient_columns[0]
            .iter()
            .all(|&value| value == inverse_len);
        let generator_digest = qa_generator_digest::<B>(&coefficient_columns)?;
        let prepared_parity = if systematic_first && coefficient_columns.len() == 2 {
            B::prepare_encoding_multipliers(&coefficient_columns[1])
        } else { None };
        Ok(Self {
            message_len,
            index: coefficient_columns.len(),
            alphas: coefficient_columns,
            prepared_parity,
            generator_digest,
            nonzero_frequency_sampling: false,
            systematic_first,
        })
    }

    /// Returns the message length.
    pub fn message_len(&self) -> usize {
        self.message_len
    }

    /// Returns the code index and inverse rate.
    pub fn index(&self) -> usize {
        self.index
    }

    /// Returns the codeword length.
    pub fn codeword_len(&self) -> Result<usize, Error> {
        self.message_len
            .checked_mul(self.index)
            .ok_or(Error::EncodingError)
    }

    /// Returns the fixed alpha columns defining the QA generator.
    pub fn alphas(&self) -> &[Vec<B>] {
        &self.alphas
    }

    /// Returns the cached digest binding every generator coefficient.
    pub fn generator_digest(&self) -> [u8; 32] {
        self.generator_digest
    }

    /// Computes the finite-length QA distance over the small base field.
    pub fn distance_bound(&self, security_bits: usize) -> Result<QaDistanceBound, Error> {
        if B::NONZERO_QA_SAMPLING {
            if !self.nonzero_frequency_sampling {
                return Err(Error::InvalidParameters(
                    "B.8 requires parameters from nonzero-frequency sampling".into(),
                ));
            }
            return qa_nonzero_distance_bound(
                self.message_len,
                self.index,
                security_bits,
                u64::try_from(B::CARDINALITY).map_err(|_| Error::EncodingError)?,
            );
        }
        qa_distance_bound_with_modulus_log2(
            self.message_len,
            self.index,
            security_bits,
            (B::CARDINALITY as f64).log2(),
        )
    }

    /// Computes the entropy-only QA relative-distance threshold over the base field.
    pub fn entropy_distance_bound(&self) -> Result<QaDistanceBound, Error> {
        qa_entropy_distance_bound_with_modulus_log2(
            self.message_len,
            self.index,
            (B::CARDINALITY as f64).log2(),
        )
    }

    /// Returns the internal target for code generation and row sampling.
    pub fn query_security_bits(&self, security_bits: usize) -> Result<usize, Error> {
        if B::NONZERO_QA_SAMPLING {
            security_bits
                .checked_add(NONZERO_QA_SECURITY_SLACK_BITS)
                .ok_or(Error::EncodingError)
        } else {
            Ok(security_bits)
        }
    }

    /// Selects the actual QA distance used by proximity checks.
    pub fn proximity_distance_bound(&self, security_bits: usize) -> Result<QaDistanceBound, Error> {
        if B::NONZERO_QA_SAMPLING || B::FINITE_LENGTH_PROXIMITY {
            self.distance_bound(self.query_security_bits(security_bits)?)
        } else {
            self.entropy_distance_bound()
        }
    }

    /// Computes the QA distance after random-foldable extension layers.
    pub fn distance_after_rf(
        &self,
        security_bits: usize,
        rf_layers: usize,
    ) -> Result<QaRfDistanceBound, Error> {
        let qa = self.proximity_distance_bound(security_bits)?;
        let relative_distance = rf_extended_relative_distance(
            qa.relative_distance,
            self.codeword_len()?,
            rf_layers,
            (B::CARDINALITY as f64).log2(),
            self.query_security_bits(security_bits)?,
        )?;
        Ok(QaRfDistanceBound {
            qa,
            relative_distance,
            rf_layers,
        })
    }

    /// Computes the RF-adjusted query count for this short-field QA code.
    pub fn num_queries(
        &self,
        security_bits: usize,
        rf_layers: usize,
        available_rows: usize,
    ) -> Result<usize, Error> {
        if available_rows == 0 {
            return Err(Error::EncodingError);
        }
        let bound = match self.distance_after_rf(security_bits, rf_layers) {
            Ok(bound) => bound,
            Err(Error::InvalidParameters(_)) => return Ok(available_rows),
            Err(error) => return Err(error),
        };
        Ok(queries_for_relative_distance(
            self.query_security_bits(security_bits)?,
            bound.relative_distance,
        )?
        .min(available_rows))
    }

    /// Encodes one base-field message into a preallocated codeword.
    pub fn encode_base_into(&self, message: &[B], output: &mut [B]) -> Result<(), Error> {
        self.encode_base_into_recorded(message, output, &mut EncodeRecorder::<false>::default())
    }

    /// Encodes with the normal code path and records coarse stage durations.
    ///
    /// WHT time includes cache-stripe gather/scatter. Stage durations are worker
    /// sums: with more than two codeword blocks, nested parity parallelism can
    /// make their sum exceed `total`, which is elapsed wall time for this call.
    #[cfg(feature = "profile-encoding")]
    pub fn encode_base_into_profiled(
        &self,
        message: &[B],
        output: &mut [B],
    ) -> Result<QaEncodeProfile, Error> {
        let started = std::time::Instant::now();
        let mut recorder = EncodeRecorder::<true>::default();
        self.encode_base_into_recorded(message, output, &mut recorder)?;
        let mut profile = recorder.into_profile();
        profile.total = started.elapsed();
        Ok(profile)
    }

    fn encode_base_into_recorded<const PROFILE: bool>(
        &self,
        message: &[B],
        output: &mut [B],
        recorder: &mut EncodeRecorder<PROFILE>,
    ) -> Result<(), Error> {
        if message.len() != self.message_len || output.len() != self.codeword_len()? {
            return Err(Error::EncodingError);
        }
        recorder.measure(EncodeStage::Copy, message.len(), || {
            output[..self.message_len].copy_from_slice(message);
        });
        self.encode_base_in_place_recorded(output, recorder)
    }

    /// Encodes a base-field message stored in the first codeword block.
    pub fn encode_base_in_place(&self, output: &mut [B]) -> Result<(), Error> {
        self.encode_base_in_place_recorded(output, &mut EncodeRecorder::<false>::default())
    }

    fn encode_base_in_place_recorded<const PROFILE: bool>(
        &self,
        output: &mut [B],
        recorder: &mut EncodeRecorder<PROFILE>,
    ) -> Result<(), Error> {
        if output.len() != self.codeword_len()? {
            return Err(Error::EncodingError);
        }
        if let Some(prepared) = &self.prepared_parity {
            // The common systematic index-two path. Each tile is consumed while
            // its freshly copied / multiplied values are still cache resident.
            let (message, spectrum) = output.split_at_mut(self.message_len);
            recorder.measure(EncodeStage::CopyWht, message.len(), || {
                for (target, source) in spectrum.chunks_mut(WHT_TILE).zip(message.chunks(WHT_TILE)) {
                    target.copy_from_slice(source);
                    short_wht_tile(target);
                }
                short_wht_finish_high(spectrum)
            })?;
            return recorder.measure(EncodeStage::MultiplyWht, spectrum.len(), || {
                for (tile, (values, alpha)) in spectrum.chunks_mut(WHT_TILE)
                    .zip(self.alphas[1].chunks(WHT_TILE)).enumerate()
                {
                    B::mul_assign_prepared_slice(values, alpha, prepared, tile * WHT_TILE);
                    short_wht_tile(values);
                }
                short_wht_finish_high(spectrum)
            });
        }
        if self.systematic_first {
            // Keep the systematic message intact and borrow the final parity
            // block as transform storage. Restoring the first block by a third
            // WHT and m normalizing multiplications is unnecessary.
            let last = output.len() - self.message_len;
            let (prefix, spectrum) = output.split_at_mut(last);
            let (message, remaining_blocks) = prefix.split_at_mut(self.message_len);
            recorder.measure(EncodeStage::Copy, message.len(), || {
                spectrum.copy_from_slice(message);
            });
            recorder.measure(EncodeStage::Wht, 0, || short_wht_blocked_in_place(spectrum))?;
            encode_remaining_base_blocks(
                spectrum,
                remaining_blocks,
                &self.alphas[1..self.index - 1],
                recorder,
            )?;
            recorder.measure(EncodeStage::Multiplication, spectrum.len(), || {
                B::mul_assign_slice(spectrum, &self.alphas[self.index - 1]);
            });
            return recorder.measure(EncodeStage::Wht, 0, || short_wht_blocked_in_place(spectrum));
        }
        let (spectrum, remaining_blocks) = output.split_at_mut(self.message_len);
        recorder.measure(EncodeStage::Wht, 0, || short_wht_blocked_in_place(spectrum))?;
        encode_remaining_base_blocks(spectrum, remaining_blocks, &self.alphas[1..], recorder)?;
        recorder.measure(EncodeStage::Multiplication, spectrum.len(), || {
            B::mul_assign_slice(spectrum, &self.alphas[0]);
        });
        recorder.measure(EncodeStage::Wht, 0, || short_wht_blocked_in_place(spectrum))
    }

    /// Encodes one extension-field message with the lifted QA generator.
    pub fn encode_extension(&self, message: &[B::Extension]) -> Result<Vec<B::Extension>, Error> {
        let mut output = vec![B::Extension::zero(); self.codeword_len()?];
        self.encode_extension_into(message, &mut output)?;
        Ok(output)
    }

    /// Encodes one extension-field message into a preallocated codeword.
    pub fn encode_extension_into(
        &self,
        message: &[B::Extension],
        output: &mut [B::Extension],
    ) -> Result<(), Error> {
        self.encode_extension_into_with_wht(
            message,
            output,
            short_wht_blocked_in_place::<B::Extension>,
        )
    }

    fn encode_extension_into_with_wht(
        &self,
        message: &[B::Extension],
        output: &mut [B::Extension],
        wht: fn(&mut [B::Extension]) -> Result<(), Error>,
    ) -> Result<(), Error> {
        if message.len() != self.message_len || output.len() != self.codeword_len()? {
            return Err(Error::EncodingError);
        }
        output[..self.message_len].copy_from_slice(message);
        if self.systematic_first {
            let last = output.len() - self.message_len;
            let (prefix, spectrum) = output.split_at_mut(last);
            let (_, remaining_blocks) = prefix.split_at_mut(self.message_len);
            spectrum.copy_from_slice(message);
            wht(spectrum)?;
            encode_remaining_extension_blocks::<B>(
                spectrum,
                remaining_blocks,
                &self.alphas[1..self.index - 1],
                wht,
            )?;
            spectrum
                .iter_mut()
                .zip(&self.alphas[self.index - 1])
                .for_each(|(value, alpha)| *value = value.mul_base(*alpha));
            return wht(spectrum);
        }
        let (spectrum, remaining_blocks) = output.split_at_mut(self.message_len);
        wht(spectrum)?;
        encode_remaining_extension_blocks::<B>(spectrum, remaining_blocks, &self.alphas[1..], wht)?;
        spectrum
            .iter_mut()
            .zip(&self.alphas[0])
            .for_each(|(value, alpha)| *value = value.mul_base(*alpha));
        wht(spectrum)
    }

    /// Applies the lifted QA generator to an extension-field right vector.
    pub fn generator_right_projection_extension(
        &self,
        right: &[B::Extension],
    ) -> Result<Vec<B::Extension>, Error> {
        // A standalone projection has no outer column parallelism.
        let mut output = vec![B::Extension::zero(); self.codeword_len()?];
        self.encode_extension_into_with_wht(
            right,
            &mut output,
            short_wht_in_place::<B::Extension>,
        )?;
        Ok(output)
    }

    /// Projects an extension-field left vector through the lifted QA generator.
    pub fn generator_left_projection_extension(
        &self,
        left: &[B::Extension],
    ) -> Result<Vec<B::Extension>, Error> {
        if left.len() != self.codeword_len()? {
            return Err(Error::EncodingError);
        }
        let mut projected = vec![B::Extension::zero(); self.message_len];
        let mut values = vec![B::Extension::zero(); self.message_len];
        // W(sum_i alpha_i * W(left_i)) needs one inverse transform after
        // frequency accumulation. A systematic first block is the identity.
        let skip = usize::from(self.systematic_first);
        for (block, alpha) in left
            .chunks_exact(self.message_len)
            .zip(&self.alphas)
            .skip(skip)
        {
            values.copy_from_slice(block);
            short_wht_in_place(&mut values)?;
            #[cfg(feature = "parallel")]
            if self.message_len >= 2 * WHT_TILE {
                projected
                    .par_chunks_mut(WHT_TILE)
                    .zip(values.par_chunks(WHT_TILE))
                    .zip(alpha.par_chunks(WHT_TILE))
                    .for_each(|((target, values), alpha)| {
                        for ((target, &value), &coefficient) in
                            target.iter_mut().zip(values).zip(alpha)
                        {
                            *target += value.mul_base(coefficient);
                        }
                    });
                continue;
            }
            for ((target, &value), &coefficient) in projected.iter_mut().zip(&values).zip(alpha) {
                *target += value.mul_base(coefficient);
            }
        }
        short_wht_in_place(&mut projected)?;
        if self.systematic_first {
            #[cfg(feature = "parallel")]
            if self.message_len >= 2 * WHT_TILE {
                projected
                    .par_chunks_mut(WHT_TILE)
                    .zip(left[..self.message_len].par_chunks(WHT_TILE))
                    .for_each(|(target, block)| {
                        for (target, &value) in target.iter_mut().zip(block) {
                            *target += value;
                        }
                    });
                return Ok(projected);
            }
            for (target, &value) in projected.iter_mut().zip(&left[..self.message_len]) {
                *target += value;
            }
        }
        Ok(projected)
    }

    /// Evaluates the lifted QA generator's multilinear extension directly.
    pub fn evaluate_generator_mle_extension(
        &self,
        left_point: &[B::Extension],
        right_point: &[B::Extension],
    ) -> Result<B::Extension, Error> {
        if left_point.len() != self.codeword_len()?.ilog2() as usize
            || right_point.len() != self.message_len.ilog2() as usize
        {
            return Err(Error::InvalidNumberOfVariables);
        }
        let left = short_tensor_weights(left_point);
        let right = short_tensor_weights(right_point);
        let projected = self.generator_right_projection_extension(&right)?;
        Ok(left
            .into_iter()
            .zip(projected)
            .map(|(left_value, projected_value)| left_value * projected_value)
            .fold(B::Extension::zero(), |sum, value| sum + value))
    }

    /// Reduces a prover-side generator claim to an MLE claim on `H_j`.
    pub fn reduce_prover(
        &self,
        claim: &ShortQaMatrixClaim<B::Extension>,
    ) -> Result<ShortQaGeneratorReduction<B::Extension>, Error> {
        let expected = self.evaluate_generator_mle_extension(&claim.point.0, &claim.point.1)?;
        let mut reduction = reduce_short_generator_claim::<B>(self.message_len, self.index, claim)?;
        reduction.ok = expected == claim.value;
        Ok(reduction)
    }

    /// Reconstructs the alpha-matrix claim without reading generator coefficients.
    pub fn verify_as_subprotocol(
        message_len: usize,
        index: usize,
        claim: &ShortQaMatrixClaim<B::Extension>,
    ) -> Result<ShortQaGeneratorReduction<B::Extension>, Error> {
        reduce_short_generator_claim::<B>(message_len, index, claim)
    }
}

/// A random-foldable source code derived from a short-field QA code.
#[derive(Clone, Debug)]
pub struct ShortQaRandomFoldableCode<B: ShortBaseField> {
    target_level: usize,
    switch_ratio: usize,
    diagonals: Vec<Vec<B>>,
}

impl<B: ShortBaseField> ShortQaRandomFoldableCode<B> {
    /// Samples `RF^log2(switch_ratio)(G_target)` over the base field.
    pub fn sample<R: RngCore>(
        target_level: usize,
        target: &ShortQaCode<B>,
        switch_ratio: usize,
        rng: &mut R,
    ) -> Result<Self, Error> {
        let layers = rf_layers_for_switch(switch_ratio, false)?;
        let mut child_len = target.codeword_len()?;
        let mut diagonals = Vec::with_capacity(layers);
        for _ in 0..layers {
            let mut diagonal = Vec::with_capacity(child_len);
            for _ in 0..child_len {
                diagonal.push(sample_nonzero_base::<B, _>(rng));
            }
            diagonals.push(diagonal);
            child_len = child_len.checked_mul(2).ok_or(Error::EncodingError)?;
        }
        Ok(Self {
            target_level,
            switch_ratio,
            diagonals,
        })
    }

    /// Returns the level of the QA generator underlying this derived code.
    pub fn target_level(&self) -> usize {
        self.target_level
    }

    /// Returns the source message length.
    pub fn message_len(&self, target: &ShortQaCode<B>) -> Result<usize, Error> {
        target
            .message_len()
            .checked_mul(self.switch_ratio)
            .ok_or(Error::EncodingError)
    }

    /// Returns the derived codeword length.
    pub fn codeword_len(&self, target: &ShortQaCode<B>) -> Result<usize, Error> {
        target
            .codeword_len()?
            .checked_mul(self.switch_ratio)
            .ok_or(Error::EncodingError)
    }

    /// Returns the number of random-foldable layers.
    pub fn layers(&self) -> usize {
        self.diagonals.len()
    }

    /// Returns one RF diagonal by its one-based layer index.
    pub fn diagonal(&self, one_based_layer: usize) -> Result<&[B], Error> {
        self.diagonals
            .get(one_based_layer.checked_sub(1).ok_or(Error::EncodingError)?)
            .map(Vec::as_slice)
            .ok_or(Error::EncodingError)
    }

    /// Encodes one base-field source message with the derived code.
    pub fn encode_base(&self, target: &ShortQaCode<B>, message: &[B]) -> Result<Vec<B>, Error> {
        let mut output = vec![B::zero(); self.codeword_len(target)?];
        self.encode_base_into(target, message, &mut output)?;
        Ok(output)
    }

    /// Encodes a base-field source message into a caller-provided buffer.
    pub fn encode_base_into(
        &self,
        target: &ShortQaCode<B>,
        message: &[B],
        output: &mut [B],
    ) -> Result<(), Error> {
        if message.len() != self.message_len(target)?
            || output.len() != self.codeword_len(target)?
        {
            return Err(Error::EncodingError);
        }
        self.encode_base_implicit_zero_into(target, message, output)
    }

    /// Encodes a base-field prefix while treating its missing suffix as zero.
    pub fn encode_base_implicit_zero_into(
        &self,
        target: &ShortQaCode<B>,
        message_prefix: &[B],
        output: &mut [B],
    ) -> Result<(), Error> {
        let message_len = self.message_len(target)?;
        let target_message_len = target.message_len();
        let target_codeword_len = target.codeword_len()?;
        if message_prefix.len() > message_len || output.len() != self.codeword_len(target)? {
            return Err(Error::EncodingError);
        }
        output.fill(B::zero());
        #[cfg(feature = "parallel")]
        output
            .par_chunks_mut(target_codeword_len)
            .enumerate()
            .try_for_each(|(block, encoded_block)| {
                let start = block
                    .checked_mul(target_message_len)
                    .ok_or(Error::EncodingError)?;
                if start < message_prefix.len() {
                    let end = start
                        .checked_add(target_message_len)
                        .ok_or(Error::EncodingError)?
                        .min(message_prefix.len());
                    encoded_block[..end - start].copy_from_slice(&message_prefix[start..end]);
                    target.encode_base_in_place(encoded_block)?;
                }
                Ok(())
            })?;
        #[cfg(not(feature = "parallel"))]
        output
            .chunks_mut(target_codeword_len)
            .enumerate()
            .try_for_each(|(block, encoded_block)| {
                let start = block
                    .checked_mul(target_message_len)
                    .ok_or(Error::EncodingError)?;
                if start < message_prefix.len() {
                    let end = start
                        .checked_add(target_message_len)
                        .ok_or(Error::EncodingError)?
                        .min(message_prefix.len());
                    encoded_block[..end - start].copy_from_slice(&message_prefix[start..end]);
                    target.encode_base_in_place(encoded_block)?;
                }
                Ok(())
            })?;
        apply_base_rf_layers(output, target_codeword_len, &self.diagonals)
    }

    /// Encodes one extension-field source message by scalar extension.
    pub fn encode_extension(
        &self,
        target: &ShortQaCode<B>,
        message: &[B::Extension],
    ) -> Result<Vec<B::Extension>, Error> {
        let mut output = vec![B::Extension::zero(); self.codeword_len(target)?];
        self.encode_extension_into(target, message, &mut output)?;
        Ok(output)
    }

    /// Encodes an extension-field source message into a caller-provided buffer.
    pub fn encode_extension_into(
        &self,
        target: &ShortQaCode<B>,
        message: &[B::Extension],
        output: &mut [B::Extension],
    ) -> Result<(), Error> {
        self.encode_with_layers_into(target, message, self.layers(), output)
    }

    /// Applies the complete derived generator to a right vector.
    pub fn right_projection_extension(
        &self,
        target: &ShortQaCode<B>,
        right: &[B::Extension],
    ) -> Result<Vec<B::Extension>, Error> {
        self.encode_extension(target, right)
    }

    /// Applies the generator obtained after a prefix of the RF layers.
    pub fn right_projection_extension_with_layers(
        &self,
        target: &ShortQaCode<B>,
        right: &[B::Extension],
        layers: usize,
    ) -> Result<Vec<B::Extension>, Error> {
        if layers > self.layers() {
            return Err(Error::EncodingError);
        }
        let scale = 1usize
            .checked_shl(layers as u32)
            .ok_or(Error::EncodingError)?;
        let output_len = target
            .codeword_len()?
            .checked_mul(scale)
            .ok_or(Error::EncodingError)?;
        let mut output = vec![B::Extension::zero(); output_len];
        self.encode_with_layers_into(target, right, layers, &mut output)?;
        Ok(output)
    }

    fn encode_with_layers_into(
        &self,
        target: &ShortQaCode<B>,
        message: &[B::Extension],
        layers: usize,
        output: &mut [B::Extension],
    ) -> Result<(), Error> {
        if layers > self.layers() {
            return Err(Error::EncodingError);
        }
        let target_message_len = target.message_len();
        let target_codeword_len = target.codeword_len()?;
        let scale = 1usize
            .checked_shl(layers as u32)
            .ok_or(Error::EncodingError)?;
        let expected_message_len = target_message_len
            .checked_mul(scale)
            .ok_or(Error::EncodingError)?;
        let expected_output_len = target_codeword_len
            .checked_mul(scale)
            .ok_or(Error::EncodingError)?;
        if message.len() != expected_message_len || output.len() != expected_output_len {
            return Err(Error::EncodingError);
        }
        output.fill(B::Extension::zero());
        #[cfg(feature = "parallel")]
        output
            .par_chunks_mut(target_codeword_len)
            .zip(message.par_chunks(target_message_len))
            .try_for_each(|(encoded_block, message_block)| {
                target.encode_extension_into(message_block, encoded_block)
            })?;
        #[cfg(not(feature = "parallel"))]
        output
            .chunks_mut(target_codeword_len)
            .zip(message.chunks(target_message_len))
            .try_for_each(|(encoded_block, message_block)| {
                target.encode_extension_into(message_block, encoded_block)
            })?;

        let mut child_len = target_codeword_len;
        for diagonal in self.diagonals.iter().take(layers) {
            if diagonal.len() != child_len {
                return Err(Error::EncodingError);
            }
            let group_len = child_len.checked_mul(2).ok_or(Error::EncodingError)?;
            #[cfg(feature = "parallel")]
            output.par_chunks_mut(group_len).for_each(|group| {
                let (left, right) = group.split_at_mut(child_len);
                for index in 0..child_len {
                    let direct = left[index];
                    let twisted = right[index].mul_base(diagonal[index]);
                    left[index] = direct + twisted;
                    right[index] = direct - twisted;
                }
            });
            #[cfg(not(feature = "parallel"))]
            output.chunks_mut(group_len).for_each(|group| {
                let (left, right) = group.split_at_mut(child_len);
                for index in 0..child_len {
                    let direct = left[index];
                    let twisted = right[index].mul_base(diagonal[index]);
                    left[index] = direct + twisted;
                    right[index] = direct - twisted;
                }
            });
            child_len = group_len;
        }
        Ok(())
    }

    /// Projects a left vector through the complete derived generator.
    pub fn left_projection_extension(
        &self,
        target: &ShortQaCode<B>,
        left: &[B::Extension],
    ) -> Result<Vec<B::Extension>, Error> {
        if left.len() != self.codeword_len(target)? {
            return Err(Error::EncodingError);
        }
        short_qa_rfc_left_projection_recursive::<B>(target, left, &self.diagonals, self.layers())
    }
}

fn sample_unbiased_base<B: ShortBaseField, R: RngCore>(rng: &mut R) -> B {
    B::sample_uniform(rng)
}

fn sample_nonzero_base<B: ShortBaseField, R: RngCore>(rng: &mut R) -> B {
    loop {
        let value = sample_unbiased_base::<B, _>(rng);
        if value != B::zero() {
            return value;
        }
    }
}

fn apply_base_rf_layers<B: ShortBaseField>(
    output: &mut [B],
    target_codeword_len: usize,
    diagonals: &[Vec<B>],
) -> Result<(), Error> {
    let mut child_len = target_codeword_len;
    for diagonal in diagonals {
        if diagonal.len() != child_len {
            return Err(Error::EncodingError);
        }
        let group_len = child_len.checked_mul(2).ok_or(Error::EncodingError)?;
        #[cfg(feature = "parallel")]
        output.par_chunks_mut(group_len).for_each(|group| {
            let (left, right) = group.split_at_mut(child_len);
            for index in 0..child_len {
                let direct = left[index];
                let twisted = diagonal[index] * right[index];
                left[index] = direct + twisted;
                right[index] = direct - twisted;
            }
        });
        #[cfg(not(feature = "parallel"))]
        output.chunks_mut(group_len).for_each(|group| {
            let (left, right) = group.split_at_mut(child_len);
            for index in 0..child_len {
                let direct = left[index];
                let twisted = diagonal[index] * right[index];
                left[index] = direct + twisted;
                right[index] = direct - twisted;
            }
        });
        child_len = group_len;
    }
    Ok(())
}

fn short_qa_rfc_left_projection_recursive<B: ShortBaseField>(
    target: &ShortQaCode<B>,
    left: &[B::Extension],
    diagonals: &[Vec<B>],
    layer: usize,
) -> Result<Vec<B::Extension>, Error> {
    if layer == 0 {
        return target.generator_left_projection_extension(left);
    }
    if left.len() % 2 != 0 {
        return Err(Error::EncodingError);
    }
    let half = left.len() / 2;
    let (top, bottom) = left.split_at(half);
    let diagonal = diagonals.get(layer - 1).ok_or(Error::EncodingError)?;
    if diagonal.len() != half {
        return Err(Error::EncodingError);
    }
    let direct = top
        .iter()
        .zip(bottom)
        .map(|(top, bottom)| *top + *bottom)
        .collect::<Vec<_>>();
    let twisted = top
        .iter()
        .zip(bottom)
        .zip(diagonal)
        .map(|((top, bottom), diagonal)| (*top - *bottom).mul_base(*diagonal))
        .collect::<Vec<_>>();
    let mut projected =
        short_qa_rfc_left_projection_recursive::<B>(target, &direct, diagonals, layer - 1)?;
    projected.extend(short_qa_rfc_left_projection_recursive::<B>(
        target,
        &twisted,
        diagonals,
        layer - 1,
    )?);
    Ok(projected)
}

fn reduce_short_generator_claim<B: ShortBaseField>(
    message_len: usize,
    index: usize,
    claim: &ShortQaMatrixClaim<B::Extension>,
) -> Result<ShortQaGeneratorReduction<B::Extension>, Error> {
    if message_len == 0
        || !message_len.is_power_of_two()
        || index <= 1
        || !index.is_power_of_two()
        || claim.point.0.len() != message_len.ilog2() as usize + index.ilog2() as usize
        || claim.point.1.len() != message_len.ilog2() as usize
    {
        return Err(Error::InvalidNumberOfVariables);
    }
    let block_variables = index.ilog2() as usize;
    let (block_point, left_message_point) = claim.point.0.split_at(block_variables);
    let (left_normalization, transformed_left) =
        transform_short_wht_tensor_point::<B>(left_message_point)?;
    let (right_normalization, transformed_right) =
        transform_short_wht_tensor_point::<B>(&claim.point.1)?;
    let merged_point = merge_short_tensor_points(&transformed_left, &transformed_right)?;
    let normalization = left_normalization * right_normalization;
    let inverse_normalization = normalization
        .try_inverse()
        .ok_or(Error::InvalidCommitment)?;
    Ok(ShortQaGeneratorReduction {
        ok: true,
        normalization,
        alpha_claim: ShortQaMatrixClaim {
            value: claim.value * inverse_normalization,
            point: (merged_point, block_point.to_vec()),
        },
    })
}

fn transform_short_wht_tensor_point<B: ShortBaseField>(
    point: &[B::Extension],
) -> Result<(B::Extension, Vec<B::Extension>), Error> {
    let one = B::Extension::one();
    let mut normalization = one;
    let mut transformed = Vec::with_capacity(point.len());
    for value in point {
        let denominator = one + *value;
        let inverse = denominator.try_inverse().ok_or(Error::InvalidCommitment)?;
        normalization *= denominator;
        transformed.push((one - *value) * inverse);
    }
    Ok((normalization, transformed))
}

fn merge_short_tensor_points<E: ShortFieldElement>(
    left: &[E],
    right: &[E],
) -> Result<Vec<E>, Error> {
    if left.len() != right.len() {
        return Err(Error::InvalidNumberOfVariables);
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left_value, right_value)| *left_value * *right_value)
        .collect())
}

/// A geometric short-prime QA sequence down to the direct base level.
#[derive(Clone, Debug)]
pub struct ShortQaFold<B: ShortBaseField> {
    witness_size: usize,
    extension_columns: usize,
    switch_ratio: usize,
    rf_layers: usize,
    recursive_index: usize,
    codes: Vec<ShortQaCode<B>>,
    generator_digest: [u8; 32],
}

impl<B: ShortBaseField> ShortQaFold<B> {
    /// Builds a code sequence for a base-field witness.
    ///
    /// Every level uses inverse rate two. Explicit inverse rate four remains
    /// available through `new_with_recursive_index` for historical comparisons.
    pub fn new(
        witness_size: usize,
        extension_columns: usize,
        top_index: usize,
        seed: [u8; 32],
    ) -> Result<Self, Error> {
        Self::new_with_recursive_index(
            witness_size,
            extension_columns,
            top_index,
            QAFOLD_RECURSIVE_INDEX,
            seed,
        )
    }

    /// Builds a sequence with inverse rate two or four below the field-selected top levels.
    pub fn new_with_recursive_index(
        witness_size: usize,
        extension_columns: usize,
        top_index: usize,
        recursive_index: usize,
        seed: [u8; 32],
    ) -> Result<Self, Error> {
        if witness_size == 0
            || !witness_size.is_power_of_two()
            || extension_columns <= 1
            || !extension_columns.is_power_of_two()
            || top_index != QAFOLD_TOP_INDEX
            || !matches!(recursive_index, 2 | 4)
        {
            return Err(Error::EncodingError);
        }
        let base_columns = extension_columns
            .checked_mul(B::EXTENSION_DEGREE)
            .ok_or(Error::EncodingError)?;
        if witness_size % base_columns != 0 {
            return Err(Error::EncodingError);
        }
        let mut message_len = witness_size / base_columns;
        let mut descending = Vec::new();
        loop {
            descending.push(message_len);
            if message_len < extension_columns {
                break;
            }
            if message_len % extension_columns != 0 {
                return Err(Error::EncodingError);
            }
            message_len /= extension_columns;
        }
        descending.reverse();
        let rf_layers = rf_layers_for_switch(extension_columns, false)?;
        let levels = descending.len();
        let codes = descending
            .into_iter()
            .enumerate()
            .map(|(level, length)| {
                ShortQaCode::sample_systematic(
                    length,
                    qafold_short_index_for_level_with_recursive_index::<B>(
                        level,
                        levels,
                        recursive_index,
                    )?,
                    derive_level_seed(&seed, level),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let generator_digest = qa_sequence_generator_digest::<B>(&codes);
        Ok(Self {
            witness_size,
            extension_columns,
            switch_ratio: extension_columns,
            rf_layers,
            recursive_index,
            codes,
            generator_digest,
        })
    }

    /// Returns the base-field witness size.
    pub fn witness_size(&self) -> usize {
        self.witness_size
    }

    /// Returns the number of extension-field columns after packing.
    pub fn extension_columns(&self) -> usize {
        self.extension_columns
    }

    /// Returns the number of base-field columns at the top level.
    pub fn base_columns(&self) -> Result<usize, Error> {
        self.extension_columns
            .checked_mul(B::EXTENSION_DEGREE)
            .ok_or(Error::EncodingError)
    }

    /// Returns the geometric switch ratio.
    pub fn switch_ratio(&self) -> usize {
        self.switch_ratio
    }

    /// Returns the number of RF layers used at each geometric code switch.
    pub fn rf_layers(&self) -> usize {
        self.rf_layers
    }

    /// Returns the RF-depth used by the active proximity model.
    pub fn proximity_rf_layers(&self) -> usize {
        match B::DECIDE_SETUP_MODE {
            ShortQaDecideSetupMode::RfV1 => self.rf_layers,
            ShortQaDecideSetupMode::DirectInterleavedV1 => 0,
        }
    }

    /// Returns the selected inverse rate below the field-selected top levels.
    pub fn recursive_index(&self) -> usize {
        self.recursive_index
    }

    /// Returns all codes in ascending message-length order.
    pub fn codes(&self) -> &[ShortQaCode<B>] {
        &self.codes
    }

    /// Returns the cached digest binding all QA generators in this sequence.
    pub fn generator_digest(&self) -> [u8; 32] {
        self.generator_digest
    }

    /// Returns the top-level base-field QA code.
    pub fn top_code(&self) -> Result<&ShortQaCode<B>, Error> {
        self.codes.last().ok_or(Error::EncodingError)
    }

    /// Returns the query count for one recursive level under the active setup policy.
    pub fn num_queries(
        &self,
        level: usize,
        security_bits: usize,
        available_rows: usize,
    ) -> Result<usize, Error> {
        self.codes
            .get(level)
            .ok_or(Error::EncodingError)?
            .num_queries(security_bits, self.proximity_rf_layers(), available_rows)
    }

    /// Materializes and encodes one group of extension-degree base columns.
    ///
    /// This is the largest live base-field buffer in the streaming prover.
    pub fn encode_top_group(&self, group: usize, witness_seed: u64) -> Result<Vec<Vec<B>>, Error> {
        if group >= self.extension_columns {
            return Err(Error::EncodingError);
        }
        let code = self.top_code()?;
        let message_len = code.message_len();
        let codeword_len = code.codeword_len()?;
        #[cfg(feature = "parallel")]
        let encoded = (0..B::EXTENSION_DEGREE)
            .into_par_iter()
            .map(|coefficient| {
                let column = group * B::EXTENSION_DEGREE + coefficient;
                let mut codeword = vec![B::zero(); codeword_len];
                codeword[..message_len]
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(row, value)| {
                        let position = column * message_len + row;
                        *value = B::from_u64(mix64(witness_seed ^ position as u64));
                    });
                code.encode_base_in_place(&mut codeword)?;
                Ok(codeword)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        #[cfg(not(feature = "parallel"))]
        let encoded = (0..B::EXTENSION_DEGREE)
            .map(|coefficient| {
                let column = group * B::EXTENSION_DEGREE + coefficient;
                let mut codeword = vec![B::zero(); codeword_len];
                for (row, value) in codeword[..message_len].iter_mut().enumerate() {
                    let position = column * message_len + row;
                    *value = B::from_u64(mix64(witness_seed ^ position as u64));
                }
                code.encode_base_in_place(&mut codeword)?;
                Ok(codeword)
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(encoded)
    }

    /// Packs one encoded base-column group and accumulates its challenge product.
    pub fn accumulate_top_group(
        &self,
        encoded_columns: &[Vec<B>],
        challenges: &[B::Extension],
        accumulator: &mut [B::Extension],
    ) -> Result<(), Error> {
        let codeword_len = self.top_code()?.codeword_len()?;
        if encoded_columns.len() != B::EXTENSION_DEGREE
            || challenges.len() != B::EXTENSION_DEGREE
            || encoded_columns
                .iter()
                .any(|column| column.len() != codeword_len)
            || accumulator.len() != codeword_len
        {
            return Err(Error::EncodingError);
        }
        #[cfg(feature = "parallel")]
        accumulator
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, target)| {
                *target += encoded_columns
                    .iter()
                    .zip(challenges)
                    .map(|(column, challenge)| challenge.mul_base(column[row]))
                    .fold(B::Extension::zero(), |left, right| left + right);
            });
        #[cfg(not(feature = "parallel"))]
        accumulator
            .iter_mut()
            .enumerate()
            .for_each(|(row, target)| {
                *target += encoded_columns
                    .iter()
                    .zip(challenges)
                    .map(|(column, challenge)| challenge.mul_base(column[row]))
                    .fold(B::Extension::zero(), |left, right| left + right);
            });
        Ok(())
    }

    /// Derives one deterministic extension-field challenge for a packed column.
    pub fn packing_challenge(group: usize, seed: u64) -> B::Extension {
        B::Extension::from_basis_coefficients_fn(|coefficient| {
            B::from_u64(mix64(seed ^ ((group as u64) << 32) ^ coefficient as u64))
        })
    }
}

fn short_tensor_weights<F: ShortFieldElement>(point: &[F]) -> Vec<F> {
    let mut weights = vec![F::one()];
    for challenge in point {
        let mut next = Vec::with_capacity(weights.len() * 2);
        for weight in weights {
            next.push(weight);
            next.push(weight * *challenge);
        }
        weights = next;
    }
    weights
}

/// Applies an unnormalized Walsh--Hadamard transform in place.
pub fn short_wht_in_place<F: ShortFieldElement>(values: &mut [F]) -> Result<(), Error> {
    if values.is_empty() || !values.len().is_power_of_two() {
        return Err(Error::EncodingError);
    }
    // Low stages stay in each cache tile. High stages divide both halves into
    // tiles, so the final butterfly stage can use every worker too.
    #[cfg(feature = "parallel")]
    if values.len() >= 2 * WHT_TILE {
        values.par_chunks_mut(WHT_TILE).for_each(short_wht_tile);
        let mut half = WHT_TILE;
        while half < values.len() {
            values.par_chunks_mut(2 * half).for_each(|block| {
                let (left, right) = block.split_at_mut(half);
                left.par_chunks_mut(WHT_TILE)
                    .zip(right.par_chunks_mut(WHT_TILE))
                    .for_each(|(left, right)| F::butterfly_slice(left, right));
            });
            half *= 2;
        }
        return Ok(());
    }
    short_wht_blocked_in_place(values)
}

const WHT_TILE: usize = 1 << 12;

fn short_wht_tile<F: ShortFieldElement>(values: &mut [F]) {
    let mut half = 1usize;
    while half < values.len() {
        let block_len = half * 2;
        values.chunks_mut(block_len).for_each(|block| {
            let (left, right) = block.split_at_mut(half);
            F::butterfly_slice(left, right);
        });
        half = block_len;
    }
}

/// Fuses the low twelve WHT stages within a tile; callers parallelize over columns.
/// Large columns transpose small stripes so their high stages stay in cache too.
fn short_wht_blocked_in_place<F: ShortFieldElement>(values: &mut [F]) -> Result<(), Error> {
    if values.is_empty() || !values.len().is_power_of_two() {
        return Err(Error::EncodingError);
    }
    for tile in values.chunks_mut(WHT_TILE) {
        short_wht_tile(tile);
    }
    short_wht_finish_high(values)
}

fn short_wht_finish_high<F: ShortFieldElement>(values: &mut [F]) -> Result<(), Error> {
    if values.len() >= 1 << 18 && short_wht_high_striped(values) {
        return Ok(());
    }
    let mut half = WHT_TILE;
    while half < values.len() {
        let block_len = half.checked_mul(2).ok_or(Error::EncodingError)?;
        for block in values.chunks_mut(block_len) {
            let (left, right) = block.split_at_mut(half);
            F::butterfly_slice(left, right);
        }
        half = block_len;
    }
    Ok(())
}

/// The low `log2(WHT_TILE)` dimensions have already been transformed. Their
/// coordinates are independent in the remaining dimensions, so gather a stripe
/// of those coordinates and finish every high stage before scattering it back.
/// Scratch is bounded per column; callers already parallelize across columns.
fn short_wht_high_striped<F: ShortFieldElement>(values: &mut [F]) -> bool {
    const SCRATCH_BYTES: usize = 256 << 10;
    let rows = values.len() / WHT_TILE;
    let max_stripe = SCRATCH_BYTES / core::mem::size_of::<F>().max(1) / rows;
    if max_stripe == 0 {
        return false;
    }
    let stripe = WHT_TILE.min(1usize << max_stripe.ilog2());
    let mut scratch = vec![F::zero(); rows * stripe];
    for offset in (0..WHT_TILE).step_by(stripe) {
        for (target, row) in scratch
            .chunks_exact_mut(stripe)
            .zip(values.chunks_exact(WHT_TILE))
        {
            target.copy_from_slice(&row[offset..offset + stripe]);
        }
        // Do not transform within a stripe: these are the already-completed
        // low coordinates, not additional high dimensions.
        let mut half = stripe;
        while half < scratch.len() {
            for block in scratch.chunks_exact_mut(2 * half) {
                let (left, right) = block.split_at_mut(half);
                F::butterfly_slice(left, right);
            }
            half *= 2;
        }
        for (source, row) in scratch
            .chunks_exact(stripe)
            .zip(values.chunks_exact_mut(WHT_TILE))
        {
            row[offset..offset + stripe].copy_from_slice(source);
        }
    }
    true
}

#[cfg(test)]
mod wht_stripe_tests {
    use super::*;
    use crate::FullPrime65521Ext8 as F;
    use rand::{SeedableRng, rngs::StdRng};

    // The previous blocked implementation, including the same SIMD butterflies,
    // is kept here solely for a like-for-like performance comparison.
    fn previous_blocked<F: ShortFieldElement>(values: &mut [F]) {
        for tile in values.chunks_mut(WHT_TILE) {
            short_wht_tile(tile);
        }
        let mut half = WHT_TILE;
        while half < values.len() {
            for block in values.chunks_exact_mut(2 * half) {
                let (left, right) = block.split_at_mut(half);
                F::butterfly_slice(left, right);
            }
            half *= 2;
        }
    }

    // Independent correctness oracle: no optimized butterfly or WHT helper.
    fn scalar_reference<F: ShortFieldElement>(values: &mut [F]) {
        let mut half = 1;
        while half < values.len() {
            for block in values.chunks_exact_mut(2 * half) {
                for i in 0..half {
                    let (a, b) = (block[i], block[i + half]);
                    block[i] = a + b;
                    block[i + half] = a - b;
                }
            }
            half *= 2;
        }
    }

    #[test]
    fn large_blocked_wht_matches_independent_scalar_reference() {
        fn exercise<B: ShortBaseField>() {
            let mut rng = StdRng::seed_from_u64(0x7374_7269_7065_7768);
            let input = (0..1 << 18)
                .map(|_| B::sample_uniform(&mut rng))
                .collect::<Vec<_>>();
            let mut actual = input.clone();
            short_wht_blocked_in_place(&mut actual).unwrap();
            let mut expected = input;
            scalar_reference(&mut expected);
            assert_eq!(actual, expected);
        }
        exercise::<Prime65521>();
        exercise::<F>();
    }

    #[test]
    #[ignore = "manual single-thread cache-stripe timing"]
    fn compare_blocked_wht_timings() {
        use std::{hint::black_box, time::Instant};
        let mut rng = StdRng::seed_from_u64(0x5748_545f_5449_4d45);
        for log_n in [18, 20, 22] {
            let input = (0..1 << log_n)
                .map(|_| F::sample_uniform(&mut rng))
                .collect::<Vec<_>>();
            let mut old = input.clone();
            let mut new = input.clone();
            previous_blocked(&mut old);
            short_wht_blocked_in_place(&mut new).unwrap();
            assert_eq!(old, new);
            let mut old_ms = Vec::new();
            let mut new_ms = Vec::new();
            for run in 0..7 {
                old.copy_from_slice(&input);
                new.copy_from_slice(&input);
                let mut old_run = || {
                    let start = Instant::now();
                    previous_blocked(black_box(&mut old));
                    old_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                };
                let mut new_run = || {
                    let start = Instant::now();
                    short_wht_blocked_in_place(black_box(&mut new)).unwrap();
                    new_ms.push(start.elapsed().as_secs_f64() * 1000.0);
                };
                if run % 2 == 0 {
                    old_run();
                    new_run();
                } else {
                    new_run();
                    old_run();
                }
                assert_eq!(old, new);
            }
            old_ms.sort_by(f64::total_cmp);
            new_ms.sort_by(f64::total_cmp);
            eprintln!(
                "WHT log_n={log_n} old_ms={:.3} striped_ms={:.3} ratio={:.3} old={old_ms:?} striped={new_ms:?}",
                old_ms[3],
                new_ms[3],
                old_ms[3] / new_ms[3]
            );
        }
    }
}

fn encode_remaining_base_blocks<B: ShortBaseField, const PROFILE: bool>(
    spectrum: &[B],
    remaining_blocks: &mut [B],
    alphas: &[Vec<B>],
    recorder: &mut EncodeRecorder<PROFILE>,
) -> Result<(), Error> {
    let message_len = spectrum.len();
    let encode_block =
        |(block, alpha): (&mut [B], &Vec<B>)| {
            let mut child = EncodeRecorder::<PROFILE>::default();
            child.measure(EncodeStage::Multiplication, message_len, || {
                block.iter_mut().zip(spectrum).zip(alpha).for_each(
                    |((target, value), coefficient)| {
                        *target = *value * *coefficient;
                    },
                );
            });
            child.measure(EncodeStage::Wht, 0, || short_wht_blocked_in_place(block))?;
            Ok::<_, Error>(child)
        };
    #[cfg(feature = "parallel")]
    {
        let blocks = remaining_blocks
            .par_chunks_mut(message_len)
            .zip(alphas.par_iter());
        #[cfg(feature = "profile-encoding")]
        if PROFILE {
            let children = blocks.map(encode_block).try_reduce(
                EncodeRecorder::<PROFILE>::default,
                |mut combined, child| {
                    combined.merge(child);
                    Ok(combined)
                },
            )?;
            recorder.merge(children);
            return Ok(());
        }
        blocks.try_for_each(|block| encode_block(block).map(|_| ()))?;
        // In an ordinary build the recorder is zero-sized and has no work.
        let _ = recorder;
    }
    #[cfg(not(feature = "parallel"))]
    remaining_blocks
        .chunks_mut(message_len)
        .zip(alphas)
        .try_for_each(|block| {
            recorder.merge(encode_block(block)?);
            Ok::<_, Error>(())
        })?;
    Ok(())
}

fn encode_remaining_extension_blocks<B: ShortBaseField>(
    spectrum: &[B::Extension],
    remaining_blocks: &mut [B::Extension],
    alphas: &[Vec<B>],
    wht: fn(&mut [B::Extension]) -> Result<(), Error>,
) -> Result<(), Error> {
    let message_len = spectrum.len();
    #[cfg(feature = "parallel")]
    remaining_blocks
        .par_chunks_mut(message_len)
        .zip(alphas.par_iter())
        .try_for_each(|(block, alpha)| {
            block
                .iter_mut()
                .zip(spectrum)
                .zip(alpha)
                .for_each(|((target, value), coefficient)| {
                    *target = value.mul_base(*coefficient);
                });
            wht(block)
        })?;
    #[cfg(not(feature = "parallel"))]
    remaining_blocks
        .chunks_mut(message_len)
        .zip(alphas)
        .try_for_each(|(block, alpha)| {
            block
                .iter_mut()
                .zip(spectrum)
                .zip(alpha)
                .for_each(|((target, value), coefficient)| {
                    *target = value.mul_base(*coefficient);
                });
            wht(block)
        })?;
    Ok(())
}

fn derive_level_seed(seed: &[u8; 32], level: usize) -> [u8; 32] {
    let mut hasher = Blake2s256::new();
    update_hash_frame(&mut hasher, b"domain", b"qafold-short/qa-level-seed/v1");
    update_hash_frame(&mut hasher, b"master-seed", seed);
    update_hash_frame(&mut hasher, b"level", &(level as u64).to_le_bytes());
    hasher.finalize().into()
}

fn sample_generator_element<B: ShortBaseField>(
    seed: &[u8; 32],
    message_len: usize,
    index: usize,
    block: usize,
    row: usize,
    nonzero_frequency: bool,
) -> Result<B, Error> {
    let range = 1u128 << 64;
    let modulus = B::MODULUS as u128;
    let alphabet_size = if nonzero_frequency {
        modulus - 1
    } else {
        modulus
    };
    let acceptance_limit = range - range % alphabet_size;
    for attempt in 0..u64::MAX {
        let mut hasher = Blake2s256::new();
        update_hash_frame(
            &mut hasher,
            b"domain",
            if nonzero_frequency {
                b"qafold-short/qa-generator-frequency-nonzero/v2"
            } else {
                b"qafold-short/qa-generator-coefficient/v1"
            },
        );
        update_hash_frame(&mut hasher, b"field-name", B::NAME.as_bytes());
        update_hash_frame(&mut hasher, b"modulus", &B::canonical_modulus_bytes());
        update_hash_frame(&mut hasher, b"seed", seed);
        for (label, value) in [
            (b"message-length".as_slice(), message_len),
            (b"index".as_slice(), index),
            (b"block".as_slice(), block),
            (b"row".as_slice(), row),
        ] {
            update_hash_frame(&mut hasher, label, &(value as u64).to_le_bytes());
        }
        update_hash_frame(&mut hasher, b"attempt", &attempt.to_le_bytes());
        let digest = hasher.finalize();
        if B::UNIFORM_BYTES > 8 {
            for lane in digest.chunks_exact(B::UNIFORM_BYTES) {
                if let Some(value) = B::from_uniform_bytes(lane) {
                    if !nonzero_frequency || value != B::zero() {
                        return Ok(value);
                    }
                }
            }
            continue;
        }
        for lane in digest.chunks_exact(8) {
            let candidate =
                u64::from_le_bytes(lane.try_into().map_err(|_| Error::EncodingError)?) as u128;
            if candidate < acceptance_limit {
                return B::from_canonical_u64_checked(
                    (candidate % alphabet_size) as u64 + u64::from(nonzero_frequency),
                )
                .ok_or(Error::EncodingError);
            }
        }
    }
    Err(Error::EncodingError)
}

fn qa_generator_digest<B: ShortBaseField>(alphas: &[Vec<B>]) -> Result<[u8; 32], Error> {
    let mut hasher = Blake2s256::new();
    update_hash_frame(&mut hasher, b"domain", b"qafold-short/qa-generator/v1");
    update_hash_frame(&mut hasher, b"field-name", B::NAME.as_bytes());
    update_hash_frame(&mut hasher, b"modulus", &B::canonical_modulus_bytes());
    update_hash_frame(
        &mut hasher,
        b"columns",
        &(alphas.len() as u64).to_le_bytes(),
    );
    let rows = alphas.first().map(Vec::len).unwrap_or_default();
    update_hash_frame(&mut hasher, b"rows", &(rows as u64).to_le_bytes());
    let mut encoded = vec![0u8; B::CANONICAL_BYTES];
    for (column, alpha) in alphas.iter().enumerate() {
        update_hash_frame(&mut hasher, b"column", &(column as u64).to_le_bytes());
        for value in alpha {
            value.write_canonical_bytes(&mut encoded)?;
            update_hash_frame(&mut hasher, b"coefficient", &encoded);
        }
    }
    Ok(hasher.finalize().into())
}

fn qa_sequence_generator_digest<B: ShortBaseField>(codes: &[ShortQaCode<B>]) -> [u8; 32] {
    let mut hasher = Blake2s256::new();
    update_hash_frame(
        &mut hasher,
        b"domain",
        b"qafold-short/qa-sequence-generators/v1",
    );
    update_hash_frame(&mut hasher, b"levels", &(codes.len() as u64).to_le_bytes());
    for (level, code) in codes.iter().enumerate() {
        update_hash_frame(&mut hasher, b"level", &(level as u64).to_le_bytes());
        update_hash_frame(
            &mut hasher,
            b"message-length",
            &(code.message_len() as u64).to_le_bytes(),
        );
        update_hash_frame(&mut hasher, b"index", &(code.index() as u64).to_le_bytes());
        update_hash_frame(&mut hasher, b"generator", &code.generator_digest());
    }
    hasher.finalize().into()
}

fn update_hash_frame(hasher: &mut Blake2s256, label: &[u8], bytes: &[u8]) {
    hasher.update((label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

#[inline]
fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

#[inline]
fn karatsuba_2_unreduced(left: [u64; 2], right: [u64; 2]) -> [u64; 3] {
    let low = left[0] * right[0];
    let high = left[1] * right[1];
    let middle = (left[0] + left[1]) * (right[0] + right[1]) - low - high;
    [low, middle, high]
}

#[inline]
fn karatsuba_4_unreduced(left: [u64; 4], right: [u64; 4]) -> [u64; 7] {
    let left_low = [left[0], left[1]];
    let left_high = [left[2], left[3]];
    let right_low = [right[0], right[1]];
    let right_high = [right[2], right[3]];
    let low = karatsuba_2_unreduced(left_low, right_low);
    let high = karatsuba_2_unreduced(left_high, right_high);
    let middle = karatsuba_2_unreduced(
        [left[0] + left[2], left[1] + left[3]],
        [right[0] + right[2], right[1] + right[3]],
    );
    let mut output = [0; 7];
    for index in 0..3 {
        output[index] += low[index];
        output[index + 4] += high[index];
        output[index + 2] += middle[index] - low[index] - high[index];
    }
    output
}

#[inline]
fn karatsuba_8_unreduced(left: [Prime65521; 8], right: [Prime65521; 8]) -> [u64; 15] {
    let left_low = array::from_fn(|i| u64::from(left[i].0));
    let left_high = array::from_fn(|i| u64::from(left[i + 4].0));
    let right_low = array::from_fn(|i| u64::from(right[i].0));
    let right_high = array::from_fn(|i| u64::from(right[i + 4].0));
    let low = karatsuba_4_unreduced(left_low, right_low);
    let high = karatsuba_4_unreduced(left_high, right_high);
    let middle = karatsuba_4_unreduced(
        array::from_fn(|index| left_low[index] + left_high[index]),
        array::from_fn(|index| right_low[index] + right_high[index]),
    );
    let mut output = [0; 15];
    for index in 0..7 {
        output[index] += low[index];
        output[index + 8] += high[index];
        output[index + 4] += middle[index] - low[index] - high[index];
    }
    output
}

/// Reduces one coefficient after X^8 = 17 and an optional canonical addend.
/// The maximum is 120*(65520^2)+65520 < 2^39. One pseudo-Mersenne fold
/// therefore fits in u32; the existing two-fold reducer finishes the job.
#[inline]
fn reduce_octic_coefficient(value: u64) -> Prime65521 {
    debug_assert!(value <= 120 * 65_520u64 * 65_520 + 65_520);
    let folded = (value & 0xffff) + 15 * (value >> 16);
    Prime65521(Prime65521::reduce_u32(folded as u32))
}
