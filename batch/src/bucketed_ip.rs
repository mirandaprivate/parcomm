//! Decreasing-size BVA metadata and packed EO-LiteBullet for byte witnesses.
//!
//! Physical witness entries remain in `i8` storage. The first packed EO step
//! converts only one field element per packing stride, after which all binary
//! folds reuse that reduced field allocation in place.

use std::{
    fmt::{Display, Formatter},
    io::Cursor,
    time::Instant,
};

use ark_crypto_primitives::sponge::Absorb;
use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};
use rayon::prelude::*;

use crate::fiat_shamir::FiatShamir;

/// A rewindable source backed by exactly one signed byte per witness entry.
pub trait BvaI8Source {
    type Error;

    fn layout(&self) -> &BvaLayout;

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[i8])) -> Result<(), Self::Error>;
}

/// Metadata for a decreasing-size bucketed vector assembly.
///
/// Each input vector owns exactly one real power-of-two cover bucket. Internal
/// suffixes and the final logical suffix are implicit zeros, not empty buckets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BvaLayout {
    logical_len: usize,
    bucket_sizes: Vec<usize>,
    vector_to_bucket: Vec<usize>,
    vector_lengths: Vec<usize>,
}

impl BvaLayout {
    pub fn from_lengths(lengths: &[usize]) -> Result<Self, BvaError> {
        if lengths.is_empty() {
            return Err(BvaError::EmptyAssembly);
        }
        if lengths.contains(&0) {
            return Err(BvaError::ZeroLengthVector);
        }

        let mut sorted = lengths
            .iter()
            .copied()
            .enumerate()
            .map(|(original_index, len)| {
                len.checked_next_power_of_two()
                    .map(|cover| (original_index, cover))
                    .ok_or(BvaError::LengthOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        sorted.sort_by(|(left_index, left_cover), (right_index, right_cover)| {
            right_cover
                .cmp(left_cover)
                .then_with(|| left_index.cmp(right_index))
        });

        let mut active_len = 0_usize;
        let mut bucket_sizes = Vec::with_capacity(lengths.len());
        let mut vector_to_bucket = vec![0_usize; lengths.len()];
        for (original_index, cover) in sorted {
            vector_to_bucket[original_index] = bucket_sizes.len();
            bucket_sizes.push(cover);
            active_len = active_len
                .checked_add(cover)
                .ok_or(BvaError::LengthOverflow)?;
        }
        let logical_len = active_len
            .checked_next_power_of_two()
            .ok_or(BvaError::LengthOverflow)?;

        Ok(Self {
            logical_len,
            bucket_sizes,
            vector_to_bucket,
            vector_lengths: lengths.to_vec(),
        })
    }

    pub fn logical_len(&self) -> usize {
        self.logical_len
    }

    pub fn bucket_sizes(&self) -> &[usize] {
        &self.bucket_sizes
    }

    pub fn vector_to_bucket(&self) -> &[usize] {
        &self.vector_to_bucket
    }

    pub fn vector_lengths(&self) -> &[usize] {
        &self.vector_lengths
    }

    pub fn active_len(&self) -> usize {
        self.bucket_sizes.iter().sum()
    }

    pub fn bucket_offsets(&self) -> Vec<usize> {
        let mut offset = 0_usize;
        self.bucket_sizes
            .iter()
            .map(|&size| {
                let current = offset;
                offset += size;
                current
            })
            .collect()
    }

    pub fn vector_offset(&self, vector_index: usize) -> Option<usize> {
        let bucket = *self.vector_to_bucket.get(vector_index)?;
        self.bucket_offsets().get(bucket).copied()
    }

    pub fn padding_bucket_count(&self) -> usize {
        0
    }

    pub fn materialized_padding_entries(&self) -> usize {
        0
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TbvTerm<F> {
    eta: F,
    factors: Vec<(F, F)>,
}

/// A compact sum of tensor terms over one BVA bucket.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TbvBucket<F> {
    terms: Vec<TbvTerm<F>>,
}

impl<F: From<u64>> TbvBucket<F> {
    pub fn new(eta: F, challenges: Vec<F>) -> Self {
        Self {
            terms: vec![TbvTerm {
                eta,
                factors: challenges
                    .into_iter()
                    .map(|challenge| (F::from(1_u64), challenge))
                    .collect(),
            }],
        }
    }

    pub fn from_terms(terms: Vec<(F, Vec<F>)>) -> Self {
        Self {
            terms: terms
                .into_iter()
                .map(|(eta, challenges)| TbvTerm {
                    eta,
                    factors: challenges
                        .into_iter()
                        .map(|challenge| (F::from(1_u64), challenge))
                        .collect(),
                })
                .collect(),
        }
    }

    fn from_factor_terms(terms: Vec<(F, Vec<(F, F)>)>) -> Self {
        Self {
            terms: terms
                .into_iter()
                .map(|(eta, factors)| TbvTerm { eta, factors })
                .collect(),
        }
    }
}

impl<F: Field> TbvBucket<F> {
    fn reduced_scalar(&self) -> F {
        self.terms.iter().fold(F::zero(), |sum, term| {
            debug_assert!(term.factors.is_empty());
            sum + term.eta
        })
    }

    fn fold_interior_owned(mut self, challenge: F) -> Self {
        for term in &mut self.terms {
            debug_assert!(!term.factors.is_empty());
            let (left, right) = term.factors.remove(0);
            term.eta *= left + challenge * right;
        }
        self
    }
}

/// A public tensor basis represented only over real BVA buckets.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorBasisVector<F> {
    layout: BvaLayout,
    buckets: Vec<TbvBucket<F>>,
}

/// What one bucket needs from the witness during the packed EO cross-term pass.
///
/// For a bucket at or above the packing stride, each tensor term contributes one
/// projection per lane, and those projections are the only place the cross terms
/// read the witness. Everything the prover derives from them costs
/// `O(stride^2)`. Publishing the request lets an accelerator that already holds
/// the witness compute the projections and hand them back, replacing a full CPU
/// pass over the witness.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedLaneRequest<F> {
    /// Where the bucket starts in the logical vector.
    pub logical_offset: usize,
    /// How many logical entries the bucket covers.
    pub cover: usize,
    /// Tensor factors above the lane bits, one entry per term.
    pub term_row_factors: Vec<Vec<(F, F)>>,
}

/// Lane projections indexed by bucket, then term, then lane.
///
/// A `None` entry leaves that bucket to the CPU, which is what buckets narrower
/// than the packing stride always need because they share a packed row.
pub type PackedLaneProjections<F> = Vec<Option<Vec<Vec<F>>>>;

/// Where one accelerated reduced vector comes from and how long it is.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PackedReducedSpec {
    /// Index of the reduced bucket the vector belongs to.
    pub bucket: usize,
    /// Where the bucket's witness entries start in the logical vector.
    pub logical_offset: usize,
    /// Entries the reduced vector holds.
    pub len: usize,
}

/// Device-side owner of the packed EO reduced vectors.
///
/// The reduced vector is the largest intermediate this reduction builds: one
/// field element per packed row, so the witness length divided by the packing
/// stride. An implementation keeps it in device memory across the rounds where
/// it is large and returns it only once folding has shrunk it, which keeps a
/// witness-sized buffer off the bus entirely.
///
/// The prover keeps every piece of protocol bookkeeping, including the cover
/// sizes, the symbolic tensor bases and the singleton merges. An implementation
/// only answers the three questions that depend on the values themselves.
pub trait PackedEoReducedVectors<F> {
    type Error: ToString;

    /// Builds one reduced vector per spec, where entry `j` is
    /// `sum_{t < stride} witness[logical_offset + j*stride + t] * inverse_powers[t]`.
    fn build(
        &mut self,
        specs: &[PackedReducedSpec],
        inverse_powers: &[F],
    ) -> Result<(), Self::Error>;

    /// `sum_i values[i] * tensor(factors[slot])[i]`, one entry per slot.
    fn weighted_sums(&mut self, factors: &[Vec<(F, F)>]) -> Result<Vec<F>, Self::Error>;

    /// `(sum_i values[2i] * T(i), sum_i values[2i+1] * T(i))` per slot, where
    /// `T` is the tensor of `factors[slot]`.
    fn even_odd_sums(&mut self, factors: &[Vec<(F, F)>]) -> Result<Vec<(F, F)>, Self::Error>;

    /// `values[i] = values[2i] + challenge * values[2i+1]`, halving every slot.
    fn fold(&mut self, challenge: F) -> Result<(), Self::Error>;

    /// Returns every still-resident slot whose length has fallen to `threshold`
    /// or below, retiring it so later calls skip it.
    ///
    /// A retired slot contributes zero to [`Self::weighted_sums`] and
    /// [`Self::even_odd_sums`] and is ignored by [`Self::fold`], because the
    /// prover has taken its values back and folds them itself.
    fn retire_below(&mut self, threshold: usize) -> Result<Vec<Option<Vec<F>>>, Self::Error>;
}

/// Commitment-first Fiat--Shamir driver for a hiding Packed EO reduction.
///
/// The driver receives each scalar only on the prover side. It must commit to
/// the ordered scalars before returning the challenge that folds them. The
/// verifier reconstructs the same challenges from the commitments, so none of
/// these values need to appear in the public proof.
pub trait PackedEoChallengeTranscript<F> {
    fn packed_challenge(&mut self, positive: &[F], negative: &[F]) -> Result<F, String>;

    fn binary_challenge(&mut self, round: usize, iota: F, rho: F) -> Result<F, String>;

    fn bind_reduced(&mut self, a_reduced: F, b_reduced: F) -> Result<(), String>;
}

/// Shortest reduced vector worth handing to an accelerator.
const ACCELERATED_MINIMUM_ENTRIES: usize = 1 << 16;

/// Length at which a reduced vector goes back to the CPU.
///
/// Below this the rounds cost less than the launches they would need, and
/// finishing on the CPU also keeps the singleton merges in one place.
const ACCELERATED_RETIRE_ENTRIES: usize = 1 << 12;

impl<F> TensorBasisVector<F> {
    pub fn new(layout: &BvaLayout, buckets: Vec<TbvBucket<F>>) -> Result<Self, BvaError> {
        if buckets.len() != layout.bucket_sizes.len() {
            return Err(BvaError::BucketCount {
                expected: layout.bucket_sizes.len(),
                actual: buckets.len(),
            });
        }
        for (bucket_index, (&bucket_size, bucket)) in
            layout.bucket_sizes.iter().zip(&buckets).enumerate()
        {
            let expected = bucket_size.ilog2() as usize;
            for term in &bucket.terms {
                if term.factors.len() != expected {
                    return Err(BvaError::ChallengeCount {
                        bucket: bucket_index,
                        expected,
                        actual: term.factors.len(),
                    });
                }
            }
        }
        Ok(Self {
            layout: layout.clone(),
            buckets,
        })
    }

    pub fn layout(&self) -> &BvaLayout {
        &self.layout
    }
}

impl<F: Field> TensorBasisVector<F> {
    /// Builds the tensor coefficients induced by low-to-high EO challenges.
    pub fn global_tensor(layout: &BvaLayout, challenges: &[F]) -> Result<Self, BvaError> {
        let expected = layout.logical_len.ilog2() as usize;
        if challenges.len() != expected {
            return Err(BvaError::FoldChallengeCount {
                expected,
                actual: challenges.len(),
            });
        }
        let offsets = layout.bucket_offsets();
        let buckets = layout
            .bucket_sizes
            .iter()
            .zip(offsets)
            .map(|(&size, offset)| {
                let interior_log = size.ilog2() as usize;
                let mut eta = F::one();
                for (bit, challenge) in challenges.iter().enumerate().skip(interior_log) {
                    if (offset >> bit) & 1 == 1 {
                        eta *= challenge;
                    }
                }
                TbvBucket::from_factor_terms(vec![(
                    eta,
                    challenges[..interior_log]
                        .iter()
                        .map(|challenge| (F::one(), *challenge))
                        .collect(),
                )])
            })
            .collect();
        Self::new(layout, buckets)
    }

    pub fn physical_coefficient_at(
        &self,
        vector_index: usize,
        entry_index: usize,
    ) -> Result<F, BvaError> {
        let Some(&bucket_index) = self.layout.vector_to_bucket.get(vector_index) else {
            return Err(BvaError::VectorIndex);
        };
        if entry_index >= self.layout.vector_lengths[vector_index] {
            return Err(BvaError::VectorLength);
        }
        Ok(self.bucket_coefficient_at(bucket_index, entry_index))
    }

    fn bucket_coefficient_at(&self, bucket_index: usize, local_index: usize) -> F {
        self.buckets[bucket_index]
            .terms
            .iter()
            .fold(F::zero(), |sum, term| {
                sum + term.factors.iter().enumerate().fold(
                    term.eta,
                    |product, (bit, (left, right))| {
                        product
                            * if (local_index >> bit) & 1 == 0 {
                                *left
                            } else {
                                *right
                            }
                    },
                )
            })
    }

    fn logical_coefficient_at(&self, offsets: &[usize], flat_index: usize) -> F {
        if flat_index >= self.layout.active_len() {
            return F::zero();
        }
        let bucket_index = offsets.partition_point(|offset| *offset <= flat_index) - 1;
        self.bucket_coefficient_at(bucket_index, flat_index - offsets[bucket_index])
    }

    pub fn fold_to_scalar(&self, challenges: &[F]) -> Result<F, BvaError> {
        let expected = self.layout.logical_len.ilog2() as usize;
        if challenges.len() != expected {
            return Err(BvaError::FoldChallengeCount {
                expected,
                actual: challenges.len(),
            });
        }
        let mut buckets = self
            .layout
            .bucket_sizes
            .iter()
            .copied()
            .zip(self.buckets.clone())
            .collect::<Vec<_>>();
        for &challenge in challenges {
            let singleton_start = buckets
                .iter()
                .position(|(size, _)| *size == 1)
                .unwrap_or(buckets.len());
            let singleton_count = buckets.len() - singleton_start;
            let mut folded = Vec::with_capacity(singleton_start + singleton_count.div_ceil(2));
            for (size, basis) in buckets.drain(..singleton_start) {
                folded.push((size / 2, basis.fold_interior_owned(challenge)));
            }
            let singletons = buckets;
            let mut pairs = singletons.chunks_exact(2);
            for pair in &mut pairs {
                folded.push((
                    1,
                    TbvBucket::from_factor_terms(vec![(
                        pair[0].1.reduced_scalar() + challenge * pair[1].1.reduced_scalar(),
                        Vec::new(),
                    )]),
                ));
            }
            if let Some(last) = pairs.remainder().first() {
                folded.push(last.clone());
            }
            buckets = folded;
        }
        if buckets.len() != 1 || buckets[0].0 != 1 {
            return Err(BvaError::MissingFinalValue);
        }
        Ok(buckets
            .pop()
            .expect("checked final bucket")
            .1
            .reduced_scalar())
    }
}

impl<F: Clone> TensorBasisVector<F> {
    /// Describes the lane projections the packed EO cross terms read from the
    /// witness, one entry per bucket.
    ///
    /// Buckets narrower than `stride` share a packed row and take a different
    /// path that also needs cross-bucket terms, so they report `None` and stay
    /// on the CPU.
    pub fn packed_lane_requests(&self, stride: usize) -> Vec<Option<PackedLaneRequest<F>>> {
        let stride_log = stride.ilog2() as usize;
        let offsets = self.layout.bucket_offsets();
        self.layout
            .bucket_sizes
            .iter()
            .zip(offsets)
            .zip(&self.buckets)
            .map(|((&cover, logical_offset), bucket)| {
                if cover < stride {
                    return None;
                }
                Some(PackedLaneRequest {
                    logical_offset,
                    cover,
                    term_row_factors: bucket
                        .terms
                        .iter()
                        .map(|term| term.factors[stride_log..].to_vec())
                        .collect(),
                })
            })
            .collect()
    }
}

impl<F: PrimeField> TensorBasisVector<F> {
    /// Computes the physical i8 witness inner product without expanding cover
    /// zeros or evaluating one tensor coefficient at a time.
    pub fn inner_product_i8_streaming<S: BvaI8Source>(
        &self,
        source: &mut S,
    ) -> Result<F, StreamingEoError<S::Error>> {
        let layout = source.layout().clone();
        if self.layout != layout {
            return Err(BvaError::LayoutMismatch.into());
        }
        let mut visited = vec![false; layout.vector_lengths.len()];
        let mut result = F::zero();
        let mut bva_error = None;
        source
            .for_each_vector(&mut |vector_index, values| {
                if bva_error.is_some() {
                    return;
                }
                let Some(&bucket_index) = layout.vector_to_bucket.get(vector_index) else {
                    bva_error = Some(BvaError::VectorIndex);
                    return;
                };
                if visited[vector_index] {
                    bva_error = Some(BvaError::DuplicateVector);
                    return;
                }
                visited[vector_index] = true;
                if values.len() != layout.vector_lengths[vector_index] {
                    bva_error = Some(BvaError::VectorLength);
                    return;
                }
                for term in &self.buckets[bucket_index].terms {
                    let (projected, _) =
                        tensor_project_i8_prefix_strided(values, 0, 1, values.len(), &term.factors);
                    result += term.eta * projected;
                }
            })
            .map_err(StreamingEoError::Source)?;
        if let Some(error) = bva_error {
            return Err(error.into());
        }
        if visited.iter().any(|was_visited| !was_visited) {
            return Err(BvaError::MissingVector.into());
        }
        Ok(result)
    }
}

impl<F: PrimeField + Absorb> TensorBasisVector<F> {
    /// Binds the exact compact BVA layout and tensor terms to an outer
    /// transcript without expanding logical coefficients or implicit zeros.
    pub fn absorb_transcript(&self, transcript: &mut FiatShamir<F>) {
        transcript.push(&0x5442_5654_5241_4e53_u64);
        transcript.push(&(self.layout.logical_len as u64));
        transcript.push(&(self.layout.vector_lengths.len() as u64));
        let offsets = self.layout.bucket_offsets();
        for vector_index in 0..self.layout.vector_lengths.len() {
            let bucket_index = self.layout.vector_to_bucket[vector_index];
            transcript.push(&(self.layout.vector_lengths[vector_index] as u64));
            transcript.push(&(offsets[bucket_index] as u64));
        }
        transcript.push(&(self.buckets.len() as u64));
        for bucket in &self.buckets {
            transcript.push(&(bucket.terms.len() as u64));
            for term in &bucket.terms {
                transcript.push(&term.eta);
                transcript.push(&(term.factors.len() as u64));
                for (left, right) in &term.factors {
                    transcript.push(left);
                    transcript.push(right);
                }
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PaddingWorkCounters {
    pub materialized_padding_entries: u64,
    pub padding_ip_field_ops: u64,
    pub padding_fold_field_ops: u64,
    pub real_ip_field_ops: u64,
    pub real_fold_field_ops: u64,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StreamingEoProfile {
    pub source_passes: usize,
    pub physical_entries_per_pass: usize,
    pub total_visited_entries: usize,
    pub maximum_live_vector_entries: usize,
    pub materialized_padding_entries: usize,
    pub padding_field_operations: u64,
    pub timings: StreamingEoTimings,
}

/// Wall-clock split of the packed EO prover stages.
#[derive(Clone, Copy, Debug, Default)]
pub struct StreamingEoTimings {
    /// First witness pass, producing the packed cross terms.
    pub cross_terms_seconds: f64,
    /// Second witness pass, folding `stride` bytes into one field element.
    pub fold_seconds: f64,
    /// Consistency check on the reduced vector.
    pub reduced_check_seconds: f64,
    /// LiteBullet rounds over the reduced vector.
    pub binary_rounds_seconds: f64,
}

// Timings vary run to run, so equality compares only the deterministic
// counters that the thread-count determinism tests care about.
impl PartialEq for StreamingEoProfile {
    fn eq(&self, other: &Self) -> bool {
        self.source_passes == other.source_passes
            && self.physical_entries_per_pass == other.physical_entries_per_pass
            && self.total_visited_entries == other.total_visited_entries
            && self.maximum_live_vector_entries == other.maximum_live_vector_entries
            && self.materialized_padding_entries == other.materialized_padding_entries
            && self.padding_field_operations == other.padding_field_operations
    }
}

impl Eq for StreamingEoProfile {}

#[derive(Debug)]
pub enum StreamingEoError<SourceError> {
    Bva(BvaError),
    Source(SourceError),
    Accelerator(String),
    ChallengeTranscript(String),
}

impl<SourceError> From<BvaError> for StreamingEoError<SourceError> {
    fn from(error: BvaError) -> Self {
        Self::Bva(error)
    }
}

impl<SourceError: Display> Display for StreamingEoError<SourceError> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bva(error) => Display::fmt(error, formatter),
            Self::Source(error) => Display::fmt(error, formatter),
            Self::Accelerator(message) => write!(formatter, "accelerator: {message}"),
            Self::ChallengeTranscript(message) => {
                write!(formatter, "challenge transcript: {message}")
            }
        }
    }
}

impl<SourceError> std::error::Error for StreamingEoError<SourceError> where
    SourceError: std::error::Error + 'static
{
}

/// Packed EO-LiteBullet for a native short-width witness and a public TBV.
#[derive(Clone, Debug, Eq, PartialEq, CanonicalDeserialize, CanonicalSerialize)]
pub struct PackedEoLiteBulletProof<F: PrimeField> {
    pub packing_stride: usize,
    pub packed_positive: Vec<F>,
    pub packed_negative: Vec<F>,
    pub binary_rounds: Vec<(F, F)>,
    pub a_reduced: F,
    pub b_reduced: F,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EoLiteBulletVerification<F> {
    pub accepted: bool,
    pub challenges: Vec<F>,
    pub challenges_inv: Vec<F>,
}

#[derive(Debug)]
pub enum EoProofDecodeError {
    Serialization(SerializationError),
    InvalidLength,
    AllocationFailed,
    TrailingBytes,
}

impl Display for EoProofDecodeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serialization(error) => Display::fmt(error, formatter),
            Self::InvalidLength => write!(formatter, "EO proof contains an invalid vector length"),
            Self::AllocationFailed => write!(formatter, "EO proof allocation failed"),
            Self::TrailingBytes => write!(formatter, "EO proof contains trailing bytes"),
        }
    }
}

impl std::error::Error for EoProofDecodeError {}

impl From<SerializationError> for EoProofDecodeError {
    fn from(error: SerializationError) -> Self {
        Self::Serialization(error)
    }
}

/// Returns the largest power-of-two packing stride supported by the field
/// storage width and the physical entry width.
pub fn packing_stride_for_bit_width<F: PrimeField>(
    logical_len: usize,
    bit_width: usize,
) -> Result<usize, BvaError> {
    if bit_width == 0 || bit_width > 32 {
        return Err(BvaError::InvalidBitWidth(bit_width));
    }
    if logical_len < 2 || !logical_len.is_power_of_two() {
        return Err(BvaError::InvalidPackingStride {
            stride: 0,
            logical_len,
        });
    }
    let field_storage_bits = (F::MODULUS_BIT_SIZE as usize).next_power_of_two();
    let ratio = field_storage_bits / bit_width;
    if ratio < 2 {
        return Err(BvaError::InvalidBitWidth(bit_width));
    }
    let stride = 1_usize << ratio.ilog2();
    Ok(stride.min(logical_len))
}

fn validate_packing_stride(logical_len: usize, stride: usize) -> Result<(), BvaError> {
    if logical_len < 2
        || !logical_len.is_power_of_two()
        || stride < 2
        || !stride.is_power_of_two()
        || stride > logical_len
    {
        return Err(BvaError::InvalidPackingStride {
            stride,
            logical_len,
        });
    }
    Ok(())
}

#[inline]
fn i8_to_field<F: PrimeField>(value: i8) -> F {
    if value >= 0 {
        F::from(value as u64)
    } else {
        -F::from((-(value as i16)) as u64)
    }
}

fn tensor_coefficients<F: Field>(factors: &[(F, F)]) -> Vec<F> {
    let mut coefficients = vec![F::one()];
    for &(left, right) in factors {
        let old_len = coefficients.len();
        coefficients.resize(2 * old_len, F::zero());
        for index in (0..old_len).rev() {
            let coefficient = coefficients[index];
            coefficients[index] = coefficient * left;
            coefficients[index + old_len] = coefficient * right;
        }
    }
    coefficients
}

fn tensor_project_i8_prefix_strided<F: PrimeField>(
    values: &[i8],
    start: usize,
    stride: usize,
    physical_len: usize,
    factors: &[(F, F)],
) -> (F, u64) {
    if physical_len == 0 {
        return (F::zero(), 0);
    }
    if factors.is_empty() {
        debug_assert_eq!(physical_len, 1);
        return (i8_to_field(values[start]), 0);
    }

    let high_bit = factors.len() - 1;
    let half = 1_usize << high_bit;
    let left_len = physical_len.min(half);
    let right_len = physical_len.saturating_sub(half);
    let recurse_left =
        || tensor_project_i8_prefix_strided(values, start, stride, left_len, &factors[..high_bit]);
    let recurse_right = || {
        tensor_project_i8_prefix_strided(
            values,
            start + half * stride,
            stride,
            right_len,
            &factors[..high_bit],
        )
    };
    let ((left, left_operations), (right, right_operations)) =
        if physical_len >= (1 << 14) && right_len != 0 {
            rayon::join(recurse_left, recurse_right)
        } else {
            (recurse_left(), recurse_right())
        };
    let (left_factor, right_factor) = factors[high_bit];
    if right_len == 0 {
        (left * left_factor, left_operations + 1)
    } else {
        (
            left * left_factor + right * right_factor,
            left_operations + right_operations + 3,
        )
    }
}

fn packed_i8_cross_terms<F: PrimeField>(
    values: &[i8],
    bucket_index: usize,
    logical_offset: usize,
    offsets: &[usize],
    tbv: &TensorBasisVector<F>,
    stride: usize,
    supplied_lanes: Option<&[Vec<F>]>,
) -> (Vec<F>, Vec<F>, F, u64) {
    let bucket_size = tbv.layout.bucket_sizes[bucket_index];
    if bucket_size >= stride {
        debug_assert_eq!(logical_offset % stride, 0);
        let stride_log = stride.ilog2() as usize;
        let mut positive = vec![F::zero(); stride - 1];
        let mut negative = vec![F::zero(); stride - 1];
        let mut trace = F::zero();
        let mut operations = 0_u64;

        for (term_index, term) in tbv.buckets[bucket_index].terms.iter().enumerate() {
            let (lane_factors, row_factors) = term.factors.split_at(stride_log);
            let lane_coefficients = tensor_coefficients(lane_factors);
            debug_assert_eq!(lane_coefficients.len(), stride);
            let lane_projections = match supplied_lanes {
                Some(lanes) => lanes[term_index].clone(),
                None => {
                    let computed = (0..stride)
                        .into_par_iter()
                        .map(|lane| {
                            let physical_len = if lane < values.len() {
                                1 + (values.len() - 1 - lane) / stride
                            } else {
                                0
                            };
                            tensor_project_i8_prefix_strided(
                                values,
                                lane,
                                stride,
                                physical_len,
                                row_factors,
                            )
                        })
                        .collect::<Vec<_>>();
                    operations += computed
                        .iter()
                        .map(|(_, term_operations)| *term_operations)
                        .sum::<u64>();
                    computed.into_iter().map(|(value, _)| value).collect()
                }
            };

            for lane in 0..stride {
                trace += term.eta * lane_projections[lane] * lane_coefficients[lane];
                operations += 3;
            }
            for distance in 1..stride {
                let mut positive_sum = F::zero();
                let mut negative_sum = F::zero();
                for lane in 0..stride - distance {
                    positive_sum += lane_projections[lane] * lane_coefficients[lane + distance];
                    negative_sum += lane_projections[lane + distance] * lane_coefficients[lane];
                    operations += 4;
                }
                positive[distance - 1] += term.eta * positive_sum;
                negative[distance - 1] += term.eta * negative_sum;
                operations += 4;
            }
        }
        return (positive, negative, trace, operations);
    }

    // Buckets smaller than one packed row may share that row. This exact
    // fallback includes cross-bucket terms while remaining O(I * stride^2).
    let vector_end = logical_offset + values.len();
    let first_row = logical_offset / stride;
    let row_count = vector_end.div_ceil(stride) - first_row;
    let (positive, negative, trace, operations, _) = (0..row_count)
        .into_par_iter()
        .with_min_len(1 << 12)
        .fold(
            || {
                (
                    vec![F::zero(); stride - 1],
                    vec![F::zero(); stride - 1],
                    F::zero(),
                    0_u64,
                    vec![F::zero(); stride],
                )
            },
            |mut state, row_index| {
                let row_start = (first_row + row_index) * stride;
                for (lane, coefficient) in state.4.iter_mut().enumerate() {
                    *coefficient = tbv.logical_coefficient_at(offsets, row_start + lane);
                }
                let entry_start = row_start.max(logical_offset);
                let entry_end = (row_start + stride).min(vector_end);
                for flat_index in entry_start..entry_end {
                    let value = values[flat_index - logical_offset];
                    if value == 0 {
                        continue;
                    }
                    let left_value = i8_to_field::<F>(value);
                    let left_lane = flat_index - row_start;
                    for (right_lane, &right_value) in state.4.iter().enumerate() {
                        if right_value.is_zero() {
                            continue;
                        }
                        let product = left_value * right_value;
                        state.3 += 1;
                        if right_lane == left_lane {
                            state.2 += product;
                        } else if right_lane > left_lane {
                            state.0[right_lane - left_lane - 1] += product;
                        } else {
                            state.1[left_lane - right_lane - 1] += product;
                        }
                    }
                }
                state
            },
        )
        .reduce(
            || {
                (
                    vec![F::zero(); stride - 1],
                    vec![F::zero(); stride - 1],
                    F::zero(),
                    0_u64,
                    vec![F::zero(); stride],
                )
            },
            |mut left, right| {
                for (left_value, right_value) in left.0.iter_mut().zip(right.0) {
                    *left_value += right_value;
                }
                for (left_value, right_value) in left.1.iter_mut().zip(right.1) {
                    *left_value += right_value;
                }
                left.2 += right.2;
                left.3 += right.3;
                left
            },
        );
    (positive, negative, trace, operations)
}

fn packed_eo_litebullet_fiat_shamir<F: PrimeField + Absorb>(
    logical_len: usize,
    packing_stride: usize,
    claimed: F,
    binding: Option<F>,
) -> FiatShamir<F> {
    let mut fs = FiatShamir::<F>::new();
    fs.push(&0x5041_434b_4544_454f_u64);
    fs.push(&(logical_len as u64));
    fs.push(&(packing_stride as u64));
    fs.push(&claimed);
    if let Some(binding) = binding {
        fs.push(&0x424f_554e_44_u64);
        fs.push(&binding);
    }
    fs
}

fn tensor_project_prefix_strided_allocation_free<F: Field>(
    values: &[F],
    start: usize,
    stride: usize,
    physical_len: usize,
    factors: &[(F, F)],
) -> (F, u64) {
    if physical_len == 0 {
        return (F::zero(), 0);
    }
    if factors.is_empty() {
        debug_assert_eq!(physical_len, 1);
        return (values[start], 0);
    }

    let high_bit = factors.len() - 1;
    let half = 1_usize << high_bit;
    let left_len = physical_len.min(half);
    let right_len = physical_len.saturating_sub(half);
    let recurse_left = || {
        tensor_project_prefix_strided_allocation_free(
            values,
            start,
            stride,
            left_len,
            &factors[..high_bit],
        )
    };
    let recurse_right = || {
        tensor_project_prefix_strided_allocation_free(
            values,
            start + half * stride,
            stride,
            right_len,
            &factors[..high_bit],
        )
    };
    let ((left, left_operations), (right, right_operations)) =
        if physical_len >= (1 << 14) && right_len != 0 {
            rayon::join(recurse_left, recurse_right)
        } else {
            (recurse_left(), recurse_right())
        };
    let (left_factor, right_factor) = factors[high_bit];
    if right_len == 0 {
        (left * left_factor, left_operations + 1)
    } else {
        (
            left * left_factor + right * right_factor,
            left_operations + right_operations + 3,
        )
    }
}

fn tensor_project_strided_allocation_free<F: Field>(
    values: &[F],
    start: usize,
    stride: usize,
    physical_len: usize,
    factors: &[(F, F)],
) -> (F, u64) {
    if physical_len == 0
        || factors
            .iter()
            .any(|(left, right)| left.is_zero() && right.is_zero())
    {
        return (F::zero(), 0);
    }

    let mut support_log = factors.len();
    let mut support_start = 0_usize;
    while support_log != 0 {
        let bit = support_log - 1;
        let (left, right) = factors[bit];
        match (left.is_zero(), right.is_zero()) {
            (true, false) => {
                support_start |= 1_usize << bit;
                support_log -= 1;
            }
            (false, true) => support_log -= 1,
            _ => break,
        }
    }
    if support_start >= physical_len {
        return (F::zero(), 0);
    }
    let support_len = 1_usize << support_log;
    let support_physical_len = support_len.min(physical_len - support_start);
    let (projected, mut operations) = tensor_project_prefix_strided_allocation_free(
        values,
        start + support_start * stride,
        stride,
        support_physical_len,
        &factors[..support_log],
    );
    if support_log == factors.len() {
        return (projected, operations);
    }
    let mut fixed_product = F::one();
    for (left, right) in &factors[support_log..] {
        fixed_product *= if left.is_zero() { *right } else { *left };
        operations += 1;
    }
    operations += 1;
    (projected * fixed_product, operations)
}

#[derive(Clone, Copy, Debug)]
struct PackedDestination {
    bucket_index: usize,
    lane_offset: usize,
}

#[derive(Clone, Debug)]
struct PackedReducedBucket<F> {
    cover_size: usize,
    values: Vec<F>,
    basis: TbvBucket<F>,
    /// Where the contributing witness entries start, so an accelerator can
    /// build this vector straight from its own copy of the witness.
    logical_offset: usize,
    /// Accelerator slot holding this bucket's values, if any.
    ///
    /// Singleton merges reorder and shorten the bucket list, so the link has to
    /// travel with the bucket rather than sit in a side table of indices.
    accelerator_slot: Option<usize>,
    /// Entries the reduced vector holds when it is built.
    ///
    /// A bucket handed to an accelerator leaves `values` empty, so its length
    /// has to be recorded separately for the build request.
    entries: usize,
}

/// Builds the reduced buckets, leaving any vector at least `unallocated_from`
/// entries long empty for an accelerator to fill.
///
/// Zeroing the largest vector costs about as much as folding it, so the buckets
/// an accelerator will own are never allocated here in the first place.
fn build_packed_reduced_buckets<F: PrimeField>(
    tbv: &TensorBasisVector<F>,
    packed_challenge: F,
    stride: usize,
    unallocated_from: usize,
) -> (Vec<PackedReducedBucket<F>>, Vec<PackedDestination>) {
    let layout = &tbv.layout;
    let stride_log = stride.ilog2() as usize;
    let offsets = layout.bucket_offsets();
    let mut challenge_powers = Vec::with_capacity(stride_log);
    let mut power = packed_challenge;
    for _ in 0..stride_log {
        challenge_powers.push(power);
        power.square_in_place();
    }
    let mut bucket_to_vector = vec![usize::MAX; layout.bucket_sizes.len()];
    for (vector_index, &bucket_index) in layout.vector_to_bucket.iter().enumerate() {
        bucket_to_vector[bucket_index] = vector_index;
    }

    let mut reduced = Vec::<PackedReducedBucket<F>>::new();
    let mut destinations = vec![
        PackedDestination {
            bucket_index: usize::MAX,
            lane_offset: 0,
        };
        layout.vector_lengths.len()
    ];
    let mut last_small_row = None::<(usize, usize)>;
    for bucket_index in 0..layout.bucket_sizes.len() {
        let bucket_size = layout.bucket_sizes[bucket_index];
        let logical_offset = offsets[bucket_index];
        let vector_index = bucket_to_vector[bucket_index];
        debug_assert_ne!(vector_index, usize::MAX);
        if bucket_size >= stride {
            debug_assert_eq!(logical_offset % stride, 0);
            let mut basis = tbv.buckets[bucket_index].clone();
            for &challenge in &challenge_powers {
                basis = basis.fold_interior_owned(challenge);
            }
            let reduced_index = reduced.len();
            let entries = layout.vector_lengths[vector_index].div_ceil(stride);
            reduced.push(PackedReducedBucket {
                cover_size: bucket_size / stride,
                values: if entries >= unallocated_from {
                    Vec::new()
                } else {
                    vec![F::zero(); entries]
                },
                basis,
                logical_offset,
                accelerator_slot: None,
                entries,
            });
            destinations[vector_index] = PackedDestination {
                bucket_index: reduced_index,
                lane_offset: 0,
            };
            continue;
        }

        let packed_row = logical_offset / stride;
        let reduced_index = match last_small_row {
            Some((previous_row, reduced_index)) if previous_row == packed_row => reduced_index,
            _ => {
                let reduced_index = reduced.len();
                reduced.push(PackedReducedBucket {
                    cover_size: 1,
                    values: vec![F::zero()],
                    basis: TbvBucket { terms: Vec::new() },
                    logical_offset,
                    accelerator_slot: None,
                    entries: 1,
                });
                last_small_row = Some((packed_row, reduced_index));
                reduced_index
            }
        };
        let lane_offset = logical_offset % stride;
        let lane_scale = packed_challenge.pow([lane_offset as u64]);
        for term in &tbv.buckets[bucket_index].terms {
            let mut eta = term.eta * lane_scale;
            for (&(left, right), &challenge) in term.factors.iter().zip(&challenge_powers) {
                eta *= left + challenge * right;
            }
            reduced[reduced_index].basis.terms.push(TbvTerm {
                eta,
                factors: Vec::new(),
            });
        }
        destinations[vector_index] = PackedDestination {
            bucket_index: reduced_index,
            lane_offset,
        };
    }
    (reduced, destinations)
}

fn packed_reduced_inner_product<F: PrimeField>(buckets: &[PackedReducedBucket<F>]) -> (F, u64) {
    buckets
        .iter()
        .map(|bucket| {
            bucket
                .basis
                .terms
                .iter()
                .fold((F::zero(), 0_u64), |(sum, operations), term| {
                    let (projected, projection_operations) = tensor_project_strided_allocation_free(
                        &bucket.values,
                        0,
                        1,
                        bucket.values.len(),
                        &term.factors,
                    );
                    (
                        sum + term.eta * projected,
                        operations + projection_operations + 2,
                    )
                })
        })
        .fold((F::zero(), 0_u64), |(sum, operations), contribution| {
            (sum + contribution.0, operations + contribution.1 + 1)
        })
}

fn packed_eo_cross_inner_products_allocation_free<F: PrimeField>(
    values: &[F],
    basis: &TbvBucket<F>,
    counters: &mut PaddingWorkCounters,
) -> (F, F) {
    let mut iota = F::zero();
    let mut rho = F::zero();
    let even_len = values.len().div_ceil(2);
    let odd_len = values.len() / 2;
    for term in &basis.terms {
        let ((left_factor, right_factor), tail) =
            term.factors.split_first().expect("non-terminal EO round");
        let ((even, even_operations), (odd, odd_operations)) = rayon::join(
            || tensor_project_strided_allocation_free(values, 0, 2, even_len, tail),
            || tensor_project_strided_allocation_free(values, 1, 2, odd_len, tail),
        );
        iota += even * term.eta * right_factor;
        rho += odd * term.eta * left_factor;
        counters.real_fold_field_ops += even_operations + odd_operations;
        counters.real_ip_field_ops += 5;
    }
    (iota, rho)
}

fn packed_eo_bucketed_cross_inner_products<F: PrimeField>(
    buckets: &[PackedReducedBucket<F>],
    counters: &mut PaddingWorkCounters,
) -> (F, F) {
    let singleton_start = buckets
        .iter()
        .position(|bucket| bucket.cover_size == 1)
        .unwrap_or(buckets.len());
    let mut iota = F::zero();
    let mut rho = F::zero();
    for bucket in &buckets[..singleton_start] {
        let (bucket_iota, bucket_rho) =
            packed_eo_cross_inner_products_allocation_free(&bucket.values, &bucket.basis, counters);
        iota += bucket_iota;
        rho += bucket_rho;
    }
    for pair in buckets[singleton_start..].chunks(2) {
        if pair.len() == 1 {
            continue;
        }
        let left_value = pair[0].values[0];
        let right_value = pair[1].values[0];
        let left_basis = pair[0].basis.reduced_scalar();
        let right_basis = pair[1].basis.reduced_scalar();
        iota += left_value * right_basis;
        rho += right_value * left_basis;
        counters.real_ip_field_ops += 4;
    }
    (iota, rho)
}

fn packed_eo_fold_prefix_in_place_parallel<F: Field>(
    values: &mut Vec<F>,
    challenge: F,
    counters: &mut PaddingWorkCounters,
) {
    let old_len = values.len();
    if old_len <= 1 {
        return;
    }
    values[0] = values[0] + challenge * values[1];
    let next_len = old_len.div_ceil(2);
    let mut start = 1_usize;
    while start < next_len {
        let end = (start * 2).min(next_len);
        let source_start = start * 2;
        let (prefix, source) = values.split_at_mut(source_start);
        let source: &[F] = source;
        prefix[start..end]
            .par_iter_mut()
            .enumerate()
            .for_each(|(offset, output)| {
                let input = offset * 2;
                let left = source[input];
                *output = if input + 1 < source.len() {
                    left + challenge * source[input + 1]
                } else {
                    left
                };
            });
        start = end;
    }
    counters.real_fold_field_ops += 2 * (old_len / 2) as u64;
    values.truncate(next_len);
}

fn packed_eo_fold_buckets_in_place<F: PrimeField>(
    mut buckets: Vec<PackedReducedBucket<F>>,
    challenge: F,
    challenge_inv: F,
    counters: &mut PaddingWorkCounters,
) -> Vec<PackedReducedBucket<F>> {
    let singleton_start = buckets
        .iter()
        .position(|bucket| bucket.cover_size == 1)
        .unwrap_or(buckets.len());
    let mut singletons = buckets.split_off(singleton_start);
    for bucket in &mut buckets {
        packed_eo_fold_prefix_in_place_parallel(&mut bucket.values, challenge_inv, counters);
        bucket.cover_size /= 2;
        let basis = std::mem::replace(&mut bucket.basis, TbvBucket { terms: Vec::new() });
        bucket.basis = basis.fold_interior_owned(challenge);
    }

    let mut singleton_iter = singletons.drain(..);
    while let Some(mut left) = singleton_iter.next() {
        if let Some(right) = singleton_iter.next() {
            left.values[0] += challenge_inv * right.values[0];
            let folded_basis =
                left.basis.reduced_scalar() + challenge * right.basis.reduced_scalar();
            left.basis = TbvBucket::from_factor_terms(vec![(folded_basis, Vec::new())]);
            counters.real_fold_field_ops += 4;
        }
        buckets.push(left);
    }
    buckets
}

impl<F: PrimeField + Absorb> PackedEoLiteBulletProof<F> {
    pub fn to_bytes(&self) -> Result<Vec<u8>, SerializationError> {
        let mut bytes = Vec::new();
        self.serialize_compressed(&mut bytes)?;
        Ok(bytes)
    }

    pub fn from_bytes_exact(bytes: &[u8]) -> Result<Self, EoProofDecodeError> {
        let mut reader = Cursor::new(bytes);
        let packing_stride = usize::deserialize_compressed(&mut reader)?;
        if packing_stride < 2 || !packing_stride.is_power_of_two() {
            return Err(EoProofDecodeError::InvalidLength);
        }
        let field_element_size = F::zero().compressed_size();
        let binary_round_size = field_element_size
            .checked_mul(2)
            .ok_or(EoProofDecodeError::InvalidLength)?;
        let packed_len = packing_stride - 1;
        let packed_positive =
            deserialize_bounded_eo_vec(&mut reader, packed_len, packed_len, field_element_size)?;
        let packed_negative =
            deserialize_bounded_eo_vec(&mut reader, packed_len, packed_len, field_element_size)?;
        let binary_rounds =
            deserialize_bounded_eo_vec(&mut reader, 0, usize::BITS as usize, binary_round_size)?;
        let a_reduced = F::deserialize_compressed(&mut reader)?;
        let b_reduced = F::deserialize_compressed(&mut reader)?;
        if reader.position() != bytes.len() as u64 {
            return Err(EoProofDecodeError::TrailingBytes);
        }
        Ok(Self {
            packing_stride,
            packed_positive,
            packed_negative,
            binary_rounds,
            a_reduced,
            b_reduced,
        })
    }

    pub fn field_element_count(&self) -> usize {
        self.packed_positive.len() + self.packed_negative.len() + 2 * self.binary_rounds.len() + 2
    }

    pub fn prove_i8_streaming<S: BvaI8Source>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        let stride = packing_stride_for_bit_width::<F>(source.layout().logical_len, bit_width)?;
        Self::prove_i8_streaming_inner(
            source, tbv, claimed, bit_width, stride, None, None, None, None,
        )
    }

    pub fn prove_i8_streaming_bound<S: BvaI8Source>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
        binding: F,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        let stride = packing_stride_for_bit_width::<F>(source.layout().logical_len, bit_width)?;
        Self::prove_i8_streaming_inner(
            source,
            tbv,
            claimed,
            bit_width,
            stride,
            Some(binding),
            None,
            None,
            None,
        )
    }

    /// Same as [`Self::prove_i8_streaming_bound`], but takes the cross-term lane
    /// projections from the caller.
    ///
    /// The projections must match [`TensorBasisVector::packed_lane_requests`]
    /// for the same stride. Any bucket left as `None` falls back to the CPU
    /// pass, so a caller may accelerate only the buckets it can.
    pub fn prove_i8_streaming_bound_with_lane_projections<S: BvaI8Source>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
        binding: F,
        lane_projections: &PackedLaneProjections<F>,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        let stride = packing_stride_for_bit_width::<F>(source.layout().logical_len, bit_width)?;
        Self::prove_i8_streaming_inner(
            source,
            tbv,
            claimed,
            bit_width,
            stride,
            Some(binding),
            Some(lane_projections),
            None,
            None,
        )
    }

    /// Same as [`Self::prove_i8_streaming_bound_with_lane_projections`], but
    /// also lets an accelerator own the reduced vectors.
    ///
    /// Together these cover both witness-sized passes of the packed EO
    /// reduction: the caller supplies the cross-term lane projections, and the
    /// accelerator builds and folds the reduced vectors without ever returning
    /// a witness-sized buffer.
    pub fn prove_i8_streaming_bound_accelerated<S: BvaI8Source>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
        binding: F,
        lane_projections: &PackedLaneProjections<F>,
        accelerator: &mut dyn PackedEoReducedVectors<F, Error = String>,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        let stride = packing_stride_for_bit_width::<F>(source.layout().logical_len, bit_width)?;
        Self::prove_i8_streaming_inner(
            source,
            tbv,
            claimed,
            bit_width,
            stride,
            Some(binding),
            Some(lane_projections),
            Some(accelerator),
            None,
        )
    }

    pub fn prove_i8_streaming_with_stride<S: BvaI8Source>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
        stride: usize,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        Self::prove_i8_streaming_inner(
            source, tbv, claimed, bit_width, stride, None, None, None, None,
        )
    }

    /// Run Packed EO with challenges derived from hiding scalar commitments.
    ///
    /// The caller owns the commitment transcript and later turns the retained
    /// scalar witnesses into a zero-knowledge scalar-identity proof.
    pub fn prove_i8_streaming_with_commitment_transcript<S: BvaI8Source>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
        transcript: &mut dyn PackedEoChallengeTranscript<F>,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        let stride = packing_stride_for_bit_width::<F>(source.layout().logical_len, bit_width)?;
        Self::prove_i8_streaming_inner(
            source,
            tbv,
            claimed,
            bit_width,
            stride,
            None,
            None,
            None,
            Some(transcript),
        )
    }

    fn prove_i8_streaming_inner<S: BvaI8Source>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
        stride: usize,
        binding: Option<F>,
        lane_projections: Option<&PackedLaneProjections<F>>,
        mut accelerator: Option<&mut dyn PackedEoReducedVectors<F, Error = String>>,
        mut commitment_transcript: Option<&mut dyn PackedEoChallengeTranscript<F>>,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        if bit_width == 0 || bit_width > 8 {
            return Err(BvaError::InvalidBitWidth(bit_width).into());
        }
        let layout = source.layout().clone();
        if tbv.layout != layout {
            return Err(BvaError::LayoutMismatch.into());
        }
        validate_packing_stride(layout.logical_len, stride)?;
        let offsets = layout.bucket_offsets();
        let mut packed_positive = vec![F::zero(); stride - 1];
        let mut packed_negative = vec![F::zero(); stride - 1];
        let mut trace = F::zero();
        let mut visited = vec![false; layout.vector_lengths.len()];
        let mut counters = PaddingWorkCounters::default();
        let mut profile = StreamingEoProfile {
            source_passes: 1,
            ..Default::default()
        };
        let mut bva_error = None;
        let cross_terms_started = Instant::now();
        source
            .for_each_vector(&mut |vector_index, values| {
                if bva_error.is_some() {
                    return;
                }
                let Some(&bucket_index) = layout.vector_to_bucket.get(vector_index) else {
                    bva_error = Some(BvaError::VectorIndex);
                    return;
                };
                if visited[vector_index] {
                    bva_error = Some(BvaError::DuplicateVector);
                    return;
                }
                visited[vector_index] = true;
                if values.len() != layout.vector_lengths[vector_index] {
                    bva_error = Some(BvaError::VectorLength);
                    return;
                }
                profile.physical_entries_per_pass += values.len();
                profile.total_visited_entries += values.len();
                if bit_width < 8
                    && values
                        .par_iter()
                        .any(|value| value.unsigned_abs() >= (1_u8 << bit_width))
                {
                    let value = values
                        .iter()
                        .copied()
                        .find(|value| value.unsigned_abs() >= (1_u8 << bit_width))
                        .expect("parallel range check found a value");
                    bva_error = Some(BvaError::ValueOutOfRange {
                        value: i32::from(value),
                        bit_width,
                    });
                    return;
                }
                let logical_offset = offsets[bucket_index];
                let supplied = lane_projections
                    .and_then(|lanes| lanes.get(bucket_index))
                    .and_then(|lanes| lanes.as_deref());
                let (positive, negative, local_trace, operations) = packed_i8_cross_terms(
                    values,
                    bucket_index,
                    logical_offset,
                    &offsets,
                    tbv,
                    stride,
                    supplied,
                );
                for (total, contribution) in packed_positive.iter_mut().zip(positive) {
                    *total += contribution;
                }
                for (total, contribution) in packed_negative.iter_mut().zip(negative) {
                    *total += contribution;
                }
                trace += local_trace;
                counters.real_ip_field_ops += operations;
            })
            .map_err(StreamingEoError::Source)?;
        profile.timings.cross_terms_seconds = cross_terms_started.elapsed().as_secs_f64();
        if let Some(error) = bva_error {
            return Err(error.into());
        }
        if visited.iter().any(|was_visited| !was_visited) {
            return Err(BvaError::MissingVector.into());
        }
        if trace != claimed {
            return Err(BvaError::ClaimMismatch.into());
        }

        let mut fs = commitment_transcript.is_none().then(|| {
            packed_eo_litebullet_fiat_shamir(layout.logical_len, stride, claimed, binding)
        });
        let packed_challenge = if let Some(transcript) = commitment_transcript.as_deref_mut() {
            transcript
                .packed_challenge(&packed_positive, &packed_negative)
                .map_err(StreamingEoError::ChallengeTranscript)?
        } else {
            let fs = fs.as_mut().expect("public transcript is initialized");
            for value in &packed_positive {
                fs.push(value);
            }
            for value in &packed_negative {
                fs.push(value);
            }
            fs.gen_challenge()
        };
        let packed_challenge_inv = packed_challenge.inverse().ok_or(BvaError::ZeroChallenge)?;
        let mut inverse_powers = Vec::with_capacity(stride);
        let mut inverse_power = F::one();
        for _ in 0..stride {
            inverse_powers.push(inverse_power);
            inverse_power *= packed_challenge_inv;
        }

        let unallocated_from = if accelerator.is_some() {
            ACCELERATED_MINIMUM_ENTRIES
        } else {
            usize::MAX
        };
        let (mut reduced_buckets, destinations) =
            build_packed_reduced_buckets(tbv, packed_challenge, stride, unallocated_from);
        profile.maximum_live_vector_entries =
            reduced_buckets.iter().map(|bucket| bucket.entries).sum();

        // Long reduced vectors move to the accelerator, which builds them from
        // its own witness copy and keeps them until folding has made them
        // small. Their CPU-side entry keeps the cover size and the symbolic
        // basis, but drops its values so nothing witness-sized is allocated
        // here.
        let mut accelerated_slots = 0_usize;
        if let Some(accelerator) = accelerator.as_deref_mut() {
            let specs = reduced_buckets
                .iter()
                .enumerate()
                .filter(|(_, bucket)| {
                    bucket.entries >= ACCELERATED_MINIMUM_ENTRIES && bucket.basis.terms.len() == 1
                })
                .map(|(bucket, entry)| PackedReducedSpec {
                    bucket,
                    logical_offset: entry.logical_offset,
                    len: entry.entries,
                })
                .collect::<Vec<_>>();
            if !specs.is_empty() {
                accelerator
                    .build(&specs, &inverse_powers)
                    .map_err(|error| StreamingEoError::Accelerator(error.to_string()))?;
                for (slot, spec) in specs.iter().enumerate() {
                    reduced_buckets[spec.bucket].accelerator_slot = Some(slot);
                }
            }
            accelerated_slots = specs.len();
        }

        visited.fill(false);
        bva_error = None;
        let fold_started = Instant::now();
        source
            .for_each_vector(&mut |vector_index, values| {
                if bva_error.is_some() {
                    return;
                }
                let Some(&bucket_index) = layout.vector_to_bucket.get(vector_index) else {
                    bva_error = Some(BvaError::VectorIndex);
                    return;
                };
                if visited[vector_index] {
                    bva_error = Some(BvaError::DuplicateVector);
                    return;
                }
                visited[vector_index] = true;
                if values.len() != layout.vector_lengths[vector_index] {
                    bva_error = Some(BvaError::VectorLength);
                    return;
                }
                profile.total_visited_entries += values.len();
                let destination = destinations[vector_index];
                if layout.bucket_sizes[bucket_index] >= stride {
                    counters.real_fold_field_ops += reduced_buckets[destination.bucket_index]
                        .values
                        .par_iter_mut()
                        .zip(values.par_chunks(stride))
                        .map(|(output, row)| {
                            let mut sum = F::zero();
                            let mut operations = 0_u64;
                            for (&value, &inverse_power) in row.iter().zip(&inverse_powers) {
                                if value == 0 {
                                    continue;
                                }
                                sum += i8_to_field::<F>(value) * inverse_power;
                                operations += 2;
                            }
                            *output = sum;
                            operations
                        })
                        .sum::<u64>();
                } else {
                    let (contribution, operations) = values
                        .par_iter()
                        .enumerate()
                        .map(|(local_index, &value)| {
                            if value == 0 {
                                (F::zero(), 0_u64)
                            } else {
                                (
                                    i8_to_field::<F>(value)
                                        * inverse_powers[destination.lane_offset + local_index],
                                    2,
                                )
                            }
                        })
                        .reduce(
                            || (F::zero(), 0_u64),
                            |left, right| (left.0 + right.0, left.1 + right.1),
                        );
                    reduced_buckets[destination.bucket_index].values[0] += contribution;
                    counters.real_fold_field_ops += operations;
                }
            })
            .map_err(StreamingEoError::Source)?;
        profile.timings.fold_seconds = fold_started.elapsed().as_secs_f64();
        profile.source_passes += 1;
        if let Some(error) = bva_error {
            return Err(error.into());
        }
        if visited.iter().any(|was_visited| !was_visited) {
            return Err(BvaError::MissingVector.into());
        }

        let reduced_check_started = Instant::now();
        let mut reduced_claim = claimed;
        let mut positive_power = packed_challenge;
        let mut negative_power = packed_challenge_inv;
        for (&positive, &negative) in packed_positive.iter().zip(&packed_negative) {
            reduced_claim += positive * positive_power + negative * negative_power;
            positive_power *= packed_challenge;
            negative_power *= packed_challenge_inv;
        }
        let (mut reduced_actual, reduced_operations) =
            packed_reduced_inner_product(&reduced_buckets);
        counters.real_ip_field_ops += reduced_operations;
        if let Some(accelerator) = accelerator.as_deref_mut() {
            if accelerated_slots != 0 {
                let mut factors = vec![Vec::new(); accelerated_slots];
                for bucket in &reduced_buckets {
                    if let Some(slot) = bucket.accelerator_slot {
                        factors[slot] = bucket.basis.terms[0].factors.clone();
                    }
                }
                let sums = accelerator
                    .weighted_sums(&factors)
                    .map_err(|error| StreamingEoError::Accelerator(error.to_string()))?;
                for bucket in &reduced_buckets {
                    if let Some(slot) = bucket.accelerator_slot {
                        reduced_actual += bucket.basis.terms[0].eta * sums[slot];
                    }
                }
            }
        }
        if reduced_actual != reduced_claim {
            return Err(BvaError::ClaimMismatch.into());
        }
        profile.timings.reduced_check_seconds = reduced_check_started.elapsed().as_secs_f64();

        let binary_rounds_started = Instant::now();
        let binary_round_count = (layout.logical_len / stride).ilog2() as usize;
        let mut binary_rounds = Vec::with_capacity(binary_round_count);
        for round in 0..binary_round_count {
            let (mut iota, mut rho) =
                packed_eo_bucketed_cross_inner_products(&reduced_buckets, &mut counters);
            if let Some(accelerator) = accelerator.as_deref_mut() {
                if accelerated_slots != 0 {
                    // The first factor pair belongs to the round itself; the
                    // tail is the tensor the even and odd halves project on.
                    let mut tails = vec![Vec::new(); accelerated_slots];
                    for bucket in &reduced_buckets {
                        if let Some(slot) = bucket.accelerator_slot {
                            tails[slot] = bucket.basis.terms[0].factors[1..].to_vec();
                        }
                    }
                    let sums = accelerator
                        .even_odd_sums(&tails)
                        .map_err(|error| StreamingEoError::Accelerator(error.to_string()))?;
                    for bucket in &reduced_buckets {
                        let Some(slot) = bucket.accelerator_slot else {
                            continue;
                        };
                        let term = &bucket.basis.terms[0];
                        let (left_factor, right_factor) = term.factors[0];
                        let (even, odd) = sums[slot];
                        iota += even * term.eta * right_factor;
                        rho += odd * term.eta * left_factor;
                    }
                }
            }
            binary_rounds.push((iota, rho));
            let challenge = if let Some(transcript) = commitment_transcript.as_deref_mut() {
                transcript
                    .binary_challenge(round, iota, rho)
                    .map_err(StreamingEoError::ChallengeTranscript)?
            } else {
                let fs = fs.as_mut().expect("public transcript is initialized");
                fs.push(&iota);
                fs.push(&rho);
                fs.gen_challenge()
            };
            let challenge_inv = challenge.inverse().ok_or(BvaError::ZeroChallenge)?;
            reduced_buckets = packed_eo_fold_buckets_in_place(
                reduced_buckets,
                challenge,
                challenge_inv,
                &mut counters,
            );
            if let Some(accelerator) = accelerator.as_deref_mut() {
                if accelerated_slots != 0 {
                    accelerator
                        .fold(challenge_inv)
                        .map_err(|error| StreamingEoError::Accelerator(error.to_string()))?;
                    // Once a vector is short enough the CPU finishes it, so the
                    // remaining rounds need no device round trips at all.
                    let mut retired = accelerator
                        .retire_below(ACCELERATED_RETIRE_ENTRIES)
                        .map_err(|error| StreamingEoError::Accelerator(error.to_string()))?;
                    for bucket in &mut reduced_buckets {
                        let Some(slot) = bucket.accelerator_slot else {
                            continue;
                        };
                        if let Some(values) = retired[slot].take() {
                            bucket.values = values;
                            bucket.accelerator_slot = None;
                        }
                    }
                }
            }
        }
        profile.timings.binary_rounds_seconds = binary_rounds_started.elapsed().as_secs_f64();
        if reduced_buckets.len() != 1
            || reduced_buckets[0].cover_size != 1
            || reduced_buckets[0].values.len() != 1
        {
            return Err(BvaError::MissingFinalValue.into());
        }
        let mut final_bucket = reduced_buckets.pop().ok_or(BvaError::MissingFinalValue)?;
        let a_reduced = final_bucket
            .values
            .pop()
            .ok_or(BvaError::MissingFinalValue)?;
        let b_reduced = final_bucket.basis.reduced_scalar();
        if let Some(transcript) = commitment_transcript.as_deref_mut() {
            transcript
                .bind_reduced(a_reduced, b_reduced)
                .map_err(StreamingEoError::ChallengeTranscript)?;
        }
        Ok((
            Self {
                packing_stride: stride,
                packed_positive,
                packed_negative,
                binary_rounds,
                a_reduced,
                b_reduced,
            },
            counters,
            profile,
        ))
    }

    fn verify_scalar_identity_inner(
        &self,
        logical_len: usize,
        claimed: F,
        binding: Option<F>,
    ) -> EoLiteBulletVerification<F> {
        let rejected = || EoLiteBulletVerification {
            accepted: false,
            challenges: Vec::new(),
            challenges_inv: Vec::new(),
        };
        if validate_packing_stride(logical_len, self.packing_stride).is_err()
            || self.packed_positive.len() != self.packing_stride - 1
            || self.packed_negative.len() != self.packing_stride - 1
            || self.binary_rounds.len() != (logical_len / self.packing_stride).ilog2() as usize
        {
            return rejected();
        }
        let mut fs =
            packed_eo_litebullet_fiat_shamir(logical_len, self.packing_stride, claimed, binding);
        for value in &self.packed_positive {
            fs.push(value);
        }
        for value in &self.packed_negative {
            fs.push(value);
        }
        let packed_challenge = fs.gen_challenge();
        let Some(packed_challenge_inv) = packed_challenge.inverse() else {
            return rejected();
        };
        let stride_log = self.packing_stride.ilog2() as usize;
        let mut challenges = Vec::with_capacity(logical_len.ilog2() as usize);
        let mut challenges_inv = Vec::with_capacity(logical_len.ilog2() as usize);
        let mut power = packed_challenge;
        let mut inverse_power = packed_challenge_inv;
        for _ in 0..stride_log {
            challenges.push(power);
            challenges_inv.push(inverse_power);
            power.square_in_place();
            inverse_power.square_in_place();
        }
        let mut reduced_claim = claimed;
        let mut positive_power = packed_challenge;
        let mut negative_power = packed_challenge_inv;
        for (&positive, &negative) in self.packed_positive.iter().zip(&self.packed_negative) {
            reduced_claim += positive * positive_power + negative * negative_power;
            positive_power *= packed_challenge;
            negative_power *= packed_challenge_inv;
        }
        for &(iota, rho) in &self.binary_rounds {
            fs.push(&iota);
            fs.push(&rho);
            let challenge = fs.gen_challenge();
            let Some(challenge_inv) = challenge.inverse() else {
                return rejected();
            };
            reduced_claim += challenge * iota + challenge_inv * rho;
            challenges.push(challenge);
            challenges_inv.push(challenge_inv);
        }
        EoLiteBulletVerification {
            accepted: self.a_reduced * self.b_reduced == reduced_claim,
            challenges,
            challenges_inv,
        }
    }

    pub fn verify_with_tbv(
        &self,
        tbv: &TensorBasisVector<F>,
        claimed: F,
    ) -> EoLiteBulletVerification<F> {
        self.verify_with_tbv_inner(tbv, claimed, None)
    }

    pub fn verify_with_tbv_bound(
        &self,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        binding: F,
    ) -> EoLiteBulletVerification<F> {
        self.verify_with_tbv_inner(tbv, claimed, Some(binding))
    }

    /// Verify the Packed EO scalar reduction before checking the terminal TBV
    /// fold. This split lets callers verify the BVA opening and the TBV inner
    /// product as two explicit end-to-end stages.
    pub fn verify_scalar_identity_bound(
        &self,
        logical_len: usize,
        claimed: F,
        binding: F,
    ) -> EoLiteBulletVerification<F> {
        self.verify_scalar_identity_inner(logical_len, claimed, Some(binding))
    }

    fn verify_with_tbv_inner(
        &self,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        binding: Option<F>,
    ) -> EoLiteBulletVerification<F> {
        let mut verification =
            self.verify_scalar_identity_inner(tbv.layout.logical_len, claimed, binding);
        verification.accepted = verification.accepted
            && tbv
                .fold_to_scalar(&verification.challenges)
                .map(|expected| expected == self.b_reduced)
                .unwrap_or(false);
        verification
    }
}

fn deserialize_bounded_eo_vec<T: CanonicalDeserialize>(
    reader: &mut Cursor<&[u8]>,
    minimum_len: usize,
    maximum_len: usize,
    encoded_element_size: usize,
) -> Result<Vec<T>, EoProofDecodeError> {
    let encoded_len = u64::deserialize_compressed(&mut *reader)?;
    let len = usize::try_from(encoded_len).map_err(|_| EoProofDecodeError::InvalidLength)?;
    if len < minimum_len || len > maximum_len {
        return Err(EoProofDecodeError::InvalidLength);
    }
    let position =
        usize::try_from(reader.position()).map_err(|_| EoProofDecodeError::InvalidLength)?;
    let remaining_bytes = reader
        .get_ref()
        .len()
        .checked_sub(position)
        .ok_or(EoProofDecodeError::InvalidLength)?;
    let required_bytes = len
        .checked_mul(encoded_element_size)
        .ok_or(EoProofDecodeError::InvalidLength)?;
    if required_bytes > remaining_bytes {
        return Err(EoProofDecodeError::InvalidLength);
    }
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| EoProofDecodeError::AllocationFailed)?;
    for _ in 0..len {
        values.push(T::deserialize_compressed(&mut *reader)?);
    }
    Ok(values)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BvaError {
    EmptyAssembly,
    ZeroLengthVector,
    LengthOverflow,
    BucketCount {
        expected: usize,
        actual: usize,
    },
    ChallengeCount {
        bucket: usize,
        expected: usize,
        actual: usize,
    },
    FoldChallengeCount {
        expected: usize,
        actual: usize,
    },
    LayoutMismatch,
    ClaimMismatch,
    VectorIndex,
    DuplicateVector,
    MissingVector,
    VectorLength,
    InvalidBitWidth(usize),
    InvalidPackingStride {
        stride: usize,
        logical_len: usize,
    },
    ValueOutOfRange {
        value: i32,
        bit_width: usize,
    },
    ZeroChallenge,
    MissingFinalValue,
}

impl Display for BvaError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyAssembly => write!(formatter, "a BVA needs at least one vector"),
            Self::ZeroLengthVector => write!(formatter, "BVA vectors must be non-empty"),
            Self::LengthOverflow => write!(formatter, "BVA logical length overflowed usize"),
            Self::BucketCount { expected, actual } => write!(
                formatter,
                "TBV bucket count mismatch: expected {expected}, got {actual}"
            ),
            Self::ChallengeCount {
                bucket,
                expected,
                actual,
            } => write!(
                formatter,
                "TBV bucket {bucket} challenge count mismatch: expected {expected}, got {actual}"
            ),
            Self::FoldChallengeCount { expected, actual } => write!(
                formatter,
                "EO-fold challenge count mismatch: expected {expected}, got {actual}"
            ),
            Self::LayoutMismatch => write!(formatter, "BVA and TBV layouts do not match"),
            Self::ClaimMismatch => {
                write!(formatter, "claimed inner product does not match BVA/TBV")
            }
            Self::VectorIndex => write!(formatter, "BVA source emitted an invalid vector index"),
            Self::DuplicateVector => write!(formatter, "BVA source emitted a vector twice"),
            Self::MissingVector => write!(formatter, "BVA source omitted a physical vector"),
            Self::VectorLength => write!(formatter, "BVA source emitted an invalid vector length"),
            Self::InvalidBitWidth(bit_width) => {
                write!(formatter, "invalid packed BVA bit width {bit_width}")
            }
            Self::InvalidPackingStride {
                stride,
                logical_len,
            } => write!(
                formatter,
                "invalid packed BVA stride {stride} for logical length {logical_len}"
            ),
            Self::ValueOutOfRange { value, bit_width } => write!(
                formatter,
                "value {value} does not fit the configured {bit_width}-bit magnitude"
            ),
            Self::ZeroChallenge => write!(formatter, "Fiat-Shamir produced a zero challenge"),
            Self::MissingFinalValue => write!(formatter, "EO-folding produced no final value"),
        }
    }
}

impl std::error::Error for BvaError {}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use crate::fiat_shamir::FiatShamir;
    use ark_bls12_381::Fr;
    use ark_ff::{Field, One, PrimeField, Zero};
    use rayon::ThreadPoolBuilder;

    use super::{
        i8_to_field, packed_eo_fold_prefix_in_place_parallel, packed_eo_litebullet_fiat_shamir,
        packing_stride_for_bit_width, BvaI8Source, BvaLayout, EoProofDecodeError,
        PackedEoLiteBulletProof, PaddingWorkCounters, TensorBasisVector,
    };

    #[derive(Clone)]
    struct ReplayI8Source {
        layout: BvaLayout,
        vectors: Vec<Vec<i8>>,
    }

    impl BvaI8Source for ReplayI8Source {
        type Error = Infallible;

        fn layout(&self) -> &BvaLayout {
            &self.layout
        }

        fn for_each_vector(
            &mut self,
            visit: &mut dyn FnMut(usize, &[i8]),
        ) -> Result<(), Self::Error> {
            for (index, vector) in self.vectors.iter().enumerate() {
                visit(index, vector);
            }
            Ok(())
        }
    }

    fn claimed_inner_product(vectors: &[Vec<i8>], tbv: &TensorBasisVector<Fr>) -> Fr {
        vectors
            .iter()
            .enumerate()
            .fold(Fr::zero(), |sum, (vector_index, vector)| {
                sum + vector.iter().enumerate().fold(
                    Fr::zero(),
                    |vector_sum, (entry_index, value)| {
                        vector_sum
                            + i8_to_field::<Fr>(*value)
                                * tbv
                                    .physical_coefficient_at(vector_index, entry_index)
                                    .unwrap()
                    },
                )
            })
    }

    fn dense_fold_witness(layout: &BvaLayout, vectors: &[Vec<i8>], challenges: &[Fr]) -> Fr {
        let mut dense = vec![Fr::zero(); layout.logical_len()];
        for (vector_index, vector) in vectors.iter().enumerate() {
            let offset = layout.vector_offset(vector_index).unwrap();
            for (entry_index, value) in vector.iter().enumerate() {
                dense[offset + entry_index] = i8_to_field::<Fr>(*value);
            }
        }
        for challenge in challenges {
            let next_len = dense.len().div_ceil(2);
            for output in 0..next_len {
                let left = dense[2 * output];
                let right = dense.get(2 * output + 1).copied().unwrap_or_default();
                dense[output] = left + *challenge * right;
            }
            dense.truncate(next_len);
        }
        dense[0]
    }

    fn dense_packed_cross_terms(
        layout: &BvaLayout,
        vectors: &[Vec<i8>],
        tbv: &TensorBasisVector<Fr>,
        stride: usize,
    ) -> (Vec<Fr>, Vec<Fr>, Fr) {
        let mut dense = vec![Fr::zero(); layout.logical_len()];
        for (vector_index, vector) in vectors.iter().enumerate() {
            let offset = layout.vector_offset(vector_index).unwrap();
            for (entry_index, value) in vector.iter().enumerate() {
                dense[offset + entry_index] = i8_to_field::<Fr>(*value);
            }
        }
        let offsets = layout.bucket_offsets();
        let mut positive = vec![Fr::zero(); stride - 1];
        let mut negative = vec![Fr::zero(); stride - 1];
        let mut trace = Fr::zero();
        for row_start in (0..layout.logical_len()).step_by(stride) {
            for left_lane in 0..stride {
                let left = dense[row_start + left_lane];
                for right_lane in 0..stride {
                    let right = tbv.logical_coefficient_at(&offsets, row_start + right_lane);
                    let product = left * right;
                    if right_lane == left_lane {
                        trace += product;
                    } else if right_lane > left_lane {
                        positive[right_lane - left_lane - 1] += product;
                    } else {
                        negative[left_lane - right_lane - 1] += product;
                    }
                }
            }
        }
        (positive, negative, trace)
    }

    fn dense_packed_proof(
        layout: &BvaLayout,
        vectors: &[Vec<i8>],
        tbv: &TensorBasisVector<Fr>,
        claimed: Fr,
        stride: usize,
    ) -> PackedEoLiteBulletProof<Fr> {
        let mut witness = vec![Fr::zero(); layout.logical_len()];
        for (vector_index, vector) in vectors.iter().enumerate() {
            let offset = layout.vector_offset(vector_index).unwrap();
            for (entry_index, value) in vector.iter().enumerate() {
                witness[offset + entry_index] = i8_to_field::<Fr>(*value);
            }
        }
        let offsets = layout.bucket_offsets();
        let basis = (0..layout.logical_len())
            .map(|index| tbv.logical_coefficient_at(&offsets, index))
            .collect::<Vec<_>>();
        let (packed_positive, packed_negative, trace) =
            dense_packed_cross_terms(layout, vectors, tbv, stride);
        assert_eq!(trace, claimed);

        let mut transcript =
            packed_eo_litebullet_fiat_shamir(layout.logical_len(), stride, claimed, None);
        for value in &packed_positive {
            transcript.push(value);
        }
        for value in &packed_negative {
            transcript.push(value);
        }
        let packed_challenge = transcript.gen_challenge();
        let packed_challenge_inv = packed_challenge.inverse().unwrap();
        let powers =
            std::iter::successors(Some(Fr::one()), |power| Some(*power * packed_challenge))
                .take(stride)
                .collect::<Vec<_>>();
        let inverse_powers =
            std::iter::successors(Some(Fr::one()), |power| Some(*power * packed_challenge_inv))
                .take(stride)
                .collect::<Vec<_>>();
        let mut reduced_witness = witness
            .chunks(stride)
            .map(|row| {
                row.iter()
                    .zip(&inverse_powers)
                    .map(|(value, power)| *value * power)
                    .sum::<Fr>()
            })
            .collect::<Vec<_>>();
        let mut reduced_basis = basis
            .chunks(stride)
            .map(|row| {
                row.iter()
                    .zip(&powers)
                    .map(|(value, power)| *value * power)
                    .sum::<Fr>()
            })
            .collect::<Vec<_>>();

        let mut binary_rounds = Vec::new();
        while reduced_witness.len() > 1 {
            let mut iota = Fr::zero();
            let mut rho = Fr::zero();
            for row in 0..reduced_witness.len() / 2 {
                iota += reduced_witness[2 * row] * reduced_basis[2 * row + 1];
                rho += reduced_witness[2 * row + 1] * reduced_basis[2 * row];
            }
            binary_rounds.push((iota, rho));
            transcript.push(&iota);
            transcript.push(&rho);
            let challenge = transcript.gen_challenge();
            let challenge_inv = challenge.inverse().unwrap();
            for row in 0..reduced_witness.len() / 2 {
                reduced_witness[row] =
                    reduced_witness[2 * row] + challenge_inv * reduced_witness[2 * row + 1];
                reduced_basis[row] =
                    reduced_basis[2 * row] + challenge * reduced_basis[2 * row + 1];
            }
            let next_len = reduced_witness.len() / 2;
            reduced_witness.truncate(next_len);
            reduced_basis.truncate(next_len);
        }
        PackedEoLiteBulletProof {
            packing_stride: stride,
            packed_positive,
            packed_negative,
            binary_rounds,
            a_reduced: reduced_witness[0],
            b_reduced: reduced_basis[0],
        }
    }

    #[test]
    fn bva_layout_matches_decreasing_paper_layout_without_empty_buckets() {
        let layout = BvaLayout::from_lengths(&[3, 5, 2]).unwrap();

        assert_eq!(layout.logical_len(), 16);
        assert_eq!(layout.active_len(), 14);
        assert_eq!(layout.bucket_sizes(), &[8, 4, 2]);
        assert_eq!(layout.vector_to_bucket(), &[1, 0, 2]);
        assert_eq!(layout.padding_bucket_count(), 0);
        assert_eq!(layout.materialized_padding_entries(), 0);
    }

    #[test]
    fn eight_bit_packing_uses_32_entries_for_a_256_bit_field() {
        assert_eq!(std::mem::size_of::<i8>(), 1);
        assert_eq!(Fr::MODULUS_BIT_SIZE, 255);
        assert_eq!(packing_stride_for_bit_width::<Fr>(64, 8).unwrap(), 32);
        assert_eq!(packing_stride_for_bit_width::<Fr>(16, 8).unwrap(), 16);
    }

    #[test]
    fn compact_tbv_transcript_binding_is_deterministic_and_term_sensitive() {
        let layout = BvaLayout::from_lengths(&[5, 3, 2]).unwrap();
        let challenges: [Fr; 4] = [2.into(), 3.into(), 5.into(), 7.into()];
        let tbv = TensorBasisVector::global_tensor(&layout, &challenges).unwrap();
        let mut left = FiatShamir::new();
        let mut right = FiatShamir::new();
        tbv.absorb_transcript(&mut left);
        tbv.absorb_transcript(&mut right);
        assert_eq!(left.state(), right.state());

        let changed =
            TensorBasisVector::global_tensor(&layout, &[2.into(), 3.into(), 5.into(), 11.into()])
                .unwrap();
        let mut different = FiatShamir::new();
        changed.absorb_transcript(&mut different);
        assert_ne!(left.state(), different.state());
    }

    #[test]
    fn packed_eo_reads_native_i8_and_crosses_bucket_boundaries() {
        let vectors = vec![vec![-3_i8, 2, 7, -1, 4, 6, -2], vec![5], vec![-4]];
        let layout =
            BvaLayout::from_lengths(&vectors.iter().map(Vec::len).collect::<Vec<_>>()).unwrap();
        assert_eq!(layout.bucket_sizes(), &[8, 1, 1]);
        assert_eq!(layout.active_len(), 10);
        assert_eq!(layout.logical_len(), 16);
        let tbv =
            TensorBasisVector::global_tensor(&layout, &[2.into(), 3.into(), 5.into(), 7.into()])
                .unwrap();
        let claimed = claimed_inner_product(&vectors, &tbv);
        let mut source = ReplayI8Source {
            layout: layout.clone(),
            vectors: vectors.clone(),
        };

        let (proof, counters, profile) = PackedEoLiteBulletProof::prove_i8_streaming_with_stride(
            &mut source,
            &tbv,
            claimed,
            8,
            4,
        )
        .unwrap();
        let verification = proof.verify_with_tbv(&tbv, claimed);

        assert!(verification.accepted);
        assert_eq!(proof.packing_stride, 4);
        assert_eq!(proof.packed_positive.len(), 3);
        assert_eq!(proof.packed_negative.len(), 3);
        assert_eq!(proof.binary_rounds.len(), 2);
        assert_eq!(proof.field_element_count(), 12);
        assert_eq!(verification.challenges.len(), 4);
        assert_eq!(profile.source_passes, 2);
        assert_eq!(profile.physical_entries_per_pass, 9);
        assert_eq!(profile.total_visited_entries, 18);
        assert_eq!(profile.maximum_live_vector_entries, 3);
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(counters.materialized_padding_entries, 0);
        assert_eq!(counters.padding_ip_field_ops, 0);
        assert_eq!(counters.padding_fold_field_ops, 0);
        assert_eq!(
            proof.a_reduced,
            dense_fold_witness(&layout, &vectors, &verification.challenges_inv)
        );

        let encoded = proof.to_bytes().unwrap();
        let decoded = PackedEoLiteBulletProof::<Fr>::from_bytes_exact(&encoded).unwrap();
        assert_eq!(decoded, proof);
        let mut trailing = encoded.clone();
        trailing.push(0);
        assert!(PackedEoLiteBulletProof::<Fr>::from_bytes_exact(&trailing).is_err());
        assert!(
            PackedEoLiteBulletProof::<Fr>::from_bytes_exact(&encoded[..encoded.len() - 1]).is_err()
        );

        let mut malicious_length = 4_u64.to_le_bytes().to_vec();
        malicious_length.extend_from_slice(&u64::MAX.to_le_bytes());
        assert!(matches!(
            PackedEoLiteBulletProof::<Fr>::from_bytes_exact(&malicious_length),
            Err(EoProofDecodeError::InvalidLength)
        ));

        let wide_stride_proof = PackedEoLiteBulletProof {
            packing_stride: 512,
            packed_positive: vec![Fr::zero(); 511],
            packed_negative: vec![Fr::zero(); 511],
            binary_rounds: Vec::new(),
            a_reduced: Fr::zero(),
            b_reduced: Fr::zero(),
        };
        let wide_stride_encoded = wide_stride_proof.to_bytes().unwrap();
        assert_eq!(
            PackedEoLiteBulletProof::<Fr>::from_bytes_exact(&wide_stride_encoded).unwrap(),
            wide_stride_proof
        );

        let mut tampered = proof;
        tampered.packed_positive[0] += Fr::from(1_u64);
        assert!(!tampered.verify_with_tbv(&tbv, claimed).accepted);
    }

    #[test]
    fn plus_one_lengths_keep_internal_cover_zeros_implicit_after_packing() {
        let vectors = (0..=8)
            .rev()
            .map(|log| {
                let length = (1_usize << log) + 1;
                (0..length)
                    .map(|index| ((index + log) % 23) as i8 - 11)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let layout =
            BvaLayout::from_lengths(&vectors.iter().map(Vec::len).collect::<Vec<_>>()).unwrap();
        assert_eq!(layout.logical_len(), 1 << 10);
        assert_eq!(layout.active_len(), (1 << 10) - 2);
        let challenges = (0..layout.logical_len().ilog2())
            .map(|round| Fr::from(u64::from(round) + 2))
            .collect::<Vec<_>>();
        let tbv = TensorBasisVector::global_tensor(&layout, &challenges).unwrap();
        let claimed = claimed_inner_product(&vectors, &tbv);
        let expected = dense_packed_cross_terms(&layout, &vectors, &tbv, 32);
        let mut source = ReplayI8Source {
            layout: layout.clone(),
            vectors: vectors.clone(),
        };

        assert_eq!(
            tbv.inner_product_i8_streaming(&mut source).unwrap(),
            claimed
        );

        let (proof, counters, profile) = PackedEoLiteBulletProof::prove_i8_streaming_with_stride(
            &mut source,
            &tbv,
            claimed,
            8,
            32,
        )
        .unwrap();
        let verification = proof.verify_with_tbv(&tbv, claimed);

        assert_eq!(
            proof,
            dense_packed_proof(&layout, &vectors, &tbv, claimed, 32)
        );
        assert_eq!(proof.packed_positive, expected.0);
        assert_eq!(proof.packed_negative, expected.1);
        assert_eq!(claimed, expected.2);
        assert!(verification.accepted);
        assert_eq!(
            proof.a_reduced,
            dense_fold_witness(&layout, &vectors, &verification.challenges_inv)
        );
        assert_eq!(profile.maximum_live_vector_entries, (1 << 4) + 5);
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(counters.materialized_padding_entries, 0);
        assert_eq!(counters.padding_ip_field_ops, 0);
        assert_eq!(counters.padding_fold_field_ops, 0);
    }

    #[test]
    fn packed_field_folding_reuses_the_same_allocation() {
        let mut values = (1_u64..=17).map(Fr::from).collect::<Vec<_>>();
        let original = values.clone();
        let pointer = values.as_ptr();
        let capacity = values.capacity();
        let challenge = Fr::from(9_u64);
        let mut counters = PaddingWorkCounters::default();

        packed_eo_fold_prefix_in_place_parallel(&mut values, challenge, &mut counters);

        let expected = original
            .chunks(2)
            .map(|pair| pair[0] + challenge * pair.get(1).copied().unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(values, expected);
        assert_eq!(values.as_ptr(), pointer);
        assert_eq!(values.capacity(), capacity);
    }

    #[test]
    fn packed_eo_is_deterministic_across_one_and_four_rayon_threads() {
        let vectors = vec![
            (0..1 << 14)
                .map(|index| (index as i8).wrapping_mul(3))
                .collect::<Vec<_>>(),
            vec![7_i8; 64],
        ];
        let layout =
            BvaLayout::from_lengths(&vectors.iter().map(Vec::len).collect::<Vec<_>>()).unwrap();
        let challenges = (0..layout.logical_len().ilog2())
            .map(|round| Fr::from(u64::from(round) + 2))
            .collect::<Vec<_>>();
        let tbv = TensorBasisVector::global_tensor(&layout, &challenges).unwrap();
        let claimed = claimed_inner_product(&vectors, &tbv);

        let prove = |threads| {
            ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    let mut source = ReplayI8Source {
                        layout: layout.clone(),
                        vectors: vectors.clone(),
                    };
                    PackedEoLiteBulletProof::prove_i8_streaming(&mut source, &tbv, claimed, 8)
                        .unwrap()
                        .0
                })
        };

        let serial = prove(1);
        let parallel = prove(4);
        assert_eq!(serial, parallel);
        assert!(parallel.verify_with_tbv(&tbv, claimed).accepted);
    }
}
