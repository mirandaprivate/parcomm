//! Decreasing-size BVA metadata and packed EO-LiteBullet for signed integers.
//!
//! Physical witness entries remain in `i8` or `i16` storage. The first packed EO step
//! converts only one field element per packing stride, after which all binary
//! folds reuse that reduced field allocation in place.

use std::{
    fmt::{Display, Formatter},
    io::Cursor,
    mem::size_of,
    time::Instant,
};

use ark_crypto_primitives::sponge::Absorb;
use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};
use mat::ShortInt;
use rayon::prelude::*;

use crate::fiat_shamir::FiatShamir;

/// Primitive [`ShortInt`] storage accepted by the packed BVA path.
///
/// The entry type is the only width selector used by Packed EO: choose `i8`
/// for native 8-bit storage or `i16` for native 16-bit storage. Both choices
/// execute the same generic projection, compression, and folding code.
pub trait BvaInteger: ShortInt + 'static {
    const BITS: usize;

    fn is_zero(self) -> bool;

    fn unsigned_abs_u64(self) -> u64;

    fn to_i32(self) -> i32;

    fn to_field<F: PrimeField>(self) -> F;
}

impl BvaInteger for i8 {
    const BITS: usize = 8;

    fn is_zero(self) -> bool {
        self == 0
    }

    fn unsigned_abs_u64(self) -> u64 {
        u64::from(self.unsigned_abs())
    }

    fn to_i32(self) -> i32 {
        i32::from(self)
    }

    fn to_field<F: PrimeField>(self) -> F {
        i8_to_field(self)
    }
}

impl BvaInteger for i16 {
    const BITS: usize = 16;

    fn is_zero(self) -> bool {
        self == 0
    }

    fn unsigned_abs_u64(self) -> u64 {
        u64::from(self.unsigned_abs())
    }

    fn to_i32(self) -> i32 {
        i32::from(self)
    }

    fn to_field<F: PrimeField>(self) -> F {
        if self >= 0 {
            F::from(self as u64)
        } else {
            -F::from((-(self as i32)) as u64)
        }
    }
}

/// A rewindable source backed by one contiguous signed-byte BVA.
pub trait BvaI8Source {
    type Error;

    fn layout(&self) -> &BvaLayout;

    /// Describe how the stored entries sit inside one logical bucket.
    ///
    /// Ordinary vectors occupy a prefix. Matrix boundary buckets override this
    /// with a row-major rectangle whose missing row and column suffixes are
    /// conceptual zeros.
    fn vector_geometry(&self, vector_index: usize) -> Option<BvaI8Geometry> {
        let layout = self.layout();
        let bucket_index = *layout.vector_to_bucket().get(vector_index)?;
        BvaI8Geometry::prefix(
            *layout.bucket_sizes().get(bucket_index)?,
            *layout.vector_lengths().get(vector_index)?,
        )
    }

    /// Number of stored zero-extension bytes in the complete BVA allocation.
    fn materialized_padding_entries(&self) -> usize {
        0
    }

    /// Return all physical vectors in canonical vector-index order when the
    /// source supports zero-copy indexed access.
    ///
    /// Packed EO uses this snapshot to schedule size-ordered buckets in
    /// strided worker lanes. Rewind-only sources may keep the default and use
    /// the serial compatibility path.
    fn indexed_vectors(&self) -> Option<Vec<&[i8]>> {
        None
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[i8])) -> Result<(), Self::Error>;
}

/// A rewindable source backed by one contiguous signed-16-bit BVA.
pub trait BvaI16Source {
    type Error;

    fn layout(&self) -> &BvaLayout;

    fn vector_geometry(&self, vector_index: usize) -> Option<BvaI8Geometry> {
        let layout = self.layout();
        let bucket_index = *layout.vector_to_bucket().get(vector_index)?;
        BvaI8Geometry::prefix(
            *layout.bucket_sizes().get(bucket_index)?,
            *layout.vector_lengths().get(vector_index)?,
        )
    }

    fn materialized_padding_entries(&self) -> usize {
        0
    }

    fn indexed_vectors(&self) -> Option<Vec<&[i16]>> {
        None
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[i16])) -> Result<(), Self::Error>;
}

/// A rewindable BVA source selected by its native [`ShortInt`] entry type.
///
/// Implement this interface when the source itself is generic over `i8` and
/// `i16`. [`BvaI8Source`] and [`BvaI16Source`] remain compatibility adapters
/// for existing sources.
pub trait BvaSource<Entry: BvaInteger> {
    type Error;

    fn layout(&self) -> &BvaLayout;

    fn vector_geometry(&self, vector_index: usize) -> Option<BvaGeometry> {
        let layout = self.layout();
        let bucket_index = *layout.vector_to_bucket().get(vector_index)?;
        BvaGeometry::prefix(
            *layout.bucket_sizes().get(bucket_index)?,
            *layout.vector_lengths().get(vector_index)?,
        )
    }

    fn materialized_padding_entries(&self) -> usize {
        0
    }

    fn indexed_vectors(&self) -> Option<Vec<&[Entry]>> {
        None
    }

    fn for_each_vector(
        &mut self,
        visit: &mut dyn FnMut(usize, &[Entry]),
    ) -> Result<(), Self::Error>;
}

impl<Source: BvaI8Source> BvaSource<i8> for Source {
    type Error = Source::Error;

    fn layout(&self) -> &BvaLayout {
        BvaI8Source::layout(self)
    }

    fn vector_geometry(&self, vector_index: usize) -> Option<BvaI8Geometry> {
        BvaI8Source::vector_geometry(self, vector_index)
    }

    fn materialized_padding_entries(&self) -> usize {
        BvaI8Source::materialized_padding_entries(self)
    }

    fn indexed_vectors(&self) -> Option<Vec<&[i8]>> {
        BvaI8Source::indexed_vectors(self)
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[i8])) -> Result<(), Self::Error> {
        BvaI8Source::for_each_vector(self, visit)
    }
}

impl<Source: BvaI16Source> BvaSource<i16> for Source {
    type Error = Source::Error;

    fn layout(&self) -> &BvaLayout {
        BvaI16Source::layout(self)
    }

    fn vector_geometry(&self, vector_index: usize) -> Option<BvaI8Geometry> {
        BvaI16Source::vector_geometry(self, vector_index)
    }

    fn materialized_padding_entries(&self) -> usize {
        BvaI16Source::materialized_padding_entries(self)
    }

    fn indexed_vectors(&self) -> Option<Vec<&[i16]>> {
        BvaI16Source::indexed_vectors(self)
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[i16])) -> Result<(), Self::Error> {
        BvaI16Source::for_each_vector(self, visit)
    }
}

/// Placement of one stored short-integer slice in a logical power-of-two bucket.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BvaGeometry {
    logical_rows: usize,
    logical_cols: usize,
    physical_rows: usize,
    physical_cols: usize,
    logical_col_log: usize,
}

impl BvaGeometry {
    pub fn prefix(logical_len: usize, physical_len: usize) -> Option<Self> {
        Self::rectangle(1, logical_len, 1, physical_len)
    }

    pub fn rectangle(
        logical_rows: usize,
        logical_cols: usize,
        physical_rows: usize,
        physical_cols: usize,
    ) -> Option<Self> {
        if logical_rows == 0
            || logical_cols == 0
            || !logical_rows.is_power_of_two()
            || !logical_cols.is_power_of_two()
            || physical_rows == 0
            || physical_cols == 0
            || physical_rows > logical_rows
            || physical_cols > logical_cols
        {
            return None;
        }
        Some(Self {
            logical_rows,
            logical_cols,
            physical_rows,
            physical_cols,
            logical_col_log: logical_cols.ilog2() as usize,
        })
    }

    pub fn logical_len(self) -> usize {
        self.logical_rows * self.logical_cols
    }

    pub fn physical_len(self) -> usize {
        self.physical_rows * self.physical_cols
    }

    pub fn logical_rows(self) -> usize {
        self.logical_rows
    }

    pub fn logical_cols(self) -> usize {
        self.logical_cols
    }

    pub fn physical_rows(self) -> usize {
        self.physical_rows
    }

    pub fn physical_cols(self) -> usize {
        self.physical_cols
    }

    pub fn logical_index(self, physical_index: usize) -> Option<usize> {
        if physical_index >= self.physical_len() {
            return None;
        }
        let row = physical_index / self.physical_cols;
        let col = physical_index % self.physical_cols;
        Some(row * self.logical_cols + col)
    }

    pub fn physical_index(self, logical_index: usize) -> Option<usize> {
        if logical_index >= self.logical_len() {
            return None;
        }
        let row = logical_index >> self.logical_col_log;
        let col = logical_index & (self.logical_cols - 1);
        (row < self.physical_rows && col < self.physical_cols)
            .then_some(row * self.physical_cols + col)
    }
}

/// Compatibility name for the geometry type used by the original i8 API.
pub type BvaI8Geometry = BvaGeometry;

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
    /// Computes the witness inner product using the native [`ShortInt`] entry
    /// type selected by `Entry`.
    pub fn inner_product_streaming<Entry, S>(
        &self,
        source: &mut S,
    ) -> Result<F, StreamingEoError<S::Error>>
    where
        Entry: BvaInteger,
        S: BvaSource<Entry>,
    {
        self.inner_product_integer_streaming::<Entry, S>(source)
    }

    /// Computes the physical i8 witness inner product without expanding cover
    /// zeros or evaluating one tensor coefficient at a time.
    pub fn inner_product_i8_streaming<S: BvaI8Source>(
        &self,
        source: &mut S,
    ) -> Result<F, StreamingEoError<S::Error>> {
        self.inner_product_streaming::<i8, S>(source)
    }

    /// Computes the physical i16 witness inner product without widening the
    /// stored BVA to a field vector.
    pub fn inner_product_i16_streaming<S: BvaI16Source>(
        &self,
        source: &mut S,
    ) -> Result<F, StreamingEoError<S::Error>> {
        self.inner_product_streaming::<i16, S>(source)
    }

    fn inner_product_integer_streaming<Entry, S>(
        &self,
        source: &mut S,
    ) -> Result<F, StreamingEoError<S::Error>>
    where
        Entry: BvaInteger,
        S: BvaSource<Entry>,
    {
        let layout = source.layout().clone();
        if self.layout != layout {
            return Err(BvaError::LayoutMismatch.into());
        }
        let geometries = source_geometries::<Entry, S>(source)?;
        if let Some(vectors) = source.indexed_vectors() {
            if vectors.len() != layout.vector_lengths.len() {
                return Err(BvaError::MissingVector.into());
            }
            let mut bucket_to_vector = vec![usize::MAX; layout.bucket_sizes.len()];
            for (vector_index, &bucket_index) in layout.vector_to_bucket.iter().enumerate() {
                if bucket_index >= bucket_to_vector.len()
                    || bucket_to_vector[bucket_index] != usize::MAX
                {
                    return Err(BvaError::VectorIndex.into());
                }
                if vectors[vector_index].len() != geometries[vector_index].physical_len() {
                    return Err(BvaError::VectorLength.into());
                }
                bucket_to_vector[bucket_index] = vector_index;
            }
            if bucket_to_vector.contains(&usize::MAX) {
                return Err(BvaError::MissingVector.into());
            }
            let task_count = eo_strided_task_count(layout.bucket_sizes.len());
            return Ok((0..task_count)
                .into_par_iter()
                .map(|task| {
                    let mut result = F::zero();
                    for bucket_index in (task..layout.bucket_sizes.len()).step_by(task_count) {
                        let vector_index = bucket_to_vector[bucket_index];
                        for term in &self.buckets[bucket_index].terms {
                            let (projected, _) = tensor_project_integer_geometry(
                                vectors[vector_index],
                                geometries[vector_index],
                                &term.factors,
                            );
                            result += term.eta * projected;
                        }
                    }
                    result
                })
                .collect::<Vec<_>>()
                .into_iter()
                .sum());
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
                let geometry = geometries[vector_index];
                if values.len() != geometry.physical_len() {
                    bva_error = Some(BvaError::VectorLength);
                    return;
                }
                for term in &self.buckets[bucket_index].terms {
                    let (projected, _) =
                        tensor_project_integer_geometry(values, geometry, &term.factors);
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
    pub packed_cross_terms_seconds: f64,
    /// Second witness pass, folding `stride` bytes into one field element.
    pub first_fold_seconds: f64,
    /// Consistency check on the reduced vector.
    pub reduced_check_seconds: f64,
    /// Cross terms computed by the later binary LiteBullet rounds.
    pub binary_cross_terms_seconds: f64,
    /// Fiat--Shamir challenge derivation in the later binary rounds.
    pub binary_challenge_seconds: f64,
    /// In-place witness and basis folds in the later binary rounds.
    pub binary_fold_seconds: f64,
}

impl StreamingEoTimings {
    /// Aggregate time for all work inside the later binary rounds.
    pub fn binary_rounds_seconds(&self) -> f64 {
        self.binary_cross_terms_seconds + self.binary_challenge_seconds + self.binary_fold_seconds
    }

    /// Sum of the mutually exclusive timed Packed EO stages.
    pub fn accounted_seconds(&self) -> f64 {
        self.packed_cross_terms_seconds
            + self.first_fold_seconds
            + self.reduced_check_seconds
            + self.binary_rounds_seconds()
    }
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

/// Returns twice the largest power-of-two packing ratio supported by the field.
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
    let base_stride = 1_usize << ratio.ilog2();
    let stride = base_stride
        .checked_mul(2)
        .ok_or(BvaError::InvalidBitWidth(bit_width))?;
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

const SHORT_INTEGER_WINDOW_BITS: usize = 4;
const SHORT_INTEGER_WINDOW_SIZE: usize = 1 << SHORT_INTEGER_WINDOW_BITS;
const MAX_SHORT_INTEGER_WINDOWS: usize = 4;
const SHORT_INTEGER_LAZY_MIN_ROWS: usize = 32;
const SHORT_INTEGER_LAZY_BLOCK_ROWS: usize = 1 << 20;

/// Precomputes signed-short-integer multiples of one field coefficient.
///
/// The packed projection reuses one row coefficient across every lane. Four-bit
/// windows replace one field conversion and one full-width field multiplication
/// per non-zero lane with a small number of field additions.
fn short_integer_window_multiples<F: Field, Entry: BvaInteger>(
    coefficient: F,
) -> (
    [[F; SHORT_INTEGER_WINDOW_SIZE]; MAX_SHORT_INTEGER_WINDOWS],
    u64,
) {
    debug_assert!(Entry::BITS <= SHORT_INTEGER_WINDOW_BITS * MAX_SHORT_INTEGER_WINDOWS);
    let window_count = Entry::BITS.div_ceil(SHORT_INTEGER_WINDOW_BITS);
    let mut multiples = [[F::zero(); SHORT_INTEGER_WINDOW_SIZE]; MAX_SHORT_INTEGER_WINDOWS];
    let mut window_unit = coefficient;
    let mut operations = 0_u64;
    for (window, values) in multiples.iter_mut().enumerate().take(window_count) {
        values[1] = window_unit;
        for digit in 2..SHORT_INTEGER_WINDOW_SIZE {
            values[digit] = values[digit - 1] + window_unit;
            operations += 1;
        }
        if window + 1 < window_count {
            for _ in 0..SHORT_INTEGER_WINDOW_BITS {
                window_unit.double_in_place();
                operations += 1;
            }
        }
    }
    (multiples, operations)
}

#[inline]
fn add_short_integer_product<F: Field, Entry: BvaInteger>(
    accumulator: &mut F,
    value: Entry,
    multiples: &[[F; SHORT_INTEGER_WINDOW_SIZE]; MAX_SHORT_INTEGER_WINDOWS],
) -> u64 {
    let subtract = value.to_i32() < 0;
    let mut magnitude = value.unsigned_abs_u64();
    let mut window = 0_usize;
    let mut operations = 0_u64;
    while magnitude != 0 {
        let digit = (magnitude & (SHORT_INTEGER_WINDOW_SIZE as u64 - 1)) as usize;
        if digit != 0 {
            if subtract {
                *accumulator -= multiples[window][digit];
            } else {
                *accumulator += multiples[window][digit];
            }
            operations += 1;
        }
        magnitude >>= SHORT_INTEGER_WINDOW_BITS;
        window += 1;
    }
    operations
}

fn project_short_integer_rows_window<F: Field, Entry: BvaInteger>(
    values: &[Entry],
    geometry: BvaI8Geometry,
    row_coefficients: &[F],
    stride: usize,
) -> (Vec<F>, u64) {
    let mut computed = vec![F::zero(); stride];
    let mut operations = 0_u64;
    for (row, &coefficient) in row_coefficients.iter().enumerate() {
        let mut multiples = None;
        for (lane, projection) in computed.iter_mut().enumerate() {
            let logical_index = row * stride + lane;
            let Some(physical_index) = geometry.physical_index(logical_index) else {
                continue;
            };
            let value = values[physical_index];
            if value.is_zero() {
                continue;
            }
            if multiples.is_none() {
                let (row_multiples, row_operations) =
                    short_integer_window_multiples::<F, Entry>(coefficient);
                multiples = Some(row_multiples);
                operations += row_operations;
            }
            operations += add_short_integer_product(
                projection,
                value,
                multiples
                    .as_ref()
                    .expect("a non-zero row initializes short multiples"),
            );
        }
    }
    (computed, operations)
}

fn project_short_integer_rows_lazy<F: PrimeField, Entry: BvaInteger>(
    values: &[Entry],
    geometry: BvaI8Geometry,
    row_coefficients: &[F],
    stride: usize,
) -> (Vec<F>, u64) {
    let limb_count = F::zero().into_bigint().as_ref().len();
    let mut computed = vec![F::zero(); stride];
    let mut accumulated = vec![0_i128; stride * limb_count];
    let mut bytes = Vec::with_capacity((limb_count + 1) * 8);
    let mut normalized = Vec::with_capacity(limb_count + 1);
    let mut operations = 0_u64;
    let reduction_radix = short_integer_reduction_radix::<F>();

    for block_start in (0..row_coefficients.len()).step_by(SHORT_INTEGER_LAZY_BLOCK_ROWS) {
        let block_end = block_start
            .saturating_add(SHORT_INTEGER_LAZY_BLOCK_ROWS)
            .min(row_coefficients.len());
        accumulated.fill(0);
        for (local_row, &coefficient) in row_coefficients[block_start..block_end].iter().enumerate()
        {
            let row = block_start + local_row;
            let coefficient = coefficient.into_bigint();
            let coefficient_limbs = coefficient.as_ref();
            operations += 1;
            for lane in 0..stride {
                let logical_index = row * stride + lane;
                let Some(physical_index) = geometry.physical_index(logical_index) else {
                    continue;
                };
                let value = values[physical_index];
                if value.is_zero() {
                    continue;
                }
                let start = lane * limb_count;
                accumulate_signed_short_integer_limbs(
                    &mut accumulated[start..start + limb_count],
                    coefficient_limbs,
                    value.to_i32(),
                );
            }
        }
        for lane in 0..stride {
            let start = lane * limb_count;
            let end = start + limb_count;
            computed[lane] += signed_accumulated_limbs_to_field::<F>(
                &accumulated[start..end],
                &mut normalized,
                &mut bytes,
                reduction_radix,
            );
            operations += 2;
        }
    }
    (computed, operations)
}

#[inline]
fn accumulate_signed_short_integer_limbs(
    accumulator: &mut [i128],
    coefficient_limbs: &[u64],
    value: i32,
) {
    let negative = value < 0;
    let magnitude = u128::from(value.unsigned_abs());
    for (accumulator, &limb) in accumulator.iter_mut().zip(coefficient_limbs) {
        let product = (u128::from(limb) * magnitude) as i128;
        if negative {
            *accumulator -= product;
        } else {
            *accumulator += product;
        }
    }
}

fn normalize_signed_accumulated_limbs(accumulated: &[i128], normalized: &mut Vec<u64>) -> bool {
    normalized.clear();
    let radix = 1_i128 << 64;
    let mut carry = 0_i128;
    for &limb in accumulated {
        let value = limb
            .checked_add(carry)
            .expect("lazy short-integer accumulator stays within its block bound");
        normalized.push(value.rem_euclid(radix) as u64);
        carry = value.div_euclid(radix);
    }
    let negative = carry < 0;
    if negative {
        let mut add_one = true;
        for limb in normalized.iter_mut() {
            let (value, carry_out) = (!*limb).overflowing_add(u64::from(add_one));
            *limb = value;
            add_one = carry_out;
        }
        let mut high = carry
            .checked_neg()
            .expect("lazy short-integer carry is not i128::MIN")
            - 1
            + i128::from(add_one);
        while high != 0 {
            normalized.push(high as u64);
            high >>= 64;
        }
    } else {
        while carry != 0 {
            normalized.push(carry as u64);
            carry >>= 64;
        }
    }
    while normalized.last() == Some(&0) {
        normalized.pop();
    }
    negative
}

fn signed_accumulated_limbs_to_field<F: PrimeField>(
    accumulated: &[i128],
    normalized: &mut Vec<u64>,
    bytes: &mut Vec<u8>,
    reduction_radix: F,
) -> F {
    let negative = normalize_signed_accumulated_limbs(accumulated, normalized);
    if normalized.is_empty() {
        return F::zero();
    }
    bytes.clear();
    for &limb in normalized.iter() {
        bytes.extend_from_slice(&limb.to_le_bytes());
    }
    let magnitude = short_integer_bytes_to_field(bytes, reduction_radix);
    if negative {
        -magnitude
    } else {
        magnitude
    }
}

#[inline]
fn short_integer_reduction_direct_bytes<F: PrimeField>() -> usize {
    (((F::MODULUS_BIT_SIZE + 7) / 8) as usize).saturating_sub(1)
}

fn short_integer_reduction_radix<F: PrimeField>() -> F {
    let direct_bytes = short_integer_reduction_direct_bytes::<F>();
    F::from(2_u64).pow([(8 * direct_bytes) as u64])
}

/// Reduce a short-integer accumulation using one high-word field multiply.
///
/// Arkworks' generic byte reduction consumes every byte above the directly
/// convertible prefix with a separate field multiplication. Packed EO
/// accumulators have at most one extra `u64` above that prefix, so parsing the
/// entire high word at once avoids repeated full-width multiplication.
fn short_integer_bytes_to_field<F: PrimeField>(bytes: &[u8], reduction_radix: F) -> F {
    let direct_bytes = short_integer_reduction_direct_bytes::<F>();
    if bytes.len() <= direct_bytes {
        return F::from_le_bytes_mod_order(bytes);
    }
    let (low_bytes, high_bytes) = bytes.split_at(direct_bytes);
    if high_bytes.len() > size_of::<u64>() {
        return F::from_le_bytes_mod_order(bytes);
    }
    let mut high_word_bytes = [0_u8; size_of::<u64>()];
    high_word_bytes[..high_bytes.len()].copy_from_slice(high_bytes);
    let high_word = u64::from_le_bytes(high_word_bytes);
    F::from_le_bytes_mod_order(low_bytes) + F::from(high_word) * reduction_radix
}

fn source_geometries<Entry, Source>(source: &Source) -> Result<Vec<BvaI8Geometry>, BvaError>
where
    Entry: BvaInteger,
    Source: BvaSource<Entry>,
{
    let layout = source.layout();
    (0..layout.vector_lengths.len())
        .map(|vector_index| {
            let geometry = source
                .vector_geometry(vector_index)
                .ok_or(BvaError::VectorGeometry)?;
            let bucket_index = *layout
                .vector_to_bucket
                .get(vector_index)
                .ok_or(BvaError::VectorIndex)?;
            if geometry.logical_len() != layout.bucket_sizes[bucket_index] {
                return Err(BvaError::VectorGeometry);
            }
            Ok(geometry)
        })
        .collect()
}

fn tensor_project_integer_geometry<F: PrimeField, Entry: BvaInteger>(
    values: &[Entry],
    geometry: BvaI8Geometry,
    factors: &[(F, F)],
) -> (F, u64) {
    debug_assert_eq!(values.len(), geometry.physical_len());
    debug_assert_eq!(1_usize << factors.len(), geometry.logical_len());
    if geometry.logical_rows() == 1 {
        return tensor_project_integer_prefix_strided(values, 0, 1, values.len(), factors);
    }

    let col_log = geometry.logical_cols().ilog2() as usize;
    let col_coefficients = tensor_coefficients(&factors[..col_log]);
    let row_coefficients = tensor_coefficients(&factors[col_log..]);
    values
        .chunks(geometry.physical_cols())
        .zip(&row_coefficients)
        .map(|(row, &row_coefficient)| {
            let mut projected = F::zero();
            let mut operations = 0_u64;
            for (&value, &col_coefficient) in row.iter().zip(&col_coefficients) {
                if value.is_zero() {
                    continue;
                }
                projected += value.to_field::<F>() * col_coefficient;
                operations += 2;
            }
            (projected * row_coefficient, operations + 1)
        })
        .fold((F::zero(), 0_u64), |left, right| {
            (left.0 + right.0, left.1 + right.1 + 1)
        })
}

fn tensor_project_integer_prefix_strided<F: PrimeField, Entry: BvaInteger>(
    values: &[Entry],
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
        return (values[start].to_field(), 0);
    }

    let high_bit = factors.len() - 1;
    let half = 1_usize << high_bit;
    let left_len = physical_len.min(half);
    let right_len = physical_len.saturating_sub(half);
    let (left, left_operations) = tensor_project_integer_prefix_strided(
        values,
        start,
        stride,
        left_len,
        &factors[..high_bit],
    );
    let (right, right_operations) = tensor_project_integer_prefix_strided(
        values,
        start + half * stride,
        stride,
        right_len,
        &factors[..high_bit],
    );
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

fn packed_integer_cross_terms<F: PrimeField, Entry: BvaInteger>(
    values: &[Entry],
    geometry: BvaI8Geometry,
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
                    let row_coefficients = tensor_coefficients(row_factors);
                    let (computed, projection_operations) = if row_coefficients.len()
                        >= SHORT_INTEGER_LAZY_MIN_ROWS
                    {
                        project_short_integer_rows_lazy(values, geometry, &row_coefficients, stride)
                    } else {
                        project_short_integer_rows_window(
                            values,
                            geometry,
                            &row_coefficients,
                            stride,
                        )
                    };
                    operations += projection_operations;
                    computed
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
    let vector_end = logical_offset + geometry.logical_len();
    let first_row = logical_offset / stride;
    let row_count = vector_end.div_ceil(stride) - first_row;
    let mut positive = vec![F::zero(); stride - 1];
    let mut negative = vec![F::zero(); stride - 1];
    let mut trace = F::zero();
    let mut operations = 0_u64;
    let mut coefficients = vec![F::zero(); stride];
    for row_index in 0..row_count {
        let row_start = (first_row + row_index) * stride;
        for (lane, coefficient) in coefficients.iter_mut().enumerate() {
            *coefficient = tbv.logical_coefficient_at(offsets, row_start + lane);
        }
        let entry_start = row_start.max(logical_offset);
        let entry_end = (row_start + stride).min(vector_end);
        for flat_index in entry_start..entry_end {
            let Some(physical_index) = geometry.physical_index(flat_index - logical_offset) else {
                continue;
            };
            let value = values[physical_index];
            if value.is_zero() {
                continue;
            }
            let left_value = value.to_field::<F>();
            let left_lane = flat_index - row_start;
            for (right_lane, &right_value) in coefficients.iter().enumerate() {
                if right_value.is_zero() {
                    continue;
                }
                let product = left_value * right_value;
                operations += 1;
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
    let (left, left_operations) = tensor_project_prefix_strided_allocation_free(
        values,
        start,
        stride,
        left_len,
        &factors[..high_bit],
    );
    let (right, right_operations) = tensor_project_prefix_strided_allocation_free(
        values,
        start + half * stride,
        stride,
        right_len,
        &factors[..high_bit],
    );
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
    geometries: &[BvaI8Geometry],
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
            let geometry = geometries[vector_index];
            let entries = geometry
                .logical_index(geometry.physical_len() - 1)
                .expect("validated geometry maps every physical entry")
                / stride
                + 1;
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

fn eo_strided_task_count(item_count: usize) -> usize {
    item_count.min(rayon::current_num_threads().max(1))
}

/// Move size-ordered items into round-robin task lanes, process each lane
/// serially, then restore canonical order without cloning item payloads.
fn map_owned_strided<T, U>(
    items: Vec<T>,
    map: impl Fn(usize, T) -> (U, u64) + Sync,
) -> (Vec<U>, u64)
where
    T: Send,
    U: Send,
{
    let item_count = items.len();
    if item_count == 0 {
        return (Vec::new(), 0);
    }
    let task_count = eo_strided_task_count(item_count);
    let mut lanes = (0..task_count)
        .map(|_| Vec::<(usize, T)>::new())
        .collect::<Vec<_>>();
    for (index, item) in items.into_iter().enumerate() {
        lanes[index % task_count].push((index, item));
    }
    let completed = lanes
        .into_par_iter()
        .map(|lane| {
            let mut operations = 0_u64;
            let mapped = lane
                .into_iter()
                .map(|(index, item)| {
                    let (item, item_operations) = map(index, item);
                    operations += item_operations;
                    (index, item)
                })
                .collect::<Vec<_>>();
            (mapped, operations)
        })
        .collect::<Vec<_>>();

    let mut restored = (0..item_count).map(|_| None).collect::<Vec<Option<U>>>();
    let mut operations = 0_u64;
    for (lane, lane_operations) in completed {
        operations += lane_operations;
        for (index, item) in lane {
            restored[index] = Some(item);
        }
    }
    (
        restored
            .into_iter()
            .map(|item| item.expect("every strided item is restored"))
            .collect(),
        operations,
    )
}

fn packed_reduced_inner_product<F: PrimeField>(buckets: &[PackedReducedBucket<F>]) -> (F, u64) {
    if buckets.is_empty() {
        return (F::zero(), 0);
    }
    let task_count = eo_strided_task_count(buckets.len());
    (0..task_count)
        .into_par_iter()
        .map(|task| {
            let mut sum = F::zero();
            let mut operations = 0_u64;
            for bucket_index in (task..buckets.len()).step_by(task_count) {
                let bucket = &buckets[bucket_index];
                let (contribution, contribution_operations) = bucket.basis.terms.iter().fold(
                    (F::zero(), 0_u64),
                    |(sum, operations), term| {
                        let (projected, projection_operations) =
                            tensor_project_strided_allocation_free(
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
                    },
                );
                sum += contribution;
                operations += contribution_operations + 1;
            }
            (sum, operations)
        })
        .collect::<Vec<_>>()
        .into_iter()
        .fold((F::zero(), 0_u64), |left, right| {
            (left.0 + right.0, left.1 + right.1)
        })
}

fn packed_eo_cross_inner_products_allocation_free<F: PrimeField>(
    values: &[F],
    basis: &TbvBucket<F>,
) -> (F, F, u64, u64) {
    let mut iota = F::zero();
    let mut rho = F::zero();
    let mut fold_operations = 0_u64;
    let mut ip_operations = 0_u64;
    let even_len = values.len().div_ceil(2);
    let odd_len = values.len() / 2;
    for term in &basis.terms {
        let ((left_factor, right_factor), tail) =
            term.factors.split_first().expect("non-terminal EO round");
        let (even, even_operations) =
            tensor_project_strided_allocation_free(values, 0, 2, even_len, tail);
        let (odd, odd_operations) =
            tensor_project_strided_allocation_free(values, 1, 2, odd_len, tail);
        iota += even * term.eta * right_factor;
        rho += odd * term.eta * left_factor;
        fold_operations += even_operations + odd_operations;
        ip_operations += 5;
    }
    (iota, rho, fold_operations, ip_operations)
}

fn packed_eo_bucketed_cross_inner_products<F: PrimeField>(
    buckets: &[PackedReducedBucket<F>],
    counters: &mut PaddingWorkCounters,
) -> (F, F) {
    let singleton_start = buckets
        .iter()
        .position(|bucket| bucket.cover_size == 1)
        .unwrap_or(buckets.len());
    let singleton_pairs = (buckets.len() - singleton_start) / 2;
    let job_count = singleton_start + singleton_pairs;
    if job_count == 0 {
        return (F::zero(), F::zero());
    }
    let task_count = eo_strided_task_count(job_count);
    let contributions = (0..task_count)
        .into_par_iter()
        .map(|task| {
            let mut iota = F::zero();
            let mut rho = F::zero();
            let mut fold_operations = 0_u64;
            let mut ip_operations = 0_u64;
            for job in (task..job_count).step_by(task_count) {
                if job < singleton_start {
                    let bucket = &buckets[job];
                    let (bucket_iota, bucket_rho, bucket_fold_ops, bucket_ip_ops) =
                        packed_eo_cross_inner_products_allocation_free(
                            &bucket.values,
                            &bucket.basis,
                        );
                    iota += bucket_iota;
                    rho += bucket_rho;
                    fold_operations += bucket_fold_ops;
                    ip_operations += bucket_ip_ops;
                    continue;
                }
                let pair_index = job - singleton_start;
                let left = &buckets[singleton_start + 2 * pair_index];
                let right = &buckets[singleton_start + 2 * pair_index + 1];
                iota += left.values[0] * right.basis.reduced_scalar();
                rho += right.values[0] * left.basis.reduced_scalar();
                ip_operations += 4;
            }
            (iota, rho, fold_operations, ip_operations)
        })
        .collect::<Vec<_>>();
    let mut iota = F::zero();
    let mut rho = F::zero();
    for (task_iota, task_rho, fold_operations, ip_operations) in contributions {
        iota += task_iota;
        rho += task_rho;
        counters.real_fold_field_ops += fold_operations;
        counters.real_ip_field_ops += ip_operations;
    }
    (iota, rho)
}

fn packed_eo_fold_prefix_in_place<F: Field>(values: &mut Vec<F>, challenge: F) -> u64 {
    let old_len = values.len();
    if old_len <= 1 {
        return 0;
    }
    let next_len = old_len.div_ceil(2);
    for output in 0..next_len {
        let input = 2 * output;
        let left = values[input];
        values[output] = if input + 1 < old_len {
            left + challenge * values[input + 1]
        } else {
            left
        };
    }
    values.truncate(next_len);
    2 * (old_len / 2) as u64
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
    let singletons = buckets.split_off(singleton_start);
    let (mut buckets, bucket_operations) = map_owned_strided(buckets, |_, mut bucket| {
        let operations = packed_eo_fold_prefix_in_place(&mut bucket.values, challenge_inv);
        bucket.cover_size /= 2;
        let basis = std::mem::replace(&mut bucket.basis, TbvBucket { terms: Vec::new() });
        bucket.basis = basis.fold_interior_owned(challenge);
        (bucket, operations)
    });
    counters.real_fold_field_ops += bucket_operations;

    let mut singleton_pairs = Vec::with_capacity(singletons.len().div_ceil(2));
    let mut singleton_iter = singletons.into_iter();
    while let Some(left) = singleton_iter.next() {
        singleton_pairs.push((left, singleton_iter.next()));
    }
    let (folded_singletons, singleton_operations) =
        map_owned_strided(singleton_pairs, |_, (mut left, right)| {
            let operations = if let Some(right) = right {
                left.values[0] += challenge_inv * right.values[0];
                let folded_basis =
                    left.basis.reduced_scalar() + challenge * right.basis.reduced_scalar();
                left.basis = TbvBucket::from_factor_terms(vec![(folded_basis, Vec::new())]);
                4
            } else {
                0
            };
            (left, operations)
        });
    counters.real_fold_field_ops += singleton_operations;
    buckets.extend(folded_singletons);
    buckets
}

struct FlatCompressionTask<'a, F> {
    witness: &'a mut [F],
    basis: &'a mut [F],
    compact_basis: TbvBucket<F>,
    source_indices: Vec<usize>,
}

fn accumulate_compact_tensor_term<F: Field>(output: &mut [F], scale: F, factors: &[(F, F)]) -> u64 {
    debug_assert_eq!(output.len(), 1_usize << factors.len());
    let Some(((left, right), prefix)) = factors.split_last() else {
        output[0] += scale;
        return 1;
    };
    let half = output.len() / 2;
    let (left_output, right_output) = output.split_at_mut(half);
    2 + accumulate_compact_tensor_term(left_output, scale * left, prefix)
        + accumulate_compact_tensor_term(right_output, scale * right, prefix)
}

/// Expand one tensor term from a scalar to its complete bucket slice.
///
/// Each round doubles the initialized prefix in place. The bucket owns a
/// disjoint output slice, so different buckets still expand concurrently in
/// the outer strided Rayon schedule.
fn expand_compact_tensor_term_in_place<F: Field>(
    output: &mut [F],
    scale: F,
    factors: &[(F, F)],
) -> u64 {
    debug_assert_eq!(output.len(), 1_usize << factors.len());
    output[0] = scale;
    let mut initialized = 1_usize;
    let mut operations = 0_u64;
    for &(left, right) in factors {
        for index in 0..initialized {
            let value = output[index];
            output[index] = value * left;
            output[index + initialized] = value * right;
        }
        operations += 2 * initialized as u64;
        initialized *= 2;
    }
    debug_assert_eq!(initialized, output.len());
    operations
}

fn compress_single_indexed_integer_source<F: PrimeField, Entry: BvaInteger>(
    output: &mut [F],
    values: &[Entry],
    geometry: BvaI8Geometry,
    lane_offset: usize,
    inverse_power_limbs: &[F::BigInt],
    stride: usize,
    reduction_radix: F,
) -> u64 {
    let limb_count = inverse_power_limbs[0].as_ref().len();
    let mut accumulated = vec![0_i128; limb_count];
    let mut normalized = Vec::with_capacity(limb_count + 1);
    let mut bytes = Vec::with_capacity((limb_count + 1) * 8);
    let mut operations = 0_u64;

    if lane_offset == 0
        && geometry.logical_rows == geometry.physical_rows
        && geometry.logical_cols == geometry.physical_cols
    {
        for (output_index, row) in values.chunks(stride).enumerate() {
            accumulated.fill(0);
            for (lane, &value) in row.iter().enumerate() {
                if value.is_zero() {
                    continue;
                }
                accumulate_signed_short_integer_limbs(
                    &mut accumulated,
                    inverse_power_limbs[lane].as_ref(),
                    value.to_i32(),
                );
            }
            output[output_index] = signed_accumulated_limbs_to_field::<F>(
                &accumulated,
                &mut normalized,
                &mut bytes,
                reduction_radix,
            );
            operations += 1;
        }
        return operations;
    }

    let mut current_output = None;
    for (physical_index, &value) in values.iter().enumerate() {
        let local_index = geometry
            .logical_index(physical_index)
            .expect("validated geometry maps every physical entry");
        let packed_index = lane_offset + local_index;
        let output_index = packed_index / stride;
        if current_output != Some(output_index) {
            if let Some(previous_output) = current_output {
                output[previous_output] = signed_accumulated_limbs_to_field::<F>(
                    &accumulated,
                    &mut normalized,
                    &mut bytes,
                    reduction_radix,
                );
                operations += 1;
            }
            accumulated.fill(0);
            current_output = Some(output_index);
        }
        if value.is_zero() {
            continue;
        }
        accumulate_signed_short_integer_limbs(
            &mut accumulated,
            inverse_power_limbs[packed_index % stride].as_ref(),
            value.to_i32(),
        );
    }
    if let Some(output_index) = current_output {
        output[output_index] = signed_accumulated_limbs_to_field::<F>(
            &accumulated,
            &mut normalized,
            &mut bytes,
            reduction_radix,
        );
        operations += 1;
    }
    operations
}

/// Compress the integer witness and compact TBV into two global flat vectors.
///
/// The transient bucket metadata only partitions disjoint output slices. It is
/// dropped before the full-width EO rounds, which operate exclusively on the
/// two contiguous vectors.
fn compress_indexed_integer_and_tbv_flat<F: PrimeField, Entry: BvaInteger>(
    layout: &BvaLayout,
    tbv: &TensorBasisVector<F>,
    geometries: &[BvaI8Geometry],
    vectors: &[&[Entry]],
    packed_challenge: F,
    inverse_powers: &[F],
    stride: usize,
) -> (Vec<F>, Vec<F>, u64) {
    let (reduced_buckets, destinations) =
        build_packed_reduced_buckets(tbv, geometries, packed_challenge, stride, 0);
    let mut destination_sources = vec![Vec::<usize>::new(); reduced_buckets.len()];
    for vector_index in 0..vectors.len() {
        destination_sources[destinations[vector_index].bucket_index].push(vector_index);
    }

    let flat_len = layout.logical_len / stride;
    let mut witness = vec![F::zero(); flat_len];
    let mut basis = vec![F::zero(); flat_len];
    let mut witness_tail = witness.as_mut_slice();
    let mut basis_tail = basis.as_mut_slice();
    let mut output_cursor = 0_usize;
    let mut tasks = Vec::with_capacity(reduced_buckets.len());
    for (bucket_index, bucket) in reduced_buckets.into_iter().enumerate() {
        let output_start = bucket.logical_offset / stride;
        debug_assert!(output_start >= output_cursor);
        let gap = output_start - output_cursor;
        let (_, witness_after_gap) = witness_tail.split_at_mut(gap);
        let (_, basis_after_gap) = basis_tail.split_at_mut(gap);
        let (witness_output, witness_next) = witness_after_gap.split_at_mut(bucket.cover_size);
        let (basis_output, basis_next) = basis_after_gap.split_at_mut(bucket.cover_size);
        witness_tail = witness_next;
        basis_tail = basis_next;
        output_cursor = output_start + bucket.cover_size;
        tasks.push(FlatCompressionTask {
            witness: witness_output,
            basis: basis_output,
            compact_basis: bucket.basis,
            source_indices: std::mem::take(&mut destination_sources[bucket_index]),
        });
    }

    let inverse_power_limbs = inverse_powers
        .iter()
        .map(|value| value.into_bigint())
        .collect::<Vec<_>>();
    let limb_count = inverse_power_limbs[0].as_ref().len();
    let reduction_radix = short_integer_reduction_radix::<F>();

    let (_, operations) = map_owned_strided(tasks, |_, task| {
        let FlatCompressionTask {
            witness,
            basis,
            compact_basis,
            source_indices,
        } = task;
        let mut operations = 0_u64;
        if source_indices.len() == 1 {
            let vector_index = source_indices[0];
            let destination = destinations[vector_index];
            operations += compress_single_indexed_integer_source(
                witness,
                vectors[vector_index],
                geometries[vector_index],
                destination.lane_offset,
                &inverse_power_limbs,
                stride,
                reduction_radix,
            );
        } else {
            let mut accumulated = vec![0_i128; witness.len() * limb_count];
            for vector_index in source_indices {
                let destination = destinations[vector_index];
                let geometry = geometries[vector_index];
                for (physical_index, &value) in vectors[vector_index].iter().enumerate() {
                    if value.is_zero() {
                        continue;
                    }
                    let local_index = geometry
                        .logical_index(physical_index)
                        .expect("validated geometry maps every physical entry");
                    let packed_index = destination.lane_offset + local_index;
                    let output_index = packed_index / stride;
                    let start = output_index * limb_count;
                    accumulate_signed_short_integer_limbs(
                        &mut accumulated[start..start + limb_count],
                        inverse_power_limbs[packed_index % stride].as_ref(),
                        value.to_i32(),
                    );
                }
            }
            let mut normalized = Vec::with_capacity(limb_count + 1);
            let mut bytes = Vec::with_capacity((limb_count + 1) * 8);
            for (output_index, output) in witness.iter_mut().enumerate() {
                let start = output_index * limb_count;
                let end = start + limb_count;
                *output = signed_accumulated_limbs_to_field::<F>(
                    &accumulated[start..end],
                    &mut normalized,
                    &mut bytes,
                    reduction_radix,
                );
                operations += 1;
            }
        }
        let mut tensor_terms = compact_basis.terms.into_iter();
        if let Some(term) = tensor_terms.next() {
            operations += expand_compact_tensor_term_in_place(basis, term.eta, &term.factors);
        }
        for term in tensor_terms {
            operations += accumulate_compact_tensor_term(basis, term.eta, &term.factors);
        }
        ((), operations)
    });
    (witness, basis, operations)
}

fn flat_inner_product<F: PrimeField>(left: &[F], right: &[F]) -> (F, u64) {
    debug_assert_eq!(left.len(), right.len());
    let value = left
        .par_iter()
        .zip(right.par_iter())
        .map(|(&left, &right)| left * right)
        .reduce(F::zero, |left, right| left + right);
    (value, 2 * left.len() as u64)
}

fn flat_eo_cross_inner_products<F: PrimeField>(witness: &[F], basis: &[F]) -> (F, F, u64) {
    debug_assert_eq!(witness.len(), basis.len());
    debug_assert_eq!(witness.len() % 2, 0);
    let (iota, rho) = witness
        .par_chunks_exact(2)
        .zip(basis.par_chunks_exact(2))
        .map(|(witness, basis)| (witness[0] * basis[1], witness[1] * basis[0]))
        .reduce(
            || (F::zero(), F::zero()),
            |left, right| (left.0 + right.0, left.1 + right.1),
        );
    (iota, rho, 4 * (witness.len() / 2) as u64)
}

/// Fold two flat vectors in place without an auxiliary field buffer.
///
/// Output index `i` depends on input indices `2i` and `2i+1`. Processing
/// breadth-first index bands keeps each parallel band disjoint from all of its
/// unread inputs, then truncates the completed prefix.
fn flat_eo_fold_pair_in_place<F: PrimeField>(
    witness: &mut Vec<F>,
    basis: &mut Vec<F>,
    challenge: F,
    challenge_inv: F,
) -> u64 {
    debug_assert_eq!(witness.len(), basis.len());
    debug_assert!(witness.len().is_power_of_two());
    let old_len = witness.len();
    if old_len <= 1 {
        return 0;
    }
    let next_len = old_len / 2;
    witness[0] = witness[0] + challenge_inv * witness[1];
    basis[0] = basis[0] + challenge * basis[1];

    let mut start = 1_usize;
    while start < next_len {
        let end = (2 * start).min(next_len);
        let (witness_prefix, witness_inputs) = witness.split_at_mut(2 * start);
        let (basis_prefix, basis_inputs) = basis.split_at_mut(2 * start);
        witness_prefix[start..end]
            .par_iter_mut()
            .zip(basis_prefix[start..end].par_iter_mut())
            .enumerate()
            .for_each(|(offset, (witness_output, basis_output))| {
                let input = 2 * offset;
                *witness_output = witness_inputs[input] + challenge_inv * witness_inputs[input + 1];
                *basis_output = basis_inputs[input] + challenge * basis_inputs[input + 1];
            });
        start = end;
    }
    witness.truncate(next_len);
    basis.truncate(next_len);
    4 * next_len as u64
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

    /// Proves Packed EO over the native [`ShortInt`] type selected by `Entry`.
    pub fn prove_streaming<Entry, S>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>>
    where
        Entry: BvaInteger,
        S: BvaSource<Entry>,
    {
        let logical_len = <S as BvaSource<Entry>>::layout(source).logical_len();
        let stride = packing_stride_for_bit_width::<F>(logical_len, bit_width)?;
        Self::prove_integer_streaming_inner::<Entry, S>(
            source, tbv, claimed, bit_width, stride, true, None, None, None, None,
        )
    }

    /// Proves transcript-bound Packed EO over the native [`ShortInt`] type
    /// selected by `Entry`.
    pub fn prove_streaming_bound<Entry, S>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
        binding: F,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>>
    where
        Entry: BvaInteger,
        S: BvaSource<Entry>,
    {
        let logical_len = <S as BvaSource<Entry>>::layout(source).logical_len();
        let stride = packing_stride_for_bit_width::<F>(logical_len, bit_width)?;
        Self::prove_integer_streaming_inner::<Entry, S>(
            source,
            tbv,
            claimed,
            bit_width,
            stride,
            true,
            Some(binding),
            None,
            None,
            None,
        )
    }

    /// Proves Packed EO with an explicit packing stride for tests and
    /// controlled performance measurements.
    pub fn prove_streaming_with_stride<Entry, S>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
        stride: usize,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>>
    where
        Entry: BvaInteger,
        S: BvaSource<Entry>,
    {
        Self::prove_integer_streaming_inner::<Entry, S>(
            source, tbv, claimed, bit_width, stride, true, None, None, None, None,
        )
    }

    pub fn prove_i8_streaming<S: BvaI8Source>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        Self::prove_streaming::<i8, S>(source, tbv, claimed, bit_width)
    }

    pub fn prove_i8_streaming_bound<S: BvaI8Source>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
        binding: F,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        Self::prove_streaming_bound::<i8, S>(source, tbv, claimed, bit_width, binding)
    }

    pub fn prove_i16_streaming<S: BvaI16Source>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        Self::prove_streaming::<i16, S>(source, tbv, claimed, bit_width)
    }

    pub fn prove_i16_streaming_bound<S: BvaI16Source>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
        binding: F,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        Self::prove_streaming_bound::<i16, S>(source, tbv, claimed, bit_width, binding)
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
        let stride =
            packing_stride_for_bit_width::<F>(BvaI8Source::layout(source).logical_len, bit_width)?;
        Self::prove_integer_streaming_inner::<i8, S>(
            source,
            tbv,
            claimed,
            bit_width,
            stride,
            true,
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
        let stride =
            packing_stride_for_bit_width::<F>(BvaI8Source::layout(source).logical_len, bit_width)?;
        Self::prove_integer_streaming_inner::<i8, S>(
            source,
            tbv,
            claimed,
            bit_width,
            stride,
            true,
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
        Self::prove_streaming_with_stride::<i8, S>(source, tbv, claimed, bit_width, stride)
    }

    /// Retained only for controlled A/B measurements against the previous
    /// per-bucket full-width rounds.
    #[doc(hidden)]
    pub fn prove_i8_streaming_with_stride_bucketed_reference<S: BvaI8Source>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
        stride: usize,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        Self::prove_integer_streaming_inner::<i8, S>(
            source, tbv, claimed, bit_width, stride, false, None, None, None, None,
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
        let stride =
            packing_stride_for_bit_width::<F>(BvaI8Source::layout(source).logical_len, bit_width)?;
        Self::prove_integer_streaming_inner::<i8, S>(
            source,
            tbv,
            claimed,
            bit_width,
            stride,
            true,
            None,
            None,
            None,
            Some(transcript),
        )
    }

    fn prove_integer_streaming_inner<Entry, S>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
        stride: usize,
        use_flat_cpu: bool,
        binding: Option<F>,
        lane_projections: Option<&PackedLaneProjections<F>>,
        mut accelerator: Option<&mut dyn PackedEoReducedVectors<F, Error = String>>,
        mut commitment_transcript: Option<&mut dyn PackedEoChallengeTranscript<F>>,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>>
    where
        Entry: BvaInteger,
        S: BvaSource<Entry>,
    {
        if bit_width == 0 || bit_width > Entry::BITS {
            return Err(BvaError::InvalidBitWidth(bit_width).into());
        }
        let layout = source.layout().clone();
        if tbv.layout != layout {
            return Err(BvaError::LayoutMismatch.into());
        }
        validate_packing_stride(layout.logical_len, stride)?;
        let geometries = source_geometries::<Entry, S>(source)?;
        let offsets = layout.bucket_offsets();
        let mut bucket_to_vector = vec![usize::MAX; layout.bucket_sizes.len()];
        for (vector_index, &bucket_index) in layout.vector_to_bucket.iter().enumerate() {
            if bucket_index >= bucket_to_vector.len()
                || bucket_to_vector[bucket_index] != usize::MAX
            {
                return Err(BvaError::VectorIndex.into());
            }
            bucket_to_vector[bucket_index] = vector_index;
        }
        if bucket_to_vector.contains(&usize::MAX) {
            return Err(BvaError::MissingVector.into());
        }
        let mut packed_positive = vec![F::zero(); stride - 1];
        let mut packed_negative = vec![F::zero(); stride - 1];
        let mut trace = F::zero();
        let mut visited = vec![false; layout.vector_lengths.len()];
        let materialized_padding_entries = source.materialized_padding_entries();
        let mut counters = PaddingWorkCounters {
            materialized_padding_entries: materialized_padding_entries as u64,
            ..Default::default()
        };
        let mut profile = StreamingEoProfile {
            source_passes: 1,
            materialized_padding_entries,
            ..Default::default()
        };
        let mut bva_error = None;
        let cross_terms_started = Instant::now();
        if let Some(vectors) = source.indexed_vectors() {
            if vectors.len() != layout.vector_lengths.len() {
                return Err(BvaError::MissingVector.into());
            }
            for (vector_index, values) in vectors.iter().enumerate() {
                if values.len() != geometries[vector_index].physical_len() {
                    return Err(BvaError::VectorLength.into());
                }
            }
            let task_count = eo_strided_task_count(layout.bucket_sizes.len());
            if bit_width < Entry::BITS {
                let bound = 1_u64 << bit_width;
                let invalid = (0..task_count)
                    .into_par_iter()
                    .map(|task| {
                        for bucket_index in (task..layout.bucket_sizes.len()).step_by(task_count) {
                            let vector_index = bucket_to_vector[bucket_index];
                            if let Some((entry_index, &value)) = vectors[vector_index]
                                .iter()
                                .enumerate()
                                .find(|(_, value)| value.unsigned_abs_u64() >= bound)
                            {
                                return Some((bucket_index, entry_index, value));
                            }
                        }
                        None
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
                    .flatten()
                    .min_by_key(|(bucket_index, entry_index, _)| (*bucket_index, *entry_index));
                if let Some((_, _, value)) = invalid {
                    return Err(BvaError::ValueOutOfRange {
                        value: value.to_i32(),
                        bit_width,
                    }
                    .into());
                }
            }
            let contributions = (0..task_count)
                .into_par_iter()
                .map(|task| {
                    let mut positive = vec![F::zero(); stride - 1];
                    let mut negative = vec![F::zero(); stride - 1];
                    let mut trace = F::zero();
                    let mut operations = 0_u64;
                    for bucket_index in (task..layout.bucket_sizes.len()).step_by(task_count) {
                        let vector_index = bucket_to_vector[bucket_index];
                        let supplied = lane_projections
                            .and_then(|lanes| lanes.get(bucket_index))
                            .and_then(|lanes| lanes.as_deref());
                        let (local_positive, local_negative, local_trace, local_operations) =
                            packed_integer_cross_terms(
                                vectors[vector_index],
                                geometries[vector_index],
                                bucket_index,
                                offsets[bucket_index],
                                &offsets,
                                tbv,
                                stride,
                                supplied,
                            );
                        for (total, contribution) in positive.iter_mut().zip(local_positive) {
                            *total += contribution;
                        }
                        for (total, contribution) in negative.iter_mut().zip(local_negative) {
                            *total += contribution;
                        }
                        trace += local_trace;
                        operations += local_operations;
                    }
                    (positive, negative, trace, operations)
                })
                .collect::<Vec<_>>();
            for (positive, negative, local_trace, operations) in contributions {
                for (total, contribution) in packed_positive.iter_mut().zip(positive) {
                    *total += contribution;
                }
                for (total, contribution) in packed_negative.iter_mut().zip(negative) {
                    *total += contribution;
                }
                trace += local_trace;
                counters.real_ip_field_ops += operations;
            }
            profile.physical_entries_per_pass = vectors.iter().map(|values| values.len()).sum();
            profile.total_visited_entries = profile.physical_entries_per_pass;
            visited.fill(true);
        } else {
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
                    let geometry = geometries[vector_index];
                    if values.len() != geometry.physical_len() {
                        bva_error = Some(BvaError::VectorLength);
                        return;
                    }
                    profile.physical_entries_per_pass += values.len();
                    profile.total_visited_entries += values.len();
                    if bit_width < Entry::BITS {
                        if let Some(&value) = values
                            .iter()
                            .find(|value| value.unsigned_abs_u64() >= (1_u64 << bit_width))
                        {
                            bva_error = Some(BvaError::ValueOutOfRange {
                                value: value.to_i32(),
                                bit_width,
                            });
                            return;
                        }
                    }
                    let logical_offset = offsets[bucket_index];
                    let supplied = lane_projections
                        .and_then(|lanes| lanes.get(bucket_index))
                        .and_then(|lanes| lanes.as_deref());
                    let (positive, negative, local_trace, operations) = packed_integer_cross_terms(
                        values,
                        geometry,
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
        }
        profile.timings.packed_cross_terms_seconds = cross_terms_started.elapsed().as_secs_f64();
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

        // The CPU indexed path materializes both packed vectors once, then
        // runs ordinary full-width EO rounds over their contiguous storage.
        // At the doubled stride the two vectors together occupy the same
        // number of field entries as the old one-sided packed witness.
        if use_flat_cpu && accelerator.is_none() {
            if let Some(vectors) = source.indexed_vectors() {
                let fold_started = Instant::now();
                let (mut flat_witness, mut flat_basis, compression_operations) =
                    compress_indexed_integer_and_tbv_flat(
                        &layout,
                        tbv,
                        &geometries,
                        &vectors,
                        packed_challenge,
                        &inverse_powers,
                        stride,
                    );
                profile.timings.first_fold_seconds = fold_started.elapsed().as_secs_f64();
                profile.source_passes += 1;
                profile.total_visited_entries +=
                    vectors.iter().map(|values| values.len()).sum::<usize>();
                profile.maximum_live_vector_entries = flat_witness
                    .len()
                    .checked_add(flat_basis.len())
                    .ok_or(BvaError::LengthOverflow)?;
                counters.real_fold_field_ops += compression_operations;

                let reduced_check_started = Instant::now();
                let mut reduced_claim = claimed;
                let mut positive_power = packed_challenge;
                let mut negative_power = packed_challenge_inv;
                for (&positive, &negative) in packed_positive.iter().zip(&packed_negative) {
                    reduced_claim += positive * positive_power + negative * negative_power;
                    positive_power *= packed_challenge;
                    negative_power *= packed_challenge_inv;
                }
                let (reduced_actual, reduced_operations) =
                    flat_inner_product(&flat_witness, &flat_basis);
                counters.real_ip_field_ops += reduced_operations;
                if reduced_actual != reduced_claim {
                    return Err(BvaError::ClaimMismatch.into());
                }
                profile.timings.reduced_check_seconds =
                    reduced_check_started.elapsed().as_secs_f64();

                let binary_round_count = (layout.logical_len / stride).ilog2() as usize;
                let mut binary_rounds = Vec::with_capacity(binary_round_count);
                for round in 0..binary_round_count {
                    let binary_cross_terms_started = Instant::now();
                    let (iota, rho, ip_operations) =
                        flat_eo_cross_inner_products(&flat_witness, &flat_basis);
                    profile.timings.binary_cross_terms_seconds +=
                        binary_cross_terms_started.elapsed().as_secs_f64();
                    counters.real_ip_field_ops += ip_operations;

                    let binary_challenge_started = Instant::now();
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
                    profile.timings.binary_challenge_seconds +=
                        binary_challenge_started.elapsed().as_secs_f64();

                    let binary_fold_started = Instant::now();
                    counters.real_fold_field_ops += flat_eo_fold_pair_in_place(
                        &mut flat_witness,
                        &mut flat_basis,
                        challenge,
                        challenge_inv,
                    );
                    profile.timings.binary_fold_seconds +=
                        binary_fold_started.elapsed().as_secs_f64();
                }
                if flat_witness.len() != 1 || flat_basis.len() != 1 {
                    return Err(BvaError::MissingFinalValue.into());
                }
                let a_reduced = flat_witness[0];
                let b_reduced = flat_basis[0];
                if let Some(transcript) = commitment_transcript.as_deref_mut() {
                    transcript
                        .bind_reduced(a_reduced, b_reduced)
                        .map_err(StreamingEoError::ChallengeTranscript)?;
                }
                return Ok((
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
                ));
            }
        }

        let unallocated_from = if accelerator.is_some() {
            ACCELERATED_MINIMUM_ENTRIES
        } else {
            usize::MAX
        };
        let (mut reduced_buckets, destinations) = build_packed_reduced_buckets(
            tbv,
            &geometries,
            packed_challenge,
            stride,
            unallocated_from,
        );
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
        if let Some(vectors) = source.indexed_vectors() {
            if vectors.len() != layout.vector_lengths.len() {
                return Err(BvaError::MissingVector.into());
            }
            let mut destination_sources = vec![Vec::<usize>::new(); reduced_buckets.len()];
            for (vector_index, values) in vectors.iter().enumerate() {
                if values.len() != geometries[vector_index].physical_len() {
                    return Err(BvaError::VectorLength.into());
                }
                let destination = destinations[vector_index].bucket_index;
                let Some(sources) = destination_sources.get_mut(destination) else {
                    return Err(BvaError::VectorIndex.into());
                };
                sources.push(vector_index);
            }
            let (filled_buckets, operations) =
                map_owned_strided(reduced_buckets, |destination_index, mut bucket| {
                    let mut operations = 0_u64;
                    for &vector_index in &destination_sources[destination_index] {
                        let values = vectors[vector_index];
                        let geometry = geometries[vector_index];
                        let bucket_index = layout.vector_to_bucket[vector_index];
                        let destination = destinations[vector_index];
                        if layout.bucket_sizes[bucket_index] >= stride {
                            for (packed_index, output) in bucket.values.iter_mut().enumerate() {
                                let mut sum = F::zero();
                                let logical_start = packed_index * stride;
                                for (lane, &inverse_power) in inverse_powers.iter().enumerate() {
                                    let Some(physical_index) =
                                        geometry.physical_index(logical_start + lane)
                                    else {
                                        continue;
                                    };
                                    let value = values[physical_index];
                                    if value.is_zero() {
                                        continue;
                                    }
                                    sum += value.to_field::<F>() * inverse_power;
                                    operations += 2;
                                }
                                *output = sum;
                            }
                            continue;
                        }
                        for (physical_index, &value) in values.iter().enumerate() {
                            if value.is_zero() {
                                continue;
                            }
                            let local_index = geometry
                                .logical_index(physical_index)
                                .expect("validated geometry maps every physical entry");
                            bucket.values[0] += value.to_field::<F>()
                                * inverse_powers[destination.lane_offset + local_index];
                            operations += 2;
                        }
                    }
                    (bucket, operations)
                });
            reduced_buckets = filled_buckets;
            counters.real_fold_field_ops += operations;
            profile.total_visited_entries +=
                vectors.iter().map(|values| values.len()).sum::<usize>();
            visited.fill(true);
        } else {
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
                    let geometry = geometries[vector_index];
                    if values.len() != geometry.physical_len() {
                        bva_error = Some(BvaError::VectorLength);
                        return;
                    }
                    profile.total_visited_entries += values.len();
                    let destination = destinations[vector_index];
                    if layout.bucket_sizes[bucket_index] >= stride {
                        for (packed_index, output) in reduced_buckets[destination.bucket_index]
                            .values
                            .iter_mut()
                            .enumerate()
                        {
                            let mut sum = F::zero();
                            let logical_start = packed_index * stride;
                            for (lane, &inverse_power) in inverse_powers.iter().enumerate() {
                                let Some(physical_index) =
                                    geometry.physical_index(logical_start + lane)
                                else {
                                    continue;
                                };
                                let value = values[physical_index];
                                if value.is_zero() {
                                    continue;
                                }
                                sum += value.to_field::<F>() * inverse_power;
                                counters.real_fold_field_ops += 2;
                            }
                            *output = sum;
                        }
                    } else {
                        for (physical_index, &value) in values.iter().enumerate() {
                            if value.is_zero() {
                                continue;
                            }
                            let local_index = geometry
                                .logical_index(physical_index)
                                .expect("validated geometry maps every physical entry");
                            reduced_buckets[destination.bucket_index].values[0] += value
                                .to_field::<F>()
                                * inverse_powers[destination.lane_offset + local_index];
                            counters.real_fold_field_ops += 2;
                        }
                    }
                })
                .map_err(StreamingEoError::Source)?;
        }
        profile.timings.first_fold_seconds = fold_started.elapsed().as_secs_f64();
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

        let binary_round_count = (layout.logical_len / stride).ilog2() as usize;
        let mut binary_rounds = Vec::with_capacity(binary_round_count);
        for round in 0..binary_round_count {
            let binary_cross_terms_started = Instant::now();
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
            profile.timings.binary_cross_terms_seconds +=
                binary_cross_terms_started.elapsed().as_secs_f64();

            let binary_challenge_started = Instant::now();
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
            profile.timings.binary_challenge_seconds +=
                binary_challenge_started.elapsed().as_secs_f64();

            let binary_fold_started = Instant::now();
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
            profile.timings.binary_fold_seconds += binary_fold_started.elapsed().as_secs_f64();
        }
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
    VectorGeometry,
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
            Self::VectorGeometry => {
                write!(
                    formatter,
                    "BVA source emitted an invalid compact vector geometry"
                )
            }
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
        add_short_integer_product, i8_to_field, packed_eo_fold_prefix_in_place,
        packed_eo_litebullet_fiat_shamir, packing_stride_for_bit_width,
        project_short_integer_rows_lazy, short_integer_bytes_to_field,
        short_integer_reduction_radix, short_integer_window_multiples, BvaI8Geometry, BvaI8Source,
        BvaInteger, BvaLayout, EoProofDecodeError, PackedEoLiteBulletProof, TensorBasisVector,
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

        fn indexed_vectors(&self) -> Option<Vec<&[i8]>> {
            Some(self.vectors.iter().map(Vec::as_slice).collect())
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

    #[derive(Clone)]
    struct StreamingOnlyI8Source {
        layout: BvaLayout,
        vectors: Vec<Vec<i8>>,
    }

    impl BvaI8Source for StreamingOnlyI8Source {
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

    struct CompactRectangleI8Source {
        layout: BvaLayout,
        values: Vec<i8>,
    }

    #[test]
    fn direct_short_integer_field_products_match_full_width_products() {
        let coefficients = [Fr::from(1_u64), Fr::from(17_u64), Fr::from(1_000_003_u64)];
        for coefficient in coefficients {
            let (i8_multiples, _) = short_integer_window_multiples::<Fr, i8>(coefficient);
            for value in i8::MIN..=i8::MAX {
                let initial = Fr::from(23_u64);
                let mut direct = initial;
                add_short_integer_product(&mut direct, value, &i8_multiples);
                assert_eq!(direct, initial + i8_to_field::<Fr>(value) * coefficient);
            }

            let (i16_multiples, _) = short_integer_window_multiples::<Fr, i16>(coefficient);
            for value in [
                i16::MIN,
                -32_767,
                -1_001,
                -128,
                -1,
                0,
                1,
                127,
                1_001,
                32_767,
                i16::MAX,
            ] {
                let initial = Fr::from(29_u64);
                let expected_value = if value >= 0 {
                    Fr::from(value as u64)
                } else {
                    -Fr::from((-(value as i32)) as u64)
                };
                let mut direct = initial;
                add_short_integer_product(&mut direct, value, &i16_multiples);
                assert_eq!(direct, initial + expected_value * coefficient);
            }
        }
    }

    #[test]
    fn short_integer_word_reduction_matches_generic_byte_reduction() {
        let radix = short_integer_reduction_radix::<Fr>();
        for byte_len in [0, 1, 30, 31, 32, 33, 35, 37, 40] {
            let bytes = (0..byte_len)
                .map(|index| (index as u8).wrapping_mul(73).wrapping_add(19))
                .collect::<Vec<_>>();
            assert_eq!(
                short_integer_bytes_to_field::<Fr>(&bytes, radix),
                Fr::from_le_bytes_mod_order(&bytes)
            );
        }
    }

    #[test]
    fn lazy_short_integer_projection_matches_full_width_with_implicit_zeros() {
        let stride = 64;
        let geometry = BvaI8Geometry::rectangle(64, 64, 63, 60).unwrap();
        let values = (0..geometry.physical_len())
            .map(|index| ((index % 256) as i16 - 128) as i8)
            .collect::<Vec<_>>();
        let coefficients = (0..64)
            .map(|row| Fr::from((row * 17 + 3) as u64))
            .collect::<Vec<_>>();
        let (actual, _) = project_short_integer_rows_lazy(&values, geometry, &coefficients, stride);
        let mut expected = vec![Fr::zero(); stride];
        for (row, &coefficient) in coefficients.iter().enumerate() {
            for (lane, projection) in expected.iter_mut().enumerate() {
                let Some(physical_index) = geometry.physical_index(row * stride + lane) else {
                    continue;
                };
                *projection += values[physical_index].to_field::<Fr>() * coefficient;
            }
        }
        assert_eq!(actual, expected);

        let stride = 32;
        let geometry = BvaI8Geometry::prefix(64 * stride, 64 * stride).unwrap();
        let values = (0..geometry.physical_len())
            .map(|index| match index % 6 {
                0 => i16::MIN,
                1 => i16::MAX,
                2 => -30_001,
                3 => 17_003,
                4 => -1,
                _ => 0,
            })
            .collect::<Vec<_>>();
        let (actual, _) = project_short_integer_rows_lazy(&values, geometry, &coefficients, stride);
        let mut expected = vec![Fr::zero(); stride];
        for (row, &coefficient) in coefficients.iter().enumerate() {
            for (lane, projection) in expected.iter_mut().enumerate() {
                *projection += values[row * stride + lane].to_field::<Fr>() * coefficient;
            }
        }
        assert_eq!(actual, expected);
    }

    impl BvaI8Source for CompactRectangleI8Source {
        type Error = Infallible;

        fn layout(&self) -> &BvaLayout {
            &self.layout
        }

        fn vector_geometry(&self, vector_index: usize) -> Option<BvaI8Geometry> {
            (vector_index == 0)
                .then(|| BvaI8Geometry::rectangle(8, 8, 8, 4).expect("valid rectangle"))
        }

        fn indexed_vectors(&self) -> Option<Vec<&[i8]>> {
            Some(vec![&self.values])
        }

        fn for_each_vector(
            &mut self,
            visit: &mut dyn FnMut(usize, &[i8]),
        ) -> Result<(), Self::Error> {
            visit(0, &self.values);
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
    fn packing_doubles_the_base_mixed_width_ratio() {
        assert_eq!(std::mem::size_of::<i8>(), 1);
        assert_eq!(std::mem::size_of::<i16>(), 2);
        assert_eq!(Fr::MODULUS_BIT_SIZE, 255);
        assert_eq!(packing_stride_for_bit_width::<Fr>(64, 8).unwrap(), 64);
        assert_eq!(packing_stride_for_bit_width::<Fr>(64, 16).unwrap(), 32);
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
        let mut streaming_source = StreamingOnlyI8Source {
            layout: layout.clone(),
            vectors: vectors.clone(),
        };
        let (streaming_proof, _, streaming_profile) =
            PackedEoLiteBulletProof::prove_i8_streaming_with_stride(
                &mut streaming_source,
                &tbv,
                claimed,
                8,
                4,
            )
            .unwrap();

        assert!(verification.accepted);
        assert_eq!(streaming_proof, proof);
        assert_eq!(proof.packing_stride, 4);
        assert_eq!(proof.packed_positive.len(), 3);
        assert_eq!(proof.packed_negative.len(), 3);
        assert_eq!(proof.binary_rounds.len(), 2);
        assert_eq!(proof.field_element_count(), 12);
        assert_eq!(verification.challenges.len(), 4);
        assert_eq!(profile.source_passes, 2);
        assert_eq!(profile.physical_entries_per_pass, 9);
        assert_eq!(profile.total_visited_entries, 18);
        assert_eq!(
            profile.maximum_live_vector_entries,
            2 * (layout.logical_len() / proof.packing_stride)
        );
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(counters.materialized_padding_entries, 0);
        assert_eq!(counters.padding_ip_field_ops, 0);
        assert_eq!(counters.padding_fold_field_ops, 0);
        for timings in [profile.timings, streaming_profile.timings] {
            for seconds in [
                timings.packed_cross_terms_seconds,
                timings.first_fold_seconds,
                timings.reduced_check_seconds,
                timings.binary_cross_terms_seconds,
                timings.binary_challenge_seconds,
                timings.binary_fold_seconds,
            ] {
                assert!(seconds.is_finite());
                assert!(seconds >= 0.0);
            }
            assert_eq!(
                timings.binary_rounds_seconds(),
                timings.binary_cross_terms_seconds
                    + timings.binary_challenge_seconds
                    + timings.binary_fold_seconds
            );
            assert_eq!(
                timings.accounted_seconds(),
                timings.packed_cross_terms_seconds
                    + timings.first_fold_seconds
                    + timings.reduced_check_seconds
                    + timings.binary_rounds_seconds()
            );
        }
        assert_eq!(
            proof.a_reduced,
            dense_fold_witness(&layout, &vectors, &verification.challenges_inv)
        );

        let mut no_binary_source = ReplayI8Source {
            layout: layout.clone(),
            vectors: vectors.clone(),
        };
        let (_, _, no_binary_profile) = PackedEoLiteBulletProof::prove_i8_streaming_with_stride(
            &mut no_binary_source,
            &tbv,
            claimed,
            8,
            layout.logical_len(),
        )
        .unwrap();
        assert_eq!(no_binary_profile.timings.binary_cross_terms_seconds, 0.0);
        assert_eq!(no_binary_profile.timings.binary_challenge_seconds, 0.0);
        assert_eq!(no_binary_profile.timings.binary_fold_seconds, 0.0);

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
    fn compact_rectangle_allocates_every_occupied_packed_row() {
        let layout = BvaLayout {
            logical_len: 64,
            bucket_sizes: vec![64],
            vector_to_bucket: vec![0],
            vector_lengths: vec![32],
        };
        let values = (0..32)
            .map(|index| (index % 17) as i8 - 8)
            .collect::<Vec<_>>();
        let challenges = (0..6)
            .map(|round| Fr::from(round as u64 + 2))
            .collect::<Vec<_>>();
        let tbv = TensorBasisVector::global_tensor(&layout, &challenges).unwrap();
        let mut source = CompactRectangleI8Source { layout, values };
        let claimed = tbv.inner_product_i8_streaming(&mut source).unwrap();

        let (proof, _, profile) = PackedEoLiteBulletProof::prove_i8_streaming_with_stride(
            &mut source,
            &tbv,
            claimed,
            8,
            32,
        )
        .unwrap();

        assert!(proof.verify_with_tbv(&tbv, claimed).accepted);
        assert_eq!(profile.maximum_live_vector_entries, 4);
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
        let expected = dense_packed_cross_terms(&layout, &vectors, &tbv, 64);
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
            64,
        )
        .unwrap();
        let verification = proof.verify_with_tbv(&tbv, claimed);

        assert_eq!(
            proof,
            dense_packed_proof(&layout, &vectors, &tbv, claimed, 64)
        );
        assert_eq!(proof.packed_positive, expected.0);
        assert_eq!(proof.packed_negative, expected.1);
        assert_eq!(claimed, expected.2);
        assert!(verification.accepted);
        assert_eq!(
            proof.a_reduced,
            dense_fold_witness(&layout, &vectors, &verification.challenges_inv)
        );
        assert_eq!(
            profile.maximum_live_vector_entries,
            2 * (layout.logical_len() / 64)
        );
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

        let operations = packed_eo_fold_prefix_in_place(&mut values, challenge);

        let expected = original
            .chunks(2)
            .map(|pair| pair[0] + challenge * pair.get(1).copied().unwrap_or_default())
            .collect::<Vec<_>>();
        assert_eq!(values, expected);
        assert_eq!(values.as_ptr(), pointer);
        assert_eq!(values.capacity(), capacity);
        assert_eq!(operations, 16);
    }

    #[test]
    fn packed_eo_is_deterministic_across_heterogeneous_strided_workers() {
        let lengths = [
            1 << 15,
            (1 << 14) + 1,
            1 << 13,
            (1 << 12) + 1,
            1 << 11,
            (1 << 10) + 1,
            513,
            257,
            129,
            65,
            33,
            17,
            7,
            3,
            1,
            1,
        ];
        let vectors = lengths
            .iter()
            .enumerate()
            .map(|(vector_index, &length)| {
                (0..length)
                    .map(|entry_index| ((entry_index + 3 * vector_index) % 127) as i8 - 63)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
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
        let eight_threads = prove(8);
        let sixty_four_threads = prove(64);
        assert_eq!(serial, eight_threads);
        assert_eq!(serial, sixty_four_threads);
        assert!(sixty_four_threads.verify_with_tbv(&tbv, claimed).accepted);
    }
}
