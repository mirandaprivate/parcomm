use ark_ff::{Field, PrimeField};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};
use rayon::prelude::*;
use std::collections::BTreeMap;
use std::convert::Infallible;
use std::fmt::{Display, Formatter};
use std::io::Cursor;

use crate::utils::fiat_shamir::FiatShamir;

/// A rewindable source for the real field-valued vectors in one BVA layout.
/// Implementations must never emit logical padding buckets or short-vector
/// suffixes.
pub trait BvaFieldSource<F: PrimeField> {
    type Error;

    fn layout(&self) -> &BvaLayout;

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[F])) -> Result<(), Self::Error>;
}

/// A rewindable source for legacy signed 32-bit BVA vectors. The byte-backed
/// packed EO path uses [`BvaI8Source`] instead.
pub trait BvaI32Source {
    type Error;

    fn layout(&self) -> &BvaLayout;

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[i32])) -> Result<(), Self::Error>;
}

/// A rewindable source whose physical witness storage is exactly one signed
/// byte per entry. The packed 8-bit EO path never widens a complete vector to
/// `i32` or to field elements.
pub trait BvaI8Source {
    type Error;

    fn layout(&self) -> &BvaLayout;

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[i8])) -> Result<(), Self::Error>;
}

/// A rewindable source for two secret field-valued BVAs with the same layout.
/// Each callback contains one matching pair of physical vectors; logical
/// padding is represented only by [`BvaLayout`].
pub trait BvaFieldPairSource<F: PrimeField> {
    type Error;

    fn layout(&self) -> &BvaLayout;

    fn for_each_vector_pair(
        &mut self,
        visit: &mut dyn FnMut(usize, &[F], &[F]),
    ) -> Result<(), Self::Error>;
}

/// A sparse source for two secret field-valued BVAs with the same layout.
/// The callback enumerates only coordinates where at least one operand is
/// nonzero. Coordinates must be unique; all omitted physical entries and all
/// logical padding are interpreted as zero.
pub trait SparseBvaFieldPairSource<F: PrimeField> {
    type Error;

    fn layout(&self) -> &BvaLayout;

    fn for_each_nonzero_pair(
        &mut self,
        visit: &mut dyn FnMut(usize, usize, F, F),
    ) -> Result<(), Self::Error>;
}

/// The metadata-only output of the decreasing-size BVA layout in the
/// BatchPCS paper. `bucket_sizes` contains exactly one real bucket per input
/// vector. Internal bucket suffixes and the final logical suffix are zeros and
/// are never allocated or represented as buckets.
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

        for (original_index, rounded) in sorted {
            vector_to_bucket[original_index] = bucket_sizes.len();
            bucket_sizes.push(rounded);
            active_len = active_len
                .checked_add(rounded)
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

/// A BVA whose payload contains only the original vectors.  Logical padding is
/// represented by `BvaLayout` and never appears in `vectors`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BucketedVector<F> {
    layout: BvaLayout,
    vectors: Vec<Vec<F>>,
}

impl<F> BucketedVector<F> {
    pub fn new(vectors: Vec<Vec<F>>) -> Result<Self, BvaError> {
        let lengths = vectors.iter().map(Vec::len).collect::<Vec<_>>();
        let layout = BvaLayout::from_lengths(&lengths)?;
        Ok(Self { layout, vectors })
    }

    pub fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    pub fn physical_entries(&self) -> usize {
        self.vectors.iter().map(Vec::len).sum()
    }

    pub fn logical_padding_entries(&self) -> usize {
        self.layout.logical_len - self.physical_entries()
    }

    pub(crate) fn vectors(&self) -> &[Vec<F>] {
        &self.vectors
    }

    pub(crate) fn vectors_mut(&mut self) -> &mut [Vec<F>] {
        &mut self.vectors
    }
}

impl<F: PrimeField> BvaFieldSource<F> for BucketedVector<F> {
    type Error = Infallible;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[F])) -> Result<(), Self::Error> {
        for (index, vector) in self.vectors.iter().enumerate() {
            visit(index, vector);
        }
        Ok(())
    }
}

impl<F: Field> BucketedVector<F> {
    /// Computes Equation (11) of the BVA construction by visiting only the
    /// original vectors. Missing suffix entries and padding buckets have no
    /// representation and therefore cannot trigger a field operation.
    pub fn inner_product(
        &self,
        tbv: &TensorBasisVector<F>,
    ) -> Result<BvaInnerProduct<F>, BvaError> {
        if self.layout != tbv.layout {
            return Err(BvaError::LayoutMismatch);
        }

        let mut counters = PaddingWorkCounters::default();
        let mut value = F::zero();
        for (vector_index, source) in self.vectors.iter().enumerate() {
            let (contribution, vector_counters) = tbv.project_physical_vector(vector_index, source);
            value += contribution;
            counters.real_ip_field_ops += vector_counters.real_ip_field_ops;
            counters.real_fold_field_ops += vector_counters.real_fold_field_ops;
        }

        Ok(BvaInnerProduct { value, counters })
    }

    /// Applies bucket-wise EO-folding until one scalar remains. Logical zero
    /// buckets are `None`; the singleton merge rules never construct `[0]`.
    pub fn fold_to_scalar(self, challenges: &[F]) -> Result<BvaFoldResult<F>, BvaError> {
        let expected = self.layout.logical_len.ilog2() as usize;
        if challenges.len() != expected {
            return Err(BvaError::FoldChallengeCount {
                expected,
                actual: challenges.len(),
            });
        }

        let mut payloads = self.vectors.into_iter().map(Some).collect::<Vec<_>>();
        let mut buckets = self
            .layout
            .bucket_sizes
            .iter()
            .copied()
            .map(|bucket_size| (bucket_size, None))
            .collect::<Vec<(usize, Option<Vec<F>>)>>();
        for (vector_index, &bucket_index) in self.layout.vector_to_bucket.iter().enumerate() {
            buckets[bucket_index].1 = payloads[vector_index].take();
        }

        let mut logical_len = self.layout.logical_len;
        let mut counters = PaddingWorkCounters::default();
        for &challenge in challenges {
            let singleton_start = buckets
                .iter()
                .position(|(bucket_size, _)| *bucket_size == 1)
                .unwrap_or(buckets.len());
            let singleton_count = buckets.len() - singleton_start;
            let mut folded = Vec::with_capacity(singleton_start + singleton_count.div_ceil(2));
            for (bucket_size, values) in &mut buckets[..singleton_start] {
                debug_assert!(*bucket_size >= 2);
                if let Some(values) = values {
                    eo_fold_prefix_in_place(values, challenge, &mut counters);
                }
                folded.push((*bucket_size / 2, values.take()));
            }
            for pair in buckets[singleton_start..].chunks_exact_mut(2) {
                let left = pair[0].1.take();
                let right = pair[1].1.take();
                let value = match (left, right) {
                    (None, None) => None,
                    (Some(left), None) => Some(left),
                    (None, Some(mut right)) => {
                        right[0] *= challenge;
                        counters.real_fold_field_ops += 1;
                        Some(right)
                    }
                    (Some(mut left), Some(right)) => {
                        left[0] += challenge * right[0];
                        counters.real_fold_field_ops += 2;
                        Some(left)
                    }
                };
                folded.push((1, value));
            }
            if singleton_count % 2 == 1 {
                let last = buckets
                    .last_mut()
                    .expect("odd singleton suffix contains a final bucket");
                folded.push((1, last.1.take()));
            }
            buckets = folded;
            logical_len /= 2;
        }

        debug_assert_eq!(logical_len, 1);
        debug_assert_eq!(buckets.len(), 1);
        let value = buckets
            .pop()
            .and_then(|(_, values)| values)
            .and_then(|mut values| values.pop())
            .ok_or(BvaError::MissingFinalValue)?;
        Ok(BvaFoldResult { value, counters })
    }
}

fn eo_fold_prefix_in_place<F: Field>(
    values: &mut Vec<F>,
    challenge: F,
    counters: &mut PaddingWorkCounters,
) {
    let next_len = values.len().div_ceil(2);
    for output_index in 0..next_len {
        let left_index = output_index * 2;
        let right_index = left_index + 1;
        let left = values[left_index];
        values[output_index] = if right_index < values.len() {
            counters.real_fold_field_ops += 2;
            left + challenge * values[right_index]
        } else {
            // The absent right entry is logical padding. Copying the real left
            // entry performs no field arithmetic involving that padding.
            left
        };
    }
    values.truncate(next_len);
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TbvBucket<F> {
    terms: Vec<TbvTerm<F>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct TbvTerm<F> {
    eta: F,
    factors: Vec<(F, F)>,
}

/// Sums one tensor term over the real prefix `[0, physical_len)`. The prefix
/// is decomposed into aligned power-of-two blocks, so work is logarithmic in
/// the logical bucket size and never visits either real entries or padding.
fn tensor_term_prefix_sum<F: Field>(eta: F, factors: &[(F, F)], physical_len: usize) -> F {
    if physical_len == 0 {
        return F::zero();
    }

    let mut lower_block_sums = Vec::with_capacity(factors.len());
    let mut complete_sum = F::one();
    for (left, right) in factors {
        lower_block_sums.push(complete_sum);
        complete_sum *= *left + *right;
    }

    let logical_len = 1_usize << factors.len();
    debug_assert!(physical_len <= logical_len);
    if physical_len == logical_len {
        return eta * complete_sum;
    }

    let mut prefix_sum = F::zero();
    let mut fixed_high_product = eta;
    for bit in (0..factors.len()).rev() {
        let (left, right) = factors[bit];
        if (physical_len >> bit) & 1 == 1 {
            prefix_sum += fixed_high_product * left * lower_block_sums[bit];
            fixed_high_product *= right;
        } else {
            fixed_high_product *= left;
        }
    }
    prefix_sum
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

    /// Represents a public coefficient vector as a sum of tensor terms over
    /// the same BVA bucket. This is needed when one committed witness column
    /// participates in more than one routed relation without duplicating that
    /// column in `C_w`.
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

    pub(crate) fn from_factor_terms(terms: Vec<(F, Vec<(F, F)>)>) -> Self {
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

    fn fold_interior_reversed_owned(mut self, challenge: F) -> Self {
        for term in &mut self.terms {
            let (left, right) = term.factors.pop().expect("validated TBV term");
            term.eta *= left + challenge * right;
        }
        self
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorBasisVector<F> {
    layout: BvaLayout,
    buckets: Vec<TbvBucket<F>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TensorVectorRoute<F> {
    Omit,
    Direct {
        target_vector: usize,
        scale: F,
    },
    /// Broadcasts one target witness entry over the complete real source
    /// vector, then routes the accumulated coefficient back to that entry.
    Point {
        target_vector: usize,
        target_index: usize,
        scale: F,
    },
}

/// Routes one aligned power-of-two segment of a source TBV vector onto one
/// target BVA vector. The target's physical suffix may be shorter than its
/// power-of-two cover; those omitted coefficients continue to represent
/// conceptual zeros.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorSegmentRoute<F> {
    pub source_vector: usize,
    pub source_offset: usize,
    pub target_vector: usize,
    pub scale: F,
}

/// Routes one contiguous physical source subsegment to one contiguous physical
/// target subsegment. The implementation decomposes the translation into
/// aligned power-of-two pieces represented as tensor terms; it never expands
/// either vector or its logical padding.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TensorSubsegmentRoute<F> {
    pub source_vector: usize,
    pub source_offset: usize,
    pub target_vector: usize,
    pub target_offset: usize,
    pub length: usize,
    pub scale: F,
}

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
    /// Builds the coefficient TBV induced by folding the complete logical BVA
    /// with one EO challenge per low-to-high index bit. Bucket placement is
    /// handled in metadata; no global coefficient vector or padding payload is
    /// allocated.
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

    fn logical_coefficient_at(&self, offsets: &[usize], flat_index: usize) -> F {
        if flat_index >= self.layout.active_len() {
            return F::zero();
        }
        let bucket_index = offsets.partition_point(|offset| *offset <= flat_index) - 1;
        let local_index = flat_index - offsets[bucket_index];
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

    /// Applies the correlated low-bit EO challenges represented by one packed
    /// challenge and rebases the result as one tensor bucket. This keeps the
    /// public query compact while the native witness becomes one flat reduced
    /// field vector.
    fn packed_fold_to_flat(&self, packed_challenge: F, stride: usize) -> Result<Self, BvaError> {
        validate_packing_stride(self.layout.logical_len, stride)?;
        let stride_log = stride.ilog2() as usize;
        let reduced_physical_len = self.layout.active_len().div_ceil(stride);
        let reduced_layout = BvaLayout::from_lengths(&[reduced_physical_len])?;
        if reduced_layout.logical_len != self.layout.logical_len / stride {
            return Err(BvaError::InvalidPackingStride {
                stride,
                logical_len: self.layout.logical_len,
            });
        }
        let reduced_log = reduced_layout.logical_len.ilog2() as usize;
        let offsets = self.layout.bucket_offsets();
        let mut challenge_powers = Vec::with_capacity(stride_log);
        let mut power = packed_challenge;
        for _ in 0..stride_log {
            challenge_powers.push(power);
            power.square_in_place();
        }

        let mut reduced_terms = Vec::new();
        for (bucket_index, (&bucket_size, &offset)) in
            self.layout.bucket_sizes.iter().zip(&offsets).enumerate()
        {
            if bucket_size >= stride {
                let reduced_size = bucket_size / stride;
                let reduced_interior_log = reduced_size.ilog2() as usize;
                let reduced_offset = offset / stride;
                for term in &self.buckets[bucket_index].terms {
                    let mut eta = term.eta;
                    for ((left, right), challenge) in
                        term.factors.iter().take(stride_log).zip(&challenge_powers)
                    {
                        eta *= *left + *challenge * right;
                    }
                    let mut factors = term.factors[stride_log..].to_vec();
                    debug_assert_eq!(factors.len(), reduced_interior_log);
                    factors.extend((reduced_interior_log..reduced_log).map(|bit| {
                        if (reduced_offset >> bit) & 1 == 0 {
                            (F::one(), F::zero())
                        } else {
                            (F::zero(), F::one())
                        }
                    }));
                    reduced_terms.push((eta, factors));
                }
            } else {
                let reduced_offset = offset / stride;
                let lane_offset = offset % stride;
                let lane_scale = packed_challenge.pow([lane_offset as u64]);
                for term in &self.buckets[bucket_index].terms {
                    let mut eta = term.eta * lane_scale;
                    for ((left, right), challenge) in term.factors.iter().zip(&challenge_powers) {
                        eta *= *left + *challenge * right;
                    }
                    let factors = (0..reduced_log)
                        .map(|bit| {
                            if (reduced_offset >> bit) & 1 == 0 {
                                (F::one(), F::zero())
                            } else {
                                (F::zero(), F::one())
                            }
                        })
                        .collect();
                    reduced_terms.push((eta, factors));
                }
            }
        }
        Self::new(
            &reduced_layout,
            vec![TbvBucket::from_factor_terms(reduced_terms)],
        )
    }

    /// Pointwise multiplies two public coefficient vectors in tensor form.
    /// Products of sums are expanded only at the metadata term level.
    pub fn pointwise_product(&self, other: &Self) -> Result<Self, BvaError> {
        if self.layout != other.layout {
            return Err(BvaError::LayoutMismatch);
        }
        let buckets = self
            .buckets
            .iter()
            .zip(&other.buckets)
            .map(|(left, right)| {
                let mut terms = Vec::with_capacity(left.terms.len() * right.terms.len());
                for left_term in &left.terms {
                    for right_term in &right.terms {
                        debug_assert_eq!(left_term.factors.len(), right_term.factors.len());
                        terms.push((
                            left_term.eta * right_term.eta,
                            left_term
                                .factors
                                .iter()
                                .zip(&right_term.factors)
                                .map(|((left_zero, left_one), (right_zero, right_one))| {
                                    (*left_zero * *right_zero, *left_one * *right_one)
                                })
                                .collect(),
                        ));
                    }
                }
                TbvBucket::from_factor_terms(terms)
            })
            .collect();
        Self::new(&self.layout, buckets)
    }

    /// Routes complete physical source vectors into another BVA layout. Each
    /// source vector can be omitted or mapped to one target vector with a
    /// public scale. Multiple routes may accumulate on the same committed
    /// witness vector.
    pub fn route_vectors(
        &self,
        target_layout: &BvaLayout,
        routes: &[Option<(usize, F)>],
    ) -> Result<Self, BvaError> {
        if routes.len() != self.layout.vector_to_bucket.len() {
            return Err(BvaError::VectorIndex);
        }
        let mut target_terms = vec![Vec::new(); target_layout.bucket_sizes.len()];
        for (source_vector, route) in routes.iter().enumerate() {
            let Some((target_vector, scale)) = route else {
                continue;
            };
            let Some(&target_bucket) = target_layout.vector_to_bucket.get(*target_vector) else {
                return Err(BvaError::VectorIndex);
            };
            if self.layout.vector_lengths[source_vector]
                != target_layout.vector_lengths[*target_vector]
            {
                return Err(BvaError::VectorLength);
            }
            let source_bucket = self.layout.vector_to_bucket[source_vector];
            if self.layout.bucket_sizes[source_bucket] != target_layout.bucket_sizes[target_bucket]
            {
                return Err(BvaError::LayoutMismatch);
            }
            target_terms[target_bucket].extend(
                self.buckets[source_bucket]
                    .terms
                    .iter()
                    .map(|term| (*scale * term.eta, term.factors.clone())),
            );
        }
        let buckets = target_terms
            .into_iter()
            .map(TbvBucket::from_factor_terms)
            .collect();
        Self::new(target_layout, buckets)
    }

    pub fn route_vector_maps(
        &self,
        target_layout: &BvaLayout,
        routes: &[TensorVectorRoute<F>],
    ) -> Result<Self, BvaError> {
        if routes.len() != self.layout.vector_to_bucket.len() {
            return Err(BvaError::VectorIndex);
        }
        let mut target_terms = vec![Vec::new(); target_layout.bucket_sizes.len()];
        for (source_vector, route) in routes.iter().enumerate() {
            match *route {
                TensorVectorRoute::Omit => {}
                TensorVectorRoute::Direct {
                    target_vector,
                    scale,
                } => {
                    let Some(&target_bucket) = target_layout.vector_to_bucket.get(target_vector)
                    else {
                        return Err(BvaError::VectorIndex);
                    };
                    if self.layout.vector_lengths[source_vector]
                        != target_layout.vector_lengths[target_vector]
                    {
                        return Err(BvaError::VectorLength);
                    }
                    let source_bucket = self.layout.vector_to_bucket[source_vector];
                    if self.layout.bucket_sizes[source_bucket]
                        != target_layout.bucket_sizes[target_bucket]
                    {
                        return Err(BvaError::LayoutMismatch);
                    }
                    target_terms[target_bucket].extend(
                        self.buckets[source_bucket]
                            .terms
                            .iter()
                            .map(|term| (scale * term.eta, term.factors.clone())),
                    );
                }
                TensorVectorRoute::Point {
                    target_vector,
                    target_index,
                    scale,
                } => {
                    let Some(&target_bucket) = target_layout.vector_to_bucket.get(target_vector)
                    else {
                        return Err(BvaError::VectorIndex);
                    };
                    if target_index >= target_layout.vector_lengths[target_vector] {
                        return Err(BvaError::VectorIndex);
                    }
                    let factors = (0..target_layout.bucket_sizes[target_bucket].ilog2() as usize)
                        .map(|bit| {
                            if (target_index >> bit) & 1 == 0 {
                                (F::one(), F::zero())
                            } else {
                                (F::zero(), F::one())
                            }
                        })
                        .collect();
                    target_terms[target_bucket].push((
                        scale * self.physical_coefficient_sum(source_vector)?,
                        factors,
                    ));
                }
            }
        }
        let buckets = target_terms
            .into_iter()
            .map(TbvBucket::from_factor_terms)
            .collect();
        Self::new(target_layout, buckets)
    }

    /// Restricts aligned source segments and routes their tensor coefficients
    /// to target vectors without expanding either source or target padding.
    pub fn route_aligned_segments(
        &self,
        target_layout: &BvaLayout,
        routes: &[TensorSegmentRoute<F>],
    ) -> Result<Self, BvaError> {
        let mut target_terms = vec![Vec::new(); target_layout.bucket_sizes.len()];
        for route in routes {
            let Some(&source_bucket) = self.layout.vector_to_bucket.get(route.source_vector) else {
                return Err(BvaError::VectorIndex);
            };
            let Some(&target_bucket) = target_layout.vector_to_bucket.get(route.target_vector)
            else {
                return Err(BvaError::VectorIndex);
            };
            let source_cover = self.layout.bucket_sizes[source_bucket];
            let target_cover = target_layout.bucket_sizes[target_bucket];
            if route.source_offset % target_cover != 0
                || route
                    .source_offset
                    .checked_add(target_cover)
                    .is_none_or(|end| end > source_cover)
            {
                return Err(BvaError::LayoutMismatch);
            }
            let target_log = target_cover.ilog2() as usize;
            for term in &self.buckets[source_bucket].terms {
                let mut eta = route.scale * term.eta;
                for (bit, (left, right)) in term.factors.iter().enumerate().skip(target_log) {
                    eta *= if (route.source_offset >> bit) & 1 == 0 {
                        *left
                    } else {
                        *right
                    };
                }
                target_terms[target_bucket].push((eta, term.factors[..target_log].to_vec()));
            }
        }
        Self::new(
            target_layout,
            target_terms
                .into_iter()
                .map(TbvBucket::from_factor_terms)
                .collect(),
        )
    }

    /// Routes arbitrary contiguous physical subsegments by decomposing each
    /// source/target translation into mutually aligned dyadic pieces.
    pub fn route_subsegments(
        &self,
        target_layout: &BvaLayout,
        routes: &[TensorSubsegmentRoute<F>],
    ) -> Result<Self, BvaError> {
        let mut target_terms = vec![Vec::new(); target_layout.bucket_sizes.len()];
        for route in routes {
            if route.length == 0 {
                return Err(BvaError::ZeroLengthVector);
            }
            let Some(&source_bucket) = self.layout.vector_to_bucket.get(route.source_vector) else {
                return Err(BvaError::VectorIndex);
            };
            let Some(&target_bucket) = target_layout.vector_to_bucket.get(route.target_vector)
            else {
                return Err(BvaError::VectorIndex);
            };
            let source_physical_len = self.layout.vector_lengths[route.source_vector];
            let target_physical_len = target_layout.vector_lengths[route.target_vector];
            let source_end = route
                .source_offset
                .checked_add(route.length)
                .ok_or(BvaError::LengthOverflow)?;
            let target_end = route
                .target_offset
                .checked_add(route.length)
                .ok_or(BvaError::LengthOverflow)?;
            if source_end > source_physical_len || target_end > target_physical_len {
                return Err(BvaError::VectorLength);
            }

            let target_log = target_layout.bucket_sizes[target_bucket].ilog2() as usize;
            let mut source_position = route.source_offset;
            let mut target_position = route.target_offset;
            let mut remaining = route.length;
            while remaining > 0 {
                let mut chunk_len = 1_usize << remaining.ilog2();
                while source_position % chunk_len != 0 || target_position % chunk_len != 0 {
                    chunk_len /= 2;
                }
                let chunk_log = chunk_len.ilog2() as usize;
                for term in &self.buckets[source_bucket].terms {
                    let mut eta = route.scale * term.eta;
                    for (bit, (left, right)) in term.factors.iter().enumerate().skip(chunk_log) {
                        eta *= if (source_position >> bit) & 1 == 0 {
                            *left
                        } else {
                            *right
                        };
                    }
                    let mut factors = term.factors[..chunk_log].to_vec();
                    factors.extend((chunk_log..target_log).map(|bit| {
                        if (target_position >> bit) & 1 == 0 {
                            (F::one(), F::zero())
                        } else {
                            (F::zero(), F::one())
                        }
                    }));
                    target_terms[target_bucket].push((eta, factors));
                }
                source_position += chunk_len;
                target_position += chunk_len;
                remaining -= chunk_len;
            }
        }
        Self::new(
            target_layout,
            target_terms
                .into_iter()
                .map(TbvBucket::from_factor_terms)
                .collect(),
        )
    }

    /// Expands coefficients only for one real source vector. The logical
    /// suffix of its bucket remains virtual and is never returned.
    pub fn physical_coefficients(&self, vector_index: usize) -> Result<Vec<F>, BvaError> {
        let Some(&bucket_index) = self.layout.vector_to_bucket.get(vector_index) else {
            return Err(BvaError::VectorIndex);
        };
        let physical_len = self.layout.vector_lengths[vector_index];
        let bucket = &self.buckets[bucket_index];
        Ok((0..physical_len)
            .map(|index| {
                bucket.terms.iter().fold(F::zero(), |sum, term| {
                    let coefficient = term.factors.iter().enumerate().fold(
                        term.eta,
                        |product, (bit, (left, right))| {
                            product
                                * if (index >> bit) & 1 == 0 {
                                    *left
                                } else {
                                    *right
                                }
                        },
                    );
                    sum + coefficient
                })
            })
            .collect())
    }

    /// Evaluates one real coefficient without allocating the physical vector
    /// or its logical zero suffix.
    pub fn physical_coefficient_at(
        &self,
        vector_index: usize,
        entry_index: usize,
    ) -> Result<F, BvaError> {
        let Some(&bucket_index) = self.layout.vector_to_bucket.get(vector_index) else {
            return Err(BvaError::VectorIndex);
        };
        if entry_index >= self.layout.vector_lengths[vector_index] {
            return Err(BvaError::VectorIndex);
        }
        let bucket = &self.buckets[bucket_index];
        Ok(bucket.terms.iter().fold(F::zero(), |sum, term| {
            sum + term
                .factors
                .iter()
                .enumerate()
                .fold(term.eta, |product, (bit, (left, right))| {
                    product
                        * if (entry_index >> bit) & 1 == 0 {
                            *left
                        } else {
                            *right
                        }
                })
        }))
    }

    /// Multiplies one real source vector by its public TBV coefficients in
    /// place, without allocating either the logical bucket suffix or a second
    /// coefficient vector.
    pub fn scale_physical_vector_in_place(
        &self,
        vector_index: usize,
        values: &mut [F],
    ) -> Result<(), BvaError> {
        let Some(&bucket_index) = self.layout.vector_to_bucket.get(vector_index) else {
            return Err(BvaError::VectorIndex);
        };
        if values.len() != self.layout.vector_lengths[vector_index] {
            return Err(BvaError::VectorLength);
        }
        let bucket = &self.buckets[bucket_index];
        for (index, value) in values.iter_mut().enumerate() {
            let coefficient = bucket.terms.iter().fold(F::zero(), |sum, term| {
                let term_value = term.factors.iter().enumerate().fold(
                    term.eta,
                    |product, (bit, (left, right))| {
                        product
                            * if (index >> bit) & 1 == 0 {
                                *left
                            } else {
                                *right
                            }
                    },
                );
                sum + term_value
            });
            *value *= coefficient;
        }
        Ok(())
    }

    pub fn physical_coefficient_sum(&self, vector_index: usize) -> Result<F, BvaError> {
        let Some(&bucket_index) = self.layout.vector_to_bucket.get(vector_index) else {
            return Err(BvaError::VectorIndex);
        };
        let physical_len = self.layout.vector_lengths[vector_index];
        let bucket = &self.buckets[bucket_index];
        Ok(bucket.terms.iter().fold(F::zero(), |sum, term| {
            sum + tensor_term_prefix_sum(term.eta, &term.factors, physical_len)
        }))
    }

    pub fn linear_combination(layout: &BvaLayout, claims: &[(F, &Self)]) -> Result<Self, BvaError> {
        if claims.is_empty() {
            return Err(BvaError::EmptyClaimBatch);
        }
        if claims.iter().any(|(_, tbv)| tbv.layout != *layout) {
            return Err(BvaError::LayoutMismatch);
        }
        let buckets = (0..layout.bucket_sizes.len())
            .map(|bucket| {
                let mut terms = Vec::new();
                for (weight, tbv) in claims {
                    terms.extend(
                        tbv.buckets[bucket]
                            .terms
                            .iter()
                            .map(|term| (*weight * term.eta, term.factors.clone())),
                    );
                }
                TbvBucket::from_factor_terms(terms)
            })
            .collect();
        Self::new(layout, buckets)
    }

    /// Consumes the component TBVs while forming their linear combination.
    /// Unlike [`Self::linear_combination`], tensor factors are moved into the
    /// result instead of cloned. This is important for large synchronized
    /// openings whose routed coefficient metadata must not exist twice at the
    /// peak, and it still never expands a physical coefficient vector or any
    /// logical padding.
    pub fn linear_combination_owned(
        layout: &BvaLayout,
        claims: Vec<(F, Self)>,
    ) -> Result<Self, BvaError> {
        if claims.is_empty() {
            return Err(BvaError::EmptyClaimBatch);
        }
        if claims.iter().any(|(_, tbv)| tbv.layout != *layout) {
            return Err(BvaError::LayoutMismatch);
        }

        let mut combined = (0..layout.bucket_sizes.len())
            .map(|_| TbvBucket { terms: Vec::new() })
            .collect::<Vec<_>>();
        for (weight, mut tbv) in claims {
            for (target, source) in combined.iter_mut().zip(&mut tbv.buckets) {
                for term in &mut source.terms {
                    term.eta *= weight;
                }
                target.terms.append(&mut source.terms);
            }
        }
        Self::new(layout, combined)
    }

    pub(crate) fn project_physical_vector(
        &self,
        vector_index: usize,
        values: &[F],
    ) -> (F, PaddingWorkCounters) {
        let bucket_index = self.layout.vector_to_bucket[vector_index];
        let bucket = &self.buckets[bucket_index];
        let mut counters = PaddingWorkCounters::default();
        let mut contribution = F::zero();
        for term in &bucket.terms {
            let folded = tensor_project_factors(values, &term.factors, &mut counters);
            counters.real_ip_field_ops += 2;
            contribution += term.eta * folded;
        }
        (contribution, counters)
    }
}

impl<F: PrimeField> TensorBasisVector<F> {
    /// Binds the exact public tensor metadata to an outer Fiat--Shamir
    /// schedule. This visits only compact TBV terms and layout metadata; it
    /// never expands physical coefficients or logical padding.
    pub fn absorb_transcript(&self, fs: &mut FiatShamir) {
        fs.push(&0x5442_5654_5241_4e53_u64);
        fs.push(&self.layout.logical_len);
        fs.push(&self.layout.vector_lengths.len());
        let offsets = self.layout.bucket_offsets();
        for vector_index in 0..self.layout.vector_lengths.len() {
            fs.push(&self.layout.vector_lengths[vector_index]);
            fs.push(&offsets[self.layout.vector_to_bucket[vector_index]]);
        }
        fs.push(&self.buckets.len());
        for bucket in &self.buckets {
            fs.push(&bucket.terms.len());
            for term in &bucket.terms {
                fs.push(&term.eta);
                fs.push(&term.factors.len());
                for (zero, one) in &term.factors {
                    fs.push(zero);
                    fs.push(one);
                }
            }
        }
    }

    /// Evaluates the public TBV at the EO folding challenges using bucket
    /// metadata only. No logical TBV or padding bucket payload is allocated.
    pub fn fold_to_scalar(&self, challenges: &[F]) -> Result<F, BvaError> {
        self.clone().fold_to_scalar_consuming(challenges)
    }

    /// Consumes the compact TBV metadata while evaluating the EO fold. This
    /// is transcript-identical to [`Self::fold_to_scalar`] and avoids keeping
    /// the original terms resident alongside a full folding clone.
    pub fn fold_to_scalar_consuming(mut self, challenges: &[F]) -> Result<F, BvaError> {
        let expected = self.layout.logical_len.ilog2() as usize;
        if challenges.len() != expected {
            return Err(BvaError::FoldChallengeCount {
                expected,
                actual: challenges.len(),
            });
        }
        // Reverse each factor list once so every consuming round can pop the
        // next factor in O(1). Removing the first factor on every round would
        // shift the remaining field elements and become quadratic in the
        // tensor dimension for production-scale TBVs.
        for bucket in &mut self.buckets {
            for term in &mut bucket.terms {
                term.factors.reverse();
            }
        }
        let mut buckets = self
            .layout
            .bucket_sizes
            .iter()
            .copied()
            .zip(self.buckets)
            .map(|(size, basis)| StreamingEoBucket {
                size,
                source: StreamingEoSource::Padding,
                basis,
            })
            .collect::<Vec<_>>();
        let mut counters = PaddingWorkCounters::default();
        for challenge in challenges {
            let challenge_inv = challenge.inverse().ok_or(BvaError::ZeroChallenge)?;
            buckets =
                streaming_eo_fold_buckets(buckets, *challenge, challenge_inv, &mut counters, true);
        }
        if buckets.len() != 1 || buckets[0].size != 1 {
            return Err(BvaError::MissingFinalValue);
        }
        Ok(buckets[0].basis.reduced_scalar())
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

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StreamingEoProfile {
    pub source_passes: usize,
    pub physical_entries_per_pass: usize,
    pub total_visited_entries: usize,
    pub maximum_live_vector_entries: usize,
    pub materialized_padding_entries: usize,
    pub padding_field_operations: u64,
}

#[derive(Debug)]
pub enum StreamingEoError<SourceError> {
    Bva(BvaError),
    Source(SourceError),
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
        }
    }
}

impl<SourceError> std::error::Error for StreamingEoError<SourceError> where
    SourceError: std::error::Error + 'static
{
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BvaInnerProduct<F> {
    pub value: F,
    pub counters: PaddingWorkCounters,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BvaFoldResult<F> {
    pub value: F,
    pub counters: PaddingWorkCounters,
}

#[derive(Clone, Debug, Eq, PartialEq, CanonicalDeserialize, CanonicalSerialize)]
pub struct EoLiteBulletProof<F: PrimeField> {
    pub rounds: Vec<(F, F)>,
    pub a_reduced: F,
    pub b_reduced: F,
}

/// Packed EO-LiteBullet for a native short-width witness and a public TBV.
/// The first `log2(packing_stride)` EO rounds share one challenge and are
/// represented by Laurent coefficients instead of binary cross terms.
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

/// EO-LiteBullet reduction for an inner product whose two operands are both
/// secret routed witness views. The terminal scalars must subsequently be
/// bound to the global witness commitment by the caller.
#[derive(Clone, Debug, Eq, PartialEq, CanonicalDeserialize, CanonicalSerialize)]
pub struct EoWitnessPairProof<F: PrimeField> {
    pub rounds: Vec<(F, F)>,
    pub a_reduced: F,
    pub b_reduced: F,
}

#[derive(Debug)]
pub enum EoProofDecodeError {
    Serialization(SerializationError),
    TrailingBytes,
}

impl Display for EoProofDecodeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serialization(error) => Display::fmt(error, formatter),
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

#[derive(Clone)]
struct EoBucket<F> {
    size: usize,
    source: Option<Vec<F>>,
    basis: TbvBucket<F>,
}

#[derive(Clone)]
struct EoPairBucket<F> {
    size: usize,
    left: Option<Vec<F>>,
    right: Option<Vec<F>>,
}

#[derive(Clone)]
enum StreamingEoSource<F> {
    Padding,
    Deferred {
        vector_index: usize,
        scalar: Option<F>,
    },
    Scalar(Option<F>),
}

#[derive(Clone)]
struct StreamingEoBucket<F> {
    size: usize,
    source: StreamingEoSource<F>,
    basis: TbvBucket<F>,
}

#[derive(Clone)]
enum StreamingPairSource<F> {
    Padding,
    Deferred {
        vector_index: usize,
        left_scalar: Option<F>,
        right_scalar: Option<F>,
    },
    Scalars(Option<(F, F)>),
}

#[derive(Clone)]
struct StreamingPairBucket<F> {
    size: usize,
    source: StreamingPairSource<F>,
}

impl<F: PrimeField> EoLiteBulletProof<F> {
    pub fn to_bytes(&self) -> Result<Vec<u8>, SerializationError> {
        let mut bytes = Vec::new();
        self.serialize_compressed(&mut bytes)?;
        Ok(bytes)
    }

    pub fn from_bytes_exact(bytes: &[u8]) -> Result<Self, EoProofDecodeError> {
        let mut reader = Cursor::new(bytes);
        let proof = Self::deserialize_compressed(&mut reader)?;
        if reader.position() != bytes.len() as u64 {
            return Err(EoProofDecodeError::TrailingBytes);
        }
        Ok(proof)
    }

    pub fn prove(
        assembly: &BucketedVector<F>,
        tbv: &TensorBasisVector<F>,
        claimed: F,
    ) -> Result<(Self, PaddingWorkCounters), BvaError> {
        Self::prove_inner(assembly, tbv, claimed, None)
    }

    /// Consumes the public tensor basis while folding it. This produces the
    /// same transcript as [`Self::prove`] without cloning a potentially large
    /// synchronized TBV before the first EO round.
    pub fn prove_consuming_tbv(
        assembly: &BucketedVector<F>,
        tbv: TensorBasisVector<F>,
        claimed: F,
    ) -> Result<(Self, PaddingWorkCounters), BvaError> {
        Self::prove_inner_consuming_tbv(assembly, tbv, claimed, None)
    }

    /// Owned, commitment-bound EO reduction. Only physical BVA payloads are
    /// cloned and folded; virtual padding buckets and short-vector suffixes
    /// remain metadata-only throughout the reduction.
    pub fn prove_bound(
        assembly: &BucketedVector<F>,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        binding: F,
    ) -> Result<(Self, PaddingWorkCounters), BvaError> {
        Self::prove_inner(assembly, tbv, claimed, Some(binding))
    }

    /// Consumes a commitment-bound TBV after the caller has already checked
    /// the exact initial inner product. `initial_counters` preserves the work
    /// accounting for that check while avoiding both a duplicate pass and a
    /// clone of the public tensor metadata.
    pub(crate) fn prove_bound_consuming_tbv_prechecked(
        assembly: &BucketedVector<F>,
        tbv: TensorBasisVector<F>,
        claimed: F,
        binding: F,
        initial_counters: PaddingWorkCounters,
    ) -> Result<(Self, PaddingWorkCounters), BvaError> {
        Self::prove_inner_consuming_tbv_prechecked(
            assembly,
            tbv,
            claimed,
            Some(binding),
            initial_counters,
        )
    }

    fn prove_inner(
        assembly: &BucketedVector<F>,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        binding: Option<F>,
    ) -> Result<(Self, PaddingWorkCounters), BvaError> {
        Self::prove_inner_consuming_tbv(assembly, tbv.clone(), claimed, binding)
    }

    fn prove_inner_consuming_tbv(
        assembly: &BucketedVector<F>,
        tbv: TensorBasisVector<F>,
        claimed: F,
        binding: Option<F>,
    ) -> Result<(Self, PaddingWorkCounters), BvaError> {
        if assembly.layout != tbv.layout {
            return Err(BvaError::LayoutMismatch);
        }
        let initial = assembly.inner_product(&tbv)?;
        if initial.value != claimed {
            return Err(BvaError::ClaimMismatch);
        }
        Self::prove_inner_consuming_tbv_prechecked(
            assembly,
            tbv,
            claimed,
            binding,
            initial.counters,
        )
    }

    fn prove_inner_consuming_tbv_prechecked(
        assembly: &BucketedVector<F>,
        tbv: TensorBasisVector<F>,
        claimed: F,
        binding: Option<F>,
        mut counters: PaddingWorkCounters,
    ) -> Result<(Self, PaddingWorkCounters), BvaError> {
        if assembly.layout != tbv.layout {
            return Err(BvaError::LayoutMismatch);
        }
        let mut sources = vec![None; assembly.layout.bucket_sizes.len()];
        for (vector_index, &bucket_index) in assembly.layout.vector_to_bucket.iter().enumerate() {
            sources[bucket_index] = Some(assembly.vectors[vector_index].clone());
        }
        let mut buckets = assembly
            .layout
            .bucket_sizes
            .iter()
            .copied()
            .zip(sources)
            .zip(tbv.buckets)
            .map(|((size, source), basis)| EoBucket {
                size,
                source,
                basis,
            })
            .collect::<Vec<_>>();
        let mut fs = eo_litebullet_fiat_shamir(assembly.layout.logical_len, claimed, binding);
        let rounds_len = assembly.layout.logical_len.ilog2() as usize;
        let mut rounds = Vec::with_capacity(rounds_len);
        for _ in 0..rounds_len {
            let (iota, rho) = eo_cross_inner_products(&buckets, &mut counters);
            rounds.push((iota, rho));
            fs.push(&iota);
            fs.push(&rho);
            let challenge = fs.gen_challenge::<F>();
            let challenge_inv = challenge.inverse().ok_or(BvaError::ZeroChallenge)?;
            buckets = eo_fold_buckets(buckets, challenge, challenge_inv, &mut counters);
        }
        if buckets.len() != 1 || buckets[0].size != 1 {
            return Err(BvaError::MissingFinalValue);
        }
        let final_bucket = buckets.pop().ok_or(BvaError::MissingFinalValue)?;
        let a_reduced = final_bucket
            .source
            .and_then(|mut values| values.pop())
            .unwrap_or_else(F::zero);
        let b_reduced = final_bucket.basis.reduced_scalar();
        Ok((
            Self {
                rounds,
                a_reduced,
                b_reduced,
            },
            counters,
        ))
    }

    /// Replays a column-streamed BVA source for each EO round. The algorithm
    /// retains at most one physical vector plus one scalar per vector whose
    /// bucket has already collapsed. It is the bounded-RAM counterpart of
    /// [`Self::prove`]; callers can later trade the extra source passes for an
    /// external fold cache without changing the transcript.
    pub fn prove_streaming<S: BvaFieldSource<F>>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        Self::prove_streaming_inner(source, tbv, claimed, None)
    }

    /// Commitment-bound streaming EO. `binding` is absorbed before the
    /// reduction challenges and must be derived from the witness commitment
    /// and the public relation domain by the caller.
    pub fn prove_streaming_bound<S: BvaFieldSource<F>>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        binding: F,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        Self::prove_streaming_inner(source, tbv, claimed, Some(binding))
    }

    fn prove_streaming_inner<S: BvaFieldSource<F>>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        binding: Option<F>,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        let layout = source.layout().clone();
        if layout != tbv.layout {
            return Err(BvaError::LayoutMismatch.into());
        }
        let (initial, mut counters, mut profile) = streaming_initial_inner_product(source, tbv)?;
        if initial != claimed {
            return Err(BvaError::ClaimMismatch.into());
        }

        let mut sources = vec![StreamingEoSource::Padding; layout.bucket_sizes.len()];
        for (vector_index, &bucket_index) in layout.vector_to_bucket.iter().enumerate() {
            sources[bucket_index] = StreamingEoSource::Deferred {
                vector_index,
                scalar: None,
            };
        }
        let mut buckets = layout
            .bucket_sizes
            .iter()
            .copied()
            .zip(sources)
            .zip(tbv.buckets.iter().cloned())
            .map(|((size, source), basis)| StreamingEoBucket {
                size,
                source,
                basis,
            })
            .collect::<Vec<_>>();
        let mut fs = eo_litebullet_fiat_shamir(layout.logical_len, claimed, binding);
        let rounds_len = layout.logical_len.ilog2() as usize;
        let mut rounds = Vec::with_capacity(rounds_len);
        let mut challenges_inv = Vec::with_capacity(rounds_len);
        for _ in 0..rounds_len {
            let (iota, rho, round_counters, round_profile) =
                streaming_eo_cross_inner_products(source, &layout, &mut buckets, &challenges_inv)?;
            accumulate_padding_counters(&mut counters, round_counters);
            accumulate_streaming_profile(&mut profile, round_profile);
            rounds.push((iota, rho));
            fs.push(&iota);
            fs.push(&rho);
            let challenge = fs.gen_challenge::<F>();
            let challenge_inv = challenge.inverse().ok_or(BvaError::ZeroChallenge)?;
            challenges_inv.push(challenge_inv);
            buckets =
                streaming_eo_fold_buckets(buckets, challenge, challenge_inv, &mut counters, false);
        }
        if buckets.len() != 1 || buckets[0].size != 1 {
            return Err(BvaError::MissingFinalValue.into());
        }
        if matches!(buckets[0].source, StreamingEoSource::Deferred { .. }) {
            let (_, _, final_counters, final_profile) =
                streaming_eo_cross_inner_products(source, &layout, &mut buckets, &challenges_inv)?;
            accumulate_padding_counters(&mut counters, final_counters);
            accumulate_streaming_profile(&mut profile, final_profile);
        }
        let final_bucket = buckets.pop().ok_or(BvaError::MissingFinalValue)?;
        let a_reduced = streaming_source_scalar(&final_bucket.source)
            .ok_or(BvaError::MissingFinalValue)?
            .unwrap_or_else(F::zero);
        Ok((
            Self {
                rounds,
                a_reduced,
                b_reduced: final_bucket.basis.reduced_scalar(),
            },
            counters,
            profile,
        ))
    }

    fn verify_scalar_identity(
        &self,
        logical_len: usize,
        claimed: F,
    ) -> EoLiteBulletVerification<F> {
        self.verify_scalar_identity_inner(logical_len, claimed, None)
    }

    /// Replays only the scalar EO identity. This is useful for a prover that
    /// has consumed a very large public TBV while producing the transcript.
    /// Verifiers must still call [`Self::verify_with_tbv`] to bind the proof
    /// to the relation coefficients.
    pub fn verify_scalar_for_prover(
        &self,
        logical_len: usize,
        claimed: F,
    ) -> EoLiteBulletVerification<F> {
        self.verify_scalar_identity(logical_len, claimed)
    }

    /// Commitment-bound counterpart of [`Self::verify_scalar_for_prover`].
    /// Verifiers must still use [`Self::verify_with_tbv_bound`].
    pub(crate) fn verify_scalar_for_prover_bound(
        &self,
        logical_len: usize,
        claimed: F,
        binding: F,
    ) -> EoLiteBulletVerification<F> {
        self.verify_scalar_identity_inner(logical_len, claimed, Some(binding))
    }

    fn verify_scalar_identity_inner(
        &self,
        logical_len: usize,
        claimed: F,
        binding: Option<F>,
    ) -> EoLiteBulletVerification<F> {
        if logical_len == 0
            || !logical_len.is_power_of_two()
            || self.rounds.len() != logical_len.ilog2() as usize
        {
            return EoLiteBulletVerification {
                accepted: false,
                challenges: Vec::new(),
                challenges_inv: Vec::new(),
            };
        }
        let mut fs = eo_litebullet_fiat_shamir(logical_len, claimed, binding);
        let mut challenges = Vec::with_capacity(self.rounds.len());
        let mut challenges_inv = Vec::with_capacity(self.rounds.len());
        let mut reduced_claim = claimed;
        for &(iota, rho) in &self.rounds {
            fs.push(&iota);
            fs.push(&rho);
            let challenge = fs.gen_challenge::<F>();
            let Some(challenge_inv) = challenge.inverse() else {
                return EoLiteBulletVerification {
                    accepted: false,
                    challenges,
                    challenges_inv,
                };
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
        let mut verification = self.verify_scalar_identity(tbv.layout.logical_len, claimed);
        verification.accepted = verification.accepted
            && tbv
                .fold_to_scalar(&verification.challenges)
                .map(|expected| expected == self.b_reduced)
                .unwrap_or(false);
        verification
    }

    /// Consuming variant of [`Self::verify_with_tbv`] for production-scale
    /// public TBVs. It preserves the scalar transcript and terminal check
    /// while eliminating a full metadata clone during verifier folding.
    pub fn verify_with_tbv_consuming(
        &self,
        tbv: TensorBasisVector<F>,
        claimed: F,
    ) -> EoLiteBulletVerification<F> {
        let mut verification = self.verify_scalar_identity(tbv.layout.logical_len, claimed);
        verification.accepted = verification.accepted
            && tbv
                .fold_to_scalar_consuming(&verification.challenges)
                .map(|expected| expected == self.b_reduced)
                .unwrap_or(false);
        verification
    }

    pub fn verify_with_tbv_bound(
        &self,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        binding: F,
    ) -> EoLiteBulletVerification<F> {
        let mut verification =
            self.verify_scalar_identity_inner(tbv.layout.logical_len, claimed, Some(binding));
        verification.accepted = verification.accepted
            && tbv
                .fold_to_scalar(&verification.challenges)
                .map(|expected| expected == self.b_reduced)
                .unwrap_or(false);
        verification
    }

    /// Commitment-bound consuming variant of
    /// [`Self::verify_with_tbv_bound`].
    pub fn verify_with_tbv_bound_consuming(
        &self,
        tbv: TensorBasisVector<F>,
        claimed: F,
        binding: F,
    ) -> EoLiteBulletVerification<F> {
        let mut verification =
            self.verify_scalar_identity_inner(tbv.layout.logical_len, claimed, Some(binding));
        verification.accepted = verification.accepted
            && tbv
                .fold_to_scalar_consuming(&verification.challenges)
                .map(|expected| expected == self.b_reduced)
                .unwrap_or(false);
        verification
    }
}

/// Chooses the largest power-of-two stride no greater than the ratio between
/// the field storage width and the physical entry width. For a 256-bit field
/// representation and 8-bit entries, this returns 32 whenever `logical_len`
/// is at least 32.
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

fn packed_i8_cross_terms<F>(
    values: &[i8],
    logical_offset: usize,
    offsets: &[usize],
    tbv: &TensorBasisVector<F>,
    stride: usize,
) -> (Vec<F>, Vec<F>, F, u64)
where
    F: PrimeField + From<i32>,
{
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
                    let left_value = F::from(i32::from(value));
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

impl<F> PackedEoLiteBulletProof<F>
where
    F: PrimeField + From<i32>,
{
    pub fn to_bytes(&self) -> Result<Vec<u8>, SerializationError> {
        let mut bytes = Vec::new();
        self.serialize_compressed(&mut bytes)?;
        Ok(bytes)
    }

    pub fn from_bytes_exact(bytes: &[u8]) -> Result<Self, EoProofDecodeError> {
        let mut reader = Cursor::new(bytes);
        let proof = Self::deserialize_compressed(&mut reader)?;
        if reader.position() != bytes.len() as u64 {
            return Err(EoProofDecodeError::TrailingBytes);
        }
        Ok(proof)
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
        Self::prove_i8_streaming_inner(source, tbv, claimed, bit_width, stride, None)
    }

    pub fn prove_i8_streaming_bound<S: BvaI8Source>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
        binding: F,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        let stride = packing_stride_for_bit_width::<F>(source.layout().logical_len, bit_width)?;
        Self::prove_i8_streaming_inner(source, tbv, claimed, bit_width, stride, Some(binding))
    }

    pub fn prove_i8_streaming_with_stride<S: BvaI8Source>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
        stride: usize,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        Self::prove_i8_streaming_inner(source, tbv, claimed, bit_width, stride, None)
    }

    fn prove_i8_streaming_inner<S: BvaI8Source>(
        source: &mut S,
        tbv: &TensorBasisVector<F>,
        claimed: F,
        bit_width: usize,
        stride: usize,
        binding: Option<F>,
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
                let (positive, negative, local_trace, operations) =
                    packed_i8_cross_terms(values, logical_offset, &offsets, tbv, stride);
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
        if let Some(error) = bva_error {
            return Err(error.into());
        }
        if visited.iter().any(|was_visited| !was_visited) {
            return Err(BvaError::MissingVector.into());
        }
        if trace != claimed {
            return Err(BvaError::ClaimMismatch.into());
        }

        let mut fs = packed_eo_litebullet_fiat_shamir(layout.logical_len, stride, claimed, binding);
        for value in &packed_positive {
            fs.push(value);
        }
        for value in &packed_negative {
            fs.push(value);
        }
        let packed_challenge = fs.gen_challenge::<F>();
        let packed_challenge_inv = packed_challenge.inverse().ok_or(BvaError::ZeroChallenge)?;
        let mut inverse_powers = Vec::with_capacity(stride);
        let mut inverse_power = F::one();
        for _ in 0..stride {
            inverse_powers.push(inverse_power);
            inverse_power *= packed_challenge_inv;
        }

        let reduced_physical_len = layout.active_len().div_ceil(stride);
        let mut reduced_values = vec![F::zero(); reduced_physical_len];
        visited.fill(false);
        bva_error = None;
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
                let logical_offset = offsets[bucket_index];
                if logical_offset % stride == 0 && values.len() % stride == 0 {
                    let reduced_start = logical_offset / stride;
                    let reduced_end = reduced_start + values.len() / stride;
                    counters.real_fold_field_ops += reduced_values[reduced_start..reduced_end]
                        .par_iter_mut()
                        .zip(values.par_chunks(stride))
                        .map(|(output, row)| {
                            let mut sum = F::zero();
                            let mut operations = 0_u64;
                            for (&value, &inverse_power) in row.iter().zip(&inverse_powers) {
                                if value == 0 {
                                    continue;
                                }
                                sum += F::from(i32::from(value)) * inverse_power;
                                operations += 2;
                            }
                            *output = sum;
                            operations
                        })
                        .sum::<u64>();
                } else {
                    for (local_index, &value) in values.iter().enumerate() {
                        if value == 0 {
                            continue;
                        }
                        let flat_index = logical_offset + local_index;
                        reduced_values[flat_index / stride] +=
                            F::from(i32::from(value)) * inverse_powers[flat_index % stride];
                        counters.real_fold_field_ops += 2;
                    }
                }
            })
            .map_err(StreamingEoError::Source)?;
        profile.source_passes += 1;
        profile.maximum_live_vector_entries = reduced_values.len();
        if let Some(error) = bva_error {
            return Err(error.into());
        }
        if visited.iter().any(|was_visited| !was_visited) {
            return Err(BvaError::MissingVector.into());
        }

        let reduced_tbv = tbv.packed_fold_to_flat(packed_challenge, stride)?;
        let mut reduced_basis = reduced_tbv
            .buckets
            .into_iter()
            .next()
            .ok_or(BvaError::MissingFinalValue)?;
        let mut reduced_claim = claimed;
        let mut positive_power = packed_challenge;
        let mut negative_power = packed_challenge_inv;
        for (&positive, &negative) in packed_positive.iter().zip(&packed_negative) {
            reduced_claim += positive * positive_power + negative * negative_power;
            positive_power *= packed_challenge;
            negative_power *= packed_challenge_inv;
        }
        let reduced_actual = reduced_basis.terms.iter().fold(F::zero(), |sum, term| {
            let (projected, _) = tensor_project_strided_allocation_free(
                &reduced_values,
                0,
                1,
                reduced_values.len(),
                &term.factors,
            );
            sum + term.eta * projected
        });
        if reduced_actual != reduced_claim {
            return Err(BvaError::ClaimMismatch.into());
        }
        let binary_round_count = (layout.logical_len / stride).ilog2() as usize;
        let mut binary_rounds = Vec::with_capacity(binary_round_count);
        for _ in 0..binary_round_count {
            let (iota, rho) = packed_eo_cross_inner_products_allocation_free(
                &reduced_values,
                &reduced_basis,
                &mut counters,
            );
            binary_rounds.push((iota, rho));
            fs.push(&iota);
            fs.push(&rho);
            let challenge = fs.gen_challenge::<F>();
            let challenge_inv = challenge.inverse().ok_or(BvaError::ZeroChallenge)?;
            packed_eo_fold_prefix_in_place_parallel(
                &mut reduced_values,
                challenge_inv,
                &mut counters,
            );
            reduced_basis = reduced_basis.fold_interior_owned(challenge);
        }
        if reduced_values.len() != 1 {
            return Err(BvaError::MissingFinalValue.into());
        }
        let a_reduced = reduced_values.pop().ok_or(BvaError::MissingFinalValue)?;
        let b_reduced = reduced_basis.reduced_scalar();
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
        let packed_challenge = fs.gen_challenge::<F>();
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
            let challenge = fs.gen_challenge::<F>();
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

impl<F: PrimeField> EoWitnessPairProof<F> {
    pub fn to_bytes(&self) -> Result<Vec<u8>, SerializationError> {
        let mut bytes = Vec::new();
        self.serialize_compressed(&mut bytes)?;
        Ok(bytes)
    }

    pub fn from_bytes_exact(bytes: &[u8]) -> Result<Self, EoProofDecodeError> {
        let mut reader = Cursor::new(bytes);
        let proof = Self::deserialize_compressed(&mut reader)?;
        if reader.position() != bytes.len() as u64 {
            return Err(EoProofDecodeError::TrailingBytes);
        }
        Ok(proof)
    }

    /// Owned, commitment-bound EO reduction for two secret BVA operands.
    /// This is transcript-equivalent to [`Self::prove_streaming_bound`] but
    /// folds each physical value once overall instead of replaying the source
    /// once per round. Virtual zero padding is never allocated or multiplied.
    pub fn prove_bound(
        left: &BucketedVector<F>,
        right: &BucketedVector<F>,
        claimed: F,
        binding: F,
    ) -> Result<(Self, PaddingWorkCounters), BvaError> {
        if left.layout != right.layout {
            return Err(BvaError::LayoutMismatch);
        }
        let mut counters = PaddingWorkCounters::default();
        let mut initial = F::zero();
        for (left_vector, right_vector) in left.vectors.iter().zip(&right.vectors) {
            if left_vector.len() != right_vector.len() {
                return Err(BvaError::VectorLength);
            }
            for (left_value, right_value) in left_vector.iter().zip(right_vector) {
                initial += *left_value * right_value;
                counters.real_ip_field_ops += 2;
            }
        }
        if initial != claimed {
            return Err(BvaError::ClaimMismatch);
        }

        let mut left_sources = vec![None; left.layout.bucket_sizes.len()];
        let mut right_sources = vec![None; right.layout.bucket_sizes.len()];
        for (vector_index, &bucket_index) in left.layout.vector_to_bucket.iter().enumerate() {
            left_sources[bucket_index] = Some(left.vectors[vector_index].clone());
            right_sources[bucket_index] = Some(right.vectors[vector_index].clone());
        }
        let mut buckets = left
            .layout
            .bucket_sizes
            .iter()
            .copied()
            .zip(left_sources)
            .zip(right_sources)
            .map(|((size, left), right)| EoPairBucket { size, left, right })
            .collect::<Vec<_>>();
        let rounds_len = left.layout.logical_len.ilog2() as usize;
        let mut rounds = Vec::with_capacity(rounds_len);
        let mut fs = eo_witness_pair_fiat_shamir(left.layout.logical_len, claimed, Some(binding));
        for _ in 0..rounds_len {
            let (iota, rho) = eo_pair_cross_inner_products(&buckets, &mut counters);
            rounds.push((iota, rho));
            fs.push(&iota);
            fs.push(&rho);
            let challenge = fs.gen_challenge::<F>();
            let challenge_inv = challenge.inverse().ok_or(BvaError::ZeroChallenge)?;
            buckets = eo_pair_fold_buckets(buckets, challenge, challenge_inv, &mut counters);
        }
        if buckets.len() != 1 || buckets[0].size != 1 {
            return Err(BvaError::MissingFinalValue);
        }
        let final_bucket = buckets.pop().ok_or(BvaError::MissingFinalValue)?;
        let a_reduced = final_bucket
            .left
            .and_then(|mut values| values.pop())
            .unwrap_or_else(F::zero);
        let b_reduced = final_bucket
            .right
            .and_then(|mut values| values.pop())
            .unwrap_or_else(F::zero);
        Ok((
            Self {
                rounds,
                a_reduced,
                b_reduced,
            },
            counters,
        ))
    }

    /// Replays a pair of column-streamed BVA views for every EO round. At most
    /// one matching physical vector pair and the collapsed bucket scalars are
    /// resident; padding buckets never acquire payloads.
    pub fn prove_streaming<S: BvaFieldPairSource<F>>(
        source: &mut S,
        claimed: F,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        Self::prove_streaming_inner(source, claimed, None)
    }

    /// The commitment-bound variant used when both secret operands are later
    /// opened against polynomial commitments. `binding` must be derived from
    /// those commitments and the public relation domain before this method is
    /// called; it is absorbed before every EO round challenge.
    pub fn prove_streaming_bound<S: BvaFieldPairSource<F>>(
        source: &mut S,
        claimed: F,
        binding: F,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        Self::prove_streaming_inner(source, claimed, Some(binding))
    }

    /// Commitment-bound EO reduction for sparse operands. Only explicitly
    /// nonzero coordinates are retained while the low-bit-first folding is
    /// replayed in a sparse map; omitted physical entries and BVA padding never
    /// become field elements or participate in arithmetic.
    pub fn prove_sparse_streaming_bound<S: SparseBvaFieldPairSource<F>>(
        source: &mut S,
        claimed: F,
        binding: F,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        let layout = source.layout().clone();
        let offsets = layout.bucket_offsets();
        let mut left = BTreeMap::<usize, F>::new();
        let mut right = BTreeMap::<usize, F>::new();
        let mut seen = BTreeMap::<(usize, usize), ()>::new();
        let mut bva_error = None;
        source
            .for_each_nonzero_pair(&mut |vector_index, entry_index, left_value, right_value| {
                if bva_error.is_some() {
                    return;
                }
                let Some(&bucket_index) = layout.vector_to_bucket.get(vector_index) else {
                    bva_error = Some(BvaError::VectorIndex);
                    return;
                };
                if entry_index >= layout.vector_lengths[vector_index] {
                    bva_error = Some(BvaError::VectorIndex);
                    return;
                }
                if seen.insert((vector_index, entry_index), ()).is_some() {
                    bva_error = Some(BvaError::DuplicateVector);
                    return;
                }
                let flat_index = offsets[bucket_index] + entry_index;
                if !left_value.is_zero() {
                    left.insert(flat_index, left_value);
                }
                if !right_value.is_zero() {
                    right.insert(flat_index, right_value);
                }
            })
            .map_err(StreamingEoError::Source)?;
        if let Some(error) = bva_error {
            return Err(error.into());
        }

        let mut counters = PaddingWorkCounters::default();
        let initial = sparse_same_index_inner_product(&left, &right, &mut counters);
        if initial != claimed {
            return Err(BvaError::ClaimMismatch.into());
        }
        let mut profile = StreamingEoProfile {
            source_passes: 1,
            physical_entries_per_pass: seen.len(),
            total_visited_entries: seen.len(),
            maximum_live_vector_entries: left.len() + right.len(),
            ..Default::default()
        };
        let mut fs = eo_witness_pair_fiat_shamir(layout.logical_len, claimed, Some(binding));
        let mut rounds = Vec::with_capacity(layout.logical_len.ilog2() as usize);
        let mut logical_len = layout.logical_len;
        while logical_len > 1 {
            let (iota, rho) = sparse_pair_cross_inner_products(&left, &right, &mut counters);
            rounds.push((iota, rho));
            fs.push(&iota);
            fs.push(&rho);
            let challenge = fs.gen_challenge::<F>();
            let challenge_inv = challenge.inverse().ok_or(BvaError::ZeroChallenge)?;
            left = sparse_fold_map(left, challenge_inv, &mut counters);
            right = sparse_fold_map(right, challenge, &mut counters);
            profile.maximum_live_vector_entries = profile
                .maximum_live_vector_entries
                .max(left.len() + right.len());
            logical_len /= 2;
        }
        let a_reduced = left.get(&0).copied().unwrap_or_else(F::zero);
        let b_reduced = right.get(&0).copied().unwrap_or_else(F::zero);
        Ok((
            Self {
                rounds,
                a_reduced,
                b_reduced,
            },
            counters,
            profile,
        ))
    }

    fn prove_streaming_inner<S: BvaFieldPairSource<F>>(
        source: &mut S,
        claimed: F,
        binding: Option<F>,
    ) -> Result<(Self, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>> {
        let layout = source.layout().clone();
        let (initial, mut counters, mut profile) = streaming_pair_inner_product(source)?;
        if initial != claimed {
            return Err(BvaError::ClaimMismatch.into());
        }

        let mut sources = vec![StreamingPairSource::Padding; layout.bucket_sizes.len()];
        for (vector_index, &bucket_index) in layout.vector_to_bucket.iter().enumerate() {
            sources[bucket_index] = StreamingPairSource::Deferred {
                vector_index,
                left_scalar: None,
                right_scalar: None,
            };
        }
        let mut buckets = layout
            .bucket_sizes
            .iter()
            .copied()
            .zip(sources)
            .map(|(size, source)| StreamingPairBucket { size, source })
            .collect::<Vec<_>>();
        let rounds_len = layout.logical_len.ilog2() as usize;
        let mut rounds = Vec::with_capacity(rounds_len);
        let mut challenges = Vec::with_capacity(rounds_len);
        let mut challenges_inv = Vec::with_capacity(rounds_len);
        let mut fs = eo_witness_pair_fiat_shamir(layout.logical_len, claimed, binding);
        for _ in 0..rounds_len {
            let (iota, rho, round_counters, round_profile) = streaming_pair_cross_inner_products(
                source,
                &layout,
                &mut buckets,
                &challenges,
                &challenges_inv,
            )?;
            accumulate_padding_counters(&mut counters, round_counters);
            accumulate_streaming_profile(&mut profile, round_profile);
            rounds.push((iota, rho));
            fs.push(&iota);
            fs.push(&rho);
            let challenge = fs.gen_challenge::<F>();
            let challenge_inv = challenge.inverse().ok_or(BvaError::ZeroChallenge)?;
            challenges.push(challenge);
            challenges_inv.push(challenge_inv);
            buckets = streaming_pair_fold_buckets(buckets, challenge, challenge_inv, &mut counters);
        }
        if buckets.len() != 1 || buckets[0].size != 1 {
            return Err(BvaError::MissingFinalValue.into());
        }
        if matches!(buckets[0].source, StreamingPairSource::Deferred { .. }) {
            let (_, _, final_counters, final_profile) = streaming_pair_cross_inner_products(
                source,
                &layout,
                &mut buckets,
                &challenges,
                &challenges_inv,
            )?;
            accumulate_padding_counters(&mut counters, final_counters);
            accumulate_streaming_profile(&mut profile, final_profile);
        }
        let final_bucket = buckets.pop().ok_or(BvaError::MissingFinalValue)?;
        let (a_reduced, b_reduced) = streaming_pair_scalars(&final_bucket.source)
            .ok_or(BvaError::MissingFinalValue)?
            .unwrap_or_else(|| (F::zero(), F::zero()));
        Ok((
            Self {
                rounds,
                a_reduced,
                b_reduced,
            },
            counters,
            profile,
        ))
    }

    pub fn verify(&self, logical_len: usize, claimed: F) -> EoLiteBulletVerification<F> {
        self.verify_inner(logical_len, claimed, None)
    }

    pub fn verify_bound(
        &self,
        logical_len: usize,
        claimed: F,
        binding: F,
    ) -> EoLiteBulletVerification<F> {
        self.verify_inner(logical_len, claimed, Some(binding))
    }

    fn verify_inner(
        &self,
        logical_len: usize,
        claimed: F,
        binding: Option<F>,
    ) -> EoLiteBulletVerification<F> {
        if logical_len == 0
            || !logical_len.is_power_of_two()
            || self.rounds.len() != logical_len.ilog2() as usize
        {
            return EoLiteBulletVerification {
                accepted: false,
                challenges: Vec::new(),
                challenges_inv: Vec::new(),
            };
        }
        let mut fs = eo_witness_pair_fiat_shamir(logical_len, claimed, binding);
        let mut challenges = Vec::with_capacity(self.rounds.len());
        let mut challenges_inv = Vec::with_capacity(self.rounds.len());
        let mut reduced_claim = claimed;
        for &(iota, rho) in &self.rounds {
            fs.push(&iota);
            fs.push(&rho);
            let challenge = fs.gen_challenge::<F>();
            let Some(challenge_inv) = challenge.inverse() else {
                return EoLiteBulletVerification {
                    accepted: false,
                    challenges,
                    challenges_inv,
                };
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
}

fn sparse_same_index_inner_product<F: PrimeField>(
    left: &BTreeMap<usize, F>,
    right: &BTreeMap<usize, F>,
    counters: &mut PaddingWorkCounters,
) -> F {
    left.iter().fold(F::zero(), |sum, (index, left_value)| {
        if let Some(right_value) = right.get(index) {
            counters.real_ip_field_ops += 2;
            sum + *left_value * right_value
        } else {
            sum
        }
    })
}

fn sparse_pair_cross_inner_products<F: PrimeField>(
    left: &BTreeMap<usize, F>,
    right: &BTreeMap<usize, F>,
    counters: &mut PaddingWorkCounters,
) -> (F, F) {
    let mut iota = F::zero();
    let mut rho = F::zero();
    for (&index, left_value) in left {
        if index & 1 == 0 {
            if let Some(right_value) = right.get(&(index + 1)) {
                iota += *left_value * right_value;
                counters.real_ip_field_ops += 2;
            }
        } else if let Some(right_value) = right.get(&(index - 1)) {
            rho += *left_value * right_value;
            counters.real_ip_field_ops += 2;
        }
    }
    (iota, rho)
}

fn sparse_fold_map<F: PrimeField>(
    values: BTreeMap<usize, F>,
    odd_challenge: F,
    counters: &mut PaddingWorkCounters,
) -> BTreeMap<usize, F> {
    let mut folded = BTreeMap::<usize, F>::new();
    for (index, value) in values {
        let contribution = if index & 1 == 0 {
            value
        } else {
            counters.real_fold_field_ops += 1;
            odd_challenge * value
        };
        let entry = folded.entry(index / 2).or_insert_with(F::zero);
        if !entry.is_zero() {
            counters.real_fold_field_ops += 1;
        }
        *entry += contribution;
    }
    folded.retain(|_, value| !value.is_zero());
    folded
}

fn streaming_pair_inner_product<F, S>(
    source: &mut S,
) -> Result<(F, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>>
where
    F: PrimeField,
    S: BvaFieldPairSource<F>,
{
    let layout = source.layout().clone();
    let mut visited = vec![false; layout.vector_to_bucket.len()];
    let mut value = F::zero();
    let mut counters = PaddingWorkCounters::default();
    let mut profile = StreamingEoProfile {
        source_passes: 1,
        ..Default::default()
    };
    let mut bva_error = None;
    source
        .for_each_vector_pair(&mut |vector_index, left, right| {
            if bva_error.is_some() {
                return;
            }
            if vector_index >= layout.vector_to_bucket.len() {
                bva_error = Some(BvaError::VectorIndex);
                return;
            }
            if visited[vector_index] {
                bva_error = Some(BvaError::DuplicateVector);
                return;
            }
            visited[vector_index] = true;
            let expected = layout.vector_lengths[vector_index];
            if left.len() != expected || right.len() != expected {
                bva_error = Some(BvaError::VectorLength);
                return;
            }
            for (left, right) in left.iter().zip(right) {
                value += *left * *right;
                counters.real_ip_field_ops += 2;
            }
            profile.physical_entries_per_pass += expected;
            profile.total_visited_entries += expected;
            profile.maximum_live_vector_entries = profile.maximum_live_vector_entries.max(expected);
        })
        .map_err(StreamingEoError::Source)?;
    if let Some(error) = bva_error {
        return Err(error.into());
    }
    if visited.iter().any(|visited| !visited) {
        return Err(BvaError::MissingVector.into());
    }
    Ok((value, counters, profile))
}

fn streaming_pair_cross_inner_products<F, S>(
    source: &mut S,
    layout: &BvaLayout,
    buckets: &mut [StreamingPairBucket<F>],
    challenges: &[F],
    challenges_inv: &[F],
) -> Result<(F, F, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>>
where
    F: PrimeField,
    S: BvaFieldPairSource<F>,
{
    let mut vector_to_state = vec![None; layout.vector_to_bucket.len()];
    for (state_index, bucket) in buckets.iter().enumerate() {
        if let StreamingPairSource::Deferred { vector_index, .. } = bucket.source {
            vector_to_state[vector_index] = Some(state_index);
        }
    }
    let mut visited = vec![false; layout.vector_to_bucket.len()];
    let mut iota = F::zero();
    let mut rho = F::zero();
    let mut counters = PaddingWorkCounters::default();
    let mut profile = StreamingEoProfile {
        source_passes: 1,
        ..Default::default()
    };
    let mut bva_error = None;
    source
        .for_each_vector_pair(&mut |vector_index, left, right| {
            if bva_error.is_some() {
                return;
            }
            let Some(&original_bucket_index) = layout.vector_to_bucket.get(vector_index) else {
                bva_error = Some(BvaError::VectorIndex);
                return;
            };
            if visited[vector_index] {
                bva_error = Some(BvaError::DuplicateVector);
                return;
            }
            visited[vector_index] = true;
            let expected = layout.vector_lengths[vector_index];
            if left.len() != expected || right.len() != expected {
                bva_error = Some(BvaError::VectorLength);
                return;
            }
            profile.physical_entries_per_pass += expected;
            profile.total_visited_entries += expected;
            profile.maximum_live_vector_entries = profile.maximum_live_vector_entries.max(expected);
            let Some(state_index) = vector_to_state[vector_index] else {
                return;
            };
            let state = &mut buckets[state_index];
            let original_log = layout.bucket_sizes[original_bucket_index].ilog2() as usize;
            let current_log = state.size.ilog2() as usize;
            let rounds_applied = original_log - current_log;
            if rounds_applied > challenges.len() || rounds_applied > challenges_inv.len() {
                bva_error = Some(BvaError::FoldChallengeCount {
                    expected: rounds_applied,
                    actual: challenges.len().min(challenges_inv.len()),
                });
                return;
            }
            let mut folded_left = left.to_vec();
            let mut folded_right = right.to_vec();
            for challenge in &challenges_inv[..rounds_applied] {
                eo_fold_prefix_in_place(&mut folded_left, *challenge, &mut counters);
            }
            for challenge in &challenges[..rounds_applied] {
                eo_fold_prefix_in_place(&mut folded_right, *challenge, &mut counters);
            }
            match &mut state.source {
                StreamingPairSource::Deferred {
                    left_scalar,
                    right_scalar,
                    ..
                } if state.size == 1 => {
                    *left_scalar = folded_left.first().copied();
                    *right_scalar = folded_right.first().copied();
                }
                StreamingPairSource::Deferred { .. } => {
                    iota += strided_inner_product(&folded_left, 0, &folded_right, 1, &mut counters);
                    rho += strided_inner_product(&folded_left, 1, &folded_right, 0, &mut counters);
                }
                StreamingPairSource::Padding | StreamingPairSource::Scalars(_) => {
                    bva_error = Some(BvaError::VectorIndex);
                }
            }
        })
        .map_err(StreamingEoError::Source)?;
    if let Some(error) = bva_error {
        return Err(error.into());
    }
    if visited.iter().any(|visited| !visited) {
        return Err(BvaError::MissingVector.into());
    }

    let singleton_start = buckets
        .iter()
        .position(|bucket| bucket.size == 1)
        .unwrap_or(buckets.len());
    for pair in buckets[singleton_start..].chunks_exact(2) {
        let left = streaming_pair_scalars(&pair[0].source).ok_or(BvaError::MissingFinalValue)?;
        let right = streaming_pair_scalars(&pair[1].source).ok_or(BvaError::MissingFinalValue)?;
        if let (Some((left_a, left_b)), Some((right_a, right_b))) = (left, right) {
            iota += left_a * right_b;
            rho += right_a * left_b;
            counters.real_ip_field_ops += 4;
        }
    }
    Ok((iota, rho, counters, profile))
}

fn strided_inner_product<F: Field>(
    left: &[F],
    left_parity: usize,
    right: &[F],
    right_parity: usize,
    counters: &mut PaddingWorkCounters,
) -> F {
    let mut value = F::zero();
    for (left, right) in left
        .iter()
        .skip(left_parity)
        .step_by(2)
        .zip(right.iter().skip(right_parity).step_by(2))
    {
        value += *left * *right;
        counters.real_ip_field_ops += 2;
    }
    value
}

fn streaming_pair_fold_buckets<F: PrimeField>(
    mut buckets: Vec<StreamingPairBucket<F>>,
    challenge: F,
    challenge_inv: F,
    counters: &mut PaddingWorkCounters,
) -> Vec<StreamingPairBucket<F>> {
    let singleton_start = buckets
        .iter()
        .position(|bucket| bucket.size == 1)
        .unwrap_or(buckets.len());
    let singleton_count = buckets.len() - singleton_start;
    let mut folded = Vec::with_capacity(singleton_start + singleton_count.div_ceil(2));
    for bucket in &mut buckets[..singleton_start] {
        folded.push(StreamingPairBucket {
            size: bucket.size / 2,
            source: std::mem::replace(&mut bucket.source, StreamingPairSource::Padding),
        });
    }
    for pair in buckets[singleton_start..].chunks_exact_mut(2) {
        let left = streaming_pair_scalars(&pair[0].source).unwrap_or(None);
        let right = streaming_pair_scalars(&pair[1].source).unwrap_or(None);
        let values = match (left, right) {
            (None, None) => None,
            (Some(values), None) => Some(values),
            (None, Some((right_a, right_b))) => {
                counters.real_fold_field_ops += 2;
                Some((challenge_inv * right_a, challenge * right_b))
            }
            (Some((left_a, left_b)), Some((right_a, right_b))) => {
                counters.real_fold_field_ops += 4;
                Some((
                    left_a + challenge_inv * right_a,
                    left_b + challenge * right_b,
                ))
            }
        };
        folded.push(StreamingPairBucket {
            size: 1,
            source: StreamingPairSource::Scalars(values),
        });
    }
    if singleton_count % 2 == 1 {
        let last = buckets
            .last_mut()
            .expect("odd singleton suffix contains a final bucket");
        folded.push(StreamingPairBucket {
            size: 1,
            source: std::mem::replace(&mut last.source, StreamingPairSource::Padding),
        });
    }
    folded
}

fn streaming_pair_scalars<F: Copy>(source: &StreamingPairSource<F>) -> Option<Option<(F, F)>> {
    match source {
        StreamingPairSource::Padding => Some(None),
        StreamingPairSource::Deferred {
            left_scalar,
            right_scalar,
            ..
        } => match (*left_scalar, *right_scalar) {
            (Some(left), Some(right)) => Some(Some((left, right))),
            (None, None) => None,
            _ => None,
        },
        StreamingPairSource::Scalars(values) => Some(*values),
    }
}

fn streaming_initial_inner_product<F, S>(
    source: &mut S,
    tbv: &TensorBasisVector<F>,
) -> Result<(F, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>>
where
    F: PrimeField,
    S: BvaFieldSource<F>,
{
    let layout = source.layout().clone();
    let mut visited = vec![false; layout.vector_to_bucket.len()];
    let mut value = F::zero();
    let mut counters = PaddingWorkCounters::default();
    let mut profile = StreamingEoProfile {
        source_passes: 1,
        ..Default::default()
    };
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
            debug_assert_eq!(bucket_index, layout.vector_to_bucket[vector_index]);
            let (contribution, vector_counters) = tbv.project_physical_vector(vector_index, values);
            value += contribution;
            accumulate_padding_counters(&mut counters, vector_counters);
            profile.physical_entries_per_pass += values.len();
            profile.total_visited_entries += values.len();
            profile.maximum_live_vector_entries =
                profile.maximum_live_vector_entries.max(values.len());
        })
        .map_err(StreamingEoError::Source)?;
    if let Some(error) = bva_error {
        return Err(error.into());
    }
    if visited.iter().any(|visited| !visited) {
        return Err(BvaError::MissingVector.into());
    }
    Ok((value, counters, profile))
}

fn streaming_eo_cross_inner_products<F, S>(
    source: &mut S,
    layout: &BvaLayout,
    buckets: &mut [StreamingEoBucket<F>],
    challenges_inv: &[F],
) -> Result<(F, F, PaddingWorkCounters, StreamingEoProfile), StreamingEoError<S::Error>>
where
    F: PrimeField,
    S: BvaFieldSource<F>,
{
    let mut vector_to_state = vec![None; layout.vector_to_bucket.len()];
    for (state_index, bucket) in buckets.iter().enumerate() {
        if let StreamingEoSource::Deferred { vector_index, .. } = bucket.source {
            vector_to_state[vector_index] = Some(state_index);
        }
    }
    let mut visited = vec![false; layout.vector_to_bucket.len()];
    let mut iota = F::zero();
    let mut rho = F::zero();
    let mut counters = PaddingWorkCounters::default();
    let mut profile = StreamingEoProfile {
        source_passes: 1,
        ..Default::default()
    };
    let mut bva_error = None;
    source
        .for_each_vector(&mut |vector_index, values| {
            if bva_error.is_some() {
                return;
            }
            let Some(&original_bucket_index) = layout.vector_to_bucket.get(vector_index) else {
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
            profile.maximum_live_vector_entries =
                profile.maximum_live_vector_entries.max(values.len());
            let Some(state_index) = vector_to_state[vector_index] else {
                return;
            };
            let state = &mut buckets[state_index];
            let original_log = layout.bucket_sizes[original_bucket_index].ilog2() as usize;
            let current_log = state.size.ilog2() as usize;
            let rounds_applied = original_log - current_log;
            if rounds_applied > challenges_inv.len() {
                bva_error = Some(BvaError::FoldChallengeCount {
                    expected: rounds_applied,
                    actual: challenges_inv.len(),
                });
                return;
            }
            let mut folded = values.to_vec();
            for challenge in &challenges_inv[..rounds_applied] {
                eo_fold_prefix_in_place(&mut folded, *challenge, &mut counters);
            }
            match &mut state.source {
                StreamingEoSource::Deferred { scalar, .. } if state.size == 1 => {
                    debug_assert_eq!(folded.len(), 1);
                    *scalar = folded.first().copied();
                }
                StreamingEoSource::Deferred { .. } => {
                    for term in &state.basis.terms {
                        let Some(((left_factor, right_factor), tail)) = term.factors.split_first()
                        else {
                            bva_error = Some(BvaError::ChallengeCount {
                                bucket: original_bucket_index,
                                expected: 1,
                                actual: 0,
                            });
                            return;
                        };
                        let even = tensor_project_strided(&folded, 0, tail, &mut counters);
                        let odd = tensor_project_strided(&folded, 1, tail, &mut counters);
                        iota += even * term.eta * right_factor;
                        rho += odd * term.eta * left_factor;
                        counters.real_ip_field_ops += 5;
                    }
                }
                StreamingEoSource::Padding | StreamingEoSource::Scalar(_) => {
                    bva_error = Some(BvaError::VectorIndex);
                }
            }
        })
        .map_err(StreamingEoError::Source)?;
    if let Some(error) = bva_error {
        return Err(error.into());
    }
    if visited.iter().any(|visited| !visited) {
        return Err(BvaError::MissingVector.into());
    }

    let singleton_start = buckets
        .iter()
        .position(|bucket| bucket.size == 1)
        .unwrap_or(buckets.len());
    for pair in buckets[singleton_start..].chunks_exact(2) {
        let left = streaming_source_scalar(&pair[0].source).ok_or(BvaError::MissingFinalValue)?;
        let right = streaming_source_scalar(&pair[1].source).ok_or(BvaError::MissingFinalValue)?;
        if let Some(left) = left {
            iota += left * pair[1].basis.reduced_scalar();
            counters.real_ip_field_ops += 2;
        }
        if let Some(right) = right {
            rho += right * pair[0].basis.reduced_scalar();
            counters.real_ip_field_ops += 2;
        }
    }
    Ok((iota, rho, counters, profile))
}

fn tensor_project_strided<F: Field>(
    values: &[F],
    parity: usize,
    factors: &[(F, F)],
    counters: &mut PaddingWorkCounters,
) -> F {
    let projected = values
        .iter()
        .skip(parity)
        .step_by(2)
        .copied()
        .collect::<Vec<_>>();
    tensor_project_factors(&projected, factors, counters)
}

fn streaming_eo_fold_buckets<F: PrimeField>(
    buckets: Vec<StreamingEoBucket<F>>,
    challenge: F,
    challenge_inv: F,
    counters: &mut PaddingWorkCounters,
    factors_reversed: bool,
) -> Vec<StreamingEoBucket<F>> {
    let singleton_start = buckets
        .iter()
        .position(|bucket| bucket.size == 1)
        .unwrap_or(buckets.len());
    let singleton_count = buckets.len() - singleton_start;
    let mut folded = Vec::with_capacity(singleton_start + singleton_count.div_ceil(2));
    let mut buckets = buckets.into_iter();
    for _ in 0..singleton_start {
        let bucket = buckets.next().expect("counted non-singleton bucket");
        let basis = if factors_reversed {
            bucket.basis.fold_interior_reversed_owned(challenge)
        } else {
            bucket.basis.fold_interior_owned(challenge)
        };
        folded.push(StreamingEoBucket {
            size: bucket.size / 2,
            source: bucket.source,
            basis,
        });
    }
    for _ in 0..singleton_count / 2 {
        let left_bucket = buckets.next().expect("counted singleton bucket");
        let right_bucket = buckets.next().expect("counted singleton bucket");
        let left = streaming_source_scalar(&left_bucket.source).unwrap_or(None);
        let right = streaming_source_scalar(&right_bucket.source).unwrap_or(None);
        let source = match (left, right) {
            (None, None) => None,
            (Some(left), None) => Some(left),
            (None, Some(right)) => {
                counters.real_fold_field_ops += 1;
                Some(challenge_inv * right)
            }
            (Some(left), Some(right)) => {
                counters.real_fold_field_ops += 2;
                Some(left + challenge_inv * right)
            }
        };
        folded.push(StreamingEoBucket {
            size: 1,
            source: StreamingEoSource::Scalar(source),
            basis: TbvBucket::new(
                left_bucket.basis.reduced_scalar()
                    + challenge * right_bucket.basis.reduced_scalar(),
                Vec::new(),
            ),
        });
    }
    if let Some(last) = buckets.next() {
        debug_assert_eq!(last.size, 1);
        folded.push(last);
    }
    debug_assert!(buckets.next().is_none());
    folded
}

fn streaming_source_scalar<F: Copy>(source: &StreamingEoSource<F>) -> Option<Option<F>> {
    match source {
        StreamingEoSource::Padding => Some(None),
        StreamingEoSource::Deferred { scalar, .. } => scalar.map(Some),
        StreamingEoSource::Scalar(value) => Some(*value),
    }
}

fn accumulate_padding_counters(total: &mut PaddingWorkCounters, current: PaddingWorkCounters) {
    total.materialized_padding_entries += current.materialized_padding_entries;
    total.padding_ip_field_ops += current.padding_ip_field_ops;
    total.padding_fold_field_ops += current.padding_fold_field_ops;
    total.real_ip_field_ops += current.real_ip_field_ops;
    total.real_fold_field_ops += current.real_fold_field_ops;
}

fn accumulate_streaming_profile(total: &mut StreamingEoProfile, current: StreamingEoProfile) {
    total.source_passes += current.source_passes;
    total.physical_entries_per_pass = total
        .physical_entries_per_pass
        .max(current.physical_entries_per_pass);
    total.total_visited_entries += current.total_visited_entries;
    total.maximum_live_vector_entries = total
        .maximum_live_vector_entries
        .max(current.maximum_live_vector_entries);
    total.materialized_padding_entries += current.materialized_padding_entries;
    total.padding_field_operations += current.padding_field_operations;
}

fn eo_litebullet_fiat_shamir<F: PrimeField>(
    logical_len: usize,
    claimed: F,
    binding: Option<F>,
) -> FiatShamir {
    let mut fs = FiatShamir::new();
    fs.push(&0x454f_4c42_u64);
    fs.push(&logical_len);
    fs.push(&claimed);
    if let Some(binding) = binding {
        fs.push(&0x424f_554e_44_u64);
        fs.push(&binding);
    }
    fs
}

fn packed_eo_litebullet_fiat_shamir<F: PrimeField>(
    logical_len: usize,
    packing_stride: usize,
    claimed: F,
    binding: Option<F>,
) -> FiatShamir {
    let mut fs = FiatShamir::new();
    fs.push(&0x5041_434b_4544_454f_u64);
    fs.push(&logical_len);
    fs.push(&packing_stride);
    fs.push(&claimed);
    if let Some(binding) = binding {
        fs.push(&0x424f_554e_44_u64);
        fs.push(&binding);
    }
    fs
}

fn eo_witness_pair_fiat_shamir<F: PrimeField>(
    logical_len: usize,
    claimed: F,
    binding: Option<F>,
) -> FiatShamir {
    let mut fs = FiatShamir::new();
    fs.push(&0x454f_5750_u64);
    fs.push(&logical_len);
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

fn packed_eo_fold_prefix_in_place_parallel<F: Field>(
    values: &mut Vec<F>,
    challenge: F,
    counters: &mut PaddingWorkCounters,
) {
    let old_len = values.len();
    if old_len <= 1 {
        return;
    }
    let first_left = values[0];
    values[0] = first_left + challenge * values[1];
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

fn eo_cross_inner_products<F: PrimeField>(
    buckets: &[EoBucket<F>],
    counters: &mut PaddingWorkCounters,
) -> (F, F) {
    let singleton_start = buckets
        .iter()
        .position(|bucket| bucket.size == 1)
        .unwrap_or(buckets.len());
    let mut iota = F::zero();
    let mut rho = F::zero();
    for pair in buckets[singleton_start..].chunks_exact(2) {
        if let Some(left) = pair[0].source.as_ref().and_then(|values| values.first()) {
            iota += *left * pair[1].basis.reduced_scalar();
            counters.real_ip_field_ops += 2;
        }
        if let Some(right) = pair[1].source.as_ref().and_then(|values| values.first()) {
            rho += *right * pair[0].basis.reduced_scalar();
            counters.real_ip_field_ops += 2;
        }
    }
    for bucket in &buckets[..singleton_start] {
        debug_assert!(bucket.size >= 2);
        let Some(source) = &bucket.source else {
            continue;
        };
        let even = source.iter().step_by(2).copied().collect::<Vec<_>>();
        let odd = source
            .iter()
            .skip(1)
            .step_by(2)
            .copied()
            .collect::<Vec<_>>();
        for term in &bucket.basis.terms {
            let ((left_factor, right_factor), tail) = term.factors.split_first().unwrap();
            if !even.is_empty() {
                let projected = tensor_project_factors(&even, tail, counters);
                iota += projected * term.eta * right_factor;
                counters.real_ip_field_ops += 3;
            }
            if !odd.is_empty() {
                let projected = tensor_project_factors(&odd, tail, counters);
                rho += projected * term.eta * left_factor;
                counters.real_ip_field_ops += 2;
            }
        }
    }
    (iota, rho)
}

fn tensor_project_factors<F: Field>(
    values: &[F],
    factors: &[(F, F)],
    counters: &mut PaddingWorkCounters,
) -> F {
    if factors
        .iter()
        .any(|(left, right)| left.is_zero() && right.is_zero())
    {
        return F::zero();
    }

    // Routed lookup slices encode their aligned dyadic support with a suffix
    // of one-hot high-bit factors. Restrict to that real support before
    // folding; arbitrary tensor terms simply fall back to the full vector.
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

    let support_len = 1_usize << support_log;
    if support_start >= values.len() {
        return F::zero();
    }
    let support_end = support_start.saturating_add(support_len).min(values.len());
    let mut folded = values[support_start..support_end].to_vec();
    for (left_factor, right_factor) in &factors[..support_log] {
        let next_len = folded.len().div_ceil(2);
        for output_index in 0..next_len {
            let left_index = output_index * 2;
            let right_index = left_index + 1;
            let left = folded[left_index] * left_factor;
            folded[output_index] = if right_index < folded.len() {
                counters.real_fold_field_ops += 3;
                left + folded[right_index] * right_factor
            } else {
                counters.real_fold_field_ops += 1;
                left
            };
        }
        folded.truncate(next_len);
    }
    debug_assert_eq!(folded.len(), 1);
    if support_log == factors.len() {
        return folded[0];
    }

    let mut fixed_product = F::one();
    for (left, right) in &factors[support_log..] {
        fixed_product *= if left.is_zero() { *right } else { *left };
        counters.real_fold_field_ops += 1;
    }
    counters.real_fold_field_ops += 1;
    folded[0] * fixed_product
}

fn eo_fold_buckets<F: PrimeField>(
    buckets: Vec<EoBucket<F>>,
    challenge: F,
    challenge_inv: F,
    counters: &mut PaddingWorkCounters,
) -> Vec<EoBucket<F>> {
    let singleton_start = buckets
        .iter()
        .position(|bucket| bucket.size == 1)
        .unwrap_or(buckets.len());
    let singleton_count = buckets.len() - singleton_start;
    let mut folded = Vec::with_capacity(singleton_start + singleton_count.div_ceil(2));
    let mut buckets = buckets.into_iter();
    for _ in 0..singleton_start {
        let mut bucket = buckets.next().expect("counted non-singleton bucket");
        if let Some(source) = &mut bucket.source {
            eo_fold_prefix_in_place(source, challenge_inv, counters);
        }
        folded.push(EoBucket {
            size: bucket.size / 2,
            source: bucket.source,
            basis: bucket.basis.fold_interior_owned(challenge),
        });
    }
    for _ in 0..singleton_count / 2 {
        let left = buckets.next().expect("counted singleton bucket");
        let right = buckets.next().expect("counted singleton bucket");
        let source = match (left.source, right.source) {
            (None, None) => None,
            (Some(left), None) => Some(left),
            (None, Some(mut right)) => {
                right[0] *= challenge_inv;
                counters.real_fold_field_ops += 1;
                Some(right)
            }
            (Some(mut left), Some(right)) => {
                left[0] += challenge_inv * right[0];
                counters.real_fold_field_ops += 2;
                Some(left)
            }
        };
        folded.push(EoBucket {
            size: 1,
            source,
            basis: TbvBucket::new(
                left.basis.reduced_scalar() + challenge * right.basis.reduced_scalar(),
                Vec::new(),
            ),
        });
    }
    if let Some(last) = buckets.next() {
        debug_assert_eq!(last.size, 1);
        folded.push(last);
    }
    debug_assert!(buckets.next().is_none());
    folded
}

fn eo_pair_cross_inner_products<F: PrimeField>(
    buckets: &[EoPairBucket<F>],
    counters: &mut PaddingWorkCounters,
) -> (F, F) {
    let singleton_start = buckets
        .iter()
        .position(|bucket| bucket.size == 1)
        .unwrap_or(buckets.len());
    let mut iota = F::zero();
    let mut rho = F::zero();
    for pair in buckets[singleton_start..].chunks_exact(2) {
        if let (Some(left), Some(right)) = (
            pair[0].left.as_ref().and_then(|values| values.first()),
            pair[1].right.as_ref().and_then(|values| values.first()),
        ) {
            iota += *left * right;
            counters.real_ip_field_ops += 2;
        }
        if let (Some(left), Some(right)) = (
            pair[1].left.as_ref().and_then(|values| values.first()),
            pair[0].right.as_ref().and_then(|values| values.first()),
        ) {
            rho += *left * right;
            counters.real_ip_field_ops += 2;
        }
    }
    for bucket in &buckets[..singleton_start] {
        debug_assert!(bucket.size >= 2);
        let (Some(left), Some(right)) = (&bucket.left, &bucket.right) else {
            continue;
        };
        for index in (0..left.len()).step_by(2) {
            if index + 1 < right.len() {
                iota += left[index] * right[index + 1];
                counters.real_ip_field_ops += 2;
            }
            if index + 1 < left.len() && index < right.len() {
                rho += left[index + 1] * right[index];
                counters.real_ip_field_ops += 2;
            }
        }
    }
    (iota, rho)
}

fn eo_pair_fold_buckets<F: PrimeField>(
    mut buckets: Vec<EoPairBucket<F>>,
    challenge: F,
    challenge_inv: F,
    counters: &mut PaddingWorkCounters,
) -> Vec<EoPairBucket<F>> {
    let singleton_start = buckets
        .iter()
        .position(|bucket| bucket.size == 1)
        .unwrap_or(buckets.len());
    let singleton_count = buckets.len() - singleton_start;
    let mut folded = Vec::with_capacity(singleton_start + singleton_count.div_ceil(2));
    for bucket in &mut buckets[..singleton_start] {
        if let Some(left) = &mut bucket.left {
            eo_fold_prefix_in_place(left, challenge_inv, counters);
        }
        if let Some(right) = &mut bucket.right {
            eo_fold_prefix_in_place(right, challenge, counters);
        }
        folded.push(EoPairBucket {
            size: bucket.size / 2,
            left: bucket.left.take(),
            right: bucket.right.take(),
        });
    }
    for pair in buckets[singleton_start..].chunks_exact_mut(2) {
        let left = match (pair[0].left.take(), pair[1].left.take()) {
            (None, None) => None,
            (Some(left), None) => Some(left),
            (None, Some(mut right)) => {
                right[0] *= challenge_inv;
                counters.real_fold_field_ops += 1;
                Some(right)
            }
            (Some(mut left), Some(right)) => {
                left[0] += challenge_inv * right[0];
                counters.real_fold_field_ops += 2;
                Some(left)
            }
        };
        let right = match (pair[0].right.take(), pair[1].right.take()) {
            (None, None) => None,
            (Some(left), None) => Some(left),
            (None, Some(mut right)) => {
                right[0] *= challenge;
                counters.real_fold_field_ops += 1;
                Some(right)
            }
            (Some(mut left), Some(right)) => {
                left[0] += challenge * right[0];
                counters.real_fold_field_ops += 2;
                Some(left)
            }
        };
        folded.push(EoPairBucket {
            size: 1,
            left,
            right,
        });
    }
    if singleton_count % 2 == 1 {
        let last = buckets
            .last_mut()
            .expect("odd singleton suffix contains a final bucket");
        folded.push(EoPairBucket {
            size: 1,
            left: last.left.take(),
            right: last.right.take(),
        });
    }
    folded
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BvaError {
    EmptyAssembly,
    EmptyClaimBatch,
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
            Self::EmptyClaimBatch => write!(formatter, "a TBV claim batch cannot be empty"),
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

    use ark_bls12_381::Fr;
    use ark_ff::Field;
    use ark_std::{One, Zero};

    use super::{
        packing_stride_for_bit_width, BucketedVector, BvaFieldPairSource, BvaFieldSource,
        BvaI8Source, BvaLayout, EoLiteBulletProof, EoWitnessPairProof, PackedEoLiteBulletProof,
        TbvBucket, TensorBasisVector, TensorSegmentRoute, TensorSubsegmentRoute, TensorVectorRoute,
    };

    #[test]
    fn tensor_term_prefix_sum_matches_explicit_irregular_prefixes() {
        let eta = Fr::from(23_u64);
        let factors = vec![
            (Fr::from(2_u64), Fr::from(3_u64)),
            (Fr::from(5_u64), Fr::from(7_u64)),
            (Fr::from(11_u64), Fr::from(13_u64)),
            (Fr::from(17_u64), Fr::from(19_u64)),
        ];

        for physical_len in 0..=1_usize << factors.len() {
            let expected = (0..physical_len).fold(Fr::zero(), |total, index| {
                total
                    + factors
                        .iter()
                        .enumerate()
                        .fold(eta, |product, (bit, (left, right))| {
                            product
                                * if (index >> bit) & 1 == 0 {
                                    *left
                                } else {
                                    *right
                                }
                        })
            });
            assert_eq!(
                super::tensor_term_prefix_sum(eta, &factors, physical_len),
                expected,
                "prefix length {physical_len}"
            );
        }
    }

    #[test]
    fn tensor_projection_visits_only_a_terms_nonzero_dyadic_support() {
        let values = (1_u64..=16).map(Fr::from).collect::<Vec<_>>();
        let assembly = BucketedVector::new(vec![values.clone()]).unwrap();
        let eta = Fr::from(23_u64);
        let factors = vec![
            (Fr::from(2_u64), Fr::from(3_u64)),
            (Fr::from(5_u64), Fr::from(7_u64)),
            (Fr::zero(), Fr::from(1_u64)),
            (Fr::from(1_u64), Fr::zero()),
        ];
        let tbv = TensorBasisVector::new(
            assembly.layout(),
            vec![TbvBucket::from_factor_terms(vec![(eta, factors.clone())])],
        )
        .unwrap();
        let expected = values
            .iter()
            .enumerate()
            .fold(Fr::zero(), |sum, (index, value)| {
                sum + *value
                    * factors
                        .iter()
                        .enumerate()
                        .fold(eta, |product, (bit, (left, right))| {
                            product
                                * if (index >> bit) & 1 == 0 {
                                    *left
                                } else {
                                    *right
                                }
                        })
            });

        let projection = assembly.inner_product(&tbv).unwrap();
        assert_eq!(projection.value, expected);
        assert!(
            projection.counters.real_fold_field_ops <= 16,
            "dyadic support should fold four values, not all sixteen: {:?}",
            projection.counters
        );
    }

    #[test]
    fn support_aware_tensor_projection_matches_naive_for_every_short_prefix() {
        let full_values = (1_u64..=16).map(Fr::from).collect::<Vec<_>>();
        for support_log in 0..=4_usize {
            for high_assignment in 0..(1_usize << (4 - support_log)) {
                let mut factors = vec![
                    (Fr::from(2_u64), Fr::from(3_u64)),
                    (Fr::from(5_u64), Fr::from(7_u64)),
                    (Fr::from(11_u64), Fr::from(13_u64)),
                    (Fr::from(17_u64), Fr::from(19_u64)),
                ];
                for (high_index, factor) in factors.iter_mut().enumerate().skip(support_log) {
                    *factor = if (high_assignment >> (high_index - support_log)) & 1 == 0 {
                        (Fr::one(), Fr::zero())
                    } else {
                        (Fr::zero(), Fr::one())
                    };
                }
                for physical_len in 1..=full_values.len() {
                    let values = &full_values[..physical_len];
                    let expected =
                        values
                            .iter()
                            .enumerate()
                            .fold(Fr::zero(), |sum, (index, value)| {
                                sum + *value
                                    * factors.iter().enumerate().fold(
                                        Fr::one(),
                                        |product, (bit, (left, right))| {
                                            product
                                                * if (index >> bit) & 1 == 0 {
                                                    *left
                                                } else {
                                                    *right
                                                }
                                        },
                                    )
                            });
                    let mut counters = super::PaddingWorkCounters::default();
                    let actual = super::tensor_project_factors(values, &factors, &mut counters);
                    assert_eq!(
                        actual, expected,
                        "support_log={support_log}, assignment={high_assignment}, len={physical_len}"
                    );
                    assert_eq!(counters.padding_fold_field_ops, 0);
                }
            }
        }
    }

    struct ReplaySource {
        layout: BvaLayout,
        vectors: Vec<Vec<Fr>>,
    }

    struct ReplayPairSource {
        layout: BvaLayout,
        left: Vec<Vec<Fr>>,
        right: Vec<Vec<Fr>>,
    }

    struct ReplayI8Source {
        layout: BvaLayout,
        vectors: Vec<Vec<i8>>,
    }

    impl BvaFieldPairSource<Fr> for ReplayPairSource {
        type Error = Infallible;

        fn layout(&self) -> &BvaLayout {
            &self.layout
        }

        fn for_each_vector_pair(
            &mut self,
            visit: &mut dyn FnMut(usize, &[Fr], &[Fr]),
        ) -> Result<(), Self::Error> {
            for (index, (left, right)) in self.left.iter().zip(&self.right).enumerate() {
                visit(index, left, right);
            }
            Ok(())
        }
    }

    impl BvaFieldSource<Fr> for ReplaySource {
        type Error = Infallible;

        fn layout(&self) -> &BvaLayout {
            &self.layout
        }

        fn for_each_vector(
            &mut self,
            visit: &mut dyn FnMut(usize, &[Fr]),
        ) -> Result<(), Self::Error> {
            for (index, vector) in self.vectors.iter().enumerate() {
                visit(index, vector);
            }
            Ok(())
        }
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

    fn two_vector_tbv(layout: &BvaLayout) -> TensorBasisVector<Fr> {
        TensorBasisVector::new(
            layout,
            vec![
                TbvBucket::new(7.into(), vec![2.into(), 3.into()]),
                TbvBucket::new(5.into(), vec![]),
            ],
        )
        .unwrap()
    }

    #[test]
    fn bva_layout_matches_the_paper_algorithm_without_materializing_padding() {
        let layout = BvaLayout::from_lengths(&[3, 5, 2]).unwrap();

        assert_eq!(layout.logical_len(), 16);
        assert_eq!(layout.active_len(), 14);
        assert_eq!(layout.bucket_sizes(), &[8, 4, 2]);
        assert_eq!(layout.vector_to_bucket(), &[1, 0, 2]);
        assert_eq!(layout.padding_bucket_count(), 0);
        assert_eq!(layout.materialized_padding_entries(), 0);
    }

    #[test]
    fn decreasing_layout_eo_matches_dense_logical_vectors() {
        for lengths in [
            vec![3, 5, 2],
            vec![1, 1, 1],
            vec![7, 4, 3, 2, 2, 2, 1, 1, 1],
        ] {
            let vectors = lengths
                .iter()
                .enumerate()
                .map(|(vector_index, length)| {
                    (0..*length)
                        .map(|entry_index| {
                            Fr::from(((vector_index + 2) * (entry_index + 3)) as u64)
                        })
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let assembly = BucketedVector::new(vectors.clone()).unwrap();
            let mut dense = vec![Fr::zero(); assembly.layout().logical_len()];
            for (vector_index, vector) in vectors.iter().enumerate() {
                let offset = assembly.layout().vector_offset(vector_index).unwrap();
                dense[offset..offset + vector.len()].copy_from_slice(vector);
            }
            let challenges = (0..assembly.layout().logical_len().ilog2())
                .map(|round| Fr::from(u64::from(round) + 2))
                .collect::<Vec<_>>();
            let mut expected = dense;
            for challenge in &challenges {
                expected = expected
                    .chunks_exact(2)
                    .map(|pair| pair[0] + *challenge * pair[1])
                    .collect();
            }
            let actual = assembly.fold_to_scalar(&challenges).unwrap();
            assert_eq!(actual.value, expected[0], "lengths={lengths:?}");
            assert_eq!(actual.counters.materialized_padding_entries, 0);
            assert_eq!(actual.counters.padding_fold_field_ops, 0);
        }
    }

    #[test]
    fn eight_bit_packing_uses_32_entries_for_a_256_bit_field() {
        assert_eq!(std::mem::size_of::<i8>(), 1);
        assert_eq!(packing_stride_for_bit_width::<Fr>(64, 8).unwrap(), 32);
        assert_eq!(packing_stride_for_bit_width::<Fr>(16, 8).unwrap(), 16);
    }

    #[test]
    fn packed_eo_litebullet_reads_native_i8_storage_and_crosses_bucket_boundaries() {
        let vectors = vec![vec![-3_i8, 2, 7, -1, 4, 6, -2], vec![5], vec![-4]];
        let layout =
            BvaLayout::from_lengths(&vectors.iter().map(Vec::len).collect::<Vec<_>>()).unwrap();
        assert_eq!(layout.bucket_sizes(), &[8, 1, 1]);
        assert_eq!(layout.active_len(), 10);
        assert_eq!(layout.logical_len(), 16);
        let tbv =
            TensorBasisVector::global_tensor(&layout, &[2.into(), 3.into(), 5.into(), 7.into()])
                .unwrap();
        let claimed = vectors
            .iter()
            .enumerate()
            .fold(Fr::zero(), |sum, (vector_index, vector)| {
                sum + vector.iter().enumerate().fold(
                    Fr::zero(),
                    |vector_sum, (entry_index, value)| {
                        vector_sum
                            + Fr::from(i32::from(*value))
                                * tbv
                                    .physical_coefficient_at(vector_index, entry_index)
                                    .unwrap()
                    },
                )
            });
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

        let field_assembly = BucketedVector::new(
            vectors
                .into_iter()
                .map(|vector| {
                    vector
                        .into_iter()
                        .map(|value| Fr::from(i32::from(value)))
                        .collect()
                })
                .collect(),
        )
        .unwrap();
        assert_eq!(
            proof.a_reduced,
            field_assembly
                .fold_to_scalar(&verification.challenges_inv)
                .unwrap()
                .value
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

        let mut tampered = proof;
        tampered.packed_positive[0] += Fr::one();
        assert!(!tampered.verify_with_tbv(&tbv, claimed).accepted);
    }

    #[test]
    fn bva_tbv_inner_product_skips_all_padding_field_operations() {
        let assembly =
            BucketedVector::<Fr>::new(vec![vec![1.into(), 2.into(), 3.into()], vec![4.into()]])
                .unwrap();
        let tbv = two_vector_tbv(assembly.layout());

        // Bucket 1 contributes 5*4. Bucket 0 contributes
        // 7*(1 + 2*2 + 3*3), for a total of 118.
        let result = assembly.inner_product(&tbv).unwrap();
        assert_eq!(result.value, Fr::from(118_u64));
        assert_eq!(result.counters.materialized_padding_entries, 0);
        assert_eq!(result.counters.padding_ip_field_ops, 0);
        assert_eq!(result.counters.padding_fold_field_ops, 0);
    }

    #[test]
    fn eo_folding_skips_middle_trailing_and_singleton_padding() {
        let assembly =
            BucketedVector::<Fr>::new(vec![vec![2.into()], vec![3.into(), 5.into(), 7.into()]])
                .unwrap();

        // The logical vector is [3, 5, 7, 0, 2, 0, 0, 0]. Three EO rounds
        // with challenges 2, 3, 5 reduce it to 44.
        let result = assembly
            .fold_to_scalar(&[2.into(), 3.into(), 5.into()])
            .unwrap();
        assert_eq!(result.value, Fr::from(44_u64));
        assert_eq!(result.counters.materialized_padding_entries, 0);
        assert_eq!(result.counters.padding_ip_field_ops, 0);
        assert_eq!(result.counters.padding_fold_field_ops, 0);
    }

    #[test]
    fn eo_litebullet_proves_bva_tbv_inner_product_without_padding_work() {
        let assembly =
            BucketedVector::<Fr>::new(vec![vec![1.into(), 2.into(), 3.into()], vec![4.into()]])
                .unwrap();
        let tbv = two_vector_tbv(assembly.layout());
        let claimed = Fr::from(118_u64);

        let (proof, counters) = EoLiteBulletProof::prove(&assembly, &tbv, claimed).unwrap();
        let verification = proof.verify_with_tbv(&tbv, claimed);
        let consuming_verification = proof.verify_with_tbv_consuming(tbv.clone(), claimed);

        let encoded = proof.to_bytes().unwrap();
        let decoded = EoLiteBulletProof::<Fr>::from_bytes_exact(&encoded).unwrap();
        assert_eq!(decoded, proof);
        assert!(decoded.verify_with_tbv(&tbv, claimed).accepted);
        let mut with_trailing_byte = encoded.clone();
        with_trailing_byte.push(0);
        assert!(EoLiteBulletProof::<Fr>::from_bytes_exact(&with_trailing_byte).is_err());
        assert!(EoLiteBulletProof::<Fr>::from_bytes_exact(&encoded[..encoded.len() - 1]).is_err());

        assert!(verification.accepted);
        assert_eq!(consuming_verification, verification);
        assert_eq!(proof.rounds.len(), 3);
        assert_eq!(counters.materialized_padding_entries, 0);
        assert_eq!(counters.padding_ip_field_ops, 0);
        assert_eq!(counters.padding_fold_field_ops, 0);
        let folded = assembly
            .clone()
            .fold_to_scalar(&verification.challenges_inv)
            .unwrap();
        assert_eq!(proof.a_reduced, folded.value);

        let mut tampered = proof.clone();
        tampered.rounds[0].0 += Fr::from(1_u64);
        assert!(!tampered.verify_with_tbv(&tbv, claimed).accepted);

        let scalar_only = proof.verify_scalar_identity(assembly.layout().logical_len(), claimed);
        let reduced_claim = proof
            .rounds
            .iter()
            .zip(
                scalar_only
                    .challenges
                    .iter()
                    .zip(&scalar_only.challenges_inv),
            )
            .fold(
                claimed,
                |accumulator, ((iota, rho), (challenge, inverse))| {
                    accumulator + *challenge * iota + *inverse * rho
                },
            );
        let mut unbound_tbv = proof.clone();
        unbound_tbv.b_reduced += Fr::from(1_u64);
        if unbound_tbv.b_reduced.is_zero() {
            unbound_tbv.b_reduced += Fr::from(1_u64);
        }
        unbound_tbv.a_reduced = reduced_claim * unbound_tbv.b_reduced.inverse().unwrap();
        assert!(
            unbound_tbv
                .verify_scalar_identity(assembly.layout().logical_len(), claimed)
                .accepted
        );
        assert!(!unbound_tbv.verify_with_tbv(&tbv, claimed).accepted);
    }

    #[test]
    fn owned_bound_eo_matches_streaming_transcript_and_rejects_binding_tamper() {
        let vectors = vec![
            vec![Fr::from(1_u64), Fr::from(2_u64), Fr::from(3_u64)],
            vec![Fr::from(4_u64)],
        ];
        let assembly = BucketedVector::new(vectors.clone()).unwrap();
        let tbv = two_vector_tbv(assembly.layout());
        let claimed = assembly.inner_product(&tbv).unwrap().value;
        let binding = Fr::from(41_u64);
        let (owned, counters) =
            EoLiteBulletProof::prove_bound(&assembly, &tbv, claimed, binding).unwrap();
        let mut source = ReplaySource {
            layout: assembly.layout().clone(),
            vectors,
        };
        let (streamed, _, _) =
            EoLiteBulletProof::prove_streaming_bound(&mut source, &tbv, claimed, binding).unwrap();

        assert_eq!(owned, streamed);
        let borrowed_verification = owned.verify_with_tbv_bound(&tbv, claimed, binding);
        let consuming_verification =
            owned.verify_with_tbv_bound_consuming(tbv.clone(), claimed, binding);
        assert!(borrowed_verification.accepted);
        assert_eq!(consuming_verification, borrowed_verification);
        assert!(
            !owned
                .verify_with_tbv_bound(&tbv, claimed, binding + Fr::from(1_u64))
                .accepted
        );
        assert_eq!(counters.materialized_padding_entries, 0);
        assert_eq!(counters.padding_ip_field_ops, 0);
        assert_eq!(counters.padding_fold_field_ops, 0);
    }

    #[test]
    fn consuming_prechecked_bound_eo_matches_borrowed_transcript() {
        let assembly =
            BucketedVector::<Fr>::new(vec![vec![1.into(), 2.into(), 3.into()], vec![4.into()]])
                .unwrap();
        let tbv = two_vector_tbv(assembly.layout());
        let initial = assembly.inner_product(&tbv).unwrap();
        let binding = Fr::from(41_u64);
        let (borrowed, borrowed_counters) =
            EoLiteBulletProof::prove_bound(&assembly, &tbv, initial.value, binding).unwrap();
        let (consuming, consuming_counters) =
            EoLiteBulletProof::prove_bound_consuming_tbv_prechecked(
                &assembly,
                tbv,
                initial.value,
                binding,
                initial.counters,
            )
            .unwrap();

        assert_eq!(consuming, borrowed);
        assert_eq!(consuming_counters, borrowed_counters);
        assert!(
            consuming
                .verify_scalar_for_prover_bound(
                    assembly.layout().logical_len(),
                    initial.value,
                    binding,
                )
                .accepted
        );
    }

    #[test]
    fn consuming_tbv_eo_matches_borrowed_transcript_without_padding_work() {
        let assembly =
            BucketedVector::<Fr>::new(vec![vec![1.into(), 2.into(), 3.into()], vec![4.into()]])
                .unwrap();
        let tbv = TensorBasisVector::new(
            assembly.layout(),
            vec![
                TbvBucket::from_terms(vec![
                    (7.into(), vec![2.into(), 3.into()]),
                    (19.into(), vec![23.into(), 29.into()]),
                ]),
                TbvBucket::new(5.into(), vec![]),
            ],
        )
        .unwrap();
        let claimed = assembly.inner_product(&tbv).unwrap().value;
        let (borrowed, _) = EoLiteBulletProof::prove(&assembly, &tbv, claimed).unwrap();
        let (consuming, counters) =
            EoLiteBulletProof::prove_consuming_tbv(&assembly, tbv.clone(), claimed).unwrap();

        assert_eq!(consuming, borrowed);
        assert!(consuming.verify_with_tbv(&tbv, claimed).accepted);
        assert_eq!(counters.materialized_padding_entries, 0);
        assert_eq!(counters.padding_ip_field_ops, 0);
        assert_eq!(counters.padding_fold_field_ops, 0);
    }

    #[test]
    fn eo_litebullet_supports_a_sum_of_tensor_terms_on_one_witness_vector() {
        let vectors = vec![vec![1.into(), 2.into(), 3.into(), 4.into()]];
        let assembly = BucketedVector::<Fr>::new(vectors.clone()).unwrap();
        let tbv = TensorBasisVector::new(
            assembly.layout(),
            vec![TbvBucket::from_terms(vec![
                (5.into(), vec![2.into(), 3.into()]),
                (7.into(), vec![11.into(), 13.into()]),
            ])],
        )
        .unwrap();
        let claimed = assembly.inner_product(&tbv).unwrap().value;
        let expected = Fr::from(5_u64)
            * (Fr::from(1_u64)
                + Fr::from(2_u64) * Fr::from(2_u64)
                + Fr::from(3_u64) * Fr::from(3_u64)
                + Fr::from(4_u64) * Fr::from(6_u64))
            + Fr::from(7_u64)
                * (Fr::from(1_u64)
                    + Fr::from(2_u64) * Fr::from(11_u64)
                    + Fr::from(3_u64) * Fr::from(13_u64)
                    + Fr::from(4_u64) * Fr::from(143_u64));
        assert_eq!(claimed, expected);

        let (owned, _) = EoLiteBulletProof::prove(&assembly, &tbv, claimed).unwrap();
        let mut source = ReplaySource {
            layout: assembly.layout().clone(),
            vectors,
        };
        let (streamed, counters, profile) =
            EoLiteBulletProof::prove_streaming(&mut source, &tbv, claimed).unwrap();

        assert_eq!(streamed, owned);
        assert!(streamed.verify_with_tbv(&tbv, claimed).accepted);
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(counters.padding_ip_field_ops, 0);
        assert_eq!(counters.padding_fold_field_ops, 0);
    }

    #[test]
    fn eo_litebullet_supports_one_tensor_lagrange_point_term() {
        let vectors = vec![(1_u64..=8).map(Fr::from).collect::<Vec<_>>()];
        let assembly = BucketedVector::<Fr>::new(vectors.clone()).unwrap();
        let tbv = TensorBasisVector::new(
            assembly.layout(),
            vec![TbvBucket::from_factor_terms(vec![(
                Fr::from(5_u64),
                vec![
                    (Fr::from(0_u64), Fr::from(1_u64)),
                    (Fr::from(1_u64), Fr::from(0_u64)),
                    (Fr::from(0_u64), Fr::from(1_u64)),
                ],
            )])],
        )
        .unwrap();
        let claimed = Fr::from(30_u64);
        assert_eq!(assembly.inner_product(&tbv).unwrap().value, claimed);

        let (owned, _) = EoLiteBulletProof::prove(&assembly, &tbv, claimed).unwrap();
        let mut source = ReplaySource {
            layout: assembly.layout().clone(),
            vectors,
        };
        let (streamed, _, profile) =
            EoLiteBulletProof::prove_streaming(&mut source, &tbv, claimed).unwrap();

        assert_eq!(streamed, owned);
        assert!(streamed.verify_with_tbv(&tbv, claimed).accepted);
        assert_eq!(profile.materialized_padding_entries, 0);
    }

    #[test]
    fn synchronized_tbv_claims_combine_before_one_streaming_eo() {
        let vectors = vec![vec![1.into(), 2.into(), 3.into(), 4.into()]];
        let assembly = BucketedVector::<Fr>::new(vectors.clone()).unwrap();
        let first = TensorBasisVector::new(
            assembly.layout(),
            vec![TbvBucket::new(5.into(), vec![2.into(), 3.into()])],
        )
        .unwrap();
        let second = TensorBasisVector::new(
            assembly.layout(),
            vec![TbvBucket::from_factor_terms(vec![(
                Fr::from(7_u64),
                vec![
                    (Fr::from(0_u64), Fr::from(1_u64)),
                    (Fr::from(1_u64), Fr::from(0_u64)),
                ],
            )])],
        )
        .unwrap();
        let first_claim = assembly.inner_product(&first).unwrap().value;
        let second_claim = assembly.inner_product(&second).unwrap().value;
        let weights = [Fr::from(11_u64), Fr::from(13_u64)];
        let combined = TensorBasisVector::linear_combination(
            assembly.layout(),
            &[(weights[0], &first), (weights[1], &second)],
        )
        .unwrap();
        let owned_combined = TensorBasisVector::linear_combination_owned(
            assembly.layout(),
            vec![(weights[0], first.clone()), (weights[1], second.clone())],
        )
        .unwrap();
        assert_eq!(owned_combined, combined);
        let combined_claim = weights[0] * first_claim + weights[1] * second_claim;
        assert_eq!(
            assembly.inner_product(&combined).unwrap().value,
            combined_claim
        );

        let mut source = ReplaySource {
            layout: assembly.layout().clone(),
            vectors,
        };
        let (proof, _, profile) =
            EoLiteBulletProof::prove_streaming(&mut source, &combined, combined_claim).unwrap();

        assert!(proof.verify_with_tbv(&combined, combined_claim).accepted);
        assert_eq!(profile.materialized_padding_entries, 0);
    }

    #[test]
    fn global_tensor_tbv_matches_eo_fold_and_multiplies_pointwise() {
        let assembly =
            BucketedVector::<Fr>::new(vec![vec![1.into(), 2.into(), 3.into()], vec![4.into()]])
                .unwrap();
        let first_challenges = vec![2.into(), 3.into(), 5.into()];
        let second_challenges = vec![7.into(), 11.into(), 13.into()];
        let first = TensorBasisVector::global_tensor(assembly.layout(), &first_challenges).unwrap();
        let second =
            TensorBasisVector::global_tensor(assembly.layout(), &second_challenges).unwrap();
        let product = first.pointwise_product(&second).unwrap();
        let product_challenges = first_challenges
            .iter()
            .zip(&second_challenges)
            .map(|(left, right)| *left * *right)
            .collect::<Vec<_>>();
        let expected_product =
            TensorBasisVector::global_tensor(assembly.layout(), &product_challenges).unwrap();

        assert_eq!(
            assembly.inner_product(&first).unwrap().value,
            assembly
                .clone()
                .fold_to_scalar(&first_challenges)
                .unwrap()
                .value
        );
        assert_eq!(
            assembly.inner_product(&product).unwrap().value,
            assembly.inner_product(&expected_product).unwrap().value
        );

        let routed_target =
            BucketedVector::<Fr>::new(vec![vec![10.into(), 20.into(), 30.into()]]).unwrap();
        let routed = first
            .route_vector_maps(
                routed_target.layout(),
                &[
                    TensorVectorRoute::Point {
                        target_vector: 0,
                        target_index: 1,
                        scale: Fr::from(1_u64),
                    },
                    TensorVectorRoute::Omit,
                ],
            )
            .unwrap();
        assert_eq!(
            routed_target.inner_product(&routed).unwrap().value,
            Fr::from(20_u64) * first.physical_coefficient_sum(0).unwrap()
        );
    }

    #[test]
    fn aligned_dense_segments_route_to_decreasing_heterogeneous_vectors() {
        let target = BucketedVector::<Fr>::new(vec![
            vec![1.into(), 2.into(), 3.into()],
            vec![4.into(), 5.into(), 6.into(), 7.into(), 8.into()],
            vec![9.into(), 10.into()],
        ])
        .unwrap();
        assert_eq!(target.layout().bucket_sizes(), &[8, 4, 2]);
        assert_eq!(target.layout().vector_to_bucket(), &[1, 0, 2]);
        assert_eq!(target.layout().logical_len(), 16);

        let offsets = target.layout().bucket_offsets();
        let mut dense_values = vec![Fr::zero(); 16];
        for (vector_index, values) in target.vectors().iter().enumerate() {
            let offset = offsets[target.layout().vector_to_bucket()[vector_index]];
            dense_values[offset..offset + values.len()].copy_from_slice(values);
        }
        let dense = BucketedVector::new(vec![dense_values]).unwrap();
        let challenges = vec![2.into(), 3.into(), 5.into(), 7.into()];
        let dense_tbv = TensorBasisVector::global_tensor(dense.layout(), &challenges).unwrap();
        let routes = (0..target.layout().vector_lengths().len())
            .map(|target_vector| TensorSegmentRoute {
                source_vector: 0,
                source_offset: offsets[target.layout().vector_to_bucket()[target_vector]],
                target_vector,
                scale: Fr::one(),
            })
            .collect::<Vec<_>>();
        let routed = dense_tbv
            .route_aligned_segments(target.layout(), &routes)
            .unwrap();

        assert_eq!(
            dense.inner_product(&dense_tbv).unwrap().value,
            target.inner_product(&routed).unwrap().value,
        );

        let mut misaligned = routes;
        misaligned[0].source_offset += 1;
        assert!(dense_tbv
            .route_aligned_segments(target.layout(), &misaligned)
            .is_err());
    }

    #[test]
    fn non_aligned_dense_subsegment_routes_without_materializing_padding() {
        let dense = BucketedVector::<Fr>::new(vec![vec![Fr::zero(); 16]]).unwrap();
        let target_values = vec![
            Fr::from(91_u64),
            Fr::from(10_u64),
            Fr::from(20_u64),
            Fr::from(30_u64),
            Fr::from(40_u64),
            Fr::from(50_u64),
            Fr::from(92_u64),
        ];
        let target = BucketedVector::<Fr>::new(vec![target_values.clone()]).unwrap();
        let challenges = vec![2.into(), 3.into(), 5.into(), 7.into()];
        let dense_tbv = TensorBasisVector::global_tensor(dense.layout(), &challenges).unwrap();
        let routed = dense_tbv
            .route_subsegments(
                target.layout(),
                &[TensorSubsegmentRoute {
                    source_vector: 0,
                    source_offset: 3,
                    target_vector: 0,
                    target_offset: 1,
                    length: 5,
                    scale: Fr::one(),
                }],
            )
            .unwrap();

        let expected = (0..5).fold(Fr::zero(), |sum, index| {
            let dense_index = 3 + index;
            let coefficient =
                challenges
                    .iter()
                    .enumerate()
                    .fold(Fr::one(), |product, (bit, challenge)| {
                        if (dense_index >> bit) & 1 == 1 {
                            product * challenge
                        } else {
                            product
                        }
                    });
            sum + target_values[1 + index] * coefficient
        });
        assert_eq!(target.inner_product(&routed).unwrap().value, expected);
    }

    #[test]
    fn streaming_witness_pair_eo_reduces_a_secret_hadamard_claim() {
        let left = vec![
            vec![Fr::from(1_u64), Fr::from(2_u64), Fr::from(3_u64)],
            vec![Fr::from(4_u64)],
        ];
        let right = vec![
            vec![Fr::from(5_u64), Fr::from(6_u64), Fr::from(7_u64)],
            vec![Fr::from(8_u64)],
        ];
        let layout = BvaLayout::from_lengths(&[3, 1]).unwrap();
        let mut source = ReplayPairSource {
            layout: layout.clone(),
            left,
            right,
        };
        let claim = Fr::from(70_u64);

        let (proof, counters, profile) =
            EoWitnessPairProof::prove_streaming(&mut source, claim).unwrap();
        let verification = proof.verify(layout.logical_len(), claim);

        assert!(verification.accepted);
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(counters.padding_ip_field_ops, 0);
        assert_eq!(counters.padding_fold_field_ops, 0);
        assert!(profile.maximum_live_vector_entries <= 3);

        let encoded = proof.to_bytes().unwrap();
        let decoded = EoWitnessPairProof::<Fr>::from_bytes_exact(&encoded).unwrap();
        assert_eq!(decoded, proof);
        assert!(decoded.verify(layout.logical_len(), claim).accepted);
        let mut with_trailing_byte = encoded.clone();
        with_trailing_byte.push(0);
        assert!(EoWitnessPairProof::<Fr>::from_bytes_exact(&with_trailing_byte).is_err());
        assert!(EoWitnessPairProof::<Fr>::from_bytes_exact(&encoded[..encoded.len() - 1]).is_err());

        let mut tampered = proof;
        tampered.a_reduced += Fr::from(1_u64);
        assert!(!tampered.verify(layout.logical_len(), claim).accepted);
    }

    #[test]
    fn owned_bound_witness_pair_matches_streaming_and_rejects_binding_tamper() {
        let left = vec![
            vec![Fr::from(1_u64), Fr::from(2_u64), Fr::from(3_u64)],
            vec![Fr::from(4_u64)],
        ];
        let right = vec![
            vec![Fr::from(5_u64), Fr::from(6_u64), Fr::from(7_u64)],
            vec![Fr::from(8_u64)],
        ];
        let left_owned = BucketedVector::new(left.clone()).unwrap();
        let right_owned = BucketedVector::new(right.clone()).unwrap();
        let claim = Fr::from(70_u64);
        let binding = Fr::from(43_u64);

        let (owned, counters) =
            EoWitnessPairProof::prove_bound(&left_owned, &right_owned, claim, binding).unwrap();
        let mut source = ReplayPairSource {
            layout: left_owned.layout().clone(),
            left,
            right,
        };
        let (streamed, _, _) =
            EoWitnessPairProof::prove_streaming_bound(&mut source, claim, binding).unwrap();

        assert_eq!(owned, streamed);
        assert!(
            owned
                .verify_bound(left_owned.layout().logical_len(), claim, binding)
                .accepted
        );
        assert!(
            !owned
                .verify_bound(
                    left_owned.layout().logical_len(),
                    claim,
                    binding + Fr::from(1_u64),
                )
                .accepted
        );
        assert_eq!(counters.materialized_padding_entries, 0);
        assert_eq!(counters.padding_ip_field_ops, 0);
        assert_eq!(counters.padding_fold_field_ops, 0);
    }

    #[test]
    fn streaming_eo_litebullet_matches_owned_proof_with_one_vector_live() {
        let vectors = vec![
            vec![Fr::from(1_u64), Fr::from(2_u64), Fr::from(3_u64)],
            vec![Fr::from(4_u64)],
        ];
        let assembly = BucketedVector::new(vectors.clone()).unwrap();
        let tbv = two_vector_tbv(assembly.layout());
        let claimed = assembly.inner_product(&tbv).unwrap().value;
        let (expected, _) = EoLiteBulletProof::prove(&assembly, &tbv, claimed).unwrap();
        let mut source = ReplaySource {
            layout: assembly.layout().clone(),
            vectors,
        };

        let (actual, counters, profile) =
            EoLiteBulletProof::prove_streaming(&mut source, &tbv, claimed).unwrap();

        assert_eq!(actual, expected);
        assert!(actual.verify_with_tbv(&tbv, claimed).accepted);
        assert_eq!(profile.maximum_live_vector_entries, 3);
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(counters.padding_ip_field_ops, 0);
        assert_eq!(counters.padding_fold_field_ops, 0);
    }

    #[test]
    fn streaming_eo_litebullet_matches_owned_across_mixed_bucket_layouts() {
        for lengths in [vec![1], vec![2, 3, 5, 1], vec![7, 4, 2, 9, 1]] {
            let vectors = lengths
                .iter()
                .enumerate()
                .map(|(vector, length)| {
                    (0..*length)
                        .map(|entry| Fr::from(((vector + 1) * (entry + 2)) as u64))
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let assembly = BucketedVector::new(vectors.clone()).unwrap();
            let tbv = TensorBasisVector::new(
                assembly.layout(),
                assembly
                    .layout()
                    .bucket_sizes()
                    .iter()
                    .enumerate()
                    .map(|(bucket, size)| {
                        TbvBucket::new(
                            Fr::from((bucket + 3) as u64),
                            (0..size.ilog2())
                                .map(|round| Fr::from(u64::from(round) + 7))
                                .collect(),
                        )
                    })
                    .collect(),
            )
            .unwrap();
            let claimed = assembly.inner_product(&tbv).unwrap().value;
            let (expected, _) = EoLiteBulletProof::prove(&assembly, &tbv, claimed).unwrap();
            let mut source = ReplaySource {
                layout: assembly.layout().clone(),
                vectors,
            };

            let (actual, counters, profile) =
                EoLiteBulletProof::prove_streaming(&mut source, &tbv, claimed).unwrap();

            assert_eq!(actual, expected);
            assert_eq!(profile.materialized_padding_entries, 0);
            assert_eq!(counters.padding_ip_field_ops, 0);
            assert_eq!(counters.padding_fold_field_ops, 0);
        }
    }
}
