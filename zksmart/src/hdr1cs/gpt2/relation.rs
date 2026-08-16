use std::collections::HashMap;
use std::convert::{TryFrom, TryInto};
use std::fmt::{Display, Formatter};
use std::io::{Cursor, Read, Seek, Write};
use std::marker::PhantomData;

use ark_bls12_381::{Bls12_381, Fr};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::{Field, PrimeField};
use ark_poly_commit::smart_pc::{
    data_structures::Trans as PcsTrans, SmartPC, UniversalParams as PcsParams,
};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError};
use ark_std::{ops::Mul, UniformRand, Zero};
use sha2::{Digest, Sha256};

use super::{
    q12_exp, q12_gelu, q12_reciprocal, q12_rsqrt, Gpt2ActivationWitness, Gpt2CheckpointError,
    Gpt2GlobalLayout, Gpt2HdR1csShape, Gpt2ProofTranscript, Gpt2PublicStatement, Gpt2Q12Checkpoint,
    Gpt2Q12Metadata, Gpt2Q12OperationKind,
};
use crate::data_structures::{ZkSRS, SRS};
use crate::hdr1cs::global_mlp::GlobalBlockId;
use crate::protocols::bucketed_ip::{
    BucketedVector, BvaFieldPairSource, BvaLayout, EoLiteBulletProof, EoWitnessPairProof,
    TbvBucket, TensorBasisVector, TensorVectorRoute,
};
use crate::protocols::streaming_grand_product::{
    BvaGrandProductHadamardProof, BvaGrandProductOpening, BvaGrandProductProof,
    BvaGrandProductRecurrenceProof,
};
use crate::protocols::vector_relation::BvaAffineRelationSource;
use crate::protocols::vector_relation::{
    BvaAffineRelationProof, BvaBooleanProof, BvaCommittedEvaluationProof, BvaRangeProof,
    BvaRoutedEqualityProof, BvaVectorOpeningProof,
};
use crate::protocols::virtualpcs::{
    bva_eo_opening_point, commit_bva_field_materialized_parallel, commit_bva_field_streaming,
    commit_sparse_bva_field_streaming, inner_product_bva_field_streaming, open_bva_field_streaming,
    BvaFieldSource, BvaI32Source, SparseBvaFieldSource,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RelationFamily {
    Linear,
    Bilinear,
    TaggedLookup,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum LinearRelationKind {
    EmbeddingGatherAdd = 0,
    ReduceSum = 1,
    EuclideanDivision = 2,
    BroadcastSubtract = 3,
    AddConstant = 4,
    BiasScaleAdd = 5,
    ResidualAdd = 6,
    MaximumDifference = 7,
    OneHotSum = 8,
    PrefixSelector = 9,
    StrictGap = 10,
    ResidualMass = 11,
    AddResidualAdjustment = 12,
    ExactRowMass = 13,
    AffineAdd = 14,
    ScaleAndDivide = 15,
    RangeRecomposition = 16,
    ClampSelectorSum = 17,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum BilinearRelationKind {
    MatrixMultiply = 0,
    LastTokenMatrixMultiply = 1,
    Square = 2,
    Hadamard = 3,
    Boolean = 4,
    SelectorZero = 5,
    CausalQk = 6,
    CausalPv = 7,
    SelectorResidual = 8,
    ClampLower = 9,
    ClampUpper = 10,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[repr(u8)]
enum TaggedLookupRelationKind {
    Gelu = 0,
    Exp = 1,
    Reciprocal = 2,
    InverseSqrt = 3,
    Range1 = 4,
    Range2 = 5,
    Range3 = 6,
    Range4 = 7,
    Range5 = 8,
    Range6 = 9,
    Range7 = 10,
    Range8 = 11,
}

const TAGGED_LOOKUP_TABLE_ORDER: [TaggedLookupRelationKind; 12] = [
    TaggedLookupRelationKind::Gelu,
    TaggedLookupRelationKind::Exp,
    TaggedLookupRelationKind::Reciprocal,
    TaggedLookupRelationKind::InverseSqrt,
    TaggedLookupRelationKind::Range1,
    TaggedLookupRelationKind::Range2,
    TaggedLookupRelationKind::Range3,
    TaggedLookupRelationKind::Range4,
    TaggedLookupRelationKind::Range5,
    TaggedLookupRelationKind::Range6,
    TaggedLookupRelationKind::Range7,
    TaggedLookupRelationKind::Range8,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
enum AuxiliaryBlockKind {
    Accumulator = 0,
    Numerator = 1,
    Remainder = 2,
    RowSum = 3,
    Mean = 4,
    Centered = 5,
    Square = 6,
    Variance = 7,
    LookupInput = 8,
    LookupOutput = 9,
    Product = 10,
    Scaled = 11,
    PackedScore = 12,
    RowMaximum = 13,
    Selector = 14,
    Gap = 15,
    PrefixSelector = 16,
    StrictGap = 17,
    Exponential = 18,
    Denominator = 19,
    Reciprocal = 20,
    BaseProbability = 21,
    ResidualMass = 22,
    ResidualAdjustment = 23,
    Probability = 24,
    ClampedSigned = 25,
    ClampedUnsigned = 26,
    ClampSlack = 27,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AuxiliaryBlockDescriptor {
    id: GlobalBlockId,
    operation_index: usize,
    kind: AuxiliaryBlockKind,
    rows: usize,
    cols: usize,
    value_bits: u16,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct RangeLimbBlockDescriptor {
    id: GlobalBlockId,
    operation_index: usize,
    source_kind: RangeLimbSource,
    source: GlobalBlockId,
    rows: usize,
    cols: usize,
    limb: usize,
    limb_bits: u8,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
enum RangeLimbSource {
    Base { block_index: usize },
    Auxiliary { block_index: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PrimitiveRelationDescriptor {
    family: RelationFamily,
    kind: u8,
    operation_index: usize,
    blocks: Vec<GlobalBlockId>,
    scalar_count: usize,
    immediate: i64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct AuxiliaryWitnessBlock {
    id: GlobalBlockId,
    columns: Vec<Vec<i128>>,
}

struct ClampWitnessColumns {
    clamped: Vec<Vec<i128>>,
    low_selector: Vec<Vec<i128>>,
    middle_selector: Vec<Vec<i128>>,
    high_selector: Vec<Vec<i128>>,
    low_slack: Vec<Vec<i128>>,
    middle_low_slack: Vec<Vec<i128>>,
    middle_high_slack: Vec<Vec<i128>>,
    high_slack: Vec<Vec<i128>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Gpt2PrimitiveWitness {
    blocks: Vec<AuxiliaryWitnessBlock>,
}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
struct TaggedLookupQuery {
    tag: TaggedLookupRelationKind,
    input: i128,
    output: i128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Gpt2LookupRegistry {
    scale: i64,
    maximum_tokens: usize,
}

struct Gpt2LookupQueryStream<'a, R> {
    checkpoint: &'a mut Gpt2Q12Checkpoint<R>,
    layout: &'a Gpt2GlobalLayout,
    activations: &'a Gpt2ActivationWitness,
    witness: &'a Gpt2PrimitiveWitness,
    manifest: &'a Gpt2PrimitiveRelationManifest,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct LookupStreamProfile {
    semantic_queries: usize,
    range_queries: usize,
    maximum_live_semantic_entries: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct StreamingLassoProducts<F: PrimeField> {
    left: F,
    right: F,
    query_count: usize,
    distinct_table_rows: usize,
    public_table_rows: usize,
    maximum_live_semantic_entries: usize,
    multiplicities: Vec<(usize, u64)>,
}

struct Gpt2LassoMultiplicitySource<F: PrimeField> {
    layout: BvaLayout,
    entries: Vec<(usize, u64)>,
    _field: PhantomData<F>,
}

#[derive(Clone, Copy)]
enum LassoPublicTableComponent {
    Value,
    Counter,
}

/// Sparse verifier-reconstructible view of the nonzero Lasso table tail.
/// The support is public in this soundness-first mode.  Query-prefix zeros,
/// unused public table rows, and BVA padding are all omitted from the source.
struct Gpt2LassoPublicTableComponentSource<F: PrimeField> {
    layout: BvaLayout,
    entries: Vec<(usize, usize, F)>,
}

/// Non-zero table support carried by the current soundness-first proof mode.
/// The three hiding scalars are intentionally disclosed because all three
/// reconstructed vectors are already public in this mode.
#[derive(Clone)]
struct Gpt2LassoPublicTableBinding<F: PrimeField> {
    multiplicities: Vec<(usize, u64)>,
    multiplicity_hiding: F,
    table_value_hiding: F,
    table_counter_hiding: F,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Gpt2LassoPublicTableBindingProfile {
    support_entries: usize,
    nonzero_group_contributions: usize,
    materialized_padding_entries: usize,
    padding_group_operations: usize,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct LassoTimestampStreamProfile {
    query_count: usize,
    vector_count: usize,
    maximum_live_entries: usize,
    maximum_timestamp: u64,
}

/// Replays the authenticated lookup stream and emits Lasso prefix timestamps
/// in bounded chunks.  Neither the full timestamp vector nor its BVA padding
/// is ever resident: every replay keeps only the per-table-row counters and
/// one physical chunk.
struct Gpt2LassoTimestampSource<'a, 'b, R, F> {
    stream: &'a mut Gpt2LookupQueryStream<'b, R>,
    registry: &'a Gpt2LookupRegistry,
    layout: BvaLayout,
    chunk_len: usize,
    expected_queries: usize,
    profile: LassoTimestampStreamProfile,
    _field: PhantomData<F>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LassoFactorSide {
    Left,
    Right,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct LassoFactorStreamProfile {
    query_factors: usize,
    table_factors: usize,
    vector_count: usize,
    maximum_live_entries: usize,
}

/// Replays one side of Lasso's challenged factor vector. Public table rows
/// with zero multiplicity are omitted from both sides because their factors
/// are identical and cancel. The source retains only prefix counters and one
/// physical factor chunk.
struct Gpt2LassoFactorSource<'a, 'b, R, F> {
    stream: &'a mut Gpt2LookupQueryStream<'b, R>,
    registry: &'a Gpt2LookupRegistry,
    multiplicities: &'a [(usize, u64)],
    layout: BvaLayout,
    chunk_len: usize,
    expected_queries: usize,
    query_vector_count: usize,
    row_challenge: F,
    product_challenge: F,
    side: LassoFactorSide,
    profile: LassoFactorStreamProfile,
}

#[derive(Clone, Copy)]
enum LassoAffineComponent {
    Value,
    QueryValue,
    TableValue,
    Counter,
    QueryCounter,
    TableCounter,
}

struct Gpt2LassoAffineComponentSource<'s, 'a, 'b, R, F> {
    source: &'s mut Gpt2LassoFactorSource<'a, 'b, R, F>,
    component: LassoAffineComponent,
}

impl<'s, 'a, 'b, R, F> Gpt2LassoAffineComponentSource<'s, 'a, 'b, R, F> {
    fn new(
        source: &'s mut Gpt2LassoFactorSource<'a, 'b, R, F>,
        component: LassoAffineComponent,
    ) -> Self {
        Self { source, component }
    }
}

impl<'a, 'b, R: Read + Seek, F: PrimeField> Gpt2LassoTimestampSource<'a, 'b, R, F> {
    fn new(
        stream: &'a mut Gpt2LookupQueryStream<'b, R>,
        registry: &'a Gpt2LookupRegistry,
        expected_queries: usize,
        chunk_len: usize,
    ) -> Result<Self, Gpt2CheckpointError> {
        if expected_queries == 0 || !chunk_len.is_power_of_two() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        // Decompose the physical stream into powers of two so each vector can
        // be fed directly to the existing range proof.  The tail is split by
        // its binary expansion instead of being padded to `chunk_len`, hence
        // no synthetic zero enters either the commitment or an inner product.
        let mut remaining = expected_queries;
        let mut lengths = Vec::new();
        while remaining != 0 {
            let length = if remaining >= chunk_len {
                chunk_len
            } else {
                1_usize << remaining.ilog2()
            };
            lengths.push(length);
            remaining -= length;
        }
        Ok(Self {
            stream,
            registry,
            layout: BvaLayout::from_lengths(&lengths)?,
            chunk_len,
            expected_queries,
            profile: LassoTimestampStreamProfile::default(),
            _field: PhantomData,
        })
    }

    fn profile(&self) -> LassoTimestampStreamProfile {
        self.profile
    }

    fn for_each_i32_vector(
        &mut self,
        visit: &mut dyn FnMut(usize, &[i32]),
    ) -> Result<(), Gpt2CheckpointError> {
        let mut counts = HashMap::<usize, u64>::new();
        let mut buffer = Vec::with_capacity(self.chunk_len);
        let mut vector_index = 0_usize;
        let mut query_count = 0_usize;
        let mut maximum_timestamp = 0_u64;
        let registry = *self.registry;
        let vector_lengths = self.layout.vector_lengths().to_vec();
        self.stream.for_each(&mut |query| {
            let table_index = registry.table_index(query)?;
            let timestamp = *counts.get(&table_index).unwrap_or(&0);
            maximum_timestamp = maximum_timestamp.max(timestamp);
            buffer.push(
                i32::try_from(timestamp).map_err(|_| Gpt2CheckpointError::ProtocolShapeMismatch)?,
            );
            query_count = query_count
                .checked_add(1)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
            let next = timestamp
                .checked_add(1)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
            counts.insert(table_index, next);
            let expected_len = vector_lengths
                .get(vector_index)
                .copied()
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            if buffer.len() == expected_len {
                visit(vector_index, &buffer);
                vector_index += 1;
                buffer.clear();
            }
            Ok(())
        })?;
        if query_count != self.expected_queries
            || vector_index != vector_lengths.len()
            || !buffer.is_empty()
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        self.profile = LassoTimestampStreamProfile {
            query_count,
            vector_count: vector_index,
            maximum_live_entries: self.chunk_len.min(query_count),
            maximum_timestamp,
        };
        Ok(())
    }

    fn evaluate_packed(&mut self, xl: &[F], xr: &[F]) -> Result<F, Gpt2CheckpointError> {
        let logical_entries = self.layout.logical_len();
        let packed_cols = 1 << (logical_entries.ilog2() as usize / 2);
        let packed_rows = logical_entries / packed_cols;
        if xl.len() != packed_rows.ilog2() as usize || xr.len() != packed_cols.ilog2() as usize {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let row_weights = crate::utils::xi::xi_from_challenges(&xl.to_vec());
        let column_weights = crate::utils::xi::xi_from_challenges(&xr.to_vec());
        let bucket_offsets = self.layout.bucket_offsets();
        let vector_to_bucket = self.layout.vector_to_bucket().to_vec();
        let mut value = F::from(0_u64);
        <Self as BvaFieldSource<F>>::for_each_vector(self, &mut |vector_index, values| {
            let offset = bucket_offsets[vector_to_bucket[vector_index]];
            for (local_index, entry) in values.iter().enumerate() {
                let flat_index = offset + local_index;
                value += *entry
                    * row_weights[flat_index / packed_cols]
                    * column_weights[flat_index % packed_cols];
            }
        })?;
        Ok(value)
    }
}

impl<R: Read + Seek, F: PrimeField> BvaFieldSource<F> for Gpt2LassoTimestampSource<'_, '_, R, F> {
    type Error = Gpt2CheckpointError;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[F])) -> Result<(), Self::Error> {
        self.for_each_i32_vector(&mut |vector_index, values| {
            let field_values = values.iter().copied().map(F::from).collect::<Vec<_>>();
            visit(vector_index, &field_values);
        })
    }
}

impl<R: Read + Seek, F: PrimeField> BvaI32Source for Gpt2LassoTimestampSource<'_, '_, R, F> {
    type Error = Gpt2CheckpointError;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[i32])) -> Result<(), Self::Error> {
        self.for_each_i32_vector(visit)
    }
}

impl<'a, 'b, R: Read + Seek, F: PrimeField> Gpt2LassoFactorSource<'a, 'b, R, F> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        stream: &'a mut Gpt2LookupQueryStream<'b, R>,
        registry: &'a Gpt2LookupRegistry,
        products: &'a StreamingLassoProducts<F>,
        row_challenge: F,
        product_challenge: F,
        chunk_len: usize,
        side: LassoFactorSide,
    ) -> Result<Self, Gpt2CheckpointError> {
        if products.query_count == 0
            || products.multiplicities.is_empty()
            || !chunk_len.is_power_of_two()
            || row_challenge.is_zero()
            || product_challenge.is_zero()
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let mut lengths = Vec::new();
        let mut remaining = products.query_count;
        while remaining != 0 {
            let length = if remaining >= chunk_len {
                chunk_len
            } else {
                1_usize << remaining.ilog2()
            };
            lengths.push(length);
            remaining -= length;
        }
        let query_vector_count = lengths.len();
        let mut remaining = products.multiplicities.len();
        while remaining != 0 {
            let length = if remaining >= chunk_len {
                chunk_len
            } else {
                1_usize << remaining.ilog2()
            };
            lengths.push(length);
            remaining -= length;
        }
        Ok(Self {
            stream,
            registry,
            multiplicities: &products.multiplicities,
            layout: BvaLayout::from_lengths(&lengths)?,
            chunk_len,
            expected_queries: products.query_count,
            query_vector_count,
            row_challenge,
            product_challenge,
            side,
            profile: LassoFactorStreamProfile::default(),
        })
    }

    fn profile(&self) -> LassoFactorStreamProfile {
        self.profile
    }

    fn for_each_affine_vector(
        &mut self,
        visit: &mut dyn FnMut(usize, &[F], &[F], &[F]),
    ) -> Result<(), Gpt2CheckpointError> {
        let mut counts = HashMap::<usize, u64>::new();
        let mut factor_buffer = Vec::with_capacity(self.chunk_len);
        let mut value_buffer = Vec::with_capacity(self.chunk_len);
        let mut counter_buffer = Vec::with_capacity(self.chunk_len);
        let mut vector_index = 0_usize;
        let mut query_factors = 0_usize;
        let vector_lengths = self.layout.vector_lengths().to_vec();
        let registry = *self.registry;
        let row_challenge = self.row_challenge;
        let product_challenge = self.product_challenge;
        let product_challenge_squared = product_challenge * product_challenge;
        let side = self.side;
        let mut push_entry = |value: F, counter: F| -> Result<(), Gpt2CheckpointError> {
            factor_buffer
                .push(F::one() + product_challenge * value + product_challenge_squared * counter);
            value_buffer.push(value);
            counter_buffer.push(counter);
            let expected_len = vector_lengths
                .get(vector_index)
                .copied()
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            if factor_buffer.len() == expected_len {
                visit(vector_index, &factor_buffer, &value_buffer, &counter_buffer);
                vector_index += 1;
                factor_buffer.clear();
                value_buffer.clear();
                counter_buffer.clear();
            }
            Ok(())
        };
        self.stream.for_each(&mut |query| {
            registry.audit_query(query)?;
            let compressed = compress_lookup_query(query, row_challenge);
            let table_index = registry.table_index(query)?;
            let count = counts.entry(table_index).or_insert(0);
            let timestamp = match side {
                LassoFactorSide::Left => *count,
                LassoFactorSide::Right => count
                    .checked_add(1)
                    .ok_or(Gpt2CheckpointError::OffsetOverflow)?,
            };
            push_entry(compressed, F::from(timestamp))?;
            *count = count
                .checked_add(1)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
            query_factors = query_factors
                .checked_add(1)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
            Ok(())
        })?;
        if query_factors != self.expected_queries {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }

        let mut previous_index = None;
        for &(table_index, multiplicity) in self.multiplicities {
            if table_index >= registry.public_table_rows()?
                || multiplicity == 0
                || previous_index.is_some_and(|previous| previous >= table_index)
                || counts.get(&table_index).copied() != Some(multiplicity)
            {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            previous_index = Some(table_index);
            let compressed = compress_lookup_query(registry.table_row(table_index)?, row_challenge);
            let table_count = match side {
                LassoFactorSide::Left => multiplicity,
                LassoFactorSide::Right => 0,
            };
            push_entry(compressed, F::from(table_count))?;
        }
        drop(push_entry);
        if vector_index != vector_lengths.len()
            || !factor_buffer.is_empty()
            || !value_buffer.is_empty()
            || !counter_buffer.is_empty()
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        self.profile = LassoFactorStreamProfile {
            query_factors,
            table_factors: self.multiplicities.len(),
            vector_count: vector_index,
            maximum_live_entries: self
                .chunk_len
                .min(query_factors + self.multiplicities.len()),
        };
        Ok(())
    }
}

impl<R: Read + Seek, F: PrimeField> BvaFieldSource<F> for Gpt2LassoFactorSource<'_, '_, R, F> {
    type Error = Gpt2CheckpointError;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[F])) -> Result<(), Self::Error> {
        self.for_each_affine_vector(&mut |vector_index, factor, _, _| {
            visit(vector_index, factor);
        })
    }
}

impl<R: Read + Seek, F: PrimeField> BvaAffineRelationSource<F>
    for Gpt2LassoFactorSource<'_, '_, R, F>
{
    type Error = Gpt2CheckpointError;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_vector(
        &mut self,
        visit: &mut dyn FnMut(usize, &[F], &[F], &[F]),
    ) -> Result<(), Self::Error> {
        self.for_each_affine_vector(visit)
    }
}

impl<R: Read + Seek, F: PrimeField> BvaFieldSource<F>
    for Gpt2LassoAffineComponentSource<'_, '_, '_, R, F>
{
    type Error = Gpt2CheckpointError;

    fn layout(&self) -> &BvaLayout {
        &self.source.layout
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[F])) -> Result<(), Self::Error> {
        let component = self.component;
        let query_vector_count = self.source.query_vector_count;
        self.source
            .for_each_affine_vector(&mut |vector_index, _, value, counter| {
                let zeros;
                let selected = match component {
                    LassoAffineComponent::Value => value,
                    LassoAffineComponent::QueryValue if vector_index < query_vector_count => value,
                    LassoAffineComponent::TableValue if vector_index >= query_vector_count => value,
                    LassoAffineComponent::Counter => counter,
                    LassoAffineComponent::QueryCounter if vector_index < query_vector_count => {
                        counter
                    }
                    LassoAffineComponent::TableCounter if vector_index >= query_vector_count => {
                        counter
                    }
                    LassoAffineComponent::QueryValue
                    | LassoAffineComponent::TableValue
                    | LassoAffineComponent::QueryCounter
                    | LassoAffineComponent::TableCounter => {
                        zeros = vec![F::from(0_u64); counter.len()];
                        &zeros
                    }
                };
                visit(vector_index, selected);
            })
    }
}

/// Routes the query prefix of the right Lasso counter vector to the canonical
/// timestamp commitment. Query counters are `timestamp + 1`; table-tail
/// counters are public zero and therefore have no target route.
fn lasso_query_timestamp_routes<F: PrimeField>(
    factor_layout: &BvaLayout,
    timestamp_layout: &BvaLayout,
    query_constant: F,
) -> Result<(Vec<Option<(usize, F)>>, Vec<F>), Gpt2CheckpointError> {
    let timestamp_vectors = timestamp_layout.vector_lengths().len();
    if factor_layout.vector_lengths().len() < timestamp_vectors
        || &factor_layout.vector_lengths()[..timestamp_vectors] != timestamp_layout.vector_lengths()
    {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    let mut routes = Vec::with_capacity(factor_layout.vector_lengths().len());
    let mut constants = Vec::with_capacity(factor_layout.vector_lengths().len());
    for vector_index in 0..factor_layout.vector_lengths().len() {
        if vector_index < timestamp_vectors {
            routes.push(Some((vector_index, F::one())));
            constants.push(query_constant);
        } else {
            routes.push(None);
            constants.push(F::zero());
        }
    }
    Ok((routes, constants))
}

impl<F: PrimeField> Gpt2LassoPublicTableComponentSource<F> {
    fn new(
        products: &StreamingLassoProducts<F>,
        registry: &Gpt2LookupRegistry,
        factor_layout: &BvaLayout,
        row_challenge: F,
        component: LassoPublicTableComponent,
    ) -> Result<Self, Gpt2CheckpointError> {
        let _validated_multiplicities = Gpt2LassoMultiplicitySource::<F>::new(products)?;
        if row_challenge == F::from(0_u64)
            || products.query_count == 0
            || products.multiplicities.is_empty()
            || products.public_table_rows != registry.public_table_rows()?
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let mut query_entries = 0_usize;
        let mut query_vector_count = 0_usize;
        while query_entries < products.query_count {
            let length = *factor_layout
                .vector_lengths()
                .get(query_vector_count)
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            query_entries = query_entries
                .checked_add(length)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
            query_vector_count += 1;
        }
        if query_entries != products.query_count {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }

        let table_lengths = &factor_layout.vector_lengths()[query_vector_count..];
        if table_lengths.iter().sum::<usize>() != products.multiplicities.len() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let mut entries = Vec::with_capacity(products.multiplicities.len());
        let mut table_cursor = 0_usize;
        for (tail_vector, &length) in table_lengths.iter().enumerate() {
            let vector_index = query_vector_count + tail_vector;
            for entry_index in 0..length {
                let &(table_index, count) = products
                    .multiplicities
                    .get(table_cursor)
                    .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                let value = match component {
                    LassoPublicTableComponent::Value => {
                        compress_lookup_query(registry.table_row(table_index)?, row_challenge)
                    }
                    LassoPublicTableComponent::Counter => F::from(count),
                };
                if value != F::from(0_u64) {
                    entries.push((vector_index, entry_index, value));
                }
                table_cursor += 1;
            }
        }
        if table_cursor != products.multiplicities.len() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        Ok(Self {
            layout: factor_layout.clone(),
            entries,
        })
    }
}

impl<F: PrimeField> SparseBvaFieldSource<F> for Gpt2LassoPublicTableComponentSource<F> {
    type Error = Gpt2CheckpointError;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_nonzero(
        &mut self,
        visit: &mut dyn FnMut(usize, usize, F),
    ) -> Result<(), Self::Error> {
        for &(vector_index, entry_index, value) in &self.entries {
            visit(vector_index, entry_index, value);
        }
        Ok(())
    }
}

impl<F: PrimeField> Gpt2LassoPublicTableBinding<F> {
    fn new(
        products: &StreamingLassoProducts<F>,
        multiplicity_hiding: F,
        table_value_hiding: F,
        table_counter_hiding: F,
    ) -> Result<Self, Gpt2CheckpointError> {
        let _validated = Gpt2LassoMultiplicitySource::<F>::new(products)?;
        Ok(Self {
            multiplicities: products.multiplicities.clone(),
            multiplicity_hiding,
            table_value_hiding,
            table_counter_hiding,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn verify<E: ark_ec::pairing::Pairing<ScalarField = F>>(
        &self,
        pp: &ark_poly_commit::smart_pc::data_structures::UniversalParams<E>,
        registry: &Gpt2LookupRegistry,
        factor_layout: &BvaLayout,
        query_count: usize,
        public_table_rows: usize,
        row_challenge: F,
        multiplicity_commitment: ark_ec::pairing::PairingOutput<E>,
        table_value_commitment: ark_ec::pairing::PairingOutput<E>,
        table_counter_commitment: ark_ec::pairing::PairingOutput<E>,
    ) -> Result<Gpt2LassoPublicTableBindingProfile, Gpt2CheckpointError> {
        let products = StreamingLassoProducts {
            left: F::from(0_u64),
            right: F::from(0_u64),
            query_count,
            distinct_table_rows: self.multiplicities.len(),
            public_table_rows,
            maximum_live_semantic_entries: 0,
            multiplicities: self.multiplicities.clone(),
        };
        let mut multiplicity_source = Gpt2LassoMultiplicitySource::<F>::new(&products)?;
        let (rebuilt_multiplicity, _, multiplicity_profile) =
            crate::protocols::virtualpcs::commit_sparse_bva_field_streaming::<E, _>(
                pp,
                &mut multiplicity_source,
                self.multiplicity_hiding,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolShapeMismatch)?;
        let mut table_value_source = Gpt2LassoPublicTableComponentSource::new(
            &products,
            registry,
            factor_layout,
            row_challenge,
            LassoPublicTableComponent::Value,
        )?;
        let (rebuilt_table_value, _, table_value_profile) =
            crate::protocols::virtualpcs::commit_sparse_bva_field_streaming::<E, _>(
                pp,
                &mut table_value_source,
                self.table_value_hiding,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolShapeMismatch)?;
        let mut table_counter_source = Gpt2LassoPublicTableComponentSource::new(
            &products,
            registry,
            factor_layout,
            row_challenge,
            LassoPublicTableComponent::Counter,
        )?;
        let (rebuilt_table_counter, _, table_counter_profile) =
            crate::protocols::virtualpcs::commit_sparse_bva_field_streaming::<E, _>(
                pp,
                &mut table_counter_source,
                self.table_counter_hiding,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolShapeMismatch)?;
        if rebuilt_multiplicity != multiplicity_commitment
            || rebuilt_table_value != table_value_commitment
            || rebuilt_table_counter != table_counter_commitment
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        Ok(Gpt2LassoPublicTableBindingProfile {
            support_entries: self.multiplicities.len(),
            nonzero_group_contributions: multiplicity_profile.nonzero_entries
                + table_value_profile.nonzero_entries
                + table_counter_profile.nonzero_entries,
            materialized_padding_entries: multiplicity_profile.materialized_padding_entries
                + table_value_profile.materialized_padding_entries
                + table_counter_profile.materialized_padding_entries,
            padding_group_operations: multiplicity_profile.padding_group_operations
                + table_value_profile.padding_group_operations
                + table_counter_profile.padding_group_operations,
        })
    }

    fn serialized_size_bytes(&self) -> usize {
        self.multiplicities.len() * (8 + 8)
            + self.multiplicity_hiding.serialized_size(Compress::Yes)
            + self.table_value_hiding.serialized_size(Compress::Yes)
            + self.table_counter_hiding.serialized_size(Compress::Yes)
    }
}

impl<F: PrimeField> Gpt2LassoMultiplicitySource<F> {
    fn new(products: &StreamingLassoProducts<F>) -> Result<Self, Gpt2CheckpointError> {
        let maximum_count = u64::try_from(products.query_count)
            .map_err(|_| Gpt2CheckpointError::ArithmeticOverflow)?;
        let mut previous_index = None;
        let mut total_count = 0_u64;
        for &(index, count) in &products.multiplicities {
            if index >= products.public_table_rows
                || count == 0
                || count > maximum_count
                || previous_index.is_some_and(|previous| previous >= index)
            {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            previous_index = Some(index);
            total_count = total_count
                .checked_add(count)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        }
        if total_count != maximum_count {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        Ok(Self {
            layout: BvaLayout::from_lengths(&[products.public_table_rows])?,
            entries: products
                .multiplicities
                .iter()
                .map(|(index, count)| (*index, *count))
                .collect(),
            _field: PhantomData,
        })
    }

    fn evaluate_packed(&self, xl: &[F], xr: &[F]) -> Result<F, Gpt2CheckpointError> {
        let logical_entries = self.layout.logical_len();
        let packed_cols = 1 << (logical_entries.ilog2() as usize / 2);
        let packed_rows = logical_entries / packed_cols;
        if xl.len() != packed_rows.ilog2() as usize || xr.len() != packed_cols.ilog2() as usize {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let row_weights = crate::utils::xi::xi_from_challenges(&xl.to_vec());
        let column_weights = crate::utils::xi::xi_from_challenges(&xr.to_vec());
        Ok(self
            .entries
            .iter()
            .fold(F::from(0_u64), |sum, (index, value)| {
                sum + F::from(*value)
                    * row_weights[*index / packed_cols]
                    * column_weights[*index % packed_cols]
            }))
    }
}

impl<F: PrimeField> SparseBvaFieldSource<F> for Gpt2LassoMultiplicitySource<F> {
    type Error = Gpt2CheckpointError;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_nonzero(
        &mut self,
        visit: &mut dyn FnMut(usize, usize, F),
    ) -> Result<(), Self::Error> {
        for &(index, value) in &self.entries {
            visit(0, index, F::from(value));
        }
        Ok(())
    }
}

impl<F: PrimeField> crate::protocols::vector_relation::SparseBvaU64Source
    for Gpt2LassoMultiplicitySource<F>
{
    type Error = Gpt2CheckpointError;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_nonzero_u64(
        &mut self,
        visit: &mut dyn FnMut(usize, usize, u64),
    ) -> Result<(), Self::Error> {
        for &(index, value) in &self.entries {
            visit(0, index, value);
        }
        Ok(())
    }
}

/// Canonical byte-limb encoding used by the range-lookup layer. Signed values
/// are shifted by `2^(bits - 1)`, so the same unsigned lookup tables prove the
/// exact interval `[-2^(bits - 1), 2^(bits - 1) - 1]` without a separate sign
/// witness or a special case for the most-negative value.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RangeEncoding {
    signed: bool,
    value_bits: u16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RangeStreamProfile {
    source_entries: usize,
    emitted_limb_entries: usize,
    maximum_live_limb_entries: usize,
}

struct RangeLimbStream<'a> {
    descriptor: &'a AuxiliaryBlockDescriptor,
    block: &'a AuxiliaryWitnessBlock,
    encoding: RangeEncoding,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum PrimitiveBvaVectorSource {
    Base { block: GlobalBlockId, column: usize },
    Auxiliary { block_index: usize, column: usize },
    RangeLimb { block_index: usize, column: usize },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Gpt2PrimitiveBvaLayout {
    layout: BvaLayout,
    vectors: Vec<PrimitiveBvaVectorSource>,
    physical_entries: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ElementwiseBilinearVector {
    left_block: GlobalBlockId,
    left_column: usize,
    left_vector: usize,
    right_block: GlobalBlockId,
    right_column: usize,
    right_vector: usize,
    right_scalar_index: Option<usize>,
    right_segmented: bool,
    right_scale: i128,
    right_subtract: [Option<(GlobalBlockId, usize, usize)>; 2],
    right_constant: i128,
    output_vector: Option<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ElementwiseBilinearRouting {
    layout: BvaLayout,
    vectors: Vec<ElementwiseBilinearVector>,
    attention_segments: Vec<usize>,
    covered_relations: usize,
}

struct Gpt2ElementwisePairSource<'a, R, F> {
    checkpoint: &'a mut Gpt2Q12Checkpoint<R>,
    base: &'a Gpt2GlobalLayout,
    activations: &'a Gpt2ActivationWitness,
    witness: &'a Gpt2PrimitiveWitness,
    routing: &'a ElementwiseBilinearRouting,
    right_coefficients: &'a TensorBasisVector<F>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DenseMatmulVector {
    input_block: GlobalBlockId,
    input_columns: Vec<usize>,
    input_vectors: Vec<usize>,
    weight_block: GlobalBlockId,
    weight_vectors: Vec<usize>,
    output_vectors: Vec<usize>,
    inner: usize,
    output_rows: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DenseMatmulRouting {
    layout: BvaLayout,
    vectors: Vec<DenseMatmulVector>,
    covered_relations: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DenseMatmulProjection<F> {
    row_challenges: Vec<F>,
    row_coefficients: Vec<F>,
    column_coefficients: Vec<F>,
    batch_weight: F,
}

struct Gpt2DenseMatmulPairSource<'a, R, F> {
    checkpoint: &'a mut Gpt2Q12Checkpoint<R>,
    base: &'a Gpt2GlobalLayout,
    activations: &'a Gpt2ActivationWitness,
    witness: &'a Gpt2PrimitiveWitness,
    routing: &'a DenseMatmulRouting,
    projections: &'a [DenseMatmulProjection<F>],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CausalQkTerm {
    qkv_block: GlobalBlockId,
    query_column: usize,
    query_vector: usize,
    query_start: usize,
    key_column: usize,
    key_vector: usize,
    key_start: usize,
    relation_index: usize,
    output_index: usize,
    head_width: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CausalQkRelation {
    output_vector: usize,
    output_len: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CausalQkRouting {
    layout: BvaLayout,
    terms: Vec<CausalQkTerm>,
    relations: Vec<CausalQkRelation>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CausalQkProjection<F> {
    output_challenges: Vec<F>,
    output_coefficients: Vec<F>,
    batch_weight: F,
}

struct Gpt2CausalQkPairSource<'a, F> {
    base: &'a Gpt2GlobalLayout,
    activations: &'a Gpt2ActivationWitness,
    routing: &'a CausalQkRouting,
    projections: &'a [CausalQkProjection<F>],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CausalPvTerm {
    probability_block: GlobalBlockId,
    probability_vector: usize,
    probability_start: usize,
    qkv_block: GlobalBlockId,
    value_index: usize,
    relation_index: usize,
    output_row: usize,
    output_column: usize,
    inner: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CausalPvRelation {
    output_vectors: Vec<usize>,
    qkv_vectors: Vec<usize>,
    output_rows: usize,
    output_columns: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CausalPvRouting {
    layout: BvaLayout,
    terms: Vec<CausalPvTerm>,
    relations: Vec<CausalPvRelation>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CausalPvProjection<F> {
    row_challenges: Vec<F>,
    row_coefficients: Vec<F>,
    column_coefficients: Vec<F>,
    batch_weight: F,
}

struct Gpt2CausalPvPairSource<'a, F> {
    base: &'a Gpt2GlobalLayout,
    activations: &'a Gpt2ActivationWitness,
    witness: &'a Gpt2PrimitiveWitness,
    routing: &'a CausalPvRouting,
    projections: &'a [CausalPvProjection<F>],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TopLevelEquationFamily {
    Io,
    Linear,
    Bilinear,
    TaggedLookup,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EquationRowSource {
    IoBlockColumn {
        block: GlobalBlockId,
        column: usize,
    },
    PrimitiveRelationChunk {
        relation_index: usize,
        scalar_offset: usize,
        scalar_len: usize,
    },
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Gpt2EquationGroupLayout {
    family: TopLevelEquationFamily,
    layout: BvaLayout,
    sources: Vec<EquationRowSource>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Gpt2GlobalEquationSchedule {
    groups: Vec<Gpt2EquationGroupLayout>,
}

impl RangeEncoding {
    fn new(signed: bool, value_bits: u16) -> Result<Self, Gpt2CheckpointError> {
        if value_bits == 0 || value_bits > 128 {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        Ok(Self { signed, value_bits })
    }

    const fn limb_count(self) -> usize {
        self.value_bits.div_ceil(8) as usize
    }

    fn limb_bits(self, limb: usize) -> Option<u8> {
        if limb >= self.limb_count() {
            return None;
        }
        let consumed = u16::try_from(limb).ok()?.checked_mul(8)?;
        Some((self.value_bits - consumed).min(8) as u8)
    }

    fn encode(self, value: i128, limbs: &mut Vec<u8>) -> Result<(), Gpt2CheckpointError> {
        let encoded = self.encoded_value(value)?;
        limbs.clear();
        limbs.reserve(self.limb_count());
        for limb in 0..self.limb_count() {
            limbs.push(((encoded >> (8 * limb)) & 0xff) as u8);
        }
        Ok(())
    }

    fn matches(self, value: i128, limbs: &[u8]) -> Result<bool, Gpt2CheckpointError> {
        if limbs.len() != self.limb_count() {
            return Ok(false);
        }
        let mut recomposed = 0_u128;
        for (limb, byte) in limbs.iter().copied().enumerate() {
            let bits = self
                .limb_bits(limb)
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            if bits < 8 && u16::from(byte) >= (1_u16 << bits) {
                return Ok(false);
            }
            recomposed |= u128::from(byte) << (8 * limb);
        }
        Ok(recomposed == self.encoded_value(value)?)
    }

    fn encoded_value(self, value: i128) -> Result<u128, Gpt2CheckpointError> {
        if self.signed {
            if self.value_bits == 128 {
                return Ok((value as u128).wrapping_add(1_u128 << 127));
            }
            let signed_limit = 1_i128 << (self.value_bits - 1);
            if value < -signed_limit || value >= signed_limit {
                return Err(Gpt2CheckpointError::DeclaredRangeViolation {
                    block: usize::MAX,
                    value_bits: self.value_bits,
                });
            }
            Ok((value + signed_limit) as u128)
        } else {
            if value < 0
                || (self.value_bits < 128 && (value as u128) >= (1_u128 << self.value_bits))
            {
                return Err(Gpt2CheckpointError::DeclaredRangeViolation {
                    block: usize::MAX,
                    value_bits: self.value_bits,
                });
            }
            Ok(value as u128)
        }
    }
}

impl<'a> RangeLimbStream<'a> {
    fn new(
        descriptor: &'a AuxiliaryBlockDescriptor,
        block: &'a AuxiliaryWitnessBlock,
    ) -> Result<Self, Gpt2CheckpointError> {
        if descriptor.id != block.id
            || block.columns.len() != descriptor.cols
            || block
                .columns
                .iter()
                .any(|column| column.len() != descriptor.rows)
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        Ok(Self {
            descriptor,
            block,
            encoding: RangeEncoding::new(
                auxiliary_kind_is_signed(descriptor.kind),
                descriptor.value_bits,
            )?,
        })
    }

    fn for_each_limb_column(
        &self,
        visit: &mut dyn FnMut(usize, usize, u8, &[u8]),
    ) -> Result<RangeStreamProfile, Gpt2CheckpointError> {
        let mut buffer = vec![0_u8; self.descriptor.rows];
        for (column_index, column) in self.block.columns.iter().enumerate() {
            for limb in 0..self.encoding.limb_count() {
                for (encoded_limb, value) in buffer.iter_mut().zip(column) {
                    let encoded = self.encoding.encoded_value(*value)?;
                    *encoded_limb = ((encoded >> (8 * limb)) & 0xff) as u8;
                }
                visit(
                    column_index,
                    limb,
                    self.encoding
                        .limb_bits(limb)
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?,
                    &buffer,
                );
            }
        }
        let source_entries = self
            .descriptor
            .rows
            .checked_mul(self.descriptor.cols)
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        Ok(RangeStreamProfile {
            source_entries,
            emitted_limb_entries: source_entries
                .checked_mul(self.encoding.limb_count())
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?,
            maximum_live_limb_entries: self.descriptor.rows,
        })
    }
}

struct Gpt2PrimitiveBvaFieldSource<'a, R, F: PrimeField> {
    checkpoint: &'a mut Gpt2Q12Checkpoint<R>,
    layout: &'a Gpt2GlobalLayout,
    activations: &'a Gpt2ActivationWitness,
    manifest: &'a Gpt2PrimitiveRelationManifest,
    witness: &'a Gpt2PrimitiveWitness,
    bva: &'a Gpt2PrimitiveBvaLayout,
    _field: PhantomData<F>,
}

/// Deterministic compiler output for the three operation families.  It is not
/// a vector of per-layer relations: every descriptor is later concatenated
/// into its family-wide selector/routing equation over the shared block IDs.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct Gpt2PrimitiveRelationManifest {
    base_block_count: usize,
    auxiliary_blocks: Vec<AuxiliaryBlockDescriptor>,
    range_limb_blocks: Vec<RangeLimbBlockDescriptor>,
    relations: Vec<PrimitiveRelationDescriptor>,
}

impl Gpt2PrimitiveRelationManifest {
    pub(super) fn compile(
        metadata: &Gpt2Q12Metadata,
        layout: &Gpt2GlobalLayout,
    ) -> Result<Self, Gpt2CheckpointError> {
        if *layout != Gpt2GlobalLayout::compile(metadata)? {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let operation_manifest = metadata.operation_manifest()?;
        if operation_manifest.len() != layout.operations().len() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let mut compiler = RelationCompiler {
            base_block_count: layout.witness_block_count(),
            auxiliary_blocks: Vec::new(),
            relations: Vec::new(),
        };
        for (operation_index, (descriptor, operation)) in operation_manifest
            .iter()
            .zip(layout.operations())
            .enumerate()
        {
            if descriptor.kind != operation.kind || descriptor.block != operation.block {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            match descriptor.kind {
                Gpt2Q12OperationKind::Embedding => {
                    compiler.compile_embedding(operation_index, layout)?
                }
                Gpt2Q12OperationKind::LayerNorm1
                | Gpt2Q12OperationKind::LayerNorm2
                | Gpt2Q12OperationKind::FinalLayerNorm => compiler.compile_layer_norm(
                    operation_index,
                    descriptor.rows,
                    descriptor.cols,
                    metadata.scale(),
                    layout,
                )?,
                Gpt2Q12OperationKind::QkvProjection
                | Gpt2Q12OperationKind::AttentionProjection
                | Gpt2Q12OperationKind::MlpExpansion
                | Gpt2Q12OperationKind::MlpProjection => compiler.compile_dense_linear(
                    operation_index,
                    descriptor.rows,
                    descriptor.cols,
                    metadata.scale(),
                    true,
                    false,
                    layout,
                )?,
                Gpt2Q12OperationKind::LmHead => compiler.compile_dense_linear(
                    operation_index,
                    descriptor.rows,
                    descriptor.cols,
                    metadata.scale(),
                    false,
                    true,
                    layout,
                )?,
                Gpt2Q12OperationKind::CausalAttention => {
                    compiler.compile_attention(operation_index, metadata, layout)?
                }
                Gpt2Q12OperationKind::AttentionResidual | Gpt2Q12OperationKind::MlpResidual => {
                    compiler.compile_residual(operation_index, layout)?
                }
                Gpt2Q12OperationKind::Gelu => {
                    compiler.compile_gelu(operation_index, metadata.scale(), layout)?
                }
            }
        }
        let range_limb_blocks = compiler.compile_range_obligations(layout)?;
        compiler
            .relations
            .sort_by_key(|relation| relation.operation_index);
        let manifest = Self {
            base_block_count: compiler.base_block_count,
            auxiliary_blocks: compiler.auxiliary_blocks,
            range_limb_blocks,
            relations: compiler.relations,
        };
        manifest.audit(layout)?;
        manifest.equation_schedule(layout)?;
        Ok(manifest)
    }

    fn audit(&self, layout: &Gpt2GlobalLayout) -> Result<(), Gpt2CheckpointError> {
        if self.base_block_count != layout.witness_block_count() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let total_blocks = self
            .base_block_count
            .checked_add(self.auxiliary_blocks.len())
            .and_then(|count| count.checked_add(self.range_limb_blocks.len()))
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        for (offset, block) in self.auxiliary_blocks.iter().enumerate() {
            if block.id.0 != self.base_block_count + offset
                || block.rows == 0
                || block.cols == 0
                || block.value_bits == 0
                || block.value_bits > 128
            {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
        }
        let mut range_sources = Vec::new();
        for (block_index, source) in layout.blocks().iter().enumerate() {
            let operation_index = match source.role {
                super::Gpt2GlobalBlockRole::Parameter(_) => 0,
                super::Gpt2GlobalBlockRole::PublicTokenIds => continue,
                super::Gpt2GlobalBlockRole::OperationOutput(operation_index) => operation_index,
            };
            range_sources.push((
                RangeLimbSource::Base { block_index },
                source.id,
                operation_index,
                source.rows,
                source.cols,
                true,
                32_u16,
            ));
        }
        for (block_index, source) in self.auxiliary_blocks.iter().enumerate() {
            range_sources.push((
                RangeLimbSource::Auxiliary { block_index },
                source.id,
                source.operation_index,
                source.rows,
                source.cols,
                auxiliary_kind_is_signed(source.kind),
                source.value_bits,
            ));
        }
        for (offset, block) in self.range_limb_blocks.iter().enumerate() {
            let (_, source, operation_index, rows, cols, signed, value_bits) = range_sources
                .iter()
                .find(|(kind, ..)| *kind == block.source_kind)
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            let encoding = RangeEncoding::new(*signed, *value_bits)?;
            if block.id.0 != self.base_block_count + self.auxiliary_blocks.len() + offset
                || block.source != *source
                || block.operation_index != *operation_index
                || block.rows != *rows
                || block.cols != *cols
                || block.limb_bits != encoding.limb_bits(block.limb).unwrap_or(0)
            {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
        }
        if self.relations.is_empty() {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        for (source_kind, source, operation_index, rows, cols, signed, value_bits) in range_sources
        {
            let limb_blocks = self
                .range_limb_blocks
                .iter()
                .filter(|block| block.source_kind == source_kind)
                .collect::<Vec<_>>();
            let encoding = RangeEncoding::new(signed, value_bits)?;
            if limb_blocks.len() != encoding.limb_count() {
                return Err(Gpt2CheckpointError::ProtocolIncomplete);
            }
            let mut expected_blocks = Vec::with_capacity(1 + limb_blocks.len());
            expected_blocks.push(source);
            expected_blocks.extend(limb_blocks.iter().map(|block| block.id));
            let recompositions = self
                .relations
                .iter()
                .filter(|relation| {
                    relation.family == RelationFamily::Linear
                        && relation.kind == LinearRelationKind::RangeRecomposition as u8
                        && relation.operation_index == operation_index
                        && relation.blocks == expected_blocks
                        && relation.scalar_count == rows * cols
                })
                .count();
            if recompositions != 1 {
                return Err(Gpt2CheckpointError::ProtocolIncomplete);
            }
            for limb in limb_blocks {
                let lookups = self
                    .relations
                    .iter()
                    .filter(|relation| {
                        relation.family == RelationFamily::TaggedLookup
                            && relation.kind == range_lookup_kind(limb.limb_bits) as u8
                            && relation.operation_index == operation_index
                            && relation.blocks == [limb.id]
                            && relation.scalar_count == limb.rows * limb.cols
                    })
                    .count();
                if lookups != 1 {
                    return Err(Gpt2CheckpointError::ProtocolIncomplete);
                }
            }
        }
        let mut referenced_auxiliary =
            vec![false; self.auxiliary_blocks.len() + self.range_limb_blocks.len()];
        let mut constrained_outputs = vec![false; layout.operations().len()];
        let mut family_present = [false; 3];
        let mut previous_operation = 0_usize;
        for (relation_index, relation) in self.relations.iter().enumerate() {
            if relation.scalar_count == 0
                || relation.operation_index >= layout.operations().len()
                || (relation_index > 0 && relation.operation_index < previous_operation)
            {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            previous_operation = relation.operation_index;
            family_present[usize::from(family_tag(relation.family))] = true;
            let output = layout.operations()[relation.operation_index].output;
            for block in &relation.blocks {
                if block.0 >= total_blocks {
                    return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                }
                if block.0 >= self.base_block_count {
                    referenced_auxiliary[block.0 - self.base_block_count] = true;
                }
                if *block == output {
                    constrained_outputs[relation.operation_index] = true;
                }
            }
        }
        if !family_present.iter().copied().all(|present| present)
            || !referenced_auxiliary
                .into_iter()
                .all(|referenced| referenced)
            || !constrained_outputs
                .into_iter()
                .all(|constrained| constrained)
        {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        Ok(())
    }

    fn compile_bva(
        &self,
        base: &Gpt2GlobalLayout,
    ) -> Result<Gpt2PrimitiveBvaLayout, Gpt2CheckpointError> {
        if self.base_block_count != base.witness_block_count() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let mut lengths = Vec::new();
        let mut vectors = Vec::new();
        let mut physical_entries = 0_usize;
        for block in base.blocks() {
            for column in 0..block.cols {
                lengths.push(block.rows);
                vectors.push(PrimitiveBvaVectorSource::Base {
                    block: block.id,
                    column,
                });
                physical_entries = physical_entries
                    .checked_add(block.rows)
                    .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
            }
        }
        for (block_index, block) in self.auxiliary_blocks.iter().enumerate() {
            for column in 0..block.cols {
                lengths.push(block.rows);
                vectors.push(PrimitiveBvaVectorSource::Auxiliary {
                    block_index,
                    column,
                });
                physical_entries = physical_entries
                    .checked_add(block.rows)
                    .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
            }
        }
        for (block_index, block) in self.range_limb_blocks.iter().enumerate() {
            for column in 0..block.cols {
                lengths.push(block.rows);
                vectors.push(PrimitiveBvaVectorSource::RangeLimb {
                    block_index,
                    column,
                });
                physical_entries = physical_entries
                    .checked_add(block.rows)
                    .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
            }
        }
        Ok(Gpt2PrimitiveBvaLayout {
            layout: BvaLayout::from_lengths(&lengths)?,
            vectors,
            physical_entries,
        })
    }

    fn equation_schedule(
        &self,
        base: &Gpt2GlobalLayout,
    ) -> Result<Gpt2GlobalEquationSchedule, Gpt2CheckpointError> {
        let schedule = Gpt2GlobalEquationSchedule::compile(base, self)?;
        schedule.audit(base, self)?;
        Ok(schedule)
    }

    fn range_recomposition_tbv<F: PrimeField>(
        &self,
        bva: &Gpt2PrimitiveBvaLayout,
        mut next_challenge: impl FnMut() -> F,
    ) -> Result<(TensorBasisVector<F>, F), Gpt2CheckpointError> {
        let block_vectors = self.bva_block_vectors(bva)?;

        let mut etas = vec![F::from(0_u64); bva.layout.bucket_sizes().len()];
        let mut challenges = bva
            .layout
            .bucket_sizes()
            .iter()
            .map(|size| vec![F::from(1_u64); size.ilog2() as usize])
            .collect::<Vec<_>>();
        let mut assigned = vec![false; bva.layout.bucket_sizes().len()];
        let mut public_claim = F::from(0_u64);
        let mut group_start = 0_usize;
        while group_start < self.range_limb_blocks.len() {
            let source_kind = self.range_limb_blocks[group_start].source_kind;
            let mut group_end = group_start + 1;
            while group_end < self.range_limb_blocks.len()
                && self.range_limb_blocks[group_end].source_kind == source_kind
            {
                group_end += 1;
            }
            let limb_blocks = &self.range_limb_blocks[group_start..group_end];
            let first = limb_blocks
                .first()
                .ok_or(Gpt2CheckpointError::ProtocolIncomplete)?;
            let (signed, value_bits) = match source_kind {
                RangeLimbSource::Base { .. } => (true, 32_u16),
                RangeLimbSource::Auxiliary { block_index } => {
                    let descriptor = self
                        .auxiliary_blocks
                        .get(block_index)
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                    (
                        auxiliary_kind_is_signed(descriptor.kind),
                        descriptor.value_bits,
                    )
                }
            };
            let bias = if signed {
                F::from_le_bytes_mod_order(&(1_u128 << (value_bits - 1)).to_le_bytes())
            } else {
                F::from(0_u64)
            };
            for column in 0..first.cols {
                let source_vector = *block_vectors
                    .get(first.source.0)
                    .and_then(|columns| columns.get(column))
                    .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                let source_bucket = bva.layout.vector_to_bucket()[source_vector];
                let row_challenges = (0..bva.layout.bucket_sizes()[source_bucket].ilog2())
                    .map(|_| next_challenge())
                    .collect::<Vec<_>>();
                let batch_weight = next_challenge();
                assign_tbv_bucket(
                    &mut etas,
                    &mut challenges,
                    &mut assigned,
                    source_bucket,
                    batch_weight,
                    &row_challenges,
                )?;
                let mut limb_coefficient = F::from(1_u64);
                for limb in limb_blocks {
                    let limb_vector = *block_vectors
                        .get(limb.id.0)
                        .and_then(|columns| columns.get(column))
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                    let limb_bucket = bva.layout.vector_to_bucket()[limb_vector];
                    if bva.layout.bucket_sizes()[limb_bucket]
                        != bva.layout.bucket_sizes()[source_bucket]
                    {
                        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                    }
                    assign_tbv_bucket(
                        &mut etas,
                        &mut challenges,
                        &mut assigned,
                        limb_bucket,
                        -batch_weight * limb_coefficient,
                        &row_challenges,
                    )?;
                    limb_coefficient *= F::from(256_u64);
                }
                if signed {
                    public_claim -=
                        batch_weight * bias * tensor_prefix_sum(first.rows, &row_challenges)?;
                }
            }
            group_start = group_end;
        }
        let buckets = etas
            .into_iter()
            .zip(challenges)
            .map(|(eta, challenges)| TbvBucket::new(eta, challenges))
            .collect();
        Ok((TensorBasisVector::new(&bva.layout, buckets)?, public_claim))
    }

    fn residual_add_tbv<F: PrimeField>(
        &self,
        base: &Gpt2GlobalLayout,
        bva: &Gpt2PrimitiveBvaLayout,
        mut next_challenge: impl FnMut() -> F,
    ) -> Result<(TensorBasisVector<F>, F), Gpt2CheckpointError> {
        let block_vectors = self.bva_block_vectors(bva)?;
        let mut terms = vec![Vec::<(F, Vec<F>)>::new(); bva.layout.bucket_sizes().len()];
        let mut relation_count = 0_usize;
        for relation in &self.relations {
            if relation.family != RelationFamily::Linear
                || relation.kind != LinearRelationKind::ResidualAdd as u8
            {
                continue;
            }
            relation_count += 1;
            if relation.blocks.len() != 3 {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            let shape = relation.blocks.iter().try_fold(None, |expected, block| {
                let actual = self.block_shape(base, *block)?;
                if expected.is_some_and(|expected| expected != actual) {
                    return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                }
                Ok::<_, Gpt2CheckpointError>(Some(actual))
            })?;
            let (rows, cols) = shape.ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            if rows
                .checked_mul(cols)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?
                != relation.scalar_count
            {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            for column in 0..cols {
                let first_vector = *block_vectors
                    .get(relation.blocks[0].0)
                    .and_then(|columns| columns.get(column))
                    .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                let first_bucket = bva.layout.vector_to_bucket()[first_vector];
                let row_challenges = (0..bva.layout.bucket_sizes()[first_bucket].ilog2())
                    .map(|_| next_challenge())
                    .collect::<Vec<_>>();
                let batch_weight = next_challenge();
                for (block, coefficient) in
                    relation
                        .blocks
                        .iter()
                        .zip([F::from(1_u64), F::from(1_u64), -F::from(1_u64)])
                {
                    let vector = *block_vectors
                        .get(block.0)
                        .and_then(|columns| columns.get(column))
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                    let bucket = bva.layout.vector_to_bucket()[vector];
                    if bva.layout.bucket_sizes()[bucket] != bva.layout.bucket_sizes()[first_bucket]
                    {
                        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                    }
                    terms[bucket].push((batch_weight * coefficient, row_challenges.clone()));
                }
            }
        }
        if relation_count == 0 {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        let buckets = terms.into_iter().map(TbvBucket::from_terms).collect();
        Ok((
            TensorBasisVector::new(&bva.layout, buckets)?,
            F::from(0_u64),
        ))
    }

    fn linear_relations_tbv<F: PrimeField>(
        &self,
        metadata: &Gpt2Q12Metadata,
        base: &Gpt2GlobalLayout,
        bva: &Gpt2PrimitiveBvaLayout,
        token_ids: &[usize],
        mut next_challenge: impl FnMut() -> F,
    ) -> Result<(TensorBasisVector<F>, F, usize), Gpt2CheckpointError> {
        if token_ids.len() != metadata.token_count
            || token_ids
                .iter()
                .any(|token| *token >= metadata.model.vocabulary)
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let block_vectors = self.bva_block_vectors(bva)?;
        let mut terms = vec![Vec::<(F, Vec<(F, F)>)>::new(); bva.layout.bucket_sizes().len()];
        let mut public_claim = F::from(0_u64);
        let mut covered = 0_usize;
        let scale = signed_i128_scalar::<F>(i128::from(metadata.scale()));
        let attention_segments = causal_row_segments(metadata)?;

        for relation in &self.relations {
            if relation.family != RelationFamily::Linear
                || relation.kind == LinearRelationKind::RangeRecomposition as u8
            {
                continue;
            }
            covered += 1;
            let kind = decode_linear_relation_kind(relation.kind)?;
            let relation_weight = next_challenge();

            let elementwise_coefficients = match kind {
                LinearRelationKind::EuclideanDivision => Some(vec![
                    F::from(1_u64),
                    -signed_i128_scalar::<F>(i128::from(relation.immediate)),
                    -F::from(1_u64),
                ]),
                LinearRelationKind::ResidualAdd | LinearRelationKind::AddResidualAdjustment => {
                    Some(vec![F::from(1_u64), F::from(1_u64), -F::from(1_u64)])
                }
                LinearRelationKind::StrictGap => {
                    Some(vec![F::from(1_u64), -F::from(1_u64), -F::from(1_u64)])
                }
                LinearRelationKind::ScaleAndDivide => Some(vec![
                    signed_i128_scalar::<F>(i128::from(relation.immediate)),
                    -scale,
                    -F::from(1_u64),
                ]),
                LinearRelationKind::ClampSelectorSum => {
                    Some(vec![F::from(1_u64), F::from(1_u64), F::from(1_u64)])
                }
                LinearRelationKind::AddConstant => Some(vec![F::from(1_u64), -F::from(1_u64)]),
                _ => None,
            };
            if let Some(coefficients) = elementwise_coefficients {
                let (rows, cols) = common_block_shape(self, base, &relation.blocks)?;
                if coefficients.len() != relation.blocks.len()
                    || rows
                        .checked_mul(cols)
                        .ok_or(Gpt2CheckpointError::OffsetOverflow)?
                        != relation.scalar_count
                {
                    return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                }
                for column in 0..cols {
                    let first = bva_vector(&block_vectors, relation.blocks[0], column)?;
                    let challenges = random_bucket_challenges(bva, first, &mut next_challenge);
                    let column_weight = relation_weight * next_challenge();
                    for (block, coefficient) in relation.blocks.iter().zip(&coefficients) {
                        let vector = bva_vector(&block_vectors, *block, column)?;
                        push_tensor_term(
                            &mut terms,
                            bva,
                            vector,
                            column_weight * coefficient,
                            &challenges,
                        )?;
                    }
                    let coefficient_sum = tensor_prefix_sum(rows, &challenges)?;
                    match kind {
                        LinearRelationKind::ClampSelectorSum => {
                            public_claim += column_weight * coefficient_sum;
                        }
                        LinearRelationKind::AddConstant => {
                            public_claim -= column_weight
                                * signed_i128_scalar::<F>(i128::from(relation.immediate))
                                * coefficient_sum;
                        }
                        _ => {}
                    }
                }
                continue;
            }

            match kind {
                LinearRelationKind::EmbeddingGatherAdd => {
                    if relation.blocks.len() != 4 {
                        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                    }
                    let token_embedding_shape = self.block_shape(base, relation.blocks[1])?;
                    let position_embedding_shape = self.block_shape(base, relation.blocks[2])?;
                    let (hidden, output_cols) = self.block_shape(base, relation.blocks[3])?;
                    if output_cols != token_ids.len()
                        || token_embedding_shape != (metadata.model.vocabulary, hidden)
                        || position_embedding_shape != (metadata.model.maximum_positions, hidden)
                    {
                        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                    }
                    for (position, token) in token_ids.iter().copied().enumerate() {
                        let output = bva_vector(&block_vectors, relation.blocks[3], position)?;
                        let challenges = random_bucket_challenges(bva, output, &mut next_challenge);
                        let weight = relation_weight * next_challenge();
                        let hidden_coefficients =
                            tensor_physical_coefficients(hidden, &challenges)?;
                        for (feature, coefficient) in hidden_coefficients.iter().enumerate() {
                            let token_column =
                                bva_vector(&block_vectors, relation.blocks[1], feature)?;
                            let position_column =
                                bva_vector(&block_vectors, relation.blocks[2], feature)?;
                            push_lagrange_point_term(
                                &mut terms,
                                bva,
                                token_column,
                                token,
                                weight * coefficient,
                            )?;
                            push_lagrange_point_term(
                                &mut terms,
                                bva,
                                position_column,
                                position,
                                weight * coefficient,
                            )?;
                        }
                        push_tensor_term(&mut terms, bva, output, -weight, &challenges)?;
                    }
                }
                LinearRelationKind::ReduceSum => {
                    if relation.blocks.len() != 2 {
                        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                    }
                    let input_shape = self.block_shape(base, relation.blocks[0])?;
                    let output_shape = self.block_shape(base, relation.blocks[1])?;
                    if output_shape.0 == 1
                        && input_shape.1 == output_shape.1
                        && relation.scalar_count == input_shape.1
                    {
                        for column in 0..input_shape.1 {
                            let weight = relation_weight * next_challenge();
                            let input = bva_vector(&block_vectors, relation.blocks[0], column)?;
                            let output = bva_vector(&block_vectors, relation.blocks[1], column)?;
                            let input_challenges = vec![F::from(1_u64); bva_bucket_log(bva, input)];
                            push_tensor_term(&mut terms, bva, input, weight, &input_challenges)?;
                            push_tensor_term(&mut terms, bva, output, -weight, &[])?;
                        }
                    } else {
                        require_segmented_shapes(
                            input_shape,
                            output_shape,
                            relation.scalar_count,
                            &attention_segments,
                        )?;
                        let input = bva_vector(&block_vectors, relation.blocks[0], 0)?;
                        let output = bva_vector(&block_vectors, relation.blocks[1], 0)?;
                        let row_challenges =
                            random_bucket_challenges(bva, output, &mut next_challenge);
                        let row_coefficients =
                            tensor_physical_coefficients(output_shape.0, &row_challenges)?;
                        let input_coefficients =
                            expand_segment_coefficients(&row_coefficients, &attention_segments)?;
                        push_dense_terms(
                            &mut terms,
                            bva,
                            input,
                            &scale_coefficients(&input_coefficients, relation_weight),
                        )?;
                        push_tensor_term(
                            &mut terms,
                            bva,
                            output,
                            -relation_weight,
                            &row_challenges,
                        )?;
                    }
                }
                LinearRelationKind::BroadcastSubtract => {
                    if relation.blocks.len() != 3 {
                        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                    }
                    let (rows, cols) = self.block_shape(base, relation.blocks[0])?;
                    if self.block_shape(base, relation.blocks[1])? != (1, cols)
                        || self.block_shape(base, relation.blocks[2])? != (rows, cols)
                    {
                        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                    }
                    for column in 0..cols {
                        let input = bva_vector(&block_vectors, relation.blocks[0], column)?;
                        let mean = bva_vector(&block_vectors, relation.blocks[1], column)?;
                        let centered = bva_vector(&block_vectors, relation.blocks[2], column)?;
                        let challenges = random_bucket_challenges(bva, input, &mut next_challenge);
                        let weight = relation_weight * next_challenge();
                        push_tensor_term(&mut terms, bva, input, weight, &challenges)?;
                        push_tensor_term(&mut terms, bva, centered, -weight, &challenges)?;
                        push_tensor_term(
                            &mut terms,
                            bva,
                            mean,
                            -weight * tensor_prefix_sum(rows, &challenges)?,
                            &[],
                        )?;
                    }
                }
                LinearRelationKind::BiasScaleAdd | LinearRelationKind::AffineAdd => {
                    if relation.blocks.len() != 3 {
                        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                    }
                    let (rows, cols) = self.block_shape(base, relation.blocks[0])?;
                    if self.block_shape(base, relation.blocks[1])? != (rows, 1)
                        || self.block_shape(base, relation.blocks[2])? != (rows, cols)
                    {
                        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                    }
                    let broadcast_coefficient = if kind == LinearRelationKind::BiasScaleAdd {
                        signed_i128_scalar::<F>(i128::from(relation.immediate))
                    } else {
                        F::from(1_u64)
                    };
                    for column in 0..cols {
                        let left = bva_vector(&block_vectors, relation.blocks[0], column)?;
                        let broadcast = bva_vector(&block_vectors, relation.blocks[1], 0)?;
                        let output = bva_vector(&block_vectors, relation.blocks[2], column)?;
                        let challenges = random_bucket_challenges(bva, left, &mut next_challenge);
                        let weight = relation_weight * next_challenge();
                        push_tensor_term(&mut terms, bva, left, weight, &challenges)?;
                        push_tensor_term(
                            &mut terms,
                            bva,
                            broadcast,
                            weight * broadcast_coefficient,
                            &challenges,
                        )?;
                        push_tensor_term(&mut terms, bva, output, -weight, &challenges)?;
                    }
                }
                LinearRelationKind::MaximumDifference => {
                    if relation.blocks.len() != 3 {
                        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                    }
                    let packed = bva_vector(&block_vectors, relation.blocks[0], 0)?;
                    let maximum = bva_vector(&block_vectors, relation.blocks[1], 0)?;
                    let gap = bva_vector(&block_vectors, relation.blocks[2], 0)?;
                    let packed_challenges =
                        random_bucket_challenges(bva, packed, &mut next_challenge);
                    let packed_coefficients = tensor_physical_coefficients(
                        attention_segments.iter().sum(),
                        &packed_challenges,
                    )?;
                    let row_coefficients =
                        sum_segment_coefficients(&packed_coefficients, &attention_segments)?;
                    push_tensor_term(
                        &mut terms,
                        bva,
                        packed,
                        -relation_weight,
                        &packed_challenges,
                    )?;
                    push_tensor_term(&mut terms, bva, gap, -relation_weight, &packed_challenges)?;
                    push_dense_terms(
                        &mut terms,
                        bva,
                        maximum,
                        &scale_coefficients(&row_coefficients, relation_weight),
                    )?;
                }
                LinearRelationKind::OneHotSum => {
                    let selector = bva_vector(&block_vectors, relation.blocks[0], 0)?;
                    let row_challenges =
                        random_log_challenges(relation.scalar_count, &mut next_challenge)?;
                    let row_coefficients =
                        tensor_physical_coefficients(relation.scalar_count, &row_challenges)?;
                    let selector_coefficients =
                        expand_segment_coefficients(&row_coefficients, &attention_segments)?;
                    push_dense_terms(
                        &mut terms,
                        bva,
                        selector,
                        &scale_coefficients(&selector_coefficients, relation_weight),
                    )?;
                    public_claim += relation_weight
                        * tensor_prefix_sum(relation.scalar_count, &row_challenges)?;
                }
                LinearRelationKind::PrefixSelector => {
                    let selector = bva_vector(&block_vectors, relation.blocks[0], 0)?;
                    let prefix = bva_vector(&block_vectors, relation.blocks[1], 0)?;
                    let challenges = random_bucket_challenges(bva, prefix, &mut next_challenge);
                    let equation_coefficients =
                        tensor_physical_coefficients(relation.scalar_count, &challenges)?;
                    let selector_coefficients =
                        suffix_segment_coefficients(&equation_coefficients, &attention_segments)?;
                    push_tensor_term(&mut terms, bva, prefix, relation_weight, &challenges)?;
                    push_dense_terms(
                        &mut terms,
                        bva,
                        selector,
                        &scale_coefficients(&selector_coefficients, relation_weight),
                    )?;
                    public_claim +=
                        relation_weight * tensor_prefix_sum(relation.scalar_count, &challenges)?;
                }
                LinearRelationKind::ResidualMass | LinearRelationKind::ExactRowMass => {
                    let row_count = relation.scalar_count;
                    if row_count != attention_segments.len() {
                        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                    }
                    let row_challenges = random_log_challenges(row_count, &mut next_challenge)?;
                    let row_coefficients =
                        tensor_physical_coefficients(row_count, &row_challenges)?;
                    let packed_coefficients =
                        expand_segment_coefficients(&row_coefficients, &attention_segments)?;
                    let packed = bva_vector(&block_vectors, relation.blocks[0], 0)?;
                    push_dense_terms(
                        &mut terms,
                        bva,
                        packed,
                        &scale_coefficients(&packed_coefficients, relation_weight),
                    )?;
                    if kind == LinearRelationKind::ResidualMass {
                        let residual = bva_vector(&block_vectors, relation.blocks[1], 0)?;
                        push_tensor_term(
                            &mut terms,
                            bva,
                            residual,
                            relation_weight,
                            &row_challenges,
                        )?;
                    }
                    public_claim += relation_weight
                        * signed_i128_scalar::<F>(i128::from(relation.immediate))
                        * tensor_prefix_sum(row_count, &row_challenges)?;
                }
                LinearRelationKind::RangeRecomposition => unreachable!(),
                LinearRelationKind::AddConstant
                | LinearRelationKind::EuclideanDivision
                | LinearRelationKind::ResidualAdd
                | LinearRelationKind::StrictGap
                | LinearRelationKind::ScaleAndDivide
                | LinearRelationKind::ClampSelectorSum
                | LinearRelationKind::AddResidualAdjustment => unreachable!(),
            }
        }
        if covered == 0 {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        let buckets = terms
            .into_iter()
            .map(TbvBucket::from_factor_terms)
            .collect();
        Ok((
            TensorBasisVector::new(&bva.layout, buckets)?,
            public_claim,
            covered,
        ))
    }

    fn public_io_tbv<F: PrimeField>(
        &self,
        base: &Gpt2GlobalLayout,
        bva: &Gpt2PrimitiveBvaLayout,
        token_ids: &[usize],
        public_logits: &[i64],
        mut next_challenge: impl FnMut() -> F,
    ) -> Result<(TensorBasisVector<F>, F), Gpt2CheckpointError> {
        let input_block = base.public_input_block();
        let output_block = base.public_output_block();
        if self.block_shape(base, input_block)? != (token_ids.len(), 1)
            || self.block_shape(base, output_block)? != (public_logits.len(), 1)
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let block_vectors = self.bva_block_vectors(bva)?;
        let input_vector = bva_vector(&block_vectors, input_block, 0)?;
        let output_vector = bva_vector(&block_vectors, output_block, 0)?;
        let input_challenges = random_bucket_challenges(bva, input_vector, &mut next_challenge);
        let output_challenges = random_bucket_challenges(bva, output_vector, &mut next_challenge);
        let input_weight = next_challenge();
        let output_weight = next_challenge();
        let input_coefficients = tensor_physical_coefficients(token_ids.len(), &input_challenges)?;
        let output_coefficients =
            tensor_physical_coefficients(public_logits.len(), &output_challenges)?;
        let mut public_claim = F::from(0_u64);
        for (token, coefficient) in token_ids.iter().zip(&input_coefficients) {
            public_claim += input_weight * F::from(*token as u64) * coefficient;
        }
        for (logit, coefficient) in public_logits.iter().zip(&output_coefficients) {
            public_claim +=
                output_weight * signed_i128_scalar::<F>(i128::from(*logit)) * coefficient;
        }
        let mut terms = vec![Vec::<(F, Vec<(F, F)>)>::new(); bva.layout.bucket_sizes().len()];
        push_tensor_term(
            &mut terms,
            bva,
            input_vector,
            input_weight,
            &input_challenges,
        )?;
        push_tensor_term(
            &mut terms,
            bva,
            output_vector,
            output_weight,
            &output_challenges,
        )?;
        let buckets = terms
            .into_iter()
            .map(TbvBucket::from_factor_terms)
            .collect();
        Ok((TensorBasisVector::new(&bva.layout, buckets)?, public_claim))
    }

    /// Authenticates every lookup-query key/value leaf against the original
    /// global witness commitment.  The public tag is returned separately so a
    /// Lasso factor source can reconstruct
    /// `tag + row_challenge * input + row_challenge^2 * output` without
    /// materializing a concatenated query vector or any BVA padding.
    fn lookup_relations_tbv<F: PrimeField>(
        &self,
        base: &Gpt2GlobalLayout,
        bva: &Gpt2PrimitiveBvaLayout,
        query_layout: Option<&BvaLayout>,
        row_challenge: F,
        mut next_challenge: impl FnMut() -> F,
    ) -> Result<(TensorBasisVector<F>, Option<TensorBasisVector<F>>, F, usize), Gpt2CheckpointError>
    {
        if row_challenge == F::from(0_u64) {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let block_vectors = self.bva_block_vectors(bva)?;
        let mut terms = vec![Vec::<(F, Vec<(F, F)>)>::new(); bva.layout.bucket_sizes().len()];
        let mut query_terms = query_layout
            .map(|layout| vec![Vec::<(F, Vec<(F, F)>)>::new(); layout.bucket_sizes().len()]);
        let mut query_relation_terms =
            query_layout.map(|_| vec![None::<Vec<(F, Vec<F>)>>; self.relations.len()]);
        let mut public_offset = F::from(0_u64);
        let mut covered = 0_usize;
        let output_coefficient = row_challenge * row_challenge;

        for (relation_index, relation) in self
            .relations
            .iter()
            .enumerate()
            .filter(|(_, relation)| relation.family == RelationFamily::TaggedLookup)
        {
            let tag = decode_lookup_relation_kind(relation.kind)?;
            let expected_blocks = if tag <= TaggedLookupRelationKind::InverseSqrt {
                2
            } else {
                1
            };
            if relation.blocks.len() != expected_blocks {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            let (rows, cols) = common_block_shape(self, base, &relation.blocks)?;
            if rows
                .checked_mul(cols)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?
                != relation.scalar_count
            {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            covered = covered
                .checked_add(1)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
            let relation_weight = next_challenge();
            let mut relation_query_terms = Vec::with_capacity(cols);
            for column in 0..cols {
                let input = bva_vector(&block_vectors, relation.blocks[0], column)?;
                let row_point = random_bucket_challenges(bva, input, &mut next_challenge);
                let weight = relation_weight * next_challenge();
                relation_query_terms.push((weight, row_point.clone()));
                push_tensor_term(&mut terms, bva, input, weight * row_challenge, &row_point)?;
                if let Some(output_block) = relation.blocks.get(1) {
                    let output = bva_vector(&block_vectors, *output_block, column)?;
                    if bva_bucket_log(bva, output) != bva_bucket_log(bva, input) {
                        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                    }
                    push_tensor_term(
                        &mut terms,
                        bva,
                        output,
                        weight * output_coefficient,
                        &row_point,
                    )?;
                }
                public_offset +=
                    weight * F::from(tag as u64) * tensor_prefix_sum(rows, &row_point)?;
            }
            if let Some(all_query_terms) = query_relation_terms.as_mut() {
                all_query_terms[relation_index] = Some(relation_query_terms);
            }
        }
        if covered == 0 {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        let buckets = terms
            .into_iter()
            .map(TbvBucket::from_factor_terms)
            .collect();
        let query_tbv = match (query_layout, query_terms.as_mut(), query_relation_terms) {
            (Some(layout), Some(terms), Some(relation_terms)) => {
                let mut query_cursor = (0_usize, 0_usize);
                for (relation_index, relation) in
                    self.relations.iter().enumerate().filter(|(_, relation)| {
                        relation.family == RelationFamily::TaggedLookup
                            && relation.kind <= TaggedLookupRelationKind::InverseSqrt as u8
                    })
                {
                    let rows = self.block_shape(base, relation.blocks[0])?.0;
                    let column_terms = relation_terms[relation_index]
                        .as_ref()
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                    for (weight, row_point) in column_terms {
                        push_lookup_query_tensor_slice(
                            terms,
                            layout,
                            &mut query_cursor,
                            rows,
                            *weight,
                            row_point,
                        )?;
                    }
                }
                let mut group_start = 0_usize;
                while group_start < self.range_limb_blocks.len() {
                    let source_kind = self.range_limb_blocks[group_start].source_kind;
                    let mut group_end = group_start + 1;
                    while group_end < self.range_limb_blocks.len()
                        && self.range_limb_blocks[group_end].source_kind == source_kind
                    {
                        group_end += 1;
                    }
                    let limb_blocks = &self.range_limb_blocks[group_start..group_end];
                    let first = limb_blocks
                        .first()
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                    for column in 0..first.cols {
                        for limb in limb_blocks {
                            let relation_index = self
                                .relations
                                .iter()
                                .position(|relation| {
                                    relation.family == RelationFamily::TaggedLookup
                                        && relation.blocks == [limb.id]
                                })
                                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                            let (weight, row_point) = relation_terms[relation_index]
                                .as_ref()
                                .and_then(|terms| terms.get(column))
                                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                            push_lookup_query_tensor_slice(
                                terms,
                                layout,
                                &mut query_cursor,
                                limb.rows,
                                *weight,
                                row_point,
                            )?;
                        }
                    }
                    group_start = group_end;
                }
                if query_cursor.1 != 0 {
                    return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                }
                let buckets = std::mem::take(terms)
                    .into_iter()
                    .map(TbvBucket::from_factor_terms)
                    .collect();
                Some(TensorBasisVector::new(layout, buckets)?)
            }
            (None, None, None) => None,
            _ => return Err(Gpt2CheckpointError::ProtocolShapeMismatch),
        };
        Ok((
            TensorBasisVector::new(&bva.layout, buckets)?,
            query_tbv,
            public_offset,
            covered,
        ))
    }

    /// Compiles the elementwise portion of the single global bilinear family
    /// into matching left/right BVA views. Matrix products and broadcasted
    /// products remain in their dedicated structured reductions.
    fn elementwise_bilinear_routing(
        &self,
        metadata: &Gpt2Q12Metadata,
        base: &Gpt2GlobalLayout,
        bva: &Gpt2PrimitiveBvaLayout,
    ) -> Result<ElementwiseBilinearRouting, Gpt2CheckpointError> {
        let block_vectors = self.bva_block_vectors(bva)?;
        let attention_segments = causal_row_segments(metadata)?;
        let mut vectors = Vec::new();
        let mut lengths = Vec::new();
        let mut covered_relations = 0_usize;
        for relation in &self.relations {
            if relation.family != RelationFamily::Bilinear {
                continue;
            }
            let kind = decode_bilinear_relation_kind(relation.kind)?;
            if matches!(
                kind,
                BilinearRelationKind::ClampLower | BilinearRelationKind::ClampUpper
            ) {
                if relation.blocks.len() != 6 {
                    return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                }
                let (rows, cols) = common_block_shape(self, base, &relation.blocks)?;
                if rows
                    .checked_mul(cols)
                    .ok_or(Gpt2CheckpointError::OffsetOverflow)?
                    != relation.scalar_count
                {
                    return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                }
                covered_relations += 1;
                for column in 0..cols {
                    let mut push_affine_product =
                        |left_block: GlobalBlockId,
                         right_block: GlobalBlockId,
                         right_scale: i128,
                         subtract_blocks: [Option<GlobalBlockId>; 2],
                         right_constant: i128|
                         -> Result<(), Gpt2CheckpointError> {
                            let mut right_subtract = [None, None];
                            for (target, block) in right_subtract.iter_mut().zip(subtract_blocks) {
                                if let Some(block) = block {
                                    *target = Some((
                                        block,
                                        column,
                                        bva_vector(&block_vectors, block, column)?,
                                    ));
                                }
                            }
                            vectors.push(ElementwiseBilinearVector {
                                left_block,
                                left_column: column,
                                left_vector: bva_vector(&block_vectors, left_block, column)?,
                                right_block,
                                right_column: column,
                                right_vector: bva_vector(&block_vectors, right_block, column)?,
                                right_scalar_index: None,
                                right_segmented: false,
                                right_scale,
                                right_subtract,
                                right_constant,
                                output_vector: None,
                            });
                            lengths.push(rows);
                            Ok(())
                        };
                    let input = relation.blocks[0];
                    let clamped = relation.blocks[1];
                    let first_selector = relation.blocks[2];
                    let second_selector = relation.blocks[3];
                    let first_slack = relation.blocks[4];
                    let second_slack = relation.blocks[5];
                    match kind {
                        BilinearRelationKind::ClampLower => {
                            push_affine_product(
                                first_selector,
                                first_selector,
                                1,
                                [None, None],
                                -1,
                            )?;
                            push_affine_product(
                                second_selector,
                                second_selector,
                                1,
                                [None, None],
                                -1,
                            )?;
                            push_affine_product(
                                first_selector,
                                clamped,
                                1,
                                [None, None],
                                -i128::from(relation.immediate),
                            )?;
                            push_affine_product(
                                second_selector,
                                clamped,
                                1,
                                [Some(input), None],
                                0,
                            )?;
                            push_affine_product(
                                first_selector,
                                input,
                                -1,
                                [Some(first_slack), None],
                                i128::from(relation.immediate),
                            )?;
                            push_affine_product(
                                second_selector,
                                input,
                                1,
                                [Some(second_slack), None],
                                -i128::from(relation.immediate),
                            )?;
                        }
                        BilinearRelationKind::ClampUpper => {
                            push_affine_product(
                                second_selector,
                                second_selector,
                                1,
                                [None, None],
                                -1,
                            )?;
                            push_affine_product(
                                second_selector,
                                clamped,
                                1,
                                [None, None],
                                -i128::from(relation.immediate),
                            )?;
                            push_affine_product(
                                first_selector,
                                input,
                                -1,
                                [Some(first_slack), None],
                                i128::from(relation.immediate),
                            )?;
                            push_affine_product(
                                second_selector,
                                input,
                                1,
                                [Some(second_slack), None],
                                -i128::from(relation.immediate),
                            )?;
                        }
                        _ => unreachable!(),
                    }
                }
                continue;
            }
            let (left_block, right_block, right_constant, output_block) = match kind {
                BilinearRelationKind::Square if relation.blocks.len() == 2 => (
                    relation.blocks[0],
                    relation.blocks[0],
                    0_i128,
                    Some(relation.blocks[1]),
                ),
                BilinearRelationKind::Hadamard if relation.blocks.len() == 3 => (
                    relation.blocks[0],
                    relation.blocks[1],
                    0_i128,
                    Some(relation.blocks[2]),
                ),
                BilinearRelationKind::Boolean if relation.blocks.len() == 1 => {
                    (relation.blocks[0], relation.blocks[0], -1_i128, None)
                }
                BilinearRelationKind::SelectorZero if relation.blocks.len() == 2 => {
                    (relation.blocks[0], relation.blocks[1], 0_i128, None)
                }
                BilinearRelationKind::SelectorResidual if relation.blocks.len() == 3 => (
                    relation.blocks[0],
                    relation.blocks[1],
                    0_i128,
                    Some(relation.blocks[2]),
                ),
                _ => continue,
            };
            let (rows, cols) = self.block_shape(base, left_block)?;
            if output_block
                .is_some_and(|output| self.block_shape(base, output).ok() != Some((rows, cols)))
            {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            let right_shape = self.block_shape(base, right_block)?;
            let right_auxiliary_kind = right_block
                .0
                .checked_sub(self.base_block_count)
                .and_then(|index| self.auxiliary_blocks.get(index))
                .map(|descriptor| descriptor.kind);
            let scalar_column_block = right_shape == (1, cols);
            let scalar_vector_block = right_shape == (cols, 1)
                && (rows != cols
                    || matches!(right_auxiliary_kind, Some(AuxiliaryBlockKind::LookupOutput)));
            let scalar_per_column = scalar_column_block || scalar_vector_block;
            let vector_broadcast = right_shape == (rows, 1) && !scalar_per_column;
            let segmented_broadcast = cols == 1
                && right_shape == (attention_segments.len(), 1)
                && rows == attention_segments.iter().sum();
            if right_shape != (rows, cols)
                && !scalar_per_column
                && !vector_broadcast
                && !segmented_broadcast
            {
                // Segmented broadcasts (for example attention row values over
                // a packed triangle) remain in the structured reducer.
                continue;
            }
            if rows
                .checked_mul(cols)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?
                != relation.scalar_count
            {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            covered_relations += 1;
            for column in 0..cols {
                let (right_column, right_scalar_index) = if scalar_column_block {
                    (column, Some(0))
                } else if scalar_vector_block {
                    (0, Some(column))
                } else if vector_broadcast || segmented_broadcast {
                    (0, None)
                } else {
                    (column, None)
                };
                vectors.push(ElementwiseBilinearVector {
                    left_block,
                    left_column: column,
                    left_vector: bva_vector(&block_vectors, left_block, column)?,
                    right_block,
                    right_column,
                    right_vector: bva_vector(&block_vectors, right_block, right_column)?,
                    right_scalar_index,
                    right_segmented: segmented_broadcast,
                    right_scale: 1_i128,
                    right_subtract: [None, None],
                    right_constant,
                    output_vector: output_block
                        .map(|block| bva_vector(&block_vectors, block, column))
                        .transpose()?,
                });
                lengths.push(rows);
            }
        }
        if vectors.is_empty() {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        Ok(ElementwiseBilinearRouting {
            layout: BvaLayout::from_lengths(&lengths)?,
            vectors,
            attention_segments,
            covered_relations,
        })
    }

    fn dense_matmul_routing(
        &self,
        base: &Gpt2GlobalLayout,
        bva: &Gpt2PrimitiveBvaLayout,
    ) -> Result<DenseMatmulRouting, Gpt2CheckpointError> {
        let block_vectors = self.bva_block_vectors(bva)?;
        let mut vectors = Vec::new();
        let mut lengths = Vec::new();
        let mut covered_relations = 0_usize;
        for relation in &self.relations {
            if relation.family != RelationFamily::Bilinear {
                continue;
            }
            let kind = decode_bilinear_relation_kind(relation.kind)?;
            if !matches!(
                kind,
                BilinearRelationKind::MatrixMultiply
                    | BilinearRelationKind::LastTokenMatrixMultiply
            ) {
                continue;
            }
            if relation.blocks.len() != 3 {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            let input_block = relation.blocks[0];
            let weight_block = relation.blocks[1];
            let output_block = relation.blocks[2];
            let (inner, input_column_count) = self.block_shape(base, input_block)?;
            let (output_rows, weight_columns) = self.block_shape(base, weight_block)?;
            let (actual_output_rows, output_columns) = self.block_shape(base, output_block)?;
            if inner != weight_columns
                || output_rows != actual_output_rows
                || output_rows
                    .checked_mul(output_columns)
                    .ok_or(Gpt2CheckpointError::OffsetOverflow)?
                    != relation.scalar_count
            {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            let input_columns = if kind == BilinearRelationKind::LastTokenMatrixMultiply {
                if output_columns != 1 || input_column_count == 0 {
                    return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                }
                vec![input_column_count - 1]
            } else {
                if input_column_count != output_columns {
                    return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                }
                (0..input_column_count).collect()
            };
            let input_vectors = input_columns
                .iter()
                .map(|column| bva_vector(&block_vectors, input_block, *column))
                .collect::<Result<Vec<_>, _>>()?;
            let weight_vectors = (0..inner)
                .map(|column| bva_vector(&block_vectors, weight_block, column))
                .collect::<Result<Vec<_>, _>>()?;
            let output_vectors = (0..output_columns)
                .map(|column| bva_vector(&block_vectors, output_block, column))
                .collect::<Result<Vec<_>, _>>()?;
            vectors.push(DenseMatmulVector {
                input_block,
                input_columns,
                input_vectors,
                weight_block,
                weight_vectors,
                output_vectors,
                inner,
                output_rows,
            });
            lengths.push(inner);
            covered_relations += 1;
        }
        if vectors.is_empty() {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        Ok(DenseMatmulRouting {
            layout: BvaLayout::from_lengths(&lengths)?,
            vectors,
            covered_relations,
        })
    }

    fn causal_qk_routing(
        &self,
        metadata: &Gpt2Q12Metadata,
        base: &Gpt2GlobalLayout,
        bva: &Gpt2PrimitiveBvaLayout,
    ) -> Result<CausalQkRouting, Gpt2CheckpointError> {
        let block_vectors = self.bva_block_vectors(bva)?;
        let model = metadata.model;
        let tokens = metadata.token_count;
        let head_width = model.head_width();
        let expected_packed = model
            .attention_heads
            .checked_mul(
                tokens
                    .checked_mul(
                        tokens
                            .checked_add(1)
                            .ok_or(Gpt2CheckpointError::OffsetOverflow)?,
                    )
                    .ok_or(Gpt2CheckpointError::OffsetOverflow)?
                    / 2,
            )
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        let mut terms = Vec::new();
        let mut relations = Vec::new();
        for relation in &self.relations {
            if relation.family != RelationFamily::Bilinear
                || decode_bilinear_relation_kind(relation.kind)? != BilinearRelationKind::CausalQk
            {
                continue;
            }
            if relation.blocks.len() != 2
                || self.block_shape(base, relation.blocks[0])? != (3 * model.hidden_width, tokens)
                || self.block_shape(base, relation.blocks[1])? != (expected_packed, 1)
                || relation.scalar_count != expected_packed
            {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            let relation_index = relations.len();
            relations.push(CausalQkRelation {
                output_vector: bva_vector(&block_vectors, relation.blocks[1], 0)?,
                output_len: expected_packed,
            });
            let mut output_index = 0_usize;
            for head in 0..model.attention_heads {
                let query_start = head * head_width;
                let key_start = model.hidden_width + query_start;
                for query_column in 0..tokens {
                    for key_column in 0..=query_column {
                        terms.push(CausalQkTerm {
                            qkv_block: relation.blocks[0],
                            query_column,
                            query_vector: bva_vector(
                                &block_vectors,
                                relation.blocks[0],
                                query_column,
                            )?,
                            query_start,
                            key_column,
                            key_vector: bva_vector(&block_vectors, relation.blocks[0], key_column)?,
                            key_start,
                            relation_index,
                            output_index,
                            head_width,
                        });
                        output_index += 1;
                    }
                }
            }
            if output_index != expected_packed {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
        }
        if terms.is_empty() {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        Ok(CausalQkRouting {
            layout: BvaLayout::from_lengths(
                &terms.iter().map(|term| term.head_width).collect::<Vec<_>>(),
            )?,
            terms,
            relations,
        })
    }

    fn causal_pv_routing(
        &self,
        metadata: &Gpt2Q12Metadata,
        base: &Gpt2GlobalLayout,
        bva: &Gpt2PrimitiveBvaLayout,
    ) -> Result<CausalPvRouting, Gpt2CheckpointError> {
        let block_vectors = self.bva_block_vectors(bva)?;
        let model = metadata.model;
        let tokens = metadata.token_count;
        let head_width = model.head_width();
        let triangular = tokens
            .checked_mul(
                tokens
                    .checked_add(1)
                    .ok_or(Gpt2CheckpointError::OffsetOverflow)?,
            )
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?
            / 2;
        let expected_packed = model
            .attention_heads
            .checked_mul(triangular)
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        let mut terms = Vec::new();
        let mut relations = Vec::new();
        for relation in &self.relations {
            if relation.family != RelationFamily::Bilinear
                || decode_bilinear_relation_kind(relation.kind)? != BilinearRelationKind::CausalPv
            {
                continue;
            }
            if relation.blocks.len() != 3
                || self.block_shape(base, relation.blocks[0])? != (expected_packed, 1)
                || self.block_shape(base, relation.blocks[1])? != (3 * model.hidden_width, tokens)
                || self.block_shape(base, relation.blocks[2])? != (model.hidden_width, tokens)
                || relation.scalar_count
                    != model
                        .hidden_width
                        .checked_mul(tokens)
                        .ok_or(Gpt2CheckpointError::OffsetOverflow)?
            {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            let relation_index = relations.len();
            relations.push(CausalPvRelation {
                output_vectors: (0..tokens)
                    .map(|column| bva_vector(&block_vectors, relation.blocks[2], column))
                    .collect::<Result<Vec<_>, _>>()?,
                qkv_vectors: (0..tokens)
                    .map(|column| bva_vector(&block_vectors, relation.blocks[1], column))
                    .collect::<Result<Vec<_>, _>>()?,
                output_rows: model.hidden_width,
                output_columns: tokens,
            });
            let probability_vector = bva_vector(&block_vectors, relation.blocks[0], 0)?;
            let mut probability_start = 0_usize;
            for head in 0..model.attention_heads {
                let value_start = 2 * model.hidden_width + head * head_width;
                for output_column in 0..tokens {
                    let inner = output_column + 1;
                    for offset in 0..head_width {
                        terms.push(CausalPvTerm {
                            probability_block: relation.blocks[0],
                            probability_vector,
                            probability_start,
                            qkv_block: relation.blocks[1],
                            value_index: value_start + offset,
                            relation_index,
                            output_row: head * head_width + offset,
                            output_column,
                            inner,
                        });
                    }
                    probability_start = probability_start
                        .checked_add(inner)
                        .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
                }
            }
            if probability_start != expected_packed {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
        }
        if terms.is_empty() {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        Ok(CausalPvRouting {
            layout: BvaLayout::from_lengths(
                &terms.iter().map(|term| term.inner).collect::<Vec<_>>(),
            )?,
            terms,
            relations,
        })
    }

    fn bva_block_vectors(
        &self,
        bva: &Gpt2PrimitiveBvaLayout,
    ) -> Result<Vec<Vec<usize>>, Gpt2CheckpointError> {
        if bva.vectors.len() != bva.layout.vector_to_bucket().len() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let total_blocks = self
            .base_block_count
            .checked_add(self.auxiliary_blocks.len())
            .and_then(|count| count.checked_add(self.range_limb_blocks.len()))
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        let mut block_vectors = vec![Vec::new(); total_blocks];
        for (vector_index, vector) in bva.vectors.iter().enumerate() {
            let (block, column) = match *vector {
                PrimitiveBvaVectorSource::Base { block, column } => (block, column),
                PrimitiveBvaVectorSource::Auxiliary {
                    block_index,
                    column,
                } => (
                    self.auxiliary_blocks
                        .get(block_index)
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?
                        .id,
                    column,
                ),
                PrimitiveBvaVectorSource::RangeLimb {
                    block_index,
                    column,
                } => (
                    self.range_limb_blocks
                        .get(block_index)
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?
                        .id,
                    column,
                ),
            };
            let columns = block_vectors
                .get_mut(block.0)
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            if column != columns.len() {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            columns.push(vector_index);
        }
        Ok(block_vectors)
    }

    fn block_shape(
        &self,
        base: &Gpt2GlobalLayout,
        block: GlobalBlockId,
    ) -> Result<(usize, usize), Gpt2CheckpointError> {
        if block.0 < self.base_block_count {
            let descriptor = base
                .blocks()
                .get(block.0)
                .filter(|descriptor| descriptor.id == block)
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            return Ok((descriptor.rows, descriptor.cols));
        }
        let auxiliary_index = block.0 - self.base_block_count;
        if let Some(descriptor) = self.auxiliary_blocks.get(auxiliary_index) {
            if descriptor.id != block {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            return Ok((descriptor.rows, descriptor.cols));
        }
        let range_index = auxiliary_index
            .checked_sub(self.auxiliary_blocks.len())
            .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
        let descriptor = self
            .range_limb_blocks
            .get(range_index)
            .filter(|descriptor| descriptor.id == block)
            .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
        Ok((descriptor.rows, descriptor.cols))
    }

    pub(super) fn bva_digest(
        &self,
        base: &Gpt2GlobalLayout,
    ) -> Result<[u8; 32], Gpt2CheckpointError> {
        let bva = self.compile_bva(base)?;
        let mut hasher = Sha256::new();
        hasher.update(b"zksmart.gpt2.primitive-global-bva-layout.v2");
        hash_usize(&mut hasher, self.base_block_count);
        hash_usize(&mut hasher, self.auxiliary_blocks.len());
        hash_usize(&mut hasher, self.range_limb_blocks.len());
        hash_usize(&mut hasher, bva.physical_entries);
        hash_usize(&mut hasher, bva.layout.logical_len());
        hash_usize(&mut hasher, bva.vectors.len());
        let bucket_offsets = bva.layout.bucket_offsets();
        for (vector_index, source) in bva.vectors.iter().enumerate() {
            hash_usize(&mut hasher, vector_index);
            match source {
                PrimitiveBvaVectorSource::Base { block, column } => {
                    hasher.update([0]);
                    hash_usize(&mut hasher, block.0);
                    hash_usize(&mut hasher, *column);
                }
                PrimitiveBvaVectorSource::Auxiliary {
                    block_index,
                    column,
                } => {
                    hasher.update([1]);
                    hash_usize(&mut hasher, *block_index);
                    hash_usize(&mut hasher, *column);
                }
                PrimitiveBvaVectorSource::RangeLimb {
                    block_index,
                    column,
                } => {
                    hasher.update([2]);
                    hash_usize(&mut hasher, *block_index);
                    hash_usize(&mut hasher, *column);
                }
            }
            let bucket = bva.layout.vector_to_bucket()[vector_index];
            hash_usize(&mut hasher, bva.layout.vector_lengths()[vector_index]);
            hash_usize(&mut hasher, bucket);
            hash_usize(&mut hasher, bucket_offsets[bucket]);
        }
        Ok(hasher.finalize().into())
    }

    pub(super) fn shape(
        &self,
        base: &Gpt2GlobalLayout,
    ) -> Result<Gpt2HdR1csShape, Gpt2CheckpointError> {
        let relation_count = |family| {
            self.relations
                .iter()
                .filter(|relation| relation.family == family)
                .count()
        };
        let bva = self.compile_bva(base)?;
        let equation_schedule = self.equation_schedule(base)?;
        let equation_rows = |family| -> Result<usize, Gpt2CheckpointError> {
            let group = equation_schedule
                .groups
                .iter()
                .find(|group| group.family == family)
                .ok_or(Gpt2CheckpointError::ProtocolIncomplete)?;
            group
                .layout
                .vector_lengths()
                .iter()
                .try_fold(0_usize, |sum, rows| {
                    sum.checked_add(*rows)
                        .ok_or(Gpt2CheckpointError::OffsetOverflow)
                })
        };
        Ok(Gpt2HdR1csShape {
            linear_relations: relation_count(RelationFamily::Linear),
            bilinear_relations: relation_count(RelationFamily::Bilinear),
            tagged_lookup_relations: relation_count(RelationFamily::TaggedLookup),
            io_equation_scalar_rows: equation_rows(TopLevelEquationFamily::Io)?,
            linear_equation_scalar_rows: equation_rows(TopLevelEquationFamily::Linear)?,
            bilinear_equation_scalar_rows: equation_rows(TopLevelEquationFamily::Bilinear)?,
            tagged_lookup_equation_scalar_rows: equation_rows(
                TopLevelEquationFamily::TaggedLookup,
            )?,
            equation_bva_logical_rows: equation_schedule.groups.iter().try_fold(
                0_usize,
                |sum, group| {
                    sum.checked_add(group.layout.logical_len())
                        .ok_or(Gpt2CheckpointError::OffsetOverflow)
                },
            )?,
            maximum_live_equation_row_entries: equation_schedule
                .groups
                .iter()
                .flat_map(|group| group.layout.vector_lengths())
                .copied()
                .max()
                .unwrap_or(0),
            equation_materialized_padding_entries: equation_schedule
                .groups
                .iter()
                .map(|group| group.layout.materialized_padding_entries())
                .sum(),
            auxiliary_blocks: self
                .auxiliary_blocks
                .len()
                .checked_add(self.range_limb_blocks.len())
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?,
            auxiliary_entries: self
                .auxiliary_blocks
                .iter()
                .map(|block| block.rows * block.cols)
                .chain(
                    self.range_limb_blocks
                        .iter()
                        .map(|block| block.rows * block.cols),
                )
                .sum(),
            resident_auxiliary_blocks: self.auxiliary_blocks.len(),
            resident_auxiliary_entries: self
                .auxiliary_blocks
                .iter()
                .map(|block| block.rows * block.cols)
                .sum(),
            streamed_range_limb_blocks: self.range_limb_blocks.len(),
            streamed_range_limb_entries: self
                .range_limb_blocks
                .iter()
                .map(|block| block.rows * block.cols)
                .sum(),
            maximum_streamed_range_limb_column_entries: self
                .range_limb_blocks
                .iter()
                .map(|block| block.rows)
                .max()
                .unwrap_or(0),
            global_witness_blocks: self
                .base_block_count
                .checked_add(self.auxiliary_blocks.len())
                .and_then(|count| count.checked_add(self.range_limb_blocks.len()))
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?,
            global_physical_entries: bva.physical_entries(),
            global_bva_logical_entries: bva.logical_entries(),
            materialized_padding_entries: bva.materialized_padding_entries(),
        })
    }

    fn lookup_scalar_count(&self) -> Result<usize, Gpt2CheckpointError> {
        self.relations
            .iter()
            .filter(|relation| {
                relation.family == RelationFamily::TaggedLookup
                    && relation.kind <= TaggedLookupRelationKind::InverseSqrt as u8
            })
            .try_fold(0_usize, |count, relation| {
                count
                    .checked_add(relation.scalar_count)
                    .ok_or(Gpt2CheckpointError::OffsetOverflow)
            })
    }

    pub(super) fn digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"zksmart.gpt2.primitive-relation-manifest.v2");
        hash_usize(&mut hasher, self.base_block_count);
        hash_usize(&mut hasher, self.auxiliary_blocks.len());
        for block in &self.auxiliary_blocks {
            hash_usize(&mut hasher, block.id.0);
            hash_usize(&mut hasher, block.operation_index);
            hasher.update([block.kind as u8]);
            hash_usize(&mut hasher, block.rows);
            hash_usize(&mut hasher, block.cols);
            hasher.update(block.value_bits.to_le_bytes());
        }
        hash_usize(&mut hasher, self.range_limb_blocks.len());
        for block in &self.range_limb_blocks {
            hash_usize(&mut hasher, block.id.0);
            hash_usize(&mut hasher, block.operation_index);
            match block.source_kind {
                RangeLimbSource::Base { block_index } => {
                    hasher.update([0]);
                    hash_usize(&mut hasher, block_index);
                }
                RangeLimbSource::Auxiliary { block_index } => {
                    hasher.update([1]);
                    hash_usize(&mut hasher, block_index);
                }
            }
            hash_usize(&mut hasher, block.source.0);
            hash_usize(&mut hasher, block.rows);
            hash_usize(&mut hasher, block.cols);
            hash_usize(&mut hasher, block.limb);
            hasher.update([block.limb_bits]);
        }
        hash_usize(&mut hasher, self.relations.len());
        for relation in &self.relations {
            hasher.update([family_tag(relation.family), relation.kind]);
            hash_usize(&mut hasher, relation.operation_index);
            hash_usize(&mut hasher, relation.blocks.len());
            for block in &relation.blocks {
                hash_usize(&mut hasher, block.0);
            }
            hash_usize(&mut hasher, relation.scalar_count);
            hasher.update(relation.immediate.to_le_bytes());
        }
        hasher.finalize().into()
    }
}

impl Gpt2GlobalEquationSchedule {
    fn compile(
        base: &Gpt2GlobalLayout,
        manifest: &Gpt2PrimitiveRelationManifest,
    ) -> Result<Self, Gpt2CheckpointError> {
        let maximum_witness_column = base
            .blocks()
            .iter()
            .map(|block| block.rows)
            .chain(manifest.auxiliary_blocks.iter().map(|block| block.rows))
            .chain(manifest.range_limb_blocks.iter().map(|block| block.rows))
            .max()
            .ok_or(Gpt2CheckpointError::ProtocolIncomplete)?;
        let mut io_blocks = base.model_blocks().to_vec();
        for block in [base.public_input_block(), base.public_output_block()] {
            if !io_blocks.contains(&block) {
                io_blocks.push(block);
            }
        }
        let mut io_lengths = Vec::new();
        let mut io_sources = Vec::new();
        for block in io_blocks {
            let descriptor = base
                .blocks()
                .get(block.0)
                .filter(|descriptor| descriptor.id == block)
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            for column in 0..descriptor.cols {
                io_lengths.push(descriptor.rows);
                io_sources.push(EquationRowSource::IoBlockColumn { block, column });
            }
        }
        let mut groups = vec![Gpt2EquationGroupLayout {
            family: TopLevelEquationFamily::Io,
            layout: BvaLayout::from_lengths(&io_lengths)?,
            sources: io_sources,
        }];
        for (top_level, relation_family) in [
            (TopLevelEquationFamily::Linear, RelationFamily::Linear),
            (TopLevelEquationFamily::Bilinear, RelationFamily::Bilinear),
            (
                TopLevelEquationFamily::TaggedLookup,
                RelationFamily::TaggedLookup,
            ),
        ] {
            let mut lengths = Vec::new();
            let mut sources = Vec::new();
            for (relation_index, relation) in manifest.relations.iter().enumerate() {
                if relation.family == relation_family {
                    let mut scalar_offset = 0_usize;
                    while scalar_offset < relation.scalar_count {
                        let scalar_len =
                            (relation.scalar_count - scalar_offset).min(maximum_witness_column);
                        lengths.push(scalar_len);
                        sources.push(EquationRowSource::PrimitiveRelationChunk {
                            relation_index,
                            scalar_offset,
                            scalar_len,
                        });
                        scalar_offset = scalar_offset
                            .checked_add(scalar_len)
                            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
                    }
                }
            }
            groups.push(Gpt2EquationGroupLayout {
                family: top_level,
                layout: BvaLayout::from_lengths(&lengths)?,
                sources,
            });
        }
        Ok(Self { groups })
    }

    fn audit(
        &self,
        base: &Gpt2GlobalLayout,
        manifest: &Gpt2PrimitiveRelationManifest,
    ) -> Result<(), Gpt2CheckpointError> {
        let expected = Self::compile(base, manifest)?;
        if *self != expected
            || self.groups.len() != 4
            || self
                .groups
                .iter()
                .any(|group| group.layout.materialized_padding_entries() != 0)
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        Ok(())
    }
}

impl Gpt2PrimitiveBvaLayout {
    fn physical_entries(&self) -> usize {
        self.physical_entries
    }

    fn logical_entries(&self) -> usize {
        self.layout.logical_len()
    }

    fn materialized_padding_entries(&self) -> usize {
        self.layout.materialized_padding_entries()
    }
}

impl ElementwiseBilinearRouting {
    fn route_left<F: PrimeField>(
        &self,
        source: &TensorBasisVector<F>,
        witness_layout: &BvaLayout,
    ) -> Result<TensorBasisVector<F>, Gpt2CheckpointError> {
        let routes = self
            .vectors
            .iter()
            .map(|vector| Some((vector.left_vector, F::from(1_u64))))
            .collect::<Vec<_>>();
        Ok(source.route_vectors(witness_layout, &routes)?)
    }

    fn route_right<F: PrimeField>(
        &self,
        source: &TensorBasisVector<F>,
        witness_bva: &Gpt2PrimitiveBvaLayout,
    ) -> Result<TensorBasisVector<F>, Gpt2CheckpointError> {
        let routes = self
            .vectors
            .iter()
            .map(|vector| {
                let scale = signed_i128_scalar::<F>(vector.right_scale);
                match vector.right_scalar_index {
                    _ if vector.right_segmented => TensorVectorRoute::Omit,
                    Some(target_index) => TensorVectorRoute::Point {
                        target_vector: vector.right_vector,
                        target_index,
                        scale,
                    },
                    None => TensorVectorRoute::Direct {
                        target_vector: vector.right_vector,
                        scale,
                    },
                }
            })
            .collect::<Vec<_>>();
        let direct = source.route_vector_maps(&witness_bva.layout, &routes)?;
        let mut dense_terms =
            vec![Vec::<(F, Vec<(F, F)>)>::new(); witness_bva.layout.bucket_sizes().len()];
        for (source_vector, vector) in self.vectors.iter().enumerate() {
            if !vector.right_segmented {
                continue;
            }
            let coefficients = source.physical_coefficients(source_vector)?;
            let row_coefficients =
                sum_segment_coefficients(&coefficients, &self.attention_segments)?;
            push_dense_terms(
                &mut dense_terms,
                witness_bva,
                vector.right_vector,
                &scale_coefficients(
                    &row_coefficients,
                    signed_i128_scalar::<F>(vector.right_scale),
                ),
            )?;
        }
        let dense = TensorBasisVector::new(
            &witness_bva.layout,
            dense_terms
                .into_iter()
                .map(TbvBucket::from_factor_terms)
                .collect(),
        )?;
        let mut combined = TensorBasisVector::linear_combination(
            &witness_bva.layout,
            &[(F::from(1_u64), &direct), (F::from(1_u64), &dense)],
        )?;
        for subtract_slot in 0..2 {
            let subtract_routes = self
                .vectors
                .iter()
                .map(|vector| match vector.right_subtract[subtract_slot] {
                    Some((_, _, target_vector)) => TensorVectorRoute::Direct {
                        target_vector,
                        scale: -F::from(1_u64),
                    },
                    None => TensorVectorRoute::Omit,
                })
                .collect::<Vec<_>>();
            let subtract = source.route_vector_maps(&witness_bva.layout, &subtract_routes)?;
            combined = TensorBasisVector::linear_combination(
                &witness_bva.layout,
                &[(F::from(1_u64), &combined), (F::from(1_u64), &subtract)],
            )?;
        }
        Ok(combined)
    }

    fn right_witness_claim<F: PrimeField>(
        &self,
        routed_coefficients: &TensorBasisVector<F>,
        reduced_view_claim: F,
    ) -> Result<F, Gpt2CheckpointError> {
        if routed_coefficients.layout() != &self.layout {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let mut public_offset = F::from(0_u64);
        for (vector_index, vector) in self.vectors.iter().enumerate() {
            public_offset += signed_i128_scalar::<F>(vector.right_constant)
                * routed_coefficients.physical_coefficient_sum(vector_index)?;
        }
        Ok(reduced_view_claim - public_offset)
    }

    fn route_output<F: PrimeField>(
        &self,
        source: &TensorBasisVector<F>,
        witness_layout: &BvaLayout,
    ) -> Result<TensorBasisVector<F>, Gpt2CheckpointError> {
        let routes = self
            .vectors
            .iter()
            .map(|vector| vector.output_vector.map(|output| (output, F::from(1_u64))))
            .collect::<Vec<_>>();
        Ok(source.route_vectors(witness_layout, &routes)?)
    }
}

impl DenseMatmulRouting {
    fn projections<F: PrimeField>(
        &self,
        mut next_challenge: impl FnMut() -> F,
    ) -> Result<Vec<DenseMatmulProjection<F>>, Gpt2CheckpointError> {
        self.vectors
            .iter()
            .map(|vector| {
                let row_challenges =
                    random_log_challenges(vector.output_rows, &mut next_challenge)?;
                let column_challenges =
                    random_log_challenges(vector.output_vectors.len(), &mut next_challenge)?;
                Ok(DenseMatmulProjection {
                    row_coefficients: tensor_physical_coefficients(
                        vector.output_rows,
                        &row_challenges,
                    )?,
                    column_coefficients: tensor_physical_coefficients(
                        vector.output_vectors.len(),
                        &column_challenges,
                    )?,
                    row_challenges,
                    batch_weight: next_challenge(),
                })
            })
            .collect()
    }

    fn route_output<F: PrimeField>(
        &self,
        projections: &[DenseMatmulProjection<F>],
        witness_bva: &Gpt2PrimitiveBvaLayout,
    ) -> Result<TensorBasisVector<F>, Gpt2CheckpointError> {
        if projections.len() != self.vectors.len() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let mut terms =
            vec![Vec::<(F, Vec<(F, F)>)>::new(); witness_bva.layout.bucket_sizes().len()];
        for (vector, projection) in self.vectors.iter().zip(projections) {
            for (column, target_vector) in vector.output_vectors.iter().copied().enumerate() {
                push_tensor_term(
                    &mut terms,
                    witness_bva,
                    target_vector,
                    projection.batch_weight * projection.column_coefficients[column],
                    &projection.row_challenges,
                )?;
            }
        }
        Ok(TensorBasisVector::new(
            &witness_bva.layout,
            terms
                .into_iter()
                .map(TbvBucket::from_factor_terms)
                .collect(),
        )?)
    }

    fn route_left<F: PrimeField>(
        &self,
        equation_coefficients: &TensorBasisVector<F>,
        projections: &[DenseMatmulProjection<F>],
        witness_bva: &Gpt2PrimitiveBvaLayout,
    ) -> Result<TensorBasisVector<F>, Gpt2CheckpointError> {
        if equation_coefficients.layout() != &self.layout || projections.len() != self.vectors.len()
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let omit = vec![TensorVectorRoute::Omit; self.vectors.len()];
        let mut combined = equation_coefficients.route_vector_maps(&witness_bva.layout, &omit)?;
        for (relation_index, (vector, projection)) in
            self.vectors.iter().zip(projections).enumerate()
        {
            for (column, target_vector) in vector.input_vectors.iter().copied().enumerate() {
                let mut routes = omit.clone();
                routes[relation_index] = TensorVectorRoute::Direct {
                    target_vector,
                    scale: projection.column_coefficients[column],
                };
                let routed =
                    equation_coefficients.route_vector_maps(&witness_bva.layout, &routes)?;
                combined = TensorBasisVector::linear_combination(
                    &witness_bva.layout,
                    &[(F::from(1_u64), &combined), (F::from(1_u64), &routed)],
                )?;
            }
        }
        Ok(combined)
    }

    fn route_right<F: PrimeField>(
        &self,
        equation_coefficients: &TensorBasisVector<F>,
        projections: &[DenseMatmulProjection<F>],
        witness_bva: &Gpt2PrimitiveBvaLayout,
    ) -> Result<TensorBasisVector<F>, Gpt2CheckpointError> {
        if equation_coefficients.layout() != &self.layout || projections.len() != self.vectors.len()
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let mut terms =
            vec![Vec::<(F, Vec<(F, F)>)>::new(); witness_bva.layout.bucket_sizes().len()];
        for (relation_index, (vector, projection)) in
            self.vectors.iter().zip(projections).enumerate()
        {
            let inner_coefficients = equation_coefficients.physical_coefficients(relation_index)?;
            if inner_coefficients.len() != vector.weight_vectors.len() {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            for (inner_coefficient, target_vector) in inner_coefficients
                .into_iter()
                .zip(vector.weight_vectors.iter().copied())
            {
                push_tensor_term(
                    &mut terms,
                    witness_bva,
                    target_vector,
                    projection.batch_weight * inner_coefficient,
                    &projection.row_challenges,
                )?;
            }
        }
        Ok(TensorBasisVector::new(
            &witness_bva.layout,
            terms
                .into_iter()
                .map(TbvBucket::from_factor_terms)
                .collect(),
        )?)
    }
}

impl CausalQkRouting {
    fn projections<F: PrimeField>(
        &self,
        mut next_challenge: impl FnMut() -> F,
    ) -> Result<Vec<CausalQkProjection<F>>, Gpt2CheckpointError> {
        self.relations
            .iter()
            .map(|relation| {
                let output_challenges =
                    random_log_challenges(relation.output_len, &mut next_challenge)?;
                Ok(CausalQkProjection {
                    output_coefficients: tensor_physical_coefficients(
                        relation.output_len,
                        &output_challenges,
                    )?,
                    output_challenges,
                    batch_weight: next_challenge(),
                })
            })
            .collect()
    }

    fn route_output<F: PrimeField>(
        &self,
        projections: &[CausalQkProjection<F>],
        witness_bva: &Gpt2PrimitiveBvaLayout,
    ) -> Result<TensorBasisVector<F>, Gpt2CheckpointError> {
        if projections.len() != self.relations.len() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let mut terms =
            vec![Vec::<(F, Vec<(F, F)>)>::new(); witness_bva.layout.bucket_sizes().len()];
        for (relation, projection) in self.relations.iter().zip(projections) {
            push_tensor_term(
                &mut terms,
                witness_bva,
                relation.output_vector,
                projection.batch_weight,
                &projection.output_challenges,
            )?;
        }
        Ok(TensorBasisVector::new(
            &witness_bva.layout,
            terms
                .into_iter()
                .map(TbvBucket::from_factor_terms)
                .collect(),
        )?)
    }

    fn route_terminal<F: PrimeField>(
        &self,
        equation_coefficients: &TensorBasisVector<F>,
        projections: &[CausalQkProjection<F>],
        witness_bva: &Gpt2PrimitiveBvaLayout,
        left: bool,
    ) -> Result<TensorBasisVector<F>, Gpt2CheckpointError> {
        if equation_coefficients.layout() != &self.layout
            || projections.len() != self.relations.len()
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let mut dense = HashMap::<usize, Vec<F>>::new();
        for (term_index, term) in self.terms.iter().enumerate() {
            let coefficients = equation_coefficients.physical_coefficients(term_index)?;
            if coefficients.len() != term.head_width {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            let (target_vector, start, scale) = if left {
                (term.query_vector, term.query_start, F::from(1_u64))
            } else {
                let projection = projections
                    .get(term.relation_index)
                    .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                (
                    term.key_vector,
                    term.key_start,
                    projection.batch_weight * projection.output_coefficients[term.output_index],
                )
            };
            let physical_len = witness_bva.layout.vector_lengths()[target_vector];
            let target = dense
                .entry(target_vector)
                .or_insert_with(|| vec![F::from(0_u64); physical_len]);
            let end = start
                .checked_add(term.head_width)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
            if end > target.len() {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            for (target, coefficient) in target[start..end].iter_mut().zip(coefficients) {
                *target += scale * coefficient;
            }
        }
        let mut terms =
            vec![Vec::<(F, Vec<(F, F)>)>::new(); witness_bva.layout.bucket_sizes().len()];
        for (target_vector, coefficients) in dense {
            push_dense_terms(&mut terms, witness_bva, target_vector, &coefficients)?;
        }
        Ok(TensorBasisVector::new(
            &witness_bva.layout,
            terms
                .into_iter()
                .map(TbvBucket::from_factor_terms)
                .collect(),
        )?)
    }
}

impl CausalPvRouting {
    fn projections<F: PrimeField>(
        &self,
        mut next_challenge: impl FnMut() -> F,
    ) -> Result<Vec<CausalPvProjection<F>>, Gpt2CheckpointError> {
        self.relations
            .iter()
            .map(|relation| {
                let row_challenges =
                    random_log_challenges(relation.output_rows, &mut next_challenge)?;
                let column_challenges =
                    random_log_challenges(relation.output_columns, &mut next_challenge)?;
                Ok(CausalPvProjection {
                    row_coefficients: tensor_physical_coefficients(
                        relation.output_rows,
                        &row_challenges,
                    )?,
                    column_coefficients: tensor_physical_coefficients(
                        relation.output_columns,
                        &column_challenges,
                    )?,
                    row_challenges,
                    batch_weight: next_challenge(),
                })
            })
            .collect()
    }

    fn output_scale<F: PrimeField>(
        &self,
        term: CausalPvTerm,
        projections: &[CausalPvProjection<F>],
    ) -> Result<F, Gpt2CheckpointError> {
        let projection = projections
            .get(term.relation_index)
            .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
        Ok(projection.batch_weight
            * projection.row_coefficients[term.output_row]
            * projection.column_coefficients[term.output_column])
    }

    fn route_output<F: PrimeField>(
        &self,
        projections: &[CausalPvProjection<F>],
        witness_bva: &Gpt2PrimitiveBvaLayout,
    ) -> Result<TensorBasisVector<F>, Gpt2CheckpointError> {
        if projections.len() != self.relations.len() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let mut terms =
            vec![Vec::<(F, Vec<(F, F)>)>::new(); witness_bva.layout.bucket_sizes().len()];
        for (relation, projection) in self.relations.iter().zip(projections) {
            for (column, target_vector) in relation.output_vectors.iter().copied().enumerate() {
                push_tensor_term(
                    &mut terms,
                    witness_bva,
                    target_vector,
                    projection.batch_weight * projection.column_coefficients[column],
                    &projection.row_challenges,
                )?;
            }
        }
        Ok(TensorBasisVector::new(
            &witness_bva.layout,
            terms
                .into_iter()
                .map(TbvBucket::from_factor_terms)
                .collect(),
        )?)
    }

    fn route_terminal<F: PrimeField>(
        &self,
        equation_coefficients: &TensorBasisVector<F>,
        projections: &[CausalPvProjection<F>],
        witness_bva: &Gpt2PrimitiveBvaLayout,
        left: bool,
    ) -> Result<TensorBasisVector<F>, Gpt2CheckpointError> {
        if equation_coefficients.layout() != &self.layout
            || projections.len() != self.relations.len()
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let mut dense = HashMap::<usize, Vec<F>>::new();
        for (term_index, term) in self.terms.iter().copied().enumerate() {
            let coefficients = equation_coefficients.physical_coefficients(term_index)?;
            if coefficients.len() != term.inner {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            if left {
                let physical_len = witness_bva.layout.vector_lengths()[term.probability_vector];
                let target = dense
                    .entry(term.probability_vector)
                    .or_insert_with(|| vec![F::from(0_u64); physical_len]);
                let end = term
                    .probability_start
                    .checked_add(term.inner)
                    .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
                if end > target.len() {
                    return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                }
                for (target, coefficient) in target[term.probability_start..end]
                    .iter_mut()
                    .zip(coefficients)
                {
                    *target += coefficient;
                }
            } else {
                let scale = self.output_scale(term, projections)?;
                let qkv_vectors = &self
                    .relations
                    .get(term.relation_index)
                    .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?
                    .qkv_vectors;
                for (key_column, coefficient) in coefficients.into_iter().enumerate() {
                    let target_vector = qkv_vectors
                        .get(key_column)
                        .copied()
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                    let physical_len = witness_bva.layout.vector_lengths()[target_vector];
                    let target = dense
                        .entry(target_vector)
                        .or_insert_with(|| vec![F::from(0_u64); physical_len]);
                    if term.value_index >= target.len() {
                        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                    }
                    target[term.value_index] += scale * coefficient;
                }
            }
        }
        let mut terms =
            vec![Vec::<(F, Vec<(F, F)>)>::new(); witness_bva.layout.bucket_sizes().len()];
        for (target_vector, coefficients) in dense {
            push_dense_terms(&mut terms, witness_bva, target_vector, &coefficients)?;
        }
        Ok(TensorBasisVector::new(
            &witness_bva.layout,
            terms
                .into_iter()
                .map(TbvBucket::from_factor_terms)
                .collect(),
        )?)
    }
}

impl<'a, R: Read + Seek, F: PrimeField> Gpt2ElementwisePairSource<'a, R, F> {
    fn new(
        checkpoint: &'a mut Gpt2Q12Checkpoint<R>,
        base: &'a Gpt2GlobalLayout,
        activations: &'a Gpt2ActivationWitness,
        witness: &'a Gpt2PrimitiveWitness,
        routing: &'a ElementwiseBilinearRouting,
        right_coefficients: &'a TensorBasisVector<F>,
    ) -> Result<Self, Gpt2CheckpointError> {
        if routing.layout != *right_coefficients.layout() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        Ok(Self {
            checkpoint,
            base,
            activations,
            witness,
            routing,
            right_coefficients,
        })
    }

    fn read_column(
        &mut self,
        block: GlobalBlockId,
        column: usize,
    ) -> Result<Vec<F>, Gpt2CheckpointError> {
        if let Some(descriptor) = self.base.blocks().get(block.0) {
            if column >= descriptor.cols {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            return match descriptor.role {
                super::Gpt2GlobalBlockRole::Parameter(_) => {
                    let mut integers = vec![0_i32; descriptor.rows];
                    self.checkpoint.read_model_block_column(
                        self.base,
                        block,
                        column,
                        &mut integers,
                    )?;
                    Ok(integers
                        .into_iter()
                        .map(|value| signed_i128_scalar::<F>(i128::from(value)))
                        .collect())
                }
                _ => {
                    let values = self
                        .activations
                        .block(block)
                        .and_then(|activation| activation.columns.get(column))
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                    if values.len() != descriptor.rows {
                        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                    }
                    Ok(values
                        .iter()
                        .map(|value| signed_i128_scalar::<F>(i128::from(*value)))
                        .collect())
                }
            };
        }
        let auxiliary_index = block
            .0
            .checked_sub(self.base.blocks().len())
            .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
        let values = self
            .witness
            .blocks
            .get(auxiliary_index)
            .filter(|witness_block| witness_block.id == block)
            .and_then(|witness_block| witness_block.columns.get(column))
            .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
        Ok(values
            .iter()
            .map(|value| signed_i128_scalar::<F>(*value))
            .collect())
    }
}

impl<R: Read + Seek, F: PrimeField> BvaFieldPairSource<F> for Gpt2ElementwisePairSource<'_, R, F> {
    type Error = Gpt2CheckpointError;

    fn layout(&self) -> &BvaLayout {
        &self.routing.layout
    }

    fn for_each_vector_pair(
        &mut self,
        visit: &mut dyn FnMut(usize, &[F], &[F]),
    ) -> Result<(), Self::Error> {
        for (vector_index, vector) in self.routing.vectors.iter().copied().enumerate() {
            let left = self.read_column(vector.left_block, vector.left_column)?;
            let mut right = self.read_column(vector.right_block, vector.right_column)?;
            if let Some(index) = vector.right_scalar_index {
                let scalar = *right
                    .get(index)
                    .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                right.clear();
                right.resize(left.len(), scalar);
            } else if vector.right_segmented {
                right = expand_segment_coefficients(&right, &self.routing.attention_segments)?;
            }
            if left.len() != right.len()
                || left.len() != self.routing.layout.vector_lengths()[vector_index]
            {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            let right_scale = signed_i128_scalar::<F>(vector.right_scale);
            if right_scale != F::from(1_u64) {
                for value in &mut right {
                    *value *= right_scale;
                }
            }
            for subtract in vector.right_subtract.iter().flatten().copied() {
                let values = self.read_column(subtract.0, subtract.1)?;
                if values.len() != right.len() {
                    return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                }
                for (right, subtract) in right.iter_mut().zip(values) {
                    *right -= subtract;
                }
            }
            let right_constant = signed_i128_scalar::<F>(vector.right_constant);
            if right_constant != F::from(0_u64) {
                for value in &mut right {
                    *value += right_constant;
                }
            }
            self.right_coefficients
                .scale_physical_vector_in_place(vector_index, &mut right)?;
            visit(vector_index, &left, &right);
        }
        Ok(())
    }
}

impl<'a, R: Read + Seek, F: PrimeField> Gpt2DenseMatmulPairSource<'a, R, F> {
    fn new(
        checkpoint: &'a mut Gpt2Q12Checkpoint<R>,
        base: &'a Gpt2GlobalLayout,
        activations: &'a Gpt2ActivationWitness,
        witness: &'a Gpt2PrimitiveWitness,
        routing: &'a DenseMatmulRouting,
        projections: &'a [DenseMatmulProjection<F>],
    ) -> Result<Self, Gpt2CheckpointError> {
        if projections.len() != routing.vectors.len() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        Ok(Self {
            checkpoint,
            base,
            activations,
            witness,
            routing,
            projections,
        })
    }

    fn read_column(
        &mut self,
        block: GlobalBlockId,
        column: usize,
    ) -> Result<Vec<F>, Gpt2CheckpointError> {
        if let Some(descriptor) = self.base.blocks().get(block.0) {
            if column >= descriptor.cols {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            return match descriptor.role {
                super::Gpt2GlobalBlockRole::Parameter(_) => {
                    let mut integers = vec![0_i32; descriptor.rows];
                    self.checkpoint.read_model_block_column(
                        self.base,
                        block,
                        column,
                        &mut integers,
                    )?;
                    Ok(integers
                        .into_iter()
                        .map(|value| signed_i128_scalar::<F>(i128::from(value)))
                        .collect())
                }
                _ => {
                    let values = self
                        .activations
                        .block(block)
                        .and_then(|activation| activation.columns.get(column))
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                    Ok(values
                        .iter()
                        .map(|value| signed_i128_scalar::<F>(i128::from(*value)))
                        .collect())
                }
            };
        }
        let auxiliary_index = block
            .0
            .checked_sub(self.base.blocks().len())
            .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
        let values = self
            .witness
            .blocks
            .get(auxiliary_index)
            .filter(|witness_block| witness_block.id == block)
            .and_then(|witness_block| witness_block.columns.get(column))
            .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
        Ok(values
            .iter()
            .map(|value| signed_i128_scalar::<F>(*value))
            .collect())
    }
}

impl<R: Read + Seek, F: PrimeField> BvaFieldPairSource<F> for Gpt2DenseMatmulPairSource<'_, R, F> {
    type Error = Gpt2CheckpointError;

    fn layout(&self) -> &BvaLayout {
        &self.routing.layout
    }

    fn for_each_vector_pair(
        &mut self,
        visit: &mut dyn FnMut(usize, &[F], &[F]),
    ) -> Result<(), Self::Error> {
        for relation_index in 0..self.routing.vectors.len() {
            let vector = self.routing.vectors[relation_index].clone();
            let projection = self.projections[relation_index].clone();
            let mut left = vec![F::from(0_u64); vector.inner];
            for (column, input_column) in vector.input_columns.iter().copied().enumerate() {
                let values = self.read_column(vector.input_block, input_column)?;
                if values.len() != vector.inner {
                    return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                }
                let coefficient = projection.column_coefficients[column];
                for (accumulator, value) in left.iter_mut().zip(values) {
                    *accumulator += coefficient * value;
                }
            }
            let mut right = Vec::with_capacity(vector.inner);
            for inner_column in 0..vector.inner {
                let values = self.read_column(vector.weight_block, inner_column)?;
                if values.len() != vector.output_rows {
                    return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                }
                let projected = values
                    .iter()
                    .zip(&projection.row_coefficients)
                    .fold(F::from(0_u64), |sum, (value, coefficient)| {
                        sum + *value * coefficient
                    });
                right.push(projection.batch_weight * projected);
            }
            visit(relation_index, &left, &right);
        }
        Ok(())
    }
}

impl<'a, F: PrimeField> Gpt2CausalQkPairSource<'a, F> {
    fn new(
        base: &'a Gpt2GlobalLayout,
        activations: &'a Gpt2ActivationWitness,
        routing: &'a CausalQkRouting,
        projections: &'a [CausalQkProjection<F>],
    ) -> Result<Self, Gpt2CheckpointError> {
        if projections.len() != routing.relations.len() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        Ok(Self {
            base,
            activations,
            routing,
            projections,
        })
    }
}

impl<F: PrimeField> BvaFieldPairSource<F> for Gpt2CausalQkPairSource<'_, F> {
    type Error = Gpt2CheckpointError;

    fn layout(&self) -> &BvaLayout {
        &self.routing.layout
    }

    fn for_each_vector_pair(
        &mut self,
        visit: &mut dyn FnMut(usize, &[F], &[F]),
    ) -> Result<(), Self::Error> {
        for (term_index, term) in self.routing.terms.iter().copied().enumerate() {
            let descriptor = self
                .base
                .blocks()
                .get(term.qkv_block.0)
                .filter(|descriptor| descriptor.id == term.qkv_block)
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            let activation = self
                .activations
                .block(term.qkv_block)
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            let query = activation
                .columns
                .get(term.query_column)
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            let key = activation
                .columns
                .get(term.key_column)
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            if query.len() != descriptor.rows || key.len() != descriptor.rows {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            let query_end = term.query_start + term.head_width;
            let key_end = term.key_start + term.head_width;
            let left = query
                .get(term.query_start..query_end)
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?
                .iter()
                .map(|value| signed_i128_scalar::<F>(i128::from(*value)))
                .collect::<Vec<_>>();
            let projection = self
                .projections
                .get(term.relation_index)
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            let scale = projection.batch_weight * projection.output_coefficients[term.output_index];
            let right = key
                .get(term.key_start..key_end)
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?
                .iter()
                .map(|value| scale * signed_i128_scalar::<F>(i128::from(*value)))
                .collect::<Vec<_>>();
            visit(term_index, &left, &right);
        }
        Ok(())
    }
}

impl<'a, F: PrimeField> Gpt2CausalPvPairSource<'a, F> {
    fn new(
        base: &'a Gpt2GlobalLayout,
        activations: &'a Gpt2ActivationWitness,
        witness: &'a Gpt2PrimitiveWitness,
        routing: &'a CausalPvRouting,
        projections: &'a [CausalPvProjection<F>],
    ) -> Result<Self, Gpt2CheckpointError> {
        if projections.len() != routing.relations.len() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        Ok(Self {
            base,
            activations,
            witness,
            routing,
            projections,
        })
    }
}

impl<F: PrimeField> BvaFieldPairSource<F> for Gpt2CausalPvPairSource<'_, F> {
    type Error = Gpt2CheckpointError;

    fn layout(&self) -> &BvaLayout {
        &self.routing.layout
    }

    fn for_each_vector_pair(
        &mut self,
        visit: &mut dyn FnMut(usize, &[F], &[F]),
    ) -> Result<(), Self::Error> {
        for (term_index, term) in self.routing.terms.iter().copied().enumerate() {
            let auxiliary_index = term
                .probability_block
                .0
                .checked_sub(self.base.blocks().len())
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            let probability = self
                .witness
                .blocks
                .get(auxiliary_index)
                .filter(|block| block.id == term.probability_block)
                .and_then(|block| block.columns.first())
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            let probability_end = term
                .probability_start
                .checked_add(term.inner)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
            let left = probability
                .get(term.probability_start..probability_end)
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?
                .iter()
                .map(|value| signed_i128_scalar::<F>(*value))
                .collect::<Vec<_>>();
            let activation = self
                .activations
                .block(term.qkv_block)
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            let scale = self.routing.output_scale(term, self.projections)?;
            let mut right = Vec::with_capacity(term.inner);
            for key_column in 0..term.inner {
                let value = activation
                    .columns
                    .get(key_column)
                    .and_then(|column| column.get(term.value_index))
                    .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                right.push(scale * signed_i128_scalar::<F>(i128::from(*value)));
            }
            visit(term_index, &left, &right);
        }
        Ok(())
    }
}

impl Gpt2PrimitiveWitness {
    fn build<R: Read + Seek>(
        checkpoint: &mut Gpt2Q12Checkpoint<R>,
        layout: &Gpt2GlobalLayout,
        activations: &Gpt2ActivationWitness,
        manifest: &Gpt2PrimitiveRelationManifest,
    ) -> Result<Self, Gpt2CheckpointError> {
        if *manifest != Gpt2PrimitiveRelationManifest::compile(checkpoint.metadata(), layout)? {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let metadata = checkpoint.metadata().clone();
        let operation_manifest = metadata.operation_manifest()?;
        let mut builder = AuxiliaryWitnessBuilder {
            manifest,
            cursor: 0,
            blocks: Vec::with_capacity(manifest.auxiliary_blocks.len()),
        };
        for (operation_index, descriptor) in operation_manifest.iter().enumerate() {
            match descriptor.kind {
                Gpt2Q12OperationKind::Embedding
                | Gpt2Q12OperationKind::AttentionResidual
                | Gpt2Q12OperationKind::MlpResidual => {}
                Gpt2Q12OperationKind::Gelu => build_gelu_witness(
                    layout,
                    activations,
                    &mut builder,
                    operation_index,
                    metadata.scale(),
                )?,
                Gpt2Q12OperationKind::LayerNorm1
                | Gpt2Q12OperationKind::LayerNorm2
                | Gpt2Q12OperationKind::FinalLayerNorm => build_layer_norm_witness(
                    checkpoint,
                    layout,
                    activations,
                    &mut builder,
                    operation_index,
                    metadata.scale(),
                )?,
                Gpt2Q12OperationKind::QkvProjection
                | Gpt2Q12OperationKind::AttentionProjection
                | Gpt2Q12OperationKind::MlpExpansion
                | Gpt2Q12OperationKind::MlpProjection => build_dense_witness(
                    checkpoint,
                    layout,
                    activations,
                    &mut builder,
                    operation_index,
                    metadata.scale(),
                    true,
                    false,
                )?,
                Gpt2Q12OperationKind::LmHead => build_dense_witness(
                    checkpoint,
                    layout,
                    activations,
                    &mut builder,
                    operation_index,
                    metadata.scale(),
                    false,
                    true,
                )?,
                Gpt2Q12OperationKind::CausalAttention => build_attention_witness(
                    layout,
                    activations,
                    &mut builder,
                    operation_index,
                    &metadata,
                )?,
            }
        }
        let witness = builder.finish()?;
        witness.audit(manifest)?;
        witness.audit_declared_ranges(manifest)?;
        Ok(witness)
    }

    fn block_count(&self) -> usize {
        self.blocks.len()
    }

    fn stored_entries(&self) -> usize {
        self.blocks
            .iter()
            .flat_map(|block| &block.columns)
            .map(Vec::len)
            .sum()
    }

    fn satisfies<R: Read + Seek>(
        &self,
        checkpoint: &mut Gpt2Q12Checkpoint<R>,
        layout: &Gpt2GlobalLayout,
        activations: &Gpt2ActivationWitness,
        manifest: &Gpt2PrimitiveRelationManifest,
    ) -> Result<bool, Gpt2CheckpointError> {
        let expected = Self::build(checkpoint, layout, activations, manifest)?;
        Ok(*self == expected)
    }

    fn lookup_queries(
        &self,
        layout: &Gpt2GlobalLayout,
        activations: &Gpt2ActivationWitness,
        manifest: &Gpt2PrimitiveRelationManifest,
    ) -> Result<Vec<TaggedLookupQuery>, Gpt2CheckpointError> {
        self.audit(manifest)?;
        let mut queries = Vec::with_capacity(manifest.lookup_scalar_count()?);
        for relation in manifest.relations.iter().filter(|relation| {
            relation.family == RelationFamily::TaggedLookup
                && relation.kind <= TaggedLookupRelationKind::InverseSqrt as u8
        }) {
            if relation.blocks.len() != 2 {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            let tag = match relation.kind {
                value if value == TaggedLookupRelationKind::Gelu as u8 => {
                    TaggedLookupRelationKind::Gelu
                }
                value if value == TaggedLookupRelationKind::Exp as u8 => {
                    TaggedLookupRelationKind::Exp
                }
                value if value == TaggedLookupRelationKind::Reciprocal as u8 => {
                    TaggedLookupRelationKind::Reciprocal
                }
                value if value == TaggedLookupRelationKind::InverseSqrt as u8 => {
                    TaggedLookupRelationKind::InverseSqrt
                }
                _ => return Err(Gpt2CheckpointError::ProtocolShapeMismatch),
            };
            let input = relation_block_values(relation.blocks[0], layout, activations, self)?;
            let output = relation_block_values(relation.blocks[1], layout, activations, self)?;
            if input.len() != relation.scalar_count || output.len() != relation.scalar_count {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            queries.extend(
                input
                    .into_iter()
                    .zip(output)
                    .map(|(input, output)| TaggedLookupQuery { tag, input, output }),
            );
        }
        Ok(queries)
    }

    fn audit(&self, manifest: &Gpt2PrimitiveRelationManifest) -> Result<(), Gpt2CheckpointError> {
        if self.blocks.len() != manifest.auxiliary_blocks.len() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        for (block, descriptor) in self.blocks.iter().zip(&manifest.auxiliary_blocks) {
            if block.id != descriptor.id
                || block.columns.len() != descriptor.cols
                || block
                    .columns
                    .iter()
                    .any(|column| column.len() != descriptor.rows)
            {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
        }
        Ok(())
    }

    fn audit_declared_ranges(
        &self,
        manifest: &Gpt2PrimitiveRelationManifest,
    ) -> Result<(), Gpt2CheckpointError> {
        self.audit(manifest)?;
        for (block, descriptor) in self.blocks.iter().zip(&manifest.auxiliary_blocks) {
            let signed = auxiliary_kind_is_signed(descriptor.kind);
            for value in block.columns.iter().flatten() {
                let in_range = if signed {
                    if descriptor.value_bits == 128 {
                        true
                    } else {
                        let limit = 1_i128 << (descriptor.value_bits - 1);
                        *value >= -limit && *value < limit
                    }
                } else if *value < 0 {
                    false
                } else if descriptor.value_bits == 128 {
                    true
                } else {
                    (*value as u128) < (1_u128 << descriptor.value_bits)
                };
                if !in_range {
                    return Err(Gpt2CheckpointError::DeclaredRangeViolation {
                        block: block.id.0,
                        value_bits: descriptor.value_bits,
                    });
                }
            }
        }
        Ok(())
    }
}

impl<'a, R: Read + Seek> Gpt2LookupQueryStream<'a, R> {
    fn new(
        checkpoint: &'a mut Gpt2Q12Checkpoint<R>,
        layout: &'a Gpt2GlobalLayout,
        activations: &'a Gpt2ActivationWitness,
        witness: &'a Gpt2PrimitiveWitness,
        manifest: &'a Gpt2PrimitiveRelationManifest,
    ) -> Result<Self, Gpt2CheckpointError> {
        witness.audit(manifest)?;
        Ok(Self {
            checkpoint,
            layout,
            activations,
            witness,
            manifest,
        })
    }

    fn for_each(
        &mut self,
        visit: &mut dyn FnMut(TaggedLookupQuery) -> Result<(), Gpt2CheckpointError>,
    ) -> Result<LookupStreamProfile, Gpt2CheckpointError> {
        let mut semantic_queries = 0_usize;
        let mut maximum_live_semantic_entries = 0_usize;
        for relation in self.manifest.relations.iter().filter(|relation| {
            relation.family == RelationFamily::TaggedLookup
                && relation.kind <= TaggedLookupRelationKind::InverseSqrt as u8
        }) {
            if relation.blocks.len() != 2 {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            let tag = semantic_lookup_kind(relation.kind)?;
            let input = relation_block_values(
                relation.blocks[0],
                self.layout,
                self.activations,
                self.witness,
            )?;
            let output = relation_block_values(
                relation.blocks[1],
                self.layout,
                self.activations,
                self.witness,
            )?;
            if input.len() != relation.scalar_count || output.len() != relation.scalar_count {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            maximum_live_semantic_entries = maximum_live_semantic_entries
                .max(input.len().checked_add(output.len()).unwrap_or(usize::MAX));
            for (input, output) in input.into_iter().zip(output) {
                visit(TaggedLookupQuery { tag, input, output })?;
                semantic_queries = semantic_queries
                    .checked_add(1)
                    .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
            }
        }

        let mut range_queries = 0_usize;
        let mut integer_buffer = Vec::new();
        let mut group_start = 0_usize;
        while group_start < self.manifest.range_limb_blocks.len() {
            let source_kind = self.manifest.range_limb_blocks[group_start].source_kind;
            let mut group_end = group_start + 1;
            while group_end < self.manifest.range_limb_blocks.len()
                && self.manifest.range_limb_blocks[group_end].source_kind == source_kind
            {
                group_end += 1;
            }
            let limb_blocks = &self.manifest.range_limb_blocks[group_start..group_end];
            match source_kind {
                RangeLimbSource::Base { block_index } => {
                    let descriptor = self
                        .layout
                        .blocks()
                        .get(block_index)
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                    let encoding = RangeEncoding::new(true, 32)?;
                    for column in 0..descriptor.cols {
                        integer_buffer.resize(descriptor.rows, 0_i32);
                        match descriptor.role {
                            super::Gpt2GlobalBlockRole::Parameter(_) => {
                                self.checkpoint.read_model_block_column(
                                    self.layout,
                                    descriptor.id,
                                    column,
                                    &mut integer_buffer,
                                )?;
                            }
                            super::Gpt2GlobalBlockRole::OperationOutput(_) => {
                                let values = self
                                    .activations
                                    .block(descriptor.id)
                                    .and_then(|block| block.columns.get(column))
                                    .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                                if values.len() != integer_buffer.len() {
                                    return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                                }
                                integer_buffer.copy_from_slice(values);
                            }
                            super::Gpt2GlobalBlockRole::PublicTokenIds => {
                                return Err(Gpt2CheckpointError::ProtocolShapeMismatch)
                            }
                        }
                        maximum_live_semantic_entries =
                            maximum_live_semantic_entries.max(integer_buffer.len());
                        for range_block in limb_blocks {
                            for value in &integer_buffer {
                                let encoded = encoding.encoded_value(i128::from(*value))?;
                                let input = ((encoded >> (8 * range_block.limb)) & 0xff) as i128;
                                visit(TaggedLookupQuery {
                                    tag: range_lookup_kind(range_block.limb_bits),
                                    input,
                                    output: 0,
                                })?;
                                range_queries = range_queries
                                    .checked_add(1)
                                    .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
                            }
                        }
                    }
                }
                RangeLimbSource::Auxiliary { block_index } => {
                    let source_descriptor = self
                        .manifest
                        .auxiliary_blocks
                        .get(block_index)
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                    let source = self
                        .witness
                        .blocks
                        .get(block_index)
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                    let encoding = RangeEncoding::new(
                        auxiliary_kind_is_signed(source_descriptor.kind),
                        source_descriptor.value_bits,
                    )?;
                    for column in &source.columns {
                        maximum_live_semantic_entries =
                            maximum_live_semantic_entries.max(column.len());
                        for range_block in limb_blocks {
                            for value in column {
                                let encoded = encoding.encoded_value(*value)?;
                                let input = ((encoded >> (8 * range_block.limb)) & 0xff) as i128;
                                visit(TaggedLookupQuery {
                                    tag: range_lookup_kind(range_block.limb_bits),
                                    input,
                                    output: 0,
                                })?;
                                range_queries = range_queries
                                    .checked_add(1)
                                    .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
                            }
                        }
                    }
                }
            }
            group_start = group_end;
        }
        Ok(LookupStreamProfile {
            semantic_queries,
            range_queries,
            maximum_live_semantic_entries,
        })
    }

    fn lasso_products<F: PrimeField>(
        &mut self,
        registry: &Gpt2LookupRegistry,
        row_challenge: F,
        product_challenge: F,
    ) -> Result<StreamingLassoProducts<F>, Gpt2CheckpointError> {
        if row_challenge == F::from(0_u64) || product_challenge == F::from(0_u64) {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let mut counts = HashMap::<usize, u64>::new();
        let mut left = F::from(1_u64);
        let mut right = F::from(1_u64);
        let product_challenge_squared = product_challenge * product_challenge;
        let profile = self.for_each(&mut |query| {
            registry.audit_query(query)?;
            let compressed = compress_lookup_query(query, row_challenge);
            let table_index = registry.table_index(query)?;
            let count = counts.entry(table_index).or_insert(0);
            let prefix_count = *count;
            left *= F::from(1_u64)
                + product_challenge * compressed
                + product_challenge_squared * F::from(prefix_count);
            right *= F::from(1_u64)
                + product_challenge * compressed
                + product_challenge_squared
                    * F::from(
                        prefix_count
                            .checked_add(1)
                            .ok_or(Gpt2CheckpointError::OffsetOverflow)?,
                    );
            *count = count
                .checked_add(1)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
            Ok(())
        })?;
        let mut multiplicities = counts.into_iter().collect::<Vec<_>>();
        multiplicities.sort_by_key(|(table_index, _)| *table_index);
        for (table_index, count) in &multiplicities {
            let compressed =
                compress_lookup_query(registry.table_row(*table_index)?, row_challenge);
            left *= F::from(1_u64)
                + product_challenge * compressed
                + product_challenge_squared * F::from(*count);
            right *= F::from(1_u64) + product_challenge * compressed;
        }
        Ok(StreamingLassoProducts {
            left,
            right,
            query_count: profile
                .semantic_queries
                .checked_add(profile.range_queries)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?,
            distinct_table_rows: multiplicities.len(),
            public_table_rows: registry.public_table_rows()?,
            maximum_live_semantic_entries: profile.maximum_live_semantic_entries,
            multiplicities,
        })
    }
}

impl<'a, R: Read + Seek, F: PrimeField> Gpt2PrimitiveBvaFieldSource<'a, R, F> {
    fn new(
        checkpoint: &'a mut Gpt2Q12Checkpoint<R>,
        layout: &'a Gpt2GlobalLayout,
        activations: &'a Gpt2ActivationWitness,
        manifest: &'a Gpt2PrimitiveRelationManifest,
        witness: &'a Gpt2PrimitiveWitness,
        bva: &'a Gpt2PrimitiveBvaLayout,
    ) -> Result<Self, Gpt2CheckpointError> {
        if *manifest != Gpt2PrimitiveRelationManifest::compile(checkpoint.metadata(), layout)?
            || *bva != manifest.compile_bva(layout)?
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        witness.audit(manifest)?;
        Ok(Self {
            checkpoint,
            layout,
            activations,
            manifest,
            witness,
            bva,
            _field: PhantomData,
        })
    }

    fn elementwise_pair_source<'b>(
        &'b mut self,
        routing: &'b ElementwiseBilinearRouting,
        right_coefficients: &'b TensorBasisVector<F>,
    ) -> Result<Gpt2ElementwisePairSource<'b, R, F>, Gpt2CheckpointError> {
        Gpt2ElementwisePairSource::new(
            self.checkpoint,
            self.layout,
            self.activations,
            self.witness,
            routing,
            right_coefficients,
        )
    }

    fn dense_matmul_pair_source<'b>(
        &'b mut self,
        routing: &'b DenseMatmulRouting,
        projections: &'b [DenseMatmulProjection<F>],
    ) -> Result<Gpt2DenseMatmulPairSource<'b, R, F>, Gpt2CheckpointError> {
        Gpt2DenseMatmulPairSource::new(
            self.checkpoint,
            self.layout,
            self.activations,
            self.witness,
            routing,
            projections,
        )
    }

    fn causal_qk_pair_source<'b>(
        &'b self,
        routing: &'b CausalQkRouting,
        projections: &'b [CausalQkProjection<F>],
    ) -> Result<Gpt2CausalQkPairSource<'b, F>, Gpt2CheckpointError> {
        Gpt2CausalQkPairSource::new(self.layout, self.activations, routing, projections)
    }

    fn causal_pv_pair_source<'b>(
        &'b self,
        routing: &'b CausalPvRouting,
        projections: &'b [CausalPvProjection<F>],
    ) -> Result<Gpt2CausalPvPairSource<'b, F>, Gpt2CheckpointError> {
        Gpt2CausalPvPairSource::new(
            self.layout,
            self.activations,
            self.witness,
            routing,
            projections,
        )
    }
}

impl<R: Read + Seek, F: PrimeField> BvaFieldSource<F> for Gpt2PrimitiveBvaFieldSource<'_, R, F> {
    type Error = Gpt2CheckpointError;

    fn layout(&self) -> &BvaLayout {
        &self.bva.layout
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[F])) -> Result<(), Self::Error> {
        let mut integer_buffer = Vec::new();
        let mut field_buffer = Vec::new();
        for (vector_index, source) in self.bva.vectors.iter().enumerate() {
            match *source {
                PrimitiveBvaVectorSource::Base { block, column } => {
                    let descriptor = self
                        .layout
                        .blocks()
                        .get(block.0)
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                    integer_buffer.resize(descriptor.rows, 0);
                    match descriptor.role {
                        super::Gpt2GlobalBlockRole::Parameter(_) => {
                            self.checkpoint.read_model_block_column(
                                self.layout,
                                block,
                                column,
                                &mut integer_buffer,
                            )?;
                        }
                        _ => {
                            let activation = self
                                .activations
                                .block(block)
                                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                            let values = activation
                                .columns
                                .get(column)
                                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                            if values.len() != integer_buffer.len() {
                                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                            }
                            integer_buffer.copy_from_slice(values);
                        }
                    }
                    field_buffer.clear();
                    field_buffer.extend(
                        integer_buffer
                            .iter()
                            .map(|value| signed_i128_scalar::<F>(i128::from(*value))),
                    );
                    visit(vector_index, &field_buffer);
                }
                PrimitiveBvaVectorSource::Auxiliary {
                    block_index,
                    column,
                } => {
                    let values = self
                        .witness
                        .blocks
                        .get(block_index)
                        .and_then(|block| block.columns.get(column))
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                    field_buffer.clear();
                    field_buffer.extend(values.iter().map(|value| signed_i128_scalar::<F>(*value)));
                    visit(vector_index, &field_buffer);
                }
                PrimitiveBvaVectorSource::RangeLimb {
                    block_index,
                    column,
                } => {
                    let range_block = self
                        .manifest
                        .range_limb_blocks
                        .get(block_index)
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                    if range_block.cols <= column {
                        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                    }
                    field_buffer.clear();
                    match range_block.source_kind {
                        RangeLimbSource::Base { block_index } => {
                            let descriptor = self
                                .layout
                                .blocks()
                                .get(block_index)
                                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                            if descriptor.id != range_block.source
                                || descriptor.rows != range_block.rows
                            {
                                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                            }
                            integer_buffer.resize(descriptor.rows, 0);
                            match descriptor.role {
                                super::Gpt2GlobalBlockRole::Parameter(_) => {
                                    self.checkpoint.read_model_block_column(
                                        self.layout,
                                        descriptor.id,
                                        column,
                                        &mut integer_buffer,
                                    )?;
                                }
                                super::Gpt2GlobalBlockRole::OperationOutput(_) => {
                                    let values = self
                                        .activations
                                        .block(descriptor.id)
                                        .and_then(|block| block.columns.get(column))
                                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                                    if values.len() != integer_buffer.len() {
                                        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                                    }
                                    integer_buffer.copy_from_slice(values);
                                }
                                super::Gpt2GlobalBlockRole::PublicTokenIds => {
                                    return Err(Gpt2CheckpointError::ProtocolShapeMismatch)
                                }
                            }
                            let encoding = RangeEncoding::new(true, 32)?;
                            field_buffer.reserve(integer_buffer.len());
                            for value in &integer_buffer {
                                let encoded = encoding.encoded_value(i128::from(*value))?;
                                let limb = (encoded >> (8 * range_block.limb)) & 0xff;
                                field_buffer.push(F::from(limb as u64));
                            }
                        }
                        RangeLimbSource::Auxiliary { block_index } => {
                            let source_descriptor = self
                                .manifest
                                .auxiliary_blocks
                                .get(block_index)
                                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                            let source = self
                                .witness
                                .blocks
                                .get(block_index)
                                .and_then(|block| block.columns.get(column))
                                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                            if source.len() != range_block.rows {
                                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                            }
                            let encoding = RangeEncoding::new(
                                auxiliary_kind_is_signed(source_descriptor.kind),
                                source_descriptor.value_bits,
                            )?;
                            field_buffer.reserve(source.len());
                            for value in source {
                                let encoded = encoding.encoded_value(*value)?;
                                let limb = (encoded >> (8 * range_block.limb)) & 0xff;
                                field_buffer.push(F::from(limb as u64));
                            }
                        }
                    }
                    visit(vector_index, &field_buffer);
                }
            }
        }
        Ok(())
    }
}

impl Gpt2LookupRegistry {
    fn exact(metadata: &Gpt2Q12Metadata) -> Self {
        Self {
            scale: i64::from(metadata.scale()),
            maximum_tokens: metadata.token_count,
        }
    }

    fn table_len(&self, tag: TaggedLookupRelationKind) -> Result<usize, Gpt2CheckpointError> {
        let scale =
            usize::try_from(self.scale).map_err(|_| Gpt2CheckpointError::ArithmeticOverflow)?;
        match tag {
            TaggedLookupRelationKind::Gelu | TaggedLookupRelationKind::Exp => scale
                .checked_mul(16)
                .and_then(|length| length.checked_add(1))
                .ok_or(Gpt2CheckpointError::OffsetOverflow),
            TaggedLookupRelationKind::Reciprocal => self
                .maximum_tokens
                .checked_mul(scale)
                .ok_or(Gpt2CheckpointError::OffsetOverflow),
            TaggedLookupRelationKind::InverseSqrt => Ok(1 << 25),
            tag @ (TaggedLookupRelationKind::Range1
            | TaggedLookupRelationKind::Range2
            | TaggedLookupRelationKind::Range3
            | TaggedLookupRelationKind::Range4
            | TaggedLookupRelationKind::Range5
            | TaggedLookupRelationKind::Range6
            | TaggedLookupRelationKind::Range7
            | TaggedLookupRelationKind::Range8) => Ok(1 << range_tag_bits(tag)),
        }
    }

    fn public_table_rows(&self) -> Result<usize, Gpt2CheckpointError> {
        TAGGED_LOOKUP_TABLE_ORDER
            .iter()
            .try_fold(0_usize, |sum, tag| {
                sum.checked_add(self.table_len(*tag)?)
                    .ok_or(Gpt2CheckpointError::OffsetOverflow)
            })
    }

    fn table_offset(&self, tag: TaggedLookupRelationKind) -> Result<usize, Gpt2CheckpointError> {
        let mut offset = 0_usize;
        for candidate in TAGGED_LOOKUP_TABLE_ORDER {
            if candidate == tag {
                return Ok(offset);
            }
            offset = offset
                .checked_add(self.table_len(candidate)?)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        }
        Err(Gpt2CheckpointError::ProtocolShapeMismatch)
    }

    fn table_index(&self, query: TaggedLookupQuery) -> Result<usize, Gpt2CheckpointError> {
        self.audit_query(query)?;
        let input =
            i64::try_from(query.input).map_err(|_| Gpt2CheckpointError::ArithmeticOverflow)?;
        let local = match query.tag {
            TaggedLookupRelationKind::Gelu => input
                .checked_add(8 * self.scale)
                .and_then(|value| usize::try_from(value).ok()),
            TaggedLookupRelationKind::Exp => usize::try_from(input).ok(),
            TaggedLookupRelationKind::Reciprocal | TaggedLookupRelationKind::InverseSqrt => input
                .checked_sub(1)
                .and_then(|value| usize::try_from(value).ok()),
            TaggedLookupRelationKind::Range1
            | TaggedLookupRelationKind::Range2
            | TaggedLookupRelationKind::Range3
            | TaggedLookupRelationKind::Range4
            | TaggedLookupRelationKind::Range5
            | TaggedLookupRelationKind::Range6
            | TaggedLookupRelationKind::Range7
            | TaggedLookupRelationKind::Range8 => usize::try_from(input).ok(),
        }
        .ok_or(Gpt2CheckpointError::LookupRegistryMismatch)?;
        if local >= self.table_len(query.tag)? {
            return Err(Gpt2CheckpointError::LookupRegistryMismatch);
        }
        self.table_offset(query.tag)?
            .checked_add(local)
            .ok_or(Gpt2CheckpointError::OffsetOverflow)
    }

    fn table_row(&self, mut index: usize) -> Result<TaggedLookupQuery, Gpt2CheckpointError> {
        for tag in TAGGED_LOOKUP_TABLE_ORDER {
            let length = self.table_len(tag)?;
            if index >= length {
                index -= length;
                continue;
            }
            let input = match tag {
                TaggedLookupRelationKind::Gelu => i64::try_from(index)
                    .ok()
                    .and_then(|index| index.checked_sub(8 * self.scale)),
                TaggedLookupRelationKind::Exp => i64::try_from(index).ok(),
                TaggedLookupRelationKind::Reciprocal | TaggedLookupRelationKind::InverseSqrt => {
                    i64::try_from(index)
                        .ok()
                        .and_then(|index| index.checked_add(1))
                }
                TaggedLookupRelationKind::Range1
                | TaggedLookupRelationKind::Range2
                | TaggedLookupRelationKind::Range3
                | TaggedLookupRelationKind::Range4
                | TaggedLookupRelationKind::Range5
                | TaggedLookupRelationKind::Range6
                | TaggedLookupRelationKind::Range7
                | TaggedLookupRelationKind::Range8 => i64::try_from(index).ok(),
            }
            .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
            let output = match tag {
                TaggedLookupRelationKind::Gelu => q12_gelu(input, self.scale)?,
                TaggedLookupRelationKind::Exp => q12_exp(
                    input
                        .checked_neg()
                        .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?,
                    self.scale,
                )?,
                TaggedLookupRelationKind::Reciprocal => {
                    q12_reciprocal(input, self.maximum_tokens, self.scale)?
                }
                TaggedLookupRelationKind::InverseSqrt => q12_rsqrt(input, self.scale)?,
                TaggedLookupRelationKind::Range1
                | TaggedLookupRelationKind::Range2
                | TaggedLookupRelationKind::Range3
                | TaggedLookupRelationKind::Range4
                | TaggedLookupRelationKind::Range5
                | TaggedLookupRelationKind::Range6
                | TaggedLookupRelationKind::Range7
                | TaggedLookupRelationKind::Range8 => 0,
            };
            return Ok(TaggedLookupQuery {
                tag,
                input: i128::from(input),
                output: i128::from(output),
            });
        }
        Err(Gpt2CheckpointError::LookupRegistryMismatch)
    }

    fn audit(&self, queries: &[TaggedLookupQuery]) -> Result<(), Gpt2CheckpointError> {
        for query in queries {
            self.audit_query(*query)?;
        }
        Ok(())
    }

    fn audit_query(&self, query: TaggedLookupQuery) -> Result<(), Gpt2CheckpointError> {
        let input =
            i64::try_from(query.input).map_err(|_| Gpt2CheckpointError::ArithmeticOverflow)?;
        let expected = match query.tag {
            TaggedLookupRelationKind::Gelu => {
                if input < -8 * self.scale || input > 8 * self.scale {
                    return Err(Gpt2CheckpointError::LookupRegistryMismatch);
                }
                i128::from(q12_gelu(input, self.scale)?)
            }
            TaggedLookupRelationKind::Exp => {
                if input < 0 || input > 16 * self.scale {
                    return Err(Gpt2CheckpointError::LookupRegistryMismatch);
                }
                let shifted = input
                    .checked_neg()
                    .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
                i128::from(q12_exp(shifted, self.scale)?)
            }
            TaggedLookupRelationKind::Reciprocal => {
                let maximum = i64::try_from(self.maximum_tokens)
                    .ok()
                    .and_then(|tokens| tokens.checked_mul(self.scale))
                    .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
                if !(1..=maximum).contains(&input) {
                    return Err(Gpt2CheckpointError::LookupRegistryMismatch);
                }
                i128::from(q12_reciprocal(input, self.maximum_tokens, self.scale)?)
            }
            TaggedLookupRelationKind::InverseSqrt => {
                if !(1..=1 << 25).contains(&input) {
                    return Err(Gpt2CheckpointError::LookupRegistryMismatch);
                }
                i128::from(q12_rsqrt(input, self.scale)?)
            }
            tag @ (TaggedLookupRelationKind::Range1
            | TaggedLookupRelationKind::Range2
            | TaggedLookupRelationKind::Range3
            | TaggedLookupRelationKind::Range4
            | TaggedLookupRelationKind::Range5
            | TaggedLookupRelationKind::Range6
            | TaggedLookupRelationKind::Range7
            | TaggedLookupRelationKind::Range8) => {
                let bits = range_tag_bits(tag);
                if query.output != 0 || input < 0 || input >= (1_i64 << bits) {
                    return Err(Gpt2CheckpointError::LookupRegistryMismatch);
                }
                0
            }
        };
        if query.output != expected {
            return Err(Gpt2CheckpointError::LookupRegistryMismatch);
        }
        Ok(())
    }
}

struct AuxiliaryWitnessBuilder<'a> {
    manifest: &'a Gpt2PrimitiveRelationManifest,
    cursor: usize,
    blocks: Vec<AuxiliaryWitnessBlock>,
}

impl AuxiliaryWitnessBuilder<'_> {
    fn push(
        &mut self,
        operation_index: usize,
        kind: AuxiliaryBlockKind,
        columns: Vec<Vec<i128>>,
    ) -> Result<GlobalBlockId, Gpt2CheckpointError> {
        let descriptor = self
            .manifest
            .auxiliary_blocks
            .get(self.cursor)
            .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
        if descriptor.operation_index != operation_index
            || descriptor.kind != kind
            || columns.len() != descriptor.cols
            || columns.iter().any(|column| column.len() != descriptor.rows)
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        self.blocks.push(AuxiliaryWitnessBlock {
            id: descriptor.id,
            columns,
        });
        self.cursor += 1;
        Ok(descriptor.id)
    }

    fn finish(self) -> Result<Gpt2PrimitiveWitness, Gpt2CheckpointError> {
        if self.cursor != self.manifest.auxiliary_blocks.len() {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        Ok(Gpt2PrimitiveWitness {
            blocks: self.blocks,
        })
    }
}

fn build_gelu_witness(
    layout: &Gpt2GlobalLayout,
    activations: &Gpt2ActivationWitness,
    builder: &mut AuxiliaryWitnessBuilder<'_>,
    operation_index: usize,
    scale: i32,
) -> Result<(), Gpt2CheckpointError> {
    let operation = operation(layout, operation_index)?;
    let input = activation_matrix(activations, required(operation.inputs[0])?)?;
    let expected = activation_matrix(activations, operation.output)?;
    if input.len() != expected.len()
        || input
            .iter()
            .zip(&expected)
            .any(|(left, right)| left.len() != right.len())
    {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    let clamp = clamp_witness_columns(&input, -8 * i128::from(scale), 8 * i128::from(scale))?;
    let computed = clamp
        .clamped
        .iter()
        .map(|column| {
            column
                .iter()
                .map(|value| {
                    Ok(i128::from(q12_gelu(
                        i64::try_from(*value)
                            .map_err(|_| Gpt2CheckpointError::ArithmeticOverflow)?,
                        i64::from(scale),
                    )?))
                })
                .collect::<Result<Vec<_>, Gpt2CheckpointError>>()
        })
        .collect::<Result<Vec<_>, _>>()?;
    if computed != expected {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    push_clamp_witness(builder, operation_index, true, clamp)?;
    Ok(())
}

fn build_layer_norm_witness<R: Read + Seek>(
    checkpoint: &mut Gpt2Q12Checkpoint<R>,
    layout: &Gpt2GlobalLayout,
    activations: &Gpt2ActivationWitness,
    builder: &mut AuxiliaryWitnessBuilder<'_>,
    operation_index: usize,
    scale: i32,
) -> Result<(), Gpt2CheckpointError> {
    let operation = operation(layout, operation_index)?;
    let input = activation_matrix(activations, required(operation.inputs[0])?)?;
    let expected = activation_matrix(activations, operation.output)?;
    let gamma = checkpoint
        .read_tensor_i64(required(operation.parameter_blocks[0])?.0)?
        .into_iter()
        .map(i128::from)
        .collect::<Vec<_>>();
    let beta = checkpoint
        .read_tensor_i64(required(operation.parameter_blocks[1])?.0)?
        .into_iter()
        .map(i128::from)
        .collect::<Vec<_>>();
    let width = input.first().map(Vec::len).unwrap_or(0);
    if width == 0
        || input.iter().any(|row| row.len() != width)
        || gamma.len() != width
        || beta.len() != width
        || expected.len() != input.len()
    {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    let divisor = width as i128;
    let q = i128::from(scale);
    let mut sums = Vec::with_capacity(input.len());
    let mut means = Vec::with_capacity(input.len());
    let mut mean_remainders = Vec::with_capacity(input.len());
    let mut centered_rows = Vec::with_capacity(input.len());
    let mut square_rows = Vec::with_capacity(input.len());
    let mut square_sums = Vec::with_capacity(input.len());
    let mut variances_2q = Vec::with_capacity(input.len());
    let mut width_remainders = Vec::with_capacity(input.len());
    let mut variances_q = Vec::with_capacity(input.len());
    let mut scale_remainders = Vec::with_capacity(input.len());
    let mut lookup_inputs = Vec::with_capacity(input.len());
    let mut inverses = Vec::with_capacity(input.len());
    let mut normalized_products = Vec::with_capacity(input.len());
    let mut normalized_rows = Vec::with_capacity(input.len());
    let mut normalized_remainders = Vec::with_capacity(input.len());
    let mut gamma_products = Vec::with_capacity(input.len());
    let mut scaled_rows = Vec::with_capacity(input.len());
    let mut gamma_remainders = Vec::with_capacity(input.len());

    for (row, expected_row) in input.iter().zip(&expected) {
        let sum = checked_sum(row)?;
        let (mean, mean_remainder) = divide_with_remainder(sum, divisor)?;
        let centered = row
            .iter()
            .map(|value| {
                value
                    .checked_sub(mean)
                    .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let squares = centered
            .iter()
            .map(|value| {
                value
                    .checked_mul(*value)
                    .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let square_sum = checked_sum(&squares)?;
        let (variance_2q, width_remainder) = divide_with_remainder(square_sum, divisor)?;
        let (variance_q, scale_remainder) = divide_with_remainder(variance_2q, q)?;
        let lookup_input = variance_q
            .checked_add(1)
            .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
        let inverse = i128::from(q12_rsqrt(
            i64::try_from(lookup_input).map_err(|_| Gpt2CheckpointError::ArithmeticOverflow)?,
            i64::from(scale),
        )?);
        let normalized_product = checked_hadamard(&centered, inverse)?;
        let (normalized, normalized_remainder) = divide_row(&normalized_product, q)?;
        let gamma_product = normalized
            .iter()
            .zip(&gamma)
            .map(|(left, right)| {
                left.checked_mul(*right)
                    .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let (scaled, gamma_remainder) = divide_row(&gamma_product, q)?;
        let output = scaled
            .iter()
            .zip(&beta)
            .map(|(left, right)| {
                left.checked_add(*right)
                    .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        if output != *expected_row {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        sums.push(sum);
        means.push(mean);
        mean_remainders.push(mean_remainder);
        centered_rows.push(centered);
        square_rows.push(squares);
        square_sums.push(square_sum);
        variances_2q.push(variance_2q);
        width_remainders.push(width_remainder);
        variances_q.push(variance_q);
        scale_remainders.push(scale_remainder);
        lookup_inputs.push(lookup_input);
        inverses.push(inverse);
        normalized_products.push(normalized_product);
        normalized_rows.push(normalized);
        normalized_remainders.push(normalized_remainder);
        gamma_products.push(gamma_product);
        scaled_rows.push(scaled);
        gamma_remainders.push(gamma_remainder);
    }

    builder.push(
        operation_index,
        AuxiliaryBlockKind::RowSum,
        scalar_columns(sums),
    )?;
    builder.push(
        operation_index,
        AuxiliaryBlockKind::Mean,
        scalar_columns(means),
    )?;
    builder.push(
        operation_index,
        AuxiliaryBlockKind::Remainder,
        scalar_columns(mean_remainders),
    )?;
    builder.push(operation_index, AuxiliaryBlockKind::Centered, centered_rows)?;
    builder.push(operation_index, AuxiliaryBlockKind::Square, square_rows)?;
    builder.push(
        operation_index,
        AuxiliaryBlockKind::RowSum,
        scalar_columns(square_sums),
    )?;
    builder.push(
        operation_index,
        AuxiliaryBlockKind::Variance,
        scalar_columns(variances_2q),
    )?;
    builder.push(
        operation_index,
        AuxiliaryBlockKind::Remainder,
        scalar_columns(width_remainders),
    )?;
    builder.push(
        operation_index,
        AuxiliaryBlockKind::Variance,
        scalar_columns(variances_q),
    )?;
    builder.push(
        operation_index,
        AuxiliaryBlockKind::Remainder,
        scalar_columns(scale_remainders),
    )?;
    let lookup_input_columns = scalar_columns(lookup_inputs);
    builder.push(
        operation_index,
        AuxiliaryBlockKind::LookupInput,
        lookup_input_columns.clone(),
    )?;
    let clamped_lookup_input = clamp_witness_columns(&lookup_input_columns, 1, 1 << 25)?;
    push_clamp_witness(builder, operation_index, false, clamped_lookup_input)?;
    builder.push(
        operation_index,
        AuxiliaryBlockKind::LookupOutput,
        scalar_columns(inverses),
    )?;
    builder.push(
        operation_index,
        AuxiliaryBlockKind::Product,
        normalized_products,
    )?;
    builder.push(operation_index, AuxiliaryBlockKind::Scaled, normalized_rows)?;
    builder.push(
        operation_index,
        AuxiliaryBlockKind::Remainder,
        normalized_remainders,
    )?;
    builder.push(operation_index, AuxiliaryBlockKind::Product, gamma_products)?;
    builder.push(operation_index, AuxiliaryBlockKind::Scaled, scaled_rows)?;
    builder.push(
        operation_index,
        AuxiliaryBlockKind::Remainder,
        gamma_remainders,
    )?;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn build_dense_witness<R: Read + Seek>(
    checkpoint: &mut Gpt2Q12Checkpoint<R>,
    layout: &Gpt2GlobalLayout,
    activations: &Gpt2ActivationWitness,
    builder: &mut AuxiliaryWitnessBuilder<'_>,
    operation_index: usize,
    scale: i32,
    has_bias: bool,
    last_token_only: bool,
) -> Result<(), Gpt2CheckpointError> {
    let operation = operation(layout, operation_index)?;
    let mut input = activation_matrix(activations, required(operation.inputs[0])?)?;
    if last_token_only {
        input = vec![input.pop().ok_or(Gpt2CheckpointError::ProtocolIncomplete)?];
    }
    let expected = activation_matrix(activations, operation.output)?;
    let weight = required(operation.parameter_blocks[0])?;
    let output_width = expected.first().map(Vec::len).unwrap_or(0);
    let input_width = input.first().map(Vec::len).unwrap_or(0);
    if input.is_empty()
        || output_width == 0
        || input.iter().any(|row| row.len() != input_width)
        || expected.len() != input.len()
    {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    let mut accumulators = vec![vec![0_i128; output_width]; input.len()];
    let mut weight_column = vec![0_i32; output_width];
    for feature in 0..input_width {
        checkpoint.read_model_block_column(layout, weight, feature, &mut weight_column)?;
        for (row, accumulator) in input.iter().zip(&mut accumulators) {
            for (value, weight_value) in accumulator.iter_mut().zip(&weight_column) {
                *value = value
                    .checked_add(
                        row[feature]
                            .checked_mul(i128::from(*weight_value))
                            .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?,
                    )
                    .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
            }
        }
    }
    builder.push(
        operation_index,
        AuxiliaryBlockKind::Accumulator,
        accumulators.clone(),
    )?;
    let numerators = if has_bias {
        let bias = checkpoint
            .read_tensor_i64(required(operation.parameter_blocks[1])?.0)?
            .into_iter()
            .map(i128::from)
            .collect::<Vec<_>>();
        if bias.len() != output_width {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let numerators = accumulators
            .iter()
            .map(|row| {
                row.iter()
                    .zip(&bias)
                    .map(|(value, bias)| {
                        value
                            .checked_add(
                                bias.checked_mul(i128::from(scale))
                                    .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?,
                            )
                            .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        builder.push(
            operation_index,
            AuxiliaryBlockKind::Numerator,
            numerators.clone(),
        )?;
        numerators
    } else {
        accumulators
    };
    let mut quotients = Vec::with_capacity(numerators.len());
    let mut remainders = Vec::with_capacity(numerators.len());
    for row in &numerators {
        let (quotient, remainder) = divide_row(row, i128::from(scale))?;
        quotients.push(quotient);
        remainders.push(remainder);
    }
    if quotients != expected {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    builder.push(operation_index, AuxiliaryBlockKind::Remainder, remainders)?;
    Ok(())
}

fn build_attention_witness(
    layout: &Gpt2GlobalLayout,
    activations: &Gpt2ActivationWitness,
    builder: &mut AuxiliaryWitnessBuilder<'_>,
    operation_index: usize,
    metadata: &Gpt2Q12Metadata,
) -> Result<(), Gpt2CheckpointError> {
    let operation = operation(layout, operation_index)?;
    let qkv = activation_matrix(activations, required(operation.inputs[0])?)?;
    let expected = activation_matrix(activations, operation.output)?;
    let model = metadata.model;
    let tokens = metadata.token_count;
    let hidden = model.hidden_width;
    let head_width = model.head_width();
    if qkv.len() != tokens
        || qkv.iter().any(|row| row.len() != 3 * hidden)
        || expected.len() != tokens
        || expected.iter().any(|row| row.len() != hidden)
    {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    let q = i128::from(metadata.scale());
    let score_scale =
        i128::from(((f64::from(metadata.scale())) / (head_width as f64).sqrt()).round() as i64);
    let tokens_plus_one = tokens
        .checked_add(1)
        .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
    let packed_capacity = checked_mul(
        model.attention_heads,
        checked_mul(tokens, tokens_plus_one)? / 2,
    )?;
    let row_capacity = checked_mul(model.attention_heads, tokens)?;
    let mut raw_scores = Vec::with_capacity(packed_capacity);
    let mut scaled_scores = Vec::with_capacity(packed_capacity);
    let mut first_remainders = Vec::with_capacity(packed_capacity);
    let mut scores = Vec::with_capacity(packed_capacity);
    let mut second_remainders = Vec::with_capacity(packed_capacity);
    let mut maxima = Vec::with_capacity(row_capacity);
    let mut selectors = Vec::with_capacity(packed_capacity);
    let mut gaps = Vec::with_capacity(packed_capacity);
    let mut prefix_selectors = Vec::with_capacity(packed_capacity);
    let mut strict_gaps = Vec::with_capacity(packed_capacity);
    let mut exponentials = Vec::with_capacity(packed_capacity);
    let mut denominators = Vec::with_capacity(row_capacity);
    let mut reciprocals = Vec::with_capacity(row_capacity);
    let mut probability_products = Vec::with_capacity(packed_capacity);
    let mut base_probabilities = Vec::with_capacity(packed_capacity);
    let mut probability_remainders = Vec::with_capacity(packed_capacity);
    let mut residual_masses = Vec::with_capacity(row_capacity);
    let mut residual_adjustments = Vec::with_capacity(packed_capacity);
    let mut probabilities = Vec::with_capacity(packed_capacity);
    let mut context_accumulators = vec![vec![0_i128; hidden]; tokens];

    for head in 0..model.attention_heads {
        let start = head * head_width;
        for row in 0..tokens {
            let mut row_raw = Vec::with_capacity(row + 1);
            let mut row_scaled = Vec::with_capacity(row + 1);
            let mut row_first_remainder = Vec::with_capacity(row + 1);
            let mut row_scores = Vec::with_capacity(row + 1);
            let mut row_second_remainder = Vec::with_capacity(row + 1);
            for column in 0..=row {
                let mut raw = 0_i128;
                for offset in 0..head_width {
                    raw = raw
                        .checked_add(
                            qkv[row][start + offset]
                                .checked_mul(qkv[column][hidden + start + offset])
                                .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?,
                        )
                        .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
                }
                let scaled_numerator = raw
                    .checked_mul(score_scale)
                    .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
                let (scaled, first_remainder) = divide_with_remainder(scaled_numerator, q)?;
                let (score, second_remainder) = divide_with_remainder(scaled, q)?;
                row_raw.push(raw);
                row_scaled.push(scaled);
                row_first_remainder.push(first_remainder);
                row_scores.push(score);
                row_second_remainder.push(second_remainder);
            }
            let maximum = *row_scores
                .iter()
                .max()
                .ok_or(Gpt2CheckpointError::ProtocolIncomplete)?;
            let maximum_index = row_scores
                .iter()
                .position(|score| *score == maximum)
                .ok_or(Gpt2CheckpointError::ProtocolIncomplete)?;
            let row_selector = (0..row_scores.len())
                .map(|index| i128::from(index == maximum_index))
                .collect::<Vec<_>>();
            let row_gaps = row_scores
                .iter()
                .map(|score| {
                    maximum
                        .checked_sub(*score)
                        .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let row_prefix_selector = (0..row_scores.len())
                .map(|index| i128::from(index < maximum_index))
                .collect::<Vec<_>>();
            let row_strict_gaps = row_gaps
                .iter()
                .zip(&row_prefix_selector)
                .map(|(gap, prefix)| {
                    let value = gap
                        .checked_sub(*prefix)
                        .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
                    if value < 0 {
                        return Err(Gpt2CheckpointError::ProtocolIncomplete);
                    }
                    Ok(value)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let row_exponentials = row_gaps
                .iter()
                .map(|gap| {
                    let input = i64::try_from(-*gap)
                        .map_err(|_| Gpt2CheckpointError::ArithmeticOverflow)?;
                    Ok(i128::from(q12_exp(input, i64::from(metadata.scale()))?))
                })
                .collect::<Result<Vec<_>, Gpt2CheckpointError>>()?;
            let denominator = checked_sum(&row_exponentials)?;
            let reciprocal = i128::from(q12_reciprocal(
                i64::try_from(denominator).map_err(|_| Gpt2CheckpointError::ArithmeticOverflow)?,
                row_scores.len(),
                i64::from(metadata.scale()),
            )?);
            let row_products = checked_hadamard(&row_exponentials, reciprocal)?;
            let (row_base_probabilities, row_probability_remainders) =
                divide_row(&row_products, q)?;
            let residual_mass = q
                .checked_sub(checked_sum(&row_base_probabilities)?)
                .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
            if residual_mass < 0 {
                return Err(Gpt2CheckpointError::ProtocolIncomplete);
            }
            let row_adjustments = row_selector
                .iter()
                .map(|selector| {
                    selector
                        .checked_mul(residual_mass)
                        .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let row_probabilities = row_base_probabilities
                .iter()
                .zip(&row_adjustments)
                .map(|(base, adjustment)| {
                    base.checked_add(*adjustment)
                        .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
                })
                .collect::<Result<Vec<_>, _>>()?;
            if checked_sum(&row_probabilities)? != q {
                return Err(Gpt2CheckpointError::ProtocolIncomplete);
            }
            for offset in 0..head_width {
                let mut accumulator = 0_i128;
                for (column, probability) in row_probabilities.iter().enumerate() {
                    accumulator = accumulator
                        .checked_add(
                            probability
                                .checked_mul(qkv[column][2 * hidden + start + offset])
                                .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?,
                        )
                        .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
                }
                context_accumulators[row][start + offset] = accumulator;
            }
            raw_scores.extend(row_raw);
            scaled_scores.extend(row_scaled);
            first_remainders.extend(row_first_remainder);
            scores.extend(row_scores);
            second_remainders.extend(row_second_remainder);
            maxima.push(maximum);
            selectors.extend(row_selector);
            gaps.extend(row_gaps);
            prefix_selectors.extend(row_prefix_selector);
            strict_gaps.extend(row_strict_gaps);
            exponentials.extend(row_exponentials);
            denominators.push(denominator);
            reciprocals.push(reciprocal);
            probability_products.extend(row_products);
            base_probabilities.extend(row_base_probabilities);
            probability_remainders.extend(row_probability_remainders);
            residual_masses.push(residual_mass);
            residual_adjustments.extend(row_adjustments);
            probabilities.extend(row_probabilities);
        }
    }
    let mut context_outputs = Vec::with_capacity(tokens);
    let mut context_remainders = Vec::with_capacity(tokens);
    for row in &context_accumulators {
        let (output, remainder) = divide_row(row, q)?;
        context_outputs.push(output);
        context_remainders.push(remainder);
    }
    if context_outputs != expected {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }

    for (kind, values) in [
        (AuxiliaryBlockKind::PackedScore, raw_scores),
        (AuxiliaryBlockKind::PackedScore, scaled_scores),
        (AuxiliaryBlockKind::Remainder, first_remainders),
        (AuxiliaryBlockKind::PackedScore, scores),
        (AuxiliaryBlockKind::Remainder, second_remainders),
        (AuxiliaryBlockKind::RowMaximum, maxima),
        (AuxiliaryBlockKind::Selector, selectors),
    ] {
        builder.push(operation_index, kind, vec![values])?;
    }
    let gap_columns = vec![gaps];
    builder.push(
        operation_index,
        AuxiliaryBlockKind::Gap,
        gap_columns.clone(),
    )?;
    let clamped_gap = clamp_witness_columns(&gap_columns, 0, 16 * q)?;
    push_clamp_witness(builder, operation_index, false, clamped_gap)?;
    for (kind, values) in [
        (AuxiliaryBlockKind::PrefixSelector, prefix_selectors),
        (AuxiliaryBlockKind::StrictGap, strict_gaps),
        (AuxiliaryBlockKind::Exponential, exponentials),
        (AuxiliaryBlockKind::Denominator, denominators),
        (AuxiliaryBlockKind::Reciprocal, reciprocals),
        (AuxiliaryBlockKind::Product, probability_products),
        (AuxiliaryBlockKind::BaseProbability, base_probabilities),
        (AuxiliaryBlockKind::Remainder, probability_remainders),
        (AuxiliaryBlockKind::ResidualMass, residual_masses),
        (AuxiliaryBlockKind::ResidualAdjustment, residual_adjustments),
        (AuxiliaryBlockKind::Probability, probabilities),
    ] {
        builder.push(operation_index, kind, vec![values])?;
    }
    builder.push(
        operation_index,
        AuxiliaryBlockKind::Accumulator,
        context_accumulators,
    )?;
    builder.push(
        operation_index,
        AuxiliaryBlockKind::Remainder,
        context_remainders,
    )?;
    Ok(())
}

fn activation_matrix(
    activations: &Gpt2ActivationWitness,
    block: GlobalBlockId,
) -> Result<Vec<Vec<i128>>, Gpt2CheckpointError> {
    let block = activations
        .block(block)
        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
    Ok(block
        .columns
        .iter()
        .map(|column| column.iter().map(|value| i128::from(*value)).collect())
        .collect())
}

fn relation_block_values(
    id: GlobalBlockId,
    layout: &Gpt2GlobalLayout,
    activations: &Gpt2ActivationWitness,
    witness: &Gpt2PrimitiveWitness,
) -> Result<Vec<i128>, Gpt2CheckpointError> {
    if id.0 < layout.witness_block_count() {
        let block = activations
            .block(id)
            .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
        return Ok(block
            .columns
            .iter()
            .flat_map(|column| column.iter().map(|value| i128::from(*value)))
            .collect());
    }
    let index =
        id.0.checked_sub(layout.witness_block_count())
            .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
    let block = witness
        .blocks
        .get(index)
        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
    if block.id != id {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    Ok(block.columns.iter().flatten().copied().collect())
}

fn scalar_columns(values: Vec<i128>) -> Vec<Vec<i128>> {
    values.into_iter().map(|value| vec![value]).collect()
}

fn clamp_witness_columns(
    input: &[Vec<i128>],
    lower: i128,
    upper: i128,
) -> Result<ClampWitnessColumns, Gpt2CheckpointError> {
    if lower >= upper || input.is_empty() || input.iter().any(Vec::is_empty) {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    let mut clamped = Vec::with_capacity(input.len());
    let mut low_selector = Vec::with_capacity(input.len());
    let mut middle_selector = Vec::with_capacity(input.len());
    let mut high_selector = Vec::with_capacity(input.len());
    let mut low_slack = Vec::with_capacity(input.len());
    let mut middle_low_slack = Vec::with_capacity(input.len());
    let mut middle_high_slack = Vec::with_capacity(input.len());
    let mut high_slack = Vec::with_capacity(input.len());
    for column in input {
        let mut clamped_column = Vec::with_capacity(column.len());
        let mut low_selector_column = Vec::with_capacity(column.len());
        let mut middle_selector_column = Vec::with_capacity(column.len());
        let mut high_selector_column = Vec::with_capacity(column.len());
        let mut low_slack_column = Vec::with_capacity(column.len());
        let mut middle_low_slack_column = Vec::with_capacity(column.len());
        let mut middle_high_slack_column = Vec::with_capacity(column.len());
        let mut high_slack_column = Vec::with_capacity(column.len());
        for value in column {
            let is_low = *value < lower;
            let is_high = *value > upper;
            let is_middle = !is_low && !is_high;
            clamped_column.push(if is_low {
                lower
            } else if is_high {
                upper
            } else {
                *value
            });
            low_selector_column.push(i128::from(is_low));
            middle_selector_column.push(i128::from(is_middle));
            high_selector_column.push(i128::from(is_high));
            low_slack_column.push(if is_low {
                lower
                    .checked_sub(*value)
                    .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?
            } else {
                0
            });
            middle_low_slack_column.push(if is_middle {
                value
                    .checked_sub(lower)
                    .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?
            } else {
                0
            });
            middle_high_slack_column.push(if is_middle {
                upper
                    .checked_sub(*value)
                    .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?
            } else {
                0
            });
            high_slack_column.push(if is_high {
                value
                    .checked_sub(upper)
                    .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?
            } else {
                0
            });
        }
        clamped.push(clamped_column);
        low_selector.push(low_selector_column);
        middle_selector.push(middle_selector_column);
        high_selector.push(high_selector_column);
        low_slack.push(low_slack_column);
        middle_low_slack.push(middle_low_slack_column);
        middle_high_slack.push(middle_high_slack_column);
        high_slack.push(high_slack_column);
    }
    Ok(ClampWitnessColumns {
        clamped,
        low_selector,
        middle_selector,
        high_selector,
        low_slack,
        middle_low_slack,
        middle_high_slack,
        high_slack,
    })
}

fn push_clamp_witness(
    builder: &mut AuxiliaryWitnessBuilder<'_>,
    operation_index: usize,
    signed: bool,
    witness: ClampWitnessColumns,
) -> Result<Vec<Vec<i128>>, Gpt2CheckpointError> {
    let clamped = witness.clamped.clone();
    builder.push(
        operation_index,
        if signed {
            AuxiliaryBlockKind::ClampedSigned
        } else {
            AuxiliaryBlockKind::ClampedUnsigned
        },
        witness.clamped,
    )?;
    for selector in [
        witness.low_selector,
        witness.middle_selector,
        witness.high_selector,
    ] {
        builder.push(operation_index, AuxiliaryBlockKind::Selector, selector)?;
    }
    for slack in [
        witness.low_slack,
        witness.middle_low_slack,
        witness.middle_high_slack,
        witness.high_slack,
    ] {
        builder.push(operation_index, AuxiliaryBlockKind::ClampSlack, slack)?;
    }
    Ok(clamped)
}

fn checked_sum(values: &[i128]) -> Result<i128, Gpt2CheckpointError> {
    values.iter().try_fold(0_i128, |sum, value| {
        sum.checked_add(*value)
            .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
    })
}

fn checked_hadamard(values: &[i128], multiplier: i128) -> Result<Vec<i128>, Gpt2CheckpointError> {
    values
        .iter()
        .map(|value| {
            value
                .checked_mul(multiplier)
                .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
        })
        .collect()
}

fn divide_with_remainder(
    numerator: i128,
    denominator: i128,
) -> Result<(i128, i128), Gpt2CheckpointError> {
    if denominator <= 0 {
        return Err(Gpt2CheckpointError::ProtocolIncomplete);
    }
    Ok((
        numerator.div_euclid(denominator),
        numerator.rem_euclid(denominator),
    ))
}

fn divide_row(
    values: &[i128],
    denominator: i128,
) -> Result<(Vec<i128>, Vec<i128>), Gpt2CheckpointError> {
    let mut quotients = Vec::with_capacity(values.len());
    let mut remainders = Vec::with_capacity(values.len());
    for value in values {
        let (quotient, remainder) = divide_with_remainder(*value, denominator)?;
        quotients.push(quotient);
        remainders.push(remainder);
    }
    Ok((quotients, remainders))
}

fn signed_i128_scalar<F: PrimeField>(value: i128) -> F {
    let magnitude = F::from_le_bytes_mod_order(&value.unsigned_abs().to_le_bytes());
    if value < 0 {
        -magnitude
    } else {
        magnitude
    }
}

const fn auxiliary_kind_is_signed(kind: AuxiliaryBlockKind) -> bool {
    matches!(
        kind,
        AuxiliaryBlockKind::Accumulator
            | AuxiliaryBlockKind::Numerator
            | AuxiliaryBlockKind::RowSum
            | AuxiliaryBlockKind::Mean
            | AuxiliaryBlockKind::Centered
            | AuxiliaryBlockKind::Product
            | AuxiliaryBlockKind::Scaled
            | AuxiliaryBlockKind::PackedScore
            | AuxiliaryBlockKind::RowMaximum
            | AuxiliaryBlockKind::ClampedSigned
    )
}

struct RelationCompiler {
    base_block_count: usize,
    auxiliary_blocks: Vec<AuxiliaryBlockDescriptor>,
    relations: Vec<PrimitiveRelationDescriptor>,
}

impl RelationCompiler {
    #[allow(clippy::too_many_arguments)]
    fn compile_clamp(
        &mut self,
        operation_index: usize,
        input: GlobalBlockId,
        rows: usize,
        cols: usize,
        clamped_bits: u16,
        signed: bool,
        slack_bits: u16,
        lower: i64,
        upper: i64,
    ) -> Result<GlobalBlockId, Gpt2CheckpointError> {
        if lower >= upper {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let entries = checked_mul(rows, cols)?;
        let clamped = self.aux(
            operation_index,
            if signed {
                AuxiliaryBlockKind::ClampedSigned
            } else {
                AuxiliaryBlockKind::ClampedUnsigned
            },
            rows,
            cols,
            clamped_bits,
        )?;
        let low_selector =
            self.aux(operation_index, AuxiliaryBlockKind::Selector, rows, cols, 1)?;
        let middle_selector =
            self.aux(operation_index, AuxiliaryBlockKind::Selector, rows, cols, 1)?;
        let high_selector =
            self.aux(operation_index, AuxiliaryBlockKind::Selector, rows, cols, 1)?;
        let mut slacks = Vec::with_capacity(4);
        for _ in 0..4 {
            slacks.push(self.aux(
                operation_index,
                AuxiliaryBlockKind::ClampSlack,
                rows,
                cols,
                slack_bits,
            )?);
        }
        self.push_linear(
            LinearRelationKind::ClampSelectorSum,
            operation_index,
            &[low_selector, middle_selector, high_selector],
            entries,
            1,
        );
        self.push(
            RelationFamily::Bilinear,
            BilinearRelationKind::ClampLower as u8,
            operation_index,
            &[
                input,
                clamped,
                low_selector,
                middle_selector,
                slacks[0],
                slacks[1],
            ],
            entries,
            lower,
        );
        self.push(
            RelationFamily::Bilinear,
            BilinearRelationKind::ClampUpper as u8,
            operation_index,
            &[
                input,
                clamped,
                middle_selector,
                high_selector,
                slacks[2],
                slacks[3],
            ],
            entries,
            upper,
        );
        Ok(clamped)
    }

    fn compile_range_obligations(
        &mut self,
        layout: &Gpt2GlobalLayout,
    ) -> Result<Vec<RangeLimbBlockDescriptor>, Gpt2CheckpointError> {
        let semantic_block_count = self.auxiliary_blocks.len();
        let mut range_blocks = Vec::new();
        let mut sources = Vec::with_capacity(layout.blocks().len() + semantic_block_count);
        for (block_index, source) in layout.blocks().iter().enumerate() {
            let operation_index = match source.role {
                super::Gpt2GlobalBlockRole::Parameter(_) => 0,
                super::Gpt2GlobalBlockRole::PublicTokenIds => continue,
                super::Gpt2GlobalBlockRole::OperationOutput(operation_index) => operation_index,
            };
            sources.push((
                RangeLimbSource::Base { block_index },
                source.id,
                operation_index,
                source.rows,
                source.cols,
                true,
                32_u16,
            ));
        }
        for (block_index, source) in self.auxiliary_blocks.iter().enumerate() {
            sources.push((
                RangeLimbSource::Auxiliary { block_index },
                source.id,
                source.operation_index,
                source.rows,
                source.cols,
                auxiliary_kind_is_signed(source.kind),
                source.value_bits,
            ));
        }
        for (source_kind, source, operation_index, rows, cols, signed, value_bits) in sources {
            let encoding = RangeEncoding::new(signed, value_bits)?;
            let mut recomposition_blocks = Vec::with_capacity(1 + encoding.limb_count());
            recomposition_blocks.push(source);
            for limb in 0..encoding.limb_count() {
                let limb_bits = encoding
                    .limb_bits(limb)
                    .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                let id = GlobalBlockId(
                    self.base_block_count
                        .checked_add(semantic_block_count)
                        .and_then(|count| count.checked_add(range_blocks.len()))
                        .ok_or(Gpt2CheckpointError::OffsetOverflow)?,
                );
                range_blocks.push(RangeLimbBlockDescriptor {
                    id,
                    operation_index,
                    source_kind,
                    source,
                    rows,
                    cols,
                    limb,
                    limb_bits,
                });
                recomposition_blocks.push(id);
                self.relations.push(PrimitiveRelationDescriptor {
                    family: RelationFamily::TaggedLookup,
                    kind: range_lookup_kind(limb_bits) as u8,
                    operation_index,
                    blocks: vec![id],
                    scalar_count: checked_mul(rows, cols)?,
                    immediate: 0,
                });
            }
            self.relations.push(PrimitiveRelationDescriptor {
                family: RelationFamily::Linear,
                kind: LinearRelationKind::RangeRecomposition as u8,
                operation_index,
                blocks: recomposition_blocks,
                scalar_count: checked_mul(rows, cols)?,
                immediate: 0,
            });
        }
        Ok(range_blocks)
    }

    fn compile_embedding(
        &mut self,
        operation_index: usize,
        layout: &Gpt2GlobalLayout,
    ) -> Result<(), Gpt2CheckpointError> {
        let operation = operation(layout, operation_index)?;
        let tokens = required(operation.inputs[0])?;
        let token_embedding = required(operation.parameter_blocks[0])?;
        let position_embedding = required(operation.parameter_blocks[1])?;
        let entries = block_entries(layout, operation.output)?;
        self.push_linear(
            LinearRelationKind::EmbeddingGatherAdd,
            operation_index,
            &[
                tokens,
                token_embedding,
                position_embedding,
                operation.output,
            ],
            entries,
            0,
        );
        Ok(())
    }

    fn compile_layer_norm(
        &mut self,
        operation_index: usize,
        token_count: usize,
        width: usize,
        scale: i32,
        layout: &Gpt2GlobalLayout,
    ) -> Result<(), Gpt2CheckpointError> {
        let operation = operation(layout, operation_index)?;
        let input = required(operation.inputs[0])?;
        let gamma = required(operation.parameter_blocks[0])?;
        let beta = required(operation.parameter_blocks[1])?;
        let value_count = checked_mul(token_count, width)?;
        let sum = self.aux(
            operation_index,
            AuxiliaryBlockKind::RowSum,
            1,
            token_count,
            64,
        )?;
        let mean = self.aux(
            operation_index,
            AuxiliaryBlockKind::Mean,
            1,
            token_count,
            64,
        )?;
        let mean_remainder = self.aux(
            operation_index,
            AuxiliaryBlockKind::Remainder,
            1,
            token_count,
            32,
        )?;
        let centered = self.aux(
            operation_index,
            AuxiliaryBlockKind::Centered,
            width,
            token_count,
            64,
        )?;
        let squares = self.aux(
            operation_index,
            AuxiliaryBlockKind::Square,
            width,
            token_count,
            128,
        )?;
        let square_sum = self.aux(
            operation_index,
            AuxiliaryBlockKind::RowSum,
            1,
            token_count,
            128,
        )?;
        let variance_2q = self.aux(
            operation_index,
            AuxiliaryBlockKind::Variance,
            1,
            token_count,
            128,
        )?;
        let width_remainder = self.aux(
            operation_index,
            AuxiliaryBlockKind::Remainder,
            1,
            token_count,
            32,
        )?;
        let variance_q = self.aux(
            operation_index,
            AuxiliaryBlockKind::Variance,
            1,
            token_count,
            64,
        )?;
        let scale_remainder = self.aux(
            operation_index,
            AuxiliaryBlockKind::Remainder,
            1,
            token_count,
            16,
        )?;
        let lookup_input = self.aux(
            operation_index,
            AuxiliaryBlockKind::LookupInput,
            1,
            token_count,
            64,
        )?;
        let clamped_lookup_input = self.compile_clamp(
            operation_index,
            lookup_input,
            1,
            token_count,
            26,
            false,
            64,
            1,
            1 << 25,
        )?;
        let inverse = self.aux(
            operation_index,
            AuxiliaryBlockKind::LookupOutput,
            1,
            token_count,
            32,
        )?;
        let normalized_product = self.aux(
            operation_index,
            AuxiliaryBlockKind::Product,
            width,
            token_count,
            96,
        )?;
        let normalized = self.aux(
            operation_index,
            AuxiliaryBlockKind::Scaled,
            width,
            token_count,
            64,
        )?;
        let normalized_remainder = self.aux(
            operation_index,
            AuxiliaryBlockKind::Remainder,
            width,
            token_count,
            16,
        )?;
        let gamma_product = self.aux(
            operation_index,
            AuxiliaryBlockKind::Product,
            width,
            token_count,
            96,
        )?;
        let scaled = self.aux(
            operation_index,
            AuxiliaryBlockKind::Scaled,
            width,
            token_count,
            64,
        )?;
        let gamma_remainder = self.aux(
            operation_index,
            AuxiliaryBlockKind::Remainder,
            width,
            token_count,
            16,
        )?;

        self.push_linear(
            LinearRelationKind::ReduceSum,
            operation_index,
            &[input, sum],
            token_count,
            0,
        );
        self.push_linear(
            LinearRelationKind::EuclideanDivision,
            operation_index,
            &[sum, mean, mean_remainder],
            token_count,
            width as i64,
        );
        self.push_linear(
            LinearRelationKind::BroadcastSubtract,
            operation_index,
            &[input, mean, centered],
            value_count,
            0,
        );
        self.push_bilinear(
            BilinearRelationKind::Square,
            operation_index,
            &[centered, squares],
            value_count,
        );
        self.push_linear(
            LinearRelationKind::ReduceSum,
            operation_index,
            &[squares, square_sum],
            token_count,
            0,
        );
        self.push_linear(
            LinearRelationKind::EuclideanDivision,
            operation_index,
            &[square_sum, variance_2q, width_remainder],
            token_count,
            width as i64,
        );
        self.push_linear(
            LinearRelationKind::EuclideanDivision,
            operation_index,
            &[variance_2q, variance_q, scale_remainder],
            token_count,
            i64::from(scale),
        );
        self.push_linear(
            LinearRelationKind::AddConstant,
            operation_index,
            &[variance_q, lookup_input],
            token_count,
            1,
        );
        self.push_lookup(
            TaggedLookupRelationKind::InverseSqrt,
            operation_index,
            &[clamped_lookup_input, inverse],
            token_count,
        );
        self.push_bilinear(
            BilinearRelationKind::Hadamard,
            operation_index,
            &[centered, inverse, normalized_product],
            value_count,
        );
        self.push_linear(
            LinearRelationKind::EuclideanDivision,
            operation_index,
            &[normalized_product, normalized, normalized_remainder],
            value_count,
            i64::from(scale),
        );
        self.push_bilinear(
            BilinearRelationKind::Hadamard,
            operation_index,
            &[normalized, gamma, gamma_product],
            value_count,
        );
        self.push_linear(
            LinearRelationKind::EuclideanDivision,
            operation_index,
            &[gamma_product, scaled, gamma_remainder],
            value_count,
            i64::from(scale),
        );
        self.push_linear(
            LinearRelationKind::AffineAdd,
            operation_index,
            &[scaled, beta, operation.output],
            value_count,
            0,
        );
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_dense_linear(
        &mut self,
        operation_index: usize,
        semantic_rows: usize,
        semantic_cols: usize,
        scale: i32,
        has_bias: bool,
        last_token_only: bool,
        layout: &Gpt2GlobalLayout,
    ) -> Result<(), Gpt2CheckpointError> {
        let operation = operation(layout, operation_index)?;
        let input = required(operation.inputs[0])?;
        let weight = required(operation.parameter_blocks[0])?;
        let output_entries = checked_mul(semantic_rows, semantic_cols)?;
        let accumulator = self.aux(
            operation_index,
            AuxiliaryBlockKind::Accumulator,
            semantic_cols,
            semantic_rows,
            128,
        )?;
        let product_kind = if last_token_only {
            BilinearRelationKind::LastTokenMatrixMultiply
        } else {
            BilinearRelationKind::MatrixMultiply
        };
        self.push_bilinear(
            product_kind,
            operation_index,
            &[input, weight, accumulator],
            output_entries,
        );
        let numerator = if has_bias {
            let bias = required(operation.parameter_blocks[1])?;
            let numerator = self.aux(
                operation_index,
                AuxiliaryBlockKind::Numerator,
                semantic_cols,
                semantic_rows,
                128,
            )?;
            self.push_linear(
                LinearRelationKind::BiasScaleAdd,
                operation_index,
                &[accumulator, bias, numerator],
                output_entries,
                i64::from(scale),
            );
            numerator
        } else {
            accumulator
        };
        let remainder = self.aux(
            operation_index,
            AuxiliaryBlockKind::Remainder,
            semantic_cols,
            semantic_rows,
            16,
        )?;
        self.push_linear(
            LinearRelationKind::EuclideanDivision,
            operation_index,
            &[numerator, operation.output, remainder],
            output_entries,
            i64::from(scale),
        );
        Ok(())
    }

    fn compile_residual(
        &mut self,
        operation_index: usize,
        layout: &Gpt2GlobalLayout,
    ) -> Result<(), Gpt2CheckpointError> {
        let operation = operation(layout, operation_index)?;
        self.push_linear(
            LinearRelationKind::ResidualAdd,
            operation_index,
            &[
                required(operation.inputs[0])?,
                required(operation.inputs[1])?,
                operation.output,
            ],
            block_entries(layout, operation.output)?,
            0,
        );
        Ok(())
    }

    fn compile_gelu(
        &mut self,
        operation_index: usize,
        scale: i32,
        layout: &Gpt2GlobalLayout,
    ) -> Result<(), Gpt2CheckpointError> {
        let operation = operation(layout, operation_index)?;
        let input = required(operation.inputs[0])?;
        let output = layout
            .blocks()
            .get(operation.output.0)
            .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
        let clamped = self.compile_clamp(
            operation_index,
            input,
            output.rows,
            output.cols,
            17,
            true,
            64,
            -8 * i64::from(scale),
            8 * i64::from(scale),
        )?;
        self.push_lookup(
            TaggedLookupRelationKind::Gelu,
            operation_index,
            &[clamped, operation.output],
            block_entries(layout, operation.output)?,
        );
        Ok(())
    }

    fn compile_attention(
        &mut self,
        operation_index: usize,
        metadata: &Gpt2Q12Metadata,
        layout: &Gpt2GlobalLayout,
    ) -> Result<(), Gpt2CheckpointError> {
        let operation = operation(layout, operation_index)?;
        let qkv = required(operation.inputs[0])?;
        let tokens = metadata.token_count;
        let heads = metadata.model.attention_heads;
        let triangular = checked_mul(
            tokens,
            tokens
                .checked_add(1)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?,
        )? / 2;
        let packed = checked_mul(heads, triangular)?;
        let rows = checked_mul(heads, tokens)?;
        let output_descriptor = layout
            .blocks()
            .get(operation.output.0)
            .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
        let output_entries = checked_mul(output_descriptor.rows, output_descriptor.cols)?;
        let score_scale = ((f64::from(metadata.scale()))
            / (metadata.model.head_width() as f64).sqrt())
        .round() as i64;

        let raw_score = self.aux(
            operation_index,
            AuxiliaryBlockKind::PackedScore,
            packed,
            1,
            128,
        )?;
        let scaled_score = self.aux(
            operation_index,
            AuxiliaryBlockKind::PackedScore,
            packed,
            1,
            96,
        )?;
        let first_remainder = self.aux(
            operation_index,
            AuxiliaryBlockKind::Remainder,
            packed,
            1,
            16,
        )?;
        let score = self.aux(
            operation_index,
            AuxiliaryBlockKind::PackedScore,
            packed,
            1,
            64,
        )?;
        let second_remainder = self.aux(
            operation_index,
            AuxiliaryBlockKind::Remainder,
            packed,
            1,
            16,
        )?;
        let maximum = self.aux(operation_index, AuxiliaryBlockKind::RowMaximum, rows, 1, 64)?;
        let selector = self.aux(operation_index, AuxiliaryBlockKind::Selector, packed, 1, 1)?;
        let gap = self.aux(operation_index, AuxiliaryBlockKind::Gap, packed, 1, 64)?;
        let clamped_gap = self.compile_clamp(
            operation_index,
            gap,
            packed,
            1,
            17,
            false,
            64,
            0,
            16 * i64::from(metadata.scale()),
        )?;
        let prefix_selector = self.aux(
            operation_index,
            AuxiliaryBlockKind::PrefixSelector,
            packed,
            1,
            32,
        )?;
        let strict_gap = self.aux(
            operation_index,
            AuxiliaryBlockKind::StrictGap,
            packed,
            1,
            64,
        )?;
        let exponential = self.aux(
            operation_index,
            AuxiliaryBlockKind::Exponential,
            packed,
            1,
            32,
        )?;
        let denominator = self.aux(
            operation_index,
            AuxiliaryBlockKind::Denominator,
            rows,
            1,
            64,
        )?;
        let reciprocal = self.aux(operation_index, AuxiliaryBlockKind::Reciprocal, rows, 1, 32)?;
        let probability_product =
            self.aux(operation_index, AuxiliaryBlockKind::Product, packed, 1, 64)?;
        let base_probability = self.aux(
            operation_index,
            AuxiliaryBlockKind::BaseProbability,
            packed,
            1,
            32,
        )?;
        let probability_remainder = self.aux(
            operation_index,
            AuxiliaryBlockKind::Remainder,
            packed,
            1,
            16,
        )?;
        let residual_mass = self.aux(
            operation_index,
            AuxiliaryBlockKind::ResidualMass,
            rows,
            1,
            16,
        )?;
        let residual_adjustment = self.aux(
            operation_index,
            AuxiliaryBlockKind::ResidualAdjustment,
            packed,
            1,
            32,
        )?;
        let probability = self.aux(
            operation_index,
            AuxiliaryBlockKind::Probability,
            packed,
            1,
            32,
        )?;
        let context_accumulator = self.aux(
            operation_index,
            AuxiliaryBlockKind::Accumulator,
            output_descriptor.rows,
            output_descriptor.cols,
            128,
        )?;
        let context_remainder = self.aux(
            operation_index,
            AuxiliaryBlockKind::Remainder,
            output_descriptor.rows,
            output_descriptor.cols,
            16,
        )?;

        self.push_bilinear(
            BilinearRelationKind::CausalQk,
            operation_index,
            &[qkv, raw_score],
            packed,
        );
        self.push_linear(
            LinearRelationKind::ScaleAndDivide,
            operation_index,
            &[raw_score, scaled_score, first_remainder],
            packed,
            score_scale,
        );
        self.push_linear(
            LinearRelationKind::EuclideanDivision,
            operation_index,
            &[scaled_score, score, second_remainder],
            packed,
            i64::from(metadata.scale()),
        );
        self.push_linear(
            LinearRelationKind::MaximumDifference,
            operation_index,
            &[score, maximum, gap],
            packed,
            0,
        );
        self.push_linear(
            LinearRelationKind::OneHotSum,
            operation_index,
            &[selector],
            rows,
            1,
        );
        self.push_bilinear(
            BilinearRelationKind::Boolean,
            operation_index,
            &[selector],
            packed,
        );
        self.push_bilinear(
            BilinearRelationKind::SelectorZero,
            operation_index,
            &[selector, gap],
            packed,
        );
        self.push_linear(
            LinearRelationKind::PrefixSelector,
            operation_index,
            &[selector, prefix_selector],
            packed,
            0,
        );
        self.push_linear(
            LinearRelationKind::StrictGap,
            operation_index,
            &[gap, prefix_selector, strict_gap],
            packed,
            0,
        );
        self.push_lookup(
            TaggedLookupRelationKind::Exp,
            operation_index,
            &[clamped_gap, exponential],
            packed,
        );
        self.push_linear(
            LinearRelationKind::ReduceSum,
            operation_index,
            &[exponential, denominator],
            rows,
            0,
        );
        self.push_lookup(
            TaggedLookupRelationKind::Reciprocal,
            operation_index,
            &[denominator, reciprocal],
            rows,
        );
        self.push_bilinear(
            BilinearRelationKind::Hadamard,
            operation_index,
            &[exponential, reciprocal, probability_product],
            packed,
        );
        self.push_linear(
            LinearRelationKind::EuclideanDivision,
            operation_index,
            &[probability_product, base_probability, probability_remainder],
            packed,
            i64::from(metadata.scale()),
        );
        self.push_linear(
            LinearRelationKind::ResidualMass,
            operation_index,
            &[base_probability, residual_mass],
            rows,
            i64::from(metadata.scale()),
        );
        self.push_bilinear(
            BilinearRelationKind::SelectorResidual,
            operation_index,
            &[selector, residual_mass, residual_adjustment],
            packed,
        );
        self.push_linear(
            LinearRelationKind::AddResidualAdjustment,
            operation_index,
            &[base_probability, residual_adjustment, probability],
            packed,
            0,
        );
        self.push_linear(
            LinearRelationKind::ExactRowMass,
            operation_index,
            &[probability],
            rows,
            i64::from(metadata.scale()),
        );
        self.push_bilinear(
            BilinearRelationKind::CausalPv,
            operation_index,
            &[probability, qkv, context_accumulator],
            output_entries,
        );
        self.push_linear(
            LinearRelationKind::EuclideanDivision,
            operation_index,
            &[context_accumulator, operation.output, context_remainder],
            output_entries,
            i64::from(metadata.scale()),
        );
        Ok(())
    }

    fn aux(
        &mut self,
        operation_index: usize,
        kind: AuxiliaryBlockKind,
        rows: usize,
        cols: usize,
        value_bits: u16,
    ) -> Result<GlobalBlockId, Gpt2CheckpointError> {
        checked_mul(rows, cols)?;
        if rows == 0 || cols == 0 {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let id = GlobalBlockId(
            self.base_block_count
                .checked_add(self.auxiliary_blocks.len())
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?,
        );
        self.auxiliary_blocks.push(AuxiliaryBlockDescriptor {
            id,
            operation_index,
            kind,
            rows,
            cols,
            value_bits,
        });
        Ok(id)
    }

    fn push_linear(
        &mut self,
        kind: LinearRelationKind,
        operation_index: usize,
        blocks: &[GlobalBlockId],
        scalar_count: usize,
        immediate: i64,
    ) {
        self.push(
            RelationFamily::Linear,
            kind as u8,
            operation_index,
            blocks,
            scalar_count,
            immediate,
        );
    }

    fn push_bilinear(
        &mut self,
        kind: BilinearRelationKind,
        operation_index: usize,
        blocks: &[GlobalBlockId],
        scalar_count: usize,
    ) {
        self.push(
            RelationFamily::Bilinear,
            kind as u8,
            operation_index,
            blocks,
            scalar_count,
            0,
        );
    }

    fn push_lookup(
        &mut self,
        kind: TaggedLookupRelationKind,
        operation_index: usize,
        blocks: &[GlobalBlockId],
        scalar_count: usize,
    ) {
        self.push(
            RelationFamily::TaggedLookup,
            kind as u8,
            operation_index,
            blocks,
            scalar_count,
            0,
        );
    }

    fn push(
        &mut self,
        family: RelationFamily,
        kind: u8,
        operation_index: usize,
        blocks: &[GlobalBlockId],
        scalar_count: usize,
        immediate: i64,
    ) {
        self.relations.push(PrimitiveRelationDescriptor {
            family,
            kind,
            operation_index,
            blocks: blocks.to_vec(),
            scalar_count,
            immediate,
        });
    }
}

fn operation(
    layout: &Gpt2GlobalLayout,
    index: usize,
) -> Result<&super::Gpt2GlobalOperation, Gpt2CheckpointError> {
    layout
        .operations()
        .get(index)
        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)
}

fn required(block: Option<GlobalBlockId>) -> Result<GlobalBlockId, Gpt2CheckpointError> {
    block.ok_or(Gpt2CheckpointError::ProtocolIncomplete)
}

fn block_entries(
    layout: &Gpt2GlobalLayout,
    block: GlobalBlockId,
) -> Result<usize, Gpt2CheckpointError> {
    let descriptor = layout
        .blocks()
        .get(block.0)
        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
    checked_mul(descriptor.rows, descriptor.cols)
}

fn checked_mul(left: usize, right: usize) -> Result<usize, Gpt2CheckpointError> {
    left.checked_mul(right)
        .ok_or(Gpt2CheckpointError::OffsetOverflow)
}

fn assign_tbv_bucket<F: PrimeField>(
    etas: &mut [F],
    challenges: &mut [Vec<F>],
    assigned: &mut [bool],
    bucket: usize,
    eta: F,
    row_challenges: &[F],
) -> Result<(), Gpt2CheckpointError> {
    let target_eta = etas
        .get_mut(bucket)
        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
    let target_challenges = challenges
        .get_mut(bucket)
        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
    let target_assigned = assigned
        .get_mut(bucket)
        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
    if *target_assigned || target_challenges.len() != row_challenges.len() {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    *target_eta = eta;
    target_challenges.copy_from_slice(row_challenges);
    *target_assigned = true;
    Ok(())
}

fn tensor_prefix_sum<F: PrimeField>(
    physical_len: usize,
    low_bit_challenges: &[F],
) -> Result<F, Gpt2CheckpointError> {
    if low_bit_challenges.len() >= usize::BITS as usize {
        return Err(Gpt2CheckpointError::OffsetOverflow);
    }
    let logical_len = 1_usize << low_bit_challenges.len();
    if physical_len > logical_len {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    if physical_len == 0 {
        return Ok(F::from(0_u64));
    }
    if low_bit_challenges.is_empty() {
        return Ok(F::from(1_u64));
    }
    let high_bit = *low_bit_challenges
        .last()
        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
    let lower_challenges = &low_bit_challenges[..low_bit_challenges.len() - 1];
    let half = logical_len / 2;
    if physical_len <= half {
        tensor_prefix_sum(physical_len, lower_challenges)
    } else {
        let lower_total = lower_challenges
            .iter()
            .fold(F::from(1_u64), |product, challenge| {
                product * (F::from(1_u64) + challenge)
            });
        Ok(lower_total + high_bit * tensor_prefix_sum(physical_len - half, lower_challenges)?)
    }
}

fn decode_linear_relation_kind(kind: u8) -> Result<LinearRelationKind, Gpt2CheckpointError> {
    match kind {
        0 => Ok(LinearRelationKind::EmbeddingGatherAdd),
        1 => Ok(LinearRelationKind::ReduceSum),
        2 => Ok(LinearRelationKind::EuclideanDivision),
        3 => Ok(LinearRelationKind::BroadcastSubtract),
        4 => Ok(LinearRelationKind::AddConstant),
        5 => Ok(LinearRelationKind::BiasScaleAdd),
        6 => Ok(LinearRelationKind::ResidualAdd),
        7 => Ok(LinearRelationKind::MaximumDifference),
        8 => Ok(LinearRelationKind::OneHotSum),
        9 => Ok(LinearRelationKind::PrefixSelector),
        10 => Ok(LinearRelationKind::StrictGap),
        11 => Ok(LinearRelationKind::ResidualMass),
        12 => Ok(LinearRelationKind::AddResidualAdjustment),
        13 => Ok(LinearRelationKind::ExactRowMass),
        14 => Ok(LinearRelationKind::AffineAdd),
        15 => Ok(LinearRelationKind::ScaleAndDivide),
        16 => Ok(LinearRelationKind::RangeRecomposition),
        17 => Ok(LinearRelationKind::ClampSelectorSum),
        _ => Err(Gpt2CheckpointError::ProtocolShapeMismatch),
    }
}

fn decode_bilinear_relation_kind(kind: u8) -> Result<BilinearRelationKind, Gpt2CheckpointError> {
    match kind {
        0 => Ok(BilinearRelationKind::MatrixMultiply),
        1 => Ok(BilinearRelationKind::LastTokenMatrixMultiply),
        2 => Ok(BilinearRelationKind::Square),
        3 => Ok(BilinearRelationKind::Hadamard),
        4 => Ok(BilinearRelationKind::Boolean),
        5 => Ok(BilinearRelationKind::SelectorZero),
        6 => Ok(BilinearRelationKind::CausalQk),
        7 => Ok(BilinearRelationKind::CausalPv),
        8 => Ok(BilinearRelationKind::SelectorResidual),
        9 => Ok(BilinearRelationKind::ClampLower),
        10 => Ok(BilinearRelationKind::ClampUpper),
        _ => Err(Gpt2CheckpointError::ProtocolShapeMismatch),
    }
}

fn decode_lookup_relation_kind(kind: u8) -> Result<TaggedLookupRelationKind, Gpt2CheckpointError> {
    match kind {
        0 => Ok(TaggedLookupRelationKind::Gelu),
        1 => Ok(TaggedLookupRelationKind::Exp),
        2 => Ok(TaggedLookupRelationKind::Reciprocal),
        3 => Ok(TaggedLookupRelationKind::InverseSqrt),
        4 => Ok(TaggedLookupRelationKind::Range1),
        5 => Ok(TaggedLookupRelationKind::Range2),
        6 => Ok(TaggedLookupRelationKind::Range3),
        7 => Ok(TaggedLookupRelationKind::Range4),
        8 => Ok(TaggedLookupRelationKind::Range5),
        9 => Ok(TaggedLookupRelationKind::Range6),
        10 => Ok(TaggedLookupRelationKind::Range7),
        11 => Ok(TaggedLookupRelationKind::Range8),
        _ => Err(Gpt2CheckpointError::ProtocolShapeMismatch),
    }
}

fn common_block_shape(
    manifest: &Gpt2PrimitiveRelationManifest,
    base: &Gpt2GlobalLayout,
    blocks: &[GlobalBlockId],
) -> Result<(usize, usize), Gpt2CheckpointError> {
    let first = *blocks
        .first()
        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
    let shape = manifest.block_shape(base, first)?;
    if blocks
        .iter()
        .skip(1)
        .any(|block| manifest.block_shape(base, *block).ok() != Some(shape))
    {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    Ok(shape)
}

fn bva_vector(
    block_vectors: &[Vec<usize>],
    block: GlobalBlockId,
    column: usize,
) -> Result<usize, Gpt2CheckpointError> {
    block_vectors
        .get(block.0)
        .and_then(|columns| columns.get(column))
        .copied()
        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)
}

fn bva_bucket_log(bva: &Gpt2PrimitiveBvaLayout, vector: usize) -> usize {
    let bucket = bva.layout.vector_to_bucket()[vector];
    bva.layout.bucket_sizes()[bucket].ilog2() as usize
}

fn random_bucket_challenges<F: PrimeField>(
    bva: &Gpt2PrimitiveBvaLayout,
    vector: usize,
    next_challenge: &mut impl FnMut() -> F,
) -> Vec<F> {
    (0..bva_bucket_log(bva, vector))
        .map(|_| next_challenge())
        .collect()
}

fn random_log_challenges<F: PrimeField>(
    physical_len: usize,
    next_challenge: &mut impl FnMut() -> F,
) -> Result<Vec<F>, Gpt2CheckpointError> {
    let logical_len = physical_len
        .checked_next_power_of_two()
        .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
    Ok((0..logical_len.ilog2()).map(|_| next_challenge()).collect())
}

fn push_tensor_term<F: PrimeField>(
    terms: &mut [Vec<(F, Vec<(F, F)>)>],
    bva: &Gpt2PrimitiveBvaLayout,
    vector: usize,
    eta: F,
    challenges: &[F],
) -> Result<(), Gpt2CheckpointError> {
    let bucket = *bva
        .layout
        .vector_to_bucket()
        .get(vector)
        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
    if challenges.len() != bva.layout.bucket_sizes()[bucket].ilog2() as usize {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    terms
        .get_mut(bucket)
        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?
        .push((
            eta,
            challenges
                .iter()
                .map(|challenge| (F::from(1_u64), *challenge))
                .collect(),
        ));
    Ok(())
}

/// Copies one contiguous tensor-coefficient slice into the sequential Lasso
/// query layout. Intersections are decomposed into aligned dyadic subcubes, so
/// the metadata cost is logarithmic in each boundary crossing instead of one
/// term per query. Table-tail vectors are never visited.
fn push_lookup_query_tensor_slice<F: PrimeField>(
    terms: &mut [Vec<(F, Vec<(F, F)>)>],
    query_layout: &BvaLayout,
    query_cursor: &mut (usize, usize),
    source_len: usize,
    eta: F,
    source_challenges: &[F],
) -> Result<(), Gpt2CheckpointError> {
    let source_capacity = 1_usize
        .checked_shl(source_challenges.len() as u32)
        .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
    if source_len == 0 || source_len > source_capacity {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    let mut source_start = 0_usize;
    while source_start < source_len {
        let vector_index = query_cursor.0;
        let destination_start = query_cursor.1;
        let destination_len = *query_layout
            .vector_lengths()
            .get(vector_index)
            .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
        if destination_start >= destination_len {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let take = (source_len - source_start).min(destination_len - destination_start);
        push_aligned_tensor_interval(
            terms,
            query_layout,
            vector_index,
            destination_start,
            source_start,
            take,
            eta,
            source_challenges,
        )?;
        source_start += take;
        query_cursor.1 += take;
        if query_cursor.1 == destination_len {
            query_cursor.0 += 1;
            query_cursor.1 = 0;
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn push_aligned_tensor_interval<F: PrimeField>(
    terms: &mut [Vec<(F, Vec<(F, F)>)>],
    destination_layout: &BvaLayout,
    destination_vector: usize,
    mut destination_start: usize,
    mut source_start: usize,
    mut remaining: usize,
    eta: F,
    source_challenges: &[F],
) -> Result<(), Gpt2CheckpointError> {
    let bucket = *destination_layout
        .vector_to_bucket()
        .get(destination_vector)
        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
    let destination_log = destination_layout.bucket_sizes()[bucket].ilog2() as usize;
    while remaining != 0 {
        let mut block_log = (remaining.ilog2() as usize)
            .min(source_challenges.len())
            .min(destination_log);
        while block_log != 0
            && ((source_start & ((1_usize << block_log) - 1)) != 0
                || (destination_start & ((1_usize << block_log) - 1)) != 0)
        {
            block_log -= 1;
        }
        let block_len = 1_usize << block_log;
        let mut term_eta = eta;
        for (bit, challenge) in source_challenges.iter().enumerate().skip(block_log) {
            if (source_start >> bit) & 1 == 1 {
                term_eta *= challenge;
            }
        }
        let factors = (0..destination_log)
            .map(|bit| {
                if bit < block_log {
                    (F::one(), source_challenges[bit])
                } else if (destination_start >> bit) & 1 == 0 {
                    (F::one(), F::zero())
                } else {
                    (F::zero(), F::one())
                }
            })
            .collect();
        terms[bucket].push((term_eta, factors));
        source_start += block_len;
        destination_start += block_len;
        remaining -= block_len;
    }
    Ok(())
}

fn push_lagrange_point_term<F: PrimeField>(
    terms: &mut [Vec<(F, Vec<(F, F)>)>],
    bva: &Gpt2PrimitiveBvaLayout,
    vector: usize,
    index: usize,
    eta: F,
) -> Result<(), Gpt2CheckpointError> {
    let physical_len = *bva
        .layout
        .vector_lengths()
        .get(vector)
        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
    if index >= physical_len {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    let bucket = bva.layout.vector_to_bucket()[vector];
    let factors = (0..bva.layout.bucket_sizes()[bucket].ilog2() as usize)
        .map(|bit| {
            if index & (1 << bit) == 0 {
                (F::from(1_u64), F::from(0_u64))
            } else {
                (F::from(0_u64), F::from(1_u64))
            }
        })
        .collect();
    terms[bucket].push((eta, factors));
    Ok(())
}

fn push_dense_terms<F: PrimeField>(
    terms: &mut [Vec<(F, Vec<(F, F)>)>],
    bva: &Gpt2PrimitiveBvaLayout,
    vector: usize,
    coefficients: &[F],
) -> Result<(), Gpt2CheckpointError> {
    let physical_len = *bva
        .layout
        .vector_lengths()
        .get(vector)
        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
    let bucket = bva.layout.vector_to_bucket()[vector];
    let logical_len = bva.layout.bucket_sizes()[bucket];
    if coefficients.len() != physical_len {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    let mut mobius = vec![F::from(0_u64); logical_len];
    mobius[..physical_len].copy_from_slice(coefficients);
    for bit in 0..logical_len.ilog2() as usize {
        for mask in 0..logical_len {
            if mask & (1 << bit) == 0 {
                let upper = mobius[mask | (1 << bit)];
                mobius[mask] -= upper;
            }
        }
    }
    for (mask, eta) in mobius.into_iter().enumerate() {
        if eta == F::from(0_u64) {
            continue;
        }
        let factors = (0..logical_len.ilog2() as usize)
            .map(|bit| (F::from(1_u64), F::from(u64::from(mask & (1 << bit) != 0))))
            .collect();
        terms[bucket].push((eta, factors));
    }
    Ok(())
}

fn tensor_physical_coefficients<F: PrimeField>(
    physical_len: usize,
    challenges: &[F],
) -> Result<Vec<F>, Gpt2CheckpointError> {
    if challenges.len() >= usize::BITS as usize || physical_len > (1_usize << challenges.len()) {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    Ok((0..physical_len)
        .map(|index| {
            challenges
                .iter()
                .enumerate()
                .filter(|(bit, _)| index & (1 << bit) != 0)
                .fold(F::from(1_u64), |coefficient, (_, challenge)| {
                    coefficient * challenge
                })
        })
        .collect())
}

fn scale_coefficients<F: PrimeField>(coefficients: &[F], scale: F) -> Vec<F> {
    coefficients
        .iter()
        .map(|coefficient| *coefficient * scale)
        .collect()
}

fn causal_row_segments(metadata: &Gpt2Q12Metadata) -> Result<Vec<usize>, Gpt2CheckpointError> {
    if metadata.token_count == 0 || metadata.model.attention_heads == 0 {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    let mut segments = Vec::with_capacity(
        metadata
            .model
            .attention_heads
            .checked_mul(metadata.token_count)
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?,
    );
    for _ in 0..metadata.model.attention_heads {
        segments.extend(1..=metadata.token_count);
    }
    Ok(segments)
}

fn require_segmented_shapes(
    input: (usize, usize),
    output: (usize, usize),
    scalar_count: usize,
    segments: &[usize],
) -> Result<(), Gpt2CheckpointError> {
    if input != (segments.iter().sum(), 1)
        || output != (segments.len(), 1)
        || scalar_count != segments.len()
    {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    Ok(())
}

fn expand_segment_coefficients<F: PrimeField>(
    row_coefficients: &[F],
    segments: &[usize],
) -> Result<Vec<F>, Gpt2CheckpointError> {
    if row_coefficients.len() != segments.len() {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    let mut result = Vec::with_capacity(segments.iter().sum());
    for (coefficient, length) in row_coefficients.iter().zip(segments) {
        result.extend(std::iter::repeat_n(*coefficient, *length));
    }
    Ok(result)
}

fn sum_segment_coefficients<F: PrimeField>(
    coefficients: &[F],
    segments: &[usize],
) -> Result<Vec<F>, Gpt2CheckpointError> {
    if coefficients.len() != segments.iter().sum() {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    let mut offset = 0_usize;
    let mut result = Vec::with_capacity(segments.len());
    for length in segments {
        result.push(
            coefficients[offset..offset + length]
                .iter()
                .fold(F::from(0_u64), |sum, coefficient| sum + coefficient),
        );
        offset += length;
    }
    Ok(result)
}

fn suffix_segment_coefficients<F: PrimeField>(
    coefficients: &[F],
    segments: &[usize],
) -> Result<Vec<F>, Gpt2CheckpointError> {
    if coefficients.len() != segments.iter().sum() {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    let mut result = Vec::with_capacity(coefficients.len());
    let mut offset = 0_usize;
    for length in segments {
        let segment = &coefficients[offset..offset + length];
        let mut suffix = vec![F::from(0_u64); *length];
        let mut running = F::from(0_u64);
        for index in (0..*length).rev() {
            running += segment[index];
            suffix[index] = running;
        }
        result.extend(suffix);
        offset += length;
    }
    Ok(result)
}

const fn family_tag(family: RelationFamily) -> u8 {
    match family {
        RelationFamily::Linear => 0,
        RelationFamily::Bilinear => 1,
        RelationFamily::TaggedLookup => 2,
    }
}

fn range_lookup_kind(bits: u8) -> TaggedLookupRelationKind {
    match bits {
        1 => TaggedLookupRelationKind::Range1,
        2 => TaggedLookupRelationKind::Range2,
        3 => TaggedLookupRelationKind::Range3,
        4 => TaggedLookupRelationKind::Range4,
        5 => TaggedLookupRelationKind::Range5,
        6 => TaggedLookupRelationKind::Range6,
        7 => TaggedLookupRelationKind::Range7,
        8 => TaggedLookupRelationKind::Range8,
        _ => unreachable!("range limbs contain between one and eight bits"),
    }
}

const fn range_tag_bits(tag: TaggedLookupRelationKind) -> u8 {
    match tag {
        TaggedLookupRelationKind::Range1 => 1,
        TaggedLookupRelationKind::Range2 => 2,
        TaggedLookupRelationKind::Range3 => 3,
        TaggedLookupRelationKind::Range4 => 4,
        TaggedLookupRelationKind::Range5 => 5,
        TaggedLookupRelationKind::Range6 => 6,
        TaggedLookupRelationKind::Range7 => 7,
        TaggedLookupRelationKind::Range8 => 8,
        _ => 0,
    }
}

fn semantic_lookup_kind(kind: u8) -> Result<TaggedLookupRelationKind, Gpt2CheckpointError> {
    match kind {
        value if value == TaggedLookupRelationKind::Gelu as u8 => {
            Ok(TaggedLookupRelationKind::Gelu)
        }
        value if value == TaggedLookupRelationKind::Exp as u8 => Ok(TaggedLookupRelationKind::Exp),
        value if value == TaggedLookupRelationKind::Reciprocal as u8 => {
            Ok(TaggedLookupRelationKind::Reciprocal)
        }
        value if value == TaggedLookupRelationKind::InverseSqrt as u8 => {
            Ok(TaggedLookupRelationKind::InverseSqrt)
        }
        _ => Err(Gpt2CheckpointError::ProtocolShapeMismatch),
    }
}

fn compress_lookup_query<F: PrimeField>(query: TaggedLookupQuery, challenge: F) -> F {
    F::from(query.tag as u64)
        + challenge * signed_i128_scalar::<F>(query.input)
        + challenge * challenge * signed_i128_scalar::<F>(query.output)
}

fn hash_usize(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

/// Public verifier parameters for the first production extraction of the
/// global GPT-2 HD-R1CS relation.  The parameters are independent of the
/// checkpoint payload; only the public statement determines the required
/// maximum polynomial dimension.
#[derive(Clone, Debug)]
pub struct Gpt2GlobalRelationParameters {
    pub pc_pp: PcsParams<Bls12_381>,
}

impl Gpt2GlobalRelationParameters {
    pub fn setup(statement: &Gpt2PublicStatement) -> Result<Self, Gpt2CheckpointError> {
        let (metadata, layout, manifest, bva) = global_relation_public_shape(statement)?;
        let _ = (&layout, &manifest);
        let bva_log = bva.logical_entries().ilog2() as usize;
        let lookup_log = Gpt2LookupRegistry::exact(&metadata)
            .public_table_rows()?
            .next_power_of_two()
            .ilog2() as usize;
        let mut rng = ark_std::rand::thread_rng();
        Ok(Self {
            pc_pp: SmartPC::<Bls12_381>::setup((bva_log.max(lookup_log) + 1) / 2, &mut rng)
                .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?,
        })
    }

    pub fn to_bytes_exact(&self) -> Result<Vec<u8>, SerializationError> {
        let mut payload = Vec::new();
        self.pc_pp.serialize_compressed(&mut payload)?;
        let mut bytes = Vec::with_capacity(20 + payload.len());
        bytes.extend_from_slice(&GPT2_PARAMETERS_MAGIC);
        bytes.extend_from_slice(&GPT2_PARAMETERS_VERSION.to_le_bytes());
        bytes.extend_from_slice(&(payload.len() as u64).to_le_bytes());
        bytes.extend_from_slice(&payload);
        Ok(bytes)
    }

    pub fn from_bytes_exact(bytes: &[u8]) -> Result<Self, Gpt2ProofDecodeError> {
        if bytes.len() > GPT2_PARAMETERS_MAX_BYTES {
            return Err(Gpt2ProofDecodeError::Oversized(bytes.len()));
        }
        if bytes.len() < 20 {
            return Err(Gpt2ProofDecodeError::Truncated);
        }
        if bytes[..8] != GPT2_PARAMETERS_MAGIC {
            return Err(Gpt2ProofDecodeError::InvalidMagic);
        }
        let version = u32::from_le_bytes(
            bytes[8..12]
                .try_into()
                .map_err(|_| Gpt2ProofDecodeError::Truncated)?,
        );
        if version != GPT2_PARAMETERS_VERSION {
            return Err(Gpt2ProofDecodeError::UnsupportedVersion(version));
        }
        let payload_len = usize::try_from(u64::from_le_bytes(
            bytes[12..20]
                .try_into()
                .map_err(|_| Gpt2ProofDecodeError::Truncated)?,
        ))
        .map_err(|_| Gpt2ProofDecodeError::Oversized(usize::MAX))?;
        if payload_len > GPT2_PARAMETERS_MAX_BYTES {
            return Err(Gpt2ProofDecodeError::Oversized(payload_len));
        }
        let expected_len = 20_usize
            .checked_add(payload_len)
            .ok_or(Gpt2ProofDecodeError::Oversized(payload_len))?;
        if bytes.len() < expected_len {
            return Err(Gpt2ProofDecodeError::Truncated);
        }
        if bytes.len() > expected_len {
            return Err(Gpt2ProofDecodeError::TrailingBytes);
        }
        let payload = &bytes[20..expected_len];
        let mut reader = Cursor::new(payload);
        let pc_pp = PcsParams::<Bls12_381>::deserialize_compressed(&mut reader)?;
        if reader.position() as usize != payload.len() {
            return Err(Gpt2ProofDecodeError::TrailingBytes);
        }
        Ok(Self { pc_pp })
    }

    fn srs(&self) -> SRS<Bls12_381> {
        SRS {
            zksrs: ZkSRS::new(self.pc_pp.g_0, self.pc_pp.tilde_g),
            pc_pp: self.pc_pp.clone(),
            h_hat: self.pc_pp.h_hat,
        }
    }
}

/// The single-commitment proof for all global I/O, linear, range, lookup-leaf
/// and bilinear GPT-2 equations.  The four bilinear reductions and all public
/// linear reductions are synchronized into one final EO reduction and one
/// SMART-PC opening of the same `global_witness_commitment`.
///
/// Lookup membership is intentionally a separate subprotocol.  The complete
/// aggregate proof must additionally bind the Lasso factor proof to
/// `lookup_claim`; this type only replaces the formerly test-only global
/// relation composition.
#[derive(Clone, Debug)]
pub struct Gpt2GlobalRelationProof {
    pub statement_digest: [u8; 32],
    pub global_witness_commitment: PairingOutput<Bls12_381>,
    pub lookup_left_product: Fr,
    pub lookup_right_product: Fr,
    pub lookup_claim: Fr,
    pub elementwise_claim: Fr,
    pub dense_matmul_claim: Fr,
    pub causal_qk_claim: Fr,
    pub causal_pv_claim: Fr,
    pub elementwise_proof: EoWitnessPairProof<Fr>,
    pub dense_matmul_proof: EoWitnessPairProof<Fr>,
    pub causal_qk_proof: EoWitnessPairProof<Fr>,
    pub causal_pv_proof: EoWitnessPairProof<Fr>,
    pub synchronized_eo: EoLiteBulletProof<Fr>,
    pub synchronized_value_commitment: PairingOutput<Bls12_381>,
    pub synchronized_opening: PcsTrans<Bls12_381>,
}

/// Production Lasso core for GPT-2's exact tagged lookup registry.  It proves
/// the challenged multiset product, factor affine equations, public sparse
/// table support/count binding, timestamp range, and both counter routes.
/// The aggregate `Gpt2Proof` additionally has to bind
/// `query_value_commitment` to the lookup projection opened from `C_w`.
pub struct Gpt2LassoProof {
    pub query_count: usize,
    pub public_table_rows: usize,
    pub factor_chunk_len: usize,
    pub timestamp_range_bits: usize,
    pub multiplicity_commitment: PairingOutput<Bls12_381>,
    pub timestamp_commitment: PairingOutput<Bls12_381>,
    pub factor_value_commitment: PairingOutput<Bls12_381>,
    pub query_value_commitment: PairingOutput<Bls12_381>,
    pub table_value_commitment: PairingOutput<Bls12_381>,
    pub left_counter_commitment: PairingOutput<Bls12_381>,
    pub left_query_counter_commitment: PairingOutput<Bls12_381>,
    pub left_table_counter_commitment: PairingOutput<Bls12_381>,
    pub right_counter_commitment: PairingOutput<Bls12_381>,
    pub left_factor_commitment: PairingOutput<Bls12_381>,
    pub right_factor_commitment: PairingOutput<Bls12_381>,
    pub expected_product: Fr,
    public_table_binding: Gpt2LassoPublicTableBinding<Fr>,
    pub timestamp_range_proof: BvaRangeProof<Bls12_381>,
    pub left_grand_product: BvaGrandProductProof<Bls12_381>,
    pub right_grand_product: BvaGrandProductProof<Bls12_381>,
    pub left_affine: BvaAffineRelationProof<Bls12_381>,
    pub right_affine: BvaAffineRelationProof<Bls12_381>,
    pub left_counter_timestamp: BvaRoutedEqualityProof<Bls12_381>,
    pub right_counter_timestamp: BvaRoutedEqualityProof<Bls12_381>,
}

/// Public-support end-to-end GPT-2 proof.  It contains one global HD-R1CS
/// relation proof, one Lasso lookup proof, and the committed-evaluation proof
/// that prevents the two subprotocols from using different query values.
pub struct Gpt2Proof {
    pub global_relation: Gpt2GlobalRelationProof,
    pub lasso: Gpt2LassoProof,
    pub query_value_binding: BvaCommittedEvaluationProof<Bls12_381>,
}

/// Scalable public-support lookup argument used by the real GPT-2 path.  The
/// sorted multiplicities are public; membership is reduced to one challenged
/// multiset product over `z - compress(tag,input,output)` and one evaluation
/// that binds those factors back to the same global `C_w`.
pub struct Gpt2PublicSupportLookupProof {
    pub query_count: usize,
    pub public_table_rows: usize,
    pub factor_chunk_len: usize,
    pub multiplicities: Vec<(usize, u64)>,
    pub factor_commitment: PairingOutput<Bls12_381>,
    pub expected_product: Fr,
    pub grand_product: BvaGrandProductProof<Bls12_381>,
    pub factor_global_binding: BvaCommittedEvaluationProof<Bls12_381>,
}

/// End-to-end real-model proof with one global HD-R1CS commitment.  This mode
/// keeps the lookup support public and is therefore intentionally non-ZK, but
/// its verifier consumes only statement, parameters, and proof.
pub struct Gpt2ScalableProof {
    pub global_relation: Gpt2GlobalRelationProof,
    pub lookup: Gpt2PublicSupportLookupProof,
}

struct Gpt2PublicSupportLookupCore {
    query_count: usize,
    public_table_rows: usize,
    factor_chunk_len: usize,
    multiplicities: Vec<(usize, u64)>,
    factor_commitment: PairingOutput<Bls12_381>,
    expected_product: Fr,
    grand_product: BvaGrandProductProof<Bls12_381>,
}

struct Gpt2PublicSupportLookupCache {
    factors: BucketedVector<Fr>,
    factor_tier_one: Vec<<Bls12_381 as Pairing>::G1>,
    factor_hiding: Fr,
    row_challenge: Fr,
    product_challenge: Fr,
}

#[derive(Clone, Copy)]
enum Gpt2LookupArgument<'a> {
    Legacy(&'a Gpt2LassoProof),
    Public(&'a Gpt2PublicSupportLookupCore),
}

impl Gpt2Proof {
    /// Proves from three independently seekable views of the same checkpoint.
    /// This avoids materializing the timestamp/query vectors merely to satisfy
    /// two-source PCS openings.  File-backed callers should open the bundle
    /// three times; no checkpoint payload is copied into RAM.
    pub fn prove_with_replays<GlobalR, LassoR, ReplayR>(
        parameters: &Gpt2GlobalRelationParameters,
        statement: &Gpt2PublicStatement,
        global_checkpoint: &mut Gpt2Q12Checkpoint<GlobalR>,
        lasso_checkpoint: &mut Gpt2Q12Checkpoint<LassoR>,
        route_replay: &mut Gpt2Q12Checkpoint<ReplayR>,
        factor_chunk_len: usize,
    ) -> Result<Self, Gpt2CheckpointError>
    where
        GlobalR: Read + Seek,
        LassoR: Read + Seek,
        ReplayR: Read + Seek,
    {
        let commitment_cache =
            commit_global_witness_cache(parameters, statement, global_checkpoint)?;
        let (lasso, lasso_cache) = Gpt2LassoProof::prove_core_with_cache(
            parameters,
            statement,
            commitment_cache.commitment,
            lasso_checkpoint,
            route_replay,
            factor_chunk_len,
        )?;
        let (global_relation, reduction) = Gpt2GlobalRelationProof::prove_internal(
            parameters,
            statement,
            global_checkpoint,
            Some(commitment_cache),
            Some(Gpt2LookupArgument::Legacy(&lasso)),
        )?;
        let query_tbv = reduction
            .query_tbv
            .as_ref()
            .ok_or(Gpt2CheckpointError::ProtocolIncomplete)?;
        if query_tbv.layout() != &lasso_cache.factor_layout {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        let query_value_binding = prove_lasso_query_value_binding(
            parameters,
            statement,
            lasso_checkpoint,
            &lasso,
            &lasso_cache,
            query_tbv,
            reduction.lookup_compressed_projection,
        )?;
        let proof = Self {
            global_relation,
            lasso,
            query_value_binding,
        };
        if !proof.verify(parameters, statement)? {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        Ok(proof)
    }

    pub fn witness_commitment_count(&self) -> usize {
        1
    }

    pub fn witness_opening_count(&self) -> usize {
        1
    }

    pub fn to_bytes_exact(&self) -> Result<Vec<u8>, SerializationError> {
        let global = self.global_relation.to_bytes_exact()?;
        let mut lasso = Vec::new();
        serialize_lasso_proof(&self.lasso, &mut lasso)?;
        let mut query_binding = Vec::new();
        serialize_committed_evaluation_proof(&self.query_value_binding, &mut query_binding)?;

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GPT2_PROOF_MAGIC);
        bytes.extend_from_slice(&GPT2_PROOF_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.global_relation.statement_digest);
        for section in [&global, &lasso, &query_binding] {
            bytes.extend_from_slice(&(section.len() as u64).to_le_bytes());
            bytes.extend_from_slice(section);
        }
        Ok(bytes)
    }

    pub fn serialized_size_bytes(&self) -> Result<usize, SerializationError> {
        Ok(self.to_bytes_exact()?.len())
    }

    pub fn from_bytes_exact(bytes: &[u8]) -> Result<Self, Gpt2ProofDecodeError> {
        if bytes.len() > GPT2_PROOF_MAX_BYTES {
            return Err(Gpt2ProofDecodeError::Oversized(bytes.len()));
        }
        if bytes.len() < 12 + 32 + 3 * 8 {
            return Err(Gpt2ProofDecodeError::Truncated);
        }
        if bytes[..8] != GPT2_PROOF_MAGIC {
            return Err(Gpt2ProofDecodeError::InvalidMagic);
        }
        let version = u32::from_le_bytes(
            bytes[8..12]
                .try_into()
                .map_err(|_| Gpt2ProofDecodeError::Truncated)?,
        );
        if version != GPT2_PROOF_VERSION {
            return Err(Gpt2ProofDecodeError::UnsupportedVersion(version));
        }
        let mut position = 12_usize;
        let mut statement_digest = [0_u8; 32];
        statement_digest.copy_from_slice(&bytes[position..position + 32]);
        position += 32;
        let mut sections = Vec::with_capacity(3);
        for _ in 0..3 {
            if position.checked_add(8).is_none_or(|end| end > bytes.len()) {
                return Err(Gpt2ProofDecodeError::Truncated);
            }
            let section_len = u64::from_le_bytes(
                bytes[position..position + 8]
                    .try_into()
                    .map_err(|_| Gpt2ProofDecodeError::Truncated)?,
            );
            position += 8;
            let section_len = usize::try_from(section_len)
                .map_err(|_| Gpt2ProofDecodeError::Oversized(usize::MAX))?;
            let end = position
                .checked_add(section_len)
                .ok_or(Gpt2ProofDecodeError::Oversized(section_len))?;
            if end > bytes.len() {
                return Err(Gpt2ProofDecodeError::Truncated);
            }
            sections.push(&bytes[position..end]);
            position = end;
        }
        if position != bytes.len() {
            return Err(Gpt2ProofDecodeError::TrailingBytes);
        }
        let global_relation = Gpt2GlobalRelationProof::from_bytes_exact(sections[0])?;
        if global_relation.statement_digest != statement_digest {
            return Err(Gpt2ProofDecodeError::InvalidMagic);
        }
        let lasso = deserialize_lasso_proof_exact(sections[1])?;
        let query_value_binding = deserialize_committed_evaluation_proof_exact(sections[2])?;
        Ok(Self {
            global_relation,
            lasso,
            query_value_binding,
        })
    }

    pub fn verify_bytes(
        parameters: &Gpt2GlobalRelationParameters,
        statement: &Gpt2PublicStatement,
        bytes: &[u8],
    ) -> Result<bool, Gpt2CheckpointError> {
        let proof = match Self::from_bytes_exact(bytes) {
            Ok(proof) => proof,
            Err(_) => return Ok(false),
        };
        proof.verify(parameters, statement)
    }

    /// Independent statement-only verification.  It never accepts a
    /// checkpoint handle, activation witness, lookup stream, or prover cache.
    pub fn verify(
        &self,
        parameters: &Gpt2GlobalRelationParameters,
        statement: &Gpt2PublicStatement,
    ) -> Result<bool, Gpt2CheckpointError> {
        statement.validate()?;
        if self.global_relation.statement_digest != statement.transcript_digest()
            || !self.lasso.verify_core(
                parameters,
                statement,
                self.global_relation.global_witness_commitment,
            )?
        {
            return Ok(false);
        }
        let (metadata, layout, manifest, bva) = global_relation_public_shape(statement)?;
        let messages = GlobalRelationMessages {
            global_witness_commitment: self.global_relation.global_witness_commitment,
            lookup_left_product: self.global_relation.lookup_left_product,
            lookup_right_product: self.global_relation.lookup_right_product,
            lookup_claim: self.global_relation.lookup_claim,
            elementwise_claim: self.global_relation.elementwise_claim,
            dense_matmul_claim: self.global_relation.dense_matmul_claim,
            causal_qk_claim: self.global_relation.causal_qk_claim,
            causal_pv_claim: self.global_relation.causal_pv_claim,
            elementwise_proof: &self.global_relation.elementwise_proof,
            dense_matmul_proof: &self.global_relation.dense_matmul_proof,
            causal_qk_proof: &self.global_relation.causal_qk_proof,
            causal_pv_proof: &self.global_relation.causal_pv_proof,
        };
        let Some(reduction) = derive_global_relation_reduction(
            statement,
            &metadata,
            &layout,
            &manifest,
            &bva,
            &messages,
            Some(Gpt2LookupArgument::Legacy(&self.lasso)),
        )?
        else {
            return Ok(false);
        };
        let synchronized_verification = self
            .global_relation
            .synchronized_eo
            .verify_with_tbv(&reduction.synchronized_tbv, reduction.synchronized_claim);
        if !synchronized_verification.accepted {
            return Ok(false);
        }
        let expected_value_commitment = Bls12_381::pairing(
            parameters
                .pc_pp
                .g_0
                .mul(self.global_relation.synchronized_eo.a_reduced),
            parameters.pc_pp.h_hat,
        );
        if self.global_relation.synchronized_value_commitment != expected_value_commitment {
            return Ok(false);
        }
        let (xl, xr) = bva_eo_opening_point(
            bva.logical_entries(),
            &synchronized_verification.challenges_inv,
        )
        .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
        if !SmartPC::<Bls12_381>::verify(
            &parameters.pc_pp,
            self.global_relation.global_witness_commitment,
            self.global_relation.synchronized_value_commitment,
            &xl,
            &xr,
            &self.global_relation.synchronized_opening,
        )
        .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?
        {
            return Ok(false);
        }
        let query_tbv = reduction
            .query_tbv
            .as_ref()
            .ok_or(Gpt2CheckpointError::ProtocolIncomplete)?;
        let factor_layout = lasso_factor_layout(
            self.lasso.query_count,
            self.lasso.public_table_binding.multiplicities.len(),
            self.lasso.factor_chunk_len,
        )?;
        Ok(self.query_value_binding.verify(
            &parameters.srs(),
            0,
            0x4c41_5651,
            statement.transcript_digest(),
            &factor_layout,
            self.lasso.query_value_commitment,
            query_tbv,
            reduction.lookup_compressed_projection,
        ))
    }
}

impl Gpt2PublicSupportLookupCore {
    fn prove<R: Read + Seek>(
        parameters: &Gpt2GlobalRelationParameters,
        statement: &Gpt2PublicStatement,
        global_witness_commitment: PairingOutput<Bls12_381>,
        checkpoint: &mut Gpt2Q12Checkpoint<R>,
        factor_chunk_len: usize,
    ) -> Result<(Self, Gpt2PublicSupportLookupCache), Gpt2CheckpointError> {
        if !factor_chunk_len.is_power_of_two() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let expected = Gpt2PublicStatement::new(
            checkpoint.metadata(),
            checkpoint.token_ids(),
            checkpoint.reference_logits(),
        )?;
        if &expected != statement {
            return Err(Gpt2CheckpointError::PublicStatementDigestMismatch);
        }
        let metadata = checkpoint.metadata().clone();
        let layout = Gpt2GlobalLayout::compile(&metadata)?;
        let (_, activations) = checkpoint.execute_activation_witness(&layout)?;
        let manifest = Gpt2PrimitiveRelationManifest::compile(&metadata, &layout)?;
        let witness = Gpt2PrimitiveWitness::build(checkpoint, &layout, &activations, &manifest)?;
        let registry = Gpt2LookupRegistry::exact(&metadata);

        gpt2_progress("lookup.support_count.start");
        let mut counts = HashMap::<usize, u64>::new();
        let profile = {
            let mut stream =
                Gpt2LookupQueryStream::new(checkpoint, &layout, &activations, &witness, &manifest)?;
            stream.for_each(&mut |query| {
                registry.audit_query(query)?;
                let table_index = registry.table_index(query)?;
                let count = counts.entry(table_index).or_insert(0);
                *count = count
                    .checked_add(1)
                    .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
                Ok(())
            })?
        };
        let query_count = profile
            .semantic_queries
            .checked_add(profile.range_queries)
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        let mut multiplicities = counts.into_iter().collect::<Vec<_>>();
        multiplicities.sort_by_key(|(table_index, _)| *table_index);
        validate_public_support(&registry, query_count, &multiplicities)?;
        gpt2_progress("lookup.support_count.done");
        let (row_challenge, product_challenge) =
            public_support_lookup_challenges(statement, global_witness_commitment, &multiplicities);
        if row_challenge.is_zero() || product_challenge.is_zero() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }

        let factor_layout = lasso_timestamp_layout(query_count, factor_chunk_len)?;
        let vector_lengths = factor_layout.vector_lengths().to_vec();
        let mut factor_vectors = vector_lengths
            .iter()
            .map(|length| Vec::with_capacity(*length))
            .collect::<Vec<_>>();
        let mut vector_index = 0_usize;
        gpt2_progress("lookup.factor_materialize.start");
        {
            let mut stream =
                Gpt2LookupQueryStream::new(checkpoint, &layout, &activations, &witness, &manifest)?;
            stream.for_each(&mut |query| {
                registry.audit_query(query)?;
                let vector = factor_vectors
                    .get_mut(vector_index)
                    .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                vector.push(product_challenge - compress_lookup_query(query, row_challenge));
                if vector.len() == vector_lengths[vector_index] {
                    vector_index += 1;
                }
                Ok(())
            })?;
        }
        if vector_index != vector_lengths.len()
            || factor_vectors
                .iter()
                .zip(&vector_lengths)
                .any(|(vector, length)| vector.len() != *length)
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let mut factors = BucketedVector::new(factor_vectors)?;
        if factors.layout() != &factor_layout {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        gpt2_progress("lookup.factor_materialize.done");
        let expected_product = public_support_expected_product(
            &registry,
            &multiplicities,
            row_challenge,
            product_challenge,
        )?;
        let mut rng = ark_std::rand::thread_rng();
        let factor_hiding = Fr::rand(&mut rng);
        gpt2_progress("lookup.factor_commit.start");
        let (factor_commitment, factor_tier_one, commit_profile) =
            commit_bva_field_materialized_parallel::<Bls12_381>(
                &parameters.pc_pp,
                &factors,
                factor_hiding,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
        if commit_profile.materialized_padding_entries != 0 {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        gpt2_progress("lookup.factor_commit.done");
        gpt2_progress("lookup.grand_product.start");
        let (grand_product, grand_product_profile) = BvaGrandProductProof::prove_materialized(
            &parameters.srs(),
            0,
            0x4750_5053,
            &mut factors,
            factor_commitment,
            &factor_tier_one,
            factor_hiding,
            expected_product,
        )
        .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
        if grand_product_profile.materialized_padding_entries != 0
            || grand_product_profile.padding_field_operations != 0
            || grand_product_profile.padding_group_operations != 0
        {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        gpt2_progress("lookup.grand_product.done");
        Ok((
            Self {
                query_count,
                public_table_rows: registry.public_table_rows()?,
                factor_chunk_len,
                multiplicities,
                factor_commitment,
                expected_product,
                grand_product,
            },
            Gpt2PublicSupportLookupCache {
                factors,
                factor_tier_one,
                factor_hiding,
                row_challenge,
                product_challenge,
            },
        ))
    }

    fn verify_core(
        &self,
        parameters: &Gpt2GlobalRelationParameters,
        statement: &Gpt2PublicStatement,
        global_witness_commitment: PairingOutput<Bls12_381>,
    ) -> Result<bool, Gpt2CheckpointError> {
        let (metadata, _, _, _) = global_relation_public_shape(statement)?;
        let registry = Gpt2LookupRegistry::exact(&metadata);
        if self.public_table_rows != registry.public_table_rows()?
            || !self.factor_chunk_len.is_power_of_two()
            || validate_public_support(&registry, self.query_count, &self.multiplicities).is_err()
        {
            return Ok(false);
        }
        let (row_challenge, product_challenge) = public_support_lookup_challenges(
            statement,
            global_witness_commitment,
            &self.multiplicities,
        );
        let expected_product = public_support_expected_product(
            &registry,
            &self.multiplicities,
            row_challenge,
            product_challenge,
        )?;
        if expected_product != self.expected_product {
            return Ok(false);
        }
        let factor_layout = lasso_timestamp_layout(self.query_count, self.factor_chunk_len)?;
        Ok(self.grand_product.verify(
            &parameters.srs(),
            0,
            0x4750_5053,
            &factor_layout,
            self.factor_commitment,
            self.expected_product,
        ))
    }
}

impl Gpt2ScalableProof {
    pub fn prove_with_replays<GlobalR: Read + Seek, LookupR: Read + Seek>(
        parameters: &Gpt2GlobalRelationParameters,
        statement: &Gpt2PublicStatement,
        global_checkpoint: &mut Gpt2Q12Checkpoint<GlobalR>,
        lookup_checkpoint: &mut Gpt2Q12Checkpoint<LookupR>,
        factor_chunk_len: usize,
    ) -> Result<Self, Gpt2CheckpointError> {
        gpt2_progress("global_witness.commit.start");
        let commitment_cache =
            commit_global_witness_cache(parameters, statement, global_checkpoint)?;
        let global_witness_commitment = commitment_cache.commitment;
        gpt2_progress("global_witness.commit.done");
        let (lookup_core, mut lookup_cache) = Gpt2PublicSupportLookupCore::prove(
            parameters,
            statement,
            global_witness_commitment,
            lookup_checkpoint,
            factor_chunk_len,
        )?;
        gpt2_progress("global_relation.prove.start");
        let (global_relation, reduction) = Gpt2GlobalRelationProof::prove_internal(
            parameters,
            statement,
            global_checkpoint,
            Some(commitment_cache),
            Some(Gpt2LookupArgument::Public(&lookup_core)),
        )?;
        gpt2_progress("global_relation.prove.done");
        if lookup_cache.row_challenge
            != public_support_lookup_challenges(
                statement,
                global_witness_commitment,
                &lookup_core.multiplicities,
            )
            .0
        {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        let query_tbv = reduction
            .query_tbv
            .ok_or(Gpt2CheckpointError::ProtocolIncomplete)?;
        if query_tbv.layout() != lookup_cache.factors.layout() {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        let physical_mask_evaluation = query_tbv
            .layout()
            .vector_lengths()
            .iter()
            .enumerate()
            .try_fold(Fr::from(0_u64), |sum, (vector_index, _)| {
                query_tbv
                    .physical_coefficient_sum(vector_index)
                    .ok()
                    .map(|value| sum + value)
            })
            .ok_or(Gpt2CheckpointError::ProtocolIncomplete)?;
        let factor_claim = lookup_cache.product_challenge * physical_mask_evaluation
            - reduction.lookup_compressed_projection;
        gpt2_progress("lookup.global_binding.start");
        let (factor_global_binding, binding_profile) =
            BvaCommittedEvaluationProof::prove_materialized_consuming_tbv(
                &parameters.srs(),
                0,
                0x5053_4642,
                statement.transcript_digest(),
                &mut lookup_cache.factors,
                lookup_core.factor_commitment,
                &lookup_cache.factor_tier_one,
                lookup_cache.factor_hiding,
                query_tbv,
                factor_claim,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
        if binding_profile.materialized_padding_entries != 0
            || binding_profile.padding_field_operations != 0
        {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        gpt2_progress("lookup.global_binding.done");
        let proof = Self {
            global_relation,
            lookup: Gpt2PublicSupportLookupProof {
                query_count: lookup_core.query_count,
                public_table_rows: lookup_core.public_table_rows,
                factor_chunk_len: lookup_core.factor_chunk_len,
                multiplicities: lookup_core.multiplicities,
                factor_commitment: lookup_core.factor_commitment,
                expected_product: lookup_core.expected_product,
                grand_product: lookup_core.grand_product,
                factor_global_binding,
            },
        };
        if !proof.verify(parameters, statement)? {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        Ok(proof)
    }

    pub fn witness_commitment_count(&self) -> usize {
        1
    }

    pub fn witness_opening_count(&self) -> usize {
        1
    }

    pub fn to_bytes_exact(&self) -> Result<Vec<u8>, SerializationError> {
        let global = self.global_relation.to_bytes_exact()?;
        let mut grand_product = Vec::new();
        serialize_grand_product_proof(&self.lookup.grand_product, &mut grand_product)?;
        let mut factor_binding = Vec::new();
        serialize_committed_evaluation_proof(
            &self.lookup.factor_global_binding,
            &mut factor_binding,
        )?;
        let mut lookup = Vec::new();
        for value in [
            self.lookup.query_count,
            self.lookup.public_table_rows,
            self.lookup.factor_chunk_len,
            self.lookup.multiplicities.len(),
        ] {
            (value as u64).serialize_compressed(&mut lookup)?;
        }
        for &(table_index, count) in &self.lookup.multiplicities {
            (table_index as u64).serialize_compressed(&mut lookup)?;
            count.serialize_compressed(&mut lookup)?;
        }
        self.lookup
            .factor_commitment
            .serialize_compressed(&mut lookup)?;
        self.lookup
            .expected_product
            .serialize_compressed(&mut lookup)?;
        for section in [&grand_product, &factor_binding] {
            (section.len() as u64).serialize_compressed(&mut lookup)?;
            lookup.extend_from_slice(section);
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GPT2_SCALABLE_PROOF_MAGIC);
        bytes.extend_from_slice(&GPT2_SCALABLE_PROOF_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.global_relation.statement_digest);
        for section in [&global, &lookup] {
            bytes.extend_from_slice(&(section.len() as u64).to_le_bytes());
            bytes.extend_from_slice(section);
        }
        Ok(bytes)
    }

    pub fn serialized_size_bytes(&self) -> Result<usize, SerializationError> {
        Ok(self.to_bytes_exact()?.len())
    }

    pub fn from_bytes_exact(bytes: &[u8]) -> Result<Self, Gpt2ProofDecodeError> {
        if bytes.len() > GPT2_PROOF_MAX_BYTES {
            return Err(Gpt2ProofDecodeError::Oversized(bytes.len()));
        }
        if bytes.len() < 12 + 32 + 16 {
            return Err(Gpt2ProofDecodeError::Truncated);
        }
        if bytes[..8] != GPT2_SCALABLE_PROOF_MAGIC {
            return Err(Gpt2ProofDecodeError::InvalidMagic);
        }
        let version = u32::from_le_bytes(
            bytes[8..12]
                .try_into()
                .map_err(|_| Gpt2ProofDecodeError::Truncated)?,
        );
        if version != GPT2_SCALABLE_PROOF_VERSION {
            return Err(Gpt2ProofDecodeError::UnsupportedVersion(version));
        }
        let mut position = 12_usize;
        let mut statement_digest = [0_u8; 32];
        statement_digest.copy_from_slice(&bytes[position..position + 32]);
        position += 32;
        let mut sections = Vec::with_capacity(2);
        for _ in 0..2 {
            if position.checked_add(8).is_none_or(|end| end > bytes.len()) {
                return Err(Gpt2ProofDecodeError::Truncated);
            }
            let section_len = u64::from_le_bytes(
                bytes[position..position + 8]
                    .try_into()
                    .map_err(|_| Gpt2ProofDecodeError::Truncated)?,
            );
            position += 8;
            let section_len = usize::try_from(section_len)
                .map_err(|_| Gpt2ProofDecodeError::Oversized(usize::MAX))?;
            let end = position
                .checked_add(section_len)
                .ok_or(Gpt2ProofDecodeError::Oversized(section_len))?;
            if end > bytes.len() {
                return Err(Gpt2ProofDecodeError::Truncated);
            }
            sections.push(&bytes[position..end]);
            position = end;
        }
        if position != bytes.len() {
            return Err(Gpt2ProofDecodeError::TrailingBytes);
        }
        let global_relation = Gpt2GlobalRelationProof::from_bytes_exact(sections[0])?;
        if global_relation.statement_digest != statement_digest {
            return Err(Gpt2ProofDecodeError::InvalidMagic);
        }
        let mut reader = Cursor::new(sections[1]);
        let query_count = deserialize_usize(&mut reader)?;
        let public_table_rows = deserialize_usize(&mut reader)?;
        let factor_chunk_len = deserialize_usize(&mut reader)?;
        let support_len = read_bounded_len(&mut reader, GPT2_SCALABLE_SUPPORT_MAX_ENTRIES)?;
        let mut multiplicities = Vec::with_capacity(support_len);
        for _ in 0..support_len {
            multiplicities.push((
                deserialize_usize(&mut reader)?,
                u64::deserialize_compressed(&mut reader)?,
            ));
        }
        let factor_commitment = PairingOutput::deserialize_compressed(&mut reader)?;
        let expected_product = Fr::deserialize_compressed(&mut reader)?;
        let grand_product_len = read_bounded_len(&mut reader, GPT2_PROOF_MAX_BYTES)?;
        let mut grand_product_bytes = vec![0_u8; grand_product_len];
        reader
            .read_exact(&mut grand_product_bytes)
            .map_err(|_| Gpt2ProofDecodeError::Truncated)?;
        let factor_binding_len = read_bounded_len(&mut reader, GPT2_PROOF_MAX_BYTES)?;
        let mut factor_binding_bytes = vec![0_u8; factor_binding_len];
        reader
            .read_exact(&mut factor_binding_bytes)
            .map_err(|_| Gpt2ProofDecodeError::Truncated)?;
        if reader.position() != sections[1].len() as u64 {
            return Err(Gpt2ProofDecodeError::TrailingBytes);
        }
        Ok(Self {
            global_relation,
            lookup: Gpt2PublicSupportLookupProof {
                query_count,
                public_table_rows,
                factor_chunk_len,
                multiplicities,
                factor_commitment,
                expected_product,
                grand_product: deserialize_grand_product_proof_exact(&grand_product_bytes)?,
                factor_global_binding: deserialize_committed_evaluation_proof_exact(
                    &factor_binding_bytes,
                )?,
            },
        })
    }

    pub fn verify_bytes(
        parameters: &Gpt2GlobalRelationParameters,
        statement: &Gpt2PublicStatement,
        bytes: &[u8],
    ) -> Result<bool, Gpt2CheckpointError> {
        let proof = match Self::from_bytes_exact(bytes) {
            Ok(proof) => proof,
            Err(_) => return Ok(false),
        };
        proof.verify(parameters, statement)
    }

    pub fn verify(
        &self,
        parameters: &Gpt2GlobalRelationParameters,
        statement: &Gpt2PublicStatement,
    ) -> Result<bool, Gpt2CheckpointError> {
        statement.validate()?;
        let core = Gpt2PublicSupportLookupCore {
            query_count: self.lookup.query_count,
            public_table_rows: self.lookup.public_table_rows,
            factor_chunk_len: self.lookup.factor_chunk_len,
            multiplicities: self.lookup.multiplicities.clone(),
            factor_commitment: self.lookup.factor_commitment,
            expected_product: self.lookup.expected_product,
            grand_product: self.lookup.grand_product.clone(),
        };
        if self.global_relation.statement_digest != statement.transcript_digest()
            || !core.verify_core(
                parameters,
                statement,
                self.global_relation.global_witness_commitment,
            )?
        {
            return Ok(false);
        }
        verify_scalable_global_and_binding(self, &core, parameters, statement)
    }
}

fn verify_scalable_global_and_binding(
    proof: &Gpt2ScalableProof,
    core: &Gpt2PublicSupportLookupCore,
    parameters: &Gpt2GlobalRelationParameters,
    statement: &Gpt2PublicStatement,
) -> Result<bool, Gpt2CheckpointError> {
    let (metadata, layout, manifest, bva) = global_relation_public_shape(statement)?;
    let messages = GlobalRelationMessages {
        global_witness_commitment: proof.global_relation.global_witness_commitment,
        lookup_left_product: proof.global_relation.lookup_left_product,
        lookup_right_product: proof.global_relation.lookup_right_product,
        lookup_claim: proof.global_relation.lookup_claim,
        elementwise_claim: proof.global_relation.elementwise_claim,
        dense_matmul_claim: proof.global_relation.dense_matmul_claim,
        causal_qk_claim: proof.global_relation.causal_qk_claim,
        causal_pv_claim: proof.global_relation.causal_pv_claim,
        elementwise_proof: &proof.global_relation.elementwise_proof,
        dense_matmul_proof: &proof.global_relation.dense_matmul_proof,
        causal_qk_proof: &proof.global_relation.causal_qk_proof,
        causal_pv_proof: &proof.global_relation.causal_pv_proof,
    };
    let Some(reduction) = derive_global_relation_reduction(
        statement,
        &metadata,
        &layout,
        &manifest,
        &bva,
        &messages,
        Some(Gpt2LookupArgument::Public(core)),
    )?
    else {
        return Ok(false);
    };
    let GlobalRelationReduction {
        synchronized_tbv,
        synchronized_claim,
        query_tbv,
        lookup_compressed_projection,
    } = reduction;
    let synchronized_verification = proof
        .global_relation
        .synchronized_eo
        .verify_with_tbv_consuming(synchronized_tbv, synchronized_claim);
    if !synchronized_verification.accepted {
        return Ok(false);
    }
    let expected_value_commitment = Bls12_381::pairing(
        parameters
            .pc_pp
            .g_0
            .mul(proof.global_relation.synchronized_eo.a_reduced),
        parameters.pc_pp.h_hat,
    );
    if proof.global_relation.synchronized_value_commitment != expected_value_commitment {
        return Ok(false);
    }
    let (xl, xr) = bva_eo_opening_point(
        bva.logical_entries(),
        &synchronized_verification.challenges_inv,
    )
    .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
    if !SmartPC::<Bls12_381>::verify(
        &parameters.pc_pp,
        proof.global_relation.global_witness_commitment,
        proof.global_relation.synchronized_value_commitment,
        &xl,
        &xr,
        &proof.global_relation.synchronized_opening,
    )
    .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?
    {
        return Ok(false);
    }

    // The synchronized reduction can dominate verifier RSS at production
    // scale. It is no longer needed once its EO and Cw opening have both
    // verified, so release it before retaining/using the lookup query TBV.
    drop(metadata);
    drop(layout);
    drop(manifest);
    drop(bva);

    let query_tbv = query_tbv.ok_or(Gpt2CheckpointError::ProtocolIncomplete)?;
    let factor_layout = lasso_timestamp_layout(core.query_count, core.factor_chunk_len)?;
    if query_tbv.layout() != &factor_layout {
        return Ok(false);
    }
    let physical_mask_evaluation = query_tbv
        .layout()
        .vector_lengths()
        .iter()
        .enumerate()
        .try_fold(Fr::from(0_u64), |sum, (vector_index, _)| {
            query_tbv
                .physical_coefficient_sum(vector_index)
                .ok()
                .map(|value| sum + value)
        });
    let Some(physical_mask_evaluation) = physical_mask_evaluation else {
        return Ok(false);
    };
    let (_, product_challenge) = public_support_lookup_challenges(
        statement,
        proof.global_relation.global_witness_commitment,
        &core.multiplicities,
    );
    let factor_claim = product_challenge * physical_mask_evaluation - lookup_compressed_projection;
    Ok(proof.lookup.factor_global_binding.verify_consuming_tbv(
        &parameters.srs(),
        0,
        0x5053_4642,
        statement.transcript_digest(),
        &factor_layout,
        core.factor_commitment,
        query_tbv,
        factor_claim,
    ))
}

struct Gpt2LassoProverCache {
    products: StreamingLassoProducts<Fr>,
    factor_layout: BvaLayout,
    row_challenge: Fr,
    product_challenge: Fr,
    query_value_tier_one: Vec<<Bls12_381 as Pairing>::G1>,
    query_value_hiding: Fr,
}

impl Gpt2LassoProof {
    pub fn prove_core<R: Read + Seek, ReplayR: Read + Seek>(
        parameters: &Gpt2GlobalRelationParameters,
        statement: &Gpt2PublicStatement,
        global_witness_commitment: PairingOutput<Bls12_381>,
        checkpoint: &mut Gpt2Q12Checkpoint<R>,
        replay_checkpoint: &mut Gpt2Q12Checkpoint<ReplayR>,
        factor_chunk_len: usize,
    ) -> Result<Self, Gpt2CheckpointError> {
        Self::prove_core_with_cache(
            parameters,
            statement,
            global_witness_commitment,
            checkpoint,
            replay_checkpoint,
            factor_chunk_len,
        )
        .map(|(proof, _)| proof)
    }

    fn prove_core_with_cache<R: Read + Seek, ReplayR: Read + Seek>(
        parameters: &Gpt2GlobalRelationParameters,
        statement: &Gpt2PublicStatement,
        global_witness_commitment: PairingOutput<Bls12_381>,
        checkpoint: &mut Gpt2Q12Checkpoint<R>,
        replay_checkpoint: &mut Gpt2Q12Checkpoint<ReplayR>,
        factor_chunk_len: usize,
    ) -> Result<(Self, Gpt2LassoProverCache), Gpt2CheckpointError> {
        if !factor_chunk_len.is_power_of_two() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let expected_statement = Gpt2PublicStatement::new(
            checkpoint.metadata(),
            checkpoint.token_ids(),
            checkpoint.reference_logits(),
        )?;
        if &expected_statement != statement {
            return Err(Gpt2CheckpointError::PublicStatementDigestMismatch);
        }
        let replay_statement = Gpt2PublicStatement::new(
            replay_checkpoint.metadata(),
            replay_checkpoint.token_ids(),
            replay_checkpoint.reference_logits(),
        )?;
        if &replay_statement != statement {
            return Err(Gpt2CheckpointError::PublicStatementDigestMismatch);
        }
        let metadata = checkpoint.metadata().clone();
        let layout = Gpt2GlobalLayout::compile(&metadata)?;
        let (_, activations) = checkpoint.execute_activation_witness(&layout)?;
        let manifest = Gpt2PrimitiveRelationManifest::compile(&metadata, &layout)?;
        let witness = Gpt2PrimitiveWitness::build(checkpoint, &layout, &activations, &manifest)?;
        let registry = Gpt2LookupRegistry::exact(&metadata);
        let mut transcript = Gpt2ProofTranscript::new(statement);
        transcript.absorb(b"global-witness-commitment", &global_witness_commitment);
        let row_challenge = transcript.challenge::<Fr>(b"lookup-row-compression");

        let mut inventory_stream =
            Gpt2LookupQueryStream::new(checkpoint, &layout, &activations, &witness, &manifest)?;
        let mut products =
            inventory_stream.lasso_products(&registry, row_challenge, Fr::from(1_u64))?;
        drop(inventory_stream);
        let srs = parameters.srs();
        let mut rng = ark_std::rand::thread_rng();

        let multiplicity_hiding = Fr::rand(&mut rng);
        let mut multiplicity_source = Gpt2LassoMultiplicitySource::<Fr>::new(&products)?;
        let (multiplicity_commitment, _, multiplicity_profile) =
            commit_sparse_bva_field_streaming::<Bls12_381, _>(
                &parameters.pc_pp,
                &mut multiplicity_source,
                multiplicity_hiding,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
        if multiplicity_profile.materialized_padding_entries != 0 {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        transcript.absorb(b"lookup-multiplicity-commitment", &multiplicity_commitment);

        let timestamp_hiding = Fr::rand(&mut rng);
        let mut timestamp_stream =
            Gpt2LookupQueryStream::new(checkpoint, &layout, &activations, &witness, &manifest)?;
        let mut timestamp_source = Gpt2LassoTimestampSource::<_, Fr>::new(
            &mut timestamp_stream,
            &registry,
            products.query_count,
            factor_chunk_len,
        )?;
        let timestamp_layout = BvaFieldSource::<Fr>::layout(&timestamp_source).clone();
        let (timestamp_commitment, timestamp_tier_one, timestamp_profile) =
            commit_bva_field_streaming::<Bls12_381, _>(
                &parameters.pc_pp,
                &mut timestamp_source,
                timestamp_hiding,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
        if timestamp_profile.materialized_padding_entries != 0 {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        drop(timestamp_source);
        drop(timestamp_stream);
        transcript.absorb(b"lookup-timestamp-commitment", &timestamp_commitment);
        let timestamp_range_bits =
            ((usize::BITS - products.query_count.saturating_sub(1).leading_zeros()) as usize)
                .max(1);
        let mut timestamp_range_stream =
            Gpt2LookupQueryStream::new(checkpoint, &layout, &activations, &witness, &manifest)?;
        let mut timestamp_range_source = Gpt2LassoTimestampSource::<_, Fr>::new(
            &mut timestamp_range_stream,
            &registry,
            products.query_count,
            factor_chunk_len,
        )?;
        let (timestamp_range_proof, timestamp_range_profile) = BvaRangeProof::prove(
            &srs,
            0,
            0x5453_4256,
            &mut timestamp_range_source,
            timestamp_commitment,
            &timestamp_tier_one,
            timestamp_hiding,
            timestamp_range_bits,
        )
        .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
        if timestamp_range_profile.materialized_padding_entries != 0 {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        drop(timestamp_range_source);
        drop(timestamp_range_stream);

        let left_value_hiding = Fr::rand(&mut rng);
        let query_value_hiding = Fr::rand(&mut rng);
        let table_value_hiding = left_value_hiding - query_value_hiding;
        let left_counter_hiding = Fr::rand(&mut rng);
        let left_query_counter_hiding = Fr::rand(&mut rng);
        let left_table_counter_hiding = left_counter_hiding - left_query_counter_hiding;
        let mut left_component_stream =
            Gpt2LookupQueryStream::new(checkpoint, &layout, &activations, &witness, &manifest)?;
        let mut left_component_source = Gpt2LassoFactorSource::new(
            &mut left_component_stream,
            &registry,
            &products,
            row_challenge,
            Fr::from(1_u64),
            factor_chunk_len,
            LassoFactorSide::Left,
        )?;
        let factor_layout = BvaFieldSource::<Fr>::layout(&left_component_source).clone();
        let (factor_value_commitment, factor_value_tier_one, _) = {
            let mut component = Gpt2LassoAffineComponentSource::new(
                &mut left_component_source,
                LassoAffineComponent::Value,
            );
            commit_bva_field_streaming::<Bls12_381, _>(
                &parameters.pc_pp,
                &mut component,
                left_value_hiding,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?
        };
        let (query_value_commitment, query_value_tier_one, _) = {
            let mut component = Gpt2LassoAffineComponentSource::new(
                &mut left_component_source,
                LassoAffineComponent::QueryValue,
            );
            commit_bva_field_streaming::<Bls12_381, _>(
                &parameters.pc_pp,
                &mut component,
                query_value_hiding,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?
        };
        let (table_value_commitment, _, _) = {
            let mut component = Gpt2LassoAffineComponentSource::new(
                &mut left_component_source,
                LassoAffineComponent::TableValue,
            );
            commit_bva_field_streaming::<Bls12_381, _>(
                &parameters.pc_pp,
                &mut component,
                table_value_hiding,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?
        };
        let (left_counter_commitment, left_counter_tier_one, _) = {
            let mut component = Gpt2LassoAffineComponentSource::new(
                &mut left_component_source,
                LassoAffineComponent::Counter,
            );
            commit_bva_field_streaming::<Bls12_381, _>(
                &parameters.pc_pp,
                &mut component,
                left_counter_hiding,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?
        };
        let (left_query_counter_commitment, left_query_counter_tier_one, _) = {
            let mut component = Gpt2LassoAffineComponentSource::new(
                &mut left_component_source,
                LassoAffineComponent::QueryCounter,
            );
            commit_bva_field_streaming::<Bls12_381, _>(
                &parameters.pc_pp,
                &mut component,
                left_query_counter_hiding,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?
        };
        let (left_table_counter_commitment, _, _) = {
            let mut component = Gpt2LassoAffineComponentSource::new(
                &mut left_component_source,
                LassoAffineComponent::TableCounter,
            );
            commit_bva_field_streaming::<Bls12_381, _>(
                &parameters.pc_pp,
                &mut component,
                left_table_counter_hiding,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?
        };
        drop(left_component_source);
        drop(left_component_stream);
        if factor_value_commitment != query_value_commitment + table_value_commitment
            || left_counter_commitment
                != left_query_counter_commitment + left_table_counter_commitment
        {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }

        let right_counter_hiding = Fr::rand(&mut rng);
        let mut right_component_stream =
            Gpt2LookupQueryStream::new(checkpoint, &layout, &activations, &witness, &manifest)?;
        let mut right_component_source = Gpt2LassoFactorSource::new(
            &mut right_component_stream,
            &registry,
            &products,
            row_challenge,
            Fr::from(1_u64),
            factor_chunk_len,
            LassoFactorSide::Right,
        )?;
        let (right_value_commitment, _, _) = {
            let mut component = Gpt2LassoAffineComponentSource::new(
                &mut right_component_source,
                LassoAffineComponent::Value,
            );
            commit_bva_field_streaming::<Bls12_381, _>(
                &parameters.pc_pp,
                &mut component,
                left_value_hiding,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?
        };
        let (right_counter_commitment, right_counter_tier_one, _) = {
            let mut component = Gpt2LassoAffineComponentSource::new(
                &mut right_component_source,
                LassoAffineComponent::Counter,
            );
            commit_bva_field_streaming::<Bls12_381, _>(
                &parameters.pc_pp,
                &mut component,
                right_counter_hiding,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?
        };
        drop(right_component_source);
        drop(right_component_stream);
        if right_value_commitment != factor_value_commitment {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }

        let public_table_binding = Gpt2LassoPublicTableBinding::new(
            &products,
            multiplicity_hiding,
            table_value_hiding,
            left_table_counter_hiding,
        )?;
        public_table_binding.verify::<Bls12_381>(
            &parameters.pc_pp,
            &registry,
            &factor_layout,
            products.query_count,
            products.public_table_rows,
            row_challenge,
            multiplicity_commitment,
            table_value_commitment,
            left_table_counter_commitment,
        )?;

        absorb_lasso_component_commitments(
            &mut transcript,
            factor_value_commitment,
            query_value_commitment,
            table_value_commitment,
            left_query_counter_commitment,
            left_table_counter_commitment,
            right_counter_commitment,
        );
        let product_challenge = transcript.challenge::<Fr>(b"lookup-product");
        if product_challenge == Fr::from(0_u64) {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        let mut challenged_stream =
            Gpt2LookupQueryStream::new(checkpoint, &layout, &activations, &witness, &manifest)?;
        let challenged_products =
            challenged_stream.lasso_products(&registry, row_challenge, product_challenge)?;
        drop(challenged_stream);
        if challenged_products.multiplicities != products.multiplicities
            || challenged_products.left != challenged_products.right
        {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        products.left = challenged_products.left;
        products.right = challenged_products.right;
        let expected_product = products.left;

        let left_factor_hiding = Fr::rand(&mut rng);
        let mut left_stream =
            Gpt2LookupQueryStream::new(checkpoint, &layout, &activations, &witness, &manifest)?;
        let mut left_source = Gpt2LassoFactorSource::new(
            &mut left_stream,
            &registry,
            &products,
            row_challenge,
            product_challenge,
            factor_chunk_len,
            LassoFactorSide::Left,
        )?;
        let (left_factor_commitment, left_factor_tier_one, _) =
            commit_bva_field_streaming::<Bls12_381, _>(
                &parameters.pc_pp,
                &mut left_source,
                left_factor_hiding,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
        let (left_grand_product, _) = BvaGrandProductProof::prove(
            &srs,
            0,
            0x4c41_5353,
            &mut left_source,
            left_factor_commitment,
            &left_factor_tier_one,
            left_factor_hiding,
            expected_product,
        )
        .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
        let (left_affine, _) = BvaAffineRelationProof::prove(
            &srs,
            0,
            0x4c41_464c,
            &mut left_source,
            left_factor_commitment,
            &left_factor_tier_one,
            left_factor_hiding,
            factor_value_commitment,
            &factor_value_tier_one,
            left_value_hiding,
            left_counter_commitment,
            &left_counter_tier_one,
            left_counter_hiding,
            Fr::from(1_u64),
            product_challenge,
            product_challenge * product_challenge,
        )
        .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
        let (left_routes, left_constants) =
            lasso_query_timestamp_routes::<Fr>(&factor_layout, &timestamp_layout, Fr::from(0_u64))?;
        let mut left_timestamp_stream = Gpt2LookupQueryStream::new(
            replay_checkpoint,
            &layout,
            &activations,
            &witness,
            &manifest,
        )?;
        let mut left_timestamp_source = Gpt2LassoTimestampSource::<_, Fr>::new(
            &mut left_timestamp_stream,
            &registry,
            products.query_count,
            factor_chunk_len,
        )?;
        let (left_counter_timestamp, _) = {
            let mut left_query_counter_source = Gpt2LassoAffineComponentSource::new(
                &mut left_source,
                LassoAffineComponent::QueryCounter,
            );
            BvaRoutedEqualityProof::prove(
                &srs,
                0,
                0x4c41_5451,
                &mut left_query_counter_source,
                left_query_counter_commitment,
                &left_query_counter_tier_one,
                left_query_counter_hiding,
                &mut left_timestamp_source,
                timestamp_commitment,
                &timestamp_tier_one,
                timestamp_hiding,
                &left_routes,
                &left_constants,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?
        };
        drop(left_source);
        drop(left_stream);
        drop(left_timestamp_source);
        drop(left_timestamp_stream);

        let right_factor_hiding = Fr::rand(&mut rng);
        let mut right_stream =
            Gpt2LookupQueryStream::new(checkpoint, &layout, &activations, &witness, &manifest)?;
        let mut right_source = Gpt2LassoFactorSource::new(
            &mut right_stream,
            &registry,
            &products,
            row_challenge,
            product_challenge,
            factor_chunk_len,
            LassoFactorSide::Right,
        )?;
        let (right_factor_commitment, right_factor_tier_one, _) =
            commit_bva_field_streaming::<Bls12_381, _>(
                &parameters.pc_pp,
                &mut right_source,
                right_factor_hiding,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
        let (right_grand_product, _) = BvaGrandProductProof::prove(
            &srs,
            0,
            0x4c41_5354,
            &mut right_source,
            right_factor_commitment,
            &right_factor_tier_one,
            right_factor_hiding,
            expected_product,
        )
        .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
        let (right_affine, _) = BvaAffineRelationProof::prove(
            &srs,
            0,
            0x4c41_4652,
            &mut right_source,
            right_factor_commitment,
            &right_factor_tier_one,
            right_factor_hiding,
            factor_value_commitment,
            &factor_value_tier_one,
            left_value_hiding,
            right_counter_commitment,
            &right_counter_tier_one,
            right_counter_hiding,
            Fr::from(1_u64),
            product_challenge,
            product_challenge * product_challenge,
        )
        .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
        let (right_routes, right_constants) =
            lasso_query_timestamp_routes::<Fr>(&factor_layout, &timestamp_layout, Fr::from(1_u64))?;
        let mut right_timestamp_stream = Gpt2LookupQueryStream::new(
            replay_checkpoint,
            &layout,
            &activations,
            &witness,
            &manifest,
        )?;
        let mut right_timestamp_source = Gpt2LassoTimestampSource::<_, Fr>::new(
            &mut right_timestamp_stream,
            &registry,
            products.query_count,
            factor_chunk_len,
        )?;
        let (right_counter_timestamp, _) = {
            let mut right_counter_source = Gpt2LassoAffineComponentSource::new(
                &mut right_source,
                LassoAffineComponent::Counter,
            );
            BvaRoutedEqualityProof::prove(
                &srs,
                0,
                0x4c41_5453,
                &mut right_counter_source,
                right_counter_commitment,
                &right_counter_tier_one,
                right_counter_hiding,
                &mut right_timestamp_source,
                timestamp_commitment,
                &timestamp_tier_one,
                timestamp_hiding,
                &right_routes,
                &right_constants,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?
        };

        let proof = Self {
            query_count: products.query_count,
            public_table_rows: products.public_table_rows,
            factor_chunk_len,
            timestamp_range_bits,
            multiplicity_commitment,
            timestamp_commitment,
            factor_value_commitment,
            query_value_commitment,
            table_value_commitment,
            left_counter_commitment,
            left_query_counter_commitment,
            left_table_counter_commitment,
            right_counter_commitment,
            left_factor_commitment,
            right_factor_commitment,
            expected_product,
            public_table_binding,
            timestamp_range_proof,
            left_grand_product,
            right_grand_product,
            left_affine,
            right_affine,
            left_counter_timestamp,
            right_counter_timestamp,
        };
        let cache = Gpt2LassoProverCache {
            products,
            factor_layout,
            row_challenge,
            product_challenge,
            query_value_tier_one,
            query_value_hiding,
        };
        Ok((proof, cache))
    }

    pub fn verify_core(
        &self,
        parameters: &Gpt2GlobalRelationParameters,
        statement: &Gpt2PublicStatement,
        global_witness_commitment: PairingOutput<Bls12_381>,
    ) -> Result<bool, Gpt2CheckpointError> {
        let (metadata, layout, manifest, _) = global_relation_public_shape(statement)?;
        let shape = manifest.shape(&layout)?;
        let expected_queries = manifest
            .lookup_scalar_count()?
            .checked_add(shape.streamed_range_limb_entries)
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        let registry = Gpt2LookupRegistry::exact(&metadata);
        if self.query_count != expected_queries
            || self.public_table_rows != registry.public_table_rows()?
            || !self.factor_chunk_len.is_power_of_two()
            || self.timestamp_range_bits
                != ((usize::BITS - self.query_count.saturating_sub(1).leading_zeros()) as usize)
                    .max(1)
            || self.factor_value_commitment
                != self.query_value_commitment + self.table_value_commitment
            || self.left_counter_commitment
                != self.left_query_counter_commitment + self.left_table_counter_commitment
        {
            return Ok(false);
        }
        let products = StreamingLassoProducts {
            left: Fr::from(0_u64),
            right: Fr::from(0_u64),
            query_count: self.query_count,
            distinct_table_rows: self.public_table_binding.multiplicities.len(),
            public_table_rows: self.public_table_rows,
            maximum_live_semantic_entries: 0,
            multiplicities: self.public_table_binding.multiplicities.clone(),
        };
        let factor_layout = lasso_factor_layout(
            self.query_count,
            self.public_table_binding.multiplicities.len(),
            self.factor_chunk_len,
        )?;
        let timestamp_layout = lasso_timestamp_layout(self.query_count, self.factor_chunk_len)?;
        let mut transcript = Gpt2ProofTranscript::new(statement);
        transcript.absorb(b"global-witness-commitment", &global_witness_commitment);
        let row_challenge = transcript.challenge::<Fr>(b"lookup-row-compression");
        transcript.absorb(
            b"lookup-multiplicity-commitment",
            &self.multiplicity_commitment,
        );
        transcript.absorb(b"lookup-timestamp-commitment", &self.timestamp_commitment);
        absorb_lasso_component_commitments(
            &mut transcript,
            self.factor_value_commitment,
            self.query_value_commitment,
            self.table_value_commitment,
            self.left_query_counter_commitment,
            self.left_table_counter_commitment,
            self.right_counter_commitment,
        );
        let product_challenge = transcript.challenge::<Fr>(b"lookup-product");
        if product_challenge == Fr::from(0_u64) {
            return Ok(false);
        }
        let binding = self.public_table_binding.verify::<Bls12_381>(
            &parameters.pc_pp,
            &registry,
            &factor_layout,
            self.query_count,
            self.public_table_rows,
            row_challenge,
            self.multiplicity_commitment,
            self.table_value_commitment,
            self.left_table_counter_commitment,
        );
        if binding.is_err() {
            return Ok(false);
        }
        let srs = parameters.srs();
        if !self.timestamp_range_proof.verify(
            &srs,
            0,
            0x5453_4256,
            &timestamp_layout,
            self.timestamp_commitment,
            self.timestamp_range_bits,
        ) || !self.left_grand_product.verify(
            &srs,
            0,
            0x4c41_5353,
            &factor_layout,
            self.left_factor_commitment,
            self.expected_product,
        ) || !self.right_grand_product.verify(
            &srs,
            0,
            0x4c41_5354,
            &factor_layout,
            self.right_factor_commitment,
            self.expected_product,
        ) || !self.left_affine.verify(
            &srs,
            0,
            0x4c41_464c,
            &factor_layout,
            self.left_factor_commitment,
            self.factor_value_commitment,
            self.left_counter_commitment,
            Fr::from(1_u64),
            product_challenge,
            product_challenge * product_challenge,
        ) || !self.right_affine.verify(
            &srs,
            0,
            0x4c41_4652,
            &factor_layout,
            self.right_factor_commitment,
            self.factor_value_commitment,
            self.right_counter_commitment,
            Fr::from(1_u64),
            product_challenge,
            product_challenge * product_challenge,
        ) {
            return Ok(false);
        }
        let (left_routes, left_constants) =
            lasso_query_timestamp_routes::<Fr>(&factor_layout, &timestamp_layout, Fr::from(0_u64))?;
        let (right_routes, right_constants) =
            lasso_query_timestamp_routes::<Fr>(&factor_layout, &timestamp_layout, Fr::from(1_u64))?;
        if !self.left_counter_timestamp.verify(
            &srs,
            0,
            0x4c41_5451,
            &factor_layout,
            self.left_query_counter_commitment,
            &timestamp_layout,
            self.timestamp_commitment,
            &left_routes,
            &left_constants,
        ) || !self.right_counter_timestamp.verify(
            &srs,
            0,
            0x4c41_5453,
            &factor_layout,
            self.right_counter_commitment,
            &timestamp_layout,
            self.timestamp_commitment,
            &right_routes,
            &right_constants,
        ) {
            return Ok(false);
        }
        let _ = products;
        Ok(true)
    }
}

#[allow(clippy::too_many_arguments)]
fn absorb_lasso_component_commitments(
    transcript: &mut Gpt2ProofTranscript,
    value: PairingOutput<Bls12_381>,
    query_value: PairingOutput<Bls12_381>,
    table_value: PairingOutput<Bls12_381>,
    left_query_counter: PairingOutput<Bls12_381>,
    left_table_counter: PairingOutput<Bls12_381>,
    right_counter: PairingOutput<Bls12_381>,
) {
    transcript.absorb(b"lookup-lasso-value-commitment", &value);
    transcript.absorb(b"lookup-lasso-query-value-commitment", &query_value);
    transcript.absorb(b"lookup-lasso-table-value-commitment", &table_value);
    transcript.absorb(
        b"lookup-lasso-left-query-counter-commitment",
        &left_query_counter,
    );
    transcript.absorb(
        b"lookup-lasso-left-table-counter-commitment",
        &left_table_counter,
    );
    transcript.absorb(b"lookup-lasso-right-counter-commitment", &right_counter);
}

fn power_of_two_chunk_layout(
    mut entries: usize,
    chunk_len: usize,
) -> Result<Vec<usize>, Gpt2CheckpointError> {
    if entries == 0 || !chunk_len.is_power_of_two() {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    let mut lengths = Vec::new();
    while entries != 0 {
        let length = if entries >= chunk_len {
            chunk_len
        } else {
            1_usize << entries.ilog2()
        };
        lengths.push(length);
        entries -= length;
    }
    Ok(lengths)
}

fn public_support_digest(multiplicities: &[(usize, u64)]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"zksmart.gpt2.public-support.v1");
    hasher.update((multiplicities.len() as u64).to_le_bytes());
    for &(table_index, count) in multiplicities {
        hasher.update((table_index as u64).to_le_bytes());
        hasher.update(count.to_le_bytes());
    }
    hasher.finalize().into()
}

fn gpt2_progress(stage: &str) {
    if std::env::var_os("ZKSMART_PROGRESS").is_some() {
        eprintln!("gpt2_stage={stage}");
    }
}

fn public_support_lookup_challenges(
    statement: &Gpt2PublicStatement,
    global_witness_commitment: PairingOutput<Bls12_381>,
    multiplicities: &[(usize, u64)],
) -> (Fr, Fr) {
    let mut transcript = Gpt2ProofTranscript::new(statement);
    transcript.absorb(b"global-witness-commitment", &global_witness_commitment);
    let row_challenge = transcript.challenge::<Fr>(b"lookup-row-compression");
    transcript.absorb(
        b"lookup-public-support-digest",
        &public_support_digest(multiplicities),
    );
    let product_challenge = transcript.challenge::<Fr>(b"lookup-product");
    (row_challenge, product_challenge)
}

fn validate_public_support(
    registry: &Gpt2LookupRegistry,
    query_count: usize,
    multiplicities: &[(usize, u64)],
) -> Result<(), Gpt2CheckpointError> {
    if query_count == 0 || multiplicities.is_empty() {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    let public_table_rows = registry.public_table_rows()?;
    let mut previous = None;
    let mut total = 0_u64;
    for &(table_index, count) in multiplicities {
        if count == 0
            || table_index >= public_table_rows
            || previous.is_some_and(|value| value >= table_index)
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        registry.audit_query(registry.table_row(table_index)?)?;
        total = total
            .checked_add(count)
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        previous = Some(table_index);
    }
    if usize::try_from(total).ok() != Some(query_count) {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    Ok(())
}

fn public_support_expected_product(
    registry: &Gpt2LookupRegistry,
    multiplicities: &[(usize, u64)],
    row_challenge: Fr,
    product_challenge: Fr,
) -> Result<Fr, Gpt2CheckpointError> {
    let mut expected = Fr::from(1_u64);
    for &(table_index, count) in multiplicities {
        let compressed = compress_lookup_query(registry.table_row(table_index)?, row_challenge);
        expected *= (product_challenge - compressed).pow([count]);
    }
    Ok(expected)
}

fn lasso_timestamp_layout(
    query_count: usize,
    chunk_len: usize,
) -> Result<BvaLayout, Gpt2CheckpointError> {
    Ok(BvaLayout::from_lengths(&power_of_two_chunk_layout(
        query_count,
        chunk_len,
    )?)?)
}

fn lasso_factor_layout(
    query_count: usize,
    support_count: usize,
    chunk_len: usize,
) -> Result<BvaLayout, Gpt2CheckpointError> {
    let mut lengths = power_of_two_chunk_layout(query_count, chunk_len)?;
    lengths.extend(power_of_two_chunk_layout(support_count, chunk_len)?);
    Ok(BvaLayout::from_lengths(&lengths)?)
}

fn commit_global_witness_cache<R: Read + Seek>(
    parameters: &Gpt2GlobalRelationParameters,
    statement: &Gpt2PublicStatement,
    checkpoint: &mut Gpt2Q12Checkpoint<R>,
) -> Result<Gpt2GlobalWitnessCommitmentCache, Gpt2CheckpointError> {
    let expected = Gpt2PublicStatement::new(
        checkpoint.metadata(),
        checkpoint.token_ids(),
        checkpoint.reference_logits(),
    )?;
    if &expected != statement {
        return Err(Gpt2CheckpointError::PublicStatementDigestMismatch);
    }
    let metadata = checkpoint.metadata().clone();
    let layout = Gpt2GlobalLayout::compile(&metadata)?;
    let (_, activations) = checkpoint.execute_activation_witness(&layout)?;
    let manifest = Gpt2PrimitiveRelationManifest::compile(&metadata, &layout)?;
    let witness = Gpt2PrimitiveWitness::build(checkpoint, &layout, &activations, &manifest)?;
    let bva = manifest.compile_bva(&layout)?;
    let mut source = Gpt2PrimitiveBvaFieldSource::<_, Fr>::new(
        checkpoint,
        &layout,
        &activations,
        &manifest,
        &witness,
        &bva,
    )?;
    gpt2_progress("global_witness.materialize.start");
    let materialized = materialize_gpt2_bva_source(&mut source)?;
    gpt2_progress("global_witness.materialize.done");
    let mut rng = ark_std::rand::thread_rng();
    let hiding = Fr::rand(&mut rng);
    gpt2_progress("global_witness.parallel_commit.start");
    let (commitment, tier_one, profile) = commit_bva_field_materialized_parallel::<Bls12_381>(
        &parameters.pc_pp,
        &materialized,
        hiding,
    )
    .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
    gpt2_progress("global_witness.parallel_commit.done");
    if profile.materialized_padding_entries != 0 {
        return Err(Gpt2CheckpointError::ProtocolIncomplete);
    }
    Ok(Gpt2GlobalWitnessCommitmentCache {
        commitment,
        tier_one,
        hiding,
        materialized,
    })
}

fn materialize_gpt2_bva_source<S>(source: &mut S) -> Result<BucketedVector<Fr>, Gpt2CheckpointError>
where
    S: BvaFieldSource<Fr, Error = Gpt2CheckpointError>,
{
    let layout = source.layout().clone();
    let mut vectors = (0..layout.vector_lengths().len())
        .map(|_| None)
        .collect::<Vec<Option<Vec<Fr>>>>();
    let mut error = None;
    source.for_each_vector(&mut |vector_index, values| {
        if error.is_some() {
            return;
        }
        let Some(slot) = vectors.get_mut(vector_index) else {
            error = Some(Gpt2CheckpointError::ProtocolShapeMismatch);
            return;
        };
        if slot.is_some() || layout.vector_lengths()[vector_index] != values.len() {
            error = Some(Gpt2CheckpointError::ProtocolShapeMismatch);
            return;
        }
        *slot = Some(values.to_vec());
    })?;
    if let Some(error) = error {
        return Err(error);
    }
    let vectors = vectors
        .into_iter()
        .collect::<Option<Vec<_>>>()
        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
    let materialized = BucketedVector::new(vectors)?;
    if materialized.layout() != &layout
        || materialized.logical_padding_entries()
            != layout.logical_len() - materialized.physical_entries()
    {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    Ok(materialized)
}

fn evaluate_gpt2_bva_source<S>(
    source: &mut S,
    materialized: Option<&BucketedVector<Fr>>,
    tbv: &TensorBasisVector<Fr>,
) -> Result<Fr, Gpt2CheckpointError>
where
    S: BvaFieldSource<Fr, Error = Gpt2CheckpointError>,
{
    if let Some(materialized) = materialized {
        return materialized
            .inner_product(tbv)
            .map(|evaluation| evaluation.value)
            .map_err(Into::into);
    }
    let evaluation = inner_product_bva_field_streaming(source, tbv)
        .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
    if evaluation.profile.materialized_padding_entries != 0 {
        return Err(Gpt2CheckpointError::ProtocolIncomplete);
    }
    Ok(evaluation.value)
}

#[allow(clippy::too_many_arguments)]
fn prove_lasso_query_value_binding<R: Read + Seek>(
    parameters: &Gpt2GlobalRelationParameters,
    statement: &Gpt2PublicStatement,
    checkpoint: &mut Gpt2Q12Checkpoint<R>,
    lasso: &Gpt2LassoProof,
    cache: &Gpt2LassoProverCache,
    query_tbv: &TensorBasisVector<Fr>,
    lookup_compressed_projection: Fr,
) -> Result<BvaCommittedEvaluationProof<Bls12_381>, Gpt2CheckpointError> {
    if cache.factor_layout != *query_tbv.layout()
        || cache.products.query_count != lasso.query_count
        || cache.products.left != lasso.expected_product
        || cache.products.right != lasso.expected_product
    {
        return Err(Gpt2CheckpointError::ProtocolIncomplete);
    }
    let expected = Gpt2PublicStatement::new(
        checkpoint.metadata(),
        checkpoint.token_ids(),
        checkpoint.reference_logits(),
    )?;
    if &expected != statement {
        return Err(Gpt2CheckpointError::PublicStatementDigestMismatch);
    }
    let metadata = checkpoint.metadata().clone();
    let layout = Gpt2GlobalLayout::compile(&metadata)?;
    let (_, activations) = checkpoint.execute_activation_witness(&layout)?;
    let manifest = Gpt2PrimitiveRelationManifest::compile(&metadata, &layout)?;
    let witness = Gpt2PrimitiveWitness::build(checkpoint, &layout, &activations, &manifest)?;
    let registry = Gpt2LookupRegistry::exact(&metadata);
    let mut stream =
        Gpt2LookupQueryStream::new(checkpoint, &layout, &activations, &witness, &manifest)?;
    let mut source = Gpt2LassoFactorSource::new(
        &mut stream,
        &registry,
        &cache.products,
        cache.row_challenge,
        cache.product_challenge,
        lasso.factor_chunk_len,
        LassoFactorSide::Left,
    )?;
    let mut component =
        Gpt2LassoAffineComponentSource::new(&mut source, LassoAffineComponent::QueryValue);
    let (proof, profile) = BvaCommittedEvaluationProof::prove(
        &parameters.srs(),
        0,
        0x4c41_5651,
        statement.transcript_digest(),
        &mut component,
        lasso.query_value_commitment,
        &cache.query_value_tier_one,
        cache.query_value_hiding,
        query_tbv,
        lookup_compressed_projection,
    )
    .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
    if profile.materialized_padding_entries != 0 {
        return Err(Gpt2CheckpointError::ProtocolIncomplete);
    }
    Ok(proof)
}

const GPT2_GLOBAL_RELATION_PROOF_MAGIC: [u8; 8] = *b"ZKSGRPF\0";
const GPT2_GLOBAL_RELATION_PROOF_VERSION: u32 = 2;
const GPT2_GLOBAL_RELATION_PROOF_MAX_BYTES: usize = 32 * 1024 * 1024;
const GPT2_GLOBAL_RELATION_PROOF_MAX_ROUNDS: usize = 256;
const GPT2_PROOF_MAGIC: [u8; 8] = *b"ZKSGP2E\0";
const GPT2_PROOF_VERSION: u32 = 2;
const GPT2_PROOF_MAX_BYTES: usize = 1024 * 1024 * 1024;
const GPT2_SCALABLE_PROOF_MAGIC: [u8; 8] = *b"ZKSGP2R\0";
const GPT2_SCALABLE_PROOF_VERSION: u32 = 2;
const GPT2_SCALABLE_SUPPORT_MAX_ENTRIES: usize = 1 << 24;
const GPT2_PARAMETERS_MAGIC: [u8; 8] = *b"ZKSGP2P\0";
const GPT2_PARAMETERS_VERSION: u32 = 2;
const GPT2_PARAMETERS_MAX_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug)]
pub enum Gpt2ProofDecodeError {
    Truncated,
    InvalidMagic,
    UnsupportedVersion(u32),
    Oversized(usize),
    Serialization(SerializationError),
    TrailingBytes,
}

impl Display for Gpt2ProofDecodeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated => write!(formatter, "GPT-2 proof is truncated"),
            Self::InvalidMagic => write!(formatter, "invalid GPT-2 proof magic"),
            Self::UnsupportedVersion(version) => {
                write!(formatter, "unsupported GPT-2 proof version {version}")
            }
            Self::Oversized(bytes) => write!(formatter, "GPT-2 proof is oversized: {bytes} bytes"),
            Self::Serialization(error) => Display::fmt(error, formatter),
            Self::TrailingBytes => write!(formatter, "GPT-2 proof contains trailing bytes"),
        }
    }
}

impl std::error::Error for Gpt2ProofDecodeError {}

impl From<SerializationError> for Gpt2ProofDecodeError {
    fn from(error: SerializationError) -> Self {
        Self::Serialization(error)
    }
}

struct GlobalRelationMessages<'a> {
    global_witness_commitment: PairingOutput<Bls12_381>,
    lookup_left_product: Fr,
    lookup_right_product: Fr,
    lookup_claim: Fr,
    elementwise_claim: Fr,
    dense_matmul_claim: Fr,
    causal_qk_claim: Fr,
    causal_pv_claim: Fr,
    elementwise_proof: &'a EoWitnessPairProof<Fr>,
    dense_matmul_proof: &'a EoWitnessPairProof<Fr>,
    causal_qk_proof: &'a EoWitnessPairProof<Fr>,
    causal_pv_proof: &'a EoWitnessPairProof<Fr>,
}

struct GlobalRelationReduction {
    synchronized_tbv: TensorBasisVector<Fr>,
    synchronized_claim: Fr,
    query_tbv: Option<TensorBasisVector<Fr>>,
    lookup_compressed_projection: Fr,
}

struct GlobalRelationProverReduction {
    query_tbv: Option<TensorBasisVector<Fr>>,
    lookup_compressed_projection: Fr,
}

struct Gpt2GlobalWitnessCommitmentCache {
    commitment: PairingOutput<Bls12_381>,
    tier_one: Vec<<Bls12_381 as Pairing>::G1>,
    hiding: Fr,
    materialized: BucketedVector<Fr>,
}

impl Gpt2GlobalRelationProof {
    pub fn witness_commitment_count(&self) -> usize {
        1
    }

    pub fn witness_opening_count(&self) -> usize {
        1
    }

    pub fn to_bytes_exact(&self) -> Result<Vec<u8>, SerializationError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GPT2_GLOBAL_RELATION_PROOF_MAGIC);
        bytes.extend_from_slice(&GPT2_GLOBAL_RELATION_PROOF_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.statement_digest);
        self.global_witness_commitment
            .serialize_compressed(&mut bytes)?;
        for scalar in [
            self.lookup_left_product,
            self.lookup_right_product,
            self.lookup_claim,
            self.elementwise_claim,
            self.dense_matmul_claim,
            self.causal_qk_claim,
            self.causal_pv_claim,
        ] {
            scalar.serialize_compressed(&mut bytes)?;
        }
        self.elementwise_proof.serialize_compressed(&mut bytes)?;
        self.dense_matmul_proof.serialize_compressed(&mut bytes)?;
        self.causal_qk_proof.serialize_compressed(&mut bytes)?;
        self.causal_pv_proof.serialize_compressed(&mut bytes)?;
        self.synchronized_eo.serialize_compressed(&mut bytes)?;
        self.synchronized_value_commitment
            .serialize_compressed(&mut bytes)?;
        self.synchronized_opening.serialize_compressed(&mut bytes)?;
        Ok(bytes)
    }

    pub fn from_bytes_exact(bytes: &[u8]) -> Result<Self, Gpt2ProofDecodeError> {
        if bytes.len() > GPT2_GLOBAL_RELATION_PROOF_MAX_BYTES {
            return Err(Gpt2ProofDecodeError::Oversized(bytes.len()));
        }
        if bytes.len() < 12 + 32 {
            return Err(Gpt2ProofDecodeError::Truncated);
        }
        if bytes[..8] != GPT2_GLOBAL_RELATION_PROOF_MAGIC {
            return Err(Gpt2ProofDecodeError::InvalidMagic);
        }
        let version = u32::from_le_bytes(
            bytes[8..12]
                .try_into()
                .map_err(|_| Gpt2ProofDecodeError::Truncated)?,
        );
        if version != GPT2_GLOBAL_RELATION_PROOF_VERSION {
            return Err(Gpt2ProofDecodeError::UnsupportedVersion(version));
        }
        let mut reader = Cursor::new(&bytes[12..]);
        let mut statement_digest = [0_u8; 32];
        reader
            .read_exact(&mut statement_digest)
            .map_err(|_| Gpt2ProofDecodeError::Truncated)?;
        let global_witness_commitment =
            PairingOutput::<Bls12_381>::deserialize_compressed(&mut reader)?;
        let lookup_left_product = Fr::deserialize_compressed(&mut reader)?;
        let lookup_right_product = Fr::deserialize_compressed(&mut reader)?;
        let lookup_claim = Fr::deserialize_compressed(&mut reader)?;
        let elementwise_claim = Fr::deserialize_compressed(&mut reader)?;
        let dense_matmul_claim = Fr::deserialize_compressed(&mut reader)?;
        let causal_qk_claim = Fr::deserialize_compressed(&mut reader)?;
        let causal_pv_claim = Fr::deserialize_compressed(&mut reader)?;
        let elementwise_proof = deserialize_witness_pair_proof(&mut reader)?;
        let dense_matmul_proof = deserialize_witness_pair_proof(&mut reader)?;
        let causal_qk_proof = deserialize_witness_pair_proof(&mut reader)?;
        let causal_pv_proof = deserialize_witness_pair_proof(&mut reader)?;
        let synchronized_eo = deserialize_eo_lite_proof(&mut reader)?;
        let synchronized_value_commitment =
            PairingOutput::<Bls12_381>::deserialize_compressed(&mut reader)?;
        let synchronized_opening = deserialize_smart_pc_transcript(&mut reader)?;
        if reader.position() != (bytes.len() - 12) as u64 {
            return Err(Gpt2ProofDecodeError::TrailingBytes);
        }
        Ok(Self {
            statement_digest,
            global_witness_commitment,
            lookup_left_product,
            lookup_right_product,
            lookup_claim,
            elementwise_claim,
            dense_matmul_claim,
            causal_qk_claim,
            causal_pv_claim,
            elementwise_proof,
            dense_matmul_proof,
            causal_qk_proof,
            causal_pv_proof,
            synchronized_eo,
            synchronized_value_commitment,
            synchronized_opening,
        })
    }

    pub fn verify_bytes(
        parameters: &Gpt2GlobalRelationParameters,
        statement: &Gpt2PublicStatement,
        bytes: &[u8],
    ) -> Result<bool, Gpt2CheckpointError> {
        let proof = match Self::from_bytes_exact(bytes) {
            Ok(proof) => proof,
            Err(_) => return Ok(false),
        };
        proof.verify(parameters, statement)
    }

    pub fn prove<R: Read + Seek>(
        parameters: &Gpt2GlobalRelationParameters,
        statement: &Gpt2PublicStatement,
        checkpoint: &mut Gpt2Q12Checkpoint<R>,
    ) -> Result<Self, Gpt2CheckpointError> {
        Self::prove_internal(parameters, statement, checkpoint, None, None).map(|(proof, _)| proof)
    }

    fn prove_internal<R: Read + Seek>(
        parameters: &Gpt2GlobalRelationParameters,
        statement: &Gpt2PublicStatement,
        checkpoint: &mut Gpt2Q12Checkpoint<R>,
        precommitment: Option<Gpt2GlobalWitnessCommitmentCache>,
        lookup_argument: Option<Gpt2LookupArgument<'_>>,
    ) -> Result<(Self, GlobalRelationProverReduction), Gpt2CheckpointError> {
        let expected = Gpt2PublicStatement::new(
            checkpoint.metadata(),
            checkpoint.token_ids(),
            checkpoint.reference_logits(),
        )?;
        if &expected != statement {
            return Err(Gpt2CheckpointError::PublicStatementDigestMismatch);
        }
        let metadata = checkpoint.metadata().clone();
        let layout = Gpt2GlobalLayout::compile(&metadata)?;
        let (_, activations) = checkpoint.execute_activation_witness(&layout)?;
        let manifest = Gpt2PrimitiveRelationManifest::compile(&metadata, &layout)?;
        let witness = Gpt2PrimitiveWitness::build(checkpoint, &layout, &activations, &manifest)?;
        let bva = manifest.compile_bva(&layout)?;

        let (global_witness_commitment, tier_one, matrix_hiding, pre_materialized) =
            if let Some(precommitment) = precommitment {
                (
                    precommitment.commitment,
                    precommitment.tier_one,
                    precommitment.hiding,
                    Some(precommitment.materialized),
                )
            } else {
                let mut source = Gpt2PrimitiveBvaFieldSource::<_, Fr>::new(
                    checkpoint,
                    &layout,
                    &activations,
                    &manifest,
                    &witness,
                    &bva,
                )?;
                let mut rng = ark_std::rand::thread_rng();
                let matrix_hiding = Fr::rand(&mut rng);
                let (commitment, tier_one, commit_profile) =
                    commit_bva_field_streaming::<Bls12_381, _>(
                        &parameters.pc_pp,
                        &mut source,
                        matrix_hiding,
                    )
                    .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
                if commit_profile.materialized_padding_entries != 0 {
                    return Err(Gpt2CheckpointError::ProtocolIncomplete);
                }
                (commitment, tier_one, matrix_hiding, None)
            };

        let mut transcript = Gpt2ProofTranscript::new(statement);
        transcript.absorb(b"global-witness-commitment", &global_witness_commitment);
        let lookup_row_challenge = transcript.challenge::<Fr>(b"lookup-row-compression");
        let lookup_product_challenge = match lookup_argument {
            Some(Gpt2LookupArgument::Legacy(lasso)) => {
                transcript.absorb(
                    b"lookup-multiplicity-commitment",
                    &lasso.multiplicity_commitment,
                );
                transcript.absorb(b"lookup-timestamp-commitment", &lasso.timestamp_commitment);
                absorb_lasso_component_commitments(
                    &mut transcript,
                    lasso.factor_value_commitment,
                    lasso.query_value_commitment,
                    lasso.table_value_commitment,
                    lasso.left_query_counter_commitment,
                    lasso.left_table_counter_commitment,
                    lasso.right_counter_commitment,
                );
                transcript.challenge::<Fr>(b"lookup-product")
            }
            Some(Gpt2LookupArgument::Public(public)) => {
                transcript.absorb(
                    b"lookup-public-support-digest",
                    &public_support_digest(&public.multiplicities),
                );
                let challenge = transcript.challenge::<Fr>(b"lookup-product");
                transcript.absorb(
                    b"lookup-public-factor-commitment",
                    &public.factor_commitment,
                );
                challenge
            }
            None => transcript.challenge::<Fr>(b"lookup-product"),
        };
        let registry = Gpt2LookupRegistry::exact(&metadata);
        let products = match lookup_argument {
            Some(Gpt2LookupArgument::Public(public)) => {
                let expected = public_support_expected_product(
                    &registry,
                    &public.multiplicities,
                    lookup_row_challenge,
                    lookup_product_challenge,
                )?;
                if expected != public.expected_product {
                    return Err(Gpt2CheckpointError::ProtocolIncomplete);
                }
                StreamingLassoProducts {
                    left: expected,
                    right: expected,
                    query_count: public.query_count,
                    distinct_table_rows: public.multiplicities.len(),
                    public_table_rows: public.public_table_rows,
                    maximum_live_semantic_entries: 0,
                    multiplicities: public.multiplicities.clone(),
                }
            }
            legacy_or_none => {
                let mut lookup_stream = Gpt2LookupQueryStream::new(
                    checkpoint,
                    &layout,
                    &activations,
                    &witness,
                    &manifest,
                )?;
                let products = lookup_stream.lasso_products(
                    &registry,
                    lookup_row_challenge,
                    lookup_product_challenge,
                )?;
                if matches!(legacy_or_none, Some(Gpt2LookupArgument::Legacy(proof)) if
                    products.left != proof.expected_product || products.right != proof.expected_product)
                {
                    return Err(Gpt2CheckpointError::ProtocolIncomplete);
                }
                products
            }
        };

        let query_layout = match lookup_argument {
            Some(Gpt2LookupArgument::Legacy(proof)) => Some(lasso_factor_layout(
                proof.query_count,
                proof.public_table_binding.multiplicities.len(),
                proof.factor_chunk_len,
            )?),
            Some(Gpt2LookupArgument::Public(public)) => Some(lasso_timestamp_layout(
                public.query_count,
                public.factor_chunk_len,
            )?),
            None => None,
        };
        let (lookup_tbv, _, lookup_public_offset, _) = manifest.lookup_relations_tbv::<Fr>(
            &layout,
            &bva,
            query_layout.as_ref(),
            lookup_row_challenge,
            || transcript.challenge(b"lookup-leaf-routing"),
        )?;
        let (range_tbv, range_claim) = manifest
            .range_recomposition_tbv::<Fr>(&bva, || transcript.challenge(b"range-routing"))?;
        let (linear_tbv, linear_claim, _) = manifest.linear_relations_tbv::<Fr>(
            &metadata,
            &layout,
            &bva,
            &statement.token_ids,
            || transcript.challenge(b"linear-routing"),
        )?;
        let (io_tbv, io_claim) = manifest.public_io_tbv::<Fr>(
            &layout,
            &bva,
            &statement.token_ids,
            &statement.public_logits,
            || transcript.challenge(b"public-io-routing"),
        )?;

        let mut source = Gpt2PrimitiveBvaFieldSource::<_, Fr>::new(
            checkpoint,
            &layout,
            &activations,
            &manifest,
            &witness,
            &bva,
        )?;
        let mut materialized_source = if let Some(materialized) = pre_materialized {
            if materialized.layout() != &bva.layout {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            Some(materialized)
        } else if matches!(lookup_argument, Some(Gpt2LookupArgument::Public(_))) {
            Some(materialize_gpt2_bva_source(&mut source)?)
        } else {
            None
        };
        let lookup_claim =
            evaluate_gpt2_bva_source(&mut source, materialized_source.as_ref(), &lookup_tbv)?;
        if lookup_claim + lookup_public_offset == Fr::from(0_u64) {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        drop(lookup_tbv);
        for (tbv, claim) in [
            (&range_tbv, range_claim),
            (&linear_tbv, linear_claim),
            (&io_tbv, io_claim),
        ] {
            let evaluation =
                evaluate_gpt2_bva_source(&mut source, materialized_source.as_ref(), tbv)?;
            if evaluation != claim {
                return Err(Gpt2CheckpointError::ProtocolIncomplete);
            }
        }
        drop(range_tbv);
        drop(linear_tbv);
        drop(io_tbv);
        gpt2_progress("global_relation.linear_checks.done");

        let elementwise_routing =
            manifest.elementwise_bilinear_routing(&metadata, &layout, &bva)?;
        let elementwise_point = (0..elementwise_routing.layout.logical_len().ilog2())
            .map(|_| transcript.challenge(b"elementwise-bilinear-routing"))
            .collect::<Vec<Fr>>();
        let elementwise_weight =
            TensorBasisVector::global_tensor(&elementwise_routing.layout, &elementwise_point)?;
        let elementwise_output_tbv =
            elementwise_routing.route_output(&elementwise_weight, &bva.layout)?;
        let elementwise_claim = evaluate_gpt2_bva_source(
            &mut source,
            materialized_source.as_ref(),
            &elementwise_output_tbv,
        )?;
        let elementwise_proof = {
            let mut pair_source =
                source.elementwise_pair_source(&elementwise_routing, &elementwise_weight)?;
            EoWitnessPairProof::prove_streaming(&mut pair_source, elementwise_claim)
                .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?
                .0
        };
        drop(elementwise_output_tbv);
        drop(elementwise_weight);
        drop(elementwise_routing);
        gpt2_progress("global_relation.elementwise.done");

        let dense_matmul_routing = manifest.dense_matmul_routing(&layout, &bva)?;
        let dense_matmul_projections = dense_matmul_routing
            .projections::<Fr>(|| transcript.challenge(b"dense-matmul-routing"))?;
        let dense_matmul_output_tbv =
            dense_matmul_routing.route_output(&dense_matmul_projections, &bva)?;
        let dense_matmul_claim = evaluate_gpt2_bva_source(
            &mut source,
            materialized_source.as_ref(),
            &dense_matmul_output_tbv,
        )?;
        let dense_matmul_proof = {
            let mut pair_source = source
                .dense_matmul_pair_source(&dense_matmul_routing, &dense_matmul_projections)?;
            EoWitnessPairProof::prove_streaming(&mut pair_source, dense_matmul_claim)
                .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?
                .0
        };
        drop(dense_matmul_output_tbv);
        drop(dense_matmul_projections);
        drop(dense_matmul_routing);
        gpt2_progress("global_relation.dense_matmul.done");

        let causal_qk_routing = manifest.causal_qk_routing(&metadata, &layout, &bva)?;
        let causal_qk_projections =
            causal_qk_routing.projections::<Fr>(|| transcript.challenge(b"causal-qk-routing"))?;
        let causal_qk_output_tbv = causal_qk_routing.route_output(&causal_qk_projections, &bva)?;
        let causal_qk_claim = evaluate_gpt2_bva_source(
            &mut source,
            materialized_source.as_ref(),
            &causal_qk_output_tbv,
        )?;
        let causal_qk_proof = {
            let mut pair_source =
                source.causal_qk_pair_source(&causal_qk_routing, &causal_qk_projections)?;
            EoWitnessPairProof::prove_streaming(&mut pair_source, causal_qk_claim)
                .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?
                .0
        };
        drop(causal_qk_output_tbv);
        drop(causal_qk_projections);
        drop(causal_qk_routing);
        gpt2_progress("global_relation.causal_qk.done");

        let causal_pv_routing = manifest.causal_pv_routing(&metadata, &layout, &bva)?;
        let causal_pv_projections =
            causal_pv_routing.projections::<Fr>(|| transcript.challenge(b"causal-pv-routing"))?;
        let causal_pv_output_tbv = causal_pv_routing.route_output(&causal_pv_projections, &bva)?;
        let causal_pv_claim = evaluate_gpt2_bva_source(
            &mut source,
            materialized_source.as_ref(),
            &causal_pv_output_tbv,
        )?;
        let causal_pv_proof = {
            let mut pair_source =
                source.causal_pv_pair_source(&causal_pv_routing, &causal_pv_projections)?;
            EoWitnessPairProof::prove_streaming(&mut pair_source, causal_pv_claim)
                .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?
                .0
        };
        drop(causal_pv_output_tbv);
        drop(causal_pv_projections);
        drop(causal_pv_routing);
        gpt2_progress("global_relation.causal_pv.done");

        let messages = GlobalRelationMessages {
            global_witness_commitment,
            lookup_left_product: products.left,
            lookup_right_product: products.right,
            lookup_claim,
            elementwise_claim,
            dense_matmul_claim,
            causal_qk_claim,
            causal_pv_claim,
            elementwise_proof: &elementwise_proof,
            dense_matmul_proof: &dense_matmul_proof,
            causal_qk_proof: &causal_qk_proof,
            causal_pv_proof: &causal_pv_proof,
        };
        gpt2_progress("global_relation.reduction.start");
        let reduction = derive_global_relation_reduction(
            statement,
            &metadata,
            &layout,
            &manifest,
            &bva,
            &messages,
            lookup_argument,
        )?
        .ok_or(Gpt2CheckpointError::ProtocolIncomplete)?;
        let GlobalRelationReduction {
            synchronized_tbv,
            synchronized_claim,
            query_tbv,
            lookup_compressed_projection,
        } = reduction;
        gpt2_progress("global_relation.reduction.done");
        let synchronized_evaluation =
            evaluate_gpt2_bva_source(&mut source, materialized_source.as_ref(), &synchronized_tbv)?;
        if synchronized_evaluation != synchronized_claim {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        let synchronized_logical_len = synchronized_tbv.layout().logical_len();
        gpt2_progress("global_relation.synchronized_eo.start");
        let (synchronized_eo, counters) = if let Some(owned) = materialized_source.as_ref() {
            EoLiteBulletProof::prove_consuming_tbv(owned, synchronized_tbv, synchronized_claim)
                .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?
        } else {
            let (proof, counters, profile) = EoLiteBulletProof::prove_streaming(
                &mut source,
                &synchronized_tbv,
                synchronized_claim,
            )
            .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
            if profile.materialized_padding_entries != 0 {
                return Err(Gpt2CheckpointError::ProtocolIncomplete);
            }
            (proof, counters)
        };
        gpt2_progress("global_relation.synchronized_eo.done");
        if counters.padding_ip_field_ops != 0 || counters.padding_fold_field_ops != 0 {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        let synchronized_verification =
            synchronized_eo.verify_scalar_for_prover(synchronized_logical_len, synchronized_claim);
        if !synchronized_verification.accepted {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        let (xl, xr) = bva_eo_opening_point(
            bva.logical_entries(),
            &synchronized_verification.challenges_inv,
        )
        .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;

        // EO currently discloses its terminal scalar.  Until the EO transcript
        // is zero-knowledge, use a zero-hiding evaluation commitment so the
        // verifier can bind that exact scalar to the SMART-PC opening.  A
        // hidden but unlinked value commitment would be unsound.
        let synchronized_value_hiding = Fr::from(0_u64);
        let synchronized_value_commitment = Bls12_381::pairing(
            parameters.pc_pp.g_0.mul(synchronized_eo.a_reduced),
            parameters.pc_pp.h_hat,
        );
        gpt2_progress("global_relation.synchronized_opening.start");
        let (synchronized_opening, opening_profile) =
            if let Some(owned) = materialized_source.as_mut() {
                open_bva_field_streaming::<Bls12_381, _>(
                    &parameters.pc_pp,
                    owned,
                    &xl,
                    &xr,
                    synchronized_value_commitment,
                    global_witness_commitment,
                    &tier_one,
                    synchronized_value_hiding,
                    matrix_hiding,
                )
                .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)
            } else {
                open_bva_field_streaming::<Bls12_381, _>(
                    &parameters.pc_pp,
                    &mut source,
                    &xl,
                    &xr,
                    synchronized_value_commitment,
                    global_witness_commitment,
                    &tier_one,
                    synchronized_value_hiding,
                    matrix_hiding,
                )
                .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)
            }?;
        gpt2_progress("global_relation.synchronized_opening.done");
        if opening_profile.materialized_projection_padding_entries != 0 {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }

        let proof = Self {
            statement_digest: statement.transcript_digest(),
            global_witness_commitment,
            lookup_left_product: products.left,
            lookup_right_product: products.right,
            lookup_claim,
            elementwise_claim,
            dense_matmul_claim,
            causal_qk_claim,
            causal_pv_claim,
            elementwise_proof,
            dense_matmul_proof,
            causal_qk_proof,
            causal_pv_proof,
            synchronized_eo,
            synchronized_value_commitment,
            synchronized_opening,
        };
        Ok((
            proof,
            GlobalRelationProverReduction {
                query_tbv,
                lookup_compressed_projection,
            },
        ))
    }

    /// Statement-only verification: no checkpoint, activation witness, query
    /// stream, or prover cache is accepted by this API.
    pub fn verify(
        &self,
        parameters: &Gpt2GlobalRelationParameters,
        statement: &Gpt2PublicStatement,
    ) -> Result<bool, Gpt2CheckpointError> {
        statement.validate()?;
        if self.statement_digest != statement.transcript_digest() {
            return Ok(false);
        }
        let (metadata, layout, manifest, bva) = global_relation_public_shape(statement)?;
        let messages = GlobalRelationMessages {
            global_witness_commitment: self.global_witness_commitment,
            lookup_left_product: self.lookup_left_product,
            lookup_right_product: self.lookup_right_product,
            lookup_claim: self.lookup_claim,
            elementwise_claim: self.elementwise_claim,
            dense_matmul_claim: self.dense_matmul_claim,
            causal_qk_claim: self.causal_qk_claim,
            causal_pv_claim: self.causal_pv_claim,
            elementwise_proof: &self.elementwise_proof,
            dense_matmul_proof: &self.dense_matmul_proof,
            causal_qk_proof: &self.causal_qk_proof,
            causal_pv_proof: &self.causal_pv_proof,
        };
        let Some(reduction) = derive_global_relation_reduction(
            statement, &metadata, &layout, &manifest, &bva, &messages, None,
        )?
        else {
            return Ok(false);
        };
        let synchronized_verification = self
            .synchronized_eo
            .verify_with_tbv(&reduction.synchronized_tbv, reduction.synchronized_claim);
        if !synchronized_verification.accepted {
            return Ok(false);
        }
        let expected_value_commitment = Bls12_381::pairing(
            parameters.pc_pp.g_0.mul(self.synchronized_eo.a_reduced),
            parameters.pc_pp.h_hat,
        );
        if self.synchronized_value_commitment != expected_value_commitment {
            return Ok(false);
        }
        let (xl, xr) = bva_eo_opening_point(
            bva.logical_entries(),
            &synchronized_verification.challenges_inv,
        )
        .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)?;
        SmartPC::<Bls12_381>::verify(
            &parameters.pc_pp,
            self.global_witness_commitment,
            self.synchronized_value_commitment,
            &xl,
            &xr,
            &self.synchronized_opening,
        )
        .map_err(|_| Gpt2CheckpointError::ProtocolIncomplete)
    }
}

fn serialize_lasso_proof(
    proof: &Gpt2LassoProof,
    writer: &mut impl Write,
) -> Result<(), SerializationError> {
    for value in [
        proof.query_count,
        proof.public_table_rows,
        proof.factor_chunk_len,
        proof.timestamp_range_bits,
    ] {
        (value as u64).serialize_compressed(&mut *writer)?;
    }
    for commitment in [
        proof.multiplicity_commitment,
        proof.timestamp_commitment,
        proof.factor_value_commitment,
        proof.query_value_commitment,
        proof.table_value_commitment,
        proof.left_counter_commitment,
        proof.left_query_counter_commitment,
        proof.left_table_counter_commitment,
        proof.right_counter_commitment,
        proof.left_factor_commitment,
        proof.right_factor_commitment,
    ] {
        commitment.serialize_compressed(&mut *writer)?;
    }
    proof.expected_product.serialize_compressed(&mut *writer)?;
    (proof.public_table_binding.multiplicities.len() as u64).serialize_compressed(&mut *writer)?;
    for &(index, count) in &proof.public_table_binding.multiplicities {
        (index as u64).serialize_compressed(&mut *writer)?;
        count.serialize_compressed(&mut *writer)?;
    }
    proof
        .public_table_binding
        .multiplicity_hiding
        .serialize_compressed(&mut *writer)?;
    proof
        .public_table_binding
        .table_value_hiding
        .serialize_compressed(&mut *writer)?;
    proof
        .public_table_binding
        .table_counter_hiding
        .serialize_compressed(&mut *writer)?;
    serialize_range_proof(&proof.timestamp_range_proof, writer)?;
    serialize_grand_product_proof(&proof.left_grand_product, writer)?;
    serialize_grand_product_proof(&proof.right_grand_product, writer)?;
    serialize_affine_proof(&proof.left_affine, writer)?;
    serialize_affine_proof(&proof.right_affine, writer)?;
    serialize_routed_equality_proof(&proof.left_counter_timestamp, writer)?;
    serialize_routed_equality_proof(&proof.right_counter_timestamp, writer)
}

fn deserialize_lasso_proof_exact(bytes: &[u8]) -> Result<Gpt2LassoProof, Gpt2ProofDecodeError> {
    let mut reader = Cursor::new(bytes);
    let query_count = deserialize_usize(&mut reader)?;
    let public_table_rows = deserialize_usize(&mut reader)?;
    let factor_chunk_len = deserialize_usize(&mut reader)?;
    let timestamp_range_bits = deserialize_usize(&mut reader)?;
    let multiplicity_commitment = PairingOutput::deserialize_compressed(&mut reader)?;
    let timestamp_commitment = PairingOutput::deserialize_compressed(&mut reader)?;
    let factor_value_commitment = PairingOutput::deserialize_compressed(&mut reader)?;
    let query_value_commitment = PairingOutput::deserialize_compressed(&mut reader)?;
    let table_value_commitment = PairingOutput::deserialize_compressed(&mut reader)?;
    let left_counter_commitment = PairingOutput::deserialize_compressed(&mut reader)?;
    let left_query_counter_commitment = PairingOutput::deserialize_compressed(&mut reader)?;
    let left_table_counter_commitment = PairingOutput::deserialize_compressed(&mut reader)?;
    let right_counter_commitment = PairingOutput::deserialize_compressed(&mut reader)?;
    let left_factor_commitment = PairingOutput::deserialize_compressed(&mut reader)?;
    let right_factor_commitment = PairingOutput::deserialize_compressed(&mut reader)?;
    let expected_product = Fr::deserialize_compressed(&mut reader)?;
    let support_len = read_bounded_len(
        &mut reader,
        public_table_rows
            .min(query_count)
            .min(bytes.len().saturating_div(16)),
    )?;
    let mut multiplicities = Vec::with_capacity(support_len);
    for _ in 0..support_len {
        multiplicities.push((
            deserialize_usize(&mut reader)?,
            u64::deserialize_compressed(&mut reader)?,
        ));
    }
    let public_table_binding = Gpt2LassoPublicTableBinding {
        multiplicities,
        multiplicity_hiding: Fr::deserialize_compressed(&mut reader)?,
        table_value_hiding: Fr::deserialize_compressed(&mut reader)?,
        table_counter_hiding: Fr::deserialize_compressed(&mut reader)?,
    };
    let timestamp_range_proof = deserialize_range_proof(&mut reader)?;
    let left_grand_product = deserialize_grand_product_proof(&mut reader)?;
    let right_grand_product = deserialize_grand_product_proof(&mut reader)?;
    let left_affine = deserialize_affine_proof(&mut reader)?;
    let right_affine = deserialize_affine_proof(&mut reader)?;
    let left_counter_timestamp = deserialize_routed_equality_proof(&mut reader)?;
    let right_counter_timestamp = deserialize_routed_equality_proof(&mut reader)?;
    if reader.position() != bytes.len() as u64 {
        return Err(Gpt2ProofDecodeError::TrailingBytes);
    }
    Ok(Gpt2LassoProof {
        query_count,
        public_table_rows,
        factor_chunk_len,
        timestamp_range_bits,
        multiplicity_commitment,
        timestamp_commitment,
        factor_value_commitment,
        query_value_commitment,
        table_value_commitment,
        left_counter_commitment,
        left_query_counter_commitment,
        left_table_counter_commitment,
        right_counter_commitment,
        left_factor_commitment,
        right_factor_commitment,
        expected_product,
        public_table_binding,
        timestamp_range_proof,
        left_grand_product,
        right_grand_product,
        left_affine,
        right_affine,
        left_counter_timestamp,
        right_counter_timestamp,
    })
}

fn serialize_vector_opening(
    opening: &BvaVectorOpeningProof<Bls12_381>,
    writer: &mut impl Write,
) -> Result<(), SerializationError> {
    opening
        .value_commitment
        .serialize_compressed(&mut *writer)?;
    opening.opening.serialize_compressed(&mut *writer)
}

fn deserialize_vector_opening(
    reader: &mut impl Read,
) -> Result<BvaVectorOpeningProof<Bls12_381>, SerializationError> {
    Ok(BvaVectorOpeningProof {
        value_commitment: <Bls12_381 as Pairing>::G1::deserialize_compressed(&mut *reader)?,
        opening: deserialize_smart_pc_transcript(reader)?,
    })
}

fn serialize_range_proof(
    proof: &BvaRangeProof<Bls12_381>,
    writer: &mut impl Write,
) -> Result<(), SerializationError> {
    proof.bit_commitments.serialize_compressed(&mut *writer)?;
    serialize_vector_opening(&proof.value_linear_opening, writer)?;
    serialize_opening_vec(&proof.bit_linear_openings, writer)?;
    serialize_opening_vec(&proof.bit_minus_one_linear_openings, writer)?;
    (proof.boolean_proofs.len() as u64).serialize_compressed(&mut *writer)?;
    for boolean in &proof.boolean_proofs {
        boolean
            .bit_minus_one_commitment
            .serialize_compressed(&mut *writer)?;
        boolean.eo_proof.serialize_compressed(&mut *writer)?;
        serialize_vector_opening(&boolean.bit_eo_opening, writer)?;
        serialize_vector_opening(&boolean.bit_minus_one_eo_opening, writer)?;
    }
    Ok(())
}

fn deserialize_range_proof(
    reader: &mut impl Read,
) -> Result<BvaRangeProof<Bls12_381>, SerializationError> {
    let bit_commitments = deserialize_bounded_vec(reader, 128)?;
    let value_linear_opening = deserialize_vector_opening(reader)?;
    let bit_linear_openings = deserialize_opening_vec(reader, 128)?;
    let bit_minus_one_linear_openings = deserialize_opening_vec(reader, 128)?;
    let boolean_len = read_bounded_len(reader, 128)?;
    let mut boolean_proofs = Vec::with_capacity(boolean_len);
    for _ in 0..boolean_len {
        boolean_proofs.push(BvaBooleanProof {
            bit_minus_one_commitment: PairingOutput::deserialize_compressed(&mut *reader)?,
            eo_proof: deserialize_witness_pair_proof(reader)?,
            bit_eo_opening: deserialize_vector_opening(reader)?,
            bit_minus_one_eo_opening: deserialize_vector_opening(reader)?,
        });
    }
    Ok(BvaRangeProof {
        bit_commitments,
        value_linear_opening,
        bit_linear_openings,
        bit_minus_one_linear_openings,
        boolean_proofs,
    })
}

fn serialize_opening_vec(
    openings: &[BvaVectorOpeningProof<Bls12_381>],
    writer: &mut impl Write,
) -> Result<(), SerializationError> {
    (openings.len() as u64).serialize_compressed(&mut *writer)?;
    for opening in openings {
        serialize_vector_opening(opening, writer)?;
    }
    Ok(())
}

fn deserialize_opening_vec(
    reader: &mut impl Read,
    maximum_len: usize,
) -> Result<Vec<BvaVectorOpeningProof<Bls12_381>>, SerializationError> {
    let len = read_bounded_len(reader, maximum_len)?;
    let mut openings = Vec::with_capacity(len);
    for _ in 0..len {
        openings.push(deserialize_vector_opening(reader)?);
    }
    Ok(openings)
}

fn serialize_grand_product_opening(
    opening: &BvaGrandProductOpening<Bls12_381>,
    writer: &mut impl Write,
) -> Result<(), SerializationError> {
    opening
        .value_commitment
        .serialize_compressed(&mut *writer)?;
    opening.opening.serialize_compressed(&mut *writer)
}

fn deserialize_grand_product_opening(
    reader: &mut impl Read,
) -> Result<BvaGrandProductOpening<Bls12_381>, SerializationError> {
    Ok(BvaGrandProductOpening {
        value_commitment: <Bls12_381 as Pairing>::G1::deserialize_compressed(&mut *reader)?,
        opening: deserialize_smart_pc_transcript(reader)?,
    })
}

fn serialize_grand_product_proof(
    proof: &BvaGrandProductProof<Bls12_381>,
    writer: &mut impl Write,
) -> Result<(), SerializationError> {
    proof.prefix_commitment.serialize_compressed(&mut *writer)?;
    proof
        .product_commitment
        .serialize_compressed(&mut *writer)?;
    proof
        .hadamard
        .product_claim
        .serialize_compressed(&mut *writer)?;
    proof.hadamard.eo_proof.serialize_compressed(&mut *writer)?;
    serialize_grand_product_opening(&proof.hadamard.factor_opening, writer)?;
    serialize_grand_product_opening(&proof.hadamard.prefix_opening, writer)?;
    serialize_grand_product_opening(&proof.hadamard.product_opening, writer)?;
    proof
        .recurrence
        .prefix_claim
        .serialize_compressed(&mut *writer)?;
    proof
        .recurrence
        .product_claim
        .serialize_compressed(&mut *writer)?;
    proof
        .recurrence
        .prefix_eo_proof
        .serialize_compressed(&mut *writer)?;
    proof
        .recurrence
        .product_eo_proof
        .serialize_compressed(&mut *writer)?;
    serialize_grand_product_opening(&proof.recurrence.prefix_opening, writer)?;
    serialize_grand_product_opening(&proof.recurrence.product_opening, writer)
}

fn deserialize_grand_product_proof(
    reader: &mut impl Read,
) -> Result<BvaGrandProductProof<Bls12_381>, SerializationError> {
    Ok(BvaGrandProductProof {
        prefix_commitment: PairingOutput::deserialize_compressed(&mut *reader)?,
        product_commitment: PairingOutput::deserialize_compressed(&mut *reader)?,
        hadamard: BvaGrandProductHadamardProof {
            product_claim: Fr::deserialize_compressed(&mut *reader)?,
            eo_proof: deserialize_witness_pair_proof(reader)?,
            factor_opening: deserialize_grand_product_opening(reader)?,
            prefix_opening: deserialize_grand_product_opening(reader)?,
            product_opening: deserialize_grand_product_opening(reader)?,
        },
        recurrence: BvaGrandProductRecurrenceProof {
            prefix_claim: Fr::deserialize_compressed(&mut *reader)?,
            product_claim: Fr::deserialize_compressed(&mut *reader)?,
            prefix_eo_proof: deserialize_eo_lite_proof(reader)?,
            product_eo_proof: deserialize_eo_lite_proof(reader)?,
            prefix_opening: deserialize_grand_product_opening(reader)?,
            product_opening: deserialize_grand_product_opening(reader)?,
        },
    })
}

fn deserialize_grand_product_proof_exact(
    bytes: &[u8],
) -> Result<BvaGrandProductProof<Bls12_381>, Gpt2ProofDecodeError> {
    let mut reader = Cursor::new(bytes);
    let proof = deserialize_grand_product_proof(&mut reader)?;
    if reader.position() != bytes.len() as u64 {
        return Err(Gpt2ProofDecodeError::TrailingBytes);
    }
    Ok(proof)
}

fn serialize_affine_proof(
    proof: &BvaAffineRelationProof<Bls12_381>,
    writer: &mut impl Write,
) -> Result<(), SerializationError> {
    serialize_vector_opening(&proof.factor_opening, writer)?;
    serialize_vector_opening(&proof.value_opening, writer)?;
    serialize_vector_opening(&proof.counter_opening, writer)
}

fn deserialize_affine_proof(
    reader: &mut impl Read,
) -> Result<BvaAffineRelationProof<Bls12_381>, SerializationError> {
    Ok(BvaAffineRelationProof {
        factor_opening: deserialize_vector_opening(reader)?,
        value_opening: deserialize_vector_opening(reader)?,
        counter_opening: deserialize_vector_opening(reader)?,
    })
}

fn serialize_routed_equality_proof(
    proof: &BvaRoutedEqualityProof<Bls12_381>,
    writer: &mut impl Write,
) -> Result<(), SerializationError> {
    proof.target_claim.serialize_compressed(&mut *writer)?;
    proof.target_eo_proof.serialize_compressed(&mut *writer)?;
    serialize_vector_opening(&proof.source_opening, writer)?;
    serialize_vector_opening(&proof.target_opening, writer)
}

fn deserialize_routed_equality_proof(
    reader: &mut impl Read,
) -> Result<BvaRoutedEqualityProof<Bls12_381>, SerializationError> {
    Ok(BvaRoutedEqualityProof {
        target_claim: Fr::deserialize_compressed(&mut *reader)?,
        target_eo_proof: deserialize_eo_lite_proof(reader)?,
        source_opening: deserialize_vector_opening(reader)?,
        target_opening: deserialize_vector_opening(reader)?,
    })
}

fn serialize_committed_evaluation_proof(
    proof: &BvaCommittedEvaluationProof<Bls12_381>,
    writer: &mut impl Write,
) -> Result<(), SerializationError> {
    proof.eo_proof.serialize_compressed(&mut *writer)?;
    serialize_vector_opening(&proof.opening, writer)
}

fn deserialize_committed_evaluation_proof_exact(
    bytes: &[u8],
) -> Result<BvaCommittedEvaluationProof<Bls12_381>, Gpt2ProofDecodeError> {
    let mut reader = Cursor::new(bytes);
    let proof = BvaCommittedEvaluationProof {
        eo_proof: deserialize_eo_lite_proof(&mut reader)?,
        opening: deserialize_vector_opening(&mut reader)?,
    };
    if reader.position() != bytes.len() as u64 {
        return Err(Gpt2ProofDecodeError::TrailingBytes);
    }
    Ok(proof)
}

fn deserialize_usize(reader: &mut impl Read) -> Result<usize, SerializationError> {
    let value = u64::deserialize_compressed(&mut *reader)?;
    usize::try_from(value).map_err(|_| SerializationError::NotEnoughSpace)
}

fn read_bounded_len(
    reader: &mut impl Read,
    maximum_len: usize,
) -> Result<usize, SerializationError> {
    let len = deserialize_usize(reader)?;
    if len > maximum_len {
        return Err(SerializationError::NotEnoughSpace);
    }
    Ok(len)
}

fn deserialize_smart_pc_transcript(
    reader: &mut impl Read,
) -> Result<PcsTrans<Bls12_381>, SerializationError> {
    Ok(PcsTrans {
        vec_l_gt: deserialize_bounded_vec(reader, GPT2_GLOBAL_RELATION_PROOF_MAX_ROUNDS)?,
        vec_r_gt: deserialize_bounded_vec(reader, GPT2_GLOBAL_RELATION_PROOF_MAX_ROUNDS)?,
        vec_l_native: deserialize_bounded_vec(reader, GPT2_GLOBAL_RELATION_PROOF_MAX_ROUNDS)?,
        vec_r_native: deserialize_bounded_vec(reader, GPT2_GLOBAL_RELATION_PROOF_MAX_ROUNDS)?,
        v_g: <Bls12_381 as Pairing>::G1::deserialize_compressed(&mut *reader)?,
        v_h: <Bls12_381 as Pairing>::G2::deserialize_compressed(&mut *reader)?,
        v_g_prime: <Bls12_381 as Pairing>::G1::deserialize_compressed(&mut *reader)?,
        v_h_prime: <Bls12_381 as Pairing>::G2::deserialize_compressed(&mut *reader)?,
        w_g: <Bls12_381 as Pairing>::G1::deserialize_compressed(&mut *reader)?,
        w_h: <Bls12_381 as Pairing>::G1::deserialize_compressed(&mut *reader)?,
        linear_f: PairingOutput::<Bls12_381>::deserialize_compressed(&mut *reader)?,
        linear_z_a: Fr::deserialize_compressed(&mut *reader)?,
        linear_z_t: Fr::deserialize_compressed(&mut *reader)?,
    })
}

fn deserialize_bounded_vec<T: CanonicalDeserialize>(
    reader: &mut impl Read,
    maximum_len: usize,
) -> Result<Vec<T>, SerializationError> {
    let len = u64::deserialize_compressed(&mut *reader)?;
    let len = usize::try_from(len).map_err(|_| SerializationError::NotEnoughSpace)?;
    if len > maximum_len {
        return Err(SerializationError::NotEnoughSpace);
    }
    let mut values = Vec::with_capacity(len);
    for _ in 0..len {
        values.push(T::deserialize_compressed(&mut *reader)?);
    }
    Ok(values)
}

fn deserialize_witness_pair_proof(
    reader: &mut impl Read,
) -> Result<EoWitnessPairProof<Fr>, SerializationError> {
    Ok(EoWitnessPairProof {
        rounds: deserialize_bounded_vec(reader, GPT2_GLOBAL_RELATION_PROOF_MAX_ROUNDS)?,
        a_reduced: Fr::deserialize_compressed(&mut *reader)?,
        b_reduced: Fr::deserialize_compressed(&mut *reader)?,
    })
}

fn deserialize_eo_lite_proof(
    reader: &mut impl Read,
) -> Result<EoLiteBulletProof<Fr>, SerializationError> {
    Ok(EoLiteBulletProof {
        rounds: deserialize_bounded_vec(reader, GPT2_GLOBAL_RELATION_PROOF_MAX_ROUNDS)?,
        a_reduced: Fr::deserialize_compressed(&mut *reader)?,
        b_reduced: Fr::deserialize_compressed(&mut *reader)?,
    })
}

fn global_relation_public_shape(
    statement: &Gpt2PublicStatement,
) -> Result<
    (
        Gpt2Q12Metadata,
        Gpt2GlobalLayout,
        Gpt2PrimitiveRelationManifest,
        Gpt2PrimitiveBvaLayout,
    ),
    Gpt2CheckpointError,
> {
    statement.validate()?;
    let metadata = Gpt2Q12Metadata {
        format_version: 2,
        model: statement.model,
        fractional_bits: statement.fractional_bits,
        parameter_entries: statement.model.physical_parameter_entries()? as u64,
        token_count: statement.token_ids.len(),
        checkpoint_sha256: statement.checkpoint_sha256,
        tokenizer_sha256: statement.tokenizer_sha256,
    };
    let layout = Gpt2GlobalLayout::compile(&metadata)?;
    let manifest = Gpt2PrimitiveRelationManifest::compile(&metadata, &layout)?;
    let bva = manifest.compile_bva(&layout)?;
    Ok((metadata, layout, manifest, bva))
}

fn absorb_global_witness_pair(
    transcript: &mut Gpt2ProofTranscript,
    label: &[u8],
    claim: Fr,
    proof: &EoWitnessPairProof<Fr>,
) {
    transcript.absorb(label, &claim);
    for (left_cross, right_cross) in &proof.rounds {
        transcript.absorb(b"witness-pair-left-cross", left_cross);
        transcript.absorb(b"witness-pair-right-cross", right_cross);
    }
    transcript.absorb(b"witness-pair-left-terminal", &proof.a_reduced);
    transcript.absorb(b"witness-pair-right-terminal", &proof.b_reduced);
}

fn derive_global_relation_reduction(
    statement: &Gpt2PublicStatement,
    metadata: &Gpt2Q12Metadata,
    layout: &Gpt2GlobalLayout,
    manifest: &Gpt2PrimitiveRelationManifest,
    bva: &Gpt2PrimitiveBvaLayout,
    messages: &GlobalRelationMessages<'_>,
    lookup_argument: Option<Gpt2LookupArgument<'_>>,
) -> Result<Option<GlobalRelationReduction>, Gpt2CheckpointError> {
    if messages.lookup_left_product != messages.lookup_right_product {
        return Ok(None);
    }
    let mut transcript = Gpt2ProofTranscript::new(statement);
    transcript.absorb(
        b"global-witness-commitment",
        &messages.global_witness_commitment,
    );
    let lookup_row_challenge = transcript.challenge::<Fr>(b"lookup-row-compression");
    let query_layout = match lookup_argument {
        Some(Gpt2LookupArgument::Legacy(lasso)) => {
            transcript.absorb(
                b"lookup-multiplicity-commitment",
                &lasso.multiplicity_commitment,
            );
            transcript.absorb(b"lookup-timestamp-commitment", &lasso.timestamp_commitment);
            absorb_lasso_component_commitments(
                &mut transcript,
                lasso.factor_value_commitment,
                lasso.query_value_commitment,
                lasso.table_value_commitment,
                lasso.left_query_counter_commitment,
                lasso.left_table_counter_commitment,
                lasso.right_counter_commitment,
            );
            if messages.lookup_left_product != lasso.expected_product
                || messages.lookup_right_product != lasso.expected_product
            {
                return Ok(None);
            }
            let layout = lasso_factor_layout(
                lasso.query_count,
                lasso.public_table_binding.multiplicities.len(),
                lasso.factor_chunk_len,
            )?;
            let _ = transcript.challenge::<Fr>(b"lookup-product");
            Some(layout)
        }
        Some(Gpt2LookupArgument::Public(public)) => {
            transcript.absorb(
                b"lookup-public-support-digest",
                &public_support_digest(&public.multiplicities),
            );
            let product_challenge = transcript.challenge::<Fr>(b"lookup-product");
            transcript.absorb(
                b"lookup-public-factor-commitment",
                &public.factor_commitment,
            );
            let registry = Gpt2LookupRegistry::exact(metadata);
            let expected = public_support_expected_product(
                &registry,
                &public.multiplicities,
                lookup_row_challenge,
                product_challenge,
            )?;
            if expected != public.expected_product
                || messages.lookup_left_product != expected
                || messages.lookup_right_product != expected
            {
                return Ok(None);
            }
            Some(lasso_timestamp_layout(
                public.query_count,
                public.factor_chunk_len,
            )?)
        }
        None => {
            let _ = transcript.challenge::<Fr>(b"lookup-product");
            None
        }
    };
    let (lookup_tbv, query_tbv, lookup_public_offset, _) = manifest.lookup_relations_tbv::<Fr>(
        layout,
        bva,
        query_layout.as_ref(),
        lookup_row_challenge,
        || transcript.challenge(b"lookup-leaf-routing"),
    )?;
    if messages.lookup_claim + lookup_public_offset == Fr::from(0_u64) {
        return Ok(None);
    }
    let (range_tbv, range_claim) =
        manifest.range_recomposition_tbv::<Fr>(bva, || transcript.challenge(b"range-routing"))?;
    let (linear_tbv, linear_claim, _) =
        manifest.linear_relations_tbv::<Fr>(metadata, layout, bva, &statement.token_ids, || {
            transcript.challenge(b"linear-routing")
        })?;
    let (io_tbv, io_claim) = manifest.public_io_tbv::<Fr>(
        layout,
        bva,
        &statement.token_ids,
        &statement.public_logits,
        || transcript.challenge(b"public-io-routing"),
    )?;

    let elementwise_routing = manifest.elementwise_bilinear_routing(metadata, layout, bva)?;
    let elementwise_point = (0..elementwise_routing.layout.logical_len().ilog2())
        .map(|_| transcript.challenge(b"elementwise-bilinear-routing"))
        .collect::<Vec<Fr>>();
    let elementwise_weight =
        TensorBasisVector::global_tensor(&elementwise_routing.layout, &elementwise_point)?;
    let elementwise_output_tbv =
        elementwise_routing.route_output(&elementwise_weight, &bva.layout)?;
    let elementwise_verification = messages.elementwise_proof.verify(
        elementwise_routing.layout.logical_len(),
        messages.elementwise_claim,
    );
    if !elementwise_verification.accepted {
        return Ok(None);
    }
    let elementwise_left_equation = TensorBasisVector::global_tensor(
        &elementwise_routing.layout,
        &elementwise_verification.challenges_inv,
    )?;
    let elementwise_right_fold = TensorBasisVector::global_tensor(
        &elementwise_routing.layout,
        &elementwise_verification.challenges,
    )?;
    let elementwise_right_equation =
        elementwise_weight.pointwise_product(&elementwise_right_fold)?;
    let elementwise_left_tbv =
        elementwise_routing.route_left(&elementwise_left_equation, &bva.layout)?;
    let elementwise_right_tbv =
        elementwise_routing.route_right(&elementwise_right_equation, bva)?;
    let elementwise_right_claim = elementwise_routing.right_witness_claim(
        &elementwise_right_equation,
        messages.elementwise_proof.b_reduced,
    )?;

    let dense_matmul_routing = manifest.dense_matmul_routing(layout, bva)?;
    let dense_matmul_projections =
        dense_matmul_routing.projections::<Fr>(|| transcript.challenge(b"dense-matmul-routing"))?;
    let dense_matmul_output_tbv =
        dense_matmul_routing.route_output(&dense_matmul_projections, bva)?;
    let dense_matmul_verification = messages.dense_matmul_proof.verify(
        dense_matmul_routing.layout.logical_len(),
        messages.dense_matmul_claim,
    );
    if !dense_matmul_verification.accepted {
        return Ok(None);
    }
    let dense_matmul_left_equation = TensorBasisVector::global_tensor(
        &dense_matmul_routing.layout,
        &dense_matmul_verification.challenges_inv,
    )?;
    let dense_matmul_right_equation = TensorBasisVector::global_tensor(
        &dense_matmul_routing.layout,
        &dense_matmul_verification.challenges,
    )?;
    let dense_matmul_left_tbv = dense_matmul_routing.route_left(
        &dense_matmul_left_equation,
        &dense_matmul_projections,
        bva,
    )?;
    let dense_matmul_right_tbv = dense_matmul_routing.route_right(
        &dense_matmul_right_equation,
        &dense_matmul_projections,
        bva,
    )?;

    let causal_qk_routing = manifest.causal_qk_routing(metadata, layout, bva)?;
    let causal_qk_projections =
        causal_qk_routing.projections::<Fr>(|| transcript.challenge(b"causal-qk-routing"))?;
    let causal_qk_output_tbv = causal_qk_routing.route_output(&causal_qk_projections, bva)?;
    let causal_qk_verification = messages.causal_qk_proof.verify(
        causal_qk_routing.layout.logical_len(),
        messages.causal_qk_claim,
    );
    if !causal_qk_verification.accepted {
        return Ok(None);
    }
    let causal_qk_left_equation = TensorBasisVector::global_tensor(
        &causal_qk_routing.layout,
        &causal_qk_verification.challenges_inv,
    )?;
    let causal_qk_right_equation = TensorBasisVector::global_tensor(
        &causal_qk_routing.layout,
        &causal_qk_verification.challenges,
    )?;
    let causal_qk_left_tbv = causal_qk_routing.route_terminal(
        &causal_qk_left_equation,
        &causal_qk_projections,
        bva,
        true,
    )?;
    let causal_qk_right_tbv = causal_qk_routing.route_terminal(
        &causal_qk_right_equation,
        &causal_qk_projections,
        bva,
        false,
    )?;

    let causal_pv_routing = manifest.causal_pv_routing(metadata, layout, bva)?;
    let causal_pv_projections =
        causal_pv_routing.projections::<Fr>(|| transcript.challenge(b"causal-pv-routing"))?;
    let causal_pv_output_tbv = causal_pv_routing.route_output(&causal_pv_projections, bva)?;
    let causal_pv_verification = messages.causal_pv_proof.verify(
        causal_pv_routing.layout.logical_len(),
        messages.causal_pv_claim,
    );
    if !causal_pv_verification.accepted {
        return Ok(None);
    }
    let causal_pv_left_equation = TensorBasisVector::global_tensor(
        &causal_pv_routing.layout,
        &causal_pv_verification.challenges_inv,
    )?;
    let causal_pv_right_equation = TensorBasisVector::global_tensor(
        &causal_pv_routing.layout,
        &causal_pv_verification.challenges,
    )?;
    let causal_pv_left_tbv = causal_pv_routing.route_terminal(
        &causal_pv_left_equation,
        &causal_pv_projections,
        bva,
        true,
    )?;
    let causal_pv_right_tbv = causal_pv_routing.route_terminal(
        &causal_pv_right_equation,
        &causal_pv_projections,
        bva,
        false,
    )?;

    transcript.absorb(b"io-claim", &io_claim);
    transcript.absorb(b"linear-claim", &linear_claim);
    transcript.absorb(b"range-claim", &range_claim);
    transcript.absorb(b"lookup-left-product", &messages.lookup_left_product);
    transcript.absorb(b"lookup-right-product", &messages.lookup_right_product);
    transcript.absorb(b"lookup-leaf-claim", &messages.lookup_claim);
    absorb_global_witness_pair(
        &mut transcript,
        b"elementwise-bilinear-proof",
        messages.elementwise_claim,
        messages.elementwise_proof,
    );
    absorb_global_witness_pair(
        &mut transcript,
        b"dense-matmul-proof",
        messages.dense_matmul_claim,
        messages.dense_matmul_proof,
    );
    absorb_global_witness_pair(
        &mut transcript,
        b"causal-qk-proof",
        messages.causal_qk_claim,
        messages.causal_qk_proof,
    );
    absorb_global_witness_pair(
        &mut transcript,
        b"causal-pv-proof",
        messages.causal_pv_claim,
        messages.causal_pv_proof,
    );
    let weights: [Fr; 16] =
        std::array::from_fn(|_| transcript.challenge(b"synchronized-witness-opening-weight"));
    let synchronized_tbv = TensorBasisVector::linear_combination_owned(
        &bva.layout,
        vec![
            (weights[0], io_tbv),
            (weights[1], linear_tbv),
            (weights[2], range_tbv),
            (weights[3], elementwise_output_tbv),
            (weights[4], elementwise_left_tbv),
            (weights[5], elementwise_right_tbv),
            (weights[6], dense_matmul_output_tbv),
            (weights[7], dense_matmul_left_tbv),
            (weights[8], dense_matmul_right_tbv),
            (weights[9], causal_qk_output_tbv),
            (weights[10], causal_qk_left_tbv),
            (weights[11], causal_qk_right_tbv),
            (weights[12], causal_pv_output_tbv),
            (weights[13], causal_pv_left_tbv),
            (weights[14], causal_pv_right_tbv),
            (weights[15], lookup_tbv),
        ],
    )?;
    let synchronized_claim = weights[0] * io_claim
        + weights[1] * linear_claim
        + weights[2] * range_claim
        + weights[3] * messages.elementwise_claim
        + weights[4] * messages.elementwise_proof.a_reduced
        + weights[5] * elementwise_right_claim
        + weights[6] * messages.dense_matmul_claim
        + weights[7] * messages.dense_matmul_proof.a_reduced
        + weights[8] * messages.dense_matmul_proof.b_reduced
        + weights[9] * messages.causal_qk_claim
        + weights[10] * messages.causal_qk_proof.a_reduced
        + weights[11] * messages.causal_qk_proof.b_reduced
        + weights[12] * messages.causal_pv_claim
        + weights[13] * messages.causal_pv_proof.a_reduced
        + weights[14] * messages.causal_pv_proof.b_reduced
        + weights[15] * messages.lookup_claim;
    Ok(Some(GlobalRelationReduction {
        synchronized_tbv,
        synchronized_claim,
        query_tbv,
        lookup_compressed_projection: messages.lookup_claim + lookup_public_offset,
    }))
}

#[cfg(test)]
mod tests {
    use std::fs::File;
    use std::io::BufReader;
    use std::io::Cursor;
    use std::time::Instant;

    use super::*;
    use crate::data_structures::{ZkSRS, SRS};
    use crate::hdr1cs::gpt2::{
        Gpt2Model, Gpt2ProofTranscript, Gpt2PublicStatement, Gpt2Q12Checkpoint,
    };
    use crate::protocols::bucketed_ip::{
        BucketedVector, EoLiteBulletProof, EoWitnessPairProof, TbvBucket, TensorBasisVector,
    };
    use crate::protocols::streaming_grand_product::BvaGrandProductProof;
    use crate::protocols::vector_relation::{
        BvaAffineRelationProof, BvaCommittedEvaluationProof, BvaRangeProof, BvaRoutedEqualityProof,
        SparseBvaUpperBoundProof,
    };
    use crate::protocols::virtualpcs::{
        bva_eo_opening_point, commit_bva_field_streaming, commit_sparse_bva_field_streaming,
        inner_product_bva_field_streaming, open_bva_field_streaming,
        open_sparse_bva_field_streaming,
    };
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_ec::pairing::Pairing;
    use ark_poly_commit::smart_pc::SmartPC;
    use ark_std::ops::Mul;

    struct TamperedBvaFieldSource<'a, S> {
        inner: &'a mut S,
        vector_index: usize,
    }

    impl<F, S> BvaFieldSource<F> for TamperedBvaFieldSource<'_, S>
    where
        F: PrimeField,
        S: BvaFieldSource<F>,
    {
        type Error = S::Error;

        fn layout(&self) -> &BvaLayout {
            self.inner.layout()
        }

        fn for_each_vector(
            &mut self,
            visit: &mut dyn FnMut(usize, &[F]),
        ) -> Result<(), Self::Error> {
            let target = self.vector_index;
            self.inner.for_each_vector(&mut |vector_index, values| {
                if vector_index == target {
                    let mut tampered = values.to_vec();
                    tampered[0] += F::from(1_u64);
                    visit(vector_index, &tampered);
                } else {
                    visit(vector_index, values);
                }
            })
        }
    }

    fn absorb_witness_pair_proof(
        transcript: &mut Gpt2ProofTranscript,
        label: &[u8],
        claimed: Fr,
        proof: &EoWitnessPairProof<Fr>,
    ) {
        transcript.absorb(label, &claimed);
        for (left_cross, right_cross) in &proof.rounds {
            transcript.absorb(b"witness-pair-left-cross", left_cross);
            transcript.absorb(b"witness-pair-right-cross", right_cross);
        }
        transcript.absorb(b"witness-pair-left-terminal", &proof.a_reduced);
        transcript.absorb(b"witness-pair-right-terminal", &proof.b_reduced);
    }

    #[test]
    fn compiler_allocates_one_shared_namespace_without_unconstrained_auxiliary_blocks() {
        let model = Gpt2Model {
            blocks: 1,
            hidden_width: 2,
            attention_heads: 1,
            mlp_width: 4,
            vocabulary: 3,
            maximum_positions: 2,
        };
        let metadata = Gpt2Q12Metadata {
            format_version: 2,
            model,
            fractional_bits: 12,
            parameter_entries: model.physical_parameter_entries().unwrap() as u64,
            token_count: 2,
            checkpoint_sha256: [0x24; 32],
            tokenizer_sha256: [0x84; 32],
        };
        let layout = Gpt2GlobalLayout::compile(&metadata).unwrap();
        let manifest = Gpt2PrimitiveRelationManifest::compile(&metadata, &layout).unwrap();
        let shape = manifest.shape(&layout).unwrap();

        assert!(manifest.audit(&layout).is_ok());
        assert_eq!(shape.top_level_equation_count(), 4);
        assert!(shape.maximum_streamed_range_limb_column_entries > 0);
        let base_range_entries = layout
            .physical_entries(&metadata)
            .unwrap()
            .checked_sub(metadata.token_count)
            .unwrap()
            * 4;
        assert!(shape.streamed_range_limb_entries >= base_range_entries);
        assert!(!manifest.range_limb_blocks.iter().any(|block| {
            matches!(
                block.source_kind,
                RangeLimbSource::Base { block_index }
                    if matches!(
                        layout.blocks()[block_index].role,
                        super::super::Gpt2GlobalBlockRole::PublicTokenIds
                    )
            )
        }));

        let mut missing_recomposition = manifest.clone();
        let position = missing_recomposition
            .relations
            .iter()
            .position(|relation| {
                relation.family == RelationFamily::Linear
                    && relation.kind == LinearRelationKind::RangeRecomposition as u8
            })
            .unwrap();
        missing_recomposition.relations.remove(position);
        assert!(missing_recomposition.audit(&layout).is_err());
    }

    #[test]
    fn equation_schedule_has_four_global_groups_and_virtual_padding_only() {
        let checkpoint = tiny_checkpoint();
        let layout = Gpt2GlobalLayout::compile(checkpoint.metadata()).unwrap();
        let manifest =
            Gpt2PrimitiveRelationManifest::compile(checkpoint.metadata(), &layout).unwrap();

        let schedule = manifest.equation_schedule(&layout).unwrap();

        assert_eq!(schedule.groups.len(), 4);
        assert_eq!(
            schedule
                .groups
                .iter()
                .map(|group| group.family)
                .collect::<Vec<_>>(),
            vec![
                TopLevelEquationFamily::Io,
                TopLevelEquationFamily::Linear,
                TopLevelEquationFamily::Bilinear,
                TopLevelEquationFamily::TaggedLookup,
            ]
        );
        assert!(schedule
            .groups
            .iter()
            .all(|group| group.layout.materialized_padding_entries() == 0));
        let maximum_witness_column = layout
            .blocks()
            .iter()
            .map(|block| block.rows)
            .chain(manifest.auxiliary_blocks.iter().map(|block| block.rows))
            .chain(manifest.range_limb_blocks.iter().map(|block| block.rows))
            .max()
            .unwrap();
        assert!(schedule.groups.iter().all(|group| {
            group
                .layout
                .vector_lengths()
                .iter()
                .all(|rows| *rows <= maximum_witness_column)
        }));
        let mut covered_relations = vec![false; manifest.relations.len()];
        for source in schedule
            .groups
            .iter()
            .skip(1)
            .flat_map(|group| &group.sources)
        {
            if let EquationRowSource::PrimitiveRelationChunk { relation_index, .. } = source {
                covered_relations[*relation_index] = true;
            }
        }
        assert!(covered_relations.into_iter().all(|covered| covered));
        assert!(schedule.audit(&layout, &manifest).is_ok());

        let mut missing_relation = schedule.clone();
        missing_relation.groups[1].sources.pop();
        assert!(missing_relation.audit(&layout, &manifest).is_err());
    }

    #[test]
    fn range_limb_stream_biases_signed_values_and_keeps_one_column_live() {
        let descriptor = AuxiliaryBlockDescriptor {
            id: GlobalBlockId(9),
            operation_index: 0,
            kind: AuxiliaryBlockKind::Centered,
            rows: 4,
            cols: 1,
            value_bits: 16,
        };
        let block = AuxiliaryWitnessBlock {
            id: descriptor.id,
            columns: vec![vec![-32_768, -1, 0, 32_767]],
        };
        let stream = RangeLimbStream::new(&descriptor, &block).unwrap();
        let mut emitted = Vec::new();
        let profile = stream
            .for_each_limb_column(&mut |column, limb, bits, values| {
                emitted.push((column, limb, bits, values.to_vec()));
            })
            .unwrap();

        assert_eq!(
            emitted,
            vec![
                (0, 0, 8, vec![0, 255, 0, 255]),
                (0, 1, 8, vec![0, 127, 128, 255]),
            ]
        );
        assert_eq!(profile.source_entries, 4);
        assert_eq!(profile.emitted_limb_entries, 8);
        assert_eq!(profile.maximum_live_limb_entries, 4);

        let encoding = RangeEncoding::new(true, 16).unwrap();
        assert!(encoding.matches(-1, &[255, 127]).unwrap());
        assert!(!encoding.matches(-1, &[254, 127]).unwrap());
    }

    #[test]
    fn range_encoding_uses_a_narrow_final_tag_for_non_byte_widths() {
        let encoding = RangeEncoding::new(false, 12).unwrap();
        let mut limbs = Vec::new();
        encoding.encode(4_095, &mut limbs).unwrap();

        assert_eq!(limbs, vec![255, 15]);
        assert_eq!(encoding.limb_bits(0), Some(8));
        assert_eq!(encoding.limb_bits(1), Some(4));
        assert_eq!(encoding.limb_bits(2), None);
        assert!(encoding.matches(4_095, &limbs).unwrap());
        assert!(encoding.encode(4_096, &mut limbs).is_err());
    }

    #[test]
    fn nonlinear_lookup_relations_use_explicit_finite_domain_clamp_blocks() {
        let checkpoint = tiny_checkpoint();
        let layout = Gpt2GlobalLayout::compile(checkpoint.metadata()).unwrap();
        let manifest =
            Gpt2PrimitiveRelationManifest::compile(checkpoint.metadata(), &layout).unwrap();
        let mut seen = [false; 3];

        for relation in manifest.relations.iter().filter(|relation| {
            relation.family == RelationFamily::TaggedLookup
                && matches!(
                    relation.kind,
                    value if value == TaggedLookupRelationKind::Gelu as u8
                        || value == TaggedLookupRelationKind::Exp as u8
                        || value == TaggedLookupRelationKind::InverseSqrt as u8
                )
        }) {
            let input = relation.blocks[0];
            let descriptor = manifest
                .auxiliary_blocks
                .get(input.0 - manifest.base_block_count)
                .unwrap();
            match relation.kind {
                value if value == TaggedLookupRelationKind::Gelu as u8 => {
                    assert_eq!(descriptor.kind, AuxiliaryBlockKind::ClampedSigned);
                    seen[0] = true;
                }
                value if value == TaggedLookupRelationKind::Exp as u8 => {
                    assert_eq!(descriptor.kind, AuxiliaryBlockKind::ClampedUnsigned);
                    seen[1] = true;
                }
                value if value == TaggedLookupRelationKind::InverseSqrt as u8 => {
                    assert_eq!(descriptor.kind, AuxiliaryBlockKind::ClampedUnsigned);
                    seen[2] = true;
                }
                _ => unreachable!(),
            }
        }
        assert!(seen.iter().copied().all(|value| value));

        let scale = i64::from(checkpoint.metadata().scale());
        let outside = 8 * scale + 1;
        let invalid_tail_query = TaggedLookupQuery {
            tag: TaggedLookupRelationKind::Gelu,
            input: i128::from(outside),
            output: i128::from(q12_gelu(outside, scale).unwrap()),
        };
        assert!(Gpt2LookupRegistry::exact(checkpoint.metadata())
            .audit_query(invalid_tail_query)
            .is_err());
        let reciprocal_tail = i64::try_from(checkpoint.metadata().token_count).unwrap() * scale + 1;
        let invalid_reciprocal_query = TaggedLookupQuery {
            tag: TaggedLookupRelationKind::Reciprocal,
            input: i128::from(reciprocal_tail),
            output: i128::from(
                q12_reciprocal(reciprocal_tail, checkpoint.metadata().token_count, scale).unwrap(),
            ),
        };
        assert!(Gpt2LookupRegistry::exact(checkpoint.metadata())
            .audit_query(invalid_reciprocal_query)
            .is_err());
    }

    #[test]
    fn clamp_witness_has_one_hot_regions_and_nonnegative_branch_slacks() {
        let witness = clamp_witness_columns(
            &[vec![-40_000, -32_768, 0, 32_768, 40_000]],
            -32_768,
            32_768,
        )
        .unwrap();

        assert_eq!(
            witness.clamped[0],
            vec![-32_768, -32_768, 0, 32_768, 32_768]
        );
        assert_eq!(witness.low_selector[0], vec![1, 0, 0, 0, 0]);
        assert_eq!(witness.middle_selector[0], vec![0, 1, 1, 1, 0]);
        assert_eq!(witness.high_selector[0], vec![0, 0, 0, 0, 1]);
        assert_eq!(witness.low_slack[0], vec![7_232, 0, 0, 0, 0]);
        assert_eq!(witness.middle_low_slack[0], vec![0, 0, 32_768, 65_536, 0]);
        assert_eq!(witness.middle_high_slack[0], vec![0, 65_536, 32_768, 0, 0]);
        assert_eq!(witness.high_slack[0], vec![0, 0, 0, 0, 7_232]);
    }

    #[test]
    fn global_relation_proof_uses_one_statement_bound_witness_commitment() {
        let mut checkpoint = tiny_checkpoint();
        let statement = Gpt2PublicStatement::new(
            checkpoint.metadata(),
            checkpoint.token_ids(),
            checkpoint.reference_logits(),
        )
        .unwrap();
        let parameters = Gpt2GlobalRelationParameters::setup(&statement).unwrap();

        let proof =
            Gpt2GlobalRelationProof::prove(&parameters, &statement, &mut checkpoint).unwrap();

        assert!(proof.verify(&parameters, &statement).unwrap());
        assert_eq!(proof.witness_commitment_count(), 1);
        assert_eq!(proof.witness_opening_count(), 1);

        let mut changed_statement = statement.clone();
        changed_statement.public_logits[0] += 1;
        assert!(!proof.verify(&parameters, &changed_statement).unwrap());

        let mut changed_commitment = proof.clone();
        changed_commitment.global_witness_commitment += Bls12_381::pairing(
            parameters.pc_pp.g_0.mul(Fr::from(1_u64)),
            parameters.pc_pp.h_hat,
        );
        assert!(!changed_commitment.verify(&parameters, &statement).unwrap());
    }

    #[test]
    fn global_relation_proof_codec_is_exact_and_statement_bound() {
        let mut checkpoint = tiny_checkpoint();
        let statement = Gpt2PublicStatement::new(
            checkpoint.metadata(),
            checkpoint.token_ids(),
            checkpoint.reference_logits(),
        )
        .unwrap();
        let parameters = Gpt2GlobalRelationParameters::setup(&statement).unwrap();
        let proof =
            Gpt2GlobalRelationProof::prove(&parameters, &statement, &mut checkpoint).unwrap();
        let bytes = proof.to_bytes_exact().unwrap();
        let decoded = Gpt2GlobalRelationProof::from_bytes_exact(&bytes).unwrap();
        assert!(decoded.verify(&parameters, &statement).unwrap());

        assert!(Gpt2GlobalRelationProof::from_bytes_exact(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(Gpt2GlobalRelationProof::from_bytes_exact(&trailing).is_err());

        let mut changed_digest = decoded;
        changed_digest.statement_digest[0] ^= 1;
        assert!(!changed_digest.verify(&parameters, &statement).unwrap());
    }

    #[test]
    fn lasso_core_proof_verifies_both_products_affine_factors_and_timestamps() {
        let mut checkpoint = tiny_checkpoint();
        let statement = Gpt2PublicStatement::new(
            checkpoint.metadata(),
            checkpoint.token_ids(),
            checkpoint.reference_logits(),
        )
        .unwrap();
        let parameters = Gpt2GlobalRelationParameters::setup(&statement).unwrap();
        let global =
            Gpt2GlobalRelationProof::prove(&parameters, &statement, &mut checkpoint).unwrap();
        let mut replay = tiny_checkpoint();
        let mut route_replay = tiny_checkpoint();
        let mut lasso = Gpt2LassoProof::prove_core(
            &parameters,
            &statement,
            global.global_witness_commitment,
            &mut replay,
            &mut route_replay,
            64,
        )
        .unwrap();

        assert!(lasso
            .verify_core(&parameters, &statement, global.global_witness_commitment,)
            .unwrap());
        lasso.expected_product += Fr::from(1_u64);
        assert!(!lasso
            .verify_core(&parameters, &statement, global.global_witness_commitment,)
            .unwrap());
    }

    #[test]
    fn end_to_end_public_support_proof_binds_lasso_queries_to_global_cw() {
        let mut global_checkpoint = tiny_checkpoint();
        let statement = Gpt2PublicStatement::new(
            global_checkpoint.metadata(),
            global_checkpoint.token_ids(),
            global_checkpoint.reference_logits(),
        )
        .unwrap();
        let parameters = Gpt2GlobalRelationParameters::setup(&statement).unwrap();
        let mut lasso_checkpoint = tiny_checkpoint();
        let mut route_replay = tiny_checkpoint();
        let mut proof = Gpt2Proof::prove_with_replays(
            &parameters,
            &statement,
            &mut global_checkpoint,
            &mut lasso_checkpoint,
            &mut route_replay,
            64,
        )
        .unwrap();

        assert!(proof.verify(&parameters, &statement).unwrap());
        assert_eq!(proof.witness_commitment_count(), 1);
        assert_eq!(proof.witness_opening_count(), 1);

        let mut changed_statement = statement.clone();
        changed_statement.public_logits[0] += 1;
        assert!(!proof.verify(&parameters, &changed_statement).unwrap());

        let commitment_delta = Bls12_381::pairing(
            parameters.pc_pp.g_0.mul(Fr::from(1_u64)),
            parameters.pc_pp.h_hat,
        );
        proof.lasso.query_value_commitment += commitment_delta;
        assert!(!proof.verify(&parameters, &statement).unwrap());
        proof.lasso.query_value_commitment -= commitment_delta;

        proof.global_relation.lookup_claim += Fr::from(1_u64);
        assert!(!proof.verify(&parameters, &statement).unwrap());
    }

    #[test]
    fn scalable_public_support_proof_uses_one_global_cw_and_rejects_factor_tamper() {
        let mut global_checkpoint = tiny_checkpoint();
        let statement = Gpt2PublicStatement::new(
            global_checkpoint.metadata(),
            global_checkpoint.token_ids(),
            global_checkpoint.reference_logits(),
        )
        .unwrap();
        let parameters = Gpt2GlobalRelationParameters::setup(&statement).unwrap();
        let mut lookup_checkpoint = tiny_checkpoint();
        let mut proof = Gpt2ScalableProof::prove_with_replays(
            &parameters,
            &statement,
            &mut global_checkpoint,
            &mut lookup_checkpoint,
            64,
        )
        .unwrap();

        assert!(proof.verify(&parameters, &statement).unwrap());
        let bytes = proof.to_bytes_exact().unwrap();
        assert!(Gpt2ScalableProof::verify_bytes(&parameters, &statement, &bytes).unwrap());
        assert!(Gpt2ScalableProof::from_bytes_exact(&bytes[..bytes.len() - 1]).is_err());
        let mut trailing = bytes;
        trailing.push(0);
        assert!(Gpt2ScalableProof::from_bytes_exact(&trailing).is_err());
        assert_eq!(proof.witness_commitment_count(), 1);
        assert_eq!(proof.witness_opening_count(), 1);
        proof.lookup.expected_product += Fr::from(1_u64);
        assert!(!proof.verify(&parameters, &statement).unwrap());
    }

    #[test]
    fn end_to_end_proof_bytes_reject_truncation_and_trailing_data() {
        let mut global_checkpoint = tiny_checkpoint();
        let statement = Gpt2PublicStatement::new(
            global_checkpoint.metadata(),
            global_checkpoint.token_ids(),
            global_checkpoint.reference_logits(),
        )
        .unwrap();
        let parameters = Gpt2GlobalRelationParameters::setup(&statement).unwrap();
        let mut lasso_checkpoint = tiny_checkpoint();
        let mut route_replay = tiny_checkpoint();
        let prove_started = Instant::now();
        let proof = Gpt2Proof::prove_with_replays(
            &parameters,
            &statement,
            &mut global_checkpoint,
            &mut lasso_checkpoint,
            &mut route_replay,
            64,
        )
        .unwrap();
        let prove_seconds = prove_started.elapsed().as_secs_f64();
        let bytes = proof.to_bytes_exact().unwrap();
        let verify_started = Instant::now();
        assert!(Gpt2Proof::verify_bytes(&parameters, &statement, &bytes).unwrap());
        let verify_seconds = verify_started.elapsed().as_secs_f64();
        println!("experiment=gpt2_end_to_end_public_support_tiny");
        println!("proof_mode=public_support_non_zk");
        println!("query_count={}", proof.lasso.query_count);
        println!(
            "lookup_support_entries={}",
            proof.lasso.public_table_binding.multiplicities.len()
        );
        println!("proof_size_bytes={}", bytes.len());
        println!("prove_seconds={prove_seconds:.6}");
        println!("verify_seconds={verify_seconds:.6}");
        println!("global_witness_commitment_count=1");
        println!("global_witness_opening_count=1");
        println!("statement_only_verify=true");
        println!("materialized_padding_entries=0");
        let mut statement_digest_tamper = bytes.clone();
        statement_digest_tamper[12] ^= 1;
        assert!(
            !Gpt2Proof::verify_bytes(&parameters, &statement, &statement_digest_tamper,).unwrap()
        );
        assert!(
            !Gpt2Proof::verify_bytes(&parameters, &statement, &bytes[..bytes.len() - 1],).unwrap()
        );
        let mut trailing = bytes;
        trailing.push(0);
        assert!(!Gpt2Proof::verify_bytes(&parameters, &statement, &trailing).unwrap());
    }

    #[test]
    fn end_to_end_artifacts_round_trip_into_a_statement_only_verifier() {
        let mut global_checkpoint = tiny_checkpoint();
        let statement = Gpt2PublicStatement::new(
            global_checkpoint.metadata(),
            global_checkpoint.token_ids(),
            global_checkpoint.reference_logits(),
        )
        .unwrap();
        let parameters = Gpt2GlobalRelationParameters::setup(&statement).unwrap();
        let mut lasso_checkpoint = tiny_checkpoint();
        let mut route_replay = tiny_checkpoint();
        let proof = Gpt2Proof::prove_with_replays(
            &parameters,
            &statement,
            &mut global_checkpoint,
            &mut lasso_checkpoint,
            &mut route_replay,
            64,
        )
        .unwrap();

        let statement_bytes = statement.to_bytes_exact().unwrap();
        let parameter_bytes = parameters.to_bytes_exact().unwrap();
        let proof_bytes = proof.to_bytes_exact().unwrap();
        let decoded_statement = Gpt2PublicStatement::from_bytes_exact(&statement_bytes).unwrap();
        let decoded_parameters =
            Gpt2GlobalRelationParameters::from_bytes_exact(&parameter_bytes).unwrap();

        assert!(
            Gpt2Proof::verify_bytes(&decoded_parameters, &decoded_statement, &proof_bytes,)
                .unwrap()
        );
        assert!(Gpt2PublicStatement::from_bytes_exact(
            &statement_bytes[..statement_bytes.len() - 1]
        )
        .is_err());
        assert!(Gpt2GlobalRelationParameters::from_bytes_exact(
            &parameter_bytes[..parameter_bytes.len() - 1]
        )
        .is_err());
        let mut statement_trailing = statement_bytes;
        statement_trailing.push(0);
        assert!(Gpt2PublicStatement::from_bytes_exact(&statement_trailing).is_err());
        let mut parameter_trailing = parameter_bytes;
        parameter_trailing.push(0);
        assert!(Gpt2GlobalRelationParameters::from_bytes_exact(&parameter_trailing).is_err());
    }

    #[test]
    fn auxiliary_witness_satisfies_the_compiled_tiny_gpt2_relation() {
        let mut checkpoint = tiny_checkpoint();
        let token_ids = checkpoint.token_ids().to_vec();
        let public_logits = checkpoint.reference_logits().to_vec();
        let metadata = checkpoint.metadata().clone();
        let statement = Gpt2PublicStatement::new(&metadata, &token_ids, &public_logits).unwrap();
        let layout = Gpt2GlobalLayout::compile(&metadata).unwrap();
        let (_, activations) = checkpoint.execute_activation_witness(&layout).unwrap();
        let manifest =
            Gpt2PrimitiveRelationManifest::compile(checkpoint.metadata(), &layout).unwrap();
        let shape = manifest.shape(&layout).unwrap();

        let witness =
            Gpt2PrimitiveWitness::build(&mut checkpoint, &layout, &activations, &manifest).unwrap();

        assert_eq!(witness.block_count(), shape.resident_auxiliary_blocks);
        assert_eq!(witness.stored_entries(), shape.resident_auxiliary_entries);
        assert_eq!(
            shape.auxiliary_blocks,
            shape.resident_auxiliary_blocks + shape.streamed_range_limb_blocks
        );
        assert_eq!(
            shape.auxiliary_entries,
            shape.resident_auxiliary_entries + shape.streamed_range_limb_entries
        );
        assert!(shape.streamed_range_limb_blocks > 0);
        assert!(shape.streamed_range_limb_entries > 0);
        assert!(witness.audit(&manifest).is_ok());
        assert!(witness.audit_declared_ranges(&manifest).is_ok());
        let mut out_of_range = witness.clone();
        let remainder_index = manifest
            .auxiliary_blocks
            .iter()
            .position(|block| block.kind == AuxiliaryBlockKind::Remainder)
            .unwrap();
        let remainder_bits = manifest.auxiliary_blocks[remainder_index].value_bits;
        out_of_range.blocks[remainder_index].columns[0][0] = 1_i128 << remainder_bits;
        assert!(out_of_range.audit_declared_ranges(&manifest).is_err());
        let registry = Gpt2LookupRegistry::exact(checkpoint.metadata());
        let queries = witness
            .lookup_queries(&layout, &activations, &manifest)
            .unwrap();
        assert!(registry.audit(&queries).is_ok());
        assert_eq!(queries.len(), manifest.lookup_scalar_count().unwrap());
        assert_eq!(registry.public_table_rows().unwrap(), 33_694_208);
        for query in &queries {
            let table_index = registry.table_index(*query).unwrap();
            assert_eq!(registry.table_row(table_index).unwrap(), *query);
        }
        let last_range_row = TaggedLookupQuery {
            tag: TaggedLookupRelationKind::Range8,
            input: 255,
            output: 0,
        };
        let last_table_index = registry.table_index(last_range_row).unwrap();
        assert_eq!(last_table_index + 1, registry.public_table_rows().unwrap());
        assert_eq!(
            registry.table_row(last_table_index).unwrap(),
            last_range_row
        );
        assert!(registry.table_row(last_table_index + 1).is_err());
        let mut invalid_queries = queries.clone();
        invalid_queries[0].output += 1;
        assert!(registry.audit(&invalid_queries).is_err());
        let bva = manifest.compile_bva(&layout).unwrap();
        assert!(witness
            .satisfies(&mut checkpoint, &layout, &activations, &manifest)
            .unwrap());

        let mut tampered = witness.clone();
        tampered.blocks[0].columns[0][0] += 1;
        assert!(!tampered
            .satisfies(&mut checkpoint, &layout, &activations, &manifest)
            .unwrap());

        assert_eq!(
            bva.physical_entries(),
            layout.physical_entries(checkpoint.metadata()).unwrap() + shape.auxiliary_entries
        );
        assert_eq!(bva.materialized_padding_entries(), 0);
        let mut source = Gpt2PrimitiveBvaFieldSource::<_, Fr>::new(
            &mut checkpoint,
            &layout,
            &activations,
            &manifest,
            &witness,
            &bva,
        )
        .unwrap();
        let bva_log = bva.logical_entries().ilog2() as usize;
        let lookup_log = registry
            .public_table_rows()
            .unwrap()
            .next_power_of_two()
            .ilog2() as usize;
        let mut rng = ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup((bva_log.max(lookup_log) + 1) / 2, &mut rng).unwrap();
        let matrix_hiding = Fr::from(73_u64);
        let (matrix_commitment, tier_one, profile) =
            commit_bva_field_streaming::<Bls12_381, _>(&pp, &mut source, matrix_hiding).unwrap();
        assert_eq!(profile.physical_entries, bva.physical_entries());
        assert_eq!(profile.materialized_padding_entries, 0);

        let mut proof_transcript = Gpt2ProofTranscript::new(&statement);
        proof_transcript.absorb(b"global-witness-commitment", &matrix_commitment);
        let lookup_row_challenge = proof_transcript.challenge::<Fr>(b"lookup-row-compression");
        // The first replay only fixes the query inventory and sparse
        // multiplicities.  The actual product challenge is sampled below,
        // after every value/counter component commitment has entered the
        // Fiat--Shamir transcript.
        let mut lookup_product_challenge = Fr::from(1_u64);
        drop(source);
        let mut stream =
            Gpt2LookupQueryStream::new(&mut checkpoint, &layout, &activations, &witness, &manifest)
                .unwrap();
        let mut products = stream
            .lasso_products(&registry, lookup_row_challenge, lookup_product_challenge)
            .unwrap();
        assert_eq!(products.left, products.right);
        assert_eq!(
            products.query_count,
            manifest.lookup_scalar_count().unwrap() + shape.streamed_range_limb_entries
        );
        assert!(products.distinct_table_rows < products.query_count);
        assert!(products.maximum_live_semantic_entries < products.query_count);
        assert_eq!(
            products.public_table_rows,
            registry.public_table_rows().unwrap()
        );
        let mut out_of_range_multiplicity = products.clone();
        out_of_range_multiplicity.multiplicities[0].1 = products.query_count as u64 + 1;
        assert!(Gpt2LassoMultiplicitySource::<Fr>::new(&out_of_range_multiplicity).is_err());
        let mut multiplicity_source = Gpt2LassoMultiplicitySource::new(&products).unwrap();
        let multiplicity_hiding = Fr::from(101_u64);
        let (multiplicity_commitment, multiplicity_tier_one, multiplicity_commit_profile) =
            commit_sparse_bva_field_streaming::<Bls12_381, _>(
                &pp,
                &mut multiplicity_source,
                multiplicity_hiding,
            )
            .unwrap();
        assert_eq!(
            multiplicity_commit_profile.nonzero_entries,
            products.distinct_table_rows
        );
        assert_eq!(multiplicity_commit_profile.materialized_padding_entries, 0);
        assert_eq!(multiplicity_commit_profile.padding_group_operations, 0);
        println!(
            "lookup_multiplicity_nonzero_entries={}",
            multiplicity_commit_profile.nonzero_entries
        );
        println!(
            "lookup_multiplicity_materialized_padding_entries={}",
            multiplicity_commit_profile.materialized_padding_entries
        );
        proof_transcript.absorb(b"lookup-multiplicity-commitment", &multiplicity_commitment);
        let multiplicity_log = multiplicity_source.layout.logical_len().ilog2() as usize;
        let multiplicity_col_log = multiplicity_log / 2;
        let multiplicity_row_log = multiplicity_log - multiplicity_col_log;
        let multiplicity_xl = (0..multiplicity_row_log)
            .map(|_| proof_transcript.challenge(b"lookup-multiplicity-row-point"))
            .collect::<Vec<Fr>>();
        let multiplicity_xr = (0..multiplicity_col_log)
            .map(|_| proof_transcript.challenge(b"lookup-multiplicity-column-point"))
            .collect::<Vec<Fr>>();
        let multiplicity_value = multiplicity_source
            .evaluate_packed(&multiplicity_xl, &multiplicity_xr)
            .unwrap();
        let multiplicity_value_hiding = Fr::from(103_u64);
        let multiplicity_value_commitment =
            Bls12_381::pairing(pp.g_0.mul(multiplicity_value), pp.h_hat)
                + pp.tilde_u.mul(multiplicity_value_hiding);
        let (multiplicity_opening, multiplicity_open_profile) =
            open_sparse_bva_field_streaming::<Bls12_381, _>(
                &pp,
                &mut multiplicity_source,
                &multiplicity_xl,
                &multiplicity_xr,
                multiplicity_value_commitment,
                multiplicity_commitment,
                &multiplicity_tier_one,
                multiplicity_value_hiding,
                multiplicity_hiding,
            )
            .unwrap();
        assert!(SmartPC::<Bls12_381>::verify(
            &pp,
            multiplicity_commitment,
            multiplicity_value_commitment,
            &multiplicity_xl,
            &multiplicity_xr,
            &multiplicity_opening,
        )
        .unwrap());
        assert_eq!(
            multiplicity_open_profile.materialized_projection_padding_entries,
            0
        );
        assert_eq!(
            multiplicity_open_profile.padding_projection_field_multiplications,
            0
        );
        proof_transcript.absorb(
            b"lookup-multiplicity-value-commitment",
            &multiplicity_value_commitment,
        );
        proof_transcript.absorb(b"lookup-multiplicity-opening", &multiplicity_opening);
        if std::env::var_os("ZKSMART_GPT2_MULTIPLICITY_RANGE_PROOF").is_some() {
            let multiplicity_range_srs = SRS::<Bls12_381> {
                zksrs: ZkSRS::new(pp.g_0, pp.tilde_g),
                pc_pp: pp.clone(),
                h_hat: pp.h_hat,
            };
            let maximum_count = products.query_count as u64;
            let prove_started = Instant::now();
            let (multiplicity_range_proof, multiplicity_range_profile) =
                SparseBvaUpperBoundProof::prove(
                    &multiplicity_range_srs,
                    0,
                    0x4d55_4c54,
                    &mut multiplicity_source,
                    multiplicity_commitment,
                    &multiplicity_tier_one,
                    multiplicity_hiding,
                    maximum_count,
                )
                .unwrap();
            let prove_seconds = prove_started.elapsed().as_secs_f64();
            let verify_started = Instant::now();
            assert!(multiplicity_range_proof.verify(
                &multiplicity_range_srs,
                0,
                0x4d55_4c54,
                &multiplicity_source.layout,
                multiplicity_commitment,
                maximum_count,
            ));
            let verify_seconds = verify_started.elapsed().as_secs_f64();
            let commitment_delta = Bls12_381::pairing(
                multiplicity_range_srs.pc_pp.g_0,
                multiplicity_range_srs.h_hat,
            );
            assert!(!multiplicity_range_proof.verify(
                &multiplicity_range_srs,
                0,
                0x4d55_4c54,
                &multiplicity_source.layout,
                multiplicity_commitment + commitment_delta,
                maximum_count,
            ));
            assert!(!multiplicity_range_proof.verify(
                &multiplicity_range_srs,
                0,
                0x4d55_4c54,
                &multiplicity_source.layout,
                multiplicity_commitment,
                maximum_count - 1,
            ));
            println!("lookup_multiplicity_range_maximum_count={maximum_count}");
            println!(
                "lookup_multiplicity_range_maximum_live_sparse_entries={}",
                multiplicity_range_profile.maximum_live_vector_entries
            );
            println!(
                "lookup_multiplicity_range_maximum_live_projection_entries={}",
                multiplicity_range_profile.maximum_live_projection_entries
            );
            println!(
                "lookup_multiplicity_range_source_passes={}",
                multiplicity_range_profile.source_passes
            );
            println!(
                "lookup_multiplicity_range_materialized_padding_entries={}",
                multiplicity_range_profile.materialized_padding_entries
            );
            println!(
                "lookup_multiplicity_range_padding_field_operations={}",
                multiplicity_range_profile.padding_field_operations
            );
            println!(
                "lookup_multiplicity_range_padding_group_operations={}",
                multiplicity_range_profile.padding_group_operations
            );
            println!("lookup_multiplicity_range_prove_seconds={prove_seconds:.6}");
            println!("lookup_multiplicity_range_verify_seconds={verify_seconds:.6}");
            println!(
                "lookup_multiplicity_range_proof_size_bytes={}",
                multiplicity_range_proof.serialized_size_bytes()
            );
            println!("lookup_multiplicity_range_verify=true");
            println!("lookup_multiplicity_range_tamper_rejected=true");
            println!("lookup_multiplicity_range_exact_public_bound=true");
        }
        let mut timestamp_source = Gpt2LassoTimestampSource::<_, Fr>::new(
            &mut stream,
            &registry,
            products.query_count,
            64,
        )
        .unwrap();
        let timestamp_hiding = Fr::from(107_u64);
        let (timestamp_commitment, timestamp_tier_one, timestamp_commit_profile) =
            commit_bva_field_streaming::<Bls12_381, _>(
                &pp,
                &mut timestamp_source,
                timestamp_hiding,
            )
            .unwrap();
        assert_eq!(
            timestamp_commit_profile.physical_entries,
            products.query_count
        );
        assert_eq!(timestamp_commit_profile.materialized_padding_entries, 0);
        assert!(
            BvaFieldSource::<Fr>::layout(&timestamp_source)
                .vector_lengths()
                .iter()
                .all(|length| length.is_power_of_two()),
            "every timestamp range-proof chunk must be a power of two"
        );
        assert!(timestamp_source.profile().maximum_live_entries <= 64);
        assert!(timestamp_source.profile().maximum_timestamp < products.query_count as u64);
        println!("lookup_timestamp_entries={}", products.query_count);
        println!(
            "lookup_timestamp_vectors={}",
            timestamp_source.profile().vector_count
        );
        println!(
            "lookup_timestamp_maximum_live_entries={}",
            timestamp_source.profile().maximum_live_entries
        );
        println!(
            "lookup_timestamp_materialized_padding_entries={}",
            timestamp_commit_profile.materialized_padding_entries
        );
        proof_transcript.absorb(b"lookup-timestamp-commitment", &timestamp_commitment);
        let timestamp_log = BvaFieldSource::<Fr>::layout(&timestamp_source)
            .logical_len()
            .ilog2() as usize;
        let timestamp_col_log = timestamp_log / 2;
        let timestamp_row_log = timestamp_log - timestamp_col_log;
        let timestamp_xl = (0..timestamp_row_log)
            .map(|_| proof_transcript.challenge(b"lookup-timestamp-row-point"))
            .collect::<Vec<Fr>>();
        let timestamp_xr = (0..timestamp_col_log)
            .map(|_| proof_transcript.challenge(b"lookup-timestamp-column-point"))
            .collect::<Vec<Fr>>();
        let timestamp_value = timestamp_source
            .evaluate_packed(&timestamp_xl, &timestamp_xr)
            .unwrap();
        let timestamp_value_hiding = Fr::from(109_u64);
        let timestamp_value_commitment = Bls12_381::pairing(pp.g_0.mul(timestamp_value), pp.h_hat)
            + pp.tilde_u.mul(timestamp_value_hiding);
        let (timestamp_opening, timestamp_open_profile) = open_bva_field_streaming::<Bls12_381, _>(
            &pp,
            &mut timestamp_source,
            &timestamp_xl,
            &timestamp_xr,
            timestamp_value_commitment,
            timestamp_commitment,
            &timestamp_tier_one,
            timestamp_value_hiding,
            timestamp_hiding,
        )
        .unwrap();
        assert!(SmartPC::<Bls12_381>::verify(
            &pp,
            timestamp_commitment,
            timestamp_value_commitment,
            &timestamp_xl,
            &timestamp_xr,
            &timestamp_opening,
        )
        .unwrap());
        assert_eq!(
            timestamp_open_profile.materialized_projection_padding_entries,
            0
        );
        assert_eq!(
            timestamp_open_profile.padding_projection_field_multiplications,
            0
        );
        let tampered_timestamp_commitment =
            timestamp_commitment + Bls12_381::pairing(pp.g_0.mul(Fr::from(1_u64)), pp.h_hat);
        assert!(!SmartPC::<Bls12_381>::verify(
            &pp,
            tampered_timestamp_commitment,
            timestamp_value_commitment,
            &timestamp_xl,
            &timestamp_xr,
            &timestamp_opening,
        )
        .unwrap());
        println!("lookup_timestamp_commit_open_verify=true");
        println!("lookup_timestamp_commitment_tamper_rejected=true");
        proof_transcript.absorb(
            b"lookup-timestamp-value-commitment",
            &timestamp_value_commitment,
        );
        proof_transcript.absorb(b"lookup-timestamp-opening", &timestamp_opening);
        let canonical_timestamp_layout =
            <Gpt2LassoTimestampSource<'_, '_, _, Fr> as BvaFieldSource<Fr>>::layout(
                &timestamp_source,
            )
            .clone();
        drop(timestamp_source);
        drop(stream);
        if std::env::var_os("ZKSMART_GPT2_TIMESTAMP_RANGE_PROOF").is_some() {
            let range_chunk_log = std::env::var("ZKSMART_GPT2_TIMESTAMP_RANGE_CHUNK_LOG")
                .ok()
                .map(|value| value.parse::<usize>().unwrap())
                .unwrap_or(6);
            let range_chunk_len = 1_usize << range_chunk_log;
            let range_bits = (usize::BITS - (products.query_count - 1).leading_zeros()) as usize;
            let range_srs = SRS::<Bls12_381> {
                zksrs: ZkSRS::new(pp.g_0, pp.tilde_g),
                pc_pp: pp.clone(),
                h_hat: pp.h_hat,
            };
            let mut range_stream = Gpt2LookupQueryStream::new(
                &mut checkpoint,
                &layout,
                &activations,
                &witness,
                &manifest,
            )
            .unwrap();
            let mut range_source = Gpt2LassoTimestampSource::<_, Fr>::new(
                &mut range_stream,
                &registry,
                products.query_count,
                range_chunk_len,
            )
            .unwrap();
            assert_eq!(
                <Gpt2LassoTimestampSource<'_, '_, _, Fr> as BvaI32Source>::layout(&range_source),
                &canonical_timestamp_layout,
                "global timestamp range proof must use the canonical commitment layout"
            );
            let prove_started = Instant::now();
            let (range_proof, range_profile) = BvaRangeProof::prove(
                &range_srs,
                0,
                0x5453_4256,
                &mut range_source,
                timestamp_commitment,
                &timestamp_tier_one,
                timestamp_hiding,
                range_bits,
            )
            .unwrap();
            let prove_seconds = prove_started.elapsed().as_secs_f64();
            let verify_started = Instant::now();
            assert!(range_proof.verify(
                &range_srs,
                0,
                0x5453_4256,
                &canonical_timestamp_layout,
                timestamp_commitment,
                range_bits,
            ));
            let verify_seconds = verify_started.elapsed().as_secs_f64();
            let proof_size_bytes = range_proof.serialized_size_bytes();
            assert!(!range_proof.verify(
                &range_srs,
                0,
                0x5453_4256,
                &canonical_timestamp_layout,
                tampered_timestamp_commitment,
                range_bits,
            ));
            println!("lookup_timestamp_range_bits={range_bits}");
            println!("lookup_timestamp_range_chunk_len={range_chunk_len}");
            println!(
                "lookup_timestamp_range_vectors={}",
                canonical_timestamp_layout.vector_lengths().len()
            );
            println!(
                "lookup_timestamp_range_maximum_live_entries={}",
                range_profile.maximum_live_vector_entries
            );
            println!(
                "lookup_timestamp_range_source_passes={}",
                range_profile.source_passes
            );
            println!(
                "lookup_timestamp_range_materialized_padding_entries={}",
                range_profile.materialized_padding_entries
            );
            println!(
                "lookup_timestamp_range_padding_field_operations={}",
                range_profile.padding_field_operations
            );
            println!(
                "lookup_timestamp_range_padding_group_operations={}",
                range_profile.padding_group_operations
            );
            println!("lookup_timestamp_range_prove_seconds={prove_seconds:.6}");
            println!("lookup_timestamp_range_verify_seconds={verify_seconds:.6}");
            println!("lookup_timestamp_range_proof_size_bytes={proof_size_bytes}");
            println!("lookup_timestamp_range_tamper_rejected=true");
            println!("lookup_timestamp_range_global_commitment_verify=true");
        }
        let mut lasso_value_binding_context = None;
        if std::env::var_os("ZKSMART_GPT2_LASSO_GRAND_PRODUCT_PROOF").is_some() {
            let factor_chunk_log = std::env::var("ZKSMART_GPT2_LASSO_FACTOR_CHUNK_LOG")
                .ok()
                .map(|value| value.parse::<usize>().unwrap())
                .unwrap_or(6);
            let factor_chunk_len = 1_usize << factor_chunk_log;
            let factor_srs = SRS::<Bls12_381> {
                zksrs: ZkSRS::new(pp.g_0, pp.tilde_g),
                pc_pp: pp.clone(),
                h_hat: pp.h_hat,
            };

            let factor_commit_started = Instant::now();
            let mut left_stream = Gpt2LookupQueryStream::new(
                &mut checkpoint,
                &layout,
                &activations,
                &witness,
                &manifest,
            )
            .unwrap();
            let mut left_source = Gpt2LassoFactorSource::new(
                &mut left_stream,
                &registry,
                &products,
                lookup_row_challenge,
                lookup_product_challenge,
                factor_chunk_len,
                LassoFactorSide::Left,
            )
            .unwrap();
            let left_hiding = Fr::from(113_u64);
            let left_query_value_hiding = Fr::from(151_u64);
            let left_table_value_hiding = Fr::from(157_u64);
            let left_value_hiding = left_query_value_hiding + left_table_value_hiding;
            let (left_value_commitment, left_value_tier_one, left_value_commit_profile) = {
                let mut component = Gpt2LassoAffineComponentSource::new(
                    &mut left_source,
                    LassoAffineComponent::Value,
                );
                commit_bva_field_streaming::<Bls12_381, _>(&pp, &mut component, left_value_hiding)
                    .unwrap()
            };
            let (
                left_query_value_commitment,
                left_query_value_tier_one,
                left_query_value_commit_profile,
            ) = {
                let mut component = Gpt2LassoAffineComponentSource::new(
                    &mut left_source,
                    LassoAffineComponent::QueryValue,
                );
                commit_bva_field_streaming::<Bls12_381, _>(
                    &pp,
                    &mut component,
                    left_query_value_hiding,
                )
                .unwrap()
            };
            let (
                left_table_value_commitment,
                left_table_value_tier_one,
                left_table_value_commit_profile,
            ) = {
                let mut component = Gpt2LassoAffineComponentSource::new(
                    &mut left_source,
                    LassoAffineComponent::TableValue,
                );
                commit_bva_field_streaming::<Bls12_381, _>(
                    &pp,
                    &mut component,
                    left_table_value_hiding,
                )
                .unwrap()
            };
            assert_eq!(
                left_value_commitment,
                left_query_value_commitment + left_table_value_commitment
            );
            assert_eq!(
                left_value_tier_one,
                left_query_value_tier_one
                    .iter()
                    .zip(&left_table_value_tier_one)
                    .map(|(query, table)| *query + table)
                    .collect::<Vec<_>>()
            );
            let left_query_counter_hiding = Fr::from(137_u64);
            let left_table_counter_hiding = Fr::from(139_u64);
            let left_counter_hiding = left_query_counter_hiding + left_table_counter_hiding;
            let (left_counter_commitment, left_counter_tier_one, left_counter_commit_profile) = {
                let mut component = Gpt2LassoAffineComponentSource::new(
                    &mut left_source,
                    LassoAffineComponent::Counter,
                );
                commit_bva_field_streaming::<Bls12_381, _>(&pp, &mut component, left_counter_hiding)
                    .unwrap()
            };
            let (
                left_query_counter_commitment,
                left_query_counter_tier_one,
                left_query_counter_commit_profile,
            ) = {
                let mut component = Gpt2LassoAffineComponentSource::new(
                    &mut left_source,
                    LassoAffineComponent::QueryCounter,
                );
                commit_bva_field_streaming::<Bls12_381, _>(
                    &pp,
                    &mut component,
                    left_query_counter_hiding,
                )
                .unwrap()
            };
            let (
                left_table_counter_commitment,
                left_table_counter_tier_one,
                left_table_counter_commit_profile,
            ) = {
                let mut component = Gpt2LassoAffineComponentSource::new(
                    &mut left_source,
                    LassoAffineComponent::TableCounter,
                );
                commit_bva_field_streaming::<Bls12_381, _>(
                    &pp,
                    &mut component,
                    left_table_counter_hiding,
                )
                .unwrap()
            };
            assert_eq!(
                left_counter_commitment,
                left_query_counter_commitment + left_table_counter_commitment
            );
            assert_eq!(
                left_counter_tier_one,
                left_query_counter_tier_one
                    .iter()
                    .zip(&left_table_counter_tier_one)
                    .map(|(query, table)| *query + table)
                    .collect::<Vec<_>>()
            );
            let factor_layout = BvaFieldSource::<Fr>::layout(&left_source).clone();
            let timestamp_vector_count = canonical_timestamp_layout.vector_lengths().len();
            assert_eq!(
                &factor_layout.vector_lengths()[..timestamp_vector_count],
                canonical_timestamp_layout.vector_lengths(),
                "Lasso query factor vectors must align with the canonical timestamp vectors"
            );
            assert_eq!(
                factor_layout.vector_lengths()[..timestamp_vector_count]
                    .iter()
                    .sum::<usize>(),
                products.query_count
            );
            drop(left_source);
            drop(left_stream);

            let mut right_stream = Gpt2LookupQueryStream::new(
                &mut checkpoint,
                &layout,
                &activations,
                &witness,
                &manifest,
            )
            .unwrap();
            let mut right_source = Gpt2LassoFactorSource::new(
                &mut right_stream,
                &registry,
                &products,
                lookup_row_challenge,
                lookup_product_challenge,
                factor_chunk_len,
                LassoFactorSide::Right,
            )
            .unwrap();
            assert_eq!(BvaFieldSource::<Fr>::layout(&right_source), &factor_layout);
            let right_hiding = Fr::from(127_u64);
            let right_value_hiding = left_value_hiding;
            let (right_value_commitment, right_value_tier_one, right_value_commit_profile) = {
                let mut component = Gpt2LassoAffineComponentSource::new(
                    &mut right_source,
                    LassoAffineComponent::Value,
                );
                commit_bva_field_streaming::<Bls12_381, _>(&pp, &mut component, right_value_hiding)
                    .unwrap()
            };
            let right_counter_hiding = Fr::from(149_u64);
            let (right_counter_commitment, right_counter_tier_one, right_counter_commit_profile) = {
                let mut component = Gpt2LassoAffineComponentSource::new(
                    &mut right_source,
                    LassoAffineComponent::Counter,
                );
                commit_bva_field_streaming::<Bls12_381, _>(
                    &pp,
                    &mut component,
                    right_counter_hiding,
                )
                .unwrap()
            };
            assert_eq!(
                right_value_commitment, left_value_commitment,
                "both Lasso sides must reuse one canonical compressed-value commitment"
            );
            assert_eq!(right_value_tier_one, left_value_tier_one);
            drop(right_source);
            drop(right_stream);

            // Soundness-first public-support binding.  The verifier rebuilds
            // all three sparse commitments from the disclosed sorted support
            // and the deterministic public lookup registry.  Only the 27
            // nonzero tiny-model rows contribute group operations.
            let public_table_binding = Gpt2LassoPublicTableBinding::new(
                &products,
                multiplicity_hiding,
                left_table_value_hiding,
                left_table_counter_hiding,
            )
            .unwrap();
            let public_table_binding_profile = public_table_binding
                .verify::<Bls12_381>(
                    &pp,
                    &registry,
                    &factor_layout,
                    products.query_count,
                    products.public_table_rows,
                    lookup_row_challenge,
                    multiplicity_commitment,
                    left_table_value_commitment,
                    left_table_counter_commitment,
                )
                .unwrap();
            assert_eq!(
                public_table_binding_profile.support_entries,
                products.distinct_table_rows
            );
            assert!(
                public_table_binding_profile.nonzero_group_contributions
                    <= 3 * products.distinct_table_rows
            );
            assert_eq!(public_table_binding_profile.materialized_padding_entries, 0);
            assert_eq!(public_table_binding_profile.padding_group_operations, 0);

            let mut tampered_public_table_binding = public_table_binding.clone();
            let donor = tampered_public_table_binding
                .multiplicities
                .iter()
                .position(|(_, count)| *count > 1)
                .unwrap();
            let recipient = usize::from(donor == 0);
            tampered_public_table_binding.multiplicities[donor].1 -= 1;
            tampered_public_table_binding.multiplicities[recipient].1 += 1;
            assert!(tampered_public_table_binding
                .verify::<Bls12_381>(
                    &pp,
                    &registry,
                    &factor_layout,
                    products.query_count,
                    products.public_table_rows,
                    lookup_row_challenge,
                    multiplicity_commitment,
                    left_table_value_commitment,
                    left_table_counter_commitment,
                )
                .is_err());

            let mut tampered_product_transcript = proof_transcript.clone();
            proof_transcript.absorb(b"lookup-lasso-value-commitment", &left_value_commitment);
            proof_transcript.absorb(
                b"lookup-lasso-query-value-commitment",
                &left_query_value_commitment,
            );
            proof_transcript.absorb(
                b"lookup-lasso-table-value-commitment",
                &left_table_value_commitment,
            );
            proof_transcript.absorb(
                b"lookup-lasso-left-query-counter-commitment",
                &left_query_counter_commitment,
            );
            proof_transcript.absorb(
                b"lookup-lasso-left-table-counter-commitment",
                &left_table_counter_commitment,
            );
            proof_transcript.absorb(
                b"lookup-lasso-right-counter-commitment",
                &right_counter_commitment,
            );
            lookup_product_challenge = proof_transcript.challenge::<Fr>(b"lookup-product");
            assert_ne!(lookup_product_challenge, Fr::from(0_u64));

            let transcript_commitment_delta =
                Bls12_381::pairing(factor_srs.pc_pp.g_0, factor_srs.h_hat);
            tampered_product_transcript
                .absorb(b"lookup-lasso-value-commitment", &left_value_commitment);
            tampered_product_transcript.absorb(
                b"lookup-lasso-query-value-commitment",
                &left_query_value_commitment,
            );
            tampered_product_transcript.absorb(
                b"lookup-lasso-table-value-commitment",
                &left_table_value_commitment,
            );
            tampered_product_transcript.absorb(
                b"lookup-lasso-left-query-counter-commitment",
                &left_query_counter_commitment,
            );
            tampered_product_transcript.absorb(
                b"lookup-lasso-left-table-counter-commitment",
                &(left_table_counter_commitment + transcript_commitment_delta),
            );
            tampered_product_transcript.absorb(
                b"lookup-lasso-right-counter-commitment",
                &right_counter_commitment,
            );
            assert_ne!(
                tampered_product_transcript.challenge::<Fr>(b"lookup-product"),
                lookup_product_challenge,
                "the product challenge must bind every component commitment"
            );
            println!("lookup_lasso_product_challenge_after_components=true");
            println!("lookup_lasso_product_challenge_component_tamper_changes_challenge=true");

            let mut product_stream = Gpt2LookupQueryStream::new(
                &mut checkpoint,
                &layout,
                &activations,
                &witness,
                &manifest,
            )
            .unwrap();
            let challenged_products = product_stream
                .lasso_products(&registry, lookup_row_challenge, lookup_product_challenge)
                .unwrap();
            assert_eq!(challenged_products.left, challenged_products.right);
            assert_eq!(challenged_products.query_count, products.query_count);
            assert_eq!(
                challenged_products.distinct_table_rows,
                products.distinct_table_rows
            );
            assert_eq!(
                challenged_products.public_table_rows,
                products.public_table_rows
            );
            assert_eq!(challenged_products.multiplicities, products.multiplicities);
            products.left = challenged_products.left;
            products.right = challenged_products.right;
            let expected_product = products.left;

            let mut left_factor_stream = Gpt2LookupQueryStream::new(
                &mut checkpoint,
                &layout,
                &activations,
                &witness,
                &manifest,
            )
            .unwrap();
            let mut left_factor_source = Gpt2LassoFactorSource::new(
                &mut left_factor_stream,
                &registry,
                &products,
                lookup_row_challenge,
                lookup_product_challenge,
                factor_chunk_len,
                LassoFactorSide::Left,
            )
            .unwrap();
            assert_eq!(
                BvaFieldSource::<Fr>::layout(&left_factor_source),
                &factor_layout
            );
            let (left_commitment, left_tier_one, left_commit_profile) =
                commit_bva_field_streaming::<Bls12_381, _>(
                    &pp,
                    &mut left_factor_source,
                    left_hiding,
                )
                .unwrap();
            let left_source_profile = left_factor_source.profile();
            drop(left_factor_source);
            drop(left_factor_stream);

            let mut right_factor_stream = Gpt2LookupQueryStream::new(
                &mut checkpoint,
                &layout,
                &activations,
                &witness,
                &manifest,
            )
            .unwrap();
            let mut right_factor_source = Gpt2LassoFactorSource::new(
                &mut right_factor_stream,
                &registry,
                &products,
                lookup_row_challenge,
                lookup_product_challenge,
                factor_chunk_len,
                LassoFactorSide::Right,
            )
            .unwrap();
            assert_eq!(
                BvaFieldSource::<Fr>::layout(&right_factor_source),
                &factor_layout
            );
            let (right_commitment, right_tier_one, right_commit_profile) =
                commit_bva_field_streaming::<Bls12_381, _>(
                    &pp,
                    &mut right_factor_source,
                    right_hiding,
                )
                .unwrap();
            let right_source_profile = right_factor_source.profile();
            drop(right_factor_source);
            drop(right_factor_stream);
            let factor_commit_seconds = factor_commit_started.elapsed().as_secs_f64();

            assert_eq!(left_commit_profile.materialized_padding_entries, 0);
            assert_eq!(right_commit_profile.materialized_padding_entries, 0);
            assert_eq!(left_commit_profile.padding_group_operations, 0);
            assert_eq!(right_commit_profile.padding_group_operations, 0);
            assert_eq!(left_value_commit_profile.materialized_padding_entries, 0);
            assert_eq!(
                left_query_value_commit_profile.materialized_padding_entries,
                0
            );
            assert_eq!(
                left_table_value_commit_profile.materialized_padding_entries,
                0
            );
            assert_eq!(left_counter_commit_profile.materialized_padding_entries, 0);
            assert_eq!(
                left_query_counter_commit_profile.materialized_padding_entries,
                0
            );
            assert_eq!(
                left_table_counter_commit_profile.materialized_padding_entries,
                0
            );
            assert_eq!(right_value_commit_profile.materialized_padding_entries, 0);
            assert_eq!(right_counter_commit_profile.materialized_padding_entries, 0);
            assert_eq!(left_source_profile.query_factors, products.query_count);
            assert_eq!(right_source_profile.query_factors, products.query_count);
            assert_eq!(
                left_source_profile.table_factors,
                products.distinct_table_rows
            );
            assert_eq!(left_source_profile, right_source_profile);

            let factor_prove_started = Instant::now();
            let mut left_stream = Gpt2LookupQueryStream::new(
                &mut checkpoint,
                &layout,
                &activations,
                &witness,
                &manifest,
            )
            .unwrap();
            let mut left_source = Gpt2LassoFactorSource::new(
                &mut left_stream,
                &registry,
                &products,
                lookup_row_challenge,
                lookup_product_challenge,
                factor_chunk_len,
                LassoFactorSide::Left,
            )
            .unwrap();
            let left_grand_product_prove_started = Instant::now();
            let (left_proof, left_profile) = BvaGrandProductProof::prove(
                &factor_srs,
                0,
                0x4c41_5353,
                &mut left_source,
                left_commitment,
                &left_tier_one,
                left_hiding,
                expected_product,
            )
            .unwrap();
            let mut grand_product_prove_seconds =
                left_grand_product_prove_started.elapsed().as_secs_f64();
            let (left_affine_proof, left_affine_profile) = BvaAffineRelationProof::prove(
                &factor_srs,
                0,
                0x4c41_464c,
                &mut left_source,
                left_commitment,
                &left_tier_one,
                left_hiding,
                left_value_commitment,
                &left_value_tier_one,
                left_value_hiding,
                left_counter_commitment,
                &left_counter_tier_one,
                left_counter_hiding,
                Fr::from(1_u64),
                lookup_product_challenge,
                lookup_product_challenge * lookup_product_challenge,
            )
            .unwrap();
            let (left_counter_routes, left_counter_constants) = lasso_query_timestamp_routes::<Fr>(
                &factor_layout,
                &canonical_timestamp_layout,
                Fr::from(0_u64),
            )
            .unwrap();
            let mut left_timestamp_replay_checkpoint = tiny_checkpoint();
            let mut left_timestamp_replay_stream = Gpt2LookupQueryStream::new(
                &mut left_timestamp_replay_checkpoint,
                &layout,
                &activations,
                &witness,
                &manifest,
            )
            .unwrap();
            let mut left_timestamp_replay_source = Gpt2LassoTimestampSource::<_, Fr>::new(
                &mut left_timestamp_replay_stream,
                &registry,
                products.query_count,
                factor_chunk_len,
            )
            .unwrap();
            let (left_counter_timestamp_proof, left_counter_timestamp_profile) = {
                let mut left_query_counter_source = Gpt2LassoAffineComponentSource::new(
                    &mut left_source,
                    LassoAffineComponent::QueryCounter,
                );
                BvaRoutedEqualityProof::prove(
                    &factor_srs,
                    0,
                    0x4c41_5451,
                    &mut left_query_counter_source,
                    left_query_counter_commitment,
                    &left_query_counter_tier_one,
                    left_query_counter_hiding,
                    &mut left_timestamp_replay_source,
                    timestamp_commitment,
                    &timestamp_tier_one,
                    timestamp_hiding,
                    &left_counter_routes,
                    &left_counter_constants,
                )
                .unwrap()
            };
            drop(left_source);
            drop(left_stream);

            let mut right_stream = Gpt2LookupQueryStream::new(
                &mut checkpoint,
                &layout,
                &activations,
                &witness,
                &manifest,
            )
            .unwrap();
            let mut right_source = Gpt2LassoFactorSource::new(
                &mut right_stream,
                &registry,
                &products,
                lookup_row_challenge,
                lookup_product_challenge,
                factor_chunk_len,
                LassoFactorSide::Right,
            )
            .unwrap();
            let right_grand_product_prove_started = Instant::now();
            let (right_proof, right_profile) = BvaGrandProductProof::prove(
                &factor_srs,
                0,
                0x4c41_5354,
                &mut right_source,
                right_commitment,
                &right_tier_one,
                right_hiding,
                expected_product,
            )
            .unwrap();
            grand_product_prove_seconds +=
                right_grand_product_prove_started.elapsed().as_secs_f64();
            let (right_affine_proof, right_affine_profile) = BvaAffineRelationProof::prove(
                &factor_srs,
                0,
                0x4c41_4652,
                &mut right_source,
                right_commitment,
                &right_tier_one,
                right_hiding,
                right_value_commitment,
                &right_value_tier_one,
                right_value_hiding,
                right_counter_commitment,
                &right_counter_tier_one,
                right_counter_hiding,
                Fr::from(1_u64),
                lookup_product_challenge,
                lookup_product_challenge * lookup_product_challenge,
            )
            .unwrap();
            let (right_counter_routes, right_counter_constants) =
                lasso_query_timestamp_routes::<Fr>(
                    &factor_layout,
                    &canonical_timestamp_layout,
                    Fr::from(1_u64),
                )
                .unwrap();
            let mut timestamp_replay_checkpoint = tiny_checkpoint();
            let mut timestamp_replay_stream = Gpt2LookupQueryStream::new(
                &mut timestamp_replay_checkpoint,
                &layout,
                &activations,
                &witness,
                &manifest,
            )
            .unwrap();
            let mut timestamp_replay_source = Gpt2LassoTimestampSource::<_, Fr>::new(
                &mut timestamp_replay_stream,
                &registry,
                products.query_count,
                factor_chunk_len,
            )
            .unwrap();
            let (right_counter_timestamp_proof, right_counter_timestamp_profile) = {
                let mut right_counter_source = Gpt2LassoAffineComponentSource::new(
                    &mut right_source,
                    LassoAffineComponent::Counter,
                );
                BvaRoutedEqualityProof::prove(
                    &factor_srs,
                    0,
                    0x4c41_5453,
                    &mut right_counter_source,
                    right_counter_commitment,
                    &right_counter_tier_one,
                    right_counter_hiding,
                    &mut timestamp_replay_source,
                    timestamp_commitment,
                    &timestamp_tier_one,
                    timestamp_hiding,
                    &right_counter_routes,
                    &right_counter_constants,
                )
                .unwrap()
            };
            let factor_prove_seconds = factor_prove_started.elapsed().as_secs_f64();

            let factor_verify_started = Instant::now();
            let grand_product_verify_started = Instant::now();
            assert!(left_proof.verify(
                &factor_srs,
                0,
                0x4c41_5353,
                &factor_layout,
                left_commitment,
                expected_product,
            ));
            assert!(right_proof.verify(
                &factor_srs,
                0,
                0x4c41_5354,
                &factor_layout,
                right_commitment,
                expected_product,
            ));
            let grand_product_verify_seconds = grand_product_verify_started.elapsed().as_secs_f64();
            assert!(left_affine_proof.verify(
                &factor_srs,
                0,
                0x4c41_464c,
                &factor_layout,
                left_commitment,
                left_value_commitment,
                left_counter_commitment,
                Fr::from(1_u64),
                lookup_product_challenge,
                lookup_product_challenge * lookup_product_challenge,
            ));
            assert!(right_affine_proof.verify(
                &factor_srs,
                0,
                0x4c41_4652,
                &factor_layout,
                right_commitment,
                right_value_commitment,
                right_counter_commitment,
                Fr::from(1_u64),
                lookup_product_challenge,
                lookup_product_challenge * lookup_product_challenge,
            ));
            assert!(right_counter_timestamp_proof.verify(
                &factor_srs,
                0,
                0x4c41_5453,
                &factor_layout,
                right_counter_commitment,
                &canonical_timestamp_layout,
                timestamp_commitment,
                &right_counter_routes,
                &right_counter_constants,
            ));
            assert!(left_counter_timestamp_proof.verify(
                &factor_srs,
                0,
                0x4c41_5451,
                &factor_layout,
                left_query_counter_commitment,
                &canonical_timestamp_layout,
                timestamp_commitment,
                &left_counter_routes,
                &left_counter_constants,
            ));
            let factor_verify_seconds = factor_verify_started.elapsed().as_secs_f64();
            let factor_commitment_delta =
                Bls12_381::pairing(factor_srs.pc_pp.g_0, factor_srs.h_hat);
            assert!(!left_proof.verify(
                &factor_srs,
                0,
                0x4c41_5353,
                &factor_layout,
                left_commitment + factor_commitment_delta,
                expected_product,
            ));
            assert!(!right_proof.verify(
                &factor_srs,
                0,
                0x4c41_5354,
                &factor_layout,
                right_commitment,
                expected_product + Fr::from(1_u64),
            ));
            assert!(!left_affine_proof.verify(
                &factor_srs,
                0,
                0x4c41_464c,
                &factor_layout,
                left_commitment,
                left_value_commitment + factor_commitment_delta,
                left_counter_commitment,
                Fr::from(1_u64),
                lookup_product_challenge,
                lookup_product_challenge * lookup_product_challenge,
            ));
            assert!(!right_affine_proof.verify(
                &factor_srs,
                0,
                0x4c41_4652,
                &factor_layout,
                right_commitment,
                right_value_commitment,
                right_counter_commitment + factor_commitment_delta,
                Fr::from(1_u64),
                lookup_product_challenge,
                lookup_product_challenge * lookup_product_challenge,
            ));
            assert!(!right_counter_timestamp_proof.verify(
                &factor_srs,
                0,
                0x4c41_5453,
                &factor_layout,
                right_counter_commitment,
                &canonical_timestamp_layout,
                timestamp_commitment + factor_commitment_delta,
                &right_counter_routes,
                &right_counter_constants,
            ));
            assert!(!left_counter_timestamp_proof.verify(
                &factor_srs,
                0,
                0x4c41_5451,
                &factor_layout,
                left_query_counter_commitment + factor_commitment_delta,
                &canonical_timestamp_layout,
                timestamp_commitment,
                &left_counter_routes,
                &left_counter_constants,
            ));

            println!(
                "lookup_lasso_factor_entries={}",
                products.query_count + products.distinct_table_rows
            );
            println!(
                "lookup_lasso_factor_vectors={}",
                factor_layout.vector_lengths().len()
            );
            println!("lookup_lasso_factor_chunk_len={factor_chunk_len}");
            println!(
                "lookup_lasso_factor_maximum_live_entries={}",
                left_profile
                    .maximum_live_vector_entries
                    .max(right_profile.maximum_live_vector_entries)
            );
            println!(
                "lookup_lasso_factor_source_passes={}",
                left_profile.source_passes + right_profile.source_passes
            );
            println!(
                "lookup_lasso_factor_materialized_padding_entries={}",
                left_profile.materialized_padding_entries
                    + right_profile.materialized_padding_entries
            );
            println!(
                "lookup_lasso_factor_padding_field_operations={}",
                left_profile.padding_field_operations + right_profile.padding_field_operations
            );
            println!(
                "lookup_lasso_factor_padding_group_operations={}",
                left_profile.padding_group_operations + right_profile.padding_group_operations
            );
            println!("lookup_lasso_factor_commit_seconds={factor_commit_seconds:.6}");
            println!("lookup_lasso_grand_product_prove_seconds={grand_product_prove_seconds:.6}");
            println!("lookup_lasso_grand_product_verify_seconds={grand_product_verify_seconds:.6}");
            println!("lookup_lasso_combined_prove_seconds={factor_prove_seconds:.6}");
            println!("lookup_lasso_combined_verify_seconds={factor_verify_seconds:.6}");
            println!(
                "lookup_lasso_grand_product_proof_size_bytes={}",
                left_proof.serialized_size_bytes() + right_proof.serialized_size_bytes()
            );
            println!("lookup_lasso_grand_product_verify=true");
            println!("lookup_lasso_grand_product_tamper_rejected=true");
            println!(
                "lookup_lasso_affine_binding_proof_size_bytes={}",
                left_affine_proof.serialized_size_bytes()
                    + right_affine_proof.serialized_size_bytes()
            );
            println!(
                "lookup_lasso_affine_binding_source_passes={}",
                left_affine_profile.source_passes + right_affine_profile.source_passes
            );
            println!(
                "lookup_lasso_affine_binding_maximum_live_entries={}",
                left_affine_profile
                    .maximum_live_vector_entries
                    .max(right_affine_profile.maximum_live_vector_entries)
            );
            println!(
                "lookup_lasso_affine_binding_materialized_padding_entries={}",
                left_affine_profile.materialized_padding_entries
                    + right_affine_profile.materialized_padding_entries
            );
            println!(
                "lookup_lasso_affine_binding_padding_field_operations={}",
                left_affine_profile.padding_field_operations
                    + right_affine_profile.padding_field_operations
            );
            println!("lookup_lasso_affine_binding_verify=true");
            println!("lookup_lasso_affine_binding_tamper_rejected=true");
            println!(
                "lookup_lasso_right_counter_timestamp_binding_proof_size_bytes={}",
                right_counter_timestamp_proof.serialized_size_bytes()
            );
            println!(
                "lookup_lasso_right_counter_timestamp_binding_source_passes={}",
                right_counter_timestamp_profile.source_passes
            );
            println!(
                "lookup_lasso_right_counter_timestamp_binding_maximum_live_entries={}",
                right_counter_timestamp_profile.maximum_live_vector_entries
            );
            println!(
                "lookup_lasso_right_counter_timestamp_binding_materialized_padding_entries={}",
                right_counter_timestamp_profile.materialized_padding_entries
            );
            println!(
                "lookup_lasso_right_counter_timestamp_binding_padding_field_operations={}",
                right_counter_timestamp_profile.padding_field_operations
            );
            println!("lookup_lasso_right_counter_timestamp_binding_verify=true");
            println!("lookup_lasso_right_counter_timestamp_binding_tamper_rejected=true");
            println!(
                "lookup_lasso_left_query_counter_timestamp_binding_proof_size_bytes={}",
                left_counter_timestamp_proof.serialized_size_bytes()
            );
            println!(
                "lookup_lasso_left_query_counter_timestamp_binding_source_passes={}",
                left_counter_timestamp_profile.source_passes
            );
            println!(
                "lookup_lasso_left_query_counter_timestamp_binding_maximum_live_entries={}",
                left_counter_timestamp_profile.maximum_live_vector_entries
            );
            println!(
                "lookup_lasso_left_query_counter_timestamp_binding_materialized_padding_entries={}",
                left_counter_timestamp_profile.materialized_padding_entries
            );
            println!(
                "lookup_lasso_left_query_counter_timestamp_binding_padding_field_operations={}",
                left_counter_timestamp_profile.padding_field_operations
            );
            println!("lookup_lasso_left_query_counter_timestamp_binding_verify=true");
            println!("lookup_lasso_left_query_counter_timestamp_binding_tamper_rejected=true");
            lasso_value_binding_context = Some((
                factor_layout.clone(),
                factor_chunk_len,
                left_query_value_commitment,
                left_query_value_tier_one.clone(),
                left_query_value_hiding,
            ));
            println!(
                "lookup_lasso_public_support_entries={}",
                products.distinct_table_rows
            );
            println!(
                "lookup_lasso_public_support_disclosure_bytes={}",
                public_table_binding.serialized_size_bytes()
            );
            println!(
                "lookup_lasso_public_support_nonzero_group_contributions={}",
                public_table_binding_profile.nonzero_group_contributions
            );
            println!("lookup_lasso_public_support_disclosed=true");
            println!("lookup_lasso_public_support_tamper_rejected=true");
            println!("lookup_lasso_public_table_zero_entries_materialized=0");
            println!("lookup_lasso_public_table_zero_entries_ip_operations=0");
            println!("lookup_lasso_table_value_registry_binding_complete=true");
            println!("lookup_lasso_left_table_counter_multiplicity_binding_complete=true");
            println!("lookup_lasso_factor_binding_complete=true");
        }
        if let Some((_, _, value_commitment, _, _)) = lasso_value_binding_context.as_ref() {
            proof_transcript.absorb(b"lookup-lasso-value-commitment", value_commitment);
        }
        let query_layout = lasso_value_binding_context
            .as_ref()
            .map(|(factor_layout, ..)| factor_layout);
        let (lookup_tbv, lookup_query_tbv, lookup_public_offset, covered_lookup_relations) =
            manifest
                .lookup_relations_tbv::<Fr>(
                    &layout,
                    &bva,
                    query_layout,
                    lookup_row_challenge,
                    || proof_transcript.challenge(b"lookup-leaf-routing"),
                )
                .unwrap();
        assert_eq!(
            covered_lookup_relations,
            manifest
                .relations
                .iter()
                .filter(|relation| relation.family == RelationFamily::TaggedLookup)
                .count()
        );
        assert_ne!(lookup_public_offset, Fr::from(0_u64));
        assert_eq!(lookup_tbv.layout(), &bva.layout);
        let mut source = Gpt2PrimitiveBvaFieldSource::<_, Fr>::new(
            &mut checkpoint,
            &layout,
            &activations,
            &manifest,
            &witness,
            &bva,
        )
        .unwrap();

        let tbv = TensorBasisVector::new(
            &bva.layout,
            bva.layout
                .bucket_sizes()
                .iter()
                .enumerate()
                .map(|(bucket, size)| {
                    TbvBucket::new(
                        Fr::from((bucket + 5) as u64),
                        (0..size.ilog2())
                            .map(|round| Fr::from(u64::from(round) + 11))
                            .collect(),
                    )
                })
                .collect(),
        )
        .unwrap();
        let mut materialized = Vec::new();
        source
            .for_each_vector(&mut |_, values| materialized.push(values.to_vec()))
            .unwrap();
        let expected = BucketedVector::new(materialized)
            .unwrap()
            .inner_product(&tbv)
            .unwrap();
        let actual = inner_product_bva_field_streaming(&mut source, &tbv).unwrap();
        assert_eq!(actual.value, expected.value);
        assert_eq!(actual.profile.physical_entries, bva.physical_entries());
        assert_eq!(actual.profile.materialized_padding_entries, 0);
        assert_eq!(actual.profile.padding_field_operations, 0);
        assert!(
            actual.profile.maximum_live_vector_entries
                <= shape.maximum_streamed_range_limb_column_entries
        );
        let (eo_proof, eo_counters, eo_profile) =
            EoLiteBulletProof::prove_streaming(&mut source, &tbv, actual.value).unwrap();
        let eo_verification = eo_proof.verify_with_tbv(&tbv, actual.value);
        assert!(eo_verification.accepted);
        assert_eq!(eo_profile.materialized_padding_entries, 0);
        assert_eq!(eo_counters.padding_ip_field_ops, 0);
        assert_eq!(eo_counters.padding_fold_field_ops, 0);
        assert!(
            eo_profile.maximum_live_vector_entries
                <= shape.maximum_streamed_range_limb_column_entries
        );
        let (xl, xr) =
            bva_eo_opening_point(bva.logical_entries(), &eo_verification.challenges_inv).unwrap();
        let value_hiding = Fr::from(79_u64);
        let value_commitment = Bls12_381::pairing(pp.g_0.mul(eo_proof.a_reduced), pp.h_hat)
            + pp.tilde_u.mul(value_hiding);
        let (opening, opening_profile) = open_bva_field_streaming::<Bls12_381, _>(
            &pp,
            &mut source,
            &xl,
            &xr,
            value_commitment,
            matrix_commitment,
            &tier_one,
            value_hiding,
            matrix_hiding,
        )
        .unwrap();
        assert!(SmartPC::<Bls12_381>::verify(
            &pp,
            matrix_commitment,
            value_commitment,
            &xl,
            &xr,
            &opening,
        )
        .unwrap());
        assert_eq!(opening_profile.materialized_projection_padding_entries, 0);
        assert_eq!(opening_profile.padding_projection_field_multiplications, 0);

        let (range_tbv, range_claim) = manifest
            .range_recomposition_tbv::<Fr>(&bva, || proof_transcript.challenge(b"range-routing"))
            .unwrap();
        let range_evaluation = inner_product_bva_field_streaming(&mut source, &range_tbv).unwrap();
        assert_eq!(range_evaluation.value, range_claim);
        assert_eq!(range_evaluation.profile.materialized_padding_entries, 0);

        let (residual_tbv, residual_claim) = manifest
            .residual_add_tbv::<Fr>(&layout, &bva, || {
                proof_transcript.challenge(b"residual-routing")
            })
            .unwrap();
        let residual_evaluation =
            inner_product_bva_field_streaming(&mut source, &residual_tbv).unwrap();
        assert_eq!(residual_evaluation.value, residual_claim);
        let (residual_eo, residual_counters, residual_profile) =
            EoLiteBulletProof::prove_streaming(&mut source, &residual_tbv, residual_claim).unwrap();
        let residual_verification = residual_eo.verify_with_tbv(&residual_tbv, residual_claim);
        assert!(residual_verification.accepted);
        assert_eq!(residual_profile.materialized_padding_entries, 0);
        assert_eq!(residual_counters.padding_ip_field_ops, 0);
        assert_eq!(residual_counters.padding_fold_field_ops, 0);
        let (residual_xl, residual_xr) =
            bva_eo_opening_point(bva.logical_entries(), &residual_verification.challenges_inv)
                .unwrap();
        let residual_value_hiding = Fr::from(83_u64);
        let residual_value_commitment =
            Bls12_381::pairing(pp.g_0.mul(residual_eo.a_reduced), pp.h_hat)
                + pp.tilde_u.mul(residual_value_hiding);
        let (residual_opening, residual_opening_profile) =
            open_bva_field_streaming::<Bls12_381, _>(
                &pp,
                &mut source,
                &residual_xl,
                &residual_xr,
                residual_value_commitment,
                matrix_commitment,
                &tier_one,
                residual_value_hiding,
                matrix_hiding,
            )
            .unwrap();
        assert!(SmartPC::<Bls12_381>::verify(
            &pp,
            matrix_commitment,
            residual_value_commitment,
            &residual_xl,
            &residual_xr,
            &residual_opening,
        )
        .unwrap());
        assert_eq!(
            residual_opening_profile.materialized_projection_padding_entries,
            0
        );

        let (linear_tbv, linear_claim, covered_relations) = manifest
            .linear_relations_tbv::<Fr>(&metadata, &layout, &bva, &token_ids, || {
                proof_transcript.challenge(b"linear-routing")
            })
            .unwrap();
        assert_eq!(
            covered_relations,
            manifest
                .relations
                .iter()
                .filter(|relation| {
                    relation.family == RelationFamily::Linear
                        && relation.kind != LinearRelationKind::RangeRecomposition as u8
                })
                .count()
        );
        let linear_evaluation =
            inner_product_bva_field_streaming(&mut source, &linear_tbv).unwrap();
        assert_eq!(linear_evaluation.value, linear_claim);
        let (linear_eo, linear_counters, linear_profile) =
            EoLiteBulletProof::prove_streaming(&mut source, &linear_tbv, linear_claim).unwrap();
        let linear_verification = linear_eo.verify_with_tbv(&linear_tbv, linear_claim);
        assert!(linear_verification.accepted);
        assert_eq!(linear_profile.materialized_padding_entries, 0);
        assert_eq!(linear_counters.padding_ip_field_ops, 0);
        assert_eq!(linear_counters.padding_fold_field_ops, 0);
        let (linear_xl, linear_xr) =
            bva_eo_opening_point(bva.logical_entries(), &linear_verification.challenges_inv)
                .unwrap();
        let linear_value_hiding = Fr::from(89_u64);
        let linear_value_commitment = Bls12_381::pairing(pp.g_0.mul(linear_eo.a_reduced), pp.h_hat)
            + pp.tilde_u.mul(linear_value_hiding);
        let (linear_opening, linear_opening_profile) = open_bva_field_streaming::<Bls12_381, _>(
            &pp,
            &mut source,
            &linear_xl,
            &linear_xr,
            linear_value_commitment,
            matrix_commitment,
            &tier_one,
            linear_value_hiding,
            matrix_hiding,
        )
        .unwrap();
        assert!(SmartPC::<Bls12_381>::verify(
            &pp,
            matrix_commitment,
            linear_value_commitment,
            &linear_xl,
            &linear_xr,
            &linear_opening,
        )
        .unwrap());
        assert_eq!(
            linear_opening_profile.materialized_projection_padding_entries,
            0
        );
        let embedding_output = manifest
            .relations
            .iter()
            .find(|relation| {
                relation.family == RelationFamily::Linear
                    && relation.kind == LinearRelationKind::EmbeddingGatherAdd as u8
            })
            .and_then(|relation| relation.blocks.get(3))
            .copied()
            .unwrap();
        let embedding_output_vector = bva
            .vectors
            .iter()
            .position(|vector| {
                matches!(
                    vector,
                    PrimitiveBvaVectorSource::Base { block, column: 0 }
                        if *block == embedding_output
                )
            })
            .unwrap();
        {
            let mut tampered_source = TamperedBvaFieldSource {
                inner: &mut source,
                vector_index: embedding_output_vector,
            };
            let tampered_linear =
                inner_product_bva_field_streaming(&mut tampered_source, &linear_tbv).unwrap();
            assert_ne!(tampered_linear.value, linear_claim);
        }
        let (io_tbv, io_claim) = manifest
            .public_io_tbv::<Fr>(&layout, &bva, &token_ids, &public_logits, || {
                proof_transcript.challenge(b"public-io-routing")
            })
            .unwrap();
        let io_evaluation = inner_product_bva_field_streaming(&mut source, &io_tbv).unwrap();
        assert_eq!(io_evaluation.value, io_claim);
        let lookup_evaluation =
            inner_product_bva_field_streaming(&mut source, &lookup_tbv).unwrap();
        let lookup_claim = lookup_evaluation.value;
        let lookup_compressed_projection = lookup_claim + lookup_public_offset;
        assert_ne!(lookup_compressed_projection, Fr::from(0_u64));
        if let Some((
            factor_layout,
            factor_chunk_len,
            value_commitment,
            value_tier_one,
            value_hiding,
        )) = lasso_value_binding_context.as_ref()
        {
            let query_tbv = lookup_query_tbv
                .as_ref()
                .expect("factor context must request the public query TBV");
            assert_eq!(query_tbv.layout(), factor_layout);
            let factor_srs = SRS::<Bls12_381> {
                zksrs: ZkSRS::new(pp.g_0, pp.tilde_g),
                pc_pp: pp.clone(),
                h_hat: pp.h_hat,
            };
            let mut value_checkpoint = tiny_checkpoint();
            let mut value_stream = Gpt2LookupQueryStream::new(
                &mut value_checkpoint,
                &layout,
                &activations,
                &witness,
                &manifest,
            )
            .unwrap();
            let mut value_source = Gpt2LassoFactorSource::new(
                &mut value_stream,
                &registry,
                &products,
                lookup_row_challenge,
                lookup_product_challenge,
                *factor_chunk_len,
                LassoFactorSide::Left,
            )
            .unwrap();
            let query_value_binding_started = Instant::now();
            let (query_value_binding_proof, query_value_binding_profile) = {
                let mut component = Gpt2LassoAffineComponentSource::new(
                    &mut value_source,
                    LassoAffineComponent::QueryValue,
                );
                BvaCommittedEvaluationProof::prove(
                    &factor_srs,
                    0,
                    0x4c41_5651,
                    statement.transcript_digest(),
                    &mut component,
                    *value_commitment,
                    value_tier_one,
                    *value_hiding,
                    query_tbv,
                    lookup_compressed_projection,
                )
                .unwrap()
            };
            let query_value_binding_prove_seconds =
                query_value_binding_started.elapsed().as_secs_f64();
            let query_value_binding_verify_started = Instant::now();
            assert!(query_value_binding_proof.verify(
                &factor_srs,
                0,
                0x4c41_5651,
                statement.transcript_digest(),
                factor_layout,
                *value_commitment,
                query_tbv,
                lookup_compressed_projection,
            ));
            let query_value_binding_verify_seconds =
                query_value_binding_verify_started.elapsed().as_secs_f64();
            let commitment_delta = Bls12_381::pairing(factor_srs.pc_pp.g_0, factor_srs.h_hat);
            assert!(!query_value_binding_proof.verify(
                &factor_srs,
                0,
                0x4c41_5651,
                statement.transcript_digest(),
                factor_layout,
                *value_commitment + commitment_delta,
                query_tbv,
                lookup_compressed_projection,
            ));
            assert!(!query_value_binding_proof.verify(
                &factor_srs,
                0,
                0x4c41_5651,
                statement.transcript_digest(),
                factor_layout,
                *value_commitment,
                query_tbv,
                lookup_compressed_projection + Fr::from(1_u64),
            ));
            let mut tampered_digest = statement.transcript_digest();
            tampered_digest[0] ^= 1;
            assert!(!query_value_binding_proof.verify(
                &factor_srs,
                0,
                0x4c41_5651,
                tampered_digest,
                factor_layout,
                *value_commitment,
                query_tbv,
                lookup_compressed_projection,
            ));
            println!(
                "lookup_lasso_query_value_global_binding_prove_seconds={query_value_binding_prove_seconds:.6}"
            );
            println!(
                "lookup_lasso_query_value_global_binding_verify_seconds={query_value_binding_verify_seconds:.6}"
            );
            println!(
                "lookup_lasso_query_value_global_binding_proof_size_bytes={}",
                query_value_binding_proof.serialized_size_bytes()
            );
            println!(
                "lookup_lasso_query_value_global_binding_source_passes={}",
                query_value_binding_profile.source_passes
            );
            println!(
                "lookup_lasso_query_value_global_binding_maximum_live_entries={}",
                query_value_binding_profile.maximum_live_vector_entries
            );
            println!(
                "lookup_lasso_query_value_global_binding_materialized_padding_entries={}",
                query_value_binding_profile.materialized_padding_entries
            );
            println!(
                "lookup_lasso_query_value_global_binding_padding_field_operations={}",
                query_value_binding_profile.padding_field_operations
            );
            println!("lookup_lasso_query_value_global_binding_verify=true");
            println!("lookup_lasso_query_value_global_binding_tamper_rejected=true");
            println!("lookup_lasso_query_value_global_binding_complete=true");
        }
        assert_eq!(lookup_evaluation.profile.materialized_padding_entries, 0);
        assert_eq!(lookup_evaluation.profile.padding_field_operations, 0);
        let bilinear_routing = manifest
            .elementwise_bilinear_routing(&metadata, &layout, &bva)
            .unwrap();
        assert_eq!(
            bilinear_routing.covered_relations,
            manifest
                .relations
                .iter()
                .filter(|relation| {
                    if relation.family != RelationFamily::Bilinear {
                        return false;
                    }
                    !matches!(
                        decode_bilinear_relation_kind(relation.kind).unwrap(),
                        BilinearRelationKind::MatrixMultiply
                            | BilinearRelationKind::LastTokenMatrixMultiply
                            | BilinearRelationKind::CausalQk
                            | BilinearRelationKind::CausalPv
                    )
                })
                .count()
        );
        assert!(bilinear_routing
            .vectors
            .iter()
            .any(|vector| vector.right_constant == -1));
        assert!(bilinear_routing
            .vectors
            .iter()
            .any(|vector| vector.right_scalar_index.is_some()));
        assert!(bilinear_routing
            .vectors
            .iter()
            .any(|vector| vector.right_segmented));
        let bilinear_point = (0..bilinear_routing.layout.logical_len().ilog2())
            .map(|_| proof_transcript.challenge(b"elementwise-bilinear-routing"))
            .collect::<Vec<_>>();
        let bilinear_weight =
            TensorBasisVector::global_tensor(&bilinear_routing.layout, &bilinear_point).unwrap();
        let bilinear_output_tbv = bilinear_routing
            .route_output(&bilinear_weight, &bva.layout)
            .unwrap();
        let bilinear_claim = inner_product_bva_field_streaming(&mut source, &bilinear_output_tbv)
            .unwrap()
            .value;
        let (bilinear_proof, bilinear_counters, bilinear_profile) = {
            let mut pair_source = source
                .elementwise_pair_source(&bilinear_routing, &bilinear_weight)
                .unwrap();
            EoWitnessPairProof::prove_streaming(&mut pair_source, bilinear_claim).unwrap()
        };
        let bilinear_verification =
            bilinear_proof.verify(bilinear_routing.layout.logical_len(), bilinear_claim);
        assert!(bilinear_verification.accepted);
        assert_eq!(bilinear_profile.materialized_padding_entries, 0);
        assert_eq!(bilinear_counters.padding_ip_field_ops, 0);
        assert_eq!(bilinear_counters.padding_fold_field_ops, 0);
        let bilinear_left_equation = TensorBasisVector::global_tensor(
            &bilinear_routing.layout,
            &bilinear_verification.challenges_inv,
        )
        .unwrap();
        let bilinear_right_fold = TensorBasisVector::global_tensor(
            &bilinear_routing.layout,
            &bilinear_verification.challenges,
        )
        .unwrap();
        let bilinear_right_equation = bilinear_weight
            .pointwise_product(&bilinear_right_fold)
            .unwrap();
        let bilinear_left_tbv = bilinear_routing
            .route_left(&bilinear_left_equation, &bva.layout)
            .unwrap();
        let bilinear_right_tbv = bilinear_routing
            .route_right(&bilinear_right_equation, &bva)
            .unwrap();
        let bilinear_right_claim = bilinear_routing
            .right_witness_claim(&bilinear_right_equation, bilinear_proof.b_reduced)
            .unwrap();
        assert_eq!(
            inner_product_bva_field_streaming(&mut source, &bilinear_left_tbv)
                .unwrap()
                .value,
            bilinear_proof.a_reduced
        );
        assert_eq!(
            inner_product_bva_field_streaming(&mut source, &bilinear_right_tbv)
                .unwrap()
                .value,
            bilinear_right_claim
        );
        let matmul_routing = manifest.dense_matmul_routing(&layout, &bva).unwrap();
        assert_eq!(
            matmul_routing.covered_relations,
            manifest
                .relations
                .iter()
                .filter(|relation| {
                    relation.family == RelationFamily::Bilinear
                        && matches!(
                            decode_bilinear_relation_kind(relation.kind).unwrap(),
                            BilinearRelationKind::MatrixMultiply
                                | BilinearRelationKind::LastTokenMatrixMultiply
                        )
                })
                .count()
        );
        let matmul_projections = matmul_routing
            .projections::<Fr>(|| proof_transcript.challenge(b"dense-matmul-routing"))
            .unwrap();
        let matmul_output_tbv = matmul_routing
            .route_output(&matmul_projections, &bva)
            .unwrap();
        let matmul_claim = inner_product_bva_field_streaming(&mut source, &matmul_output_tbv)
            .unwrap()
            .value;
        let (matmul_proof, matmul_counters, matmul_profile) = {
            let mut pair_source = source
                .dense_matmul_pair_source(&matmul_routing, &matmul_projections)
                .unwrap();
            EoWitnessPairProof::prove_streaming(&mut pair_source, matmul_claim).unwrap()
        };
        let matmul_verification =
            matmul_proof.verify(matmul_routing.layout.logical_len(), matmul_claim);
        assert!(matmul_verification.accepted);
        assert_eq!(matmul_profile.materialized_padding_entries, 0);
        assert_eq!(matmul_counters.padding_ip_field_ops, 0);
        assert_eq!(matmul_counters.padding_fold_field_ops, 0);
        let matmul_left_equation = TensorBasisVector::global_tensor(
            &matmul_routing.layout,
            &matmul_verification.challenges_inv,
        )
        .unwrap();
        let matmul_right_equation = TensorBasisVector::global_tensor(
            &matmul_routing.layout,
            &matmul_verification.challenges,
        )
        .unwrap();
        let matmul_left_tbv = matmul_routing
            .route_left(&matmul_left_equation, &matmul_projections, &bva)
            .unwrap();
        let matmul_right_tbv = matmul_routing
            .route_right(&matmul_right_equation, &matmul_projections, &bva)
            .unwrap();
        assert_eq!(
            inner_product_bva_field_streaming(&mut source, &matmul_left_tbv)
                .unwrap()
                .value,
            matmul_proof.a_reduced
        );
        assert_eq!(
            inner_product_bva_field_streaming(&mut source, &matmul_right_tbv)
                .unwrap()
                .value,
            matmul_proof.b_reduced
        );
        let causal_qk_routing = manifest
            .causal_qk_routing(&metadata, &layout, &bva)
            .unwrap();
        assert_eq!(
            causal_qk_routing.relations.len(),
            manifest
                .relations
                .iter()
                .filter(|relation| {
                    relation.family == RelationFamily::Bilinear
                        && decode_bilinear_relation_kind(relation.kind).unwrap()
                            == BilinearRelationKind::CausalQk
                })
                .count()
        );
        let causal_qk_projections = causal_qk_routing
            .projections::<Fr>(|| proof_transcript.challenge(b"causal-qk-routing"))
            .unwrap();
        let causal_qk_output_tbv = causal_qk_routing
            .route_output(&causal_qk_projections, &bva)
            .unwrap();
        let causal_qk_claim = inner_product_bva_field_streaming(&mut source, &causal_qk_output_tbv)
            .unwrap()
            .value;
        let (causal_qk_proof, causal_qk_counters, causal_qk_profile) = {
            let mut pair_source = source
                .causal_qk_pair_source(&causal_qk_routing, &causal_qk_projections)
                .unwrap();
            EoWitnessPairProof::prove_streaming(&mut pair_source, causal_qk_claim).unwrap()
        };
        let causal_qk_verification =
            causal_qk_proof.verify(causal_qk_routing.layout.logical_len(), causal_qk_claim);
        assert!(causal_qk_verification.accepted);
        assert_eq!(causal_qk_profile.materialized_padding_entries, 0);
        assert_eq!(causal_qk_counters.padding_ip_field_ops, 0);
        assert_eq!(causal_qk_counters.padding_fold_field_ops, 0);
        let causal_qk_left_equation = TensorBasisVector::global_tensor(
            &causal_qk_routing.layout,
            &causal_qk_verification.challenges_inv,
        )
        .unwrap();
        let causal_qk_right_equation = TensorBasisVector::global_tensor(
            &causal_qk_routing.layout,
            &causal_qk_verification.challenges,
        )
        .unwrap();
        let causal_qk_left_tbv = causal_qk_routing
            .route_terminal(&causal_qk_left_equation, &causal_qk_projections, &bva, true)
            .unwrap();
        let causal_qk_right_tbv = causal_qk_routing
            .route_terminal(
                &causal_qk_right_equation,
                &causal_qk_projections,
                &bva,
                false,
            )
            .unwrap();
        assert_eq!(
            inner_product_bva_field_streaming(&mut source, &causal_qk_left_tbv)
                .unwrap()
                .value,
            causal_qk_proof.a_reduced
        );
        assert_eq!(
            inner_product_bva_field_streaming(&mut source, &causal_qk_right_tbv)
                .unwrap()
                .value,
            causal_qk_proof.b_reduced
        );
        let causal_pv_routing = manifest
            .causal_pv_routing(&metadata, &layout, &bva)
            .unwrap();
        assert_eq!(
            causal_pv_routing.relations.len(),
            manifest
                .relations
                .iter()
                .filter(|relation| {
                    relation.family == RelationFamily::Bilinear
                        && decode_bilinear_relation_kind(relation.kind).unwrap()
                            == BilinearRelationKind::CausalPv
                })
                .count()
        );
        assert_eq!(
            bilinear_routing.covered_relations
                + matmul_routing.covered_relations
                + causal_qk_routing.relations.len()
                + causal_pv_routing.relations.len(),
            manifest
                .relations
                .iter()
                .filter(|relation| relation.family == RelationFamily::Bilinear)
                .count()
        );
        let causal_pv_projections = causal_pv_routing
            .projections::<Fr>(|| proof_transcript.challenge(b"causal-pv-routing"))
            .unwrap();
        let causal_pv_output_tbv = causal_pv_routing
            .route_output(&causal_pv_projections, &bva)
            .unwrap();
        let causal_pv_claim = inner_product_bva_field_streaming(&mut source, &causal_pv_output_tbv)
            .unwrap()
            .value;
        let (causal_pv_proof, causal_pv_counters, causal_pv_profile) = {
            let mut pair_source = source
                .causal_pv_pair_source(&causal_pv_routing, &causal_pv_projections)
                .unwrap();
            EoWitnessPairProof::prove_streaming(&mut pair_source, causal_pv_claim).unwrap()
        };
        let causal_pv_verification =
            causal_pv_proof.verify(causal_pv_routing.layout.logical_len(), causal_pv_claim);
        assert!(causal_pv_verification.accepted);
        assert_eq!(causal_pv_profile.materialized_padding_entries, 0);
        assert_eq!(causal_pv_counters.padding_ip_field_ops, 0);
        assert_eq!(causal_pv_counters.padding_fold_field_ops, 0);
        let causal_pv_left_equation = TensorBasisVector::global_tensor(
            &causal_pv_routing.layout,
            &causal_pv_verification.challenges_inv,
        )
        .unwrap();
        let causal_pv_right_equation = TensorBasisVector::global_tensor(
            &causal_pv_routing.layout,
            &causal_pv_verification.challenges,
        )
        .unwrap();
        let causal_pv_left_tbv = causal_pv_routing
            .route_terminal(&causal_pv_left_equation, &causal_pv_projections, &bva, true)
            .unwrap();
        let causal_pv_right_tbv = causal_pv_routing
            .route_terminal(
                &causal_pv_right_equation,
                &causal_pv_projections,
                &bva,
                false,
            )
            .unwrap();
        assert_eq!(
            inner_product_bva_field_streaming(&mut source, &causal_pv_left_tbv)
                .unwrap()
                .value,
            causal_pv_proof.a_reduced
        );
        assert_eq!(
            inner_product_bva_field_streaming(&mut source, &causal_pv_right_tbv)
                .unwrap()
                .value,
            causal_pv_proof.b_reduced
        );
        proof_transcript.absorb(b"io-claim", &io_claim);
        proof_transcript.absorb(b"linear-claim", &linear_claim);
        proof_transcript.absorb(b"range-claim", &range_claim);
        proof_transcript.absorb(b"lookup-left-product", &products.left);
        proof_transcript.absorb(b"lookup-right-product", &products.right);
        proof_transcript.absorb(b"lookup-leaf-claim", &lookup_claim);
        absorb_witness_pair_proof(
            &mut proof_transcript,
            b"elementwise-bilinear-proof",
            bilinear_claim,
            &bilinear_proof,
        );
        absorb_witness_pair_proof(
            &mut proof_transcript,
            b"dense-matmul-proof",
            matmul_claim,
            &matmul_proof,
        );
        absorb_witness_pair_proof(
            &mut proof_transcript,
            b"causal-qk-proof",
            causal_qk_claim,
            &causal_qk_proof,
        );
        absorb_witness_pair_proof(
            &mut proof_transcript,
            b"causal-pv-proof",
            causal_pv_claim,
            &causal_pv_proof,
        );
        let synchronized_weights: [Fr; 16] = std::array::from_fn(|_| {
            proof_transcript.challenge(b"synchronized-witness-opening-weight")
        });
        let synchronized_tbv = TensorBasisVector::linear_combination(
            &bva.layout,
            &[
                (synchronized_weights[0], &io_tbv),
                (synchronized_weights[1], &linear_tbv),
                (synchronized_weights[2], &range_tbv),
                (synchronized_weights[3], &bilinear_output_tbv),
                (synchronized_weights[4], &bilinear_left_tbv),
                (synchronized_weights[5], &bilinear_right_tbv),
                (synchronized_weights[6], &matmul_output_tbv),
                (synchronized_weights[7], &matmul_left_tbv),
                (synchronized_weights[8], &matmul_right_tbv),
                (synchronized_weights[9], &causal_qk_output_tbv),
                (synchronized_weights[10], &causal_qk_left_tbv),
                (synchronized_weights[11], &causal_qk_right_tbv),
                (synchronized_weights[12], &causal_pv_output_tbv),
                (synchronized_weights[13], &causal_pv_left_tbv),
                (synchronized_weights[14], &causal_pv_right_tbv),
                (synchronized_weights[15], &lookup_tbv),
            ],
        )
        .unwrap();
        let synchronized_claim = synchronized_weights[0] * io_claim
            + synchronized_weights[1] * linear_claim
            + synchronized_weights[2] * range_claim
            + synchronized_weights[3] * bilinear_claim
            + synchronized_weights[4] * bilinear_proof.a_reduced
            + synchronized_weights[5] * bilinear_right_claim;
        let synchronized_claim = synchronized_claim
            + synchronized_weights[6] * matmul_claim
            + synchronized_weights[7] * matmul_proof.a_reduced
            + synchronized_weights[8] * matmul_proof.b_reduced;
        let synchronized_claim = synchronized_claim
            + synchronized_weights[9] * causal_qk_claim
            + synchronized_weights[10] * causal_qk_proof.a_reduced
            + synchronized_weights[11] * causal_qk_proof.b_reduced;
        let synchronized_claim = synchronized_claim
            + synchronized_weights[12] * causal_pv_claim
            + synchronized_weights[13] * causal_pv_proof.a_reduced
            + synchronized_weights[14] * causal_pv_proof.b_reduced
            + synchronized_weights[15] * lookup_claim;
        let synchronized_evaluation =
            inner_product_bva_field_streaming(&mut source, &synchronized_tbv).unwrap();
        assert_eq!(synchronized_evaluation.value, synchronized_claim);
        let (synchronized_eo, synchronized_counters, synchronized_profile) =
            EoLiteBulletProof::prove_streaming(&mut source, &synchronized_tbv, synchronized_claim)
                .unwrap();
        let synchronized_verification =
            synchronized_eo.verify_with_tbv(&synchronized_tbv, synchronized_claim);
        assert!(synchronized_verification.accepted);
        assert_eq!(synchronized_profile.materialized_padding_entries, 0);
        assert_eq!(synchronized_counters.padding_ip_field_ops, 0);
        assert_eq!(synchronized_counters.padding_fold_field_ops, 0);
        let (synchronized_xl, synchronized_xr) = bva_eo_opening_point(
            bva.logical_entries(),
            &synchronized_verification.challenges_inv,
        )
        .unwrap();
        let synchronized_value_hiding = Fr::from(97_u64);
        let synchronized_value_commitment =
            Bls12_381::pairing(pp.g_0.mul(synchronized_eo.a_reduced), pp.h_hat)
                + pp.tilde_u.mul(synchronized_value_hiding);
        let (synchronized_opening, synchronized_opening_profile) =
            open_bva_field_streaming::<Bls12_381, _>(
                &pp,
                &mut source,
                &synchronized_xl,
                &synchronized_xr,
                synchronized_value_commitment,
                matrix_commitment,
                &tier_one,
                synchronized_value_hiding,
                matrix_hiding,
            )
            .unwrap();
        assert!(SmartPC::<Bls12_381>::verify(
            &pp,
            matrix_commitment,
            synchronized_value_commitment,
            &synchronized_xl,
            &synchronized_xr,
            &synchronized_opening,
        )
        .unwrap());
        assert_eq!(
            synchronized_opening_profile.materialized_projection_padding_entries,
            0
        );
        let bilinear_output_vector = bilinear_routing
            .vectors
            .iter()
            .find_map(|vector| vector.output_vector)
            .unwrap();
        {
            let mut tampered_source = TamperedBvaFieldSource {
                inner: &mut source,
                vector_index: bilinear_output_vector,
            };
            let tampered_bilinear_output =
                inner_product_bva_field_streaming(&mut tampered_source, &bilinear_output_tbv)
                    .unwrap();
            assert_ne!(tampered_bilinear_output.value, bilinear_claim);
        }
        let residual_output = manifest
            .relations
            .iter()
            .find(|relation| {
                relation.family == RelationFamily::Linear
                    && relation.kind == LinearRelationKind::ResidualAdd as u8
            })
            .and_then(|relation| relation.blocks.get(2))
            .copied()
            .unwrap();
        let residual_output_vector = bva
            .vectors
            .iter()
            .position(|vector| {
                matches!(
                    vector,
                    PrimitiveBvaVectorSource::Base { block, column: 0 }
                        if *block == residual_output
                )
            })
            .unwrap();
        {
            let mut tampered_source = TamperedBvaFieldSource {
                inner: &mut source,
                vector_index: residual_output_vector,
            };
            let tampered_residual =
                inner_product_bva_field_streaming(&mut tampered_source, &residual_tbv).unwrap();
            assert_ne!(tampered_residual.value, residual_claim);
        }
        {
            let mut tampered_source = TamperedBvaFieldSource {
                inner: &mut source,
                vector_index: embedding_output_vector,
            };
            let tampered_synchronized =
                inner_product_bva_field_streaming(&mut tampered_source, &synchronized_tbv).unwrap();
            assert_ne!(tampered_synchronized.value, synchronized_claim);
        }
        let semantic_lookup_output_vector = manifest
            .relations
            .iter()
            .find(|relation| {
                relation.family == RelationFamily::TaggedLookup && relation.blocks.len() == 2
            })
            .and_then(|relation| relation.blocks.get(1))
            .and_then(|block| {
                bva.vectors.iter().position(|vector| {
                    matches!(
                        vector,
                        PrimitiveBvaVectorSource::Auxiliary {
                            block_index,
                            column: 0
                        } if manifest.auxiliary_blocks[*block_index].id == *block
                    )
                })
            })
            .unwrap();
        {
            let mut tampered_source = TamperedBvaFieldSource {
                inner: &mut source,
                vector_index: semantic_lookup_output_vector,
            };
            let tampered_lookup =
                inner_product_bva_field_streaming(&mut tampered_source, &lookup_tbv).unwrap();
            assert_ne!(tampered_lookup.value, lookup_claim);
        }
        let range_limb_vector = bva
            .vectors
            .iter()
            .position(|vector| matches!(vector, PrimitiveBvaVectorSource::RangeLimb { .. }))
            .unwrap();
        let mut tampered_source = TamperedBvaFieldSource {
            inner: &mut source,
            vector_index: range_limb_vector,
        };
        let tampered_range =
            inner_product_bva_field_streaming(&mut tampered_source, &range_tbv).unwrap();
        assert_ne!(tampered_range.value, range_claim);
    }

    fn tiny_checkpoint() -> Gpt2Q12Checkpoint<Cursor<Vec<u8>>> {
        let model = Gpt2Model {
            blocks: 1,
            hidden_width: 2,
            attention_heads: 1,
            mlp_width: 4,
            vocabulary: 3,
            maximum_positions: 2,
        };
        let parameter_entries = model.physical_parameter_entries().unwrap();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"PR1GPT2\0");
        bytes.extend_from_slice(&2_u32.to_le_bytes());
        bytes.extend_from_slice(&12_u32.to_le_bytes());
        bytes.extend_from_slice(&(parameter_entries as u64).to_le_bytes());
        for value in [
            2_u32,
            model.vocabulary as u32,
            model.maximum_positions as u32,
            model.hidden_width as u32,
            model.attention_heads as u32,
            model.mlp_width as u32,
            model.blocks as u32,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&[0x24; 32]);
        bytes.extend_from_slice(&[0x84; 32]);
        for token in [0_u32, 1] {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        for logit in [-4_094_i64, 4_094, 0] {
            bytes.extend_from_slice(&logit.to_le_bytes());
        }
        let mut parameters = vec![0_i32; parameter_entries];
        parameters[..6].copy_from_slice(&[4_096, 0, -4_096, 0, 4_096, -4_096]);
        parameters[64..66].copy_from_slice(&[4_096, 4_096]);
        for value in parameters {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        Gpt2Q12Checkpoint::from_reader(Cursor::new(bytes)).unwrap()
    }

    #[test]
    #[ignore = "requires ZKSMART_GPT2_Q12_CHECKPOINT and the 475 MiB real bundle"]
    fn real_gpt2_small_range_shape_compiles_without_building_the_witness() {
        let path =
            std::env::var("ZKSMART_GPT2_Q12_CHECKPOINT").expect("set ZKSMART_GPT2_Q12_CHECKPOINT");
        let file = File::open(path).unwrap();
        let checkpoint =
            Gpt2Q12Checkpoint::from_reader(BufReader::with_capacity(8 << 20, file)).unwrap();
        let layout = Gpt2GlobalLayout::compile(checkpoint.metadata()).unwrap();
        let manifest =
            Gpt2PrimitiveRelationManifest::compile(checkpoint.metadata(), &layout).unwrap();
        let shape = manifest.shape(&layout).unwrap();

        println!("linear_relations={}", shape.linear_relations);
        println!("bilinear_relations={}", shape.bilinear_relations);
        println!("tagged_lookup_relations={}", shape.tagged_lookup_relations);
        println!("io_equation_scalar_rows={}", shape.io_equation_scalar_rows);
        println!(
            "linear_equation_scalar_rows={}",
            shape.linear_equation_scalar_rows
        );
        println!(
            "bilinear_equation_scalar_rows={}",
            shape.bilinear_equation_scalar_rows
        );
        println!(
            "tagged_lookup_equation_scalar_rows={}",
            shape.tagged_lookup_equation_scalar_rows
        );
        println!(
            "equation_bva_logical_rows={}",
            shape.equation_bva_logical_rows
        );
        println!(
            "maximum_live_equation_row_entries={}",
            shape.maximum_live_equation_row_entries
        );
        println!(
            "resident_auxiliary_blocks={}",
            shape.resident_auxiliary_blocks
        );
        println!(
            "resident_auxiliary_entries={}",
            shape.resident_auxiliary_entries
        );
        println!(
            "streamed_range_limb_blocks={}",
            shape.streamed_range_limb_blocks
        );
        println!(
            "streamed_range_limb_entries={}",
            shape.streamed_range_limb_entries
        );
        println!(
            "maximum_streamed_range_limb_column_entries={}",
            shape.maximum_streamed_range_limb_column_entries
        );
        println!("global_witness_blocks={}", shape.global_witness_blocks);
        println!("global_physical_entries={}", shape.global_physical_entries);
        println!(
            "global_bva_logical_entries={}",
            shape.global_bva_logical_entries
        );
        assert_eq!(shape.materialized_padding_entries, 0);
        assert_eq!(shape.equation_materialized_padding_entries, 0);
    }

    #[test]
    #[ignore = "requires ZKSMART_GPT2_Q12_CHECKPOINT and the 475 MiB real bundle"]
    fn real_gpt2_small_auxiliary_witness_matches_all_compiled_relations() {
        let path =
            std::env::var("ZKSMART_GPT2_Q12_CHECKPOINT").expect("set ZKSMART_GPT2_Q12_CHECKPOINT");
        let file = File::open(path).unwrap();
        let mut checkpoint =
            Gpt2Q12Checkpoint::from_reader(BufReader::with_capacity(8 << 20, file)).unwrap();
        let layout = Gpt2GlobalLayout::compile(checkpoint.metadata()).unwrap();
        let execute_start = Instant::now();
        let (execution, activations) = checkpoint.execute_activation_witness(&layout).unwrap();
        println!(
            "reference_execute_seconds={:.6}",
            execute_start.elapsed().as_secs_f64()
        );
        assert_eq!(execution.logits, checkpoint.reference_logits());
        let manifest =
            Gpt2PrimitiveRelationManifest::compile(checkpoint.metadata(), &layout).unwrap();
        let build_start = Instant::now();
        let witness =
            Gpt2PrimitiveWitness::build(&mut checkpoint, &layout, &activations, &manifest).unwrap();
        println!(
            "aux_build_seconds={:.6}",
            build_start.elapsed().as_secs_f64()
        );
        let shape = manifest.shape(&layout).unwrap();
        println!("auxiliary_blocks={}", witness.block_count());
        println!("auxiliary_entries={}", witness.stored_entries());
        println!("global_physical_entries={}", shape.global_physical_entries);
        println!(
            "global_bva_logical_entries={}",
            shape.global_bva_logical_entries
        );
        let queries = witness
            .lookup_queries(&layout, &activations, &manifest)
            .unwrap();
        Gpt2LookupRegistry::exact(checkpoint.metadata())
            .audit(&queries)
            .unwrap();
        println!("authenticated_lookup_queries={}", queries.len());
        println!("relation_witness_match=true");
    }
}
