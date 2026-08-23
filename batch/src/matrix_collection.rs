//! Section 6 matrix-claim collection compiled into one dense byte BVA.
//!
//! Matrix entries and their zero padding occupy one contiguous `Vec<i8>` in
//! sorted bucket order. Public matrix MLE claims are deterministically compiled
//! into a compact TBV after the dense BVA commitment fixes the collection
//! batching challenge.

use std::{
    convert::Infallible,
    fmt::{Display, Formatter},
};

use ark_bn254::Fr;
use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use bincode::Options;
use jolt_dory::DoryCommitment;
use rayon::prelude::*;
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    bucketed_ip::{
        BvaError, BvaI16Source, BvaI8Geometry, BvaI8Source, BvaInteger, BvaLayout,
        EoLiteBulletVerification, TbvBucket, TensorBasisVector,
    },
    fiat_shamir::FiatShamir,
    jolt_dory::{
        commit_bva_i16_streaming, commit_bva_i8_streaming, commit_field_vector_zk,
        commitment_binding_bytes, evaluate_field_vector_standard, open_field_vectors_batch_zk,
        packed_i8_parcomm_statement_binding, prove_packed_i16_parcomm_precommitted,
        prove_packed_i8_parcomm_precommitted, verify_field_vectors_batch_zk,
        verify_packed_i16_bva_opening_as_subprotocol, verify_packed_i16_parcomm_as_subprotocol,
        verify_packed_i8_tbv_inner_product, CommittedDoryBva, CommittedDoryFieldVector,
        DenseI16BvaSource, DenseI8BvaSource, DoryBackendError, DoryCommitProfile,
        DoryProverParameters, DoryVerifierParameters, FieldVectorDoryProof, PackedI8ParCommProfile,
        PackedI8ParCommProof, PackedI8ParCommProofDecodeError, PackedI8ParCommProverError,
        PackedI8ParCommVerificationError, TerminalDoryVerification,
    },
};

/// A dense column-major matrix backed by one short-integer type.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseShortIntMatrix<Integer: BvaInteger> {
    shape: (usize, usize),
    values: Vec<Integer>,
}

pub type DenseI8Matrix = DenseShortIntMatrix<i8>;
pub type DenseI16Matrix = DenseShortIntMatrix<i16>;

impl<Integer: BvaInteger> DenseShortIntMatrix<Integer> {
    pub fn new(shape: (usize, usize), values: Vec<Integer>) -> Result<Self, MatrixCollectionError> {
        validate_shape(0, shape)?;
        let expected = shape
            .0
            .checked_mul(shape.1)
            .ok_or(MatrixCollectionError::LengthOverflow)?;
        if values.len() != expected {
            return Err(MatrixCollectionError::MatrixLength {
                matrix: 0,
                expected,
                actual: values.len(),
            });
        }
        Ok(Self { shape, values })
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape
    }

    pub fn values(&self) -> &[Integer] {
        &self.values
    }
}

/// One public matrix MLE claim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixMleClaim<F: PrimeField> {
    pub shape: (usize, usize),
    /// Row point in low-bit-first reverse-tensor order.
    pub left_point: Vec<F>,
    /// Column point in low-bit-first reverse-tensor order.
    pub right_point: Vec<F>,
    pub value: F,
}

impl<F: PrimeField> MatrixMleClaim<F> {
    pub fn new(
        shape: (usize, usize),
        left_point: Vec<F>,
        right_point: Vec<F>,
        value: F,
    ) -> Result<Self, MatrixCollectionError> {
        validate_shape(0, shape)?;
        validate_point_lengths(0, shape, left_point.len(), right_point.len())?;
        Ok(Self {
            shape,
            left_point,
            right_point,
            value,
        })
    }
}

/// Conceptual metadata for one aligned dyadic rectangle in dense BVA order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixBucketDescriptor {
    matrix: usize,
    row_offset: usize,
    row_len: usize,
    physical_row_len: usize,
    col_offset: usize,
    col_len: usize,
    physical_col_len: usize,
    dense_offset: usize,
    source_offset: usize,
}

impl MatrixBucketDescriptor {
    pub fn matrix(&self) -> usize {
        self.matrix
    }

    pub fn row_offset(&self) -> usize {
        self.row_offset
    }

    pub fn row_len(&self) -> usize {
        self.row_len
    }

    pub fn col_offset(&self) -> usize {
        self.col_offset
    }

    pub fn col_len(&self) -> usize {
        self.col_len
    }

    pub fn physical_row_len(&self) -> usize {
        self.physical_row_len
    }

    pub fn physical_col_len(&self) -> usize {
        self.physical_col_len
    }

    pub fn dense_offset(&self) -> usize {
        self.dense_offset
    }

    pub fn source_offset(&self) -> usize {
        self.source_offset
    }

    pub fn entry_count(&self) -> usize {
        self.row_len * self.col_len
    }

    pub fn physical_entry_count(&self) -> usize {
        self.physical_row_len * self.physical_col_len
    }

    fn row_log(&self) -> usize {
        self.row_len.ilog2() as usize
    }

    fn col_log(&self) -> usize {
        self.col_len.ilog2() as usize
    }
}

/// Verifier-visible dense BVA layout and matrix-to-bucket provenance.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseBvaDescriptor {
    matrix_shapes: Vec<(usize, usize)>,
    buckets: Vec<MatrixBucketDescriptor>,
    layout: BvaLayout,
    physical_len: usize,
}

impl DenseBvaDescriptor {
    pub const SPLITTING_BUDGET: usize = 4;

    pub fn from_shapes(shapes: &[(usize, usize)]) -> Result<Self, MatrixCollectionError> {
        if shapes.is_empty() {
            return Err(MatrixCollectionError::EmptyCollection);
        }

        let mut buckets = Vec::new();
        for (matrix, &shape) in shapes.iter().enumerate() {
            validate_shape(matrix, shape)?;
            let row_intervals = capped_aligned_dyadic_intervals(shape.0, Self::SPLITTING_BUDGET);
            let col_intervals = capped_aligned_dyadic_intervals(shape.1, Self::SPLITTING_BUDGET);
            for &(row_offset, row_len) in &row_intervals {
                for &(col_offset, col_len) in &col_intervals {
                    buckets.push(MatrixBucketDescriptor {
                        matrix,
                        row_offset,
                        row_len,
                        physical_row_len: (shape.0 - row_offset).min(row_len),
                        col_offset,
                        col_len,
                        physical_col_len: (shape.1 - col_offset).min(col_len),
                        dense_offset: 0,
                        source_offset: 0,
                    });
                }
            }
        }

        buckets.sort_by(|left, right| {
            right
                .entry_count()
                .cmp(&left.entry_count())
                .then_with(|| left.matrix.cmp(&right.matrix))
                .then_with(|| left.row_offset.cmp(&right.row_offset))
                .then_with(|| left.col_offset.cmp(&right.col_offset))
        });

        let mut active_len = 0_usize;
        let mut physical_len = 0_usize;
        for bucket in &mut buckets {
            bucket.dense_offset = active_len;
            bucket.source_offset = physical_len;
            active_len = active_len
                .checked_add(bucket.entry_count())
                .ok_or(MatrixCollectionError::LengthOverflow)?;
            physical_len = physical_len
                .checked_add(bucket.physical_entry_count())
                .ok_or(MatrixCollectionError::LengthOverflow)?;
        }
        let lengths = buckets
            .iter()
            .map(MatrixBucketDescriptor::entry_count)
            .collect::<Vec<_>>();
        let layout = BvaLayout::from_lengths(&lengths)?;
        if layout.active_len() != active_len
            || layout.vector_lengths() != lengths
            || layout
                .vector_to_bucket()
                .iter()
                .copied()
                .ne(0..lengths.len())
        {
            return Err(MatrixCollectionError::NonCanonicalLayout);
        }

        Ok(Self {
            matrix_shapes: shapes.to_vec(),
            buckets,
            layout,
            physical_len,
        })
    }

    pub fn matrix_count(&self) -> usize {
        self.matrix_shapes.len()
    }

    pub fn matrix_shapes(&self) -> &[(usize, usize)] {
        &self.matrix_shapes
    }

    pub fn buckets(&self) -> &[MatrixBucketDescriptor] {
        &self.buckets
    }

    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    pub fn physical_len(&self) -> usize {
        self.physical_len
    }

    pub fn logical_len(&self) -> usize {
        self.layout.logical_len()
    }

    pub fn conceptual_final_zeros(&self) -> usize {
        self.logical_len() - self.layout.active_len()
    }

    pub fn conceptual_boundary_zeros(&self) -> usize {
        self.layout.active_len() - self.physical_len
    }

    pub fn conceptual_zero_entries(&self) -> usize {
        self.logical_len() - self.physical_len
    }

    pub fn materialized_padding_entries(&self) -> usize {
        self.conceptual_zero_entries()
    }

    pub fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    pub fn required_dory_log(&self) -> usize {
        self.logical_len().ilog2() as usize
    }

    /// Maximum Dory variable count across the dense BVA and both padded shape
    /// tables used by registration.
    pub fn required_registration_dory_log(&self) -> usize {
        let padded_rows = self.bucket_count().next_power_of_two();
        self.required_dory_log()
            .max((padded_rows * 4).ilog2() as usize)
    }

    pub fn bucket_ranges_are_a_partition(&self) -> bool {
        let mut expected_logical = 0_usize;
        let mut expected_physical = 0_usize;
        for bucket in &self.buckets {
            if bucket.dense_offset != expected_logical || bucket.source_offset != expected_physical
            {
                return false;
            }
            let Some(next_logical) = expected_logical.checked_add(bucket.entry_count()) else {
                return false;
            };
            let Some(next_physical) = expected_physical.checked_add(bucket.physical_entry_count())
            else {
                return false;
            };
            expected_logical = next_logical;
            expected_physical = next_physical;
        }
        expected_logical == self.layout.active_len() && expected_physical == self.physical_len
    }

    fn validate_canonical(&self) -> Result<(), MatrixCollectionError> {
        if Self::from_shapes(&self.matrix_shapes)? != *self {
            return Err(MatrixCollectionError::NonCanonicalLayout);
        }
        Ok(())
    }

    /// Compile public matrix claims using an already sampled batching challenge.
    pub fn compile_with_omega<F: PrimeField>(
        &self,
        claims: &[MatrixMleClaim<F>],
        omega: F,
        bit_width: usize,
    ) -> Result<CompiledMatrixClaims<F>, MatrixCollectionError> {
        self.validate_claims(claims)?;
        let powers = omega_powers(omega, claims.len());
        let buckets = self
            .buckets
            .iter()
            .map(|bucket| {
                let claim = &claims[bucket.matrix];
                let mut eta = powers[bucket.matrix];
                eta *= fixed_offset_factor(bucket.col_offset, bucket.col_log(), &claim.right_point);
                eta *= fixed_offset_factor(bucket.row_offset, bucket.row_log(), &claim.left_point);
                let mut local_point = Vec::with_capacity(bucket.row_log() + bucket.col_log());
                local_point.extend_from_slice(&claim.right_point[..bucket.col_log()]);
                local_point.extend_from_slice(&claim.left_point[..bucket.row_log()]);
                TbvBucket::new(eta, local_point)
            })
            .collect::<Vec<_>>();
        let tbv = TensorBasisVector::new(&self.layout, buckets)?;
        let claimed = claims
            .iter()
            .zip(&powers)
            .map(|(claim, power)| claim.value * power)
            .sum();
        Ok(CompiledMatrixClaims {
            omega,
            claimed,
            tbv,
            bit_width,
        })
    }

    /// Derive `omega` after `C_a` and all public claims are fixed, then compile
    /// the collection into the compact BVA--TBV statement.
    pub fn compile_for_proof(
        &self,
        matrix_commitment: &DoryCommitment,
        claims: &[MatrixMleClaim<Fr>],
        bit_width: usize,
    ) -> Result<CompiledMatrixClaims<Fr>, MatrixCollectionError> {
        self.validate_claims(claims)?;
        let omega = self.batching_challenge(matrix_commitment, None, claims, bit_width);
        self.compile_with_omega(claims, omega, bit_width)
    }

    /// Compile after both persistent shape commitments and the dense BVA
    /// commitment have been fixed.
    pub fn compile_for_registered_proof(
        &self,
        commitments: &MatrixCollectionCommitments,
        claims: &[MatrixMleClaim<Fr>],
        bit_width: usize,
    ) -> Result<CompiledMatrixClaims<Fr>, MatrixCollectionError> {
        self.validate_claims(claims)?;
        let omega = self.batching_challenge(
            &commitments.bva,
            Some(&commitments.shape),
            claims,
            bit_width,
        );
        self.compile_with_omega(claims, omega, bit_width)
    }

    fn validate_claims<F: PrimeField>(
        &self,
        claims: &[MatrixMleClaim<F>],
    ) -> Result<(), MatrixCollectionError> {
        if claims.len() != self.matrix_shapes.len() {
            return Err(MatrixCollectionError::ClaimCount {
                expected: self.matrix_shapes.len(),
                actual: claims.len(),
            });
        }
        for (matrix, (claim, &shape)) in claims.iter().zip(&self.matrix_shapes).enumerate() {
            if claim.shape != shape {
                return Err(MatrixCollectionError::ClaimShape {
                    matrix,
                    expected: shape,
                    actual: claim.shape,
                });
            }
            validate_point_lengths(
                matrix,
                shape,
                claim.left_point.len(),
                claim.right_point.len(),
            )?;
        }
        Ok(())
    }

    fn batching_challenge(
        &self,
        matrix_commitment: &DoryCommitment,
        shape_commitments: Option<&ShapeCommitments>,
        claims: &[MatrixMleClaim<Fr>],
        bit_width: usize,
    ) -> Fr {
        let mut transcript = FiatShamir::<Fr>::new();
        transcript.push(&0x4d43_4c41_494d_5331_u64);
        transcript.push(&(bit_width as u64));
        match shape_commitments {
            Some(shape) => {
                transcript.push(&1_u64);
                transcript.push_bytes(&commitment_binding_bytes(&shape.alpha));
                transcript.push_bytes(&commitment_binding_bytes(&shape.gamma));
            }
            None => transcript.push(&0_u64),
        }
        transcript.push_bytes(&commitment_binding_bytes(matrix_commitment));
        transcript.push(&(claims.len() as u64));
        for claim in claims {
            transcript.push(&(claim.shape.0 as u64));
            transcript.push(&(claim.shape.1 as u64));
            transcript.push(&(claim.left_point.len() as u64));
            for coordinate in &claim.left_point {
                transcript.push(coordinate);
            }
            transcript.push(&(claim.right_point.len() as u64));
            for coordinate in &claim.right_point {
                transcript.push(coordinate);
            }
            transcript.push(&claim.value);
        }
        transcript.push(&(self.logical_len() as u64));
        transcript.push(&(self.physical_len as u64));
        transcript.push(&(self.buckets.len() as u64));
        for bucket in &self.buckets {
            transcript.push(&(bucket.matrix as u64));
            transcript.push(&(bucket.row_offset as u64));
            transcript.push(&(bucket.row_len as u64));
            transcript.push(&(bucket.col_offset as u64));
            transcript.push(&(bucket.col_len as u64));
            transcript.push(&(bucket.dense_offset as u64));
        }
        transcript.gen_challenge()
    }
}

/// Persistent commitments to the Section 6 shape metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShapeCommitments {
    pub alpha: DoryCommitment,
    pub gamma: DoryCommitment,
}

/// Persistent registration data shared by every opening of one collection.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixCollectionCommitments {
    pub shape: ShapeCommitments,
    pub bva: DoryCommitment,
}

/// Size counters for the committed shape tables.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ShapeCommitProfile {
    pub bucket_rows: usize,
    pub padded_rows: usize,
    pub alpha_field_entries: usize,
    pub gamma_field_entries: usize,
}

/// Counters for the ordered shape-then-BVA registration phase.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct MatrixCollectionCommitProfile {
    pub shape: ShapeCommitProfile,
    pub bva: DoryCommitProfile,
}

/// Prover-only state after the shape commitments have been fixed.
pub struct CommittedMatrixShapes {
    descriptor: DenseBvaDescriptor,
    alpha_values: Vec<Fr>,
    gamma_values: Vec<Fr>,
    alpha: CommittedDoryFieldVector,
    gamma: CommittedDoryFieldVector,
    profile: ShapeCommitProfile,
}

impl CommittedMatrixShapes {
    pub fn commitments(&self) -> ShapeCommitments {
        ShapeCommitments {
            alpha: self.alpha.commitment.clone(),
            gamma: self.gamma.commitment.clone(),
        }
    }

    pub fn profile(&self) -> ShapeCommitProfile {
        self.profile
    }
}

/// Prover-only state after shape and dense BVA commitments are fixed.
pub struct RegisteredMatrixCollection {
    descriptor: DenseBvaDescriptor,
    bit_width: usize,
    commitments: MatrixCollectionCommitments,
    alpha_values: Vec<Fr>,
    gamma_values: Vec<Fr>,
    alpha: CommittedDoryFieldVector,
    gamma: CommittedDoryFieldVector,
    bva: CommittedDoryBva,
    profile: MatrixCollectionCommitProfile,
}

impl RegisteredMatrixCollection {
    pub fn commitments(&self) -> &MatrixCollectionCommitments {
        &self.commitments
    }

    pub fn profile(&self) -> MatrixCollectionCommitProfile {
        self.profile
    }
}

/// Dory openings that bind the persistent shape commitments to the descriptor.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShapePointConsistencyProof {
    pub matrices: FieldVectorDoryProof,
}

/// One registered matrix-collection opening.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RegisteredMatrixCollectionProof {
    pub packed: PackedI8ParCommProof,
    pub shape_consistency: ShapePointConsistencyProof,
}

const REGISTERED_PROOF_MAGIC: [u8; 8] = *b"JLRGTR02";
const MAX_REGISTERED_COMPONENT_BYTES: usize = 3 * 768 * 1024;

impl RegisteredMatrixCollectionProof {
    pub fn to_bytes(&self) -> Result<Vec<u8>, RegisteredMatrixCollectionProofDecodeError> {
        let packed = self.packed.to_bytes()?;
        let matrices = serialize_registered_component(&self.shape_consistency.matrices)?;
        let mut bytes = Vec::with_capacity(
            REGISTERED_PROOF_MAGIC.len()
                + 2 * std::mem::size_of::<u64>()
                + packed.len()
                + matrices.len(),
        );
        bytes.extend_from_slice(&REGISTERED_PROOF_MAGIC);
        append_registered_component(&mut bytes, &packed);
        append_registered_component(&mut bytes, &matrices);
        Ok(bytes)
    }

    pub fn from_bytes_exact(
        bytes: &[u8],
    ) -> Result<Self, RegisteredMatrixCollectionProofDecodeError> {
        let mut cursor = 0_usize;
        if take_registered_bytes(bytes, &mut cursor, REGISTERED_PROOF_MAGIC.len())?
            != REGISTERED_PROOF_MAGIC
        {
            return Err(RegisteredMatrixCollectionProofDecodeError::InvalidMagic);
        }
        let packed = take_registered_component(bytes, &mut cursor)?;
        let matrices = take_registered_component(bytes, &mut cursor)?;
        if cursor != bytes.len() {
            return Err(RegisteredMatrixCollectionProofDecodeError::TrailingBytes);
        }
        Ok(Self {
            packed: PackedI8ParCommProof::from_bytes_exact(packed)?,
            shape_consistency: ShapePointConsistencyProof {
                matrices: deserialize_registered_component(matrices)?,
            },
        })
    }
}

/// State returned after the BVA opening has been verified and before the TBV
/// and shape/point checks run.
pub struct RegisteredBvaOpeningVerification {
    pub compiled: CompiledMatrixClaims<Fr>,
    pub eo: EoLiteBulletVerification<Fr>,
}

/// Explicit terminal results in the order exercised by the end-to-end test.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct RegisteredMatrixCollectionVerification {
    pub bva_opening_accepted: bool,
    pub tbv_inner_product_accepted: bool,
    pub shape_point_consistency_accepted: bool,
}

impl RegisteredMatrixCollectionVerification {
    pub fn evaluate(&self) -> bool {
        self.bva_opening_accepted
            && self.tbv_inner_product_accepted
            && self.shape_point_consistency_accepted
    }
}

/// The one-allocation prover witness with materialized dyadic-bucket padding.
pub struct DenseMatrixBva<Integer: BvaInteger = i8> {
    descriptor: DenseBvaDescriptor,
    values: Vec<Integer>,
}

impl<Integer: BvaInteger> DenseMatrixBva<Integer> {
    /// Adopt an already converted dense BVA without widening its entries.
    ///
    /// The converter materializes boundary padding and the final suffix, so
    /// every padded entry is checked before the witness is accepted.
    pub fn from_dense_values(
        descriptor: DenseBvaDescriptor,
        values: Vec<Integer>,
    ) -> Result<Self, MatrixCollectionError> {
        descriptor.validate_canonical()?;
        let expected = descriptor.logical_len();
        if values.len() != expected {
            return Err(MatrixCollectionError::PackedBvaLength {
                expected,
                actual: values.len(),
            });
        }
        let invalid_boundary = descriptor.buckets.par_iter().find_map_any(|bucket| {
            let start = bucket.dense_offset;
            let bucket_values = &values[start..start + bucket.entry_count()];
            for local_row in 0..bucket.row_len {
                let row =
                    &bucket_values[local_row * bucket.col_len..(local_row + 1) * bucket.col_len];
                let physical_cols = if local_row < bucket.physical_row_len {
                    bucket.physical_col_len
                } else {
                    0
                };
                if let Some(local_col) = row[physical_cols..]
                    .iter()
                    .position(|value| !value.is_zero())
                {
                    return Some(start + local_row * bucket.col_len + physical_cols + local_col);
                }
            }
            None
        });
        let invalid_suffix = values[descriptor.layout.active_len()..]
            .par_iter()
            .position_any(|value| !value.is_zero())
            .map(|offset| descriptor.layout.active_len() + offset);
        if let Some(index) = invalid_boundary.or(invalid_suffix) {
            return Err(MatrixCollectionError::NonZeroPackedPadding { index });
        }
        Ok(Self { descriptor, values })
    }

    /// Adopt a converted BVA in canonical dense-bucket order.
    pub fn from_prepacked(
        shapes: &[(usize, usize)],
        values: Vec<Integer>,
    ) -> Result<Self, MatrixCollectionError> {
        Self::from_dense_values(DenseBvaDescriptor::from_shapes(shapes)?, values)
    }

    pub fn from_matrices(
        matrices: &[DenseShortIntMatrix<Integer>],
    ) -> Result<Self, MatrixCollectionError> {
        if matrices.is_empty() {
            return Err(MatrixCollectionError::EmptyCollection);
        }
        for (matrix, dense) in matrices.iter().enumerate() {
            validate_shape(matrix, dense.shape)?;
            let expected = dense
                .shape
                .0
                .checked_mul(dense.shape.1)
                .ok_or(MatrixCollectionError::LengthOverflow)?;
            if dense.values.len() != expected {
                return Err(MatrixCollectionError::MatrixLength {
                    matrix,
                    expected,
                    actual: dense.values.len(),
                });
            }
        }
        let shapes = matrices
            .iter()
            .map(DenseShortIntMatrix::shape)
            .collect::<Vec<_>>();
        Self::from_short_fn(&shapes, |matrix, row, col| {
            let dense = &matrices[matrix];
            dense.values[col * dense.shape.0 + row]
        })
    }

    /// Build the final dense BVA directly from matrix entries.
    ///
    /// This path allocates exactly one `logical_len` short-integer vector. Every sorted
    /// bucket occupies its full padded rectangle and the final BVA suffix is
    /// stored as zero in the same allocation.
    pub fn from_short_fn<ValueAt>(
        shapes: &[(usize, usize)],
        value_at: ValueAt,
    ) -> Result<Self, MatrixCollectionError>
    where
        ValueAt: Fn(usize, usize, usize) -> Integer + Sync,
    {
        let descriptor = DenseBvaDescriptor::from_shapes(shapes)?;
        let mut values = vec![Integer::from(false); descriptor.logical_len()];
        fill_bucket_range(
            &mut values[..descriptor.layout.active_len()],
            &descriptor.buckets,
            &value_at,
        );
        Ok(Self { descriptor, values })
    }

    pub fn descriptor(&self) -> &DenseBvaDescriptor {
        &self.descriptor
    }

    pub fn values(&self) -> &[Integer] {
        &self.values
    }

    pub fn values_mut(&mut self) -> &mut [Integer] {
        &mut self.values
    }

    /// Evaluate one original matrix directly from the prepacked BVA.
    pub fn matrix_mle_evaluate<F: PrimeField>(
        &self,
        matrix: usize,
        left_point: &[F],
        right_point: &[F],
    ) -> Result<F, MatrixCollectionError> {
        let &shape = self.descriptor.matrix_shapes.get(matrix).ok_or(
            MatrixCollectionError::MatrixIndex {
                matrix,
                count: self.descriptor.matrix_count(),
            },
        )?;
        validate_point_lengths(matrix, shape, left_point.len(), right_point.len())?;
        let left = tensor_weights(left_point);
        let right = tensor_weights(right_point);
        Ok(self
            .descriptor
            .buckets
            .par_iter()
            .filter(|bucket| bucket.matrix == matrix)
            .map(|bucket| {
                let start = bucket.dense_offset;
                let bucket_values = &self.values[start..start + bucket.entry_count()];
                bucket_values[..bucket.physical_row_len * bucket.col_len]
                    .par_chunks(bucket.col_len)
                    .enumerate()
                    .map(|(local_row, row)| {
                        let row_weight = left[bucket.row_offset + local_row];
                        row[..bucket.physical_col_len]
                            .iter()
                            .enumerate()
                            .map(|(local_col, &value)| {
                                value.to_field::<F>()
                                    * right[bucket.col_offset + local_col]
                                    * row_weight
                            })
                            .sum::<F>()
                    })
                    .sum::<F>()
            })
            .sum())
    }

    /// Independently evaluate the padded BVA at one global MLE point.
    pub fn evaluate_dense_at(&self, point: &[Fr]) -> Result<Fr, MatrixCollectionError> {
        if point.len() != self.descriptor.required_dory_log() {
            return Err(MatrixCollectionError::DenseEvaluation(format!(
                "point length {} does not match BVA log {}",
                point.len(),
                self.descriptor.required_dory_log()
            )));
        }
        Ok(self
            .descriptor
            .buckets
            .par_iter()
            .map(|bucket| {
                let bucket_log = bucket.row_log() + bucket.col_log();
                let fixed = fixed_offset_factor(bucket.dense_offset, bucket_log, point);
                let col_weights = tensor_weights(&point[..bucket.col_log()]);
                let row_weights = tensor_weights(&point[bucket.col_log()..bucket_log]);
                let start = bucket.dense_offset;
                let end = start + bucket.entry_count();
                self.values[start..end]
                    .par_chunks(bucket.col_len)
                    .zip(&row_weights)
                    .map(|(row, row_weight)| {
                        let row_sum = row
                            .iter()
                            .zip(&col_weights)
                            .map(|(&value, col_weight)| value.to_field::<Fr>() * col_weight)
                            .sum::<Fr>();
                        row_sum * row_weight
                    })
                    .sum::<Fr>()
                    * fixed
            })
            .sum())
    }

    /// Independently check a derived BVA claim against the dense witness.
    ///
    /// This is an optional test oracle. Shape and evaluation-point consistency
    /// is public metadata work and does not call this method.
    pub fn check_dense_bva_claim(
        &self,
        claim: &BvaMleClaim<Fr>,
    ) -> Result<bool, MatrixCollectionError> {
        Ok(self.evaluate_dense_at(&claim.point)? == claim.value)
    }
}

impl DenseMatrixBva<i8> {
    pub fn from_prepacked_i8(
        shapes: &[(usize, usize)],
        values: Vec<i8>,
    ) -> Result<Self, MatrixCollectionError> {
        Self::from_prepacked(shapes, values)
    }

    pub fn from_fn<ValueAt>(
        shapes: &[(usize, usize)],
        value_at: ValueAt,
    ) -> Result<Self, MatrixCollectionError>
    where
        ValueAt: Fn(usize, usize, usize) -> i8 + Sync,
    {
        Self::from_short_fn(shapes, value_at)
    }
}

impl DenseMatrixBva<i16> {
    pub fn from_prepacked_i16(
        shapes: &[(usize, usize)],
        values: Vec<i16>,
    ) -> Result<Self, MatrixCollectionError> {
        Self::from_prepacked(shapes, values)
    }

    pub fn from_i16_fn<ValueAt>(
        shapes: &[(usize, usize)],
        value_at: ValueAt,
    ) -> Result<Self, MatrixCollectionError>
    where
        ValueAt: Fn(usize, usize, usize) -> i16 + Sync,
    {
        Self::from_short_fn(shapes, value_at)
    }
}

impl BvaI8Source for DenseMatrixBva<i8> {
    type Error = Infallible;

    fn layout(&self) -> &BvaLayout {
        &self.descriptor.layout
    }

    fn vector_geometry(&self, vector_index: usize) -> Option<BvaI8Geometry> {
        let bucket = self.descriptor.buckets.get(vector_index)?;
        BvaI8Geometry::rectangle(
            bucket.row_len,
            bucket.col_len,
            bucket.row_len,
            bucket.col_len,
        )
    }

    fn materialized_padding_entries(&self) -> usize {
        self.descriptor.materialized_padding_entries()
    }

    fn indexed_vectors(&self) -> Option<Vec<&[i8]>> {
        Some(
            self.descriptor
                .buckets
                .iter()
                .map(|bucket| {
                    let start = bucket.dense_offset;
                    &self.values[start..start + bucket.entry_count()]
                })
                .collect(),
        )
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[i8])) -> Result<(), Self::Error> {
        for (vector, bucket) in self.descriptor.buckets.iter().enumerate() {
            let start = bucket.dense_offset;
            let end = start + bucket.entry_count();
            visit(vector, &self.values[start..end]);
        }
        Ok(())
    }
}

impl DenseI8BvaSource for DenseMatrixBva<i8> {
    fn dense_values(&self) -> &[i8] {
        &self.values
    }
}

impl BvaI16Source for DenseMatrixBva<i16> {
    type Error = Infallible;

    fn layout(&self) -> &BvaLayout {
        &self.descriptor.layout
    }

    fn vector_geometry(&self, vector_index: usize) -> Option<BvaI8Geometry> {
        let bucket = self.descriptor.buckets.get(vector_index)?;
        BvaI8Geometry::rectangle(
            bucket.row_len,
            bucket.col_len,
            bucket.row_len,
            bucket.col_len,
        )
    }

    fn materialized_padding_entries(&self) -> usize {
        self.descriptor.materialized_padding_entries()
    }

    fn indexed_vectors(&self) -> Option<Vec<&[i16]>> {
        Some(
            self.descriptor
                .buckets
                .iter()
                .map(|bucket| {
                    let start = bucket.dense_offset;
                    &self.values[start..start + bucket.entry_count()]
                })
                .collect(),
        )
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[i16])) -> Result<(), Self::Error> {
        for (vector, bucket) in self.descriptor.buckets.iter().enumerate() {
            let start = bucket.dense_offset;
            let end = start + bucket.entry_count();
            visit(vector, &self.values[start..end]);
        }
        Ok(())
    }
}

impl DenseI16BvaSource for DenseMatrixBva<i16> {
    fn dense_values(&self) -> &[i16] {
        &self.values
    }
}

/// Short-integer types supported by the matrix-collection Dory pipeline.
pub trait MatrixBvaInteger: BvaInteger {
    fn commit_bva(
        prover_parameters: &DoryProverParameters,
        source: &DenseMatrixBva<Self>,
        bit_width: usize,
    ) -> Result<CommittedDoryBva, DoryBackendError>;

    fn prove_precommitted(
        prover_parameters: &DoryProverParameters,
        source: &mut DenseMatrixBva<Self>,
        tbv: &TensorBasisVector<Fr>,
        claimed: Fr,
        bit_width: usize,
        committed: CommittedDoryBva,
    ) -> Result<
        (PackedI8ParCommProof, PackedI8ParCommProfile),
        PackedI8ParCommProverError<Infallible>,
    >;
}

impl MatrixBvaInteger for i8 {
    fn commit_bva(
        prover_parameters: &DoryProverParameters,
        source: &DenseMatrixBva<Self>,
        bit_width: usize,
    ) -> Result<CommittedDoryBva, DoryBackendError> {
        commit_bva_i8_streaming(prover_parameters, source, bit_width)
    }

    fn prove_precommitted(
        prover_parameters: &DoryProverParameters,
        source: &mut DenseMatrixBva<Self>,
        tbv: &TensorBasisVector<Fr>,
        claimed: Fr,
        bit_width: usize,
        committed: CommittedDoryBva,
    ) -> Result<
        (PackedI8ParCommProof, PackedI8ParCommProfile),
        PackedI8ParCommProverError<Infallible>,
    > {
        prove_packed_i8_parcomm_precommitted(
            prover_parameters,
            source,
            tbv,
            claimed,
            bit_width,
            committed,
        )
    }
}

impl MatrixBvaInteger for i16 {
    fn commit_bva(
        prover_parameters: &DoryProverParameters,
        source: &DenseMatrixBva<Self>,
        bit_width: usize,
    ) -> Result<CommittedDoryBva, DoryBackendError> {
        commit_bva_i16_streaming(prover_parameters, source, bit_width)
    }

    fn prove_precommitted(
        prover_parameters: &DoryProverParameters,
        source: &mut DenseMatrixBva<Self>,
        tbv: &TensorBasisVector<Fr>,
        claimed: Fr,
        bit_width: usize,
        committed: CommittedDoryBva,
    ) -> Result<
        (PackedI8ParCommProof, PackedI8ParCommProfile),
        PackedI8ParCommProverError<Infallible>,
    > {
        prove_packed_i16_parcomm_precommitted(
            prover_parameters,
            source,
            tbv,
            claimed,
            bit_width,
            committed,
        )
    }
}

/// Public collection statement after Fiat--Shamir batching and point compilation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CompiledMatrixClaims<F: PrimeField> {
    pub omega: F,
    pub claimed: F,
    pub tbv: TensorBasisVector<F>,
    pub bit_width: usize,
}

/// The dense BVA MLE claim emitted by Packed EO-LiteBullet.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BvaMleClaim<F: PrimeField> {
    /// Low-bit-first global evaluation point `y^{-1}`.
    pub point: Vec<F>,
    pub value: F,
}

impl CompiledMatrixClaims<Fr> {
    /// Replay the bound Packed EO transcript and return its dense BVA MLE claim.
    pub fn bva_mle_claim(&self, proof: &PackedI8ParCommProof) -> Option<BvaMleClaim<Fr>> {
        let binding = packed_i8_parcomm_statement_binding(
            &proof.matrix_commitment,
            &self.tbv,
            self.claimed,
            self.bit_width,
        );
        let verification = proof
            .packed_eo
            .verify_with_tbv_bound(&self.tbv, self.claimed, binding);
        self.bva_mle_claim_from_scalar_verification(proof, &verification)
    }

    /// Return the BVA claim after the scalar transcript and Dory opening have
    /// already been checked, without performing the terminal TBV fold.
    pub fn bva_mle_claim_from_scalar_verification(
        &self,
        proof: &PackedI8ParCommProof,
        verification: &EoLiteBulletVerification<Fr>,
    ) -> Option<BvaMleClaim<Fr>> {
        verification.accepted.then_some(BvaMleClaim {
            point: verification.challenges_inv.clone(),
            value: proof.packed_eo.a_reduced,
        })
    }
}

/// Commit the shape tables before committing the dense BVA.
pub fn commit_matrix_shapes(
    prover_parameters: &DoryProverParameters,
    descriptor: &DenseBvaDescriptor,
) -> Result<CommittedMatrixShapes, MatrixCollectionProverError> {
    descriptor.validate_canonical()?;
    let (alpha_values, gamma_values, profile) = shape_field_vectors(descriptor)?;
    let alpha = commit_field_vector_zk(prover_parameters, &alpha_values)
        .map_err(matrix_collection_backend_error)?;
    let gamma = commit_field_vector_zk(prover_parameters, &gamma_values)
        .map_err(matrix_collection_backend_error)?;
    Ok(CommittedMatrixShapes {
        descriptor: descriptor.clone(),
        alpha_values,
        gamma_values,
        alpha,
        gamma,
        profile,
    })
}

/// Commit the contiguous short-integer BVA after fixing the shape commitments.
pub fn commit_registered_dense_bva<Integer: MatrixBvaInteger>(
    prover_parameters: &DoryProverParameters,
    source: &DenseMatrixBva<Integer>,
    bit_width: usize,
    shape: CommittedMatrixShapes,
) -> Result<RegisteredMatrixCollection, MatrixCollectionProverError> {
    if source.descriptor != shape.descriptor {
        return Err(MatrixCollectionError::RegistrationMismatch.into());
    }
    let bva = Integer::commit_bva(prover_parameters, source, bit_width)
        .map_err(matrix_collection_backend_error)?;
    let commitments = MatrixCollectionCommitments {
        shape: shape.commitments(),
        bva: bva.commitment.clone(),
    };
    let profile = MatrixCollectionCommitProfile {
        shape: shape.profile,
        bva: bva.profile,
    };
    Ok(RegisteredMatrixCollection {
        descriptor: shape.descriptor,
        bit_width,
        commitments,
        alpha_values: shape.alpha_values,
        gamma_values: shape.gamma_values,
        alpha: shape.alpha,
        gamma: shape.gamma,
        bva,
        profile,
    })
}

/// Convenience wrapper for the ordered shape-then-BVA registration phase.
pub fn commit_matrix_collection<Integer: MatrixBvaInteger>(
    prover_parameters: &DoryProverParameters,
    source: &DenseMatrixBva<Integer>,
    bit_width: usize,
) -> Result<RegisteredMatrixCollection, MatrixCollectionProverError> {
    let shape = commit_matrix_shapes(prover_parameters, source.descriptor())?;
    commit_registered_dense_bva(prover_parameters, source, bit_width, shape)
}

/// Open a previously registered collection and append the final shape/point
/// consistency openings after Packed EO and the dense BVA opening.
pub fn prove_registered_matrix_claim_collection<Integer: MatrixBvaInteger>(
    prover_parameters: &DoryProverParameters,
    source: &mut DenseMatrixBva<Integer>,
    claims: &[MatrixMleClaim<Fr>],
    registration: &RegisteredMatrixCollection,
) -> Result<(RegisteredMatrixCollectionProof, PackedI8ParCommProfile), MatrixCollectionProverError>
{
    if source.descriptor != registration.descriptor {
        return Err(MatrixCollectionError::RegistrationMismatch.into());
    }
    let compiled = registration.descriptor.compile_for_registered_proof(
        &registration.commitments,
        claims,
        registration.bit_width,
    )?;
    let (packed, profile) = Integer::prove_precommitted(
        prover_parameters,
        source,
        &compiled.tbv,
        compiled.claimed,
        registration.bit_width,
        registration.bva.clone(),
    )
    .map_err(MatrixCollectionProverError::Packed)?;
    let challenge = shape_consistency_challenge(
        &registration.descriptor,
        &registration.commitments,
        claims,
        registration.bit_width,
        &packed,
        registration.alpha.logical_len,
        registration.gamma.logical_len,
    )?;
    let (matrices, evaluations) = open_field_vectors_batch_zk(
        prover_parameters,
        &[&registration.alpha_values, &registration.gamma_values],
        &[&registration.alpha, &registration.gamma],
        &challenge.point,
        b"section6-shape-matrices",
        &challenge.outer_binding,
    )
    .map_err(matrix_collection_backend_error)?;
    if evaluations.len() != 2 {
        return Err(MatrixCollectionError::ShapeEvaluationMismatch.into());
    }
    Ok((
        RegisteredMatrixCollectionProof {
            packed,
            shape_consistency: ShapePointConsistencyProof { matrices },
        },
        profile,
    ))
}

/// Verify the dense BVA opening while deliberately postponing the terminal
/// TBV fold and the shape/point check.
pub fn verify_registered_bva_opening(
    verifier_parameters: &DoryVerifierParameters,
    descriptor: &DenseBvaDescriptor,
    commitments: &MatrixCollectionCommitments,
    claims: &[MatrixMleClaim<Fr>],
    bit_width: usize,
    proof: &RegisteredMatrixCollectionProof,
) -> Result<RegisteredBvaOpeningVerification, MatrixCollectionVerificationError> {
    if proof.packed.matrix_commitment != commitments.bva {
        return Err(MatrixCollectionError::RegistrationMismatch.into());
    }
    let compiled = descriptor.compile_for_registered_proof(commitments, claims, bit_width)?;
    let eo = verify_packed_i16_bva_opening_as_subprotocol(
        verifier_parameters,
        &compiled.tbv,
        compiled.claimed,
        bit_width,
        &proof.packed,
    )
    .map_err(MatrixCollectionVerificationError::Packed)?;
    Ok(RegisteredBvaOpeningVerification { compiled, eo })
}

/// Check the terminal TBV inner product obtained from the Packed EO transcript.
pub fn verify_registered_tbv_inner_product(
    opening: &RegisteredBvaOpeningVerification,
    proof: &RegisteredMatrixCollectionProof,
) -> bool {
    verify_packed_i8_tbv_inner_product(&opening.compiled.tbv, &proof.packed, &opening.eo)
}

/// Verify the committed shape encoding at transcript-derived evaluation points.
/// The check uses only the descriptor, matrix evaluation points, and proof.
pub fn verify_registered_shape_point_consistency(
    verifier_parameters: &DoryVerifierParameters,
    descriptor: &DenseBvaDescriptor,
    commitments: &MatrixCollectionCommitments,
    claims: &[MatrixMleClaim<Fr>],
    bit_width: usize,
    proof: &RegisteredMatrixCollectionProof,
) -> Result<bool, MatrixCollectionVerificationError> {
    if proof.packed.matrix_commitment != commitments.bva {
        return Err(MatrixCollectionError::RegistrationMismatch.into());
    }
    descriptor.validate_claims(claims)?;
    descriptor.validate_canonical()?;
    let (alpha_values, gamma_values, _) = shape_field_vectors(descriptor)?;
    let challenge = shape_consistency_challenge(
        descriptor,
        commitments,
        claims,
        bit_width,
        &proof.packed,
        alpha_values.len(),
        gamma_values.len(),
    )?;
    let evaluations = [
        evaluate_field_vector_standard(&alpha_values, &challenge.point)
            .map_err(MatrixCollectionVerificationError::Packed)?,
        evaluate_field_vector_standard(&gamma_values, &challenge.point)
            .map_err(MatrixCollectionVerificationError::Packed)?,
    ];
    verify_field_vectors_batch_zk(
        verifier_parameters,
        &[&commitments.shape.alpha, &commitments.shape.gamma],
        alpha_values.len(),
        &challenge.point,
        &evaluations,
        &proof.shape_consistency.matrices,
        b"section6-shape-matrices",
        &challenge.outer_binding,
    )
    .map_err(MatrixCollectionVerificationError::Packed)?;
    Ok(true)
}

/// Verify every registered stage in the protocol order.
pub fn verify_registered_matrix_claim_collection(
    verifier_parameters: &DoryVerifierParameters,
    descriptor: &DenseBvaDescriptor,
    commitments: &MatrixCollectionCommitments,
    claims: &[MatrixMleClaim<Fr>],
    bit_width: usize,
    proof: &RegisteredMatrixCollectionProof,
) -> Result<RegisteredMatrixCollectionVerification, MatrixCollectionVerificationError> {
    let opening = verify_registered_bva_opening(
        verifier_parameters,
        descriptor,
        commitments,
        claims,
        bit_width,
        proof,
    )?;
    let tbv_inner_product_accepted = verify_registered_tbv_inner_product(&opening, proof);
    if !tbv_inner_product_accepted {
        return Err(MatrixCollectionVerificationError::Packed(
            DoryBackendError::PackedEoRejected,
        ));
    }
    let shape_point_consistency_accepted = verify_registered_shape_point_consistency(
        verifier_parameters,
        descriptor,
        commitments,
        claims,
        bit_width,
        proof,
    )?;
    Ok(RegisteredMatrixCollectionVerification {
        bva_opening_accepted: true,
        tbv_inner_product_accepted,
        shape_point_consistency_accepted,
    })
}

/// Commit the dense BVA, derive `omega`, compile the public claims, and run the
/// commitment-bound Packed EO plus Jolt Dory opening pipeline.
pub fn prove_matrix_claim_collection<Integer: MatrixBvaInteger>(
    prover_parameters: &DoryProverParameters,
    source: &mut DenseMatrixBva<Integer>,
    claims: &[MatrixMleClaim<Fr>],
    bit_width: usize,
) -> Result<(PackedI8ParCommProof, PackedI8ParCommProfile), MatrixCollectionProverError> {
    source.descriptor.validate_claims(claims)?;
    let committed = Integer::commit_bva(prover_parameters, source, bit_width).map_err(|error| {
        MatrixCollectionProverError::Packed(PackedI8ParCommProverError::Backend(error))
    })?;
    let compiled = source
        .descriptor
        .compile_for_proof(&committed.commitment, claims, bit_width)?;
    Integer::prove_precommitted(
        prover_parameters,
        source,
        &compiled.tbv,
        compiled.claimed,
        bit_width,
        committed,
    )
    .map_err(MatrixCollectionProverError::Packed)
}

/// Perform terminal Dory verification without exposing the dense witness.
pub fn verify_matrix_claim_collection_as_subprotocol(
    verifier_parameters: &DoryVerifierParameters,
    descriptor: &DenseBvaDescriptor,
    claims: &[MatrixMleClaim<Fr>],
    bit_width: usize,
    proof: &PackedI8ParCommProof,
) -> Result<TerminalDoryVerification, MatrixCollectionVerificationError> {
    let compiled = descriptor.compile_for_proof(&proof.matrix_commitment, claims, bit_width)?;
    verify_packed_i16_parcomm_as_subprotocol(
        verifier_parameters,
        &compiled.tbv,
        compiled.claimed,
        bit_width,
        proof,
    )
    .map_err(MatrixCollectionVerificationError::Packed)
}

pub fn verify_matrix_claim_collection(
    verifier_parameters: &DoryVerifierParameters,
    descriptor: &DenseBvaDescriptor,
    claims: &[MatrixMleClaim<Fr>],
    bit_width: usize,
    proof: &PackedI8ParCommProof,
) -> Result<bool, MatrixCollectionVerificationError> {
    Ok(verify_matrix_claim_collection_as_subprotocol(
        verifier_parameters,
        descriptor,
        claims,
        bit_width,
        proof,
    )?
    .evaluate())
}

/// Direct column-major matrix evaluation in Section 6 reverse-tensor order.
pub fn matrix_mle_evaluate<F: PrimeField>(
    matrix: &DenseShortIntMatrix<impl BvaInteger>,
    left_point: &[F],
    right_point: &[F],
) -> Result<F, MatrixCollectionError> {
    validate_point_lengths(0, matrix.shape, left_point.len(), right_point.len())?;
    let left = tensor_weights(left_point);
    let right = tensor_weights(right_point);
    let rows = matrix.shape.0;
    Ok(matrix
        .values
        .par_chunks(rows)
        .zip(&right)
        .map(|(values, column_coefficient)| {
            let column_sum = values
                .iter()
                .zip(&left)
                .map(|(&value, row_coefficient)| value.to_field::<F>() * row_coefficient)
                .sum::<F>();
            *column_coefficient * column_sum
        })
        .sum())
}

/// Evaluate a constant matrix without materializing its entries.
pub fn constant_matrix_mle_evaluate<F: PrimeField>(
    shape: (usize, usize),
    value: i8,
    left_point: &[F],
    right_point: &[F],
) -> Result<F, MatrixCollectionError> {
    validate_shape(0, shape)?;
    validate_point_lengths(0, shape, left_point.len(), right_point.len())?;
    Ok(i8_to_field::<F>(value)
        * tensor_prefix_sum(shape.0, left_point)
        * tensor_prefix_sum(shape.1, right_point))
}

#[derive(Debug)]
pub enum RegisteredMatrixCollectionProofDecodeError {
    Packed(PackedI8ParCommProofDecodeError),
    Bincode(String),
    InvalidMagic,
    InvalidLength,
    ComponentTooLarge,
    TrailingBytes,
}

impl Display for RegisteredMatrixCollectionProofDecodeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for RegisteredMatrixCollectionProofDecodeError {}

impl From<PackedI8ParCommProofDecodeError> for RegisteredMatrixCollectionProofDecodeError {
    fn from(error: PackedI8ParCommProofDecodeError) -> Self {
        Self::Packed(error)
    }
}

#[derive(Debug)]
pub enum MatrixCollectionError {
    EmptyCollection,
    ZeroDimension {
        matrix: usize,
        shape: (usize, usize),
    },
    LengthOverflow,
    MatrixLength {
        matrix: usize,
        expected: usize,
        actual: usize,
    },
    MatrixIndex {
        matrix: usize,
        count: usize,
    },
    PackedBvaLength {
        expected: usize,
        actual: usize,
    },
    NonZeroPackedPadding {
        index: usize,
    },
    PointLength {
        matrix: usize,
        side: &'static str,
        expected: usize,
        actual: usize,
    },
    ClaimCount {
        expected: usize,
        actual: usize,
    },
    ClaimShape {
        matrix: usize,
        expected: (usize, usize),
        actual: (usize, usize),
    },
    RegistrationMismatch,
    InvalidShapeEncoding,
    ShapeEvaluationMismatch,
    ShapeTranscriptEncoding,
    NonCanonicalLayout,
    Bva(BvaError),
    DenseEvaluation(String),
}

impl Display for MatrixCollectionError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyCollection => write!(formatter, "a matrix collection must be non-empty"),
            Self::ZeroDimension { matrix, shape } => write!(
                formatter,
                "matrix {matrix} has an invalid zero dimension in shape {shape:?}"
            ),
            Self::LengthOverflow => write!(formatter, "matrix collection length overflowed usize"),
            Self::MatrixLength {
                matrix,
                expected,
                actual,
            } => write!(
                formatter,
                "matrix {matrix} length mismatch: expected {expected}, got {actual}"
            ),
            Self::MatrixIndex { matrix, count } => write!(
                formatter,
                "matrix index {matrix} is outside collection of size {count}"
            ),
            Self::PackedBvaLength { expected, actual } => write!(
                formatter,
                "prepacked BVA length mismatch: expected {expected}, got {actual}"
            ),
            Self::NonZeroPackedPadding { index } => write!(
                formatter,
                "prepacked BVA padding is nonzero at dense index {index}"
            ),
            Self::PointLength {
                matrix,
                side,
                expected,
                actual,
            } => write!(
                formatter,
                "matrix {matrix} {side} point length mismatch: expected {expected}, got {actual}"
            ),
            Self::ClaimCount { expected, actual } => write!(
                formatter,
                "matrix claim count mismatch: expected {expected}, got {actual}"
            ),
            Self::ClaimShape {
                matrix,
                expected,
                actual,
            } => write!(
                formatter,
                "matrix {matrix} claim shape mismatch: expected {expected:?}, got {actual:?}"
            ),
            Self::RegistrationMismatch => {
                write!(
                    formatter,
                    "registration does not match the matrix collection"
                )
            }
            Self::InvalidShapeEncoding => {
                write!(formatter, "shape encoding has an invalid logical length")
            }
            Self::ShapeEvaluationMismatch => {
                write!(formatter, "shape opening returned an unexpected evaluation")
            }
            Self::ShapeTranscriptEncoding => {
                write!(
                    formatter,
                    "could not encode the BVA opening for the shape transcript"
                )
            }
            Self::NonCanonicalLayout => write!(formatter, "matrix BVA layout is not canonical"),
            Self::Bva(error) => Display::fmt(error, formatter),
            Self::DenseEvaluation(message) => write!(formatter, "dense BVA evaluation: {message}"),
        }
    }
}

impl std::error::Error for MatrixCollectionError {}

impl From<BvaError> for MatrixCollectionError {
    fn from(error: BvaError) -> Self {
        Self::Bva(error)
    }
}

#[derive(Debug)]
pub enum MatrixCollectionProverError {
    Collection(MatrixCollectionError),
    Packed(PackedI8ParCommProverError<Infallible>),
}

impl Display for MatrixCollectionProverError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Collection(error) => Display::fmt(error, formatter),
            Self::Packed(error) => Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for MatrixCollectionProverError {}

impl From<MatrixCollectionError> for MatrixCollectionProverError {
    fn from(error: MatrixCollectionError) -> Self {
        Self::Collection(error)
    }
}

#[derive(Debug)]
pub enum MatrixCollectionVerificationError {
    Collection(MatrixCollectionError),
    Packed(PackedI8ParCommVerificationError),
}

impl Display for MatrixCollectionVerificationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Collection(error) => Display::fmt(error, formatter),
            Self::Packed(error) => Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for MatrixCollectionVerificationError {}

impl From<MatrixCollectionError> for MatrixCollectionVerificationError {
    fn from(error: MatrixCollectionError) -> Self {
        Self::Collection(error)
    }
}

struct ShapeConsistencyChallenge {
    point: Vec<Fr>,
    outer_binding: Vec<u8>,
}

fn shape_field_vectors(
    descriptor: &DenseBvaDescriptor,
) -> Result<(Vec<Fr>, Vec<Fr>, ShapeCommitProfile), MatrixCollectionError> {
    let bucket_rows = descriptor.bucket_count();
    let padded_rows = bucket_rows
        .checked_next_power_of_two()
        .ok_or(MatrixCollectionError::LengthOverflow)?;
    let alpha_len = padded_rows
        .checked_mul(4)
        .ok_or(MatrixCollectionError::LengthOverflow)?;
    // Gamma is embedded in the same four-column rectangular cover as Alpha so
    // both whole matrices share one Dory opening point.
    let gamma_len = alpha_len;
    let mut alpha = vec![Fr::from(0_u64); alpha_len];
    let mut gamma = vec![Fr::from(0_u64); gamma_len];
    for (row, bucket) in descriptor.buckets.iter().enumerate() {
        alpha[row] = Fr::from((bucket.matrix + 1) as u64);
        alpha[padded_rows + row] = Fr::from(bucket.row_offset as u64);
        alpha[2 * padded_rows + row] = Fr::from(bucket.col_offset as u64);
        gamma[row] = Fr::from(bucket.row_log() as u64);
        gamma[padded_rows + row] = Fr::from(bucket.col_log() as u64);
    }
    Ok((
        alpha,
        gamma,
        ShapeCommitProfile {
            bucket_rows,
            padded_rows,
            alpha_field_entries: alpha_len,
            gamma_field_entries: gamma_len,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn shape_consistency_challenge(
    descriptor: &DenseBvaDescriptor,
    commitments: &MatrixCollectionCommitments,
    claims: &[MatrixMleClaim<Fr>],
    bit_width: usize,
    proof: &PackedI8ParCommProof,
    alpha_len: usize,
    gamma_len: usize,
) -> Result<ShapeConsistencyChallenge, MatrixCollectionError> {
    if alpha_len == 0 || gamma_len != alpha_len || !alpha_len.is_power_of_two() {
        return Err(MatrixCollectionError::InvalidShapeEncoding);
    }
    let mut transcript = FiatShamir::<Fr>::new();
    transcript.push(&0x5348_4150_4550_5432_u64);
    transcript.push(&(bit_width as u64));
    transcript.push_bytes(&commitment_binding_bytes(&commitments.shape.alpha));
    transcript.push_bytes(&commitment_binding_bytes(&commitments.shape.gamma));
    transcript.push_bytes(&commitment_binding_bytes(&commitments.bva));
    transcript.push(&(descriptor.matrix_count() as u64));
    for &(rows, cols) in descriptor.matrix_shapes() {
        transcript.push(&(rows as u64));
        transcript.push(&(cols as u64));
    }
    transcript.push(&(descriptor.logical_len() as u64));
    transcript.push(&(descriptor.physical_len() as u64));
    transcript.push(&(descriptor.bucket_count() as u64));
    for bucket in descriptor.buckets() {
        transcript.push(&(bucket.matrix() as u64));
        transcript.push(&(bucket.row_offset() as u64));
        transcript.push(&(bucket.row_len() as u64));
        transcript.push(&(bucket.col_offset() as u64));
        transcript.push(&(bucket.col_len() as u64));
        transcript.push(&(bucket.dense_offset() as u64));
    }
    transcript.push(&(claims.len() as u64));
    for claim in claims {
        transcript.push(&(claim.shape.0 as u64));
        transcript.push(&(claim.shape.1 as u64));
        transcript.push(&(claim.left_point.len() as u64));
        for coordinate in &claim.left_point {
            transcript.push(coordinate);
        }
        transcript.push(&(claim.right_point.len() as u64));
        for coordinate in &claim.right_point {
            transcript.push(coordinate);
        }
        transcript.push(&claim.value);
    }
    let packed_proof_bytes = proof
        .to_bytes()
        .map_err(|_| MatrixCollectionError::ShapeTranscriptEncoding)?;
    transcript.push_bytes(&packed_proof_bytes);
    transcript.push(&(alpha_len as u64));
    transcript.push(&(gamma_len as u64));
    let point = (0..alpha_len.ilog2())
        .map(|_| transcript.gen_challenge())
        .collect::<Vec<_>>();
    let mut outer_binding = Vec::new();
    transcript
        .state()
        .serialize_compressed(&mut outer_binding)
        .expect("shape transcript serialization into memory cannot fail");
    Ok(ShapeConsistencyChallenge {
        point,
        outer_binding,
    })
}

fn matrix_collection_backend_error(error: DoryBackendError) -> MatrixCollectionProverError {
    MatrixCollectionProverError::Packed(PackedI8ParCommProverError::Backend(error))
}

fn serialize_registered_component<T: Serialize>(
    value: &T,
) -> Result<Vec<u8>, RegisteredMatrixCollectionProofDecodeError> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize(value)
        .map_err(|error| RegisteredMatrixCollectionProofDecodeError::Bincode(error.to_string()))
}

fn deserialize_registered_component<T: DeserializeOwned>(
    bytes: &[u8],
) -> Result<T, RegisteredMatrixCollectionProofDecodeError> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|error| RegisteredMatrixCollectionProofDecodeError::Bincode(error.to_string()))
}

fn append_registered_component(bytes: &mut Vec<u8>, component: &[u8]) {
    bytes.extend_from_slice(&(component.len() as u64).to_le_bytes());
    bytes.extend_from_slice(component);
}

fn take_registered_component<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
) -> Result<&'a [u8], RegisteredMatrixCollectionProofDecodeError> {
    let encoded_len = take_registered_bytes(bytes, cursor, std::mem::size_of::<u64>())?;
    let mut len_bytes = [0_u8; std::mem::size_of::<u64>()];
    len_bytes.copy_from_slice(encoded_len);
    let len = usize::try_from(u64::from_le_bytes(len_bytes))
        .map_err(|_| RegisteredMatrixCollectionProofDecodeError::InvalidLength)?;
    if len > MAX_REGISTERED_COMPONENT_BYTES {
        return Err(RegisteredMatrixCollectionProofDecodeError::ComponentTooLarge);
    }
    take_registered_bytes(bytes, cursor, len)
}

fn take_registered_bytes<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    len: usize,
) -> Result<&'a [u8], RegisteredMatrixCollectionProofDecodeError> {
    let end = cursor
        .checked_add(len)
        .ok_or(RegisteredMatrixCollectionProofDecodeError::InvalidLength)?;
    let output = bytes
        .get(*cursor..end)
        .ok_or(RegisteredMatrixCollectionProofDecodeError::InvalidLength)?;
    *cursor = end;
    Ok(output)
}

fn validate_shape(matrix: usize, shape: (usize, usize)) -> Result<(), MatrixCollectionError> {
    if shape.0 == 0 || shape.1 == 0 {
        return Err(MatrixCollectionError::ZeroDimension { matrix, shape });
    }
    shape
        .0
        .checked_mul(shape.1)
        .ok_or(MatrixCollectionError::LengthOverflow)?;
    Ok(())
}

fn validate_point_lengths(
    matrix: usize,
    shape: (usize, usize),
    left_len: usize,
    right_len: usize,
) -> Result<(), MatrixCollectionError> {
    let expected_left = ceil_log2(shape.0);
    if left_len != expected_left {
        return Err(MatrixCollectionError::PointLength {
            matrix,
            side: "left",
            expected: expected_left,
            actual: left_len,
        });
    }
    let expected_right = ceil_log2(shape.1);
    if right_len != expected_right {
        return Err(MatrixCollectionError::PointLength {
            matrix,
            side: "right",
            expected: expected_right,
            actual: right_len,
        });
    }
    Ok(())
}

fn ceil_log2(value: usize) -> usize {
    value.next_power_of_two().ilog2() as usize
}

fn previous_power_of_two(value: usize) -> usize {
    1_usize << (usize::BITS - 1 - value.leading_zeros())
}

fn capped_aligned_dyadic_intervals(length: usize, budget: usize) -> Vec<(usize, usize)> {
    debug_assert!(length > 0);
    debug_assert!(budget >= 2);
    let mut sizes = Vec::with_capacity(budget);
    let mut remaining = length;
    while remaining != 0 && sizes.len() < budget {
        let size = previous_power_of_two(remaining);
        sizes.push(size);
        remaining -= size;
    }
    if remaining != 0 {
        let last = sizes
            .last_mut()
            .expect("a positive length emits a first dyadic interval");
        *last *= 2;
    }
    let mut offset = 0_usize;
    sizes
        .into_iter()
        .map(|size| {
            let interval = (offset, size);
            offset += size;
            interval
        })
        .collect()
}

fn omega_powers<F: PrimeField>(omega: F, count: usize) -> Vec<F> {
    std::iter::successors(Some(F::one()), |power| Some(*power * omega))
        .take(count)
        .collect()
}

fn fixed_offset_factor<F: PrimeField>(offset: usize, local_log: usize, point: &[F]) -> F {
    point
        .iter()
        .enumerate()
        .skip(local_log)
        .filter(|(bit, _)| (offset >> *bit) & 1 == 1)
        .map(|(_, coordinate)| *coordinate)
        .product()
}

fn tensor_weights<F: PrimeField>(point: &[F]) -> Vec<F> {
    let mut weights = vec![F::one()];
    for coordinate in point {
        let current = weights.len();
        weights.reserve(current);
        for index in 0..current {
            weights.push(weights[index] * coordinate);
        }
    }
    weights
}

fn tensor_prefix_sum<F: PrimeField>(length: usize, point: &[F]) -> F {
    if length == 1_usize << point.len() {
        return point.iter().fold(F::one(), |product, coordinate| {
            product * (F::one() + coordinate)
        });
    }
    let mut result = F::zero();
    let mut fixed_high = F::one();
    for bit in (0..point.len()).rev() {
        if (length >> bit) & 1 == 1 {
            let free_low = point[..bit].iter().fold(F::one(), |product, coordinate| {
                product * (F::one() + coordinate)
            });
            result += fixed_high * free_low;
            fixed_high *= point[bit];
        }
    }
    result
}

fn i8_to_field<F: PrimeField>(value: i8) -> F {
    if value >= 0 {
        F::from(value as u64)
    } else {
        -F::from((-(value as i16)) as u64)
    }
}

const PARALLEL_FILL_THRESHOLD: usize = 1 << 18;

fn fill_bucket_range<Integer, ValueAt>(
    values: &mut [Integer],
    buckets: &[MatrixBucketDescriptor],
    value_at: &ValueAt,
) where
    Integer: BvaInteger,
    ValueAt: Fn(usize, usize, usize) -> Integer + Sync,
{
    if buckets.is_empty() {
        debug_assert!(values.is_empty());
        return;
    }
    if buckets.len() == 1 {
        let bucket = &buckets[0];
        if values.len() >= PARALLEL_FILL_THRESHOLD {
            values.par_iter_mut().enumerate().for_each(|(flat, slot)| {
                let local_row = flat / bucket.col_len;
                let local_col = flat % bucket.col_len;
                if local_row < bucket.physical_row_len && local_col < bucket.physical_col_len {
                    *slot = value_at(
                        bucket.matrix,
                        bucket.row_offset + local_row,
                        bucket.col_offset + local_col,
                    );
                }
            });
        } else {
            for (flat, slot) in values.iter_mut().enumerate() {
                let local_row = flat / bucket.col_len;
                let local_col = flat % bucket.col_len;
                if local_row < bucket.physical_row_len && local_col < bucket.physical_col_len {
                    *slot = value_at(
                        bucket.matrix,
                        bucket.row_offset + local_row,
                        bucket.col_offset + local_col,
                    );
                }
            }
        }
        return;
    }
    if values.len() < PARALLEL_FILL_THRESHOLD {
        let mut consumed = 0_usize;
        for bucket in buckets {
            let count = bucket.entry_count();
            fill_bucket_range(
                &mut values[consumed..consumed + count],
                std::slice::from_ref(bucket),
                value_at,
            );
            consumed += count;
        }
        return;
    }

    let target = values.len() / 2;
    let mut left_len = 0_usize;
    let mut split = 0_usize;
    while split + 1 < buckets.len() && left_len + buckets[split].entry_count() <= target {
        left_len += buckets[split].entry_count();
        split += 1;
    }
    if split == 0 {
        split = 1;
        left_len = buckets[0].entry_count();
    }
    let (left_values, right_values) = values.split_at_mut(left_len);
    let (left_buckets, right_buckets) = buckets.split_at(split);
    rayon::join(
        || fill_bucket_range(left_values, left_buckets, value_at),
        || fill_bucket_range(right_values, right_buckets, value_at),
    );
}
