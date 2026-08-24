//! Streaming SmartPC binding for byte-backed decreasing-size BVAs.
//!
//! The physical witness stays in `i8` storage. Commitment construction uses
//! bounded row chunks, packed EO-LiteBullet is bound to the resulting matrix
//! commitment, and SmartPC verification is returned as a deferred sparse
//! pairing equation.

use std::marker::PhantomData;
use std::time::Instant;
use std::{
    fmt::{Display, Formatter},
    io::Cursor,
    path::{Path, PathBuf},
};

use ark_crypto_primitives::sponge::Absorb;
use ark_ec::{
    pairing::{Pairing, PairingOutput},
    AffineRepr, CurveGroup,
};
use ark_ff::{Field, UniformRand};
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};
use ark_std::{ops::Mul, Zero};
use poly_commit::smart_pc::{
    data_structures::{NativeFoldMessage, SmartPcVerificationEquation, Trans, UniversalParams},
    sub_protocols, utils as pcutils, FiatShamir, SmartPC,
};
use rayon::prelude::*;

use crate::bucketed_ip::{
    packing_stride_for_bit_width, BvaI8Source, PackedEoLiteBulletProof, PaddingWorkCounters,
    StreamingEoError, StreamingEoProfile, TensorBasisVector,
};

/// Columns handled by one parallel commitment task.
///
/// Consecutive packed columns are consecutive bytes of the witness, so a block
/// of 64 columns covers whole cache lines and every task streams the witness
/// exactly once.
const COMMIT_COLUMN_BLOCK: usize = 64;

/// Packed rows visited by one commitment pass.
///
/// A column sweep touches one cache line per row, so a bounded row count keeps
/// the working set of a 64-column block inside L1 and its pages inside the TLB.
/// The per-column bucket drain is amortized over this many entries.
const COMMIT_ROWS_PER_CHUNK: usize = 512;

/// Bounded-memory statistics for a streaming byte commitment.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BvaStreamingCommitProfile {
    /// Number of stored byte entries consumed.
    pub physical_entries: usize,
    /// Number of nonzero byte entries consumed.
    pub nonzero_entries: usize,
    /// Logical power-of-two BVA length.
    pub logical_entries: usize,
    /// Number of affine G1 bases prepared once for the whole commitment.
    pub prepared_base_entries: usize,
    /// Number of vectors processed with parallel bounded chunks.
    pub parallel_vectors: usize,
    /// Maximum number of live G1 bucket accumulators held by one parallel task.
    pub max_live_group_contributions: usize,
    /// Materialized logical-zero entries. This is always zero.
    pub materialized_padding_entries: usize,
    /// Group operations performed on logical padding. This is always zero.
    pub padding_group_operations: usize,
}

/// Statistics for the two streaming projections used by SmartPC opening.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StreamingOpenProfile {
    /// Number of complete source passes.
    pub source_passes: usize,
    /// Number of physical byte entries visited across all passes.
    pub physical_entries: usize,
    /// Number of nonzero byte entries visited across all passes.
    pub nonzero_entries: usize,
    /// Field operations involving real entries.
    pub real_projection_field_multiplications: usize,
    /// Materialized logical-zero entries. This is always zero.
    pub materialized_projection_padding_entries: usize,
    /// Field operations involving logical padding. This is always zero.
    pub padding_projection_field_multiplications: usize,
}

/// Profiles for the three prover stages.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PackedI8ParCommProfile {
    /// Streaming matrix commitment profile.
    pub commitment: BvaStreamingCommitProfile,
    /// Packed EO-LiteBullet profile.
    pub packed_eo: StreamingEoProfile,
    /// Streaming SmartPC opening profile.
    pub opening: StreamingOpenProfile,
    /// Padding counters returned by packed EO-LiteBullet.
    pub packed_eo_padding: PaddingWorkCounters,
}

/// A complete commitment-bound packed EO-LiteBullet opening.
#[derive(Clone, Debug, Eq, PartialEq, CanonicalDeserialize, CanonicalSerialize)]
pub struct PackedI8ParCommProof<E: Pairing> {
    /// Hidden commitment to the packed byte matrix.
    pub matrix_commitment: PairingOutput<E>,
    /// Non-hiding commitment to the final packed EO witness scalar.
    pub value_commitment: PairingOutput<E>,
    /// Packed EO-LiteBullet inner-product reduction.
    pub packed_eo: PackedEoLiteBulletProof<E::ScalarField>,
    /// SmartPC opening at the packed EO reduction point.
    pub opening: Trans<E>,
}

impl<E: Pairing> PackedI8ParCommProof<E> {
    /// Serializes the complete proof artifact in compressed canonical form.
    pub fn to_bytes(&self) -> Result<Vec<u8>, SerializationError> {
        let mut bytes = Vec::new();
        self.serialize_compressed(&mut bytes)?;
        Ok(bytes)
    }

    /// Decodes exactly one complete proof artifact.
    pub fn from_bytes_exact(bytes: &[u8]) -> Result<Self, PackedI8ParCommProofDecodeError> {
        let mut reader = Cursor::new(bytes);
        let proof = Self::deserialize_compressed(&mut reader)?;
        if reader.position() != bytes.len() as u64 {
            return Err(PackedI8ParCommProofDecodeError::TrailingBytes);
        }
        Ok(proof)
    }
}

/// Exact decoding errors for a complete ParComm proof artifact.
#[derive(Debug)]
pub enum PackedI8ParCommProofDecodeError {
    /// Canonical deserialization failed.
    Serialization(SerializationError),
    /// Bytes remained after one canonical proof object.
    TrailingBytes,
}

impl Display for PackedI8ParCommProofDecodeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serialization(error) => Display::fmt(error, formatter),
            Self::TrailingBytes => write!(formatter, "ParComm proof has trailing bytes"),
        }
    }
}

impl std::error::Error for PackedI8ParCommProofDecodeError {}

impl From<SerializationError> for PackedI8ParCommProofDecodeError {
    fn from(error: SerializationError) -> Self {
        Self::Serialization(error)
    }
}

/// Errors raised before or during a streaming commitment/opening operation.
#[derive(Debug)]
pub enum VirtualPcsError {
    /// Bit width is unsupported for an `i8` source.
    InvalidBitWidth(usize),
    /// The BVA logical length is not a nonzero power of two.
    InvalidBvaLogicalEntries(usize),
    /// SmartPC parameters do not cover the packed matrix shape.
    ParametersTooSmall {
        /// Required row count.
        packed_rows: usize,
        /// Required column count.
        packed_cols: usize,
        /// Available SmartPC dimension.
        q: usize,
    },
    /// A physical value does not fit the selected magnitude width.
    ValueOutOfRange {
        /// Invalid signed value.
        value: i32,
        /// Selected magnitude width.
        bit_width: usize,
    },
    /// SmartPC opening point has an inconsistent shape.
    PointShape {
        /// Expected `(left, right)` challenge lengths.
        expected: (usize, usize),
        /// Actual `(left, right)` challenge lengths.
        actual: (usize, usize),
    },
    /// One streaming projection received the wrong challenge count.
    ProjectionPointShape {
        /// Expected challenge count.
        expected: usize,
        /// Actual challenge count.
        actual: usize,
    },
    /// The retained tier-one commitment vector has the wrong length.
    TierOneLength {
        /// Expected length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// The source emitted a vector index outside the BVA layout.
    BvaVectorIndex {
        /// Invalid index.
        vector_index: usize,
        /// Expected vector count.
        vector_count: usize,
    },
    /// The source emitted the same vector more than once.
    DuplicateBvaVector(usize),
    /// The source omitted a vector.
    MissingBvaVector(usize),
    /// The source emitted the wrong physical vector length.
    BvaVectorLength {
        /// Source vector index.
        vector_index: usize,
        /// Emitted physical length.
        physical_len: usize,
        /// Expected physical length.
        expected_len: usize,
        /// Logical cover bucket length.
        bucket_size: usize,
    },
}

impl Display for VirtualPcsError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for VirtualPcsError {}

/// A streaming PCS error that preserves source failures.
#[derive(Debug)]
pub enum StreamingVirtualPcsError<SourceError> {
    /// Protocol metadata or shape failure.
    Pcs(VirtualPcsError),
    /// Failure reported by the rewindable source.
    Source(SourceError),
}

impl<SourceError> From<VirtualPcsError> for StreamingVirtualPcsError<SourceError> {
    fn from(error: VirtualPcsError) -> Self {
        Self::Pcs(error)
    }
}

impl<SourceError: Display> Display for StreamingVirtualPcsError<SourceError> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pcs(error) => Display::fmt(error, formatter),
            Self::Source(error) => Display::fmt(error, formatter),
        }
    }
}

impl<SourceError> std::error::Error for StreamingVirtualPcsError<SourceError> where
    SourceError: std::error::Error + 'static
{
}

/// Failures from the complete commitment-bound prover pipeline.
#[derive(Debug)]
pub enum PackedI8ParCommProverError<SourceError> {
    /// Streaming commitment or opening failed.
    VirtualPcs(StreamingVirtualPcsError<SourceError>),
    /// Packed EO-LiteBullet failed.
    PackedEo(StreamingEoError<SourceError>),
    /// The locally reconstructed EO transcript rejected before opening.
    PackedEoRejected,
}

impl<SourceError: Display> Display for PackedI8ParCommProverError<SourceError> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::VirtualPcs(error) => Display::fmt(error, formatter),
            Self::PackedEo(error) => Display::fmt(error, formatter),
            Self::PackedEoRejected => write!(formatter, "packed EO-LiteBullet rejected"),
        }
    }
}

impl<SourceError> std::error::Error for PackedI8ParCommProverError<SourceError> where
    SourceError: std::error::Error + 'static
{
}

/// Failures detected before a deferred SmartPC equation can be returned.
#[derive(Debug)]
pub enum PackedI8ParCommVerificationError {
    /// Packed EO-LiteBullet rejected.
    PackedEoRejected,
    /// The value commitment is not the commitment to EO's final witness.
    ValueCommitmentMismatch,
    /// The proof does not use the field-width-derived packing stride.
    PackingStrideMismatch {
        /// Required packing stride.
        expected: usize,
        /// Encoded packing stride.
        actual: usize,
    },
    /// EO challenges cannot be mapped to the packed SmartPC matrix.
    VirtualPcs(VirtualPcsError),
    /// SmartPC rejected malformed proof metadata.
    SmartPc(poly_commit::Error),
}

impl Display for PackedI8ParCommVerificationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::PackedEoRejected => write!(formatter, "packed EO-LiteBullet rejected"),
            Self::ValueCommitmentMismatch => write!(formatter, "EO value commitment mismatch"),
            Self::PackingStrideMismatch { expected, actual } => write!(
                formatter,
                "packed EO stride mismatch: expected {expected}, got {actual}"
            ),
            Self::VirtualPcs(error) => Display::fmt(error, formatter),
            Self::SmartPc(error) => Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for PackedI8ParCommVerificationError {}

/// Exact decoding errors for cached public parameters.
#[derive(Debug)]
pub enum PublicParametersDecodeError {
    /// Canonical deserialization failed.
    Serialization(SerializationError),
    /// Bytes remained after one canonical parameter object.
    TrailingBytes,
}

impl Display for PublicParametersDecodeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serialization(error) => Display::fmt(error, formatter),
            Self::TrailingBytes => write!(formatter, "public parameter file has trailing bytes"),
        }
    }
}

impl std::error::Error for PublicParametersDecodeError {}

impl From<SerializationError> for PublicParametersDecodeError {
    fn from(error: SerializationError) -> Self {
        Self::Serialization(error)
    }
}

/// Tag identifying the fixed-layout SmartPC parameter cache.
const PP_CACHE_MAGIC: [u8; 8] = *b"PARCOMM1";

/// Default on-disk location of a reusable SmartPC parameter file.
///
/// Experiments share one file per SmartPC dimension so a sweep generates the
/// parameters at most once.
pub fn default_parameter_cache_path(qlog: usize) -> PathBuf {
    Path::new("data").join(format!("smartpc-q{qlog}.params"))
}

/// Writes one group vector as fixed-size uncompressed affine records.
fn write_group_vector<G: CurveGroup>(
    points: &[G],
    bytes: &mut Vec<u8>,
) -> Result<(), SerializationError> {
    let affine = G::normalize_batch(points);
    let element_size = G::Affine::zero().uncompressed_size();
    (affine.len() as u64).serialize_uncompressed(&mut *bytes)?;
    (element_size as u64).serialize_uncompressed(&mut *bytes)?;
    let start = bytes.len();
    bytes.resize(start + affine.len() * element_size, 0);
    bytes[start..]
        .par_chunks_mut(element_size)
        .zip(affine.par_iter())
        .try_for_each(|(slot, point)| point.serialize_uncompressed(slot))
}

/// Reads one group vector written by [`write_group_vector`].
fn read_group_vector<G: CurveGroup>(
    bytes: &[u8],
    cursor: &mut usize,
) -> Result<Vec<G>, SerializationError> {
    let mut header = Cursor::new(
        bytes
            .get(*cursor..*cursor + 16)
            .ok_or(SerializationError::NotEnoughSpace)?,
    );
    let count = u64::deserialize_uncompressed(&mut header)? as usize;
    let element_size = u64::deserialize_uncompressed(&mut header)? as usize;
    *cursor += 16;
    if element_size != G::Affine::zero().uncompressed_size() {
        return Err(SerializationError::InvalidData);
    }
    let total = count
        .checked_mul(element_size)
        .ok_or(SerializationError::InvalidData)?;
    let body = bytes
        .get(*cursor..*cursor + total)
        .ok_or(SerializationError::NotEnoughSpace)?;
    *cursor += total;
    // Fixed-size uncompressed records decode in parallel, and the unchecked
    // form skips the per-point square root and subgroup check that dominate a
    // compressed load.
    body.par_chunks(element_size)
        .map(|record| G::Affine::deserialize_uncompressed_unchecked(record).map(Into::into))
        .collect()
}

/// Serializes reusable SmartPC parameters for an external cache file.
///
/// Scalars and single group elements keep the canonical encoding; the five
/// large base vectors use a fixed-size uncompressed layout so a cached file
/// loads without per-point decompression.
pub fn serialize_public_parameters<E: Pairing>(
    parameters: &UniversalParams<E>,
) -> Result<Vec<u8>, SerializationError> {
    let mut header = Vec::new();
    parameters.q.serialize_uncompressed(&mut header)?;
    parameters.num_vars.serialize_uncompressed(&mut header)?;
    parameters.g_hat.serialize_uncompressed(&mut header)?;
    parameters.h_hat.serialize_uncompressed(&mut header)?;
    parameters.s_h.serialize_uncompressed(&mut header)?;
    parameters.sq_h.serialize_uncompressed(&mut header)?;
    parameters.nu_g.serialize_uncompressed(&mut header)?;
    parameters.nu_h.serialize_uncompressed(&mut header)?;
    parameters.g_0.serialize_uncompressed(&mut header)?;
    parameters.tilde_g.serialize_uncompressed(&mut header)?;
    parameters.tilde_h.serialize_uncompressed(&mut header)?;
    parameters.u.serialize_uncompressed(&mut header)?;
    parameters.tilde_u.serialize_uncompressed(&mut header)?;

    let mut bytes = Vec::new();
    bytes.extend_from_slice(&PP_CACHE_MAGIC);
    (header.len() as u64).serialize_uncompressed(&mut bytes)?;
    bytes.extend_from_slice(&header);
    write_group_vector(&parameters.vec_g, &mut bytes)?;
    write_group_vector(&parameters.vec_h, &mut bytes)?;
    write_group_vector(&parameters.vec_g_prime, &mut bytes)?;
    write_group_vector(&parameters.vec_nu_g, &mut bytes)?;
    write_group_vector(&parameters.vec_nu_h, &mut bytes)?;
    Ok(bytes)
}

/// Decodes exactly one cached SmartPC parameter object.
pub fn deserialize_public_parameters_exact<E: Pairing>(
    bytes: &[u8],
) -> Result<UniversalParams<E>, PublicParametersDecodeError> {
    if bytes.len() < 16 || bytes[..8] != PP_CACHE_MAGIC {
        return Err(PublicParametersDecodeError::Serialization(
            SerializationError::InvalidData,
        ));
    }
    let mut length_reader = Cursor::new(&bytes[8..16]);
    let header_len = u64::deserialize_uncompressed(&mut length_reader)? as usize;
    let mut cursor = 16 + header_len;
    let header = bytes
        .get(16..cursor)
        .ok_or(SerializationError::NotEnoughSpace)?;
    let mut header = Cursor::new(header);

    let q = usize::deserialize_uncompressed(&mut header)?;
    let num_vars = usize::deserialize_uncompressed(&mut header)?;
    let g_hat = E::G1::deserialize_uncompressed_unchecked(&mut header)?;
    let h_hat = E::G2::deserialize_uncompressed_unchecked(&mut header)?;
    let s_h = E::G2::deserialize_uncompressed_unchecked(&mut header)?;
    let sq_h = E::G2::deserialize_uncompressed_unchecked(&mut header)?;
    let nu_g = E::G1::deserialize_uncompressed_unchecked(&mut header)?;
    let nu_h = E::G2::deserialize_uncompressed_unchecked(&mut header)?;
    let g_0 = E::G1::deserialize_uncompressed_unchecked(&mut header)?;
    let tilde_g = E::G1::deserialize_uncompressed_unchecked(&mut header)?;
    let tilde_h = E::G2::deserialize_uncompressed_unchecked(&mut header)?;
    let u = PairingOutput::<E>::deserialize_uncompressed_unchecked(&mut header)?;
    let tilde_u = PairingOutput::<E>::deserialize_uncompressed_unchecked(&mut header)?;
    if header.position() != header_len as u64 {
        return Err(PublicParametersDecodeError::TrailingBytes);
    }

    let vec_g = read_group_vector::<E::G1>(bytes, &mut cursor)?;
    let vec_h = read_group_vector::<E::G2>(bytes, &mut cursor)?;
    let vec_g_prime = read_group_vector::<E::G1>(bytes, &mut cursor)?;
    let vec_nu_g = read_group_vector::<E::G1>(bytes, &mut cursor)?;
    let vec_nu_h = read_group_vector::<E::G2>(bytes, &mut cursor)?;
    if cursor != bytes.len() {
        return Err(PublicParametersDecodeError::TrailingBytes);
    }

    Ok(UniversalParams {
        q,
        num_vars,
        g_hat,
        h_hat,
        s_h,
        sq_h,
        nu_g,
        nu_h,
        g_0,
        tilde_g,
        tilde_h,
        u,
        tilde_u,
        vec_g,
        vec_h,
        vec_g_prime,
        vec_nu_g,
        vec_nu_h,
    })
}

/// Derives the packed EO binding from the commitment and compact statement.
///
/// The compact TBV absorption includes the decreasing BVA layout, physical
/// vector lengths, bucket offsets, and tensor terms without expanding logical
/// zero suffixes or a length-`K` coefficient vector.
pub fn packed_i8_parcomm_statement_binding<E: Pairing>(
    commitment: &PairingOutput<E>,
    tbv: &TensorBasisVector<E::ScalarField>,
    claimed: E::ScalarField,
    bit_width: usize,
) -> E::ScalarField
where
    E::ScalarField: Absorb,
{
    let mut transcript = crate::fiat_shamir::FiatShamir::<E::ScalarField>::new();
    transcript.push(&0x5041_5243_4f4d_4d31_u64);
    transcript.push(&(bit_width as u64));
    transcript.push(commitment);
    tbv.absorb_transcript(&mut transcript);
    transcript.push(&claimed);
    transcript.gen_challenge()
}

/// Splits a logical BVA length into the packed `(rows, cols)` SmartPC shape.
pub fn packed_bva_shape(logical_entries: usize) -> Result<(usize, usize), VirtualPcsError> {
    if logical_entries == 0 || !logical_entries.is_power_of_two() {
        return Err(VirtualPcsError::InvalidBvaLogicalEntries(logical_entries));
    }
    let packed_cols = 1_usize << (logical_entries.ilog2() as usize / 2);
    Ok((logical_entries / packed_cols, packed_cols))
}

/// Combines signed magnitude buckets into `sum_k k * bucket[k]` and clears them.
///
/// The running-sum form costs `2 * max_magnitude` group additions regardless of
/// how many witness entries were accumulated, so one column pays it once instead
/// of paying `popcount(|value|)` additions per entry.
fn drain_magnitude_buckets_field<F: Field>(buckets: &mut [F], max_magnitude: usize) -> F {
    let mut running = F::zero();
    let mut total = F::zero();
    for magnitude in (1..=max_magnitude).rev() {
        running += buckets[magnitude];
        total += running;
        buckets[magnitude] = F::zero();
    }
    total
}

/// Largest magnitude in one witness vector.
fn max_magnitude(values: &[i8]) -> usize {
    usize::from(
        values
            .par_iter()
            .map(|&value| value.unsigned_abs())
            .max()
            .unwrap_or(0),
    )
}

fn drain_magnitude_buckets<E: Pairing>(buckets: &mut [E::G1], max_magnitude: usize) -> E::G1 {
    let mut running = E::G1::zero();
    let mut total = E::G1::zero();
    for magnitude in (1..=max_magnitude).rev() {
        running += buckets[magnitude];
        total += running;
        buckets[magnitude] = E::G1::zero();
    }
    total
}

/// Largest magnitude in one witness vector, or `None` when a value exceeds the
/// configured bit width.
fn scan_max_magnitude(values: &[i8], bit_width: usize) -> Result<u8, i8> {
    values
        .par_iter()
        .try_fold(
            || 0_u8,
            |current, &value| {
                let magnitude = value.unsigned_abs();
                if bit_width < 8 && magnitude >= (1_u8 << bit_width) {
                    Err(value)
                } else {
                    Ok(current.max(magnitude))
                }
            },
        )
        .try_reduce(|| 0_u8, |left, right| Ok(left.max(right)))
}

/// Commits a byte-backed BVA without widening it to a field vector.
///
/// Each parallel chunk produces at most one G1 partial sum per packed column,
/// so live group storage is bounded by `O(sqrt(K))`, not by physical witness
/// length.
pub fn commit_bva_i8_streaming<E: Pairing, S: BvaI8Source>(
    pp: &UniversalParams<E>,
    source: &mut S,
    hiding_factor: E::ScalarField,
    bit_width: usize,
) -> Result<
    (PairingOutput<E>, Vec<E::G1>, BvaStreamingCommitProfile),
    StreamingVirtualPcsError<S::Error>,
> {
    if bit_width == 0 || bit_width > 8 {
        return Err(VirtualPcsError::InvalidBitWidth(bit_width).into());
    }
    let layout = source.layout().clone();
    let logical_entries = layout.logical_len();
    let (packed_rows, packed_cols) = packed_bva_shape(logical_entries)?;
    if packed_rows > pp.vec_g.len() || packed_cols > pp.vec_h.len() {
        return Err(VirtualPcsError::ParametersTooSmall {
            packed_rows,
            packed_cols,
            q: pp.q,
        }
        .into());
    }

    // Mixed additions need affine bases, and one batch inversion replaces one
    // inversion per base.
    let base_affine = E::G1::normalize_batch(&pp.vec_g[..packed_rows]);
    let bucket_offsets = layout.bucket_offsets();
    let mut tier_one = vec![E::G1::zero(); packed_cols];
    let mut visited = vec![false; layout.vector_to_bucket().len()];
    let mut profile = BvaStreamingCommitProfile {
        logical_entries,
        prepared_base_entries: packed_rows,
        ..Default::default()
    };
    let column_block =
        COMMIT_COLUMN_BLOCK.min(packed_cols.div_ceil(rayon::current_num_threads()).max(1));
    let chunk_entries = COMMIT_ROWS_PER_CHUNK * packed_cols;
    let mut pcs_error = None;

    source
        .for_each_vector(&mut |vector_index, values| {
            if pcs_error.is_some() {
                return;
            }
            let Some(&bucket_index) = layout.vector_to_bucket().get(vector_index) else {
                pcs_error = Some(VirtualPcsError::BvaVectorIndex {
                    vector_index,
                    vector_count: visited.len(),
                });
                return;
            };
            if visited[vector_index] {
                pcs_error = Some(VirtualPcsError::DuplicateBvaVector(vector_index));
                return;
            }
            visited[vector_index] = true;
            let expected_len = layout.vector_lengths()[vector_index];
            if values.len() != expected_len {
                pcs_error = Some(VirtualPcsError::BvaVectorLength {
                    vector_index,
                    physical_len: values.len(),
                    expected_len,
                    bucket_size: layout.bucket_sizes()[bucket_index],
                });
                return;
            }
            let max_magnitude = match scan_max_magnitude(values, bit_width) {
                Ok(max_magnitude) => usize::from(max_magnitude),
                Err(value) => {
                    pcs_error = Some(VirtualPcsError::ValueOutOfRange {
                        value: i32::from(value),
                        bit_width,
                    });
                    return;
                }
            };

            profile.physical_entries += values.len();
            profile.parallel_vectors += usize::from(!values.is_empty());
            if max_magnitude == 0 {
                return;
            }
            profile.max_live_group_contributions =
                profile.max_live_group_contributions.max(max_magnitude + 1);

            let logical_offset = bucket_offsets[bucket_index];
            let flat_end = logical_offset + values.len();
            for chunk_start in (logical_offset..flat_end).step_by(chunk_entries) {
                let chunk_end = (chunk_start + chunk_entries).min(flat_end);
                let start_column = chunk_start % packed_cols;
                profile.nonzero_entries += tier_one
                    .par_chunks_mut(column_block)
                    .enumerate()
                    .map(|(block_index, block)| {
                        let mut buckets = vec![E::G1::zero(); max_magnitude + 1];
                        let mut nonzero = 0_usize;
                        for (offset, accumulator) in block.iter_mut().enumerate() {
                            let packed_col = block_index * column_block + offset;
                            let delta = (packed_col + packed_cols - start_column) % packed_cols;
                            let mut flat_index = chunk_start + delta;
                            let mut row = flat_index / packed_cols;
                            let mut column_max = 0_usize;
                            while flat_index < chunk_end {
                                let value = values[flat_index - logical_offset];
                                if value != 0 {
                                    let magnitude = usize::from(value.unsigned_abs());
                                    let base = base_affine[row];
                                    if value > 0 {
                                        buckets[magnitude] += base;
                                    } else {
                                        buckets[magnitude] -= base;
                                    }
                                    column_max = column_max.max(magnitude);
                                    nonzero += 1;
                                }
                                flat_index += packed_cols;
                                row += 1;
                            }
                            if column_max != 0 {
                                *accumulator +=
                                    drain_magnitude_buckets::<E>(&mut buckets, column_max);
                            }
                        }
                        nonzero
                    })
                    .sum::<usize>();
            }
        })
        .map_err(StreamingVirtualPcsError::Source)?;

    if let Some(error) = pcs_error {
        return Err(error.into());
    }
    if let Some(vector_index) = visited.iter().position(|visited| !visited) {
        return Err(VirtualPcsError::MissingBvaVector(vector_index).into());
    }

    let commitment = pcutils::inner_pairing_product::<E>(&tier_one, &pp.vec_h[..packed_cols])
        + pp.tilde_u.mul(hiding_factor);
    Ok((commitment, tier_one, profile))
}

/// Converts low-bit-first EO inverse challenges into SmartPC's packed
/// high-bit-first `(row, column)` challenge convention.
pub fn bva_eo_opening_point<F: Copy>(
    logical_entries: usize,
    challenges_inv: &[F],
) -> Result<(Vec<F>, Vec<F>), VirtualPcsError> {
    let (_, packed_cols) = packed_bva_shape(logical_entries)?;
    let expected = logical_entries.ilog2() as usize;
    if challenges_inv.len() != expected {
        return Err(VirtualPcsError::ProjectionPointShape {
            expected,
            actual: challenges_inv.len(),
        });
    }
    let col_log = packed_cols.ilog2() as usize;
    let xr = challenges_inv[..col_log].iter().rev().copied().collect();
    let xl = challenges_inv[col_log..].iter().rev().copied().collect();
    Ok((xl, xr))
}

#[derive(Clone, Copy)]
enum ProjectionSide {
    Left,
    Right,
}

/// Packed columns handled by one parallel task in a left projection.
const PROJECTION_COLUMN_BLOCK: usize = 64;

/// Packed rows swept before a left projection restarts at the first column.
const PROJECTION_ROWS_PER_CHUNK: usize = 512;

fn project_packed_bva_i8_streaming<E, S>(
    source: &mut S,
    challenges: &[E::ScalarField],
    side: ProjectionSide,
) -> Result<(Vec<E::ScalarField>, StreamingOpenProfile), StreamingVirtualPcsError<S::Error>>
where
    E: Pairing,
    S: BvaI8Source,
{
    let layout = source.layout().clone();
    let logical_entries = layout.logical_len();
    let (packed_rows, packed_cols) = packed_bva_shape(logical_entries)?;
    let (challenge_axis, output_axis) = match side {
        ProjectionSide::Left => (packed_rows, packed_cols),
        ProjectionSide::Right => (packed_cols, packed_rows),
    };
    let expected_challenges = challenge_axis.ilog2() as usize;
    if challenges.len() != expected_challenges {
        return Err(VirtualPcsError::ProjectionPointShape {
            expected: expected_challenges,
            actual: challenges.len(),
        }
        .into());
    }
    let weights = poly_commit::smart_pc::xi::xi_from_challenges::<E>(&challenges.to_vec());
    let bucket_offsets = layout.bucket_offsets();
    let mut projection = vec![E::ScalarField::zero(); output_axis];
    let mut visited = vec![false; layout.vector_to_bucket().len()];
    let mut profile = StreamingOpenProfile {
        source_passes: 1,
        ..Default::default()
    };
    let mut pcs_error = None;

    source
        .for_each_vector(&mut |vector_index, values| {
            if pcs_error.is_some() {
                return;
            }
            let Some(&bucket_index) = layout.vector_to_bucket().get(vector_index) else {
                pcs_error = Some(VirtualPcsError::BvaVectorIndex {
                    vector_index,
                    vector_count: visited.len(),
                });
                return;
            };
            if visited[vector_index] {
                pcs_error = Some(VirtualPcsError::DuplicateBvaVector(vector_index));
                return;
            }
            visited[vector_index] = true;
            let expected_len = layout.vector_lengths()[vector_index];
            if values.len() != expected_len {
                pcs_error = Some(VirtualPcsError::BvaVectorLength {
                    vector_index,
                    physical_len: values.len(),
                    expected_len,
                    bucket_size: layout.bucket_sizes()[bucket_index],
                });
                return;
            }
            let logical_offset = bucket_offsets[bucket_index];
            let logical_end = logical_offset + values.len();
            profile.physical_entries += values.len();
            let magnitude_bound = max_magnitude(values);
            if magnitude_bound == 0 {
                return;
            }
            // A byte witness only ever scales a weight by a small integer, so
            // accumulating weights into magnitude buckets replaces one field
            // multiplication per entry with one field addition.
            let nonzero_entries = match side {
                ProjectionSide::Left => {
                    let column_block = PROJECTION_COLUMN_BLOCK
                        .min(packed_cols.div_ceil(rayon::current_num_threads()).max(1));
                    let chunk_entries = PROJECTION_ROWS_PER_CHUNK * packed_cols;
                    let mut nonzero = 0_usize;
                    // Column-major sweeps stride by a full row, so each chunk
                    // covers only as many rows as stay resident in the TLB.
                    for chunk_start in (logical_offset..logical_end).step_by(chunk_entries) {
                        let chunk_end = (chunk_start + chunk_entries).min(logical_end);
                        let start_column = chunk_start % packed_cols;
                        nonzero += projection
                            .par_chunks_mut(column_block)
                            .enumerate()
                            .map(|(block_index, block)| {
                                let mut buckets = vec![E::ScalarField::zero(); magnitude_bound + 1];
                                let mut nonzero = 0_usize;
                                for (offset, output) in block.iter_mut().enumerate() {
                                    let packed_col = block_index * column_block + offset;
                                    let delta =
                                        (packed_col + packed_cols - start_column) % packed_cols;
                                    let mut flat_index = chunk_start + delta;
                                    let mut row = flat_index / packed_cols;
                                    let mut column_max = 0_usize;
                                    while flat_index < chunk_end {
                                        let value = values[flat_index - logical_offset];
                                        if value != 0 {
                                            let magnitude = usize::from(value.unsigned_abs());
                                            if value > 0 {
                                                buckets[magnitude] += weights[row];
                                            } else {
                                                buckets[magnitude] -= weights[row];
                                            }
                                            column_max = column_max.max(magnitude);
                                            nonzero += 1;
                                        }
                                        flat_index += packed_cols;
                                        row += 1;
                                    }
                                    if column_max != 0 {
                                        *output +=
                                            drain_magnitude_buckets_field(&mut buckets, column_max);
                                    }
                                }
                                nonzero
                            })
                            .sum::<usize>();
                    }
                    nonzero
                }
                ProjectionSide::Right => projection
                    .par_iter_mut()
                    .enumerate()
                    .map_with(
                        vec![E::ScalarField::zero(); magnitude_bound + 1],
                        |buckets, (packed_row, output)| {
                            let row_start = packed_row * packed_cols;
                            let start = row_start.max(logical_offset);
                            let end = (row_start + packed_cols).min(logical_end);
                            let mut column_max = 0_usize;
                            let mut nonzero = 0_usize;
                            for flat_index in start..end {
                                let value = values[flat_index - logical_offset];
                                if value != 0 {
                                    let magnitude = usize::from(value.unsigned_abs());
                                    let weight = weights[flat_index - row_start];
                                    if value > 0 {
                                        buckets[magnitude] += weight;
                                    } else {
                                        buckets[magnitude] -= weight;
                                    }
                                    column_max = column_max.max(magnitude);
                                    nonzero += 1;
                                }
                            }
                            if column_max != 0 {
                                *output += drain_magnitude_buckets_field(buckets, column_max);
                            }
                            nonzero
                        },
                    )
                    .sum::<usize>(),
            };
            profile.nonzero_entries += nonzero_entries;
            profile.real_projection_field_multiplications += nonzero_entries;
        })
        .map_err(StreamingVirtualPcsError::Source)?;

    if let Some(error) = pcs_error {
        return Err(error.into());
    }
    if let Some(vector_index) = visited.iter().position(|visited| !visited) {
        return Err(VirtualPcsError::MissingBvaVector(vector_index).into());
    }
    Ok((projection, profile))
}

fn check_opening_inputs<E: Pairing>(
    pp: &UniversalParams<E>,
    rows: usize,
    cols: usize,
    xl_len: usize,
    xr_len: usize,
    tier_one_len: usize,
) -> Result<(), VirtualPcsError> {
    if rows > pp.vec_g.len() || cols > pp.vec_h.len() {
        return Err(VirtualPcsError::ParametersTooSmall {
            packed_rows: rows,
            packed_cols: cols,
            q: pp.q,
        });
    }
    let expected_xl = rows.ilog2() as usize;
    let expected_xr = cols.ilog2() as usize;
    if xl_len != expected_xl || xr_len != expected_xr {
        return Err(VirtualPcsError::PointShape {
            expected: (expected_xl, expected_xr),
            actual: (xl_len, xr_len),
        });
    }
    if tier_one_len != cols {
        return Err(VirtualPcsError::TierOneLength {
            expected: cols,
            actual: tier_one_len,
        });
    }
    Ok(())
}

/// Opens the packed SmartPC view of a byte-backed BVA at one MLE point.
#[allow(clippy::too_many_arguments)]
pub fn open_bva_i8_streaming<E, S>(
    pp: &UniversalParams<E>,
    source: &mut S,
    xl: &[E::ScalarField],
    xr: &[E::ScalarField],
    value_commitment: PairingOutput<E>,
    matrix_commitment: PairingOutput<E>,
    tier_one: &[E::G1],
    value_hiding: E::ScalarField,
    matrix_hiding: E::ScalarField,
) -> Result<(Trans<E>, StreamingOpenProfile), StreamingVirtualPcsError<S::Error>>
where
    E: Pairing,
    S: BvaI8Source,
{
    let logical_entries = source.layout().logical_len();
    let (packed_rows, packed_cols) = packed_bva_shape(logical_entries)?;
    check_opening_inputs(
        pp,
        packed_rows,
        packed_cols,
        xl.len(),
        xr.len(),
        tier_one.len(),
    )?;
    let (value_current, left_profile) =
        project_packed_bva_i8_streaming::<E, S>(source, xl, ProjectionSide::Left)?;
    let mut profile = left_profile;
    let proof = open_from_left_projection::<E, StreamingVirtualPcsError<S::Error>>(
        pp,
        packed_rows,
        packed_cols,
        xl,
        xr,
        value_commitment,
        matrix_commitment,
        value_hiding,
        matrix_hiding,
        value_current,
        |challenges| {
            let (projection, right_profile) =
                project_packed_bva_i8_streaming::<E, S>(source, challenges, ProjectionSide::Right)?;
            profile.source_passes += right_profile.source_passes;
            profile.physical_entries += right_profile.physical_entries;
            profile.nonzero_entries += right_profile.nonzero_entries;
            profile.real_projection_field_multiplications +=
                right_profile.real_projection_field_multiplications;
            Ok(projection)
        },
        &mut HostColumnVectors::<E, _>::new(
            tier_one[..packed_cols].to_vec(),
            pp.vec_h[..packed_cols].to_vec(),
        ),
        &mut OpenProfile::default(),
    )?;
    Ok((proof, profile))
}

/// Runs the Dory reduction that follows the left projection.
///
/// Everything here is proportional to the packed side lengths rather than to
/// the witness, so an accelerated prover can compute the two projections
/// elsewhere and still reuse this reduction.
#[allow(clippy::too_many_arguments)]
/// The two group vectors the opening's column rounds reduce.
///
/// Each round pairs the halves of the tier-one vector and the G2 basis against
/// each other and then folds both by a challenge. An accelerator can hold them
/// in device memory for the whole reduction, so the interface never hands back
/// the vectors themselves, only the pairings and the final basis point.
pub trait OpeningColumnVectors<E: Pairing> {
    type Error;

    /// Pairs the lower half of each vector against the upper half of the other,
    /// in that order.
    fn cross_pairings(&mut self) -> Result<(PairingOutput<E>, PairingOutput<E>), Self::Error>;

    /// Folds each vector into its lower half, halving the live length.
    fn fold(&mut self, tier_one: E::ScalarField, basis: E::ScalarField) -> Result<(), Self::Error>;

    /// The single G2 point left once every round has run.
    fn reduced_basis(&self) -> Result<E::G2, Self::Error>;
}

/// Runs the column rounds on the host.
///
/// Nothing here fails, so the error type is whatever the surrounding opening
/// reports.
pub struct HostColumnVectors<E: Pairing, Error> {
    tier_one: Vec<E::G1>,
    basis: Vec<E::G2>,
    live: usize,
    reported_error: PhantomData<Error>,
}

impl<E: Pairing, Error> HostColumnVectors<E, Error> {
    pub fn new(tier_one: Vec<E::G1>, basis: Vec<E::G2>) -> Self {
        assert_eq!(tier_one.len(), basis.len());
        let live = tier_one.len();
        Self {
            tier_one,
            basis,
            live,
            reported_error: PhantomData,
        }
    }
}

impl<E: Pairing, Error> OpeningColumnVectors<E> for HostColumnVectors<E, Error> {
    type Error = Error;

    fn cross_pairings(&mut self) -> Result<(PairingOutput<E>, PairingOutput<E>), Self::Error> {
        let half = self.live / 2;
        Ok((
            pcutils::inner_pairing_product::<E>(
                &self.tier_one[..half],
                &self.basis[half..self.live],
            ),
            pcutils::inner_pairing_product::<E>(
                &self.tier_one[half..self.live],
                &self.basis[..half],
            ),
        ))
    }

    fn fold(&mut self, tier_one: E::ScalarField, basis: E::ScalarField) -> Result<(), Self::Error> {
        let half = self.live / 2;
        self.tier_one = pcutils::add_vec_g1::<E>(
            &self.tier_one[..half],
            &pcutils::scalar_mul_vec_g1::<E>(&self.tier_one[half..self.live], &tier_one),
        );
        self.basis = pcutils::add_vec_g2::<E>(
            &self.basis[..half],
            &pcutils::scalar_mul_vec_g2::<E>(&self.basis[half..self.live], &basis),
        );
        self.live = half;
        Ok(())
    }

    fn reduced_basis(&self) -> Result<E::G2, Self::Error> {
        Ok(self.basis[0])
    }
}

/// Where the SmartPC opening spends its time.
///
/// The column rounds and the row rounds cost very different things, so the
/// stages are split by the primitive they are bound by rather than by round.
#[derive(Clone, Copy, Debug, Default)]
pub struct OpenProfile {
    /// Multi-Miller loops plus final exponentiations in the column rounds.
    pub pairing_seconds: f64,
    /// Folding the tier-one vector and the G2 basis in the column rounds.
    pub column_fold_seconds: f64,
    /// The right projection between the two round loops.
    pub projection_seconds: f64,
    /// G1 multi-scalar multiplications in the row rounds.
    pub row_msm_seconds: f64,
    /// Folding the G1 basis in the row rounds.
    pub row_fold_seconds: f64,
    /// Proof-of-inner-pairing and the closing linear argument.
    pub tail_seconds: f64,
}

pub fn open_from_left_projection<E: Pairing, ProjectionError>(
    pp: &UniversalParams<E>,
    rows: usize,
    cols: usize,
    xl: &[E::ScalarField],
    xr: &[E::ScalarField],
    value_commitment: PairingOutput<E>,
    matrix_commitment: PairingOutput<E>,
    value_hiding: E::ScalarField,
    matrix_hiding: E::ScalarField,
    mut value_current: Vec<E::ScalarField>,
    mut project_right: impl FnMut(&[E::ScalarField]) -> Result<Vec<E::ScalarField>, ProjectionError>,
    columns: &mut impl OpeningColumnVectors<E, Error = ProjectionError>,
    profile: &mut OpenProfile,
) -> Result<Trans<E>, ProjectionError> {
    let mut fs = FiatShamir::new();
    fs.push(&matrix_commitment);
    fs.push(&value_commitment);
    let xl_vec = xl.to_vec();
    let xr_vec = xr.to_vec();
    fs.push(&xl_vec);
    fs.push(&xr_vec);
    let x = fs.gen_challenge::<E>();

    let log_rows = rows.ilog2() as usize;
    let log_cols = cols.ilog2() as usize;
    let rng = &mut ark_std::rand::thread_rng();
    let u_0 = E::pairing(pp.g_0, pp.h_hat);
    let l_vec = poly_commit::smart_pc::xi::xi_from_challenges::<E>(&xl_vec);
    let r_vec = poly_commit::smart_pc::xi::xi_from_challenges::<E>(&xr_vec);
    let mut accumulated_hiding = value_hiding + x * matrix_hiding;

    let mut vec_l_gt = Vec::with_capacity(log_cols);
    let mut vec_r_gt = Vec::with_capacity(log_cols);
    let mut r_current = r_vec[..cols].to_vec();
    let mut col_challenges = Vec::with_capacity(log_cols);
    let mut col_challenges_inv = Vec::with_capacity(log_cols);

    for round in 0..log_cols {
        let current_len = cols >> round;
        let midpoint = current_len / 2;
        let value_left = value_current[..midpoint].to_vec();
        let value_right = value_current[midpoint..current_len].to_vec();
        let r_left = r_current[..midpoint].to_vec();
        let r_right = r_current[midpoint..current_len].to_vec();

        let left_hiding = E::ScalarField::rand(rng);
        let right_hiding = E::ScalarField::rand(rng);
        let pairing_started = Instant::now();
        let (lower_pairing, upper_pairing) = columns.cross_pairings()?;
        profile.pairing_seconds += pairing_started.elapsed().as_secs_f64();
        let left = lower_pairing.mul(x)
            + u_0.mul(pcutils::inner_product::<E>(&value_left, &r_right))
            + pp.tilde_u.mul(left_hiding);
        let right = upper_pairing.mul(x)
            + u_0.mul(pcutils::inner_product::<E>(&value_right, &r_left))
            + pp.tilde_u.mul(right_hiding);
        vec_l_gt.push(left);
        vec_r_gt.push(right);
        fs.push(&left);
        fs.push(&right);

        let challenge = fs.gen_nonzero_challenge::<E>();
        let challenge_inv = challenge
            .inverse()
            .expect("SmartPC samples nonzero challenges");
        accumulated_hiding += left_hiding * challenge + right_hiding * challenge_inv;
        col_challenges.push(challenge);
        col_challenges_inv.push(challenge_inv);
        value_current = pcutils::add_vec_zp::<E>(
            &value_left,
            &pcutils::scalar_mul_vec_zp::<E>(&value_right, &challenge_inv),
        );
        let fold_started = Instant::now();
        columns.fold(challenge_inv, challenge)?;
        profile.column_fold_seconds += fold_started.elapsed().as_secs_f64();
        r_current = pcutils::add_vec_zp::<E>(
            &r_left,
            &pcutils::scalar_mul_vec_zp::<E>(&r_right, &challenge),
        );
    }

    let projection_started = Instant::now();
    let mut a_current = project_right(&col_challenges_inv)?;
    profile.projection_seconds += projection_started.elapsed().as_secs_f64();
    let h_reduced = columns.reduced_basis()?;
    let r_reduced = r_current[0];
    let mut g_current = pp.vec_g[..rows].to_vec();
    let mut l_current = l_vec[..rows].to_vec();
    let mut row_challenges = Vec::with_capacity(log_rows);
    let mut vec_l_native = Vec::with_capacity(log_rows);
    let mut vec_r_native = Vec::with_capacity(log_rows);

    for round in 0..log_rows {
        let current_len = rows >> round;
        let midpoint = current_len / 2;
        let a_left = a_current[..midpoint].to_vec();
        let a_right = a_current[midpoint..current_len].to_vec();
        let l_left = l_current[..midpoint].to_vec();
        let l_right = l_current[midpoint..current_len].to_vec();
        let g_left = g_current[..midpoint].to_vec();
        let g_right = g_current[midpoint..current_len].to_vec();

        let left_cancel = E::ScalarField::rand(rng);
        let left_hiding = E::ScalarField::rand(rng);
        let right_cancel = E::ScalarField::rand(rng);
        let right_hiding = E::ScalarField::rand(rng);
        let msm_started = Instant::now();
        let left = NativeFoldMessage::<E> {
            g1: pcutils::msm_g1::<E>(&g_right, &a_left).mul(x) + pp.g_0.mul(left_cancel),
            g2: pp
                .h_hat
                .mul(r_reduced * pcutils::inner_product::<E>(&a_left, &l_right))
                - h_reduced.mul(left_cancel)
                + pp.tilde_h.mul(left_hiding),
        };
        let right = NativeFoldMessage::<E> {
            g1: pcutils::msm_g1::<E>(&g_left, &a_right).mul(x) + pp.g_0.mul(right_cancel),
            g2: pp
                .h_hat
                .mul(r_reduced * pcutils::inner_product::<E>(&a_right, &l_left))
                - h_reduced.mul(right_cancel)
                + pp.tilde_h.mul(right_hiding),
        };
        profile.row_msm_seconds += msm_started.elapsed().as_secs_f64();
        vec_l_native.push(left);
        vec_r_native.push(right);
        fs.push(&left);
        fs.push(&right);

        let challenge = fs.gen_nonzero_challenge::<E>();
        let challenge_inv = challenge
            .inverse()
            .expect("SmartPC samples nonzero challenges");
        accumulated_hiding += left_hiding * challenge + right_hiding * challenge_inv;
        row_challenges.push(challenge);
        a_current = pcutils::add_vec_zp::<E>(
            &a_left,
            &pcutils::scalar_mul_vec_zp::<E>(&a_right, &challenge_inv),
        );
        let row_fold_started = Instant::now();
        g_current = pcutils::add_vec_g1::<E>(
            &g_left,
            &pcutils::scalar_mul_vec_g1::<E>(&g_right, &challenge),
        );
        profile.row_fold_seconds += row_fold_started.elapsed().as_secs_f64();
        l_current = pcutils::add_vec_zp::<E>(
            &l_left,
            &pcutils::scalar_mul_vec_zp::<E>(&l_right, &challenge),
        );
    }

    let tail_started = Instant::now();
    let a_reduced = a_current[0];
    let g_reduced = g_current[0];
    let xi_l = poly_commit::smart_pc::xi::xi_ip_from_challenges::<E>(&xl_vec, &row_challenges);
    let xi_r = poly_commit::smart_pc::xi::xi_ip_from_challenges::<E>(&xr_vec, &col_challenges);
    let base_rhs = u_0.mul(xi_l * xi_r) + E::pairing(g_reduced.mul(x), h_reduced);
    fs.push(&g_reduced);
    fs.push(&h_reduced);
    let pip_g1_challenge = fs.gen_challenge::<E>();
    let (v_g_prime, w_g) =
        sub_protocols::pip_g1_prove_with_challenge::<E>(pp, &row_challenges, pip_g1_challenge);
    fs.push(&v_g_prime);
    fs.push(&w_g);
    let pip_g2_challenge = fs.gen_challenge::<E>();
    let (v_h_prime, w_h) =
        sub_protocols::pip_g2_prove_with_challenge::<E>(pp, &col_challenges, pip_g2_challenge);
    fs.push(&v_h_prime);
    fs.push(&w_h);
    let (linear_f, linear_z_a, linear_z_t) = sub_protocols::linear_prove::<E, _>(
        base_rhs,
        pp.tilde_u,
        a_reduced,
        accumulated_hiding,
        &mut fs,
        rng,
    );

    profile.tail_seconds += tail_started.elapsed().as_secs_f64();
    Ok(Trans::<E> {
        vec_l_gt,
        vec_r_gt,
        vec_l_native,
        vec_r_native,
        v_g: g_reduced,
        v_h: h_reduced,
        v_g_prime,
        v_h_prime,
        w_g,
        w_h,
        linear_f,
        linear_z_a,
        linear_z_t,
    })
}

/// Runs commit, commitment-bound packed EO-LiteBullet, and SmartPC opening.
///
/// Public parameters are supplied by the caller so a cached setup can be
/// reused across many proofs.
pub fn prove_packed_i8_parcomm<E, S>(
    pp: &UniversalParams<E>,
    source: &mut S,
    tbv: &TensorBasisVector<E::ScalarField>,
    claimed: E::ScalarField,
    matrix_hiding: E::ScalarField,
    bit_width: usize,
) -> Result<(PackedI8ParCommProof<E>, PackedI8ParCommProfile), PackedI8ParCommProverError<S::Error>>
where
    E: Pairing,
    E::ScalarField: Absorb,
    S: BvaI8Source,
{
    let (matrix_commitment, tier_one, commitment_profile) =
        commit_bva_i8_streaming::<E, S>(pp, source, matrix_hiding, bit_width)
            .map_err(PackedI8ParCommProverError::VirtualPcs)?;
    prove_packed_i8_parcomm_precommitted::<E, S>(
        pp,
        source,
        tbv,
        claimed,
        matrix_hiding,
        bit_width,
        matrix_commitment,
        &tier_one,
        commitment_profile,
    )
}

/// Completes packed EO and the SmartPC opening from an existing byte-BVA
/// commitment.
///
/// Collection protocols use this seam to derive their batching challenge only
/// after the dense witness commitment is fixed. The supplied tier-one vector
/// and profile must come from [`commit_bva_i8_streaming`] with the same source,
/// parameters, hiding factor, and bit width.
#[allow(clippy::too_many_arguments)]
pub fn prove_packed_i8_parcomm_precommitted<E, S>(
    pp: &UniversalParams<E>,
    source: &mut S,
    tbv: &TensorBasisVector<E::ScalarField>,
    claimed: E::ScalarField,
    matrix_hiding: E::ScalarField,
    bit_width: usize,
    matrix_commitment: PairingOutput<E>,
    tier_one: &[E::G1],
    commitment_profile: BvaStreamingCommitProfile,
) -> Result<(PackedI8ParCommProof<E>, PackedI8ParCommProfile), PackedI8ParCommProverError<S::Error>>
where
    E: Pairing,
    E::ScalarField: Absorb,
    S: BvaI8Source,
{
    let binding =
        packed_i8_parcomm_statement_binding::<E>(&matrix_commitment, tbv, claimed, bit_width);
    let (packed_eo, packed_eo_padding, packed_eo_profile) =
        PackedEoLiteBulletProof::prove_i8_streaming_bound(source, tbv, claimed, bit_width, binding)
            .map_err(PackedI8ParCommProverError::PackedEo)?;
    let verification = packed_eo.verify_with_tbv_bound(tbv, claimed, binding);
    if !verification.accepted {
        return Err(PackedI8ParCommProverError::PackedEoRejected);
    }
    let (xl, xr) =
        bva_eo_opening_point(source.layout().logical_len(), &verification.challenges_inv).map_err(
            |error| PackedI8ParCommProverError::VirtualPcs(StreamingVirtualPcsError::Pcs(error)),
        )?;
    let value_commitment = E::pairing(pp.g_0.mul(packed_eo.a_reduced), pp.h_hat);
    let (opening, opening_profile) = open_bva_i8_streaming::<E, S>(
        pp,
        source,
        &xl,
        &xr,
        value_commitment,
        matrix_commitment,
        tier_one,
        E::ScalarField::zero(),
        matrix_hiding,
    )
    .map_err(PackedI8ParCommProverError::VirtualPcs)?;

    Ok((
        PackedI8ParCommProof {
            matrix_commitment,
            value_commitment,
            packed_eo,
            opening,
        },
        PackedI8ParCommProfile {
            commitment: commitment_profile,
            packed_eo: packed_eo_profile,
            opening: opening_profile,
            packed_eo_padding,
        },
    ))
}

/// Reduces verification to SmartPC's sparse deferred pairing equation.
pub fn verify_packed_i8_parcomm_as_subprotocol<E>(
    pp: &UniversalParams<E>,
    tbv: &TensorBasisVector<E::ScalarField>,
    claimed: E::ScalarField,
    bit_width: usize,
    proof: &PackedI8ParCommProof<E>,
) -> Result<SmartPcVerificationEquation<E>, PackedI8ParCommVerificationError>
where
    E: Pairing,
    E::ScalarField: Absorb,
{
    if bit_width == 0 || bit_width > 8 {
        return Err(PackedI8ParCommVerificationError::VirtualPcs(
            VirtualPcsError::InvalidBitWidth(bit_width),
        ));
    }
    let expected_stride =
        packing_stride_for_bit_width::<E::ScalarField>(tbv.layout().logical_len(), bit_width)
            .map_err(|_| {
                PackedI8ParCommVerificationError::VirtualPcs(
                    VirtualPcsError::InvalidBvaLogicalEntries(tbv.layout().logical_len()),
                )
            })?;
    if proof.packed_eo.packing_stride != expected_stride {
        return Err(PackedI8ParCommVerificationError::PackingStrideMismatch {
            expected: expected_stride,
            actual: proof.packed_eo.packing_stride,
        });
    }
    let binding =
        packed_i8_parcomm_statement_binding::<E>(&proof.matrix_commitment, tbv, claimed, bit_width);
    let verification = proof.packed_eo.verify_with_tbv_bound(tbv, claimed, binding);
    if !verification.accepted {
        return Err(PackedI8ParCommVerificationError::PackedEoRejected);
    }
    let expected_value_commitment = E::pairing(pp.g_0.mul(proof.packed_eo.a_reduced), pp.h_hat);
    if proof.value_commitment != expected_value_commitment {
        return Err(PackedI8ParCommVerificationError::ValueCommitmentMismatch);
    }
    let (xl, xr) = bva_eo_opening_point(tbv.layout().logical_len(), &verification.challenges_inv)
        .map_err(PackedI8ParCommVerificationError::VirtualPcs)?;
    SmartPC::<E>::verify_as_subprotocol(
        pp,
        proof.matrix_commitment,
        proof.value_commitment,
        &xl,
        &xr,
        &proof.opening,
    )
    .map_err(PackedI8ParCommVerificationError::SmartPc)
}

/// Evaluates the deferred sparse verifier equation.
pub fn verify_packed_i8_parcomm<E>(
    pp: &UniversalParams<E>,
    tbv: &TensorBasisVector<E::ScalarField>,
    claimed: E::ScalarField,
    bit_width: usize,
    proof: &PackedI8ParCommProof<E>,
) -> Result<bool, PackedI8ParCommVerificationError>
where
    E: Pairing,
    E::ScalarField: Absorb,
{
    Ok(verify_packed_i8_parcomm_as_subprotocol(pp, tbv, claimed, bit_width, proof)?.evaluate())
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use ark_bls12_381::{Bls12_381, Fr};
    use ark_ff::{One, PrimeField, Zero};
    use rayon::ThreadPoolBuilder;

    use super::*;
    use crate::bucketed_ip::BvaLayout;

    fn i8_to_field<F: PrimeField>(value: i8) -> F {
        if value >= 0 {
            F::from(value as u64)
        } else {
            -F::from((-(value as i16)) as u64)
        }
    }

    struct TestI8Source {
        layout: BvaLayout,
        vectors: Vec<Vec<i8>>,
    }

    impl BvaI8Source for TestI8Source {
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
            for (vector_index, values) in self.vectors.iter().enumerate() {
                visit(vector_index, values);
            }
            Ok(())
        }
    }

    fn explicit_claim(source: &TestI8Source, tbv: &TensorBasisVector<Fr>) -> Fr {
        source
            .vectors
            .iter()
            .enumerate()
            .flat_map(|(vector_index, values)| {
                values.iter().enumerate().map(move |(entry_index, &value)| {
                    i8_to_field::<Fr>(value)
                        * tbv
                            .physical_coefficient_at(vector_index, entry_index)
                            .unwrap()
                })
            })
            .sum()
    }

    #[test]
    fn per_bucket_parallel_projections_match_dense_reference() {
        let lengths = [9_usize, 5, 3, 2];
        let layout = BvaLayout::from_lengths(&lengths).unwrap();
        let vectors = lengths
            .iter()
            .enumerate()
            .map(|(vector_index, &len)| {
                (0..len)
                    .map(|entry_index| (((vector_index * 5 + entry_index * 3) % 11) as i8) - 5)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let left_challenges = vec![Fr::from(2_u64), Fr::from(3_u64), Fr::from(5_u64)];
        let right_challenges = vec![Fr::from(7_u64), Fr::from(11_u64)];
        let one_thread = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let four_threads = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let make_source = || TestI8Source {
            layout: layout.clone(),
            vectors: vectors.clone(),
        };

        let (left_one, left_one_profile) = one_thread.install(|| {
            project_packed_bva_i8_streaming::<Bls12_381, _>(
                &mut make_source(),
                &left_challenges,
                ProjectionSide::Left,
            )
            .unwrap()
        });
        let (left_four, left_four_profile) = four_threads.install(|| {
            project_packed_bva_i8_streaming::<Bls12_381, _>(
                &mut make_source(),
                &left_challenges,
                ProjectionSide::Left,
            )
            .unwrap()
        });
        let (right_one, right_one_profile) = one_thread.install(|| {
            project_packed_bva_i8_streaming::<Bls12_381, _>(
                &mut make_source(),
                &right_challenges,
                ProjectionSide::Right,
            )
            .unwrap()
        });
        let (right_four, right_four_profile) = four_threads.install(|| {
            project_packed_bva_i8_streaming::<Bls12_381, _>(
                &mut make_source(),
                &right_challenges,
                ProjectionSide::Right,
            )
            .unwrap()
        });
        assert_eq!(left_one, left_four);
        assert_eq!(right_one, right_four);
        assert_eq!(left_one_profile, left_four_profile);
        assert_eq!(right_one_profile, right_four_profile);

        let (packed_rows, packed_cols) = packed_bva_shape(layout.logical_len()).unwrap();
        let mut dense = vec![Fr::zero(); layout.logical_len()];
        let offsets = layout.bucket_offsets();
        for (vector_index, values) in vectors.iter().enumerate() {
            let offset = offsets[layout.vector_to_bucket()[vector_index]];
            for (entry_index, &value) in values.iter().enumerate() {
                dense[offset + entry_index] = i8_to_field(value);
            }
        }
        let left_weights =
            poly_commit::smart_pc::xi::xi_from_challenges::<Bls12_381>(&left_challenges);
        let right_weights =
            poly_commit::smart_pc::xi::xi_from_challenges::<Bls12_381>(&right_challenges);
        let expected_left = (0..packed_cols)
            .map(|packed_col| {
                (0..packed_rows).fold(Fr::zero(), |sum, packed_row| {
                    sum + dense[packed_row * packed_cols + packed_col] * left_weights[packed_row]
                })
            })
            .collect::<Vec<_>>();
        let expected_right = (0..packed_rows)
            .map(|packed_row| {
                (0..packed_cols).fold(Fr::zero(), |sum, packed_col| {
                    sum + dense[packed_row * packed_cols + packed_col] * right_weights[packed_col]
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(left_one, expected_left);
        assert_eq!(right_one, expected_right);

        let mut rng = ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup(3, &mut rng).unwrap();
        let hiding = Fr::from(13_u64);
        let (commitment_one, tier_one_one, profile_one) = one_thread.install(|| {
            commit_bva_i8_streaming::<Bls12_381, _>(&pp, &mut make_source(), hiding, 8).unwrap()
        });
        let (commitment_four, tier_one_four, profile_four) = four_threads.install(|| {
            commit_bva_i8_streaming::<Bls12_381, _>(&pp, &mut make_source(), hiding, 8).unwrap()
        });
        assert_eq!(commitment_one, commitment_four);
        assert_eq!(tier_one_one, tier_one_four);
        assert_eq!(profile_one, profile_four);
        assert!(profile_four.max_live_group_contributions <= 1 << 8);
        assert_eq!(profile_four.prepared_base_entries, packed_rows);
    }

    /// Lays a bucketed source out as the dense row-major matrix that an
    /// accelerated prover indexes, with bucket gaps left as logical zeros.
    fn dense_row_major(source: &TestI8Source) -> (Vec<i8>, usize, usize) {
        let layout = source.layout();
        let (rows, cols) = packed_bva_shape(layout.logical_len()).unwrap();
        let offsets = layout.bucket_offsets();
        let mut dense = vec![0_i8; layout.logical_len()];
        for (vector_index, values) in source.vectors.iter().enumerate() {
            let start = offsets[layout.vector_to_bucket()[vector_index]];
            dense[start..start + values.len()].copy_from_slice(values);
        }
        (dense, rows, cols)
    }

    /// Pins down the packing convention an off-CPU prover has to reproduce.
    ///
    /// The streaming commitment and both projections walk buckets rather than a
    /// materialized matrix, so this checks that the dense row-major view of the
    /// same source yields identical tier-one columns and identical projections.
    #[test]
    fn dense_row_major_view_matches_streaming_commit_and_projections() {
        let lengths = [9_usize, 5, 3, 2];
        let layout = BvaLayout::from_lengths(&lengths).unwrap();
        let vectors = lengths
            .iter()
            .enumerate()
            .map(|(vector_index, &len)| {
                (0..len)
                    .map(|entry_index| (((vector_index * 11 + entry_index * 7) % 15) as i8) - 7)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut source = TestI8Source {
            layout: layout.clone(),
            vectors,
        };
        let (dense, rows, cols) = dense_row_major(&source);
        assert_eq!(rows * cols, layout.logical_len());

        let mut rng = ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup(3, &mut rng).unwrap();
        let hiding = Fr::from(29_u64);
        let (commitment, tier_one, _) =
            commit_bva_i8_streaming::<Bls12_381, _>(&pp, &mut source, hiding, 8).unwrap();

        // `tier_one[col] = sum_row dense[row][col] * G[row]`.
        let dense_tier_one = (0..cols)
            .map(|col| {
                (0..rows).fold(<Bls12_381 as Pairing>::G1::zero(), |accumulator, row| {
                    let value = i64::from(dense[row * cols + col]);
                    accumulator + pp.vec_g[row] * Fr::from(value)
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(dense_tier_one, tier_one);
        assert_eq!(
            pcutils::inner_pairing_product::<Bls12_381>(&dense_tier_one, &pp.vec_h[..cols])
                + pp.tilde_u.mul(hiding),
            commitment
        );

        let left_challenges = vec![Fr::from(2_u64), Fr::from(3_u64), Fr::from(5_u64)];
        let right_challenges = vec![Fr::from(7_u64), Fr::from(11_u64)];
        assert_eq!(1_usize << left_challenges.len(), rows);
        assert_eq!(1_usize << right_challenges.len(), cols);

        let (left, _) = project_packed_bva_i8_streaming::<Bls12_381, _>(
            &mut source,
            &left_challenges,
            ProjectionSide::Left,
        )
        .unwrap();
        let left_weights =
            poly_commit::smart_pc::xi::xi_from_challenges::<Bls12_381>(&left_challenges);
        let dense_left = (0..cols)
            .map(|col| {
                (0..rows).fold(Fr::zero(), |accumulator, row| {
                    accumulator + left_weights[row] * Fr::from(i64::from(dense[row * cols + col]))
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(dense_left, left);

        let (right, _) = project_packed_bva_i8_streaming::<Bls12_381, _>(
            &mut source,
            &right_challenges,
            ProjectionSide::Right,
        )
        .unwrap();
        let right_weights =
            poly_commit::smart_pc::xi::xi_from_challenges::<Bls12_381>(&right_challenges);
        let dense_right = (0..rows)
            .map(|row| {
                (0..cols).fold(Fr::zero(), |accumulator, col| {
                    accumulator + right_weights[col] * Fr::from(i64::from(dense[row * cols + col]))
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(dense_right, right);
    }

    /// Expands `weights[index] = prod_{bit set in index} challenges[bit]`.
    fn bitwise_tensor(challenges: &[Fr]) -> Vec<Fr> {
        let mut weights = vec![Fr::one()];
        for challenge in challenges {
            let half = weights.len();
            weights.extend_from_within(..);
            for slot in &mut weights[half..] {
                *slot *= challenge;
            }
        }
        weights
    }

    /// Shows the global tensor claim is an outer product over the packed shape.
    ///
    /// `global_tensor` gives every logical index the product of the challenges
    /// its set bits select, and a row-major packing splits those bits into the
    /// column half and the row half. The claim therefore equals
    /// `<L, right_projection(R)>`, which lets a prover reuse a projection it
    /// already computes instead of streaming the witness a second time.
    #[test]
    fn global_tensor_claim_factors_through_the_right_projection() {
        let lengths = [9_usize, 5, 3, 2];
        let layout = BvaLayout::from_lengths(&lengths).unwrap();
        let vectors = lengths
            .iter()
            .enumerate()
            .map(|(vector_index, &len)| {
                (0..len)
                    .map(|entry_index| (((vector_index * 11 + entry_index * 7) % 15) as i8) - 7)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut source = TestI8Source {
            layout: layout.clone(),
            vectors,
        };
        let (rows, cols) = packed_bva_shape(layout.logical_len()).unwrap();
        let log_cols = cols.ilog2() as usize;

        let challenges = (0..layout.logical_len().ilog2())
            .map(|index| Fr::from(u64::from(index) + 2))
            .collect::<Vec<_>>();
        let tbv = TensorBasisVector::global_tensor(&layout, &challenges).unwrap();
        let claimed = tbv.inner_product_i8_streaming(&mut source).unwrap();

        let column_weights = bitwise_tensor(&challenges[..log_cols]);
        let row_weights = bitwise_tensor(&challenges[log_cols..]);
        assert_eq!(column_weights.len(), cols);
        assert_eq!(row_weights.len(), rows);

        let (dense, _, _) = dense_row_major(&source);
        let row_sums = (0..rows).map(|row| {
            (0..cols).fold(Fr::zero(), |accumulator, col| {
                accumulator + column_weights[col] * Fr::from(i64::from(dense[row * cols + col]))
            })
        });
        let factored = row_weights
            .iter()
            .zip(row_sums)
            .map(|(weight, sum)| *weight * sum)
            .sum::<Fr>();
        assert_eq!(factored, claimed);
    }

    #[test]
    fn decreasing_plus_one_vectors_verify_end_to_end_and_reject_tampering() {
        let lengths = [9_usize, 5, 3, 2];
        let layout = BvaLayout::from_lengths(&lengths).unwrap();
        assert_eq!(layout.bucket_sizes(), &[16, 8, 4, 2]);
        assert_eq!(layout.logical_len(), 32);
        assert_eq!(layout.padding_bucket_count(), 0);

        let vectors = lengths
            .iter()
            .enumerate()
            .map(|(vector_index, &len)| {
                (0..len)
                    .map(|entry_index| (((vector_index * 11 + entry_index * 7) % 15) as i8) - 7)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        assert_eq!(std::mem::size_of_val(&vectors[0][0]), 1);
        let challenges = (0..layout.logical_len().ilog2())
            .map(|index| Fr::from(u64::from(index) + 2))
            .collect::<Vec<_>>();
        let tbv = TensorBasisVector::global_tensor(&layout, &challenges).unwrap();
        let mut source = TestI8Source { layout, vectors };
        let claimed = explicit_claim(&source, &tbv);

        let mut rng = ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup(3, &mut rng).unwrap();
        let cached = serialize_public_parameters(&pp).unwrap();
        let pp = deserialize_public_parameters_exact::<Bls12_381>(&cached).unwrap();
        let (proof, profile) = prove_packed_i8_parcomm::<Bls12_381, _>(
            &pp,
            &mut source,
            &tbv,
            claimed,
            Fr::from(29_u64),
            8,
        )
        .unwrap();

        assert_eq!(
            proof.packed_eo.packing_stride,
            packing_stride_for_bit_width::<Fr>(source.layout.logical_len(), 8).unwrap()
        );
        assert_eq!(
            profile.commitment.physical_entries,
            lengths.iter().sum::<usize>()
        );
        assert_eq!(profile.commitment.materialized_padding_entries, 0);
        assert!(profile.commitment.max_live_group_contributions <= 1 << 8);
        assert_eq!(profile.packed_eo_padding.materialized_padding_entries, 0);
        assert_eq!(profile.opening.materialized_projection_padding_entries, 0);
        let proof_bytes = proof.to_bytes().unwrap();
        let decoded = PackedI8ParCommProof::<Bls12_381>::from_bytes_exact(&proof_bytes).unwrap();
        assert_eq!(decoded, proof);
        assert!(verify_packed_i8_parcomm(&pp, &tbv, claimed, 8, &decoded).unwrap());
        let mut proof_with_trailing_byte = proof_bytes;
        proof_with_trailing_byte.push(0);
        assert!(matches!(
            PackedI8ParCommProof::<Bls12_381>::from_bytes_exact(&proof_with_trailing_byte),
            Err(PackedI8ParCommProofDecodeError::TrailingBytes)
        ));
        let proof = decoded;
        let deferred =
            verify_packed_i8_parcomm_as_subprotocol(&pp, &tbv, claimed, 8, &proof).unwrap();
        assert!(deferred.evaluate());
        assert!(verify_packed_i8_parcomm(&pp, &tbv, claimed, 8, &proof).unwrap());

        let binding = packed_i8_parcomm_statement_binding::<Bls12_381>(
            &proof.matrix_commitment,
            &tbv,
            claimed,
            8,
        );
        assert_ne!(
            binding,
            packed_i8_parcomm_statement_binding::<Bls12_381>(
                &proof.matrix_commitment,
                &tbv,
                claimed,
                7,
            )
        );
        let mut altered_challenges = challenges;
        altered_challenges[0] += Fr::one();
        let altered_tbv =
            TensorBasisVector::global_tensor(source.layout(), &altered_challenges).unwrap();
        assert_ne!(
            binding,
            packed_i8_parcomm_statement_binding::<Bls12_381>(
                &proof.matrix_commitment,
                &altered_tbv,
                claimed,
                8,
            )
        );
        assert!(!verify_packed_i8_parcomm(&pp, &tbv, claimed, 7, &proof).unwrap_or(false));

        let mut commitment_tampered = proof.clone();
        commitment_tampered.matrix_commitment += pp.u;
        assert!(
            !verify_packed_i8_parcomm(&pp, &tbv, claimed, 8, &commitment_tampered).unwrap_or(false)
        );

        let mut opening_tampered = proof;
        opening_tampered.opening.linear_z_a += Fr::one();
        assert!(
            !verify_packed_i8_parcomm(&pp, &tbv, claimed, 8, &opening_tampered).unwrap_or(false)
        );

        let mut trailing = cached;
        trailing.push(0);
        assert!(matches!(
            deserialize_public_parameters_exact::<Bls12_381>(&trailing),
            Err(PublicParametersDecodeError::TrailingBytes)
        ));
        assert!(!claimed.is_zero());
    }
}
