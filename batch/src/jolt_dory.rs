//! Commitment-bound packed EO-LiteBullet using Jolt's CPU Dory backend.
//!
//! ParComm keeps the witness and Packed EO arithmetic in arkworks BN254. The
//! adapter converts only the final opening point and value to Jolt's field
//! wrapper; the dense witness remains one signed byte per physical entry.

use std::{
    fmt::{Display, Formatter},
    sync::Arc,
};

use ark_bn254::Fr;
use ark_serialize::{CanonicalSerialize, SerializationError};
use bincode::Options;
use jolt_crypto::{Bn254G1, DeriveSetup, Pedersen, PedersenSetup, VectorCommitment};
use jolt_dory::{DoryCommitment, DoryHint, DoryProverSetup, DoryScheme, DoryVerifierSetup};
use jolt_field::Fr as JoltFr;
use jolt_openings::{HomomorphicBatch, ZkBatchOpeningScheme};
use jolt_poly::{MultilinearPoly, Point};
use jolt_transcript::domain::{Label, LabelWithCount, U64Word};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rayon::prelude::*;
use serde::{de::DeserializeOwned, Serialize};

use crate::{
    bucketed_ip::{
        packing_stride_for_bit_width, BvaI8Source, EoLiteBulletVerification,
        PackedEoLiteBulletProof, PaddingWorkCounters, StreamingEoError, StreamingEoProfile,
        TensorBasisVector,
    },
    fiat_shamir::FiatShamir,
    jolt_dory_adapter::{
        commit_dense_i8_zk, commit_field_vector_zk as commit_field_vector_zk_adapter,
        normalize_monomial_point, open_dense_i8_zk,
        open_field_vector_zk as open_field_vector_zk_adapter, verify_dense_i8_zk,
        verify_field_vector_zk as verify_field_vector_zk_adapter, DenseI8DoryError,
        DenseI8DoryProof, DenseI8DorySetup, DenseI8Prefix,
    },
};

pub use crate::jolt_dory_adapter::{FieldVectorDoryProof, HiddenFieldVectorDoryProof};

const PROOF_MAGIC: [u8; 8] = *b"JLDORY01";
const MAX_COMPONENT_BYTES: usize = 768 * 1024;

/// Prover parameters for Jolt's BN254 Dory PCS.
pub struct DoryProverParameters {
    pub(crate) pcs: DoryProverSetup,
    pub(crate) blindfold: PedersenSetup<Bn254G1>,
}

/// Verifier parameters and the public evaluation-commitment basis.
pub struct DoryVerifierParameters {
    pub(crate) pcs: DoryVerifierSetup,
    evaluation: PedersenSetup<Bn254G1>,
    pub(crate) blindfold: PedersenSetup<Bn254G1>,
}

/// Generate or load Jolt's canonical even-sized Dory URS.
pub fn setup_dory_parameters(
    max_num_vars: usize,
) -> (DoryProverParameters, DoryVerifierParameters) {
    let setup = DenseI8DorySetup::new(max_num_vars);
    let blindfold_capacity = 1_usize << (max_num_vars.next_multiple_of(2) / 2);
    let blindfold = PedersenSetup::<Bn254G1>::derive(&setup.prover, blindfold_capacity);
    (
        DoryProverParameters {
            pcs: setup.prover,
            blindfold: blindfold.clone(),
        },
        DoryVerifierParameters {
            pcs: setup.verifier,
            evaluation: setup.evaluation,
            blindfold,
        },
    )
}

/// Jolt canonicalizes odd variable counts to the next even URS size.
pub fn canonical_dory_log(max_num_vars: usize) -> usize {
    max_num_vars.next_multiple_of(2)
}

/// Standard-MLE claim after converting ParComm's `(1, x)` tensor basis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NormalizedDoryClaim {
    /// Jolt point order is most-significant variable first.
    pub point_msb: Vec<Fr>,
    pub value: Fr,
    pub scale: Fr,
}

/// Normalize a low-bit-first ParComm monomial claim for Jolt.
pub fn normalize_monomial_mle_claim(
    point_low: &[Fr],
    value: Fr,
) -> Result<NormalizedDoryClaim, DoryBackendError> {
    let jolt_point = point_low
        .iter()
        .copied()
        .map(JoltFr::from)
        .collect::<Vec<_>>();
    let normalization = normalize_monomial_point(&jolt_point)?;
    let scale = Fr::from(normalization.monomial_scale);
    Ok(NormalizedDoryClaim {
        point_msb: normalization
            .standard_point_msb
            .into_iter()
            .map(Fr::from)
            .collect(),
        value: value * Fr::from(normalization.monomial_scale_inverse),
        scale,
    })
}

/// Independent standard-MLE oracle using Jolt's MSB-first convention.
pub fn standard_mle_evaluate_i8(
    values: &[i8],
    logical_len: usize,
    point_msb: &[Fr],
) -> Result<Fr, DoryBackendError> {
    let source = DenseI8Prefix::new(values, logical_len)?;
    if point_msb.len() != source.num_vars() {
        return Err(DoryBackendError::PointShape {
            expected: source.num_vars(),
            actual: point_msb.len(),
        });
    }
    let point = point_msb
        .iter()
        .copied()
        .map(JoltFr::from)
        .collect::<Vec<_>>();
    Ok(Fr::from(source.evaluate(&point)))
}

/// Evaluate a dense i8 prefix in ParComm's low-bit-first `(1, x)` basis.
pub fn monomial_mle_evaluate_i8(
    values: &[i8],
    logical_len: usize,
    point_low: &[Fr],
) -> Result<Fr, DoryBackendError> {
    let source = DenseI8Prefix::new(values, logical_len)?;
    let point = point_low
        .iter()
        .copied()
        .map(JoltFr::from)
        .collect::<Vec<_>>();
    Ok(Fr::from(source.evaluate_monomial_low(&point)?))
}

/// Commitment counters for the streaming dense-i8 path.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DoryCommitProfile {
    pub physical_entries: usize,
    pub nonzero_entries: usize,
    pub logical_entries: usize,
    pub row_commitments: usize,
    pub physical_rows: usize,
    pub materialized_field_entries: usize,
    pub materialized_padding_entries: usize,
    pub padding_group_operations: usize,
}

/// Opening counters for the lazy dense-i8 polynomial.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DoryOpenProfile {
    pub source_passes: usize,
    pub physical_entries: usize,
    pub nonzero_entries: usize,
    pub materialized_projection_entries: usize,
    pub materialized_padding_entries: usize,
    pub padding_field_operations: usize,
}

/// Profiles for commitment, Packed EO, and Dory opening.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PackedI8ParCommProfile {
    pub commitment: DoryCommitProfile,
    pub packed_eo: StreamingEoProfile,
    pub opening: DoryOpenProfile,
    pub packed_eo_padding: PaddingWorkCounters,
}

/// Complete commitment-bound Packed EO and Dory proof.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PackedI8ParCommProof {
    pub matrix_commitment: DoryCommitment,
    pub packed_eo: PackedEoLiteBulletProof<Fr>,
    pub opening: DenseI8DoryProof,
}

impl PackedI8ParCommProof {
    pub fn to_bytes(&self) -> Result<Vec<u8>, PackedI8ParCommProofDecodeError> {
        let commitment = serialize_serde(&self.matrix_commitment)?;
        let packed_eo = self.packed_eo.to_bytes()?;
        let opening = serialize_serde(&self.opening)?;
        let mut bytes = Vec::with_capacity(
            PROOF_MAGIC.len()
                + 3 * std::mem::size_of::<u64>()
                + commitment.len()
                + packed_eo.len()
                + opening.len(),
        );
        bytes.extend_from_slice(&PROOF_MAGIC);
        append_component(&mut bytes, &commitment);
        append_component(&mut bytes, &packed_eo);
        append_component(&mut bytes, &opening);
        Ok(bytes)
    }

    pub fn from_bytes_exact(bytes: &[u8]) -> Result<Self, PackedI8ParCommProofDecodeError> {
        let mut cursor = 0_usize;
        if take_bytes(bytes, &mut cursor, PROOF_MAGIC.len())? != PROOF_MAGIC {
            return Err(PackedI8ParCommProofDecodeError::InvalidMagic);
        }
        let commitment = take_component(bytes, &mut cursor)?;
        let packed_eo = take_component(bytes, &mut cursor)?;
        let opening = take_component(bytes, &mut cursor)?;
        if cursor != bytes.len() {
            return Err(PackedI8ParCommProofDecodeError::TrailingBytes);
        }
        if commitment.len() > MAX_COMPONENT_BYTES
            || packed_eo.len() > MAX_COMPONENT_BYTES
            || opening.len() > MAX_COMPONENT_BYTES
        {
            return Err(PackedI8ParCommProofDecodeError::ComponentTooLarge);
        }
        Ok(Self {
            matrix_commitment: deserialize_serde_exact(commitment)?,
            packed_eo: PackedEoLiteBulletProof::from_bytes_exact(packed_eo)
                .map_err(|_| PackedI8ParCommProofDecodeError::InvalidPackedEo)?,
            opening: deserialize_serde_exact(opening)?,
        })
    }
}

/// Exact proof decoding failures.
#[derive(Debug)]
pub enum PackedI8ParCommProofDecodeError {
    Serialization(SerializationError),
    Bincode(String),
    InvalidMagic,
    InvalidLength,
    InvalidPackedEo,
    ComponentTooLarge,
    TrailingBytes,
}

impl Display for PackedI8ParCommProofDecodeError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for PackedI8ParCommProofDecodeError {}

impl From<SerializationError> for PackedI8ParCommProofDecodeError {
    fn from(error: SerializationError) -> Self {
        Self::Serialization(error)
    }
}

/// Errors at the Jolt-Dory PCS boundary.
#[derive(Debug)]
pub enum DoryBackendError {
    InvalidBitWidth(usize),
    InvalidLogicalEntries(usize),
    DensePrefixLength { expected: usize, actual: usize },
    FieldVectorLength { expected: usize, actual: usize },
    ValueOutOfRange { value: i8, bit_width: usize },
    PointShape { expected: usize, actual: usize },
    PackingStrideMismatch { expected: usize, actual: usize },
    PackedEoRejected,
    Dory(DenseI8DoryError),
}

impl Display for DoryBackendError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Dory(error) => Display::fmt(error, formatter),
            _ => write!(formatter, "{self:?}"),
        }
    }
}

impl std::error::Error for DoryBackendError {}

impl From<DenseI8DoryError> for DoryBackendError {
    fn from(error: DenseI8DoryError) -> Self {
        Self::Dory(error)
    }
}

/// Complete-prover failures preserving source errors from Packed EO.
#[derive(Debug)]
pub enum PackedI8ParCommProverError<SourceError> {
    Backend(DoryBackendError),
    PackedEo(StreamingEoError<SourceError>),
}

impl<SourceError: Display> Display for PackedI8ParCommProverError<SourceError> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Backend(error) => Display::fmt(error, formatter),
            Self::PackedEo(error) => Display::fmt(error, formatter),
        }
    }
}

impl<SourceError> std::error::Error for PackedI8ParCommProverError<SourceError> where
    SourceError: std::error::Error + 'static
{
}

pub type PackedI8ParCommVerificationError = DoryBackendError;

/// Terminal verifier result; Dory has already executed its pairings.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TerminalDoryVerification {
    accepted: bool,
}

impl TerminalDoryVerification {
    pub fn evaluate(&self) -> bool {
        self.accepted
    }
}

/// Dense logical-prefix source required by the Jolt streaming adapter.
pub trait DenseI8BvaSource: BvaI8Source {
    fn dense_values(&self) -> &[i8];
}

/// Prover-only commitment state retained until Dory opening.
#[derive(Clone)]
pub struct CommittedDoryBva {
    pub commitment: DoryCommitment,
    pub(crate) hint: DoryHint,
    pub profile: DoryCommitProfile,
}

/// Prover-only state for a committed dense field evaluation table.
#[derive(Clone)]
pub struct CommittedDoryFieldVector {
    pub commitment: DoryCommitment,
    pub logical_len: usize,
    pub(crate) hint: Arc<DoryHint>,
    pub(crate) matrix: Option<Arc<CommittedDoryFieldMatrix>>,
    pub(crate) matrix_column: usize,
    /// Shared Section 6 grid arity, or `None` for the legacy native layout.
    pub(crate) grid_num_vars: Option<usize>,
}

/// One materialized column-major Section 6 matrix behind a single Dory
/// commitment. Column views never create additional Dory commitments.
#[derive(Clone)]
pub(crate) struct CommittedDoryFieldMatrix {
    pub values: Vec<Fr>,
    pub padded_rows: usize,
    pub logical_cols: usize,
    pub padded_cols: usize,
    pub commitment: DoryCommitment,
    pub hint: Arc<DoryHint>,
    pub grid_num_vars: Option<usize>,
}

/// Commit a power-of-two arkworks BN254 field vector with hiding Dory.
pub fn commit_field_vector_zk(
    parameters: &DoryProverParameters,
    values: &[Fr],
) -> Result<CommittedDoryFieldVector, DoryBackendError> {
    let values_jolt = values.iter().copied().map(JoltFr::from).collect::<Vec<_>>();
    let (commitment, hint) = commit_field_vector_zk_adapter(&values_jolt, &parameters.pcs)?;
    Ok(CommittedDoryFieldVector {
        commitment,
        logical_len: values.len(),
        hint: Arc::new(hint),
        matrix: None,
        matrix_column: 0,
        grid_num_vars: None,
    })
}

/// Materialize and commit one complete column-major matrix, returning cheap
/// column views that all reference the same commitment and opening hint.
pub fn commit_field_matrix_columns_zk(
    parameters: &DoryProverParameters,
    columns: &[Vec<Fr>],
    padded_cols: usize,
) -> Result<Vec<CommittedDoryFieldVector>, DoryBackendError> {
    let columns = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
    commit_field_matrix_column_slices_zk(parameters, &columns, padded_cols)
}

/// Commit one complete column-major matrix from borrowed column slices.
pub fn commit_field_matrix_column_slices_zk(
    parameters: &DoryProverParameters,
    columns: &[&[Fr]],
    padded_cols: usize,
) -> Result<Vec<CommittedDoryFieldVector>, DoryBackendError> {
    if columns.is_empty()
        || padded_cols < columns.len()
        || !padded_cols.is_power_of_two()
        || columns[0].is_empty()
        || !columns[0].len().is_power_of_two()
    {
        return Err(DoryBackendError::InvalidLogicalEntries(columns.len()));
    }
    let padded_rows = columns[0].len();
    if columns.iter().any(|column| column.len() != padded_rows) {
        return Err(DoryBackendError::FieldVectorLength {
            expected: padded_rows,
            actual: columns
                .iter()
                .map(|column| column.len())
                .find(|length| *length != padded_rows)
                .unwrap_or(0),
        });
    }
    let matrix_len = padded_rows
        .checked_mul(padded_cols)
        .ok_or(DoryBackendError::InvalidLogicalEntries(usize::MAX))?;
    let mut values = vec![Fr::from(0_u64); matrix_len];
    for (column, source) in columns.iter().enumerate() {
        let start = column * padded_rows;
        values[start..start + padded_rows].copy_from_slice(source);
    }
    let values_jolt = values.iter().copied().map(JoltFr::from).collect::<Vec<_>>();
    let (commitment, hint) = commit_field_vector_zk_adapter(&values_jolt, &parameters.pcs)?;
    let hint = Arc::new(hint);
    let matrix = Arc::new(CommittedDoryFieldMatrix {
        values,
        padded_rows,
        logical_cols: columns.len(),
        padded_cols,
        commitment: commitment.clone(),
        hint: hint.clone(),
        grid_num_vars: None,
    });
    Ok((0..columns.len())
        .map(|matrix_column| CommittedDoryFieldVector {
            commitment: commitment.clone(),
            logical_len: padded_rows,
            hint: hint.clone(),
            matrix: Some(matrix.clone()),
            matrix_column,
            grid_num_vars: None,
        })
        .collect())
}

/// Evaluate a field vector at a Jolt-standard, MSB-first MLE point.
pub fn evaluate_field_vector_standard(
    values: &[Fr],
    point_msb: &[Fr],
) -> Result<Fr, DoryBackendError> {
    if values.is_empty() || !values.len().is_power_of_two() {
        return Err(DoryBackendError::InvalidLogicalEntries(values.len()));
    }
    let num_vars = values.len().trailing_zeros() as usize;
    if point_msb.len() != num_vars {
        return Err(DoryBackendError::PointShape {
            expected: num_vars,
            actual: point_msb.len(),
        });
    }
    let values_jolt = values.iter().copied().map(JoltFr::from).collect::<Vec<_>>();
    let point_jolt = point_msb
        .iter()
        .copied()
        .map(JoltFr::from)
        .collect::<Vec<_>>();
    Ok(Fr::from(values_jolt.evaluate(&point_jolt)))
}

/// Open a committed field vector and return its public evaluation.
#[allow(clippy::too_many_arguments)]
pub fn open_field_vector_zk(
    parameters: &DoryProverParameters,
    values: &[Fr],
    committed: &CommittedDoryFieldVector,
    point_msb: &[Fr],
    role: &[u8],
    outer_binding: &[u8],
) -> Result<(FieldVectorDoryProof, Fr), DoryBackendError> {
    if values.len() != committed.logical_len {
        return Err(DoryBackendError::FieldVectorLength {
            expected: committed.logical_len,
            actual: values.len(),
        });
    }
    let evaluation = evaluate_field_vector_standard(values, point_msb)?;
    let values_jolt = values.iter().copied().map(JoltFr::from).collect::<Vec<_>>();
    let point_jolt = point_msb
        .iter()
        .copied()
        .map(JoltFr::from)
        .collect::<Vec<_>>();
    let mut transcript = Blake2bTranscript::new(b"parcomm-field-vector-dory-v1");
    let proof = open_field_vector_zk_adapter(
        &values_jolt,
        &committed.commitment,
        committed.hint.as_ref().clone(),
        &point_jolt,
        JoltFr::from(evaluation),
        role,
        outer_binding,
        &parameters.pcs,
        &mut transcript,
    )?;
    Ok((proof, evaluation))
}

/// Verify a field-vector opening and bind it to the expected public value.
#[allow(clippy::too_many_arguments)]
pub fn verify_field_vector_zk(
    parameters: &DoryVerifierParameters,
    commitment: &DoryCommitment,
    logical_len: usize,
    point_msb: &[Fr],
    expected_eval: Fr,
    proof: &FieldVectorDoryProof,
    role: &[u8],
    outer_binding: &[u8],
) -> Result<(), DoryBackendError> {
    let point_jolt = point_msb
        .iter()
        .copied()
        .map(JoltFr::from)
        .collect::<Vec<_>>();
    let mut transcript = Blake2bTranscript::new(b"parcomm-field-vector-dory-v1");
    verify_field_vector_zk_adapter(
        commitment,
        proof,
        logical_len,
        &point_jolt,
        JoltFr::from(expected_eval),
        role,
        outer_binding,
        &parameters.pcs,
        &parameters.evaluation,
        &mut transcript,
    )?;
    Ok(())
}

/// Open several whole field matrices at one common MLE point with Jolt's
/// native homomorphic Dory batch opening.
#[allow(clippy::too_many_arguments)]
pub fn open_field_vectors_batch_zk(
    parameters: &DoryProverParameters,
    values: &[&[Fr]],
    committed: &[&CommittedDoryFieldVector],
    point_msb: &[Fr],
    role: &[u8],
    outer_binding: &[u8],
) -> Result<(FieldVectorDoryProof, Vec<Fr>), DoryBackendError> {
    validate_field_vector_batch(values, committed, point_msb)?;
    let evaluations = values
        .iter()
        .map(|table| evaluate_field_vector_standard(table, point_msb))
        .collect::<Result<Vec<_>, _>>()?;
    let values_jolt = values
        .iter()
        .map(|table| {
            table
                .par_iter()
                .copied()
                .map(JoltFr::from)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let polynomials = values_jolt
        .iter()
        .map(|table| table as &dyn MultilinearPoly<JoltFr>)
        .collect::<Vec<_>>();
    let commitments = committed
        .iter()
        .map(|table| table.commitment.clone())
        .collect::<Vec<_>>();
    let hints = committed
        .iter()
        .map(|table| table.hint.as_ref().clone())
        .collect::<Vec<_>>();
    let point = point_msb
        .iter()
        .copied()
        .map(JoltFr::from)
        .collect::<Vec<_>>();
    let evaluations_jolt = evaluations
        .iter()
        .copied()
        .map(JoltFr::from)
        .collect::<Vec<_>>();
    let mut transcript = Blake2bTranscript::new(b"parcomm-matrix-batch-dory-v1");
    bind_field_vector_batch_statement(
        &mut transcript,
        role,
        outer_binding,
        committed[0].logical_len,
        &point,
        &commitments,
        &evaluations_jolt,
    );
    let opening = HomomorphicBatch::<DoryScheme>::prove_batch_zk(
        &parameters.pcs,
        Point::high_to_low(point),
        commitments,
        polynomials,
        hints,
        evaluations_jolt,
        &mut transcript,
    )
    .map_err(|error| DenseI8DoryError::Dory(error.to_string()))?;
    Ok((
        FieldVectorDoryProof {
            opening: opening.proof,
            evaluation_blind: opening.blind,
        },
        evaluations,
    ))
}

/// Verify one native Dory batch opening for several whole field matrices.
#[allow(clippy::too_many_arguments)]
pub fn verify_field_vectors_batch_zk(
    parameters: &DoryVerifierParameters,
    commitments: &[&DoryCommitment],
    logical_len: usize,
    point_msb: &[Fr],
    expected_evaluations: &[Fr],
    proof: &FieldVectorDoryProof,
    role: &[u8],
    outer_binding: &[u8],
) -> Result<(), DoryBackendError> {
    if commitments.is_empty() || commitments.len() != expected_evaluations.len() {
        return Err(DoryBackendError::InvalidLogicalEntries(commitments.len()));
    }
    if logical_len == 0 || !logical_len.is_power_of_two() {
        return Err(DoryBackendError::InvalidLogicalEntries(logical_len));
    }
    let expected_point_len = logical_len.trailing_zeros() as usize;
    if point_msb.len() != expected_point_len {
        return Err(DoryBackendError::PointShape {
            expected: expected_point_len,
            actual: point_msb.len(),
        });
    }
    let point = point_msb
        .iter()
        .copied()
        .map(JoltFr::from)
        .collect::<Vec<_>>();
    let commitments = commitments
        .iter()
        .map(|commitment| (*commitment).clone())
        .collect::<Vec<_>>();
    let evaluations = expected_evaluations
        .iter()
        .copied()
        .map(JoltFr::from)
        .collect::<Vec<_>>();

    let mut coefficient_transcript = Blake2bTranscript::new(b"parcomm-matrix-batch-dory-v1");
    bind_field_vector_batch_statement(
        &mut coefficient_transcript,
        role,
        outer_binding,
        logical_len,
        &point,
        &commitments,
        &evaluations,
    );
    let coefficients = coefficient_transcript.challenge_scalar_powers(commitments.len());
    let joint_evaluation = evaluations
        .iter()
        .zip(&coefficients)
        .fold(JoltFr::from(0_u64), |sum, (evaluation, coefficient)| {
            sum + *evaluation * *coefficient
        });

    let mut transcript = Blake2bTranscript::new(b"parcomm-matrix-batch-dory-v1");
    bind_field_vector_batch_statement(
        &mut transcript,
        role,
        outer_binding,
        logical_len,
        &point,
        &commitments,
        &evaluations,
    );
    let hiding_commitment = HomomorphicBatch::<DoryScheme>::verify_batch_zk(
        &parameters.pcs,
        Point::high_to_low(point),
        commitments,
        &proof.opening,
        &mut transcript,
    )
    .map_err(|error| DenseI8DoryError::Dory(error.to_string()))?;
    if !Pedersen::<Bn254G1>::verify(
        &parameters.evaluation,
        &hiding_commitment,
        &[joint_evaluation],
        &proof.evaluation_blind,
    ) {
        return Err(DenseI8DoryError::EvaluationCommitmentMismatch.into());
    }
    Ok(())
}

/// Batch-open hidden evaluations of several committed field tables at one
/// common point. The returned coefficients are the exact native Jolt RLC
/// coefficients used by Dory.
#[allow(clippy::too_many_arguments)]
pub fn open_hidden_field_vectors_batch_zk(
    parameters: &DoryProverParameters,
    values: &[&[Fr]],
    committed: &[&CommittedDoryFieldVector],
    point_msb: &[JoltFr],
    evaluations: &[JoltFr],
    evaluation_commitments: &[Bn254G1],
    role: &[u8],
    outer_binding: &[u8],
) -> Result<(HiddenFieldVectorDoryProof, Bn254G1, JoltFr, Vec<JoltFr>), DoryBackendError> {
    let point_ark = point_msb.iter().copied().map(Fr::from).collect::<Vec<_>>();
    validate_field_vector_batch(values, committed, &point_ark)?;
    if evaluations.len() != values.len() || evaluation_commitments.len() != values.len() {
        return Err(DoryBackendError::InvalidLogicalEntries(evaluations.len()));
    }
    let claimed_polynomials = values
        .iter()
        .zip(evaluations)
        .map(|(table, &evaluation)| ClaimedArkFieldPolynomial::new(table, point_msb, evaluation))
        .collect::<Vec<_>>();
    let polynomials = claimed_polynomials
        .iter()
        .map(|table| table as &dyn MultilinearPoly<JoltFr>)
        .collect::<Vec<_>>();
    let commitments = committed
        .iter()
        .map(|table| table.commitment.clone())
        .collect::<Vec<_>>();
    let hints = committed
        .iter()
        .map(|table| table.hint.as_ref().clone())
        .collect::<Vec<_>>();

    let coefficients = hidden_field_vector_batch_coefficients(
        role,
        outer_binding,
        committed[0].logical_len,
        point_msb,
        &commitments,
        evaluation_commitments,
    );
    let mut transcript = Blake2bTranscript::new(b"parcomm-hidden-batch-dory-v1");
    bind_hidden_field_vector_batch_statement(
        &mut transcript,
        role,
        outer_binding,
        committed[0].logical_len,
        point_msb,
        &commitments,
        evaluation_commitments,
    );
    let opening = HomomorphicBatch::<DoryScheme>::prove_batch_zk(
        &parameters.pcs,
        Point::high_to_low(point_msb.to_vec()),
        commitments,
        polynomials,
        hints,
        evaluations.to_vec(),
        &mut transcript,
    )
    .map_err(|error| DenseI8DoryError::Dory(error.to_string()))?;
    Ok((
        HiddenFieldVectorDoryProof {
            opening: opening.proof,
        },
        opening.hiding_commitment,
        opening.blind,
        coefficients,
    ))
}

#[cfg(test)]
struct ClaimedFieldPolynomial<'source, Source: MultilinearPoly<JoltFr> + ?Sized> {
    source: &'source Source,
    point: &'source [JoltFr],
    evaluation: JoltFr,
}

struct ClaimedArkFieldPolynomial<'source> {
    values: &'source [Fr],
    point: &'source [JoltFr],
    evaluation: JoltFr,
}

impl<'source> ClaimedArkFieldPolynomial<'source> {
    fn new(values: &'source [Fr], point: &'source [JoltFr], evaluation: JoltFr) -> Self {
        assert_eq!(
            point.len(),
            values.len().trailing_zeros() as usize,
            "claimed field polynomial point has the wrong length"
        );
        Self {
            values,
            point,
            evaluation,
        }
    }
}

impl MultilinearPoly<JoltFr> for ClaimedArkFieldPolynomial<'_> {
    fn num_vars(&self) -> usize {
        self.values.len().trailing_zeros() as usize
    }

    fn evaluate(&self, point: &[JoltFr]) -> JoltFr {
        assert_eq!(
            point, self.point,
            "Dory requested an unexpected opening point"
        );
        self.evaluation
    }

    fn for_each_row(&self, sigma: usize, visit: &mut dyn FnMut(usize, &[JoltFr])) {
        let row_width = 1_usize << sigma;
        for (row, values) in self.values.chunks(row_width).enumerate() {
            let converted = values.iter().copied().map(JoltFr::from).collect::<Vec<_>>();
            visit(row, &converted);
        }
    }

    fn fold_rows(&self, left: &[JoltFr], sigma: usize) -> Vec<JoltFr> {
        let row_width = 1_usize << sigma;
        assert_eq!(
            self.values.len(),
            left.len() * row_width,
            "left vector length must equal the field polynomial row count"
        );
        let row_count = left.len();
        let workers = rayon::current_num_threads().min(row_count);
        (0..workers)
            .into_par_iter()
            .map(|worker| {
                let start_row = worker * row_count / workers;
                let end_row = (worker + 1) * row_count / workers;
                let mut output = vec![JoltFr::from(0_u64); row_width];
                for row in start_row..end_row {
                    let weight = left[row];
                    let start = row * row_width;
                    output
                        .iter_mut()
                        .zip(&self.values[start..start + row_width])
                        .for_each(|(destination, &value)| {
                            *destination += weight * JoltFr::from(value);
                        });
                }
                output
            })
            .reduce(
                || vec![JoltFr::from(0_u64); row_width],
                |mut left, right| {
                    left.iter_mut()
                        .zip(right)
                        .for_each(|(destination, value)| *destination += value);
                    left
                },
            )
    }
}

#[cfg(test)]
impl<'source, Source: MultilinearPoly<JoltFr> + ?Sized> ClaimedFieldPolynomial<'source, Source> {
    fn new(source: &'source Source, point: &'source [JoltFr], evaluation: JoltFr) -> Self {
        Self {
            source,
            point,
            evaluation,
        }
    }
}

#[cfg(test)]
impl<Source: MultilinearPoly<JoltFr> + ?Sized> MultilinearPoly<JoltFr>
    for ClaimedFieldPolynomial<'_, Source>
{
    fn num_vars(&self) -> usize {
        self.source.num_vars()
    }

    fn evaluate(&self, point: &[JoltFr]) -> JoltFr {
        assert_eq!(
            point, self.point,
            "Dory requested an unexpected opening point"
        );
        self.evaluation
    }

    fn for_each_row(&self, sigma: usize, visit: &mut dyn FnMut(usize, &[JoltFr])) {
        self.source.for_each_row(sigma, visit);
    }

    fn fold_rows(&self, left: &[JoltFr], sigma: usize) -> Vec<JoltFr> {
        let Some(values) = self.source.dense_evaluations() else {
            return self.source.fold_rows(left, sigma);
        };
        let row_width = 1_usize << sigma;
        assert_eq!(
            values.len(),
            left.len() * row_width,
            "left vector length must equal the dense polynomial row count"
        );
        let row_count = left.len();
        let workers = rayon::current_num_threads().min(row_count);
        (0..workers)
            .into_par_iter()
            .map(|worker| {
                let start_row = worker * row_count / workers;
                let end_row = (worker + 1) * row_count / workers;
                let mut output = vec![JoltFr::from(0_u64); row_width];
                for row in start_row..end_row {
                    let weight = left[row];
                    let start = row * row_width;
                    output
                        .iter_mut()
                        .zip(&values[start..start + row_width])
                        .for_each(|(destination, &value)| {
                            *destination += weight * value;
                        });
                }
                output
            })
            .reduce(
                || vec![JoltFr::from(0_u64); row_width],
                |mut left, right| {
                    left.iter_mut()
                        .zip(right)
                        .for_each(|(destination, value)| *destination += value);
                    left
                },
            )
    }

    fn dense_evaluations(&self) -> Option<&[JoltFr]> {
        self.source.dense_evaluations()
    }
}

/// Verify one native hidden Dory batch opening and return its joint hidden
/// evaluation commitment and RLC coefficients.
#[allow(clippy::too_many_arguments)]
pub fn verify_hidden_field_vectors_batch_zk(
    parameters: &DoryVerifierParameters,
    commitments: &[&DoryCommitment],
    logical_len: usize,
    point_msb: &[JoltFr],
    evaluation_commitments: &[Bn254G1],
    proof: &HiddenFieldVectorDoryProof,
    role: &[u8],
    outer_binding: &[u8],
) -> Result<(Bn254G1, Vec<JoltFr>), DoryBackendError> {
    if commitments.is_empty()
        || commitments.len() != evaluation_commitments.len()
        || logical_len == 0
        || !logical_len.is_power_of_two()
    {
        return Err(DoryBackendError::InvalidLogicalEntries(logical_len));
    }
    if point_msb.len() != logical_len.trailing_zeros() as usize {
        return Err(DoryBackendError::PointShape {
            expected: logical_len.trailing_zeros() as usize,
            actual: point_msb.len(),
        });
    }
    let commitments = commitments
        .iter()
        .map(|commitment| (*commitment).clone())
        .collect::<Vec<_>>();
    let coefficients = hidden_field_vector_batch_coefficients(
        role,
        outer_binding,
        logical_len,
        point_msb,
        &commitments,
        evaluation_commitments,
    );
    let mut transcript = Blake2bTranscript::new(b"parcomm-hidden-batch-dory-v1");
    bind_hidden_field_vector_batch_statement(
        &mut transcript,
        role,
        outer_binding,
        logical_len,
        point_msb,
        &commitments,
        evaluation_commitments,
    );
    let hiding_commitment = HomomorphicBatch::<DoryScheme>::verify_batch_zk(
        &parameters.pcs,
        Point::high_to_low(point_msb.to_vec()),
        commitments,
        &proof.opening,
        &mut transcript,
    )
    .map_err(|error| DenseI8DoryError::Dory(error.to_string()))?;
    Ok((hiding_commitment, coefficients))
}

fn validate_field_vector_batch(
    values: &[&[Fr]],
    committed: &[&CommittedDoryFieldVector],
    point_msb: &[Fr],
) -> Result<(), DoryBackendError> {
    if values.is_empty() || values.len() != committed.len() {
        return Err(DoryBackendError::InvalidLogicalEntries(values.len()));
    }
    let logical_len = committed[0].logical_len;
    if logical_len == 0 || !logical_len.is_power_of_two() {
        return Err(DoryBackendError::InvalidLogicalEntries(logical_len));
    }
    if point_msb.len() != logical_len.trailing_zeros() as usize {
        return Err(DoryBackendError::PointShape {
            expected: logical_len.trailing_zeros() as usize,
            actual: point_msb.len(),
        });
    }
    for (table, commitment) in values.iter().zip(committed) {
        if commitment.logical_len != logical_len || table.len() != logical_len {
            return Err(DoryBackendError::FieldVectorLength {
                expected: logical_len,
                actual: table.len(),
            });
        }
    }
    Ok(())
}

fn bind_field_vector_batch_statement<T: Transcript<Challenge = JoltFr>>(
    transcript: &mut T,
    role: &[u8],
    outer_binding: &[u8],
    logical_len: usize,
    point: &[JoltFr],
    commitments: &[DoryCommitment],
    evaluations: &[JoltFr],
) {
    transcript.append(&Label(b"parcomm_matrix_batch_dory_v1"));
    transcript.append(&LabelWithCount(b"role", role.len() as u64));
    transcript.append_bytes(role);
    transcript.append(&LabelWithCount(
        b"outer_binding",
        outer_binding.len() as u64,
    ));
    transcript.append_bytes(outer_binding);
    transcript.append_labeled(b"logical_len", &U64Word(logical_len as u64));
    transcript.append_values(b"point", point);
    transcript.append_labeled(b"matrix_count", &U64Word(commitments.len() as u64));
    for (commitment, evaluation) in commitments.iter().zip(evaluations) {
        transcript.append_labeled(b"matrix_commitment", commitment);
        transcript.append_labeled(b"matrix_evaluation", evaluation);
    }
}

fn hidden_field_vector_batch_coefficients(
    role: &[u8],
    outer_binding: &[u8],
    logical_len: usize,
    point: &[JoltFr],
    commitments: &[DoryCommitment],
    evaluation_commitments: &[Bn254G1],
) -> Vec<JoltFr> {
    let mut transcript = Blake2bTranscript::new(b"parcomm-hidden-batch-dory-v1");
    bind_hidden_field_vector_batch_statement(
        &mut transcript,
        role,
        outer_binding,
        logical_len,
        point,
        commitments,
        evaluation_commitments,
    );
    transcript.challenge_scalar_powers(commitments.len())
}

fn bind_hidden_field_vector_batch_statement<T: Transcript<Challenge = JoltFr>>(
    transcript: &mut T,
    role: &[u8],
    outer_binding: &[u8],
    logical_len: usize,
    point: &[JoltFr],
    commitments: &[DoryCommitment],
    evaluation_commitments: &[Bn254G1],
) {
    transcript.append(&Label(b"parcomm_hidden_batch_dory_v1"));
    transcript.append(&LabelWithCount(b"role", role.len() as u64));
    transcript.append_bytes(role);
    transcript.append(&LabelWithCount(
        b"outer_binding",
        outer_binding.len() as u64,
    ));
    transcript.append_bytes(outer_binding);
    transcript.append_labeled(b"logical_len", &U64Word(logical_len as u64));
    transcript.append_values(b"point", point);
    transcript.append_labeled(b"matrix_count", &U64Word(commitments.len() as u64));
    for (commitment, evaluation_commitment) in commitments.iter().zip(evaluation_commitments) {
        transcript.append_labeled(b"matrix_commitment", commitment);
        transcript.append_labeled(b"evaluation_commitment", evaluation_commitment);
    }
}

/// Deterministic bytes used by the outer Poseidon transcript.
pub fn commitment_binding_bytes(commitment: &DoryCommitment) -> Vec<u8> {
    serialize_serde(commitment).expect("Dory commitment serialization cannot fail")
}

/// Bind the Dory commitment to the Packed EO statement.
pub fn packed_i8_parcomm_statement_binding(
    commitment: &DoryCommitment,
    tbv: &TensorBasisVector<Fr>,
    claimed: Fr,
    bit_width: usize,
) -> Fr {
    let mut transcript = FiatShamir::<Fr>::new();
    transcript.push(&0x5041_5243_444f_5259_u64);
    transcript.push(&(bit_width as u64));
    transcript.push_bytes(&commitment_binding_bytes(commitment));
    tbv.absorb_transcript(&mut transcript);
    transcript.push(&claimed);
    transcript.gen_challenge()
}

/// Commit the dense byte prefix without constructing a field evaluation table.
pub fn commit_bva_i8_streaming<S: DenseI8BvaSource>(
    parameters: &DoryProverParameters,
    source: &S,
    bit_width: usize,
) -> Result<CommittedDoryBva, DoryBackendError> {
    validate_bit_width(bit_width)?;
    let logical_len = source.layout().logical_len();
    let values = source.dense_values();
    if values.len() != source.layout().active_len() {
        return Err(DoryBackendError::DensePrefixLength {
            expected: source.layout().active_len(),
            actual: values.len(),
        });
    }
    validate_values(values, bit_width)?;
    let dense = DenseI8Prefix::new(values, logical_len)?;
    let (commitment, hint) = commit_dense_i8_zk(&dense, &parameters.pcs);
    let sigma = dense.num_vars().div_ceil(2);
    let row_width = 1_usize << sigma;
    Ok(CommittedDoryBva {
        commitment,
        hint,
        profile: DoryCommitProfile {
            physical_entries: values.len(),
            nonzero_entries: values.par_iter().filter(|&&value| value != 0).count(),
            logical_entries: logical_len,
            row_commitments: logical_len / row_width,
            physical_rows: values.len().div_ceil(row_width),
            materialized_field_entries: 0,
            materialized_padding_entries: 0,
            padding_group_operations: 0,
        },
    })
}

/// Commit, reduce with Packed EO, and open with Jolt Dory.
pub fn prove_packed_i8_parcomm<S: DenseI8BvaSource>(
    parameters: &DoryProverParameters,
    source: &mut S,
    tbv: &TensorBasisVector<Fr>,
    claimed: Fr,
    bit_width: usize,
) -> Result<(PackedI8ParCommProof, PackedI8ParCommProfile), PackedI8ParCommProverError<S::Error>> {
    let committed = commit_bva_i8_streaming(parameters, source, bit_width)
        .map_err(PackedI8ParCommProverError::Backend)?;
    prove_packed_i8_parcomm_precommitted(parameters, source, tbv, claimed, bit_width, committed)
}

/// Complete Packed EO and Dory opening from an existing commitment.
pub fn prove_packed_i8_parcomm_precommitted<S: DenseI8BvaSource>(
    parameters: &DoryProverParameters,
    source: &mut S,
    tbv: &TensorBasisVector<Fr>,
    claimed: Fr,
    bit_width: usize,
    committed: CommittedDoryBva,
) -> Result<(PackedI8ParCommProof, PackedI8ParCommProfile), PackedI8ParCommProverError<S::Error>> {
    let binding =
        packed_i8_parcomm_statement_binding(&committed.commitment, tbv, claimed, bit_width);
    let (packed_eo, packed_eo_padding, packed_eo_profile) =
        PackedEoLiteBulletProof::prove_i8_streaming_bound(source, tbv, claimed, bit_width, binding)
            .map_err(PackedI8ParCommProverError::PackedEo)?;
    let verification = packed_eo.verify_with_tbv_bound(tbv, claimed, binding);
    if !verification.accepted {
        return Err(PackedI8ParCommProverError::Backend(
            DoryBackendError::PackedEoRejected,
        ));
    }
    normalize_monomial_mle_claim(&verification.challenges_inv, packed_eo.a_reduced)
        .map_err(PackedI8ParCommProverError::Backend)?;
    let monomial_point = verification
        .challenges_inv
        .iter()
        .copied()
        .map(JoltFr::from)
        .collect::<Vec<_>>();
    let monomial_eval = JoltFr::from(packed_eo.a_reduced);
    let dense = DenseI8Prefix::new(source.dense_values(), source.layout().logical_len())
        .map_err(DoryBackendError::from)
        .map_err(PackedI8ParCommProverError::Backend)?;
    let outer_binding = field_bytes(&binding);
    let mut transcript = Blake2bTranscript::new(b"parcomm-packed-eo-dory-v1");
    let opening = open_dense_i8_zk(
        &dense,
        &committed.commitment,
        committed.hint,
        &monomial_point,
        monomial_eval,
        &outer_binding,
        &parameters.pcs,
        &mut transcript,
    )
    .map_err(DoryBackendError::from)
    .map_err(PackedI8ParCommProverError::Backend)?;
    let sigma = dense.num_vars().div_ceil(2);
    let opening_physical_entries = committed.profile.physical_entries;
    let opening_nonzero_entries = committed.profile.nonzero_entries;
    let profile = PackedI8ParCommProfile {
        commitment: committed.profile,
        packed_eo: packed_eo_profile,
        opening: DoryOpenProfile {
            source_passes: 1,
            physical_entries: opening_physical_entries,
            nonzero_entries: opening_nonzero_entries,
            materialized_projection_entries: 1_usize << sigma,
            materialized_padding_entries: 0,
            padding_field_operations: 0,
        },
        packed_eo_padding,
    };
    Ok((
        PackedI8ParCommProof {
            matrix_commitment: committed.commitment,
            packed_eo,
            opening,
        },
        profile,
    ))
}

/// Verify the Packed EO scalar reduction and the dense BVA Dory opening while
/// postponing the terminal TBV fold.
pub fn verify_packed_i8_bva_opening_as_subprotocol(
    parameters: &DoryVerifierParameters,
    tbv: &TensorBasisVector<Fr>,
    claimed: Fr,
    bit_width: usize,
    proof: &PackedI8ParCommProof,
) -> Result<EoLiteBulletVerification<Fr>, PackedI8ParCommVerificationError> {
    validate_bit_width(bit_width)?;
    let expected_stride = packing_stride_for_bit_width::<Fr>(tbv.layout().logical_len(), bit_width)
        .map_err(|_| DoryBackendError::InvalidLogicalEntries(tbv.layout().logical_len()))?;
    if proof.packed_eo.packing_stride != expected_stride {
        return Err(DoryBackendError::PackingStrideMismatch {
            expected: expected_stride,
            actual: proof.packed_eo.packing_stride,
        });
    }
    let binding =
        packed_i8_parcomm_statement_binding(&proof.matrix_commitment, tbv, claimed, bit_width);
    let verification =
        proof
            .packed_eo
            .verify_scalar_identity_bound(tbv.layout().logical_len(), claimed, binding);
    if !verification.accepted {
        return Err(DoryBackendError::PackedEoRejected);
    }
    normalize_monomial_mle_claim(&verification.challenges_inv, proof.packed_eo.a_reduced)?;
    let point = verification
        .challenges_inv
        .iter()
        .copied()
        .map(JoltFr::from)
        .collect::<Vec<_>>();
    let outer_binding = field_bytes(&binding);
    let mut transcript = Blake2bTranscript::new(b"parcomm-packed-eo-dory-v1");
    verify_dense_i8_zk(
        &proof.matrix_commitment,
        &proof.opening,
        tbv.layout().logical_len(),
        &point,
        JoltFr::from(proof.packed_eo.a_reduced),
        &outer_binding,
        &parameters.pcs,
        &parameters.evaluation,
        &mut transcript,
    )?;
    Ok(verification)
}

/// Verify the terminal TBV inner product from an already checked Packed EO
/// scalar transcript.
pub fn verify_packed_i8_tbv_inner_product(
    tbv: &TensorBasisVector<Fr>,
    proof: &PackedI8ParCommProof,
    verification: &EoLiteBulletVerification<Fr>,
) -> bool {
    verification.accepted
        && tbv
            .fold_to_scalar(&verification.challenges)
            .map(|expected| expected == proof.packed_eo.b_reduced)
            .unwrap_or(false)
}

/// Execute terminal Jolt Dory verification.
pub fn verify_packed_i8_parcomm_as_subprotocol(
    parameters: &DoryVerifierParameters,
    tbv: &TensorBasisVector<Fr>,
    claimed: Fr,
    bit_width: usize,
    proof: &PackedI8ParCommProof,
) -> Result<TerminalDoryVerification, PackedI8ParCommVerificationError> {
    let verification =
        verify_packed_i8_bva_opening_as_subprotocol(parameters, tbv, claimed, bit_width, proof)?;
    if !verify_packed_i8_tbv_inner_product(tbv, proof, &verification) {
        return Err(DoryBackendError::PackedEoRejected);
    }
    Ok(TerminalDoryVerification { accepted: true })
}

/// Verify the complete proof immediately.
pub fn verify_packed_i8_parcomm(
    parameters: &DoryVerifierParameters,
    tbv: &TensorBasisVector<Fr>,
    claimed: Fr,
    bit_width: usize,
    proof: &PackedI8ParCommProof,
) -> Result<bool, PackedI8ParCommVerificationError> {
    Ok(
        verify_packed_i8_parcomm_as_subprotocol(parameters, tbv, claimed, bit_width, proof)?
            .evaluate(),
    )
}

fn validate_bit_width(bit_width: usize) -> Result<(), DoryBackendError> {
    if bit_width == 0 || bit_width > 8 {
        return Err(DoryBackendError::InvalidBitWidth(bit_width));
    }
    Ok(())
}

fn validate_values(values: &[i8], bit_width: usize) -> Result<(), DoryBackendError> {
    if bit_width == 8 {
        return Ok(());
    }
    if let Some(&value) = values
        .par_iter()
        .find_any(|&&value| usize::from(value.unsigned_abs()) >= (1_usize << bit_width))
    {
        return Err(DoryBackendError::ValueOutOfRange { value, bit_width });
    }
    Ok(())
}

fn field_bytes(value: &Fr) -> Vec<u8> {
    let mut bytes = Vec::new();
    value
        .serialize_compressed(&mut bytes)
        .expect("field serialization into memory cannot fail");
    bytes
}

fn serialize_serde<T: Serialize>(value: &T) -> Result<Vec<u8>, PackedI8ParCommProofDecodeError> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .serialize(value)
        .map_err(|error| PackedI8ParCommProofDecodeError::Bincode(error.to_string()))
}

fn deserialize_serde_exact<T: DeserializeOwned>(
    bytes: &[u8],
) -> Result<T, PackedI8ParCommProofDecodeError> {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
        .deserialize(bytes)
        .map_err(|error| PackedI8ParCommProofDecodeError::Bincode(error.to_string()))
}

fn append_component(bytes: &mut Vec<u8>, component: &[u8]) {
    bytes.extend_from_slice(&(component.len() as u64).to_le_bytes());
    bytes.extend_from_slice(component);
}

fn take_component<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
) -> Result<&'a [u8], PackedI8ParCommProofDecodeError> {
    let length_bytes = take_bytes(bytes, cursor, 8)?;
    let mut encoded = [0_u8; 8];
    encoded.copy_from_slice(length_bytes);
    let length = usize::try_from(u64::from_le_bytes(encoded))
        .map_err(|_| PackedI8ParCommProofDecodeError::InvalidLength)?;
    take_bytes(bytes, cursor, length)
}

fn take_bytes<'a>(
    bytes: &'a [u8],
    cursor: &mut usize,
    count: usize,
) -> Result<&'a [u8], PackedI8ParCommProofDecodeError> {
    let end = cursor
        .checked_add(count)
        .ok_or(PackedI8ParCommProofDecodeError::InvalidLength)?;
    let slice = bytes
        .get(*cursor..end)
        .ok_or(PackedI8ParCommProofDecodeError::InvalidLength)?;
    *cursor = end;
    Ok(slice)
}

#[cfg(test)]
mod tests {
    use std::{
        convert::Infallible,
        sync::{
            atomic::{AtomicUsize, Ordering},
            Arc,
        },
    };

    use ark_bn254::Fr;
    use ark_ff::{One, Zero};
    use jolt_crypto::{Bn254G1, Pedersen, VectorCommitment};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr as JoltFr;
    use jolt_openings::ZkOpeningScheme;
    use jolt_poly::MultilinearPoly;
    use jolt_transcript::{Blake2bTranscript, Transcript};

    use crate::bucketed_ip::{BvaI8Source, BvaLayout, TensorBasisVector};

    use super::{
        commit_field_matrix_columns_zk, commit_field_vector_zk, evaluate_field_vector_standard,
        normalize_monomial_mle_claim, open_field_vector_zk, open_hidden_field_vectors_batch_zk,
        prove_packed_i8_parcomm, setup_dory_parameters, standard_mle_evaluate_i8,
        verify_field_vector_zk, verify_hidden_field_vectors_batch_zk, verify_packed_i8_parcomm,
        ClaimedArkFieldPolynomial, ClaimedFieldPolynomial, DenseI8BvaSource, PackedI8ParCommProof,
    };

    struct CountingPolynomial<'a> {
        values: &'a [JoltFr],
        evaluation_calls: &'a AtomicUsize,
        fold_rows_calls: &'a AtomicUsize,
        dense_access_calls: &'a AtomicUsize,
    }

    impl MultilinearPoly<JoltFr> for CountingPolynomial<'_> {
        fn num_vars(&self) -> usize {
            self.values.num_vars()
        }

        fn evaluate(&self, point: &[JoltFr]) -> JoltFr {
            self.evaluation_calls.fetch_add(1, Ordering::Relaxed);
            self.values.evaluate(point)
        }

        fn for_each_row(&self, sigma: usize, visit: &mut dyn FnMut(usize, &[JoltFr])) {
            self.values.for_each_row(sigma, visit);
        }

        fn fold_rows(&self, left: &[JoltFr], sigma: usize) -> Vec<JoltFr> {
            self.fold_rows_calls.fetch_add(1, Ordering::Relaxed);
            self.values.fold_rows(left, sigma)
        }

        fn dense_evaluations(&self) -> Option<&[JoltFr]> {
            self.dense_access_calls.fetch_add(1, Ordering::Relaxed);
            Some(self.values)
        }
    }

    struct ByteSource {
        layout: BvaLayout,
        vectors: Vec<Vec<i8>>,
        dense: Vec<i8>,
    }

    impl ByteSource {
        fn new(vectors: Vec<Vec<i8>>) -> Self {
            let layout =
                BvaLayout::from_lengths(&vectors.iter().map(Vec::len).collect::<Vec<_>>()).unwrap();
            let dense = vectors.iter().flatten().copied().collect();
            Self {
                layout,
                vectors,
                dense,
            }
        }
    }

    impl BvaI8Source for ByteSource {
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

    impl DenseI8BvaSource for ByteSource {
        fn dense_values(&self) -> &[i8] {
            &self.dense
        }
    }

    #[test]
    fn claimed_field_polynomial_skips_source_evaluation() {
        let values = [
            Fr::from(1_u64),
            Fr::from(2_u64),
            Fr::from(3_u64),
            Fr::from(4_u64),
        ];
        let point = [JoltFr::from(5_u64), JoltFr::from(7_u64)];
        let values_jolt = values.iter().copied().map(JoltFr::from).collect::<Vec<_>>();
        let expected = values_jolt.evaluate(&point);
        let evaluation_calls = AtomicUsize::new(0);
        let fold_rows_calls = AtomicUsize::new(0);
        let dense_access_calls = AtomicUsize::new(0);
        let source = CountingPolynomial {
            values: &values_jolt,
            evaluation_calls: &evaluation_calls,
            fold_rows_calls: &fold_rows_calls,
            dense_access_calls: &dense_access_calls,
        };
        let claimed = ClaimedFieldPolynomial::new(&source, &point, expected);
        let (prover, verifier) = setup_dory_parameters(2);
        let committed = commit_field_vector_zk(&prover, &values).unwrap();
        let mut prover_transcript = Blake2bTranscript::<JoltFr>::new(b"claimed-field-poly");
        let (proof, _, _) = DoryScheme::open_zk(
            &claimed,
            &point,
            expected,
            &prover.pcs,
            committed.hint.as_ref().clone(),
            &mut prover_transcript,
        )
        .unwrap();

        assert_eq!(evaluation_calls.load(Ordering::Relaxed), 0);
        assert_eq!(fold_rows_calls.load(Ordering::Relaxed), 0);
        assert_eq!(dense_access_calls.load(Ordering::Relaxed), 1);
        let mut verifier_transcript = Blake2bTranscript::<JoltFr>::new(b"claimed-field-poly");
        assert!(DoryScheme::verify_zk(
            &committed.commitment,
            &point,
            &proof,
            &verifier.pcs,
            &mut verifier_transcript,
        )
        .is_ok());
    }

    #[test]
    fn hidden_field_batch_rejects_a_claim_not_matching_the_committed_polynomial() {
        let values = [
            Fr::from(2_u64),
            Fr::from(3_u64),
            Fr::from(5_u64),
            Fr::from(7_u64),
        ];
        let point = [JoltFr::from(11_u64), JoltFr::from(13_u64)];
        let actual = values
            .iter()
            .copied()
            .map(JoltFr::from)
            .collect::<Vec<_>>()
            .evaluate(&point);
        let claimed = actual + JoltFr::from(1_u64);
        let (prover, verifier) = setup_dory_parameters(2);
        let committed = commit_field_vector_zk(&prover, &values).unwrap();
        let evaluation_blind = JoltFr::from(17_u64);
        let evaluation_commitment =
            Pedersen::<Bn254G1>::commit(&prover.blindfold, &[claimed], &evaluation_blind);
        let (proof, _, _, _) = open_hidden_field_vectors_batch_zk(
            &prover,
            &[&values],
            &[&committed],
            &point,
            &[claimed],
            &[evaluation_commitment],
            b"wrong-claim",
            b"claimed-field-batch",
        )
        .unwrap();

        assert!(verify_hidden_field_vectors_batch_zk(
            &verifier,
            &[&committed.commitment],
            values.len(),
            &point,
            &[evaluation_commitment],
            &proof,
            b"wrong-claim",
            b"claimed-field-batch",
        )
        .is_err());
    }

    #[test]
    fn claimed_ark_field_polynomial_fuses_conversion_with_row_fold() {
        let values = (1_u64..=16).map(Fr::from).collect::<Vec<_>>();
        let point = [
            JoltFr::from(2_u64),
            JoltFr::from(3_u64),
            JoltFr::from(5_u64),
            JoltFr::from(7_u64),
        ];
        let values_jolt = values.iter().copied().map(JoltFr::from).collect::<Vec<_>>();
        let expected = values_jolt.evaluate(&point);
        let source = ClaimedArkFieldPolynomial::new(&values, &point, expected);
        let left = [JoltFr::from(11_u64), JoltFr::from(13_u64)];

        assert_eq!(source.evaluate(&point), expected);
        assert_eq!(source.fold_rows(&left, 3), values_jolt.fold_rows(&left, 3));
    }

    #[test]
    fn large_field_matrix_commit_is_safe_inside_a_sixty_four_thread_pool() {
        let (prover, _) = setup_dory_parameters(18);
        let columns = (0..64)
            .map(|column| {
                (0..4096)
                    .map(|row| Fr::from((row + 1) as u64) * Fr::from(u64::MAX - column as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(64)
            .build()
            .unwrap();

        let committed = pool
            .install(|| commit_field_matrix_columns_zk(&prover, &columns, 64))
            .unwrap();

        assert_eq!(committed.len(), columns.len());
        assert!(committed
            .windows(2)
            .all(|pair| pair[0].commitment == pair[1].commitment));
    }

    #[test]
    fn field_matrix_commit_retains_the_logical_column_count() {
        let (prover, _) = setup_dory_parameters(4);
        let columns = (0..3)
            .map(|column| {
                (0..4)
                    .map(|row| Fr::from((4 * column + row + 1) as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let committed = commit_field_matrix_columns_zk(&prover, &columns, 4).unwrap();
        let matrix = committed[0].matrix.as_ref().unwrap();

        assert_eq!(matrix.logical_cols, 3);
        assert_eq!(matrix.padded_cols, 4);
        assert!(committed
            .iter()
            .all(|column| column.matrix.as_ref().unwrap().logical_cols == columns.len()));
    }

    #[test]
    fn field_matrix_columns_share_hint_and_full_matrix_opening_verifies() {
        let (prover, verifier) = setup_dory_parameters(4);
        let columns = (0..3)
            .map(|column| {
                (0..4)
                    .map(|row| Fr::from((4 * column + row + 1) as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let committed = commit_field_matrix_columns_zk(&prover, &columns, 4).unwrap();
        let matrix = committed[0].matrix.as_ref().unwrap();

        assert!(committed
            .iter()
            .all(|column| Arc::ptr_eq(&column.hint, &matrix.hint)));
        assert_eq!(Arc::strong_count(&matrix.hint), committed.len() + 1);

        let whole_matrix = super::CommittedDoryFieldVector {
            commitment: matrix.commitment.clone(),
            logical_len: matrix.values.len(),
            hint: matrix.hint.clone(),
            matrix: None,
            matrix_column: 0,
            grid_num_vars: None,
        };
        let point = [
            Fr::from(2_u64),
            Fr::from(3_u64),
            Fr::from(5_u64),
            Fr::from(7_u64),
        ];
        let (proof, evaluation) = open_field_vector_zk(
            &prover,
            &matrix.values,
            &whole_matrix,
            &point,
            b"shared-matrix-hint",
            b"shared-matrix-hint-round-trip",
        )
        .unwrap();

        assert!(verify_field_vector_zk(
            &verifier,
            &whole_matrix.commitment,
            whole_matrix.logical_len,
            &point,
            evaluation,
            &proof,
            b"shared-matrix-hint",
            b"shared-matrix-hint-round-trip",
        )
        .is_ok());
    }

    #[test]
    fn monomial_claim_normalizes_to_jolt_msb_standard_mle() {
        let values = [3_i8, -2, 5, 7];
        let point_low = [Fr::from(2_u64), Fr::from(3_u64)];
        let monomial_value = Fr::from(3_u64) - Fr::from(2_u64) * point_low[0]
            + Fr::from(5_u64) * point_low[1]
            + Fr::from(7_u64) * point_low[0] * point_low[1];
        let normalized = normalize_monomial_mle_claim(&point_low, monomial_value).unwrap();
        assert_eq!(
            standard_mle_evaluate_i8(&values, 4, &normalized.point_msb).unwrap(),
            normalized.value
        );
        assert_eq!(normalized.value * normalized.scale, monomial_value);

        assert_eq!(
            standard_mle_evaluate_i8(&values, 4, &[Fr::zero(), Fr::one()]).unwrap(),
            -Fr::from(2_u64)
        );
        assert_eq!(
            standard_mle_evaluate_i8(&values, 4, &[Fr::one(), Fr::zero()]).unwrap(),
            Fr::from(5_u64)
        );
    }

    #[test]
    fn monomial_point_with_minus_one_is_rejected() {
        let error = normalize_monomial_mle_claim(&[-Fr::one()], Fr::from(9_u64)).unwrap_err();
        assert!(error.to_string().contains("-1"));
    }

    #[test]
    fn field_vector_dory_round_trip_binds_role_outer_statement_and_evaluation() {
        let values = [
            Fr::from(1_u64),
            Fr::from(2_u64),
            Fr::from(3_u64),
            Fr::from(4_u64),
        ];
        let point = [Fr::from(2_u64), Fr::from(3_u64)];
        let one = Fr::one();
        let expected = values[0] * (one - point[0]) * (one - point[1])
            + values[1] * (one - point[0]) * point[1]
            + values[2] * point[0] * (one - point[1])
            + values[3] * point[0] * point[1];
        assert_eq!(
            evaluate_field_vector_standard(&values, &point).unwrap(),
            expected
        );

        let (prover, verifier) = setup_dory_parameters(2);
        let committed = commit_field_vector_zk(&prover, &values).unwrap();
        let (proof, evaluation) = open_field_vector_zk(
            &prover,
            &values,
            &committed,
            &point,
            b"shape-alpha",
            b"registered-matrix-collection",
        )
        .unwrap();
        assert_eq!(evaluation, expected);
        assert!(verify_field_vector_zk(
            &verifier,
            &committed.commitment,
            committed.logical_len,
            &point,
            evaluation,
            &proof,
            b"shape-alpha",
            b"registered-matrix-collection",
        )
        .is_ok());
        assert!(verify_field_vector_zk(
            &verifier,
            &committed.commitment,
            committed.logical_len,
            &point,
            evaluation + Fr::one(),
            &proof,
            b"shape-alpha",
            b"registered-matrix-collection",
        )
        .is_err());
        assert!(verify_field_vector_zk(
            &verifier,
            &committed.commitment,
            committed.logical_len,
            &point,
            evaluation,
            &proof,
            b"shape-gamma",
            b"registered-matrix-collection",
        )
        .is_err());
        assert!(verify_field_vector_zk(
            &verifier,
            &committed.commitment,
            committed.logical_len,
            &point,
            evaluation,
            &proof,
            b"shape-alpha",
            b"different-outer-statement",
        )
        .is_err());
    }

    #[test]
    fn four_bit_dory_round_trip_uses_sixty_four_packed_lanes() {
        let mut source = ByteSource::new(vec![
            (0..32).map(|index| (index % 15) as i8 - 7).collect(),
            (0..16).map(|index| 7 - (index % 15) as i8).collect(),
        ]);
        let point = (2_u64..8).map(Fr::from).collect::<Vec<_>>();
        let tbv = TensorBasisVector::global_tensor(&source.layout, &point).unwrap();
        let claimed = tbv.inner_product_i8_streaming(&mut source).unwrap();
        let (prover, verifier) = setup_dory_parameters(6);
        let (proof, profile) =
            prove_packed_i8_parcomm(&prover, &mut source, &tbv, claimed, 4).unwrap();

        assert_eq!(proof.packed_eo.packing_stride, 64);
        assert_eq!(profile.commitment.physical_entries, 48);
        assert_eq!(profile.commitment.nonzero_entries, 45);
        assert_eq!(profile.opening.physical_entries, 48);
        assert_eq!(profile.opening.nonzero_entries, 45);
        assert_eq!(profile.commitment.materialized_field_entries, 0);
        assert!(verify_packed_i8_parcomm(&verifier, &tbv, claimed, 4, &proof).unwrap());
    }

    #[test]
    fn packed_eo_dory_round_trip_rejects_tampering_and_trailing_bytes() {
        let mut source = ByteSource::new(vec![
            vec![1, -2, 3, -4, 5, -6, 7, -8],
            vec![9, -10, 11, -12],
        ]);
        let point = [
            Fr::from(2_u64),
            Fr::from(3_u64),
            Fr::from(5_u64),
            Fr::from(7_u64),
        ];
        let tbv = TensorBasisVector::global_tensor(&source.layout, &point).unwrap();
        let claimed = tbv.inner_product_i8_streaming(&mut source).unwrap();
        let (prover, verifier) = setup_dory_parameters(4);
        let (proof, profile) =
            prove_packed_i8_parcomm(&prover, &mut source, &tbv, claimed, 8).unwrap();
        assert_eq!(profile.commitment.materialized_field_entries, 0);
        assert_eq!(profile.commitment.materialized_padding_entries, 0);

        let bytes = proof.to_bytes().unwrap();
        let decoded = PackedI8ParCommProof::from_bytes_exact(&bytes).unwrap();
        assert_eq!(decoded, proof);
        assert!(verify_packed_i8_parcomm(&verifier, &tbv, claimed, 8, &decoded).unwrap());

        let mut trailing = bytes.clone();
        trailing.push(0);
        assert!(PackedI8ParCommProof::from_bytes_exact(&trailing).is_err());

        let mut tampered = decoded;
        tampered.opening.evaluation_blind += JoltFr::from(1_u64);
        assert!(verify_packed_i8_parcomm(&verifier, &tbv, claimed, 8, &tampered).is_err());
    }
}
