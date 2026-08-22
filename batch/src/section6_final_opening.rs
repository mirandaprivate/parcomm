//! One terminal Dory opening for heterogeneous Section 6 MLE claims.

use std::{collections::HashMap, fmt::Display, sync::Arc};

use ark_bn254::Fr as ArkFr;
use ark_ff::{One, Zero};
use ark_std::rand::RngCore;
use jolt_blindfold::{
    prove as prove_blindfold, BlindFoldProof, BlindFoldProtocol, BlindFoldWitness,
};
use jolt_claims::{constant, opening, Expr};
use jolt_crypto::{Bn254G1, HomomorphicCommitment, Pedersen, VectorCommitment};
use jolt_dory::{DoryCommitment, DoryHint, DoryScheme};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_openings::{
    AdditivelyHomomorphic, HomomorphicBatch, StreamingCommitment, ZkBatchOpeningScheme,
    ZkEvaluationClaim, ZkOpeningScheme,
};
use jolt_poly::{EqPolynomial, MultilinearPoly, Point, RlcSource, UnivariatePoly};
use jolt_sumcheck::{
    CommittedSumcheckBuilder, CommittedSumcheckProof, SumcheckDomainSpec, SumcheckProof,
    SumcheckStatement,
};
use jolt_transcript::domain::{Label, LabelWithCount, U64Word};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Transcript};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    hidden_relation::{HiddenRelationTable, HiddenRelationTableStatement, HiddenScalarWitness},
    jolt_dory::{
        commitment_binding_bytes, CommittedDoryBva, CommittedDoryFieldMatrix,
        CommittedDoryFieldVector, DenseI8BvaSource, DoryCommitProfile, DoryProverParameters,
        DoryVerifierParameters, HiddenFieldVectorDoryProof,
    },
    jolt_dory_adapter::DenseI8DoryError,
};

const SUMCHECK_DEGREE: usize = 2;

/// One shared setup size used by every terminal Section 6 commitment.
///
/// Each source is committed in its own balanced Dory shape and embedded in the
/// top-left of this grid only when claims are combined for the final opening.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Section6DoryGrid {
    num_vars: usize,
    logical_len: usize,
}

impl Section6DoryGrid {
    pub fn new(num_vars: usize) -> Result<Self, Section6FinalOpeningError> {
        if num_vars < 2 || num_vars >= usize::BITS as usize {
            return Err(Section6FinalOpeningError::InvalidGrid(num_vars));
        }
        Ok(Self {
            num_vars,
            logical_len: 1_usize << num_vars,
        })
    }

    pub fn num_vars(self) -> usize {
        self.num_vars
    }

    pub fn logical_len(self) -> usize {
        self.logical_len
    }

    fn row_width(self) -> usize {
        1_usize << self.num_vars.div_ceil(2)
    }

    fn row_count(self) -> usize {
        self.logical_len / self.row_width()
    }

    /// Return the balanced native Dory shape for one power-of-two table.
    pub fn native_shape(
        self,
        logical_len: usize,
    ) -> Result<Section6DoryShape, Section6FinalOpeningError> {
        validate_native_len(logical_len, self)?;
        Ok(Section6DoryShape {
            global_num_vars: self.num_vars,
            native_num_vars: logical_len.trailing_zeros() as usize,
        })
    }
}

/// A native near-square Dory matrix embedded in the top-left of the shared grid.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Section6DoryShape {
    global_num_vars: usize,
    native_num_vars: usize,
}

impl Section6DoryShape {
    fn global_row_vars(self) -> usize {
        self.global_num_vars / 2
    }

    fn global_column_vars(self) -> usize {
        self.global_num_vars.div_ceil(2)
    }

    fn native_row_vars(self) -> usize {
        self.native_num_vars / 2
    }

    fn native_column_vars(self) -> usize {
        self.native_num_vars.div_ceil(2)
    }

    fn row_selector_rounds(self) -> std::ops::Range<usize> {
        0..self.global_row_vars() - self.native_row_vars()
    }

    fn column_selector_rounds(self) -> std::ops::Range<usize> {
        self.global_row_vars()
            ..self.global_row_vars() + self.global_column_vars() - self.native_column_vars()
    }

    fn is_selector_round(self, round_index: usize) -> bool {
        self.row_selector_rounds().contains(&round_index)
            || self.column_selector_rounds().contains(&round_index)
    }

    pub fn row_count(self) -> usize {
        1_usize << self.native_row_vars()
    }

    pub fn row_width(self) -> usize {
        1_usize << self.native_column_vars()
    }

    /// Insert zero selectors independently before the native row and column axes.
    pub fn embed_native_point(
        self,
        point_msb: &[ArkFr],
    ) -> Result<Vec<ArkFr>, Section6FinalOpeningError> {
        if point_msb.len() != self.native_num_vars {
            return Err(Section6FinalOpeningError::PointLength {
                expected: self.native_num_vars,
                actual: point_msb.len(),
            });
        }
        let (native_row, native_column) = point_msb.split_at(self.native_row_vars());
        let mut embedded = Vec::with_capacity(self.global_num_vars);
        embedded.resize(
            self.global_row_vars() - self.native_row_vars(),
            ArkFr::zero(),
        );
        embedded.extend_from_slice(native_row);
        embedded.resize(
            self.global_row_vars() + self.global_column_vars() - self.native_column_vars(),
            ArkFr::zero(),
        );
        embedded.extend_from_slice(native_column);
        Ok(embedded)
    }

    /// Project a shared-grid point onto the native axes and return its selector scale.
    pub fn project_global_point(
        self,
        point_msb: &[ArkFr],
    ) -> Result<(Vec<ArkFr>, ArkFr), Section6FinalOpeningError> {
        if point_msb.len() != self.global_num_vars {
            return Err(Section6FinalOpeningError::PointLength {
                expected: self.global_num_vars,
                actual: point_msb.len(),
            });
        }
        let (global_row, global_column) = point_msb.split_at(self.global_row_vars());
        let row_extra = self.global_row_vars() - self.native_row_vars();
        let column_extra = self.global_column_vars() - self.native_column_vars();
        let mut native = Vec::with_capacity(self.native_num_vars);
        native.extend_from_slice(&global_row[row_extra..]);
        native.extend_from_slice(&global_column[column_extra..]);
        let scale = global_row[..row_extra]
            .iter()
            .chain(&global_column[..column_extra])
            .fold(ArkFr::one(), |scale, coordinate| {
                scale * (ArkFr::one() - coordinate)
            });
        Ok((native, scale))
    }

    fn project_global_jolt_point(
        self,
        point_msb: &[Fr],
    ) -> Result<(Vec<Fr>, Fr), Section6FinalOpeningError> {
        if point_msb.len() != self.global_num_vars {
            return Err(Section6FinalOpeningError::PointLength {
                expected: self.global_num_vars,
                actual: point_msb.len(),
            });
        }
        let (global_row, global_column) = point_msb.split_at(self.global_row_vars());
        let row_extra = self.global_row_vars() - self.native_row_vars();
        let column_extra = self.global_column_vars() - self.native_column_vars();
        let mut native = Vec::with_capacity(self.native_num_vars);
        native.extend_from_slice(&global_row[row_extra..]);
        native.extend_from_slice(&global_column[column_extra..]);
        let scale = global_row[..row_extra]
            .iter()
            .chain(&global_column[..column_extra])
            .fold(Fr::one(), |scale, &coordinate| {
                scale * (Fr::one() - coordinate)
            });
        Ok((native, scale))
    }
}

/// Prover-retained state for a source committed on the shared Section 6 grid.
#[derive(Clone)]
pub struct CommittedSection6DorySource {
    pub commitment: DoryCommitment,
    pub logical_len: usize,
    pub grid_num_vars: usize,
    hint: Arc<DoryHint>,
}

/// A native field table opened by the terminal auxiliary batch.
#[derive(Clone, Copy)]
pub enum Section6FinalOpeningData<'a> {
    Field(&'a [ArkFr]),
}

impl Section6FinalOpeningData<'_> {
    fn physical_len(self) -> usize {
        match self {
            Self::Field(values) => values.len(),
        }
    }

    fn value(self, index: usize) -> Fr {
        match self {
            Self::Field(values) => Fr::from(values[index]),
        }
    }
}

/// One hidden arbitrary-point MLE claim held by the prover.
#[derive(Clone, Copy)]
pub struct Section6FinalOpeningProverClaim<'a> {
    pub data: Section6FinalOpeningData<'a>,
    pub committed: &'a CommittedSection6DorySource,
    pub point_msb: &'a [ArkFr],
    pub evaluation: &'a HiddenScalarWitness,
}

/// Commitment-only view of one arbitrary-point MLE claim.
#[derive(Clone, Copy)]
pub struct Section6FinalOpeningVerifierClaim<'a> {
    pub commitment: &'a DoryCommitment,
    pub logical_len: usize,
    pub point_msb: &'a [ArkFr],
    pub evaluation_commitment: Bn254G1,
}

enum ResolvedFieldData<'a> {
    Vector(&'a [ArkFr]),
    Matrix(Arc<CommittedDoryFieldMatrix>),
}

/// Owned adapter from a relation operand to one global-opening prover claim.
pub struct ResolvedSection6FieldProverClaim<'a> {
    source: CommittedSection6DorySource,
    data: ResolvedFieldData<'a>,
    point_msb: Vec<ArkFr>,
}

impl ResolvedSection6FieldProverClaim<'_> {
    pub fn source_key(&self) -> Section6DorySourceKey {
        self.source.key()
    }

    pub fn as_claim<'a>(
        &'a self,
        evaluation: &'a HiddenScalarWitness,
    ) -> Section6FinalOpeningProverClaim<'a> {
        let values = match &self.data {
            ResolvedFieldData::Vector(values) => *values,
            ResolvedFieldData::Matrix(matrix) => matrix.values.as_slice(),
        };
        Section6FinalOpeningProverClaim {
            data: Section6FinalOpeningData::Field(values),
            committed: &self.source,
            point_msb: &self.point_msb,
            evaluation,
        }
    }
}

/// Owned adapter from a relation descriptor to one commitment-only claim.
pub struct ResolvedSection6FieldVerifierClaim<'a> {
    commitment: &'a DoryCommitment,
    logical_len: usize,
    point_msb: Vec<ArkFr>,
    evaluation_commitment: Bn254G1,
    grid_num_vars: usize,
}

impl ResolvedSection6FieldVerifierClaim<'_> {
    pub fn source_key(&self) -> Section6DorySourceKey {
        Section6DorySourceKey {
            commitment: commitment_binding_bytes(self.commitment),
            logical_len: self.logical_len,
            grid_num_vars: self.grid_num_vars,
        }
    }

    pub fn as_claim(&self) -> Section6FinalOpeningVerifierClaim<'_> {
        Section6FinalOpeningVerifierClaim {
            commitment: self.commitment,
            logical_len: self.logical_len,
            point_msb: &self.point_msb,
            evaluation_commitment: self.evaluation_commitment,
        }
    }
}

/// Resolve a relation operand. Matrix columns reuse their one whole-matrix
/// source and turn the column number into a Boolean MLE point prefix.
pub fn resolve_section6_hidden_relation_prover_claim<'a>(
    table: HiddenRelationTable<'a>,
    row_point: &[Fr],
    grid: &Section6DoryGrid,
) -> Result<ResolvedSection6FieldProverClaim<'a>, Section6FinalOpeningError> {
    if row_point.len() != table.committed.logical_len.trailing_zeros() as usize {
        return Err(Section6FinalOpeningError::PointLength {
            expected: table.committed.logical_len.trailing_zeros() as usize,
            actual: row_point.len(),
        });
    }
    let source = CommittedSection6DorySource::from_field_view(table.committed, grid)?;
    let (data, point_msb) = if let Some(matrix) = &table.committed.matrix {
        let mut point = column_point_msb(table.committed.matrix_column, matrix.padded_cols)?;
        point.extend(row_point.iter().copied().map(ArkFr::from));
        (ResolvedFieldData::Matrix(matrix.clone()), point)
    } else {
        (
            ResolvedFieldData::Vector(table.values),
            row_point.iter().copied().map(ArkFr::from).collect(),
        )
    };
    Ok(ResolvedSection6FieldProverClaim {
        source,
        data,
        point_msb,
    })
}

pub fn resolve_section6_hidden_relation_verifier_claim<'a>(
    table: HiddenRelationTableStatement<'a>,
    row_point: &[Fr],
    evaluation_commitment: Bn254G1,
    grid: &Section6DoryGrid,
) -> Result<ResolvedSection6FieldVerifierClaim<'a>, Section6FinalOpeningError> {
    if row_point.len() != table.logical_len.trailing_zeros() as usize {
        return Err(Section6FinalOpeningError::PointLength {
            expected: table.logical_len.trailing_zeros() as usize,
            actual: row_point.len(),
        });
    }
    let (logical_len, point_msb) = if let Some(matrix) = table.matrix {
        let mut point = column_point_msb(matrix.column, matrix.padded_cols)?;
        point.extend(row_point.iter().copied().map(ArkFr::from));
        let logical_len = table
            .logical_len
            .checked_mul(matrix.padded_cols)
            .ok_or(Section6FinalOpeningError::InvalidLogicalLength(usize::MAX))?;
        (logical_len, point)
    } else {
        (
            table.logical_len,
            row_point.iter().copied().map(ArkFr::from).collect(),
        )
    };
    validate_native_len(logical_len, *grid)?;
    Ok(ResolvedSection6FieldVerifierClaim {
        commitment: table.commitment,
        logical_len,
        point_msb,
        evaluation_commitment,
        grid_num_vars: grid.num_vars,
    })
}

fn column_point_msb(
    column: usize,
    padded_cols: usize,
) -> Result<Vec<ArkFr>, Section6FinalOpeningError> {
    if padded_cols == 0 || !padded_cols.is_power_of_two() || column >= padded_cols {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let num_vars = padded_cols.trailing_zeros() as usize;
    Ok((0..num_vars)
        .rev()
        .map(|bit| ArkFr::from(((column >> bit) & 1) as u64))
        .collect())
}

/// One committed claim reduction, one BlindFold proof, and one native Dory proof.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Section6FinalOpeningProof {
    pub sumcheck: CommittedSumcheckProof<Bn254G1>,
    pub dory: HiddenFieldVectorDoryProof,
    pub blindfold_transcript_state: [u8; 32],
    pub blindfold: BlindFoldProof<Fr, Bn254G1>,
}

impl Section6FinalOpeningProof {
    /// This proof deliberately contains exactly one physical PCS opening.
    pub fn physical_dory_proof_count(&self) -> usize {
        let Self {
            sumcheck: _,
            dory: _,
            blindfold_transcript_state: _,
            blindfold: _,
        } = self;
        1
    }
}

/// Stable identity for deduplicating repeated views of one committed source.
#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct Section6DorySourceKey {
    commitment: Vec<u8>,
    logical_len: usize,
    grid_num_vars: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Section6FinalOpeningError {
    EmptyClaims,
    InvalidGrid(usize),
    InvalidLogicalLength(usize),
    GridMismatch { expected: usize, actual: usize },
    SourceLength { logical: usize, physical: usize },
    PointLength { expected: usize, actual: usize },
    EvaluationMismatch(usize),
    EvaluationCommitmentMismatch(usize),
    PedersenCapacity { required: usize, available: usize },
    ProofShape,
    Sumcheck(String),
    Dory(String),
    BlindFold(String),
    Encoding(String),
}

impl Display for Section6FinalOpeningError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for Section6FinalOpeningError {}

impl From<DenseI8DoryError> for Section6FinalOpeningError {
    fn from(error: DenseI8DoryError) -> Self {
        Self::Dory(error.to_string())
    }
}

/// Commit a field table in its native balanced Dory shape.
///
/// The global setup generators are reused by prefix, so this commitment is
/// also the commitment to the native matrix embedded in the top-left corner
/// of the shared grid.
pub fn commit_section6_field_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    values: &[ArkFr],
) -> Result<CommittedSection6DorySource, Section6FinalOpeningError> {
    let shape = grid.native_shape(values.len())?;
    let row_width = shape.row_width();
    let mut partial = DoryScheme::begin(&parameters.pcs);
    let mut row = vec![Fr::zero(); row_width];
    for row_index in 0..shape.row_count() {
        row.fill(Fr::zero());
        let start = row_index * row_width;
        let end = start + row_width;
        row.iter_mut()
            .zip(&values[start..end])
            .for_each(|(destination, &value)| *destination = Fr::from(value));
        DoryScheme::feed(&mut partial, &row, &parameters.pcs);
    }
    let (commitment, hint) = DoryScheme::finish_zk(partial, &parameters.pcs);
    Ok(CommittedSection6DorySource {
        commitment,
        logical_len: values.len(),
        grid_num_vars: grid.num_vars,
        hint: Arc::new(hint),
    })
}

/// Commit signed bytes directly, without materializing a field-vector copy.
pub fn commit_section6_dense_i8_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    values: &[i8],
    logical_len: usize,
) -> Result<CommittedSection6DorySource, Section6FinalOpeningError> {
    let shape = grid.native_shape(logical_len)?;
    if values.len() > logical_len {
        return Err(Section6FinalOpeningError::SourceLength {
            logical: logical_len,
            physical: values.len(),
        });
    }
    let row_width = shape.row_width();
    let physical_rows = values.len().div_ceil(row_width);
    let primitive_count = physical_rows * row_width;
    let mut partial = DoryScheme::begin(&parameters.pcs);
    if primitive_count != 0 {
        DoryScheme::feed_i128_rows_with(
            &mut partial,
            |index| values.get(index).copied().unwrap_or(0) as i128,
            primitive_count,
            row_width,
            &parameters.pcs,
        );
    }
    DoryScheme::feed_zeros(
        &mut partial,
        row_width,
        shape.row_count() - physical_rows,
        &parameters.pcs,
    );
    let (commitment, hint) = DoryScheme::finish_zk(partial, &parameters.pcs);
    Ok(CommittedSection6DorySource {
        commitment,
        logical_len,
        grid_num_vars: grid.num_vars,
        hint: Arc::new(hint),
    })
}

impl CommittedSection6DorySource {
    pub fn key(&self) -> Section6DorySourceKey {
        Section6DorySourceKey {
            commitment: commitment_binding_bytes(&self.commitment),
            logical_len: self.logical_len,
            grid_num_vars: self.grid_num_vars,
        }
    }

    /// Recover the global-opening view of a grid-aware relation operand.
    pub fn from_field_view(
        committed: &CommittedDoryFieldVector,
        grid: &Section6DoryGrid,
    ) -> Result<Self, Section6FinalOpeningError> {
        if committed.grid_num_vars != Some(grid.num_vars) {
            return Err(Section6FinalOpeningError::GridMismatch {
                expected: grid.num_vars,
                actual: committed.grid_num_vars.unwrap_or(0),
            });
        }
        let logical_len = committed
            .matrix
            .as_ref()
            .map(|matrix| matrix.values.len())
            .unwrap_or(committed.logical_len);
        Ok(Self {
            commitment: committed.commitment.clone(),
            logical_len,
            grid_num_vars: grid.num_vars,
            hint: committed.hint.clone(),
        })
    }
}

/// Grid-aware constructor compatible with existing relation-vector APIs.
pub fn commit_section6_field_vector_view_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    values: &[ArkFr],
) -> Result<CommittedDoryFieldVector, Section6FinalOpeningError> {
    let source = commit_section6_field_on_grid_zk(parameters, grid, values)?;
    Ok(CommittedDoryFieldVector {
        commitment: source.commitment,
        logical_len: source.logical_len,
        hint: source.hint,
        matrix: None,
        matrix_column: 0,
        grid_num_vars: Some(grid.num_vars),
    })
}

/// Commit one full column-major matrix on the shared grid and return cheap
/// compatibility views for its real columns.
pub fn commit_section6_field_matrix_columns_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    columns: &[&[ArkFr]],
    padded_cols: usize,
) -> Result<Vec<CommittedDoryFieldVector>, Section6FinalOpeningError> {
    if columns.is_empty()
        || padded_cols < columns.len()
        || !padded_cols.is_power_of_two()
        || columns[0].is_empty()
        || !columns[0].len().is_power_of_two()
    {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let padded_rows = columns[0].len();
    if columns.iter().any(|column| column.len() != padded_rows) {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let matrix_len = padded_rows
        .checked_mul(padded_cols)
        .ok_or(Section6FinalOpeningError::InvalidLogicalLength(usize::MAX))?;
    let mut values = vec![ArkFr::zero(); matrix_len];
    for (column, source) in columns.iter().enumerate() {
        let start = column * padded_rows;
        values[start..start + padded_rows].copy_from_slice(source);
    }
    let source = commit_section6_field_on_grid_zk(parameters, grid, &values)?;
    let matrix = Arc::new(CommittedDoryFieldMatrix {
        values,
        padded_rows,
        logical_cols: columns.len(),
        padded_cols,
        commitment: source.commitment.clone(),
        hint: source.hint.clone(),
        grid_num_vars: Some(grid.num_vars),
    });
    Ok((0..columns.len())
        .map(|matrix_column| CommittedDoryFieldVector {
            commitment: source.commitment.clone(),
            logical_len: padded_rows,
            hint: source.hint.clone(),
            matrix: Some(matrix.clone()),
            matrix_column,
            grid_num_vars: Some(grid.num_vars),
        })
        .collect())
}

/// Grid-aware BVA commitment preserving direct signed-byte storage.
pub fn commit_section6_bva_i8_on_grid_zk<S: DenseI8BvaSource>(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    source: &S,
    bit_width: usize,
) -> Result<CommittedDoryBva, Section6FinalOpeningError> {
    if bit_width == 0 || bit_width > 8 {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let logical_len = source.layout().logical_len();
    let values = source.dense_values();
    if values.len() != source.layout().active_len() {
        return Err(Section6FinalOpeningError::SourceLength {
            logical: source.layout().active_len(),
            physical: values.len(),
        });
    }
    if bit_width < 8
        && values
            .par_iter()
            .any(|value| usize::from(value.unsigned_abs()) >= (1_usize << bit_width))
    {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let committed = commit_section6_dense_i8_on_grid_zk(parameters, grid, values, logical_len)?;
    let shape = grid.native_shape(logical_len)?;
    let row_width = shape.row_width();
    Ok(CommittedDoryBva {
        commitment: committed.commitment,
        hint: committed.hint.as_ref().clone(),
        profile: DoryCommitProfile {
            physical_entries: values.len(),
            nonzero_entries: values.par_iter().filter(|&&value| value != 0).count(),
            logical_entries: logical_len,
            row_commitments: shape.row_count(),
            physical_rows: values.len().div_ceil(row_width),
            materialized_field_entries: 0,
            materialized_padding_entries: 0,
            padding_group_operations: 0,
        },
    })
}

/// Reduce heterogeneous arbitrary-point claims and open all sources once.
#[allow(clippy::too_many_arguments)]
pub fn prove_section6_final_opening(
    role: &[u8],
    outer_binding: &[u8],
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    claims: &[Section6FinalOpeningProverClaim<'_>],
    rng: &mut impl RngCore,
) -> Result<Section6FinalOpeningProof, Section6FinalOpeningError> {
    validate_prover_claims(parameters, *grid, claims)?;
    let statements = claims
        .iter()
        .map(|claim| Section6FinalOpeningVerifierClaim {
            commitment: &claim.committed.commitment,
            logical_len: claim.committed.logical_len,
            point_msb: claim.point_msb,
            evaluation_commitment: claim.evaluation.commitment,
        })
        .collect::<Vec<_>>();
    let mut transcript = final_opening_transcript(role, outer_binding, *grid, &statements);
    let claim_coefficients = transcript.challenge_scalar_powers(claims.len());
    let mut states = ReductionStateSet::new(claims, &claim_coefficients, *grid)?;
    let claimed_values = claims
        .iter()
        .map(|claim| Fr::from(claim.evaluation.value))
        .collect::<Vec<_>>();
    let claimed_rlc = claimed_values
        .iter()
        .zip(&claim_coefficients)
        .map(|(&value, &coefficient)| coefficient * value)
        .sum::<Fr>();
    // Reduce the zero-sum polynomial
    //   sum_i gamma_i (P_i(X) eq(z_i, X) - v_i eq(0, X)).
    // This keeps the sumcheck input public while retaining every v_i as a
    // committed terminal opening.
    let mut running_claim = Fr::zero();
    let mut zero_selector_scale = Fr::one();
    let mut builder =
        CommittedSumcheckBuilder::<Fr, Pedersen<Bn254G1>, _>::new(&parameters.blindfold, &mut *rng)
            .map_err(|error| Section6FinalOpeningError::Sumcheck(error.to_string()))?;
    let mut sumcheck_point = Vec::with_capacity(grid.num_vars);
    for round_index in 0..grid.num_vars {
        let round = reduction_round(
            &states.sources,
            round_index,
            claimed_rlc,
            zero_selector_scale,
        )?;
        if round.evaluate(Fr::zero()) + round.evaluate(Fr::one()) != running_claim {
            return Err(Section6FinalOpeningError::EvaluationMismatch(usize::MAX));
        }
        let challenge = builder
            .commit_round(&round, &mut transcript)
            .map_err(|error| Section6FinalOpeningError::Sumcheck(error.to_string()))?;
        running_claim = round.evaluate(challenge);
        states.bind(round_index, challenge)?;
        zero_selector_scale *= Fr::one() - challenge;
        sumcheck_point.push(challenge);
    }
    let terminal_values = states.terminal_values()?;
    let output_values = terminal_values
        .iter()
        .chain(&claimed_values)
        .copied()
        .collect::<Vec<_>>();
    let (sumcheck, sumcheck_witness) = builder
        .finish(&output_values, &mut transcript)
        .map_err(|error| Section6FinalOpeningError::Sumcheck(error.to_string()))?;
    let SumcheckProof::Committed(sumcheck) = sumcheck else {
        return Err(Section6FinalOpeningError::ProofShape);
    };
    let claim_link_coefficients = bind_claim_link_commitments(
        &mut transcript,
        &statements
            .iter()
            .map(|statement| statement.evaluation_commitment)
            .collect::<Vec<_>>(),
    );
    let dory_outer_binding = transcript.state();
    let embedding_scales = states.embedding_scales(&sumcheck_point);
    let full_evaluations = terminal_values
        .iter()
        .zip(&embedding_scales)
        .map(|(&value, &scale)| value * scale)
        .collect::<Vec<_>>();
    let grid_polynomials = claims
        .iter()
        .zip(&full_evaluations)
        .map(|(claim, &evaluation)| {
            GridPolynomial::new(
                claim.data,
                claim.committed.logical_len,
                *grid,
                &sumcheck_point,
                evaluation,
            )
        })
        .collect::<Vec<_>>();
    drop(states);
    let (dory, dory_hiding_commitment, dory_blind, dory_coefficients) = open_grid_batch_zk(
        role,
        &dory_outer_binding,
        parameters,
        *grid,
        &grid_polynomials,
        claims,
        &sumcheck_point,
        &full_evaluations,
        &sumcheck.output_claims.commitments,
    )?;
    bind_dory_completion(&mut transcript, &dory_hiding_commitment, &dory)?;

    let terminal_sumcheck_coefficients =
        terminal_sumcheck_coefficients(*grid, &statements, &sumcheck_point, &claim_coefficients)?;
    let claimed_sumcheck_coefficients =
        claimed_sumcheck_coefficients(&sumcheck_point, &claim_coefficients);
    let terminal_dory_coefficients =
        terminal_dory_coefficients(*grid, &statements, &sumcheck_point, &dory_coefficients)?;
    bind_blindfold_coefficients(
        &mut transcript,
        &claim_coefficients,
        &terminal_sumcheck_coefficients,
        &claimed_sumcheck_coefficients,
        &claim_link_coefficients,
        &terminal_dory_coefficients,
    );
    let final_opening_mix = transcript.challenge_scalar();
    let external_commitments = statements
        .iter()
        .map(|statement| statement.evaluation_commitment)
        .collect::<Vec<_>>();
    let consistency = replay_consistency(
        role,
        outer_binding,
        *grid,
        &statements,
        &sumcheck,
        &claim_coefficients,
        &claim_link_coefficients,
    )?;
    let protocol = build_blindfold_protocol(
        parameters.blindfold.message_generators.len(),
        grid.num_vars,
        claims.len(),
        consistency,
        &sumcheck,
        &claim_coefficients,
        &terminal_sumcheck_coefficients,
        &claimed_sumcheck_coefficients,
        &claim_link_coefficients,
        combine_commitments(&external_commitments, &claim_link_coefficients),
        &terminal_dory_coefficients,
        dory_hiding_commitment,
        final_opening_mix,
    )?;
    let external_evaluation = combine_scalars(&claimed_values, &claim_link_coefficients);
    let external_blind = combine_scalars(
        &claims
            .iter()
            .map(|claim| claim.evaluation.blind)
            .collect::<Vec<_>>(),
        &claim_link_coefficients,
    );
    let dory_evaluation = combine_scalars(&full_evaluations, &dory_coefficients);
    let final_evaluation = external_evaluation + final_opening_mix * dory_evaluation;
    let final_blind = external_blind + final_opening_mix * dory_blind;
    let assigned = protocol
        .assign_witness(
            &[SumcheckDomainSpec::BooleanHypercube],
            &[&sumcheck_witness],
            &[final_evaluation],
            &[final_blind],
            &mut *rng,
        )
        .map_err(|error| Section6FinalOpeningError::BlindFold(error.to_string()))?;
    let blindfold_transcript_state = transcript.state();
    let blindfold = prove_blindfold::<Fr, Pedersen<Bn254G1>, _, _>(
        &parameters.blindfold,
        &protocol,
        &mut transcript,
        BlindFoldWitness {
            rows: &assigned.rows,
            blindings: &assigned.blindings,
            eval_outputs: &[final_evaluation],
            eval_blindings: &[final_blind],
        },
        &mut *rng,
    )
    .map_err(|error| Section6FinalOpeningError::BlindFold(error.to_string()))?;
    Ok(Section6FinalOpeningProof {
        sumcheck,
        dory,
        blindfold_transcript_state,
        blindfold,
    })
}

/// Verify the terminal proof without reading any source evaluation table.
#[allow(clippy::too_many_arguments)]
pub fn verify_section6_final_opening(
    role: &[u8],
    outer_binding: &[u8],
    parameters: &DoryVerifierParameters,
    grid: &Section6DoryGrid,
    claims: &[Section6FinalOpeningVerifierClaim<'_>],
    proof: &Section6FinalOpeningProof,
) -> Result<(), Section6FinalOpeningError> {
    validate_verifier_claims(parameters, *grid, claims)?;
    let mut transcript = final_opening_transcript(role, outer_binding, *grid, claims);
    let claim_coefficients = transcript.challenge_scalar_powers(claims.len());
    let consistency = SumcheckProof::<Fr, Bn254G1>::Committed(proof.sumcheck.clone())
        .verify_committed_consistency(
            SumcheckStatement::new(grid.num_vars, SUMCHECK_DEGREE),
            &mut transcript,
        )
        .map_err(|error| Section6FinalOpeningError::Sumcheck(error.to_string()))?;
    let sumcheck_point = consistency.challenges();
    let external_commitments = claims
        .iter()
        .map(|claim| claim.evaluation_commitment)
        .collect::<Vec<_>>();
    let claim_link_coefficients =
        bind_claim_link_commitments(&mut transcript, &external_commitments);
    let dory_outer_binding = transcript.state();
    let (dory_hiding_commitment, dory_coefficients) = verify_grid_batch_zk(
        role,
        &dory_outer_binding,
        parameters,
        *grid,
        claims,
        &sumcheck_point,
        &proof.sumcheck.output_claims.commitments,
        &proof.dory,
    )?;
    bind_dory_completion(&mut transcript, &dory_hiding_commitment, &proof.dory)?;
    let terminal_sumcheck_coefficients =
        terminal_sumcheck_coefficients(*grid, claims, &sumcheck_point, &claim_coefficients)?;
    let claimed_sumcheck_coefficients =
        claimed_sumcheck_coefficients(&sumcheck_point, &claim_coefficients);
    let terminal_dory_coefficients =
        terminal_dory_coefficients(*grid, claims, &sumcheck_point, &dory_coefficients)?;
    bind_blindfold_coefficients(
        &mut transcript,
        &claim_coefficients,
        &terminal_sumcheck_coefficients,
        &claimed_sumcheck_coefficients,
        &claim_link_coefficients,
        &terminal_dory_coefficients,
    );
    let final_opening_mix = transcript.challenge_scalar();
    let protocol = build_blindfold_protocol(
        parameters.blindfold.message_generators.len(),
        grid.num_vars,
        claims.len(),
        consistency,
        &proof.sumcheck,
        &claim_coefficients,
        &terminal_sumcheck_coefficients,
        &claimed_sumcheck_coefficients,
        &claim_link_coefficients,
        combine_commitments(&external_commitments, &claim_link_coefficients),
        &terminal_dory_coefficients,
        dory_hiding_commitment,
        final_opening_mix,
    )?;
    if transcript.state() != proof.blindfold_transcript_state {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    protocol
        .verify::<Pedersen<Bn254G1>, _>(&proof.blindfold, &parameters.blindfold, &mut transcript)
        .map_err(|error| Section6FinalOpeningError::BlindFold(format!("{error:?}")))
}

fn validate_native_len(
    logical_len: usize,
    grid: Section6DoryGrid,
) -> Result<(), Section6FinalOpeningError> {
    if logical_len == 0 || !logical_len.is_power_of_two() || logical_len > grid.logical_len {
        return Err(Section6FinalOpeningError::InvalidLogicalLength(logical_len));
    }
    Ok(())
}

fn validate_prover_claims(
    parameters: &DoryProverParameters,
    grid: Section6DoryGrid,
    claims: &[Section6FinalOpeningProverClaim<'_>],
) -> Result<(), Section6FinalOpeningError> {
    if claims.is_empty() {
        return Err(Section6FinalOpeningError::EmptyClaims);
    }
    if parameters.blindfold.message_generators.len() < SUMCHECK_DEGREE + 1 {
        return Err(Section6FinalOpeningError::PedersenCapacity {
            required: SUMCHECK_DEGREE + 1,
            available: parameters.blindfold.message_generators.len(),
        });
    }
    for (index, claim) in claims.iter().enumerate() {
        validate_native_len(claim.committed.logical_len, grid)?;
        if claim.committed.grid_num_vars != grid.num_vars {
            return Err(Section6FinalOpeningError::GridMismatch {
                expected: grid.num_vars,
                actual: claim.committed.grid_num_vars,
            });
        }
        let native_vars = claim.committed.logical_len.trailing_zeros() as usize;
        if claim.point_msb.len() != native_vars {
            return Err(Section6FinalOpeningError::PointLength {
                expected: native_vars,
                actual: claim.point_msb.len(),
            });
        }
        match claim.data {
            Section6FinalOpeningData::Field(values)
                if values.len() != claim.committed.logical_len =>
            {
                return Err(Section6FinalOpeningError::SourceLength {
                    logical: claim.committed.logical_len,
                    physical: values.len(),
                });
            }
            _ => {}
        }
        if !Pedersen::<Bn254G1>::verify(
            &parameters.blindfold,
            &claim.evaluation.commitment,
            &[Fr::from(claim.evaluation.value)],
            &claim.evaluation.blind,
        ) {
            return Err(Section6FinalOpeningError::EvaluationCommitmentMismatch(
                index,
            ));
        }
    }
    Ok(())
}

fn validate_verifier_claims(
    parameters: &DoryVerifierParameters,
    grid: Section6DoryGrid,
    claims: &[Section6FinalOpeningVerifierClaim<'_>],
) -> Result<(), Section6FinalOpeningError> {
    if claims.is_empty() {
        return Err(Section6FinalOpeningError::EmptyClaims);
    }
    if parameters.blindfold.message_generators.len() < SUMCHECK_DEGREE + 1 {
        return Err(Section6FinalOpeningError::PedersenCapacity {
            required: SUMCHECK_DEGREE + 1,
            available: parameters.blindfold.message_generators.len(),
        });
    }
    for claim in claims {
        validate_native_len(claim.logical_len, grid)?;
        let native_vars = claim.logical_len.trailing_zeros() as usize;
        if claim.point_msb.len() != native_vars {
            return Err(Section6FinalOpeningError::PointLength {
                expected: native_vars,
                actual: claim.point_msb.len(),
            });
        }
    }
    Ok(())
}

struct ReductionSourceState {
    shape: Section6DoryShape,
    selector_square: Fr,
    values: Vec<Fr>,
    eq: Vec<Fr>,
}

impl ReductionSourceState {
    fn new(
        claim: Section6FinalOpeningProverClaim<'_>,
        grid: Section6DoryGrid,
    ) -> Result<Self, Section6FinalOpeningError> {
        if claim.data.physical_len() > claim.committed.logical_len {
            return Err(Section6FinalOpeningError::SourceLength {
                logical: claim.committed.logical_len,
                physical: claim.data.physical_len(),
            });
        }
        let mut values = (0..claim.data.physical_len())
            .into_par_iter()
            .map(|index| claim.data.value(index))
            .collect::<Vec<_>>();
        values.resize(claim.committed.logical_len, Fr::zero());
        Ok(Self {
            shape: grid.native_shape(claim.committed.logical_len)?,
            selector_square: Fr::one(),
            values,
            eq: Vec::new(),
        })
    }

    fn bind(&mut self, round_index: usize, challenge: Fr) -> Result<(), Section6FinalOpeningError> {
        if self.shape.is_selector_round(round_index) {
            let selector = Fr::one() - challenge;
            self.selector_square *= selector * selector;
            return Ok(());
        }
        bind_msb(&mut self.values, challenge)?;
        bind_msb(&mut self.eq, challenge)
    }

    fn terminal_value(&self) -> Result<Fr, Section6FinalOpeningError> {
        self.values
            .first()
            .copied()
            .filter(|_| self.values.len() == 1)
            .ok_or(Section6FinalOpeningError::ProofShape)
    }

    fn remaining_inner_product(&self) -> Fr {
        assert_eq!(self.values.len(), self.eq.len());
        self.values
            .par_iter()
            .zip(self.eq.par_iter())
            .map(|(&value, &weight)| value * weight)
            .sum()
    }
}

struct ReductionClaimState {
    source_index: usize,
}

struct ReductionStateSet {
    sources: Vec<ReductionSourceState>,
    claims: Vec<ReductionClaimState>,
}

impl ReductionStateSet {
    fn new(
        prover_claims: &[Section6FinalOpeningProverClaim<'_>],
        coefficients: &[Fr],
        grid: Section6DoryGrid,
    ) -> Result<Self, Section6FinalOpeningError> {
        if prover_claims.len() != coefficients.len() {
            return Err(Section6FinalOpeningError::ProofShape);
        }
        let mut source_indices = HashMap::<Section6DorySourceKey, usize>::new();
        let mut sources = Vec::new();
        let mut claims = Vec::with_capacity(prover_claims.len());
        for (claim_index, (&claim, &coefficient)) in
            prover_claims.iter().zip(coefficients).enumerate()
        {
            let key = claim.committed.key();
            let source_index = if let Some(&source_index) = source_indices.get(&key) {
                source_index
            } else {
                let source_index = sources.len();
                sources.push(ReductionSourceState::new(claim, grid)?);
                source_indices.insert(key, source_index);
                source_index
            };
            let point = claim
                .point_msb
                .iter()
                .copied()
                .map(Fr::from)
                .collect::<Vec<_>>();
            let mut claim_eq = EqPolynomial::<Fr>::evals(&point, None);
            let source = &mut sources[source_index];
            let native_evaluation = source
                .values
                .par_iter()
                .zip(claim_eq.par_iter())
                .map(|(&value, &weight)| value * weight)
                .sum::<Fr>();
            if native_evaluation != Fr::from(claim.evaluation.value) {
                return Err(Section6FinalOpeningError::EvaluationMismatch(claim_index));
            }
            if source.eq.is_empty() {
                claim_eq
                    .par_iter_mut()
                    .for_each(|weight| *weight *= coefficient);
                source.eq = claim_eq;
            } else {
                source
                    .eq
                    .par_iter_mut()
                    .zip(claim_eq.par_iter())
                    .for_each(|(combined, &weight)| *combined += coefficient * weight);
            }
            claims.push(ReductionClaimState { source_index });
        }
        Ok(Self { sources, claims })
    }

    fn bind(&mut self, round_index: usize, challenge: Fr) -> Result<(), Section6FinalOpeningError> {
        self.sources
            .par_iter_mut()
            .try_for_each(|source| source.bind(round_index, challenge))
    }

    fn terminal_values(&self) -> Result<Vec<Fr>, Section6FinalOpeningError> {
        self.claims
            .iter()
            .map(|claim| self.sources[claim.source_index].terminal_value())
            .collect()
    }

    fn embedding_scales(&self, full_point: &[Fr]) -> Vec<Fr> {
        self.claims
            .iter()
            .map(|claim| {
                self.sources[claim.source_index]
                    .shape
                    .project_global_jolt_point(full_point)
                    .expect("validated shared-grid point")
                    .1
            })
            .collect()
    }
}

fn reduction_round(
    sources: &[ReductionSourceState],
    round_index: usize,
    claimed_rlc: Fr,
    zero_selector_scale: Fr,
) -> Result<UnivariatePoly<Fr>, Section6FinalOpeningError> {
    let two = Fr::from_u64(2);
    let mut polynomial = sources
        .par_iter()
        .map(|source| {
            if source.shape.is_selector_round(round_index) {
                let base = source.selector_square * source.remaining_inner_product();
                [base, -(two * base), base]
            } else {
                let half = source.values.len() / 2;
                (0..half)
                    .into_par_iter()
                    .map(|index| {
                        let value_zero = source.values[index];
                        let value_delta = source.values[index + half] - value_zero;
                        let eq_zero = source.eq[index];
                        let eq_delta = source.eq[index + half] - eq_zero;
                        [
                            source.selector_square * value_zero * eq_zero,
                            source.selector_square
                                * (value_zero * eq_delta + value_delta * eq_zero),
                            source.selector_square * value_delta * eq_delta,
                        ]
                    })
                    .reduce(
                        || [Fr::zero(); 3],
                        |mut left, right| {
                            for index in 0..3 {
                                left[index] += right[index];
                            }
                            left
                        },
                    )
            }
        })
        .reduce(
            || [Fr::zero(); 3],
            |mut left, right| {
                for index in 0..3 {
                    left[index] += right[index];
                }
                left
            },
        );
    let zero_term = claimed_rlc * zero_selector_scale;
    polynomial[0] -= zero_term;
    polynomial[1] += zero_term;
    Ok(UnivariatePoly::new(polynomial.to_vec()))
}

fn bind_msb(values: &mut Vec<Fr>, challenge: Fr) -> Result<(), Section6FinalOpeningError> {
    if values.len() < 2 || !values.len().is_power_of_two() {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let half = values.len() / 2;
    let (left, right) = values.split_at_mut(half);
    left.par_iter_mut()
        .zip(right.par_iter())
        .for_each(|(left, &right)| *left += challenge * (right - *left));
    values.truncate(half);
    Ok(())
}

fn terminal_sumcheck_coefficients(
    grid: Section6DoryGrid,
    claims: &[Section6FinalOpeningVerifierClaim<'_>],
    sumcheck_point: &[Fr],
    claim_coefficients: &[Fr],
) -> Result<Vec<Fr>, Section6FinalOpeningError> {
    claims
        .iter()
        .zip(claim_coefficients)
        .map(|(claim, &coefficient)| {
            let shape = grid.native_shape(claim.logical_len)?;
            let (native_point, scale) = shape.project_global_jolt_point(sumcheck_point)?;
            let point = claim
                .point_msb
                .iter()
                .copied()
                .map(Fr::from)
                .collect::<Vec<_>>();
            Ok(coefficient * scale * scale * EqPolynomial::<Fr>::mle(&point, &native_point))
        })
        .collect()
}

fn claimed_sumcheck_coefficients(sumcheck_point: &[Fr], claim_coefficients: &[Fr]) -> Vec<Fr> {
    let zero_evaluation = sumcheck_point.iter().fold(Fr::one(), |scale, &coordinate| {
        scale * (Fr::one() - coordinate)
    });
    claim_coefficients
        .iter()
        .map(|&coefficient| -(zero_evaluation * coefficient))
        .collect()
}

fn terminal_dory_coefficients(
    grid: Section6DoryGrid,
    claims: &[Section6FinalOpeningVerifierClaim<'_>],
    sumcheck_point: &[Fr],
    dory_coefficients: &[Fr],
) -> Result<Vec<Fr>, Section6FinalOpeningError> {
    claims
        .iter()
        .zip(dory_coefficients)
        .map(|(claim, &coefficient)| {
            let shape = grid.native_shape(claim.logical_len)?;
            let (_, scale) = shape.project_global_jolt_point(sumcheck_point)?;
            Ok(coefficient * scale)
        })
        .collect()
}

#[derive(Clone, Copy)]
struct GridPolynomial<'a> {
    data: Section6FinalOpeningData<'a>,
    logical_len: usize,
    grid: Section6DoryGrid,
    opening_point: &'a [Fr],
    evaluation: Fr,
}

impl<'a> GridPolynomial<'a> {
    fn new(
        data: Section6FinalOpeningData<'a>,
        logical_len: usize,
        grid: Section6DoryGrid,
        opening_point: &'a [Fr],
        evaluation: Fr,
    ) -> Self {
        Self {
            data,
            logical_len,
            grid,
            opening_point,
            evaluation,
        }
    }
}

impl MultilinearPoly<Fr> for GridPolynomial<'_> {
    fn num_vars(&self) -> usize {
        self.grid.num_vars
    }

    fn evaluate(&self, point: &[Fr]) -> Fr {
        assert_eq!(point, self.opening_point);
        self.evaluation
    }

    fn for_each_row(&self, sigma: usize, visit: &mut dyn FnMut(usize, &[Fr])) {
        assert!(sigma <= self.grid.num_vars);
        let row_width = 1_usize << sigma;
        let total_rows = self.grid.logical_len / row_width;
        let shape = self
            .grid
            .native_shape(self.logical_len)
            .expect("validated native Dory shape");
        let native_row_width = shape.row_width();
        assert!(native_row_width <= row_width);
        let mut row = vec![Fr::zero(); row_width];
        for row_index in 0..total_rows {
            row.fill(Fr::zero());
            let start = row_index * native_row_width;
            let end = (start + native_row_width).min(self.data.physical_len());
            if start < end {
                row[..native_row_width]
                    .iter_mut()
                    .zip(start..end)
                    .for_each(|(destination, index)| *destination = self.data.value(index));
            }
            visit(row_index, &row);
        }
    }

    fn fold_rows(&self, left: &[Fr], sigma: usize) -> Vec<Fr> {
        assert!(sigma <= self.grid.num_vars);
        let row_width = 1_usize << sigma;
        let row_count = self.grid.logical_len / row_width;
        assert_eq!(left.len(), row_count);
        let shape = self
            .grid
            .native_shape(self.logical_len)
            .expect("validated native Dory shape");
        let native_row_width = shape.row_width();
        assert!(native_row_width <= row_width);
        let physical_rows = self.data.physical_len().div_ceil(native_row_width);
        if physical_rows == 0 {
            return vec![Fr::zero(); row_width];
        }
        let workers = rayon::current_num_threads().min(physical_rows);
        (0..workers)
            .into_par_iter()
            .map(|worker| {
                let start_row = worker * physical_rows / workers;
                let end_row = (worker + 1) * physical_rows / workers;
                let mut output = vec![Fr::zero(); row_width];
                for row_index in start_row..end_row {
                    let start = row_index * native_row_width;
                    let end = (start + native_row_width).min(self.data.physical_len());
                    let weight = left[row_index];
                    output[..native_row_width]
                        .iter_mut()
                        .zip(start..end)
                        .for_each(|(destination, index)| {
                            *destination += weight * self.data.value(index)
                        });
                }
                output
            })
            .reduce(
                || vec![Fr::zero(); row_width],
                |mut left, right| {
                    left.iter_mut()
                        .zip(right)
                        .for_each(|(destination, value)| *destination += value);
                    left
                },
            )
    }
}

#[allow(clippy::too_many_arguments)]
fn open_grid_batch_zk(
    role: &[u8],
    outer_binding: &[u8],
    parameters: &DoryProverParameters,
    grid: Section6DoryGrid,
    polynomials: &[GridPolynomial<'_>],
    claims: &[Section6FinalOpeningProverClaim<'_>],
    point: &[Fr],
    evaluations: &[Fr],
    terminal_rows: &[Bn254G1],
) -> Result<(HiddenFieldVectorDoryProof, Bn254G1, Fr, Vec<Fr>), Section6FinalOpeningError> {
    if polynomials.len() != claims.len() || evaluations.len() != claims.len() {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let commitments = claims
        .iter()
        .map(|claim| claim.committed.commitment.clone())
        .collect::<Vec<_>>();
    let logical_lengths = claims
        .iter()
        .map(|claim| claim.committed.logical_len)
        .collect::<Vec<_>>();
    let mut transcript = grid_dory_transcript(
        role,
        outer_binding,
        grid,
        point,
        &commitments,
        &logical_lengths,
        terminal_rows,
    );
    let coefficients = transcript.challenge_scalar_powers(claims.len());
    let mut group_indices = HashMap::<Section6DorySourceKey, usize>::new();
    let mut representatives = Vec::new();
    let mut grouped_coefficients = Vec::new();
    for (claim_index, (claim, &coefficient)) in claims.iter().zip(&coefficients).enumerate() {
        let key = claim.committed.key();
        if let Some(&group_index) = group_indices.get(&key) {
            let representative = representatives[group_index];
            if evaluations[claim_index] != evaluations[representative] {
                return Err(Section6FinalOpeningError::EvaluationMismatch(claim_index));
            }
            grouped_coefficients[group_index] += coefficient;
        } else {
            let group_index = representatives.len();
            group_indices.insert(key, group_index);
            representatives.push(claim_index);
            grouped_coefficients.push(coefficient);
        }
    }
    let polynomial_refs = representatives
        .iter()
        .map(|&index| &polynomials[index] as &dyn MultilinearPoly<Fr>)
        .collect::<Vec<_>>();
    let joint_polynomial = RlcSource::new(polynomial_refs, grouped_coefficients.clone());
    let mut hints = representatives
        .iter()
        .map(|&index| claims[index].committed.hint.as_ref().clone())
        .collect::<Vec<_>>();
    let mut hint_coefficients = grouped_coefficients.clone();
    let widest_native_rows = representatives
        .iter()
        .map(|&index| grid.native_shape(claims[index].committed.logical_len))
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .map(Section6DoryShape::row_count)
        .max()
        .ok_or(Section6FinalOpeningError::EmptyClaims)?;
    if widest_native_rows < grid.row_count() {
        let mut zero_partial = DoryScheme::begin(&parameters.pcs);
        DoryScheme::feed_zeros(
            &mut zero_partial,
            grid.row_width(),
            grid.row_count(),
            &parameters.pcs,
        );
        let (_, zero_hint) = DoryScheme::finish_zk(zero_partial, &parameters.pcs);
        hints.push(zero_hint);
        hint_coefficients.push(Fr::zero());
    }
    let combined_hint =
        <DoryScheme as AdditivelyHomomorphic>::combine_hints(hints, &hint_coefficients);
    let joint_evaluation = evaluations
        .iter()
        .zip(&coefficients)
        .map(|(&evaluation, &coefficient)| coefficient * evaluation)
        .sum::<Fr>();
    let grouped_evaluation = representatives
        .iter()
        .zip(&grouped_coefficients)
        .map(|(&index, &coefficient)| coefficient * evaluations[index])
        .sum::<Fr>();
    if joint_evaluation != grouped_evaluation {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let opening_point = Point::high_to_low(point.to_vec());
    let (proof, hiding_commitment, blind) = <DoryScheme as ZkOpeningScheme>::open_zk(
        &joint_polynomial,
        opening_point.as_slice(),
        joint_evaluation,
        &parameters.pcs,
        combined_hint,
        &mut transcript,
    )
    .map_err(|error| Section6FinalOpeningError::Dory(error.to_string()))?;
    ZkEvaluationClaim::new(opening_point.as_slice(), &hiding_commitment)
        .append_to_transcript(&mut transcript);
    Ok((
        HiddenFieldVectorDoryProof { opening: proof },
        hiding_commitment,
        blind,
        coefficients,
    ))
}

#[allow(clippy::too_many_arguments)]
fn verify_grid_batch_zk(
    role: &[u8],
    outer_binding: &[u8],
    parameters: &DoryVerifierParameters,
    grid: Section6DoryGrid,
    claims: &[Section6FinalOpeningVerifierClaim<'_>],
    point: &[Fr],
    terminal_rows: &[Bn254G1],
    proof: &HiddenFieldVectorDoryProof,
) -> Result<(Bn254G1, Vec<Fr>), Section6FinalOpeningError> {
    let commitments = claims
        .iter()
        .map(|claim| claim.commitment.clone())
        .collect::<Vec<_>>();
    let logical_lengths = claims
        .iter()
        .map(|claim| claim.logical_len)
        .collect::<Vec<_>>();
    let mut coefficient_transcript = grid_dory_transcript(
        role,
        outer_binding,
        grid,
        point,
        &commitments,
        &logical_lengths,
        terminal_rows,
    );
    let coefficients = coefficient_transcript.challenge_scalar_powers(claims.len());
    let mut transcript = grid_dory_transcript(
        role,
        outer_binding,
        grid,
        point,
        &commitments,
        &logical_lengths,
        terminal_rows,
    );
    let hiding_commitment = HomomorphicBatch::<DoryScheme>::verify_batch_zk(
        &parameters.pcs,
        Point::high_to_low(point.to_vec()),
        commitments,
        &proof.opening,
        &mut transcript,
    )
    .map_err(|error| Section6FinalOpeningError::Dory(error.to_string()))?;
    Ok((hiding_commitment, coefficients))
}

fn final_opening_transcript(
    role: &[u8],
    outer_binding: &[u8],
    grid: Section6DoryGrid,
    claims: &[Section6FinalOpeningVerifierClaim<'_>],
) -> Blake2bTranscript<Fr> {
    let mut transcript = Blake2bTranscript::<Fr>::new(b"pc-section6-final-opening-v1");
    transcript.append(&Label(b"section6_final_opening"));
    transcript.append(&LabelWithCount(b"role", role.len() as u64));
    transcript.append_bytes(role);
    transcript.append(&LabelWithCount(
        b"outer_binding",
        outer_binding.len() as u64,
    ));
    transcript.append_bytes(outer_binding);
    transcript.append_labeled(b"grid_num_vars", &U64Word(grid.num_vars as u64));
    transcript.append(&LabelWithCount(b"claims", claims.len() as u64));
    for claim in claims {
        transcript.append_labeled(b"source_commitment", claim.commitment);
        transcript.append_labeled(b"source_logical_len", &U64Word(claim.logical_len as u64));
        let point = claim
            .point_msb
            .iter()
            .copied()
            .map(Fr::from)
            .collect::<Vec<_>>();
        transcript.append_values(b"source_point", &point);
        transcript.append_labeled(
            b"source_evaluation_commitment",
            &claim.evaluation_commitment,
        );
    }
    transcript
}

fn grid_dory_transcript(
    role: &[u8],
    outer_binding: &[u8],
    grid: Section6DoryGrid,
    point: &[Fr],
    commitments: &[DoryCommitment],
    logical_lengths: &[usize],
    terminal_rows: &[Bn254G1],
) -> Blake2bTranscript<Fr> {
    let mut transcript = Blake2bTranscript::<Fr>::new(b"parcomm-section6-grid-dory-v1");
    transcript.append(&Label(b"section6_grid_dory"));
    transcript.append(&LabelWithCount(b"role", role.len() as u64));
    transcript.append_bytes(role);
    transcript.append(&LabelWithCount(
        b"outer_binding",
        outer_binding.len() as u64,
    ));
    transcript.append_bytes(outer_binding);
    transcript.append_labeled(b"grid_num_vars", &U64Word(grid.num_vars as u64));
    transcript.append_values(b"grid_point", point);
    transcript.append(&LabelWithCount(b"sources", commitments.len() as u64));
    for (commitment, &logical_len) in commitments.iter().zip(logical_lengths) {
        transcript.append_labeled(b"source_commitment", commitment);
        transcript.append_labeled(b"source_logical_len", &U64Word(logical_len as u64));
    }
    transcript.append(&LabelWithCount(
        b"terminal_claim_rows",
        terminal_rows.len() as u64,
    ));
    for commitment in terminal_rows {
        transcript.append_labeled(b"terminal_claim_row", commitment);
    }
    transcript
}

fn bind_claim_link_commitments(
    transcript: &mut Blake2bTranscript<Fr>,
    commitments: &[Bn254G1],
) -> Vec<Fr> {
    transcript.append(&LabelWithCount(
        b"claim_link_commitments",
        commitments.len() as u64,
    ));
    for commitment in commitments {
        transcript.append_labeled(b"claim_link_commitment", commitment);
    }
    transcript.challenge_scalar_powers(commitments.len())
}

fn bind_dory_completion(
    transcript: &mut Blake2bTranscript<Fr>,
    hiding_commitment: &Bn254G1,
    proof: &HiddenFieldVectorDoryProof,
) -> Result<(), Section6FinalOpeningError> {
    transcript.append_labeled(b"dory_hiding_commitment", hiding_commitment);
    let bytes = bincode::serialize(proof)
        .map_err(|error| Section6FinalOpeningError::Encoding(error.to_string()))?;
    transcript.append(&LabelWithCount(b"dory_proof", bytes.len() as u64));
    transcript.append_bytes(&bytes);
    Ok(())
}

fn bind_blindfold_coefficients(
    transcript: &mut Blake2bTranscript<Fr>,
    claim_coefficients: &[Fr],
    terminal_sumcheck_coefficients: &[Fr],
    claimed_sumcheck_coefficients: &[Fr],
    claim_link_coefficients: &[Fr],
    terminal_dory_coefficients: &[Fr],
) {
    transcript.append_values(b"claim_coefficients", claim_coefficients);
    transcript.append_values(b"terminal_sumcheck_coeffs", terminal_sumcheck_coefficients);
    transcript.append_values(b"claimed_sumcheck_coeffs", claimed_sumcheck_coefficients);
    transcript.append_values(b"claim_link_coefficients", claim_link_coefficients);
    transcript.append_values(b"terminal_dory_coeffs", terminal_dory_coefficients);
}

fn replay_consistency(
    role: &[u8],
    outer_binding: &[u8],
    grid: Section6DoryGrid,
    claims: &[Section6FinalOpeningVerifierClaim<'_>],
    sumcheck: &CommittedSumcheckProof<Bn254G1>,
    expected_claim_coefficients: &[Fr],
    expected_link_coefficients: &[Fr],
) -> Result<jolt_sumcheck::CommittedSumcheckConsistency<Fr, Bn254G1>, Section6FinalOpeningError> {
    let mut transcript = final_opening_transcript(role, outer_binding, grid, claims);
    let coefficients = transcript.challenge_scalar_powers(claims.len());
    if coefficients != expected_claim_coefficients {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let consistency = SumcheckProof::<Fr, Bn254G1>::Committed(sumcheck.clone())
        .verify_committed_consistency(
            SumcheckStatement::new(grid.num_vars, SUMCHECK_DEGREE),
            &mut transcript,
        )
        .map_err(|error| Section6FinalOpeningError::Sumcheck(error.to_string()))?;
    let external_commitments = claims
        .iter()
        .map(|claim| claim.evaluation_commitment)
        .collect::<Vec<_>>();
    let link_coefficients = bind_claim_link_commitments(&mut transcript, &external_commitments);
    if link_coefficients != expected_link_coefficients {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    Ok(consistency)
}

#[allow(clippy::too_many_arguments)]
fn build_blindfold_protocol(
    row_len: usize,
    num_vars: usize,
    claim_count: usize,
    consistency: jolt_sumcheck::CommittedSumcheckConsistency<Fr, Bn254G1>,
    sumcheck: &CommittedSumcheckProof<Bn254G1>,
    claim_coefficients: &[Fr],
    terminal_sumcheck_coefficients: &[Fr],
    claimed_sumcheck_coefficients: &[Fr],
    claim_link_coefficients: &[Fr],
    claim_link_commitment: Bn254G1,
    terminal_dory_coefficients: &[Fr],
    dory_hiding_commitment: Bn254G1,
    final_opening_mix: Fr,
) -> Result<BlindFoldProtocol<Fr, Bn254G1>, Section6FinalOpeningError> {
    if [
        claim_coefficients.len(),
        terminal_sumcheck_coefficients.len(),
        claimed_sumcheck_coefficients.len(),
        claim_link_coefficients.len(),
        terminal_dory_coefficients.len(),
    ]
    .iter()
    .any(|&length| length != claim_count)
    {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let terminal_ids = (0..claim_count).collect::<Vec<_>>();
    let claimed_ids = (claim_count..2 * claim_count).collect::<Vec<_>>();
    let opening_ids = terminal_ids
        .iter()
        .chain(&claimed_ids)
        .copied()
        .collect::<Vec<_>>();
    let input_claim = constant(Fr::zero());
    let output_claim = linear_opening_expression(&terminal_ids, terminal_sumcheck_coefficients)
        + linear_opening_expression(&claimed_ids, claimed_sumcheck_coefficients);
    let final_opening_coefficients = terminal_dory_coefficients
        .iter()
        .map(|&coefficient| final_opening_mix * coefficient)
        .chain(claim_link_coefficients.iter().copied())
        .collect::<Vec<_>>();
    let final_opening_commitment = Bn254G1::linear_combine(
        &claim_link_commitment,
        &dory_hiding_commitment,
        &final_opening_mix,
    );
    BlindFoldProtocol::<Fr, Bn254G1>::builder::<usize, (), usize>()
        .stage("section6-arbitrary-point-reduction")
        .sumcheck(SumcheckStatement::new(num_vars, SUMCHECK_DEGREE))
        .domain(SumcheckDomainSpec::BooleanHypercube)
        .consistency(consistency)
        .output_claim_rows(opening_ids.clone(), row_len, sumcheck.output_claims.clone())
        .input_claim(input_claim)
        .output_claim(output_claim)
        .finish_stage()
        .map_err(|error| Section6FinalOpeningError::BlindFold(error.to_string()))?
        .final_opening(
            opening_ids,
            final_opening_coefficients,
            final_opening_commitment,
        )
        .build()
        .map_err(|error| Section6FinalOpeningError::BlindFold(error.to_string()))
}

fn linear_opening_expression(ids: &[usize], coefficients: &[Fr]) -> Expr<Fr, usize> {
    ids.iter()
        .zip(coefficients)
        .fold(constant(Fr::zero()), |expression, (&id, &coefficient)| {
            expression + constant(coefficient) * opening(id)
        })
}

fn combine_scalars(values: &[Fr], coefficients: &[Fr]) -> Fr {
    values
        .iter()
        .zip(coefficients)
        .map(|(&value, &coefficient)| value * coefficient)
        .sum()
}

fn combine_commitments(commitments: &[Bn254G1], coefficients: &[Fr]) -> Bn254G1 {
    commitments.iter().zip(coefficients).fold(
        Bn254G1::default(),
        |accumulator, (commitment, coefficient)| {
            Bn254G1::linear_combine(&accumulator, commitment, coefficient)
        },
    )
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr as ArkFr;
    use ark_ff::Zero;
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use jolt_field::Fr as JoltFr;

    use crate::{
        hidden_relation::HiddenScalarWitness,
        jolt_dory::{evaluate_field_vector_standard, setup_dory_parameters},
    };

    use super::{
        commit_section6_field_on_grid_zk, prove_section6_final_opening,
        verify_section6_final_opening, ReductionStateSet, Section6DoryGrid,
        Section6FinalOpeningData, Section6FinalOpeningProverClaim,
        Section6FinalOpeningVerifierClaim,
    };

    fn ark(value: u64) -> ArkFr {
        ArkFr::from(value)
    }

    #[test]
    fn native_balanced_shapes_convert_points_on_both_dory_axes() {
        let grid = Section6DoryGrid::new(6).unwrap();

        let odd = grid.native_shape(8).unwrap();
        assert_eq!((odd.row_count(), odd.row_width()), (2, 4));
        assert_eq!(
            odd.embed_native_point(&[ark(17), ark(19), ark(23)])
                .unwrap(),
            vec![ark(0), ark(0), ark(17), ark(0), ark(19), ark(23)]
        );
        let (odd_point, odd_scale) = odd
            .project_global_point(&[ark(2), ark(3), ark(5), ark(7), ark(11), ark(13)])
            .unwrap();
        assert_eq!(odd_point, vec![ark(5), ark(11), ark(13)]);
        assert_eq!(odd_scale, -ark(12));

        let even = grid.native_shape(16).unwrap();
        assert_eq!((even.row_count(), even.row_width()), (4, 4));
        assert_eq!(
            even.embed_native_point(&[ark(17), ark(19), ark(23), ark(29)])
                .unwrap(),
            vec![ark(0), ark(17), ark(19), ark(0), ark(23), ark(29)]
        );
        let (even_point, even_scale) = even
            .project_global_point(&[ark(2), ark(3), ark(5), ark(7), ark(11), ark(13)])
            .unwrap();
        assert_eq!(even_point, vec![ark(3), ark(5), ark(11), ark(13)]);
        assert_eq!(even_scale, ark(6));
    }

    #[test]
    fn repeated_claims_share_one_large_reduction_source() {
        let grid = Section6DoryGrid::new(4).unwrap();
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(grid.num_vars());
        let values = (0..8).map(|index| ark(index + 1)).collect::<Vec<_>>();
        let committed =
            commit_section6_field_on_grid_zk(&prover_parameters, &grid, &values).unwrap();
        let points = [vec![ark(2), ark(3), ark(5)], vec![ark(7), ark(11), ark(13)]];
        let evaluations = points
            .iter()
            .map(|point| evaluate_field_vector_standard(&values, point).unwrap())
            .collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(0x5ec7_1010);
        let witnesses = evaluations
            .iter()
            .map(|&evaluation| {
                HiddenScalarWitness::commit(&prover_parameters, evaluation, &mut rng)
            })
            .collect::<Vec<_>>();
        let claims = points
            .iter()
            .zip(&witnesses)
            .map(|(point, witness)| Section6FinalOpeningProverClaim {
                data: Section6FinalOpeningData::Field(&values),
                committed: &committed,
                point_msb: point,
                evaluation: witness,
            })
            .collect::<Vec<_>>();

        let coefficients = [JoltFr::from(ark(1)), JoltFr::from(ark(2))];
        let states = ReductionStateSet::new(&claims, &coefficients, grid).unwrap();
        assert_eq!(states.sources.len(), 1);
        assert_eq!(states.claims.len(), 2);
        assert_eq!(
            states
                .sources
                .iter()
                .map(|source| source.eq.len())
                .sum::<usize>(),
            values.len()
        );
        drop(states);

        let proof = prove_section6_final_opening(
            b"section6-repeated-source-test",
            b"outer-binding",
            &prover_parameters,
            &grid,
            &claims,
            &mut rng,
        )
        .unwrap();
        assert_eq!(proof.physical_dory_proof_count(), 1);
        let verifier_claims = points
            .iter()
            .zip(&witnesses)
            .map(|(point, witness)| Section6FinalOpeningVerifierClaim {
                commitment: &committed.commitment,
                logical_len: committed.logical_len,
                point_msb: point,
                evaluation_commitment: witness.commitment,
            })
            .collect::<Vec<_>>();
        verify_section6_final_opening(
            b"section6-repeated-source-test",
            b"outer-binding",
            &verifier_parameters,
            &grid,
            &verifier_claims,
            &proof,
        )
        .unwrap();
    }

    #[test]
    fn heterogeneous_points_and_lengths_use_one_physical_dory_proof() {
        let grid = Section6DoryGrid::new(4).unwrap();
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(grid.num_vars());
        let field_values = vec![
            ark(3),
            ark(1),
            ark(4),
            ark(1),
            ark(5),
            ark(9),
            ark(2),
            ark(6),
        ];
        let short_values = vec![ark(2), -ark(1), ark(3), ark(0)];
        let field_commitment =
            commit_section6_field_on_grid_zk(&prover_parameters, &grid, &field_values).unwrap();
        let short_commitment =
            commit_section6_field_on_grid_zk(&prover_parameters, &grid, &short_values).unwrap();
        let field_point = vec![ark(7), ark(11), ark(13)];
        let short_point = vec![ark(17), ark(19)];
        let field_eval = evaluate_field_vector_standard(&field_values, &field_point).unwrap();
        let short_eval = evaluate_field_vector_standard(&short_values, &short_point).unwrap();
        let mut rng = StdRng::seed_from_u64(0x5ec7_1006);
        let field_eval_witness =
            HiddenScalarWitness::commit(&prover_parameters, field_eval, &mut rng);
        let short_eval_witness =
            HiddenScalarWitness::commit(&prover_parameters, short_eval, &mut rng);
        let prover_claims = vec![
            Section6FinalOpeningProverClaim {
                data: Section6FinalOpeningData::Field(&field_values),
                committed: &field_commitment,
                point_msb: &field_point,
                evaluation: &field_eval_witness,
            },
            Section6FinalOpeningProverClaim {
                data: Section6FinalOpeningData::Field(&short_values),
                committed: &short_commitment,
                point_msb: &short_point,
                evaluation: &short_eval_witness,
            },
        ];
        let proof = prove_section6_final_opening(
            b"section6-test",
            b"outer-binding",
            &prover_parameters,
            &grid,
            &prover_claims,
            &mut rng,
        )
        .unwrap();
        assert_eq!(proof.physical_dory_proof_count(), 1);

        let verifier_claims = vec![
            Section6FinalOpeningVerifierClaim {
                commitment: &field_commitment.commitment,
                logical_len: field_commitment.logical_len,
                point_msb: &field_point,
                evaluation_commitment: field_eval_witness.commitment,
            },
            Section6FinalOpeningVerifierClaim {
                commitment: &short_commitment.commitment,
                logical_len: short_commitment.logical_len,
                point_msb: &short_point,
                evaluation_commitment: short_eval_witness.commitment,
            },
        ];
        verify_section6_final_opening(
            b"section6-test",
            b"outer-binding",
            &verifier_parameters,
            &grid,
            &verifier_claims,
            &proof,
        )
        .unwrap();

        let bytes = bincode::serialize(&proof).unwrap();
        let decoded = bincode::deserialize(&bytes).unwrap();
        assert_eq!(proof, decoded);
    }

    #[test]
    fn tampered_point_evaluation_or_source_commitment_is_rejected() {
        let grid = Section6DoryGrid::new(4).unwrap();
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(grid.num_vars());
        let values = vec![
            ark(8),
            ark(5),
            ark(9),
            ark(7),
            ark(9),
            ark(3),
            ark(2),
            ark(3),
        ];
        let committed =
            commit_section6_field_on_grid_zk(&prover_parameters, &grid, &values).unwrap();
        let point = vec![ark(5), ark(8), ark(9)];
        let evaluation = evaluate_field_vector_standard(&values, &point).unwrap();
        let mut rng = StdRng::seed_from_u64(0x5ec7_1007);
        let evaluation_witness =
            HiddenScalarWitness::commit(&prover_parameters, evaluation, &mut rng);
        let proof = prove_section6_final_opening(
            b"section6-test",
            b"outer-binding",
            &prover_parameters,
            &grid,
            &[Section6FinalOpeningProverClaim {
                data: Section6FinalOpeningData::Field(&values),
                committed: &committed,
                point_msb: &point,
                evaluation: &evaluation_witness,
            }],
            &mut rng,
        )
        .unwrap();

        let valid_claim = Section6FinalOpeningVerifierClaim {
            commitment: &committed.commitment,
            logical_len: committed.logical_len,
            point_msb: &point,
            evaluation_commitment: evaluation_witness.commitment,
        };
        verify_section6_final_opening(
            b"section6-test",
            b"outer-binding",
            &verifier_parameters,
            &grid,
            &[valid_claim],
            &proof,
        )
        .unwrap();

        let tampered_point = vec![ark(6), point[1], point[2]];
        let point_claim = Section6FinalOpeningVerifierClaim {
            commitment: &committed.commitment,
            logical_len: committed.logical_len,
            point_msb: &tampered_point,
            evaluation_commitment: evaluation_witness.commitment,
        };
        assert!(verify_section6_final_opening(
            b"section6-test",
            b"outer-binding",
            &verifier_parameters,
            &grid,
            &[point_claim],
            &proof,
        )
        .is_err());

        let bad_commitment =
            HiddenScalarWitness::commit(&prover_parameters, evaluation + ark(1), &mut rng)
                .commitment;
        let commitment_claim = Section6FinalOpeningVerifierClaim {
            commitment: &committed.commitment,
            logical_len: committed.logical_len,
            point_msb: &point,
            evaluation_commitment: bad_commitment,
        };
        assert!(verify_section6_final_opening(
            b"section6-test",
            b"outer-binding",
            &verifier_parameters,
            &grid,
            &[commitment_claim],
            &proof,
        )
        .is_err());

        let mut other_values = values.clone();
        other_values[0] += ark(1);
        let other_committed =
            commit_section6_field_on_grid_zk(&prover_parameters, &grid, &other_values).unwrap();
        let source_commitment_claim = Section6FinalOpeningVerifierClaim {
            commitment: &other_committed.commitment,
            logical_len: committed.logical_len,
            point_msb: &point,
            evaluation_commitment: evaluation_witness.commitment,
        };
        assert!(verify_section6_final_opening(
            b"section6-test",
            b"outer-binding",
            &verifier_parameters,
            &grid,
            &[source_commitment_claim],
            &proof,
        )
        .is_err());

        assert!(!evaluation.is_zero());
    }
}
