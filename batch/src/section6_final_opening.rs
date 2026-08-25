//! One terminal Dory opening for heterogeneous Section 6 MLE claims.

use std::{collections::HashMap, fmt::Display, sync::Arc, time::Instant};

use ark_bn254::Fr as ArkFr;
use ark_ff::{One, Zero};
use ark_std::rand::RngCore;
use jolt_blindfold::{
    prove as prove_blindfold, BlindFoldProof, BlindFoldProtocol, BlindFoldWitness,
};
use jolt_claims::{constant, opening, Expr};
use jolt_crypto::{Bn254G1, HomomorphicCommitment, JoltGroup, Pedersen, VectorCommitment};
use jolt_dory::{DoryCommitment, DoryHint, DoryScheme};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_openings::{
    AdditivelyHomomorphic, StreamingCommitment, ZkEvaluationClaim, ZkOpeningScheme,
};
use jolt_poly::{eq_index_msb, EqPolynomial, MultilinearPoly, Point, RlcSource, UnivariatePoly};
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
        commitment_binding_bytes, dense_i16_polynomial, dense_i8_polynomial, CommittedDoryBva,
        CommittedDoryFieldMatrix, CommittedDoryFieldVector, DenseI16BvaSource, DenseI8BvaSource,
        DoryCommitProfile, DoryProverParameters, DoryVerifierParameters,
        HiddenFieldVectorDoryProof,
    },
    jolt_dory_adapter::{commit_dense_i16_zk, commit_dense_i8_zk, DenseI8DoryError},
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
    WholeMatrix {
        values: &'a [ArkFr],
        padded_rows: usize,
        padded_cols: usize,
    },
    MatrixColumn {
        values: &'a [ArkFr],
        padded_rows: usize,
        padded_cols: usize,
        column: usize,
    },
}

impl Section6FinalOpeningData<'_> {
    fn physical_len(self) -> usize {
        match self {
            Self::Field(values) => values.len(),
            Self::WholeMatrix { values, .. } => values.len(),
            Self::MatrixColumn { values, .. } => values.len(),
        }
    }

    fn value(self, index: usize) -> Fr {
        match self {
            Self::Field(values) => Fr::from(values[index]),
            Self::WholeMatrix { values, .. } => Fr::from(values[index]),
            Self::MatrixColumn { values, .. } => Fr::from(values[index]),
        }
    }
}

/// Functional applied to one committed source before the terminal opening.
///
/// Point claims use the ordinary multilinear equality polynomial at
/// `point_msb`. Sparse matrix claims retain the same tensor point but restrict
/// its column equality factor to the listed logical columns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Section6FinalOpeningFunctional<'a> {
    Point,
    SparseMatrixColumns {
        padded_rows: usize,
        logical_cols: usize,
        padded_cols: usize,
        selected_columns: &'a [usize],
    },
}

/// A Pedersen commitment used by the terminal relation link.
///
/// Matrix-column reductions retain their original column commitments until
/// the final batching coefficients are known. This lets the verifier evaluate
/// all terminal commitment expressions with one global MSM.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Section6EvaluationCommitment<'a> {
    Direct(Bn254G1),
    MatrixColumns {
        logical_cols: usize,
        padded_cols: usize,
        selected_columns: &'a [usize],
        commitments: &'a [Bn254G1],
    },
}

/// One hidden arbitrary-point MLE claim held by the prover.
#[derive(Clone, Copy)]
pub struct Section6FinalOpeningProverClaim<'a> {
    pub data: Section6FinalOpeningData<'a>,
    pub committed: &'a CommittedSection6DorySource,
    pub point_msb: &'a [ArkFr],
    pub functional: Section6FinalOpeningFunctional<'a>,
    pub evaluation: &'a HiddenScalarWitness,
    pub evaluation_commitment: Section6EvaluationCommitment<'a>,
}

/// Commitment-only view of one arbitrary-point MLE claim.
#[derive(Clone, Copy)]
pub struct Section6FinalOpeningVerifierClaim<'a> {
    pub commitment: &'a DoryCommitment,
    pub logical_len: usize,
    pub point_msb: &'a [ArkFr],
    pub functional: Section6FinalOpeningFunctional<'a>,
    pub evaluation_commitment: Section6EvaluationCommitment<'a>,
}

enum ResolvedFieldData<'a> {
    Vector(&'a [ArkFr]),
    WholeMatrix(Arc<CommittedDoryFieldMatrix>),
    Matrix {
        matrix: Arc<CommittedDoryFieldMatrix>,
        column: usize,
    },
}

enum ResolvedFieldFunctional {
    Point,
    SparseMatrixColumns {
        padded_rows: usize,
        logical_cols: usize,
        padded_cols: usize,
        selected_columns: Vec<usize>,
    },
}

impl ResolvedFieldFunctional {
    fn as_functional(&self) -> Section6FinalOpeningFunctional<'_> {
        match self {
            Self::Point => Section6FinalOpeningFunctional::Point,
            Self::SparseMatrixColumns {
                padded_rows,
                logical_cols,
                padded_cols,
                selected_columns,
            } => Section6FinalOpeningFunctional::SparseMatrixColumns {
                padded_rows: *padded_rows,
                logical_cols: *logical_cols,
                padded_cols: *padded_cols,
                selected_columns,
            },
        }
    }
}

/// Owned adapter from a relation operand to one global-opening prover claim.
pub struct ResolvedSection6FieldProverClaim<'a> {
    source: CommittedSection6DorySource,
    data: ResolvedFieldData<'a>,
    point_msb: Vec<ArkFr>,
    functional: ResolvedFieldFunctional,
}

impl ResolvedSection6FieldProverClaim<'_> {
    pub fn source_key(&self) -> Section6DorySourceKey {
        self.source.key()
    }

    pub fn as_claim<'a>(
        &'a self,
        evaluation: &'a HiddenScalarWitness,
    ) -> Section6FinalOpeningProverClaim<'a> {
        self.as_claim_with_evaluation_commitment(
            evaluation,
            Section6EvaluationCommitment::Direct(evaluation.commitment),
        )
    }

    pub fn as_claim_with_evaluation_commitment<'a>(
        &'a self,
        evaluation: &'a HiddenScalarWitness,
        evaluation_commitment: Section6EvaluationCommitment<'a>,
    ) -> Section6FinalOpeningProverClaim<'a> {
        let values = match &self.data {
            ResolvedFieldData::Vector(values) => Section6FinalOpeningData::Field(values),
            ResolvedFieldData::WholeMatrix(matrix) => Section6FinalOpeningData::WholeMatrix {
                values: matrix.values.as_slice(),
                padded_rows: matrix.padded_rows,
                padded_cols: matrix.padded_cols,
            },
            ResolvedFieldData::Matrix { matrix, column } => {
                Section6FinalOpeningData::MatrixColumn {
                    values: matrix.values.as_slice(),
                    padded_rows: matrix.padded_rows,
                    padded_cols: matrix.padded_cols,
                    column: *column,
                }
            }
        };
        Section6FinalOpeningProverClaim {
            data: values,
            committed: &self.source,
            point_msb: &self.point_msb,
            functional: self.functional.as_functional(),
            evaluation,
            evaluation_commitment,
        }
    }
}

/// Owned adapter from a relation descriptor to one commitment-only claim.
pub struct ResolvedSection6FieldVerifierClaim<'a> {
    commitment: &'a DoryCommitment,
    logical_len: usize,
    point_msb: Vec<ArkFr>,
    functional: ResolvedFieldFunctional,
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
        self.as_claim_with_evaluation_commitment(Section6EvaluationCommitment::Direct(
            self.evaluation_commitment,
        ))
    }

    pub fn as_claim_with_evaluation_commitment<'a>(
        &'a self,
        evaluation_commitment: Section6EvaluationCommitment<'a>,
    ) -> Section6FinalOpeningVerifierClaim<'a> {
        Section6FinalOpeningVerifierClaim {
            commitment: self.commitment,
            logical_len: self.logical_len,
            point_msb: &self.point_msb,
            functional: self.functional.as_functional(),
            evaluation_commitment,
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
        (
            ResolvedFieldData::Matrix {
                matrix: matrix.clone(),
                column: table.committed.matrix_column,
            },
            point,
        )
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
        functional: ResolvedFieldFunctional::Point,
    })
}

/// Resolve one complete matrix source at a tensor column challenge followed
/// by the relation's terminal row point.
pub fn resolve_section6_hidden_relation_whole_matrix_prover_claim<'a>(
    table: HiddenRelationTable<'a>,
    full_point: &[Fr],
    grid: &Section6DoryGrid,
) -> Result<ResolvedSection6FieldProverClaim<'a>, Section6FinalOpeningError> {
    let matrix = table
        .committed
        .matrix
        .as_ref()
        .ok_or(Section6FinalOpeningError::ProofShape)?;
    let logical_len = matrix
        .padded_rows
        .checked_mul(matrix.padded_cols)
        .ok_or(Section6FinalOpeningError::InvalidLogicalLength(usize::MAX))?;
    if full_point.len() != logical_len.trailing_zeros() as usize {
        return Err(Section6FinalOpeningError::PointLength {
            expected: logical_len.trailing_zeros() as usize,
            actual: full_point.len(),
        });
    }
    let source = CommittedSection6DorySource::from_field_view(table.committed, grid)?;
    if source.logical_len != logical_len {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    Ok(ResolvedSection6FieldProverClaim {
        source,
        data: ResolvedFieldData::WholeMatrix(matrix.clone()),
        point_msb: full_point.iter().copied().map(ArkFr::from).collect(),
        functional: ResolvedFieldFunctional::Point,
    })
}

/// Commitment-only counterpart of the whole-matrix relation claim resolver.
pub fn resolve_section6_hidden_relation_whole_matrix_verifier_claim<'a>(
    table: HiddenRelationTableStatement<'a>,
    full_point: &[Fr],
    evaluation_commitment: Bn254G1,
    grid: &Section6DoryGrid,
) -> Result<ResolvedSection6FieldVerifierClaim<'a>, Section6FinalOpeningError> {
    let matrix = table.matrix.ok_or(Section6FinalOpeningError::ProofShape)?;
    let logical_len = table
        .logical_len
        .checked_mul(matrix.padded_cols)
        .ok_or(Section6FinalOpeningError::InvalidLogicalLength(usize::MAX))?;
    if full_point.len() != logical_len.trailing_zeros() as usize {
        return Err(Section6FinalOpeningError::PointLength {
            expected: logical_len.trailing_zeros() as usize,
            actual: full_point.len(),
        });
    }
    validate_native_len(logical_len, *grid)?;
    Ok(ResolvedSection6FieldVerifierClaim {
        commitment: table.commitment,
        logical_len,
        point_msb: full_point.iter().copied().map(ArkFr::from).collect(),
        functional: ResolvedFieldFunctional::Point,
        evaluation_commitment,
        grid_num_vars: grid.num_vars,
    })
}

/// Resolve a matrix relation opening whose terminal functional includes only
/// the selected logical columns. The selection is statement data; no missing
/// column evaluation is added to the relation proof.
pub fn resolve_section6_hidden_relation_sparse_matrix_prover_claim<'a>(
    table: HiddenRelationTable<'a>,
    full_point: &[Fr],
    selected_columns: &[usize],
    grid: &Section6DoryGrid,
) -> Result<ResolvedSection6FieldProverClaim<'a>, Section6FinalOpeningError> {
    let matrix = table
        .committed
        .matrix
        .as_ref()
        .ok_or(Section6FinalOpeningError::ProofShape)?;
    let logical_len = matrix
        .padded_rows
        .checked_mul(matrix.padded_cols)
        .ok_or(Section6FinalOpeningError::InvalidLogicalLength(usize::MAX))?;
    validate_sparse_matrix_functional(
        logical_len,
        full_point,
        matrix.padded_rows,
        matrix.logical_cols,
        matrix.padded_cols,
        selected_columns,
    )?;
    let source = CommittedSection6DorySource::from_field_view(table.committed, grid)?;
    if source.logical_len != logical_len {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    Ok(ResolvedSection6FieldProverClaim {
        source,
        data: ResolvedFieldData::WholeMatrix(matrix.clone()),
        point_msb: full_point.iter().copied().map(ArkFr::from).collect(),
        functional: ResolvedFieldFunctional::SparseMatrixColumns {
            padded_rows: matrix.padded_rows,
            logical_cols: matrix.logical_cols,
            padded_cols: matrix.padded_cols,
            selected_columns: selected_columns.to_vec(),
        },
    })
}

/// Commitment-only counterpart of the sparse matrix relation resolver.
pub fn resolve_section6_hidden_relation_sparse_matrix_verifier_claim<'a>(
    table: HiddenRelationTableStatement<'a>,
    full_point: &[Fr],
    selected_columns: &[usize],
    evaluation_commitment: Bn254G1,
    grid: &Section6DoryGrid,
) -> Result<ResolvedSection6FieldVerifierClaim<'a>, Section6FinalOpeningError> {
    let matrix = table.matrix.ok_or(Section6FinalOpeningError::ProofShape)?;
    let logical_len = table
        .logical_len
        .checked_mul(matrix.padded_cols)
        .ok_or(Section6FinalOpeningError::InvalidLogicalLength(usize::MAX))?;
    validate_sparse_matrix_functional(
        logical_len,
        full_point,
        table.logical_len,
        matrix.logical_cols,
        matrix.padded_cols,
        selected_columns,
    )?;
    validate_native_len(logical_len, *grid)?;
    Ok(ResolvedSection6FieldVerifierClaim {
        commitment: table.commitment,
        logical_len,
        point_msb: full_point.iter().copied().map(ArkFr::from).collect(),
        functional: ResolvedFieldFunctional::SparseMatrixColumns {
            padded_rows: table.logical_len,
            logical_cols: matrix.logical_cols,
            padded_cols: matrix.padded_cols,
            selected_columns: selected_columns.to_vec(),
        },
        evaluation_commitment,
        grid_num_vars: grid.num_vars,
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
        functional: ResolvedFieldFunctional::Point,
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

/// Mutually exclusive prover timings for the terminal Section 6 opening.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Section6FinalOpeningProverProfile {
    pub preparation_state_setup_seconds: f64,
    pub sumcheck_seconds: f64,
    pub dory_batch_open_seconds: f64,
    pub blindfold_preparation_seconds: f64,
    pub blindfold_prove_seconds: f64,
    pub result_assembly_seconds: f64,
    pub total_seconds: f64,
}

impl Section6FinalOpeningProverProfile {
    pub fn classified_seconds(&self) -> f64 {
        self.preparation_state_setup_seconds
            + self.sumcheck_seconds
            + self.dory_batch_open_seconds
            + self.blindfold_preparation_seconds
            + self.blindfold_prove_seconds
            + self.result_assembly_seconds
    }
}

/// Mutually exclusive verifier timings for the terminal Section 6 opening.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Section6FinalOpeningVerifierProfile {
    pub preparation_state_setup_seconds: f64,
    pub sumcheck_verify_seconds: f64,
    pub dory_batch_verify_seconds: f64,
    pub blindfold_preparation_seconds: f64,
    pub blindfold_verify_seconds: f64,
    pub result_finalization_seconds: f64,
    pub total_seconds: f64,
}

impl Section6FinalOpeningVerifierProfile {
    pub fn classified_seconds(&self) -> f64 {
        self.preparation_state_setup_seconds
            + self.sumcheck_verify_seconds
            + self.dory_batch_verify_seconds
            + self.blindfold_preparation_seconds
            + self.blindfold_verify_seconds
            + self.result_finalization_seconds
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

/// Commit signed bytes through the shared biased-u8 path, without materializing
/// a field-vector copy or trailing zero rows.
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
        DoryScheme::feed_i8_rows_with(
            &mut partial,
            |index| values.get(index).copied().unwrap_or(0),
            primitive_count,
            row_width,
            &parameters.pcs,
        );
    }
    let (commitment, hint) =
        DoryScheme::finish_zk_with_implicit_zero_rows(partial, shape.row_count(), &parameters.pcs);
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
    commit_section6_owned_field_matrix_on_grid_zk(
        parameters,
        grid,
        values,
        padded_rows,
        columns.len(),
        padded_cols,
    )
}

/// Commit an already column-major padded matrix by moving its buffer, avoiding
/// a second full-size allocation when the prover built the cover in place.
pub fn commit_section6_owned_field_matrix_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    values: Vec<ArkFr>,
    padded_rows: usize,
    logical_cols: usize,
    padded_cols: usize,
) -> Result<Vec<CommittedDoryFieldVector>, Section6FinalOpeningError> {
    if logical_cols == 0
        || padded_cols < logical_cols
        || !padded_cols.is_power_of_two()
        || padded_rows == 0
        || !padded_rows.is_power_of_two()
    {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let matrix_len = padded_rows
        .checked_mul(padded_cols)
        .ok_or(Section6FinalOpeningError::InvalidLogicalLength(usize::MAX))?;
    if values.len() != matrix_len {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let source = commit_section6_field_on_grid_zk(parameters, grid, &values)?;
    let matrix = Arc::new(CommittedDoryFieldMatrix {
        values,
        padded_rows,
        logical_cols,
        padded_cols,
        commitment: source.commitment.clone(),
        hint: source.hint.clone(),
        grid_num_vars: Some(grid.num_vars),
    });
    Ok((0..logical_cols)
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
    if bit_width < 8
        && values
            .par_iter()
            .any(|value| usize::from(value.unsigned_abs()) >= (1_usize << bit_width))
    {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let dense = dense_i8_polynomial(source)
        .map_err(|error| Section6FinalOpeningError::Dory(error.to_string()))?;
    let (commitment, hint) = commit_dense_i8_zk(&dense, &parameters.pcs);
    let shape = grid.native_shape(logical_len)?;
    let row_width = shape.row_width();
    Ok(CommittedDoryBva {
        commitment,
        hint,
        profile: DoryCommitProfile {
            physical_entries: values.len(),
            nonzero_entries: values.par_iter().filter(|&&value| value != 0).count(),
            logical_entries: logical_len,
            row_commitments: shape.row_count(),
            physical_rows: values.len().div_ceil(row_width),
            materialized_field_entries: 0,
            materialized_padding_entries: source.materialized_padding_entries(),
            padding_group_operations: 0,
        },
    })
}

/// Grid-aware BVA commitment preserving direct signed-16-bit storage.
pub fn commit_section6_bva_i16_on_grid_zk<S: DenseI16BvaSource>(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    source: &S,
    bit_width: usize,
) -> Result<CommittedDoryBva, Section6FinalOpeningError> {
    if bit_width == 0 || bit_width > 16 {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let logical_len = source.layout().logical_len();
    let values = source.dense_values();
    if bit_width < 16
        && values
            .par_iter()
            .any(|value| u64::from(value.unsigned_abs()) >= (1_u64 << bit_width))
    {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let dense = dense_i16_polynomial(source)
        .map_err(|error| Section6FinalOpeningError::Dory(error.to_string()))?;
    let (commitment, hint) = commit_dense_i16_zk(&dense, &parameters.pcs);
    let shape = grid.native_shape(logical_len)?;
    let row_width = shape.row_width();
    Ok(CommittedDoryBva {
        commitment,
        hint,
        profile: DoryCommitProfile {
            physical_entries: values.len(),
            nonzero_entries: values.par_iter().filter(|&&value| value != 0).count(),
            logical_entries: logical_len,
            row_commitments: shape.row_count(),
            physical_rows: values.len().div_ceil(row_width),
            materialized_field_entries: 0,
            materialized_padding_entries: source.materialized_padding_entries(),
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
    prove_section6_final_opening_profiled(role, outer_binding, parameters, grid, claims, rng)
        .map(|(proof, _)| proof)
}

/// Reduce heterogeneous claims once and report each terminal prover stage.
#[allow(clippy::too_many_arguments)]
pub fn prove_section6_final_opening_profiled(
    role: &[u8],
    outer_binding: &[u8],
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    claims: &[Section6FinalOpeningProverClaim<'_>],
    rng: &mut impl RngCore,
) -> Result<(Section6FinalOpeningProof, Section6FinalOpeningProverProfile), Section6FinalOpeningError>
{
    let total_started = Instant::now();
    let preparation_state_setup_started = Instant::now();
    validate_prover_claims(parameters, *grid, claims)?;
    let statements = claims
        .iter()
        .map(|claim| Section6FinalOpeningVerifierClaim {
            commitment: &claim.committed.commitment,
            logical_len: claim.committed.logical_len,
            point_msb: claim.point_msb,
            functional: claim.functional,
            evaluation_commitment: claim.evaluation_commitment,
        })
        .collect::<Vec<_>>();
    let mut transcript = final_opening_transcript(role, outer_binding, *grid, &statements);
    let claim_coefficients = transcript.challenge_scalar_powers(claims.len());
    let mut states = ReductionStateSet::new(claims, &claim_coefficients, *grid)?;
    let source_grouping = states.source_grouping();
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
    let preparation_state_setup_seconds = preparation_state_setup_started.elapsed().as_secs_f64();

    let sumcheck_started = Instant::now();
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
    let sumcheck_seconds = sumcheck_started.elapsed().as_secs_f64();

    let dory_batch_open_started = Instant::now();
    let claim_link_coefficients = bind_claim_link_commitments(&mut transcript, &statements);
    let dory_outer_binding = transcript.state();
    let embedding_scales = states.embedding_scales(&sumcheck_point);
    let full_evaluations = source_grouping
        .claim_source_indices
        .iter()
        .zip(&embedding_scales)
        .map(|(&source_index, &scale)| terminal_values[source_index] * scale)
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
    let dory_batch_open_seconds = dory_batch_open_started.elapsed().as_secs_f64();

    let blindfold_preparation_started = Instant::now();
    let terminal_sumcheck_coefficients = source_grouping.aggregate(
        &terminal_sumcheck_coefficients(*grid, &statements, &sumcheck_point, &claim_coefficients)?,
    )?;
    let claimed_sumcheck_coefficients =
        claimed_sumcheck_coefficients(&sumcheck_point, &claim_coefficients);
    let terminal_dory_coefficients = source_grouping.aggregate(&terminal_dory_coefficients(
        *grid,
        &statements,
        &sumcheck_point,
        &dory_coefficients,
    )?)?;
    bind_blindfold_coefficients(
        &mut transcript,
        &claim_coefficients,
        &terminal_sumcheck_coefficients,
        &claimed_sumcheck_coefficients,
        &claim_link_coefficients,
        &terminal_dory_coefficients,
    );
    let final_opening_mix = transcript.challenge_scalar();
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
        source_grouping.source_count,
        claims.len(),
        consistency,
        &sumcheck,
        &claim_coefficients,
        &terminal_sumcheck_coefficients,
        &claimed_sumcheck_coefficients,
        &claim_link_coefficients,
        combine_evaluation_commitments(&statements, &claim_link_coefficients)?,
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
    let dory_evaluation = combine_scalars(&terminal_values, &terminal_dory_coefficients);
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
    let blindfold_preparation_seconds = blindfold_preparation_started.elapsed().as_secs_f64();

    let blindfold_prove_started = Instant::now();
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
    let blindfold_prove_seconds = blindfold_prove_started.elapsed().as_secs_f64();

    let result_assembly_started = Instant::now();
    let proof = Section6FinalOpeningProof {
        sumcheck,
        dory,
        blindfold_transcript_state,
        blindfold,
    };
    let result_assembly_seconds = result_assembly_started.elapsed().as_secs_f64();
    let profile = Section6FinalOpeningProverProfile {
        preparation_state_setup_seconds,
        sumcheck_seconds,
        dory_batch_open_seconds,
        blindfold_preparation_seconds,
        blindfold_prove_seconds,
        result_assembly_seconds,
        total_seconds: total_started.elapsed().as_secs_f64(),
    };
    Ok((proof, profile))
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
    verify_section6_final_opening_profiled(role, outer_binding, parameters, grid, claims, proof)
        .map(|_| ())
}

/// Verify the terminal proof and report each verifier stage.
#[allow(clippy::too_many_arguments)]
pub fn verify_section6_final_opening_profiled(
    role: &[u8],
    outer_binding: &[u8],
    parameters: &DoryVerifierParameters,
    grid: &Section6DoryGrid,
    claims: &[Section6FinalOpeningVerifierClaim<'_>],
    proof: &Section6FinalOpeningProof,
) -> Result<Section6FinalOpeningVerifierProfile, Section6FinalOpeningError> {
    let total_started = Instant::now();
    let preparation_state_setup_started = Instant::now();
    validate_verifier_claims(parameters, *grid, claims)?;
    let mut transcript = final_opening_transcript(role, outer_binding, *grid, claims);
    let claim_coefficients = transcript.challenge_scalar_powers(claims.len());
    let source_grouping = SourceGrouping::from_verifier_claims(claims, *grid);
    let preparation_state_setup_seconds = preparation_state_setup_started.elapsed().as_secs_f64();

    let sumcheck_verify_started = Instant::now();
    let consistency = SumcheckProof::<Fr, Bn254G1>::Committed(proof.sumcheck.clone())
        .verify_committed_consistency(
            SumcheckStatement::new(grid.num_vars, SUMCHECK_DEGREE),
            &mut transcript,
        )
        .map_err(|error| Section6FinalOpeningError::Sumcheck(error.to_string()))?;
    let sumcheck_point = consistency.challenges();
    let sumcheck_verify_seconds = sumcheck_verify_started.elapsed().as_secs_f64();

    let dory_batch_verify_started = Instant::now();
    let claim_link_coefficients = bind_claim_link_commitments(&mut transcript, claims);
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
    let dory_batch_verify_seconds = dory_batch_verify_started.elapsed().as_secs_f64();

    let blindfold_preparation_started = Instant::now();
    let terminal_sumcheck_coefficients = source_grouping.aggregate(
        &terminal_sumcheck_coefficients(*grid, claims, &sumcheck_point, &claim_coefficients)?,
    )?;
    let claimed_sumcheck_coefficients =
        claimed_sumcheck_coefficients(&sumcheck_point, &claim_coefficients);
    let terminal_dory_coefficients = source_grouping.aggregate(&terminal_dory_coefficients(
        *grid,
        claims,
        &sumcheck_point,
        &dory_coefficients,
    )?)?;
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
        source_grouping.source_count,
        claims.len(),
        consistency,
        &proof.sumcheck,
        &claim_coefficients,
        &terminal_sumcheck_coefficients,
        &claimed_sumcheck_coefficients,
        &claim_link_coefficients,
        combine_evaluation_commitments(claims, &claim_link_coefficients)?,
        &terminal_dory_coefficients,
        dory_hiding_commitment,
        final_opening_mix,
    )?;
    if transcript.state() != proof.blindfold_transcript_state {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let blindfold_preparation_seconds = blindfold_preparation_started.elapsed().as_secs_f64();

    let blindfold_verify_started = Instant::now();
    let blindfold_result = protocol.verify::<Pedersen<Bn254G1>, _>(
        &proof.blindfold,
        &parameters.blindfold,
        &mut transcript,
    );
    let blindfold_verify_seconds = blindfold_verify_started.elapsed().as_secs_f64();

    let result_finalization_started = Instant::now();
    blindfold_result.map_err(|error| Section6FinalOpeningError::BlindFold(format!("{error:?}")))?;
    let result_finalization_seconds = result_finalization_started.elapsed().as_secs_f64();
    Ok(Section6FinalOpeningVerifierProfile {
        preparation_state_setup_seconds,
        sumcheck_verify_seconds,
        dory_batch_verify_seconds,
        blindfold_preparation_seconds,
        blindfold_verify_seconds,
        result_finalization_seconds,
        total_seconds: total_started.elapsed().as_secs_f64(),
    })
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

fn validate_sparse_matrix_functional(
    logical_len: usize,
    point_msb: &[Fr],
    padded_rows: usize,
    logical_cols: usize,
    padded_cols: usize,
    selected_columns: &[usize],
) -> Result<(), Section6FinalOpeningError> {
    if padded_rows == 0
        || padded_cols == 0
        || !padded_rows.is_power_of_two()
        || !padded_cols.is_power_of_two()
        || padded_rows.checked_mul(padded_cols) != Some(logical_len)
        || logical_cols == 0
        || logical_cols > padded_cols
        || selected_columns.is_empty()
        || selected_columns.len() >= logical_cols
        || selected_columns
            .iter()
            .any(|&column| column >= logical_cols)
        || selected_columns
            .windows(2)
            .any(|columns| columns[0] >= columns[1])
    {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let expected_point_len =
        padded_cols.trailing_zeros() as usize + padded_rows.trailing_zeros() as usize;
    if point_msb.len() != expected_point_len {
        return Err(Section6FinalOpeningError::PointLength {
            expected: expected_point_len,
            actual: point_msb.len(),
        });
    }
    Ok(())
}

fn validate_evaluation_commitment(
    logical_len: usize,
    point_msb: &[ArkFr],
    functional: Section6FinalOpeningFunctional<'_>,
    commitment: Section6EvaluationCommitment<'_>,
) -> Result<(), Section6FinalOpeningError> {
    let Section6EvaluationCommitment::MatrixColumns {
        logical_cols,
        padded_cols,
        selected_columns,
        commitments,
    } = commitment
    else {
        return Ok(());
    };
    if logical_cols == 0
        || padded_cols == 0
        || logical_cols > padded_cols
        || !padded_cols.is_power_of_two()
        || logical_len % padded_cols != 0
        || !(logical_len / padded_cols).is_power_of_two()
        || selected_columns.is_empty()
        || selected_columns.len() != commitments.len()
        || selected_columns
            .iter()
            .any(|&column| column >= logical_cols)
        || selected_columns
            .windows(2)
            .any(|columns| columns[0] >= columns[1])
        || padded_cols.trailing_zeros() as usize > point_msb.len()
    {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    match functional {
        Section6FinalOpeningFunctional::Point => {
            if selected_columns.len() != logical_cols
                || selected_columns
                    .iter()
                    .copied()
                    .enumerate()
                    .any(|(column, selected)| column != selected)
            {
                return Err(Section6FinalOpeningError::ProofShape);
            }
        }
        Section6FinalOpeningFunctional::SparseMatrixColumns {
            logical_cols: functional_logical_cols,
            padded_cols: functional_padded_cols,
            selected_columns: functional_selected_columns,
            ..
        } => {
            if logical_cols != functional_logical_cols
                || padded_cols != functional_padded_cols
                || selected_columns != functional_selected_columns
            {
                return Err(Section6FinalOpeningError::ProofShape);
            }
        }
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
            Section6FinalOpeningData::WholeMatrix {
                values,
                padded_rows,
                padded_cols,
            } if values.len() != claim.committed.logical_len
                || padded_rows == 0
                || padded_cols == 0
                || !padded_rows.is_power_of_two()
                || !padded_cols.is_power_of_two()
                || padded_rows.checked_mul(padded_cols) != Some(values.len()) =>
            {
                return Err(Section6FinalOpeningError::SourceLength {
                    logical: claim.committed.logical_len,
                    physical: values.len(),
                });
            }
            Section6FinalOpeningData::MatrixColumn {
                values,
                padded_rows,
                padded_cols,
                column,
            } if values.len() != claim.committed.logical_len
                || padded_rows == 0
                || padded_cols == 0
                || !padded_rows.is_power_of_two()
                || !padded_cols.is_power_of_two()
                || padded_rows.checked_mul(padded_cols) != Some(values.len())
                || column >= padded_cols =>
            {
                return Err(Section6FinalOpeningError::SourceLength {
                    logical: claim.committed.logical_len,
                    physical: values.len(),
                });
            }
            _ => {}
        }
        if let Section6FinalOpeningData::MatrixColumn {
            padded_cols,
            column,
            ..
        } = claim.data
        {
            let expected_prefix = column_point_msb(column, padded_cols)?;
            if !claim.point_msb.starts_with(&expected_prefix) {
                return Err(Section6FinalOpeningError::ProofShape);
            }
        }
        match (claim.data, claim.functional) {
            (
                Section6FinalOpeningData::WholeMatrix {
                    padded_rows,
                    padded_cols,
                    ..
                },
                Section6FinalOpeningFunctional::SparseMatrixColumns {
                    padded_rows: functional_rows,
                    logical_cols,
                    padded_cols: functional_cols,
                    selected_columns,
                },
            ) if padded_rows == functional_rows && padded_cols == functional_cols => {
                let point = claim
                    .point_msb
                    .iter()
                    .copied()
                    .map(Fr::from)
                    .collect::<Vec<_>>();
                validate_sparse_matrix_functional(
                    claim.committed.logical_len,
                    &point,
                    padded_rows,
                    logical_cols,
                    padded_cols,
                    selected_columns,
                )?;
            }
            (_, Section6FinalOpeningFunctional::Point) => {}
            _ => return Err(Section6FinalOpeningError::ProofShape),
        }
        validate_evaluation_commitment(
            claim.committed.logical_len,
            claim.point_msb,
            claim.functional,
            claim.evaluation_commitment,
        )?;
        if let Section6EvaluationCommitment::Direct(commitment) = claim.evaluation_commitment {
            if commitment != claim.evaluation.commitment {
                return Err(Section6FinalOpeningError::EvaluationCommitmentMismatch(
                    index,
                ));
            }
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
        if let Section6FinalOpeningFunctional::SparseMatrixColumns {
            padded_rows,
            logical_cols,
            padded_cols,
            selected_columns,
        } = claim.functional
        {
            let point = claim
                .point_msb
                .iter()
                .copied()
                .map(Fr::from)
                .collect::<Vec<_>>();
            validate_sparse_matrix_functional(
                claim.logical_len,
                &point,
                padded_rows,
                logical_cols,
                padded_cols,
                selected_columns,
            )?;
        }
        validate_evaluation_commitment(
            claim.logical_len,
            claim.point_msb,
            claim.functional,
            claim.evaluation_commitment,
        )?;
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

struct SourceGrouping {
    source_count: usize,
    claim_source_indices: Vec<usize>,
}

impl SourceGrouping {
    fn from_verifier_claims(
        claims: &[Section6FinalOpeningVerifierClaim<'_>],
        grid: Section6DoryGrid,
    ) -> Self {
        let mut source_indices = HashMap::<Section6DorySourceKey, usize>::new();
        let mut claim_source_indices = Vec::with_capacity(claims.len());
        for claim in claims {
            let key = Section6DorySourceKey {
                commitment: commitment_binding_bytes(claim.commitment),
                logical_len: claim.logical_len,
                grid_num_vars: grid.num_vars,
            };
            let next_index = source_indices.len();
            let source_index = *source_indices.entry(key).or_insert(next_index);
            claim_source_indices.push(source_index);
        }
        Self {
            source_count: source_indices.len(),
            claim_source_indices,
        }
    }

    fn aggregate(&self, values: &[Fr]) -> Result<Vec<Fr>, Section6FinalOpeningError> {
        if values.len() != self.claim_source_indices.len() {
            return Err(Section6FinalOpeningError::ProofShape);
        }
        let mut aggregated = vec![Fr::zero(); self.source_count];
        for (&value, &source_index) in values.iter().zip(&self.claim_source_indices) {
            aggregated[source_index] += value;
        }
        Ok(aggregated)
    }
}

enum ClaimEquality {
    Dense(Vec<Fr>),
    WholeMatrixTensor {
        row: Vec<Fr>,
        column: Vec<Fr>,
    },
    SparseMatrixTensor {
        row: Vec<Fr>,
        columns: Vec<(usize, Fr)>,
    },
    MatrixColumn {
        start: usize,
        values: Vec<Fr>,
    },
}

struct ReductionStateSet {
    sources: Vec<ReductionSourceState>,
    claims: Vec<ReductionClaimState>,
    #[cfg(test)]
    claim_equality_entries: usize,
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
        #[cfg(test)]
        let mut claim_equality_entries = 0;
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
            let source = &mut sources[source_index];
            let (native_evaluation, claim_eq) = match claim.data {
                Section6FinalOpeningData::Field(_) => {
                    let claim_eq = EqPolynomial::<Fr>::evals(&point, None);
                    #[cfg(test)]
                    {
                        claim_equality_entries += claim_eq.len();
                    }
                    let evaluation = source
                        .values
                        .par_iter()
                        .zip(claim_eq.par_iter())
                        .map(|(&value, &weight)| value * weight)
                        .sum::<Fr>();
                    (evaluation, ClaimEquality::Dense(claim_eq))
                }
                Section6FinalOpeningData::WholeMatrix {
                    padded_rows,
                    padded_cols,
                    ..
                } => {
                    let column_vars = padded_cols.trailing_zeros() as usize;
                    let (column_point, row_point) = point.split_at(column_vars);
                    if row_point.len() != padded_rows.trailing_zeros() as usize {
                        return Err(Section6FinalOpeningError::ProofShape);
                    }
                    let column_eq = EqPolynomial::<Fr>::evals(column_point, None);
                    let row_eq = EqPolynomial::<Fr>::evals(row_point, None);
                    match claim.functional {
                        Section6FinalOpeningFunctional::Point => {
                            #[cfg(test)]
                            {
                                claim_equality_entries += column_eq.len() + row_eq.len();
                            }
                            let evaluation = source
                                .values
                                .par_chunks(padded_rows)
                                .zip(column_eq.par_iter())
                                .map(|(column, &column_weight)| {
                                    column_weight
                                        * column
                                            .iter()
                                            .zip(&row_eq)
                                            .map(|(&value, &row_weight)| value * row_weight)
                                            .sum::<Fr>()
                                })
                                .sum::<Fr>();
                            (
                                evaluation,
                                ClaimEquality::WholeMatrixTensor {
                                    row: row_eq,
                                    column: column_eq,
                                },
                            )
                        }
                        Section6FinalOpeningFunctional::SparseMatrixColumns {
                            padded_rows: functional_rows,
                            padded_cols: functional_cols,
                            selected_columns,
                            ..
                        } if functional_rows == padded_rows && functional_cols == padded_cols => {
                            let columns = selected_columns
                                .iter()
                                .map(|&column| (column, column_eq[column]))
                                .collect::<Vec<_>>();
                            #[cfg(test)]
                            {
                                claim_equality_entries += columns.len() + row_eq.len();
                            }
                            let evaluation = columns
                                .par_iter()
                                .map(|&(column, column_weight)| {
                                    let start = column * padded_rows;
                                    column_weight
                                        * source.values[start..start + padded_rows]
                                            .iter()
                                            .zip(&row_eq)
                                            .map(|(&value, &row_weight)| value * row_weight)
                                            .sum::<Fr>()
                                })
                                .sum::<Fr>();
                            (
                                evaluation,
                                ClaimEquality::SparseMatrixTensor {
                                    row: row_eq,
                                    columns,
                                },
                            )
                        }
                        _ => return Err(Section6FinalOpeningError::ProofShape),
                    }
                }
                Section6FinalOpeningData::MatrixColumn {
                    padded_rows,
                    padded_cols,
                    column,
                    ..
                } => {
                    let column_vars = padded_cols.trailing_zeros() as usize;
                    let row_point = point
                        .get(column_vars..)
                        .ok_or(Section6FinalOpeningError::ProofShape)?;
                    if row_point.len() != padded_rows.trailing_zeros() as usize {
                        return Err(Section6FinalOpeningError::ProofShape);
                    }
                    let claim_eq = EqPolynomial::<Fr>::evals(row_point, None);
                    #[cfg(test)]
                    {
                        claim_equality_entries += claim_eq.len();
                    }
                    let start = column
                        .checked_mul(padded_rows)
                        .ok_or(Section6FinalOpeningError::InvalidLogicalLength(usize::MAX))?;
                    let end = start
                        .checked_add(padded_rows)
                        .ok_or(Section6FinalOpeningError::InvalidLogicalLength(usize::MAX))?;
                    let evaluation = source
                        .values
                        .get(start..end)
                        .ok_or(Section6FinalOpeningError::ProofShape)?
                        .par_iter()
                        .zip(claim_eq.par_iter())
                        .map(|(&value, &weight)| value * weight)
                        .sum::<Fr>();
                    (
                        evaluation,
                        ClaimEquality::MatrixColumn {
                            start,
                            values: claim_eq,
                        },
                    )
                }
            };
            if native_evaluation != Fr::from(claim.evaluation.value) {
                return Err(Section6FinalOpeningError::EvaluationMismatch(claim_index));
            }
            match claim_eq {
                ClaimEquality::Dense(mut values) => {
                    if source.eq.is_empty() {
                        values
                            .par_iter_mut()
                            .for_each(|weight| *weight *= coefficient);
                        source.eq = values;
                    } else {
                        source
                            .eq
                            .par_iter_mut()
                            .zip(values.par_iter())
                            .for_each(|(combined, &weight)| *combined += coefficient * weight);
                    }
                }
                ClaimEquality::WholeMatrixTensor { row, column } => {
                    if source.eq.is_empty() {
                        source.eq = vec![Fr::zero(); source.values.len()];
                    }
                    if source.eq.len() != row.len() * column.len() {
                        return Err(Section6FinalOpeningError::ProofShape);
                    }
                    source
                        .eq
                        .par_chunks_mut(row.len())
                        .zip(column.par_iter())
                        .for_each(|(combined, &column_weight)| {
                            combined
                                .iter_mut()
                                .zip(&row)
                                .for_each(|(combined, &row_weight)| {
                                    *combined += coefficient * column_weight * row_weight;
                                });
                        });
                }
                ClaimEquality::SparseMatrixTensor { row, columns } => {
                    if source.eq.is_empty() {
                        source.eq = vec![Fr::zero(); source.values.len()];
                    }
                    if source.eq.len() % row.len() != 0 {
                        return Err(Section6FinalOpeningError::ProofShape);
                    }
                    for (column, column_weight) in columns {
                        let start = column
                            .checked_mul(row.len())
                            .ok_or(Section6FinalOpeningError::InvalidLogicalLength(usize::MAX))?;
                        let end = start
                            .checked_add(row.len())
                            .ok_or(Section6FinalOpeningError::InvalidLogicalLength(usize::MAX))?;
                        let combined = source
                            .eq
                            .get_mut(start..end)
                            .ok_or(Section6FinalOpeningError::ProofShape)?;
                        combined.par_iter_mut().zip(row.par_iter()).for_each(
                            |(combined, &row_weight)| {
                                *combined += coefficient * column_weight * row_weight;
                            },
                        );
                    }
                }
                ClaimEquality::MatrixColumn { start, values } => {
                    if source.eq.is_empty() {
                        source.eq = vec![Fr::zero(); source.values.len()];
                    }
                    source.eq[start..start + values.len()]
                        .par_iter_mut()
                        .zip(values.par_iter())
                        .for_each(|(combined, &weight)| *combined += coefficient * weight);
                }
            }
            claims.push(ReductionClaimState { source_index });
        }
        Ok(Self {
            sources,
            claims,
            #[cfg(test)]
            claim_equality_entries,
        })
    }

    fn bind(&mut self, round_index: usize, challenge: Fr) -> Result<(), Section6FinalOpeningError> {
        self.sources
            .par_iter_mut()
            .try_for_each(|source| source.bind(round_index, challenge))
    }

    fn source_grouping(&self) -> SourceGrouping {
        SourceGrouping {
            source_count: self.sources.len(),
            claim_source_indices: self.claims.iter().map(|claim| claim.source_index).collect(),
        }
    }

    fn terminal_values(&self) -> Result<Vec<Fr>, Section6FinalOpeningError> {
        self.sources
            .iter()
            .map(ReductionSourceState::terminal_value)
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
            let functional_evaluation = match claim.functional {
                Section6FinalOpeningFunctional::Point => {
                    EqPolynomial::<Fr>::mle(&point, &native_point)
                }
                Section6FinalOpeningFunctional::SparseMatrixColumns {
                    padded_rows,
                    padded_cols,
                    selected_columns,
                    ..
                } => {
                    let column_vars = padded_cols.trailing_zeros() as usize;
                    let row_vars = padded_rows.trailing_zeros() as usize;
                    if point.len() != column_vars + row_vars
                        || native_point.len() != column_vars + row_vars
                    {
                        return Err(Section6FinalOpeningError::ProofShape);
                    }
                    let (column_point, row_point) = point.split_at(column_vars);
                    let (native_column_point, native_row_point) =
                        native_point.split_at(column_vars);
                    let column_selector = selected_columns
                        .iter()
                        .map(|&column| {
                            Ok(
                                equality_at_boolean_index(column_point, column, padded_cols)?
                                    * equality_at_boolean_index(
                                        native_column_point,
                                        column,
                                        padded_cols,
                                    )?,
                            )
                        })
                        .collect::<Result<Vec<_>, Section6FinalOpeningError>>()?
                        .into_iter()
                        .sum::<Fr>();
                    EqPolynomial::<Fr>::mle(row_point, native_row_point) * column_selector
                }
            };
            Ok(coefficient * scale * scale * functional_evaluation)
        })
        .collect()
}

fn equality_at_boolean_index(
    point: &[Fr],
    index: usize,
    domain_len: usize,
) -> Result<Fr, Section6FinalOpeningError> {
    if domain_len == 0
        || !domain_len.is_power_of_two()
        || point.len() != domain_len.trailing_zeros() as usize
        || index >= domain_len
    {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    Ok(point
        .iter()
        .enumerate()
        .fold(Fr::one(), |evaluation, (coordinate, &value)| {
            let bit = (index >> (point.len() - coordinate - 1)) & 1;
            evaluation * if bit == 0 { Fr::one() - value } else { value }
        }))
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
    let hints = representatives
        .iter()
        .map(|&index| claims[index].committed.hint.as_ref().clone())
        .collect::<Vec<_>>();
    let combined_hint =
        <DoryScheme as AdditivelyHomomorphic>::combine_hints(hints, &grouped_coefficients);
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
    let mut unique_commitments = Vec::new();
    let mut grouped_coefficients = Vec::new();
    for ((commitment, &logical_len), &coefficient) in
        commitments.iter().zip(&logical_lengths).zip(&coefficients)
    {
        let key = Section6DorySourceKey {
            commitment: commitment_binding_bytes(commitment),
            logical_len,
            grid_num_vars: grid.num_vars,
        };
        if let Some(&group_index) = group_indices.get(&key) {
            grouped_coefficients[group_index] += coefficient;
        } else {
            let group_index = unique_commitments.len();
            group_indices.insert(key, group_index);
            unique_commitments.push(commitment.clone());
            grouped_coefficients.push(coefficient);
        }
    }
    let joint_commitment =
        <DoryScheme as AdditivelyHomomorphic>::combine(&unique_commitments, &grouped_coefficients);
    let opening_point = Point::high_to_low(point.to_vec());
    let hiding_commitment = <DoryScheme as ZkOpeningScheme>::verify_zk(
        &joint_commitment,
        opening_point.as_slice(),
        &proof.opening,
        &parameters.pcs,
        &mut transcript,
    )
    .map_err(|error| Section6FinalOpeningError::Dory(error.to_string()))?;
    ZkEvaluationClaim::new(opening_point.as_slice(), &hiding_commitment)
        .append_to_transcript(&mut transcript);
    Ok((hiding_commitment, coefficients))
}

fn final_opening_transcript(
    role: &[u8],
    outer_binding: &[u8],
    grid: Section6DoryGrid,
    claims: &[Section6FinalOpeningVerifierClaim<'_>],
) -> Blake2bTranscript<Fr> {
    let mut transcript = Blake2bTranscript::<Fr>::new(b"pc-section6-final-opening-v2");
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
        match claim.functional {
            Section6FinalOpeningFunctional::Point => {
                transcript.append_labeled(b"functional_kind", &U64Word(0));
            }
            Section6FinalOpeningFunctional::SparseMatrixColumns {
                padded_rows,
                logical_cols,
                padded_cols,
                selected_columns,
            } => {
                transcript.append_labeled(b"functional_kind", &U64Word(1));
                transcript.append_labeled(b"functional_rows", &U64Word(padded_rows as u64));
                transcript.append_labeled(b"functional_lcols", &U64Word(logical_cols as u64));
                transcript.append_labeled(b"functional_pcols", &U64Word(padded_cols as u64));
                let column_vars = padded_cols.trailing_zeros() as usize;
                let split = column_vars.min(point.len());
                transcript.append_values(b"functional_col_point", &point[..split]);
                transcript.append_values(b"functional_row_point", &point[split..]);
                transcript.append(&LabelWithCount(
                    b"functional_columns",
                    selected_columns.len() as u64,
                ));
                for &column in selected_columns {
                    transcript.append_labeled(b"functional_column", &U64Word(column as u64));
                }
            }
        }
        append_evaluation_commitment(&mut transcript, claim.evaluation_commitment);
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
    claims: &[Section6FinalOpeningVerifierClaim<'_>],
) -> Vec<Fr> {
    transcript.append(&LabelWithCount(
        b"claim_link_commitments",
        claims.len() as u64,
    ));
    for claim in claims {
        append_evaluation_commitment(transcript, claim.evaluation_commitment);
    }
    transcript.challenge_scalar_powers(claims.len())
}

fn append_evaluation_commitment(
    transcript: &mut Blake2bTranscript<Fr>,
    commitment: Section6EvaluationCommitment<'_>,
) {
    match commitment {
        Section6EvaluationCommitment::Direct(commitment) => {
            transcript.append_labeled(b"eval_com_kind", &U64Word(0));
            transcript.append_labeled(b"eval_com", &commitment);
        }
        Section6EvaluationCommitment::MatrixColumns {
            logical_cols,
            padded_cols,
            selected_columns,
            commitments,
        } => {
            transcript.append_labeled(b"eval_com_kind", &U64Word(1));
            transcript.append_labeled(b"eval_com_logical_cols", &U64Word(logical_cols as u64));
            transcript.append_labeled(b"eval_com_padded_cols", &U64Word(padded_cols as u64));
            transcript.append(&LabelWithCount(
                b"eval_com_columns",
                selected_columns.len() as u64,
            ));
            for (&column, commitment) in selected_columns.iter().zip(commitments) {
                transcript.append_labeled(b"eval_com_column", &U64Word(column as u64));
                transcript.append_labeled(b"eval_com", commitment);
            }
        }
    }
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
    let link_coefficients = bind_claim_link_commitments(&mut transcript, claims);
    if link_coefficients != expected_link_coefficients {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    Ok(consistency)
}

#[allow(clippy::too_many_arguments)]
fn build_blindfold_protocol(
    row_len: usize,
    num_vars: usize,
    source_count: usize,
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
    if claim_coefficients.len() != claim_count
        || terminal_sumcheck_coefficients.len() != source_count
        || claimed_sumcheck_coefficients.len() != claim_count
        || claim_link_coefficients.len() != claim_count
        || terminal_dory_coefficients.len() != source_count
    {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let terminal_ids = (0..source_count).collect::<Vec<_>>();
    let claimed_ids = (source_count..source_count + claim_count).collect::<Vec<_>>();
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

fn combine_evaluation_commitments(
    claims: &[Section6FinalOpeningVerifierClaim<'_>],
    coefficients: &[Fr],
) -> Result<Bn254G1, Section6FinalOpeningError> {
    if claims.len() != coefficients.len() {
        return Err(Section6FinalOpeningError::ProofShape);
    }
    let mut commitments = Vec::new();
    let mut scalars = Vec::new();
    for (claim, &coefficient) in claims.iter().zip(coefficients) {
        match claim.evaluation_commitment {
            Section6EvaluationCommitment::Direct(commitment) => {
                commitments.push(commitment);
                scalars.push(coefficient);
            }
            Section6EvaluationCommitment::MatrixColumns {
                padded_cols,
                selected_columns,
                commitments: column_commitments,
                ..
            } => {
                if selected_columns.len() != column_commitments.len() {
                    return Err(Section6FinalOpeningError::ProofShape);
                }
                let column_vars = padded_cols.trailing_zeros() as usize;
                if column_vars > claim.point_msb.len() {
                    return Err(Section6FinalOpeningError::ProofShape);
                }
                let column_point = claim.point_msb[..column_vars]
                    .iter()
                    .copied()
                    .map(Fr::from)
                    .collect::<Vec<_>>();
                for (&column, &commitment) in selected_columns.iter().zip(column_commitments) {
                    commitments.push(commitment);
                    scalars.push(coefficient * eq_index_msb(&column_point, column as u128));
                }
            }
        }
    }
    Ok(Bn254G1::msm(&commitments, &scalars))
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr as ArkFr;
    use ark_ff::Zero;
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use jolt_crypto::{Bn254G1, DeriveSetup, JoltGroup, PedersenSetup};
    use jolt_field::Fr as JoltFr;
    use jolt_poly::EqPolynomial;
    use jolt_transcript::{Blake2bTranscript, Transcript};

    use crate::{
        hidden_relation::{HiddenRelationTable, HiddenRelationTableStatement, HiddenScalarWitness},
        jolt_dory::{evaluate_field_vector_standard, setup_dory_parameters},
        jolt_dory_adapter::{open_dense_i8_zk, verify_dense_i8_zk, DenseI8Prefix},
    };

    use super::{
        commit_section6_dense_i8_on_grid_zk, commit_section6_field_matrix_columns_on_grid_zk,
        commit_section6_field_on_grid_zk, prove_section6_final_opening,
        resolve_section6_hidden_relation_prover_claim,
        resolve_section6_hidden_relation_sparse_matrix_prover_claim,
        resolve_section6_hidden_relation_sparse_matrix_verifier_claim,
        resolve_section6_hidden_relation_whole_matrix_prover_claim, verify_section6_final_opening,
        ReductionStateSet, Section6DoryGrid, Section6EvaluationCommitment,
        Section6FinalOpeningData, Section6FinalOpeningError, Section6FinalOpeningFunctional,
        Section6FinalOpeningProverClaim, Section6FinalOpeningVerifierClaim,
    };

    fn ark(value: u64) -> ArkFr {
        ArkFr::from(value)
    }

    #[test]
    fn dense_i8_commitment_preserves_full_domain_and_omits_tail_rows() {
        let grid = Section6DoryGrid::new(4).unwrap();
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(grid.num_vars());
        let values = [i8::MIN, -64, -1, 0, 1, 32, 64, i8::MAX];

        let actual =
            commit_section6_dense_i8_on_grid_zk(&prover_parameters, &grid, &values, 16).unwrap();
        let dense = DenseI8Prefix::new(&values, 16).unwrap();
        let point = [
            JoltFr::from(2_u64),
            JoltFr::from(3_u64),
            JoltFr::from(4_u64),
            JoltFr::from(5_u64),
        ];
        let evaluation = dense.evaluate_monomial_low(&point).unwrap();
        let mut prover_transcript = Blake2bTranscript::new(b"section6-i8-biased-commit");
        let proof = open_dense_i8_zk(
            &dense,
            &actual.commitment,
            actual.hint.as_ref().clone(),
            &point,
            evaluation,
            b"section6-i8-biased-commit",
            &prover_parameters.pcs,
            &mut prover_transcript,
        )
        .unwrap();
        let evaluation_setup = PedersenSetup::<Bn254G1>::derive(&prover_parameters.pcs, 1);
        let mut verifier_transcript = Blake2bTranscript::new(b"section6-i8-biased-commit");
        verify_dense_i8_zk(
            &actual.commitment,
            &proof,
            16,
            &point,
            evaluation,
            b"section6-i8-biased-commit",
            &verifier_parameters.pcs,
            &evaluation_setup,
            &mut verifier_transcript,
        )
        .unwrap();

        assert_eq!(actual.hint.row_commitment_count(), 2);
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
                functional: Section6FinalOpeningFunctional::Point,
                evaluation: witness,
                evaluation_commitment: Section6EvaluationCommitment::Direct(witness.commitment),
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
                functional: Section6FinalOpeningFunctional::Point,
                evaluation_commitment: Section6EvaluationCommitment::Direct(witness.commitment),
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
    fn matrix_column_claims_expand_only_row_equality_tables() {
        let grid = Section6DoryGrid::new(6).unwrap();
        let (prover_parameters, _) = setup_dory_parameters(grid.num_vars());
        let columns = [
            (0..8).map(|index| ark(index + 1)).collect::<Vec<_>>(),
            (0..8).map(|index| ark(index + 17)).collect::<Vec<_>>(),
        ];
        let column_refs = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let committed = commit_section6_field_matrix_columns_on_grid_zk(
            &prover_parameters,
            &grid,
            &column_refs,
            4,
        )
        .unwrap();
        let points = [vec![ark(2), ark(3), ark(5)], vec![ark(7), ark(11), ark(13)]];
        let evaluations = columns
            .iter()
            .zip(&points)
            .map(|(column, point)| evaluate_field_vector_standard(column, point).unwrap())
            .collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(0x434f_4c55_4d4e_4551);
        let witnesses = evaluations
            .iter()
            .map(|&evaluation| {
                HiddenScalarWitness::commit(&prover_parameters, evaluation, &mut rng)
            })
            .collect::<Vec<_>>();
        let row_points = points
            .iter()
            .map(|point| point.iter().copied().map(JoltFr::from).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let resolved = (0..columns.len())
            .map(|column| {
                resolve_section6_hidden_relation_prover_claim(
                    HiddenRelationTable {
                        values: &columns[column],
                        committed: &committed[column],
                    },
                    &row_points[column],
                    &grid,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let claims = resolved
            .iter()
            .zip(&witnesses)
            .map(|(resolved, witness)| resolved.as_claim(witness))
            .collect::<Vec<_>>();
        let coefficients = [JoltFr::from(ark(19)), JoltFr::from(ark(23))];

        let states = ReductionStateSet::new(&claims, &coefficients, grid).unwrap();

        assert_eq!(states.sources.len(), 1);
        assert_eq!(states.claims.len(), 2);
        assert_eq!(states.claim_equality_entries, 2 * columns[0].len());
    }

    #[test]
    fn whole_matrix_claims_expand_only_tensor_equality_factors() {
        let grid = Section6DoryGrid::new(6).unwrap();
        let (prover_parameters, _) = setup_dory_parameters(grid.num_vars());
        let columns = (0..3)
            .map(|column| {
                (0..8)
                    .map(|row| ark((16 * column + row + 1) as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let column_refs = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let committed = commit_section6_field_matrix_columns_on_grid_zk(
            &prover_parameters,
            &grid,
            &column_refs,
            4,
        )
        .unwrap();
        let matrix = committed[0].matrix.as_ref().unwrap();
        assert!(matrix.values[3 * 8..].iter().all(ArkFr::is_zero));
        let point = vec![ark(2), ark(3), ark(5), ark(7), ark(11)];
        let evaluation = evaluate_field_vector_standard(&matrix.values, &point).unwrap();
        let mut rng = StdRng::seed_from_u64(0x5748_4f4c_455f_4551);
        let witness = HiddenScalarWitness::commit(&prover_parameters, evaluation, &mut rng);
        let point_jolt = point.iter().copied().map(JoltFr::from).collect::<Vec<_>>();
        let resolved = resolve_section6_hidden_relation_whole_matrix_prover_claim(
            HiddenRelationTable {
                values: &columns[0],
                committed: &committed[0],
            },
            &point_jolt,
            &grid,
        )
        .unwrap();
        let claim = resolved.as_claim(&witness);
        let states = ReductionStateSet::new(&[claim], &[JoltFr::from(ark(19))], grid).unwrap();

        assert_eq!(states.sources.len(), 1);
        assert_eq!(states.claims.len(), 1);
        assert_eq!(states.claim_equality_entries, 8 + 4);
        assert!(states.claim_equality_entries < matrix.values.len());
        assert_eq!(
            states.sources[0].remaining_inner_product(),
            JoltFr::from(ark(19)) * JoltFr::from(evaluation)
        );
    }

    #[test]
    fn sparse_matrix_functional_round_trips_and_rejects_descriptor_tampering() {
        let grid = Section6DoryGrid::new(6).unwrap();
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(grid.num_vars());
        let columns = (0..3)
            .map(|column| {
                (0..8)
                    .map(|row| ark((100 * column + 7 * row + 3) as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let column_refs = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let committed = commit_section6_field_matrix_columns_on_grid_zk(
            &prover_parameters,
            &grid,
            &column_refs,
            4,
        )
        .unwrap();
        let column_point = vec![ark(2), ark(5)];
        let row_point = vec![ark(7), ark(11), ark(13)];
        let column_point_jolt = column_point
            .iter()
            .copied()
            .map(JoltFr::from)
            .collect::<Vec<_>>();
        let row_point_jolt = row_point
            .iter()
            .copied()
            .map(JoltFr::from)
            .collect::<Vec<_>>();
        let full_point = column_point_jolt
            .iter()
            .chain(&row_point_jolt)
            .copied()
            .collect::<Vec<_>>();
        let selected_columns = vec![0, 2];
        let column_weights = EqPolynomial::<JoltFr>::evals(&column_point_jolt, None);
        let column_evaluations = selected_columns
            .iter()
            .map(|&column| {
                JoltFr::from(evaluate_field_vector_standard(&columns[column], &row_point).unwrap())
            })
            .collect::<Vec<_>>();
        let evaluation = selected_columns
            .iter()
            .zip(&column_evaluations)
            .map(|(&column, &evaluation)| column_weights[column] * evaluation)
            .sum::<JoltFr>();
        let evaluation = ArkFr::from(evaluation);
        let mut rng = StdRng::seed_from_u64(0x5350_4152_5345_4d58);
        let column_witnesses = column_evaluations
            .iter()
            .map(|&evaluation| {
                HiddenScalarWitness::commit(&prover_parameters, ArkFr::from(evaluation), &mut rng)
            })
            .collect::<Vec<_>>();
        let column_commitments = column_witnesses
            .iter()
            .map(|witness| witness.commitment)
            .collect::<Vec<_>>();
        let selected_weights = selected_columns
            .iter()
            .map(|&column| column_weights[column])
            .collect::<Vec<_>>();
        let witness = HiddenScalarWitness {
            value: evaluation,
            commitment: Bn254G1::msm(&column_commitments, &selected_weights),
            blind: column_witnesses
                .iter()
                .zip(&selected_weights)
                .map(|(witness, &weight)| witness.blind * weight)
                .sum(),
        };
        let resolved = resolve_section6_hidden_relation_sparse_matrix_prover_claim(
            HiddenRelationTable {
                values: &columns[0],
                committed: &committed[0],
            },
            &full_point,
            &selected_columns,
            &grid,
        )
        .unwrap();
        let matrix_commitment = Section6EvaluationCommitment::MatrixColumns {
            logical_cols: 3,
            padded_cols: 4,
            selected_columns: &selected_columns,
            commitments: &column_commitments,
        };
        let prover_claim =
            resolved.as_claim_with_evaluation_commitment(&witness, matrix_commitment);
        let states =
            ReductionStateSet::new(&[prover_claim], &[JoltFr::from(ark(19))], grid).unwrap();
        assert_eq!(states.claim_equality_entries, 8 + selected_columns.len());
        assert_eq!(
            states.sources[0].remaining_inner_product(),
            JoltFr::from(ark(19)) * JoltFr::from(evaluation)
        );
        drop(states);

        let proof = prove_section6_final_opening(
            b"section6-sparse-matrix",
            b"outer-binding",
            &prover_parameters,
            &grid,
            &[prover_claim],
            &mut rng,
        )
        .unwrap();
        let encoded = bincode::serialize(&proof).unwrap();
        let decoded = bincode::deserialize(&encoded).unwrap();
        let statement = HiddenRelationTableStatement::from(&committed[0]);
        let verifier_resolved = resolve_section6_hidden_relation_sparse_matrix_verifier_claim(
            statement,
            &full_point,
            &selected_columns,
            witness.commitment,
            &grid,
        )
        .unwrap();
        let verifier_claim =
            verifier_resolved.as_claim_with_evaluation_commitment(matrix_commitment);
        verify_section6_final_opening(
            b"section6-sparse-matrix",
            b"outer-binding",
            &verifier_parameters,
            &grid,
            &[verifier_claim],
            &decoded,
        )
        .unwrap();

        let mut tampered_commitments = column_commitments.clone();
        tampered_commitments[0] = Bn254G1::identity();
        let tampered_commitment = Section6FinalOpeningVerifierClaim {
            evaluation_commitment: Section6EvaluationCommitment::MatrixColumns {
                logical_cols: 3,
                padded_cols: 4,
                selected_columns: &selected_columns,
                commitments: &tampered_commitments,
            },
            ..verifier_claim
        };
        assert!(verify_section6_final_opening(
            b"section6-sparse-matrix",
            b"outer-binding",
            &verifier_parameters,
            &grid,
            &[tampered_commitment],
            &decoded,
        )
        .is_err());

        let other_columns = [1, 2];
        let tampered_selection = Section6FinalOpeningVerifierClaim {
            functional: Section6FinalOpeningFunctional::SparseMatrixColumns {
                padded_rows: 8,
                logical_cols: 3,
                padded_cols: 4,
                selected_columns: &other_columns,
            },
            ..verifier_claim
        };
        assert!(verify_section6_final_opening(
            b"section6-sparse-matrix",
            b"outer-binding",
            &verifier_parameters,
            &grid,
            &[tampered_selection],
            &decoded,
        )
        .is_err());

        let mut other_point = verifier_claim.point_msb.to_vec();
        other_point[0] += ark(1);
        let tampered_point = Section6FinalOpeningVerifierClaim {
            point_msb: &other_point,
            ..verifier_claim
        };
        assert!(verify_section6_final_opening(
            b"section6-sparse-matrix",
            b"outer-binding",
            &verifier_parameters,
            &grid,
            &[tampered_point],
            &decoded,
        )
        .is_err());

        let unsorted_columns = [2, 0];
        let unsorted = Section6FinalOpeningVerifierClaim {
            functional: Section6FinalOpeningFunctional::SparseMatrixColumns {
                padded_rows: 8,
                logical_cols: 3,
                padded_cols: 4,
                selected_columns: &unsorted_columns,
            },
            ..verifier_claim
        };
        assert!(matches!(
            verify_section6_final_opening(
                b"section6-sparse-matrix",
                b"outer-binding",
                &verifier_parameters,
                &grid,
                &[unsorted],
                &decoded,
            ),
            Err(Section6FinalOpeningError::ProofShape)
        ));
    }

    #[test]
    fn sparse_matrix_claims_share_one_source_across_distinct_row_points() {
        let grid = Section6DoryGrid::new(6).unwrap();
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(grid.num_vars());
        let columns = (0..4)
            .map(|column| {
                (0..8)
                    .map(|row| ark((80 * column + 9 * row + 5) as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let column_refs = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let committed = commit_section6_field_matrix_columns_on_grid_zk(
            &prover_parameters,
            &grid,
            &column_refs,
            4,
        )
        .unwrap();
        let column_points = [vec![ark(2), ark(5)], vec![ark(17), ark(19)]];
        let row_points = [
            vec![ark(7), ark(11), ark(13)],
            vec![ark(23), ark(29), ark(31)],
        ];
        let selections = [vec![0, 2], vec![1, 2]];
        let evaluate = |column_point: &[ArkFr], row_point: &[ArkFr], selected: &[usize]| {
            let column_point = column_point
                .iter()
                .copied()
                .map(JoltFr::from)
                .collect::<Vec<_>>();
            let weights = EqPolynomial::<JoltFr>::evals(&column_point, None);
            ArkFr::from(
                selected
                    .iter()
                    .map(|&column| {
                        weights[column]
                            * JoltFr::from(
                                evaluate_field_vector_standard(&columns[column], row_point)
                                    .unwrap(),
                            )
                    })
                    .sum::<JoltFr>(),
            )
        };
        let evaluations = (0..2)
            .map(|index| {
                evaluate(
                    &column_points[index],
                    &row_points[index],
                    &selections[index],
                )
            })
            .collect::<Vec<_>>();
        let full_points = (0..2)
            .map(|index| {
                column_points[index]
                    .iter()
                    .chain(&row_points[index])
                    .copied()
                    .map(JoltFr::from)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(0x5350_4152_5345_3252);
        let witnesses = evaluations
            .iter()
            .map(|&evaluation| {
                HiddenScalarWitness::commit(&prover_parameters, evaluation, &mut rng)
            })
            .collect::<Vec<_>>();
        let resolved = (0..2)
            .map(|index| {
                resolve_section6_hidden_relation_sparse_matrix_prover_claim(
                    HiddenRelationTable {
                        values: &columns[0],
                        committed: &committed[0],
                    },
                    &full_points[index],
                    &selections[index],
                    &grid,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let prover_claims = resolved
            .iter()
            .zip(&witnesses)
            .map(|(resolved, witness)| resolved.as_claim(witness))
            .collect::<Vec<_>>();
        let coefficients = [JoltFr::from(ark(37)), JoltFr::from(ark(41))];
        let states = ReductionStateSet::new(&prover_claims, &coefficients, grid).unwrap();
        assert_eq!(states.sources.len(), 1);
        assert_eq!(states.claims.len(), 2);
        assert_eq!(
            states.sources[0].remaining_inner_product(),
            coefficients[0] * JoltFr::from(evaluations[0])
                + coefficients[1] * JoltFr::from(evaluations[1])
        );
        drop(states);

        let proof = prove_section6_final_opening(
            b"section6-sparse-shared",
            b"outer-binding",
            &prover_parameters,
            &grid,
            &prover_claims,
            &mut rng,
        )
        .unwrap();
        let statement = HiddenRelationTableStatement::from(&committed[0]);
        let verifier_resolved = (0..2)
            .map(|index| {
                resolve_section6_hidden_relation_sparse_matrix_verifier_claim(
                    statement,
                    &full_points[index],
                    &selections[index],
                    witnesses[index].commitment,
                    &grid,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let verifier_claims = verifier_resolved
            .iter()
            .map(|resolved| resolved.as_claim())
            .collect::<Vec<_>>();
        verify_section6_final_opening(
            b"section6-sparse-shared",
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
                functional: Section6FinalOpeningFunctional::Point,
                evaluation: &field_eval_witness,
                evaluation_commitment: Section6EvaluationCommitment::Direct(
                    field_eval_witness.commitment,
                ),
            },
            Section6FinalOpeningProverClaim {
                data: Section6FinalOpeningData::Field(&short_values),
                committed: &short_commitment,
                point_msb: &short_point,
                functional: Section6FinalOpeningFunctional::Point,
                evaluation: &short_eval_witness,
                evaluation_commitment: Section6EvaluationCommitment::Direct(
                    short_eval_witness.commitment,
                ),
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
                functional: Section6FinalOpeningFunctional::Point,
                evaluation_commitment: Section6EvaluationCommitment::Direct(
                    field_eval_witness.commitment,
                ),
            },
            Section6FinalOpeningVerifierClaim {
                commitment: &short_commitment.commitment,
                logical_len: short_commitment.logical_len,
                point_msb: &short_point,
                functional: Section6FinalOpeningFunctional::Point,
                evaluation_commitment: Section6EvaluationCommitment::Direct(
                    short_eval_witness.commitment,
                ),
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
                functional: Section6FinalOpeningFunctional::Point,
                evaluation: &evaluation_witness,
                evaluation_commitment: Section6EvaluationCommitment::Direct(
                    evaluation_witness.commitment,
                ),
            }],
            &mut rng,
        )
        .unwrap();

        let valid_claim = Section6FinalOpeningVerifierClaim {
            commitment: &committed.commitment,
            logical_len: committed.logical_len,
            point_msb: &point,
            functional: Section6FinalOpeningFunctional::Point,
            evaluation_commitment: Section6EvaluationCommitment::Direct(
                evaluation_witness.commitment,
            ),
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
            functional: Section6FinalOpeningFunctional::Point,
            evaluation_commitment: Section6EvaluationCommitment::Direct(
                evaluation_witness.commitment,
            ),
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
            functional: Section6FinalOpeningFunctional::Point,
            evaluation_commitment: Section6EvaluationCommitment::Direct(bad_commitment),
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
            functional: Section6FinalOpeningFunctional::Point,
            evaluation_commitment: Section6EvaluationCommitment::Direct(
                evaluation_witness.commitment,
            ),
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
