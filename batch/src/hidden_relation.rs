//! Zero-knowledge algebraic relations over Dory-committed field tables.
//!
//! A relation is proved by a committed sumcheck of a random-point zero test.
//! Dory opens every operand at the final sumcheck point to hidden Pedersen
//! commitments, and BlindFold checks the final polynomial identity without
//! revealing the operand evaluations or their blindings.

use std::{
    collections::BTreeMap,
    fmt::{Display, Formatter},
    sync::Arc,
};

use ark_bn254::Fr as ArkFr;
use ark_ff::{One, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::rand::RngCore;
use jolt_blindfold::{
    prove as prove_blindfold, BlindFoldProof, BlindFoldProtocol, BlindFoldWitness,
};
use jolt_claims::{constant, opening, Expr};
use jolt_crypto::{Bn254G1, HomomorphicCommitment, Pedersen, VectorCommitment};
use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_sumcheck::{
    CommittedSumcheckBuilder, CommittedSumcheckProof, SumcheckDomainSpec, SumcheckProof,
    SumcheckStatement,
};
use jolt_transcript::domain::{Label, LabelWithCount, U64Word};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    jolt_dory::{
        commitment_binding_bytes, open_hidden_field_vectors_batch_zk,
        verify_hidden_field_vectors_batch_zk, CommittedDoryFieldMatrix, CommittedDoryFieldVector,
        DoryBackendError, DoryProverParameters, DoryVerifierParameters, HiddenFieldVectorDoryProof,
    },
    jolt_dory_adapter::DenseI8DoryError,
};

/// A polynomial expression over relation operands.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub enum RelationExpression {
    Constant(Fr),
    Table(usize),
    Add(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
enum RelationCheck {
    Identity,
    Sum,
}

impl RelationExpression {
    pub fn constant(value: ArkFr) -> Self {
        Self::Constant(Fr::from(value))
    }

    pub fn table(index: usize) -> Self {
        Self::Table(index)
    }

    pub fn add(self, rhs: Self) -> Self {
        Self::Add(Box::new(self), Box::new(rhs))
    }

    pub fn sub(self, rhs: Self) -> Self {
        Self::Sub(Box::new(self), Box::new(rhs))
    }

    pub fn mul(self, rhs: Self) -> Self {
        Self::Mul(Box::new(self), Box::new(rhs))
    }

    fn degree(&self) -> usize {
        match self {
            Self::Constant(_) => 0,
            Self::Table(_) => 1,
            Self::Add(left, right) | Self::Sub(left, right) => left.degree().max(right.degree()),
            Self::Mul(left, right) => left.degree() + right.degree(),
        }
    }

    fn validate(&self, table_count: usize) -> Result<(), HiddenRelationError> {
        match self {
            Self::Constant(_) => Ok(()),
            Self::Table(index) if *index < table_count => Ok(()),
            Self::Table(index) => Err(HiddenRelationError::InvalidTableIndex {
                index: *index,
                table_count,
            }),
            Self::Add(left, right) | Self::Sub(left, right) | Self::Mul(left, right) => {
                left.validate(table_count)?;
                right.validate(table_count)
            }
        }
    }

    fn evaluate(&self, table_values: &[Fr]) -> Fr {
        match self {
            Self::Constant(value) => *value,
            Self::Table(index) => table_values[*index],
            Self::Add(left, right) => left.evaluate(table_values) + right.evaluate(table_values),
            Self::Sub(left, right) => left.evaluate(table_values) - right.evaluate(table_values),
            Self::Mul(left, right) => left.evaluate(table_values) * right.evaluate(table_values),
        }
    }

    fn claim_expression(&self) -> Expr<Fr, usize> {
        match self {
            Self::Constant(value) => constant(*value),
            Self::Table(index) => opening(*index),
            Self::Add(left, right) => left.claim_expression() + right.claim_expression(),
            Self::Sub(left, right) => left.claim_expression() - right.claim_expression(),
            Self::Mul(left, right) => left.claim_expression() * right.claim_expression(),
        }
    }
}

/// One prover-side operand and its existing hiding Dory commitment.
#[derive(Clone, Copy)]
pub struct HiddenRelationTable<'a> {
    pub values: &'a [ArkFr],
    pub committed: &'a CommittedDoryFieldVector,
}

/// One verifier-known operand table.
///
/// Expression table indices list hidden tables first and public tables second.
#[derive(Clone, Copy)]
pub struct PublicRelationTable<'a> {
    pub values: &'a [ArkFr],
}

/// The verifier-side commitment metadata for one relation operand.
#[derive(Clone, Copy)]
pub struct HiddenRelationTableStatement<'a> {
    pub commitment: &'a jolt_dory::DoryCommitment,
    pub logical_len: usize,
    pub matrix: Option<HiddenMatrixColumn>,
}

/// Position of one logical column in a single column-major matrix commitment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HiddenMatrixColumn {
    pub padded_cols: usize,
    pub column: usize,
}

impl<'a> HiddenRelationTableStatement<'a> {
    pub fn vector(commitment: &'a jolt_dory::DoryCommitment, logical_len: usize) -> Self {
        Self {
            commitment,
            logical_len,
            matrix: None,
        }
    }

    pub fn matrix_column(
        commitment: &'a jolt_dory::DoryCommitment,
        logical_len: usize,
        padded_cols: usize,
        column: usize,
    ) -> Self {
        Self {
            commitment,
            logical_len,
            matrix: Some(HiddenMatrixColumn {
                padded_cols,
                column,
            }),
        }
    }
}

impl<'a> From<&'a CommittedDoryFieldVector> for HiddenRelationTableStatement<'a> {
    fn from(committed: &'a CommittedDoryFieldVector) -> Self {
        Self {
            commitment: &committed.commitment,
            logical_len: committed.logical_len,
            matrix: committed.matrix.as_ref().map(|matrix| HiddenMatrixColumn {
                padded_cols: matrix.padded_cols,
                column: committed.matrix_column,
            }),
        }
    }
}

/// Native Dory batch opening for the standalone-vector operands of a relation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenVectorDoryBatchProof {
    pub opening: HiddenFieldVectorDoryProof,
    pub blind_delta: Fr,
}

/// Native Dory batch opening for all whole matrices of one padded width.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenMatrixDoryBatchProof {
    pub padded_cols: usize,
    pub column_evaluation_commitments: Vec<Vec<Bn254G1>>,
    pub opening: HiddenFieldVectorDoryProof,
    pub blind_delta: Fr,
}

/// A complete hidden zero-test proof for one polynomial relation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenRelationProof {
    pub sumcheck: CommittedSumcheckProof<Bn254G1>,
    pub vector_dory_batch: Option<HiddenVectorDoryBatchProof>,
    pub matrix_dory_batches: Vec<HiddenMatrixDoryBatchProof>,
    pub evaluation_commitments: Vec<Bn254G1>,
    pub blindfold_transcript_state: [u8; 32],
    pub blindfold: BlindFoldProof<Fr, Bn254G1>,
}

/// Sumcheck and BlindFold portion of a hidden relation proof.
///
/// The PCS openings are deliberately kept out of this object so callers can
/// collect every terminal MLE claim before opening commitments in a batch.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenRelationReductionProof {
    pub sumcheck: CommittedSumcheckProof<Bn254G1>,
    pub evaluation_commitments: Vec<Bn254G1>,
    pub blindfold_transcript_state: [u8; 32],
    pub blindfold: BlindFoldProof<Fr, Bn254G1>,
}

/// PCS part of a hidden relation proof, produced only after reduction.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenRelationOpeningProof {
    pub vector_dory_batch: Option<HiddenVectorDoryBatchProof>,
    pub matrix_dory_batches: Vec<HiddenMatrixDoryBatchProof>,
}

/// One common-point group in a phase-level batch opening.
///
/// `relation_indices` is the deterministic, input-order partition of the
/// deferred reductions. `opening_binding` commits to the ordered outer
/// bindings of those reductions and is also used as the native Dory batch
/// transcript binding.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenRelationOpeningBatchGroupProof {
    pub relation_indices: Vec<usize>,
    pub opening_binding: [u8; 32],
    pub opening: HiddenRelationOpeningProof,
}

/// Phase-level PCS openings for a collection of deferred hidden relations.
///
/// Relations with the same logical length and terminal row point share one
/// native Dory vector batch and one native Dory batch per matrix width.
#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenRelationOpeningBatchProof {
    pub groups: Vec<HiddenRelationOpeningBatchGroupProof>,
}

/// Prover-only terminal MLE claims retained until the phase batch opening.
#[derive(Clone)]
pub struct HiddenRelationProverOpening<'a> {
    hidden_tables: Vec<HiddenRelationTable<'a>>,
    logical_len: usize,
    row_point: Vec<Fr>,
    evaluations: Vec<Fr>,
    evaluation_commitments: Vec<Bn254G1>,
    evaluation_blinds: Vec<Fr>,
    outer_binding: Vec<u8>,
}

/// Prover-owned terminal claims without references to the committed tables.
///
/// Protocols that create auxiliary tables can retain this small object next to
/// their owned auxiliary witnesses, then bind fresh table views immediately
/// before the phase-level Dory opening. This avoids self-referential state and
/// does not clone the source or auxiliary vectors.
#[derive(Clone)]
pub struct HiddenRelationProverOpeningClaims {
    logical_len: usize,
    row_point: Vec<Fr>,
    evaluations: Vec<Fr>,
    evaluation_commitments: Vec<Bn254G1>,
    evaluation_blinds: Vec<Fr>,
    outer_binding: Vec<u8>,
}

impl<'a> HiddenRelationProverOpening<'a> {
    pub fn hidden_tables(&self) -> &[HiddenRelationTable<'a>] {
        &self.hidden_tables
    }

    pub fn logical_len(&self) -> usize {
        self.logical_len
    }

    pub fn row_point(&self) -> &[Fr] {
        &self.row_point
    }

    pub fn evaluations(&self) -> &[Fr] {
        &self.evaluations
    }

    pub fn evaluation_commitments(&self) -> &[Bn254G1] {
        &self.evaluation_commitments
    }

    pub fn evaluation_blinds(&self) -> &[Fr] {
        &self.evaluation_blinds
    }

    pub fn outer_binding(&self) -> &[u8] {
        &self.outer_binding
    }

    /// Detach the small terminal claims from their borrowed committed tables.
    pub fn into_claims(self) -> HiddenRelationProverOpeningClaims {
        HiddenRelationProverOpeningClaims {
            logical_len: self.logical_len,
            row_point: self.row_point,
            evaluations: self.evaluations,
            evaluation_commitments: self.evaluation_commitments,
            evaluation_blinds: self.evaluation_blinds,
            outer_binding: self.outer_binding,
        }
    }
}

impl HiddenRelationProverOpeningClaims {
    pub fn logical_len(&self) -> usize {
        self.logical_len
    }

    pub fn row_point(&self) -> &[Fr] {
        &self.row_point
    }

    pub fn evaluations(&self) -> &[Fr] {
        &self.evaluations
    }

    pub fn evaluation_commitments(&self) -> &[Bn254G1] {
        &self.evaluation_commitments
    }

    pub fn evaluation_blinds(&self) -> &[Fr] {
        &self.evaluation_blinds
    }

    pub fn outer_binding(&self) -> &[u8] {
        &self.outer_binding
    }

    pub fn table_count(&self) -> usize {
        self.evaluations.len()
    }

    /// Reattach the exact committed tables in reduction order.
    pub fn bind_tables<'a>(
        &self,
        hidden_tables: Vec<HiddenRelationTable<'a>>,
    ) -> Result<HiddenRelationProverOpening<'a>, HiddenRelationError> {
        let pending = HiddenRelationProverOpening {
            hidden_tables,
            logical_len: self.logical_len,
            row_point: self.row_point.clone(),
            evaluations: self.evaluations.clone(),
            evaluation_commitments: self.evaluation_commitments.clone(),
            evaluation_blinds: self.evaluation_blinds.clone(),
            outer_binding: self.outer_binding.clone(),
        };
        validate_prover_opening(&pending)?;
        Ok(pending)
    }
}

/// Commitment-only verifier claims retained until the phase batch opening.
#[derive(Clone)]
pub struct HiddenRelationVerifierOpening<'a> {
    hidden_tables: Vec<HiddenRelationTableStatement<'a>>,
    logical_len: usize,
    row_point: Vec<Fr>,
    evaluation_commitments: Vec<Bn254G1>,
    outer_binding: Vec<u8>,
}

/// Commitment-only terminal claims detached from borrowed table descriptors.
#[derive(Clone)]
pub struct HiddenRelationVerifierOpeningClaims {
    logical_len: usize,
    row_point: Vec<Fr>,
    evaluation_commitments: Vec<Bn254G1>,
    outer_binding: Vec<u8>,
}

impl<'a> HiddenRelationVerifierOpening<'a> {
    pub fn hidden_tables(&self) -> &[HiddenRelationTableStatement<'a>] {
        &self.hidden_tables
    }

    pub fn logical_len(&self) -> usize {
        self.logical_len
    }

    pub fn row_point(&self) -> &[Fr] {
        &self.row_point
    }

    pub fn evaluation_commitments(&self) -> &[Bn254G1] {
        &self.evaluation_commitments
    }

    pub fn outer_binding(&self) -> &[u8] {
        &self.outer_binding
    }

    /// Detach terminal claims so a caller can own the table descriptors itself.
    pub fn into_claims(self) -> HiddenRelationVerifierOpeningClaims {
        HiddenRelationVerifierOpeningClaims {
            logical_len: self.logical_len,
            row_point: self.row_point,
            evaluation_commitments: self.evaluation_commitments,
            outer_binding: self.outer_binding,
        }
    }
}

impl HiddenRelationVerifierOpeningClaims {
    pub fn logical_len(&self) -> usize {
        self.logical_len
    }

    pub fn row_point(&self) -> &[Fr] {
        &self.row_point
    }

    pub fn evaluation_commitments(&self) -> &[Bn254G1] {
        &self.evaluation_commitments
    }

    pub fn outer_binding(&self) -> &[u8] {
        &self.outer_binding
    }

    pub fn table_count(&self) -> usize {
        self.evaluation_commitments.len()
    }

    /// Reattach commitment metadata in the exact reduction order.
    pub fn bind_tables<'a>(
        &self,
        hidden_tables: Vec<HiddenRelationTableStatement<'a>>,
    ) -> Result<HiddenRelationVerifierOpening<'a>, HiddenRelationError> {
        let pending = HiddenRelationVerifierOpening {
            hidden_tables,
            logical_len: self.logical_len,
            row_point: self.row_point.clone(),
            evaluation_commitments: self.evaluation_commitments.clone(),
            outer_binding: self.outer_binding.clone(),
        };
        validate_verifier_opening(&pending)?;
        Ok(pending)
    }
}

/// One hidden scalar and the Pedersen opening retained by the prover.
#[derive(Clone, Debug)]
pub struct HiddenScalarWitness {
    pub value: ArkFr,
    pub commitment: Bn254G1,
    pub blind: Fr,
}

impl HiddenScalarWitness {
    pub fn commit(parameters: &DoryProverParameters, value: ArkFr, rng: &mut impl RngCore) -> Self {
        let blind = Fr::random(rng);
        let commitment =
            Pedersen::<Bn254G1>::commit(&parameters.blindfold, &[Fr::from(value)], &blind);
        Self {
            value,
            commitment,
            blind,
        }
    }
}

/// BlindFold proof of a polynomial relation among Pedersen-committed scalars.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenScalarRelationProof {
    pub sumcheck: CommittedSumcheckProof<Bn254G1>,
    pub blindfold_transcript_state: [u8; 32],
    pub blindfold: BlindFoldProof<Fr, Bn254G1>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HiddenRelationError {
    EmptyTables,
    InvalidLogicalLength(usize),
    TableLength {
        table: usize,
        expected: usize,
        actual: usize,
    },
    InvalidTableIndex {
        index: usize,
        table_count: usize,
    },
    PedersenCapacity {
        required: usize,
        available: usize,
    },
    NonZeroRelation,
    ProofShape,
    Dory(DenseI8DoryError),
    Sumcheck(String),
    BlindFold(String),
    Encoding(String),
}

impl Display for HiddenRelationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for HiddenRelationError {}

impl From<DenseI8DoryError> for HiddenRelationError {
    fn from(error: DenseI8DoryError) -> Self {
        Self::Dory(error)
    }
}

impl From<DoryBackendError> for HiddenRelationError {
    fn from(error: DoryBackendError) -> Self {
        Self::Dory(DenseI8DoryError::Dory(error.to_string()))
    }
}

/// Prove that the multilinear extension of `expression` is identically zero.
pub fn prove_hidden_relation(
    label: &[u8],
    parameters: &DoryProverParameters,
    tables: &[HiddenRelationTable<'_>],
    expression: &RelationExpression,
    rng: &mut impl RngCore,
) -> Result<HiddenRelationProof, HiddenRelationError> {
    prove_hidden_relation_with_public(label, parameters, tables, &[], expression, rng)
}

/// Prove an identity relation over hidden operands followed by verifier-known operands.
pub fn prove_hidden_relation_with_public(
    label: &[u8],
    parameters: &DoryProverParameters,
    hidden_tables: &[HiddenRelationTable<'_>],
    public_tables: &[PublicRelationTable<'_>],
    expression: &RelationExpression,
    rng: &mut impl RngCore,
) -> Result<HiddenRelationProof, HiddenRelationError> {
    prove_hidden_relation_complete_with_check(
        label,
        parameters,
        hidden_tables,
        public_tables,
        expression,
        RelationCheck::Identity,
        rng,
    )
}

/// Prove that the Boolean-hypercube sum of `expression` is zero.
pub fn prove_hidden_sum_relation(
    label: &[u8],
    parameters: &DoryProverParameters,
    tables: &[HiddenRelationTable<'_>],
    expression: &RelationExpression,
    rng: &mut impl RngCore,
) -> Result<HiddenRelationProof, HiddenRelationError> {
    prove_hidden_sum_relation_with_public(label, parameters, tables, &[], expression, rng)
}

/// Prove a Boolean-hypercube sum relation with verifier-known operand tables.
pub fn prove_hidden_sum_relation_with_public(
    label: &[u8],
    parameters: &DoryProverParameters,
    hidden_tables: &[HiddenRelationTable<'_>],
    public_tables: &[PublicRelationTable<'_>],
    expression: &RelationExpression,
    rng: &mut impl RngCore,
) -> Result<HiddenRelationProof, HiddenRelationError> {
    prove_hidden_relation_complete_with_check(
        label,
        parameters,
        hidden_tables,
        public_tables,
        expression,
        RelationCheck::Sum,
        rng,
    )
}

/// Reduce an identity relation without opening any Dory commitment.
pub fn prove_hidden_relation_deferred<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    tables: &[HiddenRelationTable<'a>],
    expression: &RelationExpression,
    rng: &mut impl RngCore,
) -> Result<
    (
        HiddenRelationReductionProof,
        HiddenRelationProverOpening<'a>,
    ),
    HiddenRelationError,
> {
    prove_hidden_relation_deferred_with_check(
        label,
        parameters,
        tables,
        &[],
        expression,
        RelationCheck::Identity,
        rng,
    )
}

/// Reduce an identity relation with verifier-known tables without PCS opening.
pub fn prove_hidden_relation_deferred_with_public<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    hidden_tables: &[HiddenRelationTable<'a>],
    public_tables: &[PublicRelationTable<'_>],
    expression: &RelationExpression,
    rng: &mut impl RngCore,
) -> Result<
    (
        HiddenRelationReductionProof,
        HiddenRelationProverOpening<'a>,
    ),
    HiddenRelationError,
> {
    prove_hidden_relation_deferred_with_check(
        label,
        parameters,
        hidden_tables,
        public_tables,
        expression,
        RelationCheck::Identity,
        rng,
    )
}

/// Reduce a sum relation without opening any Dory commitment.
pub fn prove_hidden_sum_relation_deferred<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    tables: &[HiddenRelationTable<'a>],
    expression: &RelationExpression,
    rng: &mut impl RngCore,
) -> Result<
    (
        HiddenRelationReductionProof,
        HiddenRelationProverOpening<'a>,
    ),
    HiddenRelationError,
> {
    prove_hidden_relation_deferred_with_check(
        label,
        parameters,
        tables,
        &[],
        expression,
        RelationCheck::Sum,
        rng,
    )
}

/// Reduce a sum relation with verifier-known tables without PCS opening.
pub fn prove_hidden_sum_relation_deferred_with_public<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    hidden_tables: &[HiddenRelationTable<'a>],
    public_tables: &[PublicRelationTable<'_>],
    expression: &RelationExpression,
    rng: &mut impl RngCore,
) -> Result<
    (
        HiddenRelationReductionProof,
        HiddenRelationProverOpening<'a>,
    ),
    HiddenRelationError,
> {
    prove_hidden_relation_deferred_with_check(
        label,
        parameters,
        hidden_tables,
        public_tables,
        expression,
        RelationCheck::Sum,
        rng,
    )
}

fn prove_hidden_relation_complete_with_check(
    label: &[u8],
    parameters: &DoryProverParameters,
    hidden_tables: &[HiddenRelationTable<'_>],
    public_tables: &[PublicRelationTable<'_>],
    expression: &RelationExpression,
    check: RelationCheck,
    rng: &mut impl RngCore,
) -> Result<HiddenRelationProof, HiddenRelationError> {
    let (reduction, pending) = prove_hidden_relation_deferred_with_check(
        label,
        parameters,
        hidden_tables,
        public_tables,
        expression,
        check,
        rng,
    )?;
    let opening = prove_hidden_relation_opening(parameters, pending, rng)?;
    Ok(HiddenRelationProof {
        sumcheck: reduction.sumcheck,
        vector_dory_batch: opening.vector_dory_batch,
        matrix_dory_batches: opening.matrix_dory_batches,
        evaluation_commitments: reduction.evaluation_commitments,
        blindfold_transcript_state: reduction.blindfold_transcript_state,
        blindfold: reduction.blindfold,
    })
}

fn prove_hidden_relation_deferred_with_check<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    hidden_tables: &[HiddenRelationTable<'a>],
    public_tables: &[PublicRelationTable<'_>],
    expression: &RelationExpression,
    check: RelationCheck,
    rng: &mut impl RngCore,
) -> Result<
    (
        HiddenRelationReductionProof,
        HiddenRelationProverOpening<'a>,
    ),
    HiddenRelationError,
> {
    let logical_len = validate_prover_tables(hidden_tables, public_tables, expression)?;
    let table_count = hidden_tables.len() + public_tables.len();
    let num_vars = logical_len.trailing_zeros() as usize;
    let degree = expression.degree() + usize::from(check == RelationCheck::Identity);
    if degree == 0 {
        return Err(HiddenRelationError::ProofShape);
    }
    validate_capacity(
        parameters.blindfold.message_generators.len(),
        degree,
        table_count,
    )?;

    let statements = hidden_tables
        .iter()
        .map(|table| HiddenRelationTableStatement::from(table.committed))
        .collect::<Vec<_>>();
    let public_digests = public_table_digests(public_tables)?;
    let mut transcript = relation_transcript(
        label,
        &statements,
        &public_digests,
        expression,
        logical_len,
        check,
    )?;
    let zero_test_point = if check == RelationCheck::Identity {
        transcript.challenge_vector(num_vars)
    } else {
        Vec::new()
    };
    let mut current_tables = hidden_tables
        .iter()
        .map(|table| {
            table
                .values
                .iter()
                .copied()
                .map(Fr::from)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    current_tables.extend(public_tables.iter().map(|table| {
        table
            .values
            .iter()
            .copied()
            .map(Fr::from)
            .collect::<Vec<_>>()
    }));
    let mut current_eq = if check == RelationCheck::Identity {
        EqPolynomial::<Fr>::evals(&zero_test_point, None)
    } else {
        vec![Fr::one(); logical_len]
    };
    let mut running_claim = Fr::zero();
    let mut sumcheck_builder =
        CommittedSumcheckBuilder::<Fr, Pedersen<Bn254G1>, _>::new(&parameters.blindfold, &mut *rng)
            .map_err(|error| HiddenRelationError::Sumcheck(error.to_string()))?;
    let mut sumcheck_point = Vec::with_capacity(num_vars);

    for _ in 0..num_vars {
        let round = relation_round_polynomial(&current_tables, &current_eq, expression, degree)?;
        if round.evaluate(Fr::zero()) + round.evaluate(Fr::one()) != running_claim {
            return Err(HiddenRelationError::NonZeroRelation);
        }
        let challenge = sumcheck_builder
            .commit_round(&round, &mut transcript)
            .map_err(|error| HiddenRelationError::Sumcheck(error.to_string()))?;
        running_claim = round.evaluate(challenge);
        bind_all(&mut current_tables, challenge)?;
        bind_one(&mut current_eq, challenge)?;
        sumcheck_point.push(challenge);
    }

    let evaluations = current_tables
        .iter()
        .map(|table| {
            table
                .first()
                .copied()
                .ok_or(HiddenRelationError::ProofShape)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (sumcheck, sumcheck_witness) = sumcheck_builder
        .finish(&evaluations, &mut transcript)
        .map_err(|error| HiddenRelationError::Sumcheck(error.to_string()))?;
    let SumcheckProof::Committed(sumcheck) = sumcheck else {
        return Err(HiddenRelationError::ProofShape);
    };

    let outer_binding = relation_outer_binding(
        label,
        &statements,
        &public_digests,
        expression,
        logical_len,
        &zero_test_point,
        &sumcheck,
        check,
    )?;
    let mut evaluation_commitments = Vec::with_capacity(table_count);
    let mut evaluation_blinds = Vec::with_capacity(table_count);
    for (table_index, &evaluation) in evaluations[..hidden_tables.len()].iter().enumerate() {
        if let Some(previous) = (0..table_index).find(|&previous| {
            same_committed_matrix_column(
                hidden_tables[previous].committed,
                hidden_tables[table_index].committed,
            )
        }) {
            if evaluations[previous] != evaluation {
                return Err(HiddenRelationError::ProofShape);
            }
            evaluation_commitments.push(evaluation_commitments[previous]);
            evaluation_blinds.push(evaluation_blinds[previous]);
            continue;
        }
        let evaluation_blind = Fr::random(&mut *rng);
        evaluation_commitments.push(Pedersen::<Bn254G1>::commit(
            &parameters.blindfold,
            &[evaluation],
            &evaluation_blind,
        ));
        evaluation_blinds.push(evaluation_blind);
    }
    for &evaluation in &evaluations[hidden_tables.len()..] {
        evaluation_commitments.push(Pedersen::<Bn254G1>::commit(
            &parameters.blindfold,
            &[evaluation],
            &Fr::zero(),
        ));
        evaluation_blinds.push(Fr::zero());
    }
    let opening_coefficients =
        bind_evaluation_commitments(&mut transcript, &evaluation_commitments);
    let final_evaluation = combine_scalars(&evaluations, &opening_coefficients);
    let final_evaluation_blind = combine_scalars(&evaluation_blinds, &opening_coefficients);

    let protocol = build_protocol(
        expression,
        degree,
        num_vars,
        parameters.blindfold.message_generators.len(),
        &zero_test_point,
        &sumcheck_point,
        &sumcheck,
        &evaluation_commitments,
        &opening_coefficients,
        check,
        &mut relation_transcript(
            label,
            &statements,
            &public_digests,
            expression,
            logical_len,
            check,
        )?,
    )?;
    let assigned = protocol
        .assign_witness(
            &[SumcheckDomainSpec::BooleanHypercube],
            &[&sumcheck_witness],
            &[final_evaluation],
            &[final_evaluation_blind],
            &mut *rng,
        )
        .map_err(|error| HiddenRelationError::BlindFold(error.to_string()))?;
    let blindfold_transcript_state = transcript.state();
    let blindfold = prove_blindfold::<Fr, Pedersen<Bn254G1>, _, _>(
        &parameters.blindfold,
        &protocol,
        &mut transcript,
        BlindFoldWitness {
            rows: &assigned.rows,
            blindings: &assigned.blindings,
            eval_outputs: &[final_evaluation],
            eval_blindings: &[final_evaluation_blind],
        },
        &mut *rng,
    )
    .map_err(|error| HiddenRelationError::BlindFold(error.to_string()))?;

    let hidden_evaluation_commitments = evaluation_commitments[..hidden_tables.len()].to_vec();
    let hidden_evaluation_blinds = evaluation_blinds[..hidden_tables.len()].to_vec();
    let hidden_evaluations = evaluations[..hidden_tables.len()].to_vec();
    Ok((
        HiddenRelationReductionProof {
            sumcheck,
            evaluation_commitments,
            blindfold_transcript_state,
            blindfold,
        },
        HiddenRelationProverOpening {
            hidden_tables: hidden_tables.to_vec(),
            logical_len,
            row_point: sumcheck_point,
            evaluations: hidden_evaluations,
            evaluation_commitments: hidden_evaluation_commitments,
            evaluation_blinds: hidden_evaluation_blinds,
            outer_binding,
        },
    ))
}

/// Open all terminal commitments emitted by one deferred relation reduction.
pub fn prove_hidden_relation_opening(
    parameters: &DoryProverParameters,
    pending: HiddenRelationProverOpening<'_>,
    rng: &mut impl RngCore,
) -> Result<HiddenRelationOpeningProof, HiddenRelationError> {
    let (vector_dory_batch, matrix_dory_batches) = prove_relation_dory_batches(
        parameters,
        &pending.hidden_tables,
        &pending.row_point,
        &pending.evaluations,
        &pending.evaluation_commitments,
        &pending.evaluation_blinds,
        &pending.outer_binding,
        rng,
    )?;
    Ok(HiddenRelationOpeningProof {
        vector_dory_batch,
        matrix_dory_batches,
    })
}

/// Batch-open every terminal PCS claim collected during one protocol phase.
///
/// The input order is part of the statement. Pending relations are grouped by
/// identical `(logical_len, row_point)` keys, with groups ordered by the first
/// occurrence of each key and constituents ordered exactly as supplied.
pub fn prove_hidden_relation_openings_batch(
    parameters: &DoryProverParameters,
    pending: &[HiddenRelationProverOpening<'_>],
    rng: &mut impl RngCore,
) -> Result<HiddenRelationOpeningBatchProof, HiddenRelationError> {
    for opening in pending {
        validate_prover_opening(opening)?;
    }
    let groups = opening_batch_groups(
        pending
            .iter()
            .map(|opening| (opening.logical_len, opening.row_point.as_slice())),
    )?;
    let mut proofs = Vec::with_capacity(groups.len());
    for group in groups {
        let mut hidden_tables = Vec::new();
        let mut evaluations = Vec::new();
        let mut evaluation_commitments = Vec::new();
        let mut evaluation_blinds = Vec::new();
        let mut outer_bindings = Vec::with_capacity(group.relation_indices.len());
        for &relation_index in &group.relation_indices {
            let opening = &pending[relation_index];
            hidden_tables.extend(opening.hidden_tables.iter().copied());
            evaluations.extend_from_slice(&opening.evaluations);
            evaluation_commitments.extend_from_slice(&opening.evaluation_commitments);
            evaluation_blinds.extend_from_slice(&opening.evaluation_blinds);
            outer_bindings.push(opening.outer_binding.as_slice());
        }
        let opening_binding = relation_opening_batch_binding(
            group.logical_len,
            &group.row_point,
            &group.relation_indices,
            &outer_bindings,
        )?;
        let (vector_dory_batch, matrix_dory_batches) = prove_relation_dory_batches(
            parameters,
            &hidden_tables,
            &group.row_point,
            &evaluations,
            &evaluation_commitments,
            &evaluation_blinds,
            &opening_binding,
            rng,
        )?;
        proofs.push(HiddenRelationOpeningBatchGroupProof {
            relation_indices: group.relation_indices,
            opening_binding,
            opening: HiddenRelationOpeningProof {
                vector_dory_batch,
                matrix_dory_batches,
            },
        });
    }
    Ok(HiddenRelationOpeningBatchProof { groups: proofs })
}

/// Verify a hidden zero-test without receiving any operand evaluation scalar.
pub fn verify_hidden_relation(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    tables: &[HiddenRelationTableStatement<'_>],
    expression: &RelationExpression,
    proof: &HiddenRelationProof,
) -> Result<(), HiddenRelationError> {
    verify_hidden_relation_with_public(label, parameters, tables, &[], expression, proof)
}

/// Verify an identity relation with verifier-known operand tables.
pub fn verify_hidden_relation_with_public(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    hidden_tables: &[HiddenRelationTableStatement<'_>],
    public_tables: &[PublicRelationTable<'_>],
    expression: &RelationExpression,
    proof: &HiddenRelationProof,
) -> Result<(), HiddenRelationError> {
    verify_hidden_relation_complete_with_check(
        label,
        parameters,
        hidden_tables,
        public_tables,
        expression,
        proof,
        RelationCheck::Identity,
    )
}

/// Verify a hidden Boolean-hypercube sum relation.
pub fn verify_hidden_sum_relation(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    tables: &[HiddenRelationTableStatement<'_>],
    expression: &RelationExpression,
    proof: &HiddenRelationProof,
) -> Result<(), HiddenRelationError> {
    verify_hidden_sum_relation_with_public(label, parameters, tables, &[], expression, proof)
}

/// Verify a Boolean-hypercube sum relation with verifier-known operand tables.
pub fn verify_hidden_sum_relation_with_public(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    hidden_tables: &[HiddenRelationTableStatement<'_>],
    public_tables: &[PublicRelationTable<'_>],
    expression: &RelationExpression,
    proof: &HiddenRelationProof,
) -> Result<(), HiddenRelationError> {
    verify_hidden_relation_complete_with_check(
        label,
        parameters,
        hidden_tables,
        public_tables,
        expression,
        proof,
        RelationCheck::Sum,
    )
}

/// Verify an identity reduction and return its commitment-only PCS claims.
pub fn verify_hidden_relation_deferred<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    tables: &[HiddenRelationTableStatement<'a>],
    expression: &RelationExpression,
    proof: &HiddenRelationReductionProof,
) -> Result<HiddenRelationVerifierOpening<'a>, HiddenRelationError> {
    verify_hidden_relation_deferred_with_check(
        label,
        parameters,
        tables,
        &[],
        expression,
        proof,
        RelationCheck::Identity,
    )
}

/// Verify an identity reduction with public tables and defer its PCS opening.
pub fn verify_hidden_relation_deferred_with_public<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    hidden_tables: &[HiddenRelationTableStatement<'a>],
    public_tables: &[PublicRelationTable<'_>],
    expression: &RelationExpression,
    proof: &HiddenRelationReductionProof,
) -> Result<HiddenRelationVerifierOpening<'a>, HiddenRelationError> {
    verify_hidden_relation_deferred_with_check(
        label,
        parameters,
        hidden_tables,
        public_tables,
        expression,
        proof,
        RelationCheck::Identity,
    )
}

/// Verify a sum reduction and return its commitment-only PCS claims.
pub fn verify_hidden_sum_relation_deferred<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    tables: &[HiddenRelationTableStatement<'a>],
    expression: &RelationExpression,
    proof: &HiddenRelationReductionProof,
) -> Result<HiddenRelationVerifierOpening<'a>, HiddenRelationError> {
    verify_hidden_relation_deferred_with_check(
        label,
        parameters,
        tables,
        &[],
        expression,
        proof,
        RelationCheck::Sum,
    )
}

/// Verify a sum reduction with public tables and defer its PCS opening.
pub fn verify_hidden_sum_relation_deferred_with_public<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    hidden_tables: &[HiddenRelationTableStatement<'a>],
    public_tables: &[PublicRelationTable<'_>],
    expression: &RelationExpression,
    proof: &HiddenRelationReductionProof,
) -> Result<HiddenRelationVerifierOpening<'a>, HiddenRelationError> {
    verify_hidden_relation_deferred_with_check(
        label,
        parameters,
        hidden_tables,
        public_tables,
        expression,
        proof,
        RelationCheck::Sum,
    )
}

fn verify_hidden_relation_complete_with_check(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    hidden_tables: &[HiddenRelationTableStatement<'_>],
    public_tables: &[PublicRelationTable<'_>],
    expression: &RelationExpression,
    proof: &HiddenRelationProof,
    check: RelationCheck,
) -> Result<(), HiddenRelationError> {
    let reduction = HiddenRelationReductionProof {
        sumcheck: proof.sumcheck.clone(),
        evaluation_commitments: proof.evaluation_commitments.clone(),
        blindfold_transcript_state: proof.blindfold_transcript_state,
        blindfold: proof.blindfold.clone(),
    };
    let pending = verify_hidden_relation_deferred_with_check(
        label,
        parameters,
        hidden_tables,
        public_tables,
        expression,
        &reduction,
        check,
    )?;
    verify_hidden_relation_opening(
        parameters,
        pending,
        &HiddenRelationOpeningProof {
            vector_dory_batch: proof.vector_dory_batch.clone(),
            matrix_dory_batches: proof.matrix_dory_batches.clone(),
        },
    )
}

fn verify_hidden_relation_deferred_with_check<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    hidden_tables: &[HiddenRelationTableStatement<'a>],
    public_tables: &[PublicRelationTable<'_>],
    expression: &RelationExpression,
    proof: &HiddenRelationReductionProof,
    check: RelationCheck,
) -> Result<HiddenRelationVerifierOpening<'a>, HiddenRelationError> {
    let logical_len = validate_verifier_tables(hidden_tables, public_tables, expression)?;
    let table_count = hidden_tables.len() + public_tables.len();
    let num_vars = logical_len.trailing_zeros() as usize;
    let degree = expression.degree() + usize::from(check == RelationCheck::Identity);
    if degree == 0 {
        return Err(HiddenRelationError::ProofShape);
    }
    validate_capacity(
        parameters.blindfold.message_generators.len(),
        degree,
        table_count,
    )?;
    if proof.evaluation_commitments.len() != table_count {
        return Err(HiddenRelationError::ProofShape);
    }

    let public_digests = public_table_digests(public_tables)?;
    let mut transcript = relation_transcript(
        label,
        hidden_tables,
        &public_digests,
        expression,
        logical_len,
        check,
    )?;
    let zero_test_point = if check == RelationCheck::Identity {
        transcript.challenge_vector(num_vars)
    } else {
        Vec::new()
    };
    let consistency = SumcheckProof::<Fr, Bn254G1>::Committed(proof.sumcheck.clone())
        .verify_committed_consistency(SumcheckStatement::new(num_vars, degree), &mut transcript)
        .map_err(|error| HiddenRelationError::Sumcheck(error.to_string()))?;
    let sumcheck_point = consistency.challenges();
    let outer_binding = relation_outer_binding(
        label,
        hidden_tables,
        &public_digests,
        expression,
        logical_len,
        &zero_test_point,
        &proof.sumcheck,
        check,
    )?;
    let hidden_evaluation_commitments = &proof.evaluation_commitments[..hidden_tables.len()];
    let mut evaluation_commitments = hidden_evaluation_commitments.to_vec();
    for table in public_tables {
        let evaluation = evaluate_public_table(table.values, &sumcheck_point)?;
        evaluation_commitments.push(Pedersen::<Bn254G1>::commit(
            &parameters.blindfold,
            &[evaluation],
            &Fr::zero(),
        ));
    }
    if evaluation_commitments != proof.evaluation_commitments {
        return Err(HiddenRelationError::Dory(
            DenseI8DoryError::EvaluationCommitmentMismatch,
        ));
    }
    let opening_coefficients =
        bind_evaluation_commitments(&mut transcript, &evaluation_commitments);

    let protocol = build_protocol_from_consistency(
        expression,
        degree,
        num_vars,
        parameters.blindfold.message_generators.len(),
        &zero_test_point,
        &sumcheck_point,
        consistency,
        &proof.sumcheck,
        &evaluation_commitments,
        &opening_coefficients,
        check,
    )?;
    if transcript.state() != proof.blindfold_transcript_state {
        return Err(HiddenRelationError::Encoding(
            "BlindFold transcript prefix mismatch".to_owned(),
        ));
    }
    protocol
        .verify::<Pedersen<Bn254G1>, _>(&proof.blindfold, &parameters.blindfold, &mut transcript)
        .map_err(|error| HiddenRelationError::BlindFold(format!("{error:?}")))?;
    Ok(HiddenRelationVerifierOpening {
        hidden_tables: hidden_tables.to_vec(),
        logical_len,
        row_point: sumcheck_point,
        evaluation_commitments: hidden_evaluation_commitments.to_vec(),
        outer_binding,
    })
}

/// Verify the PCS openings collected from one deferred relation reduction.
pub fn verify_hidden_relation_opening(
    parameters: &DoryVerifierParameters,
    pending: HiddenRelationVerifierOpening<'_>,
    proof: &HiddenRelationOpeningProof,
) -> Result<(), HiddenRelationError> {
    verify_relation_dory_batches(
        parameters,
        &pending.hidden_tables,
        &pending.row_point,
        &pending.evaluation_commitments,
        proof.vector_dory_batch.as_ref(),
        &proof.matrix_dory_batches,
        &pending.outer_binding,
    )
}

/// Verify a phase-level batch of terminal hidden-relation PCS openings.
///
/// No witness evaluation is read. The verifier reconstructs the common-point
/// partition and every ordered group binding from deferred reduction outputs.
pub fn verify_hidden_relation_openings_batch(
    parameters: &DoryVerifierParameters,
    pending: &[HiddenRelationVerifierOpening<'_>],
    proof: &HiddenRelationOpeningBatchProof,
) -> Result<(), HiddenRelationError> {
    for opening in pending {
        validate_verifier_opening(opening)?;
    }
    let groups = opening_batch_groups(
        pending
            .iter()
            .map(|opening| (opening.logical_len, opening.row_point.as_slice())),
    )?;
    if groups.len() != proof.groups.len() {
        return Err(HiddenRelationError::ProofShape);
    }
    for (group, group_proof) in groups.iter().zip(&proof.groups) {
        if group.relation_indices != group_proof.relation_indices {
            return Err(HiddenRelationError::ProofShape);
        }
        let mut hidden_tables = Vec::new();
        let mut evaluation_commitments = Vec::new();
        let mut outer_bindings = Vec::with_capacity(group.relation_indices.len());
        for &relation_index in &group.relation_indices {
            let opening = &pending[relation_index];
            hidden_tables.extend(opening.hidden_tables.iter().copied());
            evaluation_commitments.extend_from_slice(&opening.evaluation_commitments);
            outer_bindings.push(opening.outer_binding.as_slice());
        }
        let opening_binding = relation_opening_batch_binding(
            group.logical_len,
            &group.row_point,
            &group.relation_indices,
            &outer_bindings,
        )?;
        if opening_binding != group_proof.opening_binding {
            return Err(HiddenRelationError::ProofShape);
        }
        verify_relation_dory_batches(
            parameters,
            &hidden_tables,
            &group.row_point,
            &evaluation_commitments,
            group_proof.opening.vector_dory_batch.as_ref(),
            &group_proof.opening.matrix_dory_batches,
            &opening_binding,
        )?;
    }
    Ok(())
}

/// Prove a polynomial identity among independently Pedersen-committed scalars.
///
/// Each scalar is viewed as the constant multilinear table `[value, value]`.
/// A one-round committed sumcheck and BlindFold then enforce the identity while
/// the final-opening bindings tie every scalar to its existing G1 commitment.
pub fn prove_hidden_scalar_relation(
    label: &[u8],
    parameters: &DoryProverParameters,
    witnesses: &[HiddenScalarWitness],
    expression: &RelationExpression,
    rng: &mut impl RngCore,
) -> Result<HiddenScalarRelationProof, HiddenRelationError> {
    if witnesses.is_empty() {
        return Err(HiddenRelationError::EmptyTables);
    }
    expression.validate(witnesses.len())?;
    let degree = expression.degree() + 1;
    validate_capacity(
        parameters.blindfold.message_generators.len(),
        degree,
        witnesses.len(),
    )?;
    if witnesses.iter().any(|witness| {
        !Pedersen::<Bn254G1>::verify(
            &parameters.blindfold,
            &witness.commitment,
            &[Fr::from(witness.value)],
            &witness.blind,
        )
    }) {
        return Err(HiddenRelationError::ProofShape);
    }

    let commitments = witnesses
        .iter()
        .map(|witness| witness.commitment)
        .collect::<Vec<_>>();
    let mut transcript = scalar_relation_transcript(label, &commitments, expression)?;
    let zero_test_point = transcript.challenge_vector(1);
    let mut current_tables = witnesses
        .iter()
        .map(|witness| {
            let value = Fr::from(witness.value);
            vec![value, value]
        })
        .collect::<Vec<_>>();
    let mut current_eq = EqPolynomial::<Fr>::evals(&zero_test_point, None);
    let mut sumcheck_builder =
        CommittedSumcheckBuilder::<Fr, Pedersen<Bn254G1>, _>::new(&parameters.blindfold, &mut *rng)
            .map_err(|error| HiddenRelationError::Sumcheck(error.to_string()))?;
    let round = relation_round_polynomial(&current_tables, &current_eq, expression, degree)?;
    if round.evaluate(Fr::zero()) + round.evaluate(Fr::one()) != Fr::zero() {
        return Err(HiddenRelationError::NonZeroRelation);
    }
    let challenge = sumcheck_builder
        .commit_round(&round, &mut transcript)
        .map_err(|error| HiddenRelationError::Sumcheck(error.to_string()))?;
    bind_all(&mut current_tables, challenge)?;
    bind_one(&mut current_eq, challenge)?;
    let evaluations = current_tables
        .iter()
        .map(|table| {
            table
                .first()
                .copied()
                .ok_or(HiddenRelationError::ProofShape)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let (sumcheck, sumcheck_witness) = sumcheck_builder
        .finish(&evaluations, &mut transcript)
        .map_err(|error| HiddenRelationError::Sumcheck(error.to_string()))?;
    let SumcheckProof::Committed(sumcheck) = sumcheck else {
        return Err(HiddenRelationError::ProofShape);
    };
    let opening_coefficients = bind_evaluation_commitments(&mut transcript, &commitments);
    let final_evaluation = combine_scalars(&evaluations, &opening_coefficients);
    let blinds = witnesses
        .iter()
        .map(|witness| witness.blind)
        .collect::<Vec<_>>();
    let final_evaluation_blind = combine_scalars(&blinds, &opening_coefficients);
    let protocol = build_protocol(
        expression,
        degree,
        1,
        parameters.blindfold.message_generators.len(),
        &zero_test_point,
        &[challenge],
        &sumcheck,
        &commitments,
        &opening_coefficients,
        RelationCheck::Identity,
        &mut scalar_relation_transcript(label, &commitments, expression)?,
    )?;
    let assigned = protocol
        .assign_witness(
            &[SumcheckDomainSpec::BooleanHypercube],
            &[&sumcheck_witness],
            &[final_evaluation],
            &[final_evaluation_blind],
            &mut *rng,
        )
        .map_err(|error| HiddenRelationError::BlindFold(error.to_string()))?;
    let blindfold_transcript_state = transcript.state();
    let blindfold = prove_blindfold::<Fr, Pedersen<Bn254G1>, _, _>(
        &parameters.blindfold,
        &protocol,
        &mut transcript,
        BlindFoldWitness {
            rows: &assigned.rows,
            blindings: &assigned.blindings,
            eval_outputs: &[final_evaluation],
            eval_blindings: &[final_evaluation_blind],
        },
        &mut *rng,
    )
    .map_err(|error| HiddenRelationError::BlindFold(error.to_string()))?;
    Ok(HiddenScalarRelationProof {
        sumcheck,
        blindfold_transcript_state,
        blindfold,
    })
}

/// Verify a hidden scalar relation from its ordered G1 commitments only.
pub fn verify_hidden_scalar_relation(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    commitments: &[Bn254G1],
    expression: &RelationExpression,
    proof: &HiddenScalarRelationProof,
) -> Result<(), HiddenRelationError> {
    if commitments.is_empty() {
        return Err(HiddenRelationError::EmptyTables);
    }
    expression.validate(commitments.len())?;
    let degree = expression.degree() + 1;
    validate_capacity(
        parameters.blindfold.message_generators.len(),
        degree,
        commitments.len(),
    )?;
    let mut transcript = scalar_relation_transcript(label, commitments, expression)?;
    let zero_test_point = transcript.challenge_vector(1);
    let consistency = SumcheckProof::<Fr, Bn254G1>::Committed(proof.sumcheck.clone())
        .verify_committed_consistency(SumcheckStatement::new(1, degree), &mut transcript)
        .map_err(|error| HiddenRelationError::Sumcheck(error.to_string()))?;
    let sumcheck_point = consistency.challenges();
    let opening_coefficients = bind_evaluation_commitments(&mut transcript, commitments);
    let protocol = build_protocol_from_consistency(
        expression,
        degree,
        1,
        parameters.blindfold.message_generators.len(),
        &zero_test_point,
        &sumcheck_point,
        consistency,
        &proof.sumcheck,
        commitments,
        &opening_coefficients,
        RelationCheck::Identity,
    )?;
    if transcript.state() != proof.blindfold_transcript_state {
        return Err(HiddenRelationError::Encoding(
            "BlindFold transcript prefix mismatch".to_owned(),
        ));
    }
    protocol
        .verify::<Pedersen<Bn254G1>, _>(&proof.blindfold, &parameters.blindfold, &mut transcript)
        .map_err(|error| HiddenRelationError::BlindFold(format!("{error:?}")))
}

fn scalar_relation_transcript(
    label: &[u8],
    commitments: &[Bn254G1],
    expression: &RelationExpression,
) -> Result<Blake2bTranscript<Fr>, HiddenRelationError> {
    let mut transcript = Blake2bTranscript::<Fr>::new(b"pc-hidden-scalar-rel-v1");
    transcript.append(&Label(b"hidden_scalar_relation"));
    transcript.append(&LabelWithCount(b"relation_label", label.len() as u64));
    transcript.append_bytes(label);
    transcript.append(&LabelWithCount(
        b"scalar_commitments",
        commitments.len() as u64,
    ));
    for commitment in commitments {
        transcript.append_labeled(b"scalar_commitment", commitment);
    }
    let expression_bytes = bincode::serialize(expression)
        .map_err(|error| HiddenRelationError::Encoding(error.to_string()))?;
    transcript.append(&LabelWithCount(
        b"relation_expression",
        expression_bytes.len() as u64,
    ));
    transcript.append_bytes(&expression_bytes);
    Ok(transcript)
}

fn build_protocol(
    expression: &RelationExpression,
    degree: usize,
    num_vars: usize,
    row_len: usize,
    zero_test_point: &[Fr],
    sumcheck_point: &[Fr],
    sumcheck: &CommittedSumcheckProof<Bn254G1>,
    evaluation_commitments: &[Bn254G1],
    opening_coefficients: &[Fr],
    check: RelationCheck,
    consistency_transcript: &mut Blake2bTranscript<Fr>,
) -> Result<BlindFoldProtocol<Fr, Bn254G1>, HiddenRelationError> {
    if check == RelationCheck::Identity {
        let _ = consistency_transcript.challenge_vector(num_vars);
    }
    let consistency = SumcheckProof::<Fr, Bn254G1>::Committed(sumcheck.clone())
        .verify_committed_consistency(
            SumcheckStatement::new(num_vars, degree),
            consistency_transcript,
        )
        .map_err(|error| HiddenRelationError::Sumcheck(error.to_string()))?;
    let replayed_coefficients =
        bind_evaluation_commitments(consistency_transcript, evaluation_commitments);
    if replayed_coefficients != opening_coefficients {
        return Err(HiddenRelationError::Encoding(
            "relation opening batching challenge mismatch".to_owned(),
        ));
    }
    build_protocol_from_consistency(
        expression,
        degree,
        num_vars,
        row_len,
        zero_test_point,
        sumcheck_point,
        consistency,
        sumcheck,
        evaluation_commitments,
        opening_coefficients,
        check,
    )
}

#[allow(clippy::too_many_arguments)]
fn build_protocol_from_consistency(
    expression: &RelationExpression,
    degree: usize,
    num_vars: usize,
    row_len: usize,
    zero_test_point: &[Fr],
    sumcheck_point: &[Fr],
    consistency: jolt_sumcheck::CommittedSumcheckConsistency<Fr, Bn254G1>,
    sumcheck: &CommittedSumcheckProof<Bn254G1>,
    evaluation_commitments: &[Bn254G1],
    opening_coefficients: &[Fr],
    check: RelationCheck,
) -> Result<BlindFoldProtocol<Fr, Bn254G1>, HiddenRelationError> {
    if opening_coefficients.len() != evaluation_commitments.len() {
        return Err(HiddenRelationError::ProofShape);
    }
    let eq_evaluation = if check == RelationCheck::Identity {
        EqPolynomial::<Fr>::mle(zero_test_point, sumcheck_point)
    } else {
        Fr::one()
    };
    let output_expression = constant(eq_evaluation) * expression.claim_expression();
    let opening_ids = (0..evaluation_commitments.len()).collect::<Vec<_>>();
    let mut builder = BlindFoldProtocol::<Fr, Bn254G1>::builder::<usize, (), usize>()
        .stage("hidden-relation-zero-test")
        .sumcheck(SumcheckStatement::new(num_vars, degree))
        .domain(SumcheckDomainSpec::BooleanHypercube)
        .consistency(consistency)
        .output_claim_rows(opening_ids.clone(), row_len, sumcheck.output_claims.clone())
        .input_claim(constant(Fr::zero()))
        .output_claim(output_expression)
        .finish_stage()
        .map_err(|error| HiddenRelationError::BlindFold(error.to_string()))?;
    builder = builder.final_opening(
        opening_ids,
        opening_coefficients.to_vec(),
        combine_commitments(evaluation_commitments, opening_coefficients),
    );
    builder
        .build()
        .map_err(|error| HiddenRelationError::BlindFold(error.to_string()))
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct HiddenRelationOpeningBatchGroup {
    logical_len: usize,
    row_point: Vec<Fr>,
    relation_indices: Vec<usize>,
}

fn opening_batch_groups<'a>(
    keys: impl IntoIterator<Item = (usize, &'a [Fr])>,
) -> Result<Vec<HiddenRelationOpeningBatchGroup>, HiddenRelationError> {
    let mut groups = Vec::<HiddenRelationOpeningBatchGroup>::new();
    for (relation_index, (logical_len, row_point)) in keys.into_iter().enumerate() {
        if logical_len < 2
            || !logical_len.is_power_of_two()
            || row_point.len() != logical_len.trailing_zeros() as usize
        {
            return Err(HiddenRelationError::ProofShape);
        }
        if let Some(group) = groups.iter_mut().find(|group| {
            group.logical_len == logical_len && group.row_point.as_slice() == row_point
        }) {
            group.relation_indices.push(relation_index);
        } else {
            groups.push(HiddenRelationOpeningBatchGroup {
                logical_len,
                row_point: row_point.to_vec(),
                relation_indices: vec![relation_index],
            });
        }
    }
    Ok(groups)
}

fn validate_prover_opening(
    pending: &HiddenRelationProverOpening<'_>,
) -> Result<(), HiddenRelationError> {
    let table_count = pending.hidden_tables.len();
    if pending.evaluations.len() != table_count
        || pending.evaluation_commitments.len() != table_count
        || pending.evaluation_blinds.len() != table_count
    {
        return Err(HiddenRelationError::ProofShape);
    }
    for table in &pending.hidden_tables {
        if table.values.len() != pending.logical_len
            || table.committed.logical_len != pending.logical_len
        {
            return Err(HiddenRelationError::ProofShape);
        }
    }
    Ok(())
}

fn validate_verifier_opening(
    pending: &HiddenRelationVerifierOpening<'_>,
) -> Result<(), HiddenRelationError> {
    if pending.evaluation_commitments.len() != pending.hidden_tables.len()
        || pending
            .hidden_tables
            .iter()
            .any(|table| table.logical_len != pending.logical_len)
    {
        return Err(HiddenRelationError::ProofShape);
    }
    Ok(())
}

fn relation_opening_batch_binding(
    logical_len: usize,
    row_point: &[Fr],
    relation_indices: &[usize],
    outer_bindings: &[&[u8]],
) -> Result<[u8; 32], HiddenRelationError> {
    if relation_indices.is_empty() || relation_indices.len() != outer_bindings.len() {
        return Err(HiddenRelationError::ProofShape);
    }
    let as_u64 = |value: usize| {
        u64::try_from(value).map_err(|_| {
            HiddenRelationError::Encoding("batch opening length does not fit in u64".to_owned())
        })
    };
    let mut hasher = Sha256::new();
    hasher.update(b"parcomm-hidden-relation-opening-batch-v1");
    hasher.update(as_u64(logical_len)?.to_le_bytes());
    hasher.update(as_u64(row_point.len())?.to_le_bytes());
    let mut coordinate_bytes = Vec::new();
    for &coordinate in row_point {
        coordinate_bytes.clear();
        ArkFr::from(coordinate)
            .serialize_compressed(&mut coordinate_bytes)
            .map_err(|error| HiddenRelationError::Encoding(error.to_string()))?;
        hasher.update(as_u64(coordinate_bytes.len())?.to_le_bytes());
        hasher.update(&coordinate_bytes);
    }
    hasher.update(as_u64(relation_indices.len())?.to_le_bytes());
    for (&relation_index, outer_binding) in relation_indices.iter().zip(outer_bindings) {
        hasher.update(as_u64(relation_index)?.to_le_bytes());
        hasher.update(as_u64(outer_binding.len())?.to_le_bytes());
        hasher.update(outer_binding);
    }
    Ok(hasher.finalize().into())
}

struct ProverMatrixEntry<'a> {
    matrix: &'a Arc<CommittedDoryFieldMatrix>,
    table_indices: Vec<(usize, usize)>,
}

struct ProverMatrixGroup<'a> {
    padded_cols: usize,
    matrices: Vec<ProverMatrixEntry<'a>>,
}

struct VerifierMatrixEntry<'a> {
    commitment: &'a jolt_dory::DoryCommitment,
    table_indices: Vec<(usize, usize)>,
}

struct VerifierMatrixGroup<'a> {
    padded_cols: usize,
    matrices: Vec<VerifierMatrixEntry<'a>>,
}

fn same_committed_matrix_column(
    left: &CommittedDoryFieldVector,
    right: &CommittedDoryFieldVector,
) -> bool {
    match (&left.matrix, &right.matrix) {
        (Some(left_matrix), Some(right_matrix)) => {
            left_matrix.commitment == right_matrix.commitment
                && left.matrix_column == right.matrix_column
        }
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn prove_relation_dory_batches(
    parameters: &DoryProverParameters,
    hidden_tables: &[HiddenRelationTable<'_>],
    row_point: &[Fr],
    evaluations: &[Fr],
    evaluation_commitments: &[Bn254G1],
    evaluation_blinds: &[Fr],
    outer_binding: &[u8],
    rng: &mut impl RngCore,
) -> Result<
    (
        Option<HiddenVectorDoryBatchProof>,
        Vec<HiddenMatrixDoryBatchProof>,
    ),
    HiddenRelationError,
> {
    if evaluations.len() != hidden_tables.len()
        || evaluation_commitments.len() != hidden_tables.len()
        || evaluation_blinds.len() != hidden_tables.len()
    {
        return Err(HiddenRelationError::ProofShape);
    }

    let vector_indices = hidden_tables
        .iter()
        .enumerate()
        .filter_map(|(index, table)| table.committed.matrix.is_none().then_some(index))
        .collect::<Vec<_>>();
    let vector_dory_batch = if vector_indices.is_empty() {
        None
    } else {
        let values = vector_indices
            .iter()
            .map(|&index| hidden_tables[index].values)
            .collect::<Vec<_>>();
        let committed = vector_indices
            .iter()
            .map(|&index| hidden_tables[index].committed)
            .collect::<Vec<_>>();
        let vector_evaluations = vector_indices
            .iter()
            .map(|&index| evaluations[index])
            .collect::<Vec<_>>();
        let vector_commitments = vector_indices
            .iter()
            .map(|&index| evaluation_commitments[index])
            .collect::<Vec<_>>();
        let vector_blinds = vector_indices
            .iter()
            .map(|&index| evaluation_blinds[index])
            .collect::<Vec<_>>();
        let (opening, joint_commitment, joint_blind, coefficients) =
            open_hidden_field_vectors_batch_zk(
                parameters,
                &values,
                &committed,
                row_point,
                &vector_evaluations,
                &vector_commitments,
                b"hidden-relation-vectors",
                outer_binding,
            )?;
        let combined_commitment = combine_commitments(&vector_commitments, &coefficients);
        let combined_blind = combine_scalars(&vector_blinds, &coefficients);
        let blind_delta = joint_blind - combined_blind;
        verify_dory_commitment_link(
            &parameters.blindfold,
            &combined_commitment,
            &joint_commitment,
            blind_delta,
        )?;
        Some(HiddenVectorDoryBatchProof {
            opening,
            blind_delta,
        })
    };

    let row_point_ark = row_point
        .iter()
        .copied()
        .map(ArkFr::from)
        .collect::<Vec<_>>();
    let mut matrix_dory_batches = Vec::new();
    for group in prover_matrix_groups(hidden_tables)? {
        let mut all_column_commitments = Vec::with_capacity(group.matrices.len());
        let mut all_column_blinds = Vec::with_capacity(group.matrices.len());
        let mut all_column_evaluations = Vec::with_capacity(group.matrices.len());
        for entry in &group.matrices {
            let mut column_commitments = Vec::with_capacity(group.padded_cols);
            let mut column_blinds = Vec::with_capacity(group.padded_cols);
            let column_evaluations =
                matrix_column_evaluations(entry, group.padded_cols, evaluations, |values| {
                    crate::jolt_dory::evaluate_field_vector_standard(values, &row_point_ark)
                        .map(Fr::from)
                        .map_err(HiddenRelationError::from)
                })?;
            for (column, &evaluation) in column_evaluations.iter().enumerate() {
                if let Some(&(table_index, _)) = entry
                    .table_indices
                    .iter()
                    .find(|(_, table_column)| *table_column == column)
                {
                    if evaluations[table_index] != evaluation {
                        return Err(HiddenRelationError::ProofShape);
                    }
                    column_commitments.push(evaluation_commitments[table_index]);
                    column_blinds.push(evaluation_blinds[table_index]);
                } else {
                    let blind = Fr::random(&mut *rng);
                    column_commitments.push(Pedersen::<Bn254G1>::commit(
                        &parameters.blindfold,
                        &[evaluation],
                        &blind,
                    ));
                    column_blinds.push(blind);
                }
            }
            all_column_commitments.push(column_commitments);
            all_column_blinds.push(column_blinds);
            all_column_evaluations.push(column_evaluations);
        }

        let matrix_commitments = group
            .matrices
            .iter()
            .map(|entry| &entry.matrix.commitment)
            .collect::<Vec<_>>();
        let column_point = matrix_column_point(
            outer_binding,
            hidden_tables[0].committed.logical_len,
            group.padded_cols,
            row_point,
            &matrix_commitments,
            &all_column_commitments,
        );
        let column_weights = EqPolynomial::<Fr>::evals(&column_point, None);
        let matrix_evaluations = all_column_evaluations
            .iter()
            .map(|values| combine_scalars(values, &column_weights))
            .collect::<Vec<_>>();
        let matrix_evaluation_commitments = all_column_commitments
            .iter()
            .map(|commitments| combine_commitments(commitments, &column_weights))
            .collect::<Vec<_>>();
        let matrix_evaluation_blinds = all_column_blinds
            .iter()
            .map(|blinds| combine_scalars(blinds, &column_weights))
            .collect::<Vec<_>>();
        let full_point = column_point
            .iter()
            .chain(row_point)
            .copied()
            .collect::<Vec<_>>();
        let whole_values = group
            .matrices
            .iter()
            .map(|entry| entry.matrix.values.as_slice())
            .collect::<Vec<_>>();
        let whole_committed = group
            .matrices
            .iter()
            .map(|entry| CommittedDoryFieldVector {
                commitment: entry.matrix.commitment.clone(),
                logical_len: entry.matrix.padded_rows * entry.matrix.padded_cols,
                hint: entry.matrix.hint.clone(),
                matrix: None,
                matrix_column: 0,
                grid_num_vars: entry.matrix.grid_num_vars,
            })
            .collect::<Vec<_>>();
        let whole_committed_refs = whole_committed.iter().collect::<Vec<_>>();
        let (opening, joint_commitment, joint_blind, coefficients) =
            open_hidden_field_vectors_batch_zk(
                parameters,
                &whole_values,
                &whole_committed_refs,
                &full_point,
                &matrix_evaluations,
                &matrix_evaluation_commitments,
                b"hidden-relation-matrices",
                outer_binding,
            )?;
        let combined_commitment =
            combine_commitments(&matrix_evaluation_commitments, &coefficients);
        let combined_blind = combine_scalars(&matrix_evaluation_blinds, &coefficients);
        let blind_delta = joint_blind - combined_blind;
        verify_dory_commitment_link(
            &parameters.blindfold,
            &combined_commitment,
            &joint_commitment,
            blind_delta,
        )?;
        matrix_dory_batches.push(HiddenMatrixDoryBatchProof {
            padded_cols: group.padded_cols,
            column_evaluation_commitments: all_column_commitments,
            opening,
            blind_delta,
        });
    }

    Ok((vector_dory_batch, matrix_dory_batches))
}

#[allow(clippy::too_many_arguments)]
fn verify_relation_dory_batches(
    parameters: &DoryVerifierParameters,
    hidden_tables: &[HiddenRelationTableStatement<'_>],
    row_point: &[Fr],
    evaluation_commitments: &[Bn254G1],
    vector_proof: Option<&HiddenVectorDoryBatchProof>,
    matrix_proofs: &[HiddenMatrixDoryBatchProof],
    outer_binding: &[u8],
) -> Result<(), HiddenRelationError> {
    if evaluation_commitments.len() != hidden_tables.len() {
        return Err(HiddenRelationError::ProofShape);
    }
    let vector_indices = hidden_tables
        .iter()
        .enumerate()
        .filter_map(|(index, table)| table.matrix.is_none().then_some(index))
        .collect::<Vec<_>>();
    match (vector_indices.is_empty(), vector_proof) {
        (true, None) => {}
        (false, Some(proof)) => {
            let commitments = vector_indices
                .iter()
                .map(|&index| hidden_tables[index].commitment)
                .collect::<Vec<_>>();
            let vector_evaluation_commitments = vector_indices
                .iter()
                .map(|&index| evaluation_commitments[index])
                .collect::<Vec<_>>();
            let (joint_commitment, coefficients) = verify_hidden_field_vectors_batch_zk(
                parameters,
                &commitments,
                hidden_tables[vector_indices[0]].logical_len,
                row_point,
                &vector_evaluation_commitments,
                &proof.opening,
                b"hidden-relation-vectors",
                outer_binding,
            )?;
            let combined_commitment =
                combine_commitments(&vector_evaluation_commitments, &coefficients);
            verify_dory_commitment_link(
                &parameters.blindfold,
                &combined_commitment,
                &joint_commitment,
                proof.blind_delta,
            )?;
        }
        _ => return Err(HiddenRelationError::ProofShape),
    }

    let groups = verifier_matrix_groups(hidden_tables)?;
    if groups.len() != matrix_proofs.len() {
        return Err(HiddenRelationError::ProofShape);
    }
    for (group, proof) in groups.iter().zip(matrix_proofs) {
        if proof.padded_cols != group.padded_cols
            || proof.column_evaluation_commitments.len() != group.matrices.len()
            || proof
                .column_evaluation_commitments
                .iter()
                .any(|commitments| commitments.len() != group.padded_cols)
        {
            return Err(HiddenRelationError::ProofShape);
        }
        for (entry, column_commitments) in group
            .matrices
            .iter()
            .zip(&proof.column_evaluation_commitments)
        {
            for &(table_index, column) in &entry.table_indices {
                if column_commitments[column] != evaluation_commitments[table_index] {
                    return Err(HiddenRelationError::Dory(
                        DenseI8DoryError::EvaluationCommitmentMismatch,
                    ));
                }
            }
        }
        let matrix_commitments = group
            .matrices
            .iter()
            .map(|entry| entry.commitment)
            .collect::<Vec<_>>();
        let column_point = matrix_column_point(
            outer_binding,
            hidden_tables[0].logical_len,
            group.padded_cols,
            row_point,
            &matrix_commitments,
            &proof.column_evaluation_commitments,
        );
        let column_weights = EqPolynomial::<Fr>::evals(&column_point, None);
        let matrix_evaluation_commitments = proof
            .column_evaluation_commitments
            .iter()
            .map(|commitments| combine_commitments(commitments, &column_weights))
            .collect::<Vec<_>>();
        let full_point = column_point
            .iter()
            .chain(row_point)
            .copied()
            .collect::<Vec<_>>();
        let (joint_commitment, coefficients) = verify_hidden_field_vectors_batch_zk(
            parameters,
            &matrix_commitments,
            hidden_tables[0].logical_len * group.padded_cols,
            &full_point,
            &matrix_evaluation_commitments,
            &proof.opening,
            b"hidden-relation-matrices",
            outer_binding,
        )?;
        let combined_commitment =
            combine_commitments(&matrix_evaluation_commitments, &coefficients);
        verify_dory_commitment_link(
            &parameters.blindfold,
            &combined_commitment,
            &joint_commitment,
            proof.blind_delta,
        )?;
    }
    Ok(())
}

fn prover_matrix_groups<'a>(
    tables: &'a [HiddenRelationTable<'a>],
) -> Result<Vec<ProverMatrixGroup<'a>>, HiddenRelationError> {
    let mut groups = BTreeMap::<usize, Vec<ProverMatrixEntry<'a>>>::new();
    for (table_index, table) in tables.iter().enumerate() {
        let Some(matrix) = &table.committed.matrix else {
            continue;
        };
        if matrix.padded_rows != table.committed.logical_len
            || matrix.logical_cols == 0
            || matrix.logical_cols > matrix.padded_cols
            || matrix.padded_cols == 0
            || !matrix.padded_cols.is_power_of_two()
            || table.committed.matrix_column >= matrix.logical_cols
            || table.committed.commitment != matrix.commitment
        {
            return Err(HiddenRelationError::ProofShape);
        }
        let entries = groups.entry(matrix.padded_cols).or_default();
        if let Some(entry) = entries.iter_mut().find(|entry| {
            entry.matrix.commitment == matrix.commitment
                && entry
                    .table_indices
                    .iter()
                    .all(|(_, column)| *column != table.committed.matrix_column)
        }) {
            entry
                .table_indices
                .push((table_index, table.committed.matrix_column));
        } else {
            entries.push(ProverMatrixEntry {
                matrix,
                table_indices: vec![(table_index, table.committed.matrix_column)],
            });
        }
    }
    Ok(groups
        .into_iter()
        .map(|(padded_cols, matrices)| ProverMatrixGroup {
            padded_cols,
            matrices,
        })
        .collect())
}

fn matrix_column_evaluations(
    entry: &ProverMatrixEntry<'_>,
    padded_cols: usize,
    terminal_evaluations: &[Fr],
    evaluate_column: impl Fn(&[ArkFr]) -> Result<Fr, HiddenRelationError> + Sync,
) -> Result<Vec<Fr>, HiddenRelationError> {
    (0..padded_cols)
        .into_par_iter()
        .map(|column| {
            if let Some(&(table_index, _)) = entry
                .table_indices
                .iter()
                .find(|(_, table_column)| *table_column == column)
            {
                return terminal_evaluations
                    .get(table_index)
                    .copied()
                    .ok_or(HiddenRelationError::ProofShape);
            }
            if column >= entry.matrix.logical_cols {
                return Ok(Fr::zero());
            }
            let start = column * entry.matrix.padded_rows;
            let end = start + entry.matrix.padded_rows;
            evaluate_column(&entry.matrix.values[start..end])
        })
        .collect()
}

fn verifier_matrix_groups<'a>(
    tables: &'a [HiddenRelationTableStatement<'a>],
) -> Result<Vec<VerifierMatrixGroup<'a>>, HiddenRelationError> {
    let mut groups = BTreeMap::<usize, Vec<VerifierMatrixEntry<'a>>>::new();
    for (table_index, table) in tables.iter().enumerate() {
        let Some(matrix) = table.matrix else {
            continue;
        };
        if matrix.padded_cols == 0
            || !matrix.padded_cols.is_power_of_two()
            || matrix.column >= matrix.padded_cols
        {
            return Err(HiddenRelationError::ProofShape);
        }
        let entries = groups.entry(matrix.padded_cols).or_default();
        if let Some(entry) = entries.iter_mut().find(|entry| {
            entry.commitment == table.commitment
                && entry
                    .table_indices
                    .iter()
                    .all(|(_, column)| *column != matrix.column)
        }) {
            entry.table_indices.push((table_index, matrix.column));
        } else {
            entries.push(VerifierMatrixEntry {
                commitment: table.commitment,
                table_indices: vec![(table_index, matrix.column)],
            });
        }
    }
    Ok(groups
        .into_iter()
        .map(|(padded_cols, matrices)| VerifierMatrixGroup {
            padded_cols,
            matrices,
        })
        .collect())
}

fn matrix_column_point(
    outer_binding: &[u8],
    logical_rows: usize,
    padded_cols: usize,
    row_point: &[Fr],
    matrix_commitments: &[&jolt_dory::DoryCommitment],
    column_evaluation_commitments: &[Vec<Bn254G1>],
) -> Vec<Fr> {
    let mut transcript = Blake2bTranscript::<Fr>::new(b"parcomm-matrix-column-v1");
    transcript.append(&Label(b"matrix_column_reduction_v1"));
    transcript.append(&LabelWithCount(
        b"outer_binding",
        outer_binding.len() as u64,
    ));
    transcript.append_bytes(outer_binding);
    transcript.append_labeled(b"logical_rows", &U64Word(logical_rows as u64));
    transcript.append_labeled(b"padded_cols", &U64Word(padded_cols as u64));
    transcript.append_values(b"row_point", row_point);
    transcript.append_labeled(b"matrix_count", &U64Word(matrix_commitments.len() as u64));
    for (commitment, column_commitments) in
        matrix_commitments.iter().zip(column_evaluation_commitments)
    {
        transcript.append_labeled(b"matrix_commitment", *commitment);
        transcript.append_labeled(b"column_count", &U64Word(column_commitments.len() as u64));
        for column_commitment in column_commitments {
            transcript.append_labeled(b"column_evaluation", column_commitment);
        }
    }
    transcript.challenge_vector(padded_cols.trailing_zeros() as usize)
}

fn verify_dory_commitment_link(
    setup: &jolt_crypto::PedersenSetup<Bn254G1>,
    combined_commitment: &Bn254G1,
    joint_commitment: &Bn254G1,
    blind_delta: Fr,
) -> Result<(), HiddenRelationError> {
    let blind_delta_commitment = Pedersen::<Bn254G1>::commit(setup, &[Fr::zero()], &blind_delta);
    if <Bn254G1 as HomomorphicCommitment<Fr>>::add(combined_commitment, &blind_delta_commitment)
        != *joint_commitment
    {
        return Err(HiddenRelationError::Dory(
            DenseI8DoryError::EvaluationCommitmentMismatch,
        ));
    }
    Ok(())
}

fn validate_prover_tables(
    hidden_tables: &[HiddenRelationTable<'_>],
    public_tables: &[PublicRelationTable<'_>],
    expression: &RelationExpression,
) -> Result<usize, HiddenRelationError> {
    let statements = hidden_tables
        .iter()
        .map(|table| HiddenRelationTableStatement::from(table.committed))
        .collect::<Vec<_>>();
    let logical_len = validate_verifier_tables(&statements, public_tables, expression)?;
    for (index, table) in hidden_tables.iter().enumerate() {
        if table.values.len() != logical_len {
            return Err(HiddenRelationError::TableLength {
                table: index,
                expected: logical_len,
                actual: table.values.len(),
            });
        }
        if let Some(matrix) = &table.committed.matrix {
            if matrix.padded_rows != logical_len
                || table.committed.matrix_column >= matrix.padded_cols
                || matrix.values[table.committed.matrix_column * logical_len
                    ..(table.committed.matrix_column + 1) * logical_len]
                    != *table.values
            {
                return Err(HiddenRelationError::ProofShape);
            }
        }
    }
    Ok(logical_len)
}

fn validate_verifier_tables(
    hidden_tables: &[HiddenRelationTableStatement<'_>],
    public_tables: &[PublicRelationTable<'_>],
    expression: &RelationExpression,
) -> Result<usize, HiddenRelationError> {
    let logical_len = hidden_tables
        .first()
        .map(|table| table.logical_len)
        .or_else(|| public_tables.first().map(|table| table.values.len()))
        .ok_or(HiddenRelationError::EmptyTables)?;
    if logical_len < 2 || !logical_len.is_power_of_two() {
        return Err(HiddenRelationError::InvalidLogicalLength(logical_len));
    }
    for (index, table) in hidden_tables.iter().enumerate() {
        if table.logical_len != logical_len {
            return Err(HiddenRelationError::TableLength {
                table: index,
                expected: logical_len,
                actual: table.logical_len,
            });
        }
        if let Some(matrix) = table.matrix {
            if matrix.padded_cols == 0
                || !matrix.padded_cols.is_power_of_two()
                || matrix.column >= matrix.padded_cols
            {
                return Err(HiddenRelationError::ProofShape);
            }
        }
    }
    for (public_index, table) in public_tables.iter().enumerate() {
        if table.values.len() != logical_len {
            return Err(HiddenRelationError::TableLength {
                table: hidden_tables.len() + public_index,
                expected: logical_len,
                actual: table.values.len(),
            });
        }
    }
    expression.validate(hidden_tables.len() + public_tables.len())?;
    Ok(logical_len)
}

fn public_table_digests(
    tables: &[PublicRelationTable<'_>],
) -> Result<Vec<[u8; 32]>, HiddenRelationError> {
    tables
        .iter()
        .map(|table| public_table_digest(table.values))
        .collect()
}

fn public_table_digest(values: &[ArkFr]) -> Result<[u8; 32], HiddenRelationError> {
    let mut hasher = Sha256::new();
    hasher.update(b"parcomm-hidden-relation-public-table-v1");
    hasher.update((values.len() as u64).to_le_bytes());
    let mut encoded = Vec::new();
    for value in values {
        encoded.clear();
        value
            .serialize_compressed(&mut encoded)
            .map_err(|error| HiddenRelationError::Encoding(error.to_string()))?;
        hasher.update((encoded.len() as u64).to_le_bytes());
        hasher.update(&encoded);
    }
    Ok(hasher.finalize().into())
}

fn evaluate_public_table(values: &[ArkFr], point: &[Fr]) -> Result<Fr, HiddenRelationError> {
    if values.len() < 2
        || !values.len().is_power_of_two()
        || point.len() != values.len().trailing_zeros() as usize
    {
        return Err(HiddenRelationError::ProofShape);
    }
    let mut current = values.iter().copied().map(Fr::from).collect::<Vec<_>>();
    for &challenge in point {
        bind_one(&mut current, challenge)?;
    }
    current
        .first()
        .copied()
        .ok_or(HiddenRelationError::ProofShape)
}

fn validate_capacity(
    available: usize,
    degree: usize,
    table_count: usize,
) -> Result<(), HiddenRelationError> {
    let required = (degree + 1).max(table_count.min(available));
    if degree + 1 > available {
        return Err(HiddenRelationError::PedersenCapacity {
            required,
            available,
        });
    }
    Ok(())
}

fn relation_round_polynomial(
    tables: &[Vec<Fr>],
    eq: &[Fr],
    expression: &RelationExpression,
    degree: usize,
) -> Result<UnivariatePoly<Fr>, HiddenRelationError> {
    let length = eq.len();
    if length < 2 || !length.is_power_of_two() || tables.iter().any(|table| table.len() != length) {
        return Err(HiddenRelationError::ProofShape);
    }
    let half = length / 2;
    let points = (0..=degree)
        .map(|point| Fr::from_u64(point as u64))
        .collect::<Vec<_>>();
    let evaluations = (0..half)
        .into_par_iter()
        .with_min_len(64)
        .fold(
            || (vec![Fr::zero(); degree + 1], vec![Fr::zero(); tables.len()]),
            |(mut partial, mut operand_values), row| {
                for (&point, value) in points.iter().zip(&mut partial) {
                    operand_values
                        .iter_mut()
                        .zip(tables)
                        .for_each(|(operand, table)| {
                            *operand = line(table[row], table[row + half], point);
                        });
                    *value +=
                        line(eq[row], eq[row + half], point) * expression.evaluate(&operand_values);
                }
                (partial, operand_values)
            },
        )
        .map(|(partial, _)| partial)
        .reduce(
            || vec![Fr::zero(); degree + 1],
            |mut left, right| {
                left.iter_mut()
                    .zip(right)
                    .for_each(|(left, right)| *left += right);
                left
            },
        );
    Ok(UnivariatePoly::from_evals(&evaluations))
}

fn bind_all(tables: &mut [Vec<Fr>], challenge: Fr) -> Result<(), HiddenRelationError> {
    if tables
        .iter()
        .any(|table| table.len() < 2 || !table.len().is_power_of_two())
    {
        return Err(HiddenRelationError::ProofShape);
    }
    tables.par_iter_mut().for_each(|table| {
        let half = table.len() / 2;
        let (left, right) = table.split_at_mut(half);
        left.par_iter_mut()
            .zip(right.par_iter())
            .for_each(|(left, &right)| *left = line(*left, right, challenge));
        table.truncate(half);
    });
    Ok(())
}

fn bind_one(values: &mut Vec<Fr>, challenge: Fr) -> Result<(), HiddenRelationError> {
    if values.len() < 2 || !values.len().is_power_of_two() {
        return Err(HiddenRelationError::ProofShape);
    }
    let half = values.len() / 2;
    for index in 0..half {
        values[index] = line(values[index], values[index + half], challenge);
    }
    values.truncate(half);
    Ok(())
}

fn line(left: Fr, right: Fr, point: Fr) -> Fr {
    left + point * (right - left)
}

fn relation_transcript(
    label: &[u8],
    hidden_tables: &[HiddenRelationTableStatement<'_>],
    public_digests: &[[u8; 32]],
    expression: &RelationExpression,
    logical_len: usize,
    check: RelationCheck,
) -> Result<Blake2bTranscript<Fr>, HiddenRelationError> {
    let mut transcript = Blake2bTranscript::<Fr>::new(b"parcomm-hidden-relation-v2");
    transcript.append(&Label(b"hidden_relation_statement"));
    transcript.append(&LabelWithCount(b"relation_label", label.len() as u64));
    transcript.append_bytes(label);
    transcript.append_labeled(b"logical_len", &U64Word(logical_len as u64));
    transcript.append_labeled(
        b"relation_check",
        &U64Word(u64::from(check == RelationCheck::Sum)),
    );
    transcript.append(&LabelWithCount(
        b"relation_tables",
        hidden_tables.len() as u64,
    ));
    for table in hidden_tables {
        transcript.append_labeled(b"relation_table_commitment", table.commitment);
        transcript.append_labeled(
            b"relation_table_logical_len",
            &U64Word(table.logical_len as u64),
        );
        let (padded_cols, column) = table
            .matrix
            .map(|matrix| (matrix.padded_cols, matrix.column))
            .unwrap_or((0, 0));
        transcript.append_labeled(b"relation_matrix_cols", &U64Word(padded_cols as u64));
        transcript.append_labeled(b"relation_matrix_col", &U64Word(column as u64));
    }
    if !public_digests.is_empty() {
        transcript.append(&LabelWithCount(
            b"relation_public_tables",
            public_digests.len() as u64,
        ));
        for digest in public_digests {
            transcript.append(&LabelWithCount(b"rel_public_digest", digest.len() as u64));
            transcript.append_bytes(digest);
        }
    }
    let expression_bytes = bincode::serialize(expression)
        .map_err(|error| HiddenRelationError::Encoding(error.to_string()))?;
    transcript.append(&LabelWithCount(
        b"relation_expression",
        expression_bytes.len() as u64,
    ));
    transcript.append_bytes(&expression_bytes);
    Ok(transcript)
}

fn relation_outer_binding(
    label: &[u8],
    hidden_tables: &[HiddenRelationTableStatement<'_>],
    public_digests: &[[u8; 32]],
    expression: &RelationExpression,
    logical_len: usize,
    zero_test_point: &[Fr],
    sumcheck: &CommittedSumcheckProof<Bn254G1>,
    check: RelationCheck,
) -> Result<Vec<u8>, HiddenRelationError> {
    let commitments = hidden_tables
        .iter()
        .map(|table| {
            let matrix = table
                .matrix
                .map(|matrix| (matrix.padded_cols, matrix.column));
            (
                commitment_binding_bytes(table.commitment),
                table.logical_len,
                matrix,
            )
        })
        .collect::<Vec<_>>();
    let encoded = if public_digests.is_empty() {
        bincode::serialize(&(
            label,
            logical_len,
            commitments,
            expression,
            check,
            zero_test_point,
            sumcheck,
        ))
    } else {
        bincode::serialize(&(
            b"parcomm-hidden-relation-public-binding-v1".as_slice(),
            label,
            logical_len,
            commitments,
            public_digests,
            expression,
            check,
            zero_test_point,
            sumcheck,
        ))
    };
    encoded.map_err(|error| HiddenRelationError::Encoding(error.to_string()))
}

fn bind_evaluation_commitments(
    transcript: &mut Blake2bTranscript<Fr>,
    commitments: &[Bn254G1],
) -> Vec<Fr> {
    transcript.append(&LabelWithCount(b"rel_eval_coms", commitments.len() as u64));
    for commitment in commitments {
        transcript.append_labeled(b"rel_eval_com", commitment);
    }
    let mut challenge = transcript.challenge();
    while challenge.is_zero() {
        transcript.append_labeled(b"rel_retry", &U64Word(1));
        challenge = transcript.challenge();
    }
    let mut coefficient = Fr::one();
    (0..commitments.len())
        .map(|_| {
            let current = coefficient;
            coefficient *= challenge;
            current
        })
        .collect()
}

fn combine_scalars(values: &[Fr], coefficients: &[Fr]) -> Fr {
    values
        .iter()
        .zip(coefficients)
        .map(|(&value, coefficient)| value * coefficient)
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
    use super::*;
    use crate::jolt_dory::{
        commit_field_matrix_columns_zk, commit_field_vector_zk, setup_dory_parameters,
    };
    use ark_ff::{One, Zero};
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use std::sync::atomic::{AtomicUsize, Ordering};

    #[test]
    fn relation_round_polynomial_matches_the_worked_row_reduction() {
        let tables = vec![
            [2_u64, 3, 5, 7]
                .into_iter()
                .map(Fr::from_u64)
                .collect::<Vec<_>>(),
            [11_u64, 13, 17, 19]
                .into_iter()
                .map(Fr::from_u64)
                .collect::<Vec<_>>(),
            [1_u64, 2, 3, 4]
                .into_iter()
                .map(Fr::from_u64)
                .collect::<Vec<_>>(),
        ];
        let eq = [5_u64, 7, 11, 13]
            .into_iter()
            .map(Fr::from_u64)
            .collect::<Vec<_>>();
        let expression = RelationExpression::table(0)
            .mul(RelationExpression::table(1))
            .sub(RelationExpression::table(2));

        let round = relation_round_polynomial(&tables, &eq, &expression, 3).unwrap();

        for (point, expected) in [364_u64, 2_579, 8_154, 18_601].into_iter().enumerate() {
            assert_eq!(
                round.evaluate(Fr::from_u64(point as u64)),
                Fr::from_u64(expected)
            );
        }
    }

    #[test]
    fn bind_all_matches_independent_multilinear_folds() {
        let mut tables = vec![
            [2_u64, 3, 5, 7]
                .into_iter()
                .map(Fr::from_u64)
                .collect::<Vec<_>>(),
            [11_u64, 13, 17, 19]
                .into_iter()
                .map(Fr::from_u64)
                .collect::<Vec<_>>(),
        ];

        bind_all(&mut tables, Fr::from_u64(11)).unwrap();

        assert_eq!(
            tables,
            vec![
                vec![Fr::from_u64(35), Fr::from_u64(47)],
                vec![Fr::from_u64(77), Fr::from_u64(79)],
            ]
        );
    }

    #[test]
    fn matrix_column_reduction_reuses_terminal_values_and_skips_zero_padding() {
        let (prover_parameters, _) = setup_dory_parameters(4);
        let columns = (0..3)
            .map(|column| {
                (0..4)
                    .map(|row| ArkFr::from((10 * column + row + 1) as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let committed = commit_field_matrix_columns_zk(&prover_parameters, &columns, 4).unwrap();
        let entry = ProverMatrixEntry {
            matrix: committed[0].matrix.as_ref().unwrap(),
            table_indices: vec![(0, 0), (1, 1), (2, 2)],
        };
        let terminal = [Fr::from_u64(17), Fr::from_u64(19), Fr::from_u64(23)];
        let evaluations = AtomicUsize::new(0);

        let columns = matrix_column_evaluations(&entry, 4, &terminal, |_| {
            evaluations.fetch_add(1, Ordering::Relaxed);
            Ok(Fr::from_u64(29))
        })
        .unwrap();

        assert_eq!(evaluations.load(Ordering::Relaxed), 0);
        assert_eq!(
            columns,
            vec![terminal[0], terminal[1], terminal[2], Fr::zero()]
        );
    }

    #[test]
    fn matrix_column_reduction_only_evaluates_unreferenced_logical_columns() {
        let (prover_parameters, _) = setup_dory_parameters(4);
        let columns = (0..3)
            .map(|column| {
                (0..4)
                    .map(|row| ArkFr::from((10 * column + row + 1) as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let committed = commit_field_matrix_columns_zk(&prover_parameters, &columns, 4).unwrap();
        let entry = ProverMatrixEntry {
            matrix: committed[0].matrix.as_ref().unwrap(),
            table_indices: vec![(0, 0)],
        };
        let evaluations = AtomicUsize::new(0);

        let columns = matrix_column_evaluations(&entry, 4, &[Fr::from_u64(17)], |_| {
            evaluations.fetch_add(1, Ordering::Relaxed);
            Ok(Fr::from_u64(29))
        })
        .unwrap();

        assert_eq!(evaluations.load(Ordering::Relaxed), 2);
        assert_eq!(
            columns,
            vec![
                Fr::from_u64(17),
                Fr::from_u64(29),
                Fr::from_u64(29),
                Fr::zero(),
            ]
        );
    }

    #[test]
    fn matrix_columns_reduce_by_ip_to_one_whole_matrix_dory_opening() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let left = (0..16)
            .map(|index| ArkFr::from(index as u64 + 2))
            .collect::<Vec<_>>();
        let right = (0..16)
            .map(|index| ArkFr::from(index as u64 + 5))
            .collect::<Vec<_>>();
        let product = left
            .iter()
            .zip(&right)
            .map(|(left, right)| *left * right)
            .collect::<Vec<_>>();
        let columns = vec![left, right, product];
        let committed = commit_field_matrix_columns_zk(&prover_parameters, &columns, 4).unwrap();
        assert!(committed
            .iter()
            .all(|column| column.commitment == committed[0].commitment));

        let tables = columns
            .iter()
            .zip(&committed)
            .map(|(values, committed)| HiddenRelationTable { values, committed })
            .collect::<Vec<_>>();
        let expression = RelationExpression::table(0)
            .mul(RelationExpression::table(1))
            .sub(RelationExpression::table(2));
        let mut rng = StdRng::seed_from_u64(91);
        let proof = prove_hidden_relation(
            b"whole-matrix-product",
            &prover_parameters,
            &tables,
            &expression,
            &mut rng,
        )
        .unwrap();
        assert!(proof.vector_dory_batch.is_none());
        assert_eq!(proof.matrix_dory_batches.len(), 1);

        let statements = committed
            .iter()
            .map(HiddenRelationTableStatement::from)
            .collect::<Vec<_>>();
        verify_hidden_relation(
            b"whole-matrix-product",
            &verifier_parameters,
            &statements,
            &expression,
            &proof,
        )
        .unwrap();
    }

    #[test]
    fn hidden_product_relation_verifies_and_rejects_wrong_commitment() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let left = (0..16)
            .map(|index| ArkFr::from(index as u64 + 2))
            .collect::<Vec<_>>();
        let right = (0..16)
            .map(|index| ArkFr::from(index as u64 + 7))
            .collect::<Vec<_>>();
        let product = left
            .iter()
            .zip(&right)
            .map(|(left, right)| *left * right)
            .collect::<Vec<_>>();
        let committed_left = commit_field_vector_zk(&prover_parameters, &left).unwrap();
        let committed_right = commit_field_vector_zk(&prover_parameters, &right).unwrap();
        let committed_product = commit_field_vector_zk(&prover_parameters, &product).unwrap();
        let expression = RelationExpression::table(0)
            .mul(RelationExpression::table(1))
            .sub(RelationExpression::table(2));
        let tables = [
            HiddenRelationTable {
                values: &left,
                committed: &committed_left,
            },
            HiddenRelationTable {
                values: &right,
                committed: &committed_right,
            },
            HiddenRelationTable {
                values: &product,
                committed: &committed_product,
            },
        ];
        let mut rng = StdRng::seed_from_u64(7);
        let proof = prove_hidden_relation(
            b"test-product",
            &prover_parameters,
            &tables,
            &expression,
            &mut rng,
        )
        .unwrap();
        let statements = [
            HiddenRelationTableStatement::from(&committed_left),
            HiddenRelationTableStatement::from(&committed_right),
            HiddenRelationTableStatement::from(&committed_product),
        ];
        verify_hidden_relation(
            b"test-product",
            &verifier_parameters,
            &statements,
            &expression,
            &proof,
        )
        .unwrap();

        let wrong = vec![ArkFr::one(); 16];
        let wrong_commitment = commit_field_vector_zk(&prover_parameters, &wrong).unwrap();
        let wrong_statements = [
            HiddenRelationTableStatement::from(&committed_left),
            HiddenRelationTableStatement::from(&committed_right),
            HiddenRelationTableStatement::from(&wrong_commitment),
        ];
        assert!(verify_hidden_relation(
            b"test-product",
            &verifier_parameters,
            &wrong_statements,
            &expression,
            &proof,
        )
        .is_err());
    }

    #[test]
    fn deferred_relation_verification_records_claims_before_dory_opening() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let left = (0..16)
            .map(|index| ArkFr::from(index as u64 + 3))
            .collect::<Vec<_>>();
        let right = left.clone();
        let committed_left = commit_field_vector_zk(&prover_parameters, &left).unwrap();
        let committed_right = commit_field_vector_zk(&prover_parameters, &right).unwrap();
        let tables = [
            HiddenRelationTable {
                values: &left,
                committed: &committed_left,
            },
            HiddenRelationTable {
                values: &right,
                committed: &committed_right,
            },
        ];
        let statements = [
            HiddenRelationTableStatement::from(&committed_left),
            HiddenRelationTableStatement::from(&committed_right),
        ];
        let expression = RelationExpression::table(0).sub(RelationExpression::table(1));
        let mut rng = StdRng::seed_from_u64(7331);

        let (reduction, prover_opening) = prove_hidden_relation_deferred(
            b"deferred-relation",
            &prover_parameters,
            &tables,
            &expression,
            &mut rng,
        )
        .unwrap();
        assert_eq!(prover_opening.hidden_tables().len(), 2);
        assert_eq!(prover_opening.logical_len(), 16);
        assert_eq!(prover_opening.row_point().len(), 4);
        assert_eq!(prover_opening.evaluations().len(), 2);
        assert_eq!(prover_opening.evaluation_commitments().len(), 2);
        assert_eq!(prover_opening.evaluation_blinds().len(), 2);
        assert!(!prover_opening.outer_binding().is_empty());
        let prover_claims = prover_opening.into_claims();
        assert_eq!(prover_claims.logical_len(), 16);
        assert_eq!(prover_claims.row_point().len(), 4);
        assert_eq!(prover_claims.table_count(), 2);
        assert_eq!(prover_claims.evaluations().len(), 2);
        assert_eq!(prover_claims.evaluation_commitments().len(), 2);
        assert_eq!(prover_claims.evaluation_blinds().len(), 2);
        assert!(!prover_claims.outer_binding().is_empty());
        let prover_opening = prover_claims.bind_tables(tables.to_vec()).unwrap();

        let verifier_opening = verify_hidden_relation_deferred(
            b"deferred-relation",
            &verifier_parameters,
            &statements,
            &expression,
            &reduction,
        )
        .unwrap();
        assert_eq!(verifier_opening.hidden_tables().len(), 2);
        assert_eq!(verifier_opening.logical_len(), 16);
        assert_eq!(verifier_opening.row_point().len(), 4);
        assert_eq!(verifier_opening.evaluation_commitments().len(), 2);
        assert!(!verifier_opening.outer_binding().is_empty());
        let verifier_claims = verifier_opening.into_claims();
        assert_eq!(verifier_claims.logical_len(), 16);
        assert_eq!(verifier_claims.row_point().len(), 4);
        assert_eq!(verifier_claims.table_count(), 2);
        assert_eq!(verifier_claims.evaluation_commitments().len(), 2);
        assert!(!verifier_claims.outer_binding().is_empty());
        let verifier_opening = verifier_claims.bind_tables(statements.to_vec()).unwrap();
        let opening =
            prove_hidden_relation_opening(&prover_parameters, prover_opening, &mut rng).unwrap();
        verify_hidden_relation_opening(&verifier_parameters, verifier_opening, &opening).unwrap();

        let wrong = vec![ArkFr::one(); 16];
        let wrong_commitment = commit_field_vector_zk(&prover_parameters, &wrong).unwrap();
        let wrong_statements = [
            HiddenRelationTableStatement::from(&committed_left),
            HiddenRelationTableStatement::from(&wrong_commitment),
        ];
        assert!(verify_hidden_relation_deferred(
            b"deferred-relation",
            &verifier_parameters,
            &wrong_statements,
            &expression,
            &reduction,
        )
        .is_err());
    }

    fn deferred_equality_openings<'a>(
        prover_parameters: &DoryProverParameters,
        verifier_parameters: &DoryVerifierParameters,
        left: &'a [ArkFr],
        right: &'a [ArkFr],
        committed_left: &'a CommittedDoryFieldVector,
        committed_right: &'a CommittedDoryFieldVector,
        label: &[u8],
        rng: &mut impl RngCore,
    ) -> (
        HiddenRelationProverOpening<'a>,
        HiddenRelationVerifierOpening<'a>,
    ) {
        let tables = [
            HiddenRelationTable {
                values: left,
                committed: committed_left,
            },
            HiddenRelationTable {
                values: right,
                committed: committed_right,
            },
        ];
        let statements = [
            HiddenRelationTableStatement::from(committed_left),
            HiddenRelationTableStatement::from(committed_right),
        ];
        let expression = RelationExpression::table(0).sub(RelationExpression::table(1));
        let (reduction, prover_opening) =
            prove_hidden_relation_deferred(label, prover_parameters, &tables, &expression, rng)
                .unwrap();
        let verifier_opening = verify_hidden_relation_deferred(
            label,
            verifier_parameters,
            &statements,
            &expression,
            &reduction,
        )
        .unwrap();
        (prover_opening, verifier_opening)
    }

    #[test]
    fn phase_batch_opening_groups_equal_points_and_roundtrips() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let values = (0..16)
            .map(|index| ArkFr::from(index as u64 + 13))
            .collect::<Vec<_>>();
        let committed_left = commit_field_vector_zk(&prover_parameters, &values).unwrap();
        let committed_right = commit_field_vector_zk(&prover_parameters, &values).unwrap();
        let mut rng = StdRng::seed_from_u64(8_101);
        let (first_prover, first_verifier) = deferred_equality_openings(
            &prover_parameters,
            &verifier_parameters,
            &values,
            &values,
            &committed_left,
            &committed_right,
            b"phase-batch-honest",
            &mut rng,
        );
        let mut second_prover = first_prover.clone();
        let mut second_verifier = first_verifier.clone();
        second_prover.outer_binding.extend_from_slice(b"second");
        second_verifier.outer_binding.extend_from_slice(b"second");

        let proof = prove_hidden_relation_openings_batch(
            &prover_parameters,
            &[first_prover, second_prover],
            &mut rng,
        )
        .unwrap();
        assert_eq!(proof.groups.len(), 1);
        assert_eq!(proof.groups[0].relation_indices, vec![0, 1]);
        assert!(proof.groups[0].opening.vector_dory_batch.is_some());
        let encoded = bincode::serialize(&proof).unwrap();
        let decoded: HiddenRelationOpeningBatchProof = bincode::deserialize(&encoded).unwrap();
        assert_eq!(decoded, proof);
        verify_hidden_relation_openings_batch(
            &verifier_parameters,
            &[first_verifier, second_verifier],
            &decoded,
        )
        .unwrap();
    }

    #[test]
    fn phase_batch_opening_handles_repeated_columns_of_one_matrix_commitment() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let left = (0..16)
            .map(|index| ArkFr::from(index as u64 + 3))
            .collect::<Vec<_>>();
        let right = (0..16)
            .map(|index| ArkFr::from(index as u64 + 7))
            .collect::<Vec<_>>();
        let product = left
            .iter()
            .zip(&right)
            .map(|(left, right)| *left * right)
            .collect::<Vec<_>>();
        let columns = vec![left, right, product];
        let committed = commit_field_matrix_columns_zk(&prover_parameters, &columns, 4).unwrap();
        let tables = columns
            .iter()
            .zip(&committed)
            .map(|(values, committed)| HiddenRelationTable { values, committed })
            .collect::<Vec<_>>();
        let statements = committed
            .iter()
            .map(HiddenRelationTableStatement::from)
            .collect::<Vec<_>>();
        let expression = RelationExpression::table(0)
            .mul(RelationExpression::table(1))
            .sub(RelationExpression::table(2));
        let mut rng = StdRng::seed_from_u64(8_104);
        let (reduction, first_prover) = prove_hidden_relation_deferred(
            b"phase-batch-matrix",
            &prover_parameters,
            &tables,
            &expression,
            &mut rng,
        )
        .unwrap();
        let first_verifier = verify_hidden_relation_deferred(
            b"phase-batch-matrix",
            &verifier_parameters,
            &statements,
            &expression,
            &reduction,
        )
        .unwrap();
        let mut second_prover = first_prover.clone();
        let mut second_verifier = first_verifier.clone();
        second_prover
            .outer_binding
            .extend_from_slice(b"matrix-second");
        second_verifier
            .outer_binding
            .extend_from_slice(b"matrix-second");

        let proof = prove_hidden_relation_openings_batch(
            &prover_parameters,
            &[first_prover, second_prover],
            &mut rng,
        )
        .unwrap();
        assert_eq!(proof.groups.len(), 1);
        assert!(proof.groups[0].opening.vector_dory_batch.is_none());
        assert_eq!(proof.groups[0].opening.matrix_dory_batches.len(), 1);
        assert_eq!(
            proof.groups[0].opening.matrix_dory_batches[0]
                .column_evaluation_commitments
                .len(),
            2
        );
        verify_hidden_relation_openings_batch(
            &verifier_parameters,
            &[first_verifier, second_verifier],
            &proof,
        )
        .unwrap();
    }

    #[test]
    fn phase_batch_opening_rejects_tampered_binding_and_dory_link() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let values = (0..16)
            .map(|index| ArkFr::from(index as u64 + 29))
            .collect::<Vec<_>>();
        let committed_left = commit_field_vector_zk(&prover_parameters, &values).unwrap();
        let committed_right = commit_field_vector_zk(&prover_parameters, &values).unwrap();
        let mut rng = StdRng::seed_from_u64(8_102);
        let (prover_opening, verifier_opening) = deferred_equality_openings(
            &prover_parameters,
            &verifier_parameters,
            &values,
            &values,
            &committed_left,
            &committed_right,
            b"phase-batch-tamper",
            &mut rng,
        );
        let proof =
            prove_hidden_relation_openings_batch(&prover_parameters, &[prover_opening], &mut rng)
                .unwrap();

        let mut wrong_binding = proof.clone();
        wrong_binding.groups[0].opening_binding[0] ^= 1;
        assert!(verify_hidden_relation_openings_batch(
            &verifier_parameters,
            &[verifier_opening.clone()],
            &wrong_binding,
        )
        .is_err());

        let mut wrong_link = proof;
        wrong_link.groups[0]
            .opening
            .vector_dory_batch
            .as_mut()
            .unwrap()
            .blind_delta += Fr::one();
        assert!(verify_hidden_relation_openings_batch(
            &verifier_parameters,
            &[verifier_opening],
            &wrong_link,
        )
        .is_err());
    }

    #[test]
    fn phase_batch_opening_binds_constituent_order_and_empty_input() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let values = (0..16)
            .map(|index| ArkFr::from(index as u64 + 41))
            .collect::<Vec<_>>();
        let committed_left = commit_field_vector_zk(&prover_parameters, &values).unwrap();
        let committed_right = commit_field_vector_zk(&prover_parameters, &values).unwrap();
        let mut rng = StdRng::seed_from_u64(8_103);
        let (first_prover, first_verifier) = deferred_equality_openings(
            &prover_parameters,
            &verifier_parameters,
            &values,
            &values,
            &committed_left,
            &committed_right,
            b"phase-batch-order",
            &mut rng,
        );
        let mut second_prover = first_prover.clone();
        let mut second_verifier = first_verifier.clone();
        second_prover
            .outer_binding
            .extend_from_slice(b"ordered-second");
        second_verifier
            .outer_binding
            .extend_from_slice(b"ordered-second");
        let proof = prove_hidden_relation_openings_batch(
            &prover_parameters,
            &[first_prover, second_prover],
            &mut rng,
        )
        .unwrap();
        assert!(verify_hidden_relation_openings_batch(
            &verifier_parameters,
            &[second_verifier, first_verifier],
            &proof,
        )
        .is_err());

        let empty_prover = Vec::<HiddenRelationProverOpening<'static>>::new();
        let empty_verifier = Vec::<HiddenRelationVerifierOpening<'static>>::new();
        let empty_proof =
            prove_hidden_relation_openings_batch(&prover_parameters, &empty_prover, &mut rng)
                .unwrap();
        assert_eq!(empty_proof, HiddenRelationOpeningBatchProof::default());
        verify_hidden_relation_openings_batch(&verifier_parameters, &empty_verifier, &empty_proof)
            .unwrap();
        let mut unexpected_group = empty_proof;
        unexpected_group.groups.push(proof.groups[0].clone());
        assert!(verify_hidden_relation_openings_batch(
            &verifier_parameters,
            &empty_verifier,
            &unexpected_group,
        )
        .is_err());
    }

    #[test]
    fn invalid_relation_is_rejected_before_proof_generation() {
        let (prover_parameters, _) = setup_dory_parameters(8);
        let left = vec![ArkFr::one(); 8];
        let wrong = vec![ArkFr::zero(); 8];
        let committed_left = commit_field_vector_zk(&prover_parameters, &left).unwrap();
        let committed_wrong = commit_field_vector_zk(&prover_parameters, &wrong).unwrap();
        let expression = RelationExpression::table(0).sub(RelationExpression::table(1));
        let tables = [
            HiddenRelationTable {
                values: &left,
                committed: &committed_left,
            },
            HiddenRelationTable {
                values: &wrong,
                committed: &committed_wrong,
            },
        ];
        let mut rng = StdRng::seed_from_u64(9);
        assert!(matches!(
            prove_hidden_relation(
                b"test-invalid",
                &prover_parameters,
                &tables,
                &expression,
                &mut rng,
            ),
            Err(HiddenRelationError::NonZeroRelation)
        ));
    }

    #[test]
    fn hidden_sum_relation_accepts_balanced_non_pointwise_values() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let left = vec![
            ArkFr::from(1_u64),
            ArkFr::from(2_u64),
            ArkFr::from(3_u64),
            ArkFr::from(4_u64),
            ArkFr::from(5_u64),
            ArkFr::from(6_u64),
            ArkFr::from(7_u64),
            ArkFr::from(8_u64),
        ];
        let right = vec![
            ArkFr::from(8_u64),
            ArkFr::from(7_u64),
            ArkFr::from(6_u64),
            ArkFr::from(5_u64),
            ArkFr::from(4_u64),
            ArkFr::from(3_u64),
            ArkFr::from(2_u64),
            ArkFr::from(1_u64),
        ];
        let committed_left = commit_field_vector_zk(&prover_parameters, &left).unwrap();
        let committed_right = commit_field_vector_zk(&prover_parameters, &right).unwrap();
        let expression = RelationExpression::table(0).sub(RelationExpression::table(1));
        let tables = [
            HiddenRelationTable {
                values: &left,
                committed: &committed_left,
            },
            HiddenRelationTable {
                values: &right,
                committed: &committed_right,
            },
        ];
        let mut rng = StdRng::seed_from_u64(17);
        let proof = prove_hidden_sum_relation(
            b"test-sum",
            &prover_parameters,
            &tables,
            &expression,
            &mut rng,
        )
        .unwrap();
        let statements = [
            HiddenRelationTableStatement::from(&committed_left),
            HiddenRelationTableStatement::from(&committed_right),
        ];
        verify_hidden_sum_relation(
            b"test-sum",
            &verifier_parameters,
            &statements,
            &expression,
            &proof,
        )
        .unwrap();
        assert!(verify_hidden_relation(
            b"test-sum",
            &verifier_parameters,
            &statements,
            &expression,
            &proof,
        )
        .is_err());
    }

    #[test]
    fn public_row_varying_operand_verifies_without_dory_opening_and_rejects_tampering() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let left = (0..16)
            .map(|index| ArkFr::from(index as u64 + 3))
            .collect::<Vec<_>>();
        let public_multiplier = (0..16)
            .map(|index| ArkFr::from((index * 5 + 2) as u64))
            .collect::<Vec<_>>();
        let product = left
            .iter()
            .zip(&public_multiplier)
            .map(|(left, multiplier)| *left * multiplier)
            .collect::<Vec<_>>();
        let committed_left = commit_field_vector_zk(&prover_parameters, &left).unwrap();
        let committed_product = commit_field_vector_zk(&prover_parameters, &product).unwrap();
        let expression = RelationExpression::table(0)
            .mul(RelationExpression::table(2))
            .sub(RelationExpression::table(1));
        let hidden_tables = [
            HiddenRelationTable {
                values: &left,
                committed: &committed_left,
            },
            HiddenRelationTable {
                values: &product,
                committed: &committed_product,
            },
        ];
        let public_tables = [PublicRelationTable {
            values: &public_multiplier,
        }];
        let mut rng = StdRng::seed_from_u64(23);
        let proof = prove_hidden_relation_with_public(
            b"test-public-row-varying",
            &prover_parameters,
            &hidden_tables,
            &public_tables,
            &expression,
            &mut rng,
        )
        .unwrap();
        assert_eq!(proof.evaluation_commitments.len(), 3);

        let hidden_statements = [
            HiddenRelationTableStatement::from(&committed_left),
            HiddenRelationTableStatement::from(&committed_product),
        ];
        verify_hidden_relation_with_public(
            b"test-public-row-varying",
            &verifier_parameters,
            &hidden_statements,
            &public_tables,
            &expression,
            &proof,
        )
        .unwrap();

        let mut tampered_multiplier = public_multiplier;
        tampered_multiplier[7] += ArkFr::one();
        let tampered_public_tables = [PublicRelationTable {
            values: &tampered_multiplier,
        }];
        assert!(verify_hidden_relation_with_public(
            b"test-public-row-varying",
            &verifier_parameters,
            &hidden_statements,
            &tampered_public_tables,
            &expression,
            &proof,
        )
        .is_err());
    }

    #[test]
    fn public_sum_operand_accepts_equal_hypercube_sums() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let hidden = (1..=8).map(ArkFr::from).collect::<Vec<_>>();
        let public = (1..=8).rev().map(ArkFr::from).collect::<Vec<_>>();
        let committed_hidden = commit_field_vector_zk(&prover_parameters, &hidden).unwrap();
        let expression = RelationExpression::table(0).sub(RelationExpression::table(1));
        let hidden_tables = [HiddenRelationTable {
            values: &hidden,
            committed: &committed_hidden,
        }];
        let public_tables = [PublicRelationTable { values: &public }];
        let mut rng = StdRng::seed_from_u64(29);
        let proof = prove_hidden_sum_relation_with_public(
            b"test-public-sum",
            &prover_parameters,
            &hidden_tables,
            &public_tables,
            &expression,
            &mut rng,
        )
        .unwrap();
        assert_eq!(proof.evaluation_commitments.len(), 2);

        let hidden_statements = [HiddenRelationTableStatement::from(&committed_hidden)];
        verify_hidden_sum_relation_with_public(
            b"test-public-sum",
            &verifier_parameters,
            &hidden_statements,
            &public_tables,
            &expression,
            &proof,
        )
        .unwrap();
        assert!(verify_hidden_relation_with_public(
            b"test-public-sum",
            &verifier_parameters,
            &hidden_statements,
            &public_tables,
            &expression,
            &proof,
        )
        .is_err());
    }

    #[test]
    fn logical_len_one_is_rejected_before_the_committed_stage() {
        let (prover_parameters, _) = setup_dory_parameters(2);
        let values = [ArkFr::one()];
        let public_tables = [PublicRelationTable { values: &values }];
        let expression = RelationExpression::table(0).sub(RelationExpression::table(0));
        let mut rng = StdRng::seed_from_u64(31);
        assert!(matches!(
            prove_hidden_relation_with_public(
                b"test-public-singleton",
                &prover_parameters,
                &[],
                &public_tables,
                &expression,
                &mut rng,
            ),
            Err(HiddenRelationError::InvalidLogicalLength(1))
        ));
    }

    #[test]
    fn v2_eval_commitment_batching_changes_with_ordered_commitments() {
        let (prover_parameters, _) = setup_dory_parameters(4);
        let zero = Fr::zero();
        let first =
            Pedersen::<Bn254G1>::commit(&prover_parameters.blindfold, &[Fr::from_u64(3)], &zero);
        let second =
            Pedersen::<Bn254G1>::commit(&prover_parameters.blindfold, &[Fr::from_u64(7)], &zero);

        let mut original = Blake2bTranscript::<Fr>::new(b"parcomm-hidden-relation-v2");
        original.append(&Label(b"test_eval_batching"));
        let original_coefficients = bind_evaluation_commitments(&mut original, &[first, second]);

        let mut changed = Blake2bTranscript::<Fr>::new(b"parcomm-hidden-relation-v2");
        changed.append(&Label(b"test_eval_batching"));
        let changed_coefficients = bind_evaluation_commitments(&mut changed, &[second, first]);

        assert_ne!(original_coefficients, changed_coefficients);
        assert_eq!(original_coefficients[0], Fr::one());
        assert_eq!(changed_coefficients[0], Fr::one());
    }
}
