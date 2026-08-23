//! Hidden LogUp over Dory-committed tuple columns.
//!
//! The source tuple commitments are fixed before the tuple-compression challenge.
//! The prover then commits to the compressed keys and lookup multiplicities before
//! the denominator challenge. Inverse witnesses are materialized with one batch
//! inversion and committed afterwards. One deferred identity batch binds every
//! auxiliary table to its definition, and one deferred sum batch proves the
//! LogUp fractional-sum identity. Their PCS openings are emitted only after both
//! reductions, without revealing source values, multiplicities, inverses, or
//! leaf evaluations.

use std::collections::HashMap;
use std::fmt::{Display, Formatter};

use ark_bn254::Fr as ArkFr;
use ark_ff::{batch_inversion, One, Zero};
use ark_std::rand::RngCore;
use jolt_dory::DoryCommitment;
use jolt_field::Fr as JoltFr;
use jolt_transcript::domain::{Label, LabelWithCount, U64Word};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::hidden_relation::{
    prove_hidden_relation_openings_batch, verify_hidden_relation_openings_batch,
    HiddenRelationError, HiddenRelationOpeningBatchProof, HiddenRelationProverOpening,
    HiddenRelationProverOpeningClaims, HiddenRelationTable, HiddenRelationTableStatement,
    HiddenRelationVerifierOpening, HiddenRelationVerifierOpeningClaims, PublicRelationTable,
    RelationExpression,
};
use crate::hidden_relation_batch::{
    deduplicate_prover_hidden_tables, deduplicate_verifier_hidden_tables,
    prove_hidden_relations_batch_deferred, prove_hidden_sum_relations_batch_deferred,
    verify_hidden_relations_batch_deferred, verify_hidden_sum_relations_batch_deferred,
    HiddenRelationBatchProverItem, HiddenRelationBatchReductionProof,
    HiddenRelationBatchVerifierItem,
};
use crate::jolt_dory::{
    commit_field_matrix_column_slices_zk, commit_field_vector_zk, CommittedDoryFieldVector,
    DoryProverParameters, DoryVerifierParameters,
};
use crate::section6_final_opening::{
    commit_section6_field_matrix_columns_on_grid_zk, Section6DoryGrid,
};

const LOGUP_TRANSCRIPT_DOMAIN: &[u8] = b"parcomm-hidden-logup-v1";
const LOGUP_RELATION_DOMAIN: &[u8] = b"parcomm-hidden-logup-relation-v1";

/// The Dory commitments created by one LogUp proof after the source tuples are fixed.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenLogUpAuxCommitments {
    pub query_keys: DoryCommitment,
    pub table_keys: DoryCommitment,
    pub multiplicities: DoryCommitment,
    pub query_inverses: DoryCommitment,
    pub table_inverses: DoryCommitment,
}

/// A hidden LogUp proof. No source value, multiplicity, inverse, challenge, or
/// evaluation scalar is serialized here.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenLogUpProof {
    pub auxiliary_commitments: HiddenLogUpAuxCommitments,
    pub identity_reduction: HiddenRelationBatchReductionProof,
    pub sum_reduction: HiddenRelationBatchReductionProof,
    pub relation_opening: HiddenRelationOpeningBatchProof,
}

/// Serializable single-part LogUp reduction without any PCS opening.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenLogUpReductionProof {
    pub auxiliary_commitments: HiddenLogUpAuxCommitments,
    pub identity_reduction: HiddenRelationBatchReductionProof,
    pub sum_reduction: HiddenRelationBatchReductionProof,
}

/// Auxiliary commitments for a multipart LogUp. Each field commits one
/// complete column-major part matrix.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenLogUpPartsAuxCommitments {
    pub query_keys: DoryCommitment,
    pub table_keys: DoryCommitment,
    pub multiplicities: DoryCommitment,
    pub query_inverses: DoryCommitment,
    pub table_inverses: DoryCommitment,
}

/// A multipart LogUp proof with one batch for every identity family and one
/// global fractional-sum batch across all query and table parts.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenLogUpPartsProof {
    pub auxiliary_commitments: HiddenLogUpPartsAuxCommitments,
    pub identity_reduction: HiddenRelationBatchReductionProof,
    pub sum_reduction: HiddenRelationBatchReductionProof,
    pub relation_opening: HiddenRelationOpeningBatchProof,
}

/// Serializable multipart LogUp reduction without any PCS opening.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenLogUpPartsReductionProof {
    pub auxiliary_commitments: HiddenLogUpPartsAuxCommitments,
    pub identity_reduction: HiddenRelationBatchReductionProof,
    pub sum_reduction: HiddenRelationBatchReductionProof,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HiddenLogUpSide {
    Query,
    Table,
}

/// One prover-side tuple column. Hidden and public columns may appear in any
/// tuple position; their position determines the tuple-compression coefficient.
#[derive(Clone, Copy)]
pub enum MixedLogUpColumn<'a> {
    Hidden(HiddenRelationTable<'a>),
    Public(PublicRelationTable<'a>),
}

/// One verifier-side tuple column. Public vectors are part of the statement and
/// are evaluated directly by the verifier in the key-definition relation.
#[derive(Clone, Copy)]
pub enum MixedLogUpColumnStatement<'a> {
    Hidden(HiddenRelationTableStatement<'a>),
    Public(PublicRelationTable<'a>),
}

/// Prover-owned LogUp state retained until all phase openings are collected.
pub struct HiddenLogUpProverOpeningState<'a> {
    relation_claims: [HiddenRelationProverOpeningClaims; 2],
    tables: HiddenLogUpProverTables<'a>,
}

enum HiddenLogUpProverTables<'a> {
    Single(HiddenLogUpSingleProverTables<'a>),
    Parts(HiddenLogUpPartsProverTables<'a>),
}

struct HiddenLogUpSingleProverTables<'a> {
    query_hidden_tables: Vec<HiddenRelationTable<'a>>,
    table_hidden_tables: Vec<HiddenRelationTable<'a>>,
    query_keys: Vec<ArkFr>,
    table_keys: Vec<ArkFr>,
    multiplicities: Vec<ArkFr>,
    query_inverses: Vec<ArkFr>,
    table_inverses: Vec<ArkFr>,
    committed_query_keys: CommittedDoryFieldVector,
    committed_table_keys: CommittedDoryFieldVector,
    committed_multiplicities: CommittedDoryFieldVector,
    committed_query_inverses: CommittedDoryFieldVector,
    committed_table_inverses: CommittedDoryFieldVector,
}

struct HiddenLogUpPartsProverTables<'a> {
    logical_len: usize,
    query_hidden_parts: Vec<Vec<HiddenRelationTable<'a>>>,
    table_hidden_parts: Vec<Vec<HiddenRelationTable<'a>>>,
    query_keys: Vec<Vec<ArkFr>>,
    table_keys: Vec<Vec<ArkFr>>,
    multiplicities: Vec<Vec<ArkFr>>,
    all_inverses: Vec<ArkFr>,
    committed_query_keys: Vec<CommittedDoryFieldVector>,
    committed_table_keys: Vec<CommittedDoryFieldVector>,
    committed_multiplicities: Vec<CommittedDoryFieldVector>,
    committed_query_inverses: Vec<CommittedDoryFieldVector>,
    committed_table_inverses: Vec<CommittedDoryFieldVector>,
}

/// Commitment-only LogUp claims retained by the verifier until phase opening.
pub struct HiddenLogUpVerifierOpeningState<'a> {
    relation_claims: [HiddenRelationVerifierOpeningClaims; 2],
    tables: HiddenLogUpVerifierTables<'a>,
}

enum HiddenLogUpVerifierTables<'a> {
    Single(HiddenLogUpSingleVerifierTables<'a>),
    Parts(HiddenLogUpPartsVerifierTables<'a>),
}

struct HiddenLogUpSingleVerifierTables<'a> {
    query_hidden_tables: Vec<HiddenRelationTableStatement<'a>>,
    table_hidden_tables: Vec<HiddenRelationTableStatement<'a>>,
    query_key: HiddenRelationTableStatement<'a>,
    table_key: HiddenRelationTableStatement<'a>,
    multiplicities: HiddenRelationTableStatement<'a>,
    query_inverses: HiddenRelationTableStatement<'a>,
    table_inverses: HiddenRelationTableStatement<'a>,
}

struct HiddenLogUpPartsVerifierTables<'a> {
    query_hidden_parts: Vec<Vec<HiddenRelationTableStatement<'a>>>,
    table_hidden_parts: Vec<Vec<HiddenRelationTableStatement<'a>>>,
    query_keys: Vec<HiddenRelationTableStatement<'a>>,
    table_keys: Vec<HiddenRelationTableStatement<'a>>,
    multiplicities: Vec<HiddenRelationTableStatement<'a>>,
    query_inverses: Vec<HiddenRelationTableStatement<'a>>,
    table_inverses: Vec<HiddenRelationTableStatement<'a>>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum HiddenLogUpError {
    EmptyTuple,
    EmptyParts {
        side: HiddenLogUpSide,
    },
    TupleWidth {
        query: usize,
        table: usize,
    },
    InvalidLogicalLength(usize),
    ColumnLength {
        side: HiddenLogUpSide,
        column: usize,
        expected: usize,
        actual: usize,
    },
    SourceValueLength {
        side: HiddenLogUpSide,
        column: usize,
        expected: usize,
        actual: usize,
    },
    PartTupleWidth {
        side: HiddenLogUpSide,
        part: usize,
        expected: usize,
        actual: usize,
    },
    PartColumnLength {
        side: HiddenLogUpSide,
        part: usize,
        column: usize,
        expected: usize,
        actual: usize,
    },
    PartSourceValueLength {
        side: HiddenLogUpSide,
        part: usize,
        column: usize,
        expected: usize,
        actual: usize,
    },
    ZeroCompressionChallenge,
    MissingQueryTuple {
        query_index: usize,
    },
    MultiplicityOverflow {
        table_index: usize,
    },
    SingularDenominator {
        side: HiddenLogUpSide,
        index: usize,
    },
    SingularPartDenominator {
        side: HiddenLogUpSide,
        part: usize,
        index: usize,
    },
    ProofShape,
    Encoding(String),
    Dory(String),
    Relation(HiddenRelationError),
}

impl Display for HiddenLogUpError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for HiddenLogUpError {}

impl From<HiddenRelationError> for HiddenLogUpError {
    fn from(error: HiddenRelationError) -> Self {
        Self::Relation(error)
    }
}

impl HiddenLogUpProverOpeningState<'_> {
    /// Reconstruct the two terminal relation openings without copying table data.
    pub fn relation_openings(
        &self,
    ) -> Result<[HiddenRelationProverOpening<'_>; 2], HiddenLogUpError> {
        let (identity_tables, sum_tables) = match &self.tables {
            HiddenLogUpProverTables::Single(tables) => {
                let mut identity_tables = Vec::new();
                identity_tables.extend(tables.query_hidden_tables.iter().copied());
                identity_tables.push(HiddenRelationTable {
                    values: &tables.query_keys,
                    committed: &tables.committed_query_keys,
                });
                identity_tables.extend(tables.table_hidden_tables.iter().copied());
                identity_tables.push(HiddenRelationTable {
                    values: &tables.table_keys,
                    committed: &tables.committed_table_keys,
                });
                identity_tables.extend([
                    HiddenRelationTable {
                        values: &tables.query_keys,
                        committed: &tables.committed_query_keys,
                    },
                    HiddenRelationTable {
                        values: &tables.query_inverses,
                        committed: &tables.committed_query_inverses,
                    },
                    HiddenRelationTable {
                        values: &tables.table_keys,
                        committed: &tables.committed_table_keys,
                    },
                    HiddenRelationTable {
                        values: &tables.table_inverses,
                        committed: &tables.committed_table_inverses,
                    },
                ]);
                let sum_tables = vec![
                    HiddenRelationTable {
                        values: &tables.query_inverses,
                        committed: &tables.committed_query_inverses,
                    },
                    HiddenRelationTable {
                        values: &tables.multiplicities,
                        committed: &tables.committed_multiplicities,
                    },
                    HiddenRelationTable {
                        values: &tables.table_inverses,
                        committed: &tables.committed_table_inverses,
                    },
                ];
                (identity_tables, sum_tables)
            }
            HiddenLogUpProverTables::Parts(tables) => {
                let mut identity_tables = Vec::new();
                for (part, hidden_tables) in tables.query_hidden_parts.iter().enumerate() {
                    identity_tables.extend(hidden_tables.iter().copied());
                    identity_tables.push(HiddenRelationTable {
                        values: &tables.query_keys[part],
                        committed: &tables.committed_query_keys[part],
                    });
                }
                for (part, hidden_tables) in tables.table_hidden_parts.iter().enumerate() {
                    identity_tables.extend(hidden_tables.iter().copied());
                    identity_tables.push(HiddenRelationTable {
                        values: &tables.table_keys[part],
                        committed: &tables.committed_table_keys[part],
                    });
                }
                for part in 0..tables.query_keys.len() {
                    identity_tables.push(HiddenRelationTable {
                        values: &tables.query_keys[part],
                        committed: &tables.committed_query_keys[part],
                    });
                    identity_tables.push(HiddenRelationTable {
                        values: inverse_part(&tables.all_inverses, part, tables.logical_len),
                        committed: &tables.committed_query_inverses[part],
                    });
                }
                for part in 0..tables.table_keys.len() {
                    identity_tables.push(HiddenRelationTable {
                        values: &tables.table_keys[part],
                        committed: &tables.committed_table_keys[part],
                    });
                    identity_tables.push(HiddenRelationTable {
                        values: inverse_part(
                            &tables.all_inverses,
                            tables.query_keys.len() + part,
                            tables.logical_len,
                        ),
                        committed: &tables.committed_table_inverses[part],
                    });
                }

                let mut sum_tables = Vec::new();
                for part in 0..tables.query_keys.len() {
                    sum_tables.push(HiddenRelationTable {
                        values: inverse_part(&tables.all_inverses, part, tables.logical_len),
                        committed: &tables.committed_query_inverses[part],
                    });
                }
                for part in 0..tables.table_keys.len() {
                    sum_tables.push(HiddenRelationTable {
                        values: &tables.multiplicities[part],
                        committed: &tables.committed_multiplicities[part],
                    });
                    sum_tables.push(HiddenRelationTable {
                        values: inverse_part(
                            &tables.all_inverses,
                            tables.query_keys.len() + part,
                            tables.logical_len,
                        ),
                        committed: &tables.committed_table_inverses[part],
                    });
                }
                (identity_tables, sum_tables)
            }
        };
        Ok([
            self.relation_claims[0]
                .bind_tables(deduplicate_prover_hidden_tables(identity_tables))?,
            self.relation_claims[1].bind_tables(deduplicate_prover_hidden_tables(sum_tables))?,
        ])
    }
}

impl HiddenLogUpProverOpeningState<'_> {
    /// Replace temporary multipart source descriptors with equivalent
    /// longer-lived hidden table views.
    ///
    /// Public columns are intentionally absent because terminal PCS openings
    /// only use hidden tables. This is useful when tuple normalization adds
    /// temporary public tag or zero columns.
    pub fn rebind_parts_hidden_tables<'a>(
        self,
        query_hidden_parts: Vec<Vec<HiddenRelationTable<'a>>>,
        table_hidden_parts: Vec<Vec<HiddenRelationTable<'a>>>,
    ) -> Result<HiddenLogUpProverOpeningState<'a>, HiddenLogUpError> {
        let HiddenLogUpProverTables::Parts(tables) = self.tables else {
            return Err(HiddenLogUpError::ProofShape);
        };
        if !same_prover_hidden_parts(&tables.query_hidden_parts, &query_hidden_parts)
            || !same_prover_hidden_parts(&tables.table_hidden_parts, &table_hidden_parts)
        {
            return Err(HiddenLogUpError::ProofShape);
        }
        let HiddenLogUpPartsProverTables {
            logical_len,
            query_hidden_parts: _,
            table_hidden_parts: _,
            query_keys,
            table_keys,
            multiplicities,
            all_inverses,
            committed_query_keys,
            committed_table_keys,
            committed_multiplicities,
            committed_query_inverses,
            committed_table_inverses,
        } = tables;
        Ok(HiddenLogUpProverOpeningState {
            relation_claims: self.relation_claims,
            tables: HiddenLogUpProverTables::Parts(HiddenLogUpPartsProverTables {
                logical_len,
                query_hidden_parts,
                table_hidden_parts,
                query_keys,
                table_keys,
                multiplicities,
                all_inverses,
                committed_query_keys,
                committed_table_keys,
                committed_multiplicities,
                committed_query_inverses,
                committed_table_inverses,
            }),
        })
    }
}

impl<'a> HiddenLogUpVerifierOpeningState<'a> {
    /// Return the two commitment-only terminal claims in transcript order.
    pub fn relation_openings(
        &self,
    ) -> Result<[HiddenRelationVerifierOpening<'a>; 2], HiddenLogUpError> {
        let (identity_tables, sum_tables) = match &self.tables {
            HiddenLogUpVerifierTables::Single(tables) => {
                let mut identity_tables = tables.query_hidden_tables.clone();
                identity_tables.push(tables.query_key);
                identity_tables.extend(tables.table_hidden_tables.iter().copied());
                identity_tables.push(tables.table_key);
                identity_tables.extend([
                    tables.query_key,
                    tables.query_inverses,
                    tables.table_key,
                    tables.table_inverses,
                ]);
                let sum_tables = vec![
                    tables.query_inverses,
                    tables.multiplicities,
                    tables.table_inverses,
                ];
                (identity_tables, sum_tables)
            }
            HiddenLogUpVerifierTables::Parts(tables) => {
                let mut identity_tables = Vec::new();
                for (part, hidden_tables) in tables.query_hidden_parts.iter().enumerate() {
                    identity_tables.extend(hidden_tables.iter().copied());
                    identity_tables.push(tables.query_keys[part]);
                }
                for (part, hidden_tables) in tables.table_hidden_parts.iter().enumerate() {
                    identity_tables.extend(hidden_tables.iter().copied());
                    identity_tables.push(tables.table_keys[part]);
                }
                for part in 0..tables.query_keys.len() {
                    identity_tables.push(tables.query_keys[part]);
                    identity_tables.push(tables.query_inverses[part]);
                }
                for part in 0..tables.table_keys.len() {
                    identity_tables.push(tables.table_keys[part]);
                    identity_tables.push(tables.table_inverses[part]);
                }
                let mut sum_tables = tables.query_inverses.clone();
                for part in 0..tables.table_keys.len() {
                    sum_tables.push(tables.multiplicities[part]);
                    sum_tables.push(tables.table_inverses[part]);
                }
                (identity_tables, sum_tables)
            }
        };
        Ok([
            self.relation_claims[0]
                .bind_tables(deduplicate_verifier_hidden_tables(identity_tables))?,
            self.relation_claims[1].bind_tables(deduplicate_verifier_hidden_tables(sum_tables))?,
        ])
    }
}

impl HiddenLogUpVerifierOpeningState<'_> {
    /// Replace temporary multipart source descriptors with equivalent
    /// longer-lived hidden commitment statements.
    pub fn rebind_parts_hidden_tables<'a>(
        self,
        proof: &'a HiddenLogUpPartsReductionProof,
        query_hidden_parts: Vec<Vec<HiddenRelationTableStatement<'a>>>,
        table_hidden_parts: Vec<Vec<HiddenRelationTableStatement<'a>>>,
    ) -> Result<HiddenLogUpVerifierOpeningState<'a>, HiddenLogUpError> {
        let HiddenLogUpVerifierTables::Parts(tables) = self.tables else {
            return Err(HiddenLogUpError::ProofShape);
        };
        if !same_verifier_hidden_parts(&tables.query_hidden_parts, &query_hidden_parts)
            || !same_verifier_hidden_parts(&tables.table_hidden_parts, &table_hidden_parts)
        {
            return Err(HiddenLogUpError::ProofShape);
        }
        let logical_len = tables
            .query_keys
            .first()
            .ok_or(HiddenLogUpError::ProofShape)?
            .logical_len;
        let query_keys = auxiliary_matrix_statements(
            &proof.auxiliary_commitments.query_keys,
            logical_len,
            tables.query_keys.len(),
        );
        let table_keys = auxiliary_matrix_statements(
            &proof.auxiliary_commitments.table_keys,
            logical_len,
            tables.table_keys.len(),
        );
        let multiplicities = auxiliary_matrix_statements(
            &proof.auxiliary_commitments.multiplicities,
            logical_len,
            tables.multiplicities.len(),
        );
        let query_inverses = auxiliary_matrix_statements(
            &proof.auxiliary_commitments.query_inverses,
            logical_len,
            tables.query_inverses.len(),
        );
        let table_inverses = auxiliary_matrix_statements(
            &proof.auxiliary_commitments.table_inverses,
            logical_len,
            tables.table_inverses.len(),
        );
        if !same_verifier_tables(&tables.query_keys, &query_keys)
            || !same_verifier_tables(&tables.table_keys, &table_keys)
            || !same_verifier_tables(&tables.multiplicities, &multiplicities)
            || !same_verifier_tables(&tables.query_inverses, &query_inverses)
            || !same_verifier_tables(&tables.table_inverses, &table_inverses)
        {
            return Err(HiddenLogUpError::ProofShape);
        }
        let HiddenLogUpPartsVerifierTables {
            query_hidden_parts: _,
            table_hidden_parts: _,
            query_keys: _,
            table_keys: _,
            multiplicities: _,
            query_inverses: _,
            table_inverses: _,
        } = tables;
        Ok(HiddenLogUpVerifierOpeningState {
            relation_claims: self.relation_claims,
            tables: HiddenLogUpVerifierTables::Parts(HiddenLogUpPartsVerifierTables {
                query_hidden_parts,
                table_hidden_parts,
                query_keys,
                table_keys,
                multiplicities,
                query_inverses,
                table_inverses,
            }),
        })
    }
}

fn same_prover_hidden_parts(
    left: &[Vec<HiddenRelationTable<'_>>],
    right: &[Vec<HiddenRelationTable<'_>>],
) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left_part, right_part)| {
            left_part.len() == right_part.len()
                && left_part.iter().zip(right_part).all(|(left, right)| {
                    left.values.len() == right.values.len()
                        && left.values.as_ptr() == right.values.as_ptr()
                        && std::ptr::eq(left.committed, right.committed)
                })
        })
}

fn same_verifier_hidden_parts(
    left: &[Vec<HiddenRelationTableStatement<'_>>],
    right: &[Vec<HiddenRelationTableStatement<'_>>],
) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left_part, right_part)| {
            left_part.len() == right_part.len()
                && left_part.iter().zip(right_part).all(|(left, right)| {
                    std::ptr::eq(left.commitment, right.commitment)
                        && left.logical_len == right.logical_len
                        && left.matrix == right.matrix
                })
        })
}

fn same_verifier_tables(
    left: &[HiddenRelationTableStatement<'_>],
    right: &[HiddenRelationTableStatement<'_>],
) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.commitment == right.commitment
                && left.logical_len == right.logical_len
                && left.matrix == right.matrix
        })
}

/// Produce the native relation-opening batch proof for deferred LogUp states.
///
/// Section 6 should instead forward [`HiddenLogUpProverOpeningState::relation_openings`]
/// to its shared-grid terminal accumulator.
pub fn prove_hidden_logup_openings_batch(
    parameters: &DoryProverParameters,
    states: &[&HiddenLogUpProverOpeningState<'_>],
    rng: &mut impl RngCore,
) -> Result<HiddenRelationOpeningBatchProof, HiddenLogUpError> {
    if states.is_empty() {
        return Err(HiddenLogUpError::ProofShape);
    }
    let mut pending = Vec::with_capacity(2 * states.len());
    for state in states {
        pending.extend(state.relation_openings()?);
    }
    Ok(prove_hidden_relation_openings_batch(
        parameters, &pending, rng,
    )?)
}

/// Verify the native relation-opening batch proof for deferred LogUp states.
///
/// Section 6 should instead forward [`HiddenLogUpVerifierOpeningState::relation_openings`]
/// to its shared-grid terminal accumulator.
pub fn verify_hidden_logup_openings_batch(
    parameters: &DoryVerifierParameters,
    states: &[&HiddenLogUpVerifierOpeningState<'_>],
    proof: &HiddenRelationOpeningBatchProof,
) -> Result<(), HiddenLogUpError> {
    if states.is_empty() {
        return Err(HiddenLogUpError::ProofShape);
    }
    let mut pending = Vec::with_capacity(2 * states.len());
    for state in states {
        pending.extend(state.relation_openings()?);
    }
    Ok(verify_hidden_relation_openings_batch(
        parameters, &pending, proof,
    )?)
}

/// Prove a tuple lookup. Query and table columns must use the same power-of-two
/// cover. Padding is therefore an ordinary tuple chosen by the caller, typically
/// a fixed sentinel. A row identifier is not a special protocol input: add it as
/// an ordinary tuple column whenever the lookup must be row-local.
pub fn prove_hidden_logup(
    label: &[u8],
    parameters: &DoryProverParameters,
    query_columns: &[HiddenRelationTable<'_>],
    table_columns: &[HiddenRelationTable<'_>],
    rng: &mut impl RngCore,
) -> Result<HiddenLogUpProof, HiddenLogUpError> {
    let query_columns = query_columns
        .iter()
        .map(|column| {
            MixedLogUpColumn::Hidden(HiddenRelationTable {
                values: column.values,
                committed: column.committed,
            })
        })
        .collect::<Vec<_>>();
    let table_columns = table_columns
        .iter()
        .map(|column| {
            MixedLogUpColumn::Hidden(HiddenRelationTable {
                values: column.values,
                committed: column.committed,
            })
        })
        .collect::<Vec<_>>();
    prove_mixed_hidden_logup(label, parameters, &query_columns, &table_columns, rng)
}

/// Reduce a hidden tuple lookup while deferring all terminal Dory openings.
pub fn prove_hidden_logup_deferred<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    query_columns: &[HiddenRelationTable<'a>],
    table_columns: &[HiddenRelationTable<'a>],
    rng: &mut impl RngCore,
) -> Result<(HiddenLogUpReductionProof, HiddenLogUpProverOpeningState<'a>), HiddenLogUpError> {
    let query_columns = query_columns
        .iter()
        .copied()
        .map(MixedLogUpColumn::Hidden)
        .collect::<Vec<_>>();
    let table_columns = table_columns
        .iter()
        .copied()
        .map(MixedLogUpColumn::Hidden)
        .collect::<Vec<_>>();
    prove_mixed_hidden_logup_deferred(label, parameters, &query_columns, &table_columns, rng)
}

/// Prove a tuple lookup whose source columns may independently be hidden or
/// verifier-known. Source columns are bound before tuple compression, key and
/// multiplicity commitments are bound before beta, and all denominators are
/// inverted in one batch.
pub fn prove_mixed_hidden_logup(
    label: &[u8],
    parameters: &DoryProverParameters,
    query_columns: &[MixedLogUpColumn<'_>],
    table_columns: &[MixedLogUpColumn<'_>],
    rng: &mut impl RngCore,
) -> Result<HiddenLogUpProof, HiddenLogUpError> {
    let (reduction, pending) =
        prove_mixed_hidden_logup_deferred(label, parameters, query_columns, table_columns, rng)?;
    let relation_opening = prove_hidden_logup_openings_batch(parameters, &[&pending], rng)?;
    Ok(HiddenLogUpProof {
        auxiliary_commitments: reduction.auxiliary_commitments,
        identity_reduction: reduction.identity_reduction,
        sum_reduction: reduction.sum_reduction,
        relation_opening,
    })
}

/// Reduce a mixed hidden/public tuple lookup without opening any commitment.
pub fn prove_mixed_hidden_logup_deferred<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    query_columns: &[MixedLogUpColumn<'a>],
    table_columns: &[MixedLogUpColumn<'a>],
    rng: &mut impl RngCore,
) -> Result<(HiddenLogUpReductionProof, HiddenLogUpProverOpeningState<'a>), HiddenLogUpError> {
    let logical_len = validate_mixed_prover_columns(query_columns, table_columns)?;
    let query_statements = prover_column_statements(query_columns);
    let table_statements = prover_column_statements(table_columns);

    let (mut transcript, compression_challenge) = source_transcript_and_challenge_mixed(
        label,
        &query_statements,
        &table_statements,
        logical_len,
    )?;
    let query_slices = query_columns
        .iter()
        .map(mixed_prover_values)
        .collect::<Result<Vec<_>, _>>()?;
    let table_slices = table_columns
        .iter()
        .map(mixed_prover_values)
        .collect::<Result<Vec<_>, _>>()?;
    let query_keys = compress_tuple_columns(&query_slices, compression_challenge, logical_len);
    let table_keys = compress_tuple_columns(&table_slices, compression_challenge, logical_len);
    let multiplicities = compute_multiplicities(&query_keys, &table_keys)?;

    let committed_query_keys = commit_auxiliary(parameters, &query_keys)?;
    let committed_table_keys = commit_auxiliary(parameters, &table_keys)?;
    let committed_multiplicities = commit_auxiliary(parameters, &multiplicities)?;
    let denominator_challenge = append_pre_beta_commitments_and_challenge(
        &mut transcript,
        logical_len,
        &committed_query_keys.commitment,
        &committed_table_keys.commitment,
        &committed_multiplicities.commitment,
    );
    let (query_inverses, table_inverses) =
        batch_inverse_denominators(&query_keys, &table_keys, denominator_challenge)?;
    let committed_query_inverses = commit_auxiliary(parameters, &query_inverses)?;
    let committed_table_inverses = commit_auxiliary(parameters, &table_inverses)?;

    let (query_hidden, query_public, query_expression) = prover_key_relation_inputs(
        query_columns,
        &query_keys,
        &committed_query_keys,
        compression_challenge,
    );
    let (table_hidden, table_public, table_expression) = prover_key_relation_inputs(
        table_columns,
        &table_keys,
        &committed_table_keys,
        compression_challenge,
    );
    let inverse_expression = inverse_expression(denominator_challenge);
    let query_inverse_tables = [
        HiddenRelationTable {
            values: &query_keys,
            committed: &committed_query_keys,
        },
        HiddenRelationTable {
            values: &query_inverses,
            committed: &committed_query_inverses,
        },
    ];
    let table_inverse_tables = [
        HiddenRelationTable {
            values: &table_keys,
            committed: &committed_table_keys,
        },
        HiddenRelationTable {
            values: &table_inverses,
            committed: &committed_table_inverses,
        },
    ];
    let sum_balance_tables = [
        HiddenRelationTable {
            values: &query_inverses,
            committed: &committed_query_inverses,
        },
        HiddenRelationTable {
            values: &multiplicities,
            committed: &committed_multiplicities,
        },
        HiddenRelationTable {
            values: &table_inverses,
            committed: &committed_table_inverses,
        },
    ];
    let sum_expression = sum_balance_expression();
    let identity_items = [
        HiddenRelationBatchProverItem {
            tables: &query_hidden,
            public_tables: &query_public,
            expression: &query_expression,
        },
        HiddenRelationBatchProverItem {
            tables: &table_hidden,
            public_tables: &table_public,
            expression: &table_expression,
        },
        HiddenRelationBatchProverItem {
            tables: &query_inverse_tables,
            public_tables: &[],
            expression: &inverse_expression,
        },
        HiddenRelationBatchProverItem {
            tables: &table_inverse_tables,
            public_tables: &[],
            expression: &inverse_expression,
        },
    ];
    let (identity_reduction, identity_pending) = prove_hidden_relations_batch_deferred(
        &relation_label(label, b"identity-batch"),
        parameters,
        &identity_items,
        rng,
    )?;
    let sum_items = [HiddenRelationBatchProverItem {
        tables: &sum_balance_tables,
        public_tables: &[],
        expression: &sum_expression,
    }];
    let (sum_reduction, sum_pending) = prove_hidden_sum_relations_batch_deferred(
        &relation_label(label, b"sum-batch"),
        parameters,
        &sum_items,
        rng,
    )?;

    let reduction = HiddenLogUpReductionProof {
        auxiliary_commitments: HiddenLogUpAuxCommitments {
            query_keys: committed_query_keys.commitment.clone(),
            table_keys: committed_table_keys.commitment.clone(),
            multiplicities: committed_multiplicities.commitment.clone(),
            query_inverses: committed_query_inverses.commitment.clone(),
            table_inverses: committed_table_inverses.commitment.clone(),
        },
        identity_reduction,
        sum_reduction,
    };
    let pending = HiddenLogUpProverOpeningState {
        relation_claims: [identity_pending.into_claims(), sum_pending.into_claims()],
        tables: HiddenLogUpProverTables::Single(HiddenLogUpSingleProverTables {
            query_hidden_tables: hidden_prover_tables(query_columns),
            table_hidden_tables: hidden_prover_tables(table_columns),
            query_keys,
            table_keys,
            multiplicities,
            query_inverses,
            table_inverses,
            committed_query_keys,
            committed_table_keys,
            committed_multiplicities,
            committed_query_inverses,
            committed_table_inverses,
        }),
    };
    Ok((reduction, pending))
}

/// Prove one LogUp argument over multiple fixed-size query and table parts.
/// Tuple compression is shared by every part, multiplicities are computed over
/// the union of table parts, and a single batch inversion covers all parts.
pub fn prove_hidden_logup_parts(
    label: &[u8],
    parameters: &DoryProverParameters,
    query_parts: &[Vec<MixedLogUpColumn<'_>>],
    table_parts: &[Vec<MixedLogUpColumn<'_>>],
    rng: &mut impl RngCore,
) -> Result<HiddenLogUpPartsProof, HiddenLogUpError> {
    let (reduction, pending) =
        prove_hidden_logup_parts_deferred(label, parameters, query_parts, table_parts, rng)?;
    let relation_opening = prove_hidden_logup_openings_batch(parameters, &[&pending], rng)?;
    Ok(HiddenLogUpPartsProof {
        auxiliary_commitments: reduction.auxiliary_commitments,
        identity_reduction: reduction.identity_reduction,
        sum_reduction: reduction.sum_reduction,
        relation_opening,
    })
}

/// Reduce a multipart LogUp while deferring all terminal Dory openings.
pub fn prove_hidden_logup_parts_deferred<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    query_parts: &[Vec<MixedLogUpColumn<'a>>],
    table_parts: &[Vec<MixedLogUpColumn<'a>>],
    rng: &mut impl RngCore,
) -> Result<
    (
        HiddenLogUpPartsReductionProof,
        HiddenLogUpProverOpeningState<'a>,
    ),
    HiddenLogUpError,
> {
    prove_hidden_logup_parts_deferred_with_grid(
        label,
        parameters,
        None,
        query_parts,
        table_parts,
        rng,
    )
}

/// Reduce a multipart LogUp whose five auxiliary matrices are committed as
/// low-index prefixes of the shared Section 6 Dory grid.
pub fn prove_hidden_logup_parts_deferred_on_grid<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    query_parts: &[Vec<MixedLogUpColumn<'a>>],
    table_parts: &[Vec<MixedLogUpColumn<'a>>],
    rng: &mut impl RngCore,
) -> Result<
    (
        HiddenLogUpPartsReductionProof,
        HiddenLogUpProverOpeningState<'a>,
    ),
    HiddenLogUpError,
> {
    prove_hidden_logup_parts_deferred_with_grid(
        label,
        parameters,
        Some(grid),
        query_parts,
        table_parts,
        rng,
    )
}

fn prove_hidden_logup_parts_deferred_with_grid<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    grid: Option<&Section6DoryGrid>,
    query_parts: &[Vec<MixedLogUpColumn<'a>>],
    table_parts: &[Vec<MixedLogUpColumn<'a>>],
    rng: &mut impl RngCore,
) -> Result<
    (
        HiddenLogUpPartsReductionProof,
        HiddenLogUpProverOpeningState<'a>,
    ),
    HiddenLogUpError,
> {
    let (logical_len, tuple_width) = validate_mixed_prover_parts(query_parts, table_parts)?;
    if let Some(grid) = grid {
        validate_part_matrices_fit_grid(logical_len, query_parts.len(), table_parts.len(), grid)?;
    }
    let query_statements = prover_part_statements(query_parts);
    let table_statements = prover_part_statements(table_parts);
    let (mut transcript, compression_challenge) = source_parts_transcript_and_challenge(
        label,
        &query_statements,
        &table_statements,
        logical_len,
        tuple_width,
    )?;

    let query_keys = compress_prover_parts(query_parts, compression_challenge, logical_len)?;
    let table_keys = compress_prover_parts(table_parts, compression_challenge, logical_len)?;
    let multiplicities = compute_part_multiplicities(&query_keys, &table_keys)?;
    let committed_query_keys = commit_auxiliary_matrix(parameters, grid, &query_keys)?;
    let committed_table_keys = commit_auxiliary_matrix(parameters, grid, &table_keys)?;
    let committed_multiplicities = commit_auxiliary_matrix(parameters, grid, &multiplicities)?;
    let denominator_challenge = append_pre_beta_parts_and_challenge(
        &mut transcript,
        logical_len,
        &committed_query_keys,
        &committed_table_keys,
        &committed_multiplicities,
    );
    let all_inverses = batch_inverse_part_denominators(
        &query_keys,
        &table_keys,
        denominator_challenge,
        logical_len,
    )?;
    let query_part_count = query_parts.len();
    let table_part_count = table_parts.len();
    let committed_query_inverses = commit_inverse_matrix(
        parameters,
        grid,
        &all_inverses,
        0,
        query_part_count,
        logical_len,
    )?;
    let committed_table_inverses = commit_inverse_matrix(
        parameters,
        grid,
        &all_inverses,
        query_part_count,
        table_part_count,
        logical_len,
    )?;
    let part_batch_challenge = append_post_inverse_parts_and_challenge(
        &mut transcript,
        query_part_count,
        table_part_count,
        &committed_query_inverses[0].commitment,
        &committed_table_inverses[0].commitment,
    );

    let (query_key_hidden, query_key_public, query_key_expression) =
        prover_batched_key_relation_inputs(
            query_parts,
            &query_keys,
            &committed_query_keys,
            compression_challenge,
            part_batch_challenge,
        );
    let (query_inverse_tables, query_inverse_expression) = prover_batched_inverse_inputs(
        &query_keys,
        &committed_query_keys,
        &all_inverses,
        0,
        logical_len,
        &committed_query_inverses,
        denominator_challenge,
        part_batch_challenge,
    );
    let (table_key_hidden, table_key_public, table_key_expression) =
        prover_batched_key_relation_inputs(
            table_parts,
            &table_keys,
            &committed_table_keys,
            compression_challenge,
            part_batch_challenge,
        );
    let (table_inverse_tables, table_inverse_expression) = prover_batched_inverse_inputs(
        &table_keys,
        &committed_table_keys,
        &all_inverses,
        query_part_count,
        logical_len,
        &committed_table_inverses,
        denominator_challenge,
        part_batch_challenge,
    );
    let sum_tables = prover_parts_sum_tables(
        &all_inverses,
        logical_len,
        &multiplicities,
        &committed_query_inverses,
        &committed_multiplicities,
        &committed_table_inverses,
    );
    let sum_expression = parts_sum_balance_expression(query_part_count, table_part_count);
    let identity_items = [
        HiddenRelationBatchProverItem {
            tables: &query_key_hidden,
            public_tables: &query_key_public,
            expression: &query_key_expression,
        },
        HiddenRelationBatchProverItem {
            tables: &table_key_hidden,
            public_tables: &table_key_public,
            expression: &table_key_expression,
        },
        HiddenRelationBatchProverItem {
            tables: &query_inverse_tables,
            public_tables: &[],
            expression: &query_inverse_expression,
        },
        HiddenRelationBatchProverItem {
            tables: &table_inverse_tables,
            public_tables: &[],
            expression: &table_inverse_expression,
        },
    ];
    let (identity_reduction, identity_pending) = prove_hidden_relations_batch_deferred(
        &relation_label(label, b"parts-identity-batch"),
        parameters,
        &identity_items,
        rng,
    )?;
    let sum_items = [HiddenRelationBatchProverItem {
        tables: &sum_tables,
        public_tables: &[],
        expression: &sum_expression,
    }];
    let (sum_reduction, sum_pending) = prove_hidden_sum_relations_batch_deferred(
        &relation_label(label, b"parts-sum-batch"),
        parameters,
        &sum_items,
        rng,
    )?;

    let reduction = HiddenLogUpPartsReductionProof {
        auxiliary_commitments: HiddenLogUpPartsAuxCommitments {
            query_keys: committed_query_keys[0].commitment.clone(),
            table_keys: committed_table_keys[0].commitment.clone(),
            multiplicities: committed_multiplicities[0].commitment.clone(),
            query_inverses: committed_query_inverses[0].commitment.clone(),
            table_inverses: committed_table_inverses[0].commitment.clone(),
        },
        identity_reduction,
        sum_reduction,
    };
    let pending = HiddenLogUpProverOpeningState {
        relation_claims: [identity_pending.into_claims(), sum_pending.into_claims()],
        tables: HiddenLogUpProverTables::Parts(HiddenLogUpPartsProverTables {
            logical_len,
            query_hidden_parts: query_parts
                .iter()
                .map(|part| hidden_prover_tables(part))
                .collect(),
            table_hidden_parts: table_parts
                .iter()
                .map(|part| hidden_prover_tables(part))
                .collect(),
            query_keys,
            table_keys,
            multiplicities,
            all_inverses,
            committed_query_keys,
            committed_table_keys,
            committed_multiplicities,
            committed_query_inverses,
            committed_table_inverses,
        }),
    };
    Ok((reduction, pending))
}

/// Verify a hidden tuple lookup from source commitment metadata and the proof's
/// auxiliary commitments. No source tuple or auxiliary witness is provided.
pub fn verify_hidden_logup(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    query_columns: &[HiddenRelationTableStatement<'_>],
    table_columns: &[HiddenRelationTableStatement<'_>],
    proof: &HiddenLogUpProof,
) -> Result<(), HiddenLogUpError> {
    let query_columns = query_columns
        .iter()
        .copied()
        .map(MixedLogUpColumnStatement::Hidden)
        .collect::<Vec<_>>();
    let table_columns = table_columns
        .iter()
        .copied()
        .map(MixedLogUpColumnStatement::Hidden)
        .collect::<Vec<_>>();
    verify_mixed_hidden_logup(label, parameters, &query_columns, &table_columns, proof)
}

/// Verify a hidden LogUp reduction and return commitment-only opening claims.
pub fn verify_hidden_logup_deferred<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    query_columns: &[HiddenRelationTableStatement<'a>],
    table_columns: &[HiddenRelationTableStatement<'a>],
    proof: &'a HiddenLogUpReductionProof,
) -> Result<HiddenLogUpVerifierOpeningState<'a>, HiddenLogUpError> {
    let query_columns = query_columns
        .iter()
        .copied()
        .map(MixedLogUpColumnStatement::Hidden)
        .collect::<Vec<_>>();
    let table_columns = table_columns
        .iter()
        .copied()
        .map(MixedLogUpColumnStatement::Hidden)
        .collect::<Vec<_>>();
    verify_mixed_hidden_logup_deferred(label, parameters, &query_columns, &table_columns, proof)
}

/// Verify a tuple lookup with an arbitrary hidden/public source-column layout.
/// Public values are statement data, not proof data, and their MLE evaluations
/// are recomputed by the hidden-relation verifier.
pub fn verify_mixed_hidden_logup(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    query_columns: &[MixedLogUpColumnStatement<'_>],
    table_columns: &[MixedLogUpColumnStatement<'_>],
    proof: &HiddenLogUpProof,
) -> Result<(), HiddenLogUpError> {
    let reduction = HiddenLogUpReductionProof {
        auxiliary_commitments: proof.auxiliary_commitments.clone(),
        identity_reduction: proof.identity_reduction.clone(),
        sum_reduction: proof.sum_reduction.clone(),
    };
    let pending = verify_mixed_hidden_logup_deferred(
        label,
        parameters,
        query_columns,
        table_columns,
        &reduction,
    )?;
    verify_hidden_logup_openings_batch(parameters, &[&pending], &proof.relation_opening)
}

/// Verify a mixed LogUp reduction without reading any witness or PCS opening.
pub fn verify_mixed_hidden_logup_deferred<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    query_columns: &[MixedLogUpColumnStatement<'a>],
    table_columns: &[MixedLogUpColumnStatement<'a>],
    proof: &'a HiddenLogUpReductionProof,
) -> Result<HiddenLogUpVerifierOpeningState<'a>, HiddenLogUpError> {
    let logical_len = validate_mixed_statement_columns(query_columns, table_columns)?;
    let (mut transcript, compression_challenge) =
        source_transcript_and_challenge_mixed(label, query_columns, table_columns, logical_len)?;
    let denominator_challenge = append_pre_beta_commitments_and_challenge(
        &mut transcript,
        logical_len,
        &proof.auxiliary_commitments.query_keys,
        &proof.auxiliary_commitments.table_keys,
        &proof.auxiliary_commitments.multiplicities,
    );

    let query_key_statement =
        auxiliary_statement(&proof.auxiliary_commitments.query_keys, logical_len);
    let table_key_statement =
        auxiliary_statement(&proof.auxiliary_commitments.table_keys, logical_len);
    let multiplicity_statement =
        auxiliary_statement(&proof.auxiliary_commitments.multiplicities, logical_len);
    let query_inverse_statement =
        auxiliary_statement(&proof.auxiliary_commitments.query_inverses, logical_len);
    let table_inverse_statement =
        auxiliary_statement(&proof.auxiliary_commitments.table_inverses, logical_len);

    let (query_hidden, query_public, query_expression) =
        verifier_key_relation_inputs(query_columns, query_key_statement, compression_challenge);
    let (table_hidden, table_public, table_expression) =
        verifier_key_relation_inputs(table_columns, table_key_statement, compression_challenge);
    let inverse_expression = inverse_expression(denominator_challenge);
    let query_inverse_tables = [query_key_statement, query_inverse_statement];
    let table_inverse_tables = [table_key_statement, table_inverse_statement];
    let identity_items = [
        HiddenRelationBatchVerifierItem {
            tables: &query_hidden,
            public_tables: &query_public,
            expression: &query_expression,
        },
        HiddenRelationBatchVerifierItem {
            tables: &table_hidden,
            public_tables: &table_public,
            expression: &table_expression,
        },
        HiddenRelationBatchVerifierItem {
            tables: &query_inverse_tables,
            public_tables: &[],
            expression: &inverse_expression,
        },
        HiddenRelationBatchVerifierItem {
            tables: &table_inverse_tables,
            public_tables: &[],
            expression: &inverse_expression,
        },
    ];
    let identity_pending = verify_hidden_relations_batch_deferred(
        &relation_label(label, b"identity-batch"),
        parameters,
        &identity_items,
        &proof.identity_reduction,
    )?;
    let sum_tables = [
        query_inverse_statement,
        multiplicity_statement,
        table_inverse_statement,
    ];
    let sum_expression = sum_balance_expression();
    let sum_items = [HiddenRelationBatchVerifierItem {
        tables: &sum_tables,
        public_tables: &[],
        expression: &sum_expression,
    }];
    let sum_pending = verify_hidden_sum_relations_batch_deferred(
        &relation_label(label, b"sum-batch"),
        parameters,
        &sum_items,
        &proof.sum_reduction,
    )?;

    let relation_claims = [identity_pending.into_claims(), sum_pending.into_claims()];
    Ok(HiddenLogUpVerifierOpeningState {
        relation_claims,
        tables: HiddenLogUpVerifierTables::Single(HiddenLogUpSingleVerifierTables {
            query_hidden_tables: hidden_statement_tables(query_columns),
            table_hidden_tables: hidden_statement_tables(table_columns),
            query_key: query_key_statement,
            table_key: table_key_statement,
            multiplicities: multiplicity_statement,
            query_inverses: query_inverse_statement,
            table_inverses: table_inverse_statement,
        }),
    })
}

/// Verify a multipart LogUp without receiving any hidden source or auxiliary
/// values. Public source columns remain verifier-known statement data.
pub fn verify_hidden_logup_parts(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    query_parts: &[Vec<MixedLogUpColumnStatement<'_>>],
    table_parts: &[Vec<MixedLogUpColumnStatement<'_>>],
    proof: &HiddenLogUpPartsProof,
) -> Result<(), HiddenLogUpError> {
    let reduction = HiddenLogUpPartsReductionProof {
        auxiliary_commitments: proof.auxiliary_commitments.clone(),
        identity_reduction: proof.identity_reduction.clone(),
        sum_reduction: proof.sum_reduction.clone(),
    };
    let pending = verify_hidden_logup_parts_deferred(
        label,
        parameters,
        query_parts,
        table_parts,
        &reduction,
    )?;
    verify_hidden_logup_openings_batch(parameters, &[&pending], &proof.relation_opening)
}

/// Verify a multipart LogUp reduction and return its terminal opening claims.
pub fn verify_hidden_logup_parts_deferred<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    query_parts: &[Vec<MixedLogUpColumnStatement<'a>>],
    table_parts: &[Vec<MixedLogUpColumnStatement<'a>>],
    proof: &'a HiddenLogUpPartsReductionProof,
) -> Result<HiddenLogUpVerifierOpeningState<'a>, HiddenLogUpError> {
    let (logical_len, tuple_width) = validate_mixed_statement_parts(query_parts, table_parts)?;
    validate_parts_proof_shape(query_parts.len(), table_parts.len())?;
    let (mut transcript, compression_challenge) = source_parts_transcript_and_challenge(
        label,
        query_parts,
        table_parts,
        logical_len,
        tuple_width,
    )?;
    let aux = &proof.auxiliary_commitments;
    let denominator_challenge = append_pre_beta_part_commitments_and_challenge(
        &mut transcript,
        logical_len,
        query_parts.len(),
        table_parts.len(),
        &aux.query_keys,
        &aux.table_keys,
        &aux.multiplicities,
    );

    let part_batch_challenge = append_post_inverse_parts_and_challenge(
        &mut transcript,
        query_parts.len(),
        table_parts.len(),
        &aux.query_inverses,
        &aux.table_inverses,
    );
    let query_key_statements =
        auxiliary_matrix_statements(&aux.query_keys, logical_len, query_parts.len());
    let table_key_statements =
        auxiliary_matrix_statements(&aux.table_keys, logical_len, table_parts.len());
    let multiplicity_statements =
        auxiliary_matrix_statements(&aux.multiplicities, logical_len, table_parts.len());
    let query_inverse_statements =
        auxiliary_matrix_statements(&aux.query_inverses, logical_len, query_parts.len());
    let table_inverse_statements =
        auxiliary_matrix_statements(&aux.table_inverses, logical_len, table_parts.len());

    let (query_key_hidden, query_key_public, query_key_expression) =
        verifier_batched_key_relation_inputs(
            query_parts,
            &query_key_statements,
            compression_challenge,
            part_batch_challenge,
        );
    let query_inverse_tables =
        interleave_statements(&query_key_statements, &query_inverse_statements);
    let query_inverse_expression = batched_inverse_expression(
        query_parts.len(),
        denominator_challenge,
        part_batch_challenge,
    );

    let (table_key_hidden, table_key_public, table_key_expression) =
        verifier_batched_key_relation_inputs(
            table_parts,
            &table_key_statements,
            compression_challenge,
            part_batch_challenge,
        );
    let table_inverse_tables =
        interleave_statements(&table_key_statements, &table_inverse_statements);
    let table_inverse_expression = batched_inverse_expression(
        table_parts.len(),
        denominator_challenge,
        part_batch_challenge,
    );
    let identity_items = [
        HiddenRelationBatchVerifierItem {
            tables: &query_key_hidden,
            public_tables: &query_key_public,
            expression: &query_key_expression,
        },
        HiddenRelationBatchVerifierItem {
            tables: &table_key_hidden,
            public_tables: &table_key_public,
            expression: &table_key_expression,
        },
        HiddenRelationBatchVerifierItem {
            tables: &query_inverse_tables,
            public_tables: &[],
            expression: &query_inverse_expression,
        },
        HiddenRelationBatchVerifierItem {
            tables: &table_inverse_tables,
            public_tables: &[],
            expression: &table_inverse_expression,
        },
    ];
    let identity_pending = verify_hidden_relations_batch_deferred(
        &relation_label(label, b"parts-identity-batch"),
        parameters,
        &identity_items,
        &proof.identity_reduction,
    )?;

    let mut sum_tables = query_inverse_statements.clone();
    for part in 0..table_parts.len() {
        sum_tables.push(multiplicity_statements[part]);
        sum_tables.push(table_inverse_statements[part]);
    }
    let sum_expression = parts_sum_balance_expression(query_parts.len(), table_parts.len());
    let sum_items = [HiddenRelationBatchVerifierItem {
        tables: &sum_tables,
        public_tables: &[],
        expression: &sum_expression,
    }];
    let sum_pending = verify_hidden_sum_relations_batch_deferred(
        &relation_label(label, b"parts-sum-batch"),
        parameters,
        &sum_items,
        &proof.sum_reduction,
    )?;

    let relation_claims = [identity_pending.into_claims(), sum_pending.into_claims()];
    Ok(HiddenLogUpVerifierOpeningState {
        relation_claims,
        tables: HiddenLogUpVerifierTables::Parts(HiddenLogUpPartsVerifierTables {
            query_hidden_parts: query_parts
                .iter()
                .map(|part| hidden_statement_tables(part))
                .collect(),
            table_hidden_parts: table_parts
                .iter()
                .map(|part| hidden_statement_tables(part))
                .collect(),
            query_keys: query_key_statements,
            table_keys: table_key_statements,
            multiplicities: multiplicity_statements,
            query_inverses: query_inverse_statements,
            table_inverses: table_inverse_statements,
        }),
    })
}

/// Verify a multipart LogUp reduction whose terminal claims will be opened on
/// the shared Section 6 Dory grid.
pub fn verify_hidden_logup_parts_deferred_on_grid<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    grid: &Section6DoryGrid,
    query_parts: &[Vec<MixedLogUpColumnStatement<'a>>],
    table_parts: &[Vec<MixedLogUpColumnStatement<'a>>],
    proof: &'a HiddenLogUpPartsReductionProof,
) -> Result<HiddenLogUpVerifierOpeningState<'a>, HiddenLogUpError> {
    let (logical_len, _) = validate_mixed_statement_parts(query_parts, table_parts)?;
    validate_part_matrices_fit_grid(logical_len, query_parts.len(), table_parts.len(), grid)?;
    verify_hidden_logup_parts_deferred(label, parameters, query_parts, table_parts, proof)
}

fn validate_parts_proof_shape(
    query_part_count: usize,
    table_part_count: usize,
) -> Result<(), HiddenLogUpError> {
    if query_part_count == 0 || table_part_count == 0 {
        return Err(HiddenLogUpError::ProofShape);
    }
    Ok(())
}

fn validate_mixed_prover_parts(
    query_parts: &[Vec<MixedLogUpColumn<'_>>],
    table_parts: &[Vec<MixedLogUpColumn<'_>>],
) -> Result<(usize, usize), HiddenLogUpError> {
    let query_statements = prover_part_statements(query_parts);
    let table_statements = prover_part_statements(table_parts);
    let (logical_len, tuple_width) =
        validate_mixed_statement_parts(&query_statements, &table_statements)?;
    for (side, parts) in [
        (HiddenLogUpSide::Query, query_parts),
        (HiddenLogUpSide::Table, table_parts),
    ] {
        for (part, columns) in parts.iter().enumerate() {
            for (column, source) in columns.iter().enumerate() {
                if let MixedLogUpColumn::Hidden(source) = source {
                    if source.values.len() != logical_len {
                        return Err(HiddenLogUpError::PartSourceValueLength {
                            side,
                            part,
                            column,
                            expected: logical_len,
                            actual: source.values.len(),
                        });
                    }
                }
            }
        }
    }
    Ok((logical_len, tuple_width))
}

fn validate_mixed_statement_parts(
    query_parts: &[Vec<MixedLogUpColumnStatement<'_>>],
    table_parts: &[Vec<MixedLogUpColumnStatement<'_>>],
) -> Result<(usize, usize), HiddenLogUpError> {
    if query_parts.is_empty() {
        return Err(HiddenLogUpError::EmptyParts {
            side: HiddenLogUpSide::Query,
        });
    }
    if table_parts.is_empty() {
        return Err(HiddenLogUpError::EmptyParts {
            side: HiddenLogUpSide::Table,
        });
    }
    let tuple_width = query_parts[0].len();
    if tuple_width == 0 {
        return Err(HiddenLogUpError::EmptyTuple);
    }
    if table_parts[0].len() != tuple_width {
        return Err(HiddenLogUpError::TupleWidth {
            query: tuple_width,
            table: table_parts[0].len(),
        });
    }
    let logical_len = mixed_statement_len(&query_parts[0][0]);
    if logical_len < 2 || !logical_len.is_power_of_two() {
        return Err(HiddenLogUpError::InvalidLogicalLength(logical_len));
    }
    for (side, parts) in [
        (HiddenLogUpSide::Query, query_parts),
        (HiddenLogUpSide::Table, table_parts),
    ] {
        for (part, columns) in parts.iter().enumerate() {
            if columns.len() != tuple_width {
                return Err(HiddenLogUpError::PartTupleWidth {
                    side,
                    part,
                    expected: tuple_width,
                    actual: columns.len(),
                });
            }
            for (column, statement) in columns.iter().enumerate() {
                let actual = mixed_statement_len(statement);
                if actual != logical_len {
                    return Err(HiddenLogUpError::PartColumnLength {
                        side,
                        part,
                        column,
                        expected: logical_len,
                        actual,
                    });
                }
            }
        }
    }
    Ok((logical_len, tuple_width))
}

fn prover_part_statements<'a>(
    parts: &[Vec<MixedLogUpColumn<'a>>],
) -> Vec<Vec<MixedLogUpColumnStatement<'a>>> {
    parts
        .iter()
        .map(|part| prover_column_statements(part))
        .collect()
}

fn source_parts_transcript_and_challenge(
    label: &[u8],
    query_parts: &[Vec<MixedLogUpColumnStatement<'_>>],
    table_parts: &[Vec<MixedLogUpColumnStatement<'_>>],
    logical_len: usize,
    tuple_width: usize,
) -> Result<(Blake2bTranscript<JoltFr>, ArkFr), HiddenLogUpError> {
    let mut transcript = Blake2bTranscript::<JoltFr>::new(LOGUP_TRANSCRIPT_DOMAIN);
    transcript.append(&Label(b"lu_parts_sources"));
    transcript.append(&LabelWithCount(b"lu_label", label.len() as u64));
    transcript.append_bytes(label);
    transcript.append_labeled(b"lu_len", &U64Word(logical_len as u64));
    transcript.append_labeled(b"lu_width", &U64Word(tuple_width as u64));
    transcript.append(&LabelWithCount(b"lu_query_parts", query_parts.len() as u64));
    for (part, columns) in query_parts.iter().enumerate() {
        transcript.append_labeled(b"lu_query_part", &U64Word(part as u64));
        transcript.append(&LabelWithCount(b"lu_query_cols", columns.len() as u64));
        for column in columns {
            append_source_column(&mut transcript, HiddenLogUpSide::Query, column)?;
        }
    }
    transcript.append(&LabelWithCount(b"lu_table_parts", table_parts.len() as u64));
    for (part, columns) in table_parts.iter().enumerate() {
        transcript.append_labeled(b"lu_table_part", &U64Word(part as u64));
        transcript.append(&LabelWithCount(b"lu_table_cols", columns.len() as u64));
        for column in columns {
            append_source_column(&mut transcript, HiddenLogUpSide::Table, column)?;
        }
    }
    let challenge = ArkFr::from(transcript.challenge());
    if challenge.is_zero() {
        return Err(HiddenLogUpError::ZeroCompressionChallenge);
    }
    Ok((transcript, challenge))
}

fn compress_prover_parts(
    parts: &[Vec<MixedLogUpColumn<'_>>],
    challenge: ArkFr,
    logical_len: usize,
) -> Result<Vec<Vec<ArkFr>>, HiddenLogUpError> {
    parts
        .iter()
        .map(|part| {
            let columns = part
                .iter()
                .map(mixed_prover_values)
                .collect::<Result<Vec<_>, _>>()?;
            Ok(compress_tuple_columns(&columns, challenge, logical_len))
        })
        .collect()
}

fn compute_part_multiplicities(
    query_keys: &[Vec<ArkFr>],
    table_keys: &[Vec<ArkFr>],
) -> Result<Vec<Vec<ArkFr>>, HiddenLogUpError> {
    let table_len = table_keys.first().map_or(0, Vec::len);
    let mut first_table_position = HashMap::with_capacity(table_keys.len() * table_len);
    for (part, keys) in table_keys.iter().enumerate() {
        for (index, key) in keys.iter().copied().enumerate() {
            first_table_position.entry(key).or_insert((part, index));
        }
    }
    let mut counts = vec![vec![0_u64; table_len]; table_keys.len()];
    let mut query_index = 0;
    for keys in query_keys {
        for key in keys {
            let Some(&(part, index)) = first_table_position.get(key) else {
                return Err(HiddenLogUpError::MissingQueryTuple { query_index });
            };
            counts[part][index] = counts[part][index].checked_add(1).ok_or(
                HiddenLogUpError::MultiplicityOverflow {
                    table_index: part * table_len + index,
                },
            )?;
            query_index += 1;
        }
    }
    Ok(counts
        .into_iter()
        .map(|part| part.into_iter().map(ArkFr::from).collect())
        .collect())
}

fn batch_inverse_part_denominators(
    query_keys: &[Vec<ArkFr>],
    table_keys: &[Vec<ArkFr>],
    beta: ArkFr,
    logical_len: usize,
) -> Result<Vec<ArkFr>, HiddenLogUpError> {
    let mut denominators = Vec::with_capacity((query_keys.len() + table_keys.len()) * logical_len);
    for keys in query_keys.iter().chain(table_keys) {
        denominators.extend(keys.iter().map(|key| beta + key));
    }
    for (flat_index, denominator) in denominators.iter().enumerate() {
        if denominator.is_zero() {
            let part = flat_index / logical_len;
            let index = flat_index % logical_len;
            let (side, part) = if part < query_keys.len() {
                (HiddenLogUpSide::Query, part)
            } else {
                (HiddenLogUpSide::Table, part - query_keys.len())
            };
            return Err(HiddenLogUpError::SingularPartDenominator { side, part, index });
        }
    }
    batch_inversion(&mut denominators);
    Ok(denominators)
}

fn inverse_part(values: &[ArkFr], part: usize, logical_len: usize) -> &[ArkFr] {
    &values[part * logical_len..(part + 1) * logical_len]
}

fn append_pre_beta_parts_and_challenge(
    transcript: &mut Blake2bTranscript<JoltFr>,
    logical_len: usize,
    query_keys: &[CommittedDoryFieldVector],
    table_keys: &[CommittedDoryFieldVector],
    multiplicities: &[CommittedDoryFieldVector],
) -> ArkFr {
    append_pre_beta_part_commitments_and_challenge(
        transcript,
        logical_len,
        query_keys.len(),
        table_keys.len(),
        &query_keys[0].commitment,
        &table_keys[0].commitment,
        &multiplicities[0].commitment,
    )
}

fn append_pre_beta_part_commitments_and_challenge(
    transcript: &mut Blake2bTranscript<JoltFr>,
    logical_len: usize,
    query_part_count: usize,
    table_part_count: usize,
    query_keys: &DoryCommitment,
    table_keys: &DoryCommitment,
    multiplicities: &DoryCommitment,
) -> ArkFr {
    transcript.append(&Label(b"lu_parts_pre_beta"));
    transcript.append_labeled(b"lu_query_key_parts", &U64Word(query_part_count as u64));
    transcript.append_labeled(b"lu_query_keys", query_keys);
    transcript.append_labeled(b"lu_table_key_parts", &U64Word(table_part_count as u64));
    transcript.append_labeled(b"lu_table_keys", table_keys);
    transcript.append_labeled(b"lu_mult", multiplicities);
    transcript.append_labeled(b"lu_aux_len", &U64Word(logical_len as u64));
    ArkFr::from(transcript.challenge())
}

fn append_post_inverse_parts_and_challenge(
    transcript: &mut Blake2bTranscript<JoltFr>,
    query_part_count: usize,
    table_part_count: usize,
    query_inverses: &DoryCommitment,
    table_inverses: &DoryCommitment,
) -> ArkFr {
    transcript.append(&Label(b"lu_parts_post_inverse"));
    transcript.append_labeled(b"lu_query_inverse_parts", &U64Word(query_part_count as u64));
    transcript.append_labeled(b"lu_query_inverses", query_inverses);
    transcript.append_labeled(b"lu_table_inverse_parts", &U64Word(table_part_count as u64));
    transcript.append_labeled(b"lu_table_inverses", table_inverses);
    let mut challenge = ArkFr::from(transcript.challenge());
    let mut retry = 0_u64;
    while challenge.is_zero() {
        retry += 1;
        transcript.append_labeled(b"lu_batch_retry", &U64Word(retry));
        challenge = ArkFr::from(transcript.challenge());
    }
    challenge
}

fn prover_parts_sum_tables<'a>(
    all_inverses: &'a [ArkFr],
    logical_len: usize,
    multiplicities: &'a [Vec<ArkFr>],
    query_inverses: &'a [CommittedDoryFieldVector],
    committed_multiplicities: &'a [CommittedDoryFieldVector],
    table_inverses: &'a [CommittedDoryFieldVector],
) -> Vec<HiddenRelationTable<'a>> {
    let mut tables = Vec::with_capacity(query_inverses.len() + 2 * table_inverses.len());
    for (part, committed) in query_inverses.iter().enumerate() {
        tables.push(HiddenRelationTable {
            values: inverse_part(all_inverses, part, logical_len),
            committed,
        });
    }
    for part in 0..table_inverses.len() {
        tables.push(HiddenRelationTable {
            values: &multiplicities[part],
            committed: &committed_multiplicities[part],
        });
        tables.push(HiddenRelationTable {
            values: inverse_part(all_inverses, query_inverses.len() + part, logical_len),
            committed: &table_inverses[part],
        });
    }
    tables
}

fn parts_sum_balance_expression(
    query_part_count: usize,
    table_part_count: usize,
) -> RelationExpression {
    let query_sum = (0..query_part_count)
        .fold(RelationExpression::constant(ArkFr::zero()), |sum, index| {
            sum.add(RelationExpression::table(index))
        });
    (0..table_part_count).fold(query_sum, |balance, part| {
        let multiplicity = query_part_count + 2 * part;
        balance.sub(
            RelationExpression::table(multiplicity)
                .mul(RelationExpression::table(multiplicity + 1)),
        )
    })
}

fn prover_batched_key_relation_inputs<'source, 'output>(
    parts: &'output [Vec<MixedLogUpColumn<'source>>],
    key_values: &'output [Vec<ArkFr>],
    committed_keys: &'output [CommittedDoryFieldVector],
    compression_challenge: ArkFr,
    batch_challenge: ArkFr,
) -> (
    Vec<HiddenRelationTable<'output>>,
    Vec<PublicRelationTable<'output>>,
    RelationExpression,
)
where
    'source: 'output,
{
    let total_hidden = parts.len()
        + parts
            .iter()
            .flatten()
            .filter(|column| matches!(column, MixedLogUpColumn::Hidden(_)))
            .count();
    let compression_coefficients = compression_coefficients(parts[0].len(), compression_challenge);
    let mut hidden = Vec::with_capacity(total_hidden);
    let mut public = Vec::new();
    let mut expression = RelationExpression::constant(ArkFr::zero());
    let mut batch_coefficient = ArkFr::one();
    for (part, columns) in parts.iter().enumerate() {
        let mut compressed = RelationExpression::constant(ArkFr::zero());
        for (column, compression_coefficient) in columns.iter().zip(&compression_coefficients) {
            let table_index = match column {
                MixedLogUpColumn::Hidden(column) => {
                    let index = hidden.len();
                    hidden.push(HiddenRelationTable {
                        values: column.values,
                        committed: column.committed,
                    });
                    index
                }
                MixedLogUpColumn::Public(column) => {
                    let index = total_hidden + public.len();
                    public.push(*column);
                    index
                }
            };
            compressed = compressed.add(
                RelationExpression::constant(*compression_coefficient)
                    .mul(RelationExpression::table(table_index)),
            );
        }
        let key_index = hidden.len();
        hidden.push(HiddenRelationTable {
            values: &key_values[part],
            committed: &committed_keys[part],
        });
        expression = expression.add(
            RelationExpression::constant(batch_coefficient)
                .mul(RelationExpression::table(key_index).sub(compressed)),
        );
        batch_coefficient *= batch_challenge;
    }
    (hidden, public, expression)
}

fn verifier_batched_key_relation_inputs<'a>(
    parts: &[Vec<MixedLogUpColumnStatement<'a>>],
    key_statements: &[HiddenRelationTableStatement<'a>],
    compression_challenge: ArkFr,
    batch_challenge: ArkFr,
) -> (
    Vec<HiddenRelationTableStatement<'a>>,
    Vec<PublicRelationTable<'a>>,
    RelationExpression,
) {
    let total_hidden = parts.len()
        + parts
            .iter()
            .flatten()
            .filter(|column| matches!(column, MixedLogUpColumnStatement::Hidden(_)))
            .count();
    let compression_coefficients = compression_coefficients(parts[0].len(), compression_challenge);
    let mut hidden = Vec::with_capacity(total_hidden);
    let mut public = Vec::new();
    let mut expression = RelationExpression::constant(ArkFr::zero());
    let mut batch_coefficient = ArkFr::one();
    for (part, columns) in parts.iter().enumerate() {
        let mut compressed = RelationExpression::constant(ArkFr::zero());
        for (column, compression_coefficient) in columns.iter().zip(&compression_coefficients) {
            let table_index = match column {
                MixedLogUpColumnStatement::Hidden(column) => {
                    let index = hidden.len();
                    hidden.push(*column);
                    index
                }
                MixedLogUpColumnStatement::Public(column) => {
                    let index = total_hidden + public.len();
                    public.push(*column);
                    index
                }
            };
            compressed = compressed.add(
                RelationExpression::constant(*compression_coefficient)
                    .mul(RelationExpression::table(table_index)),
            );
        }
        let key_index = hidden.len();
        hidden.push(key_statements[part]);
        expression = expression.add(
            RelationExpression::constant(batch_coefficient)
                .mul(RelationExpression::table(key_index).sub(compressed)),
        );
        batch_coefficient *= batch_challenge;
    }
    (hidden, public, expression)
}

#[allow(clippy::too_many_arguments)]
fn prover_batched_inverse_inputs<'a>(
    keys: &'a [Vec<ArkFr>],
    committed_keys: &'a [CommittedDoryFieldVector],
    all_inverses: &'a [ArkFr],
    inverse_offset: usize,
    logical_len: usize,
    committed_inverses: &'a [CommittedDoryFieldVector],
    beta: ArkFr,
    batch_challenge: ArkFr,
) -> (Vec<HiddenRelationTable<'a>>, RelationExpression) {
    let mut tables = Vec::with_capacity(keys.len() * 2);
    for part in 0..keys.len() {
        tables.push(HiddenRelationTable {
            values: &keys[part],
            committed: &committed_keys[part],
        });
        tables.push(HiddenRelationTable {
            values: inverse_part(all_inverses, inverse_offset + part, logical_len),
            committed: &committed_inverses[part],
        });
    }
    (
        tables,
        batched_inverse_expression(keys.len(), beta, batch_challenge),
    )
}

fn batched_inverse_expression(
    part_count: usize,
    beta: ArkFr,
    batch_challenge: ArkFr,
) -> RelationExpression {
    let mut expression = RelationExpression::constant(ArkFr::zero());
    let mut coefficient = ArkFr::one();
    for part in 0..part_count {
        let key = RelationExpression::table(2 * part);
        let inverse = RelationExpression::table(2 * part + 1);
        let equation = inverse
            .mul(RelationExpression::constant(beta).add(key))
            .sub(RelationExpression::constant(ArkFr::one()));
        expression = expression.add(RelationExpression::constant(coefficient).mul(equation));
        coefficient *= batch_challenge;
    }
    expression
}

fn interleave_statements<'a>(
    left: &[HiddenRelationTableStatement<'a>],
    right: &[HiddenRelationTableStatement<'a>],
) -> Vec<HiddenRelationTableStatement<'a>> {
    left.iter()
        .zip(right)
        .flat_map(|(left, right)| [*left, *right])
        .collect()
}

fn validate_mixed_prover_columns(
    query_columns: &[MixedLogUpColumn<'_>],
    table_columns: &[MixedLogUpColumn<'_>],
) -> Result<usize, HiddenLogUpError> {
    let query_statements = prover_column_statements(query_columns);
    let table_statements = prover_column_statements(table_columns);
    let logical_len = validate_mixed_statement_columns(&query_statements, &table_statements)?;
    for (side, columns) in [
        (HiddenLogUpSide::Query, query_columns),
        (HiddenLogUpSide::Table, table_columns),
    ] {
        for (column, source) in columns.iter().enumerate() {
            if let MixedLogUpColumn::Hidden(source) = source {
                if source.values.len() != logical_len {
                    return Err(HiddenLogUpError::SourceValueLength {
                        side,
                        column,
                        expected: logical_len,
                        actual: source.values.len(),
                    });
                }
            }
        }
    }
    Ok(logical_len)
}

fn validate_mixed_statement_columns(
    query_columns: &[MixedLogUpColumnStatement<'_>],
    table_columns: &[MixedLogUpColumnStatement<'_>],
) -> Result<usize, HiddenLogUpError> {
    if query_columns.is_empty() || table_columns.is_empty() {
        return Err(HiddenLogUpError::EmptyTuple);
    }
    if query_columns.len() != table_columns.len() {
        return Err(HiddenLogUpError::TupleWidth {
            query: query_columns.len(),
            table: table_columns.len(),
        });
    }
    let logical_len = mixed_statement_len(&query_columns[0]);
    if logical_len < 2 || !logical_len.is_power_of_two() {
        return Err(HiddenLogUpError::InvalidLogicalLength(logical_len));
    }
    for (side, columns) in [
        (HiddenLogUpSide::Query, query_columns),
        (HiddenLogUpSide::Table, table_columns),
    ] {
        for (column, statement) in columns.iter().enumerate() {
            let actual = mixed_statement_len(statement);
            if actual != logical_len {
                return Err(HiddenLogUpError::ColumnLength {
                    side,
                    column,
                    expected: logical_len,
                    actual,
                });
            }
        }
    }
    Ok(logical_len)
}

fn mixed_prover_values<'a>(column: &MixedLogUpColumn<'a>) -> Result<&'a [ArkFr], HiddenLogUpError> {
    match column {
        MixedLogUpColumn::Hidden(column) => Ok(column.values),
        MixedLogUpColumn::Public(column) => match column {
            PublicRelationTable::Dense(values) => Ok(values),
            PublicRelationTable::Structured { values, .. } => values.ok_or_else(|| {
                HiddenLogUpError::Encoding(
                    "structured public column is missing prover entries".to_owned(),
                )
            }),
        },
    }
}

fn hidden_prover_tables<'a>(columns: &[MixedLogUpColumn<'a>]) -> Vec<HiddenRelationTable<'a>> {
    columns
        .iter()
        .filter_map(|column| match column {
            MixedLogUpColumn::Hidden(table) => Some(*table),
            MixedLogUpColumn::Public(_) => None,
        })
        .collect()
}

fn hidden_statement_tables<'a>(
    columns: &[MixedLogUpColumnStatement<'a>],
) -> Vec<HiddenRelationTableStatement<'a>> {
    columns
        .iter()
        .filter_map(|column| match column {
            MixedLogUpColumnStatement::Hidden(table) => Some(*table),
            MixedLogUpColumnStatement::Public(_) => None,
        })
        .collect()
}

fn prover_column_statements<'a>(
    columns: &[MixedLogUpColumn<'a>],
) -> Vec<MixedLogUpColumnStatement<'a>> {
    columns
        .iter()
        .map(|column| match column {
            MixedLogUpColumn::Hidden(column) => MixedLogUpColumnStatement::Hidden(
                HiddenRelationTableStatement::from(column.committed),
            ),
            MixedLogUpColumn::Public(column) => MixedLogUpColumnStatement::Public(*column),
        })
        .collect()
}

fn mixed_statement_len(column: &MixedLogUpColumnStatement<'_>) -> usize {
    match column {
        MixedLogUpColumnStatement::Hidden(column) => column.logical_len,
        MixedLogUpColumnStatement::Public(column) => column.logical_len(),
    }
}

#[cfg(test)]
fn source_transcript_and_challenge(
    label: &[u8],
    query_columns: &[HiddenRelationTableStatement<'_>],
    table_columns: &[HiddenRelationTableStatement<'_>],
    logical_len: usize,
) -> Result<(Blake2bTranscript<JoltFr>, ArkFr), HiddenLogUpError> {
    let query_columns = query_columns
        .iter()
        .copied()
        .map(MixedLogUpColumnStatement::Hidden)
        .collect::<Vec<_>>();
    let table_columns = table_columns
        .iter()
        .copied()
        .map(MixedLogUpColumnStatement::Hidden)
        .collect::<Vec<_>>();
    source_transcript_and_challenge_mixed(label, &query_columns, &table_columns, logical_len)
}

fn source_transcript_and_challenge_mixed(
    label: &[u8],
    query_columns: &[MixedLogUpColumnStatement<'_>],
    table_columns: &[MixedLogUpColumnStatement<'_>],
    logical_len: usize,
) -> Result<(Blake2bTranscript<JoltFr>, ArkFr), HiddenLogUpError> {
    let mut transcript = Blake2bTranscript::<JoltFr>::new(LOGUP_TRANSCRIPT_DOMAIN);
    transcript.append(&Label(b"lu_sources"));
    transcript.append(&LabelWithCount(b"lu_label", label.len() as u64));
    transcript.append_bytes(label);
    transcript.append_labeled(b"lu_len", &U64Word(logical_len as u64));
    transcript.append(&LabelWithCount(
        b"lu_query_cols",
        query_columns.len() as u64,
    ));
    for statement in query_columns {
        append_source_column(&mut transcript, HiddenLogUpSide::Query, statement)?;
    }
    transcript.append(&LabelWithCount(
        b"lu_table_cols",
        table_columns.len() as u64,
    ));
    for statement in table_columns {
        append_source_column(&mut transcript, HiddenLogUpSide::Table, statement)?;
    }
    let challenge = ArkFr::from(transcript.challenge());
    if challenge.is_zero() {
        return Err(HiddenLogUpError::ZeroCompressionChallenge);
    }
    Ok((transcript, challenge))
}

fn append_source_column(
    transcript: &mut Blake2bTranscript<JoltFr>,
    side: HiddenLogUpSide,
    statement: &MixedLogUpColumnStatement<'_>,
) -> Result<(), HiddenLogUpError> {
    match (side, statement) {
        (HiddenLogUpSide::Query, MixedLogUpColumnStatement::Hidden(statement)) => {
            transcript.append_labeled(b"lu_query_com", statement.commitment);
            transcript.append_labeled(b"lu_query_len", &U64Word(statement.logical_len as u64));
        }
        (HiddenLogUpSide::Table, MixedLogUpColumnStatement::Hidden(statement)) => {
            transcript.append_labeled(b"lu_table_com", statement.commitment);
            transcript.append_labeled(b"lu_table_len", &U64Word(statement.logical_len as u64));
        }
        (HiddenLogUpSide::Query, MixedLogUpColumnStatement::Public(statement)) => {
            let digest = statement
                .transcript_digest()
                .map_err(HiddenLogUpError::Relation)?;
            transcript.append(&LabelWithCount(b"lu_query_pub", digest.len() as u64));
            transcript.append_bytes(&digest);
            transcript.append_labeled(
                b"lu_query_pub_len",
                &U64Word(statement.logical_len() as u64),
            );
        }
        (HiddenLogUpSide::Table, MixedLogUpColumnStatement::Public(statement)) => {
            let digest = statement
                .transcript_digest()
                .map_err(HiddenLogUpError::Relation)?;
            transcript.append(&LabelWithCount(b"lu_table_pub", digest.len() as u64));
            transcript.append_bytes(&digest);
            transcript.append_labeled(
                b"lu_table_pub_len",
                &U64Word(statement.logical_len() as u64),
            );
        }
    }
    Ok(())
}

fn prover_key_relation_inputs<'source, 'output>(
    columns: &[MixedLogUpColumn<'source>],
    key_values: &'output [ArkFr],
    committed_key: &'output CommittedDoryFieldVector,
    challenge: ArkFr,
) -> (
    Vec<HiddenRelationTable<'output>>,
    Vec<PublicRelationTable<'output>>,
    RelationExpression,
)
where
    'source: 'output,
{
    let mut hidden = columns
        .iter()
        .filter_map(|column| match column {
            MixedLogUpColumn::Hidden(column) => Some(HiddenRelationTable {
                values: column.values,
                committed: column.committed,
            }),
            MixedLogUpColumn::Public(_) => None,
        })
        .collect::<Vec<_>>();
    hidden.push(HiddenRelationTable {
        values: key_values,
        committed: committed_key,
    });
    let public = columns
        .iter()
        .filter_map(|column| match column {
            MixedLogUpColumn::Hidden(_) => None,
            MixedLogUpColumn::Public(column) => Some(*column),
        })
        .collect::<Vec<_>>();
    let expression = mixed_key_zero_expression(
        columns
            .iter()
            .map(|column| matches!(column, MixedLogUpColumn::Hidden(_))),
        challenge,
    );
    (hidden, public, expression)
}

fn verifier_key_relation_inputs<'a>(
    columns: &[MixedLogUpColumnStatement<'a>],
    key_statement: HiddenRelationTableStatement<'a>,
    challenge: ArkFr,
) -> (
    Vec<HiddenRelationTableStatement<'a>>,
    Vec<PublicRelationTable<'a>>,
    RelationExpression,
) {
    let mut hidden = columns
        .iter()
        .filter_map(|column| match column {
            MixedLogUpColumnStatement::Hidden(column) => Some(*column),
            MixedLogUpColumnStatement::Public(_) => None,
        })
        .collect::<Vec<_>>();
    hidden.push(key_statement);
    let public = columns
        .iter()
        .filter_map(|column| match column {
            MixedLogUpColumnStatement::Hidden(_) => None,
            MixedLogUpColumnStatement::Public(column) => Some(*column),
        })
        .collect::<Vec<_>>();
    let expression = mixed_key_zero_expression(
        columns
            .iter()
            .map(|column| matches!(column, MixedLogUpColumnStatement::Hidden(_))),
        challenge,
    );
    (hidden, public, expression)
}

fn mixed_key_zero_expression(
    hidden_positions: impl IntoIterator<Item = bool>,
    challenge: ArkFr,
) -> RelationExpression {
    let hidden_positions = hidden_positions.into_iter().collect::<Vec<_>>();
    let hidden_count = hidden_positions.iter().filter(|&&hidden| hidden).count();
    let mut hidden_index = 0;
    let mut public_index = 0;
    let mut coefficient = ArkFr::one();
    let compressed = hidden_positions.into_iter().fold(
        RelationExpression::constant(ArkFr::zero()),
        |sum, hidden| {
            let table_index = if hidden {
                let index = hidden_index;
                hidden_index += 1;
                index
            } else {
                let index = hidden_count + 1 + public_index;
                public_index += 1;
                index
            };
            let term = RelationExpression::constant(coefficient)
                .mul(RelationExpression::table(table_index));
            coefficient *= challenge;
            sum.add(term)
        },
    );
    RelationExpression::table(hidden_count).sub(compressed)
}

fn append_pre_beta_commitments_and_challenge(
    transcript: &mut Blake2bTranscript<JoltFr>,
    logical_len: usize,
    query_keys: &DoryCommitment,
    table_keys: &DoryCommitment,
    multiplicities: &DoryCommitment,
) -> ArkFr {
    transcript.append(&Label(b"lu_pre_beta"));
    transcript.append_labeled(b"lu_query_keys", query_keys);
    transcript.append_labeled(b"lu_table_keys", table_keys);
    transcript.append_labeled(b"lu_mult", multiplicities);
    transcript.append_labeled(b"lu_aux_len", &U64Word(logical_len as u64));
    ArkFr::from(transcript.challenge())
}

fn compression_coefficients(tuple_width: usize, challenge: ArkFr) -> Vec<ArkFr> {
    let mut power = ArkFr::one();
    (0..tuple_width)
        .map(|_| {
            let coefficient = power;
            power *= challenge;
            coefficient
        })
        .collect()
}

fn compress_tuple_columns(
    columns: &[&[ArkFr]],
    challenge: ArkFr,
    logical_len: usize,
) -> Vec<ArkFr> {
    let coefficients = compression_coefficients(columns.len(), challenge);
    (0..logical_len)
        .into_par_iter()
        .map(|index| {
            columns
                .iter()
                .zip(&coefficients)
                .map(|(column, coefficient)| column[index] * coefficient)
                .sum()
        })
        .collect()
}

fn compute_multiplicities(
    query_keys: &[ArkFr],
    table_keys: &[ArkFr],
) -> Result<Vec<ArkFr>, HiddenLogUpError> {
    let mut first_table_position = HashMap::with_capacity(table_keys.len());
    for (index, key) in table_keys.iter().copied().enumerate() {
        first_table_position.entry(key).or_insert(index);
    }
    let mut counts = vec![0_u64; table_keys.len()];
    for (query_index, key) in query_keys.iter().enumerate() {
        let Some(&table_index) = first_table_position.get(key) else {
            return Err(HiddenLogUpError::MissingQueryTuple { query_index });
        };
        counts[table_index] = counts[table_index]
            .checked_add(1)
            .ok_or(HiddenLogUpError::MultiplicityOverflow { table_index })?;
    }
    Ok(counts.into_iter().map(ArkFr::from).collect())
}

fn batch_inverse_denominators(
    query_keys: &[ArkFr],
    table_keys: &[ArkFr],
    beta: ArkFr,
) -> Result<(Vec<ArkFr>, Vec<ArkFr>), HiddenLogUpError> {
    let mut denominators = query_keys
        .iter()
        .chain(table_keys)
        .map(|key| beta + key)
        .collect::<Vec<_>>();
    for (index, denominator) in denominators.iter().enumerate() {
        if denominator.is_zero() {
            let (side, local_index) = if index < query_keys.len() {
                (HiddenLogUpSide::Query, index)
            } else {
                (HiddenLogUpSide::Table, index - query_keys.len())
            };
            return Err(HiddenLogUpError::SingularDenominator {
                side,
                index: local_index,
            });
        }
    }
    batch_inversion(&mut denominators);
    let table_inverses = denominators.split_off(query_keys.len());
    Ok((denominators, table_inverses))
}

fn inverse_expression(beta: ArkFr) -> RelationExpression {
    RelationExpression::table(1)
        .mul(RelationExpression::constant(beta).add(RelationExpression::table(0)))
        .sub(RelationExpression::constant(ArkFr::one()))
}

fn sum_balance_expression() -> RelationExpression {
    RelationExpression::table(0).sub(RelationExpression::table(1).mul(RelationExpression::table(2)))
}

fn commit_auxiliary(
    parameters: &DoryProverParameters,
    values: &[ArkFr],
) -> Result<CommittedDoryFieldVector, HiddenLogUpError> {
    commit_field_vector_zk(parameters, values)
        .map_err(|error| HiddenLogUpError::Dory(error.to_string()))
}

fn commit_auxiliary_matrix(
    parameters: &DoryProverParameters,
    grid: Option<&Section6DoryGrid>,
    columns: &[Vec<ArkFr>],
) -> Result<Vec<CommittedDoryFieldVector>, HiddenLogUpError> {
    let columns = columns.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let padded_cols = columns.len().next_power_of_two();
    match grid {
        Some(grid) => {
            commit_section6_field_matrix_columns_on_grid_zk(parameters, grid, &columns, padded_cols)
                .map_err(|error| HiddenLogUpError::Dory(error.to_string()))
        }
        None => commit_field_matrix_column_slices_zk(parameters, &columns, padded_cols)
            .map_err(|error| HiddenLogUpError::Dory(error.to_string())),
    }
}

fn commit_inverse_matrix(
    parameters: &DoryProverParameters,
    grid: Option<&Section6DoryGrid>,
    values: &[ArkFr],
    first_part: usize,
    part_count: usize,
    logical_len: usize,
) -> Result<Vec<CommittedDoryFieldVector>, HiddenLogUpError> {
    let columns = (0..part_count)
        .map(|part| inverse_part(values, first_part + part, logical_len))
        .collect::<Vec<_>>();
    let padded_cols = part_count.next_power_of_two();
    match grid {
        Some(grid) => {
            commit_section6_field_matrix_columns_on_grid_zk(parameters, grid, &columns, padded_cols)
                .map_err(|error| HiddenLogUpError::Dory(error.to_string()))
        }
        None => commit_field_matrix_column_slices_zk(parameters, &columns, padded_cols)
            .map_err(|error| HiddenLogUpError::Dory(error.to_string())),
    }
}

fn validate_part_matrices_fit_grid(
    logical_len: usize,
    query_part_count: usize,
    table_part_count: usize,
    grid: &Section6DoryGrid,
) -> Result<(), HiddenLogUpError> {
    let padded_parts = query_part_count
        .next_power_of_two()
        .max(table_part_count.next_power_of_two());
    let matrix_len = logical_len
        .checked_mul(padded_parts)
        .ok_or(HiddenLogUpError::ProofShape)?;
    if matrix_len > grid.logical_len() {
        return Err(HiddenLogUpError::ProofShape);
    }
    Ok(())
}

fn auxiliary_matrix_statements<'a>(
    commitment: &'a DoryCommitment,
    logical_len: usize,
    part_count: usize,
) -> Vec<HiddenRelationTableStatement<'a>> {
    (0..part_count)
        .map(|part| {
            HiddenRelationTableStatement::matrix_column(
                commitment,
                logical_len,
                part_count.next_power_of_two(),
                part,
            )
        })
        .collect()
}

fn auxiliary_statement<'a>(
    commitment: &'a DoryCommitment,
    logical_len: usize,
) -> HiddenRelationTableStatement<'a> {
    HiddenRelationTableStatement::vector(commitment, logical_len)
}

fn relation_label(instance_label: &[u8], phase: &[u8]) -> Vec<u8> {
    let mut label =
        Vec::with_capacity(LOGUP_RELATION_DOMAIN.len() + 16 + instance_label.len() + phase.len());
    label.extend_from_slice(LOGUP_RELATION_DOMAIN);
    label.extend_from_slice(&(instance_label.len() as u64).to_le_bytes());
    label.extend_from_slice(instance_label);
    label.extend_from_slice(&(phase.len() as u64).to_le_bytes());
    label.extend_from_slice(phase);
    label
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use crate::{
        jolt_dory::setup_dory_parameters,
        section6_final_opening::{
            resolve_section6_hidden_relation_prover_claim,
            resolve_section6_hidden_relation_verifier_claim, Section6DoryGrid,
        },
    };

    fn field_column(values: &[u64]) -> Vec<ArkFr> {
        values.iter().copied().map(ArkFr::from).collect()
    }

    fn commit_columns(
        parameters: &DoryProverParameters,
        columns: &[Vec<ArkFr>],
    ) -> Vec<CommittedDoryFieldVector> {
        columns
            .iter()
            .map(|column| commit_field_vector_zk(parameters, column).unwrap())
            .collect()
    }

    fn prover_columns<'a>(
        values: &'a [Vec<ArkFr>],
        commitments: &'a [CommittedDoryFieldVector],
    ) -> Vec<HiddenRelationTable<'a>> {
        values
            .iter()
            .zip(commitments)
            .map(|(values, committed)| HiddenRelationTable { values, committed })
            .collect()
    }

    fn verifier_columns(
        commitments: &[CommittedDoryFieldVector],
    ) -> Vec<HiddenRelationTableStatement<'_>> {
        commitments
            .iter()
            .map(HiddenRelationTableStatement::from)
            .collect()
    }

    fn prove_parts_with_temporary_public_tag<'a>(
        parameters: &DoryProverParameters,
        query: HiddenRelationTable<'a>,
        table: HiddenRelationTable<'a>,
        rng: &mut impl RngCore,
    ) -> (
        HiddenLogUpPartsReductionProof,
        HiddenLogUpProverOpeningState<'a>,
    ) {
        let tag = vec![ArkFr::from(9_u64); query.values.len()];
        let query_parts = vec![vec![
            MixedLogUpColumn::Public(PublicRelationTable::Dense(&tag)),
            MixedLogUpColumn::Hidden(query),
        ]];
        let table_parts = vec![vec![
            MixedLogUpColumn::Public(PublicRelationTable::Dense(&tag)),
            MixedLogUpColumn::Hidden(table),
        ]];
        let (proof, pending) = prove_hidden_logup_parts_deferred(
            b"temporary-public-tag",
            parameters,
            &query_parts,
            &table_parts,
            rng,
        )
        .unwrap();
        let pending = pending
            .rebind_parts_hidden_tables(vec![vec![query]], vec![vec![table]])
            .unwrap();
        (proof, pending)
    }

    fn verify_parts_with_temporary_public_tag<'a>(
        parameters: &DoryVerifierParameters,
        proof: &'a HiddenLogUpPartsReductionProof,
        query: HiddenRelationTableStatement<'a>,
        table: HiddenRelationTableStatement<'a>,
    ) -> HiddenLogUpVerifierOpeningState<'a> {
        let tag = vec![ArkFr::from(9_u64); query.logical_len];
        let query_parts = vec![vec![
            MixedLogUpColumnStatement::Public(PublicRelationTable::Dense(&tag)),
            MixedLogUpColumnStatement::Hidden(query),
        ]];
        let table_parts = vec![vec![
            MixedLogUpColumnStatement::Public(PublicRelationTable::Dense(&tag)),
            MixedLogUpColumnStatement::Hidden(table),
        ]];
        verify_hidden_logup_parts_deferred(
            b"temporary-public-tag",
            parameters,
            &query_parts,
            &table_parts,
            proof,
        )
        .unwrap()
        .rebind_parts_hidden_tables(proof, vec![vec![query]], vec![vec![table]])
        .unwrap()
    }

    #[test]
    fn valid_tuple_lookup_verifies_and_auxiliary_tamper_fails() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let table = vec![
            field_column(&[0, 0, 1, 1, 2, 2, 3, 3]),
            field_column(&[10, 11, 20, 21, 30, 31, 40, 41]),
        ];
        let permutation = [3, 0, 7, 2, 5, 1, 6, 4];
        let query = table
            .iter()
            .map(|column| permutation.iter().map(|&index| column[index]).collect())
            .collect::<Vec<Vec<ArkFr>>>();
        let query_commitments = commit_columns(&prover_parameters, &query);
        let table_commitments = commit_columns(&prover_parameters, &table);
        let query_prover = prover_columns(&query, &query_commitments);
        let table_prover = prover_columns(&table, &table_commitments);
        let mut rng = StdRng::seed_from_u64(11);
        let proof = prove_hidden_logup(
            b"valid-tuple-lookup",
            &prover_parameters,
            &query_prover,
            &table_prover,
            &mut rng,
        )
        .unwrap();
        let query_verifier = verifier_columns(&query_commitments);
        let table_verifier = verifier_columns(&table_commitments);
        verify_hidden_logup(
            b"valid-tuple-lookup",
            &verifier_parameters,
            &query_verifier,
            &table_verifier,
            &proof,
        )
        .unwrap();
        assert_eq!(
            proof
                .identity_reduction
                .relation
                .evaluation_commitments
                .len(),
            8
        );
        assert_eq!(proof.sum_reduction.relation.evaluation_commitments.len(), 3);
        assert_eq!(proof.relation_opening.groups.len(), 2);
        assert!(proof
            .relation_opening
            .groups
            .iter()
            .all(|group| group.opening.vector_dory_batch.is_some()));

        let mut swapped_openings = proof.clone();
        swapped_openings.relation_opening.groups.swap(0, 1);
        assert!(verify_hidden_logup(
            b"valid-tuple-lookup",
            &verifier_parameters,
            &query_verifier,
            &table_verifier,
            &swapped_openings,
        )
        .is_err());

        let wrong_values = vec![ArkFr::from(99_u64); 8];
        let wrong_commitment = commit_field_vector_zk(&prover_parameters, &wrong_values).unwrap();
        let mut tampered = proof.clone();
        tampered.auxiliary_commitments.query_inverses = wrong_commitment.commitment;
        assert!(verify_hidden_logup(
            b"valid-tuple-lookup",
            &verifier_parameters,
            &query_verifier,
            &table_verifier,
            &tampered,
        )
        .is_err());
    }

    #[test]
    fn duplicate_table_keys_use_one_aggregate_multiplicity() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let table = vec![
            field_column(&[0, 0, 1, 1, 2, 2, 3, 3]),
            field_column(&[7, 7, 9, 10, 11, 12, 13, 14]),
        ];
        let query = vec![
            field_column(&[0, 0, 0, 1, 1, 2, 3, 3]),
            field_column(&[7, 7, 7, 9, 10, 11, 13, 14]),
        ];
        let query_commitments = commit_columns(&prover_parameters, &query);
        let table_commitments = commit_columns(&prover_parameters, &table);
        let query_verifier = verifier_columns(&query_commitments);
        let table_verifier = verifier_columns(&table_commitments);
        let (_, compression_challenge) = source_transcript_and_challenge(
            b"duplicate-multiplicity",
            &query_verifier,
            &table_verifier,
            8,
        )
        .unwrap();
        let query_slices = query.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let table_slices = table.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let query_keys = compress_tuple_columns(&query_slices, compression_challenge, 8);
        let table_keys = compress_tuple_columns(&table_slices, compression_challenge, 8);
        let multiplicities = compute_multiplicities(&query_keys, &table_keys).unwrap();
        assert_eq!(multiplicities[0], ArkFr::from(3_u64));
        assert_eq!(multiplicities[1], ArkFr::zero());

        let query_prover = prover_columns(&query, &query_commitments);
        let table_prover = prover_columns(&table, &table_commitments);
        let mut rng = StdRng::seed_from_u64(23);
        let proof = prove_hidden_logup(
            b"duplicate-multiplicity",
            &prover_parameters,
            &query_prover,
            &table_prover,
            &mut rng,
        )
        .unwrap();
        verify_hidden_logup(
            b"duplicate-multiplicity",
            &verifier_parameters,
            &query_verifier,
            &table_verifier,
            &proof,
        )
        .unwrap();
    }

    #[test]
    fn row_tag_tuple_column_prevents_cross_row_lookup() {
        let (prover_parameters, _) = setup_dory_parameters(8);
        let table = vec![
            field_column(&[0, 1, 0, 1, 2, 2, 3, 3]),
            field_column(&[10, 20, 30, 40, 50, 60, 70, 80]),
        ];
        let query = vec![
            field_column(&[0, 1, 0, 1, 2, 2, 3, 3]),
            field_column(&[20, 20, 30, 40, 50, 60, 70, 80]),
        ];

        // Without a row-tag column, the value 20 is a valid table lookup.
        let value_only_query = [&query[1][..]];
        let value_only_table = [&table[1][..]];
        let value_query_keys = compress_tuple_columns(&value_only_query, ArkFr::from(17_u64), 8);
        let value_table_keys = compress_tuple_columns(&value_only_table, ArkFr::from(17_u64), 8);
        assert!(compute_multiplicities(&value_query_keys, &value_table_keys).is_ok());

        // With the row identifier as tuple column zero, (row 0, value 20) is absent.
        let query_commitments = commit_columns(&prover_parameters, &query);
        let table_commitments = commit_columns(&prover_parameters, &table);
        let query_prover = prover_columns(&query, &query_commitments);
        let table_prover = prover_columns(&table, &table_commitments);
        let mut rng = StdRng::seed_from_u64(31);
        assert!(matches!(
            prove_hidden_logup(
                b"row-tag-attack",
                &prover_parameters,
                &query_prover,
                &table_prover,
                &mut rng,
            ),
            Err(HiddenLogUpError::MissingQueryTuple { query_index: 0 })
        ));
    }

    #[test]
    fn hidden_exponent_queries_lookup_public_table() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let public_exponents = field_column(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let public_powers = field_column(&[1, 2, 4, 8, 16, 32, 64, 128]);
        let permutation = [6, 1, 7, 0, 4, 2, 5, 3];
        let query_exponents = permutation
            .iter()
            .map(|&index| public_exponents[index])
            .collect::<Vec<_>>();
        let query_powers = permutation
            .iter()
            .map(|&index| public_powers[index])
            .collect::<Vec<_>>();
        let query_values = vec![query_exponents, query_powers];
        let query_commitments = commit_columns(&prover_parameters, &query_values);
        let query = vec![
            MixedLogUpColumn::Hidden(HiddenRelationTable {
                values: &query_values[0],
                committed: &query_commitments[0],
            }),
            MixedLogUpColumn::Hidden(HiddenRelationTable {
                values: &query_values[1],
                committed: &query_commitments[1],
            }),
        ];
        let table = vec![
            MixedLogUpColumn::Public(PublicRelationTable::Dense(&public_exponents)),
            MixedLogUpColumn::Public(PublicRelationTable::Dense(&public_powers)),
        ];
        let mut rng = StdRng::seed_from_u64(41);
        let proof = prove_mixed_hidden_logup(
            b"public-exponent-table",
            &prover_parameters,
            &query,
            &table,
            &mut rng,
        )
        .unwrap();

        let query_statement = vec![
            MixedLogUpColumnStatement::Hidden(HiddenRelationTableStatement::from(
                &query_commitments[0],
            )),
            MixedLogUpColumnStatement::Hidden(HiddenRelationTableStatement::from(
                &query_commitments[1],
            )),
        ];
        let table_statement = vec![
            MixedLogUpColumnStatement::Public(PublicRelationTable::Dense(&public_exponents)),
            MixedLogUpColumnStatement::Public(PublicRelationTable::Dense(&public_powers)),
        ];
        verify_mixed_hidden_logup(
            b"public-exponent-table",
            &verifier_parameters,
            &query_statement,
            &table_statement,
            &proof,
        )
        .unwrap();
    }

    #[test]
    fn tampered_public_table_is_rejected() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let public_exponents = field_column(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let public_powers = field_column(&[1, 2, 4, 8, 16, 32, 64, 128]);
        let query_values = vec![public_exponents.clone(), public_powers.clone()];
        let query_commitments = commit_columns(&prover_parameters, &query_values);
        let query = vec![
            MixedLogUpColumn::Hidden(HiddenRelationTable {
                values: &query_values[0],
                committed: &query_commitments[0],
            }),
            MixedLogUpColumn::Hidden(HiddenRelationTable {
                values: &query_values[1],
                committed: &query_commitments[1],
            }),
        ];
        let table = vec![
            MixedLogUpColumn::Public(PublicRelationTable::Dense(&public_exponents)),
            MixedLogUpColumn::Public(PublicRelationTable::Dense(&public_powers)),
        ];
        let mut rng = StdRng::seed_from_u64(43);
        let proof = prove_mixed_hidden_logup(
            b"tampered-public-table",
            &prover_parameters,
            &query,
            &table,
            &mut rng,
        )
        .unwrap();

        let query_statement = vec![
            MixedLogUpColumnStatement::Hidden(HiddenRelationTableStatement::from(
                &query_commitments[0],
            )),
            MixedLogUpColumnStatement::Hidden(HiddenRelationTableStatement::from(
                &query_commitments[1],
            )),
        ];
        let mut tampered_powers = public_powers.clone();
        tampered_powers[3] += ArkFr::one();
        let tampered_table = vec![
            MixedLogUpColumnStatement::Public(PublicRelationTable::Dense(&public_exponents)),
            MixedLogUpColumnStatement::Public(PublicRelationTable::Dense(&tampered_powers)),
        ];
        assert!(verify_mixed_hidden_logup(
            b"tampered-public-table",
            &verifier_parameters,
            &query_statement,
            &tampered_table,
            &proof,
        )
        .is_err());
    }

    #[test]
    fn mixed_public_row_tag_prevents_cross_row_lookup() {
        let (prover_parameters, _) = setup_dory_parameters(8);
        let table_rows = field_column(&[0, 1, 0, 1, 2, 2, 3, 3]);
        let table_values = field_column(&[10, 20, 30, 40, 50, 60, 70, 80]);
        let query_rows = table_rows.clone();
        let query_values = field_column(&[20, 20, 30, 40, 50, 60, 70, 80]);

        let value_query_keys = compress_tuple_columns(&[&query_values], ArkFr::from(17_u64), 8);
        let value_table_keys = compress_tuple_columns(&[&table_values], ArkFr::from(17_u64), 8);
        assert!(compute_multiplicities(&value_query_keys, &value_table_keys).is_ok());

        let committed_query_values =
            commit_field_vector_zk(&prover_parameters, &query_values).unwrap();
        let query = vec![
            MixedLogUpColumn::Public(PublicRelationTable::Dense(&query_rows)),
            MixedLogUpColumn::Hidden(HiddenRelationTable {
                values: &query_values,
                committed: &committed_query_values,
            }),
        ];
        let table = vec![
            MixedLogUpColumn::Public(PublicRelationTable::Dense(&table_rows)),
            MixedLogUpColumn::Public(PublicRelationTable::Dense(&table_values)),
        ];
        let mut rng = StdRng::seed_from_u64(47);
        assert!(matches!(
            prove_mixed_hidden_logup(
                b"mixed-row-tag-attack",
                &prover_parameters,
                &query,
                &table,
                &mut rng,
            ),
            Err(HiddenLogUpError::MissingQueryTuple { query_index: 0 })
        ));
    }

    #[test]
    fn multipart_lookup_aggregates_table_multiplicities_without_flattening_sources() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(4);
        let table_values = vec![
            vec![field_column(&[0, 0, 0, 0]), field_column(&[10, 11, 12, 13])],
            vec![field_column(&[1, 1, 1, 1]), field_column(&[20, 21, 22, 23])],
            vec![field_column(&[2, 2, 2, 2]), field_column(&[30, 31, 32, 33])],
        ];
        let query_values = vec![
            vec![field_column(&[0, 1, 2, 0]), field_column(&[10, 21, 32, 13])],
            vec![field_column(&[2, 1, 0, 2]), field_column(&[30, 23, 11, 31])],
        ];
        let query_commitments = query_values
            .iter()
            .map(|part| commit_columns(&prover_parameters, part))
            .collect::<Vec<_>>();
        let query_parts = (0..query_values.len())
            .map(|part| {
                (0..query_values[part].len())
                    .map(|column| {
                        MixedLogUpColumn::Hidden(HiddenRelationTable {
                            values: &query_values[part][column],
                            committed: &query_commitments[part][column],
                        })
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let table_parts = table_values
            .iter()
            .map(|part| {
                part.iter()
                    .map(|values| MixedLogUpColumn::Public(PublicRelationTable::Dense(values)))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(53);
        let proof = prove_hidden_logup_parts(
            b"multipart-lookup",
            &prover_parameters,
            &query_parts,
            &table_parts,
            &mut rng,
        )
        .unwrap();

        let query_statements = query_commitments
            .iter()
            .map(|part| {
                part.iter()
                    .map(|committed| {
                        MixedLogUpColumnStatement::Hidden(HiddenRelationTableStatement::from(
                            committed,
                        ))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let table_statements = table_values
            .iter()
            .map(|part| {
                part.iter()
                    .map(|values| {
                        MixedLogUpColumnStatement::Public(PublicRelationTable::Dense(values))
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        verify_hidden_logup_parts(
            b"multipart-lookup",
            &verifier_parameters,
            &query_statements,
            &table_statements,
            &proof,
        )
        .unwrap();

        let (reduction, pending) = prove_hidden_logup_parts_deferred(
            b"multipart-deferred",
            &prover_parameters,
            &query_parts,
            &table_parts,
            &mut rng,
        )
        .unwrap();
        let HiddenLogUpPartsReductionProof {
            auxiliary_commitments: _,
            identity_reduction: _,
            sum_reduction: _,
        } = reduction.clone();
        let opening =
            prove_hidden_logup_openings_batch(&prover_parameters, &[&pending], &mut rng).unwrap();
        let verifier_pending = verify_hidden_logup_parts_deferred(
            b"multipart-deferred",
            &verifier_parameters,
            &query_statements,
            &table_statements,
            &reduction,
        )
        .unwrap();
        verify_hidden_logup_openings_batch(&verifier_parameters, &[&verifier_pending], &opening)
            .unwrap();
    }

    #[test]
    fn multipart_deferred_auxiliaries_use_one_shared_section6_grid() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(5);
        let grid = Section6DoryGrid::new(5).unwrap();
        let query_values = [field_column(&[1, 2, 3, 4])];
        let table_values = [field_column(&[4, 3, 2, 1]), field_column(&[5, 6, 7, 8])];
        let query_parts = query_values
            .iter()
            .map(|values| vec![MixedLogUpColumn::Public(PublicRelationTable::Dense(values))])
            .collect::<Vec<_>>();
        let table_parts = table_values
            .iter()
            .map(|values| vec![MixedLogUpColumn::Public(PublicRelationTable::Dense(values))])
            .collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(57);
        let (reduction, pending) = prove_hidden_logup_parts_deferred_on_grid(
            b"multipart-shared-grid",
            &prover_parameters,
            &grid,
            &query_parts,
            &table_parts,
            &mut rng,
        )
        .unwrap();

        let HiddenLogUpProverTables::Parts(tables) = &pending.tables else {
            panic!("multipart reduction returned single-part state");
        };
        for committed in tables
            .committed_query_keys
            .iter()
            .chain(&tables.committed_table_keys)
            .chain(&tables.committed_multiplicities)
            .chain(&tables.committed_query_inverses)
            .chain(&tables.committed_table_inverses)
        {
            assert_eq!(committed.grid_num_vars, Some(grid.num_vars()));
        }

        let query_statements = query_values
            .iter()
            .map(|values| {
                vec![MixedLogUpColumnStatement::Public(
                    PublicRelationTable::Dense(values),
                )]
            })
            .collect::<Vec<_>>();
        let table_statements = table_values
            .iter()
            .map(|values| {
                vec![MixedLogUpColumnStatement::Public(
                    PublicRelationTable::Dense(values),
                )]
            })
            .collect::<Vec<_>>();
        let verifier_pending = verify_hidden_logup_parts_deferred_on_grid(
            b"multipart-shared-grid",
            &verifier_parameters,
            &grid,
            &query_statements,
            &table_statements,
            &reduction,
        )
        .unwrap();
        let prover_openings = pending.relation_openings().unwrap();
        let verifier_openings = verifier_pending.relation_openings().unwrap();
        for (prover_opening, verifier_opening) in prover_openings.iter().zip(&verifier_openings) {
            for table in prover_opening.hidden_tables() {
                resolve_section6_hidden_relation_prover_claim(
                    *table,
                    prover_opening.row_point(),
                    &grid,
                )
                .unwrap();
            }
            for (table, evaluation_commitment) in verifier_opening
                .hidden_tables()
                .iter()
                .zip(verifier_opening.evaluation_commitments())
            {
                resolve_section6_hidden_relation_verifier_claim(
                    *table,
                    verifier_opening.row_point(),
                    *evaluation_commitment,
                    &grid,
                )
                .unwrap();
            }
        }
    }

    #[test]
    fn two_deferred_logups_share_one_phase_opening_and_bind_order() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let first_values = vec![field_column(&[0, 1, 2, 3, 4, 5, 6, 7])];
        let second_values = vec![field_column(&[7, 6, 5, 4, 3, 2, 1, 0])];
        let first_commitments = commit_columns(&prover_parameters, &first_values);
        let second_commitments = commit_columns(&prover_parameters, &second_values);
        let first_prover = prover_columns(&first_values, &first_commitments);
        let second_prover = prover_columns(&second_values, &second_commitments);

        let mut rng = StdRng::seed_from_u64(59);
        let (first_reduction, first_pending) = prove_hidden_logup_deferred(
            b"deferred-first",
            &prover_parameters,
            &first_prover,
            &first_prover,
            &mut rng,
        )
        .unwrap();
        let (second_reduction, second_pending) = prove_hidden_logup_deferred(
            b"deferred-second",
            &prover_parameters,
            &second_prover,
            &second_prover,
            &mut rng,
        )
        .unwrap();

        let HiddenLogUpReductionProof {
            auxiliary_commitments: _,
            identity_reduction: _,
            sum_reduction: _,
        } = first_reduction.clone();
        let opening = prove_hidden_logup_openings_batch(
            &prover_parameters,
            &[&first_pending, &second_pending],
            &mut rng,
        )
        .unwrap();

        let first_verifier = verifier_columns(&first_commitments);
        let second_verifier = verifier_columns(&second_commitments);
        let first_claims = verify_hidden_logup_deferred(
            b"deferred-first",
            &verifier_parameters,
            &first_verifier,
            &first_verifier,
            &first_reduction,
        )
        .unwrap();
        let second_claims = verify_hidden_logup_deferred(
            b"deferred-second",
            &verifier_parameters,
            &second_verifier,
            &second_verifier,
            &second_reduction,
        )
        .unwrap();
        verify_hidden_logup_openings_batch(
            &verifier_parameters,
            &[&first_claims, &second_claims],
            &opening,
        )
        .unwrap();

        assert!(verify_hidden_logup_openings_batch(
            &verifier_parameters,
            &[&second_claims, &first_claims],
            &opening,
        )
        .is_err());

        let mut tampered = opening.clone();
        tampered.groups.swap(0, 1);
        assert!(verify_hidden_logup_openings_batch(
            &verifier_parameters,
            &[&first_claims, &second_claims],
            &tampered,
        )
        .is_err());
    }

    #[test]
    fn deferred_parts_discard_temporary_public_columns_before_returning() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(4);
        let table_values = field_column(&[10, 11, 12, 13]);
        let query_values = field_column(&[13, 10, 12, 11]);
        let committed_table = commit_field_vector_zk(&prover_parameters, &table_values).unwrap();
        let committed_query = commit_field_vector_zk(&prover_parameters, &query_values).unwrap();
        let query = HiddenRelationTable {
            values: &query_values,
            committed: &committed_query,
        };
        let table = HiddenRelationTable {
            values: &table_values,
            committed: &committed_table,
        };
        let mut rng = StdRng::seed_from_u64(61);
        let (proof, pending) =
            prove_parts_with_temporary_public_tag(&prover_parameters, query, table, &mut rng);
        let opening =
            prove_hidden_logup_openings_batch(&prover_parameters, &[&pending], &mut rng).unwrap();

        let verifier_pending = verify_parts_with_temporary_public_tag(
            &verifier_parameters,
            &proof,
            HiddenRelationTableStatement::from(&committed_query),
            HiddenRelationTableStatement::from(&committed_table),
        );
        verify_hidden_logup_openings_batch(&verifier_parameters, &[&verifier_pending], &opening)
            .unwrap();
    }
}
