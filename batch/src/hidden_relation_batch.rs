//! Fiat--Shamir batching for hidden-only algebraic relations.
//!
//! Each item retains its own local table numbering. The batch compiler shifts
//! those indices into one ordered operand list, samples one commitment-bound
//! coefficient, and proves the resulting random linear combination with one
//! [`HiddenRelationProof`].

use std::collections::{HashMap, HashSet};

use ark_ff::{One, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::rand::RngCore;
use jolt_field::Fr;
use jolt_transcript::domain::{Label, LabelWithCount, U64Word};
use jolt_transcript::{Blake2bTranscript, Transcript};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    hidden_relation::{
        prove_hidden_relation_deferred_with_public, prove_hidden_relation_with_public,
        prove_hidden_sum_relation_deferred_with_public, prove_hidden_sum_relation_with_public,
        verify_hidden_relation_deferred_with_public, verify_hidden_relation_with_public,
        verify_hidden_sum_relation_deferred_with_public, verify_hidden_sum_relation_with_public,
        HiddenRelationError, HiddenRelationProof, HiddenRelationProverOpening,
        HiddenRelationReductionProof, HiddenRelationTable, HiddenRelationTableStatement,
        HiddenRelationVerifierOpening, PublicRelationTable, RelationExpression,
    },
    jolt_dory::{commitment_binding_bytes, DoryProverParameters, DoryVerifierParameters},
};

/// One identity relation supplied by the prover.
pub struct HiddenRelationBatchProverItem<'a> {
    pub tables: &'a [HiddenRelationTable<'a>],
    pub public_tables: &'a [PublicRelationTable<'a>],
    pub expression: &'a RelationExpression,
}

/// One identity relation statement supplied to the verifier.
pub struct HiddenRelationBatchVerifierItem<'a> {
    pub tables: &'a [HiddenRelationTableStatement<'a>],
    pub public_tables: &'a [PublicRelationTable<'a>],
    pub expression: &'a RelationExpression,
}

/// One random-linear-combination proof for an ordered relation batch.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenRelationBatchProof {
    pub relation: HiddenRelationProof,
}

/// One deferred random-linear-combination reduction for an ordered batch.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenRelationBatchReductionProof {
    pub relation: HiddenRelationReductionProof,
}

#[derive(Serialize)]
struct BatchBindingItem {
    hidden_tables: Vec<BatchBindingTable>,
    public_tables: Vec<BatchPublicBindingTable>,
    expression: RelationExpression,
}

#[derive(Debug, Eq, Hash, PartialEq, Serialize)]
struct BatchBindingTable {
    commitment: Vec<u8>,
    logical_len: usize,
    matrix: Option<(usize, usize, usize)>,
}

#[derive(Serialize)]
struct BatchPublicBindingTable {
    digest: [u8; 32],
    logical_len: usize,
}

#[derive(Clone, Copy, Serialize)]
enum BatchCheck {
    Identity,
    Sum,
}

struct DeduplicatedHiddenLayout {
    unique_positions: Vec<(usize, usize)>,
    local_indices: Vec<Vec<usize>>,
}

pub(crate) fn deduplicate_prover_hidden_tables<'a>(
    tables: impl IntoIterator<Item = HiddenRelationTable<'a>>,
) -> Vec<HiddenRelationTable<'a>> {
    let mut seen = HashSet::new();
    tables
        .into_iter()
        .filter(|table| {
            seen.insert(batch_binding_table(&HiddenRelationTableStatement::from(
                table.committed,
            )))
        })
        .collect()
}

pub(crate) fn deduplicate_verifier_hidden_tables<'a>(
    tables: impl IntoIterator<Item = HiddenRelationTableStatement<'a>>,
) -> Vec<HiddenRelationTableStatement<'a>> {
    let mut seen = HashSet::new();
    tables
        .into_iter()
        .filter(|table| seen.insert(batch_binding_table(table)))
        .collect()
}

/// Prove several hidden polynomial identities with one hidden-relation proof.
pub fn prove_hidden_relations_batch(
    label: &[u8],
    parameters: &DoryProverParameters,
    items: &[HiddenRelationBatchProverItem<'_>],
    rng: &mut impl RngCore,
) -> Result<HiddenRelationBatchProof, HiddenRelationError> {
    let bindings = prover_batch_bindings(items)?;
    validate_batch(&bindings)?;
    let challenge = batch_challenge(label, &bindings, BatchCheck::Identity)?;
    let (hidden_tables, public_tables, expression) =
        prover_batch_parts(items, &bindings, challenge)?;
    let relation = prove_hidden_relation_with_public(
        label,
        parameters,
        &hidden_tables,
        &public_tables,
        &expression,
        rng,
    )?;
    Ok(HiddenRelationBatchProof { relation })
}

/// Verify several hidden polynomial identities from commitments only.
pub fn verify_hidden_relations_batch(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    items: &[HiddenRelationBatchVerifierItem<'_>],
    proof: &HiddenRelationBatchProof,
) -> Result<(), HiddenRelationError> {
    let bindings = verifier_batch_bindings(items)?;
    validate_batch(&bindings)?;
    let challenge = batch_challenge(label, &bindings, BatchCheck::Identity)?;
    let (hidden_tables, public_tables, expression) =
        verifier_batch_parts(items, &bindings, challenge)?;
    verify_hidden_relation_with_public(
        label,
        parameters,
        &hidden_tables,
        &public_tables,
        &expression,
        &proof.relation,
    )
}

/// Reduce several identities while deferring their single Dory opening.
pub fn prove_hidden_relations_batch_deferred<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    items: &[HiddenRelationBatchProverItem<'a>],
    rng: &mut impl RngCore,
) -> Result<
    (
        HiddenRelationBatchReductionProof,
        HiddenRelationProverOpening<'a>,
    ),
    HiddenRelationError,
> {
    let bindings = prover_batch_bindings(items)?;
    validate_batch(&bindings)?;
    let challenge = batch_challenge(label, &bindings, BatchCheck::Identity)?;
    let (hidden_tables, public_tables, expression) =
        prover_batch_parts(items, &bindings, challenge)?;
    let (relation, opening) = prove_hidden_relation_deferred_with_public(
        label,
        parameters,
        &hidden_tables,
        &public_tables,
        &expression,
        rng,
    )?;
    Ok((HiddenRelationBatchReductionProof { relation }, opening))
}

/// Verify an identity batch reduction and return its commitment-only claims.
pub fn verify_hidden_relations_batch_deferred<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    items: &[HiddenRelationBatchVerifierItem<'a>],
    proof: &HiddenRelationBatchReductionProof,
) -> Result<HiddenRelationVerifierOpening<'a>, HiddenRelationError> {
    let bindings = verifier_batch_bindings(items)?;
    validate_batch(&bindings)?;
    let challenge = batch_challenge(label, &bindings, BatchCheck::Identity)?;
    let (hidden_tables, public_tables, expression) =
        verifier_batch_parts(items, &bindings, challenge)?;
    verify_hidden_relation_deferred_with_public(
        label,
        parameters,
        &hidden_tables,
        &public_tables,
        &expression,
        &proof.relation,
    )
}

/// Prove several hidden hypercube-sum relations with one relation proof.
pub fn prove_hidden_sum_relations_batch(
    label: &[u8],
    parameters: &DoryProverParameters,
    items: &[HiddenRelationBatchProverItem<'_>],
    rng: &mut impl RngCore,
) -> Result<HiddenRelationBatchProof, HiddenRelationError> {
    let bindings = prover_batch_bindings(items)?;
    validate_batch(&bindings)?;
    let challenge = batch_challenge(label, &bindings, BatchCheck::Sum)?;
    let (hidden_tables, public_tables, expression) =
        prover_batch_parts(items, &bindings, challenge)?;
    let relation = prove_hidden_sum_relation_with_public(
        label,
        parameters,
        &hidden_tables,
        &public_tables,
        &expression,
        rng,
    )?;
    Ok(HiddenRelationBatchProof { relation })
}

/// Verify several hidden hypercube-sum relations from commitments only.
pub fn verify_hidden_sum_relations_batch(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    items: &[HiddenRelationBatchVerifierItem<'_>],
    proof: &HiddenRelationBatchProof,
) -> Result<(), HiddenRelationError> {
    let bindings = verifier_batch_bindings(items)?;
    validate_batch(&bindings)?;
    let challenge = batch_challenge(label, &bindings, BatchCheck::Sum)?;
    let (hidden_tables, public_tables, expression) =
        verifier_batch_parts(items, &bindings, challenge)?;
    verify_hidden_sum_relation_with_public(
        label,
        parameters,
        &hidden_tables,
        &public_tables,
        &expression,
        &proof.relation,
    )
}

/// Reduce several sum relations while deferring their single Dory opening.
pub fn prove_hidden_sum_relations_batch_deferred<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    items: &[HiddenRelationBatchProverItem<'a>],
    rng: &mut impl RngCore,
) -> Result<
    (
        HiddenRelationBatchReductionProof,
        HiddenRelationProverOpening<'a>,
    ),
    HiddenRelationError,
> {
    let bindings = prover_batch_bindings(items)?;
    validate_batch(&bindings)?;
    let challenge = batch_challenge(label, &bindings, BatchCheck::Sum)?;
    let (hidden_tables, public_tables, expression) =
        prover_batch_parts(items, &bindings, challenge)?;
    let (relation, opening) = prove_hidden_sum_relation_deferred_with_public(
        label,
        parameters,
        &hidden_tables,
        &public_tables,
        &expression,
        rng,
    )?;
    Ok((HiddenRelationBatchReductionProof { relation }, opening))
}

/// Verify a sum batch reduction and return its commitment-only claims.
pub fn verify_hidden_sum_relations_batch_deferred<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    items: &[HiddenRelationBatchVerifierItem<'a>],
    proof: &HiddenRelationBatchReductionProof,
) -> Result<HiddenRelationVerifierOpening<'a>, HiddenRelationError> {
    let bindings = verifier_batch_bindings(items)?;
    validate_batch(&bindings)?;
    let challenge = batch_challenge(label, &bindings, BatchCheck::Sum)?;
    let (hidden_tables, public_tables, expression) =
        verifier_batch_parts(items, &bindings, challenge)?;
    verify_hidden_sum_relation_deferred_with_public(
        label,
        parameters,
        &hidden_tables,
        &public_tables,
        &expression,
        &proof.relation,
    )
}

fn prover_batch_bindings(
    items: &[HiddenRelationBatchProverItem<'_>],
) -> Result<Vec<BatchBindingItem>, HiddenRelationError> {
    items
        .iter()
        .map(|item| {
            let statements = item
                .tables
                .iter()
                .map(|table| HiddenRelationTableStatement::from(table.committed))
                .collect::<Vec<_>>();
            batch_binding_item(&statements, item.public_tables, item.expression)
        })
        .collect()
}

fn verifier_batch_bindings(
    items: &[HiddenRelationBatchVerifierItem<'_>],
) -> Result<Vec<BatchBindingItem>, HiddenRelationError> {
    items
        .iter()
        .map(|item| batch_binding_item(item.tables, item.public_tables, item.expression))
        .collect()
}

fn prover_batch_parts<'a>(
    items: &[HiddenRelationBatchProverItem<'a>],
    bindings: &[BatchBindingItem],
    challenge: Fr,
) -> Result<
    (
        Vec<HiddenRelationTable<'a>>,
        Vec<PublicRelationTable<'a>>,
        RelationExpression,
    ),
    HiddenRelationError,
> {
    let layout = deduplicated_hidden_layout(bindings)?;
    let expression = combine_expressions(
        bindings,
        challenge,
        &layout.local_indices,
        layout.unique_positions.len(),
    )?;
    let hidden_tables = layout
        .unique_positions
        .iter()
        .map(|&(item, table)| items[item].tables[table])
        .collect::<Vec<_>>();
    let public_tables = items
        .iter()
        .flat_map(|item| item.public_tables.iter().copied())
        .collect::<Vec<_>>();
    Ok((hidden_tables, public_tables, expression))
}

fn verifier_batch_parts<'a>(
    items: &[HiddenRelationBatchVerifierItem<'a>],
    bindings: &[BatchBindingItem],
    challenge: Fr,
) -> Result<
    (
        Vec<HiddenRelationTableStatement<'a>>,
        Vec<PublicRelationTable<'a>>,
        RelationExpression,
    ),
    HiddenRelationError,
> {
    let layout = deduplicated_hidden_layout(bindings)?;
    let expression = combine_expressions(
        bindings,
        challenge,
        &layout.local_indices,
        layout.unique_positions.len(),
    )?;
    let hidden_tables = layout
        .unique_positions
        .iter()
        .map(|&(item, table)| items[item].tables[table])
        .collect::<Vec<_>>();
    let public_tables = items
        .iter()
        .flat_map(|item| item.public_tables.iter().copied())
        .collect::<Vec<_>>();
    Ok((hidden_tables, public_tables, expression))
}

fn deduplicated_hidden_layout(
    items: &[BatchBindingItem],
) -> Result<DeduplicatedHiddenLayout, HiddenRelationError> {
    let mut unique = HashMap::<&BatchBindingTable, usize>::new();
    let mut unique_positions = Vec::new();
    let mut local_indices = Vec::with_capacity(items.len());
    for (item_index, item) in items.iter().enumerate() {
        let mut item_indices = Vec::with_capacity(item.hidden_tables.len());
        for (table_index, table) in item.hidden_tables.iter().enumerate() {
            let unique_index = match unique.get(table) {
                Some(&index) => index,
                None => {
                    let index = unique_positions.len();
                    unique.insert(table, index);
                    unique_positions.push((item_index, table_index));
                    index
                }
            };
            item_indices.push(unique_index);
        }
        local_indices.push(item_indices);
    }
    Ok(DeduplicatedHiddenLayout {
        unique_positions,
        local_indices,
    })
}

fn batch_binding_item(
    hidden_tables: &[HiddenRelationTableStatement<'_>],
    public_tables: &[PublicRelationTable<'_>],
    expression: &RelationExpression,
) -> Result<BatchBindingItem, HiddenRelationError> {
    let table_count = hidden_tables
        .len()
        .checked_add(public_tables.len())
        .ok_or(HiddenRelationError::ProofShape)?;
    validate_expression(expression, table_count)?;
    Ok(BatchBindingItem {
        hidden_tables: hidden_tables.iter().map(batch_binding_table).collect(),
        public_tables: public_tables
            .iter()
            .map(|table| {
                Ok(BatchPublicBindingTable {
                    digest: table.transcript_digest()?,
                    logical_len: table.logical_len(),
                })
            })
            .collect::<Result<Vec<_>, HiddenRelationError>>()?,
        expression: expression.clone(),
    })
}

fn batch_binding_table(table: &HiddenRelationTableStatement<'_>) -> BatchBindingTable {
    BatchBindingTable {
        commitment: commitment_binding_bytes(table.commitment),
        logical_len: table.logical_len,
        matrix: table
            .matrix
            .map(|matrix| (matrix.logical_cols, matrix.padded_cols, matrix.column)),
    }
}

fn validate_batch(items: &[BatchBindingItem]) -> Result<usize, HiddenRelationError> {
    let first = items
        .first()
        .and_then(|item| {
            item.hidden_tables
                .first()
                .map(|table| table.logical_len)
                .or_else(|| item.public_tables.first().map(|table| table.logical_len))
        })
        .ok_or(HiddenRelationError::EmptyTables)?;
    let logical_len = first;
    if logical_len < 2 || !logical_len.is_power_of_two() {
        return Err(HiddenRelationError::InvalidLogicalLength(logical_len));
    }
    let mut table_index = 0;
    for item in items {
        if item.hidden_tables.is_empty() && item.public_tables.is_empty() {
            return Err(HiddenRelationError::EmptyTables);
        }
        for table in &item.hidden_tables {
            if table.logical_len != logical_len {
                return Err(HiddenRelationError::TableLength {
                    table: table_index,
                    expected: logical_len,
                    actual: table.logical_len,
                });
            }
            table_index += 1;
        }
    }
    for item in items {
        for table in &item.public_tables {
            if table.logical_len != logical_len {
                return Err(HiddenRelationError::TableLength {
                    table: table_index,
                    expected: logical_len,
                    actual: table.logical_len,
                });
            }
            table_index += 1;
        }
    }
    Ok(logical_len)
}

fn batch_challenge(
    label: &[u8],
    items: &[BatchBindingItem],
    check: BatchCheck,
) -> Result<Fr, HiddenRelationError> {
    let encoded = bincode::serialize(items)
        .map_err(|error| HiddenRelationError::Encoding(error.to_string()))?;
    let mut transcript = Blake2bTranscript::<Fr>::new(b"parcomm-hidden-relation-batch-v1");
    transcript.append(&Label(b"hidden_relation_batch_statement"));
    transcript.append(&LabelWithCount(b"batch_label", label.len() as u64));
    transcript.append_bytes(label);
    transcript.append_labeled(b"batch_item_count", &U64Word(items.len() as u64));
    transcript.append_labeled(
        b"batch_check",
        &U64Word(u64::from(matches!(check, BatchCheck::Sum))),
    );
    transcript.append(&LabelWithCount(b"batch_statement", encoded.len() as u64));
    transcript.append_bytes(&encoded);
    let mut challenge = transcript.challenge();
    while challenge.is_zero() {
        transcript.append_labeled(b"batch_retry", &U64Word(1));
        challenge = transcript.challenge();
    }
    Ok(challenge)
}

fn combine_expressions(
    items: &[BatchBindingItem],
    challenge: Fr,
    local_hidden_indices: &[Vec<usize>],
    total_hidden_tables: usize,
) -> Result<RelationExpression, HiddenRelationError> {
    if local_hidden_indices.len() != items.len() {
        return Err(HiddenRelationError::ProofShape);
    }
    let mut combined: Option<RelationExpression> = None;
    let mut coefficient = Fr::one();
    let mut public_offset = 0_usize;
    for (item, hidden_indices) in items.iter().zip(local_hidden_indices) {
        if hidden_indices.len() != item.hidden_tables.len() {
            return Err(HiddenRelationError::ProofShape);
        }
        let shifted = remap_expression(
            &item.expression,
            item.hidden_tables.len(),
            hidden_indices,
            public_offset,
            total_hidden_tables,
        )?;
        let weighted = RelationExpression::Constant(coefficient).mul(shifted);
        combined = Some(match combined {
            Some(current) => current.add(weighted),
            None => weighted,
        });
        coefficient *= challenge;
        public_offset = public_offset
            .checked_add(item.public_tables.len())
            .ok_or(HiddenRelationError::ProofShape)?;
    }
    combined.ok_or(HiddenRelationError::EmptyTables)
}

fn remap_expression(
    expression: &RelationExpression,
    local_hidden_count: usize,
    local_hidden_indices: &[usize],
    public_offset: usize,
    total_hidden_count: usize,
) -> Result<RelationExpression, HiddenRelationError> {
    Ok(match expression {
        RelationExpression::Constant(value) => RelationExpression::Constant(*value),
        RelationExpression::Table(index) if *index < local_hidden_count => {
            RelationExpression::Table(
                *local_hidden_indices
                    .get(*index)
                    .ok_or(HiddenRelationError::ProofShape)?,
            )
        }
        RelationExpression::Table(index) => {
            let local_public_index = index - local_hidden_count;
            RelationExpression::Table(
                total_hidden_count
                    .checked_add(public_offset)
                    .and_then(|index| index.checked_add(local_public_index))
                    .ok_or(HiddenRelationError::ProofShape)?,
            )
        }
        RelationExpression::Add(left, right) => RelationExpression::Add(
            Box::new(remap_expression(
                left,
                local_hidden_count,
                local_hidden_indices,
                public_offset,
                total_hidden_count,
            )?),
            Box::new(remap_expression(
                right,
                local_hidden_count,
                local_hidden_indices,
                public_offset,
                total_hidden_count,
            )?),
        ),
        RelationExpression::Sub(left, right) => RelationExpression::Sub(
            Box::new(remap_expression(
                left,
                local_hidden_count,
                local_hidden_indices,
                public_offset,
                total_hidden_count,
            )?),
            Box::new(remap_expression(
                right,
                local_hidden_count,
                local_hidden_indices,
                public_offset,
                total_hidden_count,
            )?),
        ),
        RelationExpression::Mul(left, right) => RelationExpression::Mul(
            Box::new(remap_expression(
                left,
                local_hidden_count,
                local_hidden_indices,
                public_offset,
                total_hidden_count,
            )?),
            Box::new(remap_expression(
                right,
                local_hidden_count,
                local_hidden_indices,
                public_offset,
                total_hidden_count,
            )?),
        ),
    })
}

fn public_table_digest(values: &[ark_bn254::Fr]) -> Result<[u8; 32], HiddenRelationError> {
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

fn validate_expression(
    expression: &RelationExpression,
    table_count: usize,
) -> Result<(), HiddenRelationError> {
    match expression {
        RelationExpression::Constant(_) => Ok(()),
        RelationExpression::Table(index) if *index < table_count => Ok(()),
        RelationExpression::Table(index) => Err(HiddenRelationError::InvalidTableIndex {
            index: *index,
            table_count,
        }),
        RelationExpression::Add(left, right)
        | RelationExpression::Sub(left, right)
        | RelationExpression::Mul(left, right) => {
            validate_expression(left, table_count)?;
            validate_expression(right, table_count)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        hidden_relation::{
            prove_hidden_relation_opening, verify_hidden_relation_opening, HiddenRelationTable,
            HiddenRelationTableStatement, RelationExpression,
        },
        jolt_dory::{commit_field_vector_zk, setup_dory_parameters},
    };
    use ark_bn254::Fr;
    use ark_ff::One;
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn two_hidden_identities_share_one_serializable_relation_proof() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(4);
        let first_left = (1..=8).map(Fr::from).collect::<Vec<_>>();
        let first_right = (9..=16).map(Fr::from).collect::<Vec<_>>();
        let first_product = first_left
            .iter()
            .zip(&first_right)
            .map(|(left, right)| *left * right)
            .collect::<Vec<_>>();
        let second_left = (17..=24).map(Fr::from).collect::<Vec<_>>();
        let second_right = second_left.clone();
        let committed_first_left = commit_field_vector_zk(&prover_parameters, &first_left).unwrap();
        let committed_first_right =
            commit_field_vector_zk(&prover_parameters, &first_right).unwrap();
        let committed_first_product =
            commit_field_vector_zk(&prover_parameters, &first_product).unwrap();
        let committed_second_left =
            commit_field_vector_zk(&prover_parameters, &second_left).unwrap();
        let committed_second_right =
            commit_field_vector_zk(&prover_parameters, &second_right).unwrap();
        let first_tables = [
            HiddenRelationTable {
                values: &first_left,
                committed: &committed_first_left,
            },
            HiddenRelationTable {
                values: &first_right,
                committed: &committed_first_right,
            },
            HiddenRelationTable {
                values: &first_product,
                committed: &committed_first_product,
            },
        ];
        let second_tables = [
            HiddenRelationTable {
                values: &second_left,
                committed: &committed_second_left,
            },
            HiddenRelationTable {
                values: &second_right,
                committed: &committed_second_right,
            },
        ];
        let first_expression = RelationExpression::table(0)
            .mul(RelationExpression::table(1))
            .sub(RelationExpression::table(2));
        let second_expression = RelationExpression::table(0).sub(RelationExpression::table(1));
        let prover_items = [
            HiddenRelationBatchProverItem {
                tables: &first_tables,
                public_tables: &[],
                expression: &first_expression,
            },
            HiddenRelationBatchProverItem {
                tables: &second_tables,
                public_tables: &[],
                expression: &second_expression,
            },
        ];
        let mut rng = StdRng::seed_from_u64(701);

        let proof = prove_hidden_relations_batch(
            b"two-identities",
            &prover_parameters,
            &prover_items,
            &mut rng,
        )
        .unwrap();
        let encoded = bincode::serialize(&proof).unwrap();
        let decoded: HiddenRelationBatchProof = bincode::deserialize(&encoded).unwrap();
        assert_eq!(decoded, proof);
        assert_eq!(
            bincode::serialize(&proof).unwrap(),
            bincode::serialize(&proof.relation).unwrap()
        );

        let first_statements = [
            HiddenRelationTableStatement::from(&committed_first_left),
            HiddenRelationTableStatement::from(&committed_first_right),
            HiddenRelationTableStatement::from(&committed_first_product),
        ];
        let second_statements = [
            HiddenRelationTableStatement::from(&committed_second_left),
            HiddenRelationTableStatement::from(&committed_second_right),
        ];
        let verifier_items = [
            HiddenRelationBatchVerifierItem {
                tables: &first_statements,
                public_tables: &[],
                expression: &first_expression,
            },
            HiddenRelationBatchVerifierItem {
                tables: &second_statements,
                public_tables: &[],
                expression: &second_expression,
            },
        ];
        verify_hidden_relations_batch(
            b"two-identities",
            &verifier_parameters,
            &verifier_items,
            &decoded,
        )
        .unwrap();

        let wrong = vec![Fr::one(); 8];
        let wrong_commitment = commit_field_vector_zk(&prover_parameters, &wrong).unwrap();
        let tampered_first_statements = [
            HiddenRelationTableStatement::from(&committed_first_left),
            HiddenRelationTableStatement::from(&committed_first_right),
            HiddenRelationTableStatement::from(&wrong_commitment),
        ];
        let tampered_commitment_items = [
            HiddenRelationBatchVerifierItem {
                tables: &tampered_first_statements,
                public_tables: &[],
                expression: &first_expression,
            },
            HiddenRelationBatchVerifierItem {
                tables: &second_statements,
                public_tables: &[],
                expression: &second_expression,
            },
        ];
        assert!(verify_hidden_relations_batch(
            b"two-identities",
            &verifier_parameters,
            &tampered_commitment_items,
            &decoded,
        )
        .is_err());

        let reversed_items = [
            HiddenRelationBatchVerifierItem {
                tables: &second_statements,
                public_tables: &[],
                expression: &second_expression,
            },
            HiddenRelationBatchVerifierItem {
                tables: &first_statements,
                public_tables: &[],
                expression: &first_expression,
            },
        ];
        assert!(verify_hidden_relations_batch(
            b"two-identities",
            &verifier_parameters,
            &reversed_items,
            &decoded,
        )
        .is_err());

        let tampered_expression = RelationExpression::table(0).add(RelationExpression::table(1));
        let tampered_expression_items = [
            HiddenRelationBatchVerifierItem {
                tables: &first_statements,
                public_tables: &[],
                expression: &first_expression,
            },
            HiddenRelationBatchVerifierItem {
                tables: &second_statements,
                public_tables: &[],
                expression: &tampered_expression,
            },
        ];
        assert!(verify_hidden_relations_batch(
            b"two-identities",
            &verifier_parameters,
            &tampered_expression_items,
            &decoded,
        )
        .is_err());
    }

    #[test]
    fn two_hidden_sum_relations_share_one_relation_proof() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(4);
        let first_left = (1..=8).map(Fr::from).collect::<Vec<_>>();
        let first_right = (1..=8).rev().map(Fr::from).collect::<Vec<_>>();
        let second_left = [1_u64, 1, 1, 1, 0, 0, 0, 0]
            .into_iter()
            .map(Fr::from)
            .collect::<Vec<_>>();
        let second_right = [0_u64, 0, 0, 0, 1, 1, 1, 1]
            .into_iter()
            .map(Fr::from)
            .collect::<Vec<_>>();
        let committed_first_left = commit_field_vector_zk(&prover_parameters, &first_left).unwrap();
        let committed_first_right =
            commit_field_vector_zk(&prover_parameters, &first_right).unwrap();
        let committed_second_left =
            commit_field_vector_zk(&prover_parameters, &second_left).unwrap();
        let committed_second_right =
            commit_field_vector_zk(&prover_parameters, &second_right).unwrap();
        let first_tables = [
            HiddenRelationTable {
                values: &first_left,
                committed: &committed_first_left,
            },
            HiddenRelationTable {
                values: &first_right,
                committed: &committed_first_right,
            },
        ];
        let second_tables = [
            HiddenRelationTable {
                values: &second_left,
                committed: &committed_second_left,
            },
            HiddenRelationTable {
                values: &second_right,
                committed: &committed_second_right,
            },
        ];
        let expression = RelationExpression::table(0).sub(RelationExpression::table(1));
        let prover_items = [
            HiddenRelationBatchProverItem {
                tables: &first_tables,
                public_tables: &[],
                expression: &expression,
            },
            HiddenRelationBatchProverItem {
                tables: &second_tables,
                public_tables: &[],
                expression: &expression,
            },
        ];
        let mut rng = StdRng::seed_from_u64(709);

        let proof = prove_hidden_sum_relations_batch(
            b"two-sums",
            &prover_parameters,
            &prover_items,
            &mut rng,
        )
        .unwrap();
        let first_statements = [
            HiddenRelationTableStatement::from(&committed_first_left),
            HiddenRelationTableStatement::from(&committed_first_right),
        ];
        let second_statements = [
            HiddenRelationTableStatement::from(&committed_second_left),
            HiddenRelationTableStatement::from(&committed_second_right),
        ];
        let verifier_items = [
            HiddenRelationBatchVerifierItem {
                tables: &first_statements,
                public_tables: &[],
                expression: &expression,
            },
            HiddenRelationBatchVerifierItem {
                tables: &second_statements,
                public_tables: &[],
                expression: &expression,
            },
        ];

        verify_hidden_sum_relations_batch(
            b"two-sums",
            &verifier_parameters,
            &verifier_items,
            &proof,
        )
        .unwrap();
        assert!(verify_hidden_relations_batch(
            b"two-sums",
            &verifier_parameters,
            &verifier_items,
            &proof,
        )
        .is_err());
    }

    #[test]
    fn deferred_identity_batch_opens_after_reduction_verification() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(4);
        let first = (1..=8).map(Fr::from).collect::<Vec<_>>();
        let second = first.clone();
        let third = (11..=18).map(Fr::from).collect::<Vec<_>>();
        let fourth = third.clone();
        let committed_first = commit_field_vector_zk(&prover_parameters, &first).unwrap();
        let committed_second = commit_field_vector_zk(&prover_parameters, &second).unwrap();
        let committed_third = commit_field_vector_zk(&prover_parameters, &third).unwrap();
        let committed_fourth = commit_field_vector_zk(&prover_parameters, &fourth).unwrap();
        let first_tables = [
            HiddenRelationTable {
                values: &first,
                committed: &committed_first,
            },
            HiddenRelationTable {
                values: &second,
                committed: &committed_second,
            },
        ];
        let second_tables = [
            HiddenRelationTable {
                values: &third,
                committed: &committed_third,
            },
            HiddenRelationTable {
                values: &fourth,
                committed: &committed_fourth,
            },
        ];
        let expression = RelationExpression::table(0).sub(RelationExpression::table(1));
        let prover_items = [
            HiddenRelationBatchProverItem {
                tables: &first_tables,
                public_tables: &[],
                expression: &expression,
            },
            HiddenRelationBatchProverItem {
                tables: &second_tables,
                public_tables: &[],
                expression: &expression,
            },
        ];
        let mut rng = StdRng::seed_from_u64(719);

        let (reduction, prover_opening) = prove_hidden_relations_batch_deferred(
            b"deferred-identities",
            &prover_parameters,
            &prover_items,
            &mut rng,
        )
        .unwrap();
        assert_eq!(
            bincode::serialize(&reduction).unwrap(),
            bincode::serialize(&reduction.relation).unwrap()
        );
        let opening =
            prove_hidden_relation_opening(&prover_parameters, prover_opening, &mut rng).unwrap();
        let first_statements = [
            HiddenRelationTableStatement::from(&committed_first),
            HiddenRelationTableStatement::from(&committed_second),
        ];
        let second_statements = [
            HiddenRelationTableStatement::from(&committed_third),
            HiddenRelationTableStatement::from(&committed_fourth),
        ];
        let verifier_items = [
            HiddenRelationBatchVerifierItem {
                tables: &first_statements,
                public_tables: &[],
                expression: &expression,
            },
            HiddenRelationBatchVerifierItem {
                tables: &second_statements,
                public_tables: &[],
                expression: &expression,
            },
        ];

        let verifier_opening = verify_hidden_relations_batch_deferred(
            b"deferred-identities",
            &verifier_parameters,
            &verifier_items,
            &reduction,
        )
        .unwrap();
        verify_hidden_relation_opening(&verifier_parameters, verifier_opening, &opening).unwrap();
    }

    #[test]
    fn deferred_batch_reuses_identical_committed_operands() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(4);
        let values = (1..=8).map(Fr::from).collect::<Vec<_>>();
        let committed = commit_field_vector_zk(&prover_parameters, &values).unwrap();
        let tables = [
            HiddenRelationTable {
                values: &values,
                committed: &committed,
            },
            HiddenRelationTable {
                values: &values,
                committed: &committed,
            },
        ];
        let expression = RelationExpression::table(0).sub(RelationExpression::table(1));
        let prover_items = [
            HiddenRelationBatchProverItem {
                tables: &tables,
                public_tables: &[],
                expression: &expression,
            },
            HiddenRelationBatchProverItem {
                tables: &tables,
                public_tables: &[],
                expression: &expression,
            },
        ];
        let mut rng = StdRng::seed_from_u64(0x4445_4455_504c_4943);
        let (reduction, prover_opening) = prove_hidden_relations_batch_deferred(
            b"deduplicated-operands",
            &prover_parameters,
            &prover_items,
            &mut rng,
        )
        .unwrap();

        assert_eq!(reduction.relation.evaluation_commitments.len(), 1);

        let opening =
            prove_hidden_relation_opening(&prover_parameters, prover_opening, &mut rng).unwrap();
        let statements = [
            HiddenRelationTableStatement::from(&committed),
            HiddenRelationTableStatement::from(&committed),
        ];
        let verifier_items = [
            HiddenRelationBatchVerifierItem {
                tables: &statements,
                public_tables: &[],
                expression: &expression,
            },
            HiddenRelationBatchVerifierItem {
                tables: &statements,
                public_tables: &[],
                expression: &expression,
            },
        ];
        let verifier_opening = verify_hidden_relations_batch_deferred(
            b"deduplicated-operands",
            &verifier_parameters,
            &verifier_items,
            &reduction,
        )
        .unwrap();
        verify_hidden_relation_opening(&verifier_parameters, verifier_opening, &opening).unwrap();
    }

    #[test]
    fn deferred_mixed_batch_binds_each_public_table() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(4);
        let first_left = (2..=9).map(Fr::from).collect::<Vec<_>>();
        let first_public = (3..=10).map(Fr::from).collect::<Vec<_>>();
        let first_product = first_left
            .iter()
            .zip(&first_public)
            .map(|(left, public)| *left * public)
            .collect::<Vec<_>>();
        let second_left = (11..=18).map(Fr::from).collect::<Vec<_>>();
        let second_public = (19..=26).map(Fr::from).collect::<Vec<_>>();
        let second_product = second_left
            .iter()
            .zip(&second_public)
            .map(|(left, public)| *left * public)
            .collect::<Vec<_>>();
        let committed_first_left = commit_field_vector_zk(&prover_parameters, &first_left).unwrap();
        let committed_first_product =
            commit_field_vector_zk(&prover_parameters, &first_product).unwrap();
        let committed_second_left =
            commit_field_vector_zk(&prover_parameters, &second_left).unwrap();
        let committed_second_product =
            commit_field_vector_zk(&prover_parameters, &second_product).unwrap();
        let first_tables = [
            HiddenRelationTable {
                values: &first_left,
                committed: &committed_first_left,
            },
            HiddenRelationTable {
                values: &first_product,
                committed: &committed_first_product,
            },
        ];
        let second_tables = [
            HiddenRelationTable {
                values: &second_left,
                committed: &committed_second_left,
            },
            HiddenRelationTable {
                values: &second_product,
                committed: &committed_second_product,
            },
        ];
        let first_public_tables = [PublicRelationTable::Dense(&first_public)];
        let second_public_tables = [PublicRelationTable::Dense(&second_public)];
        let expression = RelationExpression::table(0)
            .mul(RelationExpression::table(2))
            .sub(RelationExpression::table(1));
        let prover_items = [
            HiddenRelationBatchProverItem {
                tables: &first_tables,
                public_tables: &first_public_tables,
                expression: &expression,
            },
            HiddenRelationBatchProverItem {
                tables: &second_tables,
                public_tables: &second_public_tables,
                expression: &expression,
            },
        ];
        let mut rng = StdRng::seed_from_u64(727);
        let (reduction, prover_opening) = prove_hidden_relations_batch_deferred(
            b"mixed-public-identities",
            &prover_parameters,
            &prover_items,
            &mut rng,
        )
        .unwrap();
        let opening =
            prove_hidden_relation_opening(&prover_parameters, prover_opening, &mut rng).unwrap();
        let first_statements = [
            HiddenRelationTableStatement::from(&committed_first_left),
            HiddenRelationTableStatement::from(&committed_first_product),
        ];
        let second_statements = [
            HiddenRelationTableStatement::from(&committed_second_left),
            HiddenRelationTableStatement::from(&committed_second_product),
        ];
        let verifier_items = [
            HiddenRelationBatchVerifierItem {
                tables: &first_statements,
                public_tables: &first_public_tables,
                expression: &expression,
            },
            HiddenRelationBatchVerifierItem {
                tables: &second_statements,
                public_tables: &second_public_tables,
                expression: &expression,
            },
        ];
        let verifier_opening = verify_hidden_relations_batch_deferred(
            b"mixed-public-identities",
            &verifier_parameters,
            &verifier_items,
            &reduction,
        )
        .unwrap();
        verify_hidden_relation_opening(&verifier_parameters, verifier_opening, &opening).unwrap();

        let mut tampered_public = second_public;
        tampered_public[3] += Fr::one();
        let tampered_public_tables = [PublicRelationTable::Dense(&tampered_public)];
        let tampered_items = [
            HiddenRelationBatchVerifierItem {
                tables: &first_statements,
                public_tables: &first_public_tables,
                expression: &expression,
            },
            HiddenRelationBatchVerifierItem {
                tables: &second_statements,
                public_tables: &tampered_public_tables,
                expression: &expression,
            },
        ];
        assert!(verify_hidden_relations_batch_deferred(
            b"mixed-public-identities",
            &verifier_parameters,
            &tampered_items,
            &reduction,
        )
        .is_err());
    }
}
