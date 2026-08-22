//! Zero-knowledge row-wise products over Dory-committed field columns.
//!
//! The protocol chains the generic hidden relation proof over prefix products.
//! The verifier receives only Dory commitments and the common logical length.

use std::fmt::{Display, Formatter};

use ark_bn254::Fr;
use ark_ff::{One, Zero};
use ark_std::rand::RngCore;
use jolt_dory::DoryCommitment;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    fiat_shamir::FiatShamir,
    hidden_relation::{
        prove_hidden_relation, verify_hidden_relation, HiddenMatrixColumn, HiddenRelationError,
        HiddenRelationProof, HiddenRelationTable, HiddenRelationTableStatement, RelationExpression,
    },
    jolt_dory::{
        commit_field_matrix_columns_zk, commitment_binding_bytes, CommittedDoryFieldVector,
        DoryBackendError, DoryProverParameters, DoryVerifierParameters,
    },
};

const ROW_PRODUCT_DOMAIN: &[u8] = b"parcomm-hidden-row-product-v1";

/// One prover-side row-product column and its existing hiding commitment.
#[derive(Clone, Copy)]
pub struct HiddenRowProductColumn<'a> {
    pub values: &'a [Fr],
    pub committed: &'a CommittedDoryFieldVector,
}

/// The verifier-side commitment metadata for one row-product column.
#[derive(Clone, Copy)]
pub struct HiddenRowProductColumnStatement<'a> {
    pub commitment: &'a DoryCommitment,
    pub logical_len: usize,
    pub matrix: Option<HiddenMatrixColumn>,
}

impl<'a> From<&'a CommittedDoryFieldVector> for HiddenRowProductColumnStatement<'a> {
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

impl<'a> From<HiddenRowProductColumnStatement<'a>> for HiddenRelationTableStatement<'a> {
    fn from(statement: HiddenRowProductColumnStatement<'a>) -> Self {
        Self {
            commitment: statement.commitment,
            logical_len: statement.logical_len,
            matrix: statement.matrix,
        }
    }
}

/// One auxiliary prefix matrix and the relation connecting the product chain.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenRowProductProof {
    /// One commitment to the column-major prefix matrix, when nonempty. Its
    /// final logical column is the claimed row-product vector.
    pub prefix_commitment: Option<DoryCommitment>,
    /// One batched element-wise relation for the complete product chain and
    /// the equality between the final prefix column and the output vector.
    pub relation: HiddenRelationProof,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HiddenRowProductColumnRole {
    Factor(usize),
    Scale,
    Output,
}

/// Failures detected while constructing or checking a row-product chain.
#[derive(Debug)]
pub enum HiddenRowProductError {
    EmptyFactors,
    InvalidLogicalLength(usize),
    ValuesLength {
        column: HiddenRowProductColumnRole,
        expected: usize,
        actual: usize,
    },
    CommitmentLength {
        column: HiddenRowProductColumnRole,
        expected: usize,
        actual: usize,
    },
    ProofShape {
        expected_prefix_matrix: bool,
        actual_prefix_matrix: bool,
    },
    Commitment(DoryBackendError),
    Relation {
        relation: usize,
        source: HiddenRelationError,
    },
}

impl Display for HiddenRowProductError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyFactors => formatter.write_str("row product requires at least one factor"),
            Self::InvalidLogicalLength(length) => write!(
                formatter,
                "row-product logical length {length} is not a power of two of at least two"
            ),
            Self::ValuesLength {
                column,
                expected,
                actual,
            } => write!(
                formatter,
                "{column:?} has {actual} values, expected {expected}"
            ),
            Self::CommitmentLength {
                column,
                expected,
                actual,
            } => write!(
                formatter,
                "{column:?} commitment has logical length {actual}, expected {expected}"
            ),
            Self::ProofShape {
                expected_prefix_matrix,
                actual_prefix_matrix,
            } => write!(
                formatter,
                "row-product prefix-matrix presence is {actual_prefix_matrix}, expected {expected_prefix_matrix}"
            ),
            Self::Commitment(error) => Display::fmt(error, formatter),
            Self::Relation { relation, source } => {
                write!(formatter, "row-product relation {relation} failed: {source}")
            }
        }
    }
}

impl std::error::Error for HiddenRowProductError {}

/// Prove a row-wise product of hidden factor columns.
///
/// With a row scale, the claimed output is
/// `scale[row] * product_h factors[h][row]`. Without a row scale, the first
/// factor is reused as the initial prefix. Consequently, the unscaled path
/// avoids committing a constant-one column and also avoids a trivial first
/// multiplication. The single-factor path uses a hidden equality proof so the
/// independently hiding factor and output commitments remain bound.
pub fn prove_hidden_row_product(
    label: &[u8],
    parameters: &DoryProverParameters,
    factors: &[HiddenRowProductColumn<'_>],
    row_scale: Option<HiddenRowProductColumn<'_>>,
    output: HiddenRowProductColumn<'_>,
    rng: &mut impl RngCore,
) -> Result<HiddenRowProductProof, HiddenRowProductError> {
    validate_prover_columns(factors, row_scale, output)?;
    let scaled = row_scale.is_some();

    if !scaled && factors.len() == 1 {
        let expression = equality_expression();
        let tables = [as_relation_table(factors[0]), as_relation_table(output)];
        let relation_label = relation_label(label, factors.len(), false, 0, RelationKind::Equality);
        let relation =
            prove_hidden_relation(&relation_label, parameters, &tables, &expression, rng).map_err(
                |source| HiddenRowProductError::Relation {
                    relation: 0,
                    source,
                },
            )?;
        return Ok(HiddenRowProductProof {
            prefix_commitment: None,
            relation,
        });
    }

    let (initial_values, initial_committed, first_factor) = match row_scale {
        Some(scale) => (scale.values, scale.committed, 0),
        None => (factors[0].values, factors[0].committed, 1),
    };
    let prefix_count = multiplication_count(factors.len(), scaled);
    let mut prefix_values: Vec<Vec<Fr>> = Vec::with_capacity(prefix_count);
    for step in 0..prefix_count {
        let current = if step == 0 {
            initial_values
        } else {
            &prefix_values[step - 1]
        };
        let factor = factors[first_factor + step];
        prefix_values.push(
            current
                .par_iter()
                .zip(factor.values.par_iter())
                .map(|(prefix, factor)| *prefix * factor)
                .collect::<Vec<_>>(),
        );
    }
    let committed_prefixes = if prefix_values.is_empty() {
        Vec::new()
    } else {
        commit_field_matrix_columns_zk(
            parameters,
            &prefix_values,
            prefix_values.len().next_power_of_two(),
        )
        .map_err(HiddenRowProductError::Commitment)?
    };
    let factor_statements = factors
        .iter()
        .map(|factor| HiddenRowProductColumnStatement::from(factor.committed))
        .collect::<Vec<_>>();
    let batch_challenge = row_product_batch_challenge(
        label,
        &factor_statements,
        row_scale.map(|scale| HiddenRowProductColumnStatement::from(scale.committed)),
        HiddenRowProductColumnStatement::from(output.committed),
        committed_prefixes.first().map(|column| &column.commitment),
        prefix_count,
    );
    let mut tables = Vec::with_capacity(prefix_count * 3 + 2);
    let mut expression = RelationExpression::constant(Fr::zero());
    let mut coefficient = Fr::one();
    for step in 0..prefix_count {
        let factor_index = first_factor + step;
        let current = if step == 0 {
            HiddenRelationTable {
                values: initial_values,
                committed: initial_committed,
            }
        } else {
            HiddenRelationTable {
                values: &prefix_values[step - 1],
                committed: &committed_prefixes[step - 1],
            }
        };
        let next = HiddenRelationTable {
            values: &prefix_values[step],
            committed: &committed_prefixes[step],
        };
        let table_index = tables.len();
        tables.push(current);
        tables.push(as_relation_table(factors[factor_index]));
        tables.push(next);
        let equation = RelationExpression::table(table_index)
            .mul(RelationExpression::table(table_index + 1))
            .sub(RelationExpression::table(table_index + 2));
        expression = expression.add(RelationExpression::constant(coefficient).mul(equation));
        coefficient *= batch_challenge;
    }
    let table_index = tables.len();
    tables.push(HiddenRelationTable {
        values: &prefix_values[prefix_count - 1],
        committed: &committed_prefixes[prefix_count - 1],
    });
    tables.push(as_relation_table(output));
    let output_equation =
        RelationExpression::table(table_index).sub(RelationExpression::table(table_index + 1));
    expression = expression.add(RelationExpression::constant(coefficient).mul(output_equation));
    let step_label = relation_label(
        label,
        factors.len(),
        scaled,
        first_factor,
        RelationKind::Multiplication,
    );
    let relation = prove_hidden_relation(&step_label, parameters, &tables, &expression, rng)
        .map_err(|source| HiddenRowProductError::Relation {
            relation: 0,
            source,
        })?;

    Ok(HiddenRowProductProof {
        prefix_commitment: committed_prefixes
            .first()
            .map(|column| column.commitment.clone()),
        relation,
    })
}

/// Verify a hidden row-wise product using commitment metadata only.
pub fn verify_hidden_row_product(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    factors: &[HiddenRowProductColumnStatement<'_>],
    row_scale: Option<HiddenRowProductColumnStatement<'_>>,
    output: HiddenRowProductColumnStatement<'_>,
    proof: &HiddenRowProductProof,
) -> Result<(), HiddenRowProductError> {
    let logical_len = validate_verifier_columns(factors, row_scale, output)?;
    let scaled = row_scale.is_some();
    let prefix_count = multiplication_count(factors.len(), scaled);
    let has_prefix_matrix = proof.prefix_commitment.is_some();
    if has_prefix_matrix != (prefix_count > 0) {
        return Err(HiddenRowProductError::ProofShape {
            expected_prefix_matrix: prefix_count > 0,
            actual_prefix_matrix: has_prefix_matrix,
        });
    }

    if !scaled && factors.len() == 1 {
        let tables = [factors[0].into(), output.into()];
        let step_label = relation_label(label, factors.len(), false, 0, RelationKind::Equality);
        return verify_hidden_relation(
            &step_label,
            parameters,
            &tables,
            &equality_expression(),
            &proof.relation,
        )
        .map_err(|source| HiddenRowProductError::Relation {
            relation: 0,
            source,
        });
    }

    let (initial, first_factor) = match row_scale {
        Some(scale) => (scale, 0),
        None => (factors[0], 1),
    };
    let prefix_columns = proof
        .prefix_commitment
        .as_ref()
        .map(|commitment| {
            let padded_cols = prefix_count.next_power_of_two();
            (0..prefix_count)
                .map(|column| HiddenRowProductColumnStatement {
                    commitment,
                    logical_len,
                    matrix: Some(HiddenMatrixColumn {
                        padded_cols,
                        column,
                    }),
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let batch_challenge = row_product_batch_challenge(
        label,
        factors,
        row_scale,
        output,
        proof.prefix_commitment.as_ref(),
        prefix_count,
    );
    let mut tables = Vec::with_capacity(prefix_count * 3 + 2);
    let mut expression = RelationExpression::constant(Fr::zero());
    let mut coefficient = Fr::one();
    for step in 0..prefix_count {
        let current = if step == 0 {
            initial
        } else {
            prefix_columns[step - 1]
        };
        let next = prefix_columns[step];
        let table_index = tables.len();
        tables.push(current.into());
        tables.push(factors[first_factor + step].into());
        tables.push(next.into());
        let equation = RelationExpression::table(table_index)
            .mul(RelationExpression::table(table_index + 1))
            .sub(RelationExpression::table(table_index + 2));
        expression = expression.add(RelationExpression::constant(coefficient).mul(equation));
        coefficient *= batch_challenge;
    }
    let table_index = tables.len();
    tables.push(prefix_columns[prefix_count - 1].into());
    tables.push(output.into());
    let output_equation =
        RelationExpression::table(table_index).sub(RelationExpression::table(table_index + 1));
    expression = expression.add(RelationExpression::constant(coefficient).mul(output_equation));
    let step_label = relation_label(
        label,
        factors.len(),
        scaled,
        first_factor,
        RelationKind::Multiplication,
    );
    verify_hidden_relation(
        &step_label,
        parameters,
        &tables,
        &expression,
        &proof.relation,
    )
    .map_err(|source| HiddenRowProductError::Relation {
        relation: 0,
        source,
    })
}

fn validate_prover_columns(
    factors: &[HiddenRowProductColumn<'_>],
    row_scale: Option<HiddenRowProductColumn<'_>>,
    output: HiddenRowProductColumn<'_>,
) -> Result<usize, HiddenRowProductError> {
    let factor_statements = factors
        .iter()
        .map(|factor| HiddenRowProductColumnStatement::from(factor.committed))
        .collect::<Vec<_>>();
    let scale_statement =
        row_scale.map(|scale| HiddenRowProductColumnStatement::from(scale.committed));
    let output_statement = HiddenRowProductColumnStatement::from(output.committed);
    let logical_len =
        validate_verifier_columns(&factor_statements, scale_statement, output_statement)?;

    for (index, factor) in factors.iter().enumerate() {
        validate_values(
            factor.values,
            logical_len,
            HiddenRowProductColumnRole::Factor(index),
        )?;
    }
    if let Some(scale) = row_scale {
        validate_values(scale.values, logical_len, HiddenRowProductColumnRole::Scale)?;
    }
    validate_values(
        output.values,
        logical_len,
        HiddenRowProductColumnRole::Output,
    )?;
    Ok(logical_len)
}

fn validate_verifier_columns(
    factors: &[HiddenRowProductColumnStatement<'_>],
    row_scale: Option<HiddenRowProductColumnStatement<'_>>,
    output: HiddenRowProductColumnStatement<'_>,
) -> Result<usize, HiddenRowProductError> {
    if factors.is_empty() {
        return Err(HiddenRowProductError::EmptyFactors);
    }
    let logical_len = output.logical_len;
    if logical_len < 2 || !logical_len.is_power_of_two() {
        return Err(HiddenRowProductError::InvalidLogicalLength(logical_len));
    }
    for (index, factor) in factors.iter().enumerate() {
        validate_commitment_length(
            factor.logical_len,
            logical_len,
            HiddenRowProductColumnRole::Factor(index),
        )?;
    }
    if let Some(scale) = row_scale {
        validate_commitment_length(
            scale.logical_len,
            logical_len,
            HiddenRowProductColumnRole::Scale,
        )?;
    }
    validate_commitment_length(
        output.logical_len,
        logical_len,
        HiddenRowProductColumnRole::Output,
    )?;
    Ok(logical_len)
}

fn validate_values(
    values: &[Fr],
    expected: usize,
    column: HiddenRowProductColumnRole,
) -> Result<(), HiddenRowProductError> {
    if values.len() != expected {
        return Err(HiddenRowProductError::ValuesLength {
            column,
            expected,
            actual: values.len(),
        });
    }
    Ok(())
}

fn validate_commitment_length(
    actual: usize,
    expected: usize,
    column: HiddenRowProductColumnRole,
) -> Result<(), HiddenRowProductError> {
    if actual != expected {
        return Err(HiddenRowProductError::CommitmentLength {
            column,
            expected,
            actual,
        });
    }
    Ok(())
}

fn as_relation_table(column: HiddenRowProductColumn<'_>) -> HiddenRelationTable<'_> {
    HiddenRelationTable {
        values: column.values,
        committed: column.committed,
    }
}

fn equality_expression() -> RelationExpression {
    RelationExpression::table(0).sub(RelationExpression::table(1))
}

fn row_product_batch_challenge(
    label: &[u8],
    factors: &[HiddenRowProductColumnStatement<'_>],
    row_scale: Option<HiddenRowProductColumnStatement<'_>>,
    output: HiddenRowProductColumnStatement<'_>,
    prefix_commitment: Option<&DoryCommitment>,
    prefix_count: usize,
) -> Fr {
    let mut transcript = FiatShamir::<Fr>::new();
    transcript.push_bytes(ROW_PRODUCT_DOMAIN);
    transcript.push_bytes(label);
    transcript.push(&(factors.len() as u64));
    transcript.push(&(u64::from(row_scale.is_some())));
    transcript.push(&(prefix_count as u64));
    for factor in factors {
        absorb_column_statement(&mut transcript, *factor);
    }
    if let Some(scale) = row_scale {
        absorb_column_statement(&mut transcript, scale);
    }
    absorb_column_statement(&mut transcript, output);
    transcript.push(&(u64::from(prefix_commitment.is_some())));
    if let Some(commitment) = prefix_commitment {
        transcript.push_bytes(&commitment_binding_bytes(commitment));
    }
    transcript.gen_challenge()
}

fn absorb_column_statement(
    transcript: &mut FiatShamir<Fr>,
    statement: HiddenRowProductColumnStatement<'_>,
) {
    transcript.push_bytes(&commitment_binding_bytes(statement.commitment));
    transcript.push(&(statement.logical_len as u64));
    let (padded_cols, column) = statement
        .matrix
        .map(|matrix| (matrix.padded_cols, matrix.column))
        .unwrap_or((0, 0));
    transcript.push(&(padded_cols as u64));
    transcript.push(&(column as u64));
}

fn multiplication_count(factor_count: usize, scaled: bool) -> usize {
    if scaled {
        factor_count
    } else {
        factor_count.saturating_sub(1)
    }
}

#[derive(Clone, Copy)]
enum RelationKind {
    Equality,
    Multiplication,
}

fn relation_label(
    label: &[u8],
    factor_count: usize,
    scaled: bool,
    factor_index: usize,
    kind: RelationKind,
) -> Vec<u8> {
    let mut encoded = Vec::with_capacity(ROW_PRODUCT_DOMAIN.len() + label.len() + 34);
    encoded.extend_from_slice(ROW_PRODUCT_DOMAIN);
    encoded.extend_from_slice(&(label.len() as u64).to_le_bytes());
    encoded.extend_from_slice(label);
    encoded.extend_from_slice(&(factor_count as u64).to_le_bytes());
    encoded.push(u8::from(scaled));
    encoded.extend_from_slice(&(factor_index as u64).to_le_bytes());
    encoded.push(match kind {
        RelationKind::Equality => 0,
        RelationKind::Multiplication => 1,
    });
    encoded
}

#[cfg(test)]
mod tests {
    use super::*;
    use ark_ff::{One, Zero};
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use crate::jolt_dory::{commit_field_vector_zk, setup_dory_parameters};

    fn factors(length: usize) -> Vec<Vec<Fr>> {
        (0..4)
            .map(|column| {
                (0..length)
                    .map(|row| Fr::from((column * 7 + row % 5 + 2) as u64))
                    .collect()
            })
            .collect()
    }

    fn row_product(factors: &[Vec<Fr>], scale: Option<&[Fr]>) -> Vec<Fr> {
        let length = factors[0].len();
        (0..length)
            .map(|row| {
                let initial = scale.map_or_else(Fr::one, |values| values[row]);
                factors
                    .iter()
                    .fold(initial, |product, factor| product * factor[row])
            })
            .collect()
    }

    #[test]
    fn unscaled_product_verifies_and_prefix_matrix_tampering_fails() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let factor_values = factors(16);
        let output_values = row_product(&factor_values, None);
        let committed_factors = factor_values
            .iter()
            .map(|values| commit_field_vector_zk(&prover_parameters, values).unwrap())
            .collect::<Vec<_>>();
        let committed_output = commit_field_vector_zk(&prover_parameters, &output_values).unwrap();
        let prover_factors = factor_values
            .iter()
            .zip(&committed_factors)
            .map(|(values, committed)| HiddenRowProductColumn { values, committed })
            .collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(71);
        let proof = prove_hidden_row_product(
            b"unscaled-product",
            &prover_parameters,
            &prover_factors,
            None,
            HiddenRowProductColumn {
                values: &output_values,
                committed: &committed_output,
            },
            &mut rng,
        )
        .unwrap();
        assert!(proof.prefix_commitment.is_some());
        assert_eq!(proof.relation.matrix_dory_batches.len(), 1);

        let verifier_factors = committed_factors
            .iter()
            .map(HiddenRowProductColumnStatement::from)
            .collect::<Vec<_>>();
        let output_statement = HiddenRowProductColumnStatement::from(&committed_output);
        verify_hidden_row_product(
            b"unscaled-product",
            &verifier_parameters,
            &verifier_factors,
            None,
            output_statement,
            &proof,
        )
        .unwrap();

        let wrong_values = vec![Fr::zero(); 16];
        let wrong_commitment = commit_field_vector_zk(&prover_parameters, &wrong_values).unwrap();
        let mut tampered = proof.clone();
        tampered.prefix_commitment = Some(wrong_commitment.commitment);
        assert!(verify_hidden_row_product(
            b"unscaled-product",
            &verifier_parameters,
            &verifier_factors,
            None,
            output_statement,
            &tampered,
        )
        .is_err());
    }

    #[test]
    fn factor_and_output_statement_tampering_fail() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let factor_values = factors(16);
        let output_values = row_product(&factor_values, None);
        let committed_factors = factor_values
            .iter()
            .map(|values| commit_field_vector_zk(&prover_parameters, values).unwrap())
            .collect::<Vec<_>>();
        let committed_output = commit_field_vector_zk(&prover_parameters, &output_values).unwrap();
        let prover_factors = factor_values
            .iter()
            .zip(&committed_factors)
            .map(|(values, committed)| HiddenRowProductColumn { values, committed })
            .collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(73);
        let proof = prove_hidden_row_product(
            b"statement-tamper",
            &prover_parameters,
            &prover_factors,
            None,
            HiddenRowProductColumn {
                values: &output_values,
                committed: &committed_output,
            },
            &mut rng,
        )
        .unwrap();

        let wrong_values = vec![Fr::one(); 16];
        let wrong_commitment = commit_field_vector_zk(&prover_parameters, &wrong_values).unwrap();
        let mut verifier_factors = committed_factors
            .iter()
            .map(HiddenRowProductColumnStatement::from)
            .collect::<Vec<_>>();
        verifier_factors[2] = HiddenRowProductColumnStatement::from(&wrong_commitment);
        assert!(verify_hidden_row_product(
            b"statement-tamper",
            &verifier_parameters,
            &verifier_factors,
            None,
            HiddenRowProductColumnStatement::from(&committed_output),
            &proof,
        )
        .is_err());

        let verifier_factors = committed_factors
            .iter()
            .map(HiddenRowProductColumnStatement::from)
            .collect::<Vec<_>>();
        assert!(verify_hidden_row_product(
            b"statement-tamper",
            &verifier_parameters,
            &verifier_factors,
            None,
            HiddenRowProductColumnStatement::from(&wrong_commitment),
            &proof,
        )
        .is_err());
    }

    #[test]
    fn scaled_and_single_factor_products_verify() {
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
        let factor_values = factors(16);
        let scale_values = (0..16)
            .map(|row| Fr::from((row + 3) as u64))
            .collect::<Vec<_>>();
        let output_values = row_product(&factor_values, Some(&scale_values));
        let committed_factors = factor_values
            .iter()
            .map(|values| commit_field_vector_zk(&prover_parameters, values).unwrap())
            .collect::<Vec<_>>();
        let committed_scale = commit_field_vector_zk(&prover_parameters, &scale_values).unwrap();
        let committed_output = commit_field_vector_zk(&prover_parameters, &output_values).unwrap();
        let prover_factors = factor_values
            .iter()
            .zip(&committed_factors)
            .map(|(values, committed)| HiddenRowProductColumn { values, committed })
            .collect::<Vec<_>>();
        let mut rng = StdRng::seed_from_u64(79);
        let proof = prove_hidden_row_product(
            b"scaled-product",
            &prover_parameters,
            &prover_factors,
            Some(HiddenRowProductColumn {
                values: &scale_values,
                committed: &committed_scale,
            }),
            HiddenRowProductColumn {
                values: &output_values,
                committed: &committed_output,
            },
            &mut rng,
        )
        .unwrap();
        assert!(proof.prefix_commitment.is_some());
        assert_eq!(proof.relation.matrix_dory_batches.len(), 1);
        let verifier_factors = committed_factors
            .iter()
            .map(HiddenRowProductColumnStatement::from)
            .collect::<Vec<_>>();
        verify_hidden_row_product(
            b"scaled-product",
            &verifier_parameters,
            &verifier_factors,
            Some(HiddenRowProductColumnStatement::from(&committed_scale)),
            HiddenRowProductColumnStatement::from(&committed_output),
            &proof,
        )
        .unwrap();

        let single_output = factor_values[0].clone();
        let single_committed_output =
            commit_field_vector_zk(&prover_parameters, &single_output).unwrap();
        let single_proof = prove_hidden_row_product(
            b"single-product",
            &prover_parameters,
            &prover_factors[..1],
            None,
            HiddenRowProductColumn {
                values: &single_output,
                committed: &single_committed_output,
            },
            &mut rng,
        )
        .unwrap();
        assert!(single_proof.prefix_commitment.is_none());
        verify_hidden_row_product(
            b"single-product",
            &verifier_parameters,
            &verifier_factors[..1],
            None,
            HiddenRowProductColumnStatement::from(&single_committed_output),
            &single_proof,
        )
        .unwrap();
    }
}
