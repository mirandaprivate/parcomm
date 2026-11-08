//! Short-field primitives shared by the recursive QAFold Decide phase.

use super::*;
use crate::qafold_short::{ShortQaMatrixClaim, ShortQaRandomFoldableCode};

/// One multilinear evaluation claim on a mixed setup/carry matrix.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortEvalClaim<E: ShortFieldElement> {
    /// Claimed evaluation value.
    pub value: E,
    /// Evaluation point for the row dimension.
    pub row_point: Vec<E>,
    /// Explicit weights for the real matrix columns.
    pub col_weight: Vec<E>,
}

/// One surviving multilinear evaluation claim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortEvalWithPoint<E: ShortFieldElement> {
    /// Claimed evaluation value.
    pub value: E,
    /// Column-major multilinear evaluation point.
    pub point: Vec<E>,
}

/// Result of batching many evaluations into one matrix evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortMultiEvalReduction<E: ShortFieldElement> {
    /// Whether every internal reduction check accepted.
    pub ok: bool,
    /// The surviving evaluation claim on the mixed matrix.
    pub reduced: ShortEvalWithPoint<E>,
}

/// Prover-only output retaining the mixed matrix after its column folds.
pub(super) struct ShortMultiEvalProverResult<E: ShortFieldElement> {
    /// Transcript-compatible scalar reduction shared with the verifier.
    pub(super) reduction: ShortMultiEvalReduction<E>,
    /// Column-folded matrix, before the multilinear row folds.
    pub(super) folded_vector: Vec<E>,
}

/// One MLE claim on a base-field RFC diagonal, evaluated over the extension.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ShortQaRfcTClaim<E: ShortFieldElement> {
    /// One-based RFC layer containing the claimed diagonal.
    pub layer: usize,
    /// Evaluation claim on that diagonal after scalar extension.
    pub claim: ShortEvalWithPoint<E>,
}

struct MixedColumnFoldState<'a, B: ShortBaseField> {
    rows: usize,
    logical_cols: usize,
    setup_columns: &'a [&'a [B]],
    carry_columns: &'a [&'a [B::Extension]],
    column_terms: Vec<Vec<(usize, B::Extension)>>,
}

impl<'a, B: ShortBaseField> MixedColumnFoldState<'a, B> {
    fn new(
        setup_columns: &'a [&'a [B]],
        carry_columns: &'a [&'a [B::Extension]],
        rows: usize,
        padded_cols: usize,
    ) -> Result<Self, Error> {
        let real_cols = setup_columns
            .len()
            .checked_add(carry_columns.len())
            .ok_or(Error::EncodingError)?;
        if real_cols == 0
            || real_cols > padded_cols
            || setup_columns.iter().any(|column| column.len() > rows)
            || carry_columns.iter().any(|column| column.len() > rows)
        {
            return Err(Error::InvalidNumberOfVariables);
        }
        let column_terms = (0..real_cols)
            .map(|column| vec![(column, B::Extension::one())])
            .collect();
        Ok(Self {
            rows,
            logical_cols: padded_cols,
            setup_columns,
            carry_columns,
            column_terms,
        })
    }

    fn evaluation_at_claim(
        &self,
        column: usize,
        claim: usize,
        source_evaluations: &[Vec<B::Extension>],
    ) -> B::Extension {
        self.column_terms
            .get(column)
            .map(|terms| {
                terms
                    .iter()
                    .fold(B::Extension::zero(), |sum, (source, coefficient)| {
                        sum + source_evaluations[claim][*source] * *coefficient
                    })
            })
            .unwrap_or_else(B::Extension::zero)
    }

    fn fold_columns(&mut self, challenge: B::Extension) {
        let half = self.logical_cols / 2;
        let upper = if self.column_terms.len() > half {
            self.column_terms.split_off(half)
        } else {
            Vec::new()
        };
        self.column_terms.truncate(half);
        for (index, upper) in upper.into_iter().enumerate() {
            if index == self.column_terms.len() {
                self.column_terms.push(Vec::new());
            }
            let lower = &mut self.column_terms[index];
            lower.extend(
                upper
                    .into_iter()
                    .map(|(source, coefficient)| (source, coefficient * challenge)),
            );
        }
        self.logical_cols = half;
    }

    fn into_values(self) -> Vec<B::Extension> {
        let Self {
            rows,
            logical_cols,
            setup_columns,
            carry_columns,
            column_terms,
        } = self;
        debug_assert_eq!(logical_cols, 1);
        debug_assert_eq!(column_terms.len(), 1);
        let terms = column_terms.into_iter().next().unwrap_or_default();
        #[cfg(feature = "parallel")]
        if rows >= ROW_PARALLEL_THRESHOLD {
            return (0..rows)
                .into_par_iter()
                .map(|row| {
                    terms
                        .iter()
                        .fold(B::Extension::zero(), |sum, (source, coefficient)| {
                            sum + mixed_source_value::<B>(
                                setup_columns,
                                carry_columns,
                                *source,
                                row,
                            ) * *coefficient
                        })
                })
                .collect();
        }
        (0..rows)
            .map(|row| {
                terms
                    .iter()
                    .fold(B::Extension::zero(), |sum, (source, coefficient)| {
                        sum + mixed_source_value::<B>(setup_columns, carry_columns, *source, row)
                            * *coefficient
                    })
            })
            .collect()
    }
}

fn mixed_source_value<B: ShortBaseField>(
    setup_columns: &[&[B]],
    carry_columns: &[&[B::Extension]],
    source: usize,
    row: usize,
) -> B::Extension {
    if source < setup_columns.len() {
        setup_columns[source]
            .get(row)
            .copied()
            .map(B::Extension::from_base)
            .unwrap_or_else(B::Extension::zero)
    } else {
        carry_columns[source - setup_columns.len()]
            .get(row)
            .copied()
            .unwrap_or_else(B::Extension::zero)
    }
}

#[cfg(feature = "parallel")]
const ROW_PARALLEL_THRESHOLD: usize = 1 << 12;
#[cfg(feature = "parallel")]
const COLUMN_PARALLEL_THRESHOLD: usize = 1 << 6;

struct MultiEvalColumnWeightTerm<E: ShortFieldElement> {
    scale: E,
    col_weights: Vec<E>,
}

struct MultiEvalColumnWeightState<E: ShortFieldElement> {
    logical_cols: usize,
    terms: Vec<MultiEvalColumnWeightTerm<E>>,
}

impl<E: ShortFieldElement> MultiEvalColumnWeightState<E> {
    fn new(claims: &[ShortEvalClaim<E>], powers: &[E], logical_cols: usize) -> Self {
        let terms = claims
            .iter()
            .zip(powers)
            .map(|(claim, power)| MultiEvalColumnWeightTerm {
                scale: *power,
                col_weights: claim.col_weight.clone(),
            })
            .collect();
        Self {
            logical_cols,
            terms,
        }
    }

    fn value(&self, claim: usize, column: usize) -> E {
        self.terms[claim].scale
            * self.terms[claim]
                .col_weights
                .get(column)
                .copied()
                .unwrap_or_else(E::zero)
    }

    fn fold_columns(&mut self, challenge: E) {
        for term in &mut self.terms {
            let half = self.logical_cols / 2;
            let upper = if term.col_weights.len() > half {
                term.col_weights.split_off(half)
            } else {
                Vec::new()
            };
            term.col_weights.truncate(half);
            for (index, upper) in upper.into_iter().enumerate() {
                if index == term.col_weights.len() {
                    term.col_weights.push(E::zero());
                }
                term.col_weights[index] += upper * challenge;
            }
        }
        self.logical_cols /= 2;
    }

    fn into_row_values(self, row_weights: Vec<Vec<E>>, rows: usize) -> Vec<E> {
        debug_assert_eq!(self.logical_cols, 1);
        debug_assert!(self.terms.iter().all(|term| term.col_weights.len() == 1));
        debug_assert_eq!(self.terms.len(), row_weights.len());
        let mut values = vec![E::zero(); rows];
        for (row_weights, term) in row_weights.into_iter().zip(self.terms) {
            let scale = term.scale * term.col_weights[0];
            #[cfg(feature = "parallel")]
            if rows >= ROW_PARALLEL_THRESHOLD {
                values
                    .par_iter_mut()
                    .zip(row_weights.par_iter())
                    .for_each(|(value, row_weight)| *value += scale * *row_weight);
                continue;
            }
            for (value, row_weight) in values.iter_mut().zip(row_weights) {
                *value += scale * row_weight;
            }
        }
        values
    }
}

fn validate_claims<E: ShortFieldElement>(
    claims: &[ShortEvalClaim<E>],
    log_rows: usize,
    log_cols: usize,
) -> Result<(usize, usize), Error> {
    let rows = 1usize
        .checked_shl(log_rows as u32)
        .ok_or(Error::EncodingError)?;
    let cols = 1usize
        .checked_shl(log_cols as u32)
        .ok_or(Error::EncodingError)?;
    if claims.is_empty()
        || claims.iter().any(|claim| {
            claim.row_point.len() != log_rows
                || claim.col_weight.is_empty()
                || claim.col_weight.len() > cols
        })
    {
        return Err(Error::InvalidNumberOfVariables);
    }
    Ok((rows, cols))
}

fn combined_target<E: ShortFieldElement>(claims: &[ShortEvalClaim<E>], rho: E) -> (E, Vec<E>) {
    let mut target = E::zero();
    let mut power = E::one();
    let mut powers = Vec::with_capacity(claims.len());
    for claim in claims {
        powers.push(power);
        target += power * claim.value;
        power *= rho;
    }
    (target, powers)
}

fn expected_weight_eval<E: ShortFieldElement>(
    claims: &[ShortEvalClaim<E>],
    powers: &[E],
    log_cols: usize,
    point: &[E],
) -> Result<E, Error> {
    if point.len() < log_cols {
        return Err(Error::InvalidNumberOfVariables);
    }
    let (col_point, row_point) = point.split_at(log_cols);
    let col_tensor = tensor_weights(col_point);
    claims
        .iter()
        .zip(powers)
        .try_fold(E::zero(), |sum, (claim, power)| -> Result<E, Error> {
            let col_eval = claim
                .col_weight
                .iter()
                .zip(&col_tensor)
                .fold(E::zero(), |acc, (weight, tensor)| acc + *weight * *tensor);
            Ok(sum + *power * col_eval * tensor_inner_product(&claim.row_point, row_point)?)
        })
}

fn absorb_multi_eval_statement_writer<B: ShortBaseField>(
    claims: &[ShortEvalClaim<B::Extension>],
    log_rows: usize,
    log_cols: usize,
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<(), Error> {
    transcript.absorb_bytes(b"multi-eval-domain", b"qafold-short-decide/multi-eval/v2");
    transcript.absorb_bytes(b"multi-eval-log-rows", &(log_rows as u64).to_le_bytes());
    transcript.absorb_bytes(b"multi-eval-log-cols", &(log_cols as u64).to_le_bytes());
    transcript.absorb_bytes(
        b"multi-eval-claim-count",
        &(claims.len() as u64).to_le_bytes(),
    );
    for claim in claims {
        transcript.absorb_extension(b"multi-eval-claim-value", claim.value)?;
        transcript.absorb_extensions(b"multi-eval-row-point", &claim.row_point)?;
        transcript.absorb_extensions(b"multi-eval-column-weight", &claim.col_weight)?;
    }
    Ok(())
}

fn absorb_multi_eval_statement_reader<B: ShortBaseField>(
    claims: &[ShortEvalClaim<B::Extension>],
    log_rows: usize,
    log_cols: usize,
    transcript: &mut ShortTranscriptReader<'_, B>,
) -> Result<(), Error> {
    transcript.absorb_bytes(b"multi-eval-domain", b"qafold-short-decide/multi-eval/v2");
    transcript.absorb_bytes(b"multi-eval-log-rows", &(log_rows as u64).to_le_bytes());
    transcript.absorb_bytes(b"multi-eval-log-cols", &(log_cols as u64).to_le_bytes());
    transcript.absorb_bytes(
        b"multi-eval-claim-count",
        &(claims.len() as u64).to_le_bytes(),
    );
    for claim in claims {
        transcript.absorb_extension(b"multi-eval-claim-value", claim.value)?;
        transcript.absorb_extensions(b"multi-eval-row-point", &claim.row_point)?;
        transcript.absorb_extensions(b"multi-eval-column-weight", &claim.col_weight)?;
    }
    Ok(())
}

fn source_evaluations<B: ShortBaseField>(
    setup_columns: &[&[B]],
    carry_columns: &[&[B::Extension]],
    row_weights: &[Vec<B::Extension>],
) -> Result<Vec<Vec<B::Extension>>, Error> {
    let real_cols = setup_columns
        .len()
        .checked_add(carry_columns.len())
        .ok_or(Error::EncodingError)?;
    let mut evaluations = Vec::with_capacity(row_weights.len());
    for row_weights in row_weights {
        let evaluate_serial = |source: usize| {
            if source < setup_columns.len() {
                row_weights
                    .iter()
                    .zip(setup_columns[source])
                    .fold(B::Extension::zero(), |sum, (weight, value)| {
                        sum + weight.mul_base(*value)
                    })
            } else {
                row_weights
                    .iter()
                    .zip(carry_columns[source - setup_columns.len()])
                    .fold(B::Extension::zero(), |sum, (weight, value)| {
                        sum + *weight * *value
                    })
            }
        };
        #[cfg(feature = "parallel")]
        let claim_evaluations = if real_cols >= COLUMN_PARALLEL_THRESHOLD {
            (0..real_cols)
                .into_par_iter()
                .map(evaluate_serial)
                .collect()
        } else if row_weights.len() >= ROW_PARALLEL_THRESHOLD {
            let evaluate_rows_parallel = |source: usize| {
                if source < setup_columns.len() {
                    row_weights
                        .par_iter()
                        .zip(setup_columns[source].par_iter())
                        .map(|(weight, value)| weight.mul_base(*value))
                        .reduce(B::Extension::zero, |left, right| left + right)
                } else {
                    row_weights
                        .par_iter()
                        .zip(carry_columns[source - setup_columns.len()].par_iter())
                        .map(|(weight, value)| *weight * *value)
                        .reduce(B::Extension::zero, |left, right| left + right)
                }
            };
            (0..real_cols).map(evaluate_rows_parallel).collect()
        } else {
            (0..real_cols).map(evaluate_serial).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let claim_evaluations = (0..real_cols).map(evaluate_serial).collect();
        evaluations.push(claim_evaluations);
    }
    Ok(evaluations)
}

fn claim_row_weights<E: ShortFieldElement>(
    claims: &[ShortEvalClaim<E>],
    rows: usize,
) -> Result<Vec<Vec<E>>, Error> {
    let weights = claims
        .iter()
        .map(|claim| tensor_weights(&claim.row_point))
        .collect::<Vec<_>>();
    if weights.iter().any(|weights| weights.len() != rows) {
        return Err(Error::InvalidNumberOfVariables);
    }
    Ok(weights)
}

fn mixed_cross_terms<B: ShortBaseField>(
    matrix: &MixedColumnFoldState<'_, B>,
    weights: &MultiEvalColumnWeightState<B::Extension>,
    source_evaluations: &[Vec<B::Extension>],
) -> (B::Extension, B::Extension) {
    let half = matrix.logical_cols / 2;
    let fold_column = |column: usize| {
        (0..source_evaluations.len()).fold(
            (B::Extension::zero(), B::Extension::zero()),
            |(left_cross, right_cross), claim| {
                let lower = matrix.evaluation_at_claim(column, claim, source_evaluations);
                let upper = matrix.evaluation_at_claim(column + half, claim, source_evaluations);
                (
                    left_cross + lower * weights.value(claim, column + half),
                    right_cross + upper * weights.value(claim, column),
                )
            },
        )
    };
    #[cfg(feature = "parallel")]
    if half >= COLUMN_PARALLEL_THRESHOLD {
        return (0..half).into_par_iter().map(fold_column).reduce(
            || (B::Extension::zero(), B::Extension::zero()),
            |(left_a, right_a), (left_b, right_b)| (left_a + left_b, right_a + right_b),
        );
    }
    (0..half).map(fold_column).fold(
        (B::Extension::zero(), B::Extension::zero()),
        |(left_a, right_a), (left_b, right_b)| (left_a + left_b, right_a + right_b),
    )
}

fn parallel_inner_product<E: ShortFieldElement>(left: &[E], right: &[E]) -> Result<E, Error> {
    inner_product(left, right)
}

pub(super) fn prove_multi_eval<B: ShortBaseField>(
    claims: &[ShortEvalClaim<B::Extension>],
    setup_columns: &[&[B]],
    carry_columns: &[&[B::Extension]],
    log_rows: usize,
    log_cols: usize,
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<ShortMultiEvalProverResult<B::Extension>, Error> {
    let (rows, cols) = validate_claims(claims, log_rows, log_cols)?;
    let mut matrix = MixedColumnFoldState::new(setup_columns, carry_columns, rows, cols)?;
    absorb_multi_eval_statement_writer(claims, log_rows, log_cols, transcript)?;
    let rho = transcript.challenge_extension(b"multi-eval-batch")?;
    let (mut rhs, powers) = combined_target(claims, rho);
    let row_weights = claim_row_weights(claims, rows)?;
    let source_evaluations = source_evaluations(setup_columns, carry_columns, &row_weights)?;
    let mut weights = MultiEvalColumnWeightState::new(claims, &powers, cols);
    let mut a_point = Vec::with_capacity(log_rows + log_cols);
    let mut b_point = Vec::with_capacity(log_rows + log_cols);

    for _ in 0..log_cols {
        let (left_cross, right_cross) = mixed_cross_terms(&matrix, &weights, &source_evaluations);
        transcript.push_response(left_cross)?;
        transcript.push_response(right_cross)?;
        let challenge = transcript.challenge_nonzero_extension(b"multi-eval-fold")?;
        let challenge_inverse = challenge.try_inverse().ok_or(Error::TranscriptError)?;
        rhs += left_cross * challenge + right_cross * challenge_inverse;
        matrix.fold_columns(challenge_inverse);
        weights.fold_columns(challenge);
        a_point.push(challenge_inverse);
        b_point.push(challenge);
    }

    drop(source_evaluations);
    let mut right = weights.into_row_values(row_weights, rows);
    let folded_vector = matrix.into_values();
    let mut left = folded_vector.clone();
    for _ in 0..log_rows {
        let half = left.len() / 2;
        let (left_cross, right_cross) = cross_inner_products(&left, &right);
        transcript.push_response(left_cross)?;
        transcript.push_response(right_cross)?;
        let challenge = transcript.challenge_nonzero_extension(b"multi-eval-fold")?;
        let challenge_inverse = challenge.try_inverse().ok_or(Error::TranscriptError)?;
        rhs += left_cross * challenge + right_cross * challenge_inverse;
        fold_litebullet_vectors(&mut left, &mut right, challenge_inverse, challenge);
        left.truncate(half);
        right.truncate(half);
        a_point.push(challenge_inverse);
        b_point.push(challenge);
    }
    let a_value = left[0];
    let b_value = right[0];
    transcript.push_response(a_value)?;
    transcript.push_response(b_value)?;
    let expected_weight = expected_weight_eval(claims, &powers, log_cols, &b_point)?;
    Ok(ShortMultiEvalProverResult {
        reduction: ShortMultiEvalReduction {
            ok: a_value * b_value == rhs && b_value == expected_weight,
            reduced: ShortEvalWithPoint {
                value: a_value,
                point: a_point,
            },
        },
        folded_vector,
    })
}

pub(super) fn verify_multi_eval<B: ShortBaseField>(
    claims: &[ShortEvalClaim<B::Extension>],
    log_rows: usize,
    log_cols: usize,
    transcript: &mut ShortTranscriptReader<'_, B>,
) -> Result<ShortMultiEvalReduction<B::Extension>, Error> {
    validate_claims(claims, log_rows, log_cols)?;
    absorb_multi_eval_statement_reader(claims, log_rows, log_cols, transcript)?;
    let rho = transcript.challenge_extension(b"multi-eval-batch")?;
    let (mut rhs, powers) = combined_target(claims, rho);
    let mut a_point = Vec::with_capacity(log_rows + log_cols);
    let mut b_point = Vec::with_capacity(log_rows + log_cols);
    for _ in 0..log_rows + log_cols {
        let left_cross = transcript.read_response()?;
        let right_cross = transcript.read_response()?;
        let challenge = transcript.challenge_nonzero_extension(b"multi-eval-fold")?;
        let challenge_inverse = challenge.try_inverse().ok_or(Error::TranscriptError)?;
        rhs += left_cross * challenge + right_cross * challenge_inverse;
        a_point.push(challenge_inverse);
        b_point.push(challenge);
    }
    let a_value = transcript.read_response()?;
    let b_value = transcript.read_response()?;
    let expected_weight = expected_weight_eval(claims, &powers, log_cols, &b_point)?;
    Ok(ShortMultiEvalReduction {
        ok: a_value * b_value == rhs && b_value == expected_weight,
        reduced: ShortEvalWithPoint {
            value: a_value,
            point: a_point,
        },
    })
}

fn absorb_rfc_generator_statement_writer<B: ShortBaseField>(
    source: &ShortQaRandomFoldableCode<B>,
    target: &ShortQaCode<B>,
    target_level: usize,
    value: B::Extension,
    left_point: &[B::Extension],
    right_point: &[B::Extension],
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<(), Error> {
    transcript.absorb_bytes(
        b"rfc-generator-domain",
        b"qafold-short-decide/rfc-generator/v1",
    );
    transcript.absorb_bytes(
        b"rfc-generator-target-level",
        &(target_level as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"rfc-generator-source-level",
        &(source.target_level() as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"rfc-generator-layers",
        &(source.layers() as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"rfc-generator-target-message-len",
        &(target.message_len() as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"rfc-generator-target-index",
        &(target.index() as u64).to_le_bytes(),
    );
    transcript.absorb_extension(b"rfc-generator-value", value)?;
    transcript.absorb_extensions(b"rfc-generator-left-point", left_point)?;
    transcript.absorb_extensions(b"rfc-generator-right-point", right_point)?;
    Ok(())
}

fn absorb_rfc_generator_statement_reader<B: ShortBaseField>(
    source: &ShortQaRandomFoldableCode<B>,
    target: &ShortQaCode<B>,
    target_level: usize,
    value: B::Extension,
    left_point: &[B::Extension],
    right_point: &[B::Extension],
    transcript: &mut ShortTranscriptReader<'_, B>,
) -> Result<(), Error> {
    transcript.absorb_bytes(
        b"rfc-generator-domain",
        b"qafold-short-decide/rfc-generator/v1",
    );
    transcript.absorb_bytes(
        b"rfc-generator-target-level",
        &(target_level as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"rfc-generator-source-level",
        &(source.target_level() as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"rfc-generator-layers",
        &(source.layers() as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"rfc-generator-target-message-len",
        &(target.message_len() as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"rfc-generator-target-index",
        &(target.index() as u64).to_le_bytes(),
    );
    transcript.absorb_extension(b"rfc-generator-value", value)?;
    transcript.absorb_extensions(b"rfc-generator-left-point", left_point)?;
    transcript.absorb_extensions(b"rfc-generator-right-point", right_point)?;
    Ok(())
}

pub(super) fn reduce_qa_rfc_generator_claim_prover<B: ShortBaseField>(
    source: &ShortQaRandomFoldableCode<B>,
    target: &ShortQaCode<B>,
    target_level: usize,
    value: B::Extension,
    left_point: &[B::Extension],
    right_point: &[B::Extension],
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<
    (
        ShortQaMatrixClaim<B::Extension>,
        Vec<ShortQaRfcTClaim<B::Extension>>,
    ),
    Error,
> {
    let layers = source.layers();
    if source.target_level() != target_level
        || layers == 0
        || left_point.len() < layers
        || right_point.len() < layers
    {
        return Err(Error::InvalidNumberOfVariables);
    }
    absorb_rfc_generator_statement_writer(
        source,
        target,
        target_level,
        value,
        left_point,
        right_point,
        transcript,
    )?;

    let mut current_value = value;
    let mut current_left = left_point.to_vec();
    let mut current_right = right_point.to_vec();
    let mut t_claims = Vec::with_capacity(layers);
    for remaining_layers in (1..=layers).rev() {
        if current_left.is_empty() || current_right.is_empty() {
            return Err(Error::InvalidNumberOfVariables);
        }
        let row_fold = current_left[0];
        let col_fold = current_right[0];
        let child_left_point = current_left[1..].to_vec();
        let child_right_point = current_right[1..].to_vec();
        let child_left = tensor_weights(&child_left_point);
        let child_right = tensor_weights(&child_right_point);
        let generator_right = source.right_projection_extension_with_layers(
            target,
            &child_right,
            remaining_layers - 1,
        )?;
        let diagonal = source.diagonal(remaining_layers)?;
        if child_left.len() != generator_right.len() || diagonal.len() != child_left.len() {
            return Err(Error::InvalidNumberOfVariables);
        }
        let direct_value = parallel_inner_product(&child_left, &generator_right)?;
        let weighted_t_left = child_left
            .iter()
            .zip(diagonal)
            .map(|(left, diagonal)| left.mul_base(*diagonal))
            .collect::<Vec<_>>();
        let twisted_value = parallel_inner_product(&weighted_t_left, &generator_right)?;
        transcript.push_response(direct_value)?;
        transcript.push_response(twisted_value)?;
        if current_value
            != (B::Extension::one() + row_fold) * direct_value
                + col_fold * (B::Extension::one() - row_fold) * twisted_value
        {
            return Err(Error::InvalidCommitment);
        }

        let twisted_reduction = prove_litebullet::<B>(
            twisted_value,
            weighted_t_left,
            generator_right.clone(),
            transcript,
            b"rfc-twisted-litebullet",
        )?;
        let t_point = child_left_point
            .iter()
            .zip(&twisted_reduction.a_point)
            .map(|(left, reduction)| *left * *reduction)
            .collect::<Vec<_>>();
        t_claims.push(ShortQaRfcTClaim {
            layer: remaining_layers,
            claim: ShortEvalWithPoint {
                value: twisted_reduction.a_value,
                point: t_point,
            },
        });

        let rho = transcript.challenge_extension(b"rfc-direct-twisted-batch")?;
        let combined_value = direct_value + rho * twisted_reduction.b_value;
        let twisted_right = tensor_weights(&twisted_reduction.b_point);
        let combined_left = child_left
            .iter()
            .zip(twisted_right)
            .map(|(direct, twisted)| *direct + rho * twisted)
            .collect::<Vec<_>>();
        let batch = prove_litebullet::<B>(
            combined_value,
            combined_left,
            generator_right,
            transcript,
            b"rfc-direct-twisted-litebullet",
        )?;
        current_value = batch.b_value;
        current_left = batch.b_point;
        current_right = child_right_point;
    }
    if current_left.len() != target.codeword_len()?.ilog2() as usize
        || current_right.len() != target.message_len().ilog2() as usize
    {
        return Err(Error::InvalidNumberOfVariables);
    }
    Ok((
        ShortQaMatrixClaim {
            value: current_value,
            point: (current_left, current_right),
        },
        t_claims,
    ))
}

pub(super) fn reduce_qa_rfc_generator_claim_verifier<B: ShortBaseField>(
    source: &ShortQaRandomFoldableCode<B>,
    target: &ShortQaCode<B>,
    target_level: usize,
    value: B::Extension,
    left_point: &[B::Extension],
    right_point: &[B::Extension],
    transcript: &mut ShortTranscriptReader<'_, B>,
) -> Result<
    (
        ShortQaMatrixClaim<B::Extension>,
        Vec<ShortQaRfcTClaim<B::Extension>>,
    ),
    Error,
> {
    let layers = source.layers();
    if source.target_level() != target_level
        || layers == 0
        || left_point.len() < layers
        || right_point.len() < layers
    {
        return Err(Error::InvalidNumberOfVariables);
    }
    absorb_rfc_generator_statement_reader(
        source,
        target,
        target_level,
        value,
        left_point,
        right_point,
        transcript,
    )?;

    let mut current_value = value;
    let mut current_left = left_point.to_vec();
    let mut current_right = right_point.to_vec();
    let mut t_claims = Vec::with_capacity(layers);
    for remaining_layers in (1..=layers).rev() {
        if current_left.is_empty() || current_right.is_empty() {
            return Err(Error::InvalidNumberOfVariables);
        }
        let row_fold = current_left[0];
        let col_fold = current_right[0];
        let child_left_point = current_left[1..].to_vec();
        let child_right_point = current_right[1..].to_vec();
        let child_len = source.diagonal(remaining_layers)?.len();
        if child_len == 0 || !child_len.is_power_of_two() {
            return Err(Error::InvalidNumberOfVariables);
        }
        let direct_value = transcript.read_response()?;
        let twisted_value = transcript.read_response()?;
        if current_value
            != (B::Extension::one() + row_fold) * direct_value
                + col_fold * (B::Extension::one() - row_fold) * twisted_value
        {
            return Err(Error::InvalidCommitment);
        }
        let twisted_reduction = verify_litebullet::<B>(
            twisted_value,
            child_len,
            transcript,
            b"rfc-twisted-litebullet",
        )?
        .ok_or(Error::InvalidCommitment)?;
        let t_point = child_left_point
            .iter()
            .zip(&twisted_reduction.a_point)
            .map(|(left, reduction)| *left * *reduction)
            .collect::<Vec<_>>();
        if t_point.len() != child_left_point.len() {
            return Err(Error::InvalidNumberOfVariables);
        }
        t_claims.push(ShortQaRfcTClaim {
            layer: remaining_layers,
            claim: ShortEvalWithPoint {
                value: twisted_reduction.a_value,
                point: t_point,
            },
        });

        let rho = transcript.challenge_extension(b"rfc-direct-twisted-batch")?;
        let combined_value = direct_value + rho * twisted_reduction.b_value;
        let batch = verify_litebullet::<B>(
            combined_value,
            child_len,
            transcript,
            b"rfc-direct-twisted-litebullet",
        )?
        .ok_or(Error::InvalidCommitment)?;
        let expected_left = tensor_inner_product(&child_left_point, &batch.a_point)?
            + rho * tensor_inner_product(&twisted_reduction.b_point, &batch.a_point)?;
        if batch.a_value != expected_left {
            return Err(Error::InvalidCommitment);
        }
        current_value = batch.b_value;
        current_left = batch.b_point;
        current_right = child_right_point;
    }
    if current_left.len() != target.codeword_len()?.ilog2() as usize
        || current_right.len() != target.message_len().ilog2() as usize
    {
        return Err(Error::InvalidNumberOfVariables);
    }
    Ok((
        ShortQaMatrixClaim {
            value: current_value,
            point: (current_left, current_right),
        },
        t_claims,
    ))
}

fn evaluate_base_diagonal_claim<B: ShortBaseField>(
    diagonal: &[B],
    claim: &ShortEvalWithPoint<B::Extension>,
) -> Result<bool, Error> {
    let expected_len = 1usize
        .checked_shl(claim.point.len() as u32)
        .ok_or(Error::InvalidNumberOfVariables)?;
    if diagonal.len() != expected_len {
        return Err(Error::InvalidNumberOfVariables);
    }
    let weights = tensor_weights(&claim.point);
    #[cfg(feature = "parallel")]
    let value = if diagonal.len() >= ROW_PARALLEL_THRESHOLD {
        diagonal
            .par_iter()
            .zip(weights.par_iter())
            .map(|(diagonal, weight)| weight.mul_base(*diagonal))
            .reduce(B::Extension::zero, |left, right| left + right)
    } else {
        diagonal
            .iter()
            .zip(weights)
            .fold(B::Extension::zero(), |sum, (diagonal, weight)| {
                sum + weight.mul_base(*diagonal)
            })
    };
    #[cfg(not(feature = "parallel"))]
    let value = diagonal
        .iter()
        .zip(weights)
        .fold(B::Extension::zero(), |sum, (diagonal, weight)| {
            sum + weight.mul_base(*diagonal)
        });
    Ok(value == claim.value)
}

pub(super) fn verify_base_t_claims<B: ShortBaseField>(
    source: &ShortQaRandomFoldableCode<B>,
    t_claims: &[ShortQaRfcTClaim<B::Extension>],
) -> Result<bool, Error> {
    if t_claims.len() != source.layers() {
        return Ok(false);
    }
    for (position, t_claim) in t_claims.iter().enumerate() {
        let expected_layer = source.layers() - position;
        if t_claim.layer != expected_layer
            || !evaluate_base_diagonal_claim::<B>(source.diagonal(t_claim.layer)?, &t_claim.claim)?
        {
            return Ok(false);
        }
    }
    Ok(true)
}
