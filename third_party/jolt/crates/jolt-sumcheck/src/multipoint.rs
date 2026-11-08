//! Reduction from polynomial openings at distinct points to one same-point batch opening.
//!
//! For claims `f_i(r_i) = v_i`, the protocol samples batching coefficients
//! `gamma_i` and proves the degree-two sumcheck
//!
//! `sum_x sum_i gamma_i * eq(r_i, x) * f_i(x) = sum_i gamma_i * v_i`.
//!
//! The sumcheck leaves all polynomials at one transcript-derived point `z`.
//! The verifier checks its final claim against the supplied `f_i(z)` values,
//! then delegates those values to [`HomomorphicBatch`] for one PCS proof.

use std::marker::PhantomData;

use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;
use jolt_openings::{
    AdditivelyHomomorphic, BatchOpeningScheme, EvaluationClaim, HomomorphicBatch, OpeningsError,
    VerifierOpeningClaim,
};
use jolt_poly::{EqPolynomial, MultilinearPoly, Point, Polynomial, UnivariatePoly, HIGH_TO_LOW};
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::{
    prove_batch, BatchMember, BatchPrelude, BooleanHypercube, ClearProof, ClearSumcheckRecorder,
    CompressedSumcheckProof, ProveRounds, SequentialRounds, SumcheckError, SumcheckProof,
    SumcheckRecorder, SumcheckVerifier, OPENING_CLAIM_TRANSCRIPT_LABEL,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};

const MULTIPOINT_CLAIMS_LABEL: &[u8] = b"multipoint_claims";
const MULTIPOINT_COMMITMENT_LABEL: &[u8] = b"multipoint_commitment";
const REDUCTION_DEGREE: usize = 2;

/// A protocol adapter that reduces distinct-point claims before invoking
/// [`HomomorphicBatch`] at the resulting shared point.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MultiPointBatch<PCS>(PhantomData<PCS>);

/// Proof for [`MultiPointBatch`]: the distinct-point reduction, the values at
/// its shared point, and the underlying same-point PCS proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, P: Serialize",
    deserialize = "F: for<'a> Deserialize<'a>, P: for<'a> Deserialize<'a>"
))]
pub struct MultiPointBatchProof<F: Field, P> {
    /// Compressed degree-two sumcheck reducing every original point to one point.
    pub reduction_sumcheck: CompressedSumcheckProof<F>,
    /// Claimed evaluations of the batch polynomials at the reduced point.
    pub reduced_evaluations: Vec<F>,
    /// Same-point opening proof for `reduced_evaluations`.
    pub opening_proof: P,
}

impl<PCS> BatchOpeningScheme for MultiPointBatch<PCS>
where
    PCS: AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<PCS::Field> + AppendToTranscript,
{
    type Field = PCS::Field;
    type ProverSetup = PCS::ProverSetup;
    type VerifierSetup = PCS::VerifierSetup;
    type Statement = Vec<VerifierOpeningClaim<PCS::Field, PCS::Output>>;
    type Polynomials<'a>
        = Vec<&'a (dyn MultilinearPoly<PCS::Field> + 'a)>
    where
        Self: 'a;
    type Hints = Vec<PCS::OpeningHint>;
    type Proof = MultiPointBatchProof<PCS::Field, PCS::Proof>;

    #[tracing::instrument(
        skip_all,
        name = "MultiPointBatch::prove_batch",
        fields(claims = statement.len())
    )]
    fn prove_batch<'a, T>(
        setup: &Self::ProverSetup,
        statement: Self::Statement,
        polynomials: Self::Polynomials<'a>,
        hints: Self::Hints,
        transcript: &mut T,
    ) -> Result<Self::Proof, OpeningsError>
    where
        Self: 'a,
        T: Transcript<Challenge = Self::Field>,
    {
        let num_vars = validate_prover_inputs(&statement, &polynomials, &hints)?;
        append_statement(&statement, transcript);
        let coefficients = transcript.challenge_scalar_powers(statement.len());
        let prelude = reduction_prelude(&statement, &coefficients, num_vars);

        let mut members = statement
            .iter()
            .zip(&polynomials)
            .map(|(claim, polynomial)| {
                MultiPointMember::new(*polynomial, claim.evaluation.point.as_slice())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut member_refs: Vec<&mut dyn ProveRounds<PCS::Field>> = members
            .iter_mut()
            .map(|member| member as &mut dyn ProveRounds<PCS::Field>)
            .collect();
        let mut recorder = ClearSumcheckRecorder::<PCS::Field, ()>::new();
        let proved = prove_batch(
            &prelude,
            &mut member_refs,
            &mut SequentialRounds,
            &mut recorder,
            transcript,
        )
        .map_err(|error| OpeningsError::ProveFailed(error.to_string()))?;
        drop(member_refs);

        let reduced_evaluations = members
            .iter()
            .map(MultiPointMember::final_evaluation)
            .collect::<Result<Vec<_>, _>>()?;
        let expected_final = reduction_final_claim(
            &statement,
            &coefficients,
            &proved.challenges,
            &reduced_evaluations,
        );
        if expected_final != proved.final_claim {
            return Err(OpeningsError::ProveFailed(
                "multipoint reduction final claim mismatch".to_owned(),
            ));
        }

        let recorded = recorder
            .finish(&reduced_evaluations, transcript)
            .map_err(|error| OpeningsError::ProveFailed(error.to_string()))?;
        let reduction_sumcheck = match recorded.proof {
            SumcheckProof::Clear(ClearProof::Compressed(proof)) => proof,
            SumcheckProof::Clear(ClearProof::Full(_)) => {
                return Err(OpeningsError::ProveFailed(
                    "multipoint reduction produced an unexpected full proof".to_owned(),
                ));
            }
            SumcheckProof::Committed(_) => {
                return Err(OpeningsError::ProveFailed(
                    "multipoint reduction produced an unexpected committed proof".to_owned(),
                ));
            }
        };

        let reduced_point = Point::<HIGH_TO_LOW, PCS::Field>::new(proved.challenges);
        let reduced_statement = reduced_statement(statement, &reduced_point, &reduced_evaluations);
        let opening_proof = HomomorphicBatch::<PCS>::prove_batch(
            setup,
            reduced_statement,
            polynomials,
            hints,
            transcript,
        )?;

        Ok(MultiPointBatchProof {
            reduction_sumcheck,
            reduced_evaluations,
            opening_proof,
        })
    }

    fn verify_batch<T>(
        setup: &Self::VerifierSetup,
        statement: &Self::Statement,
        proof: &Self::Proof,
        transcript: &mut T,
    ) -> Result<(), OpeningsError>
    where
        T: Transcript<Challenge = Self::Field>,
    {
        let num_vars = validate_verifier_inputs(statement, proof.reduced_evaluations.len())?;
        append_statement(statement, transcript);
        let coefficients = transcript.challenge_scalar_powers(statement.len());
        let prelude = reduction_prelude(statement, &coefficients, num_vars);
        let reduced_claim = SumcheckVerifier::verify_compressed(
            &crate::SumcheckClaim::new(num_vars, REDUCTION_DEGREE, prelude.claimed_sum),
            &proof.reduction_sumcheck,
            BooleanHypercube,
            SUMCHECK_ROUND_TRANSCRIPT_LABEL,
            transcript,
        )
        .map_err(|_| OpeningsError::VerificationFailed)?;
        for evaluation in &proof.reduced_evaluations {
            transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, evaluation);
        }

        let expected_final = reduction_final_claim(
            statement,
            &coefficients,
            reduced_claim.point.as_slice(),
            &proof.reduced_evaluations,
        );
        if expected_final != reduced_claim.value {
            return Err(OpeningsError::VerificationFailed);
        }

        let reduced_statement = reduced_statement(
            statement.clone(),
            &reduced_claim.point,
            &proof.reduced_evaluations,
        );
        HomomorphicBatch::<PCS>::verify_batch(
            setup,
            &reduced_statement,
            &proof.opening_proof,
            transcript,
        )
    }
}

fn validate_prover_inputs<F, C, H>(
    statement: &[VerifierOpeningClaim<F, C>],
    polynomials: &[&dyn MultilinearPoly<F>],
    hints: &[H],
) -> Result<usize, OpeningsError>
where
    F: Field,
{
    let num_vars = validate_statement(statement)?;
    if polynomials.len() != statement.len() || hints.len() != statement.len() {
        return Err(OpeningsError::InvalidBatch(format!(
            "{} polynomials and {} hints do not match claim count {}",
            polynomials.len(),
            hints.len(),
            statement.len()
        )));
    }
    for polynomial in polynomials {
        if polynomial.num_vars() != num_vars {
            return Err(OpeningsError::InvalidBatch(format!(
                "polynomial has {} variables but multipoint claims have {num_vars}",
                polynomial.num_vars()
            )));
        }
    }
    Ok(num_vars)
}

fn validate_verifier_inputs<F, C>(
    statement: &[VerifierOpeningClaim<F, C>],
    reduced_evaluation_count: usize,
) -> Result<usize, OpeningsError>
where
    F: Field,
{
    let num_vars = validate_statement(statement)?;
    if reduced_evaluation_count != statement.len() {
        return Err(OpeningsError::InvalidBatch(format!(
            "{} reduced evaluations do not match claim count {}",
            reduced_evaluation_count,
            statement.len()
        )));
    }
    Ok(num_vars)
}

fn validate_statement<F, C>(
    statement: &[VerifierOpeningClaim<F, C>],
) -> Result<usize, OpeningsError>
where
    F: Field,
{
    let first = statement.first().ok_or_else(|| {
        OpeningsError::InvalidBatch(
            "multipoint batch opening requires at least one claim".to_owned(),
        )
    })?;
    let num_vars = first.evaluation.point.len();
    for claim in statement.iter().skip(1) {
        if claim.evaluation.point.len() != num_vars {
            return Err(OpeningsError::InvalidBatch(
                "multipoint batch opening claims must have one common dimension".to_owned(),
            ));
        }
    }
    Ok(num_vars)
}

fn append_statement<F, C, T>(statement: &[VerifierOpeningClaim<F, C>], transcript: &mut T)
where
    F: Field,
    C: AppendToTranscript,
    T: Transcript<Challenge = F>,
{
    transcript.append(&LabelWithCount(
        MULTIPOINT_CLAIMS_LABEL,
        statement.len() as u64,
    ));
    for claim in statement {
        transcript.append(&Label(MULTIPOINT_COMMITMENT_LABEL));
        claim.commitment.append_to_transcript(transcript);
        claim.evaluation.append_to_transcript(transcript);
    }
}

fn reduction_prelude<F, C>(
    statement: &[VerifierOpeningClaim<F, C>],
    coefficients: &[F],
    num_vars: usize,
) -> BatchPrelude<F>
where
    F: Field,
{
    let members = statement
        .iter()
        .zip(coefficients)
        .map(|(claim, coefficient)| BatchMember {
            input_claim: claim.evaluation.value,
            coefficient: *coefficient,
            rounds: num_vars,
            offset: 0,
        })
        .collect();
    BatchPrelude::new(members, num_vars, REDUCTION_DEGREE)
}

fn reduction_final_claim<F, C>(
    statement: &[VerifierOpeningClaim<F, C>],
    coefficients: &[F],
    reduced_point: &[F],
    reduced_evaluations: &[F],
) -> F
where
    F: Field,
{
    statement
        .iter()
        .zip(coefficients)
        .zip(reduced_evaluations)
        .map(|((claim, coefficient), evaluation)| {
            *coefficient
                * EqPolynomial::<F>::mle(claim.evaluation.point.as_slice(), reduced_point)
                * *evaluation
        })
        .sum()
}

fn reduced_statement<F, C>(
    statement: Vec<VerifierOpeningClaim<F, C>>,
    reduced_point: &Point<HIGH_TO_LOW, F>,
    reduced_evaluations: &[F],
) -> Vec<VerifierOpeningClaim<F, C>>
where
    F: Field,
{
    statement
        .into_iter()
        .zip(reduced_evaluations)
        .map(|(claim, evaluation)| VerifierOpeningClaim {
            commitment: claim.commitment,
            evaluation: EvaluationClaim::new(reduced_point.clone(), *evaluation),
        })
        .collect()
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ReductionPhase {
    Rows,
    Columns,
}

struct MultiPointMember<'a, F: Field> {
    source: &'a dyn MultilinearPoly<F>,
    row_target: Vec<F>,
    column_target: Vec<F>,
    row_challenges: Vec<F>,
    sigma: usize,
    values: Polynomial<F>,
    equality: Polynomial<F>,
    phase: ReductionPhase,
}

impl<'a, F: Field> MultiPointMember<'a, F> {
    fn new(source: &'a dyn MultilinearPoly<F>, point: &[F]) -> Result<Self, OpeningsError> {
        let num_vars = point.len();
        let sigma = num_vars.div_ceil(2);
        let nu = num_vars - sigma;
        let (row_target, column_target) = point.split_at(nu);

        if nu == 0 {
            let values = Polynomial::new(source.fold_rows(&[F::one()], sigma));
            let equality = Polynomial::new(EqPolynomial::<F>::evals(column_target, None));
            return Ok(Self {
                source,
                row_target: Vec::new(),
                column_target: column_target.to_vec(),
                row_challenges: Vec::new(),
                sigma,
                values,
                equality,
                phase: ReductionPhase::Columns,
            });
        }

        let column_eq = EqPolynomial::<F>::evals(column_target, None);
        let row_values = source.fold_columns(&column_eq, sigma);
        if row_values.len() != 1usize << nu {
            return Err(OpeningsError::InvalidBatch(
                "multilinear source yielded an invalid balanced row view".to_owned(),
            ));
        }

        Ok(Self {
            source,
            row_target: row_target.to_vec(),
            column_target: column_target.to_vec(),
            row_challenges: Vec::with_capacity(nu),
            sigma,
            values: Polynomial::new(row_values),
            equality: Polynomial::new(EqPolynomial::<F>::evals(row_target, None)),
            phase: ReductionPhase::Rows,
        })
    }

    fn bind(&mut self, challenge: F) -> Result<(), SumcheckError<F>> {
        self.values.bind(challenge);
        self.equality.bind(challenge);
        if self.phase == ReductionPhase::Rows {
            self.row_challenges.push(challenge);
            if self.values.num_vars() == 0 && self.sigma > 0 {
                self.start_column_phase()?;
            }
        }
        Ok(())
    }

    fn start_column_phase(&mut self) -> Result<(), SumcheckError<F>> {
        let scale = self
            .equality
            .evaluations()
            .first()
            .copied()
            .ok_or(SumcheckError::MissingEvaluationSource { kind: "equality" })?;
        let row_weights = EqPolynomial::<F>::evals(&self.row_challenges, None);
        let column_values = self.source.fold_rows(&row_weights, self.sigma);
        self.values = Polynomial::new(column_values);
        self.equality = Polynomial::new(EqPolynomial::<F>::evals(&self.column_target, Some(scale)));
        self.phase = ReductionPhase::Columns;
        Ok(())
    }

    fn round_polynomial(&self) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        let values = self.values.evaluations();
        let equality = self.equality.evaluations();
        if values.len() != equality.len() || values.len() < 2 {
            return Err(SumcheckError::MissingEvaluationSource {
                kind: "multipoint reduction",
            });
        }
        let half = values.len() / 2;
        let (values_zero, values_one) = values.split_at(half);
        let (eq_zero, eq_one) = equality.split_at(half);
        let (constant, linear, quadratic) = values_zero
            .par_iter()
            .zip(values_one)
            .zip(eq_zero)
            .zip(eq_one)
            .map(|(((value_zero, value_one), eq_zero), eq_one)| {
                let value_slope = *value_one - *value_zero;
                let eq_slope = *eq_one - *eq_zero;
                (
                    *value_zero * *eq_zero,
                    *value_zero * eq_slope + value_slope * *eq_zero,
                    value_slope * eq_slope,
                )
            })
            .reduce(
                || (F::zero(), F::zero(), F::zero()),
                |left, right| (left.0 + right.0, left.1 + right.1, left.2 + right.2),
            );
        Ok(UnivariatePoly::new(vec![constant, linear, quadratic]))
    }

    fn final_evaluation(&self) -> Result<F, OpeningsError> {
        self.values.evaluations().first().copied().ok_or_else(|| {
            OpeningsError::ProveFailed(
                "multipoint reduction did not produce a final evaluation".to_owned(),
            )
        })
    }
}

impl<F: Field> ProveRounds<F> for MultiPointMember<'_, F> {
    fn num_rounds(&self) -> usize {
        self.row_target.len() + self.column_target.len()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        _previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge)?;
        }
        self.round_polynomial()
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind)
    }
}
