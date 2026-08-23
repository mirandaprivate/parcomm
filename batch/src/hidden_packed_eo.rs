//! Packed EO-LiteBullet scalar transcript for Section 6.
//!
//! Cross terms, binary-round scalars, and terminal fold scalars are sent as
//! field elements. The resulting continuation state feeds two independent
//! hidden stages: a Dory opening of the registered dense BVA and the final
//! Section 6 TBV consistency relation.

use std::{
    fmt::{Display, Formatter},
    time::Instant,
};

use ark_bn254::Fr;
use ark_ff::{Field, One, Zero};
use ark_std::rand::RngCore;
use jolt_crypto::{Bn254G1, Pedersen, PedersenSetup, VectorCommitment};
use jolt_dory::DoryCommitment;
use jolt_field::Fr as JoltFr;
use jolt_transcript::domain::{Label, LabelWithCount, U64Word};
use jolt_transcript::{Blake2bTranscript, Transcript};
use serde::{Deserialize, Serialize};

use crate::{
    bucketed_ip::{
        packing_stride_for_bit_width, PackedEoChallengeTranscript, PackedEoLiteBulletProof,
        PaddingWorkCounters, StreamingEoError, StreamingEoProfile, TensorBasisVector,
    },
    hidden_relation::{
        prove_hidden_scalar_relation, verify_hidden_scalar_relation, HiddenRelationError,
        HiddenScalarRelationProof, HiddenScalarWitness, RelationExpression,
    },
    jolt_dory::{
        dense_i8_polynomial, normalize_monomial_mle_claim, CommittedDoryBva, DenseI8BvaSource,
        DoryProverParameters, DoryVerifierParameters,
    },
    jolt_dory_adapter::{
        open_hidden_dense_i8_zk, verify_hidden_dense_i8_zk, DenseI8DoryError,
        HiddenDenseI8DoryProof,
    },
};

/// Public Packed EO reduction proof consisting only of field scalars.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenPackedEoProof {
    pub packing_stride: usize,
    pub packed_positive: Vec<JoltFr>,
    pub packed_negative: Vec<JoltFr>,
    pub binary_rounds: Vec<(JoltFr, JoltFr)>,
    pub a_reduced: JoltFr,
    pub b_reduced: JoltFr,
}

impl HiddenPackedEoProof {
    /// Number of field elements sent by the Packed EO reduction.
    pub fn field_element_count(&self) -> usize {
        self.packed_positive.len() + self.packed_negative.len() + 2 * self.binary_rounds.len() + 2
    }
}

/// Independent Dory opening of the registered dense BVA at the point emitted
/// by Packed EO-LiteBullet.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenBvaOpeningProof {
    pub dory_opening: HiddenDenseI8DoryProof,
    pub evaluation_binding: HiddenScalarRelationProof,
}

/// Mutually exclusive timings for one hidden dense-BVA opening.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HiddenBvaOpeningProverProfile {
    pub point_preparation_seconds: f64,
    pub dense_preparation_seconds: f64,
    pub transcript_preparation_seconds: f64,
    pub dory_open_seconds: f64,
    pub scalar_binding_seconds: f64,
    pub total_seconds: f64,
}

impl HiddenBvaOpeningProverProfile {
    pub fn preparation_seconds(&self) -> f64 {
        self.point_preparation_seconds
            + self.dense_preparation_seconds
            + self.transcript_preparation_seconds
    }

    pub fn accounted_seconds(&self) -> f64 {
        self.preparation_seconds() + self.dory_open_seconds + self.scalar_binding_seconds
    }

    pub fn other_seconds(&self) -> f64 {
        (self.total_seconds - self.accounted_seconds()).max(0.0)
    }
}

/// Mutually exclusive timings for verifying one hidden dense-BVA opening.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HiddenBvaOpeningVerifierProfile {
    pub point_preparation_seconds: f64,
    pub transcript_preparation_seconds: f64,
    pub dory_verify_seconds: f64,
    pub scalar_binding_seconds: f64,
    pub total_seconds: f64,
}

impl HiddenBvaOpeningVerifierProfile {
    pub fn preparation_seconds(&self) -> f64 {
        self.point_preparation_seconds + self.transcript_preparation_seconds
    }

    pub fn accounted_seconds(&self) -> f64 {
        self.preparation_seconds() + self.dory_verify_seconds + self.scalar_binding_seconds
    }

    pub fn other_seconds(&self) -> f64 {
        (self.total_seconds - self.accounted_seconds()).max(0.0)
    }
}

/// Dense-BVA evaluation reduction whose PCS claim is opened by the one
/// terminal Section 6 Dory batch.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenBvaOpeningReductionProof {
    pub normalized_evaluation_commitment: Bn254G1,
    pub evaluation_binding: HiddenScalarRelationProof,
}

/// Prover-retained normalized dense-BVA claim for the terminal Dory batch.
pub struct PendingHiddenBvaOpening {
    point_msb: Vec<Fr>,
    normalized_evaluation: HiddenScalarWitness,
}

impl PendingHiddenBvaOpening {
    pub fn point_msb(&self) -> &[Fr] {
        &self.point_msb
    }

    pub fn evaluation(&self) -> &HiddenScalarWitness {
        &self.normalized_evaluation
    }
}

/// Commitment-only dense-BVA claim reconstructed by the verifier.
pub struct HiddenBvaOpeningReductionVerification {
    pub point_msb: Vec<Fr>,
    pub normalized_evaluation_commitment: Bn254G1,
}

/// Prover continuation shared by the independent hidden BVA and TBV stages.
#[derive(Clone)]
pub struct HiddenPackedEoProverState {
    pub(crate) a_reduced: HiddenScalarWitness,
    pub(crate) b_reduced: HiddenScalarWitness,
    pub(crate) challenges: Vec<Fr>,
    pub(crate) challenges_inv: Vec<Fr>,
    pub(crate) transcript_state: [u8; 32],
}

/// Verifier output passed to the hidden BVA and final TBV consistency relations.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HiddenPackedEoVerification {
    pub(crate) a_reduced_commitment: Bn254G1,
    pub(crate) b_reduced_commitment: Bn254G1,
    pub(crate) challenges: Vec<Fr>,
    pub(crate) challenges_inv: Vec<Fr>,
    pub(crate) transcript_state: [u8; 32],
}

#[derive(Debug)]
pub enum HiddenPackedEoError {
    Shape,
    ZeroChallenge,
    Dory(DenseI8DoryError),
    Relation(HiddenRelationError),
}

impl Display for HiddenPackedEoError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for HiddenPackedEoError {}

impl From<DenseI8DoryError> for HiddenPackedEoError {
    fn from(error: DenseI8DoryError) -> Self {
        Self::Dory(error)
    }
}

impl From<HiddenRelationError> for HiddenPackedEoError {
    fn from(error: HiddenRelationError) -> Self {
        Self::Relation(error)
    }
}

#[derive(Debug)]
pub enum HiddenPackedEoProverError<SourceError> {
    Streaming(StreamingEoError<SourceError>),
    Protocol(HiddenPackedEoError),
}

impl<SourceError: Display> Display for HiddenPackedEoProverError<SourceError> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Streaming(error) => Display::fmt(error, formatter),
            Self::Protocol(error) => Display::fmt(error, formatter),
        }
    }
}

impl<SourceError> std::error::Error for HiddenPackedEoProverError<SourceError> where
    SourceError: std::error::Error + 'static
{
}

struct PublicChallengeDriver {
    transcript: Blake2bTranscript<JoltFr>,
    packing_stride: usize,
    packed_positive: Vec<Fr>,
    packed_negative: Vec<Fr>,
    binary_rounds: Vec<(Fr, Fr)>,
    a_reduced: Option<Fr>,
    b_reduced: Option<Fr>,
    challenges: Vec<Fr>,
    challenges_inv: Vec<Fr>,
}

impl PublicChallengeDriver {
    fn new(
        logical_len: usize,
        packing_stride: usize,
        bit_width: usize,
        claimed: Fr,
        matrix_commitment: &DoryCommitment,
        outer_binding: &[u8],
    ) -> Self {
        Self::with_transcript(
            packing_stride,
            packed_eo_transcript(
                logical_len,
                packing_stride,
                bit_width,
                claimed,
                matrix_commitment,
                outer_binding,
            ),
        )
    }

    fn with_transcript(packing_stride: usize, transcript: Blake2bTranscript<JoltFr>) -> Self {
        Self {
            transcript,
            packing_stride,
            packed_positive: Vec::new(),
            packed_negative: Vec::new(),
            binary_rounds: Vec::new(),
            a_reduced: None,
            b_reduced: None,
            challenges: Vec::new(),
            challenges_inv: Vec::new(),
        }
    }

    fn sample_challenge(&mut self) -> Result<(Fr, Fr), String> {
        let challenge = challenge_nonzero(&mut self.transcript);
        let challenge_ark = Fr::from(challenge);
        let challenge_inv = challenge_ark
            .inverse()
            .ok_or_else(|| "transcript returned zero challenge".to_owned())?;
        Ok((challenge_ark, challenge_inv))
    }

    fn into_output(self) -> Result<PublicChallengeOutput, String> {
        Ok(PublicChallengeOutput {
            transcript_state: self.transcript.state(),
            packed_positive: self.packed_positive,
            packed_negative: self.packed_negative,
            binary_rounds: self.binary_rounds,
            a_reduced: self
                .a_reduced
                .ok_or_else(|| "missing terminal BVA scalar".to_owned())?,
            b_reduced: self
                .b_reduced
                .ok_or_else(|| "missing terminal TBV scalar".to_owned())?,
            challenges: self.challenges,
            challenges_inv: self.challenges_inv,
        })
    }
}

struct PublicChallengeOutput {
    transcript_state: [u8; 32],
    packed_positive: Vec<Fr>,
    packed_negative: Vec<Fr>,
    binary_rounds: Vec<(Fr, Fr)>,
    a_reduced: Fr,
    b_reduced: Fr,
    challenges: Vec<Fr>,
    challenges_inv: Vec<Fr>,
}

impl PackedEoChallengeTranscript<Fr> for PublicChallengeDriver {
    fn packed_challenge(&mut self, positive: &[Fr], negative: &[Fr]) -> Result<Fr, String> {
        if !self.packed_positive.is_empty() || positive.len() != negative.len() {
            return Err("packed cross terms were sent more than once".to_owned());
        }
        self.packed_positive.extend_from_slice(positive);
        self.packed_negative.extend_from_slice(negative);
        append_field_values(&mut self.transcript, b"packed_positive", positive);
        append_field_values(&mut self.transcript, b"packed_negative", negative);
        let (challenge, challenge_inv) = self.sample_challenge()?;
        let mut power = challenge;
        let mut inverse_power = challenge_inv;
        for _ in 0..self.packing_stride.ilog2() {
            self.challenges.push(power);
            self.challenges_inv.push(inverse_power);
            power.square_in_place();
            inverse_power.square_in_place();
        }
        Ok(challenge)
    }

    fn binary_challenge(&mut self, round: usize, iota: Fr, rho: Fr) -> Result<Fr, String> {
        if round != self.binary_rounds.len() {
            return Err("binary rounds were sent out of order".to_owned());
        }
        self.transcript
            .append_labeled(b"binary_round", &U64Word(round as u64));
        self.transcript.append_labeled(b"iota", &JoltFr::from(iota));
        self.transcript.append_labeled(b"rho", &JoltFr::from(rho));
        self.binary_rounds.push((iota, rho));
        let (challenge, challenge_inv) = self.sample_challenge()?;
        self.challenges.push(challenge);
        self.challenges_inv.push(challenge_inv);
        Ok(challenge)
    }

    fn bind_reduced(&mut self, a_reduced: Fr, b_reduced: Fr) -> Result<(), String> {
        if self.a_reduced.is_some() || self.b_reduced.is_some() {
            return Err("terminal scalars were sent more than once".to_owned());
        }
        self.transcript
            .append_labeled(b"a_reduced", &JoltFr::from(a_reduced));
        self.transcript
            .append_labeled(b"b_reduced", &JoltFr::from(b_reduced));
        self.a_reduced = Some(a_reduced);
        self.b_reduced = Some(b_reduced);
        Ok(())
    }
}

/// Reduce the public Packed EO claim without opening the dense BVA commitment.
#[allow(clippy::too_many_arguments)]
pub fn prove_hidden_packed_i8_reduction<S: DenseI8BvaSource>(
    parameters: &DoryProverParameters,
    source: &mut S,
    tbv: &TensorBasisVector<Fr>,
    claimed: Fr,
    bit_width: usize,
    matrix_commitment: &DoryCommitment,
    outer_binding: &[u8],
    _rng: &mut impl RngCore,
) -> Result<
    (
        HiddenPackedEoProof,
        HiddenPackedEoProverState,
        PaddingWorkCounters,
        StreamingEoProfile,
    ),
    HiddenPackedEoProverError<S::Error>,
> {
    prove_hidden_packed_i8_reduction_inner(
        parameters,
        source,
        tbv,
        claimed,
        bit_width,
        matrix_commitment,
        outer_binding,
    )
}

#[allow(clippy::too_many_arguments)]
fn prove_hidden_packed_i8_reduction_inner<S: DenseI8BvaSource>(
    parameters: &DoryProverParameters,
    source: &mut S,
    tbv: &TensorBasisVector<Fr>,
    claimed: Fr,
    bit_width: usize,
    matrix_commitment: &DoryCommitment,
    outer_binding: &[u8],
) -> Result<
    (
        HiddenPackedEoProof,
        HiddenPackedEoProverState,
        PaddingWorkCounters,
        StreamingEoProfile,
    ),
    HiddenPackedEoProverError<S::Error>,
> {
    let logical_len = source.layout().logical_len();
    let packing_stride = packing_stride_for_bit_width::<Fr>(logical_len, bit_width)
        .map_err(|_| HiddenPackedEoProverError::Protocol(HiddenPackedEoError::Shape))?;
    let mut driver = PublicChallengeDriver::new(
        logical_len,
        packing_stride,
        bit_width,
        claimed,
        matrix_commitment,
        outer_binding,
    );
    let (private_fold, counters, profile) =
        PackedEoLiteBulletProof::prove_i8_streaming_with_commitment_transcript(
            source,
            tbv,
            claimed,
            bit_width,
            &mut driver,
        )
        .map_err(HiddenPackedEoProverError::Streaming)?;
    let output = driver
        .into_output()
        .map_err(|_| HiddenPackedEoProverError::Protocol(HiddenPackedEoError::Shape))?;
    let transcript_state = output.transcript_state;
    if output.a_reduced != private_fold.a_reduced || output.b_reduced != private_fold.b_reduced {
        return Err(HiddenPackedEoProverError::Protocol(
            HiddenPackedEoError::Shape,
        ));
    }
    verify_public_scalar_identity(claimed, packing_stride, &output.challenges, &private_fold)
        .map_err(HiddenPackedEoProverError::Protocol)?;

    let a_reduced = public_scalar_witness(parameters, output.a_reduced);
    let b_reduced = public_scalar_witness(parameters, output.b_reduced);

    Ok((
        HiddenPackedEoProof {
            packing_stride,
            packed_positive: output
                .packed_positive
                .iter()
                .copied()
                .map(JoltFr::from)
                .collect(),
            packed_negative: output
                .packed_negative
                .iter()
                .copied()
                .map(JoltFr::from)
                .collect(),
            binary_rounds: output
                .binary_rounds
                .iter()
                .map(|&(iota, rho)| (JoltFr::from(iota), JoltFr::from(rho)))
                .collect(),
            a_reduced: JoltFr::from(output.a_reduced),
            b_reduced: JoltFr::from(output.b_reduced),
        },
        HiddenPackedEoProverState {
            a_reduced,
            b_reduced,
            challenges: output.challenges,
            challenges_inv: output.challenges_inv,
            transcript_state,
        },
        counters,
        profile,
    ))
}

/// Open the registered dense BVA at the Packed EO reduction point.
pub fn prove_hidden_bva_opening<S: DenseI8BvaSource>(
    parameters: &DoryProverParameters,
    source: &S,
    committed: &CommittedDoryBva,
    packed: &HiddenPackedEoProverState,
    rng: &mut impl RngCore,
) -> Result<HiddenBvaOpeningProof, HiddenPackedEoError> {
    Ok(prove_hidden_bva_opening_profiled(parameters, source, committed, packed, rng)?.0)
}

/// Open the registered dense BVA and report mutually exclusive stage timings.
pub fn prove_hidden_bva_opening_profiled<S: DenseI8BvaSource>(
    parameters: &DoryProverParameters,
    source: &S,
    committed: &CommittedDoryBva,
    packed: &HiddenPackedEoProverState,
    rng: &mut impl RngCore,
) -> Result<(HiddenBvaOpeningProof, HiddenBvaOpeningProverProfile), HiddenPackedEoError> {
    let total_started = Instant::now();

    let point_preparation_started = Instant::now();
    let monomial_point = packed
        .challenges_inv
        .iter()
        .copied()
        .map(JoltFr::from)
        .collect::<Vec<_>>();
    let point_preparation_seconds = point_preparation_started.elapsed().as_secs_f64();

    let dense_preparation_started = Instant::now();
    let dense = dense_i8_polynomial(source)?;
    let dense_preparation_seconds = dense_preparation_started.elapsed().as_secs_f64();

    let transcript_preparation_started = Instant::now();
    let mut dory_transcript = Blake2bTranscript::<JoltFr>::new(b"pc_hidden_peo_dory");
    let transcript_preparation_seconds = transcript_preparation_started.elapsed().as_secs_f64();

    let dory_open_started = Instant::now();
    let (dory_opening, evaluation_commitment, evaluation_blind) = open_hidden_dense_i8_zk(
        &dense,
        &committed.commitment,
        committed.hint.clone(),
        &monomial_point,
        JoltFr::from(packed.a_reduced.value),
        &packed.transcript_state,
        &parameters.pcs,
        &mut dory_transcript,
    )?;
    let dory_open_seconds = dory_open_started.elapsed().as_secs_f64();

    let scalar_binding_started = Instant::now();
    let normalized = normalize_monomial_mle_claim(&packed.challenges_inv, packed.a_reduced.value)
        .map_err(|_| HiddenPackedEoError::Shape)?;
    let normalized_a = HiddenScalarWitness {
        value: normalized.value,
        commitment: evaluation_commitment,
        blind: evaluation_blind,
    };
    let evaluation_expression = RelationExpression::table(1)
        .mul(RelationExpression::constant(normalized.scale))
        .sub(RelationExpression::table(0));
    let evaluation_binding = prove_hidden_scalar_relation(
        &packed.transcript_state,
        parameters,
        &[packed.a_reduced.clone(), normalized_a],
        &evaluation_expression,
        rng,
    )?;
    let scalar_binding_seconds = scalar_binding_started.elapsed().as_secs_f64();

    let proof = HiddenBvaOpeningProof {
        dory_opening,
        evaluation_binding,
    };
    let profile = HiddenBvaOpeningProverProfile {
        point_preparation_seconds,
        dense_preparation_seconds,
        transcript_preparation_seconds,
        dory_open_seconds,
        scalar_binding_seconds,
        total_seconds: total_started.elapsed().as_secs_f64(),
    };
    Ok((proof, profile))
}

/// Reduce the dense-BVA opening to one hidden standard-MLE claim without
/// creating a Dory proof.
pub fn prepare_hidden_bva_opening(
    parameters: &DoryProverParameters,
    packed: &HiddenPackedEoProverState,
    rng: &mut impl RngCore,
) -> Result<(HiddenBvaOpeningReductionProof, PendingHiddenBvaOpening), HiddenPackedEoError> {
    let normalized = normalize_monomial_mle_claim(&packed.challenges_inv, packed.a_reduced.value)
        .map_err(|_| HiddenPackedEoError::Shape)?;
    let normalized_evaluation = HiddenScalarWitness::commit(parameters, normalized.value, rng);
    let evaluation_expression = RelationExpression::table(1)
        .mul(RelationExpression::constant(normalized.scale))
        .sub(RelationExpression::table(0));
    let evaluation_binding = prove_hidden_scalar_relation(
        &packed.transcript_state,
        parameters,
        &[packed.a_reduced.clone(), normalized_evaluation.clone()],
        &evaluation_expression,
        rng,
    )?;
    Ok((
        HiddenBvaOpeningReductionProof {
            normalized_evaluation_commitment: normalized_evaluation.commitment,
            evaluation_binding,
        },
        PendingHiddenBvaOpening {
            point_msb: normalized.point_msb,
            normalized_evaluation,
        },
    ))
}

/// Verify the public Packed EO identity without opening the dense BVA.
#[allow(clippy::too_many_arguments)]
pub fn verify_hidden_packed_i8_reduction(
    parameters: &DoryVerifierParameters,
    matrix_commitment: &DoryCommitment,
    logical_len: usize,
    claimed: Fr,
    bit_width: usize,
    outer_binding: &[u8],
    proof: &HiddenPackedEoProof,
) -> Result<HiddenPackedEoVerification, HiddenPackedEoError> {
    let expected_stride = packing_stride_for_bit_width::<Fr>(logical_len, bit_width)
        .map_err(|_| HiddenPackedEoError::Shape)?;
    if proof.packing_stride != expected_stride
        || proof.packed_positive.len() != expected_stride - 1
        || proof.packed_negative.len() != expected_stride - 1
        || proof.binary_rounds.len() != (logical_len / expected_stride).ilog2() as usize
    {
        return Err(HiddenPackedEoError::Shape);
    }
    let replay = replay_public_transcript(
        logical_len,
        expected_stride,
        bit_width,
        claimed,
        matrix_commitment,
        outer_binding,
        proof,
    )?;
    let public_fold = public_fold_from_proof(proof);
    verify_public_scalar_identity(claimed, expected_stride, &replay.challenges, &public_fold)?;

    Ok(HiddenPackedEoVerification {
        a_reduced_commitment: public_scalar_commitment(
            &parameters.blindfold,
            Fr::from(proof.a_reduced),
        ),
        b_reduced_commitment: public_scalar_commitment(
            &parameters.blindfold,
            Fr::from(proof.b_reduced),
        ),
        challenges: replay.challenges,
        challenges_inv: replay.challenges_inv,
        transcript_state: replay.transcript_state,
    })
}

/// Verify the independent dense-BVA Dory opening emitted after Packed EO.
pub fn verify_hidden_bva_opening(
    parameters: &DoryVerifierParameters,
    matrix_commitment: &DoryCommitment,
    logical_len: usize,
    packed: &HiddenPackedEoVerification,
    proof: &HiddenBvaOpeningProof,
) -> Result<(), HiddenPackedEoError> {
    verify_hidden_bva_opening_profiled(parameters, matrix_commitment, logical_len, packed, proof)?;
    Ok(())
}

/// Verify the dense-BVA opening and report mutually exclusive stage timings.
pub fn verify_hidden_bva_opening_profiled(
    parameters: &DoryVerifierParameters,
    matrix_commitment: &DoryCommitment,
    logical_len: usize,
    packed: &HiddenPackedEoVerification,
    proof: &HiddenBvaOpeningProof,
) -> Result<HiddenBvaOpeningVerifierProfile, HiddenPackedEoError> {
    let total_started = Instant::now();

    let point_preparation_started = Instant::now();
    let monomial_point = packed
        .challenges_inv
        .iter()
        .copied()
        .map(JoltFr::from)
        .collect::<Vec<_>>();
    let point_preparation_seconds = point_preparation_started.elapsed().as_secs_f64();

    let transcript_preparation_started = Instant::now();
    let mut dory_transcript = Blake2bTranscript::<JoltFr>::new(b"pc_hidden_peo_dory");
    let transcript_preparation_seconds = transcript_preparation_started.elapsed().as_secs_f64();

    let dory_verify_started = Instant::now();
    let evaluation_commitment = verify_hidden_dense_i8_zk(
        matrix_commitment,
        &proof.dory_opening,
        logical_len,
        &monomial_point,
        &packed.transcript_state,
        &parameters.pcs,
        &mut dory_transcript,
    )?;
    let dory_verify_seconds = dory_verify_started.elapsed().as_secs_f64();

    let scalar_binding_started = Instant::now();
    let normalized = normalize_monomial_mle_claim(&packed.challenges_inv, Fr::one())
        .map_err(|_| HiddenPackedEoError::Shape)?;
    let evaluation_expression = RelationExpression::table(1)
        .mul(RelationExpression::constant(normalized.scale))
        .sub(RelationExpression::table(0));
    verify_hidden_scalar_relation(
        &packed.transcript_state,
        parameters,
        &[packed.a_reduced_commitment, evaluation_commitment],
        &evaluation_expression,
        &proof.evaluation_binding,
    )?;
    let scalar_binding_seconds = scalar_binding_started.elapsed().as_secs_f64();

    Ok(HiddenBvaOpeningVerifierProfile {
        point_preparation_seconds,
        transcript_preparation_seconds,
        dory_verify_seconds,
        scalar_binding_seconds,
        total_seconds: total_started.elapsed().as_secs_f64(),
    })
}

/// Verify the dense-BVA scalar binding and return its deferred PCS claim.
pub fn prepare_hidden_bva_opening_verification(
    parameters: &DoryVerifierParameters,
    packed: &HiddenPackedEoVerification,
    proof: &HiddenBvaOpeningReductionProof,
) -> Result<HiddenBvaOpeningReductionVerification, HiddenPackedEoError> {
    let normalized = normalize_monomial_mle_claim(&packed.challenges_inv, Fr::one())
        .map_err(|_| HiddenPackedEoError::Shape)?;
    let evaluation_expression = RelationExpression::table(1)
        .mul(RelationExpression::constant(normalized.scale))
        .sub(RelationExpression::table(0));
    verify_hidden_scalar_relation(
        &packed.transcript_state,
        parameters,
        &[
            packed.a_reduced_commitment,
            proof.normalized_evaluation_commitment,
        ],
        &evaluation_expression,
        &proof.evaluation_binding,
    )?;
    Ok(HiddenBvaOpeningReductionVerification {
        point_msb: normalized.point_msb,
        normalized_evaluation_commitment: proof.normalized_evaluation_commitment,
    })
}

struct TranscriptReplay {
    transcript_state: [u8; 32],
    challenges: Vec<Fr>,
    challenges_inv: Vec<Fr>,
}

#[allow(clippy::too_many_arguments)]
fn replay_public_transcript(
    logical_len: usize,
    packing_stride: usize,
    bit_width: usize,
    claimed: Fr,
    matrix_commitment: &DoryCommitment,
    outer_binding: &[u8],
    proof: &HiddenPackedEoProof,
) -> Result<TranscriptReplay, HiddenPackedEoError> {
    let mut transcript = packed_eo_transcript(
        logical_len,
        packing_stride,
        bit_width,
        claimed,
        matrix_commitment,
        outer_binding,
    );
    append_jolt_field_values(&mut transcript, b"packed_positive", &proof.packed_positive);
    append_jolt_field_values(&mut transcript, b"packed_negative", &proof.packed_negative);
    let packed_challenge = Fr::from(challenge_nonzero(&mut transcript));
    let packed_challenge_inv = packed_challenge
        .inverse()
        .ok_or(HiddenPackedEoError::ZeroChallenge)?;
    let stride_log = packing_stride.ilog2() as usize;
    let mut challenges = Vec::with_capacity(logical_len.ilog2() as usize);
    let mut challenges_inv = Vec::with_capacity(logical_len.ilog2() as usize);
    let mut power = packed_challenge;
    let mut inverse_power = packed_challenge_inv;
    for _ in 0..stride_log {
        challenges.push(power);
        challenges_inv.push(inverse_power);
        power.square_in_place();
        inverse_power.square_in_place();
    }
    for (round, &(iota, rho)) in proof.binary_rounds.iter().enumerate() {
        transcript.append_labeled(b"binary_round", &U64Word(round as u64));
        transcript.append_labeled(b"iota", &iota);
        transcript.append_labeled(b"rho", &rho);
        let challenge = Fr::from(challenge_nonzero(&mut transcript));
        let challenge_inv = challenge
            .inverse()
            .ok_or(HiddenPackedEoError::ZeroChallenge)?;
        challenges.push(challenge);
        challenges_inv.push(challenge_inv);
    }
    transcript.append_labeled(b"a_reduced", &proof.a_reduced);
    transcript.append_labeled(b"b_reduced", &proof.b_reduced);
    Ok(TranscriptReplay {
        transcript_state: transcript.state(),
        challenges,
        challenges_inv,
    })
}

fn verify_public_scalar_identity(
    claimed: Fr,
    packing_stride: usize,
    challenges: &[Fr],
    proof: &PackedEoLiteBulletProof<Fr>,
) -> Result<(), HiddenPackedEoError> {
    let stride_log = packing_stride.ilog2() as usize;
    if proof.packing_stride != packing_stride
        || proof.packed_positive.len() != packing_stride - 1
        || proof.packed_negative.len() != packing_stride - 1
        || challenges.len() != stride_log + proof.binary_rounds.len()
    {
        return Err(HiddenPackedEoError::Shape);
    }
    let mut reduced_claim = claimed;
    if packing_stride > 1 {
        let packed_challenge = challenges[0];
        let packed_challenge_inv = packed_challenge
            .inverse()
            .ok_or(HiddenPackedEoError::ZeroChallenge)?;
        let mut positive_power = packed_challenge;
        let mut negative_power = packed_challenge_inv;
        for (&positive, &negative) in proof.packed_positive.iter().zip(&proof.packed_negative) {
            reduced_claim += positive * positive_power + negative * negative_power;
            positive_power *= packed_challenge;
            negative_power *= packed_challenge_inv;
        }
    }
    for (&(iota, rho), &challenge) in proof.binary_rounds.iter().zip(&challenges[stride_log..]) {
        let challenge_inv = challenge
            .inverse()
            .ok_or(HiddenPackedEoError::ZeroChallenge)?;
        reduced_claim += iota * challenge + rho * challenge_inv;
    }
    if proof.a_reduced * proof.b_reduced != reduced_claim {
        return Err(HiddenPackedEoError::Shape);
    }
    Ok(())
}

fn public_fold_from_proof(proof: &HiddenPackedEoProof) -> PackedEoLiteBulletProof<Fr> {
    PackedEoLiteBulletProof {
        packing_stride: proof.packing_stride,
        packed_positive: proof
            .packed_positive
            .iter()
            .copied()
            .map(Fr::from)
            .collect(),
        packed_negative: proof
            .packed_negative
            .iter()
            .copied()
            .map(Fr::from)
            .collect(),
        binary_rounds: proof
            .binary_rounds
            .iter()
            .map(|&(iota, rho)| (Fr::from(iota), Fr::from(rho)))
            .collect(),
        a_reduced: Fr::from(proof.a_reduced),
        b_reduced: Fr::from(proof.b_reduced),
    }
}

fn public_scalar_commitment(setup: &PedersenSetup<Bn254G1>, value: Fr) -> Bn254G1 {
    Pedersen::<Bn254G1>::commit(setup, &[JoltFr::from(value)], &JoltFr::zero())
}

fn public_scalar_witness(parameters: &DoryProverParameters, value: Fr) -> HiddenScalarWitness {
    HiddenScalarWitness {
        value,
        commitment: public_scalar_commitment(&parameters.blindfold, value),
        blind: JoltFr::zero(),
    }
}

#[allow(clippy::too_many_arguments)]
fn packed_eo_transcript(
    logical_len: usize,
    packing_stride: usize,
    bit_width: usize,
    claimed: Fr,
    matrix_commitment: &DoryCommitment,
    outer_binding: &[u8],
) -> Blake2bTranscript<JoltFr> {
    let mut transcript = Blake2bTranscript::<JoltFr>::new(b"pc_packed_eo_public_v1");
    transcript.append(&Label(b"packed_eo_statement"));
    transcript.append_labeled(b"logical_len", &U64Word(logical_len as u64));
    transcript.append_labeled(b"packing_stride", &U64Word(packing_stride as u64));
    transcript.append_labeled(b"bit_width", &U64Word(bit_width as u64));
    transcript.append_labeled(b"claimed", &JoltFr::from(claimed));
    transcript.append_labeled(b"matrix_commitment", matrix_commitment);
    transcript.append(&LabelWithCount(
        b"outer_binding",
        outer_binding.len() as u64,
    ));
    transcript.append_bytes(outer_binding);
    transcript
}

fn append_field_values(
    transcript: &mut Blake2bTranscript<JoltFr>,
    label: &'static [u8],
    values: &[Fr],
) {
    transcript.append(&LabelWithCount(label, values.len() as u64));
    for &value in values {
        transcript.append_labeled(b"field_element", &JoltFr::from(value));
    }
}

fn append_jolt_field_values(
    transcript: &mut Blake2bTranscript<JoltFr>,
    label: &'static [u8],
    values: &[JoltFr],
) {
    transcript.append(&LabelWithCount(label, values.len() as u64));
    for value in values {
        transcript.append_labeled(b"field_element", value);
    }
}

fn challenge_nonzero(transcript: &mut Blake2bTranscript<JoltFr>) -> JoltFr {
    let mut retry = 0_u64;
    loop {
        let challenge = transcript.challenge();
        if !challenge.is_zero() {
            return challenge;
        }
        retry += 1;
        transcript.append_labeled(b"challenge_retry", &U64Word(retry));
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use bincode::Options;

    use crate::{
        bucketed_ip::{BvaI8Source, BvaLayout},
        jolt_dory::{commit_bva_i8_streaming, setup_dory_parameters},
    };

    use super::*;

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

        fn indexed_vectors(&self) -> Option<Vec<&[i8]>> {
            Some(self.vectors.iter().map(Vec::as_slice).collect())
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

    fn fixture() -> (ByteSource, TensorBasisVector<Fr>, Fr) {
        let mut source = ByteSource::new(vec![
            (0..64).map(|index| (index % 23) as i8 - 11).collect(),
            (0..32).map(|index| 9 - (index % 19) as i8).collect(),
        ]);
        let point = (2_u64..9).map(Fr::from).collect::<Vec<_>>();
        let tbv = TensorBasisVector::global_tensor(&source.layout, &point).unwrap();
        let claimed = tbv.inner_product_i8_streaming(&mut source).unwrap();
        (source, tbv, claimed)
    }

    #[test]
    fn hidden_packed_eo_round_trip_sends_reduction_scalars_as_field_elements() {
        let (mut source, tbv, claimed) = fixture();
        let (prover, verifier) = setup_dory_parameters(7);
        let committed = commit_bva_i8_streaming(&prover, &source, 8).unwrap();
        let matrix_commitment = committed.commitment.clone();
        let mut rng = StdRng::seed_from_u64(0x4849_4444_454e);
        let (proof, claims, _, _) = prove_hidden_packed_i8_reduction(
            &prover,
            &mut source,
            &tbv,
            claimed,
            8,
            &matrix_commitment,
            b"section-6-layout",
            &mut rng,
        )
        .unwrap();
        let (bva_opening, prover_profile) =
            prove_hidden_bva_opening_profiled(&prover, &source, &committed, &claims, &mut rng)
                .unwrap();
        assert!(prover_profile.accounted_seconds() <= prover_profile.total_seconds + 1e-9);
        let verification = verify_hidden_packed_i8_reduction(
            &verifier,
            &matrix_commitment,
            source.layout.logical_len(),
            claimed,
            8,
            b"section-6-layout",
            &proof,
        )
        .unwrap();
        let verifier_profile = verify_hidden_bva_opening_profiled(
            &verifier,
            &matrix_commitment,
            source.layout.logical_len(),
            &verification,
            &bva_opening,
        )
        .unwrap();
        assert!(verifier_profile.accounted_seconds() <= verifier_profile.total_seconds + 1e-9);
        let codec = bincode::DefaultOptions::new().with_fixint_encoding();
        let packed_bytes = codec.serialize(&proof).unwrap();
        let scalar_bytes = codec.serialize(&JoltFr::zero()).unwrap().len();
        assert_eq!(
            packed_bytes.len(),
            4 * std::mem::size_of::<u64>() + proof.field_element_count() * scalar_bytes
        );
        let split_bytes = codec.serialize(&(&proof, &bva_opening)).unwrap();
        let decoded: (HiddenPackedEoProof, HiddenBvaOpeningProof) = codec
            .reject_trailing_bytes()
            .deserialize(&split_bytes)
            .unwrap();
        assert_eq!(decoded, (proof.clone(), bva_opening));
        assert_eq!(proof.a_reduced, JoltFr::from(claims.a_reduced.value));
        assert_eq!(proof.b_reduced, JoltFr::from(claims.b_reduced.value));
        assert_eq!(
            verification.a_reduced_commitment,
            claims.a_reduced.commitment
        );
        assert_eq!(
            verification.b_reduced_commitment,
            claims.b_reduced.commitment
        );
        assert_eq!(verification.challenges, claims.challenges);
        assert_eq!(
            proof.binary_rounds.len(),
            (source.layout.logical_len() / proof.packing_stride).ilog2() as usize
        );
        assert_eq!(
            proof.field_element_count(),
            2 * proof.packing_stride - 2 + 2 * proof.binary_rounds.len() + 2
        );
    }

    #[test]
    fn hidden_packed_eo_rejects_field_and_statement_tampering() {
        let (mut source, tbv, claimed) = fixture();
        let (prover, verifier) = setup_dory_parameters(7);
        let committed = commit_bva_i8_streaming(&prover, &source, 8).unwrap();
        let matrix_commitment = committed.commitment.clone();
        let mut rng = StdRng::seed_from_u64(0x5441_4d50_4552);
        let (proof, _, _, _) = prove_hidden_packed_i8_reduction(
            &prover,
            &mut source,
            &tbv,
            claimed,
            8,
            &matrix_commitment,
            b"section-6-layout",
            &mut rng,
        )
        .unwrap();

        let mut tampered = proof.clone();
        tampered.packed_positive[0] += JoltFr::one();
        assert!(verify_hidden_packed_i8_reduction(
            &verifier,
            &matrix_commitment,
            source.layout.logical_len(),
            claimed,
            8,
            b"section-6-layout",
            &tampered,
        )
        .is_err());

        let mut tampered = proof.clone();
        tampered.packed_negative[0] += JoltFr::one();
        assert!(verify_hidden_packed_i8_reduction(
            &verifier,
            &matrix_commitment,
            source.layout.logical_len(),
            claimed,
            8,
            b"section-6-layout",
            &tampered,
        )
        .is_err());

        let mut tampered = proof.clone();
        tampered.binary_rounds[0].0 += JoltFr::one();
        assert!(verify_hidden_packed_i8_reduction(
            &verifier,
            &matrix_commitment,
            source.layout.logical_len(),
            claimed,
            8,
            b"section-6-layout",
            &tampered,
        )
        .is_err());
        assert!(verify_hidden_packed_i8_reduction(
            &verifier,
            &matrix_commitment,
            source.layout.logical_len(),
            claimed + Fr::one(),
            8,
            b"section-6-layout",
            &proof,
        )
        .is_err());
        assert!(verify_hidden_packed_i8_reduction(
            &verifier,
            &matrix_commitment,
            source.layout.logical_len(),
            claimed,
            8,
            b"different-layout",
            &proof,
        )
        .is_err());

        let mut tampered_terminal = proof.clone();
        tampered_terminal.a_reduced += JoltFr::one();
        assert!(verify_hidden_packed_i8_reduction(
            &verifier,
            &matrix_commitment,
            source.layout.logical_len(),
            claimed,
            8,
            b"section-6-layout",
            &tampered_terminal,
        )
        .is_err());

        let mut tampered_terminal = proof;
        tampered_terminal.b_reduced += JoltFr::one();
        assert!(verify_hidden_packed_i8_reduction(
            &verifier,
            &matrix_commitment,
            source.layout.logical_len(),
            claimed,
            8,
            b"section-6-layout",
            &tampered_terminal,
        )
        .is_err());
    }

    #[test]
    fn hidden_bva_opening_cannot_be_swapped_between_packed_reductions() {
        let (mut source, tbv, claimed) = fixture();
        let (prover, verifier) = setup_dory_parameters(7);
        let committed = commit_bva_i8_streaming(&prover, &source, 8).unwrap();
        let matrix_commitment = committed.commitment.clone();
        let mut rng = StdRng::seed_from_u64(0x5357_4150_5f42_5641);

        let (first_proof, first_state, _, _) = prove_hidden_packed_i8_reduction(
            &prover,
            &mut source,
            &tbv,
            claimed,
            8,
            &matrix_commitment,
            b"section-6-layout",
            &mut rng,
        )
        .unwrap();
        let first_verification = verify_hidden_packed_i8_reduction(
            &verifier,
            &matrix_commitment,
            source.layout.logical_len(),
            claimed,
            8,
            b"section-6-layout",
            &first_proof,
        )
        .unwrap();
        let _first_bva =
            prove_hidden_bva_opening(&prover, &source, &committed, &first_state, &mut rng).unwrap();

        let (_second_proof, second_state, _, _) = prove_hidden_packed_i8_reduction(
            &prover,
            &mut source,
            &tbv,
            claimed,
            8,
            &matrix_commitment,
            b"section-6-layout-2",
            &mut rng,
        )
        .unwrap();
        let second_bva =
            prove_hidden_bva_opening(&prover, &source, &committed, &second_state, &mut rng)
                .unwrap();

        assert!(verify_hidden_bva_opening(
            &verifier,
            &matrix_commitment,
            source.layout.logical_len(),
            &first_verification,
            &second_bva,
        )
        .is_err());
    }
}
