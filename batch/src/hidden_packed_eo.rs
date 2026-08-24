//! Hidden Packed EO-LiteBullet scalar transcript for Section 6.
//!
//! Cross terms, binary-round scalars, and terminal fold scalars are committed
//! in G1 before their Fiat--Shamir challenges are sampled. BlindFold checks the
//! terminal scalar identity. The resulting continuation state feeds two
//! independent hidden stages: a Dory opening of the registered dense BVA and
//! the final Section 6 TBV consistency relation.

use std::{
    fmt::{Display, Formatter},
    time::Instant,
};

use ark_bn254::Fr;
use ark_ff::{Field, One, Zero};
use ark_std::rand::RngCore;
use jolt_crypto::Bn254G1;
use jolt_dory::DoryCommitment;
use jolt_field::Fr as JoltFr;
use jolt_transcript::domain::{Label, LabelWithCount, U64Word};
use jolt_transcript::{Blake2bTranscript, Transcript};
use serde::{Deserialize, Serialize};

use crate::{
    bucketed_ip::{
        packing_stride_for_bit_width, BvaInteger, BvaSource, PackedEoChallengeTranscript,
        PackedEoLiteBulletProof, PaddingWorkCounters, StreamingEoError, StreamingEoProfile,
        TensorBasisVector,
    },
    hidden_relation::{
        prove_hidden_scalar_relation, verify_hidden_scalar_relation, HiddenRelationError,
        HiddenScalarRelationProof, HiddenScalarWitness, RelationExpression,
    },
    jolt_dory::{
        dense_i16_polynomial, dense_i8_polynomial, normalize_monomial_mle_claim, CommittedDoryBva,
        DenseI16BvaSource, DenseI8BvaSource, DoryProverParameters, DoryVerifierParameters,
    },
    jolt_dory_adapter::{
        open_hidden_dense_i16_zk, open_hidden_dense_i8_zk, verify_hidden_dense_i16_zk,
        verify_hidden_dense_i8_zk, DenseI8DoryError, HiddenDenseI8DoryProof,
    },
};

/// Public proof containing commitments, never the folded field scalars.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenPackedEoProof {
    pub packing_stride: usize,
    pub packed_positive_commitments: Vec<Bn254G1>,
    pub packed_negative_commitments: Vec<Bn254G1>,
    pub binary_round_commitments: Vec<(Bn254G1, Bn254G1)>,
    pub a_reduced_commitment: Bn254G1,
    pub b_reduced_commitment: Bn254G1,
    pub scalar_identity: HiddenScalarRelationProof,
}

impl HiddenPackedEoProof {
    /// Number of scalar commitments sent by the Packed EO reduction.
    pub fn scalar_commitment_count(&self) -> usize {
        self.packed_positive_commitments.len()
            + self.packed_negative_commitments.len()
            + 2 * self.binary_round_commitments.len()
            + 2
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

struct HidingChallengeDriver<'a, R> {
    parameters: &'a DoryProverParameters,
    rng: &'a mut R,
    transcript: Blake2bTranscript<JoltFr>,
    packing_stride: usize,
    packed_positive: Vec<HiddenScalarWitness>,
    packed_negative: Vec<HiddenScalarWitness>,
    binary_rounds: Vec<(HiddenScalarWitness, HiddenScalarWitness)>,
    a_reduced: Option<HiddenScalarWitness>,
    b_reduced: Option<HiddenScalarWitness>,
    challenges: Vec<Fr>,
    challenges_inv: Vec<Fr>,
}

impl<'a, R: RngCore> HidingChallengeDriver<'a, R> {
    fn new(
        parameters: &'a DoryProverParameters,
        rng: &'a mut R,
        logical_len: usize,
        packing_stride: usize,
        bit_width: usize,
        claimed: Fr,
        matrix_commitment: &DoryCommitment,
        outer_binding: &[u8],
    ) -> Self {
        Self::with_transcript(
            parameters,
            rng,
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

    fn with_transcript(
        parameters: &'a DoryProverParameters,
        rng: &'a mut R,
        packing_stride: usize,
        transcript: Blake2bTranscript<JoltFr>,
    ) -> Self {
        Self {
            parameters,
            rng,
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

    fn commit_scalar(&mut self, value: Fr) -> HiddenScalarWitness {
        HiddenScalarWitness::commit(self.parameters, value, self.rng)
    }

    fn sample_challenge(&mut self) -> Result<(Fr, Fr), String> {
        let challenge = challenge_nonzero(&mut self.transcript);
        let challenge_ark = Fr::from(challenge);
        let challenge_inv = challenge_ark
            .inverse()
            .ok_or_else(|| "transcript returned zero challenge".to_owned())?;
        Ok((challenge_ark, challenge_inv))
    }

    fn into_output(self) -> Result<HidingChallengeOutput, String> {
        Ok(HidingChallengeOutput {
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

struct HidingChallengeOutput {
    transcript_state: [u8; 32],
    packed_positive: Vec<HiddenScalarWitness>,
    packed_negative: Vec<HiddenScalarWitness>,
    binary_rounds: Vec<(HiddenScalarWitness, HiddenScalarWitness)>,
    a_reduced: HiddenScalarWitness,
    b_reduced: HiddenScalarWitness,
    challenges: Vec<Fr>,
    challenges_inv: Vec<Fr>,
}

impl<R: RngCore> PackedEoChallengeTranscript<Fr> for HidingChallengeDriver<'_, R> {
    fn packed_challenge(&mut self, positive: &[Fr], negative: &[Fr]) -> Result<Fr, String> {
        if !self.packed_positive.is_empty() || positive.len() != negative.len() {
            return Err("packed cross terms were committed more than once".to_owned());
        }
        for &value in positive {
            let witness = self.commit_scalar(value);
            self.packed_positive.push(witness);
        }
        for &value in negative {
            let witness = self.commit_scalar(value);
            self.packed_negative.push(witness);
        }
        append_commitments(
            &mut self.transcript,
            b"packed_positive",
            self.packed_positive
                .iter()
                .map(|witness| &witness.commitment),
        );
        append_commitments(
            &mut self.transcript,
            b"packed_negative",
            self.packed_negative
                .iter()
                .map(|witness| &witness.commitment),
        );
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
            return Err("binary rounds were committed out of order".to_owned());
        }
        let iota = self.commit_scalar(iota);
        let rho = self.commit_scalar(rho);
        self.transcript
            .append_labeled(b"binary_round", &U64Word(round as u64));
        self.transcript
            .append_labeled(b"iota_commitment", &iota.commitment);
        self.transcript
            .append_labeled(b"rho_commitment", &rho.commitment);
        self.binary_rounds.push((iota, rho));
        let (challenge, challenge_inv) = self.sample_challenge()?;
        self.challenges.push(challenge);
        self.challenges_inv.push(challenge_inv);
        Ok(challenge)
    }

    fn bind_reduced(&mut self, a_reduced: Fr, b_reduced: Fr) -> Result<(), String> {
        if self.a_reduced.is_some() || self.b_reduced.is_some() {
            return Err("terminal scalars were committed more than once".to_owned());
        }
        let a_reduced = self.commit_scalar(a_reduced);
        let b_reduced = self.commit_scalar(b_reduced);
        self.transcript
            .append_labeled(b"a_reduced_commitment", &a_reduced.commitment);
        self.transcript
            .append_labeled(b"b_reduced_commitment", &b_reduced.commitment);
        self.a_reduced = Some(a_reduced);
        self.b_reduced = Some(b_reduced);
        Ok(())
    }
}

/// Reduce a hidden Packed EO claim without opening the dense BVA commitment.
#[allow(clippy::too_many_arguments)]
pub fn prove_hidden_packed_i8_reduction<S: DenseI8BvaSource>(
    parameters: &DoryProverParameters,
    source: &mut S,
    tbv: &TensorBasisVector<Fr>,
    claimed: Fr,
    bit_width: usize,
    matrix_commitment: &DoryCommitment,
    outer_binding: &[u8],
    rng: &mut impl RngCore,
) -> Result<
    (
        HiddenPackedEoProof,
        HiddenPackedEoProverState,
        PaddingWorkCounters,
        StreamingEoProfile,
    ),
    HiddenPackedEoProverError<S::Error>,
> {
    prove_hidden_packed_shortint_reduction::<i8, S>(
        parameters,
        source,
        tbv,
        claimed,
        bit_width,
        matrix_commitment,
        outer_binding,
        rng,
    )
}

/// Reduce a hidden Packed EO claim over the selected native short-integer type.
#[allow(clippy::too_many_arguments)]
pub fn prove_hidden_packed_shortint_reduction<Entry, S>(
    parameters: &DoryProverParameters,
    source: &mut S,
    tbv: &TensorBasisVector<Fr>,
    claimed: Fr,
    bit_width: usize,
    matrix_commitment: &DoryCommitment,
    outer_binding: &[u8],
    rng: &mut impl RngCore,
) -> Result<
    (
        HiddenPackedEoProof,
        HiddenPackedEoProverState,
        PaddingWorkCounters,
        StreamingEoProfile,
    ),
    HiddenPackedEoProverError<S::Error>,
>
where
    Entry: BvaInteger,
    S: BvaSource<Entry>,
{
    let logical_len = BvaSource::<Entry>::layout(source).logical_len();
    let packing_stride = packing_stride_for_bit_width::<Fr>(logical_len, bit_width)
        .map_err(|_| HiddenPackedEoProverError::Protocol(HiddenPackedEoError::Shape))?;
    let mut driver = HidingChallengeDriver::new(
        parameters,
        rng,
        logical_len,
        packing_stride,
        bit_width,
        claimed,
        matrix_commitment,
        outer_binding,
    );
    let (private_fold, counters, profile) =
        PackedEoLiteBulletProof::prove_shortint_streaming_with_commitment_transcript::<Entry, S>(
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
    if output.a_reduced.value != private_fold.a_reduced
        || output.b_reduced.value != private_fold.b_reduced
    {
        return Err(HiddenPackedEoProverError::Protocol(
            HiddenPackedEoError::Shape,
        ));
    }
    let packed_positive_commitments = output
        .packed_positive
        .iter()
        .map(|witness| witness.commitment)
        .collect::<Vec<_>>();
    let packed_negative_commitments = output
        .packed_negative
        .iter()
        .map(|witness| witness.commitment)
        .collect::<Vec<_>>();
    let binary_round_commitments = output
        .binary_rounds
        .iter()
        .map(|(iota, rho)| (iota.commitment, rho.commitment))
        .collect::<Vec<_>>();
    let a_reduced = output.a_reduced;
    let b_reduced = output.b_reduced;
    let mut identity_witnesses = Vec::with_capacity(private_fold.field_element_count());
    identity_witnesses.extend(output.packed_positive);
    identity_witnesses.extend(output.packed_negative);
    for (iota, rho) in output.binary_rounds {
        identity_witnesses.push(iota);
        identity_witnesses.push(rho);
    }
    identity_witnesses.push(a_reduced.clone());
    identity_witnesses.push(b_reduced.clone());
    let identity_expression = scalar_identity_expression(
        claimed,
        packing_stride,
        &output.challenges,
        identity_witnesses.len(),
    )
    .map_err(HiddenPackedEoProverError::Protocol)?;
    let scalar_identity = prove_hidden_scalar_relation(
        &transcript_state,
        parameters,
        &identity_witnesses,
        &identity_expression,
        rng,
    )
    .map_err(HiddenPackedEoError::from)
    .map_err(HiddenPackedEoProverError::Protocol)?;

    Ok((
        HiddenPackedEoProof {
            packing_stride,
            packed_positive_commitments,
            packed_negative_commitments,
            binary_round_commitments,
            a_reduced_commitment: a_reduced.commitment,
            b_reduced_commitment: b_reduced.commitment,
            scalar_identity,
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

/// Open a registered native-i16 BVA and report mutually exclusive timings.
pub fn prove_hidden_bva_i16_opening_profiled<S: DenseI16BvaSource>(
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
    let dense = dense_i16_polynomial(source)?;
    let dense_preparation_seconds = dense_preparation_started.elapsed().as_secs_f64();

    let transcript_preparation_started = Instant::now();
    let mut dory_transcript = Blake2bTranscript::<JoltFr>::new(b"pc_hidden_peo_dory");
    let transcript_preparation_seconds = transcript_preparation_started.elapsed().as_secs_f64();

    let dory_open_started = Instant::now();
    let (dory_opening, evaluation_commitment, evaluation_blind) = open_hidden_dense_i16_zk(
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

/// Verify the hidden Packed EO identity without opening the dense BVA.
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
        || proof.packed_positive_commitments.len() != expected_stride - 1
        || proof.packed_negative_commitments.len() != expected_stride - 1
        || proof.binary_round_commitments.len() != (logical_len / expected_stride).ilog2() as usize
    {
        return Err(HiddenPackedEoError::Shape);
    }
    let replay = replay_commitment_transcript(
        logical_len,
        expected_stride,
        bit_width,
        claimed,
        matrix_commitment,
        outer_binding,
        proof,
    )?;
    let mut identity_commitments = Vec::with_capacity(proof.scalar_commitment_count());
    identity_commitments.extend(proof.packed_positive_commitments.iter().copied());
    identity_commitments.extend(proof.packed_negative_commitments.iter().copied());
    for &(iota, rho) in &proof.binary_round_commitments {
        identity_commitments.push(iota);
        identity_commitments.push(rho);
    }
    identity_commitments.push(proof.a_reduced_commitment);
    identity_commitments.push(proof.b_reduced_commitment);
    let identity_expression = scalar_identity_expression(
        claimed,
        expected_stride,
        &replay.challenges,
        identity_commitments.len(),
    )?;
    verify_hidden_scalar_relation(
        &replay.transcript_state,
        parameters,
        &identity_commitments,
        &identity_expression,
        &proof.scalar_identity,
    )?;

    Ok(HiddenPackedEoVerification {
        a_reduced_commitment: proof.a_reduced_commitment,
        b_reduced_commitment: proof.b_reduced_commitment,
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
    verify_hidden_shortint_bva_opening_profiled(
        parameters,
        matrix_commitment,
        logical_len,
        8,
        packed,
        proof,
    )
}

/// Verify a native short-integer BVA opening selected by its committed bit width.
pub fn verify_hidden_shortint_bva_opening_profiled(
    parameters: &DoryVerifierParameters,
    matrix_commitment: &DoryCommitment,
    logical_len: usize,
    bit_width: usize,
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
    let evaluation_commitment = match bit_width {
        1..=8 => verify_hidden_dense_i8_zk(
            matrix_commitment,
            &proof.dory_opening,
            logical_len,
            &monomial_point,
            &packed.transcript_state,
            &parameters.pcs,
            &mut dory_transcript,
        )?,
        9..=16 => verify_hidden_dense_i16_zk(
            matrix_commitment,
            &proof.dory_opening,
            logical_len,
            &monomial_point,
            &packed.transcript_state,
            &parameters.pcs,
            &mut dory_transcript,
        )?,
        _ => return Err(HiddenPackedEoError::Shape),
    };
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
fn replay_commitment_transcript(
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
    append_commitments(
        &mut transcript,
        b"packed_positive",
        proof.packed_positive_commitments.iter(),
    );
    append_commitments(
        &mut transcript,
        b"packed_negative",
        proof.packed_negative_commitments.iter(),
    );
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
    for (round, &(iota, rho)) in proof.binary_round_commitments.iter().enumerate() {
        transcript.append_labeled(b"binary_round", &U64Word(round as u64));
        transcript.append_labeled(b"iota_commitment", &iota);
        transcript.append_labeled(b"rho_commitment", &rho);
        let challenge = Fr::from(challenge_nonzero(&mut transcript));
        let challenge_inv = challenge
            .inverse()
            .ok_or(HiddenPackedEoError::ZeroChallenge)?;
        challenges.push(challenge);
        challenges_inv.push(challenge_inv);
    }
    transcript.append_labeled(b"a_reduced_commitment", &proof.a_reduced_commitment);
    transcript.append_labeled(b"b_reduced_commitment", &proof.b_reduced_commitment);
    Ok(TranscriptReplay {
        transcript_state: transcript.state(),
        challenges,
        challenges_inv,
    })
}

fn scalar_identity_expression(
    claimed: Fr,
    packing_stride: usize,
    challenges: &[Fr],
    witness_count: usize,
) -> Result<RelationExpression, HiddenPackedEoError> {
    let stride_log = packing_stride.ilog2() as usize;
    if challenges.len() < stride_log {
        return Err(HiddenPackedEoError::Shape);
    }
    let packed_challenge = challenges[0];
    let packed_challenge_inv = packed_challenge
        .inverse()
        .ok_or(HiddenPackedEoError::ZeroChallenge)?;
    let packed_len = packing_stride - 1;
    let binary_round_count = challenges.len() - stride_log;
    let expected_count = 2 * packed_len + 2 * binary_round_count + 2;
    if witness_count != expected_count {
        return Err(HiddenPackedEoError::Shape);
    }
    let mut terms = vec![RelationExpression::constant(claimed)];
    let mut positive_power = packed_challenge;
    let mut negative_power = packed_challenge_inv;
    for offset in 0..packed_len {
        terms.push(
            RelationExpression::table(offset).mul(RelationExpression::constant(positive_power)),
        );
        terms.push(
            RelationExpression::table(packed_len + offset)
                .mul(RelationExpression::constant(negative_power)),
        );
        positive_power *= packed_challenge;
        negative_power *= packed_challenge_inv;
    }
    let mut index = 2 * packed_len;
    for challenge in &challenges[stride_log..] {
        let challenge_inv = challenge
            .inverse()
            .ok_or(HiddenPackedEoError::ZeroChallenge)?;
        terms.push(RelationExpression::table(index).mul(RelationExpression::constant(*challenge)));
        terms.push(
            RelationExpression::table(index + 1).mul(RelationExpression::constant(challenge_inv)),
        );
        index += 2;
    }
    terms.push(
        RelationExpression::constant(-Fr::one())
            .mul(RelationExpression::table(index).mul(RelationExpression::table(index + 1))),
    );
    balanced_sum(terms).ok_or(HiddenPackedEoError::Shape)
}

fn balanced_sum(mut terms: Vec<RelationExpression>) -> Option<RelationExpression> {
    while terms.len() > 1 {
        let mut next = Vec::with_capacity(terms.len().div_ceil(2));
        let mut pairs = terms.into_iter();
        while let Some(left) = pairs.next() {
            next.push(match pairs.next() {
                Some(right) => left.add(right),
                None => left,
            });
        }
        terms = next;
    }
    terms.pop()
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
    let mut transcript = Blake2bTranscript::<JoltFr>::new(b"pc_hidden_peo_v1");
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

fn append_commitments<'a>(
    transcript: &mut Blake2bTranscript<JoltFr>,
    label: &'static [u8],
    commitments: impl IntoIterator<Item = &'a Bn254G1>,
) {
    let commitments = commitments.into_iter().collect::<Vec<_>>();
    transcript.append(&LabelWithCount(label, commitments.len() as u64));
    for commitment in commitments {
        transcript.append_labeled(b"scalar_commitment", commitment);
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
    fn hidden_packed_eo_round_trip_keeps_reduction_scalars_committed() {
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
        let split_bytes = codec.serialize(&(&proof, &bva_opening)).unwrap();
        let decoded: (HiddenPackedEoProof, HiddenBvaOpeningProof) = codec
            .reject_trailing_bytes()
            .deserialize(&split_bytes)
            .unwrap();
        assert_eq!(decoded, (proof.clone(), bva_opening));
        assert_eq!(proof.a_reduced_commitment, claims.a_reduced.commitment);
        assert_eq!(proof.b_reduced_commitment, claims.b_reduced.commitment);
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
            proof.binary_round_commitments.len(),
            (source.layout.logical_len() / proof.packing_stride).ilog2() as usize
        );
        assert_eq!(
            proof.scalar_commitment_count(),
            2 * proof.packing_stride - 2 + 2 * proof.binary_round_commitments.len() + 2
        );
    }

    #[test]
    fn hidden_packed_eo_rejects_commitment_and_statement_tampering() {
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
        tampered.packed_positive_commitments[0] = proof.a_reduced_commitment;
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
        tampered.packed_negative_commitments[0] = proof.b_reduced_commitment;
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
        tampered.binary_round_commitments[0].0 = proof.a_reduced_commitment;
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
        tampered_terminal.a_reduced_commitment = Bn254G1::default();
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

        let mut tampered_terminal = proof.clone();
        tampered_terminal.b_reduced_commitment = Bn254G1::default();
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

        let mut tampered_identity = proof;
        tampered_identity.scalar_identity.blindfold_transcript_state[0] ^= 1;
        assert!(verify_hidden_packed_i8_reduction(
            &verifier,
            &matrix_commitment,
            source.layout.logical_len(),
            claimed,
            8,
            b"section-6-layout",
            &tampered_identity,
        )
        .is_err());
    }

    #[test]
    fn hidden_packed_eo_randomizes_commitments_for_same_statement() {
        let (mut first_source, tbv, claimed) = fixture();
        let (mut second_source, _, _) = fixture();
        let (prover, verifier) = setup_dory_parameters(7);
        let committed = commit_bva_i8_streaming(&prover, &first_source, 8).unwrap();
        let matrix_commitment = committed.commitment.clone();
        let mut first_rng = StdRng::seed_from_u64(0x4849_4444_454e_0001);
        let mut second_rng = StdRng::seed_from_u64(0x4849_4444_454e_0002);

        let (first, _, _, _) = prove_hidden_packed_i8_reduction(
            &prover,
            &mut first_source,
            &tbv,
            claimed,
            8,
            &matrix_commitment,
            b"section-6-layout",
            &mut first_rng,
        )
        .unwrap();
        let (second, _, _, _) = prove_hidden_packed_i8_reduction(
            &prover,
            &mut second_source,
            &tbv,
            claimed,
            8,
            &matrix_commitment,
            b"section-6-layout",
            &mut second_rng,
        )
        .unwrap();

        for proof in [&first, &second] {
            verify_hidden_packed_i8_reduction(
                &verifier,
                &matrix_commitment,
                first_source.layout.logical_len(),
                claimed,
                8,
                b"section-6-layout",
                proof,
            )
            .unwrap();
        }
        assert_ne!(first, second);
        assert_ne!(
            first.packed_positive_commitments,
            second.packed_positive_commitments
        );
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
