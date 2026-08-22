//! Commitment-bound aggregation of the Section 6 claim values.
//!
//! Claim values are zero-padded to a power of two and committed as one dense
//! field vector. The batching challenge is sampled only after the value
//! commitment and the caller-provided opening context have been bound. A
//! hidden Dory opening returns a Pedersen commitment to the aggregated claim,
//! which later protocols can consume without receiving the raw claim values.

use std::fmt::{Display, Formatter};

use ark_bn254::Fr as ArkFr;
use ark_ff::{One, Zero};
use ark_std::rand::RngCore;
use jolt_crypto::{Bn254G1, HomomorphicCommitment, Pedersen, VectorCommitment};
use jolt_dory::DoryCommitment;
use jolt_field::Fr as JoltFr;
use jolt_transcript::domain::{Label, LabelWithCount, U64Word};
use jolt_transcript::{Blake2bTranscript, Transcript};
use serde::{Deserialize, Serialize};

use crate::{
    hidden_relation::HiddenScalarWitness,
    jolt_dory::{
        commit_field_vector_zk, normalize_monomial_mle_claim, open_hidden_field_vectors_batch_zk,
        verify_hidden_field_vectors_batch_zk, CommittedDoryFieldVector, DoryBackendError,
        DoryProverParameters, DoryVerifierParameters,
    },
    jolt_dory_adapter::{DenseI8DoryError, HiddenFieldVectorDoryProof},
    section6_final_opening::{
        commit_section6_field_vector_view_on_grid_zk, Section6DoryGrid, Section6FinalOpeningError,
    },
};

const CLAIM_VALUE_TRANSCRIPT_DOMAIN: &[u8] = b"pc-section6-values-v1";
const CLAIM_VALUE_DORY_ROLE: &[u8] = b"section6-claim-values";

/// Commitment-only verifier statement for the ordered Section 6 claim values.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Section6ClaimValueStatement {
    pub commitment: DoryCommitment,
    pub count: usize,
}

/// Prover-only dense value vector and its Dory opening hint.
pub struct CommittedSection6ClaimValues {
    statement: Section6ClaimValueStatement,
    padded_values: Vec<ArkFr>,
    committed: CommittedDoryFieldVector,
}

impl CommittedSection6ClaimValues {
    pub fn statement(&self) -> Section6ClaimValueStatement {
        self.statement.clone()
    }
}

/// Hidden Dory opening of the committed value vector at the batching point.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Section6ClaimValueOpeningProof {
    pub opening: HiddenFieldVectorDoryProof,
    pub normalized_evaluation_commitment: Bn254G1,
    pub blind_delta: JoltFr,
}

/// Commitment-only claim-value reduction included in the joint Section 6
/// proof. Its PCS opening is supplied by the single terminal Dory batch.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Section6ClaimValueReductionProof {
    pub normalized_evaluation_commitment: Bn254G1,
}

/// Prover-only state retained until the final PCS-opening phase.
pub struct PendingSection6ClaimValueOpening<'a> {
    committed: &'a CommittedSection6ClaimValues,
    standard_point: Vec<JoltFr>,
    normalized_value: JoltFr,
    normalized_witness: HiddenScalarWitness,
    outer_binding: Vec<u8>,
}

/// Prover-side aggregate, including the Pedersen opening needed by BlindFold.
#[derive(Clone, Debug)]
pub struct Section6ClaimValueProverOutput {
    pub omega: ArkFr,
    pub aggregated_claim: HiddenScalarWitness,
}

/// Verifier-side aggregate commitment reusable by later hidden protocols.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Section6ClaimValueVerification {
    pub omega: ArkFr,
    pub normalized_evaluation_commitment: Bn254G1,
    pub aggregated_claim_commitment: Bn254G1,
}

#[derive(Debug)]
pub enum Section6ClaimValueError {
    InvalidCount(usize),
    Backend(DoryBackendError),
    Opening(DenseI8DoryError),
    FinalOpening(Section6FinalOpeningError),
}

impl Display for Section6ClaimValueError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidCount(count) => write!(formatter, "invalid claim-value count {count}"),
            Self::Backend(error) => Display::fmt(error, formatter),
            Self::Opening(error) => Display::fmt(error, formatter),
            Self::FinalOpening(error) => Display::fmt(error, formatter),
        }
    }
}

impl std::error::Error for Section6ClaimValueError {}

impl From<DoryBackendError> for Section6ClaimValueError {
    fn from(error: DoryBackendError) -> Self {
        Self::Backend(error)
    }
}

impl From<DenseI8DoryError> for Section6ClaimValueError {
    fn from(error: DenseI8DoryError) -> Self {
        Self::Opening(error)
    }
}

impl From<Section6FinalOpeningError> for Section6ClaimValueError {
    fn from(error: Section6FinalOpeningError) -> Self {
        Self::FinalOpening(error)
    }
}

/// Commit the ordered claim values as a dense field vector with a zero suffix.
pub fn commit_section6_claim_values(
    parameters: &DoryProverParameters,
    values: &[ArkFr],
) -> Result<CommittedSection6ClaimValues, Section6ClaimValueError> {
    let logical_len = claim_value_logical_len(values.len())?;
    let mut padded_values = Vec::with_capacity(logical_len);
    padded_values.extend_from_slice(values);
    padded_values.resize(logical_len, ArkFr::zero());
    let committed = commit_field_vector_zk(parameters, &padded_values)?;
    let statement = Section6ClaimValueStatement {
        commitment: committed.commitment.clone(),
        count: values.len(),
    };
    Ok(CommittedSection6ClaimValues {
        statement,
        padded_values,
        committed,
    })
}

/// Commit claim values on the shared Section 6 Dory grid.
pub fn commit_section6_claim_values_on_grid(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    values: &[ArkFr],
) -> Result<CommittedSection6ClaimValues, Section6ClaimValueError> {
    let logical_len = claim_value_logical_len(values.len())?;
    let mut padded_values = Vec::with_capacity(logical_len);
    padded_values.extend_from_slice(values);
    padded_values.resize(logical_len, ArkFr::zero());
    let committed = commit_section6_field_vector_view_on_grid_zk(parameters, grid, &padded_values)?;
    let statement = Section6ClaimValueStatement {
        commitment: committed.commitment.clone(),
        count: values.len(),
    };
    Ok(CommittedSection6ClaimValues {
        statement,
        padded_values,
        committed,
    })
}

impl CommittedSection6ClaimValues {
    pub(crate) fn values(&self) -> &[ArkFr] {
        &self.padded_values
    }

    pub(crate) fn committed(&self) -> &CommittedDoryFieldVector {
        &self.committed
    }
}

impl PendingSection6ClaimValueOpening<'_> {
    pub fn reduction_proof(&self) -> Section6ClaimValueReductionProof {
        Section6ClaimValueReductionProof {
            normalized_evaluation_commitment: self.normalized_witness.commitment,
        }
    }

    pub(crate) fn standard_point_ark(&self) -> Vec<ArkFr> {
        self.standard_point
            .iter()
            .copied()
            .map(ArkFr::from)
            .collect()
    }

    pub(crate) fn normalized_witness(&self) -> &HiddenScalarWitness {
        &self.normalized_witness
    }

    pub(crate) fn committed_values(&self) -> &CommittedSection6ClaimValues {
        self.committed
    }
}

/// Prove the hidden opening whose monomial value is
/// `sum_i omega^i * values[i]`.
pub fn prove_section6_claim_value_opening(
    parameters: &DoryProverParameters,
    committed: &CommittedSection6ClaimValues,
    outer_binding: &[u8],
    rng: &mut impl RngCore,
) -> Result<
    (
        Section6ClaimValueOpeningProof,
        Section6ClaimValueProverOutput,
    ),
    Section6ClaimValueError,
> {
    let (pending, output) =
        prepare_section6_claim_value_opening(parameters, committed, outer_binding, rng)?;
    let proof = finalize_section6_claim_value_opening(parameters, pending)?;
    Ok((proof, output))
}

/// Bind the claim-value opening and expose only the hidden aggregate
/// commitment. No PCS opening is produced until
/// [`finalize_section6_claim_value_opening`] is called.
pub fn prepare_section6_claim_value_opening<'a>(
    parameters: &DoryProverParameters,
    committed: &'a CommittedSection6ClaimValues,
    outer_binding: &[u8],
    rng: &mut impl RngCore,
) -> Result<
    (
        PendingSection6ClaimValueOpening<'a>,
        Section6ClaimValueProverOutput,
    ),
    Section6ClaimValueError,
> {
    let point = claim_value_opening_point(&committed.statement, outer_binding)?;
    let aggregated_value = aggregate_claim_values(
        &committed.padded_values[..committed.statement.count],
        point.omega,
    );
    let normalized = normalize_monomial_mle_claim(&point.monomial_point_low, aggregated_value)?;
    let standard_point = normalized
        .point_msb
        .iter()
        .copied()
        .map(JoltFr::from)
        .collect::<Vec<_>>();
    let normalized_witness = HiddenScalarWitness::commit(parameters, normalized.value, rng);
    let scale = JoltFr::from(normalized.scale);
    let aggregated_commitment = scale_commitment(&normalized_witness.commitment, scale);
    let aggregated_claim = HiddenScalarWitness {
        value: aggregated_value,
        commitment: aggregated_commitment,
        blind: normalized_witness.blind * scale,
    };
    Ok((
        PendingSection6ClaimValueOpening {
            committed,
            standard_point,
            normalized_value: JoltFr::from(normalized.value),
            normalized_witness,
            outer_binding: outer_binding.to_vec(),
        },
        Section6ClaimValueProverOutput {
            omega: point.omega,
            aggregated_claim,
        },
    ))
}

/// Produce the native Dory batch opening after every consumer has already
/// bound the hidden aggregate commitment.
pub fn finalize_section6_claim_value_opening(
    parameters: &DoryProverParameters,
    pending: PendingSection6ClaimValueOpening<'_>,
) -> Result<Section6ClaimValueOpeningProof, Section6ClaimValueError> {
    let values = [pending.committed.padded_values.as_slice()];
    let committed = [&pending.committed.committed];
    let evaluations = [pending.normalized_value];
    let evaluation_commitments = [pending.normalized_witness.commitment];
    let (opening, joint_commitment, joint_blind, coefficients) =
        open_hidden_field_vectors_batch_zk(
            parameters,
            &values,
            &committed,
            &pending.standard_point,
            &evaluations,
            &evaluation_commitments,
            CLAIM_VALUE_DORY_ROLE,
            &pending.outer_binding,
        )?;
    let coefficient = *coefficients
        .first()
        .ok_or(Section6ClaimValueError::InvalidCount(0))?;
    let combined_commitment = scale_commitment(&pending.normalized_witness.commitment, coefficient);
    let combined_blind = pending.normalized_witness.blind * coefficient;
    let blind_delta = joint_blind - combined_blind;
    verify_commitment_link(
        &parameters.blindfold,
        &combined_commitment,
        &joint_commitment,
        blind_delta,
    )?;
    Ok(Section6ClaimValueOpeningProof {
        opening,
        normalized_evaluation_commitment: pending.normalized_witness.commitment,
        blind_delta,
    })
}

/// Verify from `C_V`, the value count, and the already-bound opening context.
pub fn verify_section6_claim_value_opening(
    parameters: &DoryVerifierParameters,
    statement: &Section6ClaimValueStatement,
    outer_binding: &[u8],
    proof: &Section6ClaimValueOpeningProof,
) -> Result<Section6ClaimValueVerification, Section6ClaimValueError> {
    let verification = prepare_section6_claim_value_verification(statement, outer_binding, proof)?;
    finalize_section6_claim_value_verification(parameters, statement, outer_binding, proof)?;
    Ok(verification)
}

/// Reconstruct the hidden aggregate commitment before the final PCS batch is
/// checked. This step reads only the serialized evaluation commitment.
pub fn prepare_section6_claim_value_verification(
    statement: &Section6ClaimValueStatement,
    outer_binding: &[u8],
    proof: &Section6ClaimValueOpeningProof,
) -> Result<Section6ClaimValueVerification, Section6ClaimValueError> {
    let point = claim_value_opening_point(statement, outer_binding)?;
    let normalized = normalize_monomial_mle_claim(&point.monomial_point_low, ArkFr::one())?;
    let aggregated_claim_commitment = scale_commitment(
        &proof.normalized_evaluation_commitment,
        JoltFr::from(normalized.scale),
    );
    Ok(Section6ClaimValueVerification {
        omega: point.omega,
        normalized_evaluation_commitment: proof.normalized_evaluation_commitment,
        aggregated_claim_commitment,
    })
}

/// Reconstruct the claim-value batching challenge and aggregate commitment
/// before the one terminal PCS opening is verified.
pub fn prepare_section6_claim_value_reduction_verification(
    statement: &Section6ClaimValueStatement,
    outer_binding: &[u8],
    proof: &Section6ClaimValueReductionProof,
) -> Result<Section6ClaimValueVerification, Section6ClaimValueError> {
    let point = claim_value_opening_point(statement, outer_binding)?;
    let normalized = normalize_monomial_mle_claim(&point.monomial_point_low, ArkFr::one())?;
    let aggregated_claim_commitment = scale_commitment(
        &proof.normalized_evaluation_commitment,
        JoltFr::from(normalized.scale),
    );
    Ok(Section6ClaimValueVerification {
        omega: point.omega,
        normalized_evaluation_commitment: proof.normalized_evaluation_commitment,
        aggregated_claim_commitment,
    })
}

/// Return the standard MLE point represented by the reduction statement.
pub(crate) fn section6_claim_value_standard_point(
    statement: &Section6ClaimValueStatement,
    outer_binding: &[u8],
) -> Result<Vec<ArkFr>, Section6ClaimValueError> {
    let point = claim_value_opening_point(statement, outer_binding)?;
    let normalized = normalize_monomial_mle_claim(&point.monomial_point_low, ArkFr::one())?;
    Ok(normalized.point_msb)
}

pub(crate) fn section6_claim_value_logical_len(
    count: usize,
) -> Result<usize, Section6ClaimValueError> {
    claim_value_logical_len(count)
}

/// Verify the deferred native Dory opening in the final PCS-opening phase.
pub fn finalize_section6_claim_value_verification(
    parameters: &DoryVerifierParameters,
    statement: &Section6ClaimValueStatement,
    outer_binding: &[u8],
    proof: &Section6ClaimValueOpeningProof,
) -> Result<(), Section6ClaimValueError> {
    let logical_len = claim_value_logical_len(statement.count)?;
    let point = claim_value_opening_point(statement, outer_binding)?;
    let normalized = normalize_monomial_mle_claim(&point.monomial_point_low, ArkFr::one())?;
    let standard_point = normalized
        .point_msb
        .iter()
        .copied()
        .map(JoltFr::from)
        .collect::<Vec<_>>();
    let (joint_commitment, coefficients) = verify_hidden_field_vectors_batch_zk(
        parameters,
        &[&statement.commitment],
        logical_len,
        &standard_point,
        &[proof.normalized_evaluation_commitment],
        &proof.opening,
        CLAIM_VALUE_DORY_ROLE,
        outer_binding,
    )?;
    let coefficient = *coefficients
        .first()
        .ok_or(Section6ClaimValueError::InvalidCount(0))?;
    let combined_commitment =
        scale_commitment(&proof.normalized_evaluation_commitment, coefficient);
    verify_commitment_link(
        &parameters.blindfold,
        &combined_commitment,
        &joint_commitment,
        proof.blind_delta,
    )
}

struct ClaimValueOpeningPoint {
    omega: ArkFr,
    monomial_point_low: Vec<ArkFr>,
}

fn claim_value_opening_point(
    statement: &Section6ClaimValueStatement,
    outer_binding: &[u8],
) -> Result<ClaimValueOpeningPoint, Section6ClaimValueError> {
    let logical_len = claim_value_logical_len(statement.count)?;
    let mut transcript = Blake2bTranscript::<JoltFr>::new(CLAIM_VALUE_TRANSCRIPT_DOMAIN);
    transcript.append(&Label(b"section6_claim_value_statement"));
    transcript.append(&LabelWithCount(
        b"outer_binding",
        outer_binding.len() as u64,
    ));
    transcript.append_bytes(outer_binding);
    transcript.append_labeled(b"claim_count", &U64Word(statement.count as u64));
    transcript.append_labeled(b"logical_len", &U64Word(logical_len as u64));
    transcript.append_labeled(b"value_commitment", &statement.commitment);
    let omega = nonzero_challenge(&mut transcript);
    let mut coordinate = omega;
    let monomial_point_low = (0..logical_len.trailing_zeros() as usize)
        .map(|_| {
            let current = coordinate;
            coordinate *= coordinate;
            ArkFr::from(current)
        })
        .collect();
    Ok(ClaimValueOpeningPoint {
        omega: ArkFr::from(omega),
        monomial_point_low,
    })
}

fn nonzero_challenge(transcript: &mut Blake2bTranscript<JoltFr>) -> JoltFr {
    let mut retry = 0_u64;
    loop {
        let challenge = transcript.challenge();
        if challenge != JoltFr::from(0_u64) {
            return challenge;
        }
        retry += 1;
        transcript.append_labeled(b"challenge_retry", &U64Word(retry));
    }
}

fn claim_value_logical_len(count: usize) -> Result<usize, Section6ClaimValueError> {
    if count == 0 {
        return Err(Section6ClaimValueError::InvalidCount(count));
    }
    count
        .checked_next_power_of_two()
        .ok_or(Section6ClaimValueError::InvalidCount(count))
}

fn aggregate_claim_values(values: &[ArkFr], omega: ArkFr) -> ArkFr {
    let mut power = ArkFr::one();
    values.iter().fold(ArkFr::zero(), |sum, value| {
        let term = *value * power;
        power *= omega;
        sum + term
    })
}

fn scale_commitment(commitment: &Bn254G1, scale: JoltFr) -> Bn254G1 {
    Bn254G1::linear_combine(&Bn254G1::default(), commitment, &scale)
}

fn verify_commitment_link(
    setup: &jolt_crypto::PedersenSetup<Bn254G1>,
    combined_commitment: &Bn254G1,
    joint_commitment: &Bn254G1,
    blind_delta: JoltFr,
) -> Result<(), Section6ClaimValueError> {
    let delta = Pedersen::<Bn254G1>::commit(setup, &[JoltFr::zero()], &blind_delta);
    if <Bn254G1 as HomomorphicCommitment<JoltFr>>::add(combined_commitment, &delta)
        != *joint_commitment
    {
        return Err(Section6ClaimValueError::Opening(
            DenseI8DoryError::EvaluationCommitmentMismatch,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ark_bn254::Fr;
    use ark_ff::{Field, Zero};
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use jolt_crypto::{Bn254G1, Pedersen, VectorCommitment};
    use jolt_field::Fr as JoltFr;

    use crate::jolt_dory::setup_dory_parameters;

    use super::{
        commit_section6_claim_values, prove_section6_claim_value_opening,
        verify_section6_claim_value_opening, Section6ClaimValueStatement,
    };

    #[test]
    fn committed_claim_values_open_to_a_hidden_random_linear_combination() {
        let values = [Fr::from(2_u64), Fr::from(3_u64), Fr::from(5_u64)];
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(2);
        let committed = commit_section6_claim_values(&prover_parameters, &values).unwrap();
        let statement = committed.statement();
        let outer_binding = b"registered-shape-and-point-commitments";
        let mut rng = StdRng::seed_from_u64(1);

        let (proof, prover_output) = prove_section6_claim_value_opening(
            &prover_parameters,
            &committed,
            outer_binding,
            &mut rng,
        )
        .unwrap();
        let verifier_output = verify_section6_claim_value_opening(
            &verifier_parameters,
            &statement,
            outer_binding,
            &proof,
        )
        .unwrap();

        let expected =
            values[0] + prover_output.omega * values[1] + prover_output.omega.square() * values[2];
        assert_eq!(prover_output.aggregated_claim.value, expected);
        assert_eq!(verifier_output.omega, prover_output.omega);
        assert_eq!(
            verifier_output.aggregated_claim_commitment,
            prover_output.aggregated_claim.commitment
        );
        assert!(Pedersen::<Bn254G1>::verify(
            &prover_parameters.blindfold,
            &prover_output.aggregated_claim.commitment,
            &[JoltFr::from(expected)],
            &prover_output.aggregated_claim.blind,
        ));
    }

    #[test]
    fn claim_value_opening_rejects_a_substituted_commitment() {
        let values = [Fr::from(2_u64), Fr::from(3_u64), Fr::from(5_u64)];
        let substitute = [Fr::from(2_u64), Fr::from(3_u64), Fr::from(7_u64)];
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(2);
        let committed = commit_section6_claim_values(&prover_parameters, &values).unwrap();
        let substituted = commit_section6_claim_values(&prover_parameters, &substitute).unwrap();
        let outer_binding = b"registered-shape-and-point-commitments";
        let mut rng = StdRng::seed_from_u64(2);
        let (proof, _) = prove_section6_claim_value_opening(
            &prover_parameters,
            &committed,
            outer_binding,
            &mut rng,
        )
        .unwrap();

        assert!(verify_section6_claim_value_opening(
            &verifier_parameters,
            &substituted.statement(),
            outer_binding,
            &proof,
        )
        .is_err());
    }

    #[test]
    fn claim_value_opening_rejects_a_substituted_count() {
        let values = [
            Fr::from(2_u64),
            Fr::from(3_u64),
            Fr::from(5_u64),
            Fr::zero(),
        ];
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(2);
        let committed = commit_section6_claim_values(&prover_parameters, &values).unwrap();
        let statement = committed.statement();
        let outer_binding = b"registered-shape-and-point-commitments";
        let mut rng = StdRng::seed_from_u64(3);
        let (proof, _) = prove_section6_claim_value_opening(
            &prover_parameters,
            &committed,
            outer_binding,
            &mut rng,
        )
        .unwrap();
        let substituted = Section6ClaimValueStatement {
            commitment: statement.commitment,
            count: 3,
        };

        assert!(verify_section6_claim_value_opening(
            &verifier_parameters,
            &substituted,
            outer_binding,
            &proof,
        )
        .is_err());
    }

    #[test]
    fn claim_value_opening_rejects_a_substituted_outer_binding() {
        let values = [Fr::from(2_u64), Fr::from(3_u64), Fr::from(5_u64)];
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(2);
        let committed = commit_section6_claim_values(&prover_parameters, &values).unwrap();
        let statement = committed.statement();
        let mut rng = StdRng::seed_from_u64(4);
        let (proof, _) = prove_section6_claim_value_opening(
            &prover_parameters,
            &committed,
            b"registered-shape-and-point-commitments",
            &mut rng,
        )
        .unwrap();

        assert!(verify_section6_claim_value_opening(
            &verifier_parameters,
            &statement,
            b"different-registration-or-point-commitment",
            &proof,
        )
        .is_err());
    }

    #[test]
    fn claim_value_opening_rejects_a_substituted_hidden_evaluation_commitment() {
        let values = [Fr::from(2_u64), Fr::from(3_u64), Fr::from(5_u64)];
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(2);
        let committed = commit_section6_claim_values(&prover_parameters, &values).unwrap();
        let statement = committed.statement();
        let outer_binding = b"registered-shape-and-point-commitments";
        let mut rng = StdRng::seed_from_u64(5);
        let (mut proof, _) = prove_section6_claim_value_opening(
            &prover_parameters,
            &committed,
            outer_binding,
            &mut rng,
        )
        .unwrap();
        proof.normalized_evaluation_commitment = Bn254G1::default();

        assert!(verify_section6_claim_value_opening(
            &verifier_parameters,
            &statement,
            outer_binding,
            &proof,
        )
        .is_err());
    }
}
