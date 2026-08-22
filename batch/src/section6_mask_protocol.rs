//! End-to-end orchestration for the mask-based Section 6 protocol.
//!
//! Registration commits the three persistent shape masks `D_eta`, `D_Z`, and
//! `D_eta'` plus the dense byte BVA; no registration validity proof is
//! produced. One opening then commits the point matrix `X`, derives the
//! public batching challenge `omega` by Fiat--Shamir, commits the compiled
//! point-compilation auxiliary matrix, proves the source and compaction
//! lookups plus the eta recurrences, runs Packed EO-LiteBullet against the
//! public aggregated claim, proves the final TBV recurrences at the Packed EO
//! point, opens the native byte BVA directly, and collects every deferred
//! auxiliary claim into one terminal Dory opening.

use std::{convert::Infallible, fmt::Display, time::Instant};

use ark_bn254::Fr;
use ark_ff::{One, PrimeField, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::rand::RngCore;
use jolt_dory::DoryCommitment;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    bucketed_ip::{PaddingWorkCounters, StreamingEoProfile},
    hidden_packed_eo::{
        prove_hidden_bva_opening, prove_hidden_packed_i8_reduction, verify_hidden_bva_opening,
        verify_hidden_packed_i8_reduction, HiddenBvaOpeningProof, HiddenPackedEoError,
        HiddenPackedEoProof, HiddenPackedEoProverError,
    },
    jolt_dory::{
        commitment_binding_bytes, CommittedDoryBva, DoryProverParameters, DoryVerifierParameters,
    },
    matrix_collection::DenseMatrixBva,
    section6_claim_collection::{Section6ProverClaimCollection, Section6VerifierClaimCollection},
    section6_data::{
        section6_batching_point, validate_fixed_claims, Section6DataError, Section6MaskFinalWitness,
        Section6MaskMatrixClaim, Section6MaskPointWitness, Section6PublicLayout,
        SECTION6_FIXED_ETA_WIDTH, SECTION6_FIXED_LOCAL_WIDTH,
    },
    section6_final_atomic::{
        build_section6_final_prover_plan, build_section6_final_verifier_plan,
        commit_section6_final_aux_on_grid_zk, prove_section6_final_atomic_deferred,
        verify_section6_final_atomic_deferred, Section6FinalAtomicError, Section6FinalAtomicProof,
    },
    section6_final_opening::{
        commit_section6_bva_i8_on_grid_zk, prove_section6_final_opening,
        verify_section6_final_opening, Section6DoryGrid, Section6FinalOpeningError,
        Section6FinalOpeningProof,
    },
    section6_point_compilation::{
        build_section6_point_prover_plan, build_section6_point_verifier_plan,
        commit_section6_point_aux_on_grid_zk, commit_section6_point_matrix_on_grid_zk,
        prove_section6_point_compilation_deferred, verify_section6_point_compilation_deferred,
        Section6PointAuxColumns, Section6PointCompilationError, Section6PointCompilationProof,
        Section6PointMatrixStatement, Section6PointPublicTables,
    },
    section6_shape_masks::{
        commit_section6_shape_masks_from_descriptor_on_grid_zk, CommittedSection6ShapeMasks,
        Section6ShapeMaskCommitmentError, Section6ShapeMaskStatement,
    },
};

const TERMINAL_OPENING_ROLE: &[u8] = b"section6-mask-terminal-opening-v2";

/// Commitment-only registration output. No registration proof is serialized.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Section6MaskRegistrationStatement {
    /// Shape-hiding public layout; bucket sizes and offsets stay private.
    pub layout: Section6PublicLayout,
    pub shape: Section6ShapeMaskStatement,
    pub bva_commitment: DoryCommitment,
}

impl Section6MaskRegistrationStatement {
    pub fn physical_dory_commitment_count(&self) -> usize {
        self.shape.physical_commitment_count() + 1
    }
}

/// Prover-only state retained between registration and opening.
pub struct RegisteredSection6MaskCollection {
    statement: Section6MaskRegistrationStatement,
    shapes: CommittedSection6ShapeMasks,
    bva: CommittedDoryBva,
}

/// Wall-clock split for the two persistent registration commitments.
#[derive(Clone, Copy, Debug, Default)]
pub struct Section6MaskRegistrationProfile {
    pub shape_mask_commit_seconds: f64,
    pub bva_commit_seconds: f64,
}

impl RegisteredSection6MaskCollection {
    pub fn statement(&self) -> &Section6MaskRegistrationStatement {
        &self.statement
    }

    pub fn shapes(&self) -> &CommittedSection6ShapeMasks {
        &self.shapes
    }

    pub fn bva(&self) -> &CommittedDoryBva {
        &self.bva
    }
}

/// Every reduction emitted by one mask-based opening.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Section6MaskOpeningProof {
    /// Commitment to the point matrix `X`, made before `omega` is sampled.
    pub x: Section6PointMatrixStatement,
    /// Source lookup, compaction lookup, and eta-prefix identity batch.
    pub point: Section6PointCompilationProof,
    pub packed_eo: HiddenPackedEoProof,
    pub final_atomic: Section6FinalAtomicProof,
    pub bva_opening: HiddenBvaOpeningProof,
    pub final_opening: Section6FinalOpeningProof,
}

impl Section6MaskOpeningProof {
    /// One native-byte BVA proof plus one auxiliary-field proof.
    pub fn physical_dory_proof_count(&self) -> usize {
        1 + self.final_opening.physical_dory_proof_count()
    }

    /// `X`, the point auxiliary, and the final auxiliary each add one GT.
    pub fn auxiliary_dory_commitment_count(&self) -> usize {
        3
    }
}

/// Prover profile for the Packed EO stage and the terminal claim batch.
#[derive(Clone, Debug)]
pub struct Section6MaskOpeningProfile {
    pub padding: PaddingWorkCounters,
    pub packed_eo: StreamingEoProfile,
    /// Dense BVA entries retained in a full-field vector by the opening path.
    pub materialized_dense_bva_field_entries: usize,
    pub x_commit_seconds: f64,
    pub point_witness_seconds: f64,
    pub point_aux_commit_seconds: f64,
    pub point_prove_seconds: f64,
    pub tbv_reconstruct_seconds: f64,
    pub packed_eo_seconds: f64,
    pub bva_open_seconds: f64,
    pub final_witness_seconds: f64,
    pub final_aux_commit_seconds: f64,
    pub final_prove_seconds: f64,
    pub auxiliary_open_seconds: f64,
    pub terminal_claim_count: usize,
}

/// Public verifier output after the terminal Dory proof accepts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Section6MaskOpeningVerification {
    /// `c = sum_tau omega^{tau-1} u_tau`, recomputed from the public inputs.
    pub aggregated_claim: Fr,
    pub terminal_claim_count: usize,
}

#[derive(Debug)]
pub enum Section6MaskProtocolError {
    Data(Section6DataError),
    ShapeCommitment(Section6ShapeMaskCommitmentError),
    PointCompilation(Section6PointCompilationError),
    FinalAtomic(Section6FinalAtomicError),
    PackedProver(HiddenPackedEoProverError<Infallible>),
    PackedVerifier(HiddenPackedEoError),
    FinalOpening(Section6FinalOpeningError),
    ClaimShape,
    Encoding(String),
    LengthOverflow,
}

impl Display for Section6MaskProtocolError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for Section6MaskProtocolError {}

impl From<Section6DataError> for Section6MaskProtocolError {
    fn from(error: Section6DataError) -> Self {
        Self::Data(error)
    }
}

impl From<Section6ShapeMaskCommitmentError> for Section6MaskProtocolError {
    fn from(error: Section6ShapeMaskCommitmentError) -> Self {
        Self::ShapeCommitment(error)
    }
}

impl From<Section6PointCompilationError> for Section6MaskProtocolError {
    fn from(error: Section6PointCompilationError) -> Self {
        Self::PointCompilation(error)
    }
}

impl From<Section6FinalAtomicError> for Section6MaskProtocolError {
    fn from(error: Section6FinalAtomicError) -> Self {
        Self::FinalAtomic(error)
    }
}

impl From<HiddenPackedEoProverError<Infallible>> for Section6MaskProtocolError {
    fn from(error: HiddenPackedEoProverError<Infallible>) -> Self {
        Self::PackedProver(error)
    }
}

impl From<HiddenPackedEoError> for Section6MaskProtocolError {
    fn from(error: HiddenPackedEoError) -> Self {
        Self::PackedVerifier(error)
    }
}

impl From<Section6FinalOpeningError> for Section6MaskProtocolError {
    fn from(error: Section6FinalOpeningError) -> Self {
        Self::FinalOpening(error)
    }
}

/// Minimum shared Dory grid covering the BVA, every auxiliary matrix, the
/// LogUp part matrices, and the BlindFold rows of the identity batches.
pub fn section6_mask_required_dory_log(
    layout: &Section6PublicLayout,
) -> Result<usize, Section6MaskProtocolError> {
    layout.validate()?;
    let rows = layout.row_cover()?;
    let local_width = layout.local_width();
    let point_cols = Section6PointAuxColumns::new(local_width).padded_cols();
    let final_cols = (3 * local_width + layout.global_log + 1)
        .checked_next_power_of_two()
        .ok_or(Section6MaskProtocolError::LengthOverflow)?;
    let mask_cols = SECTION6_FIXED_ETA_WIDTH
        .checked_next_power_of_two()
        .ok_or(Section6MaskProtocolError::LengthOverflow)?;
    let lookup_cols = (local_width + 2)
        .max(SECTION6_FIXED_LOCAL_WIDTH + 1)
        .checked_next_power_of_two()
        .ok_or(Section6MaskProtocolError::LengthOverflow)?;
    let widest_cols = point_cols.max(final_cols).max(mask_cols).max(lookup_cols);
    let auxiliary_len = rows
        .checked_mul(widest_cols)
        .ok_or(Section6MaskProtocolError::LengthOverflow)?;
    let source_log = layout.logical_len.ilog2() as usize;
    let auxiliary_log = auxiliary_len.ilog2() as usize;

    // The widest identity in either batch has degree 3, and the hidden
    // relation wrapper reserves two additional Pedersen rows; eight BlindFold
    // rows therefore cover every relation, requiring a grid of at least 2^6.
    let blindfold_grid_log = 6;
    Ok(source_log.max(auxiliary_log).max(blindfold_grid_log).max(2))
}

/// Commit the three shape masks once and the native dense byte BVA once.
pub fn register_section6_mask_collection(
    parameters: &DoryProverParameters,
    source: &DenseMatrixBva,
    bit_width: usize,
) -> Result<RegisteredSection6MaskCollection, Section6MaskProtocolError> {
    Ok(register_section6_mask_collection_profiled(parameters, source, bit_width)?.0)
}

/// Commit the registered sources and retain separate commitment timings.
pub fn register_section6_mask_collection_profiled(
    parameters: &DoryProverParameters,
    source: &DenseMatrixBva,
    bit_width: usize,
) -> Result<
    (
        RegisteredSection6MaskCollection,
        Section6MaskRegistrationProfile,
    ),
    Section6MaskProtocolError,
> {
    let layout =
        crate::section6_data::Section6Layout::from_descriptor(source.descriptor(), bit_width)?;
    let public_layout = layout.public();
    let grid = Section6DoryGrid::new(section6_mask_required_dory_log(&public_layout)?)?;
    let shape_started = Instant::now();
    let shapes = commit_section6_shape_masks_from_descriptor_on_grid_zk(
        parameters,
        &grid,
        layout,
        source.descriptor(),
    )?;
    let shape_mask_commit_seconds = shape_started.elapsed().as_secs_f64();
    let bva_started = Instant::now();
    let bva = commit_section6_bva_i8_on_grid_zk(parameters, &grid, source, bit_width)?;
    let bva_commit_seconds = bva_started.elapsed().as_secs_f64();
    let statement = Section6MaskRegistrationStatement {
        layout: public_layout,
        shape: shapes.statement().clone(),
        bva_commitment: bva.commitment.clone(),
    };
    Ok((
        RegisteredSection6MaskCollection {
            statement,
            shapes,
            bva,
        },
        Section6MaskRegistrationProfile {
            shape_mask_commit_seconds,
            bva_commit_seconds,
        },
    ))
}

/// Execute the complete prover flow with a streamed byte opening.
pub fn prove_section6_mask_opening(
    parameters: &DoryProverParameters,
    registered: &RegisteredSection6MaskCollection,
    source: &mut DenseMatrixBva,
    claims: &[Section6MaskMatrixClaim<Fr>],
    rng: &mut impl RngCore,
) -> Result<(Section6MaskOpeningProof, Section6MaskOpeningProfile), Section6MaskProtocolError> {
    let layout = registered.shapes.layout().clone();
    layout.validate_descriptor(source.descriptor())?;
    validate_fixed_claims(&layout, claims)?;
    validate_registered_state(registered)?;
    let grid = Section6DoryGrid::new(section6_mask_required_dory_log(
        &registered.statement.layout,
    )?)?;

    // Commit the claim points before the batching challenge is known.
    let x_commit_started = Instant::now();
    let x = commit_section6_point_matrix_on_grid_zk(parameters, &grid, &layout, claims)?;
    let x_commit_seconds = x_commit_started.elapsed().as_secs_f64();
    let claimed_values = claims.iter().map(|claim| claim.value).collect::<Vec<_>>();
    let (omega, omega_binding) =
        derive_omega(&registered.statement, &claimed_values, x.statement())?;
    let aggregated_claim = aggregate_claim_values(&claimed_values, omega);

    // Compile and commit the point-compilation auxiliary matrix, then prove
    // the lookups and the eta recurrences with deferred openings.
    let point_witness_started = Instant::now();
    let point_witness = Section6MaskPointWitness::compile(
        &layout,
        source.descriptor(),
        registered.shapes.metadata(),
        claims,
        omega,
    )?;
    let point_witness_seconds = point_witness_started.elapsed().as_secs_f64();
    debug_assert_eq!(point_witness.aggregated_claim, aggregated_claim);
    let point_aux_commit_started = Instant::now();
    let aux = commit_section6_point_aux_on_grid_zk(parameters, &grid, &layout, point_witness)?;
    let point_aux_commit_seconds = point_aux_commit_started.elapsed().as_secs_f64();
    let point_prove_started = Instant::now();
    let public_tables = Section6PointPublicTables::derive(&registered.statement.layout)?;
    let point_plan = build_section6_point_prover_plan(&registered.shapes, &aux, &public_tables)?;
    let (point_proof, point_openings) = prove_section6_point_compilation_deferred(
        &omega_binding,
        parameters,
        &grid,
        &registered.shapes,
        &x,
        &aux,
        &public_tables,
        &point_plan,
        rng,
    )?;
    let point_prove_seconds = point_prove_started.elapsed().as_secs_f64();

    // Packed EO against the public aggregated claim over the virtual TBV.
    let packed_binding = packed_stage_binding(&omega_binding, &point_proof)?;
    let tbv_reconstruct_started = Instant::now();
    let tbv = aux.witness().reconstruct_tbv(&layout)?;
    let tbv_reconstruct_seconds = tbv_reconstruct_started.elapsed().as_secs_f64();
    let packed_eo_started = Instant::now();
    let (packed_eo, packed_state, padding, packed_profile) = prove_hidden_packed_i8_reduction(
        parameters,
        source,
        &tbv,
        aggregated_claim,
        layout.bit_width,
        &registered.statement.bva_commitment,
        &packed_binding,
        rng,
    )?;
    let packed_eo_seconds = packed_eo_started.elapsed().as_secs_f64();
    let bva_open_started = Instant::now();
    let bva_opening =
        prove_hidden_bva_opening(parameters, source, &registered.bva, &packed_state, rng)?;
    let bva_open_seconds = bva_open_started.elapsed().as_secs_f64();

    // Final TBV recurrences at the Packed EO point, with deferred openings.
    let final_witness_started = Instant::now();
    let final_witness =
        Section6MaskFinalWitness::compile(&layout, aux.witness(), &packed_state.challenges)?;
    let final_witness_seconds = final_witness_started.elapsed().as_secs_f64();
    let final_aux_commit_started = Instant::now();
    let final_aux = commit_section6_final_aux_on_grid_zk(parameters, &grid, &layout, final_witness)?;
    let final_aux_commit_seconds = final_aux_commit_started.elapsed().as_secs_f64();
    let final_prove_started = Instant::now();
    let final_plan = build_section6_final_prover_plan(
        &registered.shapes,
        &aux,
        &final_aux,
        &public_tables,
        &packed_state.challenges,
    )?;
    let (final_atomic, final_openings) = prove_section6_final_atomic_deferred(
        &omega_binding,
        parameters,
        &registered.shapes,
        &aux,
        &final_aux,
        &public_tables,
        &final_plan,
        &packed_state,
        rng,
    )?;
    let final_prove_seconds = final_prove_started.elapsed().as_secs_f64();

    // Collect every deferred claim into the one terminal Dory opening.
    let mut terminal_claims = Section6ProverClaimCollection::new();
    terminal_claims.push_relation(&point_openings.identity, &grid)?;
    let source_lookup_openings = point_openings
        .source_lookup
        .relation_openings()
        .map_err(Section6PointCompilationError::LogUp)?;
    for opening in &source_lookup_openings {
        terminal_claims.push_relation(opening, &grid)?;
    }
    let compaction_lookup_openings = point_openings
        .compaction_lookup
        .relation_openings()
        .map_err(Section6PointCompilationError::LogUp)?;
    for opening in &compaction_lookup_openings {
        terminal_claims.push_relation(opening, &grid)?;
    }
    terminal_claims.push_relation(&final_openings.identity, &grid)?;
    let v_claim = final_openings.resolve_v_terminal_claim(&final_aux, &grid)?;
    let terminal_binding = terminal_stage_binding(
        &packed_binding,
        &packed_eo,
        &final_atomic,
        &bva_opening,
    )?;
    let mut final_claims = terminal_claims.claims();
    final_claims.push(v_claim.as_claim(&final_openings.average));
    let terminal_claim_count = final_claims.len();
    let auxiliary_open_started = Instant::now();
    let final_opening = prove_section6_final_opening(
        TERMINAL_OPENING_ROLE,
        &terminal_binding,
        parameters,
        &grid,
        &final_claims,
        rng,
    )?;
    let auxiliary_open_seconds = auxiliary_open_started.elapsed().as_secs_f64();

    Ok((
        Section6MaskOpeningProof {
            x: x.statement().clone(),
            point: point_proof,
            packed_eo,
            final_atomic,
            bva_opening,
            final_opening,
        },
        Section6MaskOpeningProfile {
            padding,
            packed_eo: packed_profile,
            materialized_dense_bva_field_entries: 0,
            x_commit_seconds,
            point_witness_seconds,
            point_aux_commit_seconds,
            point_prove_seconds,
            tbv_reconstruct_seconds,
            packed_eo_seconds,
            bva_open_seconds,
            final_witness_seconds,
            final_aux_commit_seconds,
            final_prove_seconds,
            auxiliary_open_seconds,
            terminal_claim_count,
        },
    ))
}

/// Verify the complete flow without shape masks, BVA bytes, or claim points.
pub fn verify_section6_mask_opening(
    parameters: &DoryVerifierParameters,
    statement: &Section6MaskRegistrationStatement,
    claimed_values: &[Fr],
    proof: &Section6MaskOpeningProof,
) -> Result<Section6MaskOpeningVerification, Section6MaskProtocolError> {
    statement.layout.validate()?;
    statement.shape.validate(&statement.layout)?;
    if claimed_values.len() != statement.layout.matrix_count {
        return Err(Section6MaskProtocolError::ClaimShape);
    }
    let grid = Section6DoryGrid::new(section6_mask_required_dory_log(&statement.layout)?)?;
    proof
        .x
        .validate(&statement.layout)
        .map_err(Section6MaskProtocolError::Data)?;
    let (omega, omega_binding) = derive_omega(statement, claimed_values, &proof.x)?;
    let aggregated_claim = aggregate_claim_values(claimed_values, omega);
    let public_tables = Section6PointPublicTables::derive(&statement.layout)?;

    let point_plan = build_section6_point_verifier_plan(
        &statement.layout,
        &statement.shape,
        &proof.point.aux,
        &public_tables,
        &section6_batching_point(omega),
    )?;
    let point_openings = verify_section6_point_compilation_deferred(
        &omega_binding,
        parameters,
        &grid,
        &statement.layout,
        &statement.shape,
        &proof.x,
        &public_tables,
        &point_plan,
        &proof.point,
    )?;

    let packed_binding = packed_stage_binding(&omega_binding, &proof.point)?;
    let packed = verify_hidden_packed_i8_reduction(
        parameters,
        &statement.bva_commitment,
        statement.layout.logical_len,
        aggregated_claim,
        statement.layout.bit_width,
        &packed_binding,
        &proof.packed_eo,
    )?;
    verify_hidden_bva_opening(
        parameters,
        &statement.bva_commitment,
        statement.layout.logical_len,
        &packed,
        &proof.bva_opening,
    )?;

    let final_plan = build_section6_final_verifier_plan(
        &statement.layout,
        &statement.shape,
        &proof.point.aux,
        &proof.final_atomic.aux,
        &public_tables,
        &packed.challenges,
    )?;
    let final_verification = verify_section6_final_atomic_deferred(
        &omega_binding,
        parameters,
        &statement.layout,
        &statement.shape,
        &proof.point.aux,
        &public_tables,
        &final_plan,
        &packed,
        &proof.final_atomic,
    )?;

    let mut terminal_claims = Section6VerifierClaimCollection::new();
    terminal_claims.push_relation(&point_openings.identity, &grid)?;
    let source_lookup_openings = point_openings
        .source_lookup
        .relation_openings()
        .map_err(Section6PointCompilationError::LogUp)?;
    for opening in &source_lookup_openings {
        terminal_claims.push_relation(opening, &grid)?;
    }
    let compaction_lookup_openings = point_openings
        .compaction_lookup
        .relation_openings()
        .map_err(Section6PointCompilationError::LogUp)?;
    for opening in &compaction_lookup_openings {
        terminal_claims.push_relation(opening, &grid)?;
    }
    terminal_claims.push_relation(&final_verification.identity, &grid)?;
    let v_claim = final_verification.resolve_v_terminal_claim(&grid)?;
    let terminal_binding = terminal_stage_binding(
        &packed_binding,
        &proof.packed_eo,
        &proof.final_atomic,
        &proof.bva_opening,
    )?;
    let mut final_claims = terminal_claims.claims();
    final_claims.push(v_claim.as_claim());
    let terminal_claim_count = final_claims.len();
    verify_section6_final_opening(
        TERMINAL_OPENING_ROLE,
        &terminal_binding,
        parameters,
        &grid,
        &final_claims,
        &proof.final_opening,
    )?;

    Ok(Section6MaskOpeningVerification {
        aggregated_claim,
        terminal_claim_count,
    })
}

fn validate_registered_state(
    registered: &RegisteredSection6MaskCollection,
) -> Result<(), Section6MaskProtocolError> {
    if registered.shapes.statement() != &registered.statement.shape
        || registered.bva.commitment != registered.statement.bva_commitment
        || registered.shapes.layout().public() != registered.statement.layout
    {
        return Err(Section6MaskProtocolError::ClaimShape);
    }
    Ok(())
}

fn aggregate_claim_values(claimed_values: &[Fr], omega: Fr) -> Fr {
    let mut omega_power = Fr::one();
    let mut aggregated = Fr::zero();
    for value in claimed_values {
        aggregated += *value * omega_power;
        omega_power *= omega;
    }
    aggregated
}

/// Derive the public batching challenge from everything committed so far.
fn derive_omega(
    statement: &Section6MaskRegistrationStatement,
    claimed_values: &[Fr],
    x_statement: &Section6PointMatrixStatement,
) -> Result<(Fr, [u8; 32]), Section6MaskProtocolError> {
    let mut hasher = Sha256::new();
    hasher.update(b"parcomm-section6-mask-omega-v2");
    append_serialized(&mut hasher, statement)?;
    append_field_values(&mut hasher, claimed_values)?;
    append_serialized(&mut hasher, x_statement)?;
    let binding: [u8; 32] = hasher.finalize().into();
    let mut counter = 0_u64;
    loop {
        let mut candidate = Sha256::new();
        candidate.update(b"parcomm-section6-mask-omega-sample");
        candidate.update(binding);
        candidate.update(counter.to_le_bytes());
        let omega = Fr::from_le_bytes_mod_order(&candidate.finalize());
        if !omega.is_zero() {
            return Ok((omega, binding));
        }
        counter += 1;
    }
}

fn packed_stage_binding(
    omega_binding: &[u8; 32],
    point: &Section6PointCompilationProof,
) -> Result<[u8; 32], Section6MaskProtocolError> {
    let mut hasher = Sha256::new();
    hasher.update(b"parcomm-section6-mask-packed-stage-v2");
    hasher.update(omega_binding);
    append_serialized(&mut hasher, point)?;
    Ok(hasher.finalize().into())
}

fn terminal_stage_binding(
    packed_binding: &[u8; 32],
    packed: &HiddenPackedEoProof,
    final_atomic: &Section6FinalAtomicProof,
    bva: &HiddenBvaOpeningProof,
) -> Result<[u8; 32], Section6MaskProtocolError> {
    let mut hasher = Sha256::new();
    hasher.update(b"parcomm-section6-mask-terminal-stage-v2");
    hasher.update(packed_binding);
    append_serialized(&mut hasher, packed)?;
    append_serialized(&mut hasher, final_atomic)?;
    append_serialized(&mut hasher, bva)?;
    Ok(hasher.finalize().into())
}

fn append_serialized<T: Serialize>(
    hasher: &mut Sha256,
    value: &T,
) -> Result<(), Section6MaskProtocolError> {
    let encoded = bincode::serialize(value)
        .map_err(|error| Section6MaskProtocolError::Encoding(error.to_string()))?;
    hasher.update((encoded.len() as u64).to_le_bytes());
    hasher.update(encoded);
    Ok(())
}

fn append_field_values(
    hasher: &mut Sha256,
    values: &[Fr],
) -> Result<(), Section6MaskProtocolError> {
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        let mut encoded = Vec::new();
        value
            .serialize_compressed(&mut encoded)
            .map_err(|error| Section6MaskProtocolError::Encoding(error.to_string()))?;
        hasher.update((encoded.len() as u64).to_le_bytes());
        hasher.update(encoded);
    }
    Ok(())
}

#[allow(dead_code)]
fn append_commitment(hasher: &mut Sha256, commitment: &DoryCommitment) {
    let encoded = commitment_binding_bytes(commitment);
    hasher.update((encoded.len() as u64).to_le_bytes());
    hasher.update(encoded);
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};
    use bincode::Options;

    use crate::{
        jolt_dory::setup_dory_parameters,
        matrix_collection::{matrix_mle_evaluate, DenseI8Matrix},
        section6_data::Section6Layout,
    };

    use super::*;

    fn fixture() -> (DenseMatrixBva, Vec<Section6MaskMatrixClaim<Fr>>) {
        let shapes = [(1, 2), (1, 1)];
        let matrices = shapes
            .iter()
            .enumerate()
            .map(|(matrix, &(rows, columns))| {
                let values = (0..rows * columns)
                    .map(|index| ((matrix * 37 + index * 5 + 3) % 29) as i8 - 14)
                    .collect::<Vec<_>>();
                DenseI8Matrix::new((rows, columns), values).unwrap()
            })
            .collect::<Vec<_>>();
        let source = DenseMatrixBva::from_matrices(&matrices).unwrap();
        let claims = matrices
            .iter()
            .enumerate()
            .map(|(matrix, dense)| {
                let row_log = dense.shape().0.next_power_of_two().ilog2() as usize;
                let col_log = dense.shape().1.next_power_of_two().ilog2() as usize;
                let left = (0..row_log)
                    .map(|coordinate| Fr::from((11 + matrix * 7 + coordinate) as u64))
                    .collect::<Vec<_>>();
                let right = (0..col_log)
                    .map(|coordinate| Fr::from((41 + matrix * 5 + coordinate) as u64))
                    .collect::<Vec<_>>();
                let value = matrix_mle_evaluate(dense, &left, &right).unwrap();
                Section6MaskMatrixClaim::from_unpadded(&left, &right, value).unwrap()
            })
            .collect::<Vec<_>>();
        (source, claims)
    }

    #[test]
    fn complete_mask_flow_streams_i8_bva_and_batches_only_auxiliary_claims() {
        let (mut source, claims) = fixture();
        let layout = Section6Layout::from_descriptor(source.descriptor(), 8).unwrap();
        let setup_log = section6_mask_required_dory_log(&layout.public()).unwrap();
        let (prover, verifier) = setup_dory_parameters(setup_log);
        let registered = register_section6_mask_collection(&prover, &source, 8).unwrap();
        assert_eq!(registered.statement().physical_dory_commitment_count(), 4);
        assert_eq!(
            registered.statement().shape.eta_mask_width,
            SECTION6_FIXED_ETA_WIDTH
        );

        let mut rng = StdRng::seed_from_u64(0x4d41_534b_5f45_3245);
        let (proof, profile) =
            prove_section6_mask_opening(&prover, &registered, &mut source, &claims, &mut rng)
                .unwrap();
        assert_eq!(proof.physical_dory_proof_count(), 2);
        assert_eq!(proof.auxiliary_dory_commitment_count(), 3);
        assert_eq!(profile.materialized_dense_bva_field_entries, 0);
        assert!(profile.terminal_claim_count > 0);

        let codec = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .reject_trailing_bytes();
        let bytes = codec.serialize(&proof).unwrap();
        let decoded: Section6MaskOpeningProof = bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .reject_trailing_bytes()
            .deserialize(&bytes)
            .unwrap();
        let values = claims.iter().map(|claim| claim.value).collect::<Vec<_>>();
        let verified =
            verify_section6_mask_opening(&verifier, registered.statement(), &values, &decoded)
                .unwrap();
        assert_eq!(verified.terminal_claim_count, profile.terminal_claim_count);

        let mut wrong_values = values.clone();
        wrong_values[0] += Fr::one();
        assert!(verify_section6_mask_opening(
            &verifier,
            registered.statement(),
            &wrong_values,
            &decoded,
        )
        .is_err());

        let mut tampered_statement = registered.statement().clone();
        tampered_statement.bva_commitment = tampered_statement.shape.d_z_commitment.clone();
        assert!(
            verify_section6_mask_opening(&verifier, &tampered_statement, &values, &decoded,)
                .is_err()
        );

        let mut trailing = bytes;
        trailing.push(0);
        assert!(bincode::DefaultOptions::new()
            .with_fixint_encoding()
            .reject_trailing_bytes()
            .deserialize::<Section6MaskOpeningProof>(&trailing)
            .is_err());
    }

    #[test]
    fn false_private_matrix_claim_never_verifies() {
        let (mut source, mut claims) = fixture();
        let layout = Section6Layout::from_descriptor(source.descriptor(), 8).unwrap();
        let setup_log = section6_mask_required_dory_log(&layout.public()).unwrap();
        let (prover, verifier) = setup_dory_parameters(setup_log);
        let registered = register_section6_mask_collection(&prover, &source, 8).unwrap();
        claims[1].value += Fr::one();
        let mut rng = StdRng::seed_from_u64(0x4641_4c53_455f_434c);
        // The prover no longer re-evaluates its own claims, so proving may
        // succeed; soundness requires that the resulting proof is rejected.
        match prove_section6_mask_opening(&prover, &registered, &mut source, &claims, &mut rng) {
            Err(_) => {}
            Ok((proof, _)) => {
                let values = claims.iter().map(|claim| claim.value).collect::<Vec<_>>();
                assert!(verify_section6_mask_opening(
                    &verifier,
                    registered.statement(),
                    &values,
                    &proof,
                )
                .is_err());
            }
        }
    }
}
