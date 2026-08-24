//! End-to-end orchestration for the committed-shape Section 6 protocol.
//!
//! The one-time commit phase commits the mask matrix `[D_eta | D_Z]` on the
//! small auxiliary grid and commits the dense byte BVA on its native
//! source-sized grid. The optional size-private mode additionally proves the
//! shape validity argument with its own terminal Dory batch.
//!
//! Each opening then commits the claim-point matrix `X`, derives the tensor
//! batching point by Fiat--Shamir, commits the compiled TBV `[eta | Z]`, the
//! compilation auxiliary `[X' | Pi'_eta | gamma_r]`, and the key columns
//! `[q_key | t_key | V]`, proves the compilation identities plus the source
//! and power lookups, runs Packed EO-LiteBullet against the public
//! aggregated claim, proves the TBV evaluation identities at the Packed EO
//! point, opens the native byte BVA directly, and collects every other
//! deferred claim - including all LogUp auxiliary-vector openings - into one
//! terminal Dory opening on the auxiliary grid.

use std::{convert::Infallible, fmt::Display, time::Instant};

use ark_bn254::Fr;
use ark_ff::{PrimeField, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::rand::RngCore;
use jolt_dory::DoryCommitment;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    bucketed_ip::{BvaInteger, PaddingWorkCounters, StreamingEoProfile, TensorBasisVector},
    hidden_packed_eo::{
        prove_hidden_bva_i16_opening_profiled, prove_hidden_bva_opening_profiled,
        prove_hidden_packed_shortint_reduction, verify_hidden_packed_i8_reduction,
        verify_hidden_shortint_bva_opening_profiled, HiddenBvaOpeningProof,
        HiddenBvaOpeningProverProfile, HiddenBvaOpeningVerifierProfile, HiddenPackedEoError,
        HiddenPackedEoProof, HiddenPackedEoProverError, HiddenPackedEoProverState,
    },
    jolt_dory::{CommittedDoryBva, DoryProverParameters, DoryVerifierParameters},
    matrix_collection::DenseMatrixBva,
    section6_claim_collection::{Section6ProverClaimCollection, Section6VerifierClaimCollection},
    section6_data::{
        monomial_weight, validate_claims, Section6DataError, Section6FinalWitness, Section6Layout,
        Section6MatrixClaim, Section6PointWitness,
    },
    section6_final_opening::{
        commit_section6_bva_i16_on_grid_zk, commit_section6_bva_i8_on_grid_zk,
        prove_section6_final_opening_profiled, verify_section6_final_opening_profiled,
        Section6DoryGrid, Section6FinalOpeningError, Section6FinalOpeningProof,
        Section6FinalOpeningProverProfile, Section6FinalOpeningVerifierProfile,
    },
    validity_final::{
        build_final_prover_plan, build_final_verifier_plan, commit_final_aux_on_grid_zk,
        prove_final_evaluation_deferred, verify_final_evaluation_deferred, FinalAuxColumns,
        FinalEvaluationError, FinalEvaluationProof, FinalPublicTables,
    },
    validity_point::{
        build_point_prover_plan, build_point_verifier_plan, commit_point_aux_on_grid_zk,
        commit_point_keys_on_grid_zk, commit_point_matrix_on_grid_zk, commit_tbv_on_grid_zk,
        derive_point_challenges, prove_point_compilation_deferred,
        verify_point_compilation_deferred, PointAuxColumns, PointCompilationError,
        PointCompilationProof, PointKeyColumns, PointMatrixStatement, PointPublicTables,
        TbvStatement,
    },
    validity_shape::{
        commit_shape_masks_from_descriptor_on_grid_zk, prove_shape_validity,
        shape_validity_aux_padded_cols, verify_shape_validity, CommittedShapeMasks,
        ShapeCommitStatement, ShapeValidityError, ShapeValidityProof,
    },
};

const TERMINAL_OPENING_ROLE: &[u8] = b"validity-terminal-opening-v1";

/// Public commit-phase statement shared with every verifier.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ValidityCommitStatement {
    pub layout: Section6Layout,
    pub shape: ShapeCommitStatement,
    pub bva_commitment: DoryCommitment,
}

impl ValidityCommitStatement {
    /// The mask commitment plus the dense BVA commitment.
    pub fn physical_dory_commitment_count(&self) -> usize {
        2
    }
}

/// Prover-side committed state retained across openings.
pub struct CommittedOpeningCollection {
    statement: ValidityCommitStatement,
    shapes: CommittedShapeMasks,
    bva: CommittedDoryBva,
}

impl CommittedOpeningCollection {
    pub fn statement(&self) -> &ValidityCommitStatement {
        &self.statement
    }

    pub fn shapes(&self) -> &CommittedShapeMasks {
        &self.shapes
    }

    pub fn bva(&self) -> &CommittedDoryBva {
        &self.bva
    }
}

/// Size-private committed state carrying the optional one-time validity proof.
pub struct CommittedValidityCollection {
    opening: CommittedOpeningCollection,
    validity: ShapeValidityProof,
}

impl CommittedValidityCollection {
    pub fn statement(&self) -> &ValidityCommitStatement {
        self.opening.statement()
    }

    /// The one-time shape validity proof shipped alongside the statement.
    pub fn validity(&self) -> &ShapeValidityProof {
        &self.validity
    }

    pub fn shapes(&self) -> &CommittedShapeMasks {
        self.opening.shapes()
    }

    pub fn bva(&self) -> &CommittedDoryBva {
        self.opening.bva()
    }

    pub fn opening(&self) -> &CommittedOpeningCollection {
        &self.opening
    }
}

/// Commit-phase timings for the persistent commitments and their proof.
#[derive(Clone, Copy, Debug)]
pub struct ValidityCommitProfile {
    pub shape_commit_seconds: f64,
    pub validity_prove_seconds: f64,
    pub bva_commit_seconds: f64,
}

/// Serializable opening proof; only the BVA opening and the terminal batch
/// contain physical Dory openings.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ValidityOpeningProof {
    pub x: PointMatrixStatement,
    pub tbv: TbvStatement,
    pub point: PointCompilationProof,
    pub packed_eo: HiddenPackedEoProof,
    pub final_eval: FinalEvaluationProof,
    pub bva_opening: HiddenBvaOpeningProof,
    pub terminal_opening: Section6FinalOpeningProof,
}

impl ValidityOpeningProof {
    /// One dense BVA opening plus one terminal auxiliary batch opening.
    pub fn physical_dory_proof_count(&self) -> usize {
        1 + self.terminal_opening.physical_dory_proof_count()
    }

    /// `X`, the TBV, the compilation auxiliary, the key matrix, and the
    /// final auxiliary each add one GT element.
    pub fn auxiliary_dory_commitment_count(&self) -> usize {
        5
    }
}

/// Prover profile for the Packed EO stage and the terminal claim batch.
#[derive(Clone, Debug)]
pub struct ValidityOpeningProfile {
    pub padding: PaddingWorkCounters,
    pub packed_eo: StreamingEoProfile,
    pub bva_open: HiddenBvaOpeningProverProfile,
    pub terminal_open: Section6FinalOpeningProverProfile,
    pub validation_seconds: f64,
    pub batch_point_seconds: f64,
    pub x_commit_seconds: f64,
    pub point_witness_seconds: f64,
    pub tbv_commit_seconds: f64,
    pub point_aux_commit_seconds: f64,
    pub keys_commit_seconds: f64,
    pub point_prove_seconds: f64,
    pub packed_binding_seconds: f64,
    pub tbv_reconstruct_seconds: f64,
    pub packed_eo_seconds: f64,
    pub virtual_tbv_drop_seconds: f64,
    pub bva_open_seconds: f64,
    pub point_witness_drop_seconds: f64,
    pub final_witness_seconds: f64,
    pub final_aux_commit_seconds: f64,
    pub final_witness_drop_seconds: f64,
    pub final_prove_seconds: f64,
    pub terminal_claim_collection_seconds: f64,
    pub terminal_open_seconds: f64,
    pub result_assembly_seconds: f64,
    pub terminal_claim_count: usize,
}

/// Public verifier output after the terminal Dory proof accepts.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ValidityOpeningVerification {
    /// `c = sum_tau xi_tau(batch point) u_tau`, recomputed from public data.
    pub aggregated_claim: Fr,
    pub terminal_claim_count: usize,
}

/// Mutually exclusive verifier timings for one Section 6 opening.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct ValidityOpeningVerifierProfile {
    pub bva_open: HiddenBvaOpeningVerifierProfile,
    pub terminal_open: Section6FinalOpeningVerifierProfile,
    pub statement_validation_seconds: f64,
    pub batch_point_seconds: f64,
    pub point_prepare_seconds: f64,
    pub point_verify_seconds: f64,
    pub packed_eo_verify_seconds: f64,
    pub bva_open_verify_seconds: f64,
    pub final_prepare_seconds: f64,
    pub final_verify_seconds: f64,
    pub terminal_claim_collection_seconds: f64,
    pub terminal_open_verify_seconds: f64,
    pub result_assembly_seconds: f64,
    pub total_seconds: f64,
}

#[derive(Debug)]
pub enum ValidityProtocolError {
    Data(Section6DataError),
    Shape(ShapeValidityError),
    Point(PointCompilationError),
    Final(FinalEvaluationError),
    PackedProver(HiddenPackedEoProverError<Infallible>),
    PackedVerifier(HiddenPackedEoError),
    Opening(Section6FinalOpeningError),
    ClaimShape,
    Encoding(String),
    LengthOverflow,
}

impl Display for ValidityProtocolError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for ValidityProtocolError {}

impl From<Section6DataError> for ValidityProtocolError {
    fn from(error: Section6DataError) -> Self {
        Self::Data(error)
    }
}

impl From<ShapeValidityError> for ValidityProtocolError {
    fn from(error: ShapeValidityError) -> Self {
        Self::Shape(error)
    }
}

impl From<PointCompilationError> for ValidityProtocolError {
    fn from(error: PointCompilationError) -> Self {
        Self::Point(error)
    }
}

impl From<FinalEvaluationError> for ValidityProtocolError {
    fn from(error: FinalEvaluationError) -> Self {
        Self::Final(error)
    }
}

impl From<HiddenPackedEoProverError<Infallible>> for ValidityProtocolError {
    fn from(error: HiddenPackedEoProverError<Infallible>) -> Self {
        Self::PackedProver(error)
    }
}

impl From<HiddenPackedEoError> for ValidityProtocolError {
    fn from(error: HiddenPackedEoError) -> Self {
        Self::PackedVerifier(error)
    }
}

impl From<Section6FinalOpeningError> for ValidityProtocolError {
    fn from(error: Section6FinalOpeningError) -> Self {
        Self::Opening(error)
    }
}

/// Native short-integer storage supported by the Section 6 prover path.
pub trait Section6Integer: BvaInteger {
    fn commit_bva_on_grid(
        parameters: &DoryProverParameters,
        grid: &Section6DoryGrid,
        source: &DenseMatrixBva<Self>,
        bit_width: usize,
    ) -> Result<CommittedDoryBva, ValidityProtocolError>;

    #[allow(clippy::too_many_arguments)]
    fn prove_packed<R: RngCore>(
        parameters: &DoryProverParameters,
        source: &mut DenseMatrixBva<Self>,
        tbv: &TensorBasisVector<Fr>,
        claimed: Fr,
        bit_width: usize,
        matrix_commitment: &DoryCommitment,
        outer_binding: &[u8],
        rng: &mut R,
    ) -> Result<
        (
            HiddenPackedEoProof,
            HiddenPackedEoProverState,
            PaddingWorkCounters,
            StreamingEoProfile,
        ),
        ValidityProtocolError,
    >;

    fn open_bva<R: RngCore>(
        parameters: &DoryProverParameters,
        source: &DenseMatrixBva<Self>,
        committed: &CommittedDoryBva,
        packed: &HiddenPackedEoProverState,
        rng: &mut R,
    ) -> Result<(HiddenBvaOpeningProof, HiddenBvaOpeningProverProfile), ValidityProtocolError>;
}

impl Section6Integer for i8 {
    fn commit_bva_on_grid(
        parameters: &DoryProverParameters,
        grid: &Section6DoryGrid,
        source: &DenseMatrixBva<Self>,
        bit_width: usize,
    ) -> Result<CommittedDoryBva, ValidityProtocolError> {
        Ok(commit_section6_bva_i8_on_grid_zk(
            parameters, grid, source, bit_width,
        )?)
    }

    fn prove_packed<R: RngCore>(
        parameters: &DoryProverParameters,
        source: &mut DenseMatrixBva<Self>,
        tbv: &TensorBasisVector<Fr>,
        claimed: Fr,
        bit_width: usize,
        matrix_commitment: &DoryCommitment,
        outer_binding: &[u8],
        rng: &mut R,
    ) -> Result<
        (
            HiddenPackedEoProof,
            HiddenPackedEoProverState,
            PaddingWorkCounters,
            StreamingEoProfile,
        ),
        ValidityProtocolError,
    > {
        Ok(prove_hidden_packed_shortint_reduction::<i8, _>(
            parameters,
            source,
            tbv,
            claimed,
            bit_width,
            matrix_commitment,
            outer_binding,
            rng,
        )?)
    }

    fn open_bva<R: RngCore>(
        parameters: &DoryProverParameters,
        source: &DenseMatrixBva<Self>,
        committed: &CommittedDoryBva,
        packed: &HiddenPackedEoProverState,
        rng: &mut R,
    ) -> Result<(HiddenBvaOpeningProof, HiddenBvaOpeningProverProfile), ValidityProtocolError> {
        Ok(prove_hidden_bva_opening_profiled(
            parameters, source, committed, packed, rng,
        )?)
    }
}

impl Section6Integer for i16 {
    fn commit_bva_on_grid(
        parameters: &DoryProverParameters,
        grid: &Section6DoryGrid,
        source: &DenseMatrixBva<Self>,
        bit_width: usize,
    ) -> Result<CommittedDoryBva, ValidityProtocolError> {
        Ok(commit_section6_bva_i16_on_grid_zk(
            parameters, grid, source, bit_width,
        )?)
    }

    fn prove_packed<R: RngCore>(
        parameters: &DoryProverParameters,
        source: &mut DenseMatrixBva<Self>,
        tbv: &TensorBasisVector<Fr>,
        claimed: Fr,
        bit_width: usize,
        matrix_commitment: &DoryCommitment,
        outer_binding: &[u8],
        rng: &mut R,
    ) -> Result<
        (
            HiddenPackedEoProof,
            HiddenPackedEoProverState,
            PaddingWorkCounters,
            StreamingEoProfile,
        ),
        ValidityProtocolError,
    > {
        Ok(prove_hidden_packed_shortint_reduction::<i16, _>(
            parameters,
            source,
            tbv,
            claimed,
            bit_width,
            matrix_commitment,
            outer_binding,
            rng,
        )?)
    }

    fn open_bva<R: RngCore>(
        parameters: &DoryProverParameters,
        source: &DenseMatrixBva<Self>,
        committed: &CommittedDoryBva,
        packed: &HiddenPackedEoProverState,
        rng: &mut R,
    ) -> Result<(HiddenBvaOpeningProof, HiddenBvaOpeningProverProfile), ValidityProtocolError> {
        Ok(prove_hidden_bva_i16_opening_profiled(
            parameters, source, committed, packed, rng,
        )?)
    }
}

/// Minimum Dory grid covering every auxiliary matrix and the BlindFold rows
/// of the identity batches. The dense BVA is committed and opened separately
/// on its native source-sized grid.
pub fn validity_auxiliary_dory_log(
    layout: &Section6Layout,
) -> Result<usize, ValidityProtocolError> {
    layout.validate()?;
    let rows = layout.row_cover()?;
    let pair = layout.pair_width();
    let mask_cols = (layout.eta_width() + pair)
        .checked_next_power_of_two()
        .ok_or(ValidityProtocolError::LengthOverflow)?;
    let shape_aux_cols = shape_validity_aux_padded_cols(layout);
    let tbv_cols = (1 + pair)
        .checked_next_power_of_two()
        .ok_or(ValidityProtocolError::LengthOverflow)?;
    let point_aux_cols = PointAuxColumns::new(pair, layout.eta_width()).padded_cols();
    let key_cols = PointKeyColumns::padded_cols();
    let final_cols = FinalAuxColumns::new(pair).padded_cols();
    let widest_cols = mask_cols
        .max(shape_aux_cols)
        .max(tbv_cols)
        .max(point_aux_cols)
        .max(key_cols)
        .max(final_cols);
    let auxiliary_len = rows
        .checked_mul(widest_cols)
        .ok_or(ValidityProtocolError::LengthOverflow)?;
    let auxiliary_log = auxiliary_len.ilog2() as usize;

    // The widest identity in either batch has degree 3, and the hidden
    // relation wrapper reserves two additional Pedersen rows; eight BlindFold
    // rows therefore cover every relation, requiring a grid of at least 2^6.
    let blindfold_grid_log = 6;
    Ok(auxiliary_log.max(blindfold_grid_log).max(2))
}

/// Minimum setup size covering the independent BVA and auxiliary grids.
pub fn validity_required_dory_log(layout: &Section6Layout) -> Result<usize, ValidityProtocolError> {
    let auxiliary_log = validity_auxiliary_dory_log(layout)?;
    let source_log = validity_bva_dory_log(layout)?;
    Ok(source_log.max(auxiliary_log).max(2))
}

fn validity_bva_dory_log(layout: &Section6Layout) -> Result<usize, ValidityProtocolError> {
    layout.validate()?;
    Ok((layout.logical_len.ilog2() as usize).max(2))
}

/// Commit the shape masks and BVA, then produce the size-private validity proof.
pub fn commit_validity_collection<Integer: Section6Integer>(
    parameters: &DoryProverParameters,
    source: &DenseMatrixBva<Integer>,
    bit_width: usize,
    rng: &mut impl RngCore,
) -> Result<CommittedValidityCollection, ValidityProtocolError> {
    Ok(commit_validity_collection_profiled(parameters, source, bit_width, rng)?.0)
}

/// Commit the persistent state and retain separate stage timings.
pub fn commit_validity_collection_profiled<Integer: Section6Integer>(
    parameters: &DoryProverParameters,
    source: &DenseMatrixBva<Integer>,
    bit_width: usize,
    rng: &mut impl RngCore,
) -> Result<(CommittedValidityCollection, ValidityCommitProfile), ValidityProtocolError> {
    let (opening, mut profile) = commit_opening_collection_profiled(parameters, source, bit_width)?;
    let (validity, validity_prove_seconds) =
        prove_shape_validity_for_collection(parameters, &opening, rng)?;
    profile.validity_prove_seconds = validity_prove_seconds;
    Ok((CommittedValidityCollection { opening, validity }, profile))
}

/// Commit the persistent shape masks and BVA without a shape validity proof.
pub fn commit_opening_collection<Integer: Section6Integer>(
    parameters: &DoryProverParameters,
    source: &DenseMatrixBva<Integer>,
    bit_width: usize,
) -> Result<CommittedOpeningCollection, ValidityProtocolError> {
    Ok(commit_opening_collection_profiled(parameters, source, bit_width)?.0)
}

/// Commit the opening state and retain timings, with validity proof time set to zero.
pub fn commit_opening_collection_profiled<Integer: Section6Integer>(
    parameters: &DoryProverParameters,
    source: &DenseMatrixBva<Integer>,
    bit_width: usize,
) -> Result<(CommittedOpeningCollection, ValidityCommitProfile), ValidityProtocolError> {
    let layout = Section6Layout::from_descriptor(source.descriptor(), bit_width)?;
    let auxiliary_grid = Section6DoryGrid::new(validity_auxiliary_dory_log(&layout)?)?;
    let bva_grid = Section6DoryGrid::new(validity_bva_dory_log(&layout)?)?;

    let shape_started = Instant::now();
    let shapes = commit_shape_masks_from_descriptor_on_grid_zk(
        parameters,
        &auxiliary_grid,
        source.descriptor(),
        bit_width,
    )?;
    let shape_commit_seconds = shape_started.elapsed().as_secs_f64();

    let bva_started = Instant::now();
    let bva = Integer::commit_bva_on_grid(parameters, &bva_grid, source, bit_width)?;
    let bva_commit_seconds = bva_started.elapsed().as_secs_f64();

    let statement = ValidityCommitStatement {
        layout: shapes.layout().clone(),
        shape: shapes.statement().clone(),
        bva_commitment: bva.commitment.clone(),
    };
    Ok((
        CommittedOpeningCollection {
            statement,
            shapes,
            bva,
        },
        ValidityCommitProfile {
            shape_commit_seconds,
            validity_prove_seconds: 0.0,
            bva_commit_seconds,
        },
    ))
}

/// Prove shape validity explicitly for size-private collection registration.
pub fn prove_shape_validity_for_collection(
    parameters: &DoryProverParameters,
    committed: &CommittedOpeningCollection,
    rng: &mut impl RngCore,
) -> Result<(ShapeValidityProof, f64), ValidityProtocolError> {
    validate_committed_state(committed)?;
    let grid = Section6DoryGrid::new(validity_auxiliary_dory_log(&committed.statement.layout)?)?;
    let started = Instant::now();
    let proof = prove_shape_validity(parameters, &grid, &committed.shapes, rng)?;
    Ok((proof, started.elapsed().as_secs_f64()))
}

/// Verify the one-time shape validity proof against the public statement.
pub fn verify_shape_commitment(
    parameters: &DoryVerifierParameters,
    statement: &ValidityCommitStatement,
    validity: &ShapeValidityProof,
) -> Result<(), ValidityProtocolError> {
    let grid = Section6DoryGrid::new(validity_auxiliary_dory_log(&statement.layout)?)?;
    verify_shape_validity(
        parameters,
        &grid,
        &statement.layout,
        &statement.shape,
        validity,
    )?;
    Ok(())
}

/// Execute the complete prover flow with a streamed byte opening.
pub fn prove_validity_opening<Integer: Section6Integer>(
    parameters: &DoryProverParameters,
    committed: &CommittedValidityCollection,
    source: &mut DenseMatrixBva<Integer>,
    claims: &[Section6MatrixClaim<Fr>],
    rng: &mut impl RngCore,
) -> Result<(ValidityOpeningProof, ValidityOpeningProfile), ValidityProtocolError> {
    prove_opening(parameters, committed.opening(), source, claims, rng)
}

/// Execute the complete opening flow without generating a shape validity proof.
pub fn prove_opening<Integer: Section6Integer>(
    parameters: &DoryProverParameters,
    committed: &CommittedOpeningCollection,
    source: &mut DenseMatrixBva<Integer>,
    claims: &[Section6MatrixClaim<Fr>],
    rng: &mut impl RngCore,
) -> Result<(ValidityOpeningProof, ValidityOpeningProfile), ValidityProtocolError> {
    let validation_started = Instant::now();
    let layout = committed.shapes.layout().clone();
    layout.validate_descriptor(source.descriptor())?;
    validate_claims(&layout, claims)?;
    validate_committed_state(committed)?;
    let grid = Section6DoryGrid::new(validity_auxiliary_dory_log(&layout)?)?;
    let validation_seconds = validation_started.elapsed().as_secs_f64();

    // Commit the claim points before the batching point is known.
    let x_commit_started = Instant::now();
    let x = commit_point_matrix_on_grid_zk(parameters, &grid, &layout, claims)?;
    let x_commit_seconds = x_commit_started.elapsed().as_secs_f64();
    let batch_point_started = Instant::now();
    let claimed_values = claims.iter().map(|claim| claim.value).collect::<Vec<_>>();
    let (batch_point, batch_binding) =
        derive_batch_point(&committed.statement, &claimed_values, x.statement())?;
    let aggregated_claim = aggregate_claim_values(&claimed_values, &batch_point);
    let batch_point_seconds = batch_point_started.elapsed().as_secs_f64();

    // Compile the witness once; the TBV, the compilation auxiliary, and the
    // key columns are all views over it.
    let point_witness_started = Instant::now();
    let point_witness = Section6PointWitness::compile(
        &layout,
        source.descriptor(),
        committed.shapes.metadata(),
        claims,
        &batch_point,
    )?;
    let point_witness_seconds = point_witness_started.elapsed().as_secs_f64();
    debug_assert_eq!(point_witness.aggregated_claim, aggregated_claim);

    let tbv_commit_started = Instant::now();
    let tbv = commit_tbv_on_grid_zk(parameters, &grid, &layout, &point_witness)?;
    let tbv_commit_seconds = tbv_commit_started.elapsed().as_secs_f64();
    let point_aux_commit_started = Instant::now();
    let aux = commit_point_aux_on_grid_zk(parameters, &grid, &layout, &point_witness)?;
    let point_aux_commit_seconds = point_aux_commit_started.elapsed().as_secs_f64();
    let keys_commit_started = Instant::now();
    let keys = commit_point_keys_on_grid_zk(
        parameters,
        &grid,
        &batch_binding,
        &committed.shapes,
        &x,
        &tbv,
        &aux,
        &batch_point,
    )?;
    let keys_commit_seconds = keys_commit_started.elapsed().as_secs_f64();

    // Prove the compilation identities and lookups with deferred openings.
    let point_prove_started = Instant::now();
    let (challenges, _) = derive_point_challenges(
        &batch_binding,
        &layout,
        committed.shapes.statement(),
        x.statement(),
        tbv.statement(),
        aux.statement(),
        &batch_point,
    )?;
    let point_publics = PointPublicTables::derive(&layout, &challenges, true)?;
    let point_plan = build_point_prover_plan(
        &committed.shapes,
        &x,
        &tbv,
        &aux,
        &keys,
        &point_publics,
        &challenges,
        &batch_point,
    )?;
    let (point_proof, point_openings) = prove_point_compilation_deferred(
        &batch_binding,
        parameters,
        &grid,
        &committed.shapes,
        &x,
        &tbv,
        &aux,
        &keys,
        &point_publics,
        &batch_point,
        rng,
    )?;
    let point_prove_seconds = point_prove_started.elapsed().as_secs_f64();

    // Packed EO against the public aggregated claim over the virtual TBV.
    let packed_binding_started = Instant::now();
    let packed_binding = packed_stage_binding(&batch_binding, tbv.statement(), &point_proof)?;
    let packed_binding_seconds = packed_binding_started.elapsed().as_secs_f64();
    let tbv_reconstruct_started = Instant::now();
    let virtual_tbv = point_witness.reconstruct_tbv(&layout)?;
    let tbv_reconstruct_seconds = tbv_reconstruct_started.elapsed().as_secs_f64();
    let packed_eo_started = Instant::now();
    let (packed_eo, packed_state, padding, packed_profile) = Integer::prove_packed(
        parameters,
        source,
        &virtual_tbv,
        aggregated_claim,
        layout.bit_width,
        &committed.statement.bva_commitment,
        &packed_binding,
        rng,
    )?;
    let packed_eo_seconds = packed_eo_started.elapsed().as_secs_f64();
    let virtual_tbv_drop_started = Instant::now();
    drop(virtual_tbv);
    let virtual_tbv_drop_seconds = virtual_tbv_drop_started.elapsed().as_secs_f64();
    let bva_open_started = Instant::now();
    let (bva_opening, bva_open_profile) =
        Integer::open_bva(parameters, source, &committed.bva, &packed_state, rng)?;
    let bva_open_seconds = bva_open_started.elapsed().as_secs_f64();

    // TBV evaluation identities at the Packed EO point with deferred
    // openings; the committed chain ends in the claimed per-bucket values.
    let final_witness_started = Instant::now();
    let final_witness =
        Section6FinalWitness::compile(&layout, &point_witness, &packed_state.challenges)?;
    let final_witness_seconds = final_witness_started.elapsed().as_secs_f64();
    let point_witness_drop_started = Instant::now();
    drop(point_witness);
    let point_witness_drop_seconds = point_witness_drop_started.elapsed().as_secs_f64();
    let final_aux_commit_started = Instant::now();
    let final_aux = commit_final_aux_on_grid_zk(parameters, &grid, &layout, &final_witness)?;
    let final_aux_commit_seconds = final_aux_commit_started.elapsed().as_secs_f64();
    let final_witness_drop_started = Instant::now();
    drop(final_witness);
    let final_witness_drop_seconds = final_witness_drop_started.elapsed().as_secs_f64();
    let final_prove_started = Instant::now();
    let final_publics = FinalPublicTables::derive(&layout, &packed_state.challenges, true)?;
    let final_plan = build_final_prover_plan(
        &layout,
        &tbv,
        &final_aux,
        &final_publics,
        &packed_state.challenges,
    )?;
    let (final_eval, final_openings) = prove_final_evaluation_deferred(
        &batch_binding,
        parameters,
        &layout,
        &tbv,
        &final_aux,
        Some(&point_plan),
        &final_plan,
        &packed_state,
        rng,
    )?;
    let final_prove_seconds = final_prove_started.elapsed().as_secs_f64();

    // Collect every deferred claim - the identity batches, the lookups, and
    // all LogUp auxiliary-vector openings - into one terminal Dory opening.
    let terminal_claim_collection_started = Instant::now();
    let mut terminal_claims = Section6ProverClaimCollection::new();
    // The compiled and tested joint reduction includes every Point identity.
    let lookup_openings = point_openings
        .lookup
        .relation_openings()
        .map_err(PointCompilationError::LogUp)?;
    for opening in &lookup_openings {
        terminal_claims.push_relation(opening, &grid)?;
    }
    terminal_claims.push_relation(&final_openings.joint_identity, &grid)?;
    let v_claim = final_openings.resolve_v_terminal_claim(&final_aux, &grid)?;
    let terminal_binding =
        terminal_stage_binding(&packed_binding, &packed_eo, &final_eval, &bva_opening)?;
    let mut final_claims = terminal_claims.claims();
    final_claims.push(v_claim.as_claim(&final_openings.average));
    let terminal_claim_count = final_claims.len();
    let terminal_claim_collection_seconds =
        terminal_claim_collection_started.elapsed().as_secs_f64();
    let terminal_open_started = Instant::now();
    let (terminal_opening, terminal_open_profile) = prove_section6_final_opening_profiled(
        TERMINAL_OPENING_ROLE,
        &terminal_binding,
        parameters,
        &grid,
        &final_claims,
        rng,
    )?;
    let terminal_open_seconds = terminal_open_started.elapsed().as_secs_f64();

    let result_assembly_started = Instant::now();
    let proof = ValidityOpeningProof {
        x: x.statement().clone(),
        tbv: tbv.statement().clone(),
        point: point_proof,
        packed_eo,
        final_eval,
        bva_opening,
        terminal_opening,
    };
    let result_assembly_seconds = result_assembly_started.elapsed().as_secs_f64();
    Ok((
        proof,
        ValidityOpeningProfile {
            padding,
            packed_eo: packed_profile,
            bva_open: bva_open_profile,
            terminal_open: terminal_open_profile,
            validation_seconds,
            batch_point_seconds,
            x_commit_seconds,
            point_witness_seconds,
            tbv_commit_seconds,
            point_aux_commit_seconds,
            keys_commit_seconds,
            point_prove_seconds,
            packed_binding_seconds,
            tbv_reconstruct_seconds,
            packed_eo_seconds,
            virtual_tbv_drop_seconds,
            bva_open_seconds,
            point_witness_drop_seconds,
            final_witness_seconds,
            final_aux_commit_seconds,
            final_witness_drop_seconds,
            final_prove_seconds,
            terminal_claim_collection_seconds,
            terminal_open_seconds,
            result_assembly_seconds,
            terminal_claim_count,
        },
    ))
}

/// Verify the complete flow without shape data, BVA bytes, or claim points.
///
/// The one-time shape validity proof is verified separately by
/// [`verify_shape_commitment`].
pub fn verify_validity_opening(
    parameters: &DoryVerifierParameters,
    statement: &ValidityCommitStatement,
    claimed_values: &[Fr],
    proof: &ValidityOpeningProof,
) -> Result<ValidityOpeningVerification, ValidityProtocolError> {
    Ok(verify_validity_opening_profiled(parameters, statement, claimed_values, proof)?.0)
}

/// Verify an opening and retain mutually exclusive stage timings.
pub fn verify_validity_opening_profiled(
    parameters: &DoryVerifierParameters,
    statement: &ValidityCommitStatement,
    claimed_values: &[Fr],
    proof: &ValidityOpeningProof,
) -> Result<(ValidityOpeningVerification, ValidityOpeningVerifierProfile), ValidityProtocolError> {
    let total_started = Instant::now();
    let statement_validation_started = Instant::now();
    let layout = &statement.layout;
    layout.validate()?;
    statement.shape.validate(layout)?;
    if claimed_values.len() != layout.matrix_count {
        return Err(ValidityProtocolError::ClaimShape);
    }
    let grid = Section6DoryGrid::new(validity_auxiliary_dory_log(layout)?)?;
    proof.x.validate(layout)?;
    proof.tbv.validate(layout)?;
    let statement_validation_seconds = statement_validation_started.elapsed().as_secs_f64();

    let batch_point_started = Instant::now();
    let (batch_point, batch_binding) = derive_batch_point(statement, claimed_values, &proof.x)?;
    let aggregated_claim = aggregate_claim_values(claimed_values, &batch_point);
    let batch_point_seconds = batch_point_started.elapsed().as_secs_f64();

    let point_prepare_started = Instant::now();
    let (challenges, _) = derive_point_challenges(
        &batch_binding,
        layout,
        &statement.shape,
        &proof.x,
        &proof.tbv,
        &proof.point.aux,
        &batch_point,
    )?;
    let point_publics = PointPublicTables::derive(layout, &challenges, false)?;
    let point_plan = build_point_verifier_plan(
        layout,
        &statement.shape,
        &proof.x,
        &proof.tbv,
        &proof.point.aux,
        &proof.point.keys,
        &point_publics,
        &challenges,
        &batch_point,
    )?;
    let point_prepare_seconds = point_prepare_started.elapsed().as_secs_f64();
    let point_verify_started = Instant::now();
    let point_openings = verify_point_compilation_deferred(
        &batch_binding,
        parameters,
        &grid,
        layout,
        &statement.shape,
        &proof.x,
        &proof.tbv,
        &point_publics,
        &batch_point,
        &proof.point,
    )?;
    let point_verify_seconds = point_verify_started.elapsed().as_secs_f64();

    let packed_eo_verify_started = Instant::now();
    let packed_binding = packed_stage_binding(&batch_binding, &proof.tbv, &proof.point)?;
    let packed = verify_hidden_packed_i8_reduction(
        parameters,
        &statement.bva_commitment,
        layout.logical_len,
        aggregated_claim,
        layout.bit_width,
        &packed_binding,
        &proof.packed_eo,
    )?;
    let packed_eo_verify_seconds = packed_eo_verify_started.elapsed().as_secs_f64();
    let bva_open_verify_started = Instant::now();
    let bva_open_profile = verify_hidden_shortint_bva_opening_profiled(
        parameters,
        &statement.bva_commitment,
        layout.logical_len,
        layout.bit_width,
        &packed,
        &proof.bva_opening,
    )?;
    let bva_open_verify_seconds = bva_open_verify_started.elapsed().as_secs_f64();

    let final_prepare_started = Instant::now();
    let final_publics = FinalPublicTables::derive(layout, &packed.challenges, false)?;
    let final_plan = build_final_verifier_plan(
        layout,
        &proof.tbv,
        &proof.final_eval.aux,
        &final_publics,
        &packed.challenges,
    )?;
    let final_prepare_seconds = final_prepare_started.elapsed().as_secs_f64();
    let final_verify_started = Instant::now();
    let final_openings = verify_final_evaluation_deferred(
        &batch_binding,
        parameters,
        &grid,
        layout,
        &proof.tbv,
        Some(&point_plan),
        &final_plan,
        &packed,
        &proof.final_eval,
    )?;
    let final_verify_seconds = final_verify_started.elapsed().as_secs_f64();

    let terminal_claim_collection_started = Instant::now();
    let mut terminal_claims = Section6VerifierClaimCollection::new();
    // The joint reduction verified below includes every Point identity.
    let lookup_openings = point_openings
        .lookup
        .relation_openings()
        .map_err(PointCompilationError::LogUp)?;
    for opening in &lookup_openings {
        terminal_claims.push_relation(opening, &grid)?;
    }
    terminal_claims.push_relation(&final_openings.joint_identity, &grid)?;
    let v_claim = final_openings.resolve_v_terminal_claim(&grid)?;
    let terminal_binding = terminal_stage_binding(
        &packed_binding,
        &proof.packed_eo,
        &proof.final_eval,
        &proof.bva_opening,
    )?;
    let mut final_claims = terminal_claims.claims();
    final_claims.push(v_claim.as_claim());
    let terminal_claim_count = final_claims.len();
    let terminal_claim_collection_seconds =
        terminal_claim_collection_started.elapsed().as_secs_f64();
    let terminal_open_verify_started = Instant::now();
    let terminal_open_profile = verify_section6_final_opening_profiled(
        TERMINAL_OPENING_ROLE,
        &terminal_binding,
        parameters,
        &grid,
        &final_claims,
        &proof.terminal_opening,
    )?;
    let terminal_open_verify_seconds = terminal_open_verify_started.elapsed().as_secs_f64();

    let result_assembly_started = Instant::now();
    let verification = ValidityOpeningVerification {
        aggregated_claim,
        terminal_claim_count,
    };
    let result_assembly_seconds = result_assembly_started.elapsed().as_secs_f64();
    let total_seconds = total_started.elapsed().as_secs_f64();
    Ok((
        verification,
        ValidityOpeningVerifierProfile {
            bva_open: bva_open_profile,
            terminal_open: terminal_open_profile,
            statement_validation_seconds,
            batch_point_seconds,
            point_prepare_seconds,
            point_verify_seconds,
            packed_eo_verify_seconds,
            bva_open_verify_seconds,
            final_prepare_seconds,
            final_verify_seconds,
            terminal_claim_collection_seconds,
            terminal_open_verify_seconds,
            result_assembly_seconds,
            total_seconds,
        },
    ))
}

fn validate_committed_state(
    committed: &CommittedOpeningCollection,
) -> Result<(), ValidityProtocolError> {
    if committed.shapes.statement() != &committed.statement.shape
        || committed.bva.commitment != committed.statement.bva_commitment
        || committed.shapes.layout() != &committed.statement.layout
    {
        return Err(ValidityProtocolError::ClaimShape);
    }
    Ok(())
}

/// `c = sum_j u_j * prod_{bit set in j} point_bit`, the tensor batching.
fn aggregate_claim_values(claimed_values: &[Fr], batch_point: &[Fr]) -> Fr {
    let mut aggregated = Fr::zero();
    for (index, value) in claimed_values.iter().enumerate() {
        aggregated += *value * monomial_weight(batch_point, index);
    }
    aggregated
}

/// Derive the public tensor batching point from everything committed so far.
fn derive_batch_point(
    statement: &ValidityCommitStatement,
    claimed_values: &[Fr],
    x_statement: &PointMatrixStatement,
) -> Result<(Vec<Fr>, [u8; 32]), ValidityProtocolError> {
    let mut hasher = Sha256::new();
    hasher.update(b"parcomm-validity-batch-point-v1");
    append_serialized(&mut hasher, statement)?;
    append_field_values(&mut hasher, claimed_values)?;
    append_serialized(&mut hasher, x_statement)?;
    let binding: [u8; 32] = hasher.finalize().into();
    let mut point = Vec::with_capacity(statement.layout.tau_log());
    for bit in 0..statement.layout.tau_log() {
        let mut counter = 0_u64;
        loop {
            let mut candidate = Sha256::new();
            candidate.update(b"parcomm-validity-batch-point-sample");
            candidate.update(binding);
            candidate.update((bit as u64).to_le_bytes());
            candidate.update(counter.to_le_bytes());
            let coordinate = Fr::from_le_bytes_mod_order(&candidate.finalize());
            if !coordinate.is_zero() {
                point.push(coordinate);
                break;
            }
            counter += 1;
        }
    }
    Ok((point, binding))
}

fn packed_stage_binding(
    batch_binding: &[u8; 32],
    tbv_statement: &TbvStatement,
    point: &PointCompilationProof,
) -> Result<[u8; 32], ValidityProtocolError> {
    let mut hasher = Sha256::new();
    hasher.update(b"parcomm-validity-packed-stage-v1");
    hasher.update(batch_binding);
    append_serialized(&mut hasher, tbv_statement)?;
    append_serialized(&mut hasher, point)?;
    Ok(hasher.finalize().into())
}

fn terminal_stage_binding(
    packed_binding: &[u8; 32],
    packed: &HiddenPackedEoProof,
    final_eval: &FinalEvaluationProof,
    bva: &HiddenBvaOpeningProof,
) -> Result<[u8; 32], ValidityProtocolError> {
    let mut hasher = Sha256::new();
    hasher.update(b"parcomm-validity-terminal-stage-v1");
    hasher.update(packed_binding);
    append_serialized(&mut hasher, packed)?;
    append_serialized(&mut hasher, final_eval)?;
    append_serialized(&mut hasher, bva)?;
    Ok(hasher.finalize().into())
}

fn append_serialized<T: Serialize>(
    hasher: &mut Sha256,
    value: &T,
) -> Result<(), ValidityProtocolError> {
    let encoded = bincode::serialize(value)
        .map_err(|error| ValidityProtocolError::Encoding(error.to_string()))?;
    hasher.update((encoded.len() as u64).to_le_bytes());
    hasher.update(encoded);
    Ok(())
}

fn append_field_values(hasher: &mut Sha256, values: &[Fr]) -> Result<(), ValidityProtocolError> {
    hasher.update((values.len() as u64).to_le_bytes());
    for value in values {
        let mut encoded = Vec::new();
        value
            .serialize_compressed(&mut encoded)
            .map_err(|error| ValidityProtocolError::Encoding(error.to_string()))?;
        hasher.update((encoded.len() as u64).to_le_bytes());
        hasher.update(encoded);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        jolt_dory::setup_dory_parameters,
        matrix_collection::{matrix_mle_evaluate, DenseI8Matrix, DenseMatrixBva},
    };

    fn sample_matrices(seed: u64, shapes: &[(usize, usize)]) -> Vec<DenseI8Matrix> {
        let mut rng = StdRng::seed_from_u64(seed);
        shapes
            .iter()
            .map(|&(rows, cols)| {
                let values = (0..rows * cols)
                    .map(|_| ((rng.next_u32() % 17) as i32 - 8) as i8)
                    .collect::<Vec<_>>();
                DenseI8Matrix::new((rows, cols), values).unwrap()
            })
            .collect()
    }

    fn point_for(matrix: &DenseI8Matrix, seed: u64, width: usize) -> (Vec<Fr>, Vec<Fr>) {
        let row_log = matrix.shape().0.next_power_of_two().ilog2() as usize;
        let col_log = matrix.shape().1.next_power_of_two().ilog2() as usize;
        assert!(row_log <= width && col_log <= width);
        let mut rng = StdRng::seed_from_u64(seed);
        let sample = |len: usize, rng: &mut StdRng| {
            (0..len)
                .map(|_| Fr::from((rng.next_u32() % 97 + 1) as u64))
                .collect::<Vec<_>>()
        };
        (sample(row_log, &mut rng), sample(col_log, &mut rng))
    }

    #[test]
    fn validity_protocol_round_trips() {
        let shapes = [(31_usize, 31_usize), (15, 13), (4, 4)];
        let matrices = sample_matrices(0xA11C_E5, &shapes);
        let mut source = DenseMatrixBva::from_matrices(&matrices).unwrap();
        let layout = Section6Layout::from_descriptor(source.descriptor(), 8).unwrap();
        let setup_log = validity_required_dory_log(&layout).unwrap();
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(setup_log);
        let mut rng = StdRng::seed_from_u64(0x5641_4c49_4449_5459);

        let (committed, commit_profile) =
            commit_opening_collection_profiled(&prover_parameters, &source, 8).unwrap();
        assert!(commit_profile.shape_commit_seconds >= 0.0);
        assert_eq!(commit_profile.validity_prove_seconds, 0.0);

        let (private_committed, private_profile) =
            commit_validity_collection_profiled(&prover_parameters, &source, 8, &mut rng).unwrap();
        assert!(private_profile.validity_prove_seconds >= 0.0);
        verify_shape_commitment(
            &verifier_parameters,
            private_committed.statement(),
            private_committed.validity(),
        )
        .unwrap();

        let m = committed.statement().layout.point_log();
        let claims = matrices
            .iter()
            .enumerate()
            .map(|(index, matrix)| {
                let (row_point, col_point) = point_for(matrix, 900 + index as u64, m);
                let value = matrix_mle_evaluate(matrix, &row_point, &col_point).unwrap();
                Section6MatrixClaim::from_unpadded(m, &row_point, &col_point, value).unwrap()
            })
            .collect::<Vec<_>>();
        let claimed_values = claims.iter().map(|claim| claim.value).collect::<Vec<_>>();

        let (proof, profile) = prove_opening(
            &prover_parameters,
            &committed,
            &mut source,
            &claims,
            &mut rng,
        )
        .unwrap();
        assert!(profile.terminal_claim_count > 0);
        assert_eq!(proof.physical_dory_proof_count(), 2);

        let verification = verify_validity_opening(
            &verifier_parameters,
            committed.statement(),
            &claimed_values,
            &proof,
        )
        .unwrap();
        assert_eq!(
            verification.terminal_claim_count,
            profile.terminal_claim_count
        );

        // A tampered claimed value must break verification.
        let mut bad_values = claimed_values.clone();
        bad_values[0] += Fr::from(1_u64);
        assert!(verify_validity_opening(
            &verifier_parameters,
            committed.statement(),
            &bad_values,
            &proof,
        )
        .is_err());

        // A tampered TBV commitment must break verification.
        let mut bad_proof = proof.clone();
        bad_proof.tbv.commitment = proof.x.commitment.clone();
        assert!(verify_validity_opening(
            &verifier_parameters,
            committed.statement(),
            &claimed_values,
            &bad_proof,
        )
        .is_err());

        // The single joint Point/Final identity reduction is transcript-bound.
        let mut bad_joint = proof.clone();
        bad_joint
            .final_eval
            .joint_identity
            .relation
            .blindfold_transcript_state[0] ^= 1;
        assert!(verify_validity_opening(
            &verifier_parameters,
            committed.statement(),
            &claimed_values,
            &bad_joint,
        )
        .is_err());
    }

    #[test]
    fn validity_protocol_round_trips_native_i16() {
        let shapes = [(3_usize, 3_usize), (2, 2)];
        let mut source = DenseMatrixBva::<i16>::from_i16_fn(&shapes, |matrix, row, col| {
            let magnitude = 17_000 + (matrix * 101 + row * 7 + col) as i16;
            if (row + col).is_multiple_of(2) {
                magnitude
            } else {
                -magnitude
            }
        })
        .unwrap();
        let layout = Section6Layout::from_descriptor(source.descriptor(), 16).unwrap();
        let setup_log = validity_required_dory_log(&layout).unwrap();
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(setup_log);
        let mut rng = StdRng::seed_from_u64(0x4931_3653_4543_5436);
        let committed = commit_opening_collection(&prover_parameters, &source, 16).unwrap();
        let point_log = layout.point_log();
        let claims = shapes
            .iter()
            .enumerate()
            .map(|(matrix, &(rows, cols))| {
                let row_point = (0..rows.next_power_of_two().ilog2() as usize)
                    .map(|coordinate| Fr::from((matrix * 19 + coordinate + 2) as u64))
                    .collect::<Vec<_>>();
                let col_point = (0..cols.next_power_of_two().ilog2() as usize)
                    .map(|coordinate| Fr::from((matrix * 23 + coordinate + 5) as u64))
                    .collect::<Vec<_>>();
                let value = source
                    .matrix_mle_evaluate(matrix, &row_point, &col_point)
                    .unwrap();
                Section6MatrixClaim::from_unpadded(point_log, &row_point, &col_point, value)
                    .unwrap()
            })
            .collect::<Vec<_>>();
        let claimed_values = claims.iter().map(|claim| claim.value).collect::<Vec<_>>();
        let (proof, profile) = prove_opening(
            &prover_parameters,
            &committed,
            &mut source,
            &claims,
            &mut rng,
        )
        .unwrap();
        let verification = verify_validity_opening(
            &verifier_parameters,
            committed.statement(),
            &claimed_values,
            &proof,
        )
        .unwrap();
        assert_eq!(
            verification.terminal_claim_count,
            profile.terminal_claim_count
        );
        assert_eq!(proof.physical_dory_proof_count(), 2);
    }
}
