//! Shape-mask commitment and its deferred validity reduction.
//!
//! The prover commits the two Boolean shape masks `[D_eta | D_Z]` as one
//! column-major Dory matrix and proves once that the masks encode aligned
//! dyadic rectangles that exactly cover the active region of every source
//! matrix:
//!
//! * one identity batch over the bucket rows: Booleanity of both masks,
//!   the offset/prefix alignment `D_eta o D_Z = 0` on the pair columns,
//!   the ones-prefix structure of `D_Z`, and the two grand-product chains
//!   accumulating the coverage test factors at the power-of-two points;
//! * one coverage sum identity comparing the masked grand products with
//!   the generating polynomial of the exact matrix dimensions `(mu, nu)`;
//! * one keyed power lookup binding `W_theta = theta^mu` and
//!   `W_vartheta = vartheta^nu` to the public tables `{(h, theta^h)}` and
//!   `{(h, vartheta^h)}` for `h in [0, 2^m]`.
//!
//! Every auxiliary column (the two prefix chains, `mu`, `nu`, and the two
//! power columns) lives in one committed matrix. The shape relations and the
//! two power lookup families are deferred to the opening protocol, where they
//! share the joint LogUp and terminal Dory batch with the point reductions.

use std::fmt::{Display, Formatter};

use ark_bn254::Fr;
use ark_ff::{Field, One, PrimeField, Zero};
use ark_std::rand::RngCore;
use jolt_dory::DoryCommitment;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    hidden_logup::{HiddenLogUpError, MixedLogUpColumn, MixedLogUpColumnStatement},
    hidden_logup_union::{HiddenLogUpUnionProverFamilyData, HiddenLogUpUnionVerifierFamilyData},
    hidden_relation::{
        prove_hidden_sum_relation_deferred_with_public,
        verify_hidden_sum_relation_deferred_with_public, HiddenRelationError,
        HiddenRelationProverOpening, HiddenRelationReductionProof, HiddenRelationTable,
        HiddenRelationTableStatement, HiddenRelationVerifierOpening, OwnedPublicColumn,
        RelationExpression,
    },
    hidden_relation_batch::{
        deduplicate_prover_hidden_tables, deduplicate_verifier_hidden_tables,
        prove_hidden_relations_batch_deferred, verify_hidden_relations_batch_deferred,
        HiddenRelationBatchProverItem, HiddenRelationBatchReductionProof,
        HiddenRelationBatchVerifierItem,
    },
    jolt_dory::{CommittedDoryFieldVector, DoryProverParameters, DoryVerifierParameters},
    matrix_collection::DenseBvaDescriptor,
    section6_data::{
        field_from_usize, PrivateShapeMetadata, Section6DataError, Section6Layout,
        Section6ShapeMasks,
    },
    section6_final_opening::{
        commit_section6_owned_field_matrix_on_grid_zk, Section6DoryGrid, Section6FinalOpeningError,
    },
    structured_column::{ColumnPattern, ColumnSegment, StructuredColumn, StructuredColumnError},
};

const SHAPE_DOMAIN: &[u8] = b"parcomm-shape-commit-v1";

/// Verifier-visible statement for the committed shape masks.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShapeCommitStatement {
    /// Commitment to the column-major mask matrix `[D_eta | D_Z]`.
    pub mask_commitment: DoryCommitment,
    /// Shared power-of-two bucket-row cover.
    pub padded_rows: usize,
    /// Width `q = 2m + d` of `D_eta`.
    pub eta_width: usize,
    /// Width `2m` of `D_Z`.
    pub pair_width: usize,
}

impl ShapeCommitStatement {
    pub fn mask_logical_cols(&self) -> usize {
        self.eta_width + self.pair_width
    }

    pub fn mask_padded_cols(&self) -> usize {
        self.mask_logical_cols().next_power_of_two()
    }

    pub fn validate(&self, layout: &Section6Layout) -> Result<(), Section6DataError> {
        if self.padded_rows != layout.row_cover()?
            || self.eta_width != layout.eta_width()
            || self.pair_width != layout.pair_width()
        {
            return Err(Section6DataError::InvalidLayout(
                "shape statement does not match the public layout".to_owned(),
            ));
        }
        Ok(())
    }

    /// One mask column: `[0, q)` addresses `D_eta`, `[q, q + 2m)` addresses
    /// `D_Z`.
    pub fn mask_column(&self, column: usize) -> HiddenRelationTableStatement<'_> {
        HiddenRelationTableStatement::matrix_column(
            &self.mask_commitment,
            self.padded_rows,
            self.mask_logical_cols(),
            self.mask_padded_cols(),
            column,
        )
    }

    pub fn d_eta_column(&self, coordinate: usize) -> HiddenRelationTableStatement<'_> {
        self.mask_column(coordinate)
    }

    pub fn d_z_column(&self, coordinate: usize) -> HiddenRelationTableStatement<'_> {
        self.mask_column(self.eta_width + coordinate)
    }
}

/// Prover-only committed shape masks.
///
/// The compact `u8` masks stay resident for later witness construction; the
/// field-element columns live only inside the shared Dory matrix commitment.
pub struct CommittedShapeMasks {
    layout: Section6Layout,
    metadata: PrivateShapeMetadata,
    masks: Section6ShapeMasks,
    /// Canonical zero-extension dimensions of every source matrix.
    matrix_dims: Vec<(usize, usize)>,
    statement: ShapeCommitStatement,
    mask_columns: Vec<CommittedDoryFieldVector>,
}

impl CommittedShapeMasks {
    pub fn layout(&self) -> &Section6Layout {
        &self.layout
    }

    pub fn metadata(&self) -> &PrivateShapeMetadata {
        &self.metadata
    }

    pub fn masks(&self) -> &Section6ShapeMasks {
        &self.masks
    }

    pub fn matrix_dims(&self) -> &[(usize, usize)] {
        &self.matrix_dims
    }

    pub fn statement(&self) -> &ShapeCommitStatement {
        &self.statement
    }

    /// One mask column as a prover relation table; the index convention
    /// matches [`ShapeCommitStatement::mask_column`].
    pub fn mask_table(&self, column: usize) -> Option<HiddenRelationTable<'_>> {
        let committed = self.mask_columns.get(column)?;
        Some(HiddenRelationTable {
            values: committed.matrix_column_values()?,
            committed,
        })
    }

    pub fn d_eta_table(&self, coordinate: usize) -> Option<HiddenRelationTable<'_>> {
        self.mask_table(coordinate)
    }

    pub fn d_z_table(&self, coordinate: usize) -> Option<HiddenRelationTable<'_>> {
        self.mask_table(self.layout.eta_width() + coordinate)
    }
}

/// Serializable shape-validity reduction with no lookup or PCS opening.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShapeValidityProof {
    /// Commitment to the auxiliary matrix `[Pi'_eta | Pi'_Q | mu | nu |
    /// W_theta | W_vartheta]`.
    pub aux_commitment: DoryCommitment,
    pub identity: HiddenRelationBatchReductionProof,
    pub coverage: HiddenRelationReductionProof,
}

impl ShapeValidityProof {
    /// The shape reduction contributes no standalone physical Dory proof.
    pub fn physical_dory_proof_count(&self) -> usize {
        0
    }
}

/// Prover-side shape relation claims retained for the shared terminal batch.
pub struct ShapeValidityProverOpenings<'a> {
    pub identity: HiddenRelationProverOpening<'a>,
    pub coverage: HiddenRelationProverOpening<'a>,
}

/// Verifier-side shape relation claims retained for the shared terminal batch.
pub struct ShapeValidityVerifierOpenings<'a> {
    pub identity: HiddenRelationVerifierOpening<'a>,
    pub coverage: HiddenRelationVerifierOpening<'a>,
}

#[derive(Debug)]
pub enum ShapeValidityError {
    Data(Section6DataError),
    Relation(HiddenRelationError),
    LogUp(HiddenLogUpError),
    Opening(Section6FinalOpeningError),
    Encoding(String),
    ProofShape,
}

impl Display for ShapeValidityError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Data(error) => Display::fmt(error, formatter),
            Self::Relation(error) => Display::fmt(error, formatter),
            Self::LogUp(error) => Display::fmt(error, formatter),
            Self::Opening(error) => Display::fmt(error, formatter),
            Self::Encoding(message) => write!(formatter, "encoding failure: {message}"),
            Self::ProofShape => formatter.write_str("invalid shape validity proof"),
        }
    }
}

impl std::error::Error for ShapeValidityError {}

impl From<Section6DataError> for ShapeValidityError {
    fn from(error: Section6DataError) -> Self {
        Self::Data(error)
    }
}

impl From<HiddenRelationError> for ShapeValidityError {
    fn from(error: HiddenRelationError) -> Self {
        Self::Relation(error)
    }
}

impl From<HiddenLogUpError> for ShapeValidityError {
    fn from(error: HiddenLogUpError) -> Self {
        Self::LogUp(error)
    }
}

impl From<Section6FinalOpeningError> for ShapeValidityError {
    fn from(error: Section6FinalOpeningError) -> Self {
        Self::Opening(error)
    }
}

/// Commit the mask matrix `[D_eta | D_Z]` on the shared auxiliary grid.
pub fn commit_shape_masks_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    layout: Section6Layout,
    metadata: PrivateShapeMetadata,
    matrix_dims: Vec<(usize, usize)>,
) -> Result<CommittedShapeMasks, ShapeValidityError> {
    let masks = Section6ShapeMasks::from_metadata(&layout, &metadata)?;
    if matrix_dims.len() != layout.matrix_count {
        return Err(ShapeValidityError::Data(
            Section6DataError::InvalidMetadata(format!(
                "expected {} matrix dimension pairs, got {}",
                layout.matrix_count,
                matrix_dims.len()
            )),
        ));
    }
    let side = 1_usize << layout.point_log();
    for (index, &(mu, nu)) in matrix_dims.iter().enumerate() {
        if mu == 0 || nu == 0 || mu > side || nu > side {
            return Err(ShapeValidityError::Data(
                Section6DataError::InvalidMetadata(format!(
                    "matrix {index} dimensions ({mu}, {nu}) exceed the 2^m bound {side}"
                )),
            ));
        }
    }

    let rows = layout.bucket_count;
    let cover = layout.row_cover()?;
    let eta_width = layout.eta_width();
    let logical_cols = eta_width + layout.pair_width();
    let padded_cols = logical_cols.next_power_of_two();
    let padded_len = cover
        .checked_mul(padded_cols)
        .ok_or(ShapeValidityError::Data(Section6DataError::LengthOverflow))?;

    let mut values = vec![Fr::zero(); padded_len];
    values
        .par_chunks_mut(cover)
        .take(logical_cols)
        .enumerate()
        .for_each(|(column, chunk)| {
            for (row, value) in chunk[..rows].iter_mut().enumerate() {
                let bit = if column < eta_width {
                    masks.d_eta.bit(row, column)
                } else {
                    masks.d_z.bit(row, column - eta_width)
                };
                if bit {
                    *value = Fr::one();
                }
            }
        });
    let mask_columns = commit_section6_owned_field_matrix_on_grid_zk(
        parameters,
        grid,
        values,
        cover,
        logical_cols,
        padded_cols,
    )?;
    let mask_commitment = mask_columns
        .first()
        .map(|column| column.commitment.clone())
        .ok_or(ShapeValidityError::ProofShape)?;

    let statement = ShapeCommitStatement {
        mask_commitment,
        padded_rows: cover,
        eta_width,
        pair_width: layout.pair_width(),
    };
    Ok(CommittedShapeMasks {
        layout,
        metadata,
        masks,
        matrix_dims,
        statement,
        mask_columns,
    })
}

/// Convenience constructor from the dense BVA descriptor.
pub fn commit_shape_masks_from_descriptor_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    descriptor: &DenseBvaDescriptor,
    bit_width: usize,
) -> Result<CommittedShapeMasks, ShapeValidityError> {
    let layout = Section6Layout::from_descriptor(descriptor, bit_width)?;
    commit_shape_masks_for_layout(parameters, grid, layout, descriptor)
}

/// Descriptor constructor with explicit public width bounds `m` and `d`.
pub fn commit_shape_masks_from_descriptor_with_widths_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    descriptor: &DenseBvaDescriptor,
    point_width: usize,
    tau_width: usize,
    bit_width: usize,
) -> Result<CommittedShapeMasks, ShapeValidityError> {
    let layout =
        Section6Layout::from_descriptor_with_widths(descriptor, point_width, tau_width, bit_width)?;
    commit_shape_masks_for_layout(parameters, grid, layout, descriptor)
}

fn commit_shape_masks_for_layout(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    layout: Section6Layout,
    descriptor: &DenseBvaDescriptor,
) -> Result<CommittedShapeMasks, ShapeValidityError> {
    let metadata = PrivateShapeMetadata::from_descriptor(&layout, descriptor)?;
    let mut matrix_dims = vec![(0_usize, 0_usize); descriptor.matrix_count()];
    for bucket in descriptor.buckets() {
        let dims = &mut matrix_dims[bucket.matrix()];
        dims.0 = dims.0.max(
            bucket
                .row_offset()
                .checked_add(bucket.row_len())
                .ok_or(ShapeValidityError::Data(Section6DataError::LengthOverflow))?,
        );
        dims.1 = dims.1.max(
            bucket
                .col_offset()
                .checked_add(bucket.col_len())
                .ok_or(ShapeValidityError::Data(Section6DataError::LengthOverflow))?,
        );
    }
    commit_shape_masks_on_grid_zk(parameters, grid, layout, metadata, matrix_dims)
}

/// Column offsets inside the committed auxiliary matrix.
#[derive(Clone, Copy)]
struct ShapeAuxLayout {
    eta_width: usize,
    pair_width: usize,
}

impl ShapeAuxLayout {
    fn new(layout: &Section6Layout) -> Self {
        Self {
            eta_width: layout.eta_width(),
            pair_width: layout.pair_width(),
        }
    }

    /// Proper prefix `Pi'_eta[h]` for `h in [0, q - 1)`.
    fn eta_prefix_col(self, h: usize) -> usize {
        h
    }

    /// Proper prefix `Pi'_Q[h]` for `h in [0, 2m - 1)`.
    fn q_prefix_col(self, h: usize) -> usize {
        self.eta_width - 1 + h
    }

    fn mu_col(self) -> usize {
        self.eta_width + self.pair_width - 2
    }

    fn nu_col(self) -> usize {
        self.mu_col() + 1
    }

    fn w_theta_col(self) -> usize {
        self.mu_col() + 2
    }

    fn w_vartheta_col(self) -> usize {
        self.mu_col() + 3
    }

    fn logical_cols(self) -> usize {
        self.eta_width + self.pair_width + 2
    }

    fn padded_cols(self) -> usize {
        self.logical_cols().next_power_of_two()
    }
}

/// Padded column count of the validity auxiliary matrix, exposed so the
/// orchestration layer can size the shared Dory grid.
pub fn shape_validity_aux_padded_cols(layout: &Section6Layout) -> usize {
    ShapeAuxLayout::new(layout).padded_cols()
}

/// Coverage challenges sampled before the auxiliary matrix is committed.
#[derive(Clone, Copy)]
struct ShapeChallenges {
    /// Row-side (`alpha_l`, `mu`) base point.
    theta: Fr,
    /// Column-side (`alpha_r`, `nu`) base point.
    vartheta: Fr,
    /// Source-index base point.
    omega: Fr,
}

/// Combination weights sampled after the auxiliary matrix is committed.
#[derive(Clone, Copy)]
struct CombineChallenges {
    booleanity: Fr,
    alignment: Fr,
    prefix: Fr,
    eta_chain: Fr,
    q_chain: Fr,
}

fn encode_with_length(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

/// Bind the public layout and the mask commitment into one digest.
fn shape_binding(
    layout: &Section6Layout,
    statement: &ShapeCommitStatement,
) -> Result<[u8; 32], ShapeValidityError> {
    let mut hasher = Sha256::new();
    hasher.update(SHAPE_DOMAIN);
    let layout_bytes = bincode::serialize(layout)
        .map_err(|error| ShapeValidityError::Encoding(error.to_string()))?;
    encode_with_length(&mut hasher, &layout_bytes);
    let statement_bytes = bincode::serialize(statement)
        .map_err(|error| ShapeValidityError::Encoding(error.to_string()))?;
    encode_with_length(&mut hasher, &statement_bytes);
    Ok(hasher.finalize().into())
}

/// Extend the binding with the auxiliary commitment.
fn validity_binding(
    binding: &[u8; 32],
    aux_commitment: &DoryCommitment,
) -> Result<[u8; 32], ShapeValidityError> {
    let mut hasher = Sha256::new();
    hasher.update(b"parcomm-shape-validity-aux");
    hasher.update(binding);
    let aux_bytes = bincode::serialize(aux_commitment)
        .map_err(|error| ShapeValidityError::Encoding(error.to_string()))?;
    encode_with_length(&mut hasher, &aux_bytes);
    Ok(hasher.finalize().into())
}

/// Sample one field challenge outside `{0, 1}` from the binding.
fn sample_nontrivial(binding: &[u8; 32], tag: &[u8]) -> Fr {
    let mut counter = 0_u64;
    loop {
        let mut hasher = Sha256::new();
        hasher.update(b"parcomm-shape-challenge");
        hasher.update(binding);
        hasher.update((tag.len() as u64).to_le_bytes());
        hasher.update(tag);
        hasher.update(counter.to_le_bytes());
        let candidate = Fr::from_le_bytes_mod_order(&hasher.finalize());
        if !candidate.is_zero() && candidate != Fr::one() {
            return candidate;
        }
        counter += 1;
    }
}

fn shape_challenges(binding: &[u8; 32]) -> ShapeChallenges {
    ShapeChallenges {
        theta: sample_nontrivial(binding, b"theta-row"),
        vartheta: sample_nontrivial(binding, b"vartheta-col"),
        omega: sample_nontrivial(binding, b"omega-source"),
    }
}

fn combine_challenges(binding: &[u8; 32]) -> CombineChallenges {
    CombineChallenges {
        booleanity: sample_nontrivial(binding, b"booleanity"),
        alignment: sample_nontrivial(binding, b"alignment"),
        prefix: sample_nontrivial(binding, b"prefix"),
        eta_chain: sample_nontrivial(binding, b"eta-chain"),
        q_chain: sample_nontrivial(binding, b"q-chain"),
    }
}

fn scoped_label(role: &[u8], binding: &[u8; 32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(SHAPE_DOMAIN.len() + role.len() + 32 + 16);
    output.extend_from_slice(SHAPE_DOMAIN);
    output.extend_from_slice(&(role.len() as u64).to_le_bytes());
    output.extend_from_slice(role);
    output.extend_from_slice(binding);
    output
}

/// `base^{2^0}, base^{2^1}, ..., base^{2^{len-1}}`.
fn square_chain(base: Fr, len: usize) -> Vec<Fr> {
    let mut powers = Vec::with_capacity(len);
    let mut current = base;
    for _ in 0..len {
        powers.push(current);
        current.square_in_place();
    }
    powers
}

/// The `q` power-of-two test points matching the mask column order:
/// `vartheta^{2^h}` on the column-offset bits, `theta^{2^h}` on the
/// row-offset bits, then `omega^{2^h}` on the source-index bits.
fn coverage_test_points(layout: &Section6Layout, challenges: &ShapeChallenges) -> Vec<Fr> {
    let m = layout.point_log();
    let mut points = square_chain(challenges.vartheta, m);
    points.extend(square_chain(challenges.theta, m));
    points.extend(square_chain(challenges.omega, layout.tau_log()));
    points
}

fn structured_error(error: StructuredColumnError) -> ShapeValidityError {
    ShapeValidityError::Encoding(error.to_string())
}

/// Public columns used by the coverage sum and the power lookup.
///
/// Every column is piecewise regular, so the verifier evaluates each MLE in
/// `O(polylog)` time from the description alone; the prover additionally
/// materializes the dense entries for the sumcheck rounds.
struct ValidityPublicColumns {
    /// `1` on the real bucket rows, `0` on padding rows.
    indicator: OwnedPublicColumn,
    /// `omega^j` on the first `t` rows, `0` afterwards.
    geo_omega: OwnedPublicColumn,
    /// One complete table enumerating the exponents `h in [0, 2^m]`.
    exponent: OwnedPublicColumn,
    /// One complete table enumerating `theta^h`.
    theta_power: OwnedPublicColumn,
    /// One complete table enumerating `vartheta^h`.
    vartheta_power: OwnedPublicColumn,
}

/// Prover-owned shape-validity auxiliary commitment and public lookup tables.
pub struct ShapeValidityProverContext {
    aux_layout: ShapeAuxLayout,
    aux_columns: Vec<CommittedDoryFieldVector>,
    binding: [u8; 32],
    points: Vec<Fr>,
    challenges: ShapeChallenges,
    combine: CombineChallenges,
    publics: ValidityPublicColumns,
}

impl ShapeValidityProverContext {
    pub fn aux_commitment(&self) -> &DoryCommitment {
        &self.aux_columns[0].commitment
    }

    fn aux_table(&self, column: usize) -> Result<HiddenRelationTable<'_>, ShapeValidityError> {
        let committed = self
            .aux_columns
            .get(column)
            .ok_or(ShapeValidityError::ProofShape)?;
        Ok(HiddenRelationTable {
            values: committed
                .matrix_column_values()
                .ok_or(ShapeValidityError::ProofShape)?,
            committed,
        })
    }

    /// The row- and column-dimension power families appended to the joint
    /// opening-time LogUp.
    pub fn power_lookup_families(
        &self,
    ) -> Result<Vec<HiddenLogUpUnionProverFamilyData<'_>>, ShapeValidityError> {
        let mu = self.aux_table(self.aux_layout.mu_col())?;
        let w_theta = self.aux_table(self.aux_layout.w_theta_col())?;
        let nu = self.aux_table(self.aux_layout.nu_col())?;
        let w_vartheta = self.aux_table(self.aux_layout.w_vartheta_col())?;
        let theta_tables = vec![vec![
            MixedLogUpColumn::Public(self.publics.exponent.table()),
            MixedLogUpColumn::Public(self.publics.theta_power.table()),
        ]];
        let vartheta_tables = vec![vec![
            MixedLogUpColumn::Public(self.publics.exponent.table()),
            MixedLogUpColumn::Public(self.publics.vartheta_power.table()),
        ]];
        Ok(vec![
            HiddenLogUpUnionProverFamilyData {
                query_parts: vec![vec![
                    MixedLogUpColumn::Hidden(mu),
                    MixedLogUpColumn::Hidden(w_theta),
                ]],
                table_hidden_parts: (0..theta_tables.len()).map(|_| Vec::new()).collect(),
                table_parts: theta_tables,
                query_hidden_parts: vec![vec![mu, w_theta]],
            },
            HiddenLogUpUnionProverFamilyData {
                query_parts: vec![vec![
                    MixedLogUpColumn::Hidden(nu),
                    MixedLogUpColumn::Hidden(w_vartheta),
                ]],
                table_hidden_parts: (0..vartheta_tables.len()).map(|_| Vec::new()).collect(),
                table_parts: vartheta_tables,
                query_hidden_parts: vec![vec![nu, w_vartheta]],
            },
        ])
    }
}

/// Verifier-owned public data used by the deferred shape reductions.
pub struct ShapeValidityVerifierContext {
    aux_layout: ShapeAuxLayout,
    binding: [u8; 32],
    points: Vec<Fr>,
    challenges: ShapeChallenges,
    combine: CombineChallenges,
    publics: ValidityPublicColumns,
}

impl ShapeValidityVerifierContext {
    fn aux_statement<'a>(
        &self,
        layout: &Section6Layout,
        proof: &'a ShapeValidityProof,
        column: usize,
    ) -> Result<HiddenRelationTableStatement<'a>, ShapeValidityError> {
        if column >= self.aux_layout.logical_cols() {
            return Err(ShapeValidityError::ProofShape);
        }
        Ok(HiddenRelationTableStatement::matrix_column(
            &proof.aux_commitment,
            layout.row_cover()?,
            self.aux_layout.logical_cols(),
            self.aux_layout.padded_cols(),
            column,
        ))
    }

    /// The row- and column-dimension power families appended to the joint
    /// opening-time LogUp.
    pub fn power_lookup_families<'a>(
        &'a self,
        layout: &Section6Layout,
        proof: &'a ShapeValidityProof,
    ) -> Result<Vec<HiddenLogUpUnionVerifierFamilyData<'a>>, ShapeValidityError> {
        let mu = self.aux_statement(layout, proof, self.aux_layout.mu_col())?;
        let w_theta = self.aux_statement(layout, proof, self.aux_layout.w_theta_col())?;
        let nu = self.aux_statement(layout, proof, self.aux_layout.nu_col())?;
        let w_vartheta = self.aux_statement(layout, proof, self.aux_layout.w_vartheta_col())?;
        let theta_tables = vec![vec![
            MixedLogUpColumnStatement::Public(self.publics.exponent.table()),
            MixedLogUpColumnStatement::Public(self.publics.theta_power.table()),
        ]];
        let vartheta_tables = vec![vec![
            MixedLogUpColumnStatement::Public(self.publics.exponent.table()),
            MixedLogUpColumnStatement::Public(self.publics.vartheta_power.table()),
        ]];
        Ok(vec![
            HiddenLogUpUnionVerifierFamilyData {
                query_parts: vec![vec![
                    MixedLogUpColumnStatement::Hidden(mu),
                    MixedLogUpColumnStatement::Hidden(w_theta),
                ]],
                table_hidden_parts: (0..theta_tables.len()).map(|_| Vec::new()).collect(),
                table_parts: theta_tables,
                query_hidden_parts: vec![vec![mu, w_theta]],
            },
            HiddenLogUpUnionVerifierFamilyData {
                query_parts: vec![vec![
                    MixedLogUpColumnStatement::Hidden(nu),
                    MixedLogUpColumnStatement::Hidden(w_vartheta),
                ]],
                table_hidden_parts: (0..vartheta_tables.len()).map(|_| Vec::new()).collect(),
                table_parts: vartheta_tables,
                query_hidden_parts: vec![vec![nu, w_vartheta]],
            },
        ])
    }
}

fn validity_public_columns(
    layout: &Section6Layout,
    challenges: &ShapeChallenges,
    materialize: bool,
) -> Result<ValidityPublicColumns, ShapeValidityError> {
    let cover = layout.row_cover()?;
    let column = |padded_len: usize,
                  segments: Vec<ColumnSegment>|
     -> Result<StructuredColumn, ShapeValidityError> {
        StructuredColumn::with_zero_tail(padded_len, segments, Vec::new()).map_err(structured_error)
    };
    let owned = |column: StructuredColumn| -> OwnedPublicColumn {
        OwnedPublicColumn::new(column, materialize)
    };

    let indicator = owned(column(
        cover,
        vec![ColumnSegment {
            len: layout.bucket_count,
            pattern: ColumnPattern::Constant(Fr::one()),
        }],
    )?);
    let geo_omega = owned(column(
        cover,
        vec![ColumnSegment {
            len: layout.matrix_count,
            pattern: ColumnPattern::Geometric {
                scale: Fr::one(),
                ratio: challenges.omega,
            },
        }],
    )?);
    // Each power family is one complete logical table. Its power-of-two tail
    // repeats the valid pair `(0, 1)` and never creates proof-visible parts.
    let table_len = (1_usize << layout.point_log()) + 1;
    let table_cover = table_len.next_power_of_two();
    let tail_len = table_cover - table_len;
    let mut exponent_segments = vec![ColumnSegment {
        len: table_len,
        pattern: ColumnPattern::RowIndex { start: Fr::zero() },
    }];
    let mut theta_segments = vec![ColumnSegment {
        len: table_len,
        pattern: ColumnPattern::Geometric {
            scale: Fr::one(),
            ratio: challenges.theta,
        },
    }];
    let mut vartheta_segments = vec![ColumnSegment {
        len: table_len,
        pattern: ColumnPattern::Geometric {
            scale: Fr::one(),
            ratio: challenges.vartheta,
        },
    }];
    if tail_len != 0 {
        exponent_segments.push(ColumnSegment {
            len: tail_len,
            pattern: ColumnPattern::Constant(Fr::zero()),
        });
        theta_segments.push(ColumnSegment {
            len: tail_len,
            pattern: ColumnPattern::Constant(Fr::one()),
        });
        vartheta_segments.push(ColumnSegment {
            len: tail_len,
            pattern: ColumnPattern::Constant(Fr::one()),
        });
    }
    let exponent = owned(column(table_cover, exponent_segments)?);
    let theta_power = owned(column(table_cover, theta_segments)?);
    let vartheta_power = owned(column(table_cover, vartheta_segments)?);

    Ok(ValidityPublicColumns {
        indicator,
        geo_omega,
        exponent,
        theta_power,
        vartheta_power,
    })
}

/// Symbolic operand references resolved by the prover and the verifier.
#[derive(Clone, Copy)]
enum ShapeTableRef {
    DEta(usize),
    DZ(usize),
    EtaPrefix(usize),
    QPrefix(usize),
}

struct ShapeIdentityItem {
    hidden: Vec<ShapeTableRef>,
    expression: RelationExpression,
}

fn expr_constant(value: Fr) -> RelationExpression {
    RelationExpression::constant(value)
}

fn expr_table(index: usize) -> RelationExpression {
    RelationExpression::table(index)
}

fn expr_one() -> RelationExpression {
    expr_constant(Fr::one())
}

/// The validity identity batch in a fixed order shared by both sides.
///
/// Every identity holds on padding rows as well: the masks pad with zeros,
/// so each chain factor degenerates to `1` and every prefix column pads
/// with ones.
fn shape_identity_items(
    layout: &Section6Layout,
    points: &[Fr],
    combine: &CombineChallenges,
) -> Vec<ShapeIdentityItem> {
    let m = layout.point_log();
    let q = layout.eta_width();
    let pair = layout.pair_width();
    let mut items = Vec::new();

    // Booleanity of both masks: sum_h w^h (D_h - D_h^2) = 0.
    {
        let hidden = (0..q)
            .map(ShapeTableRef::DEta)
            .chain((0..pair).map(ShapeTableRef::DZ))
            .collect::<Vec<_>>();
        let mut expression = expr_constant(Fr::zero());
        let mut weight = Fr::one();
        for index in 0..q + pair {
            expression = expression.add(
                expr_constant(weight)
                    .mul(expr_table(index).sub(expr_table(index).mul(expr_table(index)))),
            );
            weight *= combine.booleanity;
        }
        items.push(ShapeIdentityItem { hidden, expression });
    }

    // Alignment on the pair columns: an offset bit cannot sit on a varying
    // coordinate, sum_h w^h D_eta_h D_Z_h = 0.
    {
        let hidden = (0..pair)
            .map(ShapeTableRef::DEta)
            .chain((0..pair).map(ShapeTableRef::DZ))
            .collect::<Vec<_>>();
        let mut expression = expr_constant(Fr::zero());
        let mut weight = Fr::one();
        for h in 0..pair {
            expression =
                expression.add(expr_constant(weight).mul(expr_table(h).mul(expr_table(pair + h))));
            weight *= combine.alignment;
        }
        items.push(ShapeIdentityItem { hidden, expression });
    }

    // Ones-prefix structure of D_Z inside each side:
    // D_Z_h (1 - D_Z_{h-1}) = 0 for h not crossing the side boundary.
    if m > 1 {
        let hidden = (0..pair).map(ShapeTableRef::DZ).collect::<Vec<_>>();
        let mut expression = expr_constant(Fr::zero());
        let mut weight = Fr::one();
        for h in 1..pair {
            if h == m {
                continue;
            }
            expression = expression.add(
                expr_constant(weight).mul(expr_table(h).mul(expr_one().sub(expr_table(h - 1)))),
            );
            weight *= combine.prefix;
        }
        items.push(ShapeIdentityItem { hidden, expression });
    }

    // eta grand-product chain: Pi'_eta[h] = F_h Pi'_eta[h-1] with
    // F_h = 1 + (pt_h - 1) D_eta_h; the last factor closes in the coverage
    // sum instead of a committed eta-hat column.
    {
        let mut hidden = (0..q - 1).map(ShapeTableRef::EtaPrefix).collect::<Vec<_>>();
        hidden.extend((0..q - 1).map(ShapeTableRef::DEta));
        let mut expression = expr_constant(Fr::zero());
        let mut weight = Fr::one();
        for h in 0..q - 1 {
            let factor =
                expr_one().add(expr_constant(points[h] - Fr::one()).mul(expr_table(q - 1 + h)));
            let previous = if h == 0 {
                expr_one()
            } else {
                expr_table(h - 1)
            };
            expression =
                expression.add(expr_constant(weight).mul(expr_table(h).sub(factor.mul(previous))));
            weight *= combine.eta_chain;
        }
        items.push(ShapeIdentityItem { hidden, expression });
    }

    // Q grand-product chain with factors 1 + D_Z_h pt_h.
    {
        let mut hidden = (0..pair - 1)
            .map(ShapeTableRef::QPrefix)
            .collect::<Vec<_>>();
        hidden.extend((0..pair - 1).map(ShapeTableRef::DZ));
        let mut expression = expr_constant(Fr::zero());
        let mut weight = Fr::one();
        for h in 0..pair - 1 {
            let factor = expr_one().add(expr_constant(points[h]).mul(expr_table(pair - 1 + h)));
            let previous = if h == 0 {
                expr_one()
            } else {
                expr_table(h - 1)
            };
            expression =
                expression.add(expr_constant(weight).mul(expr_table(h).sub(factor.mul(previous))));
            weight *= combine.q_chain;
        }
        items.push(ShapeIdentityItem { hidden, expression });
    }

    items
}

/// The coverage sum expression over the fixed operand order
/// `[D_eta[q-1], Pi'_eta[q-2], D_Z[2m-1], Pi'_Q[2m-2], W_theta, W_vartheta;
/// indicator, geo_omega]`:
///
/// `ind (1-theta)(1-vartheta) etahat Qhat - geo_omega (1-W_theta)(1-W_vartheta)`
///
/// summed to zero, where `etahat = F_{q-1} Pi'_eta[q-2]` and
/// `Qhat = G_{2m-1} Pi'_Q[2m-2]` reconstruct the full grand products.
fn coverage_expression(
    points: &[Fr],
    pair: usize,
    challenges: &ShapeChallenges,
) -> RelationExpression {
    let q = points.len();
    let eta_last = expr_one().add(expr_constant(points[q - 1] - Fr::one()).mul(expr_table(0)));
    let q_last = expr_one().add(expr_constant(points[pair - 1]).mul(expr_table(2)));
    let scale = (Fr::one() - challenges.theta) * (Fr::one() - challenges.vartheta);
    let masked = expr_constant(scale)
        .mul(expr_table(6))
        .mul(eta_last)
        .mul(expr_table(1))
        .mul(q_last)
        .mul(expr_table(3));
    let target = expr_table(7)
        .mul(expr_one().sub(expr_table(4)))
        .mul(expr_one().sub(expr_table(5)));
    masked.sub(target)
}

/// Build the auxiliary matrix `[Pi'_eta | Pi'_Q | mu | nu | W_theta |
/// W_vartheta]` column by column over the shared row cover.
fn build_validity_aux_matrix(
    layout: &Section6Layout,
    masks: &Section6ShapeMasks,
    matrix_dims: &[(usize, usize)],
    points: &[Fr],
    challenges: &ShapeChallenges,
    cover: usize,
) -> Result<Vec<Fr>, ShapeValidityError> {
    let aux = ShapeAuxLayout::new(layout);
    let q = layout.eta_width();
    let pair = layout.pair_width();
    let rows = layout.bucket_count;
    let padded_len = cover
        .checked_mul(aux.padded_cols())
        .ok_or(ShapeValidityError::Data(Section6DataError::LengthOverflow))?;
    let mut values = vec![Fr::zero(); padded_len];

    // Pi'_eta[h] = prod_{s <= h} (1 + (pt_s - 1) D_eta_s); padding rows 1.
    for h in 0..q - 1 {
        let column_start = aux.eta_prefix_col(h) * cover;
        let (before, tail) = values.split_at_mut(column_start);
        let current = &mut tail[..cover];
        let point = points[h];
        if h == 0 {
            current.par_iter_mut().enumerate().for_each(|(row, value)| {
                *value = if row < rows && masks.d_eta.bit(row, 0) {
                    point
                } else {
                    Fr::one()
                };
            });
        } else {
            let previous_start = aux.eta_prefix_col(h - 1) * cover;
            let previous = &before[previous_start..previous_start + cover];
            current
                .par_iter_mut()
                .zip(previous.par_iter())
                .enumerate()
                .for_each(|(row, (value, prev))| {
                    *value = if row < rows && masks.d_eta.bit(row, h) {
                        *prev * point
                    } else {
                        *prev
                    };
                });
        }
    }

    // Pi'_Q[h] = prod_{s <= h} (1 + D_Z_s pt_s); padding rows 1.
    for h in 0..pair - 1 {
        let column_start = aux.q_prefix_col(h) * cover;
        let (before, tail) = values.split_at_mut(column_start);
        let current = &mut tail[..cover];
        let one_plus_point = Fr::one() + points[h];
        if h == 0 {
            current.par_iter_mut().enumerate().for_each(|(row, value)| {
                *value = if row < rows && masks.d_z.bit(row, 0) {
                    one_plus_point
                } else {
                    Fr::one()
                };
            });
        } else {
            let previous_start = aux.q_prefix_col(h - 1) * cover;
            let previous = &before[previous_start..previous_start + cover];
            current
                .par_iter_mut()
                .zip(previous.par_iter())
                .enumerate()
                .for_each(|(row, (value, prev))| {
                    *value = if row < rows && masks.d_z.bit(row, h) {
                        *prev * one_plus_point
                    } else {
                        *prev
                    };
                });
        }
    }

    // mu, nu and their bound powers; padding rows carry (0, 0, 1, 1), the
    // valid lookup pair for exponent 0.
    let mu_start = aux.mu_col() * cover;
    let nu_start = aux.nu_col() * cover;
    let w_theta_start = aux.w_theta_col() * cover;
    let w_vartheta_start = aux.w_vartheta_col() * cover;
    for (index, &(mu, nu)) in matrix_dims.iter().enumerate() {
        values[mu_start + index] = field_from_usize(mu);
        values[nu_start + index] = field_from_usize(nu);
        values[w_theta_start + index] = challenges.theta.pow([mu as u64]);
        values[w_vartheta_start + index] = challenges.vartheta.pow([nu as u64]);
    }
    for row in matrix_dims.len()..cover {
        values[w_theta_start + row] = Fr::one();
        values[w_vartheta_start + row] = Fr::one();
    }

    Ok(values)
}

/// Commit the shape-validity auxiliary matrix and retain its lookup tables.
pub fn prepare_shape_validity_prover_context(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    committed: &CommittedShapeMasks,
) -> Result<ShapeValidityProverContext, ShapeValidityError> {
    let layout = &committed.layout;
    let cover = layout.row_cover()?;
    let aux = ShapeAuxLayout::new(layout);
    let binding = shape_binding(layout, &committed.statement)?;
    let challenges = shape_challenges(&binding);
    let points = coverage_test_points(layout, &challenges);
    let aux_values = build_validity_aux_matrix(
        layout,
        &committed.masks,
        &committed.matrix_dims,
        &points,
        &challenges,
        cover,
    )?;
    let aux_columns = commit_section6_owned_field_matrix_on_grid_zk(
        parameters,
        grid,
        aux_values,
        cover,
        aux.logical_cols(),
        aux.padded_cols(),
    )?;
    let aux_commitment = aux_columns
        .first()
        .map(|column| &column.commitment)
        .ok_or(ShapeValidityError::ProofShape)?;
    let binding = validity_binding(&binding, aux_commitment)?;
    let combine = combine_challenges(&binding);
    let publics = validity_public_columns(layout, &challenges, true)?;
    Ok(ShapeValidityProverContext {
        aux_layout: aux,
        aux_columns,
        binding,
        points,
        challenges,
        combine,
        publics,
    })
}

/// Reduce the shape identities while deferring their terminal PCS claims.
pub fn prove_shape_validity_deferred<'a>(
    parameters: &DoryProverParameters,
    committed: &'a CommittedShapeMasks,
    context: &'a ShapeValidityProverContext,
    rng: &mut impl RngCore,
) -> Result<(ShapeValidityProof, ShapeValidityProverOpenings<'a>), ShapeValidityError> {
    let layout = &committed.layout;
    let q = layout.eta_width();
    let pair = layout.pair_width();
    let aux = context.aux_layout;
    let resolve =
        |reference: &ShapeTableRef| -> Result<HiddenRelationTable<'_>, ShapeValidityError> {
            match reference {
                ShapeTableRef::DEta(h) => committed
                    .mask_table(*h)
                    .ok_or(ShapeValidityError::ProofShape),
                ShapeTableRef::DZ(h) => committed
                    .mask_table(q + *h)
                    .ok_or(ShapeValidityError::ProofShape),
                ShapeTableRef::EtaPrefix(h) => context.aux_table(aux.eta_prefix_col(*h)),
                ShapeTableRef::QPrefix(h) => context.aux_table(aux.q_prefix_col(*h)),
            }
        };

    let items = shape_identity_items(layout, &context.points, &context.combine);
    let resolved_items = items
        .iter()
        .map(|item| {
            item.hidden
                .iter()
                .map(&resolve)
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, ShapeValidityError>>()?;
    let batch_items = resolved_items
        .iter()
        .zip(&items)
        .map(|(tables, item)| HiddenRelationBatchProverItem {
            tables,
            public_tables: &[],
            expression: &item.expression,
        })
        .collect::<Vec<_>>();
    let identity_tables = deduplicate_prover_hidden_tables(
        resolved_items
            .iter()
            .flat_map(|tables| tables.iter().copied()),
    );
    let (identity, identity_opening) = prove_hidden_relations_batch_deferred(
        &scoped_label(b"identity", &context.binding),
        parameters,
        &batch_items,
        rng,
    )?;
    let identity_opening = identity_opening
        .into_claims()
        .bind_tables(identity_tables)?;

    let coverage_hidden = vec![
        resolve(&ShapeTableRef::DEta(q - 1))?,
        resolve(&ShapeTableRef::EtaPrefix(q - 2))?,
        resolve(&ShapeTableRef::DZ(pair - 1))?,
        resolve(&ShapeTableRef::QPrefix(pair - 2))?,
        context.aux_table(aux.w_theta_col())?,
        context.aux_table(aux.w_vartheta_col())?,
    ];
    let coverage_publics = vec![
        context.publics.indicator.table(),
        context.publics.geo_omega.table(),
    ];
    let coverage_expression = coverage_expression(&context.points, pair, &context.challenges);
    let (coverage, coverage_opening) = prove_hidden_sum_relation_deferred_with_public(
        &scoped_label(b"coverage", &context.binding),
        parameters,
        &coverage_hidden,
        &coverage_publics,
        &coverage_expression,
        rng,
    )?;
    let coverage_opening = coverage_opening
        .into_claims()
        .bind_tables(coverage_hidden)?;
    Ok((
        ShapeValidityProof {
            aux_commitment: context.aux_commitment().clone(),
            identity,
            coverage,
        },
        ShapeValidityProverOpenings {
            identity: identity_opening,
            coverage: coverage_opening,
        },
    ))
}

/// Reconstruct the public shape-validity context from commitments only.
pub fn prepare_shape_validity_verifier_context(
    layout: &Section6Layout,
    statement: &ShapeCommitStatement,
    proof: &ShapeValidityProof,
) -> Result<ShapeValidityVerifierContext, ShapeValidityError> {
    layout.validate()?;
    statement.validate(layout)?;
    let aux_layout = ShapeAuxLayout::new(layout);
    let binding = shape_binding(layout, statement)?;
    let challenges = shape_challenges(&binding);
    let points = coverage_test_points(layout, &challenges);
    let binding = validity_binding(&binding, &proof.aux_commitment)?;
    let combine = combine_challenges(&binding);
    let publics = validity_public_columns(layout, &challenges, false)?;
    Ok(ShapeValidityVerifierContext {
        aux_layout,
        binding,
        points,
        challenges,
        combine,
        publics,
    })
}

/// Verify the deferred shape reductions without performing a PCS opening.
pub fn verify_shape_validity_deferred<'a>(
    parameters: &DoryVerifierParameters,
    layout: &Section6Layout,
    statement: &'a ShapeCommitStatement,
    proof: &'a ShapeValidityProof,
    context: &'a ShapeValidityVerifierContext,
) -> Result<ShapeValidityVerifierOpenings<'a>, ShapeValidityError> {
    let q = layout.eta_width();
    let pair = layout.pair_width();
    let aux = context.aux_layout;
    let resolve = |reference: &ShapeTableRef| -> Result<HiddenRelationTableStatement<'a>, ShapeValidityError> {
        Ok(match reference {
            ShapeTableRef::DEta(h) => statement.mask_column(*h),
            ShapeTableRef::DZ(h) => statement.mask_column(q + *h),
            ShapeTableRef::EtaPrefix(h) => {
                context.aux_statement(layout, proof, aux.eta_prefix_col(*h))?
            }
            ShapeTableRef::QPrefix(h) => {
                context.aux_statement(layout, proof, aux.q_prefix_col(*h))?
            }
        })
    };

    let items = shape_identity_items(layout, &context.points, &context.combine);
    let resolved_items = items
        .iter()
        .map(|item| {
            item.hidden
                .iter()
                .map(&resolve)
                .collect::<Result<Vec<_>, _>>()
        })
        .collect::<Result<Vec<_>, ShapeValidityError>>()?;
    let batch_items = resolved_items
        .iter()
        .zip(&items)
        .map(|(tables, item)| HiddenRelationBatchVerifierItem {
            tables,
            public_tables: &[],
            expression: &item.expression,
        })
        .collect::<Vec<_>>();
    let identity_tables = deduplicate_verifier_hidden_tables(
        resolved_items
            .iter()
            .flat_map(|tables| tables.iter().copied()),
    );
    let identity_opening = verify_hidden_relations_batch_deferred(
        &scoped_label(b"identity", &context.binding),
        parameters,
        &batch_items,
        &proof.identity,
    )?;
    let identity_opening = identity_opening
        .into_claims()
        .bind_tables(identity_tables)?;

    let coverage_hidden = vec![
        resolve(&ShapeTableRef::DEta(q - 1))?,
        resolve(&ShapeTableRef::EtaPrefix(q - 2))?,
        resolve(&ShapeTableRef::DZ(pair - 1))?,
        resolve(&ShapeTableRef::QPrefix(pair - 2))?,
        context.aux_statement(layout, proof, aux.w_theta_col())?,
        context.aux_statement(layout, proof, aux.w_vartheta_col())?,
    ];
    let coverage_publics = vec![
        context.publics.indicator.table(),
        context.publics.geo_omega.table(),
    ];
    let coverage_expression = coverage_expression(&context.points, pair, &context.challenges);
    let coverage_opening = verify_hidden_sum_relation_deferred_with_public(
        &scoped_label(b"coverage", &context.binding),
        parameters,
        &coverage_hidden,
        &coverage_publics,
        &coverage_expression,
        &proof.coverage,
    )?;
    let coverage_opening = coverage_opening
        .into_claims()
        .bind_tables(coverage_hidden)?;
    Ok(ShapeValidityVerifierOpenings {
        identity: identity_opening,
        coverage: coverage_opening,
    })
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        jolt_dory::setup_dory_parameters,
        section6_claim_collection::{
            Section6ProverClaimCollection, Section6VerifierClaimCollection,
        },
    };

    #[test]
    fn shape_power_tables_are_single_full_range_columns() {
        let descriptor = DenseBvaDescriptor::from_shapes(&[(1023, 1023)]).unwrap();
        let layout = Section6Layout::from_descriptor(&descriptor, 8).unwrap();
        assert!((1_usize << layout.point_log()) + 1 > layout.row_cover().unwrap());
        let challenges = ShapeChallenges {
            theta: Fr::from(5_u64),
            vartheta: Fr::from(7_u64),
            omega: Fr::from(11_u64),
        };

        let public = validity_public_columns(&layout, &challenges, true).unwrap();
        let full_table_len = ((1_usize << layout.point_log()) + 1).next_power_of_two();

        assert_eq!(public.exponent.column.padded_len, full_table_len);
        assert_eq!(public.theta_power.column.padded_len, full_table_len);
        assert_eq!(public.vartheta_power.column.padded_len, full_table_len);

        let active_len = (1_usize << layout.point_log()) + 1;
        let exponent = public.exponent.values.as_ref().unwrap();
        let theta = public.theta_power.values.as_ref().unwrap();
        let vartheta = public.vartheta_power.values.as_ref().unwrap();
        assert_eq!(exponent[active_len - 1], field_from_usize(active_len - 1));
        assert_eq!(
            theta[active_len - 1],
            challenges.theta.pow([(active_len - 1) as u64])
        );
        assert_eq!(
            vartheta[active_len - 1],
            challenges.vartheta.pow([(active_len - 1) as u64])
        );
        assert_eq!(
            (exponent[active_len], theta[active_len]),
            (Fr::zero(), Fr::one())
        );
        assert_eq!(
            (exponent[active_len], vartheta[active_len]),
            (Fr::zero(), Fr::one())
        );
    }

    #[test]
    fn shape_commit_and_validity_round_trip() {
        let shapes = [(3, 5), (5, 2), (2, 7)];
        let descriptor = DenseBvaDescriptor::from_shapes(&shapes).unwrap();
        let grid = Section6DoryGrid::new(12).unwrap();
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(grid.num_vars());
        let mut rng = StdRng::seed_from_u64(0x5348_4150_455f_5631);

        let committed = commit_shape_masks_from_descriptor_on_grid_zk(
            &prover_parameters,
            &grid,
            &descriptor,
            8,
        )
        .unwrap();
        let context =
            prepare_shape_validity_prover_context(&prover_parameters, &grid, &committed).unwrap();
        let power_families = context.power_lookup_families().unwrap();
        assert_eq!(power_families.len(), 2);
        assert!(power_families
            .iter()
            .all(|family| family.table_parts.len() == 1));
        let (proof, prover_openings) =
            prove_shape_validity_deferred(&prover_parameters, &committed, &context, &mut rng)
                .unwrap();
        assert_eq!(proof.physical_dory_proof_count(), 0);
        let verifier_context = prepare_shape_validity_verifier_context(
            committed.layout(),
            committed.statement(),
            &proof,
        )
        .unwrap();
        let verifier_openings = verify_shape_validity_deferred(
            &verifier_parameters,
            committed.layout(),
            committed.statement(),
            &proof,
            &verifier_context,
        )
        .unwrap();
        assert_eq!(
            prover_openings.identity.evaluation_commitments(),
            verifier_openings.identity.evaluation_commitments()
        );
        assert_eq!(
            prover_openings.coverage.evaluation_commitments(),
            verifier_openings.coverage.evaluation_commitments()
        );

        let mut prover_claims = Section6ProverClaimCollection::new();
        prover_claims
            .push_relation(&prover_openings.identity, &grid)
            .unwrap();
        assert_eq!(prover_claims.len(), 2);
        prover_claims
            .push_relation(&prover_openings.coverage, &grid)
            .unwrap();
        assert_eq!(prover_claims.len(), 4);

        let mut verifier_claims = Section6VerifierClaimCollection::new();
        verifier_claims
            .push_relation(&verifier_openings.identity, &grid)
            .unwrap();
        assert_eq!(verifier_claims.len(), 2);
        verifier_claims
            .push_relation(&verifier_openings.coverage, &grid)
            .unwrap();
        assert_eq!(verifier_claims.len(), 4);

        // Replacing the mask commitment must break the validity proof.
        let mut tampered = committed.statement().clone();
        tampered.mask_commitment = proof.aux_commitment.clone();
        let tampered_context =
            prepare_shape_validity_verifier_context(committed.layout(), &tampered, &proof);
        assert!(tampered_context
            .and_then(|context| {
                verify_shape_validity_deferred(
                    &verifier_parameters,
                    committed.layout(),
                    &tampered,
                    &proof,
                    &context,
                )
                .map(|_| ())
            })
            .is_err());
    }

    #[test]
    fn validity_rejects_wrong_matrix_dimensions() {
        let shapes = [(3, 5), (5, 2), (2, 7)];
        let descriptor = DenseBvaDescriptor::from_shapes(&shapes).unwrap();
        let grid = Section6DoryGrid::new(12).unwrap();
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(grid.num_vars());
        let mut rng = StdRng::seed_from_u64(0x5348_4150_455f_5632);

        let layout = Section6Layout::from_descriptor(&descriptor, 8).unwrap();
        let metadata = PrivateShapeMetadata::from_descriptor(&layout, &descriptor).unwrap();
        let mut wrong_dims = descriptor.matrix_shapes().to_vec();
        wrong_dims[0].0 += 1;
        let committed =
            commit_shape_masks_on_grid_zk(&prover_parameters, &grid, layout, metadata, wrong_dims)
                .unwrap();
        let result = prepare_shape_validity_prover_context(&prover_parameters, &grid, &committed)
            .and_then(|context| {
                prove_shape_validity_deferred(&prover_parameters, &committed, &context, &mut rng)
                    .map(|(proof, _)| proof)
            })
            .and_then(|proof| {
                prepare_shape_validity_verifier_context(
                    committed.layout(),
                    committed.statement(),
                    &proof,
                )
                .and_then(|context| {
                    verify_shape_validity_deferred(
                        &verifier_parameters,
                        committed.layout(),
                        committed.statement(),
                        &proof,
                        &context,
                    )
                    .map(|_| ())
                })
            });
        assert!(result.is_err());
    }
}
