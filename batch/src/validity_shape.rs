//! Shape-mask commitment and its one-time validity proof.
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
//! power columns) lives in one committed matrix, and every deferred claim -
//! including the LogUp auxiliary openings - resolves in one terminal Dory
//! batch inside the validity proof.

use std::fmt::{Display, Formatter};

use ark_bn254::Fr;
use ark_ff::{Field, One, PrimeField, Zero};
use ark_std::rand::RngCore;
use jolt_dory::DoryCommitment;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    hidden_logup::{
        prove_hidden_logup_parts_deferred_on_grid, verify_hidden_logup_parts_deferred_on_grid,
        HiddenLogUpError, HiddenLogUpPartsReductionProof, MixedLogUpColumn,
        MixedLogUpColumnStatement,
    },
    hidden_relation::{
        prove_hidden_sum_relation_deferred_with_public,
        verify_hidden_sum_relation_deferred_with_public, HiddenRelationError,
        HiddenRelationReductionProof, HiddenRelationTable, HiddenRelationTableStatement,
        OwnedPublicColumn, RelationExpression,
    },
    hidden_relation_batch::{
        prove_hidden_relations_batch_deferred, verify_hidden_relations_batch_deferred,
        HiddenRelationBatchProverItem, HiddenRelationBatchReductionProof,
        HiddenRelationBatchVerifierItem,
    },
    jolt_dory::{CommittedDoryFieldVector, DoryProverParameters, DoryVerifierParameters},
    matrix_collection::DenseBvaDescriptor,
    section6_claim_collection::{Section6ProverClaimCollection, Section6VerifierClaimCollection},
    section6_data::{
        field_from_usize, PrivateShapeMetadata, Section6DataError, Section6Layout,
        Section6ShapeMasks,
    },
    section6_final_opening::{
        commit_section6_owned_field_matrix_on_grid_zk, prove_section6_final_opening,
        verify_section6_final_opening, Section6DoryGrid, Section6FinalOpeningError,
        Section6FinalOpeningProof,
    },
    structured_column::{ColumnPattern, ColumnSegment, StructuredColumn, StructuredColumnError},
};

const SHAPE_DOMAIN: &[u8] = b"parcomm-shape-commit-v1";
const VALIDITY_OPENING_ROLE: &[u8] = b"shape-validity-opening-v1";

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

/// One-time validity proof for the committed shape masks.
///
/// The terminal batch inside this proof opens every deferred claim of the
/// identity batch, the coverage sum, and the power lookup - including the
/// committed LogUp auxiliary vectors - in one Dory opening.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct ShapeValidityProof {
    /// Commitment to the auxiliary matrix `[Pi'_eta | Pi'_Q | mu | nu |
    /// W_theta | W_vartheta]`.
    pub aux_commitment: DoryCommitment,
    pub identity: HiddenRelationBatchReductionProof,
    pub coverage: HiddenRelationReductionProof,
    pub power_lookup: HiddenLogUpPartsReductionProof,
    /// Terminal batch over every deferred claim above.
    pub opening: Section6FinalOpeningProof,
}

impl ShapeValidityProof {
    pub fn physical_dory_proof_count(&self) -> usize {
        self.opening.physical_dory_proof_count()
    }
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
struct ShapeChallenges {
    /// Row-side (`alpha_l`, `mu`) base point.
    theta: Fr,
    /// Column-side (`alpha_r`, `nu`) base point.
    vartheta: Fr,
    /// Source-index base point.
    omega: Fr,
}

/// Combination weights sampled after the auxiliary matrix is committed.
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
    /// Constant `0` tag for the `theta` lookup side.
    tag_theta: OwnedPublicColumn,
    /// Constant `1` tag for the `vartheta` lookup side.
    tag_vartheta: OwnedPublicColumn,
    /// Table parts enumerating the exponents `h in [0, 2^m]`.
    exponent_parts: Vec<OwnedPublicColumn>,
    /// Table parts enumerating `theta^h`.
    theta_power_parts: Vec<OwnedPublicColumn>,
    /// Table parts enumerating `vartheta^h`.
    vartheta_power_parts: Vec<OwnedPublicColumn>,
}

fn validity_public_columns(
    layout: &Section6Layout,
    challenges: &ShapeChallenges,
    materialize: bool,
) -> Result<ValidityPublicColumns, ShapeValidityError> {
    let cover = layout.row_cover()?;
    let column = |segments: Vec<ColumnSegment>| -> Result<StructuredColumn, ShapeValidityError> {
        StructuredColumn::with_zero_tail(cover, segments, Vec::new()).map_err(structured_error)
    };
    let owned = |column: StructuredColumn| -> OwnedPublicColumn {
        OwnedPublicColumn::new(column, materialize)
    };

    let indicator = owned(column(vec![ColumnSegment {
        len: layout.bucket_count,
        pattern: ColumnPattern::Constant(Fr::one()),
    }])?);
    let geo_omega = owned(column(vec![ColumnSegment {
        len: layout.matrix_count,
        pattern: ColumnPattern::Geometric {
            scale: Fr::one(),
            ratio: challenges.omega,
        },
    }])?);
    let tag_theta = owned(column(Vec::new())?);
    let tag_vartheta = owned(column(vec![ColumnSegment {
        len: cover,
        pattern: ColumnPattern::Constant(Fr::one()),
    }])?);

    // The power tables enumerate (h, base^h) for h in [0, 2^m]; padding rows
    // repeat the valid pair (0, 1), and multiple parts cover 2^m + 1 > cover.
    let table_len = (1_usize << layout.point_log()) + 1;
    let part_count = table_len.div_ceil(cover);
    let mut exponent_parts = Vec::with_capacity(part_count);
    let mut theta_power_parts = Vec::with_capacity(part_count);
    let mut vartheta_power_parts = Vec::with_capacity(part_count);
    for part in 0..part_count {
        let first = part * cover;
        let live = (table_len - first).min(cover);
        let mut exponent_segments = vec![ColumnSegment {
            len: live,
            pattern: ColumnPattern::RowIndex {
                start: field_from_usize(first),
            },
        }];
        let mut theta_segments = vec![ColumnSegment {
            len: live,
            pattern: ColumnPattern::Geometric {
                scale: challenges.theta.pow([first as u64]),
                ratio: challenges.theta,
            },
        }];
        let mut vartheta_segments = vec![ColumnSegment {
            len: live,
            pattern: ColumnPattern::Geometric {
                scale: challenges.vartheta.pow([first as u64]),
                ratio: challenges.vartheta,
            },
        }];
        if live < cover {
            exponent_segments.push(ColumnSegment {
                len: cover - live,
                pattern: ColumnPattern::Constant(Fr::zero()),
            });
            theta_segments.push(ColumnSegment {
                len: cover - live,
                pattern: ColumnPattern::Constant(Fr::one()),
            });
            vartheta_segments.push(ColumnSegment {
                len: cover - live,
                pattern: ColumnPattern::Constant(Fr::one()),
            });
        }
        exponent_parts.push(owned(column(exponent_segments)?));
        theta_power_parts.push(owned(column(theta_segments)?));
        vartheta_power_parts.push(owned(column(vartheta_segments)?));
    }

    Ok(ValidityPublicColumns {
        indicator,
        geo_omega,
        tag_theta,
        tag_vartheta,
        exponent_parts,
        theta_power_parts,
        vartheta_power_parts,
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

/// Prove the one-time shape validity and open every deferred claim in one
/// terminal Dory batch.
pub fn prove_shape_validity(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    committed: &CommittedShapeMasks,
    rng: &mut impl RngCore,
) -> Result<ShapeValidityProof, ShapeValidityError> {
    let layout = &committed.layout;
    let cover = layout.row_cover()?;
    let q = layout.eta_width();
    let pair = layout.pair_width();
    let aux = ShapeAuxLayout::new(layout);

    let binding = shape_binding(layout, &committed.statement)?;
    let challenges = shape_challenges(&binding);
    let points = coverage_test_points(layout, &challenges);

    // The auxiliary matrix is committed only after the coverage challenges
    // are fixed; the combination weights bind to it in turn.
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
        .map(|column| column.commitment.clone())
        .ok_or(ShapeValidityError::ProofShape)?;
    let binding = validity_binding(&binding, &aux_commitment)?;
    let combine = combine_challenges(&binding);
    let publics = validity_public_columns(layout, &challenges, true)?;

    let aux_table = |column: usize| -> Result<HiddenRelationTable<'_>, ShapeValidityError> {
        let committed_column = aux_columns
            .get(column)
            .ok_or(ShapeValidityError::ProofShape)?;
        Ok(HiddenRelationTable {
            values: committed_column
                .matrix_column_values()
                .ok_or(ShapeValidityError::ProofShape)?,
            committed: committed_column,
        })
    };
    let resolve =
        |reference: &ShapeTableRef| -> Result<HiddenRelationTable<'_>, ShapeValidityError> {
            match reference {
                ShapeTableRef::DEta(h) => committed
                    .mask_table(*h)
                    .ok_or(ShapeValidityError::ProofShape),
                ShapeTableRef::DZ(h) => committed
                    .mask_table(q + *h)
                    .ok_or(ShapeValidityError::ProofShape),
                ShapeTableRef::EtaPrefix(h) => aux_table(aux.eta_prefix_col(*h)),
                ShapeTableRef::QPrefix(h) => aux_table(aux.q_prefix_col(*h)),
            }
        };

    let items = shape_identity_items(layout, &points, &combine);
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
    let (identity, identity_opening) = prove_hidden_relations_batch_deferred(
        &scoped_label(b"identity", &binding),
        parameters,
        &batch_items,
        rng,
    )?;

    let coverage_hidden = vec![
        resolve(&ShapeTableRef::DEta(q - 1))?,
        resolve(&ShapeTableRef::EtaPrefix(q - 2))?,
        resolve(&ShapeTableRef::DZ(pair - 1))?,
        resolve(&ShapeTableRef::QPrefix(pair - 2))?,
        aux_table(aux.w_theta_col())?,
        aux_table(aux.w_vartheta_col())?,
    ];
    let coverage_publics = vec![publics.indicator.table(), publics.geo_omega.table()];
    let coverage_expression = coverage_expression(&points, pair, &challenges);
    let (coverage, coverage_opening) = prove_hidden_sum_relation_deferred_with_public(
        &scoped_label(b"coverage", &binding),
        parameters,
        &coverage_hidden,
        &coverage_publics,
        &coverage_expression,
        rng,
    )?;

    // Keyed power lookup {(tag, exponent, power)} against both base tables.
    let query_parts = vec![
        vec![
            MixedLogUpColumn::Public(publics.tag_theta.table()),
            MixedLogUpColumn::Hidden(aux_table(aux.mu_col())?),
            MixedLogUpColumn::Hidden(aux_table(aux.w_theta_col())?),
        ],
        vec![
            MixedLogUpColumn::Public(publics.tag_vartheta.table()),
            MixedLogUpColumn::Hidden(aux_table(aux.nu_col())?),
            MixedLogUpColumn::Hidden(aux_table(aux.w_vartheta_col())?),
        ],
    ];
    let mut table_parts = Vec::new();
    for (exponent, value) in publics
        .exponent_parts
        .iter()
        .zip(&publics.theta_power_parts)
    {
        table_parts.push(vec![
            MixedLogUpColumn::Public(publics.tag_theta.table()),
            MixedLogUpColumn::Public(exponent.table()),
            MixedLogUpColumn::Public(value.table()),
        ]);
    }
    for (exponent, value) in publics
        .exponent_parts
        .iter()
        .zip(&publics.vartheta_power_parts)
    {
        table_parts.push(vec![
            MixedLogUpColumn::Public(publics.tag_vartheta.table()),
            MixedLogUpColumn::Public(exponent.table()),
            MixedLogUpColumn::Public(value.table()),
        ]);
    }
    let (power_lookup, lookup_state) = prove_hidden_logup_parts_deferred_on_grid(
        &scoped_label(b"power-lookup", &binding),
        parameters,
        grid,
        &query_parts,
        &table_parts,
        rng,
    )?;

    // Terminal batch: every deferred claim, including the LogUp
    // auxiliary-vector openings, resolves in one Dory opening.
    let mut claims = Section6ProverClaimCollection::new();
    claims.push_relation(&identity_opening, grid)?;
    claims.push_relation(&coverage_opening, grid)?;
    let lookup_openings = lookup_state.relation_openings()?;
    for opening in &lookup_openings {
        claims.push_relation(opening, grid)?;
    }
    let opening = prove_section6_final_opening(
        VALIDITY_OPENING_ROLE,
        &binding,
        parameters,
        grid,
        &claims.claims(),
        rng,
    )?;

    Ok(ShapeValidityProof {
        aux_commitment,
        identity,
        coverage,
        power_lookup,
        opening,
    })
}

/// Verify the one-time shape validity proof from public data only.
pub fn verify_shape_validity(
    parameters: &DoryVerifierParameters,
    grid: &Section6DoryGrid,
    layout: &Section6Layout,
    statement: &ShapeCommitStatement,
    proof: &ShapeValidityProof,
) -> Result<(), ShapeValidityError> {
    layout.validate()?;
    statement.validate(layout)?;
    let cover = layout.row_cover()?;
    let q = layout.eta_width();
    let pair = layout.pair_width();
    let aux = ShapeAuxLayout::new(layout);

    let binding = shape_binding(layout, statement)?;
    let challenges = shape_challenges(&binding);
    let points = coverage_test_points(layout, &challenges);
    let binding = validity_binding(&binding, &proof.aux_commitment)?;
    let combine = combine_challenges(&binding);
    let publics = validity_public_columns(layout, &challenges, false)?;

    let aux_statement = |column: usize| -> HiddenRelationTableStatement<'_> {
        HiddenRelationTableStatement::matrix_column(
            &proof.aux_commitment,
            cover,
            aux.padded_cols(),
            column,
        )
    };
    let resolve = |reference: &ShapeTableRef| -> HiddenRelationTableStatement<'_> {
        match reference {
            ShapeTableRef::DEta(h) => statement.mask_column(*h),
            ShapeTableRef::DZ(h) => statement.mask_column(q + *h),
            ShapeTableRef::EtaPrefix(h) => aux_statement(aux.eta_prefix_col(*h)),
            ShapeTableRef::QPrefix(h) => aux_statement(aux.q_prefix_col(*h)),
        }
    };

    let items = shape_identity_items(layout, &points, &combine);
    let resolved_items = items
        .iter()
        .map(|item| item.hidden.iter().map(&resolve).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    let batch_items = resolved_items
        .iter()
        .zip(&items)
        .map(|(tables, item)| HiddenRelationBatchVerifierItem {
            tables,
            public_tables: &[],
            expression: &item.expression,
        })
        .collect::<Vec<_>>();
    let identity_opening = verify_hidden_relations_batch_deferred(
        &scoped_label(b"identity", &binding),
        parameters,
        &batch_items,
        &proof.identity,
    )?;

    let coverage_hidden = vec![
        resolve(&ShapeTableRef::DEta(q - 1)),
        resolve(&ShapeTableRef::EtaPrefix(q - 2)),
        resolve(&ShapeTableRef::DZ(pair - 1)),
        resolve(&ShapeTableRef::QPrefix(pair - 2)),
        aux_statement(aux.w_theta_col()),
        aux_statement(aux.w_vartheta_col()),
    ];
    let coverage_publics = vec![publics.indicator.table(), publics.geo_omega.table()];
    let coverage_expression = coverage_expression(&points, pair, &challenges);
    let coverage_opening = verify_hidden_sum_relation_deferred_with_public(
        &scoped_label(b"coverage", &binding),
        parameters,
        &coverage_hidden,
        &coverage_publics,
        &coverage_expression,
        &proof.coverage,
    )?;

    let query_parts = vec![
        vec![
            MixedLogUpColumnStatement::Public(publics.tag_theta.table()),
            MixedLogUpColumnStatement::Hidden(aux_statement(aux.mu_col())),
            MixedLogUpColumnStatement::Hidden(aux_statement(aux.w_theta_col())),
        ],
        vec![
            MixedLogUpColumnStatement::Public(publics.tag_vartheta.table()),
            MixedLogUpColumnStatement::Hidden(aux_statement(aux.nu_col())),
            MixedLogUpColumnStatement::Hidden(aux_statement(aux.w_vartheta_col())),
        ],
    ];
    let mut table_parts = Vec::new();
    for (exponent, value) in publics
        .exponent_parts
        .iter()
        .zip(&publics.theta_power_parts)
    {
        table_parts.push(vec![
            MixedLogUpColumnStatement::Public(publics.tag_theta.table()),
            MixedLogUpColumnStatement::Public(exponent.table()),
            MixedLogUpColumnStatement::Public(value.table()),
        ]);
    }
    for (exponent, value) in publics
        .exponent_parts
        .iter()
        .zip(&publics.vartheta_power_parts)
    {
        table_parts.push(vec![
            MixedLogUpColumnStatement::Public(publics.tag_vartheta.table()),
            MixedLogUpColumnStatement::Public(exponent.table()),
            MixedLogUpColumnStatement::Public(value.table()),
        ]);
    }
    let lookup_state = verify_hidden_logup_parts_deferred_on_grid(
        &scoped_label(b"power-lookup", &binding),
        parameters,
        grid,
        &query_parts,
        &table_parts,
        &proof.power_lookup,
    )?;

    let mut claims = Section6VerifierClaimCollection::new();
    claims.push_relation(&identity_opening, grid)?;
    claims.push_relation(&coverage_opening, grid)?;
    let lookup_openings = lookup_state
        .relation_openings()
        .map_err(ShapeValidityError::LogUp)?;
    for opening in &lookup_openings {
        claims.push_relation(opening, grid)?;
    }
    verify_section6_final_opening(
        VALIDITY_OPENING_ROLE,
        &binding,
        parameters,
        grid,
        &claims.claims(),
        &proof.opening,
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::jolt_dory::setup_dory_parameters;

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
        let proof = prove_shape_validity(&prover_parameters, &grid, &committed, &mut rng).unwrap();
        assert_eq!(proof.physical_dory_proof_count(), 1);
        verify_shape_validity(
            &verifier_parameters,
            &grid,
            committed.layout(),
            committed.statement(),
            &proof,
        )
        .unwrap();

        // Replacing the mask commitment must break the validity proof.
        let mut tampered = committed.statement().clone();
        tampered.mask_commitment = proof.aux_commitment.clone();
        assert!(verify_shape_validity(
            &verifier_parameters,
            &grid,
            committed.layout(),
            &tampered,
            &proof,
        )
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
        let result = prove_shape_validity(&prover_parameters, &grid, &committed, &mut rng)
            .and_then(|proof| {
                verify_shape_validity(
                    &verifier_parameters,
                    &grid,
                    committed.layout(),
                    committed.statement(),
                    &proof,
                )
            });
        assert!(result.is_err());
    }
}
