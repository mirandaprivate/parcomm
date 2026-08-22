//! Deferred point-compilation argument for the mask-based Section 6 flow.
//!
//! The prover commits the point matrix `X = (X_r | X_l)` before the batching
//! challenge `omega` is sampled, then commits one auxiliary matrix holding
//! `tau`, the gathered rows `X'`, the compaction ranks `sigma` and
//! `sigma^ = D_Z o sigma`, the masked coordinates `Z~ = D_Z o X'`, the eta
//! grand-product prefixes `P`, and the compacted local points `Z`. Three
//! reductions bind them to the registered masks:
//!
//! * a keyed source-row lookup `{(tau, X')} subset {(n, X)}`;
//! * a row-local compaction lookup `{(row, sigma^, Z~)} subset
//!   {(row, h, Z_h)} union {(row, 0, 0)}`;
//! * one Fiat--Shamir identity batch for the `tau` decoding, the rank
//!   recurrence, the masked products, and the eta prefix recurrence.
//!
//! Every terminal Dory opening is deferred to the shared Section 6 batch.

use std::fmt::{Display, Formatter};

use ark_bn254::Fr;
use ark_ff::{One, Zero};
use ark_std::rand::RngCore;
use jolt_dory::DoryCommitment;
use serde::{Deserialize, Serialize};

use crate::{
    hidden_logup::{
        prove_hidden_logup_parts_deferred_on_grid, verify_hidden_logup_parts_deferred_on_grid,
        HiddenLogUpError, HiddenLogUpPartsReductionProof, HiddenLogUpProverOpeningState,
        HiddenLogUpVerifierOpeningState, MixedLogUpColumn, MixedLogUpColumnStatement,
    },
    hidden_relation::{
        HiddenRelationError, HiddenRelationProverOpening, HiddenRelationTable,
        HiddenRelationTableStatement, HiddenRelationVerifierOpening, PublicRelationTable,
        RelationExpression,
    },
    hidden_relation_batch::{
        prove_hidden_relations_batch_deferred, verify_hidden_relations_batch_deferred,
        HiddenRelationBatchProverItem, HiddenRelationBatchReductionProof,
        HiddenRelationBatchVerifierItem,
    },
    jolt_dory::{CommittedDoryFieldVector, DoryProverParameters, DoryVerifierParameters},
    section6_data::{
        field_from_usize, validate_fixed_claims, Section6DataError, Section6Layout,
        Section6MaskMatrixClaim, Section6MaskPointWitness, Section6PublicLayout,
        SECTION6_FIXED_ETA_WIDTH, SECTION6_FIXED_LOCAL_WIDTH, SECTION6_FIXED_MATRIX_LOG,
        SECTION6_FIXED_POINT_LOG,
    },
    section6_final_opening::{
        commit_section6_field_matrix_columns_on_grid_zk, Section6DoryGrid,
        Section6FinalOpeningError,
    },
    section6_shape_masks::{CommittedSection6ShapeMasks, Section6ShapeMaskStatement},
};

const POINT_DOMAIN: &[u8] = b"parcomm-section6-mask-point-v2";

/// Column offsets of the one committed point-compilation auxiliary matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Section6PointAuxColumns {
    /// Width `min(2m, l)` of the compacted `Z` block.
    pub local_width: usize,
}

impl Section6PointAuxColumns {
    pub fn new(local_width: usize) -> Self {
        Self { local_width }
    }

    pub fn tau(self) -> usize {
        0
    }

    pub fn x_selected(self, coordinate: usize) -> usize {
        1 + coordinate
    }

    pub fn sigma(self, coordinate: usize) -> usize {
        1 + SECTION6_FIXED_LOCAL_WIDTH + coordinate
    }

    pub fn sigma_masked(self, coordinate: usize) -> usize {
        1 + 2 * SECTION6_FIXED_LOCAL_WIDTH + coordinate
    }

    pub fn z_tilde(self, coordinate: usize) -> usize {
        1 + 3 * SECTION6_FIXED_LOCAL_WIDTH + coordinate
    }

    pub fn eta_prefix(self, coordinate: usize) -> usize {
        1 + 4 * SECTION6_FIXED_LOCAL_WIDTH + coordinate
    }

    /// The compiled TBV scale column `eta` is the last eta prefix.
    pub fn eta(self) -> usize {
        self.eta_prefix(SECTION6_FIXED_ETA_WIDTH - 1)
    }

    pub fn z(self, coordinate: usize) -> usize {
        1 + 4 * SECTION6_FIXED_LOCAL_WIDTH + SECTION6_FIXED_ETA_WIDTH + coordinate
    }

    pub fn count(self) -> usize {
        1 + 4 * SECTION6_FIXED_LOCAL_WIDTH + SECTION6_FIXED_ETA_WIDTH + self.local_width
    }

    pub fn padded_cols(self) -> usize {
        self.count().next_power_of_two()
    }
}

/// Verifier-visible statement for the committed point matrix `X`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Section6PointMatrixStatement {
    pub commitment: DoryCommitment,
    pub matrix_count: usize,
    pub padded_rows: usize,
    pub padded_cols: usize,
}

impl Section6PointMatrixStatement {
    pub fn validate(&self, layout: &Section6PublicLayout) -> Result<(), Section6DataError> {
        if self.matrix_count != layout.matrix_count
            || self.padded_rows != layout.row_cover()?
            || self.padded_cols != SECTION6_FIXED_LOCAL_WIDTH.next_power_of_two()
        {
            return Err(Section6DataError::InvalidPoint(
                "point matrix statement does not match the public layout".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Prover-only committed point matrix retained between omega sampling stages.
pub struct CommittedSection6PointMatrix {
    statement: Section6PointMatrixStatement,
    columns: Vec<CommittedDoryFieldVector>,
    column_values: Vec<Vec<Fr>>,
}

impl CommittedSection6PointMatrix {
    pub fn statement(&self) -> &Section6PointMatrixStatement {
        &self.statement
    }

    pub fn columns(&self) -> &[CommittedDoryFieldVector] {
        &self.columns
    }

    pub fn column_values(&self, column: usize) -> Option<&[Fr]> {
        self.column_values.get(column).map(Vec::as_slice)
    }
}

/// Verifier-visible statement for the committed auxiliary matrix.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Section6PointAuxStatement {
    pub commitment: DoryCommitment,
    pub padded_rows: usize,
    pub padded_cols: usize,
    pub local_width: usize,
}

impl Section6PointAuxStatement {
    pub fn validate(&self, layout: &Section6PublicLayout) -> Result<(), Section6DataError> {
        let columns = Section6PointAuxColumns::new(layout.local_width());
        if self.local_width != layout.local_width()
            || self.padded_rows != layout.row_cover()?
            || self.padded_cols != columns.padded_cols()
        {
            return Err(Section6DataError::InvalidPoint(
                "point auxiliary statement does not match the public layout".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Prover-only committed auxiliary matrix owning the compiled witness.
pub struct CommittedSection6PointAux {
    statement: Section6PointAuxStatement,
    witness: Section6MaskPointWitness<Fr>,
    columns: Vec<CommittedDoryFieldVector>,
}

impl CommittedSection6PointAux {
    pub fn statement(&self) -> &Section6PointAuxStatement {
        &self.statement
    }

    pub fn witness(&self) -> &Section6MaskPointWitness<Fr> {
        &self.witness
    }

    pub fn columns(&self) -> &[CommittedDoryFieldVector] {
        &self.columns
    }

    pub fn layout_columns(&self) -> Section6PointAuxColumns {
        Section6PointAuxColumns::new(self.statement.local_width)
    }

    fn column_values(&self, column: usize) -> Option<&[Fr]> {
        let columns = self.layout_columns();
        if column == columns.tau() {
            return Some(&self.witness.tau.values);
        }
        let ranges = [
            (1, &self.witness.x_selected),
            (
                1 + SECTION6_FIXED_LOCAL_WIDTH,
                &self.witness.sigma,
            ),
            (
                1 + 2 * SECTION6_FIXED_LOCAL_WIDTH,
                &self.witness.sigma_masked,
            ),
            (
                1 + 3 * SECTION6_FIXED_LOCAL_WIDTH,
                &self.witness.z_tilde,
            ),
            (
                1 + 4 * SECTION6_FIXED_LOCAL_WIDTH,
                &self.witness.eta_prefix,
            ),
            (
                1 + 4 * SECTION6_FIXED_LOCAL_WIDTH + SECTION6_FIXED_ETA_WIDTH,
                &self.witness.z,
            ),
        ];
        for (start, matrix) in ranges {
            if column >= start && column < start + matrix.logical_cols {
                return matrix.padded_column(column - start);
            }
        }
        None
    }

    /// One hidden-relation table view of an auxiliary column.
    pub fn table(&self, column: usize) -> Option<HiddenRelationTable<'_>> {
        Some(HiddenRelationTable {
            values: self.column_values(column)?,
            committed: self.columns.get(column)?,
        })
    }
}

/// Serializable point-compilation reduction with no PCS opening embedded.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Section6PointCompilationProof {
    pub aux: Section6PointAuxStatement,
    pub source_lookup: HiddenLogUpPartsReductionProof,
    pub compaction_lookup: HiddenLogUpPartsReductionProof,
    pub identity: HiddenRelationBatchReductionProof,
}

/// Prover-side deferred openings produced by one point compilation.
pub struct Section6PointProverOpenings<'a> {
    pub identity: HiddenRelationProverOpening<'a>,
    pub source_lookup: HiddenLogUpProverOpeningState<'a>,
    pub compaction_lookup: HiddenLogUpProverOpeningState<'a>,
}

/// Verifier-side deferred openings mirroring the prover structure.
pub struct Section6PointVerifierOpenings<'a> {
    pub identity: HiddenRelationVerifierOpening<'a>,
    pub source_lookup: HiddenLogUpVerifierOpeningState<'a>,
    pub compaction_lookup: HiddenLogUpVerifierOpeningState<'a>,
}

/// Public tables shared by the point-compilation relations and lookups.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Section6PointPublicTables {
    /// `1..=R` row tags for the row-local compaction lookup.
    pub row_tags: Vec<Fr>,
    /// One-based source identifiers `1..=t` padded with zeros.
    pub source_index: Vec<Fr>,
    /// `1` on the `T` real bucket rows, `0` on the padding rows.
    pub real_row_mask: Vec<Fr>,
    /// Broadcast columns `h + 1` for the compaction table parts.
    pub position_constants: Vec<Vec<Fr>>,
    /// The all-zero broadcast column of the null compaction entry.
    pub zero: Vec<Fr>,
}

impl Section6PointPublicTables {
    pub fn derive(layout: &Section6PublicLayout) -> Result<Self, Section6DataError> {
        layout.validate()?;
        let rows = layout.row_cover()?;
        let local_width = layout.local_width();
        let row_tags = (0..rows)
            .map(|row| field_from_usize::<Fr>(row + 1))
            .collect::<Vec<_>>();
        let source_index = (0..rows)
            .map(|row| {
                if row < layout.matrix_count {
                    field_from_usize::<Fr>(row + 1)
                } else {
                    Fr::zero()
                }
            })
            .collect::<Vec<_>>();
        let mut real_row_mask = vec![Fr::zero(); rows];
        real_row_mask[..layout.bucket_count].fill(Fr::one());
        let position_constants = (0..local_width)
            .map(|position| vec![field_from_usize::<Fr>(position + 1); rows])
            .collect();
        Ok(Self {
            row_tags,
            source_index,
            real_row_mask,
            position_constants,
            zero: vec![Fr::zero(); rows],
        })
    }
}

#[derive(Debug)]
pub enum Section6PointCompilationError {
    Data(Section6DataError),
    Relation(HiddenRelationError),
    LogUp(HiddenLogUpError),
    Opening(Section6FinalOpeningError),
    InvalidWitness(String),
    ProofShape,
}

impl Display for Section6PointCompilationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Data(error) => Display::fmt(error, formatter),
            Self::Relation(error) => Display::fmt(error, formatter),
            Self::LogUp(error) => Display::fmt(error, formatter),
            Self::Opening(error) => Display::fmt(error, formatter),
            Self::InvalidWitness(message) => {
                write!(formatter, "invalid point compilation witness: {message}")
            }
            Self::ProofShape => formatter.write_str("invalid point compilation proof shape"),
        }
    }
}

impl std::error::Error for Section6PointCompilationError {}

impl From<Section6DataError> for Section6PointCompilationError {
    fn from(error: Section6DataError) -> Self {
        Self::Data(error)
    }
}

impl From<HiddenRelationError> for Section6PointCompilationError {
    fn from(error: HiddenRelationError) -> Self {
        Self::Relation(error)
    }
}

impl From<HiddenLogUpError> for Section6PointCompilationError {
    fn from(error: HiddenLogUpError) -> Self {
        Self::LogUp(error)
    }
}

impl From<Section6FinalOpeningError> for Section6PointCompilationError {
    fn from(error: Section6FinalOpeningError) -> Self {
        Self::Opening(error)
    }
}

/// Commit the point matrix `X` before the batching challenge is sampled.
pub fn commit_section6_point_matrix_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    layout: &Section6Layout,
    claims: &[Section6MaskMatrixClaim<Fr>],
) -> Result<CommittedSection6PointMatrix, Section6PointCompilationError> {
    validate_fixed_claims(layout, claims)?;
    let rows = layout.row_cover()?;
    let m = SECTION6_FIXED_POINT_LOG;
    let column_values = (0..SECTION6_FIXED_LOCAL_WIDTH)
        .map(|col| {
            let mut column = vec![Fr::zero(); rows];
            for (row, claim) in claims.iter().enumerate() {
                column[row] = if col < m {
                    claim.right_point[col]
                } else {
                    claim.left_point[col - m]
                };
            }
            column
        })
        .collect::<Vec<_>>();
    let column_slices = column_values
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let padded_cols = SECTION6_FIXED_LOCAL_WIDTH.next_power_of_two();
    let columns = commit_section6_field_matrix_columns_on_grid_zk(
        parameters,
        grid,
        &column_slices,
        padded_cols,
    )?;
    let commitment = columns
        .first()
        .map(|column| column.commitment.clone())
        .ok_or(Section6PointCompilationError::ProofShape)?;
    Ok(CommittedSection6PointMatrix {
        statement: Section6PointMatrixStatement {
            commitment,
            matrix_count: layout.matrix_count,
            padded_rows: rows,
            padded_cols,
        },
        columns,
        column_values,
    })
}

/// Commit the point-compilation auxiliary matrix for one compiled witness.
pub fn commit_section6_point_aux_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    layout: &Section6Layout,
    witness: Section6MaskPointWitness<Fr>,
) -> Result<CommittedSection6PointAux, Section6PointCompilationError> {
    let rows = layout.row_cover()?;
    let local_width = layout.public().local_width();
    if witness.x_selected.padded_rows != rows
        || witness.tau.padded_len != rows
        || witness.sigma.padded_rows != rows
        || witness.sigma_masked.padded_rows != rows
        || witness.z_tilde.padded_rows != rows
        || witness.eta_prefix.padded_rows != rows
        || witness.z.padded_rows != rows
        || witness.z.logical_cols != local_width
        || witness.eta_prefix.logical_cols != SECTION6_FIXED_ETA_WIDTH
    {
        return Err(Section6PointCompilationError::InvalidWitness(
            "witness covers do not match the layout row cover".to_owned(),
        ));
    }
    let columns_layout = Section6PointAuxColumns::new(local_width);
    let mut column_slices = Vec::with_capacity(columns_layout.count());
    column_slices.push(witness.tau.values.as_slice());
    column_slices.extend(witness.x_selected.padded_columns());
    column_slices.extend(witness.sigma.padded_columns());
    column_slices.extend(witness.sigma_masked.padded_columns());
    column_slices.extend(witness.z_tilde.padded_columns());
    column_slices.extend(witness.eta_prefix.padded_columns());
    column_slices.extend(witness.z.padded_columns());
    debug_assert_eq!(column_slices.len(), columns_layout.count());
    let columns = commit_section6_field_matrix_columns_on_grid_zk(
        parameters,
        grid,
        &column_slices,
        columns_layout.padded_cols(),
    )?;
    let commitment = columns
        .first()
        .map(|column| column.commitment.clone())
        .ok_or(Section6PointCompilationError::ProofShape)?;
    Ok(CommittedSection6PointAux {
        statement: Section6PointAuxStatement {
            commitment,
            padded_rows: rows,
            padded_cols: columns_layout.padded_cols(),
            local_width,
        },
        witness,
        columns,
    })
}

/// Caller-owned backing storage for one deferred identity batch.
///
/// The batch prover and verifier borrow per-item table slices with the same
/// lifetime as the returned openings, so the resolved tables must outlive the
/// call; the orchestrator keeps this plan alive next to the openings.
pub struct Section6IdentityPlan<'a, T> {
    items: Vec<Section6IdentityPlanItem<T>>,
    mask_public: PublicRelationTable<'a>,
}

struct Section6IdentityPlanItem<T> {
    tables: Vec<T>,
    masked: bool,
    expression: RelationExpression,
}

impl<'a, T> Section6IdentityPlan<'a, T> {
    pub(crate) fn new(mask_public: PublicRelationTable<'a>) -> Self {
        Self {
            items: Vec::new(),
            mask_public,
        }
    }

    pub(crate) fn push(&mut self, tables: Vec<T>, masked: bool, expression: RelationExpression) {
        self.items.push(Section6IdentityPlanItem {
            tables,
            masked,
            expression,
        });
    }

    pub(crate) fn prover_items(&'a self) -> Vec<HiddenRelationBatchProverItem<'a>>
    where
        T: AsProverTable<'a>,
    {
        self.items
            .iter()
            .map(|item| HiddenRelationBatchProverItem {
                tables: T::as_table_slice(&item.tables),
                public_tables: if item.masked {
                    std::slice::from_ref(&self.mask_public)
                } else {
                    &[]
                },
                expression: &item.expression,
            })
            .collect()
    }

    pub(crate) fn verifier_items(&'a self) -> Vec<HiddenRelationBatchVerifierItem<'a>>
    where
        T: AsVerifierTable<'a>,
    {
        self.items
            .iter()
            .map(|item| HiddenRelationBatchVerifierItem {
                tables: T::as_table_slice(&item.tables),
                public_tables: if item.masked {
                    std::slice::from_ref(&self.mask_public)
                } else {
                    &[]
                },
                expression: &item.expression,
            })
            .collect()
    }
}

/// Marker trait mapping plan storage to prover batch table slices.
pub(crate) trait AsProverTable<'a>: Sized {
    fn as_table_slice(tables: &'a [Self]) -> &'a [HiddenRelationTable<'a>];
}

impl<'a> AsProverTable<'a> for HiddenRelationTable<'a> {
    fn as_table_slice(tables: &'a [Self]) -> &'a [HiddenRelationTable<'a>] {
        tables
    }
}

/// Marker trait mapping plan storage to verifier batch table slices.
pub(crate) trait AsVerifierTable<'a>: Sized {
    fn as_table_slice(tables: &'a [Self]) -> &'a [HiddenRelationTableStatement<'a>];
}

impl<'a> AsVerifierTable<'a> for HiddenRelationTableStatement<'a> {
    fn as_table_slice(tables: &'a [Self]) -> &'a [HiddenRelationTableStatement<'a>] {
        tables
    }
}

/// Prover-side identity plan for the point compilation.
pub type Section6PointProverPlan<'a> = Section6IdentityPlan<'a, HiddenRelationTable<'a>>;

/// Verifier-side identity plan for the point compilation.
pub type Section6PointVerifierPlan<'a> = Section6IdentityPlan<'a, HiddenRelationTableStatement<'a>>;

/// One identity of the point-compilation batch, in shared table order.
struct PointIdentityItem {
    hidden: Vec<PointTableRef>,
    masked: bool,
    expression: RelationExpression,
}

/// Symbolic operand references resolved by the prover and the verifier.
#[derive(Clone, Copy)]
enum PointTableRef {
    AuxTau,
    AuxXSelected(usize),
    AuxSigma(usize),
    AuxSigmaMasked(usize),
    AuxZTilde(usize),
    AuxPrefix(usize),
    MaskEta(usize),
    MaskZ(usize),
}

/// The complete identity batch in a fixed order shared by both sides.
fn point_identity_items(batching_point: &[Fr]) -> Vec<PointIdentityItem> {
    let mut items = Vec::new();

    // tau = mask o (1 + sum_k 2^k D_eta[2m+k]).
    {
        let mut hidden = vec![PointTableRef::AuxTau];
        for bit in 0..SECTION6_FIXED_MATRIX_LOG {
            hidden.push(PointTableRef::MaskEta(SECTION6_FIXED_LOCAL_WIDTH + bit));
        }
        let mask_index = hidden.len();
        let mut decoded = RelationExpression::constant(Fr::one());
        for bit in 0..SECTION6_FIXED_MATRIX_LOG {
            decoded = decoded.add(
                RelationExpression::constant(Fr::from(1_u64 << bit))
                    .mul(RelationExpression::table(1 + bit)),
            );
        }
        let expression = RelationExpression::table(0)
            .sub(RelationExpression::table(mask_index).mul(decoded));
        items.push(PointIdentityItem {
            hidden,
            masked: true,
            expression,
        });
    }

    // sigma_0 = D_Z[0]; sigma_k = sigma_{k-1} + D_Z[k].
    for coordinate in 0..SECTION6_FIXED_LOCAL_WIDTH {
        if coordinate == 0 {
            items.push(PointIdentityItem {
                hidden: vec![PointTableRef::AuxSigma(0), PointTableRef::MaskZ(0)],
                masked: false,
                expression: RelationExpression::table(0).sub(RelationExpression::table(1)),
            });
        } else {
            items.push(PointIdentityItem {
                hidden: vec![
                    PointTableRef::AuxSigma(coordinate),
                    PointTableRef::AuxSigma(coordinate - 1),
                    PointTableRef::MaskZ(coordinate),
                ],
                masked: false,
                expression: RelationExpression::table(0)
                    .sub(RelationExpression::table(1))
                    .sub(RelationExpression::table(2)),
            });
        }
    }

    // sigma^ = D_Z o sigma.
    for coordinate in 0..SECTION6_FIXED_LOCAL_WIDTH {
        items.push(PointIdentityItem {
            hidden: vec![
                PointTableRef::AuxSigmaMasked(coordinate),
                PointTableRef::MaskZ(coordinate),
                PointTableRef::AuxSigma(coordinate),
            ],
            masked: false,
            expression: RelationExpression::table(0)
                .sub(RelationExpression::table(1).mul(RelationExpression::table(2))),
        });
    }

    // Z~ = D_Z o X'.
    for coordinate in 0..SECTION6_FIXED_LOCAL_WIDTH {
        items.push(PointIdentityItem {
            hidden: vec![
                PointTableRef::AuxZTilde(coordinate),
                PointTableRef::MaskZ(coordinate),
                PointTableRef::AuxXSelected(coordinate),
            ],
            masked: false,
            expression: RelationExpression::table(0)
                .sub(RelationExpression::table(1).mul(RelationExpression::table(2))),
        });
    }

    // P_0 = mask o factor_0; P_k = P_{k-1} o factor_k. The first 2m factors
    // read X'; the last d factors read the public batching scalars.
    for coordinate in 0..SECTION6_FIXED_ETA_WIDTH {
        let (hidden, masked, factor) = if coordinate < SECTION6_FIXED_LOCAL_WIDTH {
            let hidden = if coordinate == 0 {
                vec![
                    PointTableRef::AuxPrefix(0),
                    PointTableRef::MaskEta(0),
                    PointTableRef::AuxXSelected(0),
                ]
            } else {
                vec![
                    PointTableRef::AuxPrefix(coordinate),
                    PointTableRef::AuxPrefix(coordinate - 1),
                    PointTableRef::MaskEta(coordinate),
                    PointTableRef::AuxXSelected(coordinate),
                ]
            };
            let mask_table = if coordinate == 0 { 1 } else { 2 };
            let point_table = if coordinate == 0 { 2 } else { 3 };
            let factor = RelationExpression::constant(Fr::one())
                .sub(RelationExpression::table(mask_table))
                .add(
                    RelationExpression::table(mask_table)
                        .mul(RelationExpression::table(point_table)),
                );
            (hidden, coordinate == 0, factor)
        } else {
            let scalar = batching_point[coordinate - SECTION6_FIXED_LOCAL_WIDTH];
            let hidden = vec![
                PointTableRef::AuxPrefix(coordinate),
                PointTableRef::AuxPrefix(coordinate - 1),
                PointTableRef::MaskEta(coordinate),
            ];
            let factor = RelationExpression::constant(Fr::one())
                .sub(RelationExpression::table(2))
                .add(RelationExpression::table(2).mul(RelationExpression::constant(scalar)));
            (hidden, false, factor)
        };
        let previous = if coordinate == 0 {
            // The first prefix multiplies the public real-row mask, which is
            // appended after the hidden tables.
            RelationExpression::table(hidden.len())
        } else {
            RelationExpression::table(1)
        };
        items.push(PointIdentityItem {
            hidden,
            masked,
            expression: RelationExpression::table(0).sub(previous.mul(factor)),
        });
    }

    items
}

fn scoped_label(label: &[u8], role: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(POINT_DOMAIN.len() + label.len() + role.len() + 16);
    output.extend_from_slice(POINT_DOMAIN);
    output.extend_from_slice(&(label.len() as u64).to_le_bytes());
    output.extend_from_slice(label);
    output.extend_from_slice(&(role.len() as u64).to_le_bytes());
    output.extend_from_slice(role);
    output
}

/// Build the caller-owned prover identity plan for one point compilation.
pub fn build_section6_point_prover_plan<'a>(
    masks: &'a CommittedSection6ShapeMasks,
    aux: &'a CommittedSection6PointAux,
    public: &'a Section6PointPublicTables,
) -> Result<Section6PointProverPlan<'a>, Section6PointCompilationError> {
    let items = point_identity_items(&aux.witness().batching_point);
    let mut plan = Section6IdentityPlan::new(PublicRelationTable {
        values: &public.real_row_mask,
    });
    for item in items {
        let tables = item
            .hidden
            .iter()
            .map(|reference| resolve_prover_table(masks, aux, reference))
            .collect::<Result<Vec<_>, _>>()?;
        plan.push(tables, item.masked, item.expression);
    }
    Ok(plan)
}

/// Build the caller-owned verifier identity plan for one point compilation.
pub fn build_section6_point_verifier_plan<'a>(
    layout: &Section6PublicLayout,
    mask_statement: &'a Section6ShapeMaskStatement,
    aux_statement: &'a Section6PointAuxStatement,
    public: &'a Section6PointPublicTables,
    batching_point: &[Fr],
) -> Result<Section6PointVerifierPlan<'a>, Section6PointCompilationError> {
    if batching_point.len() != SECTION6_FIXED_MATRIX_LOG
        || public.row_tags.len() != layout.row_cover()?
    {
        return Err(Section6PointCompilationError::ProofShape);
    }
    let items = point_identity_items(batching_point);
    let mut plan = Section6IdentityPlan::new(PublicRelationTable {
        values: &public.real_row_mask,
    });
    for item in items {
        let tables = item
            .hidden
            .iter()
            .map(|reference| resolve_verifier_table(mask_statement, aux_statement, reference))
            .collect::<Vec<_>>();
        plan.push(tables, item.masked, item.expression);
    }
    Ok(plan)
}

/// Reduce the complete point compilation while deferring every Dory opening.
#[allow(clippy::too_many_arguments)]
pub fn prove_section6_point_compilation_deferred<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    masks: &'a CommittedSection6ShapeMasks,
    x: &'a CommittedSection6PointMatrix,
    aux: &'a CommittedSection6PointAux,
    public: &'a Section6PointPublicTables,
    plan: &'a Section6PointProverPlan<'a>,
    rng: &mut impl RngCore,
) -> Result<
    (
        Section6PointCompilationProof,
        Section6PointProverOpenings<'a>,
    ),
    Section6PointCompilationError,
> {
    let layout = masks.layout();
    let public_layout = layout.public();
    let rows = layout.row_cover()?;
    let local_width = public_layout.local_width();
    if aux.statement.padded_rows != rows
        || aux.statement.local_width != local_width
        || x.statement.padded_rows != rows
        || x.statement.matrix_count != layout.matrix_count
        || public.row_tags.len() != rows
    {
        return Err(Section6PointCompilationError::InvalidWitness(
            "committed statements do not share the layout row cover".to_owned(),
        ));
    }
    let witness = aux.witness();
    for col in 0..SECTION6_FIXED_LOCAL_WIDTH {
        let committed_column = x
            .column_values(col)
            .ok_or(Section6PointCompilationError::ProofShape)?;
        let witness_column = witness
            .x
            .padded_column(col)
            .ok_or(Section6PointCompilationError::ProofShape)?;
        if committed_column != witness_column {
            return Err(Section6PointCompilationError::InvalidWitness(
                "the compiled witness X differs from the committed point matrix".to_owned(),
            ));
        }
    }

    // Source-row lookup: {(tau, X'_k)} subset {(n, X_k)}.
    let aux_columns = aux.layout_columns();
    let tau_table = aux
        .table(aux_columns.tau())
        .ok_or(Section6PointCompilationError::ProofShape)?;
    let mut source_query_part = vec![MixedLogUpColumn::Hidden(tau_table)];
    let mut source_table_part = vec![MixedLogUpColumn::Public(PublicRelationTable {
        values: &public.source_index,
    })];
    for col in 0..SECTION6_FIXED_LOCAL_WIDTH {
        source_query_part.push(MixedLogUpColumn::Hidden(
            aux.table(aux_columns.x_selected(col))
                .ok_or(Section6PointCompilationError::ProofShape)?,
        ));
        source_table_part.push(MixedLogUpColumn::Hidden(HiddenRelationTable {
            values: x
                .column_values(col)
                .ok_or(Section6PointCompilationError::ProofShape)?,
            committed: x
                .columns
                .get(col)
                .ok_or(Section6PointCompilationError::ProofShape)?,
        }));
    }
    // Padding bucket rows contribute all-zero query tuples, so the table
    // carries one public all-zero part next to the real source rows.
    let source_null_part = vec![
        MixedLogUpColumn::Public(PublicRelationTable {
            values: &public.zero,
        });
        1 + SECTION6_FIXED_LOCAL_WIDTH
    ];
    let (source_lookup, source_state) = prove_hidden_logup_parts_deferred_on_grid(
        &scoped_label(label, b"source-row-lookup"),
        parameters,
        grid,
        &[source_query_part],
        &[source_table_part, source_null_part],
        rng,
    )?;

    // Compaction lookup: {(row, sigma^_k, Z~_k)} subset
    // {(row, h+1, Z_h)} union {(row, 0, 0)}.
    let row_tag_column = MixedLogUpColumn::Public(PublicRelationTable {
        values: &public.row_tags,
    });
    let compaction_query_parts = (0..SECTION6_FIXED_LOCAL_WIDTH)
        .map(|col| {
            Ok(vec![
                row_tag_column,
                MixedLogUpColumn::Hidden(
                    aux.table(aux_columns.sigma_masked(col))
                        .ok_or(Section6PointCompilationError::ProofShape)?,
                ),
                MixedLogUpColumn::Hidden(
                    aux.table(aux_columns.z_tilde(col))
                        .ok_or(Section6PointCompilationError::ProofShape)?,
                ),
            ])
        })
        .collect::<Result<Vec<_>, Section6PointCompilationError>>()?;
    let mut compaction_table_parts = (0..local_width)
        .map(|position| {
            Ok(vec![
                row_tag_column,
                MixedLogUpColumn::Public(PublicRelationTable {
                    values: &public.position_constants[position],
                }),
                MixedLogUpColumn::Hidden(
                    aux.table(aux_columns.z(position))
                        .ok_or(Section6PointCompilationError::ProofShape)?,
                ),
            ])
        })
        .collect::<Result<Vec<_>, Section6PointCompilationError>>()?;
    compaction_table_parts.push(vec![
        row_tag_column,
        MixedLogUpColumn::Public(PublicRelationTable {
            values: &public.zero,
        }),
        MixedLogUpColumn::Public(PublicRelationTable {
            values: &public.zero,
        }),
    ]);
    let (compaction_lookup, compaction_state) = prove_hidden_logup_parts_deferred_on_grid(
        &scoped_label(label, b"compaction-lookup"),
        parameters,
        grid,
        &compaction_query_parts,
        &compaction_table_parts,
        rng,
    )?;

    // Identity batch for tau decoding, ranks, masked products, and prefixes.
    let batch_items = plan.prover_items();
    let (identity, identity_opening) = prove_hidden_relations_batch_deferred(
        &scoped_label(label, b"identity-batch"),
        parameters,
        &batch_items,
        rng,
    )?;

    Ok((
        Section6PointCompilationProof {
            aux: aux.statement.clone(),
            source_lookup,
            compaction_lookup,
            identity,
        },
        Section6PointProverOpenings {
            identity: identity_opening,
            source_lookup: source_state,
            compaction_lookup: compaction_state,
        },
    ))
}

fn resolve_prover_table<'a>(
    masks: &'a CommittedSection6ShapeMasks,
    aux: &'a CommittedSection6PointAux,
    reference: &PointTableRef,
) -> Result<HiddenRelationTable<'a>, Section6PointCompilationError> {
    let columns = aux.layout_columns();
    let table = match reference {
        PointTableRef::AuxTau => aux.table(columns.tau()),
        PointTableRef::AuxXSelected(col) => aux.table(columns.x_selected(*col)),
        PointTableRef::AuxSigma(col) => aux.table(columns.sigma(*col)),
        PointTableRef::AuxSigmaMasked(col) => aux.table(columns.sigma_masked(*col)),
        PointTableRef::AuxZTilde(col) => aux.table(columns.z_tilde(*col)),
        PointTableRef::AuxPrefix(col) => aux.table(columns.eta_prefix(*col)),
        PointTableRef::MaskEta(col) => Some(HiddenRelationTable {
            values: masks
                .d_eta_column_values(*col)
                .ok_or(Section6PointCompilationError::ProofShape)?,
            committed: masks
                .d_eta_columns()
                .get(*col)
                .ok_or(Section6PointCompilationError::ProofShape)?,
        }),
        PointTableRef::MaskZ(col) => Some(HiddenRelationTable {
            values: masks
                .d_z_column_values(*col)
                .ok_or(Section6PointCompilationError::ProofShape)?,
            committed: masks
                .d_z_columns()
                .get(*col)
                .ok_or(Section6PointCompilationError::ProofShape)?,
        }),
    };
    table.ok_or(Section6PointCompilationError::ProofShape)
}

fn resolve_verifier_table<'a>(
    mask_statement: &'a Section6ShapeMaskStatement,
    aux_statement: &'a Section6PointAuxStatement,
    reference: &PointTableRef,
) -> HiddenRelationTableStatement<'a> {
    let columns = Section6PointAuxColumns::new(aux_statement.local_width);
    let aux_column = |column: usize| {
        HiddenRelationTableStatement::matrix_column(
            &aux_statement.commitment,
            aux_statement.padded_rows,
            aux_statement.padded_cols,
            column,
        )
    };
    match reference {
        PointTableRef::AuxTau => aux_column(columns.tau()),
        PointTableRef::AuxXSelected(col) => aux_column(columns.x_selected(*col)),
        PointTableRef::AuxSigma(col) => aux_column(columns.sigma(*col)),
        PointTableRef::AuxSigmaMasked(col) => aux_column(columns.sigma_masked(*col)),
        PointTableRef::AuxZTilde(col) => aux_column(columns.z_tilde(*col)),
        PointTableRef::AuxPrefix(col) => aux_column(columns.eta_prefix(*col)),
        PointTableRef::MaskEta(col) => HiddenRelationTableStatement::matrix_column(
            &mask_statement.d_eta_commitment,
            mask_statement.padded_rows,
            Section6ShapeMaskStatement::padded_cols(mask_statement.eta_mask_width),
            *col,
        ),
        PointTableRef::MaskZ(col) => HiddenRelationTableStatement::matrix_column(
            &mask_statement.d_z_commitment,
            mask_statement.padded_rows,
            Section6ShapeMaskStatement::padded_cols(mask_statement.local_mask_width),
            *col,
        ),
    }
}

/// Verify the deferred point compilation from public statements only.
#[allow(clippy::too_many_arguments)]
pub fn verify_section6_point_compilation_deferred<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    grid: &Section6DoryGrid,
    layout: &Section6PublicLayout,
    mask_statement: &'a Section6ShapeMaskStatement,
    x_statement: &'a Section6PointMatrixStatement,
    public: &'a Section6PointPublicTables,
    plan: &'a Section6PointVerifierPlan<'a>,
    proof: &'a Section6PointCompilationProof,
) -> Result<Section6PointVerifierOpenings<'a>, Section6PointCompilationError> {
    layout.validate()?;
    mask_statement.validate(layout)?;
    x_statement.validate(layout)?;
    proof.aux.validate(layout)?;
    if public.row_tags.len() != layout.row_cover()? {
        return Err(Section6PointCompilationError::ProofShape);
    }
    let rows = layout.row_cover()?;
    let local_width = layout.local_width();
    let aux_statement = &proof.aux;
    let columns = Section6PointAuxColumns::new(local_width);
    let aux_column = |column: usize| {
        HiddenRelationTableStatement::matrix_column(
            &aux_statement.commitment,
            rows,
            aux_statement.padded_cols,
            column,
        )
    };

    let mut source_query_part = vec![MixedLogUpColumnStatement::Hidden(aux_column(
        columns.tau(),
    ))];
    let mut source_table_part = vec![MixedLogUpColumnStatement::Public(PublicRelationTable {
        values: &public.source_index,
    })];
    for col in 0..SECTION6_FIXED_LOCAL_WIDTH {
        source_query_part.push(MixedLogUpColumnStatement::Hidden(aux_column(
            columns.x_selected(col),
        )));
        source_table_part.push(MixedLogUpColumnStatement::Hidden(
            HiddenRelationTableStatement::matrix_column(
                &x_statement.commitment,
                rows,
                x_statement.padded_cols,
                col,
            ),
        ));
    }
    let source_null_part = vec![
        MixedLogUpColumnStatement::Public(PublicRelationTable {
            values: &public.zero,
        });
        1 + SECTION6_FIXED_LOCAL_WIDTH
    ];
    let source_lookup = verify_hidden_logup_parts_deferred_on_grid(
        &scoped_label(label, b"source-row-lookup"),
        parameters,
        grid,
        &[source_query_part],
        &[source_table_part, source_null_part],
        &proof.source_lookup,
    )?;

    let row_tag_column = MixedLogUpColumnStatement::Public(PublicRelationTable {
        values: &public.row_tags,
    });
    let compaction_query_parts = (0..SECTION6_FIXED_LOCAL_WIDTH)
        .map(|col| {
            vec![
                row_tag_column,
                MixedLogUpColumnStatement::Hidden(aux_column(columns.sigma_masked(col))),
                MixedLogUpColumnStatement::Hidden(aux_column(columns.z_tilde(col))),
            ]
        })
        .collect::<Vec<_>>();
    let mut compaction_table_parts = (0..local_width)
        .map(|position| {
            vec![
                row_tag_column,
                MixedLogUpColumnStatement::Public(PublicRelationTable {
                    values: &public.position_constants[position],
                }),
                MixedLogUpColumnStatement::Hidden(aux_column(columns.z(position))),
            ]
        })
        .collect::<Vec<_>>();
    compaction_table_parts.push(vec![
        row_tag_column,
        MixedLogUpColumnStatement::Public(PublicRelationTable {
            values: &public.zero,
        }),
        MixedLogUpColumnStatement::Public(PublicRelationTable {
            values: &public.zero,
        }),
    ]);
    let compaction_lookup = verify_hidden_logup_parts_deferred_on_grid(
        &scoped_label(label, b"compaction-lookup"),
        parameters,
        grid,
        &compaction_query_parts,
        &compaction_table_parts,
        &proof.compaction_lookup,
    )?;

    let batch_items = plan.verifier_items();
    let identity = verify_hidden_relations_batch_deferred(
        &scoped_label(label, b"identity-batch"),
        parameters,
        &batch_items,
        &proof.identity,
    )?;

    Ok(Section6PointVerifierOpenings {
        identity,
        source_lookup,
        compaction_lookup,
    })
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        jolt_dory::setup_dory_parameters,
        matrix_collection::DenseBvaDescriptor,
        section6_claim_collection::{
            Section6ProverClaimCollection, Section6VerifierClaimCollection,
        },
        section6_data::{
            section6_batching_point, PrivateShapeMetadata, Section6MaskMatrixClaim,
        },
        section6_final_opening::{
            prove_section6_final_opening, verify_section6_final_opening,
        },
        section6_shape_masks::commit_section6_shape_masks_on_grid_zk,
    };

    fn field_values(start: u64, len: usize) -> Vec<Fr> {
        (0..len)
            .map(|index| Fr::from(start + index as u64))
            .collect()
    }

    #[test]
    fn point_compilation_round_trips_and_defers_all_openings() {
        let shapes = [(3, 5), (5, 2)];
        let descriptor = DenseBvaDescriptor::from_shapes(&shapes).unwrap();
        let layout = Section6Layout::from_descriptor(&descriptor, 8).unwrap();
        let metadata = PrivateShapeMetadata::from_descriptor(&layout, &descriptor).unwrap();
        let claims = vec![
            Section6MaskMatrixClaim::from_unpadded(
                &field_values(11, 2),
                &field_values(21, 3),
                Fr::from(31_u64),
            )
            .unwrap(),
            Section6MaskMatrixClaim::from_unpadded(
                &field_values(41, 3),
                &field_values(51, 1),
                Fr::from(61_u64),
            )
            .unwrap(),
        ];
        let omega = Fr::from(7_u64);

        let grid = Section6DoryGrid::new(14).unwrap();
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(grid.num_vars());
        let mut rng = StdRng::seed_from_u64(0x504f_494e_545f_5632);

        let masks = commit_section6_shape_masks_on_grid_zk(
            &prover_parameters,
            &grid,
            layout.clone(),
            metadata.clone(),
        )
        .unwrap();
        let x = commit_section6_point_matrix_on_grid_zk(
            &prover_parameters,
            &grid,
            &layout,
            &claims,
        )
        .unwrap();
        let witness =
            Section6MaskPointWitness::compile(&layout, &descriptor, &metadata, &claims, omega)
                .unwrap();
        let aux =
            commit_section6_point_aux_on_grid_zk(&prover_parameters, &grid, &layout, witness)
                .unwrap();
        let public = Section6PointPublicTables::derive(&layout.public()).unwrap();

        let plan = build_section6_point_prover_plan(&masks, &aux, &public).unwrap();
        let (proof, prover_openings) = prove_section6_point_compilation_deferred(
            b"point-test",
            &prover_parameters,
            &grid,
            &masks,
            &x,
            &aux,
            &public,
            &plan,
            &mut rng,
        )
        .unwrap();

        let verifier_plan = build_section6_point_verifier_plan(
            &layout.public(),
            masks.statement(),
            &proof.aux,
            &public,
            &section6_batching_point(omega),
        )
        .unwrap();
        let verifier_openings = verify_section6_point_compilation_deferred(
            b"point-test",
            &verifier_parameters,
            &grid,
            &layout.public(),
            masks.statement(),
            x.statement(),
            &public,
            &verifier_plan,
            &proof,
        )
        .unwrap();

        // Every deferred claim resolves in one shared terminal batch.
        let mut prover_claims = Section6ProverClaimCollection::new();
        prover_claims
            .push_relation(&prover_openings.identity, &grid)
            .unwrap();
        for opening in prover_openings.source_lookup.relation_openings().unwrap() {
            prover_claims.push_relation(&opening, &grid).unwrap();
        }
        for opening in prover_openings
            .compaction_lookup
            .relation_openings()
            .unwrap()
        {
            prover_claims.push_relation(&opening, &grid).unwrap();
        }
        let terminal = prove_section6_final_opening(
            b"point-test-terminal",
            b"outer",
            &prover_parameters,
            &grid,
            &prover_claims.claims(),
            &mut rng,
        )
        .unwrap();

        let mut verifier_claims = Section6VerifierClaimCollection::new();
        verifier_claims
            .push_relation(&verifier_openings.identity, &grid)
            .unwrap();
        for opening in verifier_openings
            .source_lookup
            .relation_openings()
            .unwrap()
        {
            verifier_claims.push_relation(&opening, &grid).unwrap();
        }
        for opening in verifier_openings
            .compaction_lookup
            .relation_openings()
            .unwrap()
        {
            verifier_claims.push_relation(&opening, &grid).unwrap();
        }
        verify_section6_final_opening(
            b"point-test-terminal",
            b"outer",
            &verifier_parameters,
            &grid,
            &verifier_claims.claims(),
            &terminal,
        )
        .unwrap();

        // A tampered aux commitment must break verification.
        let mut tampered = proof.clone();
        tampered.aux.commitment = x.statement().commitment.clone();
        let tampered_plan = build_section6_point_verifier_plan(
            &layout.public(),
            masks.statement(),
            &tampered.aux,
            &public,
            &section6_batching_point(omega),
        )
        .unwrap();
        assert!(verify_section6_point_compilation_deferred(
            b"point-test",
            &verifier_parameters,
            &grid,
            &layout.public(),
            masks.statement(),
            x.statement(),
            &public,
            &tampered_plan,
            &tampered,
        )
        .is_err());
    }
}
