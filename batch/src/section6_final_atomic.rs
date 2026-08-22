//! Deferred final TBV evaluation for the mask-based Section 6 flow.
//!
//! After Packed EO fixes the evaluation point `y`, the prover commits one
//! auxiliary matrix holding the derived prefix mask `D_Z'`, the eta' prefixes
//! `P'`, the restriction points `Z' = D_Z' o (1 y^T)`, and the product
//! prefixes `Q` with `Q_0 = eta o eta'` and `v = Q_w`. One Fiat--Shamir
//! identity batch proves the recurrences against the registered offset mask
//! `D_eta'` and the point-compilation columns, and one hidden scalar relation
//! binds the mean of `v` to the Packed EO reduced claim. All Dory openings
//! are deferred to the shared terminal batch.

use std::fmt::{Display, Formatter};

use ark_bn254::Fr;
use ark_ff::{Field, One, Zero};
use ark_std::rand::RngCore;
use jolt_crypto::Bn254G1;
use jolt_dory::DoryCommitment;
use jolt_field::Fr as JoltFr;
use serde::{Deserialize, Serialize};

use crate::{
    hidden_packed_eo::{HiddenPackedEoProverState, HiddenPackedEoVerification},
    hidden_relation::{
        prove_hidden_scalar_relation, verify_hidden_scalar_relation, HiddenRelationError,
        HiddenRelationProverOpening, HiddenRelationTable, HiddenRelationTableStatement,
        HiddenRelationVerifierOpening, HiddenScalarRelationProof, HiddenScalarWitness,
        PublicRelationTable, RelationExpression,
    },
    hidden_relation_batch::{
        prove_hidden_relations_batch_deferred, verify_hidden_relations_batch_deferred,
        HiddenRelationBatchReductionProof,
    },
    jolt_dory::{CommittedDoryFieldVector, DoryProverParameters, DoryVerifierParameters},
    section6_data::{
        Section6DataError, Section6Layout, Section6MaskFinalWitness, Section6PublicLayout,
        SECTION6_FIXED_LOCAL_WIDTH,
    },
    section6_final_opening::{
        commit_section6_field_matrix_columns_on_grid_zk,
        resolve_section6_hidden_relation_prover_claim,
        resolve_section6_hidden_relation_verifier_claim, ResolvedSection6FieldProverClaim,
        ResolvedSection6FieldVerifierClaim, Section6DoryGrid, Section6FinalOpeningError,
    },
    section6_point_compilation::{
        CommittedSection6PointAux, Section6IdentityPlan, Section6PointAuxColumns,
        Section6PointAuxStatement, Section6PointCompilationError, Section6PointPublicTables,
    },
    section6_shape_masks::{CommittedSection6ShapeMasks, Section6ShapeMaskStatement},
};

const FINAL_ATOMIC_DOMAIN: &[u8] = b"parcomm-section6-mask-final-atomic-v2";

/// Column offsets of the one committed final auxiliary matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Section6FinalAuxColumns {
    /// Width `min(2m, l)` of the restriction blocks.
    pub local_width: usize,
    /// Global point length `l`.
    pub global_log: usize,
}

impl Section6FinalAuxColumns {
    pub fn new(local_width: usize, global_log: usize) -> Self {
        Self {
            local_width,
            global_log,
        }
    }

    pub fn d_z_prime(self, coordinate: usize) -> usize {
        coordinate
    }

    pub fn eta_prime_prefix(self, coordinate: usize) -> usize {
        self.local_width + coordinate
    }

    /// The restriction scale `eta'` is the last eta' prefix column.
    pub fn eta_prime(self) -> usize {
        self.eta_prime_prefix(self.global_log - 1)
    }

    pub fn z_prime(self, coordinate: usize) -> usize {
        self.local_width + self.global_log + coordinate
    }

    pub fn q(self, coordinate: usize) -> usize {
        2 * self.local_width + self.global_log + coordinate
    }

    /// The final per-bucket vector `v` is the last product prefix column.
    pub fn v(self) -> usize {
        self.q(self.local_width)
    }

    pub fn count(self) -> usize {
        3 * self.local_width + self.global_log + 1
    }

    pub fn padded_cols(self) -> usize {
        self.count().next_power_of_two()
    }
}

/// Verifier-visible statement for the committed final auxiliary matrix.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Section6FinalAuxStatement {
    pub commitment: DoryCommitment,
    pub padded_rows: usize,
    pub padded_cols: usize,
    pub local_width: usize,
    pub global_log: usize,
}

impl Section6FinalAuxStatement {
    pub fn validate(&self, layout: &Section6PublicLayout) -> Result<(), Section6DataError> {
        let columns = Section6FinalAuxColumns::new(layout.local_width(), layout.global_log);
        if self.local_width != layout.local_width()
            || self.global_log != layout.global_log
            || self.padded_rows != layout.row_cover()?
            || self.padded_cols != columns.padded_cols()
        {
            return Err(Section6DataError::InvalidPoint(
                "final auxiliary statement does not match the public layout".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Prover-only committed final auxiliary matrix owning the compiled witness.
pub struct CommittedSection6FinalAux {
    statement: Section6FinalAuxStatement,
    witness: Section6MaskFinalWitness<Fr>,
    columns: Vec<CommittedDoryFieldVector>,
}

impl CommittedSection6FinalAux {
    pub fn statement(&self) -> &Section6FinalAuxStatement {
        &self.statement
    }

    pub fn witness(&self) -> &Section6MaskFinalWitness<Fr> {
        &self.witness
    }

    pub fn layout_columns(&self) -> Section6FinalAuxColumns {
        Section6FinalAuxColumns::new(self.statement.local_width, self.statement.global_log)
    }

    fn column_values(&self, column: usize) -> Option<&[Fr]> {
        let columns = self.layout_columns();
        let ranges = [
            (columns.d_z_prime(0), &self.witness.d_z_prime),
            (columns.eta_prime_prefix(0), &self.witness.eta_prime_prefix),
            (columns.z_prime(0), &self.witness.z_prime),
            (columns.q(0), &self.witness.v_prefix),
        ];
        for (start, matrix) in ranges {
            if column >= start && column < start + matrix.logical_cols {
                return matrix.padded_column(column - start);
            }
        }
        None
    }

    /// One hidden-relation table view of a final auxiliary column.
    pub fn table(&self, column: usize) -> Option<HiddenRelationTable<'_>> {
        Some(HiddenRelationTable {
            values: self.column_values(column)?,
            committed: self.columns.get(column)?,
        })
    }

    pub fn v_values(&self) -> Option<&[Fr]> {
        self.column_values(self.layout_columns().v())
    }

    pub fn v_committed(&self) -> Option<&CommittedDoryFieldVector> {
        self.columns.get(self.layout_columns().v())
    }
}

/// Serializable final reduction with no PCS opening embedded in it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct Section6FinalAtomicProof {
    pub aux: Section6FinalAuxStatement,
    pub identity: HiddenRelationBatchReductionProof,
    /// Pedersen commitment to the all-half evaluation of `v`.
    pub average_commitment: Bn254G1,
    /// Hidden scalar identity `padded_rows * average = b_reduced`.
    pub scalar_binding: HiddenScalarRelationProof,
}

/// Prover-side deferred openings produced by the final reduction.
pub struct Section6FinalProverOpenings<'a> {
    pub identity: HiddenRelationProverOpening<'a>,
    pub average: HiddenScalarWitness,
    pub all_half_point: Vec<Fr>,
}

impl Section6FinalProverOpenings<'_> {
    /// Resolve the `v` column at the all-half point for the terminal batch.
    pub fn resolve_v_terminal_claim<'b>(
        &self,
        final_aux: &'b CommittedSection6FinalAux,
        grid: &Section6DoryGrid,
    ) -> Result<ResolvedSection6FieldProverClaim<'b>, Section6FinalAtomicError> {
        let values = final_aux
            .v_values()
            .ok_or(Section6FinalAtomicError::ProofShape)?;
        let committed = final_aux
            .v_committed()
            .ok_or(Section6FinalAtomicError::ProofShape)?;
        let point = self
            .all_half_point
            .iter()
            .copied()
            .map(JoltFr::from)
            .collect::<Vec<_>>();
        Ok(resolve_section6_hidden_relation_prover_claim(
            HiddenRelationTable { values, committed },
            &point,
            grid,
        )?)
    }
}

/// Verifier-side deferred openings mirroring the prover structure.
pub struct Section6FinalVerifierOpenings<'a> {
    pub identity: HiddenRelationVerifierOpening<'a>,
    pub average_commitment: Bn254G1,
    pub all_half_point: Vec<Fr>,
    v_statement: HiddenRelationTableStatement<'a>,
}

impl Section6FinalVerifierOpenings<'_> {
    /// Resolve the public `v` commitment view for the terminal Dory batch.
    pub fn resolve_v_terminal_claim(
        &self,
        grid: &Section6DoryGrid,
    ) -> Result<ResolvedSection6FieldVerifierClaim<'_>, Section6FinalAtomicError> {
        let point = self
            .all_half_point
            .iter()
            .copied()
            .map(JoltFr::from)
            .collect::<Vec<_>>();
        Ok(resolve_section6_hidden_relation_verifier_claim(
            self.v_statement,
            &point,
            self.average_commitment,
            grid,
        )?)
    }
}

#[derive(Debug)]
pub enum Section6FinalAtomicError {
    Data(Section6DataError),
    Relation(HiddenRelationError),
    Opening(Section6FinalOpeningError),
    Point(Section6PointCompilationError),
    InvalidWitness(String),
    ProofShape,
}

impl Display for Section6FinalAtomicError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Data(error) => Display::fmt(error, formatter),
            Self::Relation(error) => Display::fmt(error, formatter),
            Self::Opening(error) => Display::fmt(error, formatter),
            Self::Point(error) => Display::fmt(error, formatter),
            Self::InvalidWitness(message) => {
                write!(formatter, "invalid final witness: {message}")
            }
            Self::ProofShape => formatter.write_str("invalid final atomic proof shape"),
        }
    }
}

impl std::error::Error for Section6FinalAtomicError {}

impl From<Section6DataError> for Section6FinalAtomicError {
    fn from(error: Section6DataError) -> Self {
        Self::Data(error)
    }
}

impl From<HiddenRelationError> for Section6FinalAtomicError {
    fn from(error: HiddenRelationError) -> Self {
        Self::Relation(error)
    }
}

impl From<Section6FinalOpeningError> for Section6FinalAtomicError {
    fn from(error: Section6FinalOpeningError) -> Self {
        Self::Opening(error)
    }
}

impl From<Section6PointCompilationError> for Section6FinalAtomicError {
    fn from(error: Section6PointCompilationError) -> Self {
        Self::Point(error)
    }
}

/// Commit the final auxiliary matrix for one compiled final witness.
pub fn commit_section6_final_aux_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    layout: &Section6Layout,
    witness: Section6MaskFinalWitness<Fr>,
) -> Result<CommittedSection6FinalAux, Section6FinalAtomicError> {
    let rows = layout.row_cover()?;
    let public_layout = layout.public();
    let local_width = public_layout.local_width();
    if witness.d_z_prime.padded_rows != rows
        || witness.d_z_prime.logical_cols != local_width
        || witness.eta_prime_prefix.padded_rows != rows
        || witness.eta_prime_prefix.logical_cols != layout.global_log
        || witness.z_prime.padded_rows != rows
        || witness.z_prime.logical_cols != local_width
        || witness.v_prefix.padded_rows != rows
        || witness.v_prefix.logical_cols != local_width + 1
    {
        return Err(Section6FinalAtomicError::InvalidWitness(
            "final witness covers do not match the layout".to_owned(),
        ));
    }
    let columns_layout = Section6FinalAuxColumns::new(local_width, layout.global_log);
    let mut column_slices = Vec::with_capacity(columns_layout.count());
    column_slices.extend(witness.d_z_prime.padded_columns());
    column_slices.extend(witness.eta_prime_prefix.padded_columns());
    column_slices.extend(witness.z_prime.padded_columns());
    column_slices.extend(witness.v_prefix.padded_columns());
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
        .ok_or(Section6FinalAtomicError::ProofShape)?;
    Ok(CommittedSection6FinalAux {
        statement: Section6FinalAuxStatement {
            commitment,
            padded_rows: rows,
            padded_cols: columns_layout.padded_cols(),
            local_width,
            global_log: layout.global_log,
        },
        witness,
        columns,
    })
}

/// Prover-side identity plan for the final reduction.
pub type Section6FinalProverPlan<'a> = Section6IdentityPlan<'a, HiddenRelationTable<'a>>;

/// Verifier-side identity plan for the final reduction.
pub type Section6FinalVerifierPlan<'a> = Section6IdentityPlan<'a, HiddenRelationTableStatement<'a>>;

/// Build the caller-owned prover identity plan for the final reduction.
pub fn build_section6_final_prover_plan<'a>(
    masks: &'a CommittedSection6ShapeMasks,
    point_aux: &'a CommittedSection6PointAux,
    final_aux: &'a CommittedSection6FinalAux,
    public: &'a Section6PointPublicTables,
    y: &[Fr],
) -> Result<Section6FinalProverPlan<'a>, Section6FinalAtomicError> {
    let layout = masks.layout();
    if y.len() != layout.global_log {
        return Err(Section6FinalAtomicError::InvalidWitness(
            "the final point length does not match the layout".to_owned(),
        ));
    }
    let items = final_identity_items(layout.public().local_width(), layout.global_log, y);
    let mut plan = Section6IdentityPlan::new(PublicRelationTable {
        values: &public.real_row_mask,
    });
    for item in items {
        let tables = item
            .hidden
            .iter()
            .map(|reference| resolve_prover_table(masks, point_aux, final_aux, reference))
            .collect::<Result<Vec<_>, _>>()?;
        plan.push(tables, item.masked, item.expression);
    }
    Ok(plan)
}

/// Build the caller-owned verifier identity plan for the final reduction.
pub fn build_section6_final_verifier_plan<'a>(
    layout: &Section6PublicLayout,
    mask_statement: &'a Section6ShapeMaskStatement,
    point_statement: &'a Section6PointAuxStatement,
    final_statement: &'a Section6FinalAuxStatement,
    public: &'a Section6PointPublicTables,
    y: &[Fr],
) -> Result<Section6FinalVerifierPlan<'a>, Section6FinalAtomicError> {
    if y.len() != layout.global_log {
        return Err(Section6FinalAtomicError::ProofShape);
    }
    let items = final_identity_items(layout.local_width(), layout.global_log, y);
    let mut plan = Section6IdentityPlan::new(PublicRelationTable {
        values: &public.real_row_mask,
    });
    for item in items {
        let tables = item
            .hidden
            .iter()
            .map(|reference| {
                resolve_verifier_table(mask_statement, point_statement, final_statement, reference)
            })
            .collect::<Vec<_>>();
        plan.push(tables, item.masked, item.expression);
    }
    Ok(plan)
}

/// One identity of the final batch, in shared table order.
struct FinalIdentityItem {
    hidden: Vec<FinalTableRef>,
    masked: bool,
    expression: RelationExpression,
}

/// Symbolic operand references resolved by the prover and the verifier.
#[derive(Clone, Copy)]
enum FinalTableRef {
    Fin(FinalColumn),
    PointSigmaLast,
    PointEta,
    PointZ(usize),
    MaskEtaPrime(usize),
}

#[derive(Clone, Copy)]
enum FinalColumn {
    DZPrime(usize),
    EtaPrimePrefix(usize),
    ZPrime(usize),
    Q(usize),
}

/// The complete final identity batch in a fixed order shared by both sides.
fn final_identity_items(local_width: usize, global_log: usize, y: &[Fr]) -> Vec<FinalIdentityItem> {
    let mut items = Vec::new();

    // D_Z' entries are Boolean.
    for coordinate in 0..local_width {
        items.push(FinalIdentityItem {
            hidden: vec![FinalTableRef::Fin(FinalColumn::DZPrime(coordinate))],
            masked: false,
            expression: RelationExpression::table(0)
                .sub(RelationExpression::table(0).mul(RelationExpression::table(0))),
        });
    }

    // D_Z' is a monotone prefix: a one at h implies a one at h - 1.
    for coordinate in 1..local_width {
        items.push(FinalIdentityItem {
            hidden: vec![
                FinalTableRef::Fin(FinalColumn::DZPrime(coordinate)),
                FinalTableRef::Fin(FinalColumn::DZPrime(coordinate - 1)),
            ],
            masked: false,
            expression: RelationExpression::table(0)
                .sub(RelationExpression::table(0).mul(RelationExpression::table(1))),
        });
    }

    // The prefix length equals the total rank sigma_{2m-1} = gamma' per row.
    {
        let mut hidden = vec![FinalTableRef::PointSigmaLast];
        for coordinate in 0..local_width {
            hidden.push(FinalTableRef::Fin(FinalColumn::DZPrime(coordinate)));
        }
        let mut sum = RelationExpression::constant(Fr::zero());
        for coordinate in 0..local_width {
            sum = sum.add(RelationExpression::table(1 + coordinate));
        }
        items.push(FinalIdentityItem {
            hidden,
            masked: false,
            expression: RelationExpression::table(0).sub(sum),
        });
    }

    // Z' = D_Z' o (1 y^T).
    for coordinate in 0..local_width {
        items.push(FinalIdentityItem {
            hidden: vec![
                FinalTableRef::Fin(FinalColumn::ZPrime(coordinate)),
                FinalTableRef::Fin(FinalColumn::DZPrime(coordinate)),
            ],
            masked: false,
            expression: RelationExpression::table(0)
                .sub(RelationExpression::table(1).mul(RelationExpression::constant(y[coordinate]))),
        });
    }

    // P'_0 = mask o factor_0; P'_q = P'_{q-1} o factor_q with
    // factor_q = 1 - D_eta'[q] + D_eta'[q] * y_q.
    for coordinate in 0..global_log {
        let factor = |mask_table: usize| {
            RelationExpression::constant(Fr::one())
                .sub(RelationExpression::table(mask_table))
                .add(
                    RelationExpression::table(mask_table)
                        .mul(RelationExpression::constant(y[coordinate])),
                )
        };
        if coordinate == 0 {
            let hidden = vec![
                FinalTableRef::Fin(FinalColumn::EtaPrimePrefix(0)),
                FinalTableRef::MaskEtaPrime(0),
            ];
            let mask_index = hidden.len();
            items.push(FinalIdentityItem {
                hidden,
                masked: true,
                expression: RelationExpression::table(0)
                    .sub(RelationExpression::table(mask_index).mul(factor(1))),
            });
        } else {
            items.push(FinalIdentityItem {
                hidden: vec![
                    FinalTableRef::Fin(FinalColumn::EtaPrimePrefix(coordinate)),
                    FinalTableRef::Fin(FinalColumn::EtaPrimePrefix(coordinate - 1)),
                    FinalTableRef::MaskEtaPrime(coordinate),
                ],
                masked: false,
                expression: RelationExpression::table(0)
                    .sub(RelationExpression::table(1).mul(factor(2))),
            });
        }
    }

    // Q_0 = eta o eta'.
    items.push(FinalIdentityItem {
        hidden: vec![
            FinalTableRef::Fin(FinalColumn::Q(0)),
            FinalTableRef::PointEta,
            FinalTableRef::Fin(FinalColumn::EtaPrimePrefix(global_log - 1)),
        ],
        masked: false,
        expression: RelationExpression::table(0)
            .sub(RelationExpression::table(1).mul(RelationExpression::table(2))),
    });

    // Q_c = Q_{c-1} o (1 + Z_{c-1} o Z'_{c-1}).
    for coordinate in 1..=local_width {
        items.push(FinalIdentityItem {
            hidden: vec![
                FinalTableRef::Fin(FinalColumn::Q(coordinate)),
                FinalTableRef::Fin(FinalColumn::Q(coordinate - 1)),
                FinalTableRef::PointZ(coordinate - 1),
                FinalTableRef::Fin(FinalColumn::ZPrime(coordinate - 1)),
            ],
            masked: false,
            expression: RelationExpression::table(0).sub(
                RelationExpression::table(1).mul(
                    RelationExpression::constant(Fr::one())
                        .add(RelationExpression::table(2).mul(RelationExpression::table(3))),
                ),
            ),
        });
    }

    items
}

fn scoped_label(label: &[u8], role: &[u8], transcript_state: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(
        FINAL_ATOMIC_DOMAIN.len() + label.len() + role.len() + transcript_state.len() + 24,
    );
    output.extend_from_slice(FINAL_ATOMIC_DOMAIN);
    output.extend_from_slice(&(label.len() as u64).to_le_bytes());
    output.extend_from_slice(label);
    output.extend_from_slice(&(role.len() as u64).to_le_bytes());
    output.extend_from_slice(role);
    output.extend_from_slice(&(transcript_state.len() as u64).to_le_bytes());
    output.extend_from_slice(transcript_state);
    output
}

fn all_half_point(row_count: usize) -> Result<Vec<Fr>, Section6FinalAtomicError> {
    if !row_count.is_power_of_two() {
        return Err(Section6FinalAtomicError::InvalidWitness(
            "the row cover must be a power of two".to_owned(),
        ));
    }
    let half = Fr::from(2_u64)
        .inverse()
        .ok_or(Section6FinalAtomicError::ProofShape)?;
    Ok(vec![half; row_count.ilog2() as usize])
}

fn terminal_scalar_expression(row_count: usize) -> RelationExpression {
    RelationExpression::table(0)
        .mul(RelationExpression::constant(Fr::from(row_count as u64)))
        .sub(RelationExpression::table(1))
}

/// Reduce the final TBV evaluation while deferring every Dory opening.
#[allow(clippy::too_many_arguments)]
pub fn prove_section6_final_atomic_deferred<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    masks: &'a CommittedSection6ShapeMasks,
    point_aux: &'a CommittedSection6PointAux,
    final_aux: &'a CommittedSection6FinalAux,
    public: &'a Section6PointPublicTables,
    plan: &'a Section6FinalProverPlan<'a>,
    packed: &HiddenPackedEoProverState,
    rng: &mut impl RngCore,
) -> Result<(Section6FinalAtomicProof, Section6FinalProverOpenings<'a>), Section6FinalAtomicError>
{
    let layout = masks.layout();
    let rows = layout.row_cover()?;
    let local_width = layout.public().local_width();
    if packed.challenges.len() != layout.global_log {
        return Err(Section6FinalAtomicError::InvalidWitness(format!(
            "Packed EO fixed {} challenges, the layout needs {}",
            packed.challenges.len(),
            layout.global_log
        )));
    }
    if final_aux.statement.padded_rows != rows
        || final_aux.statement.local_width != local_width
        || final_aux.statement.global_log != layout.global_log
        || point_aux.statement().padded_rows != rows
        || point_aux.statement().local_width != local_width
        || public.real_row_mask.len() != rows
    {
        return Err(Section6FinalAtomicError::InvalidWitness(
            "committed statements do not share the layout row cover".to_owned(),
        ));
    }

    let batch_items = plan.prover_items();
    let identity_label = scoped_label(label, b"final-identity", &packed.transcript_state);
    let (identity, identity_opening) =
        prove_hidden_relations_batch_deferred(&identity_label, parameters, &batch_items, rng)?;

    // Bind the mean of v to the Packed EO reduced claim b_reduced.
    let v_values = final_aux
        .v_values()
        .ok_or(Section6FinalAtomicError::ProofShape)?;
    let row_count_field = Fr::from(rows as u64);
    let inverse_row_count = row_count_field
        .inverse()
        .ok_or(Section6FinalAtomicError::ProofShape)?;
    let average_value = v_values
        .iter()
        .copied()
        .fold(Fr::zero(), |sum, value| sum + value)
        * inverse_row_count;
    let average = HiddenScalarWitness::commit(parameters, average_value, rng);
    let scalar_label = scoped_label(label, b"terminal-scalar", &packed.transcript_state);
    let scalar_binding = prove_hidden_scalar_relation(
        &scalar_label,
        parameters,
        &[average.clone(), packed.b_reduced.clone()],
        &terminal_scalar_expression(rows),
        rng,
    )?;

    Ok((
        Section6FinalAtomicProof {
            aux: final_aux.statement.clone(),
            identity,
            average_commitment: average.commitment,
            scalar_binding,
        },
        Section6FinalProverOpenings {
            identity: identity_opening,
            average,
            all_half_point: all_half_point(rows)?,
        },
    ))
}

fn resolve_prover_table<'a>(
    masks: &'a CommittedSection6ShapeMasks,
    point_aux: &'a CommittedSection6PointAux,
    final_aux: &'a CommittedSection6FinalAux,
    reference: &FinalTableRef,
) -> Result<HiddenRelationTable<'a>, Section6FinalAtomicError> {
    let point_columns = point_aux.layout_columns();
    let final_columns = final_aux.layout_columns();
    let table = match reference {
        FinalTableRef::Fin(column) => {
            let index = match column {
                FinalColumn::DZPrime(coordinate) => final_columns.d_z_prime(*coordinate),
                FinalColumn::EtaPrimePrefix(coordinate) => {
                    final_columns.eta_prime_prefix(*coordinate)
                }
                FinalColumn::ZPrime(coordinate) => final_columns.z_prime(*coordinate),
                FinalColumn::Q(coordinate) => final_columns.q(*coordinate),
            };
            final_aux.table(index)
        }
        FinalTableRef::PointSigmaLast => {
            point_aux.table(point_columns.sigma(SECTION6_FIXED_LOCAL_WIDTH - 1))
        }
        FinalTableRef::PointEta => point_aux.table(point_columns.eta()),
        FinalTableRef::PointZ(coordinate) => point_aux.table(point_columns.z(*coordinate)),
        FinalTableRef::MaskEtaPrime(coordinate) => {
            return Ok(HiddenRelationTable {
                values: masks
                    .d_eta_prime_column_values(*coordinate)
                    .ok_or(Section6FinalAtomicError::ProofShape)?,
                committed: masks
                    .d_eta_prime_columns()
                    .get(*coordinate)
                    .ok_or(Section6FinalAtomicError::ProofShape)?,
            });
        }
    };
    table.ok_or(Section6FinalAtomicError::ProofShape)
}

fn resolve_verifier_table<'a>(
    mask_statement: &'a Section6ShapeMaskStatement,
    point_statement: &'a Section6PointAuxStatement,
    final_statement: &'a Section6FinalAuxStatement,
    reference: &FinalTableRef,
) -> HiddenRelationTableStatement<'a> {
    let point_columns = Section6PointAuxColumns::new(point_statement.local_width);
    let final_columns =
        Section6FinalAuxColumns::new(final_statement.local_width, final_statement.global_log);
    match reference {
        FinalTableRef::Fin(column) => {
            let index = match column {
                FinalColumn::DZPrime(coordinate) => final_columns.d_z_prime(*coordinate),
                FinalColumn::EtaPrimePrefix(coordinate) => {
                    final_columns.eta_prime_prefix(*coordinate)
                }
                FinalColumn::ZPrime(coordinate) => final_columns.z_prime(*coordinate),
                FinalColumn::Q(coordinate) => final_columns.q(*coordinate),
            };
            HiddenRelationTableStatement::matrix_column(
                &final_statement.commitment,
                final_statement.padded_rows,
                final_statement.padded_cols,
                index,
            )
        }
        FinalTableRef::PointSigmaLast => HiddenRelationTableStatement::matrix_column(
            &point_statement.commitment,
            point_statement.padded_rows,
            point_statement.padded_cols,
            point_columns.sigma(SECTION6_FIXED_LOCAL_WIDTH - 1),
        ),
        FinalTableRef::PointEta => HiddenRelationTableStatement::matrix_column(
            &point_statement.commitment,
            point_statement.padded_rows,
            point_statement.padded_cols,
            point_columns.eta(),
        ),
        FinalTableRef::PointZ(coordinate) => HiddenRelationTableStatement::matrix_column(
            &point_statement.commitment,
            point_statement.padded_rows,
            point_statement.padded_cols,
            point_columns.z(*coordinate),
        ),
        FinalTableRef::MaskEtaPrime(coordinate) => HiddenRelationTableStatement::matrix_column(
            &mask_statement.d_eta_prime_commitment,
            mask_statement.padded_rows,
            Section6ShapeMaskStatement::padded_cols(mask_statement.offset_mask_width),
            *coordinate,
        ),
    }
}

/// Verify the deferred final reduction from public statements only.
#[allow(clippy::too_many_arguments)]
pub fn verify_section6_final_atomic_deferred<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    layout: &Section6PublicLayout,
    mask_statement: &'a Section6ShapeMaskStatement,
    point_statement: &'a Section6PointAuxStatement,
    public: &'a Section6PointPublicTables,
    plan: &'a Section6FinalVerifierPlan<'a>,
    packed: &HiddenPackedEoVerification,
    proof: &'a Section6FinalAtomicProof,
) -> Result<Section6FinalVerifierOpenings<'a>, Section6FinalAtomicError> {
    layout.validate()?;
    mask_statement.validate(layout)?;
    point_statement.validate(layout)?;
    proof.aux.validate(layout)?;
    let rows = layout.row_cover()?;
    if packed.challenges.len() != layout.global_log || public.real_row_mask.len() != rows {
        return Err(Section6FinalAtomicError::ProofShape);
    }

    let batch_items = plan.verifier_items();
    let identity_label = scoped_label(label, b"final-identity", &packed.transcript_state);
    let identity = verify_hidden_relations_batch_deferred(
        &identity_label,
        parameters,
        &batch_items,
        &proof.identity,
    )?;

    let scalar_label = scoped_label(label, b"terminal-scalar", &packed.transcript_state);
    verify_hidden_scalar_relation(
        &scalar_label,
        parameters,
        &[proof.average_commitment, packed.b_reduced_commitment],
        &terminal_scalar_expression(rows),
        &proof.scalar_binding,
    )?;

    let final_columns = Section6FinalAuxColumns::new(layout.local_width(), layout.global_log);
    Ok(Section6FinalVerifierOpenings {
        identity,
        average_commitment: proof.average_commitment,
        all_half_point: all_half_point(rows)?,
        v_statement: HiddenRelationTableStatement::matrix_column(
            &proof.aux.commitment,
            rows,
            proof.aux.padded_cols,
            final_columns.v(),
        ),
    })
}
