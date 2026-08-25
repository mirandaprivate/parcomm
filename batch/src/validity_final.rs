//! TBV evaluation argument: the per-opening validity of `bar b`.
//!
//! After Packed EO fixes the folding point `y`, the prover commits one
//! auxiliary matrix `[eta' | Z' | P_0..P_{2m}]` (the paper's `C_zeta'`
//! extended with the product-prefix chain). Plain identities bind the
//! committed `eta'` and `Z'` to their public piecewise-regular
//! descriptions - `eta'_j` is the offset-bit product of `y` and
//! `Z'_{j,c} = y_c [c < gamma'_j]` - while the chain identities prove
//! `P_0 = eta o eta'` and `P_{c+1} = P_c o (1 + Z_c o Z'_c)` against the
//! committed TBV `[eta | Z]`. One hidden scalar relation binds the mean of
//! the last chain column `v` to the Packed EO reduced claim, and the `v`
//! column itself is opened at the all-half point inside the shared terminal
//! Dory batch.

use std::fmt::{Display, Formatter};

use ark_bn254::Fr;
use ark_ff::{Field, One, Zero};
use ark_std::rand::RngCore;
use jolt_crypto::{Bn254G1, JoltGroup};
use jolt_dory::DoryCommitment;
use jolt_field::Fr as JoltFr;
use serde::{Deserialize, Serialize};

use crate::{
    hidden_packed_eo::{HiddenPackedEoProverState, HiddenPackedEoVerification},
    hidden_relation::{
        HiddenRelationError, HiddenRelationProverOpening, HiddenRelationTable,
        HiddenRelationTableStatement, HiddenRelationVerifierOpening, HiddenScalarWitness,
        OwnedPublicColumn, PublicRelationTable, RelationExpression,
    },
    hidden_relation_batch::{
        prove_hidden_relations_batch_deferred, verify_hidden_relations_batch_deferred,
        HiddenRelationBatchReductionProof,
    },
    jolt_dory::{CommittedDoryFieldVector, DoryProverParameters, DoryVerifierParameters},
    section6_data::{field_from_usize, Section6DataError, Section6FinalWitness, Section6Layout},
    section6_final_opening::{
        commit_section6_field_matrix_columns_on_grid_zk,
        resolve_section6_hidden_relation_prover_claim,
        resolve_section6_hidden_relation_verifier_claim, ResolvedSection6FieldProverClaim,
        ResolvedSection6FieldVerifierClaim, Section6DoryGrid, Section6FinalOpeningError,
    },
    structured_column::{ColumnPattern, ColumnSegment, StructuredColumn},
    validity_point::{CommittedTbv, IdentityPlan, PointCompilationError, TbvStatement},
};

const FINAL_DOMAIN: &[u8] = b"parcomm-tbv-evaluation-v1";

#[derive(Debug)]
pub enum FinalEvaluationError {
    Data(Section6DataError),
    Relation(HiddenRelationError),
    Opening(Section6FinalOpeningError),
    Point(PointCompilationError),
    InvalidWitness(String),
    ProofShape,
}

impl Display for FinalEvaluationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Data(error) => Display::fmt(error, formatter),
            Self::Relation(error) => Display::fmt(error, formatter),
            Self::Opening(error) => Display::fmt(error, formatter),
            Self::Point(error) => Display::fmt(error, formatter),
            Self::InvalidWitness(message) => {
                write!(formatter, "invalid final evaluation witness: {message}")
            }
            Self::ProofShape => formatter.write_str("invalid final evaluation proof shape"),
        }
    }
}

impl std::error::Error for FinalEvaluationError {}

impl From<Section6DataError> for FinalEvaluationError {
    fn from(error: Section6DataError) -> Self {
        Self::Data(error)
    }
}

impl From<HiddenRelationError> for FinalEvaluationError {
    fn from(error: HiddenRelationError) -> Self {
        Self::Relation(error)
    }
}

impl From<Section6FinalOpeningError> for FinalEvaluationError {
    fn from(error: Section6FinalOpeningError) -> Self {
        Self::Opening(error)
    }
}

impl From<PointCompilationError> for FinalEvaluationError {
    fn from(error: PointCompilationError) -> Self {
        Self::Point(error)
    }
}

/// Column offsets inside the committed final auxiliary matrix
/// `[eta' | Z' | P_0..P_{2m}]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FinalAuxColumns {
    /// Width `2m` of the `Z'` block and the chain step count.
    pub pair_width: usize,
}

impl FinalAuxColumns {
    pub fn new(pair_width: usize) -> Self {
        Self { pair_width }
    }

    pub fn eta_prime(self) -> usize {
        0
    }

    pub fn z_prime(self, coordinate: usize) -> usize {
        1 + coordinate
    }

    pub fn p(self, coordinate: usize) -> usize {
        1 + self.pair_width + coordinate
    }

    /// The final per-bucket vector `v` is the last chain column.
    pub fn v(self) -> usize {
        self.p(self.pair_width)
    }

    pub fn count(self) -> usize {
        2 + 2 * self.pair_width
    }

    pub fn padded_cols(self) -> usize {
        self.count().next_power_of_two()
    }
}

/// Verifier-visible statement for the committed final auxiliary matrix.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FinalAuxStatement {
    pub commitment: DoryCommitment,
    pub padded_rows: usize,
    pub pair_width: usize,
}

impl FinalAuxStatement {
    pub fn columns(&self) -> FinalAuxColumns {
        FinalAuxColumns::new(self.pair_width)
    }

    pub fn validate(&self, layout: &Section6Layout) -> Result<(), Section6DataError> {
        if self.pair_width != layout.pair_width() || self.padded_rows != layout.row_cover()? {
            return Err(Section6DataError::InvalidPoint(
                "final auxiliary statement does not match the public layout".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn column(&self, column: usize) -> HiddenRelationTableStatement<'_> {
        HiddenRelationTableStatement::matrix_column(
            &self.commitment,
            self.padded_rows,
            self.columns().count(),
            self.columns().padded_cols(),
            column,
        )
    }
}

/// Prover-only committed final auxiliary matrix.
pub struct CommittedFinalAux {
    statement: FinalAuxStatement,
    columns: Vec<CommittedDoryFieldVector>,
}

impl CommittedFinalAux {
    pub fn statement(&self) -> &FinalAuxStatement {
        &self.statement
    }

    pub fn table(&self, column: usize) -> Option<HiddenRelationTable<'_>> {
        let committed = self.columns.get(column)?;
        Some(HiddenRelationTable {
            values: committed.matrix_column_values()?,
            committed,
        })
    }

    pub fn v_table(&self) -> Option<HiddenRelationTable<'_>> {
        self.table(self.statement.columns().v())
    }
}

/// Commit the final auxiliary matrix for one compiled final witness.
pub fn commit_final_aux_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    layout: &Section6Layout,
    witness: &Section6FinalWitness<Fr>,
) -> Result<CommittedFinalAux, FinalEvaluationError> {
    let rows = layout.row_cover()?;
    let pair = layout.pair_width();
    if witness.eta_prime.padded_len != rows
        || witness.z_prime.padded_rows != rows
        || witness.z_prime.logical_cols != pair
        || witness.p.padded_rows != rows
        || witness.p.logical_cols != pair + 1
    {
        return Err(FinalEvaluationError::InvalidWitness(
            "final witness covers do not match the layout".to_owned(),
        ));
    }
    let columns_layout = FinalAuxColumns::new(pair);
    let mut column_slices = Vec::with_capacity(columns_layout.count());
    column_slices.push(witness.eta_prime.values.as_slice());
    column_slices.extend(witness.z_prime.padded_columns());
    column_slices.extend(witness.p.padded_columns());
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
        .ok_or(FinalEvaluationError::ProofShape)?;
    Ok(CommittedFinalAux {
        statement: FinalAuxStatement {
            commitment,
            padded_rows: rows,
            pair_width: pair,
        },
        columns,
    })
}

/// Public restriction tables derived from the layout and the point `y`.
///
/// `eta'` follows the strided bucket offsets inside each equal-size run and
/// the `D_Z'` columns are run-length constant `0/1` masks, so the verifier
/// evaluates each MLE in `O(polylog)` time from the description alone.
pub struct FinalPublicTables {
    /// `eta'_j = prod_{bit set in offset_j} y_bit`, zero on padding rows.
    pub eta_prime: OwnedPublicColumn,
    /// Prefix masks `D_Z'_{j,c} = [c < gamma'_j]`.
    pub d_z_prime: Vec<OwnedPublicColumn>,
}

impl FinalPublicTables {
    pub fn derive(
        layout: &Section6Layout,
        y: &[Fr],
        materialize: bool,
    ) -> Result<Self, FinalEvaluationError> {
        layout.validate()?;
        if y.len() != layout.offset_width() {
            return Err(FinalEvaluationError::InvalidWitness(format!(
                "final point length mismatch: expected {}, got {}",
                layout.offset_width(),
                y.len()
            )));
        }
        let rows = layout.row_cover()?;
        let pair = layout.pair_width();
        let gamma_prime = layout.gamma_prime();
        let offsets = layout.bucket_offsets()?;
        let structured_error =
            |error: crate::structured_column::StructuredColumnError| -> FinalEvaluationError {
                FinalEvaluationError::InvalidWitness(error.to_string())
            };
        let owned = |column: StructuredColumn| OwnedPublicColumn::new(column, materialize);

        // eta' runs: consecutive buckets of one size share the stride
        // 2^{gamma'}, so each run is one strided offset-bit-product segment.
        let mut eta_segments: Vec<ColumnSegment> = Vec::new();
        let mut run: Option<(usize, u64, u64, usize)> = None; // (gamma, first, next, len)
        for (&offset, &log) in offsets.iter().zip(&gamma_prime) {
            let offset = offset as u64;
            match run.as_mut() {
                Some((gamma, _, next, len)) if *gamma == log && *next == offset => {
                    *next += 1_u64 << log;
                    *len += 1;
                }
                _ => {
                    if let Some((gamma, first, _, len)) = run.take() {
                        eta_segments.push(ColumnSegment {
                            len,
                            pattern: ColumnPattern::OffsetBitProduct {
                                offset0: first,
                                stride_log: gamma as u32,
                            },
                        });
                    }
                    run = Some((log, offset, offset + (1_u64 << log), 1));
                }
            }
        }
        if let Some((gamma, first, _, len)) = run.take() {
            eta_segments.push(ColumnSegment {
                len,
                pattern: ColumnPattern::OffsetBitProduct {
                    offset0: first,
                    stride_log: gamma as u32,
                },
            });
        }
        let eta_prime = owned(
            StructuredColumn::with_zero_tail(rows, eta_segments, y.to_vec())
                .map_err(structured_error)?,
        );

        // D_Z' columns: one on the buckets with gamma' > c, zero elsewhere;
        // consecutive equal-size buckets merge into run-length segments.
        let mut d_z_prime = Vec::with_capacity(pair);
        for coordinate in 0..pair {
            let mut segments: Vec<ColumnSegment> = Vec::new();
            let append = |segments: &mut Vec<ColumnSegment>, len: usize, value: Fr| {
                if len == 0 {
                    return;
                }
                if let Some(last) = segments.last_mut() {
                    if matches!(&last.pattern, ColumnPattern::Constant(existing) if *existing == value)
                    {
                        last.len += len;
                        return;
                    }
                }
                segments.push(ColumnSegment {
                    len,
                    pattern: ColumnPattern::Constant(value),
                });
            };
            let mut run_start = 0;
            while run_start < gamma_prime.len() {
                let live = gamma_prime[run_start] > coordinate;
                let mut run_end = run_start + 1;
                while run_end < gamma_prime.len() && (gamma_prime[run_end] > coordinate) == live {
                    run_end += 1;
                }
                let value = if live { Fr::one() } else { Fr::zero() };
                append(&mut segments, run_end - run_start, value);
                run_start = run_end;
            }
            d_z_prime.push(owned(
                StructuredColumn::with_zero_tail(rows, segments, Vec::new())
                    .map_err(structured_error)?,
            ));
        }

        Ok(Self {
            eta_prime,
            d_z_prime,
        })
    }
}

/// Serializable final reduction with no PCS opening embedded in it.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FinalEvaluationProof {
    pub aux: FinalAuxStatement,
    /// Joint Point-compilation and final-evaluation identity reduction.
    pub joint_identity: HiddenRelationBatchReductionProof,
}

/// Prover-side deferred openings produced by the final reduction.
pub struct FinalProverOpenings<'a> {
    pub joint_identity: HiddenRelationProverOpening<'a>,
    pub average: HiddenScalarWitness,
    pub all_half_point: Vec<Fr>,
}

impl FinalProverOpenings<'_> {
    /// Resolve the `v` column at the all-half point for the terminal batch.
    pub fn resolve_v_terminal_claim<'b>(
        &self,
        final_aux: &'b CommittedFinalAux,
        grid: &Section6DoryGrid,
    ) -> Result<ResolvedSection6FieldProverClaim<'b>, FinalEvaluationError> {
        let table = final_aux
            .v_table()
            .ok_or(FinalEvaluationError::ProofShape)?;
        let point = self
            .all_half_point
            .iter()
            .copied()
            .map(JoltFr::from)
            .collect::<Vec<_>>();
        Ok(resolve_section6_hidden_relation_prover_claim(
            table, &point, grid,
        )?)
    }
}

/// Verifier-side deferred openings mirroring the prover structure.
pub struct FinalVerifierOpenings<'a> {
    pub joint_identity: HiddenRelationVerifierOpening<'a>,
    pub average_commitment: Bn254G1,
    pub all_half_point: Vec<Fr>,
    v_statement: HiddenRelationTableStatement<'a>,
}

impl FinalVerifierOpenings<'_> {
    /// Resolve the public `v` commitment view for the terminal Dory batch.
    pub fn resolve_v_terminal_claim(
        &self,
        grid: &Section6DoryGrid,
    ) -> Result<ResolvedSection6FieldVerifierClaim<'_>, FinalEvaluationError> {
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

/// Prover-side identity plan for the final reduction.
pub type FinalProverPlan<'a> = IdentityPlan<'a, HiddenRelationTable<'a>>;

/// Verifier-side identity plan for the final reduction.
pub type FinalVerifierPlan<'a> = IdentityPlan<'a, HiddenRelationTableStatement<'a>>;

/// One identity of the final batch, in shared table order.
struct FinalIdentityItem {
    hidden: Vec<FinalTableRef>,
    publics: Vec<FinalPublicRef>,
    expression: RelationExpression,
}

/// Symbolic operand references resolved by the prover and the verifier.
#[derive(Clone, Copy)]
enum FinalTableRef {
    FinEtaPrime,
    FinZPrime(usize),
    FinP(usize),
    TbvEta,
    TbvZ(usize),
}

#[derive(Clone, Copy)]
enum FinalPublicRef {
    EtaPrimePub,
    DZPrime(usize),
}

/// The complete final identity batch in a fixed order shared by both sides.
///
/// Padding rows carry `eta' = 0` and hence `P = 0`, so every identity holds
/// on the whole cover without a mask; public table indices continue after
/// the hidden tables.
fn final_identity_items(pair_width: usize, y: &[Fr]) -> Vec<FinalIdentityItem> {
    let mut items = Vec::new();

    // The committed eta' equals its public offset-bit-product description.
    items.push(FinalIdentityItem {
        hidden: vec![FinalTableRef::FinEtaPrime],
        publics: vec![FinalPublicRef::EtaPrimePub],
        expression: RelationExpression::table(0).sub(RelationExpression::table(1)),
    });

    // The committed Z' equals y_c times the public prefix mask. Coordinates
    // beyond the evaluation point width have all-zero masks (no bucket can
    // exceed the global size), so their scale is immaterial and set to zero.
    for coordinate in 0..pair_width {
        let scale = y.get(coordinate).copied().unwrap_or_else(Fr::zero);
        items.push(FinalIdentityItem {
            hidden: vec![FinalTableRef::FinZPrime(coordinate)],
            publics: vec![FinalPublicRef::DZPrime(coordinate)],
            expression: RelationExpression::table(0)
                .sub(RelationExpression::constant(scale).mul(RelationExpression::table(1))),
        });
    }

    // P_0 = eta o eta' with the committed TBV scales.
    items.push(FinalIdentityItem {
        hidden: vec![
            FinalTableRef::FinP(0),
            FinalTableRef::TbvEta,
            FinalTableRef::FinEtaPrime,
        ],
        publics: Vec::new(),
        expression: RelationExpression::table(0)
            .sub(RelationExpression::table(1).mul(RelationExpression::table(2))),
    });

    // P_{c+1} = P_c o (1 + Z_c o Z'_c); the last prefix column is v.
    for coordinate in 0..pair_width {
        items.push(FinalIdentityItem {
            hidden: vec![
                FinalTableRef::FinP(coordinate + 1),
                FinalTableRef::FinP(coordinate),
                FinalTableRef::TbvZ(coordinate),
                FinalTableRef::FinZPrime(coordinate),
            ],
            publics: Vec::new(),
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

fn resolve_prover_table<'a>(
    tbv: &'a CommittedTbv,
    final_aux: &'a CommittedFinalAux,
    reference: &FinalTableRef,
) -> Result<HiddenRelationTable<'a>, FinalEvaluationError> {
    let final_columns = final_aux.statement.columns();
    let table = match reference {
        FinalTableRef::FinEtaPrime => final_aux.table(final_columns.eta_prime()),
        FinalTableRef::FinZPrime(coordinate) => final_aux.table(final_columns.z_prime(*coordinate)),
        FinalTableRef::FinP(coordinate) => final_aux.table(final_columns.p(*coordinate)),
        FinalTableRef::TbvEta => tbv.eta_table(),
        FinalTableRef::TbvZ(coordinate) => tbv.z_table(*coordinate),
    };
    table.ok_or(FinalEvaluationError::ProofShape)
}

fn resolve_verifier_table<'a>(
    tbv_statement: &'a TbvStatement,
    final_statement: &'a FinalAuxStatement,
    reference: &FinalTableRef,
) -> HiddenRelationTableStatement<'a> {
    let final_columns = final_statement.columns();
    match reference {
        FinalTableRef::FinEtaPrime => final_statement.column(final_columns.eta_prime()),
        FinalTableRef::FinZPrime(coordinate) => {
            final_statement.column(final_columns.z_prime(*coordinate))
        }
        FinalTableRef::FinP(coordinate) => final_statement.column(final_columns.p(*coordinate)),
        FinalTableRef::TbvEta => tbv_statement.eta_column(),
        FinalTableRef::TbvZ(coordinate) => tbv_statement.z_column(*coordinate),
    }
}

fn resolve_public_table<'a>(
    public: &'a FinalPublicTables,
    reference: &FinalPublicRef,
) -> PublicRelationTable<'a> {
    match reference {
        FinalPublicRef::EtaPrimePub => public.eta_prime.table(),
        FinalPublicRef::DZPrime(coordinate) => public.d_z_prime[*coordinate].table(),
    }
}

/// Build the caller-owned prover identity plan for the final reduction.
pub fn build_final_prover_plan<'a>(
    layout: &Section6Layout,
    tbv: &'a CommittedTbv,
    final_aux: &'a CommittedFinalAux,
    public: &'a FinalPublicTables,
    y: &[Fr],
) -> Result<FinalProverPlan<'a>, FinalEvaluationError> {
    let items = final_identity_items(layout.pair_width(), y);
    let mut plan = IdentityPlan::new();
    for item in items {
        let tables = item
            .hidden
            .iter()
            .map(|reference| resolve_prover_table(tbv, final_aux, reference))
            .collect::<Result<Vec<_>, _>>()?;
        let publics = item
            .publics
            .iter()
            .map(|reference| resolve_public_table(public, reference))
            .collect::<Vec<_>>();
        plan.push(tables, publics, item.expression);
    }
    Ok(plan)
}

/// Build the caller-owned verifier identity plan for the final reduction.
pub fn build_final_verifier_plan<'a>(
    layout: &Section6Layout,
    tbv_statement: &'a TbvStatement,
    final_statement: &'a FinalAuxStatement,
    public: &'a FinalPublicTables,
    y: &[Fr],
) -> Result<FinalVerifierPlan<'a>, FinalEvaluationError> {
    tbv_statement.validate(layout)?;
    final_statement.validate(layout)?;
    let items = final_identity_items(layout.pair_width(), y);
    let mut plan = IdentityPlan::new();
    for item in items {
        let tables = item
            .hidden
            .iter()
            .map(|reference| resolve_verifier_table(tbv_statement, final_statement, reference))
            .collect::<Vec<_>>();
        let publics = item
            .publics
            .iter()
            .map(|reference| resolve_public_table(public, reference))
            .collect::<Vec<_>>();
        plan.push(tables, publics, item.expression);
    }
    Ok(plan)
}

fn scoped_label(label: &[u8], role: &[u8], transcript_state: &[u8]) -> Vec<u8> {
    let mut output = Vec::with_capacity(
        FINAL_DOMAIN.len() + label.len() + role.len() + transcript_state.len() + 24,
    );
    output.extend_from_slice(FINAL_DOMAIN);
    output.extend_from_slice(&(label.len() as u64).to_le_bytes());
    output.extend_from_slice(label);
    output.extend_from_slice(&(role.len() as u64).to_le_bytes());
    output.extend_from_slice(role);
    output.extend_from_slice(&(transcript_state.len() as u64).to_le_bytes());
    output.extend_from_slice(transcript_state);
    output
}

fn all_half_point(row_count: usize) -> Result<Vec<Fr>, FinalEvaluationError> {
    if !row_count.is_power_of_two() {
        return Err(FinalEvaluationError::InvalidWitness(
            "the row cover must be a power of two".to_owned(),
        ));
    }
    let half = Fr::from(2_u64)
        .inverse()
        .ok_or(FinalEvaluationError::ProofShape)?;
    Ok(vec![half; row_count.ilog2() as usize])
}

/// Reduce the final TBV evaluation while deferring every Dory opening.
#[allow(clippy::too_many_arguments)]
pub fn prove_final_evaluation_deferred<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    layout: &Section6Layout,
    tbv: &'a CommittedTbv,
    final_aux: &'a CommittedFinalAux,
    identity_prefix: Option<&'a FinalProverPlan<'a>>,
    plan: &'a FinalProverPlan<'a>,
    packed: &HiddenPackedEoProverState,
    rng: &mut impl RngCore,
) -> Result<(FinalEvaluationProof, FinalProverOpenings<'a>), FinalEvaluationError> {
    let rows = layout.row_cover()?;
    if packed.challenges.len() != layout.offset_width() {
        return Err(FinalEvaluationError::InvalidWitness(format!(
            "Packed EO fixed {} challenges, the layout needs {}",
            packed.challenges.len(),
            layout.offset_width()
        )));
    }
    if final_aux.statement.padded_rows != rows
        || tbv.statement().padded_rows != rows
        || tbv.statement().pair_width != layout.pair_width()
    {
        return Err(FinalEvaluationError::InvalidWitness(
            "committed statements do not share the layout row cover".to_owned(),
        ));
    }

    let mut batch_items = identity_prefix
        .map(FinalProverPlan::prover_items)
        .unwrap_or_default();
    batch_items.extend(plan.prover_items());
    let identity_role = if identity_prefix.is_some() {
        b"joint-point-final-identity".as_slice()
    } else {
        b"final-identity".as_slice()
    };
    let identity_label = scoped_label(label, identity_role, &packed.transcript_state);
    let (identity, identity_opening) =
        prove_hidden_relations_batch_deferred(&identity_label, parameters, &batch_items, rng)?;

    // Bind the mean of v to the Packed EO reduced claim b_reduced.
    let v_table = final_aux
        .v_table()
        .ok_or(FinalEvaluationError::ProofShape)?;
    let inverse_row_count = field_from_usize::<Fr>(rows)
        .inverse()
        .ok_or(FinalEvaluationError::ProofShape)?;
    let average_value = v_table
        .values
        .iter()
        .copied()
        .fold(Fr::zero(), |sum, value| sum + value)
        * inverse_row_count;
    if average_value * field_from_usize::<Fr>(rows) != packed.b_reduced.value {
        return Err(FinalEvaluationError::InvalidWitness(
            "the final TBV average does not match the Packed EO reduction".to_owned(),
        ));
    }
    let inverse_row_count_jolt = JoltFr::from(inverse_row_count);
    let average = HiddenScalarWitness {
        value: average_value,
        commitment: packed
            .b_reduced
            .commitment
            .scalar_mul(&inverse_row_count_jolt),
        blind: packed.b_reduced.blind * inverse_row_count_jolt,
    };

    Ok((
        FinalEvaluationProof {
            aux: final_aux.statement.clone(),
            joint_identity: identity,
        },
        FinalProverOpenings {
            joint_identity: identity_opening,
            average,
            all_half_point: all_half_point(rows)?,
        },
    ))
}

/// Verify the deferred final reduction from public statements only.
#[allow(clippy::too_many_arguments)]
pub fn verify_final_evaluation_deferred<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    grid: &Section6DoryGrid,
    layout: &Section6Layout,
    tbv_statement: &'a TbvStatement,
    identity_prefix: Option<&'a FinalVerifierPlan<'a>>,
    plan: &'a FinalVerifierPlan<'a>,
    packed: &HiddenPackedEoVerification,
    proof: &'a FinalEvaluationProof,
) -> Result<FinalVerifierOpenings<'a>, FinalEvaluationError> {
    let _ = grid;
    layout.validate()?;
    tbv_statement.validate(layout)?;
    proof.aux.validate(layout)?;
    let rows = layout.row_cover()?;
    if packed.challenges.len() != layout.offset_width() {
        return Err(FinalEvaluationError::ProofShape);
    }

    let mut batch_items = identity_prefix
        .map(FinalVerifierPlan::verifier_items)
        .unwrap_or_default();
    batch_items.extend(plan.verifier_items());
    let identity_role = if identity_prefix.is_some() {
        b"joint-point-final-identity".as_slice()
    } else {
        b"final-identity".as_slice()
    };
    let identity_label = scoped_label(label, identity_role, &packed.transcript_state);
    let identity = verify_hidden_relations_batch_deferred(
        &identity_label,
        parameters,
        &batch_items,
        &proof.joint_identity,
    )?;

    let inverse_row_count = field_from_usize::<Fr>(rows)
        .inverse()
        .ok_or(FinalEvaluationError::ProofShape)?;
    let average_commitment = packed
        .b_reduced_commitment
        .scalar_mul(&JoltFr::from(inverse_row_count));

    Ok(FinalVerifierOpenings {
        joint_identity: identity,
        average_commitment,
        all_half_point: all_half_point(rows)?,
        v_statement: proof.aux.column(proof.aux.columns().v()),
    })
}

#[cfg(test)]
mod tests {
    use ark_std::rand::{rngs::StdRng, SeedableRng};

    use super::*;
    use crate::{
        hidden_packed_eo::HiddenPackedEoProverState,
        jolt_dory::setup_dory_parameters,
        matrix_collection::DenseBvaDescriptor,
        section6_claim_collection::{
            Section6ProverClaimCollection, Section6VerifierClaimCollection,
        },
        section6_data::{Section6MatrixClaim, Section6PointWitness},
        section6_final_opening::{prove_section6_final_opening, verify_section6_final_opening},
        validity_point::commit_tbv_on_grid_zk,
        validity_shape::commit_shape_masks_from_descriptor_on_grid_zk,
    };

    fn field_values(start: u64, len: usize) -> Vec<Fr> {
        (0..len)
            .map(|index| Fr::from(start + index as u64))
            .collect()
    }

    #[test]
    fn final_evaluation_round_trips_and_defers_all_openings() {
        let shapes = [(3, 5), (5, 2)];
        let descriptor = DenseBvaDescriptor::from_shapes(&shapes).unwrap();
        let grid = Section6DoryGrid::new(12).unwrap();
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(grid.num_vars());
        let mut rng = StdRng::seed_from_u64(0x4649_4e41_4c5f_5631);

        let committed_shapes = commit_shape_masks_from_descriptor_on_grid_zk(
            &prover_parameters,
            &grid,
            &descriptor,
            8,
        )
        .unwrap();
        let layout = committed_shapes.layout().clone();
        let metadata = committed_shapes.metadata().clone();
        let m = layout.point_log();
        let claims = vec![
            Section6MatrixClaim::from_unpadded(
                m,
                &field_values(11, 2),
                &field_values(21, 3),
                Fr::from(31_u64),
            )
            .unwrap(),
            Section6MatrixClaim::from_unpadded(
                m,
                &field_values(41, 3),
                &field_values(51, 1),
                Fr::from(61_u64),
            )
            .unwrap(),
        ];
        let batch_point = (0..layout.tau_log())
            .map(|bit| Fr::from((7 + 5 * bit) as u64))
            .collect::<Vec<_>>();
        let point_witness =
            Section6PointWitness::compile(&layout, &descriptor, &metadata, &claims, &batch_point)
                .unwrap();
        let tbv =
            commit_tbv_on_grid_zk(&prover_parameters, &grid, &layout, &point_witness).unwrap();

        let y = (0..layout.offset_width())
            .map(|bit| Fr::from((3 + 2 * bit) as u64))
            .collect::<Vec<_>>();
        let final_witness = Section6FinalWitness::compile(&layout, &point_witness, &y).unwrap();
        let final_aux =
            commit_final_aux_on_grid_zk(&prover_parameters, &grid, &layout, &final_witness)
                .unwrap();

        // A stand-in Packed EO state carrying the exact reduced TBV claim.
        let b_reduced =
            HiddenScalarWitness::commit(&prover_parameters, final_witness.evaluation, &mut rng);
        let packed = HiddenPackedEoProverState {
            a_reduced: b_reduced.clone(),
            b_reduced: b_reduced.clone(),
            challenges: y.clone(),
            challenges_inv: y.iter().map(|c| c.inverse().unwrap()).collect(),
            transcript_state: [7_u8; 32],
        };

        let publics = FinalPublicTables::derive(&layout, &y, true).unwrap();
        let plan = build_final_prover_plan(&layout, &tbv, &final_aux, &publics, &y).unwrap();
        let (proof, prover_openings) = prove_final_evaluation_deferred(
            b"final-test",
            &prover_parameters,
            &layout,
            &tbv,
            &final_aux,
            None,
            &plan,
            &packed,
            &mut rng,
        )
        .unwrap();

        let verification = HiddenPackedEoVerification {
            a_reduced_commitment: b_reduced.commitment,
            b_reduced_commitment: b_reduced.commitment,
            challenges: y.clone(),
            challenges_inv: y.iter().map(|c| c.inverse().unwrap()).collect(),
            transcript_state: [7_u8; 32],
        };
        let verifier_publics = FinalPublicTables::derive(&layout, &y, false).unwrap();
        let verifier_plan =
            build_final_verifier_plan(&layout, tbv.statement(), &proof.aux, &verifier_publics, &y)
                .unwrap();
        let verifier_openings = verify_final_evaluation_deferred(
            b"final-test",
            &verifier_parameters,
            &grid,
            &layout,
            tbv.statement(),
            None,
            &verifier_plan,
            &verification,
            &proof,
        )
        .unwrap();

        // Every deferred claim resolves in one shared terminal batch,
        // including the v column at the all-half point.
        let mut prover_claims = Section6ProverClaimCollection::new();
        prover_claims
            .push_relation(&prover_openings.joint_identity, &grid)
            .unwrap();
        let v_claim = prover_openings
            .resolve_v_terminal_claim(&final_aux, &grid)
            .unwrap();
        let prover_terminal_claims = {
            let mut claims = prover_claims.claims();
            claims.push(v_claim.as_claim(&prover_openings.average));
            claims
        };
        let terminal = prove_section6_final_opening(
            b"final-test-terminal",
            b"outer",
            &prover_parameters,
            &grid,
            &prover_terminal_claims,
            &mut rng,
        )
        .unwrap();

        let mut verifier_claims = Section6VerifierClaimCollection::new();
        verifier_claims
            .push_relation(&verifier_openings.joint_identity, &grid)
            .unwrap();
        let verifier_v_claim = verifier_openings.resolve_v_terminal_claim(&grid).unwrap();
        let verifier_terminal_claims = {
            let mut claims = verifier_claims.claims();
            claims.push(verifier_v_claim.as_claim());
            claims
        };
        verify_section6_final_opening(
            b"final-test-terminal",
            b"outer",
            &verifier_parameters,
            &grid,
            &verifier_terminal_claims,
            &terminal,
        )
        .unwrap();

        // Tampering with the reduced claim changes the derived average
        // commitment and must break the terminal Dory link to `v`.
        let wrong = HiddenScalarWitness::commit(
            &prover_parameters,
            final_witness.evaluation + Fr::one(),
            &mut rng,
        );
        let bad_verification = HiddenPackedEoVerification {
            b_reduced_commitment: wrong.commitment,
            ..verification
        };
        let bad_openings = verify_final_evaluation_deferred(
            b"final-test",
            &verifier_parameters,
            &grid,
            &layout,
            tbv.statement(),
            None,
            &verifier_plan,
            &bad_verification,
            &proof,
        )
        .unwrap();
        let mut bad_claims = Section6VerifierClaimCollection::new();
        bad_claims
            .push_relation(&bad_openings.joint_identity, &grid)
            .unwrap();
        let bad_v_claim = bad_openings.resolve_v_terminal_claim(&grid).unwrap();
        let bad_terminal_claims = {
            let mut claims = bad_claims.claims();
            claims.push(bad_v_claim.as_claim());
            claims
        };
        assert!(verify_section6_final_opening(
            b"final-test-terminal",
            b"outer",
            &verifier_parameters,
            &grid,
            &bad_terminal_claims,
            &terminal,
        )
        .is_err());
    }
}
