//! TBV compilation argument: the per-opening validity of `(eta, Z)`.
//!
//! The prover commits the claim-point matrix `X` (before the batching point
//! is sampled), the compiled TBV `[eta | Z]`, and one auxiliary matrix
//! `[X' | Pi'_eta | gamma_r]`. After those commitments are fixed, the
//! challenges `vartheta` (source keys) and `theta_z` (compaction) are
//! derived and three small key columns `[q_key | t_key | V]` are committed.
//! The argument then proves:
//!
//! * two key-definition identities and one tagged LogUp containing the source
//!   lookup `{q_key} subset {t_key}` with
//!   `q_key = ind + tau + <X'_j, vartheta>` and
//!   `t_key = rowid + <X_i, vartheta>`, binding every gathered row of `X'` to
//!   the source index encoded by the committed `D_eta`;
//! * the eta grand-product chain `Pi'_eta` with factors
//!   `1 - D_eta_h + D_eta_h X'_h` on the point block and public factors
//!   `1 + (x_b - 1) D_eta_h` on the batching block, closed by the committed
//!   TBV scale column `eta`;
//! * the compaction identity `<Z_j, theta_z> = <D_Z_r o X'_r, theta_z> +
//!   theta_z^{gamma_r - m} <D_Z_l o X'_l, theta_z>` with the committed
//!   power column `V = theta_z^{gamma_r}` bound by the second family of the
//!   same tagged LogUp.
//!
//! Every deferred claim resolves later in the shared terminal Dory batch.

use std::fmt::{Display, Formatter};

use ark_bn254::Fr;
use ark_ff::{Field, One, PrimeField, Zero};
use ark_serialize::CanonicalSerialize;
use ark_std::rand::RngCore;
use jolt_dory::DoryCommitment;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::{
    hidden_logup::{
        HiddenLogUpError, HiddenLogUpPartsReductionProof, HiddenLogUpProverOpeningState,
        HiddenLogUpVerifierOpeningState, MixedLogUpColumn, MixedLogUpColumnStatement,
    },
    hidden_logup_union::{
        prove_hidden_logup_union_deferred_on_grid_with_hidden_tables,
        verify_hidden_logup_union_deferred_on_grid_with_hidden_tables, HiddenLogUpUnionError,
        HiddenLogUpUnionProverFamily, HiddenLogUpUnionVerifierFamily,
    },
    hidden_relation::{
        HiddenRelationError, HiddenRelationTable, HiddenRelationTableStatement, OwnedPublicColumn,
        PublicRelationTable, RelationExpression,
    },
    hidden_relation_batch::{HiddenRelationBatchProverItem, HiddenRelationBatchVerifierItem},
    jolt_dory::{CommittedDoryFieldVector, DoryProverParameters, DoryVerifierParameters},
    section6_data::{
        field_from_usize, validate_claims, Section6DataError, Section6Layout, Section6MatrixClaim,
        Section6PointWitness,
    },
    section6_final_opening::{
        commit_section6_field_matrix_columns_on_grid_zk, Section6DoryGrid,
        Section6FinalOpeningError,
    },
    structured_column::{ColumnPattern, ColumnSegment, StructuredColumn},
    validity_shape::{CommittedShapeMasks, ShapeCommitStatement},
};

const POINT_DOMAIN: &[u8] = b"parcomm-tbv-compilation-v1";

#[derive(Debug)]
pub enum PointCompilationError {
    Data(Section6DataError),
    Relation(HiddenRelationError),
    LogUp(HiddenLogUpError),
    LogUpUnion(HiddenLogUpUnionError),
    Opening(Section6FinalOpeningError),
    Encoding(String),
    InvalidWitness(String),
    ProofShape,
}

impl Display for PointCompilationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Data(error) => Display::fmt(error, formatter),
            Self::Relation(error) => Display::fmt(error, formatter),
            Self::LogUp(error) => Display::fmt(error, formatter),
            Self::LogUpUnion(error) => Display::fmt(error, formatter),
            Self::Opening(error) => Display::fmt(error, formatter),
            Self::Encoding(message) => write!(formatter, "encoding failure: {message}"),
            Self::InvalidWitness(message) => {
                write!(formatter, "invalid point compilation witness: {message}")
            }
            Self::ProofShape => formatter.write_str("invalid point compilation proof shape"),
        }
    }
}

impl std::error::Error for PointCompilationError {}

impl From<Section6DataError> for PointCompilationError {
    fn from(error: Section6DataError) -> Self {
        Self::Data(error)
    }
}

impl From<HiddenRelationError> for PointCompilationError {
    fn from(error: HiddenRelationError) -> Self {
        Self::Relation(error)
    }
}

impl From<HiddenLogUpError> for PointCompilationError {
    fn from(error: HiddenLogUpError) -> Self {
        Self::LogUp(error)
    }
}

impl From<HiddenLogUpUnionError> for PointCompilationError {
    fn from(error: HiddenLogUpUnionError) -> Self {
        Self::LogUpUnion(error)
    }
}

impl From<Section6FinalOpeningError> for PointCompilationError {
    fn from(error: Section6FinalOpeningError) -> Self {
        Self::Opening(error)
    }
}

/// Verifier-visible statement for the committed point matrix `X`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PointMatrixStatement {
    pub commitment: DoryCommitment,
    pub matrix_count: usize,
    pub padded_rows: usize,
    pub padded_cols: usize,
}

impl PointMatrixStatement {
    pub fn validate(&self, layout: &Section6Layout) -> Result<(), Section6DataError> {
        if self.matrix_count != layout.matrix_count
            || self.padded_rows != layout.row_cover()?
            || self.padded_cols != layout.pair_width().next_power_of_two()
        {
            return Err(Section6DataError::InvalidPoint(
                "point matrix statement does not match the public layout".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn column(&self, column: usize) -> HiddenRelationTableStatement<'_> {
        HiddenRelationTableStatement::matrix_column(
            &self.commitment,
            self.padded_rows,
            self.padded_cols,
            column,
        )
    }
}

/// Prover-only committed point matrix retained across the opening batch.
pub struct CommittedPointMatrix {
    statement: PointMatrixStatement,
    columns: Vec<CommittedDoryFieldVector>,
}

impl CommittedPointMatrix {
    pub fn statement(&self) -> &PointMatrixStatement {
        &self.statement
    }

    pub fn table(&self, column: usize) -> Option<HiddenRelationTable<'_>> {
        let committed = self.columns.get(column)?;
        Some(HiddenRelationTable {
            values: committed.matrix_column_values()?,
            committed,
        })
    }
}

/// Verifier-visible statement for the compiled TBV `[eta | Z]`.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct TbvStatement {
    pub commitment: DoryCommitment,
    pub padded_rows: usize,
    /// Width `2m` of the `Z` block.
    pub pair_width: usize,
}

impl TbvStatement {
    pub fn logical_cols(&self) -> usize {
        1 + self.pair_width
    }

    pub fn padded_cols(&self) -> usize {
        self.logical_cols().next_power_of_two()
    }

    pub fn validate(&self, layout: &Section6Layout) -> Result<(), Section6DataError> {
        if self.padded_rows != layout.row_cover()? || self.pair_width != layout.pair_width() {
            return Err(Section6DataError::InvalidPoint(
                "TBV statement does not match the public layout".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn eta_column(&self) -> HiddenRelationTableStatement<'_> {
        HiddenRelationTableStatement::matrix_column(
            &self.commitment,
            self.padded_rows,
            self.padded_cols(),
            0,
        )
    }

    pub fn z_column(&self, coordinate: usize) -> HiddenRelationTableStatement<'_> {
        HiddenRelationTableStatement::matrix_column(
            &self.commitment,
            self.padded_rows,
            self.padded_cols(),
            1 + coordinate,
        )
    }
}

/// Prover-only committed TBV `[eta | Z]`.
pub struct CommittedTbv {
    statement: TbvStatement,
    columns: Vec<CommittedDoryFieldVector>,
}

impl CommittedTbv {
    pub fn statement(&self) -> &TbvStatement {
        &self.statement
    }

    fn table(&self, column: usize) -> Option<HiddenRelationTable<'_>> {
        let committed = self.columns.get(column)?;
        Some(HiddenRelationTable {
            values: committed.matrix_column_values()?,
            committed,
        })
    }

    pub fn eta_table(&self) -> Option<HiddenRelationTable<'_>> {
        self.table(0)
    }

    pub fn z_table(&self, coordinate: usize) -> Option<HiddenRelationTable<'_>> {
        self.table(1 + coordinate)
    }
}

/// Column offsets inside the committed auxiliary matrix
/// `[X' | Pi'_eta | gamma_r]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PointAuxColumns {
    pub pair_width: usize,
    pub eta_width: usize,
}

impl PointAuxColumns {
    pub fn new(pair_width: usize, eta_width: usize) -> Self {
        Self {
            pair_width,
            eta_width,
        }
    }

    pub fn x_selected(self, coordinate: usize) -> usize {
        coordinate
    }

    pub fn eta_prefix(self, coordinate: usize) -> usize {
        self.pair_width + coordinate
    }

    pub fn gamma_r(self) -> usize {
        self.pair_width + self.eta_width - 1
    }

    pub fn count(self) -> usize {
        self.pair_width + self.eta_width
    }

    pub fn padded_cols(self) -> usize {
        self.count().next_power_of_two()
    }
}

/// Verifier-visible statement for the committed auxiliary matrix.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PointAuxStatement {
    pub commitment: DoryCommitment,
    pub padded_rows: usize,
    pub pair_width: usize,
    pub eta_width: usize,
}

impl PointAuxStatement {
    pub fn columns(&self) -> PointAuxColumns {
        PointAuxColumns::new(self.pair_width, self.eta_width)
    }

    pub fn validate(&self, layout: &Section6Layout) -> Result<(), Section6DataError> {
        if self.pair_width != layout.pair_width()
            || self.eta_width != layout.eta_width()
            || self.padded_rows != layout.row_cover()?
        {
            return Err(Section6DataError::InvalidPoint(
                "point auxiliary statement does not match the public layout".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn column(&self, column: usize) -> HiddenRelationTableStatement<'_> {
        HiddenRelationTableStatement::matrix_column(
            &self.commitment,
            self.padded_rows,
            self.columns().padded_cols(),
            column,
        )
    }
}

/// Prover-only committed auxiliary matrix.
pub struct CommittedPointAux {
    statement: PointAuxStatement,
    columns: Vec<CommittedDoryFieldVector>,
}

impl CommittedPointAux {
    pub fn statement(&self) -> &PointAuxStatement {
        &self.statement
    }

    pub fn table(&self, column: usize) -> Option<HiddenRelationTable<'_>> {
        let committed = self.columns.get(column)?;
        Some(HiddenRelationTable {
            values: committed.matrix_column_values()?,
            committed,
        })
    }
}

/// Column offsets inside the committed key matrix `[q_key | t_key | V]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PointKeyColumns;

impl PointKeyColumns {
    pub const Q_KEY: usize = 0;
    pub const T_KEY: usize = 1;
    pub const POWER: usize = 2;
    pub const COUNT: usize = 3;

    pub fn padded_cols() -> usize {
        Self::COUNT.next_power_of_two()
    }
}

/// Verifier-visible statement for the committed key matrix.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PointKeyStatement {
    pub commitment: DoryCommitment,
    pub padded_rows: usize,
}

impl PointKeyStatement {
    pub fn validate(&self, layout: &Section6Layout) -> Result<(), Section6DataError> {
        if self.padded_rows != layout.row_cover()? {
            return Err(Section6DataError::InvalidPoint(
                "point key statement does not match the public layout".to_owned(),
            ));
        }
        Ok(())
    }

    pub fn column(&self, column: usize) -> HiddenRelationTableStatement<'_> {
        HiddenRelationTableStatement::matrix_column(
            &self.commitment,
            self.padded_rows,
            PointKeyColumns::padded_cols(),
            column,
        )
    }
}

/// Prover-only committed key matrix.
pub struct CommittedPointKeys {
    statement: PointKeyStatement,
    columns: Vec<CommittedDoryFieldVector>,
}

impl CommittedPointKeys {
    pub fn statement(&self) -> &PointKeyStatement {
        &self.statement
    }

    pub fn table(&self, column: usize) -> Option<HiddenRelationTable<'_>> {
        let committed = self.columns.get(column)?;
        Some(HiddenRelationTable {
            values: committed.matrix_column_values()?,
            committed,
        })
    }
}

/// Serializable point-compilation reduction with no PCS opening embedded.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct PointCompilationProof {
    pub aux: PointAuxStatement,
    pub keys: PointKeyStatement,
    pub lookup: HiddenLogUpPartsReductionProof,
}

/// Prover-side deferred openings produced by one point compilation.
pub struct PointProverOpenings<'a> {
    pub lookup: HiddenLogUpProverOpeningState<'a>,
}

/// Verifier-side deferred openings mirroring the prover structure.
pub struct PointVerifierOpenings<'a> {
    pub lookup: HiddenLogUpVerifierOpeningState<'a>,
}

/// Challenges derived after `X`, the TBV, and the auxiliary matrix are
/// committed and before the key columns are built.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PointChallenges {
    /// Source-key compression base.
    pub vartheta: Fr,
    /// Compaction base for the `Z` identity and the `gamma_r` power column.
    pub theta_z: Fr,
    /// Combination weight merging the eta chain steps into one identity.
    pub eta_combine: Fr,
}

fn encode_with_length(hasher: &mut Sha256, bytes: &[u8]) {
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn encode_serde<T: Serialize>(hasher: &mut Sha256, value: &T) -> Result<(), PointCompilationError> {
    let bytes = bincode::serialize(value)
        .map_err(|error| PointCompilationError::Encoding(error.to_string()))?;
    encode_with_length(hasher, &bytes);
    Ok(())
}

fn sample_nontrivial(binding: &[u8; 32], tag: &[u8]) -> Fr {
    let mut counter = 0_u64;
    loop {
        let mut hasher = Sha256::new();
        hasher.update(b"parcomm-tbv-compilation-challenge");
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

/// Derive the compilation challenges from the public statements.
pub fn derive_point_challenges(
    label: &[u8],
    layout: &Section6Layout,
    shape_statement: &ShapeCommitStatement,
    x_statement: &PointMatrixStatement,
    tbv_statement: &TbvStatement,
    aux_statement: &PointAuxStatement,
    batch_point: &[Fr],
) -> Result<(PointChallenges, [u8; 32]), PointCompilationError> {
    let mut hasher = Sha256::new();
    hasher.update(POINT_DOMAIN);
    encode_with_length(&mut hasher, label);
    encode_serde(&mut hasher, layout)?;
    encode_serde(&mut hasher, shape_statement)?;
    encode_serde(&mut hasher, x_statement)?;
    encode_serde(&mut hasher, tbv_statement)?;
    encode_serde(&mut hasher, aux_statement)?;
    let mut point_bytes = Vec::new();
    for coordinate in batch_point {
        coordinate
            .serialize_compressed(&mut point_bytes)
            .map_err(|error| PointCompilationError::Encoding(error.to_string()))?;
    }
    encode_with_length(&mut hasher, &point_bytes);
    let binding: [u8; 32] = hasher.finalize().into();
    let challenges = PointChallenges {
        vartheta: sample_nontrivial(&binding, b"vartheta-source"),
        theta_z: sample_nontrivial(&binding, b"theta-compaction"),
        eta_combine: sample_nontrivial(&binding, b"eta-combine"),
    };
    Ok((challenges, binding))
}

fn key_binding(
    binding: &[u8; 32],
    keys: &PointKeyStatement,
) -> Result<[u8; 32], PointCompilationError> {
    let mut hasher = Sha256::new();
    hasher.update(b"parcomm-tbv-compilation-keys");
    hasher.update(binding);
    encode_serde(&mut hasher, keys)?;
    Ok(hasher.finalize().into())
}

fn scoped_label(role: &[u8], binding: &[u8; 32]) -> Vec<u8> {
    let mut output = Vec::with_capacity(POINT_DOMAIN.len() + role.len() + 32 + 16);
    output.extend_from_slice(POINT_DOMAIN);
    output.extend_from_slice(&(role.len() as u64).to_le_bytes());
    output.extend_from_slice(role);
    output.extend_from_slice(binding);
    output
}

/// Commit the claim-point matrix `X` before the batching point is sampled.
pub fn commit_point_matrix_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    layout: &Section6Layout,
    claims: &[Section6MatrixClaim<Fr>],
) -> Result<CommittedPointMatrix, PointCompilationError> {
    validate_claims(layout, claims)?;
    let rows = layout.row_cover()?;
    let m = layout.point_log();
    let pair = layout.pair_width();
    let column_values = (0..pair)
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
    let column_slices = column_values.iter().map(Vec::as_slice).collect::<Vec<_>>();
    let padded_cols = pair.next_power_of_two();
    let columns = commit_section6_field_matrix_columns_on_grid_zk(
        parameters,
        grid,
        &column_slices,
        padded_cols,
    )?;
    let commitment = columns
        .first()
        .map(|column| column.commitment.clone())
        .ok_or(PointCompilationError::ProofShape)?;
    Ok(CommittedPointMatrix {
        statement: PointMatrixStatement {
            commitment,
            matrix_count: layout.matrix_count,
            padded_rows: rows,
            padded_cols,
        },
        columns,
    })
}

/// Commit the compiled TBV `[eta | Z]` on the shared grid.
pub fn commit_tbv_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    layout: &Section6Layout,
    witness: &Section6PointWitness<Fr>,
) -> Result<CommittedTbv, PointCompilationError> {
    let rows = layout.row_cover()?;
    let pair = layout.pair_width();
    if witness.eta.padded_len != rows
        || witness.z.padded_rows != rows
        || witness.z.logical_cols != pair
    {
        return Err(PointCompilationError::InvalidWitness(
            "TBV witness does not match the layout row cover".to_owned(),
        ));
    }
    let mut column_slices = Vec::with_capacity(1 + pair);
    column_slices.push(witness.eta.values.as_slice());
    column_slices.extend(witness.z.padded_columns());
    let padded_cols = (1 + pair).next_power_of_two();
    let columns = commit_section6_field_matrix_columns_on_grid_zk(
        parameters,
        grid,
        &column_slices,
        padded_cols,
    )?;
    let commitment = columns
        .first()
        .map(|column| column.commitment.clone())
        .ok_or(PointCompilationError::ProofShape)?;
    Ok(CommittedTbv {
        statement: TbvStatement {
            commitment,
            padded_rows: rows,
            pair_width: pair,
        },
        columns,
    })
}

/// Commit the auxiliary matrix `[X' | Pi'_eta | gamma_r]`.
pub fn commit_point_aux_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    layout: &Section6Layout,
    witness: &Section6PointWitness<Fr>,
) -> Result<CommittedPointAux, PointCompilationError> {
    let rows = layout.row_cover()?;
    let pair = layout.pair_width();
    let eta_width = layout.eta_width();
    if witness.x_selected.padded_rows != rows
        || witness.x_selected.logical_cols != pair
        || witness.eta_prefix.padded_rows != rows
        || witness.eta_prefix.logical_cols != eta_width - 1
        || witness.gamma_r.padded_len != rows
    {
        return Err(PointCompilationError::InvalidWitness(
            "point auxiliary witness does not match the layout row cover".to_owned(),
        ));
    }
    let columns_layout = PointAuxColumns::new(pair, eta_width);
    let mut column_slices = Vec::with_capacity(columns_layout.count());
    column_slices.extend(witness.x_selected.padded_columns());
    column_slices.extend(witness.eta_prefix.padded_columns());
    column_slices.push(witness.gamma_r.values.as_slice());
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
        .ok_or(PointCompilationError::ProofShape)?;
    Ok(CommittedPointAux {
        statement: PointAuxStatement {
            commitment,
            padded_rows: rows,
            pair_width: pair,
            eta_width,
        },
        columns,
    })
}

/// Build and commit the key matrix `[q_key | t_key | V]` after the
/// compilation challenges are fixed.
pub fn commit_point_keys_on_grid_zk(
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    label: &[u8],
    shapes: &CommittedShapeMasks,
    x: &CommittedPointMatrix,
    tbv: &CommittedTbv,
    aux: &CommittedPointAux,
    batch_point: &[Fr],
) -> Result<CommittedPointKeys, PointCompilationError> {
    let layout = shapes.layout();
    let rows = layout.bucket_count;
    let cover = layout.row_cover()?;
    let pair = layout.pair_width();
    let (challenges, _) = derive_point_challenges(
        label,
        layout,
        shapes.statement(),
        x.statement(),
        tbv.statement(),
        aux.statement(),
        batch_point,
    )?;
    let metadata = shapes.metadata();
    let aux_columns = aux.statement.columns();

    // vartheta^{h+1} weights over the pair block.
    let mut vartheta_powers = Vec::with_capacity(pair);
    let mut current = challenges.vartheta;
    for _ in 0..pair {
        vartheta_powers.push(current);
        current *= challenges.vartheta;
    }

    let mut q_key = vec![Fr::zero(); cover];
    for h in 0..pair {
        let column = aux
            .table(aux_columns.x_selected(h))
            .ok_or(PointCompilationError::ProofShape)?;
        let weight = vartheta_powers[h];
        for (value, entry) in q_key.iter_mut().zip(column.values) {
            *value += weight * entry;
        }
    }
    for (row, value) in q_key.iter_mut().take(rows).enumerate() {
        *value += field_from_usize::<Fr>(metadata.tau[row]);
    }

    let mut t_key = vec![Fr::zero(); cover];
    for h in 0..pair {
        let column = x.table(h).ok_or(PointCompilationError::ProofShape)?;
        let weight = vartheta_powers[h];
        for (value, entry) in t_key.iter_mut().zip(column.values) {
            *value += weight * entry;
        }
    }
    for (row, value) in t_key.iter_mut().take(layout.matrix_count).enumerate() {
        *value += field_from_usize::<Fr>(row + 1);
    }

    // V = theta_z^{gamma_r}; padding rows carry the valid exponent-zero pair.
    let mut power = vec![Fr::one(); cover];
    for (row, value) in power.iter_mut().take(rows).enumerate() {
        *value = challenges.theta_z.pow([metadata.gamma_r[row] as u64]);
    }

    let column_slices = [q_key.as_slice(), t_key.as_slice(), power.as_slice()];
    let columns = commit_section6_field_matrix_columns_on_grid_zk(
        parameters,
        grid,
        &column_slices,
        PointKeyColumns::padded_cols(),
    )?;
    let commitment = columns
        .first()
        .map(|column| column.commitment.clone())
        .ok_or(PointCompilationError::ProofShape)?;
    Ok(CommittedPointKeys {
        statement: PointKeyStatement {
            commitment,
            padded_rows: cover,
        },
        columns,
    })
}

/// Public columns used by the compilation identities and lookups.
pub struct PointPublicTables {
    /// `1` on the real bucket rows, `0` on padding rows.
    pub indicator: OwnedPublicColumn,
    /// `i + 1` on the first `t` rows, `0` afterwards.
    pub source_row_ids: OwnedPublicColumn,
    /// Table parts enumerating the exponents `h in [0, m]`.
    pub exponent_parts: Vec<OwnedPublicColumn>,
    /// Table parts enumerating `theta_z^h`.
    pub power_parts: Vec<OwnedPublicColumn>,
}

impl PointPublicTables {
    pub fn derive(
        layout: &Section6Layout,
        challenges: &PointChallenges,
        materialize: bool,
    ) -> Result<Self, PointCompilationError> {
        layout.validate()?;
        let cover = layout.row_cover()?;
        let column =
            |segments: Vec<ColumnSegment>| -> Result<StructuredColumn, PointCompilationError> {
                StructuredColumn::with_zero_tail(cover, segments, Vec::new())
                    .map_err(|error| PointCompilationError::Encoding(error.to_string()))
            };
        let owned = |column: StructuredColumn| -> OwnedPublicColumn {
            OwnedPublicColumn::new(column, materialize)
        };

        let indicator = owned(column(vec![ColumnSegment {
            len: layout.bucket_count,
            pattern: ColumnPattern::Constant(Fr::one()),
        }])?);
        let source_row_ids = owned(column(vec![ColumnSegment {
            len: layout.matrix_count,
            pattern: ColumnPattern::RowIndex { start: Fr::one() },
        }])?);

        let table_len = layout.point_log() + 1;
        let part_count = table_len.div_ceil(cover);
        let mut exponent_parts = Vec::with_capacity(part_count);
        let mut power_parts = Vec::with_capacity(part_count);
        for part in 0..part_count {
            let first = part * cover;
            let live = (table_len - first).min(cover);
            let mut exponent_segments = vec![ColumnSegment {
                len: live,
                pattern: ColumnPattern::RowIndex {
                    start: field_from_usize(first),
                },
            }];
            let mut power_segments = vec![ColumnSegment {
                len: live,
                pattern: ColumnPattern::Geometric {
                    scale: challenges.theta_z.pow([first as u64]),
                    ratio: challenges.theta_z,
                },
            }];
            if live < cover {
                exponent_segments.push(ColumnSegment {
                    len: cover - live,
                    pattern: ColumnPattern::Constant(Fr::zero()),
                });
                power_segments.push(ColumnSegment {
                    len: cover - live,
                    pattern: ColumnPattern::Constant(Fr::one()),
                });
            }
            exponent_parts.push(owned(column(exponent_segments)?));
            power_parts.push(owned(column(power_segments)?));
        }

        Ok(Self {
            indicator,
            source_row_ids,
            exponent_parts,
            power_parts,
        })
    }
}

/// Symbolic operand references resolved by the prover and the verifier.
#[derive(Clone, Copy)]
enum PointTableRef {
    /// Shape mask column `D_eta[h]`.
    DEta(usize),
    /// Shape mask column `D_Z[h]`.
    DZ(usize),
    /// Claim-point matrix column `X[h]`.
    X(usize),
    /// Gathered point column `X'[h]`.
    AuxX(usize),
    /// Eta chain prefix `Pi'_eta[h]`.
    EtaPrefix(usize),
    /// Column prefix lengths `gamma_r`.
    GammaR,
    /// The committed TBV scale column `eta`.
    TbvEta,
    /// The committed TBV point column `Z[c]`.
    TbvZ(usize),
    QKey,
    TKey,
    Power,
}

#[derive(Clone, Copy)]
enum PointPublicRef {
    Indicator,
    SourceRowIds,
}

struct PointIdentityItem {
    hidden: Vec<PointTableRef>,
    publics: Vec<PointPublicRef>,
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

/// The compilation identity batch in a fixed order shared by both sides.
///
/// Every identity holds on padding rows: the masks, `X'`, `Z`, `gamma_r`,
/// and both key columns pad with zeros while the eta chain and `V` pad with
/// ones.
fn point_identity_items(
    layout: &Section6Layout,
    challenges: &PointChallenges,
    batch_point: &[Fr],
) -> Vec<PointIdentityItem> {
    let m = layout.point_log();
    let pair = layout.pair_width();
    let q = layout.eta_width();
    let d = layout.tau_log();
    let mut items = Vec::new();

    // gamma_r = sum_{h<m} D_Z[h] (the column-side prefix length).
    {
        let mut hidden = vec![PointTableRef::GammaR];
        hidden.extend((0..m).map(PointTableRef::DZ));
        let mut expression = expr_table(0);
        for h in 0..m {
            expression = expression.sub(expr_table(1 + h));
        }
        items.push(PointIdentityItem {
            hidden,
            publics: Vec::new(),
            expression,
        });
    }

    // q_key = ind + (1 + <D_tau, 2>) - 1 + <X', vartheta-powers>; the public
    // indicator carries the constant one on real rows only.
    {
        let mut hidden = vec![PointTableRef::QKey];
        hidden.extend((0..d).map(|bit| PointTableRef::DEta(pair + bit)));
        hidden.extend((0..pair).map(PointTableRef::AuxX));
        let indicator = 1 + d + pair;
        let mut expression = expr_table(0).sub(expr_table(indicator));
        for bit in 0..d {
            expression = expression.sub(
                expr_constant(field_from_usize::<Fr>(1_usize << bit)).mul(expr_table(1 + bit)),
            );
        }
        let mut weight = challenges.vartheta;
        for h in 0..pair {
            expression = expression.sub(expr_constant(weight).mul(expr_table(1 + d + h)));
            weight *= challenges.vartheta;
        }
        items.push(PointIdentityItem {
            hidden,
            publics: vec![PointPublicRef::Indicator],
            expression,
        });
    }

    // t_key = source_row_ids + <X, vartheta-powers>.
    {
        let mut hidden = vec![PointTableRef::TKey];
        hidden.extend((0..pair).map(PointTableRef::X));
        let row_ids = 1 + pair;
        let mut expression = expr_table(0).sub(expr_table(row_ids));
        let mut weight = challenges.vartheta;
        for h in 0..pair {
            expression = expression.sub(expr_constant(weight).mul(expr_table(1 + h)));
            weight *= challenges.vartheta;
        }
        items.push(PointIdentityItem {
            hidden,
            publics: vec![PointPublicRef::SourceRowIds],
            expression,
        });
    }

    // Eta chain: Pi'_eta[h] = F_h Pi'_eta[h-1] with point factors on the
    // pair block and public batching factors afterwards. One weighted
    // identity covers all q - 1 steps.
    {
        let mut hidden = (0..q - 1).map(PointTableRef::EtaPrefix).collect::<Vec<_>>();
        hidden.extend((0..q - 1).map(PointTableRef::DEta));
        hidden.extend((0..pair.min(q - 1)).map(PointTableRef::AuxX));
        let deta_base = q - 1;
        let x_base = 2 * (q - 1);
        let mut expression = expr_constant(Fr::zero());
        let mut weight = Fr::one();
        for h in 0..q - 1 {
            let factor = if h < pair {
                // 1 - D + D * X'.
                expr_one()
                    .sub(expr_table(deta_base + h))
                    .add(expr_table(deta_base + h).mul(expr_table(x_base + h)))
            } else {
                // 1 + (x_b - 1) D with the public batching coordinate.
                expr_one().add(
                    expr_constant(batch_point[h - pair] - Fr::one()).mul(expr_table(deta_base + h)),
                )
            };
            let previous = if h == 0 {
                expr_one()
            } else {
                expr_table(h - 1)
            };
            expression =
                expression.add(expr_constant(weight).mul(expr_table(h).sub(factor.mul(previous))));
            weight *= challenges.eta_combine;
        }
        items.push(PointIdentityItem {
            hidden,
            publics: Vec::new(),
            expression,
        });
    }

    // Chain closure: eta = F_{q-1} Pi'_eta[q-2] with the public batching
    // factor 1 + (x_{d-1} - 1) D_eta[q-1].
    {
        let hidden = vec![
            PointTableRef::TbvEta,
            PointTableRef::EtaPrefix(q - 2),
            PointTableRef::DEta(q - 1),
        ];
        let factor =
            expr_one().add(expr_constant(batch_point[d - 1] - Fr::one()).mul(expr_table(2)));
        items.push(PointIdentityItem {
            hidden,
            publics: Vec::new(),
            expression: expr_table(0).sub(factor.mul(expr_table(1))),
        });
    }

    // Compaction identity: <Z, theta_z-powers> equals the masked column
    // block plus theta_z^{gamma_r - m} times the masked row block.
    {
        let mut hidden = (0..pair).map(PointTableRef::TbvZ).collect::<Vec<_>>();
        hidden.extend((0..pair).map(PointTableRef::DZ));
        hidden.extend((0..pair).map(PointTableRef::AuxX));
        hidden.push(PointTableRef::Power);
        let dz_base = pair;
        let x_base = 2 * pair;
        let power = 3 * pair;
        let mut expression = expr_constant(Fr::zero());
        let mut weight = Fr::one();
        for c in 0..pair {
            expression = expression.add(expr_constant(weight).mul(expr_table(c)));
            weight *= challenges.theta_z;
        }
        let mut weight = Fr::one();
        for h in 0..m {
            expression = expression.sub(
                expr_constant(weight)
                    .mul(expr_table(dz_base + h))
                    .mul(expr_table(x_base + h)),
            );
            weight *= challenges.theta_z;
        }
        let inverse_shift = challenges
            .theta_z
            .pow([m as u64])
            .inverse()
            .expect("theta_z is nonzero");
        let mut row_block = expr_constant(Fr::zero());
        let mut weight = challenges.theta_z.pow([m as u64]);
        for h in 0..m {
            row_block = row_block.add(
                expr_constant(weight)
                    .mul(expr_table(dz_base + m + h))
                    .mul(expr_table(x_base + m + h)),
            );
            weight *= challenges.theta_z;
        }
        expression = expression.sub(
            expr_constant(inverse_shift)
                .mul(expr_table(power))
                .mul(row_block),
        );
        items.push(PointIdentityItem {
            hidden,
            publics: Vec::new(),
            expression,
        });
    }

    items
}

/// Caller-owned backing storage for one deferred identity batch.
///
/// The batch prover and verifier borrow per-item table slices with the same
/// lifetime as the returned openings, so the resolved tables must outlive
/// the call; the orchestrator keeps this plan alive next to the openings.
pub struct IdentityPlan<'a, T> {
    items: Vec<IdentityPlanItem<'a, T>>,
}

struct IdentityPlanItem<'a, T> {
    tables: Vec<T>,
    publics: Vec<PublicRelationTable<'a>>,
    expression: RelationExpression,
}

impl<'a, T> IdentityPlan<'a, T> {
    pub(crate) fn new() -> Self {
        Self { items: Vec::new() }
    }

    pub(crate) fn push(
        &mut self,
        tables: Vec<T>,
        publics: Vec<PublicRelationTable<'a>>,
        expression: RelationExpression,
    ) {
        self.items.push(IdentityPlanItem {
            tables,
            publics,
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
                public_tables: &item.publics,
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
                public_tables: &item.publics,
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
pub type PointProverPlan<'a> = IdentityPlan<'a, HiddenRelationTable<'a>>;

/// Verifier-side identity plan for the point compilation.
pub type PointVerifierPlan<'a> = IdentityPlan<'a, HiddenRelationTableStatement<'a>>;

fn resolve_prover_table<'a>(
    shapes: &'a CommittedShapeMasks,
    x: &'a CommittedPointMatrix,
    tbv: &'a CommittedTbv,
    aux: &'a CommittedPointAux,
    keys: &'a CommittedPointKeys,
    reference: &PointTableRef,
) -> Result<HiddenRelationTable<'a>, PointCompilationError> {
    let aux_columns = aux.statement.columns();
    let eta_width = shapes.layout().eta_width();
    let table = match reference {
        PointTableRef::DEta(h) => shapes.mask_table(*h),
        PointTableRef::DZ(h) => shapes.mask_table(eta_width + *h),
        PointTableRef::X(h) => x.table(*h),
        PointTableRef::AuxX(h) => aux.table(aux_columns.x_selected(*h)),
        PointTableRef::EtaPrefix(h) => aux.table(aux_columns.eta_prefix(*h)),
        PointTableRef::GammaR => aux.table(aux_columns.gamma_r()),
        PointTableRef::TbvEta => tbv.eta_table(),
        PointTableRef::TbvZ(c) => tbv.z_table(*c),
        PointTableRef::QKey => keys.table(PointKeyColumns::Q_KEY),
        PointTableRef::TKey => keys.table(PointKeyColumns::T_KEY),
        PointTableRef::Power => keys.table(PointKeyColumns::POWER),
    };
    table.ok_or(PointCompilationError::ProofShape)
}

fn resolve_verifier_table<'a>(
    layout: &Section6Layout,
    shape_statement: &'a ShapeCommitStatement,
    x_statement: &'a PointMatrixStatement,
    tbv_statement: &'a TbvStatement,
    aux_statement: &'a PointAuxStatement,
    key_statement: &'a PointKeyStatement,
    reference: &PointTableRef,
) -> HiddenRelationTableStatement<'a> {
    let aux_columns = aux_statement.columns();
    let eta_width = layout.eta_width();
    match reference {
        PointTableRef::DEta(h) => shape_statement.mask_column(*h),
        PointTableRef::DZ(h) => shape_statement.mask_column(eta_width + *h),
        PointTableRef::X(h) => x_statement.column(*h),
        PointTableRef::AuxX(h) => aux_statement.column(aux_columns.x_selected(*h)),
        PointTableRef::EtaPrefix(h) => aux_statement.column(aux_columns.eta_prefix(*h)),
        PointTableRef::GammaR => aux_statement.column(aux_columns.gamma_r()),
        PointTableRef::TbvEta => tbv_statement.eta_column(),
        PointTableRef::TbvZ(c) => tbv_statement.z_column(*c),
        PointTableRef::QKey => key_statement.column(PointKeyColumns::Q_KEY),
        PointTableRef::TKey => key_statement.column(PointKeyColumns::T_KEY),
        PointTableRef::Power => key_statement.column(PointKeyColumns::POWER),
    }
}

fn resolve_public<'a>(
    publics: &'a PointPublicTables,
    reference: &PointPublicRef,
) -> PublicRelationTable<'a> {
    match reference {
        PointPublicRef::Indicator => publics.indicator.table(),
        PointPublicRef::SourceRowIds => publics.source_row_ids.table(),
    }
}

/// Build the caller-owned prover identity plan for one point compilation.
pub fn build_point_prover_plan<'a>(
    shapes: &'a CommittedShapeMasks,
    x: &'a CommittedPointMatrix,
    tbv: &'a CommittedTbv,
    aux: &'a CommittedPointAux,
    keys: &'a CommittedPointKeys,
    publics: &'a PointPublicTables,
    challenges: &PointChallenges,
    batch_point: &[Fr],
) -> Result<PointProverPlan<'a>, PointCompilationError> {
    let items = point_identity_items(shapes.layout(), challenges, batch_point);
    let mut plan = IdentityPlan::new();
    for item in items {
        let tables = item
            .hidden
            .iter()
            .map(|reference| resolve_prover_table(shapes, x, tbv, aux, keys, reference))
            .collect::<Result<Vec<_>, _>>()?;
        let public_tables = item
            .publics
            .iter()
            .map(|reference| resolve_public(publics, reference))
            .collect::<Vec<_>>();
        plan.push(tables, public_tables, item.expression);
    }
    Ok(plan)
}

/// Build the caller-owned verifier identity plan for one point compilation.
#[allow(clippy::too_many_arguments)]
pub fn build_point_verifier_plan<'a>(
    layout: &Section6Layout,
    shape_statement: &'a ShapeCommitStatement,
    x_statement: &'a PointMatrixStatement,
    tbv_statement: &'a TbvStatement,
    aux_statement: &'a PointAuxStatement,
    key_statement: &'a PointKeyStatement,
    publics: &'a PointPublicTables,
    challenges: &PointChallenges,
    batch_point: &[Fr],
) -> Result<PointVerifierPlan<'a>, PointCompilationError> {
    aux_statement.validate(layout)?;
    key_statement.validate(layout)?;
    let items = point_identity_items(layout, challenges, batch_point);
    let mut plan = IdentityPlan::new();
    for item in items {
        let tables = item
            .hidden
            .iter()
            .map(|reference| {
                resolve_verifier_table(
                    layout,
                    shape_statement,
                    x_statement,
                    tbv_statement,
                    aux_statement,
                    key_statement,
                    reference,
                )
            })
            .collect::<Vec<_>>();
        let public_tables = item
            .publics
            .iter()
            .map(|reference| resolve_public(publics, reference))
            .collect::<Vec<_>>();
        plan.push(tables, public_tables, item.expression);
    }
    Ok(plan)
}

/// Reduce the complete point compilation while deferring every Dory opening.
#[allow(clippy::too_many_arguments)]
pub fn prove_point_compilation_deferred<'a>(
    label: &[u8],
    parameters: &DoryProverParameters,
    grid: &Section6DoryGrid,
    shapes: &'a CommittedShapeMasks,
    x: &'a CommittedPointMatrix,
    tbv: &'a CommittedTbv,
    aux: &'a CommittedPointAux,
    keys: &'a CommittedPointKeys,
    publics: &'a PointPublicTables,
    batch_point: &[Fr],
    rng: &mut impl RngCore,
) -> Result<(PointCompilationProof, PointProverOpenings<'a>), PointCompilationError> {
    let layout = shapes.layout();
    let (_, binding) = derive_point_challenges(
        label,
        layout,
        shapes.statement(),
        x.statement(),
        tbv.statement(),
        aux.statement(),
        batch_point,
    )?;
    let binding = key_binding(&binding, keys.statement())?;

    // The source and power relations share one tagged LogUp. The family tag is
    // part of each tuple, so a query from one family cannot match the table of
    // the other family after the union is batched.
    let q_key_table = keys
        .table(PointKeyColumns::Q_KEY)
        .ok_or(PointCompilationError::ProofShape)?;
    let t_key_table = keys
        .table(PointKeyColumns::T_KEY)
        .ok_or(PointCompilationError::ProofShape)?;
    let source_query = vec![vec![MixedLogUpColumn::Hidden(q_key_table)]];
    let source_tables = vec![vec![MixedLogUpColumn::Hidden(t_key_table)]];
    let gamma_r_table = aux
        .table(aux.statement.columns().gamma_r())
        .ok_or(PointCompilationError::ProofShape)?;
    let power_table = keys
        .table(PointKeyColumns::POWER)
        .ok_or(PointCompilationError::ProofShape)?;
    let power_query = vec![vec![
        MixedLogUpColumn::Hidden(gamma_r_table),
        MixedLogUpColumn::Hidden(power_table),
    ]];
    let power_tables = publics
        .exponent_parts
        .iter()
        .zip(&publics.power_parts)
        .map(|(exponent, value)| {
            vec![
                MixedLogUpColumn::Public(exponent.table()),
                MixedLogUpColumn::Public(value.table()),
            ]
        })
        .collect::<Vec<_>>();
    let families = [
        HiddenLogUpUnionProverFamily {
            query_parts: &source_query,
            table_parts: &source_tables,
        },
        HiddenLogUpUnionProverFamily {
            query_parts: &power_query,
            table_parts: &power_tables,
        },
    ];
    let query_hidden_parts = vec![vec![q_key_table], vec![gamma_r_table, power_table]];
    let mut table_hidden_parts = vec![vec![t_key_table]];
    table_hidden_parts.extend((0..power_tables.len()).map(|_| Vec::new()));
    let (lookup, lookup_state) = prove_hidden_logup_union_deferred_on_grid_with_hidden_tables(
        &scoped_label(b"lookup-union", &binding),
        parameters,
        grid,
        &families,
        query_hidden_parts,
        table_hidden_parts,
        rng,
    )?;

    Ok((
        PointCompilationProof {
            aux: aux.statement.clone(),
            keys: keys.statement.clone(),
            lookup,
        },
        PointProverOpenings {
            lookup: lookup_state,
        },
    ))
}

/// Verify the deferred point compilation from public statements only.
#[allow(clippy::too_many_arguments)]
pub fn verify_point_compilation_deferred<'a>(
    label: &[u8],
    parameters: &DoryVerifierParameters,
    grid: &Section6DoryGrid,
    layout: &Section6Layout,
    shape_statement: &'a ShapeCommitStatement,
    x_statement: &'a PointMatrixStatement,
    tbv_statement: &'a TbvStatement,
    publics: &'a PointPublicTables,
    batch_point: &[Fr],
    proof: &'a PointCompilationProof,
) -> Result<PointVerifierOpenings<'a>, PointCompilationError> {
    layout.validate()?;
    shape_statement.validate(layout)?;
    x_statement.validate(layout)?;
    tbv_statement.validate(layout)?;
    proof.aux.validate(layout)?;
    proof.keys.validate(layout)?;
    let (_, binding) = derive_point_challenges(
        label,
        layout,
        shape_statement,
        x_statement,
        tbv_statement,
        &proof.aux,
        batch_point,
    )?;
    let binding = key_binding(&binding, &proof.keys)?;

    let source_query = vec![vec![MixedLogUpColumnStatement::Hidden(
        proof.keys.column(PointKeyColumns::Q_KEY),
    )]];
    let source_tables = vec![vec![MixedLogUpColumnStatement::Hidden(
        proof.keys.column(PointKeyColumns::T_KEY),
    )]];
    let power_query = vec![vec![
        MixedLogUpColumnStatement::Hidden(proof.aux.column(proof.aux.columns().gamma_r())),
        MixedLogUpColumnStatement::Hidden(proof.keys.column(PointKeyColumns::POWER)),
    ]];
    let power_tables = publics
        .exponent_parts
        .iter()
        .zip(&publics.power_parts)
        .map(|(exponent, value)| {
            vec![
                MixedLogUpColumnStatement::Public(exponent.table()),
                MixedLogUpColumnStatement::Public(value.table()),
            ]
        })
        .collect::<Vec<_>>();
    let families = [
        HiddenLogUpUnionVerifierFamily {
            query_parts: &source_query,
            table_parts: &source_tables,
        },
        HiddenLogUpUnionVerifierFamily {
            query_parts: &power_query,
            table_parts: &power_tables,
        },
    ];
    let query_hidden_parts = vec![
        vec![proof.keys.column(PointKeyColumns::Q_KEY)],
        vec![
            proof.aux.column(proof.aux.columns().gamma_r()),
            proof.keys.column(PointKeyColumns::POWER),
        ],
    ];
    let mut table_hidden_parts = vec![vec![proof.keys.column(PointKeyColumns::T_KEY)]];
    table_hidden_parts.extend((0..power_tables.len()).map(|_| Vec::new()));
    let lookup = verify_hidden_logup_union_deferred_on_grid_with_hidden_tables(
        &scoped_label(b"lookup-union", &binding),
        parameters,
        grid,
        &families,
        &proof.lookup,
        query_hidden_parts,
        table_hidden_parts,
    )?;

    Ok(PointVerifierOpenings { lookup })
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
        section6_final_opening::{prove_section6_final_opening, verify_section6_final_opening},
        validity_shape::commit_shape_masks_from_descriptor_on_grid_zk,
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
        let grid = Section6DoryGrid::new(12).unwrap();
        let (prover_parameters, verifier_parameters) = setup_dory_parameters(grid.num_vars());
        let mut rng = StdRng::seed_from_u64(0x504f_494e_545f_5631);

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
            .map(|bit| Fr::from((101 + 3 * bit) as u64))
            .collect::<Vec<_>>();

        let x =
            commit_point_matrix_on_grid_zk(&prover_parameters, &grid, &layout, &claims).unwrap();
        let witness =
            Section6PointWitness::compile(&layout, &descriptor, &metadata, &claims, &batch_point)
                .unwrap();
        let tbv = commit_tbv_on_grid_zk(&prover_parameters, &grid, &layout, &witness).unwrap();
        let aux =
            commit_point_aux_on_grid_zk(&prover_parameters, &grid, &layout, &witness).unwrap();
        let keys = commit_point_keys_on_grid_zk(
            &prover_parameters,
            &grid,
            b"point-test",
            &committed_shapes,
            &x,
            &tbv,
            &aux,
            &batch_point,
        )
        .unwrap();
        let (challenges, _) = derive_point_challenges(
            b"point-test",
            &layout,
            committed_shapes.statement(),
            x.statement(),
            tbv.statement(),
            aux.statement(),
            &batch_point,
        )
        .unwrap();
        let publics = PointPublicTables::derive(&layout, &challenges, true).unwrap();

        let (proof, prover_openings) = prove_point_compilation_deferred(
            b"point-test",
            &prover_parameters,
            &grid,
            &committed_shapes,
            &x,
            &tbv,
            &aux,
            &keys,
            &publics,
            &batch_point,
            &mut rng,
        )
        .unwrap();

        let verifier_publics = PointPublicTables::derive(&layout, &challenges, false).unwrap();
        let verifier_openings = verify_point_compilation_deferred(
            b"point-test",
            &verifier_parameters,
            &grid,
            &layout,
            committed_shapes.statement(),
            x.statement(),
            tbv.statement(),
            &verifier_publics,
            &batch_point,
            &proof,
        )
        .unwrap();

        // Every lookup claim resolves in one shared terminal batch. Point
        // identities are exercised by the end-to-end joint Point/Final test.
        let mut prover_claims = Section6ProverClaimCollection::new();
        for opening in prover_openings.lookup.relation_openings().unwrap() {
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
        for opening in verifier_openings.lookup.relation_openings().unwrap() {
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

        // A tampered auxiliary commitment must break verification.
        let mut tampered = proof.clone();
        tampered.aux.commitment = x.statement().commitment.clone();
        assert!(verify_point_compilation_deferred(
            b"point-test",
            &verifier_parameters,
            &grid,
            &layout,
            committed_shapes.statement(),
            x.statement(),
            tbv.statement(),
            &verifier_publics,
            &batch_point,
            &tampered,
        )
        .is_err());
    }
}
