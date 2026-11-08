//! Extension-field carry commitments used by short-field QAFold Decide.
//!
//! This module is a child of `qafold_short_pc`. A carry message is encoded by
//! either the scalar extension of a base-field random-foldable generator or a
//! direct lower-level QA interleaving. Its row commitment uses the same BLAKE2s
//! extension-row format as recursive PCS rounds.

use super::*;

use super::decide::{
    ShortEvalWithPoint, ShortQaRfcTClaim, reduce_qa_rfc_generator_claim_prover,
    reduce_qa_rfc_generator_claim_verifier,
};
use super::setup::verify_short_setup_rows;
use crate::qafold_short::{ShortQaMatrixClaim, ShortQaRandomFoldableCode};

const IOTA_DOMAIN: &[u8] = b"qafold-short/decide-iota/v1";
const CODE_SWITCH_DOMAIN: &[u8] = b"qafold-short/decide-code-switch/v1";
const DIRECT_CODE_SWITCH_DOMAIN: &[u8] = b"qafold-short/decide-direct-code-switch/v2";

/// Prover state retained for one derived extension-field carry commitment.
///
/// The logical shape describes the recursive witness before interleaving. The
/// encoding shape describes the columns actually passed to the lower code and
/// stored in each authenticated row. They coincide for RF carries and differ
/// for direct-interleaved carries.
pub(super) struct ShortQaDerivedCarryCache<B: ShortBaseField> {
    message_entries: Vec<B::Extension>,
    encoded_entries: Vec<B::Extension>,
    logical_rows: usize,
    logical_cols: usize,
    encoding_message_rows: usize,
    encoded_rows: usize,
    physical_cols: usize,
    tree: Blake2sMerkleTree,
}

impl<B: ShortBaseField> ShortQaDerivedCarryCache<B> {
    pub(super) fn message_entries(&self) -> &[B::Extension] {
        &self.message_entries
    }

    pub(super) fn logical_rows(&self) -> usize {
        self.logical_rows
    }

    pub(super) fn logical_columns(&self) -> usize {
        self.logical_cols
    }

    pub(super) fn encoding_message_rows(&self) -> usize {
        self.encoding_message_rows
    }

    pub(super) fn encoded_rows(&self) -> usize {
        self.encoded_rows
    }

    pub(super) fn physical_columns(&self) -> usize {
        self.physical_cols
    }

    pub(super) fn encoded_entries_len(&self) -> usize {
        self.encoded_entries.len()
    }
}

/// Encodes and commits an owned column-major carry matrix over the extension.
pub(super) fn commit_short_derived_carry<B: ShortBaseField>(
    source: &ShortQaRandomFoldableCode<B>,
    target: &ShortQaCode<B>,
    message_entries: Vec<B::Extension>,
    n_cols: usize,
) -> Result<(ShortQaFoldCommitment, ShortQaDerivedCarryCache<B>), Error> {
    let n_message_rows = source.message_len(target)?;
    let n_encoded_rows = source.codeword_len(target)?;
    let message_len = n_message_rows
        .checked_mul(n_cols)
        .ok_or(Error::EncodingError)?;
    let encoded_len = n_encoded_rows
        .checked_mul(n_cols)
        .ok_or(Error::EncodingError)?;
    if n_cols == 0 || message_entries.len() != message_len {
        return Err(Error::EncodingError);
    }

    let mut encoded_entries = vec![B::Extension::zero(); encoded_len];
    #[cfg(feature = "parallel")]
    encoded_entries
        .par_chunks_mut(n_encoded_rows)
        .zip(message_entries.par_chunks(n_message_rows))
        .try_for_each(|(output, message)| source.encode_extension_into(target, message, output))?;
    #[cfg(not(feature = "parallel"))]
    encoded_entries
        .chunks_mut(n_encoded_rows)
        .zip(message_entries.chunks(n_message_rows))
        .try_for_each(|(output, message)| source.encode_extension_into(target, message, output))?;

    let tree = build_extension_tree::<B>(&encoded_entries, n_encoded_rows, n_cols)?;
    let commitment = ShortQaFoldCommitment { root: tree.root() };
    Ok((
        commitment,
        ShortQaDerivedCarryCache {
            message_entries,
            encoded_entries,
            logical_rows: n_message_rows,
            logical_cols: n_cols,
            encoding_message_rows: n_message_rows,
            encoded_rows: n_encoded_rows,
            physical_cols: n_cols,
            tree,
        },
    ))
}

/// Encodes a logically taller carry after splitting every logical column into
/// `row_factor` columns of the next smaller QA code.
pub(super) fn commit_short_direct_interleaved_carry<B: ShortBaseField>(
    target: &ShortQaCode<B>,
    message_entries: Vec<B::Extension>,
    logical_rows: usize,
    logical_cols: usize,
    row_factor: usize,
) -> Result<(ShortQaFoldCommitment, ShortQaDerivedCarryCache<B>), Error> {
    let n_message_rows = target.message_len();
    let n_encoded_rows = target.codeword_len()?;
    let n_cols = logical_cols
        .checked_mul(row_factor)
        .ok_or(Error::EncodingError)?;
    if row_factor <= 1
        || !row_factor.is_power_of_two()
        || logical_cols == 0
        || logical_rows
            != n_message_rows
                .checked_mul(row_factor)
                .ok_or(Error::EncodingError)?
        || message_entries.len()
            != logical_rows
                .checked_mul(logical_cols)
                .ok_or(Error::EncodingError)?
    {
        return Err(Error::EncodingError);
    }

    let encoded_len = n_encoded_rows
        .checked_mul(n_cols)
        .ok_or(Error::EncodingError)?;
    let mut encoded_entries = vec![B::Extension::zero(); encoded_len];
    #[cfg(feature = "parallel")]
    encoded_entries
        .par_chunks_mut(n_encoded_rows)
        .zip(message_entries.par_chunks(n_message_rows))
        .try_for_each(|(output, message)| target.encode_extension_into(message, output))?;
    #[cfg(not(feature = "parallel"))]
    encoded_entries
        .chunks_mut(n_encoded_rows)
        .zip(message_entries.chunks(n_message_rows))
        .try_for_each(|(output, message)| target.encode_extension_into(message, output))?;

    let tree = build_extension_tree::<B>(&encoded_entries, n_encoded_rows, n_cols)?;
    let commitment = ShortQaFoldCommitment { root: tree.root() };
    Ok((
        commitment,
        ShortQaDerivedCarryCache {
            message_entries,
            encoded_entries,
            logical_rows,
            logical_cols,
            encoding_message_rows: n_message_rows,
            encoded_rows: n_encoded_rows,
            physical_cols: n_cols,
            tree,
        },
    ))
}

/// Folds the block coordinate of one logical witness vector before recursion.
///
/// The input has `row_factor` contiguous blocks of `target_rows` entries. The
/// result is one `target_rows`-entry column whose MLE at the remaining row
/// point equals the input MLE at the complete row point.
pub(super) fn fold_short_direct_interleaved_witness<E: ShortFieldElement>(
    logical_entries: Vec<E>,
    target_rows: usize,
    row_factor: usize,
    logical_row_point: &[E],
) -> Result<Vec<E>, Error> {
    if target_rows == 0
        || !target_rows.is_power_of_two()
        || row_factor <= 1
        || !row_factor.is_power_of_two()
    {
        return Err(Error::InvalidNumberOfVariables);
    }
    let logical_rows = target_rows
        .checked_mul(row_factor)
        .ok_or(Error::EncodingError)?;
    let block_log = row_factor.ilog2() as usize;
    if logical_entries.len() != logical_rows
        || logical_row_point.len() != logical_rows.ilog2() as usize
    {
        return Err(Error::InvalidNumberOfVariables);
    }
    let block_weights = tensor_weights(&logical_row_point[..block_log]);
    let fold_row = |row: usize| {
        block_weights
            .iter()
            .enumerate()
            .fold(E::zero(), |sum, (block, weight)| {
                sum + logical_entries[block * target_rows + row] * *weight
            })
    };
    #[cfg(feature = "parallel")]
    {
        Ok((0..target_rows).into_par_iter().map(fold_row).collect())
    }
    #[cfg(not(feature = "parallel"))]
    {
        Ok((0..target_rows).map(fold_row).collect())
    }
}

/// Consumes a carry cache and moves an all-row codeword directly into the proof.
pub(super) fn into_short_derived_carry_rows<B: ShortBaseField>(
    cache: ShortQaDerivedCarryCache<B>,
    selection: &RowSelection,
) -> Result<ShortQaExtensionRows<B>, Error> {
    validate_carry_selection(selection, cache.encoded_rows)?;
    if selection.is_all() {
        return Ok(ShortQaExtensionRows::AllRows {
            encoded_entries: cache.encoded_entries,
            n_rows: cache.encoded_rows,
            n_cols: cache.physical_cols,
        });
    }

    let open_row = |row: usize| {
        Ok(ShortQaExtensionRowQuery {
            row: extension_encoded_row::<B>(
                &cache.encoded_entries,
                cache.encoded_rows,
                cache.physical_cols,
                row,
            ),
            path: cache.tree.path(row)?,
        })
    };
    #[cfg(feature = "parallel")]
    let queries = selection
        .iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(open_row)
        .collect::<Result<Vec<_>, Error>>()?;
    #[cfg(not(feature = "parallel"))]
    let queries = selection
        .iter()
        .map(open_row)
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(ShortQaExtensionRows::Partial(queries))
}

/// Verifies partial paths or reconstructs an all-row carry commitment.
pub(super) fn verify_short_derived_carry_rows<B: ShortBaseField>(
    commitment: &ShortQaFoldCommitment,
    n_encoded_rows: usize,
    n_cols: usize,
    selection: &RowSelection,
    rows: &ShortQaExtensionRows<B>,
) -> Result<(), Error> {
    validate_carry_selection(selection, n_encoded_rows)?;
    if n_cols == 0 {
        return Err(Error::EncodingError);
    }
    match (selection, rows) {
        (
            RowSelection::All(count),
            ShortQaExtensionRows::AllRows {
                encoded_entries,
                n_rows,
                n_cols: encoded_cols,
            },
        ) => {
            let expected_len = n_encoded_rows
                .checked_mul(n_cols)
                .ok_or(Error::EncodingError)?;
            if *count != n_encoded_rows
                || *n_rows != n_encoded_rows
                || *encoded_cols != n_cols
                || encoded_entries.len() != expected_len
                || build_extension_tree::<B>(encoded_entries, n_encoded_rows, n_cols)?.root()
                    != commitment.root
            {
                return Err(Error::InvalidCommitment);
            }
            Ok(())
        }
        (RowSelection::Partial(expected_rows), ShortQaExtensionRows::Partial(queries)) => {
            if queries.len() != expected_rows.len()
                || queries.iter().any(|query| query.row.len() != n_cols)
            {
                return Err(Error::InvalidCommitment);
            }
            let verify = |(query, row): (&ShortQaExtensionRowQuery<B>, &usize)| {
                let leaf = hash_extension_row::<B>(&query.row)?;
                if !verify_merkle_path(leaf, *row, n_encoded_rows, &query.path, &commitment.root) {
                    return Err(Error::InvalidCommitment);
                }
                Ok(())
            };
            #[cfg(feature = "parallel")]
            {
                queries
                    .par_iter()
                    .zip(expected_rows.par_iter())
                    .try_for_each(verify)
            }
            #[cfg(not(feature = "parallel"))]
            {
                queries.iter().zip(expected_rows).try_for_each(verify)
            }
        }
        _ => Err(Error::InvalidCommitment),
    }
}

/// Claims carried from one short-field Decide code switch to the next level.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) struct ShortQaCodeSwitchCarry<E: ShortFieldElement> {
    pub(super) generator_claim: ShortQaMatrixClaim<E>,
    pub(super) folded_w_claim: ShortEvalWithPoint<E>,
    pub(super) t_claims: Vec<ShortQaRfcTClaim<E>>,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prove_short_decide_iota<B: ShortBaseField>(
    level: usize,
    setup_commitment: &ShortQaFoldCommitment,
    setup_shape: ShortQaSetupShape,
    setup_rows: &ShortQaSetupRows<B>,
    carry_commitment: Option<&ShortQaFoldCommitment>,
    carry_rows: Option<&ShortQaExtensionRows<B>>,
    carry_cols: usize,
    selection: &RowSelection,
    col_weights: &[B::Extension],
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<Vec<B::Extension>, Error> {
    validate_iota_statement(
        level,
        setup_shape,
        carry_commitment,
        carry_rows,
        carry_cols,
        selection,
        col_weights.len(),
    )?;
    absorb_iota_statement_writer(
        level,
        setup_commitment,
        setup_shape,
        carry_commitment,
        carry_cols,
        selection,
        col_weights,
        transcript,
    )?;
    let iota = compute_iota::<B>(
        setup_shape,
        setup_rows,
        carry_rows,
        carry_cols,
        selection,
        col_weights,
    )?;
    for value in &iota {
        transcript.push_response(*value)?;
    }
    Ok(iota)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn verify_short_decide_iota<B: ShortBaseField>(
    level: usize,
    setup_commitment: &ShortQaFoldCommitment,
    setup_shape: ShortQaSetupShape,
    setup_rows: &ShortQaSetupRows<B>,
    carry_commitment: Option<&ShortQaFoldCommitment>,
    carry_rows: Option<&ShortQaExtensionRows<B>>,
    carry_cols: usize,
    selection: &RowSelection,
    col_weights: &[B::Extension],
    transcript: &mut ShortTranscriptReader<'_, B>,
) -> Result<Vec<B::Extension>, Error> {
    validate_iota_statement(
        level,
        setup_shape,
        carry_commitment,
        carry_rows,
        carry_cols,
        selection,
        col_weights.len(),
    )?;
    verify_short_setup_rows(setup_commitment, setup_shape, selection, setup_rows)?;
    if let (Some(commitment), Some(rows)) = (carry_commitment, carry_rows) {
        verify_short_derived_carry_rows(
            commitment,
            setup_shape.encoded_rows,
            carry_cols,
            selection,
            rows,
        )?;
    }
    absorb_iota_statement_reader(
        level,
        setup_commitment,
        setup_shape,
        carry_commitment,
        carry_cols,
        selection,
        col_weights,
        transcript,
    )?;
    let expected = compute_iota::<B>(
        setup_shape,
        setup_rows,
        carry_rows,
        carry_cols,
        selection,
        col_weights,
    )?;
    for value in &expected {
        if transcript.read_response()? != *value {
            return Err(Error::InvalidCommitment);
        }
    }
    Ok(expected)
}

fn validate_iota_statement<B: ShortBaseField>(
    level: usize,
    setup_shape: ShortQaSetupShape,
    carry_commitment: Option<&ShortQaFoldCommitment>,
    carry_rows: Option<&ShortQaExtensionRows<B>>,
    carry_cols: usize,
    selection: &RowSelection,
    col_weights_len: usize,
) -> Result<(), Error> {
    setup_shape.validate()?;
    validate_carry_selection(selection, setup_shape.encoded_rows)?;
    let carry_present = carry_cols > 0;
    if level == 0
        || setup_shape.level != level
        || carry_commitment.is_some() != carry_present
        || carry_rows.is_some() != carry_present
    {
        return Err(Error::InvalidNumberOfVariables);
    }
    let real_cols = setup_shape
        .total_cols()?
        .checked_add(carry_cols)
        .ok_or(Error::EncodingError)?;
    let padded_cols = real_cols
        .checked_next_power_of_two()
        .ok_or(Error::EncodingError)?;
    if col_weights_len != padded_cols {
        return Err(Error::InvalidNumberOfVariables);
    }
    Ok(())
}

fn compute_iota<B: ShortBaseField>(
    setup_shape: ShortQaSetupShape,
    setup_rows: &ShortQaSetupRows<B>,
    carry_rows: Option<&ShortQaExtensionRows<B>>,
    carry_cols: usize,
    selection: &RowSelection,
    col_weights: &[B::Extension],
) -> Result<Vec<B::Extension>, Error> {
    let setup_cols = setup_shape.total_cols()?;
    let evaluate = |(ordinal, row): (usize, usize)| {
        let setup_value = setup_row_weighted_value::<B>(
            setup_rows,
            setup_shape,
            selection,
            ordinal,
            row,
            &col_weights[..setup_cols],
        )?;
        let carry_value = match carry_rows {
            Some(rows) => carry_row_weighted_value::<B>(
                rows,
                setup_shape.encoded_rows,
                carry_cols,
                selection,
                ordinal,
                row,
                &col_weights[setup_cols..setup_cols + carry_cols],
            )?,
            None => B::Extension::zero(),
        };
        Ok(setup_value + carry_value)
    };
    let indexed_rows = selection.iter().enumerate().collect::<Vec<_>>();
    #[cfg(feature = "parallel")]
    {
        indexed_rows
            .into_par_iter()
            .map(evaluate)
            .collect::<Result<Vec<_>, Error>>()
    }
    #[cfg(not(feature = "parallel"))]
    {
        indexed_rows
            .into_iter()
            .map(evaluate)
            .collect::<Result<Vec<_>, Error>>()
    }
}

fn setup_row_weighted_value<B: ShortBaseField>(
    rows: &ShortQaSetupRows<B>,
    shape: ShortQaSetupShape,
    selection: &RowSelection,
    ordinal: usize,
    row: usize,
    weights: &[B::Extension],
) -> Result<B::Extension, Error> {
    if weights.len() != shape.total_cols()? {
        return Err(Error::InvalidNumberOfVariables);
    }
    match (selection, rows) {
        (RowSelection::Partial(_), ShortQaSetupRows::Partial(queries)) => {
            let query = queries.get(ordinal).ok_or(Error::InvalidCommitment)?;
            if query.row_index != row || query.row.len() != weights.len() {
                return Err(Error::InvalidCommitment);
            }
            Ok(query
                .row
                .iter()
                .zip(weights)
                .fold(B::Extension::zero(), |sum, (value, weight)| {
                    sum + weight.mul_base(*value)
                }))
        }
        (
            RowSelection::All(_),
            ShortQaSetupRows::AllRows {
                encoded_entries,
                shape: encoded_shape,
            },
        ) => {
            if *encoded_shape != shape
                || row >= shape.encoded_rows
                || encoded_entries.len()
                    != shape
                        .encoded_rows
                        .checked_mul(weights.len())
                        .ok_or(Error::EncodingError)?
            {
                return Err(Error::InvalidCommitment);
            }
            Ok(weights
                .iter()
                .enumerate()
                .fold(B::Extension::zero(), |sum, (column, weight)| {
                    sum + weight.mul_base(encoded_entries[column * shape.encoded_rows + row])
                }))
        }
        _ => Err(Error::InvalidCommitment),
    }
}

fn carry_row_weighted_value<B: ShortBaseField>(
    rows: &ShortQaExtensionRows<B>,
    n_encoded_rows: usize,
    n_cols: usize,
    selection: &RowSelection,
    ordinal: usize,
    row: usize,
    weights: &[B::Extension],
) -> Result<B::Extension, Error> {
    if weights.len() != n_cols {
        return Err(Error::InvalidNumberOfVariables);
    }
    match (selection, rows) {
        (RowSelection::Partial(_), ShortQaExtensionRows::Partial(queries)) => {
            let query = queries.get(ordinal).ok_or(Error::InvalidCommitment)?;
            inner_product(&query.row, weights)
        }
        (
            RowSelection::All(_),
            ShortQaExtensionRows::AllRows {
                encoded_entries,
                n_rows,
                n_cols: encoded_cols,
            },
        ) => {
            if *n_rows != n_encoded_rows
                || *encoded_cols != n_cols
                || row >= n_encoded_rows
                || encoded_entries.len()
                    != n_encoded_rows
                        .checked_mul(n_cols)
                        .ok_or(Error::EncodingError)?
            {
                return Err(Error::InvalidCommitment);
            }
            extension_row_inner_product::<B>(encoded_entries, n_encoded_rows, n_cols, row, weights)
        }
        _ => Err(Error::InvalidCommitment),
    }
}

#[allow(clippy::too_many_arguments)]
fn absorb_iota_statement_writer<B: ShortBaseField>(
    level: usize,
    setup_commitment: &ShortQaFoldCommitment,
    setup_shape: ShortQaSetupShape,
    carry_commitment: Option<&ShortQaFoldCommitment>,
    carry_cols: usize,
    selection: &RowSelection,
    col_weights: &[B::Extension],
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<(), Error> {
    transcript.absorb_bytes(b"iota-domain", IOTA_DOMAIN);
    absorb_iota_public_writer(
        level,
        setup_commitment,
        setup_shape,
        carry_commitment,
        carry_cols,
        selection,
        col_weights,
        transcript,
    )
}

#[allow(clippy::too_many_arguments)]
fn absorb_iota_statement_reader<B: ShortBaseField>(
    level: usize,
    setup_commitment: &ShortQaFoldCommitment,
    setup_shape: ShortQaSetupShape,
    carry_commitment: Option<&ShortQaFoldCommitment>,
    carry_cols: usize,
    selection: &RowSelection,
    col_weights: &[B::Extension],
    transcript: &mut ShortTranscriptReader<'_, B>,
) -> Result<(), Error> {
    transcript.absorb_bytes(b"iota-domain", IOTA_DOMAIN);
    transcript.absorb_bytes(b"iota-level", &(level as u64).to_le_bytes());
    absorb_iota_shape_reader(setup_shape, transcript);
    transcript.absorb_digest(b"iota-setup-root", &setup_commitment.root);
    match carry_commitment {
        Some(commitment) => {
            transcript.absorb_bytes(b"iota-carry-present", &[1]);
            transcript.absorb_digest(b"iota-carry-root", &commitment.root);
        }
        None => transcript.absorb_bytes(b"iota-carry-present", &[0]),
    }
    transcript.absorb_bytes(b"iota-carry-cols", &(carry_cols as u64).to_le_bytes());
    absorb_selection_reader(selection, transcript);
    transcript.absorb_extensions(b"iota-column-weights", col_weights)
}

#[allow(clippy::too_many_arguments)]
fn absorb_iota_public_writer<B: ShortBaseField>(
    level: usize,
    setup_commitment: &ShortQaFoldCommitment,
    setup_shape: ShortQaSetupShape,
    carry_commitment: Option<&ShortQaFoldCommitment>,
    carry_cols: usize,
    selection: &RowSelection,
    col_weights: &[B::Extension],
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<(), Error> {
    transcript.absorb_bytes(b"iota-level", &(level as u64).to_le_bytes());
    absorb_iota_shape_writer(setup_shape, transcript);
    transcript.absorb_digest(b"iota-setup-root", &setup_commitment.root);
    match carry_commitment {
        Some(commitment) => {
            transcript.absorb_bytes(b"iota-carry-present", &[1]);
            transcript.absorb_digest(b"iota-carry-root", &commitment.root);
        }
        None => transcript.absorb_bytes(b"iota-carry-present", &[0]),
    }
    transcript.absorb_bytes(b"iota-carry-cols", &(carry_cols as u64).to_le_bytes());
    absorb_selection_writer(selection, transcript);
    transcript.absorb_extensions(b"iota-column-weights", col_weights)
}

fn absorb_iota_shape_writer<B: ShortBaseField>(
    shape: ShortQaSetupShape,
    transcript: &mut ShortTranscriptWriter<B>,
) {
    for (label, value) in iota_shape_fields(shape) {
        transcript.absorb_bytes(label, &(value as u64).to_le_bytes());
    }
}

fn absorb_iota_shape_reader<B: ShortBaseField>(
    shape: ShortQaSetupShape,
    transcript: &mut ShortTranscriptReader<'_, B>,
) {
    for (label, value) in iota_shape_fields(shape) {
        transcript.absorb_bytes(label, &(value as u64).to_le_bytes());
    }
}

fn iota_shape_fields(shape: ShortQaSetupShape) -> [(&'static [u8], usize); 7] {
    [
        (b"iota-shape-level", shape.level),
        (b"iota-shape-encoded-rows", shape.encoded_rows),
        (b"iota-shape-message-rows", shape.message_rows),
        (b"iota-shape-t-cols", shape.t_cols),
        (b"iota-shape-t-elements", shape.t_elements),
        (b"iota-shape-t-last-len", shape.t_last_len),
        (b"iota-shape-h-cols", shape.h_cols),
    ]
}

fn absorb_selection_writer<B: ShortBaseField>(
    selection: &RowSelection,
    transcript: &mut ShortTranscriptWriter<B>,
) {
    match selection {
        RowSelection::All(count) => {
            transcript.absorb_bytes(b"selection-kind", b"all");
            transcript.absorb_bytes(b"selection-count", &(*count as u64).to_le_bytes());
        }
        RowSelection::Partial(rows) => {
            transcript.absorb_bytes(b"selection-kind", b"partial");
            transcript.absorb_bytes(b"selection-count", &(rows.len() as u64).to_le_bytes());
            for row in rows {
                transcript.absorb_bytes(b"selection-row", &(*row as u64).to_le_bytes());
            }
        }
    }
}

fn absorb_selection_reader<B: ShortBaseField>(
    selection: &RowSelection,
    transcript: &mut ShortTranscriptReader<'_, B>,
) {
    match selection {
        RowSelection::All(count) => {
            transcript.absorb_bytes(b"selection-kind", b"all");
            transcript.absorb_bytes(b"selection-count", &(*count as u64).to_le_bytes());
        }
        RowSelection::Partial(rows) => {
            transcript.absorb_bytes(b"selection-kind", b"partial");
            transcript.absorb_bytes(b"selection-count", &(rows.len() as u64).to_le_bytes());
            for row in rows {
                transcript.absorb_bytes(b"selection-row", &(*row as u64).to_le_bytes());
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn prove_short_qa_code_switch_carry<B: ShortBaseField>(
    source: &ShortQaRandomFoldableCode<B>,
    target: &ShortQaCode<B>,
    target_level: usize,
    selection: &RowSelection,
    iota: &[B::Extension],
    folded_vector: &[B::Extension],
    merged_value: B::Extension,
    merged_left_point: &[B::Extension],
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<ShortQaCodeSwitchCarry<B::Extension>, Error> {
    let message_len = source.message_len(target)?;
    let n_encoded_rows = source.codeword_len(target)?;
    validate_code_switch_statement(
        source,
        target,
        target_level,
        selection,
        iota,
        merged_left_point,
    )?;
    if folded_vector.len() != message_len {
        return Err(Error::InvalidNumberOfVariables);
    }
    absorb_code_switch_statement_writer(
        source,
        target,
        target_level,
        selection,
        iota,
        merged_value,
        merged_left_point,
        transcript,
    )?;

    let selection_log = selection.len().next_power_of_two().ilog2() as usize;
    let selection_left_point = (0..selection_log)
        .map(|_| transcript.challenge_extension(b"code-switch-selection-left"))
        .collect::<Result<Vec<_>, Error>>()?;
    let selection_left = tensor_weights_prefix(&selection_left_point, selection.len())?;
    let selected_value = inner_product(&selection_left, iota)?;
    let mut sparse_left = vec![B::Extension::zero(); n_encoded_rows];
    for (row, weight) in selection.iter().zip(&selection_left) {
        sparse_left[row] += *weight;
    }
    let generator_left = source.left_projection_extension(target, &sparse_left)?;
    let rho = transcript.challenge_extension(b"code-switch-opening-batch")?;
    let merged_left = tensor_weights(merged_left_point);
    let combined_value = selected_value + rho * merged_value;
    let combined_left = generator_left
        .into_iter()
        .zip(merged_left)
        .map(|(selected, opened)| selected + rho * opened)
        .collect::<Vec<_>>();
    let message_reduction = prove_litebullet::<B>(
        combined_value,
        combined_left,
        folded_vector.to_vec(),
        transcript,
        b"decide-carry-message-litebullet",
    )?;
    let opening_left_eval = tensor_inner_product(merged_left_point, &message_reduction.a_point)?;
    let selection_left_eval = message_reduction.a_value - rho * opening_left_eval;

    let generator_right_input = tensor_weights(&message_reduction.a_point);
    let generator_right = source.right_projection_extension(target, &generator_right_input)?;
    let generator_reduction = prove_sparse_litebullet::<B>(
        selection_left_eval,
        n_encoded_rows,
        selection,
        &selection_left,
        generator_right,
        transcript,
        b"decide-carry-generator-litebullet",
    )?;
    let (generator_claim, t_claims) = reduce_qa_rfc_generator_claim_prover(
        source,
        target,
        target_level,
        generator_reduction.b_value,
        &generator_reduction.b_point,
        &message_reduction.a_point,
        transcript,
    )?;
    Ok(ShortQaCodeSwitchCarry {
        generator_claim,
        folded_w_claim: ShortEvalWithPoint {
            value: message_reduction.b_value,
            point: message_reduction.b_point,
        },
        t_claims,
    })
}

#[allow(clippy::too_many_arguments)]
pub(super) fn verify_short_qa_code_switch_carry<B: ShortBaseField>(
    source: &ShortQaRandomFoldableCode<B>,
    target: &ShortQaCode<B>,
    target_level: usize,
    selection: &RowSelection,
    iota: &[B::Extension],
    merged_value: B::Extension,
    merged_left_point: &[B::Extension],
    transcript: &mut ShortTranscriptReader<'_, B>,
) -> Result<ShortQaCodeSwitchCarry<B::Extension>, Error> {
    let message_len = source.message_len(target)?;
    let n_encoded_rows = source.codeword_len(target)?;
    validate_code_switch_statement(
        source,
        target,
        target_level,
        selection,
        iota,
        merged_left_point,
    )?;
    absorb_code_switch_statement_reader(
        source,
        target,
        target_level,
        selection,
        iota,
        merged_value,
        merged_left_point,
        transcript,
    )?;

    let selection_log = selection.len().next_power_of_two().ilog2() as usize;
    let selection_left_point = (0..selection_log)
        .map(|_| transcript.challenge_extension(b"code-switch-selection-left"))
        .collect::<Result<Vec<_>, Error>>()?;
    let selection_left = tensor_weights_prefix(&selection_left_point, selection.len())?;
    let selected_value = inner_product(&selection_left, iota)?;
    let rho = transcript.challenge_extension(b"code-switch-opening-batch")?;
    let combined_value = selected_value + rho * merged_value;
    let message_reduction = verify_litebullet::<B>(
        combined_value,
        message_len,
        transcript,
        b"decide-carry-message-litebullet",
    )?
    .ok_or(Error::InvalidCommitment)?;
    let opening_left_eval = tensor_inner_product(merged_left_point, &message_reduction.a_point)?;
    let selection_left_eval = message_reduction.a_value - rho * opening_left_eval;
    let generator_reduction = verify_litebullet::<B>(
        selection_left_eval,
        n_encoded_rows,
        transcript,
        b"decide-carry-generator-litebullet",
    )?
    .ok_or(Error::InvalidCommitment)?;
    let expected_sparse_value = selection
        .iter()
        .zip(&selection_left)
        .map(|(row, weight)| *weight * tensor_weight_at(&generator_reduction.a_point, row))
        .fold(B::Extension::zero(), |sum, term| sum + term);
    if generator_reduction.a_value != expected_sparse_value {
        return Err(Error::InvalidCommitment);
    }
    let (generator_claim, t_claims) = reduce_qa_rfc_generator_claim_verifier(
        source,
        target,
        target_level,
        generator_reduction.b_value,
        &generator_reduction.b_point,
        &message_reduction.a_point,
        transcript,
    )?;
    Ok(ShortQaCodeSwitchCarry {
        generator_claim,
        folded_w_claim: ShortEvalWithPoint {
            value: message_reduction.b_value,
            point: message_reduction.b_point,
        },
        t_claims,
    })
}

/// Proves a code switch whose committed matrix is the direct interleaving of
/// logical rows under the next smaller QA code.
#[allow(clippy::too_many_arguments)]
pub(super) fn prove_short_qa_direct_code_switch_carry<B: ShortBaseField>(
    target: &ShortQaCode<B>,
    target_level: usize,
    row_factor: usize,
    selection: &RowSelection,
    iota: &[B::Extension],
    folded_message: &[B::Extension],
    merged_value: B::Extension,
    merged_left_point: &[B::Extension],
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<ShortQaCodeSwitchCarry<B::Extension>, Error> {
    validate_direct_code_switch_statement(target, row_factor, selection, iota, merged_left_point)?;
    let block_log = row_factor.ilog2() as usize;
    let (_, base_point) = merged_left_point.split_at(block_log);
    let message_len = target.message_len();
    if folded_message.len() != message_len {
        return Err(Error::InvalidNumberOfVariables);
    }
    if mle_eval(folded_message, base_point) != Some(merged_value) {
        return Err(Error::InvalidCommitment);
    }
    absorb_direct_code_switch_statement_writer(
        target,
        target_level,
        row_factor,
        selection,
        iota,
        merged_value,
        merged_left_point,
        transcript,
    )?;

    let selection_log = selection.len().next_power_of_two().ilog2() as usize;
    let selection_left_point = (0..selection_log)
        .map(|_| transcript.challenge_extension(b"direct-code-switch-selection-left"))
        .collect::<Result<Vec<_>, Error>>()?;
    let selection_left = tensor_weights_prefix(&selection_left_point, selection.len())?;
    let selected_value = inner_product(&selection_left, iota)?;
    let n_encoded_rows = target.codeword_len()?;
    let mut sparse_left = vec![B::Extension::zero(); n_encoded_rows];
    for (row, weight) in selection.iter().zip(&selection_left) {
        sparse_left[row] += *weight;
    }
    let generator_left = target.generator_left_projection_extension(&sparse_left)?;
    let rho = transcript.challenge_extension(b"direct-code-switch-opening-batch")?;
    let merged_left = tensor_weights(base_point);
    let combined_value = selected_value + rho * merged_value;
    let combined_left = generator_left
        .into_iter()
        .zip(merged_left)
        .map(|(selected, opened)| selected + rho * opened)
        .collect::<Vec<_>>();
    let message_reduction = prove_litebullet::<B>(
        combined_value,
        combined_left,
        folded_message.to_vec(),
        transcript,
        b"decide-direct-carry-message-litebullet",
    )?;
    let opening_left_eval = tensor_inner_product(base_point, &message_reduction.a_point)?;
    let selection_left_eval = message_reduction.a_value - rho * opening_left_eval;

    let generator_right_input = tensor_weights(&message_reduction.a_point);
    let generator_right = target.generator_right_projection_extension(&generator_right_input)?;
    let generator_reduction = prove_sparse_litebullet::<B>(
        selection_left_eval,
        n_encoded_rows,
        selection,
        &selection_left,
        generator_right,
        transcript,
        b"decide-direct-carry-generator-litebullet",
    )?;
    Ok(ShortQaCodeSwitchCarry {
        generator_claim: ShortQaMatrixClaim {
            value: generator_reduction.b_value,
            point: (generator_reduction.b_point, message_reduction.a_point),
        },
        folded_w_claim: ShortEvalWithPoint {
            value: message_reduction.b_value,
            point: message_reduction.b_point,
        },
        t_claims: Vec::new(),
    })
}

/// Reconstructs the direct-interleaved code-switch reduction without reading
/// the logical carry vector.
#[allow(clippy::too_many_arguments)]
pub(super) fn verify_short_qa_direct_code_switch_carry<B: ShortBaseField>(
    target: &ShortQaCode<B>,
    target_level: usize,
    row_factor: usize,
    selection: &RowSelection,
    iota: &[B::Extension],
    merged_value: B::Extension,
    merged_left_point: &[B::Extension],
    transcript: &mut ShortTranscriptReader<'_, B>,
) -> Result<ShortQaCodeSwitchCarry<B::Extension>, Error> {
    validate_direct_code_switch_statement(target, row_factor, selection, iota, merged_left_point)?;
    absorb_direct_code_switch_statement_reader(
        target,
        target_level,
        row_factor,
        selection,
        iota,
        merged_value,
        merged_left_point,
        transcript,
    )?;
    let block_log = row_factor.ilog2() as usize;
    let (_, base_point) = merged_left_point.split_at(block_log);
    let selection_log = selection.len().next_power_of_two().ilog2() as usize;
    let selection_left_point = (0..selection_log)
        .map(|_| transcript.challenge_extension(b"direct-code-switch-selection-left"))
        .collect::<Result<Vec<_>, Error>>()?;
    let selection_left = tensor_weights_prefix(&selection_left_point, selection.len())?;
    let selected_value = inner_product(&selection_left, iota)?;
    let rho = transcript.challenge_extension(b"direct-code-switch-opening-batch")?;
    let combined_value = selected_value + rho * merged_value;
    let message_reduction = verify_litebullet::<B>(
        combined_value,
        target.message_len(),
        transcript,
        b"decide-direct-carry-message-litebullet",
    )?
    .ok_or(Error::InvalidCommitment)?;
    let opening_left_eval = tensor_inner_product(base_point, &message_reduction.a_point)?;
    let selection_left_eval = message_reduction.a_value - rho * opening_left_eval;
    let n_encoded_rows = target.codeword_len()?;
    let generator_reduction = verify_litebullet::<B>(
        selection_left_eval,
        n_encoded_rows,
        transcript,
        b"decide-direct-carry-generator-litebullet",
    )?
    .ok_or(Error::InvalidCommitment)?;
    let expected_sparse_value = selection
        .iter()
        .zip(&selection_left)
        .map(|(row, weight)| *weight * tensor_weight_at(&generator_reduction.a_point, row))
        .fold(B::Extension::zero(), |sum, term| sum + term);
    if generator_reduction.a_value != expected_sparse_value {
        return Err(Error::InvalidCommitment);
    }
    Ok(ShortQaCodeSwitchCarry {
        generator_claim: ShortQaMatrixClaim {
            value: generator_reduction.b_value,
            point: (generator_reduction.b_point, message_reduction.a_point),
        },
        folded_w_claim: ShortEvalWithPoint {
            value: message_reduction.b_value,
            point: message_reduction.b_point,
        },
        t_claims: Vec::new(),
    })
}

fn validate_direct_code_switch_statement<B: ShortBaseField>(
    target: &ShortQaCode<B>,
    row_factor: usize,
    selection: &RowSelection,
    iota: &[B::Extension],
    merged_left_point: &[B::Extension],
) -> Result<(), Error> {
    let n_encoded_rows = target.codeword_len()?;
    validate_carry_selection(selection, n_encoded_rows)?;
    let logical_len = target
        .message_len()
        .checked_mul(row_factor)
        .ok_or(Error::EncodingError)?;
    if row_factor <= 1
        || !row_factor.is_power_of_two()
        || iota.len() != selection.len()
        || merged_left_point.len() != logical_len.ilog2() as usize
    {
        return Err(Error::InvalidNumberOfVariables);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn absorb_direct_code_switch_statement_writer<B: ShortBaseField>(
    target: &ShortQaCode<B>,
    target_level: usize,
    row_factor: usize,
    selection: &RowSelection,
    iota: &[B::Extension],
    merged_value: B::Extension,
    merged_left_point: &[B::Extension],
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<(), Error> {
    transcript.absorb_bytes(b"direct-code-switch-domain", DIRECT_CODE_SWITCH_DOMAIN);
    absorb_direct_code_switch_public_writer(
        target,
        target_level,
        row_factor,
        selection,
        iota,
        merged_value,
        merged_left_point,
        transcript,
    )
}

#[allow(clippy::too_many_arguments)]
fn absorb_direct_code_switch_statement_reader<B: ShortBaseField>(
    target: &ShortQaCode<B>,
    target_level: usize,
    row_factor: usize,
    selection: &RowSelection,
    iota: &[B::Extension],
    merged_value: B::Extension,
    merged_left_point: &[B::Extension],
    transcript: &mut ShortTranscriptReader<'_, B>,
) -> Result<(), Error> {
    transcript.absorb_bytes(b"direct-code-switch-domain", DIRECT_CODE_SWITCH_DOMAIN);
    transcript.absorb_bytes(
        b"direct-code-switch-target-level",
        &(target_level as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"direct-code-switch-row-factor",
        &(row_factor as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"direct-code-switch-message-len",
        &(target.message_len() as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"direct-code-switch-encoded-len",
        &(target.codeword_len()? as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"direct-code-switch-target-index",
        &(target.index() as u64).to_le_bytes(),
    );
    transcript.absorb_digest(
        b"direct-code-switch-target-generator",
        &target.generator_digest(),
    );
    absorb_selection_reader(selection, transcript);
    transcript.absorb_extensions(b"direct-code-switch-iota", iota)?;
    transcript.absorb_extension(b"direct-code-switch-merged-value", merged_value)?;
    transcript.absorb_extensions(b"direct-code-switch-merged-left-point", merged_left_point)
}

#[allow(clippy::too_many_arguments)]
fn absorb_direct_code_switch_public_writer<B: ShortBaseField>(
    target: &ShortQaCode<B>,
    target_level: usize,
    row_factor: usize,
    selection: &RowSelection,
    iota: &[B::Extension],
    merged_value: B::Extension,
    merged_left_point: &[B::Extension],
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<(), Error> {
    transcript.absorb_bytes(
        b"direct-code-switch-target-level",
        &(target_level as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"direct-code-switch-row-factor",
        &(row_factor as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"direct-code-switch-message-len",
        &(target.message_len() as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"direct-code-switch-encoded-len",
        &(target.codeword_len()? as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"direct-code-switch-target-index",
        &(target.index() as u64).to_le_bytes(),
    );
    transcript.absorb_digest(
        b"direct-code-switch-target-generator",
        &target.generator_digest(),
    );
    absorb_selection_writer(selection, transcript);
    transcript.absorb_extensions(b"direct-code-switch-iota", iota)?;
    transcript.absorb_extension(b"direct-code-switch-merged-value", merged_value)?;
    transcript.absorb_extensions(b"direct-code-switch-merged-left-point", merged_left_point)
}

fn validate_code_switch_statement<B: ShortBaseField>(
    source: &ShortQaRandomFoldableCode<B>,
    target: &ShortQaCode<B>,
    target_level: usize,
    selection: &RowSelection,
    iota: &[B::Extension],
    merged_left_point: &[B::Extension],
) -> Result<(), Error> {
    let n_encoded_rows = source.codeword_len(target)?;
    validate_carry_selection(selection, n_encoded_rows)?;
    if source.target_level() != target_level
        || source.layers() == 0
        || iota.len() != selection.len()
        || merged_left_point.len() != source.message_len(target)?.ilog2() as usize
    {
        return Err(Error::InvalidNumberOfVariables);
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn absorb_code_switch_statement_writer<B: ShortBaseField>(
    source: &ShortQaRandomFoldableCode<B>,
    target: &ShortQaCode<B>,
    target_level: usize,
    selection: &RowSelection,
    iota: &[B::Extension],
    merged_value: B::Extension,
    merged_left_point: &[B::Extension],
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<(), Error> {
    transcript.absorb_bytes(b"code-switch-domain", CODE_SWITCH_DOMAIN);
    transcript.absorb_bytes(
        b"code-switch-target-level",
        &(target_level as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"code-switch-source-layers",
        &(source.layers() as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"code-switch-message-len",
        &(source.message_len(target)? as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"code-switch-encoded-len",
        &(source.codeword_len(target)? as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"code-switch-target-message-len",
        &(target.message_len() as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"code-switch-target-index",
        &(target.index() as u64).to_le_bytes(),
    );
    transcript.absorb_digest(b"code-switch-target-generator", &target.generator_digest());
    absorb_selection_writer(selection, transcript);
    transcript.absorb_extensions(b"code-switch-iota", iota)?;
    transcript.absorb_extension(b"code-switch-merged-value", merged_value)?;
    transcript.absorb_extensions(b"code-switch-merged-left-point", merged_left_point)
}

#[allow(clippy::too_many_arguments)]
fn absorb_code_switch_statement_reader<B: ShortBaseField>(
    source: &ShortQaRandomFoldableCode<B>,
    target: &ShortQaCode<B>,
    target_level: usize,
    selection: &RowSelection,
    iota: &[B::Extension],
    merged_value: B::Extension,
    merged_left_point: &[B::Extension],
    transcript: &mut ShortTranscriptReader<'_, B>,
) -> Result<(), Error> {
    transcript.absorb_bytes(b"code-switch-domain", CODE_SWITCH_DOMAIN);
    transcript.absorb_bytes(
        b"code-switch-target-level",
        &(target_level as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"code-switch-source-layers",
        &(source.layers() as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"code-switch-message-len",
        &(source.message_len(target)? as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"code-switch-encoded-len",
        &(source.codeword_len(target)? as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"code-switch-target-message-len",
        &(target.message_len() as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(
        b"code-switch-target-index",
        &(target.index() as u64).to_le_bytes(),
    );
    transcript.absorb_digest(b"code-switch-target-generator", &target.generator_digest());
    absorb_selection_reader(selection, transcript);
    transcript.absorb_extensions(b"code-switch-iota", iota)?;
    transcript.absorb_extension(b"code-switch-merged-value", merged_value)?;
    transcript.absorb_extensions(b"code-switch-merged-left-point", merged_left_point)
}

fn validate_carry_selection(selection: &RowSelection, n_encoded_rows: usize) -> Result<(), Error> {
    if n_encoded_rows == 0 || !n_encoded_rows.is_power_of_two() || selection.len() == 0 {
        return Err(Error::InvalidNumberOfVariables);
    }
    match selection {
        RowSelection::All(count) if *count == n_encoded_rows => Ok(()),
        RowSelection::Partial(rows) if rows.len() <= n_encoded_rows => {
            let mut sorted = rows.clone();
            sorted.sort_unstable();
            if sorted.last().is_some_and(|row| *row >= n_encoded_rows)
                || sorted.windows(2).any(|pair| pair[0] == pair[1])
            {
                return Err(Error::InvalidNumberOfVariables);
            }
            Ok(())
        }
        _ => Err(Error::InvalidNumberOfVariables),
    }
}
