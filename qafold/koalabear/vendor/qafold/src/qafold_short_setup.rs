//! Base-field setup commitments for the short-prime QAFold Decide protocol.
//!
//! This module is intended to be included as a child of `qafold_short_pc`.
//! Fixed RF diagonals and QA alpha columns stay in the short base field. Their
//! setup codeword is therefore hashed with canonical base-field encodings,
//! while later carry commitments may use the scalar-extended generator over
//! the extension field.

use super::*;

use crate::qafold_short::{ShortQaDecideSetupMode, ShortQaRandomFoldableCode};

const SETUP_LEAF_DOMAIN: &[u8] = b"qafold-short/setup-base-row/v1";
const SETUP_AGGREGATE_DOMAIN: &[u8] = b"qafold-short/setup-aggregate/v1";
const DERIVED_PARAMETERS_DOMAIN: &[u8] = b"qafold-short/derived-rfc-parameters/v1";

/// Shape metadata bound into every merged `T_j || H_j` setup leaf.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShortQaSetupShape {
    /// Recursive QA level represented by this setup commitment.
    pub level: usize,
    /// Number of encoded rows in the derived RFC codeword.
    pub encoded_rows: usize,
    /// Number of logical rows before derived-code encoding.
    pub message_rows: usize,
    /// Number of packed incoming RF-diagonal columns.
    pub t_cols: usize,
    /// Number of real incoming RF-diagonal elements before column packing.
    pub t_elements: usize,
    /// Logical length of the final T column, or zero when there are no T columns.
    pub t_last_len: usize,
    /// Number of fixed QA alpha columns.
    pub h_cols: usize,
}

impl ShortQaSetupShape {
    pub(super) fn total_cols(self) -> Result<usize, Error> {
        self.t_cols
            .checked_add(self.h_cols)
            .ok_or(Error::EncodingError)
    }

    pub(super) fn validate(self) -> Result<(), Error> {
        if self.level == 0
            || self.encoded_rows == 0
            || !self.encoded_rows.is_power_of_two()
            || self.message_rows == 0
            || !self.message_rows.is_power_of_two()
            || self.encoded_rows < self.message_rows
            || self.encoded_rows % self.message_rows != 0
            || self.h_cols == 0
            || !self.h_cols.is_power_of_two()
        {
            return Err(Error::EncodingError);
        }
        if self.t_cols == 0 {
            if self.t_elements != 0 || self.t_last_len != 0 {
                return Err(Error::EncodingError);
            }
        } else {
            if self.t_last_len == 0 || self.t_last_len > self.message_rows {
                return Err(Error::EncodingError);
            }
            let expected_elements = self
                .t_cols
                .checked_sub(1)
                .and_then(|full_columns| full_columns.checked_mul(self.message_rows))
                .and_then(|elements| elements.checked_add(self.t_last_len))
                .ok_or(Error::EncodingError)?;
            if self.t_elements != expected_elements {
                return Err(Error::EncodingError);
            }
        }
        self.total_cols().map(|_| ())
    }
}

/// Base-field RFC sources sampled for every non-base QA level.
#[derive(Clone, Debug)]
pub struct ShortQaDerivedParameters<B: ShortBaseField> {
    /// Level-indexed source codes; level zero has no derived source.
    codes: Vec<Option<ShortQaRandomFoldableCode<B>>>,
    /// Digest binding the selected field policy and every RFC source, if any.
    digest: [u8; DIGEST_BYTES],
}

impl<B: ShortBaseField> ShortQaDerivedParameters<B> {
    /// Returns whether this setup uses direct lower-code interleaving.
    pub(super) fn is_direct_interleaved(&self) -> bool {
        B::DECIDE_SETUP_MODE == ShortQaDecideSetupMode::DirectInterleavedV1
    }

    /// Returns the number of QA levels covered by the derived parameters.
    pub fn levels(&self) -> usize {
        self.len()
    }

    /// Returns the number of level-indexed derived-code slots.
    pub fn len(&self) -> usize {
        self.codes.len()
    }

    /// Returns whether the derived-code sequence has no level slots.
    pub fn is_empty(&self) -> bool {
        self.codes.is_empty()
    }

    /// Iterates over the level-indexed derived sources without mutable access.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = Option<&ShortQaRandomFoldableCode<B>>> {
        self.codes.iter().map(Option::as_ref)
    }

    /// Returns the digest binding the ordered derived-code sequence.
    pub fn digest(&self) -> [u8; DIGEST_BYTES] {
        self.digest
    }

    /// Returns the derived source for one non-base level.
    pub fn code(&self, level: usize) -> Result<&ShortQaRandomFoldableCode<B>, Error> {
        self.codes
            .get(level)
            .and_then(Option::as_ref)
            .ok_or(Error::EncodingError)
    }
}

/// Prover cache for one base-field merged setup commitment.
pub struct ShortQaFoldSetupCache<B: ShortBaseField> {
    /// Complete column-major base-field codeword.
    pub(super) encoded_entries: Vec<B>,
    /// Public shape bound by the merged leaf hashes.
    pub(super) shape: ShortQaSetupShape,
    /// Merkle tree over merged base-field rows.
    pub(super) tree: Blake2sMerkleTree,
}

/// One setup-phase commitment to base-field setup columns.
pub struct ShortQaFoldSetupLevel<B: ShortBaseField> {
    /// Merkle root of the merged setup codeword rows.
    pub commitment: ShortQaFoldCommitment,
    /// Incoming RF diagonals packed by descending layer and ascending chunk.
    ///
    /// Direct-interleaved setup has no RFC diagonals and stores this as empty.
    pub t_columns: Vec<Vec<B>>,
    /// Shape bound into every leaf.
    pub shape: ShortQaSetupShape,
    /// Prover-side encoded codeword and Merkle tree.
    pub(super) cache: ShortQaFoldSetupCache<B>,
}

/// Setup commitments indexed by QA generator level.
pub struct ShortQaFoldSetupCommitments<B: ShortBaseField> {
    levels: Vec<Option<ShortQaFoldSetupLevel<B>>>,
    setup_digest: [u8; DIGEST_BYTES],
}

impl<B: ShortBaseField> ShortQaFoldSetupCommitments<B> {
    /// Returns the setup commitment at one non-base level.
    pub fn level(&self, level: usize) -> Result<&ShortQaFoldSetupLevel<B>, Error> {
        self.levels
            .get(level)
            .and_then(Option::as_ref)
            .ok_or(Error::InvalidCommitment)
    }

    /// Returns the number of QA levels represented by this setup object.
    pub fn levels(&self) -> usize {
        self.levels.len()
    }

    /// Returns the digest binding all ordered setup roots and shapes.
    pub fn setup_digest(&self) -> [u8; DIGEST_BYTES] {
        self.setup_digest
    }
}

/// One independently authenticated row of a merged base-field setup codeword.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortQaSetupRowQuery<B: ShortBaseField> {
    /// Fiat--Shamir-selected encoded row index.
    pub row_index: usize,
    /// Merged row in `T`-then-`H` column order.
    pub row: Vec<B>,
    /// Authentication path under the setup root.
    pub path: ShortQaMerklePath,
}

/// Partial or compact all-row authentication for one setup codeword.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShortQaSetupRows<B: ShortBaseField> {
    /// Independently authenticated selected rows.
    Partial(Vec<ShortQaSetupRowQuery<B>>),
    /// Complete column-major codeword used when every row must be queried.
    AllRows {
        /// Complete base-field codeword, without extension-field lifting.
        encoded_entries: Vec<B>,
        /// Shape that determines the merged leaf encoding.
        shape: ShortQaSetupShape,
    },
}

/// Samples base-field RFC sources when the field policy uses RFC setup.
pub(super) fn sample_short_derived_parameters<B, R>(
    sequence: &ShortQaFold<B>,
    rng: &mut R,
) -> Result<ShortQaDerivedParameters<B>, Error>
where
    B: ShortBaseField,
    R: RngCore,
{
    let levels = sequence.codes().len();
    if levels == 0 {
        return Err(Error::EncodingError);
    }
    if B::DECIDE_SETUP_MODE == ShortQaDecideSetupMode::DirectInterleavedV1 {
        let codes = core::iter::repeat_with(|| None)
            .take(levels)
            .collect::<Vec<_>>();
        let digest = hash_short_derived_parameters(sequence, &codes)?;
        return Ok(ShortQaDerivedParameters { codes, digest });
    }
    let mut codes = Vec::with_capacity(levels);
    codes.push(None);
    for level in 1..levels {
        let target = sequence
            .codes()
            .get(level - 1)
            .ok_or(Error::EncodingError)?;
        let current = sequence.codes().get(level).ok_or(Error::EncodingError)?;
        let source = ShortQaRandomFoldableCode::sample(
            level - 1,
            target,
            sequence.switch_ratio(),
            &mut *rng,
        )?;
        if source.message_len(target)? != current.message_len() {
            return Err(Error::EncodingError);
        }
        codes.push(Some(source));
    }
    let digest = hash_short_derived_parameters(sequence, &codes)?;
    Ok(ShortQaDerivedParameters { codes, digest })
}

/// Packs diagonals from level `j + 1` into the base-field `T_j` columns.
///
/// Layers are ordered from the largest diagonal to the smallest. Chunks within
/// each layer retain their natural order, and only the final short chunk is
/// logically padded with base-field zeros.
pub(super) fn incoming_short_t_columns<B: ShortBaseField>(
    sequence: &ShortQaFold<B>,
    derived: &ShortQaDerivedParameters<B>,
    level: usize,
) -> Result<Vec<Vec<B>>, Error> {
    let levels = sequence.codes().len();
    if level == 0 || level >= levels || derived.codes.len() != levels {
        return Err(Error::EncodingError);
    }
    if level + 1 == levels {
        return Ok(Vec::new());
    }
    let rows = sequence
        .codes()
        .get(level)
        .ok_or(Error::EncodingError)?
        .message_len();
    let incoming = derived.code(level + 1)?;
    let diagonals = (1..=incoming.layers())
        .rev()
        .map(|layer| incoming.diagonal(layer))
        .collect::<Result<Vec<_>, Error>>()?;
    pack_short_t_columns(rows, &diagonals)
}

fn pack_short_t_columns<B: ShortBaseField>(
    rows: usize,
    diagonals: &[&[B]],
) -> Result<Vec<Vec<B>>, Error> {
    if rows == 0 {
        return Err(Error::EncodingError);
    }
    let total_elements = diagonals.iter().try_fold(0usize, |total, diagonal| {
        total
            .checked_add(diagonal.len())
            .ok_or(Error::EncodingError)
    })?;
    let mut columns = Vec::with_capacity(total_elements.div_ceil(rows));
    let mut current = Vec::with_capacity(rows);
    for diagonal in diagonals {
        for value in *diagonal {
            current.push(*value);
            if current.len() == rows {
                columns.push(core::mem::replace(&mut current, Vec::with_capacity(rows)));
            }
        }
    }
    if !current.is_empty() {
        columns.push(current);
    }
    Ok(columns)
}

/// Encodes and commits every fixed base-field setup matrix.
pub(super) fn build_short_setup_commitments<B: ShortBaseField>(
    sequence: &ShortQaFold<B>,
    derived: &ShortQaDerivedParameters<B>,
) -> Result<ShortQaFoldSetupCommitments<B>, Error> {
    match B::DECIDE_SETUP_MODE {
        ShortQaDecideSetupMode::RfV1 => build_rf_short_setup_commitments(sequence, derived),
        ShortQaDecideSetupMode::DirectInterleavedV1 => {
            build_direct_interleaved_short_setup_commitments(sequence, derived)
        }
    }
}

/// Encodes and commits the legacy RFC `T_j || H_j` setup matrices.
fn build_rf_short_setup_commitments<B: ShortBaseField>(
    sequence: &ShortQaFold<B>,
    derived: &ShortQaDerivedParameters<B>,
) -> Result<ShortQaFoldSetupCommitments<B>, Error> {
    let level_count = sequence.codes().len();
    if level_count == 0 || derived.codes.len() != level_count {
        return Err(Error::EncodingError);
    }
    let mut levels = core::iter::repeat_with(|| None)
        .take(level_count)
        .collect::<Vec<_>>();
    for level in 1..level_count {
        let target = sequence
            .codes()
            .get(level - 1)
            .ok_or(Error::EncodingError)?;
        let current = sequence.codes().get(level).ok_or(Error::EncodingError)?;
        let source = derived.code(level)?;
        let t_columns = incoming_short_t_columns(sequence, derived, level)?;
        let h_columns = current.alphas();
        let message_rows = current.message_len();
        if source.target_level() != level - 1
            || source.message_len(target)? != message_rows
            || t_columns.iter().enumerate().any(|(column, values)| {
                values.is_empty()
                    || values.len() > message_rows
                    || (column + 1 != t_columns.len() && values.len() != message_rows)
            })
            || h_columns.iter().any(|column| column.len() != message_rows)
        {
            return Err(Error::EncodingError);
        }

        let encoded_rows = source.codeword_len(target)?;
        let t_elements = t_columns.iter().try_fold(0usize, |total, column| {
            total.checked_add(column.len()).ok_or(Error::EncodingError)
        })?;
        let shape = ShortQaSetupShape {
            level,
            encoded_rows,
            message_rows,
            t_cols: t_columns.len(),
            t_elements,
            t_last_len: t_columns.last().map_or(0, Vec::len),
            h_cols: h_columns.len(),
        };
        shape.validate()?;
        let total_cols = shape.total_cols()?;
        let encoded_len = encoded_rows
            .checked_mul(total_cols)
            .ok_or(Error::EncodingError)?;
        let mut encoded_entries = vec![B::zero(); encoded_len];

        #[cfg(feature = "parallel")]
        encoded_entries
            .par_chunks_mut(encoded_rows)
            .enumerate()
            .try_for_each(|(column, output)| {
                let message = if column < shape.t_cols {
                    t_columns[column].as_slice()
                } else {
                    h_columns[column - shape.t_cols].as_slice()
                };
                if column < shape.t_cols {
                    source.encode_base_implicit_zero_into(target, message, output)
                } else {
                    source.encode_base_into(target, message, output)
                }
            })?;
        #[cfg(not(feature = "parallel"))]
        encoded_entries
            .chunks_mut(encoded_rows)
            .enumerate()
            .try_for_each(|(column, output)| {
                let message = if column < shape.t_cols {
                    t_columns[column].as_slice()
                } else {
                    h_columns[column - shape.t_cols].as_slice()
                };
                if column < shape.t_cols {
                    source.encode_base_implicit_zero_into(target, message, output)
                } else {
                    source.encode_base_into(target, message, output)
                }
            })?;

        let tree = build_short_setup_tree::<B>(&encoded_entries, shape)?;
        let commitment = ShortQaFoldCommitment { root: tree.root() };
        levels[level] = Some(ShortQaFoldSetupLevel {
            commitment,
            t_columns,
            shape,
            cache: ShortQaFoldSetupCache {
                encoded_entries,
                shape,
                tree,
            },
        });
    }
    let setup_digest = hash_short_setup_commitments::<B>(&levels)?;
    Ok(ShortQaFoldSetupCommitments {
        levels,
        setup_digest,
    })
}

/// Encodes `H_j` under `G_{j-1}` after virtually interleaving its row blocks.
///
/// Each physical column is one contiguous `k_{j-1}`-element slice of an
/// alpha column, so this never materializes a reshaped `H_j` matrix.
fn build_direct_interleaved_short_setup_commitments<B: ShortBaseField>(
    sequence: &ShortQaFold<B>,
    derived: &ShortQaDerivedParameters<B>,
) -> Result<ShortQaFoldSetupCommitments<B>, Error> {
    let level_count = sequence.codes().len();
    if level_count == 0
        || derived.codes.len() != level_count
        || derived.codes.iter().any(Option::is_some)
    {
        return Err(Error::EncodingError);
    }
    let row_factor = sequence.switch_ratio();
    if row_factor == 0 || !row_factor.is_power_of_two() {
        return Err(Error::EncodingError);
    }
    let mut levels = core::iter::repeat_with(|| None)
        .take(level_count)
        .collect::<Vec<_>>();
    for level in 1..level_count {
        let target = sequence
            .codes()
            .get(level - 1)
            .ok_or(Error::EncodingError)?;
        let current = sequence.codes().get(level).ok_or(Error::EncodingError)?;
        let message_rows = target.message_len();
        let encoded_rows = target.codeword_len()?;
        let h_columns = current.alphas();
        if current.message_len()
            != message_rows
                .checked_mul(row_factor)
                .ok_or(Error::EncodingError)?
            || h_columns.is_empty()
            || h_columns
                .iter()
                .any(|column| column.len() != current.message_len())
        {
            return Err(Error::EncodingError);
        }
        let h_cols = h_columns
            .len()
            .checked_mul(row_factor)
            .ok_or(Error::EncodingError)?;
        let shape = ShortQaSetupShape {
            level,
            encoded_rows,
            message_rows,
            t_cols: 0,
            t_elements: 0,
            t_last_len: 0,
            h_cols,
        };
        shape.validate()?;
        let encoded_len = encoded_rows
            .checked_mul(h_cols)
            .ok_or(Error::EncodingError)?;
        let mut encoded_entries = vec![B::zero(); encoded_len];

        #[cfg(feature = "parallel")]
        encoded_entries
            .par_chunks_mut(encoded_rows)
            .enumerate()
            .try_for_each(|(column, output)| {
                let alpha = h_columns
                    .get(column / row_factor)
                    .ok_or(Error::EncodingError)?;
                let block = column % row_factor;
                let start = block
                    .checked_mul(message_rows)
                    .ok_or(Error::EncodingError)?;
                let end = start
                    .checked_add(message_rows)
                    .ok_or(Error::EncodingError)?;
                target.encode_base_into(&alpha[start..end], output)
            })?;
        #[cfg(not(feature = "parallel"))]
        encoded_entries
            .chunks_mut(encoded_rows)
            .enumerate()
            .try_for_each(|(column, output)| {
                let alpha = h_columns
                    .get(column / row_factor)
                    .ok_or(Error::EncodingError)?;
                let block = column % row_factor;
                let start = block
                    .checked_mul(message_rows)
                    .ok_or(Error::EncodingError)?;
                let end = start
                    .checked_add(message_rows)
                    .ok_or(Error::EncodingError)?;
                target.encode_base_into(&alpha[start..end], output)
            })?;

        let tree = build_short_setup_tree::<B>(&encoded_entries, shape)?;
        let commitment = ShortQaFoldCommitment { root: tree.root() };
        levels[level] = Some(ShortQaFoldSetupLevel {
            commitment,
            t_columns: Vec::new(),
            shape,
            cache: ShortQaFoldSetupCache {
                encoded_entries,
                shape,
                tree,
            },
        });
    }
    let setup_digest = hash_short_setup_commitments::<B>(&levels)?;
    Ok(ShortQaFoldSetupCommitments {
        levels,
        setup_digest,
    })
}

/// Opens selected setup rows, using one compact payload when every row is read.
pub(super) fn open_short_setup_rows<B: ShortBaseField>(
    cache: &ShortQaFoldSetupCache<B>,
    selection: &RowSelection,
) -> Result<ShortQaSetupRows<B>, Error> {
    cache.shape.validate()?;
    if selection.len() == 0 || selection.len() > cache.shape.encoded_rows {
        return Err(Error::InvalidNumberOfVariables);
    }
    if selection.is_all() {
        if selection.len() != cache.shape.encoded_rows {
            return Err(Error::InvalidNumberOfVariables);
        }
        return Ok(ShortQaSetupRows::AllRows {
            encoded_entries: cache.encoded_entries.clone(),
            shape: cache.shape,
        });
    }

    let build_query = |row_index: usize| {
        if row_index >= cache.shape.encoded_rows {
            return Err(Error::InvalidNumberOfVariables);
        }
        Ok(ShortQaSetupRowQuery {
            row_index,
            row: short_setup_encoded_row(&cache.encoded_entries, cache.shape, row_index)?,
            path: cache.tree.path(row_index)?,
        })
    };
    #[cfg(feature = "parallel")]
    let queries = selection
        .iter()
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(build_query)
        .collect::<Result<Vec<_>, Error>>()?;
    #[cfg(not(feature = "parallel"))]
    let queries = selection
        .iter()
        .map(build_query)
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(ShortQaSetupRows::Partial(queries))
}

/// Verifies partial paths or reconstructs the root of an all-row setup payload.
pub(super) fn verify_short_setup_rows<B: ShortBaseField>(
    commitment: &ShortQaFoldCommitment,
    expected_shape: ShortQaSetupShape,
    selection: &RowSelection,
    rows: &ShortQaSetupRows<B>,
) -> Result<(), Error> {
    expected_shape.validate()?;
    if selection.len() == 0 || selection.len() > expected_shape.encoded_rows {
        return Err(Error::InvalidNumberOfVariables);
    }
    match (selection, rows) {
        (
            RowSelection::All(count),
            ShortQaSetupRows::AllRows {
                encoded_entries,
                shape,
            },
        ) => {
            if *count != expected_shape.encoded_rows || *shape != expected_shape {
                return Err(Error::InvalidCommitment);
            }
            let expected_len = expected_shape
                .encoded_rows
                .checked_mul(expected_shape.total_cols()?)
                .ok_or(Error::EncodingError)?;
            if encoded_entries.len() != expected_len
                || build_short_setup_tree::<B>(encoded_entries, expected_shape)?.root()
                    != commitment.root
            {
                return Err(Error::InvalidCommitment);
            }
            Ok(())
        }
        (RowSelection::Partial(expected_rows), ShortQaSetupRows::Partial(queries)) => {
            if queries.len() != expected_rows.len() {
                return Err(Error::InvalidCommitment);
            }
            let verify = |(query, expected_row): (&ShortQaSetupRowQuery<B>, &usize)| {
                if query.row_index != *expected_row
                    || query.row.len() != expected_shape.total_cols()?
                {
                    return Err(Error::InvalidCommitment);
                }
                let leaf = hash_short_setup_row::<B>(&query.row, expected_shape, query.row_index)?;
                if !verify_merkle_path(
                    leaf,
                    query.row_index,
                    expected_shape.encoded_rows,
                    &query.path,
                    &commitment.root,
                ) {
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

fn hash_short_derived_parameters<B: ShortBaseField>(
    sequence: &ShortQaFold<B>,
    codes: &[Option<ShortQaRandomFoldableCode<B>>],
) -> Result<[u8; DIGEST_BYTES], Error> {
    if codes.len() != sequence.codes().len()
        || codes.first().map_or(true, Option::is_some)
        || (B::DECIDE_SETUP_MODE == ShortQaDecideSetupMode::DirectInterleavedV1
            && codes.iter().any(Option::is_some))
    {
        return Err(Error::EncodingError);
    }
    let mut hasher = Blake2s256::new();
    absorb_framed(&mut hasher, b"domain", DERIVED_PARAMETERS_DOMAIN);
    absorb_short_field_metadata::<B>(&mut hasher);
    if B::DECIDE_SETUP_MODE == ShortQaDecideSetupMode::DirectInterleavedV1 {
        absorb_framed(
            &mut hasher,
            b"decide-setup-mode",
            B::DECIDE_SETUP_MODE.domain_label(),
        );
    }
    absorb_usize(&mut hasher, b"levels", codes.len())?;
    absorb_usize(&mut hasher, b"switch-ratio", sequence.switch_ratio())?;
    for level in 1..codes.len() {
        absorb_usize(&mut hasher, b"level", level)?;
        match (B::DECIDE_SETUP_MODE, codes[level].as_ref()) {
            (ShortQaDecideSetupMode::RfV1, Some(source)) => {
                let target = sequence
                    .codes()
                    .get(level - 1)
                    .ok_or(Error::EncodingError)?;
                absorb_usize(&mut hasher, b"target-level", source.target_level())?;
                absorb_usize(&mut hasher, b"message-length", source.message_len(target)?)?;
                absorb_usize(
                    &mut hasher,
                    b"codeword-length",
                    source.codeword_len(target)?,
                )?;
                absorb_usize(&mut hasher, b"layers", source.layers())?;
                let mut encoded = vec![0u8; B::CANONICAL_BYTES];
                for layer in 1..=source.layers() {
                    let diagonal = source.diagonal(layer)?;
                    absorb_usize(&mut hasher, b"layer", layer)?;
                    absorb_usize(&mut hasher, b"diagonal-length", diagonal.len())?;
                    for value in diagonal {
                        value.write_canonical_bytes(&mut encoded)?;
                        hasher.update(&encoded);
                    }
                }
            }
            (ShortQaDecideSetupMode::DirectInterleavedV1, None) => {
                absorb_framed(&mut hasher, b"rfc-source-present", &[0]);
            }
            _ => return Err(Error::EncodingError),
        }
    }
    Ok(hasher.finalize().into())
}

fn hash_short_setup_commitments<B: ShortBaseField>(
    levels: &[Option<ShortQaFoldSetupLevel<B>>],
) -> Result<[u8; DIGEST_BYTES], Error> {
    if levels.is_empty() || levels[0].is_some() {
        return Err(Error::EncodingError);
    }
    let mut hasher = Blake2s256::new();
    absorb_framed(&mut hasher, b"domain", SETUP_AGGREGATE_DOMAIN);
    absorb_short_field_metadata::<B>(&mut hasher);
    absorb_usize(&mut hasher, b"levels", levels.len())?;
    for (level, setup_level) in levels.iter().enumerate() {
        absorb_usize(&mut hasher, b"slot", level)?;
        match setup_level {
            None if level == 0 => absorb_framed(&mut hasher, b"present", &[0]),
            Some(setup_level) if level > 0 => {
                if setup_level.shape.level != level {
                    return Err(Error::EncodingError);
                }
                setup_level.shape.validate()?;
                absorb_framed(&mut hasher, b"present", &[1]);
                absorb_short_setup_shape(&mut hasher, setup_level.shape)?;
                absorb_framed(&mut hasher, b"root", &setup_level.commitment.root);
            }
            _ => return Err(Error::EncodingError),
        }
    }
    Ok(hasher.finalize().into())
}

fn build_short_setup_tree<B: ShortBaseField>(
    encoded_entries: &[B],
    shape: ShortQaSetupShape,
) -> Result<Blake2sMerkleTree, Error> {
    shape.validate()?;
    let expected_len = shape
        .encoded_rows
        .checked_mul(shape.total_cols()?)
        .ok_or(Error::EncodingError)?;
    if encoded_entries.len() != expected_len {
        return Err(Error::EncodingError);
    }
    #[cfg(feature = "parallel")]
    let leaves = (0..shape.encoded_rows)
        .into_par_iter()
        .map(|row| hash_short_setup_encoded_row::<B>(encoded_entries, shape, row))
        .collect::<Result<Vec<_>, Error>>()?;
    #[cfg(not(feature = "parallel"))]
    let leaves = (0..shape.encoded_rows)
        .map(|row| hash_short_setup_encoded_row::<B>(encoded_entries, shape, row))
        .collect::<Result<Vec<_>, Error>>()?;
    Blake2sMerkleTree::from_leaves(leaves)
}

fn hash_short_setup_encoded_row<B: ShortBaseField>(
    encoded_entries: &[B],
    shape: ShortQaSetupShape,
    row_index: usize,
) -> Result<[u8; DIGEST_BYTES], Error> {
    shape.validate()?;
    let total_cols = shape.total_cols()?;
    let expected_len = shape
        .encoded_rows
        .checked_mul(total_cols)
        .ok_or(Error::EncodingError)?;
    if row_index >= shape.encoded_rows || encoded_entries.len() != expected_len {
        return Err(Error::EncodingError);
    }
    let mut hasher = short_setup_row_hasher::<B>(shape, row_index)?;
    let mut encoded = [0u8; 16];
    if B::CANONICAL_BYTES > encoded.len() {
        return Err(Error::EncodingError);
    }
    for column in 0..total_cols {
        encoded_entries[column * shape.encoded_rows + row_index]
            .write_canonical_bytes(&mut encoded[..B::CANONICAL_BYTES])?;
        hasher.update(&encoded[..B::CANONICAL_BYTES]);
    }
    Ok(hasher.finalize().into())
}

fn hash_short_setup_row<B: ShortBaseField>(
    row: &[B],
    shape: ShortQaSetupShape,
    row_index: usize,
) -> Result<[u8; DIGEST_BYTES], Error> {
    shape.validate()?;
    if row_index >= shape.encoded_rows || row.len() != shape.total_cols()? {
        return Err(Error::EncodingError);
    }
    let mut hasher = short_setup_row_hasher::<B>(shape, row_index)?;
    let mut encoded = [0u8; 16];
    if B::CANONICAL_BYTES > encoded.len() {
        return Err(Error::EncodingError);
    }
    for value in row {
        value.write_canonical_bytes(&mut encoded[..B::CANONICAL_BYTES])?;
        hasher.update(&encoded[..B::CANONICAL_BYTES]);
    }
    Ok(hasher.finalize().into())
}

fn short_setup_row_hasher<B: ShortBaseField>(
    shape: ShortQaSetupShape,
    row_index: usize,
) -> Result<Blake2s256, Error> {
    shape.validate()?;
    if row_index >= shape.encoded_rows {
        return Err(Error::EncodingError);
    }
    let mut hasher = Blake2s256::new();
    absorb_framed(&mut hasher, b"domain", SETUP_LEAF_DOMAIN);
    absorb_short_field_metadata::<B>(&mut hasher);
    if B::DECIDE_SETUP_MODE == ShortQaDecideSetupMode::DirectInterleavedV1 {
        absorb_framed(
            &mut hasher,
            b"decide-setup-mode",
            B::DECIDE_SETUP_MODE.domain_label(),
        );
    }
    absorb_short_setup_shape(&mut hasher, shape)?;
    absorb_usize(&mut hasher, b"row", row_index)?;
    Ok(hasher)
}

fn absorb_short_setup_shape(
    hasher: &mut Blake2s256,
    shape: ShortQaSetupShape,
) -> Result<(), Error> {
    shape.validate()?;
    for (label, value) in [
        (b"level".as_slice(), shape.level),
        (b"encoded-rows".as_slice(), shape.encoded_rows),
        (b"message-rows".as_slice(), shape.message_rows),
        (b"t-columns".as_slice(), shape.t_cols),
        (b"t-elements".as_slice(), shape.t_elements),
        (b"t-last-length".as_slice(), shape.t_last_len),
        (b"h-columns".as_slice(), shape.h_cols),
    ] {
        absorb_usize(hasher, label, value)?;
    }
    Ok(())
}

fn short_setup_encoded_row<B: ShortBaseField>(
    encoded_entries: &[B],
    shape: ShortQaSetupShape,
    row_index: usize,
) -> Result<Vec<B>, Error> {
    shape.validate()?;
    let total_cols = shape.total_cols()?;
    if row_index >= shape.encoded_rows
        || encoded_entries.len()
            != shape
                .encoded_rows
                .checked_mul(total_cols)
                .ok_or(Error::EncodingError)?
    {
        return Err(Error::EncodingError);
    }
    Ok((0..total_cols)
        .map(|column| encoded_entries[column * shape.encoded_rows + row_index])
        .collect())
}

fn absorb_short_field_metadata<B: ShortBaseField>(hasher: &mut Blake2s256) {
    absorb_framed(hasher, b"field-name", B::NAME.as_bytes());
    absorb_framed(hasher, b"modulus", &B::canonical_modulus_bytes());
    absorb_framed(
        hasher,
        b"canonical-bytes",
        &(B::CANONICAL_BYTES as u64).to_le_bytes(),
    );
    absorb_framed(
        hasher,
        b"extension-degree",
        &(B::EXTENSION_DEGREE as u64).to_le_bytes(),
    );
}

fn absorb_usize(hasher: &mut Blake2s256, label: &[u8], value: usize) -> Result<(), Error> {
    let value = u64::try_from(value).map_err(|_| Error::EncodingError)?;
    absorb_framed(hasher, label, &value.to_le_bytes());
    Ok(())
}
