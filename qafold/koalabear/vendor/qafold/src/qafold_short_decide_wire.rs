//! Canonical wire format for recursive short-field QAFold Decide proofs.
//!
//! This module is a child of `qafold_short_pc`. Fixed setup rows contain the
//! base-field `T_j || H_j` codeword, so they use the short base-field width.
//! Dynamic carry rows and transcript responses live in the extension field.

use super::*;

const DECIDE_PROOF_MAGIC: &[u8; 8] = b"QFSDCID1";
const OPTION_NONE: u8 = 0;
const OPTION_SOME: u8 = 1;
const ROWS_PARTIAL: u8 = 0;
const ROWS_ALL: u8 = 1;
const SETUP_SHAPE_FIELDS: usize = 7;

/// Authenticated payload for one recursive short-field Decide level.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortQaDecideRound<B: ShortBaseField> {
    /// Non-base QA level discharged by this round.
    pub level: usize,
    /// Root of the dynamic carry committed for the following level.
    pub next_carry_root: Option<[u8; DIGEST_BYTES]>,
    /// Authenticated rows of the fixed short-field `T_j || H_j` setup codeword.
    pub setup_rows: ShortQaSetupRows<B>,
    /// Authenticated rows of the current extension-field carry, when present.
    pub carry_rows: Option<ShortQaExtensionRows<B>>,
}

impl<B: ShortBaseField> ShortQaDecideRound<B> {
    /// Returns the exact length of this round's canonical encoding.
    pub fn encoded_len(&self) -> usize {
        encoded_decide_round_len(self)
    }

    /// Serializes one round without the enclosing proof header.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        validate_decide_round(self)?;
        let mut output = Vec::with_capacity(self.encoded_len());
        write_decide_round::<B>(&mut output, self)?;
        if output.len() != self.encoded_len() {
            return Err(Error::EncodingError);
        }
        Ok(output)
    }

    /// Parses one standalone round and rejects trailing data.
    pub fn from_bytes(input: &[u8]) -> Result<Self, Error> {
        let mut reader = DecideByteReader::new(input);
        let round = reader.read_decide_round::<B>()?;
        if !reader.is_empty() {
            return Err(Error::InvalidCommitment);
        }
        Ok(round)
    }
}

/// Complete recursive Decide proof for one short-field QAFold opening.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortQaDecideProof<B: ShortBaseField> {
    /// Extension-field prover messages consumed by the Decide transcript.
    pub responses: Vec<B::Extension>,
    /// Recursive rounds, ordered by the Decide protocol.
    pub rounds: Vec<ShortQaDecideRound<B>>,
    /// Directly revealed terminal vector over the extension field.
    pub base_vector: Vec<B::Extension>,
}

impl<B: ShortBaseField> ShortQaDecideProof<B> {
    /// Returns the authenticated setup-row query count for each Decide round.
    pub fn round_query_counts(&self) -> Vec<usize> {
        self.rounds
            .iter()
            .map(|round| match &round.setup_rows {
                ShortQaSetupRows::Partial(queries) => queries.len(),
                ShortQaSetupRows::AllRows { shape, .. } => shape.encoded_rows,
            })
            .collect()
    }

    /// Returns the exact length of the canonical encoding.
    pub fn encoded_len(&self) -> usize {
        let extension_bytes = B::Extension::canonical_byte_len();
        DECIDE_PROOF_MAGIC.len()
            + 8
            + self.responses.len() * extension_bytes
            + 8
            + self
                .rounds
                .iter()
                .map(encoded_decide_round_len::<B>)
                .sum::<usize>()
            + 8
            + self.base_vector.len() * extension_bytes
    }

    /// Serializes the proof with canonical fixed-width field encodings.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        validate_decide_proof(self)?;
        let mut output = Vec::with_capacity(self.encoded_len());
        output.extend_from_slice(DECIDE_PROOF_MAGIC);
        write_wire_len(&mut output, self.responses.len())?;
        for value in &self.responses {
            write_wire_extension::<B>(&mut output, *value)?;
        }
        write_wire_len(&mut output, self.rounds.len())?;
        for round in &self.rounds {
            write_decide_round::<B>(&mut output, round)?;
        }
        write_wire_len(&mut output, self.base_vector.len())?;
        for value in &self.base_vector {
            write_wire_extension::<B>(&mut output, *value)?;
        }
        if output.len() != self.encoded_len() {
            return Err(Error::EncodingError);
        }
        Ok(output)
    }

    /// Parses one canonical proof and rejects malformed lengths or trailing data.
    pub fn from_bytes(input: &[u8]) -> Result<Self, Error> {
        let mut reader = DecideByteReader::new(input);
        if reader.take(DECIDE_PROOF_MAGIC.len())? != DECIDE_PROOF_MAGIC {
            return Err(Error::InvalidCommitment);
        }

        let extension_bytes = B::Extension::canonical_byte_len();
        let responses_len = reader.read_len()?;
        if responses_len > reader.remaining() / extension_bytes {
            return Err(Error::InvalidCommitment);
        }
        let mut responses = reserve_wire_vec(responses_len)?;
        for _ in 0..responses_len {
            responses.push(reader.read_extension::<B>()?);
        }

        let rounds_len = reader.read_len()?;
        const MINIMUM_ROUND_BYTES: usize = 8 + 1 + 1 + 8 + 1;
        if rounds_len
            > reader
                .remaining()
                .saturating_sub(8)
                .checked_div(MINIMUM_ROUND_BYTES)
                .unwrap_or(0)
        {
            return Err(Error::InvalidCommitment);
        }
        let mut rounds = reserve_wire_vec(rounds_len)?;
        for _ in 0..rounds_len {
            rounds.push(reader.read_decide_round::<B>()?);
        }

        let base_len = reader.read_len()?;
        if base_len > reader.remaining() / extension_bytes {
            return Err(Error::InvalidCommitment);
        }
        let mut base_vector = reserve_wire_vec(base_len)?;
        for _ in 0..base_len {
            base_vector.push(reader.read_extension::<B>()?);
        }
        if !reader.is_empty() {
            return Err(Error::InvalidCommitment);
        }

        let proof = Self {
            responses,
            rounds,
            base_vector,
        };
        validate_decide_proof(&proof).map_err(|_| Error::InvalidCommitment)?;
        Ok(proof)
    }
}

fn validate_decide_proof<B: ShortBaseField>(proof: &ShortQaDecideProof<B>) -> Result<(), Error> {
    for round in &proof.rounds {
        validate_decide_round(round)?;
    }
    Ok(())
}

fn validate_decide_round<B: ShortBaseField>(round: &ShortQaDecideRound<B>) -> Result<(), Error> {
    if round.level == 0 {
        return Err(Error::EncodingError);
    }
    validate_setup_rows::<B>(&round.setup_rows, round.level)?;
    if let Some(rows) = &round.carry_rows {
        validate_carry_rows::<B>(rows)?;
    }
    Ok(())
}

fn validate_setup_shape(shape: ShortQaSetupShape) -> Result<usize, Error> {
    if shape.level == 0
        || shape.encoded_rows == 0
        || !shape.encoded_rows.is_power_of_two()
        || shape.message_rows == 0
        || !shape.message_rows.is_power_of_two()
        || shape.encoded_rows < shape.message_rows
        || shape.encoded_rows % shape.message_rows != 0
        || shape.h_cols == 0
        || !shape.h_cols.is_power_of_two()
    {
        return Err(Error::EncodingError);
    }
    if shape.t_cols == 0 {
        if shape.t_elements != 0 || shape.t_last_len != 0 {
            return Err(Error::EncodingError);
        }
    } else {
        if shape.t_last_len == 0 || shape.t_last_len > shape.message_rows {
            return Err(Error::EncodingError);
        }
        let expected_elements = shape
            .t_cols
            .checked_sub(1)
            .and_then(|full_columns| full_columns.checked_mul(shape.message_rows))
            .and_then(|elements| elements.checked_add(shape.t_last_len))
            .ok_or(Error::EncodingError)?;
        if shape.t_elements != expected_elements {
            return Err(Error::EncodingError);
        }
    }
    shape
        .t_cols
        .checked_add(shape.h_cols)
        .ok_or(Error::EncodingError)
}

fn validate_setup_rows<B: ShortBaseField>(
    rows: &ShortQaSetupRows<B>,
    level: usize,
) -> Result<(), Error> {
    match rows {
        ShortQaSetupRows::Partial(queries) => {
            for query in queries {
                if query.row.is_empty() {
                    return Err(Error::EncodingError);
                }
            }
        }
        ShortQaSetupRows::AllRows {
            encoded_entries,
            shape,
        } => {
            let total_cols = validate_setup_shape(*shape)?;
            let expected_len = shape
                .encoded_rows
                .checked_mul(total_cols)
                .ok_or(Error::EncodingError)?;
            if shape.level != level || encoded_entries.len() != expected_len {
                return Err(Error::EncodingError);
            }
        }
    }
    Ok(())
}

fn validate_carry_rows<B: ShortBaseField>(rows: &ShortQaExtensionRows<B>) -> Result<(), Error> {
    match rows {
        ShortQaExtensionRows::Partial(queries) => {
            if queries.iter().any(|query| query.row.is_empty()) {
                return Err(Error::EncodingError);
            }
        }
        ShortQaExtensionRows::AllRows {
            encoded_entries,
            n_rows,
            n_cols,
        } => {
            let expected_len = n_rows.checked_mul(*n_cols).ok_or(Error::EncodingError)?;
            if *n_rows == 0
                || !n_rows.is_power_of_two()
                || *n_cols == 0
                || encoded_entries.len() != expected_len
            {
                return Err(Error::EncodingError);
            }
        }
    }
    Ok(())
}

fn encoded_decide_round_len<B: ShortBaseField>(round: &ShortQaDecideRound<B>) -> usize {
    8 + encoded_optional_root_len(&round.next_carry_root)
        + encoded_setup_rows_len::<B>(&round.setup_rows)
        + 1
        + round
            .carry_rows
            .as_ref()
            .map(encoded_extension_rows_len::<B>)
            .unwrap_or(0)
}

fn encoded_optional_root_len(root: &Option<[u8; DIGEST_BYTES]>) -> usize {
    1 + usize::from(root.is_some()) * DIGEST_BYTES
}

fn encoded_setup_rows_len<B: ShortBaseField>(rows: &ShortQaSetupRows<B>) -> usize {
    match rows {
        ShortQaSetupRows::Partial(queries) => {
            1 + 8
                + queries
                    .iter()
                    .map(|query| {
                        8 + 8
                            + query.row.len() * B::CANONICAL_BYTES
                            + 8
                            + query.path.siblings.len() * DIGEST_BYTES
                    })
                    .sum::<usize>()
        }
        ShortQaSetupRows::AllRows {
            encoded_entries, ..
        } => 1 + SETUP_SHAPE_FIELDS * 8 + 8 + encoded_entries.len() * B::CANONICAL_BYTES,
    }
}

fn write_decide_round<B: ShortBaseField>(
    output: &mut Vec<u8>,
    round: &ShortQaDecideRound<B>,
) -> Result<(), Error> {
    write_wire_len(output, round.level)?;
    write_optional_root(output, &round.next_carry_root);
    write_setup_rows::<B>(output, &round.setup_rows)?;
    match &round.carry_rows {
        None => output.push(OPTION_NONE),
        Some(rows) => {
            output.push(OPTION_SOME);
            write_carry_rows::<B>(output, rows)?;
        }
    }
    Ok(())
}

fn write_optional_root(output: &mut Vec<u8>, root: &Option<[u8; DIGEST_BYTES]>) {
    match root {
        None => output.push(OPTION_NONE),
        Some(root) => {
            output.push(OPTION_SOME);
            output.extend_from_slice(root);
        }
    }
}

fn write_setup_rows<B: ShortBaseField>(
    output: &mut Vec<u8>,
    rows: &ShortQaSetupRows<B>,
) -> Result<(), Error> {
    match rows {
        ShortQaSetupRows::Partial(queries) => {
            output.push(ROWS_PARTIAL);
            write_wire_len(output, queries.len())?;
            for query in queries {
                write_wire_len(output, query.row_index)?;
                write_wire_len(output, query.row.len())?;
                for value in &query.row {
                    write_wire_base::<B>(output, *value)?;
                }
                write_wire_path(output, &query.path)?;
            }
        }
        ShortQaSetupRows::AllRows {
            encoded_entries,
            shape,
        } => {
            output.push(ROWS_ALL);
            write_setup_shape(output, *shape)?;
            write_wire_len(output, encoded_entries.len())?;
            for value in encoded_entries {
                write_wire_base::<B>(output, *value)?;
            }
        }
    }
    Ok(())
}

fn write_setup_shape(output: &mut Vec<u8>, shape: ShortQaSetupShape) -> Result<(), Error> {
    validate_setup_shape(shape)?;
    for value in [
        shape.level,
        shape.encoded_rows,
        shape.message_rows,
        shape.t_cols,
        shape.t_elements,
        shape.t_last_len,
        shape.h_cols,
    ] {
        write_wire_len(output, value)?;
    }
    Ok(())
}

fn write_carry_rows<B: ShortBaseField>(
    output: &mut Vec<u8>,
    rows: &ShortQaExtensionRows<B>,
) -> Result<(), Error> {
    match rows {
        ShortQaExtensionRows::Partial(queries) => {
            output.push(ROWS_PARTIAL);
            write_wire_len(output, queries.len())?;
            for query in queries {
                write_wire_len(output, query.row.len())?;
                for value in &query.row {
                    write_wire_extension::<B>(output, *value)?;
                }
                write_wire_path(output, &query.path)?;
            }
        }
        ShortQaExtensionRows::AllRows {
            encoded_entries,
            n_rows,
            n_cols,
        } => {
            output.push(ROWS_ALL);
            write_wire_len(output, *n_rows)?;
            write_wire_len(output, *n_cols)?;
            write_wire_len(output, encoded_entries.len())?;
            for value in encoded_entries {
                write_wire_extension::<B>(output, *value)?;
            }
        }
    }
    Ok(())
}

fn write_wire_path(output: &mut Vec<u8>, path: &ShortQaMerklePath) -> Result<(), Error> {
    write_wire_len(output, path.siblings.len())?;
    for sibling in &path.siblings {
        output.extend_from_slice(sibling);
    }
    Ok(())
}

fn write_wire_base<B: ShortBaseField>(output: &mut Vec<u8>, value: B) -> Result<(), Error> {
    let start = output.len();
    output.resize(start + B::CANONICAL_BYTES, 0);
    value.write_canonical_bytes(&mut output[start..])
}

fn write_wire_extension<B: ShortBaseField>(
    output: &mut Vec<u8>,
    value: B::Extension,
) -> Result<(), Error> {
    let start = output.len();
    output.resize(start + B::Extension::canonical_byte_len(), 0);
    value.write_canonical_bytes(&mut output[start..])
}

fn write_wire_len(output: &mut Vec<u8>, value: usize) -> Result<(), Error> {
    let value = u64::try_from(value).map_err(|_| Error::EncodingError)?;
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn reserve_wire_vec<T>(len: usize) -> Result<Vec<T>, Error> {
    let mut output = Vec::new();
    output
        .try_reserve_exact(len)
        .map_err(|_| Error::InvalidCommitment)?;
    Ok(output)
}

struct DecideByteReader<'a> {
    input: &'a [u8],
    cursor: usize,
}

impl<'a> DecideByteReader<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self { input, cursor: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], Error> {
        let end = self
            .cursor
            .checked_add(len)
            .ok_or(Error::InvalidCommitment)?;
        let output = self
            .input
            .get(self.cursor..end)
            .ok_or(Error::InvalidCommitment)?;
        self.cursor = end;
        Ok(output)
    }

    fn read_len(&mut self) -> Result<usize, Error> {
        let bytes: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| Error::InvalidCommitment)?;
        usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| Error::InvalidCommitment)
    }

    fn read_tag(&mut self) -> Result<u8, Error> {
        self.take(1)?
            .first()
            .copied()
            .ok_or(Error::InvalidCommitment)
    }

    fn read_base<B: ShortBaseField>(&mut self) -> Result<B, Error> {
        B::from_canonical_bytes(self.take(B::CANONICAL_BYTES)?).ok_or(Error::InvalidCommitment)
    }

    fn read_extension<B: ShortBaseField>(&mut self) -> Result<B::Extension, Error> {
        B::Extension::from_canonical_bytes(self.take(B::Extension::canonical_byte_len())?)
            .ok_or(Error::InvalidCommitment)
    }

    fn read_optional_root(&mut self) -> Result<Option<[u8; DIGEST_BYTES]>, Error> {
        match self.read_tag()? {
            OPTION_NONE => Ok(None),
            OPTION_SOME => Ok(Some(
                self.take(DIGEST_BYTES)?
                    .try_into()
                    .map_err(|_| Error::InvalidCommitment)?,
            )),
            _ => Err(Error::InvalidCommitment),
        }
    }

    fn read_path(&mut self) -> Result<ShortQaMerklePath, Error> {
        let len = self.read_len()?;
        if len > self.remaining() / DIGEST_BYTES {
            return Err(Error::InvalidCommitment);
        }
        let mut siblings = reserve_wire_vec(len)?;
        for _ in 0..len {
            siblings.push(
                self.take(DIGEST_BYTES)?
                    .try_into()
                    .map_err(|_| Error::InvalidCommitment)?,
            );
        }
        Ok(ShortQaMerklePath { siblings })
    }

    fn read_decide_round<B: ShortBaseField>(&mut self) -> Result<ShortQaDecideRound<B>, Error> {
        let level = self.read_len()?;
        let next_carry_root = self.read_optional_root()?;
        let setup_rows = self.read_setup_rows::<B>(level)?;
        let carry_rows = match self.read_tag()? {
            OPTION_NONE => None,
            OPTION_SOME => Some(self.read_carry_rows::<B>()?),
            _ => return Err(Error::InvalidCommitment),
        };
        let round = ShortQaDecideRound {
            level,
            next_carry_root,
            setup_rows,
            carry_rows,
        };
        validate_decide_round(&round).map_err(|_| Error::InvalidCommitment)?;
        Ok(round)
    }

    fn read_setup_rows<B: ShortBaseField>(
        &mut self,
        level: usize,
    ) -> Result<ShortQaSetupRows<B>, Error> {
        match self.read_tag()? {
            ROWS_PARTIAL => {
                let len = self.read_len()?;
                const MINIMUM_QUERY_BYTES: usize = 8 + 8 + 8;
                if len > self.remaining() / MINIMUM_QUERY_BYTES {
                    return Err(Error::InvalidCommitment);
                }
                let mut queries = reserve_wire_vec(len)?;
                for _ in 0..len {
                    let row_index = self.read_len()?;
                    let row_len = self.read_len()?;
                    let bytes_before_path = self
                        .remaining()
                        .checked_sub(8)
                        .ok_or(Error::InvalidCommitment)?;
                    if row_len == 0 || row_len > bytes_before_path / B::CANONICAL_BYTES {
                        return Err(Error::InvalidCommitment);
                    }
                    let mut row = reserve_wire_vec(row_len)?;
                    for _ in 0..row_len {
                        row.push(self.read_base::<B>()?);
                    }
                    let path = self.read_path()?;
                    queries.push(ShortQaSetupRowQuery {
                        row_index,
                        row,
                        path,
                    });
                }
                Ok(ShortQaSetupRows::Partial(queries))
            }
            ROWS_ALL => {
                let shape = self.read_setup_shape()?;
                if shape.level != level {
                    return Err(Error::InvalidCommitment);
                }
                let total_cols =
                    validate_setup_shape(shape).map_err(|_| Error::InvalidCommitment)?;
                let len = self.read_len()?;
                let expected_len = shape
                    .encoded_rows
                    .checked_mul(total_cols)
                    .ok_or(Error::InvalidCommitment)?;
                if len != expected_len || len > self.remaining() / B::CANONICAL_BYTES {
                    return Err(Error::InvalidCommitment);
                }
                let mut encoded_entries = reserve_wire_vec(len)?;
                for _ in 0..len {
                    encoded_entries.push(self.read_base::<B>()?);
                }
                Ok(ShortQaSetupRows::AllRows {
                    encoded_entries,
                    shape,
                })
            }
            _ => Err(Error::InvalidCommitment),
        }
    }

    fn read_setup_shape(&mut self) -> Result<ShortQaSetupShape, Error> {
        let shape = ShortQaSetupShape {
            level: self.read_len()?,
            encoded_rows: self.read_len()?,
            message_rows: self.read_len()?,
            t_cols: self.read_len()?,
            t_elements: self.read_len()?,
            t_last_len: self.read_len()?,
            h_cols: self.read_len()?,
        };
        validate_setup_shape(shape).map_err(|_| Error::InvalidCommitment)?;
        Ok(shape)
    }

    fn read_carry_rows<B: ShortBaseField>(&mut self) -> Result<ShortQaExtensionRows<B>, Error> {
        match self.read_tag()? {
            ROWS_PARTIAL => {
                let len = self.read_len()?;
                const MINIMUM_QUERY_BYTES: usize = 8 + 8;
                if len > self.remaining() / MINIMUM_QUERY_BYTES {
                    return Err(Error::InvalidCommitment);
                }
                let mut queries = reserve_wire_vec(len)?;
                for _ in 0..len {
                    let row_len = self.read_len()?;
                    let extension_bytes = B::Extension::canonical_byte_len();
                    let bytes_before_path = self
                        .remaining()
                        .checked_sub(8)
                        .ok_or(Error::InvalidCommitment)?;
                    if row_len == 0 || row_len > bytes_before_path / extension_bytes {
                        return Err(Error::InvalidCommitment);
                    }
                    let mut row = reserve_wire_vec(row_len)?;
                    for _ in 0..row_len {
                        row.push(self.read_extension::<B>()?);
                    }
                    let path = self.read_path()?;
                    queries.push(ShortQaExtensionRowQuery { row, path });
                }
                Ok(ShortQaExtensionRows::Partial(queries))
            }
            ROWS_ALL => {
                let n_rows = self.read_len()?;
                let n_cols = self.read_len()?;
                let len = self.read_len()?;
                let expected_len = n_rows.checked_mul(n_cols).ok_or(Error::InvalidCommitment)?;
                let extension_bytes = B::Extension::canonical_byte_len();
                if n_rows == 0
                    || !n_rows.is_power_of_two()
                    || n_cols == 0
                    || len != expected_len
                    || len > self.remaining() / extension_bytes
                {
                    return Err(Error::InvalidCommitment);
                }
                let mut encoded_entries = reserve_wire_vec(len)?;
                for _ in 0..len {
                    encoded_entries.push(self.read_extension::<B>()?);
                }
                Ok(ShortQaExtensionRows::AllRows {
                    encoded_entries,
                    n_rows,
                    n_cols,
                })
            }
            _ => Err(Error::InvalidCommitment),
        }
    }

    fn remaining(&self) -> usize {
        self.input.len().saturating_sub(self.cursor)
    }

    fn is_empty(&self) -> bool {
        self.cursor == self.input.len()
    }
}
