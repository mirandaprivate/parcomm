//! Recursive Decide driver for the short-prime QAFold PCS.
//!
//! Every fixed `T_j || H_j` row is authenticated with the setup commitment's
//! short-base-field leaf format. Dynamic carry rows use extension-field leaves.
//! In direct-interleaved mode the recursive witness remains one logical column
//! and expands to exactly `t` physical columns. The driver folds the logical
//! mixed matrix without allocating its power-of-two column padding.

use super::*;

use super::decide::{
    ShortEvalWithPoint, ShortQaRfcTClaim, prove_multi_eval, verify_base_t_claims, verify_multi_eval,
};
use super::decide_carry::{
    ShortQaDerivedCarryCache, commit_short_derived_carry, commit_short_direct_interleaved_carry,
    fold_short_direct_interleaved_witness, into_short_derived_carry_rows, prove_short_decide_iota,
    prove_short_qa_code_switch_carry, prove_short_qa_direct_code_switch_carry,
    verify_short_decide_iota, verify_short_qa_code_switch_carry,
    verify_short_qa_direct_code_switch_carry,
};
use super::decide_protocol::{
    prepare_pending_h_claims_prover, prepare_pending_h_claims_verifier,
    prepare_short_decide_eval_claims,
};
const DECIDE_ROUND_DOMAIN: &[u8] = b"qafold-short/decide-round/v1";
const DECIDE_NEXT_CARRY_DOMAIN: &[u8] = b"qafold-short/decide-next-carry/v1";

impl<B: ShortBaseField> ShortQaFoldPC<B> {
    /// Proves every generator claim left by a successful short-field opening.
    pub fn prove_decide(
        &self,
        pp: &ShortQaFoldParameters<B>,
        setup: &ShortQaFoldSetupCommitments<B>,
        opening: &ShortQaFoldVerificationReduction<B::Extension>,
    ) -> Result<ShortQaDecideProof<B>, Error> {
        prove_short_qafold_decide(self, pp, setup, opening)
    }

    /// Verifies the recursive short-field Decide proof.
    pub fn verify_decide(
        &self,
        pp: &ShortQaFoldParameters<B>,
        setup: &ShortQaFoldSetupCommitments<B>,
        opening: &ShortQaFoldVerificationReduction<B::Extension>,
        proof: &ShortQaDecideProof<B>,
    ) -> Result<bool, Error> {
        verify_short_qafold_decide(self, pp, setup, opening, proof)
    }
}

fn prove_short_qafold_decide<B: ShortBaseField>(
    pc: &ShortQaFoldPC<B>,
    pp: &ShortQaFoldParameters<B>,
    setup: &ShortQaFoldSetupCommitments<B>,
    opening: &ShortQaFoldVerificationReduction<B::Extension>,
) -> Result<ShortQaDecideProof<B>, Error> {
    pc.validate_setup_commitments(pp, setup)?;
    let levels = pp.sequence.codes().len();
    if setup.levels() != levels {
        return Err(Error::InvalidCommitment);
    }

    let mut transcript = ShortTranscriptWriter::<B>::new();
    let mut h_claims = prepare_pending_h_claims_prover(pp, opening, &mut transcript)?;
    if levels <= 1 {
        if h_claims.iter().any(|claims| !claims.is_empty()) {
            return Err(Error::InvalidCommitment);
        }
        return Ok(ShortQaDecideProof {
            responses: transcript.responses,
            rounds: Vec::new(),
            base_vector: Vec::new(),
        });
    }

    let mut current_t_claims = Vec::<ShortQaRfcTClaim<B::Extension>>::new();
    let mut current_w_claim: Option<ShortEvalWithPoint<B::Extension>> = None;
    let mut current_carry: Option<ShortQaDerivedCarryCache<B>> = None;
    let mut current_carry_commitment: Option<ShortQaFoldCommitment> = None;
    let mut rounds = Vec::with_capacity(levels - 1);
    let mut base_vector = Vec::new();
    let direct_interleaved = pp.derived_parameters.is_direct_interleaved();
    let row_factor = pp.sequence.switch_ratio();

    for level in (1..levels).rev() {
        let setup_level = setup.level(level)?;
        let level_h_claims = core::mem::take(&mut h_claims[level]);
        let prepared = prepare_short_decide_eval_claims(
            pp,
            setup_level,
            level,
            &level_h_claims,
            &current_t_claims,
            current_w_claim.as_ref(),
        )?;
        validate_current_carry(
            setup_level.shape,
            prepared.carry_cols,
            direct_interleaved,
            row_factor,
            current_carry.as_ref(),
            current_carry_commitment.as_ref(),
        )?;
        absorb_round_context_writer(
            level,
            setup_level,
            current_carry_commitment.as_ref(),
            &mut transcript,
        );

        let code = pp.sequence.codes().get(level).ok_or(Error::EncodingError)?;
        let rows = code.message_len();
        let setup_columns = setup_message_columns(pp, setup_level, level)?;
        let carry_columns = match current_carry.as_ref() {
            Some(carry) => carry
                .message_entries()
                .chunks_exact(rows)
                .collect::<Vec<_>>(),
            None => Vec::new(),
        };
        if setup_columns.len() != prepared.setup_cols || carry_columns.len() != prepared.carry_cols
        {
            return Err(Error::EncodingError);
        }
        let multi_eval_result = prove_multi_eval::<B>(
            &prepared.claims,
            &setup_columns,
            &carry_columns,
            prepared.log_rows,
            prepared.log_cols,
            &mut transcript,
        )?;
        let multi_eval = multi_eval_result.reduction;
        let folded_vector = multi_eval_result.folded_vector;
        if !multi_eval.ok || multi_eval.reduced.point.len() != prepared.log_cols + prepared.log_rows
        {
            return Err(Error::InvalidCommitment);
        }
        let (col_point, merged_left_point) = multi_eval.reduced.point.split_at(prepared.log_cols);
        let col_weights = tensor_weights(col_point);
        drop(carry_columns);
        drop(setup_columns);

        let target = pp
            .sequence
            .codes()
            .get(level - 1)
            .ok_or(Error::EncodingError)?;
        let source = if direct_interleaved {
            None
        } else {
            Some(pp.derived_parameters.code(level)?)
        };
        let n_encoded_rows = match source {
            Some(source) => source.codeword_len(target)?,
            None => target.codeword_len()?,
        };
        if setup_level.shape.encoded_rows != n_encoded_rows {
            return Err(Error::EncodingError);
        }
        let selection = derive_row_selection_writer(
            target,
            source.map_or(0, |source| source.layers()),
            pp.security_bits,
            n_encoded_rows,
            &mut transcript,
        )?;
        let setup_rows = setup::open_short_setup_rows(&setup_level.cache, &selection)?;
        let carry_rows = match current_carry.take() {
            Some(carry) => Some(into_short_derived_carry_rows(carry, &selection)?),
            None => None,
        };
        let (iota_col_weights, authenticated_carry_cols) = if direct_interleaved {
            direct_interleaved_iota_weights(
                &col_weights,
                prepared.setup_cols,
                prepared.carry_cols,
                merged_left_point,
                row_factor,
            )?
        } else {
            (col_weights.clone(), prepared.carry_cols)
        };
        let iota = prove_short_decide_iota(
            level,
            &setup_level.commitment,
            setup_level.shape,
            &setup_rows,
            current_carry_commitment.as_ref(),
            carry_rows.as_ref(),
            authenticated_carry_cols,
            &selection,
            &iota_col_weights,
            &mut transcript,
        )?;

        let recursive_vector = if direct_interleaved {
            fold_short_direct_interleaved_witness(
                folded_vector,
                target.message_len(),
                row_factor,
                merged_left_point,
            )?
        } else {
            folded_vector
        };

        let mut terminal_vector = None;
        let (next_carry_commitment, next_carry) = if level > 1 {
            let next_target = pp
                .sequence
                .codes()
                .get(level - 2)
                .ok_or(Error::EncodingError)?;
            let (commitment, carry) = if direct_interleaved {
                commit_short_direct_interleaved_carry(
                    next_target,
                    recursive_vector,
                    target.message_len(),
                    1,
                    row_factor,
                )?
            } else {
                let next_source = pp.derived_parameters.code(level - 1)?;
                commit_short_derived_carry(next_source, next_target, recursive_vector, row_factor)?
            };
            (Some(commitment), Some(carry))
        } else {
            terminal_vector = Some(recursive_vector);
            (None, None)
        };
        absorb_next_carry_writer(next_carry_commitment.as_ref(), &mut transcript);
        let folded_for_switch = match (&next_carry, &terminal_vector) {
            (Some(carry), None) => carry.message_entries(),
            (None, Some(vector)) => vector.as_slice(),
            _ => return Err(Error::InvalidCommitment),
        };
        let carry = if direct_interleaved {
            prove_short_qa_direct_code_switch_carry(
                target,
                level - 1,
                row_factor,
                &selection,
                &iota,
                folded_for_switch,
                multi_eval.reduced.value,
                merged_left_point,
                &mut transcript,
            )?
        } else {
            prove_short_qa_code_switch_carry(
                source.ok_or(Error::EncodingError)?,
                target,
                level - 1,
                &selection,
                &iota,
                folded_for_switch,
                multi_eval.reduced.value,
                merged_left_point,
                &mut transcript,
            )?
        };

        if level > 1 {
            let generator_reduction = ShortQaCode::<B>::verify_as_subprotocol(
                target.message_len(),
                target.index(),
                &carry.generator_claim,
            )?;
            if !generator_reduction.ok {
                return Err(Error::InvalidCommitment);
            }
            h_claims[level - 1].push(generator_reduction.alpha_claim);
            current_t_claims = carry.t_claims;
            current_w_claim = Some(carry.folded_w_claim);
            current_carry = next_carry;
            current_carry_commitment = next_carry_commitment;
        } else {
            let terminal_vector = terminal_vector.ok_or(Error::InvalidCommitment)?;
            let t_claims_ok = match source {
                Some(source) => verify_base_t_claims(source, &carry.t_claims)?,
                None => carry.t_claims.is_empty(),
            };
            if target.evaluate_generator_mle_extension(
                &carry.generator_claim.point.0,
                &carry.generator_claim.point.1,
            )? != carry.generator_claim.value
                || !t_claims_ok
                || mle_eval(&terminal_vector, &carry.folded_w_claim.point)
                    != Some(carry.folded_w_claim.value)
            {
                return Err(Error::InvalidCommitment);
            }
            for value in &terminal_vector {
                transcript.push_response(*value)?;
            }
            base_vector = terminal_vector;
            current_carry_commitment = None;
        }
        rounds.push(ShortQaDecideRound {
            level,
            next_carry_root: next_carry_commitment.map(|commitment| commitment.root),
            setup_rows,
            carry_rows,
        });
    }
    if h_claims.iter().any(|claims| !claims.is_empty())
        || current_carry.is_some()
        || current_carry_commitment.is_some()
    {
        return Err(Error::InvalidCommitment);
    }
    Ok(ShortQaDecideProof {
        responses: transcript.responses,
        rounds,
        base_vector,
    })
}

fn verify_short_qafold_decide<B: ShortBaseField>(
    pc: &ShortQaFoldPC<B>,
    pp: &ShortQaFoldParameters<B>,
    setup: &ShortQaFoldSetupCommitments<B>,
    opening: &ShortQaFoldVerificationReduction<B::Extension>,
    proof: &ShortQaDecideProof<B>,
) -> Result<bool, Error> {
    if !opening.ok {
        return Ok(false);
    }
    pc.validate_setup_commitments(pp, setup)?;
    let levels = pp.sequence.codes().len();
    if setup.levels() != levels {
        return Ok(false);
    }
    let mut transcript = ShortTranscriptReader::<B>::new(&proof.responses);
    let mut h_claims = prepare_pending_h_claims_verifier(pp, opening, &mut transcript)?;
    if levels <= 1 {
        return Ok(h_claims.iter().all(Vec::is_empty)
            && proof.rounds.is_empty()
            && proof.base_vector.is_empty()
            && transcript.is_exhausted());
    }
    if proof.rounds.len() != levels - 1 {
        return Ok(false);
    }

    let mut current_t_claims = Vec::<ShortQaRfcTClaim<B::Extension>>::new();
    let mut current_w_claim: Option<ShortEvalWithPoint<B::Extension>> = None;
    let mut current_carry_commitment: Option<ShortQaFoldCommitment> = None;
    let direct_interleaved = pp.derived_parameters.is_direct_interleaved();
    let row_factor = pp.sequence.switch_ratio();

    for (round_index, level) in (1..levels).rev().enumerate() {
        let round = &proof.rounds[round_index];
        if round.level != level {
            return Ok(false);
        }
        let setup_level = setup.level(level)?;
        let level_h_claims = core::mem::take(&mut h_claims[level]);
        let prepared = prepare_short_decide_eval_claims(
            pp,
            setup_level,
            level,
            &level_h_claims,
            &current_t_claims,
            current_w_claim.as_ref(),
        )?;
        if current_carry_commitment.is_some() != (prepared.carry_cols > 0) {
            return Ok(false);
        }
        absorb_round_context_reader(
            level,
            setup_level,
            current_carry_commitment.as_ref(),
            &mut transcript,
        );
        let multi_eval = verify_multi_eval::<B>(
            &prepared.claims,
            prepared.log_rows,
            prepared.log_cols,
            &mut transcript,
        )?;
        if !multi_eval.ok || multi_eval.reduced.point.len() != prepared.log_cols + prepared.log_rows
        {
            return Ok(false);
        }
        let (col_point, merged_left_point) = multi_eval.reduced.point.split_at(prepared.log_cols);
        let col_weights = tensor_weights(col_point);

        let target = pp
            .sequence
            .codes()
            .get(level - 1)
            .ok_or(Error::EncodingError)?;
        let source = if direct_interleaved {
            None
        } else {
            Some(pp.derived_parameters.code(level)?)
        };
        let n_encoded_rows = match source {
            Some(source) => source.codeword_len(target)?,
            None => target.codeword_len()?,
        };
        if setup_level.shape.encoded_rows != n_encoded_rows {
            return Err(Error::EncodingError);
        }
        let selection = derive_row_selection_reader(
            target,
            source.map_or(0, |source| source.layers()),
            pp.security_bits,
            n_encoded_rows,
            &mut transcript,
        )?;
        let (iota_col_weights, authenticated_carry_cols) = if direct_interleaved {
            direct_interleaved_iota_weights(
                &col_weights,
                prepared.setup_cols,
                prepared.carry_cols,
                merged_left_point,
                row_factor,
            )?
        } else {
            (col_weights.clone(), prepared.carry_cols)
        };
        let iota = match verify_short_decide_iota(
            level,
            &setup_level.commitment,
            setup_level.shape,
            &round.setup_rows,
            current_carry_commitment.as_ref(),
            round.carry_rows.as_ref(),
            authenticated_carry_cols,
            &selection,
            &iota_col_weights,
            &mut transcript,
        ) {
            Ok(iota) => iota,
            Err(Error::InvalidCommitment | Error::TranscriptError) => return Ok(false),
            Err(error) => return Err(error),
        };

        let next_carry_commitment = match (level > 1, round.next_carry_root) {
            (true, Some(root)) => Some(ShortQaFoldCommitment { root }),
            (false, None) => None,
            _ => return Ok(false),
        };
        absorb_next_carry_reader(next_carry_commitment.as_ref(), &mut transcript);
        let carry_result = if direct_interleaved {
            verify_short_qa_direct_code_switch_carry(
                target,
                level - 1,
                row_factor,
                &selection,
                &iota,
                multi_eval.reduced.value,
                merged_left_point,
                &mut transcript,
            )
        } else {
            verify_short_qa_code_switch_carry(
                source.ok_or(Error::EncodingError)?,
                target,
                level - 1,
                &selection,
                &iota,
                multi_eval.reduced.value,
                merged_left_point,
                &mut transcript,
            )
        };
        let carry = match carry_result {
            Ok(carry) => carry,
            Err(Error::InvalidCommitment | Error::TranscriptError) => return Ok(false),
            Err(error) => return Err(error),
        };

        if level > 1 {
            let reduction = ShortQaCode::<B>::verify_as_subprotocol(
                target.message_len(),
                target.index(),
                &carry.generator_claim,
            )?;
            if !reduction.ok {
                return Ok(false);
            }
            h_claims[level - 1].push(reduction.alpha_claim);
            current_t_claims = carry.t_claims;
            current_w_claim = Some(carry.folded_w_claim);
            current_carry_commitment = next_carry_commitment;
        } else {
            let t_claims_ok = match source {
                Some(source) => verify_base_t_claims(source, &carry.t_claims)?,
                None => carry.t_claims.is_empty(),
            };
            let expected_base_len = if direct_interleaved {
                target.message_len()
            } else {
                source.ok_or(Error::EncodingError)?.message_len(target)?
            };
            if target.evaluate_generator_mle_extension(
                &carry.generator_claim.point.0,
                &carry.generator_claim.point.1,
            )? != carry.generator_claim.value
                || !t_claims_ok
                || proof.base_vector.len() != expected_base_len
            {
                return Ok(false);
            }
            for value in &proof.base_vector {
                if transcript.read_response()? != *value {
                    return Ok(false);
                }
            }
            if mle_eval(&proof.base_vector, &carry.folded_w_claim.point)
                != Some(carry.folded_w_claim.value)
            {
                return Ok(false);
            }
            current_carry_commitment = None;
        }
    }

    Ok(h_claims.iter().all(Vec::is_empty)
        && current_carry_commitment.is_none()
        && transcript.is_exhausted())
}

fn validate_current_carry<B: ShortBaseField>(
    shape: ShortQaSetupShape,
    carry_cols: usize,
    direct_interleaved: bool,
    row_factor: usize,
    carry: Option<&ShortQaDerivedCarryCache<B>>,
    commitment: Option<&ShortQaFoldCommitment>,
) -> Result<(), Error> {
    if carry.is_some() != commitment.is_some() || carry.is_some() != (carry_cols > 0) {
        return Err(Error::InvalidCommitment);
    }
    if let Some(carry) = carry {
        if direct_interleaved {
            let logical_rows = shape
                .message_rows
                .checked_mul(row_factor)
                .ok_or(Error::EncodingError)?;
            let encoded_entries = shape
                .encoded_rows
                .checked_mul(row_factor)
                .ok_or(Error::EncodingError)?;
            if row_factor <= 1
                || !row_factor.is_power_of_two()
                || carry_cols != 1
                || carry.logical_rows() != logical_rows
                || carry.logical_columns() != 1
                || carry.encoding_message_rows() != shape.message_rows
                || carry.encoded_rows() != shape.encoded_rows
                || carry.physical_columns() != row_factor
                || carry.message_entries().len() != logical_rows
                || carry.encoded_entries_len() != encoded_entries
            {
                return Err(Error::EncodingError);
            }
        } else {
            let message_entries = shape
                .message_rows
                .checked_mul(carry_cols)
                .ok_or(Error::EncodingError)?;
            let encoded_entries = shape
                .encoded_rows
                .checked_mul(carry_cols)
                .ok_or(Error::EncodingError)?;
            if carry.logical_rows() != shape.message_rows
                || carry.logical_columns() != carry_cols
                || carry.encoding_message_rows() != shape.message_rows
                || carry.encoded_rows() != shape.encoded_rows
                || carry.physical_columns() != carry_cols
                || carry.message_entries().len() != message_entries
                || carry.encoded_entries_len() != encoded_entries
            {
                return Err(Error::EncodingError);
            }
        }
    }
    Ok(())
}

fn setup_message_columns<'a, B: ShortBaseField>(
    pp: &'a ShortQaFoldParameters<B>,
    setup_level: &'a ShortQaFoldSetupLevel<B>,
    level: usize,
) -> Result<Vec<&'a [B]>, Error> {
    let code = pp.sequence.codes().get(level).ok_or(Error::EncodingError)?;
    let rows = code.message_len();
    let columns = setup_level
        .t_columns
        .iter()
        .chain(code.alphas())
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let direct_interleaved = pp.derived_parameters.is_direct_interleaved();
    let expected_physical_cols = if direct_interleaved {
        columns
            .len()
            .checked_mul(pp.sequence.switch_ratio())
            .ok_or(Error::EncodingError)?
    } else {
        columns.len()
    };
    if setup_level.shape.level != level
        || (!direct_interleaved && setup_level.shape.message_rows != rows)
        || expected_physical_cols != setup_level.shape.total_cols()?
        || setup_level
            .t_columns
            .iter()
            .any(|column| column.is_empty() || column.len() > rows)
        || code.alphas().iter().any(|column| column.len() != rows)
    {
        return Err(Error::EncodingError);
    }
    Ok(columns)
}

fn direct_interleaved_iota_weights<E: ShortFieldElement>(
    logical_padded_weights: &[E],
    logical_setup_cols: usize,
    logical_carry_cols: usize,
    logical_row_point: &[E],
    row_factor: usize,
) -> Result<(Vec<E>, usize), Error> {
    if row_factor <= 1 || !row_factor.is_power_of_two() {
        return Err(Error::InvalidNumberOfVariables);
    }
    let block_log = row_factor.ilog2() as usize;
    if logical_row_point.len() < block_log {
        return Err(Error::InvalidNumberOfVariables);
    }
    let logical_cols = logical_setup_cols
        .checked_add(logical_carry_cols)
        .ok_or(Error::EncodingError)?;
    if logical_cols == 0 || logical_cols > logical_padded_weights.len() {
        return Err(Error::InvalidNumberOfVariables);
    }
    let block_weights = tensor_weights(&logical_row_point[..block_log]);
    if block_weights.len() != row_factor {
        return Err(Error::InvalidNumberOfVariables);
    }
    let physical_cols = logical_cols
        .checked_mul(row_factor)
        .ok_or(Error::EncodingError)?;
    let padded_physical_cols = physical_cols
        .checked_next_power_of_two()
        .ok_or(Error::EncodingError)?;
    let mut physical_weights = vec![E::zero(); padded_physical_cols];
    for logical_col in 0..logical_cols {
        for (block, block_weight) in block_weights.iter().enumerate() {
            physical_weights[logical_col * row_factor + block] =
                logical_padded_weights[logical_col] * *block_weight;
        }
    }
    let physical_carry_cols = logical_carry_cols
        .checked_mul(row_factor)
        .ok_or(Error::EncodingError)?;
    Ok((physical_weights, physical_carry_cols))
}

fn derive_row_selection_writer<B: ShortBaseField>(
    target: &ShortQaCode<B>,
    rf_layers: usize,
    security_bits: usize,
    available_rows: usize,
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<RowSelection, Error> {
    let query_count = target.num_queries(security_bits, rf_layers, available_rows)?;
    let seed = transcript.challenge_digest(b"decide-row-selection");
    row_selection(query_count, available_rows, seed)
}

fn derive_row_selection_reader<B: ShortBaseField>(
    target: &ShortQaCode<B>,
    rf_layers: usize,
    security_bits: usize,
    available_rows: usize,
    transcript: &mut ShortTranscriptReader<'_, B>,
) -> Result<RowSelection, Error> {
    let query_count = target.num_queries(security_bits, rf_layers, available_rows)?;
    let seed = transcript.challenge_digest(b"decide-row-selection");
    row_selection(query_count, available_rows, seed)
}

fn row_selection(
    query_count: usize,
    available_rows: usize,
    seed: [u8; DIGEST_BYTES],
) -> Result<RowSelection, Error> {
    if query_count == 0 || query_count > available_rows {
        return Err(Error::InvalidNumberOfVariables);
    }
    if query_count == available_rows {
        Ok(RowSelection::All(available_rows))
    } else {
        Ok(RowSelection::Partial(sample_without_replacement(
            query_count,
            available_rows,
            seed,
        )?))
    }
}

fn absorb_round_context_writer<B: ShortBaseField>(
    level: usize,
    setup_level: &ShortQaFoldSetupLevel<B>,
    carry: Option<&ShortQaFoldCommitment>,
    transcript: &mut ShortTranscriptWriter<B>,
) {
    transcript.absorb_bytes(b"decide-round-domain", DECIDE_ROUND_DOMAIN);
    absorb_round_context_common_writer(level, setup_level, carry, transcript);
}

fn absorb_round_context_reader<B: ShortBaseField>(
    level: usize,
    setup_level: &ShortQaFoldSetupLevel<B>,
    carry: Option<&ShortQaFoldCommitment>,
    transcript: &mut ShortTranscriptReader<'_, B>,
) {
    transcript.absorb_bytes(b"decide-round-domain", DECIDE_ROUND_DOMAIN);
    transcript.absorb_bytes(b"decide-round-level", &(level as u64).to_le_bytes());
    for (label, value) in round_shape_fields(setup_level.shape) {
        transcript.absorb_bytes(label, &(value as u64).to_le_bytes());
    }
    transcript.absorb_digest(b"decide-round-setup-root", &setup_level.commitment.root);
    match carry {
        Some(commitment) => {
            transcript.absorb_bytes(b"decide-round-carry-present", &[1]);
            transcript.absorb_digest(b"decide-round-carry-root", &commitment.root);
        }
        None => transcript.absorb_bytes(b"decide-round-carry-present", &[0]),
    }
}

fn absorb_round_context_common_writer<B: ShortBaseField>(
    level: usize,
    setup_level: &ShortQaFoldSetupLevel<B>,
    carry: Option<&ShortQaFoldCommitment>,
    transcript: &mut ShortTranscriptWriter<B>,
) {
    transcript.absorb_bytes(b"decide-round-level", &(level as u64).to_le_bytes());
    for (label, value) in round_shape_fields(setup_level.shape) {
        transcript.absorb_bytes(label, &(value as u64).to_le_bytes());
    }
    transcript.absorb_digest(b"decide-round-setup-root", &setup_level.commitment.root);
    match carry {
        Some(commitment) => {
            transcript.absorb_bytes(b"decide-round-carry-present", &[1]);
            transcript.absorb_digest(b"decide-round-carry-root", &commitment.root);
        }
        None => transcript.absorb_bytes(b"decide-round-carry-present", &[0]),
    }
}

fn round_shape_fields(shape: ShortQaSetupShape) -> [(&'static [u8], usize); 7] {
    [
        (b"decide-round-shape-level", shape.level),
        (b"decide-round-shape-encoded-rows", shape.encoded_rows),
        (b"decide-round-shape-message-rows", shape.message_rows),
        (b"decide-round-shape-t-cols", shape.t_cols),
        (b"decide-round-shape-t-elements", shape.t_elements),
        (b"decide-round-shape-t-last-len", shape.t_last_len),
        (b"decide-round-shape-h-cols", shape.h_cols),
    ]
}

fn absorb_next_carry_writer<B: ShortBaseField>(
    carry: Option<&ShortQaFoldCommitment>,
    transcript: &mut ShortTranscriptWriter<B>,
) {
    transcript.absorb_bytes(b"decide-next-carry-domain", DECIDE_NEXT_CARRY_DOMAIN);
    match carry {
        Some(commitment) => {
            transcript.absorb_bytes(b"decide-next-carry-present", &[1]);
            transcript.absorb_digest(b"decide-next-carry-root", &commitment.root);
        }
        None => transcript.absorb_bytes(b"decide-next-carry-present", &[0]),
    }
}

fn absorb_next_carry_reader<B: ShortBaseField>(
    carry: Option<&ShortQaFoldCommitment>,
    transcript: &mut ShortTranscriptReader<'_, B>,
) {
    transcript.absorb_bytes(b"decide-next-carry-domain", DECIDE_NEXT_CARRY_DOMAIN);
    match carry {
        Some(commitment) => {
            transcript.absorb_bytes(b"decide-next-carry-present", &[1]);
            transcript.absorb_digest(b"decide-next-carry-root", &commitment.root);
        }
        None => transcript.absorb_bytes(b"decide-next-carry-present", &[0]),
    }
}
