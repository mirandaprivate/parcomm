//! Claim preparation for the recursive short-field QAFold Decide protocol.
//!
//! The fixed setup matrix is stored in `T_j || H_j` column order over the
//! short base field. Dynamic carry columns stay in the extension field. This
//! module converts every recursive claim into the common mixed-matrix form
//! consumed by the Decide `MultiEval` reduction.

use super::*;

use super::decide::{ShortEvalClaim, ShortEvalWithPoint, ShortQaRfcTClaim};
use crate::qafold_short::ShortQaMatrixClaim;

const DECIDE_DOMAIN: &[u8] = b"qafold-short/decide/v1";

fn validate_opening_pending_order<B: ShortBaseField>(
    pp: &ShortQaFoldParameters<B>,
    opening: &ShortQaFoldVerificationReduction<B::Extension>,
) -> Result<(), Error> {
    let levels = pp.sequence.codes().len();
    if !opening.ok || opening.pending_h_claims.len() != levels.saturating_sub(1) {
        return Err(Error::InvalidCommitment);
    }
    for (pending, expected_level) in opening.pending_h_claims.iter().zip((1..levels).rev()) {
        if pending.level != expected_level {
            return Err(Error::InvalidCommitment);
        }
    }
    Ok(())
}

fn absorb_opening_statement_writer<B: ShortBaseField>(
    pp: &ShortQaFoldParameters<B>,
    opening: &ShortQaFoldVerificationReduction<B::Extension>,
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<(), Error> {
    transcript.absorb_bytes(b"decide-domain", DECIDE_DOMAIN);
    transcript.absorb_digest(b"decide-parameters", &pp.parameter_digest);
    transcript.absorb_digest(b"decide-setup", &pp.setup_digest());
    transcript.absorb_bytes(
        b"decide-levels",
        &(pp.sequence.codes().len() as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(b"opening-reduction-ok", &[u8::from(opening.ok)]);
    transcript.absorb_bytes(
        b"pending-h-count",
        &(opening.pending_h_claims.len() as u64).to_le_bytes(),
    );
    for pending in &opening.pending_h_claims {
        transcript.absorb_bytes(b"pending-h-level", &(pending.level as u64).to_le_bytes());
        transcript.absorb_extension(b"pending-g-value", pending.value)?;
        transcript.absorb_extensions(b"pending-g-left", &pending.left_point)?;
        transcript.absorb_extensions(b"pending-g-right", &pending.right_point)?;
    }
    Ok(())
}

fn absorb_opening_statement_reader<B: ShortBaseField>(
    pp: &ShortQaFoldParameters<B>,
    opening: &ShortQaFoldVerificationReduction<B::Extension>,
    transcript: &mut ShortTranscriptReader<'_, B>,
) -> Result<(), Error> {
    transcript.absorb_bytes(b"decide-domain", DECIDE_DOMAIN);
    transcript.absorb_digest(b"decide-parameters", &pp.parameter_digest);
    transcript.absorb_digest(b"decide-setup", &pp.setup_digest());
    transcript.absorb_bytes(
        b"decide-levels",
        &(pp.sequence.codes().len() as u64).to_le_bytes(),
    );
    transcript.absorb_bytes(b"opening-reduction-ok", &[u8::from(opening.ok)]);
    transcript.absorb_bytes(
        b"pending-h-count",
        &(opening.pending_h_claims.len() as u64).to_le_bytes(),
    );
    for pending in &opening.pending_h_claims {
        transcript.absorb_bytes(b"pending-h-level", &(pending.level as u64).to_le_bytes());
        transcript.absorb_extension(b"pending-g-value", pending.value)?;
        transcript.absorb_extensions(b"pending-g-left", &pending.left_point)?;
        transcript.absorb_extensions(b"pending-g-right", &pending.right_point)?;
    }
    Ok(())
}

/// Converts opening generator claims to setup-committed `H_j` claims.
pub(super) fn prepare_pending_h_claims_prover<B: ShortBaseField>(
    pp: &ShortQaFoldParameters<B>,
    opening: &ShortQaFoldVerificationReduction<B::Extension>,
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<Vec<Vec<ShortQaMatrixClaim<B::Extension>>>, Error> {
    validate_opening_pending_order(pp, opening)?;
    absorb_opening_statement_writer(pp, opening, transcript)?;
    let levels = pp.sequence.codes().len();
    let mut by_level = (0..levels).map(|_| Vec::new()).collect::<Vec<_>>();
    for pending in &opening.pending_h_claims {
        let code = pp
            .sequence
            .codes()
            .get(pending.level)
            .ok_or(Error::EncodingError)?;
        let generator_claim = ShortQaMatrixClaim {
            value: pending.value,
            point: (pending.left_point.clone(), pending.right_point.clone()),
        };
        // This is the deterministic algebraic G-to-H reduction. The opening
        // protocol already produced the G claim, and the Decide proof below
        // authenticates the resulting H claim, so re-evaluating the full
        // public generator here would add linear prover work without adding a
        // verifier check.
        let reduction = ShortQaCode::<B>::verify_as_subprotocol(
            code.message_len(),
            code.index(),
            &generator_claim,
        )?;
        by_level[pending.level].push(reduction.alpha_claim);
    }
    Ok(by_level)
}

/// Reconstructs the same `G_j -> H_j` reductions without reading `H_j`.
pub(super) fn prepare_pending_h_claims_verifier<B: ShortBaseField>(
    pp: &ShortQaFoldParameters<B>,
    opening: &ShortQaFoldVerificationReduction<B::Extension>,
    transcript: &mut ShortTranscriptReader<'_, B>,
) -> Result<Vec<Vec<ShortQaMatrixClaim<B::Extension>>>, Error> {
    validate_opening_pending_order(pp, opening)?;
    absorb_opening_statement_reader(pp, opening, transcript)?;
    let levels = pp.sequence.codes().len();
    let mut by_level = (0..levels).map(|_| Vec::new()).collect::<Vec<_>>();
    for pending in &opening.pending_h_claims {
        let code = pp
            .sequence
            .codes()
            .get(pending.level)
            .ok_or(Error::EncodingError)?;
        let generator_claim = ShortQaMatrixClaim {
            value: pending.value,
            point: (pending.left_point.clone(), pending.right_point.clone()),
        };
        let reduction = ShortQaCode::<B>::verify_as_subprotocol(
            code.message_len(),
            code.index(),
            &generator_claim,
        )?;
        if !reduction.ok {
            return Err(Error::InvalidCommitment);
        }
        by_level[pending.level].push(reduction.alpha_claim);
    }
    Ok(by_level)
}

fn aligned_t_widths(rows: usize, diagonal_lengths: &[usize]) -> Result<Vec<usize>, Error> {
    if rows == 0 || !rows.is_power_of_two() {
        return Err(Error::EncodingError);
    }
    let mut prefix = 0usize;
    let mut widths = Vec::with_capacity(diagonal_lengths.len());
    for &length in diagonal_lengths {
        if length == 0 || !length.is_power_of_two() || length % rows != 0 || prefix % rows != 0 {
            return Err(Error::InvalidNumberOfVariables);
        }
        widths.push(length / rows);
        prefix = prefix.checked_add(length).ok_or(Error::EncodingError)?;
    }
    if prefix % rows != 0 {
        return Err(Error::InvalidNumberOfVariables);
    }
    Ok(widths)
}

/// Embeds aligned incoming RFC-diagonal claims into global packed `T_j` columns.
pub(super) fn packed_short_t_eval_claims<B: ShortBaseField>(
    pp: &ShortQaFoldParameters<B>,
    setup_level: &ShortQaFoldSetupLevel<B>,
    level: usize,
    t_claims: &[ShortQaRfcTClaim<B::Extension>],
) -> Result<(Vec<ShortEvalClaim<B::Extension>>, usize), Error> {
    let levels = pp.sequence.codes().len();
    if level == 0 || level >= levels || setup_level.shape.level != level {
        return Err(Error::EncodingError);
    }
    if pp.derived_parameters.is_direct_interleaved() {
        if !t_claims.is_empty()
            || !setup_level.t_columns.is_empty()
            || setup_level.shape.t_cols != 0
            || setup_level.shape.t_elements != 0
            || setup_level.shape.t_last_len != 0
        {
            return Err(Error::InvalidNumberOfVariables);
        }
        return Ok((Vec::new(), 0));
    }
    if level + 1 == levels {
        if !t_claims.is_empty() || !setup_level.t_columns.is_empty() {
            return Err(Error::InvalidNumberOfVariables);
        }
        return Ok((Vec::new(), 0));
    }

    let rows = pp.sequence.codes()[level].message_len();
    let incoming = pp.derived_parameters.code(level + 1)?;
    if t_claims.len() != incoming.layers() {
        return Err(Error::InvalidNumberOfVariables);
    }
    let layers = (1..=incoming.layers()).rev().collect::<Vec<_>>();
    let lengths = layers
        .iter()
        .map(|&layer| incoming.diagonal(layer).map(<[B]>::len))
        .collect::<Result<Vec<_>, Error>>()?;
    let widths = aligned_t_widths(rows, &lengths)?;
    let packed_cols = widths.iter().try_fold(0usize, |sum, width| {
        sum.checked_add(*width).ok_or(Error::EncodingError)
    })?;
    let total_elements = lengths.iter().try_fold(0usize, |sum, length| {
        sum.checked_add(*length).ok_or(Error::EncodingError)
    })?;
    if setup_level.t_columns.len() != packed_cols
        || setup_level.shape.t_cols != packed_cols
        || setup_level.shape.t_elements != total_elements
        || setup_level.shape.t_last_len != rows
        || setup_level
            .t_columns
            .iter()
            .any(|column| column.len() != rows)
    {
        return Err(Error::EncodingError);
    }

    let log_rows = rows.ilog2() as usize;
    let mut offset = 0usize;
    let mut claims = Vec::with_capacity(t_claims.len());
    for (((&layer, &length), &width), t_claim) in
        layers.iter().zip(&lengths).zip(&widths).zip(t_claims)
    {
        if t_claim.layer != layer
            || t_claim.claim.point.len() != length.ilog2() as usize
            || !width.is_power_of_two()
        {
            return Err(Error::InvalidNumberOfVariables);
        }
        let log_width = width.ilog2() as usize;
        if t_claim.claim.point.len() != log_width + log_rows {
            return Err(Error::InvalidNumberOfVariables);
        }
        let mut col_weight = vec![B::Extension::zero(); offset + width];
        col_weight[offset..].copy_from_slice(&tensor_weights(&t_claim.claim.point[..log_width]));
        claims.push(ShortEvalClaim {
            value: t_claim.claim.value,
            row_point: t_claim.claim.point[log_width..].to_vec(),
            col_weight,
        });
        offset = offset.checked_add(width).ok_or(Error::EncodingError)?;
    }
    if offset != packed_cols {
        return Err(Error::EncodingError);
    }
    Ok((claims, packed_cols))
}

/// Constructs the common `T_j || H_j || W_j` evaluation claims for one level.
pub(super) fn prepare_short_decide_eval_claims<B: ShortBaseField>(
    pp: &ShortQaFoldParameters<B>,
    setup_level: &ShortQaFoldSetupLevel<B>,
    level: usize,
    h_claims: &[ShortQaMatrixClaim<B::Extension>],
    t_claims: &[ShortQaRfcTClaim<B::Extension>],
    w_claim: Option<&ShortEvalWithPoint<B::Extension>>,
) -> Result<ShortQaPreparedDecideClaims<B::Extension>, Error> {
    let code = pp.sequence.codes().get(level).ok_or(Error::EncodingError)?;
    let rows = code.message_len();
    let log_rows = rows.ilog2() as usize;
    setup_level.shape.validate()?;
    let (mut claims, packed_t_cols) = packed_short_t_eval_claims(pp, setup_level, level, t_claims)?;
    let setup_cols = setup_level
        .t_columns
        .len()
        .checked_add(code.alphas().len())
        .ok_or(Error::EncodingError)?;
    if pp.derived_parameters.is_direct_interleaved() {
        let target = pp
            .sequence
            .codes()
            .get(level - 1)
            .ok_or(Error::EncodingError)?;
        let row_factor = pp.sequence.switch_ratio();
        if packed_t_cols != 0
            || setup_level.shape.message_rows != target.message_len()
            || setup_level.shape.encoded_rows != target.codeword_len()?
            || setup_level.shape.t_cols != 0
            || setup_level.shape.h_cols
                != code
                    .alphas()
                    .len()
                    .checked_mul(row_factor)
                    .ok_or(Error::EncodingError)?
            || rows
                != target
                    .message_len()
                    .checked_mul(row_factor)
                    .ok_or(Error::EncodingError)?
            || setup_cols != code.alphas().len()
        {
            return Err(Error::EncodingError);
        }
    } else if setup_level.shape.message_rows != rows
        || setup_level.shape.t_cols != packed_t_cols
        || setup_level.shape.h_cols != code.alphas().len()
        || setup_level.shape.total_cols()? != setup_cols
    {
        return Err(Error::EncodingError);
    }

    for h_claim in h_claims {
        if h_claim.point.0.len() != log_rows
            || h_claim.point.1.len() != code.alphas().len().ilog2() as usize
        {
            return Err(Error::InvalidNumberOfVariables);
        }
        let mut col_weight = vec![B::Extension::zero(); packed_t_cols];
        col_weight.extend(tensor_weights(&h_claim.point.1));
        claims.push(ShortEvalClaim {
            value: h_claim.value,
            row_point: h_claim.point.0.clone(),
            col_weight,
        });
    }

    let carry_cols = if let Some(w_claim) = w_claim {
        let mut col_weight = vec![B::Extension::zero(); setup_cols];
        if pp.derived_parameters.is_direct_interleaved() {
            if w_claim.point.len() != log_rows {
                return Err(Error::InvalidNumberOfVariables);
            }
            col_weight.push(B::Extension::one());
            claims.push(ShortEvalClaim {
                value: w_claim.value,
                row_point: w_claim.point.clone(),
                col_weight,
            });
            1
        } else {
            // The merged top opening hands over one full-length RFC column.
            // Later RFC rounds split the previous witness into switch-ratio
            // columns, matching the separate Decide pipeline.
            let carry_cols =
                if level + 1 == pp.sequence.codes().len() && w_claim.point.len() == log_rows {
                    1
                } else {
                    pp.sequence.switch_ratio()
                };
            let log_carry_cols = carry_cols.ilog2() as usize;
            if w_claim.point.len() != log_carry_cols + log_rows {
                return Err(Error::InvalidNumberOfVariables);
            }
            col_weight.extend(tensor_weights(&w_claim.point[..log_carry_cols]));
            claims.push(ShortEvalClaim {
                value: w_claim.value,
                row_point: w_claim.point[log_carry_cols..].to_vec(),
                col_weight,
            });
            carry_cols
        }
    } else {
        0
    };
    if claims.is_empty() {
        return Err(Error::InvalidNumberOfVariables);
    }
    let real_cols = setup_cols
        .checked_add(carry_cols)
        .ok_or(Error::EncodingError)?;
    Ok(ShortQaPreparedDecideClaims {
        claims,
        log_rows,
        log_cols: real_cols.next_power_of_two().ilog2() as usize,
        setup_cols,
        carry_cols,
    })
}

/// Prepared mixed-matrix claims and their logical padded shape.
pub(super) struct ShortQaPreparedDecideClaims<E: ShortFieldElement> {
    pub(super) claims: Vec<ShortEvalClaim<E>>,
    pub(super) log_rows: usize,
    pub(super) log_cols: usize,
    pub(super) setup_cols: usize,
    pub(super) carry_cols: usize,
}
