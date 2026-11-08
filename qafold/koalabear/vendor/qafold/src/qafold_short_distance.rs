//! Appendix B.8 bounds for QA generators sampled with nonzero frequencies.
//!
//! Normalizing all generators by the first (invertible) generator gives the
//! same code with a systematic first block. The remaining frequency vectors
//! are still independent and uniform over the nonzero base-field elements.

use crate::Error;
use crate::qafold::QaDistanceBound;

/// Reserve a factor 256 for code-generation and proximity events across levels.
/// A usize-sized witness has at most 64 levels and at most two row-query
/// batches per level. Algebraic/transcript errors require their own analysis.
pub const NONZERO_QA_SECURITY_SLACK_BITS: usize = 8;

fn validate(message_len: usize, index: usize, modulus: u64) -> Result<(), Error> {
    if message_len == 0
        || !message_len.is_power_of_two()
        || index <= 1
        || !index.is_power_of_two()
        || modulus < 3
        || modulus % 2 == 0
    {
        return Err(Error::EncodingError);
    }
    Ok(())
}

/// Upper-bounds log2 Pr[relative distance < delta] using the B.8 binomial tail.
/// Negative infinity denotes the empty sum, hence a deterministic guarantee.
pub fn qa_nonzero_log2_failure_bound(
    message_len: usize,
    index: usize,
    distance: f64,
    modulus: u64,
) -> Result<f64, Error> {
    validate(message_len, index, modulus)?;
    let q = modulus as f64;
    if !distance.is_finite() || distance <= 0.0 || distance >= 1.0 - q.recip() {
        return Err(Error::EncodingError);
    }
    let lower_float = distance.recip().ceil();
    if lower_float > message_len as f64 {
        return Ok(f64::NEG_INFINITY);
    }
    let lower = lower_float as usize;
    let ln_q = q.ln();
    let ln_q_minus_one = ln_q + (-q.recip()).ln_1p();
    let h_times_ln_q = distance * ln_q_minus_one
        - distance * distance.ln()
        - (1.0 - distance) * (-distance).ln_1p();
    let log_a = index as f64 * h_times_ln_q - (index - 1) as f64 * ln_q_minus_one;
    let log_eta = if lower == message_len {
        f64::NEG_INFINITY
    } else {
        ((message_len - lower) as f64).ln() - ((lower + 1) as f64).ln() + log_a
    };
    if log_eta >= 0.0 {
        return Ok(0.0);
    }
    let terms = lower.min(message_len - lower);
    // Avoid unbounded parameter-selection work. Returning one is conservative.
    if terms > 4096 {
        return Ok(0.0);
    }
    let log_binomial = (1..=terms)
        .map(|j| ((message_len - j + 1) as f64).ln() - (j as f64).ln())
        .sum::<f64>();
    let log_first = log_binomial + lower as f64 * log_a - ln_q_minus_one;
    let log_one_minus_eta = if log_eta == f64::NEG_INFINITY {
        0.0
    } else {
        (-log_eta.exp_m1()).ln()
    };
    let log_bound = log_first - log_one_minus_eta;
    // Round conservatively away from the acceptance boundary. Reference tests
    // also compare this calculation with independently computed decimal values.
    let guard = 1e-9
        * (1.0
            + index as f64
            + ln_q
            + lower as f64
            + log_binomial.abs()
            + log_first.abs()
            + log_one_minus_eta.abs());
    Ok(((log_bound + guard) / core::f64::consts::LN_2).min(0.0))
}

/// Selects a finite-length distance meeting the requested code-generation bound.
pub fn qa_nonzero_distance_bound(
    message_len: usize,
    index: usize,
    security_bits: usize,
    modulus: u64,
) -> Result<QaDistanceBound, Error> {
    validate(message_len, index, modulus)?;
    if security_bits == 0 {
        return Err(Error::EncodingError);
    }
    let mut lower = 0.0;
    let mut upper = (1.0 - 1.0 / index as f64).min(1.0 - 1.0 / modulus as f64);
    for _ in 0..96 {
        let middle = (lower + upper) * 0.5;
        if middle == lower || middle == upper {
            break;
        }
        if qa_nonzero_log2_failure_bound(message_len, index, middle, modulus)?
            <= -(security_bits as f64)
        {
            lower = middle;
        } else {
            upper = middle;
        }
    }
    let relative_distance = lower * (1.0 - 1e-10);
    if relative_distance <= 0.0 {
        return Err(Error::InvalidParameters(
            "the B.8 QA distance does not meet the requested security".into(),
        ));
    }
    let log2_failure_probability =
        qa_nonzero_log2_failure_bound(message_len, index, relative_distance, modulus)?;
    if log2_failure_probability > -(security_bits as f64) {
        return Err(Error::InvalidParameters(
            "the B.8 QA distance failed its final probability check".into(),
        ));
    }
    let ln_q = (modulus as f64).ln();
    let ln_q_minus_one = ln_q + (-(modulus as f64).recip()).ln_1p();
    let delta = relative_distance;
    let h = (delta * ln_q_minus_one - delta * delta.ln() - (1.0 - delta) * (-delta).ln_1p()) / ln_q;
    let epsilon = (index - 1) as f64 / index as f64 * ln_q_minus_one / ln_q
        - h
        - (message_len as f64).ln() / (index as f64 * ln_q);
    Ok(QaDistanceBound {
        relative_distance,
        epsilon,
        log2_failure_probability,
    })
}
