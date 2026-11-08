//! QA distance helpers extracted from Brakefold.
use crate::Error;

pub const QAFOLD_TOP_INDEX: usize = 2;
pub const QAFOLD_RECURSIVE_INDEX: usize = 2;

#[derive(Clone, Copy, Debug)]
pub struct QaDistanceBound {
    /// Guaranteed relative distance.
    pub relative_distance: f64,
    /// Positive entropy margin at the selected distance.
    pub epsilon: f64,
    /// Base-two logarithm of the smaller failure-probability bound.
    pub log2_failure_probability: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct QaRfDistanceBound {
    /// Finite-length distance bound of the underlying QA code.
    pub qa: QaDistanceBound,
    /// Relative distance after all RF extensions in this switch.
    pub relative_distance: f64,
    /// Number of RF extensions applied.
    pub rf_layers: usize,
}

pub fn qa_entropy_distance_bound_with_modulus_log2(
    message_len: usize,
    index: usize,
    modulus_log2: f64,
) -> Result<QaDistanceBound, Error> {
    if message_len == 0
        || !message_len.is_power_of_two()
        || index <= 1
        || !index.is_power_of_two()
        || !modulus_log2.is_finite()
        || modulus_log2 <= 0.0
    {
        return Err(Error::EncodingError);
    }
    let ln_p = modulus_log2 * core::f64::consts::LN_2;
    let upper = 1.0 - 1.0 / index as f64;
    let smallest = f64::EPSILON;
    if qa_epsilon_at(message_len, index, smallest, ln_p).is_none() {
        return Err(Error::InvalidParameters(
            "the QA entropy condition has no positive distance".into(),
        ));
    }

    let mut lower = smallest;
    let mut upper_bracket = upper;
    if qa_epsilon_at(message_len, index, upper, ln_p).is_some() {
        lower = upper;
    } else {
        for _ in 0..96 {
            let middle = (lower + upper_bracket) * 0.5;
            if qa_epsilon_at(message_len, index, middle, ln_p).is_some() {
                lower = middle;
            } else {
                upper_bracket = middle;
            }
        }
    }

    let relative_distance = lower * (1.0 - 1e-12);
    let (epsilon, log_failure) =
        qa_bound_at(message_len, index, relative_distance, ln_p).ok_or(Error::EncodingError)?;
    Ok(QaDistanceBound {
        relative_distance,
        epsilon,
        log2_failure_probability: log_failure / core::f64::consts::LN_2,
    })
}

pub fn qa_distance_bound_with_modulus_log2(
    message_len: usize,
    index: usize,
    security_bits: usize,
    modulus_log2: f64,
) -> Result<QaDistanceBound, Error> {
    if message_len == 0
        || !message_len.is_power_of_two()
        || index <= 1
        || !index.is_power_of_two()
        || security_bits == 0
        || !modulus_log2.is_finite()
        || modulus_log2 <= 0.0
    {
        return Err(Error::EncodingError);
    }
    let ln_p = modulus_log2 * core::f64::consts::LN_2;
    let upper = 1.0 - 1.0 / index as f64;
    let target_log_failure = -(security_bits as f64) * core::f64::consts::LN_2;
    let valid = |distance: f64| {
        qa_bound_at(message_len, index, distance, ln_p)
            .filter(|(_, log_failure)| *log_failure <= target_log_failure)
    };

    let scan_steps = 4096usize;
    let mut lower = 0.0f64;
    let mut upper_bracket = upper;
    for step in 1..=scan_steps {
        let candidate = upper * step as f64 / scan_steps as f64;
        if valid(candidate).is_some() {
            lower = candidate;
        } else if candidate > lower {
            upper_bracket = candidate;
            break;
        }
    }
    if lower == 0.0 {
        return Err(Error::InvalidParameters(
            "the QA distance bound does not meet the requested security".into(),
        ));
    }
    for _ in 0..96 {
        let middle = (lower + upper_bracket) * 0.5;
        if valid(middle).is_some() {
            lower = middle;
        } else {
            upper_bracket = middle;
        }
    }
    let relative_distance = lower * (1.0 - 1e-12);
    let (epsilon, log_failure) =
        qa_bound_at(message_len, index, relative_distance, ln_p).ok_or(Error::EncodingError)?;
    Ok(QaDistanceBound {
        relative_distance,
        epsilon,
        log2_failure_probability: log_failure / core::f64::consts::LN_2,
    })
}

fn qa_bound_at(message_len: usize, index: usize, distance: f64, ln_p: f64) -> Option<(f64, f64)> {
    let epsilon = qa_epsilon_at(message_len, index, distance, ln_p)?;
    let ln_p_minus_one = ln_p + (-(-ln_p).exp()).ln_1p();
    let denominator_log = log_one_minus_exp(-(index as f64) * epsilon * ln_p)? + ln_p_minus_one;
    let first_collision = index as f64 * (index - 1) as f64 * message_len as f64 / 2.0;
    let first_collision_log = first_collision.ln() - 2.0 * ln_p;
    let first_exponent =
        ((index - 1) as f64 / (index as f64 * distance)).ceil() * index as f64 * epsilon;
    let first_tail_log = -first_exponent * ln_p - denominator_log;
    let first_failure_log = log_add_exp(first_collision_log, first_tail_log);

    let second_collision_log = (index as f64 * message_len as f64).ln() - ln_p;
    let second_exponent = (1.0 / distance).ceil() * index as f64 * epsilon;
    let second_tail_log = -second_exponent * ln_p - denominator_log;
    let second_failure_log = log_add_exp(second_collision_log, second_tail_log);
    Some((epsilon, first_failure_log.min(second_failure_log)))
}

fn qa_epsilon_at(message_len: usize, index: usize, distance: f64, ln_p: f64) -> Option<f64> {
    if message_len == 0 || index <= 1 || !(distance > 0.0 && distance < 1.0) || !(ln_p > 0.0) {
        return None;
    }
    let ln_p_minus_one = ln_p + (-(-ln_p).exp()).ln_1p();
    let g = 1.0 - distance * ln_p_minus_one / ln_p
        + distance * distance.ln() / ln_p
        + (1.0 - distance) * (1.0 - distance).ln() / ln_p;
    let epsilon = g - (1.0 + (message_len as f64).ln() / ln_p) / index as f64;
    (epsilon > 0.0).then_some(epsilon)
}

fn log_one_minus_exp(non_positive: f64) -> Option<f64> {
    if non_positive >= 0.0 {
        return None;
    }
    if non_positive < -core::f64::consts::LN_2 {
        Some((-non_positive.exp()).ln_1p())
    } else {
        Some((-non_positive.exp_m1()).ln())
    }
}

fn log_add_exp(left: f64, right: f64) -> f64 {
    let maximum = left.max(right);
    maximum + ((left - maximum).exp() + (right - maximum).exp()).ln()
}
