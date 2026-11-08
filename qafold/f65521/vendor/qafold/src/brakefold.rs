//! RF distance helpers extracted from Brakefold.
use crate::Error;

pub fn queries_for_relative_distance(security_bits: usize, distance: f64) -> Result<usize, Error> {
    if security_bits == 0 {
        return Err(Error::EncodingError);
    }
    validate_relative_distance(distance)?;
    let soundness_base = 1.0 - distance / 3.0;
    if !(soundness_base > 0.0 && soundness_base < 1.0) {
        return Err(Error::EncodingError);
    }
    let queries = -(security_bits as f64) / soundness_base.log2();
    if !queries.is_finite() || queries < 0.0 {
        return Err(Error::EncodingError);
    }
    Ok(queries.ceil() as usize)
}

pub fn rf_extended_relative_distance(
    base_distance: f64,
    base_codeword_len: usize,
    rf_layers: usize,
    field_log2: f64,
    security_bits: usize,
) -> Result<f64, Error> {
    validate_relative_distance(base_distance)?;
    if base_codeword_len == 0
        || !field_log2.is_finite()
        || field_log2 <= 1.001
        || security_bits == 0
    {
        return Err(Error::EncodingError);
    }
    let denominator = field_log2 - 1.001;
    let mut distance = base_distance;
    let mut codeword_len = base_codeword_len;
    for _ in 0..rf_layers {
        let n = codeword_len as f64;
        let zero_bound = (1.0 - distance) * n + 1.0;
        let distance_loss = 1.0 / n
            + (security_bits as f64 + 2.002 * zero_bound + 1.2 * n) / (2.0 * n * denominator);
        distance -= distance_loss;
        if !distance.is_finite() || distance <= 0.0 {
            return Err(Error::InvalidParameters(
                "RF extensions eliminate the available distance bound".into(),
            ));
        }
        codeword_len = codeword_len.checked_mul(2).ok_or(Error::EncodingError)?;
    }
    validate_relative_distance(distance)?;
    Ok(distance)
}

pub fn rf_layers_for_switch(
    switch_ratio: usize,
    first_packed_switch: bool,
) -> Result<usize, Error> {
    if switch_ratio <= 1 || !switch_ratio.is_power_of_two() {
        return Err(Error::EncodingError);
    }
    let layers = switch_ratio.ilog2() as usize;
    if first_packed_switch {
        layers.checked_add(1).ok_or(Error::EncodingError)
    } else {
        Ok(layers)
    }
}

fn validate_relative_distance(distance: f64) -> Result<(), Error> {
    if distance.is_finite() && distance > 0.0 && distance < 1.0 {
        Ok(())
    } else {
        Err(Error::EncodingError)
    }
}
