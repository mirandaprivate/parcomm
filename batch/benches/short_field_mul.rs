use std::{hint::black_box, time::Instant};

use anyhow::{ensure, Context, Result};
use ark_bn254::Fr;
use ark_ff::{AdditiveGroup, PrimeField, Zero};
use rayon::prelude::*;

const STRIDE: usize = 64;
const WINDOW_BITS: usize = 4;
const WINDOW_SIZE: usize = 1 << WINDOW_BITS;
const WINDOWS: usize = 2;

fn main() -> Result<()> {
    let mut rows_log = 18_usize;
    let mut repetitions = 5_usize;
    let mut full_vs_short_only = false;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        if argument == "--bench" {
            continue;
        }
        if argument == "--full-vs-short-only" {
            full_vs_short_only = true;
            continue;
        }
        let value = arguments
            .next()
            .with_context(|| format!("missing value after {argument}"))?;
        match argument.as_str() {
            "--rows-log" => rows_log = value.parse()?,
            "--repetitions" => repetitions = value.parse()?,
            _ => anyhow::bail!("unknown argument {argument}"),
        }
    }
    ensure!(rows_log < usize::BITS as usize, "row log is too large");
    ensure!(repetitions != 0, "repetitions must be non-zero");

    let row_count = 1_usize << rows_log;
    let entry_count = row_count * STRIDE;
    let short_values = (0..entry_count)
        .map(|index| ((index.wrapping_mul(29) % 251) as i16 - 125) as i8)
        .collect::<Vec<_>>();
    let field_values = short_values
        .iter()
        .copied()
        .map(signed_i8_to_field)
        .collect::<Vec<_>>();
    let coefficients = (0..row_count)
        .map(|row| Fr::from((row.wrapping_mul(17) + 3) as u64))
        .collect::<Vec<_>>();

    let expected = project_preconverted(&field_values, &coefficients);
    ensure!(
        project_current_short(&short_values, &coefficients) == expected,
        "current short projection differs from the preconverted projection"
    );
    ensure!(
        project_direct_short(&short_values, &coefficients) == expected,
        "direct short projection differs from the preconverted projection"
    );
    ensure!(
        project_lazy_short(&short_values, &coefficients) == expected,
        "lazy short projection differs from the preconverted projection"
    );

    if full_vs_short_only {
        let mut full_width_samples = Vec::with_capacity(repetitions);
        let mut short_width_samples = Vec::with_capacity(repetitions);
        for repetition in 0..repetitions {
            if repetition % 2 == 0 {
                full_width_samples
                    .push(time(|| project_preconverted(&field_values, &coefficients)));
                short_width_samples.push(time(|| project_lazy_short(&short_values, &coefficients)));
            } else {
                short_width_samples.push(time(|| project_lazy_short(&short_values, &coefficients)));
                full_width_samples
                    .push(time(|| project_preconverted(&field_values, &coefficients)));
            }
        }
        let full_width = median(&full_width_samples);
        let short_width = median(&short_width_samples);
        println!("benchmark=full-width-vs-short-width-projection");
        println!("rayon_threads={}", rayon::current_num_threads());
        println!("rows={row_count}");
        println!("stride={STRIDE}");
        println!("entries={entry_count}");
        println!("repetitions={repetitions}");
        println!("outputs_equal=true");
        print_samples("full_width_fr_times_fr", &full_width_samples, entry_count);
        print_samples("short_width_i8_times_fr", &short_width_samples, entry_count);
        println!("short_width_speedup={:.6}", full_width / short_width);
        return Ok(());
    }

    let mut preconverted_samples = Vec::with_capacity(repetitions);
    let mut current_short_samples = Vec::with_capacity(repetitions);
    let mut direct_short_samples = Vec::with_capacity(repetitions);
    let mut lazy_short_samples = Vec::with_capacity(repetitions);
    for repetition in 0..repetitions {
        match repetition % 3 {
            0 => {
                preconverted_samples
                    .push(time(|| project_preconverted(&field_values, &coefficients)));
                current_short_samples
                    .push(time(|| project_current_short(&short_values, &coefficients)));
                direct_short_samples
                    .push(time(|| project_direct_short(&short_values, &coefficients)));
                lazy_short_samples.push(time(|| project_lazy_short(&short_values, &coefficients)));
            }
            1 => {
                direct_short_samples
                    .push(time(|| project_direct_short(&short_values, &coefficients)));
                preconverted_samples
                    .push(time(|| project_preconverted(&field_values, &coefficients)));
                current_short_samples
                    .push(time(|| project_current_short(&short_values, &coefficients)));
                lazy_short_samples.push(time(|| project_lazy_short(&short_values, &coefficients)));
            }
            _ => {
                current_short_samples
                    .push(time(|| project_current_short(&short_values, &coefficients)));
                direct_short_samples
                    .push(time(|| project_direct_short(&short_values, &coefficients)));
                preconverted_samples
                    .push(time(|| project_preconverted(&field_values, &coefficients)));
                lazy_short_samples.push(time(|| project_lazy_short(&short_values, &coefficients)));
            }
        }
    }

    let preconverted = median(&preconverted_samples);
    let current_short = median(&current_short_samples);
    let direct_short = median(&direct_short_samples);
    let lazy_short = median(&lazy_short_samples);
    println!("benchmark=short-field-multiplication");
    println!("rayon_threads={}", rayon::current_num_threads());
    println!("rows={row_count}");
    println!("stride={STRIDE}");
    println!("entries={entry_count}");
    println!("repetitions={repetitions}");
    println!("outputs_equal=true");
    print_samples(
        "preconverted_full_width",
        &preconverted_samples,
        entry_count,
    );
    print_samples("current_i8_to_field", &current_short_samples, entry_count);
    print_samples("direct_short_width", &direct_short_samples, entry_count);
    print_samples("lazy_short_width", &lazy_short_samples, entry_count);
    println!(
        "direct_vs_current_speedup={:.6}",
        current_short / direct_short
    );
    println!(
        "direct_vs_preconverted_speedup={:.6}",
        preconverted / direct_short
    );
    println!("lazy_vs_current_speedup={:.6}", current_short / lazy_short);
    println!(
        "lazy_vs_preconverted_speedup={:.6}",
        preconverted / lazy_short
    );
    Ok(())
}

fn project_preconverted(values: &[Fr], coefficients: &[Fr]) -> [Fr; STRIDE] {
    project_rows(coefficients.len(), |row, lane| {
        values[row * STRIDE + lane] * coefficients[row]
    })
}

fn project_current_short(values: &[i8], coefficients: &[Fr]) -> [Fr; STRIDE] {
    project_rows(coefficients.len(), |row, lane| {
        signed_i8_to_field(values[row * STRIDE + lane]) * coefficients[row]
    })
}

fn project_direct_short(values: &[i8], coefficients: &[Fr]) -> [Fr; STRIDE] {
    let task_count = coefficients.len().min(rayon::current_num_threads().max(1));
    (0..task_count)
        .into_par_iter()
        .map(|task| {
            let mut projected = [Fr::zero(); STRIDE];
            for row in (task..coefficients.len()).step_by(task_count) {
                let multiples = short_window_multiples(coefficients[row]);
                let values = &values[row * STRIDE..(row + 1) * STRIDE];
                for (lane, &value) in values.iter().enumerate() {
                    add_short_product(&mut projected[lane], value, &multiples);
                }
            }
            projected
        })
        .reduce(
            || [Fr::zero(); STRIDE],
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            },
        )
}

fn project_lazy_short(values: &[i8], coefficients: &[Fr]) -> [Fr; STRIDE] {
    let limb_count = coefficients[0].into_bigint().as_ref().len();
    let task_count = coefficients.len().min(rayon::current_num_threads().max(1));
    (0..task_count)
        .into_par_iter()
        .map(|task| {
            let mut positive = vec![0_u128; STRIDE * limb_count];
            let mut negative = vec![0_u128; STRIDE * limb_count];
            for row in (task..coefficients.len()).step_by(task_count) {
                let coefficient = coefficients[row].into_bigint();
                let coefficient_limbs = coefficient.as_ref();
                let values = &values[row * STRIDE..(row + 1) * STRIDE];
                for (lane, &value) in values.iter().enumerate() {
                    let magnitude = u128::from(value.unsigned_abs());
                    if magnitude == 0 {
                        continue;
                    }
                    let destination = if value < 0 {
                        &mut negative
                    } else {
                        &mut positive
                    };
                    let destination = &mut destination[lane * limb_count..(lane + 1) * limb_count];
                    for (accumulator, &limb) in destination.iter_mut().zip(coefficient_limbs) {
                        *accumulator += u128::from(limb) * magnitude;
                    }
                }
            }
            let mut projected = [Fr::zero(); STRIDE];
            for lane in 0..STRIDE {
                let start = lane * limb_count;
                let end = start + limb_count;
                projected[lane] = accumulated_limbs_to_field(&positive[start..end])
                    - accumulated_limbs_to_field(&negative[start..end]);
            }
            projected
        })
        .reduce(
            || [Fr::zero(); STRIDE],
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            },
        )
}

fn accumulated_limbs_to_field(accumulated: &[u128]) -> Fr {
    let mut bytes = Vec::with_capacity((accumulated.len() + 1) * 8);
    let mut carry = 0_u128;
    for &limb in accumulated {
        let normalized = limb + carry;
        bytes.extend_from_slice(&(normalized as u64).to_le_bytes());
        carry = normalized >> 64;
    }
    while carry != 0 {
        bytes.extend_from_slice(&(carry as u64).to_le_bytes());
        carry >>= 64;
    }
    Fr::from_le_bytes_mod_order(&bytes)
}

fn project_rows(row_count: usize, product: impl Fn(usize, usize) -> Fr + Sync) -> [Fr; STRIDE] {
    let task_count = row_count.min(rayon::current_num_threads().max(1));
    (0..task_count)
        .into_par_iter()
        .map(|task| {
            let mut projected = [Fr::zero(); STRIDE];
            for row in (task..row_count).step_by(task_count) {
                for (lane, projection) in projected.iter_mut().enumerate() {
                    *projection += product(row, lane);
                }
            }
            projected
        })
        .reduce(
            || [Fr::zero(); STRIDE],
            |mut left, right| {
                for (left, right) in left.iter_mut().zip(right) {
                    *left += right;
                }
                left
            },
        )
}

fn short_window_multiples(coefficient: Fr) -> [[Fr; WINDOW_SIZE]; WINDOWS] {
    let mut multiples = [[Fr::zero(); WINDOW_SIZE]; WINDOWS];
    let mut window_unit = coefficient;
    for (window, values) in multiples.iter_mut().enumerate() {
        values[1] = window_unit;
        for digit in 2..WINDOW_SIZE {
            values[digit] = values[digit - 1] + window_unit;
        }
        if window + 1 < WINDOWS {
            for _ in 0..WINDOW_BITS {
                window_unit.double_in_place();
            }
        }
    }
    multiples
}

#[inline]
fn add_short_product(accumulator: &mut Fr, value: i8, multiples: &[[Fr; WINDOW_SIZE]; WINDOWS]) {
    let subtract = value < 0;
    let mut magnitude = u64::from(value.unsigned_abs());
    let mut window = 0_usize;
    while magnitude != 0 {
        let digit = (magnitude & (WINDOW_SIZE as u64 - 1)) as usize;
        if digit != 0 {
            if subtract {
                *accumulator -= multiples[window][digit];
            } else {
                *accumulator += multiples[window][digit];
            }
        }
        magnitude >>= WINDOW_BITS;
        window += 1;
    }
}

#[inline]
fn signed_i8_to_field(value: i8) -> Fr {
    if value >= 0 {
        Fr::from(value as u64)
    } else {
        -Fr::from((-(value as i16)) as u64)
    }
}

fn time(run: impl FnOnce() -> [Fr; STRIDE]) -> f64 {
    let started = Instant::now();
    black_box(run());
    started.elapsed().as_secs_f64()
}

fn median(samples: &[f64]) -> f64 {
    let mut samples = samples.to_vec();
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

fn print_samples(label: &str, samples: &[f64], entry_count: usize) {
    let seconds = median(samples);
    println!(
        "{label}_samples_seconds={}",
        samples
            .iter()
            .map(|value| format!("{value:.6}"))
            .collect::<Vec<_>>()
            .join(",")
    );
    println!("{label}_seconds={seconds:.6}");
    println!(
        "{label}_nanoseconds_per_entry={:.3}",
        seconds * 1e9 / entry_count as f64
    );
}
