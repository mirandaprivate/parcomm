use std::{
    io::{self, Write},
    mem::size_of,
    process::Command,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use anyhow::{bail, ensure, Context, Result};
use ark_bn254::Fr;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use batch::{
    bucketed_ip::packing_stride_for_bit_width,
    jolt_dory::{canonical_dory_log, setup_dory_parameters},
    matrix_collection::{constant_matrix_mle_evaluate, DenseMatrixBva},
    section6_data::{Section6Layout, Section6MatrixClaim},
    validity_protocol::{
        commit_opening_collection_profiled, prove_opening, prove_shape_validity_for_collection,
        validity_required_dory_log, verify_shape_commitment, verify_validity_opening,
        ValidityCommitStatement, ValidityOpeningProof,
    },
    validity_shape::ShapeValidityProof,
};
use bincode::Options;
use sha2::{Digest, Sha256};

const BIT_WIDTH: usize = 8;
const DEFAULT_MAX_LOG: usize = 15;
const MAX_SUPPORTED_LOG: usize = 17;
const MIN_LOG: usize = 2;
const DEFAULT_SEED: u64 = 0x5345_4354_494f_4e36;
const PROOF_MAGIC: [u8; 8] = *b"S6VALID1";

#[derive(Debug)]
struct Arguments {
    max_log: usize,
    seed: u64,
    allow_non_64_threads: bool,
    shape_validity_proof: bool,
    /// A collection of equally sized squares, instead of the decreasing family.
    uniform: Option<UniformFamily>,
}

/// `count` copies of a `(2^log-1)x(2^log-1)` square.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct UniformFamily {
    count: usize,
    log: usize,
}

fn parse_arguments() -> Result<Option<Arguments>> {
    parse_arguments_from(std::env::args().skip(1))
}

fn parse_arguments_from(arguments: impl IntoIterator<Item = String>) -> Result<Option<Arguments>> {
    let mut max_log = DEFAULT_MAX_LOG;
    let mut seed = DEFAULT_SEED;
    let mut allow_non_64_threads = false;
    let mut shape_validity_proof = false;
    let mut target_log = None;
    let mut uniform_count = None;
    let mut uniform_log = None;
    let mut arguments = arguments.into_iter();

    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--max-log" => {
                max_log = arguments
                    .next()
                    .context("missing value after --max-log")?
                    .parse()
                    .context("invalid --max-log value")?;
            }
            "--seed" => {
                seed = arguments
                    .next()
                    .context("missing value after --seed")?
                    .parse()
                    .context("invalid --seed value")?;
            }
            "--target-log" => {
                let value = arguments
                    .next()
                    .context("missing value after --target-log")?
                    .parse::<usize>()
                    .context("invalid --target-log value")?;
                target_log = Some(value);
            }
            "--uniform-count" => {
                uniform_count = Some(
                    arguments
                        .next()
                        .context("missing value after --uniform-count")?
                        .parse::<usize>()
                        .context("invalid --uniform-count value")?,
                );
            }
            "--uniform-log" => {
                uniform_log = Some(
                    arguments
                        .next()
                        .context("missing value after --uniform-log")?
                        .parse::<usize>()
                        .context("invalid --uniform-log value")?,
                );
            }
            "--interleaved-rectangles" | "--start-with-rectangle" => {
                // Compatibility flags describing the now-fixed Section 6 family.
            }
            "--allow-non-64-threads" => allow_non_64_threads = true,
            "--shape-validity-proof" | "--shape-validity" => shape_validity_proof = true,
            "-h" | "--help" => {
                print_help();
                return Ok(None);
            }
            _ if argument.starts_with("--max-log=") => {
                max_log = argument["--max-log=".len()..]
                    .parse()
                    .context("invalid --max-log value")?;
            }
            _ if argument.starts_with("--seed=") => {
                seed = argument["--seed=".len()..]
                    .parse()
                    .context("invalid --seed value")?;
            }
            _ if argument.starts_with("--target-log=") => {
                target_log = Some(
                    argument["--target-log=".len()..]
                        .parse()
                        .context("invalid --target-log value")?,
                );
            }
            _ if argument.starts_with("--uniform-count=") => {
                uniform_count = Some(
                    argument["--uniform-count=".len()..]
                        .parse()
                        .context("invalid --uniform-count value")?,
                );
            }
            _ if argument.starts_with("--uniform-log=") => {
                uniform_log = Some(
                    argument["--uniform-log=".len()..]
                        .parse()
                        .context("invalid --uniform-log value")?,
                );
            }
            _ => bail!("unknown argument {argument}; use --help for usage"),
        }
    }
    let uniform = match (uniform_count, uniform_log) {
        (Some(count), Some(log)) => {
            ensure!(count > 0, "uniform count must be positive");
            ensure!(
                (MIN_LOG + 1..=MAX_SUPPORTED_LOG).contains(&log),
                "uniform log must be in {}..={MAX_SUPPORTED_LOG}",
                MIN_LOG + 1
            );
            Some(UniformFamily { count, log })
        }
        (None, None) => None,
        _ => bail!("--uniform-count and --uniform-log must be given together"),
    };
    ensure!(
        (MIN_LOG + 1..=MAX_SUPPORTED_LOG).contains(&max_log),
        "max log must be in {}..={MAX_SUPPORTED_LOG}",
        MIN_LOG + 1
    );
    if let Some(target_log) = target_log {
        let expected_target_log = match uniform {
            Some(family) => {
                let count_log = family
                    .count
                    .checked_next_power_of_two()
                    .context("uniform count overflowed")?
                    .ilog2() as usize;
                count_log
                    .checked_add(
                        family
                            .log
                            .checked_mul(2)
                            .context("uniform matrix log overflowed")?,
                    )
                    .context("uniform target log overflowed")?
            }
            None => max_log.checked_mul(2).context("target log overflowed")?,
        };
        ensure!(
            target_log == expected_target_log,
            "workload requires target log {expected_target_log}"
        );
    }
    Ok(Some(Arguments {
        max_log,
        seed,
        allow_non_64_threads,
        shape_validity_proof,
        uniform,
    }))
}

fn print_help() {
    println!(
        "Usage: parcomm_e2e [OPTIONS]\n\
         \n\
         Options:\n\
           --max-log N              First rectangle is (2^N-1)x(2^(N-1)-1)\n\
           --seed N                 Deterministic prover RNG seed\n\
           --target-log N           Assert that the logical domain is 2^N\n\
           --uniform-count N        Use N equal squares instead of the family\n\
           --uniform-log L          Side of each uniform square is 2^L-1\n\
           --interleaved-rectangles Compatibility flag for the fixed family\n\
           --start-with-rectangle   Compatibility flag for the fixed family\n\
           --allow-non-64-threads   Permit a local smoke run with fewer threads\n\
           --shape-validity-proof   Include the optional shape validity proof\n\
           -h, --help               Print this help\n\
         \n\
         The full experiment is the default. Run it with:\n\
         RAYON_NUM_THREADS=64 cargo run --release --example parcomm_e2e"
    );
}

fn side(log: usize) -> Result<usize> {
    1_usize
        .checked_shl(log as u32)
        .and_then(|value| value.checked_sub(1))
        .context("matrix side overflowed")
}

/// Build a collection of equally sized squares.
///
/// The decreasing family below spends most of its entries in its first few
/// matrices; this one spreads them evenly, which exercises the same total
/// domain against many buckets of the same size.
fn uniform_shapes(family: UniformFamily) -> Result<Vec<(usize, usize)>> {
    let side = side(family.log)?;
    Ok(vec![(side, side); family.count])
}

/// Build the exact decreasing family requested for the Section 6 experiment.
fn section6_shapes(max_log: usize) -> Result<Vec<(usize, usize)>> {
    ensure!(max_log > MIN_LOG, "max log must exceed {MIN_LOG}");
    let side = |log: usize| side(log);
    let mut shapes = Vec::with_capacity(2 * (max_log - MIN_LOG));
    shapes.push((side(max_log)?, side(max_log - 1)?));
    for log in (MIN_LOG..max_log).rev() {
        shapes.push((side(log)?, side(log)?));
        if log > MIN_LOG {
            shapes.push((side(log)?, side(log - 1)?));
        }
    }
    Ok(shapes)
}

fn physical_entry_count(shapes: &[(usize, usize)]) -> Result<usize> {
    shapes.iter().try_fold(0_usize, |total, &(rows, cols)| {
        let entries = rows.checked_mul(cols).context("matrix size overflowed")?;
        total.checked_add(entries).context("total size overflowed")
    })
}

fn matrix_value(matrix: usize) -> i8 {
    let magnitude = (matrix % 31 + 1) as i8;
    if matrix.is_multiple_of(2) {
        magnitude
    } else {
        -magnitude
    }
}

fn claims_for_constant_matrices(
    shapes: &[(usize, usize)],
    point_log: usize,
) -> Result<Vec<Section6MatrixClaim<Fr>>> {
    shapes
        .iter()
        .enumerate()
        .map(|(matrix, &shape)| {
            let left_len = shape.0.next_power_of_two().ilog2() as usize;
            let right_len = shape.1.next_power_of_two().ilog2() as usize;
            let left = (0..left_len)
                .map(|coordinate| Fr::from((matrix * 37 + coordinate * 11 + 2) as u64))
                .collect::<Vec<_>>();
            let right = (0..right_len)
                .map(|coordinate| Fr::from((matrix * 41 + coordinate * 13 + 5) as u64))
                .collect::<Vec<_>>();
            let value = constant_matrix_mle_evaluate(shape, matrix_value(matrix), &left, &right)?;
            Section6MatrixClaim::from_unpadded(point_log, &left, &right, value).map_err(Into::into)
        })
        .collect()
}

fn proof_codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
}

fn encode_proof_artifact(
    statement: &ValidityCommitStatement,
    validity: Option<&ShapeValidityProof>,
    opening: &ValidityOpeningProof,
) -> Result<(Vec<u8>, [usize; 3])> {
    let components = [
        proof_codec().serialize(statement)?,
        validity
            .map(|proof| proof_codec().serialize(proof))
            .transpose()?
            .unwrap_or_default(),
        proof_codec().serialize(opening)?,
    ];
    let lengths = components.each_ref().map(|component| component.len());
    let payload_len = lengths.iter().try_fold(0_usize, |total, &length| {
        total
            .checked_add(length)
            .context("proof artifact overflowed")
    })?;
    let mut artifact = Vec::with_capacity(
        PROOF_MAGIC
            .len()
            .checked_add(3 * size_of::<u64>())
            .and_then(|header| header.checked_add(payload_len))
            .context("proof artifact overflowed")?,
    );
    artifact.extend_from_slice(&PROOF_MAGIC);
    for &length in &lengths {
        artifact.extend_from_slice(&(length as u64).to_le_bytes());
    }
    for component in components {
        artifact.extend_from_slice(&component);
    }
    Ok((artifact, lengths))
}

fn decode_proof_artifact_exact(
    artifact: &[u8],
) -> Result<(
    ValidityCommitStatement,
    Option<ShapeValidityProof>,
    ValidityOpeningProof,
)> {
    let header_len = PROOF_MAGIC.len() + 3 * size_of::<u64>();
    ensure!(
        artifact.len() >= header_len,
        "truncated proof artifact header"
    );
    ensure!(
        artifact[..PROOF_MAGIC.len()] == PROOF_MAGIC,
        "invalid proof artifact magic"
    );
    let mut lengths = [0_usize; 3];
    for (component, length) in lengths.iter_mut().enumerate() {
        let start = PROOF_MAGIC.len() + component * size_of::<u64>();
        let end = start + size_of::<u64>();
        let encoded: [u8; 8] = artifact[start..end]
            .try_into()
            .expect("length field has a fixed width");
        *length = usize::try_from(u64::from_le_bytes(encoded))
            .context("proof component length does not fit usize")?;
    }
    let expected_len = lengths.iter().try_fold(header_len, |total, &length| {
        total
            .checked_add(length)
            .context("proof artifact overflowed")
    })?;
    ensure!(
        artifact.len() == expected_len,
        "proof artifact has truncated or trailing bytes"
    );
    let statement_end = header_len + lengths[0];
    let validity_end = statement_end + lengths[1];
    let statement = proof_codec().deserialize(&artifact[header_len..statement_end])?;
    let validity = if lengths[1] == 0 {
        None
    } else {
        Some(proof_codec().deserialize(&artifact[statement_end..validity_end])?)
    };
    let opening = proof_codec().deserialize(&artifact[validity_end..expected_len])?;
    Ok((statement, validity, opening))
}

fn report_stage(name: &str, started: Instant, monitor: &MemoryMonitor) -> f64 {
    let seconds = started.elapsed().as_secs_f64();
    println!("{name}_seconds={seconds:.6}");
    println!("peak_rss_after_{name}_kib={}", monitor.peak_kib());
    io::stdout().flush().expect("flush stage result");
    seconds
}

fn main() -> Result<()> {
    let Some(arguments) = parse_arguments()? else {
        return Ok(());
    };
    let threads = rayon::current_num_threads();
    if !arguments.allow_non_64_threads {
        ensure!(
            threads == 64,
            "the formal experiment requires 64 Rayon threads, got {threads}; set RAYON_NUM_THREADS=64"
        );
    }

    let experiment_started = Instant::now();
    let monitor = MemoryMonitor::spawn(Duration::from_millis(100));
    let shapes = match arguments.uniform {
        Some(family) => uniform_shapes(family)?,
        None => section6_shapes(arguments.max_log)?,
    };
    let physical_entries = physical_entry_count(&shapes)?;

    let descriptor_started = Instant::now();
    let descriptor_only = batch::matrix_collection::DenseBvaDescriptor::from_shapes(&shapes)?;
    let layout = Section6Layout::from_descriptor(&descriptor_only, BIT_WIDTH)?;
    let descriptor_seconds = report_stage("descriptor", descriptor_started, &monitor);
    let stride = packing_stride_for_bit_width::<Fr>(layout.logical_len, BIT_WIDTH)?;
    let setup_log = validity_required_dory_log(&layout)?;

    if let Some(family) = arguments.uniform {
        if family
            == (UniformFamily {
                count: 1 << 14,
                log: 10,
            })
        {
            ensure!(
                physical_entries == 17_146_331_136,
                "unexpected homogeneous physical entry count"
            );
            ensure!(
                layout.active_len == 1_usize << 34,
                "unexpected homogeneous active BVA domain"
            );
            ensure!(
                layout.logical_len == 1_usize << 34,
                "unexpected homogeneous BVA domain"
            );
            ensure!(
                layout.bucket_count == 1_usize << 18,
                "unexpected homogeneous bucket count"
            );
        } else if family
            == (UniformFamily {
                count: 1 << 10,
                log: 12,
            })
        {
            ensure!(
                physical_entries == 17_171_481_600,
                "unexpected 2^34 homogeneous physical entry count"
            );
            ensure!(
                layout.active_len == 1_usize << 34,
                "unexpected 2^34 homogeneous active BVA domain"
            );
            ensure!(
                layout.logical_len == 1_usize << 34,
                "unexpected 2^34 homogeneous BVA domain"
            );
            ensure!(
                layout.bucket_count == 1_usize << 14,
                "unexpected 2^34 homogeneous bucket count"
            );
        }
    } else if arguments.max_log == DEFAULT_MAX_LOG {
        ensure!(shapes.len() == 26, "unexpected full-scale matrix count");
        ensure!(
            physical_entries == 1_073_578_014,
            "unexpected full-scale physical entry count"
        );
        ensure!(layout.logical_len == 1_usize << 30, "unexpected BVA domain");
        ensure!(
            layout.active_len == 1_073_741_689,
            "unexpected active BVA length"
        );
        ensure!(layout.bucket_count == 383, "unexpected bucket count");
    } else if arguments.max_log == MAX_SUPPORTED_LOG {
        ensure!(shapes.len() == 30, "unexpected 2^34 matrix count");
        ensure!(
            physical_entries == 17_179_213_858,
            "unexpected 2^34 physical entry count"
        );
        ensure!(layout.logical_len == 1_usize << 34, "unexpected BVA domain");
        ensure!(
            layout.active_len == 17_179_869_049,
            "unexpected active BVA length"
        );
        ensure!(layout.bucket_count == 447, "unexpected bucket count");
    }

    let packed_field_entries = layout.active_len.div_ceil(stride);
    let packed_field_bytes = packed_field_entries
        .checked_mul(size_of::<Fr>())
        .context("packed field byte estimate overflowed")?;
    let stored_i8_entries = layout.logical_len;
    let materialized_padding_entries = stored_i8_entries - physical_entries;
    let witness_plus_packed_bytes = stored_i8_entries
        .checked_add(packed_field_bytes)
        .context("combined memory estimate overflowed")?;

    println!("experiment=parcomm-section6-hidden-cpu");
    println!("protocol=section6-capped16-packed-eo-padded-i8-dory");
    println!(
        "shape_validity_proof_enabled={}",
        arguments.shape_validity_proof
    );
    println!("backend=jolt-dory-bn254-cpu");
    println!("rayon_threads={threads}");
    println!("formal_thread_requirement=64");
    println!("seed={}", arguments.seed);
    println!("bit_width={BIT_WIDTH}");
    println!(
        "splitting_budget={}",
        batch::matrix_collection::DenseBvaDescriptor::SPLITTING_BUDGET
    );
    println!("field_storage_bits={}", size_of::<Fr>() * 8);
    println!("kappa={stride}");
    match arguments.uniform {
        Some(family) => {
            println!("family=uniform");
            println!("uniform_count={}", family.count);
            println!("uniform_log={}", family.log);
            println!("shape_formula={0}x{0} repeated", side(family.log)?);
        }
        None => {
            println!("family=section6-decreasing");
            println!("max_log={}", arguments.max_log);
            println!("min_log={MIN_LOG}");
            println!(
                "shape_formula=(2^j-1)x(2^(j-1)-1),(2^(j-1)-1)^2,then decreasing rectangle/square pairs"
            );
        }
    }
    if arguments.uniform.is_none() {
        println!(
            "shape_sequence={}",
            shapes
                .iter()
                .map(|(rows, cols)| format!("{rows}x{cols}"))
                .collect::<Vec<_>>()
                .join(",")
        );
    }
    println!("matrix_count={}", shapes.len());
    println!("physical_entries={physical_entries}");
    println!("active_logical_entries={}", layout.active_len);
    println!("logical_entries={}", layout.logical_len);
    println!(
        "boundary_padding_i8_entries={}",
        layout.active_len - physical_entries
    );
    println!(
        "final_padding_i8_entries={}",
        layout.logical_len - layout.active_len
    );
    println!("materialized_padding_entries={materialized_padding_entries}");
    println!("stored_i8_entries={stored_i8_entries}");
    println!("bucket_count={}", layout.bucket_count);
    println!(
        "shape_mask_rows={}",
        layout.bucket_count.next_power_of_two()
    );
    println!("required_dory_log={setup_log}");
    println!("canonical_dory_log={}", canonical_dory_log(setup_log));
    println!("witness_storage_bytes={stored_i8_entries}");
    println!("estimated_packed_field_entries={packed_field_entries}");
    println!("estimated_packed_field_bytes={packed_field_bytes}");
    println!("estimated_witness_plus_packed_bytes={witness_plus_packed_bytes}");
    println!("descriptor_seconds={descriptor_seconds:.6}");
    io::stdout().flush().context("flush experiment metadata")?;
    drop(descriptor_only);

    let claims_started = Instant::now();
    let claims = claims_for_constant_matrices(&shapes, layout.point_log())?;
    let claims_seconds = report_stage("claims", claims_started, &monitor);

    let generation_started = Instant::now();
    let mut source = DenseMatrixBva::from_fn(&shapes, |matrix, _, _| matrix_value(matrix))?;
    let generation_seconds = report_stage("generation", generation_started, &monitor);
    layout.validate_descriptor(source.descriptor())?;
    ensure!(
        source.values().len() == stored_i8_entries,
        "witness length mismatch"
    );

    let setup_started = Instant::now();
    let (prover_parameters, verifier_parameters) = setup_dory_parameters(setup_log);
    let setup_seconds = report_stage("setup", setup_started, &monitor);
    let mut rng = StdRng::seed_from_u64(arguments.seed);

    let commit_phase_started = Instant::now();
    let (committed, mut commit_profile) =
        commit_opening_collection_profiled(&prover_parameters, &source, BIT_WIDTH)?;
    let validity_proof = if arguments.shape_validity_proof {
        let (proof, seconds) =
            prove_shape_validity_for_collection(&prover_parameters, &committed, &mut rng)?;
        commit_profile.validity_prove_seconds = seconds;
        Some(proof)
    } else {
        None
    };
    let commit_phase_seconds = report_stage("commit_phase", commit_phase_started, &monitor);
    let statement = committed.statement().clone();
    let bva_commit_profile = committed.bva().profile;

    let opening_prove_started = Instant::now();
    let (opening_proof, opening_profile) = prove_opening(
        &prover_parameters,
        &committed,
        &mut source,
        &claims,
        &mut rng,
    )?;
    ensure!(
        opening_proof.physical_dory_proof_count() == 2,
        "Section 6 opening must use one native-i8 BVA proof and one auxiliary proof"
    );
    let opening_prove_seconds = report_stage("opening_prove", opening_prove_started, &monitor);

    let serialization_started = Instant::now();
    let sized = |bytes: u64| usize::try_from(bytes).context("proof size does not fit usize");
    let pi_ip_proof_size_bytes = sized(proof_codec().serialized_size(&opening_proof.packed_eo)?)?;
    let bva_dory_proof_size_bytes =
        sized(proof_codec().serialized_size(&opening_proof.bva_opening.dory_opening)?)?;
    let terminal_dory_proof_size_bytes =
        sized(proof_codec().serialized_size(&opening_proof.terminal_opening)?)?;
    let x_statement_size_bytes = sized(proof_codec().serialized_size(&opening_proof.x)?)?;
    let tbv_statement_size_bytes = sized(proof_codec().serialized_size(&opening_proof.tbv)?)?;
    let point_reduction_size_bytes = sized(proof_codec().serialized_size(&opening_proof.point)?)?;
    let final_evaluation_size_bytes =
        sized(proof_codec().serialized_size(&opening_proof.final_eval)?)?;
    let (proof_bytes, component_sizes) =
        encode_proof_artifact(&statement, validity_proof.as_ref(), &opening_proof)?;
    let proof_sha256 = Sha256::digest(&proof_bytes);
    let (decoded_statement, decoded_validity, decoded_opening) =
        decode_proof_artifact_exact(&proof_bytes)?;
    ensure!(
        decoded_validity.is_some() == arguments.shape_validity_proof,
        "shape validity proof mode does not match the artifact"
    );
    let mut trailing = proof_bytes.clone();
    trailing.push(0);
    ensure!(
        decode_proof_artifact_exact(&trailing).is_err(),
        "proof decoder accepted trailing bytes"
    );
    let serialization_seconds = report_stage("serialization", serialization_started, &monitor);

    let validity_verify_seconds = if let Some(validity) = decoded_validity.as_ref() {
        let validity_verify_started = Instant::now();
        verify_shape_commitment(&verifier_parameters, &decoded_statement, validity)?;
        report_stage("validity_verify", validity_verify_started, &monitor)
    } else {
        println!("validity_verify_seconds=0.000000");
        println!("peak_rss_after_validity_verify_kib={}", monitor.peak_kib());
        io::stdout()
            .flush()
            .context("flush skipped validity stage")?;
        0.0
    };

    let opening_verify_started = Instant::now();
    let claimed_values = claims.iter().map(|claim| claim.value).collect::<Vec<_>>();
    let verification = verify_validity_opening(
        &verifier_parameters,
        &decoded_statement,
        &claimed_values,
        &decoded_opening,
    )?;
    let opening_verify_seconds = report_stage("opening_verify", opening_verify_started, &monitor);

    ensure!(
        opening_profile.padding.materialized_padding_entries == materialized_padding_entries as u64,
        "Packed EO padding counter mismatch"
    );
    ensure!(
        opening_profile.packed_eo.materialized_padding_entries == materialized_padding_entries,
        "Packed EO profile padding count mismatch"
    );
    ensure!(
        opening_profile.packed_eo.physical_entries_per_pass == layout.active_len,
        "Packed EO did not visit the complete padded bucket region"
    );
    ensure!(
        bva_commit_profile.materialized_field_entries == 0
            && bva_commit_profile.materialized_padding_entries == materialized_padding_entries,
        "Dory commitment materialized a field witness or reported the wrong i8 padding"
    );

    let total_prover_seconds = commit_phase_seconds + opening_prove_seconds;
    let total_verifier_seconds = validity_verify_seconds + opening_verify_seconds;
    let total_seconds = experiment_started.elapsed().as_secs_f64();
    let peak_rss_kib = monitor.finish();

    // These top-level prover categories are mutually exclusive. The terminal
    // auxiliary opening serves both conversion phases and is added only once.
    let point_conversion_seconds = opening_profile.x_commit_seconds
        + opening_profile.point_witness_seconds
        + opening_profile.tbv_commit_seconds
        + opening_profile.point_aux_commit_seconds
        + opening_profile.keys_commit_seconds
        + opening_profile.point_prove_seconds;
    let tbv_evaluation_seconds = opening_profile.tbv_reconstruct_seconds
        + opening_profile.final_witness_seconds
        + opening_profile.final_aux_commit_seconds
        + opening_profile.final_prove_seconds;
    let point_conversion_plus_tbv_evaluation_seconds =
        point_conversion_seconds + tbv_evaluation_seconds + opening_profile.terminal_open_seconds;
    let commit_orchestration_seconds = (commit_phase_seconds
        - commit_profile.shape_commit_seconds
        - commit_profile.validity_prove_seconds
        - commit_profile.bva_commit_seconds)
        .max(0.0);
    let classified_opening_seconds = point_conversion_plus_tbv_evaluation_seconds
        + opening_profile.packed_eo_seconds
        + opening_profile.bva_open_seconds;
    let opening_orchestration_seconds =
        (opening_prove_seconds - classified_opening_seconds).max(0.0);
    let other_prover_seconds = commit_orchestration_seconds + opening_orchestration_seconds;
    let prover_breakdown_sum_seconds = commit_profile.shape_commit_seconds
        + commit_profile.validity_prove_seconds
        + commit_profile.bva_commit_seconds
        + opening_profile.packed_eo_seconds
        + opening_profile.bva_open_seconds
        + point_conversion_plus_tbv_evaluation_seconds
        + other_prover_seconds;
    ensure!(
        (prover_breakdown_sum_seconds - total_prover_seconds).abs() < 0.001,
        "prover timing breakdown is not exhaustive"
    );
    let eo_ip_accounted_seconds = opening_profile.packed_eo.timings.accounted_seconds();
    let eo_ip_other_seconds =
        (opening_profile.packed_eo_seconds - eo_ip_accounted_seconds).max(0.0);

    println!("commit_statement_size_bytes={}", component_sizes[0]);
    println!("validity_proof_size_bytes={}", component_sizes[1]);
    println!("opening_proof_size_bytes={}", component_sizes[2]);
    println!("x_statement_size_bytes={x_statement_size_bytes}");
    println!("tbv_statement_size_bytes={tbv_statement_size_bytes}");
    println!("point_reduction_size_bytes={point_reduction_size_bytes}");
    println!("pi_ip_proof_size_bytes={pi_ip_proof_size_bytes}");
    println!("final_evaluation_size_bytes={final_evaluation_size_bytes}");
    println!("bva_dory_proof_size_bytes={bva_dory_proof_size_bytes}");
    println!("terminal_dory_proof_size_bytes={terminal_dory_proof_size_bytes}");
    println!(
        "physical_dory_proof_count={}",
        opening_proof.physical_dory_proof_count()
    );
    println!(
        "proof_objects_size_bytes={}",
        component_sizes[1] + component_sizes[2]
    );
    println!("proof_size_bytes={}", proof_bytes.len());
    println!("total_communication_bytes={}", proof_bytes.len());
    println!("proof_sha256={proof_sha256:x}");
    println!("trailing_bytes_rejected=true");
    println!(
        "dory_commit_physical_entries={}",
        bva_commit_profile.physical_entries
    );
    println!(
        "dory_commit_nonzero_entries={}",
        bva_commit_profile.nonzero_entries
    );
    println!(
        "dory_commit_materialized_field_entries={}",
        bva_commit_profile.materialized_field_entries
    );
    println!(
        "dory_commit_materialized_padding_entries={}",
        bva_commit_profile.materialized_padding_entries
    );
    println!("x_commit_seconds={:.6}", opening_profile.x_commit_seconds);
    println!(
        "point_witness_seconds={:.6}",
        opening_profile.point_witness_seconds
    );
    println!(
        "tbv_commit_seconds={:.6}",
        opening_profile.tbv_commit_seconds
    );
    println!(
        "point_aux_commit_seconds={:.6}",
        opening_profile.point_aux_commit_seconds
    );
    println!(
        "keys_commit_seconds={:.6}",
        opening_profile.keys_commit_seconds
    );
    println!(
        "point_prove_seconds={:.6}",
        opening_profile.point_prove_seconds
    );
    println!(
        "tbv_reconstruct_seconds={:.6}",
        opening_profile.tbv_reconstruct_seconds
    );
    println!("packed_eo_seconds={:.6}", opening_profile.packed_eo_seconds);
    println!("bva_open_seconds={:.6}", opening_profile.bva_open_seconds);
    println!(
        "final_witness_seconds={:.6}",
        opening_profile.final_witness_seconds
    );
    println!(
        "final_aux_commit_seconds={:.6}",
        opening_profile.final_aux_commit_seconds
    );
    println!(
        "final_prove_seconds={:.6}",
        opening_profile.final_prove_seconds
    );
    println!(
        "terminal_open_seconds={:.6}",
        opening_profile.terminal_open_seconds
    );
    println!(
        "packed_eo_source_passes={}",
        opening_profile.packed_eo.source_passes
    );
    println!(
        "packed_eo_total_visited_entries={}",
        opening_profile.packed_eo.total_visited_entries
    );
    println!(
        "packed_eo_maximum_live_field_entries={}",
        opening_profile.packed_eo.maximum_live_vector_entries
    );
    println!(
        "packed_eo_maximum_live_field_bytes={}",
        opening_profile
            .packed_eo
            .maximum_live_vector_entries
            .checked_mul(size_of::<Fr>())
            .context("profile byte count overflowed")?
    );
    println!(
        "packed_eo_packed_cross_terms_seconds={:.6}",
        opening_profile.packed_eo.timings.packed_cross_terms_seconds
    );
    println!(
        "packed_eo_first_fold_seconds={:.6}",
        opening_profile.packed_eo.timings.first_fold_seconds
    );
    println!(
        "packed_eo_reduced_check_seconds={:.6}",
        opening_profile.packed_eo.timings.reduced_check_seconds
    );
    println!(
        "packed_eo_binary_cross_terms_seconds={:.6}",
        opening_profile.packed_eo.timings.binary_cross_terms_seconds
    );
    println!(
        "packed_eo_binary_challenge_seconds={:.6}",
        opening_profile.packed_eo.timings.binary_challenge_seconds
    );
    println!(
        "packed_eo_binary_fold_seconds={:.6}",
        opening_profile.packed_eo.timings.binary_fold_seconds
    );
    println!(
        "packed_eo_binary_rounds_seconds={:.6}",
        opening_profile.packed_eo.timings.binary_rounds_seconds()
    );
    println!(
        "terminal_claim_count={}",
        opening_profile.terminal_claim_count
    );
    println!("claims_seconds={claims_seconds:.6}");
    println!("generation_seconds={generation_seconds:.6}");
    println!("setup_seconds={setup_seconds:.6}");
    println!(
        "shape_commit_seconds={:.6}",
        commit_profile.shape_commit_seconds
    );
    println!(
        "bva_commit_seconds={:.6}",
        commit_profile.bva_commit_seconds
    );
    println!("commit_phase_seconds={commit_phase_seconds:.6}");
    println!(
        "validity_prove_seconds={:.6}",
        commit_profile.validity_prove_seconds
    );
    println!("opening_prove_seconds={opening_prove_seconds:.6}");
    println!(
        "shape_commitment_seconds={:.6}",
        commit_profile.shape_commit_seconds
    );
    println!(
        "dense_vector_commitment_seconds={:.6}",
        commit_profile.bva_commit_seconds
    );
    println!("dense_vector_commitment_kind=native_i8_bva");
    println!("eo_ip_seconds={:.6}", opening_profile.packed_eo_seconds);
    println!("eo_ip_accounted_seconds={eo_ip_accounted_seconds:.6}");
    println!("eo_ip_other_seconds={eo_ip_other_seconds:.6}");
    println!(
        "dense_bva_open_seconds={:.6}",
        opening_profile.bva_open_seconds
    );
    println!("point_conversion_seconds={point_conversion_seconds:.6}");
    println!("tbv_evaluation_seconds={tbv_evaluation_seconds:.6}");
    println!(
        "terminal_auxiliary_open_seconds={:.6}",
        opening_profile.terminal_open_seconds
    );
    println!(
        "point_conversion_plus_tbv_evaluation_seconds={point_conversion_plus_tbv_evaluation_seconds:.6}"
    );
    println!("commit_orchestration_seconds={commit_orchestration_seconds:.6}");
    println!("opening_orchestration_seconds={opening_orchestration_seconds:.6}");
    println!("other_prover_seconds={other_prover_seconds:.6}");
    println!("prover_breakdown_sum_seconds={prover_breakdown_sum_seconds:.6}");
    println!("serialization_seconds={serialization_seconds:.6}");
    println!("validity_verify_seconds={validity_verify_seconds:.6}");
    println!("opening_verify_seconds={opening_verify_seconds:.6}");
    println!("total_prover_seconds={total_prover_seconds:.6}");
    println!("total_verifier_seconds={total_verifier_seconds:.6}");
    println!("total_seconds={total_seconds:.6}");
    println!("peak_rss_kib={peak_rss_kib}");
    println!(
        "shape_validity_proof_status={}",
        if arguments.shape_validity_proof {
            "verified"
        } else {
            "not_requested"
        }
    );
    println!(
        "aggregated_claim_nonzero={}",
        verification.aggregated_claim != Fr::from(0_u64)
    );
    println!("decoded_proof_verified=true");
    println!("accepted=true");
    println!("verified_marker=SECTION6_E2E_VERIFIED");
    Ok(())
}

struct MemoryMonitor {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    handle: Option<thread::JoinHandle<()>>,
}

impl MemoryMonitor {
    fn spawn(sample_period: Duration) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let monitor_stop = Arc::clone(&stop);
        let monitor_peak = Arc::clone(&peak);
        let handle = thread::spawn(move || {
            while !monitor_stop.load(Ordering::Relaxed) {
                if let Some(rss_kib) = read_rss_kib() {
                    monitor_peak.fetch_max(rss_kib, Ordering::Relaxed);
                }
                thread::sleep(sample_period);
            }
            if let Some(rss_kib) = read_rss_kib() {
                monitor_peak.fetch_max(rss_kib, Ordering::Relaxed);
            }
        });
        Self {
            stop,
            peak,
            handle: Some(handle),
        }
    }

    fn peak_kib(&self) -> u64 {
        self.peak.load(Ordering::Relaxed)
    }

    fn finish(mut self) -> u64 {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            handle.join().expect("join memory monitor");
        }
        self.peak_kib()
    }
}

impl Drop for MemoryMonitor {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

fn read_rss_kib() -> Option<u64> {
    if let Ok(status) = std::fs::read_to_string("/proc/self/status") {
        if let Some(rss) = status.lines().find_map(|line| {
            line.strip_prefix("VmRSS:")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        }) {
            return Some(rss);
        }
    }
    let process = std::process::id().to_string();
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &process])
        .output()
        .ok()?;
    std::str::from_utf8(&output.stdout)
        .ok()?
        .trim()
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::{
        parse_arguments_from, physical_entry_count, section6_shapes, uniform_shapes, UniformFamily,
        DEFAULT_MAX_LOG,
    };
    use batch::matrix_collection::DenseBvaDescriptor;

    #[test]
    fn full_shape_family_is_the_requested_two_to_thirty_domain() {
        let shapes = section6_shapes(15).unwrap();
        assert_eq!(shapes.len(), 26);
        assert_eq!(shapes[0], (32_767, 16_383));
        assert_eq!(shapes[1], (16_383, 16_383));
        assert_eq!(shapes[2], (16_383, 8_191));
        assert_eq!(shapes.last(), Some(&(3, 3)));
        assert_eq!(physical_entry_count(&shapes).unwrap(), 1_073_578_014);
        assert!(shapes
            .windows(2)
            .all(|pair| pair[0].0 >= pair[1].0 && pair[0].1 >= pair[1].1));

        let descriptor = DenseBvaDescriptor::from_shapes(&shapes).unwrap();
        assert_eq!(descriptor.logical_len(), 1_usize << 30);
        assert_eq!(descriptor.layout().active_len(), 1_073_741_689);
        assert_eq!(descriptor.conceptual_boundary_zeros(), 163_675);
        assert_eq!(descriptor.conceptual_final_zeros(), 135);
        assert_eq!(descriptor.bucket_count(), 383);
        assert_eq!(descriptor.materialized_padding_entries(), 163_810);
    }

    #[test]
    fn extended_shape_family_is_the_requested_two_to_thirty_four_domain() {
        let shapes = section6_shapes(17).unwrap();
        assert_eq!(shapes.len(), 30);
        assert_eq!(shapes[0], (131_071, 65_535));
        assert_eq!(shapes[1], (65_535, 65_535));
        assert_eq!(shapes[2], (65_535, 32_767));
        assert_eq!(shapes.last(), Some(&(3, 3)));
        assert_eq!(physical_entry_count(&shapes).unwrap(), 17_179_213_858);

        let descriptor = DenseBvaDescriptor::from_shapes(&shapes).unwrap();
        assert_eq!(descriptor.logical_len(), 1_usize << 34);
        assert_eq!(descriptor.layout().active_len(), 17_179_869_049);
        assert_eq!(descriptor.conceptual_boundary_zeros(), 655_191);
        assert_eq!(descriptor.conceptual_final_zeros(), 135);
        assert_eq!(descriptor.bucket_count(), 447);
        assert_eq!(descriptor.materialized_padding_entries(), 655_326);
    }

    #[test]
    fn homogeneous_target_has_sixteen_padded_buckets_per_matrix() {
        let shapes = uniform_shapes(UniformFamily {
            count: 1 << 14,
            log: 10,
        })
        .unwrap();
        let descriptor = DenseBvaDescriptor::from_shapes(&shapes).unwrap();

        assert_eq!(physical_entry_count(&shapes).unwrap(), 17_146_331_136);
        assert_eq!(descriptor.bucket_count(), 1 << 18);
        assert_eq!(descriptor.layout().active_len(), 1 << 34);
        assert_eq!(descriptor.logical_len(), 1 << 34);
        assert_eq!(descriptor.materialized_padding_entries(), 33_538_048);
    }

    #[test]
    fn requested_homogeneous_target_is_the_two_to_thirty_four_domain() {
        let shapes = uniform_shapes(UniformFamily {
            count: 1 << 10,
            log: 12,
        })
        .unwrap();
        let descriptor = DenseBvaDescriptor::from_shapes(&shapes).unwrap();

        assert_eq!(physical_entry_count(&shapes).unwrap(), 17_171_481_600);
        assert_eq!(descriptor.bucket_count(), 1 << 14);
        assert_eq!(descriptor.layout().active_len(), 1 << 34);
        assert_eq!(descriptor.logical_len(), 1 << 34);
        assert_eq!(descriptor.materialized_padding_entries(), 8_387_584);
    }

    #[test]
    fn parser_accepts_two_to_thirty_four_and_checks_the_derived_target_log() {
        let arguments = parse_arguments_from(
            ["--max-log", "17", "--target-log", "34"]
                .into_iter()
                .map(String::from),
        )
        .unwrap()
        .unwrap();
        assert_eq!(arguments.max_log, 17);

        let defaults = parse_arguments_from(Vec::new()).unwrap().unwrap();
        assert_eq!(defaults.max_log, DEFAULT_MAX_LOG);
        assert!(!defaults.shape_validity_proof);

        let explicit_shape_validity =
            parse_arguments_from(["--shape-validity-proof"].into_iter().map(String::from))
                .unwrap()
                .unwrap();
        assert!(explicit_shape_validity.shape_validity_proof);

        let homogeneous = parse_arguments_from(
            [
                "--uniform-count",
                "1024",
                "--uniform-log",
                "12",
                "--target-log",
                "34",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            homogeneous.uniform,
            Some(UniformFamily {
                count: 1024,
                log: 12
            })
        );

        assert!(parse_arguments_from(
            ["--max-log", "17", "--target-log", "30"]
                .into_iter()
                .map(String::from),
        )
        .is_err());
        assert!(parse_arguments_from(
            [
                "--uniform-count",
                "1024",
                "--uniform-log",
                "12",
                "--target-log",
                "30",
            ]
            .into_iter()
            .map(String::from),
        )
        .is_err());
        assert!(parse_arguments_from(["--max-log", "18"].into_iter().map(String::from),).is_err());
    }
}
