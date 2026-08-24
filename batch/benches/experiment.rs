use std::{
    io::{self, Write},
    mem::size_of,
    path::PathBuf,
    process::Command,
    str::FromStr,
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
    jolt_dory::{canonical_dory_log, setup_dory_parameters_profiled},
    matrix_collection::{constant_matrix_mle_evaluate, DenseBvaDescriptor, DenseMatrixBva},
    model_bva::{default_model_data_root, ModelBvaManifest, ModelPreset},
    section6_data::{Section6Layout, Section6MatrixClaim},
    validity_protocol::{
        commit_opening_collection_profiled, prove_opening, prove_shape_validity_for_collection,
        validity_required_dory_log, verify_shape_commitment, verify_validity_opening_profiled,
        ValidityCommitStatement, ValidityOpeningProof,
    },
    validity_shape::ShapeValidityProof,
};
use bincode::Options;
use sha2::{Digest, Sha256};

const HOMOGENEOUS_BIT_WIDTH: usize = 8;
const DEFAULT_SEED: u64 = 0x4558_5045_5249_4d54;
const PROOF_MAGIC: [u8; 8] = *b"S6BENCH1";
const EXPERIMENT_STAGE_COUNT: usize = 8;

#[derive(Clone, Debug, Eq, PartialEq)]
enum Workload {
    Homogeneous {
        matrix_count: usize,
        rows: usize,
        cols: usize,
    },
    Model(ModelPreset),
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Arguments {
    workload: Workload,
    model_data_dir: PathBuf,
    manifest_override: Option<PathBuf>,
    bva_override: Option<PathBuf>,
    verify_input_hash: bool,
    target_log: Option<usize>,
    seed: u64,
    allow_non_64_threads: bool,
    shape_validity_proof: bool,
    summary_only: bool,
}

enum ExperimentSource {
    I8(DenseMatrixBva<i8>),
    I16(DenseMatrixBva<i16>),
}

impl ExperimentSource {
    fn descriptor(&self) -> &DenseBvaDescriptor {
        match self {
            Self::I8(source) => source.descriptor(),
            Self::I16(source) => source.descriptor(),
        }
    }

    fn values_len(&self) -> usize {
        match self {
            Self::I8(source) => source.values().len(),
            Self::I16(source) => source.values().len(),
        }
    }

    fn matrix_mle_evaluate(&self, matrix: usize, left: &[Fr], right: &[Fr]) -> Result<Fr> {
        match self {
            Self::I8(source) => Ok(source.matrix_mle_evaluate(matrix, left, right)?),
            Self::I16(source) => Ok(source.matrix_mle_evaluate(matrix, left, right)?),
        }
    }
}

fn main() -> Result<()> {
    run_from_args(std::env::args().skip(1))
}

pub(crate) fn run_from_args(arguments: impl IntoIterator<Item = String>) -> Result<()> {
    let Some(arguments) = parse_arguments(arguments)? else {
        return Ok(());
    };
    run(arguments)
}

fn run(arguments: Arguments) -> Result<()> {
    let threads = rayon::current_num_threads();
    if !arguments.allow_non_64_threads {
        ensure!(
            threads == 64,
            "the formal experiment requires 64 Rayon threads, got {threads}; set RAYON_NUM_THREADS=64"
        );
    }

    let detailed_output = !arguments.summary_only;
    report_active_stage(1, "Workload preparation")?;
    let experiment_started = Instant::now();
    let monitor = MemoryMonitor::spawn(Duration::from_millis(100));
    let manifest_started = Instant::now();
    let model_manifest = match &arguments.workload {
        Workload::Model(preset) => Some(
            ModelBvaManifest::load(
                preset.clone(),
                &arguments.model_data_dir,
                arguments.manifest_override.as_deref(),
                arguments.bva_override.as_deref(),
            )
            .with_context(|| format!("load converted model preset {preset}"))?,
        ),
        Workload::Homogeneous { .. } => None,
    };
    let manifest_seconds = manifest_started.elapsed().as_secs_f64();

    let (bit_width, storage_dtype, storage_element_bytes) = match model_manifest.as_ref() {
        Some(manifest) => {
            ensure!(
                matches!(
                    (manifest.storage_dtype(), manifest.bit_width()),
                    ("i8", 8) | ("i16", 16)
                ),
                "preset {} uses unsupported native {}-bit {} BVA storage",
                manifest.preset(),
                manifest.bit_width(),
                manifest.storage_dtype()
            );
            (
                manifest.bit_width(),
                manifest.storage_dtype(),
                manifest.bit_width() / 8,
            )
        }
        None => (HOMOGENEOUS_BIT_WIDTH, "i8", size_of::<i8>()),
    };

    let shapes = match (&arguments.workload, model_manifest.as_ref()) {
        (
            Workload::Homogeneous {
                matrix_count,
                rows,
                cols,
            },
            None,
        ) => vec![(*rows, *cols); *matrix_count],
        (Workload::Model(_), Some(manifest)) => manifest.matrix_shapes().to_vec(),
        _ => unreachable!("workload and model manifest stay aligned"),
    };
    let physical_entries = physical_entry_count(&shapes)?;
    let descriptor_started = Instant::now();
    let descriptor = DenseBvaDescriptor::from_shapes(&shapes)?;
    if let Some(manifest) = model_manifest.as_ref() {
        ensure!(
            descriptor == *manifest.descriptor(),
            "validated model descriptor changed before the experiment"
        );
    }
    let layout = Section6Layout::from_descriptor(&descriptor, bit_width)?;
    let descriptor_seconds =
        report_stage("descriptor", descriptor_started, &monitor, detailed_output);
    let logical_log = layout.logical_len.ilog2() as usize;
    if let Some(target_log) = arguments.target_log {
        ensure!(
            target_log == logical_log,
            "workload has logical domain 2^{logical_log}, not requested 2^{target_log}"
        );
    }
    let stride = packing_stride_for_bit_width::<Fr>(layout.logical_len, bit_width)?;
    let setup_log = validity_required_dory_log(&layout)?;
    let packed_field_entries = layout.active_len.div_ceil(stride);
    let packed_field_bytes = packed_field_entries
        .checked_mul(size_of::<Fr>())
        .context("packed field byte estimate overflowed")?;
    let stored_shortint_entries = layout.logical_len;
    let stored_shortint_bytes = stored_shortint_entries
        .checked_mul(storage_element_bytes)
        .context("stored witness byte estimate overflowed")?;
    let materialized_padding_entries = stored_shortint_entries - physical_entries;

    if detailed_output {
        println!("output_mode=detailed");
        println!("experiment=parcomm-cargo-bench");
        println!("protocol=section6-capped16-packed-eo-padded-shortint-dory");
        println!(
            "shape_validity_proof_enabled={}",
            arguments.shape_validity_proof
        );
        println!("backend=jolt-dory-bn254-cpu");
        println!("rayon_threads={threads}");
        println!("formal_thread_requirement=64");
        println!("seed={}", arguments.seed);
        println!("bit_width={bit_width}");
        println!("storage_dtype={storage_dtype}");
        println!("storage_element_bytes={storage_element_bytes}");
        println!("kappa={stride}");
        println!("splitting_budget={}", DenseBvaDescriptor::SPLITTING_BUDGET);
        match &arguments.workload {
            Workload::Homogeneous {
                matrix_count,
                rows,
                cols,
            } => {
                println!("family=homogeneous");
                println!("matrix_count={matrix_count}");
                println!("matrix_rows={rows}");
                println!("matrix_cols={cols}");
                println!("shape_formula={rows}x{cols} repeated");
                println!("input_kind=generated_i8");
            }
            Workload::Model(preset) => {
                let manifest = model_manifest
                    .as_ref()
                    .expect("model preset has a validated manifest");
                println!("family=model");
                println!("model_preset={preset}");
                println!("matrix_count={}", shapes.len());
                println!("input_kind=converted_bva");
                println!("manifest_path={}", manifest.manifest_path().display());
                println!("bva_path={}", manifest.bva_path().display());
                println!("bva_size_bytes={}", manifest.bva_size_bytes());
                println!("input_sha256_expected={}", manifest.expected_sha256());
                println!("verify_input_hash={}", arguments.verify_input_hash);
            }
        }
        println!("physical_entries={physical_entries}");
        println!("active_logical_entries={}", layout.active_len);
        println!("logical_entries={}", layout.logical_len);
        println!(
            "boundary_padding_shortint_entries={}",
            layout.active_len - physical_entries
        );
        println!(
            "final_padding_shortint_entries={}",
            layout.logical_len - layout.active_len
        );
        println!("stored_shortint_entries={stored_shortint_entries}");
        println!("stored_shortint_bytes={stored_shortint_bytes}");
        println!("bucket_count={}", layout.bucket_count);
        println!("required_dory_log={setup_log}");
        println!("canonical_dory_log={}", canonical_dory_log(setup_log));
        println!("estimated_packed_field_entries={packed_field_entries}");
        println!("estimated_packed_field_bytes={packed_field_bytes}");
        println!(
            "estimated_witness_plus_packed_bytes={}",
            stored_shortint_bytes
                .checked_add(packed_field_bytes)
                .context("combined memory estimate overflowed")?
        );
        println!("manifest_seconds={manifest_seconds:.6}");
        println!("descriptor_seconds={descriptor_seconds:.6}");
        io::stdout().flush().context("flush experiment metadata")?;
    }

    report_active_stage(2, "Input loading or generation")?;
    let generation_started = Instant::now();
    let (
        mut source,
        input_allocation_seconds,
        input_read_seconds,
        input_hash_seconds,
        input_adopt_seconds,
        input_loader_total_seconds,
    ) = match model_manifest {
        Some(manifest) => match manifest.storage_dtype() {
            "i8" => {
                let loaded = manifest.load_i8(arguments.verify_input_hash)?;
                if detailed_output {
                    println!("input_bytes_read={}", loaded.profile.bytes_read);
                    println!("input_sha256_verified={}", loaded.profile.sha256_verified);
                }
                (
                    ExperimentSource::I8(loaded.bva),
                    loaded.profile.allocation_seconds,
                    loaded.profile.read_seconds,
                    loaded.profile.sha256_seconds,
                    loaded.profile.adopt_seconds,
                    loaded.profile.total_seconds,
                )
            }
            "i16" => {
                let loaded = manifest.load_i16(arguments.verify_input_hash)?;
                if detailed_output {
                    println!("input_bytes_read={}", loaded.profile.bytes_read);
                    println!("input_sha256_verified={}", loaded.profile.sha256_verified);
                }
                (
                    ExperimentSource::I16(loaded.bva),
                    loaded.profile.allocation_seconds,
                    loaded.profile.read_seconds,
                    loaded.profile.sha256_seconds,
                    loaded.profile.adopt_seconds,
                    loaded.profile.total_seconds,
                )
            }
            dtype => bail!("unsupported native BVA storage dtype {dtype}"),
        },
        None => (
            ExperimentSource::I8(DenseMatrixBva::from_fn(&shapes, |matrix, _, _| {
                matrix_value(matrix)
            })?),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ),
    };
    let generation_seconds =
        report_stage("generation", generation_started, &monitor, detailed_output);
    layout.validate_descriptor(source.descriptor())?;
    ensure!(
        source.values_len() == stored_shortint_entries,
        "witness length mismatch"
    );

    report_active_stage(3, "Claim construction")?;
    let claims_started = Instant::now();
    let claims = match &arguments.workload {
        Workload::Homogeneous { .. } => claims_for_constant_matrices(&shapes, layout.point_log())?,
        Workload::Model(_) => claims_for_prepacked_matrices(&source, layout.point_log())?,
    };
    let claims_seconds = report_stage("claims", claims_started, &monitor, detailed_output);

    report_active_stage(4, "Dory setup")?;
    let setup_started = Instant::now();
    let (prover_parameters, verifier_parameters, setup_profile) =
        setup_dory_parameters_profiled(setup_log);
    let setup_seconds = report_stage("setup", setup_started, &monitor, detailed_output);
    let mut rng = StdRng::seed_from_u64(arguments.seed);

    report_active_stage(5, "Shape and dense BVA commitments")?;
    let prover_wall_started = Instant::now();
    let commit_started = Instant::now();
    let (committed, mut commit_profile) = match &source {
        ExperimentSource::I8(source) => {
            commit_opening_collection_profiled(&prover_parameters, source, bit_width)?
        }
        ExperimentSource::I16(source) => {
            commit_opening_collection_profiled(&prover_parameters, source, bit_width)?
        }
    };
    let validity_proof = if arguments.shape_validity_proof {
        let (proof, seconds) =
            prove_shape_validity_for_collection(&prover_parameters, &committed, &mut rng)?;
        commit_profile.validity_prove_seconds = seconds;
        Some(proof)
    } else {
        None
    };
    let commit_phase_seconds =
        report_stage("commit_phase", commit_started, &monitor, detailed_output);
    let statement_clone_started = Instant::now();
    let statement = committed.statement().clone();
    let statement_clone_seconds = statement_clone_started.elapsed().as_secs_f64();

    report_active_stage(6, "Opening prover")?;
    let opening_started = Instant::now();
    let (opening_proof, opening_profile) = match &mut source {
        ExperimentSource::I8(source) => {
            prove_opening(&prover_parameters, &committed, source, &claims, &mut rng)?
        }
        ExperimentSource::I16(source) => {
            prove_opening(&prover_parameters, &committed, source, &claims, &mut rng)?
        }
    };
    let opening_prove_seconds =
        report_stage("opening_prove", opening_started, &monitor, detailed_output);
    let prover_wall_seconds = prover_wall_started.elapsed().as_secs_f64();

    report_active_stage(7, "Proof serialization and decoding")?;
    let serialization_started = Instant::now();
    let proof_component_sizing_started = Instant::now();
    let x_statement_size_bytes = serialized_size(&opening_proof.x)?;
    let tbv_statement_size_bytes = serialized_size(&opening_proof.tbv)?;
    let point_reduction_size_bytes = serialized_size(&opening_proof.point)?;
    let point_aux_size_bytes = serialized_size(&opening_proof.point.aux)?;
    let point_keys_size_bytes = serialized_size(&opening_proof.point.keys)?;
    let point_identity_size_bytes = 0_usize;
    let point_lookup_size_bytes = serialized_size(&opening_proof.point.lookup)?;
    let point_lookup_aux_size_bytes =
        serialized_size(&opening_proof.point.lookup.auxiliary_commitments)?;
    let point_lookup_identity_size_bytes =
        serialized_size(&opening_proof.point.lookup.identity_reduction)?;
    let point_lookup_sum_size_bytes = serialized_size(&opening_proof.point.lookup.sum_reduction)?;
    let packed_eo_size_bytes = serialized_size(&opening_proof.packed_eo)?;
    let packed_eo_field_element_count = opening_proof.packed_eo.field_element_count();
    let packed_eo_stride_size_bytes = serialized_size(&opening_proof.packed_eo.packing_stride)?;
    let packed_eo_positive_size_bytes = serialized_size(&opening_proof.packed_eo.packed_positive)?;
    let packed_eo_negative_size_bytes = serialized_size(&opening_proof.packed_eo.packed_negative)?;
    let packed_eo_binary_size_bytes = serialized_size(&opening_proof.packed_eo.binary_rounds)?;
    let packed_eo_a_reduced_size_bytes = serialized_size(&opening_proof.packed_eo.a_reduced)?;
    let packed_eo_b_reduced_size_bytes = serialized_size(&opening_proof.packed_eo.b_reduced)?;
    let packed_eo_scalar_identity_size_bytes = 0_usize;
    let final_evaluation_size_bytes = serialized_size(&opening_proof.final_eval)?;
    let final_aux_size_bytes = serialized_size(&opening_proof.final_eval.aux)?;
    let final_identity_size_bytes = serialized_size(&opening_proof.final_eval.joint_identity)?;
    let joint_point_final_identity_size_bytes = final_identity_size_bytes;
    let final_average_commitment_size_bytes = 0_usize;
    let final_scalar_binding_size_bytes = 0_usize;
    let bva_opening_size_bytes = serialized_size(&opening_proof.bva_opening)?;
    let bva_dory_proof_size_bytes = serialized_size(&opening_proof.bva_opening.dory_opening)?;
    let bva_evaluation_binding_size_bytes =
        serialized_size(&opening_proof.bva_opening.evaluation_binding)?;
    let terminal_opening_size_bytes = serialized_size(&opening_proof.terminal_opening)?;
    let terminal_sumcheck_size_bytes = serialized_size(&opening_proof.terminal_opening.sumcheck)?;
    let terminal_dory_size_bytes = serialized_size(&opening_proof.terminal_opening.dory)?;
    let terminal_transcript_state_size_bytes =
        serialized_size(&opening_proof.terminal_opening.blindfold_transcript_state)?;
    let terminal_blindfold_size_bytes = serialized_size(&opening_proof.terminal_opening.blindfold)?;
    let proof_component_sizing_seconds = proof_component_sizing_started.elapsed().as_secs_f64();
    let proof_encode_started = Instant::now();
    let (proof_bytes, component_sizes) =
        encode_proof(&statement, validity_proof.as_ref(), &opening_proof)?;
    let proof_encode_seconds = proof_encode_started.elapsed().as_secs_f64();
    ensure!(
        x_statement_size_bytes
            + tbv_statement_size_bytes
            + point_reduction_size_bytes
            + packed_eo_size_bytes
            + final_evaluation_size_bytes
            + bva_opening_size_bytes
            + terminal_opening_size_bytes
            == component_sizes[2],
        "opening proof component sizes are not exhaustive"
    );
    ensure!(
        point_aux_size_bytes
            + point_keys_size_bytes
            + point_identity_size_bytes
            + point_lookup_size_bytes
            == point_reduction_size_bytes,
        "point reduction component sizes are not exhaustive"
    );
    ensure!(
        point_lookup_aux_size_bytes
            + point_lookup_identity_size_bytes
            + point_lookup_sum_size_bytes
            == point_lookup_size_bytes,
        "tagged lookup component sizes are not exhaustive"
    );
    ensure!(
        packed_eo_stride_size_bytes
            + packed_eo_positive_size_bytes
            + packed_eo_negative_size_bytes
            + packed_eo_binary_size_bytes
            + packed_eo_a_reduced_size_bytes
            + packed_eo_b_reduced_size_bytes
            + packed_eo_scalar_identity_size_bytes
            == packed_eo_size_bytes,
        "Packed EO component sizes are not exhaustive"
    );
    ensure!(
        final_aux_size_bytes
            + final_identity_size_bytes
            + final_average_commitment_size_bytes
            + final_scalar_binding_size_bytes
            == final_evaluation_size_bytes,
        "final evaluation component sizes are not exhaustive"
    );
    ensure!(
        bva_dory_proof_size_bytes + bva_evaluation_binding_size_bytes == bva_opening_size_bytes,
        "BVA opening component sizes are not exhaustive"
    );
    ensure!(
        terminal_sumcheck_size_bytes
            + terminal_dory_size_bytes
            + terminal_transcript_state_size_bytes
            + terminal_blindfold_size_bytes
            == terminal_opening_size_bytes,
        "terminal opening component sizes are not exhaustive"
    );
    let proof_hash_started = Instant::now();
    let proof_sha256 = Sha256::digest(&proof_bytes);
    let proof_hash_seconds = proof_hash_started.elapsed().as_secs_f64();
    let proof_decode_started = Instant::now();
    let (decoded_statement, decoded_validity, decoded_opening) = decode_proof(&proof_bytes)?;
    ensure!(
        decoded_validity.is_some() == arguments.shape_validity_proof,
        "decoded shape validity mode differs from the requested mode"
    );
    let proof_decode_seconds = proof_decode_started.elapsed().as_secs_f64();
    let trailing_decode_check_started = Instant::now();
    let mut trailing = proof_bytes.clone();
    trailing.push(0);
    ensure!(
        decode_proof(&trailing).is_err(),
        "proof decoder accepted trailing bytes"
    );
    let trailing_decode_check_seconds = trailing_decode_check_started.elapsed().as_secs_f64();
    let serialization_seconds = report_stage(
        "serialization",
        serialization_started,
        &monitor,
        detailed_output,
    );

    report_active_stage(8, "Verification and final validation")?;
    let validity_verify_seconds = if let Some(proof) = decoded_validity.as_ref() {
        let started = Instant::now();
        verify_shape_commitment(&verifier_parameters, &decoded_statement, proof)?;
        report_stage("validity_verify", started, &monitor, detailed_output)
    } else {
        if detailed_output {
            println!("validity_verify_seconds=0.000000");
        }
        0.0
    };
    let opening_verify_started = Instant::now();
    let claimed_values = claims.iter().map(|claim| claim.value).collect::<Vec<_>>();
    let (verification, verifier_profile) = verify_validity_opening_profiled(
        &verifier_parameters,
        &decoded_statement,
        &claimed_values,
        &decoded_opening,
    )?;
    let opening_verify_seconds = report_stage(
        "opening_verify",
        opening_verify_started,
        &monitor,
        detailed_output,
    );

    ensure!(opening_proof.physical_dory_proof_count() == 2);
    ensure!(
        opening_profile.padding.materialized_padding_entries == materialized_padding_entries as u64
    );
    ensure!(
        opening_profile.packed_eo.physical_entries_per_pass == layout.active_len,
        "Packed EO did not visit the complete padded BVA"
    );
    let total_prover_seconds = commit_phase_seconds + opening_prove_seconds;
    let total_verifier_seconds = validity_verify_seconds + opening_verify_seconds;
    let total_seconds = experiment_started.elapsed().as_secs_f64();
    let peak_rss_kib = monitor.finish();

    let commit_orchestration_seconds = (commit_phase_seconds
        - commit_profile.shape_commit_seconds
        - commit_profile.validity_prove_seconds
        - commit_profile.bva_commit_seconds)
        .max(0.0);
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
    let opening_classified_seconds = opening_profile.validation_seconds
        + opening_profile.batch_point_seconds
        + point_conversion_seconds
        + opening_profile.packed_binding_seconds
        + opening_profile.tbv_reconstruct_seconds
        + opening_profile.packed_eo_seconds
        + opening_profile.virtual_tbv_drop_seconds
        + opening_profile.bva_open_seconds
        + opening_profile.point_witness_drop_seconds
        + opening_profile.final_witness_seconds
        + opening_profile.final_aux_commit_seconds
        + opening_profile.final_witness_drop_seconds
        + opening_profile.final_prove_seconds
        + opening_profile.terminal_claim_collection_seconds
        + opening_profile.terminal_open_seconds
        + opening_profile.result_assembly_seconds;
    let opening_orchestration_seconds =
        (opening_prove_seconds - opening_classified_seconds).max(0.0);
    let prover_breakdown_sum_seconds = commit_profile.shape_commit_seconds
        + commit_profile.validity_prove_seconds
        + commit_profile.bva_commit_seconds
        + commit_orchestration_seconds
        + opening_classified_seconds
        + opening_orchestration_seconds;
    ensure!(
        (prover_breakdown_sum_seconds - total_prover_seconds).abs() < 0.001,
        "prover timing breakdown is not exhaustive"
    );
    let prover_interphase_seconds = (prover_wall_seconds - total_prover_seconds).max(0.0);
    let prover_interphase_other_seconds =
        (prover_interphase_seconds - statement_clone_seconds).max(0.0);
    let prover_wall_breakdown_sum_seconds =
        prover_breakdown_sum_seconds + prover_interphase_seconds;
    ensure!(
        (prover_wall_breakdown_sum_seconds - prover_wall_seconds).abs() < 0.001,
        "prover wall timing breakdown is not exhaustive"
    );

    let eo_ip_accounted_seconds = opening_profile.packed_eo.timings.accounted_seconds();
    let eo_ip_other_seconds =
        (opening_profile.packed_eo_seconds - eo_ip_accounted_seconds).max(0.0);

    let verifier_classified_seconds = verifier_profile.statement_validation_seconds
        + verifier_profile.batch_point_seconds
        + verifier_profile.point_prepare_seconds
        + verifier_profile.point_verify_seconds
        + verifier_profile.packed_eo_verify_seconds
        + verifier_profile.bva_open_verify_seconds
        + verifier_profile.final_prepare_seconds
        + verifier_profile.final_verify_seconds
        + verifier_profile.terminal_claim_collection_seconds
        + verifier_profile.terminal_open_verify_seconds
        + verifier_profile.result_assembly_seconds;
    let verifier_orchestration_seconds =
        (opening_verify_seconds - verifier_classified_seconds).max(0.0);
    let verifier_breakdown_sum_seconds =
        verifier_classified_seconds + verifier_orchestration_seconds;
    ensure!(
        (verifier_breakdown_sum_seconds - opening_verify_seconds).abs() < 0.001,
        "opening verifier timing breakdown is not exhaustive"
    );

    let serialization_classified_seconds = proof_component_sizing_seconds
        + proof_encode_seconds
        + proof_hash_seconds
        + proof_decode_seconds
        + trailing_decode_check_seconds;
    let serialization_orchestration_seconds =
        (serialization_seconds - serialization_classified_seconds).max(0.0);
    let preprocessing_seconds =
        manifest_seconds + descriptor_seconds + generation_seconds + claims_seconds;
    let end_to_end_classified_seconds = preprocessing_seconds
        + setup_seconds
        + prover_wall_seconds
        + serialization_seconds
        + total_verifier_seconds;
    let experiment_orchestration_seconds = (total_seconds - end_to_end_classified_seconds).max(0.0);
    let end_to_end_breakdown_sum_seconds =
        end_to_end_classified_seconds + experiment_orchestration_seconds;
    ensure!(
        (end_to_end_breakdown_sum_seconds - total_seconds).abs() < 0.01,
        "end-to-end timing breakdown is not exhaustive"
    );

    let input_loader_classified_seconds =
        input_allocation_seconds + input_read_seconds + input_hash_seconds + input_adopt_seconds;
    let input_loader_other_seconds =
        (input_loader_total_seconds - input_loader_classified_seconds).max(0.0);
    let input_generation_other_seconds = (generation_seconds - input_loader_total_seconds).max(0.0);
    let setup_classified_seconds =
        setup_profile.pcs_setup_seconds + setup_profile.blindfold_setup_seconds;
    let setup_orchestration_seconds = (setup_seconds - setup_classified_seconds).max(0.0);

    if detailed_output {
        println!("input_allocation_seconds={input_allocation_seconds:.6}");
        println!("input_read_seconds={input_read_seconds:.6}");
        println!("input_hash_seconds={input_hash_seconds:.6}");
        println!("input_adopt_seconds={input_adopt_seconds:.6}");
        println!(
            "input_validation_seconds={:.6}",
            manifest_seconds + input_hash_seconds + input_adopt_seconds
        );
        println!("input_loader_classified_seconds={input_loader_classified_seconds:.6}");
        println!("input_loader_other_seconds={input_loader_other_seconds:.6}");
        println!("input_loader_total_seconds={input_loader_total_seconds:.6}");
        println!("input_generation_other_seconds={input_generation_other_seconds:.6}");
        println!("claims_seconds={claims_seconds:.6}");
        println!("generation_seconds={generation_seconds:.6}");
        println!(
            "dory_pcs_setup_seconds={:.6}",
            setup_profile.pcs_setup_seconds
        );
        println!(
            "blindfold_setup_seconds={:.6}",
            setup_profile.blindfold_setup_seconds
        );
        println!(
            "setup_profile_total_seconds={:.6}",
            setup_profile.total_seconds
        );
        println!("setup_classified_seconds={setup_classified_seconds:.6}");
        println!("setup_orchestration_seconds={setup_orchestration_seconds:.6}");
        println!("setup_seconds={setup_seconds:.6}");
        println!(
            "shape_commit_seconds={:.6}",
            commit_profile.shape_commit_seconds
        );
        println!(
            "bva_commit_seconds={:.6}",
            commit_profile.bva_commit_seconds
        );
        println!(
            "validity_prove_seconds={:.6}",
            commit_profile.validity_prove_seconds
        );
        println!("commit_phase_seconds={commit_phase_seconds:.6}");
        println!("commit_orchestration_seconds={commit_orchestration_seconds:.6}");
        println!(
            "opening_validation_seconds={:.6}",
            opening_profile.validation_seconds
        );
        println!(
            "batch_point_seconds={:.6}",
            opening_profile.batch_point_seconds
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
            "packed_binding_seconds={:.6}",
            opening_profile.packed_binding_seconds
        );
        println!(
            "tbv_reconstruct_seconds={:.6}",
            opening_profile.tbv_reconstruct_seconds
        );
        println!("packed_eo_seconds={:.6}", opening_profile.packed_eo_seconds);
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
        println!("eo_ip_accounted_seconds={eo_ip_accounted_seconds:.6}");
        println!("eo_ip_other_seconds={eo_ip_other_seconds:.6}");
        println!(
            "virtual_tbv_drop_seconds={:.6}",
            opening_profile.virtual_tbv_drop_seconds
        );
        println!("bva_open_seconds={:.6}", opening_profile.bva_open_seconds);
        println!(
            "bva_open_point_preparation_seconds={:.6}",
            opening_profile.bva_open.point_preparation_seconds
        );
        println!(
            "bva_open_dense_preparation_seconds={:.6}",
            opening_profile.bva_open.dense_preparation_seconds
        );
        println!(
            "bva_open_transcript_preparation_seconds={:.6}",
            opening_profile.bva_open.transcript_preparation_seconds
        );
        println!(
            "bva_open_dory_seconds={:.6}",
            opening_profile.bva_open.dory_open_seconds
        );
        println!(
            "bva_open_scalar_binding_seconds={:.6}",
            opening_profile.bva_open.scalar_binding_seconds
        );
        println!(
            "bva_open_profile_other_seconds={:.6}",
            opening_profile.bva_open.other_seconds()
        );
        println!(
            "bva_open_profile_total_seconds={:.6}",
            opening_profile.bva_open.total_seconds
        );
        println!(
            "point_witness_drop_seconds={:.6}",
            opening_profile.point_witness_drop_seconds
        );
        println!(
            "final_witness_seconds={:.6}",
            opening_profile.final_witness_seconds
        );
        println!(
            "final_aux_commit_seconds={:.6}",
            opening_profile.final_aux_commit_seconds
        );
        println!(
            "final_witness_drop_seconds={:.6}",
            opening_profile.final_witness_drop_seconds
        );
        println!(
            "final_prove_seconds={:.6}",
            opening_profile.final_prove_seconds
        );
        println!(
            "terminal_claim_collection_seconds={:.6}",
            opening_profile.terminal_claim_collection_seconds
        );
        println!(
            "terminal_open_seconds={:.6}",
            opening_profile.terminal_open_seconds
        );
        println!(
            "terminal_preparation_state_setup_seconds={:.6}",
            opening_profile
                .terminal_open
                .preparation_state_setup_seconds
        );
        println!(
            "terminal_sumcheck_prove_seconds={:.6}",
            opening_profile.terminal_open.sumcheck_seconds
        );
        println!(
            "terminal_dory_batch_open_seconds={:.6}",
            opening_profile.terminal_open.dory_batch_open_seconds
        );
        println!(
            "terminal_blindfold_preparation_seconds={:.6}",
            opening_profile.terminal_open.blindfold_preparation_seconds
        );
        println!(
            "terminal_blindfold_prove_seconds={:.6}",
            opening_profile.terminal_open.blindfold_prove_seconds
        );
        println!(
            "terminal_result_assembly_seconds={:.6}",
            opening_profile.terminal_open.result_assembly_seconds
        );
        println!(
            "terminal_profile_other_seconds={:.6}",
            (opening_profile.terminal_open.total_seconds
                - opening_profile.terminal_open.classified_seconds())
            .max(0.0)
        );
        println!(
            "terminal_profile_total_seconds={:.6}",
            opening_profile.terminal_open.total_seconds
        );
        println!(
            "opening_result_assembly_seconds={:.6}",
            opening_profile.result_assembly_seconds
        );
        println!("point_conversion_seconds={point_conversion_seconds:.6}");
        println!("tbv_evaluation_seconds={tbv_evaluation_seconds:.6}");
        println!("opening_classified_seconds={opening_classified_seconds:.6}");
        println!("opening_orchestration_seconds={opening_orchestration_seconds:.6}");
        println!("opening_prove_seconds={opening_prove_seconds:.6}");
        println!("statement_clone_seconds={statement_clone_seconds:.6}");
        println!("prover_interphase_seconds={prover_interphase_seconds:.6}");
        println!("prover_interphase_other_seconds={prover_interphase_other_seconds:.6}");
        println!("prover_breakdown_sum_seconds={prover_breakdown_sum_seconds:.6}");
        println!("prover_wall_seconds={prover_wall_seconds:.6}");
        println!("prover_wall_breakdown_sum_seconds={prover_wall_breakdown_sum_seconds:.6}");
        println!(
            "terminal_claim_count={}",
            opening_profile.terminal_claim_count
        );
        println!("commit_statement_size_bytes={}", component_sizes[0]);
        println!("validity_proof_size_bytes={}", component_sizes[1]);
        println!("opening_proof_size_bytes={}", component_sizes[2]);
        println!(
            "cached_registration_size_bytes={}",
            component_sizes[0] + component_sizes[1]
        );
        println!("per_opening_proof_size_bytes={}", component_sizes[2]);
        println!("x_statement_size_bytes={x_statement_size_bytes}");
        println!("tbv_statement_size_bytes={tbv_statement_size_bytes}");
        println!("point_reduction_size_bytes={point_reduction_size_bytes}");
        println!("point_aux_size_bytes={point_aux_size_bytes}");
        println!("point_keys_size_bytes={point_keys_size_bytes}");
        println!("point_identity_size_bytes={point_identity_size_bytes}");
        println!("point_lookup_size_bytes={point_lookup_size_bytes}");
        println!("point_lookup_aux_size_bytes={point_lookup_aux_size_bytes}");
        println!("point_lookup_identity_size_bytes={point_lookup_identity_size_bytes}");
        println!("point_lookup_sum_size_bytes={point_lookup_sum_size_bytes}");
        println!("packed_eo_size_bytes={packed_eo_size_bytes}");
        println!("packed_eo_field_element_count={packed_eo_field_element_count}");
        println!("packed_eo_stride_size_bytes={packed_eo_stride_size_bytes}");
        println!("packed_eo_positive_size_bytes={packed_eo_positive_size_bytes}");
        println!("packed_eo_negative_size_bytes={packed_eo_negative_size_bytes}");
        println!("packed_eo_binary_size_bytes={packed_eo_binary_size_bytes}");
        println!("packed_eo_a_reduced_size_bytes={packed_eo_a_reduced_size_bytes}");
        println!("packed_eo_b_reduced_size_bytes={packed_eo_b_reduced_size_bytes}");
        println!("packed_eo_scalar_identity_size_bytes={packed_eo_scalar_identity_size_bytes}");
        println!("final_evaluation_size_bytes={final_evaluation_size_bytes}");
        println!("final_aux_size_bytes={final_aux_size_bytes}");
        println!("final_identity_size_bytes={final_identity_size_bytes}");
        println!("joint_point_final_identity_size_bytes={joint_point_final_identity_size_bytes}");
        println!("final_average_commitment_size_bytes={final_average_commitment_size_bytes}");
        println!("final_scalar_binding_size_bytes={final_scalar_binding_size_bytes}");
        println!("bva_opening_size_bytes={bva_opening_size_bytes}");
        println!("bva_dory_proof_size_bytes={bva_dory_proof_size_bytes}");
        println!("bva_evaluation_binding_size_bytes={bva_evaluation_binding_size_bytes}");
        println!("terminal_opening_size_bytes={terminal_opening_size_bytes}");
        println!("terminal_sumcheck_size_bytes={terminal_sumcheck_size_bytes}");
        println!("terminal_dory_size_bytes={terminal_dory_size_bytes}");
        println!("terminal_transcript_state_size_bytes={terminal_transcript_state_size_bytes}");
        println!("terminal_blindfold_size_bytes={terminal_blindfold_size_bytes}");
        println!(
            "artifact_framing_size_bytes={}",
            proof_bytes.len() - component_sizes.iter().sum::<usize>()
        );
        println!("proof_size_bytes={}", proof_bytes.len());
        println!("proof_sha256={proof_sha256:x}");
        println!("trailing_bytes_rejected=true");
        println!(
            "physical_dory_proof_count={}",
            opening_proof.physical_dory_proof_count()
        );
        println!("proof_component_sizing_seconds={proof_component_sizing_seconds:.6}");
        println!("proof_encode_seconds={proof_encode_seconds:.6}");
        println!("proof_hash_seconds={proof_hash_seconds:.6}");
        println!("proof_decode_seconds={proof_decode_seconds:.6}");
        println!("trailing_decode_check_seconds={trailing_decode_check_seconds:.6}");
        println!("serialization_orchestration_seconds={serialization_orchestration_seconds:.6}");
        println!("serialization_seconds={serialization_seconds:.6}");
        println!("validity_verify_seconds={validity_verify_seconds:.6}");
        println!(
            "verifier_statement_validation_seconds={:.6}",
            verifier_profile.statement_validation_seconds
        );
        println!(
            "verifier_batch_point_seconds={:.6}",
            verifier_profile.batch_point_seconds
        );
        println!(
            "verifier_point_prepare_seconds={:.6}",
            verifier_profile.point_prepare_seconds
        );
        println!(
            "verifier_point_verify_seconds={:.6}",
            verifier_profile.point_verify_seconds
        );
        println!(
            "verifier_packed_eo_verify_seconds={:.6}",
            verifier_profile.packed_eo_verify_seconds
        );
        println!(
            "verifier_bva_open_verify_seconds={:.6}",
            verifier_profile.bva_open_verify_seconds
        );
        println!(
            "verifier_bva_open_point_preparation_seconds={:.6}",
            verifier_profile.bva_open.point_preparation_seconds
        );
        println!(
            "verifier_bva_open_transcript_preparation_seconds={:.6}",
            verifier_profile.bva_open.transcript_preparation_seconds
        );
        println!(
            "verifier_bva_open_dory_seconds={:.6}",
            verifier_profile.bva_open.dory_verify_seconds
        );
        println!(
            "verifier_bva_open_scalar_binding_seconds={:.6}",
            verifier_profile.bva_open.scalar_binding_seconds
        );
        println!(
            "verifier_bva_open_profile_other_seconds={:.6}",
            verifier_profile.bva_open.other_seconds()
        );
        println!(
            "verifier_bva_open_profile_total_seconds={:.6}",
            verifier_profile.bva_open.total_seconds
        );
        println!(
            "verifier_final_prepare_seconds={:.6}",
            verifier_profile.final_prepare_seconds
        );
        println!(
            "verifier_final_verify_seconds={:.6}",
            verifier_profile.final_verify_seconds
        );
        println!(
            "verifier_terminal_claim_collection_seconds={:.6}",
            verifier_profile.terminal_claim_collection_seconds
        );
        println!(
            "verifier_terminal_open_verify_seconds={:.6}",
            verifier_profile.terminal_open_verify_seconds
        );
        println!(
            "verifier_terminal_preparation_state_setup_seconds={:.6}",
            verifier_profile
                .terminal_open
                .preparation_state_setup_seconds
        );
        println!(
            "verifier_terminal_sumcheck_seconds={:.6}",
            verifier_profile.terminal_open.sumcheck_verify_seconds
        );
        println!(
            "verifier_terminal_dory_batch_seconds={:.6}",
            verifier_profile.terminal_open.dory_batch_verify_seconds
        );
        println!(
            "verifier_terminal_blindfold_preparation_seconds={:.6}",
            verifier_profile.terminal_open.blindfold_preparation_seconds
        );
        println!(
            "verifier_terminal_blindfold_seconds={:.6}",
            verifier_profile.terminal_open.blindfold_verify_seconds
        );
        println!(
            "verifier_terminal_result_finalization_seconds={:.6}",
            verifier_profile.terminal_open.result_finalization_seconds
        );
        println!(
            "verifier_terminal_profile_other_seconds={:.6}",
            (verifier_profile.terminal_open.total_seconds
                - verifier_profile.terminal_open.classified_seconds())
            .max(0.0)
        );
        println!(
            "verifier_terminal_profile_total_seconds={:.6}",
            verifier_profile.terminal_open.total_seconds
        );
        println!(
            "verifier_result_assembly_seconds={:.6}",
            verifier_profile.result_assembly_seconds
        );
        println!(
            "verifier_profile_total_seconds={:.6}",
            verifier_profile.total_seconds
        );
        println!("verifier_classified_seconds={verifier_classified_seconds:.6}");
        println!("verifier_orchestration_seconds={verifier_orchestration_seconds:.6}");
        println!("verifier_breakdown_sum_seconds={verifier_breakdown_sum_seconds:.6}");
        println!("opening_verify_seconds={opening_verify_seconds:.6}");
        println!("total_prover_seconds={total_prover_seconds:.6}");
        println!("total_verifier_seconds={total_verifier_seconds:.6}");
        println!("preprocessing_seconds={preprocessing_seconds:.6}");
        println!("end_to_end_classified_seconds={end_to_end_classified_seconds:.6}");
        println!("experiment_orchestration_seconds={experiment_orchestration_seconds:.6}");
        println!("end_to_end_breakdown_sum_seconds={end_to_end_breakdown_sum_seconds:.6}");
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
    }

    println!();
    println!("ParComm experiment: VERIFIED");
    println!("verified_marker=SECTION6_E2E_VERIFIED");
    let commit_other_seconds = (commit_phase_seconds
        - commit_profile.bva_commit_seconds
        - commit_profile.shape_commit_seconds)
        .max(0.0);
    let tbv_compilation_evaluation_seconds = point_conversion_seconds + tbv_evaluation_seconds;
    let opening_other_seconds = (opening_prove_seconds
        - opening_profile.bva_open_seconds
        - opening_profile.packed_eo_seconds
        - tbv_compilation_evaluation_seconds
        - opening_profile.terminal_open_seconds)
        .max(0.0);
    let reported_proof_size_bytes = component_sizes[1]
        .checked_add(component_sizes[2])
        .context("reported proof size overflowed")?;
    let tbv_compilation_evaluation_proof_size_bytes = x_statement_size_bytes
        .checked_add(tbv_statement_size_bytes)
        .and_then(|size| size.checked_add(point_reduction_size_bytes))
        .and_then(|size| size.checked_add(final_evaluation_size_bytes))
        .context("TBV compilation and evaluation proof size overflowed")?;
    let proof_other_size_bytes = component_sizes[1];
    let reported_proof_breakdown_size_bytes = bva_opening_size_bytes
        .checked_add(packed_eo_size_bytes)
        .and_then(|size| size.checked_add(tbv_compilation_evaluation_proof_size_bytes))
        .and_then(|size| size.checked_add(terminal_opening_size_bytes))
        .and_then(|size| size.checked_add(proof_other_size_bytes))
        .context("reported proof size breakdown overflowed")?;
    ensure!(
        reported_proof_breakdown_size_bytes == reported_proof_size_bytes,
        "reported proof size breakdown is not exhaustive"
    );
    let tbv_compilation_evaluation_verify_seconds = verifier_profile.point_prepare_seconds
        + verifier_profile.point_verify_seconds
        + verifier_profile.final_prepare_seconds
        + verifier_profile.final_verify_seconds;
    let grouped_verifier_seconds = verifier_profile.bva_open_verify_seconds
        + verifier_profile.packed_eo_verify_seconds
        + tbv_compilation_evaluation_verify_seconds
        + verifier_profile.terminal_open_verify_seconds;
    let verifier_other_seconds = (total_verifier_seconds - grouped_verifier_seconds).max(0.0);

    println!();
    println!("Final results");
    println!("| Metric | Value | Breakdown |");
    println!("| --- | ---: | --- |");
    println!(
        "| Commit time | {commit_phase_seconds:.3} s | BVA commitment {:.3} s; Shape commitment {:.3} s; Other {commit_other_seconds:.3} s |",
        commit_profile.bva_commit_seconds,
        commit_profile.shape_commit_seconds,
    );
    println!(
        "| Commit size | {} | Persistent commit statement |",
        format_bytes(component_sizes[0]),
    );
    println!(
        "| Open time | {opening_prove_seconds:.3} s | BVA open {:.3} s; EO-LiteBullet {:.3} s; TBV compilation + evaluation proof {tbv_compilation_evaluation_seconds:.3} s; Terminal open {:.3} s; Other {opening_other_seconds:.3} s |",
        opening_profile.bva_open_seconds,
        opening_profile.packed_eo_seconds,
        opening_profile.terminal_open_seconds,
    );
    println!(
        "| Proof size | {} | BVA open {}; EO-LiteBullet {}; TBV compilation + evaluation proof {}; Terminal open {}; Other {} |",
        format_bytes(reported_proof_size_bytes),
        format_bytes(bva_opening_size_bytes),
        format_bytes(packed_eo_size_bytes),
        format_bytes(tbv_compilation_evaluation_proof_size_bytes),
        format_bytes(terminal_opening_size_bytes),
        format_bytes(proof_other_size_bytes),
    );
    println!(
        "| Verifier time | {total_verifier_seconds:.3} s | BVA open verify {:.3} s; EO-LiteBullet verify {:.3} s; TBV compilation + evaluation proof verify {tbv_compilation_evaluation_verify_seconds:.3} s; Terminal open verify {:.3} s; Other {verifier_other_seconds:.3} s |",
        verifier_profile.bva_open_verify_seconds,
        verifier_profile.packed_eo_verify_seconds,
        verifier_profile.terminal_open_verify_seconds,
    );
    if peak_rss_kib == 0 {
        println!("| Peak RSS | unavailable | RSS sampling returned no value |");
    } else {
        println!(
            "| Peak RSS | {:.2} GiB | {peak_rss_kib} KiB |",
            peak_rss_kib as f64 / (1024.0 * 1024.0),
        );
    }
    io::stdout().flush().context("flush experiment output")?;
    Ok(())
}

fn parse_arguments(arguments: impl IntoIterator<Item = String>) -> Result<Option<Arguments>> {
    let mut arguments = arguments.into_iter().collect::<Vec<_>>().into_iter();
    let Some(command) = arguments.next() else {
        print_help();
        bail!("missing workload")
    };
    if command == "-h" || command == "--help" {
        print_help();
        return Ok(None);
    }
    let model = ModelPreset::from_str(&command).ok();
    let homogeneous = command == "homogeneous";
    ensure!(homogeneous || model.is_some(), "unknown workload {command}");

    let mut matrix_count = None;
    let mut rows = None;
    let mut cols = None;
    let mut model_data_dir = default_model_data_root();
    let mut manifest_override = None;
    let mut bva_override = None;
    let mut verify_input_hash = true;
    let mut target_log = None;
    let mut seed = DEFAULT_SEED;
    let mut allow_non_64_threads = false;
    let mut shape_validity_proof = false;
    let mut summary_only = false;
    while let Some(argument) = arguments.next() {
        let value = |name: &str, arguments: &mut std::vec::IntoIter<String>| {
            arguments
                .next()
                .with_context(|| format!("missing value after {name}"))
        };
        match argument.as_str() {
            "--matrix-count" => {
                matrix_count = Some(value("--matrix-count", &mut arguments)?.parse()?)
            }
            "--rows" => rows = Some(value("--rows", &mut arguments)?.parse()?),
            "--cols" => cols = Some(value("--cols", &mut arguments)?.parse()?),
            "--model-data-dir" => {
                model_data_dir = PathBuf::from(value("--model-data-dir", &mut arguments)?)
            }
            "--manifest" => {
                manifest_override = Some(PathBuf::from(value("--manifest", &mut arguments)?))
            }
            "--bva" => bva_override = Some(PathBuf::from(value("--bva", &mut arguments)?)),
            "--target-log" => target_log = Some(value("--target-log", &mut arguments)?.parse()?),
            "--seed" => seed = value("--seed", &mut arguments)?.parse()?,
            "--skip-input-hash" => verify_input_hash = false,
            "--verify-input-hash" => verify_input_hash = true,
            "--bench" => {
                // Cargo appends this marker even when the benchmark harness is disabled.
            }
            "--allow-non-64-threads" => allow_non_64_threads = true,
            "--shape-validity-proof" | "--shape-validity" => shape_validity_proof = true,
            "--summary-only" => summary_only = true,
            "-h" | "--help" => {
                print_help();
                return Ok(None);
            }
            _ => bail!("unknown argument {argument}; use --help for usage"),
        }
    }

    let workload = if homogeneous {
        let matrix_count = matrix_count.context("homogeneous requires --matrix-count")?;
        let rows = rows.context("homogeneous requires --rows")?;
        let cols = cols.context("homogeneous requires --cols")?;
        ensure!(matrix_count > 0 && rows > 0 && cols > 0);
        ensure!(
            manifest_override.is_none() && bva_override.is_none(),
            "homogeneous does not accept model artifact overrides"
        );
        Workload::Homogeneous {
            matrix_count,
            rows,
            cols,
        }
    } else {
        ensure!(
            matrix_count.is_none() && rows.is_none() && cols.is_none(),
            "model presets do not accept homogeneous shape flags"
        );
        Workload::Model(model.expect("a non-homogeneous command is a model preset"))
    };
    Ok(Some(Arguments {
        workload,
        model_data_dir,
        manifest_override,
        bva_override,
        verify_input_hash,
        target_log,
        seed,
        allow_non_64_threads,
        shape_validity_proof,
        summary_only,
    }))
}

fn print_help() {
    println!(
        "Usage:\n\
         RAYON_NUM_THREADS=64 cargo bench -p batch --bench experiment -- \\\n           homogeneous --matrix-count N --rows R --cols C [OPTIONS]\n\
         RAYON_NUM_THREADS=64 cargo bench -p batch --bench experiment -- \\\n           llama8b-i8|llama8b-i16|llama70b-i8 [OPTIONS]\n\
         \n\
         Detailed stage timings and proof-size components are printed by default.\n\
         \n\
         Options:\n\
           --model-data-dir PATH    Model-data root; defaults to workspace ../model-data\n\
           --manifest PATH          Override a model preset manifest\n\
           --bva PATH               Override the BVA named by the manifest\n\
           --target-log N           Assert the exact logical BVA domain\n\
           --seed N                 Deterministic prover seed\n\
           --verify-input-hash      Verify model BVA SHA-256 (default)\n\
           --skip-input-hash        Skip only the model BVA SHA-256 pass\n\
           --allow-non-64-threads   Permit a local smoke run\n\
           --shape-validity-proof   Include the optional shape validity proof\n\
           --summary-only           Hide detailed key=value metrics and retain Final results\n\
           -h, --help               Print this help"
    );
}

fn matrix_value(matrix: usize) -> i8 {
    let magnitude = (matrix % 31 + 1) as i8;
    if matrix.is_multiple_of(2) {
        magnitude
    } else {
        -magnitude
    }
}

fn physical_entry_count(shapes: &[(usize, usize)]) -> Result<usize> {
    shapes.iter().try_fold(0_usize, |total, &(rows, cols)| {
        total
            .checked_add(rows.checked_mul(cols).context("matrix size overflowed")?)
            .context("total matrix size overflowed")
    })
}

fn deterministic_points(matrix: usize, shape: (usize, usize)) -> (Vec<Fr>, Vec<Fr>) {
    let left = (0..shape.0.next_power_of_two().ilog2() as usize)
        .map(|coordinate| Fr::from((matrix * 37 + coordinate * 11 + 2) as u64))
        .collect();
    let right = (0..shape.1.next_power_of_two().ilog2() as usize)
        .map(|coordinate| Fr::from((matrix * 41 + coordinate * 13 + 5) as u64))
        .collect();
    (left, right)
}

fn claims_for_constant_matrices(
    shapes: &[(usize, usize)],
    point_log: usize,
) -> Result<Vec<Section6MatrixClaim<Fr>>> {
    shapes
        .iter()
        .enumerate()
        .map(|(matrix, &shape)| {
            let (left, right) = deterministic_points(matrix, shape);
            let value = constant_matrix_mle_evaluate(shape, matrix_value(matrix), &left, &right)?;
            Ok(Section6MatrixClaim::from_unpadded(
                point_log, &left, &right, value,
            )?)
        })
        .collect()
}

fn claims_for_prepacked_matrices(
    source: &ExperimentSource,
    point_log: usize,
) -> Result<Vec<Section6MatrixClaim<Fr>>> {
    source
        .descriptor()
        .matrix_shapes()
        .iter()
        .enumerate()
        .map(|(matrix, &shape)| {
            let (left, right) = deterministic_points(matrix, shape);
            let value = source.matrix_mle_evaluate(matrix, &left, &right)?;
            Ok(Section6MatrixClaim::from_unpadded(
                point_log, &left, &right, value,
            )?)
        })
        .collect()
}

fn serialized_size<T: serde::Serialize>(value: &T) -> Result<usize> {
    usize::try_from(proof_codec().serialized_size(value)?).context("proof size does not fit usize")
}

fn proof_codec() -> impl Options {
    bincode::DefaultOptions::new()
        .with_fixint_encoding()
        .reject_trailing_bytes()
}

fn encode_proof(
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
    let sizes = components.each_ref().map(Vec::len);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&PROOF_MAGIC);
    for size in sizes {
        bytes.extend_from_slice(&(size as u64).to_le_bytes());
    }
    for component in components {
        bytes.extend_from_slice(&component);
    }
    Ok((bytes, sizes))
}

fn decode_proof(
    bytes: &[u8],
) -> Result<(
    ValidityCommitStatement,
    Option<ShapeValidityProof>,
    ValidityOpeningProof,
)> {
    let header = PROOF_MAGIC.len() + 3 * size_of::<u64>();
    ensure!(bytes.len() >= header && bytes[..PROOF_MAGIC.len()] == PROOF_MAGIC);
    let mut sizes = [0_usize; 3];
    for (index, size) in sizes.iter_mut().enumerate() {
        let start = PROOF_MAGIC.len() + index * size_of::<u64>();
        *size = usize::try_from(u64::from_le_bytes(bytes[start..start + 8].try_into()?))?;
    }
    let expected = sizes
        .iter()
        .try_fold(header, |total, size| total.checked_add(*size))
        .context("proof size overflowed")?;
    ensure!(
        expected == bytes.len(),
        "proof has truncated or trailing bytes"
    );
    let statement_end = header + sizes[0];
    let validity_end = statement_end + sizes[1];
    Ok((
        proof_codec().deserialize(&bytes[header..statement_end])?,
        if sizes[1] == 0 {
            None
        } else {
            Some(proof_codec().deserialize(&bytes[statement_end..validity_end])?)
        },
        proof_codec().deserialize(&bytes[validity_end..])?,
    ))
}

fn report_stage(
    name: &str,
    started: Instant,
    monitor: &MemoryMonitor,
    detailed_output: bool,
) -> f64 {
    let seconds = started.elapsed().as_secs_f64();
    if detailed_output {
        println!("{name}_seconds={seconds:.6}");
        println!("peak_rss_after_{name}_kib={}", monitor.peak_kib());
        io::stdout().flush().expect("flush benchmark stage");
    }
    seconds
}

fn report_active_stage(index: usize, name: &str) -> Result<()> {
    println!("[stage {index}/{EXPERIMENT_STAGE_COUNT}] {name}");
    io::stdout()
        .flush()
        .context("flush active experiment stage")
}

fn format_bytes(bytes: usize) -> String {
    format!("{bytes} B ({:.2} KiB)", bytes as f64 / 1024.0)
}

struct MemoryMonitor {
    stop: Arc<AtomicBool>,
    peak: Arc<AtomicU64>,
    handle: Option<thread::JoinHandle<()>>,
}

impl MemoryMonitor {
    fn spawn(period: Duration) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let peak = Arc::new(AtomicU64::new(0));
        let thread_stop = Arc::clone(&stop);
        let thread_peak = Arc::clone(&peak);
        let handle = thread::spawn(move || {
            while !thread_stop.load(Ordering::Relaxed) {
                if let Some(rss) = read_rss_kib() {
                    thread_peak.fetch_max(rss, Ordering::Relaxed);
                }
                thread::sleep(period);
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
    let pid = std::process::id().to_string();
    let output = Command::new("ps")
        .args(["-o", "rss=", "-p", &pid])
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
    #[allow(unused_imports)]
    use super::{parse_arguments, Workload};

    #[test]
    fn parses_rectangular_homogeneous_workload() {
        let arguments = parse_arguments(
            [
                "homogeneous",
                "--matrix-count",
                "7",
                "--rows",
                "13",
                "--cols",
                "9",
                "--allow-non-64-threads",
                "--summary-only",
            ]
            .into_iter()
            .map(String::from),
        )
        .unwrap()
        .unwrap();
        assert_eq!(
            arguments.workload,
            Workload::Homogeneous {
                matrix_count: 7,
                rows: 13,
                cols: 9
            }
        );
        assert!(!arguments.shape_validity_proof);
        assert!(arguments.summary_only);
    }

    #[test]
    fn parses_each_model_preset_without_enabling_shape_validity() {
        for preset in ["llama8b-i8", "llama8b-i16", "llama70b-i8"] {
            let arguments = parse_arguments([preset].into_iter().map(String::from))
                .unwrap()
                .unwrap();
            assert!(matches!(arguments.workload, Workload::Model(_)));
            assert!(!arguments.shape_validity_proof);
            assert!(!arguments.summary_only);
            assert!(arguments.verify_input_hash);
        }
    }
}
