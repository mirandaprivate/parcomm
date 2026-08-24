use std::{
    collections::BTreeSet,
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::Instant,
};

use anyhow::{bail, ensure, Context, Result};
use batch::{
    matrix_collection::{DenseBvaDescriptor, DenseMatrixBva},
    model_bva::{default_model_data_root, ModelPreset},
};
use serde::Deserialize;
use sha2::{Digest, Sha256};

#[allow(dead_code)]
#[path = "experiment.rs"]
mod experiment;

const LLAMA_8B_EXPECTED_LOGICAL_LOG: usize = 33;
const LLAMA_70B_EXPECTED_LOGICAL_LOG: usize = 37;
const PIPELINE_STAGE_COUNT: usize = 2;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StreamMode {
    Llama8bI8,
    Llama8bI16,
    Llama70bI8,
    Llama70bI16,
}

impl StreamMode {
    fn preset(self) -> ModelPreset {
        match self {
            Self::Llama8bI8 => ModelPreset::Llama8bI8,
            Self::Llama8bI16 => ModelPreset::Llama8bI16,
            Self::Llama70bI8 => ModelPreset::Llama70bI8,
            Self::Llama70bI16 => ModelPreset::Llama70bI16,
        }
    }

    fn model_family(self) -> &'static str {
        match self {
            Self::Llama8bI8 | Self::Llama70bI8 => "llama-w8a8",
            Self::Llama8bI16 | Self::Llama70bI16 => "llama-bf16",
        }
    }

    fn storage_dtype(self) -> &'static str {
        match self {
            Self::Llama8bI8 | Self::Llama70bI8 => "i8",
            Self::Llama8bI16 | Self::Llama70bI16 => "i16",
        }
    }

    fn element_bytes(self) -> usize {
        match self {
            Self::Llama8bI8 | Self::Llama70bI8 => 1,
            Self::Llama8bI16 | Self::Llama70bI16 => 2,
        }
    }

    fn bit_width(self) -> usize {
        self.element_bytes() * 8
    }

    fn source_directory(self) -> &'static str {
        match self {
            Self::Llama8bI8 => "Llama-3.1-8B-W8A8",
            Self::Llama8bI16 => "Llama-3.1-8B-BF16",
            Self::Llama70bI8 => "Llama-3.1-70B-W8A8",
            Self::Llama70bI16 => "Llama-3.1-70B",
        }
    }

    fn bench_name(self) -> &'static str {
        match self {
            Self::Llama8bI8 => "experiment_llama8B_8bit",
            Self::Llama8bI16 => "experiment_llama8B_16bit",
            Self::Llama70bI8 => "experiment_llama70B_8bit",
            Self::Llama70bI16 => "experiment_llama70B_16bit",
        }
    }

    fn model_label(self) -> &'static str {
        match self {
            Self::Llama8bI8 | Self::Llama8bI16 => "Llama-8B",
            Self::Llama70bI8 | Self::Llama70bI16 => "Llama-70B",
        }
    }

    fn pipeline_prefix(self) -> &'static str {
        match self {
            Self::Llama8bI8 | Self::Llama8bI16 => "llama8b",
            Self::Llama70bI8 | Self::Llama70bI16 => "llama70b",
        }
    }

    fn expected_logical_log(self) -> usize {
        match self {
            Self::Llama8bI8 | Self::Llama8bI16 => LLAMA_8B_EXPECTED_LOGICAL_LOG,
            Self::Llama70bI8 | Self::Llama70bI16 => LLAMA_70B_EXPECTED_LOGICAL_LOG,
        }
    }

    fn formal_min_available_gib(self) -> u64 {
        match self {
            Self::Llama8bI8 => 40,
            Self::Llama8bI16 => 64,
            Self::Llama70bI8 | Self::Llama70bI16 => 320,
        }
    }

    fn is_i8(self) -> bool {
        matches!(self, Self::Llama8bI8 | Self::Llama70bI8)
    }
}

#[derive(Debug)]
struct Arguments {
    input: PathBuf,
    python: PathBuf,
    converter: PathBuf,
    chunk_mib: usize,
    allow_non_64_threads: bool,
    allow_noncanonical_layout: bool,
    experiment_arguments: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct StreamDescription {
    format: String,
    version: u64,
    model_family: String,
    storage_dtype: String,
    element_bytes: usize,
    native_i8_tensor_count: usize,
    converted_float_tensor_count: usize,
    bitcast_bf16_tensor_count: usize,
    shapes: Vec<[usize; 2]>,
    stream_entries: usize,
    stream_bytes: u64,
    layout: StreamLayout,
}

#[derive(Debug, Deserialize)]
struct StreamLayout {
    tensor_count: usize,
    bucket_count: usize,
    physical_entries: usize,
    active_entries: usize,
    logical_entries: usize,
    boundary_padding_entries: usize,
    final_padding_entries: usize,
    splitting_budget: usize,
    bucket_order: String,
    within_bucket_order: String,
    rank1_matrix_mapping: String,
    element_dtype: String,
    element_bytes: usize,
    stream_entries: usize,
    stream_bytes: u64,
    logical_bytes: u64,
    final_padding_streamed: bool,
}

#[derive(Clone, Debug)]
struct StreamConfig {
    mode: StreamMode,
    input: PathBuf,
    python: PathBuf,
    converter: PathBuf,
    chunk_mib: usize,
    shapes: Vec<(usize, usize)>,
    active_entries: usize,
}

fn main() -> Result<()> {
    run_from_args(StreamMode::Llama70bI8, std::env::args().skip(1))
}

pub(crate) fn run_i16_from_args(arguments: impl IntoIterator<Item = String>) -> Result<()> {
    run_from_args(StreamMode::Llama70bI16, arguments)
}

pub(crate) fn run_8b_i8_from_args(arguments: impl IntoIterator<Item = String>) -> Result<()> {
    run_from_args(StreamMode::Llama8bI8, arguments)
}

pub(crate) fn run_8b_i16_from_args(arguments: impl IntoIterator<Item = String>) -> Result<()> {
    run_from_args(StreamMode::Llama8bI16, arguments)
}

fn run_from_args(mode: StreamMode, arguments: impl IntoIterator<Item = String>) -> Result<()> {
    let Some(arguments) = parse_arguments(mode, arguments)? else {
        return Ok(());
    };
    run(mode, arguments)
}

fn run(mode: StreamMode, arguments: Arguments) -> Result<()> {
    let pipeline_started = Instant::now();
    let input = fs::canonicalize(&arguments.input)
        .with_context(|| format!("canonicalize checkpoint {}", arguments.input.display()))?;
    let converter = fs::canonicalize(&arguments.converter)
        .with_context(|| format!("canonicalize converter {}", arguments.converter.display()))?;
    ensure!(
        input.is_dir(),
        "checkpoint must be a directory: {}",
        input.display()
    );
    ensure!(
        converter.is_file(),
        "Python converter does not exist: {}",
        converter.display()
    );

    println!("llama_pipeline_output_mode=detailed");
    println!(
        "pipeline={}-{}-safetensors-to-memory",
        mode.pipeline_prefix(),
        mode.storage_dtype()
    );
    println!("model_source={}", input.display());
    println!("python_executable={}", arguments.python.display());
    println!("python_converter={}", converter.display());
    println!("conversion_transport=python_stdout_active_prefix");
    println!("bva_storage=contiguous_in_memory_{}", mode.storage_dtype());
    println!("bva_disk_materialized=false");
    println!("bva_disk_bytes_written=0");
    io_flush()?;

    if !arguments.allow_non_64_threads {
        let required_gib = mode.formal_min_available_gib();
        let available_kib = available_memory_kib().context("read /proc/meminfo")?;
        let required_kib = required_gib * 1024 * 1024;
        println!("preflight_available_memory_kib={available_kib}");
        println!("preflight_required_memory_kib={required_kib}");
        ensure!(
            available_kib >= required_kib,
            "formal {} run requires at least {required_gib} GiB available memory; only {:.2} GiB is available",
            mode.model_label(),
            available_kib as f64 / (1024.0 * 1024.0)
        );
    }

    let artifacts_before = collect_intermediate_artifacts(&input)?;
    report_stage(1, "Describe checkpoint and canonical BVA layout")?;
    let description_started = Instant::now();
    let description = describe_checkpoint(mode, &arguments.python, &converter, &input)?;
    let description_seconds = description_started.elapsed().as_secs_f64();
    let shapes = validate_description(mode, &description, !arguments.allow_noncanonical_layout)?;
    let logical_log = description.layout.logical_entries.ilog2() as usize;
    println!("description_seconds={description_seconds:.6}");
    println!("selected_tensors={}", description.layout.tensor_count);
    println!("native_i8_tensors={}", description.native_i8_tensor_count);
    println!(
        "converted_float_tensors={}",
        description.converted_float_tensor_count
    );
    println!("physical_entries={}", description.layout.physical_entries);
    println!("active_entries={}", description.layout.active_entries);
    println!("logical_entries={}", description.layout.logical_entries);
    println!("bucket_count={}", description.layout.bucket_count);
    println!("stream_bytes={}", description.stream_bytes);
    println!("logical_bva_bytes={}", description.layout.logical_bytes);
    io_flush()?;

    let stream_config = StreamConfig {
        mode,
        input: input.clone(),
        python: arguments.python.clone(),
        converter: converter.clone(),
        chunk_mib: arguments.chunk_mib,
        shapes: shapes.clone(),
        active_entries: description.layout.active_entries,
    };
    let in_memory_input = experiment::InMemoryModelInput {
        preset: mode.preset(),
        storage_dtype: mode.storage_dtype(),
        bit_width: mode.bit_width(),
        source_path: input.clone(),
        matrix_shapes: shapes,
        physical_entries: description.layout.physical_entries,
        active_entries: description.layout.active_entries,
        logical_entries: description.layout.logical_entries,
        bucket_count: description.layout.bucket_count,
        native_i8_tensor_count: description.native_i8_tensor_count,
        converted_float_tensor_count: description.converted_float_tensor_count,
        bitcast_bf16_tensor_count: description.bitcast_bf16_tensor_count,
        loader: Box::new(move || stream_checkpoint_into_memory(stream_config)),
    };

    report_stage(
        2,
        "Stream safetensors into memory and run the ParComm experiment",
    )?;
    let mut experiment_arguments = vec![
        mode.preset().as_str().to_owned(),
        "--target-log".to_owned(),
        logical_log.to_string(),
    ];
    experiment_arguments.extend(arguments.experiment_arguments);
    experiment::run_from_args_with_in_memory_model(experiment_arguments, in_memory_input)?;

    let artifacts_after = collect_intermediate_artifacts(&input)?;
    ensure!(
        artifacts_after == artifacts_before,
        "the in-memory benchmark changed the checkpoint's BVA/manifest/part artifacts"
    );
    println!("bva_disk_artifacts_before={}", artifacts_before.len());
    println!("bva_disk_artifacts_after={}", artifacts_after.len());
    println!("bva_disk_bytes_written=0");
    println!("in_memory_pipeline_verified=true");
    println!(
        "{}_pipeline_total_seconds={:.6}",
        mode.pipeline_prefix(),
        pipeline_started.elapsed().as_secs_f64()
    );
    io_flush()
}

fn describe_checkpoint(
    mode: StreamMode,
    python: &Path,
    converter: &Path,
    input: &Path,
) -> Result<StreamDescription> {
    let output = Command::new(python)
        .env("PYTHONUNBUFFERED", "1")
        .arg(converter)
        .arg("describe-json")
        .arg("--input")
        .arg(input)
        .arg("--model-family")
        .arg(mode.model_family())
        .arg("--storage-dtype")
        .arg(mode.storage_dtype())
        .output()
        .with_context(|| format!("run {} describe-json", converter.display()))?;
    ensure!(
        output.status.success(),
        "checkpoint description failed with status {}: {}",
        output.status,
        String::from_utf8_lossy(&output.stderr).trim()
    );
    serde_json::from_slice(&output.stdout).with_context(|| {
        format!(
            "parse converter description: {}",
            String::from_utf8_lossy(&output.stdout)
                .chars()
                .take(200)
                .collect::<String>()
        )
    })
}

fn validate_description(
    mode: StreamMode,
    description: &StreamDescription,
    require_canonical_layout: bool,
) -> Result<Vec<(usize, usize)>> {
    ensure!(
        description.format == "parcomm-dense-bva-stream-description" && description.version == 1,
        "unsupported converter stream description"
    );
    ensure!(
        description.model_family == mode.model_family()
            && description.storage_dtype == mode.storage_dtype()
            && description.element_bytes == mode.element_bytes(),
        "converter description does not match the requested Llama stream mode"
    );
    if mode.is_i8() {
        ensure!(
            description.bitcast_bf16_tensor_count == 0
                && description.native_i8_tensor_count + description.converted_float_tensor_count
                    == description.layout.tensor_count,
            "converter tensor conversion counts are inconsistent"
        );
    } else {
        ensure!(
            description.native_i8_tensor_count == 0
                && description.converted_float_tensor_count == 0
                && description.bitcast_bf16_tensor_count == description.layout.tensor_count,
            "converter BF16 bitcast counts are inconsistent"
        );
    }
    ensure!(
        description.shapes.len() == description.layout.tensor_count,
        "converter shape count does not match tensor count"
    );
    ensure!(
        description.stream_entries == description.layout.active_entries
            && description.layout.stream_entries == description.layout.active_entries
            && description.stream_bytes == description.layout.stream_bytes
            && description.stream_bytes
                == (description.stream_entries * mode.element_bytes()) as u64,
        "converter stream length metadata is inconsistent"
    );
    ensure!(
        description.layout.logical_bytes
            == (description.layout.logical_entries * mode.element_bytes()) as u64
            && !description.layout.final_padding_streamed,
        "the converter must leave the final zero suffix implicit"
    );
    ensure!(
        description.layout.boundary_padding_entries
            == description.layout.active_entries - description.layout.physical_entries
            && description.layout.final_padding_entries
                == description.layout.logical_entries - description.layout.active_entries,
        "converter padding metadata is inconsistent"
    );
    ensure!(
        description.layout.splitting_budget == DenseBvaDescriptor::SPLITTING_BUDGET
            && description.layout.bucket_order
                == "entry_count_desc_then_tensor_index_then_row_offset_then_col_offset"
            && description.layout.within_bucket_order == "row_major"
            && description.layout.rank1_matrix_mapping == "shape_n_to_n_by_1"
            && description.layout.element_dtype == mode.storage_dtype()
            && description.layout.element_bytes == mode.element_bytes(),
        "converter layout conventions differ from the Rust BVA"
    );

    let shapes = description
        .shapes
        .iter()
        .map(|shape| (shape[0], shape[1]))
        .collect::<Vec<_>>();
    let descriptor = DenseBvaDescriptor::from_shapes(&shapes)?;
    ensure!(
        descriptor.matrix_count() == description.layout.tensor_count
            && descriptor.bucket_count() == description.layout.bucket_count
            && descriptor.physical_len() == description.layout.physical_entries
            && descriptor.layout().active_len() == description.layout.active_entries
            && descriptor.logical_len() == description.layout.logical_entries,
        "Python and Rust computed different canonical BVA layouts"
    );
    if require_canonical_layout {
        let expected_logical_log = mode.expected_logical_log();
        ensure!(
            descriptor.required_dory_log() == expected_logical_log,
            "{} checkpoint has BVA log {}, expected {expected_logical_log}",
            mode.model_label(),
            descriptor.required_dory_log()
        );
    }
    Ok(shapes)
}

fn stream_checkpoint_into_memory(config: StreamConfig) -> Result<experiment::LoadedInMemoryModel> {
    let total_started = Instant::now();
    let (source, allocation_seconds, stream_seconds, adopt_seconds, bytes_streamed, stream_sha256) =
        if config.mode.is_i8() {
            let allocation_started = Instant::now();
            let mut values = Vec::<i8>::new();
            values
                .try_reserve_exact(config.active_entries)
                .with_context(|| format!("reserve {} i8 BVA entries", config.active_entries))?;
            values.resize(config.active_entries, 0);
            let allocation_seconds = allocation_started.elapsed().as_secs_f64();
            println!("in_memory_bva_allocation_seconds={allocation_seconds:.6}");
            println!("in_memory_bva_allocated_bytes={}", values.len());
            io_flush()?;
            let (stream_seconds, stream_sha256) =
                stream_converter_bytes(&config, bytemuck::cast_slice_mut(&mut values))?;
            let adopt_started = Instant::now();
            let source = DenseMatrixBva::from_prepacked_i8(&config.shapes, values)?;
            let adopt_seconds = adopt_started.elapsed().as_secs_f64();
            (
                experiment::LoadedInMemoryBva::I8(source),
                allocation_seconds,
                stream_seconds,
                adopt_seconds,
                config.active_entries as u64,
                stream_sha256,
            )
        } else {
            ensure!(
                cfg!(target_endian = "little"),
                "native i16 BF16-bit streaming requires a little-endian target"
            );
            let allocation_started = Instant::now();
            let mut values = Vec::<i16>::new();
            values
                .try_reserve_exact(config.active_entries)
                .with_context(|| format!("reserve {} i16 BVA entries", config.active_entries))?;
            values.resize(config.active_entries, 0);
            let allocation_seconds = allocation_started.elapsed().as_secs_f64();
            let allocated_bytes = config
                .active_entries
                .checked_mul(std::mem::size_of::<i16>())
                .context("i16 BVA allocation size overflowed")?;
            println!("in_memory_bva_allocation_seconds={allocation_seconds:.6}");
            println!("in_memory_bva_allocated_bytes={allocated_bytes}");
            io_flush()?;
            let (stream_seconds, stream_sha256) =
                stream_converter_bytes(&config, bytemuck::cast_slice_mut(&mut values))?;
            let adopt_started = Instant::now();
            let source = DenseMatrixBva::from_prepacked_i16(&config.shapes, values)?;
            let adopt_seconds = adopt_started.elapsed().as_secs_f64();
            (
                experiment::LoadedInMemoryBva::I16(source),
                allocation_seconds,
                stream_seconds,
                adopt_seconds,
                allocated_bytes as u64,
                stream_sha256,
            )
        };
    let total_seconds = total_started.elapsed().as_secs_f64();
    println!("in_memory_bva_stream_seconds={stream_seconds:.6}");
    println!("in_memory_bva_adopt_seconds={adopt_seconds:.6}");
    println!("in_memory_bva_loader_seconds={total_seconds:.6}");
    println!("in_memory_bva_stream_sha256={stream_sha256}");
    println!("in_memory_bva_stream_complete=true");
    println!("bva_disk_materialized=false");
    io_flush()?;

    Ok(experiment::LoadedInMemoryModel {
        source,
        profile: experiment::InMemoryLoadProfile {
            allocation_seconds,
            stream_seconds,
            adopt_seconds,
            total_seconds,
            bytes_streamed,
            stream_sha256,
        },
    })
}

fn stream_converter_bytes(config: &StreamConfig, target: &mut [u8]) -> Result<(f64, String)> {
    let mut child = Command::new(&config.python)
        .env("PYTHONUNBUFFERED", "1")
        .arg(&config.converter)
        .arg("stream")
        .arg("--input")
        .arg(&config.input)
        .arg("--model-family")
        .arg(config.mode.model_family())
        .arg("--storage-dtype")
        .arg(config.mode.storage_dtype())
        .arg("--chunk-mib")
        .arg(config.chunk_mib.to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .with_context(|| format!("spawn {} stream", config.converter.display()))?;
    let mut stdout = child.stdout.take().context("capture converter stdout")?;
    let stream_started = Instant::now();
    let mut digest = Sha256::new();
    let mut offset = 0_usize;
    let stream_result = (|| -> Result<()> {
        while offset < target.len() {
            let read = stdout
                .read(&mut target[offset..])
                .context("read converter BVA stream")?;
            ensure!(
                read != 0,
                "converter stream ended after {offset} bytes; expected {}",
                target.len()
            );
            digest.update(&target[offset..offset + read]);
            offset += read;
        }
        let mut trailing = [0_u8; 1];
        ensure!(
            stdout
                .read(&mut trailing)
                .context("check converter stream EOF")?
                == 0,
            "converter emitted bytes beyond the declared active BVA prefix"
        );
        Ok(())
    })();
    drop(stdout);
    if let Err(error) = stream_result {
        let _ = child.kill();
        let status = child.wait().context("reap failed converter")?;
        return Err(error).context(format!("converter exited with status {status}"));
    }
    let status = child.wait().context("wait for converter stream")?;
    ensure!(
        status.success(),
        "converter stream failed with status {status}"
    );
    Ok((
        stream_started.elapsed().as_secs_f64(),
        format!("{:x}", digest.finalize()),
    ))
}

fn parse_arguments(
    mode: StreamMode,
    arguments: impl IntoIterator<Item = String>,
) -> Result<Option<Arguments>> {
    let mut arguments = arguments.into_iter().collect::<Vec<_>>().into_iter();
    let crate_directory = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_directory = crate_directory
        .parent()
        .context("the batch crate must have a workspace parent")?;
    let mut model_data_dir = default_model_data_root();
    let mut input_override = None;
    let mut python = PathBuf::from("python3");
    let mut converter = workspace_directory.join("preprocessing/modelconvert.py");
    let mut chunk_mib = 64_usize;
    let mut allow_non_64_threads = false;
    let mut allow_noncanonical_layout = false;
    let mut experiment_arguments = Vec::new();

    while let Some(argument) = arguments.next() {
        let value = |name: &str, arguments: &mut std::vec::IntoIter<String>| {
            arguments
                .next()
                .with_context(|| format!("missing value after {name}"))
        };
        match argument.as_str() {
            "--model-data-dir" => {
                model_data_dir = PathBuf::from(value("--model-data-dir", &mut arguments)?)
            }
            "--input" => input_override = Some(PathBuf::from(value("--input", &mut arguments)?)),
            "--python" => python = PathBuf::from(value("--python", &mut arguments)?),
            "--converter" => converter = PathBuf::from(value("--converter", &mut arguments)?),
            "--chunk-mib" => {
                chunk_mib = value("--chunk-mib", &mut arguments)?.parse()?;
                ensure!(chunk_mib > 0, "--chunk-mib must be positive");
            }
            "--target-log" => {
                let requested: usize = value("--target-log", &mut arguments)?.parse()?;
                let expected_logical_log = mode.expected_logical_log();
                ensure!(
                    requested == expected_logical_log,
                    "{} in-memory BVA requires --target-log {expected_logical_log}",
                    mode.model_label()
                );
            }
            "--seed" => {
                experiment_arguments.push(argument);
                experiment_arguments.push(value("--seed", &mut arguments)?);
            }
            "--allow-non-64-threads" => {
                allow_non_64_threads = true;
                experiment_arguments.push(argument);
            }
            "--allow-noncanonical-layout" | "--allow-non-70b-layout" => {
                allow_noncanonical_layout = true
            }
            "--shape-validity-proof" | "--shape-validity" | "--summary-only" => {
                experiment_arguments.push(argument)
            }
            "--bench" => {}
            "--manifest" | "--bva" | "--conversion-output-dir" | "--force-conversion" => {
                bail!("{argument} is disabled: this benchmark never materializes a disk BVA")
            }
            "-h" | "--help" => {
                print_help(mode);
                return Ok(None);
            }
            _ => bail!("unknown argument {argument}; use --help for usage"),
        }
    }
    let input = input_override.unwrap_or_else(|| model_data_dir.join(mode.source_directory()));
    ensure!(
        !allow_noncanonical_layout || allow_non_64_threads,
        "--allow-noncanonical-layout is only available with --allow-non-64-threads"
    );
    Ok(Some(Arguments {
        input,
        python,
        converter,
        chunk_mib,
        allow_non_64_threads,
        allow_noncanonical_layout,
        experiment_arguments,
    }))
}

fn available_memory_kib() -> Option<u64> {
    fs::read_to_string("/proc/meminfo")
        .ok()?
        .lines()
        .find_map(|line| {
            line.strip_prefix("MemAvailable:")?
                .split_whitespace()
                .next()?
                .parse()
                .ok()
        })
}

fn collect_intermediate_artifacts(root: &Path) -> Result<BTreeSet<PathBuf>> {
    fn visit(directory: &Path, artifacts: &mut BTreeSet<PathBuf>) -> Result<()> {
        for entry in fs::read_dir(directory)
            .with_context(|| format!("read checkpoint directory {}", directory.display()))?
        {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                visit(&path, artifacts)?;
                continue;
            }
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.contains(".bva.") || name.ends_with(".part") || name.ends_with(".manifest.json")
            {
                artifacts.insert(path);
            }
        }
        Ok(())
    }

    let mut artifacts = BTreeSet::new();
    visit(root, &mut artifacts)?;
    Ok(artifacts)
}

fn report_stage(index: usize, name: &str) -> Result<()> {
    println!("[pipeline stage {index}/{PIPELINE_STAGE_COUNT}] {name}");
    io_flush()
}

fn io_flush() -> Result<()> {
    std::io::stdout().flush().context("flush benchmark output")
}

fn print_help(mode: StreamMode) {
    let expected_logical_log = mode.expected_logical_log();
    println!(
        "Usage:\n\
         RAYON_NUM_THREADS=64 cargo bench -p batch --bench {} -- [OPTIONS]\n\
         \n\
         Streams the {} checkpoint directly into one contiguous in-memory\n\
         {} BVA active prefix and immediately runs commitment/opening.\n\
         No .bva, manifest, or partial conversion artifact is written to disk.\n\
         \n\
         Options:\n\
           --model-data-dir PATH    Model-data root; defaults to workspace ../model-data\n\
           --input PATH             Override the checkpoint directory\n\
           --python PATH            Python executable (default: python3)\n\
           --converter PATH         Converter path (default: preprocessing/modelconvert.py)\n\
           --chunk-mib N            Streaming conversion chunk size (default: 64)\n\
           --target-log N           Assert the exact 2^{expected_logical_log} BVA domain\n\
           --seed N                 Deterministic prover seed\n\
           --allow-non-64-threads   Permit a local smoke run and skip the formal RAM gate\n\
           --allow-noncanonical-layout\n\
                                   Permit a tiny checkpoint only for a local smoke run\n\
           --shape-validity-proof   Include the optional shape-validity proof\n\
           --summary-only           Hide detailed protocol metrics\n\
           -h, --help               Print this help",
        mode.bench_name(),
        mode.model_label(),
        mode.storage_dtype(),
    );
}
