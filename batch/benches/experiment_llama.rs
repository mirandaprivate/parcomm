use std::{
    fs::{self, File},
    io::{self, Write},
    path::{Path, PathBuf},
    process::Command,
    str::FromStr,
    time::Instant,
};

use anyhow::{bail, ensure, Context, Result};
use batch::model_bva::{default_model_data_root, ModelBvaManifest, ModelPreset};
use serde_json::Value;

#[allow(dead_code)]
#[path = "experiment.rs"]
mod experiment;

const LLAMA_STAGE_COUNT: usize = 2;

#[derive(Debug)]
struct Arguments {
    preset: ModelPreset,
    model_data_dir: PathBuf,
    input_override: Option<PathBuf>,
    python: PathBuf,
    converter: PathBuf,
    conversion_output_dir: Option<PathBuf>,
    conversion_output_name: Option<String>,
    chunk_mib: usize,
    force_conversion: bool,
    experiment_arguments: Vec<String>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ConversionStatus {
    Cached,
    Converted,
}

impl ConversionStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Cached => "cached",
            Self::Converted => "converted",
        }
    }
}

#[derive(Clone, Copy)]
struct ConversionSpec {
    source_directory: &'static str,
    output_name: &'static str,
    model_family: &'static str,
    storage_dtype: &'static str,
}

struct ConversionArtifacts {
    output_directory: PathBuf,
    output_name: String,
    manifest_path: PathBuf,
    bva_path: PathBuf,
}

fn main() -> Result<()> {
    run_from_args(std::env::args().skip(1))
}

pub(crate) fn run_from_args(arguments: impl IntoIterator<Item = String>) -> Result<()> {
    let Some(arguments) = parse_arguments(arguments)? else {
        return Ok(());
    };

    let summary_only = arguments
        .experiment_arguments
        .iter()
        .any(|argument| argument == "--summary-only");
    println!(
        "llama_pipeline_output_mode={}",
        if summary_only { "summary" } else { "detailed" }
    );
    let spec = conversion_spec(arguments.preset)?;
    let artifacts = conversion_artifacts(&arguments, spec)?;
    report_stage(1, "Prepare Llama BVA with Python")?;
    let conversion_started = Instant::now();
    let status = prepare_conversion(&arguments, spec, &artifacts)?;
    let conversion_seconds = conversion_started.elapsed().as_secs_f64();
    let artifacts = canonicalize_conversion_artifacts(artifacts)?;
    if matches!(
        arguments.preset,
        ModelPreset::Llama8bI16 | ModelPreset::Llama70bI16
    ) {
        validate_bf16_bitcast_manifest(&artifacts.manifest_path)?;
    }
    let manifest = ModelBvaManifest::load(
        arguments.preset,
        &arguments.model_data_dir,
        Some(&artifacts.manifest_path),
        Some(&artifacts.bva_path),
    )
    .with_context(|| {
        format!(
            "validate converted manifest {}",
            artifacts.manifest_path.display()
        )
    })?;
    println!("conversion_status={}", status.as_str());
    println!("conversion_seconds={conversion_seconds:.6}");
    println!("conversion_manifest={}", manifest.manifest_path().display());
    println!("conversion_bva={}", manifest.bva_path().display());
    println!("conversion_bva_size_bytes={}", manifest.bva_size_bytes());
    io::stdout().flush().context("flush conversion summary")?;

    report_stage(2, "Run the ParComm end-to-end experiment")?;
    let mut experiment_arguments = vec![
        arguments.preset.to_string(),
        "--model-data-dir".to_owned(),
        arguments.model_data_dir.to_string_lossy().into_owned(),
        "--manifest".to_owned(),
        artifacts.manifest_path.to_string_lossy().into_owned(),
        "--bva".to_owned(),
        artifacts.bva_path.to_string_lossy().into_owned(),
    ];
    experiment_arguments.extend(arguments.experiment_arguments);
    experiment::run_from_args(experiment_arguments)
}

fn prepare_conversion(
    arguments: &Arguments,
    spec: ConversionSpec,
    artifacts: &ConversionArtifacts,
) -> Result<ConversionStatus> {
    let source_path = arguments
        .input_override
        .clone()
        .unwrap_or_else(|| arguments.model_data_dir.join(spec.source_directory));
    let output_directory = &artifacts.output_directory;
    let manifest_path = &artifacts.manifest_path;
    let bva_path = &artifacts.bva_path;
    let partial_path = bva_path.with_file_name(format!(
        "{}.part",
        bva_path
            .file_name()
            .context("the BVA path must have a filename")?
            .to_string_lossy()
    ));
    let manifest_partial_path = manifest_path.with_file_name(format!(
        "{}.part",
        manifest_path
            .file_name()
            .context("the manifest path must have a filename")?
            .to_string_lossy()
    ));

    println!("python_executable={}", arguments.python.display());
    println!("python_converter={}", arguments.converter.display());
    println!("model_source={}", source_path.display());
    println!("conversion_output_dir={}", output_directory.display());
    println!("conversion_force={}", arguments.force_conversion);
    io::stdout().flush().context("flush conversion paths")?;

    let manifest_exists = manifest_path.is_file();
    let bva_exists = bva_path.is_file();
    if manifest_exists && bva_exists && !arguments.force_conversion {
        if arguments.input_override.is_some() {
            ensure_cached_source_matches(manifest_path, &source_path)?;
        }
        return Ok(ConversionStatus::Cached);
    }
    if !arguments.force_conversion
        && (manifest_exists
            || bva_exists
            || partial_path.exists()
            || manifest_partial_path.exists())
    {
        bail!(
            "incomplete conversion artifacts exist under {}; remove them or pass --force-conversion",
            output_directory.display()
        );
    }

    ensure!(
        source_path.exists(),
        "Llama checkpoint does not exist: {}",
        source_path.display()
    );
    ensure!(
        arguments.converter.is_file(),
        "Python converter does not exist: {}",
        arguments.converter.display()
    );

    let mut command = Command::new(&arguments.python);
    command
        .env("PYTHONUNBUFFERED", "1")
        .arg(&arguments.converter)
        .arg("convert")
        .arg("--input")
        .arg(&source_path)
        .arg("--model-family")
        .arg(spec.model_family)
        .arg("--storage-dtype")
        .arg(spec.storage_dtype)
        .arg("--output-dir")
        .arg(output_directory)
        .arg("--name")
        .arg(&artifacts.output_name)
        .arg("--chunk-mib")
        .arg(arguments.chunk_mib.to_string());
    if arguments.force_conversion {
        command.arg("--force");
    }
    let status = command
        .status()
        .with_context(|| format!("run Python converter with {}", arguments.python.display()))?;
    ensure!(
        status.success(),
        "Python conversion failed with status {status}"
    );
    ensure!(
        manifest_path.is_file() && bva_path.is_file(),
        "Python conversion succeeded without producing the requested manifest and BVA"
    );
    Ok(ConversionStatus::Converted)
}

fn ensure_cached_source_matches(manifest_path: &Path, source_path: &Path) -> Result<()> {
    let manifest: Value = serde_json::from_reader(
        File::open(manifest_path)
            .with_context(|| format!("open cached manifest {}", manifest_path.display()))?,
    )
    .with_context(|| format!("parse cached manifest {}", manifest_path.display()))?;
    let recorded = PathBuf::from(json_string(&manifest, "source.path")?);
    let requested = fs::canonicalize(source_path)
        .with_context(|| format!("canonicalize requested source {}", source_path.display()))?;
    let recorded = fs::canonicalize(&recorded)
        .with_context(|| format!("canonicalize cached source {}", recorded.display()))?;
    ensure!(
        recorded == requested,
        "cached conversion source {} does not match --input {}; pass --force-conversion",
        recorded.display(),
        requested.display()
    );
    Ok(())
}

fn conversion_artifacts(
    arguments: &Arguments,
    spec: ConversionSpec,
) -> Result<ConversionArtifacts> {
    let output_directory = match &arguments.conversion_output_dir {
        Some(path) if path.is_absolute() => path.clone(),
        Some(path) => arguments.model_data_dir.join(path),
        None => arguments
            .model_data_dir
            .join(arguments.preset.manifest_relative_path())
            .parent()
            .context("the preset manifest must have a parent directory")?
            .to_owned(),
    };
    let output_name = arguments
        .conversion_output_name
        .clone()
        .unwrap_or_else(|| spec.output_name.to_owned());
    let output_name_path = Path::new(&output_name);
    ensure!(
        !output_name.is_empty()
            && output_name_path
                .file_name()
                .is_some_and(|name| name == output_name_path),
        "--conversion-output-name must be one filename stem without path separators"
    );
    let manifest_path = output_directory.join(format!("{output_name}.manifest.json"));
    let bva_path = output_directory.join(format!("{output_name}.bva.{}", spec.storage_dtype));
    Ok(ConversionArtifacts {
        output_directory,
        output_name,
        manifest_path,
        bva_path,
    })
}

fn canonicalize_conversion_artifacts(
    mut artifacts: ConversionArtifacts,
) -> Result<ConversionArtifacts> {
    artifacts.output_directory =
        fs::canonicalize(&artifacts.output_directory).with_context(|| {
            format!(
                "canonicalize conversion directory {}",
                artifacts.output_directory.display()
            )
        })?;
    artifacts.manifest_path = fs::canonicalize(&artifacts.manifest_path).with_context(|| {
        format!(
            "canonicalize conversion manifest {}",
            artifacts.manifest_path.display()
        )
    })?;
    artifacts.bva_path = fs::canonicalize(&artifacts.bva_path).with_context(|| {
        format!(
            "canonicalize converted BVA {}",
            artifacts.bva_path.display()
        )
    })?;
    Ok(artifacts)
}

fn validate_bf16_bitcast_manifest(path: &Path) -> Result<()> {
    let manifest: Value = serde_json::from_reader(
        File::open(path).with_context(|| format!("open manifest {}", path.display()))?,
    )
    .with_context(|| format!("parse manifest {}", path.display()))?;

    ensure!(
        json_u64(&manifest, "version")? == 2,
        "BF16 bitcast manifests must use version 2"
    );
    ensure!(
        json_string(&manifest, "source.model_family")? == "llama-bf16",
        "source.model_family must be llama-bf16"
    );
    ensure!(
        json_string(&manifest, "selection.policy")? == "llama_bf16_parameter_weights",
        "selection.policy must select Llama BF16 parameter weights"
    );
    ensure!(
        json_string(&manifest, "selection.tensor_order")? == "lexicographic_name",
        "selection.tensor_order must be lexicographic_name"
    );
    ensure!(
        json_string(&manifest, "quantization.scheme")? == "bf16_bitcast_i16",
        "quantization.scheme must be bf16_bitcast_i16"
    );
    ensure!(
        json_u64(&manifest, "quantization.native_i8_tensor_count")? == 0,
        "the BF16 conversion must not contain native i8 tensors"
    );
    ensure!(
        json_u64(&manifest, "quantization.converted_float_tensor_count")? == 0,
        "the BF16 bitcast must not numerically convert floating tensors"
    );
    ensure!(
        json_string(&manifest, "quantization.storage_dtype")? == "i16",
        "quantization.storage_dtype must be i16"
    );
    ensure!(
        json_string(&manifest, "quantization.storage_encoding")?
            == "signed_twos_complement_little_endian",
        "quantization.storage_encoding must be signed little-endian two's complement"
    );
    ensure!(
        json_u64(&manifest, "quantization.storage_item_bytes")? == 2,
        "quantization.storage_item_bytes must be two"
    );
    ensure!(
        json_u64(&manifest, "quantization.bit_width")? == 16,
        "quantization.bit_width must be 16"
    );
    ensure!(
        json_bool(&manifest, "quantization.bit_preserving")?,
        "quantization.bit_preserving must be true"
    );
    ensure!(
        !json_bool(&manifest, "quantization.numeric_conversion")?,
        "quantization.numeric_conversion must be false"
    );
    ensure!(
        json_string(&manifest, "quantization.source_to_storage")? == "bitcast_bf16_u16_to_i16",
        "quantization.source_to_storage must specify the exact BF16 bitcast"
    );
    ensure!(
        json_string(&manifest, "quantization.source_storage_encoding")?
            == "bf16_bit_pattern_little_endian",
        "quantization.source_storage_encoding must preserve little-endian BF16 bits"
    );
    ensure!(
        json_string(&manifest, "quantization.field_interpretation")?
            == "signed_twos_complement_i16",
        "quantization.field_interpretation must match the native i16 proof path"
    );
    let quantization = json_at(&manifest, "quantization")?;
    for name in [
        "zero_point",
        "float_conversion_qmin",
        "float_conversion_qmax",
        "rounding",
        "float_conversion_formula",
    ] {
        ensure!(
            quantization.get(name).is_none(),
            "BF16 bitcast manifest must not contain quantization.{name}"
        );
    }

    let source_sha256 = json_string(&manifest, "source.sha256")?;
    let bva_sha256 = json_string(&manifest, "layout.bva_sha256")?;
    ensure_sha256("source.sha256", source_sha256)?;
    ensure_sha256("layout.bva_sha256", bva_sha256)?;

    let tensors = json_at(&manifest, "tensors")?
        .as_array()
        .context("manifest field tensors must be an array")?;
    ensure!(!tensors.is_empty(), "the BF16 manifest has no tensors");
    ensure!(
        json_u64(&manifest, "quantization.bitcast_bf16_tensor_count")? == tensors.len() as u64,
        "every BF16 tensor must be counted as a bitcast tensor"
    );
    for (index, tensor) in tensors.iter().enumerate() {
        ensure!(
            json_string(tensor, "source_dtype")? == "BF16",
            "tensor {index} source_dtype must be BF16"
        );
        ensure!(
            json_string(tensor, "conversion")? == "bitcast_bf16_to_i16",
            "tensor {index} conversion must be bitcast_bf16_to_i16"
        );
        ensure!(
            matches!(tensor.get("max_abs"), None | Some(Value::Null)),
            "tensor {index} must not contain max_abs"
        );
        ensure!(
            matches!(tensor.get("scale"), None | Some(Value::Null)),
            "tensor {index} must not contain scale"
        );
    }

    println!("bf16_bitcast_manifest_semantics_verified=true");
    println!("bf16_bitcast_tensor_count={}", tensors.len());
    println!("bf16_i16_representation=bit_preserving");
    println!("bf16_i16_numeric_conversion=false");
    Ok(())
}

fn json_at<'a>(value: &'a Value, path: &str) -> Result<&'a Value> {
    path.split('.').try_fold(value, |current, component| {
        current
            .get(component)
            .with_context(|| format!("manifest is missing field {path}"))
    })
}

fn json_string<'a>(value: &'a Value, path: &str) -> Result<&'a str> {
    json_at(value, path)?
        .as_str()
        .with_context(|| format!("manifest field {path} must be a string"))
}

fn json_u64(value: &Value, path: &str) -> Result<u64> {
    json_at(value, path)?
        .as_u64()
        .with_context(|| format!("manifest field {path} must be an unsigned integer"))
}

fn json_bool(value: &Value, path: &str) -> Result<bool> {
    json_at(value, path)?
        .as_bool()
        .with_context(|| format!("manifest field {path} must be boolean"))
}

fn ensure_sha256(name: &str, digest: &str) -> Result<()> {
    ensure!(
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)),
        "manifest field {name} must be a lowercase SHA-256 digest"
    );
    Ok(())
}

fn parse_arguments(arguments: impl IntoIterator<Item = String>) -> Result<Option<Arguments>> {
    let mut arguments = arguments.into_iter().collect::<Vec<_>>().into_iter();
    let Some(command) = arguments.next() else {
        print_help();
        bail!("missing Llama preset")
    };
    if command == "-h" || command == "--help" {
        print_help();
        return Ok(None);
    }
    let preset = ModelPreset::from_str(&command).with_context(|| {
        format!("unsupported preset {command}; use llama8b-i8, llama8b-i16, or llama70b-i8")
    })?;
    conversion_spec(preset)?;

    let crate_directory = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_directory = crate_directory
        .parent()
        .context("the batch crate must have a workspace parent")?;
    let mut model_data_dir = default_model_data_root();
    let mut input_override = None;
    let mut python = PathBuf::from("python3");
    let mut converter = workspace_directory.join("preprocessing/modelconvert.py");
    let mut conversion_output_dir = None;
    let mut conversion_output_name = None;
    let mut chunk_mib = 64_usize;
    let mut force_conversion = false;
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
            "--conversion-output-dir" => {
                conversion_output_dir = Some(PathBuf::from(value(
                    "--conversion-output-dir",
                    &mut arguments,
                )?))
            }
            "--conversion-output-name" => {
                conversion_output_name = Some(value("--conversion-output-name", &mut arguments)?)
            }
            "--chunk-mib" => {
                chunk_mib = value("--chunk-mib", &mut arguments)?.parse()?;
                ensure!(chunk_mib > 0, "--chunk-mib must be positive");
            }
            "--force-conversion" => force_conversion = true,
            "--target-log" | "--seed" => {
                let option_value = value(&argument, &mut arguments)?;
                experiment_arguments.push(argument);
                experiment_arguments.push(option_value);
            }
            "--skip-input-hash"
            | "--verify-input-hash"
            | "--allow-non-64-threads"
            | "--shape-validity-proof"
            | "--shape-validity"
            | "--skip-shape-validity-proof"
            | "--summary-only" => experiment_arguments.push(argument),
            "--bench" => {
                // Cargo may append this marker for a harness-free benchmark.
            }
            "-h" | "--help" => {
                print_help();
                return Ok(None);
            }
            _ => bail!("unknown argument {argument}; use --help for usage"),
        }
    }

    Ok(Some(Arguments {
        preset,
        model_data_dir,
        input_override,
        python,
        converter,
        conversion_output_dir,
        conversion_output_name,
        chunk_mib,
        force_conversion,
        experiment_arguments,
    }))
}

fn conversion_spec(preset: ModelPreset) -> Result<ConversionSpec> {
    match preset {
        ModelPreset::Llama8bI8 => Ok(ConversionSpec {
            source_directory: "Llama-3.1-8B-W8A8",
            output_name: "llama-3.1-8b-w8a8",
            model_family: "llama-w8a8",
            storage_dtype: "i8",
        }),
        ModelPreset::Llama70bI8 => Ok(ConversionSpec {
            source_directory: "Llama-3.1-70B-W8A8",
            output_name: "llama-3.1-70b-w8a8",
            model_family: "llama-w8a8",
            storage_dtype: "i8",
        }),
        ModelPreset::Llama8bI16 => Ok(ConversionSpec {
            source_directory: "Llama-3.1-8B-BF16",
            output_name: "llama-3.1-8b-bf16-bits",
            model_family: "llama-bf16",
            storage_dtype: "i16",
        }),
        ModelPreset::Llama70bI16 => Ok(ConversionSpec {
            source_directory: "Llama-3.1-70B",
            output_name: "llama-3.1-70b-bf16-bits",
            model_family: "llama-bf16",
            storage_dtype: "i16",
        }),
    }
}

fn report_stage(index: usize, name: &str) -> Result<()> {
    println!("[stage {index}/{LLAMA_STAGE_COUNT}] {name}");
    io::stdout().flush().context("flush Llama experiment stage")
}

fn print_help() {
    println!(
        "Usage:\n\
         RAYON_NUM_THREADS=64 cargo bench -p batch --bench experiment_llama -- \\\n\
           llama8b-i8|llama8b-i16|llama70b-i8 [OPTIONS]\n\
         \n\
         Detailed conversion, stage timing, and proof-size output is printed by default.\n\
         \n\
         Conversion options:\n\
           --model-data-dir PATH    Model-data root; defaults to workspace ../model-data\n\
           --input PATH             Override the source checkpoint directory\n\
           --python PATH            Python executable (default: python3)\n\
           --converter PATH         Converter path (default: preprocessing/modelconvert.py)\n\
           --conversion-output-dir PATH\n\
                                   Override the converted artifact directory\n\
           --conversion-output-name NAME\n\
                                   Override the converted artifact filename stem\n\
           --chunk-mib N            Python conversion chunk size (default: 64)\n\
           --force-conversion       Replace an existing canonical BVA conversion\n\
         \n\
         Experiment options:\n\
           --target-log N           Assert the exact logical BVA domain\n\
           --seed N                 Deterministic prover seed\n\
           --verify-input-hash      Verify the converted BVA SHA-256 (default)\n\
           --skip-input-hash        Skip only the converted BVA SHA-256 pass\n\
           --allow-non-64-threads   Permit a local smoke run\n\
           --shape-validity-proof   Explicitly enable shape validity proving (default)\n\
           --skip-shape-validity-proof\n\
                                   Disable shape validity proving for a performance-only run\n\
           --summary-only           Hide detailed metrics and retain Final results\n\
           -h, --help               Print this help"
    );
}
