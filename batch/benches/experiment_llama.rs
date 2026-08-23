use std::{
    io::{self, Write},
    path::{Path, PathBuf},
    process::Command,
    str::FromStr,
    time::Instant,
};

use anyhow::{bail, ensure, Context, Result};
use batch::model_bva::{default_model_data_root, ModelBvaManifest, ModelPreset};

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
}

fn main() -> Result<()> {
    let Some(arguments) = parse_arguments(std::env::args().skip(1))? else {
        return Ok(());
    };

    report_stage(1, "Prepare Llama BVA with Python")?;
    let conversion_started = Instant::now();
    let status = prepare_conversion(&arguments)?;
    let conversion_seconds = conversion_started.elapsed().as_secs_f64();
    let manifest_path = arguments
        .model_data_dir
        .join(arguments.preset.manifest_relative_path());
    let manifest = ModelBvaManifest::load(
        arguments.preset,
        &arguments.model_data_dir,
        Some(&manifest_path),
        None,
    )
    .with_context(|| format!("validate converted manifest {}", manifest_path.display()))?;
    ensure!(
        manifest.bit_width() == 8 && manifest.storage_dtype() == "i8",
        "the current ParComm experiment requires a native i8 BVA"
    );
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
    ];
    experiment_arguments.extend(arguments.experiment_arguments);
    experiment::run_from_args(experiment_arguments)
}

fn prepare_conversion(arguments: &Arguments) -> Result<ConversionStatus> {
    let spec = conversion_spec(arguments.preset)?;
    let source_path = arguments
        .input_override
        .clone()
        .unwrap_or_else(|| arguments.model_data_dir.join(spec.source_directory));
    let manifest_path = arguments
        .model_data_dir
        .join(arguments.preset.manifest_relative_path());
    let output_directory = manifest_path
        .parent()
        .context("the preset manifest must have an output directory")?;
    let bva_path = output_directory.join(format!("{}.bva.i8", spec.output_name));
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
        .arg("llama-w8a8")
        .arg("--storage-dtype")
        .arg("i8")
        .arg("--output-dir")
        .arg(output_directory)
        .arg("--name")
        .arg(spec.output_name)
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
        "Python conversion succeeded without producing the canonical manifest and BVA"
    );
    Ok(ConversionStatus::Converted)
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
    let preset = ModelPreset::from_str(&command)
        .with_context(|| format!("unsupported preset {command}; use llama8b-i8 or llama70b-i8"))?;
    conversion_spec(preset)?;

    let crate_directory = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace_directory = crate_directory
        .parent()
        .context("the batch crate must have a workspace parent")?;
    let mut model_data_dir = default_model_data_root();
    let mut input_override = None;
    let mut python = PathBuf::from("python3");
    let mut converter = workspace_directory.join("preprocessing/modelconvert.py");
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
        }),
        ModelPreset::Llama70bI8 => Ok(ConversionSpec {
            source_directory: "Llama-3.1-70B-W8A8",
            output_name: "llama-3.1-70b-w8a8",
        }),
        ModelPreset::Llama8bI16 => {
            bail!("llama8b-i16 is not supported by the current native-i8 proof path")
        }
    }
}

fn report_stage(index: usize, name: &str) -> Result<()> {
    println!("[llama stage {index}/{LLAMA_STAGE_COUNT}] {name}");
    io::stdout().flush().context("flush Llama experiment stage")
}

fn print_help() {
    println!(
        "Usage:\n\
         RAYON_NUM_THREADS=64 cargo bench -p batch --bench experiment_llama -- \\\n\
           llama8b-i8|llama70b-i8 [OPTIONS]\n\
         \n\
         Conversion options:\n\
           --model-data-dir PATH    Model-data root; defaults to workspace ../model-data\n\
           --input PATH             Override the source checkpoint directory\n\
           --python PATH            Python executable (default: python3)\n\
           --converter PATH         Converter path (default: preprocessing/modelconvert.py)\n\
           --chunk-mib N            Python conversion chunk size (default: 64)\n\
           --force-conversion       Replace an existing canonical BVA conversion\n\
         \n\
         Experiment options:\n\
           --target-log N           Assert the exact logical BVA domain\n\
           --seed N                 Deterministic prover seed\n\
           --verify-input-hash      Verify the converted BVA SHA-256 (default)\n\
           --skip-input-hash        Skip only the converted BVA SHA-256 pass\n\
           --allow-non-64-threads   Permit a local smoke run\n\
           --shape-validity-proof   Include the optional shape validity proof\n\
           --summary-only           Print stages and final tables without detailed metrics\n\
           -h, --help               Print this help"
    );
}
