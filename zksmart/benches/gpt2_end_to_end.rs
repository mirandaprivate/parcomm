use std::env;
use std::error::Error;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, Cursor, Read, Seek, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use sha2::{Digest, Sha256};
use zksmart::hdr1cs::gpt2::{
    Gpt2GlobalRelationParameters, Gpt2Model, Gpt2PublicStatement, Gpt2Q12Checkpoint,
    Gpt2ScalableProof,
};

#[global_allocator]
static ALLOCATOR: jemallocator::Jemalloc = jemallocator::Jemalloc;

const STATEMENT_FILE: &str = "gpt2.statement";
const PARAMETERS_FILE: &str = "gpt2.params";
const PROOF_FILE: &str = "gpt2.proof";
const MANIFEST_FILE: &str = "manifest.txt";

enum Command {
    Prove {
        checkpoint: Option<PathBuf>,
        output_dir: PathBuf,
        factor_chunk_len: usize,
    },
    Verify {
        artifact_dir: PathBuf,
    },
}

fn main() {
    if let Err(error) = run() {
        eprintln!("gpt2_end_to_end_error={error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    match parse_command()? {
        Command::Prove {
            checkpoint,
            output_dir,
            factor_chunk_len,
        } => {
            if checkpoint.is_some() {
                println!("fixture=checkpoint");
            } else {
                println!("fixture=tiny");
            }
            println!("proof_mode=scalable_public_support_non_zk");
            println!("factor_chunk_len={factor_chunk_len}");
            println!("artifact_dir={}", output_dir.display());
            if let Some(path) = checkpoint {
                println!("checkpoint={}", path.display());
                let global = open_checkpoint(&path)?;
                let lookup = open_checkpoint(&path)?;
                prove_artifacts(global, lookup, &output_dir, factor_chunk_len)?;
            } else {
                let bytes = tiny_checkpoint_bytes();
                let global = Gpt2Q12Checkpoint::from_reader(Cursor::new(bytes.clone()))?;
                let lookup = Gpt2Q12Checkpoint::from_reader(Cursor::new(bytes))?;
                prove_artifacts(global, lookup, &output_dir, factor_chunk_len)?;
            }
        }
        Command::Verify { artifact_dir } => {
            verify_artifacts(&artifact_dir)?;
        }
    }
    Ok(())
}

fn open_checkpoint(path: &Path) -> Result<Gpt2Q12Checkpoint<BufReader<File>>, Box<dyn Error>> {
    let file = File::open(path)?;
    Ok(Gpt2Q12Checkpoint::from_reader(BufReader::with_capacity(
        8 << 20,
        file,
    ))?)
}

fn prove_artifacts<GlobalR, LookupR>(
    mut global: Gpt2Q12Checkpoint<GlobalR>,
    mut lookup: Gpt2Q12Checkpoint<LookupR>,
    output_dir: &Path,
    factor_chunk_len: usize,
) -> Result<(), Box<dyn Error>>
where
    GlobalR: Read + Seek,
    LookupR: Read + Seek,
{
    if factor_chunk_len == 0 {
        return Err("factor chunk length must be nonzero".into());
    }
    fs::create_dir_all(output_dir)?;
    let statement = Gpt2PublicStatement::new(
        global.metadata(),
        global.token_ids(),
        global.reference_logits(),
    )?;
    let setup_started = Instant::now();
    let parameters = Gpt2GlobalRelationParameters::setup(&statement)?;
    let setup_seconds = setup_started.elapsed().as_secs_f64();
    let prove_started = Instant::now();
    let proof = Gpt2ScalableProof::prove_with_replays(
        &parameters,
        &statement,
        &mut global,
        &mut lookup,
        factor_chunk_len,
    )?;
    let prove_seconds = prove_started.elapsed().as_secs_f64();

    let statement_bytes = statement.to_bytes_exact()?;
    let parameters_bytes = parameters.to_bytes_exact()?;
    let proof_bytes = proof.to_bytes_exact()?;
    let witness_commitment_count = proof.witness_commitment_count();
    let witness_opening_count = proof.witness_opening_count();
    drop(proof);

    // Persist the exact prover outputs before the potentially memory-intensive
    // self-verification pass. The manifest remains the success marker and is
    // written only after verification accepts all three exact byte strings.
    write_new(&output_dir.join(STATEMENT_FILE), &statement_bytes)?;
    write_new(&output_dir.join(PARAMETERS_FILE), &parameters_bytes)?;
    write_new(&output_dir.join(PROOF_FILE), &proof_bytes)?;
    let verify_started = Instant::now();
    let verified = Gpt2ScalableProof::verify_bytes(&parameters, &statement, &proof_bytes)?;
    let verify_seconds = verify_started.elapsed().as_secs_f64();
    if !verified {
        return Err("freshly generated proof did not verify".into());
    }

    let manifest = format!(
        concat!(
            "format=zksmart.gpt2.end-to-end-artifacts.v3\n",
            "proof_mode=scalable_public_support_non_zk\n",
            "statement_sha256={}\n",
            "parameters_sha256={}\n",
            "proof_sha256={}\n",
            "statement_bytes={}\n",
            "parameters_bytes={}\n",
            "proof_bytes={}\n",
            "statement_digest={}\n",
            "global_witness_commitment_count={}\n",
            "global_witness_opening_count={}\n",
            "materialized_padding_entries=0\n",
            "fresh_proof_verified=true\n"
        ),
        sha256_hex(&statement_bytes),
        sha256_hex(&parameters_bytes),
        sha256_hex(&proof_bytes),
        statement_bytes.len(),
        parameters_bytes.len(),
        proof_bytes.len(),
        hex(&statement.transcript_digest()),
        witness_commitment_count,
        witness_opening_count,
    );
    write_new(&output_dir.join(MANIFEST_FILE), manifest.as_bytes())?;

    println!("setup_seconds={setup_seconds:.6}");
    println!("prove_seconds={prove_seconds:.6}");
    println!("verify_seconds={verify_seconds:.6}");
    println!("statement_bytes={}", statement_bytes.len());
    println!("parameters_bytes={}", parameters_bytes.len());
    println!("proof_bytes={}", proof_bytes.len());
    println!("statement_sha256={}", sha256_hex(&statement_bytes));
    println!("parameters_sha256={}", sha256_hex(&parameters_bytes));
    println!("proof_sha256={}", sha256_hex(&proof_bytes));
    println!("global_witness_commitment_count=1");
    println!("global_witness_opening_count=1");
    println!("materialized_padding_entries=0");
    println!("fresh_proof_verified=true");
    Ok(())
}

fn verify_artifacts(artifact_dir: &Path) -> Result<(), Box<dyn Error>> {
    println!("proof_mode=scalable_public_support_non_zk");
    println!("verifier_input=statement_params_proof_only");
    println!("artifact_dir={}", artifact_dir.display());
    let statement_bytes = fs::read(artifact_dir.join(STATEMENT_FILE))?;
    let parameters_bytes = fs::read(artifact_dir.join(PARAMETERS_FILE))?;
    let proof_bytes = fs::read(artifact_dir.join(PROOF_FILE))?;
    let statement = Gpt2PublicStatement::from_bytes_exact(&statement_bytes)?;
    let parameters = Gpt2GlobalRelationParameters::from_bytes_exact(&parameters_bytes)?;
    let started = Instant::now();
    let verified = Gpt2ScalableProof::verify_bytes(&parameters, &statement, &proof_bytes)?;
    println!("verify_seconds={:.6}", started.elapsed().as_secs_f64());
    println!("statement_sha256={}", sha256_hex(&statement_bytes));
    println!("parameters_sha256={}", sha256_hex(&parameters_bytes));
    println!("proof_sha256={}", sha256_hex(&proof_bytes));
    println!("statement_only_verify={verified}");
    if !verified {
        return Err("proof verification failed".into());
    }
    Ok(())
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn parse_command() -> Result<Command, Box<dyn Error>> {
    let arguments = env::args().skip(1).collect::<Vec<_>>();
    let mode = option_value(&arguments, "--mode").unwrap_or_else(|| "prove".to_owned());
    match mode.as_str() {
        "prove" => {
            let output_dir = option_value(&arguments, "--output-dir")
                .ok_or("prove requires --output-dir PATH")?;
            let checkpoint = option_value(&arguments, "--checkpoint").map(PathBuf::from);
            let tiny = arguments.iter().any(|value| value == "--tiny");
            if checkpoint.is_some() == tiny {
                return Err("prove requires exactly one of --checkpoint PATH or --tiny".into());
            }
            let factor_chunk_len = option_value(&arguments, "--factor-chunk-len")
                .unwrap_or_else(|| "64".to_owned())
                .parse()?;
            Ok(Command::Prove {
                checkpoint,
                output_dir: PathBuf::from(output_dir),
                factor_chunk_len,
            })
        }
        "verify" => {
            let artifact_dir = option_value(&arguments, "--artifact-dir")
                .ok_or("verify requires --artifact-dir PATH")?;
            Ok(Command::Verify {
                artifact_dir: PathBuf::from(artifact_dir),
            })
        }
        _ => Err("--mode must be prove or verify".into()),
    }
}

fn option_value(arguments: &[String], option: &str) -> Option<String> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == option)
        .map(|pair| pair[1].clone())
}

fn sha256_hex(bytes: &[u8]) -> String {
    hex(&Sha256::digest(bytes))
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

fn tiny_checkpoint_bytes() -> Vec<u8> {
    let model = Gpt2Model {
        blocks: 1,
        hidden_width: 2,
        attention_heads: 1,
        mlp_width: 4,
        vocabulary: 3,
        maximum_positions: 2,
    };
    let parameter_entries = model
        .physical_parameter_entries()
        .expect("tiny parameter inventory");
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"PR1GPT2\0");
    bytes.extend_from_slice(&2_u32.to_le_bytes());
    bytes.extend_from_slice(&12_u32.to_le_bytes());
    bytes.extend_from_slice(&(parameter_entries as u64).to_le_bytes());
    for value in [
        2_u32,
        model.vocabulary as u32,
        model.maximum_positions as u32,
        model.hidden_width as u32,
        model.attention_heads as u32,
        model.mlp_width as u32,
        model.blocks as u32,
    ] {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes.extend_from_slice(&[0x24; 32]);
    bytes.extend_from_slice(&[0x84; 32]);
    for token in [0_u32, 1] {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    for logit in [-4_094_i64, 4_094, 0] {
        bytes.extend_from_slice(&logit.to_le_bytes());
    }
    let mut parameters = vec![0_i32; parameter_entries];
    parameters[..6].copy_from_slice(&[4_096, 0, -4_096, 0, 4_096, -4_096]);
    parameters[64..66].copy_from_slice(&[4_096, 4_096]);
    for value in parameters {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}
