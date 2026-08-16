use std::convert::TryFrom;
use std::fs::{self, OpenOptions};
use std::io::{Cursor, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use ark_bls12_381::Bls12_381;
use sha2::{Digest, Sha256};
use zksmart::hdr1cs::fair_mlp::{hex_digest, FairMlpSpec, FAIR_MLP_FORMAT};
use zksmart::hdr1cs::global_mlp::artifact::GlobalMlpVerifierArtifact;
use zksmart::hdr1cs::global_mlp::GlobalHdR1csMlpProtocol;
use zksmart::hdr1cs::mlp::MlpManifest;

#[global_allocator]
static ALLOCATOR: jemallocator::Jemalloc = jemallocator::Jemalloc;

const STATEMENT_FILE: &str = "statement.bin";
const PARAMETERS_FILE: &str = "verifier-parameters.bin";
const PROOF_FILE: &str = "proof.bin";
const MANIFEST_FILE: &str = "manifest.txt";
const STATEMENT_MAGIC: &[u8; 8] = b"ZSMST001";

#[derive(Clone, Debug)]
struct Config {
    spec: FairMlpSpec,
    prove: bool,
    artifact_dir: Option<PathBuf>,
    tamper_tests: bool,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("fair_mlp_error={error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args().skip(1).collect::<Vec<_>>();
    if let Some(directory) = option_value(&arguments, "--verify-artifact") {
        return verify_artifact(Path::new(&directory));
    }
    let config = parse_args(&arguments)?;
    config.spec.validate()?;
    println!("experiment=fair_1024_relu_mlp");
    println!("fixture_format={FAIR_MLP_FORMAT}");
    println!("equation=x_next=ReLU(W*x+b)");
    println!("activation=exact_relu");
    println!("division_or_rescale=none");
    println!("curve=BLS12-381");
    println!("layers={}", config.spec.layers);
    println!("width={}", config.spec.width);
    println!(
        "weight_parameters={}",
        config.spec.weight_parameters().unwrap()
    );
    println!("bias_parameters={}", config.spec.bias_parameters().unwrap());
    println!(
        "model_parameters={}",
        config.spec.model_parameters().unwrap()
    );

    let digest_started = Instant::now();
    let fixture_digest = config.spec.fixture_digest()?;
    println!("fixture_sha256={}", hex_digest(&fixture_digest));
    println!(
        "fixture_digest_seconds={:.6}",
        digest_started.elapsed().as_secs_f64()
    );

    let reference_started = Instant::now();
    let reference_output = config.spec.reference_inference()?;
    if reference_output != config.spec.expected_output() {
        return Err("reference output differs from the frozen expected output".into());
    }
    println!(
        "reference_seconds={:.6}",
        reference_started.elapsed().as_secs_f64()
    );
    println!(
        "reference_output_checksum={}",
        reference_output
            .iter()
            .map(|value| i64::from(*value))
            .sum::<i64>()
    );
    println!("reference_match=true");

    if !config.prove {
        println!("mode=reference_only");
        return Ok(());
    }

    let manifest = MlpManifest {
        widths: vec![config.spec.width; config.spec.layers + 1],
        commitment_bit_width: 1,
        relu_magnitude_bits: 1,
    };
    let statement_bytes = fair_statement_bytes(
        config.spec,
        &fixture_digest,
        &config.spec.public_input(),
        &reference_output,
    )?;
    let encode_started = Instant::now();
    let encoding = GlobalHdR1csMlpProtocol::<Bls12_381>::encode(
        manifest,
        config.spec.public_input(),
        reference_output,
        config.spec.layers(),
    )?;
    println!(
        "encode_seconds={:.6}",
        encode_started.elapsed().as_secs_f64()
    );
    println!(
        "physically_stored_witness_entries={}",
        encoding.stored_witness_entries()
    );
    let shape = encoding.global_shape();
    println!("global_block_domain_n={}", shape.n);
    println!("global_witness_slots={}", shape.witness_slots);
    println!("global_gate_slots={}", shape.gate_slots);

    let setup_started = Instant::now();
    let srs = GlobalHdR1csMlpProtocol::<Bls12_381>::setup(&encoding)?;
    println!("setup_seconds={:.6}", setup_started.elapsed().as_secs_f64());

    let prove_started = Instant::now();
    let (instance, proof) = GlobalHdR1csMlpProtocol::<Bls12_381>::prove(&srs, encoding)?;
    println!("prove_seconds={:.6}", prove_started.elapsed().as_secs_f64());
    let protocol_proof_size = proof.serialized_size_bytes();
    let proof_bytes = proof.to_bytes_exact()?;
    println!("protocol_proof_size_bytes={protocol_proof_size}");
    println!("proof_size_bytes={}", proof_bytes.len());
    println!(
        "witness_commitment_count={}",
        proof.witness_commitment_count()
    );
    println!("witness_opening_count={}", proof.witness_opening_count());
    println!(
        "operation_equation_count={}",
        proof.operation_equation_count()
    );

    let verify_started = Instant::now();
    let verified = GlobalHdR1csMlpProtocol::<Bls12_381>::verify(&srs, &instance, &proof);
    println!(
        "verify_seconds={:.6}",
        verify_started.elapsed().as_secs_f64()
    );
    println!("fresh_proof_verified={verified}");
    if !verified {
        return Err("fresh proof failed verification".into());
    }
    let verifier_artifact = GlobalMlpVerifierArtifact { srs, instance };
    let parameters_bytes = verifier_artifact.to_bytes_exact()?;
    println!("statement_size_bytes={}", statement_bytes.len());
    println!("verifier_parameters_size_bytes={}", parameters_bytes.len());
    println!("statement_sha256={}", sha256_hex(&statement_bytes));
    println!(
        "verifier_parameters_sha256={}",
        sha256_hex(&parameters_bytes)
    );
    println!("proof_sha256={}", sha256_hex(&proof_bytes));
    if config.tamper_tests {
        run_tamper_tests(&statement_bytes, &parameters_bytes, &proof_bytes)?;
    }
    if let Some(directory) = config.artifact_dir {
        write_artifact(
            &directory,
            &statement_bytes,
            &parameters_bytes,
            &proof_bytes,
        )?;
        println!("artifact_dir={}", directory.display());
    }
    Ok(())
}

fn parse_args(arguments: &[String]) -> Result<Config, Box<dyn std::error::Error>> {
    let layers = option_value(&arguments, "--layers")
        .unwrap_or_else(|| "1024".to_owned())
        .parse::<usize>()?;
    let width = option_value(&arguments, "--width")
        .unwrap_or_else(|| "1024".to_owned())
        .parse::<usize>()?;
    let prove = arguments.iter().any(|argument| argument == "--prove");
    let artifact_dir = option_value(arguments, "--artifact-dir").map(PathBuf::from);
    let tamper_tests = arguments
        .iter()
        .any(|argument| argument == "--tamper-tests");
    for (index, argument) in arguments.iter().enumerate() {
        if matches!(argument.as_str(), "--layers" | "--width" | "--artifact-dir") {
            if arguments.get(index + 1).is_none() {
                return Err(format!("{argument} requires a value").into());
            }
        } else if matches!(argument.as_str(), "--prove" | "--bench" | "--tamper-tests")
            || arguments
                .get(index.wrapping_sub(1))
                .map(|previous| {
                    matches!(previous.as_str(), "--layers" | "--width" | "--artifact-dir")
                })
                .unwrap_or(false)
        {
        } else {
            return Err(format!("unknown argument: {argument}").into());
        }
    }
    Ok(Config {
        spec: FairMlpSpec { layers, width },
        prove,
        artifact_dir,
        tamper_tests,
    })
}

fn option_value(arguments: &[String], option: &str) -> Option<String> {
    arguments
        .windows(2)
        .find(|pair| pair[0] == option)
        .map(|pair| pair[1].clone())
}

fn fair_statement_bytes(
    spec: FairMlpSpec,
    fixture_digest: &[u8; 32],
    public_input: &[i32],
    public_output: &[i32],
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(STATEMENT_MAGIC);
    write_usize(spec.layers, &mut bytes)?;
    write_usize(spec.width, &mut bytes)?;
    bytes.extend_from_slice(fixture_digest);
    write_i32_vec(public_input, &mut bytes)?;
    write_i32_vec(public_output, &mut bytes)?;
    Ok(bytes)
}

fn parse_fair_statement(
    bytes: &[u8],
) -> Result<(FairMlpSpec, [u8; 32], Vec<i32>, Vec<i32>), Box<dyn std::error::Error>> {
    let mut reader = Cursor::new(bytes);
    let mut magic = [0_u8; 8];
    reader.read_exact(&mut magic)?;
    if &magic != STATEMENT_MAGIC {
        return Err("statement magic mismatch".into());
    }
    let spec = FairMlpSpec {
        layers: read_usize(&mut reader)?,
        width: read_usize(&mut reader)?,
    };
    spec.validate()?;
    let mut digest = [0_u8; 32];
    reader.read_exact(&mut digest)?;
    let public_input = read_i32_vec(&mut reader, 1_000_000)?;
    let public_output = read_i32_vec(&mut reader, 1_000_000)?;
    if reader.position() != bytes.len() as u64 {
        return Err("statement contains trailing bytes".into());
    }
    Ok((spec, digest, public_input, public_output))
}

fn verify_artifact(directory: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let statement_bytes = read_limited(&directory.join(STATEMENT_FILE), 16 * 1024 * 1024)?;
    let parameters_bytes =
        read_limited(&directory.join(PARAMETERS_FILE), 4_u64 * 1024 * 1024 * 1024)?;
    let proof_bytes = read_limited(&directory.join(PROOF_FILE), 1024 * 1024 * 1024)?;
    let started = Instant::now();
    verify_serialized(&statement_bytes, &parameters_bytes, &proof_bytes)?;
    println!("mode=independent_statement_only_verifier");
    println!("verifier_input=statement_parameters_proof_only");
    println!("statement_size_bytes={}", statement_bytes.len());
    println!("verifier_parameters_size_bytes={}", parameters_bytes.len());
    println!("proof_size_bytes={}", proof_bytes.len());
    println!("statement_sha256={}", sha256_hex(&statement_bytes));
    println!(
        "verifier_parameters_sha256={}",
        sha256_hex(&parameters_bytes)
    );
    println!("proof_sha256={}", sha256_hex(&proof_bytes));
    println!("verify_seconds={:.6}", started.elapsed().as_secs_f64());
    println!("statement_only_verify=true");
    Ok(())
}

fn verify_serialized(
    statement_bytes: &[u8],
    parameters_bytes: &[u8],
    proof_bytes: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    let (spec, fixture_digest, public_input, public_output) =
        parse_fair_statement(statement_bytes)?;
    if fixture_digest != spec.fixture_digest()? {
        return Err("statement fixture digest mismatch".into());
    }
    if public_input != spec.public_input() || public_output != spec.expected_output() {
        return Err("statement public I/O differs from frozen fixture".into());
    }
    let artifact = GlobalMlpVerifierArtifact::<Bls12_381>::from_bytes_exact(parameters_bytes)?;
    let expected_manifest = MlpManifest {
        widths: vec![spec.width; spec.layers + 1],
        commitment_bit_width: 1,
        relu_magnitude_bits: 1,
    };
    if artifact.instance.manifest != expected_manifest
        || artifact.instance.public_input != public_input
        || artifact.instance.public_output != public_output
    {
        return Err("verifier instance differs from public statement".into());
    }
    let proof =
        zksmart::hdr1cs::global_mlp::GlobalMlpProof::<Bls12_381>::from_bytes_exact(proof_bytes)?;
    if !GlobalHdR1csMlpProtocol::<Bls12_381>::verify(&artifact.srs, &artifact.instance, &proof) {
        return Err("serialized proof failed verification".into());
    }
    Ok(())
}

fn run_tamper_tests(
    statement_bytes: &[u8],
    parameters_bytes: &[u8],
    proof_bytes: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    if zksmart::hdr1cs::global_mlp::GlobalMlpProof::<Bls12_381>::from_bytes_exact(
        &proof_bytes[..proof_bytes.len() - 1],
    )
    .is_ok()
    {
        return Err("truncated proof was accepted".into());
    }
    println!("truncated_proof_rejected=true");

    let mut trailing = proof_bytes.to_vec();
    trailing.push(0);
    if zksmart::hdr1cs::global_mlp::GlobalMlpProof::<Bls12_381>::from_bytes_exact(&trailing).is_ok()
    {
        return Err("proof with trailing bytes was accepted".into());
    }
    println!("trailing_proof_rejected=true");

    let mut changed_proof = proof_bytes.to_vec();
    changed_proof[0] ^= 1;
    if verify_serialized(statement_bytes, parameters_bytes, &changed_proof).is_ok() {
        return Err("tampered proof was accepted".into());
    }
    println!("proof_tamper_rejected=true");

    let mut changed_statement = statement_bytes.to_vec();
    let last = changed_statement.len() - 1;
    changed_statement[last] ^= 1;
    if verify_serialized(&changed_statement, parameters_bytes, proof_bytes).is_ok() {
        return Err("tampered public output was accepted".into());
    }
    println!("public_io_tamper_rejected=true");

    let mut changed_parameters = parameters_bytes.to_vec();
    changed_parameters[0] ^= 1;
    if verify_serialized(statement_bytes, &changed_parameters, proof_bytes).is_ok() {
        return Err("tampered verifier parameters were accepted".into());
    }
    println!("verifier_parameters_tamper_rejected=true");
    Ok(())
}

fn write_artifact(
    directory: &Path,
    statement_bytes: &[u8],
    parameters_bytes: &[u8],
    proof_bytes: &[u8],
) -> Result<(), Box<dyn std::error::Error>> {
    fs::create_dir_all(directory)?;
    write_new(&directory.join(STATEMENT_FILE), statement_bytes)?;
    write_new(&directory.join(PARAMETERS_FILE), parameters_bytes)?;
    write_new(&directory.join(PROOF_FILE), proof_bytes)?;
    let manifest = format!(
        concat!(
            "format=zksmart-fair-MLP-artifact-v2\n",
            "statement_size_bytes={}\n",
            "statement_sha256={}\n",
            "verifier_parameters_size_bytes={}\n",
            "verifier_parameters_sha256={}\n",
            "proof_size_bytes={}\n",
            "proof_sha256={}\n",
            "fresh_proof_verified=true\n"
        ),
        statement_bytes.len(),
        sha256_hex(statement_bytes),
        parameters_bytes.len(),
        sha256_hex(parameters_bytes),
        proof_bytes.len(),
        sha256_hex(proof_bytes),
    );
    write_new(&directory.join(MANIFEST_FILE), manifest.as_bytes())?;
    Ok(())
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn read_limited(path: &Path, maximum: u64) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let metadata = fs::metadata(path)?;
    if metadata.len() > maximum {
        return Err(format!("{} exceeds artifact size bound", path.display()).into());
    }
    Ok(fs::read(path)?)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    hex_digest(&digest)
}

fn write_usize(value: usize, writer: &mut Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
    writer.extend_from_slice(&u64::try_from(value)?.to_le_bytes());
    Ok(())
}

fn read_usize(reader: &mut Cursor<&[u8]>) -> Result<usize, Box<dyn std::error::Error>> {
    let mut bytes = [0_u8; 8];
    reader.read_exact(&mut bytes)?;
    Ok(usize::try_from(u64::from_le_bytes(bytes))?)
}

fn write_i32_vec(values: &[i32], writer: &mut Vec<u8>) -> Result<(), Box<dyn std::error::Error>> {
    write_usize(values.len(), writer)?;
    for value in values {
        writer.extend_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

fn read_i32_vec(
    reader: &mut Cursor<&[u8]>,
    maximum: usize,
) -> Result<Vec<i32>, Box<dyn std::error::Error>> {
    let count = read_usize(reader)?;
    if count > maximum {
        return Err("statement vector exceeds bound".into());
    }
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        let mut bytes = [0_u8; 4];
        reader.read_exact(&mut bytes)?;
        values.push(i32::from_le_bytes(bytes));
    }
    Ok(values)
}
