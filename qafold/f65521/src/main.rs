use parcomm_qafold::{
    layout::Shape,
    tbv,
    transcript::{Transcript, digest},
    *,
};
use serde_json::json;
use std::{
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
    time::Instant,
};

fn main() {
    if let Err(error) = run() {
        eprintln!("parcomm-qafold: {error}");
        std::process::exit(1);
    }
}
fn option(args: &[String], name: &str) -> Result<Option<String>> {
    match args.iter().position(|arg| arg == name) {
        Some(index) => Ok(Some(
            args.get(index + 1).ok_or("missing option value")?.clone(),
        )),
        None => Ok(None),
    }
}
fn required(args: &[String], name: &str) -> Result<String> {
    option(args, name)?.ok_or_else(|| format!("missing {name}").into())
}
fn number(args: &[String], name: &str, default: usize) -> Result<usize> {
    Ok(match option(args, name)? {
        Some(value) => value.parse()?,
        None => default,
    })
}
fn write_json(path: impl AsRef<Path>, value: &impl serde::Serialize) -> Result<()> {
    fs::write(path, serde_json::to_vec_pretty(value)?)?;
    Ok(())
}
fn run() -> Result<()> {
    let args = std::env::args().collect::<Vec<_>>();
    let command = args.get(1).map(String::as_str).unwrap_or("help");
    if command == "help" || command == "--help" {
        println!(
            "parcomm-qafold demo --output DIR [--threads N] [--chunk-log 10..30] [--packed-log 1..6]\nparcomm-qafold prove --description JSON --input signed-i8-bva-or-- --output DIR [--threads N] [--chunk-log N] [--packed-log N] [--cache-mode cached|recompute]\nparcomm-qafold verify --output DIR [--threads 1]\nPublic shape/point/value protocol; F65521/F65521^8; signed-i8 only. Output must be a new directory for proving."
        );
        return Ok(());
    }
    if !["demo", "prove", "verify"].contains(&command) {
        return Err("unknown command".into());
    }
    let allowed = [
        "--output",
        "--threads",
        "--chunk-log",
        "--packed-log",
        "--description",
        "--input",
        "--cache-mode",
    ];
    if (args.len() - 2) % 2 != 0 {
        return Err("options must be --name value pairs".into());
    }
    for pair in args[2..].chunks_exact(2) {
        if !allowed.contains(&pair[0].as_str()) {
            return Err(format!("unknown option {}", pair[0]).into());
        }
    }
    let threads = number(&args, "--threads", if command == "verify" { 1 } else { 8 })?;
    if threads == 0 {
        return Err("threads must be positive".into());
    }
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    let output = PathBuf::from(required(&args, "--output")?);
    if command == "verify" {
        // Clear any prior success before parsing or checking a different bundle.
        write_json(
            output.join("verification.json"),
            &json!({"verified":false,"status":"verification incomplete or rejected"}),
        )?;
        let statement: Commitment =
            serde_json::from_slice(&fs::read(output.join("commitment.json"))?)?;
        let claims: Vec<tbv::Claim> =
            serde_json::from_slice(&fs::read(output.join("claims.json"))?)?;
        let start = Instant::now();
        let parameters = Parameters::new(&statement.shapes, statement.config.clone())?;
        let setup_seconds = start.elapsed().as_secs_f64();
        let start = Instant::now();
        let proof = Proof::from_bytes(&fs::read(output.join("proof.bin"))?)?;
        let profile = parameters.verify(&statement, &claims, &proof)?;
        let record = json!({"verified":true,"scope":"public-metadata ParComm complete flow","threads":threads,"setup_seconds":setup_seconds,"verification_with_read_decode_seconds":start.elapsed().as_secs_f64(),"profile":profile});
        write_json(output.join("verification.json"), &record)?;
        println!("{record}");
        return Ok(());
    }
    if output.exists() {
        return Err("output directory exists; choose a new run directory".into());
    }
    let config = Config {
        chunk_log: number(&args, "--chunk-log", 30)?,
        packed_log: number(&args, "--packed-log", 4)?,
        query_rf_bits: 100,
    };
    let shapes = if command == "demo" {
        vec![
            Shape { rows: 31, cols: 37 },
            Shape { rows: 65, cols: 17 },
            Shape { rows: 1, cols: 7 },
            Shape { rows: 9, cols: 1 },
        ]
    } else {
        let description: serde_json::Value =
            serde_json::from_slice(&fs::read(required(&args, "--description")?)?)?;
        if description["format"] != "parcomm-dense-bva-stream-description"
            || description["storage_dtype"] != "i8"
        {
            return Err("requires signed-i8 stream description".into());
        }
        let pairs: Vec<[usize; 2]> = serde_json::from_value(description["shapes"].clone())?;
        pairs
            .into_iter()
            .map(|[rows, cols]| Shape { rows, cols })
            .collect()
    };
    let start = Instant::now();
    let parameters = Parameters::new(&shapes, config)?;
    let setup_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let values = if command == "demo" {
        parameters.layout.encode(
            &shapes
                .iter()
                .enumerate()
                .map(|(index, shape)| {
                    (0..shape.rows * shape.cols)
                        .map(|entry| (entry * 73 + index * 19) as u8 as i8)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>(),
        )?
    } else {
        let description: serde_json::Value =
            serde_json::from_slice(&fs::read(required(&args, "--description")?)?)?;
        if description["stream_bytes"].as_u64() != Some(parameters.layout.active as u64)
            || description["layout"]["logical_entries"].as_u64()
                != Some(parameters.layout.active.next_power_of_two() as u64)
        {
            return Err("converter/ParComm layout mismatch".into());
        }
        let input = required(&args, "--input")?;
        let mut reader: Box<dyn Read> = if input == "-" {
            Box::new(io::stdin().lock())
        } else {
            Box::new(fs::File::open(input)?)
        };
        let mut values = vec![0i8; parameters.layout.active];
        let mut buffer = vec![0u8; 8 * 1024 * 1024];
        for chunk in values.chunks_mut(buffer.len()) {
            reader.read_exact(&mut buffer[..chunk.len()])?;
            for (value, &byte) in chunk.iter_mut().zip(&buffer) {
                *value = byte as i8;
            }
        }
        let mut extra = [0u8; 1];
        if reader.read(&mut extra)? != 0 {
            return Err("excess BVA input".into());
        }
        values
    };
    let input_seconds = start.elapsed().as_secs_f64();
    fs::create_dir_all(&output)?;
    let start = Instant::now();
    let cache_mode = option(&args, "--cache-mode")?.unwrap_or_else(|| "cached".into());
    let committed = match cache_mode.as_str() {
        "cached" => parameters.commit(values)?,
        "recompute" => parameters.commit_recomputed(values)?,
        _ => return Err("--cache-mode must be cached or recompute".into()),
    };
    let commit_seconds = start.elapsed().as_secs_f64();
    println!(
        "{}",
        json!({"stage":"committed","seconds":commit_seconds,"buckets":parameters.layout.buckets.len(),"pcs_blocks":committed.statement.roots.len()})
    );
    let statement = committed.statement.clone();
    let start = Instant::now();
    let mut transcript = Transcript::new();
    transcript.object(b"benchmark-points", &statement)?;
    let points = shapes
        .iter()
        .map(|shape| {
            (
                (0..shape.rows.next_power_of_two().ilog2())
                    .map(|_| transcript.challenge(b"row"))
                    .collect::<Vec<_>>(),
                (0..shape.cols.next_power_of_two().ilog2())
                    .map(|_| transcript.challenge(b"col"))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let claims = tbv::evaluate_claims(&parameters.layout, committed.values(), &points)?;
    let claims_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let (proof, profile) = parameters.prove(committed, &claims)?;
    let wire = proof.to_bytes()?;
    let opening_seconds = start.elapsed().as_secs_f64();
    write_json(output.join("commitment.json"), &statement)?;
    write_json(output.join("claims.json"), &claims)?;
    fs::write(output.join("proof.bin"), &wire)?;
    let start = Instant::now();
    let decoded = Proof::from_bytes(&wire)?;
    let verifier = rayon::ThreadPoolBuilder::new().num_threads(1).build()?;
    let verification = verifier.install(|| parameters.verify(&statement, &claims, &decoded))?;
    let record = json!({"verified":true,"scope":"public-metadata ParComm complete flow","field":"F65521/F65521^8","prover_threads":threads,"verifier_threads":1,"setup_seconds":setup_seconds,"input_seconds":input_seconds,"claim_generation_seconds":claims_seconds,"cache_mode":cache_mode,"commit_seconds":commit_seconds,"opening_with_serialization_seconds":opening_seconds,"total_prover_seconds":commit_seconds+opening_seconds,"verify_decode_seconds":start.elapsed().as_secs_f64(),"proof_bytes":wire.len(),"root_bytes":32*statement.roots.len(),"public_commitment_json_bytes":serde_json::to_vec(&statement)?.len(),"proof_blake2s":format!("{:x?}",digest(&wire)),"prover_profile":profile,"verifier_profile":verification,"matrix_count":shapes.len(),"bucket_count":parameters.layout.buckets.len(),"active_entries":parameters.layout.active,"logical_entries":parameters.layout.logical});
    write_json(output.join("result.json"), &record)?;
    println!("{record}");
    Ok(())
}
