//! Full-extension QAFold PCS benchmark for a packed F65521 Boolean table.
use std::{
    env,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

use blake2::{Blake2s256, Digest};
use mikros_qafold::{
    Prime65521, ShortBaseField, full_extension::FullPrime65521Ext8 as F, packed::PackedPcs,
};
use rand::{SeedableRng, rngs::StdRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde_json::{Value, json};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(Debug)]
struct Config {
    log_n: usize,
    threads: usize,
    reps: usize,
    columns: usize,
    security_bits: usize,
    output: PathBuf,
}

fn main() {
    if let Err(error) = run() {
        eprintln!("QAFold PCS benchmark failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let config = parse_args()?;
    #[cfg(feature = "parallel")]
    rayon::ThreadPoolBuilder::new()
        .num_threads(config.threads)
        .build_global()?;
    #[cfg(not(feature = "parallel"))]
    if config.threads != 1 {
        return Err("build with the parallel feature for more than one worker".into());
    }
    let mut output = Output::new(&config.output)?;
    output.line("F65521 PCS | QAFold over F_(65521^8) | packed opening")?;
    output.line(&format!(
        "Base elements: 2^{} = {} | packed extension elements: 2^{} = {}",
        config.log_n,
        1usize << config.log_n,
        config.log_n - 3,
        1usize << (config.log_n - 3),
    ))?;
    output.line(&format!(
        "Workers: {} | measured proofs: {} | warmups: 0 | extension columns: {} | query/RF target: {} bits",
        config.threads, config.reps, config.columns, config.security_bits,
    ))?;
    output.line("Online prover = commit + packing-opening reduction + QAFold opening + recursive Decide + serialization.")?;
    output.line("Setup and witness generation are reported separately. Verification includes parsing and all proof checks.")?;
    output.line("Proof bytes count the serialized opening, including packing reduction and recursive Decide; the separately reported commitment is 32 bytes.")?;
    output.line("RAM is whole-process VmHWM including setup, inputs, all samples and verification. The query/RF target is not a whole-protocol security certificate.")?;
    output.line("[setup] sampling and committing full-extension QA parameters")?;
    let start = Instant::now();
    let pcs = PackedPcs::new(
        config.log_n,
        config.columns,
        config.security_bits,
        [0x51; 32],
    )?;
    let setup_ms = start.elapsed().as_secs_f64() * 1000.0;
    output.line(&format!("[setup] complete: {setup_ms:.3} ms"))?;
    output.line(&format!(
        "Parameter digest: {}",
        hex(&pcs.parameter_digest())
    ))?;

    let mut rows = Vec::with_capacity(config.reps);
    for run in 0..config.reps {
        output.line(&format!(
            "[sample {}/{}] generating deterministic pseudorandom F65521 witness (offline)",
            run + 1,
            config.reps
        ))?;
        let start = Instant::now();
        let values = witness(config.log_n, run);
        let witness_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut rng = StdRng::seed_from_u64(0x5141_504f_494e_5400 ^ run as u64);
        let point: Vec<F> = (0..config.log_n)
            .map(|_| F::sample_uniform(&mut rng))
            .collect();
        let context = format!(
            "mikros-qafold/pcs-65521-benchmark/v1/{}/{run}",
            config.log_n
        )
        .into_bytes();

        output.line(&format!(
            "[sample {}/{}] commit start",
            run + 1,
            config.reps
        ))?;
        let start = Instant::now();
        let data = pcs.commit(values)?;
        let commit_ms = start.elapsed().as_secs_f64() * 1000.0;
        let commitment = data.commitment();
        let commit_timings = data.commit_timings();
        #[cfg(feature = "profile-encoding")]
        report_encoding_profile(&mut output, &data, run + 1)?;
        output.line(&format!(
            "  Commit stages: packing={:.3} ms copy={:.3} ms Mobius={:.3} ms allocation={:.3} ms QA encoding={:.3} ms release={:.3} ms Merkle={:.3} ms",
            commit_timings.packing_ms, commit_timings.coefficient_copy_ms,
            commit_timings.mobius_ms, commit_timings.allocation_ms, commit_timings.encoding_ms,
            commit_timings.witness_release_ms, commit_timings.merkle_ms,
        ))?;
        output.line(&format!(
            "[sample {}/{}] commit complete: {commit_ms:.3} ms; opening + Decide start",
            run + 1,
            config.reps
        ))?;
        let start = Instant::now();
        let proof = pcs.open(data, &point, &context)?;
        let serialization_start = Instant::now();
        let bytes = proof.to_bytes()?;
        let serialization_ms = serialization_start.elapsed().as_secs_f64() * 1000.0;
        let opening_ms = start.elapsed().as_secs_f64() * 1000.0;
        let prover_ms = commit_ms + opening_ms;
        output.line(&format!(
            "[sample {}/{}] opening + Decide complete: {opening_ms:.3} ms; full verify start",
            run + 1,
            config.reps
        ))?;
        let start = Instant::now();
        pcs.verify(&commitment, &point, proof.claimed_value, &bytes, &context)?;
        let verifier_ms = start.elapsed().as_secs_f64() * 1000.0;
        let peak_rss_mib = peak_rss_mib();
        let row = json!({
            "sample":run + 1, "samples":config.reps, "warmups":0,
            "base_log_n":config.log_n, "base_elements":1usize << config.log_n,
            "packed_log_n":config.log_n - 3, "packed_elements":1usize << (config.log_n - 3),
            "base_field":"F65521", "qa_field":"F65521^8", "packing_factor":8,
            "columns":config.columns, "workers":config.threads,
            "query_rf_security_bits":config.security_bits,
            "parameter_digest":hex(&pcs.parameter_digest()),
            "commitment_root":hex(&commitment.root),
            "setup_ms":setup_ms, "witness_ms":witness_ms,
            "commit_ms":commit_ms, "opening_decide_serialization_ms":opening_ms,
            "packing_ms":commit_timings.packing_ms,
            "coefficient_copy_ms":commit_timings.coefficient_copy_ms,
            "mobius_ms":commit_timings.mobius_ms,
            "qa_allocation_ms":commit_timings.allocation_ms,
            "qa_encoding_ms":commit_timings.encoding_ms,
            "witness_release_ms":commit_timings.witness_release_ms,
            "merkle_ms":commit_timings.merkle_ms,
            "projection_ms":proof.timings.projection_ms,
            "sumcheck_ms":proof.timings.sumcheck_ms,
            "native_opening_ms":proof.timings.native_opening_ms,
            "decide_ms":proof.timings.decide_ms,
            "outer_serialization_ms":serialization_ms,
            "opening_queries":proof.opening_queries,
            "decide_queries":proof.decide_queries,
            "prover_ms":prover_ms, "verifier_ms":verifier_ms,
            "proof_blake2s":hex(&Blake2s256::digest(&bytes)),
            "proof_bytes":bytes.len(), "commitment_bytes":32,
            "opening_plus_commitment_bytes":bytes.len() + 32,
            "whole_process_peak_rss_mib":peak_rss_mib,
            "verified":true,
        });
        output.line(&format!(
            "[sample {}/{}] verified: prover={prover_ms:.3} ms verifier={verifier_ms:.3} ms proof={} bytes RAM={} MiB",
            run + 1, config.reps, bytes.len(), ram_display(peak_rss_mib),
        ))?;
        output.line(&format!(
            "  Projection {:>12.3} ms | Packing sumcheck {:>12.3} ms | Native opening {:>12.3} ms | Decide {:>12.3} ms",
            proof.timings.projection_ms, proof.timings.sumcheck_ms,
            proof.timings.native_opening_ms, proof.timings.decide_ms,
        ))?;
        output.line(&format!(
            "  Opening queries {:?} | Decide queries {:?}",
            proof.opening_queries, proof.decide_queries
        ))?;
        output.line(&format!(
            "QAFOLD_PCS_SAMPLE {}",
            serde_json::to_string(&row)?
        ))?;
        rows.push(row);
    }
    output.line("")?;
    output.line("Metric                         Median")?;
    output.line("-----------------------------------------------")?;
    for (label, key, scale, unit) in [
        ("Commit", "commit_ms", 1.0, "ms"),
        ("  Packing", "packing_ms", 1.0, "ms"),
        ("  Coefficient copy", "coefficient_copy_ms", 1.0, "ms"),
        ("  Mobius", "mobius_ms", 1.0, "ms"),
        ("  Codeword allocation", "qa_allocation_ms", 1.0, "ms"),
        ("  QA encoding", "qa_encoding_ms", 1.0, "ms"),
        ("  Witness release", "witness_release_ms", 1.0, "ms"),
        ("  Merkle tree", "merkle_ms", 1.0, "ms"),
        (
            "Opening + Decide + serialize",
            "opening_decide_serialization_ms",
            1.0,
            "ms",
        ),
        ("  Projection", "projection_ms", 1.0, "ms"),
        ("  Packing sumcheck", "sumcheck_ms", 1.0, "ms"),
        ("  Native opening", "native_opening_ms", 1.0, "ms"),
        ("  Recursive Decide", "decide_ms", 1.0, "ms"),
        ("Online prover", "prover_ms", 1.0, "ms"),
        ("Full verifier", "verifier_ms", 1.0, "ms"),
        ("Proof size", "proof_bytes", 1.0 / 1024.0, "KiB"),
    ] {
        output.line(&format!(
            "{label:<30} {:>12.3} {unit}",
            median(&rows, key) * scale
        ))?;
    }
    output.line(&format!(
        "{:<30} {:>12} MiB",
        "Whole-process peak RAM",
        ram_display(peak_rss_mib())
    ))?;
    output.line(&format!(
        "Setup: {setup_ms:.3} ms (offline). Every measured proof passed full verification."
    ))?;
    output.line(&format!("Log: {}", config.output.display()))?;
    Ok(())
}

#[cfg(feature = "profile-encoding")]
fn report_encoding_profile(
    output: &mut Output,
    data: &mikros_qafold::packed::PackedData,
    sample: usize,
) -> Result<()> {
    let columns = data.encoding_columns();
    let wall_ms = data.commit_timings().encoding_ms;
    let mut totals = [0.0f64; 5];
    let mut rows = Vec::with_capacity(columns.len());
    for (index, column) in columns.iter().enumerate() {
        let p = &column.stages;
        let ms = |d: std::time::Duration| d.as_secs_f64() * 1000.0;
        let accounted = p.wht + p.multiplication + p.copy;
        // PackedPcs has systematic rate 1/2: no nested parity tasks overlap.
        // Thus these per-column stage durations partition its elapsed time.
        if p.wht_calls != 2 || accounted > p.total {
            return Err("unexpected QA profile shape or overlapping stage timers".into());
        }
        let values = [
            ms(p.total),
            ms(p.wht),
            ms(p.multiplication),
            ms(p.copy),
            ms(p.total - accounted),
        ];
        for (sum, value) in totals.iter_mut().zip(values) {
            *sum += value;
        }
        rows.push(json!({
            "column":index, "start_ms":column.start_ms, "end_ms":column.end_ms,
            "total_ms":values[0], "wht_ms":values[1], "multiplication_ms":values[2],
            "copy_ms":values[3], "other_ms":values[4],
            "wht_calls":p.wht_calls, "multiplications":p.multiplications,
            "copied_elements":p.copied_elements,
        }));
    }
    output.line(&format!(
        "  QA profile: wall={wall_ms:.3} ms | {} columns | column elapsed sum={:.3} ms | average active columns={:.2}",
        columns.len(), totals[0], totals[0] / wall_ms,
    ))?;
    output.line("  Stage                       Mean/column (ms)   Column-elapsed share")?;
    for (index, label) in [
        (1, "WHT (including cache moves)"),
        (2, "Alpha field multiplication"),
        (3, "Message/parity copy"),
        (4, "Other/timer residual"),
    ] {
        output.line(&format!(
            "  {label:<28} {:>15.3} {:>20.2}%",
            totals[index] / columns.len() as f64,
            totals[index] / totals[0] * 100.0
        ))?;
    }
    output.line("  Shares use summed overlapping column elapsed, not CPU cycles or end-to-end wall time. Allocation, Mobius, packing and Merkle are outside this profile.")?;
    output.line(&format!(
        "QAFOLD_ENCODING_PROFILE {}",
        json!({
            "sample":sample, "column_count":columns.len(), "wall_ms":wall_ms,
            "column_elapsed_ms":totals[0], "wht_ms":totals[1],
            "multiplication_ms":totals[2], "copy_ms":totals[3], "other_ms":totals[4],
            "wht_percent":totals[1] / totals[0] * 100.0,
            "multiplication_percent":totals[2] / totals[0] * 100.0,
            "copy_percent":totals[3] / totals[0] * 100.0,
            "other_percent":totals[4] / totals[0] * 100.0,
            "average_active_columns":totals[0] / wall_ms, "columns":rows,
        })
    ))?;
    Ok(())
}

fn witness(log_n: usize, run: usize) -> Vec<Prime65521> {
    // Independently seeded fixed-size chunks make the input independent of worker
    // count while avoiding a shared RNG or a huge intermediate integer array.
    const CHUNK: usize = 1 << 15;
    let mut values = vec![Prime65521::new(0); 1usize << log_n];
    let fill = |(index, chunk): (usize, &mut [Prime65521])| {
        let mut rng = StdRng::seed_from_u64(
            0x5141_5749_544e_0000 ^ (run as u64).rotate_left(32) ^ index as u64,
        );
        for value in chunk {
            *value = Prime65521::sample_uniform(&mut rng);
        }
    };
    #[cfg(feature = "parallel")]
    values.par_chunks_mut(CHUNK).enumerate().for_each(fill);
    #[cfg(not(feature = "parallel"))]
    values.chunks_mut(CHUNK).enumerate().for_each(fill);
    values
}

fn parse_args() -> Result<Config> {
    let mut config = Config {
        log_n: 30,
        threads: 64,
        reps: 1,
        columns: 32,
        security_bits: 100,
        output: PathBuf::from("target/paper-results/qafold-f65521-n30.log"),
    };
    let mut args = env::args().skip(1);
    while let Some(argument) = args.next() {
        if argument == "--bench" {
            continue;
        }
        if argument == "--help" || argument == "-h" {
            println!(
                "cargo bench -p mikros-qafold --bench pcs_65521 -- [--log-n 30] [--threads 64] [--reps 1] [--columns 32] [--security-bits 100] [--output PATH]"
            );
            std::process::exit(0);
        }
        let value = args
            .next()
            .ok_or_else(|| format!("missing value for {argument}"))?;
        match argument.as_str() {
            "--log-n" => config.log_n = value.parse()?,
            "--threads" => config.threads = value.parse()?,
            "--reps" => config.reps = value.parse()?,
            "--columns" => config.columns = value.parse()?,
            "--security-bits" => config.security_bits = value.parse()?,
            "--output" => config.output = PathBuf::from(value),
            _ => return Err(format!("unknown argument: {argument}").into()),
        }
    }
    if !(4..=30).contains(&config.log_n)
        || config.threads == 0
        || config.reps == 0
        || config.columns <= 1
        || !config.columns.is_power_of_two()
        || config.columns > 1usize << (config.log_n - 3)
    {
        return Err("unsupported size, worker count, sample count or column count".into());
    }
    Ok(config)
}

struct Output {
    file: BufWriter<File>,
}
impl Output {
    fn new(path: &PathBuf) -> io::Result<Self> {
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        Ok(Self {
            file: BufWriter::new(File::create(path)?),
        })
    }
    fn line(&mut self, line: &str) -> io::Result<()> {
        let mut stdout = io::stdout().lock();
        writeln!(stdout, "{line}")?;
        stdout.flush()?;
        writeln!(self.file, "{line}")?;
        self.file.flush()
    }
}
fn peak_rss_mib() -> Option<f64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    Some(line.split_whitespace().nth(1)?.parse::<f64>().ok()? / 1024.0)
}
fn ram_display(value: Option<f64>) -> String {
    value
        .map(|v| format!("{v:.3}"))
        .unwrap_or_else(|| "unavailable".into())
}
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
fn median(rows: &[Value], key: &str) -> f64 {
    let mut values = rows
        .iter()
        .map(|r| r[key].as_f64().unwrap())
        .collect::<Vec<_>>();
    values.sort_by(f64::total_cmp);
    let n = values.len();
    if n % 2 == 0 {
        (values[n / 2 - 1] + values[n / 2]) / 2.0
    } else {
        values[n / 2]
    }
}
