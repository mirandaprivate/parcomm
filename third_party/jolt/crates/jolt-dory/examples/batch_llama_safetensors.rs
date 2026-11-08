//! Load Llama safetensors weights straight into memory (pad to Dory's
//! balanced \(2^n\) layout), then batch-commit / batch-open — **no on-disk
//! `.bin` materialization**.
//!
//! Modes:
//! - `i8`  — W8A8 `I8` weights → `feed_i8_rows_with` (`msm_u8` + bias)
//! - `u16` — BF16 bit-reinterpret as `u16` → `feed_u16_rows_with` (`msm_u16`)
//! - `i16` — BF16 bit-reinterpret as `i16` → `feed_i128_rows_with` (legacy)
//!
//! ```text
//! cargo run --release -p jolt-dory --example batch_llama_safetensors -- \
//!   /path/to/Llama-3.1-8B-BF16 u16 64
//! ```

use std::collections::HashMap;
use std::env;
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::time::Instant;

use ark_serialize::CanonicalSerialize;
use jolt_dory::{DoryCommitment, DoryHint, DoryScheme};
use jolt_field::{Fr, FromPrimitiveInt, RandomSampling};
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, HomomorphicBatch, StreamingCommitment,
    VerifierOpeningClaim,
};
use jolt_poly::{EqPolynomial, MultilinearPoly, Point, HIGH_TO_LOW};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::prelude::*;
use serde::Deserialize;

type HomomorphicDoryBatch = HomomorphicBatch<DoryScheme>;

#[derive(Debug, Deserialize)]
struct WeightIndex {
    weight_map: HashMap<String, String>,
}

#[derive(Debug, Deserialize)]
struct TensorInfo {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

struct MatrixMeta {
    #[expect(dead_code, reason = "kept for debugging / future logging")]
    name: String,
    live_len: usize,
    pad_len: usize,
    num_vars: usize,
    sigma: usize,
    nu: usize,
}

enum Coeffs {
    I8(Vec<i8>),
    I16(Vec<i16>),
    U16(Vec<u16>),
}

struct LoadedMatrix {
    meta: MatrixMeta,
    coeffs: Coeffs,
}

fn main() {
    let model_dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("usage: batch_llama_safetensors <model_dir> <i8|u16|i16> [threads]"));
    let mode = env::args().nth(2).unwrap_or_else(|| "i8".to_string());
    let threads: usize = env::args()
        .nth(3)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    assert!(
        matches!(mode.as_str(), "i8" | "u16" | "i16"),
        "mode must be i8, u16, or i16, got {mode}"
    );

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap_or_else(|err| panic!("failed to pin rayon to {threads} threads: {err}"));

    let wall = Instant::now();
    let t0 = Instant::now();
    let matrices = load_from_safetensors(&model_dir, &mode);
    let load_s = t0.elapsed().as_secs_f64();

    let num_matrices = matrices.len();
    let total_live: u64 = matrices.iter().map(|m| m.meta.live_len as u64).sum();
    let total_pad: u64 = matrices.iter().map(|m| m.meta.pad_len as u64).sum();
    let total_bytes: u64 = matrices
        .iter()
        .map(|m| match &m.coeffs {
            Coeffs::I8(v) => v.len() as u64,
            Coeffs::I16(v) => (v.len() * 2) as u64,
            Coeffs::U16(v) => (v.len() * 2) as u64,
        })
        .sum();
    let max_vars = matrices
        .iter()
        .map(|m| m.meta.num_vars)
        .max()
        .unwrap_or(0);

    println!(
        "batch_llama_safetensors: dir={} mode={mode} matrices={num_matrices} max_vars={max_vars} \
         live≈2^{:.2} pad≈2^{:.2} bytes={:.2} GiB (in-memory only) threads={} rayon={}",
        model_dir.display(),
        (total_live as f64).log2(),
        (total_pad as f64).log2(),
        total_bytes as f64 / (1u64 << 30) as f64,
        threads,
        rayon::current_num_threads()
    );
    println!("load+pad in memory    {:>10.3}s", load_s);

    let t0 = Instant::now();
    let prover_setup = DoryScheme::setup_prover(max_vars);
    let verifier_setup = DoryScheme::setup_verifier(max_vars);
    println!("setup                 {:>10.3}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let committed: Vec<(DoryCommitment, DoryHint)> = matrices
        .par_iter()
        .map(|m| commit_matrix(m, &prover_setup))
        .collect();
    let commit_s = t0.elapsed().as_secs_f64();
    let commitment_bytes: u64 = committed
        .iter()
        .map(|(c, _)| c.transcript_payload_len().unwrap_or(0))
        .sum();
    println!(
        "batch commit          {:>10.3}s  ({num_matrices} matrices)",
        commit_s
    );
    println!(
        "commitment size       {commitment_bytes} bytes  ({:.2} KiB, {} B/commitment)",
        commitment_bytes as f64 / 1024.0,
        commitment_bytes / num_matrices as u64
    );

    let embedded: Vec<BlockEmbedded> = matrices
        .into_iter()
        .map(|m| BlockEmbedded::new(m, max_vars))
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x70b8_a11);
    let point_coords: Vec<Fr> = (0..max_vars).map(|_| Fr::random(&mut rng)).collect();
    let point = Point::<HIGH_TO_LOW, Fr>::new(point_coords.clone());

    let t0 = Instant::now();
    let evals: Vec<Fr> = embedded
        .par_iter()
        .map(|p| p.evaluate(&point_coords))
        .collect();
    let eval_s = t0.elapsed().as_secs_f64();
    println!("evaluate embeddings   {:>10.3}s", eval_s);

    let (commitments, hints): (Vec<_>, Vec<_>) = committed.into_iter().unzip();
    let claims: Vec<VerifierOpeningClaim<Fr, _>> = commitments
        .into_iter()
        .zip(evals)
        .map(|(commitment, value)| VerifierOpeningClaim {
            commitment,
            evaluation: EvaluationClaim::new(point.clone(), value),
        })
        .collect();
    let sources: Vec<&dyn MultilinearPoly<Fr>> = embedded
        .iter()
        .map(|p| p as &dyn MultilinearPoly<Fr>)
        .collect();

    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::new(b"batch-llama-safetensors");
    let proof = HomomorphicDoryBatch::prove_batch(
        &prover_setup,
        claims.clone(),
        sources,
        hints,
        &mut prover_transcript,
    )
    .unwrap_or_else(|err| panic!("batch open failed: {err:?}"));
    let open_s = t0.elapsed().as_secs_f64();
    println!("batch open            {:>10.3}s", open_s);

    let mut proof_buf = Vec::new();
    proof
        .0
        .serialize_compressed(&mut proof_buf)
        .expect("proof serialization");
    let proof_bytes = proof_buf.len();
    println!(
        "proof size            {proof_bytes} bytes  ({:.2} KiB)",
        proof_bytes as f64 / 1024.0
    );

    let t0 = Instant::now();
    let mut verifier_transcript = Blake2bTranscript::new(b"batch-llama-safetensors");
    HomomorphicDoryBatch::verify_batch(&verifier_setup, &claims, &proof, &mut verifier_transcript)
        .unwrap_or_else(|err| panic!("batch verify failed: {err:?}"));
    let verify_s = t0.elapsed().as_secs_f64();
    println!("batch verify          {:>10.3}s", verify_s);

    let peak_rss = peak_rss_bytes();
    match peak_rss {
        Some(bytes) => println!(
            "peak RSS              {:>10.3} GiB  ({bytes} bytes)",
            bytes as f64 / (1u64 << 30) as f64
        ),
        None => println!("peak RSS              (unavailable on this platform)"),
    }

    println!();
    print!(
        "summary  load={load_s:.3}s  commit={commit_s:.3}s  commitment_bytes={commitment_bytes}  \
         eval={eval_s:.3}s  open={open_s:.3}s  proof_bytes={proof_bytes}  verify={verify_s:.3}s  \
         wall={:.3}s",
        wall.elapsed().as_secs_f64()
    );
    if let Some(bytes) = peak_rss {
        println!("  peak_rss_bytes={bytes}");
    } else {
        println!();
    }
}

fn next_pow2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    1usize << (usize::BITS - (n - 1).leading_zeros())
}

fn load_from_safetensors(model_dir: &Path, mode: &str) -> Vec<LoadedMatrix> {
    let index_path = model_dir.join("model.safetensors.index.json");
    let index: WeightIndex = serde_json::from_str(
        &fs::read_to_string(&index_path).unwrap_or_else(|e| panic!("read {index_path:?}: {e}")),
    )
    .unwrap_or_else(|e| panic!("parse index: {e}"));

    // Collect (name, shard) candidates then load in parallel by tensor.
    let mut jobs: Vec<(String, PathBuf, TensorInfo)> = Vec::new();
    let mut shard_headers: HashMap<String, (HashMap<String, TensorInfo>, usize)> = HashMap::new();

    for shard in index.weight_map.values().cloned().collect::<std::collections::BTreeSet<_>>() {
        let path = model_dir.join(&shard);
        let (header, data_base) = parse_safetensors_header(&path);
        let _ = shard_headers.insert(shard.clone(), (header, data_base));
    }

    for (name, shard) in &index.weight_map {
        let (header, _) = shard_headers.get(shard).expect("shard header");
        let Some(info) = header.get(name) else {
            continue;
        };
        if !select_tensor(mode, &info.dtype, &info.shape) {
            continue;
        }
        jobs.push((
            name.clone(),
            model_dir.join(shard),
            TensorInfo {
                dtype: info.dtype.clone(),
                shape: info.shape.clone(),
                data_offsets: info.data_offsets,
            },
        ));
    }
    jobs.sort_by(|a, b| a.0.cmp(&b.0));

    // Parallel load: each job opens its shard and reads only its slice.
    let data_bases: HashMap<PathBuf, usize> = shard_headers
        .iter()
        .map(|(shard, (_, base))| (model_dir.join(shard), *base))
        .collect();

    jobs.into_par_iter()
        .map(|(name, path, info)| {
            let data_base = *data_bases.get(&path).expect("data_base");
            load_one_tensor(&name, &path, data_base, &info, mode)
        })
        .collect()
}

fn select_tensor(mode: &str, dtype: &str, shape: &[usize]) -> bool {
    if shape.len() != 2 {
        return false;
    }
    let (rows, cols) = (shape[0], shape[1]);
    if rows <= 1 || cols <= 1 {
        return false;
    }
    match mode {
        "i8" => dtype == "I8",
        "u16" | "i16" => dtype == "BF16",
        _ => false,
    }
}

fn parse_safetensors_header(path: &Path) -> (HashMap<String, TensorInfo>, usize) {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).expect("header len");
    let header_len = u64::from_le_bytes(len_buf) as usize;
    let mut header_bytes = vec![0u8; header_len];
    f.read_exact(&mut header_bytes).expect("header json");
    let raw: serde_json::Map<String, serde_json::Value> =
        serde_json::from_slice(&header_bytes).expect("header parse");
    let mut out = HashMap::new();
    for (k, v) in raw {
        if k == "__metadata__" {
            continue;
        }
        let info: TensorInfo = serde_json::from_value(v).unwrap_or_else(|e| {
            panic!("tensor info for {k}: {e}");
        });
        let _ = out.insert(k, info);
    }
    (out, 8 + header_len)
}

fn load_one_tensor(
    name: &str,
    path: &Path,
    data_base: usize,
    info: &TensorInfo,
    mode: &str,
) -> LoadedMatrix {
    let rows = info.shape[0];
    let cols = info.shape[1];
    let live_len = rows * cols;
    let pad_len = next_pow2(live_len);
    let num_vars = pad_len.trailing_zeros() as usize; // pad_len is power of two
    debug_assert_eq!(1usize << num_vars, pad_len);
    let sigma = num_vars.div_ceil(2);
    let nu = num_vars - sigma;

    let start = data_base + info.data_offsets[0];
    let end = data_base + info.data_offsets[1];
    let nbytes = end - start;

    let mut file = File::open(path).unwrap_or_else(|e| panic!("open {path:?}: {e}"));
    let _ = file
        .seek(SeekFrom::Start(start as u64))
        .unwrap_or_else(|e| panic!("seek {path:?}: {e}"));

    let coeffs = match mode {
        "i8" => {
            assert_eq!(info.dtype, "I8");
            assert_eq!(nbytes, live_len, "{name}: I8 size");
            let mut buf = vec![0u8; pad_len];
            file.read_exact(&mut buf[..live_len])
                .unwrap_or_else(|e| panic!("read {name}: {e}"));
            // Zero-pad already in place. Reinterpret as i8 without copy.
            // SAFETY: i8/u8 same size and alignment; every u8 bit pattern is a valid i8.
            let padded = unsafe {
                let ptr = buf.as_mut_ptr().cast::<i8>();
                let len = buf.len();
                let cap = buf.capacity();
                std::mem::forget(buf);
                Vec::from_raw_parts(ptr, len, cap)
            };
            Coeffs::I8(padded)
        }
        "i16" => {
            assert_eq!(info.dtype, "BF16");
            assert_eq!(nbytes, live_len * 2, "{name}: BF16 size");
            let mut padded = vec![0i16; pad_len];
            // SAFETY: reading LE bytes into an aligned i16 buffer; bf16 bits ≡ i16 patterns.
            let live_bytes = unsafe {
                std::slice::from_raw_parts_mut(padded.as_mut_ptr().cast::<u8>(), live_len * 2)
            };
            file.read_exact(live_bytes)
                .unwrap_or_else(|e| panic!("read {name}: {e}"));
            Coeffs::I16(padded)
        }
        "u16" => {
            assert_eq!(info.dtype, "BF16");
            assert_eq!(nbytes, live_len * 2, "{name}: BF16 size");
            let mut padded = vec![0u16; pad_len];
            // SAFETY: LE bf16 bit patterns reinterpreted as unsigned u16.
            let live_bytes = unsafe {
                std::slice::from_raw_parts_mut(padded.as_mut_ptr().cast::<u8>(), live_len * 2)
            };
            file.read_exact(live_bytes)
                .unwrap_or_else(|e| panic!("read {name}: {e}"));
            Coeffs::U16(padded)
        }
        other => panic!("bad mode {other}"),
    };

    LoadedMatrix {
        meta: MatrixMeta {
            name: name.to_string(),
            live_len,
            pad_len,
            num_vars,
            sigma,
            nu,
        },
        coeffs,
    }
}

fn commit_matrix(
    m: &LoadedMatrix,
    setup: &<DoryScheme as CommitmentScheme>::ProverSetup,
) -> (DoryCommitment, DoryHint) {
    let row_width = 1usize << m.meta.sigma;
    let mut partial = DoryScheme::begin(setup);
    match &m.coeffs {
        Coeffs::I8(data) => {
            DoryScheme::feed_i8_rows_with(
                &mut partial,
                |i| *data.get(i).expect("in range"),
                data.len(),
                row_width,
                setup,
            );
        }
        Coeffs::U16(data) => {
            DoryScheme::feed_u16_rows_with(
                &mut partial,
                |i| *data.get(i).expect("in range"),
                data.len(),
                row_width,
                setup,
            );
        }
        Coeffs::I16(data) => {
            DoryScheme::feed_i128_rows_with(
                &mut partial,
                |i| i128::from(*data.get(i).expect("in range")),
                data.len(),
                row_width,
                setup,
            );
        }
    }
    DoryScheme::finish_with_hint(partial, setup)
}

fn peak_rss_bytes() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        let Some(rest) = line.strip_prefix("VmHWM:") else {
            continue;
        };
        let kb: u64 = rest.split_whitespace().next()?.parse().ok()?;
        return Some(kb.saturating_mul(1024));
    }
    None
}

fn coeff_fr(coeffs: &Coeffs, i: usize) -> Fr {
    match coeffs {
        Coeffs::I8(v) => Fr::from_i64(i64::from(*v.get(i).expect("in range"))),
        Coeffs::I16(v) => Fr::from_i64(i64::from(*v.get(i).expect("in range"))),
        Coeffs::U16(v) => Fr::from_u64(u64::from(*v.get(i).expect("in range"))),
    }
}

fn coeff_len(coeffs: &Coeffs) -> usize {
    match coeffs {
        Coeffs::I8(v) => v.len(),
        Coeffs::I16(v) => v.len(),
        Coeffs::U16(v) => v.len(),
    }
}

struct BlockEmbedded {
    coeffs: Coeffs,
    sigma_table: usize,
    nu_table: usize,
    grid_vars: usize,
    sigma_grid: usize,
    nu_grid: usize,
}

impl BlockEmbedded {
    fn new(matrix: LoadedMatrix, grid_vars: usize) -> Self {
        Self {
            coeffs: matrix.coeffs,
            sigma_table: matrix.meta.sigma,
            nu_table: matrix.meta.nu,
            grid_vars,
            sigma_grid: grid_vars.div_ceil(2),
            nu_grid: grid_vars - grid_vars.div_ceil(2),
        }
    }

    fn embedding_scale(&self, point: &[Fr]) -> Fr {
        let row_hi = self.nu_grid - self.nu_table;
        let col_hi = self.sigma_grid - self.sigma_table;
        let one = Fr::from_u64(1);
        let mut scale = one;
        for challenge in point.iter().take(row_hi) {
            scale *= one - *challenge;
        }
        for challenge in point.iter().skip(self.nu_grid).take(col_hi) {
            scale *= one - *challenge;
        }
        scale
    }

    fn small_point(&self, point: &[Fr]) -> (Vec<Fr>, Vec<Fr>) {
        let row_hi = self.nu_grid - self.nu_table;
        let col_hi = self.sigma_grid - self.sigma_table;
        let row_lo = point[row_hi..self.nu_grid].to_vec();
        let col_lo = point[self.nu_grid + col_hi..].to_vec();
        (row_lo, col_lo)
    }
}

impl MultilinearPoly<Fr> for BlockEmbedded {
    fn num_vars(&self) -> usize {
        self.grid_vars
    }

    fn evaluate(&self, point: &[Fr]) -> Fr {
        let scale = self.embedding_scale(point);
        if coeff_len(&self.coeffs) == 0 {
            return Fr::from_u64(0);
        }
        let (row_lo, col_lo) = self.small_point(point);
        let eq_row = EqPolynomial::<Fr>::evals(&row_lo, None);
        let eq_col = EqPolynomial::<Fr>::evals(&col_lo, None);
        let small_cols = 1usize << self.sigma_table;
        let small_rows = 1usize << self.nu_table;
        let inner = (0..small_rows)
            .into_par_iter()
            .map(|r| {
                let eq_r = eq_row.get(r).copied().unwrap_or_else(|| Fr::from_u64(0));
                let mut acc = Fr::from_u64(0);
                for c in 0..small_cols {
                    let coeff = coeff_fr(&self.coeffs, r * small_cols + c);
                    let eq_c = eq_col.get(c).copied().unwrap_or_else(|| Fr::from_u64(0));
                    acc += eq_c * coeff;
                }
                eq_r * acc
            })
            .reduce(|| Fr::from_u64(0), |a, b| a + b);
        scale * inner
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[Fr])) {
        let num_cols = 1usize << sigma;
        let small_cols = 1usize << self.sigma_table;
        let small_rows = 1usize << self.nu_table;
        let mut row = vec![Fr::from_u64(0); num_cols];
        for r in 0..small_rows {
            for c in 0..small_cols {
                if let Some(slot) = row.get_mut(c) {
                    *slot = coeff_fr(&self.coeffs, r * small_cols + c);
                }
            }
            f(r, &row);
            row.fill(Fr::from_u64(0));
        }
    }

    fn fold_rows(&self, left: &[Fr], sigma: usize) -> Vec<Fr> {
        let num_cols = 1usize << sigma;
        let small_cols = 1usize << self.sigma_table;
        let small_rows = 1usize << self.nu_table;
        let mut result = vec![Fr::from_u64(0); num_cols];
        let cols: Vec<Fr> = (0..small_cols)
            .into_par_iter()
            .map(|c| {
                let mut acc = Fr::from_u64(0);
                for r in 0..small_rows {
                    let coeff = coeff_fr(&self.coeffs, r * small_cols + c);
                    let left_r = left.get(r).copied().unwrap_or_else(|| Fr::from_u64(0));
                    acc += left_r * coeff;
                }
                acc
            })
            .collect();
        result
            .get_mut(..small_cols)
            .expect("block columns fit")
            .copy_from_slice(&cols);
        result
    }
}
