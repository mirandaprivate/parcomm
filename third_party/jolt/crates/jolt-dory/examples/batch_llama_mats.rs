//! Batch-commit / batch-open Llama weight matrices produced by
//! `scripts/llama_to_dory_mats.py`.
//!
//! Each matrix is a zero-padded flat coefficient vector of length `2^n`,
//! committed in Dory's balanced `(2^{n-σ})×(2^σ)` layout. Batch opening
//! block-embeds every matrix into the shared `max_vars` grid (same
//! convention as Jolt precommitted polynomials) and opens once via the
//! homomorphic RLC batch.
//!
//! ```text
//! cargo run --release -p jolt-dory --example batch_llama_mats -- \
//!   /path/to/jolt-dory-i8 64
//! ```

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use ark_serialize::CanonicalSerialize;
use jolt_dory::{DoryCommitment, DoryHint, DoryScheme};
use jolt_field::{Fr, FromPrimitiveInt, MulPrimitiveInt, RandomSampling};
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
struct Manifest {
    model: String,
    mode: String,
    num_matrices: usize,
    max_vars: usize,
    total_live_coeffs: u64,
    total_pad_coeffs: u64,
    total_bytes: u64,
    matrices: Vec<MatrixMeta>,
}

#[derive(Debug, Deserialize)]
struct MatrixMeta {
    index: usize,
    name: String,
    file: String,
    live_len: usize,
    pad_len: usize,
    num_vars: usize,
    sigma: usize,
    nu: usize,
    dtype: String,
    elem_size: usize,
}

enum Coeffs {
    I8(Vec<i8>),
    I16(Vec<i16>),
}

struct LoadedMatrix {
    meta: MatrixMeta,
    coeffs: Coeffs,
}

fn main() {
    let dir = env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    let threads: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap_or_else(|err| panic!("failed to pin rayon to {threads} threads: {err}"));

    let man_path = dir.join("manifest.json");
    let manifest: Manifest = serde_json::from_str(
        &fs::read_to_string(&man_path).unwrap_or_else(|e| panic!("read {man_path:?}: {e}")),
    )
    .unwrap_or_else(|e| panic!("parse manifest: {e}"));

    println!(
        "batch_llama_mats: model={} mode={} matrices={} max_vars={} \
         live≈2^{:.2} pad≈2^{:.2} bytes={:.2} GiB threads={} rayon={}",
        manifest.model,
        manifest.mode,
        manifest.num_matrices,
        manifest.max_vars,
        (manifest.total_live_coeffs as f64).log2(),
        (manifest.total_pad_coeffs as f64).log2(),
        manifest.total_bytes as f64 / (1u64 << 30) as f64,
        threads,
        rayon::current_num_threads()
    );

    let wall = Instant::now();
    let t0 = Instant::now();
    let matrices: Vec<LoadedMatrix> = manifest
        .matrices
        .par_iter()
        .map(|m| load_matrix(&dir, m))
        .collect();
    println!(
        "load bins             {:>10.3}s",
        t0.elapsed().as_secs_f64()
    );

    let max_vars = manifest.max_vars;
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
        "batch commit          {:>10.3}s  ({} matrices)",
        commit_s, matrices.len()
    );
    println!(
        "commitment size       {commitment_bytes} bytes  ({:.2} KiB, {} B/commitment)",
        commitment_bytes as f64 / 1024.0,
        commitment_bytes / matrices.len() as u64
    );

    let embedded: Vec<BlockEmbedded> = matrices
        .into_iter()
        .map(|m| BlockEmbedded::new(m, max_vars))
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x11aa_8b16);
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
    let mut prover_transcript = Blake2bTranscript::new(b"batch-llama-mats");
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
    let mut verifier_transcript = Blake2bTranscript::new(b"batch-llama-mats");
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
        "summary  commit={commit_s:.3}s  commitment_bytes={commitment_bytes}  eval={eval_s:.3}s  \
         open={open_s:.3}s  proof_bytes={proof_bytes}  verify={verify_s:.3}s  wall={:.3}s",
        wall.elapsed().as_secs_f64()
    );
    if let Some(bytes) = peak_rss {
        println!("  peak_rss_bytes={bytes}");
    } else {
        println!();
    }
}

/// Linux high-water mark from `/proc/self/status` (`VmHWM`). Returns `None`
/// on non-Linux hosts (e.g. local macOS smoke tests).
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

fn load_matrix(dir: &Path, meta: &MatrixMeta) -> LoadedMatrix {
    let path = dir.join(&meta.file);
    let bytes = fs::read(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let expected = meta.pad_len * meta.elem_size;
    assert_eq!(
        bytes.len(),
        expected,
        "{}: got {} bytes, expected {}",
        meta.name,
        bytes.len(),
        expected
    );
    let coeffs = match meta.dtype.as_str() {
        "i8" => {
            let v: Vec<i8> = bytes.iter().map(|&b| b as i8).collect();
            Coeffs::I8(v)
        }
        "i16" => {
            assert_eq!(bytes.len() % 2, 0);
            let v: Vec<i16> = bytes
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
            Coeffs::I16(v)
        }
        other => panic!("unsupported dtype {other}"),
    };
    LoadedMatrix {
        meta: MatrixMeta {
            index: meta.index,
            name: meta.name.clone(),
            file: meta.file.clone(),
            live_len: meta.live_len,
            pad_len: meta.pad_len,
            num_vars: meta.num_vars,
            sigma: meta.sigma,
            nu: meta.nu,
            dtype: meta.dtype.clone(),
            elem_size: meta.elem_size,
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
            DoryScheme::feed_i128_rows_with(
                &mut partial,
                |i| i128::from(*data.get(i).expect("in range")),
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

fn coeff_at(coeffs: &Coeffs, i: usize) -> i64 {
    match coeffs {
        Coeffs::I8(v) => i64::from(*v.get(i).expect("in range")),
        Coeffs::I16(v) => i64::from(*v.get(i).expect("in range")),
    }
}

fn coeff_len(coeffs: &Coeffs) -> usize {
    match coeffs {
        Coeffs::I8(v) => v.len(),
        Coeffs::I16(v) => v.len(),
    }
}

/// Signed coefficient table block-embedded into the shared Dory grid.
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
        debug_assert_eq!(point.len(), self.grid_vars);
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
                    let coeff = coeff_at(&self.coeffs, r * small_cols + c);
                    if coeff != 0 {
                        let eq_c = eq_col.get(c).copied().unwrap_or_else(|| Fr::from_u64(0));
                        acc += eq_c.mul_i64(coeff);
                    }
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
                let coeff = coeff_at(&self.coeffs, r * small_cols + c);
                if let Some(slot) = row.get_mut(c) {
                    *slot = Fr::from_i64(coeff);
                }
            }
            f(r, &row);
            row.fill(Fr::from_u64(0));
        }
    }

    fn fold_rows(&self, left: &[Fr], sigma: usize) -> Vec<Fr> {
        debug_assert_eq!(left.len(), 1usize << self.grid_vars.saturating_sub(sigma));
        let num_cols = 1usize << sigma;
        let small_cols = 1usize << self.sigma_table;
        let small_rows = 1usize << self.nu_table;
        let mut result = vec![Fr::from_u64(0); num_cols];
        let cols: Vec<Fr> = (0..small_cols)
            .into_par_iter()
            .map(|c| {
                let mut acc = Fr::from_u64(0);
                for r in 0..small_rows {
                    let coeff = coeff_at(&self.coeffs, r * small_cols + c);
                    if coeff != 0 {
                        let left_r = left.get(r).copied().unwrap_or_else(|| Fr::from_u64(0));
                        acc += left_r.mul_i64(coeff);
                    }
                }
                acc
            })
            .collect();
        result
            .get_mut(..small_cols)
            .expect("block columns fit in the grid")
            .copy_from_slice(&cols);
        result
    }
}
