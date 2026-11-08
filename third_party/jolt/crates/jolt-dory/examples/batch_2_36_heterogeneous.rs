//! Heterogeneous batch: near-power-of-two u8 matrices descending from
//! (2^18-1)×(2^17-1):
//!
//! ```text
//! n=35: (2^18-1)×(2^17-1)   padded 2^18×2^17 = 2^35
//! n=34: (2^17-1)×(2^17-1)   padded 2^17×2^17 = 2^34
//! n=33: (2^17-1)×(2^16-1)   padded 2^17×2^16 = 2^33
//! ...
//! n=1:  degenerate → 1 live coefficient in a 1×2 pad
//! ```
//!
//! Padded total = 2^36 - 2 ≈ 2^36. Each matrix is committed in its own
//! balanced Dory shape (σ=⌈n/2⌉ columns); batch opening block-embeds every
//! matrix into the shared n=35 grid (Jolt's precommitted-polynomial
//! convention) and opens once at a shared 35-variable point.
//!
//! Reports: commitment time, commitment size, opening time, proof size,
//! verifier time.
//!
//! ```text
//! cargo run --release -p jolt-dory --example batch_2_36_heterogeneous -- 35 64
//! ```

use std::env;
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
use rand_core::{RngCore, SeedableRng};
use rayon::prelude::*;

type HomomorphicDoryBatch = HomomorphicBatch<DoryScheme>;

const GEN_BAND_ROWS: usize = 256;

fn main() {
    let max_vars: usize = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(35);
    let threads: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    assert!(
        (1..=37).contains(&max_vars),
        "max_vars must be in 1..=37, got {max_vars}"
    );

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap_or_else(|err| panic!("failed to pin rayon to {threads} threads: {err}"));

    let num_matrices = max_vars;
    println!(
        "batch_2_36_heterogeneous: matrices={num_matrices} (n={max_vars}..1) \
         grid_vars={max_vars} threads={} rayon={}",
        threads,
        rayon::current_num_threads()
    );

    let wall = Instant::now();
    let t0 = Instant::now();
    let prover_setup = DoryScheme::setup_prover(max_vars);
    let verifier_setup = DoryScheme::setup_verifier(max_vars);
    println!("setup                 {:>10.3}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let matrices: Vec<NearPow2Matrix> = (1..=max_vars)
        .rev()
        .map(|n| generate_matrix(n, 0x0236_e7))
        .collect();
    let live_total: usize = matrices.iter().map(|m| m.live_rows * m.live_cols).sum();
    let pad_total: usize = matrices.iter().map(|m| m.data.len()).sum();
    println!(
        "generate u8 tables    {:>10.3}s  (live={live_total} ~2^{:.2}, padded={pad_total} ~2^{:.2}, {:.2} GiB pad)",
        t0.elapsed().as_secs_f64(),
        (live_total as f64).log2(),
        (pad_total as f64).log2(),
        pad_total as f64 / (1u64 << 30) as f64
    );

    println!();
    println!(
        "{:>4}  {:>10}×{:<10}  {:>8}×{:<8}  {:>14}  {:>10}",
        "n", "live_a", "live_b", "pad_r", "pad_c", "live_len", "commit_s"
    );

    // --- batch commit ---
    let t0 = Instant::now();
    let committed: Vec<(DoryCommitment, DoryHint, f64)> = matrices
        .par_iter()
        .map(|m| {
            let t = Instant::now();
            let (c, h) = commit_matrix(m, &prover_setup);
            (c, h, t.elapsed().as_secs_f64())
        })
        .collect();
    let commit_s = t0.elapsed().as_secs_f64();

    for (m, (_c, _h, dt)) in matrices.iter().zip(committed.iter()) {
        // Print larger factor first to match the (2^18-1)×(2^17-1) order.
        let (a, b) = if m.live_cols >= m.live_rows {
            (m.live_cols, m.live_rows)
        } else {
            (m.live_rows, m.live_cols)
        };
        println!(
            "{:>4}  {:>10}×{:<10}  {:>8}×{:<8}  {:>14}  {:>10.3}",
            m.num_vars,
            a,
            b,
            m.pad_rows,
            m.pad_cols,
            m.live_rows * m.live_cols,
            dt
        );
    }
    println!("batch commit total    {:>10.3}s  (parallel wall)", commit_s);

    let commitment_bytes: u64 = committed
        .iter()
        .map(|(c, _, _)| c.transcript_payload_len().unwrap_or(0))
        .sum();
    println!(
        "commitment size       {commitment_bytes} bytes  ({} per GT commitment, {num_matrices} commitments)",
        commitment_bytes / num_matrices as u64
    );

    let embedded: Vec<BlockEmbeddedU8> = matrices
        .into_iter()
        .map(|m| BlockEmbeddedU8::new(m, max_vars))
        .collect();

    let mut rng = ChaCha20Rng::seed_from_u64(0x0b_36);
    let point_coords: Vec<Fr> = (0..max_vars).map(|_| Fr::random(&mut rng)).collect();
    let point = Point::<HIGH_TO_LOW, Fr>::new(point_coords.clone());

    let t0 = Instant::now();
    let evals: Vec<Fr> = embedded
        .par_iter()
        .map(|p| p.evaluate(&point_coords))
        .collect();
    let eval_s = t0.elapsed().as_secs_f64();
    println!("evaluate embeddings   {:>10.3}s", eval_s);

    let (commitments, hints): (Vec<_>, Vec<_>) = committed
        .into_iter()
        .map(|(c, h, _)| (c, h))
        .unzip();
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

    // --- batch open ---
    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::new(b"batch-2-36-heterogeneous");
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

    // --- batch verify ---
    let t0 = Instant::now();
    let mut verifier_transcript = Blake2bTranscript::new(b"batch-2-36-heterogeneous");
    HomomorphicDoryBatch::verify_batch(&verifier_setup, &claims, &proof, &mut verifier_transcript)
        .unwrap_or_else(|err| panic!("batch verify failed: {err:?}"));
    let verify_s = t0.elapsed().as_secs_f64();
    println!("batch verify          {:>10.3}s", verify_s);

    println!();
    println!(
        "summary  commit={commit_s:.3}s  commitment_bytes={commitment_bytes}  eval={eval_s:.3}s  \
         open={open_s:.3}s  proof_bytes={proof_bytes}  verify={verify_s:.3}s  wall={:.3}s",
        wall.elapsed().as_secs_f64()
    );
}

struct NearPow2Matrix {
    num_vars: usize,
    sigma: usize,
    nu: usize,
    live_rows: usize,
    live_cols: usize,
    pad_rows: usize,
    pad_cols: usize,
    data: Vec<u8>,
}

fn generate_matrix(n: usize, seed: u64) -> NearPow2Matrix {
    let sigma = n.div_ceil(2);
    let nu = n - sigma;
    let pad_cols = 1usize << sigma;
    let pad_rows = 1usize << nu;
    // Live block is (2^σ-1)×(2^ν-1); ν=0 (n=1) is degenerate (×0) — keep a
    // single live coefficient in the 1×2 pad.
    let live_cols = pad_cols - 1;
    let live_rows = pad_rows.saturating_sub(1);
    let (live_rows, live_cols) = if live_rows == 0 || live_cols == 0 {
        (1.min(pad_rows), 1.min(pad_cols))
    } else {
        (live_rows, live_cols)
    };

    let mut data = vec![0u8; pad_rows * pad_cols];
    // Parallel fill in bands of rows, one deterministic rng stream per band.
    data.par_chunks_mut(GEN_BAND_ROWS * pad_cols)
        .enumerate()
        .for_each(|(band, chunk)| {
            let band_seed = seed ^ ((n as u64) << 40) ^ (band as u64).wrapping_mul(0x9e37_79b9);
            let mut rng = ChaCha20Rng::seed_from_u64(band_seed);
            let first_row = band * GEN_BAND_ROWS;
            for (i, row) in chunk.chunks_mut(pad_cols).enumerate() {
                if first_row + i < live_rows {
                    rng.fill_bytes(&mut row[..live_cols]);
                }
            }
        });
    NearPow2Matrix {
        num_vars: n,
        sigma,
        nu,
        live_rows,
        live_cols,
        pad_rows,
        pad_cols,
        data,
    }
}

fn commit_matrix(
    m: &NearPow2Matrix,
    setup: &<DoryScheme as CommitmentScheme>::ProverSetup,
) -> (DoryCommitment, DoryHint) {
    let mut partial = DoryScheme::begin(setup);
    DoryScheme::feed_i128_rows_with(
        &mut partial,
        |i| i128::from(*m.data.get(i).expect("in range")),
        m.data.len(),
        m.pad_cols,
        setup,
    );
    DoryScheme::finish_with_hint(partial, setup)
}

/// 8-bit table block-embedded into the top-left of the shared Dory grid.
struct BlockEmbeddedU8 {
    data: Vec<u8>,
    sigma_table: usize,
    nu_table: usize,
    grid_vars: usize,
    sigma_grid: usize,
    nu_grid: usize,
}

impl BlockEmbeddedU8 {
    fn new(matrix: NearPow2Matrix, grid_vars: usize) -> Self {
        Self {
            data: matrix.data,
            sigma_table: matrix.sigma,
            nu_table: matrix.nu,
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

impl MultilinearPoly<Fr> for BlockEmbeddedU8 {
    fn num_vars(&self) -> usize {
        self.grid_vars
    }

    fn evaluate(&self, point: &[Fr]) -> Fr {
        debug_assert_eq!(point.len(), self.grid_vars);
        let scale = self.embedding_scale(point);
        if self.data.is_empty() {
            return Fr::from_u64(0);
        }
        let (row_lo, col_lo) = self.small_point(point);
        let eq_row = EqPolynomial::<Fr>::evals(&row_lo, None);
        let eq_col = EqPolynomial::<Fr>::evals(&col_lo, None);
        let small_cols = 1usize << self.sigma_table;
        let inner = self
            .data
            .par_chunks(small_cols)
            .zip(eq_row.par_iter())
            .map(|(row, &eq_r)| {
                row.iter()
                    .zip(eq_col.iter())
                    .map(|(&coeff, &eq_c)| eq_c.mul_u64(u64::from(coeff)))
                    .fold(Fr::from_u64(0), |acc, x| acc + x)
                    * eq_r
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
                let coeff = u64::from(*self.data.get(r * small_cols + c).expect("in range"));
                if let Some(slot) = row.get_mut(c) {
                    *slot = Fr::from_u64(coeff);
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
                    let coeff = u64::from(*self.data.get(r * small_cols + c).expect("in range"));
                    if coeff != 0 {
                        let left_r = left.get(r).copied().unwrap_or_else(|| Fr::from_u64(0));
                        acc += left_r.mul_u64(coeff);
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
