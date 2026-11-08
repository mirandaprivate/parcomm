//! Commit a tower of 8-bit matrices \(2^{N}, 2^{N-1}, \dots, 1\) with Dory,
//! then homomorphically batch-open them on the shared \(N\)-variable grid.
//!
//! Each matrix is committed in its own balanced Dory shape
//! (\(\sigma=\lceil n/2\rceil\) columns). Batch opening block-embeds every
//! smaller matrix into the top-left of the \(N\)-variable grid — the same
//! convention as Jolt's precommitted polynomials.
//!
//! ```text
//! RAYON_NUM_THREADS=64 cargo run --release -p jolt-dory --example ragged_u8_batch -- 36 64
//! ```

use std::env;
use std::time::Instant;

use jolt_dory::{DoryCommitment, DoryHint, DoryScheme};
use jolt_field::{Fr, FromPrimitiveInt, MulPrimitiveInt, RandomSampling};
use jolt_openings::{
    BatchOpeningScheme, CommitmentScheme, EvaluationClaim, HomomorphicBatch, StreamingCommitment,
    VerifierOpeningClaim,
};
use jolt_poly::{EqPolynomial, MultilinearPoly, Point, HIGH_TO_LOW};
use jolt_transcript::{Blake2bTranscript, Transcript};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};
use rayon::prelude::*;

type HomomorphicDoryBatch = HomomorphicBatch<DoryScheme>;

fn main() {
    let max_vars: usize = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(36);
    let threads: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    // Soft cap: the full tower keeps ~2^{N+1} bytes of u8 tables resident.
    // On ~368 GiB hosts, N=36 (~128 GiB) is comfortable; N=38 (~512 GiB) will OOM.
    assert!(
        (1..=37).contains(&max_vars),
        "max_vars must be in 1..=37, got {max_vars}"
    );

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap_or_else(|err| panic!("failed to pin rayon to {threads} threads: {err}"));

    println!(
        "ragged u8 batch: max_vars={max_vars} matrices={} threads={} rayon={}",
        max_vars + 1,
        threads,
        rayon::current_num_threads()
    );

    let wall = Instant::now();
    let t0 = Instant::now();
    let prover_setup = DoryScheme::setup_prover(max_vars);
    let verifier_setup = DoryScheme::setup_verifier(max_vars);
    println!("setup                 {:>10.3}s", t0.elapsed().as_secs_f64());

    let mut rng = ChaCha20Rng::seed_from_u64(0x08_b1_7_u64);
    let mut matrices = Vec::with_capacity(max_vars + 1);
    let mut gen_s = 0.0;
    for n in (0..=max_vars).rev() {
        let len = 1usize << n;
        let t = Instant::now();
        let mut data = vec![0u8; len];
        rng.fill_bytes(&mut data);
        gen_s += t.elapsed().as_secs_f64();
        matrices.push(U8Matrix { num_vars: n, data });
    }
    let bytes: usize = matrices.iter().map(|m| m.data.len()).sum();
    println!(
        "generate u8 tables    {:>10.3}s  ({:.2} GiB)",
        gen_s,
        bytes as f64 / (1u64 << 30) as f64
    );

    let mut commitments = Vec::with_capacity(matrices.len());
    let mut hints = Vec::with_capacity(matrices.len());
    let mut commit_s = 0.0;
    println!();
    println!(
        "{:>6}  {:>12}  {:>8}  {:>8}  {:>10}",
        "n", "len", "rows", "cols", "commit_s"
    );
    for matrix in &matrices {
        let sigma = matrix.num_vars.div_ceil(2);
        let nu = matrix.num_vars - sigma;
        let t = Instant::now();
        let (commitment, hint) = commit_u8_matrix(&matrix.data, matrix.num_vars, &prover_setup);
        let dt = t.elapsed().as_secs_f64();
        commit_s += dt;
        println!(
            "{:>6}  {:>12}  {:>8}  {:>8}  {:>10.3}",
            matrix.num_vars,
            matrix.data.len(),
            1usize << nu,
            1usize << sigma,
            dt
        );
        commitments.push(commitment);
        hints.push(hint);
    }
    println!("commit total          {:>10.3}s", commit_s);

    let embedded: Vec<BlockEmbeddedU8> = matrices
        .into_iter()
        .map(|m| BlockEmbeddedU8::new(m, max_vars))
        .collect();
    let point_coords: Vec<Fr> = (0..max_vars).map(|_| Fr::random(&mut rng)).collect();
    let point = Point::<HIGH_TO_LOW, Fr>::new(point_coords.clone());

    let t = Instant::now();
    let evals: Vec<Fr> = embedded.iter().map(|p| p.evaluate(&point_coords)).collect();
    let eval_s = t.elapsed().as_secs_f64();
    println!("evaluate embeddings   {:>10.3}s", eval_s);

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

    let t = Instant::now();
    let mut prover_transcript = Blake2bTranscript::new(b"ragged-u8-batch");
    let proof = HomomorphicDoryBatch::prove_batch(
        &prover_setup,
        claims.clone(),
        sources,
        hints,
        &mut prover_transcript,
    )
    .unwrap_or_else(|err| panic!("batch open failed: {err:?}"));
    let open_s = t.elapsed().as_secs_f64();
    println!("batch open            {:>10.3}s", open_s);

    let t = Instant::now();
    let mut verifier_transcript = Blake2bTranscript::new(b"ragged-u8-batch");
    HomomorphicDoryBatch::verify_batch(&verifier_setup, &claims, &proof, &mut verifier_transcript)
        .unwrap_or_else(|err| panic!("batch verify failed: {err:?}"));
    let verify_s = t.elapsed().as_secs_f64();
    println!("batch verify          {:>10.3}s", verify_s);
    println!();
    println!(
        "summary  commit={commit_s:.3}s  eval={eval_s:.3}s  open={open_s:.3}s  verify={verify_s:.3}s  wall={:.3}s",
        wall.elapsed().as_secs_f64()
    );
}

struct U8Matrix {
    num_vars: usize,
    data: Vec<u8>,
}

fn commit_u8_matrix(
    data: &[u8],
    num_vars: usize,
    setup: &<DoryScheme as CommitmentScheme>::ProverSetup,
) -> (DoryCommitment, DoryHint) {
    let sigma = num_vars.div_ceil(2);
    let row_width = 1usize << sigma;
    let mut partial = DoryScheme::begin(setup);
    DoryScheme::feed_i128_rows_with(
        &mut partial,
        |i| i128::from(*data.get(i).expect("flat index in range")),
        data.len(),
        row_width,
        setup,
    );
    DoryScheme::finish_with_hint(partial, setup)
}

/// 8-bit table block-embedded into the top-left of a larger Dory matrix.
struct BlockEmbeddedU8 {
    data: Vec<u8>,
    sigma_table: usize,
    nu_table: usize,
    grid_vars: usize,
    sigma_grid: usize,
    nu_grid: usize,
}

impl BlockEmbeddedU8 {
    fn new(matrix: U8Matrix, grid_vars: usize) -> Self {
        let sigma_table = matrix.num_vars.div_ceil(2);
        Self {
            data: matrix.data,
            sigma_table,
            nu_table: matrix.num_vars - sigma_table,
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

    fn small_point<'a>(&self, point: &'a [Fr]) -> (Vec<Fr>, Vec<Fr>) {
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
