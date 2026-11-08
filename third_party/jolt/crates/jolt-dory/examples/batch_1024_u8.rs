//! Batch-commit and batch-open 1024 matrices of shape 1023×1023 (u8).
//!
//! Total live coefficients: 1024 · 1023² ≈ 2^30. Each matrix is zero-padded
//! to 1024×1024 (= 2^20) so Dory sees a power-of-two balanced layout.
//!
//! ```text
//! cargo run --release -p jolt-dory --example batch_1024_u8 -- 8
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

const SIDE: usize = 1023;
const PAD_SIDE: usize = 1024;
const NUM_MATRICES: usize = 1024;
const NUM_VARS: usize = 20; // PAD_SIDE² = 2^20

fn main() {
    let threads: usize = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(num_cpus_fallback);

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap_or_else(|err| panic!("failed to pin rayon to {threads} threads: {err}"));

    let live_coeffs = NUM_MATRICES * SIDE * SIDE;
    let padded_coeffs = NUM_MATRICES * PAD_SIDE * PAD_SIDE;
    println!(
        "batch_1024_u8: matrices={NUM_MATRICES} shape={SIDE}x{SIDE} \
         live_coeffs={live_coeffs} (~2^{:.2}) pad={PAD_SIDE}x{PAD_SIDE} \
         threads={} rayon={}",
        (live_coeffs as f64).log2(),
        threads,
        rayon::current_num_threads()
    );
    println!(
        "padded_coeffs={padded_coeffs} (=2^{})",
        (padded_coeffs as f64).log2() as usize
    );

    let wall = Instant::now();

    let t0 = Instant::now();
    let prover_setup = DoryScheme::setup_prover(NUM_VARS);
    let verifier_setup = DoryScheme::setup_verifier(NUM_VARS);
    println!("setup                 {:>10.3}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let mut rng = ChaCha20Rng::seed_from_u64(0x1024);
    let matrices: Vec<U8Matrix> = (0..NUM_MATRICES)
        .map(|_| {
            let mut data = vec![0u8; PAD_SIDE * PAD_SIDE];
            for r in 0..SIDE {
                rng.fill_bytes(&mut data[r * PAD_SIDE..r * PAD_SIDE + SIDE]);
            }
            U8Matrix { data }
        })
        .collect();
    println!(
        "generate u8 tables    {:>10.3}s  ({:.2} GiB live / {:.2} GiB padded)",
        t0.elapsed().as_secs_f64(),
        live_coeffs as f64 / (1u64 << 30) as f64,
        padded_coeffs as f64 / (1u64 << 30) as f64
    );

    let t0 = Instant::now();
    let committed: Vec<(DoryCommitment, DoryHint)> = matrices
        .par_iter()
        .map(|m| commit_padded_u8(&m.data, &prover_setup))
        .collect();
    let commit_s = t0.elapsed().as_secs_f64();
    println!(
        "batch commit          {:>10.3}s  ({NUM_MATRICES} mats, rayon-parallel)",
        commit_s
    );

    let mut rng = ChaCha20Rng::seed_from_u64(0x9e17);
    let point_coords: Vec<Fr> = (0..NUM_VARS).map(|_| Fr::random(&mut rng)).collect();
    let point = Point::<HIGH_TO_LOW, Fr>::new(point_coords.clone());

    let t0 = Instant::now();
    let evals: Vec<Fr> = matrices
        .par_iter()
        .map(|m| m.evaluate(&point_coords))
        .collect();
    let eval_s = t0.elapsed().as_secs_f64();
    println!("evaluate at point     {:>10.3}s", eval_s);

    let (commitments, hints): (Vec<_>, Vec<_>) = committed.into_iter().unzip();
    let claims: Vec<VerifierOpeningClaim<Fr, _>> = commitments
        .into_iter()
        .zip(evals)
        .map(|(commitment, value)| VerifierOpeningClaim {
            commitment,
            evaluation: EvaluationClaim::new(point.clone(), value),
        })
        .collect();
    let sources: Vec<&dyn MultilinearPoly<Fr>> = matrices
        .iter()
        .map(|m| m as &dyn MultilinearPoly<Fr>)
        .collect();

    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::new(b"batch-1024-u8");
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

    let t0 = Instant::now();
    let mut verifier_transcript = Blake2bTranscript::new(b"batch-1024-u8");
    HomomorphicDoryBatch::verify_batch(&verifier_setup, &claims, &proof, &mut verifier_transcript)
        .unwrap_or_else(|err| panic!("batch verify failed: {err:?}"));
    let verify_s = t0.elapsed().as_secs_f64();
    println!("batch verify          {:>10.3}s", verify_s);

    println!();
    println!(
        "summary  commit={commit_s:.3}s  eval={eval_s:.3}s  open={open_s:.3}s  \
         verify={verify_s:.3}s  wall={:.3}s",
        wall.elapsed().as_secs_f64()
    );
}

fn num_cpus_fallback() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(8)
}

fn commit_padded_u8(
    data: &[u8],
    setup: &<DoryScheme as CommitmentScheme>::ProverSetup,
) -> (DoryCommitment, DoryHint) {
    debug_assert_eq!(data.len(), PAD_SIDE * PAD_SIDE);
    let mut partial = DoryScheme::begin(setup);
    DoryScheme::feed_i128_rows_with(
        &mut partial,
        |i| i128::from(*data.get(i).expect("flat index in range")),
        data.len(),
        PAD_SIDE,
        setup,
    );
    DoryScheme::finish_with_hint(partial, setup)
}

/// Dense u8 table over a 2^20 Boolean hypercube (1024×1024 row-major).
struct U8Matrix {
    data: Vec<u8>,
}

impl MultilinearPoly<Fr> for U8Matrix {
    fn num_vars(&self) -> usize {
        NUM_VARS
    }

    fn evaluate(&self, point: &[Fr]) -> Fr {
        debug_assert_eq!(point.len(), NUM_VARS);
        let eq = EqPolynomial::<Fr>::evals(point, None);
        self.data
            .par_iter()
            .zip(eq.par_iter())
            .map(|(&coeff, &e)| e.mul_u64(u64::from(coeff)))
            .reduce(|| Fr::from_u64(0), |a, b| a + b)
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[Fr])) {
        let num_cols = 1usize << sigma;
        let mut row = vec![Fr::from_u64(0); num_cols];
        for (i, chunk) in self.data.chunks(num_cols).enumerate() {
            for (slot, &coeff) in row.iter_mut().zip(chunk.iter()) {
                *slot = Fr::from_u64(u64::from(coeff));
            }
            f(i, &row);
        }
    }

    fn fold_rows(&self, left: &[Fr], sigma: usize) -> Vec<Fr> {
        let num_cols = 1usize << sigma;
        debug_assert_eq!(left.len() * num_cols, self.data.len());
        (0..num_cols)
            .into_par_iter()
            .map(|c| {
                let mut acc = Fr::from_u64(0);
                for (r, &l) in left.iter().enumerate() {
                    let coeff = u64::from(*self.data.get(r * num_cols + c).expect("in range"));
                    if coeff != 0 {
                        acc += l.mul_u64(coeff);
                    }
                }
                acc
            })
            .collect()
    }
}
