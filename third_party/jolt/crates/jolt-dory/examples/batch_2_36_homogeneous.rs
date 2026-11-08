//! Homogeneous batch: 2^16 u8 matrices of (2^10-1)×(2^10-1), each zero-padded
//! to 1024×1024 (2^20 coefficients), committed independently with Dory, then
//! batch-opened at one shared 20-variable point via the homomorphic RLC batch.
//!
//! Padded total = 2^16 · 2^20 = 2^36 coefficients (64 GiB of u8 tables).
//!
//! Reports: commitment time, commitment size, opening time, proof size,
//! verifier time.
//!
//! ```text
//! cargo run --release -p jolt-dory --example batch_2_36_homogeneous -- 16 64
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

const SIDE: usize = 1023; // 2^10 - 1 live rows/cols
const PAD_SIDE: usize = 1024; // 2^10 padded
const NUM_VARS: usize = 20; // per-matrix padded coefficients = 2^20
const NU: usize = 10; // 2^10 rows; sigma = 10 → 2^10 cols

fn main() {
    let log_count: usize = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(16);
    let threads: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let num_matrices = 1usize << log_count;

    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .unwrap_or_else(|err| panic!("failed to pin rayon to {threads} threads: {err}"));

    let live_coeffs = num_matrices * SIDE * SIDE;
    let padded_coeffs = num_matrices * PAD_SIDE * PAD_SIDE;
    println!(
        "batch_2_36_homogeneous: matrices=2^{log_count}={num_matrices} shape={SIDE}x{SIDE} \
         pad={PAD_SIDE}x{PAD_SIDE} live_coeffs={live_coeffs} (~2^{:.2}) \
         padded_coeffs={padded_coeffs} (=2^{}) threads={} rayon={}",
        (live_coeffs as f64).log2(),
        log_count + NUM_VARS,
        threads,
        rayon::current_num_threads()
    );

    let wall = Instant::now();

    let t0 = Instant::now();
    let prover_setup = DoryScheme::setup_prover(NUM_VARS);
    let verifier_setup = DoryScheme::setup_verifier(NUM_VARS);
    println!("setup                 {:>10.3}s", t0.elapsed().as_secs_f64());

    let t0 = Instant::now();
    let matrices: Vec<Vec<u8>> = (0..num_matrices)
        .into_par_iter()
        .map(|idx| {
            let seed = 0x0236_u64 ^ (idx as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let mut rng = ChaCha20Rng::seed_from_u64(seed);
            let mut data = vec![0u8; PAD_SIDE * PAD_SIDE];
            for r in 0..SIDE {
                rng.fill_bytes(&mut data[r * PAD_SIDE..r * PAD_SIDE + SIDE]);
            }
            data
        })
        .collect();
    println!(
        "generate u8 tables    {:>10.3}s  ({:.2} GiB live / {:.2} GiB padded)",
        t0.elapsed().as_secs_f64(),
        live_coeffs as f64 / (1u64 << 30) as f64,
        padded_coeffs as f64 / (1u64 << 30) as f64
    );

    // --- batch commit ---
    let t0 = Instant::now();
    let committed: Vec<(DoryCommitment, DoryHint)> = matrices
        .par_iter()
        .map(|data| commit_padded_u8(data, &prover_setup))
        .collect();
    let commit_s = t0.elapsed().as_secs_f64();
    let commitment_bytes: u64 = committed
        .iter()
        .map(|(c, _)| c.transcript_payload_len().unwrap_or(0))
        .sum();
    println!(
        "batch commit          {:>10.3}s  ({num_matrices} matrices, rayon-parallel)",
        commit_s
    );
    println!(
        "commitment size       {commitment_bytes} bytes  ({} per GT commitment, {:.2} MiB total)",
        commitment_bytes / num_matrices as u64,
        commitment_bytes as f64 / (1u64 << 20) as f64
    );

    // --- shared opening point and per-matrix evaluations ---
    let mut rng = ChaCha20Rng::seed_from_u64(0x9e17);
    let point_coords: Vec<Fr> = (0..NUM_VARS).map(|_| Fr::random(&mut rng)).collect();
    let point = Point::<HIGH_TO_LOW, Fr>::new(point_coords.clone());

    let t0 = Instant::now();
    let eq_row = EqPolynomial::<Fr>::evals(&point_coords[..NU], None);
    let eq_col = EqPolynomial::<Fr>::evals(&point_coords[NU..], None);
    let evals: Vec<Fr> = matrices
        .par_iter()
        .map(|m| eval_matrix(m, &eq_row, &eq_col))
        .collect();
    let eval_s = t0.elapsed().as_secs_f64();
    println!("evaluate at point     {:>10.3}s", eval_s);

    let sources_owned: Vec<U8Matrix> = matrices.into_iter().map(|data| U8Matrix { data }).collect();
    let (commitments, hints): (Vec<_>, Vec<_>) = committed.into_iter().unzip();
    let claims: Vec<VerifierOpeningClaim<Fr, _>> = commitments
        .into_iter()
        .zip(evals)
        .map(|(commitment, value)| VerifierOpeningClaim {
            commitment,
            evaluation: EvaluationClaim::new(point.clone(), value),
        })
        .collect();
    let sources: Vec<&dyn MultilinearPoly<Fr>> = sources_owned
        .iter()
        .map(|m| m as &dyn MultilinearPoly<Fr>)
        .collect();

    // --- batch open ---
    let t0 = Instant::now();
    let mut prover_transcript = Blake2bTranscript::new(b"batch-2-36-homogeneous");
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
    let mut verifier_transcript = Blake2bTranscript::new(b"batch-2-36-homogeneous");
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

/// eq-factorized evaluation: Σ_r eq_row[r] · Σ_c eq_col[c] · M[r][c].
fn eval_matrix(data: &[u8], eq_row: &[Fr], eq_col: &[Fr]) -> Fr {
    data.chunks(PAD_SIDE)
        .zip(eq_row.iter())
        .map(|(row, &er)| {
            let inner = row
                .iter()
                .zip(eq_col.iter())
                .map(|(&b, &ec)| ec.mul_u64(u64::from(b)))
                .fold(Fr::from_u64(0), |acc, x| acc + x);
            er * inner
        })
        .fold(Fr::from_u64(0), |acc, x| acc + x)
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
        let eq_row = EqPolynomial::<Fr>::evals(&point[..NU], None);
        let eq_col = EqPolynomial::<Fr>::evals(&point[NU..], None);
        eval_matrix(&self.data, &eq_row, &eq_col)
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
