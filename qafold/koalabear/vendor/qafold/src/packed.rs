//! Eight base-field values per QA message, with an authenticated Boolean MLE opening.
//!
//! A base-linear lane projection does not commute with extension-field evaluation.
//! Eight conjugate evaluations repair that mismatch. Their batched sumcheck ends
//! in one native QAFold opening and its complete recursive generator Decide.

use blake2::{Blake2s256, Digest};
use rand::{SeedableRng, rngs::StdRng};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use std::time::Instant;

use crate::qafold_short_pc::{
    ShortQaDecideProof, ShortQaFoldCommitment, ShortQaFoldCommitmentCache, ShortQaFoldOpeningProof,
    ShortQaFoldPC, ShortQaFoldParameters, ShortQaFoldSetupCommitments,
};
use crate::{Error, FullPrime65521Ext8 as F, Prime65521 as B, ShortBaseField, ShortFieldElement};

const MAGIC: &[u8; 8] = b"QA655PK1";
const MAX_PROOF_BYTES: usize = 256 * 1024 * 1024;
#[cfg(any(feature = "parallel", test))]
const PARALLEL_THRESHOLD: usize = 1 << 12;
const CACHE_TILE_LOG: usize = 10;
const CACHE_TILE: usize = 1 << CACHE_TILE_LOG;

// A lane accumulator contains at most 2^27 products of canonical prime-field
// coefficients. Its maximum is 2^27 * 65520^2 < 2^60, including parallel merges.
type RawLanes = [u64; 64];

pub struct PackedPcs {
    log_n: usize,
    pc: ShortQaFoldPC<F>,
    pp: ShortQaFoldParameters<F>,
    setup: ShortQaFoldSetupCommitments<F>,
}

pub struct PackedData {
    root: ShortQaFoldCommitment,
    profile: [u8; 32],
    values: Vec<F>,
    cache: ShortQaFoldCommitmentCache<F>,
    timings: CommitTimings,
}

impl PackedData {
    pub fn commitment(&self) -> ShortQaFoldCommitment {
        self.root
    }

    pub fn commit_timings(&self) -> CommitTimings {
        self.timings
    }

    #[cfg(feature = "profile-encoding")]
    pub fn encoding_columns(&self) -> &[crate::qafold_short_pc::QaEncodingColumnProfile] {
        &self.cache.encoding_columns
    }
}

/// Wall times for the online commitment stages; allocation may defer page faults
/// to the subsequent encoding pass. These measurements are not proof messages.
#[derive(Clone, Copy, Debug, Default)]
pub struct CommitTimings {
    pub packing_ms: f64,
    pub coefficient_copy_ms: f64,
    pub mobius_ms: f64,
    pub allocation_ms: f64,
    pub encoding_ms: f64,
    pub witness_release_ms: f64,
    pub merkle_ms: f64,
}

#[derive(Clone, Debug, Default)]
pub struct OpeningTimings {
    pub projection_ms: f64,
    pub sumcheck_ms: f64,
    pub native_opening_ms: f64,
    pub decide_ms: f64,
}

pub struct PackedProof {
    pub claimed_value: F,
    pub timings: OpeningTimings,
    pub opening_queries: Vec<usize>,
    pub decide_queries: Vec<usize>,
    evaluations: [F; 8],
    rounds: Vec<[F; 2]>,
    final_value: F,
    opening: Vec<u8>,
    decide: Vec<u8>,
}

impl PackedProof {
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut out = MAGIC.to_vec();
        put_field(&mut out, self.claimed_value)?;
        for value in self.evaluations {
            put_field(&mut out, value)?;
        }
        out.extend_from_slice(&(self.rounds.len() as u64).to_le_bytes());
        for round in &self.rounds {
            for &value in round {
                put_field(&mut out, value)?;
            }
        }
        put_field(&mut out, self.final_value)?;
        for part in [&self.opening, &self.decide] {
            out.extend_from_slice(&(part.len() as u64).to_le_bytes());
            out.extend_from_slice(part);
        }
        if out.len() > MAX_PROOF_BYTES {
            return Err(Error::EncodingError);
        }
        Ok(out)
    }
}

impl PackedPcs {
    /// Public preprocessing. `log_n` counts base-field values, before packing.
    pub fn new(
        log_n: usize,
        columns: usize,
        security_bits: usize,
        seed: [u8; 32],
    ) -> Result<Self, Error> {
        if !(4..=30).contains(&log_n) {
            return Err(Error::InvalidNumberOfVariables);
        }
        let pc = ShortQaFoldPC::<F>::new_with_recursive_index(
            1usize << (log_n - 3),
            columns,
            2,
            2,
            security_bits,
        )?;
        let (pp, setup) = pc.setup(&mut StdRng::from_seed(seed))?;
        pc.validate_setup_commitments(&pp, &setup)?;
        Ok(Self {
            log_n,
            pc,
            pp,
            setup,
        })
    }

    pub fn packed_len(&self) -> usize {
        1usize << (self.log_n - 3)
    }
    pub fn parameter_digest(&self) -> [u8; 32] {
        self.pp.parameter_digest
    }

    /// Consumes an MSB-first Boolean evaluation table over F_65521.
    /// Packing, the Boolean-to-monomial transform, QA encoding, and hashing are online.
    pub fn commit(&self, values: Vec<B>) -> Result<PackedData, Error> {
        if values.len() != 1usize << self.log_n {
            return Err(Error::InvalidNumberOfVariables);
        }
        let stage = Instant::now();
        #[cfg(feature = "parallel")]
        let packed: Vec<F> = values
            .par_chunks_exact(8)
            .map(|chunk| F::from_coefficients(chunk.try_into().unwrap()))
            .collect();
        #[cfg(not(feature = "parallel"))]
        let packed: Vec<F> = values
            .chunks_exact(8)
            .map(|chunk| F::from_coefficients(chunk.try_into().unwrap()))
            .collect();
        drop(values);
        let packing_ms = stage.elapsed().as_secs_f64() * 1000.0;
        let stage = Instant::now();
        #[cfg(feature = "parallel")]
        let mut coefficients: Vec<F> = packed.par_iter().copied().collect();
        #[cfg(not(feature = "parallel"))]
        let mut coefficients = packed.clone();
        let coefficient_copy_ms = stage.elapsed().as_secs_f64() * 1000.0;
        let stage = Instant::now();
        mobius_in_place(&mut coefficients);
        let mobius_ms = stage.elapsed().as_secs_f64() * 1000.0;
        let (root, cache) = self.pc.commit_owned(&self.pp, coefficients)?;
        let [allocation_ms, encoding_ms, witness_release_ms, merkle_ms] = cache.commit_timings;
        Ok(PackedData {
            root,
            profile: self.pp.parameter_digest,
            values: packed,
            cache,
            timings: CommitTimings {
                packing_ms,
                coefficient_copy_ms,
                mobius_ms,
                allocation_ms,
                encoding_ms,
                witness_release_ms,
                merkle_ms,
            },
        })
    }

    pub fn open(
        &self,
        data: PackedData,
        point: &[F],
        context: &[u8],
    ) -> Result<PackedProof, Error> {
        if point.len() != self.log_n || data.profile != self.pp.parameter_digest {
            return Err(Error::InvalidParameters(
                "packed opening profile or point mismatch".into(),
            ));
        }
        let start = Instant::now();
        let outer = &point[..self.log_n - 3];
        let (mut weights, lanes) = project_with_equality_weights(&data.values, outer);
        let low = equality_table(&point[self.log_n - 3..]);
        let claimed_value = lanes
            .iter()
            .zip(&low)
            .fold(F::zero(), |sum, (&v, &w)| sum + v * w);
        let basis = basis();
        let evaluations = std::array::from_fn(|i| {
            (0..8).fold(F::zero(), |sum, j| {
                sum + basis[j] * lanes[j].frobenius((8 - i) % 8)
            })
        });
        if projected_claim(&evaluations, &low) != claimed_value {
            return Err(Error::InvalidCommitment);
        }
        let mut transcript = self.transcript(&data.root, point, claimed_value, context)?;
        for &value in &evaluations {
            transcript.field(value)?;
        }
        let rho = transcript.challenge();
        let powers = powers8(rho);
        let mut claim = (0..8).fold(F::zero(), |sum, i| sum + powers[i] * evaluations[i]);
        // D(u) is an F_p-linear map of eq(outer,u). Precompute its basis images.
        let images: [F; 8] = std::array::from_fn(|j| {
            (0..8).fold(F::zero(), |sum, i| {
                sum + powers[i] * basis[j].frobenius((8 - i) % 8)
            })
        });
        apply_weight_images(&mut weights, &images);
        let projection_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let mut values = data.values;
        let mut rounds = Vec::with_capacity(outer.len());
        let mut reduced_point = Vec::with_capacity(outer.len());
        while values.len() > 1 {
            let half = values.len() / 2;
            let message = round_message(&values, &weights);
            transcript.field(message[0])?;
            transcript.field(message[1])?;
            let challenge = transcript.challenge();
            claim = round_evaluation(claim, message, challenge);
            fold_pair(&mut values, &mut weights, challenge);
            values.truncate(half);
            weights.truncate(half);
            rounds.push(message);
            reduced_point.push(challenge);
        }
        let final_value = values[0];
        let final_weight = batch_weight(outer, &reduced_point, &powers);
        if weights[0] != final_weight || claim != final_value * final_weight {
            return Err(Error::InvalidCommitment);
        }
        drop(values);
        drop(weights);
        let sumcheck_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let (actual, opening, reduction) =
            self.pc
                .open_with_reduction(&self.pp, &data.root, &reduced_point, data.cache)?;
        if actual != final_value || !reduction.ok {
            return Err(Error::InvalidCommitment);
        }
        let native_opening_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let decide = self.pc.prove_decide(&self.pp, &self.setup, &reduction)?;
        let decide_ms = start.elapsed().as_secs_f64() * 1000.0;
        Ok(PackedProof {
            claimed_value,
            evaluations,
            rounds,
            final_value,
            timings: OpeningTimings {
                projection_ms,
                sumcheck_ms,
                native_opening_ms,
                decide_ms,
            },
            opening_queries: opening.round_query_counts(),
            decide_queries: decide.round_query_counts(),
            opening: opening.to_bytes()?,
            decide: decide.to_bytes()?,
        })
    }

    /// Requires only the public setup, original base-table claim, and proof bytes.
    pub fn verify(
        &self,
        root: &ShortQaFoldCommitment,
        point: &[F],
        claimed_value: F,
        bytes: &[u8],
        context: &[u8],
    ) -> Result<(), Error> {
        if point.len() != self.log_n || bytes.len() > MAX_PROOF_BYTES {
            return Err(Error::InvalidCommitment);
        }
        let mut reader = Reader(bytes);
        if reader.take(8)? != MAGIC || reader.field()? != claimed_value {
            return Err(Error::InvalidCommitment);
        }
        let mut evaluations = [F::zero(); 8];
        for value in &mut evaluations {
            *value = reader.field()?;
        }
        let low = equality_table(&point[self.log_n - 3..]);
        if projected_claim(&evaluations, &low) != claimed_value {
            return Err(Error::InvalidCommitment);
        }
        let mut transcript = self.transcript(root, point, claimed_value, context)?;
        for &value in &evaluations {
            transcript.field(value)?;
        }
        let powers = powers8(transcript.challenge());
        let mut claim = (0..8).fold(F::zero(), |sum, i| sum + powers[i] * evaluations[i]);
        let count = reader.length()?;
        if count != self.log_n - 3 {
            return Err(Error::InvalidCommitment);
        }
        let mut reduced_point = Vec::with_capacity(count);
        for _ in 0..count {
            let message = [reader.field()?, reader.field()?];
            transcript.field(message[0])?;
            transcript.field(message[1])?;
            let challenge = transcript.challenge();
            claim = round_evaluation(claim, message, challenge);
            reduced_point.push(challenge);
        }
        let final_value = reader.field()?;
        let expected_weight = batch_weight(&point[..self.log_n - 3], &reduced_point, &powers);
        if claim != final_value * expected_weight {
            return Err(Error::InvalidCommitment);
        }
        let opening = ShortQaFoldOpeningProof::<F>::from_bytes(reader.part()?)?;
        let decide = ShortQaDecideProof::<F>::from_bytes(reader.part()?)?;
        if !reader.0.is_empty() {
            return Err(Error::InvalidCommitment);
        }
        let pending = self.pc.verify_opening_reduction(
            &self.pp,
            root,
            &reduced_point,
            final_value,
            &opening,
        )?;
        if !pending.ok
            || !self
                .pc
                .verify_decide(&self.pp, &self.setup, &pending, &decide)?
        {
            return Err(Error::InvalidCommitment);
        }
        Ok(())
    }

    fn transcript(
        &self,
        root: &ShortQaFoldCommitment,
        point: &[F],
        value: F,
        context: &[u8],
    ) -> Result<Transcript, Error> {
        let mut transcript = Transcript(Blake2s256::new());
        transcript.bytes(b"mikros/qafold-f65521-ext8/packed-boolean-mle/v1");
        transcript.bytes(&self.pp.parameter_digest);
        transcript.bytes(&(self.log_n as u64).to_le_bytes());
        transcript.bytes(&root.root);
        transcript.bytes(context);
        for &coordinate in point {
            transcript.field(coordinate)?;
        }
        transcript.field(value)?;
        Ok(transcript)
    }
}

fn basis() -> [F; 8] {
    std::array::from_fn(|j| {
        let mut coefficients = [B::zero(); 8];
        coefficients[j] = B::one();
        F::from_coefficients(coefficients)
    })
}

fn powers8(value: F) -> [F; 8] {
    let mut out = [F::one(); 8];
    for i in 1..8 {
        out[i] = out[i - 1] * value;
    }
    out
}

fn projected_claim(evaluations: &[F; 8], low: &[F]) -> F {
    // Inverse DFT of Frob^i(v_i) recovers each lane MLE at the original point.
    let theta_inverse = basis()[1].try_inverse().unwrap();
    let inverse_basis = powers8(theta_inverse);
    let omega_inverse = B::from_u64(57852).try_inverse().unwrap();
    let inverse_eight = B::from_u64(8).try_inverse().unwrap();
    let conjugates: [F; 8] = std::array::from_fn(|i| evaluations[i].frobenius(i));
    let mut omega_step = B::one();
    let mut claim = F::zero();
    for j in 0..8 {
        let mut phase = B::one();
        let mut sum = F::zero();
        for &v in &conjugates {
            sum += v.mul_prime(phase);
            phase *= omega_step;
        }
        let lane = (sum * inverse_basis[j]).mul_prime(inverse_eight);
        claim += low[j] * lane;
        omega_step *= omega_inverse;
    }
    claim
}

fn equality_table(point: &[F]) -> Vec<F> {
    let mut table = vec![F::one()];
    for &coordinate in point {
        let previous = table.len();
        // With MSB-first points each new coordinate is the least-significant bit.
        let mut next = vec![F::zero(); previous * 2];
        let fill = |(i, out): (usize, &mut [F])| {
            out[1] = table[i] * coordinate;
            out[0] = table[i] - out[1];
        };
        #[cfg(feature = "parallel")]
        if previous >= PARALLEL_THRESHOLD {
            next.par_chunks_exact_mut(2).enumerate().for_each(fill);
            table = next;
            continue;
        }
        next.chunks_exact_mut(2).enumerate().for_each(fill);
        table = next;
    }
    table
}

fn project_lanes(values: &[F], weights: &[F]) -> [F; 8] {
    debug_assert_eq!(values.len(), weights.len());
    let accumulate = |(values, weights): (&[F], &[F])| {
        let mut raw = [0; 64];
        for (value, weight) in values.iter().zip(weights) {
            accumulate_raw_lanes(&mut raw, value, weight);
        }
        raw
    };
    #[cfg(feature = "parallel")]
    if values.len() >= PARALLEL_THRESHOLD {
        let raw = values
            .par_chunks(CACHE_TILE)
            .zip(weights.par_chunks(CACHE_TILE))
            .map(accumulate)
            .reduce(|| [0; 64], merge_raw_lanes);
        return finish_raw_lanes(raw);
    }
    finish_raw_lanes(accumulate((values, weights)))
}

#[inline]
fn accumulate_raw_lanes(accumulator: &mut RawLanes, value: &F, weight: &F) {
    for (j, coefficient) in value.coefficients().iter().enumerate() {
        let coefficient = coefficient.to_canonical_u64();
        for (target, weight) in accumulator[j * 8..j * 8 + 8]
            .iter_mut()
            .zip(weight.coefficients())
        {
            *target += coefficient * weight.to_canonical_u64();
        }
    }
}

fn merge_raw_lanes(mut left: RawLanes, right: RawLanes) -> RawLanes {
    for (left, right) in left.iter_mut().zip(right) {
        *left += right;
    }
    left
}

fn finish_raw_lanes(raw: RawLanes) -> [F; 8] {
    std::array::from_fn(|j| {
        F::from_coefficients(std::array::from_fn(|k| B::from_u64(raw[8 * j + k])))
    })
}

/// Generate the large equality table once and consume each freshly produced
/// weight while it is in cache. The high coordinates remain the most significant.
fn project_with_equality_weights(values: &[F], point: &[F]) -> (Vec<F>, [F; 8]) {
    debug_assert_eq!(values.len(), 1usize << point.len());
    if point.len() <= CACHE_TILE_LOG {
        let weights = equality_table(point);
        let lanes = project_lanes(values, &weights);
        return (weights, lanes);
    }
    let split = point.len() - CACHE_TILE_LOG;
    let high = equality_table(&point[..split]);
    let low = equality_table(&point[split..]);
    let mut weights = vec![F::zero(); values.len()];
    let fill = |((weights, values), &high): ((&mut [F], &[F]), &F)| {
        let mut raw = [0; 64];
        for ((weight, value), &low) in weights.iter_mut().zip(values).zip(&low) {
            *weight = high * low;
            accumulate_raw_lanes(&mut raw, value, weight);
        }
        raw
    };
    #[cfg(feature = "parallel")]
    let raw = weights
        .par_chunks_mut(CACHE_TILE)
        .zip(values.par_chunks(CACHE_TILE))
        .zip(high.par_iter())
        .map(fill)
        .reduce(|| [0; 64], merge_raw_lanes);
    #[cfg(not(feature = "parallel"))]
    let raw = weights
        .chunks_mut(CACHE_TILE)
        .zip(values.chunks(CACHE_TILE))
        .zip(high.iter())
        .map(fill)
        .fold([0; 64], merge_raw_lanes);
    (weights, finish_raw_lanes(raw))
}

/// Apply the same F_p-linear map as sum_j image_j * coefficient_j, reducing
/// once per output coefficient. Eight products sum to less than 2^35.
fn apply_weight_images(weights: &mut [F], images: &[F; 8]) {
    let matrix: [[u64; 8]; 8] = std::array::from_fn(|k| {
        std::array::from_fn(|j| images[j].coefficients()[k].to_canonical_u64())
    });
    let convert = |value: &mut F| {
        let input: [u64; 8] = std::array::from_fn(|j| value.coefficients()[j].to_canonical_u64());
        *value = F::from_coefficients(std::array::from_fn(|k| {
            B::from_u64(matrix[k].iter().zip(input).map(|(&a, b)| a * b).sum())
        }));
    };
    #[cfg(feature = "parallel")]
    if weights.len() >= PARALLEL_THRESHOLD {
        weights.par_iter_mut().for_each(convert);
        return;
    }
    weights.iter_mut().for_each(convert);
}

fn round_message(left: &[F], right: &[F]) -> [F; 2] {
    let half = left.len() / 2;
    let accumulate = |mut sum: [F; 2], i: usize| {
        sum[0] = left[i].mul_add(right[i], sum[0]);
        sum[1] = (left[i + half] - left[i]).mul_add(right[i + half] - right[i], sum[1]);
        sum
    };
    #[cfg(feature = "parallel")]
    if half >= PARALLEL_THRESHOLD {
        return (0..half)
            .into_par_iter()
            .fold(|| [F::zero(); 2], accumulate)
            .reduce(|| [F::zero(); 2], |a, b| [a[0] + b[0], a[1] + b[1]]);
    }
    (0..half).fold([F::zero(); 2], accumulate)
}

fn round_evaluation(claim: F, message: [F; 2], challenge: F) -> F {
    let linear = claim - message[0] - message[0] - message[1];
    message[0] + challenge * (linear + challenge * message[1])
}

fn fold_pair(left: &mut [F], right: &mut [F], challenge: F) {
    let half = left.len() / 2;
    let (lo_l, hi_l) = left.split_at_mut(half);
    let (lo_r, hi_r) = right.split_at_mut(half);
    let fold = |(((a, b), &c), &d): (((&mut F, &mut F), &F), &F)| {
        *a = challenge.mul_add(c - *a, *a);
        *b = challenge.mul_add(d - *b, *b);
    };
    #[cfg(feature = "parallel")]
    if half >= PARALLEL_THRESHOLD {
        lo_l.par_iter_mut()
            .zip(lo_r)
            .zip(hi_l.par_iter())
            .zip(hi_r.par_iter())
            .for_each(fold);
        return;
    }
    lo_l.iter_mut()
        .zip(lo_r)
        .zip(hi_l.iter())
        .zip(hi_r.iter())
        .for_each(fold);
}

fn batch_weight(point: &[F], reduced: &[F], powers: &[F; 8]) -> F {
    (0..8).fold(F::zero(), |sum, i| {
        let weight = point.iter().zip(reduced).fold(F::one(), |v, (&p, &r)| {
            let s = p.frobenius((8 - i) % 8);
            v * ((F::one() - s) * (F::one() - r) + s * r)
        });
        sum + powers[i] * weight
    })
}

fn mobius_in_place(values: &mut [F]) {
    // Keep the low ten independent stages in a 16 KiB tile instead of scanning
    // the entire witness ten times. Higher stages retain their original order.
    let low_levels = values.len().ilog2().min(CACHE_TILE_LOG as u32);
    let tile = 1usize << low_levels;
    let transform_tile = |chunk: &mut [F]| {
        for level in 0..low_levels {
            let half = 1usize << level;
            for block in chunk.chunks_exact_mut(half * 2) {
                let (low, high) = block.split_at_mut(half);
                for (target, &source) in high.iter_mut().zip(low.iter()) {
                    *target -= source;
                }
            }
        }
    };
    #[cfg(feature = "parallel")]
    if values.len() >= PARALLEL_THRESHOLD {
        values.par_chunks_exact_mut(tile).for_each(transform_tile);
    } else {
        values.chunks_exact_mut(tile).for_each(transform_tile);
    }
    #[cfg(not(feature = "parallel"))]
    values.chunks_exact_mut(tile).for_each(transform_tile);
    for level in low_levels..values.len().ilog2() {
        let half = 1usize << level;
        let transform = |chunk: &mut [F]| {
            let (low, high) = chunk.split_at_mut(half);
            #[cfg(feature = "parallel")]
            if half >= 1 << 14 {
                high.par_iter_mut()
                    .zip(low.par_iter())
                    .for_each(|(target, &source)| *target -= source);
                return;
            }
            for (target, &source) in high.iter_mut().zip(low.iter()) {
                *target -= source;
            }
        };
        #[cfg(feature = "parallel")]
        values.par_chunks_exact_mut(half * 2).for_each(transform);
        #[cfg(not(feature = "parallel"))]
        values.chunks_exact_mut(half * 2).for_each(transform);
    }
}

struct Transcript(Blake2s256);
impl Transcript {
    fn bytes(&mut self, bytes: &[u8]) {
        self.0.update((bytes.len() as u64).to_le_bytes());
        self.0.update(bytes);
    }
    fn field(&mut self, value: F) -> Result<(), Error> {
        let mut bytes = [0; 16];
        value.write_canonical_bytes(&mut bytes)?;
        self.bytes(&bytes);
        Ok(())
    }
    fn challenge(&mut self) -> F {
        for counter in 0u64.. {
            let mut hasher = self.0.clone();
            hasher.update(b"challenge");
            hasher.update(counter.to_le_bytes());
            let bytes = hasher.finalize();
            if let Some(value) = F::from_uniform_bytes(&bytes[..16]) {
                self.bytes(&bytes);
                return value;
            }
        }
        unreachable!()
    }
}

fn put_field(out: &mut Vec<u8>, value: F) -> Result<(), Error> {
    let mut bytes = [0; 16];
    value.write_canonical_bytes(&mut bytes)?;
    out.extend_from_slice(&bytes);
    Ok(())
}

struct Reader<'a>(&'a [u8]);
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], Error> {
        if n > self.0.len() {
            return Err(Error::EncodingError);
        }
        let (head, tail) = self.0.split_at(n);
        self.0 = tail;
        Ok(head)
    }
    fn field(&mut self) -> Result<F, Error> {
        F::from_canonical_bytes(self.take(16)?).ok_or(Error::EncodingError)
    }
    fn length(&mut self) -> Result<usize, Error> {
        usize::try_from(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
            .map_err(|_| Error::EncodingError)
    }
    fn part(&mut self) -> Result<&'a [u8], Error> {
        let len = self.length()?;
        self.take(len)
    }
}

#[cfg(test)]
mod kernel_tests {
    use super::*;

    fn random_values(len: usize, rng: &mut StdRng) -> Vec<F> {
        (0..len).map(|_| F::sample_uniform(rng)).collect()
    }

    fn reference_project(values: &[F], weights: &[F]) -> [F; 8] {
        let mut lanes = [F::zero(); 8];
        for (value, &weight) in values.iter().zip(weights) {
            for (lane, &coefficient) in lanes.iter_mut().zip(value.coefficients()) {
                *lane += weight.mul_prime(coefficient);
            }
        }
        lanes
    }

    #[test]
    fn delayed_projection_matches_field_arithmetic_and_maximum_bound() {
        let mut rng = StdRng::seed_from_u64(0x7061636b5f726177);
        for len in [0, 1, CACHE_TILE - 1, CACHE_TILE, PARALLEL_THRESHOLD] {
            let values = random_values(len, &mut rng);
            let weights = random_values(len, &mut rng);
            assert_eq!(
                project_lanes(&values, &weights),
                reference_project(&values, &weights)
            );
        }
        let last = B::from_u64(65_520);
        let maximum = F::from_coefficients([last; 8]);
        let mut single = [0; 64];
        accumulate_raw_lanes(&mut single, &maximum, &maximum);
        let half: RawLanes = single.map(|entry| entry * (1u64 << 26));
        let merged = merge_raw_lanes(half, half);
        assert_eq!(merged, [576_179_311_686_451_200; 64]);
        let expected = F::from_coefficients([last * last * B::from_u64(1 << 27); 8]);
        assert_eq!(finish_raw_lanes(merged), [expected; 8]);
    }

    #[test]
    fn fused_equality_projection_matches_unsplit_order() {
        let mut rng = StdRng::seed_from_u64(0x7061636b5f657175);
        for log_n in [
            0,
            1,
            CACHE_TILE_LOG - 1,
            CACHE_TILE_LOG,
            CACHE_TILE_LOG + 1,
            14,
        ] {
            let mut point = random_values(log_n, &mut rng);
            // Include degenerate Boolean coordinates: the optimization must not
            // divide by a point coordinate or by one minus a coordinate.
            if log_n > 1 {
                point[0] = F::one();
                point[log_n - 1] = F::zero();
            }
            let values = random_values(1usize << log_n, &mut rng);
            let expected_weights = equality_table(&point);
            let expected_lanes = reference_project(&values, &expected_weights);
            let (weights, lanes) = project_with_equality_weights(&values, &point);
            assert_eq!(weights, expected_weights, "weights at log size {log_n}");
            assert_eq!(lanes, expected_lanes, "lanes at log size {log_n}");
        }
    }

    #[test]
    fn delayed_weight_map_matches_eight_field_scaled_images() {
        let mut rng = StdRng::seed_from_u64(0x7061636b5f6d6170);
        for len in [0, 1, PARALLEL_THRESHOLD - 1, PARALLEL_THRESHOLD] {
            let images: [F; 8] = std::array::from_fn(|_| F::sample_uniform(&mut rng));
            let mut weights = random_values(len, &mut rng);
            let expected: Vec<F> = weights
                .iter()
                .map(|weight| {
                    images
                        .iter()
                        .zip(weight.coefficients())
                        .fold(F::zero(), |sum, (&image, &c)| sum + image.mul_prime(c))
                })
                .collect();
            apply_weight_images(&mut weights, &images);
            assert_eq!(weights, expected);
        }
        let maximum = F::from_coefficients([B::from_u64(65_520); 8]);
        let mut weights = vec![maximum];
        apply_weight_images(&mut weights, &[maximum; 8]);
        assert_eq!(weights[0], F::from_coefficients([B::from_u64(8); 8]));
    }

    #[test]
    fn cache_blocked_mobius_matches_original_stage_order() {
        let mut rng = StdRng::seed_from_u64(0x7061636b5f6d6f62);
        for log_n in [
            0,
            1,
            CACHE_TILE_LOG - 1,
            CACHE_TILE_LOG,
            CACHE_TILE_LOG + 1,
            14,
        ] {
            let mut actual = random_values(1usize << log_n, &mut rng);
            let mut expected = actual.clone();
            for level in 0..log_n {
                let half = 1usize << level;
                for chunk in expected.chunks_exact_mut(half * 2) {
                    let (low, high) = chunk.split_at_mut(half);
                    for (target, &source) in high.iter_mut().zip(low.iter()) {
                        *target -= source;
                    }
                }
            }
            mobius_in_place(&mut actual);
            assert_eq!(actual, expected, "log size {log_n}");
        }
    }

    #[test]
    fn sumcheck_tail_and_parallel_boundary_match_serial_reference() {
        let mut rng = StdRng::seed_from_u64(0x7061636b5f73756d);
        for log_n in [1, 8, 12, 13] {
            let mut left = random_values(1usize << log_n, &mut rng);
            let mut right = random_values(1usize << log_n, &mut rng);
            let half = left.len() / 2;
            let mut expected = [F::zero(); 2];
            for i in 0..half {
                expected[0] += left[i] * right[i];
                expected[1] += (left[i + half] - left[i]) * (right[i + half] - right[i]);
            }
            assert_eq!(round_message(&left, &right), expected);
            let challenge = F::sample_uniform(&mut rng);
            let mut expected_left = left.clone();
            let mut expected_right = right.clone();
            for i in 0..half {
                expected_left[i] = left[i] + challenge * (left[i + half] - left[i]);
                expected_right[i] = right[i] + challenge * (right[i + half] - right[i]);
            }
            fold_pair(&mut left, &mut right, challenge);
            assert_eq!(left, expected_left);
            assert_eq!(right, expected_right);
        }
    }
}
