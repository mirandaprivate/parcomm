//! One Laurent Packed EO over the block axis, then one native opening/Decide.
//! Sampled codeword equations bind the folded root to every original root.
use super::*;
#[cfg(test)]
#[path = "packed_batch_tests.rs"]
mod tests;
const LANES: usize = BATCH_LANES;
const BATCH_MAGIC: &[u8; 8] = b"QAPKBAT1";

pub struct PackedBatchProof {
    pub claimed_value: F,
    pub timings: OpeningTimings,
    pub folding_ms: f64,
    pub binding_ms: f64,
    wire: Vec<u8>,
}
impl PackedBatchProof {
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        Ok(self.wire.clone())
    }
}

// Consume compact input while its packed value is in a register. Accumulator
// bounds are unchanged from project_lanes; no full-block witness is allocated.
fn project_lanes_from_fn<T: Fn(usize) -> F + Sync>(weights: &[F], source: T) -> [F; LANES] {
    let accumulate = |(tile, weights): (usize, &[F])| {
        let mut raw: RawLanes = [0; LANES * LANES];
        for (offset, weight) in weights.iter().enumerate() {
            accumulate_raw_lanes(&mut raw, &source(tile * CACHE_TILE + offset), weight);
        }
        raw
    };
    #[cfg(feature = "parallel")]
    if weights.len() >= PARALLEL_THRESHOLD {
        let raw = weights.par_chunks(CACHE_TILE).enumerate().map(accumulate)
            .reduce(|| [0; LANES * LANES], merge_raw_lanes);
        return finish_raw_lanes(raw);
    }
    finish_raw_lanes(weights.chunks(CACHE_TILE).enumerate().map(accumulate)
        .fold([0; LANES * LANES], merge_raw_lanes))
}

fn consecutive_powers(value: F, count: usize) -> Vec<F> {
    std::iter::successors(Some(F::one()), |power| Some(*power * value))
        .take(count)
        .collect()
}
fn conjugates(lanes: &[F; LANES]) -> [F; LANES] {
    let basis = basis();
    std::array::from_fn(|image| {
        (0..LANES).fold(F::zero(), |sum, lane| {
            sum + basis[lane] * lanes[lane].frobenius((LANES - image) % LANES)
        })
    })
}
fn nonzero_challenge(transcript: &mut Transcript) -> F {
    loop {
        let value = transcript.challenge();
        if value != F::zero() {
            return value;
        }
    }
}
fn absorb_cross(transcript: &mut Transcript, positive: &[F], negative: &[F]) -> Result<(), Error> {
    transcript.bytes(b"block-packed-eo-cross");
    for &value in positive.iter().chain(negative) {
        transcript.field(value)?;
    }
    Ok(())
}
fn update_eo(mut claim: F, positive: &[F], negative: &[F], challenge: F) -> F {
    let inverse = challenge.try_inverse().unwrap();
    let mut forward = challenge;
    let mut backward = inverse;
    for (&positive, &negative) in positive.iter().zip(negative) {
        claim += positive * forward + negative * backward;
        forward *= challenge;
        backward *= inverse;
    }
    claim
}
fn folded_coefficients(high: &[F], powers: &[F; LANES], challenge: F) -> [F; LANES] {
    let forward = consecutive_powers(challenge, high.len());
    std::array::from_fn(|image| {
        powers[image]
            * high
                .iter()
                .zip(&forward)
                .fold(F::zero(), |sum, (&weight, &power)| {
                    sum + power * weight.frobenius((LANES - image) % LANES)
                })
    })
}
fn binding_seed(transcript: &mut Transcript, final_value: F) -> Result<[u8; 32], Error> {
    transcript.bytes(b"authenticated-codeword-fold-queries/v1");
    transcript.field(final_value)?;
    Ok(transcript.0.clone().finalize().into())
}
fn put_part(bytes: &mut Vec<u8>, part: &[u8]) {
    bytes.extend_from_slice(&(part.len() as u64).to_le_bytes());
    bytes.extend_from_slice(part);
}

impl PackedPcs {
    fn batch_transcript(
        &self,
        roots: &[ShortQaFoldCommitment],
        point: &[F],
        value: F,
        context: &[u8],
    ) -> Result<Transcript, Error> {
        let mut transcript = self.transcript(&roots[0], point, value, context)?;
        transcript.bytes(b"one-opening-block-packed-eo/v1");
        transcript.bytes(&(roots.len() as u64).to_le_bytes());
        for root in roots {
            transcript.bytes(&root.root);
        }
        Ok(transcript)
    }

    /// Opens the concatenation of 2..128 equally sized base tables. The source is
    /// indexed by (block, offset); original roots remain persistent commitments.
    pub fn open_batch_from_fn<T: Fn(usize, usize) -> B + Sync>(
        &self, mut states: Vec<PackedData>, point: &[F], context: &[u8], source: T,
    ) -> Result<PackedBatchProof, Error> {
        if states.iter().any(|state| state.profile != self.pp.parameter_digest) {
            return Err(Error::InvalidCommitment);
        }
        let roots = states.iter().map(PackedData::commitment).collect();
        for state in &mut states { state.discard_values(); }
        self.open_batch_impl(roots, Some(states), point, context, source)
    }
    /// Same proof and transcript as the cached path; retain at most one original cache.
    pub fn open_batch_recomputed_from_fn<T: Fn(usize, usize) -> B + Sync>(
        &self, roots: Vec<ShortQaFoldCommitment>, point: &[F], context: &[u8], source: T,
    ) -> Result<PackedBatchProof, Error> {
        self.open_batch_impl(roots, None, point, context, source)
    }
    fn open_batch_impl<T: Fn(usize, usize) -> B + Sync>(
        &self, roots: Vec<ShortQaFoldCommitment>, states: Option<Vec<PackedData>>,
        point: &[F], context: &[u8], source: T,
    ) -> Result<PackedBatchProof, Error> {
        let blocks = roots.len();
        if !(2..=128).contains(&blocks) || !blocks.is_power_of_two()
            || point.len() != self.log_n + blocks.ilog2() as usize {
            return Err(Error::InvalidParameters("batched opening dimensions/profile".into()));
        }
        let start = Instant::now();
        let split = blocks.ilog2() as usize;
        let packing_log = LANES.ilog2() as usize;
        let outer = &point[split..point.len() - packing_log];
        let low = equality_table(&point[point.len() - packing_log..]);
        let high = equality_table(&point[..split]);
        let mut weights = equality_table(outer);
        let packed = |block: usize, index: usize| {
            F::from_coefficients(std::array::from_fn(|lane| {
                source(block, index * LANES + lane)
            }))
        };
        let mut evaluations_by_block = Vec::with_capacity(blocks);
        for block in 0..blocks {
            evaluations_by_block.push(conjugates(&project_lanes_from_fn(&weights, |index| packed(block, index))));
        }
        let evaluations: [F; LANES] = std::array::from_fn(|image| {
            (0..blocks).fold(F::zero(), |sum, block| {
                sum + high[block].frobenius((LANES - image) % LANES)
                    * evaluations_by_block[block][image]
            })
        });
        let claimed_value = projected_claim(&evaluations, &low);
        let mut transcript = self.batch_transcript(&roots, point, claimed_value, context)?;
        for &value in &evaluations {
            transcript.field(value)?;
        }
        let powers_vec = consecutive_powers(transcript.challenge(), LANES);
        let powers: [F; LANES] = powers_vec.try_into().unwrap();
        let mut claim = (0..LANES).fold(F::zero(), |sum, image| {
            sum + powers[image] * evaluations[image]
        });
        let cross = |witness: usize, query: usize| {
            (0..LANES).fold(F::zero(), |sum, image| {
                sum + powers[image]
                    * high[query].frobenius((LANES - image) % LANES)
                    * evaluations_by_block[witness][image]
            })
        };
        let positive = (1..blocks)
            .map(|distance| {
                (0..blocks - distance)
                    .fold(F::zero(), |sum, block| sum + cross(block, block + distance))
            })
            .collect::<Vec<_>>();
        let negative = (1..blocks)
            .map(|distance| {
                (0..blocks - distance)
                    .fold(F::zero(), |sum, block| sum + cross(block + distance, block))
            })
            .collect::<Vec<_>>();
        absorb_cross(&mut transcript, &positive, &negative)?;
        let challenge = nonzero_challenge(&mut transcript);
        claim = update_eo(claim, &positive, &negative, challenge);
        let fold_weights = consecutive_powers(challenge.try_inverse().unwrap(), blocks);
        let coefficients = folded_coefficients(&high, &powers, challenge);
        let basis = basis();
        let images = std::array::from_fn(|lane| {
            (0..LANES).fold(F::zero(), |sum, image| {
                sum + coefficients[image] * basis[lane].frobenius((LANES - image) % LANES)
            })
        });
        apply_weight_images(&mut weights, &images);
        let projection_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        // Block-major traversal preserves the sum order but reads compact input contiguously.
        let mut values = vec![F::zero(); self.packed_len()];
        let fold_tile = |(tile, output): (usize, &mut [F])| {
            let mut input = vec![F::zero(); output.len()];
            let mut coefficients = vec![F::zero(); output.len()];
            for block in 0..blocks {
                for (offset, value) in input.iter_mut().enumerate() { *value = packed(block, tile * CACHE_TILE + offset); }
                coefficients.fill(fold_weights[block]);
                F::mul_assign_slice(&mut input, &coefficients);
                for (value, &product) in output.iter_mut().zip(&input) { *value += product; }
            }
        };
        #[cfg(feature = "parallel")]
        values.par_chunks_mut(CACHE_TILE).enumerate().for_each(fold_tile);
        #[cfg(not(feature = "parallel"))]
        values.chunks_mut(CACHE_TILE).enumerate().for_each(fold_tile);
        let caches = states.as_ref().map(|states| states.iter().map(|state| &state.cache).collect::<Vec<_>>());
        let (folded_root, folded_cache) = if let Some(caches) = &caches {
            self.pc.fold_codewords(&self.pp, caches, &fold_weights)?
        } else {
            // Mobius and QA encoding are linear: encode the folded table once.
            let mut coefficients = values.clone();
            mobius_in_place(&mut coefficients);
            self.pc.commit_owned(&self.pp, coefficients)?
        };
        transcript.bytes(&folded_root.root);
        let folding_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let mut rounds = Vec::with_capacity(outer.len());
        let mut reduced_point = Vec::with_capacity(outer.len());
        while values.len() > 1 {
            let half = values.len() / 2;
            let message = round_message(&values, &weights);
            transcript.field(message[0])?;
            transcript.field(message[1])?;
            let coordinate = transcript.challenge();
            claim = round_evaluation(claim, message, coordinate);
            fold_pair(&mut values, &mut weights, coordinate);
            values.truncate(half);
            weights.truncate(half);
            rounds.push(message);
            reduced_point.push(coordinate);
        }
        let final_value = values[0];
        let final_weight = batch_weight(outer, &reduced_point, &coefficients);
        if weights[0] != final_weight || claim != final_value * final_weight {
            return Err(Error::InvalidCommitment);
        }
        drop(values);
        drop(weights);
        let sumcheck_ms = start.elapsed().as_secs_f64() * 1000.0;
        let seed = binding_seed(&mut transcript, final_value)?;
        let start = Instant::now();
        let binding = if let Some(caches) = &caches {
            self.pc.prove_codeword_fold(&self.pp, caches, &folded_cache, seed)?
        } else {
            // Seed is derived only after the folded root and local sumcheck.
            self.pc.prove_codeword_fold_recomputed(&self.pp, &roots, &folded_cache, seed, |block, workspace| {
                Ok(self.commit_from_fn_reusing(|index| source(block, index), workspace)?.cache)
            })?
        };
        drop(caches);
        drop(states);
        let binding_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let (actual, opening, reduction) =
            self.pc
                .open_with_reduction(&self.pp, &folded_root, &reduced_point, folded_cache)?;
        if actual != final_value || !reduction.ok {
            return Err(Error::InvalidCommitment);
        }
        let native_opening_ms = start.elapsed().as_secs_f64() * 1000.0;
        let start = Instant::now();
        let decide = self.pc.prove_decide(&self.pp, &self.setup, &reduction)?;
        let decide_ms = start.elapsed().as_secs_f64() * 1000.0;
        let mut wire = BATCH_MAGIC.to_vec();
        put_field(&mut wire, claimed_value)?;
        for &value in &evaluations {
            put_field(&mut wire, value)?;
        }
        for &value in positive.iter().chain(&negative) {
            put_field(&mut wire, value)?;
        }
        wire.extend_from_slice(&folded_root.root);
        for message in rounds {
            for value in message {
                put_field(&mut wire, value)?;
            }
        }
        put_field(&mut wire, final_value)?;
        put_part(&mut wire, &binding);
        put_part(&mut wire, &opening.to_bytes()?);
        put_part(&mut wire, &decide.to_bytes()?);
        if wire.len() > MAX_PROOF_BYTES {
            return Err(Error::EncodingError);
        }
        Ok(PackedBatchProof {
            claimed_value,
            folding_ms,
            binding_ms,
            wire,
            timings: OpeningTimings {
                reconstruction_ms: 0.0,
                projection_ms,
                sumcheck_ms,
                native_opening_ms,
                decide_ms,
            },
        })
    }

    pub fn verify_batch(
        &self,
        roots: &[ShortQaFoldCommitment],
        point: &[F],
        claimed_value: F,
        bytes: &[u8],
        context: &[u8],
    ) -> Result<(), Error> {
        let blocks = roots.len();
        if !(2..=128).contains(&blocks)
            || !blocks.is_power_of_two()
            || point.len() != self.log_n + blocks.ilog2() as usize
            || bytes.len() > MAX_PROOF_BYTES
        {
            return Err(Error::InvalidCommitment);
        }
        let split = blocks.ilog2() as usize;
        let packing_log = LANES.ilog2() as usize;
        let outer = &point[split..point.len() - packing_log];
        let low = equality_table(&point[point.len() - packing_log..]);
        let high = equality_table(&point[..split]);
        let mut reader = Reader(bytes);
        if reader.take(8)? != BATCH_MAGIC || reader.field()? != claimed_value {
            return Err(Error::InvalidCommitment);
        }
        let mut evaluations = [F::zero(); LANES];
        for value in &mut evaluations {
            *value = reader.field()?;
        }
        if projected_claim(&evaluations, &low) != claimed_value {
            return Err(Error::InvalidCommitment);
        }
        let mut transcript = self.batch_transcript(roots, point, claimed_value, context)?;
        for &value in &evaluations {
            transcript.field(value)?;
        }
        let powers: [F; LANES] = consecutive_powers(transcript.challenge(), LANES)
            .try_into()
            .unwrap();
        let mut claim = (0..LANES).fold(F::zero(), |sum, image| {
            sum + powers[image] * evaluations[image]
        });
        let positive = (1..blocks)
            .map(|_| reader.field())
            .collect::<Result<Vec<_>, _>>()?;
        let negative = (1..blocks)
            .map(|_| reader.field())
            .collect::<Result<Vec<_>, _>>()?;
        absorb_cross(&mut transcript, &positive, &negative)?;
        let challenge = nonzero_challenge(&mut transcript);
        claim = update_eo(claim, &positive, &negative, challenge);
        let fold_weights = consecutive_powers(challenge.try_inverse().unwrap(), blocks);
        let coefficients = folded_coefficients(&high, &powers, challenge);
        let folded_root = ShortQaFoldCommitment {
            root: reader.take(32)?.try_into().unwrap(),
        };
        transcript.bytes(&folded_root.root);
        let mut reduced_point = Vec::with_capacity(outer.len());
        for _ in 0..outer.len() {
            let message = [reader.field()?, reader.field()?];
            transcript.field(message[0])?;
            transcript.field(message[1])?;
            let coordinate = transcript.challenge();
            claim = round_evaluation(claim, message, coordinate);
            reduced_point.push(coordinate);
        }
        let final_value = reader.field()?;
        if claim != final_value * batch_weight(outer, &reduced_point, &coefficients) {
            return Err(Error::InvalidCommitment);
        }
        let seed = binding_seed(&mut transcript, final_value)?;
        if !self.pc.verify_codeword_fold(
            &self.pp,
            roots,
            &fold_weights,
            &folded_root,
            seed,
            reader.part()?,
        )? {
            return Err(Error::InvalidCommitment);
        }
        let opening = ShortQaFoldOpeningProof::<F>::from_bytes(reader.part()?)?;
        let decide = ShortQaDecideProof::<F>::from_bytes(reader.part()?)?;
        if !reader.0.is_empty() {
            return Err(Error::InvalidCommitment);
        }
        let reduction = self.pc.verify_opening_reduction(
            &self.pp,
            &folded_root,
            &reduced_point,
            final_value,
            &opening,
        )?;
        if !reduction.ok
            || !self
                .pc
                .verify_decide(&self.pp, &self.setup, &reduction, &decide)?
        {
            return Err(Error::InvalidCommitment);
        }
        Ok(())
    }
}
