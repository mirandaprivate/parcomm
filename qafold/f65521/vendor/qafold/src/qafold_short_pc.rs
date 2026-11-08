//! Authenticated QAFold openings for short-prime-field witnesses.
//!
//! The top codeword is stored over the short base field. The first Freivalds
//! fold packs the base-field columns into the extension field, and every lower
//! recursive round operates over that extension field. Use the opening
//! reduction together with `prove_decide` / `verify_decide` to authenticate
//! every deferred generator claim against preprocessed setup commitments.
//! The separate `verify` API performs direct public-generator checks.

use core::marker::PhantomData;

use crate::Error;
use blake2::Blake2s256;
use digest::Digest;
use rand::RngCore;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use crate::qafold::{QAFOLD_RECURSIVE_INDEX, QAFOLD_TOP_INDEX};
use crate::qafold_short::{
    ShortBaseField, ShortExtensionField, ShortFieldElement, ShortQaCode, ShortQaFold,
};

const DIGEST_BYTES: usize = 32;
const PROOF_MAGIC: &[u8; 8] = b"QFSHORT2";
#[path = "qafold_short_batch.rs"]
mod batch;

const TRANSCRIPT_DOMAIN: &[u8] = b"brakefold/qafold-short-pcs/v2";
const LEAF_BASE_DOMAIN: &[u8] = b"qafold-short/base-row/v1";
const LEAF_EXTENSION_DOMAIN: &[u8] = b"qafold-short/extension-row/v1";
const INNER_DOMAIN: &[u8] = b"qafold-short/merkle-inner/v1";
const TERMINAL_VECTOR_DOMAIN: &[u8] = b"qafold-short/terminal-vector/v1";
const TRANSCRIPT_SOUNDNESS_SLACK_BITS: usize = 16;
// Schedule coarse contiguous blocks: the octic products dominate per-item work,
// while small recursive tails should avoid Rayon barriers.
#[cfg(feature = "parallel")]
const OPENING_PARALLEL_GRAIN: usize = 4096;

/// A BLAKE2s Merkle root binding one encoded matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ShortQaFoldCommitment {
    /// Root digest of the row-oriented codeword tree.
    pub root: [u8; DIGEST_BYTES],
}

/// Public parameters for one short-field witness shape.
#[derive(Clone, Debug)]
pub struct ShortQaFoldParameters<B: ShortBaseField> {
    /// Complete QA sequence from the constant-size base to the top code.
    pub sequence: ShortQaFold<B>,
    /// Base-field random-foldable sources used by every recursive code switch.
    derived_parameters: ShortQaDerivedParameters<B>,
    /// Soundness target used to compute every query count.
    pub security_bits: usize,
    /// Digest binding the field, shape, soundness, QA generators, and RFC sources.
    pub parameter_digest: [u8; DIGEST_BYTES],
    validated_parameter_digest: [u8; DIGEST_BYTES],
    validated_generator_digest: [u8; DIGEST_BYTES],
    validated_derived_digest: [u8; DIGEST_BYTES],
    setup_digest: [u8; DIGEST_BYTES],
    validated_setup_digest: [u8; DIGEST_BYTES],
}

impl<B: ShortBaseField> ShortQaFoldParameters<B> {
    /// Returns the immutable base-field RFC sources used by recursive switches.
    pub fn derived_parameters(&self) -> &ShortQaDerivedParameters<B> {
        &self.derived_parameters
    }

    /// Returns the digest binding all ordered setup roots and shapes.
    pub fn setup_digest(&self) -> [u8; DIGEST_BYTES] {
        self.setup_digest
    }
}

/// Prover state retained after committing to the top codeword.
pub struct ShortQaFoldCommitmentCache<B: ShortBaseField> {
    encoded_entries: Vec<B>,
    n_ext_rows: usize,
    n_cols: usize,
    tree: Blake2sMerkleTree,
    // Allocation, QA encoding, witness release, and Merkle construction (ms).
    pub(crate) commit_timings: [f64; 4],
    #[cfg(feature = "profile-encoding")]
    pub(crate) encoding_columns: Vec<QaEncodingColumnProfile>,
}

/// One reusable original-block workspace; it never retains multiple codewords.
#[derive(Default)]
pub struct ShortQaFoldCommitWorkspace<B: ShortBaseField> {
    pub(crate) coefficients: Vec<B>,
    cache: Option<ShortQaFoldCommitmentCache<B>>,
}
impl<B: ShortBaseField> ShortQaFoldCommitWorkspace<B> {
    pub(crate) fn recycle(&mut self, cache: ShortQaFoldCommitmentCache<B>) {
        debug_assert!(self.cache.is_none());
        self.cache = Some(cache);
    }
}

/// One top-level encoding task. Offsets share the outer encoding clock;
/// overlapping column durations must not be added as end-to-end wall time.
#[cfg(feature = "profile-encoding")]
#[derive(Clone, Debug, Default)]
pub struct QaEncodingColumnProfile {
    pub start_ms: f64,
    pub end_ms: f64,
    pub stages: crate::qafold_short::QaEncodeProfile,
}

/// One BLAKE2s authentication path.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortQaMerklePath {
    /// Sibling digests ordered from the leaf level to the root.
    pub siblings: Vec<[u8; DIGEST_BYTES]>,
}

/// One authenticated top-level row over the short base field.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortQaBaseRowQuery<B: ShortBaseField> {
    /// Complete encoded row containing all base-field columns.
    pub row: Vec<B>,
    /// Authentication path for the row.
    pub path: ShortQaMerklePath,
}

/// One authenticated recursive row over the extension field.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortQaExtensionRowQuery<B: ShortBaseField> {
    /// Complete encoded row containing all extension-field columns.
    pub row: Vec<B::Extension>,
    /// Authentication path for the row.
    pub path: ShortQaMerklePath,
}

/// Authenticated data for the first, base-field round.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortQaBaseRound<B: ShortBaseField> {
    /// Partial Merkle openings or one compact complete codeword.
    pub rows: ShortQaBaseRows<B>,
}

/// Authenticated top-round payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShortQaBaseRows<B: ShortBaseField> {
    /// Independently authenticated sampled rows.
    Partial(Vec<ShortQaBaseRowQuery<B>>),
    /// Complete column-major codeword used when the distance bound queries all rows.
    AllRows {
        /// Flattened column-major encoded matrix.
        encoded_entries: Vec<B>,
        /// Number of encoded rows.
        n_rows: usize,
        /// Number of base-field columns.
        n_cols: usize,
    },
}

/// Authenticated data for one lower extension-field round.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortQaExtensionRound<B: ShortBaseField> {
    /// Merkle root of this recursively committed codeword.
    pub root: [u8; DIGEST_BYTES],
    /// Partial Merkle openings or one compact complete codeword.
    pub rows: ShortQaExtensionRows<B>,
}

/// Authenticated recursive-round payload.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShortQaExtensionRows<B: ShortBaseField> {
    /// Independently authenticated sampled rows.
    Partial(Vec<ShortQaExtensionRowQuery<B>>),
    /// Complete column-major extension codeword for an all-row query.
    AllRows {
        /// Flattened column-major encoded matrix.
        encoded_entries: Vec<B::Extension>,
        /// Number of encoded rows.
        n_rows: usize,
        /// Number of extension-field columns.
        n_cols: usize,
    },
}

/// A compact recursive short-field QAFold opening.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortQaFoldOpeningProof<B: ShortBaseField> {
    /// Prover responses consumed by the Fiat--Shamir reductions.
    pub responses: Vec<B::Extension>,
    /// Authenticated rows from the top base-field codeword.
    pub top_round: ShortQaBaseRound<B>,
    /// Authenticated rows and roots for all remaining rounds.
    pub extension_rounds: Vec<ShortQaExtensionRound<B>>,
    /// Directly revealed terminal folded vector.
    pub base_vector: Vec<B::Extension>,
}

/// One non-base QA generator claim deferred by opening verification.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortQaPendingHClaim<E> {
    /// QA sequence level that produced this generator claim.
    pub level: usize,
    /// Claimed generator-matrix MLE value.
    pub value: E,
    /// MLE point on the encoded-row dimension.
    pub left_point: Vec<E>,
    /// MLE point on the message-row dimension.
    pub right_point: Vec<E>,
}

/// Result of replaying the compact opening before Decide-style H checks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShortQaFoldVerificationReduction<E> {
    /// Whether all transcript, Merkle, folding, and terminal checks accepted.
    pub ok: bool,
    /// Non-base public-generator claims left for direct or setup-committed checks.
    pub pending_h_claims: Vec<ShortQaPendingHClaim<E>>,
}

impl<B: ShortBaseField> ShortQaFoldOpeningProof<B> {
    /// Returns the exact canonical wire length of this proof.
    pub fn encoded_len(&self) -> usize {
        let extension_bytes = B::Extension::canonical_byte_len();
        let responses = 8 + self.responses.len() * extension_bytes;
        let top = encoded_base_rows_len::<B>(&self.top_round.rows);
        let extension_rounds = 8 + self
            .extension_rounds
            .iter()
            .map(|round| DIGEST_BYTES + encoded_extension_rows_len::<B>(&round.rows))
            .sum::<usize>();
        PROOF_MAGIC.len()
            + responses
            + top
            + extension_rounds
            + 8
            + self.base_vector.len() * extension_bytes
    }

    /// Serializes this proof using fixed-width canonical field encodings.
    pub fn to_bytes(&self) -> Result<Vec<u8>, Error> {
        let mut output = Vec::with_capacity(self.encoded_len());
        output.extend_from_slice(PROOF_MAGIC);
        write_len(&mut output, self.responses.len())?;
        for value in &self.responses {
            write_extension::<B>(&mut output, *value)?;
        }
        write_base_rows::<B>(&mut output, &self.top_round.rows)?;
        write_len(&mut output, self.extension_rounds.len())?;
        for round in &self.extension_rounds {
            output.extend_from_slice(&round.root);
            write_extension_rows::<B>(&mut output, &round.rows)?;
        }
        write_len(&mut output, self.base_vector.len())?;
        for value in &self.base_vector {
            write_extension::<B>(&mut output, *value)?;
        }
        if output.len() != self.encoded_len() {
            return Err(Error::EncodingError);
        }
        Ok(output)
    }

    /// Parses one canonical proof and rejects trailing or non-canonical data.
    pub fn from_bytes(input: &[u8]) -> Result<Self, Error> {
        let mut reader = ByteReader::new(input);
        if reader.take(PROOF_MAGIC.len())? != PROOF_MAGIC {
            return Err(Error::InvalidCommitment);
        }
        let responses_len = reader.read_len()?;
        let extension_bytes = B::Extension::canonical_byte_len();
        if responses_len > reader.remaining() / extension_bytes {
            return Err(Error::InvalidCommitment);
        }
        let mut responses = reserve_decode_vec(responses_len)?;
        for _ in 0..responses_len {
            responses.push(reader.read_extension::<B>()?);
        }
        let top_rows = reader.read_base_rows::<B>()?;
        let extension_rounds_len = reader.read_len()?;
        let minimum_round_bytes = DIGEST_BYTES + 1;
        if extension_rounds_len > reader.remaining().saturating_sub(8) / minimum_round_bytes {
            return Err(Error::InvalidCommitment);
        }
        let mut extension_rounds = reserve_decode_vec(extension_rounds_len)?;
        for _ in 0..extension_rounds_len {
            let root: [u8; DIGEST_BYTES] = reader
                .take(DIGEST_BYTES)?
                .try_into()
                .map_err(|_| Error::InvalidCommitment)?;
            let rows = reader.read_extension_rows::<B>()?;
            extension_rounds.push(ShortQaExtensionRound { root, rows });
        }
        let base_len = reader.read_len()?;
        if base_len > reader.remaining() / extension_bytes {
            return Err(Error::InvalidCommitment);
        }
        let mut base_vector = reserve_decode_vec(base_len)?;
        for _ in 0..base_len {
            base_vector.push(reader.read_extension::<B>()?);
        }
        if !reader.is_empty() {
            return Err(Error::InvalidCommitment);
        }
        Ok(Self {
            responses,
            top_round: ShortQaBaseRound { rows: top_rows },
            extension_rounds,
            base_vector,
        })
    }

    /// Returns the authenticated query count for every round, top first.
    pub fn round_query_counts(&self) -> Vec<usize> {
        let mut counts = Vec::with_capacity(1 + self.extension_rounds.len());
        counts.push(base_rows_query_count(&self.top_round.rows));
        counts.extend(
            self.extension_rounds
                .iter()
                .map(|round| extension_rows_query_count(&round.rows)),
        );
        counts
    }
}

/// A self-contained short-prime-field QAFold commitment facade.
pub struct ShortQaFoldPC<B: ShortBaseField> {
    /// Number of base-field elements in the witness.
    pub n_witness: usize,
    /// Number of extension-field columns after the first fold.
    pub extension_columns: usize,
    /// Top-level QA inverse rate, fixed to two.
    pub index: usize,
    /// Inverse rate below the field-selected top levels, either two or four.
    pub recursive_index: usize,
    /// Target soundness in bits.
    pub security_bits: usize,
    _marker: PhantomData<B>,
}

impl<B: ShortBaseField> ShortQaFoldPC<B> {
    /// Constructs a PCS for a fixed witness shape.
    pub fn new(
        n_witness: usize,
        extension_columns: usize,
        index: usize,
        security_bits: usize,
    ) -> Result<Self, Error> {
        Self::new_with_recursive_index(
            n_witness,
            extension_columns,
            index,
            QAFOLD_RECURSIVE_INDEX,
            security_bits,
        )
    }

    /// Constructs a PCS with inverse rate two or four below the field-selected top levels.
    pub fn new_with_recursive_index(
        n_witness: usize,
        extension_columns: usize,
        index: usize,
        recursive_index: usize,
        security_bits: usize,
    ) -> Result<Self, Error> {
        let extension_capacity_bits =
            ((B::CARDINALITY as f64).log2() * B::EXTENSION_DEGREE as f64).floor() as usize;
        let maximum_security_bits =
            extension_capacity_bits.saturating_sub(TRANSCRIPT_SOUNDNESS_SLACK_BITS);
        if n_witness == 0
            || !n_witness.is_power_of_two()
            || extension_columns <= 1
            || !extension_columns.is_power_of_two()
            || index != QAFOLD_TOP_INDEX
            || !matches!(recursive_index, 2 | 4)
            || security_bits == 0
        {
            return Err(Error::EncodingError);
        }
        if security_bits > maximum_security_bits {
            return Err(Error::InvalidParameters(format!(
                "requested {security_bits}-bit soundness exceeds the short-field transcript limit of {maximum_security_bits} bits"
            )));
        }
        let base_columns = extension_columns
            .checked_mul(B::EXTENSION_DEGREE)
            .ok_or(Error::EncodingError)?;
        if n_witness % base_columns != 0 {
            return Err(Error::EncodingError);
        }
        Ok(Self {
            n_witness,
            extension_columns,
            index,
            recursive_index,
            security_bits,
            _marker: PhantomData,
        })
    }

    /// Samples the QA sequence and commits its fixed base-field setup matrices.
    pub fn setup<R: RngCore>(
        &self,
        rng: &mut R,
    ) -> Result<(ShortQaFoldParameters<B>, ShortQaFoldSetupCommitments<B>), Error> {
        let mut seed = [0u8; DIGEST_BYTES];
        rng.fill_bytes(&mut seed);
        let sequence = ShortQaFold::new_with_recursive_index(
            self.n_witness,
            self.extension_columns,
            self.index,
            self.recursive_index,
            seed,
        )?;
        let derived_parameters = setup::sample_short_derived_parameters(&sequence, rng)?;
        let setup_commitments =
            setup::build_short_setup_commitments(&sequence, &derived_parameters)?;
        let validated_derived_digest = derived_parameters.digest();
        let validated_setup_digest = setup_commitments.setup_digest();
        let parameter_digest = parameter_digest::<B>(
            &sequence,
            &validated_derived_digest,
            &validated_setup_digest,
            self.n_witness,
            self.extension_columns,
            self.index,
            self.security_bits,
        )?;
        let validated_generator_digest = sequence.generator_digest();
        Ok((
            ShortQaFoldParameters {
                sequence,
                derived_parameters,
                security_bits: self.security_bits,
                parameter_digest,
                validated_parameter_digest: parameter_digest,
                validated_generator_digest,
                validated_derived_digest,
                setup_digest: validated_setup_digest,
                validated_setup_digest,
            },
            setup_commitments,
        ))
    }

    /// Commits to an owned column-major base-field witness.
    pub fn commit_owned(
        &self,
        pp: &ShortQaFoldParameters<B>,
        witness: Vec<B>,
    ) -> Result<(ShortQaFoldCommitment, ShortQaFoldCommitmentCache<B>), Error> {
        self.commit_owned_inner(pp, witness, None)
    }

    pub(crate) fn commit_owned_reusing(
        &self, pp: &ShortQaFoldParameters<B>, witness: Vec<B>,
        workspace: &mut ShortQaFoldCommitWorkspace<B>,
    ) -> Result<(ShortQaFoldCommitment, ShortQaFoldCommitmentCache<B>), Error> {
        self.commit_owned_inner(pp, witness, Some(workspace))
    }

    fn commit_owned_inner(
        &self, pp: &ShortQaFoldParameters<B>, witness: Vec<B>,
        mut workspace: Option<&mut ShortQaFoldCommitWorkspace<B>>,
    ) -> Result<(ShortQaFoldCommitment, ShortQaFoldCommitmentCache<B>), Error> {
        validate_parameters(self, pp)?;
        if witness.len() != self.n_witness {
            return Err(Error::EncodingError);
        }
        let code = pp.sequence.top_code()?;
        let n_rows = code.message_len();
        let n_ext_rows = code.codeword_len()?;
        let n_cols = pp.sequence.base_columns()?;
        if witness.len() != n_rows.checked_mul(n_cols).ok_or(Error::EncodingError)? {
            return Err(Error::EncodingError);
        }

        let encoded_len = n_ext_rows.checked_mul(n_cols).ok_or(Error::EncodingError)?;
        let stage = std::time::Instant::now();
        let recycled = workspace.as_deref_mut().and_then(|workspace| workspace.cache.take());
        let (mut encoded_entries, recycled_tree) = match recycled {
            Some(cache) => (cache.encoded_entries, Some(cache.tree)),
            None => (Vec::new(), None),
        };
        if encoded_entries.len() != encoded_len {
            encoded_entries = zeroed_field_vec::<B>(encoded_len);
        }
        // Every codeword entry is overwritten by encode_base_into. Reused
        // initialized storage requires neither zeroing nor unsafe set_len.
        let allocation_ms = stage.elapsed().as_secs_f64() * 1000.0;
        let stage = std::time::Instant::now();
        #[cfg(feature = "profile-encoding")]
        let mut encoding_columns = vec![QaEncodingColumnProfile::default(); n_cols];
        #[cfg(all(feature = "profile-encoding", feature = "parallel"))]
        encoded_entries
            .par_chunks_mut(n_ext_rows)
            .zip(witness.par_chunks(n_rows))
            .zip(encoding_columns.par_iter_mut())
            .try_for_each(|((encoded, message), profile)| -> Result<(), Error> {
                profile.start_ms = stage.elapsed().as_secs_f64() * 1000.0;
                profile.stages = code.encode_base_into_profiled(message, encoded)?;
                profile.end_ms = stage.elapsed().as_secs_f64() * 1000.0;
                Ok(())
            })?;
        #[cfg(all(feature = "profile-encoding", not(feature = "parallel")))]
        encoded_entries
            .chunks_mut(n_ext_rows)
            .zip(witness.chunks(n_rows))
            .zip(encoding_columns.iter_mut())
            .try_for_each(|((encoded, message), profile)| -> Result<(), Error> {
                profile.start_ms = stage.elapsed().as_secs_f64() * 1000.0;
                profile.stages = code.encode_base_into_profiled(message, encoded)?;
                profile.end_ms = stage.elapsed().as_secs_f64() * 1000.0;
                Ok(())
            })?;
        #[cfg(all(feature = "parallel", not(feature = "profile-encoding")))]
        encoded_entries
            .par_chunks_mut(n_ext_rows)
            .zip(witness.par_chunks(n_rows))
            .try_for_each(|(encoded, message)| code.encode_base_into(message, encoded))?;
        #[cfg(all(not(feature = "parallel"), not(feature = "profile-encoding")))]
        encoded_entries
            .chunks_mut(n_ext_rows)
            .zip(witness.chunks(n_rows))
            .try_for_each(|(encoded, message)| code.encode_base_into(message, encoded))?;
        let encoding_ms = stage.elapsed().as_secs_f64() * 1000.0;
        let stage = std::time::Instant::now();
        if let Some(workspace) = workspace {
            workspace.coefficients = witness;
        } else {
            drop(witness);
        }
        let release_ms = stage.elapsed().as_secs_f64() * 1000.0;
        let stage = std::time::Instant::now();
        let tree = build_base_tree_reusing::<B>(&encoded_entries, n_ext_rows, n_cols, recycled_tree)?;
        let merkle_ms = stage.elapsed().as_secs_f64() * 1000.0;
        let commitment = ShortQaFoldCommitment { root: tree.root() };
        Ok((
            commitment,
            ShortQaFoldCommitmentCache {
                encoded_entries,
                n_ext_rows,
                n_cols,
                tree,
                commit_timings: [allocation_ms, encoding_ms, release_ms, merkle_ms],
                #[cfg(feature = "profile-encoding")]
                encoding_columns,
            },
        ))
    }

    /// Checks that setup commitments belong to these validated public parameters.
    pub fn validate_setup_commitments(
        &self,
        pp: &ShortQaFoldParameters<B>,
        setup_commitments: &ShortQaFoldSetupCommitments<B>,
    ) -> Result<(), Error> {
        validate_parameters(self, pp)?;
        validate_setup_aggregate(pp, setup_commitments)
    }

    /// Produces a recursive opening and returns its public claimed value.
    pub fn open(
        &self,
        pp: &ShortQaFoldParameters<B>,
        commitment: &ShortQaFoldCommitment,
        point: &[B::Extension],
        cache: ShortQaFoldCommitmentCache<B>,
    ) -> Result<(B::Extension, ShortQaFoldOpeningProof<B>), Error> {
        let (claimed_value, proof, _) = self.open_with_reduction(pp, commitment, point, cache)?;
        Ok((claimed_value, proof))
    }

    /// Produces an opening together with the prover-local reduction consumed by Decide.
    ///
    /// The reduction is not serialized into the opening proof. Returning it here avoids
    /// replaying Merkle verification merely to recover the generator claims that the
    /// opening prover already computed.
    pub fn open_with_reduction(
        &self,
        pp: &ShortQaFoldParameters<B>,
        commitment: &ShortQaFoldCommitment,
        point: &[B::Extension],
        cache: ShortQaFoldCommitmentCache<B>,
    ) -> Result<
        (
            B::Extension,
            ShortQaFoldOpeningProof<B>,
            ShortQaFoldVerificationReduction<B::Extension>,
        ),
        Error,
    > {
        validate_parameters(self, pp)?;
        let expected_point_len = self.n_witness.ilog2() as usize;
        if point.len() != expected_point_len || cache.tree.root() != commitment.root {
            return Err(Error::InvalidNumberOfVariables);
        }
        let top_code = pp.sequence.top_code()?;
        let top_rows = top_code.message_len();
        let top_cols = pp.sequence.base_columns()?;
        if cache.n_ext_rows != top_code.codeword_len()?
            || cache.n_cols != top_cols
            || cache.encoded_entries.len() != cache.n_ext_rows * top_cols
        {
            return Err(Error::EncodingError);
        }
        let col_log = top_cols.ilog2() as usize;
        let col_weights = tensor_weights(&point[..col_log]);
        let row_weights = tensor_weights(&point[col_log..]);
        let opening_vector = base_systematic_row_projection::<B>(
            &cache.encoded_entries,
            cache.n_ext_rows,
            top_rows,
            top_cols,
            &row_weights,
        )?;
        let claimed_value = inner_product(&opening_vector, &col_weights)?;

        let mut transcript = ShortTranscriptWriter::<B>::new();
        transcript.absorb_digest(b"parameters", &pp.parameter_digest);
        transcript.absorb_digest(b"commitment", &commitment.root);
        transcript.absorb_extensions(b"point", point)?;
        transcript.absorb_extension(b"claimed-value", claimed_value)?;

        let levels = pp.sequence.codes().len();
        let top_level = levels.saturating_sub(1);
        let proximity_rf_layers = pp.sequence.proximity_rf_layers();
        let next_code = top_level
            .checked_sub(1)
            .and_then(|level| pp.sequence.codes().get(level));
        let (top_round, mut terminal_vector, top_reduction, mut prepared_round) =
            prove_base_round::<B>(
                top_code,
                next_code,
                self.extension_columns,
                pp.security_bits,
                proximity_rf_layers,
                commitment.root,
                cache,
                point,
                opening_vector,
                row_weights,
                claimed_value,
                &mut transcript,
            )?;
        let mut pending_h_claims = Vec::with_capacity(top_level);
        if top_level == 0 {
            if top_code.evaluate_generator_mle_extension(
                &top_reduction.generator_left_point,
                &top_reduction.generator_right_point,
            )? != top_reduction.generator_value
            {
                return Err(Error::InvalidCommitment);
            }
        } else {
            pending_h_claims.push(top_reduction.pending_claim(top_level));
        }
        let mut next_value = top_reduction.next_value;
        let mut next_point = top_reduction.next_point;

        let mut extension_rounds = Vec::with_capacity(levels.saturating_sub(1));
        for level in (0..levels.saturating_sub(1)).rev() {
            let code = pp.sequence.codes().get(level).ok_or(Error::EncodingError)?;
            let cols = self.extension_columns;
            if terminal_vector.is_some() {
                return Err(Error::EncodingError);
            }
            let (root, round_cache) = prepared_round.take().ok_or(Error::InvalidCommitment)?;
            let next_code = level
                .checked_sub(1)
                .and_then(|next_level| pp.sequence.codes().get(next_level));
            let (round, next_terminal_vector, reduction, next_prepared_round) =
                prove_extension_round::<B>(
                    code,
                    next_code,
                    cols,
                    pp.security_bits,
                    proximity_rf_layers,
                    root,
                    round_cache,
                    &next_point,
                    next_value,
                    &mut transcript,
                )?;
            extension_rounds.push(round);
            if level == 0 {
                if code.evaluate_generator_mle_extension(
                    &reduction.generator_left_point,
                    &reduction.generator_right_point,
                )? != reduction.generator_value
                {
                    return Err(Error::InvalidCommitment);
                }
            } else {
                pending_h_claims.push(reduction.pending_claim(level));
            }
            terminal_vector = next_terminal_vector;
            next_value = reduction.next_value;
            next_point = reduction.next_point;
            prepared_round = next_prepared_round;
        }

        let terminal_vector = terminal_vector.ok_or(Error::InvalidCommitment)?;
        if prepared_round.is_some()
            || terminal_vector.is_empty()
            || !terminal_vector.len().is_power_of_two()
            || next_point.len() != terminal_vector.len().ilog2() as usize
            || mle_eval(&terminal_vector, &next_point) != Some(next_value)
        {
            return Err(Error::InvalidCommitment);
        }
        let proof = ShortQaFoldOpeningProof {
            responses: transcript.responses,
            top_round,
            extension_rounds,
            base_vector: terminal_vector,
        };
        Ok((
            claimed_value,
            proof,
            ShortQaFoldVerificationReduction {
                ok: true,
                pending_h_claims,
            },
        ))
    }

    /// Verifies the opening reduction and defers non-base public-generator claims.
    ///
    /// This scope mirrors `QaFoldPC::verify`: all queried rows, Freivalds folds,
    /// LiteBullet reductions, and the terminal vector are checked, while the
    /// non-base `H_j` claims are returned for a Decide phase or direct checking.
    pub fn verify_opening_reduction(
        &self,
        pp: &ShortQaFoldParameters<B>,
        commitment: &ShortQaFoldCommitment,
        point: &[B::Extension],
        claimed_value: B::Extension,
        proof: &ShortQaFoldOpeningProof<B>,
    ) -> Result<ShortQaFoldVerificationReduction<B::Extension>, Error> {
        let rejected = || ShortQaFoldVerificationReduction {
            ok: false,
            pending_h_claims: Vec::new(),
        };
        if validate_parameters(self, pp).is_err()
            || point.len() != self.n_witness.ilog2() as usize
            || proof.extension_rounds.len() != pp.sequence.codes().len().saturating_sub(1)
        {
            return Ok(rejected());
        }
        let mut transcript = ShortTranscriptReader::<B>::new(&proof.responses);
        transcript.absorb_digest(b"parameters", &pp.parameter_digest);
        transcript.absorb_digest(b"commitment", &commitment.root);
        transcript.absorb_extensions(b"point", point)?;
        transcript.absorb_extension(b"claimed-value", claimed_value)?;

        let top_code = pp.sequence.top_code()?;
        let top_cols = pp.sequence.base_columns()?;
        let top_level = pp.sequence.codes().len().saturating_sub(1);
        let proximity_rf_layers = pp.sequence.proximity_rf_layers();
        let mut pending_h_claims = Vec::with_capacity(top_level);
        let top_next_binding = if top_level == 0 {
            NextLayerBinding::TerminalVector(hash_terminal_vector::<B>(&proof.base_vector)?)
        } else {
            let root = proof
                .extension_rounds
                .first()
                .ok_or(Error::InvalidCommitment)?
                .root;
            NextLayerBinding::ExtensionRoot(root)
        };
        let top_reduction = match verify_base_round::<B>(
            top_code,
            pp.security_bits,
            proximity_rf_layers,
            commitment.root,
            top_cols,
            point,
            claimed_value,
            top_next_binding,
            &proof.top_round,
            &mut transcript,
        )? {
            Some(reduction) => reduction,
            None => return Ok(rejected()),
        };
        if top_level == 0 {
            if top_code.evaluate_generator_mle_extension(
                &top_reduction.generator_left_point,
                &top_reduction.generator_right_point,
            )? != top_reduction.generator_value
            {
                return Ok(rejected());
            }
        } else {
            pending_h_claims.push(top_reduction.pending_claim(top_level));
        }
        let mut expected_value = top_reduction.next_value;
        let mut current_point = top_reduction.next_point;

        let levels = pp.sequence.codes().len();
        for (round_index, level) in (0..levels.saturating_sub(1)).rev().enumerate() {
            let code = pp.sequence.codes().get(level).ok_or(Error::EncodingError)?;
            let proof_round = &proof.extension_rounds[round_index];
            let next_binding = if round_index + 1 < proof.extension_rounds.len() {
                NextLayerBinding::ExtensionRoot(proof.extension_rounds[round_index + 1].root)
            } else {
                NextLayerBinding::TerminalVector(hash_terminal_vector::<B>(&proof.base_vector)?)
            };
            let reduction = verify_extension_round::<B>(
                code,
                pp.security_bits,
                proximity_rf_layers,
                self.extension_columns,
                &current_point,
                expected_value,
                next_binding,
                proof_round,
                &mut transcript,
            )?;
            let Some(reduction) = reduction else {
                return Ok(rejected());
            };
            if level == 0 {
                if code.evaluate_generator_mle_extension(
                    &reduction.generator_left_point,
                    &reduction.generator_right_point,
                )? != reduction.generator_value
                {
                    return Ok(rejected());
                }
            } else {
                pending_h_claims.push(reduction.pending_claim(level));
            }
            expected_value = reduction.next_value;
            current_point = reduction.next_point;
        }

        if proof.base_vector.is_empty()
            || !proof.base_vector.len().is_power_of_two()
            || current_point.len() != proof.base_vector.len().ilog2() as usize
            || mle_eval(&proof.base_vector, &current_point) != Some(expected_value)
        {
            return Ok(rejected());
        }
        Ok(ShortQaFoldVerificationReduction {
            ok: transcript.is_exhausted(),
            pending_h_claims,
        })
    }

    /// Fully verifies an opening by directly discharging every public generator claim.
    pub fn verify(
        &self,
        pp: &ShortQaFoldParameters<B>,
        commitment: &ShortQaFoldCommitment,
        point: &[B::Extension],
        claimed_value: B::Extension,
        proof: &ShortQaFoldOpeningProof<B>,
    ) -> Result<bool, Error> {
        let reduction =
            self.verify_opening_reduction(pp, commitment, point, claimed_value, proof)?;
        if !reduction.ok {
            return Ok(false);
        }
        for claim in &reduction.pending_h_claims {
            let code = pp
                .sequence
                .codes()
                .get(claim.level)
                .ok_or(Error::EncodingError)?;
            if code.evaluate_generator_mle_extension(&claim.left_point, &claim.right_point)?
                != claim.value
            {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

struct ExtensionRoundCache<B: ShortBaseField> {
    encoded_entries: Vec<B::Extension>,
    n_ext_rows: usize,
    n_cols: usize,
    tree: Blake2sMerkleTree,
}

type PreparedExtensionRound<B> = ([u8; DIGEST_BYTES], ExtensionRoundCache<B>);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum NextLayerBinding {
    ExtensionRoot([u8; DIGEST_BYTES]),
    TerminalVector([u8; DIGEST_BYTES]),
}

#[derive(Clone, Debug)]
enum RowSelection {
    All(usize),
    Partial(Vec<usize>),
}

enum RowSelectionIter<'a> {
    All(core::ops::Range<usize>),
    Partial(core::iter::Copied<core::slice::Iter<'a, usize>>),
}

impl Iterator for RowSelectionIter<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::All(rows) => rows.next(),
            Self::Partial(rows) => rows.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::All(rows) => rows.size_hint(),
            Self::Partial(rows) => rows.size_hint(),
        }
    }
}

impl ExactSizeIterator for RowSelectionIter<'_> {}

impl RowSelection {
    fn len(&self) -> usize {
        match self {
            Self::All(len) => *len,
            Self::Partial(rows) => rows.len(),
        }
    }

    fn is_all(&self) -> bool {
        matches!(self, Self::All(_))
    }

    fn partial(&self) -> Option<&[usize]> {
        match self {
            Self::All(_) => None,
            Self::Partial(rows) => Some(rows),
        }
    }

    fn iter(&self) -> RowSelectionIter<'_> {
        match self {
            Self::All(len) => RowSelectionIter::All(0..*len),
            Self::Partial(rows) => RowSelectionIter::Partial(rows.iter().copied()),
        }
    }
}

#[derive(Clone, Debug)]
struct VerifiedRound<E: ShortFieldElement> {
    next_value: E,
    next_point: Vec<E>,
    generator_value: E,
    generator_left_point: Vec<E>,
    generator_right_point: Vec<E>,
}

impl<E: ShortFieldElement> VerifiedRound<E> {
    fn pending_claim(&self, level: usize) -> ShortQaPendingHClaim<E> {
        ShortQaPendingHClaim {
            level,
            value: self.generator_value,
            left_point: self.generator_left_point.clone(),
            right_point: self.generator_right_point.clone(),
        }
    }
}

#[derive(Clone)]
pub(crate) struct Blake2sMerkleTree {
    levels: Vec<Vec<[u8; DIGEST_BYTES]>>,
}

impl Blake2sMerkleTree {
    pub(crate) fn from_leaves(leaves: Vec<[u8; DIGEST_BYTES]>) -> Result<Self, Error> {
        Self::from_leaves_reusing(leaves, Vec::new())
    }

    fn from_leaves_reusing(leaves: Vec<[u8; DIGEST_BYTES]>, mut levels: Vec<Vec<[u8; DIGEST_BYTES]>>) -> Result<Self, Error> {
        if leaves.is_empty() || !leaves.len().is_power_of_two() {
            return Err(Error::EncodingError);
        }
        if levels.is_empty() { levels.push(leaves); } else { levels[0] = leaves; }
        let mut level = 0;
        while levels[level].len() > 1 {
            let reusable = if level + 1 < levels.len() {
                core::mem::take(&mut levels[level + 1])
            } else { Vec::new() };
            let parents = hash_parent_level_reusing(&levels[level], reusable);
            level += 1;
            if level < levels.len() { levels[level] = parents; } else { levels.push(parents); }
        }
        levels.truncate(level + 1);
        Ok(Self { levels })
    }

    pub(crate) fn root(&self) -> [u8; DIGEST_BYTES] {
        self.levels
            .last()
            .and_then(|level| level.first())
            .copied()
            .unwrap_or([0u8; DIGEST_BYTES])
    }

    pub(crate) fn path(&self, mut index: usize) -> Result<ShortQaMerklePath, Error> {
        if index >= self.levels.first().map(Vec::len).unwrap_or_default() {
            return Err(Error::InvalidCommitment);
        }
        let mut siblings = Vec::with_capacity(self.levels.len().saturating_sub(1));
        for level in self.levels.iter().take(self.levels.len().saturating_sub(1)) {
            siblings.push(level[index ^ 1]);
            index >>= 1;
        }
        Ok(ShortQaMerklePath { siblings })
    }
}

fn prepare_next_layer<B: ShortBaseField>(
    next_code: Option<&ShortQaCode<B>>,
    extension_columns: usize,
    folded: &[B::Extension],
) -> Result<(NextLayerBinding, Option<PreparedExtensionRound<B>>), Error> {
    if let Some(code) = next_code {
        let rows = code.message_len();
        if folded.len()
            != rows
                .checked_mul(extension_columns)
                .ok_or(Error::EncodingError)?
        {
            return Err(Error::EncodingError);
        }
        let (root, cache) = commit_extension_matrix::<B>(code, folded, rows, extension_columns)?;
        Ok((NextLayerBinding::ExtensionRoot(root), Some((root, cache))))
    } else {
        Ok((
            NextLayerBinding::TerminalVector(hash_terminal_vector::<B>(folded)?),
            None,
        ))
    }
}

fn absorb_next_layer_writer<B: ShortBaseField>(
    transcript: &mut ShortTranscriptWriter<B>,
    binding: NextLayerBinding,
) {
    match binding {
        NextLayerBinding::ExtensionRoot(root) => {
            transcript.absorb_bytes(b"next-layer-kind", b"extension-root");
            transcript.absorb_digest(b"next-layer-binding", &root);
        }
        NextLayerBinding::TerminalVector(digest) => {
            transcript.absorb_bytes(b"next-layer-kind", b"terminal-vector");
            transcript.absorb_digest(b"next-layer-binding", &digest);
        }
    }
}

fn absorb_next_layer_reader<B: ShortBaseField>(
    transcript: &mut ShortTranscriptReader<B>,
    binding: NextLayerBinding,
) {
    match binding {
        NextLayerBinding::ExtensionRoot(root) => {
            transcript.absorb_bytes(b"next-layer-kind", b"extension-root");
            transcript.absorb_digest(b"next-layer-binding", &root);
        }
        NextLayerBinding::TerminalVector(digest) => {
            transcript.absorb_bytes(b"next-layer-kind", b"terminal-vector");
            transcript.absorb_digest(b"next-layer-binding", &digest);
        }
    }
}

fn prove_base_round<B: ShortBaseField>(
    code: &ShortQaCode<B>,
    next_code: Option<&ShortQaCode<B>>,
    extension_columns: usize,
    security_bits: usize,
    rf_layers: usize,
    root: [u8; DIGEST_BYTES],
    cache: ShortQaFoldCommitmentCache<B>,
    point: &[B::Extension],
    opening_vector: Vec<B::Extension>,
    row_weights: Vec<B::Extension>,
    value: B::Extension,
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<
    (
        ShortQaBaseRound<B>,
        Option<Vec<B::Extension>>,
        VerifiedRound<B::Extension>,
        Option<PreparedExtensionRound<B>>,
    ),
    Error,
> {
    prove_base_round_with_next_layer(
        code,
        security_bits,
        rf_layers,
        root,
        cache,
        point,
        opening_vector,
        row_weights,
        value,
        next_code.is_none(),
        |folded| prepare_next_layer::<B>(next_code, extension_columns, folded),
        transcript,
    )
}

/// Proves the top opening while binding the caller-selected next carry encoding.
/// The resulting witness evaluation is discharged by the consuming carry round.
#[allow(clippy::too_many_arguments)]
fn prove_base_round_with_next_layer<B: ShortBaseField, T>(
    code: &ShortQaCode<B>,
    security_bits: usize,
    rf_layers: usize,
    root: [u8; DIGEST_BYTES],
    cache: ShortQaFoldCommitmentCache<B>,
    point: &[B::Extension],
    opening_vector: Vec<B::Extension>,
    row_weights: Vec<B::Extension>,
    value: B::Extension,
    retain_folded: bool,
    prepare_next: impl FnOnce(&[B::Extension]) -> Result<(NextLayerBinding, Option<T>), Error>,
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<
    (
        ShortQaBaseRound<B>,
        Option<Vec<B::Extension>>,
        VerifiedRound<B::Extension>,
        Option<T>,
    ),
    Error,
> {
    let n_rows = code.message_len();
    let n_cols = cache.n_cols;
    let col_log = n_cols.ilog2() as usize;
    if cache.n_ext_rows != code.codeword_len()?
        || cache.encoded_entries.len() != cache.n_ext_rows * n_cols
        || point.len() != col_log + n_rows.ilog2() as usize
        || opening_vector.len() != n_cols
        || cache.tree.root() != root
    {
        return Err(Error::InvalidNumberOfVariables);
    }
    transcript.absorb_digest(b"round-root", &root);
    for opening in &opening_vector {
        transcript.push_response(*opening)?;
    }
    let freivalds = (0..n_cols)
        .map(|_| transcript.challenge_extension(b"freivalds"))
        .collect::<Result<Vec<_>, _>>()?;
    let freivalds_value = inner_product(&opening_vector, &freivalds)?;
    let folded = base_systematic_fold::<B>(
        &cache.encoded_entries,
        cache.n_ext_rows,
        n_rows,
        n_cols,
        &freivalds,
    )?;
    let (next_binding, prepared_round) = prepare_next(&folded)?;
    absorb_next_layer_writer(transcript, next_binding);
    let query_count = code.num_queries(security_bits, rf_layers, cache.n_ext_rows)?;
    let query_seed = transcript.challenge_digest(b"row-selection");
    let all_rows = query_count == cache.n_ext_rows;
    let selected_rows = if all_rows {
        RowSelection::All(cache.n_ext_rows)
    } else {
        RowSelection::Partial(sample_without_replacement(
            query_count,
            cache.n_ext_rows,
            query_seed,
        )?)
    };
    let (rows, sampled_values) = if all_rows {
        #[cfg(feature = "parallel")]
        let sampled_values = (0..cache.n_ext_rows)
            .into_par_iter()
            .map(|row| {
                mixed_row_inner_product::<B>(
                    &cache.encoded_entries,
                    cache.n_ext_rows,
                    n_cols,
                    row,
                    &freivalds,
                )
            })
            .collect::<Result<Vec<_>, Error>>()?;
        #[cfg(not(feature = "parallel"))]
        let sampled_values = (0..cache.n_ext_rows)
            .map(|row| {
                mixed_row_inner_product::<B>(
                    &cache.encoded_entries,
                    cache.n_ext_rows,
                    n_cols,
                    row,
                    &freivalds,
                )
            })
            .collect::<Result<Vec<_>, Error>>()?;
        (
            ShortQaBaseRows::AllRows {
                encoded_entries: cache.encoded_entries,
                n_rows: cache.n_ext_rows,
                n_cols,
            },
            sampled_values,
        )
    } else {
        #[cfg(feature = "parallel")]
        let queried_rows = {
            let selected_rows = selected_rows.partial().ok_or(Error::EncodingError)?;
            selected_rows
                .par_iter()
                .map(|&row| {
                    Ok(ShortQaBaseRowQuery {
                        row: base_encoded_row(
                            &cache.encoded_entries,
                            cache.n_ext_rows,
                            n_cols,
                            row,
                        ),
                        path: cache.tree.path(row)?,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?
        };
        #[cfg(not(feature = "parallel"))]
        let queried_rows = {
            let selected_rows = selected_rows.partial().ok_or(Error::EncodingError)?;
            selected_rows
                .iter()
                .map(|&row| {
                    Ok(ShortQaBaseRowQuery {
                        row: base_encoded_row(
                            &cache.encoded_entries,
                            cache.n_ext_rows,
                            n_cols,
                            row,
                        ),
                        path: cache.tree.path(row)?,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?
        };
        let sampled_values = queried_rows
            .iter()
            .map(|query| mixed_inner_product::<B>(&query.row, &freivalds))
            .collect::<Result<Vec<_>, _>>()?;
        (ShortQaBaseRows::Partial(queried_rows), sampled_values)
    };
    transcript.absorb_extensions(b"sampled-values", &sampled_values)?;
    let (terminal_vector, reduction) = prove_round_reductions::<B>(
        code,
        point,
        value,
        &freivalds,
        freivalds_value,
        folded,
        row_weights,
        retain_folded,
        &selected_rows,
        &sampled_values,
        transcript,
    )?;
    Ok((
        ShortQaBaseRound { rows },
        terminal_vector,
        reduction,
        prepared_round,
    ))
}

fn prove_extension_round<B: ShortBaseField>(
    code: &ShortQaCode<B>,
    next_code: Option<&ShortQaCode<B>>,
    extension_columns: usize,
    security_bits: usize,
    rf_layers: usize,
    root: [u8; DIGEST_BYTES],
    cache: ExtensionRoundCache<B>,
    point: &[B::Extension],
    value: B::Extension,
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<
    (
        ShortQaExtensionRound<B>,
        Option<Vec<B::Extension>>,
        VerifiedRound<B::Extension>,
        Option<PreparedExtensionRound<B>>,
    ),
    Error,
> {
    let n_rows = code.message_len();
    let n_cols = cache.n_cols;
    let col_log = n_cols.ilog2() as usize;
    if cache.n_ext_rows != code.codeword_len()?
        || cache.encoded_entries.len() != cache.n_ext_rows * n_cols
        || point.len() != col_log + n_rows.ilog2() as usize
        || cache.tree.root() != root
    {
        return Err(Error::InvalidNumberOfVariables);
    }
    let row_weights = tensor_weights(&point[col_log..]);
    let opening_vector = extension_systematic_row_projection::<B>(
        &cache.encoded_entries,
        cache.n_ext_rows,
        n_rows,
        n_cols,
        &row_weights,
    )?;
    let column_weights = tensor_weights(&point[..col_log]);
    if inner_product(&opening_vector, &column_weights)? != value {
        return Err(Error::InvalidCommitment);
    }
    transcript.absorb_digest(b"round-root", &root);
    for opening in &opening_vector {
        transcript.push_response(*opening)?;
    }
    let freivalds = (0..n_cols)
        .map(|_| transcript.challenge_extension(b"freivalds"))
        .collect::<Result<Vec<_>, _>>()?;
    let freivalds_value = inner_product(&opening_vector, &freivalds)?;
    let folded = extension_systematic_fold::<B>(
        &cache.encoded_entries,
        cache.n_ext_rows,
        n_rows,
        n_cols,
        &freivalds,
    )?;
    let retain_folded = next_code.is_none();
    let (next_binding, prepared_round) =
        prepare_next_layer::<B>(next_code, extension_columns, &folded)?;
    absorb_next_layer_writer(transcript, next_binding);
    let query_count = code.num_queries(security_bits, rf_layers, cache.n_ext_rows)?;
    let query_seed = transcript.challenge_digest(b"row-selection");
    let all_rows = query_count == cache.n_ext_rows;
    let selected_rows = if all_rows {
        RowSelection::All(cache.n_ext_rows)
    } else {
        RowSelection::Partial(sample_without_replacement(
            query_count,
            cache.n_ext_rows,
            query_seed,
        )?)
    };
    let (rows, sampled_values) = if all_rows {
        #[cfg(feature = "parallel")]
        let sampled_values = (0..cache.n_ext_rows)
            .into_par_iter()
            .map(|row| {
                extension_row_inner_product::<B>(
                    &cache.encoded_entries,
                    cache.n_ext_rows,
                    n_cols,
                    row,
                    &freivalds,
                )
            })
            .collect::<Result<Vec<_>, Error>>()?;
        #[cfg(not(feature = "parallel"))]
        let sampled_values = (0..cache.n_ext_rows)
            .map(|row| {
                extension_row_inner_product::<B>(
                    &cache.encoded_entries,
                    cache.n_ext_rows,
                    n_cols,
                    row,
                    &freivalds,
                )
            })
            .collect::<Result<Vec<_>, Error>>()?;
        (
            ShortQaExtensionRows::AllRows {
                encoded_entries: cache.encoded_entries,
                n_rows: cache.n_ext_rows,
                n_cols,
            },
            sampled_values,
        )
    } else {
        #[cfg(feature = "parallel")]
        let queried_rows = {
            let selected_rows = selected_rows.partial().ok_or(Error::EncodingError)?;
            selected_rows
                .par_iter()
                .map(|&row| {
                    Ok(ShortQaExtensionRowQuery::<B> {
                        row: extension_encoded_row::<B>(
                            &cache.encoded_entries,
                            cache.n_ext_rows,
                            n_cols,
                            row,
                        ),
                        path: cache.tree.path(row)?,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?
        };
        #[cfg(not(feature = "parallel"))]
        let queried_rows = {
            let selected_rows = selected_rows.partial().ok_or(Error::EncodingError)?;
            selected_rows
                .iter()
                .map(|&row| {
                    Ok(ShortQaExtensionRowQuery::<B> {
                        row: extension_encoded_row::<B>(
                            &cache.encoded_entries,
                            cache.n_ext_rows,
                            n_cols,
                            row,
                        ),
                        path: cache.tree.path(row)?,
                    })
                })
                .collect::<Result<Vec<_>, Error>>()?
        };
        let sampled_values = queried_rows
            .iter()
            .map(|query| inner_product(&query.row, &freivalds))
            .collect::<Result<Vec<_>, _>>()?;
        (ShortQaExtensionRows::Partial(queried_rows), sampled_values)
    };
    transcript.absorb_extensions(b"sampled-values", &sampled_values)?;
    let (terminal_vector, reduction) = prove_round_reductions::<B>(
        code,
        point,
        value,
        &freivalds,
        freivalds_value,
        folded,
        row_weights,
        retain_folded,
        &selected_rows,
        &sampled_values,
        transcript,
    )?;
    Ok((
        ShortQaExtensionRound { root, rows },
        terminal_vector,
        reduction,
        prepared_round,
    ))
}

#[allow(clippy::too_many_arguments)]
fn prove_round_reductions<B: ShortBaseField>(
    code: &ShortQaCode<B>,
    point: &[B::Extension],
    _value: B::Extension,
    freivalds: &[B::Extension],
    freivalds_value: B::Extension,
    folded: Vec<B::Extension>,
    row_weights: Vec<B::Extension>,
    retain_folded: bool,
    selected_rows: &RowSelection,
    sampled_values: &[B::Extension],
    transcript: &mut ShortTranscriptWriter<B>,
) -> Result<(Option<Vec<B::Extension>>, VerifiedRound<B::Extension>), Error> {
    let n_rows = code.message_len();
    let n_ext_rows = code.codeword_len()?;
    let col_log = freivalds.len().ilog2() as usize;
    if freivalds.len() == 0
        || !freivalds.len().is_power_of_two()
        || point.len() != col_log + n_rows.ilog2() as usize
        || folded.len() != n_rows
        || row_weights.len() != n_rows
        || selected_rows.len() != sampled_values.len()
    {
        return Err(Error::InvalidNumberOfVariables);
    }
    let selection_log = selected_rows.len().next_power_of_two().ilog2() as usize;
    let selection_point = (0..selection_log)
        .map(|_| transcript.challenge_extension(b"selection-left"))
        .collect::<Result<Vec<_>, _>>()?;
    let selection_weights = tensor_weights_prefix(&selection_point, selected_rows.len())?;
    let selected_value = inner_product(&selection_weights, sampled_values)?;
    let mut pi_left = vec![B::Extension::zero(); n_ext_rows];
    for (row, &weight) in selected_rows.iter().zip(&selection_weights) {
        let target = pi_left
            .get_mut(row)
            .ok_or(Error::InvalidNumberOfVariables)?;
        *target += weight;
    }
    let mut combined_left = code.generator_left_projection_extension(&pi_left)?;
    let rho = transcript.challenge_extension(b"opening-batch")?;
    let combined_value = selected_value + rho * freivalds_value;
    add_scaled_in_place(&mut combined_left, &row_weights, rho);
    let retained_folded = retain_folded.then(|| folded.clone());
    let message_reduction = prove_litebullet::<B>(
        combined_value,
        combined_left,
        folded,
        transcript,
        b"message-litebullet",
    )?;
    let opening_left_eval = tensor_inner_product(&point[col_log..], &message_reduction.a_point)?;
    let selection_left_eval = message_reduction.a_value - rho * opening_left_eval;
    let generator_right_input = tensor_weights(&message_reduction.a_point);
    let generator_right = code.generator_right_projection_extension(&generator_right_input)?;
    let generator_reduction = prove_litebullet::<B>(
        selection_left_eval,
        pi_left,
        generator_right,
        transcript,
        b"generator-litebullet",
    )?;
    Ok((
        retained_folded,
        VerifiedRound {
            next_value: message_reduction.b_value,
            next_point: message_reduction.b_point,
            generator_value: generator_reduction.b_value,
            generator_left_point: generator_reduction.b_point,
            generator_right_point: message_reduction.a_point,
        },
    ))
}

#[allow(clippy::too_many_arguments)]
fn verify_base_round<B: ShortBaseField>(
    code: &ShortQaCode<B>,
    security_bits: usize,
    rf_layers: usize,
    root: [u8; DIGEST_BYTES],
    n_cols: usize,
    point: &[B::Extension],
    value: B::Extension,
    next_binding: NextLayerBinding,
    round: &ShortQaBaseRound<B>,
    transcript: &mut ShortTranscriptReader<B>,
) -> Result<Option<VerifiedRound<B::Extension>>, Error> {
    let n_rows = code.message_len();
    let n_ext_rows = code.codeword_len()?;
    let col_log = n_cols.ilog2() as usize;
    if n_cols == 0 || !n_cols.is_power_of_two() || point.len() != col_log + n_rows.ilog2() as usize
    {
        return Ok(None);
    }
    transcript.absorb_digest(b"round-root", &root);
    let opening_vector = (0..n_cols)
        .map(|_| transcript.read_response())
        .collect::<Result<Vec<_>, _>>()?;
    let column_weights = tensor_weights(&point[..col_log]);
    if inner_product(&opening_vector, &column_weights)? != value {
        return Ok(None);
    }
    let freivalds = (0..n_cols)
        .map(|_| transcript.challenge_extension(b"freivalds"))
        .collect::<Result<Vec<_>, _>>()?;
    let freivalds_value = inner_product(&opening_vector, &freivalds)?;
    absorb_next_layer_reader(transcript, next_binding);
    let query_count = code.num_queries(security_bits, rf_layers, n_ext_rows)?;
    let query_seed = transcript.challenge_digest(b"row-selection");
    let selected_rows = if query_count == n_ext_rows {
        RowSelection::All(n_ext_rows)
    } else {
        RowSelection::Partial(sample_without_replacement(
            query_count,
            n_ext_rows,
            query_seed,
        )?)
    };
    let Some(sampled_values) = authenticate_base_rows::<B>(
        &round.rows,
        &selected_rows,
        n_ext_rows,
        n_cols,
        &root,
        &freivalds,
    )?
    else {
        return Ok(None);
    };
    transcript.absorb_extensions(b"sampled-values", &sampled_values)?;
    verify_round_reductions::<B>(
        code,
        point,
        freivalds.len(),
        freivalds_value,
        &selected_rows,
        &sampled_values,
        transcript,
    )
}

#[allow(clippy::too_many_arguments)]
fn verify_extension_round<B: ShortBaseField>(
    code: &ShortQaCode<B>,
    security_bits: usize,
    rf_layers: usize,
    n_cols: usize,
    point: &[B::Extension],
    value: B::Extension,
    next_binding: NextLayerBinding,
    round: &ShortQaExtensionRound<B>,
    transcript: &mut ShortTranscriptReader<B>,
) -> Result<Option<VerifiedRound<B::Extension>>, Error> {
    let n_rows = code.message_len();
    let n_ext_rows = code.codeword_len()?;
    let col_log = n_cols.ilog2() as usize;
    if n_cols == 0 || !n_cols.is_power_of_two() || point.len() != col_log + n_rows.ilog2() as usize
    {
        return Ok(None);
    }
    transcript.absorb_digest(b"round-root", &round.root);
    let opening_vector = (0..n_cols)
        .map(|_| transcript.read_response())
        .collect::<Result<Vec<_>, _>>()?;
    let column_weights = tensor_weights(&point[..col_log]);
    if inner_product(&opening_vector, &column_weights)? != value {
        return Ok(None);
    }
    let freivalds = (0..n_cols)
        .map(|_| transcript.challenge_extension(b"freivalds"))
        .collect::<Result<Vec<_>, _>>()?;
    let freivalds_value = inner_product(&opening_vector, &freivalds)?;
    absorb_next_layer_reader(transcript, next_binding);
    let query_count = code.num_queries(security_bits, rf_layers, n_ext_rows)?;
    let query_seed = transcript.challenge_digest(b"row-selection");
    let selected_rows = if query_count == n_ext_rows {
        RowSelection::All(n_ext_rows)
    } else {
        RowSelection::Partial(sample_without_replacement(
            query_count,
            n_ext_rows,
            query_seed,
        )?)
    };
    let Some(sampled_values) = authenticate_extension_rows::<B>(
        &round.rows,
        &selected_rows,
        n_ext_rows,
        n_cols,
        &round.root,
        &freivalds,
    )?
    else {
        return Ok(None);
    };
    transcript.absorb_extensions(b"sampled-values", &sampled_values)?;
    verify_round_reductions::<B>(
        code,
        point,
        freivalds.len(),
        freivalds_value,
        &selected_rows,
        &sampled_values,
        transcript,
    )
}

fn verify_round_reductions<B: ShortBaseField>(
    code: &ShortQaCode<B>,
    point: &[B::Extension],
    n_cols: usize,
    freivalds_value: B::Extension,
    selected_rows: &RowSelection,
    sampled_values: &[B::Extension],
    transcript: &mut ShortTranscriptReader<B>,
) -> Result<Option<VerifiedRound<B::Extension>>, Error> {
    let n_rows = code.message_len();
    let n_ext_rows = code.codeword_len()?;
    let col_log = n_cols.ilog2() as usize;
    if n_cols == 0
        || !n_cols.is_power_of_two()
        || point.len() != col_log + n_rows.ilog2() as usize
        || selected_rows.len() != sampled_values.len()
    {
        return Ok(None);
    }
    let selection_log = selected_rows.len().next_power_of_two().ilog2() as usize;
    let selection_point = (0..selection_log)
        .map(|_| transcript.challenge_extension(b"selection-left"))
        .collect::<Result<Vec<_>, _>>()?;
    let selection_weights = tensor_weights_prefix(&selection_point, selected_rows.len())?;
    let selected_value = inner_product(&selection_weights, sampled_values)?;
    let rho = transcript.challenge_extension(b"opening-batch")?;
    let combined_value = selected_value + rho * freivalds_value;
    let message_reduction =
        verify_litebullet::<B>(combined_value, n_rows, transcript, b"message-litebullet")?;
    let Some(message_reduction) = message_reduction else {
        return Ok(None);
    };
    let opening_left_eval = tensor_inner_product(&point[col_log..], &message_reduction.a_point)?;
    let selection_left_eval = message_reduction.a_value - rho * opening_left_eval;
    let generator_reduction = verify_litebullet::<B>(
        selection_left_eval,
        n_ext_rows,
        transcript,
        b"generator-litebullet",
    )?;
    let Some(generator_reduction) = generator_reduction else {
        return Ok(None);
    };
    let expected_sparse_value = selected_rows
        .iter()
        .zip(&selection_weights)
        .map(|(row, &weight)| weight * tensor_weight_at(&generator_reduction.a_point, row))
        .fold(B::Extension::zero(), |sum, term| sum + term);
    if generator_reduction.a_value != expected_sparse_value {
        return Ok(None);
    }
    Ok(Some(VerifiedRound {
        next_value: message_reduction.b_value,
        next_point: message_reduction.b_point,
        generator_value: generator_reduction.b_value,
        generator_left_point: generator_reduction.b_point,
        generator_right_point: message_reduction.a_point,
    }))
}

fn authenticate_base_rows<B: ShortBaseField>(
    rows: &ShortQaBaseRows<B>,
    selected_rows: &RowSelection,
    n_ext_rows: usize,
    n_cols: usize,
    root: &[u8; DIGEST_BYTES],
    freivalds: &[B::Extension],
) -> Result<Option<Vec<B::Extension>>, Error> {
    match rows {
        ShortQaBaseRows::Partial(queries) => {
            if selected_rows.is_all()
                || queries.len() != selected_rows.len()
                || queries.iter().any(|query| query.row.len() != n_cols)
            {
                return Ok(None);
            }
            let mut sampled_values = Vec::with_capacity(queries.len());
            for (query, row) in queries.iter().zip(selected_rows.iter()) {
                let leaf = hash_base_row::<B>(&query.row)?;
                if !verify_merkle_path(leaf, row, n_ext_rows, &query.path, root) {
                    return Ok(None);
                }
                sampled_values.push(mixed_inner_product::<B>(&query.row, freivalds)?);
            }
            Ok(Some(sampled_values))
        }
        ShortQaBaseRows::AllRows {
            encoded_entries,
            n_rows,
            n_cols: encoded_cols,
        } => {
            if !selected_rows.is_all()
                || selected_rows.len() != n_ext_rows
                || *n_rows != n_ext_rows
                || *encoded_cols != n_cols
                || encoded_entries.len() != n_ext_rows * n_cols
                || build_base_tree::<B>(encoded_entries, n_ext_rows, n_cols)?.root() != *root
            {
                return Ok(None);
            }
            #[cfg(feature = "parallel")]
            let sampled = (0..n_ext_rows)
                .into_par_iter()
                .map(|row| {
                    mixed_row_inner_product::<B>(
                        encoded_entries,
                        n_ext_rows,
                        n_cols,
                        row,
                        freivalds,
                    )
                })
                .collect::<Result<Vec<_>, Error>>()?;
            #[cfg(not(feature = "parallel"))]
            let sampled = (0..n_ext_rows)
                .map(|row| {
                    mixed_row_inner_product::<B>(
                        encoded_entries,
                        n_ext_rows,
                        n_cols,
                        row,
                        freivalds,
                    )
                })
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Some(sampled))
        }
    }
}

fn authenticate_extension_rows<B: ShortBaseField>(
    rows: &ShortQaExtensionRows<B>,
    selected_rows: &RowSelection,
    n_ext_rows: usize,
    n_cols: usize,
    root: &[u8; DIGEST_BYTES],
    freivalds: &[B::Extension],
) -> Result<Option<Vec<B::Extension>>, Error> {
    match rows {
        ShortQaExtensionRows::Partial(queries) => {
            if selected_rows.is_all()
                || queries.len() != selected_rows.len()
                || queries.iter().any(|query| query.row.len() != n_cols)
            {
                return Ok(None);
            }
            let mut sampled_values = Vec::with_capacity(queries.len());
            for (query, row) in queries.iter().zip(selected_rows.iter()) {
                let leaf = hash_extension_row::<B>(&query.row)?;
                if !verify_merkle_path(leaf, row, n_ext_rows, &query.path, root) {
                    return Ok(None);
                }
                sampled_values.push(inner_product(&query.row, freivalds)?);
            }
            Ok(Some(sampled_values))
        }
        ShortQaExtensionRows::AllRows {
            encoded_entries,
            n_rows,
            n_cols: encoded_cols,
        } => {
            if !selected_rows.is_all()
                || selected_rows.len() != n_ext_rows
                || *n_rows != n_ext_rows
                || *encoded_cols != n_cols
                || encoded_entries.len() != n_ext_rows * n_cols
                || build_extension_tree::<B>(encoded_entries, n_ext_rows, n_cols)?.root() != *root
            {
                return Ok(None);
            }
            #[cfg(feature = "parallel")]
            let sampled = (0..n_ext_rows)
                .into_par_iter()
                .map(|row| {
                    extension_row_inner_product::<B>(
                        encoded_entries,
                        n_ext_rows,
                        n_cols,
                        row,
                        freivalds,
                    )
                })
                .collect::<Result<Vec<_>, Error>>()?;
            #[cfg(not(feature = "parallel"))]
            let sampled = (0..n_ext_rows)
                .map(|row| {
                    extension_row_inner_product::<B>(
                        encoded_entries,
                        n_ext_rows,
                        n_cols,
                        row,
                        freivalds,
                    )
                })
                .collect::<Result<Vec<_>, Error>>()?;
            Ok(Some(sampled))
        }
    }
}

pub(crate) fn verify_merkle_path(
    digest: [u8; DIGEST_BYTES],
    index: usize,
    leaf_count: usize,
    path: &ShortQaMerklePath,
    root: &[u8; DIGEST_BYTES],
) -> bool {
    verify_merkle_siblings(digest, index, leaf_count, &path.siblings, root)
}

pub(crate) fn verify_merkle_siblings(
    mut digest: [u8; DIGEST_BYTES],
    mut index: usize,
    leaf_count: usize,
    siblings: &[[u8; DIGEST_BYTES]],
    root: &[u8; DIGEST_BYTES],
) -> bool {
    if leaf_count == 0
        || !leaf_count.is_power_of_two()
        || index >= leaf_count
        || siblings.len() != leaf_count.ilog2() as usize
    {
        return false;
    }
    for sibling in siblings {
        digest = if index & 1 == 0 {
            hash_inner(&digest, sibling)
        } else {
            hash_inner(sibling, &digest)
        };
        index >>= 1;
    }
    digest == *root
}

#[derive(Clone, Debug)]
struct LiteBulletReduction<E: ShortFieldElement> {
    a_value: E,
    a_point: Vec<E>,
    b_value: E,
    b_point: Vec<E>,
}

fn prove_litebullet<B: ShortBaseField>(
    claimed_value: B::Extension,
    mut left: Vec<B::Extension>,
    mut right: Vec<B::Extension>,
    transcript: &mut ShortTranscriptWriter<B>,
    label: &[u8],
) -> Result<LiteBulletReduction<B::Extension>, Error> {
    if left.is_empty()
        || !left.len().is_power_of_two()
        || left.len() != right.len()
        || inner_product(&left, &right)? != claimed_value
    {
        return Err(Error::InvalidCommitment);
    }
    transcript.absorb_bytes(b"litebullet-domain", label);
    let mut rhs = claimed_value;
    let mut a_point = Vec::with_capacity(left.len().ilog2() as usize);
    let mut b_point = Vec::with_capacity(left.len().ilog2() as usize);
    while left.len() > 1 {
        let half = left.len() / 2;
        let (l_cross, r_cross) = cross_inner_products(&left, &right);
        transcript.push_response(l_cross)?;
        transcript.push_response(r_cross)?;
        let challenge = transcript.challenge_nonzero_extension(b"litebullet-fold")?;
        let challenge_inverse = challenge.try_inverse().ok_or(Error::TranscriptError)?;
        rhs += l_cross * challenge + r_cross * challenge_inverse;
        fold_litebullet_vectors(&mut left, &mut right, challenge_inverse, challenge);
        left.truncate(half);
        right.truncate(half);
        a_point.push(challenge_inverse);
        b_point.push(challenge);
    }
    let a_value = left[0];
    let b_value = right[0];
    transcript.push_response(a_value)?;
    transcript.push_response(b_value)?;
    if a_value * b_value != rhs {
        return Err(Error::InvalidCommitment);
    }
    Ok(LiteBulletReduction {
        a_value,
        a_point,
        b_value,
        b_point,
    })
}

fn prove_sparse_litebullet<B: ShortBaseField>(
    claimed_value: B::Extension,
    length: usize,
    selected_indices: &RowSelection,
    selected_weights: &[B::Extension],
    right: Vec<B::Extension>,
    transcript: &mut ShortTranscriptWriter<B>,
    label: &[u8],
) -> Result<LiteBulletReduction<B::Extension>, Error> {
    if length == 0
        || !length.is_power_of_two()
        || right.len() != length
        || selected_indices.len() != selected_weights.len()
        || selected_indices.iter().any(|index| index >= length)
    {
        return Err(Error::InvalidNumberOfVariables);
    }
    let mut left = vec![B::Extension::zero(); length];
    for (index, &weight) in selected_indices.iter().zip(selected_weights) {
        left[index] += weight;
    }
    prove_litebullet::<B>(claimed_value, left, right, transcript, label)
}

fn verify_litebullet<B: ShortBaseField>(
    claimed_value: B::Extension,
    length: usize,
    transcript: &mut ShortTranscriptReader<B>,
    label: &[u8],
) -> Result<Option<LiteBulletReduction<B::Extension>>, Error> {
    if length == 0 || !length.is_power_of_two() {
        return Ok(None);
    }
    transcript.absorb_bytes(b"litebullet-domain", label);
    let mut rhs = claimed_value;
    let mut current_len = length;
    let mut a_point = Vec::with_capacity(length.ilog2() as usize);
    let mut b_point = Vec::with_capacity(length.ilog2() as usize);
    while current_len > 1 {
        let l_cross = transcript.read_response()?;
        let r_cross = transcript.read_response()?;
        let challenge = transcript.challenge_nonzero_extension(b"litebullet-fold")?;
        let challenge_inverse = challenge.try_inverse().ok_or(Error::TranscriptError)?;
        rhs += l_cross * challenge + r_cross * challenge_inverse;
        a_point.push(challenge_inverse);
        b_point.push(challenge);
        current_len /= 2;
    }
    let a_value = transcript.read_response()?;
    let b_value = transcript.read_response()?;
    if a_value * b_value != rhs {
        return Ok(None);
    }
    Ok(Some(LiteBulletReduction {
        a_value,
        a_point,
        b_value,
        b_point,
    }))
}

struct ShortTranscriptWriter<B: ShortBaseField> {
    hasher: Blake2s256,
    challenge_counter: u64,
    responses: Vec<B::Extension>,
}

impl<B: ShortBaseField> ShortTranscriptWriter<B> {
    fn new() -> Self {
        let mut transcript = Self {
            hasher: Blake2s256::new(),
            challenge_counter: 0,
            responses: Vec::new(),
        };
        transcript.absorb_bytes(b"domain", TRANSCRIPT_DOMAIN);
        transcript
    }

    fn push_response(&mut self, value: B::Extension) -> Result<(), Error> {
        self.absorb_extension(b"response", value)?;
        self.responses.push(value);
        Ok(())
    }

    fn absorb_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        absorb_framed(&mut self.hasher, label, bytes);
    }

    fn absorb_digest(&mut self, label: &[u8], digest: &[u8; DIGEST_BYTES]) {
        self.absorb_bytes(label, digest);
    }

    fn absorb_extension(&mut self, label: &[u8], value: B::Extension) -> Result<(), Error> {
        let bytes = extension_bytes::<B>(value)?;
        self.absorb_bytes(label, &bytes);
        Ok(())
    }

    fn absorb_extensions(&mut self, label: &[u8], values: &[B::Extension]) -> Result<(), Error> {
        self.absorb_bytes(b"vector-label", label);
        self.absorb_bytes(b"vector-length", &(values.len() as u64).to_le_bytes());
        for value in values {
            self.absorb_extension(b"vector-element", *value)?;
        }
        Ok(())
    }

    fn challenge_digest(&mut self, label: &[u8]) -> [u8; DIGEST_BYTES] {
        let digest = derive_digest(&self.hasher, label, self.challenge_counter, 0, 0);
        self.challenge_counter = self.challenge_counter.wrapping_add(1);
        self.absorb_bytes(b"digest-challenge-label", label);
        self.absorb_digest(b"digest-challenge", &digest);
        digest
    }

    fn challenge_extension(&mut self, label: &[u8]) -> Result<B::Extension, Error> {
        self.challenge_extension_with_nonzero(label, false)
    }

    fn challenge_nonzero_extension(&mut self, label: &[u8]) -> Result<B::Extension, Error> {
        self.challenge_extension_with_nonzero(label, true)
    }

    fn challenge_extension_with_nonzero(
        &mut self,
        label: &[u8],
        require_nonzero: bool,
    ) -> Result<B::Extension, Error> {
        let counter = self.challenge_counter;
        for outer_attempt in 0..u64::MAX {
            let value =
                derive_extension_challenge::<B>(&self.hasher, label, counter, outer_attempt)?;
            if !require_nonzero || value != B::Extension::zero() {
                self.challenge_counter = self.challenge_counter.wrapping_add(1);
                self.absorb_bytes(b"field-challenge-label", label);
                self.absorb_extension(b"field-challenge", value)?;
                return Ok(value);
            }
        }
        Err(Error::TranscriptError)
    }
}

struct ShortTranscriptReader<'a, B: ShortBaseField> {
    hasher: Blake2s256,
    challenge_counter: u64,
    responses: &'a [B::Extension],
    response_cursor: usize,
}

impl<'a, B: ShortBaseField> ShortTranscriptReader<'a, B> {
    fn new(responses: &'a [B::Extension]) -> Self {
        let mut transcript = Self {
            hasher: Blake2s256::new(),
            challenge_counter: 0,
            responses,
            response_cursor: 0,
        };
        transcript.absorb_bytes(b"domain", TRANSCRIPT_DOMAIN);
        transcript
    }

    fn read_response(&mut self) -> Result<B::Extension, Error> {
        let value = self
            .responses
            .get(self.response_cursor)
            .copied()
            .ok_or(Error::TranscriptError)?;
        self.response_cursor += 1;
        self.absorb_extension(b"response", value)?;
        Ok(value)
    }

    fn absorb_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        absorb_framed(&mut self.hasher, label, bytes);
    }

    fn absorb_digest(&mut self, label: &[u8], digest: &[u8; DIGEST_BYTES]) {
        self.absorb_bytes(label, digest);
    }

    fn absorb_extension(&mut self, label: &[u8], value: B::Extension) -> Result<(), Error> {
        let bytes = extension_bytes::<B>(value)?;
        self.absorb_bytes(label, &bytes);
        Ok(())
    }

    fn absorb_extensions(&mut self, label: &[u8], values: &[B::Extension]) -> Result<(), Error> {
        self.absorb_bytes(b"vector-label", label);
        self.absorb_bytes(b"vector-length", &(values.len() as u64).to_le_bytes());
        for value in values {
            self.absorb_extension(b"vector-element", *value)?;
        }
        Ok(())
    }

    fn challenge_digest(&mut self, label: &[u8]) -> [u8; DIGEST_BYTES] {
        let digest = derive_digest(&self.hasher, label, self.challenge_counter, 0, 0);
        self.challenge_counter = self.challenge_counter.wrapping_add(1);
        self.absorb_bytes(b"digest-challenge-label", label);
        self.absorb_digest(b"digest-challenge", &digest);
        digest
    }

    fn challenge_extension(&mut self, label: &[u8]) -> Result<B::Extension, Error> {
        self.challenge_extension_with_nonzero(label, false)
    }

    fn challenge_nonzero_extension(&mut self, label: &[u8]) -> Result<B::Extension, Error> {
        self.challenge_extension_with_nonzero(label, true)
    }

    fn challenge_extension_with_nonzero(
        &mut self,
        label: &[u8],
        require_nonzero: bool,
    ) -> Result<B::Extension, Error> {
        let counter = self.challenge_counter;
        for outer_attempt in 0..u64::MAX {
            let value =
                derive_extension_challenge::<B>(&self.hasher, label, counter, outer_attempt)?;
            if !require_nonzero || value != B::Extension::zero() {
                self.challenge_counter = self.challenge_counter.wrapping_add(1);
                self.absorb_bytes(b"field-challenge-label", label);
                self.absorb_extension(b"field-challenge", value)?;
                return Ok(value);
            }
        }
        Err(Error::TranscriptError)
    }

    fn is_exhausted(&self) -> bool {
        self.response_cursor == self.responses.len()
    }
}

pub(crate) fn absorb_framed(hasher: &mut Blake2s256, label: &[u8], bytes: &[u8]) {
    hasher.update((label.len() as u64).to_le_bytes());
    hasher.update(label);
    hasher.update((bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

fn derive_digest(
    state: &Blake2s256,
    label: &[u8],
    counter: u64,
    coefficient: u64,
    attempt: u64,
) -> [u8; DIGEST_BYTES] {
    let mut hasher = state.clone();
    absorb_framed(&mut hasher, b"challenge-label", label);
    hasher.update(counter.to_le_bytes());
    hasher.update(coefficient.to_le_bytes());
    hasher.update(attempt.to_le_bytes());
    hasher.finalize().into()
}

fn derive_extension_challenge<B: ShortBaseField>(
    state: &Blake2s256,
    label: &[u8],
    counter: u64,
    outer_attempt: u64,
) -> Result<B::Extension, Error> {
    let mut coefficients = Vec::with_capacity(B::EXTENSION_DEGREE);
    for coefficient in 0..B::EXTENSION_DEGREE {
        let mut attempt = outer_attempt;
        let sampled = loop {
            let digest = derive_digest(state, label, counter, coefficient as u64, attempt);
            if let Some(value) = B::from_uniform_bytes(&digest) {
                break value;
            }
            attempt = attempt.checked_add(1).ok_or(Error::TranscriptError)?;
        };
        coefficients.push(sampled);
    }
    Ok(B::Extension::from_basis_coefficients_fn(|index| {
        coefficients[index]
    }))
}

fn validate_parameters<B: ShortBaseField>(
    pc: &ShortQaFoldPC<B>,
    pp: &ShortQaFoldParameters<B>,
) -> Result<(), Error> {
    let top = pp.sequence.top_code()?;
    let expected_parameter_digest = parameter_digest::<B>(
        &pp.sequence,
        &pp.derived_parameters.digest(),
        &pp.setup_digest,
        pc.n_witness,
        pc.extension_columns,
        pc.index,
        pc.security_bits,
    )?;
    if pp.sequence.witness_size() != pc.n_witness
        || pp.sequence.extension_columns() != pc.extension_columns
        || pp.sequence.recursive_index() != pc.recursive_index
        || top.index() != pc.index
        || pp.security_bits != pc.security_bits
        || pp.parameter_digest != pp.validated_parameter_digest
        || expected_parameter_digest != pp.validated_parameter_digest
        || pp.sequence.generator_digest() != pp.validated_generator_digest
        || pp.derived_parameters.digest() != pp.validated_derived_digest
        || pp.setup_digest != pp.validated_setup_digest
    {
        return Err(Error::InvalidCommitment);
    }
    Ok(())
}

fn validate_setup_aggregate<B: ShortBaseField>(
    pp: &ShortQaFoldParameters<B>,
    setup_commitments: &ShortQaFoldSetupCommitments<B>,
) -> Result<(), Error> {
    if setup_commitments.setup_digest() != pp.setup_digest {
        return Err(Error::InvalidCommitment);
    }
    Ok(())
}

fn parameter_digest<B: ShortBaseField>(
    sequence: &ShortQaFold<B>,
    derived_digest: &[u8; DIGEST_BYTES],
    setup_digest: &[u8; DIGEST_BYTES],
    n_witness: usize,
    extension_columns: usize,
    index: usize,
    security_bits: usize,
) -> Result<[u8; DIGEST_BYTES], Error> {
    let mut hasher = Blake2s256::new();
    absorb_framed(&mut hasher, b"domain", b"qafold-short-parameters/v4");
    absorb_framed(&mut hasher, b"field-name", B::NAME.as_bytes());
    absorb_framed(&mut hasher, b"modulus", &B::canonical_modulus_bytes());
    absorb_framed(
        &mut hasher,
        b"extension-degree",
        &(B::EXTENSION_DEGREE as u64).to_le_bytes(),
    );
    for (label, value) in [
        (b"witness".as_slice(), n_witness),
        (b"extension-columns".as_slice(), extension_columns),
        (b"index".as_slice(), index),
        (b"security".as_slice(), security_bits),
        (b"rf-layers".as_slice(), sequence.rf_layers()),
    ] {
        absorb_framed(&mut hasher, label, &(value as u64).to_le_bytes());
    }
    absorb_framed(
        &mut hasher,
        b"generator-sequence",
        &sequence.generator_digest(),
    );
    // Preserve both historical mixed-four and explicit all-two transcript encodings.
    if sequence.recursive_index() != 4 {
        absorb_framed(
            &mut hasher,
            b"recursive-index-override",
            &(sequence.recursive_index() as u64).to_le_bytes(),
        );
    }
    absorb_framed(&mut hasher, b"derived-rfc-sequence", derived_digest);
    absorb_framed(&mut hasher, b"setup-roots-and-shapes", setup_digest);
    Ok(hasher.finalize().into())
}

// Large zero fills otherwise touch every page on the calling worker before any
// column can be encoded. Indexed parallel collection initializes disjoint ranges
// directly, spreading both the writes and NUMA first touch without unsafe storage.
fn zeroed_field_vec<F: ShortFieldElement>(len: usize) -> Vec<F> {
    #[cfg(feature = "parallel")]
    if len >= 1 << 16 {
        return (0..len).into_par_iter().map(|_| F::zero()).collect();
    }
    vec![F::zero(); len]
}

fn commit_extension_matrix<B: ShortBaseField>(
    code: &ShortQaCode<B>,
    entries: &[B::Extension],
    n_rows: usize,
    n_cols: usize,
) -> Result<([u8; DIGEST_BYTES], ExtensionRoundCache<B>), Error> {
    let n_ext_rows = code.codeword_len()?;
    if n_rows != code.message_len()
        || entries.len() != n_rows.checked_mul(n_cols).ok_or(Error::EncodingError)?
    {
        return Err(Error::EncodingError);
    }
    let encoded_len = n_ext_rows.checked_mul(n_cols).ok_or(Error::EncodingError)?;
    let mut encoded_entries = zeroed_field_vec::<B::Extension>(encoded_len);
    #[cfg(feature = "parallel")]
    encoded_entries
        .par_chunks_mut(n_ext_rows)
        .zip(entries.par_chunks(n_rows))
        .try_for_each(|(encoded, message)| code.encode_extension_into(message, encoded))?;
    #[cfg(not(feature = "parallel"))]
    encoded_entries
        .chunks_mut(n_ext_rows)
        .zip(entries.chunks(n_rows))
        .try_for_each(|(encoded, message)| code.encode_extension_into(message, encoded))?;
    let tree = build_extension_tree::<B>(&encoded_entries, n_ext_rows, n_cols)?;
    let root = tree.root();
    Ok((
        root,
        ExtensionRoundCache {
            encoded_entries,
            n_ext_rows,
            n_cols,
            tree,
        },
    ))
}

fn build_base_tree<B: ShortBaseField>(
    encoded_entries: &[B],
    n_rows: usize,
    n_cols: usize,
) -> Result<Blake2sMerkleTree, Error> {
    build_base_tree_reusing(encoded_entries, n_rows, n_cols, None)
}

fn build_base_tree_reusing<B: ShortBaseField>(
    encoded_entries: &[B], n_rows: usize, n_cols: usize,
    recycled: Option<Blake2sMerkleTree>,
) -> Result<Blake2sMerkleTree, Error> {
    if n_rows == 0
        || !n_rows.is_power_of_two()
        || n_cols == 0
        || encoded_entries.len() != n_rows.checked_mul(n_cols).ok_or(Error::EncodingError)?
    {
        return Err(Error::EncodingError);
    }
    let mut levels = recycled.map(|tree| tree.levels).unwrap_or_default();
    let reusable = if levels.is_empty() { Vec::new() } else { core::mem::take(&mut levels[0]) };
    let leaves = hash_encoded_rows_reusing(
        encoded_entries,
        n_rows,
        n_cols,
        B::CANONICAL_BYTES,
        LEAF_BASE_DOMAIN,
        |value, output| value.write_canonical_bytes(output),
        reusable,
    )?;
    Blake2sMerkleTree::from_leaves_reusing(leaves, levels)
}

fn build_extension_tree<B: ShortBaseField>(
    encoded_entries: &[B::Extension],
    n_rows: usize,
    n_cols: usize,
) -> Result<Blake2sMerkleTree, Error> {
    if n_rows == 0
        || !n_rows.is_power_of_two()
        || n_cols == 0
        || encoded_entries.len() != n_rows.checked_mul(n_cols).ok_or(Error::EncodingError)?
    {
        return Err(Error::EncodingError);
    }
    let leaves = hash_encoded_rows(
        encoded_entries,
        n_rows,
        n_cols,
        B::Extension::canonical_byte_len(),
        LEAF_EXTENSION_DOMAIN,
        |value, output| value.write_canonical_bytes(output),
    )?;
    Blake2sMerkleTree::from_leaves(leaves)
}

/// Hash complete canonical rows with the same framing and byte order as the
/// scalar row helpers. Each Rayon job reuses its row buffer; the shared prefix
/// contains the domain and row width and is cloned before absorbing each row.
fn hash_encoded_rows<T: Sync, W>(
    entries: &[T],
    n_rows: usize,
    n_cols: usize,
    element_bytes: usize,
    domain: &[u8],
    write: W,
) -> Result<Vec<[u8; DIGEST_BYTES]>, Error>
where
    W: Fn(&T, &mut [u8]) -> Result<(), Error> + Sync,
{
    hash_encoded_rows_reusing(entries, n_rows, n_cols, element_bytes, domain, write, Vec::new())
}

fn hash_encoded_rows_reusing<T: Sync, W>(
    entries: &[T], n_rows: usize, n_cols: usize, element_bytes: usize,
    domain: &[u8], write: W, mut output: Vec<[u8; DIGEST_BYTES]>,
) -> Result<Vec<[u8; DIGEST_BYTES]>, Error>
where W: Fn(&T, &mut [u8]) -> Result<(), Error> + Sync,
{
    if element_bytes == 0
        || entries.len() != n_rows.checked_mul(n_cols).ok_or(Error::EncodingError)?
    {
        return Err(Error::EncodingError);
    }
    let row_bytes = n_cols
        .checked_mul(element_bytes)
        .ok_or(Error::EncodingError)?;
    let mut prefix = Vec::new();
    append_hash_frame(&mut prefix, b"domain", domain);
    append_hash_frame(&mut prefix, b"width", &(n_cols as u64).to_le_bytes());
    let message_bytes = prefix.len().checked_add(row_bytes).ok_or(Error::EncodingError)?;
    output.resize(n_rows, [0u8; DIGEST_BYTES]);
    // Eight independent BLAKE2s messages use AVX2; do not change the hash or framing.
    let hash_tile = |(tile, output): (usize, &mut [[u8; DIGEST_BYTES]])| -> Result<(), Error> {
        let mut buffer = vec![0; message_bytes * output.len()];
        for (lane, message) in buffer.chunks_exact_mut(message_bytes).enumerate() {
            message[..prefix.len()].copy_from_slice(&prefix);
            for (column, bytes) in message[prefix.len()..].chunks_exact_mut(element_bytes).enumerate() {
                write(&entries[column * n_rows + tile * HASH_TILE + lane], bytes)?;
            }
        }
        hash_messages(&buffer, message_bytes, output);
        Ok(())
    };
    #[cfg(feature = "parallel")]
    if n_rows >= 4096 {
        output.par_chunks_mut(HASH_TILE).enumerate().try_for_each(hash_tile)?;
        return Ok(output);
    }
    output.chunks_mut(HASH_TILE).enumerate().try_for_each(hash_tile)?;
    Ok(output)
}

const HASH_TILE: usize = 64;
fn append_hash_frame(output: &mut Vec<u8>, label: &[u8], bytes: &[u8]) {
    output.extend_from_slice(&(label.len() as u64).to_le_bytes());
    output.extend_from_slice(label);
    output.extend_from_slice(&(bytes.len() as u64).to_le_bytes());
    output.extend_from_slice(bytes);
}
fn hash_messages(buffer: &[u8], message_bytes: usize, output: &mut [[u8; DIGEST_BYTES]]) {
    use blake2s_simd::{Params, many::{HashManyJob, hash_many}};
    let parameters = Params::new();
    let mut jobs = buffer.chunks_exact(message_bytes)
        .map(|message| HashManyJob::new(&parameters, message)).collect::<Vec<_>>();
    hash_many(&mut jobs);
    for (output, job) in output.iter_mut().zip(jobs) {
        output.copy_from_slice(job.to_hash().as_bytes());
    }
}
fn hash_parent_level_reusing(children: &[[u8; DIGEST_BYTES]], mut output: Vec<[u8; DIGEST_BYTES]>) -> Vec<[u8; DIGEST_BYTES]> {
    let mut prefix = Vec::new();
    append_hash_frame(&mut prefix, b"domain", INNER_DOMAIN);
    let message_bytes = prefix.len() + 2 * DIGEST_BYTES;
    output.resize(children.len() / 2, [0u8; DIGEST_BYTES]);
    let hash_tile = |(tile, output): (usize, &mut [[u8; DIGEST_BYTES]])| {
        let mut buffer = vec![0; message_bytes * output.len()];
        for (lane, message) in buffer.chunks_exact_mut(message_bytes).enumerate() {
            message[..prefix.len()].copy_from_slice(&prefix);
            let child = 2 * (tile * HASH_TILE + lane);
            message[prefix.len()..prefix.len() + DIGEST_BYTES].copy_from_slice(&children[child]);
            message[prefix.len() + DIGEST_BYTES..].copy_from_slice(&children[child + 1]);
        }
        hash_messages(&buffer, message_bytes, output);
    };
    #[cfg(feature = "parallel")]
    if output.len() >= 4096 {
        output.par_chunks_mut(HASH_TILE).enumerate().for_each(hash_tile);
        return output;
    }
    output.chunks_mut(HASH_TILE).enumerate().for_each(hash_tile);
    output
}

fn hash_base_row<B: ShortBaseField>(row: &[B]) -> Result<[u8; DIGEST_BYTES], Error> {
    let mut hasher = Blake2s256::new();
    absorb_framed(&mut hasher, b"domain", LEAF_BASE_DOMAIN);
    absorb_framed(&mut hasher, b"width", &(row.len() as u64).to_le_bytes());
    let mut encoded = vec![0u8; B::CANONICAL_BYTES];
    for value in row {
        value.write_canonical_bytes(&mut encoded)?;
        hasher.update(&encoded);
    }
    Ok(hasher.finalize().into())
}

fn hash_extension_row<B: ShortBaseField>(
    row: &[B::Extension],
) -> Result<[u8; DIGEST_BYTES], Error> {
    let mut hasher = Blake2s256::new();
    absorb_framed(&mut hasher, b"domain", LEAF_EXTENSION_DOMAIN);
    absorb_framed(&mut hasher, b"width", &(row.len() as u64).to_le_bytes());
    let mut encoded = vec![0u8; B::Extension::canonical_byte_len()];
    for value in row {
        value.write_canonical_bytes(&mut encoded)?;
        hasher.update(&encoded);
    }
    Ok(hasher.finalize().into())
}

#[cfg(test)]
fn hash_base_encoded_row<B: ShortBaseField>(
    entries: &[B],
    n_rows: usize,
    n_cols: usize,
    row: usize,
) -> Result<[u8; DIGEST_BYTES], Error> {
    if row >= n_rows || entries.len() != n_rows * n_cols {
        return Err(Error::EncodingError);
    }
    let mut hasher = Blake2s256::new();
    absorb_framed(&mut hasher, b"domain", LEAF_BASE_DOMAIN);
    absorb_framed(&mut hasher, b"width", &(n_cols as u64).to_le_bytes());
    let mut encoded = vec![0u8; B::CANONICAL_BYTES];
    for column in 0..n_cols {
        entries[column * n_rows + row].write_canonical_bytes(&mut encoded)?;
        hasher.update(&encoded);
    }
    Ok(hasher.finalize().into())
}

#[cfg(test)]
fn hash_extension_encoded_row<B: ShortBaseField>(
    entries: &[B::Extension],
    n_rows: usize,
    n_cols: usize,
    row: usize,
) -> Result<[u8; DIGEST_BYTES], Error> {
    if row >= n_rows || entries.len() != n_rows * n_cols {
        return Err(Error::EncodingError);
    }
    let mut hasher = Blake2s256::new();
    absorb_framed(&mut hasher, b"domain", LEAF_EXTENSION_DOMAIN);
    absorb_framed(&mut hasher, b"width", &(n_cols as u64).to_le_bytes());
    let mut encoded = vec![0u8; B::Extension::canonical_byte_len()];
    for column in 0..n_cols {
        entries[column * n_rows + row].write_canonical_bytes(&mut encoded)?;
        hasher.update(&encoded);
    }
    Ok(hasher.finalize().into())
}

fn hash_terminal_vector<B: ShortBaseField>(
    values: &[B::Extension],
) -> Result<[u8; DIGEST_BYTES], Error> {
    let mut hasher = Blake2s256::new();
    absorb_framed(&mut hasher, b"domain", TERMINAL_VECTOR_DOMAIN);
    absorb_framed(&mut hasher, b"length", &(values.len() as u64).to_le_bytes());
    let mut encoded = vec![0u8; B::Extension::canonical_byte_len()];
    for value in values {
        value.write_canonical_bytes(&mut encoded)?;
        absorb_framed(&mut hasher, b"element", &encoded);
    }
    Ok(hasher.finalize().into())
}

fn hash_inner(left: &[u8; DIGEST_BYTES], right: &[u8; DIGEST_BYTES]) -> [u8; DIGEST_BYTES] {
    let mut hasher = Blake2s256::new();
    absorb_framed(&mut hasher, b"domain", INNER_DOMAIN);
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

fn hash_inner_prefixed(
    prefix: &Blake2s256,
    left: &[u8; DIGEST_BYTES],
    right: &[u8; DIGEST_BYTES],
) -> [u8; DIGEST_BYTES] {
    let mut hasher = prefix.clone();
    hasher.update(left);
    hasher.update(right);
    hasher.finalize().into()
}

fn base_encoded_row<B: ShortBaseField>(
    entries: &[B],
    n_rows: usize,
    n_cols: usize,
    row: usize,
) -> Vec<B> {
    (0..n_cols)
        .map(|column| entries[column * n_rows + row])
        .collect()
}

fn extension_encoded_row<B: ShortBaseField>(
    entries: &[B::Extension],
    n_rows: usize,
    n_cols: usize,
    row: usize,
) -> Vec<B::Extension> {
    (0..n_cols)
        .map(|column| entries[column * n_rows + row])
        .collect()
}

fn base_systematic_row_projection<B: ShortBaseField>(
    entries: &[B],
    n_ext_rows: usize,
    n_rows: usize,
    n_cols: usize,
    row_weights: &[B::Extension],
) -> Result<Vec<B::Extension>, Error> {
    if n_rows > n_ext_rows || row_weights.len() != n_rows || entries.len() != n_ext_rows * n_cols {
        return Err(Error::EncodingError);
    }
    #[cfg(feature = "parallel")]
    let result = (0..n_cols)
        .into_par_iter()
        .map(|column| {
            mixed_inner_product::<B>(
                &entries[column * n_ext_rows..column * n_ext_rows + n_rows],
                row_weights,
            )
        })
        .collect::<Result<Vec<_>, Error>>()?;
    #[cfg(not(feature = "parallel"))]
    let result = (0..n_cols)
        .map(|column| {
            mixed_inner_product::<B>(
                &entries[column * n_ext_rows..column * n_ext_rows + n_rows],
                row_weights,
            )
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(result)
}

fn extension_systematic_row_projection<B: ShortBaseField>(
    entries: &[B::Extension],
    n_ext_rows: usize,
    n_rows: usize,
    n_cols: usize,
    row_weights: &[B::Extension],
) -> Result<Vec<B::Extension>, Error> {
    if n_rows > n_ext_rows || row_weights.len() != n_rows || entries.len() != n_ext_rows * n_cols {
        return Err(Error::EncodingError);
    }
    #[cfg(feature = "parallel")]
    let result = (0..n_cols)
        .into_par_iter()
        .map(|column| {
            inner_product(
                &entries[column * n_ext_rows..column * n_ext_rows + n_rows],
                row_weights,
            )
        })
        .collect::<Result<Vec<_>, Error>>()?;
    #[cfg(not(feature = "parallel"))]
    let result = (0..n_cols)
        .map(|column| {
            inner_product(
                &entries[column * n_ext_rows..column * n_ext_rows + n_rows],
                row_weights,
            )
        })
        .collect::<Result<Vec<_>, Error>>()?;
    Ok(result)
}

fn base_systematic_fold<B: ShortBaseField>(
    entries: &[B],
    n_ext_rows: usize,
    n_rows: usize,
    n_cols: usize,
    weights: &[B::Extension],
) -> Result<Vec<B::Extension>, Error> {
    if n_rows > n_ext_rows || weights.len() != n_cols || entries.len() != n_ext_rows * n_cols {
        return Err(Error::EncodingError);
    }
    #[cfg(feature = "parallel")]
    let result = (0..n_rows)
        .into_par_iter()
        .map(|row| {
            (0..n_cols).fold(B::Extension::zero(), |sum, column| {
                sum + weights[column].mul_base(entries[column * n_ext_rows + row])
            })
        })
        .collect();
    #[cfg(not(feature = "parallel"))]
    let result = (0..n_rows)
        .map(|row| {
            (0..n_cols).fold(B::Extension::zero(), |sum, column| {
                sum + weights[column].mul_base(entries[column * n_ext_rows + row])
            })
        })
        .collect();
    Ok(result)
}

fn extension_systematic_fold<B: ShortBaseField>(
    entries: &[B::Extension],
    n_ext_rows: usize,
    n_rows: usize,
    n_cols: usize,
    weights: &[B::Extension],
) -> Result<Vec<B::Extension>, Error> {
    if n_rows > n_ext_rows || weights.len() != n_cols || entries.len() != n_ext_rows * n_cols {
        return Err(Error::EncodingError);
    }
    #[cfg(feature = "parallel")]
    let result = (0..n_rows)
        .into_par_iter()
        .map(|row| {
            (0..n_cols).fold(B::Extension::zero(), |sum, column| {
                sum + weights[column] * entries[column * n_ext_rows + row]
            })
        })
        .collect();
    #[cfg(not(feature = "parallel"))]
    let result = (0..n_rows)
        .map(|row| {
            (0..n_cols).fold(B::Extension::zero(), |sum, column| {
                sum + weights[column] * entries[column * n_ext_rows + row]
            })
        })
        .collect();
    Ok(result)
}

fn mixed_inner_product<B: ShortBaseField>(
    base: &[B],
    extension: &[B::Extension],
) -> Result<B::Extension, Error> {
    if base.len() != extension.len() {
        return Err(Error::InvalidNumberOfVariables);
    }
    #[cfg(feature = "parallel")]
    if base.len() >= OPENING_PARALLEL_GRAIN {
        return Ok(base
            .par_chunks(OPENING_PARALLEL_GRAIN)
            .zip(extension.par_chunks(OPENING_PARALLEL_GRAIN))
            .map(|(base, extension)| {
                base.iter()
                    .zip(extension)
                    .fold(B::Extension::zero(), |sum, (&base, &extension)| {
                        sum + extension.mul_base(base)
                    })
            })
            .reduce(B::Extension::zero, |left, right| left + right));
    }
    Ok(base
        .iter()
        .zip(extension)
        .fold(B::Extension::zero(), |sum, (&base, &extension)| {
            sum + extension.mul_base(base)
        }))
}

fn mixed_row_inner_product<B: ShortBaseField>(
    entries: &[B],
    n_rows: usize,
    n_cols: usize,
    row: usize,
    weights: &[B::Extension],
) -> Result<B::Extension, Error> {
    if row >= n_rows || weights.len() != n_cols || entries.len() != n_rows * n_cols {
        return Err(Error::InvalidNumberOfVariables);
    }
    Ok((0..n_cols).fold(B::Extension::zero(), |sum, column| {
        sum + weights[column].mul_base(entries[column * n_rows + row])
    }))
}

fn extension_row_inner_product<B: ShortBaseField>(
    entries: &[B::Extension],
    n_rows: usize,
    n_cols: usize,
    row: usize,
    weights: &[B::Extension],
) -> Result<B::Extension, Error> {
    if row >= n_rows || weights.len() != n_cols || entries.len() != n_rows * n_cols {
        return Err(Error::InvalidNumberOfVariables);
    }
    Ok((0..n_cols).fold(B::Extension::zero(), |sum, column| {
        sum + weights[column] * entries[column * n_rows + row]
    }))
}

fn inner_product<E: ShortFieldElement>(left: &[E], right: &[E]) -> Result<E, Error> {
    if left.len() != right.len() {
        return Err(Error::InvalidNumberOfVariables);
    }
    #[cfg(feature = "parallel")]
    if left.len() >= OPENING_PARALLEL_GRAIN {
        return Ok(left
            .par_chunks(OPENING_PARALLEL_GRAIN)
            .zip(right.par_chunks(OPENING_PARALLEL_GRAIN))
            .map(|(left, right)| {
                left.iter()
                    .zip(right)
                    .fold(E::zero(), |sum, (&left, &right)| sum + left * right)
            })
            .reduce(E::zero, |left, right| left + right));
    }
    Ok(left
        .iter()
        .zip(right)
        .fold(E::zero(), |sum, (&left, &right)| sum + left * right))
}

fn tensor_weights<E: ShortFieldElement>(point: &[E]) -> Vec<E> {
    let mut weights = vec![E::zero(); 1usize << point.len()];
    weights[0] = E::one();
    let mut filled = 1;
    // Append the high half using reversed coordinates. This preserves the
    // original MSB-first tensor order without reallocating at every level.
    for &challenge in point.iter().rev() {
        let (lower, upper) = weights[..2 * filled].split_at_mut(filled);
        #[cfg(feature = "parallel")]
        if filled >= OPENING_PARALLEL_GRAIN {
            upper
                .par_chunks_mut(OPENING_PARALLEL_GRAIN)
                .zip(lower.par_chunks(OPENING_PARALLEL_GRAIN))
                .for_each(|(upper, lower)| {
                    for (out, &value) in upper.iter_mut().zip(lower) {
                        *out = value * challenge;
                    }
                });
            filled *= 2;
            continue;
        }
        for (out, &value) in upper.iter_mut().zip(lower.iter()) {
            *out = value * challenge;
        }
        filled *= 2;
    }
    weights
}

/// Both cross messages share the same four vector slices and one Rayon join.
fn cross_inner_products<E: ShortFieldElement>(left: &[E], right: &[E]) -> (E, E) {
    debug_assert_eq!(left.len(), right.len());
    let half = left.len() / 2;
    let (a, b) = left.split_at(half);
    let (c, d) = right.split_at(half);
    let block = |(((a, b), c), d): (((&[E], &[E]), &[E]), &[E])| {
        a.iter().zip(b).zip(c).zip(d).fold(
            (E::zero(), E::zero()),
            |(left, right), (((&a, &b), &c), &d)| (left + a * d, right + b * c),
        )
    };
    #[cfg(feature = "parallel")]
    if half >= OPENING_PARALLEL_GRAIN {
        return a
            .par_chunks(OPENING_PARALLEL_GRAIN)
            .zip(b.par_chunks(OPENING_PARALLEL_GRAIN))
            .zip(c.par_chunks(OPENING_PARALLEL_GRAIN))
            .zip(d.par_chunks(OPENING_PARALLEL_GRAIN))
            .map(block)
            .reduce(|| (E::zero(), E::zero()), |(a, b), (c, d)| (a + c, b + d));
    }
    block((((a, b), c), d))
}

/// Fold the two vectors together so each recursive round has one scheduling barrier.
fn fold_litebullet_vectors<E: ShortFieldElement>(
    left: &mut [E],
    right: &mut [E],
    left_weight: E,
    right_weight: E,
) {
    let half = left.len() / 2;
    let (a, b) = left.split_at_mut(half);
    let (c, d) = right.split_at_mut(half);
    let block = |(((a, b), c), d): (((&mut [E], &[E]), &mut [E]), &[E])| {
        for (((a, &b), c), &d) in a.iter_mut().zip(b).zip(c).zip(d) {
            *a += b * left_weight;
            *c += d * right_weight;
        }
    };
    #[cfg(feature = "parallel")]
    if half >= OPENING_PARALLEL_GRAIN {
        a.par_chunks_mut(OPENING_PARALLEL_GRAIN)
            .zip(b.par_chunks(OPENING_PARALLEL_GRAIN))
            .zip(c.par_chunks_mut(OPENING_PARALLEL_GRAIN))
            .zip(d.par_chunks(OPENING_PARALLEL_GRAIN))
            .for_each(block);
        return;
    }
    block((((a, b), c), d));
}

fn add_scaled_in_place<E: ShortFieldElement>(target: &mut [E], source: &[E], scale: E) {
    let block = |(target, source): (&mut [E], &[E])| {
        for (target, &source) in target.iter_mut().zip(source) {
            *target += scale * source;
        }
    };
    #[cfg(feature = "parallel")]
    if target.len() >= OPENING_PARALLEL_GRAIN {
        target
            .par_chunks_mut(OPENING_PARALLEL_GRAIN)
            .zip(source.par_chunks(OPENING_PARALLEL_GRAIN))
            .for_each(block);
        return;
    }
    block((target, source));
}

fn tensor_weights_prefix<E: ShortFieldElement>(
    point: &[E],
    prefix_len: usize,
) -> Result<Vec<E>, Error> {
    let full_len = 1usize
        .checked_shl(point.len() as u32)
        .ok_or(Error::InvalidNumberOfVariables)?;
    if prefix_len == 0 || prefix_len > full_len {
        return Err(Error::InvalidNumberOfVariables);
    }
    Ok((0..prefix_len)
        .map(|index| tensor_weight_at(point, index))
        .collect())
}

fn tensor_weight_at<E: ShortFieldElement>(point: &[E], position: usize) -> E {
    let mut result = E::one();
    for bit in 0..point.len() {
        if (position >> bit) & 1 == 1 {
            result *= point[point.len() - bit - 1];
        }
    }
    result
}

fn tensor_inner_product<E: ShortFieldElement>(left: &[E], right: &[E]) -> Result<E, Error> {
    if left.len() != right.len() {
        return Err(Error::InvalidNumberOfVariables);
    }
    Ok(left
        .iter()
        .zip(right)
        .fold(E::one(), |product, (&left, &right)| {
            product * (E::one() + left * right)
        }))
}

fn mle_eval<E: ShortFieldElement>(values: &[E], point: &[E]) -> Option<E> {
    if values.is_empty() || values.len() != (1usize).checked_shl(point.len() as u32)? {
        return None;
    }
    let mut folded = values.to_vec();
    let mut current_len = folded.len();
    for challenge in point {
        let half = current_len / 2;
        let (active, _) = folded.split_at_mut(current_len);
        let (lower, upper) = active.split_at_mut(half);
        for index in 0..half {
            lower[index] += upper[index] * *challenge;
        }
        current_len = half;
    }
    Some(folded[0])
}

fn sample_without_replacement(
    count: usize,
    population: usize,
    seed: [u8; DIGEST_BYTES],
) -> Result<Vec<usize>, Error> {
    if count == 0 || count > population {
        return Err(Error::InvalidNumberOfVariables);
    }
    if count == population {
        return Ok((0..population).collect());
    }
    use std::collections::BTreeMap;
    let mut swaps = BTreeMap::new();
    let mut output = Vec::with_capacity(count);
    let mut counter = 0u64;
    for offset in 0..count {
        let remaining = population - offset;
        let sampled = sample_below(seed, &mut counter, remaining)?;
        let last = remaining - 1;
        let sampled_value = *swaps.get(&sampled).unwrap_or(&sampled);
        let last_value = *swaps.get(&last).unwrap_or(&last);
        swaps.insert(sampled, last_value);
        output.push(sampled_value);
    }
    Ok(output)
}

fn sample_below(seed: [u8; DIGEST_BYTES], counter: &mut u64, upper: usize) -> Result<usize, Error> {
    if upper == 0 {
        return Err(Error::InvalidNumberOfVariables);
    }
    let range = 1u128 << 64;
    let upper_u128 = upper as u128;
    let acceptance_limit = range - range % upper_u128;
    loop {
        let mut hasher = Blake2s256::new();
        hasher.update(b"qafold-short-index/v1");
        hasher.update(seed);
        hasher.update(counter.to_le_bytes());
        *counter = counter.checked_add(1).ok_or(Error::TranscriptError)?;
        let digest = hasher.finalize();
        let candidate =
            u64::from_le_bytes(digest[..8].try_into().map_err(|_| Error::TranscriptError)?) as u128;
        if candidate < acceptance_limit {
            return Ok((candidate % upper_u128) as usize);
        }
    }
}

fn encoded_query_len(row_len: usize, element_bytes: usize, path: &ShortQaMerklePath) -> usize {
    8 + row_len * element_bytes + 8 + path.siblings.len() * DIGEST_BYTES
}

fn encoded_base_rows_len<B: ShortBaseField>(rows: &ShortQaBaseRows<B>) -> usize {
    match rows {
        ShortQaBaseRows::Partial(queries) => {
            1 + 8
                + queries
                    .iter()
                    .map(|query| {
                        encoded_query_len(query.row.len(), B::CANONICAL_BYTES, &query.path)
                    })
                    .sum::<usize>()
        }
        ShortQaBaseRows::AllRows {
            encoded_entries, ..
        } => 1 + 8 + 8 + 8 + encoded_entries.len() * B::CANONICAL_BYTES,
    }
}

fn encoded_extension_rows_len<B: ShortBaseField>(rows: &ShortQaExtensionRows<B>) -> usize {
    match rows {
        ShortQaExtensionRows::Partial(queries) => {
            1 + 8
                + queries
                    .iter()
                    .map(|query| {
                        encoded_query_len(
                            query.row.len(),
                            B::Extension::canonical_byte_len(),
                            &query.path,
                        )
                    })
                    .sum::<usize>()
        }
        ShortQaExtensionRows::AllRows {
            encoded_entries, ..
        } => 1 + 8 + 8 + 8 + encoded_entries.len() * B::Extension::canonical_byte_len(),
    }
}

fn base_rows_query_count<B: ShortBaseField>(rows: &ShortQaBaseRows<B>) -> usize {
    match rows {
        ShortQaBaseRows::Partial(queries) => queries.len(),
        ShortQaBaseRows::AllRows { n_rows, .. } => *n_rows,
    }
}

fn extension_rows_query_count<B: ShortBaseField>(rows: &ShortQaExtensionRows<B>) -> usize {
    match rows {
        ShortQaExtensionRows::Partial(queries) => queries.len(),
        ShortQaExtensionRows::AllRows { n_rows, .. } => *n_rows,
    }
}

fn write_base_rows<B: ShortBaseField>(
    output: &mut Vec<u8>,
    rows: &ShortQaBaseRows<B>,
) -> Result<(), Error> {
    match rows {
        ShortQaBaseRows::Partial(queries) => {
            output.push(0);
            write_len(output, queries.len())?;
            for query in queries {
                write_base_query::<B>(output, query)?;
            }
        }
        ShortQaBaseRows::AllRows {
            encoded_entries,
            n_rows,
            n_cols,
        } => {
            output.push(1);
            write_len(output, *n_rows)?;
            write_len(output, *n_cols)?;
            write_len(output, encoded_entries.len())?;
            for value in encoded_entries {
                write_base::<B>(output, *value)?;
            }
        }
    }
    Ok(())
}

fn write_extension_rows<B: ShortBaseField>(
    output: &mut Vec<u8>,
    rows: &ShortQaExtensionRows<B>,
) -> Result<(), Error> {
    match rows {
        ShortQaExtensionRows::Partial(queries) => {
            output.push(0);
            write_len(output, queries.len())?;
            for query in queries {
                write_extension_query::<B>(output, query)?;
            }
        }
        ShortQaExtensionRows::AllRows {
            encoded_entries,
            n_rows,
            n_cols,
        } => {
            output.push(1);
            write_len(output, *n_rows)?;
            write_len(output, *n_cols)?;
            write_len(output, encoded_entries.len())?;
            for value in encoded_entries {
                write_extension::<B>(output, *value)?;
            }
        }
    }
    Ok(())
}

fn write_base_query<B: ShortBaseField>(
    output: &mut Vec<u8>,
    query: &ShortQaBaseRowQuery<B>,
) -> Result<(), Error> {
    write_len(output, query.row.len())?;
    for value in &query.row {
        write_base::<B>(output, *value)?;
    }
    write_path(output, &query.path)
}

fn write_extension_query<B: ShortBaseField>(
    output: &mut Vec<u8>,
    query: &ShortQaExtensionRowQuery<B>,
) -> Result<(), Error> {
    write_len(output, query.row.len())?;
    for value in &query.row {
        write_extension::<B>(output, *value)?;
    }
    write_path(output, &query.path)
}

fn write_path(output: &mut Vec<u8>, path: &ShortQaMerklePath) -> Result<(), Error> {
    write_len(output, path.siblings.len())?;
    for sibling in &path.siblings {
        output.extend_from_slice(sibling);
    }
    Ok(())
}

fn write_base<B: ShortBaseField>(output: &mut Vec<u8>, value: B) -> Result<(), Error> {
    let start = output.len();
    output.resize(start + B::CANONICAL_BYTES, 0);
    value.write_canonical_bytes(&mut output[start..])
}

fn write_extension<B: ShortBaseField>(
    output: &mut Vec<u8>,
    value: B::Extension,
) -> Result<(), Error> {
    let start = output.len();
    output.resize(start + B::Extension::canonical_byte_len(), 0);
    value.write_canonical_bytes(&mut output[start..])
}

fn write_len(output: &mut Vec<u8>, value: usize) -> Result<(), Error> {
    let value = u64::try_from(value).map_err(|_| Error::EncodingError)?;
    output.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn extension_bytes<B: ShortBaseField>(value: B::Extension) -> Result<Vec<u8>, Error> {
    let mut output = vec![0u8; B::Extension::canonical_byte_len()];
    value.write_canonical_bytes(&mut output)?;
    Ok(output)
}

fn reserve_decode_vec<T>(len: usize) -> Result<Vec<T>, Error> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| Error::InvalidCommitment)?;
    Ok(values)
}

struct ByteReader<'a> {
    input: &'a [u8],
    cursor: usize,
}

impl<'a> ByteReader<'a> {
    fn new(input: &'a [u8]) -> Self {
        Self { input, cursor: 0 }
    }

    fn take(&mut self, len: usize) -> Result<&'a [u8], Error> {
        let end = self
            .cursor
            .checked_add(len)
            .ok_or(Error::InvalidCommitment)?;
        let bytes = self
            .input
            .get(self.cursor..end)
            .ok_or(Error::InvalidCommitment)?;
        self.cursor = end;
        Ok(bytes)
    }

    fn read_len(&mut self) -> Result<usize, Error> {
        let bytes: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| Error::InvalidCommitment)?;
        usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| Error::InvalidCommitment)
    }

    fn read_base<B: ShortBaseField>(&mut self) -> Result<B, Error> {
        B::from_canonical_bytes(self.take(B::CANONICAL_BYTES)?).ok_or(Error::InvalidCommitment)
    }

    fn read_extension<B: ShortBaseField>(&mut self) -> Result<B::Extension, Error> {
        B::Extension::from_canonical_bytes(self.take(B::Extension::canonical_byte_len())?)
            .ok_or(Error::InvalidCommitment)
    }

    fn read_path(&mut self) -> Result<ShortQaMerklePath, Error> {
        let len = self.read_len()?;
        if len > self.remaining() / DIGEST_BYTES {
            return Err(Error::InvalidCommitment);
        }
        let mut siblings = reserve_decode_vec(len)?;
        for _ in 0..len {
            siblings.push(
                self.take(DIGEST_BYTES)?
                    .try_into()
                    .map_err(|_| Error::InvalidCommitment)?,
            );
        }
        Ok(ShortQaMerklePath { siblings })
    }

    fn read_base_query<B: ShortBaseField>(&mut self) -> Result<ShortQaBaseRowQuery<B>, Error> {
        let len = self.read_len()?;
        let row_bytes = self
            .remaining()
            .checked_sub(8)
            .ok_or(Error::InvalidCommitment)?;
        if len > row_bytes / B::CANONICAL_BYTES {
            return Err(Error::InvalidCommitment);
        }
        let mut row = reserve_decode_vec(len)?;
        for _ in 0..len {
            row.push(self.read_base::<B>()?);
        }
        let path = self.read_path()?;
        Ok(ShortQaBaseRowQuery { row, path })
    }

    fn read_extension_query<B: ShortBaseField>(
        &mut self,
    ) -> Result<ShortQaExtensionRowQuery<B>, Error> {
        let len = self.read_len()?;
        let element_bytes = B::Extension::canonical_byte_len();
        let row_bytes = self
            .remaining()
            .checked_sub(8)
            .ok_or(Error::InvalidCommitment)?;
        if len > row_bytes / element_bytes {
            return Err(Error::InvalidCommitment);
        }
        let mut row = reserve_decode_vec(len)?;
        for _ in 0..len {
            row.push(self.read_extension::<B>()?);
        }
        let path = self.read_path()?;
        Ok(ShortQaExtensionRowQuery { row, path })
    }

    fn read_base_rows<B: ShortBaseField>(&mut self) -> Result<ShortQaBaseRows<B>, Error> {
        let mode = *self.take(1)?.first().ok_or(Error::InvalidCommitment)?;
        match mode {
            0 => {
                let len = self.read_len()?;
                if len > self.remaining() / 16 {
                    return Err(Error::InvalidCommitment);
                }
                let mut queries = reserve_decode_vec(len)?;
                for _ in 0..len {
                    queries.push(self.read_base_query::<B>()?);
                }
                Ok(ShortQaBaseRows::Partial(queries))
            }
            1 => {
                let n_rows = self.read_len()?;
                let n_cols = self.read_len()?;
                let len = self.read_len()?;
                if len != n_rows.checked_mul(n_cols).ok_or(Error::InvalidCommitment)?
                    || len > self.remaining() / B::CANONICAL_BYTES
                {
                    return Err(Error::InvalidCommitment);
                }
                let mut encoded_entries = reserve_decode_vec(len)?;
                for _ in 0..len {
                    encoded_entries.push(self.read_base::<B>()?);
                }
                Ok(ShortQaBaseRows::AllRows {
                    encoded_entries,
                    n_rows,
                    n_cols,
                })
            }
            _ => Err(Error::InvalidCommitment),
        }
    }

    fn read_extension_rows<B: ShortBaseField>(&mut self) -> Result<ShortQaExtensionRows<B>, Error> {
        let mode = *self.take(1)?.first().ok_or(Error::InvalidCommitment)?;
        match mode {
            0 => {
                let len = self.read_len()?;
                if len > self.remaining() / 16 {
                    return Err(Error::InvalidCommitment);
                }
                let mut queries = reserve_decode_vec(len)?;
                for _ in 0..len {
                    queries.push(self.read_extension_query::<B>()?);
                }
                Ok(ShortQaExtensionRows::Partial(queries))
            }
            1 => {
                let n_rows = self.read_len()?;
                let n_cols = self.read_len()?;
                let len = self.read_len()?;
                let element_bytes = B::Extension::canonical_byte_len();
                if len != n_rows.checked_mul(n_cols).ok_or(Error::InvalidCommitment)?
                    || len > self.remaining() / element_bytes
                {
                    return Err(Error::InvalidCommitment);
                }
                let mut encoded_entries = reserve_decode_vec(len)?;
                for _ in 0..len {
                    encoded_entries.push(self.read_extension::<B>()?);
                }
                Ok(ShortQaExtensionRows::AllRows {
                    encoded_entries,
                    n_rows,
                    n_cols,
                })
            }
            _ => Err(Error::InvalidCommitment),
        }
    }

    fn remaining(&self) -> usize {
        self.input.len().saturating_sub(self.cursor)
    }

    fn is_empty(&self) -> bool {
        self.cursor == self.input.len()
    }
}

#[path = "qafold_short_decide.rs"]
mod decide;
pub use decide::{ShortEvalClaim, ShortEvalWithPoint, ShortMultiEvalReduction};

#[path = "qafold_short_setup.rs"]
mod setup;
pub use setup::{
    ShortQaDerivedParameters, ShortQaFoldSetupCache, ShortQaFoldSetupCommitments,
    ShortQaFoldSetupLevel, ShortQaSetupRowQuery, ShortQaSetupRows, ShortQaSetupShape,
};

#[path = "qafold_short_decide_wire.rs"]
mod decide_wire;
pub use decide_wire::{ShortQaDecideProof, ShortQaDecideRound};

#[path = "qafold_short_decide_carry.rs"]
mod decide_carry;

#[path = "qafold_short_decide_protocol.rs"]
mod decide_protocol;

#[path = "qafold_short_decide_driver.rs"]
mod decide_driver;

#[cfg(test)]
mod engineering_tests {
    use super::*;
    use crate::FullPrime65521Ext8 as F;
    use rand::{SeedableRng, rngs::StdRng};

    fn reference_tree(leaves: Vec<[u8; DIGEST_BYTES]>) -> Blake2sMerkleTree {
        let mut levels = vec![leaves];
        while levels.last().unwrap().len() > 1 {
            levels.push(
                levels
                    .last()
                    .unwrap()
                    .chunks_exact(2)
                    .map(|pair| hash_inner(&pair[0], &pair[1]))
                    .collect(),
            );
        }
        Blake2sMerkleTree { levels }
    }

    fn check_base_merkle_rows<B: ShortBaseField>(rows: usize, cols: usize, rng: &mut StdRng) {
        let entries: Vec<B> = (0..rows * cols).map(|_| B::sample_uniform(rng)).collect();
        let reference_leaves: Vec<_> = (0..rows)
            .map(|row| hash_base_encoded_row::<B>(&entries, rows, cols, row).unwrap())
            .collect();
        let reference = reference_tree(reference_leaves);
        let actual = build_base_tree::<B>(&entries, rows, cols).unwrap();
        assert_eq!(actual.levels, reference.levels);
        for row in [0, rows / 2, rows - 1] {
            let path = actual.path(row).unwrap();
            assert_eq!(path, reference.path(row).unwrap());
            assert!(verify_merkle_path(
                actual.levels[0][row],
                row,
                rows,
                &path,
                &actual.root()
            ));
            let mut changed = actual.root();
            changed[0] ^= 1;
            assert!(!verify_merkle_path(
                actual.levels[0][row],
                row,
                rows,
                &path,
                &changed
            ));
        }
    }

    #[test]
    fn reusable_row_hashing_preserves_base_roots_and_paths() {
        let mut rng = StdRng::seed_from_u64(0x726f775f68617368);
        for (rows, cols) in [(1, 1), (2, 3), (64, 8), (4096, 3), (8192, 32)] {
            check_base_merkle_rows::<crate::Prime65521>(rows, cols, &mut rng);
            check_base_merkle_rows::<F>(rows, cols, &mut rng);
        }
    }

    #[test]
    fn reusable_row_hashing_preserves_extension_roots_and_domain() {
        let mut rng = StdRng::seed_from_u64(0x6578745f68617368);
        for (rows, cols) in [(1, 1), (32, 3), (4096, 4)] {
            let entries: Vec<_> = (0..rows * cols)
                .map(|_| F::sample_uniform(&mut rng).0)
                .collect();
            let leaves = (0..rows)
                .map(|row| {
                    hash_extension_encoded_row::<crate::Prime65521>(&entries, rows, cols, row)
                        .unwrap()
                })
                .collect();
            let reference = reference_tree(leaves);
            let actual = build_extension_tree::<crate::Prime65521>(&entries, rows, cols).unwrap();
            assert_eq!(actual.levels, reference.levels);
            for row in [0, rows - 1] {
                assert_eq!(actual.path(row).unwrap(), reference.path(row).unwrap());
            }
            let full: Vec<F> = entries.into_iter().map(F::from).collect();
            let base = build_base_tree::<F>(&full, rows, cols).unwrap();
            assert_ne!(
                base.root(),
                actual.root(),
                "base and extension domains must remain distinct"
            );
        }
    }

    #[test]
    fn reusable_row_hashing_propagates_shape_and_encoding_errors() {
        assert!(build_base_tree::<F>(&[], 0, 1).is_err());
        assert!(build_base_tree::<F>(&[F::one()], 2, 1).is_err());
        let entries = vec![F::one(); 4096];
        let result = hash_encoded_rows(&entries, 4096, 1, 16, LEAF_BASE_DOMAIN, |_, _| {
            Err(Error::EncodingError)
        });
        assert_eq!(result, Err(Error::EncodingError));
    }

    #[test]
    fn opening_kernels_match_serial_reference() {
        let run = || {
            let mut rng = StdRng::seed_from_u64(0x4348_554e_4b53);
            for log in [0, 1, 7, 12, 13, 15] {
                let left = (0..1 << log)
                    .map(|_| F::sample_uniform(&mut rng))
                    .collect::<Vec<_>>();
                let right = (0..1 << log)
                    .map(|_| F::sample_uniform(&mut rng))
                    .collect::<Vec<_>>();
                let reference = left
                    .iter()
                    .zip(&right)
                    .fold(F::zero(), |s, (&a, &b)| s + a * b);
                assert_eq!(inner_product(&left, &right).unwrap(), reference);
                assert_eq!(mixed_inner_product::<F>(&left, &right).unwrap(), reference);
                let scale = F::sample_uniform(&mut rng);
                let mut accumulated = left.clone();
                add_scaled_in_place(&mut accumulated, &right, scale);
                assert_eq!(
                    accumulated,
                    left.iter()
                        .zip(&right)
                        .map(|(&a, &b)| a + scale * b)
                        .collect::<Vec<_>>()
                );
                if log > 0 {
                    let half = left.len() / 2;
                    let cross = (0..half).fold((F::zero(), F::zero()), |(a, b), i| {
                        (a + left[i] * right[i + half], b + left[i + half] * right[i])
                    });
                    assert_eq!(cross_inner_products(&left, &right), cross);
                    let mut folded_left = left.clone();
                    let mut folded_right = right.clone();
                    fold_litebullet_vectors(
                        &mut folded_left,
                        &mut folded_right,
                        scale,
                        scale + F::one(),
                    );
                    for i in 0..half {
                        assert_eq!(folded_left[i], left[i] + left[i + half] * scale);
                        assert_eq!(
                            folded_right[i],
                            right[i] + right[i + half] * (scale + F::one())
                        );
                    }
                }
                let point = (0..log)
                    .map(|_| F::sample_uniform(&mut rng))
                    .collect::<Vec<_>>();
                let mut legacy = vec![F::one()];
                for &challenge in &point {
                    legacy = legacy
                        .into_iter()
                        .flat_map(|weight| [weight, weight * challenge])
                        .collect();
                }
                assert_eq!(tensor_weights(&point), legacy);
            }
            assert!(inner_product(&[F::one()], &[]).is_err());
        };
        #[cfg(feature = "parallel")]
        for threads in [1, 4] {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(run);
        }
        #[cfg(not(feature = "parallel"))]
        run();
    }

    #[test]
    fn native_opening_and_decide_wire_fixture() {
        let run = || {
            let pc = ShortQaFoldPC::<F>::new(1 << 16, 16, 2, 80).unwrap();
            let mut rng = StdRng::seed_from_u64(0x7061_7261_6c6c_656c);
            let (pp, setup) = pc.setup(&mut rng).unwrap();
            let witness = (0..1 << 16).map(|_| F::sample_uniform(&mut rng)).collect();
            let point = (0..16)
                .map(|_| F::sample_uniform(&mut rng))
                .collect::<Vec<_>>();
            let (commitment, cache) = pc.commit_owned(&pp, witness).unwrap();
            let (value, opening, pending) = pc
                .open_with_reduction(&pp, &commitment, &point, cache)
                .unwrap();
            let decide = pc.prove_decide(&pp, &setup, &pending).unwrap();
            let verified = pc
                .verify_opening_reduction(&pp, &commitment, &point, value, &opening)
                .unwrap();
            assert_eq!(pending, verified);
            assert!(pc.verify_decide(&pp, &setup, &verified, &decide).unwrap());
            let mut hasher = Blake2s256::new();
            hasher.update(commitment.root);
            hasher.update(opening.to_bytes().unwrap());
            hasher.update(decide.to_bytes().unwrap());
            let digest: [u8; 32] = hasher.finalize().into();
            digest
        };
        #[cfg(feature = "parallel")]
        let digest = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap()
            .install(run);
        #[cfg(not(feature = "parallel"))]
        let digest = run();
        assert_eq!(
            digest,
            [
                0xbb, 0x49, 0x93, 0x47, 0x54, 0x32, 0xe1, 0xfb, 0x1a, 0xdb, 0x6e, 0x01, 0x81, 0xf8,
                0x0a, 0x48, 0xd6, 0x25, 0x06, 0x9a, 0x88, 0x18, 0x2a, 0x6c, 0x72, 0x17, 0x15, 0xb6,
                0x87, 0xf6, 0xd1, 0x77
            ]
        );
    }
}
