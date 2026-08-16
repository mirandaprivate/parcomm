use std::convert::TryFrom;
use std::fmt::{Display, Formatter};
use std::io::{self, Read, Seek, SeekFrom, Write};
use std::str::FromStr;

use ark_ff::PrimeField;
use ark_serialize::CanonicalSerialize;
use sha2::{Digest, Sha256};

use crate::hdr1cs::global_mlp::GlobalBlockId;
use crate::protocols::bucketed_ip::{BvaError, BvaLayout};
use crate::protocols::virtualpcs::{BvaI32Source, StackedI32Source};
use crate::utils::fiat_shamir::FiatShamir;
use crate::utils::heterogeneous::{HeteroMatrixError, MatrixBlock, MatrixDims, PaddedDomain};

// Witness construction is internal until the public protocol exposes only
// the final prove/verify seam; the statement already consumes its manifest.
#[allow(dead_code)]
mod relation;

pub use relation::{
    Gpt2GlobalRelationParameters, Gpt2GlobalRelationProof, Gpt2LassoProof, Gpt2Proof,
    Gpt2ProofDecodeError, Gpt2PublicSupportLookupProof, Gpt2ScalableProof,
};

const GPT2_Q12_MAGIC: [u8; 8] = *b"PR1GPT2\0";
const GPT2_Q12_LEGACY_VERSION: u32 = 1;
const GPT2_Q12_VERSION: u32 = 2;
pub const GPT2_HD_R1CS_PROTOCOL_VERSION: u32 = 1;
const GPT2_PUBLIC_STATEMENT_MAGIC: [u8; 8] = *b"ZKSGP2S\0";
const GPT2_PUBLIC_STATEMENT_VERSION: u32 = 1;
const GPT2_PUBLIC_STATEMENT_MAX_BYTES: usize = 16 * 1024 * 1024;
const GPT2_PUBLIC_STATEMENT_MAX_VECTOR_ENTRIES: usize = 1 << 20;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gpt2Variant {
    Small,
    Medium,
    Large,
    Xl,
}

impl Gpt2Variant {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Small => "small",
            Self::Medium => "medium",
            Self::Large => "large",
            Self::Xl => "xl",
        }
    }
}

impl FromStr for Gpt2Variant {
    type Err = Gpt2VariantParseError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "small" => Ok(Self::Small),
            "medium" => Ok(Self::Medium),
            "large" => Ok(Self::Large),
            "xl" => Ok(Self::Xl),
            _ => Err(Gpt2VariantParseError),
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gpt2VariantParseError;

impl Display for Gpt2VariantParseError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "expected one of: small, medium, large, xl")
    }
}

impl std::error::Error for Gpt2VariantParseError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gpt2Model {
    pub blocks: usize,
    pub hidden_width: usize,
    pub attention_heads: usize,
    pub mlp_width: usize,
    pub vocabulary: usize,
    pub maximum_positions: usize,
}

impl Gpt2Model {
    pub const fn small() -> Self {
        Self::for_variant(Gpt2Variant::Small)
    }

    pub const fn for_variant(variant: Gpt2Variant) -> Self {
        let (blocks, hidden_width, attention_heads) = match variant {
            Gpt2Variant::Small => (12, 768, 12),
            Gpt2Variant::Medium => (24, 1_024, 16),
            Gpt2Variant::Large => (36, 1_280, 20),
            Gpt2Variant::Xl => (48, 1_600, 25),
        };
        Self {
            blocks,
            hidden_width,
            attention_heads,
            mlp_width: hidden_width * 4,
            vocabulary: 50_257,
            maximum_positions: 1_024,
        }
    }

    pub fn head_width(&self) -> usize {
        self.hidden_width / self.attention_heads
    }

    /// Physical checkpoint tensors. The tied LM head is intentionally absent:
    /// it is a transposed view of `token_embedding`, not another payload.
    pub fn parameter_tensors(&self) -> Result<Vec<TensorSpec>, Gpt2InventoryError> {
        if self.attention_heads == 0 || self.hidden_width % self.attention_heads != 0 {
            return Err(Gpt2InventoryError::InvalidHeadLayout);
        }
        let mut tensors = vec![
            TensorSpec::matrix(
                "token_embedding",
                ParameterKind::Embedding,
                self.vocabulary,
                self.hidden_width,
            )?,
            TensorSpec::matrix(
                "position_embedding",
                ParameterKind::Embedding,
                self.maximum_positions,
                self.hidden_width,
            )?,
        ];

        for block in 0..self.blocks {
            let prefix = format!("block.{block}");
            tensors.extend([
                TensorSpec::vector(
                    format!("{prefix}.ln_1.weight"),
                    ParameterKind::LayerNorm,
                    self.hidden_width,
                )?,
                TensorSpec::vector(
                    format!("{prefix}.ln_1.bias"),
                    ParameterKind::LayerNorm,
                    self.hidden_width,
                )?,
                TensorSpec::matrix(
                    format!("{prefix}.attn.c_attn.weight"),
                    ParameterKind::Attention,
                    self.hidden_width,
                    self.hidden_width
                        .checked_mul(3)
                        .ok_or(Gpt2InventoryError::Overflow)?,
                )?,
                TensorSpec::vector(
                    format!("{prefix}.attn.c_attn.bias"),
                    ParameterKind::Attention,
                    self.hidden_width
                        .checked_mul(3)
                        .ok_or(Gpt2InventoryError::Overflow)?,
                )?,
                TensorSpec::matrix(
                    format!("{prefix}.attn.c_proj.weight"),
                    ParameterKind::Attention,
                    self.hidden_width,
                    self.hidden_width,
                )?,
                TensorSpec::vector(
                    format!("{prefix}.attn.c_proj.bias"),
                    ParameterKind::Attention,
                    self.hidden_width,
                )?,
                TensorSpec::vector(
                    format!("{prefix}.ln_2.weight"),
                    ParameterKind::LayerNorm,
                    self.hidden_width,
                )?,
                TensorSpec::vector(
                    format!("{prefix}.ln_2.bias"),
                    ParameterKind::LayerNorm,
                    self.hidden_width,
                )?,
                TensorSpec::matrix(
                    format!("{prefix}.mlp.c_fc.weight"),
                    ParameterKind::Mlp,
                    self.hidden_width,
                    self.mlp_width,
                )?,
                TensorSpec::vector(
                    format!("{prefix}.mlp.c_fc.bias"),
                    ParameterKind::Mlp,
                    self.mlp_width,
                )?,
                TensorSpec::matrix(
                    format!("{prefix}.mlp.c_proj.weight"),
                    ParameterKind::Mlp,
                    self.mlp_width,
                    self.hidden_width,
                )?,
                TensorSpec::vector(
                    format!("{prefix}.mlp.c_proj.bias"),
                    ParameterKind::Mlp,
                    self.hidden_width,
                )?,
            ]);
        }
        tensors.extend([
            TensorSpec::vector("ln_f.weight", ParameterKind::LayerNorm, self.hidden_width)?,
            TensorSpec::vector("ln_f.bias", ParameterKind::LayerNorm, self.hidden_width)?,
        ]);
        Ok(tensors)
    }

    pub fn physical_parameter_entries(&self) -> Result<usize, Gpt2InventoryError> {
        self.parameter_tensors()?
            .iter()
            .try_fold(0_usize, |sum, tensor| {
                sum.checked_add(tensor.entries)
                    .ok_or(Gpt2InventoryError::Overflow)
            })
    }

    pub fn lm_head_entries(&self) -> usize {
        self.vocabulary * self.hidden_width
    }

    pub const fn lm_head_is_tied_view(&self) -> bool {
        true
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ParameterKind {
    Embedding,
    LayerNorm,
    Attention,
    Mlp,
}

impl Display for ParameterKind {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Embedding => write!(formatter, "embedding"),
            Self::LayerNorm => write!(formatter, "layer_norm"),
            Self::Attention => write!(formatter, "attention"),
            Self::Mlp => write!(formatter, "mlp"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TensorSpec {
    pub name: String,
    pub kind: ParameterKind,
    pub rows: usize,
    pub cols: usize,
    pub entries: usize,
}

impl TensorSpec {
    pub fn matrix(
        name: impl Into<String>,
        kind: ParameterKind,
        rows: usize,
        cols: usize,
    ) -> Result<Self, Gpt2InventoryError> {
        let entries = rows.checked_mul(cols).ok_or(Gpt2InventoryError::Overflow)?;
        if entries == 0 {
            return Err(Gpt2InventoryError::ZeroSizedTensor);
        }
        Ok(Self {
            name: name.into(),
            kind,
            rows,
            cols,
            entries,
        })
    }

    pub fn vector(
        name: impl Into<String>,
        kind: ParameterKind,
        len: usize,
    ) -> Result<Self, Gpt2InventoryError> {
        Self::matrix(name, kind, 1, len)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gpt2InventoryError {
    InvalidHeadLayout,
    ZeroSizedTensor,
    Overflow,
}

impl Display for Gpt2InventoryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for Gpt2InventoryError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gpt2Q12Metadata {
    pub format_version: u32,
    pub model: Gpt2Model,
    pub fractional_bits: u32,
    pub parameter_entries: u64,
    pub token_count: usize,
    pub checkpoint_sha256: [u8; 32],
    pub tokenizer_sha256: [u8; 32],
}

/// The exact integer semantics proved by the GPT-2 Q12 relation.  Keeping the
/// nonlinear rules behind this value makes the reference executor and the
/// proof compiler share one versioned definition instead of duplicating
/// rounding behavior.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gpt2Q12Semantics {
    fractional_bits: u32,
}

impl Gpt2Q12Semantics {
    pub const fn exact() -> Self {
        Self {
            fractional_bits: 12,
        }
    }

    pub const fn fractional_bits(self) -> u32 {
        self.fractional_bits
    }

    pub const fn fixed_point_unit(self) -> i64 {
        1_i64 << self.fractional_bits
    }

    /// Stable integer Softmax semantics. The reciprocal is rounded down, so
    /// the base probabilities cannot exceed one unit in aggregate. Any
    /// residual mass is assigned to the first maximum score, making every row
    /// sum exactly to the Q12 unit without an ambiguous tie rule.
    pub fn softmax_row(self, scores: &[i64]) -> Result<Vec<i64>, Gpt2CheckpointError> {
        let maximum = *scores
            .iter()
            .max()
            .ok_or(Gpt2CheckpointError::ProtocolIncomplete)?;
        let maximum_index = scores
            .iter()
            .position(|score| *score == maximum)
            .ok_or(Gpt2CheckpointError::ProtocolIncomplete)?;
        let scale = self.fixed_point_unit();
        let exponentials = scores
            .iter()
            .map(|score| q12_exp(score - maximum, scale))
            .collect::<Result<Vec<_>, _>>()?;
        let denominator = exponentials.iter().try_fold(0_i64, |sum, &value| {
            sum.checked_add(value)
                .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
        })?;
        let reciprocal = q12_reciprocal(denominator, scores.len(), scale)?;
        let mut probabilities = exponentials
            .iter()
            .map(|value| {
                div_floor_i128(
                    i128::from(*value) * i128::from(reciprocal),
                    i128::from(scale),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let base_sum = probabilities.iter().try_fold(0_i64, |sum, &value| {
            sum.checked_add(value)
                .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
        })?;
        let residual = scale
            .checked_sub(base_sum)
            .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
        if residual < 0 {
            return Err(Gpt2CheckpointError::ProtocolIncomplete);
        }
        probabilities[maximum_index] = probabilities[maximum_index]
            .checked_add(residual)
            .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
        Ok(probabilities)
    }
}

/// Public data that defines one exact GPT-2 HD-R1CS claim. Architecture is
/// public in this first end-to-end mode: the verifier reconstructs the
/// operation manifest and BVA layout and checks their digests instead of
/// trusting prover-supplied routing metadata.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gpt2PublicStatement {
    pub protocol_version: u32,
    pub model: Gpt2Model,
    pub fractional_bits: u32,
    pub checkpoint_sha256: [u8; 32],
    pub tokenizer_sha256: [u8; 32],
    pub token_ids: Vec<usize>,
    pub public_logits: Vec<i64>,
    pub operation_manifest_digest: [u8; 32],
    pub bva_layout_digest: [u8; 32],
    pub relation_manifest_digest: [u8; 32],
    pub lookup_registry_digest: [u8; 32],
}

/// Domain-separated Fiat--Shamir schedule shared by the GPT-2 prover and
/// verifier.  Labels are absorbed before every message and challenge so that
/// reordering two otherwise well-formed subproofs changes all subsequent
/// challenges.
#[allow(dead_code)]
#[derive(Clone)]
pub(super) struct Gpt2ProofTranscript {
    fs: FiatShamir,
}

#[allow(dead_code)]
impl Gpt2ProofTranscript {
    pub(super) fn new(statement: &Gpt2PublicStatement) -> Self {
        let mut fs = FiatShamir::new();
        fs.push(&b"zksmart.gpt2.hd-r1cs.proof.v1".to_vec());
        fs.push(&statement.transcript_digest());
        Self { fs }
    }

    pub(super) fn absorb<T: CanonicalSerialize>(&mut self, label: &[u8], value: &T) {
        self.fs.push(&label.to_vec());
        self.fs.push(value);
    }

    pub(super) fn challenge<F: PrimeField>(&mut self, label: &[u8]) -> F {
        self.fs.push(&label.to_vec());
        self.fs.gen_challenge()
    }
}

/// Public size summary of the compiler-generated global HD-R1CS relation.
/// The compiler itself remains internal; verifiers deterministically
/// reconstruct this shape from the public GPT-2 model and token count.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gpt2HdR1csShape {
    pub linear_relations: usize,
    pub bilinear_relations: usize,
    pub tagged_lookup_relations: usize,
    pub io_equation_scalar_rows: usize,
    pub linear_equation_scalar_rows: usize,
    pub bilinear_equation_scalar_rows: usize,
    pub tagged_lookup_equation_scalar_rows: usize,
    pub equation_bva_logical_rows: usize,
    pub maximum_live_equation_row_entries: usize,
    pub equation_materialized_padding_entries: usize,
    pub auxiliary_blocks: usize,
    pub auxiliary_entries: usize,
    pub resident_auxiliary_blocks: usize,
    pub resident_auxiliary_entries: usize,
    pub streamed_range_limb_blocks: usize,
    pub streamed_range_limb_entries: usize,
    pub maximum_streamed_range_limb_column_entries: usize,
    pub global_witness_blocks: usize,
    pub global_physical_entries: usize,
    pub global_bva_logical_entries: usize,
    pub materialized_padding_entries: usize,
}

impl Gpt2HdR1csShape {
    pub const fn operation_family_count(self) -> usize {
        3
    }

    pub const fn top_level_equation_count(self) -> usize {
        1 + self.operation_family_count()
    }
}

impl Gpt2PublicStatement {
    pub fn new(
        metadata: &Gpt2Q12Metadata,
        token_ids: &[usize],
        public_logits: &[i64],
    ) -> Result<Self, Gpt2CheckpointError> {
        if metadata.format_version != GPT2_Q12_VERSION {
            return Err(Gpt2CheckpointError::LegacyCheckpointSemantics(
                metadata.format_version,
            ));
        }
        if token_ids.len() != metadata.token_count {
            return Err(Gpt2CheckpointError::PublicTokenCountMismatch {
                expected: metadata.token_count,
                actual: token_ids.len(),
            });
        }
        let layout = Gpt2GlobalLayout::compile(metadata)?;
        let relation_manifest =
            relation::Gpt2PrimitiveRelationManifest::compile(metadata, &layout)?;
        let statement = Self {
            protocol_version: GPT2_HD_R1CS_PROTOCOL_VERSION,
            model: metadata.model,
            fractional_bits: metadata.fractional_bits,
            checkpoint_sha256: metadata.checkpoint_sha256,
            tokenizer_sha256: metadata.tokenizer_sha256,
            token_ids: token_ids.to_vec(),
            public_logits: public_logits.to_vec(),
            operation_manifest_digest: digest_operation_manifest(metadata)?,
            bva_layout_digest: relation_manifest.bva_digest(&layout)?,
            relation_manifest_digest: relation_manifest.digest(),
            lookup_registry_digest: digest_lookup_registry(),
        };
        statement.validate()?;
        Ok(statement)
    }

    pub fn validate(&self) -> Result<(), Gpt2CheckpointError> {
        if self.protocol_version != GPT2_HD_R1CS_PROTOCOL_VERSION {
            return Err(Gpt2CheckpointError::UnsupportedProofProtocolVersion(
                self.protocol_version,
            ));
        }
        if self.fractional_bits != Gpt2Q12Semantics::exact().fractional_bits() {
            return Err(Gpt2CheckpointError::UnsupportedFractionalBits(
                self.fractional_bits,
            ));
        }
        if self.token_ids.is_empty() || self.token_ids.len() > self.model.maximum_positions {
            return Err(Gpt2CheckpointError::InvalidTokenCount(self.token_ids.len()));
        }
        if let Some((index, &token)) = self
            .token_ids
            .iter()
            .enumerate()
            .find(|(_, token)| **token >= self.model.vocabulary)
        {
            return Err(Gpt2CheckpointError::InvalidTokenId { index, token });
        }
        if self.public_logits.len() != self.model.vocabulary {
            return Err(Gpt2CheckpointError::PublicLogitCountMismatch {
                expected: self.model.vocabulary,
                actual: self.public_logits.len(),
            });
        }
        let parameter_entries = self.model.physical_parameter_entries()? as u64;
        let metadata = Gpt2Q12Metadata {
            format_version: GPT2_Q12_VERSION,
            model: self.model,
            fractional_bits: self.fractional_bits,
            parameter_entries,
            token_count: self.token_ids.len(),
            checkpoint_sha256: self.checkpoint_sha256,
            tokenizer_sha256: self.tokenizer_sha256,
        };
        let layout = Gpt2GlobalLayout::compile(&metadata)?;
        let relation_manifest =
            relation::Gpt2PrimitiveRelationManifest::compile(&metadata, &layout)?;
        if self.operation_manifest_digest != digest_operation_manifest(&metadata)?
            || self.bva_layout_digest != relation_manifest.bva_digest(&layout)?
            || self.relation_manifest_digest != relation_manifest.digest()
            || self.lookup_registry_digest != digest_lookup_registry()
        {
            return Err(Gpt2CheckpointError::PublicStatementDigestMismatch);
        }
        Ok(())
    }

    /// Canonical public-statement artifact consumed by the standalone
    /// verifier. The framing is fixed-width except for explicitly bounded
    /// token and logit vectors.
    pub fn to_bytes_exact(&self) -> Result<Vec<u8>, Gpt2CheckpointError> {
        self.validate()?;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&GPT2_PUBLIC_STATEMENT_MAGIC);
        bytes.extend_from_slice(&GPT2_PUBLIC_STATEMENT_VERSION.to_le_bytes());
        bytes.extend_from_slice(&self.protocol_version.to_le_bytes());
        for value in [
            self.model.blocks,
            self.model.hidden_width,
            self.model.attention_heads,
            self.model.mlp_width,
            self.model.vocabulary,
            self.model.maximum_positions,
        ] {
            bytes.extend_from_slice(&(value as u64).to_le_bytes());
        }
        bytes.extend_from_slice(&self.fractional_bits.to_le_bytes());
        bytes.extend_from_slice(&self.checkpoint_sha256);
        bytes.extend_from_slice(&self.tokenizer_sha256);
        bytes.extend_from_slice(&(self.token_ids.len() as u64).to_le_bytes());
        for &token in &self.token_ids {
            bytes.extend_from_slice(&(token as u64).to_le_bytes());
        }
        bytes.extend_from_slice(&(self.public_logits.len() as u64).to_le_bytes());
        for &logit in &self.public_logits {
            bytes.extend_from_slice(&logit.to_le_bytes());
        }
        bytes.extend_from_slice(&self.operation_manifest_digest);
        bytes.extend_from_slice(&self.bva_layout_digest);
        bytes.extend_from_slice(&self.relation_manifest_digest);
        bytes.extend_from_slice(&self.lookup_registry_digest);
        if bytes.len() > GPT2_PUBLIC_STATEMENT_MAX_BYTES {
            return Err(Gpt2CheckpointError::ArtifactOversized(bytes.len()));
        }
        Ok(bytes)
    }

    pub fn from_bytes_exact(bytes: &[u8]) -> Result<Self, Gpt2CheckpointError> {
        if bytes.len() > GPT2_PUBLIC_STATEMENT_MAX_BYTES {
            return Err(Gpt2CheckpointError::ArtifactOversized(bytes.len()));
        }
        let mut reader = io::Cursor::new(bytes);
        if read_array::<8>(&mut reader)? != GPT2_PUBLIC_STATEMENT_MAGIC {
            return Err(Gpt2CheckpointError::InvalidMagic);
        }
        let artifact_version = read_u32(&mut reader)?;
        if artifact_version != GPT2_PUBLIC_STATEMENT_VERSION {
            return Err(Gpt2CheckpointError::UnsupportedProofProtocolVersion(
                artifact_version,
            ));
        }
        let protocol_version = read_u32(&mut reader)?;
        let model = Gpt2Model {
            blocks: read_artifact_usize(&mut reader)?,
            hidden_width: read_artifact_usize(&mut reader)?,
            attention_heads: read_artifact_usize(&mut reader)?,
            mlp_width: read_artifact_usize(&mut reader)?,
            vocabulary: read_artifact_usize(&mut reader)?,
            maximum_positions: read_artifact_usize(&mut reader)?,
        };
        let fractional_bits = read_u32(&mut reader)?;
        let checkpoint_sha256 = read_array::<32>(&mut reader)?;
        let tokenizer_sha256 = read_array::<32>(&mut reader)?;
        let token_count = read_artifact_usize(&mut reader)?;
        if token_count > GPT2_PUBLIC_STATEMENT_MAX_VECTOR_ENTRIES {
            return Err(Gpt2CheckpointError::ArtifactOversized(token_count));
        }
        let mut token_ids = Vec::with_capacity(token_count);
        for _ in 0..token_count {
            token_ids.push(read_artifact_usize(&mut reader)?);
        }
        let logit_count = read_artifact_usize(&mut reader)?;
        if logit_count > GPT2_PUBLIC_STATEMENT_MAX_VECTOR_ENTRIES {
            return Err(Gpt2CheckpointError::ArtifactOversized(logit_count));
        }
        let mut public_logits = Vec::with_capacity(logit_count);
        for _ in 0..logit_count {
            public_logits.push(read_i64(&mut reader)?);
        }
        let statement = Self {
            protocol_version,
            model,
            fractional_bits,
            checkpoint_sha256,
            tokenizer_sha256,
            token_ids,
            public_logits,
            operation_manifest_digest: read_array::<32>(&mut reader)?,
            bva_layout_digest: read_array::<32>(&mut reader)?,
            relation_manifest_digest: read_array::<32>(&mut reader)?,
            lookup_registry_digest: read_array::<32>(&mut reader)?,
        };
        if reader.position() as usize != bytes.len() {
            return Err(Gpt2CheckpointError::ArtifactTrailingBytes);
        }
        statement.validate()?;
        Ok(statement)
    }

    pub fn relation_shape(&self) -> Result<Gpt2HdR1csShape, Gpt2CheckpointError> {
        self.validate()?;
        let metadata = Gpt2Q12Metadata {
            format_version: GPT2_Q12_VERSION,
            model: self.model,
            fractional_bits: self.fractional_bits,
            parameter_entries: self.model.physical_parameter_entries()? as u64,
            token_count: self.token_ids.len(),
            checkpoint_sha256: self.checkpoint_sha256,
            tokenizer_sha256: self.tokenizer_sha256,
        };
        let layout = Gpt2GlobalLayout::compile(&metadata)?;
        relation::Gpt2PrimitiveRelationManifest::compile(&metadata, &layout)?.shape(&layout)
    }

    /// Domain-separated digest absorbed before any prover commitment or
    /// Fiat-Shamir challenge in the complete protocol.
    pub fn transcript_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"zksmart.gpt2.hd-r1cs.public-statement.v1");
        hash_u32(&mut hasher, self.protocol_version);
        hash_model(&mut hasher, self.model);
        hash_u32(&mut hasher, self.fractional_bits);
        hasher.update(self.checkpoint_sha256);
        hasher.update(self.tokenizer_sha256);
        hash_usize(&mut hasher, self.token_ids.len());
        for &token in &self.token_ids {
            hash_usize(&mut hasher, token);
        }
        hash_usize(&mut hasher, self.public_logits.len());
        for &logit in &self.public_logits {
            hasher.update(logit.to_le_bytes());
        }
        hasher.update(self.operation_manifest_digest);
        hasher.update(self.bva_layout_digest);
        hasher.update(self.relation_manifest_digest);
        hasher.update(self.lookup_registry_digest);
        hasher.finalize().into()
    }
}

fn digest_operation_manifest(metadata: &Gpt2Q12Metadata) -> Result<[u8; 32], Gpt2CheckpointError> {
    let mut hasher = Sha256::new();
    hasher.update(b"zksmart.gpt2.operation-manifest.v1");
    hash_model(&mut hasher, metadata.model);
    hash_usize(&mut hasher, metadata.token_count);
    let tensors = metadata.model.parameter_tensors()?;
    hash_usize(&mut hasher, tensors.len());
    for tensor in tensors {
        hash_bytes(&mut hasher, tensor.name.as_bytes());
        hasher.update([parameter_kind_tag(tensor.kind)]);
        hash_usize(&mut hasher, tensor.rows);
        hash_usize(&mut hasher, tensor.cols);
        hash_usize(&mut hasher, tensor.entries);
    }
    let operations = metadata.operation_manifest()?;
    hash_usize(&mut hasher, operations.len());
    for operation in operations {
        hasher.update([operation_kind_tag(operation.kind)]);
        hash_optional_usize(&mut hasher, operation.block);
        hash_usize(&mut hasher, operation.rows);
        hash_usize(&mut hasher, operation.cols);
        for tensor in operation.parameter_tensors {
            hash_optional_usize(&mut hasher, tensor);
        }
    }
    Ok(hasher.finalize().into())
}

fn digest_lookup_registry() -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"zksmart.gpt2.q12.lookup-registry.v2");
    hasher.update(b"gelu:clamp[-8Q,8Q]:round-nearest");
    hasher.update(b"exp:clamp[-16Q,0]:round-nearest");
    hasher.update(b"reciprocal:clamp[1,nQ]:floor");
    hasher.update(b"softmax-residual:first-argmax:sum=Q");
    hasher.update(b"rsqrt:clamp[1,2^25]:round-nearest");
    hasher.update(b"signed-division:euclidean-floor");
    hasher.update(b"range-limbs:biased-signed-or-unsigned:little-endian");
    hasher.update(b"range-tags:1,2,3,4,5,6,7,8-bits");
    hash_u32(&mut hasher, Gpt2Q12Semantics::exact().fractional_bits());
    hasher.finalize().into()
}

fn hash_model(hasher: &mut Sha256, model: Gpt2Model) {
    for value in [
        model.blocks,
        model.hidden_width,
        model.attention_heads,
        model.mlp_width,
        model.vocabulary,
        model.maximum_positions,
    ] {
        hash_usize(hasher, value);
    }
}

fn hash_bytes(hasher: &mut Sha256, bytes: &[u8]) {
    hash_usize(hasher, bytes.len());
    hasher.update(bytes);
}

fn hash_u32(hasher: &mut Sha256, value: u32) {
    hasher.update(value.to_le_bytes());
}

fn hash_usize(hasher: &mut Sha256, value: usize) {
    hasher.update((value as u64).to_le_bytes());
}

fn hash_optional_usize(hasher: &mut Sha256, value: Option<usize>) {
    match value {
        Some(value) => {
            hasher.update([1]);
            hash_usize(hasher, value);
        }
        None => hasher.update([0]),
    }
}

const fn parameter_kind_tag(kind: ParameterKind) -> u8 {
    match kind {
        ParameterKind::Embedding => 0,
        ParameterKind::LayerNorm => 1,
        ParameterKind::Attention => 2,
        ParameterKind::Mlp => 3,
    }
}

const fn operation_kind_tag(kind: Gpt2Q12OperationKind) -> u8 {
    match kind {
        Gpt2Q12OperationKind::Embedding => 0,
        Gpt2Q12OperationKind::LayerNorm1 => 1,
        Gpt2Q12OperationKind::QkvProjection => 2,
        Gpt2Q12OperationKind::CausalAttention => 3,
        Gpt2Q12OperationKind::AttentionProjection => 4,
        Gpt2Q12OperationKind::AttentionResidual => 5,
        Gpt2Q12OperationKind::LayerNorm2 => 6,
        Gpt2Q12OperationKind::MlpExpansion => 7,
        Gpt2Q12OperationKind::Gelu => 8,
        Gpt2Q12OperationKind::MlpProjection => 9,
        Gpt2Q12OperationKind::MlpResidual => 10,
        Gpt2Q12OperationKind::FinalLayerNorm => 11,
        Gpt2Q12OperationKind::LmHead => 12,
    }
}

impl Gpt2Q12Metadata {
    pub fn read_from(reader: &mut impl Read) -> Result<Self, Gpt2CheckpointError> {
        let magic = read_array::<8>(reader)?;
        if magic != GPT2_Q12_MAGIC {
            return Err(Gpt2CheckpointError::InvalidMagic);
        }
        let version = read_u32(reader)?;
        if version != GPT2_Q12_LEGACY_VERSION && version != GPT2_Q12_VERSION {
            return Err(Gpt2CheckpointError::UnsupportedVersion(version));
        }
        let fractional_bits = read_u32(reader)?;
        if fractional_bits != 12 {
            return Err(Gpt2CheckpointError::UnsupportedFractionalBits(
                fractional_bits,
            ));
        }
        let parameter_entries = read_u64(reader)?;
        let token_count = read_u32(reader)? as usize;
        let vocabulary = read_u32(reader)? as usize;
        let maximum_positions = read_u32(reader)? as usize;
        let hidden_width = read_u32(reader)? as usize;
        let attention_heads = read_u32(reader)? as usize;
        let mlp_width = read_u32(reader)? as usize;
        let blocks = read_u32(reader)? as usize;
        let checkpoint_sha256 = read_array::<32>(reader)?;
        let tokenizer_sha256 = read_array::<32>(reader)?;
        let model = Gpt2Model {
            blocks,
            hidden_width,
            attention_heads,
            mlp_width,
            vocabulary,
            maximum_positions,
        };
        let expected_entries = model.physical_parameter_entries()? as u64;
        if parameter_entries != expected_entries {
            return Err(Gpt2CheckpointError::ParameterCountMismatch {
                expected: expected_entries,
                actual: parameter_entries,
            });
        }
        if token_count == 0 || token_count > maximum_positions {
            return Err(Gpt2CheckpointError::InvalidTokenCount(token_count));
        }
        Ok(Self {
            format_version: version,
            model,
            fractional_bits,
            parameter_entries,
            token_count,
            checkpoint_sha256,
            tokenizer_sha256,
        })
    }

    pub fn scale(&self) -> i32 {
        1_i32 << self.fractional_bits
    }

    pub fn operation_manifest(
        &self,
    ) -> Result<Vec<Gpt2Q12OperationDescriptor>, Gpt2CheckpointError> {
        if self.token_count == 0 || self.token_count > self.model.maximum_positions {
            return Err(Gpt2CheckpointError::InvalidTokenCount(self.token_count));
        }
        let hidden = self.model.hidden_width;
        let qkv_width = hidden
            .checked_mul(3)
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        let mut operations = Vec::with_capacity(
            self.model
                .blocks
                .checked_mul(10)
                .and_then(|count| count.checked_add(3))
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?,
        );
        let mut push = |kind, block, rows, cols, parameter_tensors| {
            operations.push(Gpt2Q12OperationDescriptor {
                kind,
                block,
                rows,
                cols,
                parameter_tensors,
            });
        };
        push(
            Gpt2Q12OperationKind::Embedding,
            None,
            self.token_count,
            hidden,
            [Some(0), Some(1)],
        );
        for block in 0..self.model.blocks {
            let base = 2 + block * 12;
            for (kind, cols, parameter_tensors) in [
                (
                    Gpt2Q12OperationKind::LayerNorm1,
                    hidden,
                    [Some(base), Some(base + 1)],
                ),
                (
                    Gpt2Q12OperationKind::QkvProjection,
                    qkv_width,
                    [Some(base + 2), Some(base + 3)],
                ),
                (Gpt2Q12OperationKind::CausalAttention, hidden, [None, None]),
                (
                    Gpt2Q12OperationKind::AttentionProjection,
                    hidden,
                    [Some(base + 4), Some(base + 5)],
                ),
                (
                    Gpt2Q12OperationKind::AttentionResidual,
                    hidden,
                    [None, None],
                ),
                (
                    Gpt2Q12OperationKind::LayerNorm2,
                    hidden,
                    [Some(base + 6), Some(base + 7)],
                ),
                (
                    Gpt2Q12OperationKind::MlpExpansion,
                    self.model.mlp_width,
                    [Some(base + 8), Some(base + 9)],
                ),
                (
                    Gpt2Q12OperationKind::Gelu,
                    self.model.mlp_width,
                    [None, None],
                ),
                (
                    Gpt2Q12OperationKind::MlpProjection,
                    hidden,
                    [Some(base + 10), Some(base + 11)],
                ),
                (Gpt2Q12OperationKind::MlpResidual, hidden, [None, None]),
            ] {
                push(kind, Some(block), self.token_count, cols, parameter_tensors);
            }
        }
        let final_base = 2 + self.model.blocks * 12;
        push(
            Gpt2Q12OperationKind::FinalLayerNorm,
            None,
            self.token_count,
            hidden,
            [Some(final_base), Some(final_base + 1)],
        );
        push(
            Gpt2Q12OperationKind::LmHead,
            None,
            1,
            self.model.vocabulary,
            [Some(0), None],
        );
        Ok(operations)
    }
}

pub struct Gpt2Q12Checkpoint<R> {
    reader: R,
    metadata: Gpt2Q12Metadata,
    token_ids: Vec<usize>,
    reference_logits: Vec<i64>,
    parameter_data_offset: u64,
    tensor_offsets: Vec<u64>,
    tensor_lengths: Vec<usize>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gpt2Q12Execution {
    pub logits: Vec<i64>,
    pub argmax: usize,
    pub checksum: i128,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gpt2Q12OperationKind {
    Embedding,
    LayerNorm1,
    QkvProjection,
    CausalAttention,
    AttentionProjection,
    AttentionResidual,
    LayerNorm2,
    MlpExpansion,
    Gelu,
    MlpProjection,
    MlpResidual,
    FinalLayerNorm,
    LmHead,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gpt2Q12OperationDescriptor {
    pub kind: Gpt2Q12OperationKind,
    pub block: Option<usize>,
    pub rows: usize,
    pub cols: usize,
    pub parameter_tensors: [Option<usize>; 2],
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Gpt2GlobalBlockRole {
    Parameter(usize),
    PublicTokenIds,
    OperationOutput(usize),
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gpt2GlobalBlockDescriptor {
    pub id: GlobalBlockId,
    pub role: Gpt2GlobalBlockRole,
    pub rows: usize,
    pub cols: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gpt2GlobalOperation {
    pub kind: Gpt2Q12OperationKind,
    pub block: Option<usize>,
    pub inputs: [Option<GlobalBlockId>; 2],
    pub parameter_blocks: [Option<GlobalBlockId>; 2],
    pub output: GlobalBlockId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gpt2GlobalLayout {
    blocks: Vec<Gpt2GlobalBlockDescriptor>,
    operations: Vec<Gpt2GlobalOperation>,
    model_blocks: Vec<GlobalBlockId>,
    public_input_block: GlobalBlockId,
    public_output_block: GlobalBlockId,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gpt2BvaVectorDescriptor {
    pub vector_index: usize,
    pub block: GlobalBlockId,
    pub column: usize,
    pub physical_len: usize,
    pub bucket_index: usize,
    pub logical_offset: usize,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gpt2GlobalBvaLayout {
    layout: BvaLayout,
    vectors: Vec<Gpt2BvaVectorDescriptor>,
    physical_entries: usize,
}

impl Gpt2GlobalBvaLayout {
    pub fn vectors(&self) -> &[Gpt2BvaVectorDescriptor] {
        &self.vectors
    }

    pub fn vector_count(&self) -> usize {
        self.vectors.len()
    }

    pub fn physical_entries(&self) -> usize {
        self.physical_entries
    }

    pub fn logical_entries(&self) -> usize {
        self.layout.logical_len()
    }

    pub fn logical_padding_entries(&self) -> usize {
        self.logical_entries() - self.physical_entries
    }

    pub fn materialized_padding_entries(&self) -> usize {
        self.layout.materialized_padding_entries()
    }

    pub fn bva_layout(&self) -> &BvaLayout {
        &self.layout
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Gpt2ActivationWitness {
    blocks: Vec<Option<MatrixBlock<i32>>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Gpt2ModelStreamProfile {
    pub physical_entries: usize,
    pub max_live_entries: usize,
}

pub struct Gpt2GlobalWitnessSource<'a, R> {
    checkpoint: &'a mut Gpt2Q12Checkpoint<R>,
    layout: &'a Gpt2GlobalLayout,
    activations: &'a Gpt2ActivationWitness,
    domain: PaddedDomain,
}

pub struct Gpt2GlobalBvaWitnessSource<'a, R> {
    global: Gpt2GlobalWitnessSource<'a, R>,
    bva: &'a Gpt2GlobalBvaLayout,
}

impl<'a, R: Read + Seek> Gpt2GlobalBvaWitnessSource<'a, R> {
    pub fn new(
        checkpoint: &'a mut Gpt2Q12Checkpoint<R>,
        layout: &'a Gpt2GlobalLayout,
        bva: &'a Gpt2GlobalBvaLayout,
        activations: &'a Gpt2ActivationWitness,
    ) -> Result<Self, Gpt2CheckpointError> {
        if *bva != layout.compile_bva()? {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        Ok(Self {
            global: Gpt2GlobalWitnessSource::new(checkpoint, layout, activations)?,
            bva,
        })
    }
}

impl<'a, R: Read + Seek> Gpt2GlobalWitnessSource<'a, R> {
    pub fn new(
        checkpoint: &'a mut Gpt2Q12Checkpoint<R>,
        layout: &'a Gpt2GlobalLayout,
        activations: &'a Gpt2ActivationWitness,
    ) -> Result<Self, Gpt2CheckpointError> {
        if *layout != Gpt2GlobalLayout::compile(checkpoint.metadata())?
            || activations.blocks.len() != layout.witness_block_count()
        {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        for descriptor in layout.blocks() {
            let activation = activations
                .blocks
                .get(descriptor.id.0)
                .and_then(Option::as_ref);
            match descriptor.role {
                Gpt2GlobalBlockRole::Parameter(_) if activation.is_some() => {
                    return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                }
                Gpt2GlobalBlockRole::Parameter(_) => {}
                _ => {
                    let activation =
                        activation.ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                    if activation.dims.rows != descriptor.rows
                        || activation.dims.cols != descriptor.cols
                    {
                        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
                    }
                }
            }
        }
        let domain = layout.padded_domain()?;
        Ok(Self {
            checkpoint,
            layout,
            activations,
            domain,
        })
    }
}

impl<R: Read + Seek> StackedI32Source for Gpt2GlobalWitnessSource<'_, R> {
    type Error = Gpt2CheckpointError;

    fn domain(&self) -> PaddedDomain {
        self.domain
    }

    fn for_each_block_column(
        &mut self,
        visit: &mut dyn FnMut(usize, usize, &[i32]),
    ) -> Result<(), Self::Error> {
        let mut buffer = Vec::new();
        for descriptor in self.layout.blocks() {
            match descriptor.role {
                Gpt2GlobalBlockRole::Parameter(_) => {
                    buffer.resize(descriptor.rows, 0);
                    for column in 0..descriptor.cols {
                        self.checkpoint.read_model_block_column(
                            self.layout,
                            descriptor.id,
                            column,
                            &mut buffer,
                        )?;
                        visit(descriptor.id.0, column, &buffer);
                    }
                }
                _ => {
                    let block = self.activations.blocks[descriptor.id.0]
                        .as_ref()
                        .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
                    for (column, values) in block.columns.iter().enumerate() {
                        visit(descriptor.id.0, column, values);
                    }
                }
            }
        }
        Ok(())
    }
}

impl<R: Read + Seek> BvaI32Source for Gpt2GlobalBvaWitnessSource<'_, R> {
    type Error = Gpt2CheckpointError;

    fn layout(&self) -> &BvaLayout {
        self.bva.bva_layout()
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[i32])) -> Result<(), Self::Error> {
        let Self { global, bva } = self;
        let vectors = &bva.vectors;
        let mut vector_index = 0_usize;
        let mut shape_mismatch = false;
        global.for_each_block_column(&mut |slot, column, values| {
            let Some(descriptor) = vectors.get(vector_index) else {
                shape_mismatch = true;
                return;
            };
            if descriptor.block.0 != slot
                || descriptor.column != column
                || descriptor.physical_len != values.len()
            {
                shape_mismatch = true;
                return;
            }
            visit(vector_index, values);
            vector_index += 1;
        })?;
        if shape_mismatch || vector_index != vectors.len() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        Ok(())
    }
}

impl Gpt2ActivationWitness {
    pub fn block(&self, id: GlobalBlockId) -> Option<&MatrixBlock<i32>> {
        self.blocks.get(id.0).and_then(Option::as_ref)
    }

    pub fn stored_entries(&self) -> usize {
        self.blocks
            .iter()
            .flatten()
            .map(|block| block.dims.entries())
            .sum()
    }
}

impl Gpt2GlobalLayout {
    pub fn compile(metadata: &Gpt2Q12Metadata) -> Result<Self, Gpt2CheckpointError> {
        let parameter_specs = metadata.model.parameter_tensors()?;
        let operation_manifest = metadata.operation_manifest()?;
        let total_blocks = parameter_specs
            .len()
            .checked_add(1)
            .and_then(|count| count.checked_add(operation_manifest.len()))
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        let mut blocks = Vec::with_capacity(total_blocks);
        let mut model_blocks = Vec::with_capacity(parameter_specs.len());
        for (tensor_index, spec) in parameter_specs.iter().enumerate() {
            let id = GlobalBlockId(tensor_index);
            let (rows, cols) = parameter_matrix_shape(spec);
            blocks.push(Gpt2GlobalBlockDescriptor {
                id,
                role: Gpt2GlobalBlockRole::Parameter(tensor_index),
                rows,
                cols,
            });
            model_blocks.push(id);
        }

        let public_input_block = GlobalBlockId(blocks.len());
        blocks.push(Gpt2GlobalBlockDescriptor {
            id: public_input_block,
            role: Gpt2GlobalBlockRole::PublicTokenIds,
            rows: metadata.token_count,
            cols: 1,
        });

        let mut operations = Vec::with_capacity(operation_manifest.len());
        let mut current_hidden = None;
        let mut block_input = None;
        let mut attention_residual = None;
        let mut last_output = None;
        for (operation_index, descriptor) in operation_manifest.iter().enumerate() {
            let output = GlobalBlockId(blocks.len());
            let inputs = match descriptor.kind {
                Gpt2Q12OperationKind::Embedding => [Some(public_input_block), None],
                Gpt2Q12OperationKind::LayerNorm1 => {
                    block_input = current_hidden;
                    [current_hidden, None]
                }
                Gpt2Q12OperationKind::AttentionResidual => {
                    let inputs = [block_input, last_output];
                    attention_residual = Some(output);
                    inputs
                }
                Gpt2Q12OperationKind::MlpResidual => [attention_residual, last_output],
                Gpt2Q12OperationKind::FinalLayerNorm => [current_hidden, None],
                _ => [last_output, None],
            };
            if inputs[0].is_none() {
                return Err(Gpt2CheckpointError::ProtocolIncomplete);
            }
            let parameter_blocks = descriptor
                .parameter_tensors
                .map(|tensor_index| tensor_index.map(GlobalBlockId));
            if parameter_blocks
                .iter()
                .flatten()
                .any(|id| id.0 >= model_blocks.len())
            {
                return Err(Gpt2CheckpointError::TensorIndex(model_blocks.len()));
            }
            blocks.push(Gpt2GlobalBlockDescriptor {
                id: output,
                role: Gpt2GlobalBlockRole::OperationOutput(operation_index),
                rows: descriptor.cols,
                cols: descriptor.rows,
            });
            operations.push(Gpt2GlobalOperation {
                kind: descriptor.kind,
                block: descriptor.block,
                inputs,
                parameter_blocks,
                output,
            });
            last_output = Some(output);
            match descriptor.kind {
                Gpt2Q12OperationKind::Embedding | Gpt2Q12OperationKind::MlpResidual => {
                    current_hidden = Some(output);
                }
                _ => {}
            }
        }
        let public_output_block = last_output.ok_or(Gpt2CheckpointError::ProtocolIncomplete)?;
        Ok(Self {
            blocks,
            operations,
            model_blocks,
            public_input_block,
            public_output_block,
        })
    }

    pub fn blocks(&self) -> &[Gpt2GlobalBlockDescriptor] {
        &self.blocks
    }

    pub fn operations(&self) -> &[Gpt2GlobalOperation] {
        &self.operations
    }

    pub fn model_blocks(&self) -> &[GlobalBlockId] {
        &self.model_blocks
    }

    pub fn public_input_block(&self) -> GlobalBlockId {
        self.public_input_block
    }

    pub fn public_output_block(&self) -> GlobalBlockId {
        self.public_output_block
    }

    pub fn witness_block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn activation_entries(&self) -> Result<usize, Gpt2CheckpointError> {
        self.blocks
            .iter()
            .filter(|block| !matches!(block.role, Gpt2GlobalBlockRole::Parameter(_)))
            .try_fold(0_usize, |sum, block| {
                block
                    .rows
                    .checked_mul(block.cols)
                    .and_then(|entries| sum.checked_add(entries))
                    .ok_or(Gpt2CheckpointError::OffsetOverflow)
            })
    }

    pub fn physical_entries(
        &self,
        metadata: &Gpt2Q12Metadata,
    ) -> Result<usize, Gpt2CheckpointError> {
        let parameters = usize::try_from(metadata.parameter_entries)
            .map_err(|_| Gpt2CheckpointError::OffsetOverflow)?;
        parameters
            .checked_add(self.activation_entries()?)
            .ok_or(Gpt2CheckpointError::OffsetOverflow)
    }

    pub fn padded_domain(&self) -> Result<PaddedDomain, Gpt2CheckpointError> {
        let slots = self
            .blocks
            .len()
            .checked_next_power_of_two()
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        let row_cap = self
            .blocks
            .iter()
            .map(|block| block.rows)
            .max()
            .ok_or(Gpt2CheckpointError::ProtocolIncomplete)?
            .checked_next_power_of_two()
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        let col_cap = self
            .blocks
            .iter()
            .map(|block| block.cols)
            .max()
            .ok_or(Gpt2CheckpointError::ProtocolIncomplete)?
            .checked_next_power_of_two()
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        PaddedDomain::new(slots, row_cap, col_cap).map_err(Into::into)
    }

    pub fn logical_padded_entries(&self) -> Result<usize, Gpt2CheckpointError> {
        let domain = self.padded_domain()?;
        domain
            .slots
            .checked_mul(domain.row_cap)
            .and_then(|entries| entries.checked_mul(domain.col_cap))
            .ok_or(Gpt2CheckpointError::OffsetOverflow)
    }

    pub fn compile_bva(&self) -> Result<Gpt2GlobalBvaLayout, Gpt2CheckpointError> {
        let vector_count = self.blocks.iter().try_fold(0_usize, |sum, block| {
            sum.checked_add(block.cols)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)
        })?;
        let mut lengths = Vec::with_capacity(vector_count);
        for block in &self.blocks {
            lengths.extend(std::iter::repeat_n(block.rows, block.cols));
        }
        let layout = BvaLayout::from_lengths(&lengths)?;
        let bucket_offsets = layout.bucket_offsets();
        let mut vectors = Vec::with_capacity(vector_count);
        let mut physical_entries = 0_usize;
        for block in &self.blocks {
            for column in 0..block.cols {
                let vector_index = vectors.len();
                let bucket_index = layout.vector_to_bucket()[vector_index];
                vectors.push(Gpt2BvaVectorDescriptor {
                    vector_index,
                    block: block.id,
                    column,
                    physical_len: block.rows,
                    bucket_index,
                    logical_offset: bucket_offsets[bucket_index],
                });
                physical_entries = physical_entries
                    .checked_add(block.rows)
                    .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
            }
        }
        Ok(Gpt2GlobalBvaLayout {
            layout,
            vectors,
            physical_entries,
        })
    }
}

fn parameter_matrix_shape(spec: &TensorSpec) -> (usize, usize) {
    if spec.rows == 1 {
        (spec.cols, 1)
    } else if spec.kind == ParameterKind::Embedding {
        (spec.rows, spec.cols)
    } else {
        (spec.cols, spec.rows)
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Gpt2Q12Values<'a> {
    Matrix(&'a [Vec<i64>]),
    Vector(&'a [i64]),
}

impl Gpt2Q12Values<'_> {
    pub fn rows(&self) -> usize {
        match self {
            Self::Matrix(rows) => rows.len(),
            Self::Vector(_) => 1,
        }
    }

    pub fn cols(&self) -> usize {
        match self {
            Self::Matrix(rows) => rows.first().map(Vec::len).unwrap_or(0),
            Self::Vector(values) => values.len(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Gpt2Q12TraceEvent<'a> {
    pub kind: Gpt2Q12OperationKind,
    pub block: Option<usize>,
    pub parameter_tensors: [Option<usize>; 2],
    pub values: Gpt2Q12Values<'a>,
}

impl<R: Read + Seek> Gpt2Q12Checkpoint<R> {
    pub fn from_reader(mut reader: R) -> Result<Self, Gpt2CheckpointError> {
        let metadata = Gpt2Q12Metadata::read_from(&mut reader)?;
        let mut token_ids = Vec::with_capacity(metadata.token_count);
        for index in 0..metadata.token_count {
            let token = read_u32(&mut reader)? as usize;
            if token >= metadata.model.vocabulary {
                return Err(Gpt2CheckpointError::InvalidTokenId { index, token });
            }
            token_ids.push(token);
        }
        let mut reference_logits = Vec::with_capacity(metadata.model.vocabulary);
        for _ in 0..metadata.model.vocabulary {
            reference_logits.push(read_i64(&mut reader)?);
        }
        let parameter_data_offset = reader.stream_position()?;
        let specs = metadata.model.parameter_tensors()?;
        let mut offset = 0_u64;
        let mut tensor_offsets = Vec::with_capacity(specs.len());
        let mut tensor_lengths = Vec::with_capacity(specs.len());
        for spec in specs {
            tensor_offsets.push(offset);
            tensor_lengths.push(spec.entries);
            offset = offset
                .checked_add(spec.entries as u64)
                .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        }
        if offset != metadata.parameter_entries {
            return Err(Gpt2CheckpointError::ParameterCountMismatch {
                expected: metadata.parameter_entries,
                actual: offset,
            });
        }
        Ok(Self {
            reader,
            metadata,
            token_ids,
            reference_logits,
            parameter_data_offset,
            tensor_offsets,
            tensor_lengths,
        })
    }

    pub fn metadata(&self) -> &Gpt2Q12Metadata {
        &self.metadata
    }

    pub fn token_ids(&self) -> &[usize] {
        &self.token_ids
    }

    pub fn reference_logits(&self) -> &[i64] {
        &self.reference_logits
    }

    /// Migrate a legacy bundle to the exact-Softmax v2 format without
    /// materializing or re-encoding model parameters. Only the header and
    /// public reference logits change; the parameter payload is copied as a
    /// bounded stream.
    pub fn write_bundle_with_reference(
        &mut self,
        writer: &mut impl Write,
        reference_logits: &[i64],
    ) -> Result<(), Gpt2CheckpointError> {
        if reference_logits.len() != self.metadata.model.vocabulary {
            return Err(Gpt2CheckpointError::PublicLogitCountMismatch {
                expected: self.metadata.model.vocabulary,
                actual: reference_logits.len(),
            });
        }
        writer.write_all(&GPT2_Q12_MAGIC)?;
        writer.write_all(&GPT2_Q12_VERSION.to_le_bytes())?;
        writer.write_all(&self.metadata.fractional_bits.to_le_bytes())?;
        writer.write_all(&self.metadata.parameter_entries.to_le_bytes())?;
        for value in [
            self.metadata.token_count,
            self.metadata.model.vocabulary,
            self.metadata.model.maximum_positions,
            self.metadata.model.hidden_width,
            self.metadata.model.attention_heads,
            self.metadata.model.mlp_width,
            self.metadata.model.blocks,
        ] {
            let value = u32::try_from(value).map_err(|_| Gpt2CheckpointError::OffsetOverflow)?;
            writer.write_all(&value.to_le_bytes())?;
        }
        writer.write_all(&self.metadata.checkpoint_sha256)?;
        writer.write_all(&self.metadata.tokenizer_sha256)?;
        for &token in &self.token_ids {
            let token = u32::try_from(token).map_err(|_| Gpt2CheckpointError::OffsetOverflow)?;
            writer.write_all(&token.to_le_bytes())?;
        }
        for &logit in reference_logits {
            writer.write_all(&logit.to_le_bytes())?;
        }

        let parameter_bytes = self
            .metadata
            .parameter_entries
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        self.reader
            .seek(SeekFrom::Start(self.parameter_data_offset))?;
        let mut payload = (&mut self.reader).take(parameter_bytes);
        let copied = io::copy(&mut payload, writer)?;
        if copied != parameter_bytes {
            return Err(Gpt2CheckpointError::ParameterPayloadByteCountMismatch {
                expected: parameter_bytes,
                actual: copied,
            });
        }
        Ok(())
    }

    pub fn read_parameter_chunk(
        &mut self,
        tensor_index: usize,
        entry_offset: usize,
        output: &mut [i32],
    ) -> Result<(), Gpt2CheckpointError> {
        let tensor_len = *self
            .tensor_lengths
            .get(tensor_index)
            .ok_or(Gpt2CheckpointError::TensorIndex(tensor_index))?;
        let end = entry_offset
            .checked_add(output.len())
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        if end > tensor_len {
            return Err(Gpt2CheckpointError::ParameterChunkOutOfBounds {
                tensor_index,
                tensor_len,
                entry_offset,
                chunk_len: output.len(),
            });
        }
        let absolute_entry = self.tensor_offsets[tensor_index]
            .checked_add(entry_offset as u64)
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        let byte_offset = absolute_entry
            .checked_mul(std::mem::size_of::<i32>() as u64)
            .and_then(|offset| self.parameter_data_offset.checked_add(offset))
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        self.reader.seek(SeekFrom::Start(byte_offset))?;
        for value in output {
            *value = read_i32(&mut self.reader)?;
        }
        Ok(())
    }

    pub fn read_model_block_column(
        &mut self,
        layout: &Gpt2GlobalLayout,
        block_id: GlobalBlockId,
        column: usize,
        output: &mut [i32],
    ) -> Result<(), Gpt2CheckpointError> {
        let descriptor = layout
            .blocks
            .get(block_id.0)
            .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
        let Gpt2GlobalBlockRole::Parameter(tensor_index) = descriptor.role else {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        };
        if column >= descriptor.cols || output.len() != descriptor.rows {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let entry_offset = column
            .checked_mul(descriptor.rows)
            .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
        self.read_parameter_chunk(tensor_index, entry_offset, output)
    }

    pub fn for_each_model_block_column(
        &mut self,
        layout: &Gpt2GlobalLayout,
        mut visit: impl FnMut(GlobalBlockId, usize, &[i32]),
    ) -> Result<Gpt2ModelStreamProfile, Gpt2CheckpointError> {
        let mut buffer = Vec::new();
        let mut physical_entries = 0_usize;
        let mut max_live_entries = 0_usize;
        for &block_id in layout.model_blocks() {
            let descriptor = &layout.blocks[block_id.0];
            max_live_entries = max_live_entries.max(descriptor.rows);
            buffer.resize(descriptor.rows, 0);
            for column in 0..descriptor.cols {
                self.read_model_block_column(layout, block_id, column, &mut buffer)?;
                physical_entries = physical_entries
                    .checked_add(buffer.len())
                    .ok_or(Gpt2CheckpointError::OffsetOverflow)?;
                visit(block_id, column, &buffer);
            }
        }
        Ok(Gpt2ModelStreamProfile {
            physical_entries,
            max_live_entries,
        })
    }

    pub fn execute(&mut self) -> Result<Gpt2Q12Execution, Gpt2CheckpointError> {
        self.execute_with_trace(|_| Ok(()))
    }

    pub fn execute_activation_witness(
        &mut self,
        layout: &Gpt2GlobalLayout,
    ) -> Result<(Gpt2Q12Execution, Gpt2ActivationWitness), Gpt2CheckpointError> {
        if *layout != Gpt2GlobalLayout::compile(&self.metadata)? {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let mut blocks = vec![None; layout.witness_block_count()];
        let token_values = self
            .token_ids
            .iter()
            .map(|&token| i32::try_from(token).map_err(|_| Gpt2CheckpointError::ArithmeticOverflow))
            .collect::<Result<Vec<_>, _>>()?;
        blocks[layout.public_input_block.0] = Some(MatrixBlock::from_columns(
            MatrixDims::new(token_values.len(), 1)?,
            vec![token_values],
        )?);

        let mut operation_index = 0_usize;
        let execution = self.execute_with_trace(|event| {
            let operation = layout
                .operations
                .get(operation_index)
                .ok_or(Gpt2CheckpointError::ProtocolShapeMismatch)?;
            if operation.kind != event.kind
                || operation.block != event.block
                || operation.parameter_blocks
                    != event
                        .parameter_tensors
                        .map(|tensor| tensor.map(GlobalBlockId))
            {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            let block = q12_values_to_block(event.values)?;
            let descriptor = &layout.blocks[operation.output.0];
            if block.dims.rows != descriptor.rows || block.dims.cols != descriptor.cols {
                return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
            }
            blocks[operation.output.0] = Some(block);
            operation_index += 1;
            Ok(())
        })?;
        if operation_index != layout.operations.len() {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        Ok((execution, Gpt2ActivationWitness { blocks }))
    }

    pub fn execute_with_trace(
        &mut self,
        mut observe: impl FnMut(Gpt2Q12TraceEvent<'_>) -> Result<(), Gpt2CheckpointError>,
    ) -> Result<Gpt2Q12Execution, Gpt2CheckpointError> {
        let model = self.metadata.model;
        let tokens = self.token_ids.clone();
        let scale = i64::from(self.metadata.scale());
        let mut hidden = vec![vec![0_i64; model.hidden_width]; tokens.len()];
        let mut token_row = vec![0_i32; model.vocabulary];
        let mut position_row = vec![0_i32; model.maximum_positions];
        for feature in 0..model.hidden_width {
            self.read_parameter_chunk(0, feature * model.vocabulary, &mut token_row)?;
            self.read_parameter_chunk(1, feature * model.maximum_positions, &mut position_row)?;
            for (position, &token) in tokens.iter().enumerate() {
                hidden[position][feature] = i64::from(token_row[token])
                    .checked_add(i64::from(position_row[position]))
                    .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
            }
        }
        observe(Gpt2Q12TraceEvent {
            kind: Gpt2Q12OperationKind::Embedding,
            block: None,
            parameter_tensors: [Some(0), Some(1)],
            values: Gpt2Q12Values::Matrix(&hidden),
        })?;

        for block in 0..model.blocks {
            let base = 2 + block * 12;
            let normalized = self.layer_norm(&hidden, base, base + 1, scale)?;
            observe(Gpt2Q12TraceEvent {
                kind: Gpt2Q12OperationKind::LayerNorm1,
                block: Some(block),
                parameter_tensors: [Some(base), Some(base + 1)],
                values: Gpt2Q12Values::Matrix(&normalized),
            })?;
            let qkv = self.linear(
                &normalized,
                base + 2,
                base + 3,
                model.hidden_width,
                3 * model.hidden_width,
                scale,
            )?;
            observe(Gpt2Q12TraceEvent {
                kind: Gpt2Q12OperationKind::QkvProjection,
                block: Some(block),
                parameter_tensors: [Some(base + 2), Some(base + 3)],
                values: Gpt2Q12Values::Matrix(&qkv),
            })?;
            let attention = q12_attention(&qkv, model, scale)?;
            observe(Gpt2Q12TraceEvent {
                kind: Gpt2Q12OperationKind::CausalAttention,
                block: Some(block),
                parameter_tensors: [None, None],
                values: Gpt2Q12Values::Matrix(&attention),
            })?;
            let attention = self.linear(
                &attention,
                base + 4,
                base + 5,
                model.hidden_width,
                model.hidden_width,
                scale,
            )?;
            observe(Gpt2Q12TraceEvent {
                kind: Gpt2Q12OperationKind::AttentionProjection,
                block: Some(block),
                parameter_tensors: [Some(base + 4), Some(base + 5)],
                values: Gpt2Q12Values::Matrix(&attention),
            })?;
            let residual = add_matrices(&hidden, &attention)?;
            observe(Gpt2Q12TraceEvent {
                kind: Gpt2Q12OperationKind::AttentionResidual,
                block: Some(block),
                parameter_tensors: [None, None],
                values: Gpt2Q12Values::Matrix(&residual),
            })?;
            let normalized = self.layer_norm(&residual, base + 6, base + 7, scale)?;
            observe(Gpt2Q12TraceEvent {
                kind: Gpt2Q12OperationKind::LayerNorm2,
                block: Some(block),
                parameter_tensors: [Some(base + 6), Some(base + 7)],
                values: Gpt2Q12Values::Matrix(&normalized),
            })?;
            let expanded = self.linear(
                &normalized,
                base + 8,
                base + 9,
                model.hidden_width,
                model.mlp_width,
                scale,
            )?;
            observe(Gpt2Q12TraceEvent {
                kind: Gpt2Q12OperationKind::MlpExpansion,
                block: Some(block),
                parameter_tensors: [Some(base + 8), Some(base + 9)],
                values: Gpt2Q12Values::Matrix(&expanded),
            })?;
            let activated = expanded
                .into_iter()
                .map(|row| {
                    row.into_iter()
                        .map(|value| q12_gelu(value, scale))
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            observe(Gpt2Q12TraceEvent {
                kind: Gpt2Q12OperationKind::Gelu,
                block: Some(block),
                parameter_tensors: [None, None],
                values: Gpt2Q12Values::Matrix(&activated),
            })?;
            let projected = self.linear(
                &activated,
                base + 10,
                base + 11,
                model.mlp_width,
                model.hidden_width,
                scale,
            )?;
            observe(Gpt2Q12TraceEvent {
                kind: Gpt2Q12OperationKind::MlpProjection,
                block: Some(block),
                parameter_tensors: [Some(base + 10), Some(base + 11)],
                values: Gpt2Q12Values::Matrix(&projected),
            })?;
            hidden = add_matrices(&residual, &projected)?;
            observe(Gpt2Q12TraceEvent {
                kind: Gpt2Q12OperationKind::MlpResidual,
                block: Some(block),
                parameter_tensors: [None, None],
                values: Gpt2Q12Values::Matrix(&hidden),
            })?;
        }

        let final_base = 2 + model.blocks * 12;
        let normalized = self.layer_norm(&hidden, final_base, final_base + 1, scale)?;
        observe(Gpt2Q12TraceEvent {
            kind: Gpt2Q12OperationKind::FinalLayerNorm,
            block: None,
            parameter_tensors: [Some(final_base), Some(final_base + 1)],
            values: Gpt2Q12Values::Matrix(&normalized),
        })?;
        let last = normalized
            .last()
            .ok_or(Gpt2CheckpointError::InvalidTokenCount(0))?;
        let mut logits = vec![0_i128; model.vocabulary];
        for (feature, &activation) in last.iter().enumerate() {
            self.read_parameter_chunk(0, feature * model.vocabulary, &mut token_row)?;
            for (accumulator, &weight) in logits.iter_mut().zip(&token_row) {
                *accumulator = accumulator
                    .checked_add(i128::from(activation) * i128::from(weight))
                    .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
            }
        }
        let logits = logits
            .into_iter()
            .map(|value| div_floor_i128(value, i128::from(scale)))
            .collect::<Result<Vec<_>, _>>()?;
        observe(Gpt2Q12TraceEvent {
            kind: Gpt2Q12OperationKind::LmHead,
            block: None,
            parameter_tensors: [Some(0), None],
            values: Gpt2Q12Values::Vector(&logits),
        })?;
        let argmax = logits
            .iter()
            .enumerate()
            .max_by_key(|(_, value)| *value)
            .map(|(index, _)| index)
            .ok_or(Gpt2CheckpointError::ProtocolIncomplete)?;
        let checksum = logits.iter().map(|&value| i128::from(value)).sum();
        Ok(Gpt2Q12Execution {
            logits,
            argmax,
            checksum,
        })
    }

    fn read_tensor_i64(&mut self, tensor_index: usize) -> Result<Vec<i64>, Gpt2CheckpointError> {
        let len = *self
            .tensor_lengths
            .get(tensor_index)
            .ok_or(Gpt2CheckpointError::TensorIndex(tensor_index))?;
        let mut values = vec![0_i32; len];
        self.read_parameter_chunk(tensor_index, 0, &mut values)?;
        Ok(values.into_iter().map(i64::from).collect())
    }

    fn linear(
        &mut self,
        input: &[Vec<i64>],
        weight_index: usize,
        bias_index: usize,
        input_width: usize,
        output_width: usize,
        scale: i64,
    ) -> Result<Vec<Vec<i64>>, Gpt2CheckpointError> {
        if input.iter().any(|row| row.len() != input_width) {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let bias = self.read_tensor_i64(bias_index)?;
        if bias.len() != output_width {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let mut accumulators = vec![vec![0_i128; output_width]; input.len()];
        let mut weight_row = vec![0_i32; output_width];
        for input_feature in 0..input_width {
            self.read_parameter_chunk(weight_index, input_feature * output_width, &mut weight_row)?;
            for (token, accumulator) in accumulators.iter_mut().enumerate() {
                let left = i128::from(input[token][input_feature]);
                for (value, &weight) in accumulator.iter_mut().zip(&weight_row) {
                    *value = value
                        .checked_add(left * i128::from(weight))
                        .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
                }
            }
        }
        accumulators
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .zip(&bias)
                    .map(|(value, &bias)| {
                        let numerator = value
                            .checked_add(i128::from(bias) * i128::from(scale))
                            .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
                        div_floor_i128(numerator, i128::from(scale))
                    })
                    .collect()
            })
            .collect()
    }

    fn layer_norm(
        &mut self,
        input: &[Vec<i64>],
        gamma_index: usize,
        beta_index: usize,
        scale: i64,
    ) -> Result<Vec<Vec<i64>>, Gpt2CheckpointError> {
        let width = self.metadata.model.hidden_width;
        if input.iter().any(|row| row.len() != width) {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        let gamma = self.read_tensor_i64(gamma_index)?;
        let beta = self.read_tensor_i64(beta_index)?;
        if gamma.len() != width || beta.len() != width {
            return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
        }
        input
            .iter()
            .map(|row| q12_layer_norm(row, &gamma, &beta, scale))
            .collect()
    }
}

fn q12_values_to_block(values: Gpt2Q12Values<'_>) -> Result<MatrixBlock<i32>, Gpt2CheckpointError> {
    match values {
        Gpt2Q12Values::Matrix(rows) => {
            let columns = rows
                .iter()
                .map(|row| {
                    row.iter()
                        .map(|&value| {
                            i32::try_from(value)
                                .map_err(|_| Gpt2CheckpointError::ActivationOutsideI32(value))
                        })
                        .collect::<Result<Vec<_>, _>>()
                })
                .collect::<Result<Vec<_>, _>>()?;
            let row_count = columns.first().map(Vec::len).unwrap_or(0);
            MatrixBlock::from_columns(MatrixDims::new(row_count, columns.len())?, columns)
                .map_err(Into::into)
        }
        Gpt2Q12Values::Vector(values) => {
            let column = values
                .iter()
                .map(|&value| {
                    i32::try_from(value)
                        .map_err(|_| Gpt2CheckpointError::ActivationOutsideI32(value))
                })
                .collect::<Result<Vec<_>, _>>()?;
            MatrixBlock::from_columns(MatrixDims::new(column.len(), 1)?, vec![column])
                .map_err(Into::into)
        }
    }
}

fn add_matrices(
    left: &[Vec<i64>],
    right: &[Vec<i64>],
) -> Result<Vec<Vec<i64>>, Gpt2CheckpointError> {
    if left.len() != right.len()
        || left
            .iter()
            .zip(right)
            .any(|(left, right)| left.len() != right.len())
    {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            left.iter()
                .zip(right)
                .map(|(&left, &right)| {
                    left.checked_add(right)
                        .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
                })
                .collect()
        })
        .collect()
}

fn q12_layer_norm(
    input: &[i64],
    gamma: &[i64],
    beta: &[i64],
    scale: i64,
) -> Result<Vec<i64>, Gpt2CheckpointError> {
    let width = i128::try_from(input.len()).map_err(|_| Gpt2CheckpointError::ArithmeticOverflow)?;
    let sum = input.iter().try_fold(0_i128, |sum, &value| {
        sum.checked_add(i128::from(value))
            .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
    })?;
    let mean = div_floor_i128(sum, width)?;
    let centered = input
        .iter()
        .map(|&value| {
            i64::try_from(i128::from(value) - i128::from(mean))
                .map_err(|_| Gpt2CheckpointError::ArithmeticOverflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let square_sum = centered.iter().try_fold(0_i128, |sum, &value| {
        sum.checked_add(i128::from(value) * i128::from(value))
            .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
    })?;
    let variance_2q = square_sum.div_euclid(width);
    let variance_q = variance_2q.div_euclid(i128::from(scale));
    let lookup_input =
        i64::try_from(variance_q + 1).map_err(|_| Gpt2CheckpointError::ArithmeticOverflow)?;
    let inverse = q12_rsqrt(lookup_input, scale)?;
    centered
        .into_iter()
        .zip(gamma)
        .zip(beta)
        .map(|((centered, &gamma), &beta)| {
            let normalized = div_floor_i128(
                i128::from(centered) * i128::from(inverse),
                i128::from(scale),
            )?;
            let scaled = div_floor_i128(
                i128::from(normalized) * i128::from(gamma),
                i128::from(scale),
            )?;
            scaled
                .checked_add(beta)
                .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
        })
        .collect()
}

fn q12_attention(
    qkv: &[Vec<i64>],
    model: Gpt2Model,
    scale: i64,
) -> Result<Vec<Vec<i64>>, Gpt2CheckpointError> {
    let tokens = qkv.len();
    if tokens == 0 || qkv.iter().any(|row| row.len() != 3 * model.hidden_width) {
        return Err(Gpt2CheckpointError::ProtocolShapeMismatch);
    }
    let head_width = model.head_width();
    let score_scale = ((scale as f64) / (head_width as f64).sqrt()).round() as i64;
    let mut output = vec![vec![0_i64; model.hidden_width]; tokens];
    for head in 0..model.attention_heads {
        let start = head * head_width;
        for row in 0..tokens {
            let mut scores = Vec::with_capacity(row + 1);
            for column in 0..=row {
                let raw = (0..head_width).try_fold(0_i128, |sum, offset| {
                    sum.checked_add(
                        i128::from(qkv[row][start + offset])
                            * i128::from(qkv[column][model.hidden_width + start + offset]),
                    )
                    .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
                })?;
                let scaled = div_floor_i128(
                    raw.checked_mul(i128::from(score_scale))
                        .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?,
                    i128::from(scale),
                )?;
                scores.push(div_floor_i128(i128::from(scaled), i128::from(scale))?);
            }
            let probabilities = Gpt2Q12Semantics::exact().softmax_row(&scores)?;
            for offset in 0..head_width {
                let raw = probabilities.iter().enumerate().try_fold(
                    0_i128,
                    |sum, (column, &probability)| {
                        sum.checked_add(
                            i128::from(probability)
                                * i128::from(qkv[column][2 * model.hidden_width + start + offset]),
                        )
                        .ok_or(Gpt2CheckpointError::ArithmeticOverflow)
                    },
                )?;
                output[row][start + offset] = div_floor_i128(raw, i128::from(scale))?;
            }
        }
    }
    Ok(output)
}

fn q12_gelu(value: i64, scale: i64) -> Result<i64, Gpt2CheckpointError> {
    let clamped = value.clamp(-8 * scale, 8 * scale);
    let x = clamped as f64 / scale as f64;
    let inner = (2.0 / std::f64::consts::PI).sqrt() * (x + 0.044_715 * x.powi(3));
    rounded_i64(0.5 * x * (1.0 + inner.tanh()) * scale as f64)
}

fn q12_exp(value: i64, scale: i64) -> Result<i64, Gpt2CheckpointError> {
    let clamped = value.clamp(-16 * scale, 0);
    rounded_i64((clamped as f64 / scale as f64).exp() * scale as f64)
}

fn q12_reciprocal(value: i64, tokens: usize, scale: i64) -> Result<i64, Gpt2CheckpointError> {
    let maximum = i64::try_from(tokens)
        .ok()
        .and_then(|tokens| tokens.checked_mul(scale))
        .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
    let clamped = value.clamp(1, maximum);
    let numerator = i128::from(scale)
        .checked_mul(i128::from(scale))
        .ok_or(Gpt2CheckpointError::ArithmeticOverflow)?;
    i64::try_from(numerator.div_euclid(i128::from(clamped)))
        .map_err(|_| Gpt2CheckpointError::ArithmeticOverflow)
}

fn q12_rsqrt(value: i64, scale: i64) -> Result<i64, Gpt2CheckpointError> {
    let clamped = value.clamp(1, 1 << 25);
    rounded_i64(((scale as f64).powi(3) / clamped as f64).sqrt())
}

fn rounded_i64(value: f64) -> Result<i64, Gpt2CheckpointError> {
    if !value.is_finite() || value < i64::MIN as f64 || value > i64::MAX as f64 {
        return Err(Gpt2CheckpointError::ArithmeticOverflow);
    }
    Ok(value.round() as i64)
}

fn div_floor_i128(numerator: i128, denominator: i128) -> Result<i64, Gpt2CheckpointError> {
    if denominator <= 0 {
        return Err(Gpt2CheckpointError::ProtocolIncomplete);
    }
    i64::try_from(numerator.div_euclid(denominator))
        .map_err(|_| Gpt2CheckpointError::ArithmeticOverflow)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Gpt2CheckpointError {
    Io(io::ErrorKind),
    Inventory(Gpt2InventoryError),
    InvalidMagic,
    UnsupportedVersion(u32),
    LegacyCheckpointSemantics(u32),
    UnsupportedProofProtocolVersion(u32),
    UnsupportedFractionalBits(u32),
    ParameterCountMismatch {
        expected: u64,
        actual: u64,
    },
    ParameterPayloadByteCountMismatch {
        expected: u64,
        actual: u64,
    },
    InvalidTokenCount(usize),
    InvalidTokenId {
        index: usize,
        token: usize,
    },
    PublicTokenCountMismatch {
        expected: usize,
        actual: usize,
    },
    PublicLogitCountMismatch {
        expected: usize,
        actual: usize,
    },
    PublicStatementDigestMismatch,
    ArtifactOversized(usize),
    ArtifactTrailingBytes,
    LookupRegistryMismatch,
    DeclaredRangeViolation {
        block: usize,
        value_bits: u16,
    },
    TensorIndex(usize),
    ParameterChunkOutOfBounds {
        tensor_index: usize,
        tensor_len: usize,
        entry_offset: usize,
        chunk_len: usize,
    },
    OffsetOverflow,
    ProtocolShapeMismatch,
    ProtocolIncomplete,
    ArithmeticOverflow,
    ActivationOutsideI32(i64),
    Heterogeneous(HeteroMatrixError),
    Bva(BvaError),
}

impl Display for Gpt2CheckpointError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for Gpt2CheckpointError {}

impl From<io::Error> for Gpt2CheckpointError {
    fn from(error: io::Error) -> Self {
        Self::Io(error.kind())
    }
}

impl From<Gpt2InventoryError> for Gpt2CheckpointError {
    fn from(error: Gpt2InventoryError) -> Self {
        Self::Inventory(error)
    }
}

impl From<HeteroMatrixError> for Gpt2CheckpointError {
    fn from(error: HeteroMatrixError) -> Self {
        Self::Heterogeneous(error)
    }
}

impl From<BvaError> for Gpt2CheckpointError {
    fn from(error: BvaError) -> Self {
        Self::Bva(error)
    }
}

fn read_array<const N: usize>(reader: &mut impl Read) -> Result<[u8; N], io::Error> {
    let mut bytes = [0_u8; N];
    reader.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn read_u32(reader: &mut impl Read) -> Result<u32, io::Error> {
    Ok(u32::from_le_bytes(read_array(reader)?))
}

fn read_u64(reader: &mut impl Read) -> Result<u64, io::Error> {
    Ok(u64::from_le_bytes(read_array(reader)?))
}

fn read_artifact_usize(reader: &mut impl Read) -> Result<usize, Gpt2CheckpointError> {
    usize::try_from(read_u64(reader)?).map_err(|_| Gpt2CheckpointError::ProtocolShapeMismatch)
}

fn read_i32(reader: &mut impl Read) -> Result<i32, io::Error> {
    Ok(i32::from_le_bytes(read_array(reader)?))
}

fn read_i64(reader: &mut impl Read) -> Result<i64, io::Error> {
    Ok(i64::from_le_bytes(read_array(reader)?))
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use super::{
        Gpt2GlobalBvaWitnessSource, Gpt2GlobalLayout, Gpt2GlobalWitnessSource, Gpt2Model,
        Gpt2ProofTranscript, Gpt2PublicStatement, Gpt2Q12Checkpoint, Gpt2Q12Metadata,
        Gpt2Q12OperationDescriptor, Gpt2Q12OperationKind, Gpt2Q12Semantics, Gpt2Variant,
    };
    use crate::protocols::virtualpcs::{
        commit_bva_i32_streaming, commit_stacked_i32, commit_stacked_i32_streaming,
    };
    use crate::utils::heterogeneous::{HeteroBlockMat, MatrixBlock, MatrixDims};
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_poly_commit::smart_pc::SmartPC;

    fn q12_header_bytes() -> Vec<u8> {
        q12_header_bytes_for(Gpt2Model::small(), 4, 124_439_808)
    }

    fn q12_header_bytes_for(model: Gpt2Model, token_count: u32, parameter_entries: u64) -> Vec<u8> {
        q12_header_bytes_for_version(model, token_count, parameter_entries, 2)
    }

    fn q12_header_bytes_for_version(
        model: Gpt2Model,
        token_count: u32,
        parameter_entries: u64,
        version: u32,
    ) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"PR1GPT2\0");
        bytes.extend_from_slice(&version.to_le_bytes());
        bytes.extend_from_slice(&12_u32.to_le_bytes());
        bytes.extend_from_slice(&parameter_entries.to_le_bytes());
        for value in [
            token_count,
            model.vocabulary as u32,
            model.maximum_positions as u32,
            model.hidden_width as u32,
            model.attention_heads as u32,
            model.mlp_width as u32,
            model.blocks as u32,
        ] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        bytes.extend_from_slice(&[0x24; 32]);
        bytes.extend_from_slice(&[0x84; 32]);
        bytes
    }

    #[test]
    fn q12_checkpoint_metadata_parses_the_canonical_small_format() {
        let metadata = Gpt2Q12Metadata::read_from(&mut Cursor::new(q12_header_bytes())).unwrap();

        assert_eq!(metadata.model, Gpt2Model::small());
        assert_eq!(metadata.format_version, 2);
        assert_eq!(metadata.fractional_bits, 12);
        assert_eq!(metadata.scale(), 4_096);
        assert_eq!(metadata.parameter_entries, 124_439_808);
        assert_eq!(metadata.token_count, 4);
        assert_eq!(metadata.checkpoint_sha256, [0x24; 32]);
        assert_eq!(metadata.tokenizer_sha256, [0x84; 32]);
    }

    #[test]
    fn q12_checkpoint_reader_exposes_reference_and_streams_parameter_chunks() {
        let mut bytes = q12_header_bytes();
        for token in [15_496_u32, 995, 11, 616] {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        for index in 0..50_257 {
            let logit = match index {
                0 => -7_i64,
                50_256 => 9_i64,
                _ => 0_i64,
            };
            bytes.extend_from_slice(&logit.to_le_bytes());
        }
        for value in [7_i32, -8, 9] {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let mut checkpoint = Gpt2Q12Checkpoint::from_reader(Cursor::new(bytes)).unwrap();
        assert_eq!(checkpoint.token_ids(), &[15_496, 995, 11, 616]);
        assert_eq!(checkpoint.reference_logits()[0], -7);
        assert_eq!(checkpoint.reference_logits()[50_256], 9);

        let mut values = [0_i32; 3];
        checkpoint.read_parameter_chunk(0, 0, &mut values).unwrap();
        assert_eq!(values, [7, -8, 9]);
    }

    #[test]
    fn q12_softmax_has_exact_fixed_point_mass_with_deterministic_correction() {
        let semantics = Gpt2Q12Semantics::exact();

        let probabilities = semantics.softmax_row(&[-8 * 4_096, -7 * 4_096]).unwrap();

        assert_eq!(probabilities, vec![1_101, 2_995]);
        assert_eq!(probabilities.iter().sum::<i64>(), 4_096);
    }

    #[test]
    fn public_statement_binds_identity_io_semantics_manifest_and_bva_layout() {
        let model = Gpt2Model {
            blocks: 1,
            hidden_width: 2,
            attention_heads: 1,
            mlp_width: 4,
            vocabulary: 3,
            maximum_positions: 2,
        };
        let parameter_entries = model.physical_parameter_entries().unwrap();
        let metadata = Gpt2Q12Metadata::read_from(&mut Cursor::new(q12_header_bytes_for(
            model,
            2,
            parameter_entries as u64,
        )))
        .unwrap();
        let statement = Gpt2PublicStatement::new(&metadata, &[0, 1], &[-4_094, 4_094, 0]).unwrap();

        assert!(statement.validate().is_ok());
        assert_ne!(statement.operation_manifest_digest, [0_u8; 32]);
        assert_ne!(statement.bva_layout_digest, [0_u8; 32]);
        assert_ne!(statement.lookup_registry_digest, [0_u8; 32]);
        assert_ne!(statement.relation_manifest_digest, [0_u8; 32]);
        let relation_shape = statement.relation_shape().unwrap();
        assert_eq!(relation_shape.top_level_equation_count(), 4);
        assert!(relation_shape.linear_relations > 0);
        assert!(relation_shape.bilinear_relations > 0);
        assert!(relation_shape.tagged_lookup_relations > 0);
        assert!(relation_shape.auxiliary_blocks > 0);

        let original_digest = statement.transcript_digest();
        let mut changed_token = statement.clone();
        changed_token.token_ids[0] = 1;
        assert_ne!(changed_token.transcript_digest(), original_digest);
        let mut changed_logit = statement.clone();
        changed_logit.public_logits[0] += 1;
        assert_ne!(changed_logit.transcript_digest(), original_digest);
        let mut changed_checkpoint = statement.clone();
        changed_checkpoint.checkpoint_sha256[0] ^= 1;
        assert_ne!(changed_checkpoint.transcript_digest(), original_digest);

        let mut prover_transcript = Gpt2ProofTranscript::new(&statement);
        prover_transcript.absorb(b"witness-commitment", &Fr::from(7_u64));
        let prover_challenge = prover_transcript.challenge::<Fr>(b"linear-routing");
        let mut verifier_transcript = Gpt2ProofTranscript::new(&statement);
        verifier_transcript.absorb(b"witness-commitment", &Fr::from(7_u64));
        assert_eq!(
            verifier_transcript.challenge::<Fr>(b"linear-routing"),
            prover_challenge
        );
        let mut tampered_transcript = Gpt2ProofTranscript::new(&statement);
        tampered_transcript.absorb(b"witness-commitment", &Fr::from(8_u64));
        assert_ne!(
            tampered_transcript.challenge::<Fr>(b"linear-routing"),
            prover_challenge
        );

        let mut invalid_manifest = statement;
        invalid_manifest.operation_manifest_digest[0] ^= 1;
        assert!(invalid_manifest.validate().is_err());

        let mut invalid_relations = changed_checkpoint;
        invalid_relations.relation_manifest_digest[0] ^= 1;
        assert!(invalid_relations.validate().is_err());
    }

    #[test]
    fn legacy_q12_bundle_can_be_read_for_migration_but_not_used_as_a_statement() {
        let model = Gpt2Model {
            blocks: 1,
            hidden_width: 2,
            attention_heads: 1,
            mlp_width: 4,
            vocabulary: 3,
            maximum_positions: 2,
        };
        let parameter_entries = model.physical_parameter_entries().unwrap();
        let metadata = Gpt2Q12Metadata::read_from(&mut Cursor::new(q12_header_bytes_for_version(
            model,
            2,
            parameter_entries as u64,
            1,
        )))
        .unwrap();

        assert_eq!(metadata.format_version, 1);
        assert!(Gpt2PublicStatement::new(&metadata, &[0, 1], &[0, 0, 0]).is_err());
    }

    #[test]
    fn legacy_bundle_can_be_streamed_into_v2_with_the_exact_softmax_reference() {
        let model = Gpt2Model {
            blocks: 1,
            hidden_width: 2,
            attention_heads: 1,
            mlp_width: 4,
            vocabulary: 3,
            maximum_positions: 2,
        };
        let parameter_entries = model.physical_parameter_entries().unwrap();
        let mut bytes = q12_header_bytes_for_version(model, 2, parameter_entries as u64, 1);
        for token in [0_u32, 1] {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        for logit in [-4_094_i64, 4_094, 0] {
            bytes.extend_from_slice(&logit.to_le_bytes());
        }
        let mut parameters = vec![0_i32; parameter_entries];
        parameters[..6].copy_from_slice(&[4_096, 0, -4_096, 0, 4_096, -4_096]);
        parameters[64..66].copy_from_slice(&[4_096, 4_096]);
        for value in parameters {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        let mut legacy = Gpt2Q12Checkpoint::from_reader(Cursor::new(bytes)).unwrap();
        let execution = legacy.execute().unwrap();
        let mut migrated_bytes = Vec::new();

        legacy
            .write_bundle_with_reference(&mut migrated_bytes, &execution.logits)
            .unwrap();

        let migrated = Gpt2Q12Checkpoint::from_reader(Cursor::new(migrated_bytes)).unwrap();
        assert_eq!(migrated.metadata().format_version, 2);
        assert_eq!(migrated.reference_logits(), execution.logits);
        assert!(Gpt2PublicStatement::new(
            migrated.metadata(),
            migrated.token_ids(),
            migrated.reference_logits(),
        )
        .is_ok());
    }

    #[test]
    fn q12_executor_matches_a_hand_checked_full_gpt2_trace() {
        let model = Gpt2Model {
            blocks: 1,
            hidden_width: 2,
            attention_heads: 1,
            mlp_width: 4,
            vocabulary: 3,
            maximum_positions: 2,
        };
        let parameter_entries = model.physical_parameter_entries().unwrap();
        assert_eq!(parameter_entries, 68);
        let mut bytes = q12_header_bytes_for(model, 2, parameter_entries as u64);
        for token in [0_u32, 1] {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        for logit in [-4_094_i64, 4_094, 0] {
            bytes.extend_from_slice(&logit.to_le_bytes());
        }

        let mut parameters = vec![0_i32; parameter_entries];
        parameters[..6].copy_from_slice(&[4_096, 0, -4_096, 0, 4_096, -4_096]);
        parameters[64..66].copy_from_slice(&[4_096, 4_096]);
        for value in parameters {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let mut checkpoint = Gpt2Q12Checkpoint::from_reader(Cursor::new(bytes)).unwrap();
        let expected_trace = checkpoint.metadata().operation_manifest().unwrap();
        let mut trace = Vec::new();
        let execution = checkpoint
            .execute_with_trace(|event| {
                trace.push(Gpt2Q12OperationDescriptor {
                    kind: event.kind,
                    block: event.block,
                    rows: event.values.rows(),
                    cols: event.values.cols(),
                    parameter_tensors: event.parameter_tensors,
                });
                Ok(())
            })
            .unwrap();

        assert_eq!(execution.logits, vec![-4_094, 4_094, 0]);
        assert_eq!(execution.argmax, 1);
        assert_eq!(execution.checksum, 0);
        assert_eq!(execution.logits, checkpoint.reference_logits());
        assert_eq!(trace, expected_trace);
        assert_eq!(trace.len(), 13);
        assert_eq!(
            trace.iter().map(|event| event.kind).collect::<Vec<_>>(),
            vec![
                Gpt2Q12OperationKind::Embedding,
                Gpt2Q12OperationKind::LayerNorm1,
                Gpt2Q12OperationKind::QkvProjection,
                Gpt2Q12OperationKind::CausalAttention,
                Gpt2Q12OperationKind::AttentionProjection,
                Gpt2Q12OperationKind::AttentionResidual,
                Gpt2Q12OperationKind::LayerNorm2,
                Gpt2Q12OperationKind::MlpExpansion,
                Gpt2Q12OperationKind::Gelu,
                Gpt2Q12OperationKind::MlpProjection,
                Gpt2Q12OperationKind::MlpResidual,
                Gpt2Q12OperationKind::FinalLayerNorm,
                Gpt2Q12OperationKind::LmHead,
            ]
        );
        assert_eq!(trace[0].parameter_tensors, [Some(0), Some(1)]);
        assert_eq!(trace[1].block, Some(0));
        assert_eq!(trace[12].rows, 1);
        assert_eq!(trace[12].cols, 3);
    }

    #[test]
    fn global_layout_uses_one_namespace_and_reuses_residual_and_tied_weight_ids() {
        let model = Gpt2Model {
            blocks: 1,
            hidden_width: 2,
            attention_heads: 1,
            mlp_width: 4,
            vocabulary: 3,
            maximum_positions: 2,
        };
        let parameter_entries = model.physical_parameter_entries().unwrap();
        let metadata = Gpt2Q12Metadata::read_from(&mut Cursor::new(q12_header_bytes_for(
            model,
            2,
            parameter_entries as u64,
        )))
        .unwrap();

        let layout = Gpt2GlobalLayout::compile(&metadata).unwrap();
        assert_eq!(layout.witness_block_count(), 30);
        assert_eq!(layout.model_blocks().len(), 16);
        assert_eq!(layout.activation_entries().unwrap(), 69);
        assert_eq!(layout.physical_entries(&metadata).unwrap(), 137);
        assert_eq!(layout.logical_padded_entries().unwrap(), 1_024);
        assert_eq!(layout.public_input_block().0, 16);
        assert_eq!(layout.public_output_block().0, 29);
        assert_eq!((layout.blocks()[0].rows, layout.blocks()[0].cols), (3, 2));
        assert_eq!((layout.blocks()[4].rows, layout.blocks()[4].cols), (6, 2));
        assert_eq!((layout.blocks()[19].rows, layout.blocks()[19].cols), (6, 2));
        assert_eq!((layout.blocks()[29].rows, layout.blocks()[29].cols), (3, 1));

        let operations = layout.operations();
        assert_eq!(operations.len(), 13);
        assert_eq!(operations[0].output.0, 17);
        assert_eq!(operations[5].inputs[0].unwrap().0, 17);
        assert_eq!(operations[5].inputs[1].unwrap().0, 21);
        assert_eq!(operations[5].output.0, 22);
        assert_eq!(operations[10].inputs[0].unwrap().0, 22);
        assert_eq!(operations[10].inputs[1].unwrap().0, 26);
        assert_eq!(operations[10].output.0, 27);
        assert_eq!(operations[12].inputs[0].unwrap().0, 28);
        assert_eq!(operations[0].parameter_blocks[0].unwrap().0, 0);
        assert_eq!(operations[12].parameter_blocks[0].unwrap().0, 0);
    }

    #[test]
    fn global_bva_layout_pads_columns_logically_instead_of_rectangularly() {
        let model = Gpt2Model {
            blocks: 1,
            hidden_width: 2,
            attention_heads: 1,
            mlp_width: 4,
            vocabulary: 3,
            maximum_positions: 2,
        };
        let parameter_entries = model.physical_parameter_entries().unwrap();
        let metadata = Gpt2Q12Metadata::read_from(&mut Cursor::new(q12_header_bytes_for(
            model,
            2,
            parameter_entries as u64,
        )))
        .unwrap();
        let layout = Gpt2GlobalLayout::compile(&metadata).unwrap();

        let bva = layout.compile_bva().unwrap();

        assert_eq!(bva.vector_count(), 50);
        assert_eq!(bva.physical_entries(), 137);
        assert!(bva.logical_entries().is_power_of_two());
        assert!(bva.logical_entries() < 8 * bva.physical_entries());
        assert!(bva.logical_entries() < layout.logical_padded_entries().unwrap());
        assert_eq!(
            bva.logical_padding_entries(),
            bva.logical_entries() - bva.physical_entries()
        );
        assert_eq!(bva.materialized_padding_entries(), 0);
        assert_eq!(bva.vectors()[0].block.0, 0);
        assert_eq!(bva.vectors()[0].column, 0);
        assert_eq!(bva.vectors()[0].physical_len, 3);
    }

    #[test]
    fn activation_witness_uses_global_ids_and_column_major_i32_blocks() {
        let model = Gpt2Model {
            blocks: 1,
            hidden_width: 2,
            attention_heads: 1,
            mlp_width: 4,
            vocabulary: 3,
            maximum_positions: 2,
        };
        let parameter_entries = model.physical_parameter_entries().unwrap();
        let mut bytes = q12_header_bytes_for(model, 2, parameter_entries as u64);
        for token in [0_u32, 1] {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        for logit in [-4_094_i64, 4_094, 0] {
            bytes.extend_from_slice(&logit.to_le_bytes());
        }
        let mut parameters = vec![0_i32; parameter_entries];
        parameters[..6].copy_from_slice(&[4_096, 0, -4_096, 0, 4_096, -4_096]);
        parameters[64..66].copy_from_slice(&[4_096, 4_096]);
        for value in parameters {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let mut checkpoint = Gpt2Q12Checkpoint::from_reader(Cursor::new(bytes)).unwrap();
        let layout = Gpt2GlobalLayout::compile(checkpoint.metadata()).unwrap();
        let (execution, witness) = checkpoint.execute_activation_witness(&layout).unwrap();

        assert_eq!(execution.logits, vec![-4_094, 4_094, 0]);
        assert_eq!(witness.stored_entries(), 69);
        assert!(witness.block(layout.model_blocks()[0]).is_none());
        let tokens = witness.block(layout.public_input_block()).unwrap();
        assert_eq!((tokens.dims.rows, tokens.dims.cols), (2, 1));
        assert_eq!(tokens.columns, vec![vec![0, 1]]);
        let logits = witness.block(layout.public_output_block()).unwrap();
        assert_eq!((logits.dims.rows, logits.dims.cols), (3, 1));
        assert_eq!(logits.columns, vec![vec![-4_094, 4_094, 0]]);

        let mut embedding_column = vec![0_i32; 3];
        checkpoint
            .read_model_block_column(&layout, layout.model_blocks()[0], 0, &mut embedding_column)
            .unwrap();
        assert_eq!(embedding_column, vec![4_096, 0, -4_096]);
        checkpoint
            .read_model_block_column(&layout, layout.model_blocks()[0], 1, &mut embedding_column)
            .unwrap();
        assert_eq!(embedding_column, vec![0, 4_096, -4_096]);
        let mut final_gamma = vec![0_i32; 2];
        checkpoint
            .read_model_block_column(&layout, layout.model_blocks()[14], 0, &mut final_gamma)
            .unwrap();
        assert_eq!(final_gamma, vec![4_096, 4_096]);

        let mut parameter_checksum = 0_i64;
        let profile = checkpoint
            .for_each_model_block_column(&layout, |_, _, column| {
                parameter_checksum += column.iter().map(|&value| i64::from(value)).sum::<i64>();
            })
            .unwrap();
        assert_eq!(profile.physical_entries, 68);
        assert_eq!(profile.max_live_entries, 6);
        assert_eq!(parameter_checksum, 8_192);
    }

    #[test]
    fn global_witness_source_commits_parameters_and_activations_as_one_matrix() {
        let model = Gpt2Model {
            blocks: 1,
            hidden_width: 2,
            attention_heads: 1,
            mlp_width: 4,
            vocabulary: 3,
            maximum_positions: 2,
        };
        let parameter_entries = model.physical_parameter_entries().unwrap();
        let mut bytes = q12_header_bytes_for(model, 2, parameter_entries as u64);
        for token in [0_u32, 1] {
            bytes.extend_from_slice(&token.to_le_bytes());
        }
        for logit in [-4_094_i64, 4_094, 0] {
            bytes.extend_from_slice(&logit.to_le_bytes());
        }
        let mut parameters = vec![0_i32; parameter_entries];
        parameters[..6].copy_from_slice(&[4_096, 0, -4_096, 0, 4_096, -4_096]);
        parameters[64..66].copy_from_slice(&[4_096, 4_096]);
        for value in parameters {
            bytes.extend_from_slice(&value.to_le_bytes());
        }

        let mut checkpoint = Gpt2Q12Checkpoint::from_reader(Cursor::new(bytes)).unwrap();
        let layout = Gpt2GlobalLayout::compile(checkpoint.metadata()).unwrap();
        let (_, activations) = checkpoint.execute_activation_witness(&layout).unwrap();
        let mut materialized_blocks = activations.blocks.clone();
        for &block_id in layout.model_blocks() {
            let descriptor = layout.blocks()[block_id.0];
            let mut columns = Vec::with_capacity(descriptor.cols);
            for column in 0..descriptor.cols {
                let mut values = vec![0_i32; descriptor.rows];
                checkpoint
                    .read_model_block_column(&layout, block_id, column, &mut values)
                    .unwrap();
                columns.push(values);
            }
            materialized_blocks[block_id.0] = Some(
                MatrixBlock::from_columns(
                    MatrixDims::new(descriptor.rows, descriptor.cols).unwrap(),
                    columns,
                )
                .unwrap(),
            );
        }
        let materialized =
            HeteroBlockMat::new(layout.padded_domain().unwrap(), materialized_blocks).unwrap();

        let mut rng = ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup(5, &mut rng).unwrap();
        let hiding = Fr::from(73_u64);
        let expected = commit_stacked_i32::<Bls12_381>(&pp, &materialized, hiding, 32).unwrap();
        let mut source =
            Gpt2GlobalWitnessSource::new(&mut checkpoint, &layout, &activations).unwrap();
        let (commitment, tier_one, profile) =
            commit_stacked_i32_streaming::<Bls12_381, _>(&pp, &mut source, hiding, 32).unwrap();

        assert_eq!((commitment, tier_one), expected);
        assert_eq!(profile.physical_entries, 137);
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(profile.padding_scalar_multiplications, 0);
        drop(source);

        let bva = layout.compile_bva().unwrap();
        let mut expanded = vec![0_i32; bva.logical_entries()];
        for vector in bva.vectors() {
            let values = &materialized.blocks[vector.block.0]
                .as_ref()
                .unwrap()
                .columns[vector.column];
            expanded[vector.logical_offset..vector.logical_offset + values.len()]
                .copy_from_slice(values);
        }
        let expanded_block = MatrixBlock::from_columns(
            MatrixDims::new(bva.logical_entries(), 1).unwrap(),
            vec![expanded],
        )
        .unwrap();
        let expanded_matrix = HeteroBlockMat::new(
            crate::utils::heterogeneous::PaddedDomain::new(1, bva.logical_entries(), 1).unwrap(),
            vec![Some(expanded_block)],
        )
        .unwrap();
        let q = (bva.logical_entries().ilog2() as usize + 1) / 2;
        let pp = SmartPC::<Bls12_381>::setup(q, &mut rng).unwrap();
        let expected = commit_stacked_i32::<Bls12_381>(&pp, &expanded_matrix, hiding, 32).unwrap();
        let mut bva_source =
            Gpt2GlobalBvaWitnessSource::new(&mut checkpoint, &layout, &bva, &activations).unwrap();
        let (commitment, tier_one, profile) =
            commit_bva_i32_streaming::<Bls12_381, _>(&pp, &mut bva_source, hiding, 32).unwrap();

        assert_eq!((commitment, tier_one), expected);
        assert_eq!(profile.physical_entries, 137);
        assert_eq!(profile.logical_entries, bva.logical_entries());
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(profile.padding_group_operations, 0);
    }

    #[test]
    fn gpt2_small_inventory_counts_tied_parameters_once() {
        let model = Gpt2Model::small();
        let tensors = model.parameter_tensors().unwrap();

        assert_eq!(tensors.len(), 148);
        assert_eq!(model.physical_parameter_entries().unwrap(), 124_439_808);
        assert_eq!(model.lm_head_entries(), 50_257 * 768);
        assert!(model.lm_head_is_tied_view());
    }

    #[test]
    fn standard_gpt2_variants_have_canonical_physical_parameter_counts() {
        let cases = [
            (Gpt2Variant::Small, 12, 768, 12, 148, 124_439_808),
            (Gpt2Variant::Medium, 24, 1_024, 16, 292, 354_823_168),
            (Gpt2Variant::Large, 36, 1_280, 20, 436, 774_030_080),
            (Gpt2Variant::Xl, 48, 1_600, 25, 580, 1_557_611_200),
        ];

        for (variant, blocks, width, heads, tensor_count, parameter_count) in cases {
            let model = Gpt2Model::for_variant(variant);
            assert_eq!(model.blocks, blocks);
            assert_eq!(model.hidden_width, width);
            assert_eq!(model.attention_heads, heads);
            assert_eq!(model.parameter_tensors().unwrap().len(), tensor_count);
            assert_eq!(model.physical_parameter_entries().unwrap(), parameter_count);
        }
    }

    #[test]
    fn gpt2_variant_names_round_trip_for_experiment_cli() {
        for name in ["small", "medium", "large", "xl"] {
            let variant = name.parse::<Gpt2Variant>().unwrap();
            assert_eq!(variant.as_str(), name);
        }
        assert!("1.5b".parse::<Gpt2Variant>().is_err());
    }
}
