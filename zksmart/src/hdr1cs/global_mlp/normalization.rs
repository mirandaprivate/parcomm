use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::{UniformRand, Zero};
use std::collections::{BTreeMap, BTreeSet};
use std::convert::TryFrom;
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;

use super::{
    commit_global_artifacts, derive_global_point, matrix_artifact, open_committed_matrix,
    prove_addition_equation, prove_direct_route, prove_multiplication_equation,
    prove_selected_route, prove_with_witness_claims, prove_witness_batch, push_block,
    push_vector_block, verify_addition_equation, verify_io_equation,
    verify_multiplication_equation, verify_selected_route, verify_witness_batch, ActivationAux,
    CommittedArtifact, CommittedGlobalMlp, CommittedMatrix, GlobalBlockId, GlobalIoProof,
    GlobalLinearProof, GlobalMlpEncoding, GlobalMlpError, GlobalMlpInstance, GlobalMlpShape,
    GlobalMultiplicationProof, GlobalOperation, GlobalOperationType, GlobalWitnessBatchProof,
    MatrixArtifact, ProverWitnessClaim, RoutedOpeningProof, VerifierWitnessClaim,
};
use crate::data_structures::SRS;
use crate::hdr1cs::mlp::MlpManifest;
use crate::protocols::{diagonal::ZkDiagonal, mul::ZkMatMul};
use crate::utils::fiat_shamir::FiatShamir;
use crate::utils::heterogeneous::{
    ColumnZeroHeteroView, ElementwiseDiagonalHeteroView, HeteroBlockMat, HeteroMatrixError,
    MatrixBlock, MatrixDims, PaddedDomain,
};
use crate::utils::zktr::{TranSeq, ZkTranSeq};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum LookupTag {
    Exp,
    Reciprocal,
    InverseSqrt,
    NonNegative,
    Boolean,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaggedLookupTable {
    pub tag: LookupTag,
    pub rows: Vec<(i32, i32)>,
}

impl TaggedLookupTable {
    pub fn new(tag: LookupTag, rows: Vec<(i32, i32)>) -> Self {
        Self { tag, rows }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NormalizationManifest {
    pub width: usize,
    pub epsilon: i32,
    pub output_shift: usize,
    pub softmax_output_shift: usize,
    pub commitment_bit_width: usize,
    pub tables: Vec<TaggedLookupTable>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NormalizationInputs {
    pub layer_norm_input: Vec<i32>,
    pub layer_norm_gamma: Vec<i32>,
    pub layer_norm_beta: Vec<i32>,
    pub softmax_scores: Vec<i32>,
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum OperationFamily {
    Linear,
    Bilinear,
    TaggedLookup,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum LinearRelationKind {
    ReduceSum,
    Affine,
    Requantize,
    RangeRecompose,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BilinearRelationKind {
    Hadamard,
    Boolean,
    SelectorZero,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TaggedLookupQuery {
    tag: LookupTag,
    input: i32,
    output: i32,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct TaggedLookupOperation {
    tag: LookupTag,
    input: GlobalBlockId,
    output: GlobalBlockId,
}

pub struct GlobalNormalizationEncoding<E: Pairing> {
    manifest: NormalizationManifest,
    inputs: NormalizationInputs,
    layer_norm_output: Vec<i32>,
    softmax_output: Vec<i32>,
    linear_relations: Vec<LinearRelationKind>,
    bilinear_relations: Vec<BilinearRelationKind>,
    lookup_queries: Vec<TaggedLookupQuery>,
    lookup_operations: Vec<TaggedLookupOperation>,
    base: GlobalMlpEncoding<E>,
    _marker: PhantomData<E>,
}

impl<E: Pairing> GlobalNormalizationEncoding<E> {
    pub fn layer_norm_output(&self) -> &[i32] {
        &self.layer_norm_output
    }

    pub fn softmax_output(&self) -> &[i32] {
        &self.softmax_output
    }

    pub fn operation_family_count(&self) -> usize {
        3
    }

    pub fn top_level_equation_count(&self) -> usize {
        1 + self.operation_family_count()
    }

    pub fn operation_count(&self, family: OperationFamily) -> usize {
        match family {
            OperationFamily::Linear => self.linear_relations.len(),
            OperationFamily::Bilinear => self.bilinear_relations.len(),
            OperationFamily::TaggedLookup => self.lookup_queries.len(),
        }
    }

    pub fn lookup_count(&self, tag: LookupTag) -> usize {
        self.lookup_queries
            .iter()
            .filter(|query| query.tag == tag)
            .count()
    }

    pub fn stored_semantic_entries(&self) -> usize {
        self.inputs.layer_norm_input.len()
            + self.inputs.layer_norm_gamma.len()
            + self.inputs.layer_norm_beta.len()
            + self.inputs.softmax_scores.len()
            + self.layer_norm_output.len()
            + self.softmax_output.len()
    }

    pub fn stored_witness_entries(&self) -> usize {
        self.base.stored_witness_entries()
    }

    pub fn logical_padded_witness_entries(&self) -> usize {
        let shape = self.base.global_shape();
        shape.witness_slots * shape.n * shape.n
    }

    pub fn witness_block_count(&self) -> usize {
        self.base.witness_block_count()
    }

    pub fn global_shape(&self) -> GlobalMlpShape {
        self.base.global_shape()
    }

    pub fn manifest(&self) -> &NormalizationManifest {
        &self.manifest
    }
}

pub struct GlobalHdR1csNormalizationProtocol<E: Pairing> {
    _marker: PhantomData<E>,
}

struct LookupArtifacts<F>
where
    F: ark_ff::PrimeField + From<i32>,
{
    selector: MatrixArtifact<F>,
    a_l: MatrixArtifact<F>,
    a_o: MatrixArtifact<F>,
}

struct CommittedLookupArtifacts<E: Pairing> {
    selector: CommittedArtifact<E>,
    a_l: CommittedArtifact<E>,
    a_o: CommittedArtifact<E>,
}

pub struct GlobalNormalizationInstance<E: Pairing> {
    base: GlobalMlpInstance<E>,
    manifest: NormalizationManifest,
    inputs: NormalizationInputs,
    layer_norm_output: Vec<i32>,
    softmax_output: Vec<i32>,
    lookup_tags: Vec<LookupTag>,
    lookup_selector: PairingOutput<E>,
    lookup_a_l: PairingOutput<E>,
    lookup_a_o: PairingOutput<E>,
}

pub struct GlobalElementwiseProof<E: Pairing> {
    output_value_commitment: E::G1,
    transcript: TranSeq<E::ScalarField, E::G1>,
    left_route: RoutedOpeningProof<E>,
    right_route: RoutedOpeningProof<E>,
    output_route: RoutedOpeningProof<E>,
}

struct RevealedLookupBlock {
    tag: LookupTag,
    input: Vec<i32>,
    output: Vec<i32>,
}

pub struct TransparentTaggedLookupProof<E: Pairing> {
    blocks: Vec<RevealedLookupBlock>,
    input_route: RoutedOpeningProof<E>,
    output_route: RoutedOpeningProof<E>,
}

pub struct GlobalNormalizationProof<E: Pairing> {
    io: GlobalIoProof<E>,
    matrix_multiplication: GlobalMultiplicationProof<E>,
    linear: GlobalLinearProof<E>,
    elementwise: GlobalElementwiseProof<E>,
    lookup: TransparentTaggedLookupProof<E>,
    witness_batch: GlobalWitnessBatchProof<E>,
}

impl<E: Pairing> GlobalNormalizationProof<E> {
    pub fn witness_commitment_count(&self) -> usize {
        1
    }

    pub fn witness_opening_count(&self) -> usize {
        1
    }

    pub fn top_level_equation_count(&self) -> usize {
        4
    }

    pub fn serialized_size_bytes(&self) -> usize {
        let elementwise = super::serialized_size(&self.elementwise.output_value_commitment)
            + super::serialized_size(&self.elementwise.transcript)
            + super::proof_size_route(&self.elementwise.left_route)
            + super::proof_size_route(&self.elementwise.right_route)
            + super::proof_size_route(&self.elementwise.output_route);
        let revealed_lookup_rows = self
            .lookup
            .blocks
            .iter()
            .map(|block| {
                std::mem::size_of::<u8>()
                    + std::mem::size_of::<u64>()
                    + std::mem::size_of_val(block.input.as_slice())
                    + std::mem::size_of_val(block.output.as_slice())
            })
            .sum::<usize>();
        let lookup = revealed_lookup_rows
            + super::proof_size_route(&self.lookup.input_route)
            + super::proof_size_route(&self.lookup.output_route);
        super::proof_size_io(&self.io)
            + super::proof_size_multiplication(&self.matrix_multiplication)
            + super::proof_size_linear(&self.linear)
            + elementwise
            + lookup
            + super::serialized_size(&self.witness_batch.transcript)
            + super::serialized_size(&self.witness_batch.opening)
    }
}

impl<E: Pairing> GlobalHdR1csNormalizationProtocol<E> {
    pub fn encode(
        manifest: NormalizationManifest,
        inputs: NormalizationInputs,
    ) -> Result<GlobalNormalizationEncoding<E>, NormalizationError> {
        let tables = validate_manifest_and_inputs(&manifest, &inputs)?;
        let mut lookup_queries = Vec::new();

        let width_i64 =
            i64::try_from(manifest.width).map_err(|_| NormalizationError::ArithmeticOverflow)?;
        let layer_sum = inputs
            .layer_norm_input
            .iter()
            .try_fold(0_i64, |sum, value| sum.checked_add(i64::from(*value)))
            .ok_or(NormalizationError::ArithmeticOverflow)?;
        let centered = inputs
            .layer_norm_input
            .iter()
            .map(|value| {
                i64::from(*value)
                    .checked_mul(width_i64)
                    .and_then(|scaled| scaled.checked_sub(layer_sum))
                    .and_then(|result| i32::try_from(result).ok())
                    .ok_or(NormalizationError::ArithmeticOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let variance = centered
            .iter()
            .try_fold(i64::from(manifest.epsilon), |sum, value| {
                i64::from(*value)
                    .checked_mul(i64::from(*value))
                    .and_then(|square| sum.checked_add(square))
            })
            .and_then(|value| i32::try_from(value).ok())
            .ok_or(NormalizationError::ArithmeticOverflow)?;
        let inverse_sqrt = lookup(
            &tables,
            LookupTag::InverseSqrt,
            variance,
            &mut lookup_queries,
        )?;
        let gamma_centered = centered
            .iter()
            .zip(&inputs.layer_norm_gamma)
            .map(|(centered, gamma)| {
                centered
                    .checked_mul(*gamma)
                    .ok_or(NormalizationError::ArithmeticOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let layer_scaled = gamma_centered
            .iter()
            .map(|value| {
                value
                    .checked_mul(inverse_sqrt)
                    .ok_or(NormalizationError::ArithmeticOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let layer_numerators = layer_scaled
            .iter()
            .zip(&inputs.layer_norm_beta)
            .map(|(scaled, beta)| {
                scaled
                    .checked_add(*beta)
                    .ok_or(NormalizationError::ArithmeticOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let layer_requantized = layer_numerators
            .iter()
            .map(|value| requantize(i64::from(*value), manifest.output_shift))
            .collect::<Result<Vec<_>, _>>()?;
        let layer_norm_output = layer_requantized
            .iter()
            .map(|(quotient, _)| *quotient)
            .collect::<Vec<_>>();
        let layer_remainders = layer_requantized
            .iter()
            .map(|(_, remainder)| *remainder)
            .collect::<Vec<_>>();

        let maximum = *inputs
            .softmax_scores
            .iter()
            .max()
            .ok_or(NormalizationError::EmptyInput)?;
        let maximum_index = inputs
            .softmax_scores
            .iter()
            .position(|score| *score == maximum)
            .ok_or(NormalizationError::EmptyInput)?;
        let shifted = inputs
            .softmax_scores
            .iter()
            .map(|score| {
                score
                    .checked_sub(maximum)
                    .ok_or(NormalizationError::ArithmeticOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let exponentials = shifted
            .iter()
            .map(|value| lookup(&tables, LookupTag::Exp, *value, &mut lookup_queries))
            .collect::<Result<Vec<_>, _>>()?;
        let denominator = exponentials
            .iter()
            .try_fold(0_i32, |sum, value| sum.checked_add(*value))
            .ok_or(NormalizationError::ArithmeticOverflow)?;
        let reciprocal = lookup(
            &tables,
            LookupTag::Reciprocal,
            denominator,
            &mut lookup_queries,
        )?;
        let softmax_products = exponentials
            .iter()
            .map(|value| {
                value
                    .checked_mul(reciprocal)
                    .ok_or(NormalizationError::ArithmeticOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let softmax_requantized = softmax_products
            .iter()
            .map(|value| requantize(i64::from(*value), manifest.softmax_output_shift))
            .collect::<Result<Vec<_>, _>>()?;
        let softmax_base_output = softmax_requantized
            .iter()
            .map(|(quotient, _)| *quotient)
            .collect::<Vec<_>>();
        let softmax_remainders = softmax_requantized
            .iter()
            .map(|(_, remainder)| *remainder)
            .collect::<Vec<_>>();
        let softmax_unit = 1_i32 << manifest.softmax_output_shift;
        let softmax_base_sum = softmax_base_output
            .iter()
            .try_fold(0_i32, |sum, value| sum.checked_add(*value))
            .ok_or(NormalizationError::ArithmeticOverflow)?;
        let softmax_residual = softmax_unit
            .checked_sub(softmax_base_sum)
            .ok_or(NormalizationError::ArithmeticOverflow)?;
        if softmax_residual < 0 {
            return Err(NormalizationError::NegativeSoftmaxResidual(
                softmax_residual,
            ));
        }
        let mut softmax_output = softmax_base_output.clone();
        softmax_output[maximum_index] = softmax_output[maximum_index]
            .checked_add(softmax_residual)
            .ok_or(NormalizationError::ArithmeticOverflow)?;

        let layer_squares = centered
            .iter()
            .map(|value| {
                value
                    .checked_mul(*value)
                    .ok_or(NormalizationError::ArithmeticOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let maximum_differences = inputs
            .softmax_scores
            .iter()
            .map(|score| {
                maximum
                    .checked_sub(*score)
                    .ok_or(NormalizationError::ArithmeticOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut one_hot = vec![0_i32; manifest.width];
        one_hot[maximum_index] = 1;

        let (base, lookup_operations) = build_base_encoding::<E>(
            &manifest,
            &inputs,
            &layer_norm_output,
            &softmax_output,
            &softmax_base_output,
            &centered,
            &layer_squares,
            variance,
            inverse_sqrt,
            &gamma_centered,
            &layer_scaled,
            &layer_numerators,
            &layer_remainders,
            maximum,
            &maximum_differences,
            &one_hot,
            &shifted,
            &exponentials,
            denominator,
            reciprocal,
            &softmax_products,
            &softmax_remainders,
            softmax_residual,
        )?;

        for value in maximum_differences
            .iter()
            .chain(layer_remainders.iter())
            .chain(softmax_remainders.iter())
            .chain(std::iter::once(&softmax_residual))
        {
            let output = lookup(&tables, LookupTag::NonNegative, *value, &mut lookup_queries)?;
            if output != *value {
                return Err(NormalizationError::NonIdentityRangeTable(*value));
            }
        }

        // These relation records are proof-IR families, not per-layer proofs.
        // All LayerNorm and Softmax instances will append to the same vectors.
        let linear_relations = vec![
            LinearRelationKind::ReduceSum,
            LinearRelationKind::Affine,
            LinearRelationKind::ReduceSum,
            LinearRelationKind::Requantize,
            LinearRelationKind::Affine,
            LinearRelationKind::RangeRecompose,
            LinearRelationKind::ReduceSum,
            LinearRelationKind::Requantize,
            LinearRelationKind::RangeRecompose,
            LinearRelationKind::Affine,
            LinearRelationKind::ReduceSum,
        ];
        let bilinear_relations = vec![
            BilinearRelationKind::Hadamard,
            BilinearRelationKind::Hadamard,
            BilinearRelationKind::Hadamard,
            BilinearRelationKind::Boolean,
            BilinearRelationKind::SelectorZero,
            BilinearRelationKind::Hadamard,
            BilinearRelationKind::Hadamard,
        ];

        Ok(GlobalNormalizationEncoding {
            manifest,
            inputs,
            layer_norm_output,
            softmax_output,
            linear_relations,
            bilinear_relations,
            lookup_queries,
            lookup_operations,
            base,
            _marker: PhantomData,
        })
    }
}

impl<E: Pairing> GlobalHdR1csNormalizationProtocol<E>
where
    E::ScalarField: From<i32>,
{
    pub fn setup(encoding: &GlobalNormalizationEncoding<E>) -> Result<SRS<E>, NormalizationError> {
        let artifacts = encoding.base.artifacts()?;
        let lookup = lookup_artifacts(encoding)?;
        let mut maximum_axis = 2_usize;
        for matrix in [
            &artifacts.witness,
            &artifacts.io,
            &artifacts.a_l.virtual_matrix,
            &artifacts.a_r.virtual_matrix,
            &artifacts.a_o.virtual_matrix,
            &artifacts.io_selector.virtual_matrix,
            &artifacts.negative.virtual_matrix,
            &lookup.selector.virtual_matrix,
            &lookup.a_l.virtual_matrix,
            &lookup.a_o.virtual_matrix,
        ] {
            let (rows, cols) = matrix.packed_stacked_shape();
            maximum_axis = maximum_axis.max(rows).max(cols);
        }
        for matrix in artifacts
            .selectors
            .values()
            .chain(artifacts.output_bits.iter())
            .chain(artifacts.negative_bits.iter())
        {
            let (rows, cols) = matrix.virtual_matrix.packed_stacked_shape();
            maximum_axis = maximum_axis.max(rows).max(cols);
        }
        Ok(SRS::<E>::setup(maximum_axis.ilog2() as usize))
    }

    pub fn prove(
        srs: &SRS<E>,
        encoding: GlobalNormalizationEncoding<E>,
    ) -> Result<(GlobalNormalizationInstance<E>, GlobalNormalizationProof<E>), NormalizationError>
    {
        let shape = encoding.base.shape();
        let lookup_artifacts = lookup_artifacts(&encoding)?;
        let committed_lookup = CommittedLookupArtifacts {
            selector: commit_public_artifact(srs, lookup_artifacts.selector)?,
            a_l: commit_public_artifact(srs, lookup_artifacts.a_l)?,
            a_o: commit_public_artifact(srs, lookup_artifacts.a_o)?,
        };
        let mut committed = commit_global_artifacts(srs, encoding.base.artifacts()?)?;
        bind_public_architecture(srs, &encoding.base, &mut committed)?;
        let base_instance = committed.instance(
            encoding.base.manifest.clone(),
            shape,
            encoding.base.public_input.clone(),
            encoding.base.public_output.clone(),
        );
        let instance = GlobalNormalizationInstance {
            base: base_instance,
            manifest: encoding.manifest.clone(),
            inputs: encoding.inputs.clone(),
            layer_norm_output: encoding.layer_norm_output.clone(),
            softmax_output: encoding.softmax_output.clone(),
            lookup_tags: encoding
                .lookup_operations
                .iter()
                .map(|operation| operation.tag)
                .collect(),
            lookup_selector: committed_lookup.selector.committed.commitment,
            lookup_a_l: committed_lookup.a_l.committed.commitment,
            lookup_a_o: committed_lookup.a_o.committed.commitment,
        };

        let (io, mut witness_claims) = prove_with_witness_claims(|claims| {
            prove_public_io_equation(srs, &instance.base, &committed, claims)
        })?;
        let (matrix_multiplication, claims) = prove_with_witness_claims(|claims| {
            prove_multiplication_equation(srs, &instance.base, &encoding.base, &committed, claims)
        })?;
        witness_claims.extend(claims);
        let (linear, claims) = prove_with_witness_claims(|claims| {
            prove_addition_equation(srs, &instance.base, &encoding.base, &committed, claims)
        })?;
        witness_claims.extend(claims);
        let (elementwise, claims) = prove_with_witness_claims(|claims| {
            prove_elementwise_equation(srs, &instance.base, &encoding.base, &committed, claims)
        })?;
        witness_claims.extend(claims);
        let mut lookup_claims = Vec::new();
        let lookup = prove_transparent_lookup(
            srs,
            &instance,
            &encoding,
            &committed,
            &committed_lookup,
            &mut lookup_claims,
        )?;
        witness_claims.extend(lookup_claims);
        let witness_batch = prove_witness_batch(srs, &committed.witness, witness_claims)?;

        Ok((
            instance,
            GlobalNormalizationProof {
                io,
                matrix_multiplication,
                linear,
                elementwise,
                lookup,
                witness_batch,
            },
        ))
    }

    pub fn verify(
        srs: &SRS<E>,
        instance: &GlobalNormalizationInstance<E>,
        proof: &GlobalNormalizationProof<E>,
    ) -> bool {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            verify_normalization_inner(srs, instance, proof)
        }))
        .unwrap_or(false)
    }
}

fn lookup_artifacts<E: Pairing>(
    encoding: &GlobalNormalizationEncoding<E>,
) -> Result<LookupArtifacts<E::ScalarField>, GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    let shape = encoding.base.shape();
    let mut selector = super::zero_columns(shape.gate_slots, shape.gate_slots);
    let mut a_l = super::zero_columns(shape.gate_slots, shape.witness_slots);
    let mut a_o = super::zero_columns(shape.gate_slots, shape.witness_slots);
    for (row, operation) in encoding.lookup_operations.iter().enumerate() {
        selector[row][row] = 1;
        a_l[operation.input.0][row] = 1;
        a_o[operation.output.0][row] = 1;
    }
    Ok(LookupArtifacts {
        selector: matrix_artifact(selector)?,
        a_l: matrix_artifact(a_l)?,
        a_o: matrix_artifact(a_o)?,
    })
}

fn commit_public_artifact<E: Pairing>(
    srs: &SRS<E>,
    artifact: MatrixArtifact<E::ScalarField>,
) -> Result<CommittedArtifact<E>, GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    Ok(CommittedArtifact {
        committed: CommittedMatrix::commit(
            srs,
            artifact.virtual_matrix,
            E::ScalarField::zero(),
            1,
        )?,
        dense: artifact.dense,
    })
}

fn bind_public_architecture<E: Pairing>(
    srs: &SRS<E>,
    encoding: &GlobalMlpEncoding<E>,
    committed: &mut CommittedGlobalMlp<E>,
) -> Result<(), GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    let artifacts = encoding.artifacts()?;
    committed.a_l = commit_public_artifact(srs, artifacts.a_l)?;
    committed.a_r = commit_public_artifact(srs, artifacts.a_r)?;
    committed.a_o = commit_public_artifact(srs, artifacts.a_o)?;
    committed.selectors = artifacts
        .selectors
        .into_iter()
        .map(|(operation, artifact)| {
            commit_public_artifact(srs, artifact).map(|committed| (operation, committed))
        })
        .collect::<Result<_, _>>()?;
    committed.io_selector = commit_public_artifact(srs, artifacts.io_selector)?;
    committed.negative = commit_public_artifact(srs, artifacts.negative)?;
    committed.output_bits = artifacts
        .output_bits
        .into_iter()
        .map(|artifact| commit_public_artifact(srs, artifact))
        .collect::<Result<_, _>>()?;
    committed.negative_bits = artifacts
        .negative_bits
        .into_iter()
        .map(|artifact| commit_public_artifact(srs, artifact))
        .collect::<Result<_, _>>()?;
    Ok(())
}

fn public_architecture_matches<E: Pairing>(
    srs: &SRS<E>,
    expected: &GlobalNormalizationEncoding<E>,
    instance: &GlobalNormalizationInstance<E>,
) -> Result<bool, NormalizationError>
where
    E::ScalarField: From<i32>,
{
    let artifacts = expected.base.artifacts()?;
    let a_l = commit_public_artifact(srs, artifacts.a_l)?;
    let a_r = commit_public_artifact(srs, artifacts.a_r)?;
    let a_o = commit_public_artifact(srs, artifacts.a_o)?;
    let selectors = artifacts
        .selectors
        .into_iter()
        .map(|(operation, artifact)| {
            commit_public_artifact(srs, artifact)
                .map(|committed| (operation, committed.committed.commitment))
        })
        .collect::<Result<BTreeMap<_, _>, _>>()?;
    let io_selector = commit_public_artifact(srs, artifacts.io_selector)?;
    let negative = commit_public_artifact(srs, artifacts.negative)?;
    let output_bits = artifacts
        .output_bits
        .into_iter()
        .map(|artifact| {
            commit_public_artifact(srs, artifact).map(|artifact| artifact.committed.commitment)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let negative_bits = artifacts
        .negative_bits
        .into_iter()
        .map(|artifact| {
            commit_public_artifact(srs, artifact).map(|artifact| artifact.committed.commitment)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let lookup = lookup_artifacts(expected)?;
    let lookup_selector = commit_public_artifact(srs, lookup.selector)?;
    let lookup_a_l = commit_public_artifact(srs, lookup.a_l)?;
    let lookup_a_o = commit_public_artifact(srs, lookup.a_o)?;

    Ok(instance.base.architecture.a_l == a_l.committed.commitment
        && instance.base.architecture.a_r == a_r.committed.commitment
        && instance.base.architecture.a_o == a_o.committed.commitment
        && instance.base.architecture.selectors == selectors
        && instance.base.architecture.io_selector == io_selector.committed.commitment
        && instance.base.architecture.negative == negative.committed.commitment
        && instance.base.architecture.output_bits == output_bits
        && instance.base.architecture.negative_bits == negative_bits
        && instance.lookup_selector == lookup_selector.committed.commitment
        && instance.lookup_a_l == lookup_a_l.committed.commitment
        && instance.lookup_a_o == lookup_a_o.committed.commitment)
}

/// Bind the committed I/O matrix to the public normalization statement at one
/// Fiat-Shamir point. The zero-hiding value commitment lets the verifier
/// independently recompute that public evaluation, while the opening and route
/// still connect it to the committed I/O matrix and the single global witness.
fn prove_public_io_equation<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    committed: &CommittedGlobalMlp<E>,
    witness_claims: &mut Vec<ProverWitnessClaim<E>>,
) -> Result<GlobalIoProof<E>, GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    let point = derive_global_point::<E>(
        instance,
        0,
        instance.shape.witness_slots * instance.shape.n,
        instance.shape.n,
    );
    let value = committed.io.matrix.proj_stacked_lr(&point.0, &point.1)?;
    let value_hiding = E::ScalarField::zero();
    let value_commitment = srs.zksrs.commit(value, value_hiding);
    let io_opening =
        open_committed_matrix(srs, &committed.io, &point, value_commitment, value_hiding)?;
    let route = prove_direct_route(
        srs,
        value_commitment,
        value_hiding,
        point,
        &committed.io_selector.committed,
        &committed.witness,
        &committed.io_selector.dense,
        instance.shape.n,
        witness_claims,
    )?;
    Ok(GlobalIoProof {
        value_commitment,
        io_opening,
        route,
    })
}

fn prove_elementwise_equation<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    encoding: &GlobalMlpEncoding<E>,
    committed: &CommittedGlobalMlp<E>,
    witness_claims: &mut Vec<ProverWitnessClaim<E>>,
) -> Result<GlobalElementwiseProof<E>, GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    let left = encoding.routed_matrix(
        GlobalOperationType::Activation,
        super::GlobalBlockRole::Left,
    )?;
    let right = encoding.routed_activation_aux(super::ActivationRoute::Negative)?;
    let output = encoding.routed_matrix(
        GlobalOperationType::Activation,
        super::GlobalBlockRole::Output,
    )?;
    let rows = instance.shape.gate_slots * instance.shape.n;
    let point = derive_global_point::<E>(instance, 0x4e42494c, rows, 1);
    let output_point = super::expand_column_zero_point(point.clone(), instance.shape.n);
    let output_value = output.proj_stacked_lr(&output_point.0, &output_point.1)?;
    let output_hiding = E::ScalarField::rand(&mut ark_std::rand::thread_rng());
    let output_value_commitment = srs.zksrs.commit(output_value, output_hiding);
    let mut transcript = ZkTranSeq::new(&srs.zksrs);
    let product = ZkMatMul::<
        i32,
        E::ScalarField,
        E::G1,
        ElementwiseDiagonalHeteroView<'_, i32>,
        ColumnZeroHeteroView<'_, i32>,
    >::new(output_value_commitment, point, rows);
    let left_diagonal = ElementwiseDiagonalHeteroView::new(&left);
    let right_vector = ColumnZeroHeteroView::new(&right);
    let (
        left_diagonal_commitment,
        right_commitment,
        left_diagonal_hiding,
        right_hiding,
        left_diagonal_point,
        right_point,
    ) = product.reduce_prover(
        &srs.zksrs,
        &mut transcript,
        &left_diagonal,
        &right_vector,
        output_hiding,
    );
    let diagonal = ZkDiagonal::<i32, E::ScalarField, E::G1, ColumnZeroHeteroView<'_, i32>>::new(
        left_diagonal_commitment,
        left_diagonal_point,
        rows,
    );
    let left_vector = ColumnZeroHeteroView::new(&left);
    let (left_commitment, left_hiding, left_point) = diagonal.reduce_prover(
        &srs.zksrs,
        &mut transcript,
        &left_vector,
        left_diagonal_hiding,
    );
    let selector = &committed.selectors[&GlobalOperationType::Activation];
    let left_route = prove_selected_route(
        srs,
        left_commitment,
        left_hiding,
        super::expand_column_zero_point(left_point, instance.shape.n),
        &selector.committed,
        &committed.a_l.committed,
        &committed.witness,
        &selector.dense,
        &committed.a_l.dense,
        instance.shape.n,
        witness_claims,
    )?;
    let right_route = prove_selected_route(
        srs,
        right_commitment,
        right_hiding,
        super::expand_column_zero_point(right_point, instance.shape.n),
        &selector.committed,
        &committed.negative.committed,
        &committed.witness,
        &selector.dense,
        &committed.negative.dense,
        instance.shape.n,
        witness_claims,
    )?;
    let output_route = prove_selected_route(
        srs,
        output_value_commitment,
        output_hiding,
        output_point,
        &selector.committed,
        &committed.a_o.committed,
        &committed.witness,
        &selector.dense,
        &committed.a_o.dense,
        instance.shape.n,
        witness_claims,
    )?;
    Ok(GlobalElementwiseProof {
        output_value_commitment,
        transcript: transcript.publish_trans(),
        left_route,
        right_route,
        output_route,
    })
}

fn verify_elementwise_equation<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    proof: &GlobalElementwiseProof<E>,
    witness_claims: &mut Vec<VerifierWitnessClaim<E>>,
) -> bool
where
    E::ScalarField: From<i32>,
{
    let rows = instance.shape.gate_slots * instance.shape.n;
    let point = derive_global_point::<E>(instance, 0x4e42494c, rows, 1);
    let mut transcript = proof.transcript.clone();
    let product = ZkMatMul::<
        i32,
        E::ScalarField,
        E::G1,
        ElementwiseDiagonalHeteroView<'static, i32>,
        ColumnZeroHeteroView<'static, i32>,
    >::new(proof.output_value_commitment, point.clone(), rows);
    let (product_ok, left_diagonal_commitment, right_commitment, left_diagonal_point, right_point) =
        product.verify_as_subprotocol(&srs.zksrs, &mut transcript);
    if !product_ok {
        return false;
    }
    let diagonal =
        ZkDiagonal::<i32, E::ScalarField, E::G1, ColumnZeroHeteroView<'static, i32>>::new(
            left_diagonal_commitment,
            left_diagonal_point,
            rows,
        );
    let (diagonal_ok, left_commitment, left_point) =
        diagonal.verify_as_subprotocol(&srs.zksrs, &mut transcript);
    if !diagonal_ok || !transcript.check_fiat_shamir() || !transcript.is_fully_consumed() {
        return false;
    }
    let selector = instance.architecture.selectors[&GlobalOperationType::Activation];
    verify_selected_route(
        srs,
        &proof.left_route,
        left_commitment,
        super::expand_column_zero_point(left_point, instance.shape.n),
        selector,
        instance.architecture.a_l,
        instance.shape,
        witness_claims,
    ) && verify_selected_route(
        srs,
        &proof.right_route,
        right_commitment,
        super::expand_column_zero_point(right_point, instance.shape.n),
        selector,
        instance.architecture.negative,
        instance.shape,
        witness_claims,
    ) && verify_selected_route(
        srs,
        &proof.output_route,
        proof.output_value_commitment,
        super::expand_column_zero_point(point, instance.shape.n),
        selector,
        instance.architecture.a_o,
        instance.shape,
        witness_claims,
    )
}

fn prove_transparent_lookup<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalNormalizationInstance<E>,
    encoding: &GlobalNormalizationEncoding<E>,
    committed: &CommittedGlobalMlp<E>,
    lookup: &CommittedLookupArtifacts<E>,
    witness_claims: &mut Vec<ProverWitnessClaim<E>>,
) -> Result<TransparentTaggedLookupProof<E>, NormalizationError>
where
    E::ScalarField: From<i32>,
{
    let mut blocks = Vec::with_capacity(encoding.lookup_operations.len());
    let mut input_blocks = vec![None; instance.base.shape.gate_slots];
    let mut output_blocks = vec![None; instance.base.shape.gate_slots];
    for (index, operation) in encoding.lookup_operations.iter().enumerate() {
        let input = block_column(&encoding.base, operation.input)?.to_vec();
        let output = block_column(&encoding.base, operation.output)?.to_vec();
        input_blocks[index] = Some(MatrixBlock::from_columns(
            MatrixDims::new(input.len(), 1)?,
            vec![input.clone()],
        )?);
        output_blocks[index] = Some(MatrixBlock::from_columns(
            MatrixDims::new(output.len(), 1)?,
            vec![output.clone()],
        )?);
        blocks.push(RevealedLookupBlock {
            tag: operation.tag,
            input,
            output,
        });
    }
    let point = derive_lookup_point::<E>(instance, &blocks);
    let input_matrix = routed_lookup_blocks(instance.base.shape, input_blocks)?;
    let output_matrix = routed_lookup_blocks(instance.base.shape, output_blocks)?;
    let input_value = input_matrix.proj_stacked_lr(&point.0, &point.1)?;
    let output_value = output_matrix.proj_stacked_lr(&point.0, &point.1)?;
    let input_commitment = srs.zksrs.commit(input_value, E::ScalarField::zero());
    let output_commitment = srs.zksrs.commit(output_value, E::ScalarField::zero());
    let input_route = prove_selected_route(
        srs,
        input_commitment,
        E::ScalarField::zero(),
        point.clone(),
        &lookup.selector.committed,
        &lookup.a_l.committed,
        &committed.witness,
        &lookup.selector.dense,
        &lookup.a_l.dense,
        instance.base.shape.n,
        witness_claims,
    )?;
    let output_route = prove_selected_route(
        srs,
        output_commitment,
        E::ScalarField::zero(),
        point,
        &lookup.selector.committed,
        &lookup.a_o.committed,
        &committed.witness,
        &lookup.selector.dense,
        &lookup.a_o.dense,
        instance.base.shape.n,
        witness_claims,
    )?;
    Ok(TransparentTaggedLookupProof {
        blocks,
        input_route,
        output_route,
    })
}

fn verify_transparent_lookup<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalNormalizationInstance<E>,
    proof: &TransparentTaggedLookupProof<E>,
    witness_claims: &mut Vec<VerifierWitnessClaim<E>>,
) -> bool
where
    E::ScalarField: From<i32>,
{
    if proof.blocks.len() != instance.lookup_tags.len() {
        return false;
    }
    let tables = match public_tables(&instance.manifest) {
        Some(tables) => tables,
        None => return false,
    };
    let mut input_blocks = vec![None; instance.base.shape.gate_slots];
    let mut output_blocks = vec![None; instance.base.shape.gate_slots];
    for (index, (block, expected_tag)) in proof.blocks.iter().zip(&instance.lookup_tags).enumerate()
    {
        if block.tag != *expected_tag
            || block.input.is_empty()
            || block.input.len() != block.output.len()
            || block.input.len() > instance.base.shape.n
        {
            return false;
        }
        let Some(table) = tables.get(&block.tag) else {
            return false;
        };
        if block
            .input
            .iter()
            .zip(&block.output)
            .any(|(input, output)| table.get(input) != Some(output))
        {
            return false;
        }
        input_blocks[index] = match MatrixDims::new(block.input.len(), 1)
            .and_then(|dims| MatrixBlock::from_columns(dims, vec![block.input.clone()]))
        {
            Ok(block) => Some(block),
            Err(_) => return false,
        };
        output_blocks[index] = match MatrixDims::new(block.output.len(), 1)
            .and_then(|dims| MatrixBlock::from_columns(dims, vec![block.output.clone()]))
        {
            Ok(block) => Some(block),
            Err(_) => return false,
        };
    }
    let point = derive_lookup_point::<E>(instance, &proof.blocks);
    let input_matrix = match routed_lookup_blocks(instance.base.shape, input_blocks) {
        Ok(matrix) => matrix,
        Err(_) => return false,
    };
    let output_matrix = match routed_lookup_blocks(instance.base.shape, output_blocks) {
        Ok(matrix) => matrix,
        Err(_) => return false,
    };
    let input_value = match input_matrix.proj_stacked_lr(&point.0, &point.1) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let output_value = match output_matrix.proj_stacked_lr(&point.0, &point.1) {
        Ok(value) => value,
        Err(_) => return false,
    };
    let input_commitment = srs.zksrs.commit(input_value, E::ScalarField::zero());
    let output_commitment = srs.zksrs.commit(output_value, E::ScalarField::zero());
    let input_ok = verify_selected_route(
        srs,
        &proof.input_route,
        input_commitment,
        point.clone(),
        instance.lookup_selector,
        instance.lookup_a_l,
        instance.base.shape,
        witness_claims,
    );
    let output_ok = verify_selected_route(
        srs,
        &proof.output_route,
        output_commitment,
        point,
        instance.lookup_selector,
        instance.lookup_a_o,
        instance.base.shape,
        witness_claims,
    );
    input_ok && output_ok
}

fn block_column<E: Pairing>(
    encoding: &GlobalMlpEncoding<E>,
    id: GlobalBlockId,
) -> Result<&[i32], NormalizationError> {
    encoding
        .blocks
        .get(id.0)
        .and_then(|block| block.columns.first())
        .map(Vec::as_slice)
        .ok_or(NormalizationError::InvalidLookupBlock)
}

fn routed_lookup_blocks(
    shape: GlobalMlpShape,
    blocks: Vec<Option<MatrixBlock<i32>>>,
) -> Result<HeteroBlockMat<i32>, HeteroMatrixError> {
    if blocks.len() > shape.gate_slots {
        return Err(HeteroMatrixError::TooManyBlocks {
            slots: shape.gate_slots,
            blocks: blocks.len(),
        });
    }
    HeteroBlockMat::new(
        PaddedDomain::new(shape.gate_slots, shape.n, shape.n)?,
        blocks,
    )
}

fn derive_lookup_point<E: Pairing>(
    instance: &GlobalNormalizationInstance<E>,
    blocks: &[RevealedLookupBlock],
) -> (Vec<E::ScalarField>, Vec<E::ScalarField>) {
    let mut fs = FiatShamir::new();
    fs.push(&0x4e4c5554_usize);
    for size in [
        instance.base.shape.n,
        instance.base.shape.witness_slots,
        instance.base.shape.gate_slots,
    ] {
        fs.push(&size);
    }
    fs.push(&instance.base.witness_commitment);
    fs.push(&instance.lookup_selector);
    fs.push(&instance.lookup_a_l);
    fs.push(&instance.lookup_a_o);
    fs.push(&blocks.len());
    for block in blocks {
        fs.push(&(block.tag as usize));
        fs.push(&block.input.len());
        for value in block.input.iter().chain(&block.output) {
            fs.push(&(*value as u32));
        }
    }
    for table in &instance.manifest.tables {
        fs.push(&(table.tag as usize));
        fs.push(&table.rows.len());
        for (key, value) in &table.rows {
            fs.push(&(*key as u32));
            fs.push(&(*value as u32));
        }
    }
    let rows = instance.base.shape.gate_slots * instance.base.shape.n;
    let cols = instance.base.shape.n;
    let left = (0..rows.ilog2())
        .map(|_| fs.gen_challenge::<E::ScalarField>())
        .collect();
    let right = (0..cols.ilog2())
        .map(|_| fs.gen_challenge::<E::ScalarField>())
        .collect();
    (left, right)
}

fn public_tables(
    manifest: &NormalizationManifest,
) -> Option<BTreeMap<LookupTag, BTreeMap<i32, i32>>> {
    let mut tables = BTreeMap::new();
    for table in &manifest.tables {
        let mut rows = BTreeMap::new();
        for (input, output) in &table.rows {
            if rows.insert(*input, *output).is_some() {
                return None;
            }
        }
        if tables.insert(table.tag, rows).is_some() {
            return None;
        }
    }
    Some(tables)
}

fn verify_normalization_inner<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalNormalizationInstance<E>,
    proof: &GlobalNormalizationProof<E>,
) -> bool
where
    E::ScalarField: From<i32>,
{
    let expected = match GlobalHdR1csNormalizationProtocol::<E>::encode(
        instance.manifest.clone(),
        instance.inputs.clone(),
    ) {
        Ok(encoding) => encoding,
        Err(_) => return false,
    };
    if expected.layer_norm_output != instance.layer_norm_output
        || expected.softmax_output != instance.softmax_output
        || expected
            .lookup_operations
            .iter()
            .map(|operation| operation.tag)
            .collect::<Vec<_>>()
            != instance.lookup_tags
        || instance.base.shape != expected.base.shape()
        || ![
            GlobalOperationType::Multiplication,
            GlobalOperationType::Addition,
            GlobalOperationType::Activation,
        ]
        .iter()
        .all(|operation| instance.base.architecture.selectors.contains_key(operation))
    {
        return false;
    }
    if !matches!(
        public_architecture_matches(srs, &expected, instance),
        Ok(true)
    ) {
        return false;
    }

    let expected_artifacts = match expected.base.artifacts() {
        Ok(artifacts) => artifacts,
        Err(_) => return false,
    };
    let public_io_point = derive_global_point::<E>(
        &instance.base,
        0,
        instance.base.shape.witness_slots * instance.base.shape.n,
        instance.base.shape.n,
    );
    let public_io_value = match expected_artifacts
        .io
        .proj_stacked_lr(&public_io_point.0, &public_io_point.1)
    {
        Ok(value) => value,
        Err(_) => return false,
    };
    if proof.io.value_commitment != srs.zksrs.commit(public_io_value, E::ScalarField::zero()) {
        return false;
    }

    let mut witness_claims = Vec::new();
    let io = verify_io_equation(srs, &instance.base, &proof.io, &mut witness_claims);
    let multiplication = verify_multiplication_equation(
        srs,
        &instance.base,
        &proof.matrix_multiplication,
        &mut witness_claims,
    );
    let linear = verify_addition_equation(srs, &instance.base, &proof.linear, &mut witness_claims);
    let elementwise =
        verify_elementwise_equation(srs, &instance.base, &proof.elementwise, &mut witness_claims);
    let lookup = verify_transparent_lookup(srs, instance, &proof.lookup, &mut witness_claims);
    io && multiplication
        && linear
        && elementwise
        && lookup
        && verify_witness_batch(
            srs,
            instance.base.witness_commitment,
            &proof.witness_batch,
            witness_claims,
        )
}

#[allow(clippy::too_many_arguments)]
fn build_base_encoding<E: Pairing>(
    manifest: &NormalizationManifest,
    inputs: &NormalizationInputs,
    layer_norm_output: &[i32],
    softmax_output: &[i32],
    softmax_base_output: &[i32],
    centered: &[i32],
    layer_squares: &[i32],
    variance: i32,
    inverse_sqrt: i32,
    gamma_centered: &[i32],
    layer_scaled: &[i32],
    layer_numerators: &[i32],
    layer_remainders: &[i32],
    maximum: i32,
    maximum_differences: &[i32],
    one_hot: &[i32],
    shifted: &[i32],
    exponentials: &[i32],
    denominator: i32,
    reciprocal: i32,
    softmax_products: &[i32],
    softmax_remainders: &[i32],
    softmax_residual: i32,
) -> Result<(GlobalMlpEncoding<E>, Vec<TaggedLookupOperation>), NormalizationError> {
    let width = manifest.width;
    let padded_width = width.next_power_of_two();
    let mut blocks = Vec::new();
    let mut io_blocks = Vec::new();
    let mut multiplications = Vec::new();
    let mut additions = Vec::new();
    let mut hadamards = Vec::new();
    let mut activation_aux = Vec::new();
    let mut lookups = Vec::new();

    let mut public_vector = |values: Vec<i32>| -> Result<GlobalBlockId, NormalizationError> {
        let id = push_vector_block(&mut blocks, values)?;
        io_blocks.push(id);
        Ok(id)
    };
    let x = public_vector(inputs.layer_norm_input.clone())?;
    let gamma = public_vector(inputs.layer_norm_gamma.clone())?;
    let beta = public_vector(inputs.layer_norm_beta.clone())?;
    let scores = public_vector(inputs.softmax_scores.clone())?;
    let layer_output = public_vector(layer_norm_output.to_vec())?;
    let softmax_output_id = public_vector(softmax_output.to_vec())?;
    let negative_ones = public_vector(vec![-1; width])?;
    let width_values = public_vector(vec![
        i32::try_from(width).map_err(|_| {
            NormalizationError::ArithmeticOverflow
        })?;
        width
    ])?;
    let layer_divisor = public_vector(vec![1_i32 << manifest.output_shift; width])?;
    let softmax_divisor = public_vector(vec![1_i32 << manifest.softmax_output_shift; width])?;
    let softmax_unit = public_vector(vec![1_i32 << manifest.softmax_output_shift])?;
    let zero = public_vector(vec![0; width])?;
    let one = public_vector(vec![1])?;
    let epsilon = public_vector(vec![manifest.epsilon])?;
    drop(public_vector);

    let ones_row = MatrixBlock::from_columns(
        MatrixDims::new(1, width)?,
        (0..width).map(|_| vec![1_i32]).collect(),
    )?;
    let ones_row = push_block(&mut blocks, ones_row);
    io_blocks.push(ones_row);
    let ones_column =
        MatrixBlock::from_columns(MatrixDims::new(width, 1)?, vec![vec![1_i32; width]])?;
    let ones_column = push_block(&mut blocks, ones_column);
    io_blocks.push(ones_column);
    let negative_ones_column =
        MatrixBlock::from_columns(MatrixDims::new(width, 1)?, vec![vec![-1_i32; width]])?;
    let negative_ones_column = push_block(&mut blocks, negative_ones_column);
    io_blocks.push(negative_ones_column);

    let mut push_private_vector = |values: Vec<i32>| -> Result<GlobalBlockId, NormalizationError> {
        Ok(push_vector_block(&mut blocks, values)?)
    };
    let softmax_base_output_id = push_private_vector(softmax_base_output.to_vec())?;
    let layer_sum_value = inputs.layer_norm_input.iter().sum::<i32>();
    let layer_sum = push_private_vector(vec![layer_sum_value])?;
    multiplications.push(GlobalOperation {
        left: ones_row,
        right: Some(x),
        output: layer_sum,
    });
    let width_scaled_values = inputs
        .layer_norm_input
        .iter()
        .map(|value| value.checked_mul(i32::try_from(width).unwrap()).unwrap())
        .collect::<Vec<_>>();
    let width_scaled = push_private_vector(width_scaled_values)?;
    push_hadamard(
        width_values,
        x,
        width_scaled,
        zero,
        &mut hadamards,
        &mut activation_aux,
    );
    let negative_sum_values = vec![
        layer_sum_value
            .checked_neg()
            .ok_or(NormalizationError::ArithmeticOverflow)?;
        width
    ];
    let negative_sum = push_private_vector(negative_sum_values)?;
    multiplications.push(GlobalOperation {
        left: negative_ones_column,
        right: Some(layer_sum),
        output: negative_sum,
    });
    let centered_id = push_private_vector(centered.to_vec())?;
    additions.push(GlobalOperation {
        left: width_scaled,
        right: Some(negative_sum),
        output: centered_id,
    });
    let squares = push_private_vector(layer_squares.to_vec())?;
    push_hadamard(
        centered_id,
        centered_id,
        squares,
        zero,
        &mut hadamards,
        &mut activation_aux,
    );
    let variance_without_epsilon = layer_squares.iter().sum::<i32>();
    let variance_sum = push_private_vector(vec![variance_without_epsilon])?;
    multiplications.push(GlobalOperation {
        left: ones_row,
        right: Some(squares),
        output: variance_sum,
    });
    let variance_id = push_private_vector(vec![variance])?;
    additions.push(GlobalOperation {
        left: variance_sum,
        right: Some(epsilon),
        output: variance_id,
    });
    let inverse_sqrt_id = push_private_vector(vec![inverse_sqrt])?;
    lookups.push(TaggedLookupOperation {
        tag: LookupTag::InverseSqrt,
        input: variance_id,
        output: inverse_sqrt_id,
    });
    let gamma_centered_id = push_private_vector(gamma_centered.to_vec())?;
    push_hadamard(
        gamma,
        centered_id,
        gamma_centered_id,
        zero,
        &mut hadamards,
        &mut activation_aux,
    );
    let inverse_sqrt_vector = push_private_vector(vec![inverse_sqrt; width])?;
    multiplications.push(GlobalOperation {
        left: ones_column,
        right: Some(inverse_sqrt_id),
        output: inverse_sqrt_vector,
    });
    let layer_scaled_id = push_private_vector(layer_scaled.to_vec())?;
    push_hadamard(
        gamma_centered_id,
        inverse_sqrt_vector,
        layer_scaled_id,
        zero,
        &mut hadamards,
        &mut activation_aux,
    );
    let layer_numerator = push_private_vector(layer_numerators.to_vec())?;
    additions.push(GlobalOperation {
        left: layer_scaled_id,
        right: Some(beta),
        output: layer_numerator,
    });
    let layer_scaled_output_values = layer_norm_output
        .iter()
        .map(|value| value.checked_mul(1_i32 << manifest.output_shift).unwrap())
        .collect::<Vec<_>>();
    let layer_scaled_output = push_private_vector(layer_scaled_output_values)?;
    push_hadamard(
        layer_divisor,
        layer_output,
        layer_scaled_output,
        zero,
        &mut hadamards,
        &mut activation_aux,
    );
    let layer_remainder = push_private_vector(layer_remainders.to_vec())?;
    additions.push(GlobalOperation {
        left: layer_scaled_output,
        right: Some(layer_remainder),
        output: layer_numerator,
    });
    lookups.push(TaggedLookupOperation {
        tag: LookupTag::NonNegative,
        input: layer_remainder,
        output: layer_remainder,
    });

    let maximum_id = push_private_vector(vec![maximum])?;
    let maximum_vector = push_private_vector(vec![maximum; width])?;
    multiplications.push(GlobalOperation {
        left: ones_column,
        right: Some(maximum_id),
        output: maximum_vector,
    });
    let negative_maximum_vector = push_private_vector(vec![
        maximum.checked_neg().ok_or(
            NormalizationError::ArithmeticOverflow
        )?;
        width
    ])?;
    multiplications.push(GlobalOperation {
        left: negative_ones_column,
        right: Some(maximum_id),
        output: negative_maximum_vector,
    });
    let shifted_id = push_private_vector(shifted.to_vec())?;
    additions.push(GlobalOperation {
        left: scores,
        right: Some(negative_maximum_vector),
        output: shifted_id,
    });
    let negative_scores_values = inputs
        .softmax_scores
        .iter()
        .map(|value| {
            value
                .checked_neg()
                .ok_or(NormalizationError::ArithmeticOverflow)
        })
        .collect::<Result<Vec<_>, _>>()?;
    let negative_scores = push_private_vector(negative_scores_values)?;
    push_hadamard(
        negative_ones,
        scores,
        negative_scores,
        zero,
        &mut hadamards,
        &mut activation_aux,
    );
    let differences = push_private_vector(maximum_differences.to_vec())?;
    additions.push(GlobalOperation {
        left: maximum_vector,
        right: Some(negative_scores),
        output: differences,
    });
    lookups.push(TaggedLookupOperation {
        tag: LookupTag::NonNegative,
        input: differences,
        output: differences,
    });
    let one_hot_id = push_private_vector(one_hot.to_vec())?;
    let one_hot_sum = push_private_vector(vec![1])?;
    multiplications.push(GlobalOperation {
        left: ones_row,
        right: Some(one_hot_id),
        output: one_hot_sum,
    });
    // Reuse the public scalar-one block as the required output of the reduction.
    additions.push(GlobalOperation {
        left: one_hot_sum,
        right: Some(push_private_vector(vec![0])?),
        output: one,
    });
    let one_hot_minus_one_values = one_hot.iter().map(|value| value - 1).collect::<Vec<_>>();
    let one_hot_minus_one = push_private_vector(one_hot_minus_one_values)?;
    additions.push(GlobalOperation {
        left: one_hot_id,
        right: Some(negative_ones),
        output: one_hot_minus_one,
    });
    push_hadamard(
        one_hot_id,
        one_hot_minus_one,
        zero,
        zero,
        &mut hadamards,
        &mut activation_aux,
    );
    push_hadamard(
        one_hot_id,
        differences,
        zero,
        zero,
        &mut hadamards,
        &mut activation_aux,
    );

    let exponential_id = push_private_vector(exponentials.to_vec())?;
    lookups.push(TaggedLookupOperation {
        tag: LookupTag::Exp,
        input: shifted_id,
        output: exponential_id,
    });
    let denominator_id = push_private_vector(vec![denominator])?;
    multiplications.push(GlobalOperation {
        left: ones_row,
        right: Some(exponential_id),
        output: denominator_id,
    });
    let reciprocal_id = push_private_vector(vec![reciprocal])?;
    lookups.push(TaggedLookupOperation {
        tag: LookupTag::Reciprocal,
        input: denominator_id,
        output: reciprocal_id,
    });
    let reciprocal_vector = push_private_vector(vec![reciprocal; width])?;
    multiplications.push(GlobalOperation {
        left: ones_column,
        right: Some(reciprocal_id),
        output: reciprocal_vector,
    });
    let softmax_product = push_private_vector(softmax_products.to_vec())?;
    push_hadamard(
        exponential_id,
        reciprocal_vector,
        softmax_product,
        zero,
        &mut hadamards,
        &mut activation_aux,
    );
    let softmax_scaled_output_values = softmax_base_output
        .iter()
        .map(|value| {
            value
                .checked_mul(1_i32 << manifest.softmax_output_shift)
                .unwrap()
        })
        .collect::<Vec<_>>();
    let softmax_scaled_output = push_private_vector(softmax_scaled_output_values)?;
    push_hadamard(
        softmax_divisor,
        softmax_base_output_id,
        softmax_scaled_output,
        zero,
        &mut hadamards,
        &mut activation_aux,
    );
    let softmax_remainder = push_private_vector(softmax_remainders.to_vec())?;
    additions.push(GlobalOperation {
        left: softmax_scaled_output,
        right: Some(softmax_remainder),
        output: softmax_product,
    });
    lookups.push(TaggedLookupOperation {
        tag: LookupTag::NonNegative,
        input: softmax_remainder,
        output: softmax_remainder,
    });

    // Exact Qq normalization. The lookup-derived/requantized base row may
    // lose mass to integer rounding. Bind the residual to S-sum(base), route
    // it through the already constrained argmax one-hot selector, and prove
    // that the final public row sums to the fixed-point unit S=2^q.
    let softmax_base_sum_value = softmax_base_output.iter().sum::<i32>();
    let softmax_base_sum = push_private_vector(vec![softmax_base_sum_value])?;
    multiplications.push(GlobalOperation {
        left: ones_row,
        right: Some(softmax_base_output_id),
        output: softmax_base_sum,
    });
    let softmax_residual_id = push_private_vector(vec![softmax_residual])?;
    additions.push(GlobalOperation {
        left: softmax_base_sum,
        right: Some(softmax_residual_id),
        output: softmax_unit,
    });
    lookups.push(TaggedLookupOperation {
        tag: LookupTag::NonNegative,
        input: softmax_residual_id,
        output: softmax_residual_id,
    });
    let softmax_residual_vector = push_private_vector(vec![softmax_residual; width])?;
    multiplications.push(GlobalOperation {
        left: ones_column,
        right: Some(softmax_residual_id),
        output: softmax_residual_vector,
    });
    let softmax_correction_values = one_hot
        .iter()
        .map(|selected| selected.checked_mul(softmax_residual).unwrap())
        .collect::<Vec<_>>();
    let softmax_correction = push_private_vector(softmax_correction_values)?;
    push_hadamard(
        one_hot_id,
        softmax_residual_vector,
        softmax_correction,
        zero,
        &mut hadamards,
        &mut activation_aux,
    );
    additions.push(GlobalOperation {
        left: softmax_base_output_id,
        right: Some(softmax_correction),
        output: softmax_output_id,
    });
    multiplications.push(GlobalOperation {
        left: ones_row,
        right: Some(softmax_output_id),
        output: softmax_unit,
    });

    let operations = BTreeMap::from([
        (GlobalOperationType::Multiplication, multiplications),
        (GlobalOperationType::Addition, additions),
        (GlobalOperationType::Activation, hadamards),
    ]);
    let base_manifest = MlpManifest {
        widths: vec![padded_width, padded_width],
        commitment_bit_width: manifest.commitment_bit_width,
        relu_magnitude_bits: 1,
    };
    let base = GlobalMlpEncoding {
        manifest: base_manifest,
        public_input: inputs.layer_norm_input.clone(),
        public_output: [layer_norm_output, softmax_output].concat(),
        blocks,
        io_blocks,
        model_blocks: Vec::new(),
        operations,
        activation_aux,
        _marker: PhantomData,
    };
    Ok((base, lookups))
}

fn push_hadamard(
    left: GlobalBlockId,
    right: GlobalBlockId,
    output: GlobalBlockId,
    dummy_bit: GlobalBlockId,
    operations: &mut Vec<GlobalOperation>,
    auxiliaries: &mut Vec<ActivationAux>,
) {
    operations.push(GlobalOperation {
        left,
        right: None,
        output,
    });
    auxiliaries.push(ActivationAux {
        negative: right,
        output_bits: vec![dummy_bit],
        negative_bits: vec![dummy_bit],
    });
}

fn validate_manifest_and_inputs(
    manifest: &NormalizationManifest,
    inputs: &NormalizationInputs,
) -> Result<BTreeMap<LookupTag, BTreeMap<i32, i32>>, NormalizationError> {
    if manifest.width == 0 {
        return Err(NormalizationError::EmptyInput);
    }
    if manifest.commitment_bit_width == 0 || manifest.commitment_bit_width > 32 {
        return Err(NormalizationError::InvalidCommitmentBitWidth);
    }
    if manifest.output_shift > 30 || manifest.softmax_output_shift > 30 {
        return Err(NormalizationError::InvalidShift);
    }
    for values in [
        &inputs.layer_norm_input,
        &inputs.layer_norm_gamma,
        &inputs.layer_norm_beta,
        &inputs.softmax_scores,
    ] {
        if values.len() != manifest.width {
            return Err(NormalizationError::WidthMismatch);
        }
    }

    let mut tables = BTreeMap::new();
    for table in &manifest.tables {
        if tables.contains_key(&table.tag) {
            return Err(NormalizationError::DuplicateLookupTag(table.tag));
        }
        let mut rows = BTreeMap::new();
        for (input, output) in &table.rows {
            if rows.insert(*input, *output).is_some() {
                return Err(NormalizationError::DuplicateLookupInput {
                    tag: table.tag,
                    input: *input,
                });
            }
        }
        tables.insert(table.tag, rows);
    }
    let required = [
        LookupTag::Exp,
        LookupTag::Reciprocal,
        LookupTag::InverseSqrt,
    ];
    let present = tables.keys().copied().collect::<BTreeSet<_>>();
    for tag in required {
        if !present.contains(&tag) {
            return Err(NormalizationError::MissingLookupTable(tag));
        }
    }
    Ok(tables)
}

fn lookup(
    tables: &BTreeMap<LookupTag, BTreeMap<i32, i32>>,
    tag: LookupTag,
    input: i32,
    queries: &mut Vec<TaggedLookupQuery>,
) -> Result<i32, NormalizationError> {
    let output = *tables
        .get(&tag)
        .and_then(|table| table.get(&input))
        .ok_or(NormalizationError::LookupMiss { tag, input })?;
    queries.push(TaggedLookupQuery { tag, input, output });
    Ok(output)
}

fn requantize(value: i64, shift: usize) -> Result<(i32, i32), NormalizationError> {
    let divisor = 1_i64
        .checked_shl(u32::try_from(shift).map_err(|_| NormalizationError::InvalidShift)?)
        .ok_or(NormalizationError::InvalidShift)?;
    let quotient = value.div_euclid(divisor);
    let remainder = value.rem_euclid(divisor);
    Ok((
        i32::try_from(quotient).map_err(|_| NormalizationError::ArithmeticOverflow)?,
        i32::try_from(remainder).map_err(|_| NormalizationError::ArithmeticOverflow)?,
    ))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum NormalizationError {
    Global(GlobalMlpError),
    Matrix(HeteroMatrixError),
    EmptyInput,
    WidthMismatch,
    InvalidCommitmentBitWidth,
    InvalidShift,
    DuplicateLookupTag(LookupTag),
    DuplicateLookupInput { tag: LookupTag, input: i32 },
    MissingLookupTable(LookupTag),
    LookupMiss { tag: LookupTag, input: i32 },
    ArithmeticOverflow,
    NegativeSoftmaxResidual(i32),
    NonIdentityRangeTable(i32),
    InvalidLookupBlock,
}

impl From<GlobalMlpError> for NormalizationError {
    fn from(error: GlobalMlpError) -> Self {
        Self::Global(error)
    }
}

impl From<HeteroMatrixError> for NormalizationError {
    fn from(error: HeteroMatrixError) -> Self {
        Self::Matrix(error)
    }
}

impl Display for NormalizationError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for NormalizationError {}

#[cfg(test)]
mod tests {
    use ark_bls12_381::Bls12_381;

    use super::{
        GlobalHdR1csNormalizationProtocol, LookupTag, NormalizationInputs, NormalizationManifest,
        OperationFamily, TaggedLookupTable,
    };

    fn worked_manifest() -> NormalizationManifest {
        NormalizationManifest {
            width: 4,
            epsilon: 1,
            output_shift: 3,
            softmax_output_shift: 4,
            commitment_bit_width: 16,
            tables: vec![
                TaggedLookupTable::new(LookupTag::Exp, vec![(-3, 1), (-2, 2), (-1, 4), (0, 8)]),
                TaggedLookupTable::new(LookupTag::Reciprocal, vec![(15, 16)]),
                TaggedLookupTable::new(LookupTag::InverseSqrt, vec![(81, 8)]),
                TaggedLookupTable::new(
                    LookupTag::NonNegative,
                    (0..=15).map(|value| (value, value)).collect(),
                ),
                TaggedLookupTable::new(LookupTag::Boolean, vec![(0, 0), (1, 1)]),
            ],
        }
    }

    fn worked_inputs() -> NormalizationInputs {
        NormalizationInputs {
            layer_norm_input: vec![1, 2, 3, 4],
            layer_norm_gamma: vec![1, 1, 1, 1],
            layer_norm_beta: vec![0, 0, 0, 0],
            softmax_scores: vec![0, 1, 2, 3],
        }
    }

    #[test]
    fn compiler_merges_layer_norm_and_softmax_into_three_operation_families() {
        let encoding = GlobalHdR1csNormalizationProtocol::<Bls12_381>::encode(
            worked_manifest(),
            worked_inputs(),
        )
        .unwrap();

        assert_eq!(encoding.layer_norm_output(), &[-6, -2, 2, 6]);
        assert_eq!(encoding.softmax_output(), &[1, 2, 4, 9]);
        assert_eq!(encoding.softmax_output().iter().sum::<i32>(), 1 << 4);
        assert_eq!(encoding.operation_family_count(), 3);
        assert_eq!(encoding.top_level_equation_count(), 4);
        assert!(encoding.operation_count(OperationFamily::Linear) > 0);
        assert!(encoding.operation_count(OperationFamily::Bilinear) > 0);
        assert_eq!(encoding.operation_count(OperationFamily::TaggedLookup), 19);
        assert_eq!(encoding.lookup_count(LookupTag::Exp), 4);
        assert_eq!(encoding.lookup_count(LookupTag::Reciprocal), 1);
        assert_eq!(encoding.lookup_count(LookupTag::InverseSqrt), 1);
    }

    #[test]
    fn normalization_proof_uses_one_global_witness_opening() {
        let encoding = GlobalHdR1csNormalizationProtocol::<Bls12_381>::encode(
            worked_manifest(),
            worked_inputs(),
        )
        .unwrap();
        let srs = GlobalHdR1csNormalizationProtocol::<Bls12_381>::setup(&encoding).unwrap();
        let (instance, proof) =
            GlobalHdR1csNormalizationProtocol::<Bls12_381>::prove(&srs, encoding).unwrap();

        assert_eq!(proof.witness_commitment_count(), 1);
        assert_eq!(proof.witness_opening_count(), 1);
        assert_eq!(proof.top_level_equation_count(), 4);
        assert!(GlobalHdR1csNormalizationProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
    }

    #[test]
    fn tampered_public_input_is_rejected() {
        let encoding = GlobalHdR1csNormalizationProtocol::<Bls12_381>::encode(
            worked_manifest(),
            worked_inputs(),
        )
        .unwrap();
        let srs = GlobalHdR1csNormalizationProtocol::<Bls12_381>::setup(&encoding).unwrap();
        let (mut instance, proof) =
            GlobalHdR1csNormalizationProtocol::<Bls12_381>::prove(&srs, encoding).unwrap();

        // This changes a public beta and its remainder witness while preserving
        // the rounded output, so output-only binding would miss the tamper.
        instance.inputs.layer_norm_beta[0] = 1;
        assert!(!GlobalHdR1csNormalizationProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
    }

    #[test]
    fn tampered_lookup_tag_and_rounding_witness_are_rejected() {
        let encoding = GlobalHdR1csNormalizationProtocol::<Bls12_381>::encode(
            worked_manifest(),
            worked_inputs(),
        )
        .unwrap();
        let srs = GlobalHdR1csNormalizationProtocol::<Bls12_381>::setup(&encoding).unwrap();
        let (instance, mut proof) =
            GlobalHdR1csNormalizationProtocol::<Bls12_381>::prove(&srs, encoding).unwrap();

        let original_tag = proof.lookup.blocks[0].tag;
        proof.lookup.blocks[0].tag = LookupTag::Exp;
        assert!(!GlobalHdR1csNormalizationProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
        proof.lookup.blocks[0].tag = original_tag;

        // (1, 1) is itself a valid NonNegative table row. Replacing the true
        // zero remainder therefore exercises witness routing, not membership.
        proof.lookup.blocks[1].input[0] = 1;
        proof.lookup.blocks[1].output[0] = 1;
        assert!(!GlobalHdR1csNormalizationProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
    }
}
