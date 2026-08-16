use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::{One, UniformRand, Zero};
use ark_poly_commit::smart_pc::{data_structures::Trans as PcsTrans, SmartPC};
use ark_serialize::{CanonicalSerialize, Compress};
use ark_std::ops::Mul;
use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;

use crate::data_structures::SRS;
use crate::hdr1cs::mlp::{MlpLayer, MlpManifest, MlpProofError};
use crate::protocols::{
    batchpcs::BatchProjectionPoints,
    diagonal::ZkDiagonal,
    kronecker::ZkKronecker,
    mul::ZkMatMul,
    virtualpcs::{commit_stacked_i32, open_stacked_i32, VirtualPcsError},
};
use crate::utils::{
    fiat_shamir::FiatShamir,
    heterogeneous::{
        ColumnZeroHeteroView, DiagonalHeteroView, ElementwiseDiagonalHeteroView, HeteroBlockMat,
        HeteroMatrixError, MatrixBlock, MatrixDims, PaddedDomain, StackedHeteroView,
    },
    linear,
    sparse::{DenseMatCM, MatOps, SparseKronecker},
    xi,
    zktr::{TranElem, TranSeq, ZkTranSeq},
};

pub mod artifact;
pub mod normalization;

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum GlobalOperationType {
    Multiplication,
    Addition,
    Activation,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GlobalBlockRole {
    Left,
    Right,
    Output,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlobalBlockId(pub usize);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct GlobalOperation {
    left: GlobalBlockId,
    right: Option<GlobalBlockId>,
    output: GlobalBlockId,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ActivationAux {
    negative: GlobalBlockId,
    output_bits: Vec<GlobalBlockId>,
    negative_bits: Vec<GlobalBlockId>,
}

pub struct GlobalMlpEncoding<E: Pairing> {
    pub manifest: MlpManifest,
    pub public_input: Vec<i32>,
    pub public_output: Vec<i32>,
    blocks: Vec<MatrixBlock<i32>>,
    io_blocks: Vec<GlobalBlockId>,
    model_blocks: Vec<GlobalBlockId>,
    operations: BTreeMap<GlobalOperationType, Vec<GlobalOperation>>,
    activation_aux: Vec<ActivationAux>,
    _marker: PhantomData<E>,
}

impl<E: Pairing> GlobalMlpEncoding<E> {
    pub fn operation_count(&self, operation_type: GlobalOperationType) -> usize {
        self.operations
            .get(&operation_type)
            .map(Vec::len)
            .unwrap_or(0)
    }

    pub fn routed_block(
        &self,
        operation_type: GlobalOperationType,
        operation_index: usize,
        role: GlobalBlockRole,
    ) -> Option<GlobalBlockId> {
        let operation = self.operations.get(&operation_type)?.get(operation_index)?;
        match role {
            GlobalBlockRole::Left => Some(operation.left),
            GlobalBlockRole::Right => operation.right,
            GlobalBlockRole::Output => Some(operation.output),
        }
    }

    pub fn witness_block_count(&self) -> usize {
        self.blocks.len()
    }

    pub fn io_block_count(&self) -> usize {
        self.io_blocks.len()
    }

    pub fn stored_witness_entries(&self) -> usize {
        self.blocks.iter().map(|block| block.dims.entries()).sum()
    }

    pub fn global_shape(&self) -> GlobalMlpShape {
        self.shape()
    }

    fn shape(&self) -> GlobalMlpShape {
        let n = *self.manifest.widths.iter().max().unwrap();
        let witness_slots = self.blocks.len().next_power_of_two();
        let gate_count = self.operations.values().map(Vec::len).sum::<usize>();
        GlobalMlpShape {
            n,
            witness_slots,
            gate_slots: gate_count.next_power_of_two(),
            relu_magnitude_bits: self.manifest.relu_magnitude_bits,
            commitment_bit_width: self.manifest.commitment_bit_width,
        }
    }

    fn artifacts(&self) -> Result<GlobalMlpArtifacts<E::ScalarField>, GlobalMlpError>
    where
        E::ScalarField: From<i32>,
    {
        let shape = self.shape();
        let witness = HeteroBlockMat::new(
            PaddedDomain::new(shape.witness_slots, shape.n, shape.n)?,
            self.blocks.iter().cloned().map(Some).collect(),
        )?;
        let io_set = self
            .io_blocks
            .iter()
            .map(|id| id.0)
            .collect::<std::collections::BTreeSet<_>>();
        let io = HeteroBlockMat::new(
            PaddedDomain::new(shape.witness_slots, shape.n, shape.n)?,
            self.blocks
                .iter()
                .enumerate()
                .map(|(index, block)| io_set.contains(&index).then(|| block.clone()))
                .collect(),
        )?;
        let model_set = self
            .model_blocks
            .iter()
            .map(|id| id.0)
            .collect::<std::collections::BTreeSet<_>>();
        let model = HeteroBlockMat::new(
            PaddedDomain::new(shape.witness_slots, shape.n, shape.n)?,
            self.blocks
                .iter()
                .enumerate()
                .map(|(index, block)| model_set.contains(&index).then(|| block.clone()))
                .collect(),
        )?;

        let mut a_l = zero_columns(shape.gate_slots, shape.witness_slots);
        let mut a_r = zero_columns(shape.gate_slots, shape.witness_slots);
        let mut a_o = zero_columns(shape.gate_slots, shape.witness_slots);
        let mut selectors = BTreeMap::new();
        let mut row_start = 0;
        for operation_type in [
            GlobalOperationType::Multiplication,
            GlobalOperationType::Addition,
            GlobalOperationType::Activation,
        ] {
            let operations = &self.operations[&operation_type];
            let mut selector = zero_columns(shape.gate_slots, shape.gate_slots);
            for (local_row, operation) in operations.iter().enumerate() {
                let row = row_start + local_row;
                a_l[operation.left.0][row] = 1;
                if let Some(right) = operation.right {
                    a_r[right.0][row] = 1;
                }
                a_o[operation.output.0][row] = 1;
                selector[row][row] = 1;
            }
            selectors.insert(operation_type, matrix_artifact(selector)?);
            row_start += operations.len();
        }

        let mut io_selector = zero_columns(shape.witness_slots, shape.witness_slots);
        for block in &self.io_blocks {
            io_selector[block.0][block.0] = 1;
        }
        let mut model_selector = zero_columns(shape.witness_slots, shape.witness_slots);
        for block in &self.model_blocks {
            model_selector[block.0][block.0] = 1;
        }

        let activation_start = self.operation_count(GlobalOperationType::Multiplication)
            + self.operation_count(GlobalOperationType::Addition);
        let mut negative = zero_columns(shape.gate_slots, shape.witness_slots);
        let mut output_bits = (0..shape.relu_magnitude_bits)
            .map(|_| zero_columns(shape.gate_slots, shape.witness_slots))
            .collect::<Vec<_>>();
        let mut negative_bits = (0..shape.relu_magnitude_bits)
            .map(|_| zero_columns(shape.gate_slots, shape.witness_slots))
            .collect::<Vec<_>>();
        for (index, auxiliary) in self.activation_aux.iter().enumerate() {
            let row = activation_start + index;
            negative[auxiliary.negative.0][row] = 1;
            for (bit, block) in auxiliary.output_bits.iter().enumerate() {
                output_bits[bit][block.0][row] = 1;
            }
            for (bit, block) in auxiliary.negative_bits.iter().enumerate() {
                negative_bits[bit][block.0][row] = 1;
            }
        }

        Ok(GlobalMlpArtifacts {
            shape,
            witness,
            io,
            model,
            a_l: matrix_artifact(a_l)?,
            a_r: matrix_artifact(a_r)?,
            a_o: matrix_artifact(a_o)?,
            selectors,
            io_selector: matrix_artifact(io_selector)?,
            model_selector: matrix_artifact(model_selector)?,
            negative: matrix_artifact(negative)?,
            output_bits: output_bits
                .into_iter()
                .map(matrix_artifact)
                .collect::<Result<_, _>>()?,
            negative_bits: negative_bits
                .into_iter()
                .map(matrix_artifact)
                .collect::<Result<_, _>>()?,
        })
    }

    fn routed_matrix(
        &self,
        operation_type: GlobalOperationType,
        role: GlobalBlockRole,
    ) -> Result<HeteroBlockMat<i32>, GlobalMlpError> {
        let shape = self.shape();
        let start = match operation_type {
            GlobalOperationType::Multiplication => 0,
            GlobalOperationType::Addition => {
                self.operation_count(GlobalOperationType::Multiplication)
            }
            GlobalOperationType::Activation => {
                self.operation_count(GlobalOperationType::Multiplication)
                    + self.operation_count(GlobalOperationType::Addition)
            }
        };
        let mut blocks = vec![None; shape.gate_slots];
        for (index, operation) in self.operations[&operation_type].iter().enumerate() {
            let id = match role {
                GlobalBlockRole::Left => Some(operation.left),
                GlobalBlockRole::Right => operation.right,
                GlobalBlockRole::Output => Some(operation.output),
            };
            if let Some(id) = id {
                blocks[start + index] = Some(self.blocks[id.0].clone());
            }
        }
        Ok(HeteroBlockMat::new(
            PaddedDomain::new(shape.gate_slots, shape.n, shape.n)?,
            blocks,
        )?)
    }

    fn routed_activation_aux(
        &self,
        which: ActivationRoute,
    ) -> Result<HeteroBlockMat<i32>, GlobalMlpError> {
        let shape = self.shape();
        let start = self.operation_count(GlobalOperationType::Multiplication)
            + self.operation_count(GlobalOperationType::Addition);
        let mut blocks = vec![None; shape.gate_slots];
        for (index, auxiliary) in self.activation_aux.iter().enumerate() {
            let id = match which {
                ActivationRoute::Negative => auxiliary.negative,
                ActivationRoute::OutputBit(bit) => auxiliary.output_bits[bit],
                ActivationRoute::NegativeBit(bit) => auxiliary.negative_bits[bit],
            };
            blocks[start + index] = Some(self.blocks[id.0].clone());
        }
        Ok(HeteroBlockMat::new(
            PaddedDomain::new(shape.gate_slots, shape.n, shape.n)?,
            blocks,
        )?)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GlobalMlpShape {
    pub n: usize,
    pub witness_slots: usize,
    pub gate_slots: usize,
    pub relu_magnitude_bits: usize,
    pub commitment_bit_width: usize,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ActivationRoute {
    Negative,
    OutputBit(usize),
    NegativeBit(usize),
}

struct MatrixArtifact<F>
where
    F: ark_ff::PrimeField + From<i32>,
{
    virtual_matrix: HeteroBlockMat<i32>,
    dense: DenseMatCM<i32, F>,
}

struct GlobalMlpArtifacts<F>
where
    F: ark_ff::PrimeField + From<i32>,
{
    shape: GlobalMlpShape,
    witness: HeteroBlockMat<i32>,
    io: HeteroBlockMat<i32>,
    model: HeteroBlockMat<i32>,
    a_l: MatrixArtifact<F>,
    a_r: MatrixArtifact<F>,
    a_o: MatrixArtifact<F>,
    selectors: BTreeMap<GlobalOperationType, MatrixArtifact<F>>,
    io_selector: MatrixArtifact<F>,
    model_selector: MatrixArtifact<F>,
    negative: MatrixArtifact<F>,
    output_bits: Vec<MatrixArtifact<F>>,
    negative_bits: Vec<MatrixArtifact<F>>,
}

fn zero_columns(rows: usize, cols: usize) -> Vec<Vec<i32>> {
    vec![vec![0; rows]; cols]
}

fn matrix_artifact<F>(columns: Vec<Vec<i32>>) -> Result<MatrixArtifact<F>, GlobalMlpError>
where
    F: ark_ff::PrimeField + From<i32>,
{
    let cols = columns.len();
    let rows = columns.first().map(Vec::len).unwrap_or(0);
    let block = MatrixBlock::from_columns(MatrixDims::new(rows, cols)?, columns.clone())?;
    Ok(MatrixArtifact {
        virtual_matrix: HeteroBlockMat::new(PaddedDomain::new(1, rows, cols)?, vec![Some(block)])?,
        dense: DenseMatCM::from_data(columns),
    })
}

fn public_io_matrix(
    shape: GlobalMlpShape,
    public_input: &[i32],
    public_output: &[i32],
) -> Result<HeteroBlockMat<i32>, GlobalMlpError> {
    if shape.witness_slots < 2
        || public_input.is_empty()
        || public_input.len() > shape.n
        || public_output.is_empty()
        || public_output.len() > shape.n
    {
        return Err(GlobalMlpError::ProtocolShapeMismatch);
    }
    let mut blocks = vec![None; shape.witness_slots];
    blocks[0] = Some(MatrixBlock::from_columns(
        MatrixDims::new(public_input.len(), 1)?,
        vec![public_input.to_vec()],
    )?);
    blocks[1] = Some(MatrixBlock::from_columns(
        MatrixDims::new(public_output.len(), 1)?,
        vec![public_output.to_vec()],
    )?);
    Ok(HeteroBlockMat::new(
        PaddedDomain::new(shape.witness_slots, shape.n, shape.n)?,
        blocks,
    )?)
}

fn public_io_selector<F>(shape: GlobalMlpShape) -> Result<MatrixArtifact<F>, GlobalMlpError>
where
    F: ark_ff::PrimeField + From<i32>,
{
    if shape.witness_slots < 2 {
        return Err(GlobalMlpError::ProtocolShapeMismatch);
    }
    let mut columns = zero_columns(shape.witness_slots, shape.witness_slots);
    columns[0][0] = 1;
    columns[1][1] = 1;
    matrix_artifact(columns)
}

fn expected_mlp_shape(manifest: &MlpManifest) -> Result<GlobalMlpShape, GlobalMlpError> {
    manifest.validate()?;
    let layers = manifest.layers();
    let blocks_per_layer = manifest
        .relu_magnitude_bits
        .checked_mul(2)
        .and_then(|bits| bits.checked_add(6))
        .ok_or(GlobalMlpError::ProtocolShapeMismatch)?;
    let witness_blocks = layers
        .checked_mul(blocks_per_layer)
        .and_then(|blocks| blocks.checked_add(1))
        .ok_or(GlobalMlpError::ProtocolShapeMismatch)?;
    let gate_count = layers
        .checked_mul(3)
        .ok_or(GlobalMlpError::ProtocolShapeMismatch)?;
    let witness_slots = witness_blocks
        .checked_next_power_of_two()
        .ok_or(GlobalMlpError::ProtocolShapeMismatch)?;
    let gate_slots = gate_count
        .checked_next_power_of_two()
        .ok_or(GlobalMlpError::ProtocolShapeMismatch)?;
    Ok(GlobalMlpShape {
        n: *manifest
            .widths
            .iter()
            .max()
            .ok_or(GlobalMlpError::ProtocolShapeMismatch)?,
        witness_slots,
        gate_slots,
        relu_magnitude_bits: manifest.relu_magnitude_bits,
        commitment_bit_width: manifest.commitment_bit_width,
    })
}

fn public_model_selector<F>(
    manifest: &MlpManifest,
    shape: GlobalMlpShape,
) -> Result<MatrixArtifact<F>, GlobalMlpError>
where
    F: ark_ff::PrimeField + From<i32>,
{
    if expected_mlp_shape(manifest)? != shape {
        return Err(GlobalMlpError::ProtocolShapeMismatch);
    }
    let mut columns = zero_columns(shape.witness_slots, shape.witness_slots);
    let mut cursor = 2_usize;
    for layer in 0..manifest.layers() {
        cursor = cursor
            .checked_add(2)
            .ok_or(GlobalMlpError::ProtocolShapeMismatch)?;
        if layer + 1 != manifest.layers() {
            cursor = cursor
                .checked_add(1)
                .ok_or(GlobalMlpError::ProtocolShapeMismatch)?;
        }
        cursor = cursor
            .checked_add(1 + 2 * manifest.relu_magnitude_bits)
            .ok_or(GlobalMlpError::ProtocolShapeMismatch)?;
        for _ in 0..2 {
            if cursor >= shape.witness_slots {
                return Err(GlobalMlpError::ProtocolShapeMismatch);
            }
            columns[cursor][cursor] = 1;
            cursor += 1;
        }
    }
    let expected_blocks = manifest
        .layers()
        .checked_mul(6 + 2 * manifest.relu_magnitude_bits)
        .and_then(|blocks| blocks.checked_add(1))
        .ok_or(GlobalMlpError::ProtocolShapeMismatch)?;
    if cursor != expected_blocks {
        return Err(GlobalMlpError::ProtocolShapeMismatch);
    }
    matrix_artifact(columns)
}

pub struct GlobalHdR1csMlpProtocol<E: Pairing> {
    _marker: PhantomData<E>,
}

pub struct GlobalMlpInstance<E: Pairing> {
    pub manifest: MlpManifest,
    pub shape: GlobalMlpShape,
    pub public_input: Vec<i32>,
    pub public_output: Vec<i32>,
    pub witness_commitment: PairingOutput<E>,
    pub io_commitment: PairingOutput<E>,
    pub model_commitment: PairingOutput<E>,
    pub architecture: GlobalArchitectureCommitments<E>,
}

pub struct GlobalArchitectureCommitments<E: Pairing> {
    pub a_l: PairingOutput<E>,
    pub a_r: PairingOutput<E>,
    pub a_o: PairingOutput<E>,
    pub selectors: BTreeMap<GlobalOperationType, PairingOutput<E>>,
    pub io_selector: PairingOutput<E>,
    pub model_selector: PairingOutput<E>,
    pub negative: PairingOutput<E>,
    pub output_bits: Vec<PairingOutput<E>>,
    pub negative_bits: Vec<PairingOutput<E>>,
}

struct CommittedMatrix<E: Pairing> {
    matrix: HeteroBlockMat<i32>,
    hiding: E::ScalarField,
    commitment: PairingOutput<E>,
    tier_one: Vec<E::G1>,
}

impl<E: Pairing> CommittedMatrix<E>
where
    E::ScalarField: From<i32>,
{
    fn commit(
        srs: &SRS<E>,
        matrix: HeteroBlockMat<i32>,
        hiding: E::ScalarField,
        bit_width: usize,
    ) -> Result<Self, GlobalMlpError> {
        let (commitment, tier_one) =
            commit_stacked_i32::<E>(&srs.pc_pp, &matrix, hiding, bit_width)?;
        Ok(Self {
            matrix,
            hiding,
            commitment,
            tier_one,
        })
    }
}

pub struct RoutedOpeningProof<E: Pairing> {
    transcript: TranSeq<E::ScalarField, E::G1>,
    selector_opening: PcsTrans<E>,
    wiring_opening: Option<PcsTrans<E>>,
}

struct ProverWitnessClaim<E: Pairing> {
    value_commitment: E::G1,
    value_hiding: E::ScalarField,
    point: (Vec<E::ScalarField>, Vec<E::ScalarField>),
}

struct VerifierWitnessClaim<E: Pairing> {
    value_commitment: E::G1,
    point: (Vec<E::ScalarField>, Vec<E::ScalarField>),
}

fn prove_with_witness_claims<E, T>(
    prove: impl FnOnce(&mut Vec<ProverWitnessClaim<E>>) -> Result<T, GlobalMlpError>,
) -> Result<(T, Vec<ProverWitnessClaim<E>>), GlobalMlpError>
where
    E: Pairing,
{
    let mut claims = Vec::new();
    let proof = prove(&mut claims)?;
    Ok((proof, claims))
}

pub struct GlobalWitnessBatchProof<E: Pairing> {
    transcript: TranSeq<E::ScalarField, E::G1>,
    opening: PcsTrans<E>,
}

struct CommittedArtifact<E: Pairing> {
    committed: CommittedMatrix<E>,
    dense: DenseMatCM<i32, E::ScalarField>,
}

struct CommittedGlobalMlp<E: Pairing> {
    witness: CommittedMatrix<E>,
    io: CommittedMatrix<E>,
    model: CommittedMatrix<E>,
    a_l: CommittedArtifact<E>,
    a_r: CommittedArtifact<E>,
    a_o: CommittedArtifact<E>,
    selectors: BTreeMap<GlobalOperationType, CommittedArtifact<E>>,
    io_selector: CommittedArtifact<E>,
    model_selector: CommittedArtifact<E>,
    negative: CommittedArtifact<E>,
    output_bits: Vec<CommittedArtifact<E>>,
    negative_bits: Vec<CommittedArtifact<E>>,
}

pub struct GlobalIoProof<E: Pairing> {
    value_commitment: E::G1,
    io_opening: PcsTrans<E>,
    route: RoutedOpeningProof<E>,
}

pub struct GlobalMultiplicationProof<E: Pairing> {
    output_value_commitment: E::G1,
    transcript: TranSeq<E::ScalarField, E::G1>,
    left_route: RoutedOpeningProof<E>,
    right_route: RoutedOpeningProof<E>,
    output_route: RoutedOpeningProof<E>,
}

pub struct GlobalLinearProof<E: Pairing> {
    value_commitments: Vec<E::G1>,
    routes: Vec<RoutedOpeningProof<E>>,
}

pub struct GlobalZeroProductProof<E: Pairing> {
    transcript: TranSeq<E::ScalarField, E::G1>,
    left_route: RoutedOpeningProof<E>,
    right_route: RoutedOpeningProof<E>,
}

pub struct GlobalBooleanProof<E: Pairing> {
    transcript: TranSeq<E::ScalarField, E::G1>,
    direct_route: RoutedOpeningProof<E>,
    complement_route: RoutedOpeningProof<E>,
}

pub struct GlobalRangeProof<E: Pairing> {
    value_commitment: E::G1,
    bit_commitments: Vec<E::G1>,
    routes: Vec<RoutedOpeningProof<E>>,
    boolean_proofs: Vec<GlobalBooleanProof<E>>,
}

pub struct GlobalActivationProof<E: Pairing> {
    linear: GlobalLinearProof<E>,
    zero_product: GlobalZeroProductProof<E>,
    output_range: GlobalRangeProof<E>,
    negative_range: GlobalRangeProof<E>,
}

pub struct GlobalMlpProof<E: Pairing> {
    io: GlobalIoProof<E>,
    model: GlobalIoProof<E>,
    multiplication: GlobalMultiplicationProof<E>,
    addition: GlobalLinearProof<E>,
    activation: GlobalActivationProof<E>,
    witness_batch: GlobalWitnessBatchProof<E>,
}

impl<E: Pairing> GlobalMlpProof<E> {
    pub fn witness_commitment_count(&self) -> usize {
        1
    }

    pub fn witness_opening_count(&self) -> usize {
        1
    }

    pub fn operation_equation_count(&self) -> usize {
        4
    }

    pub fn serialized_size_bytes(&self) -> usize {
        proof_size_io(&self.io)
            + proof_size_io(&self.model)
            + proof_size_multiplication(&self.multiplication)
            + proof_size_linear(&self.addition)
            + proof_size_activation(&self.activation)
            + serialized_size(&self.witness_batch.transcript)
            + serialized_size(&self.witness_batch.opening)
    }
}

fn serialized_size<T: CanonicalSerialize>(value: &T) -> usize {
    value.serialized_size(Compress::Yes)
}

fn proof_size_route<E: Pairing>(proof: &RoutedOpeningProof<E>) -> usize {
    serialized_size(&proof.transcript)
        + serialized_size(&proof.selector_opening)
        + proof
            .wiring_opening
            .as_ref()
            .map(serialized_size)
            .unwrap_or(0)
}

fn proof_size_io<E: Pairing>(proof: &GlobalIoProof<E>) -> usize {
    serialized_size(&proof.value_commitment)
        + serialized_size(&proof.io_opening)
        + proof_size_route(&proof.route)
}

fn proof_size_multiplication<E: Pairing>(proof: &GlobalMultiplicationProof<E>) -> usize {
    serialized_size(&proof.output_value_commitment)
        + serialized_size(&proof.transcript)
        + proof_size_route(&proof.left_route)
        + proof_size_route(&proof.right_route)
        + proof_size_route(&proof.output_route)
}

fn proof_size_linear<E: Pairing>(proof: &GlobalLinearProof<E>) -> usize {
    proof
        .value_commitments
        .iter()
        .map(serialized_size)
        .sum::<usize>()
        + proof.routes.iter().map(proof_size_route).sum::<usize>()
}

fn proof_size_boolean<E: Pairing>(proof: &GlobalBooleanProof<E>) -> usize {
    serialized_size(&proof.transcript)
        + proof_size_route(&proof.direct_route)
        + proof_size_route(&proof.complement_route)
}

fn proof_size_range<E: Pairing>(proof: &GlobalRangeProof<E>) -> usize {
    serialized_size(&proof.value_commitment)
        + proof
            .bit_commitments
            .iter()
            .map(serialized_size)
            .sum::<usize>()
        + proof.routes.iter().map(proof_size_route).sum::<usize>()
        + proof
            .boolean_proofs
            .iter()
            .map(proof_size_boolean)
            .sum::<usize>()
}

fn proof_size_activation<E: Pairing>(proof: &GlobalActivationProof<E>) -> usize {
    proof_size_linear(&proof.linear)
        + serialized_size(&proof.zero_product.transcript)
        + proof_size_route(&proof.zero_product.left_route)
        + proof_size_route(&proof.zero_product.right_route)
        + proof_size_range(&proof.output_range)
        + proof_size_range(&proof.negative_range)
}

fn commit_artifact<E: Pairing>(
    srs: &SRS<E>,
    artifact: MatrixArtifact<E::ScalarField>,
) -> Result<CommittedArtifact<E>, GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    let hiding = E::ScalarField::rand(&mut ark_std::rand::thread_rng());
    Ok(CommittedArtifact {
        committed: CommittedMatrix::commit(srs, artifact.virtual_matrix, hiding, 1)?,
        dense: artifact.dense,
    })
}

fn commit_global_artifacts<E: Pairing>(
    srs: &SRS<E>,
    artifacts: GlobalMlpArtifacts<E::ScalarField>,
) -> Result<CommittedGlobalMlp<E>, GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    let rng = &mut ark_std::rand::thread_rng();
    let witness = CommittedMatrix::commit(
        srs,
        artifacts.witness,
        E::ScalarField::rand(rng),
        artifacts.shape.commitment_bit_width,
    )?;
    let io = CommittedMatrix::commit(
        srs,
        artifacts.io,
        E::ScalarField::rand(rng),
        artifacts.shape.commitment_bit_width,
    )?;
    let model = CommittedMatrix::commit(
        srs,
        artifacts.model,
        E::ScalarField::rand(rng),
        artifacts.shape.commitment_bit_width,
    )?;
    let mut selectors = BTreeMap::new();
    for (operation_type, selector) in artifacts.selectors {
        selectors.insert(operation_type, commit_artifact(srs, selector)?);
    }
    Ok(CommittedGlobalMlp {
        witness,
        io,
        model,
        a_l: commit_artifact(srs, artifacts.a_l)?,
        a_r: commit_artifact(srs, artifacts.a_r)?,
        a_o: commit_artifact(srs, artifacts.a_o)?,
        selectors,
        io_selector: CommittedArtifact {
            committed: CommittedMatrix::commit(
                srs,
                artifacts.io_selector.virtual_matrix,
                E::ScalarField::zero(),
                1,
            )?,
            dense: artifacts.io_selector.dense,
        },
        model_selector: CommittedArtifact {
            committed: CommittedMatrix::commit(
                srs,
                artifacts.model_selector.virtual_matrix,
                E::ScalarField::zero(),
                1,
            )?,
            dense: artifacts.model_selector.dense,
        },
        negative: commit_artifact(srs, artifacts.negative)?,
        output_bits: artifacts
            .output_bits
            .into_iter()
            .map(|artifact| commit_artifact(srs, artifact))
            .collect::<Result<_, _>>()?,
        negative_bits: artifacts
            .negative_bits
            .into_iter()
            .map(|artifact| commit_artifact(srs, artifact))
            .collect::<Result<_, _>>()?,
    })
}

impl<E: Pairing> CommittedGlobalMlp<E> {
    fn instance(
        &self,
        manifest: MlpManifest,
        shape: GlobalMlpShape,
        public_input: Vec<i32>,
        public_output: Vec<i32>,
    ) -> GlobalMlpInstance<E> {
        GlobalMlpInstance {
            manifest,
            shape,
            public_input,
            public_output,
            witness_commitment: self.witness.commitment,
            io_commitment: self.io.commitment,
            model_commitment: self.model.commitment,
            architecture: GlobalArchitectureCommitments {
                a_l: self.a_l.committed.commitment,
                a_r: self.a_r.committed.commitment,
                a_o: self.a_o.committed.commitment,
                selectors: self
                    .selectors
                    .iter()
                    .map(|(operation_type, artifact)| {
                        (*operation_type, artifact.committed.commitment)
                    })
                    .collect(),
                io_selector: self.io_selector.committed.commitment,
                model_selector: self.model_selector.committed.commitment,
                negative: self.negative.committed.commitment,
                output_bits: self
                    .output_bits
                    .iter()
                    .map(|artifact| artifact.committed.commitment)
                    .collect(),
                negative_bits: self
                    .negative_bits
                    .iter()
                    .map(|artifact| artifact.committed.commitment)
                    .collect(),
            },
        }
    }
}

fn identity_matrix<F>(n: usize) -> DenseMatCM<i32, F>
where
    F: ark_ff::PrimeField + From<i32>,
{
    let mut columns = zero_columns(n, n);
    for index in 0..n {
        columns[index][index] = 1;
    }
    DenseMatCM::from_data(columns)
}

fn prove_selected_route<E: Pairing>(
    srs: &SRS<E>,
    value_commitment: E::G1,
    value_hiding: E::ScalarField,
    point: (Vec<E::ScalarField>, Vec<E::ScalarField>),
    selector: &CommittedMatrix<E>,
    wiring: &CommittedMatrix<E>,
    witness: &CommittedMatrix<E>,
    selector_dense: &DenseMatCM<i32, E::ScalarField>,
    wiring_dense: &DenseMatCM<i32, E::ScalarField>,
    n: usize,
    witness_claims: &mut Vec<ProverWitnessClaim<E>>,
) -> Result<RoutedOpeningProof<E>, GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    let selected_columns =
        linear::mat_mul_shortint::<i32>(&selector_dense.data, &wiring_dense.data);
    let selected = DenseMatCM::<i32, E::ScalarField>::from_data(selected_columns.clone());
    let identity = identity_matrix::<E::ScalarField>(n);
    let kronecker = SparseKronecker::<i32, E::ScalarField>::from_data((
        selected_columns,
        identity.data.clone(),
    ));
    let mut transcript = ZkTranSeq::new(&srs.zksrs);

    let routing = ZkMatMul::<
        i32,
        E::ScalarField,
        E::G1,
        SparseKronecker<i32, E::ScalarField>,
        StackedHeteroView<'_, i32>,
    >::new(value_commitment, point, witness.matrix.stacked_shape().0);
    let witness_view = StackedHeteroView::new(&witness.matrix);
    let (
        kronecker_value_commitment,
        witness_value_commitment,
        kronecker_value_hiding,
        witness_value_hiding,
        kronecker_point,
        witness_point,
    ) = routing.reduce_prover(
        &srs.zksrs,
        &mut transcript,
        &kronecker,
        &witness_view,
        value_hiding,
    );

    let kronecker_reduction = ZkKronecker::<
        i32,
        E::ScalarField,
        E::G1,
        DenseMatCM<i32, E::ScalarField>,
        DenseMatCM<i32, E::ScalarField>,
    >::new(
        kronecker_value_commitment,
        kronecker_point,
        selected.get_shape(),
    );
    let (
        selected_value_commitment,
        _identity_value_commitment,
        selected_value_hiding,
        identity_value_hiding,
        selected_point,
        _identity_point,
    ) = kronecker_reduction.reduce_prover(
        &srs.zksrs,
        &mut transcript,
        &selected,
        &identity,
        kronecker_value_hiding,
    );
    transcript.push_without_blinding(identity_value_hiding);

    let selection = ZkMatMul::<
        i32,
        E::ScalarField,
        E::G1,
        DenseMatCM<i32, E::ScalarField>,
        DenseMatCM<i32, E::ScalarField>,
    >::new(
        selected_value_commitment,
        selected_point,
        selector_dense.get_shape().1,
    );
    let (
        selector_value_commitment,
        wiring_value_commitment,
        selector_value_hiding,
        wiring_value_hiding,
        selector_point,
        wiring_point,
    ) = selection.reduce_prover(
        &srs.zksrs,
        &mut transcript,
        selector_dense,
        wiring_dense,
        selected_value_hiding,
    );

    witness_claims.push(ProverWitnessClaim {
        value_commitment: witness_value_commitment,
        value_hiding: witness_value_hiding,
        point: witness_point,
    });
    let selector_opening = open_committed_matrix(
        srs,
        selector,
        &selector_point,
        selector_value_commitment,
        selector_value_hiding,
    )?;
    let wiring_opening = open_committed_matrix(
        srs,
        wiring,
        &wiring_point,
        wiring_value_commitment,
        wiring_value_hiding,
    )?;
    Ok(RoutedOpeningProof {
        transcript: transcript.publish_trans(),
        selector_opening,
        wiring_opening: Some(wiring_opening),
    })
}

fn prove_direct_route<E: Pairing>(
    srs: &SRS<E>,
    value_commitment: E::G1,
    value_hiding: E::ScalarField,
    point: (Vec<E::ScalarField>, Vec<E::ScalarField>),
    route: &CommittedMatrix<E>,
    witness: &CommittedMatrix<E>,
    route_dense: &DenseMatCM<i32, E::ScalarField>,
    n: usize,
    witness_claims: &mut Vec<ProverWitnessClaim<E>>,
) -> Result<RoutedOpeningProof<E>, GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    let identity = identity_matrix::<E::ScalarField>(n);
    let kronecker = SparseKronecker::<i32, E::ScalarField>::from_data((
        route_dense.data.clone(),
        identity.data.clone(),
    ));
    let mut transcript = ZkTranSeq::new(&srs.zksrs);
    let routing = ZkMatMul::<
        i32,
        E::ScalarField,
        E::G1,
        SparseKronecker<i32, E::ScalarField>,
        StackedHeteroView<'_, i32>,
    >::new(value_commitment, point, witness.matrix.stacked_shape().0);
    let witness_view = StackedHeteroView::new(&witness.matrix);
    let (
        kronecker_value_commitment,
        witness_value_commitment,
        kronecker_value_hiding,
        witness_value_hiding,
        kronecker_point,
        witness_point,
    ) = routing.reduce_prover(
        &srs.zksrs,
        &mut transcript,
        &kronecker,
        &witness_view,
        value_hiding,
    );
    let kronecker_reduction = ZkKronecker::<
        i32,
        E::ScalarField,
        E::G1,
        DenseMatCM<i32, E::ScalarField>,
        DenseMatCM<i32, E::ScalarField>,
    >::new(
        kronecker_value_commitment,
        kronecker_point,
        route_dense.get_shape(),
    );
    let (
        route_value_commitment,
        _identity_value_commitment,
        route_value_hiding,
        identity_value_hiding,
        route_point,
        _identity_point,
    ) = kronecker_reduction.reduce_prover(
        &srs.zksrs,
        &mut transcript,
        route_dense,
        &identity,
        kronecker_value_hiding,
    );
    transcript.push_without_blinding(identity_value_hiding);
    witness_claims.push(ProverWitnessClaim {
        value_commitment: witness_value_commitment,
        value_hiding: witness_value_hiding,
        point: witness_point,
    });
    let selector_opening = open_committed_matrix(
        srs,
        route,
        &route_point,
        route_value_commitment,
        route_value_hiding,
    )?;
    Ok(RoutedOpeningProof {
        transcript: transcript.publish_trans(),
        selector_opening,
        wiring_opening: None,
    })
}

fn open_committed_matrix<E: Pairing>(
    srs: &SRS<E>,
    matrix: &CommittedMatrix<E>,
    point: &(Vec<E::ScalarField>, Vec<E::ScalarField>),
    value_commitment: E::G1,
    value_hiding: E::ScalarField,
) -> Result<PcsTrans<E>, GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    let packed_point = linear::reshape_points_keep_projection(point);
    Ok(open_stacked_i32::<E>(
        &srs.pc_pp,
        &matrix.matrix,
        &packed_point.0,
        &packed_point.1,
        E::pairing(value_commitment, srs.h_hat),
        matrix.commitment,
        &matrix.tier_one,
        value_hiding,
        matrix.hiding,
    )?)
}

#[allow(clippy::too_many_arguments)]
fn verify_selected_route<E: Pairing>(
    srs: &SRS<E>,
    proof: &RoutedOpeningProof<E>,
    value_commitment: E::G1,
    point: (Vec<E::ScalarField>, Vec<E::ScalarField>),
    selector_commitment: PairingOutput<E>,
    wiring_commitment: PairingOutput<E>,
    shape: GlobalMlpShape,
    witness_claims: &mut Vec<VerifierWitnessClaim<E>>,
) -> bool
where
    E::ScalarField: From<i32>,
{
    if proof.wiring_opening.is_none() {
        return false;
    }
    let mut transcript = proof.transcript.clone();
    let routing = ZkMatMul::<
        i32,
        E::ScalarField,
        E::G1,
        SparseKronecker<i32, E::ScalarField>,
        StackedHeteroView<'static, i32>,
    >::new(value_commitment, point, shape.witness_slots * shape.n);
    let (
        route_ok,
        kronecker_value_commitment,
        witness_value_commitment,
        kronecker_point,
        witness_point,
    ) = routing.verify_as_subprotocol(&srs.zksrs, &mut transcript);
    if !route_ok {
        return false;
    }
    let kronecker_reduction = ZkKronecker::<
        i32,
        E::ScalarField,
        E::G1,
        DenseMatCM<i32, E::ScalarField>,
        DenseMatCM<i32, E::ScalarField>,
    >::new(
        kronecker_value_commitment,
        kronecker_point,
        (shape.gate_slots, shape.witness_slots),
    );
    let (
        kronecker_ok,
        selected_value_commitment,
        identity_value_commitment,
        selected_point,
        identity_point,
    ) = kronecker_reduction.verify_as_subprotocol(&srs.zksrs, &mut transcript);
    if !kronecker_ok
        || !verify_public_identity(
            srs,
            &mut transcript,
            identity_value_commitment,
            &identity_point,
        )
    {
        return false;
    }
    let selection = ZkMatMul::<
        i32,
        E::ScalarField,
        E::G1,
        DenseMatCM<i32, E::ScalarField>,
        DenseMatCM<i32, E::ScalarField>,
    >::new(selected_value_commitment, selected_point, shape.gate_slots);
    let (
        selection_ok,
        selector_value_commitment,
        wiring_value_commitment,
        selector_point,
        wiring_point,
    ) = selection.verify_as_subprotocol(&srs.zksrs, &mut transcript);
    let openings_ok = selection_ok
        && transcript.check_fiat_shamir()
        && transcript.is_fully_consumed()
        && verify_virtual_opening(
            srs,
            selector_commitment,
            selector_value_commitment,
            &selector_point,
            &proof.selector_opening,
        )
        && verify_virtual_opening(
            srs,
            wiring_commitment,
            wiring_value_commitment,
            &wiring_point,
            proof.wiring_opening.as_ref().unwrap(),
        );
    if openings_ok {
        witness_claims.push(VerifierWitnessClaim {
            value_commitment: witness_value_commitment,
            point: witness_point,
        });
    }
    openings_ok
}

#[allow(clippy::too_many_arguments)]
fn verify_direct_route<E: Pairing>(
    srs: &SRS<E>,
    proof: &RoutedOpeningProof<E>,
    value_commitment: E::G1,
    point: (Vec<E::ScalarField>, Vec<E::ScalarField>),
    route_commitment: PairingOutput<E>,
    route_shape: (usize, usize),
    witness_slots: usize,
    n: usize,
    witness_claims: &mut Vec<VerifierWitnessClaim<E>>,
) -> bool
where
    E::ScalarField: From<i32>,
{
    if proof.wiring_opening.is_some() {
        return false;
    }
    let mut transcript = proof.transcript.clone();
    let routing = ZkMatMul::<
        i32,
        E::ScalarField,
        E::G1,
        SparseKronecker<i32, E::ScalarField>,
        StackedHeteroView<'static, i32>,
    >::new(value_commitment, point, witness_slots * n);
    let (
        route_ok,
        kronecker_value_commitment,
        witness_value_commitment,
        kronecker_point,
        witness_point,
    ) = routing.verify_as_subprotocol(&srs.zksrs, &mut transcript);
    if !route_ok {
        return false;
    }
    let kronecker_reduction = ZkKronecker::<
        i32,
        E::ScalarField,
        E::G1,
        DenseMatCM<i32, E::ScalarField>,
        DenseMatCM<i32, E::ScalarField>,
    >::new(kronecker_value_commitment, kronecker_point, route_shape);
    let (
        kronecker_ok,
        route_value_commitment,
        identity_value_commitment,
        route_point,
        identity_point,
    ) = kronecker_reduction.verify_as_subprotocol(&srs.zksrs, &mut transcript);
    let openings_ok = kronecker_ok
        && verify_public_identity(
            srs,
            &mut transcript,
            identity_value_commitment,
            &identity_point,
        )
        && transcript.check_fiat_shamir()
        && transcript.is_fully_consumed()
        && verify_virtual_opening(
            srs,
            route_commitment,
            route_value_commitment,
            &route_point,
            &proof.selector_opening,
        );
    if openings_ok {
        witness_claims.push(VerifierWitnessClaim {
            value_commitment: witness_value_commitment,
            point: witness_point,
        });
    }
    openings_ok
}

fn verify_public_identity<E: Pairing>(
    srs: &SRS<E>,
    transcript: &mut TranSeq<E::ScalarField, E::G1>,
    value_commitment: E::G1,
    point: &(Vec<E::ScalarField>, Vec<E::ScalarField>),
) -> bool {
    let Some(TranElem::Field(hiding)) = transcript.data.get(transcript.pointer).cloned() else {
        return false;
    };
    transcript.pointer += 1;
    let value = xi::xi_ip_from_challenges(&point.0, &point.1);
    srs.zksrs.commit(value, hiding) == value_commitment
}

fn verify_virtual_opening<E: Pairing>(
    srs: &SRS<E>,
    matrix_commitment: PairingOutput<E>,
    value_commitment: E::G1,
    point: &(Vec<E::ScalarField>, Vec<E::ScalarField>),
    opening: &PcsTrans<E>,
) -> bool {
    let packed_point = linear::reshape_points_keep_projection(point);
    if !opening.has_rounds(packed_point.0.len(), packed_point.1.len()) {
        return false;
    }
    SmartPC::<E>::verify(
        &srs.pc_pp,
        matrix_commitment,
        E::pairing(value_commitment, srs.h_hat),
        &packed_point.0,
        &packed_point.1,
        opening,
    )
    .unwrap_or(false)
}

fn prove_witness_batch<E: Pairing>(
    srs: &SRS<E>,
    witness: &CommittedMatrix<E>,
    claims: Vec<ProverWitnessClaim<E>>,
) -> Result<GlobalWitnessBatchProof<E>, GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    if claims.is_empty() {
        return Err(GlobalMlpError::ProtocolIncomplete);
    }
    let mut batch = BatchProjectionPoints::<i32, E::ScalarField, E::G1>::new();
    for claim in &claims {
        batch.push_point(claim.value_commitment, claim.point.clone());
    }
    let mut transcript = ZkTranSeq::new(&srs.zksrs);
    let witness_view = StackedHeteroView::new(&witness.matrix);
    let (value_commitment, value_hiding, point) = batch.reduce_prover(
        &srs.zksrs,
        &mut transcript,
        &witness_view,
        claims.iter().map(|claim| claim.value_hiding).collect(),
    );
    let opening = open_committed_matrix(srs, witness, &point, value_commitment, value_hiding)?;
    Ok(GlobalWitnessBatchProof {
        transcript: transcript.publish_trans(),
        opening,
    })
}

fn verify_witness_batch<E: Pairing>(
    srs: &SRS<E>,
    witness_commitment: PairingOutput<E>,
    proof: &GlobalWitnessBatchProof<E>,
    claims: Vec<VerifierWitnessClaim<E>>,
) -> bool
where
    E::ScalarField: From<i32>,
{
    if claims.is_empty() {
        return false;
    }
    let mut batch = BatchProjectionPoints::<i32, E::ScalarField, E::G1>::new();
    for claim in claims {
        batch.push_point(claim.value_commitment, claim.point);
    }
    let mut transcript = proof.transcript.clone();
    let (batch_ok, value_commitment, point) =
        batch.verify_as_subprotocol(&srs.zksrs, &mut transcript);
    batch_ok
        && transcript.check_fiat_shamir()
        && transcript.is_fully_consumed()
        && verify_virtual_opening(
            srs,
            witness_commitment,
            value_commitment,
            &point,
            &proof.opening,
        )
}

fn derive_global_point<E: Pairing>(
    instance: &GlobalMlpInstance<E>,
    tag: usize,
    rows: usize,
    cols: usize,
) -> (Vec<E::ScalarField>, Vec<E::ScalarField>) {
    let mut fs = FiatShamir::new();
    fs.push(&0x48445231_usize);
    fs.push(&tag);
    fs.push(&instance.manifest.widths.len());
    for width in &instance.manifest.widths {
        fs.push(width);
    }
    fs.push(&instance.manifest.commitment_bit_width);
    fs.push(&instance.manifest.relu_magnitude_bits);
    for size in [
        instance.shape.n,
        instance.shape.witness_slots,
        instance.shape.gate_slots,
        instance.shape.relu_magnitude_bits,
        instance.shape.commitment_bit_width,
    ] {
        fs.push(&size);
    }
    fs.push(&instance.public_input.len());
    for value in &instance.public_input {
        fs.push(&u32::from_le_bytes(value.to_le_bytes()));
    }
    fs.push(&instance.public_output.len());
    for value in &instance.public_output {
        fs.push(&u32::from_le_bytes(value.to_le_bytes()));
    }
    fs.push(&instance.witness_commitment);
    fs.push(&instance.io_commitment);
    fs.push(&instance.model_commitment);
    fs.push(&instance.architecture.a_l);
    fs.push(&instance.architecture.a_r);
    fs.push(&instance.architecture.a_o);
    fs.push(&instance.architecture.io_selector);
    fs.push(&instance.architecture.model_selector);
    fs.push(&instance.architecture.negative);
    for operation_type in [
        GlobalOperationType::Multiplication,
        GlobalOperationType::Addition,
        GlobalOperationType::Activation,
    ] {
        fs.push(&instance.architecture.selectors[&operation_type]);
    }
    for commitment in instance
        .architecture
        .output_bits
        .iter()
        .chain(instance.architecture.negative_bits.iter())
    {
        fs.push(commitment);
    }
    let left = (0..rows.ilog2())
        .map(|_| fs.gen_challenge::<E::ScalarField>())
        .collect();
    let right = (0..cols.ilog2())
        .map(|_| fs.gen_challenge::<E::ScalarField>())
        .collect();
    (left, right)
}

fn prove_io_equation<E: Pairing>(
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
    let hiding = E::ScalarField::zero();
    let value_commitment = srs.zksrs.commit(value, hiding);
    let io_opening = open_committed_matrix(srs, &committed.io, &point, value_commitment, hiding)?;
    let route = prove_direct_route(
        srs,
        value_commitment,
        hiding,
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

fn verify_io_equation<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    proof: &GlobalIoProof<E>,
    witness_claims: &mut Vec<VerifierWitnessClaim<E>>,
) -> bool
where
    E::ScalarField: From<i32>,
{
    let point = derive_global_point::<E>(
        instance,
        0,
        instance.shape.witness_slots * instance.shape.n,
        instance.shape.n,
    );
    verify_virtual_opening(
        srs,
        instance.io_commitment,
        proof.value_commitment,
        &point,
        &proof.io_opening,
    ) && verify_direct_route(
        srs,
        &proof.route,
        proof.value_commitment,
        point,
        instance.architecture.io_selector,
        (instance.shape.witness_slots, instance.shape.witness_slots),
        instance.shape.witness_slots,
        instance.shape.n,
        witness_claims,
    )
}

fn prove_model_equation<E: Pairing>(
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
        0x4d4f444c,
        instance.shape.witness_slots * instance.shape.n,
        instance.shape.n,
    );
    let value = committed.model.matrix.proj_stacked_lr(&point.0, &point.1)?;
    let hiding = E::ScalarField::rand(&mut ark_std::rand::thread_rng());
    let value_commitment = srs.zksrs.commit(value, hiding);
    let io_opening =
        open_committed_matrix(srs, &committed.model, &point, value_commitment, hiding)?;
    let route = prove_direct_route(
        srs,
        value_commitment,
        hiding,
        point,
        &committed.model_selector.committed,
        &committed.witness,
        &committed.model_selector.dense,
        instance.shape.n,
        witness_claims,
    )?;
    Ok(GlobalIoProof {
        value_commitment,
        io_opening,
        route,
    })
}

fn verify_model_equation<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    proof: &GlobalIoProof<E>,
    witness_claims: &mut Vec<VerifierWitnessClaim<E>>,
) -> bool
where
    E::ScalarField: From<i32>,
{
    let point = derive_global_point::<E>(
        instance,
        0x4d4f444c,
        instance.shape.witness_slots * instance.shape.n,
        instance.shape.n,
    );
    verify_virtual_opening(
        srs,
        instance.model_commitment,
        proof.value_commitment,
        &point,
        &proof.io_opening,
    ) && verify_direct_route(
        srs,
        &proof.route,
        proof.value_commitment,
        point,
        instance.architecture.model_selector,
        (instance.shape.witness_slots, instance.shape.witness_slots),
        instance.shape.witness_slots,
        instance.shape.n,
        witness_claims,
    )
}

fn prove_multiplication_equation<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    encoding: &GlobalMlpEncoding<E>,
    committed: &CommittedGlobalMlp<E>,
    witness_claims: &mut Vec<ProverWitnessClaim<E>>,
) -> Result<GlobalMultiplicationProof<E>, GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    let left =
        encoding.routed_matrix(GlobalOperationType::Multiplication, GlobalBlockRole::Left)?;
    let right =
        encoding.routed_matrix(GlobalOperationType::Multiplication, GlobalBlockRole::Right)?;
    let output =
        encoding.routed_matrix(GlobalOperationType::Multiplication, GlobalBlockRole::Output)?;
    let point = derive_global_point::<E>(
        instance,
        1,
        instance.shape.gate_slots * instance.shape.n,
        instance.shape.n,
    );
    let output_value = output.proj_stacked_lr(&point.0, &point.1)?;
    let output_hiding = E::ScalarField::rand(&mut ark_std::rand::thread_rng());
    let output_value_commitment = srs.zksrs.commit(output_value, output_hiding);
    let mut transcript = ZkTranSeq::new(&srs.zksrs);
    let left_diagonal = DiagonalHeteroView::new(&left);
    let right_stacked = StackedHeteroView::new(&right);
    let multiplication = ZkMatMul::<
        i32,
        E::ScalarField,
        E::G1,
        DiagonalHeteroView<'_, i32>,
        StackedHeteroView<'_, i32>,
    >::new(
        output_value_commitment,
        point.clone(),
        instance.shape.gate_slots * instance.shape.n,
    );
    let (
        left_diagonal_value_commitment,
        right_value_commitment,
        left_diagonal_value_hiding,
        right_value_hiding,
        left_diagonal_point,
        right_point,
    ) = multiplication.reduce_prover(
        &srs.zksrs,
        &mut transcript,
        &left_diagonal,
        &right_stacked,
        output_hiding,
    );
    let left_stacked = StackedHeteroView::new(&left);
    let diagonal = ZkDiagonal::<i32, E::ScalarField, E::G1, StackedHeteroView<'_, i32>>::new(
        left_diagonal_value_commitment,
        left_diagonal_point,
        instance.shape.gate_slots,
    );
    let (left_value_commitment, left_value_hiding, left_point) = diagonal.reduce_prover(
        &srs.zksrs,
        &mut transcript,
        &left_stacked,
        left_diagonal_value_hiding,
    );
    let selector = &committed.selectors[&GlobalOperationType::Multiplication];
    let left_route = prove_selected_route(
        srs,
        left_value_commitment,
        left_value_hiding,
        left_point,
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
        right_value_commitment,
        right_value_hiding,
        right_point,
        &selector.committed,
        &committed.a_r.committed,
        &committed.witness,
        &selector.dense,
        &committed.a_r.dense,
        instance.shape.n,
        witness_claims,
    )?;
    let output_route = prove_selected_route(
        srs,
        output_value_commitment,
        output_hiding,
        point,
        &selector.committed,
        &committed.a_o.committed,
        &committed.witness,
        &selector.dense,
        &committed.a_o.dense,
        instance.shape.n,
        witness_claims,
    )?;
    Ok(GlobalMultiplicationProof {
        output_value_commitment,
        transcript: transcript.publish_trans(),
        left_route,
        right_route,
        output_route,
    })
}

fn verify_multiplication_equation<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    proof: &GlobalMultiplicationProof<E>,
    witness_claims: &mut Vec<VerifierWitnessClaim<E>>,
) -> bool
where
    E::ScalarField: From<i32>,
{
    let point = derive_global_point::<E>(
        instance,
        1,
        instance.shape.gate_slots * instance.shape.n,
        instance.shape.n,
    );
    let mut transcript = proof.transcript.clone();
    let multiplication = ZkMatMul::<
        i32,
        E::ScalarField,
        E::G1,
        DiagonalHeteroView<'static, i32>,
        StackedHeteroView<'static, i32>,
    >::new(
        proof.output_value_commitment,
        point.clone(),
        instance.shape.gate_slots * instance.shape.n,
    );
    let (
        multiplication_ok,
        left_diagonal_value_commitment,
        right_value_commitment,
        left_diagonal_point,
        right_point,
    ) = multiplication.verify_as_subprotocol(&srs.zksrs, &mut transcript);
    if !multiplication_ok {
        return false;
    }
    let diagonal = ZkDiagonal::<i32, E::ScalarField, E::G1, StackedHeteroView<'static, i32>>::new(
        left_diagonal_value_commitment,
        left_diagonal_point,
        instance.shape.gate_slots,
    );
    let (diagonal_ok, left_value_commitment, left_point) =
        diagonal.verify_as_subprotocol(&srs.zksrs, &mut transcript);
    let selector_commitment = instance.architecture.selectors[&GlobalOperationType::Multiplication];
    diagonal_ok
        && transcript.check_fiat_shamir()
        && transcript.is_fully_consumed()
        && verify_selected_route(
            srs,
            &proof.left_route,
            left_value_commitment,
            left_point,
            selector_commitment,
            instance.architecture.a_l,
            instance.shape,
            witness_claims,
        )
        && verify_selected_route(
            srs,
            &proof.right_route,
            right_value_commitment,
            right_point,
            selector_commitment,
            instance.architecture.a_r,
            instance.shape,
            witness_claims,
        )
        && verify_selected_route(
            srs,
            &proof.output_route,
            proof.output_value_commitment,
            point,
            selector_commitment,
            instance.architecture.a_o,
            instance.shape,
            witness_claims,
        )
}

fn prove_addition_equation<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    encoding: &GlobalMlpEncoding<E>,
    committed: &CommittedGlobalMlp<E>,
    witness_claims: &mut Vec<ProverWitnessClaim<E>>,
) -> Result<GlobalLinearProof<E>, GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    let left = encoding.routed_matrix(GlobalOperationType::Addition, GlobalBlockRole::Left)?;
    let right = encoding.routed_matrix(GlobalOperationType::Addition, GlobalBlockRole::Right)?;
    let output = encoding.routed_matrix(GlobalOperationType::Addition, GlobalBlockRole::Output)?;
    let point = derive_global_point::<E>(
        instance,
        2,
        instance.shape.gate_slots * instance.shape.n,
        instance.shape.n,
    );
    let left_value = left.proj_stacked_lr(&point.0, &point.1)?;
    let right_value = right.proj_stacked_lr(&point.0, &point.1)?;
    let output_value = output.proj_stacked_lr(&point.0, &point.1)?;
    if left_value + right_value != output_value {
        return Err(GlobalMlpError::UnsatisfiedAddition);
    }

    let rng = &mut ark_std::rand::thread_rng();
    let left_hiding = E::ScalarField::rand(rng);
    let right_hiding = E::ScalarField::rand(rng);
    let output_hiding = left_hiding + right_hiding;
    let value_commitments = vec![
        srs.zksrs.commit(left_value, left_hiding),
        srs.zksrs.commit(right_value, right_hiding),
        srs.zksrs.commit(output_value, output_hiding),
    ];
    let selector = &committed.selectors[&GlobalOperationType::Addition];
    let routes = vec![
        prove_selected_route(
            srs,
            value_commitments[0],
            left_hiding,
            point.clone(),
            &selector.committed,
            &committed.a_l.committed,
            &committed.witness,
            &selector.dense,
            &committed.a_l.dense,
            instance.shape.n,
            witness_claims,
        )?,
        prove_selected_route(
            srs,
            value_commitments[1],
            right_hiding,
            point.clone(),
            &selector.committed,
            &committed.a_r.committed,
            &committed.witness,
            &selector.dense,
            &committed.a_r.dense,
            instance.shape.n,
            witness_claims,
        )?,
        prove_selected_route(
            srs,
            value_commitments[2],
            output_hiding,
            point,
            &selector.committed,
            &committed.a_o.committed,
            &committed.witness,
            &selector.dense,
            &committed.a_o.dense,
            instance.shape.n,
            witness_claims,
        )?,
    ];
    Ok(GlobalLinearProof {
        value_commitments,
        routes,
    })
}

fn verify_addition_equation<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    proof: &GlobalLinearProof<E>,
    witness_claims: &mut Vec<VerifierWitnessClaim<E>>,
) -> bool
where
    E::ScalarField: From<i32>,
{
    if proof.value_commitments.len() != 3 || proof.routes.len() != 3 {
        return false;
    }
    if proof.value_commitments[0] + proof.value_commitments[1] != proof.value_commitments[2] {
        return false;
    }
    let point = derive_global_point::<E>(
        instance,
        2,
        instance.shape.gate_slots * instance.shape.n,
        instance.shape.n,
    );
    let selector = instance.architecture.selectors[&GlobalOperationType::Addition];
    [
        instance.architecture.a_l,
        instance.architecture.a_r,
        instance.architecture.a_o,
    ]
    .iter()
    .enumerate()
    .all(|(index, wiring)| {
        verify_selected_route(
            srs,
            &proof.routes[index],
            proof.value_commitments[index],
            point.clone(),
            selector,
            *wiring,
            instance.shape,
            witness_claims,
        )
    })
}

fn expand_column_zero_point<F: ark_ff::PrimeField>(
    point: (Vec<F>, Vec<F>),
    n: usize,
) -> (Vec<F>, Vec<F>) {
    debug_assert!(point.1.is_empty());
    (point.0, vec![F::zero(); n.ilog2() as usize])
}

fn prove_activation_linear<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    encoding: &GlobalMlpEncoding<E>,
    committed: &CommittedGlobalMlp<E>,
    witness_claims: &mut Vec<ProverWitnessClaim<E>>,
) -> Result<GlobalLinearProof<E>, GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    let preactivation =
        encoding.routed_matrix(GlobalOperationType::Activation, GlobalBlockRole::Left)?;
    let negative = encoding.routed_activation_aux(ActivationRoute::Negative)?;
    let output =
        encoding.routed_matrix(GlobalOperationType::Activation, GlobalBlockRole::Output)?;
    let point = derive_global_point::<E>(
        instance,
        3,
        instance.shape.gate_slots * instance.shape.n,
        instance.shape.n,
    );
    let preactivation_value = preactivation.proj_stacked_lr(&point.0, &point.1)?;
    let negative_value = negative.proj_stacked_lr(&point.0, &point.1)?;
    let output_value = output.proj_stacked_lr(&point.0, &point.1)?;
    if preactivation_value + negative_value != output_value {
        return Err(GlobalMlpError::UnsatisfiedActivation);
    }
    let rng = &mut ark_std::rand::thread_rng();
    let preactivation_hiding = E::ScalarField::rand(rng);
    let negative_hiding = E::ScalarField::rand(rng);
    let output_hiding = preactivation_hiding + negative_hiding;
    let value_commitments = vec![
        srs.zksrs.commit(preactivation_value, preactivation_hiding),
        srs.zksrs.commit(negative_value, negative_hiding),
        srs.zksrs.commit(output_value, output_hiding),
    ];
    let selector = &committed.selectors[&GlobalOperationType::Activation];
    let routes = vec![
        prove_selected_route(
            srs,
            value_commitments[0],
            preactivation_hiding,
            point.clone(),
            &selector.committed,
            &committed.a_l.committed,
            &committed.witness,
            &selector.dense,
            &committed.a_l.dense,
            instance.shape.n,
            witness_claims,
        )?,
        prove_selected_route(
            srs,
            value_commitments[1],
            negative_hiding,
            point.clone(),
            &selector.committed,
            &committed.negative.committed,
            &committed.witness,
            &selector.dense,
            &committed.negative.dense,
            instance.shape.n,
            witness_claims,
        )?,
        prove_selected_route(
            srs,
            value_commitments[2],
            output_hiding,
            point,
            &selector.committed,
            &committed.a_o.committed,
            &committed.witness,
            &selector.dense,
            &committed.a_o.dense,
            instance.shape.n,
            witness_claims,
        )?,
    ];
    Ok(GlobalLinearProof {
        value_commitments,
        routes,
    })
}

fn verify_activation_linear<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    proof: &GlobalLinearProof<E>,
    witness_claims: &mut Vec<VerifierWitnessClaim<E>>,
) -> bool
where
    E::ScalarField: From<i32>,
{
    if proof.value_commitments.len() != 3
        || proof.routes.len() != 3
        || proof.value_commitments[0] + proof.value_commitments[1] != proof.value_commitments[2]
    {
        return false;
    }
    let point = derive_global_point::<E>(
        instance,
        3,
        instance.shape.gate_slots * instance.shape.n,
        instance.shape.n,
    );
    let selector = instance.architecture.selectors[&GlobalOperationType::Activation];
    [
        instance.architecture.a_l,
        instance.architecture.negative,
        instance.architecture.a_o,
    ]
    .iter()
    .enumerate()
    .all(|(index, wiring)| {
        verify_selected_route(
            srs,
            &proof.routes[index],
            proof.value_commitments[index],
            point.clone(),
            selector,
            *wiring,
            instance.shape,
            witness_claims,
        )
    })
}

fn prove_activation_zero_product<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    encoding: &GlobalMlpEncoding<E>,
    committed: &CommittedGlobalMlp<E>,
    witness_claims: &mut Vec<ProverWitnessClaim<E>>,
) -> Result<GlobalZeroProductProof<E>, GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    let output =
        encoding.routed_matrix(GlobalOperationType::Activation, GlobalBlockRole::Output)?;
    let negative = encoding.routed_activation_aux(ActivationRoute::Negative)?;
    let rows = instance.shape.gate_slots * instance.shape.n;
    let point = derive_global_point::<E>(instance, 4, rows, 1);
    let mut transcript = ZkTranSeq::new(&srs.zksrs);
    let product = ZkMatMul::<
        i32,
        E::ScalarField,
        E::G1,
        ElementwiseDiagonalHeteroView<'_, i32>,
        ColumnZeroHeteroView<'_, i32>,
    >::new(E::G1::zero(), point, rows);
    let output_diagonal = ElementwiseDiagonalHeteroView::new(&output);
    let negative_vector = ColumnZeroHeteroView::new(&negative);
    let (
        output_diagonal_commitment,
        negative_commitment,
        output_diagonal_hiding,
        negative_hiding,
        output_diagonal_point,
        negative_point,
    ) = product.reduce_prover(
        &srs.zksrs,
        &mut transcript,
        &output_diagonal,
        &negative_vector,
        E::ScalarField::zero(),
    );
    let diagonal = ZkDiagonal::<i32, E::ScalarField, E::G1, ColumnZeroHeteroView<'_, i32>>::new(
        output_diagonal_commitment,
        output_diagonal_point,
        rows,
    );
    let output_vector = ColumnZeroHeteroView::new(&output);
    let (output_commitment, output_hiding, output_point) = diagonal.reduce_prover(
        &srs.zksrs,
        &mut transcript,
        &output_vector,
        output_diagonal_hiding,
    );
    let selector = &committed.selectors[&GlobalOperationType::Activation];
    let left_route = prove_selected_route(
        srs,
        output_commitment,
        output_hiding,
        expand_column_zero_point(output_point, instance.shape.n),
        &selector.committed,
        &committed.a_o.committed,
        &committed.witness,
        &selector.dense,
        &committed.a_o.dense,
        instance.shape.n,
        witness_claims,
    )?;
    let right_route = prove_selected_route(
        srs,
        negative_commitment,
        negative_hiding,
        expand_column_zero_point(negative_point, instance.shape.n),
        &selector.committed,
        &committed.negative.committed,
        &committed.witness,
        &selector.dense,
        &committed.negative.dense,
        instance.shape.n,
        witness_claims,
    )?;
    Ok(GlobalZeroProductProof {
        transcript: transcript.publish_trans(),
        left_route,
        right_route,
    })
}

fn verify_activation_zero_product<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    proof: &GlobalZeroProductProof<E>,
    witness_claims: &mut Vec<VerifierWitnessClaim<E>>,
) -> bool
where
    E::ScalarField: From<i32>,
{
    let rows = instance.shape.gate_slots * instance.shape.n;
    let point = derive_global_point::<E>(instance, 4, rows, 1);
    let mut transcript = proof.transcript.clone();
    let product = ZkMatMul::<
        i32,
        E::ScalarField,
        E::G1,
        ElementwiseDiagonalHeteroView<'static, i32>,
        ColumnZeroHeteroView<'static, i32>,
    >::new(E::G1::zero(), point, rows);
    let (
        product_ok,
        output_diagonal_commitment,
        negative_commitment,
        output_diagonal_point,
        negative_point,
    ) = product.verify_as_subprotocol(&srs.zksrs, &mut transcript);
    if !product_ok {
        return false;
    }
    let diagonal =
        ZkDiagonal::<i32, E::ScalarField, E::G1, ColumnZeroHeteroView<'static, i32>>::new(
            output_diagonal_commitment,
            output_diagonal_point,
            rows,
        );
    let (diagonal_ok, output_commitment, output_point) =
        diagonal.verify_as_subprotocol(&srs.zksrs, &mut transcript);
    let selector = instance.architecture.selectors[&GlobalOperationType::Activation];
    let transcript_ok = transcript.check_fiat_shamir() && transcript.is_fully_consumed();
    let direct_ok = verify_selected_route(
        srs,
        &proof.left_route,
        output_commitment,
        expand_column_zero_point(output_point, instance.shape.n),
        selector,
        instance.architecture.a_o,
        instance.shape,
        witness_claims,
    );
    let complement_ok = verify_selected_route(
        srs,
        &proof.right_route,
        negative_commitment,
        expand_column_zero_point(negative_point, instance.shape.n),
        selector,
        instance.architecture.negative,
        instance.shape,
        witness_claims,
    );
    diagonal_ok && transcript_ok && direct_ok && complement_ok
}

fn prove_boolean_route<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    matrix: &HeteroBlockMat<i32>,
    selector: &CommittedArtifact<E>,
    wiring: &CommittedArtifact<E>,
    committed: &CommittedGlobalMlp<E>,
    tag: usize,
    witness_claims: &mut Vec<ProverWitnessClaim<E>>,
) -> Result<GlobalBooleanProof<E>, GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    let rows = instance.shape.gate_slots * instance.shape.n;
    let point = derive_global_point::<E>(instance, tag, rows, 1);
    let mut transcript = ZkTranSeq::new(&srs.zksrs);
    let boolean = ZkMatMul::<
        i32,
        E::ScalarField,
        E::G1,
        ElementwiseDiagonalHeteroView<'_, i32>,
        ColumnZeroHeteroView<'_, i32>,
    >::new(E::G1::zero(), point, rows);
    let diagonal_view = ElementwiseDiagonalHeteroView::new(matrix);
    let complement_view = ColumnZeroHeteroView::one_minus(matrix);
    let (
        diagonal_commitment,
        complement_commitment,
        diagonal_hiding,
        complement_hiding,
        diagonal_point,
        complement_point,
    ) = boolean.reduce_prover(
        &srs.zksrs,
        &mut transcript,
        &diagonal_view,
        &complement_view,
        E::ScalarField::zero(),
    );
    let diagonal = ZkDiagonal::<i32, E::ScalarField, E::G1, ColumnZeroHeteroView<'_, i32>>::new(
        diagonal_commitment,
        diagonal_point,
        rows,
    );
    let vector_view = ColumnZeroHeteroView::new(matrix);
    let (direct_commitment, direct_hiding, direct_point) =
        diagonal.reduce_prover(&srs.zksrs, &mut transcript, &vector_view, diagonal_hiding);
    let ones = vec![E::ScalarField::one(); complement_point.0.len()];
    let ones_at_point = xi::xi_ip_from_challenges(&complement_point.0, &ones);
    let derived_commitment =
        srs.zksrs.commit(ones_at_point, E::ScalarField::zero()) - complement_commitment;
    let direct_route = prove_selected_route(
        srs,
        direct_commitment,
        direct_hiding,
        expand_column_zero_point(direct_point, instance.shape.n),
        &selector.committed,
        &wiring.committed,
        &committed.witness,
        &selector.dense,
        &wiring.dense,
        instance.shape.n,
        witness_claims,
    )?;
    let complement_route = prove_selected_route(
        srs,
        derived_commitment,
        -complement_hiding,
        expand_column_zero_point(complement_point, instance.shape.n),
        &selector.committed,
        &wiring.committed,
        &committed.witness,
        &selector.dense,
        &wiring.dense,
        instance.shape.n,
        witness_claims,
    )?;
    Ok(GlobalBooleanProof {
        transcript: transcript.publish_trans(),
        direct_route,
        complement_route,
    })
}

#[allow(clippy::too_many_arguments)]
fn verify_boolean_route<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    proof: &GlobalBooleanProof<E>,
    selector_commitment: PairingOutput<E>,
    wiring_commitment: PairingOutput<E>,
    tag: usize,
    witness_claims: &mut Vec<VerifierWitnessClaim<E>>,
) -> bool
where
    E::ScalarField: From<i32>,
{
    let rows = instance.shape.gate_slots * instance.shape.n;
    let point = derive_global_point::<E>(instance, tag, rows, 1);
    let mut transcript = proof.transcript.clone();
    let boolean = ZkMatMul::<
        i32,
        E::ScalarField,
        E::G1,
        ElementwiseDiagonalHeteroView<'static, i32>,
        ColumnZeroHeteroView<'static, i32>,
    >::new(E::G1::zero(), point, rows);
    let (boolean_ok, diagonal_commitment, complement_commitment, diagonal_point, complement_point) =
        boolean.verify_as_subprotocol(&srs.zksrs, &mut transcript);
    if !boolean_ok {
        return false;
    }
    let diagonal =
        ZkDiagonal::<i32, E::ScalarField, E::G1, ColumnZeroHeteroView<'static, i32>>::new(
            diagonal_commitment,
            diagonal_point,
            rows,
        );
    let (diagonal_ok, direct_commitment, direct_point) =
        diagonal.verify_as_subprotocol(&srs.zksrs, &mut transcript);
    let ones = vec![E::ScalarField::one(); complement_point.0.len()];
    let ones_at_point = xi::xi_ip_from_challenges(&complement_point.0, &ones);
    let derived_commitment =
        srs.zksrs.commit(ones_at_point, E::ScalarField::zero()) - complement_commitment;
    let transcript_ok = transcript.check_fiat_shamir() && transcript.is_fully_consumed();
    let direct_ok = verify_selected_route(
        srs,
        &proof.direct_route,
        direct_commitment,
        expand_column_zero_point(direct_point, instance.shape.n),
        selector_commitment,
        wiring_commitment,
        instance.shape,
        witness_claims,
    );
    let complement_ok = verify_selected_route(
        srs,
        &proof.complement_route,
        derived_commitment,
        expand_column_zero_point(complement_point, instance.shape.n),
        selector_commitment,
        wiring_commitment,
        instance.shape,
        witness_claims,
    );
    diagonal_ok && transcript_ok && direct_ok && complement_ok
}

#[allow(clippy::too_many_arguments)]
fn prove_range_equation<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    value: &HeteroBlockMat<i32>,
    bits: &[HeteroBlockMat<i32>],
    value_wiring: &CommittedArtifact<E>,
    bit_wirings: &[CommittedArtifact<E>],
    committed: &CommittedGlobalMlp<E>,
    tag: usize,
    witness_claims: &mut Vec<ProverWitnessClaim<E>>,
) -> Result<GlobalRangeProof<E>, GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    if bits.len() != instance.shape.relu_magnitude_bits || bits.len() != bit_wirings.len() {
        return Err(GlobalMlpError::ProtocolShapeMismatch);
    }
    let rows = instance.shape.gate_slots * instance.shape.n;
    let vector_point = derive_global_point::<E>(instance, tag, rows, 1);
    let point = expand_column_zero_point(vector_point, instance.shape.n);
    let value_at_point = value.proj_stacked_lr(&point.0, &point.1)?;
    let bit_values = bits
        .iter()
        .map(|bit| bit.proj_stacked_lr(&point.0, &point.1))
        .collect::<Result<Vec<_>, _>>()?;
    let recombined = bit_values
        .iter()
        .enumerate()
        .map(|(bit, value)| *value * E::ScalarField::from(1_i32 << bit))
        .sum::<E::ScalarField>();
    if recombined != value_at_point {
        return Err(GlobalMlpError::UnsatisfiedActivation);
    }
    let rng = &mut ark_std::rand::thread_rng();
    let bit_hidings = bits
        .iter()
        .map(|_| E::ScalarField::rand(rng))
        .collect::<Vec<_>>();
    let value_hiding = bit_hidings
        .iter()
        .enumerate()
        .map(|(bit, hiding)| *hiding * E::ScalarField::from(1_i32 << bit))
        .sum::<E::ScalarField>();
    let value_commitment = srs.zksrs.commit(value_at_point, value_hiding);
    let bit_commitments = bit_values
        .iter()
        .zip(&bit_hidings)
        .map(|(value, hiding)| srs.zksrs.commit(*value, *hiding))
        .collect::<Vec<_>>();
    let selector = &committed.selectors[&GlobalOperationType::Activation];
    let mut routes = Vec::with_capacity(bits.len() + 1);
    routes.push(prove_selected_route(
        srs,
        value_commitment,
        value_hiding,
        point.clone(),
        &selector.committed,
        &value_wiring.committed,
        &committed.witness,
        &selector.dense,
        &value_wiring.dense,
        instance.shape.n,
        witness_claims,
    )?);
    for (index, wiring) in bit_wirings.iter().enumerate() {
        routes.push(prove_selected_route(
            srs,
            bit_commitments[index],
            bit_hidings[index],
            point.clone(),
            &selector.committed,
            &wiring.committed,
            &committed.witness,
            &selector.dense,
            &wiring.dense,
            instance.shape.n,
            witness_claims,
        )?);
    }
    let boolean_proofs = bits
        .iter()
        .zip(bit_wirings)
        .enumerate()
        .map(|(index, (bit, wiring))| {
            prove_boolean_route(
                srs,
                instance,
                bit,
                selector,
                wiring,
                committed,
                tag + 1 + index,
                witness_claims,
            )
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok(GlobalRangeProof {
        value_commitment,
        bit_commitments,
        routes,
        boolean_proofs,
    })
}

#[allow(clippy::too_many_arguments)]
fn verify_range_equation<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    proof: &GlobalRangeProof<E>,
    value_wiring: PairingOutput<E>,
    bit_wirings: &[PairingOutput<E>],
    tag: usize,
    witness_claims: &mut Vec<VerifierWitnessClaim<E>>,
) -> bool
where
    E::ScalarField: From<i32>,
{
    let bits = instance.shape.relu_magnitude_bits;
    if proof.bit_commitments.len() != bits
        || proof.routes.len() != bits + 1
        || proof.boolean_proofs.len() != bits
        || bit_wirings.len() != bits
    {
        return false;
    }
    let recombined = proof
        .bit_commitments
        .iter()
        .enumerate()
        .map(|(bit, commitment)| commitment.mul(E::ScalarField::from(1_i32 << bit)))
        .sum::<E::G1>();
    if recombined != proof.value_commitment {
        return false;
    }
    let rows = instance.shape.gate_slots * instance.shape.n;
    let point = expand_column_zero_point(
        derive_global_point::<E>(instance, tag, rows, 1),
        instance.shape.n,
    );
    let selector = instance.architecture.selectors[&GlobalOperationType::Activation];
    if !verify_selected_route(
        srs,
        &proof.routes[0],
        proof.value_commitment,
        point.clone(),
        selector,
        value_wiring,
        instance.shape,
        witness_claims,
    ) {
        return false;
    }
    for index in 0..bits {
        let bit_route_ok = verify_selected_route(
            srs,
            &proof.routes[index + 1],
            proof.bit_commitments[index],
            point.clone(),
            selector,
            bit_wirings[index],
            instance.shape,
            witness_claims,
        );
        if !bit_route_ok {
            return false;
        }
    }
    for index in 0..bits {
        let boolean_ok = verify_boolean_route(
            srs,
            instance,
            &proof.boolean_proofs[index],
            selector,
            bit_wirings[index],
            tag + 1 + index,
            witness_claims,
        );
        if !boolean_ok {
            return false;
        }
    }
    true
}

fn prove_activation_equation<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    encoding: &GlobalMlpEncoding<E>,
    committed: &CommittedGlobalMlp<E>,
    witness_claims: &mut Vec<ProverWitnessClaim<E>>,
) -> Result<GlobalActivationProof<E>, GlobalMlpError>
where
    E::ScalarField: From<i32>,
{
    let output =
        encoding.routed_matrix(GlobalOperationType::Activation, GlobalBlockRole::Output)?;
    let negative = encoding.routed_activation_aux(ActivationRoute::Negative)?;
    let output_bits = (0..instance.shape.relu_magnitude_bits)
        .map(|bit| encoding.routed_activation_aux(ActivationRoute::OutputBit(bit)))
        .collect::<Result<Vec<_>, _>>()?;
    let negative_bits = (0..instance.shape.relu_magnitude_bits)
        .map(|bit| encoding.routed_activation_aux(ActivationRoute::NegativeBit(bit)))
        .collect::<Result<Vec<_>, _>>()?;
    let (
        (linear, linear_claims),
        (zero_product, zero_product_claims),
        (output_range, output_range_claims),
        (negative_range, negative_range_claims),
    ) = if std::env::var_os("ZKSMART_PARALLEL_EQUATIONS").is_some() {
        let ((linear, zero_product), (output_range, negative_range)) = rayon::join(
            || {
                rayon::join(
                    || {
                        prove_with_witness_claims(|claims| {
                            prove_activation_linear(srs, instance, encoding, committed, claims)
                        })
                    },
                    || {
                        prove_with_witness_claims(|claims| {
                            prove_activation_zero_product(
                                srs, instance, encoding, committed, claims,
                            )
                        })
                    },
                )
            },
            || {
                rayon::join(
                    || {
                        prove_with_witness_claims(|claims| {
                            prove_range_equation(
                                srs,
                                instance,
                                &output,
                                &output_bits,
                                &committed.a_o,
                                &committed.output_bits,
                                committed,
                                100,
                                claims,
                            )
                        })
                    },
                    || {
                        prove_with_witness_claims(|claims| {
                            prove_range_equation(
                                srs,
                                instance,
                                &negative,
                                &negative_bits,
                                &committed.negative,
                                &committed.negative_bits,
                                committed,
                                200,
                                claims,
                            )
                        })
                    },
                )
            },
        );
        (linear?, zero_product?, output_range?, negative_range?)
    } else {
        (
            prove_with_witness_claims(|claims| {
                prove_activation_linear(srs, instance, encoding, committed, claims)
            })?,
            prove_with_witness_claims(|claims| {
                prove_activation_zero_product(srs, instance, encoding, committed, claims)
            })?,
            prove_with_witness_claims(|claims| {
                prove_range_equation(
                    srs,
                    instance,
                    &output,
                    &output_bits,
                    &committed.a_o,
                    &committed.output_bits,
                    committed,
                    100,
                    claims,
                )
            })?,
            prove_with_witness_claims(|claims| {
                prove_range_equation(
                    srs,
                    instance,
                    &negative,
                    &negative_bits,
                    &committed.negative,
                    &committed.negative_bits,
                    committed,
                    200,
                    claims,
                )
            })?,
        )
    };
    witness_claims.extend(linear_claims);
    witness_claims.extend(zero_product_claims);
    witness_claims.extend(output_range_claims);
    witness_claims.extend(negative_range_claims);
    Ok(GlobalActivationProof {
        linear,
        zero_product,
        output_range,
        negative_range,
    })
}

fn verify_activation_equation<E: Pairing>(
    srs: &SRS<E>,
    instance: &GlobalMlpInstance<E>,
    proof: &GlobalActivationProof<E>,
    witness_claims: &mut Vec<VerifierWitnessClaim<E>>,
) -> bool
where
    E::ScalarField: From<i32>,
{
    let linear = verify_activation_linear(srs, instance, &proof.linear, witness_claims);
    let zero_product =
        verify_activation_zero_product(srs, instance, &proof.zero_product, witness_claims);
    let output_range = verify_range_equation(
        srs,
        instance,
        &proof.output_range,
        instance.architecture.a_o,
        &instance.architecture.output_bits,
        100,
        witness_claims,
    );
    let negative_range = verify_range_equation(
        srs,
        instance,
        &proof.negative_range,
        instance.architecture.negative,
        &instance.architecture.negative_bits,
        200,
        witness_claims,
    );
    linear && zero_product && output_range && negative_range
}

impl<E: Pairing> GlobalHdR1csMlpProtocol<E> {
    pub fn setup(encoding: &GlobalMlpEncoding<E>) -> Result<SRS<E>, GlobalMlpError>
    where
        E::ScalarField: From<i32>,
    {
        let artifacts = encoding.artifacts()?;
        let mut maximum_axis = 2;
        for matrix in [
            &artifacts.witness,
            &artifacts.io,
            &artifacts.model,
            &artifacts.a_l.virtual_matrix,
            &artifacts.a_r.virtual_matrix,
            &artifacts.a_o.virtual_matrix,
            &artifacts.io_selector.virtual_matrix,
            &artifacts.model_selector.virtual_matrix,
            &artifacts.negative.virtual_matrix,
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
        encoding: GlobalMlpEncoding<E>,
    ) -> Result<(GlobalMlpInstance<E>, GlobalMlpProof<E>), GlobalMlpError>
    where
        E::ScalarField: From<i32>,
    {
        let shape = encoding.shape();
        let profile = std::env::var_os("ZKSMART_PROFILE_GLOBAL_MLP").is_some();
        let mut phase_start = std::time::Instant::now();
        let committed = commit_global_artifacts(srs, encoding.artifacts()?)?;
        if profile {
            eprintln!(
                "profile_commit_artifacts_seconds={:.6}",
                phase_start.elapsed().as_secs_f64()
            );
            phase_start = std::time::Instant::now();
        }
        let instance = committed.instance(
            encoding.manifest.clone(),
            shape,
            encoding.public_input.clone(),
            encoding.public_output.clone(),
        );
        let (model, model_claims) = prove_with_witness_claims(|claims| {
            prove_model_equation(srs, &instance, &committed, claims)
        })?;
        let (
            (io, io_claims),
            (multiplication, multiplication_claims),
            (addition, addition_claims),
            (activation, activation_claims),
        ) = if std::env::var_os("ZKSMART_PARALLEL_EQUATIONS").is_some() {
            let ((io, multiplication), (addition, activation)) = rayon::join(
                || {
                    rayon::join(
                        || {
                            prove_with_witness_claims(|claims| {
                                prove_io_equation(srs, &instance, &committed, claims)
                            })
                        },
                        || {
                            prove_with_witness_claims(|claims| {
                                prove_multiplication_equation(
                                    srs, &instance, &encoding, &committed, claims,
                                )
                            })
                        },
                    )
                },
                || {
                    rayon::join(
                        || {
                            prove_with_witness_claims(|claims| {
                                prove_addition_equation(
                                    srs, &instance, &encoding, &committed, claims,
                                )
                            })
                        },
                        || {
                            prove_with_witness_claims(|claims| {
                                prove_activation_equation(
                                    srs, &instance, &encoding, &committed, claims,
                                )
                            })
                        },
                    )
                },
            );
            if profile {
                eprintln!(
                    "profile_parallel_equations_seconds={:.6}",
                    phase_start.elapsed().as_secs_f64()
                );
            }
            (io?, multiplication?, addition?, activation?)
        } else {
            let io = prove_with_witness_claims(|claims| {
                prove_io_equation(srs, &instance, &committed, claims)
            })?;
            if profile {
                eprintln!(
                    "profile_io_equation_seconds={:.6}",
                    phase_start.elapsed().as_secs_f64()
                );
                phase_start = std::time::Instant::now();
            }
            let multiplication = prove_with_witness_claims(|claims| {
                prove_multiplication_equation(srs, &instance, &encoding, &committed, claims)
            })?;
            if profile {
                eprintln!(
                    "profile_multiplication_equation_seconds={:.6}",
                    phase_start.elapsed().as_secs_f64()
                );
                phase_start = std::time::Instant::now();
            }
            let addition = prove_with_witness_claims(|claims| {
                prove_addition_equation(srs, &instance, &encoding, &committed, claims)
            })?;
            if profile {
                eprintln!(
                    "profile_addition_equation_seconds={:.6}",
                    phase_start.elapsed().as_secs_f64()
                );
                phase_start = std::time::Instant::now();
            }
            let activation = prove_with_witness_claims(|claims| {
                prove_activation_equation(srs, &instance, &encoding, &committed, claims)
            })?;
            if profile {
                eprintln!(
                    "profile_activation_equation_seconds={:.6}",
                    phase_start.elapsed().as_secs_f64()
                );
            }
            (io, multiplication, addition, activation)
        };
        let mut witness_claims = io_claims;
        witness_claims.extend(model_claims);
        witness_claims.extend(multiplication_claims);
        witness_claims.extend(addition_claims);
        witness_claims.extend(activation_claims);
        phase_start = std::time::Instant::now();
        let witness_batch = prove_witness_batch(srs, &committed.witness, witness_claims)?;
        if profile {
            eprintln!(
                "profile_witness_batch_seconds={:.6}",
                phase_start.elapsed().as_secs_f64()
            );
        }
        let proof = GlobalMlpProof {
            io,
            model,
            multiplication,
            addition,
            activation,
            witness_batch,
        };
        Ok((instance, proof))
    }

    pub fn verify(srs: &SRS<E>, instance: &GlobalMlpInstance<E>, proof: &GlobalMlpProof<E>) -> bool
    where
        E::ScalarField: From<i32>,
    {
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            Self::verify_inner(srs, instance, proof)
        }))
        .unwrap_or(false)
    }

    fn verify_inner(
        srs: &SRS<E>,
        instance: &GlobalMlpInstance<E>,
        proof: &GlobalMlpProof<E>,
    ) -> bool
    where
        E::ScalarField: From<i32>,
    {
        let expected_shape = match expected_mlp_shape(&instance.manifest) {
            Ok(shape) => shape,
            Err(_) => return false,
        };
        if instance.shape.n == 0
            || instance.shape != expected_shape
            || !instance.shape.n.is_power_of_two()
            || instance.shape.witness_slots == 0
            || !instance.shape.witness_slots.is_power_of_two()
            || instance.shape.gate_slots == 0
            || !instance.shape.gate_slots.is_power_of_two()
            || instance.shape.relu_magnitude_bits == 0
            || instance.shape.relu_magnitude_bits > 30
            || instance.shape.commitment_bit_width == 0
            || instance.shape.commitment_bit_width > 32
            || instance.public_input.is_empty()
            || instance.public_input.len() != instance.manifest.widths[0]
            || instance.public_output.is_empty()
            || instance.public_output.len() != *instance.manifest.widths.last().unwrap()
            || instance.architecture.output_bits.len() != instance.shape.relu_magnitude_bits
            || instance.architecture.negative_bits.len() != instance.shape.relu_magnitude_bits
            || instance.architecture.selectors.len() != 3
            || [
                GlobalOperationType::Multiplication,
                GlobalOperationType::Addition,
                GlobalOperationType::Activation,
            ]
            .iter()
            .any(|operation_type| !instance.architecture.selectors.contains_key(operation_type))
        {
            return false;
        }
        let expected_io = match public_io_matrix(
            instance.shape,
            &instance.public_input,
            &instance.public_output,
        ) {
            Ok(matrix) => matrix,
            Err(_) => return false,
        };
        let io_point = derive_global_point::<E>(
            instance,
            0,
            instance.shape.witness_slots * instance.shape.n,
            instance.shape.n,
        );
        let expected_io_value = match expected_io.proj_stacked_lr(&io_point.0, &io_point.1) {
            Ok(value) => value,
            Err(_) => return false,
        };
        if proof.io.value_commitment != srs.zksrs.commit(expected_io_value, E::ScalarField::zero())
        {
            return false;
        }
        let expected_io_selector = match public_io_selector::<E::ScalarField>(instance.shape) {
            Ok(selector) => selector,
            Err(_) => return false,
        };
        let expected_io_selector = match CommittedMatrix::commit(
            srs,
            expected_io_selector.virtual_matrix,
            E::ScalarField::zero(),
            1,
        ) {
            Ok(selector) => selector.commitment,
            Err(_) => return false,
        };
        if instance.architecture.io_selector != expected_io_selector {
            return false;
        }
        let expected_model_selector =
            match public_model_selector::<E::ScalarField>(&instance.manifest, instance.shape) {
                Ok(selector) => selector,
                Err(_) => return false,
            };
        let expected_model_selector = match CommittedMatrix::commit(
            srs,
            expected_model_selector.virtual_matrix,
            E::ScalarField::zero(),
            1,
        ) {
            Ok(selector) => selector.commitment,
            Err(_) => return false,
        };
        if instance.architecture.model_selector != expected_model_selector {
            return false;
        }
        let mut witness_claims = Vec::new();
        let io = verify_io_equation(srs, instance, &proof.io, &mut witness_claims);
        let model = verify_model_equation(srs, instance, &proof.model, &mut witness_claims);
        let multiplication = verify_multiplication_equation(
            srs,
            instance,
            &proof.multiplication,
            &mut witness_claims,
        );
        let addition =
            verify_addition_equation(srs, instance, &proof.addition, &mut witness_claims);
        let activation =
            verify_activation_equation(srs, instance, &proof.activation, &mut witness_claims);
        io && model
            && multiplication
            && addition
            && activation
            && verify_witness_batch(
                srs,
                instance.witness_commitment,
                &proof.witness_batch,
                witness_claims,
            )
    }

    pub fn encode<I>(
        manifest: MlpManifest,
        public_input: Vec<i32>,
        expected_public_output: Vec<i32>,
        layers: I,
    ) -> Result<GlobalMlpEncoding<E>, GlobalMlpError>
    where
        I: IntoIterator<Item = MlpLayer>,
    {
        manifest.validate()?;
        if public_input.len() != manifest.widths[0] {
            return Err(GlobalMlpError::InputWidthMismatch);
        }
        if expected_public_output.len() != *manifest.widths.last().unwrap() {
            return Err(GlobalMlpError::OutputWidthMismatch);
        }

        let layers = layers.into_iter().collect::<Vec<_>>();
        if layers.len() != manifest.layers() {
            return Err(GlobalMlpError::LayerCountMismatch);
        }
        for (layer_index, (layer, widths)) in
            layers.iter().zip(manifest.widths.windows(2)).enumerate()
        {
            layer.weight.validate_layout()?;
            if layer.weight.dims != MatrixDims::new(widths[1], widths[0])? {
                return Err(GlobalMlpError::LayerShapeMismatch(layer_index));
            }
            if layer.bias.len() != widths[1] {
                return Err(GlobalMlpError::BiasShapeMismatch(layer_index));
            }
        }

        let mut blocks = Vec::new();
        let input_id = push_vector_block(&mut blocks, public_input.clone())?;
        let output_id = push_vector_block(&mut blocks, expected_public_output.clone())?;
        let io_blocks = vec![input_id, output_id];
        let mut model_blocks = Vec::with_capacity(layers.len() * 2);
        let mut multiplication = Vec::with_capacity(layers.len());
        let mut addition = Vec::with_capacity(layers.len());
        let mut activation = Vec::with_capacity(layers.len());
        let mut activation_aux = Vec::with_capacity(layers.len());
        let mut current_values = public_input.clone();
        let mut current_id = input_id;

        let layer_count = layers.len();
        for (layer_index, layer) in layers.into_iter().enumerate() {
            let product = matrix_vector_product(&layer.weight, &current_values)?;
            let product_id = push_vector_block(&mut blocks, product.clone())?;
            let preactivation = add_vectors(&product, &layer.bias)?;
            let preactivation_id = push_vector_block(&mut blocks, preactivation.clone())?;
            let next_values = preactivation
                .iter()
                .map(|value| (*value).max(0))
                .collect::<Vec<_>>();
            let negative_values = preactivation
                .iter()
                .map(|value| {
                    if *value < 0 {
                        value
                            .checked_neg()
                            .ok_or(GlobalMlpError::ArithmeticOverflow)
                    } else {
                        Ok(0)
                    }
                })
                .collect::<Result<Vec<_>, _>>()?;
            let activation_limit = 1_u32 << manifest.relu_magnitude_bits;
            if next_values
                .iter()
                .chain(negative_values.iter())
                .any(|value| *value < 0 || (*value as u32) >= activation_limit)
            {
                return Err(GlobalMlpError::ValueOutsideActivationRange);
            }
            let next_id = if layer_index + 1 == layer_count {
                if next_values != expected_public_output {
                    return Err(GlobalMlpError::UnexpectedPublicOutput);
                }
                output_id
            } else {
                push_vector_block(&mut blocks, next_values.clone())?
            };
            let negative_id = push_vector_block(&mut blocks, negative_values.clone())?;
            let output_bits =
                push_bit_blocks(&mut blocks, &next_values, manifest.relu_magnitude_bits)?;
            let negative_bits =
                push_bit_blocks(&mut blocks, &negative_values, manifest.relu_magnitude_bits)?;
            let weight_id = push_block(&mut blocks, layer.weight);
            let bias_id = push_vector_block(&mut blocks, layer.bias)?;
            model_blocks.extend([weight_id, bias_id]);
            multiplication.push(GlobalOperation {
                left: weight_id,
                right: Some(current_id),
                output: product_id,
            });
            addition.push(GlobalOperation {
                left: product_id,
                right: Some(bias_id),
                output: preactivation_id,
            });
            activation.push(GlobalOperation {
                left: preactivation_id,
                right: None,
                output: next_id,
            });
            activation_aux.push(ActivationAux {
                negative: negative_id,
                output_bits,
                negative_bits,
            });
            current_values = next_values;
            current_id = next_id;
        }

        let operations = BTreeMap::from([
            (GlobalOperationType::Multiplication, multiplication),
            (GlobalOperationType::Addition, addition),
            (GlobalOperationType::Activation, activation),
        ]);
        Ok(GlobalMlpEncoding {
            manifest,
            public_input,
            public_output: expected_public_output,
            blocks,
            io_blocks,
            model_blocks,
            operations,
            activation_aux,
            _marker: PhantomData,
        })
    }
}

fn push_bit_blocks(
    blocks: &mut Vec<MatrixBlock<i32>>,
    values: &[i32],
    bits: usize,
) -> Result<Vec<GlobalBlockId>, GlobalMlpError> {
    (0..bits)
        .map(|bit| {
            push_vector_block(
                blocks,
                values
                    .iter()
                    .map(|value| (((*value as u32) >> bit) & 1) as i32)
                    .collect(),
            )
        })
        .collect()
}

fn push_block(blocks: &mut Vec<MatrixBlock<i32>>, block: MatrixBlock<i32>) -> GlobalBlockId {
    let id = GlobalBlockId(blocks.len());
    blocks.push(block);
    id
}

fn push_vector_block(
    blocks: &mut Vec<MatrixBlock<i32>>,
    values: Vec<i32>,
) -> Result<GlobalBlockId, GlobalMlpError> {
    let dims = MatrixDims::new(values.len(), 1)?;
    Ok(push_block(
        blocks,
        MatrixBlock::from_columns(dims, vec![values])?,
    ))
}

fn matrix_vector_product(
    weight: &MatrixBlock<i32>,
    input: &[i32],
) -> Result<Vec<i32>, GlobalMlpError> {
    if weight.dims.cols != input.len() {
        return Err(GlobalMlpError::InputWidthMismatch);
    }
    let mut output = vec![0_i64; weight.dims.rows];
    for (column, input_value) in weight.columns.iter().zip(input) {
        for (accumulator, weight_value) in output.iter_mut().zip(column) {
            *accumulator = accumulator
                .checked_add((*weight_value as i64) * (*input_value as i64))
                .ok_or(GlobalMlpError::ArithmeticOverflow)?;
        }
    }
    output
        .into_iter()
        .map(|value| i32::try_from(value).map_err(|_| GlobalMlpError::ArithmeticOverflow))
        .collect()
}

fn add_vectors(left: &[i32], right: &[i32]) -> Result<Vec<i32>, GlobalMlpError> {
    if left.len() != right.len() {
        return Err(GlobalMlpError::VectorLengthMismatch);
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            left.checked_add(*right)
                .ok_or(GlobalMlpError::ArithmeticOverflow)
        })
        .collect()
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GlobalMlpError {
    Manifest(MlpProofError),
    Matrix(HeteroMatrixError),
    InputWidthMismatch,
    OutputWidthMismatch,
    LayerCountMismatch,
    LayerShapeMismatch(usize),
    BiasShapeMismatch(usize),
    VectorLengthMismatch,
    ArithmeticOverflow,
    UnexpectedPublicOutput,
    ValueOutsideActivationRange,
    UnsatisfiedAddition,
    UnsatisfiedActivation,
    ProtocolShapeMismatch,
    Pcs(VirtualPcsError),
    ProtocolIncomplete,
}

impl Display for GlobalMlpError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for GlobalMlpError {}

impl From<MlpProofError> for GlobalMlpError {
    fn from(error: MlpProofError) -> Self {
        Self::Manifest(error)
    }
}

impl From<HeteroMatrixError> for GlobalMlpError {
    fn from(error: HeteroMatrixError) -> Self {
        Self::Matrix(error)
    }
}

impl From<VirtualPcsError> for GlobalMlpError {
    fn from(error: VirtualPcsError) -> Self {
        Self::Pcs(error)
    }
}

#[cfg(test)]
mod tests {
    use ark_bls12_381::Bls12_381;
    use ark_ec::pairing::Pairing;

    use super::{GlobalBlockRole, GlobalHdR1csMlpProtocol, GlobalOperationType};
    use crate::hdr1cs::mlp::{MlpLayer, MlpManifest};
    use crate::utils::heterogeneous::{MatrixBlock, MatrixDims};
    use crate::utils::zktr::TranElem;

    fn block(rows: usize, cols: usize, columns: Vec<Vec<i32>>) -> MatrixBlock<i32> {
        MatrixBlock::from_columns(MatrixDims::new(rows, cols).unwrap(), columns).unwrap()
    }

    fn worked_two_layer_mlp() -> (MlpManifest, Vec<i32>, Vec<i32>, Vec<MlpLayer>) {
        let manifest = MlpManifest {
            widths: vec![4, 2, 1],
            commitment_bit_width: 8,
            relu_magnitude_bits: 5,
        };
        let input = vec![1, 2, 3, 4];
        let output = vec![16];
        let layers = vec![
            MlpLayer {
                weight: block(2, 4, vec![vec![1, 0], vec![0, 1], vec![1, 0], vec![0, 1]]),
                bias: vec![1, -1],
            },
            MlpLayer {
                weight: block(1, 2, vec![vec![1], vec![2]]),
                bias: vec![1],
            },
        ];
        (manifest, input, output, layers)
    }

    fn tiny_one_layer_mlp() -> (MlpManifest, Vec<i32>, Vec<i32>, Vec<MlpLayer>) {
        (
            MlpManifest {
                widths: vec![2, 1],
                commitment_bit_width: 8,
                relu_magnitude_bits: 1,
            },
            vec![1, 1],
            vec![1],
            vec![MlpLayer {
                weight: block(1, 2, vec![vec![1], vec![0]]),
                bias: vec![0],
            }],
        )
    }

    #[test]
    fn global_encoder_groups_by_operation_type_and_shares_cross_layer_wire() {
        let (manifest, input, output, layers) = worked_two_layer_mlp();
        let encoding =
            GlobalHdR1csMlpProtocol::<Bls12_381>::encode(manifest, input, output, layers).unwrap();

        assert_eq!(
            encoding.operation_count(GlobalOperationType::Multiplication),
            2
        );
        assert_eq!(encoding.operation_count(GlobalOperationType::Addition), 2);
        assert_eq!(encoding.operation_count(GlobalOperationType::Activation), 2);

        let first_activation_output = encoding
            .routed_block(GlobalOperationType::Activation, 0, GlobalBlockRole::Output)
            .unwrap();
        let second_multiplication_input = encoding
            .routed_block(
                GlobalOperationType::Multiplication,
                1,
                GlobalBlockRole::Right,
            )
            .unwrap();
        assert_eq!(first_activation_output, second_multiplication_input);
    }

    #[test]
    fn global_protocol_uses_one_witness_commitment_and_type_equations() {
        let (manifest, input, output, layers) = worked_two_layer_mlp();
        let encoding =
            GlobalHdR1csMlpProtocol::<Bls12_381>::encode(manifest, input, output, layers).unwrap();
        let srs = GlobalHdR1csMlpProtocol::<Bls12_381>::setup(&encoding).unwrap();
        let (instance, proof) =
            GlobalHdR1csMlpProtocol::<Bls12_381>::prove(&srs, encoding).unwrap();

        assert_eq!(proof.witness_commitment_count(), 1);
        assert_eq!(proof.witness_opening_count(), 1);
        assert_eq!(proof.operation_equation_count(), 4);
        assert!(GlobalHdR1csMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
    }

    #[test]
    fn global_verifier_rejects_commitment_and_transcript_tampering() {
        let (manifest, input, output, layers) = tiny_one_layer_mlp();
        let encoding =
            GlobalHdR1csMlpProtocol::<Bls12_381>::encode(manifest, input, output, layers).unwrap();
        let srs = GlobalHdR1csMlpProtocol::<Bls12_381>::setup(&encoding).unwrap();
        let (mut instance, mut proof) =
            GlobalHdR1csMlpProtocol::<Bls12_381>::prove(&srs, encoding).unwrap();
        assert!(GlobalHdR1csMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));

        let delta = Bls12_381::pairing(srs.pc_pp.g_0, srs.h_hat);
        instance.witness_commitment += delta;
        assert!(!GlobalHdR1csMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
        instance.witness_commitment -= delta;

        instance.architecture.a_l += delta;
        assert!(!GlobalHdR1csMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
        instance.architecture.a_l -= delta;

        *instance
            .architecture
            .selectors
            .get_mut(&GlobalOperationType::Activation)
            .unwrap() += delta;
        assert!(!GlobalHdR1csMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
        *instance
            .architecture
            .selectors
            .get_mut(&GlobalOperationType::Activation)
            .unwrap() -= delta;

        let last = proof.multiplication.transcript.data.pop().unwrap();
        assert!(!GlobalHdR1csMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
        proof.multiplication.transcript.data.push(last);
        proof.multiplication.transcript.data.push(TranElem::Size(0));
        assert!(!GlobalHdR1csMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
        proof.multiplication.transcript.data.pop();

        let batch_last = proof.witness_batch.transcript.data.pop().unwrap();
        assert!(!GlobalHdR1csMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
        proof.witness_batch.transcript.data.push(batch_last);
        assert!(GlobalHdR1csMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
    }

    #[test]
    fn global_verifier_binds_public_input_and_output() {
        let (manifest, input, output, layers) = tiny_one_layer_mlp();
        let encoding =
            GlobalHdR1csMlpProtocol::<Bls12_381>::encode(manifest, input, output, layers).unwrap();
        let srs = GlobalHdR1csMlpProtocol::<Bls12_381>::setup(&encoding).unwrap();
        let (mut instance, proof) =
            GlobalHdR1csMlpProtocol::<Bls12_381>::prove(&srs, encoding).unwrap();

        assert!(GlobalHdR1csMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));

        instance.public_input[0] += 1;
        assert!(!GlobalHdR1csMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
        instance.public_input[0] -= 1;

        instance.public_output[0] += 1;
        assert!(!GlobalHdR1csMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
    }

    #[test]
    fn global_verifier_binds_the_declared_model_commitment() {
        let (manifest, input, output, layers) = tiny_one_layer_mlp();
        let encoding =
            GlobalHdR1csMlpProtocol::<Bls12_381>::encode(manifest, input, output, layers).unwrap();
        let srs = GlobalHdR1csMlpProtocol::<Bls12_381>::setup(&encoding).unwrap();
        let (mut instance, proof) =
            GlobalHdR1csMlpProtocol::<Bls12_381>::prove(&srs, encoding).unwrap();

        assert!(GlobalHdR1csMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
        let delta = Bls12_381::pairing(srs.pc_pp.g_0, srs.h_hat);
        instance.model_commitment += delta;
        assert!(!GlobalHdR1csMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
    }

    #[test]
    fn global_verifier_binds_manifest_and_model_layout() {
        let (manifest, input, output, layers) = tiny_one_layer_mlp();
        let encoding =
            GlobalHdR1csMlpProtocol::<Bls12_381>::encode(manifest, input, output, layers).unwrap();
        let srs = GlobalHdR1csMlpProtocol::<Bls12_381>::setup(&encoding).unwrap();
        let (mut instance, proof) =
            GlobalHdR1csMlpProtocol::<Bls12_381>::prove(&srs, encoding).unwrap();

        assert!(GlobalHdR1csMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));

        instance.manifest.widths[0] *= 2;
        assert!(!GlobalHdR1csMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
        instance.manifest.widths[0] /= 2;

        let (manifest, input, output, layers) = tiny_one_layer_mlp();
        let mut encoding =
            GlobalHdR1csMlpProtocol::<Bls12_381>::encode(manifest, input, output, layers).unwrap();
        encoding.model_blocks.clear();
        let srs = GlobalHdR1csMlpProtocol::<Bls12_381>::setup(&encoding).unwrap();
        let (instance, proof) =
            GlobalHdR1csMlpProtocol::<Bls12_381>::prove(&srs, encoding).unwrap();
        assert!(!GlobalHdR1csMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof,
        ));
    }
}
