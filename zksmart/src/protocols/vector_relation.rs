use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::{Field, PrimeField, UniformRand};
use ark_poly_commit::smart_pc::{data_structures::Trans as PcsTrans, SmartPC};
use ark_serialize::{CanonicalSerialize, Compress};
use ark_std::{ops::Mul, Zero};
use std::convert::Infallible;
use std::fmt::{Display, Formatter};
use std::marker::PhantomData;

use crate::data_structures::SRS;
use crate::protocols::{
    boolcheck::ZkBoolCheck,
    bucketed_ip::{
        BucketedVector, BvaError, BvaFieldPairSource, BvaLayout, EoLiteBulletProof,
        EoWitnessPairProof, PaddingWorkCounters, SparseBvaFieldPairSource, StreamingEoError,
        StreamingEoProfile, TensorBasisVector,
    },
    hadamard::ZkHadamard,
    virtualpcs::{
        bva_eo_opening_point, commit_bva_field_streaming, commit_sparse_bva_field_streaming,
        commit_stacked_i32, inner_product_bva_field_streaming,
        inner_product_sparse_bva_field_streaming, open_bva_field_streaming,
        open_sparse_bva_field_streaming, open_stacked_i32, BvaFieldSource, BvaI32Source,
        BvaStreamingCommitProfile, SparseBvaFieldSource, StreamingOpenProfile,
        StreamingVirtualPcsError, VirtualPcsError,
    },
};
use crate::utils::{
    fiat_shamir::FiatShamir,
    heterogeneous::{HeteroBlockMat, HeteroMatrixError, MatrixBlock, MatrixDims, PaddedDomain},
    linear,
    sparse::DenseMatCM,
    zktr::{TranSeq, ZkTranSeq},
};

pub(crate) struct CommittedVectorWitness<E: Pairing> {
    pub values: Vec<i32>,
    pub matrix: HeteroBlockMat<i32>,
    pub hiding: E::ScalarField,
    pub commitment: PairingOutput<E>,
    pub tier_one: Vec<E::G1>,
}

impl<E: Pairing> CommittedVectorWitness<E>
where
    E::ScalarField: From<i32>,
{
    pub fn commit(
        srs: &SRS<E>,
        values: Vec<i32>,
        hiding: E::ScalarField,
        bit_width: usize,
    ) -> Result<Self, VectorProofError> {
        if values.is_empty() || !values.len().is_power_of_two() {
            return Err(VectorProofError::InvalidVectorLength(values.len()));
        }
        let width = values.len();
        let block = MatrixBlock::from_columns(MatrixDims::new(width, 1)?, vec![values.clone()])?;
        let matrix = HeteroBlockMat::new(PaddedDomain::new(1, width, 1)?, vec![Some(block)])?;
        let (commitment, tier_one) =
            commit_stacked_i32::<E>(&srs.pc_pp, &matrix, hiding, bit_width)?;
        Ok(Self {
            values,
            matrix,
            hiding,
            commitment,
            tier_one,
        })
    }

    fn value_at(&self, point: &[E::ScalarField]) -> Result<E::ScalarField, VectorProofError> {
        Ok(self.matrix.proj_stacked_lr(point, &[])?)
    }

    fn open_at(
        &self,
        srs: &SRS<E>,
        logical_point: &[E::ScalarField],
        value_commitment: E::G1,
        value_hiding: E::ScalarField,
    ) -> Result<VectorOpeningProof<E>, VectorProofError> {
        let packed_point =
            linear::reshape_points_keep_projection(&(logical_point.to_vec(), Vec::new()));
        let opening = open_stacked_i32::<E>(
            &srs.pc_pp,
            &self.matrix,
            &packed_point.0,
            &packed_point.1,
            E::pairing(value_commitment, srs.h_hat),
            self.commitment,
            &self.tier_one,
            value_hiding,
            self.hiding,
        )?;
        Ok(VectorOpeningProof {
            value_commitment,
            opening,
        })
    }
}

pub struct VectorOpeningProof<E: Pairing> {
    pub value_commitment: E::G1,
    pub opening: PcsTrans<E>,
}

pub struct ZeroLinearProof<E: Pairing> {
    pub evaluations: Vec<VectorOpeningProof<E>>,
}

impl<E: Pairing> ZeroLinearProof<E>
where
    E::ScalarField: From<i32>,
{
    pub(crate) fn prove(
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        tensors: &[&CommittedVectorWitness<E>],
        coefficients: &[E::ScalarField],
    ) -> Result<Self, VectorProofError> {
        check_linear_inputs(tensors, coefficients)?;
        let commitments = tensors
            .iter()
            .map(|tensor| tensor.commitment)
            .collect::<Vec<_>>();
        let point = derive_vector_point::<E>(tag, layer, tensors[0].values.len(), &commitments);
        let rng = &mut ark_std::rand::thread_rng();
        let last = coefficients.len() - 1;
        let last_inverse = coefficients[last]
            .inverse()
            .ok_or(VectorProofError::ZeroFinalCoefficient)?;
        let mut evaluation_hidings = (0..last)
            .map(|_| E::ScalarField::rand(rng))
            .collect::<Vec<_>>();
        let accumulated = coefficients[..last]
            .iter()
            .zip(evaluation_hidings.iter())
            .map(|(coefficient, hiding)| *coefficient * hiding)
            .sum::<E::ScalarField>();
        evaluation_hidings.push(-accumulated * last_inverse);

        let mut evaluations = Vec::with_capacity(tensors.len());
        for (tensor, hiding) in tensors.iter().zip(evaluation_hidings) {
            let value = tensor.value_at(&point)?;
            let value_commitment = srs.zksrs.commit(value, hiding);
            evaluations.push(tensor.open_at(srs, &point, value_commitment, hiding)?);
        }
        Ok(Self { evaluations })
    }

    pub(crate) fn verify(
        &self,
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        width: usize,
        commitments: &[PairingOutput<E>],
        coefficients: &[E::ScalarField],
    ) -> bool {
        if commitments.len() != coefficients.len()
            || self.evaluations.len() != coefficients.len()
            || commitments.is_empty()
            || !width.is_power_of_two()
        {
            return false;
        }
        let combined = self
            .evaluations
            .iter()
            .zip(coefficients)
            .map(|(evaluation, coefficient)| evaluation.value_commitment.mul(coefficient))
            .sum::<E::G1>();
        if !combined.is_zero() {
            return false;
        }
        let point = derive_vector_point::<E>(tag, layer, width, commitments);
        commitments
            .iter()
            .zip(&self.evaluations)
            .all(|(commitment, evaluation)| {
                verify_vector_opening(srs, *commitment, &point, evaluation)
            })
    }
}

pub struct ZeroHadamardProof<E: Pairing> {
    pub transcript: TranSeq<E::ScalarField, E::G1>,
    pub left_opening: VectorOpeningProof<E>,
    pub right_opening: VectorOpeningProof<E>,
}

impl<E: Pairing> ZeroHadamardProof<E>
where
    E::ScalarField: From<i32>,
{
    pub(crate) fn prove(
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        left: &CommittedVectorWitness<E>,
        right: &CommittedVectorWitness<E>,
    ) -> Result<Self, VectorProofError> {
        if left.values.len() != right.values.len() {
            return Err(VectorProofError::VectorLengthMismatch);
        }
        let point = derive_vector_point::<E>(
            tag,
            layer,
            left.values.len(),
            &[left.commitment, right.commitment],
        );
        let left_dense = DenseMatCM::<i32, E::ScalarField>::from_data(vec![left.values.clone()]);
        let right_dense = DenseMatCM::<i32, E::ScalarField>::from_data(vec![right.values.clone()]);
        let protocol = ZkHadamard::<
            i32,
            i32,
            E::ScalarField,
            E::G1,
            DenseMatCM<i32, E::ScalarField>,
            DenseMatCM<i32, E::ScalarField>,
        >::new(E::G1::zero(), (point, Vec::new()));
        let mut transcript = ZkTranSeq::new(&srs.zksrs);
        let (
            left_value_commitment,
            right_value_commitment,
            left_value_hiding,
            right_value_hiding,
            left_point,
            right_point,
        ) = protocol.reduce_prover(
            &srs.zksrs,
            &mut transcript,
            &left_dense,
            &right_dense,
            E::ScalarField::zero(),
        );
        let left_opening =
            left.open_at(srs, &left_point.0, left_value_commitment, left_value_hiding)?;
        let right_opening = right.open_at(
            srs,
            &right_point.0,
            right_value_commitment,
            right_value_hiding,
        )?;
        Ok(Self {
            transcript: transcript.publish_trans(),
            left_opening,
            right_opening,
        })
    }

    pub(crate) fn verify(
        &self,
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        width: usize,
        left_commitment: PairingOutput<E>,
        right_commitment: PairingOutput<E>,
    ) -> bool {
        if width == 0
            || !width.is_power_of_two()
            || self.transcript.pointer != 0
            || self.transcript.len() != hadamard_transcript_len(width)
        {
            return false;
        }
        let point =
            derive_vector_point::<E>(tag, layer, width, &[left_commitment, right_commitment]);
        let protocol = ZkHadamard::<
            i32,
            i32,
            E::ScalarField,
            E::G1,
            DenseMatCM<i32, E::ScalarField>,
            DenseMatCM<i32, E::ScalarField>,
        >::new(E::G1::zero(), (point, Vec::new()));
        let mut transcript = self.transcript.clone();
        let (ok, left_value_commitment, right_value_commitment, left_point, right_point) =
            protocol.verify_as_subprotocol(&srs.zksrs, &mut transcript);
        ok && transcript.check_fiat_shamir()
            && transcript.is_fully_consumed()
            && left_value_commitment == self.left_opening.value_commitment
            && right_value_commitment == self.right_opening.value_commitment
            && verify_vector_opening(srs, left_commitment, &left_point.0, &self.left_opening)
            && verify_vector_opening(srs, right_commitment, &right_point.0, &self.right_opening)
    }
}

pub struct BooleanVectorProof<E: Pairing> {
    pub transcript: TranSeq<E::ScalarField, E::G1>,
    pub opening: VectorOpeningProof<E>,
}

pub struct RangeProof<E: Pairing> {
    pub bit_commitments: Vec<PairingOutput<E>>,
    pub value_linear_opening: VectorOpeningProof<E>,
    pub bit_linear_openings: Vec<VectorOpeningProof<E>>,
    pub boolean_proofs: Vec<BooleanVectorProof<E>>,
}

impl<E: Pairing> RangeProof<E>
where
    E::ScalarField: From<i32>,
{
    pub(crate) fn prove(
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        value: &CommittedVectorWitness<E>,
        bits: usize,
    ) -> Result<Self, VectorProofError> {
        if bits == 0 || bits > 30 {
            return Err(VectorProofError::InvalidRangeBits(bits));
        }
        let limit = 1_u32 << bits;
        if value
            .values
            .iter()
            .any(|entry| *entry < 0 || (*entry as u32) >= limit)
        {
            return Err(VectorProofError::ValueOutsideRange);
        }
        let rng = &mut ark_std::rand::thread_rng();
        let bit_commitment_hidings = (0..bits)
            .map(|_| E::ScalarField::rand(rng))
            .collect::<Vec<_>>();
        let mut bit_commitments = Vec::with_capacity(bits);
        for bit in 0..bits {
            let entries = value
                .values
                .iter()
                .map(|entry| ((*entry as u32 >> bit) & 1) as i32)
                .collect::<Vec<_>>();
            let bit_witness =
                CommittedVectorWitness::commit(srs, entries, bit_commitment_hidings[bit], 1)?;
            bit_commitments.push(bit_witness.commitment);
        }
        let mut point_commitments = vec![value.commitment];
        point_commitments.extend(bit_commitments.iter().copied());
        let point = derive_vector_point::<E>(tag, layer, value.values.len(), &point_commitments);
        let bit_evaluation_hidings = (0..bits)
            .map(|_| E::ScalarField::rand(rng))
            .collect::<Vec<_>>();
        let value_evaluation_hiding = bit_evaluation_hidings
            .iter()
            .enumerate()
            .map(|(bit, hiding)| E::ScalarField::from(1_i32 << bit) * hiding)
            .sum::<E::ScalarField>();
        let value_at_point = value.value_at(&point)?;
        let value_commitment = srs.zksrs.commit(value_at_point, value_evaluation_hiding);
        let value_linear_opening =
            value.open_at(srs, &point, value_commitment, value_evaluation_hiding)?;
        // Recreate one bit plane at a time after deriving the common opening
        // point.  This trades an extra commitment pass for O(width), rather
        // than O(bits * width), live witness memory.
        let mut bit_linear_openings = Vec::with_capacity(bits);
        let mut boolean_proofs = Vec::with_capacity(bits);
        for bit_index in 0..bits {
            let entries = value
                .values
                .iter()
                .map(|entry| ((*entry as u32 >> bit_index) & 1) as i32)
                .collect::<Vec<_>>();
            let bit =
                CommittedVectorWitness::commit(srs, entries, bit_commitment_hidings[bit_index], 1)?;
            debug_assert_eq!(bit.commitment, bit_commitments[bit_index]);
            let evaluation_hiding = bit_evaluation_hidings[bit_index];
            let evaluation = bit.value_at(&point)?;
            let evaluation_commitment = srs.zksrs.commit(evaluation, evaluation_hiding);
            bit_linear_openings.push(bit.open_at(
                srs,
                &point,
                evaluation_commitment,
                evaluation_hiding,
            )?);

            let dense = DenseMatCM::<i32, E::ScalarField>::from_data(vec![bit.values.clone()]);
            let protocol =
                ZkBoolCheck::<i32, E::ScalarField, E::G1, DenseMatCM<i32, E::ScalarField>>::new(
                    bit.values.len(),
                    1,
                );
            let mut transcript = ZkTranSeq::new(&srs.zksrs);
            let (evaluation_commitment, evaluation_hiding, evaluation_point) =
                protocol.reduce_prover(&srs.zksrs, &mut transcript, &dense);
            let opening = bit.open_at(
                srs,
                &evaluation_point.0,
                evaluation_commitment,
                evaluation_hiding,
            )?;
            boolean_proofs.push(BooleanVectorProof {
                transcript: transcript.publish_trans(),
                opening,
            });
        }

        Ok(Self {
            bit_commitments,
            value_linear_opening,
            bit_linear_openings,
            boolean_proofs,
        })
    }

    pub(crate) fn verify(
        &self,
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        width: usize,
        value_commitment: PairingOutput<E>,
        bits: usize,
    ) -> bool {
        if bits == 0
            || bits > 30
            || self.bit_commitments.len() != bits
            || self.bit_linear_openings.len() != bits
            || self.boolean_proofs.len() != bits
        {
            return false;
        }
        let recombined = self
            .bit_linear_openings
            .iter()
            .enumerate()
            .map(|(bit, opening)| {
                opening
                    .value_commitment
                    .mul(E::ScalarField::from(1_i32 << bit))
            })
            .sum::<E::G1>();
        if recombined != self.value_linear_opening.value_commitment {
            return false;
        }
        let mut commitments = vec![value_commitment];
        commitments.extend(self.bit_commitments.iter().copied());
        let point = derive_vector_point::<E>(tag, layer, width, &commitments);
        if !verify_vector_opening(srs, value_commitment, &point, &self.value_linear_opening) {
            return false;
        }
        for ((bit_commitment, linear_opening), boolean_proof) in self
            .bit_commitments
            .iter()
            .zip(&self.bit_linear_openings)
            .zip(&self.boolean_proofs)
        {
            if boolean_proof.transcript.pointer != 0
                || boolean_proof.transcript.len() != boolcheck_transcript_len(width)
            {
                return false;
            }
            if !verify_vector_opening(srs, *bit_commitment, &point, linear_opening) {
                return false;
            }
            let protocol =
                ZkBoolCheck::<i32, E::ScalarField, E::G1, DenseMatCM<i32, E::ScalarField>>::new(
                    width, 1,
                );
            let mut transcript = boolean_proof.transcript.clone();
            let (ok, evaluation_commitment, evaluation_point) =
                protocol.verify_as_subprotocol(&srs.zksrs, &mut transcript);
            if !ok
                || !transcript.check_fiat_shamir()
                || !transcript.is_fully_consumed()
                || evaluation_commitment != boolean_proof.opening.value_commitment
                || !verify_vector_opening(
                    srs,
                    *bit_commitment,
                    &evaluation_point.0,
                    &boolean_proof.opening,
                )
            {
                return false;
            }
        }
        true
    }
}

/// A self-contained bounded-range proof for one physical, power-of-two chunk.
///
/// The prover-only values and commitment randomness are intentionally absent
/// from this object.  Verification therefore uses only the public shape, the
/// commitment carried by the proof, the SRS, and the Fiat-Shamir domain.
pub struct BoundedRangeChunkProof<E: Pairing> {
    pub width: usize,
    pub bits: usize,
    pub value_commitment: PairingOutput<E>,
    pub range_proof: RangeProof<E>,
}

impl<E: Pairing> BoundedRangeChunkProof<E>
where
    E::ScalarField: From<i32>,
{
    pub fn prove(
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        values: Vec<i32>,
        bits: usize,
    ) -> Result<Self, VectorProofError> {
        Self::prove_with_hiding(
            srs,
            layer,
            tag,
            values,
            bits,
            E::ScalarField::rand(&mut ark_std::rand::thread_rng()),
        )
    }

    pub(crate) fn prove_with_hiding(
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        values: Vec<i32>,
        bits: usize,
        value_hiding: E::ScalarField,
    ) -> Result<Self, VectorProofError> {
        if bits == 0 || bits > 30 {
            return Err(VectorProofError::InvalidRangeBits(bits));
        }
        let limit = 1_u32 << bits;
        if values
            .iter()
            .any(|entry| *entry < 0 || (*entry as u32) >= limit)
        {
            return Err(VectorProofError::ValueOutsideRange);
        }
        let width = values.len();
        let value = CommittedVectorWitness::commit(srs, values, value_hiding, bits)?;
        let value_commitment = value.commitment;
        let range_proof = RangeProof::prove(srs, layer, tag, &value, bits)?;
        Ok(Self {
            width,
            bits,
            value_commitment,
            range_proof,
        })
    }

    pub fn verify(&self, srs: &SRS<E>, layer: usize, tag: usize) -> bool {
        self.width != 0
            && self.width.is_power_of_two()
            && self.range_proof.verify(
                srs,
                layer,
                tag,
                self.width,
                self.value_commitment,
                self.bits,
            )
    }

    /// Accounts for the canonical-compressed payload fields in this chunk.
    /// An eventual aggregate proof codec must additionally count its vector
    /// framing and outer envelope.
    pub fn serialized_size_bytes(&self) -> usize {
        let opening_size = |opening: &VectorOpeningProof<E>| {
            opening.value_commitment.serialized_size(Compress::Yes)
                + opening.opening.serialized_size(Compress::Yes)
        };
        2 * std::mem::size_of::<u64>()
            + self.value_commitment.serialized_size(Compress::Yes)
            + self
                .range_proof
                .bit_commitments
                .iter()
                .map(|commitment| commitment.serialized_size(Compress::Yes))
                .sum::<usize>()
            + opening_size(&self.range_proof.value_linear_opening)
            + self
                .range_proof
                .bit_linear_openings
                .iter()
                .map(opening_size)
                .sum::<usize>()
            + self
                .range_proof
                .boolean_proofs
                .iter()
                .map(|proof| {
                    proof.transcript.serialized_size(Compress::Yes) + opening_size(&proof.opening)
                })
                .sum::<usize>()
    }
}

/// One SmartPC opening of a field-valued BVA commitment. The evaluation is
/// Pedersen committed in G1, while the matrix commitment remains in GT.
#[derive(Clone, Debug)]
pub struct BvaVectorOpeningProof<E: Pairing> {
    pub value_commitment: E::G1,
    pub opening: PcsTrans<E>,
}

/// Global Booleanity proof for one bit plane. `bit_minus_one_commitment` is
/// linked to the bit plane at the common range point, and the EO reduction
/// proves their pointwise inner product is zero.
pub struct BvaBooleanProof<E: Pairing> {
    pub bit_minus_one_commitment: PairingOutput<E>,
    pub eo_proof: EoWitnessPairProof<E::ScalarField>,
    pub bit_eo_opening: BvaVectorOpeningProof<E>,
    pub bit_minus_one_eo_opening: BvaVectorOpeningProof<E>,
}

/// A bounded-RAM range proof over one canonical BVA commitment. Bit planes
/// and `bit - 1` planes are replayed from the source one physical vector at a
/// time. Logical BVA padding remains metadata-only in commitments, openings,
/// and EO inner products.
pub struct BvaRangeProof<E: Pairing> {
    pub bit_commitments: Vec<PairingOutput<E>>,
    pub value_linear_opening: BvaVectorOpeningProof<E>,
    pub bit_linear_openings: Vec<BvaVectorOpeningProof<E>>,
    pub bit_minus_one_linear_openings: Vec<BvaVectorOpeningProof<E>>,
    pub boolean_proofs: Vec<BvaBooleanProof<E>>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BvaRangeProfile {
    pub source_passes: usize,
    pub total_visited_entries: usize,
    pub maximum_live_vector_entries: usize,
    pub maximum_live_projection_entries: usize,
    pub materialized_padding_entries: usize,
    pub padding_field_operations: u64,
    pub padding_group_operations: usize,
}

impl BvaRangeProfile {
    fn absorb_commit(&mut self, profile: BvaStreamingCommitProfile) {
        self.source_passes += 1;
        self.total_visited_entries += profile.physical_entries;
        self.materialized_padding_entries += profile.materialized_padding_entries;
        self.padding_group_operations += profile.padding_group_operations;
    }

    fn absorb_open(&mut self, profile: StreamingOpenProfile) {
        self.source_passes += profile.source_passes;
        self.total_visited_entries += profile.physical_entries;
        self.materialized_padding_entries += profile.materialized_projection_padding_entries;
        self.padding_field_operations += profile.padding_projection_field_multiplications as u64;
    }

    fn absorb_eo(&mut self, counters: PaddingWorkCounters, profile: StreamingEoProfile) {
        self.source_passes += profile.source_passes;
        self.total_visited_entries += profile.total_visited_entries;
        self.maximum_live_vector_entries = self
            .maximum_live_vector_entries
            .max(profile.maximum_live_vector_entries);
        self.materialized_padding_entries += profile.materialized_padding_entries;
        self.padding_field_operations += counters.padding_ip_field_ops
            + counters.padding_fold_field_ops
            + profile.padding_field_operations;
    }
}

struct CommittedBvaPlane<E: Pairing> {
    commitment: PairingOutput<E>,
    tier_one: Vec<E::G1>,
    hiding: E::ScalarField,
}

#[derive(Debug)]
enum BvaRangeSourceError<SourceError> {
    Source(SourceError),
    Bva(BvaError),
    ValueOutsideRange,
}

impl<SourceError: Display> Display for BvaRangeSourceError<SourceError> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Source(error) => Display::fmt(error, formatter),
            Self::Bva(error) => Display::fmt(error, formatter),
            Self::ValueOutsideRange => write!(formatter, "BVA value is outside the claimed range"),
        }
    }
}

impl<SourceError> std::error::Error for BvaRangeSourceError<SourceError> where
    SourceError: std::error::Error + 'static
{
}

#[derive(Debug)]
pub enum BvaRangeProofError<SourceError> {
    Bva(BvaError),
    VirtualPcs(VirtualPcsError),
    Source(SourceError),
    InvalidRangeBits(usize),
    ValueOutsideRange,
    RelationMismatch,
}

impl<SourceError: Display> Display for BvaRangeProofError<SourceError> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bva(error) => Display::fmt(error, formatter),
            Self::VirtualPcs(error) => Display::fmt(error, formatter),
            Self::Source(error) => Display::fmt(error, formatter),
            Self::InvalidRangeBits(bits) => write!(formatter, "invalid BVA range width {bits}"),
            Self::ValueOutsideRange => write!(formatter, "BVA value is outside the claimed range"),
            Self::RelationMismatch => write!(formatter, "BVA range relation mismatch"),
        }
    }
}

impl<SourceError> std::error::Error for BvaRangeProofError<SourceError> where
    SourceError: std::error::Error + 'static
{
}

impl<E: Pairing> BvaRangeProof<E>
where
    E::ScalarField: From<i32>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn prove<S: BvaI32Source>(
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        source: &mut S,
        value_commitment: PairingOutput<E>,
        value_tier_one: &[E::G1],
        value_hiding: E::ScalarField,
        bits: usize,
    ) -> Result<(Self, BvaRangeProfile), BvaRangeProofError<S::Error>> {
        if bits == 0 || bits > 30 {
            return Err(BvaRangeProofError::InvalidRangeBits(bits));
        }
        let layout = source.layout().clone();
        let mut profile = BvaRangeProfile {
            maximum_live_vector_entries: layout
                .vector_lengths()
                .iter()
                .copied()
                .max()
                .unwrap_or(0)
                .saturating_mul(2),
            maximum_live_projection_entries: bva_maximum_projection_entries(&layout),
            ..Default::default()
        };
        let rng = &mut ark_std::rand::thread_rng();
        let mut bit_planes = Vec::with_capacity(bits);
        let mut bit_minus_one_planes = Vec::with_capacity(bits);

        for bit in 0..bits {
            let hiding = E::ScalarField::rand(rng);
            let mut adapter = BitPlaneBvaFieldAdapter::<S, E::ScalarField>::new(source, bit, bits);
            let (commitment, tier_one, commit_profile) =
                commit_bva_field_streaming::<E, _>(&srs.pc_pp, &mut adapter, hiding)
                    .map_err(map_bva_range_pcs_error)?;
            profile.absorb_commit(commit_profile);
            bit_planes.push(CommittedBvaPlane {
                commitment,
                tier_one,
                hiding,
            });

            let hiding = E::ScalarField::rand(rng);
            let mut adapter =
                BitMinusOneBvaFieldAdapter::<S, E::ScalarField>::new(source, bit, bits);
            let (commitment, tier_one, commit_profile) =
                commit_bva_field_streaming::<E, _>(&srs.pc_pp, &mut adapter, hiding)
                    .map_err(map_bva_range_pcs_error)?;
            profile.absorb_commit(commit_profile);
            bit_minus_one_planes.push(CommittedBvaPlane {
                commitment,
                tier_one,
                hiding,
            });
        }

        let bit_commitments = bit_planes
            .iter()
            .map(|plane| plane.commitment)
            .collect::<Vec<_>>();
        let bit_minus_one_commitments = bit_minus_one_planes
            .iter()
            .map(|plane| plane.commitment)
            .collect::<Vec<_>>();
        let range_challenges = derive_bva_range_challenges::<E>(
            layer,
            tag,
            &layout,
            value_commitment,
            &bit_commitments,
            &bit_minus_one_commitments,
            bits,
        );
        let range_tbv = TensorBasisVector::global_tensor(&layout, &range_challenges)
            .map_err(BvaRangeProofError::Bva)?;
        let mask_value = (0..layout.vector_lengths().len())
            .try_fold(E::ScalarField::zero(), |sum, vector_index| {
                range_tbv
                    .physical_coefficient_sum(vector_index)
                    .map(|value| sum + value)
            })
            .map_err(BvaRangeProofError::Bva)?;
        let (range_xl, range_xr) = bva_eo_opening_point(layout.logical_len(), &range_challenges)
            .map_err(BvaRangeProofError::VirtualPcs)?;

        let mut value_adapter = I32BvaFieldAdapter::<S, E::ScalarField>::new(source);
        let value_at_point = inner_product_bva_field_streaming(&mut value_adapter, &range_tbv)
            .map_err(map_plain_bva_range_pcs_error)?;
        profile.source_passes += value_at_point.profile.source_passes;
        profile.total_visited_entries += value_at_point.profile.physical_entries;
        profile.maximum_live_vector_entries = profile
            .maximum_live_vector_entries
            .max(value_at_point.profile.maximum_live_vector_entries);
        profile.materialized_padding_entries += value_at_point.profile.materialized_padding_entries;
        profile.padding_field_operations += value_at_point.profile.padding_field_operations;

        let bit_evaluation_hidings = (0..bits)
            .map(|_| E::ScalarField::rand(rng))
            .collect::<Vec<_>>();
        let value_evaluation_hiding = bit_evaluation_hidings
            .iter()
            .enumerate()
            .map(|(bit, hiding)| E::ScalarField::from(1_i32 << bit) * hiding)
            .sum::<E::ScalarField>();
        let value_at_point_commitment = srs
            .zksrs
            .commit(value_at_point.value, value_evaluation_hiding);
        let value_linear_opening = {
            let mut adapter = I32BvaFieldAdapter::<S, E::ScalarField>::new(source);
            let (opening, open_profile) = open_bva_field_streaming::<E, _>(
                &srs.pc_pp,
                &mut adapter,
                &range_xl,
                &range_xr,
                E::pairing(value_at_point_commitment, srs.h_hat),
                value_commitment,
                value_tier_one,
                value_evaluation_hiding,
                value_hiding,
            )
            .map_err(map_plain_bva_range_pcs_error)?;
            profile.absorb_open(open_profile);
            BvaVectorOpeningProof {
                value_commitment: value_at_point_commitment,
                opening,
            }
        };

        let mut bit_linear_openings = Vec::with_capacity(bits);
        let mut bit_minus_one_linear_openings = Vec::with_capacity(bits);
        let mut boolean_proofs = Vec::with_capacity(bits);
        let mut recombined_value = E::ScalarField::zero();

        for bit in 0..bits {
            let mut bit_adapter =
                BitPlaneBvaFieldAdapter::<S, E::ScalarField>::new(source, bit, bits);
            let bit_at_point = inner_product_bva_field_streaming(&mut bit_adapter, &range_tbv)
                .map_err(map_bva_range_pcs_error)?;
            profile.source_passes += bit_at_point.profile.source_passes;
            profile.total_visited_entries += bit_at_point.profile.physical_entries;
            profile.maximum_live_vector_entries = profile
                .maximum_live_vector_entries
                .max(bit_at_point.profile.maximum_live_vector_entries);
            profile.materialized_padding_entries +=
                bit_at_point.profile.materialized_padding_entries;
            profile.padding_field_operations += bit_at_point.profile.padding_field_operations;

            let mut bit_minus_one_adapter =
                BitMinusOneBvaFieldAdapter::<S, E::ScalarField>::new(source, bit, bits);
            let bit_minus_one_at_point =
                inner_product_bva_field_streaming(&mut bit_minus_one_adapter, &range_tbv)
                    .map_err(map_bva_range_pcs_error)?;
            profile.source_passes += bit_minus_one_at_point.profile.source_passes;
            profile.total_visited_entries += bit_minus_one_at_point.profile.physical_entries;
            profile.maximum_live_vector_entries = profile
                .maximum_live_vector_entries
                .max(bit_minus_one_at_point.profile.maximum_live_vector_entries);
            profile.materialized_padding_entries +=
                bit_minus_one_at_point.profile.materialized_padding_entries;
            profile.padding_field_operations +=
                bit_minus_one_at_point.profile.padding_field_operations;

            if bit_at_point.value - bit_minus_one_at_point.value != mask_value {
                return Err(BvaRangeProofError::RelationMismatch);
            }
            recombined_value += E::ScalarField::from(1_i32 << bit) * bit_at_point.value;
            let evaluation_hiding = bit_evaluation_hidings[bit];
            let bit_value_commitment = srs.zksrs.commit(bit_at_point.value, evaluation_hiding);
            let bit_minus_one_value_commitment = srs
                .zksrs
                .commit(bit_minus_one_at_point.value, evaluation_hiding);

            let bit_linear_opening = {
                let mut adapter =
                    BitPlaneBvaFieldAdapter::<S, E::ScalarField>::new(source, bit, bits);
                let (opening, open_profile) = open_bva_field_streaming::<E, _>(
                    &srs.pc_pp,
                    &mut adapter,
                    &range_xl,
                    &range_xr,
                    E::pairing(bit_value_commitment, srs.h_hat),
                    bit_planes[bit].commitment,
                    &bit_planes[bit].tier_one,
                    evaluation_hiding,
                    bit_planes[bit].hiding,
                )
                .map_err(map_bva_range_pcs_error)?;
                profile.absorb_open(open_profile);
                BvaVectorOpeningProof {
                    value_commitment: bit_value_commitment,
                    opening,
                }
            };
            let bit_minus_one_linear_opening = {
                let mut adapter =
                    BitMinusOneBvaFieldAdapter::<S, E::ScalarField>::new(source, bit, bits);
                let (opening, open_profile) = open_bva_field_streaming::<E, _>(
                    &srs.pc_pp,
                    &mut adapter,
                    &range_xl,
                    &range_xr,
                    E::pairing(bit_minus_one_value_commitment, srs.h_hat),
                    bit_minus_one_planes[bit].commitment,
                    &bit_minus_one_planes[bit].tier_one,
                    evaluation_hiding,
                    bit_minus_one_planes[bit].hiding,
                )
                .map_err(map_bva_range_pcs_error)?;
                profile.absorb_open(open_profile);
                BvaVectorOpeningProof {
                    value_commitment: bit_minus_one_value_commitment,
                    opening,
                }
            };

            let eo_binding = derive_bva_boolean_binding::<E>(
                layer,
                tag,
                &layout,
                value_commitment,
                bit_planes[bit].commitment,
                bit_minus_one_planes[bit].commitment,
                bits,
                bit,
            );
            let boolean_challenges = derive_bva_boolean_challenges::<E>(
                layer,
                tag,
                &layout,
                value_commitment,
                bit_planes[bit].commitment,
                bit_minus_one_planes[bit].commitment,
                bits,
                bit,
            );
            let boolean_tbv = TensorBasisVector::global_tensor(&layout, &boolean_challenges)
                .map_err(BvaRangeProofError::Bva)?;
            let mut pair_adapter =
                BitPairBvaFieldAdapter::<S, E::ScalarField>::new(source, bit, bits, boolean_tbv);
            let (eo_proof, counters, eo_profile) = EoWitnessPairProof::prove_streaming_bound(
                &mut pair_adapter,
                E::ScalarField::zero(),
                eo_binding,
            )
            .map_err(map_bva_range_eo_error)?;
            profile.absorb_eo(counters, eo_profile);
            let eo_verification =
                eo_proof.verify_bound(layout.logical_len(), E::ScalarField::zero(), eo_binding);
            if !eo_verification.accepted {
                return Err(BvaRangeProofError::RelationMismatch);
            }

            let (bit_eo_xl, bit_eo_xr) =
                bva_eo_opening_point(layout.logical_len(), &eo_verification.challenges_inv)
                    .map_err(BvaRangeProofError::VirtualPcs)?;
            let bit_eo_value_commitment =
                srs.zksrs.commit(eo_proof.a_reduced, E::ScalarField::zero());
            let bit_eo_opening = {
                let mut adapter =
                    BitPlaneBvaFieldAdapter::<S, E::ScalarField>::new(source, bit, bits);
                let (opening, open_profile) = open_bva_field_streaming::<E, _>(
                    &srs.pc_pp,
                    &mut adapter,
                    &bit_eo_xl,
                    &bit_eo_xr,
                    E::pairing(bit_eo_value_commitment, srs.h_hat),
                    bit_planes[bit].commitment,
                    &bit_planes[bit].tier_one,
                    E::ScalarField::zero(),
                    bit_planes[bit].hiding,
                )
                .map_err(map_bva_range_pcs_error)?;
                profile.absorb_open(open_profile);
                BvaVectorOpeningProof {
                    value_commitment: bit_eo_value_commitment,
                    opening,
                }
            };

            let weighted_right_challenges = boolean_challenges
                .iter()
                .zip(&eo_verification.challenges)
                .map(|(boolean, eo)| *boolean * eo)
                .collect::<Vec<_>>();
            let (bit_minus_one_eo_xl, bit_minus_one_eo_xr) =
                bva_eo_opening_point(layout.logical_len(), &weighted_right_challenges)
                    .map_err(BvaRangeProofError::VirtualPcs)?;
            let bit_minus_one_eo_value_commitment =
                srs.zksrs.commit(eo_proof.b_reduced, E::ScalarField::zero());
            let bit_minus_one_eo_opening = {
                let mut adapter =
                    BitMinusOneBvaFieldAdapter::<S, E::ScalarField>::new(source, bit, bits);
                let (opening, open_profile) = open_bva_field_streaming::<E, _>(
                    &srs.pc_pp,
                    &mut adapter,
                    &bit_minus_one_eo_xl,
                    &bit_minus_one_eo_xr,
                    E::pairing(bit_minus_one_eo_value_commitment, srs.h_hat),
                    bit_minus_one_planes[bit].commitment,
                    &bit_minus_one_planes[bit].tier_one,
                    E::ScalarField::zero(),
                    bit_minus_one_planes[bit].hiding,
                )
                .map_err(map_bva_range_pcs_error)?;
                profile.absorb_open(open_profile);
                BvaVectorOpeningProof {
                    value_commitment: bit_minus_one_eo_value_commitment,
                    opening,
                }
            };

            bit_linear_openings.push(bit_linear_opening);
            bit_minus_one_linear_openings.push(bit_minus_one_linear_opening);
            boolean_proofs.push(BvaBooleanProof {
                bit_minus_one_commitment: bit_minus_one_planes[bit].commitment,
                eo_proof,
                bit_eo_opening,
                bit_minus_one_eo_opening,
            });
        }

        if recombined_value != value_at_point.value {
            return Err(BvaRangeProofError::RelationMismatch);
        }

        Ok((
            Self {
                bit_commitments,
                value_linear_opening,
                bit_linear_openings,
                bit_minus_one_linear_openings,
                boolean_proofs,
            },
            profile,
        ))
    }

    pub fn verify(
        &self,
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        layout: &BvaLayout,
        value_commitment: PairingOutput<E>,
        bits: usize,
    ) -> bool {
        if bits == 0
            || bits > 30
            || self.bit_commitments.len() != bits
            || self.bit_linear_openings.len() != bits
            || self.bit_minus_one_linear_openings.len() != bits
            || self.boolean_proofs.len() != bits
        {
            return false;
        }
        let bit_minus_one_commitments = self
            .boolean_proofs
            .iter()
            .map(|proof| proof.bit_minus_one_commitment)
            .collect::<Vec<_>>();
        let range_challenges = derive_bva_range_challenges::<E>(
            layer,
            tag,
            layout,
            value_commitment,
            &self.bit_commitments,
            &bit_minus_one_commitments,
            bits,
        );
        let Ok(range_tbv) = TensorBasisVector::global_tensor(layout, &range_challenges) else {
            return false;
        };
        let Some(mask_value) = (0..layout.vector_lengths().len()).try_fold(
            E::ScalarField::zero(),
            |sum, vector_index| {
                range_tbv
                    .physical_coefficient_sum(vector_index)
                    .ok()
                    .map(|value| sum + value)
            },
        ) else {
            return false;
        };
        let mask_commitment = srs.zksrs.commit(mask_value, E::ScalarField::zero());
        if !verify_bva_vector_opening(
            srs,
            value_commitment,
            layout,
            &range_challenges,
            &self.value_linear_opening,
        ) {
            return false;
        }
        let recombined = self
            .bit_linear_openings
            .iter()
            .enumerate()
            .map(|(bit, opening)| {
                opening
                    .value_commitment
                    .mul(E::ScalarField::from(1_i32 << bit))
            })
            .sum::<E::G1>();
        if recombined != self.value_linear_opening.value_commitment {
            return false;
        }

        for bit in 0..bits {
            if self.bit_linear_openings[bit].value_commitment
                - self.bit_minus_one_linear_openings[bit].value_commitment
                != mask_commitment
                || !verify_bva_vector_opening(
                    srs,
                    self.bit_commitments[bit],
                    layout,
                    &range_challenges,
                    &self.bit_linear_openings[bit],
                )
                || !verify_bva_vector_opening(
                    srs,
                    bit_minus_one_commitments[bit],
                    layout,
                    &range_challenges,
                    &self.bit_minus_one_linear_openings[bit],
                )
            {
                return false;
            }
            let eo_binding = derive_bva_boolean_binding::<E>(
                layer,
                tag,
                layout,
                value_commitment,
                self.bit_commitments[bit],
                bit_minus_one_commitments[bit],
                bits,
                bit,
            );
            let boolean = &self.boolean_proofs[bit];
            let boolean_challenges = derive_bva_boolean_challenges::<E>(
                layer,
                tag,
                layout,
                value_commitment,
                self.bit_commitments[bit],
                bit_minus_one_commitments[bit],
                bits,
                bit,
            );
            let eo_verification = boolean.eo_proof.verify_bound(
                layout.logical_len(),
                E::ScalarField::zero(),
                eo_binding,
            );
            if !eo_verification.accepted
                || boolean.bit_eo_opening.value_commitment
                    != srs
                        .zksrs
                        .commit(boolean.eo_proof.a_reduced, E::ScalarField::zero())
                || boolean.bit_minus_one_eo_opening.value_commitment
                    != srs
                        .zksrs
                        .commit(boolean.eo_proof.b_reduced, E::ScalarField::zero())
                || !verify_bva_vector_opening(
                    srs,
                    self.bit_commitments[bit],
                    layout,
                    &eo_verification.challenges_inv,
                    &boolean.bit_eo_opening,
                )
                || !verify_bva_vector_opening(
                    srs,
                    bit_minus_one_commitments[bit],
                    layout,
                    &boolean_challenges
                        .iter()
                        .zip(&eo_verification.challenges)
                        .map(|(boolean, eo)| *boolean * eo)
                        .collect::<Vec<_>>(),
                    &boolean.bit_minus_one_eo_opening,
                )
            {
                return false;
            }
        }
        true
    }

    pub fn serialized_size_bytes(&self) -> usize {
        let opening_size = |opening: &BvaVectorOpeningProof<E>| {
            opening.value_commitment.serialized_size(Compress::Yes)
                + opening.opening.serialized_size(Compress::Yes)
        };
        self.bit_commitments
            .iter()
            .map(|commitment| commitment.serialized_size(Compress::Yes))
            .sum::<usize>()
            + opening_size(&self.value_linear_opening)
            + self
                .bit_linear_openings
                .iter()
                .map(opening_size)
                .sum::<usize>()
            + self
                .bit_minus_one_linear_openings
                .iter()
                .map(opening_size)
                .sum::<usize>()
            + self
                .boolean_proofs
                .iter()
                .map(|proof| {
                    proof
                        .bit_minus_one_commitment
                        .serialized_size(Compress::Yes)
                        + proof.eo_proof.serialized_size(Compress::Yes)
                        + opening_size(&proof.bit_eo_opening)
                        + opening_size(&proof.bit_minus_one_eo_opening)
                })
                .sum::<usize>()
    }
}

/// A rewindable sparse source for nonnegative integer BVA entries. Omitted
/// physical coordinates and all logical padding are the value zero.
pub trait SparseBvaU64Source {
    type Error;

    fn layout(&self) -> &BvaLayout;

    fn for_each_nonzero_u64(
        &mut self,
        visit: &mut dyn FnMut(usize, usize, u64),
    ) -> Result<(), Self::Error>;
}

/// Booleanity of one sparse bit plane, expressed as the random-MLE identity
/// `b * b = b`. This avoids the dense `b - 1` vector that would otherwise put
/// `-1` at every omitted public-table row.
pub struct SparseBvaBooleanProof<E: Pairing> {
    pub bit_boolean_value: E::ScalarField,
    pub bit_boolean_opening: BvaVectorOpeningProof<E>,
    pub eo_proof: EoWitnessPairProof<E::ScalarField>,
    pub bit_eo_opening: BvaVectorOpeningProof<E>,
    pub bit_weighted_eo_opening: BvaVectorOpeningProof<E>,
}

/// Range proof for a sparse nonnegative BVA commitment. Bit commitments,
/// openings, and Boolean EO reductions enumerate only nonzero count entries;
/// zero table rows and BVA padding remain metadata-only.
pub struct SparseBvaRangeProof<E: Pairing> {
    pub bit_commitments: Vec<PairingOutput<E>>,
    pub value_linear_opening: BvaVectorOpeningProof<E>,
    pub bit_linear_openings: Vec<BvaVectorOpeningProof<E>>,
    pub boolean_proofs: Vec<SparseBvaBooleanProof<E>>,
}

impl<E: Pairing> SparseBvaRangeProof<E>
where
    E::ScalarField: From<i32>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn prove<S: SparseBvaU64Source>(
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        source: &mut S,
        value_commitment: PairingOutput<E>,
        value_tier_one: &[E::G1],
        value_hiding: E::ScalarField,
        bits: usize,
    ) -> Result<(Self, BvaRangeProfile), BvaRangeProofError<S::Error>> {
        if bits == 0 || bits > 30 {
            return Err(BvaRangeProofError::InvalidRangeBits(bits));
        }
        let layout = source.layout().clone();
        let mut profile = BvaRangeProfile {
            maximum_live_projection_entries: bva_maximum_projection_entries(&layout),
            ..Default::default()
        };
        let rng = &mut ark_std::rand::thread_rng();
        let mut bit_planes = Vec::with_capacity(bits);
        for bit in 0..bits {
            let hiding = E::ScalarField::rand(rng);
            let mut adapter =
                SparseBitPlaneBvaFieldAdapter::<S, E::ScalarField>::new(source, bit, bits);
            let (commitment, tier_one, commit_profile) =
                commit_sparse_bva_field_streaming::<E, _>(&srs.pc_pp, &mut adapter, hiding)
                    .map_err(map_bva_range_pcs_error)?;
            profile.source_passes += 1;
            profile.total_visited_entries += commit_profile.nonzero_entries;
            profile.maximum_live_vector_entries = profile
                .maximum_live_vector_entries
                .max(commit_profile.nonzero_entries);
            profile.materialized_padding_entries += commit_profile.materialized_padding_entries;
            profile.padding_group_operations += commit_profile.padding_group_operations;
            bit_planes.push(CommittedBvaPlane {
                commitment,
                tier_one,
                hiding,
            });
        }
        let bit_commitments = bit_planes
            .iter()
            .map(|plane| plane.commitment)
            .collect::<Vec<_>>();
        let range_challenges = derive_sparse_bva_range_challenges::<E>(
            layer,
            tag,
            &layout,
            value_commitment,
            &bit_commitments,
            bits,
        );
        let range_tbv = TensorBasisVector::global_tensor(&layout, &range_challenges)
            .map_err(BvaRangeProofError::Bva)?;
        let (range_xl, range_xr) = bva_eo_opening_point(layout.logical_len(), &range_challenges)
            .map_err(BvaRangeProofError::VirtualPcs)?;

        let value_at_point = {
            let mut adapter = SparseU64BvaFieldAdapter::<S, E::ScalarField>::new(source);
            inner_product_sparse_bva_field_streaming(&mut adapter, &range_tbv)
                .map_err(map_bva_range_pcs_error)?
        };
        profile.source_passes += value_at_point.profile.source_passes;
        profile.total_visited_entries += value_at_point.profile.physical_entries;
        profile.maximum_live_vector_entries = profile
            .maximum_live_vector_entries
            .max(value_at_point.profile.maximum_live_vector_entries);

        let bit_evaluation_hidings = (0..bits)
            .map(|_| E::ScalarField::rand(rng))
            .collect::<Vec<_>>();
        let value_evaluation_hiding = bit_evaluation_hidings
            .iter()
            .enumerate()
            .map(|(bit, hiding)| E::ScalarField::from(1_i32 << bit) * hiding)
            .sum::<E::ScalarField>();
        let value_at_point_commitment = srs
            .zksrs
            .commit(value_at_point.value, value_evaluation_hiding);
        let value_linear_opening = {
            let mut adapter = SparseU64BvaFieldAdapter::<S, E::ScalarField>::new(source);
            let (opening, open_profile) = open_sparse_bva_field_streaming::<E, _>(
                &srs.pc_pp,
                &mut adapter,
                &range_xl,
                &range_xr,
                E::pairing(value_at_point_commitment, srs.h_hat),
                value_commitment,
                value_tier_one,
                value_evaluation_hiding,
                value_hiding,
            )
            .map_err(map_bva_range_pcs_error)?;
            profile.absorb_open(open_profile);
            BvaVectorOpeningProof {
                value_commitment: value_at_point_commitment,
                opening,
            }
        };

        let mut bit_linear_openings = Vec::with_capacity(bits);
        let mut boolean_proofs = Vec::with_capacity(bits);
        let mut recombined_value = E::ScalarField::zero();
        for bit in 0..bits {
            let mut bit_adapter =
                SparseBitPlaneBvaFieldAdapter::<S, E::ScalarField>::new(source, bit, bits);
            let bit_at_point =
                inner_product_sparse_bva_field_streaming(&mut bit_adapter, &range_tbv)
                    .map_err(map_bva_range_pcs_error)?;
            profile.source_passes += bit_at_point.profile.source_passes;
            profile.total_visited_entries += bit_at_point.profile.physical_entries;
            profile.maximum_live_vector_entries = profile
                .maximum_live_vector_entries
                .max(bit_at_point.profile.maximum_live_vector_entries);
            recombined_value += E::ScalarField::from(1_i32 << bit) * bit_at_point.value;
            let bit_value_commitment = srs
                .zksrs
                .commit(bit_at_point.value, bit_evaluation_hidings[bit]);
            let bit_linear_opening = {
                let mut adapter =
                    SparseBitPlaneBvaFieldAdapter::<S, E::ScalarField>::new(source, bit, bits);
                let (opening, open_profile) = open_sparse_bva_field_streaming::<E, _>(
                    &srs.pc_pp,
                    &mut adapter,
                    &range_xl,
                    &range_xr,
                    E::pairing(bit_value_commitment, srs.h_hat),
                    bit_planes[bit].commitment,
                    &bit_planes[bit].tier_one,
                    bit_evaluation_hidings[bit],
                    bit_planes[bit].hiding,
                )
                .map_err(map_bva_range_pcs_error)?;
                profile.absorb_open(open_profile);
                BvaVectorOpeningProof {
                    value_commitment: bit_value_commitment,
                    opening,
                }
            };

            let boolean_challenges = derive_sparse_bva_boolean_challenges::<E>(
                layer,
                tag,
                &layout,
                value_commitment,
                bit_planes[bit].commitment,
                bits,
                bit,
            );
            let boolean_tbv = TensorBasisVector::global_tensor(&layout, &boolean_challenges)
                .map_err(BvaRangeProofError::Bva)?;
            let bit_boolean_value = {
                let mut adapter =
                    SparseBitPlaneBvaFieldAdapter::<S, E::ScalarField>::new(source, bit, bits);
                let evaluation =
                    inner_product_sparse_bva_field_streaming(&mut adapter, &boolean_tbv)
                        .map_err(map_bva_range_pcs_error)?;
                profile.source_passes += evaluation.profile.source_passes;
                profile.total_visited_entries += evaluation.profile.physical_entries;
                evaluation.value
            };
            let bit_boolean_value_commitment =
                srs.zksrs.commit(bit_boolean_value, E::ScalarField::zero());
            let bit_boolean_opening = {
                let (xl, xr) = bva_eo_opening_point(layout.logical_len(), &boolean_challenges)
                    .map_err(BvaRangeProofError::VirtualPcs)?;
                let mut adapter =
                    SparseBitPlaneBvaFieldAdapter::<S, E::ScalarField>::new(source, bit, bits);
                let (opening, open_profile) = open_sparse_bva_field_streaming::<E, _>(
                    &srs.pc_pp,
                    &mut adapter,
                    &xl,
                    &xr,
                    E::pairing(bit_boolean_value_commitment, srs.h_hat),
                    bit_planes[bit].commitment,
                    &bit_planes[bit].tier_one,
                    E::ScalarField::zero(),
                    bit_planes[bit].hiding,
                )
                .map_err(map_bva_range_pcs_error)?;
                profile.absorb_open(open_profile);
                BvaVectorOpeningProof {
                    value_commitment: bit_boolean_value_commitment,
                    opening,
                }
            };

            let eo_binding = derive_sparse_bva_boolean_binding::<E>(
                layer,
                tag,
                &layout,
                value_commitment,
                bit_planes[bit].commitment,
                bits,
                bit,
            );
            let mut pair_adapter = SparseBitPairBvaFieldAdapter::<S, E::ScalarField>::new(
                source,
                bit,
                bits,
                boolean_tbv,
            );
            let (eo_proof, counters, eo_profile) =
                EoWitnessPairProof::prove_sparse_streaming_bound(
                    &mut pair_adapter,
                    bit_boolean_value,
                    eo_binding,
                )
                .map_err(map_bva_range_eo_error)?;
            profile.absorb_eo(counters, eo_profile);
            let eo_verification =
                eo_proof.verify_bound(layout.logical_len(), bit_boolean_value, eo_binding);
            if !eo_verification.accepted {
                return Err(BvaRangeProofError::RelationMismatch);
            }

            let bit_eo_opening = {
                let (xl, xr) =
                    bva_eo_opening_point(layout.logical_len(), &eo_verification.challenges_inv)
                        .map_err(BvaRangeProofError::VirtualPcs)?;
                let value_commitment = srs.zksrs.commit(eo_proof.a_reduced, E::ScalarField::zero());
                let mut adapter =
                    SparseBitPlaneBvaFieldAdapter::<S, E::ScalarField>::new(source, bit, bits);
                let (opening, open_profile) = open_sparse_bva_field_streaming::<E, _>(
                    &srs.pc_pp,
                    &mut adapter,
                    &xl,
                    &xr,
                    E::pairing(value_commitment, srs.h_hat),
                    bit_planes[bit].commitment,
                    &bit_planes[bit].tier_one,
                    E::ScalarField::zero(),
                    bit_planes[bit].hiding,
                )
                .map_err(map_bva_range_pcs_error)?;
                profile.absorb_open(open_profile);
                BvaVectorOpeningProof {
                    value_commitment,
                    opening,
                }
            };
            let weighted_right_challenges = boolean_challenges
                .iter()
                .zip(&eo_verification.challenges)
                .map(|(boolean, eo)| *boolean * eo)
                .collect::<Vec<_>>();
            let bit_weighted_eo_opening = {
                let (xl, xr) =
                    bva_eo_opening_point(layout.logical_len(), &weighted_right_challenges)
                        .map_err(BvaRangeProofError::VirtualPcs)?;
                let value_commitment = srs.zksrs.commit(eo_proof.b_reduced, E::ScalarField::zero());
                let mut adapter =
                    SparseBitPlaneBvaFieldAdapter::<S, E::ScalarField>::new(source, bit, bits);
                let (opening, open_profile) = open_sparse_bva_field_streaming::<E, _>(
                    &srs.pc_pp,
                    &mut adapter,
                    &xl,
                    &xr,
                    E::pairing(value_commitment, srs.h_hat),
                    bit_planes[bit].commitment,
                    &bit_planes[bit].tier_one,
                    E::ScalarField::zero(),
                    bit_planes[bit].hiding,
                )
                .map_err(map_bva_range_pcs_error)?;
                profile.absorb_open(open_profile);
                BvaVectorOpeningProof {
                    value_commitment,
                    opening,
                }
            };
            bit_linear_openings.push(bit_linear_opening);
            boolean_proofs.push(SparseBvaBooleanProof {
                bit_boolean_value,
                bit_boolean_opening,
                eo_proof,
                bit_eo_opening,
                bit_weighted_eo_opening,
            });
        }
        if recombined_value != value_at_point.value {
            return Err(BvaRangeProofError::RelationMismatch);
        }
        Ok((
            Self {
                bit_commitments,
                value_linear_opening,
                bit_linear_openings,
                boolean_proofs,
            },
            profile,
        ))
    }

    pub fn verify(
        &self,
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        layout: &BvaLayout,
        value_commitment: PairingOutput<E>,
        bits: usize,
    ) -> bool {
        if bits == 0
            || bits > 30
            || self.bit_commitments.len() != bits
            || self.bit_linear_openings.len() != bits
            || self.boolean_proofs.len() != bits
        {
            return false;
        }
        let range_challenges = derive_sparse_bva_range_challenges::<E>(
            layer,
            tag,
            layout,
            value_commitment,
            &self.bit_commitments,
            bits,
        );
        if !verify_bva_vector_opening(
            srs,
            value_commitment,
            layout,
            &range_challenges,
            &self.value_linear_opening,
        ) {
            return false;
        }
        let recombined = self
            .bit_linear_openings
            .iter()
            .enumerate()
            .map(|(bit, opening)| {
                opening
                    .value_commitment
                    .mul(E::ScalarField::from(1_i32 << bit))
            })
            .sum::<E::G1>();
        if recombined != self.value_linear_opening.value_commitment {
            return false;
        }
        for bit in 0..bits {
            if !verify_bva_vector_opening(
                srs,
                self.bit_commitments[bit],
                layout,
                &range_challenges,
                &self.bit_linear_openings[bit],
            ) {
                return false;
            }
            let boolean = &self.boolean_proofs[bit];
            let boolean_challenges = derive_sparse_bva_boolean_challenges::<E>(
                layer,
                tag,
                layout,
                value_commitment,
                self.bit_commitments[bit],
                bits,
                bit,
            );
            if boolean.bit_boolean_opening.value_commitment
                != srs
                    .zksrs
                    .commit(boolean.bit_boolean_value, E::ScalarField::zero())
                || !verify_bva_vector_opening(
                    srs,
                    self.bit_commitments[bit],
                    layout,
                    &boolean_challenges,
                    &boolean.bit_boolean_opening,
                )
            {
                return false;
            }
            let eo_binding = derive_sparse_bva_boolean_binding::<E>(
                layer,
                tag,
                layout,
                value_commitment,
                self.bit_commitments[bit],
                bits,
                bit,
            );
            let eo_verification = boolean.eo_proof.verify_bound(
                layout.logical_len(),
                boolean.bit_boolean_value,
                eo_binding,
            );
            if !eo_verification.accepted
                || boolean.bit_eo_opening.value_commitment
                    != srs
                        .zksrs
                        .commit(boolean.eo_proof.a_reduced, E::ScalarField::zero())
                || boolean.bit_weighted_eo_opening.value_commitment
                    != srs
                        .zksrs
                        .commit(boolean.eo_proof.b_reduced, E::ScalarField::zero())
                || !verify_bva_vector_opening(
                    srs,
                    self.bit_commitments[bit],
                    layout,
                    &eo_verification.challenges_inv,
                    &boolean.bit_eo_opening,
                )
                || !verify_bva_vector_opening(
                    srs,
                    self.bit_commitments[bit],
                    layout,
                    &boolean_challenges
                        .iter()
                        .zip(&eo_verification.challenges)
                        .map(|(boolean, eo)| *boolean * eo)
                        .collect::<Vec<_>>(),
                    &boolean.bit_weighted_eo_opening,
                )
            {
                return false;
            }
        }
        true
    }

    pub fn serialized_size_bytes(&self) -> usize {
        let opening_size = |opening: &BvaVectorOpeningProof<E>| {
            opening.value_commitment.serialized_size(Compress::Yes)
                + opening.opening.serialized_size(Compress::Yes)
        };
        self.bit_commitments
            .iter()
            .map(|commitment| commitment.serialized_size(Compress::Yes))
            .sum::<usize>()
            + opening_size(&self.value_linear_opening)
            + self
                .bit_linear_openings
                .iter()
                .map(opening_size)
                .sum::<usize>()
            + self
                .boolean_proofs
                .iter()
                .map(|proof| {
                    proof.bit_boolean_value.serialized_size(Compress::Yes)
                        + opening_size(&proof.bit_boolean_opening)
                        + proof.eo_proof.serialized_size(Compress::Yes)
                        + opening_size(&proof.bit_eo_opening)
                        + opening_size(&proof.bit_weighted_eo_opening)
                })
                .sum::<usize>()
    }
}

/// Exact sparse upper-bound proof. For every nonzero value `m`, the prover
/// commits a hidden support bit `s` and nonnegative slack `d` satisfying
/// `m + d = maximum * s`; range proofs for `m`, `d`, and `s` then establish
/// `0 <= m <= maximum` without filling zero table rows with `maximum`.
pub struct SparseBvaUpperBoundProof<E: Pairing> {
    pub slack_commitment: PairingOutput<E>,
    pub support_commitment: PairingOutput<E>,
    pub value_range_proof: SparseBvaRangeProof<E>,
    pub slack_range_proof: SparseBvaRangeProof<E>,
    pub support_range_proof: SparseBvaRangeProof<E>,
    pub value_bound_opening: BvaVectorOpeningProof<E>,
    pub slack_bound_opening: BvaVectorOpeningProof<E>,
    pub support_bound_opening: BvaVectorOpeningProof<E>,
}

impl<E: Pairing> SparseBvaUpperBoundProof<E>
where
    E::ScalarField: From<i32> + From<u64>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn prove<S: SparseBvaU64Source>(
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        source: &mut S,
        value_commitment: PairingOutput<E>,
        value_tier_one: &[E::G1],
        value_hiding: E::ScalarField,
        maximum: u64,
    ) -> Result<(Self, BvaRangeProfile), BvaRangeProofError<S::Error>> {
        let bits = sparse_upper_bound_bits(maximum)?;
        let layout = source.layout().clone();
        let mut outside_range = false;
        source
            .for_each_nonzero_u64(&mut |_, _, value| {
                outside_range |= value == 0 || value > maximum;
            })
            .map_err(BvaRangeProofError::Source)?;
        if outside_range {
            return Err(BvaRangeProofError::ValueOutsideRange);
        }

        let rng = &mut ark_std::rand::thread_rng();
        let slack_hiding = E::ScalarField::rand(rng);
        let support_hiding = E::ScalarField::rand(rng);
        let (slack_commitment, slack_tier_one, slack_commit_profile) = {
            let mut adapter = SparseSlackU64Adapter::new(source, maximum);
            commit_sparse_bva_field_streaming::<E, _>(&srs.pc_pp, &mut adapter, slack_hiding)
                .map_err(map_plain_bva_range_pcs_error)?
        };
        let (support_commitment, support_tier_one, support_commit_profile) = {
            let mut adapter = SparseSupportU64Adapter::new(source);
            commit_sparse_bva_field_streaming::<E, _>(&srs.pc_pp, &mut adapter, support_hiding)
                .map_err(map_plain_bva_range_pcs_error)?
        };
        let mut profile = BvaRangeProfile {
            source_passes: 3,
            total_visited_entries: slack_commit_profile.nonzero_entries
                + support_commit_profile.nonzero_entries,
            maximum_live_vector_entries: slack_commit_profile
                .nonzero_entries
                .max(support_commit_profile.nonzero_entries),
            materialized_padding_entries: slack_commit_profile.materialized_padding_entries
                + support_commit_profile.materialized_padding_entries,
            padding_group_operations: slack_commit_profile.padding_group_operations
                + support_commit_profile.padding_group_operations,
            ..Default::default()
        };

        let (value_range_proof, value_range_profile) = SparseBvaRangeProof::prove(
            srs,
            layer,
            sparse_upper_bound_range_tag(tag, 0),
            source,
            value_commitment,
            value_tier_one,
            value_hiding,
            bits,
        )?;
        absorb_bva_range_profile(&mut profile, value_range_profile);
        let (slack_range_proof, slack_range_profile) = {
            let mut adapter = SparseSlackU64Adapter::new(source, maximum);
            SparseBvaRangeProof::prove(
                srs,
                layer,
                sparse_upper_bound_range_tag(tag, 1),
                &mut adapter,
                slack_commitment,
                &slack_tier_one,
                slack_hiding,
                bits,
            )?
        };
        absorb_bva_range_profile(&mut profile, slack_range_profile);
        let (support_range_proof, support_range_profile) = {
            let mut adapter = SparseSupportU64Adapter::new(source);
            SparseBvaRangeProof::prove(
                srs,
                layer,
                sparse_upper_bound_range_tag(tag, 2),
                &mut adapter,
                support_commitment,
                &support_tier_one,
                support_hiding,
                1,
            )?
        };
        absorb_bva_range_profile(&mut profile, support_range_profile);

        let bound_challenges = derive_sparse_upper_bound_challenges::<E>(
            layer,
            tag,
            &layout,
            value_commitment,
            slack_commitment,
            support_commitment,
            maximum,
        );
        let bound_tbv = TensorBasisVector::global_tensor(&layout, &bound_challenges)
            .map_err(BvaRangeProofError::Bva)?;
        let (bound_xl, bound_xr) = bva_eo_opening_point(layout.logical_len(), &bound_challenges)
            .map_err(BvaRangeProofError::VirtualPcs)?;
        let value_at_bound = {
            let mut adapter = SparseU64BvaFieldAdapter::<S, E::ScalarField>::new(source);
            inner_product_sparse_bva_field_streaming(&mut adapter, &bound_tbv)
                .map_err(map_bva_range_pcs_error)?
                .value
        };
        let slack_at_bound = {
            let mut adapter = SparseSlackU64Adapter::new(source, maximum);
            inner_product_sparse_bva_field_streaming(&mut adapter, &bound_tbv)
                .map_err(map_plain_bva_range_pcs_error)?
                .value
        };
        let support_at_bound = {
            let mut adapter = SparseSupportU64Adapter::new(source);
            inner_product_sparse_bva_field_streaming(&mut adapter, &bound_tbv)
                .map_err(map_plain_bva_range_pcs_error)?
                .value
        };
        let maximum_field = E::ScalarField::from(maximum);
        if value_at_bound + slack_at_bound != maximum_field * support_at_bound {
            return Err(BvaRangeProofError::RelationMismatch);
        }
        profile.source_passes += 3;
        profile.total_visited_entries +=
            slack_commit_profile.nonzero_entries + support_commit_profile.nonzero_entries * 2;

        let value_evaluation_hiding = E::ScalarField::rand(rng);
        let slack_evaluation_hiding = E::ScalarField::rand(rng);
        let support_evaluation_hiding = (value_evaluation_hiding + slack_evaluation_hiding)
            * maximum_field
                .inverse()
                .ok_or(BvaRangeProofError::RelationMismatch)?;
        let value_evaluation_commitment = srs.zksrs.commit(value_at_bound, value_evaluation_hiding);
        let slack_evaluation_commitment = srs.zksrs.commit(slack_at_bound, slack_evaluation_hiding);
        let support_evaluation_commitment = srs
            .zksrs
            .commit(support_at_bound, support_evaluation_hiding);
        let value_bound_opening = {
            let mut adapter = SparseU64BvaFieldAdapter::<S, E::ScalarField>::new(source);
            let (opening, open_profile) = open_sparse_bva_field_streaming::<E, _>(
                &srs.pc_pp,
                &mut adapter,
                &bound_xl,
                &bound_xr,
                E::pairing(value_evaluation_commitment, srs.h_hat),
                value_commitment,
                value_tier_one,
                value_evaluation_hiding,
                value_hiding,
            )
            .map_err(map_bva_range_pcs_error)?;
            profile.absorb_open(open_profile);
            BvaVectorOpeningProof {
                value_commitment: value_evaluation_commitment,
                opening,
            }
        };
        let slack_bound_opening = {
            let mut adapter = SparseSlackU64Adapter::new(source, maximum);
            let (opening, open_profile) = open_sparse_bva_field_streaming::<E, _>(
                &srs.pc_pp,
                &mut adapter,
                &bound_xl,
                &bound_xr,
                E::pairing(slack_evaluation_commitment, srs.h_hat),
                slack_commitment,
                &slack_tier_one,
                slack_evaluation_hiding,
                slack_hiding,
            )
            .map_err(map_plain_bva_range_pcs_error)?;
            profile.absorb_open(open_profile);
            BvaVectorOpeningProof {
                value_commitment: slack_evaluation_commitment,
                opening,
            }
        };
        let support_bound_opening = {
            let mut adapter = SparseSupportU64Adapter::new(source);
            let (opening, open_profile) = open_sparse_bva_field_streaming::<E, _>(
                &srs.pc_pp,
                &mut adapter,
                &bound_xl,
                &bound_xr,
                E::pairing(support_evaluation_commitment, srs.h_hat),
                support_commitment,
                &support_tier_one,
                support_evaluation_hiding,
                support_hiding,
            )
            .map_err(map_plain_bva_range_pcs_error)?;
            profile.absorb_open(open_profile);
            BvaVectorOpeningProof {
                value_commitment: support_evaluation_commitment,
                opening,
            }
        };

        Ok((
            Self {
                slack_commitment,
                support_commitment,
                value_range_proof,
                slack_range_proof,
                support_range_proof,
                value_bound_opening,
                slack_bound_opening,
                support_bound_opening,
            },
            profile,
        ))
    }

    pub fn verify(
        &self,
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        layout: &BvaLayout,
        value_commitment: PairingOutput<E>,
        maximum: u64,
    ) -> bool {
        let Ok(bits) = sparse_upper_bound_bits::<std::convert::Infallible>(maximum) else {
            return false;
        };
        if !self.value_range_proof.verify(
            srs,
            layer,
            sparse_upper_bound_range_tag(tag, 0),
            layout,
            value_commitment,
            bits,
        ) || !self.slack_range_proof.verify(
            srs,
            layer,
            sparse_upper_bound_range_tag(tag, 1),
            layout,
            self.slack_commitment,
            bits,
        ) || !self.support_range_proof.verify(
            srs,
            layer,
            sparse_upper_bound_range_tag(tag, 2),
            layout,
            self.support_commitment,
            1,
        ) {
            return false;
        }
        let challenges = derive_sparse_upper_bound_challenges::<E>(
            layer,
            tag,
            layout,
            value_commitment,
            self.slack_commitment,
            self.support_commitment,
            maximum,
        );
        verify_bva_vector_opening(
            srs,
            value_commitment,
            layout,
            &challenges,
            &self.value_bound_opening,
        ) && verify_bva_vector_opening(
            srs,
            self.slack_commitment,
            layout,
            &challenges,
            &self.slack_bound_opening,
        ) && verify_bva_vector_opening(
            srs,
            self.support_commitment,
            layout,
            &challenges,
            &self.support_bound_opening,
        ) && self.value_bound_opening.value_commitment + self.slack_bound_opening.value_commitment
            == self
                .support_bound_opening
                .value_commitment
                .mul(E::ScalarField::from(maximum))
    }

    pub fn serialized_size_bytes(&self) -> usize {
        let opening_size = |opening: &BvaVectorOpeningProof<E>| {
            opening.value_commitment.serialized_size(Compress::Yes)
                + opening.opening.serialized_size(Compress::Yes)
        };
        self.slack_commitment.serialized_size(Compress::Yes)
            + self.support_commitment.serialized_size(Compress::Yes)
            + self.value_range_proof.serialized_size_bytes()
            + self.slack_range_proof.serialized_size_bytes()
            + self.support_range_proof.serialized_size_bytes()
            + opening_size(&self.value_bound_opening)
            + opening_size(&self.slack_bound_opening)
            + opening_size(&self.support_bound_opening)
    }
}

/// Rewindable source for one pointwise affine relation over three committed
/// BVA vectors. Every replay must visit the same physical vectors in canonical
/// order; logical padding is omitted from all three slices.
pub trait BvaAffineRelationSource<F: PrimeField> {
    type Error;

    fn layout(&self) -> &BvaLayout;

    fn for_each_vector(
        &mut self,
        visit: &mut dyn FnMut(usize, &[F], &[F], &[F]),
    ) -> Result<(), Self::Error>;
}

/// Random-point proof of `factor = constant + alpha * value + beta * counter`.
/// The three MLE evaluations are Pedersen committed with correlated hiding,
/// so the verifier can check the affine equality without learning them.
pub struct BvaAffineRelationProof<E: Pairing> {
    pub factor_opening: BvaVectorOpeningProof<E>,
    pub value_opening: BvaVectorOpeningProof<E>,
    pub counter_opening: BvaVectorOpeningProof<E>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BvaAffineRelationProfile {
    pub source_passes: usize,
    pub total_visited_entries: usize,
    pub maximum_live_vector_entries: usize,
    pub materialized_padding_entries: usize,
    pub padding_field_operations: u64,
}

impl BvaAffineRelationProfile {
    fn absorb_open(&mut self, profile: StreamingOpenProfile) {
        self.source_passes += profile.source_passes;
        self.total_visited_entries += profile.physical_entries;
        self.materialized_padding_entries += profile.materialized_projection_padding_entries;
        self.padding_field_operations += profile.padding_projection_field_multiplications as u64;
    }
}

#[derive(Debug)]
pub enum BvaAffineRelationError<SourceError> {
    Bva(BvaError),
    VirtualPcs(VirtualPcsError),
    Source(SourceError),
    RelationMismatch,
}

impl<SourceError: Display> Display for BvaAffineRelationError<SourceError> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bva(error) => Display::fmt(error, formatter),
            Self::VirtualPcs(error) => Display::fmt(error, formatter),
            Self::Source(error) => Display::fmt(error, formatter),
            Self::RelationMismatch => write!(formatter, "BVA affine relation mismatch"),
        }
    }
}

impl<SourceError> std::error::Error for BvaAffineRelationError<SourceError> where
    SourceError: std::error::Error + 'static
{
}

#[derive(Clone, Copy)]
enum BvaAffineView {
    Factor,
    Value,
    Counter,
}

struct BvaAffineViewSource<'a, S, F> {
    source: &'a mut S,
    layout: BvaLayout,
    view: BvaAffineView,
    _field: PhantomData<F>,
}

impl<'a, S, F: PrimeField> BvaAffineViewSource<'a, S, F>
where
    S: BvaAffineRelationSource<F>,
{
    fn new(source: &'a mut S, view: BvaAffineView) -> Self {
        Self {
            layout: source.layout().clone(),
            source,
            view,
            _field: PhantomData,
        }
    }
}

impl<S, F> BvaFieldSource<F> for BvaAffineViewSource<'_, S, F>
where
    S: BvaAffineRelationSource<F>,
    F: PrimeField,
{
    type Error = S::Error;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[F])) -> Result<(), Self::Error> {
        let view = self.view;
        self.source
            .for_each_vector(&mut |vector_index, factor, value, counter| {
                visit(
                    vector_index,
                    match view {
                        BvaAffineView::Factor => factor,
                        BvaAffineView::Value => value,
                        BvaAffineView::Counter => counter,
                    },
                );
            })
    }
}

impl<E: Pairing> BvaAffineRelationProof<E> {
    #[allow(clippy::too_many_arguments)]
    pub fn prove<S: BvaAffineRelationSource<E::ScalarField>>(
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        source: &mut S,
        factor_commitment: PairingOutput<E>,
        factor_tier_one: &[E::G1],
        factor_hiding: E::ScalarField,
        value_commitment: PairingOutput<E>,
        value_tier_one: &[E::G1],
        value_hiding: E::ScalarField,
        counter_commitment: PairingOutput<E>,
        counter_tier_one: &[E::G1],
        counter_hiding: E::ScalarField,
        constant: E::ScalarField,
        value_coefficient: E::ScalarField,
        counter_coefficient: E::ScalarField,
    ) -> Result<(Self, BvaAffineRelationProfile), BvaAffineRelationError<S::Error>> {
        let layout = source.layout().clone();
        let challenges = derive_bva_affine_relation_challenges::<E>(
            layer,
            tag,
            &layout,
            factor_commitment,
            value_commitment,
            counter_commitment,
            constant,
            value_coefficient,
            counter_coefficient,
        );
        let tbv = TensorBasisVector::global_tensor(&layout, &challenges)
            .map_err(BvaAffineRelationError::Bva)?;
        let mut factor_evaluation = E::ScalarField::zero();
        let mut value_evaluation = E::ScalarField::zero();
        let mut counter_evaluation = E::ScalarField::zero();
        let mut physical_mask_evaluation = E::ScalarField::zero();
        let mut visited = vec![false; layout.vector_lengths().len()];
        let mut relation_error = None;
        let mut physical_entries = 0_usize;
        let mut maximum_live_vector_entries = 0_usize;
        source
            .for_each_vector(&mut |vector_index, factor, value, counter| {
                if relation_error.is_some() {
                    return;
                }
                if vector_index >= visited.len() || visited[vector_index] {
                    relation_error = Some(BvaError::VectorIndex);
                    return;
                }
                let expected_len = layout.vector_lengths()[vector_index];
                if factor.len() != expected_len
                    || value.len() != expected_len
                    || counter.len() != expected_len
                {
                    relation_error = Some(BvaError::VectorLength);
                    return;
                }
                visited[vector_index] = true;
                physical_entries += expected_len;
                maximum_live_vector_entries = maximum_live_vector_entries.max(expected_len);
                factor_evaluation += tbv.project_physical_vector(vector_index, factor).0;
                value_evaluation += tbv.project_physical_vector(vector_index, value).0;
                counter_evaluation += tbv.project_physical_vector(vector_index, counter).0;
                match tbv.physical_coefficient_sum(vector_index) {
                    Ok(value) => physical_mask_evaluation += value,
                    Err(error) => relation_error = Some(error),
                }
            })
            .map_err(BvaAffineRelationError::Source)?;
        if let Some(error) = relation_error {
            return Err(BvaAffineRelationError::Bva(error));
        }
        if visited.iter().any(|visited| !visited) {
            return Err(BvaAffineRelationError::Bva(BvaError::MissingVector));
        }
        if factor_evaluation
            != constant * physical_mask_evaluation
                + value_coefficient * value_evaluation
                + counter_coefficient * counter_evaluation
        {
            return Err(BvaAffineRelationError::RelationMismatch);
        }

        let rng = &mut ark_std::rand::thread_rng();
        let value_evaluation_hiding = E::ScalarField::rand(rng);
        let counter_evaluation_hiding = E::ScalarField::rand(rng);
        let factor_evaluation_hiding = value_coefficient * value_evaluation_hiding
            + counter_coefficient * counter_evaluation_hiding;
        let factor_value_commitment = srs
            .zksrs
            .commit(factor_evaluation, factor_evaluation_hiding);
        let value_value_commitment = srs.zksrs.commit(value_evaluation, value_evaluation_hiding);
        let counter_value_commitment = srs
            .zksrs
            .commit(counter_evaluation, counter_evaluation_hiding);
        let (xl, xr) = bva_eo_opening_point(layout.logical_len(), &challenges)
            .map_err(BvaAffineRelationError::VirtualPcs)?;
        let mut profile = BvaAffineRelationProfile {
            source_passes: 1,
            total_visited_entries: physical_entries.saturating_mul(3),
            maximum_live_vector_entries,
            ..Default::default()
        };

        let factor_opening = {
            let mut view = BvaAffineViewSource::new(source, BvaAffineView::Factor);
            let (opening, open_profile) = open_bva_field_streaming::<E, _>(
                &srs.pc_pp,
                &mut view,
                &xl,
                &xr,
                E::pairing(factor_value_commitment, srs.h_hat),
                factor_commitment,
                factor_tier_one,
                factor_evaluation_hiding,
                factor_hiding,
            )
            .map_err(map_bva_affine_pcs_error)?;
            profile.absorb_open(open_profile);
            BvaVectorOpeningProof {
                value_commitment: factor_value_commitment,
                opening,
            }
        };
        let value_opening = {
            let mut view = BvaAffineViewSource::new(source, BvaAffineView::Value);
            let (opening, open_profile) = open_bva_field_streaming::<E, _>(
                &srs.pc_pp,
                &mut view,
                &xl,
                &xr,
                E::pairing(value_value_commitment, srs.h_hat),
                value_commitment,
                value_tier_one,
                value_evaluation_hiding,
                value_hiding,
            )
            .map_err(map_bva_affine_pcs_error)?;
            profile.absorb_open(open_profile);
            BvaVectorOpeningProof {
                value_commitment: value_value_commitment,
                opening,
            }
        };
        let counter_opening = {
            let mut view = BvaAffineViewSource::new(source, BvaAffineView::Counter);
            let (opening, open_profile) = open_bva_field_streaming::<E, _>(
                &srs.pc_pp,
                &mut view,
                &xl,
                &xr,
                E::pairing(counter_value_commitment, srs.h_hat),
                counter_commitment,
                counter_tier_one,
                counter_evaluation_hiding,
                counter_hiding,
            )
            .map_err(map_bva_affine_pcs_error)?;
            profile.absorb_open(open_profile);
            BvaVectorOpeningProof {
                value_commitment: counter_value_commitment,
                opening,
            }
        };

        Ok((
            Self {
                factor_opening,
                value_opening,
                counter_opening,
            },
            profile,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &self,
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        layout: &BvaLayout,
        factor_commitment: PairingOutput<E>,
        value_commitment: PairingOutput<E>,
        counter_commitment: PairingOutput<E>,
        constant: E::ScalarField,
        value_coefficient: E::ScalarField,
        counter_coefficient: E::ScalarField,
    ) -> bool {
        let challenges = derive_bva_affine_relation_challenges::<E>(
            layer,
            tag,
            layout,
            factor_commitment,
            value_commitment,
            counter_commitment,
            constant,
            value_coefficient,
            counter_coefficient,
        );
        let Ok(tbv) = TensorBasisVector::global_tensor(layout, &challenges) else {
            return false;
        };
        let Some(physical_mask_evaluation) = layout.vector_lengths().iter().enumerate().try_fold(
            E::ScalarField::zero(),
            |sum, (vector_index, _)| {
                tbv.physical_coefficient_sum(vector_index)
                    .ok()
                    .map(|value| sum + value)
            },
        ) else {
            return false;
        };
        let public_constant_commitment = srs
            .zksrs
            .commit(constant * physical_mask_evaluation, E::ScalarField::zero());
        self.factor_opening.value_commitment
            == public_constant_commitment
                + self.value_opening.value_commitment.mul(value_coefficient)
                + self
                    .counter_opening
                    .value_commitment
                    .mul(counter_coefficient)
            && verify_bva_vector_opening(
                srs,
                factor_commitment,
                layout,
                &challenges,
                &self.factor_opening,
            )
            && verify_bva_vector_opening(
                srs,
                value_commitment,
                layout,
                &challenges,
                &self.value_opening,
            )
            && verify_bva_vector_opening(
                srs,
                counter_commitment,
                layout,
                &challenges,
                &self.counter_opening,
            )
    }

    pub fn serialized_size_bytes(&self) -> usize {
        let opening_size = |opening: &BvaVectorOpeningProof<E>| {
            opening.value_commitment.serialized_size(Compress::Yes)
                + opening.opening.serialized_size(Compress::Yes)
        };
        opening_size(&self.factor_opening)
            + opening_size(&self.value_opening)
            + opening_size(&self.counter_opening)
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_bva_affine_relation_challenges<E: Pairing>(
    layer: usize,
    tag: usize,
    layout: &BvaLayout,
    factor_commitment: PairingOutput<E>,
    value_commitment: PairingOutput<E>,
    counter_commitment: PairingOutput<E>,
    constant: E::ScalarField,
    value_coefficient: E::ScalarField,
    counter_coefficient: E::ScalarField,
) -> Vec<E::ScalarField> {
    let mut fs = FiatShamir::new();
    fs.push(&0x4256_4146_4649_4e45_u64);
    fs.push(&tag);
    fs.push(&layer);
    fs.push(&layout.logical_len());
    let offsets = layout.bucket_offsets();
    for vector_index in 0..layout.vector_lengths().len() {
        fs.push(&layout.vector_lengths()[vector_index]);
        fs.push(&offsets[layout.vector_to_bucket()[vector_index]]);
    }
    fs.push(&factor_commitment);
    fs.push(&value_commitment);
    fs.push(&counter_commitment);
    fs.push(&constant);
    fs.push(&value_coefficient);
    fs.push(&counter_coefficient);
    (0..layout.logical_len().ilog2())
        .map(|_| fs.gen_challenge::<E::ScalarField>())
        .collect()
}

fn map_bva_affine_pcs_error<SourceError>(
    error: StreamingVirtualPcsError<SourceError>,
) -> BvaAffineRelationError<SourceError> {
    match error {
        StreamingVirtualPcsError::Pcs(error) => BvaAffineRelationError::VirtualPcs(error),
        StreamingVirtualPcsError::Source(error) => BvaAffineRelationError::Source(error),
    }
}

/// Proves an arbitrary public TBV evaluation against one committed BVA source.
/// The TBV is reconstructed by the verifier from public routing metadata; only
/// its claimed scalar is carried by the surrounding protocol. Logical padding
/// remains absent from the source, EO reduction, and SmartPC opening.
#[derive(Clone, Debug)]
pub struct BvaCommittedEvaluationProof<E: Pairing> {
    pub eo_proof: EoLiteBulletProof<E::ScalarField>,
    pub opening: BvaVectorOpeningProof<E>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BvaCommittedEvaluationProfile {
    pub source_passes: usize,
    pub total_visited_entries: usize,
    pub maximum_live_vector_entries: usize,
    pub materialized_padding_entries: usize,
    pub padding_field_operations: u64,
}

impl BvaCommittedEvaluationProfile {
    fn absorb_inner(
        &mut self,
        profile: crate::protocols::virtualpcs::BvaStreamingInnerProductProfile,
    ) {
        self.source_passes += profile.source_passes;
        self.total_visited_entries += profile.physical_entries;
        self.maximum_live_vector_entries = self
            .maximum_live_vector_entries
            .max(profile.maximum_live_vector_entries);
        self.materialized_padding_entries += profile.materialized_padding_entries;
        self.padding_field_operations += profile.padding_field_operations;
    }

    fn absorb_open(&mut self, profile: StreamingOpenProfile) {
        self.source_passes += profile.source_passes;
        self.total_visited_entries += profile.physical_entries;
        self.materialized_padding_entries += profile.materialized_projection_padding_entries;
        self.padding_field_operations += profile.padding_projection_field_multiplications as u64;
    }

    fn absorb_eo(&mut self, counters: PaddingWorkCounters, profile: StreamingEoProfile) {
        self.source_passes += profile.source_passes;
        self.total_visited_entries += profile.total_visited_entries;
        self.maximum_live_vector_entries = self
            .maximum_live_vector_entries
            .max(profile.maximum_live_vector_entries);
        self.materialized_padding_entries += profile.materialized_padding_entries;
        self.padding_field_operations += counters.padding_ip_field_ops
            + counters.padding_fold_field_ops
            + profile.padding_field_operations;
    }
}

#[derive(Debug)]
pub enum BvaCommittedEvaluationError<SourceError> {
    Bva(BvaError),
    VirtualPcs(VirtualPcsError),
    Source(SourceError),
    RelationMismatch,
}

impl<SourceError: Display> Display for BvaCommittedEvaluationError<SourceError> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bva(error) => Display::fmt(error, formatter),
            Self::VirtualPcs(error) => Display::fmt(error, formatter),
            Self::Source(error) => Display::fmt(error, formatter),
            Self::RelationMismatch => write!(formatter, "BVA committed evaluation mismatch"),
        }
    }
}

impl<SourceError> std::error::Error for BvaCommittedEvaluationError<SourceError> where
    SourceError: std::error::Error + 'static
{
}

impl<E: Pairing> BvaCommittedEvaluationProof<E> {
    #[allow(clippy::too_many_arguments)]
    pub fn prove_materialized(
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        relation_digest: [u8; 32],
        source: &mut BucketedVector<E::ScalarField>,
        commitment: PairingOutput<E>,
        tier_one: &[E::G1],
        hiding: E::ScalarField,
        tbv: &TensorBasisVector<E::ScalarField>,
        claim: E::ScalarField,
    ) -> Result<(Self, BvaCommittedEvaluationProfile), BvaCommittedEvaluationError<Infallible>>
    {
        let layout = source.layout().clone();
        if tbv.layout() != &layout {
            return Err(BvaCommittedEvaluationError::RelationMismatch);
        }
        let binding = derive_bva_committed_evaluation_binding::<E>(
            layer,
            tag,
            relation_digest,
            &layout,
            commitment,
            tbv,
            claim,
        );
        if binding.is_zero() {
            return Err(BvaCommittedEvaluationError::RelationMismatch);
        }
        let evaluation = source
            .inner_product(tbv)
            .map_err(BvaCommittedEvaluationError::Bva)?;
        if evaluation.value != claim {
            return Err(BvaCommittedEvaluationError::RelationMismatch);
        }
        let (eo_proof, counters) = EoLiteBulletProof::prove_bound(source, tbv, claim, binding)
            .map_err(BvaCommittedEvaluationError::Bva)?;
        let verification = eo_proof.verify_with_tbv_bound(tbv, claim, binding);
        if !verification.accepted {
            return Err(BvaCommittedEvaluationError::RelationMismatch);
        }
        let (xl, xr) = bva_eo_opening_point(layout.logical_len(), &verification.challenges_inv)
            .map_err(BvaCommittedEvaluationError::VirtualPcs)?;
        let value_commitment = srs.zksrs.commit(eo_proof.a_reduced, E::ScalarField::zero());
        let (opening, opening_profile) = open_bva_field_streaming::<E, _>(
            &srs.pc_pp,
            source,
            &xl,
            &xr,
            E::pairing(value_commitment, srs.h_hat),
            commitment,
            tier_one,
            E::ScalarField::zero(),
            hiding,
        )
        .map_err(map_committed_evaluation_pcs_error)?;
        let mut profile = BvaCommittedEvaluationProfile {
            total_visited_entries: source.physical_entries().saturating_mul(2),
            maximum_live_vector_entries: source.physical_entries(),
            materialized_padding_entries: counters.materialized_padding_entries as usize,
            padding_field_operations: counters.padding_ip_field_ops
                + counters.padding_fold_field_ops,
            ..Default::default()
        };
        profile.absorb_open(opening_profile);
        Ok((
            Self {
                eo_proof,
                opening: BvaVectorOpeningProof {
                    value_commitment,
                    opening,
                },
            },
            profile,
        ))
    }

    /// Materialized committed evaluation that consumes the public TBV after
    /// one exact claim check. This preserves the transcript while avoiding a
    /// duplicate inner-product pass and a potentially large TBV clone.
    #[allow(clippy::too_many_arguments)]
    pub fn prove_materialized_consuming_tbv(
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        relation_digest: [u8; 32],
        source: &mut BucketedVector<E::ScalarField>,
        commitment: PairingOutput<E>,
        tier_one: &[E::G1],
        hiding: E::ScalarField,
        tbv: TensorBasisVector<E::ScalarField>,
        claim: E::ScalarField,
    ) -> Result<(Self, BvaCommittedEvaluationProfile), BvaCommittedEvaluationError<Infallible>>
    {
        let layout = source.layout().clone();
        if tbv.layout() != &layout {
            return Err(BvaCommittedEvaluationError::RelationMismatch);
        }
        let binding = derive_bva_committed_evaluation_binding::<E>(
            layer,
            tag,
            relation_digest,
            &layout,
            commitment,
            &tbv,
            claim,
        );
        if binding.is_zero() {
            return Err(BvaCommittedEvaluationError::RelationMismatch);
        }
        let evaluation = source
            .inner_product(&tbv)
            .map_err(BvaCommittedEvaluationError::Bva)?;
        if evaluation.value != claim {
            return Err(BvaCommittedEvaluationError::RelationMismatch);
        }
        let (eo_proof, counters) = EoLiteBulletProof::prove_bound_consuming_tbv_prechecked(
            source,
            tbv,
            claim,
            binding,
            evaluation.counters,
        )
        .map_err(BvaCommittedEvaluationError::Bva)?;
        let verification =
            eo_proof.verify_scalar_for_prover_bound(layout.logical_len(), claim, binding);
        if !verification.accepted {
            return Err(BvaCommittedEvaluationError::RelationMismatch);
        }
        let (xl, xr) = bva_eo_opening_point(layout.logical_len(), &verification.challenges_inv)
            .map_err(BvaCommittedEvaluationError::VirtualPcs)?;
        let value_commitment = srs.zksrs.commit(eo_proof.a_reduced, E::ScalarField::zero());
        let (opening, opening_profile) = open_bva_field_streaming::<E, _>(
            &srs.pc_pp,
            source,
            &xl,
            &xr,
            E::pairing(value_commitment, srs.h_hat),
            commitment,
            tier_one,
            E::ScalarField::zero(),
            hiding,
        )
        .map_err(map_committed_evaluation_pcs_error)?;
        let mut profile = BvaCommittedEvaluationProfile {
            total_visited_entries: source.physical_entries(),
            maximum_live_vector_entries: source.physical_entries(),
            materialized_padding_entries: counters.materialized_padding_entries as usize,
            padding_field_operations: counters.padding_ip_field_ops
                + counters.padding_fold_field_ops,
            ..Default::default()
        };
        profile.absorb_open(opening_profile);
        Ok((
            Self {
                eo_proof,
                opening: BvaVectorOpeningProof {
                    value_commitment,
                    opening,
                },
            },
            profile,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prove<S>(
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        relation_digest: [u8; 32],
        source: &mut S,
        commitment: PairingOutput<E>,
        tier_one: &[E::G1],
        hiding: E::ScalarField,
        tbv: &TensorBasisVector<E::ScalarField>,
        claim: E::ScalarField,
    ) -> Result<(Self, BvaCommittedEvaluationProfile), BvaCommittedEvaluationError<S::Error>>
    where
        S: BvaFieldSource<E::ScalarField>,
    {
        let layout = source.layout().clone();
        if tbv.layout() != &layout {
            return Err(BvaCommittedEvaluationError::RelationMismatch);
        }
        let binding = derive_bva_committed_evaluation_binding::<E>(
            layer,
            tag,
            relation_digest,
            &layout,
            commitment,
            tbv,
            claim,
        );
        if binding.is_zero() {
            return Err(BvaCommittedEvaluationError::RelationMismatch);
        }
        let evaluation = inner_product_bva_field_streaming(source, tbv)
            .map_err(map_committed_evaluation_pcs_error)?;
        if evaluation.value != claim {
            return Err(BvaCommittedEvaluationError::RelationMismatch);
        }
        let mut profile = BvaCommittedEvaluationProfile::default();
        profile.absorb_inner(evaluation.profile);
        let (eo_proof, counters, eo_profile) =
            EoLiteBulletProof::prove_streaming_bound(source, tbv, claim, binding)
                .map_err(map_committed_evaluation_eo_error)?;
        profile.absorb_eo(counters, eo_profile);
        let verification = eo_proof.verify_with_tbv_bound(tbv, claim, binding);
        if !verification.accepted {
            return Err(BvaCommittedEvaluationError::RelationMismatch);
        }
        let (xl, xr) = bva_eo_opening_point(layout.logical_len(), &verification.challenges_inv)
            .map_err(BvaCommittedEvaluationError::VirtualPcs)?;
        let value_commitment = srs.zksrs.commit(eo_proof.a_reduced, E::ScalarField::zero());
        let (opening, opening_profile) = open_bva_field_streaming::<E, _>(
            &srs.pc_pp,
            source,
            &xl,
            &xr,
            E::pairing(value_commitment, srs.h_hat),
            commitment,
            tier_one,
            E::ScalarField::zero(),
            hiding,
        )
        .map_err(map_committed_evaluation_pcs_error)?;
        profile.absorb_open(opening_profile);
        Ok((
            Self {
                eo_proof,
                opening: BvaVectorOpeningProof {
                    value_commitment,
                    opening,
                },
            },
            profile,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &self,
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        relation_digest: [u8; 32],
        layout: &BvaLayout,
        commitment: PairingOutput<E>,
        tbv: &TensorBasisVector<E::ScalarField>,
        claim: E::ScalarField,
    ) -> bool {
        if tbv.layout() != layout {
            return false;
        }
        let binding = derive_bva_committed_evaluation_binding::<E>(
            layer,
            tag,
            relation_digest,
            layout,
            commitment,
            tbv,
            claim,
        );
        if binding.is_zero() {
            return false;
        }
        let verification = self.eo_proof.verify_with_tbv_bound(tbv, claim, binding);
        verification.accepted
            && self.opening.value_commitment
                == srs
                    .zksrs
                    .commit(self.eo_proof.a_reduced, E::ScalarField::zero())
            && verify_bva_vector_opening(
                srs,
                commitment,
                layout,
                &verification.challenges_inv,
                &self.opening,
            )
    }

    /// Verifies the committed evaluation while consuming the compact public
    /// TBV. This is transcript-identical to [`Self::verify`], but avoids
    /// retaining the original TBV beside the verifier's folding state.
    #[allow(clippy::too_many_arguments)]
    pub fn verify_consuming_tbv(
        &self,
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        relation_digest: [u8; 32],
        layout: &BvaLayout,
        commitment: PairingOutput<E>,
        tbv: TensorBasisVector<E::ScalarField>,
        claim: E::ScalarField,
    ) -> bool {
        if tbv.layout() != layout {
            return false;
        }
        let binding = derive_bva_committed_evaluation_binding::<E>(
            layer,
            tag,
            relation_digest,
            layout,
            commitment,
            &tbv,
            claim,
        );
        if binding.is_zero() {
            return false;
        }
        let verification = self
            .eo_proof
            .verify_with_tbv_bound_consuming(tbv, claim, binding);
        verification.accepted
            && self.opening.value_commitment
                == srs
                    .zksrs
                    .commit(self.eo_proof.a_reduced, E::ScalarField::zero())
            && verify_bva_vector_opening(
                srs,
                commitment,
                layout,
                &verification.challenges_inv,
                &self.opening,
            )
    }

    pub fn serialized_size_bytes(&self) -> usize {
        self.eo_proof.serialized_size(Compress::Yes)
            + self.opening.value_commitment.serialized_size(Compress::Yes)
            + self.opening.opening.serialized_size(Compress::Yes)
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_bva_committed_evaluation_binding<E: Pairing>(
    layer: usize,
    tag: usize,
    relation_digest: [u8; 32],
    layout: &BvaLayout,
    commitment: PairingOutput<E>,
    tbv: &TensorBasisVector<E::ScalarField>,
    claim: E::ScalarField,
) -> E::ScalarField {
    let mut fs = FiatShamir::new();
    fs.push(&0x4256_4143_4556_414c_u64);
    fs.push(&tag);
    fs.push(&layer);
    fs.push(&relation_digest);
    push_bva_layout(&mut fs, layout);
    fs.push(&commitment);
    tbv.absorb_transcript(&mut fs);
    fs.push(&claim);
    fs.gen_challenge::<E::ScalarField>()
}

fn map_committed_evaluation_pcs_error<SourceError>(
    error: StreamingVirtualPcsError<SourceError>,
) -> BvaCommittedEvaluationError<SourceError> {
    match error {
        StreamingVirtualPcsError::Pcs(error) => BvaCommittedEvaluationError::VirtualPcs(error),
        StreamingVirtualPcsError::Source(error) => BvaCommittedEvaluationError::Source(error),
    }
}

fn map_committed_evaluation_eo_error<SourceError>(
    error: StreamingEoError<SourceError>,
) -> BvaCommittedEvaluationError<SourceError> {
    match error {
        StreamingEoError::Bva(error) => BvaCommittedEvaluationError::Bva(error),
        StreamingEoError::Source(error) => BvaCommittedEvaluationError::Source(error),
    }
}

/// Equality proof between one auxiliary BVA commitment and a routed view of a
/// canonical commitment. For source vector `j`, `routes[j] = (k, scale)` means
/// `source[j] = scale * target[k] + public_constants[j]`; an omitted route is
/// a public constant vector. Only physical entries participate.
pub struct BvaRoutedEqualityProof<E: Pairing> {
    pub target_claim: E::ScalarField,
    pub target_eo_proof: EoLiteBulletProof<E::ScalarField>,
    pub source_opening: BvaVectorOpeningProof<E>,
    pub target_opening: BvaVectorOpeningProof<E>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct BvaRoutedEqualityProfile {
    pub source_passes: usize,
    pub total_visited_entries: usize,
    pub maximum_live_vector_entries: usize,
    pub materialized_padding_entries: usize,
    pub padding_field_operations: u64,
}

impl BvaRoutedEqualityProfile {
    fn absorb_inner(
        &mut self,
        profile: crate::protocols::virtualpcs::BvaStreamingInnerProductProfile,
    ) {
        self.source_passes += profile.source_passes;
        self.total_visited_entries += profile.physical_entries;
        self.maximum_live_vector_entries = self
            .maximum_live_vector_entries
            .max(profile.maximum_live_vector_entries);
        self.materialized_padding_entries += profile.materialized_padding_entries;
        self.padding_field_operations += profile.padding_field_operations;
    }

    fn absorb_open(&mut self, profile: StreamingOpenProfile) {
        self.source_passes += profile.source_passes;
        self.total_visited_entries += profile.physical_entries;
        self.materialized_padding_entries += profile.materialized_projection_padding_entries;
        self.padding_field_operations += profile.padding_projection_field_multiplications as u64;
    }

    fn absorb_eo(&mut self, counters: PaddingWorkCounters, profile: StreamingEoProfile) {
        self.source_passes += profile.source_passes;
        self.total_visited_entries += profile.total_visited_entries;
        self.maximum_live_vector_entries = self
            .maximum_live_vector_entries
            .max(profile.maximum_live_vector_entries);
        self.materialized_padding_entries += profile.materialized_padding_entries;
        self.padding_field_operations += counters.padding_ip_field_ops
            + counters.padding_fold_field_ops
            + profile.padding_field_operations;
    }
}

#[derive(Debug)]
pub enum BvaRoutedEqualityError<SourceError, TargetError> {
    Bva(BvaError),
    VirtualPcs(VirtualPcsError),
    Source(SourceError),
    Target(TargetError),
    RelationMismatch,
}

impl<SourceError: Display, TargetError: Display> Display
    for BvaRoutedEqualityError<SourceError, TargetError>
{
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Bva(error) => Display::fmt(error, formatter),
            Self::VirtualPcs(error) => Display::fmt(error, formatter),
            Self::Source(error) => Display::fmt(error, formatter),
            Self::Target(error) => Display::fmt(error, formatter),
            Self::RelationMismatch => write!(formatter, "BVA routed equality mismatch"),
        }
    }
}

impl<SourceError, TargetError> std::error::Error
    for BvaRoutedEqualityError<SourceError, TargetError>
where
    SourceError: std::error::Error + 'static,
    TargetError: std::error::Error + 'static,
{
}

struct BvaRoutedEqualityChallenges<F> {
    source_point: Vec<F>,
    target_eo_binding: F,
}

impl<E: Pairing> BvaRoutedEqualityProof<E> {
    #[allow(clippy::too_many_arguments)]
    pub fn prove<S, T>(
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        source: &mut S,
        source_commitment: PairingOutput<E>,
        source_tier_one: &[E::G1],
        source_hiding: E::ScalarField,
        target: &mut T,
        target_commitment: PairingOutput<E>,
        target_tier_one: &[E::G1],
        target_hiding: E::ScalarField,
        routes: &[Option<(usize, E::ScalarField)>],
        public_constants: &[E::ScalarField],
    ) -> Result<(Self, BvaRoutedEqualityProfile), BvaRoutedEqualityError<S::Error, T::Error>>
    where
        S: BvaFieldSource<E::ScalarField>,
        T: BvaFieldSource<E::ScalarField>,
    {
        let source_layout = source.layout().clone();
        let target_layout = target.layout().clone();
        if routes.len() != source_layout.vector_lengths().len()
            || public_constants.len() != routes.len()
            || routes.iter().all(Option::is_none)
        {
            return Err(BvaRoutedEqualityError::RelationMismatch);
        }
        let challenges = derive_bva_routed_equality_challenges::<E>(
            layer,
            tag,
            &source_layout,
            source_commitment,
            &target_layout,
            target_commitment,
            routes,
            public_constants,
        );
        if challenges.target_eo_binding.is_zero() {
            return Err(BvaRoutedEqualityError::RelationMismatch);
        }
        let source_tbv = TensorBasisVector::global_tensor(&source_layout, &challenges.source_point)
            .map_err(BvaRoutedEqualityError::Bva)?;
        let target_tbv = source_tbv
            .route_vectors(&target_layout, routes)
            .map_err(BvaRoutedEqualityError::Bva)?;
        let public_offset = public_constants
            .iter()
            .enumerate()
            .try_fold(E::ScalarField::zero(), |sum, (vector_index, constant)| {
                source_tbv
                    .physical_coefficient_sum(vector_index)
                    .map(|coefficient_sum| sum + *constant * coefficient_sum)
            })
            .map_err(BvaRoutedEqualityError::Bva)?;
        let source_evaluation = inner_product_bva_field_streaming(source, &source_tbv)
            .map_err(map_routed_source_pcs_error)?;
        let target_evaluation = inner_product_bva_field_streaming(target, &target_tbv)
            .map_err(map_routed_target_pcs_error)?;
        if source_evaluation.value != target_evaluation.value + public_offset {
            return Err(BvaRoutedEqualityError::RelationMismatch);
        }
        let mut profile = BvaRoutedEqualityProfile::default();
        profile.absorb_inner(source_evaluation.profile);
        profile.absorb_inner(target_evaluation.profile);

        let (target_eo_proof, target_eo_counters, target_eo_profile) =
            EoLiteBulletProof::prove_streaming_bound(
                target,
                &target_tbv,
                target_evaluation.value,
                challenges.target_eo_binding,
            )
            .map_err(map_routed_target_eo_error)?;
        profile.absorb_eo(target_eo_counters, target_eo_profile);
        let target_eo_verification = target_eo_proof.verify_with_tbv_bound(
            &target_tbv,
            target_evaluation.value,
            challenges.target_eo_binding,
        );
        if !target_eo_verification.accepted {
            return Err(BvaRoutedEqualityError::RelationMismatch);
        }

        let (source_xl, source_xr) =
            bva_eo_opening_point(source_layout.logical_len(), &challenges.source_point)
                .map_err(BvaRoutedEqualityError::VirtualPcs)?;
        let source_value_commitment = srs
            .zksrs
            .commit(source_evaluation.value, E::ScalarField::zero());
        let (source_opening, source_open_profile) = open_bva_field_streaming::<E, _>(
            &srs.pc_pp,
            source,
            &source_xl,
            &source_xr,
            E::pairing(source_value_commitment, srs.h_hat),
            source_commitment,
            source_tier_one,
            E::ScalarField::zero(),
            source_hiding,
        )
        .map_err(map_routed_source_pcs_error)?;
        profile.absorb_open(source_open_profile);
        let (target_xl, target_xr) = bva_eo_opening_point(
            target_layout.logical_len(),
            &target_eo_verification.challenges_inv,
        )
        .map_err(BvaRoutedEqualityError::VirtualPcs)?;
        let target_value_commitment = srs
            .zksrs
            .commit(target_eo_proof.a_reduced, E::ScalarField::zero());
        let (target_opening, target_open_profile) = open_bva_field_streaming::<E, _>(
            &srs.pc_pp,
            target,
            &target_xl,
            &target_xr,
            E::pairing(target_value_commitment, srs.h_hat),
            target_commitment,
            target_tier_one,
            E::ScalarField::zero(),
            target_hiding,
        )
        .map_err(map_routed_target_pcs_error)?;
        profile.absorb_open(target_open_profile);

        Ok((
            Self {
                target_claim: target_evaluation.value,
                target_eo_proof,
                source_opening: BvaVectorOpeningProof {
                    value_commitment: source_value_commitment,
                    opening: source_opening,
                },
                target_opening: BvaVectorOpeningProof {
                    value_commitment: target_value_commitment,
                    opening: target_opening,
                },
            },
            profile,
        ))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn verify(
        &self,
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        source_layout: &BvaLayout,
        source_commitment: PairingOutput<E>,
        target_layout: &BvaLayout,
        target_commitment: PairingOutput<E>,
        routes: &[Option<(usize, E::ScalarField)>],
        public_constants: &[E::ScalarField],
    ) -> bool {
        if routes.len() != source_layout.vector_lengths().len()
            || public_constants.len() != routes.len()
            || routes.iter().all(Option::is_none)
        {
            return false;
        }
        let challenges = derive_bva_routed_equality_challenges::<E>(
            layer,
            tag,
            source_layout,
            source_commitment,
            target_layout,
            target_commitment,
            routes,
            public_constants,
        );
        if challenges.target_eo_binding.is_zero() {
            return false;
        }
        let Ok(source_tbv) =
            TensorBasisVector::global_tensor(source_layout, &challenges.source_point)
        else {
            return false;
        };
        let Ok(target_tbv) = source_tbv.route_vectors(target_layout, routes) else {
            return false;
        };
        let Ok(public_offset) = public_constants.iter().enumerate().try_fold(
            E::ScalarField::zero(),
            |sum, (vector_index, constant)| {
                source_tbv
                    .physical_coefficient_sum(vector_index)
                    .map(|coefficient_sum| sum + *constant * coefficient_sum)
            },
        ) else {
            return false;
        };
        let target_eo_verification = self.target_eo_proof.verify_with_tbv_bound(
            &target_tbv,
            self.target_claim,
            challenges.target_eo_binding,
        );
        target_eo_verification.accepted
            && self.source_opening.value_commitment
                == srs
                    .zksrs
                    .commit(self.target_claim + public_offset, E::ScalarField::zero())
            && self.target_opening.value_commitment
                == srs
                    .zksrs
                    .commit(self.target_eo_proof.a_reduced, E::ScalarField::zero())
            && verify_bva_vector_opening(
                srs,
                source_commitment,
                source_layout,
                &challenges.source_point,
                &self.source_opening,
            )
            && verify_bva_vector_opening(
                srs,
                target_commitment,
                target_layout,
                &target_eo_verification.challenges_inv,
                &self.target_opening,
            )
    }

    pub fn serialized_size_bytes(&self) -> usize {
        let opening_size = |opening: &BvaVectorOpeningProof<E>| {
            opening.value_commitment.serialized_size(Compress::Yes)
                + opening.opening.serialized_size(Compress::Yes)
        };
        self.target_claim.serialized_size(Compress::Yes)
            + self.target_eo_proof.serialized_size(Compress::Yes)
            + opening_size(&self.source_opening)
            + opening_size(&self.target_opening)
    }
}

#[allow(clippy::too_many_arguments)]
fn derive_bva_routed_equality_challenges<E: Pairing>(
    layer: usize,
    tag: usize,
    source_layout: &BvaLayout,
    source_commitment: PairingOutput<E>,
    target_layout: &BvaLayout,
    target_commitment: PairingOutput<E>,
    routes: &[Option<(usize, E::ScalarField)>],
    public_constants: &[E::ScalarField],
) -> BvaRoutedEqualityChallenges<E::ScalarField> {
    let mut fs = FiatShamir::new();
    fs.push(&0x4256_4152_5445_5155_u64);
    fs.push(&tag);
    fs.push(&layer);
    push_bva_layout(&mut fs, source_layout);
    push_bva_layout(&mut fs, target_layout);
    fs.push(&source_commitment);
    fs.push(&target_commitment);
    fs.push(&routes.len());
    for route in routes {
        match route {
            Some((target_vector, scale)) => {
                fs.push(&1_u64);
                fs.push(target_vector);
                fs.push(scale);
            }
            None => fs.push(&0_u64),
        }
    }
    for constant in public_constants {
        fs.push(constant);
    }
    BvaRoutedEqualityChallenges {
        source_point: (0..source_layout.logical_len().ilog2())
            .map(|_| fs.gen_challenge::<E::ScalarField>())
            .collect(),
        target_eo_binding: fs.gen_challenge::<E::ScalarField>(),
    }
}

fn push_bva_layout(fs: &mut FiatShamir, layout: &BvaLayout) {
    fs.push(&layout.logical_len());
    let offsets = layout.bucket_offsets();
    for vector_index in 0..layout.vector_lengths().len() {
        fs.push(&layout.vector_lengths()[vector_index]);
        fs.push(&offsets[layout.vector_to_bucket()[vector_index]]);
    }
}

fn map_routed_source_pcs_error<SourceError, TargetError>(
    error: StreamingVirtualPcsError<SourceError>,
) -> BvaRoutedEqualityError<SourceError, TargetError> {
    match error {
        StreamingVirtualPcsError::Pcs(error) => BvaRoutedEqualityError::VirtualPcs(error),
        StreamingVirtualPcsError::Source(error) => BvaRoutedEqualityError::Source(error),
    }
}

fn map_routed_target_pcs_error<SourceError, TargetError>(
    error: StreamingVirtualPcsError<TargetError>,
) -> BvaRoutedEqualityError<SourceError, TargetError> {
    match error {
        StreamingVirtualPcsError::Pcs(error) => BvaRoutedEqualityError::VirtualPcs(error),
        StreamingVirtualPcsError::Source(error) => BvaRoutedEqualityError::Target(error),
    }
}

fn map_routed_target_eo_error<SourceError, TargetError>(
    error: StreamingEoError<TargetError>,
) -> BvaRoutedEqualityError<SourceError, TargetError> {
    match error {
        StreamingEoError::Bva(error) => BvaRoutedEqualityError::Bva(error),
        StreamingEoError::Source(error) => BvaRoutedEqualityError::Target(error),
    }
}

fn sparse_upper_bound_bits<SourceError>(
    maximum: u64,
) -> Result<usize, BvaRangeProofError<SourceError>> {
    if maximum == 0 {
        return Err(BvaRangeProofError::InvalidRangeBits(0));
    }
    let bits = (u64::BITS - maximum.leading_zeros()) as usize;
    if bits > 30 {
        return Err(BvaRangeProofError::InvalidRangeBits(bits));
    }
    Ok(bits)
}

fn sparse_upper_bound_range_tag(tag: usize, lane: usize) -> usize {
    tag ^ 0x5350_0000 ^ lane
}

fn absorb_bva_range_profile(total: &mut BvaRangeProfile, current: BvaRangeProfile) {
    total.source_passes += current.source_passes;
    total.total_visited_entries += current.total_visited_entries;
    total.maximum_live_vector_entries = total
        .maximum_live_vector_entries
        .max(current.maximum_live_vector_entries);
    total.maximum_live_projection_entries = total
        .maximum_live_projection_entries
        .max(current.maximum_live_projection_entries);
    total.materialized_padding_entries += current.materialized_padding_entries;
    total.padding_field_operations += current.padding_field_operations;
    total.padding_group_operations += current.padding_group_operations;
}

fn bva_maximum_projection_entries(layout: &BvaLayout) -> usize {
    let logical_log = layout.logical_len().ilog2() as usize;
    1_usize << (logical_log - logical_log / 2)
}

struct SparseSlackU64Adapter<'a, S> {
    source: &'a mut S,
    layout: BvaLayout,
    maximum: u64,
}

impl<'a, S: SparseBvaU64Source> SparseSlackU64Adapter<'a, S> {
    fn new(source: &'a mut S, maximum: u64) -> Self {
        Self {
            layout: source.layout().clone(),
            source,
            maximum,
        }
    }
}

impl<S: SparseBvaU64Source> SparseBvaU64Source for SparseSlackU64Adapter<'_, S> {
    type Error = S::Error;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_nonzero_u64(
        &mut self,
        visit: &mut dyn FnMut(usize, usize, u64),
    ) -> Result<(), Self::Error> {
        let maximum = self.maximum;
        self.source
            .for_each_nonzero_u64(&mut |vector_index, entry_index, value| {
                debug_assert!(value <= maximum);
                let slack = maximum.saturating_sub(value);
                if slack != 0 {
                    visit(vector_index, entry_index, slack);
                }
            })
    }
}

impl<S, F> SparseBvaFieldSource<F> for SparseSlackU64Adapter<'_, S>
where
    S: SparseBvaU64Source,
    F: PrimeField,
{
    type Error = S::Error;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_nonzero(
        &mut self,
        visit: &mut dyn FnMut(usize, usize, F),
    ) -> Result<(), Self::Error> {
        self.for_each_nonzero_u64(&mut |vector_index, entry_index, value| {
            visit(vector_index, entry_index, F::from(value));
        })
    }
}

struct SparseSupportU64Adapter<'a, S> {
    source: &'a mut S,
    layout: BvaLayout,
}

impl<'a, S: SparseBvaU64Source> SparseSupportU64Adapter<'a, S> {
    fn new(source: &'a mut S) -> Self {
        Self {
            layout: source.layout().clone(),
            source,
        }
    }
}

impl<S: SparseBvaU64Source> SparseBvaU64Source for SparseSupportU64Adapter<'_, S> {
    type Error = S::Error;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_nonzero_u64(
        &mut self,
        visit: &mut dyn FnMut(usize, usize, u64),
    ) -> Result<(), Self::Error> {
        self.source
            .for_each_nonzero_u64(&mut |vector_index, entry_index, _| {
                visit(vector_index, entry_index, 1);
            })
    }
}

impl<S, F> SparseBvaFieldSource<F> for SparseSupportU64Adapter<'_, S>
where
    S: SparseBvaU64Source,
    F: PrimeField,
{
    type Error = S::Error;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_nonzero(
        &mut self,
        visit: &mut dyn FnMut(usize, usize, F),
    ) -> Result<(), Self::Error> {
        self.for_each_nonzero_u64(&mut |vector_index, entry_index, value| {
            visit(vector_index, entry_index, F::from(value));
        })
    }
}

struct SparseU64BvaFieldAdapter<'a, S, F> {
    source: &'a mut S,
    layout: BvaLayout,
    _field: PhantomData<F>,
}

impl<'a, S: SparseBvaU64Source, F> SparseU64BvaFieldAdapter<'a, S, F> {
    fn new(source: &'a mut S) -> Self {
        Self {
            layout: source.layout().clone(),
            source,
            _field: PhantomData,
        }
    }
}

impl<S, F> SparseBvaFieldSource<F> for SparseU64BvaFieldAdapter<'_, S, F>
where
    S: SparseBvaU64Source,
    F: PrimeField,
{
    type Error = BvaRangeSourceError<S::Error>;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_nonzero(
        &mut self,
        visit: &mut dyn FnMut(usize, usize, F),
    ) -> Result<(), Self::Error> {
        self.source
            .for_each_nonzero_u64(&mut |vector_index, entry_index, value| {
                visit(vector_index, entry_index, F::from(value));
            })
            .map_err(BvaRangeSourceError::Source)
    }
}

struct SparseBitPlaneBvaFieldAdapter<'a, S, F> {
    source: &'a mut S,
    layout: BvaLayout,
    bit: usize,
    bits: usize,
    _field: PhantomData<F>,
}

impl<'a, S: SparseBvaU64Source, F> SparseBitPlaneBvaFieldAdapter<'a, S, F> {
    fn new(source: &'a mut S, bit: usize, bits: usize) -> Self {
        Self {
            layout: source.layout().clone(),
            source,
            bit,
            bits,
            _field: PhantomData,
        }
    }
}

impl<S, F> SparseBvaFieldSource<F> for SparseBitPlaneBvaFieldAdapter<'_, S, F>
where
    S: SparseBvaU64Source,
    F: PrimeField,
{
    type Error = BvaRangeSourceError<S::Error>;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_nonzero(
        &mut self,
        visit: &mut dyn FnMut(usize, usize, F),
    ) -> Result<(), Self::Error> {
        let bit = self.bit;
        let limit = 1_u64 << self.bits;
        let mut outside_range = false;
        self.source
            .for_each_nonzero_u64(&mut |vector_index, entry_index, value| {
                if outside_range {
                    return;
                }
                if value >= limit {
                    outside_range = true;
                } else if (value >> bit) & 1 == 1 {
                    visit(vector_index, entry_index, F::one());
                }
            })
            .map_err(BvaRangeSourceError::Source)?;
        if outside_range {
            Err(BvaRangeSourceError::ValueOutsideRange)
        } else {
            Ok(())
        }
    }
}

struct SparseBitPairBvaFieldAdapter<'a, S, F> {
    source: &'a mut S,
    layout: BvaLayout,
    bit: usize,
    bits: usize,
    boolean_tbv: TensorBasisVector<F>,
}

impl<'a, S: SparseBvaU64Source, F: PrimeField> SparseBitPairBvaFieldAdapter<'a, S, F> {
    fn new(source: &'a mut S, bit: usize, bits: usize, boolean_tbv: TensorBasisVector<F>) -> Self {
        Self {
            layout: source.layout().clone(),
            source,
            bit,
            bits,
            boolean_tbv,
        }
    }
}

impl<S, F> SparseBvaFieldPairSource<F> for SparseBitPairBvaFieldAdapter<'_, S, F>
where
    S: SparseBvaU64Source,
    F: PrimeField,
{
    type Error = BvaRangeSourceError<S::Error>;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_nonzero_pair(
        &mut self,
        visit: &mut dyn FnMut(usize, usize, F, F),
    ) -> Result<(), Self::Error> {
        let bit = self.bit;
        let limit = 1_u64 << self.bits;
        let boolean_tbv = &self.boolean_tbv;
        let mut error = None;
        self.source
            .for_each_nonzero_u64(&mut |vector_index, entry_index, value| {
                if error.is_some() {
                    return;
                }
                if value >= limit {
                    error = Some(BvaRangeSourceError::ValueOutsideRange);
                } else if (value >> bit) & 1 == 1 {
                    match boolean_tbv.physical_coefficient_at(vector_index, entry_index) {
                        Ok(coefficient) => visit(vector_index, entry_index, F::one(), coefficient),
                        Err(bva_error) => error = Some(BvaRangeSourceError::Bva(bva_error)),
                    }
                }
            })
            .map_err(BvaRangeSourceError::Source)?;
        if let Some(error) = error {
            Err(error)
        } else {
            Ok(())
        }
    }
}

struct BitPlaneBvaFieldAdapter<'a, S, F> {
    source: &'a mut S,
    layout: BvaLayout,
    bit: usize,
    bits: usize,
    _field: PhantomData<F>,
}

impl<'a, S: BvaI32Source, F> BitPlaneBvaFieldAdapter<'a, S, F> {
    fn new(source: &'a mut S, bit: usize, bits: usize) -> Self {
        Self {
            layout: source.layout().clone(),
            source,
            bit,
            bits,
            _field: PhantomData,
        }
    }
}

impl<S, F> BvaFieldSource<F> for BitPlaneBvaFieldAdapter<'_, S, F>
where
    S: BvaI32Source,
    F: PrimeField + From<i32>,
{
    type Error = BvaRangeSourceError<S::Error>;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[F])) -> Result<(), Self::Error> {
        let bit = self.bit;
        let limit = 1_u32 << self.bits;
        let mut outside_range = false;
        self.source
            .for_each_vector(&mut |vector_index, values| {
                if outside_range {
                    return;
                }
                if values
                    .iter()
                    .any(|value| *value < 0 || (*value as u32) >= limit)
                {
                    outside_range = true;
                    return;
                }
                let field_values = values
                    .iter()
                    .map(|value| F::from(((*value as u32 >> bit) & 1) as i32))
                    .collect::<Vec<_>>();
                visit(vector_index, &field_values);
            })
            .map_err(BvaRangeSourceError::Source)?;
        if outside_range {
            Err(BvaRangeSourceError::ValueOutsideRange)
        } else {
            Ok(())
        }
    }
}

struct BitMinusOneBvaFieldAdapter<'a, S, F> {
    source: &'a mut S,
    layout: BvaLayout,
    bit: usize,
    bits: usize,
    _field: PhantomData<F>,
}

impl<'a, S: BvaI32Source, F> BitMinusOneBvaFieldAdapter<'a, S, F> {
    fn new(source: &'a mut S, bit: usize, bits: usize) -> Self {
        Self {
            layout: source.layout().clone(),
            source,
            bit,
            bits,
            _field: PhantomData,
        }
    }
}

impl<S, F> BvaFieldSource<F> for BitMinusOneBvaFieldAdapter<'_, S, F>
where
    S: BvaI32Source,
    F: PrimeField + From<i32>,
{
    type Error = BvaRangeSourceError<S::Error>;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[F])) -> Result<(), Self::Error> {
        let bit = self.bit;
        let limit = 1_u32 << self.bits;
        let mut outside_range = false;
        self.source
            .for_each_vector(&mut |vector_index, values| {
                if outside_range {
                    return;
                }
                if values
                    .iter()
                    .any(|value| *value < 0 || (*value as u32) >= limit)
                {
                    outside_range = true;
                    return;
                }
                let field_values = values
                    .iter()
                    .map(|value| F::from((((*value as u32 >> bit) & 1) as i32) - 1))
                    .collect::<Vec<_>>();
                visit(vector_index, &field_values);
            })
            .map_err(BvaRangeSourceError::Source)?;
        if outside_range {
            Err(BvaRangeSourceError::ValueOutsideRange)
        } else {
            Ok(())
        }
    }
}

struct BitPairBvaFieldAdapter<'a, S, F> {
    source: &'a mut S,
    layout: BvaLayout,
    bit: usize,
    bits: usize,
    boolean_tbv: TensorBasisVector<F>,
    _field: PhantomData<F>,
}

impl<'a, S: BvaI32Source, F: PrimeField> BitPairBvaFieldAdapter<'a, S, F> {
    fn new(source: &'a mut S, bit: usize, bits: usize, boolean_tbv: TensorBasisVector<F>) -> Self {
        Self {
            layout: source.layout().clone(),
            source,
            bit,
            bits,
            boolean_tbv,
            _field: PhantomData,
        }
    }
}

impl<S, F> BvaFieldPairSource<F> for BitPairBvaFieldAdapter<'_, S, F>
where
    S: BvaI32Source,
    F: PrimeField + From<i32>,
{
    type Error = BvaRangeSourceError<S::Error>;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_vector_pair(
        &mut self,
        visit: &mut dyn FnMut(usize, &[F], &[F]),
    ) -> Result<(), Self::Error> {
        let bit = self.bit;
        let limit = 1_u32 << self.bits;
        let boolean_tbv = &self.boolean_tbv;
        let mut outside_range = false;
        self.source
            .for_each_vector(&mut |vector_index, values| {
                if outside_range {
                    return;
                }
                if values
                    .iter()
                    .any(|value| *value < 0 || (*value as u32) >= limit)
                {
                    outside_range = true;
                    return;
                }
                let left = values
                    .iter()
                    .map(|value| F::from(((*value as u32 >> bit) & 1) as i32))
                    .collect::<Vec<_>>();
                let mut right = left
                    .iter()
                    .map(|value| *value - F::one())
                    .collect::<Vec<_>>();
                if boolean_tbv
                    .scale_physical_vector_in_place(vector_index, &mut right)
                    .is_err()
                {
                    outside_range = true;
                    return;
                }
                visit(vector_index, &left, &right);
            })
            .map_err(BvaRangeSourceError::Source)?;
        if outside_range {
            Err(BvaRangeSourceError::ValueOutsideRange)
        } else {
            Ok(())
        }
    }
}

fn map_bva_range_pcs_error<SourceError>(
    error: StreamingVirtualPcsError<BvaRangeSourceError<SourceError>>,
) -> BvaRangeProofError<SourceError> {
    match error {
        StreamingVirtualPcsError::Pcs(error) => BvaRangeProofError::VirtualPcs(error),
        StreamingVirtualPcsError::Source(BvaRangeSourceError::Source(error)) => {
            BvaRangeProofError::Source(error)
        }
        StreamingVirtualPcsError::Source(BvaRangeSourceError::Bva(error)) => {
            BvaRangeProofError::Bva(error)
        }
        StreamingVirtualPcsError::Source(BvaRangeSourceError::ValueOutsideRange) => {
            BvaRangeProofError::ValueOutsideRange
        }
    }
}

fn map_plain_bva_range_pcs_error<SourceError>(
    error: StreamingVirtualPcsError<SourceError>,
) -> BvaRangeProofError<SourceError> {
    match error {
        StreamingVirtualPcsError::Pcs(error) => BvaRangeProofError::VirtualPcs(error),
        StreamingVirtualPcsError::Source(error) => BvaRangeProofError::Source(error),
    }
}

fn map_bva_range_eo_error<SourceError>(
    error: StreamingEoError<BvaRangeSourceError<SourceError>>,
) -> BvaRangeProofError<SourceError> {
    match error {
        StreamingEoError::Bva(error) => BvaRangeProofError::Bva(error),
        StreamingEoError::Source(BvaRangeSourceError::Source(error)) => {
            BvaRangeProofError::Source(error)
        }
        StreamingEoError::Source(BvaRangeSourceError::Bva(error)) => BvaRangeProofError::Bva(error),
        StreamingEoError::Source(BvaRangeSourceError::ValueOutsideRange) => {
            BvaRangeProofError::ValueOutsideRange
        }
    }
}

fn derive_sparse_bva_range_challenges<E: Pairing>(
    layer: usize,
    tag: usize,
    layout: &BvaLayout,
    value_commitment: PairingOutput<E>,
    bit_commitments: &[PairingOutput<E>],
    bits: usize,
) -> Vec<E::ScalarField> {
    let mut fs = FiatShamir::new();
    fs.push(&0x5350_4256_5241_4e47_u64);
    fs.push(&tag);
    fs.push(&layer);
    fs.push(&bits);
    fs.push(&layout.logical_len());
    fs.push(&value_commitment);
    let offsets = layout.bucket_offsets();
    for vector_index in 0..layout.vector_lengths().len() {
        fs.push(&layout.vector_lengths()[vector_index]);
        fs.push(&offsets[layout.vector_to_bucket()[vector_index]]);
    }
    for commitment in bit_commitments {
        fs.push(commitment);
    }
    (0..layout.logical_len().ilog2())
        .map(|_| fs.gen_challenge::<E::ScalarField>())
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn derive_sparse_bva_boolean_binding<E: Pairing>(
    layer: usize,
    tag: usize,
    layout: &BvaLayout,
    value_commitment: PairingOutput<E>,
    bit_commitment: PairingOutput<E>,
    bits: usize,
    bit: usize,
) -> E::ScalarField {
    let mut fs = FiatShamir::new();
    fs.push(&0x5350_4256_424f_554e_u64);
    fs.push(&tag);
    fs.push(&layer);
    fs.push(&bits);
    fs.push(&bit);
    fs.push(&layout.logical_len());
    fs.push(&value_commitment);
    fs.push(&bit_commitment);
    let offsets = layout.bucket_offsets();
    for vector_index in 0..layout.vector_lengths().len() {
        fs.push(&layout.vector_lengths()[vector_index]);
        fs.push(&offsets[layout.vector_to_bucket()[vector_index]]);
    }
    fs.gen_challenge::<E::ScalarField>()
}

#[allow(clippy::too_many_arguments)]
fn derive_sparse_bva_boolean_challenges<E: Pairing>(
    layer: usize,
    tag: usize,
    layout: &BvaLayout,
    value_commitment: PairingOutput<E>,
    bit_commitment: PairingOutput<E>,
    bits: usize,
    bit: usize,
) -> Vec<E::ScalarField> {
    let mut fs = FiatShamir::new();
    fs.push(&0x5350_4256_424f_4f4c_u64);
    fs.push(&tag);
    fs.push(&layer);
    fs.push(&bits);
    fs.push(&bit);
    fs.push(&layout.logical_len());
    fs.push(&value_commitment);
    fs.push(&bit_commitment);
    let offsets = layout.bucket_offsets();
    for vector_index in 0..layout.vector_lengths().len() {
        fs.push(&layout.vector_lengths()[vector_index]);
        fs.push(&offsets[layout.vector_to_bucket()[vector_index]]);
    }
    (0..layout.logical_len().ilog2())
        .map(|_| fs.gen_challenge::<E::ScalarField>())
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn derive_sparse_upper_bound_challenges<E: Pairing>(
    layer: usize,
    tag: usize,
    layout: &BvaLayout,
    value_commitment: PairingOutput<E>,
    slack_commitment: PairingOutput<E>,
    support_commitment: PairingOutput<E>,
    maximum: u64,
) -> Vec<E::ScalarField> {
    let mut fs = FiatShamir::new();
    fs.push(&0x5350_4256_5550_5045_u64);
    fs.push(&tag);
    fs.push(&layer);
    fs.push(&maximum);
    fs.push(&layout.logical_len());
    fs.push(&value_commitment);
    fs.push(&slack_commitment);
    fs.push(&support_commitment);
    let offsets = layout.bucket_offsets();
    for vector_index in 0..layout.vector_lengths().len() {
        fs.push(&layout.vector_lengths()[vector_index]);
        fs.push(&offsets[layout.vector_to_bucket()[vector_index]]);
    }
    (0..layout.logical_len().ilog2())
        .map(|_| fs.gen_challenge::<E::ScalarField>())
        .collect()
}

fn derive_bva_range_challenges<E: Pairing>(
    layer: usize,
    tag: usize,
    layout: &BvaLayout,
    value_commitment: PairingOutput<E>,
    bit_commitments: &[PairingOutput<E>],
    bit_minus_one_commitments: &[PairingOutput<E>],
    bits: usize,
) -> Vec<E::ScalarField> {
    let mut fs = FiatShamir::new();
    fs.push(&0x4256_4152_414e_4745_u64);
    fs.push(&tag);
    fs.push(&layer);
    fs.push(&bits);
    fs.push(&layout.logical_len());
    fs.push(&value_commitment);
    let offsets = layout.bucket_offsets();
    for vector_index in 0..layout.vector_lengths().len() {
        fs.push(&layout.vector_lengths()[vector_index]);
        fs.push(&offsets[layout.vector_to_bucket()[vector_index]]);
    }
    for (bit_commitment, bit_minus_one_commitment) in
        bit_commitments.iter().zip(bit_minus_one_commitments)
    {
        fs.push(bit_commitment);
        fs.push(bit_minus_one_commitment);
    }
    (0..layout.logical_len().ilog2())
        .map(|_| fs.gen_challenge::<E::ScalarField>())
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn derive_bva_boolean_binding<E: Pairing>(
    layer: usize,
    tag: usize,
    layout: &BvaLayout,
    value_commitment: PairingOutput<E>,
    bit_commitment: PairingOutput<E>,
    bit_minus_one_commitment: PairingOutput<E>,
    bits: usize,
    bit: usize,
) -> E::ScalarField {
    let mut fs = FiatShamir::new();
    fs.push(&0x4256_4142_4f4f_4c_u64);
    fs.push(&tag);
    fs.push(&layer);
    fs.push(&bits);
    fs.push(&bit);
    fs.push(&layout.logical_len());
    fs.push(&value_commitment);
    fs.push(&bit_commitment);
    fs.push(&bit_minus_one_commitment);
    let offsets = layout.bucket_offsets();
    for vector_index in 0..layout.vector_lengths().len() {
        fs.push(&layout.vector_lengths()[vector_index]);
        fs.push(&offsets[layout.vector_to_bucket()[vector_index]]);
    }
    fs.gen_challenge::<E::ScalarField>()
}

#[allow(clippy::too_many_arguments)]
fn derive_bva_boolean_challenges<E: Pairing>(
    layer: usize,
    tag: usize,
    layout: &BvaLayout,
    value_commitment: PairingOutput<E>,
    bit_commitment: PairingOutput<E>,
    bit_minus_one_commitment: PairingOutput<E>,
    bits: usize,
    bit: usize,
) -> Vec<E::ScalarField> {
    let mut fs = FiatShamir::new();
    fs.push(&0x4256_4142_4d4c_45_u64);
    fs.push(&tag);
    fs.push(&layer);
    fs.push(&bits);
    fs.push(&bit);
    fs.push(&layout.logical_len());
    fs.push(&value_commitment);
    fs.push(&bit_commitment);
    fs.push(&bit_minus_one_commitment);
    let offsets = layout.bucket_offsets();
    for vector_index in 0..layout.vector_lengths().len() {
        fs.push(&layout.vector_lengths()[vector_index]);
        fs.push(&offsets[layout.vector_to_bucket()[vector_index]]);
    }
    (0..layout.logical_len().ilog2())
        .map(|_| fs.gen_challenge::<E::ScalarField>())
        .collect()
}

fn verify_bva_vector_opening<E: Pairing>(
    srs: &SRS<E>,
    commitment: PairingOutput<E>,
    layout: &BvaLayout,
    challenges: &[E::ScalarField],
    proof: &BvaVectorOpeningProof<E>,
) -> bool {
    let Ok((xl, xr)) = bva_eo_opening_point(layout.logical_len(), challenges) else {
        return false;
    };
    proof.opening.has_rounds(xl.len(), xr.len())
        && SmartPC::<E>::verify(
            &srs.pc_pp,
            commitment,
            E::pairing(proof.value_commitment, srs.h_hat),
            &xl,
            &xr,
            &proof.opening,
        )
        .unwrap_or(false)
}

/// Proves that the independently range-checked chunk commitments encode the
/// same physical vectors as one canonical BVA commitment.  The proof contains
/// only random-point openings; the chunk values and all commitment randomness
/// remain prover-side.
pub struct BvaRangeBindingProof<E: Pairing> {
    pub global_value_commitment: E::G1,
    pub global_opening: PcsTrans<E>,
    pub chunk_openings: Vec<VectorOpeningProof<E>>,
}

impl<E: Pairing> BvaRangeBindingProof<E>
where
    E::ScalarField: From<i32>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn prove<S: BvaI32Source>(
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        source: &mut S,
        global_commitment: PairingOutput<E>,
        global_tier_one: &[E::G1],
        global_matrix_hiding: E::ScalarField,
        chunk_commitments: &[PairingOutput<E>],
        chunk_commitment_hidings: &[E::ScalarField],
        bit_width: usize,
    ) -> Result<Self, BvaRangeBindingError<S::Error>> {
        let layout = source.layout().clone();
        if chunk_commitments.len() != layout.vector_lengths().len()
            || chunk_commitment_hidings.len() != chunk_commitments.len()
        {
            return Err(BvaRangeBindingError::ChunkArityMismatch);
        }
        let challenges = derive_bva_binding_challenges::<E>(
            layer,
            tag,
            &layout,
            global_commitment,
            chunk_commitments,
        );
        let coordinates = bva_binding_coordinates(&layout, &challenges)
            .ok_or(BvaRangeBindingError::InvalidChunkLayout)?;
        let tbv = TensorBasisVector::global_tensor(&layout, &challenges)
            .map_err(BvaRangeBindingError::Bva)?;
        let global_value = {
            let mut adapter = I32BvaFieldAdapter::<S, E::ScalarField>::new(source);
            inner_product_bva_field_streaming(&mut adapter, &tbv)
                .map_err(BvaRangeBindingError::Pcs)?
                .value
        };
        let rng = &mut ark_std::rand::thread_rng();
        let global_evaluation_hiding = E::ScalarField::rand(rng);
        let global_value_commitment = srs.zksrs.commit(global_value, global_evaluation_hiding);
        let (global_xl, global_xr) = bva_eo_opening_point(layout.logical_len(), &challenges)
            .map_err(BvaRangeBindingError::VirtualPcs)?;
        let global_opening = {
            let mut adapter = I32BvaFieldAdapter::<S, E::ScalarField>::new(source);
            open_bva_field_streaming::<E, _>(
                &srs.pc_pp,
                &mut adapter,
                &global_xl,
                &global_xr,
                E::pairing(global_value_commitment, srs.h_hat),
                global_commitment,
                global_tier_one,
                global_evaluation_hiding,
                global_matrix_hiding,
            )
            .map_err(BvaRangeBindingError::Pcs)?
            .0
        };

        let pivot = coordinates
            .iter()
            .rposition(|(coefficient, _)| !coefficient.is_zero())
            .ok_or(BvaRangeBindingError::ZeroBindingCoefficient)?;
        let mut evaluation_hidings = (0..chunk_commitments.len())
            .map(|_| E::ScalarField::rand(rng))
            .collect::<Vec<_>>();
        let accumulated = coordinates
            .iter()
            .zip(&evaluation_hidings)
            .enumerate()
            .filter(|(index, _)| *index != pivot)
            .map(|(_, ((coefficient, _), hiding))| *coefficient * hiding)
            .sum::<E::ScalarField>();
        evaluation_hidings[pivot] = (global_evaluation_hiding - accumulated)
            * coordinates[pivot]
                .0
                .inverse()
                .ok_or(BvaRangeBindingError::ZeroBindingCoefficient)?;

        let mut chunk_openings = (0..chunk_commitments.len())
            .map(|_| None)
            .collect::<Vec<Option<VectorOpeningProof<E>>>>();
        let mut visited = vec![false; chunk_commitments.len()];
        let mut binding_error = None;
        source
            .for_each_vector(&mut |vector_index, values| {
                if binding_error.is_some() {
                    return;
                }
                if vector_index >= visited.len() {
                    binding_error = Some(BvaRangeBindingError::ChunkIndex(vector_index));
                    return;
                }
                if visited[vector_index] {
                    binding_error = Some(BvaRangeBindingError::DuplicateChunk(vector_index));
                    return;
                }
                visited[vector_index] = true;
                let witness = match CommittedVectorWitness::commit(
                    srs,
                    values.to_vec(),
                    chunk_commitment_hidings[vector_index],
                    bit_width,
                ) {
                    Ok(witness) => witness,
                    Err(error) => {
                        binding_error = Some(BvaRangeBindingError::Vector(error));
                        return;
                    }
                };
                if witness.commitment != chunk_commitments[vector_index] {
                    binding_error =
                        Some(BvaRangeBindingError::ChunkCommitmentMismatch(vector_index));
                    return;
                }
                let point = &coordinates[vector_index].1;
                let value = match witness.value_at(point) {
                    Ok(value) => value,
                    Err(error) => {
                        binding_error = Some(BvaRangeBindingError::Vector(error));
                        return;
                    }
                };
                let value_hiding = evaluation_hidings[vector_index];
                let value_commitment = srs.zksrs.commit(value, value_hiding);
                match witness.open_at(srs, point, value_commitment, value_hiding) {
                    Ok(opening) => chunk_openings[vector_index] = Some(opening),
                    Err(error) => binding_error = Some(BvaRangeBindingError::Vector(error)),
                }
            })
            .map_err(|error| BvaRangeBindingError::Pcs(StreamingVirtualPcsError::Source(error)))?;
        if let Some(error) = binding_error {
            return Err(error);
        }
        if let Some(index) = visited.iter().position(|visited| !visited) {
            return Err(BvaRangeBindingError::MissingChunk(index));
        }
        let chunk_openings = chunk_openings
            .into_iter()
            .enumerate()
            .map(|(index, opening)| opening.ok_or(BvaRangeBindingError::MissingChunk(index)))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            global_value_commitment,
            global_opening,
            chunk_openings,
        })
    }

    pub fn verify(
        &self,
        srs: &SRS<E>,
        layer: usize,
        tag: usize,
        layout: &BvaLayout,
        global_commitment: PairingOutput<E>,
        chunk_commitments: &[PairingOutput<E>],
    ) -> bool {
        if chunk_commitments.len() != layout.vector_lengths().len()
            || self.chunk_openings.len() != chunk_commitments.len()
        {
            return false;
        }
        let challenges = derive_bva_binding_challenges::<E>(
            layer,
            tag,
            layout,
            global_commitment,
            chunk_commitments,
        );
        let Some(coordinates) = bva_binding_coordinates(layout, &challenges) else {
            return false;
        };
        let Ok((global_xl, global_xr)) = bva_eo_opening_point(layout.logical_len(), &challenges)
        else {
            return false;
        };
        if !self
            .global_opening
            .has_rounds(global_xl.len(), global_xr.len())
            || !SmartPC::<E>::verify(
                &srs.pc_pp,
                global_commitment,
                E::pairing(self.global_value_commitment, srs.h_hat),
                &global_xl,
                &global_xr,
                &self.global_opening,
            )
            .unwrap_or(false)
        {
            return false;
        }
        let recombined = coordinates
            .iter()
            .zip(&self.chunk_openings)
            .map(|((coefficient, _), opening)| opening.value_commitment.mul(coefficient))
            .sum::<E::G1>();
        recombined == self.global_value_commitment
            && chunk_commitments
                .iter()
                .zip(&self.chunk_openings)
                .zip(coordinates)
                .all(|((commitment, opening), (_, point))| {
                    verify_vector_opening(srs, *commitment, &point, opening)
                })
    }

    pub fn serialized_size_bytes(&self) -> usize {
        self.global_value_commitment.serialized_size(Compress::Yes)
            + self.global_opening.serialized_size(Compress::Yes)
            + self
                .chunk_openings
                .iter()
                .map(|opening| {
                    opening.value_commitment.serialized_size(Compress::Yes)
                        + opening.opening.serialized_size(Compress::Yes)
                })
                .sum::<usize>()
    }
}

struct I32BvaFieldAdapter<'a, S, F> {
    source: &'a mut S,
    layout: BvaLayout,
    _field: PhantomData<F>,
}

impl<'a, S: BvaI32Source, F> I32BvaFieldAdapter<'a, S, F> {
    fn new(source: &'a mut S) -> Self {
        Self {
            layout: source.layout().clone(),
            source,
            _field: PhantomData,
        }
    }
}

impl<S, F> BvaFieldSource<F> for I32BvaFieldAdapter<'_, S, F>
where
    S: BvaI32Source,
    F: ark_ff::PrimeField + From<i32>,
{
    type Error = S::Error;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[F])) -> Result<(), Self::Error> {
        self.source.for_each_vector(&mut |vector_index, values| {
            let field_values = values.iter().copied().map(F::from).collect::<Vec<_>>();
            visit(vector_index, &field_values);
        })
    }
}

fn derive_bva_binding_challenges<E: Pairing>(
    layer: usize,
    tag: usize,
    layout: &BvaLayout,
    global_commitment: PairingOutput<E>,
    chunk_commitments: &[PairingOutput<E>],
) -> Vec<E::ScalarField> {
    let mut fs = FiatShamir::new();
    fs.push(&0x4256_4152_usize);
    fs.push(&tag);
    fs.push(&layer);
    fs.push(&layout.logical_len());
    fs.push(&global_commitment);
    let offsets = layout.bucket_offsets();
    for (vector_index, commitment) in chunk_commitments.iter().enumerate() {
        fs.push(&layout.vector_lengths()[vector_index]);
        fs.push(&offsets[layout.vector_to_bucket()[vector_index]]);
        fs.push(commitment);
    }
    (0..layout.logical_len().ilog2())
        .map(|_| fs.gen_challenge::<E::ScalarField>())
        .collect()
}

fn bva_binding_coordinates<F: Field>(
    layout: &BvaLayout,
    challenges: &[F],
) -> Option<Vec<(F, Vec<F>)>> {
    if challenges.len() != layout.logical_len().ilog2() as usize {
        return None;
    }
    let offsets = layout.bucket_offsets();
    layout
        .vector_lengths()
        .iter()
        .enumerate()
        .map(|(vector_index, &width)| {
            if !width.is_power_of_two() {
                return None;
            }
            let interior_log = width.ilog2() as usize;
            let bucket = layout.vector_to_bucket()[vector_index];
            if layout.bucket_sizes()[bucket] != width {
                return None;
            }
            let offset = offsets[bucket];
            let mut coefficient = F::one();
            for (bit, challenge) in challenges.iter().enumerate().skip(interior_log) {
                if (offset >> bit) & 1 == 1 {
                    coefficient *= challenge;
                }
            }
            let point = challenges[..interior_log].iter().rev().copied().collect();
            Some((coefficient, point))
        })
        .collect()
}

#[derive(Debug)]
pub enum BvaRangeBindingError<SourceError> {
    Bva(BvaError),
    VirtualPcs(VirtualPcsError),
    Pcs(StreamingVirtualPcsError<SourceError>),
    Vector(VectorProofError),
    ChunkArityMismatch,
    InvalidChunkLayout,
    ZeroBindingCoefficient,
    ChunkIndex(usize),
    DuplicateChunk(usize),
    MissingChunk(usize),
    ChunkCommitmentMismatch(usize),
}

impl<SourceError: Display + std::fmt::Debug> Display for BvaRangeBindingError<SourceError> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl<SourceError: Display + std::fmt::Debug> std::error::Error
    for BvaRangeBindingError<SourceError>
{
}

fn hadamard_transcript_len(width: usize) -> usize {
    // LiteBullet: public commitment/size (2), folding rounds (3 each),
    // terminal commitments (2), and the scalar multiplication proof (13).
    3 * width.ilog2() as usize + 17
}

fn boolcheck_transcript_len(width: usize) -> usize {
    // BoolCheck samples log(width) coordinates, runs one Hadamard proof,
    // then batches two openings with a coin, LiteBullet, and Schnorr (5).
    7 * width.ilog2() as usize + 40
}

fn check_linear_inputs<E: Pairing>(
    tensors: &[&CommittedVectorWitness<E>],
    coefficients: &[E::ScalarField],
) -> Result<(), VectorProofError> {
    if tensors.is_empty() || tensors.len() != coefficients.len() {
        return Err(VectorProofError::LinearArityMismatch);
    }
    let width = tensors[0].values.len();
    if tensors.iter().any(|tensor| tensor.values.len() != width) {
        return Err(VectorProofError::VectorLengthMismatch);
    }
    Ok(())
}

fn derive_vector_point<E: Pairing>(
    tag: usize,
    layer: usize,
    width: usize,
    commitments: &[PairingOutput<E>],
) -> Vec<E::ScalarField> {
    let mut fs = FiatShamir::new();
    fs.push(&0x4d4c50_usize);
    fs.push(&tag);
    fs.push(&layer);
    fs.push(&width);
    for commitment in commitments {
        fs.push(commitment);
    }
    (0..width.ilog2())
        .map(|_| fs.gen_challenge::<E::ScalarField>())
        .collect()
}

fn verify_vector_opening<E: Pairing>(
    srs: &SRS<E>,
    commitment: PairingOutput<E>,
    logical_point: &[E::ScalarField],
    proof: &VectorOpeningProof<E>,
) -> bool {
    let packed_point =
        linear::reshape_points_keep_projection(&(logical_point.to_vec(), Vec::new()));
    if !proof
        .opening
        .has_rounds(packed_point.0.len(), packed_point.1.len())
    {
        return false;
    }
    SmartPC::<E>::verify(
        &srs.pc_pp,
        commitment,
        E::pairing(proof.value_commitment, srs.h_hat),
        &packed_point.0,
        &packed_point.1,
        &proof.opening,
    )
    .unwrap_or(false)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum VectorProofError {
    Matrix(HeteroMatrixError),
    Pcs(VirtualPcsError),
    InvalidVectorLength(usize),
    VectorLengthMismatch,
    LinearArityMismatch,
    ZeroFinalCoefficient,
    InvalidRangeBits(usize),
    ValueOutsideRange,
}

impl From<HeteroMatrixError> for VectorProofError {
    fn from(error: HeteroMatrixError) -> Self {
        Self::Matrix(error)
    }
}

impl From<VirtualPcsError> for VectorProofError {
    fn from(error: VirtualPcsError) -> Self {
        Self::Pcs(error)
    }
}

impl Display for VectorProofError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for VectorProofError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::bucketed_ip::BvaLayout;
    use crate::protocols::virtualpcs::{
        commit_bva_field_streaming, commit_bva_i32_streaming, commit_sparse_bva_field_streaming,
        BvaFieldSource, BvaI32Source, SparseBvaFieldSource,
    };
    use ark_bls12_381::{Bls12_381, Fr};

    #[derive(Clone)]
    struct TestBvaI32Source {
        layout: BvaLayout,
        vectors: Vec<Vec<i32>>,
        reverse: bool,
    }

    impl TestBvaI32Source {
        fn new(vectors: Vec<Vec<i32>>) -> Self {
            let lengths = vectors.iter().map(Vec::len).collect::<Vec<_>>();
            Self {
                layout: BvaLayout::from_lengths(&lengths).unwrap(),
                vectors,
                reverse: false,
            }
        }

        fn reversed(mut self) -> Self {
            self.reverse = true;
            self
        }
    }

    impl BvaI32Source for TestBvaI32Source {
        type Error = std::convert::Infallible;

        fn layout(&self) -> &BvaLayout {
            &self.layout
        }

        fn for_each_vector(
            &mut self,
            visit: &mut dyn FnMut(usize, &[i32]),
        ) -> Result<(), Self::Error> {
            if self.reverse {
                for index in (0..self.vectors.len()).rev() {
                    visit(index, &self.vectors[index]);
                }
            } else {
                for (index, vector) in self.vectors.iter().enumerate() {
                    visit(index, vector);
                }
            }
            Ok(())
        }
    }

    #[derive(Clone)]
    struct TestSparseBvaU64Source {
        layout: BvaLayout,
        entries: Vec<(usize, usize, u64)>,
    }

    impl SparseBvaU64Source for TestSparseBvaU64Source {
        type Error = std::convert::Infallible;

        fn layout(&self) -> &BvaLayout {
            &self.layout
        }

        fn for_each_nonzero_u64(
            &mut self,
            visit: &mut dyn FnMut(usize, usize, u64),
        ) -> Result<(), Self::Error> {
            for &(vector_index, entry_index, value) in &self.entries {
                visit(vector_index, entry_index, value);
            }
            Ok(())
        }
    }

    impl SparseBvaFieldSource<Fr> for TestSparseBvaU64Source {
        type Error = std::convert::Infallible;

        fn layout(&self) -> &BvaLayout {
            &self.layout
        }

        fn for_each_nonzero(
            &mut self,
            visit: &mut dyn FnMut(usize, usize, Fr),
        ) -> Result<(), Self::Error> {
            for &(vector_index, entry_index, value) in &self.entries {
                visit(vector_index, entry_index, Fr::from(value));
            }
            Ok(())
        }
    }

    #[derive(Clone)]
    struct TestBvaFieldSource {
        layout: BvaLayout,
        vectors: Vec<Vec<Fr>>,
    }

    impl TestBvaFieldSource {
        fn new(vectors: Vec<Vec<Fr>>) -> Self {
            let lengths = vectors.iter().map(Vec::len).collect::<Vec<_>>();
            Self {
                layout: BvaLayout::from_lengths(&lengths).unwrap(),
                vectors,
            }
        }
    }

    impl BvaFieldSource<Fr> for TestBvaFieldSource {
        type Error = std::convert::Infallible;

        fn layout(&self) -> &BvaLayout {
            &self.layout
        }

        fn for_each_vector(
            &mut self,
            visit: &mut dyn FnMut(usize, &[Fr]),
        ) -> Result<(), Self::Error> {
            for (index, vector) in self.vectors.iter().enumerate() {
                visit(index, vector);
            }
            Ok(())
        }
    }

    struct TestBvaAffineRelationSource {
        layout: BvaLayout,
        factor: Vec<Vec<Fr>>,
        value: Vec<Vec<Fr>>,
        counter: Vec<Vec<Fr>>,
    }

    impl BvaAffineRelationSource<Fr> for TestBvaAffineRelationSource {
        type Error = std::convert::Infallible;

        fn layout(&self) -> &BvaLayout {
            &self.layout
        }

        fn for_each_vector(
            &mut self,
            visit: &mut dyn FnMut(usize, &[Fr], &[Fr], &[Fr]),
        ) -> Result<(), Self::Error> {
            for index in 0..self.factor.len() {
                visit(
                    index,
                    &self.factor[index],
                    &self.value[index],
                    &self.counter[index],
                );
            }
            Ok(())
        }
    }

    #[test]
    fn bva_affine_relation_binds_factor_value_and_counter_commitments() {
        type E = Bls12_381;
        let value = vec![
            vec![Fr::from(2_u64), Fr::from(3_u64), Fr::from(5_u64)],
            vec![Fr::from(7_u64)],
        ];
        let counter = vec![
            vec![Fr::from(1_u64), Fr::from(0_u64), Fr::from(4_u64)],
            vec![Fr::from(2_u64)],
        ];
        let constant = Fr::from(3_u64);
        let value_coefficient = Fr::from(5_u64);
        let counter_coefficient = Fr::from(7_u64);
        let factor = value
            .iter()
            .zip(&counter)
            .map(|(values, counters)| {
                values
                    .iter()
                    .zip(counters)
                    .map(|(value, counter)| {
                        constant + value_coefficient * value + counter_coefficient * counter
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let layout = BvaLayout::from_lengths(&[3, 1]).unwrap();
        let srs = SRS::<E>::setup(2);
        let factor_hiding = Fr::from(149_u64);
        let value_hiding = Fr::from(151_u64);
        let counter_hiding = Fr::from(157_u64);

        let mut factor_commit_source = TestBvaFieldSource::new(factor.clone());
        let (factor_commitment, factor_tier_one, _) = commit_bva_field_streaming::<E, _>(
            &srs.pc_pp,
            &mut factor_commit_source,
            factor_hiding,
        )
        .unwrap();
        let mut value_commit_source = TestBvaFieldSource::new(value.clone());
        let (value_commitment, value_tier_one, _) =
            commit_bva_field_streaming::<E, _>(&srs.pc_pp, &mut value_commit_source, value_hiding)
                .unwrap();
        let mut counter_commit_source = TestBvaFieldSource::new(counter.clone());
        let (counter_commitment, counter_tier_one, _) = commit_bva_field_streaming::<E, _>(
            &srs.pc_pp,
            &mut counter_commit_source,
            counter_hiding,
        )
        .unwrap();
        let mut source = TestBvaAffineRelationSource {
            layout: layout.clone(),
            factor,
            value,
            counter,
        };
        let (mut proof, profile) = BvaAffineRelationProof::prove(
            &srs,
            13,
            0x4146_4649,
            &mut source,
            factor_commitment,
            &factor_tier_one,
            factor_hiding,
            value_commitment,
            &value_tier_one,
            value_hiding,
            counter_commitment,
            &counter_tier_one,
            counter_hiding,
            constant,
            value_coefficient,
            counter_coefficient,
        )
        .unwrap();

        assert!(proof.verify(
            &srs,
            13,
            0x4146_4649,
            &layout,
            factor_commitment,
            value_commitment,
            counter_commitment,
            constant,
            value_coefficient,
            counter_coefficient,
        ));
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(profile.padding_field_operations, 0);
        assert!(profile.maximum_live_vector_entries <= 3);

        let commitment_delta = E::pairing(srs.pc_pp.g_0, srs.h_hat);
        assert!(!proof.verify(
            &srs,
            13,
            0x4146_4649,
            &layout,
            factor_commitment + commitment_delta,
            value_commitment,
            counter_commitment,
            constant,
            value_coefficient,
            counter_coefficient,
        ));
        assert!(!proof.verify(
            &srs,
            13,
            0x4146_4649,
            &layout,
            factor_commitment,
            value_commitment + commitment_delta,
            counter_commitment,
            constant,
            value_coefficient,
            counter_coefficient,
        ));
        assert!(!proof.verify(
            &srs,
            13,
            0x4146_4649,
            &layout,
            factor_commitment,
            value_commitment,
            counter_commitment + commitment_delta,
            constant,
            value_coefficient,
            counter_coefficient,
        ));
        assert!(!proof.verify(
            &srs,
            13,
            0x4146_4649,
            &layout,
            factor_commitment,
            value_commitment,
            counter_commitment,
            constant + Fr::from(1_u64),
            value_coefficient,
            counter_coefficient,
        ));
        proof.factor_opening.value_commitment += srs.pc_pp.g_0;
        assert!(!proof.verify(
            &srs,
            13,
            0x4146_4649,
            &layout,
            factor_commitment,
            value_commitment,
            counter_commitment,
            constant,
            value_coefficient,
            counter_coefficient,
        ));
    }

    #[test]
    fn bva_routed_equality_binds_auxiliary_component_to_canonical_commitment() {
        type E = Bls12_381;
        let target = vec![vec![Fr::from(2_u64), Fr::from(4_u64), Fr::from(6_u64)]];
        let source = vec![
            vec![Fr::from(3_u64), Fr::from(5_u64), Fr::from(7_u64)],
            vec![Fr::from(0_u64)],
        ];
        let source_layout = BvaLayout::from_lengths(&[3, 1]).unwrap();
        let target_layout = BvaLayout::from_lengths(&[3]).unwrap();
        let routes = vec![Some((0, Fr::from(1_u64))), None];
        let public_constants = vec![Fr::from(1_u64), Fr::from(0_u64)];
        let srs = SRS::<E>::setup(2);
        let source_hiding = Fr::from(163_u64);
        let target_hiding = Fr::from(167_u64);
        let mut source_commit_source = TestBvaFieldSource::new(source.clone());
        let (source_commitment, source_tier_one, _) = commit_bva_field_streaming::<E, _>(
            &srs.pc_pp,
            &mut source_commit_source,
            source_hiding,
        )
        .unwrap();
        let mut target_commit_source = TestBvaFieldSource::new(target.clone());
        let (target_commitment, target_tier_one, _) = commit_bva_field_streaming::<E, _>(
            &srs.pc_pp,
            &mut target_commit_source,
            target_hiding,
        )
        .unwrap();
        let mut source = TestBvaFieldSource::new(source);
        let mut target = TestBvaFieldSource::new(target);
        let (mut proof, profile) = BvaRoutedEqualityProof::prove(
            &srs,
            17,
            0x5254_4551,
            &mut source,
            source_commitment,
            &source_tier_one,
            source_hiding,
            &mut target,
            target_commitment,
            &target_tier_one,
            target_hiding,
            &routes,
            &public_constants,
        )
        .unwrap();

        assert!(proof.verify(
            &srs,
            17,
            0x5254_4551,
            &source_layout,
            source_commitment,
            &target_layout,
            target_commitment,
            &routes,
            &public_constants,
        ));
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(profile.padding_field_operations, 0);
        assert!(profile.maximum_live_vector_entries <= 3);

        let commitment_delta = E::pairing(srs.pc_pp.g_0, srs.h_hat);
        assert!(!proof.verify(
            &srs,
            17,
            0x5254_4551,
            &source_layout,
            source_commitment + commitment_delta,
            &target_layout,
            target_commitment,
            &routes,
            &public_constants,
        ));
        assert!(!proof.verify(
            &srs,
            17,
            0x5254_4551,
            &source_layout,
            source_commitment,
            &target_layout,
            target_commitment + commitment_delta,
            &routes,
            &public_constants,
        ));
        let mut wrong_constants = public_constants.clone();
        wrong_constants[0] += Fr::from(1_u64);
        assert!(!proof.verify(
            &srs,
            17,
            0x5254_4551,
            &source_layout,
            source_commitment,
            &target_layout,
            target_commitment,
            &routes,
            &wrong_constants,
        ));
        proof.target_claim += Fr::from(1_u64);
        assert!(!proof.verify(
            &srs,
            17,
            0x5254_4551,
            &source_layout,
            source_commitment,
            &target_layout,
            target_commitment,
            &routes,
            &public_constants,
        ));
    }

    #[test]
    fn bva_committed_evaluation_binds_an_arbitrary_public_tbv_claim() {
        type E = Bls12_381;
        let vectors = vec![
            vec![Fr::from(2_u64), Fr::from(3_u64), Fr::from(5_u64)],
            vec![Fr::from(7_u64)],
        ];
        let layout = BvaLayout::from_lengths(&[3, 1]).unwrap();
        let tbv = TensorBasisVector::global_tensor(
            &layout,
            &[Fr::from(11_u64), Fr::from(13_u64), Fr::from(17_u64)],
        )
        .unwrap();
        let srs = SRS::<E>::setup(2);
        let hiding = Fr::from(173_u64);
        let mut commit_source = TestBvaFieldSource::new(vectors.clone());
        let (commitment, tier_one, commit_profile) =
            commit_bva_field_streaming::<E, _>(&srs.pc_pp, &mut commit_source, hiding).unwrap();
        assert_eq!(commit_profile.materialized_padding_entries, 0);
        let mut source = TestBvaFieldSource::new(vectors);
        let claim = inner_product_bva_field_streaming(&mut source, &tbv)
            .unwrap()
            .value;
        let relation_digest = [23_u8; 32];
        let (mut proof, profile) = BvaCommittedEvaluationProof::prove(
            &srs,
            19,
            0x4345_5641,
            relation_digest,
            &mut source,
            commitment,
            &tier_one,
            hiding,
            &tbv,
            claim,
        )
        .unwrap();

        assert!(proof.verify(
            &srs,
            19,
            0x4345_5641,
            relation_digest,
            &layout,
            commitment,
            &tbv,
            claim,
        ));
        assert!(proof.verify_consuming_tbv(
            &srs,
            19,
            0x4345_5641,
            relation_digest,
            &layout,
            commitment,
            tbv.clone(),
            claim,
        ));
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(profile.padding_field_operations, 0);
        assert!(profile.maximum_live_vector_entries <= 3);

        let commitment_delta = E::pairing(srs.pc_pp.g_0, srs.h_hat);
        assert!(!proof.verify(
            &srs,
            19,
            0x4345_5641,
            relation_digest,
            &layout,
            commitment + commitment_delta,
            &tbv,
            claim,
        ));
        assert!(!proof.verify(
            &srs,
            19,
            0x4345_5641,
            [29_u8; 32],
            &layout,
            commitment,
            &tbv,
            claim,
        ));
        assert!(!proof.verify(
            &srs,
            19,
            0x4345_5641,
            relation_digest,
            &layout,
            commitment,
            &tbv,
            claim + Fr::from(1_u64),
        ));
        proof.opening.value_commitment += srs.pc_pp.g_0;
        assert!(!proof.verify(
            &srs,
            19,
            0x4345_5641,
            relation_digest,
            &layout,
            commitment,
            &tbv,
            claim,
        ));
    }

    #[test]
    fn sparse_bva_range_proof_never_materializes_zero_table_rows() {
        type E = Bls12_381;
        let layout = BvaLayout::from_lengths(&[33_694_208]).unwrap();
        let mut source = TestSparseBvaU64Source {
            layout: layout.clone(),
            entries: vec![(0, 5, 2), (0, 123, 3), (0, 33_694_207, 7)],
        };
        let srs = SRS::<E>::setup(13);
        let value_hiding = Fr::from(131_u64);
        let (value_commitment, value_tier_one, commit_profile) =
            commit_sparse_bva_field_streaming::<E, _>(&srs.pc_pp, &mut source, value_hiding)
                .unwrap();
        assert_eq!(commit_profile.nonzero_entries, 3);
        assert_eq!(commit_profile.materialized_padding_entries, 0);

        let (mut proof, profile) = SparseBvaRangeProof::prove(
            &srs,
            9,
            0x5350_5253,
            &mut source,
            value_commitment,
            &value_tier_one,
            value_hiding,
            4,
        )
        .unwrap();
        assert!(proof.verify(&srs, 9, 0x5350_5253, &layout, value_commitment, 4));
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(profile.padding_field_operations, 0);
        assert_eq!(profile.padding_group_operations, 0);
        assert!(profile.maximum_live_vector_entries <= 6);

        let commitment_delta = E::pairing(srs.pc_pp.g_0, srs.h_hat);
        proof.bit_commitments[0] += commitment_delta;
        assert!(!proof.verify(&srs, 9, 0x5350_5253, &layout, value_commitment, 4));
        proof.bit_commitments[0] -= commitment_delta;
        proof.boolean_proofs[0].eo_proof.a_reduced += Fr::from(1_u64);
        assert!(!proof.verify(&srs, 9, 0x5350_5253, &layout, value_commitment, 4));

        let mut out_of_range = TestSparseBvaU64Source {
            layout,
            entries: vec![(0, 5, 16)],
        };
        assert!(matches!(
            SparseBvaRangeProof::prove(
                &srs,
                9,
                0x5350_5253,
                &mut out_of_range,
                value_commitment,
                &value_tier_one,
                value_hiding,
                4,
            ),
            Err(BvaRangeProofError::ValueOutsideRange)
        ));
    }

    #[test]
    fn sparse_bva_upper_bound_proves_exact_public_limit() {
        type E = Bls12_381;
        let layout = BvaLayout::from_lengths(&[17]).unwrap();
        let mut source = TestSparseBvaU64Source {
            layout: layout.clone(),
            entries: vec![(0, 1, 2), (0, 8, 3), (0, 16, 7)],
        };
        let srs = SRS::<E>::setup(3);
        let value_hiding = Fr::from(137_u64);
        let (value_commitment, value_tier_one, _) =
            commit_sparse_bva_field_streaming::<E, _>(&srs.pc_pp, &mut source, value_hiding)
                .unwrap();
        let (mut proof, profile) = SparseBvaUpperBoundProof::prove(
            &srs,
            11,
            0x5350_5542,
            &mut source,
            value_commitment,
            &value_tier_one,
            value_hiding,
            7,
        )
        .unwrap();
        assert!(proof.verify(&srs, 11, 0x5350_5542, &layout, value_commitment, 7));
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(profile.padding_field_operations, 0);
        assert_eq!(profile.padding_group_operations, 0);
        assert!(!proof.verify(&srs, 11, 0x5350_5542, &layout, value_commitment, 6));

        let commitment_delta = E::pairing(srs.pc_pp.g_0, srs.h_hat);
        proof.slack_commitment += commitment_delta;
        assert!(!proof.verify(&srs, 11, 0x5350_5542, &layout, value_commitment, 7));

        let mut out_of_range = TestSparseBvaU64Source {
            layout,
            entries: vec![(0, 1, 8)],
        };
        assert!(matches!(
            SparseBvaUpperBoundProof::prove(
                &srs,
                11,
                0x5350_5542,
                &mut out_of_range,
                value_commitment,
                &value_tier_one,
                value_hiding,
                7,
            ),
            Err(BvaRangeProofError::ValueOutsideRange)
        ));
    }

    #[test]
    fn bva_range_binding_verifies_without_values_and_rejects_commitment_or_opening_tampering() {
        type E = Bls12_381;
        let srs = SRS::<Bls12_381>::setup(3);
        let vectors = vec![vec![3, 1, 4, 1], vec![5, 9], vec![2]];
        let mut source = TestBvaI32Source::new(vectors.clone());
        let layout = source.layout().clone();
        let chunk_hidings = vec![Fr::from(11_u64), Fr::from(13_u64), Fr::from(17_u64)];
        let chunks = vectors
            .into_iter()
            .zip(chunk_hidings.iter().copied())
            .enumerate()
            .map(|(index, (values, hiding))| {
                BoundedRangeChunkProof::prove_with_hiding(
                    &srs,
                    5,
                    0x4256_0000 + index,
                    values,
                    4,
                    hiding,
                )
                .unwrap()
            })
            .collect::<Vec<_>>();
        let chunk_commitments = chunks
            .iter()
            .map(|proof| proof.value_commitment)
            .collect::<Vec<_>>();
        let global_hiding = Fr::from(19_u64);
        let (global_commitment, global_tier_one, _) =
            commit_bva_i32_streaming::<E, _>(&srs.pc_pp, &mut source, global_hiding, 4).unwrap();
        let mut binding = BvaRangeBindingProof::prove(
            &srs,
            5,
            0x4256,
            &mut source,
            global_commitment,
            &global_tier_one,
            global_hiding,
            &chunk_commitments,
            &chunk_hidings,
            4,
        )
        .unwrap();

        assert!(binding.verify(
            &srs,
            5,
            0x4256,
            &layout,
            global_commitment,
            &chunk_commitments,
        ));

        let mut reversed_source = source.clone().reversed();
        let reversed_binding = BvaRangeBindingProof::prove(
            &srs,
            5,
            0x4256,
            &mut reversed_source,
            global_commitment,
            &global_tier_one,
            global_hiding,
            &chunk_commitments,
            &chunk_hidings,
            4,
        )
        .unwrap();
        assert!(reversed_binding.verify(
            &srs,
            5,
            0x4256,
            &layout,
            global_commitment,
            &chunk_commitments,
        ));

        let commitment_delta = E::pairing(srs.pc_pp.g_0, srs.h_hat);
        let mut tampered_chunks = chunk_commitments.clone();
        tampered_chunks[0] += commitment_delta;
        assert!(!binding.verify(
            &srs,
            5,
            0x4256,
            &layout,
            global_commitment,
            &tampered_chunks,
        ));

        binding.global_value_commitment += srs.zksrs.com_base;
        assert!(!binding.verify(
            &srs,
            5,
            0x4256,
            &layout,
            global_commitment,
            &chunk_commitments,
        ));
        binding.global_value_commitment -= srs.zksrs.com_base;
        binding.chunk_openings[0].value_commitment += srs.zksrs.com_base;
        assert!(!binding.verify(
            &srs,
            5,
            0x4256,
            &layout,
            global_commitment,
            &chunk_commitments,
        ));
        binding.chunk_openings[0].value_commitment -= srs.zksrs.com_base;
        assert!(binding.verify(
            &srs,
            5,
            0x4256,
            &layout,
            global_commitment,
            &chunk_commitments,
        ));

        let reordered_layout = BvaLayout::from_lengths(&[2, 4, 1]).unwrap();
        assert!(!binding.verify(
            &srs,
            5,
            0x4256,
            &reordered_layout,
            global_commitment,
            &chunk_commitments,
        ));

        let mut mismatched_source = source.clone();
        mismatched_source.vectors[0][0] += 1;
        assert!(matches!(
            BvaRangeBindingProof::prove(
                &srs,
                5,
                0x4256,
                &mut mismatched_source,
                global_commitment,
                &global_tier_one,
                global_hiding,
                &chunk_commitments,
                &chunk_hidings,
                4,
            ),
            Err(BvaRangeBindingError::ChunkCommitmentMismatch(0))
        ));
    }

    #[test]
    fn bva_range_proof_binds_one_global_commitment_without_materializing_padding() {
        type E = Bls12_381;
        let srs = SRS::<E>::setup(3);
        let vectors = vec![vec![0, 1, 3, 7], vec![2, 5], vec![6]];
        let mut source = TestBvaI32Source::new(vectors);
        let layout = source.layout().clone();
        let global_hiding = Fr::from(19_u64);
        let (global_commitment, global_tier_one, commit_profile) =
            commit_bva_i32_streaming::<E, _>(&srs.pc_pp, &mut source, global_hiding, 3).unwrap();
        assert_eq!(commit_profile.materialized_padding_entries, 0);

        let (mut proof, profile) = BvaRangeProof::prove(
            &srs,
            5,
            0x4256_4152,
            &mut source,
            global_commitment,
            &global_tier_one,
            global_hiding,
            3,
        )
        .unwrap();

        assert!(proof.verify(&srs, 5, 0x4256_4152, &layout, global_commitment, 3,));
        assert_eq!(profile.materialized_padding_entries, 0);
        assert_eq!(profile.padding_field_operations, 0);
        assert_eq!(profile.padding_group_operations, 0);
        assert_eq!(profile.maximum_live_vector_entries, 8);
        assert!(proof.serialized_size_bytes() > 0);

        let commitment_delta = E::pairing(srs.pc_pp.g_0, srs.h_hat);
        proof.bit_commitments[0] += commitment_delta;
        assert!(!proof.verify(&srs, 5, 0x4256_4152, &layout, global_commitment, 3,));
        proof.bit_commitments[0] -= commitment_delta;

        proof.boolean_proofs[0].bit_minus_one_commitment += commitment_delta;
        assert!(!proof.verify(&srs, 5, 0x4256_4152, &layout, global_commitment, 3,));
        proof.boolean_proofs[0].bit_minus_one_commitment -= commitment_delta;

        proof.boolean_proofs[0].eo_proof.a_reduced += Fr::from(1_u64);
        assert!(!proof.verify(&srs, 5, 0x4256_4152, &layout, global_commitment, 3,));
        proof.boolean_proofs[0].eo_proof.a_reduced -= Fr::from(1_u64);

        proof.boolean_proofs[0]
            .bit_minus_one_eo_opening
            .value_commitment += srs.zksrs.com_base;
        assert!(!proof.verify(&srs, 5, 0x4256_4152, &layout, global_commitment, 3,));
        proof.boolean_proofs[0]
            .bit_minus_one_eo_opening
            .value_commitment -= srs.zksrs.com_base;

        proof.value_linear_opening.value_commitment += srs.zksrs.com_base;
        assert!(!proof.verify(&srs, 5, 0x4256_4152, &layout, global_commitment, 3,));
        proof.value_linear_opening.value_commitment -= srs.zksrs.com_base;
        assert!(proof.verify(&srs, 5, 0x4256_4152, &layout, global_commitment, 3,));

        let reordered_layout = BvaLayout::from_lengths(&[2, 4, 1]).unwrap();
        assert!(!proof.verify(
            &srs,
            5,
            0x4256_4152,
            &reordered_layout,
            global_commitment,
            3,
        ));
        assert!(!proof.verify(&srs, 5, 0x4256_4152, &layout, global_commitment, 2,));

        let mut out_of_range = TestBvaI32Source::new(vec![vec![0, 8], vec![1]]);
        let out_of_range_layout = out_of_range.layout().clone();
        let out_of_range_hiding = Fr::from(23_u64);
        let (out_of_range_commitment, out_of_range_tier_one, _) =
            commit_bva_i32_streaming::<E, _>(&srs.pc_pp, &mut out_of_range, out_of_range_hiding, 4)
                .unwrap();
        assert_ne!(out_of_range_layout, layout);
        assert!(matches!(
            BvaRangeProof::prove(
                &srs,
                5,
                0x4256_4152,
                &mut out_of_range,
                out_of_range_commitment,
                &out_of_range_tier_one,
                out_of_range_hiding,
                3,
            ),
            Err(BvaRangeProofError::ValueOutsideRange)
        ));
    }

    #[test]
    fn bounded_range_chunk_verifies_without_witness_and_rejects_tampering() {
        let srs = SRS::<Bls12_381>::setup(3);
        let mut proof =
            BoundedRangeChunkProof::prove(&srs, 7, 19, vec![0, 1, 2, 7, 8, 15, 3, 4], 4).unwrap();

        assert!(proof.verify(&srs, 7, 19));
        assert!(!proof.verify(&srs, 7, 20));

        let commitment_delta = Bls12_381::pairing(srs.pc_pp.g_0, srs.h_hat);
        proof.value_commitment += commitment_delta;
        assert!(!proof.verify(&srs, 7, 19));
        proof.value_commitment -= commitment_delta;

        proof.range_proof.value_linear_opening.value_commitment += srs.zksrs.com_base;
        assert!(!proof.verify(&srs, 7, 19));
        proof.range_proof.value_linear_opening.value_commitment -= srs.zksrs.com_base;

        proof.range_proof.boolean_proofs[0].transcript.pointer = 1;
        assert!(!proof.verify(&srs, 7, 19));
        proof.range_proof.boolean_proofs[0].transcript.pointer = 0;
        assert!(proof.verify(&srs, 7, 19));

        assert!(matches!(
            BoundedRangeChunkProof::<Bls12_381>::prove(
                &srs,
                7,
                19,
                vec![0, 1, 2, 7, 8, 15, 3, 4],
                3,
            ),
            Err(VectorProofError::ValueOutsideRange)
        ));
    }
}
