use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::UniformRand;
use std::convert::TryFrom;
use std::fmt::{Display, Formatter};

use crate::data_structures::SRS;
use crate::hdr1cs::heterogeneous::{
    HeteroMulBucketHidings, HeteroMulError, HeteroMulInstance, HeteroMulProof, HeteroMulProtocol,
    HeteroMulWitness,
};
use crate::protocols::vector_relation::{
    CommittedVectorWitness, RangeProof, VectorProofError, ZeroHadamardProof, ZeroLinearProof,
};
use crate::utils::heterogeneous::{
    HeteroBlockMat, HeteroMatrixError, MatrixBlock, MatrixDims, PaddedDomain,
};

const ADDITION_TAG: usize = 0x414444;
const RELU_LINEAR_TAG: usize = 0x524c4c;
const RELU_PRODUCT_TAG: usize = 0x524c50;
const RELU_OUTPUT_RANGE_TAG: usize = 0x524f52;
const RELU_NEGATIVE_RANGE_TAG: usize = 0x524e52;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MlpManifest {
    pub widths: Vec<usize>,
    pub commitment_bit_width: usize,
    pub relu_magnitude_bits: usize,
}

impl MlpManifest {
    pub fn validate(&self) -> Result<(), MlpProofError> {
        if self.widths.len() < 2 {
            return Err(MlpProofError::NoLayers);
        }
        if self
            .widths
            .iter()
            .any(|width| *width == 0 || !width.is_power_of_two())
        {
            return Err(MlpProofError::NonPowerOfTwoWidth);
        }
        if self.commitment_bit_width == 0 || self.commitment_bit_width > 32 {
            return Err(MlpProofError::InvalidCommitmentBitWidth(
                self.commitment_bit_width,
            ));
        }
        if self.relu_magnitude_bits == 0 || self.relu_magnitude_bits > 30 {
            return Err(MlpProofError::InvalidReluMagnitudeBits(
                self.relu_magnitude_bits,
            ));
        }
        Ok(())
    }

    pub fn layers(&self) -> usize {
        self.widths.len() - 1
    }
}

pub struct MlpLayer {
    pub weight: MatrixBlock<i32>,
    pub bias: Vec<i32>,
}

pub struct MlpLayerInstance<E: Pairing> {
    pub multiplication: HeteroMulInstance<E>,
    pub bias: PairingOutput<E>,
    pub preactivation: PairingOutput<E>,
    pub activation: PairingOutput<E>,
    pub negative: PairingOutput<E>,
}

pub struct MlpInstance<E: Pairing> {
    pub manifest: MlpManifest,
    pub public_input: Vec<i32>,
    pub public_output: Vec<i32>,
    pub layers: Vec<MlpLayerInstance<E>>,
}

pub struct MlpLayerProof<E: Pairing> {
    pub multiplication: HeteroMulProof<E>,
    pub addition: ZeroLinearProof<E>,
    pub relu_linear: ZeroLinearProof<E>,
    pub relu_product: ZeroHadamardProof<E>,
    pub activation_range: RangeProof<E>,
    pub negative_range: RangeProof<E>,
}

pub struct MlpProof<E: Pairing> {
    pub layers: Vec<MlpLayerProof<E>>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct MlpRamProfile {
    pub total_weight_entries: usize,
    pub max_live_weight_entries: usize,
    pub total_activation_entries: usize,
    pub range_bit_entries: usize,
}

impl MlpRamProfile {
    pub fn for_manifest(manifest: &MlpManifest) -> Result<Self, MlpProofError> {
        manifest.validate()?;
        let mut total_weight_entries = 0_usize;
        let mut max_live_weight_entries = 0_usize;
        let mut total_activation_entries = manifest.widths[0];
        for widths in manifest.widths.windows(2) {
            let entries = widths[0]
                .checked_mul(widths[1])
                .ok_or(MlpProofError::SizeOverflow)?;
            total_weight_entries = total_weight_entries
                .checked_add(entries)
                .ok_or(MlpProofError::SizeOverflow)?;
            max_live_weight_entries = max_live_weight_entries.max(entries);
            total_activation_entries = total_activation_entries
                .checked_add(widths[1])
                .ok_or(MlpProofError::SizeOverflow)?;
        }
        let range_bit_entries = manifest.widths[1..]
            .iter()
            .try_fold(0_usize, |total, width| {
                total.checked_add(
                    width
                        .checked_mul(manifest.relu_magnitude_bits)
                        .and_then(|entries| entries.checked_mul(2))?,
                )
            })
            .ok_or(MlpProofError::SizeOverflow)?;
        Ok(Self {
            total_weight_entries,
            max_live_weight_entries,
            total_activation_entries,
            range_bit_entries,
        })
    }
}

pub struct HeteroMlpProtocol<E: Pairing> {
    _marker: std::marker::PhantomData<E>,
}

impl<E: Pairing> HeteroMlpProtocol<E>
where
    E::ScalarField: From<i32>,
{
    pub fn setup(manifest: &MlpManifest) -> Result<SRS<E>, MlpProofError> {
        manifest.validate()?;
        let weight_axes = manifest
            .widths
            .windows(2)
            .map(|widths| {
                widths[0]
                    .checked_mul(widths[1])
                    .map(packed_max_axis)
                    .ok_or(MlpProofError::SizeOverflow)
            })
            .collect::<Result<Vec<_>, _>>()?;
        let maximum_axis = weight_axes
            .into_iter()
            .chain(manifest.widths.iter().copied().map(packed_max_axis))
            .max()
            .unwrap_or(2)
            .max(2);
        Ok(SRS::<E>::setup(maximum_axis.ilog2() as usize))
    }

    pub fn prove<I>(
        srs: &SRS<E>,
        manifest: MlpManifest,
        public_input: Vec<i32>,
        expected_public_output: Vec<i32>,
        layers: I,
    ) -> Result<(MlpInstance<E>, MlpProof<E>), MlpProofError>
    where
        I: IntoIterator<Item = MlpLayer>,
    {
        manifest.validate()?;
        if public_input.len() != manifest.widths[0] {
            return Err(MlpProofError::InputWidthMismatch);
        }
        if expected_public_output.len() != *manifest.widths.last().unwrap() {
            return Err(MlpProofError::OutputWidthMismatch);
        }

        let rng = &mut ark_std::rand::thread_rng();
        let mut layer_source = layers.into_iter();
        let mut current = CommittedVectorWitness::<E>::commit(
            srs,
            public_input.clone(),
            E::ScalarField::from(0_i32),
            manifest.commitment_bit_width,
        )?;
        let mut layer_instances = Vec::with_capacity(manifest.layers());
        let mut layer_proofs = Vec::with_capacity(manifest.layers());

        for (layer_index, widths) in manifest.widths.windows(2).enumerate() {
            let layer = layer_source
                .next()
                .ok_or(MlpProofError::LayerCountMismatch)?;
            let input_width = widths[0];
            let output_width = widths[1];
            validate_layer(layer_index, &layer, input_width, output_width)?;
            let product = matrix_vector_product(&layer.weight, &current.values)?;
            let preactivation = add_vectors(&product, &layer.bias)?;
            let activation = preactivation
                .iter()
                .map(|value| (*value).max(0))
                .collect::<Vec<_>>();
            let negative = preactivation
                .iter()
                .map(|value| value.checked_neg().unwrap_or(i32::MIN).max(0))
                .collect::<Vec<_>>();
            if preactivation
                .iter()
                .zip(activation.iter().zip(&negative))
                .any(|(z, (y, n))| *z != *y - *n)
            {
                return Err(MlpProofError::ArithmeticOverflow);
            }

            let product_witness = CommittedVectorWitness::<E>::commit(
                srs,
                product.clone(),
                E::ScalarField::rand(rng),
                manifest.commitment_bit_width,
            )?;
            let bias_witness = CommittedVectorWitness::<E>::commit(
                srs,
                layer.bias.clone(),
                E::ScalarField::rand(rng),
                manifest.commitment_bit_width,
            )?;
            let preactivation_witness = CommittedVectorWitness::<E>::commit(
                srs,
                preactivation,
                E::ScalarField::rand(rng),
                manifest.commitment_bit_width,
            )?;
            let is_last = layer_index + 1 == manifest.layers();
            let activation_hiding = if is_last {
                E::ScalarField::from(0_i32)
            } else {
                E::ScalarField::rand(rng)
            };
            let activation_witness = CommittedVectorWitness::<E>::commit(
                srs,
                activation,
                activation_hiding,
                manifest.commitment_bit_width,
            )?;
            let negative_witness = CommittedVectorWitness::<E>::commit(
                srs,
                negative,
                E::ScalarField::rand(rng),
                manifest.commitment_bit_width,
            )?;

            let multiplication_witness = single_mul_witness(
                layer.weight,
                current.values.clone(),
                product,
                input_width,
                output_width,
            )?;
            let (multiplication_instance, multiplication_proof) =
                HeteroMulProtocol::<E>::prove_owned_with_hidings(
                    srs,
                    multiplication_witness,
                    manifest.commitment_bit_width,
                    vec![HeteroMulBucketHidings {
                        left: E::ScalarField::rand(rng),
                        right: current.hiding,
                        output: product_witness.hiding,
                    }],
                )?;
            if multiplication_instance.buckets.len() != 1
                || multiplication_instance.buckets[0].right != current.commitment
                || multiplication_instance.buckets[0].output != product_witness.commitment
            {
                return Err(MlpProofError::CommitmentLinkMismatch);
            }

            let addition = ZeroLinearProof::prove(
                srs,
                layer_index,
                ADDITION_TAG,
                &[&product_witness, &bias_witness, &preactivation_witness],
                &[
                    E::ScalarField::from(1_i32),
                    E::ScalarField::from(1_i32),
                    E::ScalarField::from(-1_i32),
                ],
            )?;
            let relu_linear = ZeroLinearProof::prove(
                srs,
                layer_index,
                RELU_LINEAR_TAG,
                &[
                    &preactivation_witness,
                    &activation_witness,
                    &negative_witness,
                ],
                &[
                    E::ScalarField::from(1_i32),
                    E::ScalarField::from(-1_i32),
                    E::ScalarField::from(1_i32),
                ],
            )?;
            let relu_product = ZeroHadamardProof::prove(
                srs,
                layer_index,
                RELU_PRODUCT_TAG,
                &activation_witness,
                &negative_witness,
            )?;
            let activation_range = RangeProof::prove(
                srs,
                layer_index,
                RELU_OUTPUT_RANGE_TAG,
                &activation_witness,
                manifest.relu_magnitude_bits,
            )?;
            let negative_range = RangeProof::prove(
                srs,
                layer_index,
                RELU_NEGATIVE_RANGE_TAG,
                &negative_witness,
                manifest.relu_magnitude_bits,
            )?;

            layer_instances.push(MlpLayerInstance {
                multiplication: multiplication_instance,
                bias: bias_witness.commitment,
                preactivation: preactivation_witness.commitment,
                activation: activation_witness.commitment,
                negative: negative_witness.commitment,
            });
            layer_proofs.push(MlpLayerProof {
                multiplication: multiplication_proof,
                addition,
                relu_linear,
                relu_product,
                activation_range,
                negative_range,
            });
            current = activation_witness;
        }
        if layer_source.next().is_some() {
            return Err(MlpProofError::LayerCountMismatch);
        }
        if current.values != expected_public_output {
            return Err(MlpProofError::UnexpectedPublicOutput);
        }

        Ok((
            MlpInstance {
                manifest,
                public_input,
                public_output: expected_public_output,
                layers: layer_instances,
            },
            MlpProof {
                layers: layer_proofs,
            },
        ))
    }

    pub fn verify(srs: &SRS<E>, instance: &MlpInstance<E>, proof: &MlpProof<E>) -> bool {
        if instance.manifest.validate().is_err()
            || instance.layers.len() != instance.manifest.layers()
            || proof.layers.len() != instance.manifest.layers()
            || instance.public_input.len() != instance.manifest.widths[0]
            || instance.public_output.len() != *instance.manifest.widths.last().unwrap()
        {
            return false;
        }
        let mut expected_input = match CommittedVectorWitness::<E>::commit(
            srs,
            instance.public_input.clone(),
            E::ScalarField::from(0_i32),
            instance.manifest.commitment_bit_width,
        ) {
            Ok(input) => input.commitment,
            Err(_) => return false,
        };

        for (layer_index, ((widths, layer), layer_proof)) in instance
            .manifest
            .widths
            .windows(2)
            .zip(&instance.layers)
            .zip(&proof.layers)
            .enumerate()
        {
            if layer.multiplication.buckets.len() != 1
                || layer.multiplication.shape.gates.len() != 1
                || layer.multiplication.buckets[0].right != expected_input
                || !shape_matches(&layer.multiplication, widths[0], widths[1])
                || !HeteroMulProtocol::<E>::verify(
                    srs,
                    &layer.multiplication,
                    &layer_proof.multiplication,
                )
            {
                return false;
            }
            let product = layer.multiplication.buckets[0].output;
            let addition_commitments = [product, layer.bias, layer.preactivation];
            if !layer_proof.addition.verify(
                srs,
                layer_index,
                ADDITION_TAG,
                widths[1],
                &addition_commitments,
                &[
                    E::ScalarField::from(1_i32),
                    E::ScalarField::from(1_i32),
                    E::ScalarField::from(-1_i32),
                ],
            ) {
                return false;
            }
            let relu_commitments = [layer.preactivation, layer.activation, layer.negative];
            if !layer_proof.relu_linear.verify(
                srs,
                layer_index,
                RELU_LINEAR_TAG,
                widths[1],
                &relu_commitments,
                &[
                    E::ScalarField::from(1_i32),
                    E::ScalarField::from(-1_i32),
                    E::ScalarField::from(1_i32),
                ],
            ) || !layer_proof.relu_product.verify(
                srs,
                layer_index,
                RELU_PRODUCT_TAG,
                widths[1],
                layer.activation,
                layer.negative,
            ) || !layer_proof.activation_range.verify(
                srs,
                layer_index,
                RELU_OUTPUT_RANGE_TAG,
                widths[1],
                layer.activation,
                instance.manifest.relu_magnitude_bits,
            ) || !layer_proof.negative_range.verify(
                srs,
                layer_index,
                RELU_NEGATIVE_RANGE_TAG,
                widths[1],
                layer.negative,
                instance.manifest.relu_magnitude_bits,
            ) {
                return false;
            }
            expected_input = layer.activation;
        }

        let public_output = match CommittedVectorWitness::<E>::commit(
            srs,
            instance.public_output.clone(),
            E::ScalarField::from(0_i32),
            instance.manifest.commitment_bit_width,
        ) {
            Ok(output) => output.commitment,
            Err(_) => return false,
        };
        expected_input == public_output
    }
}

fn single_mul_witness(
    weight: MatrixBlock<i32>,
    input: Vec<i32>,
    output: Vec<i32>,
    input_width: usize,
    output_width: usize,
) -> Result<HeteroMulWitness, MlpProofError> {
    let left = HeteroBlockMat::new(
        PaddedDomain::new(1, output_width, input_width)?,
        vec![Some(weight)],
    )?;
    let right = HeteroBlockMat::new(
        PaddedDomain::new(1, input_width, 1)?,
        vec![Some(MatrixBlock::from_columns(
            MatrixDims::new(input_width, 1)?,
            vec![input],
        )?)],
    )?;
    let output = HeteroBlockMat::new(
        PaddedDomain::new(1, output_width, 1)?,
        vec![Some(MatrixBlock::from_columns(
            MatrixDims::new(output_width, 1)?,
            vec![output],
        )?)],
    )?;
    Ok(HeteroMulWitness::new(left, right, output)?)
}

fn validate_layer(
    layer_index: usize,
    layer: &MlpLayer,
    input_width: usize,
    output_width: usize,
) -> Result<(), MlpProofError> {
    if layer.weight.dims != MatrixDims::new(output_width, input_width)? {
        return Err(MlpProofError::LayerShapeMismatch { layer: layer_index });
    }
    layer.weight.validate_layout()?;
    if layer.bias.len() != output_width {
        return Err(MlpProofError::BiasWidthMismatch { layer: layer_index });
    }
    Ok(())
}

fn matrix_vector_product(
    weight: &MatrixBlock<i32>,
    input: &[i32],
) -> Result<Vec<i32>, MlpProofError> {
    if weight.dims.cols != input.len() {
        return Err(MlpProofError::InputWidthMismatch);
    }
    let mut output = vec![0_i64; weight.dims.rows];
    for (column, input_value) in weight.columns.iter().zip(input) {
        for (accumulator, weight_value) in output.iter_mut().zip(column) {
            *accumulator = accumulator
                .checked_add((*weight_value as i64) * (*input_value as i64))
                .ok_or(MlpProofError::ArithmeticOverflow)?;
        }
    }
    output
        .into_iter()
        .map(|value| i32::try_from(value).map_err(|_| MlpProofError::ArithmeticOverflow))
        .collect()
}

fn add_vectors(left: &[i32], right: &[i32]) -> Result<Vec<i32>, MlpProofError> {
    if left.len() != right.len() {
        return Err(MlpProofError::VectorLengthMismatch);
    }
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            left.checked_add(*right)
                .ok_or(MlpProofError::ArithmeticOverflow)
        })
        .collect()
}

fn shape_matches<E: Pairing>(
    instance: &HeteroMulInstance<E>,
    input_width: usize,
    output_width: usize,
) -> bool {
    let Some(Some(gate)) = instance.shape.gates.first() else {
        return false;
    };
    gate.left
        == MatrixDims {
            rows: output_width,
            cols: input_width,
        }
        && gate.right
            == MatrixDims {
                rows: input_width,
                cols: 1,
            }
        && gate.output
            == MatrixDims {
                rows: output_width,
                cols: 1,
            }
}

fn packed_max_axis(entries: usize) -> usize {
    let log_entries = entries.ilog2() as usize;
    let columns = 1 << (log_entries / 2);
    (entries / columns).max(columns)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum MlpProofError {
    Matrix(HeteroMatrixError),
    Multiplication(HeteroMulError),
    Vector(VectorProofError),
    NoLayers,
    NonPowerOfTwoWidth,
    InvalidCommitmentBitWidth(usize),
    InvalidReluMagnitudeBits(usize),
    InputWidthMismatch,
    OutputWidthMismatch,
    LayerCountMismatch,
    LayerShapeMismatch { layer: usize },
    BiasWidthMismatch { layer: usize },
    VectorLengthMismatch,
    ArithmeticOverflow,
    CommitmentLinkMismatch,
    UnexpectedPublicOutput,
    SizeOverflow,
}

impl From<HeteroMatrixError> for MlpProofError {
    fn from(error: HeteroMatrixError) -> Self {
        Self::Matrix(error)
    }
}

impl From<HeteroMulError> for MlpProofError {
    fn from(error: HeteroMulError) -> Self {
        Self::Multiplication(error)
    }
}

impl From<VectorProofError> for MlpProofError {
    fn from(error: VectorProofError) -> Self {
        Self::Vector(error)
    }
}

impl Display for MlpProofError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for MlpProofError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::zktr::TranElem;
    use ark_bls12_381::Bls12_381;
    use ark_std::Zero;

    fn manifest() -> MlpManifest {
        MlpManifest {
            widths: vec![4, 2, 1],
            commitment_bit_width: 32,
            relu_magnitude_bits: 4,
        }
    }

    fn layers() -> Vec<MlpLayer> {
        vec![
            MlpLayer {
                weight: MatrixBlock::from_columns(
                    MatrixDims::new(2, 4).unwrap(),
                    vec![vec![1, 0], vec![0, 1], vec![1, 1], vec![0, -1]],
                )
                .unwrap(),
                bias: vec![-5, 2],
            },
            MlpLayer {
                weight: MatrixBlock::from_columns(
                    MatrixDims::new(1, 2).unwrap(),
                    vec![vec![2], vec![-1]],
                )
                .unwrap(),
                bias: vec![3],
            },
        ]
    }

    fn prove() -> (SRS<Bls12_381>, MlpInstance<Bls12_381>, MlpProof<Bls12_381>) {
        let manifest = manifest();
        let srs = HeteroMlpProtocol::<Bls12_381>::setup(&manifest).unwrap();
        let (instance, proof) = HeteroMlpProtocol::<Bls12_381>::prove(
            &srs,
            manifest,
            vec![1, -2, 3, 1],
            vec![1],
            layers(),
        )
        .unwrap();
        (srs, instance, proof)
    }

    #[test]
    fn end_to_end_relu_mlp_accepts_one_shared_activation_chain() {
        let (srs, instance, proof) = prove();

        assert!(HeteroMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof
        ));
        assert_eq!(
            instance.layers[0].activation,
            instance.layers[1].multiplication.buckets[0].right
        );
    }

    #[test]
    fn end_to_end_relu_mlp_rejects_public_output_tampering() {
        let (srs, mut instance, proof) = prove();
        instance.public_output[0] = 2;
        assert!(!HeteroMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof
        ));
    }

    #[test]
    fn end_to_end_relu_mlp_rejects_bias_commitment_tampering() {
        let (srs, mut instance, proof) = prove();
        instance.layers[0].bias = PairingOutput::<Bls12_381>::zero();
        assert!(!HeteroMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof
        ));
    }

    #[test]
    fn end_to_end_relu_mlp_rejects_truncated_activation_transcript() {
        let (srs, instance, mut proof) = prove();
        proof.layers[0].relu_product.transcript.data.pop();
        assert!(!HeteroMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof
        ));
    }

    #[test]
    fn end_to_end_relu_mlp_rejects_missing_layer_proof() {
        let (srs, instance, mut proof) = prove();
        proof.layers.pop();
        assert!(!HeteroMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof
        ));
    }

    #[test]
    fn end_to_end_relu_mlp_rejects_cross_layer_link_tampering() {
        let (srs, mut instance, proof) = prove();
        instance.layers[1].multiplication.buckets[0].right = PairingOutput::zero();
        assert!(!HeteroMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof
        ));
    }

    #[test]
    fn end_to_end_relu_mlp_rejects_activation_commitment_tampering() {
        let (srs, mut instance, proof) = prove();
        instance.layers[0].activation = PairingOutput::zero();
        assert!(!HeteroMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof
        ));
    }

    #[test]
    fn end_to_end_relu_mlp_rejects_trailing_transcript_data() {
        let (srs, instance, mut proof) = prove();
        proof.layers[0]
            .relu_product
            .transcript
            .data
            .push(TranElem::Size(1));
        assert!(!HeteroMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof
        ));
    }

    #[test]
    fn end_to_end_relu_mlp_rejects_nonzero_transcript_pointer() {
        let (srs, instance, mut proof) = prove();
        proof.layers[0].activation_range.boolean_proofs[0]
            .transcript
            .pointer = 1;
        assert!(!HeteroMlpProtocol::<Bls12_381>::verify(
            &srs, &instance, &proof
        ));
    }
}
