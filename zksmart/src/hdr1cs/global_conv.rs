use std::{
    convert::TryFrom,
    fmt::{Display, Formatter},
    marker::PhantomData,
    time::Instant,
};

use ark_bls12_381::{Bls12_381, Fr};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ff::{Field, One, UniformRand, Zero};
use ark_poly_commit::smart_pc::{
    data_structures::Trans as PcsTrans, SmartPC, UniversalParams as PcsParams,
};
use ark_std::{ops::Mul, rand::RngCore};
use sha2::{Digest, Sha256};

use crate::protocols::{
    bucketed_ip::{
        BucketedVector, BvaFieldSource, BvaLayout, EoLiteBulletProof, EoWitnessPairProof,
        TbvBucket, TensorBasisVector,
    },
    virtualpcs::{
        bva_eo_opening_point, commit_bva_i32_streaming, inner_product_bva_field_streaming,
        open_bva_field_streaming, BvaI32Source,
    },
};
use crate::utils::fiat_shamir::FiatShamir;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Conv2dShape {
    pub input_h: usize,
    pub input_w: usize,
    pub kernel_h: usize,
    pub kernel_w: usize,
    pub in_channels: usize,
    pub out_channels: usize,
}

impl Conv2dShape {
    pub fn new(
        input_h: usize,
        input_w: usize,
        kernel_h: usize,
        kernel_w: usize,
        in_channels: usize,
        out_channels: usize,
    ) -> Result<Self, GlobalConvError> {
        let shape = Self {
            input_h,
            input_w,
            kernel_h,
            kernel_w,
            in_channels,
            out_channels,
        };
        shape.validate()?;
        Ok(shape)
    }

    pub fn validate(self) -> Result<(), GlobalConvError> {
        if [
            self.input_h,
            self.input_w,
            self.kernel_h,
            self.kernel_w,
            self.in_channels,
            self.out_channels,
        ]
        .contains(&0)
            || !self.input_h.is_power_of_two()
            || !self.input_w.is_power_of_two()
            || !self.out_channels.is_power_of_two()
        {
            return Err(GlobalConvError::InvalidShape);
        }
        self.input_len()?;
        self.kernel_len()?;
        self.output_len()?;
        Ok(())
    }

    pub fn output_h(self) -> Result<usize, GlobalConvError> {
        self.input_h
            .checked_add(self.kernel_h)
            .and_then(|value| value.checked_sub(1))
            .ok_or(GlobalConvError::ArithmeticOverflow)
    }

    pub fn output_w(self) -> Result<usize, GlobalConvError> {
        self.input_w
            .checked_add(self.kernel_w)
            .and_then(|value| value.checked_sub(1))
            .ok_or(GlobalConvError::ArithmeticOverflow)
    }

    pub fn input_len(self) -> Result<usize, GlobalConvError> {
        checked_product(&[self.in_channels, self.input_h, self.input_w])
    }

    pub fn kernel_len(self) -> Result<usize, GlobalConvError> {
        checked_product(&[
            self.in_channels,
            self.out_channels,
            self.kernel_h,
            self.kernel_w,
        ])
    }

    pub fn output_len(self) -> Result<usize, GlobalConvError> {
        checked_product(&[self.out_channels, self.output_h()?, self.output_w()?])
    }
}

fn checked_product(values: &[usize]) -> Result<usize, GlobalConvError> {
    values.iter().try_fold(1_usize, |product, value| {
        product
            .checked_mul(*value)
            .ok_or(GlobalConvError::ArithmeticOverflow)
    })
}

pub fn full_cross_correlation(
    shape: Conv2dShape,
    input: &[i32],
    kernel: &[i32],
) -> Result<Vec<i32>, GlobalConvError> {
    shape.validate()?;
    if input.len() != shape.input_len()? {
        return Err(GlobalConvError::InputLength);
    }
    if kernel.len() != shape.kernel_len()? {
        return Err(GlobalConvError::KernelLength);
    }
    let output_h = shape.output_h()?;
    let output_w = shape.output_w()?;
    let input_plane = checked_product(&[shape.input_h, shape.input_w])?;
    let kernel_plane = checked_product(&[shape.kernel_h, shape.kernel_w])?;
    let output_plane = checked_product(&[output_h, output_w])?;
    let mut output = vec![0_i32; shape.output_len()?];
    for output_channel in 0..shape.out_channels {
        for output_row in 0..output_h {
            for output_column in 0..output_w {
                let mut accumulator = 0_i64;
                for input_channel in 0..shape.in_channels {
                    for kernel_row in 0..shape.kernel_h {
                        for kernel_column in 0..shape.kernel_w {
                            let input_row = output_row as isize - (shape.kernel_h as isize - 1)
                                + kernel_row as isize;
                            let input_column = output_column as isize
                                - (shape.kernel_w as isize - 1)
                                + kernel_column as isize;
                            if input_row < 0
                                || input_row >= shape.input_h as isize
                                || input_column < 0
                                || input_column >= shape.input_w as isize
                            {
                                continue;
                            }
                            let input_index = input_channel * input_plane
                                + input_row as usize * shape.input_w
                                + input_column as usize;
                            let kernel_index = (input_channel * shape.out_channels
                                + output_channel)
                                * kernel_plane
                                + kernel_row * shape.kernel_w
                                + kernel_column;
                            accumulator = accumulator
                                .checked_add(
                                    i64::from(input[input_index])
                                        .checked_mul(i64::from(kernel[kernel_index]))
                                        .ok_or(GlobalConvError::ArithmeticOverflow)?,
                                )
                                .ok_or(GlobalConvError::ArithmeticOverflow)?;
                        }
                    }
                }
                output[output_channel * output_plane + output_row * output_w + output_column] =
                    i32::try_from(accumulator).map_err(|_| GlobalConvError::ArithmeticOverflow)?;
            }
        }
    }
    Ok(output)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlobalConvPublicLayer {
    pub shape: Conv2dShape,
    pub input: Vec<i32>,
    pub output: Vec<i32>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlobalConvStatement {
    pub layers: Vec<GlobalConvPublicLayer>,
    pub commitment_bit_width: usize,
}

impl GlobalConvStatement {
    pub fn new(
        layers: Vec<GlobalConvPublicLayer>,
        commitment_bit_width: usize,
    ) -> Result<Self, GlobalConvError> {
        let statement = Self {
            layers,
            commitment_bit_width,
        };
        statement.validate()?;
        Ok(statement)
    }

    pub fn validate(&self) -> Result<(), GlobalConvError> {
        if self.layers.is_empty()
            || self.commitment_bit_width == 0
            || self.commitment_bit_width > 32
        {
            return Err(GlobalConvError::InvalidStatement);
        }
        for layer in &self.layers {
            layer.shape.validate()?;
            if layer.input.len() != layer.shape.input_len()?
                || layer.output.len() != layer.shape.output_len()?
            {
                return Err(GlobalConvError::InvalidStatement);
            }
        }
        Ok(())
    }

    pub fn transcript_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(b"zksmart-global-hd-r1cs-conv-v1");
        hasher.update((self.commitment_bit_width as u64).to_le_bytes());
        hasher.update((self.layers.len() as u64).to_le_bytes());
        for layer in &self.layers {
            for value in [
                layer.shape.input_h,
                layer.shape.input_w,
                layer.shape.kernel_h,
                layer.shape.kernel_w,
                layer.shape.in_channels,
                layer.shape.out_channels,
            ] {
                hasher.update((value as u64).to_le_bytes());
            }
            for value in layer.input.iter().chain(&layer.output) {
                hasher.update(value.to_le_bytes());
            }
        }
        hasher.finalize().into()
    }
}

pub struct GlobalConvParameters {
    pub pc_pp: PcsParams<Bls12_381>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlobalConvInstance {
    pub statement_digest: [u8; 32],
    pub global_witness_commitment: PairingOutput<Bls12_381>,
    pub model_commitment: PairingOutput<Bls12_381>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct GlobalConvProof {
    pub statement_digest: [u8; 32],
    pub conv_claim: Fr,
    pub conv_proof: EoWitnessPairProof<Fr>,
    pub model_claim: Fr,
    pub model_eo: EoLiteBulletProof<Fr>,
    pub model_value_commitment: PairingOutput<Bls12_381>,
    pub model_opening: PcsTrans<Bls12_381>,
    pub synchronized_eo: EoLiteBulletProof<Fr>,
    pub synchronized_value_commitment: PairingOutput<Bls12_381>,
    pub synchronized_opening: PcsTrans<Bls12_381>,
}

impl GlobalConvProof {
    pub fn witness_commitment_count(&self) -> usize {
        1
    }

    pub fn witness_opening_count(&self) -> usize {
        1
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct GlobalConvProverProfile {
    pub witness_physical_entries: usize,
    pub witness_logical_entries: usize,
    pub model_physical_entries: usize,
    pub model_logical_entries: usize,
    pub pair_physical_entries: usize,
    pub pair_logical_entries: usize,
    pub materialized_padding_entries: usize,
    pub witness_commit_seconds: f64,
    pub model_commit_seconds: f64,
    pub convolution_reduction_seconds: f64,
    pub model_binding_seconds: f64,
    pub synchronized_reduction_seconds: f64,
    pub opening_seconds: f64,
}

pub struct GlobalHdR1csConvProtocol {
    _marker: PhantomData<Bls12_381>,
}

#[derive(Clone, Debug)]
struct ConvLayerMap {
    input_start: usize,
    kernel_start: usize,
    output_start: usize,
}

#[derive(Clone, Debug)]
struct ConvLayouts {
    witness: BvaLayout,
    model: BvaLayout,
    pair: BvaLayout,
    public_io: BvaLayout,
    layers: Vec<ConvLayerMap>,
    model_to_witness: Vec<usize>,
    public_io_to_witness: Vec<usize>,
}

fn compile_layouts(statement: &GlobalConvStatement) -> Result<ConvLayouts, GlobalConvError> {
    statement.validate()?;
    let mut witness_lengths = Vec::new();
    let mut model_lengths = Vec::new();
    let mut pair_lengths = Vec::new();
    let mut public_io_lengths = Vec::new();
    let mut layers = Vec::with_capacity(statement.layers.len());
    let mut model_to_witness = Vec::new();
    let mut public_io_to_witness = Vec::new();

    for layer in &statement.layers {
        let shape = layer.shape;
        let input_plane = checked_product(&[shape.input_h, shape.input_w])?;
        let output_plane = checked_product(&[shape.output_h()?, shape.output_w()?])?;
        let input_start = witness_lengths.len();
        for _ in 0..shape.in_channels {
            public_io_to_witness.push(witness_lengths.len());
            public_io_lengths.push(input_plane);
            witness_lengths.push(input_plane);
        }

        let kernel_start = witness_lengths.len();
        for _ in 0..checked_product(&[shape.in_channels, shape.kernel_h, shape.kernel_w])? {
            model_to_witness.push(witness_lengths.len());
            model_lengths.push(shape.out_channels);
            witness_lengths.push(shape.out_channels);
        }

        let output_start = witness_lengths.len();
        for _ in 0..shape.out_channels {
            public_io_to_witness.push(witness_lengths.len());
            public_io_lengths.push(output_plane);
            witness_lengths.push(output_plane);
        }
        pair_lengths.push(shape.in_channels);
        layers.push(ConvLayerMap {
            input_start,
            kernel_start,
            output_start,
        });
    }

    Ok(ConvLayouts {
        witness: BvaLayout::from_lengths(&witness_lengths)
            .map_err(|_| GlobalConvError::ProtocolShapeMismatch)?,
        model: BvaLayout::from_lengths(&model_lengths)
            .map_err(|_| GlobalConvError::ProtocolShapeMismatch)?,
        pair: BvaLayout::from_lengths(&pair_lengths)
            .map_err(|_| GlobalConvError::ProtocolShapeMismatch)?,
        public_io: BvaLayout::from_lengths(&public_io_lengths)
            .map_err(|_| GlobalConvError::ProtocolShapeMismatch)?,
        layers,
        model_to_witness,
        public_io_to_witness,
    })
}

struct ConvVectorSource<'a> {
    layout: &'a BvaLayout,
    vectors: &'a [Vec<i32>],
}

impl BvaI32Source for ConvVectorSource<'_> {
    type Error = GlobalConvError;

    fn layout(&self) -> &BvaLayout {
        self.layout
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[i32])) -> Result<(), Self::Error> {
        for (index, vector) in self.vectors.iter().enumerate() {
            visit(index, vector);
        }
        Ok(())
    }
}

impl BvaFieldSource<Fr> for ConvVectorSource<'_> {
    type Error = GlobalConvError;

    fn layout(&self) -> &BvaLayout {
        self.layout
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[Fr])) -> Result<(), Self::Error> {
        for (index, vector) in self.vectors.iter().enumerate() {
            let field = vector.iter().copied().map(Fr::from).collect::<Vec<_>>();
            visit(index, &field);
        }
        Ok(())
    }
}

struct ConvSubsetSource<'a> {
    layout: &'a BvaLayout,
    vectors: &'a [Vec<i32>],
    source_indices: &'a [usize],
}

impl BvaI32Source for ConvSubsetSource<'_> {
    type Error = GlobalConvError;

    fn layout(&self) -> &BvaLayout {
        self.layout
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[i32])) -> Result<(), Self::Error> {
        for (index, source_index) in self.source_indices.iter().copied().enumerate() {
            let vector = self
                .vectors
                .get(source_index)
                .ok_or(GlobalConvError::ProtocolShapeMismatch)?;
            visit(index, vector);
        }
        Ok(())
    }
}

impl BvaFieldSource<Fr> for ConvSubsetSource<'_> {
    type Error = GlobalConvError;

    fn layout(&self) -> &BvaLayout {
        self.layout
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[Fr])) -> Result<(), Self::Error> {
        for (index, source_index) in self.source_indices.iter().copied().enumerate() {
            let vector = self
                .vectors
                .get(source_index)
                .ok_or(GlobalConvError::ProtocolShapeMismatch)?;
            let field = vector.iter().copied().map(Fr::from).collect::<Vec<_>>();
            visit(index, &field);
        }
        Ok(())
    }
}

fn build_witness_vectors(
    statement: &GlobalConvStatement,
    layouts: &ConvLayouts,
    kernels: Vec<Vec<i32>>,
) -> Result<Vec<Vec<i32>>, GlobalConvError> {
    if kernels.len() != statement.layers.len() {
        return Err(GlobalConvError::KernelLength);
    }
    let mut vectors = Vec::with_capacity(layouts.witness.vector_lengths().len());
    for ((layer, map), kernel) in statement.layers.iter().zip(&layouts.layers).zip(kernels) {
        let shape = layer.shape;
        if kernel.len() != shape.kernel_len()? {
            return Err(GlobalConvError::KernelLength);
        }
        if full_cross_correlation(shape, &layer.input, &kernel)? != layer.output {
            return Err(GlobalConvError::WitnessOutputMismatch);
        }
        let input_plane = checked_product(&[shape.input_h, shape.input_w])?;
        for channel in 0..shape.in_channels {
            vectors.push(layer.input[channel * input_plane..(channel + 1) * input_plane].to_vec());
        }
        if vectors.len() != map.kernel_start {
            return Err(GlobalConvError::ProtocolShapeMismatch);
        }
        for input_channel in 0..shape.in_channels {
            for kernel_row in 0..shape.kernel_h {
                for kernel_column in 0..shape.kernel_w {
                    let mut across_outputs = Vec::with_capacity(shape.out_channels);
                    for output_channel in 0..shape.out_channels {
                        let index = ((input_channel * shape.out_channels + output_channel)
                            * shape.kernel_h
                            + kernel_row)
                            * shape.kernel_w
                            + kernel_column;
                        across_outputs.push(kernel[index]);
                    }
                    vectors.push(across_outputs);
                }
            }
        }
        if vectors.len() != map.output_start {
            return Err(GlobalConvError::ProtocolShapeMismatch);
        }
        let output_plane = checked_product(&[shape.output_h()?, shape.output_w()?])?;
        for channel in 0..shape.out_channels {
            vectors
                .push(layer.output[channel * output_plane..(channel + 1) * output_plane].to_vec());
        }
    }
    if vectors.len() != layouts.witness.vector_lengths().len()
        || vectors
            .iter()
            .map(Vec::len)
            .ne(layouts.witness.vector_lengths().iter().copied())
    {
        return Err(GlobalConvError::ProtocolShapeMismatch);
    }
    Ok(vectors)
}

struct ConvRelationChallenges {
    gammas: Vec<Fr>,
    betas: Vec<Fr>,
    public_io_point: Vec<Fr>,
    model_point: Vec<Fr>,
    pair_binding: Fr,
    model_binding: Fr,
}

fn relation_challenges(
    statement: &GlobalConvStatement,
    layouts: &ConvLayouts,
    instance: &GlobalConvInstance,
) -> (ConvRelationChallenges, FiatShamir) {
    let mut fs = FiatShamir::new();
    fs.push(&0x434f_4e56_4844_5231_u64);
    fs.push(&statement.transcript_digest().to_vec());
    fs.push(&instance.global_witness_commitment);
    fs.push(&instance.model_commitment);
    let mut gammas = Vec::with_capacity(statement.layers.len());
    let mut betas = Vec::with_capacity(statement.layers.len());
    for _ in &statement.layers {
        gammas.push(fs.gen_challenge::<Fr>());
        betas.push(fs.gen_challenge::<Fr>());
    }
    let public_io_point = (0..layouts.public_io.logical_len().ilog2())
        .map(|_| fs.gen_challenge::<Fr>())
        .collect();
    let model_point = (0..layouts.model.logical_len().ilog2())
        .map(|_| fs.gen_challenge::<Fr>())
        .collect();
    let pair_binding = fs.gen_challenge::<Fr>();
    let model_binding = fs.gen_challenge::<Fr>();
    (
        ConvRelationChallenges {
            gammas,
            betas,
            public_io_point,
            model_point,
            pair_binding,
            model_binding,
        },
        fs,
    )
}

fn absorb_pair_proof(fs: &mut FiatShamir, proof: &EoWitnessPairProof<Fr>) {
    for (left, right) in &proof.rounds {
        fs.push(left);
        fs.push(right);
    }
    fs.push(&proof.a_reduced);
    fs.push(&proof.b_reduced);
}

fn absorb_eo_proof(fs: &mut FiatShamir, proof: &EoLiteBulletProof<Fr>) {
    for (left, right) in &proof.rounds {
        fs.push(left);
        fs.push(right);
    }
    fs.push(&proof.a_reduced);
    fs.push(&proof.b_reduced);
}

fn synchronized_weights_and_binding(
    mut fs: FiatShamir,
    conv_claim: Fr,
    conv_proof: &EoWitnessPairProof<Fr>,
    model_claim: Fr,
    model_eo: &EoLiteBulletProof<Fr>,
    model_value_commitment: PairingOutput<Bls12_381>,
) -> ([Fr; 5], Fr) {
    fs.push(&conv_claim);
    absorb_pair_proof(&mut fs, conv_proof);
    fs.push(&model_claim);
    absorb_eo_proof(&mut fs, model_eo);
    fs.push(&model_value_commitment);
    let weights = std::array::from_fn(|_| fs.gen_challenge::<Fr>());
    let binding = fs.gen_challenge::<Fr>();
    (weights, binding)
}

fn field_power(base: Fr, exponent: usize) -> Result<Fr, GlobalConvError> {
    Ok(base.pow([u64::try_from(exponent).map_err(|_| GlobalConvError::ArithmeticOverflow)?]))
}

fn linear_power_challenges(
    base: Fr,
    stride: usize,
    bits: usize,
) -> Result<Vec<Fr>, GlobalConvError> {
    (0..bits)
        .map(|bit| {
            let power = stride
                .checked_mul(
                    1_usize
                        .checked_shl(
                            u32::try_from(bit).map_err(|_| GlobalConvError::ArithmeticOverflow)?,
                        )
                        .ok_or(GlobalConvError::ArithmeticOverflow)?,
                )
                .ok_or(GlobalConvError::ArithmeticOverflow)?;
            field_power(base, power)
        })
        .collect()
}

fn input_power_challenges(shape: Conv2dShape, gamma: Fr) -> Result<Vec<Fr>, GlobalConvError> {
    let column_bits = shape.input_w.ilog2() as usize;
    let row_bits = shape.input_h.ilog2() as usize;
    let mut challenges = linear_power_challenges(gamma, 1, column_bits)?;
    challenges.extend(linear_power_challenges(gamma, shape.output_w()?, row_bits)?);
    Ok(challenges)
}

fn add_tbv_term(
    layout: &BvaLayout,
    terms: &mut [Vec<(Fr, Vec<Fr>)>],
    vector_index: usize,
    eta: Fr,
    challenges: Vec<Fr>,
) -> Result<(), GlobalConvError> {
    let bucket_index = *layout
        .vector_to_bucket()
        .get(vector_index)
        .ok_or(GlobalConvError::ProtocolShapeMismatch)?;
    if challenges.len() != layout.bucket_sizes()[bucket_index].ilog2() as usize {
        return Err(GlobalConvError::ProtocolShapeMismatch);
    }
    terms[bucket_index].push((eta, challenges));
    Ok(())
}

fn tbv_from_terms(
    layout: &BvaLayout,
    terms: Vec<Vec<(Fr, Vec<Fr>)>>,
) -> Result<TensorBasisVector<Fr>, GlobalConvError> {
    TensorBasisVector::new(
        layout,
        terms.into_iter().map(TbvBucket::from_terms).collect(),
    )
    .map_err(|_| GlobalConvError::ProtocolShapeMismatch)
}

fn public_io_vectors(statement: &GlobalConvStatement) -> Result<Vec<Vec<Fr>>, GlobalConvError> {
    let mut vectors = Vec::new();
    for layer in &statement.layers {
        let shape = layer.shape;
        let input_plane = checked_product(&[shape.input_h, shape.input_w])?;
        for channel in 0..shape.in_channels {
            vectors.push(
                layer.input[channel * input_plane..(channel + 1) * input_plane]
                    .iter()
                    .copied()
                    .map(Fr::from)
                    .collect(),
            );
        }
        let output_plane = checked_product(&[shape.output_h()?, shape.output_w()?])?;
        for channel in 0..shape.out_channels {
            vectors.push(
                layer.output[channel * output_plane..(channel + 1) * output_plane]
                    .iter()
                    .copied()
                    .map(Fr::from)
                    .collect(),
            );
        }
    }
    Ok(vectors)
}

fn public_io_tbv_and_claim(
    statement: &GlobalConvStatement,
    layouts: &ConvLayouts,
    challenges: &[Fr],
) -> Result<(TensorBasisVector<Fr>, Fr), GlobalConvError> {
    let source_tbv = TensorBasisVector::global_tensor(&layouts.public_io, challenges)
        .map_err(|_| GlobalConvError::ProtocolShapeMismatch)?;
    let public = BucketedVector::new(public_io_vectors(statement)?)
        .map_err(|_| GlobalConvError::ProtocolShapeMismatch)?;
    let claim = public
        .inner_product(&source_tbv)
        .map_err(|_| GlobalConvError::ProtocolShapeMismatch)?
        .value;
    let routes = layouts
        .public_io_to_witness
        .iter()
        .copied()
        .map(|target| Some((target, Fr::one())))
        .collect::<Vec<_>>();
    let routed = source_tbv
        .route_vectors(&layouts.witness, &routes)
        .map_err(|_| GlobalConvError::ProtocolShapeMismatch)?;
    Ok((routed, claim))
}

fn model_tbv(
    layouts: &ConvLayouts,
    challenges: &[Fr],
) -> Result<TensorBasisVector<Fr>, GlobalConvError> {
    TensorBasisVector::global_tensor(&layouts.model, challenges)
        .map_err(|_| GlobalConvError::ProtocolShapeMismatch)
}

fn routed_model_tbv(
    layouts: &ConvLayouts,
    model_tbv: &TensorBasisVector<Fr>,
) -> Result<TensorBasisVector<Fr>, GlobalConvError> {
    let routes = layouts
        .model_to_witness
        .iter()
        .copied()
        .map(|target| Some((target, Fr::one())))
        .collect::<Vec<_>>();
    model_tbv
        .route_vectors(&layouts.witness, &routes)
        .map_err(|_| GlobalConvError::ProtocolShapeMismatch)
}

fn evaluate_power_series(values: &[i32], point: Fr, stride: usize) -> Result<Fr, GlobalConvError> {
    let step = field_power(point, stride)?;
    let mut power = Fr::one();
    let mut value = Fr::zero();
    for entry in values {
        value += Fr::from(*entry) * power;
        power *= step;
    }
    Ok(value)
}

fn convolution_pair_assemblies(
    statement: &GlobalConvStatement,
    layouts: &ConvLayouts,
    witness_vectors: &[Vec<i32>],
    challenges: &ConvRelationChallenges,
) -> Result<(BucketedVector<Fr>, BucketedVector<Fr>, Fr), GlobalConvError> {
    let mut left_vectors = Vec::with_capacity(statement.layers.len());
    let mut right_vectors = Vec::with_capacity(statement.layers.len());
    let mut claim = Fr::zero();
    for (layer_index, (layer, map)) in statement.layers.iter().zip(&layouts.layers).enumerate() {
        let shape = layer.shape;
        let gamma = challenges.gammas[layer_index];
        let beta = challenges.betas[layer_index];
        let output_w = shape.output_w()?;
        let output_plane = checked_product(&[shape.output_h()?, output_w])?;
        let input_factors = input_power_challenges(shape, gamma)?;
        let mut left = Vec::with_capacity(shape.in_channels);
        let mut right = Vec::with_capacity(shape.in_channels);
        for input_channel in 0..shape.in_channels {
            let input_vector = witness_vectors
                .get(map.input_start + input_channel)
                .ok_or(GlobalConvError::ProtocolShapeMismatch)?;
            let input_bucket_log = input_vector.len().ilog2() as usize;
            if input_factors.len() != input_bucket_log {
                return Err(GlobalConvError::ProtocolShapeMismatch);
            }
            let mut input_value = Fr::zero();
            for (index, entry) in input_vector.iter().copied().enumerate() {
                let row = index / shape.input_w;
                let column = index % shape.input_w;
                let exponent = row
                    .checked_mul(output_w)
                    .and_then(|value| value.checked_add(column))
                    .ok_or(GlobalConvError::ArithmeticOverflow)?;
                input_value += Fr::from(entry) * field_power(gamma, exponent)?;
            }
            left.push(input_value);

            let mut kernel_value = Fr::zero();
            for kernel_row in 0..shape.kernel_h {
                for kernel_column in 0..shape.kernel_w {
                    let vector_index = map.kernel_start
                        + (input_channel * shape.kernel_h + kernel_row) * shape.kernel_w
                        + kernel_column;
                    let across_outputs = witness_vectors
                        .get(vector_index)
                        .ok_or(GlobalConvError::ProtocolShapeMismatch)?;
                    let output_channel_polynomial =
                        evaluate_power_series(across_outputs, gamma, output_plane)?;
                    let spatial_exponent = (shape.kernel_h - 1 - kernel_row)
                        .checked_mul(output_w)
                        .and_then(|value| value.checked_add(shape.kernel_w - 1 - kernel_column))
                        .ok_or(GlobalConvError::ArithmeticOverflow)?;
                    kernel_value +=
                        output_channel_polynomial * field_power(gamma, spatial_exponent)?;
                }
            }
            right.push(beta * kernel_value);
        }
        left_vectors.push(left);
        right_vectors.push(right);
        claim += beta * evaluate_power_series(&layer.output, gamma, 1)?;
    }
    let left =
        BucketedVector::new(left_vectors).map_err(|_| GlobalConvError::ProtocolShapeMismatch)?;
    let right =
        BucketedVector::new(right_vectors).map_err(|_| GlobalConvError::ProtocolShapeMismatch)?;
    if left.layout() != &layouts.pair || right.layout() != &layouts.pair {
        return Err(GlobalConvError::ProtocolShapeMismatch);
    }
    Ok((left, right, claim))
}

fn convolution_output_tbv(
    statement: &GlobalConvStatement,
    layouts: &ConvLayouts,
    challenges: &ConvRelationChallenges,
) -> Result<TensorBasisVector<Fr>, GlobalConvError> {
    let mut terms = vec![Vec::new(); layouts.witness.bucket_sizes().len()];
    for (layer_index, (layer, map)) in statement.layers.iter().zip(&layouts.layers).enumerate() {
        let shape = layer.shape;
        let gamma = challenges.gammas[layer_index];
        let beta = challenges.betas[layer_index];
        let output_plane = checked_product(&[shape.output_h()?, shape.output_w()?])?;
        for output_channel in 0..shape.out_channels {
            let vector_index = map.output_start + output_channel;
            let bucket = layouts.witness.vector_to_bucket()[vector_index];
            let factors = linear_power_challenges(
                gamma,
                1,
                layouts.witness.bucket_sizes()[bucket].ilog2() as usize,
            )?;
            let exponent = output_channel
                .checked_mul(output_plane)
                .ok_or(GlobalConvError::ArithmeticOverflow)?;
            add_tbv_term(
                &layouts.witness,
                &mut terms,
                vector_index,
                beta * field_power(gamma, exponent)?,
                factors,
            )?;
        }
    }
    tbv_from_terms(&layouts.witness, terms)
}

fn convolution_terminal_tbvs(
    statement: &GlobalConvStatement,
    layouts: &ConvLayouts,
    challenges: &ConvRelationChallenges,
    pair_verification_challenges: &[Fr],
    pair_verification_challenges_inv: &[Fr],
) -> Result<(TensorBasisVector<Fr>, TensorBasisVector<Fr>), GlobalConvError> {
    let pair_left =
        TensorBasisVector::global_tensor(&layouts.pair, pair_verification_challenges_inv)
            .map_err(|_| GlobalConvError::ProtocolShapeMismatch)?;
    let pair_right = TensorBasisVector::global_tensor(&layouts.pair, pair_verification_challenges)
        .map_err(|_| GlobalConvError::ProtocolShapeMismatch)?;
    let mut left_terms = vec![Vec::new(); layouts.witness.bucket_sizes().len()];
    let mut right_terms = vec![Vec::new(); layouts.witness.bucket_sizes().len()];

    for (layer_index, (layer, map)) in statement.layers.iter().zip(&layouts.layers).enumerate() {
        let shape = layer.shape;
        let gamma = challenges.gammas[layer_index];
        let beta = challenges.betas[layer_index];
        let output_w = shape.output_w()?;
        let output_plane = checked_product(&[shape.output_h()?, output_w])?;
        let input_factors = input_power_challenges(shape, gamma)?;
        let kernel_factors =
            linear_power_challenges(gamma, output_plane, shape.out_channels.ilog2() as usize)?;
        for input_channel in 0..shape.in_channels {
            let left_scale = pair_left
                .physical_coefficient_at(layer_index, input_channel)
                .map_err(|_| GlobalConvError::ProtocolShapeMismatch)?;
            add_tbv_term(
                &layouts.witness,
                &mut left_terms,
                map.input_start + input_channel,
                left_scale,
                input_factors.clone(),
            )?;

            let right_scale = pair_right
                .physical_coefficient_at(layer_index, input_channel)
                .map_err(|_| GlobalConvError::ProtocolShapeMismatch)?;
            for kernel_row in 0..shape.kernel_h {
                for kernel_column in 0..shape.kernel_w {
                    let vector_index = map.kernel_start
                        + (input_channel * shape.kernel_h + kernel_row) * shape.kernel_w
                        + kernel_column;
                    let spatial_exponent = (shape.kernel_h - 1 - kernel_row)
                        .checked_mul(output_w)
                        .and_then(|value| value.checked_add(shape.kernel_w - 1 - kernel_column))
                        .ok_or(GlobalConvError::ArithmeticOverflow)?;
                    add_tbv_term(
                        &layouts.witness,
                        &mut right_terms,
                        vector_index,
                        right_scale * beta * field_power(gamma, spatial_exponent)?,
                        kernel_factors.clone(),
                    )?;
                }
            }
        }
    }
    Ok((
        tbv_from_terms(&layouts.witness, left_terms)?,
        tbv_from_terms(&layouts.witness, right_terms)?,
    ))
}

fn synchronized_relation(
    layouts: &ConvLayouts,
    weights: [Fr; 5],
    public_io: TensorBasisVector<Fr>,
    public_io_claim: Fr,
    output: TensorBasisVector<Fr>,
    conv_claim: Fr,
    left: TensorBasisVector<Fr>,
    left_claim: Fr,
    right: TensorBasisVector<Fr>,
    right_claim: Fr,
    model: TensorBasisVector<Fr>,
    model_claim: Fr,
) -> Result<(TensorBasisVector<Fr>, Fr), GlobalConvError> {
    let tbv = TensorBasisVector::linear_combination_owned(
        &layouts.witness,
        vec![
            (weights[0], public_io),
            (weights[1], output),
            (weights[2], left),
            (weights[3], right),
            (weights[4], model),
        ],
    )
    .map_err(|_| GlobalConvError::ProtocolShapeMismatch)?;
    let claim = weights[0] * public_io_claim
        + weights[1] * conv_claim
        + weights[2] * left_claim
        + weights[3] * right_claim
        + weights[4] * model_claim;
    Ok((tbv, claim))
}

impl GlobalHdR1csConvProtocol {
    pub fn setup_with_rng<R: RngCore>(
        statement: &GlobalConvStatement,
        rng: &mut R,
    ) -> Result<GlobalConvParameters, GlobalConvError> {
        statement.validate()?;
        let layouts = compile_layouts(statement)?;
        let maximum_log = [
            layouts.witness.logical_len(),
            layouts.model.logical_len(),
            layouts.pair.logical_len(),
            layouts.public_io.logical_len(),
        ]
        .iter()
        .map(|length| length.ilog2() as usize)
        .max()
        .ok_or(GlobalConvError::ProtocolShapeMismatch)?;
        let q = (maximum_log + 1) / 2;
        let pc_pp =
            SmartPC::<Bls12_381>::setup(q, rng).map_err(|_| GlobalConvError::ProtocolIncomplete)?;
        Ok(GlobalConvParameters { pc_pp })
    }

    pub fn prove<R: RngCore>(
        parameters: &GlobalConvParameters,
        statement: &GlobalConvStatement,
        kernels: Vec<Vec<i32>>,
        rng: &mut R,
    ) -> Result<(GlobalConvInstance, GlobalConvProof, GlobalConvProverProfile), GlobalConvError>
    {
        statement.validate()?;
        let started = Instant::now();
        let layouts = compile_layouts(statement)?;
        let witness_vectors = build_witness_vectors(statement, &layouts, kernels)?;

        let witness_hiding = Fr::rand(rng);
        let mut witness_source = ConvVectorSource {
            layout: &layouts.witness,
            vectors: &witness_vectors,
        };
        let (global_witness_commitment, witness_tier_one, witness_commit_profile) =
            commit_bva_i32_streaming::<Bls12_381, _>(
                &parameters.pc_pp,
                &mut witness_source,
                witness_hiding,
                statement.commitment_bit_width,
            )
            .map_err(|_| GlobalConvError::ProtocolIncomplete)?;
        let witness_commit_seconds = started.elapsed().as_secs_f64();

        let model_hiding = Fr::rand(rng);
        let mut model_source = ConvSubsetSource {
            layout: &layouts.model,
            vectors: &witness_vectors,
            source_indices: &layouts.model_to_witness,
        };
        let (model_commitment, model_tier_one, _model_commit_profile) =
            commit_bva_i32_streaming::<Bls12_381, _>(
                &parameters.pc_pp,
                &mut model_source,
                model_hiding,
                statement.commitment_bit_width,
            )
            .map_err(|_| GlobalConvError::ProtocolIncomplete)?;
        let model_commit_seconds = started.elapsed().as_secs_f64() - witness_commit_seconds;

        if witness_commit_profile.materialized_padding_entries != 0 {
            return Err(GlobalConvError::ProtocolIncomplete);
        }
        let instance = GlobalConvInstance {
            statement_digest: statement.transcript_digest(),
            global_witness_commitment,
            model_commitment,
        };
        let (challenges, fs) = relation_challenges(statement, &layouts, &instance);

        let model_tbv = model_tbv(&layouts, &challenges.model_point)?;
        let mut model_field_source = ConvSubsetSource {
            layout: &layouts.model,
            vectors: &witness_vectors,
            source_indices: &layouts.model_to_witness,
        };
        let model_eval = inner_product_bva_field_streaming(&mut model_field_source, &model_tbv)
            .map_err(|_| GlobalConvError::ProtocolIncomplete)?;
        let model_claim = model_eval.value;
        let model_eo = EoLiteBulletProof::prove_streaming_bound(
            &mut model_field_source,
            &model_tbv,
            model_claim,
            challenges.model_binding,
        )
        .map_err(|_| GlobalConvError::ProtocolIncomplete)?
        .0;
        let model_verification =
            model_eo.verify_with_tbv_bound(&model_tbv, model_claim, challenges.model_binding);
        if !model_verification.accepted {
            return Err(GlobalConvError::ProtocolIncomplete);
        }
        let model_value_commitment = Bls12_381::pairing(
            parameters.pc_pp.g_0.mul(model_eo.a_reduced),
            parameters.pc_pp.h_hat,
        );
        let (model_xl, model_xr) = bva_eo_opening_point(
            layouts.model.logical_len(),
            &model_verification.challenges_inv,
        )
        .map_err(|_| GlobalConvError::ProtocolIncomplete)?;
        let mut model_open_source = ConvSubsetSource {
            layout: &layouts.model,
            vectors: &witness_vectors,
            source_indices: &layouts.model_to_witness,
        };
        let (model_opening, _) = open_bva_field_streaming::<Bls12_381, _>(
            &parameters.pc_pp,
            &mut model_open_source,
            &model_xl,
            &model_xr,
            model_value_commitment,
            model_commitment,
            &model_tier_one,
            Fr::zero(),
            model_hiding,
        )
        .map_err(|_| GlobalConvError::ProtocolIncomplete)?;

        let (pair_left, pair_right, conv_claim) =
            convolution_pair_assemblies(statement, &layouts, &witness_vectors, &challenges)?;
        let pair_started = Instant::now();
        let (conv_proof, _) = EoWitnessPairProof::prove_bound(
            &pair_left,
            &pair_right,
            conv_claim,
            challenges.pair_binding,
        )
        .map_err(|_| GlobalConvError::ProtocolIncomplete)?;
        let convolution_reduction_seconds = pair_started.elapsed().as_secs_f64();
        let pair_verification = conv_proof.verify_bound(
            layouts.pair.logical_len(),
            conv_claim,
            challenges.pair_binding,
        );
        if !pair_verification.accepted {
            return Err(GlobalConvError::ProtocolIncomplete);
        }
        let (left_tbv, right_tbv) = convolution_terminal_tbvs(
            statement,
            &layouts,
            &challenges,
            &pair_verification.challenges,
            &pair_verification.challenges_inv,
        )?;
        let (public_io_tbv, public_io_claim) =
            public_io_tbv_and_claim(statement, &layouts, &challenges.public_io_point)?;
        let output_tbv = convolution_output_tbv(statement, &layouts, &challenges)?;
        let routed_model = routed_model_tbv(&layouts, &model_tbv)?;
        let (weights, synchronized_binding) = synchronized_weights_and_binding(
            fs,
            conv_claim,
            &conv_proof,
            model_claim,
            &model_eo,
            model_value_commitment,
        );
        let (synchronized_tbv, synchronized_claim) = synchronized_relation(
            &layouts,
            weights,
            public_io_tbv,
            public_io_claim,
            output_tbv,
            conv_claim,
            left_tbv,
            conv_proof.a_reduced,
            right_tbv,
            conv_proof.b_reduced,
            routed_model,
            model_claim,
        )?;
        let mut synchronized_source = ConvVectorSource {
            layout: &layouts.witness,
            vectors: &witness_vectors,
        };
        let synchronized_eval =
            inner_product_bva_field_streaming(&mut synchronized_source, &synchronized_tbv)
                .map_err(|_| GlobalConvError::ProtocolIncomplete)?;
        if synchronized_eval.value != synchronized_claim {
            return Err(GlobalConvError::ProtocolIncomplete);
        }
        let synchronized_started = Instant::now();
        let synchronized_eo = EoLiteBulletProof::prove_streaming_bound(
            &mut synchronized_source,
            &synchronized_tbv,
            synchronized_claim,
            synchronized_binding,
        )
        .map_err(|_| GlobalConvError::ProtocolIncomplete)?
        .0;
        let synchronized_reduction_seconds = synchronized_started.elapsed().as_secs_f64();
        let synchronized_verification = synchronized_eo.verify_with_tbv_bound(
            &synchronized_tbv,
            synchronized_claim,
            synchronized_binding,
        );
        if !synchronized_verification.accepted {
            return Err(GlobalConvError::ProtocolIncomplete);
        }
        let synchronized_value_commitment = Bls12_381::pairing(
            parameters.pc_pp.g_0.mul(synchronized_eo.a_reduced),
            parameters.pc_pp.h_hat,
        );
        let (synchronized_xl, synchronized_xr) = bva_eo_opening_point(
            layouts.witness.logical_len(),
            &synchronized_verification.challenges_inv,
        )
        .map_err(|_| GlobalConvError::ProtocolIncomplete)?;
        let opening_started = Instant::now();
        let (synchronized_opening, _) = open_bva_field_streaming::<Bls12_381, _>(
            &parameters.pc_pp,
            &mut synchronized_source,
            &synchronized_xl,
            &synchronized_xr,
            synchronized_value_commitment,
            global_witness_commitment,
            &witness_tier_one,
            Fr::zero(),
            witness_hiding,
        )
        .map_err(|_| GlobalConvError::ProtocolIncomplete)?;
        let opening_seconds = opening_started.elapsed().as_secs_f64();

        let profile = GlobalConvProverProfile {
            witness_physical_entries: layouts.witness.vector_lengths().iter().sum(),
            witness_logical_entries: layouts.witness.logical_len(),
            model_physical_entries: layouts.model.vector_lengths().iter().sum(),
            model_logical_entries: layouts.model.logical_len(),
            pair_physical_entries: pair_left.physical_entries(),
            pair_logical_entries: layouts.pair.logical_len(),
            materialized_padding_entries: 0,
            witness_commit_seconds,
            model_commit_seconds,
            convolution_reduction_seconds,
            model_binding_seconds: 0.0,
            synchronized_reduction_seconds,
            opening_seconds,
        };
        let proof = GlobalConvProof {
            statement_digest: statement.transcript_digest(),
            conv_claim,
            conv_proof,
            model_claim,
            model_eo,
            model_value_commitment,
            model_opening,
            synchronized_eo,
            synchronized_value_commitment,
            synchronized_opening,
        };
        Ok((instance, proof, profile))
    }

    pub fn verify(
        parameters: &GlobalConvParameters,
        statement: &GlobalConvStatement,
        instance: &GlobalConvInstance,
        proof: &GlobalConvProof,
    ) -> Result<bool, GlobalConvError> {
        statement.validate()?;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let layouts = compile_layouts(statement)?;
            if instance.statement_digest != statement.transcript_digest()
                || proof.statement_digest != instance.statement_digest
            {
                return Ok(false);
            }
            let (challenges, fs) = relation_challenges(statement, &layouts, instance);
            let model_tbv = model_tbv(&layouts, &challenges.model_point)?;
            let model_verification = proof.model_eo.verify_with_tbv_bound(
                &model_tbv,
                proof.model_claim,
                challenges.model_binding,
            );
            if !model_verification.accepted {
                return Ok(false);
            }
            let expected_model_value_commitment = Bls12_381::pairing(
                parameters.pc_pp.g_0.mul(proof.model_eo.a_reduced),
                parameters.pc_pp.h_hat,
            );
            if proof.model_value_commitment != expected_model_value_commitment {
                return Ok(false);
            }
            let (model_xl, model_xr) = bva_eo_opening_point(
                layouts.model.logical_len(),
                &model_verification.challenges_inv,
            )
            .map_err(|_| GlobalConvError::ProtocolIncomplete)?;
            if !SmartPC::<Bls12_381>::verify(
                &parameters.pc_pp,
                instance.model_commitment,
                proof.model_value_commitment,
                &model_xl,
                &model_xr,
                &proof.model_opening,
            )
            .map_err(|_| GlobalConvError::ProtocolIncomplete)?
            {
                return Ok(false);
            }

            let pair_verification = proof.conv_proof.verify_bound(
                layouts.pair.logical_len(),
                proof.conv_claim,
                challenges.pair_binding,
            );
            if !pair_verification.accepted {
                return Ok(false);
            }
            let (left_tbv, right_tbv) = convolution_terminal_tbvs(
                statement,
                &layouts,
                &challenges,
                &pair_verification.challenges,
                &pair_verification.challenges_inv,
            )?;
            let (public_io_tbv, public_io_claim) =
                public_io_tbv_and_claim(statement, &layouts, &challenges.public_io_point)?;
            let output_tbv = convolution_output_tbv(statement, &layouts, &challenges)?;
            let routed_model = routed_model_tbv(&layouts, &model_tbv)?;
            let (weights, synchronized_binding) = synchronized_weights_and_binding(
                fs,
                proof.conv_claim,
                &proof.conv_proof,
                proof.model_claim,
                &proof.model_eo,
                proof.model_value_commitment,
            );
            let (synchronized_tbv, synchronized_claim) = synchronized_relation(
                &layouts,
                weights,
                public_io_tbv,
                public_io_claim,
                output_tbv,
                proof.conv_claim,
                left_tbv,
                proof.conv_proof.a_reduced,
                right_tbv,
                proof.conv_proof.b_reduced,
                routed_model,
                proof.model_claim,
            )?;
            let synchronized_verification = proof.synchronized_eo.verify_with_tbv_bound(
                &synchronized_tbv,
                synchronized_claim,
                synchronized_binding,
            );
            if !synchronized_verification.accepted {
                return Ok(false);
            }
            let expected_synchronized_value_commitment = Bls12_381::pairing(
                parameters.pc_pp.g_0.mul(proof.synchronized_eo.a_reduced),
                parameters.pc_pp.h_hat,
            );
            if proof.synchronized_value_commitment != expected_synchronized_value_commitment {
                return Ok(false);
            }
            let (synchronized_xl, synchronized_xr) = bva_eo_opening_point(
                layouts.witness.logical_len(),
                &synchronized_verification.challenges_inv,
            )
            .map_err(|_| GlobalConvError::ProtocolIncomplete)?;
            SmartPC::<Bls12_381>::verify(
                &parameters.pc_pp,
                instance.global_witness_commitment,
                proof.synchronized_value_commitment,
                &synchronized_xl,
                &synchronized_xr,
                &proof.synchronized_opening,
            )
            .map_err(|_| GlobalConvError::ProtocolIncomplete)
        }));
        result.unwrap_or(Ok(false))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum GlobalConvError {
    InvalidShape,
    ArithmeticOverflow,
    InputLength,
    KernelLength,
    InvalidStatement,
    ProtocolShapeMismatch,
    WitnessOutputMismatch,
    ProtocolIncomplete,
}

impl Display for GlobalConvError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{self:?}")
    }
}

impl std::error::Error for GlobalConvError {}

#[cfg(test)]
mod tests {
    use ark_ff::One;
    use rand_chacha::{rand_core::SeedableRng, ChaCha20Rng};

    use super::{
        full_cross_correlation, Conv2dShape, GlobalConvPublicLayer, GlobalConvStatement,
        GlobalHdR1csConvProtocol,
    };

    fn tiny_fixture() -> (GlobalConvStatement, Vec<Vec<i32>>) {
        let shape = Conv2dShape::new(2, 2, 2, 2, 2, 2).unwrap();
        let input = vec![1, 2, 3, 4, -1, 0, 1, 2];
        let kernel = vec![
            1, 0, 0, 1, // c0 -> d0
            0, 1, 1, 0, // c0 -> d1
            1, 1, 0, 0, // c1 -> d0
            0, 0, 1, 1, // c1 -> d1
        ];
        let output = full_cross_correlation(shape, &input, &kernel).unwrap();
        (
            GlobalConvStatement::new(
                vec![GlobalConvPublicLayer {
                    shape,
                    input,
                    output,
                }],
                8,
            )
            .unwrap(),
            vec![kernel],
        )
    }

    #[test]
    fn reference_matches_literal_multichannel_full_cross_correlation() {
        let shape = Conv2dShape::new(2, 2, 2, 2, 2, 2).unwrap();
        let input = vec![1, 2, 3, 4, -1, 0, 1, 2];
        let kernel = vec![
            1, 0, 0, 1, // c0 -> d0
            0, 1, 1, 0, // c0 -> d1
            1, 1, 0, 0, // c1 -> d0
            0, 0, 1, 1, // c1 -> d1
        ];
        assert_eq!(
            full_cross_correlation(shape, &input, &kernel).unwrap(),
            vec![
                1, 2, 0, 2, 4, 2, 1, 6, 6, // d0
                -1, 0, 2, 2, 8, 6, 3, 4, 0, // d1
            ]
        );
    }

    #[test]
    fn reference_rejects_wrong_dense_lengths() {
        let shape = Conv2dShape::new(2, 2, 2, 2, 1, 1).unwrap();
        assert!(full_cross_correlation(shape, &[1, 2, 3], &[1, 2, 3, 4]).is_err());
        assert!(full_cross_correlation(shape, &[1, 2, 3, 4], &[1, 2, 3]).is_err());
    }

    #[test]
    fn global_conv_proves_with_one_witness_commitment_and_one_opening() {
        let (statement, kernels) = tiny_fixture();
        let mut rng = ChaCha20Rng::seed_from_u64(7);
        let parameters = GlobalHdR1csConvProtocol::setup_with_rng(&statement, &mut rng).unwrap();
        let (instance, proof, profile) =
            GlobalHdR1csConvProtocol::prove(&parameters, &statement, kernels, &mut rng).unwrap();
        assert!(
            GlobalHdR1csConvProtocol::verify(&parameters, &statement, &instance, &proof).unwrap()
        );
        assert_eq!(proof.witness_commitment_count(), 1);
        assert_eq!(proof.witness_opening_count(), 1);
        assert_eq!(profile.materialized_padding_entries, 0);
    }

    #[test]
    fn global_conv_rejects_public_output_and_proof_tampering() {
        let (statement, kernels) = tiny_fixture();
        let mut rng = ChaCha20Rng::seed_from_u64(11);
        let parameters = GlobalHdR1csConvProtocol::setup_with_rng(&statement, &mut rng).unwrap();
        let (instance, proof, _) =
            GlobalHdR1csConvProtocol::prove(&parameters, &statement, kernels, &mut rng).unwrap();

        let mut wrong_statement = statement.clone();
        wrong_statement.layers[0].output[0] += 1;
        assert!(!GlobalHdR1csConvProtocol::verify(
            &parameters,
            &wrong_statement,
            &instance,
            &proof
        )
        .unwrap());

        let mut wrong_proof = proof.clone();
        wrong_proof.conv_claim += ark_bls12_381::Fr::one();
        assert!(!GlobalHdR1csConvProtocol::verify(
            &parameters,
            &statement,
            &instance,
            &wrong_proof
        )
        .unwrap());
    }
}
