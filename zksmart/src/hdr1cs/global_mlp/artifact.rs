use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::fmt::{Display, Formatter};
use std::io::{Cursor, Read};

use ark_ec::pairing::Pairing;
use ark_poly_commit::smart_pc::UniversalParams as PcsParams;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError};

use crate::data_structures::{ZkSRS, SRS};
use crate::hdr1cs::mlp::MlpManifest;
use crate::utils::zktr::{TranElem, TranSeq};

use super::{
    GlobalActivationProof, GlobalArchitectureCommitments, GlobalBooleanProof, GlobalIoProof,
    GlobalLinearProof, GlobalMlpInstance, GlobalMlpProof, GlobalMlpShape,
    GlobalMultiplicationProof, GlobalOperationType, GlobalRangeProof, GlobalWitnessBatchProof,
    GlobalZeroProductProof, RoutedOpeningProof,
};
use ark_poly_commit::smart_pc::data_structures::Trans as PcsTrans;

const VERIFIER_MAGIC: &[u8; 8] = b"ZSMVFY02";
const PROOF_MAGIC: &[u8; 8] = b"ZSMPRO02";
const MAX_VECTOR_LEN: usize = 1_000_000;
const MAX_TRANSCRIPT_LEN: usize = 1_000_000;

#[derive(Debug)]
pub enum GlobalMlpArtifactError {
    Serialization(SerializationError),
    Format(&'static str),
    Bounds(&'static str),
}

impl Display for GlobalMlpArtifactError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Serialization(error) => write!(formatter, "serialization error: {error}"),
            Self::Format(message) => write!(formatter, "invalid artifact: {message}"),
            Self::Bounds(message) => write!(formatter, "artifact bound exceeded: {message}"),
        }
    }
}

impl std::error::Error for GlobalMlpArtifactError {}

impl From<SerializationError> for GlobalMlpArtifactError {
    fn from(error: SerializationError) -> Self {
        Self::Serialization(error)
    }
}

pub struct GlobalMlpVerifierArtifact<E: Pairing> {
    pub srs: SRS<E>,
    pub instance: GlobalMlpInstance<E>,
}

impl<E: Pairing> GlobalMlpVerifierArtifact<E> {
    pub fn to_bytes_exact(&self) -> Result<Vec<u8>, GlobalMlpArtifactError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(VERIFIER_MAGIC);
        write_srs(&self.srs, &mut bytes)?;
        write_instance(&self.instance, &mut bytes)?;
        Ok(bytes)
    }

    pub fn from_bytes_exact(bytes: &[u8]) -> Result<Self, GlobalMlpArtifactError> {
        let mut reader = Cursor::new(bytes);
        read_magic(&mut reader, VERIFIER_MAGIC)?;
        let srs = read_srs(&mut reader)?;
        let instance = read_instance(&mut reader)?;
        ensure_consumed(&reader, bytes.len())?;
        Ok(Self { srs, instance })
    }
}

impl<E: Pairing> GlobalMlpProof<E> {
    pub fn to_bytes_exact(&self) -> Result<Vec<u8>, GlobalMlpArtifactError> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(PROOF_MAGIC);
        write_io_proof(&self.io, &mut bytes)?;
        write_io_proof(&self.model, &mut bytes)?;
        write_multiplication_proof(&self.multiplication, &mut bytes)?;
        write_linear_proof(&self.addition, &mut bytes)?;
        write_activation_proof(&self.activation, &mut bytes)?;
        write_witness_batch_proof(&self.witness_batch, &mut bytes)?;
        Ok(bytes)
    }

    pub fn from_bytes_exact(bytes: &[u8]) -> Result<Self, GlobalMlpArtifactError> {
        let mut reader = Cursor::new(bytes);
        read_magic(&mut reader, PROOF_MAGIC)?;
        let proof = Self {
            io: read_io_proof(&mut reader)?,
            model: read_io_proof(&mut reader)?,
            multiplication: read_multiplication_proof(&mut reader)?,
            addition: read_linear_proof(&mut reader)?,
            activation: read_activation_proof(&mut reader)?,
            witness_batch: read_witness_batch_proof(&mut reader)?,
        };
        ensure_consumed(&reader, bytes.len())?;
        Ok(proof)
    }
}

fn write_srs<E: Pairing>(srs: &SRS<E>, writer: &mut Vec<u8>) -> Result<(), GlobalMlpArtifactError> {
    write_value(&srs.zksrs.com_base, writer)?;
    write_value(&srs.zksrs.blind_base, writer)?;
    write_value(&srs.pc_pp, writer)?;
    write_value(&srs.h_hat, writer)
}

fn read_srs<E: Pairing>(reader: &mut Cursor<&[u8]>) -> Result<SRS<E>, GlobalMlpArtifactError> {
    let com_base = read_value(reader)?;
    let blind_base = read_value(reader)?;
    let pc_pp = PcsParams::<E>::deserialize_compressed(&mut *reader)?;
    let h_hat = read_value(reader)?;
    Ok(SRS {
        zksrs: ZkSRS::new(com_base, blind_base),
        pc_pp,
        h_hat,
    })
}

fn write_instance<E: Pairing>(
    instance: &GlobalMlpInstance<E>,
    writer: &mut Vec<u8>,
) -> Result<(), GlobalMlpArtifactError> {
    write_usize_vec(&instance.manifest.widths, writer)?;
    write_usize(instance.manifest.commitment_bit_width, writer)?;
    write_usize(instance.manifest.relu_magnitude_bits, writer)?;
    for value in [
        instance.shape.n,
        instance.shape.witness_slots,
        instance.shape.gate_slots,
        instance.shape.relu_magnitude_bits,
        instance.shape.commitment_bit_width,
    ] {
        write_usize(value, writer)?;
    }
    write_i32_vec(&instance.public_input, writer)?;
    write_i32_vec(&instance.public_output, writer)?;
    write_value(&instance.witness_commitment, writer)?;
    write_value(&instance.io_commitment, writer)?;
    write_value(&instance.model_commitment, writer)?;
    write_architecture(&instance.architecture, writer)
}

fn read_instance<E: Pairing>(
    reader: &mut Cursor<&[u8]>,
) -> Result<GlobalMlpInstance<E>, GlobalMlpArtifactError> {
    let manifest = MlpManifest {
        widths: read_usize_vec(reader, MAX_VECTOR_LEN)?,
        commitment_bit_width: read_usize(reader)?,
        relu_magnitude_bits: read_usize(reader)?,
    };
    let shape = GlobalMlpShape {
        n: read_usize(reader)?,
        witness_slots: read_usize(reader)?,
        gate_slots: read_usize(reader)?,
        relu_magnitude_bits: read_usize(reader)?,
        commitment_bit_width: read_usize(reader)?,
    };
    Ok(GlobalMlpInstance {
        manifest,
        shape,
        public_input: read_i32_vec(reader, MAX_VECTOR_LEN)?,
        public_output: read_i32_vec(reader, MAX_VECTOR_LEN)?,
        witness_commitment: read_value(reader)?,
        io_commitment: read_value(reader)?,
        model_commitment: read_value(reader)?,
        architecture: read_architecture(reader)?,
    })
}

fn write_architecture<E: Pairing>(
    architecture: &GlobalArchitectureCommitments<E>,
    writer: &mut Vec<u8>,
) -> Result<(), GlobalMlpArtifactError> {
    write_value(&architecture.a_l, writer)?;
    write_value(&architecture.a_r, writer)?;
    write_value(&architecture.a_o, writer)?;
    for operation in operation_types() {
        let commitment = architecture
            .selectors
            .get(&operation)
            .ok_or(GlobalMlpArtifactError::Format("missing operation selector"))?;
        write_value(commitment, writer)?;
    }
    write_value(&architecture.io_selector, writer)?;
    write_value(&architecture.model_selector, writer)?;
    write_value(&architecture.negative, writer)?;
    write_vec(&architecture.output_bits, writer)?;
    write_vec(&architecture.negative_bits, writer)
}

fn read_architecture<E: Pairing>(
    reader: &mut Cursor<&[u8]>,
) -> Result<GlobalArchitectureCommitments<E>, GlobalMlpArtifactError> {
    let a_l = read_value(reader)?;
    let a_r = read_value(reader)?;
    let a_o = read_value(reader)?;
    let mut selectors = BTreeMap::new();
    for operation in operation_types() {
        selectors.insert(operation, read_value(reader)?);
    }
    Ok(GlobalArchitectureCommitments {
        a_l,
        a_r,
        a_o,
        selectors,
        io_selector: read_value(reader)?,
        model_selector: read_value(reader)?,
        negative: read_value(reader)?,
        output_bits: read_vec(reader, 128)?,
        negative_bits: read_vec(reader, 128)?,
    })
}

fn operation_types() -> [GlobalOperationType; 3] {
    [
        GlobalOperationType::Multiplication,
        GlobalOperationType::Addition,
        GlobalOperationType::Activation,
    ]
}

fn write_route<E: Pairing>(
    proof: &RoutedOpeningProof<E>,
    writer: &mut Vec<u8>,
) -> Result<(), GlobalMlpArtifactError> {
    write_transcript::<E>(&proof.transcript, writer)?;
    write_pcs_transcript(&proof.selector_opening, writer)?;
    match &proof.wiring_opening {
        Some(opening) => {
            writer.push(1);
            write_pcs_transcript(opening, writer)
        }
        None => {
            writer.push(0);
            Ok(())
        }
    }
}

fn read_route<E: Pairing>(
    reader: &mut Cursor<&[u8]>,
) -> Result<RoutedOpeningProof<E>, GlobalMlpArtifactError> {
    let transcript = read_transcript::<E>(reader)?;
    let selector_opening = read_pcs_transcript(reader)?;
    let wiring_opening = match read_byte(reader)? {
        0 => None,
        1 => Some(read_pcs_transcript(reader)?),
        _ => return Err(GlobalMlpArtifactError::Format("invalid option tag")),
    };
    Ok(RoutedOpeningProof {
        transcript,
        selector_opening,
        wiring_opening,
    })
}

fn write_io_proof<E: Pairing>(
    proof: &GlobalIoProof<E>,
    writer: &mut Vec<u8>,
) -> Result<(), GlobalMlpArtifactError> {
    write_value(&proof.value_commitment, writer)?;
    write_pcs_transcript(&proof.io_opening, writer)?;
    write_route(&proof.route, writer)
}

fn read_io_proof<E: Pairing>(
    reader: &mut Cursor<&[u8]>,
) -> Result<GlobalIoProof<E>, GlobalMlpArtifactError> {
    Ok(GlobalIoProof {
        value_commitment: read_value(reader)?,
        io_opening: read_pcs_transcript(reader)?,
        route: read_route(reader)?,
    })
}

fn write_multiplication_proof<E: Pairing>(
    proof: &GlobalMultiplicationProof<E>,
    writer: &mut Vec<u8>,
) -> Result<(), GlobalMlpArtifactError> {
    write_value(&proof.output_value_commitment, writer)?;
    write_transcript::<E>(&proof.transcript, writer)?;
    write_route(&proof.left_route, writer)?;
    write_route(&proof.right_route, writer)?;
    write_route(&proof.output_route, writer)
}

fn read_multiplication_proof<E: Pairing>(
    reader: &mut Cursor<&[u8]>,
) -> Result<GlobalMultiplicationProof<E>, GlobalMlpArtifactError> {
    Ok(GlobalMultiplicationProof {
        output_value_commitment: read_value(reader)?,
        transcript: read_transcript::<E>(reader)?,
        left_route: read_route(reader)?,
        right_route: read_route(reader)?,
        output_route: read_route(reader)?,
    })
}

fn write_linear_proof<E: Pairing>(
    proof: &GlobalLinearProof<E>,
    writer: &mut Vec<u8>,
) -> Result<(), GlobalMlpArtifactError> {
    write_vec(&proof.value_commitments, writer)?;
    write_usize(proof.routes.len(), writer)?;
    for route in &proof.routes {
        write_route(route, writer)?;
    }
    Ok(())
}

fn read_linear_proof<E: Pairing>(
    reader: &mut Cursor<&[u8]>,
) -> Result<GlobalLinearProof<E>, GlobalMlpArtifactError> {
    let value_commitments = read_vec(reader, 128)?;
    let route_count = read_bounded_len(reader, 128)?;
    let mut routes = Vec::with_capacity(route_count);
    for _ in 0..route_count {
        routes.push(read_route(reader)?);
    }
    Ok(GlobalLinearProof {
        value_commitments,
        routes,
    })
}

fn write_zero_product_proof<E: Pairing>(
    proof: &GlobalZeroProductProof<E>,
    writer: &mut Vec<u8>,
) -> Result<(), GlobalMlpArtifactError> {
    write_transcript::<E>(&proof.transcript, writer)?;
    write_route(&proof.left_route, writer)?;
    write_route(&proof.right_route, writer)
}

fn read_zero_product_proof<E: Pairing>(
    reader: &mut Cursor<&[u8]>,
) -> Result<GlobalZeroProductProof<E>, GlobalMlpArtifactError> {
    Ok(GlobalZeroProductProof {
        transcript: read_transcript::<E>(reader)?,
        left_route: read_route(reader)?,
        right_route: read_route(reader)?,
    })
}

fn write_boolean_proof<E: Pairing>(
    proof: &GlobalBooleanProof<E>,
    writer: &mut Vec<u8>,
) -> Result<(), GlobalMlpArtifactError> {
    write_transcript::<E>(&proof.transcript, writer)?;
    write_route(&proof.direct_route, writer)?;
    write_route(&proof.complement_route, writer)
}

fn read_boolean_proof<E: Pairing>(
    reader: &mut Cursor<&[u8]>,
) -> Result<GlobalBooleanProof<E>, GlobalMlpArtifactError> {
    Ok(GlobalBooleanProof {
        transcript: read_transcript::<E>(reader)?,
        direct_route: read_route(reader)?,
        complement_route: read_route(reader)?,
    })
}

fn write_range_proof<E: Pairing>(
    proof: &GlobalRangeProof<E>,
    writer: &mut Vec<u8>,
) -> Result<(), GlobalMlpArtifactError> {
    write_value(&proof.value_commitment, writer)?;
    write_vec(&proof.bit_commitments, writer)?;
    write_usize(proof.routes.len(), writer)?;
    for route in &proof.routes {
        write_route(route, writer)?;
    }
    write_usize(proof.boolean_proofs.len(), writer)?;
    for boolean in &proof.boolean_proofs {
        write_boolean_proof(boolean, writer)?;
    }
    Ok(())
}

fn read_range_proof<E: Pairing>(
    reader: &mut Cursor<&[u8]>,
) -> Result<GlobalRangeProof<E>, GlobalMlpArtifactError> {
    let value_commitment = read_value(reader)?;
    let bit_commitments = read_vec(reader, 128)?;
    let route_count = read_bounded_len(reader, 128)?;
    let mut routes = Vec::with_capacity(route_count);
    for _ in 0..route_count {
        routes.push(read_route(reader)?);
    }
    let boolean_count = read_bounded_len(reader, 128)?;
    let mut boolean_proofs = Vec::with_capacity(boolean_count);
    for _ in 0..boolean_count {
        boolean_proofs.push(read_boolean_proof(reader)?);
    }
    Ok(GlobalRangeProof {
        value_commitment,
        bit_commitments,
        routes,
        boolean_proofs,
    })
}

fn write_activation_proof<E: Pairing>(
    proof: &GlobalActivationProof<E>,
    writer: &mut Vec<u8>,
) -> Result<(), GlobalMlpArtifactError> {
    write_linear_proof(&proof.linear, writer)?;
    write_zero_product_proof(&proof.zero_product, writer)?;
    write_range_proof(&proof.output_range, writer)?;
    write_range_proof(&proof.negative_range, writer)
}

fn read_activation_proof<E: Pairing>(
    reader: &mut Cursor<&[u8]>,
) -> Result<GlobalActivationProof<E>, GlobalMlpArtifactError> {
    Ok(GlobalActivationProof {
        linear: read_linear_proof(reader)?,
        zero_product: read_zero_product_proof(reader)?,
        output_range: read_range_proof(reader)?,
        negative_range: read_range_proof(reader)?,
    })
}

fn write_witness_batch_proof<E: Pairing>(
    proof: &GlobalWitnessBatchProof<E>,
    writer: &mut Vec<u8>,
) -> Result<(), GlobalMlpArtifactError> {
    write_transcript::<E>(&proof.transcript, writer)?;
    write_pcs_transcript(&proof.opening, writer)
}

fn read_witness_batch_proof<E: Pairing>(
    reader: &mut Cursor<&[u8]>,
) -> Result<GlobalWitnessBatchProof<E>, GlobalMlpArtifactError> {
    Ok(GlobalWitnessBatchProof {
        transcript: read_transcript::<E>(reader)?,
        opening: read_pcs_transcript(reader)?,
    })
}

fn write_transcript<E: Pairing>(
    transcript: &TranSeq<E::ScalarField, E::G1>,
    writer: &mut Vec<u8>,
) -> Result<(), GlobalMlpArtifactError> {
    if transcript.pointer != 0 {
        return Err(GlobalMlpArtifactError::Format(
            "prover transcript pointer is nonzero",
        ));
    }
    write_usize(transcript.data.len(), writer)?;
    for element in &transcript.data {
        match element {
            TranElem::Field(value) => {
                writer.push(0);
                write_value(value, writer)?;
            }
            TranElem::Group(value) => {
                writer.push(1);
                write_value(value, writer)?;
            }
            TranElem::Size(value) => {
                writer.push(2);
                write_usize(*value, writer)?;
            }
            TranElem::Coin(value) => {
                writer.push(3);
                write_value(value, writer)?;
            }
        }
    }
    Ok(())
}

fn read_transcript<E: Pairing>(
    reader: &mut Cursor<&[u8]>,
) -> Result<TranSeq<E::ScalarField, E::G1>, GlobalMlpArtifactError> {
    let count = read_bounded_len(reader, MAX_TRANSCRIPT_LEN)?;
    let mut transcript = TranSeq::new();
    for _ in 0..count {
        match read_byte(reader)? {
            0 => transcript.push(TranElem::Field(read_value(reader)?)),
            1 => transcript.push(TranElem::Group(read_value(reader)?)),
            2 => transcript.push(TranElem::Size(read_usize(reader)?)),
            3 => {
                let expected = read_value(reader)?;
                if transcript.gen_challenge() != expected {
                    return Err(GlobalMlpArtifactError::Format(
                        "Fiat-Shamir coin differs from transcript",
                    ));
                }
            }
            _ => return Err(GlobalMlpArtifactError::Format("unknown transcript tag")),
        }
    }
    Ok(transcript)
}

fn write_pcs_transcript<E: Pairing>(
    transcript: &PcsTrans<E>,
    writer: &mut Vec<u8>,
) -> Result<(), GlobalMlpArtifactError> {
    write_vec(&transcript.vec_l_gt, writer)?;
    write_vec(&transcript.vec_r_gt, writer)?;
    write_vec(&transcript.vec_l_native, writer)?;
    write_vec(&transcript.vec_r_native, writer)?;
    write_value(&transcript.v_g, writer)?;
    write_value(&transcript.v_h, writer)?;
    write_value(&transcript.v_g_prime, writer)?;
    write_value(&transcript.v_h_prime, writer)?;
    write_value(&transcript.w_g, writer)?;
    write_value(&transcript.w_h, writer)?;
    write_value(&transcript.linear_f, writer)?;
    write_value(&transcript.linear_z_a, writer)?;
    write_value(&transcript.linear_z_t, writer)
}

fn read_pcs_transcript<E: Pairing>(
    reader: &mut Cursor<&[u8]>,
) -> Result<PcsTrans<E>, GlobalMlpArtifactError> {
    Ok(PcsTrans {
        vec_l_gt: read_vec(reader, 128)?,
        vec_r_gt: read_vec(reader, 128)?,
        vec_l_native: read_vec(reader, 128)?,
        vec_r_native: read_vec(reader, 128)?,
        v_g: read_value(reader)?,
        v_h: read_value(reader)?,
        v_g_prime: read_value(reader)?,
        v_h_prime: read_value(reader)?,
        w_g: read_value(reader)?,
        w_h: read_value(reader)?,
        linear_f: read_value(reader)?,
        linear_z_a: read_value(reader)?,
        linear_z_t: read_value(reader)?,
    })
}

fn write_value<T: CanonicalSerialize>(
    value: &T,
    writer: &mut Vec<u8>,
) -> Result<(), GlobalMlpArtifactError> {
    value.serialize_compressed(writer)?;
    Ok(())
}

fn read_value<T: CanonicalDeserialize>(
    reader: &mut Cursor<&[u8]>,
) -> Result<T, GlobalMlpArtifactError> {
    Ok(T::deserialize_compressed(&mut *reader)?)
}

fn write_vec<T: CanonicalSerialize>(
    values: &[T],
    writer: &mut Vec<u8>,
) -> Result<(), GlobalMlpArtifactError> {
    write_usize(values.len(), writer)?;
    for value in values {
        write_value(value, writer)?;
    }
    Ok(())
}

fn read_vec<T: CanonicalDeserialize>(
    reader: &mut Cursor<&[u8]>,
    maximum: usize,
) -> Result<Vec<T>, GlobalMlpArtifactError> {
    let count = read_bounded_len(reader, maximum)?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(read_value(reader)?);
    }
    Ok(values)
}

fn write_usize_vec(values: &[usize], writer: &mut Vec<u8>) -> Result<(), GlobalMlpArtifactError> {
    write_usize(values.len(), writer)?;
    for value in values {
        write_usize(*value, writer)?;
    }
    Ok(())
}

fn read_usize_vec(
    reader: &mut Cursor<&[u8]>,
    maximum: usize,
) -> Result<Vec<usize>, GlobalMlpArtifactError> {
    let count = read_bounded_len(reader, maximum)?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        values.push(read_usize(reader)?);
    }
    Ok(values)
}

fn write_i32_vec(values: &[i32], writer: &mut Vec<u8>) -> Result<(), GlobalMlpArtifactError> {
    write_usize(values.len(), writer)?;
    for value in values {
        writer.extend_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

fn read_i32_vec(
    reader: &mut Cursor<&[u8]>,
    maximum: usize,
) -> Result<Vec<i32>, GlobalMlpArtifactError> {
    let count = read_bounded_len(reader, maximum)?;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        let mut bytes = [0_u8; 4];
        reader
            .read_exact(&mut bytes)
            .map_err(|_| GlobalMlpArtifactError::Format("truncated i32 vector"))?;
        values.push(i32::from_le_bytes(bytes));
    }
    Ok(values)
}

fn write_usize(value: usize, writer: &mut Vec<u8>) -> Result<(), GlobalMlpArtifactError> {
    let value = u64::try_from(value).map_err(|_| GlobalMlpArtifactError::Bounds("usize"))?;
    writer.extend_from_slice(&value.to_le_bytes());
    Ok(())
}

fn read_usize(reader: &mut Cursor<&[u8]>) -> Result<usize, GlobalMlpArtifactError> {
    let mut bytes = [0_u8; 8];
    reader
        .read_exact(&mut bytes)
        .map_err(|_| GlobalMlpArtifactError::Format("truncated length"))?;
    usize::try_from(u64::from_le_bytes(bytes)).map_err(|_| GlobalMlpArtifactError::Bounds("usize"))
}

fn read_bounded_len(
    reader: &mut Cursor<&[u8]>,
    maximum: usize,
) -> Result<usize, GlobalMlpArtifactError> {
    let value = read_usize(reader)?;
    if value > maximum {
        return Err(GlobalMlpArtifactError::Bounds("vector length"));
    }
    Ok(value)
}

fn read_magic(
    reader: &mut Cursor<&[u8]>,
    expected: &[u8; 8],
) -> Result<(), GlobalMlpArtifactError> {
    let mut actual = [0_u8; 8];
    reader
        .read_exact(&mut actual)
        .map_err(|_| GlobalMlpArtifactError::Format("truncated magic"))?;
    if &actual != expected {
        return Err(GlobalMlpArtifactError::Format("magic mismatch"));
    }
    Ok(())
}

fn read_byte(reader: &mut Cursor<&[u8]>) -> Result<u8, GlobalMlpArtifactError> {
    let mut byte = [0_u8; 1];
    reader
        .read_exact(&mut byte)
        .map_err(|_| GlobalMlpArtifactError::Format("truncated tag"))?;
    Ok(byte[0])
}

fn ensure_consumed(reader: &Cursor<&[u8]>, expected: usize) -> Result<(), GlobalMlpArtifactError> {
    if reader.position() != expected as u64 {
        return Err(GlobalMlpArtifactError::Format("trailing bytes"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_poly_commit::smart_pc::SmartPC;

    use super::{ensure_consumed, read_pcs_transcript, write_pcs_transcript, Cursor};

    #[test]
    fn smart_pc_transcript_codec_round_trips_native_hiding() {
        let mut rng = ark_std::test_rng();
        let pp = SmartPC::<Bls12_381>::setup(2, &mut rng).unwrap();
        let matrix = (0..4)
            .map(|column| {
                (0..4)
                    .map(|row| Fr::from((column * 4 + row + 1) as u64))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let matrix_hiding = Fr::from(9_u64);
        let (matrix_commitment, tier_one) =
            SmartPC::<Bls12_381>::commit_full(&pp, &matrix, matrix_hiding).unwrap();
        let xl = vec![Fr::from(2_u64), Fr::from(3_u64)];
        let xr = vec![Fr::from(5_u64), Fr::from(7_u64)];
        let (value_commitment, value_hiding) = SmartPC::<Bls12_381>::eval(&pp, &matrix, &xl, &xr);
        let proof = SmartPC::<Bls12_381>::open(
            &pp,
            &matrix,
            &xl,
            &xr,
            value_commitment,
            matrix_commitment,
            &tier_one,
            value_hiding,
            matrix_hiding,
        )
        .unwrap();

        let mut encoded = Vec::new();
        write_pcs_transcript(&proof, &mut encoded).unwrap();
        let mut reader = Cursor::new(encoded.as_slice());
        let decoded = read_pcs_transcript::<Bls12_381>(&mut reader).unwrap();
        ensure_consumed(&reader, encoded.len()).unwrap();
        assert_eq!(decoded, proof);
        assert!(SmartPC::<Bls12_381>::verify(
            &pp,
            matrix_commitment,
            value_commitment,
            &xl,
            &xr,
            &decoded,
        )
        .unwrap());

        let mut truncated = Cursor::new(&encoded[..encoded.len() - 1]);
        assert!(read_pcs_transcript::<Bls12_381>(&mut truncated).is_err());
    }
}
