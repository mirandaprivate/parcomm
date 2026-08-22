//! Dense signed-byte adapter for Jolt's Dory polynomial commitment scheme.
//!
//! The physical witness is a byte prefix of a power-of-two logical vector.
//! Commitment streams primitive integers directly into Dory, and opening
//! materializes only square-root-sized field vectors.

use std::fmt::{Display, Formatter};

use jolt_crypto::{Bn254G1, DeriveSetup, Pedersen, PedersenSetup, VectorCommitment};
use jolt_dory::{
    DoryCommitment, DoryHint, DoryProof, DoryProverSetup, DoryScheme, DoryVerifierSetup,
};
use jolt_field::{Fr, FromPrimitiveInt, Invertible, MulPrimitiveInt};
use jolt_openings::{CommitmentScheme, StreamingCommitment, ZkOpeningScheme};
use jolt_poly::{EqPolynomial, MultilinearPoly};
use jolt_transcript::domain::{Label, LabelWithCount, U64Word};
use jolt_transcript::Transcript;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Errors rejected before entering Dory, plus terminal opening failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DenseI8DoryError {
    InvalidLogicalLength(usize),
    PhysicalPrefixTooLong {
        physical: usize,
        logical: usize,
    },
    PointLength {
        expected: usize,
        actual: usize,
    },
    SingularMonomialCoordinate(usize),
    Dory(String),
    EvaluationCommitmentMismatch,
    ProofLayout {
        expected_nu: usize,
        expected_sigma: usize,
        actual_nu: usize,
        actual_sigma: usize,
    },
}

impl Display for DenseI8DoryError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidLogicalLength(length) => write!(
                formatter,
                "logical length {length} is not a nonzero power of two"
            ),
            Self::PhysicalPrefixTooLong { physical, logical } => write!(
                formatter,
                "physical prefix length {physical} exceeds logical length {logical}"
            ),
            Self::PointLength { expected, actual } => write!(
                formatter,
                "evaluation point has length {actual}, expected {expected}"
            ),
            Self::SingularMonomialCoordinate(index) => write!(
                formatter,
                "monomial coordinate {index} is -1 and cannot be normalized"
            ),
            Self::Dory(message) => formatter.write_str(message),
            Self::EvaluationCommitmentMismatch => {
                formatter.write_str("Dory evaluation commitment does not open to the public claim")
            }
            Self::ProofLayout {
                expected_nu,
                expected_sigma,
                actual_nu,
                actual_sigma,
            } => write!(
                formatter,
                "Dory proof layout ({actual_nu}, {actual_sigma}) does not match canonical layout ({expected_nu}, {expected_sigma})"
            ),
        }
    }
}

impl std::error::Error for DenseI8DoryError {}

/// Dory setup plus the Pedersen basis binding its hidden evaluation.
pub struct DenseI8DorySetup {
    pub prover: DoryProverSetup,
    pub verifier: DoryVerifierSetup,
    pub evaluation: PedersenSetup<Bn254G1>,
}

impl DenseI8DorySetup {
    pub fn new(max_num_vars: usize) -> Self {
        let prover = DoryScheme::setup_prover(max_num_vars);
        let verifier = DoryScheme::verifier_setup(&prover);
        let evaluation = PedersenSetup::<Bn254G1>::derive(&prover, 1);
        Self {
            prover,
            verifier,
            evaluation,
        }
    }
}

/// A dense physical i8 prefix followed by an implicit logical zero suffix.
pub struct DenseI8Prefix<'a> {
    values: &'a [i8],
    logical_len: usize,
    num_vars: usize,
}

impl<'a> DenseI8Prefix<'a> {
    pub fn new(values: &'a [i8], logical_len: usize) -> Result<Self, DenseI8DoryError> {
        if logical_len == 0 || !logical_len.is_power_of_two() {
            return Err(DenseI8DoryError::InvalidLogicalLength(logical_len));
        }
        if values.len() > logical_len {
            return Err(DenseI8DoryError::PhysicalPrefixTooLong {
                physical: values.len(),
                logical: logical_len,
            });
        }
        Ok(Self {
            values,
            logical_len,
            num_vars: logical_len.trailing_zeros() as usize,
        })
    }

    pub fn num_vars(&self) -> usize {
        self.num_vars
    }

    /// Evaluate the dense prefix in ParComm's low-bit-first `(1, x)` basis.
    pub fn evaluate_monomial_low(&self, point_low: &[Fr]) -> Result<Fr, DenseI8DoryError> {
        if point_low.len() != self.num_vars {
            return Err(DenseI8DoryError::PointLength {
                expected: self.num_vars,
                actual: point_low.len(),
            });
        }
        let column_log = self.num_vars.div_ceil(2);
        let row_weights = monomial_tensor(&point_low[column_log..]);
        let column_weights = monomial_tensor(&point_low[..column_log]);
        Ok(self
            .fold_rows(&row_weights, column_log)
            .into_iter()
            .zip(column_weights)
            .map(|(folded, weight)| folded * weight)
            .sum())
    }
}

fn monomial_tensor(point_low: &[Fr]) -> Vec<Fr> {
    let mut weights = vec![Fr::from_u64(1)];
    for coordinate in point_low {
        let current = weights.len();
        weights.reserve(current);
        for index in 0..current {
            weights.push(weights[index] * *coordinate);
        }
    }
    weights
}

impl MultilinearPoly<Fr> for DenseI8Prefix<'_> {
    fn num_vars(&self) -> usize {
        self.num_vars
    }

    fn evaluate(&self, point: &[Fr]) -> Fr {
        assert_eq!(
            point.len(),
            self.num_vars,
            "point length must match num_vars"
        );
        let sigma = self.num_vars.div_ceil(2);
        let nu = self.num_vars - sigma;
        let left = EqPolynomial::new(point[..nu].to_vec()).evaluations();
        let right = EqPolynomial::new(point[nu..].to_vec()).evaluations();
        self.fold_rows(&left, sigma)
            .into_iter()
            .zip(right)
            .map(|(folded, basis)| folded * basis)
            .sum()
    }

    fn for_each_row(&self, sigma: usize, visit: &mut dyn FnMut(usize, &[Fr])) {
        assert!(sigma <= self.num_vars, "sigma exceeds num_vars");
        let row_width = 1_usize << sigma;
        let row_count = self.logical_len / row_width;
        let mut row = vec![Fr::from_u64(0); row_width];
        for row_index in 0..row_count {
            row.fill(Fr::from_u64(0));
            let start = row_index * row_width;
            let end = (start + row_width).min(self.values.len());
            if start < end {
                for (destination, value) in row.iter_mut().zip(&self.values[start..end]) {
                    *destination = Fr::from_i8(*value);
                }
            }
            visit(row_index, &row);
        }
    }

    fn fold_rows(&self, left: &[Fr], sigma: usize) -> Vec<Fr> {
        assert!(sigma <= self.num_vars, "sigma exceeds num_vars");
        let row_width = 1_usize << sigma;
        let row_count = self.logical_len / row_width;
        assert_eq!(left.len(), row_count, "left vector has the wrong length");
        let physical_rows = self.values.len().div_ceil(row_width);
        if physical_rows == 0 {
            return vec![Fr::from_u64(0); row_width];
        }
        let workers = rayon::current_num_threads().min(physical_rows);
        (0..workers)
            .into_par_iter()
            .map(|worker| {
                let start_row = worker * physical_rows / workers;
                let end_row = (worker + 1) * physical_rows / workers;
                let mut output = vec![Fr::from_u64(0); row_width];
                for row_index in start_row..end_row {
                    let start = row_index * row_width;
                    let end = (start + row_width).min(self.values.len());
                    let row_weight = left[row_index];
                    let scaled: [Fr; 256] = core::array::from_fn(|encoded| {
                        row_weight.mul_i64((encoded as u8 as i8) as i64)
                    });
                    for (destination, value) in output.iter_mut().zip(&self.values[start..end]) {
                        *destination += scaled[*value as u8 as usize];
                    }
                }
                output
            })
            .reduce(
                || vec![Fr::from_u64(0); row_width],
                |mut left, right| {
                    left.iter_mut()
                        .zip(right)
                        .for_each(|(destination, value)| *destination += value);
                    left
                },
            )
    }
}

/// Conversion from ParComm's monomial basis to Jolt's standard MLE basis.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MleNormalization {
    pub standard_point_msb: Vec<Fr>,
    pub monomial_scale: Fr,
    pub monomial_scale_inverse: Fr,
}

/// Normalize a low-bit-first `(1, x)` point and reverse it for Jolt.
pub fn normalize_monomial_point(
    monomial_point_low: &[Fr],
) -> Result<MleNormalization, DenseI8DoryError> {
    let mut standard_point_low = Vec::with_capacity(monomial_point_low.len());
    let mut scale = Fr::from_u64(1);
    let mut scale_inverse = Fr::from_u64(1);
    for (index, coordinate) in monomial_point_low.iter().copied().enumerate() {
        let denominator = Fr::from_u64(1) + coordinate;
        let Some(denominator_inverse) = denominator.inverse() else {
            return Err(DenseI8DoryError::SingularMonomialCoordinate(index));
        };
        standard_point_low.push(coordinate * denominator_inverse);
        scale *= denominator;
        scale_inverse *= denominator_inverse;
    }
    standard_point_low.reverse();
    Ok(MleNormalization {
        standard_point_msb: standard_point_low,
        monomial_scale: scale,
        monomial_scale_inverse: scale_inverse,
    })
}

/// Stream the i8 prefix and implicit zero rows into a hiding Dory commitment.
pub fn commit_dense_i8_zk(
    source: &DenseI8Prefix<'_>,
    setup: &DoryProverSetup,
) -> (DoryCommitment, DoryHint) {
    let sigma = source.num_vars.div_ceil(2);
    let row_width = 1_usize << sigma;
    let total_rows = source.logical_len / row_width;
    let physical_rows = source.values.len().div_ceil(row_width);
    let primitive_count = physical_rows * row_width;
    let mut partial = DoryScheme::begin(setup);
    if primitive_count != 0 {
        DoryScheme::feed_i128_rows_with(
            &mut partial,
            |index| source.values.get(index).copied().unwrap_or(0) as i128,
            primitive_count,
            row_width,
            setup,
        );
    }
    DoryScheme::feed_zeros(&mut partial, row_width, total_rows - physical_rows, setup);
    DoryScheme::finish_zk(partial, setup)
}

/// Public opening data for a dense field vector.
///
/// The blind is public because this adapter binds the hidden Dory evaluation
/// commitment to an evaluation that is already part of the public statement.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct FieldVectorDoryProof {
    pub opening: DoryProof,
    pub evaluation_blind: Fr,
}

/// A Dory opening whose evaluation and blinding stay inside a BlindFold proof.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenFieldVectorDoryProof {
    pub opening: DoryProof,
}

/// Dense-i8 Dory opening whose normalized evaluation remains hidden.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct HiddenDenseI8DoryProof {
    pub opening: DoryProof,
}

fn validate_field_vector_shape(
    values_len: usize,
    point_len: Option<usize>,
) -> Result<usize, DenseI8DoryError> {
    if values_len == 0 || !values_len.is_power_of_two() {
        return Err(DenseI8DoryError::InvalidLogicalLength(values_len));
    }
    let num_vars = values_len.trailing_zeros() as usize;
    if let Some(actual) = point_len {
        if actual != num_vars {
            return Err(DenseI8DoryError::PointLength {
                expected: num_vars,
                actual,
            });
        }
    }
    Ok(num_vars)
}

fn validate_dory_proof_layout(proof: &DoryProof, num_vars: usize) -> Result<(), DenseI8DoryError> {
    let expected_sigma = num_vars.div_ceil(2);
    let expected_nu = num_vars - expected_sigma;
    if proof.0.nu != expected_nu || proof.0.sigma != expected_sigma {
        return Err(DenseI8DoryError::ProofLayout {
            expected_nu,
            expected_sigma,
            actual_nu: proof.0.nu,
            actual_sigma: proof.0.sigma,
        });
    }
    Ok(())
}

/// Commit a power-of-two field evaluation table in Dory's hiding mode.
pub fn commit_field_vector_zk(
    values: &[Fr],
    setup: &DoryProverSetup,
) -> Result<(DoryCommitment, DoryHint), DenseI8DoryError> {
    validate_field_vector_shape(values.len(), None)?;
    DoryScheme::commit_zk(values, setup).map_err(|error| DenseI8DoryError::Dory(error.to_string()))
}

fn bind_field_vector_statement<T: Transcript<Challenge = Fr>>(
    transcript: &mut T,
    role: &[u8],
    outer_binding: &[u8],
    commitment: &DoryCommitment,
    logical_len: usize,
    standard_point_msb: &[Fr],
    standard_eval: Fr,
) {
    transcript.append(&Label(b"parcomm_field_vector_dory_v1"));
    transcript.append(&LabelWithCount(b"role", role.len() as u64));
    transcript.append_bytes(role);
    transcript.append(&LabelWithCount(
        b"outer_binding",
        outer_binding.len() as u64,
    ));
    transcript.append_bytes(outer_binding);
    transcript.append_labeled(b"logical_len", &U64Word(logical_len as u64));
    transcript.append_labeled(b"commitment", commitment);
    transcript.append_values(b"standard_point", standard_point_msb);
    transcript.append_labeled(b"standard_eval", &standard_eval);
}

/// Open a dense field vector at a Jolt-standard, MSB-first MLE point.
#[allow(clippy::too_many_arguments)]
pub fn open_field_vector_zk<T: Transcript<Challenge = Fr>>(
    values: &[Fr],
    commitment: &DoryCommitment,
    hint: DoryHint,
    standard_point_msb: &[Fr],
    standard_eval: Fr,
    role: &[u8],
    outer_binding: &[u8],
    setup: &DoryProverSetup,
    transcript: &mut T,
) -> Result<FieldVectorDoryProof, DenseI8DoryError> {
    validate_field_vector_shape(values.len(), Some(standard_point_msb.len()))?;
    bind_field_vector_statement(
        transcript,
        role,
        outer_binding,
        commitment,
        values.len(),
        standard_point_msb,
        standard_eval,
    );
    let (opening, _evaluation_commitment, evaluation_blind) = DoryScheme::open_zk(
        values,
        standard_point_msb,
        standard_eval,
        setup,
        hint,
        transcript,
    )
    .map_err(|error| DenseI8DoryError::Dory(error.to_string()))?;
    Ok(FieldVectorDoryProof {
        opening,
        evaluation_blind,
    })
}

/// Verify a dense field-vector opening and its public evaluation binding.
#[allow(clippy::too_many_arguments)]
pub fn verify_field_vector_zk<T: Transcript<Challenge = Fr>>(
    commitment: &DoryCommitment,
    proof: &FieldVectorDoryProof,
    logical_len: usize,
    standard_point_msb: &[Fr],
    standard_eval: Fr,
    role: &[u8],
    outer_binding: &[u8],
    verifier_setup: &DoryVerifierSetup,
    evaluation_setup: &PedersenSetup<Bn254G1>,
    transcript: &mut T,
) -> Result<(), DenseI8DoryError> {
    let num_vars = validate_field_vector_shape(logical_len, Some(standard_point_msb.len()))?;
    validate_dory_proof_layout(&proof.opening, num_vars)?;
    bind_field_vector_statement(
        transcript,
        role,
        outer_binding,
        commitment,
        logical_len,
        standard_point_msb,
        standard_eval,
    );
    let evaluation_commitment = DoryScheme::verify_zk(
        commitment,
        standard_point_msb,
        &proof.opening,
        verifier_setup,
        transcript,
    )
    .map_err(|error| DenseI8DoryError::Dory(error.to_string()))?;
    if !Pedersen::<Bn254G1>::verify(
        evaluation_setup,
        &evaluation_commitment,
        &[standard_eval],
        &proof.evaluation_blind,
    ) {
        return Err(DenseI8DoryError::EvaluationCommitmentMismatch);
    }
    Ok(())
}

fn bind_hidden_field_vector_statement<T: Transcript<Challenge = Fr>>(
    transcript: &mut T,
    role: &[u8],
    outer_binding: &[u8],
    commitment: &DoryCommitment,
    logical_len: usize,
    standard_point_msb: &[Fr],
) {
    transcript.append(&Label(b"parcomm_hidden_dory_v1"));
    transcript.append(&LabelWithCount(b"role", role.len() as u64));
    transcript.append_bytes(role);
    transcript.append(&LabelWithCount(
        b"outer_binding",
        outer_binding.len() as u64,
    ));
    transcript.append_bytes(outer_binding);
    transcript.append_labeled(b"logical_len", &U64Word(logical_len as u64));
    transcript.append_labeled(b"commitment", commitment);
    transcript.append_values(b"standard_point", standard_point_msb);
}

/// Open a field vector while keeping the evaluation scalar and its blind private.
#[allow(clippy::too_many_arguments)]
pub fn open_hidden_field_vector_zk<T: Transcript<Challenge = Fr>>(
    values: &[Fr],
    commitment: &DoryCommitment,
    hint: DoryHint,
    standard_point_msb: &[Fr],
    standard_eval: Fr,
    role: &[u8],
    outer_binding: &[u8],
    setup: &DoryProverSetup,
    transcript: &mut T,
) -> Result<(HiddenFieldVectorDoryProof, Bn254G1, Fr), DenseI8DoryError> {
    validate_field_vector_shape(values.len(), Some(standard_point_msb.len()))?;
    bind_hidden_field_vector_statement(
        transcript,
        role,
        outer_binding,
        commitment,
        values.len(),
        standard_point_msb,
    );
    let (opening, evaluation_commitment, evaluation_blind) = DoryScheme::open_zk(
        values,
        standard_point_msb,
        standard_eval,
        setup,
        hint,
        transcript,
    )
    .map_err(|error| DenseI8DoryError::Dory(error.to_string()))?;
    Ok((
        HiddenFieldVectorDoryProof { opening },
        evaluation_commitment,
        evaluation_blind,
    ))
}

/// Verify a hidden field-vector opening and return its Pedersen evaluation commitment.
#[allow(clippy::too_many_arguments)]
pub fn verify_hidden_field_vector_zk<T: Transcript<Challenge = Fr>>(
    commitment: &DoryCommitment,
    proof: &HiddenFieldVectorDoryProof,
    logical_len: usize,
    standard_point_msb: &[Fr],
    role: &[u8],
    outer_binding: &[u8],
    verifier_setup: &DoryVerifierSetup,
    transcript: &mut T,
) -> Result<Bn254G1, DenseI8DoryError> {
    let num_vars = validate_field_vector_shape(logical_len, Some(standard_point_msb.len()))?;
    validate_dory_proof_layout(&proof.opening, num_vars)?;
    bind_hidden_field_vector_statement(
        transcript,
        role,
        outer_binding,
        commitment,
        logical_len,
        standard_point_msb,
    );
    DoryScheme::verify_zk(
        commitment,
        standard_point_msb,
        &proof.opening,
        verifier_setup,
        transcript,
    )
    .map_err(|error| DenseI8DoryError::Dory(error.to_string()))
}

fn bind_hidden_dense_i8_statement<T: Transcript<Challenge = Fr>>(
    transcript: &mut T,
    outer_binding: &[u8],
    commitment: &DoryCommitment,
    logical_len: usize,
    standard_point_msb: &[Fr],
) {
    transcript.append(&Label(b"parcomm_hidden_i8_v1"));
    transcript.append(&LabelWithCount(
        b"outer_binding",
        outer_binding.len() as u64,
    ));
    transcript.append_bytes(outer_binding);
    transcript.append_labeled(b"logical_len", &U64Word(logical_len as u64));
    transcript.append_labeled(b"commitment", commitment);
    transcript.append_values(b"standard_point", standard_point_msb);
}

/// Open a dense i8 source while retaining the normalized evaluation and blind.
#[allow(clippy::too_many_arguments)]
pub fn open_hidden_dense_i8_zk<T: Transcript<Challenge = Fr>>(
    source: &DenseI8Prefix<'_>,
    commitment: &DoryCommitment,
    hint: DoryHint,
    monomial_point_low: &[Fr],
    monomial_eval: Fr,
    outer_binding: &[u8],
    setup: &DoryProverSetup,
    transcript: &mut T,
) -> Result<(HiddenDenseI8DoryProof, Bn254G1, Fr), DenseI8DoryError> {
    if monomial_point_low.len() != source.num_vars {
        return Err(DenseI8DoryError::PointLength {
            expected: source.num_vars,
            actual: monomial_point_low.len(),
        });
    }
    let normalization = normalize_monomial_point(monomial_point_low)?;
    let standard_eval = monomial_eval * normalization.monomial_scale_inverse;
    bind_hidden_dense_i8_statement(
        transcript,
        outer_binding,
        commitment,
        source.logical_len,
        &normalization.standard_point_msb,
    );
    let opening_source = ClaimedOpeningSource {
        source,
        point: &normalization.standard_point_msb,
        evaluation: standard_eval,
    };
    let (opening, evaluation_commitment, evaluation_blind) = DoryScheme::open_zk(
        &opening_source,
        &normalization.standard_point_msb,
        standard_eval,
        setup,
        hint,
        transcript,
    )
    .map_err(|error| DenseI8DoryError::Dory(error.to_string()))?;
    Ok((
        HiddenDenseI8DoryProof { opening },
        evaluation_commitment,
        evaluation_blind,
    ))
}

/// Verify a hidden dense-i8 opening and return its normalized evaluation commitment.
#[allow(clippy::too_many_arguments)]
pub fn verify_hidden_dense_i8_zk<T: Transcript<Challenge = Fr>>(
    commitment: &DoryCommitment,
    proof: &HiddenDenseI8DoryProof,
    logical_len: usize,
    monomial_point_low: &[Fr],
    outer_binding: &[u8],
    verifier_setup: &DoryVerifierSetup,
    transcript: &mut T,
) -> Result<Bn254G1, DenseI8DoryError> {
    if logical_len == 0 || !logical_len.is_power_of_two() {
        return Err(DenseI8DoryError::InvalidLogicalLength(logical_len));
    }
    let num_vars = logical_len.trailing_zeros() as usize;
    if monomial_point_low.len() != num_vars {
        return Err(DenseI8DoryError::PointLength {
            expected: num_vars,
            actual: monomial_point_low.len(),
        });
    }
    validate_dory_proof_layout(&proof.opening, num_vars)?;
    let normalization = normalize_monomial_point(monomial_point_low)?;
    bind_hidden_dense_i8_statement(
        transcript,
        outer_binding,
        commitment,
        logical_len,
        &normalization.standard_point_msb,
    );
    DoryScheme::verify_zk(
        commitment,
        &normalization.standard_point_msb,
        &proof.opening,
        verifier_setup,
        transcript,
    )
    .map_err(|error| DenseI8DoryError::Dory(error.to_string()))
}

/// Public opening data; revealing this blind is safe because Packed EO reveals the value.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct DenseI8DoryProof {
    pub opening: DoryProof,
    pub evaluation_blind: Fr,
}

struct ClaimedOpeningSource<'source, 'values> {
    source: &'source DenseI8Prefix<'values>,
    point: &'source [Fr],
    evaluation: Fr,
}

impl MultilinearPoly<Fr> for ClaimedOpeningSource<'_, '_> {
    fn num_vars(&self) -> usize {
        self.source.num_vars()
    }

    fn evaluate(&self, point: &[Fr]) -> Fr {
        assert_eq!(
            point, self.point,
            "Dory requested an unexpected opening point"
        );
        self.evaluation
    }

    fn for_each_row(&self, sigma: usize, visit: &mut dyn FnMut(usize, &[Fr])) {
        self.source.for_each_row(sigma, visit);
    }

    fn fold_rows(&self, left: &[Fr], sigma: usize) -> Vec<Fr> {
        self.source.fold_rows(left, sigma)
    }
}

fn bind_public_statement<T: Transcript<Challenge = Fr>>(
    transcript: &mut T,
    outer_binding: &[u8],
    commitment: &DoryCommitment,
    logical_len: usize,
    standard_point_msb: &[Fr],
    standard_eval: Fr,
) {
    transcript.append(&Label(b"parcomm_dense_i8_dory_v1"));
    transcript.append(&LabelWithCount(
        b"outer_binding",
        outer_binding.len() as u64,
    ));
    transcript.append_bytes(outer_binding);
    transcript.append_labeled(b"logical_len", &U64Word(logical_len as u64));
    transcript.append_labeled(b"commitment", commitment);
    transcript.append_values(b"standard_point", standard_point_msb);
    transcript.append_labeled(b"standard_eval", &standard_eval);
}

/// Open the dense i8 source at a public monomial-basis claim.
#[allow(clippy::too_many_arguments)]
pub fn open_dense_i8_zk<T: Transcript<Challenge = Fr>>(
    source: &DenseI8Prefix<'_>,
    commitment: &DoryCommitment,
    hint: DoryHint,
    monomial_point_low: &[Fr],
    monomial_eval: Fr,
    outer_binding: &[u8],
    setup: &DoryProverSetup,
    transcript: &mut T,
) -> Result<DenseI8DoryProof, DenseI8DoryError> {
    if monomial_point_low.len() != source.num_vars {
        return Err(DenseI8DoryError::PointLength {
            expected: source.num_vars,
            actual: monomial_point_low.len(),
        });
    }
    let normalization = normalize_monomial_point(monomial_point_low)?;
    let standard_eval = monomial_eval * normalization.monomial_scale_inverse;
    bind_public_statement(
        transcript,
        outer_binding,
        commitment,
        source.logical_len,
        &normalization.standard_point_msb,
        standard_eval,
    );
    let opening_source = ClaimedOpeningSource {
        source,
        point: &normalization.standard_point_msb,
        evaluation: standard_eval,
    };
    let (opening, _evaluation_commitment, evaluation_blind) = DoryScheme::open_zk(
        &opening_source,
        &normalization.standard_point_msb,
        standard_eval,
        setup,
        hint,
        transcript,
    )
    .map_err(|error| DenseI8DoryError::Dory(error.to_string()))?;
    Ok(DenseI8DoryProof {
        opening,
        evaluation_blind,
    })
}

/// Verify Dory and bind its hidden evaluation commitment to the public claim.
#[allow(clippy::too_many_arguments)]
pub fn verify_dense_i8_zk<T: Transcript<Challenge = Fr>>(
    commitment: &DoryCommitment,
    proof: &DenseI8DoryProof,
    logical_len: usize,
    monomial_point_low: &[Fr],
    monomial_eval: Fr,
    outer_binding: &[u8],
    verifier_setup: &DoryVerifierSetup,
    evaluation_setup: &PedersenSetup<Bn254G1>,
    transcript: &mut T,
) -> Result<(), DenseI8DoryError> {
    if logical_len == 0 || !logical_len.is_power_of_two() {
        return Err(DenseI8DoryError::InvalidLogicalLength(logical_len));
    }
    let num_vars = logical_len.trailing_zeros() as usize;
    if monomial_point_low.len() != num_vars {
        return Err(DenseI8DoryError::PointLength {
            expected: num_vars,
            actual: monomial_point_low.len(),
        });
    }
    validate_dory_proof_layout(&proof.opening, num_vars)?;
    let normalization = normalize_monomial_point(monomial_point_low)?;
    let standard_eval = monomial_eval * normalization.monomial_scale_inverse;
    bind_public_statement(
        transcript,
        outer_binding,
        commitment,
        logical_len,
        &normalization.standard_point_msb,
        standard_eval,
    );
    let evaluation_commitment = DoryScheme::verify_zk(
        commitment,
        &normalization.standard_point_msb,
        &proof.opening,
        verifier_setup,
        transcript,
    )
    .map_err(|error| DenseI8DoryError::Dory(error.to_string()))?;
    if !Pedersen::<Bn254G1>::verify(
        evaluation_setup,
        &evaluation_commitment,
        &[standard_eval],
        &proof.evaluation_blind,
    ) {
        return Err(DenseI8DoryError::EvaluationCommitmentMismatch);
    }
    Ok(())
}
