//! Defines the data structures used in SMART-PC
//! use crate::*;
//! AdditiveGroup, AffineRepr
//!
// use ark_ff::ToConstraintField;
#![deny(missing_docs, trivial_numeric_casts, variant_size_differences)]
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_serialize::{
    CanonicalDeserialize, CanonicalSerialize, Compress, SerializationError, Valid, Validate,
};
use ark_std::io::{Read, Write};
use ark_std::{
    ops::{Add, Mul},
    Zero,
};

// `UniversalParams` are the universal parameters for the SMART-PC scheme.
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
/// Universal structured reference string (SRS) parameters containing all group bases
/// and vectors required for commitments and openings in the SMART-PC scheme.
pub struct UniversalParams<E: Pairing> {
    /// Maximum supported matrix dimension / polynomial size upper bound.
    pub q: usize,
    /// Number of variables (e.g. multilinear variables or vector length).
    pub num_vars: usize,
    /// Generator (or normalized base) in G1.
    pub g_hat: E::G1,
    /// Generator (or normalized base) in G2.
    pub h_hat: E::G2,
    /// Randomization element s^h in G2.
    pub s_h: E::G2,
    /// Randomization element (s^q)^h in G2.
    pub sq_h: E::G2,
    /// Auxiliary random base in G1 (for encoding / hiding).
    pub nu_g: E::G1,
    /// Auxiliary random base in G2.
    pub nu_h: E::G2,
    /// Public base g_0 in G1.
    pub g_0: E::G1,
    /// Variant base \~g in G1 (used for hiding / separation).
    pub tilde_g: E::G1,
    /// G2 hiding base satisfying `e(g_0, tilde_h) = tilde_u`.
    pub tilde_h: E::G2,
    /// Base element u = e(g, h) in GT.
    pub u: PairingOutput<E>,
    /// Independent GT base \~u used for randomness.
    pub tilde_u: PairingOutput<E>,
    /// Vector of G1 bases for primary commitments.
    pub vec_g: Vec<E::G1>,
    /// Vector of G2 bases for primary commitments.
    pub vec_h: Vec<E::G2>,
    /// G1 bases used in opening phase (g').
    pub vec_g_prime: Vec<E::G1>,
    /// Additional G1 bases for randomization / zero-knowledge (nu_g).
    pub vec_nu_g: Vec<E::G1>,
    /// Additional G2 bases for randomization / zero-knowledge (nu_h).
    pub vec_nu_h: Vec<E::G2>,
}

impl<E: Pairing> Valid for UniversalParams<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.q.check()?;
        self.num_vars.check()?;
        self.g_hat.check()?;
        self.h_hat.check()?;
        self.s_h.check()?;
        self.sq_h.check()?;
        self.nu_g.check()?;
        self.nu_h.check()?;
        self.g_0.check()?;
        self.tilde_g.check()?;
        self.tilde_h.check()?;
        self.u.check()?;
        self.tilde_u.check()?;
        self.vec_g.check()?;
        self.vec_h.check()?;
        self.vec_g_prime.check()?;
        self.vec_nu_g.check()?;
        self.vec_nu_h.check()?;
        if self.g_0.is_zero()
            || self.tilde_h.is_zero()
            || self.u.is_zero()
            || self.tilde_u.is_zero()
            || E::pairing(self.g_0, self.h_hat) != self.u
            || E::pairing(self.g_0, self.tilde_h) != self.tilde_u
            || E::pairing(self.tilde_g, self.h_hat) != self.tilde_u
        {
            return Err(SerializationError::InvalidData);
        }
        Ok(())
    }
}

impl<E: Pairing> CanonicalSerialize for UniversalParams<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.q.serialize_with_mode(&mut writer, compress)?;
        self.num_vars.serialize_with_mode(&mut writer, compress)?;
        self.g_hat.serialize_with_mode(&mut writer, compress)?;
        self.h_hat.serialize_with_mode(&mut writer, compress)?;
        self.s_h.serialize_with_mode(&mut writer, compress)?;
        self.sq_h.serialize_with_mode(&mut writer, compress)?;
        self.nu_g.serialize_with_mode(&mut writer, compress)?;
        self.nu_h.serialize_with_mode(&mut writer, compress)?;
        self.g_0.serialize_with_mode(&mut writer, compress)?;
        self.tilde_g.serialize_with_mode(&mut writer, compress)?;
        self.tilde_h.serialize_with_mode(&mut writer, compress)?;
        self.u.serialize_with_mode(&mut writer, compress)?;
        self.tilde_u.serialize_with_mode(&mut writer, compress)?;
        self.vec_g.serialize_with_mode(&mut writer, compress)?;
        self.vec_h.serialize_with_mode(&mut writer, compress)?;
        self.vec_g_prime
            .serialize_with_mode(&mut writer, compress)?;
        self.vec_nu_g.serialize_with_mode(&mut writer, compress)?;
        self.vec_nu_h.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.q.serialized_size(compress)
            + self.num_vars.serialized_size(compress)
            + self.g_hat.serialized_size(compress)
            + self.h_hat.serialized_size(compress)
            + self.s_h.serialized_size(compress)
            + self.sq_h.serialized_size(compress)
            + self.nu_g.serialized_size(compress)
            + self.nu_h.serialized_size(compress)
            + self.g_0.serialized_size(compress)
            + self.tilde_g.serialized_size(compress)
            + self.tilde_h.serialized_size(compress)
            + self.u.serialized_size(compress)
            + self.tilde_u.serialized_size(compress)
            + self.vec_g.serialized_size(compress)
            + self.vec_h.serialized_size(compress)
            + self.vec_g_prime.serialized_size(compress)
            + self.vec_nu_g.serialized_size(compress)
            + self.vec_nu_h.serialized_size(compress)
    }
}

impl<E: Pairing> CanonicalDeserialize for UniversalParams<E> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let q = usize::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        let num_vars = usize::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        let g_hat = E::G1::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        let h_hat = E::G2::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        let s_h = E::G2::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        let sq_h = E::G2::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        let nu_g = E::G1::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        let nu_h = E::G2::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        let g_0 = E::G1::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        let tilde_g = E::G1::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        let tilde_h = E::G2::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        let u = PairingOutput::<E>::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        let tilde_u =
            PairingOutput::<E>::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        let vec_g = Vec::<E::G1>::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        let vec_h = Vec::<E::G2>::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        let vec_g_prime = Vec::<E::G1>::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        let vec_nu_g = Vec::<E::G1>::deserialize_with_mode(&mut reader, compress, Validate::No)?;
        let vec_nu_h = Vec::<E::G2>::deserialize_with_mode(&mut reader, compress, Validate::No)?;

        let result = Self {
            q,
            num_vars,
            g_hat,
            h_hat,
            s_h,
            sq_h,
            nu_g,
            nu_h,
            g_0,
            tilde_g,
            tilde_h,
            u,
            tilde_u,
            vec_g,
            vec_h,
            vec_g_prime,
            vec_nu_g,
            vec_nu_h,
        };
        if let Validate::Yes = validate {
            result.check()?;
        }

        Ok(result)
    }
}

/// One row-fold message represented and hidden in its source groups.
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Copy(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
pub struct NativeFoldMessage<E: Pairing> {
    /// The G1 component paired with the reduced G2 basis.
    pub g1: E::G1,
    /// The G2 component paired with `g_0`.
    pub g2: E::G2,
}

impl<E: Pairing> Valid for NativeFoldMessage<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.g1.check()?;
        self.g2.check()
    }
}

impl<E: Pairing> CanonicalSerialize for NativeFoldMessage<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.g1.serialize_with_mode(&mut writer, compress)?;
        self.g2.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.g1.serialized_size(compress) + self.g2.serialized_size(compress)
    }
}

impl<E: Pairing> CanonicalDeserialize for NativeFoldMessage<E> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let result = Self {
            g1: E::G1::deserialize_with_mode(&mut reader, compress, Validate::No)?,
            g2: E::G2::deserialize_with_mode(&mut reader, compress, Validate::No)?,
        };
        if let Validate::Yes = validate {
            result.check()?;
        }
        Ok(result)
    }
}

// `Trans` are the transcript for the SMART-PC scheme.
#[derive(Derivative)]
#[derivative(
    Clone(bound = ""),
    Debug(bound = ""),
    PartialEq(bound = ""),
    Eq(bound = "")
)]
/// Transcript capturing source-group fold messages and the final linear proof.
pub struct Trans<E: Pairing> {
    /// Column-fold L messages, which are irreducible GT inner-pairing products.
    pub vec_l_gt: Vec<PairingOutput<E>>,
    /// Column-fold R messages, which are irreducible GT inner-pairing products.
    pub vec_r_gt: Vec<PairingOutput<E>>,
    /// Row-fold L messages hidden in their source groups.
    pub vec_l_native: Vec<NativeFoldMessage<E>>,
    /// Row-fold R messages hidden in their source groups.
    pub vec_r_native: Vec<NativeFoldMessage<E>>,
    /// Commitment component V_G in G1.
    pub v_g: E::G1,
    /// Commitment component V_H in G2.
    pub v_h: E::G2,
    /// Extended component V_G' in G1.
    pub v_g_prime: E::G1,
    /// Extended component V_H' in G2.
    pub v_h_prime: E::G2,
    /// Intermediate commitment W_G in G1.
    pub w_g: E::G1,
    /// Intermediate commitment W_H in G1.
    pub w_h: E::G1,
    /// Announcement for the final linear representation proof in GT.
    pub linear_f: PairingOutput<E>,
    /// Response for the reduced matrix value.
    pub linear_z_a: E::ScalarField,
    /// Response for the accumulated hiding factor.
    pub linear_z_t: E::ScalarField,
}

impl<E: Pairing> Trans<E> {
    /// Create an empty transcript (vectors cleared, group elements set to default, scalars zero).
    pub fn new() -> Self {
        Self {
            vec_l_gt: Vec::new(),
            vec_r_gt: Vec::new(),
            vec_l_native: Vec::new(),
            vec_r_native: Vec::new(),
            v_g: <E::G1 as Default>::default(),
            v_h: <E::G2 as Default>::default(),
            v_g_prime: <E::G1 as Default>::default(),
            v_h_prime: <E::G2 as Default>::default(),
            w_g: <E::G1 as Default>::default(),
            w_h: <E::G1 as Default>::default(),
            linear_f: PairingOutput::<E>::default(),
            linear_z_a: E::ScalarField::zero(),
            linear_z_t: E::ScalarField::zero(),
        }
    }

    /// Returns whether every folding-message vector has the expected length.
    pub fn has_rounds(&self, row_rounds: usize, column_rounds: usize) -> bool {
        self.vec_l_gt.len() == column_rounds
            && self.vec_r_gt.len() == column_rounds
            && self.vec_l_native.len() == row_rounds
            && self.vec_r_native.len() == row_rounds
    }
}

impl<E: Pairing> Valid for Trans<E> {
    fn check(&self) -> Result<(), SerializationError> {
        self.vec_l_gt.check()?;
        self.vec_r_gt.check()?;
        self.vec_l_native.check()?;
        self.vec_r_native.check()?;
        self.v_g.check()?;
        self.v_h.check()?;
        self.v_g_prime.check()?;
        self.v_h_prime.check()?;
        self.w_g.check()?;
        self.w_h.check()?;
        self.linear_f.check()?;
        self.linear_z_a.check()?;
        self.linear_z_t.check()?;
        Ok(())
    }
}

impl<E: Pairing> CanonicalSerialize for Trans<E> {
    fn serialize_with_mode<W: Write>(
        &self,
        mut writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.vec_l_gt.serialize_with_mode(&mut writer, compress)?;
        self.vec_r_gt.serialize_with_mode(&mut writer, compress)?;
        self.vec_l_native
            .serialize_with_mode(&mut writer, compress)?;
        self.vec_r_native
            .serialize_with_mode(&mut writer, compress)?;
        self.v_g.serialize_with_mode(&mut writer, compress)?;
        self.v_h.serialize_with_mode(&mut writer, compress)?;
        self.v_g_prime.serialize_with_mode(&mut writer, compress)?;
        self.v_h_prime.serialize_with_mode(&mut writer, compress)?;
        self.w_g.serialize_with_mode(&mut writer, compress)?;
        self.w_h.serialize_with_mode(&mut writer, compress)?;
        self.linear_f.serialize_with_mode(&mut writer, compress)?;
        self.linear_z_a.serialize_with_mode(&mut writer, compress)?;
        self.linear_z_t.serialize_with_mode(&mut writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.vec_l_gt.serialized_size(compress)
            + self.vec_r_gt.serialized_size(compress)
            + self.vec_l_native.serialized_size(compress)
            + self.vec_r_native.serialized_size(compress)
            + self.v_g.serialized_size(compress)
            + self.v_h.serialized_size(compress)
            + self.v_g_prime.serialized_size(compress)
            + self.v_h_prime.serialized_size(compress)
            + self.w_g.serialized_size(compress)
            + self.w_h.serialized_size(compress)
            + self.linear_f.serialized_size(compress)
            + self.linear_z_a.serialized_size(compress)
            + self.linear_z_t.serialized_size(compress)
    }
}

impl<E: Pairing> CanonicalDeserialize for Trans<E> {
    fn deserialize_with_mode<R: Read>(
        mut reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        let result = Self {
            vec_l_gt: Vec::<PairingOutput<E>>::deserialize_with_mode(
                &mut reader,
                compress,
                Validate::No,
            )?,
            vec_r_gt: Vec::<PairingOutput<E>>::deserialize_with_mode(
                &mut reader,
                compress,
                Validate::No,
            )?,
            vec_l_native: Vec::<NativeFoldMessage<E>>::deserialize_with_mode(
                &mut reader,
                compress,
                Validate::No,
            )?,
            vec_r_native: Vec::<NativeFoldMessage<E>>::deserialize_with_mode(
                &mut reader,
                compress,
                Validate::No,
            )?,
            v_g: E::G1::deserialize_with_mode(&mut reader, compress, Validate::No)?,
            v_h: E::G2::deserialize_with_mode(&mut reader, compress, Validate::No)?,
            v_g_prime: E::G1::deserialize_with_mode(&mut reader, compress, Validate::No)?,
            v_h_prime: E::G2::deserialize_with_mode(&mut reader, compress, Validate::No)?,
            w_g: E::G1::deserialize_with_mode(&mut reader, compress, Validate::No)?,
            w_h: E::G1::deserialize_with_mode(&mut reader, compress, Validate::No)?,
            linear_f: PairingOutput::<E>::deserialize_with_mode(
                &mut reader,
                compress,
                Validate::No,
            )?,
            linear_z_a: E::ScalarField::deserialize_with_mode(&mut reader, compress, Validate::No)?,
            linear_z_t: E::ScalarField::deserialize_with_mode(&mut reader, compress, Validate::No)?,
        };
        if let Validate::Yes = validate {
            result.check()?;
        }
        Ok(result)
    }
}

/// A deferred pairing row whose two source-group sides are sparse linear combinations.
pub struct SparsePairingRow<E: Pairing> {
    /// Scalar coefficients and G1 points to combine before pairing.
    pub g1_terms: Vec<(E::ScalarField, E::G1)>,
    /// Scalar coefficients and G2 points to combine before pairing.
    pub g2_terms: Vec<(E::ScalarField, E::G2)>,
}

/// A deferred SmartPC verifier equation.
///
/// Subprotocol verification records only coefficients. Evaluation performs the
/// G1 and G2 linear combinations first and executes one multi-pairing afterward.
pub struct SmartPcVerificationEquation<E: Pairing> {
    /// Scalar coefficients and irreducible GT terms.
    pub gt_terms: Vec<(E::ScalarField, PairingOutput<E>)>,
    /// Factored pairing rows.
    pub pairing_rows: Vec<SparsePairingRow<E>>,
}

impl<E: Pairing> SmartPcVerificationEquation<E> {
    /// Creates an empty deferred equation.
    pub fn new() -> Self {
        Self {
            gt_terms: Vec::new(),
            pairing_rows: Vec::new(),
        }
    }

    /// Adds another deferred equation multiplied by `coefficient`.
    pub fn append_scaled(&mut self, other: Self, coefficient: E::ScalarField) {
        self.gt_terms.extend(
            other
                .gt_terms
                .into_iter()
                .map(|(scalar, value)| (scalar * coefficient, value)),
        );
        self.pairing_rows
            .extend(other.pairing_rows.into_iter().map(|mut row| {
                for (scalar, _) in &mut row.g1_terms {
                    *scalar *= coefficient;
                }
                row
            }));
    }

    /// Evaluates the sparse source-group operations and the final pairing equation.
    pub fn evaluate(&self) -> bool {
        let mut g1_values = Vec::with_capacity(self.pairing_rows.len());
        let mut g2_values = Vec::with_capacity(self.pairing_rows.len());
        for row in &self.pairing_rows {
            let g1 = row
                .g1_terms
                .iter()
                .fold(E::G1::zero(), |acc, (scalar, value)| {
                    acc.add(value.mul(scalar))
                });
            let g2 = row
                .g2_terms
                .iter()
                .fold(E::G2::zero(), |acc, (scalar, value)| {
                    acc.add(value.mul(scalar))
                });
            g1_values.push(g1);
            g2_values.push(g2);
        }

        let mut result = E::multi_pairing(g1_values, g2_values);
        for (scalar, value) in &self.gt_terms {
            result = result.add(value.mul(scalar));
        }
        result.is_zero()
    }
}

impl<E: Pairing> Default for SmartPcVerificationEquation<E> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::smart_pc::SmartPC;
    use ark_bls12_381::{Bls12_381, Fr};
    use ark_serialize::CanonicalSerialize;
    use ark_std::test_rng;

    #[test]
    fn transcript_compressed_round_trip_preserves_a_valid_proof() {
        let rng = &mut test_rng();
        let pp = SmartPC::<Bls12_381>::setup(2, rng).unwrap();
        let matrix = (0..4)
            .map(|column| {
                (0..4)
                    .map(|row| Fr::from(column * 4 + row + 1))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let matrix_hiding = Fr::from(17_u64);
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
        proof.serialize_compressed(&mut encoded).unwrap();
        let decoded = Trans::<Bls12_381>::deserialize_compressed(encoded.as_slice()).unwrap();

        assert_eq!(decoded, proof);
        assert_eq!(decoded.serialized_size(Compress::Yes), encoded.len());
        assert!(SmartPC::<Bls12_381>::verify(
            &pp,
            matrix_commitment,
            value_commitment,
            &xl,
            &xr,
            &decoded,
        )
        .unwrap());
    }
}
