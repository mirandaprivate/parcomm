use mikros_qafold::qafold_short::{ShortQaCode, ShortQaDecideSetupMode};
use mikros_qafold::qafold_short_pc::{
    ShortQaBaseRows, ShortQaDecideProof, ShortQaFoldOpeningProof, ShortQaSetupRows,
};
use mikros_qafold::{
    FullPrime65521Ext8 as F, FullQaFoldPC, Prime65521 as P, ShortBaseField, ShortFieldElement,
};
use rand::{SeedableRng, rngs::StdRng};

fn pow(mut value: F, mut exponent: u128) -> F {
    let mut result = F::one();
    while exponent != 0 {
        if exponent & 1 != 0 {
            result *= value;
        }
        value *= value;
        exponent >>= 1;
    }
    result
}

fn trim(polynomial: &mut Vec<P>) {
    while polynomial.last() == Some(&P::zero()) {
        polynomial.pop();
    }
}

fn remainder(mut dividend: Vec<P>, divisor: &[P]) -> Vec<P> {
    let inverse = divisor.last().unwrap().try_inverse().unwrap();
    trim(&mut dividend);
    while dividend.len() >= divisor.len() {
        let offset = dividend.len() - divisor.len();
        let scale = *dividend.last().unwrap() * inverse;
        for (i, coefficient) in divisor.iter().enumerate() {
            dividend[offset + i] -= scale * *coefficient;
        }
        trim(&mut dividend);
    }
    dividend
}

#[test]
fn modulus_is_prime_and_octic_polynomial_is_irreducible() {
    for divisor in 2u64..=256 {
        assert_ne!(65_521 % divisor, 0);
    }
    let x = F::from_coefficients(core::array::from_fn(|i| {
        if i == 1 { P::one() } else { P::zero() }
    }));
    assert_eq!(pow(x, 8), F::from_u64(17));
    // Rabin's criterion for degree eight: X^(p^8)=X and
    // gcd(X^(p^(8/2))-X, X^8-17)=1. Two is the only prime divisor of eight.
    assert_eq!(pow(x, F::CARDINALITY), x);
    let p4_minus_x = pow(x, 65_521u128.pow(4)) - x;
    let mut modulus = vec![P::zero(); 9];
    modulus[0] = -P::from_u64(17);
    modulus[8] = P::one();
    let mut other = p4_minus_x.coefficients().to_vec();
    trim(&mut other);
    while !other.is_empty() {
        let next = remainder(modulus, &other);
        modulus = other;
        other = next;
    }
    assert_eq!(modulus.len(), 1);
}

#[test]
fn arithmetic_and_frobenius_use_the_polynomial_field() {
    let mut rng = StdRng::seed_from_u64(0x6552_1008);
    assert_eq!(F::from_u64(65_521), F::zero());
    for _ in 0..128 {
        let a = F::sample_uniform(&mut rng);
        let b = F::sample_uniform(&mut rng);
        let c = F::sample_uniform(&mut rng);
        assert_eq!(a * (b + c), a * b + a * c);
        assert_eq!(a * (b * c), (a * b) * c);
        assert_eq!(a - b, a + -b);
        assert_eq!(a * a.try_inverse().unwrap(), F::one());
        assert_eq!(a.frobenius(1), pow(a, 65_521));
        assert_eq!(a.frobenius(8), a);
        let scalar = P::sample_uniform(&mut rng);
        assert_eq!(
            a.mul_prime(scalar),
            a * F::from_u64(scalar.to_canonical_u64())
        );
    }
    assert_eq!(F::zero().try_inverse(), None);
}

/// Independent integer schoolbook convolution followed by polynomial reduction.
fn reference_product(a: F, b: F, addend: F) -> F {
    let mut coefficients = [0u128; 15];
    for i in 0..8 {
        for j in 0..8 {
            coefficients[i + j] +=
                u128::from(a.coefficients()[i].as_u16()) * u128::from(b.coefficients()[j].as_u16());
        }
    }
    F::from_coefficients(core::array::from_fn(|i| {
        let high = if i < 7 { coefficients[i + 8] } else { 0 };
        let integer = coefficients[i] + 17 * high + u128::from(addend.coefficients()[i].as_u16());
        P::new((integer % 65_521) as u32)
    }))
}

#[test]
fn lazy_product_and_fused_multiply_add_match_integer_convolution() {
    let mut rng = StdRng::seed_from_u64(0x6552_10ca_7715);
    let edge = [0, 1, 15, 32_760, 32_761, 65_519, 65_520];
    for &left in &edge {
        for &right in &edge {
            for lane in 0..=8 {
                let a = F::from_coefficients(core::array::from_fn(|i| {
                    P::new(if lane == 8 || i == lane { left } else { 65_520 })
                }));
                let b = F::from_coefficients([P::new(right); 8]);
                let addend = F::from_coefficients([P::new(65_520); 8]);
                assert_eq!(a * b, reference_product(a, b, F::zero()));
                assert_eq!(a.mul_add(b, addend), reference_product(a, b, addend));
            }
        }
    }
    for _ in 0..4096 {
        let a = F::sample_uniform(&mut rng);
        let b = F::sample_uniform(&mut rng);
        let addend = F::sample_uniform(&mut rng);
        assert_eq!(a * b, reference_product(a, b, F::zero()));
        assert_eq!(a.mul_add(b, addend), reference_product(a, b, addend));
    }
}

#[test]
fn direct_frobenius_table_matches_independent_exponentiation() {
    let mut rng = StdRng::seed_from_u64(0xf80b_6552_1008);
    for _ in 0..64 {
        let a = F::sample_uniform(&mut rng);
        let mut expected = a;
        for k in 0..16 {
            assert_eq!(a.frobenius(k), expected);
            expected = pow(expected, 65_521);
        }
    }
}

#[test]
fn butterfly_dispatch_matches_canonical_integer_arithmetic() {
    let mut rng = StdRng::seed_from_u64(0x6552_1b77);
    for length in [0, 1, 2, 3, 4, 5, 7, 8, 9, 16, 63, 257, 4097] {
        let mut left: Vec<_> = (0..length + 2)
            .map(|_| F::sample_uniform(&mut rng))
            .collect();
        let mut right: Vec<_> = (0..length + 2)
            .map(|_| F::sample_uniform(&mut rng))
            .collect();
        let before_left = left.clone();
        let before_right = right.clone();
        F::butterfly_slice(&mut left[1..length + 1], &mut right[1..length + 1]);
        for i in 1..length + 1 {
            for j in 0..8 {
                let a = u32::from(before_left[i].coefficients()[j].as_u16());
                let b = u32::from(before_right[i].coefficients()[j].as_u16());
                assert_eq!(
                    u32::from(left[i].coefficients()[j].as_u16()),
                    (a + b) % 65_521
                );
                assert_eq!(
                    u32::from(right[i].coefficients()[j].as_u16()),
                    (a + 65_521 - b) % 65_521
                );
            }
        }
        assert_eq!(left[0], before_left[0]);
        assert_eq!(right[0], before_right[0]);
        assert_eq!(left[length + 1], before_left[length + 1]);
        assert_eq!(right[length + 1], before_right[length + 1]);
    }
}

#[test]
fn canonical_codec_and_unbiased_full_field_sampling() {
    let mut rng = StdRng::seed_from_u64(0xc0de_cafe);
    let mut high_nonzero = [false; 8];
    for _ in 0..256 {
        let value = F::sample_uniform(&mut rng);
        let mut bytes = [0; 16];
        value.write_canonical_bytes(&mut bytes).unwrap();
        assert_eq!(F::from_canonical_bytes(&bytes), Some(value));
        assert_eq!(F::from_uniform_bytes(&bytes), Some(value));
        assert_eq!(
            F::from_canonical_u128_checked(value.to_canonical_u128()),
            Some(value)
        );
        for (observed, coefficient) in high_nonzero.iter_mut().zip(value.coefficients()) {
            *observed |= *coefficient != P::zero();
        }
    }
    assert!(high_nonzero.into_iter().all(|observed| observed));
    for lane in 0..8 {
        for invalid in 65_521u16..=u16::MAX {
            let mut bytes = [0; 16];
            bytes[2 * lane..2 * lane + 2].copy_from_slice(&invalid.to_le_bytes());
            assert_eq!(F::from_canonical_bytes(&bytes), None);
            assert_eq!(F::from_uniform_bytes(&bytes), None);
        }
    }
    assert_eq!(F::from_canonical_bytes(&[0; 15]), None);
    assert_eq!(F::from_canonical_bytes(&[0; 17]), None);
    // A valid coefficient representation can exceed the cardinality as an
    // integer; accepting it is required, and reducing it modulo q is wrong.
    let highest = F::from_coefficients([P::from_u64(65_520); 8]);
    assert!(highest.to_canonical_u128() > F::CARDINALITY);
    assert_eq!(
        F::from_canonical_u128_checked(highest.to_canonical_u128()),
        Some(highest)
    );
}

#[test]
fn qa_generator_and_distance_use_the_full_message_field() {
    let full = ShortQaCode::<F>::sample_systematic(64, 2, [7; 32]).unwrap();
    let prime = ShortQaCode::<P>::sample_systematic(64, 2, [7; 32]).unwrap();
    assert!(full.alphas()[1].iter().any(|value| {
        value.coefficients()[1..]
            .iter()
            .any(|coefficient| *coefficient != P::zero())
    }));
    assert_ne!(full.generator_digest(), prime.generator_digest());
    assert!(
        full.entropy_distance_bound().unwrap().relative_distance
            > prime.entropy_distance_bound().unwrap().relative_distance
    );
    assert_eq!(F::EXTENSION_DEGREE, 1);
    assert_eq!(
        F::DECIDE_SETUP_MODE,
        ShortQaDecideSetupMode::DirectInterleavedV1
    );
    assert!(FullQaFoldPC::new(256, 4, 2, 111).is_ok());
    assert!(FullQaFoldPC::new(256, 4, 2, 112).is_err());
}

fn coefficient_evaluation(mut witness: Vec<F>, point: &[F]) -> F {
    for challenge in point {
        let half = witness.len() / 2;
        for i in 0..half {
            witness[i] = witness[i] + witness[i + half] * *challenge;
        }
        witness.truncate(half);
    }
    witness[0]
}

#[test]
fn full_opening_recursive_decide_and_tamper_checks() {
    let pc = FullQaFoldPC::new(256, 4, 2, 80).unwrap();
    let mut rng = StdRng::seed_from_u64(0xf011_6552_1008);
    let (pp, setup) = pc.setup(&mut rng).unwrap();
    let witness: Vec<_> = (0..256).map(|_| F::sample_uniform(&mut rng)).collect();
    let point: Vec<_> = (0..8).map(|_| F::sample_uniform(&mut rng)).collect();
    let expected = coefficient_evaluation(witness.clone(), &point);
    let (commitment, cache) = pc.commit_owned(&pp, witness).unwrap();
    let (value, proof, prover_reduction) = pc
        .open_with_reduction(&pp, &commitment, &point, cache)
        .unwrap();
    assert_eq!(value, expected);
    let proof_bytes = proof.to_bytes().unwrap();
    assert_eq!(proof_bytes.len(), proof.encoded_len());
    let proof = ShortQaFoldOpeningProof::<F>::from_bytes(&proof_bytes).unwrap();
    let reduction = pc
        .verify_opening_reduction(&pp, &commitment, &point, value, &proof)
        .unwrap();
    assert_eq!(reduction, prover_reduction);
    assert!(reduction.ok && reduction.pending_h_claims.len() >= 2);
    let decide = pc.prove_decide(&pp, &setup, &prover_reduction).unwrap();
    let decide_bytes = decide.to_bytes().unwrap();
    assert_eq!(decide_bytes.len(), decide.encoded_len());
    let decide = ShortQaDecideProof::<F>::from_bytes(&decide_bytes).unwrap();
    assert!(pc.verify_decide(&pp, &setup, &reduction, &decide).unwrap());
    assert!(pc.verify(&pp, &commitment, &point, value, &proof).unwrap());

    assert!(
        !pc.verify_opening_reduction(&pp, &commitment, &point, value + F::one(), &proof)
            .unwrap()
            .ok
    );
    let mut altered = proof.clone();
    altered.responses[0] += F::one();
    assert!(
        !pc.verify_opening_reduction(&pp, &commitment, &point, value, &altered)
            .unwrap()
            .ok
    );
    let mut altered = proof.clone();
    match &mut altered.top_round.rows {
        ShortQaBaseRows::Partial(rows) => rows[0].row[0] += F::one(),
        ShortQaBaseRows::AllRows {
            encoded_entries, ..
        } => encoded_entries[0] += F::one(),
    }
    assert!(
        !pc.verify_opening_reduction(&pp, &commitment, &point, value, &altered)
            .unwrap()
            .ok
    );
    let mut altered = decide.clone();
    altered.responses[0] += F::one();
    assert!(
        !pc.verify_decide(&pp, &setup, &reduction, &altered)
            .unwrap_or(false)
    );
    let mut altered = decide.clone();
    match &mut altered.rounds[0].setup_rows {
        ShortQaSetupRows::Partial(rows) => rows[0].row[0] += F::one(),
        ShortQaSetupRows::AllRows {
            encoded_entries, ..
        } => encoded_entries[0] += F::one(),
    }
    assert!(
        !pc.verify_decide(&pp, &setup, &reduction, &altered)
            .unwrap_or(false)
    );
    let mut altered = reduction.clone();
    altered.pending_h_claims[0].value += F::one();
    assert!(
        !pc.verify_decide(&pp, &setup, &altered, &decide)
            .unwrap_or(false)
    );
    let mut malformed = proof_bytes.clone();
    malformed[16..18].copy_from_slice(&65_521u16.to_le_bytes());
    assert!(ShortQaFoldOpeningProof::<F>::from_bytes(&malformed).is_err());
    let mut trailing = decide_bytes;
    trailing.push(0);
    assert!(ShortQaDecideProof::<F>::from_bytes(&trailing).is_err());
}
