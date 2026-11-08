use mikros_qafold::qafold_short::{ShortQaCode, short_wht_in_place};
use mikros_qafold::{
    FullPrime65521Ext8 as F, Prime65521 as P, ShortBaseField, ShortExtensionField,
    ShortFieldElement,
};
use rand::{SeedableRng, rngs::StdRng};

// Intentionally independent of the optimized butterfly hook and WHT helpers.
fn reference_wht<E: ShortFieldElement>(values: &mut [E]) {
    let mut half = 1;
    while half < values.len() {
        for block in values.chunks_exact_mut(2 * half) {
            for i in 0..half {
                let (a, b) = (block[i], block[i + half]);
                block[i] = a + b;
                block[i + half] = a - b;
            }
        }
        half *= 2;
    }
}

fn reference_encode<B: ShortBaseField>(
    code: &ShortQaCode<B>,
    message: &[B::Extension],
) -> Vec<B::Extension> {
    let mut spectrum = message.to_vec();
    reference_wht(&mut spectrum);
    let mut output = Vec::new();
    for alpha in code.alphas() {
        let mut values = spectrum
            .iter()
            .zip(alpha)
            .map(|(&value, &alpha)| value.mul_base(alpha))
            .collect::<Vec<_>>();
        reference_wht(&mut values);
        output.extend(values);
    }
    output
}

fn reference_left<B: ShortBaseField>(
    code: &ShortQaCode<B>,
    left: &[B::Extension],
) -> Vec<B::Extension> {
    let mut output = vec![B::Extension::zero(); code.message_len()];
    for (block, alpha) in left.chunks_exact(code.message_len()).zip(code.alphas()) {
        let mut values = block.to_vec();
        reference_wht(&mut values);
        for (value, &alpha) in values.iter_mut().zip(alpha) {
            *value = value.mul_base(alpha);
        }
        reference_wht(&mut values);
        for (out, value) in output.iter_mut().zip(values) {
            *out += value;
        }
    }
    output
}

fn sample_extension<B: ShortBaseField>(rng: &mut StdRng) -> B::Extension {
    B::Extension::from_basis_coefficients_fn(|_| B::sample_uniform(rng))
}

fn exercise<B: ShortBaseField>() {
    let mut rng = StdRng::seed_from_u64(0x5141_454e_434f_4445);
    for log in [0, 1, 5, 12, 14] {
        let n = 1 << log;
        for rate in [2, 4] {
            for systematic in [true, false] {
                let code = if systematic {
                    ShortQaCode::<B>::sample_systematic(n, rate, [23; 32]).unwrap()
                } else {
                    let mut coefficients = (0..rate)
                        .map(|_| {
                            (0..n)
                                .map(|_| B::sample_uniform(&mut rng))
                                .collect::<Vec<_>>()
                        })
                        .collect::<Vec<_>>();
                    coefficients[0][0] = B::from_u64(2);
                    ShortQaCode::<B>::from_group_algebra(coefficients).unwrap()
                };
                let message = (0..n)
                    .map(|_| B::sample_uniform(&mut rng))
                    .collect::<Vec<_>>();
                let embedded = message
                    .iter()
                    .copied()
                    .map(B::Extension::from_base)
                    .collect::<Vec<_>>();
                let expected = reference_encode::<B>(&code, &embedded);
                let mut output = vec![B::one(); n * rate];
                code.encode_base_into(&message, &mut output).unwrap();
                assert_eq!(
                    output
                        .iter()
                        .copied()
                        .map(B::Extension::from_base)
                        .collect::<Vec<_>>(),
                    expected
                );
                output.fill(B::one());
                output[..n].copy_from_slice(&message);
                code.encode_base_in_place(&mut output).unwrap();
                assert_eq!(
                    output
                        .iter()
                        .copied()
                        .map(B::Extension::from_base)
                        .collect::<Vec<_>>(),
                    expected
                );
                if systematic {
                    assert_eq!(&output[..n], message.as_slice());
                }
                let extension = (0..n)
                    .map(|_| sample_extension::<B>(&mut rng))
                    .collect::<Vec<_>>();
                let expected = reference_encode::<B>(&code, &extension);
                assert_eq!(code.encode_extension(&extension).unwrap(), expected);
                assert_eq!(
                    code.generator_right_projection_extension(&extension)
                        .unwrap(),
                    expected
                );
                let left = (0..n * rate)
                    .map(|_| sample_extension::<B>(&mut rng))
                    .collect::<Vec<_>>();
                assert_eq!(
                    code.generator_left_projection_extension(&left).unwrap(),
                    reference_left::<B>(&code, &left)
                );
                let mut actual_wht = extension.clone();
                let mut expected_wht = extension;
                short_wht_in_place(&mut actual_wht).unwrap();
                reference_wht(&mut expected_wht);
                assert_eq!(actual_wht, expected_wht);
                let zeros = vec![B::zero(); n];
                code.encode_base_into(&zeros, &mut output).unwrap();
                assert!(output.iter().all(|&value| value == B::zero()));
                assert!(
                    code.encode_base_into(&message[..n - 1], &mut output)
                        .is_err()
                );
                assert!(
                    code.encode_base_in_place(&mut output[..n * rate - 1])
                        .is_err()
                );
            }
        }
    }
    assert!(short_wht_in_place::<B>(&mut []).is_err());
    assert!(short_wht_in_place(&mut [B::one(); 3]).is_err());
}

#[test]
fn qa_encoding_and_projection_match_independent_scalar_reference() {
    #[cfg(feature = "parallel")]
    for threads in [1, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                exercise::<P>();
                exercise::<F>();
            });
    }
    #[cfg(not(feature = "parallel"))]
    {
        exercise::<P>();
        exercise::<F>();
    }
}

#[test]
fn fused_f65521_encoding_crosses_high_stripe_boundary() {
    let n = 1 << 18;
    let code = ShortQaCode::<F>::sample_systematic(n, 2, [81; 32]).unwrap();
    let mut rng = StdRng::seed_from_u64(418);
    let message = (0..n).map(|_| F::sample_uniform(&mut rng)).collect::<Vec<_>>();
    let expected = reference_encode::<F>(&code, &message);
    let mut actual = vec![F::one(); n * 2];
    code.encode_base_into(&message, &mut actual).unwrap();
    assert_eq!(actual, expected);
}
