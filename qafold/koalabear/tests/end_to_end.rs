use parcomm_qafold_koalabear_8b::{
    field::*,
    layout::*,
    tbv::{self, Claim},
    transcript::Transcript,
    *,
};

fn fixture(shapes: &[Shape]) -> (Parameters, Vec<i16>, Vec<Claim>) {
    let parameters = Parameters::new(
        shapes,
        Config {
            chunk_log: 10,
            packed_log: 4,
            query_rf_bits: 100,
        },
    )
    .unwrap();
    let matrices = shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            (0..shape.rows * shape.cols)
                .map(|entry| (entry * 73 + index * 19) as u16 as i16)
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let values = parameters.layout.encode(&matrices).unwrap();
    let mut transcript = Transcript::new();
    let points = shapes
        .iter()
        .map(|shape| {
            (
                (0..shape.rows.next_power_of_two().ilog2())
                    .map(|_| transcript.challenge(b"row"))
                    .collect::<Vec<_>>(),
                (0..shape.cols.next_power_of_two().ilog2())
                    .map(|_| transcript.challenge(b"col"))
                    .collect::<Vec<_>>(),
            )
        })
        .collect::<Vec<_>>();
    let claims = tbv::evaluate_claims(&parameters.layout, &values, &points).unwrap();
    // Independent original-matrix oracle: never use BVA offsets or tensor helpers.
    for (((shape, matrix), (row, col)), claim) in
        shapes.iter().zip(&matrices).zip(&points).zip(&claims)
    {
        let mut expected = F::zero();
        for row_index in 0..shape.rows {
            for col_index in 0..shape.cols {
                let mut weight = F::one();
                for (bit, &coordinate) in row.iter().enumerate() {
                    if row_index & (1 << bit) != 0 {
                        weight *= coordinate;
                    }
                }
                for (bit, &coordinate) in col.iter().enumerate() {
                    if col_index & (1 << bit) != 0 {
                        weight *= coordinate;
                    }
                }
                expected += weight * lift(matrix[row_index * shape.cols + col_index]);
            }
        }
        assert_eq!(claim.value.field().unwrap(), expected);
    }
    (parameters, values, claims)
}

#[test]
fn heterogeneous_collection_roundtrip_and_adversarial_verification() {
    let (parameters, values, claims) = fixture(&[
        Shape { rows: 31, cols: 37 },
        Shape { rows: 65, cols: 17 },
        Shape { rows: 1, cols: 7 },
        Shape { rows: 9, cols: 1 },
        Shape { rows: 1, cols: 1 },
    ]);
    let committed = parameters.commit(values).unwrap();
    let statement = committed.statement.clone();
    let (proof, _) = parameters.prove(committed, &claims).unwrap();
    let wire = proof.to_bytes().unwrap();
    let proof = Proof::from_bytes(&wire).unwrap();
    // A fresh verifier reconstructs parameters from public shapes/config alone.
    let verifier = Parameters::new(&statement.shapes, statement.config.clone()).unwrap();
    verifier.verify(&statement, &claims, &proof).unwrap();
    assert!(statement.roots.len() > 1);
    assert_eq!(proof.openings.len(), 1);
    let profile = verifier.verify(&statement, &claims, &proof).unwrap();
    assert_eq!(profile.native_openings, 1);
    assert_eq!(profile.authenticated_roots, statement.roots.len());
    let mut swapped = statement.clone();
    swapped.roots.swap(0, 1);
    assert!(verifier.verify(&swapped, &claims, &proof).is_err());
    let reject = |proof: &Proof| assert!(verifier.verify(&statement, &claims, proof).is_err());
    let mut altered = proof.clone();
    altered.eo.positive[0] = (altered.eo.positive[0].field().unwrap() + F::one()).into();
    reject(&altered);
    let mut altered = proof.clone();
    altered.eo.negative[0] = (altered.eo.negative[0].field().unwrap() + F::one()).into();
    reject(&altered);
    let mut altered = proof.clone();
    altered.eo.rounds[0][0] = (altered.eo.rounds[0][0].field().unwrap() + F::one()).into();
    reject(&altered);
    let mut altered = proof.clone();
    altered.eo.query = (altered.eo.query.field().unwrap() + F::one()).into();
    reject(&altered);
    let mut altered = proof.clone();
    altered.eo.value = (altered.eo.value.field().unwrap() + F::one()).into();
    reject(&altered);
    let mut altered = proof.clone();
    altered.openings[0].value = (altered.openings[0].value.field().unwrap() + F::one()).into();
    reject(&altered);
    let mut altered = proof.clone();
    altered.openings[0].proof[0] ^= 1;
    reject(&altered);
    let mut altered = proof.clone();
    altered.openings.push(altered.openings[0].clone());
    reject(&altered);
    let mut altered = proof.clone();
    altered.openings.pop();
    reject(&altered);
    let mut altered = proof.clone();
    altered.eo.rounds.pop();
    reject(&altered);
    let mut altered = proof.clone();
    altered.eo.value = Scalar([255; 16]);
    reject(&altered);
    let mut altered_claims = claims.clone();
    altered_claims[0].value = (altered_claims[0].value.field().unwrap() + F::one()).into();
    assert!(
        verifier
            .verify(&statement, &altered_claims, &proof)
            .is_err()
    );
    let mut altered_claims = claims.clone();
    altered_claims[0].col[0] = (altered_claims[0].col[0].field().unwrap() + F::one()).into();
    assert!(
        verifier
            .verify(&statement, &altered_claims, &proof)
            .is_err()
    );
    let mut altered_statement = statement.clone();
    altered_statement.roots[0][0] ^= 1;
    assert!(
        verifier
            .verify(&altered_statement, &claims, &proof)
            .is_err()
    );
    let mut altered_statement = statement.clone();
    altered_statement.shapes[0].rows -= 1;
    assert!(
        verifier
            .verify(&altered_statement, &claims, &proof)
            .is_err()
    );
    let mut altered_statement = statement.clone();
    altered_statement.parameter_digest[0] ^= 1;
    assert!(
        verifier
            .verify(&altered_statement, &claims, &proof)
            .is_err()
    );
    assert!(Proof::from_bytes(&wire[..wire.len() - 1]).is_err());
    let mut trailing = wire;
    trailing.push(0);
    assert!(Proof::from_bytes(&trailing).is_err());
}

#[test]
fn packing_widths_and_zero_one_query_coordinates() {
    for packed_log in 1..=6 {
        let parameters = Parameters::new(
            &[Shape { rows: 5, cols: 7 }],
            Config {
                packed_log,
                ..Default::default()
            },
        )
        .unwrap();
        let values = parameters.layout.encode(&[vec![-128; 35]]).unwrap();
        let points = [(
            vec![F::zero(), F::one(), -F::one()],
            vec![F::one(), F::zero(), -F::one()],
        )];
        let claims = tbv::evaluate_claims(&parameters.layout, &values, &points).unwrap();
        let committed = parameters.commit(values).unwrap();
        let statement = committed.statement.clone();
        let (proof, _) = parameters.prove(committed, &claims).unwrap();
        parameters.verify(&statement, &claims, &proof).unwrap();
    }
}

#[test]
fn malformed_inputs_rejected() {
    assert!(Layout::new(&[]).is_err());
    assert!(Layout::new(&[Shape { rows: 0, cols: 1 }]).is_err());
    assert!(
        Layout::new(&[Shape {
            rows: usize::MAX,
            cols: 2
        }])
        .is_err()
    );
    let (parameters, values, mut claims) = fixture(&[Shape { rows: 3, cols: 5 }]);
    assert!(
        parameters
            .commit(values[..values.len() - 1].to_vec())
            .is_err()
    );
    let committed = parameters.commit(values).unwrap();
    claims[0].row.pop();
    assert!(parameters.prove(committed, &claims).is_err());
}

#[test]
fn compact_tbv_matches_dense_query_and_padding_support() {
    let (parameters, values, claims) =
        fixture(&[Shape { rows: 31, cols: 31 }, Shape { rows: 3, cols: 2 }]);
    let mut transcript = Transcript::new();
    let query = tbv::compile(&parameters.layout, &claims, &mut transcript).unwrap();
    let mut dense = vec![F::zero(); parameters.layout.logical];
    for term in &query.terms {
        for index in 0..term.len() {
            dense[term.offset + index] += term.weight(index);
        }
    }
    let claim = values
        .iter()
        .zip(&dense)
        .fold(F::zero(), |sum, (&v, &weight)| sum + lift(v) * weight);
    assert_eq!(claim, query.claim);
    let point = (0..parameters.layout.logical.ilog2())
        .map(|_| transcript.challenge(b"evaluation"))
        .collect::<Vec<_>>();
    let mut reduced = dense.clone();
    for &coordinate in &point {
        reduced = reduced
            .chunks_exact(2)
            .map(|pair| pair[0] + coordinate * pair[1])
            .collect();
    }
    assert_eq!(query.evaluation(&point), reduced[0]);
    let stride = 16;
    let (positive, negative) = query.cross_terms(&values, stride);
    for distance in 1..stride {
        let mut expected_positive = F::zero();
        let mut expected_negative = F::zero();
        for start in (0..parameters.layout.logical).step_by(stride) {
            for lane in 0..stride - distance {
                expected_positive += lift(values.get(start + lane).copied().unwrap_or(0))
                    * dense[start + lane + distance];
                expected_negative +=
                    lift(values.get(start + lane + distance).copied().unwrap_or(0))
                        * dense[start + lane];
            }
        }
        assert_eq!(positive[distance - 1], expected_positive);
        assert_eq!(negative[distance - 1], expected_negative);
    }
}

#[test]
fn real_8b_layout_matches_the_existing_converter_exactly() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("fixtures/llama8b-layout.json")).unwrap();
    let shapes: Vec<[usize; 2]> = serde_json::from_value(fixture["shapes"].clone()).unwrap();
    let layout = Layout::new(
        &shapes
            .into_iter()
            .map(|[rows, cols]| Shape { rows, cols })
            .collect::<Vec<_>>(),
    )
    .unwrap();
    assert_eq!(layout.shapes.len(), 291);
    assert_eq!(layout.buckets.len(), 489);
    assert_eq!(layout.active, 8053329920);
    assert_eq!(layout.logical, 8589934592);
    for (bucket, original) in layout
        .buckets
        .iter()
        .zip(fixture["buckets"].as_array().unwrap())
    {
        for (actual, key) in [
            (bucket.matrix, "tensor_index"),
            (bucket.offset, "dense_offset"),
            (bucket.row_offset, "row_offset"),
            (bucket.col_offset, "col_offset"),
            (bucket.rows, "row_len"),
            (bucket.cols, "col_len"),
            (bucket.physical_rows, "physical_row_len"),
            (bucket.physical_cols, "physical_col_len"),
        ] {
            assert_eq!(actual as u64, original[key].as_u64().unwrap(), "{key}");
        }
    }
}

#[test]
fn large_public_dimensions_preserve_integer_values() {
    let (parameters, values, claims) = fixture(&[Shape {
        rows: 65523,
        cols: 1,
    }]);
    assert_eq!(parameters.layout.shapes[0].rows, 65523);
    let committed = parameters.commit(values).unwrap();
    let statement = committed.statement.clone();
    let (proof, _) = parameters.prove(committed, &claims).unwrap();
    parameters.verify(&statement, &claims, &proof).unwrap();
}
