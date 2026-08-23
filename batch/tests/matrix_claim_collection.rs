use ark_bn254::Fr;
use ark_ff::{One, Zero};
use batch::jolt_dory::setup_dory_parameters;
use batch::matrix_collection::{
    commit_matrix_shapes, commit_registered_dense_bva, matrix_mle_evaluate,
    prove_registered_matrix_claim_collection, verify_registered_bva_opening,
    verify_registered_matrix_claim_collection, verify_registered_shape_point_consistency,
    verify_registered_tbv_inner_product, DenseI16Matrix, DenseI8Matrix, DenseMatrixBva,
    MatrixMleClaim, RegisteredMatrixCollectionProof,
};
use jolt_field::Fr as JoltFr;

fn point(values: &[u64]) -> Vec<Fr> {
    values.iter().copied().map(Fr::from).collect()
}

fn make_claim(
    matrix: &DenseI8Matrix,
    left_point: Vec<Fr>,
    right_point: Vec<Fr>,
) -> MatrixMleClaim<Fr> {
    let value = matrix_mle_evaluate(matrix, &left_point, &right_point).unwrap();
    MatrixMleClaim::new(matrix.shape(), left_point, right_point, value).unwrap()
}

#[test]
fn witness_matrix_is_column_major_but_each_bva_bucket_is_row_major() {
    // The native matrix columns are [1, 2], [3, 4], and [5, 6]. Section 6
    // stores the leading 2x2 bucket row-major, followed by the final 2x1
    // bucket row-major.
    let matrix = DenseI8Matrix::new((2, 3), vec![1, 2, 3, 4, 5, 6]).unwrap();
    let left = point(&[2]);
    let right = point(&[3, 5]);
    assert_eq!(
        matrix_mle_evaluate(&matrix, &left, &right).unwrap(),
        Fr::from(123_u64)
    );

    let dense_bva = DenseMatrixBva::from_matrices(&[matrix]).unwrap();
    assert_eq!(dense_bva.values(), &[1, 3, 2, 4, 5, 6, 0, 0]);
}

#[test]
fn dense_short_int_bva_uses_native_i16_storage_end_to_end() {
    let matrix = DenseI16Matrix::new(
        (2, 3),
        vec![i16::MIN, 30_000, -1_024, 17_000, 12_345, -23_456],
    )
    .unwrap();
    let left = point(&[2]);
    let right = point(&[3, 5]);
    let claim = MatrixMleClaim::new(
        matrix.shape(),
        left.clone(),
        right.clone(),
        matrix_mle_evaluate(&matrix, &left, &right).unwrap(),
    )
    .unwrap();
    let built = DenseMatrixBva::<i16>::from_matrices(&[matrix]).unwrap();
    let descriptor = built.descriptor().clone();
    let mut source = DenseMatrixBva::<i16>::from_prepacked_i16(
        descriptor.matrix_shapes(),
        built.values().to_vec(),
    )
    .unwrap();
    assert_eq!(
        std::mem::size_of_val(source.values()),
        2 * source.values().len()
    );
    assert!(source
        .values()
        .iter()
        .any(|value| value.unsigned_abs() > 127));

    let (prover, verifier) = setup_dory_parameters(descriptor.required_registration_dory_log());
    let committed_shapes = commit_matrix_shapes(&prover, &descriptor).unwrap();
    let registration = commit_registered_dense_bva(&prover, &source, 16, committed_shapes).unwrap();
    let commitments = registration.commitments().clone();
    let (proof, profile) = prove_registered_matrix_claim_collection(
        &prover,
        &mut source,
        &[claim.clone()],
        &registration,
    )
    .unwrap();

    assert_eq!(proof.packed.packed_eo.packing_stride, 8);
    assert_eq!(profile.commitment.materialized_field_entries, 0);
    assert!(verify_registered_matrix_claim_collection(
        &verifier,
        &descriptor,
        &commitments,
        &[claim],
        16,
        &proof,
    )
    .unwrap()
    .evaluate());
}

#[test]
fn public_matrix_claim_collection_binds_dense_bva_opening_and_rejects_tampering() {
    // Asymmetric matrices and distinct left/right points catch transposition and
    // point-ordering errors in the Section 6 row-major bucket compilation.
    let matrices = vec![
        DenseI8Matrix::new(
            (3, 5),
            vec![1, 2, -3, 4, 5, -2, 7, 1, 0, -4, 3, 6, 2, -1, 5],
        )
        .unwrap(),
        DenseI8Matrix::new((5, 2), vec![-1, 4, 2, 3, 5, -2, 7, 1, -3, 6]).unwrap(),
        DenseI8Matrix::new((2, 3), vec![6, -2, 1, 4, -5, 3]).unwrap(),
    ];
    let claims = vec![
        make_claim(&matrices[0], point(&[2, 7]), point(&[3, 5, 11])),
        make_claim(&matrices[1], point(&[13, 17, 19]), point(&[23])),
        make_claim(&matrices[2], point(&[29]), point(&[31, 37])),
    ];

    let mut dense_bva = DenseMatrixBva::from_matrices(&matrices).unwrap();
    let descriptor = dense_bva.descriptor().clone();
    assert_eq!(dense_bva.values().len(), 32);
    assert_eq!(descriptor.physical_len(), 31);
    assert_eq!(descriptor.logical_len(), 32);
    assert_eq!(descriptor.materialized_padding_entries(), 1);
    assert!(descriptor.bucket_ranges_are_a_partition());
    assert_eq!(descriptor.matrix_count(), claims.len());

    let (prover_parameters, verifier_parameters) =
        setup_dory_parameters(descriptor.required_registration_dory_log());

    // Registration is intentionally ordered: shape metadata is committed
    // before the contiguous dense BVA.
    let committed_shapes = commit_matrix_shapes(&prover_parameters, &descriptor).unwrap();
    assert_eq!(
        committed_shapes.profile().alpha_field_entries,
        committed_shapes.profile().gamma_field_entries
    );
    let shape_commitments = committed_shapes.commitments();
    let registration =
        commit_registered_dense_bva(&prover_parameters, &dense_bva, 8, committed_shapes).unwrap();
    assert_eq!(registration.commitments().shape, shape_commitments);
    let commitments = registration.commitments().clone();
    let (proof, profile) = prove_registered_matrix_claim_collection(
        &prover_parameters,
        &mut dense_bva,
        &claims,
        &registration,
    )
    .unwrap();
    assert_eq!(proof.packed.matrix_commitment, commitments.bva);
    assert_eq!(
        profile.commitment.physical_entries,
        descriptor.logical_len()
    );
    assert_eq!(profile.commitment.materialized_padding_entries, 1);
    assert_eq!(profile.packed_eo_padding.materialized_padding_entries, 1);
    assert_eq!(profile.opening.materialized_padding_entries, 1);

    let bytes = proof.to_bytes().unwrap();
    let decoded = RegisteredMatrixCollectionProof::from_bytes_exact(&bytes).unwrap();
    assert_eq!(decoded, proof);
    let proof = decoded;

    // The verifier performs the requested terminal sequence explicitly:
    // open BVA, verify the TBV inner product, then check shape/point consistency.
    let opening = verify_registered_bva_opening(
        &verifier_parameters,
        &descriptor,
        &commitments,
        &claims,
        8,
        &proof,
    )
    .unwrap();
    assert!(opening.eo.accepted);
    assert!(verify_registered_tbv_inner_product(&opening, &proof));
    assert!(verify_registered_shape_point_consistency(
        &verifier_parameters,
        &descriptor,
        &commitments,
        &claims,
        8,
        &proof,
    )
    .unwrap());
    let terminal = verify_registered_matrix_claim_collection(
        &verifier_parameters,
        &descriptor,
        &commitments,
        &claims,
        8,
        &proof,
    )
    .unwrap();
    assert!(terminal.bva_opening_accepted);
    assert!(terminal.tbv_inner_product_accepted);
    assert!(terminal.shape_point_consistency_accepted);
    assert!(terminal.evaluate());

    // This optional witness-side oracle is independent of the public
    // shape/point consistency check.
    let compiled = descriptor
        .compile_for_registered_proof(&commitments, &claims, 8)
        .unwrap();
    let dense_claim = compiled
        .bva_mle_claim_from_scalar_verification(&proof.packed, &opening.eo)
        .unwrap();
    assert_eq!(
        dense_claim.point.len(),
        descriptor.logical_len().ilog2() as usize
    );
    assert_eq!(dense_claim.value, proof.packed.packed_eo.a_reduced);
    assert!(dense_bva.check_dense_bva_claim(&dense_claim).unwrap());

    let mut bad_dense_value = dense_claim.clone();
    bad_dense_value.value += Fr::one();
    assert!(!dense_bva.check_dense_bva_claim(&bad_dense_value).unwrap());

    let mut bad_dense_point = dense_claim.clone();
    bad_dense_point.point[0] += Fr::one();
    assert!(!dense_bva.check_dense_bva_claim(&bad_dense_point).unwrap());

    let mut bad_value = claims.clone();
    bad_value[1].value += Fr::one();
    assert!(verify_registered_matrix_claim_collection(
        &verifier_parameters,
        &descriptor,
        &commitments,
        &bad_value,
        8,
        &proof,
    )
    .is_err());

    let mut bad_gamma = proof.clone();
    bad_gamma.shape_consistency.matrices.evaluation_blind += JoltFr::from(1_u64);
    assert!(verify_registered_shape_point_consistency(
        &verifier_parameters,
        &descriptor,
        &commitments,
        &claims,
        8,
        &bad_gamma,
    )
    .is_err());

    let mut swapped_shape_commitments = commitments.clone();
    swapped_shape_commitments.shape.alpha = commitments.shape.gamma.clone();
    assert!(verify_registered_shape_point_consistency(
        &verifier_parameters,
        &descriptor,
        &swapped_shape_commitments,
        &claims,
        8,
        &proof,
    )
    .is_err());

    let mut unrelated_packed_commitment = proof.clone();
    unrelated_packed_commitment.packed.matrix_commitment = Default::default();
    assert!(verify_registered_shape_point_consistency(
        &verifier_parameters,
        &descriptor,
        &commitments,
        &claims,
        8,
        &unrelated_packed_commitment,
    )
    .is_err());

    let different_descriptor = DenseMatrixBva::from_fn(&[(3, 5), (5, 2), (3, 2)], |_, _, _| 1)
        .unwrap()
        .descriptor()
        .clone();
    assert!(verify_registered_shape_point_consistency(
        &verifier_parameters,
        &different_descriptor,
        &commitments,
        &claims,
        8,
        &proof,
    )
    .is_err());

    let mut bad_point = claims.clone();
    bad_point[0].right_point[1] += Fr::one();
    assert!(verify_registered_matrix_claim_collection(
        &verifier_parameters,
        &descriptor,
        &commitments,
        &bad_point,
        8,
        &proof,
    )
    .is_err());

    assert!(verify_registered_matrix_claim_collection(
        &verifier_parameters,
        &descriptor,
        &commitments,
        &claims,
        7,
        &proof,
    )
    .is_err());

    let mut bad_commitment = commitments.clone();
    bad_commitment.bva = Default::default();
    assert!(verify_registered_matrix_claim_collection(
        &verifier_parameters,
        &descriptor,
        &bad_commitment,
        &claims,
        8,
        &proof,
    )
    .is_err());

    let mut bad_opening = proof.clone();
    bad_opening.packed.opening.evaluation_blind += JoltFr::from(1_u64);
    assert!(verify_registered_matrix_claim_collection(
        &verifier_parameters,
        &descriptor,
        &commitments,
        &claims,
        8,
        &bad_opening,
    )
    .is_err());

    // A shape-proof tamper is rejected only in the final, witness-free stage;
    // the BVA opening and TBV inner product remain valid.
    let mut bad_shape = proof.clone();
    bad_shape.shape_consistency.matrices.evaluation_blind += JoltFr::from(1_u64);
    let opening = verify_registered_bva_opening(
        &verifier_parameters,
        &descriptor,
        &commitments,
        &claims,
        8,
        &bad_shape,
    )
    .unwrap();
    assert!(verify_registered_tbv_inner_product(&opening, &bad_shape));
    assert!(verify_registered_shape_point_consistency(
        &verifier_parameters,
        &descriptor,
        &commitments,
        &claims,
        8,
        &bad_shape,
    )
    .is_err());

    let mut wrong_witness = DenseMatrixBva::from_matrices(&matrices).unwrap();
    wrong_witness.values_mut()[0] ^= 1;
    assert!(commit_registered_dense_bva(
        &prover_parameters,
        &wrong_witness,
        8,
        commit_matrix_shapes(&prover_parameters, &descriptor).unwrap(),
    )
    .and_then(|registration| {
        prove_registered_matrix_claim_collection(
            &prover_parameters,
            &mut wrong_witness,
            &claims,
            &registration,
        )
        .map(|_| ())
    })
    .is_err());

    let mut bad_magic = bytes.clone();
    bad_magic[0] ^= 1;
    assert!(RegisteredMatrixCollectionProof::from_bytes_exact(&bad_magic).is_err());
    assert!(RegisteredMatrixCollectionProof::from_bytes_exact(&bytes[..bytes.len() - 1]).is_err());
    let mut oversized_component = bytes.clone();
    oversized_component[8..16].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(RegisteredMatrixCollectionProof::from_bytes_exact(&oversized_component).is_err());
    let mut trailing = bytes;
    trailing.push(0);
    assert!(RegisteredMatrixCollectionProof::from_bytes_exact(&trailing).is_err());
}

#[test]
fn dense_bva_oracle_uses_one_contiguous_padded_i8_vector() {
    let shapes = [(3, 5), (5, 2), (2, 3)];
    let dense_bva = DenseMatrixBva::from_fn(&shapes, |matrix, row, col| {
        let magnitude = ((matrix * 17 + row * 5 + col + 1) % 31 + 1) as i8;
        if (matrix + row + col).is_multiple_of(2) {
            magnitude
        } else {
            -magnitude
        }
    })
    .unwrap();
    assert_eq!(dense_bva.values().len(), 32);
    assert_eq!(dense_bva.values().last(), Some(&0));
    assert_eq!(dense_bva.descriptor().logical_len(), 32);
    assert_eq!(dense_bva.descriptor().materialized_padding_entries(), 1);

    let point = vec![
        -Fr::one(),
        Fr::from(3_u64),
        Fr::from(5_u64),
        Fr::from(7_u64),
        Fr::from(11_u64),
    ];
    let mut weights = vec![Fr::one()];
    for coordinate in &point {
        let current = weights.len();
        for index in 0..current {
            weights.push(weights[index] * coordinate);
        }
    }
    let expected = dense_bva
        .values()
        .iter()
        .zip(weights)
        .map(|(&value, weight)| {
            let field_value = if value >= 0 {
                Fr::from(value as u64)
            } else {
                -Fr::from((-(value as i16)) as u64)
            };
            field_value * weight
        })
        .sum::<Fr>();

    assert_eq!(dense_bva.evaluate_dense_at(&point).unwrap(), expected);
}

#[test]
fn constant_matrix_claim_has_expected_zero_and_unit_cases() {
    let zero = DenseI8Matrix::new((1, 1), vec![0]).unwrap();
    assert_eq!(
        matrix_mle_evaluate::<Fr>(&zero, &[], &[]).unwrap(),
        Fr::zero()
    );
    let unit = DenseI8Matrix::new((1, 1), vec![1]).unwrap();
    assert_eq!(
        matrix_mle_evaluate::<Fr>(&unit, &[], &[]).unwrap(),
        Fr::one()
    );
}

#[test]
fn prepacked_bva_evaluates_without_reconstructing_the_matrix() {
    let shapes = [(3, 5), (2, 3)];
    let generated = DenseMatrixBva::from_fn(&shapes, |matrix, row, col| {
        (1 + matrix * 17 + row * 5 + col) as i8
    })
    .unwrap();
    let prepacked =
        DenseMatrixBva::from_prepacked_i8(&shapes, generated.values().to_vec()).unwrap();

    for matrix in 0..shapes.len() {
        let shape = shapes[matrix];
        let left = (0..shape.0.next_power_of_two().ilog2())
            .map(|index| Fr::from(3 + index as u64))
            .collect::<Vec<_>>();
        let right = (0..shape.1.next_power_of_two().ilog2())
            .map(|index| Fr::from(11 + index as u64))
            .collect::<Vec<_>>();
        let dense = DenseI8Matrix::new(
            shape,
            (0..shape.1)
                .flat_map(|col| {
                    (0..shape.0).map(move |row| (1 + matrix * 17 + row * 5 + col) as i8)
                })
                .collect(),
        )
        .unwrap();
        assert_eq!(
            prepacked
                .matrix_mle_evaluate::<Fr>(matrix, &left, &right)
                .unwrap(),
            matrix_mle_evaluate::<Fr>(&dense, &left, &right).unwrap()
        );
    }
}

#[test]
fn prepacked_bva_rejects_nonzero_materialized_padding() {
    let shapes = [(3, 3)];
    let generated = DenseMatrixBva::from_fn(&shapes, |_, row, col| (row + col + 1) as i8).unwrap();
    let mut values = generated.values().to_vec();
    let padding = generated
        .descriptor()
        .buckets()
        .iter()
        .find_map(|bucket| {
            (bucket.physical_row_len() < bucket.row_len())
                .then(|| bucket.dense_offset() + bucket.physical_row_len() * bucket.col_len())
        })
        .unwrap_or(generated.descriptor().layout().active_len());
    values[padding] = 1;
    assert!(DenseMatrixBva::from_prepacked_i8(&shapes, values).is_err());

    let generated_i16 =
        DenseMatrixBva::<i16>::from_i16_fn(&shapes, |_, row, col| (row + col + 1) as i16).unwrap();
    let mut values_i16 = generated_i16.values().to_vec();
    values_i16[padding] = 1;
    assert!(DenseMatrixBva::from_prepacked_i16(&shapes, values_i16).is_err());
}

#[test]
fn column_major_reverse_tensor_order_matches_section_six() {
    let matrix = DenseI8Matrix::new((2, 2), vec![1, 2, 3, 4]).unwrap();
    let left = Fr::from(5_u64);
    let right = Fr::from(7_u64);
    let expected = Fr::one()
        + Fr::from(2_u64) * left
        + Fr::from(3_u64) * right
        + Fr::from(4_u64) * left * right;
    assert_eq!(
        matrix_mle_evaluate(&matrix, &[left], &[right]).unwrap(),
        expected
    );

    let column = DenseI8Matrix::new((4, 1), vec![1, 2, 3, 4]).unwrap();
    let low = Fr::from(11_u64);
    let high = Fr::from(13_u64);
    let expected =
        Fr::one() + Fr::from(2_u64) * low + Fr::from(3_u64) * high + Fr::from(4_u64) * low * high;
    assert_eq!(
        matrix_mle_evaluate(&column, &[low, high], &[]).unwrap(),
        expected
    );
}
