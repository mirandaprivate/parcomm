use ark_bn254::Fr;
use ark_ff::One;
use batch::{
    jolt_dory::setup_dory_parameters,
    matrix_collection::{
        commit_matrix_shapes, commit_registered_dense_bva, constant_matrix_mle_evaluate,
        prove_registered_matrix_claim_collection, verify_registered_bva_opening,
        verify_registered_matrix_claim_collection, verify_registered_shape_point_consistency,
        verify_registered_tbv_inner_product, DenseBvaDescriptor, DenseMatrixBva, MatrixMleClaim,
        RegisteredMatrixCollectionProof,
    },
};
use sha2::{Digest, Sha256};

fn family_shapes(max_log: usize) -> Vec<(usize, usize)> {
    (2..=max_log)
        .rev()
        .map(|log| {
            let side = (1_usize << log) - 1;
            (side, side)
        })
        .collect()
}

fn family_claims(shapes: &[(usize, usize)], constants: &[i8]) -> Vec<MatrixMleClaim<Fr>> {
    shapes
        .iter()
        .enumerate()
        .map(|(matrix, &shape)| {
            let point_len = shape.0.next_power_of_two().ilog2() as usize;
            let left_point = (0..point_len)
                .map(|coordinate| Fr::from((matrix * 37 + coordinate * 11 + 2) as u64))
                .collect::<Vec<_>>();
            let right_point = (0..point_len)
                .map(|coordinate| Fr::from((matrix * 41 + coordinate * 13 + 5) as u64))
                .collect::<Vec<_>>();
            let value =
                constant_matrix_mle_evaluate(shape, constants[matrix], &left_point, &right_point)
                    .unwrap();
            MatrixMleClaim::new(shape, left_point, right_point, value).unwrap()
        })
        .collect()
}

fn run_family(max_log: usize, run_witness_oracle: bool) {
    let shapes = family_shapes(max_log);
    let constants = (0..shapes.len())
        .map(|matrix| (matrix % 15 + 1) as i8)
        .collect::<Vec<_>>();
    let claims = family_claims(&shapes, &constants);
    let mut source = DenseMatrixBva::from_fn(&shapes, |matrix, _, _| constants[matrix]).unwrap();
    let descriptor = source.descriptor().clone();

    let expected_physical = shapes
        .iter()
        .map(|&(rows, cols)| rows * cols)
        .sum::<usize>();
    let expected_buckets = (2..=max_log)
        .map(|log| log.min(DenseBvaDescriptor::SPLITTING_BUDGET).pow(2))
        .sum::<usize>();
    assert_eq!(descriptor.physical_len(), expected_physical);
    assert_eq!(descriptor.bucket_count(), expected_buckets);
    assert_eq!(source.values().len(), descriptor.logical_len());
    assert_eq!(
        descriptor.materialized_padding_entries(),
        descriptor.logical_len() - expected_physical
    );
    assert!(descriptor.bucket_ranges_are_a_partition());

    let (prover_parameters, verifier_parameters) =
        setup_dory_parameters(descriptor.required_registration_dory_log());
    let committed_shapes = commit_matrix_shapes(&prover_parameters, &descriptor).unwrap();
    let registration =
        commit_registered_dense_bva(&prover_parameters, &source, 8, committed_shapes).unwrap();
    let commitments = registration.commitments().clone();
    let (proof, profile) = prove_registered_matrix_claim_collection(
        &prover_parameters,
        &mut source,
        &claims,
        &registration,
    )
    .unwrap();
    assert_eq!(
        proof.packed.packed_eo.packing_stride,
        32.min(descriptor.logical_len())
    );
    assert_eq!(
        profile.commitment.physical_entries,
        descriptor.logical_len()
    );
    assert_eq!(
        profile.commitment.materialized_padding_entries,
        descriptor.materialized_padding_entries()
    );
    assert_eq!(
        profile.packed_eo_padding.materialized_padding_entries,
        descriptor.materialized_padding_entries() as u64
    );
    assert_eq!(
        profile.opening.materialized_padding_entries,
        descriptor.materialized_padding_entries()
    );

    let proof_bytes = proof.to_bytes().unwrap();
    let proof_sha256 = Sha256::digest(&proof_bytes)
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect::<String>();
    let decoded = RegisteredMatrixCollectionProof::from_bytes_exact(&proof_bytes).unwrap();
    assert_eq!(decoded, proof);
    let proof = decoded;
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
    assert!(verify_registered_matrix_claim_collection(
        &verifier_parameters,
        &descriptor,
        &commitments,
        &claims,
        8,
        &proof,
    )
    .unwrap()
    .evaluate());

    let compiled = descriptor
        .compile_for_registered_proof(&commitments, &claims, 8)
        .unwrap();
    let bva_claim = compiled
        .bva_mle_claim_from_scalar_verification(&proof.packed, &opening.eo)
        .unwrap();
    assert_eq!(
        bva_claim.point.len(),
        descriptor.logical_len().ilog2() as usize
    );
    assert_eq!(bva_claim.value, proof.packed.packed_eo.a_reduced);
    if run_witness_oracle {
        assert!(source.check_dense_bva_claim(&bva_claim).unwrap());
    }

    println!(
        "matrix_collection_e2e max_log={max_log} matrices={} physical_entries={} buckets={} logical_entries={} final_implicit_zeros={} proof_bytes={} proof_sha256={} accepted=true",
        shapes.len(),
        descriptor.physical_len(),
        descriptor.bucket_count(),
        descriptor.logical_len(),
        descriptor.conceptual_final_zeros(),
        proof_bytes.len(),
        proof_sha256,
    );

    if max_log <= 7 {
        let mut bad_value = claims.clone();
        bad_value[0].value += Fr::one();
        assert!(verify_registered_matrix_claim_collection(
            &verifier_parameters,
            &descriptor,
            &commitments,
            &bad_value,
            8,
            &proof,
        )
        .is_err());
        let mut bad_point = claims.clone();
        bad_point[0].left_point[0] += Fr::one();
        assert!(verify_registered_matrix_claim_collection(
            &verifier_parameters,
            &descriptor,
            &commitments,
            &bad_point,
            8,
            &proof,
        )
        .is_err());
    }
}

#[test]
fn heterogeneous_matrix_collection_small_end_to_end() {
    run_family(7, true);
}

#[test]
fn capped_split_stores_a_padded_1023_square_in_sixteen_buckets() {
    let shapes = [(1023, 1023)];
    let descriptor = DenseBvaDescriptor::from_shapes(&shapes).unwrap();

    assert_eq!(descriptor.bucket_count(), 16);
    assert_eq!(descriptor.physical_len(), 1023 * 1023);
    assert_eq!(descriptor.layout().active_len(), 1024 * 1024);
    assert_eq!(descriptor.logical_len(), 1024 * 1024);
    assert_eq!(descriptor.materialized_padding_entries(), 2047);
    assert!(descriptor.bucket_ranges_are_a_partition());

    let row_intervals = descriptor
        .buckets()
        .iter()
        .map(|bucket| (bucket.row_offset(), bucket.row_len()))
        .collect::<std::collections::BTreeSet<_>>();
    let col_intervals = descriptor
        .buckets()
        .iter()
        .map(|bucket| (bucket.col_offset(), bucket.col_len()))
        .collect::<std::collections::BTreeSet<_>>();
    let expected = [(0, 512), (512, 256), (768, 128), (896, 128)]
        .into_iter()
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(row_intervals, expected);
    assert_eq!(col_intervals, expected);

    let source = DenseMatrixBva::from_fn(&shapes, |_, row, col| {
        assert!(row < 1023);
        assert!(col < 1023);
        7
    })
    .unwrap();
    assert_eq!(source.values().len(), 1024 * 1024);
    assert_eq!(
        source.values().iter().filter(|&&value| value == 0).count(),
        2047
    );
}

#[test]
fn capped_split_uses_at_most_sixteen_buckets_per_matrix() {
    let shapes = [(31, 13), (15, 15), (1023, 511), (1, 1)];
    let descriptor = DenseBvaDescriptor::from_shapes(&shapes).unwrap();
    let mut counts = vec![0_usize; shapes.len()];
    for bucket in descriptor.buckets() {
        counts[bucket.matrix()] += 1;
    }
    assert_eq!(counts, vec![12, 16, 16, 1]);
    assert!(counts.into_iter().all(|count| count <= 16));
}

#[test]
fn heterogeneous_matrix_collection_full_descriptor_is_exact() {
    let descriptor = DenseBvaDescriptor::from_shapes(&family_shapes(15)).unwrap();
    assert_eq!(descriptor.matrix_count(), 14);
    assert_eq!(descriptor.physical_len(), 1_431_524_710);
    assert_eq!(descriptor.layout().active_len(), 1_431_655_707);
    assert_eq!(descriptor.bucket_count(), 205);
    assert_eq!(descriptor.logical_len(), 1_usize << 31);
    assert_eq!(descriptor.conceptual_boundary_zeros(), 130_997);
    assert_eq!(descriptor.conceptual_final_zeros(), 715_827_941);
    assert_eq!(descriptor.required_dory_log(), 31);
    assert_eq!(
        descriptor.materialized_padding_entries(),
        descriptor.logical_len() - descriptor.physical_len()
    );
    assert!(descriptor.bucket_ranges_are_a_partition());
}

#[test]
#[ignore = "requires multi-GiB RAM and a release build"]
fn heterogeneous_matrix_collection_2pow30_end_to_end() {
    let shapes = family_shapes(15);
    let physical = shapes
        .iter()
        .map(|&(rows, cols)| rows * cols)
        .sum::<usize>();
    assert_eq!(shapes.len(), 14);
    assert_eq!(physical, 1_431_524_710);
    assert_eq!(
        (2_usize..=15)
            .map(|log| log.min(DenseBvaDescriptor::SPLITTING_BUDGET).pow(2))
            .sum::<usize>(),
        205
    );
    assert_eq!(physical.next_power_of_two(), 1_usize << 31);
    assert_eq!((1_usize << 31) - 1_431_655_707, 715_827_941);
    run_family(15, false);
}
