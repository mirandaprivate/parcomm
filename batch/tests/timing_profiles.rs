use ark_bn254::Fr;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use batch::{
    hidden_relation::HiddenScalarWitness,
    jolt_dory::{
        evaluate_field_vector_standard, setup_dory_parameters, setup_dory_parameters_profiled,
    },
    section6_final_opening::{
        commit_section6_field_on_grid_zk, prove_section6_final_opening_profiled,
        verify_section6_final_opening_profiled, Section6DoryGrid, Section6FinalOpeningData,
        Section6FinalOpeningProverClaim, Section6FinalOpeningVerifierClaim,
    },
};

#[test]
fn dory_setup_profile_classifies_pcs_and_blindfold_work() {
    let (_, _, profile) = setup_dory_parameters_profiled(2);
    let classified_seconds = profile.pcs_setup_seconds + profile.blindfold_setup_seconds;
    assert!(profile.total_seconds >= classified_seconds);
}

#[test]
fn terminal_opening_profiles_are_exhaustive_and_the_proof_verifies() {
    let grid = Section6DoryGrid::new(4).unwrap();
    let (prover_parameters, verifier_parameters) = setup_dory_parameters(grid.num_vars());
    let values = (1_u64..=8).map(Fr::from).collect::<Vec<_>>();
    let committed = commit_section6_field_on_grid_zk(&prover_parameters, &grid, &values).unwrap();
    let point = vec![Fr::from(2_u64), Fr::from(3_u64), Fr::from(5_u64)];
    let evaluation = evaluate_field_vector_standard(&values, &point).unwrap();
    let mut rng = StdRng::seed_from_u64(0x5449_4d49_4e47);
    let witness = HiddenScalarWitness::commit(&prover_parameters, evaluation, &mut rng);
    let prover_claim = Section6FinalOpeningProverClaim {
        data: Section6FinalOpeningData::Field(&values),
        committed: &committed,
        point_msb: &point,
        evaluation: &witness,
    };

    let (proof, prover_profile) = prove_section6_final_opening_profiled(
        b"section6-timing-profile-test",
        b"outer-binding",
        &prover_parameters,
        &grid,
        &[prover_claim],
        &mut rng,
    )
    .unwrap();
    assert!(prover_profile.total_seconds >= prover_profile.classified_seconds());
    assert_eq!(proof.physical_dory_proof_count(), 1);

    let verifier_claim = Section6FinalOpeningVerifierClaim {
        commitment: &committed.commitment,
        logical_len: committed.logical_len,
        point_msb: &point,
        evaluation_commitment: witness.commitment,
    };
    let verifier_profile = verify_section6_final_opening_profiled(
        b"section6-timing-profile-test",
        b"outer-binding",
        &verifier_parameters,
        &grid,
        &[verifier_claim],
        &proof,
    )
    .unwrap();
    assert!(verifier_profile.total_seconds >= verifier_profile.classified_seconds());
}
