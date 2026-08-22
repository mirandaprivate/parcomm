use ark_bn254::Fr;
use ark_std::rand::{rngs::StdRng, SeedableRng};
use batch::{
    hidden_relation::{
        prove_hidden_relation, verify_hidden_relation, HiddenRelationProof, HiddenRelationTable,
        HiddenRelationTableStatement, RelationExpression,
    },
    jolt_dory::{
        commit_field_matrix_columns_zk, setup_dory_parameters, CommittedDoryFieldVector,
        DoryProverParameters,
    },
};
use rayon::ThreadPoolBuilder;

fn prove_matrix_product(
    thread_count: usize,
    parameters: &DoryProverParameters,
    columns: &[Vec<Fr>],
    committed: &[CommittedDoryFieldVector],
) -> HiddenRelationProof {
    let tables = columns
        .iter()
        .zip(committed)
        .map(|(values, committed)| HiddenRelationTable { values, committed })
        .collect::<Vec<_>>();
    let expression = RelationExpression::table(0)
        .mul(RelationExpression::table(1))
        .sub(RelationExpression::table(2));
    let pool = ThreadPoolBuilder::new()
        .num_threads(thread_count)
        .build()
        .unwrap();
    let mut rng = StdRng::seed_from_u64(0x5eed_1234);
    pool.install(|| {
        prove_hidden_relation(
            b"hidden-relation-parallel-equivalence",
            parameters,
            &tables,
            &expression,
            &mut rng,
        )
        .unwrap()
    })
}

fn serialized_transcript_projection(proof: &HiddenRelationProof) -> Vec<u8> {
    let matrix_batches = proof
        .matrix_dory_batches
        .iter()
        .map(|batch| (batch.padded_cols, &batch.column_evaluation_commitments))
        .collect::<Vec<_>>();
    bincode::serialize(&(
        &proof.sumcheck,
        proof.vector_dory_batch.is_some(),
        matrix_batches,
        &proof.evaluation_commitments,
        proof.blindfold_transcript_state,
        &proof.blindfold,
    ))
    .unwrap()
}

#[test]
fn parallel_prover_preserves_serialized_transcript_and_proof_layout() {
    let rows = 64;
    let (prover_parameters, verifier_parameters) = setup_dory_parameters(8);
    let left = (0..rows)
        .map(|index| Fr::from(index as u64 + 2))
        .collect::<Vec<_>>();
    let right = (0..rows)
        .map(|index| Fr::from((3 * index + 5) as u64))
        .collect::<Vec<_>>();
    let product = left
        .iter()
        .zip(&right)
        .map(|(left, right)| *left * right)
        .collect::<Vec<_>>();
    let columns = vec![left, right, product];
    let committed = commit_field_matrix_columns_zk(&prover_parameters, &columns, 4).unwrap();

    let sequential = prove_matrix_product(1, &prover_parameters, &columns, &committed);
    let parallel = prove_matrix_product(4, &prover_parameters, &columns, &committed);
    let sequential_bytes = bincode::serialize(&sequential).unwrap();
    let parallel_bytes = bincode::serialize(&parallel).unwrap();

    assert_eq!(
        serialized_transcript_projection(&parallel),
        serialized_transcript_projection(&sequential)
    );
    assert_eq!(parallel_bytes.len(), sequential_bytes.len());

    let statements = committed
        .iter()
        .map(HiddenRelationTableStatement::from)
        .collect::<Vec<_>>();
    let expression = RelationExpression::table(0)
        .mul(RelationExpression::table(1))
        .sub(RelationExpression::table(2));
    verify_hidden_relation(
        b"hidden-relation-parallel-equivalence",
        &verifier_parameters,
        &statements,
        &expression,
        &parallel,
    )
    .unwrap();
    verify_hidden_relation(
        b"hidden-relation-parallel-equivalence",
        &verifier_parameters,
        &statements,
        &expression,
        &sequential,
    )
    .unwrap();
}
