use mikros_qafold::{
    ShortBaseField, ShortFieldElement,
    koalabear::{KoalaBear as B, KoalaBearExt4 as F},
    packed_koalabear::{PackedPcs, PackedCommitWorkspace},
};
#[test]
fn reconstructed_witness_keeps_proof_bytes_and_rejects_wrong_source() {
    let pcs = PackedPcs::new(10, 32, 100, [17; 32]).unwrap();
    let source = |index| B::from_u64((index * 17 + 9) as u64);
    let point = vec![F::from_u64(13); 10];
    let values = (0..1024).map(source).collect::<Vec<_>>();
    let original = pcs.commit(values.clone()).unwrap();
    let root = original.commitment();
    let proof = pcs.open(original, &point, b"cache-test").unwrap();
    let mut compact = pcs.commit(values.clone()).unwrap();
    compact.discard_values();
    let rebuilt = pcs
        .open_from_fn(compact, &point, b"cache-test", source)
        .unwrap();
    assert_eq!(proof.to_bytes().unwrap(), rebuilt.to_bytes().unwrap());
    pcs.verify(
        &root,
        &point,
        rebuilt.claimed_value,
        &rebuilt.to_bytes().unwrap(),
        b"cache-test",
    )
    .unwrap();
    let mut wrong = pcs.commit(values).unwrap();
    wrong.discard_values();
    assert!(
        pcs.open_from_fn(wrong, &point, b"cache-test", |index| source(index)
            + B::one())
            .is_err()
    );
}

#[test]
fn recycled_commit_buffers_preserve_roots_and_proof_bytes() {
    let mut workspace = PackedCommitWorkspace::default();
    // Repeated size, grow, shrink, and overwrite nonzero data with a zero block.
    for (log, salt) in [(10, 7), (10, 0), (13, 19), (8, 23), (10, 31)] {
        let pcs = PackedPcs::new(log, 32, 100, [17; 32]).unwrap();
        let source = |index| B::from_u64((salt * (index * 17 + 9)) as u64);
        let original = pcs.commit_from_fn(source).unwrap();
        let recycled = pcs.commit_from_fn_reusing(source, &mut workspace).unwrap();
        let root = recycled.commitment();
        assert_eq!(root, original.commitment());
        recycled.recycle(&mut workspace);
        let recycled = pcs.commit_from_fn_reusing(source, &mut workspace).unwrap();
        let point = vec![F::from_u64(13); log];
        let expected = pcs.open_from_fn(original, &point, b"recycle-test", source).unwrap();
        let actual = pcs.open_from_fn(recycled, &point, b"recycle-test", source).unwrap();
        assert_eq!(actual.to_bytes().unwrap(), expected.to_bytes().unwrap());
        pcs.verify(&root, &point, actual.claimed_value, &actual.to_bytes().unwrap(), b"recycle-test").unwrap();
        pcs.commit_from_fn_reusing(source, &mut workspace).unwrap().recycle(&mut workspace);
    }
}
