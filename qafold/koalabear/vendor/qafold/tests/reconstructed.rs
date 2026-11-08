use mikros_qafold::{
    ShortBaseField, ShortFieldElement,
    koalabear::{KoalaBear as B, KoalaBearExt4 as F},
    packed_koalabear::PackedPcs,
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
