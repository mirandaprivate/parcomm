
use parcomm_qafold::{layout::{Layout, Shape}, tbv, field::F, Config, Parameters, Proof};
use mikros_qafold::ShortBaseField;

#[test]
fn recomputation_preserves_complete_parcomm_proof() {
    let shapes = [Shape { rows: 67, cols: 59 }, Shape { rows: 33, cols: 29 }];
    for packed_log in [4, 6] {
        let config = Config { chunk_log: 10, packed_log, query_rf_bits: 100 };
        let parameters = Parameters::new(&shapes, config).unwrap();
        let matrices = shapes.iter().map(|shape| (0..shape.rows * shape.cols)
            .map(|index| index.wrapping_mul(73).wrapping_add(19) as i8).collect()).collect::<Vec<_>>();
        let values = parameters.layout.encode(&matrices).unwrap();
        let points = shapes.iter().map(|shape| (
            vec![F::from_u64(3); shape.rows.next_power_of_two().ilog2() as usize],
            vec![F::from_u64(7); shape.cols.next_power_of_two().ilog2() as usize])).collect::<Vec<_>>();
        let claims = tbv::evaluate_claims(&parameters.layout, &values, &points).unwrap();
        let cached = parameters.commit(values.clone()).unwrap();
        let rebuilt = parameters.commit_recomputed(values).unwrap();
        assert_eq!(cached.statement, rebuilt.statement);
        let statement = cached.statement.clone();
        let (old, _) = parameters.prove(cached, &claims).unwrap();
        let (new, _) = parameters.prove(rebuilt, &claims).unwrap();
        assert_eq!(old.to_bytes().unwrap(), new.to_bytes().unwrap());
        let decoded = Proof::from_bytes(&new.to_bytes().unwrap()).unwrap();
        let verified = parameters.verify(&statement, &claims, &decoded).unwrap();
        assert_eq!(verified.native_openings, 1);
        assert_eq!(verified.authenticated_roots, statement.roots.len());
    }
}

#[test]
fn layout_supports_70b_without_allocating_the_witness() {
    let layout = Layout::new(&[Shape { rows: 1 << 18, cols: (1 << 18) + 16 }]).unwrap();
    assert!(layout.active > 1usize << 36);
    assert_eq!(layout.logical, 1usize << 37);
    assert_eq!(layout.logical / (1usize << 30), 128);
    assert!(Layout::new(&[Shape { rows: 1 << 20, cols: 1 << 20 }]).is_err());
}
