use super::*;

fn source(block: usize, index: usize) -> B {
    B::from_u64(
        (index as u64)
            .wrapping_mul(7927)
            .wrapping_add(block as u64 * 65519 + 31),
    )
}
fn states(pcs: &PackedPcs, blocks: usize) -> Vec<PackedData> {
    (0..blocks)
        .map(|block| {
            pcs.commit(
                (0..1usize << pcs.log_n)
                    .map(|index| source(block, index))
                    .collect(),
            )
            .unwrap()
        })
        .collect()
}
fn reference(blocks: usize, log_n: usize, point: &[F]) -> F {
    let mut table = (0..blocks)
        .flat_map(|block| {
            (0..1usize << log_n)
                .map(move |index| F::from_u64(source(block, index).to_canonical_u64()))
        })
        .collect::<Vec<_>>();
    for &coordinate in point {
        let half = table.len() / 2;
        for index in 0..half {
            table[index] =
                table[index] * (F::one() - coordinate) + table[index + half] * coordinate;
        }
        table.truncate(half);
    }
    table[0]
}

#[test]
fn batch_mle_matches_direct_oracle_including_boolean_points() {
    let mut rng = StdRng::seed_from_u64(9173);
    for (blocks, log_n) in [(2usize, 10), (4, 10), (8, 10), (64, 10), (2, 16)] {
        let pcs = PackedPcs::new(log_n, 32, 100, [0x51; 32]).unwrap();
        let mut point = (0..log_n + blocks.ilog2() as usize)
            .map(|_| F::sample_uniform(&mut rng))
            .collect::<Vec<_>>();
        if blocks == 4 {
            point[0] = F::zero();
            point[1] = F::one();
            point[log_n] = F::one();
        }
        let data = states(&pcs, blocks);
        let roots = data.iter().map(PackedData::commitment).collect::<Vec<_>>();
        let proof = pcs
            .open_batch_from_fn(data, &point, b"batch-test", source)
            .unwrap();
        assert_eq!(proof.claimed_value, reference(blocks, log_n, &point));
        pcs.verify_batch(
            &roots,
            &point,
            proof.claimed_value,
            &proof.wire,
            b"batch-test",
        )
        .unwrap();
        assert!(
            pcs.verify_batch(
                &roots,
                &point,
                proof.claimed_value,
                &proof.wire,
                b"different-context"
            )
            .is_err()
        );
        assert!(
            pcs.verify_batch(
                &roots[..blocks - 1],
                &point,
                proof.claimed_value,
                &proof.wire,
                b"batch-test"
            )
            .is_err()
        );
        for block in 0..blocks {
            let mut bad = roots.clone();
            bad[block].root[0] ^= 1;
            assert!(
                pcs.verify_batch(
                    &bad,
                    &point,
                    proof.claimed_value,
                    &proof.wire,
                    b"batch-test"
                )
                .is_err()
            );
        }
        let cross_start = 8 + 16 + LANES * 16;
        let folded_root_start = cross_start + 2 * (blocks - 1) * 16;
        let final_start = folded_root_start + 32 + (log_n - LANES.ilog2() as usize) * 32;
        let binding_start = final_start + 16 + 8;
        for offset in [
            0,
            8,
            24,
            cross_start,
            folded_root_start,
            final_start,
            binding_start,
            proof.wire.len() - 1,
        ] {
            let mut wire = proof.wire.clone();
            wire[offset] ^= 1;
            assert!(
                pcs.verify_batch(&roots, &point, proof.claimed_value, &wire, b"batch-test")
                    .is_err(),
                "accepted corruption at {offset}"
            );
        }
        let mut wire = proof.wire.clone();
        wire.push(0);
        assert!(
            pcs.verify_batch(&roots, &point, proof.claimed_value, &wire, b"batch-test")
                .is_err()
        );
        assert!(
            pcs.verify_batch(
                &roots,
                &point,
                proof.claimed_value,
                &proof.wire[..proof.wire.len() - 1],
                b"batch-test"
            )
            .is_err()
        );
    }
}

#[test]
fn valid_folded_commitment_with_wrong_linear_relation_is_rejected() {
    let pcs = PackedPcs::new(10, 32, 100, [0x51; 32]).unwrap();
    let data = states(&pcs, 2);
    let roots = data.iter().map(PackedData::commitment).collect::<Vec<_>>();
    let caches = data.iter().map(|state| &state.cache).collect::<Vec<_>>();
    let weights = [F::one(), F::from_u64(2)];
    let wrong = [F::one(), F::one()];
    let (root, cache) = pcs.pc.fold_codewords(&pcs.pp, &caches, &wrong).unwrap();
    let binding = pcs
        .pc
        .prove_codeword_fold(&pcs.pp, &caches, &cache, [0x78; 32])
        .unwrap();
    assert!(
        pcs.pc
            .verify_codeword_fold(&pcs.pp, &roots, &wrong, &root, [0x78; 32], &binding)
            .unwrap()
    );
    assert!(
        !pcs.pc
            .verify_codeword_fold(&pcs.pp, &roots, &weights, &root, [0x78; 32], &binding)
            .unwrap()
    );
}

#[test]
fn replacement_witness_cannot_open_original_roots() {
    let pcs = PackedPcs::new(10, 32, 100, [0x51; 32]).unwrap();
    let point = vec![F::from_u64(7); 13];
    let result =
        pcs.open_batch_from_fn(states(&pcs, 8), &point, b"wrong-source", |block, index| {
            source(block, index) + B::one()
        });
    assert!(result.is_err());
}

#[test]
fn direct_commit_preserves_root_and_complete_opening() {
    let mut rng = StdRng::seed_from_u64(7729);
    for log_n in [10usize, 16] {
        let pcs = PackedPcs::new(log_n, 32, 100, [0x51; 32]).unwrap();
        let ordinary = pcs.commit((0..1usize << log_n).map(|index| source(3, index)).collect()).unwrap();
        let direct = pcs.commit_from_fn(|index| source(3, index)).unwrap();
        assert_eq!(ordinary.commitment().root, direct.commitment().root);
        assert!(direct.values.is_empty());
        let point = (0..log_n).map(|_| F::sample_uniform(&mut rng)).collect::<Vec<_>>();
        let old = pcs.open(ordinary, &point, b"direct-commit-test").unwrap();
        let new = pcs.open_from_fn(direct, &point, b"direct-commit-test", |index| source(3, index)).unwrap();
        assert_eq!(old.to_bytes().unwrap(), new.to_bytes().unwrap());
    }
}


#[test]
fn recomputed_batch_is_byte_identical_and_checks_sources() {
    let mut rng = StdRng::seed_from_u64(91247);
    for (blocks, log_n) in [(2usize, 10), (8, 10), (128, 10), (2, 16)] {
        let pcs = PackedPcs::new(log_n, 32, 100, [0x51; 32]).unwrap();
        let point = (0..log_n + blocks.ilog2() as usize)
            .map(|_| F::sample_uniform(&mut rng)).collect::<Vec<_>>();
        let input = |block, index| if block < blocks / 2 { source(block, index) } else { B::zero() };
        let data = (0..blocks).map(|block| pcs.commit_from_fn(|index| input(block, index)).unwrap()).collect::<Vec<_>>();
        let roots = data.iter().map(PackedData::commitment).collect::<Vec<_>>();
        let cached = pcs.open_batch_from_fn(data, &point, b"recompute", input).unwrap();
        let rebuilt = pcs.open_batch_recomputed_from_fn(roots.clone(), &point, b"recompute", input).unwrap();
        assert_eq!(cached.claimed_value, rebuilt.claimed_value);
        assert_eq!(cached.wire, rebuilt.wire);
        pcs.verify_batch(&roots, &point, rebuilt.claimed_value, &rebuilt.wire, b"recompute").unwrap();
        if blocks == 2 && log_n == 10 {
            let mut corrupted = roots.clone(); corrupted[0].root[0] ^= 1;
            assert!(pcs.open_batch_recomputed_from_fn(corrupted, &point, b"recompute", input).is_err());
            assert!(pcs.open_batch_recomputed_from_fn(roots, &point, b"recompute", |block, index| input(block, index) + B::one()).is_err());
        }
    }
}
