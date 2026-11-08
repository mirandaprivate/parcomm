//! Independent checks of the base-table statement authenticated by the packed PCS.

use mikros_qafold::{
    FullPrime65521Ext8 as F, Prime65521 as B, ShortBaseField, ShortFieldElement, packed::PackedPcs,
};
use rand::{Rng, SeedableRng, rngs::StdRng};

const CONTEXT: &[u8] = b"packed-pcs/integration-tests/v1";

fn values(log_n: usize) -> Vec<B> {
    let mut rng = StdRng::seed_from_u64(0x6c616e655f746573);
    (0..1usize << log_n)
        .map(|_| B::from_u64(rng.gen_range(0..65_521)))
        .collect()
}

fn point(log_n: usize) -> Vec<F> {
    let mut rng = StdRng::seed_from_u64(0x706f696e745f7465);
    (0..log_n)
        .map(|_| {
            F::from_coefficients(std::array::from_fn(|_| {
                B::from_u64(rng.gen_range(0..65_521))
            }))
        })
        .collect()
}

/// Evaluates the original, unpacked Boolean table without the packing helpers.
fn original_mle(values: &[B], point: &[F]) -> F {
    assert_eq!(values.len(), 1usize << point.len());
    let mut table: Vec<F> = values
        .iter()
        .map(|value| F::from_u64(value.to_canonical_u64()))
        .collect();
    for &coordinate in point {
        let half = table.len() / 2;
        for i in 0..half {
            table[i] = table[i] * (F::one() - coordinate) + table[half + i] * coordinate;
        }
        table.truncate(half);
    }
    table[0]
}

fn pcs(log_n: usize) -> PackedPcs {
    PackedPcs::new(log_n, 4, 32, [91; 32]).unwrap()
}

#[test]
fn extension_query_authenticates_original_base_table() {
    let log_n = 9;
    let scheme = pcs(log_n);
    let values = values(log_n);
    let point = point(log_n);
    assert!(point.iter().all(|value| {
        value.coefficients()[1..]
            .iter()
            .any(|coefficient| *coefficient != B::zero())
    }));
    let expected = original_mle(&values, &point);
    let data = scheme.commit(values).unwrap();
    let root = data.commitment();
    let proof = scheme.open(data, &point, CONTEXT).unwrap();
    assert_eq!(proof.claimed_value, expected);
    let bytes = proof.to_bytes().unwrap();
    scheme
        .verify(&root, &point, expected, &bytes, CONTEXT)
        .unwrap();
    assert!(proof.opening_queries.iter().any(|&count| count != 0));
    assert!(proof.decide_queries.iter().any(|&count| count != 0));
}

#[test]
fn adjacent_eight_lanes_use_the_last_three_msb_first_coordinates() {
    let log_n = 7;
    let scheme = pcs(log_n);
    let values = values(log_n);
    for lane in 0..8 {
        let mut query = point(log_n);
        for bit in 0..3 {
            query[log_n - 3 + bit] = F::from_u64(((lane >> (2 - bit)) & 1) as u64);
        }
        let expected = original_mle(&values, &query);
        let data = scheme.commit(values.clone()).unwrap();
        let root = data.commitment();
        let proof = scheme.open(data, &query, CONTEXT).unwrap();
        assert_eq!(proof.claimed_value, expected, "lane {lane}");
        scheme
            .verify(&root, &query, expected, &proof.to_bytes().unwrap(), CONTEXT)
            .unwrap();
    }
}

#[test]
fn boolean_boundary_points_select_the_original_entry() {
    let log_n = 6;
    let scheme = pcs(log_n);
    let values = values(log_n);
    for index in [0usize, 1, 7, 8, (1usize << log_n) - 1] {
        let query: Vec<F> = (0..log_n)
            .map(|bit| F::from_u64(((index >> (log_n - 1 - bit)) & 1) as u64))
            .collect();
        let expected = F::from_u64(values[index].to_canonical_u64());
        let data = scheme.commit(values.clone()).unwrap();
        let root = data.commitment();
        let proof = scheme.open(data, &query, CONTEXT).unwrap();
        assert_eq!(proof.claimed_value, expected, "original index {index}");
        scheme
            .verify(&root, &query, expected, &proof.to_bytes().unwrap(), CONTEXT)
            .unwrap();
    }
}

fn read_len(bytes: &[u8], offset: usize) -> usize {
    u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap()) as usize
}

/// Replaces one encoded field element with a different, still canonical value.
fn change_field(bytes: &mut [u8], offset: usize) {
    let value = <F as ShortBaseField>::from_canonical_bytes(&bytes[offset..offset + 16]).unwrap();
    (value + F::one())
        .write_canonical_bytes(&mut bytes[offset..offset + 16])
        .unwrap();
}

#[test]
fn rejects_modified_statements_messages_and_noncanonical_encodings() {
    let log_n = 9;
    let scheme = pcs(log_n);
    let query = point(log_n);
    let data = scheme.commit(values(log_n)).unwrap();
    let root = data.commitment();
    let proof = scheme.open(data, &query, CONTEXT).unwrap();
    let claim = proof.claimed_value;
    let bytes = proof.to_bytes().unwrap();
    scheme
        .verify(&root, &query, claim, &bytes, CONTEXT)
        .unwrap();

    let mut changed_root = root;
    changed_root.root[0] ^= 1;
    assert!(
        scheme
            .verify(&changed_root, &query, claim, &bytes, CONTEXT)
            .is_err()
    );
    assert!(
        scheme
            .verify(&root, &query, claim + F::one(), &bytes, CONTEXT)
            .is_err()
    );
    for coordinate in [0, log_n - 1] {
        let mut changed_query = query.clone();
        changed_query[coordinate] += F::one();
        assert!(
            scheme
                .verify(&root, &changed_query, claim, &bytes, CONTEXT)
                .is_err()
        );
    }
    assert!(
        scheme
            .verify(&root, &query, claim, &bytes, b"another context")
            .is_err()
    );
    let other_setup = PackedPcs::new(log_n, 4, 32, [92; 32]).unwrap();
    assert!(
        other_setup
            .verify(&root, &query, claim, &bytes, CONTEXT)
            .is_err()
    );

    // The wrapper stores: magic, claim, 8 conjugate values, count, 2 fields/round,
    // terminal value, then two length-prefixed native proof payloads.
    let count_offset = 8 + 16 + 8 * 16;
    assert_eq!(read_len(&bytes, count_offset), log_n - 3);
    let rounds_offset = count_offset + 8;
    let final_offset = rounds_offset + 32 * (log_n - 3);
    let opening_length_offset = final_offset + 16;
    let opening_offset = opening_length_offset + 8;
    let opening_length = read_len(&bytes, opening_length_offset);
    let decide_length_offset = opening_offset + opening_length;
    let decide_offset = decide_length_offset + 8;
    let decide_length = read_len(&bytes, decide_length_offset);
    assert_eq!(decide_offset + decide_length, bytes.len());
    assert!(opening_length > 16 && decide_length > 16);

    let mut field_offsets = vec![8, final_offset];
    field_offsets.extend((0..8).map(|i| 8 + 16 + 16 * i));
    field_offsets.extend((0..2 * (log_n - 3)).map(|i| rounds_offset + 16 * i));
    for offset in field_offsets {
        let mut changed = bytes.clone();
        change_field(&mut changed, offset);
        assert!(
            scheme
                .verify(&root, &query, claim, &changed, CONTEXT)
                .is_err(),
            "accepted changed wrapper field at byte {offset}"
        );
    }
    for offset in [
        opening_offset + opening_length / 2,
        decide_offset + decide_length / 2,
    ] {
        let mut changed = bytes.clone();
        changed[offset] ^= 1;
        assert!(
            scheme
                .verify(&root, &query, claim, &changed, CONTEXT)
                .is_err(),
            "accepted changed native proof byte {offset}"
        );
    }
    for offset in [count_offset, opening_length_offset, decide_length_offset] {
        let mut changed = bytes.clone();
        changed[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(
            scheme
                .verify(&root, &query, claim, &changed, CONTEXT)
                .is_err()
        );
    }
    let mut noncanonical = bytes.clone();
    noncanonical[8 + 16..8 + 16 + 2].copy_from_slice(&65_521u16.to_le_bytes());
    assert!(
        scheme
            .verify(&root, &query, claim, &noncanonical, CONTEXT)
            .is_err()
    );
    for end in [
        0,
        7,
        8,
        count_offset,
        final_offset,
        opening_offset,
        decide_offset,
        bytes.len() - 1,
    ] {
        assert!(
            scheme
                .verify(&root, &query, claim, &bytes[..end], CONTEXT)
                .is_err(),
            "accepted truncation at byte {end}"
        );
    }
    let mut trailing = bytes.clone();
    trailing.push(0);
    assert!(
        scheme
            .verify(&root, &query, claim, &trailing, CONTEXT)
            .is_err()
    );
    let mut magic = bytes;
    magic[0] ^= 1;
    assert!(
        scheme
            .verify(&root, &query, claim, &magic, CONTEXT)
            .is_err()
    );
}

#[test]
fn rejects_shape_and_profile_mismatches() {
    assert!(PackedPcs::new(3, 4, 32, [91; 32]).is_err());
    assert!(PackedPcs::new(31, 4, 32, [91; 32]).is_err());
    let log_n = 7;
    let scheme = pcs(log_n);
    assert!(scheme.commit(vec![]).is_err());
    assert!(scheme.commit(values(log_n - 1)).is_err());
    let query = point(log_n);
    let data = scheme.commit(values(log_n)).unwrap();
    assert!(scheme.open(data, &query[..log_n - 1], CONTEXT).is_err());
    let other = PackedPcs::new(log_n, 4, 32, [92; 32]).unwrap();
    let data = scheme.commit(values(log_n)).unwrap();
    assert!(other.open(data, &query, CONTEXT).is_err());
}
