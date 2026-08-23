use std::convert::Infallible;

use ark_bn254::Fr;
use batch::bucketed_ip::{
    BvaInteger, BvaLayout, BvaSource, PackedEoLiteBulletProof, TensorBasisVector,
};

struct OwnedBva<Entry: BvaInteger> {
    layout: BvaLayout,
    vectors: Vec<Vec<Entry>>,
}

impl<Entry: BvaInteger> BvaSource<Entry> for OwnedBva<Entry> {
    type Error = Infallible;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn indexed_vectors(&self) -> Option<Vec<&[Entry]>> {
        Some(self.vectors.iter().map(Vec::as_slice).collect())
    }

    fn for_each_vector(
        &mut self,
        visit: &mut dyn FnMut(usize, &[Entry]),
    ) -> Result<(), Self::Error> {
        for (index, vector) in self.vectors.iter().enumerate() {
            visit(index, vector);
        }
        Ok(())
    }
}

fn prove_with_short_int<Entry>() -> usize
where
    Entry: BvaInteger + From<i8>,
{
    let vectors = [33_usize, 17, 7]
        .into_iter()
        .enumerate()
        .map(|(vector, length)| {
            (0..length)
                .map(|entry| Entry::from(((vector * 11 + entry * 7) % 31) as i8 - 15))
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let layout =
        BvaLayout::from_lengths(&vectors.iter().map(Vec::len).collect::<Vec<_>>()).unwrap();
    let challenges = (0..layout.logical_len().ilog2())
        .map(|round| Fr::from(u64::from(round) + 2))
        .collect::<Vec<_>>();
    let tbv = TensorBasisVector::global_tensor(&layout, &challenges).unwrap();
    let mut source = OwnedBva { layout, vectors };
    let claimed = tbv
        .inner_product_streaming::<Entry, _>(&mut source)
        .unwrap();
    let (proof, counters, profile) = PackedEoLiteBulletProof::prove_streaming::<Entry, _>(
        &mut source,
        &tbv,
        claimed,
        Entry::BITS,
    )
    .unwrap();

    assert!(proof.verify_with_tbv(&tbv, claimed).accepted);
    assert_eq!(counters.materialized_padding_entries, 0);
    assert_eq!(profile.materialized_padding_entries, 0);
    proof.packing_stride
}

#[test]
fn short_int_selects_native_i8_or_i16_packed_eo() {
    assert_eq!(prove_with_short_int::<i8>(), 64);
    assert_eq!(prove_with_short_int::<i16>(), 32);
}
