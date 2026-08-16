use std::{
    convert::Infallible,
    io::{self, Write},
    mem::size_of,
    time::Instant,
};

use ark_bls12_381::Fr;
use ark_serialize::{CanonicalSerialize, Compress};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use zksmart::protocols::{
    bucketed_ip::{
        packing_stride_for_bit_width, BvaI8Source, BvaLayout, PackedEoLiteBulletProof,
        TensorBasisVector,
    },
    virtualpcs::inner_product_bva_i8_streaming,
};

struct ByteBvaSource {
    layout: BvaLayout,
    vectors: Vec<Vec<i8>>,
}

impl BvaI8Source for ByteBvaSource {
    type Error = Infallible;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[i8])) -> Result<(), Self::Error> {
        for (index, vector) in self.vectors.iter().enumerate() {
            visit(index, vector);
        }
        Ok(())
    }
}

fn parse_lengths() -> Vec<usize> {
    let mut arguments = std::env::args().skip(1);
    let mut log_largest = 16_usize;
    let mut geometric_max_log = None;
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--log-largest" => {
                log_largest = arguments
                    .next()
                    .expect("missing --log-largest value")
                    .parse()
                    .expect("invalid --log-largest value");
            }
            "--geometric-max-log" => {
                geometric_max_log = Some(
                    arguments
                        .next()
                        .expect("missing --geometric-max-log value")
                        .parse::<usize>()
                        .expect("invalid --geometric-max-log value"),
                );
            }
            _ => panic!("unknown argument {}", argument),
        }
    }
    if let Some(max_log) = geometric_max_log {
        assert!(max_log <= 30);
        return (0..=max_log).rev().map(|log| 1_usize << log).collect();
    }
    assert!((8..=28).contains(&log_largest));
    let largest = 1_usize << log_largest;
    vec![
        largest - 3,
        largest / 2 - 5,
        largest / 4 - 1,
        17,
        9,
        3,
        1,
        1,
        1,
    ]
}

fn main() {
    let lengths = parse_lengths();
    let requested_physical_entries = lengths.iter().sum::<usize>();
    println!("rayon_threads={}", rayon::current_num_threads());
    println!("vector_count={}", lengths.len());
    println!("requested_physical_entries={requested_physical_entries}");
    io::stdout().flush().expect("flush benchmark metadata");
    let generation_started = Instant::now();
    let vectors = lengths
        .par_iter()
        .enumerate()
        .map(|(vector_index, length)| {
            let magnitude = (vector_index % 31 + 1) as i8;
            let value = if vector_index & 1 == 0 {
                magnitude
            } else {
                -magnitude
            };
            vec![value; *length]
        })
        .collect::<Vec<_>>();
    let generation_seconds = generation_started.elapsed().as_secs_f64();
    let physical_entries = vectors.iter().map(Vec::len).sum::<usize>();
    assert_eq!(physical_entries, requested_physical_entries);
    let layout = BvaLayout::from_lengths(&lengths).expect("valid decreasing BVA layout");
    let packing_stride =
        packing_stride_for_bit_width::<Fr>(layout.logical_len(), 8).expect("valid i8 packing");
    let challenges = (0..layout.logical_len().ilog2())
        .map(|round| Fr::from(u64::from(round) + 2))
        .collect::<Vec<_>>();
    let tbv = TensorBasisVector::global_tensor(&layout, &challenges).expect("valid global TBV");
    let mut source = ByteBvaSource { layout, vectors };

    let claim_started = Instant::now();
    let claim = inner_product_bva_i8_streaming(&mut source, &tbv)
        .expect("native i8 TBV evaluation")
        .value;
    let claim_seconds = claim_started.elapsed().as_secs_f64();

    let prove_started = Instant::now();
    let (proof, counters, profile) =
        PackedEoLiteBulletProof::prove_i8_streaming(&mut source, &tbv, claim, 8)
            .expect("packed EO proof");
    let prove_seconds = prove_started.elapsed().as_secs_f64();

    let verify_started = Instant::now();
    let verification = proof.verify_with_tbv(&tbv, claim);
    let verify_seconds = verify_started.elapsed().as_secs_f64();
    assert!(verification.accepted);
    let proof_sha256 = Sha256::digest(proof.to_bytes().expect("serialize packed EO proof"));
    let proof_sha256_hex = proof_sha256
        .iter()
        .map(|byte| format!("{:02x}", byte))
        .collect::<String>();

    let field_element_storage_bytes = size_of::<Fr>();
    let maximum_live_reduced_field_bytes = profile
        .maximum_live_vector_entries
        .checked_mul(field_element_storage_bytes)
        .expect("reduced field storage size");
    let minimum_simultaneous_witness_and_reduced_bytes = physical_entries
        .checked_add(maximum_live_reduced_field_bytes)
        .expect("simultaneous witness and reduced storage size");
    assert_eq!(field_element_storage_bytes, 32);
    assert_eq!(
        profile.maximum_live_vector_entries,
        source.layout.active_len().div_ceil(packing_stride)
    );

    println!("physical_entries={physical_entries}");
    println!("witness_storage_bytes={physical_entries}");
    println!("field_element_storage_bytes={field_element_storage_bytes}");
    println!("active_cover_entries={}", source.layout.active_len());
    println!("logical_entries={}", source.layout.logical_len());
    println!("packing_stride={packing_stride}");
    println!("proof_field_elements={}", proof.field_element_count());
    println!("proof_size_bytes={}", proof.serialized_size(Compress::Yes));
    println!("proof_sha256={proof_sha256_hex}");
    println!("generation_seconds={generation_seconds:.6}");
    println!("claim_seconds={claim_seconds:.6}");
    println!("prove_seconds={prove_seconds:.6}");
    println!("verify_seconds={verify_seconds:.6}");
    println!("source_passes={}", profile.source_passes);
    println!(
        "maximum_live_reduced_field_entries={}",
        profile.maximum_live_vector_entries
    );
    println!("maximum_live_reduced_field_bytes={maximum_live_reduced_field_bytes}");
    println!(
        "minimum_simultaneous_witness_and_reduced_bytes={minimum_simultaneous_witness_and_reduced_bytes}"
    );
    println!(
        "materialized_padding_entries={}",
        counters.materialized_padding_entries
    );
    println!("accepted={}", verification.accepted);
}
