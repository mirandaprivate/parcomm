use std::collections::BTreeSet;
use std::convert::TryFrom;
use std::mem::MaybeUninit;
use std::time::Instant;

use ark_bls12_381::Bls12_381;
use zksmart::hdr1cs::global_mlp::normalization::{
    GlobalHdR1csNormalizationProtocol, LookupTag, NormalizationInputs, NormalizationManifest,
    OperationFamily, TaggedLookupTable,
};

#[global_allocator]
static ALLOCATOR: jemallocator::Jemalloc = jemallocator::Jemalloc;

fn main() {
    let width = parse_width();
    let (manifest, inputs) = fixture(width);

    println!("experiment=global_hd_r1cs_layernorm_softmax");
    println!("proof_organization=one_global_instance_not_per_layer");
    println!("operation_families=linear,bilinear,tagged_lookup");
    println!("lookup_mode=transparent_rows_not_zero_knowledge");
    println!("fixture=synthetic_integer_tables");
    println!("width={width}");
    report_memory("before");

    let encode_start = Instant::now();
    let encoding = GlobalHdR1csNormalizationProtocol::<Bls12_381>::encode(manifest, inputs)
        .expect("normalization encoding must succeed");
    println!("encode_seconds={:.6}", encode_start.elapsed().as_secs_f64());
    println!(
        "layer_norm_output_len={}",
        encoding.layer_norm_output().len()
    );
    println!("softmax_output_len={}", encoding.softmax_output().len());
    println!(
        "operation_family_count={}",
        encoding.operation_family_count()
    );
    println!(
        "linear_relation_count={}",
        encoding.operation_count(OperationFamily::Linear)
    );
    println!(
        "bilinear_relation_count={}",
        encoding.operation_count(OperationFamily::Bilinear)
    );
    println!(
        "tagged_lookup_query_count={}",
        encoding.operation_count(OperationFamily::TaggedLookup)
    );
    println!("witness_block_count={}", encoding.witness_block_count());
    println!(
        "physically_stored_witness_entries={}",
        encoding.stored_witness_entries()
    );
    println!(
        "logical_padded_witness_entries={}",
        encoding.logical_padded_witness_entries()
    );
    let shape = encoding.global_shape();
    println!("global_block_domain_n={}", shape.n);
    println!("global_witness_slots={}", shape.witness_slots);
    println!("global_gate_slots={}", shape.gate_slots);
    report_memory("after_encode");

    let setup_start = Instant::now();
    let srs = GlobalHdR1csNormalizationProtocol::<Bls12_381>::setup(&encoding)
        .expect("normalization setup must succeed");
    println!("setup_seconds={:.6}", setup_start.elapsed().as_secs_f64());
    report_memory("after_setup");

    let prove_start = Instant::now();
    let (instance, proof) = GlobalHdR1csNormalizationProtocol::<Bls12_381>::prove(&srs, encoding)
        .expect("normalization proof must succeed");
    println!("prove_seconds={:.6}", prove_start.elapsed().as_secs_f64());
    println!("proof_size_bytes={}", proof.serialized_size_bytes());
    println!(
        "witness_commitment_count={}",
        proof.witness_commitment_count()
    );
    println!("witness_opening_count={}", proof.witness_opening_count());
    println!(
        "top_level_equation_count={}",
        proof.top_level_equation_count()
    );
    report_memory("after_prove");

    let verify_start = Instant::now();
    let verified = GlobalHdR1csNormalizationProtocol::<Bls12_381>::verify(&srs, &instance, &proof);
    println!("verify_seconds={:.6}", verify_start.elapsed().as_secs_f64());
    println!("verified={verified}");
    report_memory("final");
    assert!(verified);
}

fn fixture(width: usize) -> (NormalizationManifest, NormalizationInputs) {
    assert!(width >= 4 && width.is_power_of_two());
    let layer_norm_input = (0..width)
        .map(|index| i32::try_from(index % 4 + 1).unwrap())
        .collect::<Vec<_>>();
    let softmax_scores = (0..width)
        .map(|index| i32::try_from(index % 4).unwrap())
        .collect::<Vec<_>>();
    let width_i32 = i32::try_from(width).unwrap();
    let sum = layer_norm_input.iter().sum::<i32>();
    let centered = layer_norm_input
        .iter()
        .map(|value| width_i32 * value - sum)
        .collect::<Vec<_>>();
    let epsilon = 1_i32;
    let variance = centered
        .iter()
        .map(|value| value.checked_mul(*value).unwrap())
        .sum::<i32>()
        .checked_add(epsilon)
        .unwrap();

    let softmax_output_shift = 12_usize;
    let divisor = 1_i32 << softmax_output_shift;
    let exponential_rows = vec![(-3, 1), (-2, 2), (-1, 4), (0, 8)];
    let denominator = softmax_scores
        .iter()
        .map(|score| {
            exponential_rows
                .iter()
                .find(|row| row.0 == score - 3)
                .unwrap()
                .1
        })
        .sum::<i32>();
    // The requantization relation divides the product by S, so the lookup
    // multiplier approximates S/z in Qq representation, i.e. S^2/z.
    let reciprocal = divisor
        .checked_mul(divisor)
        .unwrap()
        .checked_div(denominator)
        .unwrap();
    let softmax_base_sum = softmax_scores
        .iter()
        .map(|score| {
            exponential_rows
                .iter()
                .find(|row| row.0 == score - 3)
                .unwrap()
                .1
        })
        .map(|exponential| exponential * reciprocal / divisor)
        .sum::<i32>();
    let softmax_residual = divisor - softmax_base_sum;
    let mut nonnegative = BTreeSet::from([0_i32, 1, 2, 3]);
    for exponential in [1_i32, 2, 4, 8] {
        nonnegative.insert((exponential * reciprocal).rem_euclid(divisor));
    }
    nonnegative.insert(softmax_residual);

    (
        NormalizationManifest {
            width,
            epsilon,
            output_shift: 3,
            softmax_output_shift,
            commitment_bit_width: 24,
            tables: vec![
                TaggedLookupTable::new(LookupTag::Exp, exponential_rows),
                TaggedLookupTable::new(LookupTag::Reciprocal, vec![(denominator, reciprocal)]),
                TaggedLookupTable::new(LookupTag::InverseSqrt, vec![(variance, 8)]),
                TaggedLookupTable::new(
                    LookupTag::NonNegative,
                    nonnegative
                        .into_iter()
                        .map(|value| (value, value))
                        .collect(),
                ),
                TaggedLookupTable::new(LookupTag::Boolean, vec![(0, 0), (1, 1)]),
            ],
        },
        NormalizationInputs {
            layer_norm_input,
            layer_norm_gamma: vec![1; width],
            layer_norm_beta: vec![0; width],
            softmax_scores,
        },
    )
}

fn parse_width() -> usize {
    let args = std::env::args().collect::<Vec<_>>();
    let mut width = 4_usize;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--bench" => {}
            "--width" => {
                index += 1;
                width = args[index].parse().unwrap();
            }
            other => panic!("unknown argument: {}", other),
        }
        index += 1;
    }
    assert!((4..=64).contains(&width) && width.is_power_of_two());
    width
}

fn report_memory(phase: &str) {
    println!("peak_rss_{phase}_bytes={}", peak_rss_bytes());
}

fn peak_rss_bytes() -> u64 {
    let mut usage = MaybeUninit::<libc::rusage>::zeroed();
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return 0;
    }
    let rss = unsafe { usage.assume_init() }.ru_maxrss as u64;
    #[cfg(target_os = "macos")]
    {
        rss
    }
    #[cfg(not(target_os = "macos"))]
    {
        rss * 1024
    }
}
