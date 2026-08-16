use std::fs;
use std::mem::{size_of, MaybeUninit};
use std::time::Instant;

use ark_bls12_381::{Bls12_381, Fr};
use rayon::prelude::*;
use zksmart::hdr1cs::global_mlp::normalization::{
    GlobalHdR1csNormalizationProtocol, LookupTag, NormalizationInputs, NormalizationManifest,
    TaggedLookupTable,
};
use zksmart::hdr1cs::gpt2::{Gpt2Model, Gpt2Variant};
use zksmart::protocols::bucketed_ip::{BucketedVector, BvaLayout, TbvBucket, TensorBasisVector};

#[global_allocator]
static ALLOCATOR: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[derive(Clone, Copy, Debug)]
struct ExperimentConfig {
    variant: Gpt2Variant,
    entry_divisor: usize,
    layout_only: bool,
    run_normalization_proof: bool,
    sequence_len: usize,
}

fn main() {
    let config = parse_args();
    let model = Gpt2Model::for_variant(config.variant);
    let specs = model.parameter_tensors().expect("valid GPT-2 inventory");
    let full_lengths = specs
        .iter()
        .map(|tensor| tensor.entries)
        .collect::<Vec<_>>();
    let full_layout = BvaLayout::from_lengths(&full_lengths).expect("valid full BVA layout");
    let full_physical_entries = model
        .physical_parameter_entries()
        .expect("GPT-2 parameter count must fit usize");
    let full_padding_entries = full_layout.logical_len() - full_physical_entries;

    println!("experiment=gpt2_hd_r1cs_bva_eo_parameter_sweep");
    println!("model=gpt2-{}", config.variant.as_str());
    println!("blocks={}", model.blocks);
    println!("hidden_width={}", model.hidden_width);
    println!("attention_heads={}", model.attention_heads);
    println!("head_width={}", model.head_width());
    println!("mlp_width={}", model.mlp_width);
    println!("vocabulary={}", model.vocabulary);
    println!("maximum_positions={}", model.maximum_positions);
    println!("parameter_tensor_count={}", specs.len());
    println!("physical_parameter_entries={full_physical_entries}");
    println!("lm_head_tied_view={}", model.lm_head_is_tied_view());
    println!("lm_head_extra_physical_entries=0");
    println!("full_bva_logical_entries={}", full_layout.logical_len());
    println!("full_bva_padding_entries={full_padding_entries}");
    println!(
        "full_bva_explicit_padding_bytes_avoided={}",
        full_padding_entries as u128 * size_of::<Fr>() as u128
    );
    println!("materialized_padding_entries=0");
    println!("padding_representation=metadata_only_null_bucket");
    println!("folding=EO_bit_reversal_equivalent_to_LR");
    println!("sequence_len={}", config.sequence_len);
    println!(
        "gpt2_attention_softmax_row_count={}",
        model.blocks * model.attention_heads * config.sequence_len
    );
    println!(
        "kernel_scope=parameter_BVA_TBV_inner_product_and_EO_folding_not_full_GPT2_inference_proof"
    );
    println!("parameter_fixture=deterministic_nonzero_field_elements_not_checkpoint_values");
    println!("entry_divisor={}", config.entry_divisor);
    println!("layout_only={}", config.layout_only);
    report_memory("initial");

    if !config.layout_only {
        run_bva_kernel(&specs, config.entry_divisor);
    }
    if config.run_normalization_proof {
        run_normalization_proof();
    }
    report_memory("final");
}

fn run_bva_kernel(specs: &[zksmart::hdr1cs::gpt2::TensorSpec], entry_divisor: usize) {
    let executed_lengths = specs
        .iter()
        .map(|tensor| tensor.entries.div_ceil(entry_divisor))
        .collect::<Vec<_>>();
    let executed_entries = executed_lengths.iter().sum::<usize>();
    println!("kernel_full_shape={}", entry_divisor == 1);
    println!("kernel_physical_entries={executed_entries}");
    println!(
        "kernel_physical_field_payload_bytes={}",
        executed_entries as u128 * size_of::<Fr>() as u128
    );

    let materialize_start = Instant::now();
    let mut vectors = Vec::with_capacity(specs.len());
    for (tensor_index, (&len, tensor)) in executed_lengths.iter().zip(specs).enumerate() {
        eprintln!(
            "materializing tensor {}/{} {} shape={}x{} executed_entries={len}",
            tensor_index + 1,
            specs.len(),
            tensor.name,
            tensor.rows,
            tensor.cols,
        );
        let values = (0..len)
            .into_par_iter()
            .map(|entry| {
                let nonzero = ((tensor_index as u64 * 131 + entry as u64) % 251) + 1;
                Fr::from(nonzero)
            })
            .collect::<Vec<_>>();
        vectors.push(values);
    }
    println!(
        "materialize_true_entries_seconds={:.6}",
        materialize_start.elapsed().as_secs_f64()
    );
    report_memory("after_true_entries");

    let assembly = BucketedVector::new(vectors).expect("valid executed BVA");
    println!(
        "kernel_bva_logical_entries={}",
        assembly.layout().logical_len()
    );
    println!(
        "kernel_bva_padding_entries={}",
        assembly.logical_padding_entries()
    );
    println!(
        "kernel_bva_padding_bucket_count={}",
        assembly.layout().padding_bucket_count()
    );
    println!("kernel_materialized_padding_entries=0");

    let tbv_buckets = assembly
        .layout()
        .bucket_sizes()
        .iter()
        .enumerate()
        .map(|(bucket_index, &bucket_size)| {
            let eta = Fr::from((bucket_index as u64 % 251) + 1);
            let challenges = (0..bucket_size.ilog2() as usize)
                .map(|round| Fr::from(((bucket_index + round) as u64 % 251) + 2))
                .collect();
            TbvBucket::new(eta, challenges)
        })
        .collect();
    let tbv = TensorBasisVector::new(assembly.layout(), tbv_buckets).expect("matching TBV");

    let ip_start = Instant::now();
    let ip = assembly.inner_product(&tbv).expect("BVA/TBV IP succeeds");
    println!("bva_tbv_ip_seconds={:.6}", ip_start.elapsed().as_secs_f64());
    println!("bva_tbv_ip_checksum={}", ip.value);
    println!(
        "ip_materialized_padding_entries={}",
        ip.counters.materialized_padding_entries
    );
    println!("padding_ip_field_ops={}", ip.counters.padding_ip_field_ops);
    println!(
        "padding_ip_fold_field_ops={}",
        ip.counters.padding_fold_field_ops
    );
    println!("real_ip_field_ops={}", ip.counters.real_ip_field_ops);
    println!("real_ip_fold_field_ops={}", ip.counters.real_fold_field_ops);
    assert_eq!(ip.counters.materialized_padding_entries, 0);
    assert_eq!(ip.counters.padding_ip_field_ops, 0);
    assert_eq!(ip.counters.padding_fold_field_ops, 0);
    report_memory("after_ip");

    let fold_challenges = (0..assembly.layout().logical_len().ilog2() as usize)
        .map(|round| Fr::from((round as u64 % 251) + 2))
        .collect::<Vec<_>>();
    let fold_start = Instant::now();
    let folded = assembly
        .fold_to_scalar(&fold_challenges)
        .expect("bucket-wise EO folding succeeds");
    println!("eo_fold_seconds={:.6}", fold_start.elapsed().as_secs_f64());
    println!("eo_fold_checksum={}", folded.value);
    println!(
        "eo_fold_materialized_padding_entries={}",
        folded.counters.materialized_padding_entries
    );
    println!(
        "padding_fold_field_ops={}",
        folded.counters.padding_fold_field_ops
    );
    println!(
        "real_fold_field_ops={}",
        folded.counters.real_fold_field_ops
    );
    assert_eq!(folded.counters.materialized_padding_entries, 0);
    assert_eq!(folded.counters.padding_ip_field_ops, 0);
    assert_eq!(folded.counters.padding_fold_field_ops, 0);
    report_memory("after_eo_fold");
}

fn run_normalization_proof() {
    let manifest = NormalizationManifest {
        width: 4,
        epsilon: 1,
        output_shift: 3,
        softmax_output_shift: 4,
        commitment_bit_width: 16,
        tables: vec![
            TaggedLookupTable::new(LookupTag::Exp, vec![(-3, 1), (-2, 2), (-1, 4), (0, 8)]),
            TaggedLookupTable::new(LookupTag::Reciprocal, vec![(15, 16)]),
            TaggedLookupTable::new(LookupTag::InverseSqrt, vec![(81, 8)]),
            TaggedLookupTable::new(
                LookupTag::NonNegative,
                (0..=15).map(|value| (value, value)).collect(),
            ),
            TaggedLookupTable::new(LookupTag::Boolean, vec![(0, 0), (1, 1)]),
        ],
    };
    let inputs = NormalizationInputs {
        layer_norm_input: vec![1, 2, 3, 4],
        layer_norm_gamma: vec![1, 1, 1, 1],
        layer_norm_beta: vec![0, 0, 0, 0],
        softmax_scores: vec![0, 1, 2, 3],
    };
    let encode_start = Instant::now();
    let encoding = GlobalHdR1csNormalizationProtocol::<Bls12_381>::encode(manifest, inputs)
        .expect("normalization encoding succeeds");
    println!("softmax_slice_scope=one_operator_row_with_global_operation_type_equations");
    println!("softmax_fixed_point_shift=4");
    println!("softmax_fixed_point_unit=16");
    println!("softmax_output={:?}", encoding.softmax_output());
    println!(
        "softmax_output_row_sum={}",
        encoding.softmax_output().iter().sum::<i32>()
    );
    println!(
        "normalization_encode_seconds={:.6}",
        encode_start.elapsed().as_secs_f64()
    );

    let setup_start = Instant::now();
    let srs = GlobalHdR1csNormalizationProtocol::<Bls12_381>::setup(&encoding)
        .expect("normalization setup succeeds");
    println!(
        "normalization_setup_seconds={:.6}",
        setup_start.elapsed().as_secs_f64()
    );
    let prove_start = Instant::now();
    let (instance, proof) = GlobalHdR1csNormalizationProtocol::<Bls12_381>::prove(&srs, encoding)
        .expect("normalization proof succeeds");
    println!(
        "normalization_prove_seconds={:.6}",
        prove_start.elapsed().as_secs_f64()
    );
    println!(
        "normalization_proof_size_bytes={}",
        proof.serialized_size_bytes()
    );
    println!(
        "normalization_witness_commitment_count={}",
        proof.witness_commitment_count()
    );
    println!(
        "normalization_witness_opening_count={}",
        proof.witness_opening_count()
    );
    let verify_start = Instant::now();
    let verified = GlobalHdR1csNormalizationProtocol::<Bls12_381>::verify(&srs, &instance, &proof);
    println!(
        "normalization_verify_seconds={:.6}",
        verify_start.elapsed().as_secs_f64()
    );
    println!("normalization_verified={verified}");
    assert!(verified);
}

fn parse_args() -> ExperimentConfig {
    let args = std::env::args().collect::<Vec<_>>();
    let mut config = ExperimentConfig {
        variant: Gpt2Variant::Small,
        entry_divisor: 4_096,
        layout_only: false,
        run_normalization_proof: true,
        sequence_len: 4,
    };
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--bench" => {}
            "--full" => config.entry_divisor = 1,
            "--layout-only" => config.layout_only = true,
            "--skip-normalization-proof" => config.run_normalization_proof = false,
            "--model" => {
                index += 1;
                config.variant = args[index].parse().expect("valid GPT-2 model variant");
            }
            "--entry-divisor" => {
                index += 1;
                config.entry_divisor = args[index].parse().expect("valid entry divisor");
            }
            "--sequence-len" => {
                index += 1;
                config.sequence_len = args[index].parse().expect("valid sequence length");
            }
            other => panic!("unknown argument: {}", other),
        }
        index += 1;
    }
    assert!(config.entry_divisor > 0);
    assert!((1..=1_024).contains(&config.sequence_len));
    config
}

fn report_memory(phase: &str) {
    println!("current_rss_{phase}_bytes={}", current_rss_bytes());
    println!("peak_rss_{phase}_bytes={}", peak_rss_bytes());
}

fn current_rss_bytes() -> u64 {
    let Ok(status) = fs::read_to_string("/proc/self/status") else {
        return 0;
    };
    status
        .lines()
        .find_map(|line| {
            line.strip_prefix("VmRSS:")
                .and_then(|value| value.split_whitespace().next())
                .and_then(|value| value.parse::<u64>().ok())
                .map(|kilobytes| kilobytes * 1024)
        })
        .unwrap_or(0)
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
        rss * 1_024
    }
}
