use std::mem::MaybeUninit;
use std::time::Instant;

use ark_bls12_381::Bls12_381;
use zksmart::hdr1cs::heterogeneous::{HeteroMulProtocol, HeteroMulWitness};
use zksmart::utils::heterogeneous::{HeteroBlockMat, MatrixBlock, MatrixDims, PaddedDomain};

#[global_allocator]
static ALLOCATOR: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[derive(Clone, Copy, Debug)]
struct ExperimentConfig {
    log_input_width: usize,
    materialize: bool,
    prove: bool,
    bit_width: usize,
}

#[derive(Clone, Copy, Debug)]
struct AnalyticalProfile {
    layers: usize,
    weight_parameters: u128,
    bias_parameters: u128,
    stored_witness_entries: u128,
    monolithic_padded_entries: u128,
    bucket_padded_entries: u128,
    max_projection_entries: usize,
    max_pcs_axis: usize,
}

fn main() {
    let config = parse_args();
    let widths = halving_widths(config.log_input_width);
    let profile = analytical_profile(&widths);

    println!("experiment=heterogeneous_halving_mlp");
    println!("equation=Y=ReLU(Wx+b)");
    println!("fixture=sparse_one_hot_i32_weights_unit_inputs_zero_biases_unit_outputs");
    println!("input_width={}", widths[0]);
    println!("output_width={}", widths[widths.len() - 1]);
    println!("layers={}", profile.layers);
    println!("weight_parameters={}", profile.weight_parameters);
    println!("bias_parameters={}", profile.bias_parameters);
    println!(
        "model_parameters={}",
        profile.weight_parameters + profile.bias_parameters
    );
    println!("stored_witness_entries={}", profile.stored_witness_entries);
    println!(
        "monolithic_padded_entries={}",
        profile.monolithic_padded_entries
    );
    println!("bucket_padded_entries={}", profile.bucket_padded_entries);
    println!("max_projection_entries={}", profile.max_projection_entries);
    println!("max_pcs_axis={}", profile.max_pcs_axis);
    println!(
        "monolithic_to_bucket_ratio={:.3}",
        profile.monolithic_padded_entries as f64 / profile.bucket_padded_entries as f64
    );
    println!("peak_rss_before_bytes={}", peak_rss_bytes());

    if !config.materialize {
        println!("mode=analytical_profile_only");
        println!("verified=false");
        return;
    }

    println!(
        "mode={}",
        if config.prove {
            "materialize_prove_verify"
        } else {
            "materialize_only"
        }
    );
    let build_start = Instant::now();
    let witness = build_sparse_relu_witness(&widths);
    let build_seconds = build_start.elapsed().as_secs_f64();
    let runtime_profile = witness.ram_profile();
    println!("build_seconds={build_seconds:.6}");
    println!("peak_rss_after_build_bytes={}", peak_rss_bytes());
    println!(
        "runtime_stored_witness_entries={}",
        runtime_profile.stored_witness_entries
    );
    println!(
        "runtime_monolithic_padded_entries={}",
        runtime_profile.monolithic_padded_entries
    );
    println!(
        "runtime_bucket_padded_entries={}",
        runtime_profile.bucket_padded_entries
    );
    if !config.prove {
        println!("verified=false");
        return;
    }

    println!("proof_scope=heterogeneous_Wx_only; zero_bias_and_ReLU_are_computed_fixture_values");

    let setup_start = Instant::now();
    let srs = HeteroMulProtocol::<Bls12_381>::setup(&witness).unwrap();
    let setup_seconds = setup_start.elapsed().as_secs_f64();
    println!("setup_seconds={setup_seconds:.6}");
    println!("peak_rss_after_setup_bytes={}", peak_rss_bytes());

    let prove_start = Instant::now();
    let (instance, proof) =
        HeteroMulProtocol::<Bls12_381>::prove_owned(&srs, witness, config.bit_width).unwrap();
    let prove_seconds = prove_start.elapsed().as_secs_f64();
    println!("prove_seconds={prove_seconds:.6}");
    println!("proof_buckets={}", proof.buckets.len());
    println!("peak_rss_after_prove_bytes={}", peak_rss_bytes());

    let verify_start = Instant::now();
    let verified = HeteroMulProtocol::<Bls12_381>::verify(&srs, &instance, &proof);
    let verify_seconds = verify_start.elapsed().as_secs_f64();
    println!("verify_seconds={verify_seconds:.6}");
    println!("verified={verified}");
    println!("peak_rss_final_bytes={}", peak_rss_bytes());
    assert!(verified);
}

fn parse_args() -> ExperimentConfig {
    let args: Vec<String> = std::env::args().collect();
    let mut config = ExperimentConfig {
        log_input_width: 15,
        materialize: false,
        prove: false,
        bit_width: 1,
    };
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--bench" => {}
            "--materialize" => config.materialize = true,
            "--prove" => {
                config.materialize = true;
                config.prove = true;
            }
            "--log-input-width" => {
                index += 1;
                config.log_input_width = args[index].parse().unwrap();
            }
            "--bit-width" => {
                index += 1;
                config.bit_width = args[index].parse().unwrap();
            }
            other => panic!("unknown argument: {}", other),
        }
        index += 1;
    }
    assert!((1..=30).contains(&config.log_input_width));
    config
}

fn halving_widths(log_input_width: usize) -> Vec<usize> {
    (0..=log_input_width)
        .map(|step| 1_usize << (log_input_width - step))
        .collect()
}

fn analytical_profile(widths: &[usize]) -> AnalyticalProfile {
    let layers = widths.len() - 1;
    let weight_parameters = widths
        .windows(2)
        .map(|pair| pair[0] as u128 * pair[1] as u128)
        .sum::<u128>();
    let bias_parameters = widths[1..].iter().map(|&width| width as u128).sum();
    let input_entries = widths[..layers]
        .iter()
        .map(|&width| width as u128)
        .sum::<u128>();
    let output_entries = bias_parameters;
    let stored_witness_entries = weight_parameters + input_entries + output_entries;
    let slots = layers.next_power_of_two() as u128;
    let max_input = widths[0] as u128;
    let max_output = widths[1] as u128;
    let monolithic_padded_entries = slots * (max_input * max_output + max_input + max_output);
    let max_weight_entries = widths[0] * widths[1];
    let max_pcs_axis = packed_axes(max_weight_entries)
        .0
        .max(packed_axes(max_weight_entries).1);

    AnalyticalProfile {
        layers,
        weight_parameters,
        bias_parameters,
        stored_witness_entries,
        monolithic_padded_entries,
        bucket_padded_entries: stored_witness_entries,
        max_projection_entries: widths[0],
        max_pcs_axis,
    }
}

fn packed_axes(entries: usize) -> (usize, usize) {
    let cols = 1 << (entries.ilog2() as usize / 2);
    (entries / cols, cols)
}

fn build_sparse_relu_witness(widths: &[usize]) -> HeteroMulWitness {
    let layers = widths.len() - 1;
    let mut weights = Vec::with_capacity(layers);
    let mut inputs = Vec::with_capacity(layers);
    let mut outputs = Vec::with_capacity(layers);

    for (layer, pair) in widths.windows(2).enumerate() {
        let input_width = pair[0];
        let output_width = pair[1];
        eprintln!(
            "materializing layer {}/{}: {}x{} W",
            layer + 1,
            layers,
            output_width,
            input_width,
        );
        weights.push(Some(one_hot_weight_block(output_width, input_width)));
        inputs.push(Some(unit_vector_block(input_width)));
        outputs.push(Some(unit_vector_block(output_width)));
    }

    let slots = layers.next_power_of_two();
    let max_input = widths[0];
    let max_output = widths[1];
    let left = HeteroBlockMat::new(
        PaddedDomain::new(slots, max_output, max_input).unwrap(),
        weights,
    )
    .unwrap();
    let right =
        HeteroBlockMat::new(PaddedDomain::new(slots, max_input, 1).unwrap(), inputs).unwrap();
    let output =
        HeteroBlockMat::new(PaddedDomain::new(slots, max_output, 1).unwrap(), outputs).unwrap();
    HeteroMulWitness::new(left, right, output).unwrap()
}

fn one_hot_weight_block(rows: usize, cols: usize) -> MatrixBlock<i32> {
    let mut columns = Vec::with_capacity(cols);
    columns.push(vec![1_i32; rows]);
    columns.extend((1..cols).map(|_| vec![0_i32; rows]));
    MatrixBlock::from_columns(MatrixDims::new(rows, cols).unwrap(), columns).unwrap()
}

fn unit_vector_block(rows: usize) -> MatrixBlock<i32> {
    MatrixBlock::from_columns(MatrixDims::new(rows, 1).unwrap(), vec![vec![1_i32; rows]]).unwrap()
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
