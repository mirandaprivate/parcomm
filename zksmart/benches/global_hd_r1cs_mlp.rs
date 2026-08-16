use std::fs;
use std::mem::MaybeUninit;
use std::time::Instant;

use ark_bls12_381::Bls12_381;
use zksmart::hdr1cs::global_mlp::GlobalHdR1csMlpProtocol;
use zksmart::hdr1cs::mlp::{MlpLayer, MlpManifest};
use zksmart::utils::heterogeneous::{MatrixBlock, MatrixDims};

#[global_allocator]
static ALLOCATOR: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[derive(Clone, Copy, Debug)]
struct ExperimentConfig {
    log_input_width: usize,
    prove: bool,
}

fn main() {
    let config = parse_args();
    let widths = halving_widths(config.log_input_width);
    let weight_parameters = widths
        .windows(2)
        .map(|pair| pair[0] as u128 * pair[1] as u128)
        .sum::<u128>();
    let bias_parameters = widths[1..].iter().map(|&width| width as u128).sum::<u128>();
    let model_parameters = weight_parameters + bias_parameters;

    println!("experiment=global_hd_r1cs_heterogeneous_halving_mlp");
    println!("equation=y_i=ReLU(W_i*x_i+b_i)");
    println!("sigma_semantics=exact_relu");
    println!("fixture=dense_balanced_sign_weights_unit_input_zero_biases");
    println!(
        "proof_scope=one_global_instance+one_global_Cw+io_equation+per_operation_type_equations"
    );
    println!("witness_opening_scope=all_Cw_claims_batched_to_one_opening");
    println!("proof_organization=global_not_per_layer");
    println!("parameter_binding=offline_io_commitment_and_global_witness_commitment");
    println!("input_width={}", widths[0]);
    println!("output_width={}", widths[widths.len() - 1]);
    println!("layers={}", widths.len() - 1);
    println!("weight_parameters={weight_parameters}");
    println!("bias_parameters={bias_parameters}");
    println!("model_parameters={model_parameters}");
    println!("commitment_bit_width=1");
    println!("relu_magnitude_bits=1");
    report_memory("before");

    if !config.prove {
        println!("mode=analytical_profile_only");
        println!("verified=false");
        return;
    }

    let manifest = MlpManifest {
        widths: widths.clone(),
        commitment_bit_width: 1,
        relu_magnitude_bits: 1,
    };
    let public_input = vec![1_i32; widths[0]];
    let public_output = vec![0_i32; *widths.last().unwrap()];
    let encode_start = Instant::now();
    let encoding = GlobalHdR1csMlpProtocol::<Bls12_381>::encode(
        manifest,
        public_input,
        public_output,
        HalvingLayers::new(widths.clone()),
    )
    .unwrap();
    println!("encode_seconds={:.6}", encode_start.elapsed().as_secs_f64());
    let shape = encoding.global_shape();
    println!("global_block_domain_n={}", shape.n);
    println!("global_witness_slots={}", shape.witness_slots);
    println!("global_gate_slots={}", shape.gate_slots);
    println!(
        "logical_witness_entries={}",
        shape.witness_slots as u128 * shape.n as u128 * shape.n as u128
    );
    println!(
        "physically_stored_witness_entries={}",
        encoding.stored_witness_entries()
    );
    report_memory("after_encode");

    let setup_start = Instant::now();
    let srs = GlobalHdR1csMlpProtocol::<Bls12_381>::setup(&encoding).unwrap();
    println!("setup_seconds={:.6}", setup_start.elapsed().as_secs_f64());
    report_memory("after_setup");

    let prove_start = Instant::now();
    let (instance, proof) = GlobalHdR1csMlpProtocol::<Bls12_381>::prove(&srs, encoding).unwrap();
    println!("prove_seconds={:.6}", prove_start.elapsed().as_secs_f64());
    println!(
        "witness_commitment_count={}",
        proof.witness_commitment_count()
    );
    println!("witness_opening_count={}", proof.witness_opening_count());
    println!(
        "operation_equation_count={}",
        proof.operation_equation_count()
    );
    println!("proof_size_bytes={}", proof.serialized_size_bytes());
    println!("instance_n={}", instance.shape.n);
    report_memory("after_prove");

    let verify_start = Instant::now();
    let verified = GlobalHdR1csMlpProtocol::<Bls12_381>::verify(&srs, &instance, &proof);
    println!("verify_seconds={:.6}", verify_start.elapsed().as_secs_f64());
    println!("verified={verified}");
    report_memory("final");
    assert!(verified);
}

struct HalvingLayers {
    widths: Vec<usize>,
    next_layer: usize,
}

impl HalvingLayers {
    fn new(widths: Vec<usize>) -> Self {
        Self {
            widths,
            next_layer: 0,
        }
    }
}

impl Iterator for HalvingLayers {
    type Item = MlpLayer;

    fn next(&mut self) -> Option<Self::Item> {
        let pair = self.widths.get(self.next_layer..self.next_layer + 2)?;
        self.next_layer += 1;
        eprintln!(
            "materializing layer {}/{}: {}x{} dense W",
            self.next_layer,
            self.widths.len() - 1,
            pair[1],
            pair[0],
        );
        Some(MlpLayer {
            weight: dense_balanced_weight_block(pair[1], pair[0]),
            bias: vec![0_i32; pair[1]],
        })
    }
}

fn dense_balanced_weight_block(rows: usize, cols: usize) -> MatrixBlock<i32> {
    let columns = (0..cols)
        .map(|column| vec![if column % 2 == 0 { 1_i32 } else { -1_i32 }; rows])
        .collect();
    MatrixBlock::from_columns(MatrixDims::new(rows, cols).unwrap(), columns).unwrap()
}

fn parse_args() -> ExperimentConfig {
    let args = std::env::args().collect::<Vec<_>>();
    let mut config = ExperimentConfig {
        log_input_width: 5,
        prove: false,
    };
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--bench" => {}
            "--prove" => config.prove = true,
            "--log-input-width" => {
                index += 1;
                config.log_input_width = args[index].parse().unwrap();
            }
            other => panic!("unknown argument: {}", other),
        }
        index += 1;
    }
    assert!((1..=15).contains(&config.log_input_width));
    config
}

fn halving_widths(log_input_width: usize) -> Vec<usize> {
    (0..=log_input_width)
        .map(|step| 1_usize << (log_input_width - step))
        .collect()
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
        rss * 1024
    }
}
