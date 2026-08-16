use std::mem::MaybeUninit;
use std::time::Instant;

use ark_bls12_381::Bls12_381;
use zksmart::hdr1cs::mlp::{HeteroMlpProtocol, MlpLayer, MlpManifest, MlpRamProfile};
use zksmart::utils::heterogeneous::{MatrixBlock, MatrixDims};

#[global_allocator]
static ALLOCATOR: jemallocator::Jemalloc = jemallocator::Jemalloc;

#[derive(Clone, Copy, Debug)]
struct ExperimentConfig {
    log_input_width: usize,
    prove: bool,
    commitment_bit_width: usize,
    relu_bits: usize,
    dense_balanced: bool,
}

fn main() {
    let config = parse_args();
    let widths = halving_widths(config.log_input_width);
    let manifest = MlpManifest {
        widths: widths.clone(),
        commitment_bit_width: config.commitment_bit_width,
        relu_magnitude_bits: config.relu_bits,
    };
    let profile = MlpRamProfile::for_manifest(&manifest).unwrap();
    let bias_parameters = widths[1..].iter().map(|&width| width as u128).sum::<u128>();

    println!("experiment=end_to_end_heterogeneous_halving_mlp");
    println!("equation=y_i=ReLU(W_i*x_i+b_i)");
    println!(
        "fixture={}",
        if config.dense_balanced {
            "dense_balanced_sign_weights_unit_inputs_unit_biases"
        } else {
            "sparse_first_column_weights_unit_inputs_zero_biases"
        }
    );
    println!("proof_scope=matmul+bias_addition+exact_relu+cross_layer_wiring+public_io");
    println!("input_width={}", widths[0]);
    println!("output_width={}", widths[widths.len() - 1]);
    println!("layers={}", widths.len() - 1);
    println!("weight_parameters={}", profile.total_weight_entries);
    println!("bias_parameters={bias_parameters}");
    println!(
        "model_parameters={}",
        profile.total_weight_entries as u128 + bias_parameters
    );
    println!(
        "max_live_weight_entries={}",
        profile.max_live_weight_entries
    );
    println!(
        "total_activation_entries={}",
        profile.total_activation_entries
    );
    println!("range_bit_entries={}", profile.range_bit_entries);
    println!("commitment_bit_width={}", config.commitment_bit_width);
    println!("relu_magnitude_bits={}", config.relu_bits);
    println!("peak_rss_before_bytes={}", peak_rss_bytes());

    if !config.prove {
        println!("mode=analytical_profile_only");
        println!("verified=false");
        return;
    }

    println!("mode=streaming_prove_verify");
    let setup_start = Instant::now();
    let srs = HeteroMlpProtocol::<Bls12_381>::setup(&manifest).unwrap();
    println!("setup_seconds={:.6}", setup_start.elapsed().as_secs_f64());
    println!("peak_rss_after_setup_bytes={}", peak_rss_bytes());

    let public_input = vec![1_i32; widths[0]];
    let public_output = vec![1_i32; *widths.last().unwrap()];
    let layers = HalvingLayers::new(widths.clone(), config.dense_balanced);
    let prove_start = Instant::now();
    let (instance, proof) =
        HeteroMlpProtocol::<Bls12_381>::prove(&srs, manifest, public_input, public_output, layers)
            .unwrap();
    println!("prove_seconds={:.6}", prove_start.elapsed().as_secs_f64());
    println!("proof_layers={}", proof.layers.len());
    println!("peak_rss_after_prove_bytes={}", peak_rss_bytes());

    let verify_start = Instant::now();
    let verified = HeteroMlpProtocol::<Bls12_381>::verify(&srs, &instance, &proof);
    println!("verify_seconds={:.6}", verify_start.elapsed().as_secs_f64());
    println!("verified={verified}");
    println!("peak_rss_final_bytes={}", peak_rss_bytes());
    assert!(verified);
}

struct HalvingLayers {
    widths: Vec<usize>,
    next_layer: usize,
    dense_balanced: bool,
}

impl HalvingLayers {
    fn new(widths: Vec<usize>, dense_balanced: bool) -> Self {
        Self {
            widths,
            next_layer: 0,
            dense_balanced,
        }
    }
}

impl Iterator for HalvingLayers {
    type Item = MlpLayer;

    fn next(&mut self) -> Option<Self::Item> {
        let pair = self.widths.get(self.next_layer..self.next_layer + 2)?;
        self.next_layer += 1;
        eprintln!(
            "materializing layer {}/{}: {}x{} W",
            self.next_layer,
            self.widths.len() - 1,
            pair[1],
            pair[0],
        );
        let (weight, bias) = if self.dense_balanced {
            (
                dense_balanced_weight_block(pair[1], pair[0]),
                vec![1_i32; pair[1]],
            )
        } else {
            (
                first_column_weight_block(pair[1], pair[0]),
                vec![0_i32; pair[1]],
            )
        };
        Some(MlpLayer { weight, bias })
    }
}

fn first_column_weight_block(rows: usize, cols: usize) -> MatrixBlock<i32> {
    let mut columns = Vec::with_capacity(cols);
    columns.push(vec![1_i32; rows]);
    columns.extend((1..cols).map(|_| vec![0_i32; rows]));
    MatrixBlock::from_columns(MatrixDims::new(rows, cols).unwrap(), columns).unwrap()
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
        log_input_width: 15,
        prove: false,
        commitment_bit_width: 1,
        relu_bits: 1,
        dense_balanced: false,
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
            "--commitment-bit-width" => {
                index += 1;
                config.commitment_bit_width = args[index].parse().unwrap();
            }
            "--relu-bits" => {
                index += 1;
                config.relu_bits = args[index].parse().unwrap();
            }
            "--dense-balanced" => config.dense_balanced = true,
            other => panic!("unknown argument: {}", other),
        }
        index += 1;
    }
    assert!((1..=20).contains(&config.log_input_width));
    config
}

fn halving_widths(log_input_width: usize) -> Vec<usize> {
    (0..=log_input_width)
        .map(|step| 1_usize << (log_input_width - step))
        .collect()
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
