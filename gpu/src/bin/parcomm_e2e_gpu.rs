//! End-to-end ParComm experiment with the witness-proportional stages on GPU.
//!
//! The packed witness is uploaded once and stays in device memory; the tier-one
//! commitment MSMs and both opening projections read it from there. Everything
//! whose cost only depends on the packed side lengths runs on the CPU.
//!
//! ```text
//! cargo run --release --bin parcomm_e2e_gpu -- --geometric-max-log 31 --pp-log 18
//! ```

use std::{
    convert::Infallible,
    fs,
    io::{self, Write},
    path::PathBuf,
    time::Instant,
};

use ark_bls12_381::{Bls12_381, Fr};
use ark_ec::pairing::{Pairing, PairingOutput};
use ark_ec::CurveGroup;
use ark_ff::UniformRand;
use ark_std::{ops::Mul, Zero};
use batch::{
    bucketed_ip::{
        packing_stride_for_bit_width, BvaI8Source, BvaLayout, PackedEoLiteBulletProof,
        TensorBasisVector,
    },
    virtualpcs::{
        bva_eo_opening_point, commit_bva_i8_streaming, default_parameter_cache_path,
        deserialize_public_parameters_exact, open_from_left_projection, packed_bva_shape,
        packed_i8_parcomm_statement_binding, serialize_public_parameters, verify_packed_i8_parcomm,
        OpenProfile, OpeningColumnVectors, PackedI8ParCommProof,
    },
};
use parcomm_gpu::{
    dense_packed_witness, smart_pc::GpuSmartPcError, GpuPairingVectors, GpuReducedVectors,
    GpuWitness, PACKING_STRIDE,
};
use poly_commit::smart_pc::{data_structures::UniversalParams, xi::xi_from_challenges, SmartPC};
use rayon::prelude::*;

const BIT_WIDTH: usize = 8;

/// Presents the device-resident column vectors to the opening reduction.
struct GpuColumnVectors {
    inner: GpuPairingVectors,
}

impl GpuColumnVectors {
    fn new(inner: GpuPairingVectors) -> Self {
        Self { inner }
    }
}

impl OpeningColumnVectors<Bls12_381> for GpuColumnVectors {
    type Error = GpuSmartPcError;

    fn cross_pairings(
        &mut self,
    ) -> Result<(PairingOutput<Bls12_381>, PairingOutput<Bls12_381>), Self::Error> {
        self.inner.cross_pairings()
    }

    fn fold(&mut self, tier_one: Fr, basis: Fr) -> Result<(), Self::Error> {
        self.inner.fold(tier_one, basis)
    }

    fn reduced_basis(&self) -> Result<<Bls12_381 as Pairing>::G2, Self::Error> {
        Ok(self.inner.reduced_basis()?.into())
    }
}

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
        for (vector_index, vector) in self.vectors.iter().enumerate() {
            visit(vector_index, vector);
        }
        Ok(())
    }
}

struct Arguments {
    max_log: usize,
    pp_log: Option<usize>,
    devices: Vec<i32>,
    check_cpu: bool,
}

fn parse_arguments() -> Arguments {
    let mut max_log = 12_usize;
    let mut pp_log = None;
    let mut devices = Vec::new();
    let mut check_cpu = false;
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--geometric-max-log" => {
                max_log = arguments
                    .next()
                    .expect("missing --geometric-max-log value")
                    .parse()
                    .expect("invalid --geometric-max-log value");
            }
            "--pp-log" => {
                pp_log = Some(
                    arguments
                        .next()
                        .expect("missing --pp-log value")
                        .parse()
                        .expect("invalid --pp-log value"),
                );
            }
            "--devices" => {
                devices = arguments
                    .next()
                    .expect("missing --devices value")
                    .split(',')
                    .map(|entry| entry.trim().parse().expect("invalid device id"))
                    .collect();
            }
            "--check-cpu" => check_cpu = true,
            _ => panic!("unknown argument {argument}"),
        }
    }
    if devices.is_empty() {
        devices = vec![0, 1, 2, 3];
    }
    assert!(
        (3..=35).contains(&max_log),
        "geometric max log must be between 3 and 35"
    );
    Arguments {
        max_log,
        pp_log,
        devices,
        check_cpu,
    }
}

fn load_or_setup_parameters(
    path: &PathBuf,
    requested_qlog: usize,
) -> (UniversalParams<Bls12_381>, &'static str) {
    if path.exists() {
        let bytes = fs::read(path).expect("read SmartPC parameter cache");
        if let Ok(parameters) = deserialize_public_parameters_exact::<Bls12_381>(&bytes) {
            return (parameters, "cache");
        }
        eprintln!("pp_cache_rejected path={}", path.display());
    }
    let mut rng = ark_std::rand::thread_rng();
    let parameters = SmartPC::<Bls12_381>::setup(requested_qlog, &mut rng)
        .expect("generate SmartPC public parameters");
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent).expect("create SmartPC parameter directory");
    }
    let bytes =
        serialize_public_parameters(&parameters).expect("serialize SmartPC public parameters");
    fs::write(path, bytes).expect("write SmartPC parameter cache");
    (parameters, "generated")
}

fn main() -> Result<(), GpuSmartPcError> {
    let arguments = parse_arguments();
    parcomm_gpu::initialize_cuda_backend()?;

    let lengths = (0..=arguments.max_log)
        .rev()
        .map(|log_size| (1_usize << log_size) + 1)
        .collect::<Vec<_>>();
    let layout = BvaLayout::from_lengths(&lengths).expect("valid decreasing BVA layout");
    let physical_entries = lengths.iter().sum::<usize>();
    let logical_entries = layout.logical_len();
    let log_k = logical_entries.ilog2() as usize;
    let requested_qlog = arguments.pp_log.unwrap_or(log_k.div_ceil(2));

    println!("experiment=parcomm-packed-i8-end-to-end-gpu");
    println!("rayon_threads={}", rayon::current_num_threads());
    println!("cuda_devices={:?}", arguments.devices);
    println!("geometric_max_log={}", arguments.max_log);
    println!("physical_entries={physical_entries}");
    println!("logical_entries={logical_entries}");
    println!("requested_pp_log={requested_qlog}");
    io::stdout().flush().expect("flush metadata");

    let setup_started = Instant::now();
    let (pp, pp_source) = load_or_setup_parameters(
        &default_parameter_cache_path(requested_qlog),
        requested_qlog,
    );
    println!("pp_source={pp_source}");
    println!(
        "setup_or_load_seconds={:.6}",
        setup_started.elapsed().as_secs_f64()
    );

    let generation_started = Instant::now();
    let vectors = lengths
        .par_iter()
        .enumerate()
        .map(|(vector_index, &length)| {
            let magnitude = (vector_index % 31 + 1) as i8;
            let value = if vector_index & 1 == 0 {
                magnitude
            } else {
                -magnitude
            };
            vec![value; length]
        })
        .collect::<Vec<_>>();
    println!(
        "generation_seconds={:.6}",
        generation_started.elapsed().as_secs_f64()
    );
    let mut source = ByteBvaSource { layout, vectors };

    // The device kernels index a dense row-major matrix, so the bucketed
    // vectors are laid out once and then never touched by the host again.
    let dense_started = Instant::now();
    let dense = dense_packed_witness(&mut source)?;
    let (rows, cols) = (dense.layout.rows, dense.layout.cols);
    let stored_rows = dense.layout.live_rows.len();
    println!(
        "dense_seconds={:.6}",
        dense_started.elapsed().as_secs_f64()
    );
    assert_eq!((rows, cols), packed_bva_shape(logical_entries).unwrap());
    println!("stored_rows={stored_rows} of {rows}");

    let upload_started = Instant::now();
    let mut gpu = GpuWitness::upload(&pp, dense, &arguments.devices)?;
    println!(
        "gpu_upload_seconds={:.6}",
        upload_started.elapsed().as_secs_f64()
    );
    println!("packed_rows={rows} packed_cols={cols}");

    let challenges = (0..log_k)
        .map(|round| Fr::from(u64::try_from(round).expect("challenge index fits u64") + 2))
        .collect::<Vec<_>>();
    let tbv = TensorBasisVector::global_tensor(source.layout(), &challenges)
        .expect("valid global tensor basis vector");
    let claim_started = Instant::now();
    let claimed = gpu.global_tensor_claim(&challenges)?;
    println!(
        "claim_seconds={:.6}",
        claim_started.elapsed().as_secs_f64()
    );

    if arguments.check_cpu {
        let cpu_started = Instant::now();
        let cpu_claimed = tbv
            .inner_product_i8_streaming(&mut source)
            .expect("compute streaming i8 claim");
        println!(
            "cpu_claim_seconds={:.6}",
            cpu_started.elapsed().as_secs_f64()
        );
        println!("claim_matches_cpu={}", cpu_claimed == claimed);
        assert_eq!(cpu_claimed, claimed, "GPU claim disagrees with the CPU prover");
    }

    let packing_stride =
        packing_stride_for_bit_width::<Fr>(logical_entries, BIT_WIDTH).expect("valid i8 packing");
    assert_eq!(packing_stride, 32);

    let mut prover_rng = ark_std::rand::thread_rng();
    let matrix_hiding = Fr::rand(&mut prover_rng);
    let commit_started = Instant::now();
    let (matrix_commitment, tier_one) = gpu.commit(pp.tilde_u, matrix_hiding)?;
    let commit_seconds = commit_started.elapsed().as_secs_f64();
    println!("commit_seconds={commit_seconds:.6}");
    println!(
        "commit_msm_seconds={:.6} commit_pairing_seconds={:.6}",
        gpu.profile.commit_msm_seconds, gpu.profile.commit_pairing_seconds
    );

    if arguments.check_cpu {
        let cpu_started = Instant::now();
        let (cpu_commitment, cpu_tier_one, _) =
            commit_bva_i8_streaming::<Bls12_381, _>(&pp, &mut source, matrix_hiding, BIT_WIDTH)
                .expect("streaming SmartPC commitment");
        println!(
            "cpu_commit_seconds={:.6}",
            cpu_started.elapsed().as_secs_f64()
        );
        assert_eq!(
            cpu_tier_one.len(),
            tier_one.len(),
            "tier-one length mismatch"
        );
        let mismatches = cpu_tier_one
            .par_iter()
            .zip(tier_one.par_iter())
            .filter(|(cpu, gpu)| cpu != gpu)
            .count();
        println!("tier_one_mismatches={mismatches}");
        println!(
            "commitment_matches_cpu={}",
            cpu_commitment == matrix_commitment
        );
        assert_eq!(mismatches, 0, "GPU tier-one disagrees with the CPU prover");
        assert_eq!(cpu_commitment, matrix_commitment);
    }

    let binding = packed_i8_parcomm_statement_binding::<Bls12_381>(
        &matrix_commitment,
        &tbv,
        claimed,
        BIT_WIDTH,
    );
    let packed_eo_started = Instant::now();
    let lane_requests = tbv.packed_lane_requests(PACKING_STRIDE);
    let lane_projections = gpu.packed_lane_projections(&lane_requests)?;
    println!(
        "cross_terms_gpu_seconds={:.6}",
        gpu.profile.cross_terms_seconds
    );
    let mut reduced_vectors = GpuReducedVectors::new(&gpu);
    let (packed_eo, _padding, packed_eo_profile) =
        PackedEoLiteBulletProof::prove_i8_streaming_bound_accelerated(
            &mut source,
            &tbv,
            claimed,
            BIT_WIDTH,
            binding,
            &lane_projections,
            &mut reduced_vectors,
        )
        .expect("commitment-bound packed EO-LiteBullet proof");
    println!(
        "reduced_build_seconds={:.6} reduced_round_seconds={:.6}",
        reduced_vectors.build_seconds, reduced_vectors.round_seconds
    );
    println!(
        "packed_eo_seconds={:.6}",
        packed_eo_started.elapsed().as_secs_f64()
    );
    let packed_eo_timings = packed_eo_profile.timings;
    println!(
        "packed_eo_cross_terms_seconds={:.6} packed_eo_fold_seconds={:.6} \
         packed_eo_reduced_check_seconds={:.6} packed_eo_binary_rounds_seconds={:.6}",
        packed_eo_timings.cross_terms_seconds,
        packed_eo_timings.fold_seconds,
        packed_eo_timings.reduced_check_seconds,
        packed_eo_timings.binary_rounds_seconds,
    );
    if arguments.check_cpu {
        let cpu_started = Instant::now();
        let (cpu_packed_eo, _, cpu_profile) = PackedEoLiteBulletProof::prove_i8_streaming_bound(
            &mut source,
            &tbv,
            claimed,
            BIT_WIDTH,
            binding,
        )
        .expect("CPU packed EO-LiteBullet proof");
        println!(
            "cpu_packed_eo_seconds={:.6} cpu_cross_terms_seconds={:.6}",
            cpu_started.elapsed().as_secs_f64(),
            cpu_profile.timings.cross_terms_seconds,
        );
        let gpu_bytes = packed_eo.to_bytes().expect("serialize GPU packed EO proof");
        let cpu_bytes = cpu_packed_eo
            .to_bytes()
            .expect("serialize CPU packed EO proof");
        println!("packed_eo_matches_cpu={}", gpu_bytes == cpu_bytes);
        assert_eq!(
            gpu_bytes, cpu_bytes,
            "GPU cross terms produced a different packed EO proof"
        );
    }
    let packed_eo_verification = packed_eo.verify_with_tbv_bound(&tbv, claimed, binding);
    assert!(packed_eo_verification.accepted);

    let (xl, xr) = bva_eo_opening_point(logical_entries, &packed_eo_verification.challenges_inv)
        .expect("map packed EO challenges to SmartPC point");
    let value_commitment = Bls12_381::pairing(pp.g_0.mul(packed_eo.a_reduced), pp.h_hat);

    let mut open_profile = OpenProfile::default();
    // The column rounds pair halves of the tier-one vector, so the largest call
    // is half the column count.
    let mut column_vectors = GpuColumnVectors::new(
        GpuPairingVectors::new(
            0,
            &<Bls12_381 as Pairing>::G1::normalize_batch(&tier_one[..cols]),
            &<Bls12_381 as Pairing>::G2::normalize_batch(&pp.vec_h[..cols]),
        )
        .expect("upload opening column vectors"),
    );
    let opening_started = Instant::now();
    let left_weights = xi_from_challenges::<Bls12_381>(&xl.to_vec());
    let value_current = gpu.project_left(&left_weights)?;
    let opening = open_from_left_projection::<Bls12_381, GpuSmartPcError>(
        &pp,
        rows,
        cols,
        &xl,
        &xr,
        value_commitment,
        matrix_commitment,
        Fr::zero(),
        matrix_hiding,
        value_current,
        |challenges| {
            let weights = xi_from_challenges::<Bls12_381>(&challenges.to_vec());
            gpu.project_right(&weights)
        },
        &mut column_vectors,
        &mut open_profile,
    )?;
    let opening_seconds = opening_started.elapsed().as_secs_f64();
    println!("opening_seconds={opening_seconds:.6}");
    println!(
        "project_left_seconds={:.6} project_right_seconds={:.6}",
        gpu.profile.project_left_seconds, gpu.profile.project_right_seconds
    );
    println!(
        "open_pairing_seconds={:.6} open_column_fold_seconds={:.6} \
         open_projection_seconds={:.6} open_row_msm_seconds={:.6} open_row_fold_seconds={:.6} \
         open_tail_seconds={:.6}",
        open_profile.pairing_seconds,
        open_profile.column_fold_seconds,
        open_profile.projection_seconds,
        open_profile.row_msm_seconds,
        open_profile.row_fold_seconds,
        open_profile.tail_seconds,
    );

    let proof = PackedI8ParCommProof {
        matrix_commitment,
        value_commitment,
        packed_eo,
        opening,
    };
    let proof_bytes = proof.to_bytes().expect("serialize complete ParComm proof");
    let decoded = PackedI8ParCommProof::<Bls12_381>::from_bytes_exact(&proof_bytes)
        .expect("decode exactly one complete ParComm proof");
    assert_eq!(decoded, proof);
    println!("proof_size_bytes={}", proof_bytes.len());

    let verify_started = Instant::now();
    let accepted = verify_packed_i8_parcomm(&pp, &tbv, claimed, BIT_WIDTH, &decoded)
        .expect("verify decoded ParComm proof");
    println!(
        "verify_seconds={:.6}",
        verify_started.elapsed().as_secs_f64()
    );
    println!("accepted={accepted}");
    assert!(accepted, "GPU-produced proof was rejected");
    Ok(())
}
