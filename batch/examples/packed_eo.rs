use std::{
    convert::Infallible,
    io::{self, Write},
    mem::size_of,
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc,
    },
    thread,
    time::{Duration, Instant},
};

use ark_bls12_381::Fr;
use ark_serialize::{CanonicalSerialize, Compress};
use batch::bucketed_ip::{
    packing_stride_for_bit_width, BvaI8Source, BvaLayout, PackedEoLiteBulletProof,
    TensorBasisVector,
};
use rayon::prelude::*;
use sha2::{Digest, Sha256};

struct ByteBvaSource {
    layout: BvaLayout,
    vectors: Vec<Vec<i8>>,
}

impl BvaI8Source for ByteBvaSource {
    type Error = Infallible;

    fn layout(&self) -> &BvaLayout {
        &self.layout
    }

    fn indexed_vectors(&self) -> Option<Vec<&[i8]>> {
        Some(self.vectors.iter().map(Vec::as_slice).collect())
    }

    fn for_each_vector(&mut self, visit: &mut dyn FnMut(usize, &[i8])) -> Result<(), Self::Error> {
        for (index, vector) in self.vectors.iter().enumerate() {
            visit(index, vector);
        }
        Ok(())
    }
}

struct Arguments {
    max_log: usize,
    allow_non_64_threads: bool,
}

fn parse_arguments() -> Arguments {
    let mut max_log = 20_usize;
    let mut allow_non_64_threads = false;
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
            "--allow-non-64-threads" => allow_non_64_threads = true,
            _ => panic!("unknown argument {argument}"),
        }
    }
    assert!(
        max_log <= 29,
        "geometric workload exceeds the 1G i8 benchmark limit"
    );
    Arguments {
        max_log,
        allow_non_64_threads,
    }
}

fn main() {
    let arguments = parse_arguments();
    let threads = rayon::current_num_threads();
    if !arguments.allow_non_64_threads {
        assert_eq!(threads, 64, "experiments must run with 64 Rayon threads");
    }

    let lengths = (0..=arguments.max_log)
        .rev()
        .map(|log| (1_usize << log) + 1)
        .collect::<Vec<_>>();
    let requested_physical_entries = lengths.iter().sum::<usize>();
    let (stop, peak_rss_kib, monitor) = spawn_memory_monitor(Duration::from_millis(100));

    println!("experiment=packed-eo-litebullet-i8");
    println!("rayon_threads={threads}");
    println!("geometric_max_log={}", arguments.max_log);
    println!("length_formula=2^j+1");
    println!("vector_count={}", lengths.len());
    println!("requested_physical_entries={requested_physical_entries}");
    io::stdout().flush().expect("flush benchmark metadata");

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
    let generation_seconds = generation_started.elapsed().as_secs_f64();
    let physical_entries = vectors.iter().map(Vec::len).sum::<usize>();
    assert_eq!(physical_entries, requested_physical_entries);
    println!(
        "rss_after_generation_kib={}",
        peak_rss_kib.load(Ordering::Relaxed)
    );

    let layout = BvaLayout::from_lengths(&lengths).expect("valid decreasing BVA layout");
    let packing_stride =
        packing_stride_for_bit_width::<Fr>(layout.logical_len(), 8).expect("valid i8 packing");
    let challenges = (0..layout.logical_len().ilog2())
        .map(|round| Fr::from(u64::from(round) + 2))
        .collect::<Vec<_>>();
    let tbv = TensorBasisVector::global_tensor(&layout, &challenges).expect("valid global TBV");
    let mut source = ByteBvaSource { layout, vectors };

    let claim_started = Instant::now();
    let claim = tbv
        .inner_product_i8_streaming(&mut source)
        .expect("streaming i8 inner product");
    let claim_seconds = claim_started.elapsed().as_secs_f64();
    println!(
        "rss_after_claim_kib={}",
        peak_rss_kib.load(Ordering::Relaxed)
    );

    let prove_started = Instant::now();
    let (proof, counters, profile) =
        PackedEoLiteBulletProof::prove_i8_streaming(&mut source, &tbv, claim, 8)
            .expect("packed EO proof");
    let prove_seconds = prove_started.elapsed().as_secs_f64();
    println!(
        "rss_after_prove_kib={}",
        peak_rss_kib.load(Ordering::Relaxed)
    );

    let verify_started = Instant::now();
    let verification = proof.verify_with_tbv(&tbv, claim);
    let verify_seconds = verify_started.elapsed().as_secs_f64();
    assert!(verification.accepted, "packed EO proof verification failed");
    println!(
        "rss_after_verify_kib={}",
        peak_rss_kib.load(Ordering::Relaxed)
    );

    let proof_bytes = proof.to_bytes().expect("serialize packed EO proof");
    let proof_sha256 = Sha256::digest(&proof_bytes);
    let field_storage_bytes = size_of::<Fr>();
    let maximum_live_reduced_field_bytes = profile
        .maximum_live_vector_entries
        .checked_mul(field_storage_bytes)
        .expect("reduced field storage size");
    let simultaneous_witness_and_reduced_bytes = physical_entries
        .checked_add(maximum_live_reduced_field_bytes)
        .expect("simultaneous witness and reduced storage size");

    assert_eq!(size_of::<i8>(), 1);
    assert_eq!(field_storage_bytes, 32);
    assert_eq!(packing_stride, 64.min(source.layout.logical_len()));
    assert_eq!(
        profile.maximum_live_vector_entries,
        2 * (source.layout.logical_len() / packing_stride)
    );
    assert_eq!(counters.materialized_padding_entries, 0);

    stop.store(true, Ordering::Relaxed);
    monitor.join().expect("join memory monitor");

    println!("physical_entries={physical_entries}");
    println!("witness_storage_bytes={physical_entries}");
    println!("active_cover_entries={}", source.layout.active_len());
    println!("logical_entries={}", source.layout.logical_len());
    println!(
        "internal_implicit_zero_entries={}",
        source.layout.active_len() - physical_entries
    );
    println!(
        "final_implicit_zero_entries={}",
        source.layout.logical_len() - source.layout.active_len()
    );
    println!(
        "total_implicit_zero_entries={}",
        source.layout.logical_len() - physical_entries
    );
    println!("packing_stride={packing_stride}");
    println!("field_element_storage_bytes={field_storage_bytes}");
    println!(
        "maximum_live_reduced_field_entries={}",
        profile.maximum_live_vector_entries
    );
    println!(
        "reduced_field_boundary_overhead_entries={}",
        profile.maximum_live_vector_entries - physical_entries.div_ceil(packing_stride)
    );
    println!("maximum_live_reduced_field_bytes={maximum_live_reduced_field_bytes}");
    println!("simultaneous_witness_and_reduced_bytes={simultaneous_witness_and_reduced_bytes}");
    println!("proof_field_elements={}", proof.field_element_count());
    println!("proof_size_bytes={}", proof.serialized_size(Compress::Yes));
    println!("proof_sha256={proof_sha256:x}");
    println!("generation_seconds={generation_seconds:.6}");
    println!("claim_seconds={claim_seconds:.6}");
    println!("prove_seconds={prove_seconds:.6}");
    println!("verify_seconds={verify_seconds:.6}");
    println!("source_passes={}", profile.source_passes);
    println!(
        "materialized_padding_entries={}",
        counters.materialized_padding_entries
    );
    println!("peak_rss_kib={}", peak_rss_kib.load(Ordering::Relaxed));
    println!("accepted=true");
}

fn spawn_memory_monitor(
    sample_period: Duration,
) -> (Arc<AtomicBool>, Arc<AtomicU64>, thread::JoinHandle<()>) {
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let monitor_stop = Arc::clone(&stop);
    let monitor_peak = Arc::clone(&peak);
    let handle = thread::spawn(move || {
        while !monitor_stop.load(Ordering::Relaxed) {
            if let Some(rss_kib) = read_rss_kib() {
                monitor_peak.fetch_max(rss_kib, Ordering::Relaxed);
            }
            thread::sleep(sample_period);
        }
        if let Some(rss_kib) = read_rss_kib() {
            monitor_peak.fetch_max(rss_kib, Ordering::Relaxed);
        }
    });
    (stop, peak, handle)
}

fn read_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    status.lines().find_map(|line| {
        line.strip_prefix("VmRSS:")?
            .split_whitespace()
            .next()?
            .parse()
            .ok()
    })
}
