use std::{
    io::{self, Write},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
    thread,
    time::{Duration, Instant},
    env,
};

use ark_bls12_381::{Bls12_381, Fr};
use batch::batchpcs::{BatchPCS, BatchPCSTrans};
use batch::test_data::{
    generate_deterministic_vecs_decreasing_sizes,
    generate_deterministic_vecs_uniform_sizes,
    generate_random_challenges,
};
use batch::utils::eo_fold_iterative_myint_to_field;
use batch::MyInt;


const LOG_TOTAL_MIN: usize = 20;
const LOG_TOTAL_MAX: usize = 30;
const LOG_NUM_VEC_MIN: usize = 10;
const LOG_NUM_VEC_MAX: usize = 10;
const HETEROGENEOUS: bool = false;

fn main() {


    let rayon_env = env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "not set".to_string());
    println!(
        "RAYON_NUM_THREADS env: {}, effective threads: {}",
        rayon_env,
        rayon::current_num_threads()
    );

    println!("========================================================");

    for log_num_vec in (LOG_NUM_VEC_MIN..=LOG_NUM_VEC_MAX).step_by(2) {
        for log_n in (LOG_TOTAL_MIN..=LOG_TOTAL_MAX).step_by(2) {

            let log_single_vec_size = log_n - log_num_vec;

            println!("========================================================\n\n");
            println!(" *******************************************************\n\n");
            println!("========================================================\n\n");
            println!(
                "Running batch PCS with 2^{} scalar entries, heterogeneous: {}, single vec size for uniform: 2^{}",
                log_n,
                HETEROGENEOUS,
                if HETEROGENEOUS { 0} else { log_single_vec_size }
            );

            println!("----------------------------------------------------------");

            let (_stop, peak_kb, monitor) = spawn_mem_monitor(Duration::from_millis(200));
            let report_peak = |label: &str| {
                println!("⬜ Peak RSS after {}: {} KB", label, peak_kb.load(Ordering::Relaxed));
            };

            println!("Setting up BatchPCS with log_n = {}", log_n);
            let setup_timer = Instant::now();
            let mut batch_pcs = BatchPCS::<Bls12_381>::setup(log_n);
            println!("🕒 \x1b[1m Setup time: {:.3}s \x1b[0m", setup_timer.elapsed().as_secs_f64());
            report_peak("setup");

            // 1. Generate random data
            println!("Generating random vectors...");
            let gen_timer = Instant::now();

            let vecs: Vec<Vec<MyInt>>;
            if HETEROGENEOUS {
                vecs = generate_deterministic_vecs_decreasing_sizes(log_n - 1);
            } else {
                vecs = generate_deterministic_vecs_uniform_sizes(log_single_vec_size, log_n);
            }
            let num_vecs = vecs.len();
            println!("num_vecs = {}", num_vecs);


            println!("🕒 \x1b[1m Data generation time: {:.3}ms \x1b[0m", gen_timer.elapsed().as_secs_f64() * 1000.0);
            report_peak("data generation");


            println!("Evaluating...");
            let timer_eval = Instant::now();
            let mut points = Vec::with_capacity(num_vecs);
            let mut values = Vec::with_capacity(num_vecs);

            let results: Vec<(Vec<Fr>, Fr)> = (0..num_vecs).into_iter().map(|i| {
                let log_k = vecs[i].len().ilog2() as usize;
                let point = generate_random_challenges::<Fr>(log_k);

                // Ground truth via iterative fold (avoids allocating xi vector of size 2^log_k)
                let val = eo_fold_iterative_myint_to_field::<Fr>(&vecs[i], point.clone());

                (point, val)
            }).collect();

            for (point, val) in results {
                points.push(point);
                values.push(val);
            }
            let time_eval = timer_eval.elapsed();
            println!("🕒 \x1b[1m Evaluation time: {:.3}s \x1b[0m", time_eval.as_secs_f64());

            batch_pcs.set_prover_input(vecs);
            println!("🕒 \x1b[1m Data generation + Evaluation time: {:.3}s \x1b[0m", gen_timer.elapsed().as_secs_f64());
            report_peak("data generation");

            // 2. Commit
            println!("Committing...");
            let commit_timer = Instant::now();
            let (a_com, i_com, a_tier1, i_tier1) = batch_pcs.commit();
            println!("🕒 \x1b[1m Commit time: {:.3}s \x1b[0m", commit_timer.elapsed().as_secs_f64());
            report_peak("commit");

            let (z_com, z_tier1) = batch_pcs.commit_points(&points);

            // 3. Open (reduction + PC)
            println!("Prover: Opening (reduction + PC)...");
            let open_timer = Instant::now();
            let mut trans_prover = BatchPCSTrans::<Bls12_381>::new();
            batch_pcs.open(
                &mut trans_prover,
                &values,
                &points,
                a_com,
                i_com,
                z_com,
                a_tier1,
                i_tier1,
                z_tier1,
            );
            println!("🕒 \x1b[1m Open time: {:.3}s \x1b[0m", open_timer.elapsed().as_secs_f64());
            report_peak("open");

            let proof_size = trans_prover.get_proof_size();
            println!("⬜ \x1b[1m Proof size: {} bytes \x1b[0m", proof_size);

            // 4. Verify
            println!("Verifier: Verifying full flow...");
            let verify_timer = Instant::now();
            let mut trans_verifier = trans_prover.clone();
            trans_verifier.reset_pointer();
            let ok = batch_pcs.verify(
                &mut trans_verifier,
                &values,
                &points,
                a_com,
                i_com,
                z_com,
            );
            println!("🕒 \x1b[1m Verify time: {:.3}ms \x1b[0m", verify_timer.elapsed().as_secs_f64() * 1000.0);
            report_peak("verify");
            println!("✅ \x1b[1m Verification result: {} \x1b[0m", ok);
            assert!(ok, "BatchPCS verification failed");

            println!("🎉 \x1b[1m All steps completed successfully! \x1b[0m");

            // stop memory monitoring and report peak
            _stop.store(true, Ordering::Relaxed);
            let _ = monitor.join();
            let peak = peak_kb.load(Ordering::Relaxed);
            println!("⬜ \x1b[1m Peak RSS during run: {} KB \x1b[0m", peak);
        }
    }
}

fn spawn_mem_monitor(sample: Duration) -> (Arc<AtomicBool>, Arc<AtomicU64>, thread::JoinHandle<()>) {
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicU64::new(0));
    let stop_cloned = stop.clone();
    let peak_cloned = peak.clone();
    let handle = thread::spawn(move || {
        while !stop_cloned.load(Ordering::Relaxed) {
            if let Some(rss) = read_rss_kb() {
                let _ = peak_cloned.fetch_max(rss, Ordering::Relaxed);
            }
            thread::sleep(sample);
        }
    });
    (stop, peak, handle)
}

fn read_rss_kb() -> Option<u64> {
    // print '⌛' every 30 seconds as a heartbeat
    static LAST_HEARTBEAT: OnceLock<Mutex<Instant>> = OnceLock::new();
    if let Ok(mut t) = LAST_HEARTBEAT.get_or_init(|| Mutex::new(Instant::now())).lock() {
        if t.elapsed() >= Duration::from_secs(30) {
            print!("⌛");
            let _ = io::stdout().flush();
            *t = Instant::now();
        }
    }

    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if let Some(rest) = line.strip_prefix("VmRSS:") {
            let parts: Vec<&str> = rest.split_whitespace().collect();
            if let Some(val) = parts.get(0) {
                if let Ok(num) = val.parse::<u64>() {
                    return Some(num);
                }
            }
        }
    }
    None
}
