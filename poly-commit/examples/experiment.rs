use std::{
    env, fs,
    io::{self, Write},
    sync::{
        atomic::{AtomicBool, AtomicU64, Ordering},
        Arc, Mutex, OnceLock,
    },
    thread,
    time::{Duration, Instant},
};

use ark_bls12_381::{Bls12_381, Fr};
use ark_ff::{UniformRand, Zero};
use ark_serialize::{CanonicalSerialize, Compress};
use ark_std::{rand::Rng, test_rng};
use poly_commit::{smart_pc::SmartPC, MyInt};
use rayon::prelude::*;

const LOG_TOTAL: usize = 30; // vector length = 2^20
const TOTAL: usize = 1 << LOG_TOTAL;
const QLOG: usize = (LOG_TOTAL + 1) / 2; // matches square reshape to 2^10 x 2^10

fn main() {
    let (_stop, peak_kb, monitor) = spawn_mem_monitor(Duration::from_millis(200));

    let mut rng = test_rng();
    let rayon_env = env::var("RAYON_NUM_THREADS").unwrap_or_else(|_| "not set".to_string());
    println!(
        "RAYON_NUM_THREADS env: {}, effective threads: {}",
        rayon_env,
        rayon::current_num_threads()
    );
    println!(
        "Running SMART-PC (flat) with 2^{} = {} i8 vector (curve: BLS12-381)",
        LOG_TOTAL, TOTAL
    );

    // generate a flat vector of length 2^20
    let flat: Vec<MyInt> = (0..TOTAL)
        .into_par_iter()
        .map(|_| {
            let mut r = test_rng();
            r.gen_range(-128i8..=127i8) as MyInt
        })
        .collect();

    // 1) setup
    let setup_start = Instant::now();
    let pp = SmartPC::<Bls12_381>::setup(QLOG, &mut rng).expect("setup should succeed");
    println!("Setup time: {:.3}s", setup_start.elapsed().as_secs_f64());
    println!(
        "Public parameter size: {} bytes",
        pp.serialized_size(Compress::Yes)
    );

    // 2) commit (flat)
    let commit_start = Instant::now();
    let (mat_com, tier1) = SmartPC::<Bls12_381>::commit_square_myint_flat(&pp, &flat, Fr::zero())
        .expect("commit should succeed");
    println!("Commit time: {:.3}s", commit_start.elapsed().as_secs_f64());
    println!(
        "Commitment size: {} bytes",
        mat_com.serialized_size(Compress::Yes)
    );

    // 3) random point evaluation: original shape 1 × 2^20 (eval_my_int_flat projects directly on flat vector without reconstructing matrix)
    let xl_raw: Vec<_> = Vec::new(); // 2^0 = 1 row
    let xr_raw: Vec<_> = (0..LOG_TOTAL)
        .into_par_iter()
        .map(|_| {
            let mut r = test_rng();
            Fr::rand(&mut r)
        })
        .collect(); // 2^20 columns

    let (v_com, v_tilde) = SmartPC::<Bls12_381>::eval_my_int_flat(&pp, &flat, &xl_raw, &xr_raw);

    // 4) open
    let open_start = Instant::now();
    let proof = SmartPC::<Bls12_381>::open_square_myint_flat(
        &pp,
        &flat,
        &xl_raw,
        &xr_raw,
        v_com,
        mat_com,
        &tier1,
        v_tilde,
        Fr::zero(),
    )
    .expect("open should succeed");
    println!("Open time: {:.3}s", open_start.elapsed().as_secs_f64());
    println!("Proof size: {} bytes", proof.serialized_size(Compress::Yes));

    // 5) verify
    let verify_start = Instant::now();
    let ok =
        SmartPC::<Bls12_381>::verify_square(&pp, mat_com, v_com, &xl_raw, &xr_raw, &proof).unwrap();
    println!(
        "Verify time: {:.3}s, result: {}",
        verify_start.elapsed().as_secs_f64(),
        ok
    );

    // stop memory monitoring and report peak
    _stop.store(true, Ordering::Relaxed);
    let _ = monitor.join();
    let peak = peak_kb.load(Ordering::Relaxed);
    if let Some(curr) = read_rss_kb() {
        println!("Current RSS: {} KB", curr);
    }
    println!("Peak RSS during run: {} KB", peak);
}

fn spawn_mem_monitor(
    sample: Duration,
) -> (Arc<AtomicBool>, Arc<AtomicU64>, thread::JoinHandle<()>) {
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
    if let Ok(mut t) = LAST_HEARTBEAT
        .get_or_init(|| Mutex::new(Instant::now()))
        .lock()
    {
        if t.elapsed() >= Duration::from_secs(30) {
            print!("⌛");
            let _ = io::stdout().flush();
            *t = Instant::now();
        }
    }

    let status = fs::read_to_string("/proc/self/status").ok()?;
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
