use libc;
use std::fs::{self, File};
use std::io;
use std::os::unix::io::AsRawFd;

use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use ark_ec::pairing::Pairing;

use zksmart::hdr1cs::lookups::HdR1csLookup;
use zksmart::hdr1cs::muls::HdR1csMuls;
use zksmart::utils::zktr::ZkTranSeq;

const MIN_LOG_NUM: usize = 1;
const MAX_LOG_NUM: usize = 11;
const MIN_LOG_N: usize = 10;
const MAX_LOG_N: usize = 11;
const BIT_WIDTH: usize = 8;

fn main() -> io::Result<()> {
    // let log_file = File::create("experiment.log")?;
    // // experiment_lookups();

    // println!("Redirecting stdout to experiment.log...");

    // unsafe {
    //     let stdout_fd = io::stdout().as_raw_fd();
    //     let _ = libc::dup2(log_file.as_raw_fd(), stdout_fd);
    // }

    println!("\n ============================================\n");
    println!("Running experiments...");

    match std::env::var("RAYON_NUM_THREADS") {
        Ok(value) => println!("RAYON_NUM_THREADS: {}", value),
        Err(e) => println!("Couldn't read RAYON_NUM_THREADS: {}", e),
    }

    experiment_lookups();

    // experiment_muls();

    Ok(())
}

fn get_memory_usage() -> Option<u64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    for line in status.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            return parts.get(1).and_then(|s| s.parse().ok());
        }
    }
    None
}

fn experiment_muls() {
    type E = ark_bls12_381::Bls12_381;
    type F = <E as Pairing>::ScalarField;
    type G = <E as Pairing>::G1;

    for log_num in (MIN_LOG_NUM..MAX_LOG_NUM).step_by(1) {
        for log_n in (MIN_LOG_N..MAX_LOG_N).step_by(1) {
            let max_memory = Arc::new(Mutex::new(0));

            let max_memory_clone = Arc::clone(&max_memory);
            let monitor_thread = thread::spawn(move || {
                loop {
                    if let Some(memory_usage) = get_memory_usage() {
                        let mut max = max_memory_clone.lock().unwrap();
                        if memory_usage > *max {
                            *max = memory_usage;
                        }
                    }
                    thread::sleep(Duration::from_secs(1)); // 暂停 1 秒
                }
            });

            println!("\n\n============================================\n");

            println!(
                "Muls for 2e{:?} numbers of 2e{} x 2e{} matrices",
                log_num, log_n, log_n
            );

            let num = 1 << log_num;
            let n = 1 << log_n;

            let mut protocol_prover = HdR1csMuls::<E>::gen_mock(num, n, BIT_WIDTH);

            let srs = &protocol_prover.setup();

            let zk_trans_seq = &mut ZkTranSeq::<F, G>::new(&srs.zksrs);

            let (com_container, pc_proof) = protocol_prover.prove(srs, zk_trans_seq);

            let trans_seq = &mut zk_trans_seq.publish_trans();

            let mut protocol_verifier = HdR1csMuls::<E>::new(num, n, BIT_WIDTH);

            let check = protocol_verifier.verify(srs, trans_seq, &com_container, &pc_proof);

            assert_eq!(check, true);

            let max = max_memory.lock().unwrap();
            println!("\n ****Max RAM: {} KB\n", *max);
        }
    }
}

fn experiment_lookups() {
    type E = ark_bls12_381::Bls12_381;
    type F = <E as Pairing>::ScalarField;
    type G = <E as Pairing>::G1;

    let k = 2 * BIT_WIDTH;

    for log_num in (MIN_LOG_NUM..MAX_LOG_NUM).step_by(1) {
        for log_n in (MIN_LOG_N..MAX_LOG_N).step_by(1) {
            let max_memory = Arc::new(Mutex::new(0));

            let max_memory_clone = Arc::clone(&max_memory);
            let monitor_thread = thread::spawn(move || {
                loop {
                    if let Some(memory_usage) = get_memory_usage() {
                        let mut max = max_memory_clone.lock().unwrap();
                        if memory_usage > *max {
                            *max = memory_usage;
                        }
                    }
                    thread::sleep(Duration::from_secs(1)); // 暂停 1 秒
                }
            });

            println!("\n\n============================================\n");

            println!(
                "Lookups for 2e{:?} numbers of 2e{} x 2e{} matrices",
                log_num, log_n, log_n
            );

            println!("Lookup table size: 2e{}", k);

            let num = 1 << log_num;
            let n = 1 << log_n;

            let mut protocol_prover = HdR1csLookup::<E>::gen_mock(num, n, k);

            let srs = &protocol_prover.setup();

            let zk_trans_seq = &mut ZkTranSeq::<F, G>::new(&srs.zksrs);

            let (com_container, pc_proof) = protocol_prover.prove(srs, zk_trans_seq);

            let trans_seq = &mut zk_trans_seq.publish_trans();

            let mut protocol_verifier = HdR1csLookup::<E>::new(num, n, k);

            let check = protocol_verifier.verify(srs, trans_seq, &com_container, &pc_proof);

            assert_eq!(check, true);

            let max = max_memory.lock().unwrap();
            println!("\n ****Max RAM: {} KB\n", *max);
        }
    }
}
