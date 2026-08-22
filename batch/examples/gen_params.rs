//! Preloads Jolt Dory's reusable CPU universal reference string.
//!
//! Jolt canonicalizes an odd variable count to the next even value and uses
//! dory-pcs's disk cache. The cache location follows the platform cache
//! directory, for example `~/.cache/dory/` on Ubuntu/WSL.
//!
//! ```text
//! cargo run --release --example gen_params -- --pp-log 18
//! ```

use std::time::Instant;

use batch::jolt_dory::{canonical_dory_log, setup_dory_parameters};

fn main() {
    let mut pp_logs = Vec::new();
    let mut arguments = std::env::args().skip(1);
    while let Some(argument) = arguments.next() {
        match argument.as_str() {
            "--pp-log" => pp_logs.push(
                arguments
                    .next()
                    .expect("missing --pp-log value")
                    .parse::<usize>()
                    .expect("invalid --pp-log value"),
            ),
            _ => panic!("unknown argument {argument}"),
        }
    }
    if pp_logs.is_empty() {
        pp_logs.push(18);
    }

    for requested_log in pp_logs {
        let canonical_log = canonical_dory_log(requested_log);
        let started = Instant::now();
        let (_prover, _verifier) = setup_dory_parameters(requested_log);
        println!(
            "requested_log={requested_log} canonical_log={canonical_log} setup_seconds={:.3} status=ready",
            started.elapsed().as_secs_f64(),
        );
    }
}
