# ParComm

ParComm is a Rust implementation of collection commitments for heterogeneous
low-bit-width matrix collections.

The paper source is available in [`latex/`](./latex/).

## Requirements

- Ubuntu Linux or WSL2
- [Rustup](https://rustup.rs/) with Rust 1.95
- Python 3 with NumPy for the Llama experiments

The Rust toolchain is pinned in [`rust-toolchain.toml`](./rust-toolchain.toml)
and is selected automatically by Cargo.

## Quick Start

Clone and build the repository:

```bash
git clone git@github.com:mirandaprivate/parcomm.git
cd parcomm
rustup toolchain install 1.95
cargo build --locked --workspace --release
```

Run a locally generated homogeneous experiment:

```bash
RAYON_NUM_THREADS=8 cargo bench --locked --quiet -p batch \
  --bench experiment -- \
  homogeneous \
  --matrix-count 128 \
  --rows 127 \
  --cols 127 \
  --allow-non-64-threads
```

This command generates 128 matrices of shape `127 x 127` in memory and runs
commitment, opening, and verification. It prints detailed stage timings,
proof-size components, peak RSS, and the final result table.

## Tests

Run the focused end-to-end tests:

```bash
cargo test --locked -p batch --test packed_eo_short_int
cargo test --locked -p batch --test matrix_collection_e2e -- --test-threads=1
```

Run the complete workspace test suite:

```bash
cargo test --locked --workspace -- --test-threads=1
```

## Homogeneous Experiments

Use `--matrix-count`, `--rows`, and `--cols` to select the generated matrix
collection. Paper-scale runs use 64 Rayon threads:

```bash
mkdir -p logs

RAYON_NUM_THREADS=64 cargo bench --locked --quiet -p batch \
  --bench experiment -- \
  homogeneous \
  --matrix-count 1024 \
  --rows 4095 \
  --cols 4095 \
  2>&1 | tee logs/result_homogeneous.log
```

## Llama Experiments

The Llama benchmarks read sharded safetensors checkpoints from
`../model-data` by default and stream the converted BVA directly into memory.

| Benchmark | Default checkpoint directory |
| --- | --- |
| Llama-8B 8-bit | `../model-data/Llama-3.1-8B-W8A8` |
| Llama-8B 16-bit | `../model-data/Llama-3.1-8B-BF16` |
| Llama-70B 8-bit | `../model-data/Llama-3.1-70B-W8A8` |
| Llama-70B 16-bit | `../model-data/Llama-3.1-70B` |

Run the four model experiments from the repository root:

```bash
mkdir -p logs

RAYON_NUM_THREADS=64 cargo bench --locked --quiet -p batch \
  --bench experiment_llama8B_8bit \
  2>&1 | tee logs/result_llama8B_8bit.log

RAYON_NUM_THREADS=64 cargo bench --locked --quiet -p batch \
  --bench experiment_llama8B_16bit \
  2>&1 | tee logs/result_llama8B_16bit.log

RAYON_NUM_THREADS=64 cargo bench --locked --quiet -p batch \
  --bench experiment_llama70B_8bit \
  2>&1 | tee logs/result_llama70B_8bit.log

RAYON_NUM_THREADS=64 cargo bench --locked --quiet -p batch \
  --bench experiment_llama70B_16bit \
  2>&1 | tee logs/result_llama70B_16bit.log
```

Pass `--input PATH` after `--` to use another compatible checkpoint directory:

```bash
RAYON_NUM_THREADS=64 cargo bench --locked --quiet -p batch \
  --bench experiment_llama70B_16bit -- \
  --input /path/to/llama-70b-bf16
```

## Output

Detailed stage timings and proof-size components are printed by default. A
successful end-to-end run finishes with a `Final results` table and:

```text
verified_marker=SECTION6_E2E_VERIFIED
```

Print the available options with:

```bash
cargo bench --locked --quiet -p batch --bench experiment -- --help
cargo bench --locked --quiet -p batch --bench experiment_llama8B_8bit -- --help
```
