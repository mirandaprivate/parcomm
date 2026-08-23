# ParComm

One commitment for heterogeneous low-bit-width matrix collections.

ParComm is an experimental research implementation of collection commitments
for heterogeneously shaped matrices. The current benchmark stores matrix
entries and their padded coordinates in one contiguous, native-width Boolean
Vector Assembly (BVA), avoiding an input-sized field-element expansion. It
represents MLE queries with compact Tensor Basis Vectors (TBVs), proves the
resulting inner product with Packed EO-LiteBullet, and uses Jolt's CPU Dory PCS
over BN254 for terminal polynomial openings.

## Resources

### Paper

[ParComm: Padding-Free Shape-Private Commitments to Low-Bit-Width Matrix
Collections](./latex/main.tex)

_Anonymous USENIX Security 2027 submission._

The paper source is in [`latex/`](./latex/), and standalone protocol listings
are in [`algorithm/`](./algorithm/).

## Quickstart

> [!NOTE]
> ParComm is a research prototype and has not been audited for production use.

Run a small end-to-end experiment from the repository root:

```bash
RAYON_NUM_THREADS=8 cargo bench --quiet -p batch --bench experiment -- \
  homogeneous \
  --matrix-count 2 \
  --rows 3 \
  --cols 3 \
  --allow-non-64-threads \
  --summary-only
```

A successful run decodes and verifies the serialized proof, rejects trailing
bytes, prints four summary tables, and finishes with:

```text
verified_marker=SECTION6_E2E_VERIFIED
```

## Installation

ParComm requires the Rust toolchain pinned in
[`rust-toolchain.toml`](./rust-toolchain.toml). With `rustup` installed, Cargo
selects it automatically on the first invocation.

Clone the repository and confirm the active toolchain:

```bash
git clone git@github.com:mirandaprivate/parcomm.git
cd parcomm
rustup show
```

Ubuntu or WSL is recommended for reproducible experiment runs. Large formal
experiments also require an explicit Rayon thread count and sufficient RAM for
the selected workload.

## Build

Build the complete Rust workspace:

```bash
cargo build --locked --workspace --release
```

For a faster development check of the main experiment target:

```bash
cargo check --locked -p batch --bench experiment
```

## Test

Run the focused Packed EO and matrix-collection end-to-end tests:

```bash
cargo test --locked -p batch --test packed_eo_short_int
cargo test --locked -p batch --test matrix_collection_e2e -- --test-threads=1
```

Run the complete workspace test suite when sufficient memory is available:

```bash
cargo test --locked --workspace -- --test-threads=1
```

## Experiments

The `experiment` benchmark supports generated homogeneous matrices and
preconverted model BVA artifacts. Formal measurements require 64 Rayon threads;
local smoke tests must pass `--allow-non-64-threads` explicitly.

### Homogeneous matrices

Specify the number of matrices and their common shape:

```bash
RAYON_NUM_THREADS=64 cargo bench --quiet -p batch --bench experiment -- \
  homogeneous \
  --matrix-count 1024 \
  --rows 4095 \
  --cols 4095 \
  --target-log 34 \
  --summary-only
```

This workload contains 1,024 matrices of shape `4095 x 4095` and stores a
`2^34`-entry native-i8 BVA, including its padded coordinates. It is a
large-memory experiment; reserve at least 40 GiB of available memory.
Shape-validity proving is optional and disabled by default.

### Converted model matrices

The CPU end-to-end experiment currently accepts native-i8 BVA artifacts for
the Llama 8B and Llama 70B presets. By default, manifests and BVA files are
loaded from `../model-data` relative to the workspace.

```bash
RAYON_NUM_THREADS=64 cargo bench --quiet -p batch --bench experiment -- \
  llama8b-i8 \
  --summary-only

RAYON_NUM_THREADS=64 cargo bench --quiet -p batch --bench experiment -- \
  llama70b-i8 \
  --summary-only
```

Use `--model-data-dir`, `--manifest`, or `--bva` to override the artifact
locations. Input SHA-256 verification is enabled by default.

To prepare the canonical BVA from the safetensors checkpoint and then run the
same end-to-end experiment through one direct Cargo command, use:

```bash
RAYON_NUM_THREADS=64 cargo bench --quiet -p batch --bench experiment_llama -- \
  llama8b-i8 \
  --summary-only
```

The benchmark invokes `python3 preprocessing/modelconvert.py` directly. It
uses `../model-data/Llama-3.1-8B-W8A8` by default, reuses a complete canonical
conversion when present, and then verifies its manifest and BVA before proving.
Pass `--force-conversion` only when the existing conversion must be replaced.
The converter requires NumPy in the selected Python environment. Llama 70B i8
uses the same command with `llama70b-i8`.

### Options

```text
--target-log N           Assert the exact logical BVA domain
--seed N                 Set the deterministic prover seed
--model-data-dir PATH    Select the converted model-data root
--manifest PATH          Override a model manifest
--bva PATH               Override the BVA file
--skip-input-hash        Skip the model BVA SHA-256 pass
--allow-non-64-threads   Permit a local smoke run
--shape-validity-proof   Include the optional shape-validity proof
--summary-only           Print benchmark stages and final tables only
```

Print the complete command-line help with:

```bash
cargo bench --quiet -p batch --bench experiment -- --help
```

## Benchmark output

The end-to-end benchmark reports four groups of measurements:

| Group | Contents |
| --- | --- |
| Workload | Matrix count and shape, logical entries, bucket count, threads, packing stride, and shape-validity mode |
| Timing | Input preparation, Dory setup, commitments, Packed EO sub-stages, BVA and auxiliary openings, prover, verifier, and end-to-end time |
| Proof | Total and component proof sizes, Dory proof count, and proof SHA-256 |
| Verification and memory | Exact decoding, trailing-byte rejection, acceptance markers, and peak RSS |

The final `VERIFIED` result is printed only after proof serialization,
independent decoding, and verification have all succeeded.

Omit `--summary-only` when collecting the complete machine-readable
`key=value` metric stream in addition to the stage messages and final tables.

## Performance profiling

Measure Packed EO-LiteBullet independently of the Dory opening path:

```bash
RAYON_NUM_THREADS=8 cargo bench --quiet -p batch --bench packed_eo_ab -- \
  --bucket-count 1024 \
  --bucket-log 18 \
  --repetitions 3
```

The benchmark reports Packed Cross Terms, First Fold, Reduced Check, and the
subsequent full-width binary folding stages separately.

Validated experiment CSV files can be plotted with:

```bash
python3 python/plot.py \
  --input python/result.csv \
  --output python/result.png
```

## Repository layout

| Path | Purpose |
| --- | --- |
| [`batch/`](./batch/) | BVA construction, Packed EO-LiteBullet, Dory integration, and end-to-end experiments |
| [`atomic_proof/`](./atomic_proof/) | Atomic matrix protocols and shared reduction types |
| [`mat/`](./mat/) | Matrix storage, low-bit-width values, and MLE utilities |
| [`fs/`](./fs/) | Fiat-Shamir transcript utilities |
| [`poly-commit/`](./poly-commit/) | Polynomial-commitment primitives retained by the workspace |
| [`gpu/`](./gpu/) | Separate experimental four-GPU SmartPC commitment path |
| [`latex/`](./latex/) | Paper source |
| [`python/`](./python/) | Result collection and plotting utilities |

## Acknowledgements

The CPU terminal PCS integrates the Dory implementation from
[Jolt](https://github.com/a16z/jolt). Field and elliptic-curve arithmetic are
provided by Arkworks; the local Arkworks patch is documented in
[`vendor/ark-ec/PATCHES.md`](./vendor/ark-ec/PATCHES.md).
