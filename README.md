# ParComm

This repository contains the Jolt-backed ParComm implementation and the small-field
QAFold public-metadata implementation, with reference results for Llama 8B/70B at
8/16 bits. Start with this document; no server login or private Git credentials
are needed to build the included source snapshots.

## Contents

| Path | Purpose |
|---|---|
| `Cargo.toml`, `batch/`, `atomic_proof/`, `fs/`, `mat/`, `poly-commit/` | Jolt-backed ParComm workspace |
| `qafold/f65521/` | Optimized 8-bit public-metadata ParComm |
| `qafold/koalabear/` | Optimized 16-bit public-metadata ParComm |
| `third_party/` | Exact local Jolt snapshot and pinned Git dependency source trees |
| `.cargo/jolt-patch.toml` | Local source overrides for the required customized dependencies |
| `scripts/run.py` | Unified build, test, smoke-test and model experiment entry point |
| `scripts/summarize.py` | Check saved results and export reference CSV tables |
| `results/reference/` | Eight original result logs, QAFold measurements and provenance |
| `validation/` | Build and correctness validation evidence; these are not new model timings |
| `benchmarks/` | Sources and portable runners for the FlashProve, Hyrax and implicit-eq EO result logs; see [benchmark reproduction](benchmarks/README.md) |

## Scope and timing

| Property | Jolt | QAFold |
|---|---|---|
| Field | BN254 | F65521/degree-8 extension (8-bit); KoalaBear/degree-4 extension (16-bit) |
| Statement visibility | Existing hidden-relation/BlindFold implementation | Public shape, query point and value |
| CPU only | Yes | Yes |
| Model-scale reference runs | Existing four archived logs | Four low-memory runs, one invocation each |
| Prover time | Commit + open from the original final-results table | Commit + opening/serialization, including reconstruction |
| Setup | Original setup fields retained | Separately reported, excluded from prover and verifier |

The two backends do not have identical hiding guarantees. Do not interpret a
timing comparison as a comparison of identical security configurations. These
experiments commit to and open model parameter matrices; they do not prove a
neural-network inference. BF16 inputs preserve their 16-bit payloads, not a new
16-bit quantization scheme. QAFold verifies all original roots through one native
PCS opening. Input conversion and claim generation are separate from prover time.

## Requirements

Linux x86-64 (Ubuntu 22.04 or WSL2), Python >=3.10, NumPy 2.2.6 for checkpoint
conversion, GNU `time`, and Rustup. No CUDA or GPU is used.

```bash
rustup toolchain install nightly-2026-03-21 --profile minimal
rustup toolchain install 1.97.1 --profile minimal
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Toolchain/Python installation and the first Rust build require Internet access.
Rust dependencies are downloaded by Cargo with `--locked`; dependency mirrors
and build products are excluded from Git. If optional local `dependencies/` or
`dependencies-jolt/` mirrors and their vendor configurations exist, the runner
uses them with `--offline --locked`. Required customized source snapshots remain
in `third_party/` and `vendor/`. QAFold's reference
server runs used Rust 1.95; artifact smoke validation uses 1.97.1. Jolt uses the
historical nightly. Timing is hardware/compiler dependent.

The reference server had two Xeon Gold 6342 CPUs, 48 physical/96 logical cores,
AVX-512, and approximately 377 GiB RAM. Small smoke tests need no model weights
and use two threads. Allow several GiB RAM and ample disk space for Rust builds.
For all four model-scale runs, use a server with approximately 384 GiB RAM and
sufficient checkpoint storage. QAFold observed prover RSS peaks were 23.91,
37.94, 130.16 and 196.45 GiB respectively; input conversion and the OS also need
memory. These are observations, not enforced memory limits. Run large cases
sequentially. The reference runs used 48 QAFold prover threads and one verifier
thread; Jolt uses 64 Rayon threads.

## Quick smoke tests

From this directory (each output must be new):

```bash
python3 scripts/run.py smoke --backend qafold --case 8B-8bit --output runs/smoke-qafold8
python3 scripts/run.py smoke --backend qafold --case 8B-16bit --output runs/smoke-qafold16
python3 scripts/run.py smoke --backend jolt --output runs/smoke-jolt
```

Each command builds first, runs a small generated instance, checks acceptance,
and writes `status.txt` containing `success`. QAFold also runs an independent
verifier. Jolt acceptance is `verified_marker=SECTION6_E2E_VERIFIED` in `run.log`.
Use `--target-dir /path/with/space` to relocate build products.

## Correctness tests

```bash
python3 scripts/run.py test --backend qafold --case 8B-8bit --output runs/test-qafold8
python3 scripts/run.py test --backend qafold --case 8B-16bit --output runs/test-qafold16
python3 scripts/run.py test --backend jolt --output runs/test-jolt
```

The QAFold suite includes reconstruction equivalence, encoding reference checks,
Merkle-root consistency and rejection of altered proofs/statements. The Jolt
entry runs the focused packed-EO and matrix-collection end-to-end tests.

## Model experiments

Obtain compatible safetensors checkpoints separately under their respective
licenses. Weights and access credentials are not distributed. Use local paths:

| Case | Reference checkpoint directory name | QAFold chunk/packed log |
|---|---|---|
| `8B-8bit` | `Llama-3.1-8B-W8A8` | 30 / 4 |
| `8B-16bit` | `Llama-3.1-8B-BF16` | 30 / 4 |
| `70B-8bit` | `Llama-3.1-70B-W8A8` | 30 / 6 |
| `70B-16bit` | `Llama-3.1-70B-BF16-community` | 30 / 6 |

```bash
python3 scripts/run.py model --backend qafold --case 8B-8bit \
  --input /data/Llama-3.1-8B-W8A8 --output runs/qafold-8b8
python3 scripts/run.py model --backend jolt --case 8B-8bit \
  --input /data/Llama-3.1-8B-W8A8 --output runs/jolt-8b8
```

Repeat with the other three case/path pairs. Each invocation runs exactly once;
there is no automatic retry or median aggregation. QAFold always uses
`recompute` low-memory mode here. It streams converted input, records SHA256,
retains all reconstruction cost inside the prover timer, and writes the proof,
statement, claims, profile and independent verification result. Recorded commands
and process resource logs accompany each run. Avoid concurrent benchmarking.

## Inspect supplied results without rerunning models

```bash
python3 scripts/summarize.py
```

The summary script validates acceptance and writes `results/summary.csv` and
`results/summary.md`. Reference data are preserved verbatim, including historical
machine paths; those paths are provenance, not paths required by this runner.
The 70B 16-bit QAFold result comes from the preceding optimized run, and the
other three QAFold results from the subsequent model sweep. No model-scale
experiment was repeated just to package this artifact.

See `PROVENANCE.md`, `LICENSES.md` and `validation/STATUS.md` for source origins,
repository-layout changes, attribution and the exact validation boundary.
The repository is the distribution unit; no separate archive is required.
Historical experiment, dependency-mirror and maintenance directories are ignored
by Git. Required source snapshots preserve their line endings through
`.gitattributes`.
