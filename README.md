# ParComm

ParComm is a collection commitment for decreasing-size heterogeneous vectors.
The current implementation keeps low-bit-width witnesses in native storage,
uses one real bucket per input vector, and applies packed EO-LiteBullet before
opening the resulting dense byte BVA with Jolt's CPU Dory PCS over BN254.

## Workspace

- `atomic_proof/`: atomic proof protocols and shared reduction types.
- `batch/`: decreasing BVA, packed EO-LiteBullet, and the end-to-end ParComm flow.
- `fs/`: Fiat-Shamir transcript primitives.
- `mat/`: matrix and multilinear-evaluation utilities.
- `poly-commit/`: the legacy SmartPC implementation retained for compatibility.

## Prerequisites

- Rust toolchain `1.95`.
- Ubuntu or WSL2 for benchmark runs.
- An explicit `RAYON_NUM_THREADS` value for reproducible experiments.

Install the pinned toolchain if needed:

```bash
rustup toolchain install 1.95
```

## Build and Test

```bash
CARGO_TARGET_DIR=/tmp/parcomm-target \
  cargo +1.95 build --locked --workspace --release

CARGO_TARGET_DIR=/tmp/parcomm-target \
  cargo +1.95 test --locked --workspace --release
```

## Local End-to-End Run

This run uses square matrices with sides `2^7-1, 2^6-1, ..., 2^2-1`.
It first commits the padded `alpha = (matrix, row offset, column offset)` and
`gamma = (row log, column log)` shape tables, then commits the contiguous dense
BVA. Verification opens the BVA, checks the terminal TBV inner product, and
finally checks the committed shape encoding against the public descriptor and
matrix evaluation points. The complete proof, including both shape openings,
is exactly decoded before verification.

```bash
RAYON_NUM_THREADS=8 CARGO_TARGET_DIR=/tmp/parcomm-target \
  cargo +1.95 run --locked --release -p batch \
  --example parcomm_e2e -- \
  --max-log 7
```

The output reports stage timings, proof size and SHA-256, physical i8 storage,
the maximum live packed-field state, one materialized Dory projection vector,
zero-padding counters, the derived Dory setup size, separate BVA-opening/TBV-IP/
shape-consistency results, and the final verification result. Add
`--dense-witness-oracle` to independently
recompute the final MLE from the contiguous dense i8 witness. This diagnostic is
kept outside the formal consistency timing and is disabled for large runs.

The packed EO-LiteBullet layer can also be measured without Dory:

```bash
RAYON_NUM_THREADS=8 CARGO_TARGET_DIR=/tmp/parcomm-target \
  cargo +1.95 run --locked --release -p batch \
  --example packed_eo -- \
  --geometric-max-log 12 --allow-non-64-threads
```

## 64-Thread Experiment

The full matrix workload uses sides `2^15-1, 2^14-1, ..., 2^2-1` and requires
multi-GiB memory. Its dense BVA has 1,431,524,710 physical i8 entries and a
`2^31` logical length. Jolt canonicalizes its requested 31-variable Dory setup
to 32 variables and automatically reuses the dory-pcs disk cache.

```bash
RAYON_NUM_THREADS=64 CARGO_TARGET_DIR=/tmp/parcomm-target \
  cargo +1.95 run --locked --release -p batch --example gen_params -- \
  --pp-log 31

RAYON_NUM_THREADS=64 CARGO_TARGET_DIR=/tmp/parcomm-target \
  cargo +1.95 run --locked --release -p batch \
  --example parcomm_e2e -- --max-log 15
```

Generated parameters, proofs, logs, and experiment results are ignored by Git.

## Limitations

- Proof generation uses fresh opening randomness, so the proof SHA-256 identifies
  one run and is not expected to remain constant across runs.
- The CPU Dory verifier is terminal: it performs pairings immediately and does
  not yet emit the recursive sparse verifier equation used by legacy SmartPC.
- The GPU path remains unchanged and is not wired to the Dory backend.
- The current CPU path proves that committed shape tables equal the public
  canonical descriptor. It does not yet hide shapes: the hidden-shape Section 6
  lookup, grand-product, and committed-evaluation relation proofs remain future
  work.
- Reported witness and materialized-field byte counts exclude public
  parameters, group state, allocator metadata, and the proof object.

## License

Licensed under either the Apache License, Version 2.0, or the MIT license.
