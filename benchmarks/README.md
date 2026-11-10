# Sources for the additional benchmark logs

| Result in repository root | Source | Measured scope |
|---|---|---|
| `result_flashprove_4scenarios.log` | `flashprove/cpu/`, `flashprove/upstream/blocks/` | CPU streaming native LIN PCS, four model cases |
| `result_hyrax_4scenarios.log` | `hyrax/src/`, `hyrax/native-poly-commit/` | Native BN254 Hyrax with homomorphic size-grouped batching |
| `result_packed_eo_1B.log` and `result_packed_eo_litebullet_bitwidth.log` | `packed_eo/` plus the root Rust workspace | Hidden EO-LiteBullet component, N=2^30, implicit MLE eq weights |

The two EO logs are identical. None of these additional benchmarks measures a
complete ParComm model proof or inference. Read each implementation's scope
below before comparing its timings. The root Jolt and QAFold implementations
remain unchanged.

## Build and synthetic validation

Run from the repository root on Linux/WSL as an ordinary user, with Python 3.10+.
FlashProve requires g++ 13+ and OpenMP. Hyrax uses Rust 1.95; EO uses the root
workspace's nightly-2026-03-21. Install these separately if needed; the runner
does not install software or invoke sudo. Cargo downloads public registry
dependencies on the first build; `--offline` uses an existing local cache.
Customized EO dependencies are already tracked under `third_party/` and `vendor/`.

```sh
python3 benchmarks/run.py flashprove smoke --output runs/flashprove-smoke
python3 benchmarks/run.py hyrax smoke --output runs/hyrax-smoke
python3 benchmarks/run.py packed_eo smoke --output runs/eo-smoke
```

Each output directory must be new. Use `--cxx /path/to/g++` for an isolated
compiler and `--target-dir /path/to/build` to relocate build products. `build`
instead of `smoke` builds without running. Successful smoke runs write `status.txt`
with `success`, plus the exact commands, settings and validation logs.

FlashProve runs 81 differential configurations, native negative tests, a synthetic
file benchmark and independent verification. Hyrax tests two widths with both
same-size and different-size matrices, independent verification and rejection of
corrupted proofs, wrong points and changed constituent commitments. EO tests all
seven widths, signed extrema and Boolean eq points (42 cases), comparing complete
proofs to an explicit full-field reference and independently checking endpoints.

## Reproduce model PCS measurements

The scripts run each model case once, sequentially, and require model checkpoints
obtained separately. Models and credentials are not distributed. `modelconvert.py`
is the exact converter from the measured source archive; its NumPy dependencies
are covered by the root `requirements.txt`. `--models` names the parent directory
of the four checkpoint directories documented in the root README. Input hashes
must match the historical inputs or the runners stop before timing.

```sh
python3 benchmarks/run.py flashprove build --output runs/flashprove-build
python3 benchmarks/flashprove/run_models.py --models /data/models --scratch /data/scratch-flash \
  --output runs/flashprove-models --binary "$PWD/target/benchmarks/model_bench" \
  --converter "$PWD/benchmarks/modelconvert.py" --threads 48

python3 benchmarks/run.py hyrax build --output runs/hyrax-build
python3 benchmarks/hyrax/run_models.py --models /data/models --scratch /data/scratch-hyrax \
  --output runs/hyrax-models --binary "$PWD/target/benchmarks/release/hyrax-model-bench" \
  --converter "$PWD/benchmarks/modelconvert.py" --threads 48
```

FlashProve measures one public evaluation of one native table, including one-time
proximity testing in the opening. This is a CPU port, not CUDA throughput.
Hyrax verifies hidden evaluation commitments; `public_evaluation_checked=false`.
Its upstream setup is testing-only, not a production setup. Original model
commitments are aggregated by the verifier and that work is included in its time.
Each runner retains an independently verified proof and detailed timing boundaries.

## Reproduce EO bitwidth measurements

```sh
python3 benchmarks/run.py packed_eo bench --threads 64 --log-n 30 --output runs/eo-1B
```

Seven configurations run once each; one verifier thread. `a` is a native signed
4/8/16/32/64/128-bit vector or full BN254 scalar field vector. `b` is the implicit
MLE equality vector in LSB-first coordinate order. Adaptive packing is retained.
Both projection and first folding use real radix-256 byte-plane arithmetic for
integer inputs, without dummy work. Only packed row coefficients are materialized.

The measured component includes hiding scalar commitments and BlindFold; input
PCS commitments and terminal PCS openings are excluded. Setup, generation,
initial claim preparation and serialization are reported separately. Eq row
coefficient generation is included in prover time. The transcript fixture is
not a commitment to either measured input. This remains a component benchmark.
Peak RSS is process lifetime high-water memory; no strict time proportionality
is promised. Full-width input uses native field arithmetic.

`packed_eo/prepare.py` generates the ignored `packed_eo/source/` workspace from
the tracked root crates and adds the measured adapter and kernels. Its dependency
paths are resolved locally, with no hard-coded user or server paths. The generated
workspace is a snapshot: regenerate it after changing the root implementation.

## Provenance and cleanup

`source-origin.json` records original paths and SHA-256 hashes before portable
wrapper/documentation edits. Measured arithmetic was checked byte-for-byte against
the original FlashProve, Hyrax batched and EO v6 source archives. The EO adapter's
preparation logic is retained; only its dependency-path resolution was made portable.
FlashProve upstream revision is `9e2bdd8ff5bf9c62f18263d7dac189b0400b260c`.
Only its shared blocks and required CPU port are distributed here, with upstream
LICENSE, NOTICE and third-party attribution. GPU inference and official full
inference-verifier sources are not needed by this PCS benchmark.

Historical raw measurements and existing source archives remain local under
ignored `experiments/`. Obsolete unpacked EO v1-v5 source copies are removed only
after confirming archive coverage. Build products, downloaded dependencies,
proof files, compiler packages, model data and old process scripts are not committed.
