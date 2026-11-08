# Evaluation guide

## What reviewers can evaluate

1. Build both CPU backends from the supplied source and locked dependencies.
2. Run small generated end-to-end examples without model access.
3. Run correctness/tamper tests for the included protocols.
4. Inspect eight archived model results without rerunning expensive experiments.
5. Reproduce one model experiment per invocation with separately supplied weights.

The executable instructions are in README.md. There are no required server
accounts, fixed server paths, scheduled jobs or GPU drivers. Python and Rust
toolchains must be installed before building. A normal clone downloads locked
registry dependencies on its first build; offline mirrors are optional local files.

## Evidence map

| Claim | Evidence | Acceptance criterion |
|---|---|---|
| Jolt end-to-end implementation executes | `scripts/run.py smoke --backend jolt ...` | SECTION6_E2E_VERIFIED marker and exit 0 |
| Small-field complete public flow executes | QAFold smoke commands, for both fields | result/verification verified=true; all roots authenticated; one native opening |
| Reconstruction preserves correctness | QAFold test command | Byte-equivalence and negative tests pass |
| Optional local mirrors support offline builds | `validation/offline-resolution.json` | Historical check: all three manifests resolved offline with local mirrors; mirrors are not tracked |
| Reported model measurements are retained | `results/reference/`, summary script | Eight successful records, separate backend/timing interpretation |
| Distribution is reproducible | Git revision, lockfiles and vendor checksums | Offline locked resolution and smoke tests pass |

## Expected cost

Building from scratch takes minutes to tens of minutes depending on CPU and
filesystem, and requires additional disk space beyond the source archive. Smoke
proofs are small; correctness tests should not require large model memory.
For full models, input conversion is a significant part of wall time. On the
reference server QAFold low-memory prover times were about 41 s, 72 s, 200 s and
352 s for 8B8, 8B16, 70B8 and 70B16. These exclude setup, input conversion and
claim generation. Total QAFold process times were about 94 s, 163 s, 617 s and
1176 s respectively. Model runs should be sequential and must fit host memory.

## Limits of the evidence

Reference model results are historical individual measurements, not an average
or confidence interval. Packaging smoke tests do not reproduce full-model
performance. The Jolt snapshot was obtained from the actual server override;
its relation to the private Git revision cannot be verified from missing Git
metadata. QAFold currently exposes shape/point/value and does not establish the
zero-knowledge hiding properties of the Jolt path. No neural inference is proven.
No artifact-evaluation badge or conference acceptance is claimed.
