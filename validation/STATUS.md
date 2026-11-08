# Packaging validation — 2026-11-08

## Git repository layout validation

The supported files now live directly in the repository; the duplicate
`artifact/` tree and submission archive were removed. All three root-level
smoke commands passed again using separate initially empty Cargo caches:
`repository-jolt-isolated/`, `repository-qafold8-isolated/`, and
`repository-qafold16-isolated/`. Each completed the full small proof and
verification flow. Offline locked metadata resolution also passed after the
move. `git-distribution-audit.json` records that required distribution files
are not excluded by Git ignore rules. `.gitattributes` preserves vendored bytes.
The result summary script was rerun from the repository root successfully.

## Earlier validation of the same implementation

| Check | Result |
|---|---|
| F65521 generated low-memory complete proof and fresh verifier | Passed (`qafold8-smoke-final/`) |
| KoalaBear generated low-memory complete proof and fresh verifier | Passed (`qafold16-smoke/`) |
| F65521 root and backend tests | Passed (`qafold8-tests/`); manual timing test ignored |
| KoalaBear root and backend tests | Passed (`qafold16-tests/`); manual timing test ignored |
| Jolt generated heterogeneous-collection complete flow | Passed (`jolt-smoke-final/`); SECTION6_E2E_VERIFIED |
| Jolt packed EO focused test | 1 passed (`jolt-tests/`) |
| Jolt matrix-collection end-to-end suite | 4 passed; 1 multi-GiB test ignored (`jolt-tests/`) |
| All three dependency graphs, empty Cargo cache | Passed offline and locked (`offline-resolution.json`) |
| Eight reference result acceptance checks | Passed; `results/summary.csv` generated |
| Python entry-point syntax | Passed |

Validation used local WSL Linux, Rust 1.97.1 for QAFold and
nightly-2026-03-21 for Jolt. Build/test logs include original warnings. Smoke
timings are not model-scale measurements and are not added to the result table.
The new model-runner commands have not been rerun on the full 8B/70B inputs in
this packaging task. The reference results are the preserved completed runs.

Only dependency-location manifests and lockfile identities changed for Jolt;
Rust implementation source files were copied unchanged. QAFold implementation
files were checked against the frozen experiment source manifest.
