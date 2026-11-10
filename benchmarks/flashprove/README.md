# FlashProve native PCS, parallel CPU streaming port

Upstream revision: `9e2bdd8ff5bf9c62f18263d7dac189b0400b260c`.
This experiment is separate from Jolt and ParComm/QAFold.

The CPU keeper implements the upstream `IndexHolder` contract. It evaluates
aligned RS codeword intervals, incrementally hashes columns in row slabs, keeps
only upper Merkle nodes, and recomputes queried subtrees. It checks reconstructed
nodes before emitting opening bytes. All field, hash, transcript, proximity-test,
and verifier routines come from upstream. The full codeword is never stored.
OpenMP parallelizes row encoding, column hashing, tree levels, record generation,
and table folding. The small-oracle `EngineCpu` remains upstream sequential code.

The standalone benchmark uses one public MLE evaluation of one table. It runs
LIN at its **one-time** proximity-test radius in each opening: no prior table
verification is assumed, and no test-only trusted-table or hand-written error
bound is enabled. This measures commitment plus a complete self-contained native
PCS opening, not ParComm, full model inference, or production CUDA throughput.
The planner checks a 128-bit per-round target with no proof of work. This is not
a claim of 128-bit security for the whole compiled implementation.

One native request-mask oracle is retained to match the upstream LIN planner;
its coefficient in the public evaluation claim is zero, so the external value
is bound directly and is not replaced by a prover-chosen masked value.

## Inputs and timing

The existing QAFold converter supplies identical compact integer input bytes.
The runner requires the recorded SHA256 and element count of each previous
workload. BF16 is preserved as signed 16-bit payloads, as in those experiments;
this is not floating-point arithmetic. Signed i8/i16 entries are embedded directly
in Goldilocks. The flattened input fills table rows contiguously, with explicit
zero padding. This standalone layout is not the Llama-4 production GPU layout.

- One run per model/bit-width, 48 prover threads; sequential verifier.
- Conversion/input and parameter setup are reported separately.
- Prover = commit (including record) + opening (including evaluation and proof
  serialization). Proof file disk writes are outside prover time.
- Independent verifier reconstructs parameters outside its timer, then decodes
  the public statement/proof and verifies. Its reported time includes file I/O.
- Proof byte count includes four 8-byte framing lengths; public statement/key
  bytes are saved separately.
- Every run saves proof, statement, logs, input hash, commands and peak RSS.

## Build and validation

Use Linux, ordinary user, g++ 13+ and OpenMP. Set `ROOT` to this directory and
use an output directory outside `upstream/blocks`. Example compiler commands:

```
g++ -std=c++17 -O3 -fopenmp -DBLK_FIELD=2 -DBLK_HASH=1 -I upstream/blocks -I cpu cpu/equivalence_test.cpp -o BUILD/equivalence
g++ -std=c++17 -O3 -fopenmp -DSERVING_BUILD -DBLK_FIELD=2 -DBLK_HASH=1 -I upstream/blocks -I cpu cpu/native_test.cpp -o BUILD/native_test
g++ -std=c++17 -O3 -fopenmp -DSERVING_BUILD -DBLK_FIELD=2 -DBLK_HASH=1 -I upstream/blocks -I cpu cpu/model_bench.cpp -o BUILD/model_bench
```

`equivalence_test` is synthetic only and compares 81 configurations against the
upstream CPU reference: exact encoding, root, record, folded values, Merkle opening
bytes and whole LIN proof bytes, plus verification and tampering rejection.
`native_test` uses actual upstream plans without test-only bypasses, accepts an
honest opening, and rejects a wrong public value and a modified proof.

`model_bench INPUT WIDTH THREADS OUTPUT_PREFIX MATRIX_LOG [plan]` runs a model.
`model_bench verify OUTPUT_PREFIX` replays verification in a fresh process, without
access to weights. The public statement must be the one retained from the trusted
benchmark run; receiving an arbitrary statement alongside a proof does not establish
which model was committed.

Use `run_models.py --help` for the sequential four-case runner. Its output directory
must be new. It deletes only its own temporary compact input after independent
verification succeeds. It never deletes checkpoints or earlier experiment results.

## Completed measurements

All four model cases completed once on the server and passed independent verification.
Server GCC 13.4 build, differential tests, native negative checks and synthetic replay passed.
The official stored example returned YES (exit 0) in the historical experiment.
See `../../result_flashprove_4scenarios.log` for the four measured cases and
`../README.md` for portable build and reproduction commands. This compact source
directory contains the shared blocks required by the CPU PCS, not the full GPU
inference implementation or official inference-verifier executable.

