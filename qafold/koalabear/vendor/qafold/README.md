# QAFold over F_(65521^8)

This crate implements QAFold with QA generators, recursive setup, and challenges
over `F_(65521^8) = F_65521[X]/(X^8 - 17)`. Its packed PCS commits to a Boolean
evaluation table over `F65521`: eight adjacent base-field elements occupy one
extension-field element. The packed-opening reduction authenticates the original
base-field MLE claim before completing the QAFold opening and recursive Decide.

The opening reduction uses eight Frobenius-conjugate evaluations and batches
them into one degree-two sumcheck and one QAFold opening. The native QA encoder
accepts monomial coefficients, so commitment includes a Boolean-to-monomial
transform. Packing reduces the number of field elements eightfold; eight
2-byte base elements and one 16-byte extension element have equal payload size.

The implementation uses delayed coefficient reduction for octic Karatsuba
multiplication and the packing projections, a fixed Frobenius basis map, and
cache-blocked Boolean-to-monomial conversion. Large opening inner products and
folds run in parallel; small tails run serially. These optimizations preserve
the field, code parameters, transcript, and serialized proof. Each benchmark
record includes a proof digest to check byte equality across implementations.

Systematic QA encoding preserves the message block directly and uses its final
parity block as transform workspace. At rate one half this needs two WHTs per
column instead of three. The generator's transpose projection also combines
frequency-domain contributions before its final WHT. On x86-64, butterflies use
contiguous 16-bit coefficients with SSE2/AVX2/AVX-512BW dispatch; other platforms
retain the scalar implementation. Large codeword initialization and coefficient
copies are parallel. Merkle construction reuses row serialization buffers and
fixed hash prefixes, preserving every root and proof byte.
For large columns, upper WHT stages gather narrow stripes into at most 256 KiB
of scratch per active column, transform them in cache, then scatter them back.

The main modules are:

| Module | Role |
|---|---|
| `qafold_short` | F65521/F65521^8 arithmetic, QA coding and generator reductions |
| `full_extension` | Full-extension generators, sampling and code parameters |
| `qafold_short_pc` | Merkle commitments, opening transcript and recursive Decide |
| `packed` | Base-field packing and original-MLE opening reduction |
| `benches/pcs_65521.rs` | Standalone PCS benchmark and log output |

Run the requested PCS experiment on `2^30` base-field elements (`2^27` packed
extension-field elements), using 64 workers and one complete measured proof:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench -p mikros-qafold --bench pcs_65521 -- \
  --log-n 30 --threads 64 --reps 1 --columns 32 \
  --output target/paper-results/qafold-f65521-n30.log
```

A small validation run uses the same protocol:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench -p mikros-qafold --bench pcs_65521 -- \
  --log-n 14 --threads 1 --reps 1 --columns 8 \
  --output target/paper-results/qafold-f65521-smoke.log
```

The benchmark reports setup and witness generation separately. Online prover
time includes commitment, packing/opening reduction, native QAFold opening,
recursive Decide, and serialization. Verifier time includes parsing and every
proof check. Each sample prints progress, prover/verifier time, canonical proof
bytes, and whole-process peak RSS, followed by a median summary. The same output,
including machine-readable `QAFOLD_PCS_SAMPLE` records, is written to the selected
log. RSS is Linux `VmHWM`, including setup, inputs, all samples and verification;
it is reported as unavailable on platforms without `/proc/self/status`.
Proof bytes count the serialized packing reduction, opening, and Decide; the
32-byte commitment root is reported separately in each machine-readable record.
Commit timing is split into packing, coefficient copy, Mobius conversion,
codeword allocation, QA encoding, witness release, and Merkle construction.
Allocation/page-fault cost can move between allocation and encoding depending
on the allocator, so compare their sum as well as the total commitment time.

For a single-run encoding breakdown, add `--features profile-encoding` before
the benchmark's `--`. This opt-in build times WHTs, alpha multiplications,
message/parity copies, and the residual in each top-level column. It preserves
the encoder's parallel schedule. The report gives the outer wall time separately
from shares of summed column elapsed times; concurrent columns overlap, so these
shares are not CPU-cycle shares or a wall-time decomposition. WHT includes its
cache rearrangement and scratch allocation, and multiplication includes array
reads and writes. Packing, Mobius conversion, codeword allocation and Merkle
hashing remain separate. The default build does not enable these internal timers.

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench -p mikros-qafold --bench pcs_65521 \
  --features profile-encoding -- --log-n 30 --threads 64 --reps 1 --columns 32 \
  --output target/paper-results/qafold-encoding-profile.log
```

Parameters and security accounting remain explicit: the default `100` is the
query/RF target, not a claim of 100-bit or 128-bit whole-protocol security.
`F65521^8` has approximately 128 bits of field cardinality. Code-distance checks
use the full extension cardinality. The separate `Prime65521` short-code backend
retains its original small-field parameter policy and is not the main benchmark.
Source provenance and the exact adaptation boundary are in `PROVENANCE.md` and
`SOURCE_MANIFEST.json`; the upstream license is retained in `LICENSE_MIT`.
