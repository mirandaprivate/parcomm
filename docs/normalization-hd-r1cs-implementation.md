# Global HD-R1CS normalization implementation and experiment

## Status

Implemented and locally verified on 2026-08-10. The implementation proves one
manifest-defined integer LayerNorm row and one stable integer Softmax row in one
global HD-R1CS instance. It is the nonlinear normalization vertical slice for
the GPT-2 design, not a complete GPT-2 proof.

Code and experiment entry points:

- `zksmart/src/hdr1cs/global_mlp/normalization.rs`
- `zksmart/benches/global_hd_r1cs_normalization.rs`

## Integer statement

For width `d`, the LayerNorm path computes

```text
s       = sum(x_i)
c_i     = d * x_i - s
v       = sum(c_i * c_i) + epsilon
r       = InverseSqrtLUT(v)
t_i     = gamma_i * c_i * r + beta_i
t_i     = 2^output_shift * y_i + remainder_i
```

The division uses Euclidean quotient/remainder. The public manifest fixes
`epsilon`, shifts, bit width, and all lookup tables.

The Softmax path computes

```text
m       = max(score_i)
u_i     = score_i - m
e_i     = ExpLUT(u_i)
z       = sum(e_i)
r       = ReciprocalLUT(z)
p_i     = e_i * r
p_i     = 2^softmax_output_shift * base_i + remainder_i
delta   = 2^softmax_output_shift - sum(base_i)
correction_i = one_hot_i * delta
y_i     = base_i + correction_i
sum(y_i) = 2^softmax_output_shift
```

Maximum certification uses nonnegative differences, one-hot boolean witnesses,
`sum(one_hot)=1`, and `one_hot_i * (m-score_i)=0`. The same constrained
one-hot vector routes the quantization residual, so each Qq Softmax row sums
exactly to the integer unit `2^q` (and therefore represents probabilities that
sum to 1).

## Global proof organization

The compiler emits one global witness namespace. It does not prove LayerNorm
and Softmax separately and does not create per-layer witness commitments.

There are four algebraic top-level groups:

1. public I/O binding;
2. one structured linear family;
3. one structured bilinear family;
4. one tagged lookup family.

The bilinear family currently has two backend reductions: ordinary rectangular
matrix multiplication and virtual-diagonal Hadamard multiplication. They share
the same global block IDs and are reduced into the same final witness-claim
batch. Thus “four groups” is an algebraic-family count, not a claim that the
implementation emits only four transcripts.

The proof collects 12 routed `C_w` claims for this fixed slice: one I/O, three
matrix-multiplication, three linear, three elementwise, and two lookup-routing
claims. They are reduced to one final PCS opening of one global witness
commitment.

Public routing, selectors, and lookup wiring use deterministic zero-hiding
commitments. The verifier reconstructs them from the manifest and compares all
commitments. Public I/O is bound by a zero-hiding evaluation commitment at a
Fiat--Shamir point, an opening of the committed I/O matrix, and a route to the
same global witness.

## Tagged lookup boundary

`Exp`, `Reciprocal`, `InverseSqrt`, `NonNegative`, and `Boolean` have distinct
tags. All selected lookup input/output vectors are aggregated into one routed
lookup matrix before the proof is generated.

This first implementation is transparent: the lookup rows appear in the proof.
The verifier checks public table membership directly and then verifies that the
revealed rows are evaluations routed from `C_w`. It is sound as a transparent
membership-and-routing construction under the underlying transcript and PCS
assumptions, but it does not provide zero knowledge for lookup rows. Replacing
this component with a hiding lookup argument should preserve the compiler API
and global block IDs.

`NonNegative` is an identity lookup table and therefore only proves the range
enumerated by the manifest. Production manifests must use fixed, audited ranges;
they must not be synthesized from secret witness values.

## Tests

The focused command is:

```bash
cargo test -p zksmart hdr1cs::global_mlp::normalization
```

It passed 4 tests:

- compiler output has exactly three operation families;
- setup/prove/verify uses one `C_w` commitment and one final opening;
- changing public `beta` while preserving the rounded output is rejected;
- changing a lookup tag is rejected;
- replacing a true remainder with another table-valid row is rejected by
  witness routing.

The existing global MLP regression also passed:

```bash
cargo test -p zksmart global_protocol_uses_one_witness_commitment_and_type_equations
```

Finally, after the BVA/GPT-2 additions, `cargo test -p zksmart` passed all 58
unit tests and the empty doc-test suite (`58 passed; 0 failed`) locally and on
the authorized Linux server.

## Experiment

Command template:

```bash
cargo bench -p zksmart --bench global_hd_r1cs_normalization -- --width 64
```

Environment: local arm64 macOS/Darwin host, Rust/Cargo 1.91.0-nightly, release
bench profile, BLS12-381, jemalloc. Each size was run sequentially in a fresh
benchmark process. Peak RSS is `getrusage(RUSAGE_SELF).ru_maxrss`; on macOS the
reported unit is bytes.

The fixture repeats LayerNorm inputs `[1,2,3,4]` and Softmax scores `[0,1,2,3]`.
It uses synthetic public integer lookup tables. This measures the proof path and
memory representation, not approximation accuracy against floating-point GPT-2.

| Width | Stored witness entries | Logical padded entries | Lookup queries | Prove (s) | Verify (s) | Proof (B) | Peak RSS (B) | Verified |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 4 | 155 | 1,024 | 18 | 2.197151 | 0.700156 | 422,390 | 13,778,944 | yes |
| 8 | 299 | 4,096 | 34 | 2.279552 | 0.710166 | 431,286 | 14,254,080 | yes |
| 16 | 587 | 16,384 | 66 | 2.370923 | 0.723496 | 440,310 | 15,335,424 | yes |
| 32 | 1,163 | 65,536 | 130 | 2.525728 | 0.728608 | 449,590 | 18,481,152 | yes |
| 64 | 2,315 | 262,144 | 258 | 2.809057 | 0.755843 | 459,382 | 26,542,080 | yes |

The table above records the earlier pre-row-sum prototype and is retained as
historical evidence; its lookup counts, proof sizes, and timings do not include
the residual lookup/correction/final row-sum relations. The updated Q4 server
slice produced `[1,2,4,9]`, row sum 16, proof size 462,087 bytes, proving time
8.264028 s, and verification time 1.610686 s. See
`outputs/gpt2_hd_r1cs_full_server_result.md`.

For every row, the proof reports one witness commitment, one final witness
opening, and four algebraic top-level groups. Physical witness storage grows
approximately linearly with width, while the logical padded domain grows
quadratically. The width-64 run stores only 2,315 semantic entries rather than
materializing 262,144 padded entries; nevertheless PCS setup and projection work
still depend on the logical domain, so low stored RAM does not imply constant
prover work.

`proof_size_bytes` is a wire-size accounting of canonically serialized protocol
elements plus an explicit compact encoding for transparent lookup tags, lengths,
and `i32` rows. It excludes the public instance and SRS.

### Server replication

The final source increment was also deployed to the independent directory
`/home/mscong1/zksmart-normalization-20260810` on the authorized Linux server.
The server used its existing Rust 1.97.1 toolchain. The full test suite again
passed all 54 tests in 45.70 seconds.

Server command outputs are preserved at:

- `outputs/normalization-full-test.log`
- `outputs/normalization-width-sweep.log`

| Width | Prove (s) | Verify (s) | Proof (B) | Peak RSS (B) | Verified |
|---:|---:|---:|---:|---:|:---:|
| 4 | 9.226713 | 1.548554 | 422,390 | 42,467,328 | yes |
| 16 | 10.146795 | 1.570396 | 440,310 | 41,680,896 | yes |
| 64 | 12.007549 | 1.637141 | 459,382 | 41,680,896 | yes |

On this Linux server, `ru_maxrss` was already about 40 MiB at the first
measurement in each fresh process, and the small width-4--64 workload did not
raise that lifetime maximum. The proving path did include `C_w` commitment and
the final PCS opening: every server run reported exactly one of each. Therefore
this RSS result means that a 2,315-entry width-64 physical witness fits below the
runtime/toolchain baseline; it does not mean witness commitment memory was
excluded. Larger GPT-2/sequence experiments require separate phase-level RSS
measurements because this small fixture cannot predict their peak.

## Remaining work before a GPT-2 claim

- compile all rows across all transformer blocks and sequence positions;
- add GELU, causal-mask routing, embeddings, attention, residuals, and LM head;
- split reusable parameter commitment `C_theta` from online `C_w`;
- replace transparent lookup rows with a hiding/authenticated lookup argument;
- add static no-wrap bounds, production quantization tables, clamp/saturation,
  and independent bit-exact reference checks;
- run sequence-length and GPT-2 Small end-to-end experiments.

Accordingly, the verified claim is the normalization slice above. It is not a
proof of a complete GPT-2 model and not a proof of PyTorch floating-point
LayerNorm or Softmax.
