# QAFold source provenance

This package adapts the current Brakefold short-prime QAFold implementation
(upstream checkout HEAD `c0cac85b05cf8e048f7c9830b38fddb1680b95a0`).
`SOURCE_MANIFEST.json` records the exact upstream source hashes and initial adapted
source hashes. The upstream MIT permission notice is retained in `LICENSE_MIT`.

The short-code implementation preserves the `Prime65521` / `Prime65521Ext8` backend,
base-field top commitment, extension-field folding, transcript domain labels,
canonical proof encoding, authenticated recursive Decide, and parameter checks.
The degree-eight extension is F_65521[X]/(X^8 - 17). Its canonical representation
has eight 16-bit base-field coefficients; base witnesses occupy two bytes each.

Porting changes: unrelated Mersenne31, Goldilocks, and Mersenne127 field backends,
their upstream test modules, and the alternative non-accumulating proof driver
are omitted. The required numerical distance helpers are extracted into small
modules. `ark_std` RNG/containers are replaced by the same rand 0.8 interfaces
and the standard library. The six used error variants are provided locally.
No external repository path or Arkworks/Plonky3 runtime dependency is required.
Unused helpers belonging only to omitted upstream tests and the non-accumulating
driver are removed. The mixed multi-evaluation row fold also includes a serial
fallback: upstream accidentally compiled the entire fold out without `parallel`.

The original Prime65521 short-code setup policy is retained: QA generators and random-
foldable sources are over the base field, RFv1 Decide setup, an entropy-distance
model, and complete-row fallback when the distance calculation cannot support
sampling. Full-extension challenges do not repair a base-code distance bound.
The requested security-bit parameter is a query/RF target, not a verified
whole-protocol security certificate; its upstream maximum is 111 bits.

The main packed PCS instead uses the added `FullPrime65521Ext8` adapter. Its QA
generators, recursive setup coefficients, and challenges all occupy the full
octic field. Field characteristic and cardinality have separate interfaces;
arithmetic is never performed modulo the composite integer `65521^8`. The full
adapter uses a distinct hash identity, canonical eight-limb rejection sampling,
and direct-interleaved setup. The `packed` module adds an authenticated reduction
from a base-field Boolean MLE to the full-field packed commitment. These additions
are local work and are not represented as unmodified upstream QAFold code.

The low-level `commit_owned` input is a monomial coefficient table (MSB-first
tensor order), not a Boolean-evaluation table. A caller opening a Boolean MLE
must apply the usual Boolean-to-monomial transform before commitment and include
that transform in online commitment time. Full succinct verification requires
`verify_opening_reduction` **and** `verify_decide`; `verify` is the separate
direct public-generator check and must not be reported as succinct verification.
