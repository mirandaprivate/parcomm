---
status: accepted
---

# Batch all global witness claims into one opening

All routed evaluations of the online witness commitment `C_w` are reduced with
shared Fiat-Shamir challenges to one point and then discharged by one SmartPC
opening. The equal-size implementation's dense `BatchPoints` random linear
combination cannot be used directly because it would construct vectors over
the 137,438,953,472-entry logical padded domain. The heterogeneous adaptation
therefore performs virtual synchronized projections, uses one final hiding,
and requires the verifier to check that every reduced commitment is identical
before accepting the single opening.

## Consequences

The equality check is a soundness requirement, not an optimization: without
it, individually valid projections against different matrices could be
followed by an opening of only the first reduced value. Architecture and I/O
commitments remain separate; “one opening” refers specifically to `C_w`.
