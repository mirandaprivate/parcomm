# zkSmart HD-R1CS

This context describes zero-knowledge proofs for matrix circuits whose witness
blocks may have different true dimensions while sharing one global algebraic
statement.

## Language

**Global HD-R1CS instance**:
One circuit-wide relation whose gates are grouped by operation type and share a
single witness namespace.
_Avoid_: Per-layer proof, layer composition

**Global witness**:
The complete collection of circuit values addressed in one common block domain
and bound by one online commitment.
_Avoid_: Layer witness set

**Witness block**:
A true-size matrix or vector placed in a slot of the global witness domain.
Unused cells in its slot are virtual zeros.

**Global block ID**:
The stable identity of a witness block across all equations. Reusing an ID is
the meaning of cross-layer wire equality.
_Avoid_: Layer-local commitment link

**Operation type**:
One of multiplication, addition, or activation. All gates of the same type are
constrained by one global equation group.

**Routing**:
The committed relationship from global left, right, and output wire positions
to witness block IDs.
_Avoid_: Layer boundary equality proof

**Architecture commitment**:
A commitment to routing, selectors, I/O extraction, or activation auxiliary
structure. It is distinct from the online witness commitment.

**Witness commitment (`C_w`)**:
The single online commitment that binds the global witness.
_Avoid_: Per-route witness commitment, per-layer witness commitment

**Routed witness claim**:
A claimed evaluation of `C_w` reached through a routing or direct-witness
relation at a particular point.

**Synchronized projection batch**:
A collection of routed witness claims reduced with shared challenges to one
point, with equality of their final reduced commitments enforced before one
`C_w` opening.

**Heterogeneous block domain**:
A common power-of-two slot domain whose blocks have different true row and
column dimensions.

**Virtual zero**:
A padded-domain entry defined to be zero without being physically stored.
_Avoid_: Materialized padding

**Physical witness entries**:
The true entries actually stored across all witness blocks.

**Logical padded entries**:
The full common-domain size used by the algebraic relation, including virtual
zeros.

**Exact ReLU relation**:
An activation relation using nonnegative output and negative-part range
decompositions plus a zero-product constraint.
_Avoid_: Quantized sigmoid lookup
