---
status: accepted
---

# Use one global HD-R1CS instance grouped by operation type

The heterogeneous MLP proof is one circuit-wide HD-R1CS statement with one
global witness namespace, committed routing, and equation groups for I/O,
multiplication, addition, and activation. We rejected a vector of per-layer
proofs because it changes the LaTeX statement, replaces hidden global wiring
with layer-boundary commitment equalities, and makes low RAM a consequence of
a weaker composition boundary rather than an implementation of global
HD-R1CS. Different layer sizes are represented inside the global statement by
true-size blocks and virtual zeros.

## Consequences

Cross-layer equality is represented by shared global block IDs. RAM must be
controlled through virtual storage, replay, and scheduling without changing
the statement into layer-local proofs.
