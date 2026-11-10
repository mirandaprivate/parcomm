# Hyrax homomorphic batching benchmark

Scope: local poly-commit Hyrax over BN254, CPU only, 48 prover threads, one run
per scenario. Not zkLLM CUDA or ParComm end-to-end. Library commit/open/check are
unchanged. An isolated snapshot adds two private-prover-state helper interfaces
for reading blinding factors and constructing an aggregated state.

Every original model tensor has its own commitment. Tensors with the same padded
polynomial size share one opening point and one random-linear-combination proof.
The batch transcript binds the complete ordered metadata and all commitments
before deriving coefficients and group query points. Matrices, commitments and
blinding factors use identical coefficients. The independent verifier derives the
same challenges and aggregates all original commitments; its cost is timed.

Two input passes avoid retaining all matrices in field form. Raw streams must match
prior FlashProve/QAFold hashes. BVA boundary zeros are removed when reconstructing
tensors; native Hyrax pads each flattened tensor to an even variable count. Record
physical, stream and padded counts separately. Rank-one tensors are retained.

Prover = commit + aggregation + native opening/serialization. Input conversion,
field loading, parameter setup/trim and proof file writing are separately excluded.
Opening includes matrix/blinding/commitment aggregation and constructing its state.
Independent verifier includes commitment/proof reads, decoding, transcript challenges,
commitment aggregation and native check; parameter setup and shape parsing are excluded.
Proof size sums group proofs, excludes original commitments and public metadata.

Native check validates a hidden evaluation commitment, ignoring its public-values
argument. The benchmark reports public_evaluation_checked=false. Native setup is
explicitly testing-only; no production security claim is made.

Local synthetic 8-bit and 16-bit checks passed: two same-size matrices aggregate
into one opening while a differently sized matrix has a separate opening. Tests
include boundary padding and negative checks for corrupt proof, wrong point and
changed constituent commitment. Both independently replay without input weights.

Use Rust 1.95 (already installed on the server); the server default Rust 1.88 is
older than the locked enum-ordinalize dependency requires. Native root repository
files and previous experiment results remain unchanged.
