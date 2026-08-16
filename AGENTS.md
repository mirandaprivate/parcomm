## Agent skills

### Issue tracker

Issues and PRDs are tracked in GitHub Issues using the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Use the default five-label triage vocabulary. See `docs/agents/triage-labels.md`.

### Domain docs

This is a single-context repository. See `docs/agents/domain.md`.

## Global Constraints (Must Follow)

1. Language constraint:
   - All inline code comments, docstrings, and explanations must be in English.
   - No Chinese comments are allowed in code.

2. Execution environment constraint:
   - All commands, scripts, and paths must be compatible with WSL.
   - Do not use Windows-native paths (e.g., `C:\Users\...`).
   - Use Linux-style paths (e.g., `/mnt/c/...` if needed).

3. Environment reproducibility:
   - Assume Ubuntu (WSL) as the default runtime.
   - wsl -e bash -lc "cd /mnt/c/... && ..."
   - Avoid OS-specific behavior unless explicitly required.

## Repository Workflow

1. Search and inspection:
   - Prefer `rg` and `rg --files` for code search.
   - Prefer small, targeted file reads before editing.
   - Use Linux-style paths in commands, especially when invoking commands through WSL.

2. Editing:
   - Keep changes scoped to the requested protocol or utility layer.
   - Do not reintroduce deleted legacy modules unless explicitly requested.
   - Avoid unrelated refactors while fixing protocol logic.
   - Use English names and English comments only.

3. Testing:
   - After changing one protocol file, run the narrowest relevant unit tests first.
   - After crate-level protocol changes, run the affected crate tests.
   - For `plonkish/zktools`, avoid running the full lib test binary with default parallelism on memory-constrained WSL. Prefer exact tests, grouped tests, or `-- --test-threads=1`.
   - If a full test run exits with `SIGKILL`, check for OOM and leftover `cargo` or test-binary processes before treating it as a logic failure.

## Code Style

1. Helper functions:
   - Put reusable, general-purpose matrix and challenge helpers in utility modules such as `composite/src/utils/matop.rs`.
   - Inline very short helper functions when extracting them makes the protocol harder to read.
   - Do not hide simple local checks behind one-line helper functions.

2. Matrix data movement:
   - Avoid unnecessary clones on large matrices or vectors.
   - If a function may return the original matrix without modification, prefer taking ownership instead of taking a reference and cloning.
   - Only pad or copy matrix data when the shape actually changes.
   - Do not materialize padded matrices just to satisfy a power-of-two folding protocol. Prefer passing the original shape plus a padded folding shape, and treat out-of-shape entries as zero inside the protocol.
   - Avoid example-only witness clones before prover calls. If a protocol object already owns the witness matrix, its `reduce_prover` path should borrow from `self` instead of asking the caller to clone the matrix.
   - When a later subprotocol needs a large intermediate matrix already owned by an earlier subprotocol, move it out explicitly after the earlier reduction instead of cloning it.

3. Parallelism:
   - Use Rayon for large independent prover-side vector and matrix computations.
   - Combine denominator construction and zero checks when possible, instead of building data and scanning it again.
   - Use parallel reductions for large inner-product style sums.
   - Keep verifier-side work lightweight and deterministic.
   - For very wide column-major matrices with a small row count, avoid nested parallelism inside each column. Parallelize across columns and use serial loops within the small column slices.

4. Transcript usage:
   - Transcript interaction should be explicit and local to the reduction being produced or verified.
   - Do not use transcript indices as a substitute for returning structured reduction data.

## Protocol Design

1. Atomic protocols:
   - Protocols should expose clean `reduce_prover` and `verify_as_subprotocol` flows.
   - `reduce_prover` should return the reduction result directly.
   - `verify_as_subprotocol` should reduce to the same structured result without requiring witness matrices.
   - Do not reintroduce POP concepts such as `atomic_pop`, `prepare_atomic_pop`, `synthesize_atomic_pop`, `hat_c_index`, `point_c_index`, or output-passing responsibilities.
   - Use shared reduction structs from `atomic_proof/src/atomic_protocol.rs`, including `EvalWithPointMat` and `EvalWithPointVec`, instead of protocol-local duplicate claim structs.

2. Reduction responsibilities:
   - A protocol reduction should describe the claims that still need to be opened or checked.
   - The caller or test should verify that returned evaluation claims match the committed matrices or vectors.
   - Do not nest protocol reductions when the outer protocol can directly return the final evaluation claims needed by the caller.

3. Verifier inputs:
   - `verify_as_subprotocol` should not consume prover witness matrices or vectors.
   - It may use public metadata such as matrix shapes, vector lengths, and table lengths.
   - It should reconstruct transcript challenges and compute public expected values from reductions and public metadata.

4. Challenge and MLE utilities:
   - Use `xi_ip_from_challenges` whenever an inner product with a tensor equality vector can be computed directly.
   - Avoid calling `xi_from_challenges` in verifiers when only the inner product value is needed.
   - When a weighted MLE point is composed from two tensor points, convert it by element-wise multiplication of the corresponding challenge vectors.
   - If a protocol proves a claim over a logically padded matrix, returned evaluation points should still be valid points for the original unpadded matrix, with missing padded entries interpreted as zero.

5. Performance workflow:
   - For large prover benchmarks, add stage-level timings around data generation, lookup construction, denominator construction, batch inversion, reduction calls, and final claim construction.
   - When a stage is unexpectedly slow, first identify whether the time is arithmetic, allocation, flattening, cloning, or padding.
   - Prefer removing large data movement over micro-optimizing arithmetic loops.
   - Keep benchmark examples focused on prover time, verifier time, proof size, and memory usage. Do not include artificial clone/setup costs in prover time unless the benchmark explicitly measures them.

## LogUp Protocol

1. Main shape:
   - The prover computes target and table denominators, batch-inverts them, and reduces the lookup proof to a small set of MLE evaluation claims.
   - The prover should compute weighted inverse objects directly, such as target inverse values multiplied by row and column weights, and table inverse values multiplied by the table weight vector.
   - The reduction should use one matrix inner-product style claim for the target side and one vector inner-product style claim for the table side when that is sufficient.

2. RHS computation:
   - Compute `f` from the table-side weighted inverse and multiplicities.
   - Compute terms such as `<l, 1> * <r, 1>` and `<xi, 1>` with `xi_ip_from_challenges`.
   - Do not recompute expected products when they are already equal to the RHS derived from the batching equation.

3. Evaluation claims:
   - Denominator-product reductions should be converted back into evaluation claims for the original objects, such as `F`, `f`, `L`, `J`, and `mu`, with the correct composed points.
   - Tests must verify that these evaluation claims agree with the actual MLE evaluations of the protocol input.

## Sparse Protocol

1. Interface:
   - `reduce_prover` and `verify_as_subprotocol` should take the claimed value `v` and opening point as explicit inputs.
   - If the sparse matrix object already stores the non-zero values and row indices, `reduce_prover` should use those fields directly and should not require cloned witness matrices as inputs.
   - The verifier should use public shape and table-length metadata instead of witness matrices.

2. Reduction result:
   - The sparse reduction result should include the LogUp reduction claims.
   - It should also include the evaluation claim for the sparse value matrix `E` using `EvalWithPointMat`.

3. Internal protocol choice:
   - Use the matrix-native Hadamard protocol for the sparse value/index consistency layer when proving element-wise products.
   - Do not replace this part with MatIP unless the protocol design explicitly changes.
   - Hadamard reductions for sparse openings should accept original unpadded matrices together with a padded folding shape. Missing rows or columns are zero by definition.
   - When possible, pass the sparse value matrix `E` by reference into the Hadamard prover and consume the lookup-value matrix, instead of cloning `E`.

4. Tests:
   - Sparse tests should check that the protocol reduces to matrix/vector opening claims.
   - Tests should also verify that each returned evaluation value matches the actual MLE evaluation of the corresponding matrix or vector.
   - Include at least one sparse/Hadamard test where the original matrix shape is not a power of two but the folding shape is, so implicit zero padding is covered.

## Hadamard Matrix Protocol

1. Matrix-native folding:
   - Prefer `HadamardMat` for large matrix Hadamard reductions.
   - Fold columns first and rows second so the transcript and returned points match col-major LiteBullet semantics.
   - Do not flatten full matrices into vectors before folding unless the caller explicitly needs the vector protocol.

2. Implicit padding:
   - `shape_a` and `shape_b` describe the actual witness matrices.
   - `shape_c` describes the power-of-two folding shape and may be larger than the witness shape.
   - Cross terms, folds, and row folds must treat missing rows and columns as zero instead of allocating zero-filled columns.

3. Data ownership:
   - For sparse openings, support a path where the value matrix `A` is borrowed and the lookup matrix `B` is consumed.
   - The first folding round may construct a smaller owned matrix from borrowed `A`; avoid constructing a full weighted clone of `A` when a first column fold can reduce it immediately.

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues for this repository. See `docs/agents/issue-tracker.md`.

### Triage labels

The repository uses the five default canonical triage labels. See `docs/agents/triage-labels.md`.

### Domain docs

The repository uses a single-context domain documentation layout. See `docs/agents/domain.md`.
