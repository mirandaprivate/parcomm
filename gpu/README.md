# Four-GPU SmartPC tier-one commitment

This module implements the BLS12-381 SmartPC `commit_full` path with four CUDA devices and
parallel CPU pairing. The matrix is column-major, matching the existing SmartPC implementation.
Each GPU owns exactly one quarter of the columns.

| Stage | Placement |
| --- | --- |
| Fiat-Shamir challenge generation | CPU memory |
| Matrix scalar storage | Evenly sharded across four GPU memories |
| Tier-one column MSMs | ICICLE CUDA MSM, one column shard per GPU |
| Tier-one result transfer | One G1 result per column, GPU to CPU |
| Second-tier pairing | Four CPU workers, using the same column shards |
| Four GT shard products and hiding term | Constant-size CPU merge |

ICICLE v4 provides the CUDA runtime and tier-one MSM. Arkworks performs the four CPU pairing
shards. The executable compares every tier-one point and the final commitment against
`SmartPC::commit_full` before reporting success.

## Signed i8 input

The signed-i8 path keeps the raw matrix evenly sharded across the four GPUs. Each GPU expands
one chunk at a time into positive and negative 8-bit magnitudes, runs two ICICLE MSMs with
`bitsize = 8`, and subtracts the negative result from the positive result. This preserves signed
field semantics without converting negative values into full-width scalar representatives.
Pairing remains in the four CPU workers.

For a square matrix containing `2^30` signed bytes, run:

```bash
cargo run --release --bin smart_pc_i8_commit -- --log-elements 30 --chunk-columns 256
```

The executable validates a four-column warmup and five columns distributed across the full
matrix before reporting success.

## Build and run

Ubuntu, CUDA, four visible GPUs, an ICICLE v4 frontend installation, and the ICICLE CUDA backend
are required. The runtime paths are explicit so the build is reproducible.

```bash
export ICICLE_FRONTEND_INSTALL_DIR=/opt/icicle/lib
export ICICLE_BACKEND_INSTALL_DIR=/opt/icicle/lib/backend
export LD_LIBRARY_PATH=/opt/icicle/lib:${LD_LIBRARY_PATH:-}
export CUDA_VISIBLE_DEVICES=0,1,2,3
cargo run --release --bin smart_pc_commit -- --log-q 8 --repeats 3
```
