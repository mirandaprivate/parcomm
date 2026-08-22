# ParComm patch

This crate is based on `a16z/arkworks-algebra`, branch `dev/twist-shout`, commit
`76bb3a4518928f1ff7f15875f940d614bb9845e6`.

ParComm changes `msm_bigint_wnaf` to reuse the current Rayon pool. The upstream
fork creates a new two-thread pool for every WNAF chunk. Jolt Dory already
parallelizes dense commitments across rows, so the nested pool construction can
create thousands of operating-system threads and fail before the commitment
finishes. The patched implementation retains the same chunk computation and
ordered group reduction while scheduling all work on the existing pool.
