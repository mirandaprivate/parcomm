#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=${1:-/home/mscong1/zksmart-gpt2-sweep-20260810}
TARGET_DIR=${CARGO_TARGET_DIR:-/home/mscong1/zksmart-normalization-20260810/target}
export PATH=/home/mscong1/.local/rust-1.97.1/bin:$PATH
export RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-32}

cd "$ROOT_DIR"

for model in small medium large xl; do
    log="server-gpt2-${model}-full.log"
    status_file="server-gpt2-${model}-full.exit"
    set +e
    CARGO_TARGET_DIR="$TARGET_DIR" /usr/bin/time -v \
        cargo bench -p zksmart --bench gpt2_hd_r1cs -- \
        --model "$model" --full --sequence-len 32 --skip-normalization-proof \
        2>&1 | tee "$log"
    status=${PIPESTATUS[0]}
    set -e
    printf '%s\n' "$status" > "$status_file"
    if [[ $status -ne 0 ]]; then
        exit "$status"
    fi
done
