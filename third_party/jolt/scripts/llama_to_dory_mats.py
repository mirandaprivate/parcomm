#!/usr/bin/env python3
"""Convert Llama safetensors weights into Dory-friendly padded matrices.

For each selected 2D weight:
  1. read raw i8 (W8A8) or bf16-bit-reinterpreted-as-i16 (BF16)
  2. flatten row-major and zero-pad to the next power-of-two length
  3. write a dense .bin (little-endian) plus a JSON manifest

Dory commits each matrix in the balanced layout
  rows = 2^{n - ceil(n/2)}, cols = 2^{ceil(n/2)}
where n = log2(pad_len). The .bin is exactly that flat coefficient vector.

Examples:
  python3 scripts/llama_to_dory_mats.py \\
    --input ~/git/model-data/Llama-3.1-8B-W8A8 \\
    --mode i8 --output ~/git/model-data/Llama-3.1-8B-W8A8/jolt-dory-i8

  python3 scripts/llama_to_dory_mats.py \\
    --input ~/git/model-data/Llama-3.1-8B-BF16 \\
    --mode i16 --output ~/git/model-data/Llama-3.1-8B-BF16/jolt-dory-i16
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
from collections import Counter


def next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def parse_safetensors_header(path: str) -> tuple[dict, int]:
    with open(path, "rb") as f:
        header_len = int.from_bytes(f.read(8), "little")
        header = json.loads(f.read(header_len))
    return header, 8 + header_len


def iter_shard_tensors(model_dir: str):
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]
    shards = sorted(set(weight_map.values()))
    for shard in shards:
        path = os.path.join(model_dir, shard)
        header, data_base = parse_safetensors_header(path)
        for name, info in header.items():
            if name == "__metadata__":
                continue
            yield name, info, path, data_base


def select_tensors(mode: str, name: str, dtype: str, shape: list[int]) -> bool:
    if len(shape) != 2:
        return False
    rows, cols = shape
    if rows == 1 or cols == 1:
        # skip per-channel scales / bias-like (N,1) tensors
        return False
    if mode == "i8":
        return dtype == "I8"
    if mode == "i16":
        # commit all real 2D BF16 weights (layers + embed/lm_head)
        return dtype == "BF16"
    raise ValueError(f"unknown mode {mode}")


def read_tensor_bytes(path: str, data_base: int, info: dict) -> bytes:
    start, end = info["data_offsets"]
    with open(path, "rb") as f:
        f.seek(data_base + start)
        return f.read(end - start)


def convert_one(
    raw: bytes,
    dtype: str,
    shape: list[int],
    mode: str,
) -> tuple[bytes, dict]:
    rows, cols = shape
    live_len = rows * cols
    pad_len = next_pow2(live_len)
    num_vars = pad_len.bit_length() - 1
    sigma = (num_vars + 1) // 2
    nu = num_vars - sigma

    if mode == "i8":
        if dtype != "I8" or len(raw) != live_len:
            raise ValueError(f"I8 size mismatch: {len(raw)} vs {live_len}")
        out = bytearray(pad_len)  # zero padded
        out[:live_len] = raw
        elem_size = 1
        storage = "i8"
    elif mode == "i16":
        if dtype != "BF16" or len(raw) != live_len * 2:
            raise ValueError(f"BF16 size mismatch: {len(raw)} vs {live_len * 2}")
        # Reinterpret bf16 bit patterns as little-endian i16 — no float math.
        out = bytearray(pad_len * 2)
        out[: live_len * 2] = raw
        elem_size = 2
        storage = "i16"
    else:
        raise ValueError(mode)

    meta = {
        "live_rows": rows,
        "live_cols": cols,
        "live_len": live_len,
        "pad_len": pad_len,
        "num_vars": num_vars,
        "sigma": sigma,
        "nu": nu,
        "pad_rows": 1 << nu,
        "pad_cols": 1 << sigma,
        "dtype": storage,
        "elem_size": elem_size,
        "source_dtype": dtype,
    }
    return bytes(out), meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="HF model directory with safetensors shards")
    ap.add_argument("--output", required=True, help="output directory for .bin + manifest.json")
    ap.add_argument("--mode", choices=["i8", "i16"], required=True)
    ap.add_argument("--name", default=None, help="manifest model name (default: basename)")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model_name = args.name or os.path.basename(os.path.abspath(args.input))
    t0 = time.time()

    matrices = []
    shape_hist = Counter()
    total_live = 0
    total_pad = 0
    total_bytes = 0
    idx = 0

    for name, info, path, data_base in iter_shard_tensors(args.input):
        dtype = info["dtype"]
        shape = list(info["shape"])
        if not select_tensors(args.mode, name, dtype, shape):
            continue
        raw = read_tensor_bytes(path, data_base, info)
        blob, meta = convert_one(raw, dtype, shape, args.mode)
        fname = f"{idx:04d}.bin"
        out_path = os.path.join(args.output, fname)
        with open(out_path, "wb") as f:
            f.write(blob)
        entry = {
            "index": idx,
            "name": name,
            "file": fname,
            **meta,
        }
        matrices.append(entry)
        shape_hist[tuple(shape)] += 1
        total_live += meta["live_len"]
        total_pad += meta["pad_len"]
        total_bytes += len(blob)
        idx += 1
        if idx % 32 == 0 or idx == 1:
            print(
                f"[{idx}] {name} {shape} -> pad_len={meta['pad_len']} "
                f"(n={meta['num_vars']}) {len(blob)/(1<<20):.1f} MiB",
                flush=True,
            )

    if not matrices:
        print("ERROR: no tensors selected", file=sys.stderr)
        return 1

    max_vars = max(m["num_vars"] for m in matrices)
    manifest = {
        "model": model_name,
        "mode": args.mode,
        "source": os.path.abspath(args.input),
        "num_matrices": len(matrices),
        "max_vars": max_vars,
        "total_live_coeffs": total_live,
        "total_pad_coeffs": total_pad,
        "total_bytes": total_bytes,
        "shape_histogram": {f"{a}x{b}": c for (a, b), c in sorted(shape_hist.items())},
        "matrices": matrices,
        "notes": (
            "Each .bin is a zero-padded flat coefficient vector of length pad_len. "
            "Dory balanced layout uses pad_cols=2^ceil(n/2), pad_rows=2^{n-ceil(n/2)}. "
            "i16 mode stores raw BF16 bit patterns as little-endian i16."
        ),
    }
    man_path = os.path.join(args.output, "manifest.json")
    with open(man_path, "w") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")

    elapsed = time.time() - t0
    print()
    print(f"wrote {len(matrices)} matrices to {args.output}")
    print(f"manifest {man_path}")
    print(
        f"live_coeffs={total_live} (~2^{total_live.bit_length()-1 if total_live else 0}+) "
        f"pad_coeffs={total_pad} (~2^{total_pad.bit_length()-1}) "
        f"bytes={total_bytes/(1<<30):.2f} GiB max_vars={max_vars}"
    )
    print(f"shapes {dict(shape_hist)}")
    print(f"elapsed_s={elapsed:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
