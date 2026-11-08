#!/usr/bin/env python3
"""Expand a jolt-dory i8 matrix directory into synthetic i16 by a fixed scale.

For each coefficient c in an i8 .bin:
    out = int16(c) * scale   (little-endian)

Default scale=256 (= left-shift 8). Must keep c*scale inside i16 for all i8 c.

Matrices are expanded in parallel across CPU workers (ProcessPoolExecutor).
Per-file conversion uses NumPy vectorized multiply when available, else a
bytearray lookup table (still much faster than struct.pack per chunk).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def _expand_with_numpy(src_path: str, dst_path: str, scale: int) -> int:
    import numpy as np

    data = np.fromfile(src_path, dtype=np.int8)
    out = (data.astype(np.int16) * np.int16(scale))
    out.tofile(dst_path)
    return int(out.nbytes)


def _expand_with_lut(src_path: str, dst_path: str, scale: int, chunk: int = 1 << 24) -> int:
    # Precompute i8 -> two LE bytes for every possible signed byte.
    lut = bytearray(256 * 2)
    for u in range(256):
        c = u if u < 128 else u - 256
        v = c * scale
        lut[u * 2] = v & 0xFF
        lut[u * 2 + 1] = (v >> 8) & 0xFF
    out_bytes = 0
    with open(src_path, "rb") as fin, open(dst_path, "wb") as fout:
        while True:
            buf = fin.read(chunk)
            if not buf:
                break
            # Translate each i8 byte through the LUT into 2 output bytes.
            mv = memoryview(buf)
            out = bytearray(len(buf) * 2)
            for i, b in enumerate(mv):
                base = b * 2
                out[i * 2] = lut[base]
                out[i * 2 + 1] = lut[base + 1]
            fout.write(out)
            out_bytes += len(out)
    return out_bytes


def expand_bin(src_path: str, dst_path: str, scale: int) -> int:
    try:
        return _expand_with_numpy(src_path, dst_path, scale)
    except ImportError:
        return _expand_with_lut(src_path, dst_path, scale)


def _worker(args: tuple[str, str, int, int, str]) -> tuple[int, str, int]:
    """Process-pool worker: (index, name, nbytes)."""
    index, src, dst, scale, name = args
    n = expand_bin(src, dst, scale)
    return index, name, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="jolt-dory-i8 directory with manifest.json")
    ap.add_argument("--output", required=True, help="output jolt-dory-i16 directory")
    ap.add_argument(
        "--scale",
        type=int,
        default=256,
        help="fixed i8→i16 multiplier (default 256 = left-shift 8)",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 8) // 2),
        help="parallel worker processes (default: half of CPUs)",
    )
    args = ap.parse_args()

    if not (-32768 <= -128 * args.scale <= 32767 and -32768 <= 127 * args.scale <= 32767):
        print(
            f"ERROR: scale={args.scale} overflows i16 for some i8 values "
            f"(range of c*scale must fit in [-32768, 32767])",
            file=sys.stderr,
        )
        return 1

    man_path = os.path.join(args.input, "manifest.json")
    with open(man_path) as f:
        man = json.load(f)
    if man.get("mode") != "i8":
        print(f"WARNING: input mode is {man.get('mode')!r}, expected 'i8'", file=sys.stderr)

    os.makedirs(args.output, exist_ok=True)
    matrices = man["matrices"]
    n_mats = len(matrices)
    workers = max(1, min(args.workers, n_mats))
    print(
        f"expand i8→i16 scale={args.scale} matrices={n_mats} workers={workers}",
        flush=True,
    )

    jobs = []
    for m in matrices:
        src = os.path.join(args.input, m["file"])
        dst = os.path.join(args.output, m["file"])
        jobs.append((m["index"], src, dst, args.scale, m["name"]))

    t0 = time.time()
    results: dict[int, int] = {}
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_worker, job) for job in jobs]
        for fut in as_completed(futures):
            index, name, nbytes = fut.result()
            results[index] = nbytes
            done += 1
            if done % 32 == 0 or done == 1 or done == n_mats:
                elapsed = time.time() - t0
                rate = sum(results.values()) / elapsed / (1 << 30) if elapsed > 0 else 0
                print(
                    f"[{done}/{n_mats}] {name} -> {nbytes/(1<<20):.1f} MiB  "
                    f"({rate:.2f} GiB/s wall)",
                    flush=True,
                )

    new_mats = []
    total_bytes = 0
    for m in matrices:
        entry = dict(m)
        entry["dtype"] = "i16"
        entry["elem_size"] = 2
        entry["source_dtype"] = "I8xSCALE"
        entry["i8_scale"] = args.scale
        new_mats.append(entry)
        total_bytes += results[m["index"]]

    out_man = {
        "model": man.get("model", "unknown") + f"-i8x{args.scale}",
        "mode": "i16",
        "source": os.path.abspath(args.input),
        "expansion": {
            "from_mode": "i8",
            "scale": args.scale,
            "formula": "i16 = i8 * scale (LE)",
            "workers": workers,
        },
        "num_matrices": len(new_mats),
        "max_vars": man["max_vars"],
        "total_live_coeffs": man["total_live_coeffs"],
        "total_pad_coeffs": man["total_pad_coeffs"],
        "total_bytes": total_bytes,
        "shape_histogram": man.get("shape_histogram", {}),
        "matrices": new_mats,
        "notes": (
            f"Synthetic i16 expanded from i8 by multiplying each coefficient by {args.scale}. "
            "Same pad_len / num_vars / balanced Dory layout as the i8 source."
        ),
    }
    out_path = os.path.join(args.output, "manifest.json")
    with open(out_path, "w") as f:
        json.dump(out_man, f, indent=2)
        f.write("\n")

    elapsed = time.time() - t0
    print()
    print(f"wrote {len(new_mats)} i16 matrices to {args.output}")
    print(
        f"scale={args.scale} workers={workers} total_bytes={total_bytes/(1<<30):.2f} GiB "
        f"elapsed_s={elapsed:.3f} ({total_bytes/elapsed/(1<<30):.2f} GiB/s)"
    )
    return 0


if __name__ == "__main__":
    # Required on some platforms for ProcessPoolExecutor spawn.
    sys.exit(main())
