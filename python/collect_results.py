#!/usr/bin/env python3
"""Collect validated ParComm sweep logs into one CSV file."""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path


REQUIRED_TRUE_MARKERS = (
    "accepted",
    "decoded_proof_verified",
    "trailing_bytes_rejected",
)

SECTION6_TIMING_FIELDS = (
    "point_commitment_seconds",
    "point_compilation_seconds",
    "packed_eo_reduction_seconds",
    "bva_opening_seconds",
    "final_tbv_seconds",
    "auxiliary_pcs_seconds",
    "auxiliary_opening_seconds",
)

SECTION6_PROOF_SIZE_FIELDS = (
    "pi_ip_proof_size_bytes",
    "pi_a_proof_size_bytes",
)

CSV_FIELDS = (
    "size_log",
    "size_label",
    "geometric_max_log",
    "rayon_threads",
    "vector_count",
    "physical_entries",
    "active_cover_entries",
    "logical_entries",
    "setup_or_load_seconds",
    "generation_seconds",
    "claim_seconds",
    "commit_seconds",
    "packed_eo_seconds",
    "opening_seconds",
    "open_seconds",
    *SECTION6_TIMING_FIELDS,
    "serialization_seconds",
    "verifier_reduction_seconds",
    "verifier_pairing_seconds",
    "verify_seconds",
    "convenience_verify_seconds",
    "measured_stage_seconds",
    "wall_seconds",
    "proof_size_bytes",
    "packed_eo_size_bytes",
    *SECTION6_PROOF_SIZE_FIELDS,
    "peak_rss_kib",
    "time_max_rss_kib",
    "peak_rss_gib",
    "accepted",
    "decoded_proof_verified",
    "trailing_bytes_rejected",
    "exit_code",
    "proof_sha256",
    "log_file",
)


def parse_elapsed(value: str) -> float:
    """Convert GNU time's m:ss or h:mm:ss value into seconds."""
    parts = value.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return int(minutes) * 60 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    raise ValueError(f"unsupported elapsed time: {value}")


def format_optional_float(values: dict[str, str], name: str) -> str:
    """Format an optional timing while leaving legacy logs representable."""
    value = values.get(name)
    return "" if value in (None, "") else f"{float(value):.6f}"


def exact_log2(value: int, *, path: Path) -> int:
    """Return log2(value), rejecting non-power-of-two logical sizes."""
    if value <= 0 or value & (value - 1):
        raise ValueError(f"{path}: logical_entries={value} is not a power of two")
    return value.bit_length() - 1


def parse_log(path: Path) -> dict[str, str] | None:
    """Parse one complete, verified experiment log."""
    values: dict[str, str] = {}
    max_rss_kib = None
    wall_seconds = None

    for raw_line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        match = re.fullmatch(r"([A-Za-z][A-Za-z0-9_]*)=(.*)", line)
        if match:
            values[match.group(1)] = match.group(2).strip()
            continue
        if line.startswith("Maximum resident set size (kbytes):"):
            max_rss_kib = int(line.rsplit(":", 1)[1].strip())
        elif line.startswith("Elapsed (wall clock) time"):
            wall_seconds = parse_elapsed(line.split("): ", 1)[1])

    if values.get("EXPERIMENT_EXIT_CODE") != "0":
        return None
    if any(values.get(marker) != "true" for marker in REQUIRED_TRUE_MARKERS):
        return None

    is_section6 = "opening_prove_seconds" in values
    logical_entries = int(values["logical_entries"])
    if is_section6:
        max_log = int(values["max_log"])
        size_log = exact_log2(logical_entries, path=path)
    else:
        max_log = int(values["geometric_max_log"])
        size_log = max_log + 1
    threads = int(values["rayon_threads"])
    physical_entries = int(values["physical_entries"])
    active_cover_entries = values.get("active_cover_entries", "")

    if not 20 <= size_log <= 34:
        raise ValueError(f"{path}: unexpected size 2^{size_log}")
    if threads != 64:
        raise ValueError(f"{path}: expected 64 Rayon threads, got {threads}")
    if is_section6:
        if not 0 < physical_entries <= logical_entries:
            raise ValueError(
                f"{path}: physical_entries={physical_entries} exceeds "
                f"logical_entries={logical_entries}"
            )
    else:
        expected_physical = (1 << size_log) + size_log - 1
        expected_active_cover = (1 << (size_log + 1)) - 2
        expected_logical = 1 << (size_log + 1)
        if physical_entries != expected_physical:
            raise ValueError(
                f"{path}: physical_entries={physical_entries}, "
                f"expected {expected_physical}"
            )
        if int(active_cover_entries) != expected_active_cover:
            raise ValueError(
                f"{path}: active_cover_entries={active_cover_entries}, "
                f"expected {expected_active_cover}"
            )
        if logical_entries != expected_logical:
            raise ValueError(
                f"{path}: logical_entries={logical_entries}, "
                f"expected {expected_logical}"
            )
    if wall_seconds is None or max_rss_kib is None:
        raise ValueError(f"{path}: missing GNU time wall/RSS measurements")
    if len(values.get("proof_sha256", "")) != 64:
        raise ValueError(f"{path}: missing SHA-256 proof digest")

    if is_section6:
        packed_eo_seconds = float(
            values.get(
                "packed_eo_reduction_seconds",
                sum(
                    float(values.get(name, 0.0))
                    for name in (
                        "packed_eo_cross_terms_seconds",
                        "packed_eo_fold_seconds",
                        "packed_eo_reduced_check_seconds",
                        "packed_eo_binary_rounds_seconds",
                    )
                ),
            )
        )
        open_seconds = float(values["opening_prove_seconds"])
        stages = {
            "setup_or_load_seconds": float(values["setup_seconds"]),
            "generation_seconds": float(values["generation_seconds"]),
            "claim_seconds": float(values["claims_seconds"]),
            "commit_seconds": float(values["shape_commit_seconds"])
            + float(values["bva_commit_seconds"]),
            "packed_eo_seconds": packed_eo_seconds,
            "opening_seconds": max(0.0, open_seconds - packed_eo_seconds),
            "serialization_seconds": float(values["serialization_seconds"]),
            "verifier_reduction_seconds": float(
                values["registration_verify_seconds"]
            ),
            "verifier_pairing_seconds": float(values["opening_verify_seconds"]),
        }
        verify_seconds = float(
            values.get(
                "total_verifier_seconds",
                stages["verifier_reduction_seconds"]
                + stages["verifier_pairing_seconds"],
            )
        )
        measured_stage_seconds = (
            stages["setup_or_load_seconds"]
            + stages["generation_seconds"]
            + stages["claim_seconds"]
            + stages["commit_seconds"]
            + float(values["registration_prove_seconds"])
            + open_seconds
            + stages["serialization_seconds"]
            + verify_seconds
        )
    else:
        stage_names = (
            "setup_or_load_seconds",
            "generation_seconds",
            "claim_seconds",
            "commit_seconds",
            "packed_eo_seconds",
            "opening_seconds",
            "serialization_seconds",
            "verifier_reduction_seconds",
            "verifier_pairing_seconds",
        )
        stages = {name: float(values[name]) for name in stage_names}
        open_seconds = stages["packed_eo_seconds"] + stages["opening_seconds"]
        verify_seconds = (
            stages["verifier_reduction_seconds"]
            + stages["verifier_pairing_seconds"]
        )
        measured_stage_seconds = sum(stages.values())
    peak_rss_kib = max(int(values["peak_rss_kib"]), max_rss_kib)

    row = {
        "size_log": str(size_log),
        "size_label": f"2^{size_log}",
        "geometric_max_log": str(max_log),
        "rayon_threads": str(threads),
        "vector_count": values.get("vector_count", values.get("matrix_count", "")),
        "physical_entries": str(physical_entries),
        "active_cover_entries": str(active_cover_entries),
        "logical_entries": str(logical_entries),
        **{name: f"{number:.6f}" for name, number in stages.items()},
        "open_seconds": f"{open_seconds:.6f}",
        **{
            name: format_optional_float(values, name)
            for name in SECTION6_TIMING_FIELDS
        },
        "verify_seconds": f"{verify_seconds:.6f}",
        "convenience_verify_seconds": values.get(
            "convenience_verify_seconds", ""
        ),
        "measured_stage_seconds": f"{measured_stage_seconds:.6f}",
        "wall_seconds": f"{wall_seconds:.6f}",
        "proof_size_bytes": values["proof_size_bytes"],
        "packed_eo_size_bytes": values.get(
            "packed_eo_size_bytes", values.get("pi_ip_proof_size_bytes", "")
        ),
        **{name: values.get(name, "") for name in SECTION6_PROOF_SIZE_FIELDS},
        "peak_rss_kib": str(peak_rss_kib),
        "time_max_rss_kib": str(max_rss_kib),
        "peak_rss_gib": f"{peak_rss_kib / (1024 ** 2):.6f}",
        "accepted": values["accepted"],
        "decoded_proof_verified": values["decoded_proof_verified"],
        "trailing_bytes_rejected": values["trailing_bytes_rejected"],
        "exit_code": values["EXPERIMENT_EXIT_CODE"],
        "proof_sha256": values["proof_sha256"],
        "log_file": path.name,
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    parser.add_argument("--output", type=Path, default=Path("python/result.csv"))
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    rows_by_log: dict[int, dict[str, str]] = {}
    patterns = (
        str(args.log_dir / "parcomm_2p*_64.log"),
        str(args.log_dir / "section6_2p*.log"),
    )
    log_names = sorted({name for pattern in patterns for name in glob.glob(pattern)})
    for name in log_names:
        path = Path(name)
        if ".smoke." in path.name or ".nested-" in path.name:
            continue
        row = parse_log(path)
        if row is not None and int(row["size_log"]) % 2 == 0:
            rows_by_log[int(row["size_log"])] = row

    rows = [rows_by_log[size_log] for size_log in sorted(rows_by_log)]
    if args.require_complete:
        expected = set(range(20, 35, 2))
        missing = sorted(expected - set(rows_by_log))
        if missing:
            missing_text = ", ".join(f"2^{size_log}" for size_log in missing)
            raise SystemExit(f"missing complete verified logs: {missing_text}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(args.output)
    print(f"wrote {len(rows)} validated rows to {args.output}")


if __name__ == "__main__":
    main()
