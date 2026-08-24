#!/usr/bin/env python3
"""Convert Hugging Face safetensors checkpoints into ParComm dense short-word BVA files.

The converter intentionally has no PyTorch or safetensors package dependency.
It parses safetensors headers directly, memory-maps tensor data, preserves
native i8 weights, preserves Llama BF16 payloads bit-for-bit in 16-bit storage,
quantizes other floating parameters, and writes the exact dense bucket order
constructed by ``DenseBvaDescriptor::from_shapes`` in ParComm.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import mmap
import os
import re
import struct
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np


LEGACY_FORMAT_VERSION = 1
FORMAT_VERSION = 2
SPLITTING_BUDGET = 4
BF16_BITCAST_CONVERSION = "bitcast_bf16_to_i16"
BF16_BITCAST_SCHEME = "bf16_bitcast_i16"
BF16_SOURCE_TO_STORAGE = "bitcast_bf16_u16_to_i16"
GPT2_BUFFER_PATTERN = re.compile(
    r"^(?:transformer\.)?h\.\d+\.attn\.(?:bias|masked_bias)$"
)

DTYPE_INFO: dict[str, tuple[np.dtype[Any], int]] = {
    "F32": (np.dtype("<f4"), 4),
    "F16": (np.dtype("<f2"), 2),
    # NumPy does not expose bfloat16 on every supported version. BF16 values
    # are mapped as uint16 and expanded explicitly when a floating view is used.
    "BF16": (np.dtype("<u2"), 2),
    "I8": (np.dtype("i1"), 1),
}

STORAGE_INFO: dict[str, tuple[np.dtype[Any], int, int, int]] = {
    # Numerically converted floating inputs use symmetric ranges and
    # deliberately leave the most negative two's-complement value unused.
    # The BF16 bitcast path bypasses these numeric limits entirely.
    "i8": (np.dtype("i1"), 1, -127, 127),
    "i16": (np.dtype("<i2"), 2, -32767, 32767),
}


class ConversionError(RuntimeError):
    """Raised when the checkpoint or generated ParComm artifact is invalid."""


@dataclass
class TensorSpec:
    index: int
    name: str
    source_dtype: str
    source_shape: list[int]
    matrix_shape: list[int]
    source_file: str
    source_data_offsets: list[int]
    element_count: int
    conversion: str
    quantization_scale_tensor: str | None = None
    max_abs: float | None = None
    scale: float | None = None
    bucket_indices: list[int] = field(default_factory=list)


@dataclass
class BucketSpec:
    index: int
    tensor_index: int
    row_offset: int
    row_len: int
    physical_row_len: int
    col_offset: int
    col_len: int
    physical_col_len: int
    dense_offset: int = 0
    source_offset: int = 0

    @property
    def entry_count(self) -> int:
        return self.row_len * self.col_len

    @property
    def physical_entry_count(self) -> int:
        return self.physical_row_len * self.physical_col_len

    def manifest_record(self) -> dict[str, int]:
        record = asdict(self)
        record["entry_count"] = self.entry_count
        record["physical_entry_count"] = self.physical_entry_count
        return record


@dataclass(frozen=True)
class LayoutSummary:
    tensor_count: int
    bucket_count: int
    physical_entries: int
    active_entries: int
    logical_entries: int
    boundary_padding_entries: int
    final_padding_entries: int


class SafetensorsFile:
    """Minimal read-only safetensors mapping backed by one mmap."""

    def __init__(self, path: Path):
        self.path = path
        self._stream = path.open("rb")
        prefix = self._stream.read(8)
        if len(prefix) != 8:
            raise ConversionError(f"{path} is too short to be a safetensors file")
        self.header_len = struct.unpack("<Q", prefix)[0]
        header_bytes = self._stream.read(self.header_len)
        if len(header_bytes) != self.header_len:
            raise ConversionError(f"{path} has a truncated safetensors header")
        try:
            self.header: dict[str, Any] = json.loads(header_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ConversionError(f"{path} has an invalid safetensors header") from error
        self.data_start = 8 + self.header_len
        self.file_size = path.stat().st_size
        self._mapping = mmap.mmap(self._stream.fileno(), 0, access=mmap.ACCESS_READ)

    def close(self) -> None:
        self._mapping.close()
        self._stream.close()

    def __enter__(self) -> "SafetensorsFile":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def tensor_view(self, tensor: TensorSpec) -> np.ndarray[Any, Any]:
        if tensor.source_file != self.path.name:
            raise ConversionError(
                f"tensor {tensor.name!r} belongs to {tensor.source_file}, not {self.path.name}"
            )
        dtype, _ = DTYPE_INFO[tensor.source_dtype]
        offset = self.data_start + tensor.source_data_offsets[0]
        return np.ndarray(
            shape=tuple(tensor.source_shape),
            dtype=dtype,
            buffer=self._mapping,
            offset=offset,
            order="C",
        )

    def tensor_records(self) -> list[tuple[str, Mapping[str, Any], str]]:
        return [
            (name, info, self.path.name)
            for name, info in self.header.items()
            if name != "__metadata__"
        ]

    def source_paths(self) -> list[Path]:
        return [self.path]

    def metadata(self) -> Mapping[str, Any]:
        return self.header.get("__metadata__", {})

    def data_start_for(self, source_file: str) -> int:
        if source_file != self.path.name:
            raise ConversionError(f"unknown source file {source_file!r}")
        return self.data_start

    def file_size_for(self, source_file: str) -> int:
        if source_file != self.path.name:
            raise ConversionError(f"unknown source file {source_file!r}")
        return self.file_size


class ShardedSafetensorsCheckpoint:
    """Read-only mapping for a safetensors index and all referenced shards."""

    def __init__(self, path: Path):
        self.index_path = (
            path / "model.safetensors.index.json" if path.is_dir() else path
        )
        if not self.index_path.is_file():
            raise ConversionError(
                f"cannot find model.safetensors.index.json at {self.index_path}"
            )
        try:
            self.index: dict[str, Any] = json.loads(self.index_path.read_text())
            weight_map = {
                str(name): str(shard)
                for name, shard in self.index["weight_map"].items()
            }
        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ConversionError(f"invalid safetensors index {self.index_path}") from error
        if not weight_map:
            raise ConversionError(f"empty weight map in {self.index_path}")
        self._weight_map = weight_map
        self._shards: dict[str, SafetensorsFile] = {}
        try:
            for shard_name in sorted(set(weight_map.values())):
                shard_path = self.index_path.parent / shard_name
                self._shards[shard_name] = SafetensorsFile(shard_path)
            records = self.tensor_records()
            names = {name for name, _, _ in records}
            if names != set(weight_map):
                missing = sorted(set(weight_map) - names)[:3]
                extra = sorted(names - set(weight_map))[:3]
                raise ConversionError(
                    f"safetensors index/header mismatch; missing={missing}, extra={extra}"
                )
            for name, _, shard_name in records:
                if weight_map[name] != shard_name:
                    raise ConversionError(
                        f"tensor {name!r} is in {shard_name}, index says {weight_map[name]}"
                    )
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        for shard in self._shards.values():
            shard.close()
        self._shards.clear()

    def __enter__(self) -> "ShardedSafetensorsCheckpoint":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def tensor_records(self) -> list[tuple[str, Mapping[str, Any], str]]:
        records: list[tuple[str, Mapping[str, Any], str]] = []
        for shard_name, shard in self._shards.items():
            shard_records = shard.tensor_records()
            if any(record[2] != shard_name for record in shard_records):
                raise ConversionError(f"unexpected shard filename for {shard_name}")
            records.extend(shard_records)
        return records

    def tensor_view(self, tensor: TensorSpec) -> np.ndarray[Any, Any]:
        try:
            shard = self._shards[tensor.source_file]
        except KeyError as error:
            raise ConversionError(
                f"tensor {tensor.name!r} references unknown shard {tensor.source_file}"
            ) from error
        return shard.tensor_view(tensor)

    def source_paths(self) -> list[Path]:
        return [self.index_path] + [
            self._shards[name].path for name in sorted(self._shards)
        ]

    def metadata(self) -> Mapping[str, Any]:
        return {
            "index": self.index.get("metadata", {}),
            "shards": {
                name: shard.header.get("__metadata__", {})
                for name, shard in sorted(self._shards.items())
            },
        }

    def data_start_for(self, source_file: str) -> int:
        try:
            return self._shards[source_file].data_start
        except KeyError as error:
            raise ConversionError(f"unknown source file {source_file!r}") from error

    def file_size_for(self, source_file: str) -> int:
        try:
            return self._shards[source_file].file_size
        except KeyError as error:
            raise ConversionError(f"unknown source file {source_file!r}") from error


def open_checkpoint(path: Path, model_family: str) -> Any:
    if model_family == "gpt2":
        if not path.is_file():
            raise ConversionError("GPT-2 input must be one safetensors file")
        return SafetensorsFile(path)
    if model_family in {"llama-w8a8", "llama-bf16"}:
        return ShardedSafetensorsCheckpoint(path)
    raise ConversionError(f"unsupported model family {model_family!r}")


def validate_storage_mode(model_family: str, storage_dtype: str) -> None:
    if model_family == "llama-bf16" and storage_dtype != "i16":
        raise ConversionError(
            "Llama BF16 bit-preserving storage requires --storage-dtype i16"
        )


def product(values: Sequence[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


def next_power_of_two(value: int) -> int:
    if value <= 0:
        raise ConversionError(f"cannot cover non-positive length {value}")
    return 1 << (value - 1).bit_length()


def capped_aligned_dyadic_intervals(
    length: int, budget: int = SPLITTING_BUDGET
) -> list[tuple[int, int]]:
    """Mirror matrix_collection.rs::capped_aligned_dyadic_intervals."""
    if length <= 0 or budget < 2:
        raise ConversionError(f"invalid interval request length={length}, budget={budget}")
    sizes: list[int] = []
    remaining = length
    while remaining and len(sizes) < budget:
        size = 1 << (remaining.bit_length() - 1)
        sizes.append(size)
        remaining -= size
    if remaining:
        sizes[-1] *= 2
    offset = 0
    intervals: list[tuple[int, int]] = []
    for size in sizes:
        intervals.append((offset, size))
        offset += size
    return intervals


def parse_tensor_specs(
    source: Any, model_family: str
) -> tuple[list[TensorSpec], list[dict[str, Any]]]:
    if model_family not in {"gpt2", "llama-w8a8", "llama-bf16"}:
        raise ConversionError(f"unsupported model family {model_family!r}")
    tensors: list[TensorSpec] = []
    excluded: list[dict[str, Any]] = []
    occupied: list[tuple[str, int, int, str]] = []
    records = sorted(source.tensor_records(), key=lambda record: record[0])
    all_names = {name for name, _, _ in records}
    info_by_name = {name: info for name, info, _ in records}
    for name, info, source_file in records:
        try:
            dtype_name = str(info["dtype"])
            shape = [int(value) for value in info["shape"]]
            offsets = [int(value) for value in info["data_offsets"]]
        except (KeyError, TypeError, ValueError) as error:
            raise ConversionError(f"tensor {name!r} has invalid metadata") from error
        if dtype_name not in DTYPE_INFO:
            raise ConversionError(f"tensor {name!r} uses unsupported dtype {dtype_name}")
        if not shape or any(dimension <= 0 for dimension in shape):
            raise ConversionError(f"tensor {name!r} has invalid shape {shape}")
        if len(offsets) != 2 or offsets[0] < 0 or offsets[1] < offsets[0]:
            raise ConversionError(f"tensor {name!r} has invalid offsets {offsets}")
        element_count = product(shape)
        expected_bytes = element_count * DTYPE_INFO[dtype_name][1]
        if offsets[1] - offsets[0] != expected_bytes:
            raise ConversionError(
                f"tensor {name!r} occupies {offsets[1] - offsets[0]} bytes, "
                f"expected {expected_bytes}"
            )
        if source.data_start_for(source_file) + offsets[1] > source.file_size_for(
            source_file
        ):
            raise ConversionError(
                f"tensor {name!r} extends past source file {source_file}"
            )
        occupied.append((source_file, offsets[0], offsets[1], name))

        exclusion_reason: str | None = None
        linked_weight: str | None = None
        if model_family == "gpt2" and GPT2_BUFFER_PATTERN.fullmatch(name):
            exclusion_reason = "fixed_causal_attention_mask_buffer"
        elif model_family == "llama-w8a8" and name.endswith(".weight_scale"):
            exclusion_reason = "public_quantization_scale"
            linked_weight = name.removesuffix("_scale")
        if exclusion_reason is not None:
            excluded.append(
                {
                    "name": name,
                    "reason": exclusion_reason,
                    "dtype": dtype_name,
                    "shape": shape,
                    "element_count": element_count,
                    "source_file": source_file,
                    "source_data_offsets": offsets,
                    "linked_weight": linked_weight,
                }
            )
            continue

        if model_family in {"llama-w8a8", "llama-bf16"} and not name.endswith(
            ".weight"
        ):
            raise ConversionError(f"unexpected Llama tensor {name!r}")
        if len(shape) == 1:
            matrix_shape = [shape[0], 1]
        elif len(shape) == 2:
            matrix_shape = shape.copy()
        else:
            raise ConversionError(
                f"parameter tensor {name!r} has unsupported rank {len(shape)}"
            )
        if model_family == "llama-bf16" and dtype_name != "BF16":
            raise ConversionError(
                f"Llama BF16 tensor {name!r} uses unexpected dtype {dtype_name}"
            )
        if dtype_name == "I8":
            conversion = "identity_i8"
        elif model_family == "llama-bf16":
            conversion = BF16_BITCAST_CONVERSION
        else:
            conversion = "symmetric_per_tensor"
        quantization_scale_tensor = None
        if model_family == "llama-w8a8" and dtype_name == "I8":
            quantization_scale_tensor = f"{name}_scale"
            if quantization_scale_tensor not in all_names:
                raise ConversionError(
                    f"native I8 weight {name!r} has no {quantization_scale_tensor!r}"
                )
        tensors.append(
            TensorSpec(
                index=len(tensors),
                name=name,
                source_dtype=dtype_name,
                source_shape=shape,
                matrix_shape=matrix_shape,
                source_file=source_file,
                source_data_offsets=offsets,
                element_count=element_count,
                conversion=conversion,
                quantization_scale_tensor=quantization_scale_tensor,
            )
        )

    occupied.sort()
    for previous, current in zip(occupied, occupied[1:]):
        if previous[0] == current[0] and previous[2] > current[1]:
            raise ConversionError(
                f"source tensors {previous[3]!r} and {current[3]!r} overlap"
            )
    if not tensors:
        raise ConversionError("the checkpoint contains no selected parameter tensors")
    for tensor in tensors:
        if tensor.quantization_scale_tensor is None:
            continue
        scale_info = info_by_name[tensor.quantization_scale_tensor]
        scale_shape = [int(value) for value in scale_info["shape"]]
        expected_scale_shape = [tensor.matrix_shape[0], 1]
        if str(scale_info["dtype"]) not in {"BF16", "F16", "F32"}:
            raise ConversionError(
                f"scale tensor {tensor.quantization_scale_tensor!r} is not floating point"
            )
        if scale_shape != expected_scale_shape:
            raise ConversionError(
                f"scale tensor {tensor.quantization_scale_tensor!r} has shape "
                f"{scale_shape}, expected {expected_scale_shape}"
            )
    return tensors, excluded


def build_layout(
    tensors: Sequence[TensorSpec], budget: int = SPLITTING_BUDGET
) -> tuple[list[BucketSpec], LayoutSummary]:
    buckets: list[BucketSpec] = []
    for tensor in tensors:
        tensor.bucket_indices.clear()
        rows, cols = tensor.matrix_shape
        for row_offset, row_len in capped_aligned_dyadic_intervals(rows, budget):
            for col_offset, col_len in capped_aligned_dyadic_intervals(cols, budget):
                buckets.append(
                    BucketSpec(
                        index=0,
                        tensor_index=tensor.index,
                        row_offset=row_offset,
                        row_len=row_len,
                        physical_row_len=min(rows - row_offset, row_len),
                        col_offset=col_offset,
                        col_len=col_len,
                        physical_col_len=min(cols - col_offset, col_len),
                    )
                )
    buckets.sort(
        key=lambda bucket: (
            -bucket.entry_count,
            bucket.tensor_index,
            bucket.row_offset,
            bucket.col_offset,
        )
    )
    dense_offset = 0
    source_offset = 0
    for index, bucket in enumerate(buckets):
        bucket.index = index
        bucket.dense_offset = dense_offset
        bucket.source_offset = source_offset
        tensors[bucket.tensor_index].bucket_indices.append(index)
        dense_offset += bucket.entry_count
        source_offset += bucket.physical_entry_count
    logical_entries = next_power_of_two(dense_offset)
    expected_physical = sum(tensor.element_count for tensor in tensors)
    if source_offset != expected_physical:
        raise ConversionError(
            f"layout covers {source_offset} entries, expected {expected_physical}"
        )
    summary = LayoutSummary(
        tensor_count=len(tensors),
        bucket_count=len(buckets),
        physical_entries=source_offset,
        active_entries=dense_offset,
        logical_entries=logical_entries,
        boundary_padding_entries=dense_offset - source_offset,
        final_padding_entries=logical_entries - dense_offset,
    )
    return buckets, summary


def iter_float_chunks(
    raw: np.ndarray[Any, Any], dtype_name: str, chunk_elements: int
) -> Iterator[np.ndarray[Any, np.dtype[np.float32]]]:
    flat = raw.reshape(-1)
    for start in range(0, flat.size, chunk_elements):
        block = flat[start : start + chunk_elements]
        if dtype_name == "BF16":
            bits = np.asarray(block, dtype=np.uint16).astype(np.uint32)
            yield np.left_shift(bits, 16).view(np.float32)
        else:
            yield np.asarray(block, dtype=np.float32)


def as_float32(raw: np.ndarray[Any, Any], dtype_name: str) -> np.ndarray[Any, Any]:
    if dtype_name == "BF16":
        bits = np.asarray(raw, dtype=np.uint16).astype(np.uint32)
        return np.left_shift(bits, 16).view(np.float32)
    return np.asarray(raw, dtype=np.float32)


def bitcast_bf16_to_i16(raw: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Return an i16 view with exactly the same little-endian 16-bit payloads."""
    return np.asarray(raw, dtype=np.dtype("<u2")).view(np.dtype("<i2"))


def compute_quantization_scales(
    source: Any,
    tensors: Sequence[TensorSpec],
    chunk_elements: int,
    quantization_max: int,
) -> None:
    started = time.monotonic()
    processed = 0
    total = sum(tensor.element_count for tensor in tensors)
    next_report = 0.05
    for tensor in tensors:
        raw = source.tensor_view(tensor)
        max_abs = 0.0
        for block in iter_float_chunks(raw, tensor.source_dtype, chunk_elements):
            if not bool(np.isfinite(block).all()):
                raise ConversionError(f"tensor {tensor.name!r} contains NaN or infinity")
            if block.size:
                max_abs = max(max_abs, float(np.max(np.abs(block))))
        if tensor.conversion == "identity_i8":
            scale = np.float32(1.0)
        else:
            scale = (
                np.float32(max_abs / quantization_max)
                if max_abs != 0.0
                else np.float32(1.0)
            )
        if not math.isfinite(float(scale)) or scale <= 0:
            raise ConversionError(f"tensor {tensor.name!r} produced invalid scale {scale}")
        tensor.max_abs = max_abs
        tensor.scale = float(scale)
        processed += tensor.element_count
        ratio = processed / total
        if ratio >= next_report or processed == total:
            elapsed = time.monotonic() - started
            rate = processed / max(elapsed, 1e-9) / (1 << 20)
            print(
                f"scale_progress={ratio:.1%} processed={processed} "
                f"rate_melements_per_second={rate:.1f}",
                flush=True,
            )
            next_report += 0.05


def quantize_block(
    values: np.ndarray[Any, Any],
    scale: float,
    output_dtype: np.dtype[Any],
    quantization_min: int,
    quantization_max: int,
) -> np.ndarray[Any, Any]:
    scaled = np.empty(values.shape, dtype=np.float32)
    np.divide(values, np.float32(scale), out=scaled)
    np.rint(scaled, out=scaled)
    np.clip(scaled, quantization_min, quantization_max, out=scaled)
    return scaled.astype(output_dtype)


def prepare_output(path: Path, size: int, force: bool) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    conflicts = [candidate for candidate in (path, partial) if candidate.exists()]
    if conflicts and not force:
        joined = ", ".join(str(candidate) for candidate in conflicts)
        raise ConversionError(f"output already exists: {joined}; pass --force to replace it")
    if force:
        for candidate in conflicts:
            candidate.unlink()
    with partial.open("xb") as stream:
        stream.truncate(size)
    return partial


def write_dense_bva(
    source: Any,
    tensors: Sequence[TensorSpec],
    buckets: Sequence[BucketSpec],
    summary: LayoutSummary,
    output_path: Path,
    chunk_elements: int,
    force: bool,
    storage_dtype: str,
) -> None:
    output_dtype, item_size, quantization_min, quantization_max = STORAGE_INFO[
        storage_dtype
    ]
    partial = prepare_output(
        output_path, summary.logical_entries * item_size, force
    )
    output = np.memmap(
        partial, dtype=output_dtype, mode="r+", shape=(summary.logical_entries,)
    )
    started = time.monotonic()
    processed = 0
    next_report = 0.05
    try:
        for bucket in buckets:
            tensor = tensors[bucket.tensor_index]
            if tensor.conversion == "symmetric_per_tensor" and tensor.scale is None:
                raise ConversionError(f"tensor {tensor.name!r} has no quantization scale")
            raw = source.tensor_view(tensor).reshape(tuple(tensor.matrix_shape))
            destination = output[
                bucket.dense_offset : bucket.dense_offset + bucket.entry_count
            ].reshape((bucket.row_len, bucket.col_len))
            physical_cols = bucket.physical_col_len
            rows_per_chunk = max(1, chunk_elements // max(physical_cols, 1))
            for local_start in range(0, bucket.physical_row_len, rows_per_chunk):
                local_end = min(bucket.physical_row_len, local_start + rows_per_chunk)
                source_block = raw[
                    bucket.row_offset + local_start : bucket.row_offset + local_end,
                    bucket.col_offset : bucket.col_offset + physical_cols,
                ]
                if tensor.conversion == "identity_i8":
                    destination[local_start:local_end, :physical_cols] = source_block
                elif tensor.conversion == BF16_BITCAST_CONVERSION:
                    if storage_dtype != "i16" or tensor.source_dtype != "BF16":
                        raise ConversionError(
                            f"tensor {tensor.name!r} has an invalid BF16 bitcast configuration"
                        )
                    destination[local_start:local_end, :physical_cols] = (
                        bitcast_bf16_to_i16(source_block)
                    )
                elif tensor.conversion == "symmetric_per_tensor":
                    float_block = as_float32(source_block, tensor.source_dtype)
                    destination[local_start:local_end, :physical_cols] = quantize_block(
                        float_block,
                        tensor.scale,
                        output_dtype,
                        quantization_min,
                        quantization_max,
                    )
                else:
                    raise ConversionError(
                        f"tensor {tensor.name!r} has unknown conversion mode "
                        f"{tensor.conversion!r}"
                    )
            processed += bucket.physical_entry_count
            ratio = processed / summary.physical_entries
            if ratio >= next_report or processed == summary.physical_entries:
                output.flush()
                elapsed = time.monotonic() - started
                rate = processed / max(elapsed, 1e-9) / (1 << 20)
                print(
                    f"write_progress={ratio:.1%} processed={processed} "
                    f"rate_melements_per_second={rate:.1f}",
                    flush=True,
                )
                next_report += 0.05
        output.flush()
    except BaseException:
        del output
        raise
    del output
    os.replace(partial, output_path)


def scalar_quantize(value: float, scale: float, storage_dtype: str) -> int:
    output_dtype, _, quantization_min, quantization_max = STORAGE_INFO[
        storage_dtype
    ]
    encoded = np.empty(1, dtype=np.float32)
    encoded[0] = np.float32(value)
    return int(
        quantize_block(
            encoded,
            scale,
            output_dtype,
            quantization_min,
            quantization_max,
        )[0]
    )


def verify_layout_samples(
    source: Any,
    bva_path: Path,
    tensors: Sequence[TensorSpec],
    buckets: Sequence[BucketSpec],
    summary: LayoutSummary,
    storage_dtype: str,
) -> int:
    output_dtype, item_size, _, _ = STORAGE_INFO[storage_dtype]
    expected_size = summary.logical_entries * item_size
    if bva_path.stat().st_size != expected_size:
        raise ConversionError(
            f"BVA file has {bva_path.stat().st_size} bytes, "
            f"expected {expected_size}"
        )
    output = np.memmap(
        bva_path, dtype=output_dtype, mode="r", shape=(summary.logical_entries,)
    )
    checked = 0
    for bucket in buckets:
        tensor = tensors[bucket.tensor_index]
        if tensor.conversion == "symmetric_per_tensor" and tensor.scale is None:
            raise ConversionError(f"tensor {tensor.name!r} has no quantization scale")
        raw = source.tensor_view(tensor).reshape(tuple(tensor.matrix_shape))
        coordinates = {
            (0, 0),
            (bucket.physical_row_len - 1, bucket.physical_col_len - 1),
            (
                (bucket.physical_row_len - 1) // 2,
                (bucket.physical_col_len - 1) // 2,
            ),
        }
        for local_row, local_col in coordinates:
            source_sample = raw[
                bucket.row_offset + local_row : bucket.row_offset + local_row + 1,
                bucket.col_offset + local_col : bucket.col_offset + local_col + 1,
            ]
            position = bucket.dense_offset + local_row * bucket.col_len + local_col
            if tensor.conversion == BF16_BITCAST_CONVERSION:
                expected_bits = int(
                    np.asarray(source_sample, dtype=np.dtype("<u2"))[0, 0]
                )
                actual_bits = int(output[position : position + 1].view("<u2")[0])
                if actual_bits != expected_bits:
                    raise ConversionError(
                        f"BF16 bit mismatch in {tensor.name!r} at "
                        f"({bucket.row_offset + local_row}, "
                        f"{bucket.col_offset + local_col}): "
                        f"0x{actual_bits:04x} != 0x{expected_bits:04x}"
                    )
                checked += 1
                continue
            if tensor.conversion == "identity_i8":
                expected = int(source_sample[0, 0])
            elif tensor.conversion == "symmetric_per_tensor":
                expected = scalar_quantize(
                    float(as_float32(source_sample, tensor.source_dtype)[0, 0]),
                    tensor.scale,
                    storage_dtype,
                )
            else:
                raise ConversionError(
                    f"tensor {tensor.name!r} has unknown conversion mode "
                    f"{tensor.conversion!r}"
                )
            actual = int(output[position])
            if actual != expected:
                raise ConversionError(
                    f"sample mismatch in {tensor.name!r} at "
                    f"({bucket.row_offset + local_row}, "
                    f"{bucket.col_offset + local_col}): {actual} != {expected}"
                )
            checked += 1

        destination = output[
            bucket.dense_offset : bucket.dense_offset + bucket.entry_count
        ].reshape((bucket.row_len, bucket.col_len))
        if bucket.physical_col_len < bucket.col_len and bool(
            np.any(destination[: bucket.physical_row_len, bucket.physical_col_len :])
        ):
            raise ConversionError(f"nonzero right padding in bucket {bucket.index}")
        if bucket.physical_row_len < bucket.row_len and bool(
            np.any(destination[bucket.physical_row_len :, :])
        ):
            raise ConversionError(f"nonzero bottom padding in bucket {bucket.index}")

    chunk = 64 << 20
    for start in range(summary.active_entries, summary.logical_entries, chunk):
        if bool(np.any(output[start : min(summary.logical_entries, start + chunk)])):
            raise ConversionError("the final BVA zero suffix contains nonzero bytes")
    del output
    return checked


def sha256_file(path: Path, chunk_bytes: int = 16 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def source_checksums(source: Any) -> tuple[list[dict[str, Any]], str]:
    records: list[dict[str, Any]] = []
    for path in source.source_paths():
        records.append(
            {
                "name": path.name,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if len(records) == 1:
        aggregate = records[0]["sha256"]
    else:
        canonical = json.dumps(
            [
                {
                    "name": record["name"],
                    "size_bytes": record["size_bytes"],
                    "sha256": record["sha256"],
                }
                for record in records
            ],
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        aggregate = hashlib.sha256(canonical).hexdigest()
    return records, aggregate


def make_manifest(
    source_path: Path,
    source: Any,
    model_family: str,
    bva_path: Path,
    tensors: Sequence[TensorSpec],
    excluded: Sequence[Mapping[str, Any]],
    buckets: Sequence[BucketSpec],
    summary: LayoutSummary,
    source_files: Sequence[Mapping[str, Any]],
    source_sha256: str,
    bva_sha256: str,
    sample_count: int,
    storage_dtype: str,
) -> dict[str, Any]:
    output_dtype, item_size, quantization_min, quantization_max = STORAGE_INFO[
        storage_dtype
    ]
    storage_limits = np.iinfo(output_dtype)
    native_i8_tensors = sum(tensor.conversion == "identity_i8" for tensor in tensors)
    converted_tensors = sum(
        tensor.conversion == "symmetric_per_tensor" for tensor in tensors
    )
    bitcast_bf16_tensors = sum(
        tensor.conversion == BF16_BITCAST_CONVERSION for tensor in tensors
    )
    if model_family == "gpt2":
        selection_policy = "gpt2_trainable_parameters"
    elif model_family == "llama-w8a8":
        selection_policy = "llama_w8a8_weights_without_public_weight_scales"
    else:
        selection_policy = "llama_bf16_parameter_weights"
    storage_metadata = {
        "storage_dtype": storage_dtype,
        "storage_encoding": "signed_twos_complement_little_endian",
        "storage_item_bytes": item_size,
        "bit_width": item_size * 8,
    }
    if model_family == "llama-bf16":
        if (
            storage_dtype != "i16"
            or native_i8_tensors != 0
            or converted_tensors != 0
            or bitcast_bf16_tensors != len(tensors)
        ):
            raise ConversionError("invalid Llama BF16 bitcast manifest state")
        quantization_metadata = {
            "scheme": BF16_BITCAST_SCHEME,
            "native_i8_tensor_count": 0,
            "converted_float_tensor_count": 0,
            "bitcast_bf16_tensor_count": bitcast_bf16_tensors,
            "bit_preserving": True,
            "numeric_conversion": False,
            "source_to_storage": BF16_SOURCE_TO_STORAGE,
            "source_storage_encoding": "bf16_bit_pattern_little_endian",
            "field_interpretation": "signed_twos_complement_i16",
            **storage_metadata,
        }
    else:
        quantization_scheme = (
            "symmetric_per_tensor"
            if native_i8_tensors == 0
            else "native_i8_plus_symmetric_per_tensor"
        )
        quantization_metadata = {
            "scheme": quantization_scheme,
            "native_i8_tensor_count": native_i8_tensors,
            "converted_float_tensor_count": converted_tensors,
            "bitcast_bf16_tensor_count": 0,
            "zero_point": 0,
            **storage_metadata,
            "storage_qmin": int(storage_limits.min),
            "storage_qmax": int(storage_limits.max),
            "float_conversion_qmin": quantization_min,
            "float_conversion_qmax": quantization_max,
            "rounding": "nearest_ties_to_even",
            "float_conversion_formula": (
                "q=clip(round(value/scale),"
                f"{quantization_min},{quantization_max})"
            ),
            "native_i8_formula": "q=source_i8",
        }
    return {
        "format": f"parcomm-dense-{storage_dtype}-bva",
        "version": FORMAT_VERSION,
        "source": {
            "path": str(source_path.resolve()),
            "format": "safetensors" if len(source_files) == 1 else "sharded_safetensors",
            "size_bytes": sum(int(record["size_bytes"]) for record in source_files),
            "sha256": source_sha256,
            "sha256_definition": (
                "file_bytes"
                if len(source_files) == 1
                else "sha256_of_canonical_filename_size_and_sha256_records"
            ),
            "files": list(source_files),
            "metadata": source.metadata(),
            "model_family": model_family,
        },
        "selection": {
            "policy": selection_policy,
            "tensor_order": "lexicographic_name",
            "excluded_tensors": list(excluded),
        },
        "quantization": quantization_metadata,
        "layout": {
            **asdict(summary),
            "splitting_budget": SPLITTING_BUDGET,
            "bucket_order": (
                "entry_count_desc_then_tensor_index_then_row_offset_then_col_offset"
            ),
            "within_bucket_order": "row_major",
            "rank1_matrix_mapping": "shape_n_to_n_by_1",
            "element_dtype": storage_dtype,
            "element_bytes": item_size,
            "bva_path": str(bva_path.resolve()),
            "bva_size_bytes": bva_path.stat().st_size,
            "bva_sha256": bva_sha256,
        },
        "tensors": [asdict(tensor) for tensor in tensors],
        "buckets": [bucket.manifest_record() for bucket in buckets],
        "verification": {
            "layout_sample_count": sample_count,
            "internal_padding_zero": True,
            "final_suffix_zero": True,
            "file_size_matches_logical_entries": True,
        },
    }


def write_json_atomic(path: Path, value: Mapping[str, Any], force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    conflicts = [candidate for candidate in (path, partial) if candidate.exists()]
    if conflicts and not force:
        joined = ", ".join(str(candidate) for candidate in conflicts)
        raise ConversionError(f"manifest already exists: {joined}; pass --force to replace it")
    if force:
        for candidate in conflicts:
            candidate.unlink()
    with partial.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def inspect_checkpoint(
    source_path: Path, model_family: str, storage_dtype: str
) -> tuple[
    list[TensorSpec], list[dict[str, Any]], list[BucketSpec], LayoutSummary
]:
    validate_storage_mode(model_family, storage_dtype)
    with open_checkpoint(source_path, model_family) as source:
        tensors, excluded = parse_tensor_specs(source, model_family)
        buckets, summary = build_layout(tensors)
    print(f"source={source_path}")
    print(f"model_family={model_family}")
    print(f"selected_tensors={summary.tensor_count}")
    print(f"excluded_tensors={len(excluded)}")
    print(
        "native_i8_tensors="
        f"{sum(tensor.conversion == 'identity_i8' for tensor in tensors)}"
    )
    print(
        "converted_float_tensors="
        f"{sum(tensor.conversion == 'symmetric_per_tensor' for tensor in tensors)}"
    )
    print(
        "bitcast_bf16_tensors="
        f"{sum(tensor.conversion == BF16_BITCAST_CONVERSION for tensor in tensors)}"
    )
    print(f"bucket_count={summary.bucket_count}")
    print(f"physical_entries={summary.physical_entries}")
    print(f"active_entries={summary.active_entries}")
    print(f"logical_entries={summary.logical_entries}")
    item_size = STORAGE_INFO[storage_dtype][1]
    bva_size_bytes = summary.logical_entries * item_size
    print(f"storage_dtype={storage_dtype}")
    print(f"bva_size_bytes={bva_size_bytes}")
    print(f"bva_size_gib={bva_size_bytes / (1 << 30):.6f}")
    print(f"boundary_padding_entries={summary.boundary_padding_entries}")
    print(f"final_padding_entries={summary.final_padding_entries}")
    return tensors, excluded, buckets, summary


def convert_checkpoint(arguments: argparse.Namespace) -> None:
    source_path = arguments.input.resolve()
    model_family = arguments.model_family
    storage_dtype = arguments.storage_dtype
    validate_storage_mode(model_family, storage_dtype)
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if arguments.name:
        prefix = arguments.name
    elif model_family == "gpt2":
        prefix = "gpt2-xl"
    elif model_family == "llama-w8a8":
        prefix = "llama-3.1-8b-w8a8"
    else:
        prefix = "llama-3.1-8b-bf16-bits"
    bva_path = output_dir / f"{prefix}.bva.{storage_dtype}"
    manifest_path = output_dir / f"{prefix}.manifest.json"
    chunk_elements = max(1, (arguments.chunk_mib << 20) // 4)

    started = time.monotonic()
    with open_checkpoint(source_path, model_family) as source:
        tensors, excluded = parse_tensor_specs(source, model_family)
        buckets, summary = build_layout(tensors)
        print(f"model_family={model_family}")
        print(f"selected_tensors={summary.tensor_count}")
        print(f"excluded_tensors={len(excluded)}")
        print(
            "native_i8_tensors="
            f"{sum(tensor.conversion == 'identity_i8' for tensor in tensors)}"
        )
        print(
            "converted_float_tensors="
            f"{sum(tensor.conversion == 'symmetric_per_tensor' for tensor in tensors)}"
        )
        print(
            "bitcast_bf16_tensors="
            f"{sum(tensor.conversion == BF16_BITCAST_CONVERSION for tensor in tensors)}"
        )
        print(f"physical_entries={summary.physical_entries}")
        print(f"bucket_count={summary.bucket_count}")
        print(f"logical_entries={summary.logical_entries}")
        print(f"storage_dtype={storage_dtype}")
        print(
            f"bva_size_bytes={summary.logical_entries * STORAGE_INFO[storage_dtype][1]}"
        )
        print("stage=source_sha256", flush=True)
        source_files, source_sha256 = source_checksums(source)
        if (
            arguments.expected_source_sha256
            and source_sha256.lower() != arguments.expected_source_sha256.lower()
        ):
            raise ConversionError(
                f"source SHA-256 {source_sha256} does not match expected "
                f"{arguments.expected_source_sha256}"
            )
        print(f"source_sha256={source_sha256}")
        scale_tensors = [
            tensor
            for tensor in tensors
            if tensor.conversion != BF16_BITCAST_CONVERSION
        ]
        if scale_tensors:
            print("stage=quantization_scales", flush=True)
            compute_quantization_scales(
                source,
                scale_tensors,
                chunk_elements,
                STORAGE_INFO[storage_dtype][3],
            )
        else:
            print("stage=bf16_bitcast", flush=True)
            print("quantization_scale_pass=skipped", flush=True)
        print("stage=write_bva", flush=True)
        write_dense_bva(
            source,
            tensors,
            buckets,
            summary,
            bva_path,
            chunk_elements,
            arguments.force,
            storage_dtype,
        )
        print("stage=verify_layout", flush=True)
        sample_count = verify_layout_samples(
            source, bva_path, tensors, buckets, summary, storage_dtype
        )
        print(f"verified_samples={sample_count}")
        print("stage=bva_sha256", flush=True)
        bva_sha256 = sha256_file(bva_path)
        print(f"bva_sha256={bva_sha256}")
        manifest = make_manifest(
            source_path,
            source,
            model_family,
            bva_path,
            tensors,
            excluded,
            buckets,
            summary,
            source_files,
            source_sha256,
            bva_sha256,
            sample_count,
            storage_dtype,
        )
    write_json_atomic(manifest_path, manifest, arguments.force)
    elapsed = time.monotonic() - started
    print(f"bva_path={bva_path}")
    print(f"manifest_path={manifest_path}")
    print(f"elapsed_seconds={elapsed:.6f}")
    print("conversion_verified=true")


def validate_manifest_conversion_metadata(
    manifest: Mapping[str, Any],
    model_family: str,
    tensors: Sequence[TensorSpec],
) -> None:
    if model_family != "llama-bf16":
        return
    quantization = manifest.get("quantization")
    if not isinstance(quantization, Mapping):
        raise ConversionError("manifest quantization metadata is missing")
    expected = {
        "scheme": BF16_BITCAST_SCHEME,
        "native_i8_tensor_count": 0,
        "converted_float_tensor_count": 0,
        "bitcast_bf16_tensor_count": len(tensors),
        "bit_preserving": True,
        "numeric_conversion": False,
        "source_to_storage": BF16_SOURCE_TO_STORAGE,
        "source_storage_encoding": "bf16_bit_pattern_little_endian",
        "field_interpretation": "signed_twos_complement_i16",
    }
    for name, value in expected.items():
        if quantization.get(name) != value:
            raise ConversionError(
                f"manifest quantization.{name} does not match BF16 bitcast semantics"
            )
    forbidden = {
        "zero_point",
        "float_conversion_qmin",
        "float_conversion_qmax",
        "rounding",
        "float_conversion_formula",
    }
    present = sorted(forbidden.intersection(quantization))
    if present:
        raise ConversionError(
            "BF16 bitcast manifest contains numeric conversion metadata: "
            + ", ".join(present)
        )


def verify_artifacts(arguments: argparse.Namespace) -> None:
    with arguments.manifest.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("version") not in {LEGACY_FORMAT_VERSION, FORMAT_VERSION}:
        raise ConversionError("unsupported ParComm manifest format")
    storage_dtype = manifest.get("quantization", {}).get("storage_dtype")
    if storage_dtype not in STORAGE_INFO:
        raise ConversionError(f"unsupported storage dtype {storage_dtype!r}")
    if manifest.get("format") != f"parcomm-dense-{storage_dtype}-bva":
        raise ConversionError("manifest format does not match its storage dtype")
    source_path = arguments.input.resolve()
    bva_path = arguments.bva.resolve()
    model_family = manifest.get("source", {}).get("model_family")
    if model_family not in {"gpt2", "llama-w8a8", "llama-bf16"}:
        raise ConversionError(f"unsupported manifest model family {model_family!r}")
    if model_family == "llama-bf16" and manifest.get("version") != FORMAT_VERSION:
        raise ConversionError("Llama BF16 bitcast manifests must use format version 2")
    validate_storage_mode(model_family, storage_dtype)
    with open_checkpoint(source_path, model_family) as source:
        tensors, excluded = parse_tensor_specs(source, model_family)
        buckets, summary = build_layout(tensors)
        validate_manifest_conversion_metadata(manifest, model_family, tensors)
        manifest_tensors = manifest.get("tensors", [])
        if len(manifest_tensors) != len(tensors):
            raise ConversionError("manifest tensor count does not match the checkpoint")
        for tensor, record in zip(tensors, manifest_tensors):
            if tensor.name != record.get("name"):
                raise ConversionError("manifest tensor ordering does not match the checkpoint")
            if tensor.conversion != record.get("conversion", tensor.conversion):
                raise ConversionError(
                    f"manifest conversion mode does not match tensor {tensor.name!r}"
                )
            if tensor.conversion == BF16_BITCAST_CONVERSION:
                if record.get("max_abs") is not None or record.get("scale") is not None:
                    raise ConversionError(
                        f"BF16 bitcast tensor {tensor.name!r} contains numeric scale metadata"
                    )
            else:
                tensor.max_abs = float(record["max_abs"])
                tensor.scale = float(record["scale"])
        expected_excluded = [record["name"] for record in excluded]
        actual_excluded = [
            record["name"]
            for record in manifest.get("selection", {}).get("excluded_tensors", [])
        ]
        if expected_excluded != actual_excluded:
            raise ConversionError("manifest exclusion list does not match the checkpoint")
        sample_count = verify_layout_samples(
            source, bva_path, tensors, buckets, summary, storage_dtype
        )
        _, actual_source_hash = source_checksums(source)
    expected_source_hash = manifest["source"]["sha256"]
    if actual_source_hash != expected_source_hash:
        raise ConversionError(
            f"source SHA-256 {actual_source_hash} does not match manifest "
            f"{expected_source_hash}"
        )
    expected_hash = manifest["layout"]["bva_sha256"]
    actual_hash = sha256_file(bva_path)
    if actual_hash != expected_hash:
        raise ConversionError(
            f"BVA SHA-256 {actual_hash} does not match manifest {expected_hash}"
        )
    print(f"verified_samples={sample_count}")
    print(f"source_sha256={actual_source_hash}")
    print(f"bva_sha256={actual_hash}")
    print("conversion_verified=true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect", help="inspect a checkpoint and compute its ParComm BVA layout"
    )
    inspect_parser.add_argument("--input", type=Path, required=True)
    inspect_parser.add_argument(
        "--model-family",
        choices=("gpt2", "llama-w8a8", "llama-bf16"),
        default="gpt2",
    )
    inspect_parser.add_argument(
        "--storage-dtype", choices=tuple(STORAGE_INFO), default="i8"
    )

    convert_parser = subparsers.add_parser(
        "convert", help="convert checkpoint parameters into a dense ParComm integer BVA"
    )
    convert_parser.add_argument("--input", type=Path, required=True)
    convert_parser.add_argument(
        "--model-family",
        choices=("gpt2", "llama-w8a8", "llama-bf16"),
        default="gpt2",
    )
    convert_parser.add_argument(
        "--storage-dtype", choices=tuple(STORAGE_INFO), default="i8"
    )
    convert_parser.add_argument("--output-dir", type=Path, required=True)
    convert_parser.add_argument("--name")
    convert_parser.add_argument(
        "--chunk-mib",
        type=int,
        default=64,
        help="maximum working chunk size in MiB (default: 64)",
    )
    convert_parser.add_argument("--expected-source-sha256")
    convert_parser.add_argument("--force", action="store_true")

    verify_parser = subparsers.add_parser(
        "verify", help="verify an existing BVA against its source checkpoint"
    )
    verify_parser.add_argument("--input", type=Path, required=True)
    verify_parser.add_argument("--bva", type=Path, required=True)
    verify_parser.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "inspect":
            inspect_checkpoint(
                arguments.input.resolve(),
                arguments.model_family,
                arguments.storage_dtype,
            )
        elif arguments.command == "convert":
            if arguments.chunk_mib <= 0:
                raise ConversionError("--chunk-mib must be positive")
            convert_checkpoint(arguments)
        elif arguments.command == "verify":
            verify_artifacts(arguments)
        else:
            parser.error(f"unsupported command {arguments.command!r}")
    except (ConversionError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
