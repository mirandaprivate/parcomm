from __future__ import annotations

import contextlib
import io
import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from preprocessing import modelconvert


class Bf16BitcastConversionTest(unittest.TestCase):
    def test_bf16_payloads_are_preserved_exactly(self) -> None:
        patterns = np.asarray(
            [
                0x0000,
                0x8000,
                0x3F80,
                0xBF80,
                0x7F80,
                0xFF80,
                0x7FC1,
                0xFFC1,
                0x0001,
                0xFFFF,
            ],
            dtype=np.dtype("<u2"),
        )
        words = np.resize(patterns, 31 * 3).astype(np.dtype("<u2"), copy=False)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            output = root / "output"
            tensor_name = self._write_checkpoint(source, words, [31, 3])
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stdout):
                result = modelconvert.main(
                    [
                        "convert",
                        "--input",
                        str(source),
                        "--model-family",
                        "llama-bf16",
                        "--storage-dtype",
                        "i16",
                        "--output-dir",
                        str(output),
                        "--name",
                        "tiny-bf16-bits",
                        "--chunk-mib",
                        "1",
                    ]
                )
            self.assertEqual(result, 0, stdout.getvalue())
            self.assertIn("quantization_scale_pass=skipped", stdout.getvalue())

            manifest_path = output / "tiny-bf16-bits.manifest.json"
            bva_path = output / "tiny-bf16-bits.bva.i16"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["version"], 2)
            quantization = manifest["quantization"]
            self.assertEqual(quantization["scheme"], "bf16_bitcast_i16")
            self.assertEqual(quantization["bitcast_bf16_tensor_count"], 1)
            self.assertEqual(quantization["converted_float_tensor_count"], 0)
            self.assertTrue(quantization["bit_preserving"])
            self.assertFalse(quantization["numeric_conversion"])
            self.assertEqual(
                quantization["source_to_storage"], "bitcast_bf16_u16_to_i16"
            )
            for numeric_field in (
                "zero_point",
                "float_conversion_qmin",
                "float_conversion_qmax",
                "rounding",
                "float_conversion_formula",
            ):
                self.assertNotIn(numeric_field, quantization)
            tensor = manifest["tensors"][0]
            self.assertEqual(tensor["name"], tensor_name)
            self.assertEqual(tensor["conversion"], "bitcast_bf16_to_i16")
            self.assertIsNone(tensor["max_abs"])
            self.assertIsNone(tensor["scale"])

            actual = np.fromfile(bva_path, dtype=np.dtype("<u2"))
            expected = np.zeros(manifest["layout"]["logical_entries"], dtype="<u2")
            source_matrix = words.reshape((31, 3))
            for bucket in manifest["buckets"]:
                destination = expected[
                    bucket["dense_offset"] : bucket["dense_offset"]
                    + bucket["entry_count"]
                ].reshape((bucket["row_len"], bucket["col_len"]))
                destination[
                    : bucket["physical_row_len"], : bucket["physical_col_len"]
                ] = source_matrix[
                    bucket["row_offset"] : bucket["row_offset"]
                    + bucket["physical_row_len"],
                    bucket["col_offset"] : bucket["col_offset"]
                    + bucket["physical_col_len"],
                ]
            np.testing.assert_array_equal(actual, expected)
            self.assertGreater(manifest["layout"]["boundary_padding_entries"], 0)
            self.assertGreater(manifest["layout"]["final_padding_entries"], 0)

            verify_stdout = io.StringIO()
            with contextlib.redirect_stdout(verify_stdout), contextlib.redirect_stderr(
                verify_stdout
            ):
                verify_result = modelconvert.main(
                    [
                        "verify",
                        "--input",
                        str(source),
                        "--bva",
                        str(bva_path),
                        "--manifest",
                        str(manifest_path),
                    ]
                )
            self.assertEqual(verify_result, 0, verify_stdout.getvalue())
            self.assertIn("conversion_verified=true", verify_stdout.getvalue())

    def test_llama_bf16_rejects_byte_storage(self) -> None:
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            result = modelconvert.main(
                [
                    "inspect",
                    "--input",
                    "unused",
                    "--model-family",
                    "llama-bf16",
                    "--storage-dtype",
                    "i8",
                ]
            )
        self.assertEqual(result, 1)
        self.assertIn("requires --storage-dtype i16", stderr.getvalue())

    def test_legacy_version_one_i8_artifact_still_verifies(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "gpt2.safetensors"
            output = root / "output"
            values = np.asarray(
                [[-1.5, 0.0], [1.0, 3.25]], dtype=np.dtype("<f4")
            )
            self._write_safetensors(
                source,
                "transformer.wte.weight",
                "F32",
                [2, 2],
                values.tobytes(),
            )
            conversion_output = io.StringIO()
            with contextlib.redirect_stdout(conversion_output), contextlib.redirect_stderr(
                conversion_output
            ):
                result = modelconvert.main(
                    [
                        "convert",
                        "--input",
                        str(source),
                        "--model-family",
                        "gpt2",
                        "--storage-dtype",
                        "i8",
                        "--output-dir",
                        str(output),
                        "--name",
                        "legacy-i8",
                    ]
                )
            self.assertEqual(result, 0, conversion_output.getvalue())

            manifest_path = output / "legacy-i8.manifest.json"
            bva_path = output / "legacy-i8.bva.i8"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["version"] = 1
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            verify_output = io.StringIO()
            with contextlib.redirect_stdout(verify_output), contextlib.redirect_stderr(
                verify_output
            ):
                verify_result = modelconvert.main(
                    [
                        "verify",
                        "--input",
                        str(source),
                        "--bva",
                        str(bva_path),
                        "--manifest",
                        str(manifest_path),
                    ]
                )
            self.assertEqual(verify_result, 0, verify_output.getvalue())
            self.assertIn("conversion_verified=true", verify_output.getvalue())

    @staticmethod
    def _write_checkpoint(
        directory: Path, words: np.ndarray, shape: list[int]
    ) -> str:
        directory.mkdir(parents=True)
        tensor_name = "model.layers.0.self_attn.q_proj.weight"
        shard_name = "model-00001-of-00001.safetensors"
        data = np.asarray(words, dtype=np.dtype("<u2")).tobytes()
        Bf16BitcastConversionTest._write_safetensors(
            directory / shard_name, tensor_name, "BF16", shape, data
        )
        index = {
            "metadata": {"total_size": len(data)},
            "weight_map": {tensor_name: shard_name},
        }
        (directory / "model.safetensors.index.json").write_text(
            json.dumps(index), encoding="utf-8"
        )
        return tensor_name

    @staticmethod
    def _write_safetensors(
        path: Path, tensor_name: str, dtype: str, shape: list[int], data: bytes
    ) -> None:
        header = {
            tensor_name: {
                "dtype": dtype,
                "shape": shape,
                "data_offsets": [0, len(data)],
            }
        }
        encoded_header = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded_header += b" " * (-len(encoded_header) % 8)
        path.write_bytes(
            struct.pack("<Q", len(encoded_header)) + encoded_header + data
        )


if __name__ == "__main__":
    unittest.main()
