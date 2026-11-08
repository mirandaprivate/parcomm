"""Exercise the actual stdin -> serialized bundle -> fresh verifier boundary."""
import json
from pathlib import Path
import subprocess
import tempfile

binary = Path(__file__).resolve().parents[1] / "target/release/parcomm-qafold"
with tempfile.TemporaryDirectory(prefix="parcomm-qafold-cli-") as temp:
    root = Path(temp)
    description = root / "input.json"
    description.write_text(json.dumps({
        "format": "parcomm-dense-bva-stream-description", "storage_dtype": "i8",
        "shapes": [[32, 32]], "stream_bytes": 1024,
        "layout": {"logical_entries": 1024},
    }))
    data = bytes((index * 73) % 256 for index in range(1024))
    def prove(output, witness):
        return subprocess.run([str(binary), "prove", "--description", str(description),
                               "--input", "-", "--output", str(output), "--threads", "2"],
                              input=witness, capture_output=True)
    output = root / "valid"
    result = prove(output, data)
    assert result.returncode == 0, result.stderr.decode()
    # This process has no input data, model path or previous prover state.
    verified = subprocess.run([str(binary), "verify", "--output", str(output), "--threads", "1"],
                              capture_output=True)
    assert verified.returncode == 0, verified.stderr.decode()
    assert json.loads(verified.stdout)["verified"]
    assert prove(root / "short", data[:-1]).returncode != 0
    assert prove(root / "long", data + b"\0").returncode != 0
    # Existing result directories cannot be overwritten by another prove call.
    assert prove(output, data).returncode != 0
    proof = output / "proof.bin"
    damaged = bytearray(proof.read_bytes())
    damaged[-1] ^= 1
    proof.write_bytes(damaged)
    rejected = subprocess.run([str(binary), "verify", "--output", str(output)], capture_output=True)
    assert rejected.returncode != 0
    assert json.loads((output / "verification.json").read_text())["verified"] is False
print("PASS: streamed input, complete proving, fresh-process verify, truncated/excess input, overwrite prevention, tampered proof rejection")
