"""Exercise the actual little-endian i16 input path, including malformed lengths."""
import json
import os
from pathlib import Path
import struct
import subprocess

root = Path(__file__).resolve().parents[1]
target = Path(os.environ.get('CARGO_TARGET_DIR', root / 'target'))
binary = target / 'release/pcs'
command = [str(binary), '10', '10', '1024', '4']
payload = b''.join(struct.pack('<h', value) for value in [-32768, -1, 0, 32767] * 256)
ok = subprocess.run(command, input=payload, capture_output=True, check=True)
records = [json.loads(line) for line in ok.stdout.splitlines()]
assert records[-1]['stage'] == 'complete' and records[-1]['verified']
assert records[0]['input_bytes'] == 2048
for label, bad in [('truncated', payload[:-1]), ('excess', payload+b'\0')]:
    result = subprocess.run(command, input=bad, capture_output=True)
    assert result.returncode != 0, label
    assert b'"stage":"complete"' not in result.stdout, label
print('PASS: signed i16 limits, actual stdin, truncated and excess inputs')
