#!/usr/bin/env python3
"""Run each requested native CPU PCS model once, preserving all evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

CASES = {
    '8B-8bit': ('Llama-3.1-8B-W8A8', 'llama-w8a8', 'i8', 8053329920, '5b3bccda178bea933b34100f2112fad6819d281be85cc9a9131728e4d883db5d'),
    '8B-16bit': ('Llama-3.1-8B-BF16', 'llama-bf16', 'i16', 8053329920, '4a402790ceb49665c11d0f1466f0893b3b32d9445740756ee7eaf1a65c8d1c5f'),
    '70B-8bit': ('Llama-3.1-70B-W8A8', 'llama-w8a8', 'i8', 70599843840, '1831cb4b5de22794fb7a98f0de0a54fc303738d572ca17934bc695cdee4bc6f8'),
    '70B-16bit': ('Llama-3.1-70B-BF16-community', 'llama-bf16', 'i16', 70599843840, '0aa5428f88d876241ce08db0a16c5a1d7a395cf6fde2a563052dc02a9ad6c7c3'),
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--scratch', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--converter', type=Path, required=True)
    parser.add_argument('--case', choices=CASES, nargs='+', default=list(CASES))
    parser.add_argument('--threads', type=int, default=48)
    parser.add_argument('--matrix-log', type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.scratch.mkdir(parents=True, exist_ok=True)
    history = []
    def record(cmd):
        history.append([str(x) for x in cmd])
        (args.output/'commands.json').write_text(json.dumps(history, indent=2)+'\n')
    (args.output/'configuration.json').write_text(json.dumps(vars(args), default=str, indent=2)+'\n')
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads), OMP_PROC_BIND='spread', OMP_PLACES='cores')
    for name in args.case:
        model, family, dtype, entries, expected_sha = CASES[name]
        width = 1 if dtype == 'i8' else 2
        out = args.output/name
        out.mkdir()
        data = args.scratch/(name+'.input')
        if data.exists(): raise RuntimeError(f'refusing to overwrite {data}')
        if shutil.disk_usage(args.scratch).free < entries*width + 2*(1 << 30):
            raise RuntimeError('scratch does not have enough free space for compact input')
        cmd = [sys.executable, str(args.converter), 'stream', '--input', str(args.models/model), '--model-family', family, '--storage-dtype', dtype]
        record(cmd)
        digest = hashlib.sha256(); count = 0; start = time.monotonic()
        print(f'{name}: input conversion started', flush=True)
        with data.open('xb') as dst, (out/'conversion.log').open('w') as log:
            producer = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=log)
            try:
                for chunk in iter(lambda: producer.stdout.read(8*1024*1024), b''):
                    dst.write(chunk); digest.update(chunk); count += len(chunk)
                producer.stdout.close()
                if producer.wait(): raise RuntimeError('converter failed')
            finally:
                if producer.poll() is None: producer.terminate(); producer.wait()
        (out/'input.json').write_text(json.dumps(dict(bytes=count, sha256=digest.hexdigest(), conversion_seconds=time.monotonic()-start, expected_sha256=expected_sha), indent=2)+'\n')
        if count != entries*width or digest.hexdigest() != expected_sha:
            raise RuntimeError('input differs from the previous QAFold workload; no benchmark started')
        prefix = out/'result'
        cmd = ['/usr/bin/time', '-v', '-o', str(out/'resources.log'), str(args.binary.resolve()), str(data), str(width), str(args.threads), str(prefix), str(args.matrix_log)]
        record(cmd)
        print(f'{name}: commit/open/verify started', flush=True)
        with (out/'run.log').open('w') as log, (out/'progress.log').open('w') as err:
            subprocess.run(cmd, stdout=log, stderr=err, env=env, check=True)
        cmd = [str(args.binary.resolve()), 'verify', str(prefix)]
        record(cmd)
        with (out/'independent-verify.json').open('w') as log:
            subprocess.run(cmd, stdout=log, env=dict(env, OMP_NUM_THREADS='1'), check=True)
        if not json.loads((out/'independent-verify.json').read_text())['verified']:
            raise RuntimeError('independent verification failed')
        # Only this script's successfully checked compact input is temporary.
        data.unlink()
        print(f'{name}: completed and independently verified', flush=True)

if __name__ == '__main__': main()
