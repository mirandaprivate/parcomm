#!/usr/bin/env python3
"""CPU artifact entry point. Run from any directory; never overwrite an output."""
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    '8B-8bit': ('f65521', 'llama-w8a8', 'i8', 4),
    '8B-16bit': ('koalabear', 'llama-bf16', 'i16', 4),
    '70B-8bit': ('f65521', 'llama-w8a8', 'i8', 6),
    '70B-16bit': ('koalabear', 'llama-bf16', 'i16', 6),
}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['build', 'test', 'smoke', 'model'])
    p.add_argument('--backend', choices=['jolt', 'qafold'], required=True)
    p.add_argument('--case', choices=CASES, default='8B-8bit')
    p.add_argument('--input', type=Path, help='Local safetensors checkpoint directory')
    p.add_argument('--output', type=Path, required=True, help='New directory for commands and outputs')
    p.add_argument('--threads', type=int, help='Default: smoke 2, Jolt model 64, QAFold model 48')
    p.add_argument('--target-dir', type=Path, default=ROOT/'build')
    a = p.parse_args()
    if a.action == 'model' and (a.input is None or not a.input.is_dir()): p.error('model requires an existing --input directory')
    threads = a.threads or (2 if a.action == 'smoke' else 64 if a.backend == 'jolt' else 48)
    if threads < 1: p.error('--threads must be positive')
    out = a.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    field, family, dtype, packed = CASES[a.case]
    source = ROOT if a.backend == 'jolt' else ROOT/'qafold'/field
    target = a.target_dir.resolve()/('jolt' if a.backend == 'jolt' else field)
    env = os.environ.copy(); env.update(CARGO_TARGET_DIR=str(target), RAYON_NUM_THREADS=str(threads))
    if a.backend == 'qafold': env['RUSTFLAGS'] = '-C target-cpu=native'
    toolchain = 'nightly-2026-03-21' if a.backend == 'jolt' else '1.97.1'
    config = ROOT/'.cargo'/('jolt-vendor-config.toml' if a.backend == 'jolt' else 'qafold-vendor-config.toml')
    mirror = ROOT/('dependencies-jolt' if a.backend == 'jolt' else 'dependencies')
    offline = config.is_file() and mirror.is_dir()
    cargo = ['cargo', '+'+toolchain]
    if offline: cargo += ['--config', str(config)]
    if a.backend == 'jolt': cargo += ['--config', str(ROOT/'.cargo/jolt-patch.toml')]
    common = ['--release'] + (['--offline'] if offline else []) + ['--locked', '--manifest-path', str(source/'Cargo.toml')]
    history = []
    def record(cmd):
        history.append(cmd); (out/'commands.json').write_text(json.dumps(history, indent=2)+'\n')
    def run(cmd, label):
        record(cmd)
        with (out/(label+'.log')).open('w') as log:
            subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    (out/'configuration.json').write_text(json.dumps(dict(backend=a.backend,case=a.case,threads=threads,toolchain=toolchain,runs=1,input=str(a.input),rustflags=env.get('RUSTFLAGS','')),indent=2)+'\n')
    if a.backend == 'jolt':
        if a.action == 'test':
            for test in ['packed_eo_short_int','matrix_collection_e2e']:
                run(cargo+['test']+common+['-p','batch','--test',test,'--','--test-threads=1'],test)
        elif a.action == 'build': run(cargo+['build']+common+['--workspace'],'build')
        else:
            bench = 'experiment' if a.action == 'smoke' else 'experiment_llama'+a.case.replace('-','_')
            bench_common=[flag for flag in common if flag != '--release']
            run(cargo+['bench','--no-run']+bench_common+['-p','batch','--bench',bench],'build')
            args = ['homogeneous','--matrix-count','2','--rows','31','--cols','37'] if a.action == 'smoke' else ['--input',str(a.input.resolve()),'--converter',str(source/'preprocessing/modelconvert.py')]
            if threads != 64: args.append('--allow-non-64-threads')
            # Compile separately. Cargo's small invocation overhead remains in wall time only.
            run(['/usr/bin/time','-v']+cargo+['bench']+bench_common+['-p','batch','--bench',bench,'--']+args,'run')
            assert 'verified_marker=SECTION6_E2E_VERIFIED' in (out/'run.log').read_text(), 'Missing end-to-end acceptance marker'
    else:
        name = 'parcomm-qafold' if field == 'f65521' else 'parcomm-qafold-koalabear-8b'
        if a.action == 'test':
            run(cargo+['test']+common,'tests')
            run(cargo+['test','--release']+(['--offline'] if offline else [])+['--locked','--manifest-path',str(source/'vendor/qafold/Cargo.toml'),'--features','profile-encoding'],'backend-tests')
        else:
            run(cargo+['build']+common+['--bin',name],'build')
            if a.action != 'build':
                binary=str(target/'release'/name); bundle=out/'bundle'
                if a.action == 'smoke':
                    run([binary,'demo','--output',str(bundle),'--threads',str(threads),'--chunk-log','10','--packed-log','4','--cache-mode','recompute'],'run')
                else:
                    converter=[sys.executable,str(source/'modelconvert.py')]
                    args=['--input',str(a.input.resolve()),'--model-family',family,'--storage-dtype',dtype]
                    desc=converter+['describe-json']+args; record(desc)
                    with (out/'description.json').open('w') as f: subprocess.run(desc,stdout=f,check=True,env=env)
                    cmd=['/usr/bin/time','-v','-o',str(out/'resources.log'),binary,'prove','--description',str(out/'description.json'),'--input','-','--output',str(bundle),'--threads',str(threads),'--chunk-log','30','--packed-log',str(packed),'--cache-mode','recompute']
                    convert=converter+['stream']+args; record(cmd); record(convert)
                    h=hashlib.sha256(); count=0; start=time.monotonic()
                    with (out/'run.log').open('w') as log,(out/'stages.log').open('w') as err,(out/'conversion.log').open('w') as cerr:
                        prover=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=log,stderr=err,env=env)
                        producer=subprocess.Popen(convert,stdout=subprocess.PIPE,stderr=cerr,env=env)
                        try:
                            for block in iter(lambda:producer.stdout.read(8*1024*1024),b''):
                                h.update(block); count+=len(block); prover.stdin.write(block)
                            producer.stdout.close(); assert producer.wait()==0
                            prover.stdin.close(); assert prover.wait()==0
                        finally:
                            for proc in [producer,prover]:
                                if proc.poll() is None: proc.terminate(); proc.wait()
                    description=json.loads((out/'description.json').read_text())
                    assert count==description['stream_bytes']
                    (out/'input-audit.json').write_text(json.dumps(dict(bytes=count,sha256=h.hexdigest(),pipeline_wall_seconds=time.monotonic()-start),indent=2)+'\n')
                run([binary,'verify','--output',str(bundle),'--threads','1'],'verify')
                r=json.loads((bundle/'result.json').read_text()); v=json.loads((bundle/'verification.json').read_text())
                assert r['verified'] and v['verified'] and r['cache_mode']=='recompute'
                assert v['profile']['native_openings']==1
                statement=json.loads((bundle/'commitment.json').read_text())
                assert v['profile']['authenticated_roots']==len(statement['roots'])
    (out/'status.txt').write_text('success\n')
    print('SUCCESS:',out)

if __name__ == '__main__': main()
