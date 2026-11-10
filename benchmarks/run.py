#!/usr/bin/env python3
"""Portable CPU benchmark build and synthetic checks; never reruns model timings."""
import argparse,json,os,struct,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('backend',choices=['packed_eo','flashprove','hyrax'])
p.add_argument('action',choices=['build','smoke','bench'])
p.add_argument('--output',type=Path,required=True)
p.add_argument('--target-dir',type=Path,default=ROOT/'target/benchmarks')
p.add_argument('--cxx',default=os.environ.get('CXX','g++'))
p.add_argument('--threads',type=int,default=2)
p.add_argument('--offline',action='store_true')
p.add_argument('--log-n',type=int,default=30)
a=p.parse_args()
if a.action=='bench' and a.backend!='packed_eo':p.error('model runners are documented in README.md')
out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
target=a.target_dir.resolve();target.mkdir(parents=True,exist_ok=True)
env=dict(os.environ,RAYON_NUM_THREADS=str(a.threads),OMP_NUM_THREADS=str(a.threads))
for name in ['EO_TEST_PATTERN','EO_EQ_PATTERN','EO_CROSS_KERNEL','EO_CROSS_SIMD','EO_SMALL_LEAF']:env.pop(name,None)
commands=[]
def run(command,label,cwd=ROOT,settings=None):
    command=list(map(str,command));commands.append(dict(command=command,cwd=str(cwd),settings=settings or {}))
    (out/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
    with (out/(label+'.log')).open('w') as stream:
        subprocess.run(command,cwd=cwd,env=env|(settings or {}),stdout=stream,stderr=subprocess.STDOUT,check=True)
    print(label,'PASS',flush=True)
(out/'configuration.json').write_text(json.dumps(vars(a),default=str,indent=2)+'\n')
base=HERE/a.backend
if a.backend=='packed_eo':
    source=base/'source'
    if not source.exists():run([sys.executable,base/'prepare.py'],'prepare')
    env['CARGO_TARGET_DIR']=str(target)
    cargo=['cargo','+nightly-2026-03-21','--config',source/'local-patch.toml']
    if (source/'local-vendor.toml').exists():cargo+=['--config',source/'local-vendor.toml']
    run(cargo+['build','--release','--locked']+(['--offline'] if a.offline else [])+
        ['-p','batch','--example','packed_eo_bitwidth_v6'],'build',source)
    binary=target/'release/examples/packed_eo_bitwidth_v6'
    if a.action!='build':
        for width in ['4','8','16','32','64','128','full']:
            if a.action=='smoke':
                for pattern in ['mixed','min','max']:
                    run([binary,width,'16','--smoke'],f'{width}-{pattern}',settings={'EO_TEST_PATTERN':pattern})
                for point in ['zero','one','alternating']:
                    run([binary,width,'8','--smoke'],f'{width}-point-{point}',settings={'EO_EQ_PATTERN':point})
            else:run(['/usr/bin/time','-v','-o',out/f'{width}-resources.log',binary,width,str(a.log_n)],width)
elif a.backend=='hyrax':
    env['CARGO_TARGET_DIR']=str(target)
    run(['cargo','+1.95','build','--release','--locked']+(['--offline'] if a.offline else []),'build',base)
    binary=target/'release/hyrax-model-bench'
    if a.action=='smoke':
        for width in [1,2]:
            data=out/f'input-{width}.bin'
            data.write_bytes(b''.join(struct.pack('<b' if width==1 else '<h',(index%101)-50) for index in range(180)))
            result=out/f'case-{width}'
            run([binary,data,width,base/'batch-smoke-description.json',result,a.threads],f'{width}-prove')
            run([binary,'verify',result,'negative'],f'{width}-verify',settings={'RAYON_NUM_THREADS':'1'})
            record=next(line for line in reversed((out/f'{width}-verify.log').read_text().splitlines()) if line.startswith('{'))
            assert json.loads(record)['verified_hidden_evaluation']
else:
    for name in ['equivalence_test','native_test','model_bench']:
        cmd=[a.cxx,'-std=c++17','-O3','-fopenmp','-DBLK_FIELD=2','-DBLK_HASH=1',
             '-I',base/'upstream/blocks','-I',base/'cpu',base/f'cpu/{name}.cpp','-o',target/name]
        if name!='equivalence_test':cmd.insert(1,'-DSERVING_BUILD')
        run(cmd,f'build-{name}')
    if a.action=='smoke':
        run([target/'equivalence_test'],'equivalence')
        run([target/'native_test'],'native')
        data=out/'input.bin';data.write_bytes(bytes((index%251 for index in range(1024))))
        prefix=out/'model'
        run([target/'model_bench',data,1,a.threads,prefix,11],'model')
        run([target/'model_bench','verify',prefix],'independent-verify',settings={'OMP_NUM_THREADS':'1'})
        assert json.loads((out/'independent-verify.log').read_text())['verified']
(out/'status.txt').write_text('success\n')
