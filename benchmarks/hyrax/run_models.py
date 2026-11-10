#!/usr/bin/env python3
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np

def main():
 p=argparse.ArgumentParser();p.add_argument('--models',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--scratch',type=Path,required=True);p.add_argument('--binary',type=Path,required=True);p.add_argument('--converter',type=Path,required=True);p.add_argument('--threads',type=int,default=48);a=p.parse_args()
 cases=json.loads(Path(__file__).with_name('cases.json').read_text());a.output.mkdir(parents=True,exist_ok=False);a.scratch.mkdir(parents=True,exist_ok=True)
 (a.output/'configuration.json').write_text(json.dumps(vars(a),default=str,indent=2));history=[]
 def record(cmd):
  history.append([str(x) for x in cmd]);(a.output/'commands.json').write_text(json.dumps(history,indent=2))
 for name,(model,family,dtype,count,expected) in cases.items():
  out=a.output/name;out.mkdir();width=1 if dtype=='i8' else 2;npdtype=np.dtype('i1' if width==1 else '<i2');raw=a.scratch/(name+'.input');matrix=a.scratch/(name+'.matrix')
  assert not raw.exists() and not matrix.exists()
  base=['--input',str(a.models/model),'--model-family',family,'--storage-dtype',dtype]
  cmd=[sys.executable,str(a.converter),'describe-json']+base;record(cmd);desc=json.loads(subprocess.check_output(cmd));(out/'description.json').write_text(json.dumps(desc,indent=2))
  assert desc['stream_entries']==count
  cmd=[sys.executable,str(a.converter),'stream']+base;record(cmd);h=hashlib.sha256();n=0;t=time.monotonic();print(name,'conversion',flush=True)
  with raw.open('xb') as dest,(out/'conversion.log').open('w') as err:
   proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=err)
   try:
    for chunk in iter(lambda:proc.stdout.read(8<<20),b''):dest.write(chunk);h.update(chunk);n+=len(chunk)
    assert proc.wait()==0
   finally:
    if proc.poll() is None:proc.terminate();proc.wait()
  (out/'input.json').write_text(json.dumps(dict(bytes=n,sha256=h.hexdigest(),expected_sha256=expected,conversion_seconds=time.monotonic()-t),indent=2));assert n==count*width and h.hexdigest()==expected
  pcs=out/'pcs';cmd=['/usr/bin/time','-v','-o',str(out/'resources.log'),str(a.binary),str(raw),str(width),str(out/'description.json'),str(pcs),str(a.threads)];record(cmd)
  print(name,'homomorphic commit/open started',flush=True)
  with (out/'run.log').open('w') as log,(out/'progress.log').open('w') as err:subprocess.run(cmd,stdout=log,stderr=err,check=True)
  cmd=[str(a.binary),'verify',str(pcs)];record(cmd)
  with (out/'independent-verify.json').open('w') as log:subprocess.run(cmd,stdout=log,env=dict(os.environ,RAYON_NUM_THREADS='1'),check=True)
  replay=json.loads((out/'independent-verify.json').read_text());assert replay['verified_hidden_evaluation']
  summary=json.loads((pcs/'summary.json').read_text());summary.update(scenario=name,independent_verify_seconds=replay['verify_decode_io_seconds'],verifier_setup_seconds=replay['setup_seconds'],verifier_commitment_aggregation_seconds=replay['commitment_aggregation_seconds_in_verify'],verified_hidden_evaluation=True)
  (out/'summary.json').write_text(json.dumps(summary,indent=2));print(name,'COMPLETE',json.dumps(summary),flush=True);raw.unlink()
if __name__=='__main__':main()


