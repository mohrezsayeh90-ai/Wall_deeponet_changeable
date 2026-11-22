from pathlib import Path
import os,sys,json,time,shutil,subprocess,hashlib
R=Path('/home/marimo/wall_soap_sweep50k_v1');PY='/home/marimo/wall_soap_trials_v1/venv/bin/python'
def read(p):return json.loads(p.read_text())
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
os.close(os.open(R/'supervisor.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600))
cases=[('constant_1e4',1e-4,'constant'),('constant_2e5',2e-5,'constant'),('constant_5e5',5e-5,'constant'),('constant_2e4',2e-4,'constant'),('cosine_1e4_2e5',1e-4,'cosine'),('constant_1e4_extend25k',1e-4,'constant')]
completed=0
try:
 for name,lr,mode in cases:
  d=R/name;d.mkdir();extension=name.endswith('extend25k');anchor=347400 if extension else 342400;updates=25000 if extension else 5000
  source=R/'constant_1e4/continuation/snapshots/checkpoint_347400.npz' if extension else R/'base.npz'
  shutil.copy2(source,d/'base.npz');shutil.copy2(R/'constant_1e4/continuation/final_fem.json' if extension else R/'start_fem.json',d/'start_fem.json')
  cfg=dict(anchor=anchor,updates=updates,optimizer_count=60001 if extension else 55001,mode=mode,lr=lr,end_lr=2e-5 if mode=='cosine' else lr,base_sha256=sha(source),case=name)
  write(d/'run_config.json',cfg);write(R/'status.json',dict(phase='running',active_case=name,completed_updates=completed,budget=50000,case_target=updates,case_anchor=anchor))
  p=subprocess.Popen([PY,str(R/'worker.py'),str(d)],stdout=(d/'console.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
  write(R/'process.json',dict(worker_pid=p.pid,case=name))
  rc=p.wait(timeout=28800)
  if rc:raise RuntimeError(f'{name} failed exit{rc}; inspect its failure.json; no automatic restart')
  done=read(d/'continuation/completed.json');completed+=done['new_updates']
  if not done['completed']:raise RuntimeError(f'{name} stopped early')
 assert completed==50000
 result={'budget_used':completed,'comparison_scope':'All5 screens share identical checkpoint342400, complete optimizer/RNG state and5000 fresh-batch updates. Extension is a longer-budget observation, not an equal-budget comparison.','screens':{},'extension':read(R/'constant_1e4_extend25k/continuation/final_fem.json'),'decision':'No200k training automatically launched. Use control/FEM and retained audit to choose; all repeated cohorts are development sets.'}
 for name,lr,mode in cases[:5]:
  d=R/name;probes=[json.loads(l) for l in (d/'continuation/probe.jsonl').read_text().splitlines()]
  result['screens'][name]={'config':read(d/'run_config.json'),'final_probe':probes[-1],'final_fem':read(d/'continuation/final_fem.json'),'completed':read(d/'continuation/completed.json')}
 write(R/'comparison.json',result);write(R/'status.json',dict(phase='completed',completed_updates=completed,budget=50000))
except BaseException as e:
 write(R/'failure.json',dict(error=type(e).__name__,message=str(e),completed_updates=completed));raise
