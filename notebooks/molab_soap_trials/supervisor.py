from pathlib import Path
import os,sys,time,subprocess,signal,json,zipfile
R=Path('/home/marimo/wall_soap_trials_v1')
os.close(os.open(R/'supervisor.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600))
assert json.loads((R/'setup-status.json').read_text())['phase']=='ready'
env=os.environ.copy();env.update(JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',JAX_COMPILATION_CACHE_DIR=str(R/'jax-cache'),MPLCONFIGDIR=str(R/'mpl'))
py=R/'venv/bin/python'
with (R/'console.log').open('w') as f:
 p=subprocess.Popen([str(py),str(R/'worker.py')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 (R/'supervisor-status.json').write_text(json.dumps({'phase':'running','worker_pid':p.pid,'started_unix':time.time()}))
 try:rc=p.wait(timeout=9000)
 except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);rc=124
(R/'supervisor-status.json').write_text(json.dumps({'phase':'completed' if rc==0 else 'failed','exit_code':rc,'finished_unix':time.time()}))
if rc==0:
 subprocess.run([str(py),str(R/'finalize.py')],check=True,env=env)
 with zipfile.ZipFile(R/'results.tmp.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
  for name in ['soap_2e5','soap_1e4']:
   for p in (R/name).rglob('*'):
    if p.is_file():z.write(p,p.relative_to(R))
  for p in R.iterdir():
   if p.is_file() and p.suffix in ['.py','.json','.jsonl','.log','.png','.pdf']:z.write(p,p.name)
 with zipfile.ZipFile(R/'results.tmp.zip') as z:assert z.testzip() is None
 (R/'results.tmp.zip').replace(R/'results.zip')
