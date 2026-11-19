from pathlib import Path
import sys,subprocess,os,json,signal,zipfile,time
R=Path('/home/marimo/wall_rad_ab_15k_v1')
def main():
 os.close(os.open(R/'supervisor.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600))
 env=os.environ.copy();env.update(JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLCONFIGDIR=str(R/'mpl'),JAX_COMPILATION_CACHE_DIR=str(R/'jax-cache'))
 subprocess.run([sys.executable,str(R/'build_core.py'),'--root',str(R)],check=True,env=env)
 with (R/'console.log').open('w') as f:
  p=subprocess.Popen([sys.executable,str(R/'worker.py')],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
  (R/'supervisor-status.json').write_text(json.dumps({'phase':'running','worker_pid':p.pid,'started_unix':time.time()}))
  try:rc=p.wait(timeout=14400)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGTERM);rc=124
 (R/'supervisor-status.json').write_text(json.dumps({'phase':'completed' if rc==0 else 'failed','exit_code':rc,'finished_unix':time.time()}))
 if rc==0:
  subprocess.run([sys.executable,str(R/'finalize.py')],check=True,env=env)
  with zipfile.ZipFile(R/'results.tmp.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
   for d in [R/'baseline',R/'rad']:
    for p in d.rglob('*'):
     if p.is_file():z.write(p,p.relative_to(R))
   for p in R.iterdir():
    if p.is_file() and p.suffix in ['.json','.jsonl','.py','.png','.pdf','.log']:z.write(p,p.name)
  with zipfile.ZipFile(R/'results.tmp.zip') as z:assert z.testzip() is None
  (R/'results.tmp.zip').replace(R/'results.zip')
if __name__=='__main__':main()
