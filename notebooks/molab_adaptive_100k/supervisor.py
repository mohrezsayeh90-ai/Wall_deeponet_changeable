import argparse,os,subprocess,sys,json,zipfile
from pathlib import Path
from datetime import datetime,timezone

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);r=ap.parse_args().root
    lock=os.open(r/'supervisor.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(lock)
    state={'phase':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),'base_step':287400,'target_step':387400}
    env=os.environ.copy();env.update(JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLCONFIGDIR=str(r/'mpl-cache'),JAX_COMPILATION_CACHE_DIR=str(r/'jax-cache'))
    with (r/'console.log').open('w') as f:
        p=subprocess.Popen([sys.executable,str(r/'worker.py'),'--root',str(r)],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        state['worker_pid']=p.pid;(r/'supervisor-status.json').write_text(json.dumps(state,indent=2))
        try:code=p.wait(timeout=18000)
        except subprocess.TimeoutExpired:p.terminate();code=124
    state.update(phase='completed' if code==0 else 'failed',exit_code=code,finished_at_utc=datetime.now(timezone.utc).isoformat());(r/'supervisor-status.json').write_text(json.dumps(state,indent=2))
    if code==0:
        tmp=r/'results.tmp.zip'
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for folder in [r/'run',r/'snapshots']:
                for p in folder.rglob('*'):
                    if p.is_file():z.write(p,p.relative_to(r))
            for p in r.iterdir():
                if p.is_file() and p.suffix in ['.json','.jsonl','.py','.npz','.log']:z.write(p,p.name)
        with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
        tmp.replace(r/'adaptive_results.zip')
if __name__=='__main__':main()
