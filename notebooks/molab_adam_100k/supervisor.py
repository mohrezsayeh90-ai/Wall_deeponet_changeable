import argparse,json,os,subprocess,sys,time,zipfile
from pathlib import Path
from datetime import datetime,timezone

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);root=ap.parse_args().root
    fd=os.open(root/'launch.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    state={'phase':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),'base_step':187400,'target_step':287400,'additional_updates':100000}
    def write():
        tmp=root/'state.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(root/'supervisor-status.json')
    env=os.environ.copy();env.update(JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLCONFIGDIR=str(root/'mpl-cache'),WALL_LR_CONFIG=str(root/'lr.json'))
    with (root/'console.log').open('w') as log:
        p=subprocess.Popen([sys.executable,str(root/'worker.py'),'--root',str(root),'--case','cosine_100k'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['worker_pid']=p.pid;write()
        try:code=p.wait(timeout=15000)
        except subprocess.TimeoutExpired:p.terminate();code=124
    state.update(phase='completed' if code==0 else 'failed',exit_code=code,finished_at_utc=datetime.now(timezone.utc).isoformat());write()
    if code==0:
        tmp=root/'results.tmp.zip'
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for p in sorted(root.rglob('*')):
                if p.is_file() and p.suffix!='.zip' and 'project' not in p.relative_to(root).parts:z.write(p,arcname=p.relative_to(root))
        with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
        tmp.replace(root/'adam_100k_results.zip')
if __name__=='__main__':main()
