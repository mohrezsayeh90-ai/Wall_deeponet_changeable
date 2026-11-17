"""Sequential, isolated 15k-update trials; no chaining of candidate weights."""
import argparse,json,os,subprocess,sys,time,zipfile
from pathlib import Path
from datetime import datetime,timezone

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);root=ap.parse_args().root.resolve()
    fd=os.open(root/'trials.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
    manifest=json.loads((root/'manifest.json').read_text());state={'phase':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),'completed':[]}
    def write():
        tmp=root/'suite.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(root/'suite-status.json')
    env=os.environ.copy();env.update(JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLCONFIGDIR=str(root/'mpl-cache'))
    hashes=[];first_losses=[]
    for case in manifest['cases']:
        state.update(active_case=case['id']);write()
        folder=root/'cases'/case['id'];folder.mkdir(parents=True,exist_ok=True)
        cfg={**case,'base_step':manifest['base_step']};(folder/'lr.json').write_text(json.dumps(cfg,indent=2));env['WALL_LR_CONFIG']=str(folder/'lr.json')
        with (folder/'console.log').open('w') as log:
            proc=subprocess.Popen([sys.executable,str(root/'trial_worker.py'),'--root',str(root),'--case',case['id']],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            state['worker_pid']=proc.pid;write()
            try:code=proc.wait(timeout=7500)
            except subprocess.TimeoutExpired:proc.terminate();code=124
        if code:
            state.update(phase='failed',failed_case=case['id'],exit_code=code);write();return
        resume=json.loads((folder/'resume.json').read_text());hashes.append(resume['first_batch_sha256'])
        first=json.loads((folder/'training.jsonl').read_text().splitlines()[0]);first_losses.append(first['loss'])
        if len(set(hashes))!=1 or max(first_losses)-min(first_losses)>1e-6:
            state.update(phase='failed',reason='Paired RNG/initial-objective check failed');write();return
        state['completed'].append(case['id']);write()
    state.update(phase='completed',finished_at_utc=datetime.now(timezone.utc).isoformat());write()
    with zipfile.ZipFile(root/'lr_trials_results.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix!='.zip' and 'project' not in p.relative_to(root).parts:z.write(p,arcname=p.relative_to(root))

if __name__=='__main__':main()
