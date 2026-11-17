"""Launch once, retain full-state snapshots, and record the trainer's exit."""
import argparse,hashlib,json,os,shutil,signal,subprocess,sys,time,zipfile
from datetime import datetime,timezone
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);root=parser.parse_args().root.resolve()
    assert json.loads((root/'preflight-status.json').read_text())['state']=='passed'
    # The lock is deliberately retained after exit to prevent accidental second runs.
    lock=os.open(root/'launch.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(lock,str(os.getpid()).encode());os.close(lock)
    run=root/'run';run.mkdir(exist_ok=True);(root/'snapshots').mkdir(exist_ok=True)
    base=root/'frozen/output/project/runs/adam_next50k/checkpoint.npz'
    assert not (run/'training.jsonl').exists(),'Refusing to append to another training run'
    shutil.copy2(base,run/'checkpoint.npz')
    now=datetime.now(timezone.utc);state={'phase':'starting','started_at_utc':now.isoformat(),'base_step':132216,'training_budget_seconds':3600,
        'budget_scope':'Frozen trainer clock includes JIT compilation; import/preflight time excluded; final update/save can slightly exceed the limit.',
        'watchdog_seconds':3780,'root':str(root),'checkpoint_sha256_before_training':hashlib.sha256(base.read_bytes()).hexdigest(),
        'optimizer':'Adam','optimizer_and_rng_restored':True,'frozen_code_revision':'f6b32ae7e95853f6e48142e822d6718597eb4708'}
    def write():
        tmp=root/'supervisor-status.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(root/'supervisor-status.json')
    env=os.environ.copy();env.update(JAX_ENABLE_X64='true',XLA_PYTHON_CLIENT_PREALLOCATE='false',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MPLCONFIGDIR=str(root/'mpl-cache'))
    begin=time.monotonic()
    with (root/'training-console.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,str(root/'worker.py'),'--root',str(root)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(phase='running',worker_pid=proc.pid);write();snapshot_at=begin+600;interrupted_at=None
        while proc.poll() is None:
            elapsed=time.monotonic()-begin
            if time.monotonic()>=snapshot_at and (run/'status.json').exists():
                tmp=root/'snapshots/current.tmp.npz';shutil.copyfile(run/'checkpoint.npz',tmp)
                with np.load(tmp,allow_pickle=False) as d:step=json.loads(str(d['metadata']))['step']
                tmp.replace(root/f'snapshots/checkpoint_{step}.npz');snapshot_at+=600
            if elapsed>3780 and interrupted_at is None:
                os.killpg(proc.pid,signal.SIGINT);interrupted_at=time.monotonic();state['watchdog_triggered']=True;write()
            if interrupted_at and time.monotonic()-interrupted_at>30:os.killpg(proc.pid,signal.SIGTERM)
            time.sleep(2)
    code=proc.wait();state.update(phase='completed' if code==0 else 'failed',exit_code=code,finished_at_utc=datetime.now(timezone.utc).isoformat(),supervisor_elapsed_seconds=time.monotonic()-begin)
    if (run/'status.json').exists():state['final_trainer_status']=json.loads((run/'status.json').read_text())
    state['checkpoint_sha256_after_training']=hashlib.sha256((run/'checkpoint.npz').read_bytes()).hexdigest();write()
    # Downloadable final bundle, without copying the immutable upstream dataset/archive.
    with zipfile.ZipFile(root/'molab_adam_1h_results.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for folder in [run,root/'snapshots']:
            for p in folder.iterdir():
                if p.is_file():z.write(p,arcname=str(p.relative_to(root)))
        for name in ['worker.py','supervisor.py','launch-manifest.json','preflight-status.json','preflight.log','supervisor-status.json','training-console.log']:
            p=root/name
            if p.exists():z.write(p,arcname=name)
    if code:raise SystemExit(code)

if __name__=='__main__':main()
