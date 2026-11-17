"""One-hour Adam continuation, using the unmodified archived Kaggle trainer."""
import argparse,hashlib,json,os,sys
from pathlib import Path

BASE_STEP=132216
BASE_SHA='dabe75cd4121d1f4c6e2c920dced4c38cf033e2da7b1d74c4805ee90cf1e1544'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True);args=parser.parse_args()
    root=args.root.resolve();project=root/'frozen/output/project';checkpoint=project/'runs/adam_next50k/checkpoint.npz'
    pre=json.loads((root/'preflight-status.json').read_text());assert pre['state']=='passed'
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==BASE_SHA
    os.chdir(project);sys.path.insert(0,str(project))
    import jax
    jax.config.update('jax_enable_x64',True)
    assert jax.default_backend()=='gpu'
    from geometry_operator.checkpoint import load
    from geometry_operator.train import main as train_main
    _,config,meta,_,optimizer=load(checkpoint,optimizer=True)
    assert meta['step']==BASE_STEP and meta['optimizer_present'] and config.dtype=='float32'
    assert optimizer is not None
    sys.argv=['train','--pool',str(project/'data/geometries'),'--config',str(project/'configs/runtime.json'),
              '--resume',str(checkpoint),'--output',str(root/'run'),'--steps','100000000',
              '--batch','32','--points','128','--outlet','48','--max-seconds','3600',
              '--save-every','100','--log-every','10','--require-gpu',
              '--code-revision','f6b32ae7e95853f6e48142e822d6718597eb4708']
    train_main()

if __name__=='__main__':main()
