"""Package a private Kaggle continuation of the exact collected v1 trainer."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build():
    source=ROOT/'artifacts/kaggle/main-v1/source/geometry-pi-deeponet-main.py'
    prefix=source.read_text().split('from geometry_operator.profile import run\n',1)[0]
    output=ROOT/'notebooks/kaggle_continue_50k';output.mkdir(exist_ok=True)
    continuation='''
# Reuse the original frozen trainer and optimizer, not fresh training.
import hashlib,shutil
import numpy as np
expected='a7e9e35b91d6e3c7f3a37404b940289427bfe78f9b38e39661c10382babd5235'
candidates=[p for p in Path('/kaggle/input').rglob('checkpoint.npz')
            if hashlib.sha256(p.read_bytes()).hexdigest()==expected]
assert len(candidates)==1, 'Expected exactly one matching original checkpoint input'
resume=candidates[0];original=resume.parents[2]
from geometry_operator.checkpoint import load
params,config,meta,norm,optimizer=load(resume,optimizer=True)
assert meta['step']==32216 and meta['optimizer_present']
assert config.dtype=='float32'
shutil.copytree(original/'data/geometries',root/'data/geometries',dirs_exist_ok=True)
shutil.copy2(original/'configs/runtime.json',root/'configs/runtime.json')
if (original/'data/references').exists():
    shutil.copytree(original/'data/references',root/'data/references',dirs_exist_ok=True)
from geometry_operator.sampling import load_pool
pool,manifest=load_pool('data/geometries')
assert manifest['splits']['train']['sha256']==meta['train_pool_sha256']
assert json.loads(Path('configs/runtime.json').read_text())['model']==config.json()
with np.load('data/geometries/normalization.npz',allow_pickle=False) as normal:
    np.testing.assert_allclose(normal['mean'],norm[0],rtol=1e-6,atol=1e-8)
    np.testing.assert_allclose(normal['std'],norm[1],rtol=1e-6,atol=1e-8)
resume_info=dict(source_kernel='mohrezsayeh/geometry-pi-deeponet-main',
    checkpoint_sha256=expected,start_step=32216,additional_steps=50000,target_step=82216,
    optimizer_restored=True,rng_restored_by_trainer=True,dtype=config.dtype,
    source_revision='f6b32ae7e95853f6e48142e822d6718597eb4708',
    max_training_seconds=36000,global_batch=32,interior_points=128,outlet_points=48)
Path('resume_manifest.json').write_text(json.dumps(resume_info,indent=2))
print('RESUME_VERIFIED',json.dumps(resume_info),flush=True)
from geometry_operator.train import main
sys.argv=['train','--pool','data/geometries','--config','configs/runtime.json',
 '--resume',str(resume),'--output','runs/continued_50k','--steps','82216',
 '--batch','32','--points','128','--outlet','48','--max-seconds','36000',
 '--save-every','100','--log-every','10','--require-gpu',
 '--code-revision','f6b32ae7e95853f6e48142e822d6718597eb4708']
main()
# Same pilot only; do not open the locked test or claim acceptance.
from geometry_operator.evaluate import main as evaluate
sys.argv=['evaluate','--checkpoint','runs/continued_50k/checkpoint.npz',
 '--pool','data/geometries','--references','data/references',
 '--output','runs/continued_50k/development','--split','dev','--limit','8','--build-references']
evaluate()
'''
    script=prefix+continuation
    compile(script,str(output/'run.py'),'exec')
    (output/'run.py').write_text(script)
    meta=dict(id='mohrezsayeh/geometry-pi-deeponet-continue-50k',title='Geometry PI DeepONet Continue 50k',code_file='run.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[],competition_sources=[],kernel_sources=['mohrezsayeh/geometry-pi-deeponet-main'])
    (output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(path=str(output),script_sha256=hashlib.sha256(script.encode()).hexdigest(),target_step=82216,additional_steps=50000)))


if __name__=='__main__':build()
