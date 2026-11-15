"""Build a frozen-v1 Kaggle package with only a new deterministic L-BFGS stage."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def main():
    source=(ROOT/'artifacts/kaggle/main-v1/source/geometry-pi-deeponet-main.py').read_text()
    prefix=source.split('from geometry_operator.profile import run\n',1)[0]
    prefix=prefix.replace("subprocess.run([sys.executable,'-m','pytest','-q'],check=True)\n",'')
    module=(ROOT/'geometry_operator/train_lbfgs.py').read_text();test=(ROOT/'tests/test_lbfgs.py').read_text()
    script=prefix+f"\nPath('geometry_operator/train_lbfgs.py').write_text({module!r})\nPath('tests/test_lbfgs.py').write_text({test!r})\n"
    script+='''
subprocess.run([sys.executable,'-m','pytest','-q'],check=True)
import hashlib,shutil
from dataclasses import replace
expected='84276c11feecd5bf230c79d1077250018b356ae118491678a4ed69e46ab83140'
candidates=[p for p in Path('/kaggle/input').rglob('checkpoint.npz') if hashlib.sha256(p.read_bytes()).hexdigest()==expected]
assert len(candidates)==1, 'Exactly one matching step-82216 checkpoint is required'
resume=candidates[0];original=resume.parents[2]
from geometry_operator.checkpoint import load
params,config,meta,norm,_=load(resume)
assert meta['step']==82216
shutil.copytree(original/'data/geometries',root/'data/geometries',dirs_exist_ok=True)
shutil.copytree(original/'data/references',root/'data/references',dirs_exist_ok=True)
shutil.copytree(original/'runs/continued_50k/development',root/'runs/base_adam/development',dirs_exist_ok=True)
Path('resume').mkdir(exist_ok=True);shutil.copy2(resume,'resume/base_checkpoint.npz')
Path('configs/lbfgs.json').write_text(json.dumps({'model':replace(config,dtype='float64').json()},indent=2))
manifest=dict(base_checkpoint_sha256=expected,base_adam_step=82216,optimizer_stage='L-BFGS-B only; no new Adam updates',fixed_training_geometries=512,disjoint_train_probe_geometries=128,points=128,outlet=48,seed=2026111521,max_iterations=500,max_function_evaluations=1000,max_training_seconds=21600,float64=True,training_labels=0,locked_test_opened=False)
Path('lbfgs_launch.json').write_text(json.dumps(manifest,indent=2))
print('LBFGS_RESUME_VERIFIED',json.dumps(manifest),flush=True)
from geometry_operator.train_lbfgs import main
sys.argv=['train_lbfgs','--checkpoint','resume/base_checkpoint.npz','--pool','data/geometries',
 '--output','runs/lbfgs','--fixed-geometries','512','--probe-geometries','128',
 '--points','128','--outlet','48','--maxiter','500','--maxfun','1000',
 '--max-seconds','21600','--history','20','--save-every','5','--require-gpu']
main()
from geometry_operator.evaluate import main as evaluate
sys.argv=['evaluate','--checkpoint','runs/lbfgs/checkpoint.npz','--pool','data/geometries',
 '--references','data/references','--output','runs/lbfgs/development',
 '--split','dev','--limit','8','--build-references']
evaluate()
p=Path('runs/lbfgs/development/report.json');report=json.loads(p.read_text())
status=json.loads(Path('runs/lbfgs/status.json').read_text())
report.update(optimizer_stage='L-BFGS-B',base_adam_step=82216,lbfgs_iterations=status['lbfgs_iterations'],base_checkpoint_sha256=expected,step_semantics='step is the retained base Adam count; L-BFGS iterations are separate')
p.write_text(json.dumps(report,indent=2))
'''
    compile(script,'run.py','exec')
    output=ROOT/'notebooks/kaggle_lbfgs';output.mkdir(exist_ok=True)
    (output/'run.py').write_text(script)
    meta=dict(id='mohrezsayeh/geometry-pi-deeponet-l-bfgs-refinement',title='Geometry PI DeepONet L-BFGS Refinement',code_file='run.py',language='python',kernel_type='script',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[],competition_sources=[],kernel_sources=['mohrezsayeh/geometry-pi-deeponet-continue-50k'])
    (output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('Built',output,'SHA256',hashlib.sha256(script.encode()).hexdigest())


if __name__=='__main__':main()
