"""Build a self-contained Kaggle script embedding this exact source tree.

No tokens or local environment are copied. Existing GPU JAX is retained.
"""
import argparse,base64,io,json,subprocess,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def build(output,kind='main'):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w',zipfile.ZIP_DEFLATED) as z:
        for folder in ['geometry_operator','configs','tests']:
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and '__pycache__' not in str(p):z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'pyproject.toml','pyproject.toml')
    payload=base64.b64encode(stream.getvalue()).decode()
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    script='''import os,sys,base64,zipfile,io,subprocess,json,time
from pathlib import Path
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['MPLBACKEND']='Agg'
root=Path('/kaggle/working/project');root.mkdir(exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode(PAYLOAD))).extractall(root)
os.chdir(root);sys.path.insert(0,str(root))
# Preserve the managed CUDA JAX build; install only independent reference/test tools.
subprocess.run([sys.executable,'-m','pip','install','scikit-fem==11.0.0','pytest','--quiet'],check=True)
import jax
jax.config.update('jax_enable_x64',True)
assert jax.default_backend()=='gpu', 'Refusing to consume main budget on CPU'
print('RUNTIME',jax.__version__,jax.devices(),flush=True)
subprocess.run([sys.executable,'-m','pytest','-q'],check=True)
'''.replace('PAYLOAD',repr(payload))
    if kind=='profile':
        script+='''from geometry_operator.profile import run
run('profile.json')
'''
    else:
        script+='''from geometry_operator.profile import run
profile=run('profile.json')
from geometry_operator.sampling import generate_pools
generate_pools('data/geometries')
# Config selected by a preregistered precision agreement check, not by test data.
config=json.loads(Path('configs/production.json').read_text())
config['model']['dtype']=profile['selected_dtype']
Path('configs/runtime.json').write_text(json.dumps(config,indent=2))
from geometry_operator.train import main
sys.argv=['train','--pool','data/geometries','--config','configs/runtime.json',
 '--output','runs/main','--steps','100000','--batch','32','--points','128','--outlet','48',
 '--max-seconds','17400','--save-every','100','--log-every','10','--require-gpu',
 '--code-revision',REVISION]
main()
# No automatic locked-test access. A partial run is saved as unaccepted.
from geometry_operator.evaluate import main as evaluate
sys.argv=['evaluate','--checkpoint','runs/main/checkpoint.npz','--pool','data/geometries',
 '--references','data/references','--output','runs/main/development','--split','dev','--limit','8','--build-references']
evaluate()
'''.replace('REVISION',repr(revision))
    (output/'run.py').write_text(script)
    slug='geometry-pi-deeponet-'+kind
    meta=dict(id='mohrezsayeh/'+slug,title='Geometry PI DeepONet '+kind.title(),code_file='run.py',language='python',
        kernel_type='script',is_private=True,enable_gpu=True,enable_internet=True,dataset_sources=[],competition_sources=[],kernel_sources=[])
    (output/'kernel-metadata.json').write_text(json.dumps(meta,indent=2))
    return output

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='notebooks/kaggle_main');p.add_argument('--kind',choices=['main','profile'],default='main')
    a=p.parse_args();print(build(a.output,a.kind))
