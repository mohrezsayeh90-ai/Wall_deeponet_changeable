from pathlib import Path
import subprocess,sys,os,json,hashlib,shutil
r=Path('/home/marimo/wall_adaptive_100k');status=r/'setup-status.json'
def record(**x):status.write_text(json.dumps(x,indent=2));print(json.dumps(x),flush=True)
try:
 assert not (r/'supervisor.lock').exists(),'Refusing setup during an existing launch'
 repo=r/'repo';record(phase='restoring_inputs')
 if not (repo/'.git').exists():subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse','https://github.com/mohrezsayeh90-ai/Wall_deeponet_changeable.git',str(repo)],check=True)
 patterns=['/scripts/restore_kaggle.py','/artifacts/kaggle/adam-next50k-v1/inventory.json','/artifacts/kaggle/adam-next50k-v1/raw-output.zip.part*','/artifacts/molab/adam-100k-v1/checkpoint.npz','/geometry_operator/','/configs/','/tests/']
 subprocess.run(['git','sparse-checkout','set','--no-cone',*patterns],cwd=repo,check=True)
 subprocess.run([sys.executable,'scripts/restore_kaggle.py','--artifact-dir','artifacts/kaggle/adam-next50k-v1','--output',str(r/'frozen')],cwd=repo,check=True)
 base=repo/'artifacts/molab/adam-100k-v1/checkpoint.npz'
 expected='72bbe88ddc539258b9bfb433709aa434776393bd5f7e163d90478cbf7fa0e6c2'
 assert hashlib.sha256(base.read_bytes()).hexdigest()==expected
 shutil.copy2(base,r/'base.npz');project=r/'frozen/output/project';sys.path.insert(0,str(project))
 import jax,numpy as np
 jax.config.update('jax_enable_x64',True);assert jax.default_backend()=='gpu'
 from geometry_operator.checkpoint import load
 from geometry_operator.sampling import load_pool,batch
 p,c,m,n,opt=load(r/'base.npz',optimizer=True)
 assert m['step']==287400 and m['optimizer_present'] and c.dtype=='float32'
 assert all(np.isfinite(np.asarray(a)).all() for a in jax.tree_util.tree_leaves((p,opt)))
 pool,manifest=load_pool(project/'data/geometries');assert m['train_pool_sha256']==manifest['splits']['train']['sha256']
 assert json.loads((project/'configs/runtime.json').read_text())['model']==c.json()
 with np.load(project/'data/geometries/normalization.npz') as norm:
  np.testing.assert_allclose(norm['mean'],n[0],rtol=1e-6,atol=1e-8);np.testing.assert_allclose(norm['std'],n[1],rtol=1e-6,atol=1e-8)
 dev,_=load_pool(project/'data/geometries','dev');rng=np.random.default_rng(2026111801);ids=rng.choice(len(dev['coeff']),64,replace=False)
 data=batch(rng,dev,64,128,48);data['coeff']=dev['coeff'][ids];data['sensors']=dev['sensors'][ids]
 for label,sl in [('control',slice(0,32)),('audit',slice(32,64))]:np.savez_compressed(r/(label+'.npz'),**{k:v[sl] for k,v in data.items()})
 (r/'probe_ids.json').write_text(json.dumps({'control':ids[:32].tolist(),'audit':ids[32:].tolist(),'seed':2026111801,'disjoint_geometry_ids':True},indent=2))
 record(phase='ready',base_step=m['step'],base_sha256=expected,jax=jax.__version__,devices=[str(x) for x in jax.devices()],optimizer_restored=True,normalization_verified=True,pool_sha256=m['train_pool_sha256'])
except BaseException as e:record(phase='failed',error_type=type(e).__name__);raise
