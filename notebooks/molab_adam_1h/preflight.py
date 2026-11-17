import os,sys,json,hashlib,subprocess
from pathlib import Path
root=Path('/home/marimo/wall_adam_1h');project=root/'frozen/output/project';os.chdir(project);sys.path.insert(0,str(project))
status=root/'preflight-status.json'
try:
 import jax,numpy as np
 from geometry_operator.checkpoint import load
 from geometry_operator.sampling import load_pool,batch
 from geometry_operator.api import FlowOperator
 checkpoint=project/'runs/adam_next50k/checkpoint.npz'
 expected='dabe75cd4121d1f4c6e2c920dced4c38cf033e2da7b1d74c4805ee90cf1e1544'
 assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==expected
 params,config,meta,norm,optimizer=load(checkpoint,optimizer=True)
 assert meta['step']==132216 and meta['optimizer_present'] and config.dtype=='float32'
 assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves((params,optimizer)))
 assert jax.default_backend()=='gpu'
 pool,manifest=load_pool(project/'data/geometries')
 assert manifest['splits']['train']['sha256']==meta['train_pool_sha256']
 assert json.loads((project/'configs/runtime.json').read_text())['model']==config.json()
 with np.load(project/'data/geometries/normalization.npz') as n:
  for key,value in zip(['mean','std'],norm):np.testing.assert_allclose(n[key],value,rtol=1e-6,atol=1e-8)
 rng=np.random.default_rng();rng.bit_generator.state=meta['rng_state'];first=batch(rng,pool,32,128,48)
 batchsha=hashlib.sha256(b''.join(np.asarray(first[k]).tobytes() for k in sorted(first))).hexdigest()
 with np.load(project/'data/geometries/dev.npz') as dev,np.load(project/'runs/adam_next50k/development/0000.npz') as saved:
  model=FlowOperator(checkpoint,allow_unvalidated=True,chunk_size=128)
  prediction=model.evaluate(dev['coeff'][0],float(dev['reynolds'][0]),saved['xy'][:128])
  difference=float(abs(prediction-saved['predicted'][:128]).max());assert difference<1e-5
 result=dict(state='validating_tests',start_step=132216,checkpoint_sha256=expected,optimizer_restored=True,rng_restored=True,first_batch_sha256=batchsha,pool_sha256=meta['train_pool_sha256'],config=config.json(),jax=jax.__version__,devices=[str(d) for d in jax.devices()],pilot_inference_max_difference=difference)
 status.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 subprocess.run([sys.executable,'-m','pytest','-q','tests/test_geometry_operator.py','-k','not reference_flat'],check=True)
 result.update(state='passed',test_scope='Original frozen physical derivatives/boundaries, geometry, checkpoint and gradient/deterministic-update tests; independent FEM test excluded because no FEM solve is needed for training.')
 status.write_text(json.dumps(result,indent=2));print('PREFLIGHT_PASSED',flush=True)
except BaseException as e:
 status.write_text(json.dumps(dict(state='failed',error_type=type(e).__name__)));raise
