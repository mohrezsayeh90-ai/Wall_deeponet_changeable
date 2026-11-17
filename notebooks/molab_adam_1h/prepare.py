import os,subprocess,sys,json,time
from pathlib import Path
root=Path('/home/marimo/wall_adam_1h');state=root/'setup-status.json'
def record(**x):state.write_text(json.dumps(x,indent=2));print(json.dumps(x),flush=True)
try:
 record(phase='checking_gpu')
 env=os.environ.copy();env['JAX_ENABLE_X64']='true';env['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
 r=subprocess.run([sys.executable,'-c',"import jax,jax.numpy as j; print(jax.__version__,jax.devices()); assert jax.default_backend()=='gpu'; print(jax.jit(lambda x:x@x)(j.ones((32,32))).block_until_ready()[0,0])"],env=env,text=True,check=True)
 record(phase='cloning')
 repo=root/'repo'
 if not (repo/'.git').exists():subprocess.run(['git','clone','--depth','1','--filter=blob:none','--sparse','https://github.com/mohrezsayeh90-ai/Wall_deeponet_changeable.git',str(repo)],check=True)
 subprocess.run(['git','sparse-checkout','set','geometry_operator','configs','tests','scripts','artifacts/kaggle/adam-next50k-v1'],cwd=repo,check=True)
 record(phase='restoring_frozen_kaggle_output')
 subprocess.run([sys.executable,'-m','scripts.restore_kaggle','--artifact-dir','artifacts/kaggle/adam-next50k-v1','--output',str(root/'frozen')],cwd=repo,check=True)
 record(phase='ready_for_preflight')
except BaseException as e:
 record(phase='failed',error_type=type(e).__name__);raise
