from pathlib import Path
import sys,subprocess,os,shutil,json,hashlib
R=Path('/home/marimo/wall_soap_trials_v1');OLD=Path('/home/marimo/wall_rad_ab_15k_v1')
def status(**kw):(R/'setup-status.json').write_text(json.dumps(kw,indent=2))
try:
 status(phase='installing_isolated_dependencies')
 assert json.loads((OLD/'status.json').read_text())['phase']=='completed'
 base=OLD/'base.npz';sha=hashlib.sha256(base.read_bytes()).hexdigest();assert sha=='72bbe88ddc539258b9bfb433709aa434776393bd5f7e163d90478cbf7fa0e6c2'
 shutil.copy2(base,R/'base.npz')
 for name in ['control.npz','audit.npz','probe_ids.json','base_fem.json']:shutil.copy2(OLD/name,R/name)
 uv=shutil.which('uv');assert uv
 subprocess.run([uv,'venv','--system-site-packages',str(R/'venv')],check=True)
 py=R/'venv/bin/python'
 subprocess.run([uv,'pip','install','--python',str(py),'jax==0.11.2','jaxlib==0.11.2','optax==0.2.6','chex==0.1.91','jaxtyping==0.3.3'],check=True)
 subprocess.run([str(py),'-c',"import jax,optax,chex,jaxtyping;assert jax.__version__=='0.11.2';assert jax.default_backend()=='gpu';print('GPU_AND_DEPENDENCIES_OK',jax.devices())"],check=True)
 status(phase='ready',base_step=287400,base_sha256=sha,python=str(py),dependency_pins={'jax':'0.11.2','jaxlib':'0.11.2','optax':'0.2.6','chex':'0.1.91','jaxtyping':'0.3.3'},soap_source_commit='a7c92f174c49b13136294a94ae0d46266b4f671f')
except BaseException as e:status(phase='failed',error=type(e).__name__,message=str(e));raise
