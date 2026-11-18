"""Read-only checkpoint comparison; no updates to training state."""
from pathlib import Path
import os,sys,json,urllib.request,hashlib,shutil,time
root=Path('/home/marimo/wall_adaptive_100k');out=root/'diagnostic';out.mkdir(exist_ok=True)
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
project=root/'frozen/output/project';sys.path.insert(0,str(project))
import jax,jax.numpy as jnp,numpy as np
jax.config.update('jax_enable_x64',True)
from geometry_operator.checkpoint import load
from geometry_operator.adaptive_core import make_step
from geometry_operator.api import FlowOperator
from geometry_operator.sampling import load_pool
commit='eea5bf11f5b48c41589c340edb35acef208deb73'
for name in ['fixed_probe.npz','best_checkpoint.npz']:
 dest=out/name
 if not dest.exists():
  urllib.request.urlretrieve(f'https://raw.githubusercontent.com/mohrezsayeh90-ai/Wall_deeponet_changeable/{commit}/artifacts/molab/adam-100k-v1/{name}',dest)
assert hashlib.sha256((out/'best_checkpoint.npz').read_bytes()).hexdigest()=='4333a74c3eebcc7d21f83a29138eb5353e6d812d341e5d48feb4b8448ce34463'
shutil.copy2(root/'run/checkpoint.npz',out/'observed_checkpoint.npz')
_,config,_,(mean,std),_=load(root/'base.npz');_,one=make_step(config,mean,std)
probes={}
for label,path in [('previous_fixed',out/'fixed_probe.npz'),('control',root/'control.npz'),('audit',root/'audit.npz')]:
 with np.load(path,allow_pickle=False) as d:data={k:jnp.asarray(d[k],dtype=jnp.float32) for k in d.files}
 def fun(p,data=data):
  def f(t,item):return t+one(p,item)[0],None
  return jax.lax.scan(f,jnp.array(0.,dtype=jnp.float32),data)[0]/data['re'].shape[0]
 probes[label]=jax.jit(fun)
dev,_=load_pool(project/'data/geometries','dev');results={}
for label,path in [('base_287400',root/'base.npz'),('previous_best_282400',out/'best_checkpoint.npz'),('observed_current',out/'observed_checkpoint.npz')]:
 p,_,meta,_,opt=load(path,optimizer=True)
 rec={'step':meta['step'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'probe':{k:float(f(p)) for k,f in probes.items()},'optimizer_present':meta['optimizer_present'],'rng_state':meta['rng_state']}
 if label=='observed_current':
  with np.load(root/'base.npz') as b,np.load(path) as d:
   rec['parameter_relative_change']=float(np.sqrt(sum(np.sum((d[k].astype(float)-b[k].astype(float))**2) for k in d.files if k.startswith('param_'))/sum(np.sum(b[k].astype(float)**2) for k in b.files if k.startswith('param_'))))
   rec['parameter_arrays_changed']=sum(not np.array_equal(d[k],b[k]) for k in d.files if k.startswith('param_'))
   rec['all_state_finite']=all(np.isfinite(d[k]).all() for k in d.files if k!='metadata')
 model=FlowOperator(path,allow_unvalidated=True,chunk_size=256);rows=[]
 for i in range(8):
  with np.load(project/f'runs/adam_next50k/development/{i:04d}.npz') as ref:
   pred=model.evaluate(dev['coeff'][i],float(dev['reynolds'][i]),ref['xy']);truth=ref['truth'];w=ref['weights']
   def rel(a,b):
    ww=w.reshape((-1,)+(1,)*(a.ndim-1));return float(np.sqrt(np.sum(ww*(a-b)**2)/max(np.sum(ww*b*b),1e-16)))
   rows.append({'case':i,'velocity_l2':rel(pred[:,:2],truth[:,:2]),'pressure_l2':rel(pred[:,2],truth[:,2])})
 rec['fem_cases']=rows;rec['median_velocity_l2']=float(np.median([x['velocity_l2'] for x in rows]));rec['median_pressure_l2']=float(np.median([x['pressure_l2'] for x in rows]));results[label]=rec
 (out/'comparison.json').write_text(json.dumps(results,indent=2));print(label,json.dumps(rec),flush=True)
(out/'evaluation_complete.json').write_text(json.dumps({'completed':True,'observed_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}))
