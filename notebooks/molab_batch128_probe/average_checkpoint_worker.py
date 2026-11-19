"""Read-only stochastic weight average; no training labels or optimizer state reset."""
from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np
R=Path('/home/marimo/wall_weight_average_v1');CUR=Path('/home/marimo/wall_soap_continue20k_v1');OLD=Path('/home/marimo/wall_rad_ab_15k_v1');P=OLD/'frozen/output/project';PREV=Path('/home/marimo/wall_soap_trials_v1');sys.path.insert(0,str(P))
import jax,jax.numpy as jnp
jax.config.update('jax_enable_x64',True)
from geometry_operator.checkpoint import load,save
from geometry_operator.sampling import load_pool
from geometry_operator.soap_gradient import make_step
from geometry_operator.api import FlowOperator
paths=[CUR/f'continuation/snapshots/checkpoint_{s}.npz' for s in [311400,312400,313400,314400,315400]];assert all(p.exists() for p in paths)
p,c,md,(mean,std),_=load(paths[-1]);trees=[];sources=[]
for path in paths:
 pp,_,meta,_,_=load(path);trees.append(pp);sources.append({'step':meta['step'],'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
avg=jax.tree_util.tree_map(lambda *a:jnp.asarray(np.mean(np.stack([np.asarray(x,dtype=np.float64) for x in a]),axis=0),dtype=a[0].dtype),*trees)
save(R/'average.npz',avg,c,315400,md['rng_state'],mean,std,metadata={'accepted':False,'training_labels':0,'averaging_method':'Uniform parameter mean of five adjacent1k SOAP checkpoints','source_checkpoints':sources,'optimizer_resume_supported':False,'note':'Inference-only averaged parameters; retain original full SOAP states for training'})
_,one=make_step(c,mean,std);probes={}
for label in ['control','audit']:
 with np.load(PREV/f'{label}.npz') as d:data={k:jnp.asarray(d[k],dtype=jnp.float32) for k in d.files}
 def loss(p,data=data):
  def acc(t,x):return t+one(p,x)[0],None
  return jax.lax.scan(acc,jnp.array(0.,dtype=jnp.float32),data)[0]/data['re'].shape[0]
 probes[label]=jax.jit(loss)
dev,_=load_pool(P/'data/geometries','dev');refs=[]
for row in json.loads((OLD/'repo/reports/adam_next50k_comparison/selection.json').read_text())['population']:
 with np.load(OLD/'repo/reports'/row['group']/row['directory']/'fields.npz') as d:refs.append((str(row['case_id']),d['coeff'].copy(),30.,d['xy'].copy(),d['weights'].copy(),d['fem'].copy()))
for i in range(8):
 with np.load(P/f'runs/adam_next50k/development/{i:04d}.npz') as d:refs.append((f'pilot_{i}',dev['coeff'][i],float(dev['reynolds'][i]),d['xy'].copy(),d['weights'].copy(),d['truth'].copy()))
result={'read_only':True,'method':'Weight averaging, not smoothing a loss chart','sources':sources,'methods':{}}
for name,path in [('online',paths[-1]),('average',R/'average.npz')]:
 par,_,_,_,_=load(path);model=FlowOperator(path,allow_unvalidated=True,chunk_size=256);rows=[]
 for key,coeff,re,xy,w,truth in refs:
  pred=model.evaluate(coeff,re,xy)
  def rel(a,b):
   ww=w.reshape((-1,)+(1,)*(a.ndim-1));return float(np.sqrt(np.sum(ww*(a-b)**2)/max(np.sum(ww*b*b),1e-16)))
  rows.append({'case':key,'velocity_l2':rel(pred[:,:2],truth[:,:2]),'pressure_l2':rel(pred[:,2],truth[:,2])})
 rec={'probe':{k:float(f(par)) for k,f in probes.items()},'cases':rows}
 for group,rr in [('saved20',rows[:20]),('pilot8',rows[20:])]:rec[group]={f'{field}_{stat}':float(fun([x[field] for x in rr])) for field in ['velocity_l2','pressure_l2'] for stat,fun in [('median',np.median),('max',np.max)]}
 result['methods'][name]=rec
(R/'comparison.json').write_text(json.dumps(result,indent=2));print('AVERAGE_COMPARISON_COMPLETE',json.dumps({k:{f:v for f,v in val.items() if f!='cases'} for k,val in result['methods'].items()}),flush=True)
