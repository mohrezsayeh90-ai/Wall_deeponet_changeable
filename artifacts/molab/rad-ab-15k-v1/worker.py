"""Paired baseline/RAD continuation, with shared geometry/Re/base points and independent adaptive RNG."""
from pathlib import Path
import sys,os,json,time,copy,shutil,hashlib
import numpy as np
R=Path('/home/marimo/wall_rad_ab_15k_v1');P=R/'frozen/output/project';sys.path.insert(0,str(P))
import jax,jax.numpy as jnp
jax.config.update('jax_enable_x64',True)
from geometry_operator.checkpoint import load,save
from geometry_operator.sampling import load_pool,batch
from geometry_operator.model import context
from geometry_operator.geometry import forward
from geometry_operator.operators import residual
from geometry_operator.adaptive_core import make_step
from geometry_operator.train import make_step as original_make
from geometry_operator.api import FlowOperator

def write(path,obj):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def rel(a,b,w):
 ww=w.reshape((-1,)+(1,)*(a.ndim-1));return float(np.sqrt(np.sum(ww*(a-b)**2)/max(np.sum(ww*b*b),1e-16)))
def main():
 assert json.loads((R/'setup-status.json').read_text())['phase']=='ready'
 assert jax.default_backend()=='gpu'
 os.close(os.open(R/'worker.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600))
 base,cfg,meta,(mean,std),(bm,bv)=load(R/'base.npz',optimizer=True);pool,_=load_pool(P/'data/geometries');dev,_=load_pool(P/'data/geometries','dev')
 step,one=make_step(cfg,mean,std);old,_=original_make(cfg,mean,std)
 probes={}
 for label in ['control','audit']:
  with np.load(R/f'{label}.npz') as d:data={k:jnp.asarray(d[k],dtype=jnp.float32) for k in d.files}
  def loss(p,data=data):
   def acc(t,x):return t+one(p,x)[0],None
   return jax.lax.scan(acc,jnp.array(0.,dtype=jnp.float32),data)[0]/data['re'].shape[0]
  probes[label]=jax.jit(loss)
 def scores(p,data):
  def scan(_,item):
   b=context(p,item['sensors'],item['re'],mean,std)
   rr=jax.vmap(lambda z:residual(p,b,item['coeff'],item['re'],forward(item['coeff'],z),cfg))(item['z'])
   return None,jnp.sqrt(jnp.sum(rr[:,:2]**2,axis=1))
  return jax.lax.scan(scan,None,data)[1]
 score=jax.jit(scores)
 # Same production batch, same LR: exact frozen physics/Adam versus dynamic core.
 test_rng=np.random.default_rng();test_rng.bit_generator.state=copy.deepcopy(meta['rng_state']);sample=batch(test_rng,pool,32,128,48);jj={k:jnp.asarray(v,dtype=jnp.float32) for k,v in sample.items()}
 a=old(base,bm,bv,jnp.asarray(287401),jj);b=step(base,bm,bv,jnp.asarray(287401),jj,jnp.asarray(2e-5,dtype=jnp.float32))
 for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):np.testing.assert_allclose(np.asarray(x),np.asarray(y),rtol=1e-5,atol=1e-7)
 test_extra=dict(sample,z=batch(np.random.default_rng(45),pool,32,256,48)['z'])
 test_scores=np.asarray(score(base,{k:jnp.asarray(x,dtype=jnp.float32) for k,x in test_extra.items()}));assert test_scores.shape==(32,256) and np.isfinite(test_scores).all() and (test_scores>=0).all()
 write(R/'preflight.json',{'passed':True,'dynamic_matches_original':True,'base_sha256':hashlib.sha256((R/'base.npz').read_bytes()).hexdigest(),'jax':jax.__version__})
 selection=json.loads((R/'repo/reports/adam_next50k_comparison/selection.json').read_text())['population']
 refs=[]
 for row in selection:
  path=R/'repo/reports'/row['group']/row['directory']/'fields.npz'
  with np.load(path) as d:refs.append((str(row['case_id']),d['coeff'].copy(),30.,d['xy'].copy(),d['weights'].copy(),d['fem'].copy()))
 for i in range(8):
  with np.load(P/f'runs/adam_next50k/development/{i:04d}.npz') as d:refs.append((f'pilot_{i}',dev['coeff'][i],float(dev['reynolds'][i]),d['xy'].copy(),d['weights'].copy(),d['truth'].copy()))
 def fem(path):
  model=FlowOperator(path,allow_unvalidated=True,chunk_size=256);rows=[]
  for name,coeff,re,xy,w,truth in refs:
   pred=model.evaluate(coeff,re,xy);rows.append(dict(case=name,velocity_l2=rel(pred[:,:2],truth[:,:2],w),pressure_l2=rel(pred[:,2],truth[:,2],w)))
  result={'cases':rows}
  for label,rr in [('saved20',rows[:20]),('pilot8',rows[20:])]:
   result[label]={f'{field}_{stat}':float(fun([x[field] for x in rr])) for field in ['velocity_l2','pressure_l2'] for stat,fun in [('median',np.median),('max',np.max)]}
  return result
 basefem=fem(R/'base.npz');write(R/'base_fem.json',basefem)
 for mode in ['baseline','rad']:
  folder=R/mode;folder.mkdir();snap=folder/'snapshots';snap.mkdir();p,cfg,md,(mean,std),(m,v)=load(R/'base.npz',optimizer=True)
  rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(md['rng_state']);arng=np.random.default_rng(2026111901);start=time.monotonic();count=287400
  def persist():
   save(folder/'checkpoint.npz',p,cfg,count,rng.bit_generator.state,mean,std,m,v,dict(train_pool_sha256=md['train_pool_sha256'],accepted=False,mode=mode,lr=2e-5,adaptive_rng_state=arng.bit_generator.state,training_labels=0,selection='FEM development comparison; no promotion',target_steps=302400))
   write(R/'status.json',dict(phase='running',branch=mode,step=count,offset=count-287400,target_step=302400,lr=2e-5,elapsed_branch_seconds=time.monotonic()-start))
  def evaluate():
   rec=dict(step=count,offset=count-287400,control=float(probes['control'](p)),audit=float(probes['audit'](p)))
   with (folder/'probe.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
   persist()
   if count==287400 or (count-287400)%5000==0:
    stable=snap/f'checkpoint_{count}.npz';shutil.copy2(folder/'checkpoint.npz',stable)
    ff=basefem if count==287400 else fem(stable);write(folder/f'fem_{count}.json',ff)
    # Ready marker makes immutable checkpoints safe for the independent collector.
    write(snap/f'checkpoint_{count}.ready.json',{'step':count,'sha256':hashlib.sha256(stable.read_bytes()).hexdigest(),'bytes':stable.stat().st_size})
  evaluate()
  with (folder/'training.jsonl').open('a',buffering=1) as log:
   for count in range(287401,302401):
    data=batch(rng,pool,32,128,48)
    if mode=='rad':
     candidate=batch(arng,pool,32,256,48)['z'];extra=dict(data,z=candidate)
     ss=np.asarray(score(p,{k:jnp.asarray(x,dtype=jnp.float32) for k,x in extra.items()}));assert np.isfinite(ss).all() and (ss>=0).all()
     zz=np.empty_like(data['z'])
     for i in range(32):
      prob=ss[i].astype(float)/max(float(ss[i].mean()),1e-12)+1.;prob/=prob.sum()
      keep=arng.choice(128,64,replace=False);choose=arng.choice(256,64,replace=False,p=prob)
      zz[i,:64]=data['z'][i,keep];zz[i,64:]=candidate[i,choose]
     assert zz.shape==(32,128,2) and np.isfinite(zz).all() and (zz>0).all() and (zz<1).all();data['z']=zz
    jj={k:jnp.asarray(x,dtype=jnp.float32) for k,x in data.items()};t=time.monotonic();p,m,v,value,terms,gn=step(p,m,v,jnp.asarray(count),jj,jnp.asarray(2e-5,dtype=jnp.float32));value=float(value)
    if not np.isfinite(value):raise RuntimeError('Nonfinite training loss; previous checkpoint retained')
    if count==287401 or count%10==0:log.write(json.dumps(dict(step=count,loss=value,terms=np.asarray(terms).tolist(),gradient_norm=float(gn),lr=2e-5,step_seconds=time.monotonic()-t,elapsed_seconds=time.monotonic()-start))+'\n')
    if count==287401 or count%100==0:persist()
    if count==287500:
     stable=snap/f'checkpoint_{count}.npz';shutil.copy2(folder/'checkpoint.npz',stable);write(snap/f'checkpoint_{count}.ready.json',{'step':count,'sha256':hashlib.sha256(stable.read_bytes()).hexdigest(),'bytes':stable.stat().st_size})
    if (count-287400)%2500==0:evaluate()
  write(folder/'completed.json',{'completed':True,'step':count,'elapsed_seconds':time.monotonic()-start})
 write(R/'status.json',{'phase':'completed','branches':['baseline','rad'],'step':302400,'updates_each':15000,'automatic_promotion':False})
if __name__=='__main__':
 try:main()
 except BaseException as e:
  write(R/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
