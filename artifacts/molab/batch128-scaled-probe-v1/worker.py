"""Same-state, same geometry-draw budget comparison of batch128 vs cached batch32."""
from pathlib import Path
import sys,json,time,hashlib,copy,os,importlib.util
import numpy as np
R=Path('/home/marimo/wall_batch128_scaled_probe_v1');PREV=Path('/home/marimo/wall_soap_trials_v1');OLD=Path('/home/marimo/wall_rad_ab_15k_v1');P=OLD/'frozen/output/project'
spec=importlib.util.spec_from_file_location('soap_helper',PREV/'worker.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);jax,jnp,optax=h.jax,h.jnp,h.optax
from geometry_operator.checkpoint import load
from geometry_operator.sampling import batch,load_pool
from geometry_operator.soap_gradient import make_step
from geometry_operator.api import FlowOperator

def main():
 os.close(os.open(R/'worker.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600));assert jax.default_backend()=='gpu'
 tx=h.make_optimizer(4e-5);p,st,rng,md=h.restore_full(R/'base.npz',tx);_,cfg,_,(mean,std),_=load(R/'base.npz');anchor=md['step'];assert anchor==313400
 assert h.sha(R/'base.npz')=='1f66f97353558c44f8af33be70f202bfb034dba9d1749682476893dad8636130'
 assert h.sha(R/'small_batch_reference.npz')=='a500fac7a8f6d07e2fdf8058b5d77c1303c45098278d28e664c2b5ffacd1dc25'
 pool,manifest=load_pool(P/'data/geometries');assert md['train_pool_sha256']==manifest['splits']['train']['sha256'];grad,one=make_step(cfg,mean,std);up=jax.jit(lambda g,s,p:tx.update(g,s,p));jd=lambda data:{k:jnp.asarray(v,dtype=jnp.float32) for k,v in data.items()}
 def get_big(rr):
  micro=[batch(rr,pool,32,128,48) for _ in range(4)];return {k:np.concatenate([x[k] for x in micro],axis=0) for k in micro[0]},micro
 # Exact average of four baseline batches, then clip once by unchanged global norm10.
 tr=np.random.default_rng();tr.bit_generator.state=copy.deepcopy(rng.bit_generator.state);big,micro=get_big(tr);gg,ll,tt,nn=grad(p,jd(big));mg=[];ml=[]
 for x in micro:
  g,l,t,n=grad(p,jd(x));assert float(n)<10;mg.append(g);ml.append(float(l))
 avg=jax.tree_util.tree_map(lambda *x:sum(x)/4,*mg)
 for x,y in zip(jax.tree_util.tree_leaves(avg),jax.tree_util.tree_leaves(gg)):np.testing.assert_allclose(np.asarray(x),np.asarray(y),rtol=4e-5,atol=2e-7)
 np.testing.assert_allclose(float(ll),np.mean(ml),rtol=1e-5,atol=1e-7)
 h.write(R/'preflight.json',dict(passed=True,anchor_step=anchor,global_batch=128,microbatch=32,microbatches=4,gradient_matches_four_microbatches=True,loss_matches_four_microbatches=True,same_physics_objective=True,learning_rate=4e-5,no_optimizer_reset=True))
 probes={}
 for name in ['control','audit']:
  with np.load(PREV/f'{name}.npz') as d:data=jd({k:d[k] for k in d.files})
  def loss(p,data=data):
   def acc(t,x):return t+one(p,x)[0],None
   return jax.lax.scan(acc,jnp.array(0.,dtype=jnp.float32),data)[0]/data['re'].shape[0]
  probes[name]=jax.jit(loss)
 dev,_=load_pool(P/'data/geometries','dev');refs=[]
 for row in json.loads((OLD/'repo/reports/adam_next50k_comparison/selection.json').read_text())['population']:
  with np.load(OLD/'repo/reports'/row['group']/row['directory']/'fields.npz') as d:refs.append((str(row['case_id']),d['coeff'].copy(),30.,d['xy'].copy(),d['weights'].copy(),d['fem'].copy()))
 for i in range(8):
  with np.load(P/f'runs/adam_next50k/development/{i:04d}.npz') as d:refs.append((f'pilot_{i}',dev['coeff'][i],float(dev['reynolds'][i]),d['xy'].copy(),d['weights'].copy(),d['truth'].copy()))
 def fem(path):
  model=FlowOperator(path,allow_unvalidated=True,chunk_size=256);rows=[]
  for name,coeff,re,xy,w,truth in refs:
   pred=model.evaluate(coeff,re,xy)
   def rel(a,b):
    ww=w.reshape((-1,)+(1,)*(a.ndim-1));return float(np.sqrt(np.sum(ww*(a-b)**2)/max(np.sum(ww*b*b),1e-16)))
   rows.append(dict(case=name,velocity_l2=rel(pred[:,:2],truth[:,:2]),pressure_l2=rel(pred[:,2],truth[:,2])))
  out={'cases':rows}
  for group,rr in [('saved20',rows[:20]),('pilot8',rows[20:])]:out[group]={f'{field}_{stat}':float(fun([x[field] for x in rr])) for field in ['velocity_l2','pressure_l2'] for stat,fun in [('median',np.median),('max',np.max)]}
  return out
 base_metrics={'probe':{k:float(f(p)) for k,f in probes.items()},'fem':fem(R/'base.npz')};h.write(R/'base_metrics.json',base_metrics)
 rp,_,reference_meta,_,_=load(R/'small_batch_reference.npz');small_metrics={'probe':{k:float(f(rp)) for k,f in probes.items()},'fem':fem(R/'small_batch_reference.npz'),'optimizer_updates':2000,'global_batch':32};h.write(R/'reference_metrics.json',small_metrics)
 folder=R/'batch128';folder.mkdir();snap=folder/'snapshots';snap.mkdir();begin=time.monotonic();count=0
 def save(snapshot=False):
  leaves,tree=jax.tree_util.tree_flatten(st);metadata=dict(md,step=anchor+count,rng_state=rng.bit_generator.state,soap_tree=str(tree),soap_leaves=len(leaves),global_batch=128,soap_learning_rate=4e-5,parameter_updates_since_anchor=count,equivalent_batch32_updates=count*4,equivalent_batch32_step=anchor+count*4,elapsed_seconds=time.monotonic()-begin,training_labels=0,accepted=False)
  arrays=dict(mean=np.asarray(mean),std=np.asarray(std),metadata=np.array(json.dumps(metadata)))
  arrays.update({f'param_{i}':np.asarray(x) for i,x in enumerate(jax.tree_util.tree_leaves(p))});arrays.update({f'soap_{i}':np.asarray(x) for i,x in enumerate(leaves)})
  dst=folder/'checkpoint.npz';tmp=folder/'checkpoint.write.npz';np.savez_compressed(tmp,**arrays);tmp.replace(dst)
  h.write(R/'status.json',dict(phase='running',optimizer_updates=count,target_optimizer_updates=500,equivalent_batch32_updates=count*4,target_equivalent_batch32_updates=2000,global_batch=128,anchor_step=anchor,elapsed_seconds=time.monotonic()-begin))
  if snapshot:
   dst2=snap/f'checkpoint_{anchor+count}.npz';__import__('shutil').copy2(dst,dst2);h.write(dst2.with_suffix('.ready.json'),dict(step=anchor+count,bytes=dst2.stat().st_size,sha256=h.sha(dst2),global_batch=128))
 with (folder/'training.jsonl').open('a',buffering=1) as log:
  for count in range(1,501):
   data,_=get_big(rng);t0=time.monotonic();g,value,terms,gn=grad(p,jd(data));value=float(value);assert np.isfinite(value)
   updates,st=up(g,st,p);p=optax.apply_updates(p,updates);jax.block_until_ready(p)
   if count==1 or count%5==0:log.write(json.dumps(dict(step=anchor+count,optimizer_updates=count,equivalent_batch32_updates=count*4,loss=value,gradient_norm=float(gn),step_seconds=time.monotonic()-t0,elapsed_seconds=time.monotonic()-begin))+'\n')
   if count==1 or count%50==0:h.write(R/'status.json',dict(phase='running',optimizer_updates=count,target_optimizer_updates=500,equivalent_batch32_updates=count*4,target_equivalent_batch32_updates=2000,global_batch=128,anchor_step=anchor,elapsed_seconds=time.monotonic()-begin))
   if count in [100,250,500]:save(snapshot=True)
   if count%100==0:
    rec={'optimizer_updates':count,'equivalent_batch32_updates':count*4,**{k:float(f(p)) for k,f in probes.items()}}
    with (folder/'probe.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
 save(snapshot=True)
 assert rng.bit_generator.state==reference_meta['rng_state'],'Matched data-draw RNG endpoints must agree'
 result={'base':base_metrics,'batch32_reference':small_metrics,'batch128':{'probe':{k:float(f(p)) for k,f in probes.items()},'fem':fem(folder/'checkpoint.npz'),'optimizer_updates':500,'global_batch':128,'equivalent_batch32_updates':2000,'elapsed_seconds':time.monotonic()-begin},'same_geometry_draw_budget':True,'rng_endpoints_exactly_match':True,'optimizer':'SOAP','lr':4e-5,'sampling_objective_unchanged':True,'no_automatic_promotion':True}
 h.write(R/'comparison.json',result);h.write(R/'status.json',dict(phase='completed',optimizer_updates=500,equivalent_batch32_updates=2000,global_batch=128,no_automatic_promotion=True))
if __name__=='__main__':
 try:main()
 except BaseException as e:h.write(R/'failure.json',dict(error=type(e).__name__,message=str(e)));raise
