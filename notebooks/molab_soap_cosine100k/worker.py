"""Resume the complete selected SOAP state; no new basis initialization or warm-up."""
from pathlib import Path
import sys,os,json,time,copy,hashlib,shutil,importlib.util
import numpy as np
R=Path('/home/marimo/wall_soap_cosine100k_v1');PREV=Path('/home/marimo/wall_soap_trials_v1');OLD=Path('/home/marimo/wall_rad_ab_15k_v1');P=OLD/'frozen/output/project'
spec=importlib.util.spec_from_file_location('previous_soap_worker',PREV/'worker.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
jax,jnp,optax=h.jax,h.jnp,h.optax
from geometry_operator.checkpoint import load
from geometry_operator.sampling import batch,load_pool
from geometry_operator.soap_gradient import make_step
from geometry_operator.api import FlowOperator

def main():
 os.close(os.open(R/'worker.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600));assert jax.default_backend()=='gpu'
 source=R/'base.npz';expected='768c56f1c08515b3c0aa24d20ed19308cdf977e43746110db6b309c380f2b3e2'
 assert h.sha(source)==expected
 tx=h.make_optimizer(2e-5);p,st,rng,meta=h.restore_full(R/'base.npz',tx);_,cfg,_,(mean,std),_=load(R/'base.npz')
 h.BASE_META=meta
 assert meta['step']==322400 and int(st[0].count)==35001
 assert h.sha(P/'geometry_operator/soap_gradient.py')==json.loads((PREV/'gradient_source.json').read_text())['gradient_sha256']
 pool,manifest=load_pool(P/'data/geometries');assert meta['train_pool_sha256']==manifest['splits']['train']['sha256']
 grad,one=make_step(cfg,mean,std)
 def lr_at(offset):
  q=np.clip((float(offset)-1.)/99999.,0.,1.)
  return float(5e-6+.5*(1e-4-5e-6)*(1+np.cos(np.pi*q)))
 assert lr_at(1)==1e-4 and np.isclose(lr_at(100000),5e-6,rtol=0,atol=1e-20)
 assert all(lr_at(i+1)<=lr_at(i) for i in range(1,100000))
 def scheduled_update(g,s,p,rate):
  u,ns=tx.update(g,s,p)
  return jax.tree_util.tree_map(lambda x:x*(rate/2e-5),u),ns
 update=jax.jit(scheduled_update)
 def save_state(path,p,cfg,count,rng,mean,std,st,rate,elapsed):
  h.save_full(path,p,cfg,count,rng,mean,std,st,rate,elapsed)
  with np.load(path,allow_pickle=False) as d:arr={k:d[k] for k in d.files}
  mm=json.loads(str(arr['metadata']));mm.update(resume_step=322400,schedule='cosine',schedule_start=1e-4,schedule_end=5e-6,schedule_updates=100000,schedule_index=count-322400,last_applied_learning_rate=rate,next_learning_rate=lr_at(count-322400+1),batch_size=32)
  arr['metadata']=np.array(json.dumps(mm));tmp=path.with_suffix('.schedule.npz');np.savez_compressed(tmp,**arr);tmp.replace(path)

 jd=lambda d:{k:jnp.asarray(x,dtype=jnp.float32) for k,x in d.items()}
 # Restore a second complete state, compare the next real gradient/update and data stream without consuming training RNG.
 test=R/'preflight.npz';save_state(test,p,cfg,322400,rng,mean,std,st,2e-5,0.)
 rp,rs,rr,_=h.restore_full(test,tx)
 for a,b in zip(jax.tree_util.tree_leaves((p,st)),jax.tree_util.tree_leaves((rp,rs))):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
 tr=np.random.default_rng();tr.bit_generator.state=copy.deepcopy(rng.bit_generator.state);a=batch(tr,pool,32,128,48);b=batch(rr,pool,32,128,48)
 for k in a:np.testing.assert_array_equal(a[k],b[k])
 g,l,t,n=grad(p,jd(a));u,s1=update(g,st,p,jnp.asarray(lr_at(1),dtype=jnp.float32));v,s2=update(g,rs,rp,jnp.asarray(lr_at(1),dtype=jnp.float32))
 for x,y in zip(jax.tree_util.tree_leaves((u,s1)),jax.tree_util.tree_leaves((v,s2))):np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
 assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves((u,s1))) and int(s1[0].count)==35002
 direct=h.make_optimizer(1e-4);du,ds=jax.jit(lambda g,s,p:direct.update(g,s,p))(g,st,p)
 for x,y in zip(jax.tree_util.tree_leaves(u),jax.tree_util.tree_leaves(du)):np.testing.assert_allclose(np.asarray(x),np.asarray(y),rtol=2e-6,atol=1e-9)
 for x,y in zip(jax.tree_util.tree_leaves(s1),jax.tree_util.tree_leaves(ds)):np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
 h.write(R/'preflight.json',dict(passed=True,source_sha256=expected,resumed_step=322400,restored_optimizer_count=35001,next_optimizer_count=35002,first_lr=lr_at(1),final_lr=lr_at(100000),scheduled_update_matches_direct_1e4=True,schedule_state_added=False,parameters_and_state_exact=True,rng_next_batch_exact=True,next_update_after_roundtrip_exact=True,no_optimizer_reinitialization=True,extra_basis_initialization_updates=0))
 test.unlink()
 probes={}
 for label in ['control','audit']:
  shutil.copy2(PREV/f'{label}.npz',R/f'{label}.npz')
  with np.load(R/f'{label}.npz') as d:data=jd({k:d[k] for k in d.files})
  def loss(p,data=data):
   def acc(t,x):return t+one(p,x)[0],None
   return jax.lax.scan(acc,jnp.array(0.,dtype=jnp.float32),data)[0]/data['re'].shape[0]
  probes[label]=jax.jit(loss)
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
  f={'cases':rows}
  for group,rrr in [('saved20',rows[:20]),('pilot8',rows[20:])]:f[group]={f'{field}_{stat}':float(fun([x[field] for x in rrr])) for field in ['velocity_l2','pressure_l2'] for stat,fun in [('median',np.median),('max',np.max)]}
  return f
 start_fem=json.loads((R/'start_fem.json').read_text());h.write(R/'start_fem.json',start_fem)
 folder=R/'continuation';folder.mkdir();snap=folder/'snapshots';snap.mkdir();count=322400;begin=time.monotonic();bad_streak=0;stop_reason='target_completed'
 def persist(snapshot=False):
  save_state(folder/'checkpoint.npz',p,cfg,count,rng,mean,std,st,lr_at(count-322400),time.monotonic()-begin)
  h.write(R/'status.json',dict(phase='running',step=count,offset=count-322400,target_step=422400,learning_rate=lr_at(count-322400),optimizer_count=int(st[0].count),elapsed_seconds=time.monotonic()-begin))
  if snapshot:
   dst=snap/f'checkpoint_{count}.npz';shutil.copy2(folder/'checkpoint.npz',dst);h.write(dst.with_suffix('.ready.json'),dict(step=count,bytes=dst.stat().st_size,sha256=h.sha(dst),optimizer='soap'))
 def evaluate():
  rec=dict(step=count,offset=count-322400,control=float(probes['control'](p)),audit=float(probes['audit'](p)),elapsed_seconds=time.monotonic()-begin)
  assert np.isfinite([rec['control'],rec['audit']]).all()
  with (folder/'probe.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
 evaluate();persist()
 with (folder/'training.jsonl').open('a',buffering=1) as log:
  for count in range(322401,422401):
   data=jd(batch(rng,pool,32,128,48));t0=time.monotonic();gg,value,terms,gn=grad(p,data);value=float(value)
   if not np.isfinite(value):raise RuntimeError('Nonfinite loss; retain previous verified checkpoint')
   u,st=update(gg,st,p,jnp.asarray(lr_at(count-322400),dtype=jnp.float32));p=optax.apply_updates(p,u);jax.block_until_ready(p)
   if count==322401 or count%10==0:log.write(json.dumps(dict(step=count,loss=value,terms=np.asarray(terms).tolist(),gradient_norm=float(gn),learning_rate=lr_at(count-322400),elapsed_seconds=time.monotonic()-begin,step_seconds=time.monotonic()-t0))+'\n')
   if count%100==0:h.write(R/'status.json',dict(phase='running',step=count,offset=count-322400,target_step=422400,learning_rate=lr_at(count-322400),optimizer_count=int(st[0].count),elapsed_seconds=time.monotonic()-begin))
   if count==322500 or (count-322400)%5000==0:persist(snapshot=True)
   if (count-322400)%1000==0:evaluate()
   if (count-322400)%5000==0:
    persist(snapshot=True);ff=fem(folder/'checkpoint.npz');h.write(folder/f'fem_{count}.json',ff)
    # Heuristic sustained-regression guard; it is not a statistical significance test.
    bad=any(ff[group][key]>1.10*start_fem[group][key] for group in ['saved20','pilot8'] for key in ['velocity_l2_median','pressure_l2_median'])
    bad_streak=bad_streak+1 if bad else 0;h.write(R/'regression_guard.json',dict(step=count,bad_streak=bad_streak,threshold_ratio=1.10,required_consecutive=2))
    if bad_streak>=2:h.write(R/'regression_warning.json',dict(step=count,bad_streak=bad_streak,action='warning; user requested full100k schedule'))
   if time.monotonic()-begin>28800:stop_reason='wall_clock_budget';persist(snapshot=True);break
 assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves((p,st)))
 persist(snapshot=True);h.write(folder/'final_fem.json',fem(folder/'checkpoint.npz'))
 h.write(folder/'completed.json',dict(completed=stop_reason=='target_completed',step=count,new_updates=count-322400,total_soap_parameter_updates=count-287400,optimizer_count=int(st[0].count),elapsed_seconds=time.monotonic()-begin,stop_reason=stop_reason))
 h.write(R/'status.json',dict(phase='completed' if stop_reason=='target_completed' else 'stopped',step=count,offset=count-322400,target_step=422400,stop_reason=stop_reason,automatic_promotion=False))
if __name__=='__main__':
 try:main()
 except BaseException as e:h.write(R/'failure.json',dict(error=type(e).__name__,message=str(e)));raise
