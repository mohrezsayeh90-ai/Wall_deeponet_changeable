"""Two bounded SOAP continuations, common RNG and frozen physics, with portable full optimizer state."""
from pathlib import Path
import os,sys,time,json,hashlib,copy,shutil
import numpy as np
R=Path('/home/marimo/wall_soap_trials_v1');OLD=Path('/home/marimo/wall_rad_ab_15k_v1');P=OLD/'frozen/output/project'
sys.path[:0]=[str(R/'vendor'),str(P)]
import jax,jax.numpy as jnp,optax
from soap_jax import soap
jax.config.update('jax_enable_x64',True)
from geometry_operator.checkpoint import load,save
from geometry_operator.sampling import batch,load_pool
from geometry_operator.api import FlowOperator
from geometry_operator.train import make_step as original_make

def write(path,obj):
 tmp=path.with_suffix('.json.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def make_optimizer(lr):return soap(learning_rate=lr,b1=.95,b2=.95,weight_decay=0.,precondition_frequency=10,max_precond_dim=1024,precondition_1d=False)
def save_full(path,p,config,count,rng,mean,std,state,rate,elapsed):
 leaves,tree=jax.tree_util.tree_flatten(state)
 meta=dict(optimizer='soap',soap_state_present=True,soap_tree=str(tree),soap_leaves=len(leaves),soap_learning_rate=rate,soap_version_commit='a7c92f174c49b13136294a94ae0d46266b4f671f',train_pool_sha256=BASE_META['train_pool_sha256'],accepted=False,training_labels=0,elapsed_seconds=elapsed,soap_init_gradient_evaluations=1,base_step=287400,soap_hyperparameters=dict(b1=.95,b2=.95,precondition_frequency=10,max_precond_dim=1024,precondition_1d=False,weight_decay=0.))
 # Standard model fields plus SOAP leaves in one atomic, pickle-free NPZ. Adam resume deliberately unavailable.
 tmp=path.with_suffix('.base.npz');save(tmp,p,config,count,rng.bit_generator.state,mean,std,metadata=meta)
 with np.load(tmp,allow_pickle=False) as d:arrays={k:d[k] for k in d.files}
 arrays.update({f'soap_{i}':np.asarray(x) for i,x in enumerate(leaves)})
 final=path.with_suffix('.write.npz');np.savez_compressed(final,**arrays);final.replace(path);tmp.unlink()
def restore_full(path,tx):
 p,c,meta,norm,_=load(path)
 _,tree=jax.tree_util.tree_flatten(tx.init(p));assert str(tree)==meta['soap_tree']
 with np.load(path,allow_pickle=False) as d:state=jax.tree_util.tree_unflatten(tree,[jnp.asarray(d[f'soap_{i}']) for i in range(meta['soap_leaves'])])
 rng=np.random.default_rng();rng.bit_generator.state=meta['rng_state'];return p,state,rng,meta

def main():
 global BASE_META
 assert json.loads((R/'setup-status.json').read_text())['phase']=='ready'
 os.close(os.open(R/'worker.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600));assert jax.default_backend()=='gpu'
 base,cfg,BASE_META,(mean,std),(am,av)=load(R/'base.npz',optimizer=True)
 assert BASE_META['step']==287400 and sha(R/'base.npz')=='72bbe88ddc539258b9bfb433709aa434776393bd5f7e163d90478cbf7fa0e6c2'
 pool,manifest=load_pool(P/'data/geometries');assert manifest['splits']['train']['sha256']==BASE_META['train_pool_sha256']
 source=(P/'geometry_operator/train.py').read_text();assert source.count('def step(params,m,v,count,data):')==1
 prefix=source[:source.index('        m=jax.tree_util.tree_map')]
 suffix=source[source.index('        return params,m,v,loss,terms,norm')+len('        return params,m,v,loss,terms,norm'):]
 modified=(prefix+'        return g,loss,terms,norm'+suffix).replace('def step(params,m,v,count,data):','def step(params,data):')
 (P/'geometry_operator/soap_gradient.py').write_text(modified);(R/'gradient_core.py').write_text(modified)
 write(R/'gradient_source.json',{'original_sha256':hashlib.sha256(source.encode()).hexdigest(),'gradient_sha256':hashlib.sha256(modified.encode()).hexdigest(),'changes':'Return identical clipped batch gradient, loss, terms and norm before original Adam update; physics untouched'})
 from geometry_operator.soap_gradient import make_step
 grad,one=make_step(cfg,mean,std);original,_=original_make(cfg,mean,std)
 def jdata(data):return {k:jnp.asarray(x,dtype=jnp.float32) for k,x in data.items()}
 probes={}
 for label in ['control','audit']:
  with np.load(R/f'{label}.npz') as d:data=jdata({k:d[k] for k in d.files})
  def loss(p,data=data):
   def acc(t,x):return t+one(p,x)[0],None
   return jax.lax.scan(acc,jnp.array(0.,dtype=jnp.float32),data)[0]/data['re'].shape[0]
  probes[label]=jax.jit(loss)
 def checkfinite(tree):return all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(tree))
 # Production-size gradient equivalence and SOAP initialization/serialization checks.
 rr=np.random.default_rng();rr.bit_generator.state=copy.deepcopy(BASE_META['rng_state']);raw_first=batch(rr,pool,32,128,48);first=jdata(raw_first);g,loss,terms,gn=grad(base,first)
 ref=original(base,am,av,jnp.asarray(287401),first)
 np.testing.assert_allclose(np.asarray(loss),np.asarray(ref[3]),rtol=1e-5,atol=1e-7)
 expected_m=jax.tree_util.tree_map(lambda m,x:.9*m+.1*x,am,g)
 for x,y in zip(jax.tree_util.tree_leaves(expected_m),jax.tree_util.tree_leaves(ref[1])):np.testing.assert_allclose(np.asarray(x),np.asarray(y),rtol=1e-5,atol=1e-7)
 tx=make_optimizer(2e-5);up=jax.jit(lambda gg,ss,pp:tx.update(gg,ss,pp));st=tx.init(base)
 u,st=up(g,st,base);assert all(np.all(np.asarray(x)==0) for x in jax.tree_util.tree_leaves(u))
 u,st=up(g,st,base);pp=optax.apply_updates(base,u);assert checkfinite((pp,st));assert any(not np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(jax.tree_util.tree_leaves(base),jax.tree_util.tree_leaves(pp)))
 test=R/'preflight_state.npz';save_full(test,pp,cfg,287401,rr,mean,std,st,2e-5,0.);rp,rs,_,_=restore_full(test,tx)
 for x,y in zip(jax.tree_util.tree_leaves((pp,st)),jax.tree_util.tree_leaves((rp,rs))):np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
 a=up(g,st,pp);b=up(g,rs,rp)
 for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)):np.testing.assert_array_equal(np.asarray(x),np.asarray(y))
 assert checkfinite(a);assert any(not np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(jax.tree_util.tree_leaves(st[0].exp_avg),jax.tree_util.tree_leaves(a[1][0].exp_avg)))
 write(R/'preflight.json',dict(passed=True,production_gradient_matches_original=True,initialization_zero_update=True,actual_update_finite_nonzero=True,serialization_exact=True,resumed_next_update_exact=True,off_precondition_state_updates=True,full_state_bytes=test.stat().st_size,jax=jax.__version__,device=str(jax.devices()[0]),first_batch_sha256=hashlib.sha256(b''.join(np.asarray(raw_first[k]).tobytes() for k in sorted(raw_first))).hexdigest()))
 test.unlink()
 dev,_=load_pool(P/'data/geometries','dev');refs=[]
 population=json.loads((OLD/'repo/reports/adam_next50k_comparison/selection.json').read_text())['population']
 for x in population:
  with np.load(OLD/'repo/reports'/x['group']/x['directory']/'fields.npz') as d:refs.append((str(x['case_id']),d['coeff'].copy(),30.,d['xy'].copy(),d['weights'].copy(),d['fem'].copy()))
 for i in range(8):
  with np.load(P/f'runs/adam_next50k/development/{i:04d}.npz') as d:refs.append((f'pilot_{i}',dev['coeff'][i],float(dev['reynolds'][i]),d['xy'].copy(),d['weights'].copy(),d['truth'].copy()))
 def fem(path):
  model=FlowOperator(path,allow_unvalidated=True,chunk_size=256);rows=[]
  for name,coeff,re,xy,w,truth in refs:
   pred=model.evaluate(coeff,re,xy)
   def rel(a,b):
    ww=w.reshape((-1,)+(1,)*(a.ndim-1));return float(np.sqrt(np.sum(ww*(a-b)**2)/max(np.sum(ww*b*b),1e-16)))
   rows.append(dict(case=name,velocity_l2=rel(pred[:,:2],truth[:,:2]),pressure_l2=rel(pred[:,2],truth[:,2])))
  result={'cases':rows}
  for group,rrr in [('saved20',rows[:20]),('pilot8',rows[20:])]:result[group]={f'{field}_{stat}':float(fun([x[field] for x in rrr])) for field in ['velocity_l2','pressure_l2'] for stat,fun in [('median',np.median),('max',np.max)]}
  return result
 for name,lr in [('soap_2e5',2e-5),('soap_1e4',1e-4)]:
  folder=R/name;folder.mkdir();snap=folder/'snapshots';snap.mkdir();p=base;tx=make_optimizer(lr);update=jax.jit(lambda gg,ss,pp:tx.update(gg,ss,pp));state=tx.init(p)
  rng=np.random.default_rng();rng.bit_generator.state=copy.deepcopy(BASE_META['rng_state']);counter=287400;begin=time.monotonic()
  # One extra gradient evaluation initializes the basis without consuming the common training RNG.
  u,state=update(g,state,p);assert all(np.all(np.asarray(x)==0) for x in jax.tree_util.tree_leaves(u))
  def persist(snapshot=False):
   path=folder/'checkpoint.npz';save_full(path,p,cfg,counter,rng,mean,std,state,lr,time.monotonic()-begin)
   write(R/'status.json',dict(phase='running',branch=name,step=counter,offset=counter-287400,target_step=292400,learning_rate=lr,elapsed_seconds=time.monotonic()-begin))
   if snapshot:
    dst=snap/f'checkpoint_{counter}.npz';shutil.copy2(path,dst);write(dst.with_suffix('.ready.json'),{'step':counter,'sha256':sha(dst),'bytes':dst.stat().st_size,'optimizer':'soap'})
  def evaluate():
   v=dict(step=counter,offset=counter-287400,control=float(probes['control'](p)),audit=float(probes['audit'](p)),elapsed_seconds=time.monotonic()-begin)
   with (folder/'probe.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
   assert np.isfinite([v['control'],v['audit']]).all();persist()
  evaluate()
  with (folder/'training.jsonl').open('a',buffering=1) as log:
   for counter in range(287401,292401):
    data=jdata(batch(rng,pool,32,128,48));t=time.monotonic();gg,value,tt,nn=grad(p,data);value=float(value)
    if not np.isfinite(value):raise RuntimeError('Nonfinite physics loss; previous verified checkpoint retained')
    updates,state=update(gg,state,p);p=optax.apply_updates(p,updates)
    # Synchronize parameters for meaningful timing including preconditioning.
    jax.block_until_ready(p)
    if counter==287401 or counter%10==0:log.write(json.dumps(dict(step=counter,loss=value,terms=np.asarray(tt).tolist(),gradient_norm=float(nn),learning_rate=lr,step_seconds=time.monotonic()-t,elapsed_seconds=time.monotonic()-begin))+'\n')
    if counter%100==0:write(R/'status.json',dict(phase='running',branch=name,step=counter,offset=counter-287400,target_step=292400,learning_rate=lr,elapsed_seconds=time.monotonic()-begin))
    if counter==287500 or (counter-287400)%1000==0:persist(snapshot=True)
    if (counter-287400)%1000==0:evaluate()
    if (counter-287400)%2500==0:
     persist(snapshot=True);write(folder/f'fem_{counter}.json',fem(folder/'checkpoint.npz'))
    if time.monotonic()-begin>3600:raise RuntimeError('One-hour branch budget reached; checkpoint preserved')
  assert checkfinite((p,state));write(folder/'completed.json',dict(completed=True,step=counter,parameter_updates=5000,initialization_gradient_evaluations=1,elapsed_seconds=time.monotonic()-begin))
 write(R/'status.json',dict(phase='completed',branches=['soap_2e5','soap_1e4'],updates_each=5000,step=292400,automatic_promotion=False))
if __name__=='__main__':
 try:main()
 except BaseException as e:write(R/'failure.json',{'error':type(e).__name__,'message':str(e)});raise
