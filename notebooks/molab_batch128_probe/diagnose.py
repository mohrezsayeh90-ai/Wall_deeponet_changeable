"""Read-only same-state precision and stochastic-gradient diagnostics."""
from pathlib import Path
import os,sys,json,time,copy,hashlib,shutil,importlib.util
import numpy as np
R=Path('/home/marimo/wall_precision_noise_diag_v1');CUR=Path('/home/marimo/wall_soap_continue20k_v1');PREV=Path('/home/marimo/wall_soap_trials_v1');P=Path('/home/marimo/wall_rad_ab_15k_v1/frozen/output/project')
spec=importlib.util.spec_from_file_location('soap_helper',PREV/'worker.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h);jax,jnp=h.jax,h.jnp
from geometry_operator.checkpoint import load
from geometry_operator.sampling import load_pool,batch
from geometry_operator.soap_gradient import make_step
ready=sorted((CUR/'continuation/snapshots').glob('*.ready.json'),key=lambda p:int(p.name.split('_')[1].split('.')[0]))[-1];src=ready.with_name(ready.name.replace('.ready.json','.npz'));meta=json.loads(ready.read_text());assert h.sha(src)==meta['sha256'];shutil.copy2(src,R/'base.npz')
p,cfg,md,(mean,std),_=load(R/'base.npz');pool,_=load_pool(P/'data/geometries');rng=np.random.default_rng();rng.bit_generator.state=md['rng_state'];jd=lambda d:{k:jnp.asarray(x,dtype=jnp.float32) for k,x in d.items()}
samples=[jd(batch(rng,pool,32,128,48)) for _ in range(8)];grad,one=make_step(cfg,mean,std)
def flat(tree):return np.concatenate([np.asarray(x).reshape(-1) for x in jax.tree_util.tree_leaves(tree)]).astype(np.float64)
out={'step':md['step'],'source_sha256':meta['sha256'],'dtype':cfg.dtype,'jax':jax.__version__,'default_matmul_precision':str(jax.config.jax_default_matmul_precision),'samples':8,'batch_size':32,'read_only':True}
start=time.monotonic();gs=[];loss=[];gn=[];groups={k:[] for k in p}
for i,data in enumerate(samples):
 g,l,t,n=grad(p,data);gs.append(flat(g));loss.append(float(l));gn.append(float(n))
 for k in p:groups[k].append(flat(g[k]))
 print('MICROBATCH',i,float(l),float(n),flush=True)
G=np.array(gs);m=G.mean(0);signal=float(m@m);variance=float(np.sum(np.var(G,axis=0,ddof=1)));gram=G@G.T;norm=np.sqrt(np.maximum(np.diag(gram),1e-30));cos=gram/(norm[:,None]*norm[None,:])
out.update(losses=loss,gradient_norms=gn,mean_gradient_norm=float(np.sqrt(signal)),gradient_noise_trace=variance,noise_to_squared_mean_ratio=variance/max(signal,1e-30),cosines=cos.tolist(),measured_default_seconds=time.monotonic()-start)
out['groups']={}
for k,v in groups.items():
 a=np.array(v);mm=a.mean(0);ss=float(mm@mm);vv=float(np.sum(np.var(a,axis=0,ddof=1)));out['groups'][k]={'mean_grad_norm':float(np.sqrt(ss)),'noise_to_squared_mean_ratio':vv/max(ss,1e-30)}
# The identical parameters and first batch, with a new JIT precision context.
with jax.default_matmul_precision('highest'):
 gh,lh,th,nh=grad(p,samples[0]);gh=flat(gh);lh=float(lh)
out['highest_vs_default']={'default_loss':loss[0],'highest_loss':lh,'relative_loss_change':float((lh-loss[0])/max(abs(loss[0]),1e-30)),'relative_gradient_difference':float(np.linalg.norm(gh-G[0])/max(np.linalg.norm(G[0]),1e-30)),'gradient_cosine':float(gh@G[0]/max(np.linalg.norm(gh)*np.linalg.norm(G[0]),1e-30))}
(R/'diagnostic.json').write_text(json.dumps(out,indent=2));print('DIAGNOSTIC_COMPLETE',json.dumps(out),flush=True)
