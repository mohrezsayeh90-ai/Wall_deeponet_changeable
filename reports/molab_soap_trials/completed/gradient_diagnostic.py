from pathlib import Path
import sys,json,hashlib
import numpy as np
R=Path('/home/marimo/wall_soap_trials_v1');P=Path('/home/marimo/wall_rad_ab_15k_v1/frozen/output/project');sys.path.insert(0,str(P))
import jax,jax.numpy as jnp
jax.config.update('jax_enable_x64',True)
from geometry_operator.checkpoint import load
from geometry_operator.sampling import batch,load_pool
from geometry_operator.model import context
from geometry_operator.geometry import forward
from geometry_operator.operators import residual,traction
base,cfg,md,(mean,std),_=load(R/'base.npz');pool,_=load_pool(P/'data/geometries');rng=np.random.default_rng();rng.bit_generator.state=md['rng_state'];raw=batch(rng,pool,32,128,48);data={k:jnp.asarray(v,dtype=jnp.float32) for k,v in raw.items()}
def one(p,item):
 b=context(p,item['sensors'],item['re'],mean,std)
 rr=jax.vmap(lambda z:residual(p,b,item['coeff'],item['re'],forward(item['coeff'],z),cfg))(item['z'])
 tt=jax.vmap(lambda q:traction(p,b,item['coeff'],item['re'],q,cfg))(item['outlet'])
 return jnp.array([jnp.mean(rr[:,0]**2),jnp.mean(rr[:,1]**2),10*jnp.mean(tt[:,0]**2),10*jnp.mean(tt[:,1]**2)])
jac=jax.jacrev(one)
@jax.jit
def summary(p):
 zeros=jax.tree_util.tree_map(lambda x:jnp.zeros((4,)+x.shape,dtype=x.dtype),p)
 def accum(carry,item):
  g,val=carry;v=one(p,item);gg=jac(p,item)
  return (jax.tree_util.tree_map(jnp.add,g,gg),val+v),None
 (g,v),_=jax.lax.scan(accum,(zeros,jnp.zeros(4,dtype=jnp.float32)),data)
 g=jax.tree_util.tree_map(lambda x:x/32,g);v/=32
 gram=sum(x.reshape(4,-1)@x.reshape(4,-1).T for x in jax.tree_util.tree_leaves(g));norm=jnp.sqrt(jnp.maximum(jnp.diag(gram),0));cos=gram/jnp.maximum(norm[:,None]*norm[None,:],1e-30)
 return v,norm,cos
out={'terms':['x_momentum','y_momentum','weighted_outlet_tx','weighted_outlet_ty'],'scope':'One identical fresh training batch, 32 geometries. Diagnostic only, not a full distribution estimate. Unclipped per-term gradients; no optimizer updates.','batch_sha256':hashlib.sha256(b''.join(raw[k].tobytes() for k in sorted(raw))).hexdigest(),'methods':{}}
for name,path in [('base',R/'base.npz'),('soap_2e5',R/'soap_2e5/snapshots/checkpoint_292400.npz')]:
 p,_,meta,_,_=load(path);val,norm,cos=summary(p);out['methods'][name]={'step':meta['step'],'loss_terms':np.asarray(val).tolist(),'gradient_norms':np.asarray(norm).tolist(),'gradient_cosines':np.asarray(cos).tolist()}
(R/'gradient_diagnostic.json').write_text(json.dumps(out,indent=2));print('DIAGNOSTIC_COMPLETE',json.dumps(out),flush=True)
