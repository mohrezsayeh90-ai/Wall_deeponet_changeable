"""Development-only linear projection diagnostic; no model training or promotion."""
from pathlib import Path
import sys,json
import numpy as np
R=Path('/home/marimo/wall_trunk_capacity_diag_v1');OLD=Path('/home/marimo/wall_rad_ab_15k_v1');P=OLD/'frozen/output/project';sys.path.insert(0,str(P))
import jax,jax.numpy as jnp
jax.config.update('jax_enable_x64',True)
from geometry_operator.checkpoint import load
from geometry_operator.geometry import inverse,local
from geometry_operator.model import mlp,context
from geometry_operator.api import FlowOperator
source=Path('/home/marimo/wall_batch128_probe_v1/small_batch_reference.npz');p,cfg,md,(mean,std),_=load(source)
from geometry_operator.geometry import sensors
@jax.jit
def basis(coeff,xy):
 def at(q):
  z=inverse(coeff,q);xi,eta=z;c,g=local(coeff,5*xi);cp,gp=local(coeff,5*xi,1)
  feat=jnp.concatenate([2*z-1,jnp.array([c/.35,g/.5,(cp-gp/2)/.7,(cp+gp/2)/.7])])
  mask=(-jnp.expm1(-xi/cfg.inlet_length))**2*16*eta**2*(1-eta)**2
  return (3*eta**2-2*eta**3),mask*mlp(p['psi_trunk'],feat)/jnp.sqrt(float(cfg.rank)),mlp(p['p_trunk'],feat)/jnp.sqrt(float(cfg.rank))
 def one(q):
  base,psi,pres=at(q);d0=jax.jacfwd(lambda z:at(z)[0])(q);dpsi=jax.jacfwd(lambda z:at(z)[1])(q)
  return jnp.array([d0[1],-d0[0]]),jnp.stack([dpsi[:,1],-dpsi[:,0]]),pres
 return jax.vmap(one)(xy)
model=FlowOperator(source,allow_unvalidated=True,chunk_size=256);population=json.loads((OLD/'repo/reports/adam_next50k_comparison/selection.json').read_text())['population'];out={'scope':'Four already-used development examples. Ridge projection against FEM uses a fit split and a disjoint point split; not a trained network or accepted result. No model parameters are changed.','checkpoint_step':md['step'],'cases':[]}
for row in population:
 if row['case_id'] not in [1,12,16,20]:continue
 with np.load(OLD/'repo/reports'/row['group']/row['directory']/'fields.npz') as d:coeff,xy,w,truth=d['coeff'],d['xy'],d['weights'],d['fem']
 rng=np.random.default_rng(20261119+row['case_id']);ix=rng.choice(len(xy),min(3000,len(xy)),replace=False);xy,w,truth=xy[ix],w[ix],truth[ix];u0,ub,pb=basis(jnp.asarray(coeff,dtype=jnp.float32),jnp.asarray(xy,dtype=jnp.float32));u0,ub,pb=map(np.asarray,[u0,ub,pb]);n=len(xy)//2;fit=np.arange(n);test=np.arange(n,len(xy));pred=model.evaluate(coeff,30.,xy)
 def relative(a,b,weights):
  ww=weights.reshape((-1,)+(1,)*(a.ndim-1));return float(np.sqrt(np.sum(ww*(a-b)**2)/max(np.sum(ww*b*b),1e-16)))
 def ridge(A,y,ww):
  Aw=A*np.sqrt(ww)[:,None];yw=y*np.sqrt(ww);lam=1e-8*np.sum(Aw*Aw)/A.shape[1];return np.linalg.lstsq(np.vstack([Aw,np.sqrt(lam)*np.eye(A.shape[1])]),np.r_[yw,np.zeros(A.shape[1])],rcond=None)[0]
 cp=ridge(pb[fit],truth[fit,2]-12*(5-xy[fit,0])/30.,w[fit]);cu=ridge(ub[fit].reshape(-1,cfg.rank),(truth[fit,:2]-u0[fit]).reshape(-1),np.repeat(w[fit],2))
 pu=u0[test]+np.einsum('ijk,k->ij',ub[test],cu);pp=12*(5-xy[test,0])/30.+pb[test]@cp
 branch=np.asarray(context(p,jnp.asarray(sensors(coeff),dtype=jnp.float32),jnp.asarray(30.,dtype=jnp.float32),mean,std))
 rec={'case':row['case_id'],'online_velocity_l2':relative(pred[test,:2],truth[test,:2],w[test]),'online_pressure_l2':relative(pred[test,2],truth[test,2],w[test]),'projected_velocity_l2':relative(pu,truth[test,:2],w[test]),'projected_pressure_l2':relative(pp,truth[test,2],w[test]),'pressure_coefficient_norm_ratio':float(np.linalg.norm(cp)/max(np.linalg.norm(branch[1]),1e-30)),'velocity_coefficient_norm_ratio':float(np.linalg.norm(cu)/max(np.linalg.norm(branch[0]),1e-30))}
 out['cases'].append(rec);(R/'diagnostic.json').write_text(json.dumps(out,indent=2));print('CASE',json.dumps(rec),flush=True)
