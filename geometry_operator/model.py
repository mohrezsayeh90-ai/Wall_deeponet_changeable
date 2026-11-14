"""Independent gated branch/trunk pairs; hard physical streamfunction ansatz."""
from dataclasses import dataclass, asdict
import jax
import jax.numpy as jnp
import numpy as np
from .geometry import local, inverse

@dataclass(frozen=True)
class Config:
    sensors: int = 257
    width: int = 192
    depth: int = 4
    rank: int = 128
    seed: int = 20261115
    dtype: str = 'float64'
    local_features: bool = True
    inlet_length: float = .05

    def json(self):return asdict(self)


def initialize(config):
    rng=np.random.default_rng(config.seed);dt=getattr(jnp,config.dtype)
    def layer(n,m,small=False):
        return dict(w=jnp.asarray(rng.normal(0,1e-3 if small else np.sqrt(2/(n+m)),(n,m)),dtype=dt),b=jnp.zeros(m,dtype=dt))
    def net(n):return dict(u=layer(n,config.width),v=layer(n,config.width),
        gates=[layer(n if i==0 else config.width,config.width) for i in range(config.depth)],out=layer(config.width,config.rank,True))
    return {key:net(2*config.sensors+1 if 'branch' in key else (6 if config.local_features else 2))
            for key in ['psi_branch','psi_trunk','p_branch','p_trunk']}


def mlp(p,x):
    dense=lambda q,z:z@q['w']+q['b']
    u,v=jnp.tanh(dense(p['u'],x)),jnp.tanh(dense(p['v'],x))
    h=x
    for q in p['gates']:
        z=jnp.tanh(dense(q,h));h=z*u+(1-z)*v
    return dense(p['out'],h)


def context(params,wall_samples,re,mean,std):
    x=jnp.concatenate([(wall_samples-mean)/std,jnp.array([jnp.log(re/10)/jnp.log(10.)])])
    return jnp.stack([mlp(params['psi_branch'],x),mlp(params['p_branch'],x)])


def potentials(params,branch,coeff,re,z,config):
    xi,eta=z;x=5*xi
    features=2*z-1
    if config.local_features:
        c,g=local(coeff,x);cp,gp=local(coeff,x,1)
        features=jnp.concatenate([features,jnp.array([c/.35,g/.5,(cp-gp/2)/.7,(cp+gp/2)/.7])])
    npsi=jnp.sum(branch[0]*mlp(params['psi_trunk'],features))/jnp.sqrt(float(config.rank))
    npres=jnp.sum(branch[1]*mlp(params['p_trunk'],features))/jnp.sqrt(float(config.rank))
    mask=(-jnp.expm1(-xi/config.inlet_length))**2*16*eta**2*(1-eta)**2
    return jnp.array([3*eta**2-2*eta**3+mask*npsi,12*(5-x)/re+npres])


def physical_potentials(params,branch,coeff,re,xy,config):
    return potentials(params,branch,coeff,re,inverse(coeff,xy),config)


def fields(params,branch,coeff,re,xy,config):
    fn=lambda pos:physical_potentials(params,branch,coeff,re,pos,config)
    dp=jax.jacfwd(fn)(xy)
    return jnp.array([dp[0,1],-dp[0,0],fn(xy)[1]])
