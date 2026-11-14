"""Frozen physical-coordinate inference, with explicit validation status."""
import json, hashlib
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .checkpoint import load
from .geometry import validate,sensors,sample_numpy,reconstruct_samples
from .model import context,fields


class FlowOperator:
    def __init__(self,checkpoint,allow_unvalidated=False,chunk_size=256):
        self.path=Path(checkpoint);self.sha256=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.params,self.config,self.metadata,(self.mean,self.std),_=load(checkpoint)
        rp=self.path.with_suffix('.validation.json');report=json.loads(rp.read_text()) if rp.exists() else {}
        self.accepted=report.get('checkpoint_sha256')==self.sha256 and report.get('all_gates_passed') is True
        if not self.accepted and not allow_unvalidated:
            raise ValueError('Checkpoint has no passing independent benchmark; use allow_unvalidated=True for research')
        if chunk_size<1:raise ValueError('chunk_size must be positive')
        self.chunk_size=chunk_size
        self._branch=jax.jit(lambda s,r:context(self.params,s,r,self.mean,self.std))
        self._fields=jax.jit(lambda b,c,r,q:jax.vmap(lambda xy:fields(self.params,b,c,r,xy,self.config))(q))
        self._cache={}

    def evaluate(self,walls,reynolds,xy,allow_stress=False):
        validate(walls,allow_stress)
        if not np.isfinite(reynolds) or not 10<=reynolds<=100:raise ValueError('Re must lie in [10,100]')
        xy=np.asarray(xy,dtype=float)
        if xy.ndim!=2 or xy.shape[1]!=2 or not np.isfinite(xy).all():raise ValueError('xy must be finite (N,2) physical coordinates')
        if not len(xy):return np.empty((0,3))
        c,dg=sample_numpy(walls,xy[:,0]);b=.5+c-(1+dg)/2;t=b+1+dg
        if np.any((xy[:,0]<0)|(xy[:,0]>5)|(xy[:,1]<b-1e-10)|(xy[:,1]>t+1e-10)):
            raise ValueError('Query lies outside the physical channel')
        key=(hashlib.sha256(np.asarray(walls).tobytes()).hexdigest(),float(reynolds))
        if key not in self._cache:
            if len(self._cache)>128:self._cache.clear()
            self._cache[key]=self._branch(jnp.asarray(sensors(walls,self.config.sensors)),jnp.asarray(float(reynolds)))
        branch=self._cache[key];result=[]
        for start in range(0,len(xy),self.chunk_size):
            q=xy[start:start+self.chunk_size];n=len(q);q=np.pad(q,((0,self.chunk_size-n),(0,0)),mode='edge')
            result.append(np.asarray(self._fields(branch,jnp.asarray(walls),jnp.asarray(float(reynolds)),jnp.asarray(q)))[:n])
        result=np.concatenate(result)
        if not np.isfinite(result).all():raise RuntimeError('Nonfinite prediction')
        return result

    def evaluate_samples(self,bottom,top,reynolds,xy):
        coeff,diagnostics=reconstruct_samples(bottom,top)
        return self.evaluate(coeff,reynolds,xy),diagnostics
