"""Atomic non-pickle checkpoints including exact RNG and optimizer state."""
import json
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .model import Config, initialize

FORMAT='geometry-pi-deeponet-v1'


def save(path,params,config,step,rng_state,mean,std,m=None,v=None,metadata=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    arrays=dict(mean=np.asarray(mean),std=np.asarray(std))
    for name,tree in [('param',params),('adam_m',m),('adam_v',v)]:
        if tree is not None:
            arrays.update({f'{name}_{i}':np.asarray(x) for i,x in enumerate(jax.tree_util.tree_leaves(tree))})
    arrays['metadata']=np.array(json.dumps(dict(format=FORMAT,config=config.json(),step=int(step),rng_state=rng_state,
        optimizer_present=m is not None,**(metadata or {}))))
    tmp=path.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arrays);tmp.replace(path)


def load(path,optimizer=False):
    with np.load(path,allow_pickle=False) as d:
        meta=json.loads(str(d['metadata']))
        if meta['format']!=FORMAT:raise ValueError('Incompatible checkpoint')
        config=Config(**meta['config']);flat,structure=jax.tree_util.tree_flatten(initialize(config))
        def tree(prefix):
            xs=[jnp.asarray(d[f'{prefix}_{i}']) for i in range(len(flat))]
            if any(x.shape!=y.shape for x,y in zip(xs,flat)):raise ValueError('Corrupt checkpoint shapes')
            return jax.tree_util.tree_unflatten(structure,xs)
        state=None
        if optimizer:
            if not meta['optimizer_present']:raise ValueError('No optimizer state')
            state=(tree('adam_m'),tree('adam_v'))
        return tree('param'),config,meta,(d['mean'].copy(),d['std'].copy()),state
