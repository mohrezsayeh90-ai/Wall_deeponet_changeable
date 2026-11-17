"""Only changed hyperparameter: LR as a function of paired-trial offset."""
import json,os
from pathlib import Path
import jax.numpy as jnp
SPEC=json.loads(Path(os.environ['WALL_LR_CONFIG']).read_text())

def learning_rate(count):
    progress=jnp.clip((count-SPEC['base_step']-1)/14999.,0.,1.)
    if SPEC['kind']=='constant':return jnp.asarray(SPEC['high'],dtype=jnp.float32)
    if SPEC['kind']=='cosine':return SPEC['low']+.5*(SPEC['high']-SPEC['low'])*(1+jnp.cos(jnp.pi*progress))
    raise ValueError('Unsupported schedule')
