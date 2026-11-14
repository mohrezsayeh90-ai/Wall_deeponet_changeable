"""Compare one global Adam update with two-device gradient averaging.

CPU test: XLA_FLAGS=--xla_force_host_platform_device_count=2 python -m scripts.verify_parallel
On GPU, requires at least two devices. This script does no training beyond one
matched update per path. Never reduces the declared global batch.
"""
import json
import jax
import jax.numpy as jnp
import numpy as np
from geometry_operator.geometry import generate,sensors
from geometry_operator.model import Config,initialize
from geometry_operator.train import make_step

def main():
    if len(jax.local_devices())<2:raise RuntimeError('Two devices required')
    c=Config(sensors=65,width=8,depth=1,rank=4);p=initialize(c);zero=jax.tree_util.tree_map(jnp.zeros_like,p)
    coeff=np.array([generate(np.random.default_rng(i),i) for i in [1,5]])
    data=dict(coeff=jnp.asarray(coeff),sensors=jnp.asarray(np.array([sensors(x,65) for x in coeff])),re=jnp.array([30.,80.]),
        z=jnp.array([[[.3,.4],[.6,.7]],[[.4,.3],[.7,.6]]]),outlet=jnp.array([[.2,.6],[.3,.8]]))
    serial,_=make_step(c,jnp.zeros(130),jnp.ones(130));parallel,_=make_step(c,jnp.zeros(130),jnp.ones(130),'devices')
    expected=serial(p,zero,zero,1,data)
    replicate=lambda tree:jax.tree_util.tree_map(lambda x:jnp.broadcast_to(x,(2,)+x.shape),tree)
    sharded={k:x.reshape((2,1)+x.shape[1:]) for k,x in data.items()}
    actual=parallel(replicate(p),replicate(zero),replicate(zero),1,sharded)
    worst=0.
    for a,b in zip(jax.tree_util.tree_leaves(expected),jax.tree_util.tree_leaves(actual)):
        np.testing.assert_allclose(a,b[0],rtol=1e-10,atol=1e-10)
        worst=max(worst,float(np.max(abs(np.asarray(a)-np.asarray(b[0])))))
    print(json.dumps(dict(serial_parallel_update_match=True,max_absolute_difference=worst,devices=[str(x) for x in jax.devices()])))
if __name__=='__main__':main()
