"""Compare precision on full-size model and profile memory-bounded updates."""
import gc,json,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .model import Config,initialize,context
from .geometry import generate,sensors,forward
from .operators import residual
from .train import make_step


def run(output):
    coeff=generate(np.random.default_rng(721),5);ss=sensors(coeff)
    results={};preds={}
    for dtype in ['float64','float32']:
        dt=getattr(jnp,dtype);config=Config(dtype=dtype);p=initialize(config)
        c=jnp.asarray(coeff,dtype=dt);s=jnp.asarray(ss,dtype=dt)
        mean=jnp.zeros(514,dtype=dt);std=jnp.ones(514,dtype=dt)
        b=context(p,s,jnp.asarray(55.,dtype=dt),mean,std)
        fn=jax.jit(lambda z:residual(p,b,c,jnp.asarray(55.,dtype=dt),forward(c,z),config))
        preds[dtype]=np.asarray(jax.vmap(fn)(jnp.array([[.3,.2],[.5,.4],[.7,.8]],dtype=dt)))
        step,_=make_step(config,mean,std);zero=jax.tree_util.tree_map(jnp.zeros_like,p)
        data=dict(coeff=c[None],sensors=s[None],re=jnp.array([55.],dtype=dt),
            z=jnp.asarray(np.random.default_rng(1).uniform(.01,.99,(1,128,2)),dtype=dt),
            outlet=jnp.asarray(np.random.default_rng(2).uniform(0,1,(1,48)),dtype=dt))
        t=time.monotonic();out=step(p,zero,zero,1,data);jax.block_until_ready(out);compile_seconds=time.monotonic()-t
        t=time.monotonic();out=step(*out[:3],2,data);jax.block_until_ready(out);seconds=time.monotonic()-t
        results[dtype]=dict(compilation_seconds=compile_seconds,seconds_per_geometry=seconds,
            estimated_100k_global_seconds=seconds*32*100000,loss=float(out[3]),finite=bool(np.isfinite(float(out[3]))))
        del fn,step,p,out,zero;gc.collect();jax.clear_caches()
    err=float(np.max(abs(preds['float32']-preds['float64'])))
    relative=float(np.linalg.norm(preds['float32']-preds['float64'])/max(np.linalg.norm(preds['float64']),1e-12))
    # Both absolute and relative tolerances must pass, and training must be finite.
    chosen='float32' if err<.001 and relative<.01 and results['float32']['finite'] else 'float64'
    result=dict(results=results,residual_max_precision_difference=err,residual_relative_precision_difference=relative,
        selected_dtype=chosen,backend=jax.default_backend(),jax=jax.__version__,devices=[str(x) for x in jax.devices()],
        precision_scope='initialization,3 interior points; final checkpoint must separately pass float64 validation')
    Path(output).write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return result
