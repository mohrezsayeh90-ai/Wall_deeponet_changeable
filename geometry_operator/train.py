"""Physics-only global training with microbatch gradient accumulation.

No FEM arrays or field targets enter this module. Full wall pool is fixed;
Re and collocation points regenerate each update. Losses are unweighted
strong-form momentum plus 10 times natural outlet traction.
"""
import argparse, json, time, platform, subprocess
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .model import Config,initialize,context
from .geometry import forward
from .operators import residual,traction
from .sampling import load_pool,batch,sha
from .checkpoint import save,load


def make_step(config,mean,std):
    def one(params,item):
        coeff,samples,re,z,eta=(item[k] for k in ['coeff','sensors','re','z','outlet'])
        b=context(params,samples,re,mean,std)
        rr=jax.vmap(lambda q:residual(params,b,coeff,re,forward(coeff,q),config))(z)
        tt=jax.vmap(lambda q:traction(params,b,coeff,re,q,config))(eta)
        terms=jnp.array([jnp.mean(jnp.sum(rr[:,:2]**2,axis=1)),jnp.mean(tt[:,0]**2),jnp.mean(tt[:,1]**2),jnp.mean(rr[:,2]**2)])
        return terms[0]+10*(terms[1]+terms[2]),terms
    vg=jax.value_and_grad(one,has_aux=True)
    def step(params,m,v,count,data):
        zero=jax.tree_util.tree_map(jnp.zeros_like,params)
        def accum(carry,item):
            gs,ls,ts=carry;(loss,terms),grad=vg(params,item)
            return (jax.tree_util.tree_map(jnp.add,gs,grad),ls+loss,ts+terms),None
        (g,loss,terms),_=jax.lax.scan(accum,(zero,jnp.array(0.,dtype=getattr(jnp,config.dtype)),jnp.zeros(4,dtype=getattr(jnp,config.dtype))),data)
        n=data['re'].shape[0];g=jax.tree_util.tree_map(lambda x:x/n,g);loss/=n;terms/=n
        norm=jnp.sqrt(sum(jnp.sum(x*x) for x in jax.tree_util.tree_leaves(g)))
        g=jax.tree_util.tree_map(lambda x:x*jnp.minimum(1.,10/(norm+1e-20)),g)
        m=jax.tree_util.tree_map(lambda a,b:.9*a+.1*b,m,g);v=jax.tree_util.tree_map(lambda a,b:.999*a+.001*b*b,v,g)
        lr=jnp.maximum(2e-5,1e-3*.95**(count/2000))
        params=jax.tree_util.tree_map(lambda p,a,b:p-lr*(a/(1-.9**count))/(jnp.sqrt(b/(1-.999**count))+1e-8),params,m,v)
        return params,m,v,loss,terms,norm
    return jax.jit(step),jax.jit(one)


def train(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    pool,manifest=load_pool(args.pool);norm=np.load(Path(args.pool)/'normalization.npz')
    config=Config(**json.loads(Path(args.config).read_text())['model'])
    params=initialize(config);m=jax.tree_util.tree_map(jnp.zeros_like,params);v=m;start=0
    mean,std=jnp.asarray(norm['mean'],dtype=getattr(jnp,config.dtype)),jnp.asarray(norm['std'],dtype=getattr(jnp,config.dtype));rng=np.random.default_rng(config.seed+1)
    if args.resume:
        params,loaded,meta,(mean,std),state=load(args.resume,True)
        if loaded!=config or meta['train_pool_sha256']!=manifest['splits']['train']['sha256']:
            raise ValueError('Resume config/pool mismatch')
        m,v=state;start=meta['step'];rng.bit_generator.state=meta['rng_state']
    if args.require_gpu and jax.default_backend()!='gpu':raise RuntimeError('GPU required; refusing CPU main run')
    step,_=make_step(config,mean,std);begin=time.monotonic();compiled=None
    metadata=dict(train_pool_sha256=manifest['splits']['train']['sha256'],
        code_revision=args.code_revision,accepted=False,training_labels=0,backend=jax.default_backend(),
        devices=[str(d) for d in jax.devices()],jax=jax.__version__,python=platform.python_version(),
        target_steps=args.steps,global_batch=args.batch,interior_points=args.points,outlet_points=args.outlet,
        sampling='biased mixture; unweighted strong momentum and traction',selection='latest; FEM selection pending')
    Path(output/'runtime.json').write_text(json.dumps(metadata,indent=2))
    def persist(count,reason):
        metadata.update(elapsed_seconds=time.monotonic()-begin,compilation_seconds=compiled,stop_reason=reason)
        save(output/'checkpoint.npz',params,config,count,rng.bit_generator.state,mean,std,m,v,metadata)
        (output/'status.json').write_text(json.dumps(dict(step=count,**metadata),indent=2))
    with open(output/'training.jsonl','a',buffering=1) as log:
        count=start
        try:
            for count in range(start+1,args.steps+1):
                data={k:jnp.asarray(x,dtype=getattr(jnp,config.dtype)) for k,x in batch(rng,pool,args.batch,args.points,args.outlet).items()}
                t=time.monotonic();params,m,v,value,terms,gn=step(params,m,v,jnp.asarray(count),data)
                value=float(value);elapsed=time.monotonic()-t
                if compiled is None:compiled=elapsed
                if not np.isfinite(value):raise RuntimeError('Nonfinite training loss')
                if count==start+1 or count%args.log_every==0:
                    record=dict(step=count,loss=value,terms=np.asarray(terms).tolist(),gradient_norm=float(gn),step_seconds=elapsed,elapsed_seconds=time.monotonic()-begin)
                    log.write(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
                if count%args.save_every==0 or count==start+1:persist(count,'running')
                if time.monotonic()-begin>args.max_seconds:
                    persist(count,'wall_clock_budget');break
            else:persist(count,'target_completed')
        except BaseException:
            # State may be nonfinite on a numerical failure: retain prior atomic
            # checkpoint, save failure metadata without overwriting it.
            (output/'failure.json').write_text(json.dumps(dict(last_attempt=count,elapsed_seconds=time.monotonic()-begin)))
            raise
    return output/'checkpoint.npz'


def main():
    p=argparse.ArgumentParser();p.add_argument('--pool',default='data/geometries');p.add_argument('--config',default='configs/production.json')
    p.add_argument('--output',default='runs/main');p.add_argument('--steps',type=int,default=100000);p.add_argument('--batch',type=int,default=32)
    p.add_argument('--points',type=int,default=128);p.add_argument('--outlet',type=int,default=48)
    p.add_argument('--save-every',type=int,default=500);p.add_argument('--log-every',type=int,default=100)
    p.add_argument('--max-seconds',type=float,default=18000);p.add_argument('--resume');p.add_argument('--require-gpu',action='store_true');p.add_argument('--code-revision',default='uncommitted')
    a=p.parse_args()
    if min(a.steps,a.batch,a.points,a.outlet,a.save_every,a.log_every)<1:p.error('Positive counts required')
    train(a)
if __name__=='__main__':main()
