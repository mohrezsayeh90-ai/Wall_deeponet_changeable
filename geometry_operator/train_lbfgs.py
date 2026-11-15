"""Deterministic L-BFGS-B refinement of saved weights; no Adam updates or FEM labels."""
import argparse,json,time,hashlib
from dataclasses import replace
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize
from .checkpoint import load,save
from .sampling import load_pool,batch
from .model import context
from .geometry import forward
from .operators import residual,traction


class TimeBudget(Exception):pass


def make_objective(config,mean,std):
    def one(params,item):
        b=context(params,item['sensors'],item['re'],mean,std)
        rr=jax.vmap(lambda z:residual(params,b,item['coeff'],item['re'],forward(item['coeff'],z),config))(item['z'])
        tt=jax.vmap(lambda eta:traction(params,b,item['coeff'],item['re'],eta,config))(item['outlet'])
        return jnp.mean(jnp.sum(rr[:,:2]**2,axis=1))+10*jnp.mean(jnp.sum(tt**2,axis=1))
    vg=jax.value_and_grad(one)
    @jax.jit
    def objective(params,data):
        zero=jax.tree_util.tree_map(jnp.zeros_like,params)
        def accumulate(carry,item):
            value,gradient=vg(params,item)
            return (carry[0]+value,jax.tree_util.tree_map(jnp.add,carry[1],gradient)),None
        (value,gradient),_=jax.lax.scan(accumulate,(jnp.array(0.,dtype=jnp.float64),zero),data)
        n=data['re'].shape[0]
        return value/n,jax.tree_util.tree_map(lambda x:x/n,gradient)
    return objective


def fixed_banks(pool,count,probe_count,points,outlet,seed):
    if count+probe_count>len(pool['coeff']):raise ValueError('Disjoint banks require enough training geometries')
    rng=np.random.default_rng(seed);indices=rng.choice(len(pool['coeff']),count+probe_count,replace=False)
    banks=[]
    for ix in [indices[:count],indices[count:]]:
        data=batch(rng,pool,len(ix),points,outlet)
        data['coeff']=pool['coeff'][ix];data['sensors']=pool['sensors'][ix]
        banks.append(data)
    return banks,indices,rng.bit_generator.state


def train(args):
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    if (output/'status.json').exists():raise ValueError('Use a fresh output directory; refusing to overwrite an experiment')
    params,old_config,base,(mean,std),_=load(args.checkpoint)
    if base['step']!=args.expected_step:raise ValueError('Unexpected base checkpoint step')
    if args.require_gpu and jax.default_backend()!='gpu':raise RuntimeError('GPU required')
    config=replace(old_config,dtype='float64');params=jax.tree_util.tree_map(lambda x:x.astype(jnp.float64),params)
    mean,std=jnp.asarray(mean,dtype=jnp.float64),jnp.asarray(std,dtype=jnp.float64)
    pool,manifest=load_pool(args.pool)
    if base['train_pool_sha256']!=manifest['splits']['train']['sha256']:raise ValueError('Training pool mismatch')
    banks,indices,rng_state=fixed_banks(pool,args.fixed_geometries,args.probe_geometries,args.points,args.outlet,args.seed)
    for name,data,ix in zip(['fixed_bank','probe_bank'],banks,[indices[:args.fixed_geometries],indices[args.fixed_geometries:]]):
        np.savez_compressed(output/f'{name}.npz',**data,train_indices=ix)
    bank,probe=[{k:jnp.asarray(v,dtype=jnp.float64) for k,v in data.items()} for data in banks]
    objective=make_objective(config,mean,std);x0,unravel=ravel_pytree(params);x0=np.asarray(x0,dtype=np.float64)
    start=time.monotonic();state={'x':x0.copy(),'iteration':0,'evaluations':0,'loss':None};cache={}
    metadata=dict(optimizer='L-BFGS-B',optimizer_bounds=None,base_adam_step=base['step'],base_checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),train_pool_sha256=base['train_pool_sha256'],accepted=False,training_labels=0,dtype_transition=f'{old_config.dtype} weights promoted to float64 without changing their stored numerical values',fixed_geometries=args.fixed_geometries,probe_geometries=args.probe_geometries,interior_points=args.points,outlet_points=args.outlet,bank_seed=args.seed,history_size=args.history,max_iterations=args.maxiter,max_function_evaluations=args.maxfun,max_seconds=args.max_seconds,jax=jax.__version__,scipy=__import__('scipy').__version__,backend=jax.default_backend(),objective='Fixed train-only geometry/Re/interior/outlet bank; momentum + 10*traction; no gradient clipping',resume_note='Parameter-only L-BFGS restart supported; SciPy curvature history is not serialized. Adam moments are not used.',step_semantics='step retains base Adam count; lbfgs_iterations counts accepted refinement iterations')
    def persist(reason):
        meta=dict(metadata,lbfgs_iterations=state['iteration'],function_evaluations=state['evaluations'],fixed_loss=state['loss'],elapsed_seconds=time.monotonic()-start,stop_reason=reason)
        save(output/'checkpoint.npz',unravel(jnp.asarray(state['x'])),config,base['step'],rng_state,mean,std,metadata=meta)
        (output/'status.json').write_text(json.dumps(dict(step=base['step'],**meta),indent=2)+'\n')
    persist('initialized')
    def fg(x):
        if 'x' in cache and np.array_equal(x,cache['x']):return cache['value'],cache['gradient']
        if time.monotonic()-start>args.max_seconds:raise TimeBudget()
        t=time.monotonic();value,gradient=objective(unravel(jnp.asarray(x)),bank);gradient=np.asarray(ravel_pytree(gradient)[0],dtype=np.float64);value=float(value)
        if not np.isfinite(value) or not np.isfinite(gradient).all():raise RuntimeError('Nonfinite objective/gradient')
        state['evaluations']+=1;cache.update(x=np.array(x,copy=True),value=value,gradient=gradient)
        row=dict(event='objective',evaluation=state['evaluations'],loss=value,gradient_norm=float(np.linalg.norm(gradient)),seconds=time.monotonic()-t,elapsed_seconds=time.monotonic()-start)
        with (output/'objective.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True);return value,gradient
    def callback(intermediate_result):
        state.update(x=np.array(intermediate_result.x,copy=True),iteration=state['iteration']+1,loss=float(intermediate_result.fun))
        row=dict(event='accepted_iteration',iteration=state['iteration'],loss=state['loss'],function_evaluations=state['evaluations'],elapsed_seconds=time.monotonic()-start)
        with (output/'iterations.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        print(json.dumps(row),flush=True)
        if state['iteration']==1 or state['iteration']%args.save_every==0:persist('running')
        if time.monotonic()-start>args.max_seconds:raise TimeBudget()
    initial_probe,_=objective(params,probe)
    if not np.isfinite(float(initial_probe)):raise RuntimeError('Nonfinite initial probe loss')
    metadata['initial_probe_loss']=float(initial_probe);print('INITIAL_PROBE',float(initial_probe),flush=True)
    reason='optimizer_stopped'
    try:
        value,_=fg(x0);state['loss']=value;metadata['initial_fixed_loss']=value;persist('running')
        result=minimize(fg,x0,method='L-BFGS-B',jac=True,callback=callback,options=dict(maxiter=args.maxiter,maxfun=args.maxfun,maxcor=args.history,ftol=1e-12,gtol=1e-7,maxls=30))
        metadata.update(scipy_success=bool(result.success),scipy_message=str(result.message),scipy_status=int(result.status))
        reason='converged' if result.success else 'optimizer_limit_or_failure'
        # Keep the last callback-confirmed accepted point, never an unaccepted trial.
    except TimeBudget:reason='wall_clock_budget'
    except BaseException:
        persist('failed');raise
    persist(reason)
    try:
        final_probe,_=objective(unravel(jnp.asarray(state['x'])),probe)
        if not np.isfinite(float(final_probe)):raise RuntimeError('Nonfinite final probe loss')
        metadata['final_probe_loss']=float(final_probe)
        print('FINAL_PROBE',float(final_probe),flush=True)
    finally:persist(reason)
    return output/'checkpoint.npz'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True);p.add_argument('--pool',required=True);p.add_argument('--output',required=True)
    p.add_argument('--expected-step',type=int,default=82216);p.add_argument('--fixed-geometries',type=int,default=512);p.add_argument('--probe-geometries',type=int,default=128);p.add_argument('--points',type=int,default=128);p.add_argument('--outlet',type=int,default=48);p.add_argument('--seed',type=int,default=2026111521);p.add_argument('--history',type=int,default=20);p.add_argument('--maxiter',type=int,default=500);p.add_argument('--maxfun',type=int,default=1000);p.add_argument('--max-seconds',type=float,default=21600);p.add_argument('--save-every',type=int,default=5);p.add_argument('--require-gpu',action='store_true')
    args=p.parse_args()
    if min(args.fixed_geometries,args.probe_geometries,args.points,args.outlet,args.history,args.maxiter,args.maxfun,args.save_every,args.max_seconds)<=0:p.error('All counts and budgets must be positive')
    train(args)


if __name__=='__main__':main()
