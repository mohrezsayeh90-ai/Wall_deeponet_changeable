"""Design channel walls for inertial particle targeting using a frozen DeepONet."""
from pathlib import Path
import argparse
import json
import shutil
import time

import jax
import jax.numpy as jnp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import LinearConstraint, brentq, minimize
from scipy.interpolate import PPoly

from geometry_operator.checkpoint import load
from geometry_operator.geometry import BREAKS, _canonical, sample_numpy, sensors, validate
from geometry_operator.model import context, fields

RE = 30.0
TARGETS = np.arange(1, 6) / 6.0
TAUS = np.array([0.75, 0.9, 1.8])
SEED = 1729


def design_basis():
    knots = np.r_[np.zeros(6), BREAKS[1:-1], np.full(6, 5.)]
    x = np.array([np.mean(knots[i+1:i+6]) for i in range(len(knots)-6)])
    shapes = []
    for center, radius in [(1.25,.8),(1.85,1.25),(2.55,1.45),(3.15,1.2),(3.85,1.1),(4.1,.75)]:
        z = (x-center)/radius
        shapes.append(np.exp(1-1/np.maximum(1-z*z,1e-12))*(abs(z)<1))
    zero = np.zeros_like(x)
    basis = np.stack([_canonical([b,zero]) for b in shapes] +
                     [_canonical([zero,b]) for b in shapes[::2]+shapes[-1:]])
    grid = np.linspace(0,5,1001)
    c,g = np.array([sample_numpy(b, grid) for b in basis]).transpose(1,2,0)
    cp,gp = np.array([sample_numpy(b, grid,1) for b in basis]).transpose(1,2,0)
    matrix = np.vstack((c,g,cp-gp/2,cp+gp/2))
    lo = np.repeat([-.348,-.398,-.697,-.697],len(grid))
    hi = np.repeat([.348,.498,.697,.697],len(grid))
    return basis, matrix, lo, hi


def predictor(checkpoint, basis):
    params,cfg,meta,(mean,std),_ = load(checkpoint)
    dtype = getattr(jnp,cfg.dtype)
    mean,std = jnp.asarray(mean,dtype),jnp.asarray(std,dtype)
    coeff_basis = jnp.asarray(basis,dtype)
    sensor_basis = jnp.asarray(np.stack([sensors(b,cfg.sensors) for b in basis]),dtype)
    re = jnp.asarray(RE,dtype)

    def path(weights, tau, dt, n):
        coeff = jnp.einsum('i,ijkl->jkl',weights,coeff_basis)
        branch = context(params,weights@sensor_basis,re,mean,std)
        def flow(z):
            return fields(params,branch,coeff,re,z[:2],cfg)[:2]
        def advance(carry, _):
            state, ended = carry
            def step(z):
                u = flow(z)
                mid = jnp.concatenate([z[:2]+dt/2*z[2:],u+(z[2:]-u)*jnp.exp(-dt/(2*tau))])
                um = flow(mid)
                vel = um+(z[2:]-um)*jnp.exp(-dt/tau)
                return jnp.concatenate([z[:2]+dt*mid[2:],vel])
            trial = jax.lax.cond(ended,lambda z:z,step,state)
            return (trial,ended|(trial[0]>=5)),trial
        z0=jnp.array([0.,.5,0.,0.],dtype=dtype)
        _, history=jax.lax.scan(advance,(z0,False),None,length=n)
        return jnp.concatenate([z0[None,:],history],axis=0)

    @jax.jit
    def endpoints(weights, taus):
        def one(w,tau):
            h = path(w,tau,.04,260)
            k = jnp.argmax(h[:,0]>=5)
            lo,hi=h[jnp.maximum(k-1,0)],h[k]
            dy=(5-lo[0])/jnp.maximum(hi[0]-lo[0],1e-10)
            return jnp.where(k>0,lo[1]+dy*(hi[1]-lo[1]),jnp.nan)
        return jax.vmap(one)(weights,taus)
    @jax.jit
    def velocity(coeff, branch, xy):
        return fields(params, branch, coeff, re, xy, cfg)[:2]

    def flow_for(coeff):
        jc = jnp.asarray(coeff, dtype)
        branch = context(params, jnp.asarray(sensors(coeff,cfg.sensors), dtype), re, mean, std)
        return lambda xy: np.asarray(velocity(jc, branch, jnp.asarray(xy,dtype)),float)
    return endpoints, flow_for


def screen(args):
    basis,A,lo,hi=design_basis()
    endpoints,_=predictor(args.checkpoint,basis)
    rng=np.random.default_rng(SEED)
    x=rng.normal(size=(args.samples,len(basis)))
    x[0]=0
    values=x@A.T
    limit=np.maximum((values/hi).max(axis=1),(values/lo).max(axis=1))
    x*=np.divide(rng.uniform(.35,1,size=len(x)),np.maximum(limit,1e-12))[:,None]
    for tau in TAUS if args.particle is None else TAUS[args.particle-1:args.particle]:
        y=np.concatenate([np.asarray(endpoints(w.astype('float32'),np.full(len(w),tau,'float32')))
                          for w in np.array_split(x,max(1,(len(x)+63)//64))])
        print('SCREEN',tau, float(np.nanmin(y)),float(np.nanmax(y)),flush=True)
        np.savez(args.output/f'screen_{tau}.npz',weights=x,y=y)


def integrate(coeff, flow, tau, *, max_step=.04, rtol=2e-7, contains=None):
    c,g=[PPoly(a,BREAKS) for a in coeff]
    def bounds(x):
        return c(x)-g(x)/2,1+c(x)+g(x)/2
    def rhs(t,z):
        # Adaptive RK stages can overshoot terminal boundaries before event localization.
        x=float(np.clip(z[0],0,5))
        b,top=bounds(x)
        y=float(np.clip(z[1],b+1e-10,top-1e-10))
        xy=np.array([x,y])
        if contains is not None and not contains(xy[None,:])[0]:
            raise ValueError('Particle integration reached the FEM wall approximation')
        return np.r_[z[2:],(flow(xy)-z[2:])/tau]
    def outlet(t,z):return z[0]-5.
    def bottom(t,z):return z[1]-bounds(z[0])[0]
    def top(t,z):return bounds(z[0])[1]-z[1]
    outlet.terminal=True;outlet.direction=1
    for event in [bottom,top]:event.terminal=True;event.direction=-1
    result=solve_ivp(rhs,(0,30),[0,.5,0,0],events=[outlet,bottom,top],
                     rtol=rtol,atol=rtol*.05,max_step=max_step)
    state=result.y.T
    clearance=np.min(np.minimum(state[:,1]-bounds(state[:,0])[0],bounds(state[:,0])[1]-state[:,1]))
    exit_y=float(result.y_events[0][0,1]) if len(result.t_events[0]) else None
    return dict(exit_y=exit_y, time=float(result.t[-1]),clearance=float(clearance),
                state=state,t=result.t,status='exited' if exit_y is not None else 'no_outlet')


def optimize(args):
    basis,A,lo,hi=design_basis()
    endpoints,flow_for=predictor(args.checkpoint,basis)
    constraints=LinearConstraint(A[::4],lo[::4],hi[::4])
    for tau in TAUS if args.particle is None else TAUS[args.particle-1:args.particle]:
        with np.load(args.output/f'screen_{tau}.npz') as z:
            weights=z['weights']; observed=z['y']

        def batch_y(w):
            return np.asarray(endpoints(np.asarray(w,dtype='float32'),np.full(len(w),tau,'float32')),float)

        def extreme(sign):
            order=np.argsort(np.nan_to_num(sign*observed,nan=np.inf))
            best=(float(sign*observed[order[0]]),weights[order[0]])
            for i in order[:args.starts]:
                def objective(w):
                    return float(np.nan_to_num(sign*batch_y(w[None,:])[0],nan=10.))
                def jac(w):
                    h=.0003
                    offsets=np.eye(len(w))*h
                    y=batch_y(np.vstack((w+offsets,w-offsets)))
                    return np.nan_to_num(sign*(y[:len(w)]-y[len(w):])/(2*h))
                r=minimize(objective,weights[i],jac=jac,constraints=[constraints],method='SLSQP',
                           options={'maxiter':args.max_iterations,'ftol':args.optimization_tolerance})
                w=r.x
                ext=A@w
                w=w/max(1.,np.max(ext/hi),np.max(ext/lo))
                if objective(w)<best[0]:
                    try:
                        validate(np.einsum('i,ijkl->jkl',w,basis))
                        best=objective(w),w
                    except ValueError:
                        pass
            return best[1]

        low,high=extreme(1),extreme(-1)
        print('EXTREMES',tau,batch_y(np.stack((low,high))).tolist(),flush=True)
        for target_i,target in enumerate(TARGETS,1):
            direction=low if target<.5 else high
            if target==.5:direction=np.zeros_like(direction)
            def evaluated(alpha,**kwargs):
                coeff=np.einsum('i,ijkl->jkl',alpha*direction,basis)
                return coeff,integrate(coeff,flow_for(coeff),tau,**kwargs)
            _,a=evaluated(0);_,b=evaluated(1)
            if (a['exit_y'] is not None and b['exit_y'] is not None and
                    (a['exit_y']-target)*(b['exit_y']-target)<0):
                alpha=brentq(lambda x:evaluated(x)[1]['exit_y']-target,0,1,xtol=1e-5)
            else:
                alpha=min([0,1],key=lambda x: abs((a if x==0 else b)['exit_y']-target)
                          if (a if x==0 else b)['exit_y'] is not None else np.inf)
            coeff,pred=evaluated(alpha, max_step=.025,rtol=3e-8)
            validate(coeff)
            name=f'P{list(TAUS).index(tau)+1}_T{target_i}'
            tmp=args.output/f'{name}.partial.npz'
            np.savez_compressed(tmp,coeff=coeff,tau=tau,target=target,
                                reynolds=RE,model_state=pred['state'],model_time=pred['t'],
                                model_status=pred['status'])
            tmp.replace(args.output/f'{name}.npz')
            print('DESIGN',name, target,pred['exit_y'],pred['clearance'],flush=True)


def retarget_first(args):
    """Use the lower-steering pilot shape and its reflection for particle 1."""
    basis,_,_,_=design_basis()
    _,flow_for=predictor(args.checkpoint,basis)
    with np.load(args.pilot_shape,allow_pickle=False) as z:
        template=z['coeff'].copy()
    tau=TAUS[0]
    for i,target in enumerate(TARGETS,1):
        direction=template.copy()
        if target>.5:direction[0]*=-1.
        if target==.5:direction[:]=0.
        def evaluate(alpha):
            coeff=alpha*direction
            return integrate(coeff,flow_for(coeff),tau,max_step=.025,rtol=3e-8)
        left,right=(evaluate(a)['exit_y'] for a in (0,1))
        assert left is not None and right is not None
        if (left-target)*(right-target)<0:
            alpha=brentq(lambda a:evaluate(a)['exit_y']-target,0,1,xtol=2e-6)
        else:
            alpha=0 if abs(left-target)<abs(right-target) else 1
        res=evaluate(alpha)
        coeff=alpha*direction
        validate(coeff)
        np.savez_compressed(args.output/f'P1_T{i}.npz',coeff=coeff,tau=tau,target=target,
                            reynolds=RE,model_state=res['state'],model_time=res['t'],
                            model_status=res['status'],template=direction,scale=alpha)
        print('RETARGET',i,tau,target,res['exit_y'],flush=True)


def verify(args):
    from geometry_operator.reference import solve_reference

    # Run both meshes for each distinct geometry; reuse solutions across particles.
    prefix='P[123]' if args.particle is None else f'P{args.particle}'
    paths=sorted(args.output.glob(prefix+'_T[12345].npz'))
    if args.wait:
        expected=15 if args.particle is None else 5
        deadline=time.monotonic()+900
        while len(paths)<expected and time.monotonic()<deadline:
            time.sleep(3)
            paths=sorted(args.output.glob(prefix+'_T[12345].npz'))
        if len(paths)<expected:
            raise RuntimeError(f'Only {len(paths)} of {expected} designs became available')
    for path in paths:
        with np.load(path) as z:
            design={k:z[k] for k in z.files}
        tau=float(design['tau']);coeff=design['coeff']
        result_path=path.with_suffix('.json')
        if result_path.exists():
            continue
        model_exit=float(design['model_state'][-1,1]) if np.isclose(design['model_state'][-1,0],5.) else None
        record=dict(case=path.stem,tau=tau,target=float(design['target']),
                    model_exit=model_exit,geometry=validate(coeff),fem={})
        model=design['model_state']
        center,gap=sample_numpy(coeff,model[:,0])
        record['model_clearance']=float(np.minimum(model[:,1]-center+gap/2,
                                                    1+center+gap/2-model[:,1]).min())
        for label,nx,ny in [('coarse',48,16),('fine',72,24)]:
            try:
                flow,meta=solve_reference(coeff,RE,nx,ny)
                pred=integrate(coeff,lambda p:flow(p[None,:])[0,:2],tau,contains=flow.contains)
                record['fem'][label]={k:v for k,v in pred.items() if k not in ['state','t']}
                record['fem'][label]['solver']=meta
                design[label+'_state']=pred['state'];design[label+'_time']=pred['t']
            except (RuntimeError,ValueError) as e:
                record['fem'][label]={'status':'unresolved','detail':str(e),'exit_y':None}
        c,f=[record['fem'][k]['exit_y'] for k in ['coarse','fine']]
        record['mesh_exit_difference']=abs(c-f) if c is not None and f is not None else None
        np.savez_compressed(path,**design)
        result_path.write_text(json.dumps(record,indent=2)+'\n')
        print('FEM',path.stem,record['target'],record['model_exit'],c,f,flush=True)


def refine(args):
    """Adjust missed targets using coarse FEM, then independently check the fine mesh."""
    from geometry_operator.reference import solve_reference
    from geometry_operator.geometry import descriptors

    basis,_,_,_=design_basis()
    _,flow_for=predictor(args.checkpoint,basis)
    args.output.mkdir(exist_ok=True)
    initial=args.initial
    for i in ([args.particle] if args.particle else [1,2,3]):
        for j in range(1,6):
            name=f'P{i}_T{j}'
            deadline=time.monotonic()+600
            while not (initial/f'{name}.json').exists() and time.monotonic()<deadline:
                time.sleep(3)
            first=json.loads((initial/f'{name}.json').read_text())
            with np.load(initial/f'{name}.npz',allow_pickle=False) as z:
                data={k:z[k] for k in z.files}
            coeff,tau,target=data['coeff'],float(data['tau']),float(data['target'])
            coarse,fine=[first['fem'][label]['exit_y'] for label in ['coarse','fine']]
            mesh=first['mesh_exit_difference']
            if fine is not None and mesh is not None and abs(fine-target)+mesh<=.01:
                shutil.copy2(initial/f'{name}.npz',args.output/f'{name}.npz')
                record=dict(first,design_method='DeepONet; checked by two FEM meshes')
            elif coarse is None or fine is None:
                shutil.copy2(initial/f'{name}.npz',args.output/f'{name}.npz')
                record=dict(first,design_method='DeepONet; FEM trajectory unresolved')
            else:
                d=descriptors(coeff)
                cap=min(.349/max(d['center_deviation'],1e-12),
                        .399/max(1-d['min_gap'],1e-12),.499/max(d['max_gap']-1,1e-12),
                        .699/max(d['max_slope'],1e-12),2.)
                samples=[(0.,.5),(1.,coarse)]
                best=None
                def solve(alpha,nx,ny):
                    geometry=alpha*coeff
                    validate(geometry)
                    flow,meta=solve_reference(geometry,RE,nx,ny)
                    res=integrate(geometry,lambda p:flow(p[None,:])[0,:2],tau,contains=flow.contains)
                    res['solver']=meta
                    return res
                for _ in range(5):
                    low,high=sorted(samples,key=lambda v:abs(v[1]-target))[:2]
                    if abs(high[1]-low[1])<1e-10:break
                    alpha=float(np.clip(low[0]+(target-low[1])*(high[0]-low[0])/(high[1]-low[1]),.01,cap))
                    if any(abs(alpha-old)<1e-5 for old,_ in samples):break
                    try:
                        trial=solve(alpha,48,16)
                    except (RuntimeError,ValueError):
                        break
                    if trial['exit_y'] is None:break
                    y=trial['exit_y'];samples.append((alpha,y))
                    if best is None or abs(y-target)<abs(best[1]['exit_y']-target):best=alpha,trial
                    if abs(y-target)<.0015:break
                record=dict(first,design_method='DeepONet; original design retained',refinement_trials=samples)
                if best is not None:
                    alpha,co=best
                    try:
                        fi=solve(alpha,72,24)
                    except (RuntimeError,ValueError):
                        fi=None
                    if fi is not None and fi['exit_y'] is not None and abs(fi['exit_y']-target)<abs(fine-target):
                        new_coeff=coeff*alpha
                        model=integrate(new_coeff,flow_for(new_coeff),tau,max_step=.025,rtol=3e-8)
                        data.update(coeff=new_coeff,model_state=model['state'],model_time=model['t'],
                                    model_status=model['status'],coarse_state=co['state'],
                                    coarse_time=co['t'],fine_state=fi['state'],fine_time=fi['t'])
                        record.update(model_exit=model['exit_y'],geometry=validate(new_coeff),
                                      model_clearance=model['clearance'],refinement_scale=alpha,
                                      design_method='DeepONet initial design; amplitude tuned on coarse FEM',
                                      fem={k:{n:v for n,v in z.items() if n not in ['state','t']}
                                           for k,z in [('coarse',co),('fine',fi)]},
                                      mesh_exit_difference=abs(co['exit_y']-fi['exit_y']))
                np.savez_compressed(args.output/f'{name}.npz',**data)
            record['initial_deeponet_exit']=first['model_exit']
            record['initial_fem_fine_exit']=fine
            (args.output/f'{name}.json').write_text(json.dumps(record,indent=2)+'\n')
            print('FINAL',name,target,record['fem']['fine']['exit_y'],record['design_method'],flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--samples',type=int,default=512)
    parser.add_argument('--particle',type=int,choices=[1,2,3])
    parser.add_argument('--wait',action='store_true')
    parser.add_argument('--stage',choices=['screen','optimize','retarget_first','verify','refine'],default='screen')
    parser.add_argument('--pilot-shape',type=Path)
    parser.add_argument('--initial',type=Path)
    parser.add_argument('--first-tau',type=float,default=.75)
    parser.add_argument('--starts',type=int,default=2)
    parser.add_argument('--max-iterations',type=int,default=30)
    parser.add_argument('--optimization-tolerance',type=float,default=5e-6)
    args=parser.parse_args()
    TAUS[0]=args.first_tau
    args.output.mkdir(parents=True,exist_ok=True)
    {'screen':screen,'optimize':optimize,'retarget_first':retarget_first,'verify':verify,'refine':refine}[args.stage](args)


if __name__=='__main__':
    main()
