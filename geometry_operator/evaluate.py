"""Independent case-level metrics; missing references can never pass release."""
import argparse,hashlib,json,time
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from .api import FlowOperator
from .geometry import forward,local
from .model import fields
from .operators import residual,traction
from .reference import build_case,pressure_sections
from .sampling import load_pool,sha


def weighted_relative(pred,truth,w):
    pred=np.asarray(pred).reshape(len(w),-1);truth=np.asarray(truth).reshape(len(w),-1)
    return float(np.sqrt(np.sum(w[:,None]*(pred-truth)**2)/max(np.sum(w[:,None]*truth**2),1e-16)))


def diagnostics(model,coeff,re,xy,weights):
    # Explicit float64 verification even for a float32-trained checkpoint.
    params=jax.tree_util.tree_map(lambda p:p.astype(jnp.float64),model.params)
    from .model import context
    from .geometry import sensors
    c=jnp.asarray(coeff,dtype=jnp.float64);rr=jnp.array(re,dtype=jnp.float64)
    branch=context(params,jnp.asarray(sensors(coeff,model.config.sensors)),rr,jnp.asarray(model.mean),jnp.asarray(model.std))
    res=jax.jit(jax.vmap(lambda q:residual(params,branch,c,rr,q,model.config)))
    values=[]
    for start in range(0,len(xy),64):
        q=xy[start:start+64];n=len(q);q=np.pad(q,((0,64-n),(0,0)),mode='edge')
        values.append(np.asarray(res(jnp.asarray(q)))[:n])
    values=np.concatenate(values);weights=weights/np.sum(weights)
    e,w=np.polynomial.legendre.leggauss(40);e=(e+1)/2
    tt=np.asarray(jax.jit(jax.vmap(lambda eta:traction(params,branch,c,rr,eta,model.config)))(jnp.asarray(e)))
    pts=[]
    for x in np.linspace(0,5,31):
        for eta in [0.,1.]:pts.append(np.asarray(forward(c,jnp.array([x/5,eta]))))
    wall=model.evaluate(coeff,re,np.array(pts),allow_stress=True)
    inlet_xy=np.column_stack([np.zeros(40),e]);inlet=model.evaluate(coeff,re,inlet_xy,allow_stress=True)
    flux=[]
    for x in np.linspace(0,5,21):
        q=np.asarray(jax.vmap(lambda eta:forward(c,jnp.array([x/5,eta])))(jnp.asarray(e)))
        u=model.evaluate(coeff,re,q,allow_stress=True);gap=float(1+local(c,x)[1]);flux.append(float(w@u[:,0]*gap/2))
    return dict(momentum_rms=float(np.sqrt(np.sum(weights*np.sum(values[:,:2]**2,axis=1)))),
        divergence_rms=float(np.sqrt(np.sum(weights*values[:,2]**2))),
        traction_rms=float(np.sqrt(np.sum(w*np.sum(tt**2,axis=1))/2)),
        wall_max=float(abs(wall[:,:2]).max()),
        inlet_max=float(abs(inlet[:,:2]-np.column_stack([6*e*(1-e),np.zeros(len(e))])).max()),
        flux_max=float(np.max(abs(np.array(flux)-1))))


def plot_case(path,xy,truth,pred):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(3,3,figsize=(14,8),layout='constrained')
    for row,title in enumerate(['u','v','p']):
        vmin=min(truth[:,row].min(),pred[:,row].min());vmax=max(truth[:,row].max(),pred[:,row].max())
        for col,data in enumerate([truth[:,row],pred[:,row],pred[:,row]-truth[:,row]]):
            ax=axes[row,col]
            im=ax.scatter(xy[:,0],xy[:,1],c=data,s=5,cmap='coolwarm' if col==2 else 'viridis',
                vmin=None if col==2 else vmin,vmax=None if col==2 else vmax)
            ax.set_aspect('equal');ax.set_title(f'{title}: '+['FEM','operator','difference'][col]);fig.colorbar(im,ax=ax,shrink=.7)
    fig.savefig(path,dpi=160);plt.close(fig)


def evaluate(args):
    model=FlowOperator(args.checkpoint,allow_unvalidated=True)
    pool,manifest=load_pool(args.pool,args.split);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    config=json.loads(Path('configs/production.json').read_text());records=[]
    indices=range(min(len(pool['coeff']),args.limit or len(pool['coeff'])))
    if args.split=='test':
        if not args.unlock_test:raise ValueError('Locked test requires explicit --unlock-test after checkpoint selection')
        marker=Path(args.pool)/'test_opened.json'
        if marker.exists() and json.loads(marker.read_text())['checkpoint_sha256']!=model.sha256:
            raise ValueError('Test already opened for different weights; final claims require a new untouched split')
        marker.write_text(json.dumps(dict(checkpoint_sha256=model.sha256,split_sha256=manifest['splits']['test']['sha256'])))
    for i in indices:
        coeff=pool['coeff'][i];re=float(pool['reynolds'][i]);case=manifest['splits'][args.split]['cases'][i]
        ref=Path(args.references)/args.split/f'{i:04d}.npz'
        try:
            if args.build_references and not ref.exists():build_case(coeff,re,ref,nx=40,ny=12)
            if not ref.exists():raise FileNotFoundError('Independent mesh-converged reference missing')
            with np.load(ref,allow_pickle=False) as d:
                xy,w,truth=d['xy'],d['weights'],d['fields'];rm=json.loads(str(d['metadata']))
            t=time.monotonic();pred=model.evaluate(coeff,re,xy,allow_stress=args.split=='stress');first=time.monotonic()-t
            t=time.monotonic();model.evaluate(coeff,re,xy,allow_stress=args.split=='stress');steady=time.monotonic()-t
            drop=pressure_sections(lambda q:model.evaluate(coeff,re,q,allow_stress=args.split=='stress'),coeff)
            metrics=dict(velocity_relative_l2=weighted_relative(pred[:,:2],truth[:,:2],w),
                pressure_relative_l2=weighted_relative(pred[:,2],truth[:,2],w),
                v_absolute_rms=float(np.sqrt(np.sum(w*(pred[:,1]-truth[:,1])**2)/sum(w))),
                pressure_drop_relative=abs(drop-rm['pressure_drop'])/max(abs(rm['pressure_drop']),1e-8),
                pressure_drop_predicted=drop,pressure_drop_reference=rm['pressure_drop'],
                first_call_seconds=first,steady_seconds=steady,**diagnostics(model,coeff,re,xy,w))
            records.append(dict(**case,status='evaluated',metrics=metrics,reference_sha256=sha(ref),mesh_uncertainty=rm['mesh_uncertainty']))
            np.savez_compressed(output/f'{i:04d}.npz',xy=xy,weights=w,truth=truth,predicted=pred)
            if i<8:plot_case(output/f'{i:04d}.png',xy,truth,pred)
        except Exception as e:records.append(dict(**case,status='failed',error=str(e)))
        print(json.dumps(records[-1]),flush=True)
        (output/'cases.json').write_text(json.dumps(records,indent=2))
    valid=[r for r in records if r['status']=='evaluated'];summary={};gates={}
    thresholds=config['release']
    if valid:
        for name in valid[0]['metrics']:
            vv=[r['metrics'][name] for r in valid]
            summary[name]=dict(median=float(np.median(vv)),p95=float(np.percentile(vv,95)),max=float(max(vv)))
        mapping={'velocity_relative_l2':('velocity_median','velocity_p95'),
                 'pressure_relative_l2':('pressure_median','pressure_p95'),
                 'pressure_drop_relative':('pressure_drop_median','pressure_drop_p95')}
        for metric,(med,p95) in mapping.items():
            gates[med]=summary[metric]['median']<=thresholds[med];gates[p95]=summary[metric]['p95']<=thresholds[p95]
        gates['family_medians']=all(np.median([r['metrics']['velocity_relative_l2'] for r in valid if r['family']==family])<=thresholds['family_velocity_median'] for family in {r['family'] for r in valid})
        for metric,key,stat in [('flux_max','flux_max','max'),('wall_max','boundary_max','max'),('inlet_max','boundary_max','max'),
                                ('divergence_rms','divergence_rms','max'),('momentum_rms','momentum_p95','p95'),('traction_rms','traction_p95','p95')]:
            gates[metric]=summary[metric][stat]<=thresholds[key]
    gates['complete_split']=len(valid)==len(pool['coeff']) and len(records)==len(pool['coeff'])
    gates['locked_final_test']=args.split=='test' and args.unlock_test
    # Required baseline/ablation, stress, Re sweep and wall-stress evidence are
    # separate release prerequisites, not inferred from this single split.
    gates['release_comparisons_complete']=False
    report=dict(checkpoint_sha256=model.sha256,step=model.metadata['step'],split=args.split,
        split_sha256=manifest['splits'][args.split]['sha256'],expected_cases=len(pool['coeff']),evaluated=len(valid),
        failed=len(records)-len(valid),summary=summary,gates=gates,all_gates_passed=all(gates.values()),
        timing='frozen branch and batched trunk; no optimization or solver in prediction',
        limitations=['Wall stress and required matched baselines/ablations not yet completed.'])
    (output/'report.json').write_text(json.dumps(report,indent=2))
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--pool',default='data/geometries');p.add_argument('--references',default='data/references')
    p.add_argument('--split',choices=['dev','test','stress'],default='dev');p.add_argument('--output',default='runs/evaluation');p.add_argument('--limit',type=int)
    p.add_argument('--unlock-test',action='store_true');p.add_argument('--build-references',action='store_true')
    evaluate(p.parse_args())
if __name__=='__main__':main()
