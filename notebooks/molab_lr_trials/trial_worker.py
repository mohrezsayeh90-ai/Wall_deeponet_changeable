"""Paired LR trial: frozen Adam implementation, common state/RNG, fixed dev probe."""
import argparse,hashlib,json,os,sys,time
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--case',required=True);a=ap.parse_args()
    root=a.root.resolve();manifest=json.loads((root/'manifest.json').read_text());spec=next(c for c in manifest['cases'] if c['id']==a.case)
    base=root/'base.npz';assert hashlib.sha256(base.read_bytes()).hexdigest()==manifest['base_sha256']
    project=root/'project';sys.path.insert(0,str(project));os.chdir(project)
    import jax,jax.numpy as jnp,numpy as np
    jax.config.update('jax_enable_x64',True);assert jax.default_backend()=='gpu'
    from geometry_operator.checkpoint import load
    from geometry_operator.sampling import load_pool,batch
    import geometry_operator.train as train
    from geometry_operator.lr_control import learning_rate
    params,config,meta,(mean,std),optimizer=load(base,optimizer=True)
    assert meta['step']==manifest['base_step'] and meta['optimizer_present'] and config.dtype=='float32'
    pool,pm=load_pool(manifest['pool']);assert pm['splits']['train']['sha256']==meta['train_pool_sha256']
    rng=np.random.default_rng();rng.bit_generator.state=meta['rng_state'];first=batch(rng,pool,32,128,48)
    bh=hashlib.sha256(b''.join(np.asarray(first[k]).tobytes() for k in sorted(first))).hexdigest()
    out=root/'cases'/a.case;out.mkdir(parents=True,exist_ok=True);assert not (out/'training.jsonl').exists()
    probe=np.load(root/'fixed_probe.npz',allow_pickle=False);data={k:jnp.asarray(probe[k],dtype=getattr(jnp,config.dtype)) for k in probe.files}
    _,one=train.make_step(config,mean,std)
    def total(p):
        def f(carry,item):
            value,terms=one(p,item);return (carry[0]+value,carry[1]+terms),None
        (val,terms),_=jax.lax.scan(f,(jnp.array(0.,dtype=jnp.float32),jnp.zeros(4,dtype=jnp.float32)),data)
        return val/data['re'].shape[0],terms/data['re'].shape[0]
    evaluate=jax.jit(total);seen=set()
    def log_probe(p,count):
        loss,terms=evaluate(p);loss=float(loss);assert np.isfinite(loss)
        with (out/'probe.jsonl').open('a') as f:f.write(json.dumps({'step':count,'offset':count-meta['step'],'loss':loss,'terms':np.asarray(terms).tolist(),'learning_rate':float(learning_rate(jnp.asarray(count)))})+'\n')
        seen.add(count)
    log_probe(params,meta['step'])
    original_save=train.save
    def save(*args,**kwargs):
        original_save(*args,**kwargs)
        count=args[3]
        if count not in seen and (count-meta['step'])%5000==0:log_probe(args[1],count)
    train.save=save
    (out/'resume.json').write_text(json.dumps({'base_step':meta['step'],'base_sha256':manifest['base_sha256'],'first_batch_sha256':bh,'optimizer_restored':True,'rng_restored':True,'spec':spec},indent=2))
    sys.argv=['train','--pool',manifest['pool'],'--config',manifest['config'],'--resume',str(base),'--output',str(out),
        '--steps',str(meta['step']+15000),'--batch','32','--points','128','--outlet','48','--max-seconds','7200',
        '--save-every','100','--log-every','10','--require-gpu','--code-revision','frozen-plus-explicit-LR-control']
    train.main()
    status=json.loads((out/'status.json').read_text());assert status['step']==meta['step']+15000 and status['stop_reason']=='target_completed'
    from geometry_operator.api import FlowOperator
    def weighted_relative(a,b,w):
        e=np.asarray(a)-b;ww=w.reshape((-1,)+(1,)*(e.ndim-1))
        return float(np.sqrt(np.sum(ww*e**2)/max(np.sum(ww*np.asarray(b)**2),1e-16)))
    dev,_=load_pool(manifest['pool'],'dev');model=FlowOperator(out/'checkpoint.npz',allow_unvalidated=True,chunk_size=256);metrics=[]
    for i in range(8):
        with np.load(Path(manifest['reference_pilot'])/f'{i:04d}.npz',allow_pickle=False) as ref:
            prediction=model.evaluate(dev['coeff'][i],float(dev['reynolds'][i]),ref['xy'])
            metrics.append({'case':i,'velocity_relative_l2':weighted_relative(prediction[:,:2],ref['truth'][:,:2],ref['weights']),
                'pressure_relative_l2':weighted_relative(prediction[:,2],ref['truth'][:,2],ref['weights'])})
    (out/'fem_pilot.json').write_text(json.dumps({'cases':metrics,'median_velocity_l2':float(np.median([m['velocity_relative_l2'] for m in metrics])),
        'median_pressure_l2':float(np.median([m['pressure_relative_l2'] for m in metrics]))},indent=2))


if __name__=='__main__':main()
