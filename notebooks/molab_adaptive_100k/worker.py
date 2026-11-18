"""Adaptive Adam via paired short LR probes; accepted path has exactly 100k updates."""
import argparse,copy,hashlib,json,os,sys,time,platform,shutil
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);root=ap.parse_args().root.resolve()
    assert json.loads((root/'setup-status.json').read_text())['phase']=='ready'
    assert json.loads((root/'core-check.json').read_text())['passed']
    lock=os.open(root/'launch.lock',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(lock)
    project=root/'frozen/output/project';sys.path.insert(0,str(project))
    import jax,jax.numpy as jnp,numpy as np
    jax.config.update('jax_enable_x64',True);assert jax.default_backend()=='gpu'
    from geometry_operator.checkpoint import load,save
    from geometry_operator.sampling import load_pool,batch
    from geometry_operator.adaptive_core import make_step
    params,config,base,(mean,std),(m,v)=load(root/'base.npz',optimizer=True)
    assert base['step']==287400 and base['optimizer_present'];count=base['step'];start=count;target=start+100000
    pool,manifest=load_pool(project/'data/geometries');assert manifest['splits']['train']['sha256']==base['train_pool_sha256']
    rng=np.random.default_rng();rng.bit_generator.state=base['rng_state'];rate=2e-5
    step,one=make_step(config,mean,std)
    def build_probe(path):
        with np.load(path,allow_pickle=False) as d:data={k:jnp.asarray(d[k],dtype=jnp.float32) for k in d.files}
        def loss(p):
            def f(t,item):return t+one(p,item)[0],None
            val,_=jax.lax.scan(f,jnp.array(0.,dtype=jnp.float32),data)
            return val/data['re'].shape[0]
        return jax.jit(loss)
    control=build_probe(root/'control.npz');audit=build_probe(root/'audit.npz')
    run=root/'run';run.mkdir(exist_ok=True);(root/'snapshots').mkdir(exist_ok=True)
    assert not (run/'training.jsonl').exists()
    begin=time.monotonic();best=float('inf');audit_best=float('inf');total_work=0
    state={'phase':'running','start_step':start,'target_step':target,'initial_lr':rate,'current_lr':rate,'accepted_step':count,'physical_updates':0}
    def write_state():
        state.update(accepted_step=count,current_lr=rate,elapsed_seconds=time.monotonic()-begin,physical_updates=total_work)
        tmp=root/'state.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(root/'status.json')
    def persist(reason='running'):
        md=dict(train_pool_sha256=base['train_pool_sha256'],code_revision='frozen-dynamic-LR-paired-controller',accepted=False,training_labels=0,
            backend=jax.default_backend(),devices=[str(d) for d in jax.devices()],jax=jax.__version__,python=platform.python_version(),
            target_steps=target,global_batch=32,interior_points=128,outlet_points=48,current_lr=rate,elapsed_seconds=time.monotonic()-begin,stop_reason=reason)
        save(run/'checkpoint.npz',params,config,count,rng.bit_generator.state,mean,std,m,v,md);write_state()
    def evaluate():
        nonlocal best,audit_best
        c=float(control(params));a=float(audit(params));assert np.isfinite([c,a]).all()
        rec={'step':count,'offset':count-start,'control_loss':c,'audit_loss':a,'lr':rate,'elapsed_seconds':time.monotonic()-begin}
        with (run/'validation.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
        persist()
        if c<best:
            best=c;shutil.copy2(run/'checkpoint.npz',run/'best_control.npz');(run/'best_control.json').write_text(json.dumps(rec,indent=2))
        if a<audit_best:
            audit_best=a;shutil.copy2(run/'checkpoint.npz',run/'best_audit.npz');(run/'best_audit.json').write_text(json.dumps(rec,indent=2))
        if count>start and (count-start)%10000==0:shutil.copy2(run/'checkpoint.npz',root/f'snapshots/checkpoint_{count}.npz')
        return c,a
    def advance(p,mm,vv,rr,initial,n,lr,path,accepted):
        nonlocal params,m,v,rng,count,total_work
        with path.open('a',buffering=1) as log:
            for k in range(initial+1,initial+n+1):
                data={key:jnp.asarray(x,dtype=jnp.float32) for key,x in batch(rr,pool,32,128,48).items()}
                t=time.monotonic();p,mm,vv,value,terms,gn=step(p,mm,vv,jnp.asarray(k),data,jnp.asarray(lr,dtype=jnp.float32))
                value=float(value);total_work+=1
                if not np.isfinite(value):raise RuntimeError('Nonfinite training loss')
                if k==initial+1 or k%10==0:
                    rec=dict(step=k,loss=value,terms=np.asarray(terms).tolist(),gradient_norm=float(gn),lr=lr,step_seconds=time.monotonic()-t,elapsed_seconds=time.monotonic()-begin)
                    log.write(json.dumps(rec)+'\n')
                if accepted:
                    params,m,v,rng,count=p,mm,vv,rr,k
                    if k%100==0 or k==initial+1:persist()
        return p,mm,vv,rr
    try:
        evaluate();write_state()
        while count<target:
            offset=count-start
            if offset>0 and offset%10000==0:
                state['phase']='comparing_rates';write_state();candidates=sorted(set([max(2e-6,rate/2),rate,min(4e-5,rate*2)]));results=[]
                for lr in candidates:
                    state['candidate_lr']=lr;write_state();rr=np.random.default_rng();rr.bit_generator.state=copy.deepcopy(rng.bit_generator.state)
                    path=root/f'candidate_{count}_{lr:.1e}.jsonl'
                    p,mm,vv,rr=advance(params,m,v,rr,count,500,lr,path,False)
                    score=float(control(p))
                    if np.isfinite(score):results.append((score,lr,p,mm,vv,rr,path))
                current=next(r for r in results if r[1]==rate);winner=min(results,key=lambda r:r[0])
                if winner[0]>=current[0]*.995:winner=current
                before=count;previous=rate;score,rate,params,m,v,rng,path=winner;count+=500
                with (run/'training.jsonl').open('a') as f:f.write(path.read_text())
                record={'anchor_step':before,'chosen_lr':rate,'previous_lr':previous,'chosen_control_loss':score,'candidates':[{'lr':r[1],'control_loss':r[0]} for r in results],
                        'selection_rule':'Switch only if >=0.5% better than same-batch current-rate pilot; audit set is not consulted'}
                with (run/'controller.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                state.pop('candidate_lr',None);state['phase']='running';evaluate()
            next_check=min(target,start+((count-start)//5000+1)*5000)
            params,m,v,rng=advance(params,m,v,rng,count,next_check-count,rate,run/'training.jsonl',True)
            evaluate()
        persist('target_completed');state.update(phase='completed');write_state()
        from geometry_operator.api import FlowOperator
        dev,_=load_pool(project/'data/geometries','dev');reports={}
        for label,file in [('final','checkpoint.npz'),('best_control','best_control.npz'),('best_audit','best_audit.npz')]:
            model=FlowOperator(run/file,allow_unvalidated=True,chunk_size=256);rows=[]
            for i in range(8):
                with np.load(project/f'runs/adam_next50k/development/{i:04d}.npz') as ref:
                    p=model.evaluate(dev['coeff'][i],float(dev['reynolds'][i]),ref['xy']);t=ref['truth'];w=ref['weights']
                    def rel(a,b):
                        ww=w.reshape((-1,)+(1,)*(a.ndim-1));return float(np.sqrt(np.sum(ww*(a-b)**2)/max(np.sum(ww*b*b),1e-16)))
                    rows.append({'case':i,'velocity_l2':rel(p[:,:2],t[:,:2]),'pressure_l2':rel(p[:,2],t[:,2])})
            reports[label]={'step':model.metadata['step'],'cases':rows,'median_velocity_l2':float(np.median([x['velocity_l2'] for x in rows])),'median_pressure_l2':float(np.median([x['pressure_l2'] for x in rows]))}
        (run/'fem_pilot.json').write_text(json.dumps(reports,indent=2))

    except BaseException as exc:
        state.update(phase='failed',error_type=type(exc).__name__);write_state();raise

if __name__=='__main__':main()
