"""Matched saved-FEM evaluation, final/best checkpoints and two result chains."""
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry_operator.api import FlowOperator
from geometry_operator.evaluate import weighted_relative
from scripts.channel_chain import render
ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/'artifacts/molab/adam-100k-v1';OUT=ROOT/'reports/molab_adam_100k/completed'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    import geometry_operator.reference as ref
    def forbidden(*a,**k):raise RuntimeError('Saved FEM only; no re-solving permitted')
    ref.solve_reference=forbidden;ref.build_case=forbidden
    OUT.mkdir(parents=True,exist_ok=True)
    models={name:FlowOperator(ART/file,allow_unvalidated=True) for name,file in [('base','base_checkpoint.npz'),('final','checkpoint.npz'),('best','best_checkpoint.npz')]}
    assert [models[n].metadata['step'] for n in ['base','final','best']]==[187400,287400,282400]
    selection=json.loads((ROOT/'reports/adam_next50k_comparison/selection.json').read_text());records=[];chains=[];gallery=OUT/'selected';gallery.mkdir(exist_ok=True)
    for c in selection['population']:
        source=ROOT/'reports'/c['group']/c['directory'];target=OUT/'cases'/f"{c['case_id']:02d}_{c['directory']}";target.mkdir(parents=True,exist_ok=True)
        with np.load(source/'fields.npz') as d:xy,w,truth,coeff=d['xy'],d['weights'],d['fem'],d['coeff']
        predictions={n:m.evaluate(coeff,30.,xy) for n,m in models.items()};metrics={}
        for n,p in predictions.items():
            e=p-truth
            metrics[n]={'velocity_relative_l2':weighted_relative(p[:,:2],truth[:,:2],w),'pressure_relative_l2':weighted_relative(p[:,2],truth[:,2],w),
                        'u_relative_l2':weighted_relative(p[:,0],truth[:,0],w),'v_absolute_rms':float(np.sqrt(w@e[:,1]**2/w.sum())),
                        'velocity_mae':float(w@np.linalg.norm(e[:,:2],axis=1)/w.sum()),'pressure_mae':float(w@abs(e[:,2])/w.sum())}
        np.savez_compressed(target/'fields.npz',xy=xy,weights=w,fem=truth,coeff=coeff,**predictions)
        shutil.copy2(source/'geometry.npz',target/'geometry.npz');shutil.copy2(source/'reference.npz',target/'reference.npz')
        record={**c,'directory':str(target.relative_to(OUT)),'metrics':metrics,'reference_sha256':sha(source/'reference.npz')};records.append(record)
        (target/'metrics.json').write_text(json.dumps(record,indent=2)+'\n')
        if c['case_id'] in [12,16]:
            manifest=json.loads((source.parent/'manifest.json').read_text());info=next(i for i in manifest['cases'] if i['index']==c['original_index'])
            gd=gallery/target.name;gd.mkdir(exist_ok=True)
            for name in ['geometry.npz','reference.npz']:shutil.copy2(source/name,gd/name)
            chains.append(dict(info,index=c['case_id'],directory=target.name))
        print('Evaluated saved case',c['case_id'],flush=True)
    # Independent same eight saved FEM pilot cases, Re=10, all three checkpoints.
    pilot=[]
    with np.load(ROOT/'runs/adam-next50k-v1/output/project/data/geometries/dev.npz') as dev:
        for i in range(8):
            with np.load(ROOT/f'artifacts/kaggle/adam-next50k-v1/development/{i:04d}.npz') as f:
                xy,w,t=f['xy'],f['weights'],f['truth']
            mm={}
            for n,m in models.items():
                p=m.evaluate(dev['coeff'][i],float(dev['reynolds'][i]),xy)
                mm[n]={'velocity_relative_l2':weighted_relative(p[:,:2],t[:,:2],w),'pressure_relative_l2':weighted_relative(p[:,2],t[:,2],w)}
            pilot.append({'case':i,'metrics':mm})
    med=lambda rs,n,k:float(np.median([r['metrics'][n][k] for r in rs]))
    aggregates={name:{k:med(records,name,k) for k in ['velocity_relative_l2','pressure_relative_l2']} for name in models}
    pilot_med={name:{k:med(pilot,name,k) for k in ['velocity_relative_l2','pressure_relative_l2']} for name in models}
    gpu=json.loads((ART/'fem_pilot.json').read_text())
    np.testing.assert_allclose(pilot_med['final']['velocity_relative_l2'],gpu['median_velocity_l2'],rtol=1e-6,atol=1e-9)
    np.testing.assert_allclose(pilot_med['final']['pressure_relative_l2'],gpu['median_pressure_l2'],rtol=1e-6,atol=1e-9)
    logs=[json.loads(l) for l in (ART/'training.jsonl').read_text().splitlines()];probes=[json.loads(l) for l in (ART/'probe.jsonl').read_text().splitlines()]
    rolling=np.array([r['loss'] for r in logs]);xs=np.array([r['step'] for r in logs]);averages=np.array([rolling[max(0,i-99):i+1].mean() for i in range(len(rolling))])
    fig,axes=plt.subplots(2,1,figsize=(12,8),layout='constrained');axes[0].plot(xs,rolling,alpha=.2,lw=.5,label='Logged loss');axes[0].plot(xs,averages,label='Mean over up to 100 records',color='#7c3aed')
    axes[1].plot([p['step'] for p in probes],[p['loss'] for p in probes],'o-',color='#0f766e');axes[1].axvline(282400,color='black',ls=':',label='Best fixed-probe checkpoint')
    for ax in axes:ax.grid(alpha=.2);ax.set(yscale='log',xlabel='Global Adam step');ax.legend()
    axes[0].set(title='Completed 100,000-update training',ylabel='Stochastic training loss');axes[1].set(title='Unchanged development probe',ylabel='Fixed-probe objective')
    fig.savefig(OUT/'training_and_probe.png',dpi=180);fig.savefig(OUT/'training_and_probe.pdf');plt.close(fig)
    summary={'final_step':287400,'best_probe_step':282400,'start_step':187400,'updates':100000,'accepted_model':False,'fem_rerun':False,
        'checkpoints':{n:{'step':m.metadata['step'],'sha256':m.sha256} for n,m in models.items()},'all_20_medians':aggregates,'pilot_8_medians':pilot_med,
        'fixed_probe_initial':probes[0]['loss'],'fixed_probe_final':probes[-1]['loss'],'fixed_probe_best':min(p['loss'] for p in probes),
        'training_mean_first_1000_records':float(rolling[:1000].mean()),'training_mean_last_1000_records':float(rolling[-1000:].mean()),'cases':records,'pilot':pilot}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (OUT/'comparison.csv').open('w',newline='') as f:
        rows=[{'case':r['case_id'],'title':r['title'],'checkpoint':n,'step':models[n].metadata['step'],**r['metrics'][n]} for r in records for n in models]
        wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
    manifest={'checkpoint_step':282400,'checkpoint_sha256':models['best'].sha256,'checkpoint':str(models['best'].path.relative_to(ROOT)),'cases':chains,'case_count':len(chains),'fem_rerun':False}
    (gallery/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def callback(output,manifest,rr,plt,colors):(output/'summary.json').write_text(json.dumps({'cases':rr,'checkpoint_step':282400},indent=2)+'\n')
    render(gallery,manifest,models['best'].path,summary_function=callback,pdf_name='two_result_chains.pdf')
    text='''# Completed molab 100k continuation: final and best checkpoints

**187,400 → 287,400: exactly 100,000 updates completed.** Best fixed-probe checkpoint: **282,400**. Both full-state checkpoints and all periodic snapshots are preserved. Selection by the development probe does not imply universal FEM superiority or full benchmark acceptance.

[Two complete result chains PDF](selected/two_result_chains.pdf) · [Comparison CSV](comparison.csv) · [Full metrics](summary.json) · [Download outputs ZIP](results_and_examples.zip)

![Training and probe](training_and_probe.png)

| Re=30, median over the same 20 saved cases | Start 187,400 | Final 287,400 | Best probe 282,400 |
| --- | ---: | ---: | ---: |
'''
    for key,label in [('velocity_relative_l2','Velocity L2'),('pressure_relative_l2','Pressure L2')]:text+=f"| {label} | {aggregates['base'][key]:.3%} | {aggregates['final'][key]:.3%} | {aggregates['best'][key]:.3%} |\n"
    text+='\nAll geometry, FEM fields, points and physical weights are reused exactly. Pressure has no fitted shift. These 20 illustrative Re=30 cases and the separate eight-case Re=10 pilot are not the full validation benchmark. All regressions are retained. Final CPU FEM metrics match the saved GPU pilot within tolerance.\n\n'
    for c in chains:text+=f"## {c['title']}\n\n![New best-probe result chain](selected/{c['directory']}/chain.png)\n\n"
    (OUT/'README.md').write_text(text)
    with zipfile.ZipFile(OUT/'results_and_examples.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix!='.zip':z.write(p,arcname=p.relative_to(OUT.parent))
    with zipfile.ZipFile(OUT/'results_and_examples.zip') as z:
        assert z.testzip() is None
        for n in z.namelist():assert z.read(n)==(OUT.parent/n).read_bytes()
    print(json.dumps({k:v for k,v in summary.items() if k not in ['cases','pilot']},indent=2))
if __name__=='__main__':main()
