"""New checkpoint predictions on all 25 saved geometries; never rerun FEM."""
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
from scripts.channel_chain import render
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/checkpoint_82216'
GROUPS=[('channel_chain_15',15,False),('random_channel_chain_5',5,False),('extreme_channel_chain_5',5,True)]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def metric_pair(pred,truth,w,vector=False):
    error=pred-truth
    e=np.linalg.norm(error,axis=1) if vector else abs(error)
    f=np.linalg.norm(truth,axis=1) if vector else abs(truth)
    return dict(mae=float(w@e/w.sum()),relative_l1=float(w@e/max(w@f,1e-16)),reference_mean_magnitude=float(w@f/w.sum()),relative_l1_denominator_floored=bool(w@f<1e-16))


def summarize_group(source,output,manifest,records,plt):
    rows=[];pairs=[]
    for r in records:
        old_dir=source/r['directory'];new_dir=output/r['directory']
        assert sha(old_dir/'reference.npz')==sha(new_dir/'reference.npz')
        assert sha(old_dir/'geometry.npz')==sha(new_dir/'geometry.npz')
        with np.load(old_dir/'fields.npz',allow_pickle=False) as old,np.load(new_dir/'fields.npz',allow_pickle=False) as new:
            for name in ['xy','weights','fem','coeff']:np.testing.assert_array_equal(old[name],new[name])
            assert str(new['checkpoint_sha256'])==manifest['checkpoint_sha256']
            truth=new['fem'];pred=new['predicted'];prior=old['predicted'];w=new['weights'];assert np.isfinite(pred).all() and np.all(w>0)
            err=pred-truth
            actual=np.sqrt((w@np.sum(err[:,:2]**2,axis=1))/(w@np.sum(truth[:,:2]**2,axis=1)))
            np.testing.assert_allclose(actual,r['metrics']['velocity_relative_l2'],rtol=1e-12)
            mean={}
            for quantity,a,b,f,vector in [('velocity_vector',prior[:,:2],pred[:,:2],truth[:,:2],True),('u',prior[:,0],pred[:,0],truth[:,0],False),('v',prior[:,1],pred[:,1],truth[:,1],False),('pressure',prior[:,2],pred[:,2],truth[:,2],False),('speed',np.linalg.norm(prior[:,:2],axis=1),np.linalg.norm(pred[:,:2],axis=1),np.linalg.norm(truth[:,:2],axis=1),False)]:
                before=metric_pair(a,f,w,vector);after=metric_pair(b,f,w,vector);mean[quantity]=dict(old=before,new=after)
                rows.append(dict(case=r['index'],title=r['title'],quantity=quantity,old_mae=before['mae'],new_mae=after['mae'],old_relative_l1_percent=100*before['relative_l1'],new_relative_l1_percent=100*after['relative_l1'],relative_denominator_floored=after['relative_l1_denominator_floored']))
        prior_metrics=json.loads((old_dir/'metrics.json').read_text())['metrics']
        pair=dict(index=r['index'],title=r['title'],directory=r['directory'],geometry_id=r['geometry_id'],reference_sha256=sha(old_dir/'reference.npz'),original_reference_path=str((old_dir/'reference.npz').relative_to(ROOT)),old=prior_metrics,new=r['metrics'],mean_errors=mean)
        r['mean_errors']=mean;(new_dir/'metrics.json').write_text(json.dumps(r,indent=2)+'\n');pairs.append(pair)
    summary=dict(checkpoint_step=82216,old_checkpoint_step=32216,checkpoint_sha256=manifest['checkpoint_sha256'],reynolds=30.,expected=len(records),passed=len(records),failed=0,fem_rerun=False,same_reference_points_weights_and_geometry_verified=True,stress_extrapolation=manifest['stress_extrapolation'],accepted_model=False,cases=records,comparisons=pairs)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (output/'mean_errors.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    keys=['velocity_relative_l2','pressure_relative_l2','pressure_drop_relative']
    with (output/'comparison.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['case','title','metric','old_32216','new_82216','relative_reduction_percent'])
        for p in pairs:
            for key in keys:writer.writerow([p['index'],p['title'],key,p['old'][key],p['new'][key],100*(1-p['new'][key]/max(p['old'][key],1e-16))])
    fig,axes=plt.subplots(3,1,figsize=(13,11),layout='constrained');x=np.arange(len(records))
    for ax,key,title in zip(axes,keys,['Velocity relative L2 (%)','Pressure relative L2 (%)','Pressure drop relative error (%)']):
        ax.bar(x-.2,[100*p['old'][key] for p in pairs],.4,label='Step 32,216',color='#94b6cd');ax.bar(x+.2,[100*p['new'][key] for p in pairs],.4,label='Step 82,216',color='#0f766e');ax.set(title=title,xticks=x,xticklabels=[p['index'] for p in pairs],xlabel='Original case order');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.suptitle('Same geometry, Re=30, FEM fields, points and weights | original vs new checkpoint',fontsize=15);fig.savefig(output/'before_after.png',dpi=160);fig.savefig(output/'before_after.pdf');plt.close(fig)
    if (source/'geometry_overview.png').exists():shutil.copy2(source/'geometry_overview.png',output/'geometry_overview.png')
    lines=[f"# {len(records)} cases refreshed with checkpoint 82,216",'', '[All case chains PDF](all_chains.pdf) · [Old/new comparison CSV](comparison.csv) · [Absolute MAE and relative L1 CSV](mean_errors.csv) · [Full numerical metrics](summary.json)','', '**Only model inference was rerun**, using the exact saved geometry and FEM quadrature points. Reference NPZ files, FEM fields, weights and geometry arrays were verified unchanged. The original results remain in `'+str(source.relative_to(ROOT))+'`. All cases use Re=30. New weights: step 82,216; old weights: step 32,216. Research weights remain unaccepted.','', '**All five cases in this group are stress/extrapolation queries outside at least one training-geometry bound. They are not evidence of accuracy inside the training envelope.**' if manifest['stress_extrapolation'] else 'These are the original prescribed examples inside the declared geometry envelope, not the locked final benchmark.','', '![Before and after](before_after.png)','', '| Case | Velocity L2 old → new | Pressure L2 old → new | Drop error old → new |','| --- | ---: | ---: | ---: |']
    for p in pairs:lines.append(f"| {p['index']:02d} {p['title']} | {p['old'][keys[0]]:.2%} → {p['new'][keys[0]]:.2%} | {p['old'][keys[1]]:.2%} → {p['new'][keys[1]]:.2%} | {p['old'][keys[2]]:.2%} → {p['new'][keys[2]]:.2%} |")
    lines+=['', 'Relative L2 uses physical-area quadrature weights and the FEM field norm as denominator. Velocity uses the vector norm. MAE is the area-weighted mean absolute scalar error or Euclidean vector error. Relative L1 divides integrated error magnitude by integrated reference magnitude (1e-16 floor). It is not pointwise percentage error. CSV L1 values are percentages; absolute MAE is in nondimensional field units. For nearly zero reference v, use absolute MAE/RMS rather than interpreting an ill-conditioned relative v ratio; the floor flag and reference mean magnitude are in JSON. Pressure is not shifted to fit FEM. Colors match model and FEM per field/case; absolute-error scales are per panel. Original common-domain exclusions and FEM uncertainty are retained.','']
    for p in pairs:
        d=p['directory'];lines += [f"## {p['index']:02d}. {p['title']}",'',f'[Geometry]({d}/01_geometry.png) → [New model]({d}/02_model.png) → [Saved FEM]({d}/03_fem.png) → [New errors]({d}/04_error.png) · [Fields]({d}/fields.npz) · [Metrics]({d}/metrics.json)','',f'![New comparison chain]({d}/chain.png)','']
    (output/'README.md').write_text('\n'.join(lines))


def main():
    import geometry_operator.reference as reference
    def forbidden(*args,**kwargs):raise RuntimeError('FEM solving is forbidden in this inference-only refresh')
    reference.solve_reference=forbidden;reference.build_case=forbidden
    latest=json.loads((ROOT/'artifacts/kaggle/latest_checkpoint.json').read_text());checkpoint=ROOT/latest['path'];assert latest['step']==82216 and sha(checkpoint)==latest['sha256']
    OUT.mkdir(exist_ok=True);downloads=OUT/'downloads';downloads.mkdir(exist_ok=True);index=[]
    for name,count,stress in GROUPS:
        source=ROOT/'reports'/name;target=OUT/name;target.mkdir(exist_ok=True)
        original=json.loads((source/'manifest.json').read_text());assert len(original['cases'])==count
        manifest=dict(original);manifest.update(checkpoint=latest['path'],checkpoint_sha256=latest['sha256'],checkpoint_step=82216,source_report=str(source.relative_to(ROOT)),stress_extrapolation=stress,fem_rerun=False)
        (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        for case in manifest['cases']:
            directory=target/case['directory'];directory.mkdir(exist_ok=True)
            for filename in ['geometry.npz','reference.npz']:shutil.copy2(source/case['directory']/filename,directory/filename)
        def summary_callback(output,manifest,records,plt,colors):summarize_group(source,output,manifest,records,plt)
        render(target,manifest,checkpoint,summary_function=summary_callback,pdf_name='all_chains.pdf',allow_stress=stress)
        report=json.loads((target/'summary.json').read_text())
        bundle=downloads/f'{name}_step_82216.zip'
        with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
            for p in sorted(target.rglob('*')):
                if p.is_file():archive.write(p,p.relative_to(OUT))
        index.append(dict(group=name,cases=count,archive=bundle.name,archive_bytes=bundle.stat().st_size,archive_sha256=sha(bundle),median_old={k:float(np.median([p['old'][k] for p in report['comparisons']])) for k in ['velocity_relative_l2','pressure_relative_l2']},median_new={k:float(np.median([p['new'][k] for p in report['comparisons']])) for k in ['velocity_relative_l2','pressure_relative_l2']}))
    (OUT/'index.json').write_text(json.dumps(dict(checkpoint_step=82216,checkpoint_sha256=latest['sha256'],total_cases=25,fem_rerun=False,groups=index),indent=2)+'\n')
    lines=['# Updated predictions for all 25 existing cases','', '**New checkpoint: step 82,216. Original FEM reused exactly. No FEM solve or training was run. Original step-32,216 reports remain unchanged.**','', '| Set | Cases | New results | PDF | Complete ZIP |','| --- | ---: | --- | --- | --- |']
    for item in index:
        name=item['group'];lines.append(f"| {name} | {item['cases']} | [Gallery]({name}/README.md) | [PDF]({name}/all_chains.pdf) | [Download ZIP](downloads/{item['archive']}) |")
    lines+=['','Each ZIP contains all new model/FEM/error figures, complete per-case chain images, numerical predictions, unchanged FEM references and geometry, old/new comparison metrics, MAE/relative L1 tables and the multipage PDF. Unzip and open the included README/PDF; no model execution is needed. Archive SHA-256 values are recorded in [index.json](index.json).','', 'All 25 cases are Re=30. The extreme set is explicitly outside the training geometry envelope and is a separate extrapolation experiment. Results are not uniformly improved; the paired tables retain every case and metric. The new checkpoint remains unaccepted.','', 'Reproduce locally: `python -m scripts.refresh_saved_cases` in the documented CPU environment. The script disables FEM solver entry points during the refresh, verifies identical references/points/weights and labels figures with the actual checkpoint step.','']
    (OUT/'README.md').write_text('\n'.join(lines));print(json.dumps(index,indent=2))


if __name__=='__main__':main()
