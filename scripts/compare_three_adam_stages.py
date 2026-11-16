"""Frozen inference on all 20 saved in-envelope cases; seeded six-case chain gallery."""
import csv,hashlib,json,shutil,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry_operator.api import FlowOperator
from geometry_operator.evaluate import weighted_relative
from geometry_operator.geometry import validate
from geometry_operator.reference import pressure_sections
from scripts.channel_chain import render

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/adam_next50k_comparison'
CP=ROOT/'artifacts/kaggle/adam-next50k-v1/checkpoint.npz'
STEPS=[32216,82216,132216]
KEYS=['velocity_relative_l2','u_relative_l2','v_absolute_rms','pressure_relative_l2','pressure_drop_relative','velocity_mae','pressure_mae','velocity_relative_l1','pressure_relative_l1']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def measurements(p,f,w,drop,refdrop):
    e=p-f;norm=w/w.sum()
    return dict(velocity_relative_l2=weighted_relative(p[:,:2],f[:,:2],w),u_relative_l2=weighted_relative(p[:,0],f[:,0],w),
        v_absolute_rms=float(np.sqrt(norm@e[:,1]**2)),pressure_relative_l2=weighted_relative(p[:,2],f[:,2],w),
        pressure_drop_relative=abs(drop-refdrop)/max(abs(refdrop),1e-8),
        velocity_mae=float(norm@np.linalg.norm(e[:,:2],axis=1)),pressure_mae=float(norm@abs(e[:,2])),
        velocity_relative_l1=float(w@np.linalg.norm(e[:,:2],axis=1)/max(w@np.linalg.norm(f[:,:2],axis=1),1e-16)),
        pressure_relative_l1=float(w@abs(e[:,2])/max(w@abs(f[:,2]),1e-16)))



def add_improvements(records,aggregates):
    rows=[]
    for r in records:
        for key in KEYS:
            a,b,c=[r['metrics'][str(step)][key] for step in STEPS]
            rows.append({'case':r['case_id'],'title':r['title'],'metric':key,'step_32216':a,'step_82216':b,'step_132216':c,
                         'reduction_percent_vs_32216':100*(1-c/max(a,1e-16)),
                         'reduction_percent_vs_82216':100*(1-c/max(b,1e-16))})
    with (OUT/'improvements.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    p=OUT/'README.md';text=p.read_text()
    marker='## Per-group progress and regressions'
    if marker not in text:
        lines=['',marker,'','[Per-case reduction percentages CSV](improvements.csv). Positive reduction means improved; negative means worsened. Percent reductions compare error magnitudes, not percentage-point differences.','',
               '| Group | Velocity median 82k → 132k | Pressure median 82k → 132k |','| --- | ---: | ---: |']
        for group in ['channel_chain_15','random_channel_chain_5']:
            a=aggregates[group]['velocity_relative_l2']['medians'];b=aggregates[group]['pressure_relative_l2']['medians']
            lines.append(f"| {group} | {a['82216']:.2%} → {a['132216']:.2%} | {b['82216']:.2%} → {b['132216']:.2%} |")
        lines+=['','The varied five-case group has a higher median velocity error after the new run, despite the overall 20-case median improving. Improvements are not uniform across geometries or quantities.','']
        p.write_text(text+'\n'.join(lines))


def main():
    import geometry_operator.reference as reference
    def forbidden(*args,**kwargs):raise RuntimeError('FEM solves are forbidden: reuse saved references')
    reference.solve_reference=forbidden;reference.build_case=forbidden
    selection=json.loads((OUT/'selection.json').read_text());selected=set(selection['selected_case_ids'])
    status=json.loads((CP.parent/'status.json').read_text());assert status['step']==132216 and status['stop_reason']=='target_completed'
    model=FlowOperator(CP,allow_unvalidated=True,chunk_size=512);assert model.metadata['step']==132216
    assert model.metadata['optimizer_present']
    dataout=OUT/'cases';dataout.mkdir(exist_ok=True);gallery=OUT/'selected';gallery.mkdir(exist_ok=True)
    records=[];chains=[]
    for item in selection['population']:
        gid=item['case_id'];group=item['group'];dirname=item['directory'];source=ROOT/'reports'/group/dirname
        middle=ROOT/'reports/checkpoint_82216'/group/dirname
        original_manifest=json.loads((source.parent/'manifest.json').read_text())
        info=next(c for c in original_manifest['cases'] if c['index']==item['original_index'])
        target=dataout/f'{gid:02d}_{info["slug"]}';target.mkdir(exist_ok=True)
        for name in ['geometry.npz','reference.npz']:shutil.copy2(source/name,target/name)
        assert sha(source/'reference.npz')==sha(middle/'reference.npz')
        with np.load(source/'fields.npz',allow_pickle=False) as old,np.load(middle/'fields.npz',allow_pickle=False) as mid,np.load(source/'reference.npz',allow_pickle=False) as ref:
            for key in ['coeff','xy','weights','fem']:np.testing.assert_array_equal(old[key],mid[key])
            coeff=old['coeff'];xy=old['xy'];w=old['weights'];f=old['fem'];validate(coeff)
            np.testing.assert_array_equal(f,ref['fields']);np.testing.assert_array_equal(w,ref['weights']);np.testing.assert_array_equal(xy,ref['xy'])
            rm=json.loads(str(ref['metadata']));assert max(rm['mesh_uncertainty'].values())<=.01
            pred=model.evaluate(coeff,30.,xy)
            drop=pressure_sections(lambda q:model.evaluate(coeff,30.,q),coeff)
            stages={}
            for step,fields,folder in [(32216,old['predicted'],source),(82216,mid['predicted'],middle),(132216,pred,None)]:
                dp=drop if folder is None else json.loads((folder/'metrics.json').read_text())['metrics']['pressure_drop_predicted']
                stages[str(step)]=measurements(fields,f,w,dp,rm['pressure_drop'])
            assert np.isfinite(pred).all() and (w>0).all()
            np.savez_compressed(target/'fields.npz',xy=xy,weights=w,fem=f,predicted=pred,coeff=coeff,reynolds=30.,checkpoint_sha256=model.sha256)
        record={**item,'result_directory':str(target.relative_to(OUT)),'selected_for_chain':gid in selected,'metrics':stages,
                'reference_sha256':sha(source/'reference.npz'),'geometry_sha256':sha(source/'geometry.npz'),'mesh_uncertainty':rm['mesh_uncertainty']}
        (target/'metrics.json').write_text(json.dumps(record,indent=2)+'\n');records.append(record)
        if gid in selected:
            g=gallery/target.name;g.mkdir(exist_ok=True)
            for name in ['geometry.npz','reference.npz']:shutil.copy2(source/name,g/name)
            chains.append(dict(info,index=gid,original_index=item['original_index'],source_group=group,directory=target.name))
        print(json.dumps({'case':gid,'title':item['title'],'velocity_errors':[stages[str(s)]['velocity_relative_l2'] for s in STEPS]}),flush=True)
    manifest={'checkpoint':str(CP.relative_to(ROOT)),'checkpoint_sha256':model.sha256,'checkpoint_step':132216,'reynolds':30.,
              'case_count':len(chains),'cases':chains,'selection':selection,'fem_rerun':False,'accepted_model':False}
    (gallery/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    def callback(output,manifest,rendered,plt,colors):
        for r in rendered:
            original=next(c for c in records if c['case_id']==r['index'])
            np.testing.assert_allclose(r['metrics']['velocity_relative_l2'],original['metrics']['132216']['velocity_relative_l2'],rtol=1e-12)
        (output/'summary.json').write_text(json.dumps({'checkpoint_step':132216,'cases':rendered,'fem_rerun':False},indent=2)+'\n')
    render(gallery,manifest,CP,summary_function=callback,pdf_name='six_random_chains.pdf')
    rows=[]
    for r in records:
        for step in STEPS:rows.append({'case':r['case_id'],'title':r['title'],'group':r['group'],'step':step,**r['metrics'][str(step)]})
    with (OUT/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    aggregates={}
    for label,rr in [('all_20',records),('channel_chain_15',records[:15]),('random_channel_chain_5',records[15:])]:
        aggregates[label]={}
        for key in KEYS:
            vals={str(s):float(np.median([r['metrics'][str(s)][key] for r in rr])) for s in STEPS}
            old=np.array([r['metrics']['82216'][key] for r in rr]);new=np.array([r['metrics']['132216'][key] for r in rr])
            aggregates[label][key]={'medians':vals,'improved_vs_82216':int((new<old).sum()),'worsened_vs_82216':int((new>old).sum()),
                                   'median_error_reduction_percent_vs_82216':100*(1-vals['132216']/max(vals['82216'],1e-16))}
    summary={'checkpoint_step':132216,'checkpoint_sha256':model.sha256,'checkpoint_path':str(CP.relative_to(ROOT)),
             'steps':STEPS,'seed':selection['seed'],'selected_case_ids':sorted(selected),'cases':records,'aggregates':aggregates,
             'reynolds':30.,'same_geometry_reference_points_weights_verified':True,'fem_rerun':False,'accepted_model':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fig,axes=plt.subplots(3,1,figsize=(14,12),layout='constrained');x=np.arange(1,21)
    for ax,key,title,percent in zip(axes,['velocity_relative_l2','pressure_relative_l2','v_absolute_rms'],['Velocity relative L2 (%)','Pressure relative L2 (%)','Transverse velocity absolute RMS'],[True,True,False]):
        for step,color in zip(STEPS,['#94a3b8','#0f766e','#7c3aed']):
            ax.plot(x,[(100 if percent else 1)*r['metrics'][str(step)][key] for r in records],'o-',color=color,label=f'Step {step:,}',ms=4)
        ax.axvline(15.5,color='gray',ls=':');ax.set(title=title,xlabel='Saved case ID (1-15 original; 16-20 varied)',xticks=x);ax.grid(alpha=.2);ax.legend()
    fig.suptitle('All 20 unchanged cases | same FEM, geometry, Re=30, points and weights',fontsize=15)
    fig.savefig(OUT/'three_stage_errors.png',dpi=180);fig.savefig(OUT/'three_stage_errors.pdf');plt.close(fig)
    lines=['# Three Adam stages: all 20 saved cases and six seeded random chains','',
           '**New checkpoint: 132,216. Previous stages: 32,216 and 82,216. All 20 standard-envelope examples use Re=30.** Same geometry, FEM fields, physical quadrature points and weights were verified exactly. No FEM solve or retraining was performed. Full benchmark acceptance is not claimed.','',
           '[Six random chains PDF](selected/six_random_chains.pdf) · [All-error comparison CSV](comparison.csv) · [Full metrics](summary.json) · [Complete ZIP](three_stage_20_cases.zip)','',
           '![Three-stage errors](three_stage_errors.png)','',
           '| Metric (median over all 20) | 32,216 | 82,216 | 132,216 | Improved cases vs 82,216 |','| --- | ---: | ---: | ---: | ---: |']
    for key,label in [('velocity_relative_l2','Velocity L2'),('pressure_relative_l2','Pressure L2'),('pressure_drop_relative','Pressure drop')]:
        a=aggregates['all_20'][key];v=a['medians'];lines.append(f"| {label} | {v['32216']:.2%} | {v['82216']:.2%} | {v['132216']:.2%} | {a['improved_vs_82216']}/20 |")
    lines+=['','| Case | Velocity L2: 32k → 82k → 132k | Pressure L2: 32k → 82k → 132k |','| --- | ---: | ---: |']
    for r in records:
        text=lambda k:' → '.join(f"{r['metrics'][str(s)][k]:.2%}" for s in STEPS)
        lines.append(f"| {r['case_id']:02d} {r['title']} | {text('velocity_relative_l2')} | {text('pressure_relative_l2')} |")
    lines+=['','## Random chain gallery','','Seed **20261116** selects IDs '+', '.join(map(str,sorted(selected)))+' uniformly without replacement from all 20 before evaluating the new errors. The selection was not adjusted for favorable results. Each chain shows geometry → new u/v/speed/p → unchanged FEM → absolute error. Separate panels and numerical fields are included.','']
    for r in records:
        if r['selected_for_chain']:
            d='selected/'+Path(r['result_directory']).name
            lines += [f"### {r['case_id']:02d}. {r['title']}",'',f'[Full chain]({d}/chain.png) · [Metrics]({r["result_directory"]}/metrics.json)','',f'![Chain]({d}/chain.png)','']
    lines+=['## Error definitions and limits','','Relative L2 = sqrt(sum(w |prediction-FEM|²) / sum(w |FEM|²)); velocity uses both components. MAE is the physical-area mean absolute scalar error or vector error magnitude. Relative L1 divides integrated absolute error by integrated reference magnitude. Denominator floors match the existing protocol. Pressure has no fitted gauge shift. Near-zero transverse velocity is reported using absolute RMS. CSV/JSON errors are ratios, while README L2 entries are percentages. Every regression is retained. These illustrative cases do not cover all Re or the locked test.','',
            'Reproduce: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_ENABLE_X64=true .venv/bin/python -m scripts.compare_three_adam_stages`. ZIP contains all 20 new numerical cases, all six illustrated chains, PDF, selection manifest and three-stage metrics; model weights are stored separately at the recorded checkpoint path.','']
    (OUT/'README.md').write_text('\n'.join(lines))
    add_improvements(records,aggregates)
    with zipfile.ZipFile(OUT/'three_stage_20_cases.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.suffix!='.zip':z.write(p,arcname=p.relative_to(OUT.parent))
    with zipfile.ZipFile(OUT/'three_stage_20_cases.zip') as z:
        assert z.testzip() is None
        for name in z.namelist():assert z.read(name)==(OUT.parent/name).read_bytes()
    print(json.dumps(aggregates,indent=2),flush=True)

if __name__=='__main__':main()
