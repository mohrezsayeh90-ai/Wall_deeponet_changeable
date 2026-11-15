"""Fifteen prescribed channels: geometry -> frozen operator -> FEM -> error."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import multiprocessing
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def designs():
    from geometry_operator.geometry import BREAKS, _canonical, validate, geometry_id
    knots = np.r_[np.zeros(6), BREAKS[1:-1], np.full(6, 5.)]
    x = np.array([np.mean(knots[i+1:i+6]) for i in range(len(knots)-6)])
    def bump(center, width):
        z = (x-center)/width
        return np.exp(1-1/np.maximum(1-z*z, 1e-12))*(abs(z)<1)
    a, b, c = bump(1.45,.65), bump(2.5,.8), bump(3.55,.65)
    narrow = [bump(p,.48) for p in [1.15,2.05,2.95,3.85]]
    zero = np.zeros_like(x)
    mixed_bottom=.20*bump(1.2,.50)-.16*bump(2.5,.62)+.14*bump(3.7,.50)
    mixed_top=-.15*bump(1.7,.55)+.16*bump(3.1,.55)-.08*bump(3.9,.40)
    # Values are centerline displacement and gap displacement, respectively.
    specs = [
        ('straight', 'Straight channel', zero, zero),
        ('mild_constriction', 'Mild symmetric constriction', zero, -.08*b),
        ('moderate_constriction', 'Moderate symmetric constriction', zero, -.28*b),
        ('expansion', 'Symmetric expansion', zero, .36*b),
        ('bottom_bump', 'One bump on bottom wall', .11*b, -.22*b),
        ('top_bump', 'One bump on top wall', -.13*b, -.26*b),
        ('centerline_bend', 'Constant-gap centerline bend', .20*b, zero),
        ('s_bend', 'Constant-gap S bend', .15*(a-c), zero),
        ('double_constriction', 'Two symmetric constrictions', zero, -.24*(a+c)),
        ('constriction_expansion', 'Constriction followed by expansion', zero, -.25*a+.30*c),
        ('staggered_walls', 'Staggered independent wall bumps', .09*a-.10*c, -.18*a-.20*c),
        ('triple_constriction', 'Three symmetric constrictions', zero, -.23*(bump(1.3,.5)+bump(2.5,.5)+bump(3.7,.5))),
        ('oscillating_gap', 'Four alternating gap features', zero, sum((-1)**(i+1)*.25*v for i,v in enumerate(narrow))),
        ('oscillating_walls', 'Oscillating centerline and gap', sum((-1)**i*.12*v for i,v in enumerate(narrow)), sum((-1)**(i//2)*.20*v for i,v in enumerate(narrow))),
        ('mixed_independent', 'Mixed independent multiscale walls', (mixed_bottom+mixed_top)/2, mixed_top-mixed_bottom),
    ]
    result=[]
    for index,(slug,title,center,gap) in enumerate(specs,1):
        coeff = _canonical([center,gap])
        # Keep every prescribed channel inside the operator's geometric envelope.
        from geometry_operator.geometry import descriptors, sample_numpy
        dd=descriptors(coeff);cc,gg=sample_numpy(coeff,np.linspace(0,5,4001))
        scale=min(1.,.399/max(-gg.min(),1e-12),.499/max(gg.max(),1e-12),.349/max(abs(cc).max(),1e-12),.699/max(dd['max_slope'],1e-12))
        coeff*=scale
        description=validate(coeff)
        result.append((dict(index=index,slug=slug,title=title,geometry_id=geometry_id(coeff),reynolds=30.,amplitude_scale=scale,descriptors=description),coeff))
    return result


def prepare(output, checkpoint):
    from geometry_operator.geometry import sample_numpy
    output.mkdir(parents=True,exist_ok=True)
    records=[]
    for info,coeff in designs():
        case=output/f"{info['index']:02d}_{info['slug']}";case.mkdir(exist_ok=True)
        path=case/'geometry.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as old:
                np.testing.assert_array_equal(old['coeff'],coeff)
        else:
            x=np.linspace(0,5,1001);center,gap=sample_numpy(coeff,x)
            np.savez_compressed(path,coeff=coeff,x=x,bottom=center-gap/2,top=1+center+gap/2)
        records.append(dict(**info,directory=case.name))
    manifest=dict(schema='channel-chain-15-v1',created_at_utc=datetime.now(timezone.utc).isoformat(),checkpoint=str(checkpoint.relative_to(ROOT)),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),checkpoint_step=32216,reynolds=30.,case_count=15,selection='Prescribed analytic B-spline wall designs ordered by illustrative structural complexity; fixed before error evaluation, no selection using model errors.',locked_test_used=False,trained_model_accepted=False,cases=records)
    old=output/'manifest.json'
    if old.exists():
        existing=json.loads(old.read_text());assert existing['cases']==records and existing['checkpoint_sha256']==manifest['checkpoint_sha256']
    else:old.write_text(json.dumps(manifest,indent=2)+'\n')
    return json.loads(old.read_text())


def reference_task(task):
    directory,re=task
    directory=Path(directory)
    from geometry_operator.reference import build_case
    with np.load(directory/'geometry.npz',allow_pickle=False) as d:coeff=d['coeff']
    try:
        meta=build_case(coeff,re,directory/'reference.npz',nx=40,ny=12,mesh_rtol=.01)
        return dict(directory=directory.name,status='passed',reference=meta)
    except Exception as exc:
        return dict(directory=directory.name,status='failed',error=str(exc))


def references(output,manifest,workers):
    status={}
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as executor:
        futures=[executor.submit(reference_task,(str(output/c['directory']),c['reynolds'])) for c in manifest['cases']]
        for future in as_completed(futures):
            item=future.result();status[item['directory']]=item
            print(json.dumps(item),flush=True)
            (output/'reference-status.json').write_text(json.dumps(dict(expected=len(manifest['cases']),completed=len(status),cases=[status[k] for k in sorted(status)]),indent=2)+'\n')
    if any(x['status']!='passed' for x in status.values()):raise RuntimeError('Unresolved FEM cases; see reference-status.json')


def render(output,manifest,checkpoint,summary_function=None,pdf_name='all_15_chains.pdf'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib.backends.backend_pdf import PdfPages
    from geometry_operator.api import FlowOperator
    from geometry_operator.geometry import sample_numpy
    from geometry_operator.reference import pressure_sections
    from geometry_operator.evaluate import weighted_relative
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.facecolor':'white','savefig.facecolor':'white'})
    model=FlowOperator(checkpoint,allow_unvalidated=True,chunk_size=256)
    assert model.sha256==manifest['checkpoint_sha256']
    results=[]
    colors=['#155e75','#b45309','#7c3aed','#be123c']
    def walls_on(ax,x,b,t):
        ax.plot(x,b,color='#111827',lw=1.1);ax.plot(x,t,color='#111827',lw=1.1)
        ax.set(xlim=(0,5),ylim=(-.48,1.48),xlabel='x',ylabel='y',aspect='equal')
    with PdfPages(output/pdf_name) as pdf:
        for info in manifest['cases']:
            case=output/info['directory']
            with np.load(case/'geometry.npz',allow_pickle=False) as d:coeff=d['coeff'];x=d['x'];bottom=d['bottom'];top=d['top']
            with np.load(case/'reference.npz',allow_pickle=False) as d:xy=d['xy'];weights=d['weights'];truth=d['fields'];refmeta=json.loads(str(d['metadata']))
            pred=model.evaluate(coeff,info['reynolds'],xy)
            error=pred-truth;speed_ref=np.linalg.norm(truth[:,:2],axis=1);speed_pred=np.linalg.norm(pred[:,:2],axis=1)
            w=weights/weights.sum()
            drop=pressure_sections(lambda q:model.evaluate(coeff,info['reynolds'],q),coeff)
            metrics=dict(velocity_relative_l2=weighted_relative(pred[:,:2],truth[:,:2],weights),u_relative_l2=weighted_relative(pred[:,0],truth[:,0],weights),v_absolute_rms=float(np.sqrt(w@(error[:,1]**2))),pressure_relative_l2=weighted_relative(pred[:,2],truth[:,2],weights),speed_relative_l2=weighted_relative(speed_pred,speed_ref,weights),velocity_absolute_rms=float(np.sqrt(w@np.sum(error[:,:2]**2,axis=1))),pressure_absolute_rms=float(np.sqrt(w@(error[:,2]**2))),velocity_max_abs_at_quadrature=float(np.linalg.norm(error[:,:2],axis=1).max()),pressure_max_abs_at_quadrature=float(abs(error[:,2]).max()),pressure_drop_predicted=drop,pressure_drop_reference=refmeta['pressure_drop'],pressure_drop_relative=abs(drop-refmeta['pressure_drop'])/max(abs(refmeta['pressure_drop']),1e-8),physical_area_evaluated=float(weights.sum()),common_domain_excluded_area_fraction=refmeta['common_domain_excluded_area_fraction'])
            if not np.isfinite(np.r_[pred.ravel(),list(metrics.values())]).all():raise RuntimeError('Nonfinite result')
            np.savez_compressed(case/'fields.npz',xy=xy,weights=weights,fem=truth,predicted=pred,error=error,speed_fem=speed_ref,speed_predicted=speed_pred,coeff=coeff,reynolds=info['reynolds'],checkpoint_sha256=model.sha256)
            record=dict(**info,status='passed',checkpoint_sha256=model.sha256,metrics=metrics,mesh_uncertainty=refmeta['mesh_uncertainty'],fem_coarse=refmeta['coarse'],fem_fine=refmeta['fine'])
            (case/'metrics.json').write_text(json.dumps(record,indent=2)+'\n');results.append(record)
            cc_grid,gg_grid=sample_numpy(coeff,xy[:,0]);eta=(xy[:,1]-(cc_grid-gg_grid/2))/(1+gg_grid)
            reftri=mtri.Triangulation(xy[:,0],eta)
            tri=mtri.Triangulation(*xy.T,triangles=reftri.triangles)
            cent=xy[tri.triangles].mean(axis=1);cc,gg=sample_numpy(coeff,cent[:,0]);bb=cc-gg/2
            tri.set_mask((cent[:,1]<bb)|(cent[:,1]>bb+1+gg))
            values_ref=[truth[:,0],truth[:,1],speed_ref,truth[:,2]];values_pred=[pred[:,0],pred[:,1],speed_pred,pred[:,2]]
            labels=['u: axial velocity','v: transverse velocity','Speed magnitude','p: pressure']
            limits=[]
            for j,(f,p) in enumerate(zip(values_ref,values_pred)):
                if j==1:
                    mm=max(abs(f).max(),abs(p).max(),1e-8);limits.append((-mm,mm))
                else:limits.append((min(f.min(),p.min()),max(f.max(),p.max())+1e-12))
            def geometry_panel(ax):
                ax.fill_between(x,bottom,top,color='#dbeafe');walls_on(ax,x,bottom,top)
                ax.plot(x,(bottom+top)/2,'--',color='#0e7490',lw=.9,label='Centerline')
                ax.annotate('Inlet: Q = 1',xy=(0,.5),xytext=(.35,1.25),arrowprops={'arrowstyle':'->'})
                ax.text(4.0,1.27,'Zero-traction outlet',ha='center',fontsize=9)
                ax.set_title(f"{info['index']:02d}  {info['title']}  |  Re = 30",fontweight='bold')
            def field_panels(axes,values,error_row=False):
                for j,(ax,value) in enumerate(zip(axes,values)):
                    if error_row:lo,hi=0,max(value.max(),1e-12);cmap='magma'
                    else:lo,hi=limits[j];cmap='RdBu_r' if j==1 else 'viridis'
                    im=ax.tripcolor(tri,value,shading='gouraud',cmap=cmap,vmin=lo,vmax=hi,rasterized=True)
                    walls_on(ax,x,bottom,top);ax.set_title(('Absolute error: ' if error_row else '')+labels[j],fontsize=10)
                    ax.figure.colorbar(im,ax=ax,orientation='horizontal',pad=.30,shrink=.92,format='%.3g',fraction=.06)
            fig,ax=plt.subplots(figsize=(11,3.4),layout='constrained');geometry_panel(ax);fig.savefig(case/'01_geometry.png',dpi=150);plt.close(fig)
            for stage,title,vals,err in [('02_model','2. Frozen DeepONet',values_pred,False),('03_fem','3. Independent mesh-checked FEM',values_ref,False),('04_error','4. Pointwise absolute error',list(map(lambda z:abs(z[0]-z[1]),zip(values_pred,values_ref))),True)]:
                fig,axes=plt.subplots(2,2,figsize=(12,6.5),layout='constrained');field_panels(axes.flat,vals,err)
                fig.suptitle(f"{info['index']:02d} {info['title']} | {title} | Re = 30",fontweight='bold')
                fig.savefig(case/f'{stage}.png',dpi=145);plt.close(fig)
            fig=plt.figure(figsize=(17,12),layout='constrained');gs=fig.add_gridspec(4,4,height_ratios=[1.2,1,1,1])
            geometry_panel(fig.add_subplot(gs[0,:2]));text=fig.add_subplot(gs[0,2:]);text.axis('off')
            text.text(0,.95,f"1 Geometry -> 2 Model -> 3 FEM -> 4 Error\n\nVelocity relative L2: {metrics['velocity_relative_l2']:.2%}     Pressure relative L2: {metrics['pressure_relative_l2']:.2%}\nPressure-drop error: {metrics['pressure_drop_relative']:.2%}     v absolute RMS: {metrics['v_absolute_rms']:.4g}\nFEM fine mesh: {refmeta['fine']['nx']} x {refmeta['fine']['ny']}     Mesh uncertainty (velocity): {refmeta['mesh_uncertainty']['velocity']:.2%}\n\nFrozen checkpoint: 32,216 steps | research weights, not accepted\nSame color limits for model and FEM; error scales are per panel.",va='top',fontsize=12,linespacing=1.6)
            for row,vals,err,title in [(1,values_pred,False,'2 MODEL'),(2,values_ref,False,'3 FEM'),(3,[abs(p-f) for p,f in zip(values_pred,values_ref)],True,'4 ERROR')]:
                axes=[fig.add_subplot(gs[row,j]) for j in range(4)];field_panels(axes,vals,err);axes[0].set_ylabel(title+'\ny')
            fig.suptitle(f"Case {info['index']:02d}: {info['title']} | Re = 30",fontsize=18,fontweight='bold')
            fig.savefig(case/'chain.png',dpi=145);pdf.savefig(fig,dpi=145);plt.close(fig)
            print(json.dumps({'case':info['index'],'velocity_error':metrics['velocity_relative_l2'],'pressure_error':metrics['pressure_relative_l2']}),flush=True)
    (summary_function or summarize)(output,manifest,results,plt,colors)


def summarize(output,manifest,records,plt,colors):
    import csv,html
    metrics=['velocity_relative_l2','pressure_relative_l2','pressure_drop_relative','v_absolute_rms']
    with (output/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,lineterminator='\n',fieldnames=['index','title','reynolds']+metrics+['mesh_velocity_uncertainty','mesh_pressure_uncertainty','mesh_drop_uncertainty'])
        writer.writeheader()
        for r in records:writer.writerow({**{k:r[k] for k in ['index','title','reynolds']},**{k:r['metrics'][k] for k in metrics},**{'mesh_'+n+'_uncertainty':r['mesh_uncertainty'][m] for n,m in [('velocity','velocity'),('pressure','pressure'),('drop','pressure_drop')]}})
    report=dict(checkpoint_sha256=manifest['checkpoint_sha256'],checkpoint_step=32216,reynolds=30.,expected=15,passed=len(records),failed=15-len(records),all_fem_mesh_checks_passed=all(max(r['mesh_uncertainty'].values())<=.01 for r in records),locked_test_used=False,accepted_model=False,case_selection=manifest['selection'],cases=records)
    (output/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
    fig,axes=plt.subplots(1,2,figsize=(13,4.7),layout='constrained')
    for color,metric,label in zip(colors,metrics[:3],['Velocity vector L2','Pressure L2','Pressure drop']):axes[0].plot(range(1,16),[100*r['metrics'][metric] for r in records],'o-',label=label,color=color)
    axes[0].set(xlabel='Prescribed case order (simple to complex)',ylabel='Relative error (%)',xticks=range(1,16));axes[0].legend();axes[0].grid(alpha=.2)
    for color,metric in zip(colors,['velocity','pressure','pressure_drop']):axes[1].plot(range(1,16),[100*r['mesh_uncertainty'][metric] for r in records],'o-',label=metric,color=color)
    axes[1].axhline(1,color='black',ls='--',label='1% refinement criterion');axes[1].set(xlabel='Case',ylabel='Coarse/fine FEM discrepancy (%)',xticks=range(1,16));axes[1].legend();axes[1].grid(alpha=.2)
    fig.suptitle('Frozen DeepONet vs independent FEM | 15 prescribed geometries | Re = 30');fig.savefig(output/'error_summary.png',dpi=170);plt.close(fig)
    fig,axes=plt.subplots(5,3,figsize=(15,12),layout='constrained')
    for ax,r in zip(axes.flat,records):
        with np.load(output/r['directory']/'geometry.npz',allow_pickle=False) as d:x,b,t=d['x'],d['bottom'],d['top']
        ax.fill_between(x,b,t,color='#dbeafe');ax.plot(x,b,color='#155e75');ax.plot(x,t,color='#155e75');ax.set(xlim=(0,5),ylim=(-.48,1.48),aspect='equal');ax.set_title(f"{r['index']:02d} {r['title']}",fontsize=10)
    fig.suptitle('15 prescribed channels: simple to structurally complex | Re = 30',fontsize=18);fig.savefig(output/'geometry_overview.png',dpi=160);plt.close(fig)
    header='''<!doctype html><html lang="fa" dir="rtl"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>۱۵ کانال: مدل، FEM و خطا</title><style>body{font-family:system-ui,sans-serif;background:#f4f7fb;color:#172b4d;max-width:1400px;margin:auto;padding:24px;line-height:1.8}section,header{background:white;border-radius:12px;padding:22px;margin:20px 0}img{width:100%;height:auto}a{color:#075985}table{border-collapse:collapse;width:100%;direction:ltr}td,th{border-bottom:1px solid #ddd;padding:8px;text-align:left}.links{display:flex;gap:22px;flex-wrap:wrap}.note{background:#fff5d6;padding:16px;border-radius:8px}</style><header><h1>۱۵ کانال؛ از هندسه تا مقایسه و خطا</h1><p>۱. شکل کانال ← ۲. سرعت و فشار مدل ← ۳. حل مستقل FEM ← ۴. خطای نقطه‌ای و معیارهای کمی</p><p>عدد رینولدز برای همه موارد ۳۰ است. وزن‌ها ثابت و متعلق به گام ۳۲٬۲۱۶ هستند؛ آموزش جدیدی انجام نشده است.</p><p class="note">این ۱۵ شکل از پیش طراحی شده‌اند و آزمون نهایی قفل‌شده نیستند. مدل هنوز تأیید نهایی نشده است. رنگ‌های مدل و FEM در هر مورد یکسان‌اند؛ مقیاس رنگ خطا در هر پنل جداست.</p><p><a href="all_15_chains.pdf">PDF تمام ۱۵ زنجیره</a> · <a href="summary.csv">جدول CSV</a> · <a href="summary.json">گزارش و داده‌های معیارها</a></p></header><section><img src="geometry_overview.png" alt="15 geometries"><img src="error_summary.png" alt="Errors and FEM mesh checks"></section>'''
    cards=[];rows=[]
    for r in records:
        d=r['directory'];title=html.escape(r['title']);m=r['metrics']
        rows.append(f"<tr><td><a href='#{d}'>{r['index']:02d} {title}</a></td><td>{m['velocity_relative_l2']:.2%}</td><td>{m['pressure_relative_l2']:.2%}</td><td>{m['pressure_drop_relative']:.2%}</td></tr>")
        cards.append(f"<section id='{d}'><h2 dir='ltr'>{r['index']:02d}. {title}</h2><div class='links'><a href='{d}/01_geometry.png'>۱. هندسه</a><a href='{d}/02_model.png'>۲. مدل</a><a href='{d}/03_fem.png'>۳. FEM</a><a href='{d}/04_error.png'>۴. خطا</a><a href='{d}/metrics.json'>معیارها</a><a href='{d}/fields.npz'>داده میدان‌ها</a></div><img loading='lazy' src='{d}/chain.png' alt='{title}: geometry model FEM error'><p>خطای نسبی سرعت: {m['velocity_relative_l2']:.2%}؛ فشار: {m['pressure_relative_l2']:.2%}؛ افت فشار: {m['pressure_drop_relative']:.2%}</p></section>")
    table="<section><table><tr><th>Case</th><th>Velocity L2</th><th>Pressure L2</th><th>Pressure drop</th></tr>"+''.join(rows)+"</table></section>"
    (output/'index.html').write_text(header+table+''.join(cards)+'</html>')
    md=['# Fifteen channels: geometry → model → FEM → error','', 'Frozen step-32,216 checkpoint, **Re=30 for every case**. These illustrative, prescribed cases do not open or replace the locked test. No new model training was performed.','', '[Interactive gallery](index.html) (download/open locally; GitHub does not render HTML) · [All 15 chains PDF](all_15_chains.pdf) · [CSV metrics](summary.csv) · [Full JSON](summary.json)','', '![Geometry overview](geometry_overview.png)','', '![Errors and mesh checks](error_summary.png)','', '| Case | Velocity L2 error | Pressure L2 error | Pressure-drop error |','| --- | ---: | ---: | ---: |']
    for r in records:
        m=r['metrics'];md.append(f"| [{r['index']:02d} {r['title']}](#{r['directory']}) | {m['velocity_relative_l2']:.2%} | {m['pressure_relative_l2']:.2%} | {m['pressure_drop_relative']:.2%} |")
    md+=['','## Method and limits','','Geometry uses the same exact C4 quintic wall representation as the operator. The order describes increasing structural variety (single feature → multiple features → independent mixed walls), not a mathematically monotonic complexity score or a ranking by prediction error. All designs are fixed before solving, validated within the declared geometry envelope, and retain flat inlet/outlet strips. Coefficients, sampled walls and geometric descriptors are saved for each case.','','Independent Taylor–Hood P2/P1 FEM uses the same inlet, no-slip walls, Laplacian viscous form and natural zero-traction outlet as the model. Meshes start at 40×12 and 80×24 and refine as necessary. All three coarse/fine discrepancies (velocity, pressure, pressure drop) must be ≤1%; algebraic convergence is checked. This controls a mesh-discrepancy estimate, not a rigorous error bound. Exact curved walls and straight-edged FEM meshes differ; only their common physical quadrature domain is used and the excluded area fraction is recorded (≤0.5%).','','At quadrature samples, error = model − FEM. Relative field L2 = sqrt(sum(w |model−FEM|²) / max(sum(w |FEM|²), 1e−16)), with physical-area weights. Velocity error uses the vector (u,v). Pressure is compared with its physical outlet anchoring, without a fitted pressure shift. Transverse velocity also has absolute RMS because its relative norm is ill-conditioned in a straight channel. Speed = sqrt(u²+v²). Pressure drop is section-averaged p(0.25)−p(4.75); relative-drop denominator has a 1e−8 floor. Full signed errors, weights, reference fields and predictions are saved in fields.npz.','','Model/FEM color limits match per case and field; transverse velocity uses a symmetric color scale. Absolute-error scales are per panel. Maps interpolate only the saved common-domain quadrature points on masked triangles; the small unsampled edge strip is blank, not extrapolated. Max errors refer to the quadrature samples, not the continuous domain. All quantities are nondimensional.','','Research weights remain unaccepted. These cases at one Reynolds number do not establish the full-domain benchmark, wall-stress accuracy, learned-baseline comparisons or completion of the 100,000-step training target.','','## Reproduce','','```bash','export MPLBACKEND=Agg MPLCONFIGDIR=/workspace/.cache/matplotlib JAX_PLATFORMS=cpu','export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1','  .venv/bin/python -m scripts.channel_chain --workers 2','```','','The run is restartable: existing geometry is checked, reference caches bind geometry and Re, and failed FEM cases are recorded explicitly. `--stage prepare`, `--stage references`, and `--stage render` can be run separately. Reproduction uses the pinned CPU environment documented at the repository root.','']
    for r in records:
        d=r['directory'];md += [f'<a id="{d}"></a>',f"## {r['index']:02d}. {r['title']}",'',f'[1 Geometry]({d}/01_geometry.png) → [2 Model]({d}/02_model.png) → [3 FEM]({d}/03_fem.png) → [4 Error]({d}/04_error.png) · [Metrics]({d}/metrics.json) · [Fields]({d}/fields.npz) · [FEM reference]({d}/reference.npz)','',f'![Complete result chain]({d}/chain.png)','']
    (output/'README.md').write_text('\n'.join(md))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=ROOT/'reports/channel_chain_15');p.add_argument('--checkpoint',type=Path,default=ROOT/'artifacts/kaggle/main-v1/checkpoint.npz');p.add_argument('--workers',type=int,default=2);p.add_argument('--stage',choices=['all','prepare','references','render'],default='all');a=p.parse_args()
    a.output=a.output.resolve();a.checkpoint=a.checkpoint.resolve()
    if a.workers<1:p.error('--workers must be positive')
    manifest=prepare(a.output,a.checkpoint)
    if a.stage in ['all','references']:references(a.output,manifest,a.workers)
    if a.stage in ['all','render']:render(a.output,manifest,a.checkpoint)


if __name__=='__main__':main()
