"""Plot saved 15-channel errors in their prescribed simple-to-complex order."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'reports/channel_chain_15'
OUTPUT=ROOT/'reports/complexity_error_curves_15'


def main():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    original=json.loads((SOURCE/'summary.json').read_text())
    cases=sorted(original['cases'],key=lambda r:r['index'])
    assert [r['index'] for r in cases]==list(range(1,16))
    rows=[];hashes={}
    for case in cases:
        path=SOURCE/case['directory']/'fields.npz'
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as d:
            w=d['weights'];truth=d['fem'];error=d['predicted']-truth
            assert np.all(w>0) and np.isfinite(error).all()
            energy=float(w@np.sum(truth[:,:2]**2,axis=1))
            urms=float(np.sqrt(w@(error[:,0]**2)/w.sum()))
            vrms=float(np.sqrt(w@(error[:,1]**2)/w.sum()))
            prms=float(np.sqrt(w@(error[:,2]**2)/w.sum()))
            u=float(np.sqrt(w@(error[:,0]**2)/max(float(w@(truth[:,0]**2)),1e-16)))
            p=float(np.sqrt(w@(error[:,2]**2)/max(float(w@(truth[:,2]**2)),1e-16)))
            vector=float(np.sqrt(w@np.sum(error[:,:2]**2,axis=1)/energy))
            uc=float(np.sqrt(w@(error[:,0]**2)/energy));vc=float(np.sqrt(w@(error[:,1]**2)/energy))
            np.testing.assert_allclose([u,vrms,p,vector],[case['metrics'][k] for k in ['u_relative_l2','v_absolute_rms','pressure_relative_l2','velocity_relative_l2']],rtol=1e-12)
            np.testing.assert_allclose(uc*uc+vc*vc,vector*vector,rtol=1e-12)
            assert str(d['checkpoint_sha256'])==original['checkpoint_sha256']
        rows.append(dict(case=case['index'],title=case['title'],reynolds=case['reynolds'],u_relative_l2_percent=100*u,v_absolute_rms=vrms,p_relative_l2_percent=100*p,velocity_relative_l2_percent=100*vector,speed_relative_l2_percent=100*case['metrics']['speed_relative_l2'],pressure_drop_relative_percent=100*case['metrics']['pressure_drop_relative'],u_absolute_rms=urms,p_absolute_rms=prms,u_error_over_reference_velocity_norm_percent=100*uc,v_error_over_reference_velocity_norm_percent=100*vc))
    OUTPUT.mkdir(exist_ok=True)
    with (OUTPUT/'errors_by_case.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (OUTPUT/'data.json').write_text(json.dumps(dict(checkpoint_step=32216,checkpoint_sha256=original['checkpoint_sha256'],reynolds=30.,case_order='Original prescribed qualitative order, not a calibrated complexity score; no reordering by errors.',source_field_sha256=hashes,all_saved_metrics_verified=True,cases=rows),indent=2)+'\n')
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    xx=np.arange(1,16)
    def panel(ax,key,title,ylabel,color):
        yy=[r[key] for r in rows]
        ax.plot(xx,yy,'o-',color=color,lw=1.8,ms=5)
        ax.set(title=title,xlabel='Case order: simpler to more structured',ylabel=ylabel,xticks=xx,xlim=(.6,15.4),ylim=(0,max(yy)*1.18));ax.grid(alpha=.22)
        i=int(np.argmax(yy));ax.annotate(f'{yy[i]:.3g}',(xx[i],yy[i]),xytext=(0,8),textcoords='offset points',ha='center',fontsize=9)
    panels=[('u_relative_l2_percent','u: axial velocity','Relative L2 error (%)','#075985'),('v_absolute_rms','v: transverse velocity','Absolute RMS error (nondimensional)','#7c3aed'),('p_relative_l2_percent','p: pressure','Relative L2 error (%)','#b45309'),('velocity_relative_l2_percent','Full velocity vector (u,v)','Relative vector L2 error (%)','#0f766e'),('speed_relative_l2_percent','Speed magnitude','Relative L2 error (%)','#be123c'),('pressure_drop_relative_percent','Pressure drop','Relative pressure-drop error (%)','#334155')]
    fig,axes=plt.subplots(2,3,figsize=(17,9),layout='constrained')
    for ax,(key,title,label,color) in zip(axes.flat,panels):panel(ax,key,title,label,color)
    fig.suptitle('Errors across 15 prescribed channels | frozen step 32,216 | Re = 30',fontsize=17,fontweight='bold')
    fig.savefig(OUTPUT/'error_dashboard.png',dpi=170);fig.savefig(OUTPUT/'error_dashboard.pdf');plt.close(fig)
    for slug,definition in zip(['u_error','v_error','p_error'],panels[:3]):
        fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained');panel(ax,*definition)
        fig.suptitle('15 saved geometry/FEM comparisons | Re = 30 | same frozen checkpoint',fontsize=13)
        fig.savefig(OUTPUT/f'{slug}.png',dpi=170);fig.savefig(OUTPUT/f'{slug}.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,5.5),layout='constrained')
    for key,color,label in [('u_error_over_reference_velocity_norm_percent','#075985','u error / reference velocity-vector norm'),('v_error_over_reference_velocity_norm_percent','#7c3aed','v error / reference velocity-vector norm'),('velocity_relative_l2_percent','#0f766e','Total velocity-vector error')]:
        axes[0].plot(xx,[r[key] for r in rows],'o-',color=color,label=label)
    axes[0].set(xticks=xx,xlabel='Prescribed case order',ylabel='Error / reference vector L2 norm (%)',title='u and v with a shared reference scale');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    for key,color,label in [('u_absolute_rms','#075985','u absolute RMS'),('v_absolute_rms','#7c3aed','v absolute RMS')]:axes[1].plot(xx,[r[key] for r in rows],'o-',color=color,label=label)
    axes[1].set(xticks=xx,xlabel='Prescribed case order',ylabel='Absolute RMS velocity error (nondimensional)',title='Absolute component errors');axes[1].legend();axes[1].grid(alpha=.2)
    fig.suptitle('Velocity components: comparison without dividing by nearly zero reference v',fontsize=14)
    fig.savefig(OUTPUT/'velocity_components.png',dpi=170);fig.savefig(OUTPUT/'velocity_components.pdf');plt.close(fig)
    lines=['# Error versus prescribed channel complexity order','', '[Six-panel PDF](error_dashboard.pdf) · [Numerical CSV](errors_by_case.csv) · [Data, source hashes and validation](data.json) · [Original geometry gallery](../channel_chain_15/README.md)','', '![Errors across the 15 cases](error_dashboard.png)','', 'The x-axis preserves the original 01–15 simple-to-structurally-complex design order. It is a **qualitative ordering**, not a calibrated or strictly increasing geometric-complexity score. Markers are actual cases; connecting lines only guide the eye. No fit or smoothing is applied, and cases were not reordered by error. The curves do not establish that complexity alone causes error or that error must rise monotonically. Geometry type and deformation magnitude also vary.','', 'All data come from the already saved FEM/model fields and metrics at **Re=30**, using the **original step-32,216 checkpoint**. No training, prediction or FEM solve was rerun; these are not results from the ongoing continuation. All plotted u/v/p and vector metrics were independently recomputed from saved fields and checked against the original reports. Original 15-case outputs remain unchanged.','', '## Individual u, v and p curves','', '[u axial velocity](u_error.png) ([PDF](u_error.pdf)) · [v transverse velocity](v_error.png) ([PDF](v_error.pdf)) · [p pressure](p_error.png) ([PDF](p_error.pdf))','', 'u, p, full velocity and speed use physical-area-weighted **relative L2**. For a field f this is sqrt(sum(w |model−FEM|²) / max(sum(w |FEM|²), 1e−16)); velocity uses the two-component vector norm. Pressure drop uses the original section-average drop comparison with its 1e−8 denominator floor.','', '**v is plotted as absolute RMS**, sqrt(sum(w (v_model−v_FEM)²)/sum(w)), because reference v vanishes in the straight-channel case and a relative v error would be ill-conditioned. v RMS is not a percentage and cannot be compared numerically to the percent axes in the other panels.','', '## Direct comparison of velocity components','', '![Velocity component errors](velocity_components.png)','', 'For a common percent scale, the left panel divides both component error norms by the **full FEM velocity-vector norm**, not by each component norm. Thus u_contribution² + v_contribution² = total_vector_error². The right panel compares absolute u/v RMS in the same velocity units. These common-scale quantities are named explicitly in the CSV and differ from relative u or relative v error. All quantities are nondimensional and use the original common physical quadrature domain.','', '| Case | Geometry | u relative L2 | v absolute RMS | p relative L2 | Vector relative L2 |','| --- | --- | ---: | ---: | ---: | ---: |']
    for r in rows:lines.append(f"| {r['case']:02d} | {r['title']} | {r['u_relative_l2_percent']:.2f}% | {r['v_absolute_rms']:.5f} | {r['p_relative_l2_percent']:.2f}% | {r['velocity_relative_l2_percent']:.2f}% |")
    lines+=['','Reproduce: `MPLBACKEND=Agg MPLCONFIGDIR=/workspace/.cache/matplotlib XDG_CACHE_HOME=/workspace/.cache .venv/bin/python -m scripts.plot_complexity_errors`.','']
    (OUTPUT/'README.md').write_text('\n'.join(lines))
    print('Verified and plotted all 15 cases; no training, inference or FEM solves performed.')


if __name__=='__main__':main()
