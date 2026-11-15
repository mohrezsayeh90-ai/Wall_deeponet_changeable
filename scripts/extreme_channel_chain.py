"""Five extreme exact-wall flow comparisons, entirely on local CPU."""
import json,hashlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scripts.channel_chain import references,render
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/extreme_channel_chain_5'
SEED=2026111517


def prepare():
    from geometry_operator.geometry import BREAKS,_canonical,descriptors,validate,sample_numpy,geometry_id
    from scipy.ndimage import gaussian_filter1d
    rng=np.random.default_rng(SEED)
    knots=np.r_[np.zeros(6),BREAKS[1:-1],np.full(6,5.)]
    x=np.array([np.mean(knots[i+1:i+6]) for i in range(len(knots)-6)])
    z=np.clip((x-.5)/4,0,1);env=np.sin(np.pi*z)**4
    plateau=.5*(np.tanh((x-1.25)/.45)-np.tanh((x-3.75)/.45))
    bottom=gaussian_filter1d(rng.normal(size=len(x)),1.4)*env
    top=gaussian_filter1d(rng.normal(size=len(x)),1.7)*env
    specs=[('hourglass','Severely pinched hourglass',np.zeros_like(x),.70*env*np.cos(2*np.pi*z)),
           ('serpentine','Strong serpentine displacement',.65*env*np.sin(2*np.pi*z),-.16*env*np.cos(4*np.pi*z)),
           ('interlocking','Interlocking rounded wall teeth',.32*env*np.sin(6*np.pi*z),-.42*env),
           ('wide_cavity','Large asymmetric plateau cavity',.34*plateau,.68*plateau),
           ('rough_multiscale','Independent rough multiscale walls',.7*(bottom+top)/2,.7*(top-bottom))]
    OUT.mkdir(exist_ok=True);cases=[]
    for i,(slug,title,c,g) in enumerate(specs,1):
        coeff=_canonical([c,g]);d=descriptors(coeff);cc,gg=sample_numpy(coeff,np.linspace(0,5,4001))
        scale=min(.579/max(-gg.min(),1e-12),.699/max(gg.max(),1e-12),.449/max(abs(cc).max(),1e-12),1.099/max(d['max_slope'],1e-12))
        coeff*=scale;desc=validate(coeff,stress=True)
        try:validate(coeff);domain='within_training_geometry_envelope'
        except ValueError:domain='outside_training_geometry_envelope_stress_test'
        case=OUT/f'{i:02d}_{slug}';case.mkdir(exist_ok=True);xx=np.linspace(0,5,1001);cc,gg=sample_numpy(coeff,xx)
        path=case/'geometry.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as old:np.testing.assert_array_equal(old['coeff'],coeff)
        else:np.savez_compressed(path,coeff=coeff,x=xx,bottom=cc-gg/2,top=1+cc+gg/2)
        cases.append(dict(index=i,slug=slug,title=title,directory=case.name,reynolds=30.,geometry_id=geometry_id(coeff),descriptors=desc,amplitude_scale=scale,domain_status=domain))
    cp=ROOT/'artifacts/kaggle/main-v1/checkpoint.npz';manifest=dict(seed=SEED,cases=cases,case_count=5,checkpoint_step=32216,checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),reynolds=30.,selection='Prescribed extreme mechanisms, radially scaled to the declared stress envelope before error evaluation.',kaggle_accessed=False,locked_test_used=False,created_at_utc=datetime.now(timezone.utc).isoformat())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest,cp


def summarize(output,manifest,records,plt,colors):
    import csv
    for r in records:
        with np.load(output/r['directory']/'fields.npz',allow_pickle=False) as data:
            w=data['weights'];error=data['predicted']-data['fem'];f=data['fem']
            ev=np.linalg.norm(error[:,:2],axis=1);fv=np.linalg.norm(f[:,:2],axis=1)
            r['metrics'].update(velocity_mae=float(w@ev/w.sum()),velocity_relative_l1=float(w@ev/max(w@fv,1e-16)),pressure_mae=float(w@abs(error[:,2])/w.sum()),pressure_relative_l1=float(w@abs(error[:,2])/max(w@abs(f[:,2]),1e-16)))
        (output/r['directory']/'metrics.json').write_text(json.dumps(r,indent=2)+'\n')
    result=dict(expected=5,passed=len(records),failed=5-len(records),checkpoint_step=32216,checkpoint_sha256=manifest['checkpoint_sha256'],reynolds=30.,kaggle_accessed=False,new_training_performed=False,locked_test_used=False,accepted_model=False,cases=records)
    (output/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['case','geometry','domain_status','velocity_relative_l2','pressure_relative_l2','pressure_drop_relative','max_mesh_discrepancy','common_area_excluded_fraction'])
        for r in records:w.writerow([r['index'],r['title'],r['domain_status'],*[r['metrics'][k] for k in ['velocity_relative_l2','pressure_relative_l2','pressure_drop_relative']],max(r['mesh_uncertainty'].values()),r['metrics']['common_domain_excluded_area_fraction']])
    fig,axes=plt.subplots(3,2,figsize=(14,10),layout='constrained')
    for ax,r in zip(axes.flat,records):
        with np.load(output/r['directory']/'geometry.npz',allow_pickle=False) as d:
            ax.fill_between(d['x'],d['bottom'],d['top'],color='#ffedd5');ax.plot(d['x'],d['bottom'],color='#9a3412');ax.plot(d['x'],d['top'],color='#9a3412')
        ax.set(xlim=(0,5),ylim=(-.5,1.75),aspect='equal');ax.set_title(f"{r['index']:02d}. {r['title']}")
    axes.flat[-1].axis('off');fig.suptitle('Extreme wall geometries | stress-envelope inputs | Re = 30',fontsize=17);fig.savefig(output/'geometry_overview.png',dpi=170);plt.close(fig)
    lines=['# Five extreme local-CPU channel comparisons','', '**Kaggle was not accessed.** These calculations use the existing **step-32,216 checkpoint** in the repository, not the ongoing continuation. All cases have Re=30. No training is performed.','', '[Five-page result PDF](all_5_chains.pdf) · [Numerical CSV](summary.csv) · [Full metrics and FEM metadata](summary.json)','', '![Extreme geometries](geometry_overview.png)','', '## Domain limits and interpretation','','The training geometry envelope is gap 0.60–1.50, centerline deviation ≤0.35 and wall slope ≤0.70. The supported stress-input envelope is gap 0.42–1.70, centerline deviation ≤0.45 and slope ≤1.10. Shapes are designed first and radially scaled until reaching a stress-envelope boundary. Each case is labeled against the training envelope below. Stress permission only permits an extrapolation query; it does **not** establish that the checkpoint was trained on these shapes or is accurate there. The channel remains smooth C4, nonbranching, with flat inlet/outlet strips; overhangs, sharp corners and obstacles are not represented.','', 'FEM is independent P2/P1, checked with coarse/fine meshes and automatic refinement until velocity, pressure and pressure-drop discrepancies are ≤1%. The exact-wall/FEM common-domain area excluded must be ≤0.5%. Errors are physical-area weighted and pressure is not shifted to fit the reference. The [original method definitions](../channel_chain_15/README.md) apply. Model/FEM colors match per case and quantity; absolute-error panels use separate scales. The default API rejects out-of-envelope queries; this experiment explicitly enables the documented stress path. All numerical arrays, signed errors and exact geometries are retained.','', '| Case | Domain | Min/max gap | Max slope | Velocity L2 | Pressure L2 | Drop error | Max mesh discrepancy |','| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for r in records:
        m=r['metrics'];d=r['descriptors'];domain='Extrapolation' if r['domain_status'].startswith('outside') else 'Within training envelope'
        lines.append(f"| {r['index']:02d} {r['title']} | {domain} | {d['min_gap']:.3f}/{d['max_gap']:.3f} | {d['max_slope']:.3f} | {m['velocity_relative_l2']:.2%} | {m['pressure_relative_l2']:.2%} | {m['pressure_drop_relative']:.2%} | {max(r['mesh_uncertainty'].values()):.2%} |")
    lines+=['','## Area-weighted mean errors','','Absolute MAE = sum(w |error|)/sum(w); relative L1 = sum(w |error|)/sum(w |FEM|), with a 1e-16 denominator floor. Velocity uses the Euclidean vector magnitude. MAE is in nondimensional field units, not percent.','', '| Case | Velocity MAE | Velocity relative L1 | Pressure MAE | Pressure relative L1 |','| --- | ---: | ---: | ---: | ---: |']
    for r in records:
        m=r['metrics'];lines.append(f"| {r['index']:02d} | {m['velocity_mae']:.5f} | {m['velocity_relative_l1']:.2%} | {m['pressure_mae']:.5f} | {m['pressure_relative_l1']:.2%} |")
    lines+=['','Reproduce locally: `python -m scripts.extreme_channel_chain` in the documented CPU environment. Reference caches bind geometry and Re. Out-of-domain results are stress experiments and do not grant model acceptance.','']
    for r in records:
        d=r['directory'];lines += [f"## {r['index']:02d}. {r['title']}",'',f'[Geometry]({d}/01_geometry.png) → [Model]({d}/02_model.png) → [FEM]({d}/03_fem.png) → [Error]({d}/04_error.png) · [Metrics]({d}/metrics.json) · [Fields]({d}/fields.npz)','',f'![Comparison chain]({d}/chain.png)','']
    (output/'README.md').write_text('\n'.join(lines))


def main():
    manifest,cp=prepare();references(OUT,manifest,2);render(OUT,manifest,cp,summary_function=summarize,pdf_name='all_5_chains.pdf',allow_stress=True)


if __name__=='__main__':main()
