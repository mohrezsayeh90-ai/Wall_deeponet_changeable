"""Five seeded, visually distinct exact-wall inputs and independent flow checks."""
import csv
import hashlib
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from scripts.channel_chain import references,render
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/random_channel_chain_5'
SEED=2026111505


def prepare():
    from geometry_operator.geometry import BREAKS,_canonical,descriptors,validate,sample_numpy,geometry_id
    from scipy.ndimage import gaussian_filter1d
    rng=np.random.default_rng(SEED)
    knots=np.r_[np.zeros(6),BREAKS[1:-1],np.full(6,5.)]
    x=np.array([np.mean(knots[i+1:i+6]) for i in range(len(knots)-6)])
    z=np.clip((x-.5)/4,0,1);env=np.sin(np.pi*z)**4
    plateau=.5*(np.tanh((x-rng.uniform(1.1,1.4))/.27)-np.tanh((x-rng.uniform(3.6,3.9))/.35))
    chirp=env*np.sin(2*np.pi*(rng.uniform(.7,1.1)*z+2.7*z*z)+rng.uniform(0,6.28))
    step_rng=np.random.default_rng(SEED+3)
    terrace=env*(.65*np.tanh((x-step_rng.uniform(1.5,1.9))/.22)-.45*np.tanh((x-step_rng.uniform(3.,3.4))/.25))
    random_bottom=env*gaussian_filter1d(rng.normal(size=len(x)),2.0)
    random_top=env*gaussian_filter1d(rng.normal(size=len(x)),2.8)
    def fourier():
        return env*sum(rng.normal()/k*np.sin(2*np.pi*k*z+rng.uniform(0,2*np.pi)) for k in range(1,5))
    specs=[('roof_plateau','Rounded plateau roof',np.zeros_like(x),.28*plateau),
           ('chirped_wave','Waves with changing wavelength',.14*chirp,-.11*env*np.sin(2*np.pi*(1.1*z+2.3*z*z)+.6)),
           ('stepped_s_bend','Rounded stepped S-shaped walls',.22*terrace-.10*env,-.16*terrace+.10*env),
           ('random_spline','Independent irregular spline walls',.48*random_bottom,.48*random_top),
           ('random_fourier','Independent random Fourier walls',.16*fourier(),.16*fourier())]
    OUT.mkdir(exist_ok=True);records=[]
    for i,(slug,title,bottom,top) in enumerate(specs,1):
        coeff=_canonical([(bottom+top)/2,top-bottom]);d=descriptors(coeff);cc,gg=sample_numpy(coeff,np.linspace(0,5,4001))
        scale=min(1.,.399/max(-gg.min(),1e-12),.499/max(gg.max(),1e-12),.349/max(abs(cc).max(),1e-12),.699/max(d['max_slope'],1e-12));coeff*=scale
        desc=validate(coeff);case=OUT/f'{i:02d}_{slug}';case.mkdir(exist_ok=True)
        xx=np.linspace(0,5,1001);cc,gg=sample_numpy(coeff,xx)
        path=case/'geometry.npz'
        if path.exists():
            with np.load(path,allow_pickle=False) as old:np.testing.assert_array_equal(old['coeff'],coeff)
        else:np.savez_compressed(path,coeff=coeff,x=xx,bottom=cc-gg/2,top=1+cc+gg/2)
        records.append(dict(index=i,slug=slug,title=title,directory=case.name,reynolds=30.,geometry_id=geometry_id(coeff),descriptors=desc,amplitude_scale=scale))
    cp=ROOT/'artifacts/kaggle/main-v1/checkpoint.npz'
    manifest=dict(seed=SEED,cases=records,case_count=5,checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),selection='Seeded shape mechanisms, fixed before FEM/model evaluation; not selected by low prediction error.',checkpoint_step=32216,reynolds=30.,locked_test_used=False,created_at_utc=datetime.now(timezone.utc).isoformat())
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');return manifest,cp


def summarize(output,manifest,records,plt,colors):
    summary=dict(expected=5,passed=len(records),failed=5-len(records),checkpoint_sha256=manifest['checkpoint_sha256'],checkpoint_step=32216,reynolds=30.,seed=SEED,accepted_model=False,locked_test_used=False,all_fem_mesh_checks_passed=all(max(r['mesh_uncertainty'].values())<=.01 for r in records),cases=records)
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    with (output/'summary.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['case','title','velocity_relative_l2','pressure_relative_l2','pressure_drop_relative','max_fem_mesh_discrepancy'])
        for r in records:w.writerow([r['index'],r['title'],*[r['metrics'][k] for k in ['velocity_relative_l2','pressure_relative_l2','pressure_drop_relative']],max(r['mesh_uncertainty'].values())])
    fig,axes=plt.subplots(3,2,figsize=(13,9),layout='constrained')
    for ax,r in zip(axes.flat,records):
        with np.load(output/r['directory']/'geometry.npz',allow_pickle=False) as d:
            ax.fill_between(d['x'],d['bottom'],d['top'],color='#dbeafe');ax.plot(d['x'],d['bottom'],color='#075985');ax.plot(d['x'],d['top'],color='#075985')
        ax.set(xlim=(0,5),ylim=(-.48,1.48),aspect='equal');ax.set_title(f"{r['index']:02d}. {r['title']}")
    axes.flat[-1].axis('off');fig.suptitle('Five distinct wall-shape mechanisms | Re = 30 | seeded geometry inputs',fontsize=16)
    fig.savefig(output/'geometry_overview.png',dpi=170);plt.close(fig)
    md=['# Five varied random channel inputs and flow predictions','', 'The flow operator predicts velocity and pressure **given wall geometry**; it is not a geometry generator. The previous 15 shapes used repeated compact bump functions by design. This set uses rounded plateaus, chirped sinusoidal waves, rounded steps, smoothed random spline coefficients and independent Fourier mixtures instead. All use the exact canonical C4 quintic wall representation, flat inlet/outlet strips and the declared gap/centerline/slope envelope. Sharp corners, discontinuous steps, branching channels and obstacles are not supported by this model.','',f'Random seed: `{SEED}`. Parameters and shapes were fixed before evaluation. These are illustrative custom inputs, not a certified held-out benchmark or an exhaustive random sample. All cases use **Re=30** and the **original step-32,216 research checkpoint**. The ongoing continuation is not used. Input validity does not guarantee accurate predictions; the FEM comparisons below determine the observed errors.','', '[All five result chains PDF](all_5_chains.pdf) · [CSV errors](summary.csv) · [Full metrics JSON](summary.json)','', '![Five geometries](geometry_overview.png)','', '| Case | Velocity L2 error | Pressure L2 error | Pressure-drop error | Max mesh discrepancy |','| --- | ---: | ---: | ---: | ---: |']
    for r in records:
        m=r['metrics'];md.append(f"| {r['index']:02d} {r['title']} | {m['velocity_relative_l2']:.2%} | {m['pressure_relative_l2']:.2%} | {m['pressure_drop_relative']:.2%} | {max(r['mesh_uncertainty'].values()):.2%} |")
    md+=['','## Method','','The same independent P2/P1 FEM and error protocol as the [15-case report](../channel_chain_15/README.md) is used: coarse/fine mesh discrepancies ≤1% for velocity, pressure and pressure drop; common-domain excluded area ≤0.5%; positive physical-area quadrature weights; pressure compared without fitted gauge shift. All quantities are nondimensional. Model/FEM color limits match per case; absolute-error scales differ per panel. Field maxima refer to saved quadrature samples. Errors and signed fields, exact geometry coefficients, references and metadata are retained. The checkpoint remains unaccepted, and the locked test is not opened.','','Reproduce: `python -m scripts.random_channel_chain` in the documented CPU environment. Cached references bind geometry and Re. The original 15-case output files are retained unchanged.','']
    for r in records:
        d=r['directory'];md += [f"## {r['index']:02d}. {r['title']}",'',f'[Geometry]({d}/01_geometry.png) → [Model]({d}/02_model.png) → [FEM]({d}/03_fem.png) → [Error]({d}/04_error.png) · [Metrics]({d}/metrics.json) · [Fields]({d}/fields.npz)','',f'![Complete result chain]({d}/chain.png)','']
    (output/'README.md').write_text('\n'.join(md))


def main():
    manifest,cp=prepare();references(OUT,manifest,2);render(OUT,manifest,cp,summary_function=summarize,pdf_name='all_5_chains.pdf')


if __name__=='__main__':main()
