"""Paired Reynolds-number comparison for the same saved two-step geometry."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from geometry_operator.geometry import validate
from geometry_operator.evaluate import weighted_relative

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/double_step_re_comparison_82216'


def main():
    cases={50:ROOT/'reports/double_step_re50_82216',100:ROOT/'reports/double_step_82216'}
    OUT.mkdir(parents=True,exist_ok=True)
    summaries={re:json.loads((p/'summary.json').read_text()) for re,p in cases.items()}
    data={}
    for re,path in cases.items():
        with np.load(path/'fields.npz',allow_pickle=False) as f:data[re]={key:f[key] for key in f.files}
        validate(data[re]['coeff'])
        assert float(data[re]['reynolds'])==re
        s=summaries[re]
        assert hashlib.sha256((ROOT/s['checkpoint']).read_bytes()).hexdigest()==s['checkpoint_sha256']
        assert max(s['fem_reference']['mesh_uncertainty'].values())<=.01
        f=data[re];w=f['quadrature_weights'];assert (w>0).all()
        assert abs(weighted_relative(f['quadrature_model'][:,:2],f['quadrature_fem'][:,:2],w)-s['metrics']['velocity_relative_l2'])<1e-12
        for key in ['model','fem']:assert np.isfinite(f[key][f['mask']]).all()
    for key in ['coeff','x','y','bottom','top','mask','quadrature_xy','quadrature_weights']:
        np.testing.assert_array_equal(data[50][key],data[100][key])
    assert summaries[50]['checkpoint_sha256']==summaries[100]['checkpoint_sha256']
    min_u=min(float(np.nanmin(data[re][key][:,:,0])) for re in cases for key in ['model','fem'])
    norm=TwoSlopeNorm(vmin=min(-.005,min_u),vcenter=0,vmax=.3)
    fig,axes=plt.subplots(2,2,figsize=(17,6),layout='constrained')
    for col,re in enumerate([50,100]):
        f=data[re];x,y,b,t=f['x'],f['y'],f['bottom'],f['top']
        for row,key in enumerate(['model','fem']):
            ax=axes[row,col];field=f[key]
            u=np.ma.masked_invalid(field[:,:,0]);v=np.ma.masked_invalid(field[:,:,1])
            pc=ax.pcolormesh(x,y,u,cmap='coolwarm',norm=norm,shading='auto',rasterized=True)
            ax.streamplot(x,y,u,v,color='#172554',density=(7,6),linewidth=.48,arrowsize=.7)
            if np.nanmin(u)<-.001:ax.contour(x,y,u,levels=[-.001],colors=['#dc2626'],linewidths=1.3)
            ax.plot(x,b,'k',lw=1.5);ax.fill_between(x,-.02,b,color='#cbd5e1',zorder=4)
            ax.set(xlim=(1.4,3.65),ylim=(-.01,.5),aspect='equal',xlabel='x',ylabel='y',title=f"{'PINN / DeepONet' if key=='model' else 'Independent FEM'} | Re = {re}")
    fig.colorbar(pc,ax=axes.ravel().tolist(),label='Axial velocity u (blue = reverse flow); common scale',shrink=.7,extend='max')
    fig.suptitle('Same two-step geometry and Adam checkpoint 82,216 | Re = 50 versus Re = 100\nIdentical sampling, limits and color scale; red contours: u = -0.001',fontsize=15)
    fig.savefig(OUT/'re50_vs_re100.png',dpi=200);fig.savefig(OUT/'re50_vs_re100.pdf');plt.close(fig)
    rows=[]
    for re in [50,100]:
        s=summaries[re];r=s['reverse_flow']
        rows.append({'reynolds':re,**s['metrics'],'model_min_u_inter_step':r['model']['minimum_u_in_inter_step_region'],
                     'fem_min_u_inter_step':r['fem']['minimum_u_in_inter_step_region'],
                     'model_reverse_area_estimate':r['model']['approx_reversed_area_in_diagnostic_region'],
                     'fem_reverse_area_estimate':r['fem']['approx_reversed_area_in_diagnostic_region'],
                     'max_fem_mesh_discrepancy':max(s['fem_reference']['mesh_uncertainty'].values())})
    with (OUT/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (OUT/'comparison.json').write_text(json.dumps({'checkpoint_step':82216,'checkpoint_sha256':summaries[50]['checkpoint_sha256'],
        'same_geometry_and_all_sampling_arrays_verified':True,'training_performed':False,'cases':rows,'case_summaries':summaries},indent=2)+'\n')
    md=['# Same two-step channel at Re=50 and Re=100','',
        'Frozen Adam step **82,216**. No retraining. Exact geometry coefficients, plot grid, mask, FEM quadrature points and physical weights are **array-identical** across Reynolds numbers. All normal geometry limits pass. The comparison figure uses common axes and axial-velocity color limits.','',
        '[Comparison PNG](re50_vs_re100.png) · [PDF](re50_vs_re100.pdf) · [CSV](comparison.csv) · [Complete comparison ZIP](double_step_re50_vs_re100.zip) · [Re=50 case](../double_step_re50_82216/README.md) · [Re=100 case](../double_step_82216/README.md)','',
        '![Reynolds-number comparison](re50_vs_re100.png)','',
        '| Re | Velocity relative L2 | Pressure relative L2 | Model min inter-step u | FEM min inter-step u | Max FEM mesh discrepancy |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        md.append(f"| {r['reynolds']} | {r['velocity_relative_l2']:.2%} | {r['pressure_relative_l2']:.2%} | {r['model_min_u_inter_step']:.5f} | {r['fem_min_u_inter_step']:.5f} | {r['max_fem_mesh_discrepancy']:.2%} |")
    a,b=rows
    md+=['',f"Velocity error changes from {b['velocity_relative_l2']:.2%} at Re=100 to {a['velocity_relative_l2']:.2%} at Re=50. Pressure error changes from {b['pressure_relative_l2']:.2%} to {a['pressure_relative_l2']:.2%}.",'',
         'Reverse-flow diagnostics use 1.9<x<3.1, bottom(x)+0.02<y<0.36 and threshold u<-0.001. The excluded wall band avoids treating tiny near-wall signs as a resolved bubble. The two FEM solves are independent and pass the existing <=1% global mesh checks; these do not certify local wall shear or exact reattachment lengths. Negative-u contours alone do not certify closed streamlines. All quantities are nondimensional; pressure is compared without a fitted gauge shift.','',
         'FEM shows a smaller and weaker reverse-flow region at Re=50; the stored model misses inter-step reversal at both Reynolds numbers. The minimum FEM u in the diagnostic region changes from approximately -0.05770 at Re=100 to -0.00534 at Re=50.','',
         'A permitted geometry does not guarantee accurate separation prediction. This paired experiment tests the effect of changing Re for one geometry and one checkpoint; it does not establish a general error trend across all channels.','',
         'Reproduce Re=50: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_ENABLE_X64=true .venv/bin/python -m scripts.double_step_flow --re 50`; then `.venv/bin/python -m scripts.compare_double_step_re`. The ZIP includes this comparison and both complete case folders, excluding nested ZIPs and model weights.','']
    (OUT/'README.md').write_text('\n'.join(md))
    with zipfile.ZipFile(OUT/'double_step_re50_vs_re100.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for folder in [OUT,*cases.values()]:
            for p in sorted(folder.iterdir()):
                if p.is_file() and p.suffix!='.zip':z.write(p,arcname=folder.name+'/'+p.name)
    with zipfile.ZipFile(OUT/'double_step_re50_vs_re100.zip') as z:
        assert z.testzip() is None
        for name in z.namelist():assert z.read(name)==(ROOT/'reports'/name).read_bytes()
    print(json.dumps(rows,indent=2))

if __name__=='__main__':main()
