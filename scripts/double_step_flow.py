"""Two smooth bottom-wall steps: frozen operator, independent FEM, dense streamlines."""
import json
from pathlib import Path
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from geometry_operator.geometry import BREAKS, _canonical, sample_numpy, validate, geometry_id
from geometry_operator.api import FlowOperator
from geometry_operator.reference import build_case, solve_reference
from geometry_operator.evaluate import weighted_relative

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/double_step_82216'


def render():
    with np.load(OUT/'fields.npz',allow_pickle=False) as f:
        x,y,b,t=f['x'],f['y'],f['bottom'],f['top']
        fields=[f['model'],f['fem']]
    plt.rcParams.update({'font.size':11,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    for zoom,filename in [(False,'streamlines'),(True,'between_steps_zoom')]:
        fig,axes=plt.subplots(2,1,figsize=(13,8 if zoom else 7),layout='constrained')
        vmax=max(float(np.nanmax(np.linalg.norm(f[:,:,:2],axis=2))) for f in fields)
        vmin=min(-.005,min(float(np.nanmin(f[:,:,0])) for f in fields))
        for ax,field,label in zip(axes,fields,['Frozen PINN / DeepONet - Adam 82,216','Independent P2/P1 FEM']):
            u=np.ma.masked_invalid(field[:,:,0]);v=np.ma.masked_invalid(field[:,:,1])
            if zoom:
                pc=ax.pcolormesh(x,y,u,cmap='coolwarm',norm=TwoSlopeNorm(vmin=vmin,vcenter=0,vmax=.3),shading='auto',rasterized=True)
            else:
                pc=ax.pcolormesh(x,y,np.linalg.norm(field[:,:,:2],axis=2),cmap='viridis',vmin=0,vmax=vmax,shading='auto',rasterized=True)
            ax.streamplot(x,y,u,v,color='#172554' if zoom else 'white',density=(7,6) if zoom else (5,3.5),linewidth=.5,arrowsize=.75)
            if np.nanmin(u)<-.001:ax.contour(x,y,u,levels=[-.001],colors=['#dc2626' if zoom else '#fb7185'],linewidths=1.4)
            ax.plot(x,b,'k',lw=1.8);ax.plot(x,t,'k',lw=1.8)
            ax.fill_between(x,-.025,b,color='#cbd5e1',zorder=4)
            ax.set(xlim=(1.4,3.65) if zoom else (0,5),ylim=(-.01,.5) if zoom else (-.02,1.02),aspect='equal',xlabel='x',ylabel='y',title=label)
            fig.colorbar(pc,ax=ax,label='Axial velocity u; blue = reverse flow' if zoom else 'Speed |U|',shrink=.8,extend='max' if zoom else 'neither')
        fig.suptitle('Two rounded steps on the lower wall | Re = 100 | standard geometry bounds\n'+('Enlarged inter-step region; equal physical x/y scale' if zoom else 'Dense streamlines; pink contour: u = -0.001'),fontsize=14)
        fig.savefig(OUT/(filename+'.png'),dpi=210);fig.savefig(OUT/(filename+'.pdf'));plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,3),layout='constrained')
    ax.fill_between(x,b,t,color='#dbeafe');ax.fill_between(x,0,b,color='#cbd5e1');ax.plot(x,b,'k',lw=2);ax.plot(x,t,'k',lw=2)
    ax.set(xlim=(0,5),ylim=(-.02,1.02),aspect='equal',xlabel='x',ylabel='y',title='Exact smooth two-step geometry | no stress override')
    fig.savefig(OUT/'geometry.png',dpi=200);plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checkpoint=ROOT/'artifacts/kaggle/continue-50k-v1/checkpoint.npz'
    knots=np.r_[np.zeros(6),BREAKS[1:-1],np.full(6,5.)]
    gx=np.array([np.mean(knots[i+1:i+6]) for i in range(len(knots)-6)])
    bottom=sum(.38*.5*(np.tanh((gx-left)/.28)-np.tanh((gx-right)/.28)) for left,right in [(.95,1.85),(3.15,4.05)])
    coeff=_canonical([bottom/2,-bottom]);desc=validate(coeff);re=100.
    np.savez_compressed(OUT/'geometry.npz',coeff=coeff)
    print(json.dumps({'phase':'geometry_validated','descriptors':desc}),flush=True)
    meta=build_case(coeff,re,OUT/'reference.npz',nx=40,ny=12)
    print(json.dumps({'phase':'fem_mesh_passed','metadata':meta}),flush=True)
    model=FlowOperator(checkpoint,allow_unvalidated=True,chunk_size=512)
    assert model.metadata['step']==82216
    with np.load(OUT/'reference.npz',allow_pickle=False) as f:q,w,truth=f['xy'],f['weights'],f['fields']
    pred=model.evaluate(coeff,re,q)
    metrics={key:weighted_relative(pred[:,ids],truth[:,ids],w) for key,ids in [('velocity_relative_l2',slice(0,2)),('pressure_relative_l2',slice(2,3))]}
    fine=meta['fine'];fem,solver=solve_reference(coeff,re,nx=fine['nx'],ny=fine['ny'])
    x=np.linspace(0,5,501);y=np.linspace(-.02,1.02,201)
    c,dg=sample_numpy(coeff,x);b=c-dg/2;t=1+c+dg/2
    xx,yy=np.meshgrid(x,y);xy=np.column_stack([xx.ravel(),yy.ravel()])
    mask=(yy>b[None,:]+1e-6)&(yy<t[None,:]-1e-6)&fem.contains(xy).reshape(xx.shape)
    ii=np.flatnonzero(mask.ravel());mp=np.full((len(xy),3),np.nan);fp=mp.copy()
    mp[ii]=model.evaluate(coeff,re,xy[ii])
    for start in range(0,len(ii),1000):
        jj=ii[start:start+1000];fp[jj]=fem(xy[jj])
    mp=mp.reshape((*xx.shape,3));fp=fp.reshape((*xx.shape,3))
    region=mask&(xx>1.9)&(xx<3.1)&(yy>b[None,:]+.02)&(yy<.36)
    reverse={}
    for name,field in [('model',mp),('fem',fp)]:
        reversed_points=region&(field[:,:,0]<-.001)
        reverse[name]={'minimum_u_in_inter_step_region':float(np.min(field[:,:,0][region])),
                       'inter_step_points_u_below_minus_0_001':int(reversed_points.sum()),
                       'approx_reversed_area_in_diagnostic_region':float(reversed_points.sum()*(x[1]-x[0])*(y[1]-y[0]))}
    np.savez_compressed(OUT/'fields.npz',x=x,y=y,bottom=b,top=t,mask=mask,model=mp,fem=fp,coeff=coeff,
                        quadrature_xy=q,quadrature_weights=w,quadrature_model=pred,quadrature_fem=truth,
                        reynolds=re,checkpoint_sha256=model.sha256)
    summary={'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_step':82216,'checkpoint_sha256':model.sha256,
             'reynolds':re,'geometry_id':geometry_id(coeff),'descriptors':desc,'metrics':metrics,
             'reverse_flow':reverse,'diagnostic_region':'1.9<x<3.1, bottom(x)+0.02<y<0.36; uniform-grid estimate, threshold u<-0.001',
             'fem_reference':meta,'display_fem_solver':solver,'accepted_model':False,'training_performed':False,
             'selection':'Two prescribed rounded plateaus patterned after the user sketch; geometry fixed before solving, no error-based selection.',
             'geometry_scope':'C4 single-valued lower wall, fixed upper wall, standard bounds including flat inlet/outlet; no stress override.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    render()
    finding=('FEM shows a clear inter-step recirculating region, while the stored model misses the reversed flow in the diagnostic region.'
             if reverse['fem']['inter_step_points_u_below_minus_0_001']>0 and reverse['model']['inter_step_points_u_below_minus_0_001']==0
             else 'Compare the signed velocities and streamline patterns below; reversal diagnostics are region-specific.')
    summary['finding']=finding
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    statements=[]
    for name in ['model','fem']:
        r=reverse[name]
        statements.append(f"- {name}: minimum inter-step u = {r['minimum_u_in_inter_step_region']:.6f}; approximate reversed area in the stated diagnostic region = {r['approx_reversed_area_in_diagnostic_region']:.6f}.")
    text=f'''# Two smooth steps on the bottom wall

Two rounded raised floor sections approximate the supplied sketch. Frozen Adam **step 82,216**, **Re=100**, no retraining or Kaggle use. The upper wall is flat; inlet/outlet use the original model boundary conditions. C4 splines and **all standard geometry limits** pass without stress overrides. Sharp vertical steps are outside this model's geometry domain.

[Full flow PNG](streamlines.png) · [Full flow PDF](streamlines.pdf) · [Between-step zoom PNG](between_steps_zoom.png) · [Zoom PDF](between_steps_zoom.pdf) · [Complete ZIP](double_step_82216.zip) · [Fields](fields.npz) · [Metrics/provenance](summary.json)

![Geometry](geometry.png)

Minimum gap **{desc['min_gap']:.4f}**, maximum gap **{desc['max_gap']:.4f}**, maximum slope **{desc['max_slope']:.4f}**, maximum centerline deviation **{desc['center_deviation']:.4f}**. Geometry parameters were fixed before solving.

![Full flow](streamlines.png)

![Inter-step region](between_steps_zoom.png)

Velocity relative L2 error **{metrics['velocity_relative_l2']:.2%}**; pressure relative L2 error **{metrics['pressure_relative_l2']:.2%}**. Physical-area weighted FEM quadrature; no fitted pressure shift. Maximum coarse/fine FEM discrepancy **{max(meta['mesh_uncertainty'].values()):.2%}**; excluded common-domain area **{meta['common_domain_excluded_area_fraction']:.3%}**. Mesh checks address global fields and pressure drop; local wall shear and exact reattachment length are not certified.

## Reverse flow between the steps

**{finding}** Standard input validity does not guarantee an accurate flow prediction.

'''+'\n'.join(statements)+'''

Diagnostics use **1.9<x<3.1**, **bottom(x)+0.02<y<0.36**, and reversal threshold **u<-0.001**. Area is an approximate uniform plot-grid estimate within that region, not the full recirculation-bubble area. Streamlines use evaluated velocity vectors with arrows; red/pink contours delimit u=-0.001. Inspect both streamline direction and the signed axial velocity. A negative-u contour alone does not establish a closed vortex or exact separation/reattachment points. The research checkpoint has no passing full validation benchmark.

Reproduce: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_ENABLE_X64=true .venv/bin/python -m scripts.double_step_flow`. Re-render existing fields without solving: `from scripts.double_step_flow import render; render()`. Complete ZIP contains fields, geometry, FEM quadrature/reference, plots and metadata; weights remain at the recorded checkpoint path and SHA256.
'''
    (OUT/'README.md').write_text(text)
    with zipfile.ZipFile(OUT/'double_step_82216.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.iterdir()):
            if path.suffix!='.zip':z.write(path,arcname=OUT.name+'/'+path.name)
    with zipfile.ZipFile(OUT/'double_step_82216.zip') as z:
        assert z.testzip() is None
        for name in z.namelist():assert z.read(name)==(OUT/Path(name).name).read_bytes()
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
