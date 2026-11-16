"""Bounded rounded backward-facing-step analogue; frozen Adam inference and FEM."""
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

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/rounded_backward_step_82216'



def render_zoom():
    with np.load(OUT/'fields.npz',allow_pickle=False) as f:
        x,y,b,t=f['x'],f['y'],f['bottom'],f['top']
        fields=[f['model'],f['fem']]
    fig,axes=plt.subplots(2,1,figsize=(12,7),layout='constrained')
    for ax,field,label in zip(axes,fields,['PINN / DeepONet','Independent FEM']):
        u=np.ma.masked_invalid(field[:,:,0]);v=np.ma.masked_invalid(field[:,:,1])
        pc=ax.pcolormesh(x,y,u,cmap='coolwarm',norm=TwoSlopeNorm(vmin=-.035,vcenter=0,vmax=.15),shading='auto',rasterized=True)
        ax.streamplot(x,y,u,v,color='#111827',density=(7,6),linewidth=.5,arrowsize=.75)
        ax.contour(x,y,u,levels=[-.001],colors=['#dc2626'],linewidths=1.4)
        ax.plot(x,b,'k',lw=2);ax.fill_between(x,0,b,color='#d1d5db',zorder=4)
        ax.set(xlim=(2.2,3.4),ylim=(0,.27),aspect='equal',xlabel='x',ylabel='y',title=label)
        fig.colorbar(pc,ax=ax,label='Axial velocity u (blue = reversed)',shrink=.8)
    fig.suptitle('Downstream recirculation detail | rounded step | Re = 100 | Adam 82,216')
    fig.savefig(OUT/'recirculation_zoom.png',dpi=200)
    fig.savefig(OUT/'recirculation_zoom.pdf');plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / 'artifacts/kaggle/continue-50k-v1/checkpoint.npz'
    knots = np.r_[np.zeros(6), BREAKS[1:-1], np.full(6, 5.)]
    gx = np.array([np.mean(knots[i+1:i+6]) for i in range(len(knots)-6)])
    # Finite upstream raised floor, smooth descent into full-height channel.
    # Return to zero is mandatory on both flat boundary strips.
    bottom = .35 * .5 * (np.tanh((gx-1.05)/.29)-np.tanh((gx-2.4)/.27))
    coeff = _canonical([bottom/2, -bottom])
    d = validate(coeff)
    re = 100.
    np.savez_compressed(OUT/'geometry.npz', coeff=coeff)
    print(json.dumps({'phase':'geometry', 'descriptors':d}), flush=True)
    meta = build_case(coeff, re, OUT/'reference.npz', nx=40, ny=12)
    print(json.dumps({'phase':'mesh_verified', 'metadata':meta}), flush=True)
    model = FlowOperator(checkpoint, allow_unvalidated=True, chunk_size=512)
    assert model.metadata['step'] == 82216
    with np.load(OUT/'reference.npz', allow_pickle=False) as f:
        q, w, truth = f['xy'], f['weights'], f['fields']
    pred = model.evaluate(coeff, re, q)
    metrics = {name: weighted_relative(pred[:, ids], truth[:, ids], w)
               for name, ids in [('velocity_relative_l2', slice(0,2)), ('pressure_relative_l2', slice(2,3))]}
    fine = meta['fine']
    fem, solver_meta = solve_reference(coeff, re, nx=fine['nx'], ny=fine['ny'])
    x = np.linspace(0,5,401)
    c, dg = sample_numpy(coeff,x); b=c-dg/2; t=1+c+dg/2
    y = np.linspace(-.025,1.025,151)
    xx, yy = np.meshgrid(x,y)
    # Intersection of exact and chordal domains: never evaluate/extrapolate outside.
    mask = (yy>b[None,:]+1e-6)&(yy<t[None,:]-1e-6)
    xy = np.column_stack([xx.ravel(),yy.ravel()])
    mask &= fem.contains(xy).reshape(mask.shape)
    valid = mask.ravel()
    mp = np.full((xy.shape[0],3),np.nan); fp=mp.copy()
    mp[valid]=model.evaluate(coeff,re,xy[valid])
    # Chunk FEM interpolation to keep element-finder memory bounded.
    indices=np.flatnonzero(valid)
    for start in range(0,len(indices),1000):
        ii=indices[start:start+1000]; fp[ii]=fem(xy[ii])
    mp=mp.reshape((*xx.shape,3)); fp=fp.reshape((*xx.shape,3))
    # Exclude a 0.02 wall band from reverse-flow diagnostics; report raw grid minimum too.
    interior=mask & (yy>b[None,:]+.02) & (yy<t[None,:]-.02) & (xx>2.2) & (xx<4.5)
    reverse={}
    for name, fields in [('model',mp),('fem',fp)]:
        uu=fields[:,:,0]
        reverse[name]={'minimum_u_on_plot_grid':float(np.nanmin(uu)),
                       'minimum_u_in_downstream_interior':float(np.min(uu[interior])),
                       'downstream_interior_points_u_below_minus_0_001':int(np.sum(interior & (uu<-.001)))}
    np.savez_compressed(OUT/'fields.npz', x=x,y=y,bottom=b,top=t,mask=mask,model=mp,fem=fp,
                        quadrature_xy=q,quadrature_weights=w,quadrature_model=pred,quadrature_fem=truth,
                        coeff=coeff,reynolds=re,checkpoint_sha256=model.sha256)
    plt.rcParams.update({'font.size':10,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False})
    fig, axes=plt.subplots(3,1,figsize=(13,10),layout='constrained')
    speedmax=max(float(np.nanmax(np.linalg.norm(z[:,:,:2],axis=2))) for z in [mp,fp])
    for ax,fields,label in zip(axes[:2],[mp,fp],['Frozen PINN / DeepONet - Adam step 82,216','Independent P2/P1 FEM']):
        speed=np.linalg.norm(fields[:,:,:2],axis=2)
        pc=ax.pcolormesh(x,y,speed,cmap='viridis',vmin=0,vmax=speedmax,shading='auto',rasterized=True)
        u=np.ma.masked_invalid(fields[:,:,0]); v=np.ma.masked_invalid(fields[:,:,1])
        ax.streamplot(x,y,u,v,color='white',density=(2.5,1.3),linewidth=.65,arrowsize=.75)
        if np.nanmin(fields[:,:,0])<-.001:
            ax.contour(x,y,fields[:,:,0],levels=[-.001],colors=['#fb7185'],linewidths=1.3)
        ax.plot(x,b,'k',lw=2);ax.plot(x,t,'k',lw=2)
        ax.fill_between(x,0,b,color='#d1d5db',zorder=4)
        ax.set(xlim=(0,5),ylim=(-.03,1.03),aspect='equal',xlabel='x',ylabel='y',title=label)
        fig.colorbar(pc,ax=ax,label='Speed |U|',shrink=.85)
    ax=axes[2]
    for fields,label,color in [(mp,'PINN','#2563eb'),(fp,'FEM','#dc2626')]:
        profile=np.min(np.where(interior,fields[:,:,0],np.inf),axis=0)
        profile[~np.isfinite(profile)]=np.nan
        ax.plot(x,profile,label=label,color=color)
    ax.axhline(0,color='black',lw=.8);ax.axhline(-.001,color='#777',lw=.6,ls=':')
    ax.set(xlim=(2.2,4.5),xlabel='x',ylabel='min u over interior y',title='Downstream reverse-flow diagnostic (wall distance > 0.02)')
    ax.legend();ax.grid(alpha=.2)
    fig.suptitle('Rounded backward-facing-step analogue | Re = 100 | normal geometry envelope\nSmooth finite raised floor; not the classical sharp, unequal-height inlet/outlet benchmark',fontsize=13)
    fig.savefig(OUT/'streamlines.png',dpi=190);fig.savefig(OUT/'streamlines.pdf');plt.close(fig)
    summary={'checkpoint':str(checkpoint.relative_to(ROOT)),'checkpoint_step':82216,'checkpoint_sha256':model.sha256,
             'reynolds':re,'geometry_id':geometry_id(coeff),'descriptors':d,'metrics':metrics,
             'reverse_flow_diagnostics':reverse,'fem_reference':meta,'display_fem_solver':solver_meta,
             'geometry_scope':'Normal envelope, no stress override. Rounded finite upstream floor; not a classical sharp backward-facing step.',
             'reverse_flow_threshold':-.001,'wall_exclusion_for_downstream_diagnostic':.02,
             'accepted_model':False,'training_performed':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    render_zoom()
    femreverse=reverse['fem']['downstream_interior_points_u_below_minus_0_001']
    note=('Both solutions show downstream reverse flow. The model predicts a weaker, smaller reversed-flow region: minimum u approximately -0.01168 versus FEM -0.02738 on the saved grid. Closed-looking streamline loops are visible in the plots; exact separation/reattachment lengths and wall shear are not certified.' if femreverse else
          'FEM does not detect downstream reverse flow below u=-0.001 outside the 0.02 wall band. No separation bubble is claimed for this smooth, bounded case.')
    text=f'''# Rounded backward-facing-step analogue

Frozen Adam checkpoint **82,216**, Re=100. No retraining or Kaggle access. This is a **smooth analogue**, not the classical sharp-step benchmark: C4 walls, slope <=0.7, and equal flat inlet/outlet heights are mandatory. The raised upstream floor must therefore rise smoothly after the inlet and descend smoothly downstream. It cannot reproduce the classical vertical corner or arbitrary expansion ratio.

[Streamlines PNG](streamlines.png) · [PDF](streamlines.pdf) · [Numerical fields](fields.npz) · [Metrics and provenance](summary.json) · [Download complete case](rounded_step_82216.zip)

![Flow and streamlines](streamlines.png)

[Recirculation close-up PDF](recirculation_zoom.pdf)

![Recirculation close-up](recirculation_zoom.png)

Velocity relative L2 error: **{metrics['velocity_relative_l2']:.2%}**. Pressure relative L2 error: **{metrics['pressure_relative_l2']:.2%}**. Errors use physical-area weighted FEM quadrature, no fitted pressure shift. Maximum FEM mesh discrepancy: **{max(meta['mesh_uncertainty'].values()):.2%}**.

{note}

White curves are streamlines integrated through the evaluated velocity field, with arrows for direction. Pink contours, when present, delimit u=-0.001. Downstream diagnostics use 2.2<x<4.5 and distance >0.02 from each wall. A local negative velocity or a drawn streamline alone does not certify a closed recirculation bubble. The model remains an unaccepted research checkpoint.

Geometry gap range: {d['min_gap']:.4f}–{d['max_gap']:.4f}; maximum wall slope {d['max_slope']:.4f}; centerline deviation {d['center_deviation']:.4f}. All standard geometry checks passed without stress overrides.

Other compatible **benchmark analogues**: plane Poiseuille flow (exact straight channel); smooth converging-diverging channel / Venturi; symmetric smooth stenosis; smooth asymmetric diffuser; smooth S-shaped channel. These retain this model's inlet profile, Reynolds range and outlet condition. Lid-driven cavity, cylinder flow, sharp backward-facing step, and branching channels need different geometry or boundary conditions.

Reproduce: `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 JAX_ENABLE_X64=true .venv/bin/python -m scripts.rounded_step_flow` from the repository root. See the repository CPU environment pins. The checkpoint is referenced by its SHA256 above and in summary.json; it is not bundled in the result ZIP.
'''
    (OUT/'README.md').write_text(text)
    with zipfile.ZipFile(OUT/'rounded_step_82216.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.iterdir()):
            if path.suffix!='.zip':z.write(path,arcname=OUT.name+'/'+path.name)
    with zipfile.ZipFile(OUT/'rounded_step_82216.zip') as z:assert z.testzip() is None
    print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
