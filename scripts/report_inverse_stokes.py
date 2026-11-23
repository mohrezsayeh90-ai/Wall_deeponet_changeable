"""Package an inverse-particle review with the actual outlet residuals."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from geometry_operator.geometry import sample_numpy


TOLERANCE = .01


def mark_outlet_divisions(panel, target):
    """Draw six equal outlet intervals with labels for the five interior targets."""
    boundaries = np.linspace(0, 1, 7)
    ruler_left, ruler_right = 5.10, 5.22
    for index, (bottom, top) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        fill = '#dbeafe' if index % 2 == 0 else '#f1f5f9'
        panel.fill_betweenx([bottom, top], ruler_left, ruler_right, color=fill, zorder=3)
    panel.vlines([ruler_left, ruler_right], 0, 1, colors='#94a3b8', linewidth=.6, zorder=4)
    for index, height in enumerate(boundaries):
        selected = np.isclose(height, target)
        color = '#be123c' if selected else '#64748b'
        panel.hlines(height, ruler_left-.03, ruler_right+.03, colors=color,
                     linewidth=1.4 if selected else .65, zorder=5)
        if 0 < index < 6:
            panel.hlines(height, 4.84, ruler_left, colors=color, linestyles=':',
                         linewidth=.8, alpha=.75, zorder=2)
            panel.text(ruler_right+.08, height, f'{index}/6', va='center', ha='left',
                       fontsize=7.5, color=color, weight='bold' if selected else 'normal')
    panel.set_xlim(-.15, 5.67)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    args=parser.parse_args()
    folder=args.output
    cases=[json.loads(p.read_text()) for p in sorted(folder.glob('P[123]_T[12345].json'))]
    assert len(cases)==15, len(cases)
    rows=[]
    fig,axes=plt.subplots(3,5,figsize=(21,9),layout='constrained',sharex=True,sharey=True)
    with PdfPages(folder/'designs_and_trajectories.pdf') as pdf:
        for case,ax in zip(cases,axes.flat):
            name=case['case'];target=case['target']
            model=case['model_exit'];fem=case['fem']['fine']['exit_y']
            coarse=case['fem']['coarse']['exit_y']
            difference=case['mesh_exit_difference']
            error_model=abs(model-target) if model is not None else None
            error_fem=abs(fem-target) if fem is not None else None
            outcome=('hit' if error_fem+difference<=TOLERANCE else
                     'miss' if error_fem-difference>TOLERANCE else 'near tolerance') if difference is not None else 'unresolved'
            rows.append(dict(case=name,tau=case['tau'],target_y_over_W=target,
                             model_exit_y_over_W=model,model_absolute_error_over_W=error_model,
                             model_hit=error_model is not None and error_model<=TOLERANCE,
                             initial_deeponet_exit_y_over_W=case['initial_deeponet_exit'],
                             initial_design_fem_exit_y_over_W=case['initial_fem_fine_exit'],
                             fem_coarse_exit_y_over_W=coarse,fem_fine_exit_y_over_W=fem,
                             fem_absolute_error_over_W=error_fem,mesh_delta_over_W=difference,
                             outcome=outcome,design_method=case['design_method']))
            with np.load(folder/f'{name}.npz') as z:
                coeff=z['coeff'];x=np.linspace(0,5,501);c,g=sample_numpy(coeff,x)
                def draw(panel, *, overview=False):
                    panel.fill_between(x,c-g/2,1+c+g/2,color='#eff6ff')
                    panel.plot(x,c-g/2,color='#1f2937')
                    panel.plot(x,1+c+g/2,color='#1f2937')
                    panel.plot([5,5],[0,1],':',color='#64748b')
                    for key,color,label in [('model','#2563eb','DeepONet'),('fine','#059669','FEM fine')]:
                        if key+'_state' in z:
                            state=z[key+'_state'];panel.plot(state[:,0],state[:,1],color=color,label=label)
                            panel.plot(*state[-1,:2],'o',color=color,ms=4)
                    panel.plot(0,.5,'o',color='#231942',ms=4,label='Release at rest')
                    if overview:
                        panel.plot(5, target, marker='o', markerfacecolor='none',
                                   markeredgecolor='#be123c', markeredgewidth=1.15,
                                   markersize=8, zorder=8, label='Target')
                    else:
                        panel.plot(5,target,'x',color='#e11d48',ms=7,mew=2,label='Target')
                    panel.set(xlim=(-.15,5.2),ylim=(-.48,1.48),xlabel='x/W',ylabel='y/W')
                    panel.grid(alpha=.15)
                draw(ax, overview=True)
                mark_outlet_divisions(ax, target)
                if fem is not None:
                    ax.plot(5, fem, 'o', color='#059669', markeredgecolor='white',
                            markeredgewidth=.7, markersize=4.5, zorder=7)
                err=f'{100*error_fem:.2f}% W' if error_fem is not None else 'unresolved'
                ax.set_title(f'{name}  target {target:.3f} | {outcome}\nFEM offset {err}',fontsize=10)
                detail,panel=plt.subplots(figsize=(10,4.3),layout='constrained')
                draw(panel)
                panel.legend(ncol=4,loc='lower center',fontsize=8)
                panel.set_title(f'{name} · response time {case["tau"]:g} W/U · target {target:.5f} W\n'
                                f'FEM offset {err} · {outcome}\n{case["design_method"]}')
                pdf.savefig(detail);plt.close(detail)
        handles,labels=axes[0,0].get_legend_handles_labels()
        fig.legend(handles,labels,ncol=len(labels),loc='outside upper center')
        fig.savefig(folder/'all_15_designs.png',dpi=140)
        pdf.savefig(fig)
    plt.close(fig)
    with (folder/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert len({r['case'] for r in rows})==15
    record=dict(model_step=442400,checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                reynolds=30.,tolerance_over_W=TOLERANCE,results=rows)
    (folder/'results.json').write_text(json.dumps(record,indent=2)+'\n')
    for row in rows:print(json.dumps(row),flush=True)
    local_sources=['inverse_stokes_particles.py','report_inverse_stokes.py']
    with zipfile.ZipFile(folder/'inverse_particle_results.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and not p.name.startswith('screen_') and p.suffix in {'.json','.npz','.png','.pdf','.csv','.md'}:
                z.write(p,p.relative_to(folder))
        for name in local_sources:z.write(Path(__file__).with_name(name),name)
    with zipfile.ZipFile(folder/'inverse_particle_results.zip') as z:assert z.testzip() is None


if __name__=='__main__':
    main()
