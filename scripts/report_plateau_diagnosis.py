"""Reproduce fixed-objective/FEM comparisons of bounded plateau interventions."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];D=R/'reports/plateau_diagnosis'
def read(p):return json.loads(p.read_text())
b=read(D/'batch128_lr2e5_result.json');ref=b['batch32_reference'];rows=[('Batch32 reference',ref['probe'],ref['fem']['saved20'])]
for file,key,name in [('batch128_lr2e5_result.json','batch128','Batch128, LR 2e-5'),('batch128_lr4e5_result.json','batch128','Batch128, LR 4e-5')]:
 x=read(D/file)[key];rows.append((name,x['probe'],x['fem']['saved20']))
x=read(D/'weight_average_result.json')['methods']['average'];rows.append(('Weight average (inference only)',x['probe'],x['saved20']))
p=D/'branch_only_result.json'
if p.exists():
 x=read(p)['branch_only'];rows.append(('Frozen trunk, branch-only',x['probe'],x['fem']['saved20']))
with (D/'comparison.csv').open('w') as f:
 w=csv.writer(f);w.writerow(['method','control_loss','audit_loss','median_velocity_relative_L2','median_pressure_relative_L2','reference','scope'])
 for name,probe,med in rows:w.writerow([name,probe['control'],probe['audit'],med['velocity_l2_median'],med['pressure_l2_median'],'batch32 full SOAP at315400','development only; weight average is not a trained checkpoint'])
fig,axs=plt.subplots(1,2,figsize=(12,5.8),layout='constrained',sharey=True);names=[x[0].replace('Weight average (inference only)','Weight average\n(inference only)').replace('Frozen trunk, branch-only','Frozen trunk\nbranch-only') for x in rows[1:]];ix=np.arange(len(names));colors=['#e67e22','#c0392b','#16a085','#8e44ad'][:len(names)]
changes=[100*(x[1]['audit']/ref['probe']['audit']-1) for x in rows[1:]]
axs[0].barh(ix,changes,color=colors);axs[0].set(yticks=ix,yticklabels=names,xlabel='Change in fixed audit loss (%)',title='Fixed audit loss')
for i,v in enumerate(changes):axs[0].text(v,i,f' {v:+.2f}%',va='center',ha='left' if v>=0 else 'right')
for field,offset,color,label in [('velocity_l2_median',-.17,'#2563eb','Velocity'),('pressure_l2_median',.17,'#e67e22','Pressure')]:
 values=[100*(x[2][field]/ref['fem']['saved20'][field]-1) for x in rows[1:]];axs[1].barh(ix+offset,values,height=.32,color=color,label=label)
axs[1].set(yticks=ix,yticklabels=names,xlabel='Change in median relative L2 error (%)',title='20-case FEM medians');axs[1].legend()
for a in axs:a.axvline(0,color='black',lw=.8);a.grid(axis='x',alpha=.2)
axs[0].set_xlim(min(changes)-.65,max(changes)+.65)
fig.suptitle('Plateau interventions vs matched batch32 continuation — negative is better',fontsize=12)
fig.savefig(D/'comparison.png',dpi=180);fig.savefig(D/'comparison.pdf');plt.close(fig)
