from pathlib import Path
import json,csv,numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=Path(__file__).resolve().parents[3];out=r/'reports/molab_adaptive_100k/diagnostic';d=json.loads((out/'observation.json').read_text());old=json.loads((out/'previous_run_windows.json').read_text());comp=json.loads((out/'comparison.json').read_text()) if (out/'comparison.json').exists() else {}
with (out/'training_windows.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=['run','start_step','end_step','records','mean','median','complete_10000_steps']);w.writeheader()
 for run,base,rows in [('previous_cosine',187400,old['windows']),('current_adaptive',287400,d['windows'])]:
  for a in rows:w.writerow(dict(run=run,start_step=base+a['offset']+1,end_step=a.get('last',base+a['offset']+10000),records=a['n'],mean=a['mean'],median=a['median'],complete_10000_steps=a.get('last',base+a['offset']+10000)==base+a['offset']+10000))
fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained')
ax=axs[0,0]
for label,base,rows,color in [('Previous cosine',187400,old['windows'],'#64748b'),('Current adaptive',287400,d['windows'],'#7c3aed')]:
 xs=[(base+a['offset']+1+a.get('last',base+a['offset']+10000))/2 for a in rows];ys=[a['mean'] for a in rows];ax.plot(xs,ys,'o-',label=label,color=color)
ax.axvline(287400,color='k',ls=':',alpha=.5);ax.set(title='Mean of logged training loss in 10k-step windows',ylabel='Loss',xlabel='Global Adam step');ax.text(.02,.03,'Last current window is partial',transform=ax.transAxes,fontsize=9)
ax=axs[0,1];ax.plot([x['step'] for x in old['probe']],[x['loss'] for x in old['probe']],'o-',color='#64748b',label='Previous fixed probe')
if comp:
 for label,marker,color in [('base_287400','s','#64748b'),('previous_best_282400','*','#0d9488'),('observed_current','o','#7c3aed')]:
  x=comp[label];ax.scatter(x['step'],x['probe']['previous_fixed'],marker=marker,s=75,color=color,label=f"Re-evaluated {x['step']:,}")
ax.set(title='Same previous fixed probe: comparable checkpoint evaluations',ylabel='Loss',xlabel='Global Adam step')
ax=axs[1,0]
for k,label,color in [('control_loss','LR-control cohort','#2563eb'),('audit_loss','Disjoint audit cohort','#dc2626')]:
 v=d['validation'];ax.plot([x['step'] for x in v],[100*(x[k]/v[0][k]-1) for x in v],'o-',color=color,label=label)
ax.axhline(0,color='k',ls=':',alpha=.5);ax.set(title='Current fixed cohorts: change from starting checkpoint',ylabel='Loss change (%)',xlabel='Global Adam step')
ax=axs[1,1];xs=np.linspace(187400,287400,200);ax.plot(xs,2e-5+.5*8e-5*(1+np.cos(np.pi*(xs-187400)/100000)),label='Previous cosine',color='#64748b');t=d['sampled_training'];ax.step([x['step'] for x in t],[x['lr'] for x in t],where='post',color='#7c3aed',label='Current adaptive');ax.set(yscale='log',title='Learning rates',ylabel='Learning rate',xlabel='Global Adam step')
for ax in axs.flat:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Training observed '+d['observed_utc']+' | Checkpoint snapshot: 373,800',fontsize=14)
fig.savefig(out/'continuation_diagnostic.png',dpi=170);fig.savefig(out/'continuation_diagnostic.pdf');plt.close(fig)
print('PLOT_READY',out/'continuation_diagnostic.png')
