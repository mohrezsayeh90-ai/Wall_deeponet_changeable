from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/marimo/wall_rad_ab_15k_v1');b=json.loads((R/'base_fem.json').read_text());results={}
fig,axs=plt.subplots(2,1,figsize=(12,8),layout='constrained')
for name in ['baseline','rad']:
 r=[json.loads(l) for l in (R/name/'training.jsonl').read_text().splitlines()];v=[json.loads(l) for l in (R/name/'probe.jsonl').read_text().splitlines()];y=np.array([x['loss'] for x in r]);axs[0].plot([x['step'] for x in r][99:],np.convolve(y,np.ones(100)/100,mode='valid'),label=name)
 axs[1].plot([x['step'] for x in v],[x['audit'] for x in v],'o-',label=name+' audit')
 f=json.loads((R/name/'fem_302400.json').read_text());results[name]={'fem':f,'final_probe':v[-1],'elapsed_seconds':json.loads((R/name/'completed.json').read_text())['elapsed_seconds']}
 for group in ['saved20','pilot8']:
  results[name][group+'_median_ratios']={key:f[group][key]/b[group][key] for key in ['velocity_l2_median','pressure_l2_median']}
 for field in ['velocity_l2','pressure_l2']:results[name][field+'_worsened_cases']=sum(x[field]>z[field] for x,z in zip(f['cases'],b['cases']))
for ax in axs:ax.set(yscale='log',xlabel='Global Adam step',ylabel='Loss');ax.grid(alpha=.2);ax.legend()
axs[0].set_title('Training objectives differ under RAD: use fixed audit and FEM for comparison');axs[1].set_title('Identical fixed audit: no FEM labels used for training')
fig.savefig(R/'comparison.png',dpi=170);fig.savefig(R/'comparison.pdf');plt.close(fig)
(R/'comparison.json').write_text(json.dumps({'base_step':287400,'final_steps':302400,'base_fem':b,'branches':results,'promotion':'No automatic promotion; inspect all case regressions and worst errors, not only medians','scope':'20 saved Re30 and 8 Re10 pilot cases; not a full generalization benchmark'},indent=2))
