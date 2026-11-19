from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/marimo/wall_soap_trials_v1');OLD=Path('/home/marimo/wall_rad_ab_15k_v1');results={}
fig,axs=plt.subplots(2,1,figsize=(11,7),layout='constrained')
for name,folder in [('adam',OLD/'baseline'),('soap_2e5',R/'soap_2e5'),('soap_1e4',R/'soap_1e4')]:
 logs=[json.loads(l) for l in (folder/'training.jsonl').read_text().splitlines()];logs=[x for x in logs if x['step']<=292400]
 vv=[json.loads(l) for l in (folder/'probe.jsonl').read_text().splitlines()];vv=[x for x in vv if x['step']<=292400]
 y=np.array([x['loss'] for x in logs]);axs[0].plot([x['step']-287400 for x in logs][99:],np.convolve(y,np.ones(100)/100,mode='valid'),label=name)
 axs[1].plot([x['offset'] for x in vv],[x['audit'] for x in vv],'o-',label=name+' audit')
 results[name]={'fem':json.loads((folder/'fem_292400.json').read_text()),'last_probe':vv[-1],'elapsed_to_last_5000_log':logs[-1]['elapsed_seconds'],'scope':'Same starting parameters and baseline RNG; SOAP own fresh state and extra initialization gradient; Adam continues saved state'}
 if name!='adam':results[name]['completion']=json.loads((folder/'completed.json').read_text())
for ax in axs:ax.set(yscale='log',xlabel='Parameter updates from same base',ylabel='Loss');ax.grid(alpha=.2);ax.legend()
axs[0].set_title('Identical baseline sampling: 100-record mean');axs[1].set_title('Identical fixed audit cohort')
fig.savefig(R/'comparison.png',dpi=160);fig.savefig(R/'comparison.pdf');plt.close(fig)
(R/'comparison.json').write_text(json.dumps({'base_step':287400,'updates_each':5000,'base_fem':json.loads((R/'base_fem.json').read_text()),'methods':results,'selection':'No automatic promotion. Compare individual errors and worst cases, equal updates and measured elapsed time. Existing Adam15k endpoint is an additional longer-budget reference, not equal-updates SOAP comparison.'},indent=2))
