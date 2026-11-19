from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/marimo/wall_soap_continue10k_v1');PREV=Path('/home/marimo/wall_soap_trials_v1');OLD=Path('/home/marimo/wall_rad_ab_15k_v1')
load=lambda p:json.loads(p.read_text())
rows=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
fig,axs=plt.subplots(2,1,figsize=(11,7),layout='constrained')
for name,logs,probe in [('Adam15k',rows(OLD/'baseline/training.jsonl'),rows(OLD/'baseline/probe.jsonl')),('SOAP continued',rows(PREV/'soap_2e5/training.jsonl')+rows(R/'continuation/training.jsonl'),rows(PREV/'soap_2e5/probe.jsonl')+rows(R/'continuation/probe.jsonl'))]:
 y=np.array([x['loss'] for x in logs]);axs[0].plot([x['step']-287400 for x in logs][99:],np.convolve(y,np.ones(100)/100,mode='valid'),label=name)
 axs[1].plot([x['step']-287400 for x in probe],[x['audit'] for x in probe],'o-',label=name)
for ax in axs:ax.set(yscale='log',xlabel='Total updates since base287400',ylabel='Loss');ax.grid(alpha=.2);ax.legend()
fig.savefig(R/'comparison.png',dpi=160);fig.savefig(R/'comparison.pdf');plt.close(fig)
report={'status':load(R/'status.json'),'base287400':load(OLD/'base_fem.json'),'soap_start292400':load(R/'start_fem.json'),'soap_final':load(R/'continuation/final_fem.json'),'adam15k':load(OLD/'baseline/fem_302400.json'),'completion':load(R/'continuation/completed.json'),'initial_soap5k_elapsed':load(PREV/'soap_2e5/completed.json')['elapsed_seconds'],'selection':'No automatic model promotion; inspect all individual regressions. If SOAP stopped early, Adam15k is not a matched-update comparison.'}
(R/'comparison.json').write_text(json.dumps(report,indent=2))
