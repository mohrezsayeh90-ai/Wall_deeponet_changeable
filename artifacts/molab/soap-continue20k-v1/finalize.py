from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/marimo/wall_soap_continue20k_v1');S1=Path('/home/marimo/wall_soap_trials_v1');S2=Path('/home/marimo/wall_soap_continue10k_v1');OLD=Path('/home/marimo/wall_rad_ab_15k_v1')
load=lambda p:json.loads(p.read_text());rows=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
hist=load(R/'historical_adam_training.json');hv=load(R/'historical_adam_probe.json')['validation']
soaplog=rows(S1/'soap_2e5/training.jsonl')+rows(S2/'continuation/training.jsonl')+rows(R/'continuation/training.jsonl');soapv=rows(S1/'soap_2e5/probe.jsonl')+rows(S2/'continuation/probe.jsonl')+rows(R/'continuation/probe.jsonl');end=soaplog[-1]['step']
fig,axs=plt.subplots(3,1,figsize=(11,10),layout='constrained')
for name,logs,probe,color in [('Historical Adam (adaptive LR)',[dict(step=x[0],loss=x[1],learning_rate=x[2]) for x in hist['rows'] if x[0]<=end],[dict(step=x['step'],audit=x['audit_loss']) for x in hv if x['step']<=end],'#2563eb'),('SOAP (fixed LR)',soaplog,soapv,'#f97316')]:
 y=np.array([x['loss'] for x in logs]);axs[0].plot([x['step']-287400 for x in logs][99:],np.convolve(y,np.ones(100)/100,mode='valid'),label=name,color=color);axs[1].plot([x['step']-287400 for x in probe],[x['audit'] for x in probe],'o-',label=name,color=color);axs[2].step([x['step']-287400 for x in logs],[x.get('learning_rate',2e-5) for x in logs],where='post',label=name,color=color)
for ax in axs:ax.set(yscale='log',xlabel='Updates since common checkpoint287400');ax.grid(alpha=.2);ax.legend()
axs[0].set_ylabel('Training loss mean');axs[1].set_ylabel('Same fixed audit loss');axs[2].set_ylabel('Learning rate')
fig.savefig(R/'comparison.png',dpi=160);fig.savefig(R/'comparison.pdf');plt.close(fig)
report={'status':load(R/'status.json'),'base287400':load(OLD/'base_fem.json'),'soap_start302400':load(R/'start_fem.json'),'soap_final':load(R/'continuation/final_fem.json'),'adam15k_shorter_budget_reference':load(OLD/'baseline/fem_302400.json'),'completion':load(R/'continuation/completed.json'),'historical_adam_log_last_step':hist['rows'][-1][0],'selection':'Training/audit curves use the actual earlier adaptive-Adam history through378600, clipped to common SOAP progress; LR policies differ. The FEM Adam15k reference has a shorter budget; later historical Adam weights are not archived for same-step FEM.'}
(R/'comparison.json').write_text(json.dumps(report,indent=2))
