from pathlib import Path
import json,csv,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'reports/historical_adam_vs_soap'
def rows(p):return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []
def main():
 O.mkdir(exist_ok=True);hp=R/'reports/full_training_history/current_run_observation.json';vp=R/'reports/molab_adaptive_100k/diagnostic/observation.json';hist=json.loads(hp.read_text());hv=json.loads(vp.read_text())['validation']
 folders=[R/'artifacts/molab/soap-trials-v1/soap_2e5',R/'artifacts/molab/soap-continue10k-v1/continuation',R/'artifacts/molab/soap-continue20k-v1/continuation'];st=sum([rows(p/'training.jsonl') for p in folders],[]);sv=sum([rows(p/'probe.jsonl') for p in folders],[]);end=min(max(x['step'] for x in st),hist['rows'][-1][0]);st=[x for x in st if x['step']<=end];ht=[dict(step=x[0],loss=x[1],learning_rate=x[2]) for x in hist['rows'] if x[0]<=end]
 hmap={x['step']:x for x in hv};smap={x['step']:x for x in sv};common=sorted(set(hmap)&set(smap));out=[]
 for step in common:
  h,s=hmap[step],smap[step];out.append(dict(step=step,updates=step-287400,adam_control=h['control_loss'],soap_control=s['control'],adam_audit=h['audit_loss'],soap_audit=s['audit'],soap_audit_reduction_vs_adam_percent=100*(1-s['audit']/h['audit_loss']),historical_adam_lr=h['lr']))
 with (O/'same_step_probes.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(out[0]));w.writeheader();w.writerows(out)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(3,1,figsize=(12,10),layout='constrained')
 for label,t,v,col in [('Historical adaptive Adam',ht,[dict(step=x['step'],audit=x['audit_loss']) for x in hv if x['step']<=end],'#2563eb'),('SOAP, fixed2e-5',st,[x for x in sv if x['step']<=end],'#f97316')]:
  y=np.array([x['loss'] for x in t]);axs[0].plot([x['step'] for x in t][99:],np.convolve(y,np.ones(100)/100,mode='valid'),label=label,color=col);axs[1].plot([x['step'] for x in v],[x['audit'] for x in v],'o-',label=label,color=col);axs[2].step([x['step'] for x in t],[x.get('learning_rate',2e-5) for x in t],where='post',label=label,color=col)
 for ax in axs:ax.set(yscale='log',xlabel='Global continuation step');ax.grid(alpha=.2);ax.legend()
 axs[0].set_ylabel('Training loss:100-record mean');axs[1].set_ylabel('Identical fixed audit loss');axs[2].set_ylabel('Learning rate');fig.suptitle(f'Actual historical Adam vs SOAP | shared base287400 | through{end:,}',fontsize=14)
 fig.savefig(O/'comparison.png',dpi=170);fig.savefig(O/'comparison.pdf');plt.close(fig)
 summary={'historical_log_last_step':hist['rows'][-1][0],'historical_audit_last_step':hv[-1]['step'],'common_display_last_step':end,'last_exact_common_probe':out[-1],'historical_sources':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [hp,vp]},'note':'Earlier Adam used adaptive learning rates; SOAP uses constant2e-5. Same starting parameters and fixed control/audit cohorts. Different evaluation cadence: compare only exact common recorded steps. No loss extrapolation or additional Adam training.'}
 (O/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 (O/'README.md').write_text(f'''# Historical Adam (logged through378600) versus SOAP

![Actual common-range comparison](comparison.png)

[PDF](comparison.pdf) · [Exact same-step fixed probes CSV](same_step_probes.csv) · [Coverage and source hashes](summary.json).

The previously archived Adam continuation **287400→378600** supplies the blue curve. SOAP restarts from the same287400 weights and is plotted through its latest collected step **{end}**. Both curves show actual log data, with a trailing100-record arithmetic mean, without extrapolation. Earlier use of only the separate15k Adam constant-rate trial omitted this available longer history.

The earlier Adam schedule was adaptive (starting2e-5, then1e-5 and lower); SOAP is fixed2e-5. This is a comparison of the two actual optimizer/schedule combinations, not isolation of optimizer alone under identical LR. Both use the original resampling and common fixed control/audit preparation; base probe values agree. Their evaluation cadences differ, so the CSV compares only exactly matching recorded steps. No probe values are invented between evaluations. The historical audit snapshot ends372400, while raw training log ends378600.

The live notebook now uses this historical Adam curve in blue and the growing SOAP curve in orange. The standalone Adam15k trial remains archived and remains the available15k FEM comparison. Historical Adam checkpoint weights at matching later SOAP steps were not archived, so this chart does not manufacture a later FEM comparison. The older recorded373800 FEM diagnostic remains separate.

No new training, optimizer-state modification, or changes to the active SOAP run were made for this presentation correction. Reproduce with `python -m scripts.compare_historical_adam_soap` after restoring the input archives.
''')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
