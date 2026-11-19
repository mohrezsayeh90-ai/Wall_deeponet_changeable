from pathlib import Path
import json,csv,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];A=R/'artifacts/molab/soap-continue10k-v1';O=R/'reports/molab_soap_continue10k/completed'
def main():
 c=json.loads((O/'observation.json').read_text());sources={k:c[k] for k in ['base287400','soap_start292400','soap_final','adam15k']};rows=[];stats={}
 for name,d in sources.items():
  stats[name]={'saved20':d['saved20'],'pilot8':d['pilot8']}
  for rr in d['cases']:rows.append(dict(model=name,**rr))
  if name=='soap_final':
   for comparator in ['base287400','soap_start292400','adam15k']:
    stats[name]['improved_vs_'+comparator]={f'{field}_{label}':sum(x[field]<y[field] for x,y in zip(d['cases'][sl],sources[comparator]['cases'][sl])) for field in ['velocity_l2','pressure_l2'] for label,sl in [('saved20',slice(0,20)),('pilot8',slice(20,28))]}
 with (O/'per_case_errors.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
 for ax,(group,field) in zip(axs.flat,[('saved20','velocity_l2_median'),('saved20','pressure_l2_median'),('pilot8','velocity_l2_median'),('pilot8','pressure_l2_median')]):
  names=['base287400','adam15k','soap_start292400','soap_final'];v=[100*sources[n][group][field] for n in names];b=ax.bar(['Base','Adam15k','SOAP5k','SOAP15k'],v,color=['#94a3b8','#3b82f6','#a78bfa','#0d9488']);ax.bar_label(b,fmt='%.3f%%',padding=3);ax.set(ylim=(0,max(v)*1.25),ylabel='Median relative L2 (%)',title=group+' | '+('velocity' if field.startswith('velocity') else 'pressure'));ax.grid(axis='y',alpha=.2)
 fig.suptitle('Completed SOAP15k versus Adam15k, same original checkpoint',fontsize=15);fig.savefig(O/'fem_comparison.png',dpi=170);fig.savefig(O/'fem_comparison.pdf');plt.close(fig)
 total=c['initial_soap5k_elapsed']+c['completion']['elapsed_seconds'];adam_time=json.loads((R/'artifacts/molab/rad-ab-15k-v1/baseline/completed.json').read_text())['elapsed_seconds'];stats['timing']={'soap_two_stage_seconds':total,'adam15k_seconds':adam_time,'ratio':total/adam_time,'scope':'Includes branch evaluation/checkpoint overhead; excludes extra separate continuation preflight/setup time. Not a matched-time run.'}
 (O/'analysis.json').write_text(json.dumps(stats,indent=2)+'\n')
 base=c['base287400']['saved20'];final=c['soap_final']['saved20'];adam=c['adam15k']['saved20'];pct=lambda x,y:100*(1-x/y)
 text='''# Completed SOAP10k continuation: SOAP15k total

**292400→302400 completed,10000 additional updates, exit0.** The complete SOAP state was preserved, reaching optimizer count15001 for15000 parameter updates plus its original zero-update basis initialization. No restart, loss reweighting, RAD or model architecture change was introduced. The regression guard did not stop the run; this only means its specified thresholds were not triggered, not that every case improved.

![Same-reference FEM comparison](fem_comparison.png)

[PDF](fem_comparison.pdf) · [Every-case CSV](per_case_errors.csv) · [Analysis JSON](analysis.json) · [Original combined training curves](../../../artifacts/molab/soap-continue10k-v1/comparison.png) · [Snapshots and raw results](../../../artifacts/molab/soap-continue10k-v1/).

| Median relative L2 | Original287400 | Adam15k | SOAP5k | SOAP15k |
| --- | ---: | ---: | ---: | ---: |
'''
 for group,key,label in [('saved20','velocity_l2_median','20 Re30: velocity'),('saved20','pressure_l2_median','20 Re30: pressure'),('pilot8','velocity_l2_median','8 Re10: velocity'),('pilot8','pressure_l2_median','8 Re10: pressure')]:text+='| '+label+' | '+' | '.join(f'{100*sources[n][group][key]:.4f}%' for n in ['base287400','adam15k','soap_start292400','soap_final'])+' |\n'
 text+=f'''
SOAP15k lowers the20-case velocity/pressure medians by **{pct(final['velocity_l2_median'],base['velocity_l2_median']):.2f}% / {pct(final['pressure_l2_median'],base['pressure_l2_median']):.2f}%** relative to the original checkpoint, and **{pct(final['velocity_l2_median'],adam['velocity_l2_median']):.2f}% / {pct(final['pressure_l2_median'],adam['pressure_l2_median']):.2f}%** versus Adam at the same15000-update budget. These are relative changes in median relative-L2 metrics, not percentage-point changes. On the Re10 pilot, SOAP has better pressure median but slightly worse velocity median than Adam; it is not superior in every metric.

Worst20-case velocity/pressure errors remain **{100*final['velocity_l2_max']:.3f}% / {100*final['pressure_l2_max']:.3f}%**. The pilot worst pressure error increased versus the original checkpoint. Thus an aggregate improvement does not resolve difficult geometries or establish whole-domain accuracy. The20Re30 and8Re10 cohort gives no direct validation atRe60/100. The complete per-case CSV retains regressions.

Observed branch times total **{total/60:.2f}min SOAP** versus **{adam_time/60:.2f}min Adam** (about{100*(total/adam_time-1):.1f}% longer for SOAP). Extra separate setup/preflight time is excluded, and checkpoint/evaluation schedules differ. No equal-wall-time or large multiplicative speedup is established.

## Decision and next step

Keep SOAP at **2e-5** as the preferred optimizer candidate. Preserve the original and Adam checkpoints; do not claim universal model promotion. The latest full SOAP state302400 is the continuation source, not a fresh optimizer.

Before investing in another long run, expand the **development FEM validation to32 fresh cases balanced across8 geometry families and Re10,30,60,100**, excluding the existing pilot and fixed control/audit geometry IDs. Use independent mesh-converged FEM, not new labels in training; keep locked test unopened. Compare original287400, Adam15k and SOAP15k on exactly these same inputs. This addresses the current coverage gap and tells us whether pressure gains extend beyond the repeatedly examined cases.

If the broader cases confirm improvement without material systematic regression, a **bounded20k continuation** at2e-5 from302400 is the next training candidate, with periodic FEM and retained states. If higher-Re or difficult-case velocity still regresses, diagnose residual/field maps before adding compute or changing architecture. Do not return to LR1e-4, add RAD, change loss weights, or launch100k simply because the stochastic training curve is lower.

**No new validation batch or training continuation was launched by this analysis.** This report records a concrete next proposal, not completed validation or an authorized model release.
'''
 (O/'README.md').write_text(text)
 p=O.parent/'README.md';s=p.read_text();s='**Completed at302400. [Final analysis and next proposal](completed/README.md).**\n\n'+s if 'Completed at302400.' not in s else s;p.write_text(s)
 print(json.dumps(stats['soap_final'],indent=2));print(stats['timing'])
if __name__=='__main__':main()
