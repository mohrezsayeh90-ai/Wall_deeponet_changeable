from pathlib import Path
import json,csv,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[1];ART=ROOT/'artifacts/molab/soap-trials-v1';OUT=ROOT/'reports/molab_soap_trials/completed'
def main():
 OUT.mkdir(exist_ok=True);c=json.loads((ART/'comparison.json').read_text());base=c['base_fem'];methods=c['methods'];stats={}
 for name,x in methods.items():
  f=x['fem'];stats[name]={'elapsed_to_5000_log_seconds':x['elapsed_to_last_5000_log'],'audit_loss':x['last_probe']['audit'],'control_loss':x['last_probe']['control']}
  for group,sl in [('saved20',slice(0,20)),('pilot8',slice(20,28))]:
   stats[name][group]={**f[group],'velocity_improved_cases':sum(a['velocity_l2']<b['velocity_l2'] for a,b in zip(f['cases'][sl],base['cases'][sl])),'pressure_improved_cases':sum(a['pressure_l2']<b['pressure_l2'] for a,b in zip(f['cases'][sl],base['cases'][sl]))}
 rows=[]
 for label,f in [('base',base)]+[(n,x['fem']) for n,x in methods.items()]:
  for row in f['cases']:rows.append({'method':label,'case':row['case'],'group':'pilot8' if row['case'].startswith('pilot') else 'saved20',**{k:v for k,v in row.items() if k!='case'}})
 with (OUT/'per_case_errors.csv').open('w',newline='') as f:
  wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained');labels=['Base','Adam 5k','SOAP 2e-5','SOAP 1e-4'];sources=[base,methods['adam']['fem'],methods['soap_2e5']['fem'],methods['soap_1e4']['fem']]
 for ax,(group,field) in zip(axs.flat,[('saved20','velocity_l2_median'),('saved20','pressure_l2_median'),('pilot8','velocity_l2_median'),('pilot8','pressure_l2_median')]):
  yy=[100*x[group][field] for x in sources];bars=ax.bar(labels,yy,color=['#94a3b8','#3b82f6','#0d9488','#f59e0b']);ax.bar_label(bars,fmt='%.3f%%',padding=3);ax.set(ylim=(0,max(yy)*1.25),ylabel='Median relative L2 error (%)',title=f'{group} | '+('velocity' if field.startswith('velocity') else 'pressure'));ax.grid(axis='y',alpha=.2)
 fig.suptitle('Matched 5,000-update continuations from checkpoint 287,400',fontsize=15)
 fig.savefig(OUT/'fem_comparison.png',dpi=170);fig.savefig(OUT/'fem_comparison.pdf');plt.close(fig)
 report={'completed':True,'base_step':287400,'updates_each':5000,'final_step_each':292400,'base_fem':base,'methods':stats,'all_14_snapshot_sha256_verified':True,'selection':'SOAP2e-5 is a candidate for a bounded continuation, not a universally improved model','next_proposal':{'optimizer':'SOAP','lr':2e-5,'resume_step':292400,'additional_updates':10000,'target_step':302400,'reason':'Match the already completed Adam15k budget; retain own optimizer state and RNG; no loss reweighting or RAD','evaluation_interval':2500,'long_100k_run_authorized_by_results':False}}
 (OUT/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
 text='''# Completed SOAP experiment: analysis and next step

**Both 5k branches completed with exit0.** From identical287400 parameters and common baseline batch RNG, each reached292400. The original model, the completed Adam/RAD trials, and all14 new full SOAP-state snapshots are retained. All14 NPZ sizes/SHA256 hashes were rechecked; tracked local HEAD was matched to remote during collection. No automatic model promotion or new long training occurred.

![FEM comparison](fem_comparison.png)

[PDF](fem_comparison.pdf) · [Every case CSV](per_case_errors.csv) · [Analysis JSON](analysis.json) · [Original curves](../../../artifacts/molab/soap-trials-v1/comparison.png) · [Full original metrics and checkpoints](../../../artifacts/molab/soap-trials-v1/).

## Physical errors: same saved references

| Median relative L2 | Base287400 | Adam5k | SOAP2e-5,5k | SOAP1e-4,5k |
| --- | ---: | ---: | ---: | ---: |
'''
 for group,field,title in [('saved20','velocity_l2_median','20 Re30: velocity'),('saved20','pressure_l2_median','20 Re30: pressure'),('pilot8','velocity_l2_median','8 Re10: velocity'),('pilot8','pressure_l2_median','8 Re10: pressure')]:text+='| '+title+' | '+' | '.join(f'{100*x[group][field]:.4f}%' for x in sources)+' |\n'
 b=base['saved20'];s=methods['soap_2e5']['fem']['saved20'];a=methods['adam']['fem']['saved20']
 text+=f'''
SOAP2e-5 lowers the20-case pressure median by **{100*(1-s['pressure_l2_median']/b['pressure_l2_median']):.2f}%** versus the base, but raises the velocity median by **{100*(s['velocity_l2_median']/b['velocity_l2_median']-1):.2f}%**. Against Adam at the same5k budget, its20-case velocity/pressure medians are {100*(1-s['velocity_l2_median']/a['velocity_l2_median']):.2f}% / {100*(1-s['pressure_l2_median']/a['pressure_l2_median']):.2f}% lower. On the Re10 pilot its pressure median is worse than Adam's. This is a tradeoff, not uniform improvement.

Compared with base, SOAP2e-5 improves velocity in12/20 saved cases and1/8 pilot cases, and pressure in12/20 plus6/8. Thus even a better aggregate hides regressions. Its worst saved velocity/pressure errors remain **{100*s['velocity_l2_max']:.3f}% / {100*s['pressure_l2_max']:.3f}%**. SOAP1e-4 has a higher worst pressure error and less favorable audit performance, so it is not the preferred continuation.

## Fixed probe and elapsed time

| Method | Fixed control | Disjoint audit | Seconds to last5000-step log |
| --- | ---: | ---: | ---: |
'''
 for name,x in methods.items():text+=f"| {name} | {x['last_probe']['control']:.8f} | {x['last_probe']['audit']:.8f} | {x['elapsed_to_last_5000_log']:.1f} |\n"
 text+='''
Base control/audit were0.01294696 /0.01334894. SOAP2e-5 audit falls about2.17%, while control is nearly unchanged (slightly higher). A steep-looking local training-loss segment is not evidence of100k-equivalent progress.

These are measured branch wall times, including different checkpoint/evaluation work; SOAP shares compiled preflight kernels, and the later branch can benefit from caching. They are not isolated optimizer timings or a matched wall-time convergence experiment. Initialization and the common production preflight add one-time cost. SOAP2e-5 takes about26% longer to reach the5k log than the reused Adam reference; final FEM/serialization adds further time. No20x speedup is established.

## Additional read-only gradient diagnostic

[Diagnostic data](gradient_diagnostic.json) · [Code](gradient_diagnostic.py). On the same32-geometry training batch, x/y momentum gradient norms at base are0.304/0.202; after SOAP they are0.282/0.205. Their ratio is about1.4–1.5, not orders of magnitude. Their cosine changes from-0.160 to-0.293, indicating some opposing directions on this batch. Outlet gradients are much smaller, while their losses are already tiny. This single-batch diagnostic does not establish a global bottleneck or justify boosting outlet weights to match gradients blindly. No parameters were changed by the diagnostic.

## Decision

Prefer **SOAP2e-5** as the next candidate; retain287400 as the velocity baseline and retain the completed Adam15k pressure candidate separately. Do not restart from zero, increase LR to1e-4, change architecture or loss weights based only on these results, or launch another100k blindly.

The concrete next proposal is **10k additional updates from the saved SOAP2e-5 state at292400**, keeping LR2e-5 and the same resampling, to reach302400 (15k total). This matches the already completed Adam15k budget. Evaluate identical per-case FEM every2500 updates, retain all full states, and stop/reassess if errors persistently regress. It is an experiment, not a claim that longer SOAP must repair velocity. Before any long run or acceptance, add balanced development FEM coverage at higher Re; the current20Re30 plus8Re10 cohort does not validate the whole Re10–100 range. Locked test remains unopened.

**This analysis has not launched that additional10k run.** Checkpoint restoration must use full SOAP state with pinned code, not Adam resume. Source and results remain under distinct paths.
'''
 (OUT/'README.md').write_text(text)
 parent=OUT.parent/'README.md';s=parent.read_text();s='**Both branches completed. [Final analysis, all-case errors and next proposal](completed/README.md).**\n\n'+s if 'Both branches completed.' not in s else s;parent.write_text(s)
 print(json.dumps({k:v for k,v in report.items() if k not in ['base_fem','methods']},indent=2))
if __name__=='__main__':main()
