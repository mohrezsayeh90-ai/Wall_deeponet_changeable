"""Same-case archived100k predictions versus current independent FEM metrics."""
from pathlib import Path
import json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'reports/molab_soap_highlr50k/versus100k';oldroot=ROOT/'reports/molab_adam_100k/completed';read=lambda p:json.loads(p.read_text());new=read(O/'current_fem.json');summary=read(oldroot/'summary.json');hard={x['case'] for x in new['cases'] if x['hard']};population=read(ROOT/'reports/adam_next50k_comparison/selection.json')['population'];models={'adam100k_final287400':[],'adam100k_probe_best282400':[],'soap_cosine_partial342400':[],'new_highlr'+str(new['step']):new['cases']};verified=[]
for r in summary['cases']:
 case=str(r['case_id']);p=oldroot/r['directory']/'fields.npz';orig=next(x for x in population if str(x['case_id'])==case)
 with np.load(p,allow_pickle=False) as z,np.load(ROOT/'reports'/orig['group']/orig['directory']/'fields.npz',allow_pickle=False) as ref:
  for k in ['xy','weights','fem','coeff']:np.testing.assert_array_equal(z[k],ref[k])
  w=z['weights'];truth=z['fem']
  def rel(a,b):
   ww=w.reshape((-1,)+(1,)*(a.ndim-1));den=float(np.sum(ww*b*b));return float(np.sqrt(np.sum(ww*(a-b)**2)/den)) if den>1e-12 else None
  for tag,key in [('adam100k_final287400','final'),('adam100k_probe_best282400','best')]:
   pred=z[key];v=dict(case=case,hard=case in hard,velocity_l2=rel(pred[:,:2],truth[:,:2]),pressure_l2=rel(pred[:,2],truth[:,2]),u_l2=rel(pred[:,0],truth[:,0]),v_l2=rel(pred[:,1],truth[:,1]));models[tag].append(v)
   np.testing.assert_allclose(v['velocity_l2'],r['metrics'][key]['velocity_relative_l2'],rtol=1e-8,atol=1e-12)
   np.testing.assert_allclose(v['pressure_l2'],r['metrics'][key]['pressure_relative_l2'],rtol=1e-8,atol=1e-12)
 verified.append(case)
partial=read(ROOT/'artifacts/molab/soap-highlr50k-v1/fem/fem_342400.json');models['soap_cosine_partial342400']=partial['cases'];out={'current_step':new['step'],'current_updates':new['offset'],'same_fields_verified_case_ids':verified,'missing_100k_SOAP_final422400':True,'models':{},'comparisons':{}};csvrows=[]
for name,rr in models.items():
 out['models'][name]={}
 for group,rows in [('all20',rr),('same_curvature12',[x for x in rr if x['case'] in hard])]:
  out['models'][name][group]={}
  for field in ['velocity_l2','pressure_l2','u_l2','v_l2']:
   vals=[x[field] for x in rows if x[field] is not None];out['models'][name][group][field]={'median':float(np.median(vals)),'max':float(np.max(vals)),'count':len(vals)}
 csvrows.extend(dict(model=name,**r) for r in rr)
current=models['new_highlr'+str(new['step'])]
for name,rr in models.items():
 if name.startswith('new_'):continue
 by={x['case']:x for x in rr};out['comparisons'][name]={}
 for group,selected in [('all20',current),('same_curvature12',[x for x in current if x['case'] in hard])]:
  out['comparisons'][name][group]={}
  for field in ['velocity_l2','pressure_l2','u_l2','v_l2']:
   n=out['models']['new_highlr'+str(new['step'])][group][field]['median'];b=out['models'][name][group][field]['median']
   out['comparisons'][name][group][field]={'median_relative_reduction_percent':100*(1-n/b),'improved_cases':sum(x[field]<by[x['case']][field] for x in selected if x[field] is not None and by[x['case']][field] is not None)}
(O/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
with (O/'per_case_errors.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(csvrows[0]),lineterminator='\n');w.writeheader();w.writerows(csvrows)
labels=['Adam100k final','Adam best probe','SOAP partial20k',f"New high-LR{new['offset']/1000:g}k"];fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,(group,field) in zip(axs.flat,[(g,f) for g in ['all20','same_curvature12'] for f in ['velocity_l2','pressure_l2']]):
 vals=[100*out['models'][n][group][field]['median'] for n in models];bars=ax.bar(labels,vals,color=['#94a3b8','#60a5fa','#f59e0b','#0d9488']);ax.bar_label(bars,fmt='%.3f%%',padding=3);ax.set(title=group+' '+field,ylabel='Median relative L2 (%)',ylim=(0,max(vals)*1.2));ax.tick_params(axis='x',rotation=12);ax.grid(axis='y',alpha=.2)
fig.suptitle('Same20 FEM cases; unequal training lineage/budgets, not an optimizer ablation');fig.savefig(O/'comparison.png',dpi=170);fig.savefig(O/'comparison.pdf');plt.close(fig)
print(json.dumps(out['models'],indent=2));print(json.dumps(out['comparisons'],indent=2))
