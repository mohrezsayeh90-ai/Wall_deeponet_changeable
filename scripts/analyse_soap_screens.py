"""Compare completed equal-budget SOAP screens without resuming any training."""
from pathlib import Path
import json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];A=ROOT/'artifacts/molab/soap-sweep50k-v1';O=ROOT/'reports/molab_soap_sweep50k/comparison';O.mkdir(exist_ok=True)
names=['constant_2e5','constant_5e5','constant_1e4','constant_2e4','cosine_1e4_2e5'];labels=['2e-5','5e-5','1e-4','2e-4','Cosine1e-4→2e-5']
read=lambda p:json.loads(p.read_text());rows=lambda p:[json.loads(x) for x in p.read_text().splitlines()]
base=read(A/'start_fem.json');out={'scope':'Five paired5000-update development screens; extension separate; no new run started','methods':{},'baseline_fem':base,'verified_snapshots':[]};rngs=[];table=[];case_rows=[]
fig,axs=plt.subplots(2,1,figsize=(10,7),layout='constrained')
for name,label in zip(names,labels):
 d=A/name;cfg=read(d/'run_config.json');p=rows(d/'continuation/probe.jsonl');f=read(d/'continuation/final_fem.json');done=read(d/'continuation/completed.json');assert done['completed'] and done['new_updates']==5000
 assert cfg['base_sha256']=='b49cc0e7c38809a5fe9f9913ae24c35563c6d8c5287af0bfe0aec283a7522018'
 path=d/'continuation/snapshots/checkpoint_347400.npz';ready=read(path.with_suffix('.ready.json'));assert path.stat().st_size==ready['bytes'];assert hashlib.sha256(path.read_bytes()).hexdigest()==ready['sha256']
 with np.load(path,allow_pickle=False) as z:
  meta=json.loads(str(z['metadata']));assert meta['soap_state_present'] and meta['step']==347400;assert int(z['soap_0'])==60001;rngs.append(meta['rng_state'])
 out['verified_snapshots'].append(dict(case=name,**ready))
 v={'config':cfg,'endpoint':p[-1],'final3_probe_mean':{k:float(np.mean([x[k] for x in p[-3:]])) for k in ['control','audit']},'relative_probe_change_percent':{k:100*(p[-1][k]/p[0][k]-1) for k in ['control','audit']},'fem':f,'per_case_improvements':{}}
 for group,sl in [('saved20',slice(0,20)),('pilot8',slice(20,28))]:
  v['per_case_improvements'][group]={field:sum(x[field]<y[field] for x,y in zip(f['cases'][sl],base['cases'][sl])) for field in ['velocity_l2','pressure_l2']}
 out['methods'][name]=v
 table.append(dict(case=name,control=p[-1]['control'],audit=p[-1]['audit'],control_last3_mean=v['final3_probe_mean']['control'],audit_last3_mean=v['final3_probe_mean']['audit'],**{group+'_'+k:f[group][k] for group in ['saved20','pilot8'] for k in ['velocity_l2_median','pressure_l2_median','velocity_l2_max','pressure_l2_max']}))
 case_rows.extend(dict(case_method=name,**x) for x in f['cases'])
 for ax,key in zip(axs,['control','audit']):ax.plot([x['offset'] for x in p],[x[key] for x in p],'o-',label=label);ax.set(ylabel=key+' physics loss',xlabel='Paired updates from identical342400 state');ax.grid(alpha=.2)
assert all(x==rngs[0] for x in rngs);out['identical_screen_rng_endpoints']=True
for ax in axs:ax.legend(fontsize=8)
fig.savefig(O/'fixed_probes.png',dpi=170);fig.savefig(O/'fixed_probes.pdf');plt.close(fig)
fig,axs=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for ax,(group,field) in zip(axs.flat,[(g,f) for g in ['saved20','pilot8'] for f in ['velocity_l2_median','pressure_l2_median']]):
 values=[base[group][field]]+[out['methods'][n]['fem'][group][field] for n in names]
 b=ax.bar(['Base']+labels,np.array(values)*100,color=['#94a3b8']+['#0d9488','#eab308','#3b82f6','#ef4444','#8b5cf6']);ax.bar_label(b,fmt='%.3f',padding=3,fontsize=8);ax.tick_params(axis='x',rotation=20);ax.set(ylabel='Median relative L2 (%)',title=group+' '+field,ylim=(0,max(values)*125));ax.grid(axis='y',alpha=.2)
fig.savefig(O/'fem_comparison.png',dpi=170);fig.savefig(O/'fem_comparison.pdf');plt.close(fig)
for file,data in [('common5k.csv',table),('per_case_fem.csv',case_rows)]:
 with (O/file).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(data[0]),lineterminator='\n');w.writeheader();w.writerows(data)
ext=A/'constant_1e4_extend25k/continuation';p=rows(ext/'probe.jsonl');last=rows(ext/'training.jsonl')[-1]['step'];ff=read(ext/'fem_352400.json');out['extension']={'scope':'Unequal-budget observation; paused, not a completed25k extension','last_logged_step':last,'additional_logged_updates':last-347400,'latest_fixed_probe':p[-1],'fem_at_total10000_updates':ff};out['budget']={'screens':25000,'extension_at_least':last-347400,'total_at_least':25000+last-347400,'remaining_at_most':50000-(25000+last-347400),'uncertainty':'Logs every10 updates; live paused state may include up to9 further unlogged updates.'}
(O/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'matched_rng':True,'budget':out['budget'],'rows':table},indent=2))
