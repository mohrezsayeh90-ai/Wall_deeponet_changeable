"""Generate an equal-budget screening report after the supervisor finishes."""
from pathlib import Path
import json,time,csv
R=Path('/home/marimo/wall_soap_sweep50k_v1')
while not (R/'comparison.json').exists():
 if (R/'failure.json').exists():raise SystemExit('Trial failed; keep partial results and inspect failure.json')
 time.sleep(15)
c=json.loads((R/'comparison.json').read_text());rows=[]
for name,x in c['screens'].items():
 row={'case':name,'updates':5000,'control':x['final_probe']['control'],'audit':x['final_probe']['audit']}
 for group in ['saved20','pilot8']:
  for field in ['velocity_l2_median','pressure_l2_median']:row[group+'_'+field]=x['final_fem'][group][field]
 rows.append(row)
with (R/'common5k.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
import numpy as np
rng=[]
for name in c['screens']:
 with np.load(R/name/'continuation/checkpoint.npz',allow_pickle=False) as z:rng.append(json.loads(str(z['metadata']))['rng_state'])
assert all(x==rng[0] for x in rng),'Screen data streams must end identically'
c['matched_screen_rng_endpoints']=True;c['minimum_control_at5k']=min(rows,key=lambda x:x['control'])['case'];c['promotion']='No200k run started; compare FEM regressions and validation coverage before choosing.'
(R/'comparison.json').write_text(json.dumps(c,indent=2))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axs=plt.subplots(2,1,figsize=(11,8),layout='constrained')
for name in c['screens']:
 p=[json.loads(l) for l in (R/name/'continuation/probe.jsonl').read_text().splitlines()]
 axs[0].plot([x['offset'] for x in p],[x['control'] for x in p],'o-',label=name)
 axs[1].plot([x['offset'] for x in p],[x['audit'] for x in p],'o-',label=name)
for ax in axs:ax.set(yscale='log',xlabel='New updates from identical342400 state');ax.grid(alpha=.2);ax.legend(fontsize=8)
axs[0].set_ylabel('Fixed control loss');axs[1].set_ylabel('Fixed audit loss')
fig.savefig(R/'comparison.png',dpi=170);fig.savefig(R/'comparison.pdf');plt.close(fig)
text='# Completed50k SOAP experiment\n\nFive5000-update screens plus25000 further constant1e-4 updates completed. Constant1e-4 has30000 total updates. All screen RNG endpoints exactly match.\n\nMinimum5k control: **'+c['minimum_control_at5k']+'**. This is not a universal accuracy winner; inspect the FEM columns and every-case data. No200k training was launched automatically. The cosine test compresses a decay into5000 updates and does not validate that horizon for200k. All28 FEM cases and both probes are repeatedly used development cohorts; FEM coverage is Re10/30, not fullRe10–100.\n\n![Equal-budget fixed probes](comparison.png)\n\n[CSV](common5k.csv) · [Full results](comparison.json). Constant30k is a longer-budget observation, not an equal-cost competitor to the5k screens.\n'
(R/'analysis.md').write_text(text)
(R/'analysis_ready.json').write_text(json.dumps({'complete':True,'total_updates':50000,'matched_rng_endpoints':True}))