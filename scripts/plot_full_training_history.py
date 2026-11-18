"""Plot the accepted Adam lineage without connecting missing history or parallel trials."""
from pathlib import Path
import argparse,csv,hashlib,json
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path,default=ROOT/'reports/full_training_history');ap.add_argument('--rebuild-data',action='store_true');a=ap.parse_args();out=a.output;out.mkdir(parents=True,exist_ok=True)
 if a.rebuild_data:
  inputs=[('kaggle_main','artifacts/kaggle/main-v1/training.jsonl',0,32216),('kaggle_plus50k','artifacts/kaggle/continue-50k-v1/training.jsonl',32216,82216),('kaggle_next50k','artifacts/kaggle/adam-next50k-v1/training.jsonl',82216,132216),('selected_cosine_trial','runs/verified-molab-lr/cases/cosine_1e4_2e5/training.jsonl',172400,187400),('molab_cosine100k','artifacts/molab/adam-100k-v1/training.jsonl',187400,287400)]
  rows=[];sources=[]
  for phase,file,lo,hi in inputs:
   p=ROOT/file;d=[json.loads(l) for l in p.read_text().splitlines() if l.strip()];d=[x for x in d if lo<x['step']<=hi];assert all(x['loss']>0 and np.isfinite(x['loss']) for x in d)
   if all('terms' in x for x in d):np.testing.assert_allclose([x['loss'] for x in d],[x['terms'][0]+10*(x['terms'][1]+x['terms'][2]) for x in d],rtol=3e-6,atol=1e-7)
   rows.extend(dict(step=x['step'],loss=x['loss'],phase=phase,kind='raw_batch',window_start='',window_end='',source=file) for x in d)
   sources.append(dict(phase=phase,path=file,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),records=len(d),first_step=d[0]['step'],last_step=d[-1]['step']))
  p=out/'current_run_observation.json';d=json.loads(p.read_text());rows.extend(dict(step=x[0],loss=x[1],phase='molab_adaptive100k',kind='raw_batch',window_start='',window_end='',source='current_run_observation.json') for x in d['rows']);sources.append(dict(phase='molab_adaptive100k',path='current_run_observation.json',sha256=hashlib.sha256(p.read_bytes()).hexdigest(),records=len(d['rows']),first_step=d['rows'][0][0],last_step=d['rows'][-1][0],observed_utc=d['observed_utc'],status=d['status']))
  p=ROOT/'reports/molab_adam_1h/loss_audit/molab.json';d=json.loads(p.read_text());derived=d['hundred_step_means'];rows.extend(dict(step=int(s),loss=y,phase='molab_first_hour',kind='mean_100_steps',window_start=int(s)-100,window_end=int(s),source='reports/molab_adam_1h/loss_audit/molab.json') for s,y in derived)
  first=d['first_record'];rows.append(dict(step=first['step'],loss=first['loss'],phase='molab_first_hour',kind='raw_batch',window_start='',window_end='',source='reports/molab_adam_1h/loss_audit/molab.json'))
  sources.append(dict(phase='molab_first_hour',path='reports/molab_adam_1h/loss_audit/molab.json',sha256=hashlib.sha256(p.read_bytes()).hexdigest(),aggregate_records=len(derived),raw_records=1,aggregate_end=derived[-1][0],missing_interval=[147216,172400],note='Saved 100-step means cover only 132216–147216; raw full first-hour log unavailable in accessible storage. No synthetic points inserted.'))
  rows.sort(key=lambda x:(x['step'],x['kind']));raw=[x['step'] for x in rows if x['kind']=='raw_batch'];assert len(raw)==len(set(raw)),'Duplicate raw steps from parallel branches'
  with (out/'training_history.csv').open('w',newline='') as f:
   w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
  (out/'sources.json').write_text(json.dumps(sources,indent=2)+'\n')
 with (out/'training_history.csv').open() as f:rows=list(csv.DictReader(f))
 for x in rows:x['step']=int(x['step']);x['loss']=float(x['loss'])
 raw=[x for x in rows if x['kind']=='raw_batch'];last=max(x['step'] for x in raw);first=min(x['step'] for x in raw)
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
 fig,axs=plt.subplots(2,1,figsize=(15,10),layout='constrained',gridspec_kw={'height_ratios':[1.3,1]})
 phases=list(dict.fromkeys(x['phase'] for x in rows));smooth=[]
 for ax in axs:
  labelled=False
  for phase in phases:
   rr=[x for x in raw if x['phase']==phase];xx=np.array([x['step'] for x in rr]);yy=np.array([x['loss'] for x in rr]);ax.plot(xx,yy,color='#76abc6',alpha=.24,lw=.45,label='Actual logged batch loss' if not labelled else None);labelled=True
   if len(rr)>=100:
    sy=np.convolve(yy,np.ones(100)/100,mode='valid');sx=xx[99:];ax.plot(sx,sy,color='#6d28d9',lw=1.8,label='Trailing mean: 100 logged records (~1,000 updates)' if phase=='kaggle_main' else None)
    if ax is axs[0]:smooth.extend(dict(step=int(s),mean=float(y),phase=phase,records=100) for s,y in zip(sx,sy))
  ag=[x for x in rows if x['kind']=='mean_100_steps'];xx=np.array([x['step'] for x in ag]);yy=np.array([x['loss'] for x in ag]);sy=np.convolve(yy,np.ones(10)/10,mode='valid');ax.plot(xx[9:],sy,'--',color='#0369a1',lw=1.8,label='First molab: saved means, smoothed over 1,000 updates')
  ax.axvspan(147216,172400,facecolor='#e2e8f0',alpha=.65,hatch='///',edgecolor='#94a3b8',lw=0,label='Detailed loss history unavailable')
  for s in [32216,82216,132216,172400,187400,287400]:ax.axvline(s,color='#64748b',ls=':',lw=.9,alpha=.6)
  ax.grid(alpha=.18);ax.set(xlabel='Global accepted Adam update step (not epochs)',ylabel='Physics training loss')
 axs[0].set(yscale='log',xlim=(0,last+2000),title='Entire training history — logarithmic loss scale')
 axs[0].set_title('Entire training history — logarithmic loss scale',fontsize=13,pad=28)
 axs[0].legend(fontsize=9,loc='upper right',framealpha=.95)
 for lo,hi,label in [(0,32216,'Kaggle 1'),(32216,82216,'Kaggle +50k'),(82216,132216,'Kaggle +50k'),(132216,172400,'Molab 1h'),(172400,187400,'LR trial'),(187400,287400,'Cosine 100k'),(287400,last,'Adaptive')]:
  axs[0].text((lo+hi)/2,1.015,label,transform=axs[0].get_xaxis_transform(),ha='center',va='bottom',fontsize=8)
 axs[1].set(xlim=(100000,last+2000),ylim=(.005,.035),title='Later training — linear loss scale (vertical range 0.005–0.035)')
 axs[1].text(.99,.97,f'Last logged step: {last:,}',ha='right',va='top',transform=axs[1].transAxes,bbox=dict(facecolor='white',alpha=.9,edgecolor='#cbd5e1'))
 fig.suptitle(f'Geometry-conditioned PI DeepONet | accepted Adam lineage: step {first:,}–{last:,}',fontweight='bold',fontsize=16)
 fig.savefig(out/'loss_from_start.png',dpi=180);fig.savefig(out/'loss_from_start.pdf');plt.close(fig)
 with (out/'smoothed_history.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=['step','mean','phase','records']);w.writeheader();w.writerows(smooth)
 summary=dict(first_logged_step=first,last_logged_step=last,raw_batch_records=len(raw),aggregate_records=sum(x['kind']!='raw_batch' for x in rows),missing_loss_interval=[147216,172400],smoothing='Per-phase trailing mean of 100 logged records, approximately 1000 updates. Aggregate-only segment uses 10 saved 100-step means. No cross-gap smoothing.',lineage='Accepted cosine trial only; other LR branches and rejected LBFGS excluded.',axes='Global accepted Adam updates, not epochs.',last_logged_loss=raw[-1]['loss'])
 (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 (out/'README.md').write_text(f'''# Training loss from the beginning to step {last:,}

![Entire Adam training history](loss_from_start.png)

[PDF](loss_from_start.pdf) · [All plotted values CSV](training_history.csv) · [Smoothed values CSV](smoothed_history.csv) · [Source hashes and coverage](sources.json) · [Summary](summary.json).

The chart follows the actual accepted checkpoint lineage: Kaggle main (1–32,216), Kaggle +50k (to82,216), next +50k (to132,216), first molab segment (to172,400), the selected cosine LR trial (to187,400), cosine100k (to287,400), then the current adaptive continuation. Parallel LR branches and rejected L-BFGS trajectories do not belong to this lineage and are not concatenated.

- Horizontal axes show **Adam updates, not epochs**: batches resample geometries, Reynolds numbers and collocation points.
- The upper panel uses logarithmic loss; the lower panel shows the later history on a linear scale with an explicitly limited vertical range.
- Faint blue curves show every available raw logged batch. Purple curves are arithmetic means over the preceding100 logged records (approximately1,000 updates), independently within each phase. The first99 smoothed points of each phase are omitted. Averages never cross missing history or a phase boundary.
- **Coverage limitation:** first molab raw history is unavailable in accessible storage. Saved arithmetic means over100-update windows cover132,216–147,216 and are drawn as a separate dashed curve after averaging10 such windows. **147,216–172,400 has no plotted loss values** and is shown as a hatched band. This is missing historical data, not a pause in training. No loss is inferred from checkpoint weights, endpoints or screenshots.
- {len(raw):,} raw logged values and150 aggregate values are included. Snapshot coverage ends at **{last:,}**; no measurement is invented for unlogged steps or later live progress. Current state can advance during collection.
- Fixed development probes are excluded from the training-loss curve because their cohorts changed. The previous [same-probe/FEM audit](../molab_adaptive_100k/diagnostic/) remains the appropriate checkpoint comparison.
- The selected LR-trial raw log is preserved in the verified [lr-trials archive](../../artifacts/molab/lr-trials-v1/). The consolidated CSV and current-run observation make this chart reproducible without reconnecting to molab. A source hash identifies each input.

Re-render from the saved consolidated CSV:

```bash
MPLCONFIGDIR=/tmp/mpl-full-history .venv/bin/python -m scripts.plot_full_training_history
```

Use `--rebuild-data` only when all archived upstream logs have been restored to the paths recorded in the script. No training was launched or modified to produce this chart.
''')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
