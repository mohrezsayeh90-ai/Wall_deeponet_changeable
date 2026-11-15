"""Compare arithmetic mean training loss in fixed 10,000-update windows."""
import argparse,csv,hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/training_windows_10k'


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--live-log',type=Path);a=p.parse_args();OUT.mkdir(exist_ok=True)
    if a.live_log:
        captured={}
        for line in a.live_log.read_text().splitlines():
            try:r=json.loads(json.loads(line).get('data',''))
            except (ValueError,TypeError):continue
            if isinstance(r,dict) and 'step' in r:captured[r['step']]=r
        assert captured
        (OUT/'continuation_snapshot.jsonl').write_text(''.join(json.dumps(captured[k])+'\n' for k in sorted(captured)))
        (OUT/'observation.json').write_text(json.dumps(dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),kaggle_status='RUNNING',last_logged_step=max(captured),source='mohrezsayeh/geometry-pi-deeponet-continue-50k live numerical log snapshot'),indent=2)+'\n')
    sources=[ROOT/'artifacts/kaggle/main-v1/training.jsonl',OUT/'continuation_snapshot.jsonl'];combined={}
    for source in sources:
        for line in source.read_text().splitlines():
            r=json.loads(line)
            if r['step']>=30000:
                if r['step'] in combined:assert combined[r['step']]==r
                combined[r['step']]=r
    rows=[combined[k] for k in sorted(combined)];steps=np.array([r['step'] for r in rows]);loss=np.array([r['loss'] for r in rows]);latest=int(steps[-1])
    assert steps[0]==30000 and np.all(np.diff(steps)>0) and np.isfinite(loss).all() and np.all(loss>0)
    windows=[];previous=None
    for start in range(30000,latest+1,10000):
        end=start+10000;selected=loss[(steps>=start)&(steps<end)];full=latest>=end
        if not len(selected):continue
        mean=float(np.mean(selected));median=float(np.median(selected));w=dict(start_inclusive=start,end_exclusive=end,last_observed_step=int(steps[(steps>=start)&(steps<end)][-1]),complete=full,logged_records=len(selected),arithmetic_mean=mean,median=median,min=float(selected.min()),p10=float(np.quantile(selected,.1)),p90=float(np.quantile(selected,.9)),reduction_vs_previous_complete_percent=None,log_decrease_per_10000_steps=None)
        if full and previous is not None:
            w['reduction_vs_previous_complete_percent']=100*(1-mean/previous)
            w['log_decrease_per_10000_steps']=float(np.log(previous/mean))
        if full:previous=mean
        windows.append(w)
    smoothed=np.r_[np.full(99,np.nan),np.mean(np.lib.stride_tricks.sliding_window_view(loss,100),axis=1)]
    report=dict(start_step=30000,last_logged_step=latest,window_steps=10000,smoothing_records=100,smoothing_approx_steps=1000,actual_epochs=False,equivalent_pool_passes_per_window=32,mean_definition='Arithmetic mean of available logged batch losses; not a loss evaluated every step on a fixed validation set.',source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources},windows=windows)
    (OUT/'window_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    with (OUT/'window_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(windows[0]),lineterminator='\n');w.writeheader();w.writerows(windows)
    with (OUT/'history.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['step','loss','trailing_100_record_mean'])
        for x,y,z in zip(steps,loss,smoothed):w.writerow([x,y,'' if np.isnan(z) else z])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,axes=plt.subplots(3,1,figsize=(13,13),layout='constrained',height_ratios=[1.6,1,1])
    ax=axes[0];ax.plot(steps,loss,color='#94b6cd',alpha=.35,lw=.5,label='Actual logged batches');ax.plot(steps,smoothed,color='#075985',lw=1.8,label='Trailing arithmetic mean: 100 logged records')
    for i,w in enumerate(windows):
        finish=min(w['end_exclusive'],latest);color='#0f766e' if w['complete'] else '#b45309'
        ax.hlines(w['arithmetic_mean'],w['start_inclusive'],finish,color=color,lw=3,label=('Complete 10k-window means' if i==0 else 'Partial-window mean' if not w['complete'] else None))
    ax.axvline(32216,color='#64748b',ls='--',lw=1,label='Resume boundary: step 32,216');ax.set(yscale='log',xlabel='Optimizer step',ylabel='Physics training loss (log scale)',title='A. Logged loss, rolling mean and fixed-window means');ax.legend(fontsize=9,ncol=2);ax.grid(alpha=.2)
    labels=[f"{w['start_inclusive']/1000:.0f}–{w['end_exclusive']/1000:.0f}k"+('' if w['complete'] else '\nPARTIAL') for w in windows]
    bars=axes[1].bar(range(len(windows)),[w['arithmetic_mean'] for w in windows],color=['#0f766e' if w['complete'] else '#fbbf24' for w in windows])
    for b,w in zip(bars,windows):
        if not w['complete']:b.set_hatch('//')
        axes[1].text(b.get_x()+b.get_width()/2,b.get_height(),f"{w['arithmetic_mean']:.5f}\nn={w['logged_records']}",ha='center',va='bottom',fontsize=10)
    axes[1].set(xticks=range(len(windows)),xticklabels=labels,ylabel='Arithmetic mean loss',title='B. Mean of recorded losses in each 10,000-update window',ylim=(0,max(w['arithmetic_mean'] for w in windows)*1.3));axes[1].grid(axis='y',alpha=.2)
    comparisons=[w for w in windows if w['reduction_vs_previous_complete_percent'] is not None]
    if comparisons:
        rate=[w['reduction_vs_previous_complete_percent'] for w in comparisons]
        axes[2].bar(range(len(rate)),rate,color=['#0f766e' if v>=0 else '#be123c' for v in rate])
        axes[2].set(xticks=range(len(rate)),xticklabels=[f"{w['start_inclusive']/1000:.0f}–{w['end_exclusive']/1000:.0f}k vs previous 10k" for w in comparisons])
        for i,v in enumerate(rate):axes[2].annotate(f'{v:.2f}%',(i,v),xytext=(0,6 if v>=0 else -16),textcoords='offset points',ha='center')
        axes[2].margins(y=.3)
    else:axes[2].text(.5,.5,'Not enough complete windows for comparison',transform=axes[2].transAxes,ha='center')
    axes[2].axhline(0,color='#64748b',lw=1);axes[2].set(ylabel='Reduction in mean loss (%)',title='C. Relative improvement per 10k: 100 × (1 − new mean / previous mean)');axes[2].grid(axis='y',alpha=.2)
    fig.suptitle(f'Actual training progress from step 30,000 through {latest:,} | 10k-update windows',fontsize=16,fontweight='bold')
    fig.savefig(OUT/'training_windows.png',dpi=170);fig.savefig(OUT/'training_windows.pdf');plt.close(fig)
    lines=['# Training slowdown review: fixed 10,000-update windows','', '![Training window comparison](training_windows.png)','', '[PDF](training_windows.pdf) · [Window CSV](window_summary.csv) · [Exact summary and hashes](window_summary.json) · [All plotted records](history.csv) · [Live snapshot observation](observation.json)','', '**These are optimizer steps, not actual epochs.** Each 10,000-step window equals 32 equivalent geometry-pool passes (batch 32 / pool 10,000), not 100 epochs. The rolling display uses 100 logged records, approximately 1,000 steps. A hypothetical 100 equivalent pool passes would span 31,250 steps.','', 'Original-run logs and resumed-run numerical logs are merged by global step without resetting the x-axis. The resume boundary is marked. No training or validation computation is rerun. The snapshot stops at the last actual recorded step; it is not a live-updating chart.','', '## Arithmetic means','', '| Window (steps) | Complete? | Records | Mean loss | Median loss | Reduction from previous full window |','| --- | --- | ---: | ---: | ---: | ---: |']
    for w in windows:
        reduction='—' if w['reduction_vs_previous_complete_percent'] is None else f"{w['reduction_vs_previous_complete_percent']:.2f}%"
        lines.append(f"| {w['start_inclusive']:,} ≤ step < {w['end_exclusive']:,} | {'Yes' if w['complete'] else 'No; partial'} | {w['logged_records']} | {w['arithmetic_mean']:.6f} | {w['median']:.6f} | {reduction} |")
    lines+=['','Averages are over actual **logged batches**, normally recorded every 10 steps, not all individual optimizer steps. The extra first resumed update at 32,217 is retained; no missing values are filled. All batches are equally weighted in this arithmetic mean. Random geometry, Re and collocation sampling create batch variability.','', 'Only complete adjacent windows are used for the percent-reduction chart. Positive values mean lower mean loss; a smaller positive value means slower relative improvement over that pair of 10k windows. Negative values mean a higher window mean. The JSON also contains ln(previous mean / new mean), the decrease in log mean per 10k steps. Partial-window means are provisional and excluded from rate comparisons. This is descriptive training-loss progress, not a validation/FEM accuracy curve or a statistical guarantee of convergence.','', 'Reproduce the frozen snapshot: `python -m scripts.plot_training_windows`. To update, capture fresh live logs using the existing collector and pass `--live-log path/to/live-log.jsonl`.','']
    (OUT/'README.md').write_text('\n'.join(lines));print(json.dumps(report,indent=2))


if __name__=='__main__':main()
