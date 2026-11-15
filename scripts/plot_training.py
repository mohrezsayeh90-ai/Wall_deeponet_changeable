"""Plot actual Kaggle training records without inventing epochs or validation loss."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--log',type=Path,default=ROOT/'artifacts/kaggle/main-v1/training.jsonl')
    p.add_argument('--output',type=Path,default=ROOT/'reports/training_curves')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    records=[json.loads(s) for s in a.log.read_text().splitlines() if s.strip()]
    status=json.loads((ROOT/'artifacts/kaggle/main-v1/status.json').read_text())
    step=np.array([r['step'] for r in records]);loss=np.array([r['loss'] for r in records])
    terms=np.array([r['terms'] for r in records]);hours=np.array([r['elapsed_seconds'] for r in records])/3600
    grad=np.array([r['gradient_norm'] for r in records])
    assert len(records)>100 and np.all(np.diff(step)>0)
    assert np.isfinite(np.r_[loss,terms.ravel(),hours,grad]).all() and np.all(loss>0)
    np.testing.assert_allclose(loss,terms[:,0]+10*(terms[:,1]+terms[:,2]),rtol=3e-6,atol=1e-7)
    window=100
    def smooth(values):
        return np.r_[np.full(window-1,np.nan),np.median(np.lib.stride_tricks.sliding_window_view(values,window),axis=1)]
    median=smooth(loss)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    def loss_plot(ax,x,label):
        ax.plot(x,loss,color='#94b6cd',alpha=.42,lw=.6,label='Actual logged batch loss')
        ax.plot(x,median,color='#075985',lw=2,label='Trailing median: 100 logged records')
        ax.set(yscale='log',xlabel=label,ylabel='Physics training loss');ax.grid(alpha=.2,which='major');ax.legend(fontsize=9)
    fig,ax=plt.subplots(figsize=(12,7.5));loss_plot(ax,step,'Optimizer update step')
    ax.axvline(status['step'],color='#b45309',ls='--',lw=1,label='Final checkpoint')
    fig.suptitle('Geometry-conditioned PI DeepONet | actual Kaggle training history',fontweight='bold',fontsize=15,y=.97)
    secondary=ax.secondary_xaxis('top',functions=(lambda x:x*32/10000,lambda x:x*10000/32))
    secondary.set_xlabel('Equivalent geometry-pool passes = step × 32 / 10,000 (not actual epochs)',labelpad=9)
    ax.text(.98,.96,f"Logged: steps {step[0]:,}–{step[-1]:,}\nCheckpoint: {status['step']:,} / {status['target_steps']:,}\nStop: wall-clock budget",transform=ax.transAxes,ha='right',va='top',bbox={'facecolor':'white','alpha':.9,'edgecolor':'#cbd5e1'},fontsize=10)
    fig.subplots_adjust(left=.09,right=.98,bottom=.15,top=.79)
    fig.text(.5,.025,'Training batches resample geometries, Reynolds numbers and physics points. No per-step FEM validation loss was recorded.',ha='center',fontsize=9,color='#475569')
    fig.savefig(a.output/'loss_vs_step.png',dpi=180);fig.savefig(a.output/'loss_vs_step.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(14,9),layout='constrained')
    loss_plot(axes[0,0],step,'Optimizer update step');axes[0,0].set_title('A. Total physics loss')
    for value,color,label in [(terms[:,0],'#075985','Momentum'),(10*terms[:,1],'#b45309','10 × outlet Tx squared'),(10*terms[:,2],'#7c3aed','10 × outlet Ty squared')]:
        axes[0,1].plot(step,value,color=color,alpha=.18,lw=.5)
        axes[0,1].plot(step,smooth(value),color=color,lw=1.6,label=label)
    axes[0,1].set(yscale='log',xlabel='Optimizer update step',ylabel='Contribution to total loss',title='B. Loss components (actual weights)');axes[0,1].legend(fontsize=9);axes[0,1].grid(alpha=.2)
    axes[1,0].plot(step,grad,color='#c4b5fd',alpha=.5,lw=.6);axes[1,0].plot(step,smooth(grad),color='#6d28d9',lw=1.6)
    axes[1,0].set(yscale='log',xlabel='Optimizer update step',ylabel='Gradient L2 norm, before clipping',title='C. Gradient norm (raw + trailing median)');axes[1,0].grid(alpha=.2)
    loss_plot(axes[1,1],hours,'Elapsed training time (hours)');axes[1,1].set_title('D. Loss versus elapsed time')
    fig.suptitle('Kaggle v1 training | 3,222 logged records | batch 32 | no new training',fontsize=16,fontweight='bold')
    fig.savefig(a.output/'training_dashboard.png',dpi=170);fig.savefig(a.output/'training_dashboard.pdf');plt.close(fig)
    with (a.output/'training_history.csv').open('w',newline='') as f:
        writer=csv.writer(f,lineterminator='\n');writer.writerow(['step','equivalent_pool_passes_not_epochs','elapsed_hours','loss','trailing_100_record_median','momentum','weighted_traction_x','weighted_traction_y','divergence_squared_diagnostic_not_in_loss','gradient_norm_before_clipping','step_seconds'])
        for i,r in enumerate(records):writer.writerow([int(step[i]),step[i]*32/10000,hours[i],loss[i],'' if np.isnan(median[i]) else median[i],terms[i,0],10*terms[i,1],10*terms[i,2],terms[i,3],grad[i],r['step_seconds']])
    summary=dict(log_sha256=hashlib.sha256(a.log.read_bytes()).hexdigest(),records=len(records),first_logged_step=int(step[0]),last_logged_step=int(step[-1]),checkpoint_step=status['step'],target_steps=status['target_steps'],stop_reason=status['stop_reason'],first_logged_loss=float(loss[0]),last_logged_loss=float(loss[-1]),first_100_records_median=float(np.median(loss[:100])),last_100_records_median=float(median[-1]),checkpoint_geometry_draws=status['step']*32,equivalent_pool_passes_at_checkpoint=status['step']*32/10000,actual_epochs_defined=False,per_step_validation_loss_available=False,log_loss_matches_weighted_terms=True,smoothing='Trailing median of 100 recorded batches, about 1,000 optimizer updates; first 99 smoothed entries unavailable.')
    (a.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    doc=f'''# Actual training history

![Training loss](loss_vs_step.png)

[Download curve PDF](loss_vs_step.pdf) · [Dashboard PDF](training_dashboard.pdf) · [All logged numerical records CSV](training_history.csv) · [Summary](summary.json)

![Training dashboard](training_dashboard.png)

## Steps, not epochs

This is physics-only training. Each optimizer update draws a batch of **32 geometries with replacement** from a 10,000-geometry training pool, resamples Re and draws new interior/outlet collocation points. There is no ordered complete dataset traversal defining an actual epoch. FEM targets are not training labels.

The primary x-axis is the actual optimizer step. The secondary axis is only **equivalent geometry-pool passes = step × 32 / 10,000**. At checkpoint step 32,216 this equals **{summary['equivalent_pool_passes_at_checkpoint']:.4f}** pool passes ({summary['checkpoint_geometry_draws']:,} geometry draws). This does not mean every geometry was visited equally, nor 103 actual training epochs. It does not count a fixed collection of Re/physics-point examples.

## What the records establish

- All **{len(records):,}** log records are plotted, from step 1 through **32,210**. The final checkpoint is at **32,216**; no loss measurement is invented for the last six steps.
- The job stopped at its **17,400-second wall-clock budget**, before its **100,000-step** target.
- First logged loss: **{loss[0]:.8g}**. Last logged loss: **{loss[-1]:.8g}**. These come from different random batches.
- Median of first 100 logged batches: **{summary['first_100_records_median']:.8g}**. Median of last 100: **{summary['last_100_records_median']:.8g}**.
- Raw batches are shown faintly; the dark curve is a trailing median over **100 log records**, roughly 1,000 updates. No missing log records are synthesized. The first 99 median values are absent until a full window exists.
- Total loss = momentum mean-square + **10 × (Tx mean-square + Ty mean-square)**. This identity was checked against every logged record. The divergence-squared value is a diagnostic, not another loss contribution. Gradient norms are the pre-clipping norms from the trainer.
- There is **no validation/FEM error curve versus training step** in these logs. The collected eight-case FEM evaluation and later 15-channel examples evaluate the final checkpoint; they cannot be relabeled as training-time validation histories. Lower physics loss alone does not certify solution accuracy or model acceptance.

The plotted records are from the actual archived Kaggle v1 log, not a new training run. Original log SHA-256: `{summary['log_sha256']}`.

## Reproduce

From the repository root in its documented CPU environment:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/workspace/.cache/matplotlib XDG_CACHE_HOME=/workspace/.cache .venv/bin/python -m scripts.plot_training
```

The original [JSONL log](../../artifacts/kaggle/main-v1/training.jsonl), [final status](../../artifacts/kaggle/main-v1/status.json) and [collection report](../KAGGLE_COMPLETED_RUN.md) remain unchanged.
'''
    (a.output/'README.md').write_text(doc)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
