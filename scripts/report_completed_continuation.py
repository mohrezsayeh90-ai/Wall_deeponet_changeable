"""Reproduce full training plots and paired eight-case checkpoint comparison."""
import csv,hashlib,json,sys
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT/'artifacts/kaggle/main-v1'
NEW=ROOT/'artifacts/kaggle/continue-50k-v1'
OUT=ROOT/'reports/continuation_50k/completed'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    old_report=json.loads((OLD/'development/report.json').read_text());new_report=json.loads((NEW/'development/report.json').read_text())
    old_cases=json.loads((OLD/'development/cases.json').read_text());new_cases=json.loads((NEW/'development/cases.json').read_text())
    assert len(old_cases)==len(new_cases)==8 and new_report['step']==82216
    quantities=['velocity_relative_l2','pressure_relative_l2','pressure_drop_relative','momentum_rms']
    paired=[]
    for a,b in zip(old_cases,new_cases):
        assert a['id']==b['id'] and a['reynolds']==b['reynolds']==10 and a['reference_sha256']==b['reference_sha256']
        for key in ['xy','weights','truth']:
            with np.load(OLD/f"development/{a['index']:04d}.npz",allow_pickle=False) as x,np.load(NEW/f"development/{b['index']:04d}.npz",allow_pickle=False) as y:np.testing.assert_array_equal(x[key],y[key])
        paired.append(dict(index=a['index'],geometry_id=a['id'],family=a['family'],reynolds=b['reynolds'],reference_sha256=a['reference_sha256'],old={k:a['metrics'][k] for k in quantities},new={k:b['metrics'][k] for k in quantities}))
    comparison=dict(old_checkpoint_sha256=old_report['checkpoint_sha256'],new_checkpoint_sha256=new_report['checkpoint_sha256'],old_step=32216,new_step=82216,same_geometries_references_points_weights=True,evaluated=8,expected_development_cases=400,reynolds=10,old_summary=old_report['summary'],new_summary=new_report['summary'],new_gates=new_report['gates'],accepted=False,cases=paired)
    (OUT/'pilot_comparison.json').write_text(json.dumps(comparison,indent=2)+'\n')
    with (OUT/'pilot_comparison.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['dev_index','family','metric','old_32216','new_82216','relative_reduction_percent'])
        for r in paired:
            for key in quantities:w.writerow([r['index'],r['family'],key,r['old'][key],r['new'][key],100*(1-r['new'][key]/max(r['old'][key],1e-16))])
    combined={}
    for path in [OLD/'training.jsonl',NEW/'training.jsonl']:
        for line in path.read_text().splitlines():
            r=json.loads(line);assert r['step'] not in combined;combined[r['step']]=r
    records=[combined[k] for k in sorted(combined)];steps=np.array([r['step'] for r in records]);loss=np.array([r['loss'] for r in records]);assert len(records)==8223 and steps[-1]==82210 and np.isfinite(loss).all()
    rolling=np.r_[np.full(99,np.nan),np.mean(np.lib.stride_tricks.sliding_window_view(loss,100),axis=1)]
    with (OUT/'full_training_history.csv').open('w',newline='') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['step','loss','elapsed_seconds_in_own_run','trailing_100_record_mean'])
        for r,z in zip(records,rolling):w.writerow([r['step'],r['loss'],r['elapsed_seconds'],'' if np.isnan(z) else z])
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,ax=plt.subplots(figsize=(13,6),layout='constrained')
    ax.plot(steps,loss,color='#94b6cd',alpha=.35,lw=.5,label='Actual logged batches');ax.plot(steps,rolling,color='#075985',lw=1.8,label='Trailing arithmetic mean: 100 records')
    ax.axvline(32216,color='#b45309',ls='--',label='Resume from step 32,216');ax.axvline(82216,color='#0f766e',ls='--',label='Target completed: 82,216')
    ax.set(yscale='log',xlabel='Global optimizer step',ylabel='Physics training loss',title='Complete original + continuation training history | no loss invented for unlogged steps');ax.legend();ax.grid(alpha=.2)
    fig.savefig(OUT/'full_training.png',dpi=170);fig.savefig(OUT/'full_training.pdf');plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    for ax,key,title in zip(axes.flat,quantities,['Velocity relative L2 (%)','Pressure relative L2 (%)','Pressure-drop relative error (%)','Momentum RMS']):
        factor=1 if key=='momentum_rms' else 100;x=np.arange(8)
        ax.bar(x-.2,[r['old'][key]*factor for r in paired],.4,color='#94b6cd',label='Step 32,216');ax.bar(x+.2,[r['new'][key]*factor for r in paired],.4,color='#0f766e',label='Step 82,216')
        ax.set(title=title,xlabel='Development case index',xticks=x);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
    fig.suptitle('Paired FEM comparison: same 8 geometries, Re=10, references, points and weights',fontsize=14)
    fig.savefig(OUT/'pilot_comparison.png',dpi=170);fig.savefig(OUT/'pilot_comparison.pdf');plt.close(fig)
    windows=OUT/'windows_10k';windows.mkdir(exist_ok=True)
    (windows/'continuation_snapshot.jsonl').write_bytes((NEW/'training.jsonl').read_bytes())
    (windows/'observation.json').write_text(json.dumps(dict(observed_at_utc=datetime.now(timezone.utc).isoformat(),kaggle_status='COMPLETE',checkpoint_step=82216,last_logged_step=82210,source='Complete downloaded continuation v1 log'),indent=2)+'\n')
    from scripts import plot_training_windows
    plot_training_windows.OUT=windows;sys.argv=['plot_training_windows'];plot_training_windows.main()
    window_readme=windows/'README.md'
    text=window_readme.read_text().replace('Partial-window means are provisional and excluded from rate comparisons.', 'This run is complete: the final 80–90k bin is a shorter terminal segment because the requested target was 82,216; its last logged point is 82,210. Its mean is final for the available logged segment but is excluded from equal-10k rate comparisons.')
    window_readme.write_text(text)
    lines=['# Completed 50,000-update continuation','', '**Verified final checkpoint: 82,216; stop_reason=target_completed. Exactly 50,000 updates after step 32,216.** Training duration was 26,840.254 seconds (7 h 27 min 20 s). The complete output archive is in [continue-50k-v1](../../../artifacts/kaggle/continue-50k-v1/README.md). The old checkpoint and all previous geometry results remain available separately.','', '[Latest training PDF](full_training.pdf) · [Complete-window review](windows_10k/README.md) · [Paired FEM comparison CSV](pilot_comparison.csv) · [Comparison JSON](pilot_comparison.json)','', '![Complete training history](full_training.png)','', 'The two logs contain 8,223 records in total. The first resumed update is 32,217. The final logged update is 82,210 with loss 0.018711606; the checkpoint is at 82,216, whose loss is not separately logged. Geometry/Re/physics points are randomly resampled; x is optimizer steps, not actual epochs. A 100-log-record trailing arithmetic mean shows the trend without deleting raw batches. Training loss is not a validation-error curve.','', '## Paired eight-case FEM pilot','', '![Paired comparison](pilot_comparison.png)','', 'Both checkpoints were evaluated on exactly the same eight development geometries, all at Re=10. Reference SHA-256, geometry identity, coordinates, FEM fields and physical-area weights were verified identical. The following are medians across those eight cases, not a complete 400-case development result.','', '| Metric | Step 32,216 | Step 82,216 |','| --- | ---: | ---: |']
    for key,label in zip(quantities,['Relative velocity L2','Relative pressure L2','Relative pressure drop','Momentum RMS']):
        a=old_report['summary'][key]['median'];b=new_report['summary'][key]['median'];fmt=(lambda x:f'{x:.5f}') if key=='momentum_rms' else (lambda x:f'{x:.3%}')
        lines.append(f'| {label} | {fmt(a)} | {fmt(b)} |')
    lines+=['', 'The velocity-median gate now passes on this pilot. **Family-median and momentum gates still fail**, and complete-split, final-test and comparison requirements remain incomplete. The model remains **unaccepted**. The pilot establishes improvement on these specific cases; it does not establish full-domain accuracy. All original case-level outputs are preserved, including any worsening metric.','', 'The existing 15-channel gallery, five varied examples and five stress examples still use step 32,216. They have **not** been relabeled as results of the new checkpoint or rerun here. Stress results remain extrapolation results, not evidence of accuracy within the training envelope.','', 'Reproduce plots/comparisons from the archived data: `python -m scripts.report_completed_continuation`. This does not retrain or rerun FEM.','']
    (OUT/'README.md').write_text('\n'.join(lines))


if __name__=='__main__':main()
