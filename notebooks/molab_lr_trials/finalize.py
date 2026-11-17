"""Wait for all trials, then publish an atomic comparison bundle and ranking."""
import argparse,json,time,csv,zipfile
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);root=ap.parse_args().root
    while True:
        p=root/'suite-status.json'
        try:state=json.loads(p.read_text())
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(10);continue
        if state['phase']=='failed':return
        if state['phase']=='completed':break
        time.sleep(10)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    m=json.loads((root/'manifest.json').read_text());rows=[]
    fig,axes=plt.subplots(2,1,figsize=(12,9),layout='constrained')
    for c in m['cases']:
        folder=root/'cases'/c['id'];r=[json.loads(l) for l in (folder/'training.jsonl').read_text().splitlines()]
        probe=[json.loads(l) for l in (folder/'probe.jsonl').read_text().splitlines()];fem=json.loads((folder/'fem_pilot.json').read_text())
        y=np.array([t['loss'] for t in r]);x=[t['step']-m['base_step'] for t in r]
        smooth=[y[max(0,i-99):i+1].mean() for i in range(len(y))]
        axes[0].plot(x,smooth,label=c['label']);axes[1].plot([t['offset'] for t in probe],[t['loss'] for t in probe],'o-',label=c['label'])
        rows.append({'case':c['id'],'label':c['label'],'final_step':r[-1]['step'],'last_300_record_mean_loss':float(y[-300:].mean()),
            'last_300_record_median_loss':float(np.median(y[-300:])), 'initial_probe_loss':probe[0]['loss'],'final_probe_loss':probe[-1]['loss'],
            'probe_reduction_percent':100*(1-probe[-1]['loss']/probe[0]['loss']),'fem_velocity_median':fem['median_velocity_l2'],'fem_pressure_median':fem['median_pressure_l2']})
    for ax in axes:ax.grid(alpha=.2);ax.legend();ax.set(xlabel='Updates from identical baseline checkpoint',ylabel='Loss',yscale='log')
    axes[0].set_title('Training mean over up to 100 logged records');axes[1].set_title('Fixed development probe objective')
    fig.savefig(root/'comparison.png',dpi=180);fig.savefig(root/'comparison.pdf');plt.close(fig)
    rows.sort(key=lambda r:r['final_probe_loss'])
    report={'ranked_by':'final fixed-development-probe objective; no automatic model promotion','base_step':m['base_step'],'cases':rows}
    (root/'comparison.json').write_text(json.dumps(report,indent=2))
    with (root/'comparison.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (root/'README.md').write_text('# Paired learning-rate comparison\n\nAll four trials start from identical parameters, Adam moments and RNG and run 15,000 updates. Ranking uses the fixed dev objective, with matched FEM pilot errors retained separately. No checkpoint is automatically promoted. See comparison.csv, comparison.json and comparison.png.\n')
    tmp=root/'final-results.tmp.zip'
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for p in sorted(root.rglob('*')):
            if p.is_file() and p.suffix!='.zip' and 'project' not in p.relative_to(root).parts:z.write(p,arcname=p.relative_to(root))
    with zipfile.ZipFile(tmp) as z:assert z.testzip() is None
    tmp.replace(root/'lr_trials_results.zip');(root/'comparison_ready.json').write_text(json.dumps({'ready':True,'ranked_cases':[r['case'] for r in rows]}))

if __name__=='__main__':main()
