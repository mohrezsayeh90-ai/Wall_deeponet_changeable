"""Select a globally frozen checkpoint using development FEM only."""
import argparse,json,shutil
from pathlib import Path
from types import SimpleNamespace
from geometry_operator.evaluate import evaluate
from geometry_operator.sampling import sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--candidates',default='runs/main');p.add_argument('--pool',default='data/geometries')
    p.add_argument('--references',default='data/references');p.add_argument('--output',default='runs/selected');p.add_argument('--limit',type=int)
    a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True);scores=[]
    for path in sorted(Path(a.candidates).glob('checkpoint*.npz')):
        report=evaluate(SimpleNamespace(checkpoint=str(path),pool=a.pool,references=a.references,split='dev',
            output=str(out/path.stem),limit=a.limit,unlock_test=False,build_references=False))
        if not report['evaluated']:continue
        s=report['summary']
        score=max(s['velocity_relative_l2']['p95']/.05,s['pressure_relative_l2']['p95']/.1,
                  s['pressure_drop_relative']['p95']/.07,s['momentum_rms']['p95']/.05,s['traction_rms']['p95']/.01)
        scores.append(dict(path=str(path),sha256=sha(path),score=score,step=report['step'],evaluated=report['evaluated'],complete_development=report['gates']['complete_split']))
    if not scores:raise RuntimeError('No candidate has independent development reference evidence')
    best=min(scores,key=lambda s:s['score']);shutil.copyfile(best['path'],out/'checkpoint.npz')
    (out/'selection.json').write_text(json.dumps(dict(selected=best,candidates=scores,
        split='development; reused for selection, not final test',criterion='worst P95/threshold across velocity,pressure,drop,momentum,traction',
        provisional=not best['complete_development'],accepted=False),indent=2))
    print(json.dumps(best))
if __name__=='__main__':main()
