"""Parallel independent FEM generation, with incremental failure visibility."""
import argparse,json,os
from concurrent.futures import ProcessPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from geometry_operator.sampling import load_pool

def solve_case(task):
    i,coeff,re,path=task
    from geometry_operator.reference import build_case
    try:return dict(index=i,status='passed',**build_case(coeff,re,path,40,12))
    except Exception as e:return dict(index=i,status='failed',error=str(e))

def main():
    p=argparse.ArgumentParser();p.add_argument('--pool',default='data/geometries');p.add_argument('--split',default='dev',choices=['dev','stress'])
    p.add_argument('--output',default='data/references');p.add_argument('--workers',type=int,default=2);a=p.parse_args()
    pool,m=load_pool(a.pool,a.split);out=Path(a.output)/a.split;out.mkdir(parents=True,exist_ok=True);records=[]
    tasks=[(i,c,float(r),str(out/f'{i:04d}.npz')) for i,(c,r) in enumerate(zip(pool['coeff'],pool['reynolds']))]
    with ProcessPoolExecutor(max_workers=a.workers) as executor:
        futures=[executor.submit(solve_case,t) for t in tasks]
        for future in as_completed(futures):
            r=future.result();records.append(r);print(json.dumps(r),flush=True)
            tmp=out/'status.tmp.json';tmp.write_text(json.dumps(dict(split=a.split,pool_sha256=m['splits'][a.split]['sha256'],
                expected=len(tasks),cases=sorted(records,key=lambda x:x['index'])),indent=2));tmp.replace(out/'status.json')
    if any(r['status']=='failed' for r in records):raise SystemExit(1)
if __name__=='__main__':main()
