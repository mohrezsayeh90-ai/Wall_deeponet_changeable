"""Example: same frozen checkpoint on two unseen development geometries."""
import argparse
from pathlib import Path
import numpy as np
from geometry_operator.api import FlowOperator
from geometry_operator.geometry import sample_numpy
from geometry_operator.sampling import load_pool

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--pool',default='data/geometries');p.add_argument('--output',default='runs/predictions');p.add_argument('--allow-unvalidated',action='store_true');a=p.parse_args()
    model=FlowOperator(a.checkpoint,allow_unvalidated=a.allow_unvalidated);pool,_=load_pool(a.pool,'dev')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for i in [1,5]:
        coeff=pool['coeff'][i];x=np.linspace(0,5,101);eta=np.linspace(0,1,51);c,g=sample_numpy(coeff,x);gap=1+g;b=c-g/2
        xy=np.stack([np.broadcast_to(x[:,None],(101,51)),b[:,None]+gap[:,None]*eta],-1).reshape(-1,2)
        field=model.evaluate(coeff,55.,xy)
        np.savez_compressed(out/f'geometry_{i}.npz',xy=xy,fields=field,coeff=coeff,reynolds=55.,checkpoint_sha256=model.sha256)
if __name__=='__main__':main()
