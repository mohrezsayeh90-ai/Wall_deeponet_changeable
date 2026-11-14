"""Physical-aspect gallery and split descriptor distributions."""
from pathlib import Path
import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from geometry_operator.geometry import sample_numpy,FAMILIES
from geometry_operator.sampling import load_pool

def main():
    p=argparse.ArgumentParser();p.add_argument('--pool',default='data/geometries');p.add_argument('--output',default='reports/geometry_gallery.png');a=p.parse_args()
    pool,m=load_pool(a.pool,'dev');fig,axes=plt.subplots(4,2,figsize=(12,9),layout='constrained');x=np.linspace(0,5,1001)
    for i,ax in enumerate(axes.flat):
        c,g=sample_numpy(pool['coeff'][i],x);b=c-g/2;t=1+c+g/2
        ax.fill_between(x,b,t,alpha=.15);ax.plot(x,b,'k');ax.plot(x,t,'k');ax.set_aspect('equal');ax.set_title(FAMILIES[i]);ax.set_xlabel('x / H0');ax.set_ylabel('y / H0')
    Path(a.output).parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=160);plt.close(fig)
if __name__=='__main__':main()
