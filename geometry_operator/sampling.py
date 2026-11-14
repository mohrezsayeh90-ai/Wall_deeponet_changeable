"""Reproducible immutable geometry pools; no solved field data."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from .geometry import generate, geometry_id, sensors, descriptors, FAMILIES


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate_pools(output, counts=(10000,400,1000,200),seed=20261115):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    manifest=output/'manifest.json'
    if manifest.exists():
        m=json.loads(manifest.read_text())
        if m['counts']!=list(counts) or m['seed']!=seed:raise ValueError('Refusing to overwrite different immutable split')
        for s in m['splits']:
            if sha(output/f'{s}.npz')!=m['splits'][s]['sha256']:raise ValueError('Pool hash mismatch')
        return m
    rng=np.random.default_rng(seed);ids=set();prior={i:[] for i in range(8)};splits={}
    for split,count in zip(['train','dev','test','stress'],counts):
        if count%8:raise ValueError('Counts must be divisible by eight')
        records=[];coeffs=[];packed=[];trees={i:cKDTree(prior[i]) if prior[i] else None for i in range(8)}
        for i in range(count):
            family=i%8
            for attempt in range(1000):
                coeff=generate(rng,family,split=='stress');gid=geometry_id(coeff)
                vec=sensors(coeff,33);canonical=[vec,vec.reshape(2,-1)[:,::-1].reshape(-1),
                    (vec.reshape(2,-1)*np.array([[-1],[1]])).reshape(-1),
                    (vec.reshape(2,-1)[:,::-1]*np.array([[-1],[1]])).reshape(-1)]
                near=trees[family] is not None and any(trees[family].query(v)[0]/np.sqrt(66)<.001 for v in canonical)
                if gid not in ids and not near:break
            else:raise RuntimeError('Cannot generate sufficiently separated geometry')
            ids.add(gid);coeffs.append(coeff);packed.append(sensors(coeff))
            prior[family].extend(canonical)
            re=float([10,30,60,100][(i//8)%4])
            records.append(dict(id=gid,family=FAMILIES[family],index=i,reynolds=re,**descriptors(coeff)))
        path=output/f'{split}.npz'
        np.savez_compressed(path,coeff=np.array(coeffs),sensors=np.array(packed),
                            reynolds=np.array([r['reynolds'] for r in records]))
        splits[split]=dict(count=count,sha256=sha(path),cases=records)
        print(f'Generated {split}: {count}',flush=True)
    # Statistics solely from training geometries.
    with np.load(output/'train.npz') as train:
        mean=train['sensors'].mean(0);std=np.maximum(train['sensors'].std(0),.02)
    np.savez(output/'normalization.npz',mean=mean,std=std)
    m=dict(schema='geometry-splits-v1',seed=seed,counts=list(counts),splits=splits,
        near_duplicate_rule='33-point center/gap RMS >= .001 across splits, including axial and centerline reflections',
        sweep_test_indices=list(range(min(80,counts[2]))),test_opened=False)
    manifest.write_text(json.dumps(m,indent=2))
    return m


def load_pool(path,split='train'):
    path=Path(path);m=json.loads((path/'manifest.json').read_text())
    if sha(path/f'{split}.npz')!=m['splits'][split]['sha256']:raise ValueError('Pool modified')
    with np.load(path/f'{split}.npz') as data:pool={k:data[k] for k in data.files}
    return pool,m


def batch(rng,pool,batch_size,points,outlet):
    ix=rng.integers(len(pool['coeff']),size=batch_size)
    z=rng.uniform(.0001,.9999,(batch_size,points,2))
    # Explicit biased training objective: 50% uniform,25% near-wall,25%
    # interior-axial mixture. Evaluation uses physical-area quadrature instead.
    n=points//4
    z[:,:n,1]=rng.beta(.5,.5,(batch_size,n))*.9998+.0001
    z[:,n:2*n,0]=rng.uniform(.1,.9,(batch_size,n))
    re=rng.uniform(10,100,batch_size);endpoint=rng.random(batch_size)<.2
    re[endpoint]=rng.choice([10.,100.],endpoint.sum())
    return dict(coeff=pool['coeff'][ix],sensors=pool['sensors'][ix],re=re,z=z,
                outlet=rng.uniform(0,1,(batch_size,outlet)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='data/geometries');p.add_argument('--small',action='store_true')
    a=p.parse_args();generate_pools(a.output,(32,8,8,8) if a.small else (10000,400,1000,200))
if __name__=='__main__':main()
