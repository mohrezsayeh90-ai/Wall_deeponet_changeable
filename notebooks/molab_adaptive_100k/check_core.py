from pathlib import Path
import os,sys,json,hashlib
root=Path('/home/marimo/wall_adaptive_100k');project=root/'frozen/output/project';sys.path.insert(0,str(project))
import jax,jax.numpy as jnp,numpy as np
jax.config.update('jax_enable_x64',True)
from geometry_operator.checkpoint import load
from geometry_operator.sampling import load_pool,batch
from geometry_operator.train import make_step as old_make
from geometry_operator.adaptive_core import make_step
try:
    p,c,meta,(mean,std),(m,v)=load(root/'base.npz',optimizer=True)
    pool,_=load_pool(project/'data/geometries');rng=np.random.default_rng();rng.bit_generator.state=meta['rng_state']
    sample=batch(rng,pool,32,128,48);data={k:jnp.asarray(x,dtype=jnp.float32) for k,x in sample.items()};count=jnp.asarray(meta['step']+1)
    old,_=old_make(c,mean,std);new,_=make_step(c,mean,std)
    reference=old(p,m,v,count,data);changed=new(p,m,v,count,data,jnp.asarray(2e-5,dtype=jnp.float32))
    for a,b in zip(jax.tree_util.tree_leaves(reference),jax.tree_util.tree_leaves(changed)):np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=1e-5,atol=1e-7)
    higher=new(p,m,v,count,data,jnp.asarray(4e-5,dtype=jnp.float32))
    for a,b in zip(jax.tree_util.tree_leaves(changed[1:]),jax.tree_util.tree_leaves(higher[1:])):np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
    delta=lambda tree:sum(np.sum((np.asarray(a,dtype=float)-np.asarray(b,dtype=float))**2) for a,b in zip(jax.tree_util.tree_leaves(p),jax.tree_util.tree_leaves(tree)))**.5
    ratio=delta(higher[0])/delta(changed[0]);assert 1.9<ratio<2.1
    ids=json.loads((root/'probe_ids.json').read_text());assert len(set(ids['control']+ids['audit']))==64
    result={'passed':True,'base_step':meta['step'],'dynamic_core_matches_original_at_2e5':True,'moments_and_gradients_independent_of_lr_for_same_first_batch':True,
        'parameter_update_scale_ratio_4e5_vs_2e5':ratio,'disjoint_control_audit_geometries':True,'first_batch_sha256':hashlib.sha256(b''.join(np.asarray(sample[k]).tobytes() for k in sorted(sample))).hexdigest()}
    (root/'core-check.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
except BaseException as e:
    (root/'core-check.json').write_text(json.dumps({'passed':False,'error_type':type(e).__name__}));raise
