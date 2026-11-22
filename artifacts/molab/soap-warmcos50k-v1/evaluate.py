"""Read-only FEM comparison in a separate process; never returns labels to trainer."""
from pathlib import Path
import sys,json,time,hashlib
import numpy as np
R=Path('/home/marimo/wall_soap_warmcos50k_v1');P=Path('/home/marimo/wall_rad_ab_15k_v1/frozen/output/project');OLD=P.parents[2]
sys.path.insert(0,str(P))
import jax
jax.config.update('jax_enable_x64',True)
from geometry_operator.geometry import validate,descriptors,geometry_id
from geometry_operator.api import FlowOperator
def write(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
population=json.loads((OLD/'repo/reports/adam_next50k_comparison/selection.json').read_text())['population'];refs=[]
for row in population:
 with np.load(OLD/'repo/reports'/row['group']/row['directory']/'fields.npz') as z:
  c=z['coeff'].copy();validate(c,False);d=descriptors(c)
  refs.append(dict(case=str(row['case_id']),title=row['title'],coeff=c,xy=z['xy'].copy(),w=z['weights'].copy(),truth=z['fem'].copy(),descriptors=d,geometry_id=geometry_id(c)))
hard=set(x['case'] for x in sorted(refs,key=lambda x:x['descriptors']['max_curvature'],reverse=True)[:12])
write(R/'fem_cohort.json',dict(used_for_training=False,training_labels=0,selection='12of20 highest wall curvature, chosen solely from geometry before evaluating new model',hard_cases=sorted(hard,key=int),reynolds=30,cases=[{k:v for k,v in x.items() if k in ['case','title','descriptors','geometry_id']} for x in refs],bounds=dict(min_gap=.6,max_gap=1.5,center_deviation=.35,max_slope=.7),all_validate_passed=True))
def assess(path,step):
 model=FlowOperator(path,allow_unvalidated=True,chunk_size=256);rows=[]
 for r in refs:
  pred=model.evaluate(r['coeff'],30.,r['xy']);truth=r['truth'];w=r['w']
  def rel(a,b):
   ww=w.reshape((-1,)+(1,)*(a.ndim-1));den=float(np.sum(ww*b*b))
   return float(np.sqrt(np.sum(ww*(a-b)**2)/den)) if den>1e-12 else None
  rows.append(dict(case=r['case'],hard=r['case'] in hard,velocity_l2=rel(pred[:,:2],truth[:,:2]),u_l2=rel(pred[:,0],truth[:,0]),v_l2=rel(pred[:,1],truth[:,1]),pressure_l2=rel(pred[:,2],truth[:,2])))
 out=dict(step=step,offset=step-392400,training_labels=0,checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),cases=rows)
 for group,rr in [('all20',rows),('hard12',[x for x in rows if x['hard']])]:
  out[group]={}
  for k in ['velocity_l2','u_l2','v_l2','pressure_l2']:
   vals=[x[k] for x in rr if x[k] is not None];out[group][k+'_median']=float(np.median(vals));out[group][k+'_max']=float(np.max(vals));out[group][k+'_valid_count']=len(vals)
 write(R/'fem'/f'fem_{step}.json',out)
(R/'fem').mkdir(exist_ok=True);assess(R/'base.npz',392400);seen=set()
while True:
 for ready in sorted((R/'continuation/snapshots').glob('*.ready.json')):
  m=json.loads(ready.read_text());step=m['step']
  if step in seen:continue
  path=ready.with_name(ready.name.replace('.ready.json','.npz'));assert hashlib.sha256(path.read_bytes()).hexdigest()==m['sha256'];assess(path,step);seen.add(step)
 if (R/'failure.json').exists():break
 if (R/'continuation/completed.json').exists():
  d=json.loads((R/'continuation/completed.json').read_text())
  if d['step'] in seen:break
 time.sleep(10)
write(R/'fem_evaluation_done.json',dict(evaluated_steps=sorted(seen),read_only=True))