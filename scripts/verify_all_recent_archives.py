from pathlib import Path
import json,hashlib
import numpy as np
R=Path(__file__).resolve().parents[1];O=R/'reports/molab_soap_warmcos50k/completed';mapping={'wall_soap_sweep50k_v1':'soap-sweep50k-v1','wall_soap_highlr50k_v1':'soap-highlr50k-v1','wall_soap_warmcos50k_v1':'soap-warmcos50k-v1'}
inv=json.loads((O/'remote_inventory.json').read_text());index={};exact=[];missing=[];aliases=[]
for name in mapping.values():
 for p in (R/'artifacts/molab'/name).rglob('*'):
  if p.is_file() and not p.name.endswith('.download'):index.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(),str(p.relative_to(R)))
p=R/'artifacts/molab/soap-cosine100k-v1/continuation/snapshots/checkpoint_342400.npz';index.setdefault(hashlib.sha256(p.read_bytes()).hexdigest(),str(p.relative_to(R)))
for x in inv:
 p=R/'artifacts/molab'/mapping[x['root']]/x['path'];rec=dict(remote_root=x['root'],remote_path=x['path'],bytes=x['bytes'],sha256=x['sha256'])
 if p.exists() and p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']:exact.append(dict(rec,archive_path=str(p.relative_to(R))))
 elif x['sha256'] in index:aliases.append(dict(rec,identical_content_archived_at=index[x['sha256']]))
 else:missing.append(dict(rec,target_path=str(p)))
checks=[]
for name,step,count in [('soap-highlr50k-v1',392400,105001),('soap-warmcos50k-v1',442400,155001)]:
 d=R/'artifacts/molab'/name/'continuation/snapshots';ready=list(d.glob('*.ready.json'));assert len(ready)==21
 for m in ready:
  x=json.loads(m.read_text());p=m.with_name(m.name.replace('.ready.json','.npz'));assert p.stat().st_size==x['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==x['sha256']
 p=d/f'checkpoint_{step}.npz'
 with np.load(p,allow_pickle=False) as z:
  md=json.loads(str(z['metadata']));assert md['step']==step and md['soap_state_present'] and int(z['soap_0'])==count
  for k in z.files:
   if k.startswith(('param_','soap_')):assert np.isfinite(z[k]).all(),k
 checks.append(dict(archive=name,full_state_snapshots=21,final_step=step,optimizer_counter=count,all_snapshot_hashes_passed=True,final_parameters_and_state_finite=True))
out=dict(remote_file_count=len(inv),exact_file_matches=exact,identical_content_aliases=aliases,missing=missing,final_state_checks=checks,complete=not missing)
(O/'archive_verification.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'remote_files':len(inv),'exact':len(exact),'aliases':len(aliases),'missing':missing,'final_state_checks':checks},indent=2))
