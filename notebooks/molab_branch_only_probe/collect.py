"""Independent cloud-side collector: checksum snapshots and push them periodically using existing Git auth.
Token is read only through hidden stdin and retained in memory, never written to disk.
"""
from pathlib import Path
import requests,getpass,json,re,base64,hashlib,time,subprocess,sys,numpy as np
repo=Path('/workspace/Wall_deeponet_changeable');dest=repo/'artifacts/molab/branch-only-probe-v1';dest.mkdir(parents=True,exist_ok=True)
secret=getpass.getpass('Connection token (hidden): ');s=requests.Session();s.headers['Authorization']='Bearer '+secret
url=sys.argv[1].rstrip('/');resp=s.get(url+'/api/sessions',timeout=30);resp.raise_for_status();sessions=resp.json();assert len(sessions)==1;sid=next(iter(sessions))
def execute(code):
 r=s.post(url+'/api/kernel/execute',headers={'Marimo-Session-Id':sid},json={'code':code},stream=True,timeout=90);r.raise_for_status();event='';out=[];done=False
 for raw in r.iter_lines():
  line=raw.decode()
  if line.startswith('event:'):event=line[6:].strip()
  elif line.startswith('data:'):
   obj=json.loads(line[5:].strip())
   if event=='stdout':out.append(obj.get('data',''))
   if event=='done':
    assert obj.get('success',True);done=True;break
 r.close();assert done;return ''.join(out)
def fetch(remote,local,info):
 tmp=local.with_suffix(local.suffix+'.download');local.parent.mkdir(parents=True,exist_ok=True)
 with tmp.open('wb') as f:
  for pos in range(0,info['bytes'],524288):
   code="import base64 as _b\nwith open("+repr(remote)+",'rb') as _f:\n _f.seek("+str(pos)+");print('CHUNK::'+_b.b64encode(_f.read("+str(min(524288,info['bytes']-pos))+")).decode())"
   text=execute(code);data=base64.b64decode(re.search(r'CHUNK::([A-Za-z0-9+/=]+)',text).group(1));assert len(data)==min(524288,info['bytes']-pos);f.write(data)
 assert hashlib.sha256(tmp.read_bytes()).hexdigest()==info['sha256'];tmp.replace(local)
 if local.suffix=='.npz' and 'checkpoint_' in local.name:
  with np.load(local,allow_pickle=False) as d:assert json.loads(str(d['metadata']))['soap_state_present']
execute('import marimo._code_mode as cm; help(cm)')
root='/home/marimo/wall_branch_only_probe_v1';seen={};published=False
while True:
 try:
  code="""from pathlib import Path as _P
import json as _j,hashlib as _h
_r=_P(ROOT);_files=[]
for _p in _r.iterdir():
 if _p.is_file() and (_p.suffix in ['.json','.jsonl','.py','.log','.png','.pdf'] or _p.name in ['control.npz','audit.npz']):_files.append(_p)
for _name in ['branch_only']:
 _d=_r/_name
 if _d.exists():
  _files.extend(p for p in _d.iterdir() if p.is_file() and p.suffix in ['.json','.jsonl'])
  for _p in (_d/'snapshots').glob('*.ready.json'):
   _meta=_j.loads(_p.read_text());_f=_p.with_name(_p.name.replace('.ready.json','.npz'))
   if _meta['step']>287400:_files.extend([_p,_f])
_o=[]
for _p in _files:
 _data=_p.read_bytes();_o.append({'path':str(_p.relative_to(_r)),'bytes':len(_data),'sha256':_h.sha256(_data).hexdigest()})
print('FILES::'+_j.dumps(_o))
""".replace('ROOT',repr(root))
  listing=json.loads(re.search(r'FILES::([^\n]+)',execute(code)).group(1));changed=[]
  for info in listing:
   name=info['path'];local=dest/name
   if seen.get(name)==info['sha256']:continue
   if local.exists() and hashlib.sha256(local.read_bytes()).hexdigest()==info['sha256']:seen[name]=info['sha256'];continue
   try:fetch(root+'/'+name,local,info)
   except AssertionError:
    print('FILE_CHANGED_DURING_COLLECTION',name,flush=True);continue
   seen[name]=info['sha256'];changed.append(name)
  (dest/'collection_status.json').write_text(json.dumps({'last_success_unix':time.time(),'verified_files':len(seen),'sha256':seen,'collector_lifetime':'Runs in Codex cloud; reconnect/restart needed if that environment stops. No molab persistence guarantee.'},indent=2))
  newweights=[x for x in changed if x.endswith('.npz') and 'checkpoint_' in x]
  finished=(dest/'comparison.json').exists() and all(seen.get(info['path'])==info['sha256'] for info in listing)
  if newweights or finished or not published:
   subprocess.run(['git','add','--',str(dest.relative_to(repo))],cwd=repo,check=True)
   staged=subprocess.run(['git','diff','--cached','--quiet','--',str(dest.relative_to(repo))],cwd=repo).returncode
   if staged:
    subprocess.run(['git','commit','--only','-m','Archive verified SOAP trial progress and full-state snapshots','--',str(dest.relative_to(repo))],cwd=repo,check=True,stdout=subprocess.DEVNULL)
    subprocess.run(['git','push','origin','HEAD:main'],cwd=repo,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
   published=True;print('BACKUP_PUSHED',newweights,'files',len(seen),flush=True)
  if finished:print('COLLECTION_COMPLETE',flush=True);break
 except Exception as e:
  published=False
  print('COLLECTOR_RETRY',type(e).__name__,str(e).replace(secret,'[REDACTED]')[:200],flush=True)
 time.sleep(45)
