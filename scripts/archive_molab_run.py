"""Archive an authenticated, downloaded molab ZIP with complete SHA256 inventory."""
import argparse,hashlib,json,shutil,zipfile
from pathlib import Path
from datetime import datetime,timezone

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--download-dir',type=Path,required=True);ap.add_argument('--artifact-dir',type=Path,required=True);ap.add_argument('--case');a=ap.parse_args()
    archive=a.download_dir/'results.zip';download=json.loads((a.download_dir/'download-verification.json').read_text());assert sha(archive)==download['sha256']
    dest=a.artifact_dir;dest.mkdir(parents=True,exist_ok=True);records=[]
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for item in z.infolist():
            assert not item.is_dir();name=item.filename;assert not name.startswith('/') and '..' not in Path(name).parts
            body=z.read(name);records.append({'path':name,'bytes':len(body),'sha256':hashlib.sha256(body).hexdigest()})
        if a.case:
            if 'base.npz' in z.namelist():(dest/'base_checkpoint.npz').write_bytes(z.read('base.npz'))
            prefix='cases/'+a.case+'/'
            for name in z.namelist():
                if name.startswith(prefix) and Path(name).name in ['checkpoint.npz','best_checkpoint.npz','best.json','training.jsonl','probe.jsonl','status.json','runtime.json','resume.json','fem_pilot.json']:
                    (dest/Path(name).name).write_bytes(z.read(name))
        for name in ['manifest.json','lr.json','worker.py','supervisor.py','lr_control.py','supervisor-status.json','suite-status.json','comparison.json','comparison.csv','comparison.png','comparison.pdf','fixed_probe.npz','transition.json']:
            if name in z.namelist():(dest/name).write_bytes(z.read(name))
    parts=[]
    with archive.open('rb') as f:
        for i in range(10000):
            block=f.read(40*1024*1024)
            if not block:break
            p=dest/f'raw-output.zip.part{i:03d}';p.write_bytes(block);parts.append({'path':p.name,'bytes':len(block),'sha256':sha(p)})
    inv={'schema':'molab-complete-output-v1','collected_at_utc':datetime.now(timezone.utc).isoformat(),'original_files':len(records),
         'archive':{'bytes':archive.stat().st_size,'sha256':sha(archive),'parts':parts},'files':records,'zip_crc_passed':True,'download_sha256_matches_remote':True}
    (dest/'inventory.json').write_text(json.dumps(inv,indent=2)+'\n');print('Archived',len(records),'files in',len(parts),'verified parts.')
if __name__=='__main__':main()
