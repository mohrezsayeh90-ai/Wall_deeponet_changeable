"""Archive a downloaded Kaggle run losslessly with per-file SHA-256 inventory."""
import argparse,hashlib,json,re,shutil,zipfile
from datetime import datetime,timezone
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--download-dir',type=Path,required=True);p.add_argument('--artifact-dir',type=Path,required=True);p.add_argument('--run-directory',required=True);p.add_argument('--kernel',required=True);a=p.parse_args()
    raw=a.download_dir;dest=a.artifact_dir;project=raw/'output/project';main=project/'runs'/a.run_directory
    inv=json.loads((raw/'download-inventory.json').read_text());files=[p for folder in ['output','source'] for p in sorted((raw/folder).rglob('*')) if p.is_file()]
    actual={str(p.relative_to(raw)) for p in (raw/'output').rglob('*') if p.is_file()}
    assert set(inv['files'])==actual and not inv['next_page_token']
    for f in files:
        if f.suffix in ['.json','.jsonl','.log','.py','.txt']:
            text=f.read_text(errors='replace')
            assert not re.search(r'KGAT_[A-Za-z0-9_]+|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN .*PRIVATE KEY-----|"accessToken"\s*:\s*"[^" ]+',text),f'Sensitive pattern in {f.name}'
    dest.mkdir(parents=True,exist_ok=True)
    for name in ['checkpoint.npz','status.json','runtime.json','training.jsonl']:shutil.copy2(main/name,dest/name)
    shutil.copytree(main/'development',dest/'development',dirs_exist_ok=True);shutil.copytree(raw/'source',dest/'source',dirs_exist_ok=True)
    shutil.copy2(project/'configs/runtime.json',dest/'model-config.json')
    for name in ['resume_manifest.json','profile.json']:
        if (project/name).exists():shutil.copy2(project/name,dest/name)
    for f in (raw/'output').glob('*.log'):shutil.copy2(f,dest/'kernel.log')
    records=[dict(path=p.relative_to(raw).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    archive=raw/'raw-output.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for f in files:z.write(f,f.relative_to(raw).as_posix())
    parts=[]
    with archive.open('rb') as f:
        i=0
        while block:=f.read(40*1024*1024):
            name=f'raw-output.zip.part{i:03d}';(dest/name).write_bytes(block);parts.append(dict(path=name,bytes=len(block),sha256=hashlib.sha256(block).hexdigest()));i+=1
    meta=dict(schema='kaggle-completed-output-v1',kernel=a.kernel,collected_at_utc=datetime.now(timezone.utc).isoformat(),kaggle_status='COMPLETE',output_file_count=len(actual),source_file_count=len(files)-len(actual),pagination_complete=True,archive=dict(format='concatenated ZIP parts in manifest order',bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),parts=parts),files=records)
    (dest/'inventory.json').write_text(json.dumps(meta,indent=2)+'\n');print('Archived',len(actual),'output files and',len(files)-len(actual),'source files in',len(parts),'parts.')


if __name__=='__main__':main()
