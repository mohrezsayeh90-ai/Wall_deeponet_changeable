"""Restore the complete collected Kaggle v1 output, verifying every SHA-256."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import zipfile


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def restore(artifact_dir, output):
    manifest = json.loads((artifact_dir / 'inventory.json').read_text())
    output = output.resolve()
    expected = {item['path']: item for item in manifest['files']}
    if len(expected) != len(manifest['files']):
        raise ValueError('Duplicate inventory path')
    with tempfile.TemporaryFile() as combined:
        digest = hashlib.sha256()
        total = 0
        for item in manifest['archive']['parts']:
            part = artifact_dir / item['path']
            if part.stat().st_size != item['bytes'] or sha(part) != item['sha256']:
                raise ValueError(f'Archive part checksum mismatch: {part.name}')
            with part.open('rb') as source:
                while chunk := source.read(1024 * 1024):
                    combined.write(chunk)
                    digest.update(chunk)
                    total += len(chunk)
        if total != manifest['archive']['bytes'] or digest.hexdigest() != manifest['archive']['sha256']:
            raise ValueError('Combined archive checksum mismatch')
        combined.seek(0)
        with zipfile.ZipFile(combined) as archive:
            if len(archive.infolist()) != len(expected) or set(archive.namelist()) != set(expected):
                raise ValueError('Archive file list differs from inventory')
            # Verify every file and existing destination before writing any output.
            for name, item in expected.items():
                target = (output / name).resolve()
                if not target.is_relative_to(output) or target == output:
                    raise ValueError('Unsafe archive path')
                with archive.open(name) as source:
                    if hashlib.file_digest(source, 'sha256').hexdigest() != item['sha256']:
                        raise ValueError(f'File checksum mismatch: {name}')
                if archive.getinfo(name).file_size != item['bytes']:
                    raise ValueError(f'File size mismatch: {name}')
                if target.exists() and (not target.is_file() or sha(target) != item['sha256']):
                    raise ValueError(f'Refusing to replace different existing file: {name}')
            for name in expected:
                target = output / name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    with archive.open(name) as source, target.open('xb') as dest:
                        shutil.copyfileobj(source, dest)
    print(f'Verified and restored {len(expected)} files to {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifact-dir', type=Path,
                        default=Path(__file__).resolve().parents[1] / 'artifacts/kaggle/main-v1')
    parser.add_argument('--output', type=Path, default=Path('runs/restored-kaggle-v1'))
    args = parser.parse_args()
    restore(args.artifact_dir, args.output)


if __name__ == '__main__':
    main()
