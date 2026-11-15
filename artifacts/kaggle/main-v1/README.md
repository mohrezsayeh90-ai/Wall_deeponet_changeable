# Kaggle v1: completed job, provisional research checkpoint

Collected from [mohrezsayeh/geometry-pi-deeponet-main, version 1](https://www.kaggle.com/code/mohrezsayeh/geometry-pi-deeponet-main/versions/1).
The downloaded script matches the explicitly pinned v1 source. Its embedded revision
and the checkpoint identify `f6b32ae7e95853f6e48142e822d6718597eb4708`.

Kaggle status is **COMPLETE**, but training stopped because of its time budget at
**32,216/100,000 steps**, after 17,400.49 seconds. The checkpoint contains weights,
Adam moments, RNG state, normalization and model config. It is **not accepted**.

## Contents

| Path | Contents |
| --- | --- |
| `checkpoint.npz` | Original final checkpoint, with optimizer and RNG state |
| `model-config.json` | Exact runtime config; float32 training, full-size architecture |
| `status.json`, `runtime.json` | Original trainer completion/runtime metadata |
| `training.jsonl` | All 3,222 logged updates, ending at step 32,210 |
| `kernel.log` | Complete downloaded Kaggle console log |
| `development/` | Original report, case metrics, 8 predicted/FEM field files and 8 plots |
| `precision-profile.json` | Initialization precision/timing profile |
| `source/` | Downloaded self-contained script and Kaggle metadata |
| `raw-output.zip.part000`, `raw-output.zip.part001` | Complete archive, split below Git's large-file warning threshold |
| `inventory.json` | SHA-256, sizes and paths for all 71 archived files and both parts |
| `collection-validation.json` | Checks performed after collection |
| `kaggle-observation.json` | Confirmed notebook version, final API status and observation time |

The archive preserves **all 69 downloaded output files**, including generated Python
bytecode/cache files, exact executed project sources, 10,000/400/1,000/200 geometry
pools, normalization, immutable manifest, eight FEM references and every run output.
It also includes both downloaded source/metadata files. The readable files above
are byte-identical copies for convenience. Nothing is silently omitted.
No account credentials or signed download URLs are needed to restore this archive.

## Restore everything

From the repository root, using Python 3.12 (standard library only):

```bash
python scripts/restore_kaggle.py --output runs/restored-kaggle-v1
```

The script verifies both parts, the combined ZIP, every file hash/size and safe
paths before writing. It refuses to replace different existing files. Repeating
the command is supported. The resulting layout is:

```text
runs/restored-kaggle-v1/
  source/                 # downloaded Kaggle script and metadata
  output/
    geometry-pi-deeponet-main.log
    project/
      data/geometries/    # complete immutable pools and normalization
      data/references/dev/
      runs/main/          # checkpoint, logs and eight-case development results
      configs/
      geometry_operator/ # actual frozen source, not current checkout code
```

The Python 3.13 bytecode is preserved for provenance, not for use on Python 3.12.
Use current checkout code and its documented environment when loading the weights.
All four remote pool hashes match `reports/split_manifest.json`. The locked-test
pool is archived; it was not used for evaluation or selection during collection.

## Use the checkpoint for research

In the documented CPU environment, from the repository root:

```python
import numpy as np
from geometry_operator.api import FlowOperator

model = FlowOperator('artifacts/kaggle/main-v1/checkpoint.npz', allow_unvalidated=True)
uvp = model.evaluate(np.zeros((2, 6, 80)), 55., np.array([[2.5, .5]]))
```

`allow_unvalidated=True` is explicit research use; it does not grant acceptance.
The API's default remains rejection without a passing independent benchmark.
The original checkpoint SHA-256 is
`a7e9e35b91d6e3c7f3a37404b940289427bfe78f9b38e39661c10382babd5235`.

For a future GPU continuation, use the exact `model-config.json`, restored train
pool and this checkpoint; the optimizer/RNG state was verified to load. The
following command is documented for a GPU machine and was **not executed** during
collection:

```bash
python -m geometry_operator.train \
  --pool runs/restored-kaggle-v1/output/project/data/geometries \
  --config artifacts/kaggle/main-v1/model-config.json \
  --resume artifacts/kaggle/main-v1/checkpoint.npz \
  --output runs/continued --steps 100000 --batch 32 --points 128 --outlet 48 \
  --devices 1 --require-gpu
```

Here `--steps 100000` is the total target; 67,784 updates remain. The current
checkout's trainer is newer than the frozen v1 code. CPU loading/inference was
verified; GPU continuation with the newer trainer was not run.

## Evaluation limits

Only **8/400** development geometries, one from each family and all at **Re=10**,
were evaluated. No failures were reported for those eight cases. Median velocity,
pressure and pressure-drop relative errors are **2.59%, 2.68%, 2.53%**; velocity
P95 is **3.58%**. These pilot numbers are not the complete balanced benchmark.
Velocity-median, family-median and momentum gates fail; median momentum RMS is
0.35536 (P95 0.57213). The complete development split, locked test, Reynolds
sweeps, stress evaluation, matched learned baselines/ablations and wall-stress
verification remain outstanding. `all_gates_passed` is false.

See [the collection report](../../../reports/KAGGLE_COMPLETED_RUN.md) and the
original [case-level report](development/report.json).
