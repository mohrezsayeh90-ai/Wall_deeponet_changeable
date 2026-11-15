# Completed continuation v1 — step 82,216

This is the research checkpoint after **50,000 additional updates**, resumed from
step 32,216 with the original Adam/RNG state. Kaggle reports COMPLETE; the trainer
records `step=82216`, `target_steps=82216`, `stop_reason=target_completed`.
Training took **26,840.254 seconds** (7 h 27 min 20 s). This is the requested
continuation target, not completion of the original 100,000-step research budget.
The checkpoint remains **unaccepted** pending full independent benchmarking.

- [Final checkpoint with Adam/RNG state](checkpoint.npz), SHA-256:
  `84276c11feecd5bf230c79d1077250018b356ae118491678a4ed69e46ab83140`.
- [Exact model config](model-config.json), [final status](status.json),
  [runtime metadata](runtime.json), [resume provenance](resume_manifest.json).
- [Complete training log](training.jsonl): 5,001 records, step 32,217 through
  82,210. The last logged loss is 0.018711606. No loss is invented at 82,216.
- [Complete console log](kernel.log), [downloaded source and metadata](source/).
- [Original eight-case development outputs](development/): metrics, predictions,
  reference fields and plots. Only 8/400 development cases, all Re=10, were evaluated.
- [Collection verification](collection-validation.json) and [13 passing tests](tests.txt).
- [Full training plots and paired before/after comparison](../../../reports/continuation_50k/completed/README.md).
- Original step-32,216 weights remain in [main-v1](../main-v1/README.md).

## Restore every output

The two `raw-output.zip.part*` files preserve **all 68 Kaggle output files** plus
**two source/metadata files**, including complete geometry pools, normalization,
FEM reference caches, bytecode/test caches, exact frozen project source and all
run outputs. No files from the downloaded run are silently omitted. Parts are at
most 40 MiB; [inventory.json](inventory.json) contains every file hash/size and
both part hashes plus the combined ZIP hash. The checkpoint, reports and logs
above are byte-identical convenience copies of archived originals.

From the repository root:

```bash
python scripts/restore_kaggle.py \
  --artifact-dir artifacts/kaggle/continue-50k-v1 \
  --output runs/restored-continuation-v1
```

This verifies both parts and every archived file before restoring the original
`output/project/runs/continued_50k/` layout. Repeating restoration is supported;
different existing files are never replaced silently. The original Python 3.13
bytecode is preserved only as provenance; use checkout source with the documented
Python 3.12 CPU environment to load the checkpoint.

## Research inference

```python
import numpy as np
from geometry_operator.api import FlowOperator
model = FlowOperator('artifacts/kaggle/continue-50k-v1/checkpoint.npz',
                     allow_unvalidated=True)
uvp = model.evaluate(np.zeros((2, 6, 80)), 55., np.array([[2.5, .5]]))
```

The matching original geometry pools/config are available through restoration.
Optimizer state and finite arrays were verified to load; CPU predictions at six
points on each of development cases 0 and 5 reproduce the saved GPU fields to
maximum absolute difference 4.45e-16. This is a portability check, not another
full benchmark. All four pool hashes match the original immutable manifest.

The pilot median velocity/pressure/drop errors are **1.831% / 1.049% / 0.304%**.
Family-median and momentum gates still fail, as do incomplete full-benchmark
requirements. No accepted validation sidecar is supplied and the default API's
acceptance protection remains unchanged. Prior 15-case, random-five and stress-five
reports retain their original step-32,216 weights and are not relabeled.
