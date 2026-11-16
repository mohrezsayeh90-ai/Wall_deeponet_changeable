# Completed Adam continuation: 82,216 → 132,216

**Verified complete:** exactly 50,000 additional updates, final step **132,216**, `target_completed`. Training took **7 h 20 min 41 s**. All 68 output files and two source files are preserved in [the artifact archive](../../artifacts/kaggle/adam-next50k-v1/inventory.json); archive parts and every downloaded file were hash-checked. Adam moments, configuration, normalization and training-pool identity were checked. CPU inference agrees with saved GPU predictions to <1e-14 on 256 saved pilot points. [Completion record](completion.json).

[Twenty-case three-stage comparison and six random result chains](../adam_next50k_comparison/README.md). Old checkpoints/reports remain available. Model acceptance is still false; the eight-case pilot is not the full benchmark.

Private run: https://www.kaggle.com/code/mohrezsayeh/geometry-pi-deeponet-adam-next50k

This new run continues the successful **Adam** checkpoint from the first +50k
continuation. It **does not use the L-BFGS weights**. The request for another run
with the same previous procedure is implemented as **another 50,000 Adam updates**.

- Base: **82,216**, SHA-256
  `84276c11feecd5bf230c79d1077250018b356ae118491678a4ed69e46ab83140`.
- Target total: **132,216**. First expected new update: **82,217**.
- Same frozen v1 model/physics/trainer source at revision
  `f6b32ae7e95853f6e48142e822d6718597eb4708`.
- Original Adam moments and exact NumPy RNG state restored. Learning-rate schedule
  continues with the global step count; neither optimizer nor schedule restarts.
- **float32**, global batch **32**, **128** interior and **48** outlet points per
  geometry. Random train-pool geometry/Re/collocation sampling resumes; this does
  not use L-BFGS's fixed 512-geometry bank.
- GPU: T4. Training budget **10 hours**; session timeout **10.5 hours**, within the
  approximately **11.43 GPU hours** remaining at submission. Previous measured
  speed implies about **7–8 hours** for the requested updates; this is an estimate.
- Initial resumed update and every **100** updates are saved; logging every 10.
  Final checkpoint records `target_completed` only if step 132,216 is reached.
  A wall-clock/quota stop is partial, not 50,000 completed updates.
- Original weights/results and both earlier jobs are preserved. The latest
  collected research-weight pointer now refers to Adam step 132,216 after downloading
  and checking the completed outputs.

The first job action checks GPU availability and runs the original frozen nine-test
suite. Before optimization, it verifies exact input checkpoint hash, base step,
optimizer presence, float32 config, immutable train-pool hash and normalization.
Then the same eight-case Re=10 FEM pilot is run after training; no locked test is
opened and no model-acceptance claim is made. The pilot alone is not the full
400-case development benchmark.

`submission.json` records accepted Kaggle version and initial state;
`launch-observation.json` preserves the initial launch snapshot. See `completion.json` for the verified final state.
The script and metadata are in `../../notebooks/kaggle_adam_next50k/`.

## Historical packaging / monitoring commands

`python -m scripts.build_adam_next50k` recreates only the new launch files from
the preserved previous Adam launch. It does not launch another session. The embedded
physics source is retained verbatim; the source attachment, checkpoint identity,
start/target counts and output directory change.

```bash
KAGGLE_CONFIG_DIR=/workspace/.config/kaggle .venv/bin/python scripts/collect_kaggle.py \
  --kernel mohrezsayeh/geometry-pi-deeponet-adam-next50k \
  --logs --seconds 30 --output runs/adam-next50k-monitor
```

The local preflight successfully loaded the production weights, Adam moments and
RNG; matched config, pool hash and normalization; compiled the packaged script and
verified **132,216 − 82,216 = 50,000**. This validates packaging and local resume
prerequisites, not production GPU progress. New outputs will be under
`project/runs/adam_next50k/`.
