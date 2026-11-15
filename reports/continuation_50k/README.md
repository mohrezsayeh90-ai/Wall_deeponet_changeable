# Completed continuation: 50,000 additional updates

**Update:** the final checkpoint at step **82,216** was collected and verified with `target_completed`. See the [complete collection/comparison report](completed/README.md) and [new artifact archive](../../artifacts/kaggle/continue-50k-v1/README.md). The launch observation and initial logs below are retained as historical evidence, not live status.

Private Kaggle run: https://www.kaggle.com/code/mohrezsayeh/geometry-pi-deeponet-continue-50k

- Source: completed `mohrezsayeh/geometry-pi-deeponet-main` v1.
- Resume checkpoint SHA-256: `a7e9e35b91d6e3c7f3a37404b940289427bfe78f9b38e39661c10382babd5235`.
- Start: **32,216**. Requested additional updates: **50,000**. Total target: **82,216**.
- Resume uses original weights, Adam moments, exact saved NumPy RNG state,
  float32 runtime config and immutable geometry pool. It does not initialize a new training run from step zero.
- The executed model/trainer source is the exact embedded v1 code at revision
  `f6b32ae7e95853f6e48142e822d6718597eb4708`. Only the Kaggle launch wrapper changes.
  Single-device accumulated global batch 32, 128 interior and 48 outlet points
  are retained. CUDA JAX is preserved; the original frozen test suite runs first.
- The input is Kaggle's read-only output attachment. Launch verifies the exact
  checkpoint hash, step, optimizer presence, train-pool hash, model config and
  normalization before training. No pool regeneration or precision selection is performed.
- Kaggle accepted version 1 with T4 GPU acceleration and an 11.5-hour session
  timeout. The trainer has a 10-hour training budget to leave room for startup,
  checkpoints and the same eight-case development pilot after training.
- At submission, Kaggle reported 30 GPU hours total and 4.98198 hours used
  (about 25.018 hours remaining). The previous observed update rate implies
  approximately 7–8 hours for 50,000 additional updates; this is an estimate.
- First resumed update is saved, then checkpointing occurs every 100 steps.
  A final atomic checkpoint is saved at target completion or the trainer's time budget.
  An external hard session termination can lose work since the last checkpoint.
- Completion is not asserted until retrieved status/checkpoint show step 82,216
  and `stop_reason=target_completed`. A partial or still-running job is not 50,000
  completed updates. `launch-observation.json` records the most recently observed state.
- No locked-test evaluation or accepted-model claim is made. A decreasing physics
  training loss must be followed by independent field/FEM validation.

## Files and monitoring

`submission.json` records Kaggle's accepted run/version. The [launch script and metadata](../../notebooks/kaggle_continue_50k) and are reproducible with
`python -m scripts.build_continuation`. Rebuilding does not launch the job. Do not
push another version of the running kernel merely to monitor it.

```bash
KAGGLE_CONFIG_DIR=/workspace/.config/kaggle .venv/bin/python scripts/collect_kaggle.py \
  --kernel mohrezsayeh/geometry-pi-deeponet-continue-50k \
  --logs --seconds 30 --output runs/continue-50k-monitor
```

Once complete, download with the same collector using `--download` and a fresh
output directory, verify checkpoint step and pool/source hashes, and preserve the
original run and new run separately. New training outputs will be in
`project/runs/continued_50k/`; the pilot evaluation is its `development/` subdirectory.
Compare final field errors against the original checkpoint before any acceptance claim.

## Verified start

The actual GPU runtime reports JAX 0.11.1 and two visible CUDA devices. The frozen
nine-test suite passed. `RESUME_VERIFIED` confirmed the original checkpoint and
configuration. The **first actual optimization update was step 32,217**, not step 1.
The initial numerical training records are preserved in `initial-training.jsonl`.
At the recorded observation, the run was **RUNNING** and the last logged step was
**32310**. This is a launch snapshot, not live progress or a completion claim.
