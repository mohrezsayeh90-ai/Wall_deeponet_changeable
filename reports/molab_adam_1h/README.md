# One-hour Adam continuation on molab

**Launched and progressing**, from the completed Kaggle **step 132,216** checkpoint. GPU: **NVIDIA RTX PRO 6000 Blackwell Server Edition, approximately 96 GiB**. No L-BFGS weights are used. This report is a launch observation, not a completed-run result.

Start: **2026-11-17 14:13:33 Tehran**. Expected stop: approximately **15:13 Tehran**, after the trainer's 3,600-second budget. Its clock includes JIT compilation; preflight/import time is excluded and the final update/save can slightly exceed the limit. An external watchdog handles a stalled trainer.

[Observed launch/runtime/training status](launch-observation.json) · [Worker and supervisor source](../../notebooks/molab_adam_1h/) · [Launch configuration](../../notebooks/molab_adam_1h/launch-manifest.json).

## Resume and validation

- Exact base SHA256: `dabe75cd4121d1f4c6e2c920dced4c38cf033e2da7b1d74c4805ee90cf1e1544`.
- All 70 files of the Kaggle archive were restored and hash-verified. Training uses the archived, unchanged source from revision `f6b32ae7e95853f6e48142e822d6718597eb4708`.
- Adam moments, step, RNG, training-pool hash, normalization and model configuration were validated. The next update is 132,217. Pool, batch=32, 128 interior and 48 outlet points, float32 and global learning-rate schedule are preserved.
- JAX **0.11.0** on molab, versus **0.11.1** in the preceding Kaggle run. GPU and software differ, so bitwise-identical training trajectories are not promised. Saved-pilot inference at 128 points agreed to **8.88e-16** maximum absolute difference.
- Original frozen training-related checks: **8 passed, 1 deselected**. The independent FEM reference test was excluded because this job performs physics-only training and no new FEM solves.

At the recorded observation: latest logged step **132,300**, loss **0.01141327**, GPU **NVIDIA RTX PRO 6000 Blackwell Server Edition, 94 %, 1979 MiB**. All dashboard cells reported no errors. The checkpoint pointer in GitHub remains step 132,216 until final molab weights are collected and checked.

## Live notebook and outputs

The user's existing molab notebook now opens with an automatically refreshed dashboard (10 seconds), a time-progress bar, current/new steps, latest loss, saved checkpoint and a live logarithmic loss plot with a rolling mean. Dashboard code is collapsed and the visible dashboard is placed first. A download control is available for the completed result bundle. The dashboard cell source is preserved in `notebooks/molab_adam_1h/dashboard_cell.py` and is intended for the live notebook, not standalone execution. Training runs as a separate background process, so UI refreshes do not restart optimization. A retained lock prevents accidental duplicate launches.

Files on molab:

- `/home/marimo/wall_adam_1h/run/checkpoint.npz`: latest atomic checkpoint, including Adam moments and RNG; saved every 100 updates and after the first resumed update.
- `/home/marimo/wall_adam_1h/run/training.jsonl`: loss, residual terms, gradient norm and timing, every 10 updates.
- `/home/marimo/wall_adam_1h/run/status.json`: saved checkpoint step and trainer stop reason.
- `/home/marimo/wall_adam_1h/snapshots/`: additional full-state snapshots approximately every 10 minutes.
- `/home/marimo/wall_adam_1h/supervisor-status.json`: process and completion status.
- `/home/marimo/wall_adam_1h/molab_adam_1h_results.zip`: created after exit, containing checkpoint(s), logs, configuration, preflight evidence and launcher code; downloadable inside the notebook.

The high step ceiling is solely a nonbinding loop limit. This run is governed by **time**, with expected normal stop reason `wall_clock_budget`; the final step count is measured after completion. Model acceptance and improved FEM accuracy are not inferred from training progress.
