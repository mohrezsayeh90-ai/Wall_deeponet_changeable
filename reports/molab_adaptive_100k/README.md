# Adaptive Adam continuation: 287,400 → 387,400

**Launched; completion and accuracy results are pending.** Continues the final previous checkpoint (step **287,400**, final learning rate **2e-5**) with its exact Adam moments and RNG. The final previous checkpoint was chosen because it had better velocity metrics on the saved 20 cases and better median metrics on the eight-case FEM pilot than the old best-probe checkpoint. That old checkpoint remains preserved; no universal superiority is claimed.

[Launch observation](launch-observation.json) · [Manifest](manifest.json) · [Driver, supervisor, dashboard and setup source](../../notebooks/molab_adaptive_100k/).

## Bounded adaptive controller

- Exactly **100,000 accepted updates**, target global step **387,400**. First 10,000 updates use the previous terminal rate **2e-5**.
- At accepted offsets **10k, 20k, …, 90k**, compare current rate, half rate and double rate, bounded to **[2e-6, 4e-5]**. Duplicate boundary candidates are removed. High 1e-4 rates were not retained because the previous constant-rate experiment performed worse.
- Each candidate takes **500 updates from identical anchor parameters, Adam moments, global step and RNG**. Discarded branches never become the next anchor. The winner's parameters, moments and RNG are adopted; its 500 updates count toward the 100k accepted path.
- Switch rate only when the candidate's fixed control objective is at least **0.5% lower than the same-anchor current-rate pilot**; otherwise retain the current-rate pilot. The controller can raise or lower LR without jumping back to 1e-3 or restarting Adam.
- Up to **9,000 extra discarded pilot updates** add computational cost. All branches and decisions are logged, with accepted progress and total physical updates reported separately. Extra probes and evaluation may affect duration.

This is a deliberately limited development experiment, not a proven globally optimal learning-rate policy. A 500-update pilot and a small fixed cohort can misrank long-term behavior. The 0.5% switching threshold is a heuristic, not a statistical significance test. No guarantee of improved FEM accuracy is implied.

## Control and independent monitoring

A seeded draw selects **64 unique development geometries**, split into **32 control** and **32 audit** geometries with disjoint IDs. Re and collocation arrays are fixed. Both use the existing momentum + 10×outlet-traction objective. The **audit objective is not consulted for LR decisions**. It supplies an additional development monitor, not an unopened final test; saving its best checkpoint is retrospective model selection, not a claim of an unbiased test result. Locked test data are not opened.

Control/audit are evaluated every **5,000 accepted updates** and after a selected pilot. Latest checkpoint is saved every 100 accepted updates; best-control and best-audit full-state checkpoints are kept separately, as are 10k snapshots. The same eight previously saved FEM pilot cases are evaluated for final/best-control/best-audit weights after training. No FEM labels enter gradients.

## Verified execution

Base SHA256: `72bbe88ddc539258b9bfb433709aa434776393bd5f7e163d90478cbf7fa0e6c2`. Archive restoration verified 70 upstream files. Adam moments, configuration, pool SHA256 and normalization passed checks. GPU: RTX PRO 6000 Blackwell (~96 GiB).

Only the LR input in the frozen core was changed: the original schedule expression is replaced by an explicit dynamic scalar. A production-sized first-batch check on the new GPU found the dynamic core at 2e-5 numerically equivalent to the original update within tolerance. At 4e-5, gradients and moments matched the 2e-5 update, and the parameter displacement scaled by **1.999999**. Disjoint probe IDs were verified. JAX on the new molab runtime is **0.11.2**, versus 0.11.0 previously; bitwise-identical trajectories across runtimes are not promised.

At launch observation, the first resumed update **287,401** was verified, last logged step **287,810**, checkpoint **287,800**, and all dashboard cells had no errors. Baseline control objective **0.01294696**, audit **0.01334894**. These are new cohorts and should not be numerically compared as if identical to the older 16-geometry probe.

## Live outputs

Existing notebook now shows accepted progress, actual work, current/candidate learning rate, loss curve, control/audit curves and controller decisions. Refreshing does not relaunch training. Files on molab:

- `/home/marimo/wall_adaptive_100k/run/checkpoint.npz`: latest state, Adam and RNG.
- `run/best_control.npz`, `run/best_audit.npz`: separately selected weights.
- `run/training.jsonl`, `run/validation.jsonl`, `run/controller.jsonl`: accepted training path, fixed metrics and LR decisions.
- `candidate_*.jsonl`: trial trajectories, including discarded work.
- `snapshots/checkpoint_*.npz`: 10k accepted-step states.
- `status.json`, `supervisor-status.json`, `core-check.json`, `probe_ids.json`: runtime and reproducibility evidence.
- `adaptive_results.zip`: produced and CRC-verified after successful completion; download appears in the notebook.

Estimated duration approximately **2–3 hours**, based on observed update speed plus pilot/evaluation overhead. A 5h process watchdog is a fallback, not a requested extension. GitHub's collected research-checkpoint pointer stays at the previous completed run until new outputs are collected and checked.
