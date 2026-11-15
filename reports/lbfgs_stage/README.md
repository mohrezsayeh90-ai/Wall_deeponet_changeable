# L-BFGS refinement after the completed Adam continuation

Run: https://www.kaggle.com/code/mohrezsayeh/geometry-pi-deeponet-l-bfgs-refinement

This is a separate private experiment, initialized from the original research
checkpoint at **Adam step 82,216**, SHA-256
`84276c11feecd5bf230c79d1077250018b356ae118491678a4ed69e46ab83140`.
The old weights and reports are preserved. No additional Adam updates are run.
The launch state is recorded in `launch-observation.json`; it is a snapshot,
not an automatically refreshed live status or a convergence claim.

## Optimizer and objective

- SciPy **L-BFGS-B with no bounds**, i.e. an unconstrained limited-memory BFGS
  refinement with line search, not the bound-constrained variant applied to physics.
- Parameters and normalization are promoted from their stored values to float64
  for more reliable line search. The architecture and original frozen v1 physics
  code remain unchanged. CUDA JAX is preserved.
- **512 distinct train-pool geometries**, fixed Re draws, 128 fixed interior points
  and 48 fixed outlet points per geometry define one deterministic objective.
  This is 65,536 interior and 24,576 outlet queries per function evaluation.
- A **disjoint 128-geometry train-pool probe bank**, with separate fixed Re/points,
  is evaluated before/after. Its values do not enter gradients, line search or
  checkpoint selection. It checks generalization away from the optimization bank;
  it is not the development FEM benchmark.
- Seed 2026111521; banks and exact train indices are saved as NPZ for reproduction.
  Training loss is the mean of momentum squared plus 10 times outlet traction
  squared, identical to the Adam objective's mathematical terms. No gradient
  clipping is applied to the derivative supplied to SciPy. No FEM labels enter
  optimization, and no locked-test data are used.
- Limit: **500 accepted optimizer iterations**, **1,000 function evaluations**,
  or **6 hours** of training, whichever stops the optimizer first. One iteration
  may use several function evaluations during line search. The session timeout
  is 11.5 hours to leave room for compilation, probes and the same eight-case FEM
  pilot. Limits are upper bounds; convergence or line-search failure may stop earlier.
- `ftol=1e-12`, `gtol=1e-7`, history length 20, up to 30 line-search trials.
- Checkpoints save the initial point, first accepted iterate, every five accepted
  iterates and final accepted point. A rejected line-search trial is never stored
  as the final model. Abrupt session termination can lose work since the last save.

**Step semantics:** the checkpoint retains `step=82216` as its base Adam count.
`lbfgs_iterations` and `function_evaluations` record this new phase separately;
L-BFGS iterations are not equivalent to Adam steps or epochs.
Adam moments are intentionally not continued. SciPy curvature history is not
serialized; a parameter-only restart can restart L-BFGS, but is not an exact
optimizer-state continuation of a previous L-BFGS run.

## Judging whether it helps

A falling fixed-bank loss alone can reflect specialization to that bank. Compare
the final probe loss and the **same eight development FEM cases** against the
base Adam checkpoint. The previous evaluation outputs are copied to
`project/runs/base_adam/development/`; new outputs are in
`project/runs/lbfgs/development/`. Original reference caches and the frozen v1
evaluator are retained, so the geometry, Re=10, points and reference labels match.
The output report labels the optimizer stage and accepted iteration count.

This changes optimizer, precision and sampling policy together; it is an
exploratory refinement, not a controlled optimizer-only ablation. It cannot
establish model acceptance, full-domain accuracy or full 400-case development
performance. Results may improve or worsen on geometries outside the fixed bank.

## Local validation and reproduction

The 15-test current CPU suite passed. New tests verify objective agreement with
the existing trainer, a finite-difference directional gradient, deterministic
repeatability, disjoint reproducible banks and actual short L-BFGS loss reduction.
A small end-to-end CPU smoke run checked SciPy callbacks/checkpointing: three
accepted iterations reduced fixed-bank loss from 0.0762902 to 0.0184997, while its
probe changed from 0.0806432 to 0.0785856. **This tiny-model smoke test is not a
result for the production checkpoint.** See `local-validation.json` and test logs.

Rebuild the standalone script with `python -m scripts.build_lbfgs_kaggle`.
It embeds the exact frozen original source plus the new L-BFGS module/tests;
no credentials or local datasets are embedded. Rebuilding does not launch a run.
Do not push another version merely to monitor the already submitted job.

```bash
KAGGLE_CONFIG_DIR=/workspace/.config/kaggle .venv/bin/python scripts/collect_kaggle.py \
  --kernel mohrezsayeh/geometry-pi-deeponet-l-bfgs-refinement \
  --logs --seconds 30 --output runs/lbfgs-monitor
```

When complete, collect the separate outputs and inspect `stop_reason`, accepted
iterations, fixed/probe losses and paired FEM metrics. An API COMPLETE state alone
must not be described as numerical convergence or improved solution accuracy.

## Restart: version 2

The user canceled the queued version 1. Cancellation was verified as
`CANCEL_ACKNOWLEDGED` before resubmission. Version 2 was submitted with
identical code (SHA-256 `812b55726c58afeecfa0279a67277509b07ef39fdc21ad7420774fc6fb8980ae`), base weights and settings.
Status immediately after submission: **QUEUED**. See
[resubmission evidence](resubmission-v2.json); this is a snapshot.
