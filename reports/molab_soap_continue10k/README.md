**Completed at302400. [Final analysis and next proposal](completed/README.md).**

# SOAP continuation: 292400 → 302400, at2e-5

Resume the complete selected SOAP state from `artifacts/molab/soap-trials-v1/soap_2e5/snapshots/checkpoint_292400.npz`, SHA256 `a65fc5a53f906f873b96e20e0dd61c2f169fe6963900d6aa7e037df6adff579d`. Exactly10000 additional parameter updates are targeted, reaching15000 SOAP updates since287400, matching the previously completed Adam15k budget. No new long run or automatic model promotion is introduced.

## Preserved state and checks

Same optimizer configuration: LR2e-5, b1=b2=.95, frequency10, max_precond_dim1024, no1D preconditioning/weight decay. Same pinned optimizer source/environment from the previous SOAP stage. All225 optimizer-state leaves, parameters, normalization and training RNG are restored. SOAP count5001 (including its historical zero-update basis initialization) continues to5002 on the next update. The basis is not reinitialized. A production-batch preflight reserializes/restores state, compares the next sampled batch exactly, and requires identical next updates plus finite state before starting. These checks are considered passed only after recorded in `preflight.json`.

Training still resamples32 geometries, Re10–100,128 interior and48 outlet points per update from the same training pool. No RAD, loss reweighting, architecture or precision change is introduced. The gradient/source and pool hashes are checked.

## Evaluation and stopping

Same32+32 development probes every1000 new updates; same20 Re30 and8 Re10 cached FEM cases every2500. All case errors, medians and maxima are retained. These limited development cases do not validate the entire geometry/Re range; locked test remains unopened.

Heuristic regression guard: if any of the four monitored velocity/pressure medians (20-case and8-case) exceeds1.10 times its292400 value at **two consecutive** FEM evaluations, stop and preserve the current checkpoint. This is a conservative experimental guard, not a statistical significance test. A90min training-stage cap and2h supervisor watchdog bound time. Final comparison explicitly labels early stopping so an incomplete SOAP run is not called matched to Adam15k.

## Storage and dashboard

Full SOAP state is saved after the first100 new updates, every1000 and every2500, as well as on stopping/completion. The independent collector SHA256-verifies snapshots and pushes while the Codex cloud environment and credentials remain available. A successful Git push is the evidence of backup, not an assurance that an ephemeral molab instance is persistent.

[Source](../../notebooks/molab_soap_continue10k/) · [Collected outputs](../../artifacts/molab/soap-continue10k-v1/). Remote root `/home/marimo/wall_soap_continue10k_v1`. Input dependencies, isolated venv, pinned optimizer helper/vendor and frozen references remain in the prior stage roots specified in the worker; restore those archives/source for reproduction. No token is saved in repository files.

New dashboard will show the10000-step continuation and combined15000-step SOAP history alongside the existing Adam15k history. Final plots/metrics and `results.zip` are produced after completion or the regression guard stops the run. No improvement is claimed before observing FEM results.

## Launch verification

Preflight passed on the production GPU: restored parameters and all optimizer leaves exactly, optimizer count5001→5002, identical next sampled batch and next update after a full save/restore. No basis warm-up or optimizer reset occurred. The first resumed training branch reached step292700 (300 new updates) with no dashboard error. This records launch, not a claim of improved FEM accuracy.
