# SOAP learning-rate study:50000 total parameter updates

User authorized a50000-update experiment before choosing a200000-update continuation. Restore the latest **available** complete checkpoint342400, SHA256 `b49cc0e7c38809a5fe9f9913ae24c35563c6d8c5287af0bfe0aec283a7522018`. The missing422400 model cannot be used; its cached plot is not a checkpoint. This tunes continuation of an already-trained operator, not the best LR for training from scratch.

## Why test1e-4 instead of assuming it is best?

The original cosine run's exact archived fixed-control loss rose0.01240382→0.01334285 in its first1000 updates (about7.6% worse); audit rose0.01248427→0.01324098 (about6.1% worse). Its first5000-update control was still0.01316424, while audit was0.01200599. The full retained image eventually shows improvement, but fast early decrease from a transient peak does not establish faster progress versus a lower-LR policy. That motivates matched short screens plus a longer constant-LR observation.

## Budget and cases

| Case | LR policy | Updates |
|---|---|---:|
| constant_1e4 | Fixed1e-4 |5000 |
| constant_2e5 | Fixed2e-5 |5000 |
| constant_5e5 | Fixed5e-5 |5000 |
| constant_2e4 | Fixed2e-4 |5000 |
| cosine_1e4_2e5 | Continuous cosine1e-4→2e-5 |5000 |
| constant_1e4_extend25k | Continue the first case's full state at fixed1e-4 |25000 |
| **Total** | **The fixed1e-4 trajectory has30000 updates total** | **50000** |

All five screens start from exactly the same parameters, complete225-leaf SOAP state, normalization and RNG. Each sees the same5000 fresh baseline batch draws, not fixed collocation. Base optimizer count55001 continues to55002. The extension resumes the first case's347400 full state/count60001 and ends372400; no reset or repeated basis initialization. Common-screen ending RNG states are compared at completion. Additional preflight gradient evaluations do not consume training RNG or apply model updates and are recorded separately from the parameter-update budget.

Batch32, Reynolds10–100 sampling,128 interior and48 outlet points, hard constraints, loss/clipping, precision and SOAP betas.95/.95/frequency10/max dimension1024 remain fixed. The tested factors are starting LR and constant-versus-cosine policy. Changing betas/batch/physics simultaneously would confound this LR study. The compressed5000-update cosine schedule tests a short-run policy; it does not validate a200000-update decay horizon.

## Evaluation and decisions

Log training loss every10 updates; fixed control/audit every1000; complete snapshots after100 and every2500;28 cached FEM cases every2500 and at endpoints. Compare all five methods at the **same5000-update budget**, using fixed control, audit trajectory, per-case FEM velocity/pressure errors, medians/worst cases and relative reduction per update. The constant30000 trajectory is a separate longer-budget observation. No winner is selected from one low stochastic loss or from unequal-budget endpoints.

The finalizer produces an equal-budget CSV, control/audit plots and complete metrics; the minimum-control candidate is labeled only as such. It does not automatically authorize or launch200000 updates. A large-run choice must account for FEM regressions, stability and longer-run behavior. Repeatedly inspected control/audit/FEM are development cohorts, not blind tests. The28 FEM references cover20Re30 and8Re10; they do not establish generalization across allRe10–100. Locked test stays unopened.

Nonfinite loss or failed preflight stops the sequence with retained prior snapshots; it is not silently restarted. The old FEM-regression monitor records warnings. A partial run is reported as partial, and50000 is a maximum planned budget, not proof that it completed. Each worker has an8h wall-time cap.

## Execution and preservation

Molab root `/home/marimo/wall_soap_sweep50k_v1`; isolated pinned GPU runtime and vendor match the prior SOAP experiment. All56 restored input files were verified against a SHA256 manifest. The first production preflight passed: exact weights/state serialization, identical next RNG batch, identical next restored update, direct-LR update equivalence, and no optimizer reset. The live notebook shows the active case, cumulative budget, training means, fixed-control curves and actual rates.

[Source](../../notebooks/molab_soap_sweep50k/) · [Raw progress and snapshots](../../artifacts/molab/soap-sweep50k-v1/). The collector checks snapshot length/SHA256 and pushes while the Codex cloud process/auth remain alive. **It cannot guarantee uploads after the chat environment stops.** Molab's retained notebook image alone does not preserve run files or weights; download completed result files before stopping its instance. Historical stages and the recovered100k image stay preserved in their existing directories.
