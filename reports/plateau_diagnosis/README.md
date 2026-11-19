# Plateau diagnosis and batch-size probe

User requested investigation/correction of the stalled physics loss. At immutable checkpoint313400, read-only diagnostics evaluated eight independent batch32 draws with identical fixed parameters and compared default versus highest matmul precision. No optimizer update occurred during diagnosis.

## Evidence

Actual losses ranged0.00417–0.02519 on the same weights. The trace of sample gradient covariance divided by squared norm of the eight-batch mean was6.50; this is a limited finite-sample estimate, not a global noise measurement. Individual gradients often had near-zero or negative pairwise cosine. This supports a larger-batch ablation; it does not prove batch size is the sole cause of the plateau.

Highest matmul precision changed that batch loss by only0.00324%, gradient relative norm by3.19%, and direction cosine was0.999505. No large numerical-loss floor was established. Precision is therefore kept unchanged for the batch-size probe, isolating the sampling-noise change. [Data](precision_and_noise.json) · [Diagnostic source](../../notebooks/molab_batch128_probe/diagnose.py).

## Intervention

The original20k worker was SIGSTOP-paused for1061.45 seconds (17.69 minutes), then **SIGCONT-resumed with its full live state** after both batch128 probes failed. No optimizer reset and no duplicate original worker. Pause metadata is archived in [pause_status.json](../../artifacts/molab/soap-continue20k-v1/pause_status.json). Wall-clock durations include this pause and diagnostic/GPU contention, and must not be interpreted as pure training-speed measurements. The original model was not replaced by any rejected probe.

Compare from the exact same complete SOAP state313400:

- Cached batch32 reference:2000 already-executed updates, ending315400.
- New batch128 probe:500 optimizer updates, each using the concatenation of four consecutive original batch32 draws. Both see **64000 geometry draws** and identical Re/interior/outlet draws in the same stream, verified by identical ending RNG states. LR remains2e-5 and all physics/loss weights remain unchanged.

The source SHA256 is `1f66f97353558c44f8af33be70f202bfb034dba9d1749682476893dad8636130`; cached reference SHA256 is `a500fac7a8f6d07e2fdf8058b5d77c1303c45098278d28e664c2b5ffacd1dc25`. The old branch is reused, not re-trained. A preflight verifies that the batch128 gradient/loss equals the mean of four batch32 gradients on the same parameters (when those microgradients do not trigger clipping). The batch128 core clips the combined gradient once with the original norm10 rule.

Compare unchanged control/audit objectives and the same28 cached FEM cases at both endpoints. A smoother training curve alone is not success. Same-data-draw budgets are matched; optimizer update counts differ. Wall time is observed with compilation/evaluation/checkpoint overhead and possible diagnostic interference, not asserted identical.

Candidate checkpoints contain all SOAP state, RNG and explicit batch128/equivalent-batch32 metadata. The final candidate's actual step is313900 (500 updates), but its geometry-draw budget equals the batch32 reference315400. Do not relabel it as2000 parameter updates or merge it into the old step plot without adjusting axes. No automatic model promotion occurs before endpoint comparison.

[Source](../../notebooks/molab_batch128_probe/) · [Collected files](../../artifacts/molab/batch128-probe-v1/). Molab root `/home/marimo/wall_batch128_probe_v1`. Full-state snapshots are collected and SHA256 verified by the independent cloud collector while the cloud environment/auth remain available. This is a bounded500-update probe, not a promise to eliminate the error floor.

## Decision criteria, recorded before endpoint results

A larger-batch candidate is considered promising only if fixed audit is at least1% below the cached batch32 endpoint, control is no more than1% higher, both20-case FEM medians are within2% of reference with at least one improving2%, neither8-case median regresses more than5%, and neither20-case maximum regresses more than5%. These thresholds are heuristic development criteria, not statistical significance or full acceptance. No selection is made from a smoother stochastic training curve.

## First result and bounded LR-scaling follow-up

At batch128/LR2e-5 the audit endpoint was0.01233968 versus0.01230275 for the matched cached batch32 reference;20-case medians were1.3770% velocity and1.2177% pressure versus1.3698% and1.1947%. It did not meet the predeclared gates and is not adopted. [Complete first result](batch128_lr2e5_result.json).

A second and final bounded500-update probe starts from exactly the same state/data stream with batch128 and LR4e-5, testing a square-root-style LR adjustment for the4x batch increase. Same cached batch32 reference, same FEM/probes and decision gates; the original worker was temporarily paused during these two probes and was resumed afterward. The first probe is the same-LR batch ablation; the follow-up changes LR and is labeled separately.


## Final batch-size decision

The LR4e-5 probe also failed: fixed audit0.01245287 (+1.22% vs reference), median velocity1.3609%, median pressure1.2578% (+5.28% pressure error). **Neither batch128 candidate is adopted.** The noisy-gradient observation did not establish the principal plateau cause. [Complete second result](batch128_lr4e5_result.json).

## Weight averaging: small inference-only effect

Uniformly average parameter tensors from311400,312400,313400,314400,315400, accumulated in float64 then cast back to float32. No further training. Compared to315400, audit decreases0.80%,20-case velocity median changes1.3698%→1.3534% and pressure1.1947%→1.1610%. The8-case worst pressure regresses1.5929%→1.7052%; it is not a uniform improvement or a large plateau fix. The averaged file has **no optimizer state** and is not selected for continuation. Original full-state checkpoints remain available. [All cases](weight_average_result.json) · [Inference-only weights](../../artifacts/molab/weight-average-v1/average.npz).

## Fixed-trunk representation diagnostic

Four already-used Re30 development geometries were evaluated. With the current trunk fixed, independent per-case branch coefficients were fitted by weighted ridge least squares to FEM on half of up to3000 sampled points and assessed on the disjoint point half. Current versus projected velocity errors (%): straight0.234→0.000015; triple constriction3.684→0.192; roof plateau7.323→0.073; Fourier5.142→0.106. Current versus projected pressure (%):0.391→~0;1.052→0.165;13.910→0.070;1.477→0.095.

**These are FEM-assisted diagnostic projections, not trained operator predictions, not held-out-geometry results, and not accepted model accuracy.** They do not verify PDE derivatives/outlet traction, nor prove that the branch MLP can learn the optimal coefficient map. Difficult-case coefficient norms are2.5–5.5 times current norms. Nevertheless these observations provide no evidence that enlarging the current spatial basis is the first necessary change. [Data](trunk_capacity_diagnostic.json) · [Source](../../notebooks/molab_batch128_probe/trunk_capacity_diagnostic.py).

## Bounded branch-only ablation

A separate2000-update SOAP trial starts from identical full state313400 and uses exactly the same fresh batch32 geometry/Re/interior/outlet draw stream as the cached315400 reference. Same LR2e-5, same objective and all original constraints. Only psi_branch/p_branch receive parameter updates; both trunks are checked bitwise unchanged on save. The original full SOAP state is restored; branch moments/preconditioners are retained. Trunk gradients/updates are zeroed, so trunk-state accumulators may evolve but no trunk parameter moves. Metadata explicitly records branch-only mode. No FEM labels enter training; no automatic promotion. [Source](../../notebooks/molab_branch_only_probe/) · [Artifacts](../../artifacts/molab/branch-only-probe-v1/).

[Endpoint chart](comparison.png) · [PDF](comparison.pdf) · [CSV](comparison.csv) · Reproduce: `.venv/bin/python scripts/report_plateau_diagnosis.py`.
