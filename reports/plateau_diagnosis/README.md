# Plateau diagnosis and batch-size probe

User requested investigation/correction of the stalled physics loss. At immutable checkpoint313400, read-only diagnostics evaluated eight independent batch32 draws with identical fixed parameters and compared default versus highest matmul precision. No optimizer update occurred during diagnosis.

## Evidence

Actual losses ranged0.00417–0.02519 on the same weights. The trace of sample gradient covariance divided by squared norm of the eight-batch mean was6.50; this is a limited finite-sample estimate, not a global noise measurement. Individual gradients often had near-zero or negative pairwise cosine. This supports a larger-batch ablation; it does not prove batch size is the sole cause of the plateau.

Highest matmul precision changed that batch loss by only0.00324%, gradient relative norm by3.19%, and direction cosine was0.999505. No large numerical-loss floor was established. Precision is therefore kept unchanged for the batch-size probe, isolating the sampling-noise change. [Data](precision_and_noise.json) · [Diagnostic source](../../notebooks/molab_batch128_probe/diagnose.py).

## Intervention

The original20k continuation worker is **temporarily SIGSTOP-paused**, with its live state preserved and immutable source/matched reference already saved. Original supervisor remains alive. It can be resumed with SIGCONT to its verified process group; do not launch a duplicate worker. Pause metadata is on molab at `wall_soap_continue20k_v1/pause_status.json`. The pause has not reset parameters or discarded previous GitHub checkpoints.

Compare from the exact same complete SOAP state313400:

- Cached batch32 reference:2000 already-executed updates, ending315400.
- New batch128 probe:500 optimizer updates, each using the concatenation of four consecutive original batch32 draws. Both see **64000 geometry draws** and identical Re/interior/outlet draws in the same stream, verified by identical ending RNG states. LR remains2e-5 and all physics/loss weights remain unchanged.

The source SHA256 is `1f66f97353558c44f8af33be70f202bfb034dba9d1749682476893dad8636130`; cached reference SHA256 is `a500fac7a8f6d07e2fdf8058b5d77c1303c45098278d28e664c2b5ffacd1dc25`. The old branch is reused, not re-trained. A preflight verifies that the batch128 gradient/loss equals the mean of four batch32 gradients on the same parameters (when those microgradients do not trigger clipping). The batch128 core clips the combined gradient once with the original norm10 rule.

Compare unchanged control/audit objectives and the same28 cached FEM cases at both endpoints. A smoother training curve alone is not success. Same-data-draw budgets are matched; optimizer update counts differ. Wall time is observed with compilation/evaluation/checkpoint overhead and possible diagnostic interference, not asserted identical.

Candidate checkpoints contain all SOAP state, RNG and explicit batch128/equivalent-batch32 metadata. The final candidate's actual step is313900 (500 updates), but its geometry-draw budget equals the batch32 reference315400. Do not relabel it as2000 parameter updates or merge it into the old step plot without adjusting axes. No automatic model promotion occurs before endpoint comparison.

[Source](../../notebooks/molab_batch128_probe/) · [Collected files](../../artifacts/molab/batch128-probe-v1/). Molab root `/home/marimo/wall_batch128_probe_v1`. Full-state snapshots are collected and SHA256 verified by the independent cloud collector while the cloud environment/auth remain available. This is a bounded500-update probe, not a promise to eliminate the error floor.

## Decision criteria, recorded before endpoint results

A larger-batch candidate is considered promising only if fixed audit is at least1% below the cached batch32 endpoint, control is no more than1% higher, both20-case FEM medians are within2% of reference with at least one improving2%, neither8-case median regresses more than5%, and neither20-case maximum regresses more than5%. These thresholds are heuristic development criteria, not statistical significance or full acceptance. No selection is made from a smoother stochastic training curve.
