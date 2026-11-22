# SOAP continuation:322400 →422400

**2026-11-22 recovery:** the retained notebook reports completion at422400 and its full final plot was recovered. The current Molab instance has no original run directories or final weights/logs. The highest archived checkpoint remains342400. [Recovered figure, approximate loss reductions and exact archive limits](retained/README.md). Do not treat the retained plot as a recovered terminal model or final FEM evaluation.

User requested100000 further updates from the complete final SOAP checkpoint322400, batch32, with a continuous decrease in learning rate from1e-4 to5e-6. The new Molab instance was empty of training files. The original frozen physics source, immutable train/dev pools, fixed probes, cached28 FEM references, pinned SOAP vendor and full checkpoint were restored from the existing archives. All55 transferred files were verified against a SHA256 manifest. No locked test data were transferred or opened.

Source checkpoint SHA256: `768c56f1c08515b3c0aa24d20ed19308cdf977e43746110db6b309c380f2b3e2`. Parameters, normalization, RNG and all225 SOAP-state leaves are restored. SOAP counter35001 continues to35002; the original basis-only initialization is not repeated.

For new update index j=1,…,100000:

`lr(j)=5e-6 + (1e-4−5e-6)/2 × [1+cos(π(j−1)/99999)]`.

The first actual update uses1e-4 and the last uses5e-6. The schedule is continuous, monotonically decreasing and has no restarts or step reductions. The LR scalar is supplied to each JIT update. Internally the restored constant-LR SOAP transformation is retained and its final update is multiplied by `lr/2e-5`; with zero weight decay this preserves the existing optimizer tree and is equivalent to applying the requested LR. Preflight compares the first update against a direct1e-4 SOAP transformation and checks identical optimizer-state evolution. No extra scheduler counter, optimizer reset or stale global-step offset is introduced.

Baseline sampling remains32 fresh geometry draws, Reynolds10–100,128 interior and48 outlet points per geometry per update. All four branch/trunk networks train. Same physics objective, clipping and SOAP hyperparameters as the source: betas.95/.95, precondition frequency10, max dimension1024, no1D preconditioning or weight decay. Isolated runtime: JAX/JAXLIB/CUDA13 plugin0.11.2, optax0.2.6, chex0.1.91, numpy2.5.5, scipy1.18.1. Vendor commit `a7c92f174c49b13136294a94ae0d46266b4f671f`.

Logs every10 updates; fixed disjoint control/audit probes every1000; full-state checkpoint after100 and every5000 updates; same28 cached FEM cases every5000 updates and at completion. The old10%-regression monitor records warnings; this user-requested100k schedule is not automatically stopped for that heuristic. Nonfinite training loss raises an error and leaves prior immutable snapshots intact. Training wall-time cap is8h; a cap stop is recorded distinctly from completion. No universal accuracy improvement is assumed from launching this higher initial LR.

Molab root `/home/marimo/wall_soap_cosine100k_v1`. [Worker and dashboard source](../../notebooks/molab_soap_cosine100k/) · [Collected progress and verified snapshots](../../artifacts/molab/soap-cosine100k-v1/). The independent collector checks sizes/SHA256 and pushes snapshots to GitHub while the Codex environment and connection remain alive; it cannot guarantee collection after either machine stops. The live notebook displays progress, actual applied LR, stochastic loss and unchanged fixed probe losses.

Launch is confirmed only after `preflight.json` passes and `status.json` records real updates; see the collected evidence for the latest observed step.

## Verified launch,2026-11-21

Preflight passed on the restored RTX PRO6000 Blackwell GPU: exact parameter/optimizer roundtrip, identical next RNG batch, identical resumed next update, state counter35001→35002 and agreement with direct SOAP1e-4. First/final schedule endpoints passed and all100000 scheduled rates were checked monotonically decreasing. Training subsequently exceeded322800 (400 new updates) with finite logged losses, GPU utilization90% at observation and no live dashboard errors. This establishes a real launch, not completion or an accuracy gain. Check the collected status/logs for current progress.
