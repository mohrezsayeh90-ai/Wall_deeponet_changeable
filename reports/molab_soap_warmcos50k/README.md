# Smooth warm-up then cosine:392400→442400

User requested50000 additional SOAP updates from the completed latest run: smooth increase1e-4→5e-4 in5000 updates, then cosine decay5e-4→8e-5 in45000 updates. The complete392400 state is preserved: SHA256 `fba423df7dd80c9eeea4082e1d12d2167dfe707fede76c00162a6dc275025c72`, SOAP counter105001, normalization and sampling RNG. Final target442400, counter155001. Batch32 and the physics training workflow stay unchanged.

For update j=1…5000: `LR=1e-4+(5e-4−1e-4)/2×(1−cos(π(j−1)/4999))`.

For update j=5001…50000: `LR=8e-5+(5e-4−8e-5)/2×(1+cos(π(j−5001)/44999))`.

This gives exactly1e-4 at the first update,5e-4 at5000 and5001, and8e-5 at50000. Both segments have zero endpoint slope; the transition has no jump. Preflight checks the endpoints, monotonicity in each segment and equality at the join, exact optimizer/model restoration, identical next batch/update after serialization, and equivalence to the direct requested SOAP rate. No new optimizer/basis initialization occurs.

The previous50000 high-LR updates completed at392400 and their final full-state checkpoint was downloaded and its SHA256 and finite56 parameter/225 SOAP leaves verified before continuation. No missing historical422400 state is involved. This is a further authorized50000-update continuation, not a restart from the old common342400 anchor.

FEM evaluation stays in a separate read-only process. The trainer never loads FEM labels or uses FEM to change LR, sampling, gradients, loss, stopping or model selection. The same20 Re30 geometries and identical12 higher-curvature subgroup are checked against original geometry bounds and displayed with per-case velocity/pressure and u/v statistics. The same previously documented zero-reference-v exclusion applies. These are development comparisons, not full-domain acceptance.

The existing training/probe/LR and FEM displays retain the earlier runs and append the new continuation on the cumulative axis from342400; the new stage begins at50000 on that axis. Its own progress card starts at0/50000 from392400. Snapshots after100 and every2500 updates; fixed probes every1000; FEM on every immutable snapshot. Latest-state download is available in Molab.

The independent collector verifies and pushes files while the Codex environment/auth stay alive. Cached notebook pictures alone do not preserve weights. Training errors/nonfinite state and the8h time cap are explicit stops, not silent restarts. The old unconditional sweep extension remains paused.

[Source](../../notebooks/molab_soap_warmcos50k/) · [Raw progress and weights](../../artifacts/molab/soap-warmcos50k-v1/). Molab root `/home/marimo/wall_soap_warmcos50k_v1`. Launch/schedule validity does not establish an accuracy improvement.
