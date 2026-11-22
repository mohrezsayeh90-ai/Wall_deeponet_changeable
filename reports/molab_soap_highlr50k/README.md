# New50000-update high-LR SOAP run

[Same-case comparison at374900 versus the preserved100k Adam model and the available partial SOAP100k checkpoint](versus100k/README.md). The terminal SOAP422400 model remains unavailable; it is explicitly not used or substituted silently.

User requested an additional50000-update run with continuous cosine LR **1e-3→1e-4**, overlaid on the existing comparison plots with the new update axis starting at zero. This run restores the **same common full checkpoint342400**, not the paused extension, random initialization, or missing422400 weights. Full SOAP counter55001 and RNG are retained; the target is392400. This is a newly authorized50000-update budget, separate from the previously held experiment. The previous unconditional extension remains paused.

For new update j=1,…,50000: `lr(j)=1e-4+(1e-3−1e-4)/2×[1+cos(π(j−1)/49999)]`. First actual LR1e-3, final1e-4; no resets, step reductions or adaptive LR decisions. Batch32 and the original fresh geometry/Re/interior/outlet sampler, hard constraints, loss, clipping, precision and SOAP hyperparameters remain unchanged. The source SHA256 is `b49cc0e7c38809a5fe9f9913ae24c35563c6d8c5287af0bfe0aec283a7522018`.

## FEM answers are strictly for display and comparison

The trainer and FEM evaluator are **separate processes**. The trainer reads only geometry pools, normalization, saved optimizer/model state and fixed physics-only control/audit probes. Its source contains no FEM-reference loading, truth arrays, FlowOperator inference or FEM evaluation calls. Gradients and optimizer updates use only the original PDE/outlet objective. Every case uses the same prescribed LR schedule and retains periodic states; **FEM does not influence LR, loss, gradients, sampling, stopping or checkpoint selection**.

The evaluator opens immutable saved checkpoints, predicts with geometry/Re/coordinates, and then compares detached NumPy predictions with cached FEM fields. The true FEM u/v/p values are not network inputs. It has no optimizer or parameter-update path and writes only error metrics/cohort metadata. No new FEM labels or solved flow data were added to the training dataset.

All20 saved channels atRe30 are retained, including multiple constrictions, irregular walls, rounded step, plateau and Fourier examples. Each geometry must pass the original `validate(...,stress=False)` constraints: gap0.60–1.50, center displacement≤0.35 and slope≤0.7 plus the original smoothness/endpoint checks. The12 highest-curvature geometries form a predeclared geometry-based subgroup (60% of the cohort), selected before evaluating the new checkpoint; high curvature is a difficulty proxy, not proof of difficult flow. Individual-case curves show **all20**, including complex low-curvature cases outside that subgroup. Geometry IDs, descriptors and subgroup membership are in `fem_cohort.json`. No out-of-envelope stress example is silently substituted.

Weighted physical-area relative L2 errors are reported separately for vector velocity, u, v and pressure. A relative component error with essentially zero reference norm is recorded as undefined (`null`), not given a fabricated denominator; plot medians omit it and valid counts are stored. This mostly concerns straight-channel v. Both all20 and higher-curvature12 medians/worst cases and individual errors are saved. This cohort tests Re30 only and does not validate allRe10–100.

## Display and preservation

The live notebook retains old training/control/LR curves, adds the new trajectory from zero and displays FEM velocity/pressure comparisons on the same geometry subgroup. Additional panels show new u/v errors and all individual velocity/pressure case trends. Curves stop at actual observations; old5000-update branches are not extrapolated to50000.

Complete snapshots after100 and every2500 updates; control/audit every1000; read-only FEM evaluation at the base and each immutable snapshot. Numeric FEM histories are under `fem/`. A direct-download button exposes the latest saved full state. The collector verifies size/SHA256 before pushing weights/metrics while Codex cloud and authentication remain active. It cannot guarantee collection after this environment stops; cached notebook pictures do not preserve weights.

Training stops on nonfinite loss/state or its8h wall-time cap, with prior immutable states retained. FEM error itself never stops this run. Launch/preflight is not evidence that the higher LR improves accuracy. The new high rate is experimental; evaluate observed losses/FEM rather than assuming improvement.

[Sources](../../notebooks/molab_soap_highlr50k/) · [Collected weights and metrics](../../artifacts/molab/soap-highlr50k-v1/). Molab root `/home/marimo/wall_soap_highlr50k_v1`.

## Verified launch

Complete-state roundtrip, identical next sampled batch, identical resumed update and agreement with direct SOAP1e-3 passed; optimizer count55001→55002. The first and last rates are1e-3/1e-4. Training passed1500 updates. All20 geometries passed original bounds. Baseline and first saved snapshot FEM metrics were produced by the separate evaluator. Additional isolation/baseline checks and browser rendering evidence are collected when available; see `evaluation_isolation_checks.json` and `dashboard.png`.
