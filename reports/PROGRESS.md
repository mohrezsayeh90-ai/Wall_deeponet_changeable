# Implementation and experiment status

This is an initial implementation with a running full-size experiment, **not a
completed accuracy-validated research release**. See `../STATUS.json` for the
last observed training step and reference-generation progress. Training loss is
not a substitute for independent field errors.

## Verified

- Authenticated GitHub read and push to the requested project's `main`.
- Authenticated Kaggle access and a completed GPU probe: JAX 0.11.1, two CUDA
  devices, float64 calculation verified. The requested machine shape was T4.
- Thirteen local tests passed. Nine tests in the initial frozen source bundle
  passed inside the actual Kaggle GPU runtime.
- Geometry/map, physical chain rule, Laplacian, exact streamfunction constraints,
  Poiseuille PDE/outlet, optimizer gradient, serialization, immutable pool and
  reference identity protections are tested.
- A two-update CPU smoke train produced a reloadable checkpoint and physical API
  predictions. This smoke checkpoint is not the main trained result.
- A matched serial/two-device gradient-averaged Adam update agreed within
  5.6e-17 on two virtual CPU devices. The two-device path has not been validated
  on Kaggle GPU yet; the current v1 run uses one device.
- Local train/dev/test/stress pools contain 10,000/400/1,000/200 geometries.
  The immutable local manifest is committed. The remote run generated the same
  declared counts; exact artifact hashes still need collection/comparison.
- Eight independent curved-channel FEM pilot references passed coarse/fine
  velocity, pressure and drop checks below 1%. All pilot cases are Re=10 and
  span the eight families. They are not the complete development benchmark.
- The zero-correction analytic lift baseline on these eight cases has median
  velocity relative L2 **6.52%**, P95 **13.27%**, median pressure error **10.26%**.
  Structural constraints pass while momentum and field-accuracy gates fail.
  Raw numerical references and metrics/plots are retained beside this report.

## Running experiment

Kaggle: https://www.kaggle.com/code/mohrezsayeh/geometry-pi-deeponet-main/versions/1

Version 1 embeds source revision `f6b32ae`, complete-width architecture, 257 wall
sensors, rank 128, four gated updates, global batch 32, 128 interior and 48 outlet
queries. Its live logs show actual optimization updates and finite losses.

Precision profiling selected float32 after an initialization residual comparison
with float64: maximum absolute discrepancy about 1.08e-6. This check covers three
points at initialization; it is not a final precision-accuracy guarantee.
Production structural evaluation remains float64.

The single-geometry profile initially estimated about 31 hours for 100k updates.
The measured global accumulated update is faster, about .53 seconds, implying
approximately **15 hours** for 100k updates. Available Kaggle quota at launch was
**6 GPU hours**. The run therefore has a 17,400-second training stop budget and
will retain a provisional checkpoint if it reaches the budget before 100k.
The time limit does not count as successful completion of the requested budget.

The later GitHub trainer adds periodic inference archives and two-device
gradient averaging for continuation. It is a different source snapshot from
the already-running v1 and is not retroactively described as that run's code.
The v1 keeps its latest atomic optimizer/RNG checkpoint every 100 updates.

Independent references for all 400 development cases are being generated on the
cloud CPU in a separate process. The operation is restartable and skips verified
caches. Reference files explicitly bind geometry ID and Re. A provenance-only
metadata migration for references generated earlier in this task is recorded;
their numerical arrays were not changed.

## Remaining release requirements

1. Collect main checkpoint, exact remote pool manifests, runtime metadata and
   complete training logs; resume the global run if its budget is exhausted.
2. Finish development references and choose one checkpoint using development
   evidence. Do not use the locked test for model/hyperparameter selection.
3. Run the complete locked benchmark, the predeclared 80-geometry Reynolds sweep,
   stress cases, and record all failures and reference uncertainty.
4. Complete the matched learned baselines and sensor/local-feature/constraint
   ablations requested in the specification. Only the analytic-lift comparison
   has been executed so far.
5. Verify wall stress and sensor-only reconstruction/flow accuracy on nontrivial
   held-out walls. The current sensor-only API is an implementation, not a
   validated sensor-only accuracy claim.
6. Publish frozen weights, full case-level metrics and representative prediction
   examples only after evaluating them. The default API rejects unaccepted
   weights; explicit research use does not change acceptance.

## External requirements

The current cloud proxy permits Kaggle API/status/log access, but downloading
the generated artifact from `www.kaggleusercontent.com` returned a proxy 403.
Network domain additions were saved in the environment draft for
`www.kaggle.com`, `api.kaggle.com`, `www.kaggleusercontent.com` and
`storage.googleapis.com`; the tool also preserved/added `kaggle.com` for the
existing binding. The current runtime does not yet show the artifact domains
as applied. The user needs to review/save the environment settings and publish
the environment before retrying that download path. Saved configuration is not
proof of active runtime access.

Finishing the 100k-update budget also needs more GPU time than the observed
current allocation, whether by renewed Kaggle quota or another GPU environment.
No paid capacity was enabled. GPU training and CPU reference generation are
still running; neither is represented as an accepted final model.
