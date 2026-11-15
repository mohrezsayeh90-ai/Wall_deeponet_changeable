# Implementation and experiment status

The original Kaggle v1 job and its 50,000-update continuation are both complete
and fully archived. The latest collected checkpoint is step **82,216**. The
requested continuation target was reached; the original run had stopped at 32,216
because of its time budget. This is **not a
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

## Collected experiment

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
**6 GPU hours**. The run had a 17,400-second training stop budget and stopped at step 32,216
with a provisional checkpoint (optimizer and RNG included).
The time limit does not count as successful completion of the requested budget.

The later GitHub trainer adds periodic inference archives and two-device
gradient averaging for continuation. It is a different source snapshot from
the collected v1 and is not retroactively described as that run's code.
The v1 keeps its latest atomic optimizer/RNG checkpoint every 100 updates.

Historical cloud CPU generation recorded 27/400 development references. That
process is not running in this task; only eight references were produced by Kaggle
and are included in its collected outputs. The operation is restartable and skips verified
caches. Reference files explicitly bind geometry ID and Re. A provenance-only
metadata migration for references generated earlier in this task is recorded;
their numerical arrays were not changed.

## Remaining release requirements

1. Both collections are complete: original and continuation outputs are preserved
   separately. Requested +50,000 updates reached step 82,216. A further 17,784
   updates would be needed only to reach the original 100,000-step research budget.
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

## Collection and current requirements

The previous Kaggle download-network blocker is resolved in the published cloud
environment. Authenticated status is COMPLETE, all 69 output files were downloaded,
and all geometry-pool hashes match the committed immutable manifest. Checkpoint
hash: `a7e9e35b91d6e3c7f3a37404b940289427bfe78f9b38e39661c10382babd5235`.

The current CPU environment loads its optimizer state and reproduces saved GPU
predictions on two development geometries to a maximum absolute difference of
8.89e-16. All 13 current tests pass. Full archive restoration verifies all 71
archived files (69 output plus two source files) and is repeatable.

The eight-case Re=10 pilot has median velocity error 2.59%, median pressure error
2.68%, and median pressure-drop error 2.53%. The velocity-median, family-median,
momentum and complete-split gates fail. Only 8/400 development cases are evaluated;
no final test was opened. See [the collection report](KAGGLE_COMPLETED_RUN.md).

After the completed continuation, reaching the original 100,000-step research
budget would require 17,784 more GPU updates and subsequent independent evaluation. No paid capacity was enabled, no training was restarted,
and no accepted-model claim is made.

## Fifteen illustrative channel result chains

[Complete gallery and numerical results](channel_chain_15/README.md) now cover
15 prescribed geometries from a straight channel to independent mixed walls, all
at Re=30. The fixed step-32,216 checkpoint was used without training or selection
on the locked test. Each chain contains geometry, model u/v/speed/pressure,
independent FEM and absolute-error maps, plus area-weighted metrics.

All 15 coarse/fine FEM checks passed: maximum discrepancy across velocity, pressure
and pressure drop was 0.7205%; maximum excluded common-domain area was 0.2296%.
The straight-channel FEM solution matches the analytic solution within 1.44e-8.
All errors were independently recomputed from saved arrays; all 13 current tests
passed. Geometry, reference identity, 77 PNGs, 15 PDF pages and gallery links were
checked. A cached reference rerun succeeded for all 15 cases.

Median relative velocity error is 2.93% (maximum 7.81%), median pressure error
6.15% (maximum 16.18%), and median pressure-drop error 3.15% (maximum 16.27%).
The complete figures include poorer cases as well as simpler successful ones.
These illustrative geometries at a single Re do not replace the 400-case
development benchmark or the locked final test and do not grant model acceptance.

## Completed 50,000-update continuation

The new run reached **82,216** with `target_completed` after 26,840.254 seconds.
All 68 output files plus two source files are archived separately in
`../artifacts/kaggle/continue-50k-v1`; original weights and previous figures are
preserved. Full training history and paired evaluation are in
[the completed continuation report](continuation_50k/completed/README.md).

On identical eight-case Re=10 FEM references/points/weights, median velocity error
improved 2.59% to 1.83%, pressure 2.68% to 1.05%, pressure drop 2.53% to 0.30%,
and momentum RMS 0.35536 to 0.25713. The family-median and momentum gates still
fail; full development/final-test/comparison requirements are incomplete.
The new checkpoint is research-only and unaccepted. Existing 15/random/stress
examples remain step-32,216 results and were not rerun or relabeled here.

CPU optimizer loading, finite arrays, two-case prediction portability, all 13
current tests, paired reference identity and complete archive restoration were
verified. Full archive restoration was repeated successfully. The exact Kaggle
source was downloaded and matches the submitted continuation script.
