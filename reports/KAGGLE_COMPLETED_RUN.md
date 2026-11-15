# Collected Kaggle GPU run

All outputs of [Kaggle v1](https://www.kaggle.com/code/mohrezsayeh/geometry-pi-deeponet-main/versions/1)
are now preserved in [the artifact directory](../artifacts/kaggle/main-v1/README.md).
The collection timestamp and SHA-256 inventory are recorded there. No new training
or locked-test evaluation was performed.

## Completion and provenance

- Kaggle reports `COMPLETE` with no failure message. This means the job ended;
  it does not mean the requested training budget or model acceptance was achieved.
- Final checkpoint: step **32,216 of 100,000**; stop reason `wall_clock_budget`;
  elapsed training time **17,400.489 seconds** (4 h 50 min). **67,784** steps remain.
- Frozen source revision: `f6b32ae7e95853f6e48142e822d6718597eb4708`.
  The downloaded script matches an explicit version-1 source pull byte for byte.
- Kaggle used JAX **0.11.1**, Python **3.13.16**, float32 training. Two CUDA devices
  were visible; the frozen v1 trainer performed single-device accumulated updates.
  The global batch was 32, with 128 interior and 48 outlet points per geometry.
- All **69 output files** and **2 downloaded source/metadata files** are preserved
  in a checksummed ZIP split into two parts, each at most 40 MiB. Geometry data,
  references, executed code, logs, plots, weights, caches and metadata are included.
- All four geometry-pool hashes match the committed immutable split manifest.
  Pool counts are 10,000 train, 400 development, 1,000 locked test and 200 stress.
- Checkpoint SHA-256:
  `a7e9e35b91d6e3c7f3a37404b940289427bfe78f9b38e39661c10382babd5235`.
  Original Adam moments, RNG state, normalization and model config are retained.
- All **3,222** training log records were collected. Logging ends at step 32,210
  with loss **0.0750331655**; the final checkpoint/status record six further steps.
  The final logged loss must not be mislabeled as loss measured at step 32,216.

## Eight-case development pilot

| Metric | Median | P95 |
| --- | ---: | ---: |
| Relative velocity L2 | 2.5874% | 3.5785% |
| Relative pressure L2 | 2.6825% | 5.1849% |
| Relative pressure drop | 2.5277% | 4.3282% |
| Momentum RMS | 0.35536 | 0.57213 |
| Outlet traction RMS | 0.003839 | 0.009728 |

All eight cases are Re=10 and span the eight wall families. They are only **8/400**
development cases. The original report lists **0 failed cases**, but velocity-median,
family-median, momentum, complete-split and final-release gates are false. Wall,
inlet, flux and divergence checks pass in this pilot. These structural results do
not establish field accuracy across the full geometry/Re domain.

The original report, per-case metrics, field arrays and plots are published without
alteration. Its `all_gates_passed=false` and checkpoint `accepted=false` are retained.
No model-selection claim, accepted validation sidecar or final-test result was created.

## Collection verification

- Loaded the original checkpoint, including optimizer state, using current checkout
  code with CPU JAX 0.4.38; all checkpoint arrays are finite.
- Recomputed six physical-coordinate predictions on each of development cases 0
  and 5. Maximum absolute difference from stored GPU predictions was **8.89e-16**
  (asserted tolerance: rtol 2e-5, atol 2e-6). This is a portability smoke check,
  not another full field-accuracy benchmark.
- Current repository test suite: **13 passed**, no failures, in 20.80 seconds.
- Verified pool/checkpoint hashes and complete download pagination.
- Reassembled the archive, verified all 71 file sizes/hashes and restored the
  original layout. Repeated restoration successfully without changing outputs.

## Remaining work

Continue training only as a separate GPU task if the 100,000-step budget is still
required. Then finish all development FEM references and checkpoint selection,
followed by the locked test, Reynolds sweeps and stress benchmark. Matched learned
baselines/ablations and wall-stress verification are also incomplete. Historical
local progress of 27 development references is separate from the eight references
actually returned by this Kaggle job; no reference-generation process is active
in this collection task.

The previous network/authentication blocker was resolved in the published cloud
environment. No environment settings were reapplied or published during collection.
