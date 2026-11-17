# Paired 15,000-update Adam learning-rate trials on molab

**Started; results pending.** The user requested stopping the one-hour baseline and comparing independent LR continuations. The latest intact full-state checkpoint at **step 172,400** was copied and SHA256-verified before the old worker was interrupted. The baseline remains immutable on molab at `/home/marimo/wall_lr_trials/base.npz`. [Manifest](manifest.json) · [Launch observation](launch-observation.json) · [Worker/controller/dashboard source](../../notebooks/molab_lr_trials/).

| Order | Schedule | Updates | Start → target |
| --- | --- | ---: | --- |
| 1 | Constant 2e-5 (existing floor baseline) | 15,000 | 172,400 → 187,400 |
| 2 | Constant 5e-5 | 15,000 | 172,400 → 187,400 |
| 3 | Constant 1e-4 | 15,000 | 172,400 → 187,400 |
| 4 | Cosine 1e-4 → 2e-5 | 15,000 | 172,400 → 187,400 |

All four restart from the same parameter arrays, Adam moments, global step and RNG. They do **not** chain candidate weights. Same batch 32, interior 128, outlet 48, float32, training pool and frozen physics. The copied trainer changes only its LR expression and imports the explicit controller. Compiled schedule checks cover all 15,000 values, positivity, endpoints and cosine monotonicity. Initial sampled-batch hashes and first loss values are compared across completed trials as an additional pairing check.

At launch observation, the first trial had reached **173,030**, with a full optimizer checkpoint saved at **173,000**. The first resumed update was **172,401**. This is not a completed comparison or a claim that any controller is better.

## Fixed evaluation and display

A seeded, immutable development probe contains 16 sampled dev geometries with fixed Re and collocation points. Its objective is evaluated at offsets **0, 5k, 10k, 15k** without entering the training gradients. The baseline probe loss was **0.0136732142**. This probe is a controlled optimization diagnostic, not a full generalization benchmark. After each complete trial, the same eight previously saved FEM cases are evaluated without resolving FEM. Their median velocity and pressure errors are retained alongside loss results.

The existing molab dashboard preserves the earlier loss history and overlays all new branches from the same global step. Thin curves show logged batch loss and bold curves show means over up to 100 logged records. A second plot compares the fixed dev probe. Progress advances across the four sequential trials; total training work is **60,000 updates**. Estimated total duration **75–85 minutes**, including repeated compilation and evaluation, based on observed speed; actual timing is recorded.

An independent report finalizer waits for completion, ranks trials by final fixed-probe loss, retains separate matched FEM metrics, exports PNG/PDF/CSV/JSON, and makes an atomic ZIP available in the dashboard. It does not automatically replace the main model or promote a winner. Any improvement is determined after the actual runs, not from minimum noisy batch loss.

## Files on molab

- `/home/marimo/wall_lr_trials/suite-status.json`: active trial, completed trials and process state.
- `/home/marimo/wall_lr_trials/cases/<case>/`: training.jsonl, probe.jsonl, resume.json, lr.json, full-state checkpoint.npz, status.json and final FEM pilot metrics.
- `/home/marimo/wall_lr_trials/comparison.png`, `comparison.pdf`, `comparison.csv`, `comparison.json`: generated after all cases finish.
- `/home/marimo/wall_lr_trials/lr_trials_results.zip`: final complete bundle; the dashboard download is enabled after verification.

The interrupted older run and its logs/checkpoints remain under `/home/marimo/wall_adam_1h`. The stop was user-requested; an interrupt-related legacy failure marker does not indicate a corrupt saved baseline. The latest collected GitHub checkpoint remains 132,216 until later molab outputs are retrieved and verified. No additional LR trials are started by refreshing the dashboard.
