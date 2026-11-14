# Geometry-conditioned physics-informed DeepONet

اپراتور فیزیک‌آگاه برای جریان پایای دوبعدی در کانالی با **دو دیوارهٔ مستقل** و `10 <= Re <= 100`؛ یک checkpoint برای همهٔ هندسه‌ها. مشخصات درخواست در [سند پروژه](reports/PROJECT_SPECIFICATION.md) حفظ شده است.

**وضعیت: پیاده‌سازی و آزمون‌های اولیه انجام شده؛ اجرای GPU در حال انجام است. مدل آموزش‌دیده و تأییدشده هنوز منتشر نشده است.** [وضعیت ماشین‌خوان](STATUS.json) و [گزارش پیشرفت](reports/PROGRESS.md) میان موفقیت کد، پایان آموزش و صحت میدان تفاوت می‌گذارند. این مخزن بازتولید آمادهٔ حل NS از مقاله نیست؛ سازگارکردن جدید آن برای هندسهٔ متغیر است.

- [اجرای اصلی GPU در Kaggle](https://www.kaggle.com/code/mohrezsayeh/geometry-pi-deeponet-main)؛ خصوصی در حساب مالک.
- [آزمایش دسترسی GPU](https://www.kaggle.com/code/mohrezsayeh/geometry-pi-deeponet-gpu-probe).
- [گالری هشت خانوادهٔ دیواره](reports/geometry_gallery.png).
- [شناسه‌ها، پارامترهای هندسی و هش مجموعه‌های ثابت](reports/split_manifest.json).

## Model and physics

The branch receives 257 samples each of `centerline - .5` and `gap - 1`, plus normalized log(Re). Independent streamfunction/pressure DeepONets use width 192, four gated updates and rank 128. The trunk receives reference coordinates and differentiable local wall features. Walls are C4 quintic splines, flat on the inlet/outlet 10% strips. The production mode has access to the complete wall functions. Sensor-only reconstruction is a separate API and is not claimed equivalent to exact-wall mode.

`x=5*xi`, `y=b(x)+eta*(t(x)-b(x))`. Production autodiff wraps the full inverse map and differentiates in **physical Cartesian coordinates**, including all geometry terms. A separate mapped-derivative path is tested against analytic physical functions.

```text
psi = 3*eta^2 - 2*eta^3
      + (1-exp(-xi/.05))^2 * 16*eta^2*(1-eta)^2 * N_psi
p = 12*(5-x)/Re + N_p
u = dpsi/dy, v = -dpsi/dx
L = mean(Ru^2+Rv^2) + 10*mean(Tx^2+Ty^2)
```

No-slip, prescribed parabolic inlet, unit flux and incompressibility follow from the ansatz. They do not prove momentum balance or field accuracy. Outlet conditions are `du/dx/Re-p=0`, `dv/dx/Re=0`; no extra pressure gauge is imposed. The independent P2/P1 FEM uses this same Laplacian weak form.

## Install and verify

Python 3.12 CPU setup tested in this cloud environment:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r reports/environment-cpu.txt
python -m pytest -q
```

**Do not install the CPU lock over working CUDA JAX on Kaggle.** The self-contained Kaggle script preserves the existing CUDA JAX and installs only scikit-fem and pytest. Precision profiling compares initialization residuals and update timing in float64/float32; the selected precision and its limited comparison scope are logged. Final structural verification uses float64.

The tests cover map recovery, physical chain rules and Laplacian, mixed derivatives, random-correction wall/inlet/flux/divergence, exact Poiseuille momentum/traction, independent FEM, finite training gradients, checkpoint equality, immutable split hashes and sensor-only input rejection. Numerical field validation is separate.

## Generate immutable geometries

```bash
python -m geometry_operator.sampling --output data/geometries
python -m scripts.geometry_gallery
```

10,000 train / 400 development / 1,000 locked test / 200 stress geometries. Each split is family-balanced. Re is independently drawn during training; reference Re values are preassigned and balanced. Near duplicates across splits are screened on 33-point wall vectors including axial and centerline reflections, with a declared RMS threshold. This is a numerical grouping criterion, not proof of distinctness in every possible function-space norm. Re-running generation checks immutable hashes. Training-only normalization is retained.

## Train and resume

```bash
python -m geometry_operator.train --pool data/geometries \
  --config configs/production.json --output runs/main \
  --steps 100000 --batch 32 --points 128 --outlet 48 --require-gpu

python -m geometry_operator.train --pool data/geometries \
  --config configs/runtime.json --output runs/continued \
  --steps 100000 --batch 32 --points 128 --outlet 48 \
  --resume runs/main/checkpoint.npz --require-gpu
```

Use the exact config saved with the checkpoint; `configs/runtime.json` is produced by the Kaggle precision probe. With the unmodified production float64 config, resume with that config instead. Checkpoints retain Adam moments, exact NumPy RNG state, step, wall-pool hash, model config and normalization. One-geometry microbatches accumulate the global batch without reducing it. There is no per-geometry optimizer in inference. FEM labels do not enter training.

The later trainer supports `--devices 2` (global gradient averaging) and `--archive-every 1000` for development checkpoint selection. Matched serial/parallel updates were checked on two virtual CPU devices. The already-running Kaggle v1 is the earlier single-device snapshot, not this later trainer. After archives/references are available, use `python -m scripts.select_checkpoint --candidates runs/main --output runs/selected`; its selection is provisional until all 400 development references are evaluated.

Kaggle's observed quota was **6 GPU hours**, not an unlimited training allocation. The run targets 100,000 steps with a time-budget stop and atomic checkpoints. A time-budget checkpoint remains provisional; the code does not label a partial run completed or accepted. Selecting by full development FEM and implementing the remaining comparisons are outstanding release tasks.

## Kaggle packaging and collection

```bash
python scripts/build_kaggle.py --output notebooks/kaggle_main --kind main
kaggle kernels push -p notebooks/kaggle_main --timeout 20400 --accelerator NvidiaTeslaT4
python scripts/collect_kaggle.py
python scripts/collect_kaggle.py --logs --seconds 30
python scripts/collect_kaggle.py --download --output runs/kaggle
```

The pushed script embeds the exact source archive and revision and is private by default. It runs tests, profiles precision, generates the full pool and starts the global run. The first eight development references after training are a **pilot at Re=10**, not a balanced substitute for the 400-case benchmark. No final test is opened automatically. Live numeric training logs can be read through the Kaggle API. Artifact downloads require network grants to `www.kaggleusercontent.com` and the actual redirect destination, commonly `storage.googleapis.com`; a successful status/log call does not prove artifact download access.

## Independent reference and evaluation

```bash
python -m scripts.reference_batch --pool data/geometries --split dev --workers 2
python -m geometry_operator.evaluate --checkpoint runs/main/checkpoint.npz \
  --pool data/geometries --references data/references --split dev \
  --output runs/main/development
```

Mapped meshes retain inlet/top/bottom/outlet facet tags. The solver uses Reynolds continuation and checks the complete nonlinear velocity-pressure update. Coarse/fine references compare velocity, pressure and pressure drop and refine when necessary. Interpolators reject points outside the mesh. For straight-sided wall approximation, the common interior is explicit and its excluded physical quadrature area is reported, capped at .5%. Wall stress has not yet been verified.

Evaluation saves per-case physical-area-weighted velocity/pressure error, absolute transverse RMS, pressure drop between x=.25 and 4.75, momentum/traction, wall/inlet/divergence/flux, mesh uncertainty, first-call and steady timings, fields and plots. Missing/failed references remain visible and prevent acceptance. Relative-field metrics use a norm floor of 1e-8; pressure-drop denominator uses an absolute floor of 1e-8.

Only after model selection, generate final test references and explicitly open the test:

```bash
python -m geometry_operator.reference --pool data/geometries --split test --unlock-test
python -m geometry_operator.evaluate --checkpoint runs/frozen/checkpoint.npz \
  --split test --unlock-test --output runs/final
```

Opening evaluation records the checkpoint hash and prevents silently using the same locked test for a different model. The full specification additionally requires 80-geometry Re sweeps, stress results and matched baseline/ablation evidence. The evaluator deliberately leaves `release_comparisons_complete=false` until those separate requirements are supplied; no automatic accepted-model claim is possible yet.

## Physical-coordinate API

```python
from geometry_operator.api import FlowOperator

model = FlowOperator('runs/main/checkpoint.npz', allow_unvalidated=True)
# walls: canonical (2,6,80) smooth coefficients; xy: physical (N,2) array
uvp = model.evaluate(walls, reynolds=55., xy=xy)
# Alternative, explicitly reconstructed geometry:
uvp, geometry_diagnostics = model.evaluate_samples(bottom257, top257, 55., xy)
```

The default rejects a checkpoint without a matching passing benchmark report. `allow_unvalidated=True` is for research; it does not certify the prediction. Geometry envelope, Re and physical domain checks run before prediction. Branch embeddings are cached for repeated queries. Generate two example predictions with the same weights:

```bash
python -m scripts.predict --checkpoint runs/main/checkpoint.npz --allow-unvalidated
```

## Sources

Wang, Wang & Perdikaris, *Science Advances* (2021), DOI [10.1126/sciadv.abi8605](https://doi.org/10.1126/sciadv.abi8605); [reference repository](https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets/tree/77e7c747d653b34e79e43df7d88bad87de5c27d8). The user's [fixed-channel research code](https://github.com/mohrezsayeh90-ai/inverse_fromtrajectory_to_FlowReconstruction/tree/92897a6962e537a4bea88042609be102095db037) supplies the architectural/FEM lineage. Its old weights are not used or represented as geometry-generalizing weights.
