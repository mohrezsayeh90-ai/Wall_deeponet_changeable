# Mean absolute and relative errors for the five examples

[All quantities CSV](metrics.csv) · [Full precision JSON and source hashes](metrics.json) · [Original figures and L2 comparison](../README.md)

**Absolute MAE** = sum(w × magnitude(model − FEM)) / sum(w).

**Relative L1 (%)** = 100 × sum(w × magnitude(model − FEM)) / max(sum(w × magnitude(FEM)), 1e−16).

For velocity, magnitude is the Euclidean norm of the two-component vector (u,v). For pressure or another scalar, it is absolute value. This uses the same positive physical-area weights, common-domain quadrature points and original step-32,216 checkpoint as the five-case L2 report, all at Re=30. No new model inference, FEM solve or training is needed. Original field arrays and figures remain unchanged.

MAE is an absolute error in the corresponding nondimensional field units, **not a percentage**. Relative L1 is a ratio of weighted integrals, not the mean of pointwise percentage errors (MAPE). Pressure is compared without a fitted constant shift. Near-zero reference v can make its relative error large even when absolute v error is small; the reference mean magnitude and denominator-floor flag are included in JSON/CSV. These ratios are not guaranteed to be smaller than relative L2.

## Velocity vector and pressure

| Case | Velocity MAE | Velocity relative L1 | Pressure MAE | Pressure relative L1 |
| --- | ---: | ---: | ---: | ---: |
| 01 Rounded plateau roof | 0.092433 | 10.590% | 0.218332 | 24.615% |
| 02 Waves with changing wavelength | 0.029826 | 2.984% | 0.047837 | 4.405% |
| 03 Rounded stepped S-shaped walls | 0.020163 | 1.917% | 0.037190 | 3.226% |
| 04 Independent irregular spline walls | 0.026160 | 2.626% | 0.035323 | 3.450% |
| 05 Independent random Fourier walls | 0.043543 | 4.179% | 0.026218 | 2.081% |

## Components and speed magnitude

| Case | Quantity | Absolute MAE | Relative L1 |
| --- | --- | ---: | ---: |
| 01 | u | 0.084609 | 9.721% |
| 01 | v | 0.024615 | 50.917% |
| 01 | speed | 0.085026 | 9.742% |
| 02 | u | 0.024782 | 2.483% |
| 02 | v | 0.012325 | 53.889% |
| 02 | speed | 0.025016 | 2.503% |
| 03 | u | 0.018931 | 1.801% |
| 03 | v | 0.004454 | 21.792% |
| 03 | speed | 0.018992 | 1.806% |
| 04 | u | 0.022199 | 2.230% |
| 04 | v | 0.009434 | 52.610% |
| 04 | speed | 0.022336 | 2.242% |
| 05 | u | 0.040098 | 3.860% |
| 05 | v | 0.011367 | 23.179% |
| 05 | speed | 0.039937 | 3.833% |

Reproduce from repository root: `python -m scripts.mean_errors`. All 25 quantity-level MAE/relative identities are verified during generation.
