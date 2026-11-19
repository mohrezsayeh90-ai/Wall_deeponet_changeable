**Both branches completed. [Final analysis, all-case errors and next proposal](completed/README.md).**

# SOAP continuation experiment: two independent 5k branches

Both branches start from archived step **287400**, SHA256 `72bbe88ddc539258b9bfb433709aa434776393bd5f7e163d90478cbf7fa0e6c2`. Parameters are retained; **SOAP statistics start fresh**. The saved Adam state remains in the original checkpoint and is not reinterpreted as SOAP state. Each branch performs5000 parameter updates, ending at continuation step292400, at fixed LR **2e-5** or **1e-4**. One extra gradient evaluation initializes each SOAP basis and makes no parameter update; it does not advance the shared training RNG. No later long run or promotion occurs automatically.

## Fixed inputs and comparison

Batch32,128 interior/48 outlet points, identical baseline sampling and original training RNG. No RAD, architecture change, new loss weight or training FEM target is introduced. Compare against the already completed Adam checkpoint292400 (5000 updates from the same base), avoiding duplicate training. Same disjoint32+32 fixed development cohorts are evaluated every1000 updates. Same20 cached Re30 FEM cases plus8 Re10 pilot cases are evaluated at2500 and5000. Report all case errors, worst errors, medians and wall time; these cohorts do not establish full benchmark acceptance.

SOAP costs include QR/basis updates, initialization, serialization and compilation overhead. Equal-update results are not equal-time results; no new Adam run is claimed. Existing Adam15000 is an additional longer-budget reference, not an equal-updates competitor.

## Implementation and safeguards

Vendored, licensed **unofficial JAX SOAP** implementation pinned to upstream commit `a7c92f174c49b13136294a94ae0d46266b4f671f` (version0.2.2). Source: https://github.com/haydn-jones/SOAP_JAX. This version contains the upstream correction to off-preconditioning EMA updates. No claimed performance improvement is assumed from literature.

SOAP b1=b2=.95, precondition_frequency10, max_precond_dim1024, no1D preconditioning or weight decay. Its internal matrix precision uses the upstream highest setting. Frozen model gradients remain as before. An isolated venv with shared system packages pins JAX/JAXLIB0.11.2, optax0.2.6, chex0.1.91 and jaxtyping0.3.3; the notebook and former training environment are unchanged. Resolved versions and GPU checks are captured in setup logs.

The gradient core returns the original clipped gradient before Adam's update. Production-size preflight verifies agreement with the original loss and Adam first-moment update, finite nonzero SOAP updates, zero-update initialization, exact pickle-free state serialization and the same next update after restore. It checks EMA changes on an off-preconditioning step. Training starts only after these pass. No tests are claimed passed before the preflight file records success.

Full checkpoint NPZs contain standard model fields plus SOAP state leaves, optimizer settings, tree signature and RNG. **`optimizer_present=false` describes absent Adam moments; `soap_state_present=true` describes the actual optimizer state.** Use `restore_full` with the pinned vendor code, not Adam resume. The full SOAP state is essential for continuation. Immutable snapshots follow100,1000,2000,2500,3000,4000,5000 updates; periodic latest state and logs are also maintained. A one-hour branch guard and2.5h supervisor watchdog bound compute.

The independent collector verifies file SHA256 and pushes immutable full-state snapshots during training; it requires this Codex cloud environment and credentials to remain available. A successful push is the evidence of backup. Failed/mutating transfers are retried and completion requires all listed files to be verified. Remote `results.zip` is available after completion; its large download is separate from the individually collected GitHub files.

[Source and pinned SOAP implementation](../../notebooks/molab_soap_trials/) · [Collected outputs](../../artifacts/molab/soap-trials-v1/).

Remote root: `/home/marimo/wall_soap_trials_v1`. Reproduction requires restoring the existing frozen Kaggle source/pool/FEM inputs and the completed RAD-suite references at the paths recorded in the worker. No locked test is opened.

## Launch verification

Production preflight passed: clipped-gradient agreement, finite nonzero SOAP update, exact full-state round trip and identical next update after restore. Off-preconditioning EMA changes were verified. First common-batch SHA256 matches the earlier287400 continuation (`091545e904a2e5998e3dc4d0901ab3bb68807f96d6337254c7504d8c51551357`). A full model+SOAP NPZ is about31.7MB. The first low-rate branch reached its first100 parameter updates; high-rate branch is queued. No convergence result is claimed from launch.
