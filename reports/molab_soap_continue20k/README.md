# SOAP20k continuation: 302400 → 322400

User requested immediate20000 additional steps. Resume the **complete SOAP state** from step302400, SHA256 `6f38542948ca8a1d4264ef50e85739046a3d3348b8b153704633f0e6addf6630`, preserving parameters, all225 optimizer leaves, RNG and normalization. Fixed LR **2e-5** and unchanged SOAP hyperparameters (betas.95/.95, frequency10, max dimension1024, no1D preconditioning/decay). Optimizer counter15001 continues to15002; no new basis initialization or optimizer reset occurs. Target35000 total SOAP parameter updates since287400.

The same production preflight verifies identical parameters/state after save/restore, identical next sampled batch and identical next update. Finite next-state checks precede training. Do not claim checks passed until `preflight.json` records success. Same pinned vendor, isolated venv and frozen gradient source from the earlier SOAP stages are used; source/pool hashes are verified.

Baseline sampling continues to regenerate32 geometries, Re10–100,128 interior and48 outlet points each update. No loss reweighting, RAD, architecture or matrix-precision change is introduced. The fresh32-case FEM expansion was deferred per user's instruction. Same28 cached FEM cases and fixed32+32 probes remain development monitors; coverage is limited to20Re30 and8Re10 FEM references, not full-domain acceptance. Locked test remains unopened.

FEM every2500 updates, probes every1000. Retain the prior heuristic guard: stop if any of four monitored velocity/pressure medians exceeds1.10 times its302400 value at two successive FEM checks. This threshold is not a statistical test. Training-stage90min cap and2h watchdog bound runtime. A guarded stop is recorded distinctly from completing20000 steps.

Full-state snapshots after100, every1000 and every2500 additional steps, plus final/stopped state. Independent collector verifies sizes/SHA256 and pushes during training while this cloud environment and auth remain alive. Only a successful Git push proves backup. All earlier stages stay preserved; no automatic model promotion occurs.

[Source](../../notebooks/molab_soap_continue20k/) · [Collected outputs](../../artifacts/molab/soap-continue20k-v1/). Molab root `/home/marimo/wall_soap_continue20k_v1`. Reproduction needs the pinned helper/vendor/venv and frozen inputs restored at the prior-stage paths recorded in the worker. No credentials are saved in files.

Live dashboard combines the previous15k and this20k SOAP history. The existing Adam15k curve is explicitly a **shorter-budget reference**, not a matched35000-step comparison. Final metrics and a ZIP bundle are generated after completion/guarded stop. No accuracy gain is presumed from launching.
