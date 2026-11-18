# Improving stalled PINN / DeepONet training: model-specific research review

Reviewed **18 November 2026** against the current geometry-conditioned, steady 2D Navier–Stokes DeepONet. This review changes no live training configuration. It uses author-maintained research code, documentation and linked paper descriptions; it is not a claim to have read every full paper or to have verified published gains on this task. [Source provenance](sources.json).

## What our evidence supports

- The operator has gated modified-MLP branch/trunk networks, rank128, float32 parameters, and hard wall/inlet/flux enforcement through a streamfunction. It is not a plain unconstrained MLP PINN.
- Strong-form momentum requires derivatives through the streamfunction up to third spatial order. Numerical conditioning and precision therefore deserve attention.
- Twenty illustrative Re=30 cases have improved median FEM errors, but a varied roof case still has roughly 7.8% velocity and 14.2% pressure error for one collected checkpoint; the earlier two-step Re=100 case missed recirculation. The gap between average and difficult-case performance motivates localized diagnostics, not a claim that network capacity is definitely insufficient.
- Current sampling already resamples points and includes a fixed near-wall/axial bias. It does **not** rank candidate points or training geometries by the current residual.
- The current adaptive LR run adjusts only scalar step size. It does not change point placement, gradient conditioning or network expressivity.

## Prioritized options

| Priority | Method | Fit to this task | Continue existing weights? | Main caution |
| --- | --- | --- | --- | --- |
| First diagnostic | Explicit highest matmul precision | JAX-PI flags sensitivity to matrix precision; our code does not explicitly set it | Yes; keep Adam state, record the precision change | Not proven to be the present bottleneck; test residuals/gradients and wall time |
| 1 | Residual-adaptive sampling, RAD / RAR-D | Allocate points to unresolved momentum features and difficult **training** geometries, especially separated/near-wall regions | Yes; Adam state can be retained, but the training distribution changes | Retain broad geometry/Re coverage; validate physical errors independently |
| 2 | SOAP preconditioned optimization | A scalar LR cannot correct very different gradient scales/directions | Yes, from the same parameters; SOAP needs its **own** initialized optimizer state | Adam moments cannot simply be relabeled SOAP state; steps may be slower/more memory hungry |
| 3 | Gradient-norm / NTK loss balancing | Test imbalance between momentum components and outlet traction | Yes, as a separate controlled objective change | Measure per-term gradients first; loss magnitudes alone are insufficient |
| Architecture branch | PirateNet / multiscale Fourier trunk | Improve optimization and representation of local, multiscale structures | Not a drop-in load into the existing parameter tree; retraining, distillation or a carefully constructed residual adapter is needed | Must remain a geometry-conditioned operator, preserve hard constraints and smooth derivatives |

## 1. Residual-adaptive sampling: best initial model-specific experiment

The author repository for Wu et al. (CMAME 2023) includes RAD and RAR-D. Its Burgers RAD example evaluates candidate PDE residuals and samples with probabilities proportional to `|residual|^k / mean(|residual|^k) + c`. This supports sampling where the equation is currently poorly satisfied. It does **not** prove equivalent gains for geometry-conditioned Navier–Stokes; its example also changes training stages, so its aggregate results cannot be attributed to sampling alone without a matched ablation.

For our operator, use a bounded mixture such as 50% current baseline sampler and 50% residual-adaptive points, refreshed periodically. Both the 50/50 split and refresh interval are proposed experiment settings, not paper-prescribed optima. Rank points separately across selected geometries/Re, and consider a training-only geometry sampler so a few high-residual cases do not consume the entire budget. Preserve random replay across all eight geometry families and Re10–100. Never train on the fixed control/audit cohorts merely because their residual is high.

Changing sampling without weights changes the expected objective. If preserving the current target sampling density q is required, use properly supported mixture p and weights q/p; raw 1/p is only correct for an appropriate uniform target. Alternatively, explicitly define the new nonuniform objective and evaluate it against unchanged physical-area FEM metrics. A low residual alone does not guarantee the right separated-flow solution.

Primary source: [PINN-sampling](https://github.com/lu-group/pinn-sampling/tree/c20433036c3eaec21758384e1710e15024745d95), [RAD Burgers implementation](https://github.com/lu-group/pinn-sampling/blob/c20433036c3eaec21758384e1710e15024745d95/src/burgers/RAD.py). Paper: *A comprehensive study of non-adaptive and residual-based adaptive sampling for physics-informed neural networks*, CMAME403 (2023), 115671, DOI 10.1016/j.cma.2022.115671.

## 2. SOAP: optimization beyond LR

The JAX-PI pirate branch implements an optimizer factory with Adam and SOAP (among others), and links *Gradient Alignment in Physics-informed Neural Networks: A Second-Order Optimization Perspective* (2025). The mechanism of interest is preconditioning/alignment, not a larger scalar LR. This is a promising optimizer experiment for stiff higher-order derivative losses.

Transfer only the same checkpoint parameters; initialize SOAP's own statistics. Maintain the same geometry/point stream and fixed validation sets to isolate the optimizer, while allowing a small, equally budgeted LR calibration for each optimizer. Compare equal wall time as well as equal updates, including preconditioner overhead. Do not combine SOAP, adaptive sampling and a new architecture in the first experiment.

JAX-PI imports an **unofficial** JAX SOAP implementation. Its README documents a 2026 correction to stale EMA updates and warns that trajectories change. Pin and inspect the version, test tree structure and checkpoint restore, and avoid silently following latest packages. This source issue is not evidence that our current Adam implementation is broken.

Primary sources: [JAX-PI optimizer factory](https://github.com/PredictiveIntelligenceLab/jaxpi/blob/9b5196b846285c32a8f2e337982d19699de85956/jaxpi/models.py), [SOAP_JAX](https://github.com/haydn-jones/SOAP_JAX), [official original SOAP implementation](https://github.com/nikhilvyas/SOAP). A full L-BFGS retry is lower priority because our previous fixed-bank L-BFGS trial worsened the independent probe/FEM metrics; that does not establish that all second-order methods will fail.

## 3. Gradient balancing: diagnose before reweighting

Wang, Teng and Perdikaris (2021) directly study unbalanced back-propagated gradients and provide a lid-driven cavity example. JAX-PI implements `grad_norm` and `ntk` loss weighting. Measure gradient norms and directional alignment of x-momentum, y-momentum and outlet terms on a fixed diagnostic batch, periodically rather than on every large batch.

Our earlier logs showed momentum contributing around 99.94% of the scalar objective; this **does not prove** its gradient dominates. Raising the outlet weight solely from this ratio is unjustified. Hard-enforced wall/inlet/divergence constraints should not be given adaptive weights that try to balance an identically zero loss. Clamp/smooth estimated weights and keep validation objective definitions fixed so a smaller newly weighted loss is not mistaken for improved flow accuracy.

Primary sources: [GradientPathologiesPINNs](https://github.com/PredictiveIntelligenceLab/GradientPathologiesPINNs/tree/93e752b0e3b541818d5cca49b681f4957bc36808), [JAX-PI grad-norm/NTK implementation](https://github.com/PredictiveIntelligenceLab/jaxpi/blob/9b5196b846285c32a8f2e337982d19699de85956/jaxpi/models.py).

## 4. PirateNet / Fourier features: credible architecture branch, not an automatic upgrade

The official JAX-PI architecture source implements PirateNet using embedded inputs, gated u/v features and adaptive residual blocks, along with Fourier embeddings and random weight factorization. These are designed to help train continuous representations and PINNs. Our existing network already has gated u/v modified-MLP components, so simply adding the name “improved MLP” would not add a missing mechanism.

A new DeepONet may use a PirateNet-style trunk (and potentially branch) while retaining wall sensors, Re conditioning and the branch/trunk operator construction. A stand-alone spatial PirateNet for one channel would lose the operator goal. Fourier scales need tuning: third derivatives amplify high frequencies; blindly copying a frequency scale from a different PDE can worsen stiffness.

Changing trunk architecture, sensor count or latent rank generally changes the parameter tree. Preserve the current model as a baseline; train a separate model, distill it, or design a function-preserving residual adapter in streamfunction/pressure space with smooth hard-constraint masks. If using an initially zero correction, ensure its parameterization still has a gradient path; setting both factors of a new branch–trunk product to zero can create a dead path. Such an adapter needs mathematical/boundary/checkpoint tests before training.

Primary source: [PirateNet and embeddings](https://github.com/PredictiveIntelligenceLab/jaxpi/blob/9b5196b846285c32a8f2e337982d19699de85956/jaxpi/archs.py). Paper: Wang et al., *PirateNets: Physics-informed Deep Learning with Residual Adaptive Networks*, arXiv2402.00326 (2024). General methods: *An Expert's Guide to Training Physics-informed Neural Networks*, arXiv2308.08468.

## Precision and less relevant suggestions

JAX-PI's README explicitly recommends highest matrix precision for reproducibility. Check `jax_default_matmul_precision` and test `highest` on a separate branch. Enabling x64 support does not automatically convert this project's float32 training arrays to float64. This check may affect speed, derivatives or convergence; no present precision fault has been established.

Causal time weighting is not a direct priority for a **steady** channel PDE. A Reynolds curriculum could be tested with replay of the full Re range; training only low Re late in the run risks forgetting high-Re behavior. Domain decomposition / mixtures of experts may help localized features, but smooth interfaces, conservation and operator coverage introduce extra work. These are later options, not immediate fixes supported by our current diagnostics.

## Recommended experimental order

Keep the already authorized adaptive-LR run as its own experiment. After a verified snapshot, inspect precision and per-loss gradient statistics, then branch matched short tests for baseline sampling versus RAD. Test SOAP separately from identical starting weights. Track fixed control/audit residuals, unchanged eight-case FEM pilot, the 20 saved cases, and the two-step recirculation case with streamline/reversal diagnostics. Use matched wall-time budgets and retain regressions; do not select on only minimum training loss. Move to a new PirateNet/Fourier architecture only if these experiments and localized residual/FEM maps suggest a representation bottleneck.

**No new optimization, sampling or architecture experiment was launched by this review.** Published improvements on other PDEs do not establish a particular multiplicative improvement on this model.
