# Geometry-conditioned physics-informed DeepONet for channel flow

**Final model specification and implementation handoff · 14 November 2026**

**Deliverable.** Implement, train, and independently validate one geometry-conditioned physics-informed DeepONet for steady 2D incompressible Navier–Stokes. The deliverable comprises a frozen checkpoint, a physical-coordinate inference API, immutable split manifests, an independent FEM benchmark, and plots and machine-readable metrics. Use one checkpoint across all validated geometries and Reynolds numbers. Section 7 defines the proposed release criteria.

**Production target.** Train on a fixed pool of 10,000 distinct channel geometries distributed across eight wall-shape families, with Reynolds numbers spanning 10–100. Each geometry contains two independently variable smooth walls. The inlet velocity profile remains prescribed and identical across cases so the experiment isolates geometry and Re. The preceding variable-inlet DeepONet used 10,000 input profiles and motivates the starting scale of this pool. Its accuracy does not transfer automatically to the geometry problem.

**Implementation boundary.** This note defines the final model and the evidence required to call it accurate. No geometry-conditioned training or full PDE convergence was performed for this revision. The analytic map and hard constraints were checked previously on controlled examples. Short unit and integration checks are part of development. The requested outcome is a completed full-domain training run and a benchmarked checkpoint.

## 1. What the supplied work gives us

The two PDFs are versions of the same work by Sifan Wang, Hanwen Wang and Paris Perdikaris: the 2021 *Science Advances* article and arXiv:2103.10974v1. Both show how to represent an input function by sensor values, combine branch and trunk features, and differentiate the output with respect to query coordinates. For each function, the residual and boundary constraints are evaluated on a set of query points. One set of trained weights represents a family of solutions.

I inspected the linked repository at commit `77e7c747d653b34e79e43df7d88bad87de5c27d8`.

| Source | What the notebook actually does | What to reuse |
|---|---|---|
| `Eikonal/PI_DeepONet_Eikonal_airfoil.ipynb` | Flattens ordered sampled boundary coordinates into a branch vector. The trunk evaluates a scalar field at a two-dimensional point. It uses the boundary-zero loss and an Eikonal residual. | The pattern for feeding a sampled geometry into the branch and holding that geometry fixed across many queries. |
| `Eikonal/PI_DeepOnet_Eikonal_circle.ipynb` | Generates circles by radius and samples their coordinates at fixed angles. The network learns a distance field. | A controlled geometry-family generator and a geometry-only smoke test. |
| `Antiderivative/PI_DeepONet_antideriv.ipynb` | Combines branch–trunk dot products with automatic differentiation. Its actual training cell also generates antiderivative solution targets using `odeint`. | The dot-product and derivative logic. Implement a separate residual-only training objective for this project. |
| `Burger/PI_DeepONet_Burger.ipynb` | Uses separate initial, boundary and residual data paths for variable initial conditions. | Separate samplers and loss accounting for each physical constraint. |

The airfoil notebook defines the distance field on a surrounding box. Its eikonal constraint is a first-order scalar equation. For the proposed flow operator, collocation belongs inside each channel, the wall constraints move with the geometry, and pressure and both velocity components must satisfy the coupled equations. The legacy notebooks are useful worked examples. Build the new modules with the current JAX environment used by the user's flow operator.

Relevant prior work already includes PhyGeoNet on geometry-adaptive physics-informed flow, GAPI-DeepONet on geometry-adaptive physics-informed operators, and neural operators for steady Navier–Stokes on stenosed and bifurcating domains. A broad claim of being first to solve flow on varying geometries would be weak. A better question is how much **hard wall and flux enforcement, analytic geometry mapping, and wall-sensor resolution** improve zero-shot transfer in an experiment that holds training budget and test geometries fixed. Report which shape families, minimum gaps and Reynolds numbers are supported.

## 2. Declare the final geometry and flow domain

Use a fixed reference height H₀ and reference mean inlet speed U₀ for every case. Scale x and y by H₀, velocity by U₀, and pressure by ρU₀² relative to the exterior reference. In the following, all equations use dimensionless quantities. Set L = 5 and let

```text
Ω_g = {(x,y) : 0 < x < L, b(x) < y < t(x)}
g(x) = t(x) − b(x) > 0.
```

The design input is one pair of smooth functions (b,t). The production scope covers steady single-inlet/single-outlet 2D channels with fixed walls and one connected vertical section at every x. Bifurcations, internal obstacles and wall overhangs require a topology-aware extension. Arbitrary shapes within the declared family are accepted for evaluation subject to the geometry checks. Accuracy is established on the reported benchmark.

- Keep the first and last 10% of length flat, with b=0 and t=1. The interior has independent variation in centerline and gap. Pool families include asymmetric, multi-feature and compound walls.
- Production Re range: [10,100], based on H₀ and U₀. Both the geometry and Re vary in the main run. Report performance across their joint distribution.
- Starting declared envelope: gap 0.60≤g≤1.50, |c−0.5|≤0.35, max(|b′|,|t′|)≤0.7, nominal feature widths ≥0.08L. Use explicit bandwidth limits and verify actual assembled walls, because overlapping features may create steeper or narrower structures. Log curvature and test the full declared envelope. Treat tighter gaps and stronger slopes outside it as separate stress cases.
- Keep the prescribed inlet axial profile f(y)=6y(1−y), transverse speed zero, stationary no-slip walls, and the same outlet convention as the current code. Thus Q=1. State inlet and outlet conditions in every benchmark.

Suggested parameterization for data generation: c(x)=(b+t)/2 and g(x)=t−b. Set c=0.5+C(x), g=1+G(x), b=c−g/2, t=c+g/2. Generate C and G with smooth compactly supported bumps or endpoint-constrained B-splines. Use a differentiable high-order representation with continuous derivatives of order three for the strong streamfunction residual. A quintic B-spline with simple interior knots is a practical option, with explicit end-strip constraints. Use analytic bumps and smooth spline families in the production generator. One concrete family uses B(z)=exp(1−1/(1−z²)) for |z|<1 and zero outside, z=(x−x_c)/w. For a symmetric constriction use G=−aB and C=0. For a one-wall constriction use G=−aB and C=+aB/2, which moves only the lower wall upward. Use positive a for expansion with G=+aB. Enforce the specified full feature width 2w and keep the support inside the interior between the two flat strips. In JAX, define a numerically safe denominator before applying the mask. Unit-test derivatives around the compact-support joins. A piecewise-linear interpolation of the wall samples would leave slope jumps and invalidate strong second/third derivative calculations at joins.

### Sensors and queries

Place M fixed sensors x_j=jL/(M−1) on the common axial parameter. Sensing both walls may mean sampling known CAD curves. It requires no measured flow labels. Feed [c(x_j)−0.5, g(x_j)−1] and normalized log(Re) to the branch. Use M=257 as the production default: 514 wall values plus normalized log(Re), for 515 Branch features. Test M=65 and 129 as resolution ablations. Publish the final sensor grid and normalization. Keep training-family feature widths well above the sensor spacing and verify reconstruction of narrow cases. Raw wall-height samples are an equivalent representation to test.

The exact smooth walls used for physics are available separately from the branch features. Distinguish two modes:

1. **Known-wall mode.** Analytic/canonical spline walls define domain and derivatives. Branch receives their M-point samples. The physics and physical-coordinate API also have access to the full walls.
2. **Sensor-only geometry mode.** Reconstruct both walls by a named, deterministic smooth interpolant from the same sensor values. Use those reconstructed walls everywhere in inference. Compare reconstructed geometry to the original manufactured geometry first, and then compare the physical solutions. This is essential if the intended claim is that sensors alone specify geometry. Report wall and derivative error separately.

A query uses (ξ,η) in a fixed unit square and is independent of the branch sensor count. “Arbitrary query points” means evaluation of a continuous network within the supported domain. Resolution studies still need to establish accuracy on finer queries. A conventional MLP branch has a fixed input dimension. If input sensor locations or counts must vary, first resample to the canonical x_j; a masked set or graph encoder is an optional later experiment.

## 3. Map every channel to the same square

Use the analytic map and inverse

```text
x = L ξ
y = b(L ξ) + η g(L ξ)
ξ = x/L,  η = [y − b(x)]/g(x)
A = b′(x) + η g′(x),   det J = L g(x).

D_x = (1/L)∂_ξ − (A/g)∂_η
D_y = (1/g)∂_η
Δ_phys F = D_x(D_x F) + D_y(D_y F).
```

Here D_x and D_y are physical-coordinate derivatives written on the reference square. In the second application the derivatives act on the coefficients A/g and 1/g as well. This includes geometry curvature and metric terms. Coordinate autodiff must compute geometry features inside the differentiated function; precomputing them as detached arrays silently drops those terms. Geometry samples or coefficients in the branch are constant with respect to coordinates. Leave x,η and the local geometric features differentiable in the trunk and in the operators.

An implementation alternative is to wrap the reference-network prediction in the inverse map (x,y)→(ξ,η), then differentiate directly in physical coordinates. Implement one as the production path and compare it to the other in derivative tests. Either way, network outputs represent physical Cartesian components or potentials. They are not contravariant velocities on the square.

Use `L*g` and appropriate sampling weights if estimating a physical-area norm from reference samples. If training with an intentional near-wall mixture, state that training uses a biased sampling objective. For validation, compute physical-area-weighted norms with a fixed common quadrature per geometry. Surface terms on the actual walls use arclength when reporting physical integrals. Boundary normals and mesh markers follow the physical curves.

### Equations

For velocity U=(u,v) and pressure p:

```text
R_u = u D_x u + v D_y u + D_x p − Re⁻¹ Δ_phys u
R_v = u D_x v + v D_y v + D_y p − Re⁻¹ Δ_phys v
R_c = D_x u + D_y v.
```

At the vertical outlet use Re⁻¹D_x u−p=0 and Re⁻¹D_x v=0, the natural condition for the existing Laplacian weak formulation with zero prescribed outlet traction. Keep this convention in the FEM baseline. A symmetric-stress outlet would be a different operator. The stated outflow condition anchors the pressure level; do not additionally force p=0 on all outlet samples or project pressure to a zero-mean gauge for this problem. Check solved flows for substantial outlet recirculation. If needed, extend the flat outlet section consistently in all methods and rerun the affected comparison.

## 4. Architecture and exact constraints

Use separate scalar DeepONet pairs for streamfunction and pressure. Use smooth tanh subnetworks with width 192, four gated updates and rank 128 as the production default. Reuse the existing gated update structure and keep independent streamfunction and pressure pairs. Compare capacity and feature changes on validation with a fixed budget. Normalize branch inputs with training-only statistics. Use (ξ,η) with local differentiable geometry features [c−0.5,g−1,b′,t′] in the production trunk. Normalize the features consistently and backpropagate through their spatial dependence. Compare to a coordinates-only trunk on validation. Cache branch embeddings at inference for many queries on the same geometry.

```text
N_ψ(z; geometry,Re) = ⟨B_ψ,T_ψ⟩ / √r
N_p(z; geometry,Re) = ⟨B_p,T_p⟩ / √r.

Φ(η) = 3η² − 2η³
ψ = Q Φ(η) + α(ξ) 16η²(1−η)² N_ψ
p = 12Q(L−x)/Re + N_p
u = D_y ψ,   v = −D_x ψ.
```

Use α(ξ)=(1−exp(−ξ/ℓ))² with fixed positive ℓ, e.g. 0.05, as an initial inlet mask. Its value and first derivative vanish at the inlet. Use the full smooth geometry in the baseline and masks. For a flat unit-height channel and N_ψ=N_p=0, these equations give the exact Poiseuille solution and the matched traction. In a curved channel the same lift ensures boundary conditions but leaves physical momentum to training. No outlet mask is needed on ψ. Learn the remaining outlet traction residual.

Why the constraints hold:

- Incompressibility is exact for sufficiently smooth ψ because physical mixed derivatives commute.
- Both wall traces have constant ψ and zero physical gradient, giving stationary no slip. The double zeros in η are important.
- In the flat inlet strip with g=1, the lift gives u=f and v=0. The inlet correction and its first derivative vanish.
- The vertical section flux ∫u dy is ψ(x,t(x))−ψ(x,b(x))=Q for every x. Use flux as a numerical check. Strong exact constraints alone do not establish velocity-profile or pressure accuracy.

For an input inlet profile f(η) on a flat inlet of constant height g(0), define I(η)=g(0)∫₀^η f(a)da and Q=I(1). Replace the lift by QΦ(η)+d(ξ)[I(η)−QΦ(η)], with d(0)=1, d′(0)=0, and f(0)=f(1)=0. Keep the squared correction masks. Add f samples as a branch input only after the geometry-only benchmark passes. This is compatible with the full-primitive idea in the current channel model. A 2D streamfunction does not directly provide a 3D divergence-free construction.

### Loss options and diagnostic fallback

Train with mean squared momentum residuals plus λ_out times mean squared outlet traction. Initialize λ_out=10 as in the current channel model. Tune or balance the loss on development data, then freeze the configuration before final testing. Track unweighted momentum and the two traction terms. Record divergence, no slip, inlet and flux errors independently. Avoid importing the old inlet-curvature weighting when only walls vary. If testing geometry-related weights, keep unweighted validation and document the change.

The strong ψ formulation requires third derivatives of ψ and smooth geometry, so compare cost and conditioning on a reduced experiment. Two useful fallbacks are:

- Direct physical outputs (u,v,p) with hard velocity masks and a compatible inlet lift. Add a continuity term and monitor flux. Momentum then needs second derivatives. Treat this as a separate baseline because its incompressibility is approximate.
- A weak formulation on the physical domain with tests pulled back to the square. For test vector w zero on prescribed-velocity boundaries, the residual is ∫[(U·∇)U·w + Re⁻¹∇U:∇w − p∇·w]dΩ for the zero-traction outlet. Use the physical Jacobian and quadrature, several local test functions, and a separate validation of outlet traction. This reduces the order of derivatives in the momentum residual. It is an engineering option if strong autodiff is the bottleneck.

## 5. Production geometry pool and training protocol

Generate 10,000 unique accepted wall pairs before the production run, with 1,250 in each bucket. Assign a fixed bucket priority or disjoint generator definitions and log geometric descriptors independently of bucket labels. Exact flat flow is a deterministic reference check. Do not fill the near-flat bucket with repeated copies of that geometry.

| Family | Variation |
|---|---|
| Near-straight and mild | Small-amplitude independent deformations of both walls |
| Centerline-dominant | Smooth bends and S-bends with weak gap variations |
| Symmetric gap variation | Single smooth narrowing or widening along the centerline |
| One-wall deformation | Local positive or negative deformation on either wall |
| Multiple gap features | Two or more spaced narrowing and widening features |
| Independent multi-feature walls | Distinct feature counts, positions and signs on each wall |
| Smooth oscillatory walls | Band-limited mixtures under a smooth end-strip envelope |
| Mixed spline geometries | Independent multiscale centerline and gap shapes, including challenging valid variations |

Use stratified draws over amplitude, position, width and complexity. In a fixed part of each family, target the challenging edge of the declared gap or slope envelope. Screen by dense samples and spline extrema or certified bounds as appropriate. Store and deduplicate wall representations. Retain the seed, family, generator parameters, reconstruction coefficients, sensor vectors and continuous geometry diagnostics. Reynolds number is sampled independently during training, with coverage across the full range and a controlled fraction at 10 and 100. The 10,000 count is distinct **geometries**, regardless of the number of Reynolds draws or spatial coordinates.

Create separate, immutable pools of 400 development geometries and 1,000 locked in-distribution test geometries. Stratify both by family and difficulty. Group near duplicates, related parameter variants and reflections in the same split. Predeclare a separate stress pool of at least 200 geometries with larger slope, tighter gaps, unseen combinations or stronger frequencies. Report these cases separately. For independent FEM, solve all 400 development and 1,000 locked geometries at one preassigned Reynolds condition each, balanced across the Re range. Add a 10/30/60/100 Reynolds sweep on a preselected stratified subset of 80 locked geometries, reusing reference runs where conditions overlap. Resolve reference mesh uncertainty before classifying model errors. Reference failures must be visible in the report. Do not declare full coverage using a silently filtered subset.

**Main training run.** Use JAX, batched geometry and coordinate evaluation, and normalize with training data only. Start with 32 geometry–Re conditions per global step, 128 interior queries and 48 outlet samples per condition. If device memory requires smaller microbatches, accumulate gradients to preserve the global batch. Regenerate the collocation coordinates and Re draws each step from the fixed wall pool. Mix uniform, near-wall, and feature-focused interior samples. Keep a nonzero full-domain component and track unweighted physical residuals on independent coordinates. Do not mix FEM labels into the physics-only training objective. Use FEM metrics on the development set for model selection.

Use Adam with initial learning rate 1e-3 and decay toward 2e-5. Budget 100,000 updates as the initial **full-scale training run**, with checkpointing, deterministic restarts, and wall-clock and accelerator usage logging. Verify JAX autodiff in float64. Compare dtype and residual stability on the development set before fixing production precision. Start timing at the beginning of training and separately log compilation. Reuse branch embeddings per geometry, differentiate local features in the query path, and profile third-derivative memory use. A second-stage optimizer or other refinement is allowed only as part of global training, with the final configuration chosen on development data.

**Accuracy recovery.** Choose the checkpoint on stratified development evidence, including worst-family accuracy and pressure drop. If the development curves indicate inadequate shape coverage, expand the train-only pool to 20,000 or 40,000 shapes by preserving the original collection and adding genuinely different examples in underrepresented regions. If errors track sensor reconstruction, increase resolution or improve the declared interpolant. If strong autodiff or optimization stalls, compare the weak or direct-output formulation described above under the same validation protocol. Change one main factor at a time where possible. Do not freeze in an undertrained checkpoint just because the initial budget is exhausted. Open the locked test split once for the selected configuration. Any tuning after seeing it requires an untouched replacement test set for the final claim.

## 6. Build an independent FEM reference

Current research code: [inverse_fromtrajectory_to_FlowReconstruction](https://github.com/mohrezsayeh90-ai/inverse_fromtrajectory_to_FlowReconstruction), reviewed source lineage `92897a6962e537a4bea88042609be102095db037`. Read `flow_operator/model.py`, `train.py`, `api.py`, `checkpoint.py`, and `validation/fem.py`.

The current model already uses independent branch–trunk pairs for streamfunction and pressure, a hard inlet, and JAX spatial derivatives. Its boundary ansatz, API range checks, and FEM mesh currently assume a rectangle. Adapt these pieces in a new package or explicit new geometry mode. The old selected weight at step 94,000 is a fixed-domain comparison and did not pass all reported gates. Its weights are not a boundary-generalization checkpoint. Both geometry models and fixed-domain comparisons should keep that distinction in their reports.

Reuse the Taylor–Hood P2/P1 assembly, convection linearization, and zero-traction weak form as a basis. Replace the mesh construction and boundary tagging:

1. Create a rectangular reference triangulation with named bottom, top, inlet and outlet facets. Preserve facet IDs under the map. Do not reuse a rule like y=0 or y=1 to identify a curved wall.
2. Map the vertices with the geometry. For complex walls use high-order/curved meshes if supported. With straight triangles, quantify geometric approximation and refine near constrictions. Check element orientation and geometry resolution.
3. Apply no-slip on both physical walls and the exact inlet profile at inlet velocity DOFs. Use the same outlet form in FEM and the operator. Track pressure reference consistency.
4. Double mesh density in relevant regions and compare velocity, pressure drop, flow rate and wall stress. Use mesh-converged references on both ordinary and difficult geometries. Escalate failing references through smaller Reynolds continuation steps or a stronger nonlinear solver. Record unresolved cases.
5. Compare solutions on common physical locations. Some exact-wall-near points can land outside a chordal FEM approximation. Evaluate on a common valid interior or use a sufficiently accurate curved mesh and quantify the excluded boundary strip. Use explicit member checks. Evaluate wall conditions and wall stress on their declared geometry independently. Never silently extrapolate a FEM interpolator.

For the cross-section pressure-drop metric define and record the two x stations and average pressure with the appropriate physical cross-section measure. If reporting viscous wall shear, use the tangential projection of physical Cauchy stress, e.g. t̂·[Re⁻¹(∇U+∇Uᵀ)]n̂, including the wall normals and tangents. This diagnostic does not change the chosen FEM outlet weak form.

## 7. Metrics, baselines and success criteria

Define success using weights frozen before opening the locked test split. A prediction on a held-out geometry should use one branch evaluation and batched trunk queries. There should be no per-geometry gradient update or solver call in the zero-shot timing. Optional local correction may be studied and timed separately.

Report per-geometry physical-area-weighted relative L² errors for vector velocity and pressure. Report absolute RMS for v, or normalize by U₀, because the relative transverse denominator can vanish. Include pressure drop, boundary errors, momentum residual, flux drift, and near-wall velocity or wall shear where relevant. Separate interpolation from extrapolation. Show medians, 90th percentiles and supported worst cases. Supply case-level errors and IDs. Confidence intervals, if used, should resample independent geometries, rather than individual query points.

**Proposed final acceptance criteria.** Lock the chosen thresholds before the final test. For the full declared in-distribution scope, use physical-area-weighted vector-velocity relative L² with median ≤2% and P95 ≤5%, pressure-field relative L² with median ≤5% and P95 ≤10%, and relative pressure-drop error with median ≤3% and P95 ≤7%. Require every geometry family to have median vector-velocity error ≤3%. Require max relative section-flux deviation ≤0.1%. For dimensionless structural checks, evaluate max wall and inlet velocity mismatch ≤1e-5 and RMS divergence ≤1e-5 in float64 verification. As initial physics targets, use P95 per-case RMS physical momentum residual ≤0.05 and P95 per-case RMS outlet traction residual ≤0.01, with unweighted dimensionless residuals and the prescribed pressure convention. Compute momentum RMS with physical-area quadrature and traction RMS with outlet-line quadrature. Record any threshold adjustment before unlocking the test set. Report near-wall errors, worst cases, reference discretization uncertainty and any unresolved test cases. Use a preregistered absolute-error rule for any relative pressure metric with near-zero reference denominator. These thresholds are proposed goals and may be refined with documented development evidence before the final split is opened.

A final release means a frozen checkpoint, uncertainty-aware FEM comparisons and a reported result for each gate. If a gate fails, publish the measured accuracy and supported scope. A changed model requires independent final evaluation after development. Stress-set results establish how the model degrades outside the declared domain and are reported separately.

Essential comparisons:

1. **Analytic lift alone.** This is a strong flat-channel baseline. It may satisfy continuity and flux while missing pressure and the actual velocity distribution in contractions.
2. **Parameter-conditioned PINN.** Use the same map and PDE with a joint MLP of shape coefficients and coordinates. This tests whether branch–trunk factorization improves over a conventional conditioned network.
3. **Sensor branch versus coefficient branch.** The coefficient branch gives a reference ceiling for a known fixed generator. The sensor branch tests portability across profile representations.
4. **Hard streamfunction constraints versus soft continuity.** Match geometry family, data and budget. Compare cost, divergence and solution accuracy.
5. **Relevant geometry operator.** A small supervised or physics-aware reference-domain operator can serve as a separate comparison. Disclose its solution-label budget. Match the test set and training geometry coverage.

Recommended ablations: 65/129/257 wall sensors; with/without local geometry features in the trunk; fixed sampling versus targeted sampling; single wall versus independent walls. Run multiple seeds where compute permits. Use first-call JIT and steady-state latency separately, report hardware, and compare downstream price of training with total cost across many geometries. For an amortization estimate, N_break_even ≈ T_train/(T_FEM−T_infer) when the denominator is positive. Compare identical workloads and tolerances. The original DeepONet paper's speedups are evidence for its benchmarks only.

## 8. Implementation work packages and acceptance checks

Suggested repo-local package names. Match the repository's existing conventions.

```text
geometry_operator/
  geometry.py       wall functions, smooth coefficients, sensor encoding and map
  operators.py      physical gradients, momentum and traction
  model.py          branch/trunk pairs, masks and streamfunction lift
  sampling.py       immutable geometry splits and collocation
  train.py          optimizer, monitoring and checkpointing
  reference.py      boundary-tagged physical FEM
  evaluate.py       geometry-wise metrics, timings and figure-ready arrays
  api.py            frozen physical/reference queries and domain validation
configs/geometry_operator/
tests/test_geometry_operator.py
reports/geometry_operator/
```

Reuse the fixed-channel project as the implementation base. Put geometry-conditioned assets in a distinct package and version the checkpoints. Preserve fixed-domain compatibility. Keep metadata small and include the config, seeds, shape IDs, hashes and baseline settings. Save numerical arrays separately from plots. A reference API can accept `evaluate(walls, Re, xy)`; a second API can accept canonical wall samples and reconstruct a declared smooth shape. Provide helpful rejection messages for crossed walls, undersampling, extrapolated Re requests and out-of-domain query points. Map valid queries to the square and evaluate them in batches.

Unit checks before training:

- Rectangular-map recovery x=Lξ, y=η for flat walls; forward/inverse round trip; positive determinant L for the flat channel and Lg for the mapped channel.
- Chain-rule gradients, Laplacian and mixed derivatives on a curved geometry against analytic physical functions. Include a polynomial and a smooth nonpolynomial example.
- Hard wall, inlet, divergence and flux checks for random corrections. Evaluate even if the learned correction is initialized small.
- Poiseuille momentum and outlet traction for a null correction. Check pressure sign and physical scaling.
- FEM mesh tags, wall normals, boundary DOF values, open-outlet condition, and reference uncertainty.
- Sensor pack ordering, reflection/canonical orientation, normalization, reproducible draws and split leakage checks.
- A shape under-resolved at a given sensor count can pass the gap rule but lose a narrow feature. Flag that result and report profile-reconstruction error before comparing velocity.
- For plotting, preserve physical aspect where relevant. Compare operating conditions with common color limits. Deliver gallery of shapes, FEM/network fields and errors, pressure-drop scatter, sensor-density sensitivity, and an error-vs-gap panel.

**Analytic verification performed for this note.** I checked the map against directly differentiated physical fields on a curved channel, including repeated derivatives and the physical Laplacian. The proposed streamfunction lift gave divergence within the numerical tolerance of the analytic test, exact wall velocity and exact section flux in that test. I also checked the flat Poiseuille momentum balance and outlet traction. These checks verify the derivation on controlled functions. Full PDE convergence, training, and data efficiency are future milestones.

## 9. Required release bundle

- Source, unit tests, a pinned production config, and commands for geometry generation, training/resume, FEM generation, evaluation and zero-shot prediction.
- Frozen model weights plus geometry schema, normalization, geometry/Reynolds envelope, train/dev/test manifests, and per-geometry gate results.
- Reproducible example using two different unseen wall pairs with the same checkpoint. Export u, v, p and pressure drop. Include verified velocity, pressure and error plots for several distinct wall families.
- A benchmark report including Re sweep, unseen walls, worst cases, latency with and without compilation, reference mesh uncertainty, and training cost. Keep sensor-only results separate from full-wall mode when making a sample-only inference claim. In sensor-only mode compare against FEM on the reconstructed geometry and, when a generating truth exists, separately quantify the effect of geometry reconstruction against FEM on the generating geometry.

## 10. Extensions that follow from this project

1. **Trajectory inference on new geometries.** Reuse the thesis's one-way coupling and given particle model. Query velocity through the new physical-coordinate API at each integration step, with wall collision events based on physical wall clearance and particle radius. Compare full trajectories and events against the FEM-driven reference. Keep point-particle and finite-radius models distinct.
2. **Geometry or inflow estimation from sparse tracks.** Optimize smooth wall coefficients, mean flow, or an inlet profile with the forward operator frozen. Start with one unknown family at a time. Check sensitivities and identifiability, validate against known geometry and a held-out trajectory, and use sufficient spatial coverage. Different wall and inlet combinations can create nearly identical observed paths, so an inverse claim needs geometry error.
3. **Shapes beyond two graph walls.** Bifurcations, holes and multi-valued wall curves require a new representation. Use a point-cloud or signed-distance encoder with a physical-domain sampler, multi-block maps, or established geometry-aware operators. Treat geometric resolution, topology, and boundary tags as first-class inputs. For SDFs, use a sufficiently smooth wall function or appropriate soft constraints near non-smooth distance loci. Extend to unsteady flow only after defining initial conditions and time-dependent forcing.
4. **A paper-ready claim.** Quantify how accurately and quickly one residual-trained operator solves new smooth wall geometries, and show where its accuracy breaks as the wall gap decreases or the sensor spacing grows. A separate downstream result is how flow errors affect particle paths. The contribution is strongest when backed by a matched hard-constraint and sensor-resolution ablation plus independent FEM across the declared shape family.

## Sources

- Wang, Wang & Perdikaris. *Learning the solution operator of parametric partial differential equations with physics-informed DeepONets*. Science Advances (2021). https://doi.org/10.1126/sciadv.abi8605
- Uploaded preprint version and full appendix. https://arxiv.org/abs/2103.10974v1
- Inspected upstream notebooks, pinned commit. https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets/tree/77e7c747d653b34e79e43df7d88bad87de5c27d8
- Current project, fixed-channel implementation. https://github.com/mohrezsayeh90-ai/inverse_fromtrajectory_to_FlowReconstruction/tree/92897a6962e537a4bea88042609be102095db037
- Gao, Sun & Wang. *PhyGeoNet: Physics-informed geometry-adaptive convolutional neural networks for solving parameterized steady-state PDEs on irregular domain*. JCP (2021). https://doi.org/10.1016/j.jcp.2020.110079
- *A geometry-adaptive physics-informed operator framework generalized for arbitrary geometries*. Engineering Applications of Artificial Intelligence (2026). https://doi.org/10.1016/j.engappai.2026.114457
- *Learning the solution operator of two-dimensional incompressible Navier–Stokes equations using physics-aware convolutional neural networks*. JCP (2025). https://doi.org/10.1016/j.jcp.2025.114027
- *Operator learning for hemodynamics on variable domains: 2D stenosis and bifurcation*. PubMed record. https://pubmed.ncbi.nlm.nih.gov/42770673/
- Li et al. *Geometry-Informed Neural Operator for Large-Scale 3D PDEs*. NeurIPS (2023). https://arxiv.org/abs/2309.00583
