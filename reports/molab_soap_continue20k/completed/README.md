# SOAP20k continuation completed at322400

All20000 additional updates completed from302400 without optimizer reset;35000 SOAP parameter updates total from287400. Full optimizer count35001 includes the original basis-only initialization. The original run was paused1061.45s for two batch-size diagnostics and resumed with the same live state. Both diagnostics were rejected; they were separate branches. Final SHA256: `768c56f1c08515b3c0aa24d20ed19308cdf977e43746110db6b309c380f2b3e2`.

[Every-case CSV](per_case_errors.csv) · [Data](observation.json) · [FEM plot](fem_comparison.png) · [PDF](fem_comparison.pdf) · [Raw artifacts and full-state snapshots](../../../artifacts/molab/soap-continue20k-v1/).

| Median relative L2 |302400 |322400 |
|---|---:|---:|
|20 Re30: velocity|1.3609%|1.3404%|
|20 Re30: pressure|1.1358%|1.1709%|
|8 Re10: velocity|1.1027%|1.0780%|
|8 Re10: pressure|0.7342%|0.6558%|

Fixed audit starts0.01281514 and ends0.01248427, only **2.58% lower in20000 updates**. Control starts0.01269784 and ends0.01240382 (2.32% lower). Audit is not monotonic: the smallest recorded audit was at308400, not the final step; the minimum control was at320400. This is a small improvement in a physics objective, not proof of equivalent FEM-error reduction.

The20-case velocity median improves1.51%, but pressure median regresses3.09%. Worst20-case velocity7.4896%→7.3484%, pressure13.8567%→12.7089%. The8-case worst pressure0.0163867→0.0183183 as a fraction also regresses. All individual changes are retained in the CSV. **No large, consistent plateau fix or universal model improvement has been established.** No automatic production promotion or further long continuation was launched on this evidence.

The elapsed training-stage wall time2950.8s includes the17.69-minute pause and diagnostic resource contention; it is not pure GPU training time. Same28 repeatedly used FEM development cases remain20 atRe30 and8 atRe10; no new evidence atRe60/100 and no locked-test opening. The guard failing to trigger does not mean every case improved.

[Controlled plateau investigation and decisions](../../plateau_diagnosis/README.md). The fixed-trunk FEM projection is a diagnostic only and must not be presented as actual PINN accuracy. Changing batch size, LR scaling, or freezing trunk did not establish the large reduction requested.


## Archive verified

All25 full-state snapshots were downloaded, their lengths and SHA256 digests verified, and pushed to GitHub. Final322400 contains56 model leaves and225 SOAP-state leaves, all finite, read with `allow_pickle=False`. [Final checkpoint verification](checkpoint_verification.json) · [Every snapshot verification](snapshot_verification.json) · [Download final full-state NPZ](../../../artifacts/molab/soap-continue20k-v1/continuation/snapshots/checkpoint_322400.npz). The generic `optimizer_present=false` flag refers to the absent Adam state; `soap_state_present=true` records the preserved SOAP state. Use the archived SOAP restore helper, not Adam resume.
