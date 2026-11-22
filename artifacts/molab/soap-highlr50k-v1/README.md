# High-LR SOAP50000-update archive

[Run configuration and FEM separation](../../../reports/molab_soap_highlr50k/README.md).

Common source342400, fresh50000 updates, cosine1e-3→1e-4 and batch32. The trainer never loads FEM labels; the independent evaluator writes only metrics under `fem/`. `fem_cohort.json` records all20 valid geometries and the12 higher-curvature subgroup. Checkpoint NPZ files retain complete SOAP state and RNG; ready manifests and collector SHA256 checks prove collected files. Runtime status is a last-observed value, not a guarantee of current connectivity.

The collector's life depends on the Codex cloud process and authenticated notebook. Download weights before closing Molab. Earlier trials and cached plots remain in their separate archives.
