# Smooth-warmup/cosine continuation archive

Full-state392400→442400,5000-update smooth warm-up1e-4→5e-4 followed by45000-update cosine decay to8e-5. [Schedule and preservation](../../../reports/molab_soap_warmcos50k/README.md).

Checkpoints retain parameters, complete SOAP state and RNG. FEM labels are loaded only by the separate read-only evaluator; no feedback to training. The collector verifies size/SHA256 before pushing; collection depends on the active Codex process and auth. Download completed weights from Molab before stopping its instance. Cached plots alone do not contain model weights.
