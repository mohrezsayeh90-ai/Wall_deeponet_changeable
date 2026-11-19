# Collected SOAP trial outputs

[Experiment protocol and source](../../../reports/molab_soap_trials/).

The first full SOAP snapshot at continuation step **287500** was downloaded, SHA256 verified, inspected for finite parameters/optimizer leaves, and committed/pushed. It includes model parameters, RNG and complete SOAP state. Later periodic snapshots arrive while the independent cloud collector and credentials remain active. Latest status is an observation, not a live dashboard.

These are experimental checkpoints. They have `soap_state_present=true` and `optimizer_present=false` because the standard `optimizer_present` flag refers to Adam. Resume using `restore_full` in the pinned worker and vendored optimizer; do not use the Adam resume function. NPZ files use no pickle. Keep the model metadata, all SOAP leaves, pinned source and hyperparameters together.

Snapshots and `.ready.json` SHA256 records are inside each branch's `snapshots/` folder. `collection_status.json` records collected hashes. The baseline at287400 remains preserved in `../adam-100k-v1/checkpoint.npz`. No trained model is automatically promoted. Both5000-update branches completed with exit0. All14 immutable full-state snapshots were rechecked for size, SHA256 and Git tracking. See the [completed analysis](../../../reports/molab_soap_trials/completed/README.md) for per-case results and limitations.
