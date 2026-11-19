# Paired RAD trials: collected research outputs

Starting checkpoint:287400, preserved at `../adam-100k-v1/checkpoint.npz`. Two branches target302400 independently; baseline runs first. [Protocol and source](../../../reports/molab_rad_ab_15k/).

The first new full-state baseline checkpoint **287500** was downloaded, SHA256-verified, checked for Adam state, and pushed to GitHub. Snapshots and their `.ready.json` hashes are under `baseline/snapshots/` and later `rad/snapshots/`. They are experimental weights, not automatically promoted models. The collector publishes later snapshots as available while this cloud environment and authentication remain active. Runtime status is a snapshot, not a live GitHub dashboard.

`base_fem.json` re-evaluates exactly the prior20 Re30 cases plus8 Re10 pilot cases. Results match the collected287400 baseline. Training loss values under RAD and baseline use different sampling distributions; use identical fixed probes and per-case FEM for comparison.

All downloaded files are checked against remote SHA256; `collection_status.json` records hashes at its observation time. Incomplete transfers are ignored. Molab download bundle and final figures become available when both trials finish.
