# Collected SOAP10k continuation

[Protocol and source](../../../reports/molab_soap_continue10k/). From step292400, full SOAP state resumes at LR2e-5 toward302400, unless the documented FEM regression guard stops it.

The first full-state snapshot at **292500** was downloaded, SHA256-verified, and checked for finite model/optimizer leaves. Its SOAP counter is5101, preserving5001 from the prior stage plus100 new updates. The snapshot is committed and pushed. Later snapshots are collected while this cloud process and credentials remain available.

No optimizer reset or new basis initialization occurred. Use the pinned previous-stage `restore_full` and vendor code. `optimizer_present=false` refers to absent Adam moments; `soap_state_present=true` identifies the complete SOAP state inside these pickle-free NPZs. Original source checkpoint SHA256 is recorded in manifest/preflight. No automatic model acceptance is implied.

`continuation/snapshots/` contains NPZs and ready/hash records. Status is a collected observation, not a live GitHub dashboard. Final per-case errors and comparison appear after completion or a guarded stop. The original292400 checkpoint remains separately archived in `../soap-trials-v1/soap_2e5/snapshots/`.

**Completed at302400:** all10000 additional updates finished with exit0,13 full-state snapshots were size/SHA256/Git-tracking verified, and final optimizer counter is15001. [Final analysis and next proposal](../../../reports/molab_soap_continue10k/completed/README.md).
