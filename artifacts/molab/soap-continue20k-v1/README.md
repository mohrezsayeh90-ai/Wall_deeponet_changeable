# SOAP20k continuation outputs

[Protocol/source](../../../reports/molab_soap_continue20k/). Resume the complete SOAP state302400 at LR2e-5 toward322400, with the retained FEM regression guard. No expanded32-case FEM cohort is launched in this stage, per the user request.

Production preflight verified exact parameter/optimizer-state restore, matching next sampled batch and next update, with optimizer count15001→15002 and no basis initialization. Periodic checkpoints include all SOAP leaves and RNG; use the pinned earlier-stage `restore_full`, not Adam resume. `soap_state_present=true` distinguishes SOAP from the standard Adam flag.

Immutable NPZs and SHA256 ready records are under `continuation/snapshots/`. `collection_status.json` records verified downloads; a successful Git push proves an off-molab backup. This independent collector requires the Codex cloud environment and credentials to remain available. Incomplete transfers are ignored. Runtime status is a collected observation, not a live GitHub dashboard.

Final metrics/figures will compare with the starting302400 SOAP state and original287400. Adam15k is only a shorter-budget reference; no matched35000-step Adam run was performed. No automatic model acceptance is implied. Earlier source checkpoints stay preserved in the prior-stage archives.

**First new full snapshot302500 was SHA256-verified and pushed**, including finite model/SOAP state and optimizer counter15101. This preserves the prior15001 count plus100 new updates.
