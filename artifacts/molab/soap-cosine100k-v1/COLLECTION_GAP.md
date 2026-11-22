# Archive check on2026-11-22

**Follow-up after publication:** authenticated access succeeded. The current instance has no original run directories, checkpoint422400 or raw training/probe/FEM files. The notebook's retained HTML/session cache and original full final plot were recovered and SHA256 verified. [Recovery analysis and preserved outputs](../../../reports/molab_soap_cosine100k/retained/README.md). The network blocker below describes the earlier check; it is now resolved. Final model artifacts remain unavailable on the inspected instance.

This archive is **partial**. The user's Molab screenshot shows100000 new updates completed (target422400), but the live kernel and final files have not yet been inspected. The new Molab hostname was denied by this Codex environment's network policy before authentication. The required hostname was added to the configuration draft; the tool reports that publication is required before it is applied.

Files already received locally were checked and preserved:

- Five immutable, full-SOAP-state checkpoints at322500,327400,332400,337400,342400. Every size/SHA256 and checkpoint step/state flag matched its ready manifest.
-2501 finite training-loss records through347400.
- Fixed control/audit records through346400.
- Last received status reports347400; this is a historical observation from2026-11-21, not current live status.

The final422400 checkpoint, later snapshots, final FEM results and completion metadata are **not present in this archive**. Do not infer that they are lost on Molab; access is currently blocked and they may still be available there. Reconnect to the user-provided new hostname after the environment grant is applied, then collect and verify all remaining files before claiming a complete backup. No new training job was started by this archive check.
