# SOAP cosine100k progress archive

Continuation322400→422400, batch32 and continuous cosine LR1e-4→5e-6. [Configuration and launch checks](../../../reports/molab_soap_cosine100k/README.md).

The independent collector copies logs/configuration and immutable full SOAP checkpoints after100 and each5000 new updates. It verifies SHA256 before renaming each `.download` file and pushing to main. `collection_status.json` records its last successful collection and verified-file hashes. The collector depends on the active Codex cloud process and authenticated Molab session; restarting either requires reconnecting/restarting collection. No token is saved in these files. Existing earlier runs and rejected trials are preserved in their separate directories.
