# Complete molab archive: adam-100k-v1

**All 30 original ZIP members preserved**, with exact sizes and SHA256 hashes. The original remote download SHA256, all archive parts, reconstructed archive, every file and ZIP CRC were verified. [Inventory](inventory.json) · [Verification](verification.json).

The original ZIP is stored as 4 ordered parts, each at most 40 MiB, so all contents fit ordinary GitHub file limits. Restore everything with:

```bash
python -m scripts.restore_kaggle --artifact-dir artifacts/molab/adam-100k-v1 --output runs/restored-adam-100k-v1
```

The generic restore command verifies every part, the concatenated ZIP and every member before extraction, and refuses to overwrite different existing files. Earlier Kaggle/molab artifacts remain unchanged.

This includes **all ten periodic full-state snapshots**, final checkpoint at **287,400**, best fixed-probe checkpoint at **282,400**, resumed baseline at **187,400**, Adam moments/RNG, 10,001 training records, all 21 probe records, runtime/status/resume data, FEM pilot, frozen-controller/worker code, fixed probe and full console log. The immutable geometry pool and original physics source remain reproducible from the preceding Kaggle archive referenced by manifest/pool hashes.

Convenient direct files: [final checkpoint](checkpoint.npz), [best-probe checkpoint](best_checkpoint.npz), [base checkpoint](base_checkpoint.npz), [training log](training.jsonl), [fixed probe log](probe.jsonl), [FEM pilot](fem_pilot.json).

[Evaluation on 20 unchanged FEM cases and two complete result chains](../../../reports/molab_adam_100k/completed/README.md). Best by fixed-probe objective is not best for every FEM quantity. These are unaccepted research checkpoints.
