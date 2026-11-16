# Completed second Adam continuation

Kaggle: `mohrezsayeh/geometry-pi-deeponet-adam-next50k/1`. Exactly 82,216 → 132,216; 50,000 additional updates; `target_completed`. Adam moments and RNG restored from the first continuation; L-BFGS weights are not used.

All 68 outputs and two source files are archived losslessly. [Inventory and per-file SHA256](inventory.json). The two `raw-output.zip.part*` files concatenate in manifest order to the original ZIP; archive parts and all downloaded files were hash-verified. Convenient checkpoint, optimizer state, training log, configuration, resume manifest, pilot plots/fields and source are also available directly. Older artifact directories remain unchanged.

Restore: `python -m scripts.restore_kaggle --artifact-dir artifacts/kaggle/adam-next50k-v1 --output runs/restored-adam-next50k-v1`.

[Twenty-case comparison and six random chains](../../../reports/adam_next50k_comparison/README.md). [Completion record](../../../reports/adam_next50k/completion.json). [CPU/GPU inference check](cpu-inference-check.json). This is a research checkpoint with no passing full benchmark; the saved eight-case development pilot is not full model acceptance.
