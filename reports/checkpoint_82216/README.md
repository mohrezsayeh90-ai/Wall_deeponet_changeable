# Updated predictions for all 25 existing cases

**New checkpoint: step 82,216. Original FEM reused exactly. No FEM solve or training was run. Original step-32,216 reports remain unchanged.**

| Set | Cases | New results | PDF | Complete ZIP |
| --- | ---: | --- | --- | --- |
| channel_chain_15 | 15 | [Gallery](channel_chain_15/README.md) | [PDF](channel_chain_15/all_chains.pdf) | [Download ZIP](downloads/channel_chain_15_step_82216.zip) |
| random_channel_chain_5 | 5 | [Gallery](random_channel_chain_5/README.md) | [PDF](random_channel_chain_5/all_chains.pdf) | [Download ZIP](downloads/random_channel_chain_5_step_82216.zip) |
| extreme_channel_chain_5 | 5 | [Gallery](extreme_channel_chain_5/README.md) | [PDF](extreme_channel_chain_5/all_chains.pdf) | [Download ZIP](downloads/extreme_channel_chain_5_step_82216.zip) |

Each ZIP contains all new model/FEM/error figures, complete per-case chain images, numerical predictions, unchanged FEM references and geometry, old/new comparison metrics, MAE/relative L1 tables and the multipage PDF. Unzip and open the included README/PDF; no model execution is needed. Archive SHA-256 values are recorded in [index.json](index.json).

All 25 cases are Re=30. The extreme set is explicitly outside the training geometry envelope and is a separate extrapolation experiment. Results are not uniformly improved; the paired tables retain every case and metric. The new checkpoint remains unaccepted.

Reproduce locally: `python -m scripts.refresh_saved_cases` in the documented CPU environment. The script disables FEM solver entry points during the refresh, verifies identical references/points/weights and labels figures with the actual checkpoint step.
