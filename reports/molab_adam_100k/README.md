# Selected cosine Adam continuation: 187,400 → 287,400

**100,000 additional updates authorized and launched on molab.** First resumed update **187,401** verified; at the recorded observation, training had reached **188,810** and checkpoint **188,800** was saved with no dashboard errors. [Launch observation](launch-observation.json) · [Exact manifest](manifest.json) · [Results used for selection](../molab_lr_trials/README.md) · [Worker/controller/dashboard source](../../notebooks/molab_adam_100k/).

Source: cosine 15k trial endpoint. Base SHA256 `7b53db720ac5e3644739704da16f77baa8828012792ccc433528ebcb5ca7bebd`. Full parameters, Adam moments and RNG are restored; source checkpoint was checked finite and optimizer-present. Model, pool, normalization, physics, batch32, interior128, outlet48 and float32 are preserved. Only the learning-rate controller changes relative to the frozen trainer.

The new cosine schedule is `2e-5 + 0.5*(1e-4-2e-5)*(1+cos(pi*t))`, with `t=(step-187401)/99999` clipped to [0,1]. The 100k values were JIT-tested for finite positive rates, correct endpoints and monotonic decay. This is an extension of the winning short-trial schedule; its longer horizon is not yet validated.

Output root on molab: `/home/marimo/wall_adam_100k`. Active output: `cases/cosine_100k`. Latest full-state checkpoint every 100 updates, best checkpoint by fixed dev-probe loss (including starting weights), fixed probe every 5k, additional full-state copies every 10k. A matched eight-case FEM pilot is computed after completion. An automatic final ZIP is exposed in the notebook after creation and CRC verification.

The dashboard preserves the prior run/trial curves and adds the new continuation, step progress, loss and fixed-probe checks. The latest GitHub collected checkpoint is not automatically replaced by an uncollected remote run. All earlier weights are retained.

Expected duration is approximately **two hours**, depending on evaluation and storage overhead. The step target governs this run; a 4h internal safety budget and outer process watchdog are fallback limits. This report records launch, not completion or guaranteed accuracy improvement.
