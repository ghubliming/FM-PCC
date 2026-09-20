# v3 → v2 · 2026-09-20 · `04_method.tex:1437` claims a shared **action weight**. It is not shared.

**One sentence to change.** `v2/thesis_v2.tex:2128` (= `chapters/04_method.tex:1437` in both drafts) reads:

> …same plan horizon, number of diffusion steps, action weight and number of candidate plans --- on the…

The clause **"action weight"** is true on D3IL-avoiding for the diffusion baseline and for FM, and false
everywhere else.

## What is actually trained

| environment | diffusion baseline | flow-based models |
| :-- | :-- | :-- |
| D3IL-avoiding | 10 | FM 10; MeanFM / CI-MeanFM: key set to 10 but **never applied to the loss** |
| D3IL-aligning | 10 | **1** |
| UAV, all three scenes | **1** | **1** |

Sources: `config/avoiding-d3il.py:323, 1265, 1400, 1482` with the FIX-3 comments at `:885, :1014` and
`flow_matcher_v3_meanflow/models/mf_diffusion.py:49-52`; `config/aligning-d3il-visual.py:455, 539, 880`;
`config/uav_mix.py:265` with the declared deviation at `:497-503`. Verified against the checkpoint paths
in the aligning corpus of record (`_aw10` for the diffusion arms, `_aw1` for every flow arm).

## Suggested fix

Drop the two words:

> …same plan horizon, number of diffusion steps and number of candidate plans --- on the…

Everything else in that sentence holds. **v3 has removed the action weight from Chapters 5 and 6
entirely** at v3.43, on the author's instruction, and records the flaw in
[`../../data_status/FLAW_20260920_known_flaws.md`](../../data_status/FLAW_20260920_known_flaws.md).
If Ch 1–4 keeps the clause, the two halves of the thesis will disagree.

Related, already closed: the v3.14 item *"Training is not uniform across models"* (✅ v2.17) covered batch,
learning rate and EMA. This sentence is the one place the action weight survived.
