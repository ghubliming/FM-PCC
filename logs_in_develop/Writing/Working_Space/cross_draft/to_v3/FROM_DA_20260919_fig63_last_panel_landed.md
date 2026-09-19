# FROM DA → v3 · 2026-09-19 — `fig:raw-plans` is complete. The last panel was trained.

**Supersedes** [`FROM_DA_20260919_fig63_diffusion_panel_answer.md`](FROM_DA_20260919_fig63_diffusion_panel_answer.md),
sent hours earlier, which told you to keep the cell empty. That was correct at the time and is now overtaken
by a run: the checkpoint that did not exist was trained. **Two edits are wanted in §6.1 — the eighth panel,
and one sentence of the caption.**

Full record: [`PENDING_20260919_fig63_diffusion_K2_panel.md`](../../data_status/PENDING_20260919_fig63_diffusion_K2_panel.md)
(the runbook is inside it) · figure store: `Data_Analysis/DA_in_Paper/figures/demo/fig_raw_plans_diffusion_K2.png`

## 1 · The panel exists and is in the store

`fig_raw_plans_diffusion_K2` is built, cut to the same `(2312, 92, 2715, 492)` box as the three 08-19
panels, and exported. The 4 × 2 matrix has **no empty cell left**.

| | $\nfe=1$ | $\nfe=2$ |
| :-- | :-- | :-- |
| MeanFM | ✅ | ✅ |
| FM | ✅ | ✅ |
| CI-MeanFM | ✅ | ✅ |
| **Diffusion** | ✅ | ✅ **new** |

It is the only one of the eight that was a **run** rather than a fetch: no $K{=}2$ diffusion checkpoint
existed anywhere, so one was trained (job 25965, 2 h 41 m on an A5000) and evaluated (25966), both
2026-09-19, seed 6, `both-hard`, unprojected, two episodes — the protocol of the other panels.

## 2 · The caption: the claim stays, one sentence goes

The v3.41 caption says the step count of a diffusion model is fixed when its noise schedule is discretised
at training, **so a panel at that budget would need a checkpoint trained there.** Every word of that is
still true and is the reason this took a GPU training rather than an evaluation. Only the clause that
*therefore the cell is empty* is now wrong.

Both diffusion panels are **trained at their own budget** — $K{=}1$ from the 08-19 run, $K{=}2$ from this
one — so the bottom row is internally consistent and reads the same way as the other three. The existing
$\nfe=1$ panel is untouched; nothing else in the figure moves.

Suggested shape, yours to word: keep the mechanism sentence, then say the $K{=}2$ panel required its own
training run, which is what the mechanism predicts. That turns the former gap into evidence for the
asymmetry §6.1.1.4 argues, instead of an apology for a missing cell.

## 2b · The two exact edits, and the one step only you can trigger

`chapters/06_results.tex`, the `fig:raw-plans` block:

* **line ~537** — replace
  `\todofigure[0.92\linewidth]{\scriptsize Diffusion, $\nfe=2$}`
  with `\includegraphics[width=\linewidth]{fig_raw_plans_diffusion_K2}`
* **the caption** — the sentence beginning *"The eighth cell is empty because it cannot be filled…"*.
  Its mechanism is right; only its conclusion changed. Note the caption's last clause,
  *"which is not the same model as the baseline"*, is **still true and worth keeping**: this panel is a
  separately trained $K{=}2$ checkpoint, not the $K{=}20$ baseline read at a lower budget.
* the earlier caption sentence *"the seven come from two evaluation campaigns"* is now **eight from
  three** — the new panel is its own campaign (2026-09-19, two episodes).

⚠️ **`export_to_draft.py` copies only figures the draft references**, so `fig_raw_plans_diffusion_K2.png`
is in the store but **not yet in `v3/figures/`**. Once the `\includegraphics` line exists, run
`python3 Data_Analysis/DA_in_Paper/plotting/export_to_draft.py v3` (or ask DA to) and it lands. The other
33 figures were re-exported today and are current.

## 3 · What the panel actually shows, in case it changes the sentence beside it

At $\nfe=1$ the baseline's unprojected fan is a dense scribble filling the corridor — many short,
incoherent plans. At $\nfe=2$ it is a **single coherent bundle** tracking the right-hand obstacle column
from start to goal. One extra denoising step is the difference between noise and a committed route.

Worth knowing before that reads as a success: the $K{=}2$ unprojected arm still scores **0.00** S&C
(constraints satisfied 0.00, 20.5 violating steps per episode). The bundle is coherent **and** it runs
along the wrong side of the constraint. The panel shows the generative model getting its act together,
not the constraint problem going away — which is the argument for the projector, in one picture.

## 4 · A second thing this run produced, for §6.1.2 — read the caveat

25966 ran the full 13-variant × 3-geometry sweep, so **the diffusion baseline at $\nfe=2$ now exists**
(ledger A4 said it was absent everywhere, at either protocol). Seed 6, two episodes per geometry,
aggregated by geometry:

| variant | S&C | steps | ms/step |
| :-- | --: | --: | --: |
| `dpcc-r-tightened` | 1.00 | 64.3 | 225 |
| `dpcc-c-tightened` | 1.00 | 61.3 | 226 |
| `dpcc-t-tightened` | 0.83 | 62.7 | 276 |
| `dpcc-r` / `dpcc-c` / `dpcc-t` (untightened) | 0.33 / 0.67 / 0.33 | 68.8–77.8 | 198–224 |
| `diffuser` (unprojected) | 0.00 | 59.0 | 18 |

🔴 **Seed 6 only.** Every other diffusion row in `tab:avoiding-dpcc-protocol` is five seeds, so this
**cannot be dropped into that table as a peer row**. Either it waits for seeds 7–10 (four more trainings;
cluster disk was at 43 GB free, so not now) or it is reported separately and labelled single-seed. Please
do not table it silently.

The one line it does support, if you want it in prose: at $\nfe=2$ the baseline still costs **≈ 225 ms per
step** against flow matching's 17.3 ms at $\nfe=1$. Cutting the budget from 20 to 2 does not recover the
gap, because the cost sits in the projector, not the denoiser. That is the same conclusion `fig:k-ladder`
draws, now with the baseline's own $K{=}2$ point rather than an inference from $K{=}1$ and $K{=}5$.

## 5 · One naming trap, since it cost a job

In the batch CSVs, `Dmodels.diffusion.GaussianDiffusion` (note the extra `.diffusion.`) under a
`flow_matching_v3_*` prefix is a **flow** model under its pre-26-May class name — commit `cac7cc6a`
renamed that class to `FlowMatchingODE`. The diffusion baseline is `Dmodels.GaussianDiffusion` under
`plans/diffusion/`. Reading the first as the second is what briefly suggested this panel was a cheap
evaluation. No DA cell was ever affected; it is noted here only so the draft never cites one as the other.
