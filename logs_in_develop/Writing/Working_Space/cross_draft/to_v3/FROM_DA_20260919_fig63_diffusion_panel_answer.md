# FROM DA → v3 · 2026-09-19 — Fig 6.3's empty panel: your caption was right, keep it

**No draft change is needed.** This answers the question that opened
[`PENDING_20260919_fig63_diffusion_K2_panel.md`](../../data_status/PENDING_20260919_fig63_diffusion_K2_panel.md)
(ledger R24) and closes the correction that note opened against your caption.

## The short version

The note claimed the caption was too strong — that a second route existed where the diffusion engine is
sampled at any $K$ off one checkpoint, making the empty cell an evaluation away. **That claim was wrong,
and the run that tested it proved it wrong in four seconds** (job 25964).

The folder it rested on, `plans/flow_matching_v3_ode_selectable/H8_Dmodels.diffusion.GaussianDiffusion_…`,
is not a diffusion run. It is a **flow-matching** run from before commit `cac7cc6a` (25 May 2026), which
renamed the class `GaussianDiffusion` → `FlowMatchingODE` in that module. The path kept the old name; the
model was always the flow model. Loading it as a diffusion engine fails on the class that no longer exists.

## What this means for Chapter 6

* **`fig:raw-plans` keeps its empty cell, and the v3.41 caption stands word for word.** A diffusion
  model's step count is fixed when its noise schedule is discretised at training, so that panel needs a
  checkpoint trained at $\nfe=2$. There is no cheaper route.
* **The existing diffusion $\nfe=1$ panel stays.** It was going to be replaced so the row would not mix a
  trained-at-1 model with a sampled-at-2 one; with no sampled-at-2 model, nothing changes.
* Filling the cell now means a **training run** — author's call, not a fetch and not a quick eval.
* `fig:k-ladder` is untouched by this. Its §6 correction (three separately trained checkpoints) was
  independent and is confirmed.
* The claim the chapter builds on this — for diffusion the budget is a training decision a later choice
  cannot undo, while for a flow model it is an inference-time dial — is **strengthened**, not weakened.
  The one apparent counter-example turned out to be a flow model all along.

## One trap worth knowing, since it fooled the audit

In the batch CSVs, `Dmodels.diffusion.GaussianDiffusion` (with the extra `.diffusion.`) under a
`flow_matching_v3_*` prefix is a **flow** row; the diffusion baseline is `Dmodels.GaussianDiffusion` under
`plans/diffusion/`. Every DA cell was re-checked against this and none is mislabelled, so no table or
figure reads a diffusion budget as an inference-time choice.
