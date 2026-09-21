# TO v3 — the ninth `fig:raw-plans` panel, and a Table 6.2 cell that is short a geometry

> # ✅ UPDATE, same day — THE PANEL LANDED. Section 1 below is superseded.
>
> Section 1 told you to keep the `\todofigure`. **Don't** — `fig_raw_plans_diffusion_K20` is in the
> store and verified. The run named in the request really is truncated, but a **complete sibling
> campaign** had the same dashboard: `H8_K20_Dmodels.GaussianDiffusion_aw10_thres0.5`, seed 6,
> `both-hard`. Same checkpoint, same $\nfe=20$, same scene and seed. `thres0.5` is a *projection*
> threshold and this is the *unprojected* arm, so the projector never ran and cannot have shaped the
> plans drawn. Replace the `\todofigure` with `\includegraphics{fig_raw_plans_diffusion_K20}`.
>
> **Sections 2–4 are unchanged and still matter** — the truncation still leaves one published table
> cell averaged over two geometries, and that is not fixed by the figure landing.


**2026-09-20 · from the DA side (figure pipeline), not from another draft.**
Nothing in `v3/` has been touched. Two items below: one changes what you should expect, one changes a
number you have already published.

## 1 · ~~`fig:raw-plans`, diffusion at $\nfe=20$ — keep the `\todofigure`~~ (SUPERSEDED — see the box above)

The ADDITION of 2026-09-20 to `REQUEST_20260916_cluster_fm_plan_panels.md` specifies this panel as a
**fetch**: the twenty-episode diffusion campaign was supposed to have written the same dashboard the
other eight panels come from. It did not. The cluster PLAN resolved 20 of its 22 cells and failed on
exactly this one, and the folder says why:

```
…/H8_K20_T0.5_Dmodels.GaussianDiffusion_msg20trials/6/results/halfspace_both-hard/
```

holds `dpcc-r`, `dpcc-r-tightened`, `dpcc-c`, `dpcc-c-tightened`, `dpcc-t` — and then stops.
`eval_dpcc-t-tightened.log` is 109 bytes with no artefacts beside it, and that timestamp
(2026-08-18 12:32) is also the newest mtime of the **whole run folder**. The job ended mid-variant. A
complete leaf holds thirteen variants; this one holds five. The `diffuser` dashboard the panel needs is
among the eight that were never reached.

~~So: keep the `\todofigure`.~~ **Resolved the same day** — see the box at the top. The dashboard
exists in the `thres0.5` campaign, which is complete across all three geometries and five seeds.

The constraint that made this work still holds, and is worth stating because it ruled out the
easier substitutions: a `diffuser` dashboard for this model also exists at `top-left-hard` and
`top-right-hard`, and **those must not be used**. The other eight panels are all seed 6 / `both-hard`,
the caption promises "same scene and seed", and a panel from a different geometry would show the reader
a different obstacle layout. The `thres0.5` source is acceptable precisely because it changes neither.

**One thing I had to fix that the request got wrong.** The 2026-09-20 addition specifies the
twenty-episode crop box with the `(403,400)` resize. The file that exists is $3000\times1000$ — the
*two*-episode layout — so it takes the 08-19 box and **no** resize. Measured, not assumed; verified
after cropping that all nine panels are `(403,400)` with the plot frame at $(56,28)$–$(388,378)$.

## 2 · Table 6.2, diffusion $\nfe=20$, per-step tightened — one cell is short a geometry

This is the part that affects text already written.

Every published cell of `tab:avoiding-dpcc-protocol` and `tab:state-headline` is five training seeds ×
three constraint geometries. `avoiding_table_spread.py` prints the realised counts. Of the 48 cells it
prints, **exactly one** is not `geos=3`:

```
extended (5 seeds x 20 episodes)
  Diffusion  K=20  t  S&C=0.960 +/- 0.045  steps=82.8 +/- 14.7  ms=564.3 +/- 29.5  seeds=5 geos=2
```

`dpcc-t-tightened` is precisely the variant the job died on, so **`both-hard` contributes nothing** to
that row. The $r$ and $c$ rows of the same cell are fine at `geos=3` — the job got through
`dpcc-c-tightened` nineteen minutes before it stopped.

**What this is not.** It is not a flattered baseline. On `n_success_and_constraints`, `both-hard` is
not the harder geometry — every model scores 1.000 there under the tightened-$c$ rule. The number is
not known to be wrong; it rests on two thirds of the evidence every neighbouring number rests on.

**What to do now** is a judgement call that is yours, and both options are defensible:

- leave the number and add a `\provisional` or a footnote saying that cell averages two geometries; or
- leave it silent until the re-run lands, since one re-run repairs it.

What should **not** happen is the row being described, in the table caption or in §6.1, as five seeds
and three geometries — that sentence is true of the other 47 cells and not of this one.

## 3 · One re-run still closes the other two

Diffusion $\nfe=20$, twenty-episode protocol, seed 6, geometry `both-hard`, the eight variants from
`dpcc-t-tightened` onward. That single evaluation cell:

1. ~~produces the `diffuser` dashboard → the ninth `fig:raw-plans` panel~~ — **done, see above**;
2. produces `dpcc-t-tightened.npz` → the diffusion row of the new D3IL-avoiding executed-path figure;
3. repairs the `geos=2` cell in Table 6.2.

No training is involved — the K20 checkpoint is the one every Chapter 6 number already uses.

## 4 · For the record, the rest of the 20-trials coverage

Checked while bounding the above. Seeds present per geometry:

| run | both-hard | top-left | top-right |
| :-- | --: | --: | --: |
| Diffusion $\nfe=20$ | **1** (partial) | 5 | 5 |
| FM $\nfe=20$ | **3** | 5 | 5 |
| MeanFM $\nfe=20$ | **0** | 3 | 5 |
| every K1 / K2 / K5 / K10 run | 5 | 5 | 5 |

The CI-MeanFM `msgafon02_s6` runs read 1/1/1 by design — the `_s6` in the tag *is* seed 6, and seeds
7–10 are already tracked as **R25**. Not a new gap.

FM $\nfe=20$ **is** a table row, and MeanFM $\nfe=20$ feeds `fig:k-ladder`. Their seed-means are
computed over the geometries each seed covers, so the values stand — but neither should be described as
five seeds across three geometries either.

## 5 · Unrelated and still true

The other 20 cells of the fetch resolved: the D3IL-avoiding executed paths are on the cluster and
staging. That figure has no builder yet, so it is not a `\hole` you can fill this round.

Recorded in
[`LEDGER_20260918_v3_figure_artefact_fetch.md`](../../../../../Data_Analysis/analysis_results_checkpoint/LEDGER_20260918_v3_figure_artefact_fetch.md),
2026-09-20 section, and as §8 of
[`PENDING_20260922_all_lacking_runs.md`](../../data_status/PENDING_20260922_all_lacking_runs.md).
