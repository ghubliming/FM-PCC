# NOTES — plan quality, smoothness, and why downstream metrics hide it

**Created:** 2026-09-12 · **Type:** a claim this writing must not lose, plus the measurement it owes
**Thesis home:** `sec:res:state` (the result) · `sec:disc:threats` (the general form) ·
`sec:setup:metrics` (why stage 1 of the funnel is the unprojected arm) · `sec:disc:limitations` (the gap)
**Written into:** [`../Working_Space/v3/`](../Working_Space/v3/README.md) as of **v3.5** — see that
CHANGELOG before re-adding it anywhere.
**Evidence of record:**
[`Report_20260819_MF_UNet`](../../../Data_Analysis/DA_Result_Curated_MD/Report_20260819_MF_UNet/README.md) §7 (figs 6a–6c) ·
[`Report_20260903_AF_UNet`](../../../Data_Analysis/DA_Result_Curated_MD/Report_20260903_AF_UNet/README.md) §7–§8 (fig 7)

---

## 1. The claim, in one paragraph

**Every metric this thesis reports is measured after a projector and a tracking controller have
acted on the plan, and both of those are low-pass.** A projector pulls a plan toward the feasible
set; a controller turns setpoints into commands through inverse kinematics or a cascaded geometric
law. Either absorbs a great deal of incoherence. So a generator that emits chaotic plans and one
that emits clean plans can post similar downstream numbers — and at one network evaluation, that is
exactly what happens.

Switch the projector off and the two separate:

| unprojected, $K = 1$, seed 6, both-hard, 10 episodes | violations | steps | s/step |
| :-- | --: | --: | --: |
| MeanFlow, U-Net 4.0 M | **6.0** | 58.0 | 0.0097 |
| diffusion baseline, U-Net 4.0 M | **28.0** | 56.5 | 0.0094 |
| MeanFlow, $K = 2$ | 12.0 | — | 0.0186 |

**4.7× fewer violations before any projection, at matched compute** (3 % apart) and matched backbone
size. The qualitative half is the picture: MeanFlow's replans form a tight ribbon around the executed
path, each a bounded goal-directed curve; the baseline's are a high-frequency scribble spanning the
workspace. **The baseline's *commanded* traces are smooth anyway — because the controller integrates
the scribble away.**

> ### Why this matters more than it looks
> It is not only "MeanFlow is better". It is that **the usual reporting cannot see the difference**.
> An evaluation that measures only what survives the control stack measures *the stack's tolerance*
> as much as the planner's quality. That is a threat to validity for the whole area, found in this
> project's own numbers, and it costs one extra configuration per cell to defend against.

And it is the missing half of the headline: the MeanFlow engine is not merely **cheaper** at one
evaluation — its plans are **usable** at one evaluation, where the baseline's are not. Without this,
a reader can reasonably ask whether the 31× is bought by degrading the plan.

## 2. The consistency target on the same arm

| unprojected, $K = 1$, top-right-hard, seed 6, 20 trials, **successes-only** step basis | goal reached | steps |
| :-- | --: | --: |
| consistency target, floor $\alpha = 0.2$ | **1.00** (20/20) | 60.90 |
| consistency target, floor $\alpha = 0.05$ | 0.90 (18/20) | 62.28 |
| MeanFlow | 0.85 (17/20) | 62.41 |
| plain flow matching | 0.85 (17/20) | 62.45 |
| diffusion baseline, $K = 20$ | 0.60 (12/20) | 57.25 |

🔴 **Do not turn this into a ladder.** Three episodes on one seed, and **the margin does not survive
projection** — under the projector the program repairs both plans toward the same place. It is
consistent with the standing negative result ([`Naming`](Naming/NAMING_20260910_master_table.md) §1,
`FALLBACK_20260910` §2): what the raw arm supports is *the few-step objectives draw better plans
than diffusion at one evaluation*, not an ordering among them.

⚠️ **Two `n_steps` definitions exist in the toolchain** — the eval log averages over *successful*
episodes, the DA CSV over *all* of them. This table is the successes-only basis, which is the only
correct one when success rates differ. Quoting it next to an all-episode number is a real error;
`Report_20260903` §7 has the reconciliation (116 of 318 cells mismatch, every one with success < 1).

## 3. 🔴 The measurement this owes

**No quantitative smoothness metric exists anywhere in this pipeline.** The argument currently rests
on violation counts (quantitative) plus inspection of the plan panels (not).

- **What to compute:** jerk, path length, or curvature over the saved plan files.
- **Cost: no new runs.** One pass over plans already on disk. Directories are listed at the end of
  `Report_20260903_AF_UNet` §8.
- **Why it is worth taking:** it converts the strongest qualitative claim in the thesis into a
  quantitative one, and it is the cheapest outstanding measurement on the whole list — cheaper than
  every run in `DATASTATUS_20260910` §8.
- `Report_20260903_AF_UNet` §8 additionally has **eight plan panels still marked placeholder**
  (α-Flow / MeanFlow / flow matching / diffusion at $K \in \{1,2\}$). The `Report_20260819` panels
  exist and are the ones the thesis uses.

## 4. Figure provenance — checked, and worth re-checking

The four panels used are **ours**: verified absent from `/workspaces/aux_repo/`. They are registered
in `Working_Space/v3/plots/sources.py` under `VENDORED`, each with its provenance string, and
`figures/MANIFEST.md` prints them in their own table.

> 🚨 **Why the registry exists.** `figures/avoiding*.png` at the repo root are **byte-identical to
> the baseline authors' own copies** in `aux_repo/dpcc/figures/`, and were very nearly used as this
> thesis's environment figures. Any panel that is copied rather than generated gets checked against
> `aux_repo/` first and carries its provenance afterwards. See `v3/CHANGELOG.md` v3.3.

## 5. Related standing rules

- The comparison is **plan quality, not constraint satisfaction** — all unprojected arms are
  unconstrained, so all of them violate. Never read the raw violation counts as a safety result.
- On `both-hard` the projector still recovers the baseline at $K=1$ to S&C 0.90–1.00; the larger
  collapse (0.50–0.60) is on `top-left-hard` / `top-right-hard`. The raw-plan gap and the projected
  gap are different sizes, and the text says which it is quoting.
- This is why stage 1 of the funnel ([`../Working_Space/v3/chapters/05_setup.tex`](../Working_Space/v3/chapters/05_setup.tex),
  `sec:setup:metrics`) is evaluated on the unprojected arm — a decision that predates this note and
  is justified by it.
