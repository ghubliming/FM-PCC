# Gen9 E2 — Where We Diverged From the Published DPCC Code (Comparability Ledger)

**Date:** 2026-06-13
**Author:** Claude Opus 4.8
**Companion to:** `U4/CHANGELOG.md` (the fix pass) and
`logs_in_develop/Gen3v2/Audit_After_Gen8/Gen3v2_FMv3ODE_DPCC_CrossCheck_vs_Fable.md`
(where these were traced back to the published DPCC repo at `/workspaces/dpcc`).

---

## Why this document exists

The Fable audit (`U3/U3_audit_Fable.md`) listed bugs B1–B10 and U4 fixed the code-fixable
ones. The cross-check against `/workspaces/dpcc` then revealed something not obvious at the
time: **four of those "bugs" (B6, B7, B8, B9) are not ours — they exist verbatim in the
published DPCC code.** So when U4 fixed them, it **deviated from the published baseline**.

That is a double-edged action. A fix can be an improvement *and* a loss of comparability with
the paper at the same time. This ledger records exactly which fixes diverged from DPCC, so the
divergence is never silently forgotten when we later put numbers next to the DPCC paper.

---

## The key idea: there are TWO different "comparabilities," and they are not the same

| Comparison axis | What it means | Affected by our fixes? |
|---|---|---|
| **A. Within Gen9: visual-FM vs visual-DPCC** | The actual thesis question — does FM beat/match DPCC under identical infra? | **No — preserved.** Both forks got the *same* B6/B7/B8/B9 fixes, so any change is common-mode and cancels. |
| **B. Gen9 visual-DPCC vs the DPCC *paper* table** | "Do our numbers reproduce the published DPCC numbers?" | **Was never valid anyway** (visual ≠ state task) *and* now further diverged by B6/B7. |
| **C. State baseline (`diffuser/`) vs DPCC paper** | The faithful reproduction of the published state-only avoiding result | **No — fully intact.** U4 touched only the visual fork; `diffuser/` is unchanged. |

The worry "if we fixed DPCC's bugs we lost comparability" resolves cleanly once you separate
these:

- **Axis A (the one that matters for the thesis) is safe** — common-mode fixes.
- **Axis C (faithful DPCC reproduction) is safe** — we never touched the state baseline.
- **Axis B is the only thing affected, and it was already not a real comparison** — Gen9 is
  *single-camera visual* avoiding; the DPCC paper is *state-only* avoiding. You cannot lay a
  vision-conditioned success rate beside a state-conditioned one as a "reproduction" claim
  regardless of EMA or split choices. B1/B2 alone (the model literally sees images) already
  make it a different experiment.

---

## What DPCC actually does vs what Gen9 E2 now does (the divergences)

These are the **inherited-from-DPCC** findings. For each: the published DPCC behaviour, the
Gen9 E2 U4 change, and whether it moves us away from the paper.

| # | Published DPCC (`/workspaces/dpcc`) | Gen9 E2 U4 now does | Diverges from DPCC? | When it bites |
|---|---|---|---|---|
| **B6** EMA | Evals **raw** `trainer.model` (`serialization.py:75`; `diffusion_epoch:'best'` → raw weights) | Visual eval returns `trainer.ema_model` (`eval_visual_avoiding_dpcc.py:179-180`) | **YES — eval-side, active now** | Immediately, on existing checkpoints |
| **B7** split | Window-level `random_split` over stride-1 windows; `'best'` chosen on leaky test loss | `episode_split()` — hard episode boundary (`sequence.py:144`, `training.py:74-75`) | **YES — training-side** | Next retrain only |
| **B8** final save | No terminal save; last checkpoint at step 80k of 100k | `self.save(self.step)` at end of `train()` (`training.py:203`) | **YES, but largely inert** (DPCC evals `'best'`, not `'latest'`) | Next retrain, only if evaluating `'latest'` |
| **B9** eval-mode | `test()` leaves model in `.eval()` | `self.model.train()` restores mode (`training.py:223`) | **YES technically, zero practical effect** (no mode-dependent layers) | Next retrain; effectively never |

### The two that genuinely change numbers

- **B6 (EMA)** is the sharpest divergence: it is **eval-side and active right now** on existing
  checkpoints. DPCC's published numbers are on raw weights; Gen9 visual numbers are now on
  EMA-smoothed weights. EMA usually *helps*, so Gen9 visual-DPCC will tend to look a little
  better than a raw-weights run of the same checkpoint — **not** because the model is better,
  but because of the eval choice. Since visual-FM also uses EMA now, Axis A stays fair.
- **B7 (episode split)** changes *which* checkpoint `'best'` selects and removes the train/test
  leakage. It only matters on the **next retrain**. Again common-mode across FM and DPCC.

### The two that are bookkeeping-only

- **B8** is inert for any run evaluated with `diffusion_epoch:'best'` (which is the default),
  because `'best'` is saved independently throughout training. It only matters if someone
  evaluates `'latest'`.
- **B9** has no effect on the produced weights given the GroupNorm + manual-CFG-mask
  architecture (no `BatchNorm`/`nn.Dropout` module responds to `.eval()`). It is insurance,
  not a numbers change.

---

## Fixes that did NOT diverge from DPCC (safe — visual-only or no-op)

These U4 fixes do **not** create any comparability gap with the published DPCC, because the
issues do not exist in DPCC at all (they are artifacts of the visual rewrite) or change no
behaviour:

| # | Why it's safe |
|---|---|
| **B1** RGB/BGR | DPCC has no images — pure visual-rewrite bug. Fixing it only undoes a visual regression. |
| **B2** render resolution | Same — no images in DPCC. |
| **B3** `trajectory_selection` | DPCC implements selection correctly (`scripts/eval.py:209-215`); the visual code had dead code. U4 removed the *dead* code — behaviour unchanged. (Note: visual still does not implement selection, so visual `dpcc-c/-t` variants remain RNG-only — a *pre-existing* visual gap, not introduced or closed by U4.) |
| **B5** constraint source | No code change. |
| **B10** pkl banner / `mpc_batch_size` | The banner is a warning (no behaviour change); `mpc_batch_size` wiring makes the config value authoritative — applies identically to FM and DPCC, common-mode. |

---

## The reassurance, stated plainly

1. **The state-based DPCC baseline is untouched and still paper-faithful.** Verified: `diffuser/`
   has **no** `episode_split`, **no** terminal `self.save(self.step)`, **no** `model.train()`
   restore, and `serialization.py` is byte-identical to `/workspaces/dpcc`. If you want to quote
   a number that reproduces the DPCC paper, run the **state** baseline — it is exactly the
   published code (plus our resume/logging additions, which do not touch the bug lines).

2. **The FM-vs-DPCC comparison inside Gen9 is still fair.** Both visual forks received the
   identical B6/B7/B8/B9 changes, so the comparison is apples-to-apples; the fixes cancel in
   the difference.

3. **The only thing we "lost" was a comparison that was never valid** — visual (image-conditioned)
   numbers were never a reproduction of the state-only DPCC paper table.

---

## What to do to keep everything clean

- **When reporting Gen9 visual numbers:** label them "EMA-eval, episode-split" — do **not**
  present them as reproducing the DPCC paper. Present them only against the **Gen9 visual-DPCC**
  baseline (same infra).
- **If you ever want a head-to-head against the DPCC paper:** use the untouched **state**
  `diffuser/` baseline, which still matches `/workspaces/dpcc`.
- **If you want to isolate the EMA effect:** run the visual-DPCC checkpoint once with raw
  weights and once with EMA (a labelled `dpcc-raw` vs `dpcc-ema` pair) rather than assuming the
  gain is model quality. This is the cheapest way to quantify how much B6 alone moved the number.
- **Do not back-port B6/B7/B8/B9 into the state `diffuser/` baseline** unless you intend to stop
  reproducing the paper — and if you do, keep an unmodified copy as the reference column.

---

## Caveats

- File:line read on `2026-06-13`, branch `update_into_FM`; re-verify if files move.
- Nothing was executed (Docker = AI-coding only). The EMA "usually helps" and B7 "mild"
  statements are reasoned, not measured here — confirm with the labelled A/B runs above.
- "DPCC paper table" claims assume `/workspaces/dpcc` is the published code/commit; it may
  differ from what produced the actual paper figures (see the Gen3v2 audit's repo≠paper caveat).
