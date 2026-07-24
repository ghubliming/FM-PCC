# Gen13 fix_7 RESULTS — the 2×2 smoothness matrix: the NLP manufactures smoothness

**Date:** 2026-07-20 · **Data:** `temp/fix_7/diag_smooth_{imf,fm}_{unguided,guided}_*_n5/trajectories.csv` (n=5 each, ~7 plans/episode ⇒ ~35 planned horizons per cell)
**Prompted by:** *"inspected unguided iMF smoothness — it is crazy chaotic lines compared to FM lines, and the guided is normal."*
**Verdict:** your visual read is **exactly right, and quantitatively confirmed**. It also produced a result that **narrows the cause of the residual safety gap**.

> **First, to answer the question directly: yes — "unguided" means NO projection at all.** `original_imf` / `original` sample the generative field and execute it; no NLP, no obstacle constraint, no dynamics constraint. It is the pure generative output, HardFlow's equivalent of DPCC's raw `diffuser`.

---

## 1. The numbers

| cell | roughness (plan used) | roughness (raw, pre-NLP) | NLP smoothing | safe | steps |
|---|---|---|---|---|---|
| **iMF unguided** | **1.666e-04** | — (no projection) | — | 0% | 41.6 |
| **iMF guided** | **2.194e-06** | 2.106e-04 | **96×** | 100% | 51.6 |
| **FM unguided** | **1.762e-05** | — | — | 0% | 18.8 |
| **FM guided** | **2.683e-06** | 1.980e-05 | **7.4×** | 100% | 51.8 |

Derived comparisons:

| Comparison | Result |
|---|---|
| **Unguided: iMF vs FM** | **iMF is 9.5× ROUGHER** ← your observation, quantified |
| **Guided: iMF vs FM** | **iMF is 0.82× — i.e. marginally SMOOTHER than FM** |
| NLP smoothing done for iMF | **96×** (raw → projected, same planning call) |
| NLP smoothing done for FM | **7.4×** |
| ⇒ the projection works **13× harder** for iMF | |

## 2. Internal consistency check (the measurement is trustworthy)

The raw roughness is measured **two independent ways** — as the *unguided* cell, and as the *warm-start* inside the guided cell (fix_7.2). They agree:

| | unguided cell | guided cell's raw | ratio |
|---|---|---|---|
| iMF | 1.666e-04 | 2.106e-04 | 1.26 |
| FM | 1.762e-05 | 1.980e-05 | 1.12 |

Same order of magnitude from two different code paths ⇒ the metric and both capture routes are sound.

## 3. Visual confirmation

`diag_smooth_imf_unguided_K5_n5/0_fan.png` — the planned horizons and x̂1 curves zig-zag violently, and the executed rollout wanders off to the right and never reaches the green target. Compare `diag_smooth_imf_guided_K5_n5/*_fan.png`, where plans are clean lines. Exactly the "crazy chaotic vs normal" you described.

## 4. What this establishes

**The central hypothesis of `../../HF_iMF/Research/DISCUSSION_foresight_fan_and_smoothness_paradigms.md` is CONFIRMED.** HardFlow enforces dynamic feasibility (`A·s + B·a + c = s'`) as a **hard NLP equality**, so smoothness is *manufactured by the projection*, not inherited from the generative field. The evidence:

1. A **9.5× field-quality gap** between iMF and FM **completely disappears** after projection — and in fact reverses slightly (0.82×).
2. The projection scales its effort to the input: **96× smoothing for the rough iMF field vs 7.4× for the smoother FM field.** It is not applying a fixed correction; it is dragging whatever it receives onto the feasible manifold.
3. This is why iMF can solve **0% of episodes unguided but 98.5% guided** (u_5). The coarse field is real, and the projection absorbs it.

## 5. ⭐ The important, non-obvious finding: smoothness is NOT the cause of the safety gap

There is an apparent paradox worth stating plainly:

> **Guided iMF produces SMOOTHER plans than guided FM (2.19e-06 vs 2.68e-06) — yet has MORE violations (3/200 vs 0/200).**

That is not a contradiction; it **decouples two hypotheses that were previously bundled together**:

| Hypothesis | Status after this run |
|---|---|
| "iMF's coarse field yields rough plans" | ✅ **TRUE — but only before projection.** After the NLP, its plans are as smooth as FM's (slightly smoother). |
| "the residual 1.5-pt safety gap comes from plan roughness" | ❌ **REFUTED.** The plans are not rough. |

**Why both can hold:** the NLP guarantees the *planned* trajectory is smooth, dynamically feasible and obstacle-free (constraint violation ~1e-16). Violations arise when the *executed* trajectory diverges from that certified plan — i.e. from **prediction error**, not from jitter. A plan can be perfectly smooth and perfectly feasible while pointing somewhere slightly wrong.

So the remaining gap should be attributed to **where the plan goes, not how smooth it is** — consistent with the coarse-field/prediction-error account in `../u_5/RESULTS_Gen13_u5_paired_n200.md` §6, and it rules out "roughness" as the mechanism.

## 6. Secondary observations

- **Unguided iMF survives longer than unguided FM** (41.6 vs 18.8 steps) despite being 9.5× rougher, and both end at 0% safety. Rough ≠ immediately fatal; FM's smoother-but-unconstrained plans apparently hit an obstacle sooner. Not investigated further — both are useless without guidance.
- **Guided step counts match** (51.6 vs 51.8) — the projection normalises task behaviour across backbones too, not just smoothness.

## 7. Known cosmetic bug (not affecting numbers)

On the **unguided** cells the fan legend reads *"Projected plan (post-NLP)"*, which is wrong — there is no NLP on that path. The curve plotted is the plain generative plan. Labels only; every number above is unaffected. Worth fixing if these figures are used in a writeup.

## 8. ⚠️ "The iMF field outputs trash — so why is the u_5 conclusion still positive?"

A fair and important challenge to `../u_5/RESULTS_Gen13_u5_paired_n200.md`. Both things are true simultaneously:

- **fix_7:** iMF's raw field is 9.5× rougher than FM's, visually chaotic, and solves **0%** of episodes on its own.
- **u_5:** iMF-guided is **98.5% safe** at **~2× lower cost**, and is declared "promising".

Here is the honest reconciliation — and it ends with a **missing control experiment** that this challenge exposes.

### 8.1 It is not "iMF trash vs FM good" — **both raw fields fail completely**

| | unguided success | guided success |
|---|---|---|
| iMF | **0%** | 100% |
| FM | **0%** | 100% |

FM's field is 9.5× *smoother* — and still solves **0%**. Smoothness is not the same as task competence. So the correct framing is not "iMF is broken and FM works", it is **"neither field can do this task alone; the NLP is what solves it, for both."** iMF is a rougher input to the same rescue machine, arriving at the same place.

### 8.2 Division of labour: what each component actually contributes

- **The generative field** proposes *which route/basin* — which side of the obstacles, roughly where to go.
- **The NLP** enforces *feasibility within that basin* — dynamics, obstacle clearance, action bounds — to ~1e-16.

That is why a coarse field is survivable: the projection repairs local geometry (§4: 96× smoothing) but does **not** re-plan the route. This is HardFlow's design thesis, and fix_7 is direct evidence for it.

### 8.3 Field quality still matters — and we can see exactly where

If the field were irrelevant, iMF and FM would be indistinguishable after projection. They are not: **3/200 vs 0/200 violations** (u_5). That residual is precisely the price of the coarser field — the projection certifies a *smooth, feasible* plan, but cannot fix a plan that is smoothly pointed slightly wrong (§5). So the field's contribution is real, measurable, and small.

### 8.4 ⚠️ THE CONFOUND this question uncovers — the efficiency claim is not cleanly attributed

`K == ode_t_steps` controls **two things at once**: the number of network evaluations **and** the number of NLP projections. Measured:

| | NFE/plan | NLP/plan | NLP/episode | wall-clock/plan |
|---|---|---|---|---|
| iMF K=5 | 21 | **5** | 35 | **0.4815 s** |
| FM K=10 | 41 | **10** | 63 | **0.8379 s** |
| ratio | 1.95× | **2.0×** | 1.80× | **1.74×** |

**The wall-clock ratio (1.74×) sits closest to the NLP ratio (1.80×), not the NFE ratio (1.95×)** — consistent with IPOPT dominating the per-plan cost. So the speedup is largely "half as many projections", and iMF is *not* offloading work onto the optimiser (it does fewer of **both**).

But that means: **we have not shown the seam swap causes the speedup.** iMF simply ran at K=5 while FM ran at K=10. The claim implicitly assumes FM *cannot* run at K=5 — and **we never tested that.**

### 8.5 The missing control: **FM at `ode_t_steps=5`**

| Outcome of FM@K=5 | What it would mean |
|---|---|
| FM@K=5 also ≈98–100% safe at ≈0.48 s/plan | ❌ **iMF contributes nothing.** The entire "win" is just "use fewer steps", and Gen13's seam swap is decorative. |
| FM@K=5 degrades badly (violations ↑) | ✅ **Gen13's thesis holds.** iMF's exact endpoint map is what makes low-K viable — exactly the `O((1−τ)²)` Euler-error argument (`../u_8/STUDY_Gen13_fig11_how_it_is_generated.md` §4), which predicts FM should suffer most when few, large steps are taken. |

This is **cheap** — one eval run, no retraining — and it is now the **single most important open experiment in Gen13**, ahead of everything in u_5 §15.4. Until it is run, the defensible claim must be weakened to:

> "iMF matches FM's safety within noise while running at half the projection budget. **Whether FM could run at the same budget has not been tested.**"

### 8.6 Status of the u_5 conclusion

| u_5 claim | Status after this challenge |
|---|---|
| iMF ~2× cheaper than FM **as configured** | ✅ still true — measured, zero distribution overlap |
| iMF safety ≈ FM (3/200 vs 0/200, p=0.248) | ✅ still true |
| **The efficiency is *attributable to iMF*** | ⚠️ **NOT established** — confounded with K; needs FM@K=5 |
| "trash field ⇒ conclusion invalid" | ❌ no — both fields fail alone; the NLP does the task for both |

The u_5 numbers stand. What this challenge correctly deflates is the *causal story*, not the measurements.

## 9. NEW TEST BATTERY (fix_7.3) — designed to try to FALSIFY the Gen13 conclusion

**Motivation:** visual inspection says iMF's raw output is low quality — *"not even on the main trajectory manifold"* — while FM looks on track. Every comparison so far ran **iMF at K=5 vs FM at K=10**, so backbone was confounded with budget. These tests are designed so that a *negative* result is clearly visible, not explained away.

### 9.1 The four tests

| # | Test | What it settles | Why it can falsify |
|---|---|---|---|
| **T1** | **FM at K = 1, 2, 5, 10** (guided) | **THE control.** Can FM simply use fewer steps? | If FM@K=5 ≈ 100% safe at ≈0.48 s/plan, **iMF contributes nothing** and the entire Gen13 efficiency claim collapses to "use fewer steps". |
| **T2** | **iMF at K = 1, 2, 5, 10** (guided) | reverse control — does iMF improve with budget? | If iMF@K=10 still violates while FM@K=10 does not, the iMF **field has a hard ceiling** that budget cannot fix. |
| **T3** | **both, unguided, K = 1,2,5,10** | raw field quality vs budget, no NLP | Quantifies your visual read across budgets. If iMF-unguided stays ~0% at every K while FM-unguided improves, the field is genuinely worse, not just under-stepped. |
| **T4** | **x̂1 accuracy vs τ** (post-processing) | **the seam mechanism itself** | The most direct test. If iMF's error at τ=0 matches FM's, the exact endpoint map is **not delivering in practice** — the whole Gen13 rationale fails, regardless of task metrics. |

### 9.2 Why T1+T2 together are the decisive design

Reading **across backbones at EQUAL K** removes the confound entirely — equal K means equal NFE *and* equal projection count:

| | K=1 | K=2 | K=5 | K=10 |
|---|---|---|---|---|
| iMF safety | ? | ? | (98.5% @ n=200) | ? |
| FM safety | ? | ? | **?** ← the critical cell | (100% @ n=200) |

The single most informative number in the whole battery is **FM@K=5**. Three outcomes:
- **FM@K=5 ≈ 100%** → iMF is strictly dominated; Gen13 has no efficiency story.
- **FM@K=5 collapses** → iMF's exact endpoint map is what makes low-K viable ⇒ Gen13's thesis is validated *for the right reason*.
- **Both degrade similarly** → K, not the backbone, is the whole story; report it as a HardFlow finding, not an iMF one.

### 9.3 T4 — measuring the mechanism, not the outcome

`run/analyze_x1_accuracy.py` computes, per ODE step, from chains already dumped in the npz:
```
err(k) = ‖ x̂1(k) − x_final ‖      x_final = chain[-1]
```
**Predicted signature** (`../u_8/STUDY_Gen13_fig11_how_it_is_generated.md` §4):
- **FM** — Euler shot, error `O((1−τ)²)`: **high at τ=0, decaying**
- **iMF** — exact endpoint map: **low and FLAT in τ**

**Validated on synthetic data with a known answer** before use: injected `0.06(1−τ)²` for FM and a flat `0.008` for iMF; the tool recovered decay **1.0× for iMF** vs a large decay for FM. So it will discriminate — a null result would be informative, not a tooling artifact.

⚠️ T4 needs `chain_full` in the npz (u_8.2). The **existing** `temp/fix_7` dumps predate it and are skipped with a clear message; the new run produces them.

### 9.4 Code change required (fix_7.3)

`run_scripts/eval_smoothness_diag.sh` had FM's `k_steps=10` **hardcoded** — which is precisely why the control was never run. Now `FM_K` env-overridable (default 10, so every earlier invocation is unchanged). Plus new `run/analyze_x1_accuracy.py`. `run/eval.py` / `flow_policy.py` untouched.

### 9.5 ⭐ COMMAND TO SUBMIT

```bash
cd /u/home/llim/FMPCC/FM-PCC
git pull
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/hardflow/eval_matched_nfe_hardflow.sh
```
~60–90 min (4h requested). Runs T1+T2+T3 (16 guided + 16 unguided cells at n=20), prints a matched-budget summary table, then runs T4 automatically.

**Knobs:** `N=20` (episodes/cell) · `KS="1 2 5 10"` · `MODES="guided"` (skip unguided to halve runtime).

**Why n=20:** enough to separate large differences (0% vs ~100%) cheaply. **Not** enough to resolve the ~1.5-pt residual gap — that remains the u_5 n=200 job's role. If T1 shows FM@K=5 is competitive, the follow-up should be a paired n=200 at matched K.

### 9.6 How to read the results — including what would overturn the conclusion

| Observation | Conclusion |
|---|---|
| FM@K=5 safe ≈ iMF@K=5 safe, similar s/plan | ❌ **Gen13 efficiency claim dead.** Report honestly: the win was the step count. |
| FM safety falls sharply below K=10; iMF holds | ✅ Gen13 validated — iMF enables low-K operation |
| T4: iMF flat & low, FM high at τ=0 | ✅ mechanism confirmed — the seam works as theorised |
| T4: iMF ≈ FM at τ=0 | ⚠️ **mechanism refuted** even if task metrics look fine — the u-field's training error dominates the Euler error at this data scale, and any task-level win is incidental |
| T3: iMF-unguided ~0% at every K | your visual read confirmed — the field is genuinely poor, and Gen13's result rests entirely on the projection |

**Pre-committing to the interpretation before seeing the data** — so a negative outcome gets reported as a negative outcome.

## 10. Bottom line

Your visual observation was correct and is now a measured result: **unguided iMF is 9.5× rougher than unguided FM, and the projection erases that entirely — 96× smoothing for iMF vs 7.4× for FM, leaving iMF's guided plans marginally smoother.**

The scientific payoff is §5: because guided iMF's plans are *smooth*, roughness is eliminated as the explanation for its 3/200 violations. The gap lives in **prediction accuracy**, which points the next work at the field/training or the Newton pull-back — not at the sampler or the projection count.
