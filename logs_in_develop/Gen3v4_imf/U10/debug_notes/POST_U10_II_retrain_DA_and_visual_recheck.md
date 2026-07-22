# POST-U10-II — the retrained `imf_official` run: DA comparison + visual re-inspection

**Date:** 2026-07-19 · **Revised:** 2026-07-22 (major — §1 was wrong; see banner)
**Source:** `temp/candidates_multidimensional_aggregated.csv` (multidimensional DA aggregation)
**Compared:** DA candidates **45–48** vs **49–52**
**Companion:** `../K2_train_eval/ANALYSIS_imf_official_K2_train_curve_and_eval.md` (the first-setup K-sweep analysis, §0–§9)
**Successor:** [`POST_U10_III_large_batch_test_and_theory_corrections.md`](POST_U10_III_large_batch_test_and_theory_corrections.md)

---

## 🔴 REVISION BANNER — what the 2026-07-19 version got wrong

The original §1 claimed *"What actually changed between OLD and NEW is **not recorded in this repo**"*
and built §5/§6 on that premise. **This was false.** The delta was recorded the whole time, per-run, by
the Smart Config Snapshot mechanism (`flow_matcher_v3_imeanflow/utils/setup.py:187` `snapshot_configs()`),
which copies the live config into `<savepath>/config_snapshot_avoiding-d3il/` at every launch.

The original author checked the **working tree** (`git status` on `config/avoiding-d3il.py`) instead of the
**run folders**. The working tree is not the record of what a run used — the snapshot is.

**Four parameters changed, not zero. One of them halves the training budget.** Everything the old note
framed as "an unattributable wash" is now attributable, and the "wash" reading itself does not survive:
the two runs were never a controlled A/B. Corrected content in §1–§2; the statistics argument (old §0)
and the DA tables (old §2/§3) stand and are retained as §3–§5.

---

## §1 — The parameter delta (verified, this session)

`diff` of the two per-run snapshots in `temp/gen3v4u10/`:

| knob | OLD (setup1) | NEW (setup2) | direction |
|---|---|---|---|
| `n_train_steps` | 100000 | **50000** | 🔴 **half the training** |
| `meanflow_data_proportion` | 0.5 | **0.25** | 🔴 half the FM anchors |
| `p_std` | 1.0 | **1.4** | wider time distribution |
| `meanflow_cfg_smax` | 7.0 | **3.0** | narrower train-time CFG range |

`projection_eval.yaml` is **byte-identical** between the two (`md5 93537ac0…`) — the projection/eval side did
not move. Only the training block of `avoiding-d3il.py` changed; the diff is exactly 4 lines
(`avoiding-d3il.py:488,489,532,550`).

**Snapshot timestamps** — these also settle the OLD/NEW identification independently of the manual rename:

| snapshot dir | stamps |
|---|---|
| `…(setup1)` | 2026-07-16 21:35:34 / 21:38:05 / 21:40:45 |
| `…(setup2)` | 2026-07-18 20:51:10 / 20:53:30 / 20:56:00 |

setup1 precedes setup2 by two days, matching OLD→retrain. The old note's K100-based argument reached the
same conclusion by inference; it is now **verified**. (Old §7 listed this as "inferred" — upgrade it.)

The **working tree today matches setup1 (OLD)**: `n_train_steps=100000`, `p_std=1.0`,
`meanflow_data_proportion=0.5`, `meanflow_cfg_smax=7.0`. So the NEW run's parameters were never committed —
they exist *only* in that run's snapshot. That is why `git status` looked clean and why the original note
concluded nothing had changed.

### 1.1 Why the folder names collided (this part of the old §1 was right)

`args_to_watch_fmv3_imf_train` (`config/avoiding-d3il.py:66-76`) is
`prefix, horizon, diffusion, time_beta_alpha_v3, time_beta_beta_v3, action_weight, imf_objective, imf_backbone, t_schedule`.
**None of the four changed knobs is in that list**, so both runs generate the byte-identical folder
`…_a1.0_b1.0_aw10_objimf_official_bbdit_tslogit_normal`. The old note's observation was correct; only its
inference ("therefore unrecorded") was wrong.

🔴 **This is a live overwrite hazard, not a cosmetic one.** Without the manual `(setup1)` rename, the NEW
run would have written its checkpoints into the OLD run's directory. Commit `4d402847` ("derive experiment
name and checkpoint dynamically … preventing silent overwrites") addresses this class of bug on the iMF
pipeline — this run pair predates it.

### 1.2 What each knob actually does

Read against `flow_matcher_v3_imeanflow/models/imf_diffusion.py::_p_losses_imf_official` (L660-700):

- **`meanflow_data_proportion`** (L671) is the fraction of the batch forced to `r == t`:
  `fm_mask = rand(B) < data_proportion; r = where(fm_mask, t, r)`. Those are the **FM anchors** — the
  `h = t − r = 0` samples, i.e. the *only* samples that supervise the **instantaneous velocity field**.
  The rest (`r < t`) supervise the averaged/jump field.
- **`p_std`** (L667-668) is the spread of `τ ~ sigmoid(N(−p_mean, p_std))`, drawn twice; `t = max(τ₁,τ₂)`,
  `r = min(τ₁,τ₂)`. Widening it pushes mass toward both endpoints, so the **gap `h` grows**.
- **`meanflow_cfg_smax`** (L687 → `_sample_cfg_scale`, L494) sets the per-sample guidance draw
  `ω = exp(u·log1p(s_max)) ∈ [1, 1+s_max]`. 7.0 → 3.0 narrows the trained guidance manifold from
  **[1, 8] to [1, 4]**.
- **`n_train_steps`** — optimizer steps. No further comment needed.

### 1.3 Quantified: the NEW run trained the instantaneous field **4× less**

Combining the two red rows (Monte-Carlo, 4×10⁵ draws, exact sampler reproduction):

| quantity | OLD | NEW | ratio |
|---|---|---|---|
| optimizer steps | 100 000 | 50 000 | **0.5×** |
| FM-anchor (`h=0`) sample budget = `dp × steps` | 50 000 | 12 500 | 🔴 **0.25×** |
| mean-flow (`h>0`) sample budget | 50 000 | 37 500 | 0.75× |
| `E[h]` per training sample | 0.116 | 0.220 | **1.9×** |
| `E[h ∣ h>0]` | 0.233 | 0.294 | 1.26× |
| `P(h > 0.5)` — long-jump samples | 0.042 | 0.140 | **3.3×** |

So NEW is not "the same run with a tweak". It is **half the compute, a quarter of the instantaneous-field
supervision, and roughly 3× the share of hard long-jump samples**, with a narrower guidance range.

### 1.4 The eval block was not updated to match (provenance hazard, not a correctness bug)

The plan/eval block still carries the OLD values — `meanflow_cfg_smax: 7.0`, `meanflow_data_proportion: 0.5`
(`config/avoiding-d3il.py:887-888`), `p_std: 1.0` (L858) — while the NEW checkpoint was trained at 3.0 /
0.25 / 1.4. Under config-overrides-pkl these overwrite the checkpoint's recorded values.

**This does not corrupt the eval numbers.** For `imf_objective='imf_official'`, all three knobs are read
only inside `_p_losses_imf_official`; `loss()` short-circuits to it at L414, and sampling uses the eval
operating point `meanflow_cfg_omega=1.0` (guidance off) with `flow_steps_v3`. They are genuinely train-only
here, exactly as the config comment claims. But it does mean **the eval-side record of this checkpoint
states parameters the checkpoint was not trained with** — the same provenance failure that produced the
original §1.

---

## §2 — What §1 does to the comparison

**The A/B is confounded four ways, and one confound points the same direction as the null result.**

1. **Half the steps.** Any "NEW ≈ OLD" reading is a comparison at *half the budget*, which is a
   materially different statement from "the parameter change did nothing". Both are consistent with
   the data; the old note asserted neither correctly.
2. **The FM-anchor cut is the wrong-direction knob.** §6's standing diagnosis — the model is under-fit
   in the **instantaneous-velocity limit** — names precisely the quantity that `data_proportion`
   supervises. The retrain cut that budget 4×. Under the note's own hypothesis, this should have made
   raw smoothness *worse*.
3. **`p_std` compounds it.** More mass at large `h` moves capacity further toward long jumps and away
   from the small-`h` regime where smoothness lives.
4. **`smax` 7→3 is the one plausibly-helping change.** Eval runs at ω=1; concentrating the trained
   guidance manifold near the operating point should improve conditioning *there*. This is a
   candidate mechanism for the single positive in §4A (raw `n_success` = 1.00 at every K) — untested,
   but it is now a named hypothesis rather than an unexplained blip.

**Net:** the retrain was not an ablation of anything. Four knobs moved at once, two of them large, in
opposing directions, at half budget. **No single-cause attribution is available from this run pair** — but
unlike the original note's version of that statement, the reason is *over-determination, not missing data*.

---

## §3 — Sample size: read this before any number below

**The two runs are statistically indistinguishable by construction. This DA cannot decide "small upgrade vs small downgrade".**

| | iMF runs (45–52) | FM/DPCC baselines in the same CSV (54–56) |
|---|---|---|
| seeds aggregated (`count`) | **1** | **5** |
| `Missing_Seeds` | `[7, 8, 9, 10]` | `` (none) |
| granularity of `n_success` | **0.5** → 2 episodes per cell | 0.1 → 10 episodes per cell |

Every iMF cell is **2 episodes, 1 seed** (the old eval log confirms a single seed folder:
`…/iMeanFlowODE/6`). A cell moving `1.00 → 0.50` is **one episode flipping**. In the 3-halfspace averages
below the quantum is `1/6 ≈ 0.17` — again one episode.

The user's own reading ("maybe a small percentage upgrade/downgrade but it is not obvious") is the correct
one: **there is no resolution here to be obvious with.** The baselines were measured with 5× the samples,
so iMF-vs-baseline comparisons in this CSV are also not like-for-like.

This section is unchanged from the original note and is the part of it that held up.

---

## §4 — Headline DA tables

> Numbers below are carried over from the 2026-07-19 session. `temp/candidates_multidimensional_aggregated.csv`
> is no longer present in the working tree, so they were **not re-verified on 2026-07-22**. The §1–§2
> corrections do not depend on them.

### 4A `diffuser` (raw, unprojected — the variant the smoothness eyeballing refers to)

Averaged over the 3 halfspace settings (quantum = 0.17 = one episode):

| metric | O-K1 | O-K2 | O-K10 | O-K50 | O-K100 | N-K1 | N-K2 | N-K10 | N-K50 |
|---|---|---|---|---|---|---|---|---|---|
| `n_success` | 0.83 | 1.00 | 0.83 | 1.00 | 0.83 | **1.00** | **1.00** | **1.00** | **1.00** |
| `n_success_and_constraints` | 0.17 | 0.33 | 0.00 | 0.17 | 0.00 | 0.17 | 0.33 | 0.17 | 0.17 |
| `n_violations` | 18.0 | 11.0 | 18.2 | 18.2 | 17.5 | 12.7 | 16.3 | 16.2 | 16.3 |
| `n_steps` | 66.8 | 63.5 | 77.2 | 66.0 | 72.0 | 62.0 | 67.0 | 66.0 | 66.0 |

The single cleanest NEW signal: **raw `n_success` is 1.00 at every K**. OLD lost `top-right-hard` at
K1/K10/K100 (0.50); NEW solves it at all four K — 3 recovered episodes. Suggestive of a slightly
better-conditioned raw field; §2 point 4 gives `smax` 7→3 as the candidate mechanism. Raw `n_violations`
did **not** improve (18.0→12.7 at K1 but 11.0→16.3 at K2) — noise-dominated.

### 4B `dpcc-r-tightened` (the "dpcc-rtc" variant the user quotes), per halfspace

`n_success_and_constraints`:

| halfspace | O-K1 | O-K2 | O-K10 | O-K50 | O-K100 | N-K1 | N-K2 | N-K10 | N-K50 |
|---|---|---|---|---|---|---|---|---|---|
| top-left-hard | 1.00 | 1.00 | 0.50 | 1.00 | 0.50 | 1.00 | 1.00 | 0.50 | 1.00 |
| **top-right-hard** | 0.00 | 1.00 | 1.00 | 1.00 | 0.50 | 0.50 | 0.50 | 0.50 | **0.00** |
| both-hard | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

- `both-hard` is **1.00 everywhere, OLD and NEW, at every K** — unchanged from the first analysis.
- `top-left-hard` is identical OLD↔NEW cell-for-cell, including the K10 dip and its 3–3.5 violations.
- **All movement is confined to `top-right-hard`** — 6 episodes total across the 4 K values. NEW is worse
  there (0.5/0.5/0.5/0.0 vs 0/1/1/1). NEW-K50's `n_steps = 40.0 ± 25.0` with 0 success is a *failure*
  signature, not a speed-up (episodes ending early at ~15 and ~65 steps).

### 4C Cost

`avg_time` is unchanged (OLD 0.60 s vs NEW 0.61 s mean over all 156 comparable cells) and scales with K as
expected — `diffuser` 0.01/0.03/0.13/0.67 s for K1/K2/K10/K50. Nothing regressed on compute. Note this is
**inference** cost; NEW's *training* cost was half (§1.3).

---

## §5 — Aggregate over all 13 projection variants × 3 halfspaces × 4 K (156 cells)

| metric | result |
|---|---|
| `n_success` | NEW better in **7** cells, worse in **28**, identical in **121** |
| `n_success_and_constraints` | NEW better in **29**, worse in **43**, identical in **84** |
| `n_violations` | OLD mean **7.06** → NEW **6.87** (flat) |
| `n_steps` | OLD mean **71.96** → NEW **67.86** (**−4.1 steps**, the most consistent difference) |
| `avg_time` | OLD 0.60 → NEW 0.61 (flat) |

Reading: a slight net regression on success-type metrics, a slight net improvement on episode length,
violations flat. Both directions are within the one-episode quantum, and the cells are *not independent* —
all 13 projection variants replay the same 2 underlying episodes per condition, so a single bad episode
propagates into ~13 "worse" cells. The 43-vs-29 split is therefore closer to **~3 real episodes** than to
43 real observations.

**Corrected verdict.** The original note concluded "a wash — the retrain neither fixed nor broke anything."
The defensible statement is narrower: **at 2 episodes/cell, this eval cannot resolve the difference between
a 100k-step run and a 50k-step run with a 4× smaller instantaneous-field budget.** That NEW roughly matches
OLD *at half the steps* is a mildly interesting datum in its own right — it is consistent with the
COMPARE §8.2 reading (a blind loss direction) that POST-U10-III develops, where extra steps along a
degenerate direction buy nothing. It is not evidence that the parameter change was neutral.

---

## §6 — Visual smoothness: unchanged, and now *predicted*

User's inspection of the NEW run: *low-K NFE still not smooth / somewhat chaotic but not exploding; high-K
smoother but still not as smooth as the DPCC / FMv3ODE baseline.*

Same qualitative picture as the first setup (K2_train_eval §9): raw smoothness rises mildly with K, never
reaching FM/DPCC quality, while low K stays angular.

**The original note called this "the retrain did not move it" and treated it as an open puzzle. Given §1 it
is the expected outcome.** Smoothness at low NFE is a property of the instantaneous-velocity limit
(`h → 0`); the only samples supervising it are the FM anchors; NEW trained them **4× less** (§1.3) and
shifted the remaining mass toward large `h` (`P(h>0.5)` up 3.3×). A retrain that cuts the budget for the
exact quantity under diagnosis, and does not degrade it visibly, says the model is **not anchor-starved at
these budgets** — which is a genuine (if weak) result, and a different one from "the retrain changed
nothing".

This observation is also **not sample-starved**, unlike §4–§5 — a coarse-looking trajectory is coarse in
every episode. The §1/§2 diagnosis of the first analysis (instantaneous-velocity limit under-fit) remains
unrefuted.

Three separate notions keep getting merged here; the distinction from
`HF_iMF/Research/DISCUSSION_foresight_fan_and_smoothness_paradigms.md` §2 applies directly:

- **S1 generative smoothness** — what is being eyeballed on the `diffuser` variant. Still poor. Model-side.
- **S2 dynamic feasibility** — what the DPCC projection enforces. This is why "after projection all is smooth".
- **S3 cross-replan consistency** — untested here.

"Raw is chaotic but projected is smooth" is **expected pipeline behaviour**, not a new symptom. The open
question is only whether S1 quality buys anything downstream once the projection is on — and §4B/§5 say: at
this sample size, no measurable amount.

---

## §7 — What this run does and does not establish

**Establishes:**
1. The exact parameter delta OLD→NEW: `n_train_steps` 100k→50k, `data_proportion` 0.5→0.25, `p_std` 1.0→1.4,
   `cfg_smax` 7.0→3.0 — and that the OLD/NEW identification is confirmed by snapshot timestamps, not just
   by the K100 argument.
2. The retrain did **not** regress the pipeline at half the training budget (violations flat, inference time
   flat, `both-hard` perfect at all K).
3. Raw `diffuser` success is 1.00 at every K including K1 — the only positive worth naming; `smax` 7→3 at an
   ω=1 eval point is the standing hypothesis for it.
4. Cutting FM anchors 4× did **not** visibly degrade S1 smoothness ⇒ the model is not anchor-starved at
   these budgets.
5. The DA evidence base for iMF is **1 seed / 2 episodes per cell**, 5× thinner than the FM/DPCC baselines
   it is being compared against.
6. Two runs of this arm generate **identical folder names** — a silent-overwrite hazard, survived here only
   by a manual rename.

**Does not establish (and cannot, as run):**
- Any success/violation ranking between OLD and NEW, in either direction.
- The effect of **any individual** parameter — four moved at once, in opposing directions, at half budget.
- Whether the parameter change helped the field — needs the **train-loss curve**, compared at *matched step
  counts* (§8.1).
- Any K-ordering claim. The first analysis's §9 finding (low K commits to a mode, high K mode-averages) is
  neither confirmed nor contradicted; NEW's `top-right-hard` K50 failure is weakly consistent with it, on
  2 episodes.

---

## §8 — Recommended next steps, in order of information-per-GPU-hour

1. **Post the NEW train log — and compare at step 50 000, not at the end.** Read `raw_mse` / `aux_loss` /
   `a0_loss` against the OLD run's *value at the same step*, not its final plateau (OLD 11.3 → ~2.3 for
   `raw_mse` over 100k). A final-vs-final comparison here measures the step-count difference, not the
   parameter change. This is free — both logs exist. **Also confirm which checkpoint the eval loaded**;
   with `n_train_steps` halved, checkpoint numbering between the two runs is not comparable by index.
   (`loss` / `test/loss` are flat by construction — see §0 of the K2 analysis: `adp(L)=L/(L+0.01)` saturates.)
2. **Fix the sample size before running any more comparisons.** Filling seeds 7–10 (`Missing_Seeds`) takes
   the quantum from 0.5 to 0.1 and makes iMF rows comparable to the 54–56 baselines. Every A/B question
   asked of this pipeline is unanswerable until this is done — running more K values at n=2 adds cost and
   no information.
3. **Change one knob at a time.** The four-way delta cost this comparison its interpretability outright.
   If the target is S1 smoothness, `meanflow_data_proportion` is the knob that addresses it — and it should
   go **up** from 0.5, not down.
4. **Close the provenance gaps** (all three are real, and #1 caused the original §1 error):
   - **Read the run's `config_snapshot_avoiding-d3il/`, never the working tree,** when reconstructing what
     a run used. The snapshot is written by `snapshot_configs()` into `<savepath>/` at every launch.
   - **Add the training knobs to `args_to_watch_fmv3_imf_train`** — or rely on the dynamic experiment
     naming from `4d402847` — so two runs of this arm cannot collide on one folder.
   - **Keep the plan/eval block in sync with the training block** (`smax`, `data_proportion`, `p_std`).
     Inert at eval for this objective (§1.4), but it currently mislabels the checkpoint.
5. **Stop treating raw smoothness as the primary readout, or commit to measuring it.** Either judge the
   model by projected task metrics at fixed NFE (the HardFlow lens), or turn S1 into a number — e.g. mean
   squared second difference of the planned trajectory, pre- vs post-projection — so "not as smooth as
   DPCC" becomes a value trackable across retrains instead of re-eyeballed each time.

---

## §9 — Verified vs inferred

**Verified 2026-07-22 (this revision):** the 4-line parameter delta and the identical
`projection_eval.yaml` (`diff` + `md5sum` of the two snapshot dirs in `temp/gen3v4u10/`); the snapshot
timestamps and hence the OLD/NEW ordering; that the working tree matches setup1; that none of the four
knobs is in `args_to_watch_fmv3_imf_train` (`config/avoiding-d3il.py:66-76`); that `snapshot_configs()`
writes into each run's `savepath` (`flow_matcher_v3_imeanflow/utils/setup.py:187`); the code paths for
`data_proportion` / `cfg_smax` / `p_std` (`imf_diffusion.py:414, 667-671, 687, 494`) and that all three are
train-only under `imf_official`; the eval block's stale 7.0 / 0.5 / 1.0 (`avoiding-d3il.py:858, 887-888`).
The `E[h]`, `P(h>0.5)` and anchor-budget figures in §1.3 are Monte-Carlo (4×10⁵ draws) reproducing the
sampler at L667-671 — derived, not measured from the runs.

**Verified 2026-07-19 (original session), not re-checked:** `count=1` seed for 45–52 vs `count=5` for
baseline 54; `Missing_Seeds=[7,8,9,10]`; the 0.5 granularity of `n_success`; all numbers in §4 and §5; the
single seed folder `…/iMeanFlowODE/6` and the CFG-off eval overrides in the OLD eval log. The source CSV is
no longer in the working tree.

**Inferred:** that `smax` 7→3 explains §4A's raw success improvement (mechanism is plausible and the eval
operating point is ω=1, but untested); that the 43-vs-29 flip split reflects ~3 real episodes rather than 43
independent observations (follows from the shared-episode structure of the projection variants, not
separately measured); that NEW actually ran to its configured 50 000 steps (from the config value — the
train log would confirm).

**Not claimed:** that the retrain helped or hurt. On this evidence it did neither measurably — and because
four parameters moved at once at half budget, this run pair could not have shown which one mattered even
with adequate sample size.

**Retracted from the 2026-07-19 version:** "What actually changed … is not recorded in this repo" (§1);
"whatever changed is a knob that does not enter `args_to_watch`" as an argument that the change was
*unrecorded* (the folder-name observation itself stands, §1.1); "Net verdict: a wash" (§3) and "the retrain
did not deliver a visible S1 improvement — the central hypothesis it was meant to test" (§5), both of which
presumed a controlled A/B that did not exist.
