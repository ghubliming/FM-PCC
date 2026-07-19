# POST-U10-II — the retrained `imf_official` run: DA comparison + visual re-inspection

**Date:** 2026-07-19
**Source:** `temp/candidates_multidimensional_aggregated.csv` (multidimensional DA aggregation)
**Compared:** DA candidates **45–48** vs **49–52**
**Companion:** `../K2_train_eval/ANALYSIS_imf_official_K2_train_curve_and_eval.md` (the first-setup K-sweep analysis, §0–§9)

---

## Which candidate is which

| DA candidates | Log folder | Read as |
|---|---|---|
| 44, 45, 46, 47, 48 | `…_objimf_official_bbdit_tslogit_normal**(setup1)**` → K100, K10, K1, K2, K50 | **OLD** — the first setup, analysed in `K2_train_eval/ANALYSIS…md` |
| 49, 50, 51, 52 | `…_objimf_official_bbdit_tslogit_normal` (unsuffixed) → K10, K1, K2, K50 | **NEW** — the retrain after the parameter update |

Assignment rationale (worth stating because both folders differ only by a manual `(setup1)` rename):
the `(setup1)` set is the only one that carries **K100**, which matches the first-setup sweep exactly (K1/K2/K10/K50/K100);
the NEW set is K1/K2/K10/K50 only. Both were written to the *same* generated folder name, i.e. **no
name-carrying hyperparameter changed** (`a1.0_b1.0_aw10_objimf_official_bbdit_tslogit_normal` is identical).

---

## §0 — Read this before reading any number below

**The two runs are statistically indistinguishable by construction. This DA cannot decide "small upgrade vs small downgrade".**

| | iMF runs (45–52) | FM/DPCC baselines in the same CSV (54–56) |
|---|---|---|
| seeds aggregated (`count`) | **1** | **5** |
| `Missing_Seeds` | `[7, 8, 9, 10]` | `` (none) |
| granularity of `n_success` | **0.5** → 2 episodes per cell | 0.1 → 10 episodes per cell |

So every iMF cell is **2 episodes, 1 seed** (the old eval log confirms a single seed folder: `…/iMeanFlowODE/6`).
A cell moving `1.00 → 0.50` is **one episode flipping**. In the 3-halfspace averages further down, the
quantum is `1/6 ≈ 0.17` — again one episode.

This is exactly the reason the user's own reading ("maybe a small percentage upgrade/downgrade but it is
not obvious") is the correct reading: **there is no resolution here to be obvious with.** The baselines
were measured with 5× the samples, so iMF-vs-baseline comparisons in this CSV are also not like-for-like.

---

## §1 — What actually changed between OLD and NEW is *not recorded in this repo*

- `config/avoiding-d3il.py` in the working tree is **unmodified** (`git status` clean for it; last touching
  commit is `19935220`, the config→pkl two-tier fix). Current values are still `n_train_steps=100000`,
  `p_mean=-0.4`, `p_std=1.0`, `meanflow_data_proportion=0.5`, `meanflow_cfg_smax=7.0`, `meanflow_cfg_omega=4.0`, `lr=5e-4`.
- The generated folder name is byte-identical between OLD and NEW, so whatever changed is a knob that does
  **not** enter `args_to_watch_fmv3_imf_train`.

**Consequence:** this note can describe *what the retrain did to the metrics*, but cannot attribute it to a
cause. **To close this, the NEW train log (`*_imf_train_*.log`) is needed** — specifically the
`raw_mse` / `aux_loss` / `a0_loss` plateau heights, which are the only real convergence signals
(`loss` and `test/loss` are flat by construction — see §0 of the K2 analysis: `adp(L)=L/(L+0.01)` saturates).
Without it there is no way to tell whether the field got better and the eval noise hid it, or the field
did not move at all.

---

## §2 — Headline DA tables

### 2A `diffuser` (raw, unprojected — this is the variant the smoothness eyeballing refers to)

Averaged over the 3 halfspace settings (quantum = 0.17 = one episode):

| metric | O-K1 | O-K2 | O-K10 | O-K50 | O-K100 | N-K1 | N-K2 | N-K10 | N-K50 |
|---|---|---|---|---|---|---|---|---|---|
| `n_success` | 0.83 | 1.00 | 0.83 | 1.00 | 0.83 | **1.00** | **1.00** | **1.00** | **1.00** |
| `n_success_and_constraints` | 0.17 | 0.33 | 0.00 | 0.17 | 0.00 | 0.17 | 0.33 | 0.17 | 0.17 |
| `n_violations` | 18.0 | 11.0 | 18.2 | 18.2 | 17.5 | 12.7 | 16.3 | 16.2 | 16.3 |
| `n_steps` | 66.8 | 63.5 | 77.2 | 66.0 | 72.0 | 62.0 | 67.0 | 66.0 | 66.0 |

The single cleanest NEW signal: **raw `n_success` is 1.00 at every K**. OLD lost `top-right-hard` at
K1/K10/K100 (0.50); NEW solves it at all four K. That is 3 recovered episodes — suggestive of a slightly
better-conditioned raw field, and consistent with "low K is no longer exploding, and now also not failing".
Raw `n_violations` did **not** improve (18.0→12.7 at K1 but 11.0→16.3 at K2) — noise-dominated.

### 2B `dpcc-r-tightened` (the "dpcc-rtc" variant the user quotes), per halfspace

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

### 2C Cost

`avg_time` is unchanged (OLD 0.60 s vs NEW 0.61 s mean over all 156 comparable cells) and scales with K as
expected — `diffuser` 0.01/0.03/0.13/0.67 s for K1/K2/K10/K50. Nothing regressed on compute.

---

## §3 — Aggregate over all 13 projection variants × 3 halfspaces × 4 K (156 cells)

| metric | result |
|---|---|
| `n_success` | NEW better in **7** cells, worse in **28**, identical in **121** |
| `n_success_and_constraints` | NEW better in **29**, worse in **43**, identical in **84** |
| `n_violations` | OLD mean **7.06** → NEW **6.87** (flat) |
| `n_steps` | OLD mean **71.96** → NEW **67.86** (**−4.1 steps**, the most consistent difference) |
| `avg_time` | OLD 0.60 → NEW 0.61 (flat) |

Reading: **a slight net regression on success-type metrics, a slight net improvement on episode length,
violations flat.** Both directions are within the one-episode quantum, and the cells are *not independent*
(all 13 projection variants replay the same 2 underlying episodes per condition, so a single bad episode
propagates into ~13 "worse" cells). The 43-vs-29 split is therefore closer to **~3 real episodes** than to
43 real observations.

**Net verdict: a wash.** The retrain neither fixed nor broke anything measurable at this sample size. The
one arguably-real positive is raw `diffuser` success reaching 1.00 at all K (§2A).

---

## §4 — Visual smoothness: unchanged, and that is the informative part

User's inspection of the NEW run: *low-K NFE still not smooth / somewhat chaotic but not exploding; high-K
smoother but still not as smooth as the DPCC / FMv3ODE baseline.*

This is **the same qualitative picture as the first setup** (K2_train_eval §9): raw smoothness rises mildly
with K, never reaching FM/DPCC quality, while low K stays angular. The retrain did not move it.

That matters more than the DA numbers, because unlike the DA it is **not sample-starved** — a coarse-looking
trajectory is coarse in every episode. It says the **instantaneous-velocity limit of the learned average
field is still under-fit**, which was the §1/§2 diagnosis of the first analysis and remains unrefuted.

Three separate notions keep getting merged here; the distinction from
`HF_iMF/Research/DISCUSSION_foresight_fan_and_smoothness_paradigms.md` §2 applies directly:

- **S1 generative smoothness** — what is being eyeballed on the `diffuser` variant. Still poor. Model-side.
- **S2 dynamic feasibility** — what the DPCC projection enforces. This is why "after projection all is smooth".
- **S3 cross-replan consistency** — untested here.

So "raw is chaotic but projected is smooth" is **expected behaviour of the pipeline**, not a new symptom.
The open question is only whether S1 quality still buys anything downstream once the projection is on — and
§2B/§3 say: at this sample size, no measurable amount.

---

## §5 — What this run does and does not establish

**Establishes:**
1. The retrain did **not** regress the pipeline (violations flat, time flat, `both-hard` perfect at all K).
2. The retrain did **not** deliver a visible S1 (smoothness) improvement — the central hypothesis it was meant to test.
3. Raw `diffuser` success is now 1.00 at every K including K1 — the only positive worth naming.
4. The DA evidence base for iMF is **1 seed / 2 episodes per cell**, 5× thinner than the FM/DPCC baselines it is being compared against.

**Does not establish (and cannot, as run):**
- Any success/violation ranking between OLD and NEW, in either direction.
- Whether the changed training parameters helped the field at all — that needs the **train-loss curve**, not the eval.
- Any K-ordering claim. The first analysis's §9 finding (low K commits to a mode, high K mode-averages) is
  neither confirmed nor contradicted here; NEW's `top-right-hard` K50 failure is weakly consistent with it,
  on 2 episodes.

---

## §6 — Recommended next steps, in order of information-per-GPU-hour

1. **Post the NEW train log.** Compare `raw_mse` / `aux_loss` / `a0_loss` plateau heights against the OLD
   run's (11.3 → ~2.3 for `raw_mse`). This is a *free* comparison — the log already exists — and it is the
   only thing that can say whether the parameter change did anything. If `raw_mse` did not drop below ~2,
   the eval result is fully explained and no further eval is needed.
2. **Fix the sample size before running any more comparisons.** Filling seeds 7–10 (`Missing_Seeds`) takes
   the quantum from 0.5 to 0.1 and makes iMF rows comparable to the 54–56 baselines. Every A/B question
   asked of this pipeline is unanswerable until this is done — running more K values at n=2 adds cost and
   no information.
3. **Record the parameter delta.** The OLD/NEW folders are name-identical, so the two setups are
   distinguishable only by a manual `(setup1)` rename. Whatever knob changed should either be added to
   `args_to_watch_fmv3_imf_train` or written down in the changelog, or the next A/B will be unattributable
   in the same way.
4. **Stop treating raw smoothness as the primary readout, or commit to measuring it.** Either judge the
   model by projected task metrics at fixed NFE (the HardFlow lens), or turn S1 into a number — e.g. mean
   squared second difference of the planned trajectory, pre- vs post-projection — so "not as smooth as
   DPCC" becomes a value that can be tracked across retrains instead of re-eyeballed each time.

---

## §7 — Verified vs inferred

**Verified from the CSV / logs (this session):** the candidate→folder mapping and the `(setup1)` split; `count=1`
seed for 45–52 vs `count=5` for baseline 54; `Missing_Seeds=[7,8,9,10]`; the 0.5 granularity of `n_success`
(vs 0.1 for baselines); all numbers in §2 and §3; the single seed folder `…/iMeanFlowODE/6` and the CFG-off
eval overrides in the OLD eval log; `config/avoiding-d3il.py` unmodified in the working tree.

**Inferred:** that `(setup1)` is the OLD run (strongly supported — it is the only set containing K100, matching
the first sweep — but it rests on a manual rename, not on metadata); that the 43-vs-29 flip split reflects
~3 real episodes rather than 43 independent observations (follows from the shared-episode structure of the
projection variants, not separately measured).

**Not claimed:** that the retrain helped or hurt. On this evidence it did neither measurably.
