# `avoiding-d3il-visual` — env status, per projector, and the `af ~ mf > fm > dpcc` question

> **SNAPSHOT 2026-08-26.** Whole-env status across every visual-avoiding candidate in
> `batch_avoiding_combined_20260825_143212`. Regenerated as new batches land; use the newest
> `SNAPSHOT_<date>_*` in this folder.

**Batch:** `temp/2508/batch_avoiding_combined_20260825_143212/candidates_multidimensional_raw.csv`
**Scope:** the **8 visual candidates** out of 173 in the batch — `logs/avoiding-d3il-visual/plans/*`
(Gen9) + `logs/avoiding-d3il-visual-mix/plans/*` (Gen16). The other 165 are state-only and are
covered by [`DA_20260819_ntrials20_…`](DA_20260819_ntrials20_DPCC_vs_FM_vs_MeanFlow_vs_AlphaFlow.md).
**Companions:**
[`Gen16/init/DA_20260823_…mf_visual_avoiding…`](../../logs_in_develop/Gen16/init/DA_20260823_Gen16_mf_visual_avoiding_first_results.md) (the `mf` row, in depth) ·
[`SNAPSHOT_20260823_visual_aligning_env_status.md`](SNAPSHOT_20260823_visual_aligning_env_status.md) (the sibling env, where the premise comes from) ·
[`DA_20260819_DPCC_K20_aw10_ntrials20_vs_ntrials2.md`](DA_20260819_DPCC_K20_aw10_ntrials20_vs_ntrials2.md) (why `n_trials=2` cells cannot be ranked).

> ### The four answers up front
>
> **0. This env cannot answer a *perception* question at all — and that is a property of the task,
> not of the sample size.** The observation is clean (no obstacle coordinates ever reach the policy,
> exactly as in visual aligning — **not** a cheat), but the obstacle geometry reaches the controller
> by two routes that bypass the camera: it is **hardcoded into the projector**
> (`config/avoiding-d3il-visual.py:122-136`) and it **never varies** — `_reset_env` ignores `random`
> and `context` (`avoiding.py:257-262`). Aligning has neither route. The vision branch here is
> **redundant, not unused**, and §4b measures the cost: **+35 % per network call and no gain**. §0.2.1, §4b.
>
> **1. `af` on visual avoiding does not exist.** Zero rollouts, zero checkpoints, zero cells. The
> `af ~ mf` half of the question has no data behind it and cannot be answered from any batch. §1.
>
> **2. `mf > fm` does NOT reproduce on visual avoiding.** On the success gate `mf` K2 is *below*
> every `fm` row on its worst geometry (**0.83** arm B / **0.87** arm C vs `fm` filmv1 **1.00**), and
> on cost the comparison is not a model result at all — `mf` ran at K=2, `fm` and `dpcc` only ever at
> K=20. Per-NFE generation cost is **11.8–12.4 ms for all three** and per-NLP-solve cost at matched K
> is the same ~7 ms it is for state FM. **The 16× wall-clock gap is entirely K.** §3, §4.
>
> **3. Nothing here is decidable anyway, because the opponents are `n_trials=2`.** Every `fm` and
> `dpcc` visual cell is 2 episodes per (seed × geometry); the `mf` cell is 30. The three rows that
> score a perfect 1.00 are exactly the three rows with **6 episodes total**. Wilson 95 % CI on 2/2 is
> **[0.34, 1.00]**. §2. **The one run that would settle the question is `fm` visual at K=2 at
> `n_trials=30`** — it has never been run. §8.

---

## Reporting rules

🔴 **1. No aggregation across geometries.** The unit is the **cell** = `(model × projector ×
halfspace × seed)`. `top-left-hard`, `top-right-hard` and `both-hard` behave differently for every
model in this file; a mean over the three hides the only thing that separates them.

🔴 **2. Model-vs-model uses each model's OWN best projector, and always names it.** Best is chosen on
**worst-geometry S&C**, then on `s/ep`.

🔴 **3. Arm B (`dpcc-*`) and arm C (`hardflow_new-*`) are never mixed into one row.** Only the `mf`
candidate has arm C at all on this env.

🔴 **4. Every row carries `n` = episodes behind the cell.** `n = seeds × n_trials`. A row at `n = 2`
supports no ranking claim — see §2. Rows are never compared across different `n` without saying so.

🔴 **5. `n_trials` is a property of the eval YAML, not of the model.** `config/visual_avoiding_eval.yaml:27`
sets `n_trials: 2` (Gen9 tree: all `fm`, all `dpcc`); `config/visual_avoiding_mix_eval.yaml:33` sets
`n_trials: 30` (Gen16 tree: `mf`). Nobody chose an unfair comparison; the two trees were simply
never reconciled.

🔴 **6. Wall-clock is only comparable between candidates whose `diffuser` per-NFE cost matches.**
Two of the four `fm` candidates generate at 58–68 ms/NFE against 12 ms/NFE for everything else at
the same K and the same U-Net — those runs are node-contaminated and their timings are quarantined
(§4.1).

---

## 0. Definitions

### 0.1 Metrics

| symbol | CSV field | meaning | dir |
|---|---|---|---|
| **`S&C`** | `n_success_and_constraints` | Fraction of episodes that reach the goal **and** violate no constraint at any step. The gate. | higher |
| `succ` | `n_success` | Goal reached, constraints ignored. Reported beside `S&C` so failures can be split into *task* failures (`S&C = succ`) and *constraint* failures (`S&C < succ`). | higher |
| `viol` | `n_violations` | Mean violating steps per episode. | lower |
| **`steps`** | `n_steps` | Control steps to reach the goal, over successful episodes. | lower |
| `s/step` | `avg_time` | Wall-clock s per control step = one plan (K net calls × MPC fan 4) **plus** the projection solve. | lower |
| **`s/ep`** | derived | `steps × s/step`, per cell, then averaged. The deployment number. | lower |
| **`n`** | derived | `seeds × n_trials` — episodes behind the cell. | — |
| `[a,b]` | derived | Wilson 95 % interval on `S&C` at that `n`. | — |

### 0.2 What "visual" means on this task — vision is **additive**, not substitutive

> ⚠️ The premise "not feeding the state data, only the visual data" **does not describe this
> pipeline.** It is worth being explicit, because it changes what the numbers below mean.

D3IL avoiding's state-only observation is already `obs_dim = 4` = `[des_xy(2), c_xy(2)]` — desired and
current 2-D robot position, **no obstacle positions**
(`logs_in_develop/Gen9/…/PLAN_SINGLE_CAMERA_VISUAL_AVOIDING.md:37`, §12 audit "Option C"). The
visual variants keep that identical 4-D state (`config/avoiding-d3il-visual.py:147`,
`config/avoiding-d3il-visual-mix.py:259`) and **add** a 64-D single-camera (`bp_cam`) latent as
conditioning. The trajectory dim is 6 = `[act(2) | des_xy(2) | c_xy(2)]` in both cases.

So on avoiding, "visual" = *the same state, plus an image*. Read any visual-vs-state gap as **"what
the encoder costs"**, not as *"what perception achieves"* — §4b measures exactly that.

### 0.2.1 Is it cheating? No. Is it "pure visual"? Also no. These are different questions.

Two distinct worries get conflated here, so they are separated once, with evidence.

**Q1 — does the policy get object ground truth handed to it?** **No, and this is clean.** Neither
task's network ever sees an obstacle or box coordinate in its observation:

| | obs fed to the net | object/obstacle pose in obs? |
|---|---|---|
| visual **aligning** | 6-D `[des_c_pos(3), c_pos(3)]` (`config/aligning-d3il-visual.py:392`) | ❌ no — its state-only sibling has them at **20-D** (`:785`, `+box(3)+box_q(4)+tgt(3)+tgt_q(4)`) |
| visual **avoiding** | 4-D `[des_xy(2), c_xy(2)]` (`config/avoiding-d3il-visual.py:147`) | ❌ no — never in either arm |

On that test **visual avoiding is exactly as clean as visual aligning.** No leak, no shortcut. The
proprioception both receive (desired + current EE position) is what a real robot reads off its own
joint encoders, and it is structurally required: the trajectory *is* `[action | obs]`, `cond[0]` is
hard-inpainted into it at index 0 every sampling step (`mix_visual_avoiding/models/helpers.py:159-163`),
and the DPCC projector enforces Euler dynamics on the `c_xy` slice. Remove it and there is no
projector.

**Q2 — is perception *load-bearing*?** **Aligning yes, avoiding no.** The obstacle geometry reaches
the avoiding controller by **two routes that bypass the camera entirely**:

1. **Hardcoded into the projector.** `config/avoiding-d3il-visual.py:122-136` declares the six
   obstacles as `sphere_outside` constraints, *"sourced verbatim from …avoiding_objects.py:68-82"*.
   The controller is told where they are.
2. **They never move.** `_reset_env` ignores both `random` and `context`
   (`d3il/…/gym_avoiding/envs/avoiding.py:257-262`); obstacles are six literal constants, every geom
   `static=True` (`…/objects/avoiding_objects.py:68-80`); the start is one fixed `init_end_eff_pos`
   (`:5`); the randomization block at `avoiding.py:382-388` is commented out. So the scene is a
   compile-time constant the network can memorise from training, image or no image.

Aligning has **neither** route: the box and target poses are re-sampled every episode
(`d3il/…/gym_aligning/envs/aligning.py:375`, `self.manager.start(random=random, context=context)`),
and the projector is *not* told them — the `obstacle_only_*` entries in
`config/visual_aligning_eval.yaml:265-280` are ablation-only geometries with `PLACEHOLDER` centres,
absent from the `combined_5` sets the main tables use.

| | image is used by the net? | image carries robot pose? | image carries **episode-varying task state**? | perception required? |
|---|---|---|---|---|
| visual **aligning** | ✅ | ✅ | ✅ **box + target pose** | ✅ **yes** |
| visual **avoiding** | ✅ | ✅ (richer than 4-D obs — it renders the full arm) | ❌ scene is constant | ❌ **no** |

> 🔴 **The consequence for this file.** The image is *used* — the encoder is trained end-to-end
> (`mf_freeze_vision_encoder` defaults `False`) and FiLM-conditions every U-Net block, so "the vision
> branch does nothing" would be false. But it is a **redundant path**, not a load-bearing one. Every
> result below is therefore a valid *engine* comparison **run with a camera attached**, and none of
> it can support a claim of the form *"the method works from images"*. **`SNAPSHOT_20260823_visual_aligning…`
> is where that claim lives; this file cannot corroborate or refute it at any sample size.**

### 0.3 Projectors

| class | variants | role |
|---|---|---|
| **Arm B — DPCC projector** | `dpcc-{r,c,t}` · `-tightened` (0.025 margin) · `post_processing` | ✅ reported; model-vs-model uses each model's best |
| **Arm C — HardFlow** | `hardflow_new-{r,c,t}[-tightened]` | ✅ reported separately (§5); **`mf` only** |
| **Arm A — reference** | `diffuser` (no projection) | ⚪ the no-projector control; also the clean per-NFE generation cost (§4) |
| **Study-only — excluded** | `gradient*` · `model_free*` · `dpcc-c-tightened-dt{0p25,0p5,2p0,4p0}` | ❌ not controllers |

### 0.4 Geometries

`TL` = `top-left-hard`, `TR` = `top-right-hard`, `BH` = `both-hard`. All three run for every
candidate below except where the table says otherwise.

---

## 1. Roster — and the engine that is missing

| C | tree | gen | engine | bone / cond | K | aw | seeds | `n_trials` | **n / geo** |
|---|---|---|---|---|---|---|---|---|---|
| **1** | `avoiding-d3il-visual-mix` | **Gen16** | **`mf`** (VisualMeanFlow) | U-Net FiLM v1 | **2** | **1** | 6 | **30** | **30** |
| **4** | `avoiding-d3il-visual` | Gen9 | **`fm`** (VisualFlowMatching) | U-Net FiLM v1 | 20 | 1 | 6 | 2 | **2** |
| 5 | `avoiding-d3il-visual` | Gen9 | `fm` | U-Net FiLM v2 | 20 | 1 | 6 | 2 | 2 |
| 3 | `avoiding-d3il-visual` | Gen9 | `fm` | U-Net, no FiLM ⚠️ | 20 | 1 | 6–10 | 2 | 10 (BH: 2) |
| **7** | `avoiding-d3il-visual` | Gen9 | **`dpcc`** (VisualGaussianDiffusion) | U-Net | 20 | **10** | 6–10 | 2 | **10** |
| 6 | `avoiding-d3il-visual` | Gen9 | `dpcc` | U-Net | 20 | 10 | 6 | 2 | 2 |
| — | — | — | **`af` (AlphaFlow)** | — | — | — | — | — | **none — never run** |
| 2, 173 | `…(unfull)`, `avoiding-d3il/plans/fm_visual_avoiding` | Gen9 | `fm` | — | 20 / 100 | 1 | 6 | 2 | ❌ quarantined (§7) |

⚠️ C3 is the `(old_no_mpc_traj)` tree — a superseded MPC path; kept only because it is the one `fm`
row with 5 seeds.

**The bones are matched.** C1 and C4 are both `_filmv1` U-Nets (`_mix_bone_keys()` emits `film_mode`
only on the U-Net branch — `config/avoiding-d3il-visual-mix.py:505-524`); C6/C7 are the same U-Net
without the FiLM knob. **No SiT/DiT anywhere on this env**, so unlike the state study there is no
backbone confound to unpick.

**`aw` is matched between `mf` and `fm` (both `aw1`) and mismatched against `dpcc` (`aw10`)** —
`config/avoiding-d3il-visual.py:151` vs `:186`. So the `mf`-vs-`fm` comparison is clean on the loss
weight and the `*`-vs-`dpcc` comparison is not. This is the reverse of the Gen16 DA's framing, which
compared `mf`/`aw1` against a state `aw10` target and correctly called `aw` its first confound.

---

## 2. 🔴 The sample-size problem, before any result

| model | worst-geometry `S&C` | **n / geo** | episodes total |
|---|---|---|---|
| `fm` K20 filmv1 | **1.00** | 2 | **6** |
| `fm` K20 filmv2 | **1.00** | 2 | **6** |
| `dpcc` K20, seed 6 only | **1.00** | 2 | **6** |
| `mf` K2, arm C | 0.87 | **30** | **90** |
| `mf` K2, arm B | 0.83 | **30** | **90** |
| `fm` K20 nofilm | 0.80 | 10 (BH 2) | 22 |
| `dpcc` K20, 5 seeds | **0.70** | 10 | 30 |

**The ranking is a perfect inverse of the sample size.** Every row that scores 1.00 has six episodes
behind it; the two best-sampled rows are the two lowest-scoring. Wilson 95 % on 2/2 is
**[0.34, 1.00]** — a cell that scored 1.00 here is consistent with a true rate of one in three.

The predecessor DA measured exactly this bias on the state DPCC baseline when it went 2 → 20 trials:
*"any `1.00` was inflated … any 'perfect' n=2 number should be read as '≥0.90 with ±0.10
resolution'"*, and mid-range cells moved by up to 0.35 **in both directions**. That was at 10
episodes/cell. Here the `fm` and `dpcc` cells have **two**.

The same DA's closing rule applies verbatim, with the sides swapped:

> **"Trial-count parity is now mandatory in any table: an n=2 method row against an n=20 baseline row
> is not a comparison."**

Here it is the *method* that is well-sampled and the *baselines* that are not. **§3 is reported for
completeness; it is not evidence.**

---

## 3. Head-to-head — each model at its own best projector

All rows below are the **same U-Net**, on the **same three geometries**. `s/ep` bold.

| model | projector | geo | n | S&C | 95 % CI | succ | viol | steps | s/step | s/ep |
|---|---|---|---|---|---|---|---|---|---|---|
| **`mf` K2 filmv1** | `dpcc-t-tightened` | TL | 30 | **1.00** | [0.89,1.00] | 1.00 | 0.00 | 55.6 | 0.031 | **1.72** |
| **`mf` K2 filmv1** | `dpcc-t-tightened` | TR | 30 | **0.83** | [0.66,0.93] | 0.83 | 0.00 | 63.7 | 0.032 | **2.04** |
| **`mf` K2 filmv1** | `dpcc-t-tightened` | BH | 30 | **1.00** | [0.89,1.00] | 1.00 | 0.00 | 51.9 | 0.040 | **2.09** |
| `mf` K2 filmv1 | `dpcc-c-tightened` | TL | 30 | 1.00 | [0.89,1.00] | 1.00 | 0.00 | 55.9 | 0.031 | 1.73 |
| `mf` K2 filmv1 | `dpcc-c-tightened` | TR | 30 | 0.77 | [0.59,0.88] | 0.77 | 0.00 | 62.4 | 0.033 | 2.04 |
| `mf` K2 filmv1 | `dpcc-c-tightened` | BH | 30 | 1.00 | [0.89,1.00] | 1.00 | 0.00 | 52.0 | 0.044 | 2.28 |
| **`fm` K20 filmv1** | `dpcc-c-tightened` | TL | **2** | 1.00 | [0.34,1.00] | 1.00 | 0.00 | 72.0 | 0.370 | **26.64** |
| **`fm` K20 filmv1** | `dpcc-c-tightened` | TR | **2** | 1.00 | [0.34,1.00] | 1.00 | 0.00 | 83.5 | 0.392 | **32.76** |
| **`fm` K20 filmv1** | `dpcc-c-tightened` | BH | **2** | 1.00 | [0.34,1.00] | 1.00 | 0.00 | 56.0 | 2.429 | **136.02** ⚠️ |
| **`dpcc` K20 (5 seeds)** | `dpcc-c-tightened` | TL | 10 | **0.70** | [0.40,0.89] | 1.00 | 9.50 | 70.1 | 0.736 | **58.79** |
| **`dpcc` K20 (5 seeds)** | `dpcc-c-tightened` | TR | 10 | 1.00 | [0.72,1.00] | 1.00 | 0.00 | 66.1 | 0.516 | **33.76** |
| **`dpcc` K20 (5 seeds)** | `dpcc-c-tightened` | BH | 10 | 1.00 | [0.72,1.00] | 1.00 | 0.00 | 65.7 | 0.566 | **37.13** |
| `dpcc` K20 (seed 6) | `dpcc-c-tightened` | TL | 2 | 1.00 | [0.34,1.00] | 1.00 | 0.00 | 52.5 | 0.547 | 28.71 |
| `dpcc` K20 (seed 6) | `dpcc-c-tightened` | TR | 2 | 1.00 | [0.34,1.00] | 1.00 | 0.00 | 77.5 | 0.451 | 34.95 |
| `dpcc` K20 (seed 6) | `dpcc-c-tightened` | BH | 2 | 1.00 | [0.34,1.00] | 1.00 | 0.00 | 54.5 | 0.476 | 25.92 |
| `fm` K20 filmv2 | `dpcc-c-tightened` | TL | 2 | 1.00 | [0.34,1.00] | 1.00 | 0.00 | 56.5 | 2.247 | 126.97 ⚠️ |
| `fm` K20 filmv2 | `dpcc-c-tightened` | TR | 2 | 1.00 | [0.34,1.00] | 1.00 | 0.00 | 66.0 | 4.169 | 275.14 ⚠️ |
| `fm` K20 filmv2 | `dpcc-c-tightened` | BH | 2 | 1.00 | [0.34,1.00] | 1.00 | 0.00 | 58.5 | 5.721 | 334.66 ⚠️ |
| `fm` K20 nofilm | `dpcc-c-tightened` | TL | 10 | 1.00 | [0.72,1.00] | 1.00 | 0.00 | 62.6 | 5.753 | 326.28 ⚠️ |
| `fm` K20 nofilm | `dpcc-c-tightened` | TR | 10 | 0.80 | [0.49,0.94] | 1.00 | 0.50 | 70.2 | 10.820 | 767.95 ⚠️ |
| `fm` K20 nofilm | `dpcc-c-tightened` | BH | 2 | 1.00 | [0.34,1.00] | 1.00 | 0.00 | 53.5 | 22.802 | 1219.90 ⚠️ |

⚠️ = wall-clock quarantined by §4.1 (`fm` filmv2 / nofilm entirely; `fm` filmv1's BH cell only).

### 3.1 Paired at seed 6 — the only seed all three share

| | S&C TL / TR / BH | steps TL / TR / BH | **s/ep TL / TR / BH** |
|---|---|---|---|
| `mf` K2 `dpcc-t-tightened` (n = 30) | 1.00 / **0.83** / 1.00 | **55.6 / 63.7 / 51.9** | **1.72 / 2.04 / 2.09** |
| `fm` K20 filmv1 `dpcc-c-tightened` (n = 2) | 1.00 / 1.00 / 1.00 | 72.0 / 83.5 / 56.0 | 26.64 / 32.76 / 136.02 |
| `dpcc` K20 `dpcc-c-tightened` (n = 2) | 1.00 / 1.00 / 1.00 | 52.5 / 77.5 / 54.5 | 28.49 / 34.63 / 25.70 |
| **`mf` speed-up vs `fm`** | | | **15.5× / 16.1× / 65.1×** |
| **`mf` speed-up vs `dpcc`** | | | **16.6× / 17.0× / 12.3×** |

**Read this as: `mf` is 12–17× cheaper per episode and uses ~10–20 fewer control steps, and it is the
only one of the three whose success rate is measured at all.** Its single deficit is `top-right-hard`,
and §5 shows that deficit is a goal-reaching failure, not a constraint failure.

**`fm` does not beat `dpcc` on this env.** At seed 6 both sit at 1.00/1.00/1.00 (n = 2, i.e.
indistinguishable by construction), `fm` is 1.06–1.07× cheaper on TL/TR and **5.3× more expensive on
BH**. There is no visual analogue of the state-side "FM Pareto-dominates DPCC" result — because the
K=2 operating point that produced it was never run on visual avoiding.

---

## 4. Where the time goes — the 16× is K, and only K

Splitting `dpcc-c-tightened`'s `s/step` into generation (= the `diffuser` arm, no projection) and
projection overhead, and normalising each by its own budget (K network calls; `K/2` NLP solves per
control step at `T = 0.5`):

| model | K | gen/NFE (ms) TL / TR / BH | proj/solve (ms) TL / TR / BH |
|---|---|---|---|
| **`mf`** K2 | 2 | **11.9 / 12.4 / 12.0** | **7.2 / 7.9 / 19.7** |
| **`fm`** K20 filmv1 | 20 | **11.9 / 11.8 / 11.8** | 13.2 / 15.7 / **219.3** ⚠️ |
| **`dpcc`** K20 (seed 6) | 20 | **12.2 / 12.3 / 12.2** | 30.2 / 20.5 / 23.2 |
| `dpcc` K20 (5 seeds) | 20 | 12.2 / 12.1 / 12.1 | 49.3 / 27.4 / 32.3 |
| `fm` K20 filmv2 ⚠️ | 20 | 67.6 / 68.1 / 67.9 | 89.6 / 280.7 / 436.2 |
| `fm` K20 nofilm ⚠️ | 20 | 58.2 / 58.1 / 58.6 | 458.9 / 965.8 / 2163.0 |

**Generation is 11.8–12.4 ms per network call for `mf`, `fm` and `dpcc` alike** — as it must be for
three identically-sized visual U-Nets. So the 10× generation gap between the rows is exactly
K = 2 vs K = 20, nothing else.

**Per-solve projection cost is *also* a function of K, not only of the model.** The state DA measured
FM's own per-solve cost climbing `7 → 15 → 22 ms` across K = 2 / 5 / 20 as the projector's warm start
degrades. `mf`'s **7.2 ms at K = 2 is that same number**; `fm`'s 13.2–15.7 ms at K = 20 is that same
curve's other end. **There is no measurable model effect in this data** — only the K effect, seen
twice.

Corrected decomposition of the observed ~16×:

| factor | ratio | attribution |
|---|---|---|
| fewer NFE (K 20 → 2) | ~10× | budget |
| fewer NLP solves (10 → 1 per control step) | ~10× | budget (same K reduction, projection side) |
| cheaper per solve at **matched** K | **not measured** | ⚠️ no visual `fm`/`dpcc` run exists at K < 20 |

Both budget factors are the *same* K reduction, and together with step count they land on the
observed 34.6 → 2.0 s/ep. **The residual model term — the one the whole `mf > fm` claim would rest
on — has no data.**

### 4.1 Timing quarantine

`fm` filmv2 and `fm` nofilm generate at **58–68 ms/NFE** against 12 ms/NFE for the same U-Net at the
same K. That is a 5× discrepancy that cannot be a model property; those runs are from 2026-06-20 …
06-30 and are node-contaminated. Their `s/step` and `s/ep` are unusable. Their `S&C` and `steps` are
unaffected and are kept.

`fm` filmv1's **BH** cell is a separate anomaly: generation is clean (11.8 ms/NFE, identical to its
own TL/TR) but projection costs **219 ms/solve against its own 13–16 ms on TL/TR** — a 14× blow-up
confined to one geometry, on the model whose plans have to satisfy two halfspaces at once. `dpcc`
shows no such blow-up on BH (23 ms vs 20–30 ms elsewhere). This is plausibly real — `fm`'s raw plans
land far from the feasible set under two constraints and the NLP grinds — but it is **one cell of two
episodes**, so it is flagged, not claimed.

---

## 4b. The vision branch in the U-Net — where it enters, and what it costs the state model

§0.2.1 argued from the env that the image is redundant on avoiding. This section measures it.

### 4b.1 How the image enters

One `bp_cam` frame `(3, 96, 96)` → `MultiImageObsEncoder` (ResNet-18 per camera, `share_rgb_model:
False`) → 64-D latent, `LATENT_DIM = N_CAMERAS × RGB_OUTPUT_SIZE = 1 × 64`
(`mix_visual_avoiding/models/visual_spec.py:56-64`, **derived, never hand-set**). The latent is
mean-pooled over the image window and FiLM-conditions every U-Net block. `T_win = 1`
(`window_size: 1`, `obs_seq_len: 1`, `config/avoiding-d3il-visual.py:215-216`), so there is **no
temporal signal in the image either** — a single frame, pooled over itself.

Crucially the state never goes *through* the encoder. It enters on a separate, exact path:
`cond[0]` → `apply_conditioning` → trajectory index 0. So the two channels are additive, and the
FiLM branch is the *only* thing the state model does not have.

> 🔴 **The two gens call the encoder in different places, and it is not cosmetic.**
>
> | gen | call site | encoder runs |
> |---|---|---|
> | **Gen9** (`fm_visual_avoiding`, `diffuser_visual_avoiding`) | inside `forward()` — `visual_unet.py:117` / `:127` | **once per network call → K× per plan** |
> | **Gen16** (`mix_visual_avoiding`) | pre-encoded upstream into `cond['visual_latent']`; `resolve_visual_cond` short-circuits (`visual_unet_twotime.py:168-172`, `visual_mf_diffusion.py:87-91`) | **once per plan**, reused across all K steps |
>
> At K = 2 that saves one encode and is worth little. At K = 20 it would save 19. **Any future
> Gen9-vs-Gen16 wall-clock comparison must account for this**; it is a second budget effect stacked
> on top of K, not a model property. (Gen16's motivation was the JVP, not speed — re-running ResNets
> under forward-mode AD is wasteful and possibly unimplemented — but the speed consequence is real.)

### 4b.2 What the encoder costs — the clean measurement

**`dpcc` K20 `aw10`, `dpcc-c-tightened`, seeds 6–10, `n_trials = 2` on both sides.** Same engine,
same K, same `aw`, same projector, same seeds, same trial count. **The only difference is
`if_vision`.** This is the one fully-controlled visual-vs-state pair in the batch.

| | S&C TL / TR / BH | steps TL / TR / BH | s/step TL / TR / BH | s/ep TL / TR / BH |
|---|---|---|---|---|
| **state** (C16) | **1.00 / 1.00 / 1.00** | 68.4 / 77.2 / 64.8 | 0.565 / 0.505 / 0.590 | 38.9 / 38.4 / 38.3 |
| **visual** (C7) | **0.70** / 1.00 / 1.00 | 70.1 / 66.1 / 65.7 | 0.736 / 0.516 / 0.566 | **58.8** / 33.8 / 37.1 |

**Adding the vision branch buys nothing and costs on two axes:**

1. **Success: −0.30 on `top-left-hard`**, tie on the other two. ⚠️ n = 10 per cell, and the deficit is
   entirely seeds 7 (0.0) and 8 (0.5) — so the *size* is not resolved, only the direction, and the
   state row's own n = 20 value is 1.00 / 0.95 / 1.00 (C11). Do not quote 0.30 as a measured penalty.
2. **Generation: +35 % per network call, flat across geometries.** From the unprojected `diffuser`
   arm, which isolates generation:

   | | gen/NFE TL / TR / BH |
   |---|---|
   | state `dpcc` K20 | **8.94 / 8.93 / 9.04 ms** |
   | visual `dpcc` K20 | **12.19 / 12.11 / 12.14 ms** |
   | Δ = the encoder | **+3.25 / +3.18 / +3.10 ms** |

   ~3.2 ms is one ResNet-18 pass on a 96×96 frame at MPC fan 4. It is **paid K times in Gen9**, so a
   K = 20 visual plan spends ~64 ms/step purely re-encoding an image that has not changed.
3. **Steps: a wash** (70.1/66.1/65.7 vs 68.4/77.2/64.8). The encoder does not shorten the path.

**This is the expected result if the image is redundant, and it is what §0.2.1 predicts from the env
code.** The encoder consumes capacity and compute to re-derive a robot position the model is already
handed exactly, plus a constant obstacle map the projector is also handed explicitly.

### 4b.3 The same comparison for `fm` and `mf` — confounded, listed for completeness

Neither pair is controlled the way §4b.2 is; both are shown so the file does not appear to select the
one unflattering comparison.

| engine | arm | S&C TL / TR / BH | steps TL / TR / BH | s/ep TL / TR / BH | why confounded |
|---|---|---|---|---|---|
| `fm` K20 | state, `aw10`, n=100 (C163) | 0.91 / 1.00 / 1.00 | 69.1 / 69.2 / 58.5 | 31.4 / 26.7 / 34.6 | — |
| `fm` K20 | visual, `aw1`, n=2 (C4) | 1.00 / 1.00 / 1.00 | 72.0 / 83.5 / 56.0 | 26.6 / 32.8 / 136.0 | **`aw1` vs `aw10`**; n = 2 vs 100 |
| `mf` K2 | state, `aw10`, `A0.5 B1`, n=100 (C146) | 0.99 / 0.93 / **0.86** | 95.0 / 100.2 / 98.8 | 2.56 / 2.66 / 2.65 | — |
| `mf` K2 | visual, `aw1`, `a1.5 b1.0`, n=30 (C1) | 1.00 / **0.83** / 1.00 | **55.6 / 63.7 / 51.9** | 1.72 / 2.04 / 2.09 | **`aw1` vs `aw10`**, **different `a`/`b`**, 1 seed |

⚠️ **The `mf` row is the one place vision does *not* look like a pure tax:** the visual model reaches
the goal in **~40 % fewer control steps** on every geometry (52–64 vs 95–100), wins `both-hard`
(1.00 vs 0.86) and loses `top-right` (0.83 vs 0.93). That is a large, interesting difference — but it
sits on top of **two** training-config changes (`aw`, and the `a`/`b` time-schedule), so it is **not
attributable to vision**. Reproducing it at matched `aw10` is gap 5 in §8. Do not cite it as a vision
effect.

---

## 5. `mf`'s failures are task failures; the untightened projectors are the real defect

In all nine `mf` `*-tightened` cells `viol = 0.00` and **`S&C` is exactly equal to `succ`** (see §3).
Where `mf` loses on `top-right-hard`, the constraint layer is doing its job and the *generative model*
is failing to reach the goal — consistent with `mf`'s unguided `diffuser` arm scoring `succ = 0.73`
on TR against 1.00 on TL/BH. The hole is upstream of any projector.

**Untightened arm B collapses for every model on at least one geometry:**

| model | untightened `dpcc-c` worst cell | S&C | succ | viol |
|---|---|---|---|---|
| `mf` K2 | TL | **0.00** | 1.00 | 11.6 |
| `fm` K20 filmv1 | TL (0.00 on all three) | **0.00** | 1.00 | 53.0 |
| `dpcc` K20 (5 seeds) | BH | **0.10** | 1.00 | 8.7 |

`S&C = 0.00` at `succ = 1.00` means the policy reaches the goal every time and clips an obstacle every
time. **Tightening is load-bearing on visual avoiding, not an optimisation** — for all three engines.

### 5.1 Arm C (`mf` only)

`hardflow_new-c-tightened` is `mf`'s best row on the failing geometry: **0.87 on TR vs 0.77
(`dpcc-c-tightened`) / 0.83 (`dpcc-t-tightened`)**, tying at 1.00 on TL and BH, at **3.5×** the cost
(7.0–7.6 vs 2.0–2.1 s/ep). Both arms ran at `hf_act_threshold = 0.5` and `hf_batch_size = 4`
(verified in the CSV), so this is a cost-for-success trade, **not** the claim the benchmark hierarchy
asks for — HardFlow has to beat the DPCC projector *at a lower projection threshold*. A threshold
sweep is still missing. No `fm` or `dpcc` visual candidate has arm C at all.

---

## 6. Verdict on `af ~ mf > fm > dpcc`

| claim | visual **aligning** (where the premise comes from) | **state** avoiding | **visual avoiding** (this file) |
|---|---|---|---|
| `af ~ mf` | ✅ both beat K20 engines | ⚠️ `af` SiT-confounded | ❌ **`af` has no run** |
| `mf > fm` | ✅ yes | ❌ **no — `mf` never beats naive FM at any K** | ❌ **no** — worst-geo 0.83/0.87 vs 1.00 (n=2) |
| `fm > dpcc` | ✅ at each's own K | ✅ at own K / ⚖️ a wash at matched NFE | ⚖️ **tie at seed 6**, and 5.3× worse on BH |
| `mf` cheaper than `fm`/`dpcc` | ✅ | ✅ | ✅ **12–17×** — but attributable entirely to K (§4) |

**The premise is a visual-*aligning* result, and it is already contradicted on state avoiding.**
[`SNAPSHOT_20260823_visual_aligning_env_status.md`](SNAPSHOT_20260823_visual_aligning_env_status.md)
§3 answers "does `mf`/`af` at K=2 beat `fm`/`diffusion` at K=20?" with **yes, on every axis**. The
state-avoiding DA's §3/§6 answer the same question with **"MeanFlow must beat naive FM to justify
itself. It does not, at any K, on any halfspace"** — naive FM is that study's winner. So the ordering
was never task-invariant, and visual avoiding is a third data point rather than a tie-break.

**What this file can actually assert:**

0. 🔴 **It is not a perception result.** The obs is clean — no cheat — but the obstacles are hardcoded
   into the projector and never move, so the camera is a redundant path (§0.2.1). Controlled
   visual-vs-state on the same engine/K/aw/trials shows the encoder costing **+35 % per network call
   for no gain** (§4b.2). Any "works from images" claim belongs to visual *aligning*, not here.
1. ✅ Gen16's visual `mf` pipeline runs end-to-end on avoiding, arms A/B/C, and is the only
   visual-avoiding candidate evaluated at a usable sample size.
2. ✅ It is **12–17× cheaper per episode** and uses **~10–20 fewer control steps** than both
   K=20 visual engines, on all three geometries.
3. ✅ Its constraint layer is exact — `viol = 0.00` in every tightened cell.
4. ⚖️ It is **non-dominated, not better**: `top-right-hard` 0.83/0.87 against opponents that report
   1.00 on six episodes. Per `pareto-definition-of-good`, this is a **trade-off**.
5. ❌ **`af ~ mf > fm > dpcc` is not supported, not refuted, and not testable from this batch.**

---

## 7. Quarantine

| C | reason |
|---|---|
| 2 | `fm_visual_avoiding(unfull)` — 1–2 cells, geometries incomplete, several variants with a single cell. |
| 173 | `H8_K100_…VisualFlowMatching` under the **state** tree; one snapshot, no `S&C`/`avg_time` at all. |
| 3, 5 | Kept for `S&C`/`steps`; **all wall-clock quarantined** by §4.1. C3 additionally is the superseded `(old_no_mpc_traj)` path. |
| `post_processing*` on every n=2 cell | The predecessor DA showed `post_processing ≡ dpcc-r` on the n=2 DPCC baseline at K=20, where the identity is impossible — a genuine defect in n=2 data. Same shape here; not reported. |

---

## 8. Gaps — in the order that unblocks the question

1. 🔴 **`fm` visual avoiding at K = 2, `n_trials = 30`, seeds 6–10.** *The* missing run. It supplies
   the matched-K, matched-sample opponent that turns §3 from an artefact into a result, and it is the
   only way to separate "`mf` is better" from "K = 2 is cheaper". Cheap — K = 2 is ~1/10 of a K = 20 slot.
2. 🔴 **Re-run the Gen9 visual tree at `n_trials = 30`** (`config/visual_avoiding_eval.yaml:27`,
   one-line change) for `fm` filmv1 and `dpcc`. Without this, every `fm`/`dpcc` visual number in the
   repo is 6 episodes.
3. **`mf` seeds 7–10.** One seed measures no seed variance; the state study's per-seed spread
   (DPCC's TL 0.70 here is *entirely* seeds 7 and 8) is large enough to swallow the 0.83-vs-1.00 gap.
4. **`af` visual avoiding — train and evaluate.** Still zero data; the `af ~ mf` half of the question
   is unanswerable until this exists. Must be a **U-Net** bone to stay architecture-matched (the state
   `af` rows are SiT and are confounded for exactly this reason).
5. **`mf` at `aw10`.** `mf`/`fm` are `aw1`, `dpcc` is `aw10` (§1). Retraining `mf` at `aw10` removes
   the only training-config mismatch against the `dpcc` baseline and directly targets TR.
6. **Diagnose `top-right-hard`.** The only geometry where anything fails, failures are goal-reaching
   not violations (§5), and `mf`'s *unguided* `succ` is 0.73 there vs 1.00 elsewhere.
7. **`hf_act_threshold` sweep on arm C**, so §5.1 can answer HardFlow-vs-DPCC properly.
8. **Hoist Gen9's encoder call out of `forward()`** (`fm_visual_avoiding/models/visual_unet.py:127`,
   `diffuser_visual_avoiding/models/visual_unet.py:117`) to match Gen16's encode-once path
   (§4b.1). At K = 20 this is ~19 redundant ResNet passes per plan, ≈64 ms/control-step — a large
   share of the visual K20 rows' wall clock, and it makes every Gen9-vs-Gen16 timing comparison
   carry a second budget effect on top of K.
9. 🔴 **If a perception result on avoiding is actually wanted, the task has to change — and it is
   three coupled changes, not one.** Randomise the layout per episode **in training and eval**
   (re-collect demos; `avoiding.py:382-388` holds the commented-out block), feed the true layout to
   the projector per episode instead of the hardcoded `_AVOIDING_OBSTACLES`, and upgrade the state
   baseline to 16-D obs so the comparison stays fair. **Full argument, and the same check applied to
   the UAV env before its visual arm is built, in §8b.** Scope this before spending compute on gaps
   1–3 if perception — rather than engine ranking — is the goal.

---

## 8b. Design note — when does a visual arm mean anything? (avoiding now, UAV next)

This file's §0.2.1 and §4b are a specific instance of a general rule that is worth stating once,
because it decides whether a visual arm is worth building **before** any compute is spent on it.

### 8b.1 The criterion

A vision branch can only earn its cost if there is something the model **must** see. That requires
**all three** of the following. Break any one and the encoder is a redundant path.

| # | condition | avoiding today | aligning today |
|---|---|---|---|
| **1** | The task-relevant scene state **varies per episode** | ❌ `_reset_env` ignores `random`/`context`; obstacles are literal constants | ✅ `manager.start(random, context)` re-samples box + target |
| **2** | It is **absent from the obs vector** | ✅ 4-D robot only | ✅ 6-D robot only (state sibling has it at 20-D) |
| **3** | It is **not handed to the controller by another route** | ❌ `_AVOIDING_OBSTACLES` hardcodes all six into the projector | ✅ `obstacle_only_*` are `PLACEHOLDER` ablations, not in `combined_5` |

Avoiding satisfies only #2 — which is why it is **clean but uninformative**: no cheat, no perception.
**Aligning satisfies all three, and is therefore the only env in this repo where a "works from
images" claim is currently supportable.** It is also the existence proof that the pipeline can do
it, so the gap on avoiding is a *task-design* gap, not a code gap.

### 8b.2 What it would take to make visual avoiding mean something

Randomising the obstacles is necessary but **not sufficient** — three things move together:

1. **Randomise the layout per episode, in training *and* eval.** Training so the encoder ever learns
   to read the field; eval so it is tested. The env already has the hook: the randomisation block at
   `d3il/…/gym_avoiding/envs/avoiding.py:382-388` is written and commented out. **Training data would
   have to be re-collected** — the current demos are all from the one fixed layout, so this is not a
   config flip.
2. **Feed the true layout to the projector per episode instead of hardcoding it.**
   `config/avoiding-d3il-visual.py:122-136` is a static list; it would become a per-episode input.
   ⚠️ **This is not a cheat to remove** — DPCC *assumes* the constraints are known; that assumption is
   the method. Perception has to be load-bearing for the **generative policy**, not for the projector.
3. **Upgrade the state baseline to receive obstacle coordinates in obs** (4-D → 16-D:
   `[des_xy(2), c_xy(2), obs_xy(2)×6]`). Without this, state-vs-visual becomes trivially unfair in the
   *opposite* direction — the state arm would have no way to know where anything is.
   🔴 This is exactly **"Option B"**, which the Gen9 plan explicitly **rejected**
   (`logs_in_develop/Gen9/…/PLAN_SINGLE_CAMERA_VISUAL_AVOIDING.md:44`) on the grounds that *"positions
   are constants… adds redundant capacity; not recommended."* **That reasoning was correct then and
   inverts under randomisation:** the moment the layout varies, Option B stops being redundant and
   becomes the only fair state baseline. The audit that produced Option C should be re-opened, not
   worked around.

⚠️ **A small finite scene set is a weak version of this.** Drawing from 3–5 fixed layouts lets the
network classify the scene from a handful of pixels — technically perception, but a near-trivial
instance of it. Continuous randomisation of the obstacle positions is the stronger design and costs
no more to collect.

### 8b.3 The same defect is pre-loaded into the UAV env

**UAV currently has no visual arm at all** — `mix_uav/models/` contains no `visual_*.py`, no
`MultiImageObsEncoder`, no camera key; the only `visual` strings in `config/uav.py` are comments
citing the aligning config's naming conventions. So this is a warning about a build that has not
happened yet, not a finding about a run.

Checked against the criterion, **UAV would fail #1 exactly the way avoiding does, for a different
reason.** Four scenes exist — `corridor`, `pillars`, `s_curve`, `empty` — but per
[`SNAPSHOT_20260825_uav_mix_env_status_PILOT.md`](SNAPSHOT_20260825_uav_mix_env_status_PILOT.md) the
entire deployable core is **13 candidates × 2 943 rollouts, all `corridor`, one seed**; `pillars` and
`s_curve` ran n = 3 and are quarantined. **One scene per run ⇒ the geometry is constant within the
experiment ⇒ a camera would have nothing episode-varying to report**, exactly as here.

The fix is cheaper for UAV than for avoiding, and should be settled **before** an encoder is written:
**mix the scenes inside one train + eval run** rather than one scene per run. The scenes already
exist and the XMLs already differ; only the run protocol is single-scene. Do that and UAV satisfies
#1 and #2 — leaving #3 as the open design question (the corridor walls are currently projector
constraints, so the same "known-constraints vs perceived-constraints" split as §8b.2 step 2 applies).

⚠️ One thing a camera would **not** fix on UAV: the seeded `homotopy` (which side of the corridor to
fly, pool 4 L / 3 C / 3 R) that the policy is *"never told"* (that file's §0.1 `match` row). That is an
intended **mode**, not an observable scene property — no sensor can read it off the arena. Do not
motivate a UAV camera with it.

### 8b.4 What a fixed-scene visual run is still good for

Not nothing, and this is how the Gen16 result should be cited:

- ✅ **Pipeline validation.** Gen16 proved the visual `mf` chain runs end-to-end on a second task,
  arms A/B/C, with NFE instrumentation and candidate-fan parity (§1). That is a real engineering
  result and it does not need the image to be informative.
- ✅ **A cost measurement.** §4b.2 is only possible *because* the image is redundant: it isolates the
  encoder's price with the benefit pinned at zero — **+35 % per network call, +3.2 ms/NFE.** That
  number transfers to any future visual arm as the floor it must beat.
- ❌ **Not** evidence about perception, engine ranking under perception, or sim-to-real transfer.

**Bottom line: build the visual arm where the scene varies. On the current avoiding and UAV
configurations it cannot pay for itself, and §4b.2 measures what it costs instead.**

---

## 9. One-line summary

**Visual avoiding is a clean run but not a vision benchmark — the obs hides the obstacles from the
policy (no cheat), yet the projector is handed them and they never move, so the encoder costs +35 %
per network call and buys nothing. Within that: `af` does not exist, `mf` does not beat `fm` on the
success gate, `fm` does not beat `dpcc` at all, and `mf`'s 12–17× wall-clock win is entirely
attributable to K = 2 vs K = 20 — and none of it is decidable anyway, because every `fm` and `dpcc`
visual cell is two episodes while the `mf` cell is thirty. Run `fm` visual at K = 2 with
`n_trials = 30` to settle the engine question; randomise the obstacles to make it a vision question.**
