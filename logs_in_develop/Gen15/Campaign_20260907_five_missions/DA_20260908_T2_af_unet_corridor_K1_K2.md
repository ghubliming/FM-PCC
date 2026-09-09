# DA — Mission 2: af_unet on `corridor`, K=1 / K=2 (jobs 25494 / 25495 / 25498)

*Gen15 · campaign `Campaign_20260907_five_missions` · 2026-09-08.
Source batch: **`temp/0809/batch_uav_20260908_153947`** (`DA_UAV_v1`, generated 2026-09-08 15:40:29,
1263 units loaded, 0 failed). Candidates **C29** (af K=1) and **C30** (af K=2), with **C44** (mf K=2)
and **C38** (fm K=2) as the matched comparators.*

## 0. TL;DR — three answers

1. **`corridor` is saturated for both U-Net engines.** S&C = 1.000 in **all 20** af cells (10 variants
   × 2 K) and **all 10** mf cells. The scene has almost no dynamic range left, so it cannot rank
   af against mf. §2
2. **af_unet ≡ mf_unet at K=2 — a dead tie, not a win.** Head-to-head on `steps_to_goal`: af leads on
   5 of 10 variants, mf on the other 5, mean |Δ| = **0.83 steps** against per-cell σ of 5–14.
   🔴 **The claim-ladder step `af > mf` is NOT supported on `corridor`.** §3
3. **The one thing corridor *does* separate is `fm`.** It is the only engine that breaks S&C — 0.800
   on all three `temporal_consistency` rows — and it carries **3–6× the step-count dispersion** of
   af/mf on every variant. `af, mf > fm` on reliability holds here, weakly. §4

Secondary: doubling the budget K=1→2 buys af **nothing** (§5); the projector trades tracking
fidelity for path length monotonically and identically in all three engines (§6).

---

## 1. Provenance and gates

| | |
|---|---|
| jobs | train **25494** (Sep 7 04:22→07:15), eval **25495** (K=1), **25498** (K=2) |
| model | `AlphaFlowODE_9D_as1_ae0.2_bbunet` — **U-Net, 3,969,222 params (3.97 M)**, α floored at 0.2 |
| eval tag | `Eaf_K{1,2}_mpc4_pid_stopgo_T0.5_EPlatest_u7hg` |
| scene / geo | `corridor`, `corridor_hg_bounds+dynamics+geo_bounds+halfspace+obstacles` (bounds, hs=2, obs=4) |
| seed / trials | seed **6** only · **n = 10** rollouts per cell · `mpc_batch=4` · `controller=pid_stopgo` |
| variants | **10** (`diffuser`, `dpcc-{r,c,t}`, `×-tightened`, `×-geo_free`) |

**Gates green** on all four candidates: `n_cb_tripped = 0`, `cb_sentinel = 0`, `timing_missing = 0`,
`hf_degenerate = 0`, `n_rollouts = 10`, `n_diagnostics_json = 10`, `source = npz`.
(`npz_complete` is blank for *every* candidate in this batch, so it carries no signal either way.)

### 1.1 What is comparable, and what is not

| K | af | mf | fm | verdict |
|---|---|---|---|---|
| **1** | C29 ✅ | — (C41 is pre-`u7hg`) | — (C35 is pre-`u7hg`) | **af only — nothing to rank against** |
| **2** | C30 ✅ | **C44** ✅ | **C38** ✅ | ✅ **genuine three-way at matched K** |

Only `u7hg` candidates are admissible: the U7 honest-geometry change altered the constraint set, so
the untagged `corridor` rows (C34–C37, C39–C43, C45) and the af **SiT** rows (C31–C33, `ae0`,
Aug-23) are **not** comparable and are excluded throughout.

🔴 **Two absences that bound every claim below.** (a) **No HardFlow row exists** — at K=1 and K=2 with
A=0.5 the arm is degenerate (`n_genuine=0`) and was correctly blocked, so mission 2 says *nothing*
about HardFlow. (b) **No `engine=diffusion` arm exists on `corridor` at all**, so the DPCC-diffusion
Target required by the benchmark hierarchy cannot be placed here. This DA orders af/mf/fm among
themselves and no further.

---

## 2. The scene is saturated

`n_success_and_constraints = 1.000` with σ = 0.000 in **30 of the 40** engine × variant cells at
K=2, including **every** af and mf cell, and in **all 20** af cells across both K.

At n = 10 a clean 1.000 has a Wilson 95 % lower bound of ≈ 0.72 — so "1.000" here means *no failure
in ten*, not *proven perfect*. With the primary axis pinned at the ceiling for two of the three
engines, all remaining separation must come from the secondary axes (`steps_to_goal`, `track_err`,
time), and those are the weakest evidence in the hierarchy.

---

## 3. af vs mf at K=2 — a tie

`steps_to_goal`, mean ± σ (n = 10 unless noted):

| variant | **af** K=2 | **mf** K=2 | Δ (af−mf) |
|---|---|---|---|
| `diffuser` | 270.9 ± 5.04 | 271.7 ± 4.47 | −0.8 |
| `dpcc-r` | 260.8 ± 12.51 | 261.1 ± 13.67 | −0.3 |
| `dpcc-c` | 265.4 ± 7.60 | 265.3 ± 8.93 | +0.1 |
| `dpcc-t` | 259.6 ± 6.57 | 261.8 ± 7.80 | −2.2 |
| `dpcc-r-tightened` | 249.7 ± 12.88 | 248.7 ± 13.72 | +1.0 |
| `dpcc-c-tightened` | 251.1 ± 11.51 | 251.0 ± 10.95 | +0.1 |
| `dpcc-t-tightened` | 249.2 ± 11.78 | 248.5 ± 10.90 | +0.7 |
| `dpcc-r-geo_free` | 264.9 ± 13.88 | 266.5 ± 14.23 | −1.6 |
| `dpcc-c-geo_free` | 268.5 ± 6.84 | 269.5 ± 7.38 | −1.0 |
| `dpcc-t-geo_free` | 262.8 ± 7.44 | 262.3 ± 8.73 | +0.5 |

**af leads 5/10, mf leads 5/10; mean |Δ| = 0.83 steps.** Every Δ is an order of magnitude inside its
own σ. S&C is 1.000 for both everywhere, and `avg_time_ms` matches within 3 % on every row
(e.g. `dpcc-r` 66.93 vs 64.92 ms; `dpcc-c-tightened` 169.88 vs 184.73 ms — the only row exceeding 5 %,
and it favours af).

Per the Pareto rule this is **neither a win nor a trade-off — it is indistinguishable**. On this
scene the α-Flow objective buys nothing over MeanFlow at equal budget and equal backbone.

---

## 4. fm is the one engine corridor can break

**S&C = 0.800 on `dpcc-t`, `dpcc-t-tightened`, `dpcc-t-geo_free`** — all three `temporal_consistency`
selector rows, and only those. `phys_safe = 1.000` on every fm row, so the two lost rollouts per row
are **goal-reach failures, not crashes or constraint violations**. Note the steps means on those rows
are computed over **n = 8 survivors**, i.e. survivorship-biased and not directly comparable.

Dispersion is the sharper signal, and it is uniform:

| variant | af σ | mf σ | **fm σ** | ratio fm/af |
|---|---|---|---|---|
| `dpcc-r` | 12.51 | 13.67 | **44.44** | 3.6× |
| `dpcc-c` | 7.60 | 8.93 | **30.54** | 4.0× |
| `dpcc-r-tightened` | 12.88 | 13.72 | **48.14** | 3.7× |
| `dpcc-r-geo_free` | 13.88 | 14.23 | **51.33** | 3.7× |
| `dpcc-c-geo_free` | 6.84 | 7.38 | **32.41** | 4.7× |

`fm`'s `dpcc-r` ranges 236–365 steps; af's ranges 250–293. Both U-Net engines are markedly more
*consistent* than naive FM at matched K — which is the ladder step `mf, af > fm`, supported here but
on a single seed and near the ceiling.

fm is not uniformly worse: its `track_err` is the **best** of the three (0.413–0.523 vs af
0.504–0.562), and its projection is cheaper (`proj_ms` ≈ 31 vs ≈ 47 on `dpcc-r`). It reaches the goal
less often and less predictably, but tracks the reference more tightly when it does.

---

## 5. K=1 → K=2 buys af nothing

Averaged over the 10 matched variants:

| | steps_to_goal | track_err | fm_ms | S&C |
|---|---|---|---|---|
| af **K=1** | **255.53** | 0.5356 | **9.49** | 1.000 (all) |
| af **K=2** | 260.29 | **0.5257** | 18.24 | 1.000 (all) |
| Δ | **+4.8 (worse)** | −0.010 (better) | **+8.75 (+92 %)** | — |

Doubling the NFE costs 92 % more network time to gain 1.9 % tracking and lose 1.9 % path length —
both changes well inside σ. **On `corridor` the budget axis is flat for af.** This is consistent in
direction with the Sep-7 `pillars` finding that raw-plan success *falls* with K
(`../DA/DA_20260907_af_unet_uav_s_curve_pillars_K_sweep.md`), though here it is a wash rather than a
regression.

---

## 6. The projector's mechanism — monotone, and engine-independent

Ordering by projection strength (af K=2 shown; af K=1, mf K=2 and fm K=2 all reproduce it):

| projector | steps_to_goal | track_err | proj_ms |
|---|---|---|---|
| `diffuser` (off) | 270.9 — **most** | 0.504 — **best** | 0.00 |
| `-geo_free` | 262.8 – 268.5 | 0.509 – 0.518 | 11.2 |
| plain `dpcc` | 259.6 – 265.4 | 0.516 – 0.526 | 44.7 – 48.7 |
| `-tightened` | 249.2 – 251.1 — **fewest** | 0.545 – 0.549 — **worst** | 151.5 – 154.7 |

**Tightening the constraint set makes the drone cut the corridor corners**: the path shortens
monotonically and the deviation from the reference grows monotonically, in lockstep, across every
engine. That is a coherent mechanism, not noise — and it means `steps_to_goal` and `track_err` are
*not* independent quality axes on this scene; they are two readings of the same knob.

`-geo_free` is the efficiency sweet spot: within ~1 % of `diffuser` on both quality axes at
**1/14** the projection cost of `-tightened` (11.2 ms vs ≈ 153 ms).

**Timing note.** `avg_time_ms` and `total_ms_p95` are reported raw. Per repo convention the
`budget = 30.3 ms` / 33 Hz line is a **data-rate artefact plus cluster latency, not a real-time
target** — the `real_time_OVER×N` counters in the job logs are not a pass/fail verdict and are
deliberately not reproduced here.

---

## 7. The claim ladder, rung by rung — `corridor` holds one of three

| rung | verdict on `corridor` | evidence |
|---|---|---|
| **af > mf** | ✗ **not supported** | tie on every axis — primary saturated (both 1.000), secondary a coin-flip (5/10 vs 5/10, mean \|Δ\| = 0.83 steps vs σ = 5–14) · §3 |
| **mf > fm** | ✓ **supported**, weakly | fm alone drops to S&C 0.800 on all three `-t` rows, and carries 3.6–4.7× the step dispersion · §4 |
| **fm > diffusion** | — **untestable** | no `engine=diffusion` arm exists on `corridor` · §1.1 |

🔴 **`af > mf` is undetectable here, not refuted.** Two engines both pinned at S&C 1.000 cannot be
ranked on the primary axis, and the tiebreaker axes returned noise. This bounds the *scene*, not
α-Flow: a scene with real dynamic range could still separate them. `corridor` cannot ask the question.

### 7.1 🔴 The baseline rung is untestable campaign-wide, not just here

`batch_uav_20260908_153947` contains exactly **two** `engine=diffusion` candidates in total —
**C75 and C91, both `s_curve` K=20, both untagged (pre-U7 geometry)**. There is no `u7hg` diffusion
arm on any scene.

So the bottom rung cannot be evaluated anywhere in this campaign: mission 4's `s_curve` runs are
`u7hg`, the only diffusion arms are pre-U7, and comparing across that boundary confounds the U7
geometry change with the engine change. **A `diffusion` `s_curve` K=20 rerun under `u7hg` is the one
missing job that would make the baseline comparable** — and it is not among the five missions.

---

## 7. What this licenses

**Supported.** `corridor` is saturated at K=1 and K=2 for af_unet. af_unet and mf_unet are
indistinguishable at matched K=2 and matched backbone. Both are more reliable and far less variable
than fm, which loses 20 % of rollouts on every temporal-consistency row. Extra NFE is flat for af.
The projector trades tracking for path length monotonically.

**Not supported / out of scope.** Anything about HardFlow (no genuine row exists at K≤2). Anything
against the DPCC-diffusion baseline (no diffusion arm on this scene). Any multi-seed claim (seed 6
only). Any claim that af *beats* mf — the data says tie.

**Consequence for the campaign.** Mission 2 confirms the af_unet pipeline is sound end-to-end on a
third scene, but `corridor` joins `s_curve` (floored at 0.00) and pre-U7 `pillars` (void) as a scene
that cannot rank the arm. **The af_unet UAV arm still has no scene with usable dynamic range** — the
open question mission 5 (25514, mjpc vs pid) was designed to attack.

---

## 8. Next

- A `corridor` arm at **K=5 or higher** would put a genuine HardFlow row on this scene and re-open
  the budget axis, which is flat at K≤2.
- Missions 3 and 4 (25503, 25502 running; 25542/25543 queued) carry the K=5/K=10/K=20 evidence.
- 🔴 A **`diffusion` arm under `u7hg`** is owed before *any* headline claim — see §7.1. The
  cheapest form is `s_curve` K=20 with the `u7hg` tag, which would make C75/C91 comparable.
