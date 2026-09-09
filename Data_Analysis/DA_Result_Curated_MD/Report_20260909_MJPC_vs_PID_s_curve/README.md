# The tracker, not the planner — `mjpc` vs `pid_stopgo` on UAV `s_curve`

**Date:** 2026-09-09 · **Task:** `uav-s_curve` (Gen15 UAV Mix-ML) · **Engine:** MeanFlow, U-Net backbone (`bbunet`), K=10
**Batch:** `temp/0909/batch_uav_20260909_205118` · **Candidates:** C94 (`mjpc`) vs C95 (`pid_stopgo`)
**Jobs:** 25514 → **25554** (mjpc, n=3) · **25502** (pid_stopgo, n=8 variants, n=10) — both completed

📄 **Full working analysis:**
[`logs_in_develop/Gen15/Campaign_20260907_five_missions/DA_20260909_T5_mjpc_vs_pid_s_curve.md`](../../../logs_in_develop/Gen15/Campaign_20260907_five_missions/DA_20260909_T5_mjpc_vs_pid_s_curve.md)
· campaign index:
[`MASTER_20260908_five_missions_campaign.md`](../../../logs_in_develop/Gen15/Campaign_20260907_five_missions/MASTER_20260908_five_missions_campaign.md)

---

`s_curve` has been the UAV scene where every engine scores S&C ≈ 0 — the result that killed the
α-Flow UAV arm on 2026-09-07. This experiment asks whether that floor is a **planner** failure or a
**tracker** failure, by swapping only the low-level controller (`pid_stopgo` → MuJoCo MPC) on an
otherwise byte-identical evaluation.

## What it shows

| # | Finding | Strength |
|---|---|---|
| **F1** | **The raw MeanFlow plan was flyable; `pid_stopgo` could not fly it.** Goal reached **0/10** under PID, **3/3** under MJPC, on the *same* three initial conditions. Goal distance 2.86/2.72/2.89 → 0.299/0.298/0.294. | Fisher exact **p = 0.0035**, paired |
| **F2** | **PID fails by stalling, not diverging.** All 10 rollouts exhaust the 871-step budget while airborne and stable (`min_z` 0.96–1.24) and hold the *tightest* tracking in the study (`track_err` 0.294–0.341). Tight tracking + zero goal reach ⇒ following the reference without advancing along it. | 10/10, unambiguous |
| **F3** | **The DPCC-projected plan is unflyable by either tracker** — 0/10 and 0/3, goal distance ≈ 2.6–2.9 under both. A stronger controller changes nothing. This localises a **projector** defect. | 13/13 failures |
| **F4** | **On the HardFlow arm PID "succeeds" by dragging along the floor.** `phys_min_z` ≈ 0 or negative on **all 10** PID rollouts (min −0.009) vs **1.05–1.14 m** under MJPC, at 3.0× fewer violations per step and **half** the projection time. | 10/10 vs 3/3 |
| **F5** | 🔴 **S&C remains 0.000 in all six cells.** MJPC converts *"never arrives"* into *"arrives, still violates"*. The scene is **not** rescued as a ranking instrument. | all cells |
| **F6** | MJPC's per-step violation rate is near-constant across variants (0.068–0.077) while PID's spans **15×** (0.016–0.235). PID's numbers are dictated by *which pathology it falls into*; MJPC has a characteristic cost. | 3 variants |

---

## Figures

> **Placeholders — figures to be added.** Drop the files into this folder under exactly these
> names and the links resolve. Build specs and data sources in the [appendix](#appendix--figure-specs).

![Fig 1 — paired goal distance, PID → MJPC, per initial condition](fig1_paired_goal_distance.svg)

![Fig 2 — minimum altitude per rollout: the floor-scraping failure](fig2_min_altitude.svg)

![Fig 3 — constraint violations per step, normalised by episode length](fig3_violations_per_step.svg)

![Fig 4 — tracking error vs goal reached: the anti-correlation](fig4_trackerr_vs_goal.svg)

![Fig 5 — example trajectories on a shared initial condition](fig5_trajectories_idx0.svg)

---

## 1. Design — matched, and paired

Everything except the controller is identical: same trained MeanFlow U-Net checkpoint, same scene,
K=10, same `u7hg` honest-geometry constraint set, same seed 6, same `mpc_batch=4`, same projection
mode. Only the tracker differs, and because `controller` is a results-path key the two runs write to
separate folders and cannot collide:

```
Emf_K10_mpc4_pid_stopgo_T0.5_u7hg/     ← C95, job 25502
Emf_K10_mpc4_mjpc_T0.5_u7hg/           ← C94, job 25554
```

**The comparison is paired.** `phys_min_z` at a given `rollout_idx` is identical across *different
variants* within a run (1.106 / 1.186 / 1.154 at indices 0/1/2 for both `diffuser` and `dpcc-r` under
PID), so `rollout_idx` names the same initial condition in both jobs. MJPC's rollouts 0–2 sit
directly against PID's rollouts 0–2.

**Trial counts are deliberately unmatched: n=3 (MJPC) vs n=10 (PID).** The MJPC arm was specified as
a cheap probe. A bare 3/3 carries a Wilson 95 % lower bound of only 0.29, so no claim here rests on
an MJPC rate in isolation — F1 survives because PID's 0/10 has an *upper* bound of 0.31 and the
intervals are disjoint.

**HardFlow genuineness:** at K=10 with activation threshold A=0.5 the arm is non-degenerate
(`n_genuine ≥ 1`), so the `hardflow_sls-r` rows carry real HardFlow arithmetic and are citable. ✅

---

## 2. F1/F2 — the raw plan, and the shape of PID's failure

| idx | PID `goal_dist` | PID `n_steps` | **MJPC `goal_dist`** | **MJPC `steps_to_goal`** |
|---|---|---|---|---|
| 0 | 2.863 | 871 *(budget exhausted)* | **0.299** | **633** |
| 1 | 2.718 | 871 | **0.298** | **650** |
| 2 | 2.889 | 871 | **0.294** | **625** |
| 3–9 | 2.617 – 2.873 | 871 × 7 | — | — |

Not one PID rollout diverges or crashes. The drone is airborne and stable for the entire episode and
simply does not arrive.

The mechanism is named by an inversion that would otherwise look like an error:

| controller | `track_err` | `goal_reached` |
|---|---|---|
| `pid_stopgo` | **0.294 – 0.341** *(tightest in the study)* | **0.000** |
| `mjpc` | 0.422 – 0.490 *(visibly looser)* | **1.000** |

**Tracking error is anti-correlated with task success.** A controller that hugs the reference while
never advancing along it is not failing to track — it is failing to make **progress**. That is
`pid_stopgo`'s stop-and-go logic saturating on a reference that demands sustained forward velocity.
MJPC accepts a looser corridor and converts it into motion.

---

## 3. F3/F4 — three variants, three different failures

| variant | PID | MJPC | who is at fault |
|---|---|---|---|
| `diffuser` (raw plan) | 0/10 goal — stalls at 871 steps | **3/3 goal** | **the controller** |
| `dpcc-r` (DPCC projection) | 0/10 goal | **0/3 goal** | **the projector** |
| `hardflow_sls-r` (HardFlow-SLSQP) | 7/10 goal, but `min_z` ≈ 0 | **3/3 goal at 1.1 m** | **the controller** |

### `dpcc-r` — not a controller problem

Goal distance moves 2.665 → 2.571 and violations *rise* 29.1 → 59.0. The tracker that completely
rescued the raw plan does nothing here. PID's ten rollouts split into 8 stalls (`goal_dist` ≈ 2.85,
7–11 violations) and 2 scrapes (idx 3–4: `goal_dist` 1.80/2.02, 99/120 violations, `min_z` 0.545,
`contact_frac` 0.172). **Whatever the projector emits on `s_curve`, no controller can fly it** —
and the projection dominates the control step at `proj_ms` ≈ 2122 (PID) / 2420 (MJPC).

### `hardflow_sls-r` — the aggregate hides the failure

`goal_reached` = 0.700 under PID looks respectable until the altitudes are read:

| ctrl | `phys_min_z`, per rollout |
|---|---|
| **PID** | −0.005, 0.005, 0.014, 0.002, 0.022, −0.002, 0.020, −0.009, 0.013, 0.019 |
| **MJPC** | **1.050, 1.143, 1.112** |

Every PID rollout descends to ground level or below; it reaches the goal by sliding along the floor.
On the three paired initial conditions the goal distance is unchanged (0.295→0.299, 0.296→0.297,
0.300→0.296) while altitude jumps from ≈ 0 to ≈ 1.1 m — **same outcome, entirely different flight.**
MJPC additionally halves the projection cost (`proj_ms` 1587.5 → 761.8), the NLP converging faster
from better-conditioned states.

---

## 4. F6 — violations must be normalised by episode length

Raw violation totals are confounded: a stalled drone accrues few violations because it never
traverses the scene. Per step:

| variant | ctrl | violations | steps | **viol / step** | `min_z` |
|---|---|---|---|---|---|
| `diffuser` | PID | 13.7 | 871.0 | **0.0157** | 1.126 |
| `diffuser` | **MJPC** | 48.7 | 636.0 | **0.0765** *(4.9× higher)* | 1.100 |
| `dpcc-r` | PID | 29.1 | 871.0 | 0.0334 | 1.046 |
| `dpcc-r` | **MJPC** | 59.0 | 871.0 | 0.0677 | 1.093 |
| `hardflow_sls-r` | PID | 166.9 | 709.1 | **0.2354** | **0.008** |
| `hardflow_sls-r` | **MJPC** | 48.0 | 618.0 | **0.0777** *(3.0× lower)* | **1.102** |

Two things follow, and they point in opposite directions:

* On the **raw plan** MJPC genuinely violates more — 4.9× more per step. It is buying progress with
  constraint adherence, not merely accumulating violations by flying further. **This is a real
  trade, not a normalisation artefact.**
* On the **HardFlow arm** MJPC violates 3.0× *less* per step *and* flies at altitude *and* halves the
  projection cost. There it is a strict improvement.

MJPC's per-step rate is nearly constant across all three variants (**0.068 – 0.077**), while PID's
spans a 15× range (0.016 – 0.235). PID's violation statistics are dictated by which failure mode it
falls into; MJPC has a characteristic, predictable cost.

---

## 5. 🔴 What this does *not* show

**S&C = 0.000 in all six cells, under both controllers.** Under MJPC the raw plan reaches `success`
0.667 and `phys_safe` 0.667 — but S&C requires success *and* a clean constraint record, and the
rollouts that arrive carry 28–63 violations.

So the experiment answers its own question — *yes, the tracker was not powerful enough* — while
**failing to restore `s_curve` as a scene that can rank engines.** The floor that made it useless is
still at 0.00; the failure has only moved from *"cannot reach the goal"* to *"reaches the goal
through the constraints"*.

Also out of scope: any multi-seed claim (seed 6 only); any MJPC rate beyond the paired `diffuser`
contrast at n=3; the five variants of C95 (`dpcc-c`, `dpcc-t`, `hardflow_sls`, `-c`, `-t`) that have
no MJPC counterpart.

**Timing note.** `avg_time_ms` is reported raw (88–2513 ms). The `budget = 30.3 ms` / 33 Hz line that
appears in the job logs is a data-rate artefact plus cluster latency, **not** a real-time target, and
the `real_time_OVER×N` counters are deliberately not reproduced as a pass/fail verdict.

---

## 6. Implications

1. **A UAV result that reports `goal_reached` without `phys_min_z` can be an artefact of the
   tracker.** F4 is the cautionary case: a 70 % goal rate achieved at zero altitude.
2. **`track_err` is not a proxy for task quality on this scene** — here it is inversely related to
   success. Reporting it alone would have ranked PID above MJPC.
3. **`dpcc-r` on `s_curve` is the sharpest open defect in the UAV line**: a projected plan that no
   controller can fly. It is now isolated from the controller question and can be attacked directly.
4. **Scene dynamic range is the blocker, not the controller.** `corridor` saturates at S&C 1.000,
   pre-U7 `pillars` is void for ranking, and `s_curve` floors at 0.000 under both trackers. This is a
   constraint-set problem.

**Recommended next runs.** (a) MJPC at **n=10** over the full 8-variant subset — the effect is large
and the run cost only 1.6 h; (b) a `dpcc-r`-focused `s_curve` investigation; (c) do **not** budget
for a controller change to make `s_curve` rankable.

---

## 7. Reproduction

```bash
# PID arm (C95) — job 25502
UAV_EVAL_HOURS=24 FMPCC_SAFE_EPS_MODE=scaled FMPCC_UAV_EVAL_TAG=u7hg \
UAV_MIX_VARIANTS='diffuser,dpcc-r,dpcc-c,dpcc-t,hardflow_new,hardflow_new-r,hardflow_new-c,hardflow_new-t' \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh mf s_curve "6" "10"

# MJPC arm (C94) — job 25514 -> 25554.  UAV_MIX_CONTROLLER selects the FMPCC_mjx conda env.
UAV_EVAL_HOURS=24 FMPCC_SAFE_EPS_MODE=scaled FMPCC_UAV_EVAL_TAG=u7hg \
UAV_MIX_CONTROLLER=mjpc UAV_MIX_VARIANTS='diffuser,dpcc-r,hardflow_new-r' \
  ./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_mix/eval_k_sweep.sh mf s_curve "6" "10"
```

`UAV_MIX_CONTROLLER` and the `FMPCC_mjx` env-selection fix are Gen15 U10
(`logs_in_develop/Gen15/U10/CHANGELOG_20260907_mjpc_controller_override.md`, commit `9bda071c`);
job 25554 is the first cluster run that exercised them, and validated all four first-run checks.

### Paper table (LaTeX)

```latex
\begin{tabular}{llrrrrr}
\toprule
Variant & Tracker & S\&C & Goal & $d_{\text{goal}}$ & $z_{\min}$ & viol/step \\
\midrule
Raw plan        & PID  & 0.00 & 0.00 & 2.809 & 1.126 & 0.0157 \\
Raw plan        & MJPC & 0.00 & \textbf{1.00} & \textbf{0.297} & 1.100 & 0.0765 \\
DPCC            & PID  & 0.00 & 0.00 & 2.665 & 1.046 & 0.0334 \\
DPCC            & MJPC & 0.00 & 0.00 & 2.571 & 1.093 & 0.0677 \\
HardFlow-SLSQP  & PID  & 0.00 & 0.70 & 0.834 & \textbf{0.008} & 0.2354 \\
HardFlow-SLSQP  & MJPC & 0.00 & \textbf{1.00} & \textbf{0.297} & \textbf{1.102} & \textbf{0.0777} \\
\bottomrule
\end{tabular}
```

*MeanFlow U-Net, `s_curve`, K=10, seed 6, honest geometry (`u7hg`). PID n=10, MJPC n=3.*

---

## Appendix — figure specs

**Data source:** `temp/0909/batch_uav_20260909_205118` — `per_rollout_detail.csv`
(filter `Candidate in {94, 95}`) and `candidates_per_variant.csv`. C95 = `pid_stopgo` (n=10),
C94 = `mjpc` (n=3). `rollout_idx` names the same initial condition in both runs.

| file | what it must show | carries |
|---|---|---|
| `fig1_paired_goal_distance.svg` | Dumbbell/slope plot: `goal_dist` PID → MJPC, one line per shared `rollout_idx` (0,1,2), panelled by variant. Mark the goal threshold. | **F1** — raw plan collapses 2.86 → 0.30 on all three pairs; `dpcc-r` does not move |
| `fig2_min_altitude.svg` | `phys_min_z` per rollout, PID (10) vs MJPC (3), grouped by variant; horizontal line at z=0. | **F4** — every PID HardFlow rollout sits at/below 0; MJPC at ~1.1 m |
| `fig3_violations_per_step.svg` | Grouped bars, `n_violations / n_steps`, PID vs MJPC × 3 variants. Annotate the 4.9× (raw) and 3.0× (HardFlow) ratios. | **F6** — the trade reverses by variant; MJPC's rate is near-constant |
| `fig4_trackerr_vs_goal.svg` | Scatter `track_err_mean` (x) vs `goal_reached` (y), coloured by controller, marker by variant. | **F2** — tight tracking with zero goal reach |
| `fig5_trajectories_idx0.svg` | Top-down `s_curve` trajectory overlay for `rollout_idx=0`, raw plan: PID vs MJPC, with obstacles/half-spaces and the reference. Optional z-vs-time inset. | **F1/F2** — the stall made visible |

Keep both themes legible (no pure-black/pure-white fills), embed fonts, prefer `.svg`.
The sibling `../Report_20260903_AF_UNet/make_figs.py` is the closest existing template.
