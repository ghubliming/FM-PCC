# DA 2026-09-24 · UAV-s-curve R44a: the raw generative grid (Table 6.14 `tab:uav-scurve`), complete

**Analysis of record for `tab:uav-scurve` and for its `\hole` (which configuration Table 6.15 is flown with).**
- **Corpus:** the DA_UAV_v1 batch over the whole UAV_MIX tree, `temp/23-09-FULL/24-09-1000/batch_uav_20260924_081422`
  (`per_rollout_detail.csv`, `data_quality.csv`).
- **Cells:** run tag `p23scgrid` (CI-MeanFM `EPlatest_p23scgrid`), geometry `s_curve_hg`, variant `diffuser`, cascaded
  geometric controller (`pid_stopgo`), seed 6, ten flights of the one route.
- **Jobs:** 26167–26176, submitted 23-09 21:26 and finished by 22:29 UTC.
- **Script:** [`scurve_r44.py`](scurve_r44.py). Runbook: `data_status/SLURM_RUNBOOK_20260923_uav_scurve_R44.md`; spec:
  `data_status/PENDING_20260923_uav_scurve_R44_raw_first.md`.

Chapter 6 (v3 draft) is **not** edited here; the writing agent transcribes from §2 (cross-draft note
`cross_draft/to_v3/FROM_DA_20260924_scurve_R44a_ready.md`).

> **Verdict.** The grid is complete and clean: ten cells, 100 flights, and every check passes. The six cells the table
> printed from older tags reproduce **flight for flight** (0 of 600 outcome values differ); only the time per action moves
> by ≤ 1.2 ms. The four new cells: MeanFM 3/10 at nfe = 1 and 0/10 at nfe = 20; CI-MeanFM 1/10 at nfe = 20; **FM 9/10 at
> nfe = 1**, the most crossings in the grid. **The selection rule therefore picks FM at nfe = 1**, a budget with no guiding
> step for endpoint projection. **No flight in any cell is violation-free (S&C 0/10 everywhere).** The flights track their
> plans with a mean error of 0.30–0.33 m, while the demonstrated route clears the constraint boundaries by only 0.121 m
> (the evaluation's own feasibility check).

## 1 · Checks

| check | result |
| :-- | :-- |
| job logs 26167–26176 (`Slurm_Codes/logs/2026-09-23/21_26_50_p23scgrid_*`) | all ten `Job completed successfully`; each prints tag `p23scgrid`, `SAFE_EPS scaled/1e-3`, `ENGINE … SCENE: s_curve SEEDS: 6 N_TRIALS: 10 K: <k>`, geometry `s_curve_hg` only, `variants=['diffuser']`, controller `pid_stopgo` (FMPCC env); CI-MeanFM `unet`, α_end 0.2, checkpoint `latest` |
| checkpoints loaded (logs) | MeanFM `state_best.pt` step 95000 · CI-MeanFM `state_100000.pt` · FM and diffusion `state_best.pt` step 91000: **the same files the old rows loaded** |
| batch coverage | 10 cells × 10 flights; no other cell under the tag |
| `data_quality.csv` | 10 rows, 10 rollouts each; 0 projection cut-off trips, timing present, 10 diagnostics files per cell |
| reproduction against `analysis_results_checkpoint/15-09/batch_uav_20260915_100816` | MeanFM 2, CI-MeanFM 1 and 2 (`EPlatest_u6unet_ae02`), FM 2 and 20, diffusion 20 (`u7hg`): **0 differing values** over 10 flights × 10 outcome metrics each (success, S&C, distance, aborted, abort step, steps, violating steps, crossed, min altitude, tracking error). ms/step old → new: 18.1 → 17.9, 9.2 → 9.3, 18.5 → 18.0, 17.4 → 17.2, 169.5 → 168.3, 173.9 → 174.3 |

Consequences:
- **The evaluation is deterministic for a fixed configuration** (checkpoint, seed, nfe, variant, geometry, controller,
  SAFE_EPS). A cell re-flown under a new tag is the old cell.
- The open question of the dataref (whether `EPlatest_u6unet_ae02` is the `u7hg` CI-MeanFM checkpoint) is closed. It is:
  the same file and the same flights.

## 2 · Table 6.14 `tab:uav-scurve`

Ten flights per cell; *success* = crossed the finish line in safe flight (see §3.4); *S&C* = success on a violation-free
flight; *distance* = mean final distance to the route's goal point; *aborted* = ended by the divergence guard (inverted);
*ms/step* = time to compute one action (network only, nothing is projected here).

| model | nfe | success | S&C | distance [m] | aborted | ms/step |
| :-- | --: | --: | --: | --: | --: | --: |
| MeanFM | 1 | **3/10** | 0/10 | 2.023 | 7/10 | 9.2 |
| MeanFM | 2 | 0/10 | 0/10 | 2.515 | 9/10 | 17.9 |
| MeanFM | 20 | **0/10** | 0/10 | 2.772 | 10/10 | 176.4 |
| CI-MeanFM, α_end 0.2 | 1 | 6/10 | 0/10 | 1.306 | 4/10 | 9.3 |
| CI-MeanFM, α_end 0.2 | 2 | 1/10 | 0/10 | 2.082 | 7/10 | 18.0 |
| CI-MeanFM, α_end 0.2 | 20 | **1/10** | 0/10 | 2.511 | 9/10 | 180.4 |
| FM | 1 | **9/10** | 0/10 | 0.571 | 1/10 | 8.9 |
| FM | 2 | 7/10 | 0/10 | 1.110 | 3/10 | 17.2 |
| FM | 20 | 6/10 | 0/10 | 1.193 | 4/10 | 168.3 |
| ★ Diffusion | 20 | 0/10 | 0/10 | 2.432 | 8/10 | 174.3 |

Bold: the four cells that were *pending (R44a)*. The other six equal the printed values in every outcome column. Their
ms/step now comes from the new tag, and the dataref moves to this batch and tag. LaTeX rows: `scurve_r44.py` prints them
ready to paste.

## 3 · What the table shows

### 3.1 Success falls as the budget grows, for every flow-based model

| model | nfe 1 | nfe 2 | nfe 20 |
| :-- | --: | --: | --: |
| FM | 9 (1 aborted) | 7 (3) | 6 (4) |
| CI-MeanFM | 6 (4) | 1 (7) | 1 (9) |
| MeanFM | 3 (7) | 0 (9) | 0 (10) |

Crossings never rise with the budget and aborts never fall, in all three models. A failed flight is almost always an
inverted one: 54 of the 57 failures in the flow cells are aborts, and the other three crossed the line after reaching the
floor (§3.4). No flow-model flight ran to the episode limit. Flights invert between control steps 386 and 530, 11.6–15.9 s into the
flight; the successful flights reach the line at step 590–635.

*Reading, not tested here:* one sampling step returns a plan nearer the conditional mean, which is smoother and easier to
track. More steps sharpen the plan toward the demonstrated turn, which the cascaded geometric controller cannot hold. The
plans in the npz can test this (§5). The table alone does not.

### 3.2 No flight is violation-free (S&C 0/10 in all ten cells)

> ⚠️ **Corrected 24-09 by [`DA_20260924_scurve_R44bc_projection_controller.md`](DA_20260924_scurve_R44bc_projection_controller.md) §4, §6.**
> The mechanism below is wrong. The 0.30–0.33 m is the setpoint's *lead* along the path; the flown path stays within about
> 1 cm of the commanded path. The violations come from the commanded path itself, which cuts the second inside corner of
> the crossover by 8–19 cm on every flight. The counts in this section stand.

- Every flight of every cell has violating steps. The flow models have 13–26 per flight on average, the baseline 119
  (± 199).
- The flow models' flights track their plans with a mean error of 0.30–0.33 m (FM nfe 1: 0.303 m). The evaluation's
  feasibility check prints that the demonstrated route clears the planning set by 0.121 m ("need ≥ 0.30 m to absorb the
  policy tracking error").
- With that tracking error and that clearance, violation-free flight is bounded near zero for any model. This is a
  property of the scene and the controller, not a ranking of the models.
- The baseline tracks worst (0.558 m).

### 3.3 The baseline

Diffusion at its own twenty steps crosses on none of ten flights: 8 inverted (steps 388–501), and the other 2 ran to the
episode limit (871 steps) without crossing. Mean final distance 2.432 m. As before, it is the only model with a tracking error above 0.5 m.

### 3.4 What *success* counts (for the writing agent)

`success_relaxed` = crossed the finish line **and** safe: contact within the scene's limit, **airborne (lowest altitude
> 0.2 m)**, and not aborted (`eval_mix_uav.py:1994–2027`). Three flights crossed the line after dropping to the floor
(lowest altitude 0.00 m) and are therefore not successes:
- MeanFM nfe 2, flight 3;
- CI-MeanFM nfe 2, flights 3 and 9.

Chapter 5's definition ("the vehicle crosses the finish line of its scene", `sec:setup:metrics:uav`) states the abort
exclusion but not the floor. One clause closes it.

## 4 · The selection (the table's `\hole`)

The chapter's rule is the configuration whose plans cross the finish line on the most flights (ties: fewer aborted, then
smaller distance):

1. **FM, nfe = 1: 9/10**, 1 aborted, 0.571 m;
2. FM, nfe = 2: 7/10;
3. FM, nfe = 20: 6/10;
4. CI-MeanFM, nfe = 1: 6/10.

**Table 6.15 is therefore flown on FM at nfe = 1.** At that budget endpoint projection has no guiding step (activation
threshold 0.5), and the evaluation does not run it. The endpoint rows print "no guiding step", as the section's own
sentence ("none at one or two") anticipates. Spec §2 leaves the choice to the author: (i) FM at nfe = 20 with both
methods, or (ii) per-step only at nfe = 1. See the pitch in the spec §2 and the runbook.

Context from cells already in this batch (plain or tightened sets, older tags, **not** part of Table 6.15):
- FM at nfe = 2, tightened per-step (`u7hg`): 3, 5 and 5 of ten crossing, 4–5 aborted, S&C 0, 143–165 ms/step.
- FM at nfe = 20, plain set: per-step 0–2/10 crossing with 8–10 inverted at 3.2–4.7 s per control step; endpoint 2–8/10
  at 1.1–3.7 s (`u7hg`, `u18sc`).
- S&C 0 in all of them.

## 5 · Caveats and what is not in this DA

- **FLAW §B (recorded, not fixed):** FM, and only FM, is sampled from a half-scale initial noise, while the average-velocity
  models use unit scale (`mix_uav/models/engine_registry.py:274`; `data_status/FLAW_20260920_known_flaws.md` §B). A
  half-scale prior concentrates the sample toward the centre of the learned distribution. UAV-s-curve is the one scene
  where FM leads, and it is the selected configuration. Any sentence that compares FM with MeanFM or CI-MeanFM here
  inherits §B2's caveat. FLAW §B2's table entry for the quadrotor ("the only flow model that never reaches S&C 1.00")
  does not describe this scene, where no model reaches S&C above 0.
- One training seed, ten flights of one route. Counts only (thesis rule: no tests).
- ms/step is cluster latency: compare between configurations, never against 30.3 ms.
- **Not done, needs the npz** (not in this drop):
  - whether the commanded setpoint was itself clean at the violating steps (tracking versus plan; `corridor_v3_grid.py`
    has the method);
  - plan smoothness against nfe (§3.1);
  - a flown-path figure of the selected configuration.

  Fetch: `Slurm_Codes/temp_bash/fetch_20260924_R44_scurve.sh`, best run once after R44b/c.

Claude (Opus 5.5, Claude Code, R44 run chat) · 2026-09-24.
