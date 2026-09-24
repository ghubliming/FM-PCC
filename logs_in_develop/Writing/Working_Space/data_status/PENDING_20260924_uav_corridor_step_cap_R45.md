# R45 — UAV-corridor: the 396-step budget decides the baseline's result (problem · fast DA fix · runs for later)

**2026-09-24 · written by the v3 agent (v3.81) at the author's request · for the DA agent (R45a) and the run agent
(R45b/c).** Nothing in the thesis is changed yet: Chapter 6 §6.3.2 still reads the corpus at the 396-step finish-line
rule. The author picks the path; v3 rewrites the corridor section after the DA lands.

> **Status 2026-09-24 — R45a DONE (DA, U19 corridor chat):** [`DA_20260924_corridor_v3_R45a_clear_line.md`](../../../../Data_Analysis/DA_in_Paper/analysis/DA_20260924_corridor_v3_R45a_clear_line.md), note `cross_draft/to_v3/FROM_DA_20260924_R45a_corridor_clear_line.md`. The preview is verified. Read at the eval's latch timing (the position after each step) hump r is **10/10**, not 9: the one flight at x = 2.0 is 1.998785 m recorded and 2.0167 m after its last step. The x ≈ 2.26 flight of §2 is c **trial 8**, not 7. R45b and R45c have not started.
> **Code, 24-09 (author's go-ahead):** `eval_mix_uav.py` now scores corridor success at x′ = 2.0 itself
> (`SCENE_CLEAR_LINE_X`). The flights are unchanged: no new early stop. The goal-plane rule is kept as
> `success.relaxed_goal_line`. This is **not** R45c item 1: that ends flights at the line and would break identity.
> Changelog: `logs_in_develop/Gen15/U19/CHANGELOG_20260924_U19_corridor_finish_line_coding2.md`.

> **v3.82 (24-09): R45a read into the draft** — Ch 5 defines the corridor's finish line (x = 2.0 m) and the near-satisfaction bar; Ch 6 §6.3.2 and §6.3.5 rewritten on the DA's reading after each step (hump r 10/10), goal-plane counts beside. R45b/R45c not started.

## 0 · In short

- **Problem.** The corridor's step budget (396 control steps) was sized for the *level* corridor's demonstrations.
  On corridor v3 the diffusion baseline, once projected, flies slowly and runs out of steps **while still moving
  toward the finish line**: 42 of its 60 projected flights. That, not a failure to fly, is why its S&C reads 0/10 on
  the tilt and 3–6/10 on the hump.
- **Fast fix, data only (R45a, author's preference).** Re-score the existing corpus with the finish line moved to the
  **end of the corridor, x′ = 2.0 m** (author, 24-09), where the walls and both test-time constraints end. Only the
  six baseline projected cells change in S&C; no flow-model cell does. Preview: baseline S&C tilt r/c/t 0/0/0 →
  **3/8/3**, hump 3/3/6 → **9/9/9**.
- **"Continue the runs from step 396" is not possible** (§3). A longer budget is a **new sample**; if the measured
  outcome is wanted instead of the moved line, re-fly only the six baseline projected cells (R45b, ≈ 2 h wall).
- **For any later corridor run**, raise the budget (R45c): `--max-episode-length 792`, plus two small code fixes that
  need the author's go-ahead.

## 1 · The problem, with the evidence

**The rule** (`mix_uav_test/eval_mix_uav.py`):
- A flight succeeds when, within the budget, it comes within 0.30 m of the route's goal point (2.8, y_route, z_trial),
  or crosses the vertical plane x = 2.8 through that point (latch `:1910–1912`). Only the 0.30 m sphere stops a flight
  early (`:1916–1917`, break at `:1983`).
- It must stay in controlled flight: no abort, lowest altitude above 0.2 m, wall contact ≤ 2 % of physics steps
  (`:1992–2027`).
- The budget is `SCENE_MAX_EPISODE_LENGTH['corridor'] = 396` (`:195`) = 1.2 × 330, the slowest *level-corridor*
  demonstration (10 s at 33 Hz). That is about 12 s of flight.

**The corpus** (R33, `batch_uav_20260924_081422` + npz in `temp/23-09-Corridor-TEMP/plans`, first ten flights of all
136 cells). Each trial's goal is rebuilt with the evaluation's own seeding: `default_rng(10000 + i)`, route
`LCR[i % 3]`, `generator._build_traj_and_init`.

| | flights |
| :-- | --: |
| read | 1 360 |
| not successful | 222 |
| … of which on the 396-step cap without reaching the line | **222 (all of them)** |
| … aborted, crashed, below 0.2 m, wall contact | 0 |
| flow models, per-step projection, K ≤ 2, over the hump (18 cells × 10) | 180 |
| diffusion baseline, projected (tilt r 8, c 10, t 7; hump r 7, c 7, t 3) | 42 |
| unprojected, or flow at K ≥ 3 | 0 |

**Trend at the cap** (mean pace over the last 30 control steps; windows of 10 and 60 steps agree):

- **Baseline, 42 flights: every one still moving toward the line**, at 2–5 mm per step on the tilt (unprojected pace
  ≈ 20 mm). At that pace 6 would cross within +10 % more steps, 37 within +25 %, and all 42 within +50 %.
  - They end 0.41–0.74 m (tilt) and 0.36–1.60 m (hump) from the goal point.
  - Most are at y ≈ −0.5: after the tilt they do not return to the route, so for them only the x = 2.8 plane is in
    reach.
- **Flow models, 180 flights: stalled** before the hump's apex (median movement 2.4 cm over the last 60 steps). A longer
  budget cannot help them. Six of them (MeanFM and CI-MeanFM, K = 2, rule t) drift sideways across the wall's boundary;
  they violate the wall constraint and stay at S&C 0 under any budget.
- **Beyond the constrained section nothing is violated:** 0 violating steps at x ≥ 2.31 m in all 1 360 flights.

Scripts (preview, not the DA of record): `Data_Analysis/DA_in_Paper/analysis/R45_preview/cap_check.py` and
`relaxed_line.py`.

## 2 · R45a — the DA with the finish line at the end of the constrained section (data processing only) — DA agent

**Definition (author, 24-09: "the new goal line … direct at end of the corridor if good").**
- **x′ = 2.0 m, the end of the corridor.** The walls and both test-time constraints (the tilt, the roof) are active
  only for x ∈ [−2, 2]. For a vehicle inside the corridor nothing can be violated past x = 2.0: the wall-end caps
  (disks r = 0.05 at x = 2, y = ±1, plus the 0.31 m reach) lie outside the inflated walls (|y| > 0.64).
- Checked on the corpus: past x = 2.0 there are 25 violating steps in all 1 360 flights. They come from three flights
  (hump, MeanFM K2 t trial 9; CI-MeanFM K2 t trials 8, 9) that had already crossed the wall's boundary sideways
  (y −0.73 to −0.97) and clip a cap at x 2.00–2.23. They are violating anyway (S&C 0 under any line). Past x = 2.31
  there are none.
- `success_clear` = `success_relaxed` **or** (the flight's centre reaches x ≥ x′ within the 396 steps, in controlled
  flight).
- `S&C_clear` = `success_clear` **and** violation-free, with violations counted over the flight exactly as now
  (§1: none lie beyond x′).

**Conditions to check on every flight the moved line turns into a success** (the author's condition):
1. It is past x′ before the cap, so it has finished every constraint.
2. No abort, and safe.
3. It is still moving toward the goal at the cap: report the pace.
4. It has no violating step past x′.

**Scope.**
- Apply the rule to every cell, not only the baseline, and report both rules side by side.
- Tables: `tab:uav-corridor-raw` does not change (every unprojected flight already succeeds); `tab:uav-corridor` and
  `tab:uav-corridor-projection` change in the baseline rows only.
- Also rerun the frontier (DA §3.4), the budget facts (§3.2) and the §3.7 conclusion line.

**Preview** (v3 agent; verify):

| baseline, projected | S&C now | **x′ = 2.00** | x′ = 2.31 | x′ = 2.36 | x′ = 2.50 |
| :-- | :-- | :-- | :-- | :-- | :-- |
| tilt r / c / t | 0 / 0 / 0 | **3 / 8 / 3** | 3 / 8 / 3 | 3 / 8 / 3 | 3 / 3 / 3 |
| hump r / c / t | 3 / 3 / 6 | **9 / 9 / 9** | 9 / 8 / 9 | 9 / 8 / 9 | 6 / 4 / 9 |

At x′ = 2.0 three flow flights also gain a success (the drift flights above: MeanFM K2 t 1, CI-MeanFM K2 t 2); their
S&C stays 0.

Reading if it holds:
- **Hump:** FM K20 endpoint single (8/10, 291 ms per action) and the baseline under per-step projection (9/10 under
  every rule; best rule c, 623 ms, by the chapter's tie-break) are both non-dominated. That is a **trade-off**, no
  longer "FM ahead of the baseline".
- **Tilt:** the baseline per-step c (8/10, 666 ms) is the only configuration with any S&C.

The results barely depend on the line's position between 2.0 and 2.36 m: one hump baseline flight (rule c, trial 7,
which ends at x = 2.26) changes. At 2.50 m, where aligned flights enter the goal sphere, the counts are lower, because
that line reaches back into the free last half metre.

**Honesty clause for the thesis.** This moves the success definition after the data were seen. It is defensible only
because:
- it is fixed by the geometry (the end of the corridor, where every constraint ends), not by the outcome;
- it is applied to every configuration;
- the original finish-line counts are printed beside it.

Ch 5's success clause must state it (v3 does that after the author approves).

## 3 · R45b — the measured outcome instead of the moved line (optional) — run agent

**Why "continue from the current point" cannot be done.** The npz stores `obs_all` = [commanded position, position,
velocity]. It does not store the attitude, the body rates, the cascaded controller's integrator states or the planner's
random state. A resumed flight would not be the same flight.

**Why a longer-budget re-run is a new sample, not an extension.** Torch is seeded once per job, when the model is built
(`mix_uav/utils/setup.py:166` via `build_experiment`, `eval_mix_uav.py:2667`). The random stream is then shared by every
trial and variant of the job (`_run_variant`, `:2734`). Every flight after the first capped one draws different noise.

**If wanted**, re-fly only the cells a budget can change:
- The six baseline projected cells: diffusion K20 × {`corridor_v3_tilt`, `corridor_v3_ablation_hump`} ×
  `dpcc-{r,c,t}-bounds_free-pdes-tightened`.
- Settings: `--max-episode-length 792`, a new tag (e.g. `p24cv3cap792`), ten flights.
- The DA counts violations up to each flight's first crossing of the finish line. Without the R45c(i) fix, a flight
  that crosses the plane off the sphere flies on to the cap and could drift toward the workspace box.

**Cost.** About 12 flights × up to 792 steps × 0.65 s ≈ 1.7 h per cell; the six jobs in parallel ≈ 2 h wall.

**No flow cell needs it.** Their unsuccessful flights are stalled (or violate), and their successful ones had finished.
No budget changes their S&C.

## 4 · R45c — the budget for any later corridor run — run agent (config) / author (code)

- **Budget:** pass `--max-episode-length 792` (2 × 396; the flag exists, `eval_mix_uav.py:356`). The per-scene yaml
  key `max_episode_length: {corridor: 792}` is read at `:2399–2403` but is not set in `config/uav_projection.yaml`
  today; check that the eval's `config` carries it before relying on it.
- **Code, needs the author's go-ahead** (each breaks flight-for-flight identity with the current corpus):
  1. End a flight at its first crossing of the finish line, not only inside the 0.30 m sphere. Today 87 flow and 7
     baseline successes crossed the plane off the sphere and flew on to the cap.
  2. Seed torch per trial (e.g. `torch.manual_seed(base + trial_index)` at the start of `rollout_one`), so that a
     budget change re-samples nothing.
  3. Optionally, store the full end state so that a flight can be extended.

## 5 · What v3 changes once R45a (or R45b) lands — not applied now

- Ch 5 `sec:setup:metrics:uav`: the corridor success clause (leaving the corridor at x = 2.0 m within 396 steps),
  with the old finish-line count kept.
- Ch 6 §6.3.2: the baseline rows of `tab:uav-corridor` and `tab:uav-corridor-projection`; the sentence "run out of their
  396 control steps"; the budget, frontier and projection-methods paragraphs; the §6.3.5 corridor paragraph.
- The → v4 note (Ch 8 facts) and §6.4 (🔒, the author's).
