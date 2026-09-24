# DA 2026-09-24 · UAV-s-curve R44b/c: projection and the two controllers on FM nfe 1 — complete

**Analysis of record for `tab:uav-scurve-projection` (Table 6.15), `tab:uav-controller` (Table 6.16, three of four rows),
the mechanism behind every violation in Tables 6.14–6.16, and the controller cost on the selected configuration.**
- **Corpus:** the raw result folders of the two R44 fetches of 24-09 (`fetch_20260924_R44_scurve.sh`, git `5092e056`):
  16:17 (621 files) and 17:31 (`TAGS=p23scmjpc`, 83 files), every md5 of both manifests verified. The second was merged
  into the first, in `temp/23-09-FULL/S_CURVE-P2/R44_scurve_20260924_161747`: its 42 C0 files were byte-identical to the
  first fetch's, and only the job-id ledger gained C1's line.
- **Tags:** `p23scgrid` (A, jobs 26167–26176), `p23scproj` (B1, job 26195), `p23scmjpc` (C0, job 26196; C1, job 26204).
- **Source:** no DA_UAV_v1 batch (author). The sources are `results.json`, the npz and the job records.
- **Script:** [`scurve_r44_raw.py`](scurve_r44_raw.py) (python3.14). Table 6.14 itself is in
  [`DA_20260924_scurve_R44a_raw_grid.md`](DA_20260924_scurve_R44a_raw_grid.md); its §3.2 reading is corrected here (§6).
- **Figure 6.9:** rebuilt on this configuration (§8).

> **Verdict: UAV-s-curve shows the controller on an extreme trajectory, and three limits separate.**
>
> 1. **The tracking controller's limit is stability at the second turn.** 54 of the 57 flow-model flights lost in
>    Table 6.14 invert. 48 of those 54 inversions happen at flown x 0.2–1.2 m (median 0.38 m), just past the crossover
>    as the route turns into the second straight.
>    - On the selected plans (FM, nfe 1, unprojected), **MuJoCo MPC inverts on none of ten flights** (cascaded
>      geometric: one), and it brings every flight within 0.3 m of the goal point (mean distance 0.297 against 0.571 m).
>      Successes are 9/10 under both; one MPC flight is scored unsafe for contact.
> 2. **The controller that rescues the unprojected plans loses every projected one.**
>    - Under per-step projection (random rule), MuJoCo MPC succeeds on **0 of 10** flights. Seven invert 15.7–20.7 s in,
>      pressed against the outer wall of the second straight (flown x 0.72–2.09 m, y ≈ 1.15). The other three reach the
>      goal region with contact above the scene limit (contact on 16 % of steps).
>    - The cascaded geometric controller flies 5/10 of the same kind of plans. So the gain from changing the controller
>      depends on the plan.
> 3. **Constraint satisfaction is the plan's limit, not the tracker's.** No flight among the 40 of Table 6.16 is
>    violation-free.
>    - On every flight of every cell, the commanded path cuts the second inside corner of the crossover. It passes
>      8–19 cm inside the corner's keep-out (the 0.05 m corner plus the 0.31 m rotor reach).
>    - The cascaded geometric controller follows it within 1 cm on average, and never comes more than 1.4 cm closer to
>      the corner than the command. The "tracking error" of 0.30 m is the setpoint's lead along the path.
>    - Per-step projection binds the plan's *measured* position (DPCC's binding), not the commanded setpoint, so the cut
>      stays (8–9 cm inside). Projection drops the successes from 9 to 5 under every rule.
> 4. **The price of the stable controller:** 125.4 ms per control step for MuJoCo MPC and the simulator step, against
>    5.2 ms for the cascaded law (×24), ten against ten on the unprojected plans. The pilot's 125.8 against 6.5 ms is
>    reproduced. On the projected plans the whole control loop costs 348.6 against 159.4 ms per step.

## 1 · Checks

| check | result |
| :-- | :-- |
| fetch | 621 files, **all md5 verified**, none empty |
| job log 26204 (C1) | complete; tag `p23scmjpc`, K1, `dpcc-r-tightened` only, controller `mjpc` → `FMPCC_mjx`, checkpoint step 91000 |
| job logs 26195 (B1), 26196 (C0) | complete; tag, K1, variants (`diffuser` + `dpcc-{r,c,t}-tightened` / `diffuser`), controller (`pid_stopgo` → FMPCC env / `mjpc` → `FMPCC_mjx`), checkpoint `state_best.pt` step 91000 as in Phase A; the endpoint arm is dropped at K1, as designed |
| geometry snapshot of every run vs repo `s_curve_hg` | identical (16 cells) |
| violating steps re-scored from the flown path with the evaluation's own scorer | **0 flights disagree** (16 cells, 160 flights) |
| projection health | 0 cut-off trips, 0 backstop hits |
| determinism | B1's `diffuser` (job 26195) against A7 (job 26171): 0 differing outcome fields over 10 flights; ms/step 8.8 against 8.9 |
| Table 6.14 recomputed from the raw folders | equals the printed table in every cell |

## 2 · Table 6.15 `tab:uav-scurve-projection`: FM, nfe 1, cascaded geometric controller

| projection | selection rule | success | violating steps | distance [m] | aborted | ms/step |
| :-- | :-- | --: | --: | --: | --: | --: |
| none | — | 9/10 | 23.0 ± 5.4 | 0.571 | 1/10 | 8.9 |
| per-step | random | 5/10 | 71.3 ± 66.6 | 0.572 | 3/10 | 157.1 |
| per-step | cumulative cost | 5/10 | 74.7 ± 95.9 | 1.007 | 3/10 | 163.8 |
| per-step | temporal consistency | 5/10 | 37.2 ± 19.6 | 0.817 | 5/10 | 135.9 |
| endpoint | — | no guiding step at nfe 1 | | | | |

```latex
    none      & ---                   & 9/10 & $23.0 \pm 5.4$  & 0.571 & 1/10 & 8.9 \\
    \addlinespace[3pt]
    per-step  & random                & 5/10 & $71.3 \pm 66.6$ & 0.572 & 3/10 & 157.1 \\
    per-step  & cumulative cost       & 5/10 & $74.7 \pm 95.9$ & 1.007 & 3/10 & 163.8 \\
    per-step  & temporal consistency  & 5/10 & $37.2 \pm 19.6$ & 0.817 & 5/10 & 135.9 \\
```

**Reading.**
- S&C is 0/10 in every row.
- Projection costs four successes under every rule, adds aborts, and raises violating steps. Temporal consistency is
  least affected (37.2), but it aborts most (5).
- The large spreads of random and cumulative cost come from two flights each that drop to the floor after crossing and
  then sit outside the box. Three flights cross the line without counting as successes: two floor, one inverted after
  crossing.
- Projection time is 126–155 ms per step against 8.8 ms of network time.
- The unprojected row is A7's (8.9 ms/step); B1's own unprojected cell is the same ten flights (8.8 ms/step).

## 3 · Table 6.16 `tab:uav-controller`: FM, nfe 1, the same plans under the two controllers, ten flights each (complete)

| projection | controller | success | violating steps | distance [m] | aborted |
| :-- | :-- | --: | --: | --: | --: |
| none | cascaded geometric | 9/10 | 23.0 ± 5.4 | 0.571 | 1/10 |
| none | MuJoCo MPC | 9/10 | 41.5 ± 18.4 | 0.297 | 0/10 |
| per-step (random) | cascaded geometric | 5/10 | 71.3 ± 66.6 | 0.572 | 3/10 |
| per-step (random) | MuJoCo MPC | **0/10** | 105.5 ± 46.0 | 1.626 | **7/10** |

```latex
    none      & cascaded geometric & 9/10 & $23.0 \pm 5.4$  & 0.571 & 1/10 \\
    none      & MuJoCo MPC         & 9/10 & $41.5 \pm 18.4$ & 0.297 & 0/10 \\
    \addlinespace[3pt]
    per-step  & cascaded geometric & 5/10 & $71.3 \pm 66.6$ & 0.572 & 3/10 \\
    per-step  & MuJoCo MPC         & 0/10 & $105.5 \pm 46.0$ & 1.626 & 7/10 \\
```

More per cell (not in the table):

| cell | S&C | within 0.3 m of the goal point | not safe | contact fraction (max) | lowest altitude [m] |
| :-- | --: | --: | :-- | :-- | --: |
| none, cascaded | 0/10 | 9/10 | 1 inverted (step 461) | 0.001 (0.004) | 0.96 |
| none, MuJoCo MPC | 0/10 | **10/10** | 1 contact above the scene limit (0.096) | 0.035 (0.096) | 0.84 |
| per-step random, cascaded | 0/10 | 4/10 | 3 inverted (steps 492–660, at x ≈ 3: at or after the line), 2 floor | 0.031 (0.093) | 0.00 |
| per-step random, MuJoCo MPC | 0/10 | 3/10 | 7 inverted (steps 523–690, 15.7–20.7 s, x 0.72–2.09 m, y ≈ 1.15 against the outer wall), 3 contact above the limit | **0.159 (0.312)** | 0.89 |

**Which per-step rule rides in Table 6.16.**
- All three rules tie at 5/10 successes.
- Random and cumulative cost abort 3, temporal consistency 5.
- Random has fewer violating steps than cumulative cost (71.3 against 74.7) and a smaller distance (0.572 against
  1.007 m).
- The order is the one in `PENDING_20260923_uav_scurve_R44_raw_first.md` §4. **C1 = `dpcc-r-tightened`.**

## 4 · Where the violations come from: the commanded path, not the tracker

*Lead* = |p_des − p|, which the evaluation reports as the tracking error. *Cross-track* = distance of the flown position
from the polyline of the flight's commanded setpoints. *Corner clearance* = closest approach to an inside corner minus
its keep-out radius, 0.36 m (negative = inside).

| cell | lead [m] | cross-track, mean / p95 / max [cm] | corner clearance, commanded path | …flown path | inside, commanded / flown (flights) |
| :-- | --: | :-- | --: | --: | :-- |
| FM nfe 1, cascaded | 0.296 | 0.8 / 5.7 / 10.9 | −0.076 | −0.062 | 10/10 · 10/10 |
| FM nfe 1, MuJoCo MPC | 0.430 | 5.2 / 18.1 / 28.8 | −0.203 | −0.059 | 10/10 · 10/10 |
| FM nfe 1, per-step random | 0.371 | 8.5 (flight means; max flight 20.8) | −0.092 | −0.065 | 10/10 · 10/10 |
| FM nfe 1, per-step random, MuJoCo MPC | 0.540 | 12.1 (flight means; max flight 24.1) | −0.209 | −0.080 | 10/10 · 10/10 |
| FM nfe 1, per-step cumulative cost | 0.364 | 6.9 (max flight 16.0) | −0.084 | −0.055 | 10/10 · 10/10 |
| FM nfe 1, per-step temporal consistency | 0.385 | 6.0 (max flight 7.9) | −0.080 | −0.063 | 10/10 · 10/10 |
| MeanFM nfe 1, cascaded | 0.306 | 1.0 / 5.8 / 22.3 | −0.133 | −0.147 | 10/10 · 10/10 |
| CI-MeanFM nfe 1, cascaded | 0.308 | 0.9 / 5.5 / 32.0 | −0.093 | −0.059 | 10/10 · 10/10 |
| FM nfe 20, cascaded | 0.306 | 1.1 (max flight 2.5) | −0.100 | −0.072 | 10/10 · 10/10 |
| Diffusion nfe 20, cascaded | 0.527 | 9.3 (max flight 44.4) | −0.187 | −0.200 | 10/10 · 10/10 |

**The nearest corner is the second inside corner, (0.5, 0.3), on every flight.** The flown violations concentrate there:
- FM nfe 1, cascaded: 223 of 230 violating steps touch a corner, at x 0.18–0.24; the 12 box steps are ceiling crossings
  of the one flight that inverts.
- Walls are rare in the unprojected flow cells (0–16 steps). They appear in the projected cells (66–77), under MuJoCo MPC
  (39), and for the baseline (834 of its 1194 steps). Projected plans under MuJoCo MPC violate the walls on 654 of their
  1055 violating steps: the flights hug the outer wall of the second straight, where seven invert.

What this establishes:
1. **The plans cut the corner that the demonstrations clear.** The demonstrated route clears the constraint set by
   0.121 m (the evaluation's feasibility check). Every model's commanded path passes 8–19 cm inside the corner's keep-out.
2. **The cascaded geometric controller follows what it is commanded.** Sideways it stays within 1 cm on average, every
   flow model. Near the corner it departs by up to 11 cm on the selected configuration (single flights of MeanFM and
   CI-MeanFM up to 22–32 cm). Its closest approach to the corner is 1.4 and 3.4 cm wider than the command's on FM and
   CI-MeanFM, and 1.3–1.4 cm deeper on MeanFM and the baseline. The tracker adds at most 1.4 cm to a cut that is 8–19 cm
   deep in the command.
3. **The 0.30 m "tracking error" is the setpoint's lead**: the position it is sent to lies about 0.3 m ahead on the path.
   It does not explain the violations.
4. **MuJoCo MPC follows the commanded path less closely** (5 cm mean, 29 cm max). The plans generated from its states cut
   the corner deeper (−0.203 m). Its flown clearance is the same as the cascaded controller's (−0.059 m), but it spends
   more steps there (384 corner steps against 223) and adds 39 wall steps.
5. **Per-step projection does not move the commanded path out of the corner** (−0.080 to −0.092 m). On this scene the
   projector binds the geometry to the plan's measured position p, not to the commanded setpoint p_des
   (`eval_mix_uav.py:1400–1418`, DPCC's binding). UAV-corridor v3 uses the setpoint binding (`-pdes`); UAV-s-curve does
   not (§7).

## 5 · Where the flights invert

The flown position at the divergence abort, all cells of Table 6.14:
- MeanFM nfe 1/2/20: x 0.21–1.15, y 0.25–1.16 (26 flights).
- CI-MeanFM: x −0.02–1.26 (20 flights).
- FM: x −0.19–1.93 (8 flights).
- Diffusion: x 0.33–1.56 (8 flights).

48 of the 54 flow-model inversions lie at x 0.2–1.2 m (median 0.38 m), and 7 of the baseline's 8 at x 0.33–0.43 m. That
is just past the crossover, where the route turns into the second straight (y ≈ 0.8 from x = 0.5). Flights invert at the
second turn, 11.6–15.9 s into the flight. MuJoCo MPC flies the selected plans through it on every flight.

## 6 · Correction to the R44a reading (and to the v3 prose built on it)

**Wrong:** `DA_20260924_scurve_R44a_raw_grid.md` §3.2 and the note to v3 said: "the flights track their plans with a mean
error of 0.30–0.33 m, while the route clears the boundaries by 0.121 m, so violation-free flight is bounded near zero for
any model". v3 now prints a sentence built on that ("Every flight that crosses violates the constraints on the way: … a
mean tracking error of 0.30 to 0.33 m … while the demonstrated route clears the constraint boundaries by 0.121 m"), and
its provisional conclusion repeats it. The numbers are right; the mechanism is not:
- 0.30 m is the setpoint's lead along the path.
- The flown path stays within about 1 cm of the commanded path.
- The violations come from the commanded path itself, which cuts the second inside corner by 8–19 cm on every flight.

**Right:** "No flight is free of violations because the plans cut the second inside corner of the crossover, which the
demonstrated route clears by 0.121 m. The commanded path passes 8 to 19 cm inside the corner's keep-out on every flight of
every configuration, and the vehicle follows it: the cascaded geometric controller keeps within 1 cm of the commanded path
on average, and it never comes more than 1.4 cm closer to the corner than the command does." The tracking-error number
(0.30–0.33 m) can stay only if it is called the distance to the commanded position.

## 7 · The one open question: a projector that binds the commanded setpoint (optional run, author's call)

On UAV-corridor the chapter already says that the projection there "constrains the commanded position in addition to the
measured one" (`-bounds_free-pdes-tightened`, Gen15 U16). The reason is that the vehicle lags its setpoint, so binding the
plan's measured position lets the command overshoot the constraint. §4 shows the same mechanism on UAV-s-curve: the
command cuts the corner under DPCC's binding.

The run that answers whether projection can remove the corner cut once it binds what is commanded:
- FM nfe 1, `dpcc-{r,c,t}-bounds_free-pdes-tightened` (the corridor's stack), ten flights, cascaded geometric;
- one job, about 1 h (per-step at nfe 1 is 0.13–0.16 s per step), tag `p23scproj` (new variant folders, nothing
  overwritten);
- if it succeeds within the constraints, its best rule is the natural per-step row for Table 6.16, with a MuJoCo MPC twin
  (about 40 min).

Not run, not required for the section as written: the chapter can state DPCC's binding as the configuration and point to
the corridor. This needs a driver change (a Phase-B variant list with the stack), and waits for the author's go.

## 8 · Done for the section: C1, Figure 6.9, the cost table

1. **C1** (job 26204): the fourth row of Table 6.16 (§3). Table 6.16 is complete, ten against ten, unprojected and
   per-step.
2. **Figure 6.9 `fig:uav-scurve-paths`, rebuilt** in `DA_in_Paper/figures/da/fig_uav_scurve_paths.{svg,png}` (PNG rendered
   after the SVG, so `export_to_draft.py` accepts it). The draft copy is made by `python3 plotting/export_to_draft.py v3`
   (writing agent).
   - FM nfe 1, ten flights per panel, 2 × 2: rows unprojected / per-step projection (random, tightened); columns cascaded
     geometric / MuJoCo MPC.
   - Green = success, red = no success, a cross = collided or lost control.
   - Panel counts: 9/10 · 9/10 · 5/10 · 0/10 success, 0/10 violation-free everywhere.
   - Visible in it: the corner cut (every path passes through the keep-out circle of the second inside corner), the one
     cascaded inversion just past it, the looser MPC paths, and the MPC-projected flights ending against the outer wall.
   - Data: `data/uav_paths.json` `figures['scurve']`, written by the new `plotting/extract/scurve_r44_paths.py`, which
     replaces only that key (checked: the pillars, pillars CI-MeanFM, corridor and corridor-altitude entries are
     unchanged). `builders/paths.py` now credits that extractor. The MeanFM nfe 10 pilot figure is superseded; its data are
     still in `temp/18-09-2026` if needed.
3. **Table 6.17 on the selected configuration**, if the writing agent moves it off the pilot. Job records: elapsed time per
   flight divided by executed control steps.

   | ms per executed control step | cascaded geometric | MuJoCo MPC |
   | :-- | --: | --: |
   | the control loop | 14.0 | 135.8 |
   | the planner | 8.9 | 10.4 |
   | the controller and the simulator step | 5.2 | 125.4 |

   - Pilot: 94.9 / 88.4 / 6.5 against 215.6 / 89.8 / 125.8.
   - The planner is 1.5 ms slower under MPC: the same network runs in the `FMPCC_mjx` environment with MJX sharing the GPU.
   - First flight 15.1 / 138.1 ms, medians 14.6 / 135.8: no compilation outlier.
   - Projected plans: the whole loop costs 159.4 (cascaded) against 348.6 ms (MPC) per step. The planner's share does not
     split cleanly there: the projection's time doubles under MPC (157.1 → 319.6 ms), most likely because the CPU-side
     solve and MJX compete for the node. The controller's own cost is therefore read on the unprojected pair only, as the
     chapter already does.

## 8b · The finish line at the end of the walls (code change, 24-09)

The eval now scores the s-curve at the end of its walls, x = 3.0 m, like the corridor: `SCENE_CLEAR_LINE_X`, changelog
`logs_in_develop/Gen15/U19/CHANGELOG_20260924_U19_scurve_finish_line_coding3.md`. All 160 flights of the 16 cells were
re-scored from their flown paths (`scurve_r44_raw.py` §6b): **0 flights change**, so every count in this DA and in
Tables 6.14–6.16 holds under both lines. The one flight within 3 cm of the line without crossing (per-step random,
cascaded, flight 7, x 2.9932) inverted, so it is unsafe under any line.

**Supporting fact for §4:** the demonstrations fly a Z-shaped route through the crossover, with its crossing leg straight
up the gap's middle at x = 0, 0.50 m from both corners. Their design notes record why a line toward the corners is
infeasible: "the diagonal (−0.5, y1) → (+0.5, y2) passes 0.291 m from both corners — inside the 0.31 m rotor reach on the
nominal path alone" (`uav_expert_data_collect/trajectories.py`, `s_curve_scene_path`, U7 C1).

The learned plans cross the gap **late**. Their commanded paths cross y = 0 at x = 0.212 m (FM nfe 1; flights
0.18–0.24), 0.231 m (CI-MeanFM nfe 1), 0.266 m (MeanFM nfe 1) and 0.236 m (FM nfe 20), instead of at x = 0. That shift of
about 0.2 m toward the second corner is the corner cut measured in §4.

## 9 · Caveats

- One training seed, ten flights of one route. Counts only (no tests, thesis rule).
- ms figures are cluster latency: compare between configurations only.
- FLAW §B: FM, the selected model, is sampled from half-scale initial noise. This does not touch the controller comparison
  (the same plans under both controllers); it touches any ranking of FM above the average-velocity models.
- Cross-track is measured against the polyline of the commanded setpoints, sampled every third step for the flight means
  and every step for p95 and max.

Claude (Opus 5.5, Claude Code, R44 run chat) · 2026-09-24.
