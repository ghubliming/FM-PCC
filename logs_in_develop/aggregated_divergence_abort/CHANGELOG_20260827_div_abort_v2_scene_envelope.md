# Div_Abort v2 — abort on the ROBOT leaving its scene, not on the commanded point

**Date:** 2026-08-27
**Supersedes:** the trigger table in `CHANGELOG_20260823_divergence_abort_and_plot_clamp.md` §3.1.
Everything else in that changelog (logging surfaces, sentinels, plot clamp, the aligning-side
guard) is unchanged.
**Scope:** BOTH families the 08-23 commit `1288118a` touched — UAV (`mix_uav_test/` Gen15 +
`FM_v3_uav_test/` Gen11) **and** visual aligning (`fm_visual_aligning_test/` Gen7,
`mix_visual_aligning_test/` Gen14, `imf_visual_aligning_test/` Gen3v4,
`diffuser_visual_aligning_test/` Gen9). Same bug, same fix, fixed together.
**Status:** written and syntax-checked in Docker — **NOT run** (no numpy in this container).
Needs a cluster run, see §5.

---

## 1. The report

> "delete it and find a way to do the abort I want ie stop when the UAV flying very fast and
> into abnormal position for each uav scene"
>
> "the small svg is because the uav flying out of the scene normal range/scope, so if it flying
> too far from pillar/walls, or has suddenly speed it is sign it is failed! then abort will
> solve the small svg problem"

Trigger for the report: a `s_curve` / `diffuser` run in which **5/5 trials aborted at FM step
23–24** with `reason=p_des_runaway`, while the drone itself was at `z ≈ 0.73–0.92 m`, doing
1–2 m/s, well inside the arena.

## 2. Why `p_des_runaway` was wrong

The v1 check was `|p_des - p| > 5.0 m` — the **commanded-point lead**, which is
**direction-blind**. The same 5 m means three different things:

| gap direction | what the PID does (`Kp_pos = [4, 4, 8]`) | really lost? |
|---|---|---|
| `p_des` 5 m **below** `p` | `a_cmd_z = -42 m/s²` → `F_world_z < 0`. A quadrotor cannot pull negative thrust: `T` clamps to `thrust_floor = 0.1·m·g` (free fall) and `b3_des` points down, so the attitude loop commands a **flip**. | yes |
| `p_des` 5 m **above** `p` | `F_world_z = m(42 + 9.81) > 0`. Thrust saturates at `u_max = 2·u_hover`; the drone climbs at max acceleration and **catches up**. | **no** |
| `p_des` 5 m **sideways** | `a_cmd_xy = -20 m/s²` → a 64° tilt. Aggressive, upright, still flying. | **no** |

v1 aborted all three identically, on a quantity that is not the aircraft: `p_des` is a
free-running integrator, and a large lead is a *symptom* the flight may be lost, never a
measurement that it is.

### 2.1 In fairness to v1 — the record, so nobody re-litigates this

Measured over the 644 pre-guard rollouts still on disk (corridor / pillars / s_curve, ~20
projection variants, one controller `pid_stopgo`): **v1 would have aborted 0 of 84 strict
successes.** Highest `max |p_des - p|` among successes was 2.76 m against the 5.0 m threshold
(1.81x margin); only 2 of 106 goal-reaching rollouts crossed it and both were already
`safe=False, strict=False`; it fired on 408 of 536 failures. The `s_curve` aborts that prompted
this rewrite were also CORRECT calls — that drone lands at step ~74 and never moves again.

So v1's problem was never its observed hit rate. It was that **zero observed false positives is
not zero risk when the mechanism is unsound**: the archive only ever contained DOWNWARD
runaways, the one direction where the lead really does imply saturation. An upward lead
(recoverable climb) or a sideways lead (64° tilt, still flying) would have been killed
identically, and neither is represented in that data. `pid_const_v`, which feeds a large
`v_des` into the PID and so produces a bigger sustained lead by design, is entirely untested
against it.

Note the uncomfortable corollary for v2: **v1's threshold has more evidence behind it than
v2's do.** v2's numbers come from scene geometry and free-fall arithmetic — better reasoning,
but zero measurement. See §5.2 and §6.5.

## 3. What changed

`p_des` is **gone from the guard entirely**. `_check_divergence()` reads `p`, `v` and the
attitude quaternion — nothing else. `_divergence_arena(geo_config)` → `_flight_envelope(scene)`.

| tag | condition | default |
|---|---|---|
| `nan_state` | non-finite `p` / `v` | — |
| `off_map` | `\|x\|,\|y\| > 10 m`, `z > 10 m`, or `z < -0.5 m` — the drone has left the world | MuJoCo floor plane is 10×10 m |
| `off_route` | `p` outside its scene's flight envelope ⊕ slack — too far from the walls/pillars the route is defined by | slack **2.0 m** |
| `overspeed` | `\|v\|` above cap | **6.0 m/s** |
| `inverted` | body z-axis · world z < 0 | — |
| ~~`p_des_runaway`~~ | **DELETED** | — |

**`off_route` and `overspeed` fire INDEPENDENTLY (OR, not AND).** Either one alone is already
a sign the rollout has failed, and either one alone already wrecks the SVG. An earlier draft
required both; that was wrong for the same reason it would have been safe — it could not fire
on a slow drift far off-route, which is exactly the "flying out of the scene's normal scope"
case the report is about.

### 3.1 The per-scene flight envelope

`SCENE_FLIGHT_ENVELOPE` — the box containing every **expert** trajectory of that scene. Read
off `d3il/environments/d3il/models/mj/robot/quadrotor/scenes/scene_<scene>.xml` (floor plane
±10 m, walls 1.5 m tall) and `uav_expert_data_collect/generator.py` (altitude drawn
`U(0.90, 1.30)` at the start, `U(0.70, 1.10)` at the goal):

| scene | lb | ub | source |
|---|---|---|---|
| `empty` | (-1.8, -1.80, 0.70) | (1.8, 1.80, 1.30) | start/goal drawn in `U(-1.8, 1.8)` |
| `corridor` | (-2.8, -0.45, 0.70) | (2.8, 0.45, 1.30) | path spans x = ±2.8; wall inner faces y = ∓0.45 |
| `pillars` | (-3.2, -1.11, 0.70) | (3.2, 1.11, 1.30) | path spans x = ±3.2; outer channel y = ±1.11 |
| `s_curve` | (-3.2, -1.25, 0.70) | (3.2, 1.25, 1.30) | path spans x = ±3.2; corridor band \|y\| ≤ 1.25 |

An unknown scene falls back to the union of the four, so a new scene can never abort
spuriously before someone measures its envelope and adds a row.

**Deliberately NOT `geo_config['workspace_bounds']`.** That box shrinks per geo ablation
(`geo_bounds_only`, tightened `combined_*`), so keying the guard off it would let the same
flight abort under one variant and survive under another. The envelope is a fixed physical
property of the scene, identical across every projection variant.

### 3.2 Why these numbers

* **Slack 2.0 m** — wider than the entire corridor/s_curve wall gap. Puts the ceiling trigger
  at `z = 3.30 m`: 1.8 m clear of the tallest wall (1.5 m) and 2.0 m above any expert altitude.
  For `s_curve` the live box is `(-5.2, -3.25, -1.30) .. (5.2, 3.25, 3.30)`.
* **Speed 6.0 m/s** — the expert covers ≤ 8 m of path in 6–22 s (`generator.py:145/153/174`),
  i.e. ~0.4–0.9 m/s mean and well under 2 m/s peak. Because this now fires *alone*, it is set
  above every speed the arena can produce innocently: a free fall from the top of the altitude
  draw (1.30 m) lands at `sqrt(2·9.81·1.30) = 5.05 m/s`, so 6.0 m/s cannot be reached by
  merely dropping out of cruise — it takes powered divergence. (v1's 12 m/s was never derived
  from anything.)

### 3.3 Env overrides

`FMPCC_UAV_DIVERGENCE_ABORT=0` (disable all), `FMPCC_UAV_DIV_SLACK_M`,
`FMPCC_UAV_DIV_SPEED_MS`, `FMPCC_UAV_DIV_MAP_XY_M`, `FMPCC_UAV_DIV_MAP_Z_M`.
`FMPCC_UAV_DIV_LEAD_M` is **removed** — it no longer does anything.

`divergence['thresholds']` in `results.json` changes shape:
`{envelope_slack_m, speed_max_ms, map_xy_m, map_z_m}` (was `{arena_slack_m, speed_max_ms,
p_des_lead_m}`). `divergence['p_des']` / `['p_des_lead']` are **kept as diagnostics** — the
foresight SVG still draws the dotted leader to the commanded point — they are simply no longer
triggers.

## 4. What this does for the small-SVG problem

Two different causes, two different fixes, and they are not interchangeable:

* **`p_des` runs away, drone stays in-scene.** Handled entirely by the **view clamp**
  (`eval_artifacts.view_window()`, 08-23 §3.3): every panel scales to the FLOWN PATH (2–98 pct)
  plus enforced geometry, and `p_des` may widen it by at most `VIEW_MAX_GROW = 1.0` core span.
  A commanded point at −600 m cannot shrink a plot. No abort needed.
* **The DRONE leaves the scene.** The clamp cannot fix this — the flown path *is* the core the
  window scales to, so a fly-away sets a huge percentile band and the real flight compresses
  anyway. **Only the abort fixes this**, by ending the episode on the step the aircraft is lost
  so the core band stays on the arena. This is what `off_route` / `overspeed` are for.

### 4.1 Behaviour on the reported run

`s_curve` / `diffuser`, `p = [-3.18, -0.81, 0.73]`, `|v| = 2.10 m/s`:
`off_map` no · `off_route` no (`p` is inside `(-5.2,-3.25,-1.30)..(5.2,3.25,3.30)`) ·
`overspeed` no (`2.10 < 6.0`) · `inverted` no. → **no abort**; all 5 trials fly the full episode.

**Known and accepted gap.** The actual `s_curve` failure is a *downward* runaway: the drone
falls, lands around step ~74, and then sits at `z ≈ 0.31`, `|v| ≈ 0`, upright, in-envelope, for
the remaining ~800 steps (confirmed against a pre-guard archive run, 10/10 seeds). **No trigger
in this table fires on that** — it is in-scene and stationary. It burns the full budget and is
scored a miss on its own merits (`safe=False`, `reached=False`). Its SVG stays readable because
of the view clamp, so nothing is broken; only compute is wasted. If that becomes worth
reclaiming, the right addition is a separate opt-in `grounded` trigger (`z < 0.35 m` and
`|v| < 0.05 m/s` sustained for N steps), NOT a return to a `p_des` term.

## 5. Files touched / what still needs a cluster run

```
mix_uav_test/eval_mix_uav.py       guard block rewritten; call sites + thresholds dict
FM_v3_uav_test/eval_fm_uav.py      same edits — guard block verified byte-identical
mix_uav_test/eval_artifacts.py     stale reason-tag list in the npz comment
FM_v3_uav_test/eval_artifacts.py   same (siblings still byte-identical)
```

No CLI flags, no sbatch change (the guard is default-on and env-tuned only).

1. **The abort code path has still never executed on the cluster.** The 08-23 §6 items were
   never run; the `s_curve` job that produced this report was its first real firing, and it
   fired on the trigger that has now been deleted. Both a true-positive and a false-positive
   run are still outstanding.
2. **False positive — `overspeed` is the one to check.** It now fires alone, so it is the only
   threshold that can end a healthy rollout on its own. Run one healthy variant per scene and
   confirm `divergence_aborts : 0/N` and no `DIVERGENCE_ABORT.txt`; if anything comes close to
   6 m/s, raise `FMPCC_UAV_DIV_SPEED_MS`. `off_route` is bounded by scene geometry and is much
   harder to trip innocently.
3. **True positive.** Force one (`FMPCC_UAV_DIV_SPEED_MS=0.5` on a healthy variant makes
   `overspeed` reachable) and confirm the ✖ marker, banner and `DIVERGENCE_ABORT.txt` land.
4. **Metric drift.** Any UAV run made between 2026-08-23 and today may carry
   `p_des_runaway`-truncated per-step counts (`n_violations`, `contact_frac`, `track_err_mean`,
   `goal.dist`). Those are **not** comparable to pre-08-23 or post-08-27 numbers. Re-run rather
   than mix.

## 6. Visual aligning — the same bug, same fix

Commit `1288118a` shipped this guard to BOTH families, so the aligning side carried the same
defect and is corrected in the same pass. **Two** checks were command-based there, not one:

| deleted tag | what it was | why it goes |
|---|---|---|
| `des_runaway` | `\|des_c_pos - c_pos\|_xy > 0.25 m` | the direct twin of `p_des_runaway` — a direction-blind lead check on a free-running integrator |
| `des_out_of_arena` | `des_c_pos` outside the arena box | the same signal as a bound instead of a lead. It bounds the COMMANDED point, which is not the robot either |

`des_runaway`'s 0.25 m was also the threshold §6.1 of the 08-23 changelog flagged as "the one
to watch" and asked to be checked against `outcome.max_physical_tracking_error` — that check
was never run.

### 6.1 New aligning trigger table

| tag | condition | default |
|---|---|---|
| `nan_state` | non-finite `des_c_pos` / `c_pos` | — |
| `off_table` | `c_pos` outside the physical table box `(-0.30,-1.20,-0.50)..(1.60,1.20,1.50)` | arm has left the workspace |
| `ee_off_route` | `c_pos` outside the aligning TASK envelope ⊕ slack | envelope `(0.20,-0.45,0.02)..(0.80,0.45,0.50)`, slack **0.15 m** |
| `ee_overspeed` | TCP speed (finite difference of `c_pos` over one control step at `RT_CONTROL_HZ`) | **disabled (0)** — see below |
| ~~`des_runaway`~~, ~~`des_out_of_arena`~~ | **DELETED** | — |

`nan_state` still looks at `des_c_pos`: a non-finite command is an unambiguous integrator
blow-up, not a judgement call about control authority.

### 6.2 Why `ee_overspeed` ships DISABLED

The UAV's speed cap is anchored to a hard physical constant — a free fall from the top of the
altitude draw lands at 5.05 m/s, so 6.0 m/s cannot be reached innocently. **The arm has no such
constant**, `max_action_delta` is `null` by default so it provides no cap either, and there is
no measured TCP speed distribution in-repo. Shipping a guessed threshold that fires ALONE is
precisely the mistake `des_runaway` was, so the code is in place and off. To enable: measure
`c_pos_history` from a healthy cluster run, then set `FMPCC_ALIGN_DIV_SPEED_MS=<value>`.

`ee_off_route` carries the aligning guard on its own until then. It is well-grounded: the
envelope is the widest Cartesian surface any shipped geo entry declares (`x∈[0.30,0.70]`,
`y∈[±0.35]`, `z∈[0.05,0.40]` in `config/visual_aligning_eval.yaml`) opened out to the reachable
aligning area, and it is FIXED — deliberately not `geo_config['workspace_bounds']`, which
shrinks per ablation and would make the same motion abort under one variant and survive under
another.

### 6.3 Aligning env overrides / schema

`FMPCC_ALIGN_DIVERGENCE_ABORT=0` (disable all), `FMPCC_ALIGN_DIV_SLACK_M`,
`FMPCC_ALIGN_DIV_SPEED_MS`. **`FMPCC_ALIGN_DIV_LEAD_M` is removed** — it no longer does
anything. `divergence['thresholds']` becomes `{route_slack_m, ee_speed_ms}` (was
`{arena_slack_m, lead_m}`); `arena_lb/ub` become `route_lb/ub` + `table_lb/ub`.
`divergence['des_c_pos']` and `['lead']` are **kept as diagnostics** — the foresight SVG still
draws the leader to the commanded point — they are simply no longer triggers.

`d3il/simulation/aligning_sim.py` is **unchanged**: its break is keyed off
`getattr(agent, 'abort_episode', False)` and knows nothing about reason tags.

### 6.4 Aligning files touched

```
fm_visual_aligning_test/eval_fm_visual_aligning.py        (Gen7)
mix_visual_aligning_test/eval_mix_visual_aligning.py      (Gen14)
imf_visual_aligning_test/eval_imf_visual_aligning.py      (Gen3v4)
diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py (Gen9 / DPCC baseline)
```
Guard block verified byte-identical across all four before and after the edit.

### 6.5 Aligning cluster checks

1. **False positive on `ee_off_route`.** One healthy variant → expect no `DIVERGENCE_ABORT.txt`.
   The live box is `(0.05,-0.60,-0.13)..(0.95,0.60,0.65)`; confirm no healthy `c_pos_history`
   comes near it.
2. **Measure TCP speed** from the same run's `c_pos_history` and pick a real
   `FMPCC_ALIGN_DIV_SPEED_MS` before enabling `ee_overspeed`.
3. **Metric drift.** Any ALIGNING run made between 2026-08-23 and today may carry a
   `des_runaway` / `des_out_of_arena` truncation. Unlike the UAV path the env `success` was
   never overridden, but per-step counts still cover fewer steps. Re-run rather than mix.
