# DA — Gen15 UAV Mix-ML: s_curve three-way K-sweep (`fm` / `mf` / `diffusion`)

**Date:** 2026-08-27
**Data root:** `temp/2808/` — train jobs 25007/25008/25009 (`temp/2808/2026-08-24/`), eval jobs 25072/25073/25075/25077–25084 (`temp/2808/2026-08-25/`), batch DA `temp/2808/batch_uav_20260827_161320/`
**Git rev of the runs:** training `7111fb2`; eval `6c2df73` (25072/25073/25075), `30dd65a` (25077/25078/25079), `94ea2d1` (25080–25084)
**Scene:** `s_curve` (hardest; `max_episode_length = 871`) · **Seed:** 6 (single seed) · **n_trials:** 5 per variant
**Arms:** `fm` K∈{1,2,5,10,20} · `mf` K∈{1,2,5,10,20} · `diffusion` K=20 (train-time)
**Protocol:** `mpc_batch = 4`, `controller = pid_stopgo`, `threshold = 0.5`, `UAV_MIX_HF_OFF=1` (HardFlow arm disabled), geo tag `s_curve_bounds+dynamics+geo_bounds+halfspace+obstacles`
**Status:** 10 of 11 eval jobs completed cleanly; job 25075 (`diffusion`) hit the 24 h wall on variant 20/20

---

## 0. TL;DR

**Nothing solves s_curve.** Across **220 variant-arms / 1 097 rollouts**, exactly **one rollout**
scored success+constraints: `mf` K=2 `geo_free`, 1/5. Every other row in the sweep is **0/5**.

**993 of 1 097 rollouts (90.5 %) ended in a divergence abort.** The abort reasons are
`p_des_runaway` (736) and `inverted` (259) — the drone either flips upside-down or the
free-running `p_des` integrator walks away from the airframe. Median abort lands at step
**248–427 of 871**, i.e. **28–49 % into the episode**. This is a **flight-stability failure, not a
planning failure**: the rollout dies before the goal is ever in play.

**The DPCC projector does not fail open — it fails closed.** On every `mf` arm, full `dpcc-c` is
**perfectly clean** (`n viol = 0.0`, `Σ viol = 0.000`, CF **5/5**) at **every K**, and **never
arrives**: `goal_dist` pins at **3.06–3.08 m**, 5/5 rollouts, K=1 through K=20. The projector keeps
the drone safe by keeping it stuck.

**The DPCC/DDPM baseline is the worst arm on this scene.** `diffusion` K=20 + `dpcc-c`:
**203.0** violating steps, **Σ viol = 131.4**, `goal_dist = 2.34 m`, **5 347.5 ms/step**, and
**88/97 rollouts aborted** — 75 of them `inverted`, a failure mode that dominates *only* on this arm
(`fm`/`mf` are 3–5× more often `p_des_runaway`).

**More NFE makes it worse.** Abort rate rises monotonically with K on both flow arms
(`fm` 79 → 95 %, `mf` 88 → 98 %), abort step falls (`fm` 427 → 327, `mf` 376 → 248), and
goal-line crossings collapse (`fm` 21 → 5 per 100, `mf` 11 → 2 per 100). K=1 is the **best**
budget on s_curve for both.

**What this run is good for.** It is a clean, matched, three-way **negative result** that localises
the blocker to the **controller/scene**, not to the generative engine. It does **not** rank
`fm` vs `mf` vs `diffusion` in any meaningful way — at 0/5 everywhere there is nothing to rank.

---

## 1. What ran

### 1.1 Training — all three completed, same rev

| job | engine | scene | start (UTC) | end (UTC) | wall | final test loss |
|---|---|---|---|---|---|---|
| 25007 | `fm` | s_curve | 2026-08-24 14:42 | 2026-08-24 17:10 | 2 h 28 m | 0.00425 |
| 25008 | `mf` | s_curve | 2026-08-24 17:10 | 2026-08-25 00:55 | 7 h 45 m | 0.93949 (MF composite; `test/raw_mse` 0.60348) |
| 25009 | `diffusion` | s_curve | 2026-08-25 00:55 | 2026-08-25 03:21 | 2 h 26 m | 0.00181 |

All three at `GIT REV 7111fb2`, seed 6, `status completed`. That rev contains all four commits that
invalidated the previous (Aug-16/18) s_curve data — `1ce49201` (checkpoint→`best`), `0f1aa7fc`
(HardFlow batch parity / rename), `1288118a` (divergence-abort guard), `ba361e1e` (diffusion arm).

### 1.2 Eval — 11 jobs, one truncated

| job | engine | K | start (UTC) | end (UTC) | wall | outcome |
|---|---|---|---|---|---|---|
| 25072 | `fm` | 20 | 08-25 20:31 | 08-26 09:32 | 13 h 01 m | completed |
| 25073 | `mf` | 20 | 08-25 21:54 | 08-26 14:39 | 16 h 45 m | completed |
| 25075 | `diffusion` | 20 (plan block) | 08-26 09:32 | 08-27 09:32 | 24 h 00 m | **CANCELLED — TIME LIMIT** |
| 25077 | `fm` | 1 | 08-26 14:39 | 08-26 15:28 | 0 h 48 m | completed |
| 25078 | `fm` | 2 | 08-26 15:28 | 08-26 16:24 | 0 h 56 m | completed |
| 25079 | `fm` | 5 | 08-26 16:24 | 08-26 19:45 | 3 h 21 m | completed |
| 25080 | `fm` | 10 | 08-26 19:45 | 08-27 01:32 | 5 h 47 m | completed |
| 25081 | `mf` | 1 | 08-27 01:33 | 08-27 01:58 | 0 h 25 m | completed |
| 25082 | `mf` | 2 | 08-27 01:58 | 08-27 02:26 | 0 h 28 m | completed |
| 25083 | `mf` | 5 | 08-27 02:26 | 08-27 06:19 | 3 h 52 m | completed |
| 25084 | `mf` | 10 | 08-27 06:19 | 08-27 12:32 | 6 h 13 m | completed |

Jobs serialised on `i6-gpu-1` (each starts as the previous ends). Total ≈ 76 GPU-hours.

**Rev drift is benign.** The evals span three commits, but `30dd65a3` touches only
`Slurm_Codes/sbatch/eval_dpcc_job.sh`, `config/avoiding-d3il.py` and `scripts/eval.py` (the Gen0
baseline path) — no `mix_uav*` file — and `94ea2d12` is documentation only. No `mix_uav` code
changed between the first and last eval job.

---

## 2. Headline — success + constraints (out of 5)

Best S&C achieved by **any** of the 20 projection variants, per arm:

| arm | K | max S&C | max goal_reached | max CF | min `n viol` | min `goal_dist` |
|---|---:|---:|---:|---:|---:|---:|
| `fm` @ unet | 1 | **0/5** | 3/5 | 0/5 | 15.8 | 0.308 |
| `fm` @ unet | 2 | **0/5** | 1/5 | 1/5 | 16.0 | 0.406 |
| `fm` @ unet | 5 | **0/5** | 0/5 | 2/5 | 7.8 | 0.543 |
| `fm` @ unet | 10 | **0/5** | 1/5 | 2/5 | 6.2 | 0.968 |
| `fm` @ unet | 20 | **0/5** | 2/5 | 5/5 | 0.0 | 1.349 |
| `mf` @ unet | 1 | **0/5** | 4/5 | 5/5 | 0.0 | 0.609 |
| `mf` @ unet | 2 | **1/5** | 3/5 | 5/5 | 0.0 | 0.694 |
| `mf` @ unet | 5 | **0/5** | 1/5 | 5/5 | 0.0 | 2.215 |
| `mf` @ unet | 10 | **0/5** | 2/5 | 5/5 | 0.0 | 1.815 |
| `mf` @ unet | 20 | **0/5** | 1/5 | 5/5 | 0.0 | 2.064 |
| `diffusion` @ unet | 20 | **0/5** | 0/5 | 5/5 | 0.0 | 2.025 |

The `max` columns are **taken over different variants** — no single variant achieves several of them
at once. That is the whole story of this scene: *goal* and *constraints* are attained by disjoint
sets of rows.

**The only S&C > 0 row in the sweep:**

| field | value |
|---|---|
| arm | `mf` @ unet, K=2 |
| variant | `geo_free` |
| S&C | **1/5** |
| success / goal_reached | 2/5 · 2/5 |
| CF completed | 1/5 |
| `n viol` / `Σ viol` | 9.6 · 0.713 |
| `goal_dist` | 1.817 m |
| `steps_to_goal` | 605.5 / 871 |
| `avg_time` | 29.5 ms (`fm_ms` 18.1 · `proj_ms` 11.4) |

`geo_free` disables geometry enforcement **inside the projector**, but `_exec_constraint_violations`
scores the flown path against the **full raw constraint set** regardless of variant — so this 1/5 is
a real, fully-scored success, not an artefact of a relaxed metric.

---

## 3. The dominant failure mode: divergence abort

`1288118a`'s guard is active in every job here. It fires on **993 / 1 097 rollouts (90.5 %)**.

| arm | rollouts | aborted | % | median abort step | p10 | p90 | median / 871 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fm` K=1 | 100 | 79 | 79.0 | 427 | 52 | 518 | 0.49 |
| `fm` K=2 | 100 | 80 | 80.0 | 422 | 61 | 551 | 0.48 |
| `fm` K=5 | 100 | 90 | 90.0 | 371 | 76 | 417 | 0.43 |
| `fm` K=10 | 100 | 91 | 91.0 | 367 | 87 | 416 | 0.42 |
| `fm` K=20 | 100 | 95 | 95.0 | 327 | 7 | 409 | 0.38 |
| `mf` K=1 | 100 | 88 | 88.0 | 376 | 15 | 394 | 0.43 |
| `mf` K=2 | 100 | 90 | 90.0 | 380 | 18 | 398 | 0.44 |
| `mf` K=5 | 100 | 98 | 98.0 | 288 | 22 | 390 | 0.33 |
| `mf` K=10 | 100 | 96 | 96.0 | 287 | 24 | 385 | 0.33 |
| `mf` K=20 | 100 | 98 | 98.0 | 248 | 5 | 394 | 0.28 |
| `diffusion` K=20 | 97 | 88 | 90.7 | 397 | 5 | 544 | 0.46 |

### 3.1 Reasons — the DDPM arm fails differently

| arm | `p_des_runaway` | `inverted` | ratio |
|---|---:|---:|---:|
| `fm` K=1 | 55 | 24 | 2.3 |
| `fm` K=2 | 65 | 15 | 4.3 |
| `fm` K=5 | 63 | 27 | 2.3 |
| `fm` K=10 | 68 | 23 | 3.0 |
| `fm` K=20 | 78 | 17 | 4.6 |
| `mf` K=1 | 72 | 16 | 4.5 |
| `mf` K=2 | 74 | 16 | 4.6 |
| `mf` K=5 | 78 | 20 | 3.9 |
| `mf` K=10 | 78 | 18 | 4.3 |
| `mf` K=20 | 90 | 8 | 11.3 |
| **`diffusion` K=20** | **15** | **75** | **0.20** |
| **total** | **736** | **259** | 2.8 |

Both flow arms are dominated by `p_des_runaway` (`|p_des − p| > 5.0 m` — the commanded point
outruns the airframe). The `diffusion` arm inverts the drone **five times more often than it runs
away**, the only arm with that signature. Typical logged trip:
`inverted: body z-axis · world z = −0.16 < 0`.

`phys_min_z` goes negative on **133 / 1 097** rollouts (min **−0.021 m**) — the airframe clips
through the floor plane during those upsets.

### 3.2 Only 10 of 220 variant-arms never diverged

| engine | K | variant | rollouts | aborts | goal_reached |
|---|---:|---|---:|---:|---:|
| `mf` | 1 | `gradient` | 5 | 0 | 4 |
| `fm` | 1 | `geo_free-bounds_free` | 5 | 0 | 3 |
| `fm` | 1 | `gradient-tightened` | 5 | 0 | 1 |
| `fm` | 1 | `gradient` | 5 | 0 | 0 |
| `fm` | 2 | `geo_free-bounds_free` | 5 | 0 | 0 |
| `fm` | 2 | `gradient` | 5 | 0 | 0 |
| `fm` | 2 | `gradient-tightened` | 5 | 0 | 0 |
| `fm` | 5 | `geo_free` | 5 | 0 | 0 |
| `fm` | 5 | `geo_free-bounds_free` | 5 | 0 | 0 |
| `diffusion` | 20 | `dpcc-t-tightened` ⚠ | 2 | 0 | 0 |

⚠ = partial unit: job 25075 was killed mid-variant, only 2 of 5 rollouts exist. Excluded from every aggregate elsewhere in this document.

Every one of them is a **weak or ablated projector** (`gradient`, `geo_free`, `bounds_free`) at
**K ≤ 5**. **No full `dpcc-*` variant, on any arm, at any K, produced a single non-diverging
rollout** — with the sole exception of the truncated `diffusion` `dpcc-t-tightened` pair.

---

## 4. The projector wall — safe, and stuck at 3.07 m

Matched projector, full constraint set (`dpcc-c`), all 11 arms:

| arm | K | S&C | goal | CF | `n viol` | `Σ viol` | `goal_dist` | avg ms | aborts |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `fm` | 1 | 0/5 | 0/5 | 0/5 | 63.2 | 1.994 | 2.60 | 64.4 | 5/5 |
| `fm` | 2 | 0/5 | 0/5 | 0/5 | 49.6 | 2.270 | 2.79 | 76.4 | 5/5 |
| `fm` | 5 | 0/5 | 0/5 | 0/5 | 16.4 | 3.299 | 3.14 | 651.3 | 5/5 |
| `fm` | 10 | 0/5 | 0/5 | 0/5 | 12.6 | 2.635 | 3.12 | 987.5 | 5/5 |
| `fm` | 20 | 0/5 | 0/5 | 0/5 | 14.4 | 2.869 | 2.99 | 3 046.8 | 5/5 |
| `mf` | 1 | 0/5 | 0/5 | **5/5** | **0.0** | **0.000** | 3.08 | 40.0 | 5/5 |
| `mf` | 2 | 0/5 | 0/5 | **5/5** | **0.0** | **0.000** | 3.07 | 48.3 | 5/5 |
| `mf` | 5 | 0/5 | 0/5 | **5/5** | **0.0** | **0.000** | 3.07 | 1 152.4 | 5/5 |
| `mf` | 10 | 0/5 | 0/5 | **5/5** | **0.0** | **0.000** | 3.06 | 1 776.7 | 5/5 |
| `mf` | 20 | 0/5 | 0/5 | **5/5** | **0.0** | **0.000** | 3.06 | 4 976.6 | 5/5 |
| `diffusion` | 20 | 0/5 | 0/5 | 0/5 | **203.0** | **131.423** | 2.34 | 5 347.5 | 4/5 |

Per-rollout `goal_dist` on the `mf` rows is not merely low-variance, it is **degenerate**:

```
mf K1   3.04 3.07 3.08 3.09 3.10
mf K2   3.05 3.06 3.07 3.08 3.10
mf K5   3.05 3.05 3.06 3.08 3.09
mf K10  3.04 3.04 3.04 3.08 3.10
mf K20  3.02 3.03 3.06 3.08 3.09
```

25 rollouts, five NFE budgets, spread **0.08 m**. The drone is being parked at a fixed obstruction
~3.07 m from the finish. Combined with `n viol = 0.000` and `CF = 5/5`, this is the projector
holding a feasible-but-stationary solution: **constraint satisfaction achieved by not moving.**

`fm` under the same projector gets neither property — 12.6–63.2 violating steps *and* 2.60–3.14 m
short. `diffusion` is worse than both on every column except `goal_dist`, where it is closer only
because it is diverging in a different direction.

---

## 5. Generator alone — `geo_free` (projector enforces no geometry)

Violations are still scored against the **raw** constraint set, so this isolates the generator.

| arm | K | S&C | goal | CF | `n viol` | `Σ viol` | `goal_dist` | avg ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fm` | 1 | 0/5 | 3/5 | 0/5 | 501.4 | 31.275 | 1.67 | 23.8 |
| `fm` | 2 | 0/5 | 0/5 | 0/5 | 621.0 | 82.868 | 0.86 | 33.8 |
| `fm` | 5 | 0/5 | 0/5 | 0/5 | 674.4 | 131.901 | **0.55** | 89.8 |
| `fm` | 10 | 0/5 | 1/5 | 0/5 | 258.0 | 50.674 | 0.97 | 163.6 |
| `fm` | 20 | 0/5 | 2/5 | 0/5 | 21.0 | 2.719 | 1.81 | 281.4 |
| `mf` | 1 | 0/5 | 2/5 | 0/5 | 20.4 | 1.871 | 1.63 | 20.7 |
| `mf` | 2 | **1/5** | 2/5 | 1/5 | 9.6 | 0.713 | 1.82 | 29.5 |
| `mf` | 5 | 0/5 | 1/5 | 0/5 | 17.6 | 1.328 | 2.22 | 78.0 |
| `mf` | 10 | 0/5 | 1/5 | 0/5 | 14.2 | 0.831 | 1.86 | 147.6 |
| `mf` | 20 | 0/5 | 1/5 | 0/5 | 17.0 | 2.088 | 2.11 | 330.8 |
| `diffusion` | 20 | 0/5 | 0/5 | 0/5 | 86.0 | 20.563 | 2.28 | 318.0 |

The two flow arms fail on **opposite** axes. `fm` drives the drone close — down to **0.55 m** at
K=5 — by **flying through the geometry**: 674.4 violating steps out of a 871-step episode, i.e.
**77 % of the flight is inside an obstacle**, `Σ viol = 131.9`. `mf` keeps penetration two orders of
magnitude lower (`Σ viol` 0.71–2.09) but stalls 1.6–2.2 m out. Neither combination is a solution;
`mf` K=2 is the only row where the two curves brush past each other for one rollout out of five.

---

## 6. Least-bad row per arm (ranked by `goal_crossed_line`, ties → fewer violating steps)

`goal_crossed_line` = past the finish half-plane **or** inside the 0.30 m ball.

| arm | best variant | crossed | goal | CF | S&C | `n viol` | `Σ viol` | `goal_dist` | avg ms |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fm` K=1 | `gradient-tightened` | **5/5** | 1/5 | 0/5 | 0/5 | 454.4 | 40.629 | 0.46 | 12.7 |
| `fm` K=2 | `gradient-tightened` | **5/5** | 0/5 | 0/5 | 0/5 | 297.2 | 21.700 | 0.42 | 22.4 |
| `fm` K=5 | `bounds_free-tightened` | 0/5 | 0/5 | 0/5 | 0/5 | 7.8 | 0.938 | 3.11 | 276.8 |
| `fm` K=10 | `geo_free` | 2/5 | 1/5 | 0/5 | 0/5 | 258.0 | 50.674 | 0.97 | 163.6 |
| `fm` K=20 | `geo_free-bounds_free` | 3/5 | 2/5 | 0/5 | 0/5 | 60.4 | 8.182 | 1.35 | 261.4 |
| `mf` K=1 | `gradient-tightened` | 4/5 | **4/5** | 0/5 | 0/5 | **31.2** | **1.460** | 0.61 | **10.6** |
| `mf` K=2 | `gradient-tightened` | 4/5 | 3/5 | 0/5 | 0/5 | 131.4 | 20.754 | 0.69 | 19.5 |
| `mf` K=5 | `geo_free-bounds_free` | 1/5 | 1/5 | 0/5 | 0/5 | 9.0 | 0.689 | 2.32 | 71.9 |
| `mf` K=10 | `geo_free-bounds_free` | 2/5 | 2/5 | 0/5 | 0/5 | 11.8 | 0.673 | 1.81 | 138.2 |
| `mf` K=20 | `geo_free` | 1/5 | 1/5 | 0/5 | 0/5 | 17.0 | 2.088 | 2.11 | 330.8 |
| `diffusion` K=20 | `model_free` | 2/5 | 0/5 | 0/5 | 0/5 | 197.8 | 46.267 | 2.03 | 374.7 |

**`mf` K=1 `gradient-tightened`** is the closest thing to a working configuration anywhere in the
sweep: 4/5 goal, 31.2 violating steps, `Σ viol = 1.460`, 0.61 m mean final distance, **10.6 ms/step**,
1/5 divergence aborts. It still scores **S&C 0/5**, because CF is 0/5 — every rollout clips
geometry somewhere.

`fm` K=1/K=2 `gradient-tightened` cross the line **5/5** — but with 297–454 violating steps
(34–52 % of the episode inside geometry), so the crossings are worthless.

---

## 7. K is anti-monotone on this scene

| arm | K=1 | K=2 | K=5 | K=10 | K=20 |
|---|---:|---:|---:|---:|---:|
| `fm` — goal-line crossings / 100 | **21** | 13 | 0 | 4 | 5 |
| `fm` — abort rate % | **79.0** | 80.0 | 90.0 | 91.0 | 95.0 |
| `fm` — median abort step | **427** | 422 | 371 | 367 | 327 |
| `mf` — goal-line crossings / 100 | **11** | 10 | 2 | 4 | 2 |
| `mf` — abort rate % | **88.0** | 90.0 | 98.0 | 96.0 | 98.0 |
| `mf` — median abort step | 376 | **380** | 288 | 287 | 248 |

Every column moves the wrong way as K rises. On corridor the K axis buys success; on s_curve it
buys **earlier divergence**. The cheapest budget is also the best one on all three measures, for
both engines. This inverts the Gen15 premise that K is the axis worth sweeping here — on s_curve
the ordering carries no information about generator quality, only about how fast the controller
destabilises.

---

## 8. Cost

Per-step wall time on the matched `dpcc-c` row, split generator / projector:

| arm | K | `fm_ms` | `proj_ms` | total ms | proj share |
|---|---:|---:|---:|---:|---:|
| `fm` | 1 | 10.9 | 53.4 | 64.4 | 83.0 % |
| `fm` | 2 | 20.8 | 55.6 | 76.4 | 72.8 % |
| `fm` | 5 | 52.2 | 599.2 | 651.3 | 92.0 % |
| `fm` | 10 | 85.4 | 902.1 | 987.5 | 91.4 % |
| `fm` | 20 | 169.4 | 2 877.4 | 3 046.8 | 94.4 % |
| `mf` | 1 | 9.3 | 30.7 | 40.0 | 76.7 % |
| `mf` | 2 | 18.1 | 30.2 | 48.3 | 62.4 % |
| `mf` | 5 | 44.6 | 1 107.8 | 1 152.4 | 96.1 % |
| `mf` | 10 | 91.8 | 1 684.9 | 1 776.7 | 94.8 % |
| `mf` | 20 | 211.9 | 4 764.6 | 4 976.6 | 95.7 % |
| `diffusion` | 20 | 176.8 | 5 170.7 | 5 347.5 | 96.7 % |

Two things to note.

**The projector, not the generator, is the cost.** 62–97 % of wall time is `proj_ms` on every row.

**`proj_ms` is not K-independent, and the jump is at K=2→5.** `fm`: 55.6 → 599.2 ms (**10.8×** for a
2.5× K increase). `mf`: 30.2 → 1 107.8 ms (**36.7×**). The NLP does not change size with K, so this
is the solver working harder — higher-K trajectories land further outside the feasible set and cost
more IPOPT iterations to project back. Since abort rate *also* jumps at K=5 (§7), the two are likely
the same phenomenon seen from opposite ends.

Generator-only cost (`geo_free-bounds_free`, dynamics projection only) is well-behaved and roughly
linear in K: `fm` 21.6 → 261.4 ms, `mf` 18.5 → 311.3 ms, `diffusion` 278.7 ms at K=20.

---

## 9. Data-quality notes

**Clean.** Over the 298 s_curve data-quality records: `n_cb_tripped = 0`, `cb_tripped_rate = 0.0`,
`cb_skipped_steps = 0`, `cb_trips = 0`, `backstop_hits = 0`, `cb_sentinel = 0` — no projection
callback fired anywhere. `timing_missing` on 2 records. All Gen15 rows carry
`n_diagnostics_json = 5`, matching `n_trials = 5`.

**Three exclusions applied to every table above.**

1. **`mf` K=10 carries 3 stale HardFlow rows** — `hardflow_new` (`n_rollouts = 3`), `hardflow_new-c`
   (3), `hardflow_new-t` (1) — left over from the Aug-15 run in the same folder. `UAV_MIX_HF_OFF=1`
   means no HardFlow ran in *this* sweep. Dropped; `mf` K=10 is 20 variants like every other arm.
2. **`diffusion` `dpcc-t-tightened` is partial** (`n_rollouts = 2`, job killed mid-variant). Reported
   only in §3.2 with the ⚠ marker; excluded from every aggregate.
3. **`candidates_per_variant.csv` ships duplicated rows** — deduped on `Candidate|variant|geo`
   before any aggregation.

**The `diffusion` arm is 97 rollouts, not 100.** 19 complete variants × 5 + 2. Since all 19 complete
variants score 0/5 S&C, the missing 3 rollouts cannot change any conclusion drawn here.

**Single seed.** Seed 6 only, 5 rollouts per variant. At S&C = 0/5 nearly everywhere the sample size
is more than adequate to establish *failure*; it would not be adequate to establish success.

---

## 10. Verdict against the benchmark hierarchy

| claim | verdict |
|---|---|
| Does any arm solve s_curve? | **No.** 1 successful rollout in 1 097 (0.09 %). |
| Does FM beat the diffusion-DPCC baseline? | **Unresolvable at this operating point.** Both are 0/5 S&C on every variant. `fm` is cheaper (3 046.8 vs 5 347.5 ms on `dpcc-c`) and aborts less on the `inverted` mode, but a comparison between two arms that both score zero is not a win. |
| Do MF/AF beat naive FM? | **Not tested for AF** (no `af` s_curve checkpoint exists). `mf` vs `fm`: `mf` holds violations 1–2 orders of magnitude lower under every projector and owns the only S&C > 0 row, `fm` gets physically closer to the goal. **Non-dominated — a trade-off, not a win.** |
| Does HardFlow beat the DPCC projector? | **Not tested.** `UAV_MIX_HF_OFF=1` on all 11 jobs; the `x_active` NLP-rebuild bug in `mix_uav/sampling/hardflow_projection.py` is still open (`:200` builds the NLP once in `__init__`, so `eval_mix_uav.py:1178-1179`'s `rebuild_projector` is a no-op). |
| Is K the right axis on s_curve? | **No.** Anti-monotone on abort rate, abort step and goal-line crossings for both flow arms (§7). |
| Is the blocker the generator? | **No.** 90.5 % of rollouts die to `p_des_runaway` / `inverted` before the goal is in play, and the full DPCC projector produces a degenerate 3.07 m stall independent of which generator feeds it (§4). The blocker is downstream of the engine. |

---

## 11. What to run next

Ranked by what actually unblocks the scene.

1. **Fix the controller before re-running anything on s_curve.** `p_des_runaway` is 736 of 993
   aborts. The `pid_stopgo` `p_des` integrator free-runs; s_curve's tighter turns let the commanded
   point outrun the airframe past the 5.0 m guard. Candidates: clamp `|p_des − p|` at the
   integrator, or re-anchor `p_des` to the current position each MPC step
   (`pid_stopgo_anchorP` already exists and was used by Gen11 candidate 25).
2. **Diagnose the 3.07 m stall directly.** One `mf` K=1 `dpcc-c` rollout, `--record gif`, and read
   which constraint is active where the drone parks. If it is a halfspace, this is the same
   `x_active` pathology that blocks HardFlow, and it needs fixing in the DPCC projector too — not
   just in `hardflow_projection.py`.
3. **Re-run the `diffusion` arm.** Job 25075 lost variant 20/20 to the 24 h wall at
   5 347.5 ms/step. Either raise `--time` beyond 24 h (needs splitting the job) or cut the variant
   list; at ~1.5 h/variant the full 20 does not fit.
4. **Do not spend GPU on an `af` s_curve run until 1–2 land.** The corridor result
   (`DA_20260824_af_sit_K_sweep_corridor.md`) says `af@sit` wins on generator quality; nothing in
   this sweep suggests generator quality is what s_curve is gated on.
5. **Prefer the low-K end if s_curve is re-run.** K=1 is the best budget for both flow arms on every
   measure here, and it is 10–50× cheaper per step than K=20.

---

## 12. One-line summary

On s_curve, all three Gen15 arms score **0/5 success+constraints** on **219 of 220 variant-arms**;
**90.5 %** of rollouts diverge before reaching the goal (**736** `p_des_runaway`, **259**
`inverted`), the full DPCC projector parks every `mf` rollout at a **3.06–3.08 m** stall with
**zero** violations, and **raising K makes all of it worse** — so the scene is gated on the
controller and the projector, not on the generative engine.
