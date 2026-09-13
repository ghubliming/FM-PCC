# DA — U14 `corridor_gate` full wave (job 25706)

**Date:** 2026-09-13 · **Gen:** 15 · **Scene:** `corridor` · **Geo:** `corridor_gate` (`_hgg`)
**Engine:** mf (MeanFlow U-Net) · **K:** 2 / 5 / 10 · **Seed:** 6 · **n:** 10 rollouts per cell (L×4, C×3, R×3)
**Change under test:** [`CHANGELOG_20260912_corridor_gate.md`](CHANGELOG_20260912_corridor_gate.md)

## Verdict

> **The gate bites and the projector does not solve it. The executed path does not move.**
> Across 90 C/R rollouts with geometry enforced (5 projected variants × 3 K × 6), **0 are
> collision-free**, and the largest lateral difference from the unprojected arm is **1.0 mm**,
> which is the logging resolution. The only variant that moves at all is `-tightened` (≤ 3 mm,
> toward L, correct direction), which is 2.5% of the 120 mm detour to the clean L channel.
>
> The failure is independent of K (2 = 5 = 10), of candidate selection (r = c = t), and of the
> action-magnitude cap (`bounds_free` = `dpcc-t`, identical to the millimetre).

| gate | test | result |
| :-- | :-- | :-- |
| 1 | `diffuser` violates | ✅ **PASS** — C 12.7 / R 34.0 violation steps, 0/6 C/R clean at every K |
| 2 | projector solves it (**C/R rollouts**) | ❌ **FAIL** — 0 / 90 collision-free |
| 3 | executed y differs from `diffuser` by > 1 mm | ❌ **FAIL** — max 1.0 mm (log resolution); `-tightened` 3.0 mm |

⚠️ **Gate 2 as written in the changelog was flawed.** It said `collision_free_completed > 0`,
which the L channel satisfies **by construction**. The gate is clear of y = −0.12, so every L
rollout is clean with or without a projector. That is why every cell reads exactly **0.40** = 4/10.
The correct gate 2 counts C/R only, and that is the one reported above.

## 1. Data and provenance

**Input:** the raw per-variant result folders from job 25706: `results.json` ×21, rollout traces
×210, npz, diagnostics. Downloaded to `temp/1309/`. The DA_UAV_v1 batch CSVs were not produced for
this wave. Consequences:

- ⚠️ DA_UAV_v1's data-quality and circuit-breaker checks were **not** run. The same things were
  checked by hand instead: `projection_health.n_tripped_trials = 0` and `divergence.n_aborted_trials = 0`
  in all 21 cells, and all 21 cells have 10/10 rollouts with homotopy labels matching filenames.
- One seed, n = 10. Every result below is an **exact tie or an exact zero**, not a mean difference,
  so the verdict does not depend on sample size. Any small violation-count difference between arms
  is not interpreted as an effect (see §4).
- No diffusion K20 baseline in this wave. This is a projector-mechanics test, not an engine-ladder
  claim, so the `da-target` comparison does not apply.

| folder in `temp/1309/` | K | identified by |
| :-- | :-- | :-- |
| `corridor_hgg_…obstacles` | 2 | `fm_ms` ≈ 18 (no K in the folder name) |
| `K5` | 5 | `fm_ms` ≈ 45 |
| `corridor_hgg_…obstacles_K10` | 10 | `fm_ms` ≈ 90 |

**Extraction:** `temp/1309/_da/extract.py` → `temp/1309/_da/u14_extract.json` (stdlib only, not versioned).

### 1.1 Validation of the trace-based analysis

The executed trajectories (`STATE p=(x,y,z)` lines) were re-scored against the gate geometry using a
replica of `_exec_constraint_violations` (`eval_mix_uav.py:770`), and compared with the eval's own
`n_violations`:

- **205/210 exact, 210/210 within one step.** The ±1 comes from the 3-decimal rounding of the logged positions.
- **Every violating step in all 210 rollouts is from the gate.** Walls 0, wall-end caps 0, altitude box 0.

So the traces are a faithful record of what was scored, and the gate is the only constraint being
exercised.

## 2. Outcome — every cell identical

| K | variant | success | S&C | collision-free | n_viol | steps | proj_ms |
| --: | :-- | --: | --: | --: | --: | --: | --: |
| 2 | diffuser | 1.00 | 0.40 | 0.40 | 14.0 | 271.7 | — |
| 2 | dpcc-r | 1.00 | 0.40 | 0.40 | 12.5 | 257.1 | 81.5 |
| 2 | dpcc-c | 1.00 | 0.40 | 0.40 | 12.5 | 259.0 | 88.7 |
| 2 | dpcc-t | 1.00 | 0.40 | 0.40 | 12.9 | 255.7 | 78.1 |
| 2 | dpcc-t-tightened | 1.00 | 0.40 | 0.40 | 13.2 | 243.5 | 179.9 |
| 2 | dpcc-t-geo_free | 1.00 | 0.40 | 0.40 | 14.4 | 261.7 | 11.5 |
| 2 | dpcc-t-bounds_free | 1.00 | 0.40 | 0.40 | 13.0 | 257.3 | 44.7 |
| 5 | diffuser | 1.00 | 0.40 | 0.40 | 14.0 | 269.2 | — |
| 5 | dpcc-r | 1.00 | 0.40 | 0.40 | 12.5 | 260.6 | 251.2 |
| 5 | dpcc-c | 1.00 | 0.40 | 0.40 | 12.4 | 261.0 | 264.2 |
| 5 | dpcc-t | 1.00 | 0.40 | 0.40 | 13.3 | 266.6 | 271.0 |
| 5 | dpcc-t-tightened | 1.00 | 0.40 | 0.40 | 13.3 | 230.4 | 1008.5 |
| 5 | dpcc-t-geo_free | 1.00 | 0.40 | 0.40 | 15.1 | 278.9 | 33.5 |
| 5 | dpcc-t-bounds_free | 1.00 | 0.40 | 0.40 | 13.3 | 265.5 | 213.7 |
| 10 | diffuser | 1.00 | 0.40 | 0.40 | 13.9 | 269.1 | — |
| 10 | dpcc-r | 1.00 | 0.40 | 0.40 | 12.5 | 259.6 | 360.0 |
| 10 | dpcc-c | 1.00 | 0.40 | 0.40 | 12.5 | 260.7 | 364.3 |
| 10 | dpcc-t | 1.00 | 0.40 | 0.40 | 13.2 | 267.8 | 356.9 |
| 10 | dpcc-t-tightened | 1.00 | 0.40 | 0.40 | 13.4 | 229.2 | 1738.0 |
| 10 | dpcc-t-geo_free | 1.00 | 0.40 | 0.40 | 15.2 | 279.4 | 57.9 |
| 10 | dpcc-t-bounds_free | 1.00 | 0.40 | 0.40 | 13.4 | 267.4 | 309.6 |

Every rollout reached the goal and crossed the line. The drone is never physically stopped. The
gate is a scored constraint, not a physical wall in the simulator, so an unsolved gate shows up
as violations with success still 1.00. S&C and collision-free are 0.40 in **all 21 cells**.

### 2.1 Per homotopy — where the 0.40 comes from

| K | variant | L clean | C clean | R clean | C n_viol | R n_viol | flown L\|C\|R |
| --: | :-- | :-: | :-: | :-: | --: | --: | :-- |
| 2 | diffuser | 4/4 | 0/3 | 0/3 | 12.7 | 34.0 | LLLL\|CCC\|RRR |
| 2 | dpcc-t | 4/4 | 0/3 | 0/3 | 12.7 | 30.3 | LLLL\|CCC\|RRR |
| 2 | dpcc-t-bounds_free | 4/4 | 0/3 | 0/3 | 13.0 | 30.3 | LLLL\|CCC\|RRR |
| 5 | diffuser | 4/4 | 0/3 | 0/3 | 12.7 | 34.0 | LLLL\|CCC\|RRR |
| 5 | dpcc-t | 4/4 | 0/3 | 0/3 | 13.3 | 31.0 | LLLL\|CCC\|RRR |
| 10 | diffuser | 4/4 | 0/3 | 0/3 | 12.3 | 34.0 | LLLL\|CCC\|RRR |
| 10 | dpcc-t | 4/4 | 0/3 | 0/3 | 13.3 | 30.7 | LLLL\|CCC\|RRR |

The other four projected variants (`dpcc-r`, `dpcc-c`, `-tightened`, `-geo_free`) have **the same
0/3 C and 0/3 R at every K**. Full table in `u14_extract.json → homotopy`.

**No rollout changed channel.** 0 of 108 projected C/R rollouts (6 variants × 3 K × 6) were flown as a
different homotopy than planned. The clean L channel was 120 mm away and never used.

## 3. Gate 3 — did the path move?

Each projected rollout was compared with the `diffuser` rollout of the **same homotopy and episode
id**, step-aligned over the whole flight and x-aligned inside the gate window (x ∈ [0.40, 2.00]).
C and R rollouts only (n = 6 per cell).

| K | variant | max \|Δy\| (whole flight) | max \|Δy\| (gate window) | mean Δy (gate window) | max \|Δz\| |
| --: | :-- | --: | --: | --: | --: |
| 2 | dpcc-r | 1.0 mm | 1.0 mm | −0.04 mm | 1.0 mm |
| 2 | dpcc-c | 1.0 mm | 1.0 mm | −0.01 mm | 1.0 mm |
| 2 | dpcc-t | 1.0 mm | 1.0 mm | −0.05 mm | 1.0 mm |
| 2 | **dpcc-t-tightened** | **3.0 mm** | **3.0 mm** | **−1.15 mm** | 1.0 mm |
| 2 | dpcc-t-bounds_free | 1.0 mm | 1.0 mm | −0.05 mm | 1.0 mm |
| 2 | dpcc-t-geo_free | 0.0 mm | 0.0 mm | 0.00 mm | 1.0 mm |
| 5 | dpcc-r / -c / -t / -bounds_free | 1.0 mm | 0.0 mm | 0.00 mm | 1.0 mm |
| 5 | **dpcc-t-tightened** | **3.0 mm** | **3.0 mm** | **−1.07 mm** | 1.0 mm |
| 5 | dpcc-t-geo_free | 0.0 mm | 0.0 mm | 0.00 mm | 1.0 mm |
| 10 | dpcc-r / -c / -t / -bounds_free | 1.0 mm | 0.0 mm | 0.00 mm | 1.0 mm |
| 10 | **dpcc-t-tightened** | **3.0 mm** | **3.0 mm** | **−1.08 mm** | 1.0 mm |
| 10 | dpcc-t-geo_free | 0.0 mm | 0.0 mm | 0.00 mm | 1.0 mm |

**Lowest y inside the gate window, dpcc-t vs diffuser (per rollout):**

| K | C (dpcc-t) | C (diffuser) | R (dpcc-t) | R (diffuser) |
| --: | :-- | :-- | :-- | :-- |
| 2 | +0.007, −0.012, −0.027 | +0.008, −0.012, −0.027 | +0.120 ×3 | +0.120 ×3 |
| 5 | +0.008, −0.012, −0.027 | +0.008, −0.012, −0.027 | +0.120 ×3 | +0.120 ×3 |
| 10 | +0.008, −0.012, −0.027 | +0.008, −0.012, −0.027 | +0.120 ×3 | +0.120 ×3 |

**Worst gate penetration:** diffuser C 0.092 / R 0.202 m · dpcc-t C 0.090 / R 0.201 m ·
bounds_free C 0.091 / R 0.202 m. The drone flies just as deep into the blocked side with the projector on.

The R rollouts never leave y = +0.120, the trained channel centre, in any arm. L rollouts (the gate is never
active for them) show 0.0 mm difference, which is what the method should report when nothing is
there to move the path.

## 4. The violation-count differences are not path differences

dpcc-t reads 30.3–31.0 violation steps on R against diffuser's 34.0, which would look like a ~10%
improvement. It is not:

| K | arm | R violation steps | R steps in blocked window x ∈ [0.40, 1.43] | violations per window step |
| --: | :-- | --: | --: | --: |
| 2 | diffuser | 34.0 | 34.0 | **1.000** |
| 2 | dpcc-t | 30.3 | 30.3 | **1.000** |
| 5 | diffuser | 34.0 | 34.0 | **1.000** |
| 5 | dpcc-t | 31.0 | 31.0 | **1.000** |
| 10 | diffuser | 34.0 | 34.0 | **1.000** |
| 10 | dpcc-t | 30.7 | 30.7 | **1.000** |

Every logged step inside the window violates in both arms. The projected arm logs about 3.5 fewer steps
crossing the same stretch of corridor at the same y, so it accumulates fewer violation steps.
It is the same step-count artefact identified in U12 (0.99× per step there). **Violation counts
on this scene must not be read as a projector effect without the trace check.**

`-tightened` also finishes in fewer steps (229–244 vs 267–272). Its 3 mm lateral shift is too
small to explain that. It is logged here as an observation only, not interpreted.

## 5. What this refutes and what it leaves open

### 5.1 The prediction held, but the mechanism I stated is refuted by this run

The changelog §4 predicted failure and gave the cause as the **`action_bounds='auto'` cap**
(Δy ≤ 2.2e-05 m/step). The outcome held. **The mechanism does not survive this run's own
control:** `dpcc-t-bounds_free` removes exactly that constraint family (`eval_mix_uav.py:1259`)
and its executed path is **identical to `dpcc-t` to the millimetre** at every K, with 0/18 C/R clean.
The action-magnitude constraint in the projector is **not** what blocks the detour. The same thing
was already visible in U12 (`bounds_free` 27.70 violations vs 27.10 unprojected, cf 0.000), and
it should not have been re-asserted as the cause afterwards.

What is still true: the corridor dataset's Δy channel **is** degenerate. The eval's own line
(`action_bounds=auto → lb=[1.24e-04 −2.20e-05 −2.20e-05]`) printed again in 25707–25709, and it
differs from pillars (3.97e-02), where the projector does move paths. The **correlation** stands.
The **specific path** it acts through is not the `bounds` constraint. It is the normalized solver box (§5.3).

### 5.2 Ruled out by this run

| candidate blocker | test in this wave | result |
| :-- | :-- | :-- |
| number of ODE steps projected | K = 2 / 5 / 10 | identical |
| candidate selection | dpcc-r / -c / -t | identical |
| action-magnitude cap | `dpcc-t-bounds_free` | identical to dpcc-t |
| surface too close to the feasible boundary | `-tightened` (+0.025 m) | moves 3 mm, still 0/18 |
| constraint family | halfspace (U14) vs sphere (U11–U13) | same null result |

### 5.3 ✅ Cause found — code reading, 2026-09-13 (no new run)

**The projector solves in normalized coordinates with a hard box on every coordinate, and the box
size comes from the dataset's range.**

`mix_uav/sampling/projection.py:205` (verbatim from upstream `aux_repo/dpcc/diffuser/sampling/projection.py:140`):

```python
res = minimize(..., method='SLSQP',
               bounds=Bounds(-5 * np.ones_like(x0), 5 * np.ones_like(x0)), ...)
```

The normalization is `z = 2(x − min)/(max − min) − 1` (`projection.py:598`), so the box z ∈ [−5, 5]
allows a physical action of up to ±2.5 × (max − min) per step. This is hardcoded in the solver call. It is
**not** the `bounds` constraint family, which is why `bounds_free` changed nothing.

| scene | Δy normalizer width | box allows Δy per step | over one 7-step plan |
| :-- | --: | --: | --: |
| **corridor** | 4.4e-05 m | ±1.1e-04 m | **0.77 mm** |
| s_curve | 2.29e-02 m | ±5.7e-02 m | 401 mm |
| pillars | 7.94e-02 m | ±1.99e-01 m | 1390 mm |

The corridor gate needs the plan to move ~100–200 mm sideways within the horizon, and the box allows 0.77 mm.
No solution exists inside it. On non-convergence the projector **keeps SLSQP's last iterate
silently** (`projection.py`, SolverSwap comment: "DPCC itself still silently keeps `res.x` on
non-convergence"), which stays next to the unprojected plan. So the executed path does not move.

**Why every variant fails:** they all share this solver call. r/c/t only choose among candidates;
K only changes which ODE steps get projected; `-tightened` shifts the surface; `bounds_free` removes a
different constraint. None of them touch the box. **Why pillars works with the same algorithm:**
its expert moves sideways, so its box is 1800× wider.

⚠️ The box arithmetic is exact. That the solves actually failed in these runs is inferred from the code
path. The DPCC arm does not log `last_solve_success`, so the failure count was not measured.

**Fix directions (not applied, need a go-ahead):**
- give degenerate channels a physically meaningful normalizer width instead of the Fix_16 epsilon;
- make the solver box physical rather than normalized;
- collect corridor demonstrations that move sideways.

## 6. Standing

- **Corridor cannot demonstrate projection-driven avoidance on this stack.** Five geometries have
  now been tried: sphere r = 0.35 / 0.12 / 0.05 / 0.01 and the slanted halfspace. Across all five
  variants, K and selection rules, the projected executed path equals the unprojected one to log resolution.
- The per-scene contrast is the finding. Pillars, whose dataset has lateral action range, shows
  real avoidance (U12 reference: af 197.70 → 22.90 violations, cf 0.000 → 0.900). Corridor, whose
  dataset has none, shows none.
- **Do not cite any corridor violation-count reduction as a projector effect.** §4 shows the counts
  move while the path does not.
