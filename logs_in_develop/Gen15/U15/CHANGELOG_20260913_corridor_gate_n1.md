# U15 — `corridor_gate_n1`: U14's gate, with `FMPCC_SAFE_EPS_FRAC=1.0`

> ⚠️ **Superseded by rev2** ([`CHANGELOG_20260913_u15r2_slide_and_plots.md`](CHANGELOG_20260913_u15r2_slide_and_plots.md)):
> the gate is now a slide from the top wall, and the eval draws the geometry + a GIF. §1–§6 below
> describe rev1 (job 25728).

**Date:** 2026-09-13 · **Gen:** 15 · **Scope:** config (one entry appended) + one submit script — **no code changed**

## 1. The one change under test

```bash
FMPCC_SAFE_EPS_FRAC=1.0      # default 1e-3 — an existing setting (mix_uav/datasets/normalization.py:33)
```

Everything else is identical to U14 (job 25706): FM checkpoint, DPCC projector, HardFlow, PID
controller, violation scorer, `FMPCC_SAFE_EPS_MODE=scaled`, `diffusion_timestep_threshold: 0.5`,
and the gate geometry (verified byte-identical to `corridor_gate` by parsing both entries).

| file | change |
| :-- | :-- |
| `config/uav_projection.yaml` | + `corridor_gate_n1` (suffix `_hggn`), a copy of `corridor_gate` under a new name so results never pool with U14 |
| `Slurm_Codes/temp_bash/eval_20260913_u15_corridor_gate_n1.sh` | injection submitter (new, gitignored dir → `git add -f`) |
| `temp/geo_demo/make_gate_gif.py` | GIF builder from rollout traces (local tool, not version-controlled) |

## 2. Why this setting — the cause found in U14

From [`U14/DA_20260913`](../U14/DA_20260913_corridor_gate_full_wave.md) §5.3:

1. The corridor expert flies dead straight, so `actions[1]` (Δy) and `actions[2]` (Δz) are constant
   in the data.
2. Fix_16 `scaled` (a **Gen15 addition, not upstream DPCC**) gives a constant channel a width of
   `median(non-constant half-widths) × FRAC` = 0.02188 × 1e-3 = **2.2e-05 m/step**.
3. The DPCC projector solves in normalized units inside a hard box `z ∈ [−5, 5]`
   (`mix_uav/sampling/projection.py:205`, verbatim from `aux_repo/dpcc/.../projection.py:140`).
   On corridor Δy that allows **0.77 mm** of sideways reach per 7-step plan.
4. The gate needs about 100 mm, so no feasible solution exists. SLSQP's last iterate is kept silently,
   the unprojected plan flies, and all 21 U14 cells show an identical path.

Pillars' Δy is non-constant, never reaches rule 2, and gets 1390 mm of reach from the same code.

## 3. What `FRAC=1.0` does

A channel the expert never used gets **the same scale as the action channel it did use**.

| corridor | FRAC = 1e-3 (U14) | FRAC = 1.0 (U15) |
| :-- | --: | --: |
| Δy, Δz half-width | 2.2e-05 m/step | **0.02188 m/step** (= forward-step half-width) |
| projector sideways reach / 7-step plan | 0.77 mm | **766 mm** |
| gate ride needs 8.8e-03 m/step | 399× over | **40% of cap** |

### Why this is not a corridor-specific change

- **One rule, every scene.** It is an existing global setting; no code or geometry is special-cased.
- **The checkpoint stays valid.** A constant channel normalizes to 0 for any eps
  (`normalization.py` docstring), so the generator's training target is unchanged. Only the
  physical scale seen at unnormalize and projection time changes.
- **Physically grounded.** The unused axis gets the scene's own real step size. The rejected
  alternative `FMPCC_SAFE_EPS_MODE=legacy` gives 1.0 m/step, 45× more than any real move in the data.

### ⚠️ Honest scope

- **Pillars is not re-run.** Its Δy is untouched mathematically, but its Δz is also constant and
  would widen 3.1e-05 → 0.031. Any claim spanning corridor and pillars under this setting needs
  pillars re-run under it first.
- **Risk:** generator output noise on Δy/Δz is now scaled by 0.022 m instead of 2.2e-05 m. The
  `diffuser` arm is the check for drift.
- Channels touched in the corridor eval: `actions[1]`, `actions[2]`, `rewards[0]` (unused). No
  observation channel is constant.

## 4. Injection test

| | |
| :-- | :-- |
| variants | `diffuser` (control) · `dpcc-t-bounds_free` (least-constrained PCC) · `hardflow_new` (HF) |
| K | 3 — HardFlow runs no real math at K ≤ 2 (A = 0.5); K = 3 gives 1 genuine step |
| trials | 2 — `homotopies[i % 3]` (`eval_mix_uav.py:2120`): trial 0 = **L** (gate never active), trial 1 = **C** (blocked x ∈ [0.40, 0.83]) |
| env | `FMPCC_SAFE_EPS_FRAC=1.0`, `FMPCC_SAFE_EPS_MODE=scaled`, `UAV_EVAL_HOURS=24` |
| target | < 5 min wall (estimate; HardFlow cost at K = 3 not yet measured on corridor) |

`dpcc-t-bounds_free` is the least-constrained PCC that still enforces the gate. `model_free` would
cut the `p ← act` link, so a projected plan would no longer change the flown command.

**First thing to check in the child log:** `Fix_16 DEGENERATE actions[1] ... eps=2.188e-02`. If it
still reads `2.2e-05`, the setting did not reach the job and the run is void.

## 5. Pass criteria

| # | arm | pass |
| :-- | :-- | :-- |
| 1 | `diffuser` | no drift, no wall/floor contact, L and C paths close to U14's diffuser |
| 2 | `dpcc-t-bounds_free` | C rollout ≥ 86 mm under the gate line at x = 0.40, `collision_free` |
| 3 | `hardflow_new` | same on C |

Criterion 1 failing means the setting is unsafe, independent of 2 and 3.

## 6. GIF

`python3.14 temp/geo_demo/make_gate_gif.py <downloaded corridor_hggn_... folder>` → one GIF per
episode in `<folder>/_gif/`. Top-down corridor with the gate and its drone-inflated boundary. All
three arms fly at the same time with trails, a y-zoomed second panel, and each dot turns red while inside
the gate. Built from the rollout traces, no cluster rendering. Tested on the U14 folder.

## 7. Run record

| job | submitted | status |
| --: | :-- | :-- |
| **25728** | 2026-09-13 (cluster) | DONE — results in `temp/1309/Emf_K3_mpc4_pid_stopgo_T0.5_u7hg/6/`. |

### 7.1 First read of 25728 (2 trials, not a DA)

The setting reached the job. The child log with the eps line was not downloaded, but the path moved:
`dpcc-t-bounds_free` C differs from `diffuser` by **273 mm** in y (U14: ≤ 1 mm).

| arm | L | C | note |
| :-- | :-- | :-- | :-- |
| `diffuser` | clean, success | 24 viol steps, 12.3 cm deep, success | **drifts**: L y −0.122 → −0.058, C 0.008 → 0.075. Criterion 1 not clean |
| `dpcc-t-bounds_free` | 2 viol, success | 16 viol, 9.8 cm deep, **no success** (396 steps, overshoots to y −0.377 after x = 2) | the projector now bends the path, but not cleanly |
| `hardflow_new` (folder `hardflow_sls`) | divergence abort, **inverted** at step 166 | divergence abort, **inverted** at step 174 | HF loses the drone under this setting |

**Verdict:**
- Criterion 1 is borderline, with 6–7 cm drift in the unprojected arm.
- Criterion 2 fails: moved but still violating, and missed the goal.
- Criterion 3 fails: both rollouts flipped.

The mechanism diagnosis (§2) is confirmed, since widening the scale unlocks sideways projection. The
setting is not a clean fix.

**Visual complaint (user):** "0 block" in the plots.
- The eval's per-variant plot draws neither the gate nor the drone's width. The gate does block the
  body (diffuser C, 24 steps, 12 cm).
- A diagonal that visibly crosses the drawn route leaves a gap smaller than the 0.62 m drone in the
  0.90 m corridor. See `temp/geo_demo/U15_block_truth.png`.
