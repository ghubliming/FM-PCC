# U_13 — Investigation + Plan: deterministic episode length & DPCC-style goal/success termination for UAV eval

**Date:** 2026-07-09. Status: **PLAN ONLY — no code changed yet.**
**Trigger:** UAV corridor eval shows random per-rollout step counts, and rollouts that look
"on track" terminate before the goal. Root cause diagnosed in the Fix_12 discussion; this
doc pins it to code, compares against the **DPCC d3il-avoiding** reference the user wants us
to follow, and lays out the change.

**Goal (user's words):** "all the steps & if stop cross the goal / success_relaxed /
success reached should not be random — follow the d3il-avoiding task used by DPCC."

---

## 1. Root cause (UAV eval today)

The episode length is a **per-trial random draw** the policy never sees:

- `uav_expert_data_collect/generator.py:_build_traj_and_init` samples a random duration per
  scene, per trial:
  - corridor `dur = rng.uniform(6.0, 10.0)` (line 145)
  - pillars  `dur = rng.uniform(10.0, 16.0)` (line 174)
  - s_curve  `dur = rng.uniform(16.0, 22.0)` (line 153)
  - empty    `dur = max(4.0, dist/0.4)` (line 135)
- `FM_v3_uav_test/eval_fm_uav.py:864` → `n_fm = int(round(dur * DATASET_HZ))` (33 Hz), so
  the **step budget itself is random** (corridor: 198–330 steps).
- `eval_fm_uav.py:889` → `for k in range(n_fm):` with **no early termination**. The eval
  docstring (line 30, "SUCCESS_RELAXED (U7)") states this is deliberate: *"episodes never
  terminate early on goal-reach — they always run the full fixed FM-step budget."*
- `success` is checked **only at the final step** (`goal_reached = goal_dist < goal_radius`
  on `p_final`, line 1008/1015).

**Consequence** (see Fix_12 report for the full numeric derivation): the FM policy is
**unconditioned on `dur`** — it flies at ~one learned average speed. So over a *short*-`dur`
budget it covers only part of the corridor and is scored a goal-miss despite flying
perfectly; over a *long*-`dur` budget it reaches the goal and then drifts/overshoots for the
remaining steps. Step count and goal/success outcomes therefore vary trial-to-trial for
reasons unrelated to policy quality — exactly the "randomness" observed.

---

## 2. The reference: how DPCC d3il-avoiding does it

`/workspaces/aux_repo/dpcc/scripts/eval.py` (mirrored by the local `scripts/eval.py`) and
`config/avoiding-d3il.py`:

- **Fixed step budget for ALL trials:** `config/avoiding-d3il.py:68` → `max_episode_length:
  200` (a single constant; `max_path_length: 150` is the separate training-horizon param,
  "longest [expert episode]: 106"). There is **no per-trial random duration**.
- **Deterministic seeding:** `eval.py:180-181` → `torch.manual_seed(i)`, `env_seed = i`.
- **Per-step success from the environment:** `eval.py:241` `success = info['success']`
  (avoiding: `info[1]`, from `avoiding.py:step` → `check_success()` every step, latching
  `self.success`/`self.terminated`).
- **Early termination** (`eval.py:264-268`):
  ```python
  if success or terminated or _ == args.max_episode_length - 1:
      n_steps[i] = _
      if success and collision_free_completed[i]: n_success_and_constraints[i] = 1
      break
  ```
- **`n_steps[i] = _`** — the step at which the episode ACTUALLY ended → a deterministic,
  meaningful *time-to-goal* (DPCC even reports `Avg number of steps` over successes,
  `eval.py:315`).
- **Constraint compliance latched up to the break** (`collision_free_completed`,
  `eval.py:214/221/261`), and success-and-constraints computed at the break.

### Key properties to replicate
| Property | DPCC avoiding | UAV eval now |
|---|---|---|
| Step budget | **fixed** (`max_episode_length`), same every trial | **random** (`round(dur·33)`), per trial |
| Goal/success check | **every step**, from env | **final step only**, computed in eval |
| Early stop on success | **yes** (`break`) | **no** (U7: always run full budget) |
| `n_steps` meaning | time-to-goal (deterministic) | == the random budget (uninformative) |
| Seeding | deterministic per trial | deterministic per trial ✓ (already) |

---

## 3. Plan (implement in a later fix under this folder)

Adopt the DPCC avoiding loop shape in `rollout_one` / `_run_variant`, keeping UAV specifics
(3D flight, PID/MJPC tracker, per-step behaviour log). Concretely:

### 3.1 Fixed per-scene step budget (replaces random `n_fm`)
- Stop deriving the loop budget from the per-trial `dur`. Introduce a **fixed
  `max_episode_length` per scene**, sized from the scene's MAX expert duration plus a margin
  so a slower-than-expert policy can still finish:
  `max_steps = ceil(SCENE_MAX_DUR · DATASET_HZ · SAFETY)`, `SAFETY ≈ 1.5`.
  - corridor (max dur 10 s): ~495 steps
  - pillars (16 s): ~792
  - s_curve (22 s): ~1089
  - empty (random goal): keep a fixed generous budget too (e.g. from its max plausible dist)
- Put these in one place (a `SCENE_MAX_EPISODE_LENGTH` dict next to `SCENES`, or a yaml
  `max_episode_length` per scene, mirroring `config/avoiding-d3il.py`). CLI override
  (`--max-episode-length`) like DPCC's `args.max_episode_length`.
- **`dur` is still sampled** — but used ONLY to compute the fixed `goal` endpoint and the
  finish-line direction (both are already `dur`-independent in value for goal-path scenes:
  `traverse_line`/`pillar_path`/`s_curve` all return the fixed route endpoint at `t≥T`), and
  the initial pose. The *budget* no longer depends on it. (Optionally compute `goal` from a
  large `t` to drop the `dur`→goal coupling entirely.)

### 3.2 Per-step goal check + early termination (DPCC `break` logic)
- Inside the physics-decimation inner loop we already compute per-step position; compute
  `goal_reached_now = ||p - goal|| < goal_radius` each FM step and **latch** it (this is what
  `crossed_line` already does at eval_fm_uav.py:881-883 — reuse/extend it).
- After each FM step, replicate DPCC:
  ```
  if success_now or crashed or k == max_steps - 1:
      n_steps = k; break
  ```
  where `success_now = goal_reached_now and safe_so_far` (scene-aware, matching the current
  `success` definition), and `crashed` is an optional hard-failure terminate (see 3.4).
- Record **`n_steps = k`** at the break → deterministic time-to-goal, same semantics as
  DPCC `n_steps[i]`.

### 3.3 Reconcile with the U7 "never terminate early" decision
- U7 kept episodes running so `success` (final-position) wouldn't be corrupted by a drone
  that reached the goal then drifted, and added `success_relaxed`/`crossed_line` as the
  "ever crossed the finish line" latch. **Early termination on goal-reach makes that
  corruption impossible by construction** (we stop AT the goal), so:
  - `success` (strict) becomes "reached goal within budget AND safe up to that point" —
    cleaner than the current final-only check, and matches DPCC.
  - `success_relaxed`/`crossed_line` is retained for backward-compat but now nearly
    coincides with strict success (still useful for the "grazed the line but stopped just
    outside `goal_radius`" case). Keep both keys; document the semantic shift.
- Update the docstring block at `eval_fm_uav.py:19-33` to describe the new
  fixed-budget + early-stop semantics (and note it now follows DPCC avoiding).

### 3.4 Safety / constraint accounting up to the break (mirror DPCC)
- `safe = contact_free AND airborne` should be measured over the **actual flown steps up to
  the break** (as now, just fewer steps). Contact fraction denominator = executed physics
  steps up to break — already correct since counters accumulate in-loop.
- Optional hard-failure terminate (DPCC's `terminated`): break early on an unrecoverable
  crash (e.g. `min_z` below floor, or contact fraction already over the scene limit). Keep
  this OPTIONAL and off by default in the first cut to avoid changing `contact_frac`
  statistics; add behind a flag if wanted.

### 3.5 Metrics / artifacts / downstream
- `n_fm_steps` in the result dict (eval_fm_uav.py:1095) becomes the real stop step; NPZ
  `n_steps` (eval_artifacts.save_npz) inherits it — good, since the DA pipeline already
  reports it (`Data_Analysis/DA_Visual_Aligning/*`, `DA_Code_v3/data_loader.py`,
  DPCC-style "avg number of steps"). No schema change; values just become meaningful.
- Behaviour log / GIF / foresight already iterate over whatever steps ran → automatically
  shorter, no change needed.
- `pos_tracking_errors`-style fixed-width arrays: UAV uses lists, not fixed-width — no
  `max_episode_length-1` preallocation issue (DPCC preallocates; UAV appends). Confirm.

### 3.6 Determinism guarantee
- With a fixed budget + per-step latched goal check + deterministic `trial_seed = 10_000+i`,
  `n_steps`, `crossed_line`, `success_relaxed`, and `success` become **fully deterministic
  functions of (checkpoint, trial index, variant)** — no dependence on a random `dur`.
  This is the property the user asked for.

---

## 4. Files this will touch (later)
- `FM_v3_uav_test/eval_fm_uav.py` — `rollout_one` loop (fixed budget, per-step goal check,
  early break, `n_steps`), docstring 19-33; possibly `_run_variant` / arg parsing for
  `--max-episode-length`.
- `config/uav_projection.yaml` or a scene constant in `eval_fm_uav.py` /
  `uav_expert_data_collect/generator.py` — `max_episode_length` per scene (mirror
  `config/avoiding-d3il.py:68`).
- **No change** to `generator._build_traj_and_init`'s `dur` sampling itself (still used for
  goal/init) — unless we choose to also make `goal` fully `dur`-independent.
- Changelog in this folder when implemented.

## 5. Open decisions for the author
1. **`SAFETY` multiplier / exact per-scene budgets** — 1.5× max-expert-dur, or a hand-set
   constant per scene? (DPCC just hand-set 200.)
2. **Hard-failure early terminate (3.4)** — include now or defer? (Changes `contact_frac`
   denominator distribution.)
3. **Keep `success_relaxed`/`crossed_line`** as-is for compat, or collapse into strict
   success now that early-stop removes the drift-after-arrival case?
4. Budget as **yaml** (`max_episode_length`, DPCC-faithful) vs **code constant** — yaml is
   more consistent with the avoiding config the user pointed to.

## 6. Verification (on cluster — no Python in Docker)
- `py_compile` locally.
- Re-run corridor eval; confirm: (a) all trials now show the SAME max budget when they miss,
  and a *deterministic, distance-proportional* `n_steps` when they reach; (b) rollouts that
  were "on track but cut short" now reach the goal; (c) `n_steps`/`success`/`crossed_line`
  identical across two runs of the same seed.

## References
- DPCC eval loop: `/workspaces/aux_repo/dpcc/scripts/eval.py:179-268`; budget
  `config/avoiding-d3il.py:68`; env success `d3il/.../gym_avoiding/envs/avoiding.py:168-246`.
- UAV diagnosis: `logs_in_develop/Gen11/Epoch9_PCC_Constraints/Fix_12/REPORT_fix12_*` (Q on
  step-count randomness) and the Fix_12 changelog.
