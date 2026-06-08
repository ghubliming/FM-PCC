# Gen11 Epoch 3 — UAV in Real Env: Changelog (code drop, pre-execution)

**Date**: 2026-05-31
**Branch**: `update_into_FM`
**Scope**: New code + scene XMLs for testing the Epoch 2 cascaded-PID controller inside real MuJoCo scenes (floor + lighting + skybox + obstacles).
**Plan**: [`PLAN.md`](PLAN.md)
**Discipline**: copy from Epoch 2, modify in new dir; **no edits to Epoch 2 files**.
**Runtime status**: ⏭ **Not yet executed on cluster.** Phase 3-γ verification pending Slurm submission.

---

## Files Created

### `uav_env_test/` — new parallel of Epoch 2's `uav_naive_test/`

| File | Lines | Provenance | Purpose |
|---|---|---|---|
| `__init__.py` | 0 | `cp` from `uav_naive_test/` | package marker |
| `flight_controller.py` | 143 | **verbatim `cp`** from `uav_naive_test/` (byte-identical) | Same X2, same dynamics → controller works unchanged. No reason to fork. |
| `trajectories.py` | 143 | `cp` from `uav_naive_test/` (lines 1-59 verbatim) **+ appended 84 lines** | Epoch 2's `hover_at`/`step_to`/`circle` kept intact; new factories `traverse_line` (cosine-profile point-to-point), `s_curve_path` (multi-waypoint), `weave` (sinusoidal y over linear x) appended |
| `smoke_load_env.py` | 58 | `cp` from `smoke_load.py` + scene-path swap | Loads a scene wrapper (default `empty`), drops drone, reports model dims + active contact count. `--scene` arg selects from {empty, corridor, s_curve, pillars}. |
| `run_env.py` | 279 | `cp` from `run_naive.py` + scene dispatch + new tasks + contact logging | Adds `--scene` (required), 3 new tasks (`traverse`, `s_curve`, `weave`), per-step obstacle-contact count (floor contacts ignored), warn-on-touch (never abort). Output → `logs/uav_env/<scene>_<task>/`. |
| `README.md` | — | new | Orientation pointer to PLAN.md + Slurm commands |

### `d3il/environments/d3il/models/mj/robot/quadrotor/scenes/` — new dir

| File | Lines | Why |
|---|---|---|
| `scene_empty.xml` | 22 | Baseline: `<include>` of Epoch 1's `quadrotor_modified.xml` + floor + skybox + 2 lights. Fixes the "drone in a dark void" GIF problem from Epoch 2. |
| `scene_corridor.xml` | 28 | `scene_empty` + 2 parallel wall boxes (4 m × 0.1 m × 1.5 m) with 1 m gap along the x-axis |
| `scene_s_curve.xml` | 32 | `scene_empty` + 2 offset corridor segments forcing a north-shift between legs |
| `scene_pillars.xml` | 33 | `scene_empty` + 6 cylinder pillars (radius 0.12 m, height 2 m) in two staggered columns at x ∈ {-2, 0, 2} |

XML is hand-written from MuJoCo XML reference + Menagerie `scene.xml` template — no LLM model-content synthesis. Each scene `<include>`s the Epoch 1 model verbatim; the scene only adds world bodies.

### `Slurm_Codes/sbatch/uav_env/`

| File | Lines | Why |
|---|---|---|
| `run_env.sh` | 90 | `cp` from `uav_naive/run_naive.sh` + scene-dispatch case. Dispatches on `smoke_<scene>`, `<scene> <task>`, or `all` (runs the 4 default scene/task pairings). Same `$SLURM_SUBMIT_DIR` + marker-walk repo-resolve idiom. Marker updated to `scenes/` dir so Epoch 3 layout is what's expected. |

### `logs_in_develop/Gen11/Epoch_3_uav_in_env/`

| File | Why |
|---|---|
| `PLAN.md` | Strategy + decision points (already present from prior step) |
| `CHANGELOG.md` | This file |

---

## Files Modified

**None.** Epoch 2 (`uav_naive_test/`, `Slurm_Codes/sbatch/uav_naive/`, `logs/uav_naive/`, all Epoch 2 docs) is **bit-for-bit unchanged**.

Verified:
- `diff -q uav_naive_test/flight_controller.py uav_env_test/flight_controller.py` → identical
- `diff uav_naive_test/trajectories.py uav_env_test/trajectories.py` → only appended lines past line 59; Epoch 2's content intact

---

## Files Deleted

**None.**

---

## Discipline Compliance

| Rule | Compliance |
|---|---|
| No touch to Epoch 2 code | ✅ Verified via diff |
| Copy, don't rewrite | ✅ 6 files `cp` from Epoch 2 + small surgical edits; only 4 scene XMLs hand-written from scratch (no Epoch 2 analogue) |
| No LLM-synthesized model XML | ✅ Scene wrappers use straight MuJoCo reference syntax; only `<include>` of Epoch 1's vendored model |
| File size cap ≤ 300 lines | ✅ Largest is `run_env.py` at 279 |
| Python syntax clean | ✅ `ast.parse` on all 4 py files |
| Shell syntax clean | ✅ `bash -n` on run_env.sh |
| XML well-formed | ✅ `ElementTree.parse` on all 4 scene XMLs |
| Reversible | ✅ `rm -rf uav_env_test/ Slurm_Codes/sbatch/uav_env/ d3il/environments/d3il/models/mj/robot/quadrotor/scenes/ logs/uav_env/` undoes Epoch 3 cleanly |

---

## Phase Status (per PLAN §5)

| Phase | Deliverable | Code Status | Runtime Status |
|---|---|---|---|
| 3-α | Copy + syntax check | ✅ Done | — |
| 3-β | `scene_empty.xml` + `smoke_load_env.py` | ✅ Written | ⏭ Pending Slurm |
| 3-γ | Re-run Task C circle 9D in `scene_empty` | ✅ Driver supports | ⏭ Pending Slurm |
| 3-δ | `scene_corridor.xml` + `traverse_line` traj | ✅ Written | ⏭ Pending Slurm |
| 3-ε | `scene_s_curve.xml` + `s_curve_path` traj | ✅ Written | ⏭ Pending Slurm |
| 3-ζ | `scene_pillars.xml` + `weave` traj | ✅ Written | ⏭ Pending Slurm |
| 3-η | `obstacles.py` SDF stub | ⏭ Deferred to Epoch 4 (per PLAN §10 D2) | — |
| 3-θ | Closure changelog | ⏭ Pending 3-γ/δ/ε/ζ results | — |

---

## How to Execute on Cluster

Submit in this order. **Gate on `smoke_empty` before scene runs.**

```bash
cd /path/to/FM-PCC

# Gate
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh smoke_empty

# Must-pass (PLAN §8 items 1-4)
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh empty C

# Scope expansion (item 5)
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh corridor traverse
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh s_curve s_curve
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh pillars weave

# Or all-in-one:
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh all
```

Output directory: `logs/uav_env/<scene>_<task_label>/{log.json, metrics.txt, controller.txt, rollout.gif}`.

`logs/uav_env/` is already gitignored via `logs/*` (line 3 of `.gitignore`).

---

## Expected Results (per PLAN §8 pass criteria)

| Run | Pass condition | Notes |
|---|---|---|
| `smoke_empty` | `nq=7 nv=6 nu=4`, `qpos_z ≈ 0.05-0.15` (settled on floor), `ncon > 0` at end | Confirms scene loads + drone contacts floor |
| `empty C` | RMS ≤ 0.058 m (2× Epoch 2's 0.029) | Proves controller is scene-agnostic |
| `corridor traverse` | Drone visibly traverses corridor, contact count low (ideally 0) | Hand-coded trajectory may or may not clip — that's information |
| `s_curve s_curve` | Drone visibly threads the curve | Same caveat |
| `pillars weave` | Drone visibly weaves the pillar field | Same caveat |

---

## How to Reverse Epoch 3

```bash
rm -rf uav_env_test
rm -rf Slurm_Codes/sbatch/uav_env
rm -rf d3il/environments/d3il/models/mj/robot/quadrotor/scenes
rm -rf logs/uav_env
```

Repository state then identical to immediately after Epoch 2 closure.
