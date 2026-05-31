# Gen11 Epoch 2 — Naive Fly Test: Changelog (code drop, pre-execution)

**Date**: 2026-05-30
**Branch**: `update_into_FM`
**Scope**: Code only — controller, trajectories, driver, SLURM wrapper. Phases 2-α through 2-ε.
**Predecessor**: [`../Epoch1_UAV_model/CHANGELOG.md`](../Epoch1_UAV_model/CHANGELOG.md)
**Plans**: [`PREP_PLAN.md`](PREP_PLAN.md) (architecture), [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md) (step-by-step).
**Runtime status**: ⏭ **Not yet executed on cluster** — Phases 2-ζ (run + iterate) and 2-η (close) pending user Slurm submission.

---

## Files Created

| File | Lines | Why |
|---|---|---|
| `uav_naive_test/__init__.py` | 0 | Marks the directory as a Python package; permits relative imports from the driver. |
| `uav_naive_test/smoke_load.py` | 39 | Phase 2-α: minimal "does the Epoch 1 XML load on Slurm" check. Prints `nq/nv/nu/qpos_z` + mass + gravity + timestep. This is the runtime check that local Docker couldn't run (no Python in container per project convention). |
| `uav_naive_test/flight_controller.py` | 143 | Phase 2-β: cascaded PID flight controller (Lee/Mellinger SO(3) structure). Position PD + feed-forward → desired body-z + yaw → SO(3) attitude PD + gyroscopic comp → motor allocation via 4×4 matrix built from `model.site_pos` and `model.actuator_gear`. Zero hardcoded geometry — every constant comes from the loaded model. |
| `uav_naive_test/trajectories.py` | 59 | Phase 2-γ: hand-coded reference trajectories. Three factories (`hover_at`, `step_to`, `circle`) returning `traj(t) → (p, v, a, yaw)`. Pure functions, no MuJoCo dependency. |
| `uav_naive_test/run_naive.py` | 205 | Phase 2-δ: driver. Parses CLI, loads X2 from Epoch 1's `quadrotor_modified.xml`, sets initial state, builds the controller, runs the sim loop at MuJoCo's native timestep, logs per-step state/target/control to JSON, writes metrics (final/mean/RMS/max position error), optionally renders a GIF via `mujoco.Renderer` with the X2's `track` camera. |
| `uav_naive_test/README.md` | — | One-pager pointing at PREP_PLAN + EXECUTION_PLAN and listing the Slurm commands to run each task. |
| `Slurm_Codes/sbatch/uav_naive/run_naive.sh` | 98 | Phase 2-ε: SLURM wrapper. 1 GPU, 30 min walltime, EGL + PYOPENGL_PLATFORM exported before Python starts. Repo root resolved via `$SLURM_SUBMIT_DIR` + upward marker-dir search (same idiom as `visual_avoiding/collect_visual_avoiding.sh`, which we know works post-Fix-1). Dispatches on `$1 ∈ {smoke, A, B, C, all}` and `$2 ∈ {6D, 9D}`. |

## Files Modified

**None.** Epoch 2 is purely additive. No edits to `d3il/`, `config/`, `fm_visual_aligning/`, `diffuser_visual_aligning/`, or any existing SLURM script.

## Files Deleted

**None.**

---

## Code Discipline (per EXECUTION_PLAN §1)

| Rule | Compliance |
|---|---|
| All files ≤ 300 lines | ✅ Largest is `run_naive.py` at 205 |
| No LLM-synthesized library code | ✅ Cascaded PID structure follows the well-known Lee 2010 / Mellinger 2011 design and was hand-typed against that spec — no vendored library |
| No edits to existing source files | ✅ Inventory above is all "Created", zero "Modified" |
| Syntax-checked before commit | ✅ `bash -n run_naive.sh` clean; `ast.parse` clean on all 4 Python files |
| Reversible | ✅ `rm -rf uav_naive_test Slurm_Codes/sbatch/uav_naive logs_in_develop/Gen11/Epoch2_env/results` fully undoes Epoch 2 |

---

## What Was NOT Done (Per Plan)

- **No execution on cluster yet.** Phases 2-ζ (gain tuning iteration loop) and 2-η (closure / format decision) pending Slurm submission.
- **No D3IL env class.** No `MjQuadrotor`, no `gym_quadrotor_env/`. Driver loads the raw MuJoCo model directly.
- **No FM-PCC integration.** No `config/`, no training script, no eval script touched.
- **No learned policy.** Trajectories are deterministic hand-coded functions.
- **No obstacle world / DPCC plumbing.** That is Epoch 3.

---

## Phase Status (per EXECUTION_PLAN §3)

| Phase | Deliverable | Code Status | Runtime Status |
|---|---|---|---|
| 2-α | `smoke_load.py` | ✅ Written | ⏭ Pending Slurm |
| 2-β | `flight_controller.py` | ✅ Written, syntax-checked | ⏭ Pending Slurm |
| 2-γ | `trajectories.py` | ✅ Written, syntax-checked | ⏭ Pending Slurm |
| 2-δ | `run_naive.py` | ✅ Written, syntax-checked | ⏭ Pending Slurm |
| 2-ε | `Slurm_Codes/sbatch/uav_naive/run_naive.sh` | ✅ Written, `bash -n` clean, chmod +x | ⏭ Pending Slurm |
| 2-ζ | Run + iterate on cluster | — | ⏭ User action |
| 2-η | Closure changelog + format decision | — | ⏭ Pending 2-ζ results |

---

## How to Execute (User on Cluster)

Submit in this order. **Each later step assumes the previous one passed.**

```bash
cd /path/to/FM-PCC
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh smoke    # Phase 2-α: must pass first
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh A        # hover
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh B        # step
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh C 6D     # circle, 6-D traj
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh C 9D     # circle, 9-D traj
# or all-in-one:
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh all
```

Results land in `logs_in_develop/Gen11/Epoch2_env/results/<task_label>/`:
- `log.json` — per-step state, target, control
- `metrics.txt` — final/mean/RMS/max position error
- `rollout.gif` — visual sanity GIF
- `controller.txt` — PID gains and allocation matrix dump

**If `smoke` fails on mesh path resolution:** apply Epoch 1 §11.5 allowed edit (add `<compiler meshdir="assets" texturedir="assets"/>` to the XML), then re-submit.

**If hover (Task A) diverges:** see EXECUTION_PLAN §5 fallbacks. Most likely cause is gain miscalibration; the `controller.txt` dump tells you the starting numbers.

---

## How to Reverse Epoch 2

```bash
rm -rf uav_naive_test
rm -rf Slurm_Codes/sbatch/uav_naive
rm -rf logs_in_develop/Gen11/Epoch2_env/results
```

Repository state then identical to immediately after Epoch 1.
