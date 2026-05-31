# Gen11 Epoch 2 — Execution Plan (the "how to do it")

**Date**: 2026-05-30
**Branch**: `update_into_FM`
**Status**: Plan only — no code or SLURM submission yet.
**Companion**: [`PREP_PLAN.md`](PREP_PLAN.md) — the architectural hypothesis & task definitions this executes against.

This document specifies **the concrete steps and file structure** for executing Epoch 2. `PREP_PLAN.md` says *what we're testing and why*; this document says *exactly which files get written, in which order, on which machine, with which verification gates*.

---

## 1. Execution Discipline (same rules as Epoch 1)

Carrying over from Epoch 1 §11.1:

1. **No LLM-synthesized XML or vendored library code.** Anything brought in from upstream (controller references, MuJoCo Python idioms) comes via `cp` from a known source or is hand-typed from a paper / spec we can cite.
2. **Code we write ourselves** (cascaded PID, trajectory authoring, driver) is written fresh, but **kept small and reviewable** (each file <300 lines).
3. **The `Edit` tool is for code edits;** larger structural changes go through `Write` with explicit purpose.
4. **Reversible:** entire Epoch 2 lives under `uav_naive_test/`, `Slurm_Codes/sbatch/uav_naive/`, and `logs_in_develop/Gen11/Epoch2_env/`. `rm -rf` of those three paths fully undoes Epoch 2.

---

## 2. File Inventory To Be Created

Exhaustive list. Anything not on this list is out of scope.

```
uav_naive_test/
├── __init__.py                      ← empty package marker
├── flight_controller.py             ← cascaded PID, ~150 lines
├── trajectories.py                  ← hand-coded Tasks A/B/C, ~100 lines
├── run_naive.py                     ← driver: load X2 + run task + log + render
└── README.md                        ← one-paragraph orientation, references PREP_PLAN

Slurm_Codes/sbatch/uav_naive/
└── run_naive.sh                     ← SLURM wrapper (GPU node, EGL, FMPCC env)

logs_in_develop/Gen11/Epoch2_env/
├── PREP_PLAN.md                     ← already created
├── EXECUTION_PLAN.md                ← this file
├── results/                         ← created at run time
│   ├── task_A_hover/
│   ├── task_B_step/
│   ├── task_C_circle_6D/
│   └── task_C_circle_9D/
└── CHANGELOG.md                     ← written at end of Epoch 2
```

**Files NOT created:**
- No `d3il/.../MjQuadrotor.py` (deferred to a later epoch).
- No `d3il/.../envs/gym_quadrotor_env/` (deferred).
- No `config/` change.
- No edit to any existing source file (the X2 XMLs in `d3il/environments/.../quadrotor/` stay byte-identical to Epoch 1 output).

---

## 3. Step-by-Step Execution Sequence

### Phase 2-α — Cluster smoke load (the deferred Epoch 1 step 5)

Before writing controller / trajectories, prove the patched XML actually loads on Slurm. This is the smoke check that local Docker couldn't run.

**Action:** create a minimal one-file script, submit to Slurm.

`uav_naive_test/smoke_load.py` (~15 lines):
- `import mujoco`
- `mujoco.MjModel.from_xml_path(<path to quadrotor_modified.xml>)`
- `mujoco.MjData(model)`
- 100 × `mj_step` with zero ctrl
- Print `nq`, `nv`, `nu`, final `qpos[2]`.

**Pass condition:** `nq=7, nv=6, nu=4` printed; `qpos[2]` ≈ 0.052 m (drone falls under gravity from 0.1 m start over 0.1 s).

**Fail handling:**
- Asset path resolution error → apply Epoch 1 §11.5 allowed edit: add `<compiler meshdir="assets" texturedir="assets"/>` to the XML.
- Anything else → stop, investigate before proceeding.

**Gate:** do NOT proceed to Phase 2-β until 2-α passes on Slurm.

### Phase 2-β — Cascaded PID controller

**Action:** write `uav_naive_test/flight_controller.py`.

Public API the driver will call:
```
class CascadedPID:
    def __init__(self, model_constants):
        # ingest mass, gravity, inertia, motor geometry from MuJoCo model
        ...

    def compute_thrusts(self,
                        state,            # (p, q, v, ω) from sensors
                        target_p, target_v=None, target_a=None,
                        target_yaw=0.0):
        # returns u in R^4 (motor thrusts)
        ...
```

Internal structure (standard cascade):
1. **Position loop** (P + D on position error) → desired acceleration `a_des` in world frame.
2. **Add gravity compensation** and feedforward `target_a` if provided → required body z-thrust direction.
3. **Attitude reconstruction** — from required thrust direction + desired yaw, compute desired rotation matrix / quaternion.
4. **Attitude loop** (P + D on rotation error) → desired body torques.
5. **Thrust mixing** — given desired (total thrust, roll/pitch/yaw torques), invert the X2 geometry to get individual motor thrusts. Clip to `[0, u_max]`.

Initial gains: hand-tune on Task A. Start with `Kp_pos = [4, 4, 8]`, `Kd_pos = [3, 3, 4]`, `Kp_att = [70, 70, 4]`, `Kd_att = [10, 10, 2]`. These are typical Crazyflie-scale starting points — re-tune for X2's larger mass.

**Pass condition:** standalone — `import flight_controller` succeeds, `CascadedPID(...)` constructs without error using constants pulled from a loaded X2 model.

### Phase 2-γ — Trajectories module

**Action:** write `uav_naive_test/trajectories.py`.

Three trajectory generators, each returning a callable `traj(t) -> (p, v, a, yaw)`:

- `hover_at(point, duration)` → constant position.
- `step_to(p_from, p_to, t_step, duration)` → step function (with optional smoothing if needed).
- `circle(center, radius, period, altitude, duration, fmt='9D')` → parametric circle. `fmt='6D'` returns `(p, v, None, 0)`, `fmt='9D'` returns `(p, v, a, 0)`.

Pure functions, no MuJoCo dependency. Should be unit-checkable by printing values at sample times.

### Phase 2-δ — Driver

**Action:** write `uav_naive_test/run_naive.py`.

CLI:
```
python run_naive.py --task {A|B|C} [--trajectory-format {6D|9D}] [--seed N] [--render]
```

Driver logic per task:
1. Load X2 from `d3il/environments/d3il/models/mj/robot/quadrotor/quadrotor_modified.xml`.
2. Construct `MjData`, set initial pose per task spec.
3. Build trajectory generator + controller.
4. Loop at `dt = 0.002 s` (500 Hz physics) for the task duration:
   - Read sensors → `(p, q, v, ω)`.
   - Sample trajectory at current sim time → `(p_des, v_des, a_des, yaw_des)`.
   - Call controller → 4 thrusts.
   - Write to `data.ctrl[:4]`.
   - `mj_step`.
   - Log state, target, control to a list each step.
5. After loop:
   - Save logs to `logs_in_develop/Gen11/Epoch2_env/results/task_X_*/log.csv` + `.json` (state per step, target per step, control per step).
   - Compute pass-fail metrics per PREP_PLAN §5 / §8.
   - If `--render`, save a GIF/MP4 from a third-person tracking camera.

### Phase 2-ε — SLURM script

**Action:** write `Slurm_Codes/sbatch/uav_naive/run_naive.sh`.

Standard template, mirrors `Slurm_Codes/sbatch/visual_avoiding/collect_visual_avoiding.sh`:
- 1 GPU (for EGL offscreen rendering).
- 1 CPU task, ~8 GB RAM.
- ~30 min walltime (3 tasks × max 30 s sim + render).
- Sets `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`.
- Resolves `$REPO` via `$SLURM_SUBMIT_DIR` walking up to the marker dir (same idiom as the visual_avoiding sbatch).
- Args: `$1 = task (A|B|C|all)`, `$2 = trajectory-format (6D|9D, default 9D, only meaningful for C)`.

Inside: activates `FMPCC` conda env, runs `python uav_naive_test/run_naive.py …` once per task.

### Phase 2-ζ — Run + iterate

Submit:
```
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh A
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh B
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh C 6D
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh C 9D
```

Inspect results:
- Plot `target_p[:, i]` vs actual `p[:, i]` for each axis.
- Check pass-fail metrics from PREP_PLAN §8.
- If Task A overshoots: lower `Kp_pos`, raise `Kd_pos`.
- If Task A oscillates: lower attitude gains.
- If Task C tracking is poor in 6-D but acceptable in 9-D → confirms the feed-forward acceleration term matters → lock in 9-D for Epoch 3.
- If both 6-D and 9-D fail Task C → switch to SE(3) geometric controller. Adds ~2 h.

Limit: max **5 gain-tuning iterations** before considering the cascaded PID inadequate and switching to SE(3).

### Phase 2-η — Closure

When pass conditions are met:
1. Write `logs_in_develop/Gen11/Epoch2_env/CHANGELOG.md` (per Epoch 1's template).
2. Commit selected trajectory format (6-D or 9-D) as a one-line decision recorded in the changelog.
3. State the recommended next epoch (Epoch 3 — obstacle world).

---

## 4. Gate Conditions Between Phases

Don't proceed past a gate without the prior phase's pass condition met.

| Gate | Must be true before next phase |
|---|---|
| α → β | Smoke load on Slurm prints expected `nq/nv/nu` and reasonable `qpos[2]` |
| β → γ | `flight_controller.py` imports clean and constructs `CascadedPID` without error |
| γ → δ | Each trajectory generator returns expected shapes (`p,v,a` all 3-D, yaw scalar) |
| δ → ε | `run_naive.py --task A` runs to completion locally syntax-check (`python -m py_compile`) |
| ε → ζ | `run_naive.sh -c` shell-syntax check passes |
| ζ → η | All 4 SLURM jobs (A, B, C-6D, C-9D) finished with non-error exit codes |

---

## 5. Failure Modes and Fallbacks

| Symptom | Likely cause | Fallback |
|---|---|---|
| Smoke load: `mesh file not found` | Asset path resolution | Add `<compiler meshdir="assets" texturedir="assets"/>` to XML (Epoch 1 §11.5 allowed edit). Re-run smoke load. |
| Smoke load: GLEW / EGL error | Cluster MuJoCo GL backend mismatch | Verify `MUJOCO_GL=egl` exported BEFORE `import mujoco`. If still fails, fall back to `osmesa`. |
| Hover diverges immediately | Wrong sign on gravity comp / wrong motor mixing | Re-check actuator gear values (`±0.0201` yaw torques) and motor site positions. |
| Hover oscillates with bounded amplitude | Gains too high | Halve `Kp_pos` first, then `Kp_att`. |
| Step settles but overshoots > 50 cm | Position D-gain too low | Double `Kd_pos`. |
| Circle: phase lag > 90° between target and actual | No feed-forward velocity (using 3-D target) | Confirm using 6-D or 9-D format; check `target_v` is being passed to controller. |
| Circle tracking RMS > 30 cm in both formats | Cascaded PID structurally inadequate | Switch to SE(3) controller (PREP_PLAN §4). |

---

## 6. Time Budget

Estimated effort, accounting for the iteration loop:

| Phase | Time |
|---|---|
| α — smoke load | 20 min |
| β — cascaded PID | 1.5 h |
| γ — trajectories | 30 min |
| δ — driver | 1 h |
| ε — SLURM script | 30 min |
| ζ — run + iterate (avg 3 iterations) | 1.5 h |
| η — changelog + commit decision | 30 min |
| **Total** | **~5.5 h** |

Buffer to **8 h** if SE(3) controller becomes necessary.

---

## 7. What Success Looks Like at End of Epoch 2

A directory `logs_in_develop/Gen11/Epoch2_env/results/` containing:
- Four sub-directories (Task A, B, C-6D, C-9D) each with `log.csv`, `log.json`, `rollout.gif`, `metrics.txt`.
- Pass/fail status for each task.
- A locked trajectory format choice (6-D or 9-D) recorded in the changelog with the tracking-RMS comparison that motivated it.
- A pointer to Epoch 3 (obstacle world).

That's the entirety of Epoch 2's surface. No new D3IL plumbing, no FM-PCC touches, no learned policy. Just *"can the X2 fly trajectories under a separate controller?"* — answered.

---

## 8. Greenlight Checklist

Before I start writing `flight_controller.py`:

- [ ] PREP_PLAN.md trajectory-format decision (6-D, 9-D, or both) confirmed
- [ ] Controller choice (cascaded PID vs SE(3)) confirmed — default cascaded PID
- [ ] SLURM resource limits confirmed acceptable (1 GPU, ~30 min walltime)
- [ ] OK to use `uav_naive_test/` as the home directory (not `tools/`, not `scripts/`)
- [ ] OK to write the SLURM script under `Slurm_Codes/sbatch/uav_naive/`

Defaults are applied where the user hasn't specified. Ready to execute Phase 2-α on greenlight.
