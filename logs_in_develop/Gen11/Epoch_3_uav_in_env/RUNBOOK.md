# Gen11 Epoch 3 — Runbook

How to execute, how to verify success, how to diagnose failure.
Mirrors Epoch 2's RUNBOOK in structure. Read top-to-bottom in order.

---

## 1. Submit (in order, on cluster)

```bash
cd /path/to/FM-PCC   # repo root (where d3il/ + uav_env_test/ live)

# Gate: must pass before anything else
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh smoke_empty

# Must-pass: prove the controller is scene-agnostic (PLAN §8 items 1-4)
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh empty C

# Scope-expansion: obstacle scenes (PLAN §8 item 5)
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh corridor traverse
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh s_curve s_curve
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh pillars weave

# Optional: also smoke-load the obstacle scenes
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh smoke_corridor
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh smoke_s_curve
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh smoke_pillars

# All-in-one (4 default scene/task pairings):
sbatch Slurm_Codes/sbatch/uav_env/run_env.sh all
```

Submit from the repo root so `$SLURM_SUBMIT_DIR` resolves correctly. Each
result lands in `logs/uav_env/<scene>_<task_label>/`.

---

## 2. What to expect (success path)

### 2.1 `smoke_empty` (gate)

**Stdout in SLURM log should contain:**
```
OK scene=empty  nq=7 nv=6 nu=4  qpos_z=0.0510
[ smoke ] body_mass(x2) total = 1.3250 kg
[ smoke ] gravity = [ 0. 0. -9.81 ]
[ smoke ] timestep = 0.01
[ smoke ] ngeom = 2   (drone body + floor plane)
[ smoke ] ncon at end = 1   (drone resting on floor)
```

- `nq=7 nv=6 nu=4` — load-bearing; same as Epoch 2.
- `qpos_z ≈ 0.04-0.10` — drone fell from 0.1 m and is now resting on the floor (its body sits a few cm above z=0 due to drone geometry).
- `ngeom > 1` — the scene added at least the floor plane (Epoch 2 had `ngeom=1`).
- `ncon > 0` at end — drone is in contact with the floor; confirms collision detection works.

### 2.2 `empty C` (must-pass — scene-agnostic controller proof)

**Look in** `logs/uav_env/empty_task_C_circle_9D/metrics.txt`:

| Metric | Expected | Threshold |
|---|---|---|
| `rms_pos_err_m` | 0.02 – 0.06 | **< 0.058** (= 2× Epoch 2's 0.029 m baseline) |
| `final_pos_err_m` | 0.02 – 0.06 | < 0.06 |
| `max_pos_err_m` | < 0.10 | < 0.15 |
| `obstacle_contact_steps` | **0** | 0 |
| `obstacle_contact_fraction` | 0.000 | 0.000 |

`rollout.gif` should now show: **drone + floor (checker grid) + sky gradient**, drone tracing a clean circle at z=0.75. This is the "GIF is finally legible" moment — fixes the Epoch 2 dark-void problem.

**If RMS exceeds 0.058:** the scene is somehow affecting the controller (shouldn't happen — the X2 model is unchanged). Investigate before proceeding to obstacle scenes.

### 2.3 `corridor traverse`

8-second cosine-profile flight from `(-2.5, 0, 0.75)` to `(+2.5, 0, 0.75)` through a 1 m wide × 1.5 m tall corridor.

| Metric | Expected | Pass |
|---|---|---|
| `final_pos_err_m` | < 0.05 | < 0.10 |
| `rms_pos_err_m` | < 0.05 | informational |
| `obstacle_contact_steps` | **0** (drone stays at y=0, walls at y=±0.5) | < 50 |

`rollout.gif`: drone visibly flies down the corridor, exits the far end. Wall contact would show as the drone visibly clipping the wall — should not happen.

### 2.4 `s_curve s_curve`

3-leg path through 2 offset corridor segments, with the lane-change happening in the open space between segments.

Waypoints:
```
(-3.0, -0.8, 0.75) → (-0.5, -0.8, 0.75)    leg 1: traverse seg 1
(-0.5, -0.8, 0.75) → ( 0.5,  0.8, 0.75)    leg 2: shift north in open space
( 0.5,  0.8, 0.75) → ( 3.0,  0.8, 0.75)    leg 3: traverse seg 2
```
15 seconds total, 5 s per leg, zero velocity at every waypoint.

| Metric | Expected | Pass |
|---|---|---|
| `final_pos_err_m` | < 0.10 | < 0.20 |
| `rms_pos_err_m` | < 0.15 | informational |
| `obstacle_contact_steps` | 0 (lane-shift happens in open space at x ∈ [-0.5, 0.5]) | < 100 |

`rollout.gif`: drone enters left corridor, exits, shifts north between segments, enters right corridor. **The visible "S" trace is the demo.**

### 2.5 `pillars weave`

Sinusoidal weave: x sweeps -3→+3 over 10 s; y oscillates as `1.0·sin(2π·t/4)`; constant z=1.0. Pillars at `x ∈ {-2, 0, +2}`, `y ∈ {-0.6, +0.6}`, radius 0.12 m.

| Metric | Expected | Pass |
|---|---|---|
| `final_pos_err_m` | < 0.10 | < 0.20 |
| `rms_pos_err_m` | < 0.20 | informational |
| `obstacle_contact_steps` | **likely > 0** — see analysis below | < 200 |

**Geometry note (may produce small contact):** with weave amplitude 1.0 m and pillars offset y=±0.6 m (radius 0.12 m), the closest the drone path gets to a pillar is **~0.1 m** when crossing x ≈ ±2 at y ≈ 0.5. That's *less* than the pillar radius — meaning a few contact ticks are expected, not a failure. The drone may "graze" the inner edge of a pillar. **Treat any contact count < ~200 as expected demo behavior**, not a controller failure.

If you want a clean no-contact weave, lower the amplitude (`y_amplitude=0.4` in `run_env.py`) — then the drone passes between the two pillar columns rather than around them. Trade-off: looks less impressive.

---

## 3. How to tell it didn't work

### 3.1 Hard failures (stop, fix before continuing)

| Symptom | Where to look | Cause | Fix |
|---|---|---|---|
| Smoke: `ERROR: scene XML not found at .../scene_empty.xml` | SLURM stdout | The new `scenes/` dir didn't sync to cluster | Verify `d3il/environments/d3il/models/mj/robot/quadrotor/scenes/` exists on cluster; re-sync if missing |
| Smoke: `XMLError ... mesh file not found` | SLURM stdout | `<include>` resolved but mesh path inside `quadrotor_modified.xml` didn't | Apply Epoch 1 §11.5 allowed edit (add `<compiler meshdir="assets" texturedir="assets"/>`) — should already be in place |
| Smoke: `qpos_z` is NaN or extremely negative | SLURM stdout | Integration explosion (rare with zero ctrl) | Re-verify Epoch 1 assets unchanged |
| Smoke: `ncon = 0` at end of run | SLURM stdout | Drone passed through the floor — `contype` / `conaffinity` mismatch | Floor geom has `conaffinity` unset → defaults to 1, should collide with drone. Inspect XML if this fires. |
| `empty C` RMS > 0.10 m | metrics.txt | Scene is affecting controller in unexpected way | The X2 model is unchanged — should not happen. Compare against Epoch 2 `task_C_circle_9D/metrics.txt` to confirm regression. |
| Slurm job exits seconds in, non-zero status | `tail` SLURM log | Python import or shell error | Check `conda activate FMPCC` succeeded; verify `uav_env_test/` synced to cluster |
| `log.json` missing or empty | results dir | Driver crashed mid-loop | Inspect SLURM log for Python traceback |

### 3.2 Soft failures (controller behavior different from Epoch 2)

The X2 model is unchanged between Epoch 2 and Epoch 3 — the only difference is the scene around it. Controller behavior should be **identical**. If you see:

| Symptom | Likely cause | Fix |
|---|---|---|
| `empty C` RMS materially worse than Epoch 2's 0.029 | Scene wrapper changed something we didn't intend (extra light affecting collision? gravity overridden?) | `diff scene_empty.xml` against the file in this repo — there should be no `<option>` block, no `gravity=` override |
| Drone visibly tips or oscillates in any scene | Same Epoch-2 hover-instability bug surfacing — but `empty C` should NOT trigger it (the FF-accel masking is intact) | Confirm `--trajectory-format 9D` (the default), not 6D |
| Obstacle scenes never contact obstacles even when geometry says they should | `_is_drone_contact` filter too aggressive — only floor contacts being seen | Inspect `log.json` for `n_obstacle_contacts > 0` at any step; if always 0, the filter is over-broad |

### 3.3 Expected non-failures (treat as information)

| Observation | Interpretation |
|---|---|
| Pillar weave has 50-200 obstacle-contact steps | Expected: weave amplitude > pillar offset (§2.5). Demo behavior. |
| S-curve final_pos_err near 0.15 | Cosine-blend at waypoints settles slowly when leg duration is short. Increase `segment_duration` if you want tighter tracking. |
| Corridor traverse has 0 contacts but RMS jumps near t=4 | Mid-corridor cosine peak velocity moment. Expected. |

---

## 4. Quick verification commands (run on cluster after jobs complete)

```bash
# 1. Did all expected runs produce results?
ls -la logs/uav_env/

# 2. Compare empty-scene circle vs. Epoch 2 baseline
echo "Epoch 2 baseline:"; cat logs/uav_naive/task_C_circle_9D/metrics.txt | grep rms_pos
echo "Epoch 3 empty C:";  cat logs/uav_env/empty_task_C_circle_9D/metrics.txt | grep rms_pos
# Both should report rms_pos_err_m within ~2x of each other.

# 3. Read all obstacle-scene metrics at once
for d in corridor_task_traverse s_curve_task_s_curve pillars_task_weave; do
  echo "=== $d ==="; cat logs/uav_env/$d/metrics.txt
done

# 4. Sanity-check that log.json sizes match expected step counts
python -c "
import json
for name, expected_dur in [('empty_task_C_circle_9D', 30.0),
                            ('corridor_task_traverse', 8.0),
                            ('s_curve_task_s_curve', 15.0),
                            ('pillars_task_weave', 10.0)]:
    p = f'logs/uav_env/{name}/log.json'
    n = len(json.load(open(p)))
    print(f'{name}: {n} steps  (expected ~{int(expected_dur/0.01)})')
"
```

---

## 5. After all scenes complete

When the must-pass + scope-expansion runs are done:

1. **Decide closure status** against PLAN §8:
   - 5/5 must-pass items met → close Epoch 3 cleanly.
   - 4/5 (e.g. one obstacle scene fails tracking) → close with documented caveat, like Epoch 2 did.
2. **Preserve GIFs** somewhere outside `logs/uav_env/` if you want them long-term (per `.gitignore:3`, `logs/*` is auto-cleaned).
3. **Write the closure changelog** at `logs_in_develop/Gen11/Epoch_3_uav_in_env/EPOCH3_CLOSURE.md` (mirror Epoch 2's structure).
4. **Greenlight Epoch 4** — DPCC obstacle-avoidance: write `obstacles.py` SDF stub (PLAN §3, deferred from Epoch 3), wire halfspaces into the projector.

---

## 6. Reversal (if Epoch 3 results unusable)

```bash
rm -rf uav_env_test
rm -rf Slurm_Codes/sbatch/uav_env
rm -rf d3il/environments/d3il/models/mj/robot/quadrotor/scenes
rm -rf logs/uav_env
```

Repo state then identical to immediately after Epoch 2 closure. Epoch 2
itself (`uav_naive_test/`, `Slurm_Codes/sbatch/uav_naive/`,
`logs/uav_naive/`) is **not touched** by this reversal — those stay
intact.
