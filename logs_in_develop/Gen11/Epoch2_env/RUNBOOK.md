# Gen11 Epoch 2 — Runbook

How to execute, how to verify success, how to diagnose failure. Read top-to-bottom in order.

---

## 1. Submit (in order, on cluster)

```bash
cd /path/to/FM-PCC   # repo root (where d3il/ lives)

# Gate: must pass before anything else
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh smoke

# Then, one at a time, or all together:
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh A         # hover
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh B         # step
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh C 6D      # circle, position+velocity
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh C 9D      # circle, position+velocity+accel
# OR all-in-one:
sbatch Slurm_Codes/sbatch/uav_naive/run_naive.sh all
```

Submit from the repo root so `$SLURM_SUBMIT_DIR` resolves correctly.

---

## 2. What to expect (success path)

### 2.1 `smoke` (Phase α — model loads)

**Stdout in SLURM log should contain:**
```
OK nq=7 nv=6 nu=4 qpos_z=0.0517
[ smoke ] body_mass(x2) total = 0.6450 kg   (or similar; depends on Menagerie version)
[ smoke ] gravity = [ 0. 0. -9.81 ]
[ smoke ] timestep = 0.002
```

- `nq=7`, `nv=6`, `nu=4` are **load-bearing** — these are the free-joint quadrotor's signature.
- `qpos_z ≈ 0.052 m`: drone fell from `0.1 m` start over 100 ticks × ~2 ms ≈ 0.2 s under gravity. The exact value depends on `timestep`; anything between `0.04` and `0.07` is fine.
- If timestep is different from 0.002 s, the falling distance scales as `½·g·(N·dt)²`.

### 2.2 Task A (hover at `(0, 0, 0.5)` for 5 s)

**Look in** `logs_in_develop/Gen11/Epoch2_env/results/task_A_hover/metrics.txt`:

| Metric | Expected | Threshold for pass |
|---|---|---|
| `final_pos_err_m` | 0.000 – 0.020 | < 0.02 |
| `mean_pos_err_m` | 0.00 – 0.10 | (settling included; informational) |
| `rms_pos_err_m` | 0.00 – 0.10 | < 0.10 |
| `max_pos_err_m` | 0.20 – 0.40 | < 0.50 (the initial 0.2 m climb) |

`rollout.gif` should show the drone smoothly climbing from `(0,0,0.3)` to roughly `(0,0,0.5)` and holding.

### 2.3 Task B (step from `(0,0,0.5)` to `(1,0,0.5)` at t=2s)

| Metric | Expected | Threshold for pass |
|---|---|---|
| `final_pos_err_m` | 0.000 – 0.050 | < 0.05 |
| `max_pos_err_m` | ≈ 1.0 (the step magnitude at moment of step) | informational |
| Settling time (read off `log.json`) | < 3 s after t=2 | < 3 s |
| Overshoot beyond `x=1` | < 0.20 m | < 0.20 |

`rollout.gif` should show: hover for 2 s → fly forward to roughly `(1, 0, 0.5)` → hold.

### 2.4 Task C (circle, radius 0.5 m, period 10 s, 30 s total, altitude 0.75 m)

| Metric | Expected (6D) | Expected (9D) | Threshold for pass |
|---|---|---|---|
| `rms_pos_err_m` | 0.05 – 0.30 (phase-lag dominated) | 0.02 – 0.10 (accel FF reduces lag) | < 0.10 in at least one of {6D, 9D} |
| `max_pos_err_m` | < 0.40 | < 0.20 | < 0.40 |
| Tracking shape (in GIF) | slightly lagged but circular | tighter circle, less lag | visible smooth circular motion |

**Trajectory format decision:** if 9D < 6D by a clear margin (~2× or better RMS), lock in **9D `[p, v, a]`** for Epoch 3. If they're comparable, 6D is acceptable (simpler downstream).

---

## 3. How to tell it didn't work

### 3.1 Hard failures (stop, fix before continuing)

| Symptom | Where to look | Cause | Fix |
|---|---|---|---|
| SLURM log: `mesh file not found` or asset path error | Smoke stdout | XML can't resolve mesh path | Apply Epoch 1 §11.5 allowed edit: add `<compiler meshdir="assets" texturedir="assets"/>` near top of `quadrotor.xml` AND `quadrotor_modified.xml`. Resubmit smoke. |
| `qpos_z` is NaN, or very negative (< −1) | Smoke stdout | Model integration explosion (rare with zero ctrl) | Re-fetch Epoch 1 sources; verify `quadrotor_modified.xml` byte-identical to a known good copy. |
| `metrics.txt` says `max_pos_err_m > 5.0` for any task | Task results dir | Controller diverged | Skip to §3.2 (soft failure handling) |
| Slurm job exits within seconds with non-zero status | `tail` the SLURM log | Python import error, missing dep | Check `conda activate FMPCC` succeeded and `mujoco`, `numpy`, `imageio` are installed in the env. |
| No `log.json` produced | Result dir empty | Driver crashed mid-loop | Inspect SLURM log for Python traceback. |

### 3.2 Soft failures (tune gains, retry)

If a task ran to completion but the metrics are bad:

| Symptom | Likely cause | Fix (edit `flight_controller.py:Kp_*` constants) |
|---|---|---|
| Hover never reaches 0.5 m (`final_pos_err > 0.10`) | Position P-gain too low OR mass mis-extracted | Double `Kp_pos[2]`. Check `controller.txt` shows `mass ≈ 0.65 kg`. |
| Hover oscillates with bounded amplitude (RMS > 0.15, max > 0.50) | Position D-gain too low or attitude gain too high | Double `Kd_pos`; if persists, halve `Kp_att`. |
| Step (Task B) overshoots > 30 cm | Position D-gain too low | Double `Kd_pos[0]` (forward direction). |
| Step settles too slowly (> 4 s) | Position P-gain too low | 1.5× `Kp_pos[0]`. |
| Circle (C) phase lag > 90° (drone trails target by quarter-cycle) | FF acceleration not being passed | Verify `--trajectory-format 9D`, check `log.json` `a_des` field is non-zero. |
| Circle RMS > 30 cm in BOTH 6D and 9D | Cascaded PID structurally too weak for this trajectory | Either (a) slow the circle: edit `trajectories.circle` `period=20`, or (b) switch to SE(3) controller (PREP_PLAN §4) — adds ~2 h. |
| Drone yaws / spins uncontrollably | Yaw gain wrong sign or allocation matrix wrong | Inspect `controller.txt` — `allocation M` row 4 should have alternating signs `[-0.0201, +0.0201, +0.0201, -0.0201]` (or all positive — depends on motor naming order; what matters is alternation). |

Max **5 gain-tuning iterations** before considering the controller inadequate and switching to SE(3).

---

## 4. Quick verification commands (run on cluster after a job completes)

```bash
# Were results written?
ls -la logs_in_develop/Gen11/Epoch2_env/results/

# How did the task score?
cat logs_in_develop/Gen11/Epoch2_env/results/task_A_hover/metrics.txt

# Did the controller initialise correctly?
cat logs_in_develop/Gen11/Epoch2_env/results/task_A_hover/controller.txt

# Sanity check that log.json isn't empty
python -c "import json; print(len(json.load(open('logs_in_develop/Gen11/Epoch2_env/results/task_A_hover/log.json'))))"
# expect a number near duration_s / timestep — Task A = 5 / 0.002 ≈ 2500
```

---

## 5. After Phase ζ completes

When all 4 tasks pass:

1. Record which trajectory format won (6D or 9D) in a follow-up changelog.
2. Save the GIFs somewhere outside `temp/` if you want them preserved (the `temp/` and `results/` directories are listed as reversible in the changelog).
3. Greenlight Epoch 3 (obstacle world + DPCC halfspace function).

If only A and B passed but C failed in both formats: still a partial success. Document the cap in a follow-up changelog and decide whether to invest in SE(3) before moving to Epoch 3.

---

## 6. Reversal (if Epoch 2 results unusable)

```bash
rm -rf temp/uav_naive_test
rm -rf Slurm_Codes/sbatch/uav_naive
rm -rf logs_in_develop/Gen11/Epoch2_env/results
```

Repo state then identical to immediately after Epoch 1.
