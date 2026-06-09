# Gen11 Epoch4 U4 — Fix Plan

**Date:** 2026-06-09  
**Source:** `U4/EVAL.md` — two hard blockers, one informational flag  
**Status:** Plan only — not yet implemented

---

## Summary of Issues

| ID | Issue | Severity | Scenes |
|---|---|---|---|
| Fix A | s_curve 90.5% rejection — PID inertia overshoot causes floor crash | ❌ Blocking | s_curve |
| Fix B | R_R_R PKLs all corrupt (truncated transfer) | ❌ Blocking | pillars |
| Fix C | End-of-episode speed spikes >2.5 m/s | ℹ️ Informational | pillars |

---

## Fix A — s_curve Inertia Overshoot

### Root Cause (from PKL inspection of episode 0000010)

The s_curve path (`s_curve_scene_path`) has three `traverse_line` segments with zero velocity at each junction:

```
Seg A: (-3.2, y1, z) → (-0.5, y1, z)   pure-x, 2.7 m
Seg B: (-0.5, y1, z) → ( 0.5, y2, z)   diagonal, 1.89 m, Δy = 1.6 m
Seg C: ( 0.5, y2, z) → ( 3.2, y2, z)   pure-x, 2.7 m
```

Direct tracking data from episode 0000010 reveals the failure sequence:

| Dataset step | p_des (x, y) | p_actual (x, y) | err_y | err_z |
|---|---|---|---|---|
| 250 | (-0.51, -0.82) | (-0.51, -0.82) | 0.000 | 0.000 |
| 310 | (-0.32, -0.53) | (-0.33, -0.64) | **-0.106** | -0.003 |
| 330 | (-0.17, -0.29) | (-0.18, -0.47) | **-0.185** | +0.003 |
| 340 | (-0.08, -0.15) | (-0.10, -0.14) | +0.016 | -0.042 |
| 350 | (+0.00, -0.02) | (-0.14, +0.37) | **+0.387** | -0.118 |
| 381 (min_z) | (+0.26, +0.39) | (+0.05, +0.91) | +0.533 | **-0.397** |
| 450 | | | | 0.000 (recovered) |

**Failure mechanism: lag-then-overshoot in y:**

1. **Lag phase (t=300→330):** Seg B begins commanding `+y` acceleration. The drone lags 0.18m behind the commanded y because the PID and physical inertia cannot track the cosine acceleration ramp perfectly. While lagging, the PID applies aggressive `+y` corrective force on top of the trajectory feed-forward.

2. **Overshoot phase (t=330→350):** The cosine profile begins decelerating (commanding `−y` acceleration), but the drone has accumulated `+y` momentum from the correction phase. In 0.6 s the drone swings from y=−0.474 to y=+0.370 — a 0.844 m lateral swing in 0.6 s (1.41 m/s) vs the commanded 0.272 m/0.6 s = 0.45 m/s. Fully uncontrolled overshoot.

3. **Altitude collapse (t=350→381):** The PID must simultaneously correct a 0.53 m y-error AND a 0.40 m z-error. The combined thrust demand plus the required attitude tilt depresses available z-thrust, or the attitude inner loop saturates, causing the drone to fall while the y-error is corrected. Floor crash occurs when z < 0.50 m.

**Why 90.5% of seeds fail:** Starting altitude is `z ~ U(0.70, 1.10)`. The altitude drop during overshoot is consistently 0.30–0.45 m. With z_start = 0.70 m (minimum), the drone hits the 0.50 m floor after a 0.20 m drop. Only seeds with z_start ≥ 0.90 m AND a small overshoot (< 0.40 m drop) survive — roughly 10% of draws.

---

### Fix A Code Change

**Root fix:** Insert explicit hover pauses at the segment junctions `(-0.5, y1, z)` and `(+0.5, y2, z)`. Each hover gives the drone time to stabilize its lateral velocity to near-zero before starting the next segment. This breaks the lag-overshoot cycle entirely: there is no accumulated `+y` momentum to overshoot with if the drone is hovering.

**File:** `uav_expert_data_collect/trajectories.py` — `s_curve_scene_path`

```diff
 def s_curve_scene_path(altitude, duration, y_jitter=0.0, yaw=0.0):
+    """
+    Fix A: add hover_pause at each segment junction to eliminate y-overshoot.
+    Duration budget: 1.0 s per junction (2.0 s total), remainder proportional.
+    """
     z  = float(altitude)
     T  = float(duration)
     y1 = -0.8 + y_jitter
     y2 =  0.8 + y_jitter
 
+    T_HOVER  = 1.0   # seconds to hover at each junction
+    T_move   = T - 2.0 * T_HOVER   # time available for A, B, C
+
     d_a = 2.7
     d_b = float(np.sqrt(1.0**2 + (y2 - y1)**2))   # ≈ 1.89 m when jitter=0
     d_c = 2.7
     d_total = d_a + d_b + d_c

-    t_a = T * d_a / d_total
-    t_b = T * d_b / d_total
-    t_c = T * d_c / d_total
+    t_a = T_move * d_a / d_total
+    t_b = T_move * d_b / d_total
+    t_c = T_move * d_c / d_total
 
     seg_a = traverse_line((-3.2, y1, z), (-0.5, y1, z), t_a, yaw)
+    hov_1 = hover_at((-0.5, y1, z), yaw)    # stabilise before diagonal
     seg_b = traverse_line((-0.5, y1, z), ( 0.5, y2, z), t_b, yaw)
+    hov_2 = hover_at(( 0.5, y2, z), yaw)    # stabilise after diagonal
     seg_c = traverse_line(( 0.5, y2, z), ( 3.2, y2, z), t_c, yaw)

+    t_ends = [t_a, t_a + T_HOVER, t_a + T_HOVER + t_b,
+              t_a + T_HOVER + t_b + T_HOVER]

     def traj(t):
-        if t < t_a:
+        if t < t_ends[0]:
             return seg_a(t)
-        elif t < t_a + t_b:
-            return seg_b(t - t_a)
-        else:
-            return seg_c(t - t_a - t_b)
+        elif t < t_ends[1]:
+            return hov_1(t)
+        elif t < t_ends[2]:
+            return seg_b(t - t_ends[1])
+        elif t < t_ends[3]:
+            return hov_2(t)
+        else:
+            return seg_c(t - t_ends[3])

     return traj
```

**Duration range adjustment in `generator.py`:**

Current: `dur = float(rng.uniform(16.0, 22.0))`

The hover pauses consume 2.0 s. Total episode is 2.0 s longer for the same manoeuvre time. Raise the minimum to keep T_move ≥ 14s:

```diff
-    dur = float(rng.uniform(16.0, 22.0))
+    dur = float(rng.uniform(18.0, 24.0))   # +2 s for the two 1.0 s hover pauses
```

**Expected outcome:**

| Metric | Before Fix A | After Fix A (expected) |
|---|---|---|
| y-lag at junction | ~0.185 m | ~0.01–0.02 m (settled hover) |
| Overshoot | 0.53 m | ~0 (drone is stationary at junction) |
| z-drop during crossing | 0.40 m | < 0.10 m (z-correction focused) |
| Rejection rate | 90.5% → ABORT | Target < 30% |

The two 1.0 s hover windows are visible in GIF inspection as brief pauses at the two tunnel mouths. These are physically correct (a human pilot would also pause to align before the crossing). The data is still valid for training.

---

## Fix B — Pillars R_R_R Corrupt PKLs

### Root Cause

All 121 R_R_R PKLs are truncated on read (`EOFError: Ran out of input`). L_L_L, L_R_L, R_L_R files are intact. This is a data transfer problem: R_R_R files were present on disk but their content was not fully copied from the cluster.

### Fix B Steps

**Step 1 — Check cluster originals first:**

```bash
# On cluster: verify if originals are intact
python3 -c "
import pickle, glob
pkls = sorted(glob.glob('logs/uav_expert_data/pillars/R_R_R/*.pkl'))
ok = bad = 0
for p in pkls:
    try: pickle.load(open(p,'rb')); ok += 1
    except: bad += 1
print(f'ok={ok} bad={bad}')
"
```

**Step 2a — If cluster originals are intact: re-copy only R_R_R**

```bash
rsync -av --progress \
  <cluster>:FMPCC/FM-PCC/logs/uav_expert_data/pillars/R_R_R/ \
  temp/Gen11E4U3/F3/uav_expert_data/pillars/R_R_R/
```

**Step 2b — If cluster originals are also corrupt: re-run R_R_R only**

```bash
# collect.py --homotopy "(R,R,R)" runs only R_R_R, same seeds 0–499
./Slurm_Codes/submit.sh \
  Slurm_Codes/sbatch/uav_expert_data/collect.sh pillars 500 pid_default 0 "(R,R,R)"
```

Note: collect.sh must pass `--homotopy "(R,R,R)"` to collect.py. Verify the sbatch script accepts a 5th positional argument for homotopy filter. If not, add it.

**Expected outcome:** 121 R_R_R episodes added. Final pillars dataset: 396 + 121 = 517 (slightly over-target due to the 104 rejections in the original run). Homotopy distribution becomes approximately balanced: L_L_L≈117, L_R_L≈84, R_L_R≈74, R_R_R≈121 → slight imbalance on L_R_L/R_L_R but acceptable.

---

## Fix C — Pillars Speed Spikes (Informational, Deferred)

### Observation

98/272 readable pillar episodes (36%) have at least one step with physical speed > 2.5 m/s. The worst is 6.3 m/s at dataset step t=465/470 (episode pillars_R_L_R_pid_default_0000210). Speed spikes occur at the LAST 10 steps of the episode.

### Root Cause Hypothesis

Same lag-overshoot mechanism as s_curve. For the large-diagonal homotopies in pillars (R_L_R and L_R_L require Δy=2.22m diagonal crossings at seg 2 and seg 4), the PID accumulates lateral momentum. At episode end, the trajectory commands return-to-y=0 (the exit waypoint), and the drone is decelerating. For short episode durations (~10 s), the final deceleration can't fully arrest the lateral velocity before the episode ends. The saved `v` reflects actual physics velocity, not commanded velocity.

The z-height at the spike is always normal (0.71–0.97 m). This is NOT a floor crash. The speed spike is a velocity artifact, not a physical hazard.

### Why Not Fixing Now

1. These episodes are physically valid (no floor crash, no obstacle contact)
2. The model may learn the deceleration behaviour as a valid transition
3. Adding a max-speed filter risks discarding too many episodes, especially R_L_R and L_R_L which already have the lowest counts (84 and 74)
4. The p95 speed is 1.28 m/s — fine. Outliers at 36% are at single steps only

### If Needed Later: Filter Option

```python
# In generator.py run_trial, after min_z check:
MAX_SPEED = 3.0   # m/s — tune this
max_speed = max(np.linalg.norm(s['v']) for s in steps)
if max_speed > MAX_SPEED:
    return None
```

At threshold 3.0 m/s: 95/272 (34.9%) would be filtered. Pillars would need ~500/0.65 ≈ 770 trial attempts to get 500 saved. Feasible but changes the dataset character (removes aggressive manoeuvres).

---

## Implementation Priority

| Step | Action | Estimated time | Required before E6? |
|---|---|---|---|
| B.1 | Verify R_R_R originals on cluster | 5 min | Yes |
| B.2 | Re-copy or re-run R_R_R (121 eps) | 10 min job | Yes |
| A.1 | Apply hover-pause fix to `trajectories.py` | 30 min code | Yes |
| A.2 | Update duration range in `generator.py` | 5 min code | Yes |
| A.3 | Smoke test s_curve (20 trials) | 10 min job | Yes |
| A.4 | Full s_curve collection (500 trials) | ~15 min job | Yes |
| C   | Speed spike filter (optional) | 1 hr + re-run | No |

---

## Re-run Plan After Fixes Applied

Once Fix A and Fix B are implemented:

```bash
# Fix B: pillars R_R_R only
./Slurm_Codes/submit.sh collect.sh pillars 130 pid_default 0 "(R,R,R)"

# Fix A: s_curve smoke test first (20 trials)
./Slurm_Codes/submit.sh collect.sh s_curve 20 pid_default 0

# If smoke test < 30% rejection: full s_curve run
./Slurm_Codes/submit.sh collect.sh s_curve 500 pid_default 0
```

Target dataset after all fixes:

| Scene | Target | Expected |
|---|---|---|
| empty | 500 | 500 ✅ (already clean) |
| corridor | 500 | 500 ✅ (already clean) |
| pillars | 500 | ~517 (396 existing + 121 R_R_R) |
| s_curve | 500 | ~350–400 (with <30% rejection) |
| **Total** | 2000 | **~1867–1917** |
