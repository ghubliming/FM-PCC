# Gen11 Epoch4 U4 — Changelog

**Date:** 2026-06-09  
**Based on:** E4 U3 (with Fix_2 Z_FLOOR_MARGIN applied)  
**Trigger:** E4 U4 re-collection (F3) exposed two hard blockers — see `EVAL.md`  
**Plan:** `FIX_PLAN.md`

---

## Fix A — s_curve: hover pauses at segment junctions

**Root cause (confirmed from PKL tracking data, episode 0000010):**

The 3-segment `s_curve_scene_path` uses `traverse_line` with cosine-profile velocity.
At segment junctions the commanded velocity is zero, but the drone's physical inertia
cannot track the cosine acceleration ramp perfectly:

1. **Lag phase:** drone lags 0.18 m behind commanded y at the mid-crossing (err_y = −0.185 m
   at dataset step 330). PID applies aggressive `+y` correction on top of trajectory feed-forward.
2. **Overshoot:** the cosine profile starts decelerating while the drone has accumulated
   `+y` momentum. In 0.6 s the drone swings 0.844 m laterally (1.41 m/s vs commanded 0.45 m/s).
3. **Altitude collapse:** combined y-error (0.53 m) + z-error (0.40 m) correction demand
   depresses available z-thrust → drone drops to z = 0.682 m. At lower starting altitudes
   (z ≈ 0.70 m minimum) the same drop produces z < 0.50 m → Z_FLOOR_MARGIN rejection.

This mechanism affected ~90 % of s_curve trials, causing the 90.5 % rejection ABORT in F3.

---

### A1 — `uav_expert_data_collect/trajectories.py`

**Function:** `s_curve_scene_path`

Added two 1.0 s `hover_at` pauses at the segment junctions. The drone settles to near-zero
lateral velocity during the hover before each transition, eliminating the lag-overshoot cycle.

```diff
-    t_a = T * d_a / d_total
-    t_b = T * d_b / d_total
-    t_c = T * d_c / d_total
-
-    seg_a = traverse_line((-3.2, y1, z), (-0.5, y1, z), t_a, yaw)
-    seg_b = traverse_line((-0.5, y1, z), ( 0.5, y2, z), t_b, yaw)
-    seg_c = traverse_line(( 0.5, y2, z), ( 3.2, y2, z), t_c, yaw)
-
-    def traj(t):
-        if t < t_a:
-            return seg_a(t)
-        elif t < t_a + t_b:
-            return seg_b(t - t_a)
-        else:
-            return seg_c(t - t_a - t_b)
+    T_HOVER = 1.0
+    T_move  = T - 2.0 * T_HOVER
+
+    t_a = T_move * d_a / d_total
+    t_b = T_move * d_b / d_total
+    t_c = T_move * d_c / d_total
+
+    seg_a = traverse_line((-3.2, y1, z), (-0.5, y1, z), t_a, yaw)
+    hov_1 = hover_at((-0.5, y1, z), yaw)
+    seg_b = traverse_line((-0.5, y1, z), ( 0.5, y2, z), t_b, yaw)
+    hov_2 = hover_at(( 0.5, y2, z), yaw)
+    seg_c = traverse_line(( 0.5, y2, z), ( 3.2, y2, z), t_c, yaw)
+
+    t1 = t_a
+    t2 = t_a + T_HOVER
+    t3 = t2 + t_b
+    t4 = t3 + T_HOVER
+
+    def traj(t):
+        if t < t1:
+            return seg_a(t)
+        elif t < t2:
+            return hov_1(t)
+        elif t < t3:
+            return seg_b(t - t2)
+        elif t < t4:
+            return hov_2(t)
+        else:
+            return seg_c(t - t4)
```

`T_move = dur - 2.0` preserves the same distance-proportional speed budget for the
three traverse segments. The hover phases are pure `hover_at` — zero velocity and
zero acceleration commanded — so the PID holds position quietly.

---

### A2 — `uav_expert_data_collect/generator.py`

**s_curve duration range:** `[16.0, 22.0]` → `[18.0, 24.0]` s

```diff
-        # Fix_4: revert duration [22,30]→[16,22]s to match the Fix_2 config that
-        # achieved 61.9% rejection (best so far).  Longer duration worsened things.
-        dur = float(rng.uniform(16.0, 22.0))
+        # U4 Fix A: raise duration range [16,22]→[18,24]s to account for the two
+        # 1.0 s hover pauses added to s_curve_scene_path (2.0 s total pause budget).
+        # T_move = dur - 2.0 stays in [16, 22]s — same manoeuvre time as before.
+        dur = float(rng.uniform(18.0, 24.0))
```

`T_move ∈ [16, 22]` s is identical to the previous range; the old Fix_4 reasoning
still holds for the manoeuvre-only time budget.

---

## Fix B — pillars R_R_R: homotopy filter in collect.sh

**Issue:** All 121 R_R_R PKLs from F3 are truncated (data transfer failure). Need to
re-run or re-copy just the R_R_R homotopy without re-collecting the other three.

### B1 — `Slurm_Codes/sbatch/uav_expert_data/collect.sh`

Added `$5` positional argument for homotopy filter, forwarded to `collect.py --homotopy`.

```diff
 # Args:
 #   $1 = scene       (empty|corridor|s_curve|pillars|all_scenes)  [default: empty]
 #   $2 = n_trials    [default: 200]
 #   $3 = gain        (pid_default|pid_high_gain|pid_low_gain)      [default: pid_default]
 #   $4 = seed_offset (added to SLURM_ARRAY_TASK_ID * 10000)        [default: 0]
+#   $5 = homotopy    (all | specific label e.g. "(R,R,R)")         [default: all]

+HOMOTOPY="${5:-all}"

-    --homotopy     all \
+    --homotopy     "$HOMOTOPY" \
```

Default is `all` — fully backward compatible. Existing usages unchanged.

**Usage for Fix B:**

```bash
# Step 1: verify cluster originals
#   ssh <cluster>
#   python3 -c "import pickle,glob; ..."   (see FIX_PLAN.md §Fix B)

# Step 2a: if originals intact — re-copy
#   rsync -av cluster:FMPCC/FM-PCC/logs/uav_expert_data/pillars/R_R_R/ \
#         temp/Gen11E4U3/F3/uav_expert_data/pillars/R_R_R/

# Step 2b: if originals corrupt — re-run R_R_R only
./Slurm_Codes/submit.sh \
    Slurm_Codes/sbatch/uav_expert_data/collect.sh \
    pillars 130 pid_default 0 "(R,R,R)"
```

Note: re-running with seed 0 + homotopy `(R,R,R)` produces episode IDs
`pillars_R_R_R_pid_default_0000000` … `_0000129`. These match the IDs from the
original run and will overwrite the corrupted files cleanly.

---

## Re-run Commands (after this changelog)

```bash
# s_curve smoke test — 20 trials to verify rejection rate < 30 %
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh s_curve 20

# s_curve full collection (after smoke passes)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh s_curve 500

# pillars R_R_R only (after checking cluster originals)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh pillars 130 pid_default 0 "(R,R,R)"
```

---

## Files Changed

| File | Change |
|---|---|
| `uav_expert_data_collect/trajectories.py` | `s_curve_scene_path`: hover pauses at junctions |
| `uav_expert_data_collect/generator.py` | s_curve duration range `[16,22]` → `[18,24]` s |
| `Slurm_Codes/sbatch/uav_expert_data/collect.sh` | `$5` homotopy filter argument |
