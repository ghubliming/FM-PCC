# Gen11 Epoch 4 — Expert Data Sourcing for FM-PCC UAV Training: Research

**Date**: 2026-06-01
**Status**: Research only — no code in this epoch.
**Predecessors**: Epoch 1 (X2 vendored), Epoch 2 (cascaded PID validated), Epoch 3 (env scenes + record_sim_frame).
**Roadmap position**: This epoch executes path_temp_initial.md step 3 — "Mirror UAV-Flow trajectory statistics. Implement a simple expert controller in MuJoCo that generates similar nice paths so you have supervised trajectories for FM."

---

## TL;DR

For training FM-PCC on the UAV task, we need **(state, action) trajectory
data in our specific format** (H-step chunks of `[act ‖ state]`,
sampled at our chosen rate, in our chosen state/action conventions).

Three candidate sources:

| Source | Pre-existing data? | Aligned with our format? | Realistic for FM-PCC training? |
|---|---|---|---|
| **MuJoCo MPC** (mjpc/tasks/quadrotor) | ❌ No — it's a real-time PLANNER, not a dataset | ✅ Same physics, same state space (would generate aligned data) | ✅ — but requires building a "MJPC-as-data-collector" harness |
| **UAV-Flow Colosseum** | ✅ ~100 JSON trajectory files (`UAV-Flow-Eval/test_jsons/`, 210k lines total) | ❌ Wrong sim (UE4), wrong units (cm), wrong physics (Unreal black-box), wrong action format (waypoints only — no recorded actions) | ❌ Cannot be used as direct training data; usable for **statistics only** |
| **Manual generation in our MuJoCo stack** | N/A — we generate fresh | ✅ Native format, full control | ✅ **Recommended primary path** |

**Verdict**: **Manual generation is the primary path.** It's the only
source that ships data in our format with no conversion losses.
MJPC-as-data-collector is a viable secondary option for racing-style
demos. UAV-Flow is useful for statistical targets and pretraining
(but not as a direct training set for our MuJoCo policy).

---

## 1. What "expert data" means for FM-PCC

FM-PCC training consumes a tensor of shape `(B, H, D)` per sample, where:
- `B` = batch size
- `H` = trajectory horizon (= 8 in our existing aligning runs; same value
  TBD for UAV but likely 8 to reuse the U-Net architecture)
- `D` = transition_dim per timestep, structured as
  `[action ‖ obs]` with the obs anchor pinned at step 0 via
  `apply_conditioning`.

For the UAV task, plausible state/action conventions:

| Variant | Action dim | Obs dim | Trajectory dim D | Comment |
|---|---|---|---|---|
| **High-level position target** | 3 (`(p_x, p_y, p_z)`) | 6 (`[p(3), v(3)]`) | 9 | Mirrors aligning's 9-D. Inner cascaded PID converts position target → motor thrusts. |
| **High-level position + velocity** | 6 (`(p, v)` target) | 6 | 12 | Closer to ACT's chunked output but with velocity feedforward. |
| **Full state + raw thrust** | 4 (`u₁..u₄`) | 13 (`[p, q, v, ω]`) | 17 | Lowest level. Hardest to learn. |

The choice depends on what we want FM to plan. For initial experiments,
**3-D position target + 6-D obs (`p, v`) → 9-D trajectory** is the
right default — it matches aligning's architecture exactly and lets
the cascaded PID handle low-level control.

Each training trajectory must therefore contain, per timestep:
- The robot's state at that timestep (`p, v`).
- The desired position at that timestep (becomes the "action").
- (Optional) Box/target/obstacle state if non-visual.

---

## 2. Source A — MuJoCo MPC (run-and-record)

### 2.1 What's actually in mujoco_mpc

Per Epoch 1 audit (`MUJOCO_MPC_UAV_MATH.md`):
- `mjpc/tasks/quadrotor/task.xml`: Skydio X2 + 11 waypoints + 8 racing gates.
- `quadrotor.cc/.h`: C++ residual + transition functions for MJPC's
  sampling-based planner.
- `task.xml <custom>` block: MJPC-specific hyperparameters (32
  candidates per replan, 0.5 s horizon, etc.).

**No JSON, no pickle, no recorded rollouts.** MJPC computes the
trajectory online via predictive sampling — it's a controller, not a
dataset.

### 2.2 Could we run MJPC and record rollouts?

Yes, but with engineering work. The pipeline would be:

```
1. Build mujoco_mpc from source (C++/CMake, includes the X2 task).
2. Run the MJPC GUI or headless binary on the quadrotor racing task.
3. Modify the MJPC source (or write a logger) to dump per-step (state, action) to disk.
4. Convert dumped state/action sequences into our (H, D) chunks.
```

**Effort:** 1–2 days. Mostly building MJPC and wiring the logger.

**Pros:**
- Trajectories are physically feasible in OUR same MuJoCo dynamics.
- MJPC is a well-tuned controller — produces sensible racing
  trajectories.
- Can generate as much data as we want (infinite trajectory horizon
  by varying initial conditions).

**Cons:**
- **Wrong task**: MJPC's task is RACING (chase waypoints through 8
  gates). We want OBSTACLE AVOIDANCE (corridors/pillars). The
  trajectories' kinematic statistics will be racing-flavored
  (aggressive accelerations, tight turns).
- Build dependency: requires MJPC's full C++ toolchain on the
  cluster. Not trivial.
- Doesn't generalize: if we change scenes (different obstacle layouts),
  we'd need to re-design MJPC's residual + transition for each new
  scene.

### 2.3 Verdict on MJPC

Useful as a **racing-task benchmark** generator if we ever want to
test FM-PCC on drone racing. Not useful for the obstacle-avoidance
task that path_temp_initial.md sets as the goal. **Skip for the
primary training set.**

---

## 3. Source B — UAV-Flow Colosseum

### 3.1 What's actually in UAV-Flow

Per the inventory at `/workspaces/UAV-Flow/UAV-Flow-Eval/test_jsons/`:
- ~100 JSON files, 210,000 lines total.
- Each JSON: instruction (natural language), initial_pos, end_pos,
  target_pos, `reference_path_raw` (list of 6-DOF waypoints).
- Format per waypoint: `[x, y, z, roll, pitch, yaw]` in UE4 coordinates
  (cm).

Example trajectory (`2025-05-06_19-14-24.json` per the temp_Ideas.md
report):
```
Initial: [-412.117, 190.853, 300.0, 0, -174.722, 0]   (cm, UE4)
End:     [-988.554, 135.718, 300.0, 0, -174.722, 0]
Distance: ~580 cm = 5.8 m
Altitude: constant 3 m
Path: 80-150 waypoints
```

### 3.2 Why this isn't directly usable

**Five hard alignment problems:**

1. **Units**: UAV-Flow uses centimeters; MuJoCo uses meters. Scaling
   factor 0.01. Trivial to fix.

2. **Coordinate system**: UE4 (X-forward, Y-right, Z-up) vs. MuJoCo
   (typically Z-up but with X/Y swapped by convention). Need a
   rotation matrix to align. Doable.

3. **No actions recorded**: only WAYPOINTS are stored. To turn this
   into `(state, action)` training pairs, we'd have to:
   - Finite-difference consecutive waypoints to get velocities.
   - Either treat velocities as "actions" (mid-level control) or
     differentiate again to get accelerations (would-be thrust
     commands).
   - This is approximate — we never see what the demonstrator
     actually did.

4. **Wrong dynamics**: UAV-Flow's waypoints were produced by Unreal
   Engine's black-box flight model with proprietary controller. Our
   MuJoCo Skydio X2 has different mass, inertia, thrust curve, etc.
   Replaying UAV-Flow's exact waypoints in our MuJoCo via our PID
   would NOT produce the recorded trajectory — different state
   evolution, different tracking error.

5. **Wrong scenes**: UAV-Flow trajectories were recorded in
   photorealistic UE4 environments (urban, suburban, indoor). Our
   MuJoCo scenes are abstract geometries (corridor, S-curve, pillars).
   The OBSTACLE STRUCTURE is fundamentally different. UAV-Flow data
   teaches "fly through this specific city street" — we need "weave
   between these specific pillars."

### 3.3 What UAV-Flow IS useful for

**Statistical targets** for our manual data generation. Per the
temp_Ideas.md report:

| UAV-Flow statistic | Value | How we'd use it |
|---|---|---|
| Typical altitude | 300 cm (3 m) | Set our hover height ≈ 1.0 m for indoor-scale scenes; or 3 m if matching exactly |
| Max horizontal velocity | 30 cm/s (0.3 m/s) | Cap our trajectory generator's velocity at ≈ 0.3 m/s |
| Max yaw rate | 200°/s | Set yaw rate cap on our cascaded PID's reference |
| Episode length | 80-150 waypoints | At ~30 Hz that's ~3-5 s; matches our Epoch-2 trajectory duration |
| Path length | 200-1000 cm (2-10 m) | Configure our scene extents accordingly |
| Δ position per step | 2-5 cm | At 30 Hz that's 60-150 cm/s — actually higher than the 30 cm/s headline; suggests UAV-Flow samples are downsampled. Need to verify. |
| Smoothness | Smooth curves (no jerks) | Use cosine-blended or polynomial trajectory generators (we already do this in Epoch 2's `traverse_line`, `s_curve_path`, `weave`) |

These statistics give us "what good UAV trajectories look like" so
our manual generator produces FM training data that's
distribution-similar to real-world UAV flight, even if the underlying
scenes differ.

### 3.4 Verdict on UAV-Flow

**Mine for statistics, don't train on directly.** Add to Epoch 4
backlog a small utility that loads `UAV-Flow-Eval/test_jsons/*.json`,
extracts the kinematic distributions (velocity histogram,
acceleration histogram, altitude profile, path-length CDF), and
saves them as our trajectory-generator's TARGET distribution.

---

## 4. Source C — Manual generation (primary path)

### 4.1 What this looks like

For each obstacle scene from Epoch 3 (corridor, S-curve, pillars,
empty), and a chosen set of conditions (start pose, goal pose, scene
parameters), we use our **already-validated** Epoch 2 cascaded PID to
fly hand-designed reference trajectories. The PID's INPUT (position
target per step) becomes the "expert action"; the X2's STATE at each
step becomes the "expert state."

```python
for scene in [empty, corridor, s_curve, pillars]:
    for trial in range(N_PER_SCENE):
        env = load_scene(scene)
        start_pose = sample_random_start(scene)
        target_traj = generate_reference(start_pose, scene)   # e.g. traverse_line + jitter
        rollout = run_pid(env, target_traj)
        save_rollout(rollout, format='fm_pcc_9D')
```

Per-step output (FM-PCC 9-D format, hypothetical):
```
state_t = [p_x, p_y, p_z, v_x, v_y, v_z]              # 6-D obs
action_t = [p_x_des, p_y_des, p_z_des]                # 3-D position target
trajectory_chunk_t = [action_t ‖ state_t]              # 9-D per timestep
```

Save to disk as pickled (or HDF5) per-episode files mirroring the
D3IL aligning dataset format.

### 4.2 What we already have from Epoch 2

| Component | Status | Re-usable? |
|---|---|---|
| `flight_controller.py` cascaded PID | ✅ working | Yes |
| `trajectories.py` factories (`hover_at`, `step_to`, `circle`, `traverse_line`, `s_curve_path`, `weave`) | ✅ working | Yes |
| `run_naive.py` driver | ✅ rolls out one task at a time | Needs extension to loop over many trials + save in dataset format |
| Scene XMLs (Epoch 3) | ✅ four scenes | Yes |
| Logging format (log.json) | ✅ per-step state, target, control | Re-shape into 9-D trajectory chunks |

### 4.3 What's still to build for Epoch 4

| Component | Effort | Description |
|---|---|---|
| **Trial randomizer** | 1 h | Sample random (start, goal, scene-params) per trial |
| **Per-scene reference generator** | 2 h | Each scene gets a parameterized trajectory family (corridor → traverse_line w/ random altitudes; pillars → weave w/ random amplitudes; etc.) |
| **Dataset writer** | 1 h | Save per-trial rollouts as (state, action) pickles in the FM-PCC dataset format |
| **Stats validator** | 1 h | Check generated data matches UAV-Flow statistics (velocity ranges, altitude profile, etc.) — sanity, not gating |
| **SLURM job** | 30 min | Parallel rollouts across N seeds; ~100-500 trials per scene |
| **Verification** | 1 h | Spot-check a few trials' rollout.gif to ensure trajectories look reasonable |
| **Total** | **~6-8 h** | Doable in one day on cluster |

### 4.4 How much data do we need?

For D3IL aligning visual baselines: ~900 episodes for training.
Conservative target for UAV: **~500-1000 trials per scene × 4 scenes
= 2000-4000 trials**. With ~5 s per trial at 100 Hz, that's
~5000-10000 timesteps per scene, ~20k-40k timesteps total, ~3-5k
9D-chunks of horizon-8 trajectories.

This is small by deep-learning standards but matches what aligning
uses; should be enough for a first FM-PCC drone experiment.

### 4.5 What's NOT in this manual approach

- **No multi-modal demos**: each trial follows one cosine-profile
  trajectory. If we want multi-modality (different homotopy classes
  through obstacles), we need multiple trajectory generators per
  scene.
- **No real "expert" — it's a hand-designed reference**: FM will learn
  to mimic the cosine profile, not real-world UAV flight. That's
  acceptable for a research demo (the goal is "does FM-PCC work on
  drones at all"), but unsuitable for sim-to-real transfer claims.
- **No natural-language instructions**: unlike UAV-Flow, our trials
  don't have language labels. If we want language-conditioned UAV
  policy, we'd need to add a labeling step (or generate trials per
  instruction template).

---

## 5. Recommended Epoch 4 plan

### 5.1 Three sub-phases

| Phase | What | Time |
|---|---|---|
| **4-α** | Mine UAV-Flow JSONs for statistics → save target distribution table | 1 h |
| **4-β** | Extend Epoch 2's driver into a trial-randomizing dataset generator (per §4.3) | 4-6 h |
| **4-γ** | Run on SLURM → 500-1000 trials × 4 scenes → save as FM-PCC dataset | overnight job, then 1 h to spot-check |

### 5.2 What this does NOT include

- ❌ Training an FM-PCC model on the generated data (that's Epoch 5).
- ❌ Visual encoder / image rendering during dataset collection (state-only first, per path_temp step 5).
- ❌ MJPC-as-data-collector pipeline (skipped per §2.3 — wrong task).
- ❌ UAV-Flow waypoint-to-action conversion (skipped per §3.2 — wrong sim).

### 5.3 Deliverables

- `uav_expert_data_collect/` (new dir at repo root, mirroring
  `collect_visual_avoiding_data/` and `uav_naive_test/` conventions)
  - `generator.py` — trial randomizer + reference trajectory family per scene
  - `dataset_writer.py` — save rollouts as FM-PCC-compatible pickles
  - `collect.py` — driver script
- `Slurm_Codes/sbatch/uav_expert_data/collect.sh` — SLURM wrapper
- `logs/uav_expert_data/<scene>/env_<id>.pkl` — output (gitignored under `logs/*`)
- `logs_in_develop/Gen11/Epoch4_expert_data/CHANGELOG.md` — closure
  with stats validation table

### 5.4 Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Cascaded PID instability on randomized trials | Medium | Use only "moderate" randomization ranges (e.g. velocity ≤ 0.3 m/s). Drop trials with controller divergence. Known PID limitation from Epoch 2 closure. |
| Generated data lacks multi-modality | Low for the first experiment, High for sim-to-real | Accept for Epoch 5 baseline. Add multi-modal generators in a later epoch if needed. |
| Distribution mismatch with UAV-Flow statistics | Low | Already accounted for by using UAV-Flow statistics as Phase 4-α target. Adjust generator parameters until histograms align. |
| Scene-specific trajectory failures (e.g. weave can't fit pillar field) | Medium | Run Phase 4-β on smoke trials first (10-20 per scene) before full collection. |

---

## 6. Open questions before launching Epoch 4

| # | Question | Default if not specified |
|---|---|---|
| Q1 | Trajectory format: 9-D `[act(3) ‖ p(3), v(3)]` or 12-D `[act(6) ‖ p(3), v(3)]` (include velocity target in action)? | **9-D** — matches aligning architecture, simpler FM. |
| Q2 | Sampling rate for the saved dataset: physics 100 Hz, or downsampled to 30 Hz (matches UAV-Flow)? | **30 Hz** for the dataset; physics still 100 Hz internally. Downsample 3-to-1 during dataset write. |
| Q3 | Include obstacle state in obs? Or only robot state? | **Robot-only state for first cut.** Obstacles enter via the DPCC projector (Epoch 5+), not via obs. Mirrors visual aligning's choice. |
| Q4 | Are we collecting on the cluster (SLURM, parallel) or locally? | **SLURM** — same as everything else in this project. Faster + matches workflow. |
| Q5 | How tightly should our generated trajectories match UAV-Flow statistics? | **Loose match.** Aim for velocity range overlap and altitude similarity. Don't try to replicate UAV-Flow exactly — our scenes are different. |

---

## 7. One-line summary

For Epoch 4 expert-data sourcing: **manual generation in our own
MuJoCo stack is the only path that ships training-ready data**. MJPC
is wrong-task; UAV-Flow is wrong-sim and lacks recorded actions; both
serve only as references (UAV-Flow's statistics shape our generator's
output distribution; MJPC's racing flavor sits aside for a possible
future racing demo). Concrete next step: extend Epoch 2's PID +
trajectory factories into a trial-randomizing dataset generator,
collect ~500-1000 trials per scene, validate distributions against
UAV-Flow statistics, save in FM-PCC format. ~one cluster-day of work
including overnight collection.
