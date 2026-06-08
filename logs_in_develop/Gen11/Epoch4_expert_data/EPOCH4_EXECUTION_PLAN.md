# Gen11 Epoch 4 — Expert Data Collection: Execution Plan

**Date**: 2026-06-04  
**Status**: Blueprint — no code in this document.  
**Inputs**: [RESEARCH.md](Materials/RESEARCH.md), [AUDIT.md](Materials/AUDIT.md), [EXPERT_DATA_PROVENANCE.md](Materials/EXPERT_DATA_PROVENANCE.md)  
**Goal**: Produce a training-ready `(state, action)` dataset for FM-PCC on the UAV task, in our native MuJoCo stack, following the format and conventions already validated by Epochs 1–3.

---

## 0. Guiding Principles

These are the non-negotiable constraints that govern every decision below.

1. **Manual generation is the primary path.** MJPC is wrong-task (racing ≠ avoidance). UAV-Flow is wrong-sim (UE4 ≠ MuJoCo) and lacks recorded actions. Both serve only as references. *(RESEARCH §7, PROVENANCE §4)*
2. **Position-delta as action.** D3IL, UAV-Flow, and our own pipeline all converge on `Δp_des = p_des[t+1] − p_des[t]` as the action convention. The dataset must use **deltas, not absolute targets**. *(PROVENANCE §3a.6, AUDIT Risk 3)*
3. **Two-stage collection (state → images).** Stage 1 records state-only pickles headlessly. Stage 2 replays and renders cameras. The two never run simultaneously. *(PROVENANCE §3c)*
4. **FM theory is demonstrator-agnostic.** A scripted PID controller is a legitimate expert for our research question ("does FM-PCC work on drones?"). Real-pilot data is correct for a *different* question ("can we deploy?"). *(PROVENANCE §3.2(f), §3.5)*
5. **9D format is the default; 12D is the fallback.** `[Δp_des(3) ‖ p(3), v(3)]` matches D3IL's aligning architecture. If PID hover instability re-emerges during low-speed FM planning, upgrade to 12D `[Δp_des(3), Δv_des(3) ‖ p(3), v(3)]`. *(RESEARCH §1, AUDIT R4)*

---

## 1. Three Pre-Flight Decisions (Gate: resolve ALL before writing code)

> [!CAUTION]
> These are **blocking** — writing `dataset_writer.py` or `generator.py` without resolving them risks producing an unusable dataset.

### Decision 1 — Action Convention Lock

| Option | Convention | D3IL alignment | Risk |
|---|---|---|---|
| **A (recommended)** | `Δp_des` — position *delta* per step | ✅ Matches `vel_state = des_c_pos[1:] - des_c_pos[:-1]` | None |
| B | `p_des` — absolute position target | ❌ Contradicts D3IL and UAV-Flow | FM velocity-field semantics are wrong |

**Default**: **A**. Lock this and never revisit.  
*(Source: AUDIT Risk 3, PROVENANCE §3a.2)*

### Decision 2 — PID Stability Fix

The Epoch 2 closure identified that `Kp_omega` needs correction to `[2.5, 2.5, 1.0]`. Running 1000 randomized trials with the broken gain will corrupt ≥30% of them with limit-cycle instability.

**Action**: Apply the one-line fix to `flight_controller.py` *before* any Phase 4-β work. Verify with a 10-trial smoke test on `s_curve` scene.  
*(Source: AUDIT Risk 1, Epoch 2 §3)*

### Decision 3 — Dataset Schema Lock

Define the per-episode pickle schema *before* implementing `dataset_writer.py`. Proposed schema:

```
Per-episode file:  uav_expert_data/{scene}/{ep_id}.pkl
Keys:
  episode_id    : str
  scene         : str           ("corridor" | "s_curve" | "pillars" | "empty")
  homotopy      : str           ("(L,R,L)" or "N/A")
  controller    : str           ("pid_default" | "pid_high_gain" | "mjpc" | "teleop")
  dt            : float         (1/30 for 30 Hz dataset; physics at 100 Hz internally)
  obs           : np.ndarray    (T, 6)  — [p(3), v(3)]
  actions       : np.ndarray    (T-1, 3) — [Δp_des(3)]
  targets       : np.ndarray    (T, 3)  — absolute position reference (for debugging)
  obstacles     : list[dict]    [{type, center, radius/half_extents}, ...]
  metadata      : dict          {start_pos, end_pos, total_time, controller_gains}
```

> [!TIP]
> Including `obstacles` per-episode enables a future obstacle-conditioned policy without re-collecting data. SafeFlowMPC does this (`p_cols` in `conditional_data1`).

*(Source: AUDIT R8)*

---

## 2. Execution Phases

### Phase 4-α — Mine UAV-Flow for Statistical Targets

**Duration**: ~1 hour  
**Objective**: Extract kinematic distributions from 273 UAV-Flow eval trajectories to set our generator's target parameters.

**What to extract**:
- Velocity histogram (`Δp/Δt` per step) — resolve the 0.3 m/s vs 0.6–1.5 m/s inconsistency
- Acceleration histogram
- Altitude profile distribution
- Path-length CDF
- Episode-length histogram

**Output**: A reference table of target ranges (velocity, altitude, path-length, duration) saved as a JSON or markdown table inside this epoch's logs.

> [!WARNING]
> RESEARCH §3.3 reports "max velocity 0.3 m/s" but per-step deltas suggest 5× higher. Do NOT set velocity caps until the actual sweep is complete.

*(Source: RESEARCH §3.3, AUDIT R7)*

---

### Phase 4-β — Build the Generator Pipeline

**Duration**: ~4–6 hours  
**Objective**: Extend Epoch 2's validated PID + trajectory factories into a trial-randomizing dataset generator.

#### Components to build:

| Component | Description | Builds on |
|---|---|---|
| **Trial randomizer** | Sample random `(start, goal, scene_params)` per trial. Include ≥2 homotopy classes per obstacle scene. | Epoch 3 scene XMLs |
| **Per-scene reference generator** | Corridor → `traverse_line` w/ random altitude. Pillars → `weave` w/ random amplitude + homotopy label `(L/R)^N`. S-curve → `s_curve_path` w/ random speed. Empty → random waypoints. | Epoch 2 `trajectories.py` |
| **Dataset writer** | Save per-trial rollouts as pickles in the locked schema (Decision 3). Downsample from 100 Hz physics to 30 Hz dataset. Compute `Δp_des` deltas. | New |
| **Stats validator** | Compare generated data against Phase 4-α target distributions. Log histograms. Sanity check, not a hard gate. | Phase 4-α output |
| **Trajectory noise augmentation** | Add `ε ~ N(0, σ²I)`, σ ≈ 0.01–0.05 m, to expert positions *during dataset creation* to thicken the data manifold. | New |

#### Multi-modality strategy:

For each obstacle scene, define **≥2 homotopy classes** (topologically distinct routes):

```
Pillars (3 obstacles):
  Homotopy A: (L, L, L)  — pass left of all
  Homotopy B: (L, R, L)  — pass left, right, left
  Homotopy C: (R, L, R)
  Homotopy D: (R, R, R)  — pass right of all
```

Each class gets its own reference trajectory generator. Within a class, vary start/end positions and speed profiles. This restores multi-modality without needing human pilots.  
*(Source: AUDIT R3, R5)*

#### PID gain variation:

Additionally, vary `Kp_pos`, `Kp_vel` by ±20% across trajectory batches to produce behaviourally distinct controllers — further thickening the data manifold and avoiding the "thin-curve" covariate-shift problem.  
*(Source: AUDIT R5)*

---

### Phase 4-γ — Cluster Collection + Validation

**Duration**: ~2 cluster-days (realistic estimate, not the optimistic "one day")  
**Objective**: Generate 2000–4000 training episodes across 4 scenes.

#### Scale targets:

| Scene | Episodes | Homotopy classes | Controller variants |
|---|---|---|---|
| Empty | 500–1000 | N/A | `pid_default`, `pid_high_gain` |
| Corridor | 500–1000 | 2 (left-wall-hug, right-wall-hug) | `pid_default`, `pid_high_gain` |
| S-curve | 500–1000 | 1 (topologically unique) | `pid_default`, `pid_high_gain` |
| Pillars | 500–1000 | 4 (L/R per pillar) | `pid_default`, `pid_high_gain` |

#### Collection workflow:

1. **Smoke test**: 10–20 trials per scene locally. Spot-check rollout GIFs. Catch s-curve instability or schema bugs.
2. **SLURM batch**: Parallel rollouts across N seeds. `Slurm_Codes/sbatch/uav_expert_data/collect.sh`.
3. **Overnight run**: 500–1000 trials × 4 scenes.
4. **Morning validation**: Stats comparison vs 4-α targets. Spot-check 5 GIFs per scene. Log rejection rate (trials where PID diverged).
5. **Format check**: Load 10 random episodes through the FM-PCC dataloader to verify tensor shapes match `(B, H=8, D=9)`.

#### Day 2 budget:

| Activity | Time |
|---|---|
| Debug MuJoCo rendering on Slurm (EGL/GPU) | 1–2 h |
| Fix dataset format mismatches (first pickle → FM dataloader) | 2–3 h |
| Re-run after fixes | overnight |
| Final spot-check + stats table for CHANGELOG | 1 h |

*(Source: AUDIT R9)*

---

## 3. Sanity Gate — Mini-FM Training Checkpoint

> [!IMPORTANT]
> Do this **after** Phase 4-β, **before** the full Phase 4-γ collection.

Train a **tiny FM** on ~100 manually generated trajectories from the `empty` scene (no obstacles). If FM can reproduce the PID's empty-scene trajectories at < 0.1 m RMS, the data pipeline is correct. If not, fix data issues before scaling up.

This prevents the "collect 4000 episodes, discover in Epoch 5 that the action convention was wrong" failure mode.

**What this tests**:
- Schema correctness (pickle → dataloader → training tensor)
- Action convention correctness (Δp vs absolute p)
- Horizon/dim compatibility (H=8, D=9)
- Basic FM convergence on a trivially learnable distribution

**What this does NOT test**:
- Obstacle avoidance quality (no obstacles in empty scene)
- Multi-modality (only one controller variant needed)
- DPCC projection (Epoch 5 scope)

*(Source: AUDIT Risk 2)*

---

## 4. Data Source Layering (the A→F Spectrum)

Not all data needs to come from one source. Layer them by purpose:

| Priority | Spectrum Row | Source | Volume | Purpose |
|---|---|---|---|---|
| **Primary** | **D** — Parametrised scripted | PID flying parameterised routes with homotopy labels | ~1k–10k | Bulk distributional coverage |
| Secondary | **F** — Optimal-control oracle | MJPC via Python API (if buildable) | ~500–1k | High-quality upper-bound expert trajectories |
| Tertiary | **C** — Joystick teleop | `tools/teleop_uav.py` (MuJoCo viewer + pygame) | ~100–200 | Stylistic multi-modality (the FM sweet spot) |
| Optional | **A** — Hand-drawn seeds | Matplotlib click-to-waypoint tool | ~10–50 | Topologically hard cases; paper figures |

> [!NOTE]
> Row **D** is the only one required for Epoch 4 closure. Rows F, C, A are enhancements that can be added mid-to-late Epoch 4 or deferred to Epoch 5.

*(Source: PROVENANCE §3d.4)*

---

## 5. Two-Stage Architecture (State → Images)

This is the structural pattern inherited from Gen9 Epoch 1's `replay_and_capture()`.

```
Stage 1 (headless, CPU)          Stage 2 (GPU, per-frame render)
─────────────────────────        ──────────────────────────────────
PID/teleop flies drone    ──►   Load state pickle
Records (t, p, v, q, ω,         Replay action sequence in MuJoCo
 des_p, thrusts)                 Render bp-cam (overhead world)
                                 Render fpv-cam (body-frame FPV)
Outputs:                         Outputs:
  state_pickles/{ep}.pkl           images/bp-cam/{ep}/{frame}.jpg
                                   images/inhand-cam/{ep}/{frame}.jpg
```

**Why two stages, not one**:
- Iterate on controller without re-rendering
- Render multiple camera variants from one state set
- Domain randomization at render time without corrupting physics
- Skip Stage 2 entirely for non-visual baselines (Epoch 4 primary)
- Cheaply scale Stage 1 to 10k trajectories; render only what you need

**Epoch 4 scope**: **Stage 1 only.** Stage 2 activates when visual FM-PCC is in scope (Epoch 5+).

*(Source: PROVENANCE §3c)*

---

## 6. Deliverables Checklist

### Code deliverables:

| File | Purpose | Phase |
|---|---|---|
| `uav_expert_data_collect/generator.py` | Trial randomizer + per-scene reference trajectory family + homotopy labelling | 4-β |
| `uav_expert_data_collect/dataset_writer.py` | Save rollouts as schema-locked pickles; 100→30 Hz downsample; Δp_des computation | 4-β |
| `uav_expert_data_collect/collect.py` | Driver script (CLI: scene, N_trials, seed, gain_variant) | 4-β |
| `uav_expert_data_collect/stats_validator.py` | Compare generated data vs 4-α targets; produce histograms | 4-β |
| `Slurm_Codes/sbatch/uav_expert_data/collect.sh` | SLURM wrapper for parallel collection | 4-γ |
| `tools/teleop_uav.py` *(optional)* | MuJoCo viewer + joystick teleop | 4-late |

### Documentation deliverables:

| File | Purpose |
|---|---|
| `logs_in_develop/Gen11/Epoch4_expert_data/CHANGELOG.md` | Epoch closure with stats validation table |
| Phase 4-α stats table | JSON/markdown with UAV-Flow kinematic target ranges |

### Data deliverables (gitignored):

| Path | Content |
|---|---|
| `logs/uav_expert_data/{scene}/env_{id}.pkl` | Per-episode state+action pickles |

---

## 7. Risk Register

| # | Risk | Severity | Mitigation | Owner |
|---|---|---|---|---|
| 1 | PID instability on randomized starts (s-curve 41% contact rate) | 🔴 High | Fix `Kp_omega = [2.5, 2.5, 1.0]` before 4-β. Smoke-test 20 trials. | Decision 2 |
| 2 | Action convention mismatch (absolute vs delta) | 🔴 Critical | Lock Decision 1 (delta). Verify in mini-FM gate. | Decision 1 |
| 3 | Dataset schema doesn't match FM dataloader expectations | 🟡 High | Lock Decision 3. Validate with format-check step in 4-γ. | Decision 3 |
| 4 | Velocity cap set 5× too low (UAV-Flow stat inconsistency) | 🟡 High | Resolve in Phase 4-α before parameterising the generator. | Phase 4-α |
| 5 | Mono-modal data wastes FM's multi-modal capability | 🟡 High | ≥2 homotopy classes per obstacle scene + gain variation. | Phase 4-β |
| 6 | "One cluster-day" estimate is optimistic | 🟠 Medium | Budget 2 cluster-days + 1 contingency. | Phase 4-γ |
| 7 | Thin data manifold → covariate shift at inference | 🟠 Medium | Trajectory noise augmentation (σ ≈ 0.01–0.05 m) + gain variation + homotopy diversity. | Phase 4-β |
| 8 | No way to validate data quality until Epoch 5 FM training | 🟡 High | Mini-FM sanity gate on 100 empty-scene episodes. | §3 |

---

## 8. What Epoch 4 Does NOT Include

Explicit out-of-scope items to prevent creep:

- ❌ Training an FM-PCC model on the generated data *(Epoch 5)*
- ❌ Visual encoder / image rendering during collection *(Stage 2, Epoch 5+)*
- ❌ MJPC-as-data-collector pipeline *(wrong task — racing ≠ avoidance)*
- ❌ UAV-Flow waypoint-to-action conversion *(wrong sim — UE4 ≠ MuJoCo)*
- ❌ SafeFlowMPC safety-filter-during-training *(Epoch 5 architecture decision)*
- ❌ Real-world pilot demonstrations *(Epoch 7+ if deployment is in scope)*
- ❌ Language-conditioned data collection *(out of project scope)*
- ❌ DAgger-style interactive correction *(Epoch 5+ enhancement)*

---

## 9. Epoch 5 Forward-Look (ideas to file, not to act on now)

These emerged from the AUDIT and are worth remembering but **must not block Epoch 4**:

| Idea | Source | When |
|---|---|---|
| Safety-filter-during-training (DPCC projection at FM interpolation steps) | AUDIT R6, SafeFlowMPC | Epoch 5 architecture |
| `CondOTScheduler` from Meta's `flow_matching` library | AUDIT R1 | Epoch 5 training |
| `Beta(1.5, 1.0)` time sampling | AUDIT R1 | Epoch 5 training |
| EMA model averaging | AUDIT R1 | Epoch 5 training |
| DAgger-lite (one round of on-policy correction) | AUDIT R5.5 | Epoch 5+ |
| Upgrade to 12D format if PID hover instability reappears | AUDIT R4 | Decision gate after mini-FM |

---

## 10. One-Line Summary

> **Resolve three blocking decisions (delta actions, PID fix, schema lock) → mine UAV-Flow stats (4-α, 1 h) → build a homotopy-aware, gain-varied, noise-augmented generator (4-β, 4–6 h) → collect 2k–4k episodes on SLURM (4-γ, 2 cluster-days) → validate with a mini-FM sanity gate before shipping to Epoch 5.**

---

## 11. Addendum — Codebase Verification (2026-06-04)

**Method**: Direct file reads of `uav_naive_test/trajectories.py`, `uav_naive_test/flight_controller.py`, and `uav_naive_test/run_naive.py` against the Phase 4-β assumptions in §2.

**Overall verdict**: Plan logic and research conclusions are sound. One material gap found that affects Phase 4-β scoping.

---

### Gap 1 — CORRECTED: Trajectory factories exist in `uav_env_test/`, not `uav_naive_test/`

Phase 4-β's "Builds on: Epoch 2 `trajectories.py`" referred to the wrong module. Direct file read reveals:

| Factory | `uav_naive_test/trajectories.py` | `uav_env_test/trajectories.py` |
|---|---|---|
| `hover_at` | ✅ | ✅ |
| `step_to` | ✅ | ✅ |
| `circle` | ✅ | ✅ |
| `traverse_line` | ❌ | ✅ (Epoch 3, cosine-profile) |
| `s_curve_path` | ❌ | ✅ (Epoch 3, piecewise) |
| `weave` | ❌ | ✅ (Epoch 3, sinusoidal) |

**Impact**: Original Gap 1 finding was based on the wrong file. `uav_env_test/trajectories.py` is the correct base — all three factories exist. Phase 4-β effort estimate is accurate (4–6 h), not understated.

**Action**: `uav_expert_data_collect/` imports from `uav_env_test.trajectories` and adds homotopy-labelled scene-specific wrappers on top. No new base factories need to be built from scratch.

---

### Gap 2 — `Kp_omega` is currently `[10.0, 10.0, 2.0]` (Decision 2 not yet applied)

`flight_controller.py:70` sets `self.Kp_omega = np.array([10.0, 10.0, 2.0])`. Decision 2 specifies the fix as `[2.5, 2.5, 1.0]`. **This one-line change must be made before any Phase 4-β trial runs.**

The fix is in Decision 2 of this document — the codebase gap just confirms it has not yet been applied.

---

### Confirmed-good items

| Assumption | Status |
|---|---|
| `uav_naive_test/flight_controller.py` — CascadedPID exists | ✅ |
| `uav_naive_test/run_naive.py` — rollout driver exists | ✅ |
| Scene XMLs: `scene_corridor.xml`, `scene_empty.xml`, `scene_pillars.xml`, `scene_s_curve.xml` | ✅ at `d3il/environments/d3il/models/mj/robot/quadrotor/scenes/` |
| `collect_visual_avoiding_data/collect_visual_avoiding_data.py` — Stage 2 template | ✅ |
| `Slurm_Codes/sbatch/uav_env/` — SLURM job scaffolding | ✅ |
| No `uav_expert_data_collect/` dir yet — correct, to be created | ✅ |
