# Gen11 Epoch 4 — Technical Audit & Recommendations

**Date**: 2026-06-03  
**Scope**: Audit of [RESEARCH.md](file:///workspaces/FM-PCC/logs_in_develop/Gen11/Epoch4_expert_data/RESEARCH.md) and [EXPERT_DATA_PROVENANCE.md](file:///workspaces/FM-PCC/logs_in_develop/Gen11/Epoch4_expert_data/EXPERT_DATA_PROVENANCE.md)  
**Method**: Cross-referenced against codebases at `/workspaces/{mujoco_mpc, UAV-Flow, SafeFlowMPC, d3il}`, Epoch 2/3 closure docs, and current literature on flow matching for constrained planning.

---

## Executive Summary

The two documents are **unusually strong** for a research planning phase — the provenance analysis is genuinely publication-grade, and the RESEARCH.md makes defensible engineering choices. This audit identifies **no fatal flaws**, but surfaces **9 actionable recommendations** and **2 major opportunities** that the documents miss entirely. A notable adjacent reference: **SafeFlowMPC** (ICRA 2026, vendored at `/workspaces/SafeFlowMPC/`) tackles a related problem class and its dataset-creation ideas are worth borrowing — but see R1 caution before reading further.

---

## Part I: What the Documents Get Right

> [!TIP]
> These are genuine strengths worth preserving, not just pleasantries.

### ✅ S1. The "position-delta as action" convergence finding is load-bearing

PROVENANCE §3a demonstrates that D3IL (`vel_state = des_c_pos[1:] - des_c_pos[:-1]`), UAV-Flow (`_transform_to_local_frame` → body-frame Δpose), and your planned manual generation all converge on the same action convention. This is the single most important finding in either document — it proves format alignment without conversion loss. **Keep this front and center.**

### ✅ S2. The phased answer (§3.5 → §3c.5 → §3d.4) is honest engineering

The "manual gen for validation, real data for deployment" split, evolved across three refinements, is the right answer. Importantly, the docs *don't* overclaim — they acknowledge that manual data is wrong for sim-to-real and language-conditioning, which are out of scope.

### ✅ S3. The two-stage collection pattern (§3c) is architecturally clean

State-first, images-later via `replay_and_capture` is correct, matches Gen9 precedent, and decouples the two hardest iteration cycles. The determinism analysis (§3c.4) is appropriately cautious.

### ✅ S4. The spectrum decomposition (§3d) resolves a real ambiguity

The A→F spectrum is a genuine contribution to your own internal clarity. Earlier sections *were* ambiguous about "manual," and the spectrum makes the layered recommendation (bulk D, optional C/F/A) defensible.

### ✅ S5. MJPC dismissal is correct but for a subtle reason

RESEARCH §2.3 dismisses MJPC as "wrong task" (racing vs. avoidance). This is correct, but the *deeper* reason (which the doc doesn't fully articulate) is that MJPC's predictive-sampling planner produces **optimal** trajectories for its cost function — training FM on optimal-only demos can cause issues (see R5 below).

### ✅ S6. The 9D format lock from Epoch 2 is carried forward cleanly

RESEARCH §1's default of 3-D position target + 6-D `[p, v]` obs matches the Epoch 2 closure's "9D locked" decision. No format surgery needed.

### ✅ S7. UAV-Flow trajectory count correction

PROVENANCE §3a.5 corrects RESEARCH's "~100" to 273 based on actual file count. Good practice — the original document should be updated per PROVENANCE §5.

---

## Part II: Recommendations

### R1. Study SafeFlowMPC's design ideas as a reference — but keep FM-PCC as the base

> [!IMPORTANT]
> This is a **reference-only** recommendation.

> [!CAUTION]
> **SafeFlowMPC is an auxiliary idea-source, not a superior system to adopt.** FM-PCC is the base architecture of this project. SafeFlowMPC solves a related but different problem (manipulator trajectory planning for a Franka arm, joint-space, with Acados as the safety backend), and its codebase should not be treated as something FM-PCC needs to conform to or be replaced by. The value is in **selectively borrowing design ideas** that are portable into the FM-PCC stack — nothing more. Do not let studying SafeFlowMPC create scope creep or architectural second-guessing of FM-PCC's established decisions.

**SafeFlowMPC** (ICRA 2026, [README](file:///workspaces/SafeFlowMPC/README.md)) is a flow matching policy with safety projection for robot arm trajectory planning. Its dataset-creation pipeline reveals a few design choices that are *conceptually* worth understanding, even though FM-PCC's implementation will differ significantly in dynamics, state space, action space, and safety formulation:

| SafeFlowMPC design idea | What they do | Portable to FM-PCC? |
|---|---|---|
| **Safety filter on training data** | Each training sample is passed through `SafetyFilter.step()` at every FM interpolation step — the velocity field learns constraint-respecting *intermediate* states, not just endpoints | ⚠️ Conceptually interesting. FM-PCC currently projects only at inference (via DPCC). Worth noting as a future Epoch 5 experiment, but **do not block Epoch 4 data collection on this** |
| **Pre-train unsafe → fine-tune safe** | Two-phase training: first without safety, then fine-tune with safety-filtered intermediate states | 🔵 Future only — relevant to Epoch 5 training design, not Epoch 4 data collection |
| **Conditional OT scheduler** | Uses `CondOTScheduler` from Meta's `flow_matching` library | 🔵 Check if FM-PCC already uses this; if not, worth a one-line swap in Epoch 5 |
| **Beta(1.5, 1.0) time sampling** | Biases training samples toward t≈1 (near the target), not uniform t | 🔵 Low-effort improvement to check in Epoch 5 training |
| **4000 VP-STO trajectories** | Their expert = variational-path optimizer, not PID | ❌ Their expert is not portable (different robot, different physics). PID + MJPC oracle (R2) is FM-PCC's equivalent |

**Action (scoped)**: Skim SafeFlowMPC's [create_imitation_dataset_vpsto.py](file:///workspaces/SafeFlowMPC/dataset_creation/create_imitation_dataset_vpsto.py) for the safety-filter-during-interpolation idea only. File the training-phase ideas (OT scheduler, Beta sampling, unsafe→safe fine-tuning) as Epoch 5 candidates. Do not restructure Epoch 4 data collection around SafeFlowMPC's pipeline.

---

### R2. MJPC is more accessible than RESEARCH §2 suggests — reconsider as a secondary data source

RESEARCH §2.2 estimates "1–2 days" to build a C++ MJPC logger and dismisses it. But the vendored `mujoco_mpc` has a **Python API** that the document doesn't mention:

```python
# From /workspaces/mujoco_mpc/python/mujoco_mpc/demos/agent/cartpole.py
agent = agent_lib.Agent(task_id="Quadrotor", model=model)
for t in range(T):
    agent.set_state(time=data.time, qpos=data.qpos, qvel=data.qvel, ...)
    for _ in range(10):
        agent.planner_step()
    data.ctrl = agent.get_action()
    mujoco.mj_step(model, data)
    # Log (data.qpos, data.qvel, data.ctrl) → your format
```

The [cartpole demo](file:///workspaces/mujoco_mpc/python/mujoco_mpc/demos/agent/cartpole.py) already does exactly "run MJPC and record rollouts" in ~50 lines of Python. The quadrotor task XML at [task.xml](file:///workspaces/mujoco_mpc/mjpc/tasks/quadrotor/task.xml) has 11 waypoints + 8 racing gates — and the `TransitionLocked` function at [quadrotor.cc:60](file:///workspaces/mujoco_mpc/mjpc/tasks/quadrotor/quadrotor.cc#L60) auto-advances waypoints when position error < 0.5 m.

**The blocker** is building the C++ `agent_server` binary (requires gRPC + CMake), which is nontrivial on Slurm. But on a local machine it's feasible. MJPC trajectories would serve as your spectrum row **F** (optimal-control oracle) demos with ~1 day of effort, not 2 days.

**Action**: Update RESEARCH §2 to note the Python API exists. If you have a local machine with CMake, try `cd /workspaces/mujoco_mpc && mkdir build && cd build && cmake .. && make -j` as a side-quest — the payoff is high-quality optimal trajectories that complement PID's simpler demos.

---

### R3. Address the mono-modality risk more concretely

Both documents acknowledge the mono-modality issue (RESEARCH §4.5, PROVENANCE §3.2(d)) but the mitigation is vague: "multiple gain sets / multiple goal biases." This needs to be more concrete for the dataset generator.

**Concrete proposal for multi-modal PID data**:

For each obstacle layout, define **2–4 homotopy classes** — topologically distinct routes through the obstacles. For a pillar field with 3 pillars, the homotopy classes are "pass left/right of pillar N":

```
Homotopy A: (L, L, L) — pass left of all three pillars
Homotopy B: (L, R, L) — pass left, right, left
Homotopy C: (R, L, R) — opposite of B
Homotopy D: (R, R, R) — pass right of all three
```

Each homotopy class gets its own reference trajectory generator. Within a class, vary start/end positions and speed profiles. The FM velocity field then has to model `P(route | start, goal)` as a multi-modal distribution — which is exactly the property that justifies using flow matching instead of a regression baseline.

**Action**: Add a `homotopy_label` field to the dataset schema. When generating trajectories for pillar/corridor scenes, enumerate at least 2 homotopy classes per scene.

---

### R4. The 9D format has a silent assumption about acceleration

Epoch 2's closure locked 9D `[p, v, a]` for the *controller* (feed-forward acceleration is load-bearing for PID tracking). But RESEARCH §1 proposes 9D `[act(3) ‖ p(3), v(3)]` for the *FM dataset* — note that `a` (acceleration) is in the controller format but **not** in the FM format.

This creates a gap: the controller *needs* acceleration feed-forward (Epoch 2 §4: "Task C 9D works *because* of `a_des`"), but the FM policy will only output position targets. The policy's output goes through `vel_state = des_c_pos[1:] - des_c_pos[:-1]` to become velocity, and a second difference to become acceleration — but finite-differencing introduces noise.

**Two options**:

| Option | FM output | Controller input | Risk |
|---|---|---|---|
| A (current plan) | 3D position target only | PID must compute `a` via double-differencing | Noisy `a` → hover instability may resurface at low speeds |
| B (richer format) | 6D `(p_target, v_target)` or 9D `(p, v, a)` | PID uses explicit velocity/acceleration feed-forward | Harder to learn but safer controller performance |

RESEARCH §1 already lists this as a variant (12-D with velocity target). The Epoch 2 evidence strongly suggests **at minimum providing velocity feed-forward** alongside position targets. Consider 12D `[act(6)=[p_des, v_des] ‖ obs(6)=[p, v]]` as the default, with 9D as a fallback.

> [!WARNING]
> If you proceed with 9D position-target-only and the controller double-differences to get acceleration, the same limit-cycle instability from Epoch 2 §3 may reappear during low-speed FM-planned segments. This is not theoretical — Epoch 2 proved it happens.

---

### R5. Beware of "optimal-only" training data for flow matching

Flow matching learns a velocity field that transports noise to data. If all training data comes from a single deterministic expert (one PID with one gain set), the "data manifold" is a **thin curve**, not a distribution. The velocity field will correctly point toward that curve but will have no useful gradient *along* the curve — any perturbation at inference time pushes you off-manifold into an untrained region.

This is the flow-matching analogue of **covariate shift** in behavioral cloning. Recent literature (2024–2025) confirms that:
- Flow matching benefits from **multi-modal** demonstrations (multiple valid solutions per initial condition)
- **Action chunking** (your H=8 horizon) partially mitigates single-step compounding errors but doesn't fix thin-manifold issues
- **DAgger-style** interactive correction is the gold standard but requires simulator-in-the-loop at training time

**Mitigations** (choose ≥2):

1. **Add Gaussian noise to expert trajectories** during dataset creation (not at training time). `p_noisy = p_expert + ε`, `ε ~ N(0, σ²I)` with σ ≈ 0.01–0.05 m. This thickens the data manifold.
2. **Multiple homotopy classes** (R3 above).
3. **Multiple PID gain sets** — vary `Kp_pos`, `Kp_vel` by ±20% across trajectory batches.
4. **Random initial-condition perturbations** that push the drone off the nominal trajectory early, then let PID recover. The recovery sub-trajectory teaches the policy what "corrective" actions look like.
5. **DAgger-lite**: After training an initial FM policy, roll out that policy in MuJoCo, query the PID expert for the "correct" action at the policy's *actual* visited states, add those `(state_visited, action_expert)` pairs to the dataset, retrain. One round of DAgger dramatically reduces compounding error.

> [!NOTE]
> SafeFlowMPC addresses this differently: their training samples include **safety-filtered intermediate states** at 50 interpolation steps between noise and target. This implicitly teaches the velocity field how to handle off-manifold states by always projecting them back to feasibility. Your DPCC projector could serve the same role if applied during training, not just inference.

---

### R6. Consider applying DPCC projection during training, not just inference

Currently, FM-PCC's architecture is:
```
Training:  FM learns v(x,t) from expert (state, action) pairs
Inference: FM generates action → DPCC projects action onto constraint set
```

SafeFlowMPC's architecture is:
```
Training:  Safety filter applied at EVERY interpolation step during flow matching
           → velocity field learns to produce constraint-respecting intermediate states
Inference: Acados NLP solves for safe trajectory given FM's output
```

The difference is profound. SafeFlowMPC's FM *already knows* how to stay feasible because it was trained on feasibility-filtered interpolants. Your FM has never seen a constraint; it relies entirely on post-hoc DPCC projection to save it.

**Potential improvement for Epoch 5+**: During FM training, at each time step t ∈ [0,1], compute `x_t = (1-t)·noise + t·data`, apply DPCC projection to `x_t`, and train the velocity field against the *projected* interpolant's velocity. This teaches the velocity field to route around obstacles *during generation*, not just at the endpoint.

---

### R7. The UAV-Flow stats table has a suspicious velocity inconsistency

RESEARCH §3.3 reports "Max horizontal velocity: 30 cm/s (0.3 m/s)" but also "Δ position per step: 2-5 cm." At 30 Hz, 2–5 cm/step = 60–150 cm/s, which is **5× higher** than the 30 cm/s headline. The document notes "need to verify" but doesn't resolve it.

This matters because the velocity cap on your trajectory generator (§3.3: "cap velocity at ≈ 0.3 m/s") may be set 5× too low. If UAV-Flow trajectories actually move at 0.6–1.5 m/s, your manually generated data at 0.3 m/s will be unrealistically slow — the FM policy will learn a "timid" planner.

**Action**: Run the actual stats sweep on the 273 trajectories (PROVENANCE §3a.5 confirms we have them). Compute `Δp/Δt` per step, plot the velocity histogram, and set the generator's velocity range accordingly. This is Phase 4-α work (RESEARCH §5.1) — 30 minutes, not 1 hour.

---

### R8. Missing: dataset schema specification

RESEARCH §4.1 shows the 9D per-step format but doesn't specify the **dataset file schema** — what fields go in each pickle, how episodes are indexed, how the dataloader discovers files. D3IL's schema (at [aligning_dataset.py](file:///workspaces/FM-PCC/d3il/environments/dataset/aligning_dataset.py)) uses `train_files_{fraction}_.pkl` with `env_state['robot']['des_c_pos']` etc. SafeFlowMPC uses `.npz` with keys `trajectories`, `c_data`, `samples`, `dsamples`, `t_samples`.

Your dataset writer (`dataset_writer.py` in §5.3) needs a schema defined *before* implementation. Proposal:

```python
# Per-episode file: uav_expert_data/{scene}/{ep_id}.pkl
{
    "episode_id": str,
    "scene": str,           # "corridor" | "s_curve" | "pillars" | "empty"
    "homotopy": str,        # "(L,R,L)" or "N/A" for empty
    "controller": str,      # "pid_default" | "pid_high_gain" | "mjpc" | "teleop"
    "dt": float,            # seconds between steps (1/30 for 30 Hz)
    "obs": np.ndarray,      # (T, obs_dim)  — [p(3), v(3)]
    "actions": np.ndarray,  # (T-1, act_dim) — [Δp_des(3)] or [p_des(3)]
    "targets": np.ndarray,  # (T, 3) — the reference position target at each step
    "obstacles": list,      # [{type, center, radius/half_extents}, ...]
    "metadata": {
        "start_pos": list,
        "end_pos": list,
        "total_time": float,
        "controller_gains": dict,
    }
}
```

> [!TIP]
> Including `obstacles` per-episode enables training a **goal-conditioned** or **obstacle-conditioned** policy later without re-collecting data. SafeFlowMPC conditions on collision-body positions (`p_cols` in their `conditional_data1`); you should too.

---

### R9. The "one cluster-day" estimate is optimistic

RESEARCH §4.3 estimates ~6–8 hours total. But this doesn't account for:

| Hidden cost | Time |
|---|---|
| Debugging the first trial's MuJoCo rendering on Slurm (GPU/EGL issues) | 1–2 h |
| Discovering the s_curve PID instability affects random starts (Epoch 3 §4) | 1 h to diagnose + fix `Kp_omega` |
| Dataset format iteration (your first pickle won't match what the FM dataloader expects) | 2–3 h |
| UAV-Flow stats sweep (Phase 4-α) including the velocity inconsistency (R7) | 1 h |
| Spot-checking GIFs for 4 scenes × multiple homotopy classes | 1–2 h |

A more realistic estimate: **2 cluster-days** (one for state generation + debugging, one for validation + format iteration), with a **third day** if you add the teleop rig (§3b.4).

---

## Part III: Risks Not Covered in the Documents

### ⚠️ Risk 1: PID instability on randomized initial conditions

RESEARCH §5.4 lists this as "Medium" severity. Based on Epoch 2 and 3 evidence, it should be **High** for any trial where the drone starts with nonzero velocity toward an obstacle. The s_curve failure (41% obstacle contacts) was triggered by zero-velocity waypoints; randomized starts with aggressive angles will trigger the same mode.

**Mitigation**: Fix `Kp_omega` *before* Epoch 4 data collection, not after. It's a one-line change and Epoch 2 already specified the fix: `Kp_omega = [2.5, 2.5, 1.0]`. Running 1000 trials with an unstable controller will waste the cluster-day.

### ⚠️ Risk 2: No validation that FM can actually learn from this data

Neither document addresses: *how will you know the data is good enough for FM training?* The implicit assumption is "collect data → train FM in Epoch 5 → see if it works." But if FM fails in Epoch 5, you won't know whether the failure is:
- (a) Bad data (wrong distribution, too few homotopy classes, mono-modal)
- (b) Bad FM hyperparameters (wrong H, wrong t-schedule, wrong architecture)
- (c) Bad DPCC projection (constraint formulation issue)

**Mitigation**: Before full-scale collection, train a **tiny FM** on 100 manually generated trajectories from the `empty` scene (no obstacles). If FM can reproduce the PID's empty-scene trajectories at < 0.1 m RMS, the data pipeline is correct. If not, fix data issues before scaling up.

### ⚠️ Risk 3: Action convention ambiguity — absolute position vs. delta position

RESEARCH §4.1 shows `action_t = [p_x_des, p_y_des, p_z_des]` — **absolute** position targets. But D3IL's convention (PROVENANCE §3a.2) is `vel_state = des_c_pos[1:] - des_c_pos[:-1]` — **delta** positions. UAV-Flow is also deltas (`_transform_to_local_frame`).

The FM training code treats actions as things to be predicted — if you mix absolute and delta conventions, the velocity field's output semantics are wrong. Both D3IL and UAV-Flow use deltas. **Your dataset should also use deltas** (`Δp_des = p_des[t+1] - p_des[t]`) unless you have a specific reason to use absolute targets.

The PROVENANCE doc identifies this convergence (§3a.6) but RESEARCH §4.1 contradicts it by showing absolute targets. Resolve this before writing `dataset_writer.py`.

---

## Part IV: Opportunities

### 💡 Opportunity 1: Borrow specific training ideas from SafeFlowMPC — not the codebase

> [!CAUTION]
> **Do not treat SafeFlowMPC as a direct code reference to adopt.** FM-PCC is the base system; SafeFlowMPC is a separate ICRA 2026 paper solving a different problem (Franka arm, joint-space planning, Acados backend). Its architecture, data format, and safety formulation are not directly portable. The opportunity here is to cherry-pick **two or three isolated ideas** from their training setup — nothing more.

The specific ideas worth borrowing (all Epoch 5 scope, not Epoch 4):
- **`flow_matching` library** (`from flow_matching.path import AffineProbPath`, `CondOTScheduler`): a drop-in scheduler improvement. Check if FM-PCC's training already uses conditional OT; if not, it's a one-line swap.
- **Beta-distributed time sampling** (`Beta(1.5, 1.0)`, biases toward t≈1): improves convergence near the data manifold. One-line change to the training loop.
- **EMA model averaging**: stabilises training. If FM-PCC doesn't already use EMA, worth adding.

These are **library-level ideas**, not architectural ones. The `TemporalUnet` itself may or may not match FM-PCC's U-Net — verify before drawing any conclusions. Ignore their VP-STO expert, Acados safety filter, and joint-space data format entirely.

### 🚀 Opportunity 2: MJPC Python API as a "free" oracle baseline

The [agent.py](file:///workspaces/mujoco_mpc/python/mujoco_mpc/agent.py) `best_trajectory()` method returns the *full planned trajectory* (states + actions + times) in a single call. If you can build MJPC, you get an oracle that produces dynamically feasible, cost-optimal trajectories for any waypoint configuration — no PID tuning, no trajectory factory, no gain-set enumeration. These would serve as your "upper-bound expert" (spectrum row F) at near-zero marginal cost per trajectory.

---

## Priority-Ordered Action List

| # | Action | Effort | Impact | When |
|---|---|---|---|---|
| **1** | Read SafeFlowMPC's dataset pipeline (R1) | 2 h | 🔴 Critical — may change your training architecture | Before writing any Epoch 4 code |
| **2** | Fix `Kp_omega` before data collection (Risk 1) | 5 min | 🔴 Critical — unfixed PID will corrupt 30%+ of trials | Before Phase 4-β |
| **3** | Resolve absolute vs. delta action convention (Risk 3) | 30 min | 🔴 Critical — wrong convention = unusable dataset | Before writing `dataset_writer.py` |
| **4** | Run UAV-Flow velocity stats (R7) | 30 min | 🟡 High — velocity cap may be 5× wrong | Phase 4-α |
| **5** | Define dataset schema (R8) | 1 h | 🟡 High — blocks `dataset_writer.py` | Phase 4-α |
| **6** | Add homotopy labels + multi-class generation (R3) | 2 h | 🟡 High — mono-modal data wastes FM's capability | Phase 4-β |
| **7** | Train tiny FM on 100 empty-scene trials (Risk 2) | 4 h | 🟡 High — catches pipeline bugs early | After Phase 4-β, before 4-γ |
| **8** | Consider 12D format with velocity feed-forward (R4) | 1 h design | 🟠 Medium — mitigates PID instability at low speed | Phase 4-α (decision point) |
| **9** | Explore MJPC Python API build (R2) | 4 h | 🟠 Medium — high-quality secondary data | Side-quest, any time |
| **10** | Add trajectory noise augmentation (R5) | 1 h | 🟠 Medium — thickens data manifold | Phase 4-β |
| **11** | Study safety-filter-during-training idea (R6) | Research only | 🔵 Future — Epoch 5 architecture decision | After Epoch 4 data lands |

---

## One-Line Verdict

The Epoch 4 plan is **sound in its conclusions** (manual generation is right, phased approach is right, two-stage collection is right) but **incomplete in its execution details** — the action convention is ambiguous, the PID controller is still broken, the dataset schema is unspecified, and a directly relevant ICRA 2026 paper (SafeFlowMPC) in your own workspace has been overlooked. Fix the top-3 items before writing code and this epoch will land cleanly.
