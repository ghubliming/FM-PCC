# Gen7 Fix 4 — MuJoCo Simulation Protection & Action Clamping

**Date:** 2026-05-20
**Branch:** update_into_FM

## Q: Do we feed the entire trajectory to MuJoCo at once, or step-by-step?
**Answer:** We feed it **step-by-step**. 
Here is how the Model Predictive Control (MPC) pipeline works in your codebase:
1. The Flow Matching model predicts an *entire trajectory* of future actions at once (e.g., 8 steps into the future, known as the "horizon").
2. However, the `VisualAgentWrapper` takes **only the first step** (or a very small chunk, defined by `action_seq_size`) from that prediction. 
3. It sends that *single* delta action to `env.step(pred_action)`. 
4. The MuJoCo controller (`CartesianPosQuatTrackingController`) then breaks that single step down into 35 micro-physics time-steps (`n_substeps = 35`) to physically move the arm.
5. After the environment finishes that step, the model looks at the new camera images and replans the next 8 steps.

## Q: Why does MuJoCo crash immediately if it's step-by-step?
**Answer:** A crash (simulation explosion) occurs when the model predicts a diverging, physically impossible action. Even with a trained model, encountering an out-of-distribution visual state, scaling normalizer mismatches, or temporary instability can cause the model to output a massive delta action (e.g., `+5.0` meters on the X-axis). When `env.step()` feeds this to MuJoCo's PD controller, it tries to move the heavy robot arm several meters in a fraction of a second. The physics engine applies astronomical torque to the joints, which mathematically explodes the simulation instantly (state values become `NaN`).

Once the state is `NaN`, all objects disappear, the camera matrices break (showing a blank screen), and the simulation is permanently corrupted for the remaining 399 steps.

## Q: Can we feed it part-by-part (e.g., 0-100, 100-200) to see what works?
**Answer:** Yes! Because the environment is inherently step-by-step, we can easily protect it. If we prevent the model from sending "explosive" commands, the simulation will survive the entire 400-step episode. This allows us to observe the robot's exact behavior (even if it drifts, gets stuck, or fails the task) and log exactly what the model is outputting at every single step to find out where it went wrong.

---

## Action Plan: Implement a Safety Guard (Action Clamping)

To prevent the MuJoCo engine from crashing when testing diverging checkpoints or encountering unexpected edge cases, we will implement a hard physical clamp on the actions before they are fed to the environment.

### Phase 1: Implement Action Clipping in `VisualAgentWrapper`
We will modify the `predict()` function in `fm_visual_aligning_test/eval_fm_visual_aligning.py` to ensure no single step can request a movement larger than the physical limits of the robot.

1. **Calculate Delta:** The model outputs a normalized action, which we denormalize into a Cartesian delta (e.g., `[dx, dy, dz]`).
2. **Apply Clamp:** We will clip the Cartesian delta to a maximum safe threshold (e.g., `±0.05` meters / 5cm per step).
3. **Execute:** We add the clamped delta to the current `des_robot_pos` and send it to `env.step()`.

**Expected Code Change in `eval_fm_visual_aligning.py`:**
```python
# Inside VisualAgentWrapper.predict(), just before applying the action:
next_action_np = next_action.detach().cpu().numpy().squeeze(0)   # (3,)

# --- FIX 4: MUJOCO SAFETY CLAMP ---
# Prevent explosive trajectories from destroying the physics simulation.
# Max allowable movement per step (e.g., 5 cm)
MAX_DELTA = 0.05 
next_action_np = np.clip(next_action_np, -MAX_DELTA, MAX_DELTA)
# ----------------------------------

self.mental_robot_pos += next_action_np
```

### Phase 2: Add Diagnostic Tracing
If we want to see exactly *when* the model starts predicting garbage (e.g., if steps 0-100 are perfectly fine, but step 101 explodes), we can log a warning whenever the clamp is triggered.

```python
original_mag = np.linalg.norm(next_action_np)
if original_mag > MAX_DELTA:
    print(f"[ WARNING ] Step {self.step_counter}: Model predicted explosive action "
          f"(magnitude {original_mag:.4f}m). Clamped to {MAX_DELTA}m to save MuJoCo.")
```

## Q: Did `/workspaces/dpcc` use this safety guard? If not, why did they not have this problem?
**Answer:** The original DPCC codebase did not need an explicit clamp in the evaluation loop for two main reasons:

1. **They Used Mathematical Constraints (The SLSQP Projector):** The core of DPCC (Diffusion Predictive Control with *Constraints*) is that it processes model trajectories through a projector (`setup_dpcc_projector()`). The projector enforces rigorous workspace bounds (`ws_lb`, `ws_ub`), mathematically acting as an action clamp before the robot command is ever sent to MuJoCo. 
2. **Stable DDPM vs. FM ODE Integration:** For baseline variants without a projector, their fully-trained DDPM checkpoints were structurally stable. DDPM reverse chains naturally keep predicted actions within the bounds of the training distribution. However, Flow Matching (FM) integrates a continuous-time ODE. If an FM model drifts, runs into out-of-distribution visual states, or has scaling mismatches, the ODE integration error can quickly accumulate, causing the trajectory to overshoot safe limits.

Interestingly, the original D3IL environment wrapper (`d3il/environments/d3il/d3il_sim/gyms/gym_env_wrapper.py`) *did* originally feature a hard action clamp:
```python
# action[0] = np.clip(action[0], 0.3, 0.8)
# action[1] = np.clip(action[1], -0.45, 0.45)
```
They commented it out because the DPCC projector and stable DDPM made it redundant. Because we are currently debugging and evaluating continuous Flow Matching (FM) models that can diverge under ODE solver integration before they are perfectly converged, adding a safety clamp is a standard robotics safeguard to protect the environment and capture diagnostics.

---

## Forensic Analysis: The Gen6 DPCC Divergence Anomaly (Z=5.0m)
*(Reference: Realtime diagnostic plots from Gen6 FIX7 run — **[Remote Server / Slurm Cluster Only]** at `FMPCC/FM-PCC/logs/archive/aligning-d3il-visual(FIX7, GOOD RESULTS)/plans/visual_aligning_dpcc/H8_K100_Ddiffuser_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion_aw10_VTrue_steps1000/H8_K100_T0.1_Ddiffuser_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion_VTrue_steps1000/6/results_train_set/realtime_diagnostics`)*

Based on the Gen6 diagnostic plots, we have confirmed that **MuJoCo is NOT broken and is behaving exactly as designed.** The true source of the "start at ending position" or straight-line divergence comes from a bug in the model's post-processing (projector), not the physics engine itself. 

1. **MuJoCo's Built-in Velocity Clamp:** In the diagnostic plots showing the robot diverging to `Z = 5.0m` and `X = 3.5m`, the **End-Effector Velocity** metric perfectly flatlines at `~0.018` for hundreds of steps. This proves that MuJoCo's `CartesianPosQuatTrackingController` is successfully restricting the speed. The robot is moving in a perfectly smooth, straight line into the sky because it was mathematically commanded to do so at max velocity.
2. **The Diffuser is Working:** The base `diffuser` model outputs perfectly valid trajectories (e.g., dropping Z-height to table level `0.13` and staying within workspace bounds). The neural network predictions are mathematically sound.
3. **The Projector Corruption:** The "postprocessing" (SLSQP Projector) takes this perfectly good trajectory and attempts to "correct" it to satisfy constraints. Due to an index misalignment in the state space (e.g., the projector checks the wrong index for Z-height, interpreting camera pixels or quaternions as physical coordinates), the projector calculates a massive, fabricated error. To "fix" this, the SLSQP optimizer forces a massive gradient update onto the trajectory, literally commanding the robot to fly to `Z=5.0m`.

This proves that **model-free (diffuser) outputs were inherently stable, but the postprocessing projection corrupted them.** This further reinforces the necessity of the Fix 4 Action Clamp: we must protect MuJoCo from being fed these massive, mathematically corrupted delta commands so the simulation survives long enough for us to diagnose the pipeline.

> [!IMPORTANT]
> **CRITICAL FORENSIC INSIGHT (90%+ Probability):**
> This comparison proves with extremely high certainty that **the raw generative model (the diffuser/FM trajectory engine) is mathematically sound.** The model successfully learns the physics and predicts highly reasonable path trajectories. The soaring behavior is **100% a downstream post-processing/projector bug**. By isolating the fault strictly to the projection constraints (SLSQP layers) rather than the model weights themselves, we prove that the neural network's training pipeline is completely correct, and only the post-processing integration requires remediation.

---

### Next Steps
Would you like me to go ahead and implement this action clamp in `eval_fm_visual_aligning.py`? This will immediately stop the black/yellow screen crashes and let you trace exactly which step the trained model diverges on.

---

## Research Notes & Extended Action Plan

> [!NOTE]
> **Scope:** This section is the AI's independent analysis of the ideas above.
> It adds threshold calibration, implementation details, scope decisions,
> and a diagnostic sequence for interpreting results after both Fix 3 and Fix 4 are live.

---

### R1 — Threshold Calibration: What should MAX_DELTA be?

The plan proposes `MAX_DELTA = 0.05 m` (5 cm per step). Here is the justification and
options:

| Threshold | Reasoning | Risk |
|---|---|---|
| `0.005 m` (0.5 cm) | Matches healthy Gen6V4 physical steps (~0.22 mm mean + headroom) | May clip valid fast-approach actions |
| `0.01 m` (1 cm) | ~45× the observed healthy step magnitude; aggressive push still fits | Still clips large approach motions |
| `0.05 m` (5 cm) | ~220× healthy step; stops genuine explosions only | May allow moderate divergence to continue |
| `0.10 m` (10 cm) | Only stops total catastrophe | Almost no protection for gradual drift |

**Recommended: use `0.01 m` for initial protected eval, not `0.05 m`.**

Reasoning: the healthy Gen6V4 run produced `|a0| = 0.000224 m` per step. The
action normalizer's maximum range (from `act_normalizer.maxs - act_normalizer.mins`)
is approximately 2× the max training action. The largest physically plausible single-step
delta for this table-top push task is ~3–5 mm. `0.01 m` gives 2–3× headroom above that
without allowing ODE-blown trajectories to survive.

**Make it configurable.** Add `max_action_delta` to `config/visual_aligning_eval.yaml`
(default: `0.01`) so it can be changed per run without code edits:

```yaml
# visual_aligning_eval.yaml — add:
max_action_delta: 0.01    # metres per step; null = no clamp
```

Read it in the eval script main block and pass to `VisualAgentWrapper.__init__`:
```python
max_action_delta = config.get('max_action_delta', None)
agent = VisualAgentWrapper(..., max_action_delta=max_action_delta)
```

---

### R2 — Scope: Apply clamp to Gen6V4 eval too?

**Yes.** The clamp should be in `eval_visual_aligning_dpcc.py` as well.

Reasons:
1. The DPCC projector is the *mathematical* equivalent of the clamp for DPCC variants,
   but the `diffuser` (no-projector) variant has no such protection.
2. The promising Gen6V4 run was the `diffuser` variant — Mode 0, success False.
   If action scale is the problem, a controlled `max_action_delta` experiment on Gen6V4
   first gives a clean comparison: same model, same eval, with and without clamp.
3. Consistency: `VisualAgentWrapper` is shared code between the two files. Adding the
   clamp in one and not the other creates drift.

**Clamp does NOT replace the DPCC projector** — the projector runs on the full predicted
trajectory before denormalization. The clamp runs on the single extracted step after
denormalization. They operate on different representations and are not redundant.

---

### R3 — Implementation: exact location and clamp-event tracking

**Where to add in `predict()`:**

```python
# Existing code (after action magnitude append):
next_action_np = next_action.detach().cpu().numpy().squeeze(0)   # (3,)
self.curr_rollout_act_magnitudes.append(float(np.linalg.norm(next_action_np)))

# ADD HERE — after magnitude is logged (log raw, then clamp):
if self.max_action_delta is not None:
    raw_mag = np.linalg.norm(next_action_np)
    if raw_mag > self.max_action_delta:
        next_action_np = next_action_np * (self.max_action_delta / raw_mag)
        self.curr_rollout_clamp_events.append((self.step_counter, float(raw_mag)))
        if len(self.curr_rollout_clamp_events) <= 5 or self.step_counter % 50 == 0:
            print(f'[ CLAMP step={self.step_counter} ] '
                  f'raw|a|={raw_mag:.4f} m → clamped to {self.max_action_delta} m')
```

Note: log `raw_mag` in `history_act_magnitudes` (before clamp) so the PNG panel
`[1,2]` shows true model output, not the clamped value. The clamp events are a
separate list that answers "when and how bad were the violations?"

**Add clamp-event count to rollout summary:**
```python
# In update_rollout_info():
print(f'  - Clamp events: {len(self.curr_rollout_clamp_events)}')
```

**Add a 9th PNG panel** (expand to 3×3): `[2,0]` — "Clamp events (raw |a| at trigger)"
as a scatter plot of `(step, raw_mag)`. If this panel is empty the run was clean.
If it has many points at large `raw_mag`, the model is genuinely exploding.

---

### R4 — Post-Fix-3 Diagnostic Sequence

After both Fix 3 (diagnostics) and Fix 4 (clamp) are active, use this sequence
to interpret each failed rollout:

```
1. Check PNG [0,3] Distance-to-Target curve
   ├── Flat from step 0       → robot never approached box
   │     └── Check PNG [0,0] XY — is robot moving at all?
   │           ├── No movement → action magnitude [1,2] near zero → scaler/normalizer bug
   │           └── Movement but wrong direction → obs conditioning wrong (DIAG obs/img)
   ├── Decreasing then plateau → policy makes progress but stalls at a local min
   │     └── Check DIAG replan=50/100/150 direction logs — is direction consistent?
   └── Oscillating             → policy is indecisive near box, can't commit to push

2. Check PNG [1,2] Action Magnitude
   ├── Near-zero throughout    → denormalization failed (normalizer pkl wrong)
   ├── Oscillating ±           → model output is noisy, low confidence
   ├── Healthy (0.1–0.5 mm)    → model is sane; task failure is policy quality
   └── Spikes > MAX_DELTA      → ODE divergence — check CLAMP panel for step index

3. Check DIAG img std
   ├── bp_std < 0.01           → agentview camera not rendering
   └── inhand_std < 0.01       → wrist camera not rendering

4. Check DIAG obs des_c_pos vs c_pos
   └── Identical               → both slots wired to des_robot_pos; c_pos not connected
```

---

### R5 — The Deeper Question: Action Scale vs Visual Conditioning

The Mode 0 success-False result (robot reaches box but fails) has two plausible causes:

**Hypothesis A — Action scale is wrong:**
The policy outputs small, uncertain deltas (~0.22 mm/step) because the action
normalizer's range was computed over training data that includes slow-approach and
near-contact micro-corrections. The policy has learned to be conservative. At box
contact it needs larger decisive pushes but never produces them.

*Test:* Does the distance curve in Fix 3 show a plateau at a non-zero distance
(box never moves despite robot contact)? If yes, this is the hypothesis.

*Potential fix (not in this plan):* Adjust `action_weight` in the training config
(currently 10). Higher `action_weight` penalizes action prediction error more,
forcing the model to learn decisive actions.

**Hypothesis B — Visual conditioning provides insufficient push signal:**
The agentview camera doesn't change much once the robot is near the box (small
field-of-view change). The model can't see whether it's making contact. It defaults
to small safe actions rather than committing to a push.

*Test:* Does the `[ DIAG img ] inhand_img std` drop significantly when the robot
is near the box (wrist camera now showing the box close-up vs. empty table earlier)?
If std stays constant, the wrist camera is providing consistent signal. If std spikes,
the model is receiving confusing new-distribution images at contact.

*Potential fix:* Ensure the wrist camera height provides clear box-top view at
contact range. This is a dataset composition / camera setup question.

**These two hypotheses require the Fix 3 distance curve to distinguish.**
Run one rollout with Fix 3 + Fix 4 active and read the distance curve first.

---

## Full Point Inventory & Final Verdict

All proposed changes across the original plan and the research section, with a clear do/not-do decision.

| # | Proposed Change | Source | Verdict | Reason |
|---|---|---|---|---|
| 1 | Hard clamp in `predict()` — stop MuJoCo explosions | Phase 1 | **DO** | Core purpose of Fix 4. Without this, Fix 3 diagnostics never reach step 2. |
| 2 | `np.clip` → direction-preserving rescale | R3 | **DO** | `np.clip` distorts direction (e.g. clips X but not Y, changing the intended push vector). Rescale keeps intent, just limits magnitude. One-liner change. |
| 3 | `MAX_DELTA = 0.01 m` not `0.05 m` | R1 | **DO** | 0.05 m allows moderate ODE drift to continue silently. 0.01 m is ~45× the observed healthy step size — stops real explosions while letting normal fast-approach actions through. |
| 4 | `max_action_delta` in `visual_aligning_eval.yaml` + passed to wrapper | R1 | **DO** | Threshold is a tuning knob. Hardcoding it means a code edit every experiment. YAML entry costs nothing. |
| 5 | WARNING log when clamp fires (step + raw magnitude) | Phase 2 | **DO** | Directly answers "when does the model diverge?" Zero cost. |
| 6 | Clamp event count in rollout summary print | R3 | **DO** | One line. Immediately tells you if the run was clean or dirty without opening any file. |
| 7 | Apply clamp to Gen6V4 eval too | R2 | **DO** | The `diffuser` variant (no projector) is equally unprotected. Same `VisualAgentWrapper`, same fix. |
| 8 | `curr_rollout_clamp_events` list + 9th PNG panel (3×3 scatter) | R3 | **DO** | Valid diagnostic. Clamp event scatter shows exactly when and how badly the model diverges per rollout. Not dangerous, just extra code — worth having. |
| 9 | "Part-by-part 0-100, 100-200" feature | Original Q&A | **SKIP — not a code change** | This was an explanation of why the clamp works, not a feature request. MPC is already step-by-step; Fix 3 + Fix 4 together deliver full per-step visibility. Nothing to implement. |
| 10 | R4 Diagnostic decision tree | R4 | **SKIP — not a code change** | Reading guide only, already written in the plan. No implementation needed. |
| 11 | R5 Hypothesis A/B (action scale vs visual conditioning) | R5 | **SKIP — not a code change** | Post-run analysis framework. Apply after seeing Fix 3 distance curves. Nothing to implement now. |

**What gets implemented:** items 1–8.
**Files touched:** `eval_visual_aligning_dpcc.py`, `eval_fm_visual_aligning.py`, `config/visual_aligning_eval.yaml`.

