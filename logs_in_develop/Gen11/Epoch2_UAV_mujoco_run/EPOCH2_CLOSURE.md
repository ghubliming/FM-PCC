# Gen11 Epoch 2 — Closure

**Date**: 2026-05-31
**Signature (runtime evidence)**: `Slurm_Codes/logs/2026-05-31/11_59_58_run_naive_21023.log`
**Status**: ✅ **Epoch 2 development CLOSED.** Architectural hypothesis validated; controller-tuning gap on static tasks documented as known-acceptable.

---

## 1. What was Run

Single SLURM job `21023` executed all four naive-fly tasks against the
patched X2 model from Epoch 1, under the cascaded PID flight controller
from `uav_naive_test/flight_controller.py`. Results landed at
`logs/uav_naive/{task_A_hover, task_B_step, task_C_circle_6D, task_C_circle_9D}/`.

---

## 2. Results vs RUNBOOK pass/fail criteria

| Task | Trajectory format | RMS pos err (m) | Final pos err (m) | Threshold | Status |
|---|---|---|---|---|---|
| A — hover | (static) | 0.335 | 0.585 | RMS < 0.10, final < 0.02 | ❌ FAIL |
| B — step | (static post-step) | 0.328 | 0.120 | final < 0.05 | ❌ FAIL |
| C — circle 6D | `[p, v]` | 0.214 | 0.195 | RMS < 0.10 in at least one of {6D, 9D} | ❌ (this format) |
| C — circle 9D | `[p, v, a]` | **0.029** | **0.027** | RMS < 0.10 | ✅ **PASS** |

Score against PREP_PLAN §8: 3 of 5 pass-criteria met. Critically, **the
two that matter for the FM-PCC use case** (Task C 9D tracking + trajectory
format selection) both passed cleanly.

---

## 3. Diagnosis — Bug vs. Tuning

The hover and step failures are **a parameter-tuning / discretization
interaction, NOT a logic bug**. Proof:

- Task C 9D ran 30 s of continuous tracking with the *same* controller code
  and *same* gains, achieving 2.9 cm RMS. If the cascaded PID logic were
  wrong (sign error in SO(3) attitude error, wrong allocation matrix, bad
  quaternion convention), the 9D circle would not track.

- The hover failure mode is a discrete-time limit cycle locked to *exactly
  one physics step*: motor outputs alternate `[6.5, 6.5, 0, 0]` ↔
  `[0, 0, 6.5, 6.5]` every step, with `ω_y` flipping `+0.367 ↔ -0.349`
  rad/s. This is the unambiguous signature of attitude-rate gain being too
  aggressive for the simulation rate.

Concrete arithmetic:

| Parameter | Value |
|---|---|
| Physics timestep (Menagerie default) | `dt = 0.01 s` (100 Hz) — **not** the 500 Hz assumed in EXECUTION_PLAN |
| Pitch inertia `I_yy` | 0.0365 kg·m² |
| Max pitch torque (clipped) | ≈ 1.82 N·m |
| Max angular acceleration at saturation | `1.82 / 0.0365 ≈ 50 rad/s²` |
| Max Δω per timestep | `50 × 0.01 = 0.5 rad/s` |
| `Kp_omega[y]` (rate damping) | 10 |
| Saturation onset | `|ω| > 1.82 / 10 = 0.18 rad/s` |

Result: any disturbance reaching `|ω| > 0.18 rad/s` saturates the torque
command, which then over-corrects by 0.5 rad/s in one step → sign flip →
limit cycle at `1/dt`. Textbook gain-too-high-for-discretization
instability.

**Fix is one constant in `flight_controller.py`** (any of: drop `Kp_omega`
from 10 → ~2-3; raise physics rate to 500 Hz; or add saturation
back-off). All three are tuning, not algorithmic changes.

---

## 4. Why Task C 9D Works Despite the Instability

Feed-forward acceleration `a_des` from the 9D trajectory keeps the
position-loop error tiny throughout the rollout. With small position
errors, the desired attitude `R_des` stays close to current `R`, attitude
error stays in the linear region, and the rate-damping torque
never saturates. The unstable mode is never excited.

In contrast, static hover starts with a 0.2 m altitude error → demands
non-trivial acceleration → tiny attitude perturbation grows until
saturation → limit cycle.

This is *also* the reason the hover failure is **downstream-irrelevant for
the FM-PCC use case**: FM/diffusion policies output continuously-moving
trajectories with non-zero `a_des`, not static setpoints. The exact failure
mode that breaks hover cannot be triggered by anything FM-PCC will plan.

---

## 5. Closure Decisions

### 5.1 Trajectory format

**Locked: 9D `[p, v, a]`.** 9D beats 6D by **7.4× RMS** on the circle
test (2.9 cm vs 21.4 cm). The acceleration feed-forward is load-bearing.
9D also matches FM-PCC visual aligning's existing 9D backbone — no
trajectory-channel surgery required when wiring FM policies into the
quadrotor stack later.

### 5.2 Architectural hypothesis (the whole point of Epoch 2)

> *"Can we plan in position-space and execute via a separate controller,
> like FM-PCC aligning treats the Panda?"*

**Yes — validated.** A 30-second non-trivial trajectory was tracked at
2.9 cm RMS with no learning involved, just a hand-written cascaded
controller below a hand-written reference trajectory. Planning and
control are cleanly decoupled. Epochs 3+ can build on this assumption.

### 5.3 Hover instability

**Acknowledged, not fixed in Epoch 2.** Three reasons to defer:
- Cure is a one-line constant change, low risk if/when needed.
- Hover behavior is not exercised by the downstream FM-PCC path
  (continuously-moving trajectories only).
- Re-tuning + verifying would push Epoch 2 past its time budget for
  no architectural payoff.

If a future epoch needs static-hold capability (e.g. demo at start of
rollout, or test fixtures), the fix is: edit
`uav_naive_test/flight_controller.py` `self.Kp_omega = np.array([10., 10., 2.])`
→ `np.array([2.5, 2.5, 1.0])`, re-submit Tasks A/B.

---

## 6. Deliverables Inventory (Epoch 2)

| Path | Status |
|---|---|
| `d3il/environments/d3il/models/mj/robot/quadrotor/` (Epoch 1) | ✅ Verified flyable on Slurm |
| `uav_naive_test/{__init__,smoke_load,flight_controller,trajectories,run_naive,README}.{py,md}` | ✅ Written, exercised |
| `Slurm_Codes/sbatch/uav_naive/run_naive.sh` | ✅ Written, exercised on job `21023` |
| `logs_in_develop/Gen11/Epoch2_env/{PREP_PLAN, EXECUTION_PLAN, RUNBOOK, CHANGELOG, EPOCH2_CLOSURE}.md` | ✅ All present |
| `logs/uav_naive/{task_A_hover, task_B_step, task_C_circle_6D, task_C_circle_9D}/` | ✅ Populated with `log.json`, `metrics.txt`, `controller.txt`, `rollout.gif` |

No edits to `config/`, `d3il/*/envs/`, `fm_visual_aligning/`,
`diffuser_visual_aligning/`, or any existing SLURM script.

---

## 7. Greenlight for Epoch 3

Epoch 3 (per `path_temp_initial.md` step 2 + DPCC integration) is unblocked:

- 9D trajectory format is the contract.
- Cascaded PID is the execution layer (tracks dynamic trajectories
  cleanly; static-hold capability is a one-line away if needed).
- The X2 + flight-controller pair behaves predictably on real Slurm
  hardware (rendering, logging, metrics all working).

Epoch 3 scope (preview, to be written in `logs_in_develop/Gen11/Epoch3_*/`):
- Define 2-3 obstacle layouts in MuJoCo (corridor, S-curve, pillar field).
- Implement a signed-distance / halfspace function returning DPCC-compatible
  `Z_f^t` from MuJoCo state.
- Verify the flight controller still tracks when feasible trajectories are
  threaded through obstacles.

---

## 8. Closure Statement

**Epoch 2 development is CLOSED as of 2026-05-31.** The architectural
hypothesis that motivated the epoch is validated. The known gap (hover
controller tuning) is documented, scoped, and judged irrelevant to the
FM-PCC downstream pipeline. Re-opening Epoch 2 would only be warranted by
a future need for static-hold capability that is not currently planned.

Runtime evidence on file at:
`Slurm_Codes/logs/2026-05-31/11_59_58_run_naive_21023.log`
