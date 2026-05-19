# Gen6V4 Code Structure — Audit Guide

**Date:** 2026-05-19
**Purpose:** Orientation map for an auditor reviewing the Gen6V4 Visual-DPCC codebase
after Fix 7. This document describes *what is where and why* — not findings.

---

## What Gen6V4 Is

Gen6V4 is the Visual-DPCC model for the D3IL Aligning task. It extends the original
DPCC (Diffusion Policy with Constraints and Control) architecture to condition on
camera observations (RGB frames) in addition to proprioceptive state.

Core invariants that must not change:
- **9D trajectory:** `[act(3) | des_c_pos(3) | c_pos(3)]`
- **Euler integration:** `c_pos[t+1] = c_pos[t] + act[t] * dt`
- **SLSQP projector:** projects denoised trajectory onto feasible constraint manifold at each diffusion step

---

## Repository Layout

```
FM-PCC/
├── diffuser_visual_aligning/          # Core model library (copy-modify; never edit originals in diffuser/)
│   ├── models/
│   │   ├── visual_gaussian_diffusion.py   # VisualGaussianDiffusion — main diffusion model
│   │   ├── visual_unet.py                 # VisualUNet — 1D temporal U-Net + image encoder backbone
│   │   ├── unet1d_temporal_cond.py        # Temporal conditioning U-Net (shared with non-visual path)
│   │   └── diffusion.py                   # Base GaussianDiffusion (parent class)
│   ├── datasets/
│   │   ├── sequence.py                    # ParityAligningDataset — loads 9D windows + image pairs
│   │   └── normalization.py               # LimitsNormalizer (clips to [min,max]; critical for K=100)
│   └── sampling/
│       └── projection.py                  # SLSQP projector — DPCC constraint engine (DO NOT TOUCH)
│
├── diffuser_visual_aligning_test/     # Train + eval entry points
│   ├── train_visual_aligning_dpcc.py      # Training script
│   └── eval_visual_aligning_dpcc.py       # Eval script (VisualAgentWrapper + rollout loop)
│
├── config/
│   ├── aligning-d3il-visual.py            # Hydra config: model architecture, horizon H, steps K
│   └── visual_aligning_eval.yaml          # Eval config: diffusion_timestep_threshold, seeds, etc.
│
└── d3il/                              # Vendored D3IL (patched — see Deltas section below)
    ├── simulation/
    │   └── aligning_sim.py                # Aligning_Sim — multiprocess rollout harness (key file)
    ├── environments/
    │   ├── dataset/
    │   │   └── aligning_dataset.py        # Raw image loading (BGR, no cv2.cvtColor — FIX_7.2)
    │   └── d3il/
    │       ├── envs/gym_aligning_env/gym_aligning/envs/
    │       │   └── aligning.py            # Robot_Push_Env + BPCageCam (FIX_7.3)
    │       ├── models/mj/robot/
    │       │   └── panda_rod_invisible.xml # Rod tip collision flags (FIX_7.3)
    │       ├── d3il_sim/sims/mj_beta/
    │       │   ├── MjLoadable.py          # atexit cleanup handler (FM-PCC patch, active)
    │       │   └── MjRobot.py             # fsync guarantee on robot XML write (FM-PCC patch, active)
    │       └── d3il_sim/utils/
    │           └── sim_path.py            # D3IL_DIR env var override (FM-PCC patch, active)
    └── agents/models/bet/libraries/mingpt/
        └── trainer.py                     # Progress bar throttle — prevents Slurm stdout overflow
```

---

## Data Flow (Eval)

```
eval_visual_aligning_dpcc.py
  │
  ├─ loads checkpoint → VisualGaussianDiffusion (wraps VisualUNet)
  ├─ loads obs_normalizer / act_normalizer (LimitsNormalizer)
  ├─ builds VisualAgentWrapper (holds video_frames, rollout history)
  │
  └─ Aligning_Sim.test_agent(agent)
       │
       └─ eval_agent() [single process, n_cores=1]
            │
            ├─ Robot_Push_Env (MuJoCo, if_vision=True)
            ├─ for each context/rollout:
            │    agent.reset()
            │    env.reset(context=test_contexts[ctx])    ← or train_contexts if eval_on_train
            │    while not done:
            │        agent.predict(bp_image, inhand_image, des_robot_pos)
            │          └─ VisualUNet → encode images → 1D U-Net → SLSQP project → return act[0]
            │        env.step(action)
            │        collect bp_image frame into video_frames
            │    agent.update_rollout_info(info)           ← triggers video/gif save
            │
            └─ return success_rate, mode_encoding, successes, mean_distance
```

---

## Key File: `aligning_sim.py` — Patched Deltas vs Original D3IL

This is the most-patched file. Every delta is intentional and gated:

| Delta | Guard | Purpose |
|---|---|---|
| `eval_on_train: bool = False` param | Default = test contexts = D3IL behavior | Allows in-distribution sanity checks |
| CPU pinning gated on `not self.if_vision` | State-only path is byte-for-byte D3IL | Unpinned for visual: prevents single-core starvation under K=100 |
| `hasattr(agent, 'update_rollout_info')` hook | Guard skips for D3IL agents | Triggers video/gif save after each rollout |
| Returns 4 values `(success_rate, mode_encoding, successes, mean_distance)` | FM-PCC eval script expects 4 | Exposes per-context success and distance tensors |

**Rule:** If D3IL parity reverts are ever re-run on this file, the CPU bypass gate and
`update_rollout_info` hook must be explicitly preserved — they cover behavior D3IL never defined.

---

## Key File: `eval_visual_aligning_dpcc.py` — Structure

- **`VisualAgentWrapper`** (line ~214): wraps the diffusion model. Owns `video_frames`,
  `master_rollout_history`, normalizers, and the SLSQP projector reference.
  - `predict()`: runs one inference step; collects one frame (BGR→RGB for display)
  - `update_rollout_info()`: called by `aligning_sim` at rollout end → triggers `_save_diagnostics()`
  - `reset()`: clears state at rollout start (called by `aligning_sim` before each rollout)
- **`generate_expert_reference()`** (line ~144): saves 3 ground-truth expert rollout videos for comparison
- **Main eval loop** (line ~700+): iterates seeds → loads checkpoint → runs `sim.test_agent(agent)`

---

## Non-negotiable Invariants (for the auditor)

1. **Trajectory format:** 9D = `[act(3) | des_c_pos(3) | c_pos(3)]`. Any code that reshapes,
   slices, or normalizes trajectories must respect this split.
2. **`clip_denoised=False`:** Must be forced at eval. Any checkpoint trained with K≥100 will
   diverge if the ±5 clamp fires. Verified forced in eval script.
3. **BGR image pipeline:** `cv2.imread` returns BGR. No `cv2.cvtColor` conversion anywhere in
   the training or eval image loading path (FIX_7.2). The display/video path does convert BGR→RGB
   for human viewing only — this does not affect model inputs.
4. **SLSQP projector:** `diffuser_visual_aligning/sampling/projection.py`. Must not be modified.
   Threshold controlled by `diffusion_timestep_threshold` in `visual_aligning_eval.yaml`.
5. **`eval_on_train=False` default:** Eval runs on held-out test contexts by default. Pass
   `--eval_on_train` only for in-distribution sanity checks during development.

---

## Fix 7 Change Log (summary for auditor)

Full detail in `../Manual_Legacy_retrieval_FIX_7/`. Brief map:

| Fix | File(s) | What |
|---|---|---|
| FIX_7.1 | `aligning_sim.py`, `eval_ddpm_encdec_vision.py` | Removed `max_episode_length` plumbing that crashed env construction |
| FIX_7.2 | `d3il/environments/dataset/aligning_dataset.py` | Removed spurious BGR→RGB conversion (channel distribution shift) |
| FIX_7.3 | `aligning_sim.py`, `aligning.py`, `panda_rod_invisible.xml` | Restored D3IL parity: BPCageCam constructor, rod:tip non-colliding, CPU pinning, return signature |
| FIX_7.4 | `aligning_sim.py`, `eval_visual_aligning_dpcc.py` | Restored `eval_on_train` as optional flag (default False); removed `max_episode_length` from visual eval call |
| FIX_7.5 | `aligning_sim.py` | CPU bypass for vision mode; `update_rollout_info` hook for video save; return expanded to 4 values |
