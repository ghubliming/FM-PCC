# D3IL Visual Aligning Baseline — Implementation Changelog

**Date**: 2026-05-28
**Branch**: `update_into_FM`

---

## Summary

Standalone pipeline to train and evaluate any pre-trained D3IL aligning agent
(BC, DDPM-EncDec-Vision, BESO…) with rich per-rollout recording — GIF/video,
JSON stats, summary PNG — matching the DPCC/FM-PCC eval output schema for direct
A/B comparison.  Zero permanent changes to `d3il/` (aside from one genuine bug fix
noted below); all outputs land under `logs/d3il_visual_aligning_baseline/`
(already gitignored by root `.gitignore`).

---

## New Files

### `d3il_visual_aligning_baseline_test/train_d3il_visual_aligning.py`

Training wrapper for the aligning task, mirroring `d3il/run_vision.py`.

- `@hydra.main(config_path="../d3il/configs", config_name="aligning_vision_config")` —
  resolves configs relative to its own location, works when called from repo root.
  No `version_base` parameter (cluster Hydra is pre-1.2).
- W&B on by default; `???` placeholders in `wandb.entity/project` are handled safely
  (`throw_on_missing=False` + guard), defaulting project to `'d3il-baseline'`.
- Dispatches on `agent.model.visual_input`:
  - **Vision agents** → `_train_vision(agent)` — outer epoch loop calling
    `train_vision_agent()` once per epoch, with val-loss checkpointing via
    `_eval_vision_loss()`.  Mirrors `d3il/run_vision.py`; avoids MuJoCo during train.
  - **State agents** → `agent.train_agent()` — has its own complete epoch loop.
- `_eval_vision_loss()` — computes val loss by calling `agent.model(state, None,
  action=action, if_train=True)` directly (vision tuple); bypasses state-only
  `agent.evaluate()` which calls `scaler.scale_input(state)` on a plain tensor.
- `hydra.run.dir` is overridden from the SLURM CLI to
  `logs/d3il_visual_aligning_baseline/{agent_name}/seed_{s}/weights/`.
  Hydra changes CWD → `agent.working_dir = os.getcwd()` resolves there →
  checkpoints land in the gitignored tree.

### `d3il_visual_aligning_baseline_test/eval_d3il_visual_aligning.py`

Main evaluation script (~380 lines).

**Key design choices vs D3IL's `run_sim.py`:**

| `run_sim.py` / `aligning_sim` | This script |
|---|---|
| Multiprocessing (`n_cores`) | Single-process (enables per-rollout recording) |
| W&B only output | GIF/MP4 + JSON stats + summary PNG per rollout |
| No final box position | UF-16.4 hook — reads `env.push_box` pos+quat |
| No summary PNG | 2-panel: XY trajectory + distance-to-target curve |

**`D3ILBaselineWrapper` class:**
- `reset()` — clears per-rollout buffers and calls agent reset
- `record_context_info(context, context_idx)` — stores box init XY/angle, target
  XY/angle, init XY dist from `(pos, quat, target_pos, target_quat)` context tuple
- `record_step(des_robot_pos, frame_bgr, mean_distance)` — called from rollout loop
  after each `env.step()`; accumulates EE trajectory, mean_distance curve, video frames
  (BGR→RGB for imageio)
- `update_rollout_info(info)` — UF-16.4: extracts `final_box_pos`/`final_box_quat`,
  computes final angle (exact Z-quat formula), final XY dist to target; prints console
  summary; calls `_export_rollout_realtime`
- `predict(obs, if_vision)` — aligning_sim-compatible: unwraps 4-tuple
  `(bp, inh, des_pos, robot_pos)` to the 3-tuple `(bp, inh, des_pos)` that D3IL agents
  expect (4th element is actual robot pos, only needed by sim-side logic)

**`build_agent()`:**
Uses `hydra.initialize_config_dir + compose` (programmatic Hydra — no CWD change,
no Hydra output dirs).  `OmegaConf.register_new_resolver("add", ...)` called before
compose to resolve `${add:...}` interpolations in DDPM agent config.  Checkpoint
loaded via `agent.load_pretrained_model(ckpt_dir, sv_name=eval_model_name)`.
`agent_cfg_group` is always derived from `agent_name` at runtime (not from YAML),
so `--agent-name ddpm_vision` correctly selects `ddpm_vision_agent` config.

**Rollout loop:**
Follows the `if_vision / else` pattern from `aligning_sim.eval_agent` exactly:
visual path unpacks `(env_state, bp_image, inhand_image)` from `env.reset()` and
`(robot_pos, bp_image, inhand_image)` from `env.step()`; state path uses full obs vector.

**Output artifacts per rollout** (in `{save_root}/{agent_name}/seed_{s}/diagnostics/`):

| File | Content |
|---|---|
| `rollout_N.gif` | bp_cam frames at fps=10 |
| `rollout_N.mp4` | bp_cam frames at fps=20 |
| `rollout_N_stats.json` | success, steps, mean_distance, mode, context_info (incl. final box XY/angle) |
| `rollout_N_summary.png` | 2-panel: XY trajectory + distance curve |
| `rollout_N_data.pkl` | full rollout dict (ee_traj, dist_curve, context_info) |

**Output per seed** (`{save_root}/{agent_name}/seed_{s}/results_seed_{s}.json`):
success_rate, mean_distance mean/std, final_xy_dist mean/std, final_angle mean/std,
n_steps mean/std, mode_0_rate.

**Cross-seed aggregate** (`{save_root}/{agent_name}/aggregate_results.json`):
written when multiple seeds are run; mean ± std of all scalar metrics across seeds.

### `d3il_visual_aligning_baseline_test/d3il_eval_config.yaml`

Plain YAML (no Hydra) read by `eval_d3il_visual_aligning.py` via `yaml.safe_load`.
Configures agent name, Hydra config group, checkpoint root, seeds, n_contexts,
if_vision, record_mode.  All keys overridable by CLI args.

### `Slurm_Codes/sbatch/d3il_visual_aligning_baseline/train_d3il_baseline.sh`

SLURM train job. Mirrors `train_visual_aligning_dpcc.sh` header (GPU, mem, time).
Args: `$1=agent_name` (default `ddpm_encdec_vision`), `$2=seed` (default 42).
Auto-selects config: agents with `_vision` suffix → `aligning_vision_config`;
others → `aligning_config`.
Calls `train_d3il_visual_aligning.py` with `hydra.run.dir` override to
`logs/d3il_visual_aligning_baseline/{agent_name}/seed_{s}/weights`.

### `Slurm_Codes/sbatch/d3il_visual_aligning_baseline/eval_d3il_baseline.sh`

SLURM eval job. Mirrors `eval_visual_aligning_dpcc.sh` header.
Args: `$1=agent_name`, `$2=seed` (optional), `$3=record_mode` (default `all`).
Blank seed → eval runs all seeds listed in `d3il_eval_config.yaml`.

### `Slurm_Codes/sbatch/d3il_visual_aligning_baseline/pipeline_d3il_baseline.sh`

Pipeline manager — submits train then eval with `--dependency=afterok:$TRAIN_ID`.
Mirrors `visual_aligning_pipeline_dpcc.sh` exactly.
Args: `$1=agent_name`, `$2=seed`, `$3=record_mode`.

---

## Updated Files

### `Slurm_Codes/submit.sh`

**Bug fix**: was silently dropping all arguments after `$1` (script path).
Added `shift` after capturing `$1` and stored remaining args in `SCRIPT_ARGS=("$@")`;
appended `"${SCRIPT_ARGS[@]}"` to the `sbatch` call.
Without this fix, `submit.sh train_d3il_baseline.sh ddpm_vision 6` would submit with
default `agent_name=ddpm_encdec_vision seed=42` regardless of what was passed.

### `d3il/agents/models/diffusion/diffusion_models.py`

**Bug fix in `DiffusionMLPNetwork.forward()` 3D path** (vision + sequence input):
The code rearranged the timestep embedding `t` to `[B, 1, t_dim]` then tried
`torch.cat([x, t, state], dim=2)` where `x = [B, T, action_dim]` and `state = [B, T, obs_dim]`.
The concat fails because dim-1 size is `T` for `x`/`state` but `1` for `t`.
Fix: `.expand(-1, x.shape[1], -1)` to broadcast `t` over the sequence length.

This is a genuine omission in D3IL's code — the 3D branch was written but the
expand was never added. The 2D (state-only) path is unaffected.
Only `ddpm_vision` hits this path; `ddpm_encdec_vision` uses a Transformer backbone
with a different forward pass.

### `logs_in_develop/D3IL_Visual_Aligning_RUN/PLAN.md`

Updated with:
- Output path findings: root `.gitignore` already covers `logs/*`; D3IL has no
  `.gitignore`; running from repo root resolves `logs/` to gitignored path.
- Train script design section (Hydra config_path approach, no `test_agent`).
- Eval script section updated: own rollout loop rationale, Hydra compose, `predict`
  4→3-tuple unwrap noted.

---

## Debugging Iteration — Runtime Fixes on Cluster

The following bugs were discovered by running on the remote SLURM cluster and fixed
iteratively (each triggered a new job submission after git-pull).

| # | Error | Root Cause | Fix |
|---|---|---|---|
| 1 | `TypeError: main() got unexpected kwarg 'version_base'` | Cluster Hydra is pre-1.2; `version_base` didn't exist | Removed from `@hydra.main()` and `initialize_config_dir()` |
| 2 | `agent=ddpm_encdec_vision seed=42` despite passing `ddpm_vision 6` | `submit.sh` dropped all args after script path | `shift` + `SCRIPT_ARGS` fix in `submit.sh` |
| 3 | `ValueError: too many values to unpack` in `train_agent` (expects 3-tuple, got 5) | `aligning_config` (state dataset) was used instead of `aligning_vision_config` | SLURM script auto-selects config by `*_vision*` in agent name; Python wrapper also updated |
| 4 | `MissingMandatoryValue: entity` (W&B crash) | `aligning_vision_config.yaml` has `wandb: {entity: ???}` as mandatory missing; `throw_on_missing=True` crashes | `throw_on_missing=False` + guard against `None`/`'???'` |
| 5 | `ValueError: too many values to unpack (expected 3)` in `train_agent` line 217 | `train_agent()` is state-only (expects 3-tuple); vision dataset returns 5-tuple | Detect `agent.model.visual_input`; dispatch to `train_vision_agent()` |
| 6 | `RuntimeError: Sizes of tensors must match…Expected 8 got 1` in `DiffusionMLPNetwork.forward` | `t=[B,1,4]` can't cat with `x=[B,8,3]` on dim=2 — expand missing | Add `.expand(-1, x.shape[1], -1)` in 3D branch of `diffusion_models.py` |
| 7 | Only 1 epoch trained (config says `epoch=4`) | `train_vision_agent()` is one epoch only; `run_vision.py` wraps it in an outer loop; we called it once | Added `_train_vision()` outer epoch loop matching `run_vision.py` |
| 8 | `AttributeError: 'tuple' has no attribute 'shape'` in `agent.evaluate()` | `evaluate()` is state-only; we passed a vision tuple `(bp_imgs, inhand_imgs, obs)` | Replaced with `_eval_vision_loss()` that calls `agent.model()` directly |

---

## Unchanged

- `diffuser_visual_aligning_test/`, `fm_visual_aligning_test/`, `config/` — untouched

---

## Usage Reference

```bash
# Train one agent/seed (from repo root or via SLURM):
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/d3il_visual_aligning_baseline/train_d3il_baseline.sh \
    ddpm_vision 6

# Eval one agent/seed:
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/d3il_visual_aligning_baseline/eval_d3il_baseline.sh \
    ddpm_vision 6 all

# Full pipeline (train → eval chained):
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/d3il_visual_aligning_baseline/pipeline_d3il_baseline.sh \
    ddpm_vision 6 all

# Local smoke test (5 epochs):
python d3il_visual_aligning_baseline_test/train_d3il_visual_aligning.py \
    --config-name aligning_vision_config \
    "agents=ddpm_vision_agent" "agent_name=ddpm_vision" \
    "seed=6" "epoch=5" \
    "hydra.run.dir=logs/d3il_visual_aligning_baseline/ddpm_vision/seed_6/weights"
```
