# D3IL Visual Aligning Baseline — Implementation Changelog

**Date**: 2026-05-28
**Branch**: `update_into_FM`

---

## Summary

Standalone pipeline to train and evaluate any pre-trained D3IL aligning agent
(BC, DDPM-EncDec-Vision, BESO…) with rich per-rollout recording — GIF/video,
JSON stats, summary PNG — matching the DPCC/FM-PCC eval output schema for direct
A/B comparison.  Zero permanent changes to `d3il/`; all outputs land under
`logs/d3il_visual_aligning_baseline/` (already gitignored by root `.gitignore`).

---

## New Files

### `d3il_visual_aligning_baseline_test/train_d3il_visual_aligning.py`

Thin Hydra training wrapper for the aligning task.

- `@hydra.main(config_path="../d3il/configs", config_name="aligning_config")` —
  resolves configs relative to its own location, works when called from repo root.
- Calls `agent.train_agent()` only — no `env_sim.test_agent()` at end (eval is
  handled separately by the rich eval script).
- W&B disabled by default; enabled when `use_wandb=True` is passed as Hydra override.
- `hydra.run.dir` is overridden from the SLURM CLI to
  `logs/d3il_visual_aligning_baseline/{agent_name}/seed_{s}/weights/`.
  Hydra changes CWD to that dir → `agent.working_dir = os.getcwd()` resolves there →
  `agent.store_model_weights()` writes checkpoints there, inside the gitignored tree.

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

### `logs_in_develop/D3IL_Visual_Aligning_RUN/PLAN.md`

Updated with:
- Output path findings: root `.gitignore` already covers `logs/*`; D3IL has no
  `.gitignore`; running from repo root resolves `logs/` to gitignored path.
- Train script design section (Hydra config_path approach, no `test_agent`).
- Eval script section updated: own rollout loop rationale, Hydra compose, `predict`
  4→3-tuple unwrap noted.

---

## Unchanged

- `d3il/` folder — zero edits beyond UF-16.4 hook already committed
- `diffuser_visual_aligning_test/`, `fm_visual_aligning_test/`, `config/` — untouched

---

## Usage Reference

```bash
# Train one agent/seed (from repo root or via SLURM):
python d3il_visual_aligning_baseline_test/train_d3il_visual_aligning.py \
    "agents=ddpm_encdec_vision_agent" "agent_name=ddpm_encdec_vision" \
    "seed=42" "hydra.run.dir=logs/d3il_visual_aligning_baseline/ddpm_encdec_vision/seed_42/weights"

# Eval one agent/seed:
python d3il_visual_aligning_baseline_test/eval_d3il_visual_aligning.py \
    --agent-name ddpm_encdec_vision --seed 42 --record all

# Full pipeline via SLURM:
sbatch Slurm_Codes/sbatch/d3il_visual_aligning_baseline/pipeline_d3il_baseline.sh \
    ddpm_encdec_vision 42 all
```
