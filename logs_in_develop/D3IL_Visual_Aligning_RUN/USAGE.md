# D3IL Visual Aligning Baseline — Usage & Outputs

**Scripts**: `d3il_visual_aligning_baseline_test/`
**SLURM jobs**: `Slurm_Codes/sbatch/d3il_visual_aligning_baseline/`

---

## Prerequisites

- A trained D3IL aligning checkpoint in:
  `logs/d3il_visual_aligning_baseline/{agent_name}/seed_{s}/weights/`
  containing `eval_best_{name}.pth` (written by the train step).
- Conda env `FMPCC` active; run everything from repo root (`cd $REPO`).

---

## Step 1 — Train

### Via SLURM (recommended)

Epoch defaults follow the D3IL paper: **200 for vision agents, 500 for state agents**,
with `eval_every_n_epochs = epoch / 10` (paper: "evaluate after every 1/10th of training").

```bash
# Full pipeline: train (paper epochs) → eval chained automatically
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/d3il_visual_aligning_baseline/pipeline_d3il_baseline.sh \
    ddpm_encdec_vision  42          # epoch=200 by default (vision), record=all
#   ^agent_name         ^seed

# With explicit epoch override (e.g. reproduce exactly or custom length)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/d3il_visual_aligning_baseline/pipeline_d3il_baseline.sh \
    ddpm_encdec_vision  42  200  all
#   ^agent_name         ^seed  ^epoch  ^record_mode
```

```bash
# Train only (paper defaults)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/d3il_visual_aligning_baseline/train_d3il_baseline.sh \
    ddpm_encdec_vision  42        # → 200 epochs, eval every 20

# Train with explicit epoch
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/d3il_visual_aligning_baseline/train_d3il_baseline.sh \
    ddpm_encdec_vision  42  200
```

### Locally (smoke check)

```bash
# Quick smoke: 5 epochs to verify pipeline end-to-end
python d3il_visual_aligning_baseline_test/train_d3il_visual_aligning.py \
    --config-name aligning_vision_config \
    "agents=ddpm_encdec_vision_agent" \
    "agent_name=ddpm_encdec_vision" \
    "seed=42" \
    "epoch=5" \
    "eval_every_n_epochs=1" \
    "hydra.run.dir=logs/d3il_visual_aligning_baseline/ddpm_encdec_vision/seed_42/weights"
```

See **[Choosing a Model](#choosing-a-model)** section below for all options.

---

## Step 2 — Eval

### Via SLURM

```bash
# Eval one seed
sbatch Slurm_Codes/sbatch/d3il_visual_aligning_baseline/eval_d3il_baseline.sh \
    ddpm_encdec_vision  42  all
#   ^agent_name         ^seed  ^record_mode (all|gif|video|none)

# Eval all seeds in d3il_eval_config.yaml (leave seed blank)
sbatch Slurm_Codes/sbatch/d3il_visual_aligning_baseline/eval_d3il_baseline.sh \
    ddpm_encdec_vision
```

### Locally

```bash
# Smoke: 3 contexts, record GIF only
python d3il_visual_aligning_baseline_test/eval_d3il_visual_aligning.py \
    --agent-name ddpm_encdec_vision \
    --seed 42 \
    --n-contexts 3 \
    --record gif

# All CLI options
python d3il_visual_aligning_baseline_test/eval_d3il_visual_aligning.py --help
```

### Edit default settings

Open `d3il_visual_aligning_baseline_test/d3il_eval_config.yaml` to change:
- `seeds`, `n_contexts`, `n_trajectories_per_context`
- `if_vision: true/false` — false skips image capture (faster, no video)
- `record_mode: all | gif | video | none`

---

## Outputs

```
logs/d3il_visual_aligning_baseline/
  {agent_name}/
    seed_{s}/
      weights/                          ← from train step
        eval_best_{name}.pth            ← best val-loss checkpoint
        last_{name}.pth                 ← final epoch checkpoint
      diagnostics/                      ← from eval step
        rollout_0.gif                   ← bp_cam video (fps=10)
        rollout_0.mp4                   ← bp_cam video (fps=20)
        rollout_0_stats.json            ← per-rollout metrics (see below)
        rollout_0_summary.png           ← 2-panel plot (see below)
        rollout_0_data.pkl              ← full data dict
        rollout_1.gif / …
      results_seed_{s}.json             ← seed-level aggregate
    aggregate_results.json              ← cross-seed aggregate (multi-seed runs)
```

### `rollout_N_stats.json` fields

```json
{
  "rollout_index": 0,
  "success":       false,
  "steps":         400,
  "mean_distance": 0.0812,
  "mode":          0,
  "context_info": {
    "context_idx":        0,
    "box_init_xy":        [0.581, -0.204],
    "box_init_angle_deg": -43.5,
    "target_xy":          [0.498,  0.333],
    "target_angle_deg":   -58.3,
    "init_xy_dist":       0.5374,
    "final_box_xy":       [0.501,  0.330],
    "final_box_angle_deg":-57.1,
    "final_xy_dist":      0.0043
  }
}
```

`mean_distance` = `0.5*(3D_pos_m + rot/π)` — the D3IL combined metric; success if < 0.033.
`final_xy_dist` = 2D XY only, no rotation.
`mode` = 0 means robot still in contact with box at last step (normal for push task).

### `rollout_N_summary.png`

Two panels:
- **Left**: XY top-down trajectory of the robot EE (black line); box init position (red ×); target position (green ★); final box position (red triangle).
- **Right**: `mean_distance` over episode steps; green dashed line = success threshold (0.033).

### `results_seed_{s}.json` fields

| Field | Meaning |
|---|---|
| `success_rate` | Fraction of rollouts that succeeded |
| `mean_distance_mean/std` | Combined D3IL metric across rollouts |
| `final_xy_dist_mean/std` | 2D XY distance box→target at episode end |
| `final_angle_deg_mean/std` | Box final angle (degrees) |
| `n_steps_mean/std` | Episode length (max 400) |
| `mode_0_rate` | Fraction of rollouts ending with robot in contact (mode=0) |

### `aggregate_results.json`

Same fields as `results_seed`, but mean ± std computed **across seeds**.
Written only when multiple seeds are run.

---

## Choosing a Model

The D3IL benchmark (ICLR 2024) evaluates 11 imitation learning methods on the aligning task.
All have a **vision variant** (`*_vision`) and a **state-only variant**.

### Visual agents (`if_vision: true` in config)

These are the primary baselines — they consume camera images the same way our FM-PCC/DPCC
policies do, so results are directly comparable.

| `agent_name` to pass | Config file | Checkpoint saved as | Architecture family | Notes |
|---|---|---|---|---|
| `ddpm_encdec_vision` | `ddpm_encdec_vision_agent.yaml` | `eval_best_ddpm.pth` | Diffusion (EncDec Transformer) | **Default in `aligning_vision_config.yaml` — start here** |
| `beso_vision` | `beso_vision_agent.yaml` | `eval_best_beso.pth` | Diffusion (score-based, BESO) | Strong multi-modal baseline |
| `act_vision` | `act_vision_agent.yaml` | `eval_best_act.pth` | Transformer (ACT / VAE) | From ACT paper; uses action chunking |
| `bet_vision` | `bet_vision_agent.yaml` | `eval_best_bet.pth` | Transformer (BeT + MinGPT) | Behavior Transformer with latent actions |
| `bet_mlp_vision` | `bet_mlp_vision_agent.yaml` | `eval_best_bet_mlp.pth` | MLP + BeT offset head | Lighter BeT variant |
| `gpt_vision` | `gpt_vision_agent.yaml` | `eval_best_gpt.pth` | Transformer (GPT-BC) | Autoregressive BC |
| `cvae_vision` | `cvae_vision_agent.yaml` | `eval_best_cvae.pth` | VAE / latent-variable | CVAE policy |
| `ddpm_vision` | `ddpm_vision_agent.yaml` | `eval_best_ddpm.pth` | Diffusion (goal-conditioned MLP) | Simpler DDPM variant |
| `ddpm_transformer_vision` | `ddpm_transformer_vision_agent.yaml` | `eval_best_ddpm.pth` | Diffusion (Transformer backbone) | Heavier DDPM |
| `ibc_vision` | `ibc_vision_agent.yaml` | `eval_best_ibc.pth` | Energy-based (IBC) | Slow inference; EBM |
| `bc_vision` | `bc_vision_agent.yaml` | `eval_best_bc.pth` | MLP (deterministic BC) | Weakest baseline; fast to train |

### State-only agents (`if_vision: false` in config)

Useful for **ablation**: train without images to isolate how much visual input helps.
Use the same pipeline — just set `if_vision: false` in `d3il_eval_config.yaml` and
drop the `_vision` suffix from `agent_name`.

| `agent_name` | Config file | Notes |
|---|---|---|
| `ddpm_encdec` | `ddpm_encdec_agent.yaml` | State counterpart of primary visual baseline |
| `beso` | `beso_agent.yaml` | |
| `act` | `act_agent.yaml` | |
| `bet` | `bet_agent.yaml` | |
| `bet_mlp` | `bet_mlp_agent.yaml` | |
| `bc` | `bc_agent.yaml` | Fastest to train |
| `bc_gmm` | `bc_gmm_agent.yaml` | BC with Gaussian mixture output |

### How to select

1. **For comparison with FM-PCC/DPCC**: use `ddpm_encdec_vision` — it is the D3IL paper's
   default for visual aligning and the closest architectural match (diffusion policy).
2. **For full D3IL paper reproduction**: run all visual agents; compare `success_rate`
   across models.
3. **For a fast smoke test**: `bc_vision` trains in minutes and confirms the pipeline works
   end-to-end before committing GPU hours to diffusion models.

### Do different models clash in outputs?

**No.** `agent_name` is baked into every output path:

```
logs/d3il_visual_aligning_baseline/
  ddpm_encdec_vision / seed_42 / weights / eval_best_ddpm.pth
  beso_vision        / seed_42 / weights / eval_best_beso.pth
  act_vision         / seed_42 / weights / eval_best_act.pth
  bc_vision          / seed_42 / weights / eval_best_bc.pth
```

You can train and eval all agents in parallel SLURM jobs without any interference.
The SLURM train script takes `agent_name` as `$1`, so fan-out is straightforward:

```bash
for AGENT in ddpm_encdec_vision beso_vision act_vision bc_vision; do
    sbatch Slurm_Codes/sbatch/d3il_visual_aligning_baseline/pipeline_d3il_baseline.sh \
        "$AGENT" 42 all
done
```

---

## Comparing to DPCC / FM-PCC

All three evals share the same output schema subset, so you can load and compare them directly:

| Field | D3IL Baseline | DPCC / FM-PCC |
|---|---|---|
| `success_rate` | ✅ | ✅ |
| `mean_distance` | ✅ | ✅ |
| `final_box_xy / angle` | ✅ | ✅ |
| `final_xy_dist` | ✅ | ✅ |
| GIF / MP4 | ✅ (bp_cam only) | ✅ |
| Constraint metrics | — (no projector) | ✅ |
| MPC foresight plot | — | ✅ |

The D3IL eval runs a **single-process** loop (no `aligning_sim.test_agent()` multiprocessing) so it is slower per-rollout than the original D3IL eval, but produces full recording artifacts.
