# D3IL Visual Aligning Baseline — Run Plan

**Date**: 2026-05-28
**Branch**: `update_into_FM`
**Purpose**: Run the original D3IL Visual Aligning pipeline (any agent: BC, DDPM, BESO…)
             with rich output logging (GIF/video, JSON diagnostics, success/distance metrics)
             comparable to our DPCC and FM-PCC eval outputs, so we can A/B against them.

---

## Goals

1. Run D3IL aligning agents (pre-trained) in eval mode with zero permanent changes to `d3il/`.
2. Capture per-rollout GIF/video + JSON diagnostics + summary PNG.
3. Produce a cross-rollout aggregate JSON matching the DPCC/FM output schema.
4. Provide a SLURM job script alongside existing `diffuser_visual_aligning/` scripts.
5. Keep the new code fully standalone — a reader of `d3il/` should not even notice this exists.

---

## Output Path — No D3IL Pollution

**Root `.gitignore` already covers `logs/*`.**
D3IL has no `.gitignore` of its own, so anything written inside `d3il/` goes into git.

Hydra resolves `log_dir: logs/aligning/` relative to the **caller's CWD**:

| Launched from | Logs land at | Gitignored? |
|---|---|---|
| `cd d3il && python run.py` | `d3il/logs/aligning/...` | ❌ infects repo |
| `cd $REPO && python d3il_visual_aligning_baseline_test/train_d3il_visual_aligning.py` | `logs/d3il_visual_aligning_baseline/...` (repo root) | ✅ covered |

All SLURM scripts use `cd "$REPO"` before invoking Python — so as long as we
override `hydra.run.dir` to a path under `logs/`, **nothing ever writes inside `d3il/`**.
No `d3il/.gitignore` addition needed.

**Predictable checkpoint path via Hydra CLI override:**
```
hydra.run.dir=logs/d3il_visual_aligning_baseline/{agent_name}/seed_{s}/weights
```
The eval script reads from `{root}/{agent_name}/seed_{s}/weights/eval_best_{name}.pth`.

---

## Constraint: Minimal D3IL changes

The only existing D3IL change we depend on is the UF-16.4 hook already in
`d3il/simulation/aligning_sim.py`:

```python
if hasattr(agent, 'update_rollout_info'):
    agent.update_rollout_info({**info, 'context': context,
                               'final_box_pos':  _fbox_pos,
                               'final_box_quat': _fbox_quat})
```

This hook is already committed. No further D3IL edits are needed.

---

## New Files Overview

```
d3il_visual_aligning_baseline_test/          ← new standalone folder
    train_d3il_visual_aligning.py            ← Hydra train wrapper (~40 lines)
    eval_d3il_visual_aligning.py             ← main eval script (~380 lines)
    d3il_eval_config.yaml                    ← seeds, agent type, rollout config

Slurm_Codes/sbatch/d3il_visual_aligning_baseline/
    train_d3il_baseline.sh                   ← single agent/seed train job
    eval_d3il_baseline.sh                    ← single agent/seed eval job
    pipeline_d3il_baseline.sh               ← train → eval pipeline chain
```

No changes to `d3il/`, `diffuser_visual_aligning_test/`, `fm_visual_aligning_test/`, or `config/`.

---

## Train Script Design (`train_d3il_visual_aligning.py`)

Thin Hydra wrapper — mirrors `d3il/run.py` but:
- `config_path="../d3il/configs"` (relative to file) → always resolves to `d3il/configs/`
- Trains only — no `env_sim.test_agent()` at end (we eval separately with rich recording)
- `wandb mode="disabled"` by default; W&B enabled via `--wandb` flag if needed
- `hydra.run.dir` overridden from SLURM CLI → `logs/d3il_visual_aligning_baseline/{agent}/seed_{s}/weights/`
- Weights saved there by `agent.store_model_weights(agent.working_dir, ...)` (Hydra sets CWD)

SLURM call pattern:
```bash
python d3il_visual_aligning_baseline_test/train_d3il_visual_aligning.py \
    "agents=ddpm_encdec_vision_agent" \
    "agent_name=ddpm_encdec_vision" \
    "seed=42" \
    "hydra.run.dir=logs/d3il_visual_aligning_baseline/ddpm_encdec_vision/seed_42/weights"
```

---

## Eval Script Design (`eval_d3il_visual_aligning.py`)

### No `aligning_sim.test_agent()` — own rollout loop

`aligning_sim.test_agent()` uses multiprocessing which complicates per-rollout recording.
Instead we write our **own single-process rollout loop** (pattern copied from
`aligning_sim.eval_agent`), calling `Robot_Push_Env` directly. The UF-16.4 hook
(`update_rollout_info`) is called manually at the end of each episode from our loop.

### Config loading

- Reads `d3il_eval_config.yaml` (plain YAML, no Hydra) for:
  - `agent_cfg`: path to D3IL hydra agent config (e.g. `d3il/configs/agents/ddpm_encdec_vision_agent.yaml`)
  - `checkpoint_path`: path to saved `.pt` / `.pkl` model weights
  - `seeds: [42, 43, ...]`
  - `n_contexts`, `n_trajectories_per_context`
  - `eval_on_train: false`
  - `record_mode`: `all` / `none` / `failures`
  - `save_root`: output root dir

- Agent instantiation via `hydra.initialize_config_dir + compose` (programmatic, no CWD
  change, no Hydra output directories created). `OmegaConf.register_new_resolver("add",...)`
  registered before compose so `${add:...}` interpolations in DDPM config resolve.

### `D3ILBaselineWrapper`

Wraps any D3IL agent. Presents the same interface as `VisualAgentWrapper` so
`aligning_sim.py`'s `test_agent()` call works unchanged.

Key responsibilities:
- `predict(obs)` → delegates to wrapped D3IL agent's `predict()` or `get_action()`
- `reset()` → clears rollout buffers, calls wrapped agent's reset
- `update_rollout_info(info)` → receives the UF-16.4 hook data:
  - extracts `success`, `mean_distance`, `mode`, `final_box_pos`, `final_box_quat`
  - computes `final_box_angle_deg`, `final_xy_dist` (same formula as UF-16.4)
  - builds `curr_context_info`, appends to `history_context_info`
  - calls `_export_rollout_realtime(ridx)`
- `record_step(des_robot_pos, frame_bgr, mean_distance)` → accumulates EE traj, frames,
  distance curve (called from our own loop after each `env.step()`)
- `record_context_info(context, context_idx)` → stores init XY, target XY, init angle
- `predict(obs, if_vision)` → aligning_sim-compatible 4-tuple handler; unwraps to 3-tuple
  for the D3IL agent (which only uses `(bp, inhand, des_robot_pos)`)

No projection, no MPC foresight, no constraint metrics — those don't exist in D3IL.

### `_export_rollout_realtime(rollout_idx)`

Writes to `{save_root}/seed_{s}/diagnostics/`:

| Artifact | Content |
|---|---|
| `rollout_{N}.gif` / `.mp4` | rendered frames at fps=10/20 |
| `rollout_{N}_stats.json` | success, mean_distance, mode, final_box_xy, final_box_angle_deg, final_xy_dist, init_box_xy, target_xy, n_steps |
| `rollout_{N}_summary.png` | 2-panel: XY trajectory trace + success/distance annotation |

### Summary block (after all rollouts)

Console + JSON `results_seed_{s}.json` at `{save_root}/seed_{s}/`:

| Field | Source |
|---|---|
| `success_rate` | fraction of rollouts where `success=True` |
| `mean_distance_mean/std` | across rollouts |
| `final_xy_dist_mean/std` | 2D XY only |
| `final_angle_deg_mean/std` | |
| `n_steps_mean/std` | steps per rollout |
| `mode_probs` | mode 0/1 distribution over successful rollouts |

### Cross-seed aggregate (`aggregate_results.json`)

After all seeds: mean ± std of all scalar metrics, written once at `{save_root}/`.

---

## Output Directory Layout

```
logs/d3il_visual_aligning_baseline/
  {agent_name}/
    seed_{s}/
      diagnostics/
        rollout_0.gif
        rollout_0_stats.json
        rollout_0_summary.png
        rollout_1.gif
        …
      results_seed_{s}.json
    aggregate_results.json
```

This mirrors the DPCC/FM structure:
```
logs/aligning-d3il-visual/plans/visual_aligning_dpcc/{exp}/{seed}/results/
```
so the same Data_Analysis notebooks can load both.

---

## SLURM Script Design (`eval_d3il_baseline.sh`)

Model: copy structure of `eval_visual_aligning_dpcc.sh` exactly.

- `#SBATCH` header: same GPU/CPU/mem/time config
- Conda env `FMPCC`, PYTHONPATH setup
- `MUJOCO_GL=egl`, `MPLBACKEND=agg`
- Positional args: `$1` = seed (optional override), `$2` = record_mode (default `all`)
- Calls: `python d3il_visual_aligning_baseline_test/eval_d3il_visual_aligning.py`
         `$SEED_ARG --record "$RECORD_MODE"`

---

## Config YAML (`d3il_baseline_eval_config.yaml`)

```yaml
agent_cfg:        "d3il/configs/agents/ddpm_encdec_vision_agent.yaml"  # or bc_agent etc.
checkpoint_path:  "logs/aligning/runs/ddpm_encdec_vision/..."           # pre-trained weights
agent_type:       "ddpm_encdec_vision"   # used for output dir naming

seeds:            [42, 43, 44]
n_contexts:       3          # smoke: 3; full: 30
n_trajectories_per_context: 1
eval_on_train:    false
max_episode_length: 400
if_vision:        true       # true = capture frames; false = state-only eval

record_mode:      "all"      # all | failures | none
save_root:        "logs/d3il_visual_aligning_baseline"
```

---

## Comparison Table (D3IL vs DPCC/FM)

| Aspect | D3IL Baseline | DPCC / FM-PCC |
|---|---|---|
| Policy | D3IL agent (BC/DDPM/BESO/ACT…) | FM flow matching + SLSQP |
| Constraint projection | None | Yes (DPCC/SLSQP) |
| MPC foresight | None | Yes (H-step lookahead) |
| Output metrics | success, dist, final XY/angle, steps | + constraint metrics, plan viol rates |
| GIF/video | Yes (same pipeline) | Yes |
| Summary PNG | 2-panel XY trace | 9-panel (MPC+constraints+dist) |
| Aggregate JSON | Same schema subset | Full constraint metrics extra |

---

## Implementation Sequence (when ready to code)

1. Write `d3il_baseline_eval_config.yaml` — config skeleton only
2. Write `eval_d3il_visual_aligning.py`:
   - Config loader + arg parser
   - `D3ILBaselineWrapper` class
   - `_export_rollout_realtime` (GIF, JSON, PNG)
   - Summary + aggregate block
3. Write `Slurm_Codes/sbatch/d3il_visual_aligning_baseline/eval_d3il_baseline.sh`
4. Smoke test (1 seed, 2 contexts, `record_mode=all`) on cluster

---

## Open Questions (resolve at coding time)

| Question | Default assumption |
|---|---|
| Which D3IL agent is the primary baseline? | `ddpm_encdec_vision` (visual DDPM) — also support `bc` as secondary |
| Agent load path: hydra instantiate or manual `torch.load`? | Prefer manual load (no Hydra in eval script) unless agent `__init__` is too complex |
| Rendering: `if_vision=True` needed for video? | Yes — env must be started with `if_vision=True` to get frames; falls back to state-only if False |
| Summary PNG: full trajectory or only final state? | XY trajectory trace over the episode (same approach as DPCC) |
