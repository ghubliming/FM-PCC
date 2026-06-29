# U3 Changelog — Replicate d3il Checkpoint Selection

**Root cause fixed:** val-loss selection → simulation-success selection (exact d3il match)
**Expected impact:** recover toward paper `0.278 ± 0.071` success / `0.139 ± 0.054` entropy

---

## Files Changed

### `d3il_visual_aligning_baseline_test/train_d3il_visual_aligning.py`

**Removed** `_eval_vision_loss()` — offline MSE val-loss on static dataset.

**Replaced** `_train_vision(agent)` with `_train_vision(agent, train_sim)`:

```python
# before (WRONG — picked most overfitted checkpoint):
avg_val = _eval_vision_loss(agent)
if avg_val < best_val_loss:
    agent.store_model_weights(...)

# after (matches d3il/run_vision.py exactly):
successrate, _ = train_sim.test_agent(agent)
if successrate > best_success:
    agent.store_model_weights(...)
```

**In `main()`:** instantiate `train_sim` from `cfg.train_simulation` before calling `_train_vision`.
`cfg.train_simulation` already exists in `aligning_vision_config.yaml` — no config changes needed.
Uses `n_contexts=1, n_trajectories_per_context=1` → single rollout per check, negligible overhead.

**WandB:** `val_loss` → `train_sim_success`.

### `Slurm_Codes/sbatch/d3il_visual_aligning_baseline/train_d3il_baseline.sh`

Updated comment line 71: "best val-loss" → "best simulation-success (U3)".

---

## To Rerun

```bash
bash Slurm_Codes/sbatch/d3il_visual_aligning_baseline/run_all_seeds_d3il_baseline.sh
```

Submits all 6 seeds (0 1 2 3 4 42) as independent train→eval pipeline pairs in one command.
