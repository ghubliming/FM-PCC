# U3 — Replicate d3il Behavior Exactly: Checkpoint Selection + Audit

**Based on:** U2.3 audit + U2.4 covariate-shift confirmation + cross-code read of `/workspaces/d3il/run_vision.py`
**Scope:** training wrapper only — eval script already verified correct (U2.3 §A)

---

## All Divergences Found (Exhaustive Audit)

### D1 — CRITICAL: Checkpoint Selection (val-loss vs simulation success)

| | FM-PCC `train_d3il_visual_aligning.py` | d3il `run_vision.py` |
|---|---|---|
| Every `eval_every_n_epochs` | compute offline MSE val-loss | run `train_sim.test_agent(agent)` (1 rollout) |
| Save criterion | `avg_val < best_val_loss` | `successrate > best_success` |
| What gets saved | most overfitted checkpoint | checkpoint that actually solves the task |

**Root cause of ~0% success.** Confirmed by U2.4.

**Fix:** In `_train_vision(agent, train_sim)`:
- Remove `_eval_vision_loss` call and `best_val_loss`
- Add `successrate, _ = train_sim.test_agent(agent)`
- Save on `successrate > best_success`
- Pass `train_sim = hydra.utils.instantiate(cfg.train_simulation)` from `main()` into `_train_vision`

`train_simulation` config is already in `aligning_vision_config.yaml`:
```yaml
train_simulation:
  n_contexts: 1          # ← only 1 rollout per check — very fast
  n_trajectories_per_context: 1
  n_cores: 1
```
No config changes needed.

**WandB:** replace `wandb.log({"val_loss": avg_val})` with `wandb.log({"train_sim_success": successrate})`

---

### D2 — MINOR: Post-training final simulation (d3il does it; FM-PCC delegates)

d3il `run_vision.py` lines 76-81 after training finishes:
```python
agent.load_pretrained_model(agent.working_dir, sv_name=agent.eval_model_name)
env_sim = hydra.utils.instantiate(cfg.simulation)   # 60 ctx × 8 traj × 5 cores
env_sim.test_agent(agent)
```

FM-PCC: the training script does NOT do this. Instead it chains `afterok` to the separate `eval_d3il_baseline.sh` sbatch job, which loads `eval_best_ddpm.pth` and runs its own rollout loop.

**Verdict: acceptable — NOT a bug.** The separate eval job does the equivalent and is actually more thorough (60 ctx × 18 traj with entropy). The only risk is if eval is NOT chained (e.g. manual train-only run) — then no post-train eval. Document only, no code change needed.

---

### D3 — COSMETIC: Train sbatch comment is wrong

`train_d3il_baseline.sh` line 72:
```
# D3IL paper (ICLR 2024): 200 epochs for image agents, 500 for state agents;
# eval every 1/10th of total training; best val-loss checkpoint saved.   ← WRONG after D1 fix
```

After D1 fix, update to: `best simulation-success checkpoint saved`.

---

### D4 — COSMETIC: WandB mode during training

d3il `run_vision.py`: `wandb.init(mode="disabled")` — intentionally silent.
FM-PCC: conditional on `use_wandb` (defaults online). Not a correctness issue.
Keep FM-PCC behavior (logging is useful). No change.

---

## What Was Ruled Out (Do NOT re-investigate)

From U2.3 — all verified correct, no divergence:

| Item | Status |
|---|---|
| Inference loop logic | ✓ byte-for-byte match to `aligning_sim.py` |
| Success definition (`_check_early_termination`, latching, lag) | ✓ correct |
| Obs preprocessing (BGR/RGB, `/255`, transpose) | ✓ correct |
| Action convention (`pred + des_robot_pos`, quat `[0,1,0,0]`) | ✓ correct |
| Agent history reset between rollouts | ✓ correct |
| EMA save/load | ✓ correct |
| Dataset / config files | ✓ identical to d3il |
| `epoch=200`, `eval_every_n_epochs=20` (epoch/10) | ✓ matches paper |

---

## Fix Summary (what to code in U3)

**1 file to change: `d3il_visual_aligning_baseline_test/train_d3il_visual_aligning.py`**

- Delete `_eval_vision_loss()` function
- Change `_train_vision(agent)` → `_train_vision(agent, train_sim)`
- Inside: replace val-loss block with `successrate, _ = train_sim.test_agent(agent)` + `best_success` logic
- In `main()`: add `train_sim = hydra.utils.instantiate(cfg.train_simulation)` before calling `_train_vision`
- Update `wandb.log` key

**1 comment to fix: `Slurm_Codes/sbatch/d3il_visual_aligning_baseline/train_d3il_baseline.sh`**
- Line 72: update comment from "best val-loss" to "best simulation-success"

That is all. Two files, one real code change.

---

## Expected Outcome After Fix

Training now saves the checkpoint that actually achieves highest MuJoCo success during training
(same as d3il). The 1-rollout sim check is cheap (n_ctx=1, n_traj=1) — negligible overhead.
Eval should recover toward the paper's `0.278 ± 0.071` success / `0.139 ± 0.054` entropy.
