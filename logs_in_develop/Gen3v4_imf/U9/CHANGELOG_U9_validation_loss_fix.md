# U9 Changelog: Validation Loss Visibility & Integrity Fixes

**Date:** July 7, 2026
**Follows:** `INVESTIGATION_validation_loss.md` (same folder). Simple fixes only (§5.1–5.4 of the investigation).
**Status:** code-complete, NOT runtime-verified — run on cluster (checklist in investigation §6).

## Changes

1. **`config/avoiding-d3il.py`** (iMF training block): added explicit `train_test_split: 0.9`. Was missing — validation only ran via a silent `getattr(..., 0.9)` fallback in the train script, so the split never appeared in the W&B run config.

2. **`FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py`**:
   - `log_wandb_from_losses` now takes an `after_step` cursor and returns the last logged step, so it can be called incrementally; also logs the new `val/raw_mse` metric and sets `run.summary['final_test_loss']` / `['final_val_raw_mse']` (was: full one-shot replay of train/test loss only, no summary keys).
   - Training now flushes losses to W&B **after every epoch** via the new `on_epoch_end` callback (was: one replay only after the seed fully completed — a SLURM 24h timeout left the W&B run permanently empty).

3. **`flow_matcher_v3_imeanflow/utils/training.py`** (Trainer):
   - Train/test `random_split` now uses a seeded generator (`split_seed=42`, new ctor param). Was unseeded: each resume re-split differently, leaking old test trajectories into training and making `best_test_loss`/`state_best.pt` selection compare across different test sets. Fixed seed also means all training seeds share the same held-out set (cross-seed comparable).
   - `train()` accepts optional `on_epoch_end(epoch)` callback (used for the per-epoch W&B flush).
   - `test()` now also aggregates and returns `raw_mse` (3-tuple, was 2-tuple; sole caller updated); tracked as `test_raw_mse_losses` in `losses.pkl` and in `state_*.pt`/`state_best.pt` payloads.
   - `test()` now restores `self.model.train()` after evaluating. Pre-existing DPCC-inherited bug: `eval()` was never undone, so the model sat in eval mode for all training after step 0. **No behavior change for current backbones** (only Bernoulli-mask condition dropout, unaffected by module mode) — hygiene/future-proofing for any nn.Dropout/BatchNorm backbone.

4. **`flow_matcher_v3_imeanflow/models/imf_diffusion.py`**: both loss paths now expose `info['raw_mse']` — the held-out companion metric behind `val/raw_mse`.
   - `meanflow_jvp` path: `per_sample.mean()` before the adaptive `1/(mse+c)^p` reweighting (the adaptive-weighted `test/loss` is self-referential and scale-compressed, weak for cross-run comparison).
   - `fm_equivalent` path: equals `main_loss` (no adaptive reweighting exists there) — same key for cross-objective parity.

## Not changed (deliberate)

- `test/loss` name and semantics kept (cross-gen continuity); `state_best.pt` still selected by it.
- No iMF-style sampling validation (1-NFE trajectory MSE) — deferred, see investigation §5.5.
- Sibling script `train_flow_matching_v3_ode_selectable.py` untouched (imports its own `flow_matcher_v3_ode_selectable.utils` trainer — verified isolated).

## Verify on cluster

Short training run → confirm W&B shows `train/loss`, `test/loss`, `val/raw_mse` live (per epoch), `train_test_split` in run config, and `losses.pkl` contains `test_raw_mse_losses`. Old `losses.pkl` files without the new key replay fine (`.get(..., [])`).
