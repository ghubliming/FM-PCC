# U9 Changelog: Metric-Parity Pass — full DPCC + imeanflow loss set into W&B

**Date:** July 8, 2026
**Follows:** `CHANGELOG_U9_validation_loss_fix.md` and `INVESTIGATION_validation_loss.md` §7.4 (metric inventory) in this folder. Purpose: debugging visibility — close every gap in the §7.4 table so training W&B carries all DPCC/Gen0 metrics **and** the imeanflow-repo loss analogs.
**Status:** code-complete, NOT runtime-verified — run on cluster.

## W&B metrics after this pass (per step, live per epoch)

| W&B key | Quantity | Parity with |
|---|---|---|
| `train/loss` | adaptive-weighted objective, train batch | (existing) DPCC `loss` / imeanflow `loss` |
| `test/loss` | same, held-out avg over 100 batches | (existing) DPCC `loss_test` |
| `val/raw_mse` | held-out raw (un-reweighted) MSE | (existing, U9) — no upstream analog |
| `train/a0_loss` | **new** first-action loss, train side | DPCC/Gen0 `train/a0_loss` |
| `test/a0_loss` | **new** first-action loss, held-out | DPCC/Gen0 `test/a0_loss` |
| `train/raw_mse` | **new** raw MSE, train side | imeanflow `loss_u` (raw component) |
| `train/aux_loss` | **new** aux v-head MSE, train side | imeanflow `loss_v` |
| `test/aux_loss` | **new** aux v-head MSE, held-out | (imeanflow has no val side) |
| `train/lr` | **new** learning rate (cosine+warmup schedule) | DPCC `lr` (tqdm-only there, never persisted) |

Summaries: `final_train_loss` (**new**, Gen0 parity), `final_test_loss`, `final_val_raw_mse` (existing).

## Changes

1. **`flow_matcher_v3_imeanflow/utils/training.py`** (Trainer):
   - New tracked series: `train_raw_mse_losses`, `train_aux_losses`, `test_aux_losses` (appended at the same `log_freq` steps as the existing series; presence-guarded on the `infos` keys), and `lr_history` (`lr_scheduler.get_last_lr()[0]`, always present). The lr curve is a debugging metric, not a loss: it verifies the cosine+warmup scheduler resumed correctly (it is rebuilt from scratch on every resume — a wrong warmup/total-step count shows up as a saw-tooth or re-warmup) and explains loss spikes/plateaus.
   - `test()` now also aggregates `aux_loss` → returns a **4-tuple** `(test_loss, test_a0_loss, test_raw_mse, test_aux)`; `test_aux` is **`None`** (not 0) when the objective exposes no aux head — e.g. `meanflow_jvp` with `meanflow_aux_weight=0` — so no fake-zero curves get logged. Sole caller (`train_epoch`) updated.
   - New series persisted everywhere the old ones are: `state_{epoch}.pt` / `state_best.pt` payloads, `losses.pkl` + `.json` (keys `training_raw_mse_losses`, `training_aux_losses`, `test_aux_losses`, `lr_history`; the resume-merge in `save_losses` is key-generic, no change needed), and both `load()` restore paths (checkpoint and pkl fallback, `.get(..., [])` so old checkpoints load fine).
   - tqdm postfix now also shows `raw_mse` / `aux_loss` when present (console debugging on the cluster `.log`s).

2. **`FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py`** (`log_wandb_from_losses`):
   - Replaced the hand-rolled per-metric dicts with a `companion_keys` pkl-key→W&B-key map covering the full table above; missing pkl keys (old runs, no-aux objectives) are skipped silently.
   - Added `run.summary['final_train_loss']`.
   - Incremental `after_step` cursor semantics unchanged (per-epoch live flush still works).

## Semantics notes (no behavior change)

- `state_best.pt` selection still uses `test/loss` — unchanged.
- No new quantity was invented: `raw_mse` and `aux_loss` already existed in the model's `info` dict (`imf_diffusion.py`, both objective paths); this pass only records and uploads them. `meanflow_jvp` sets `aux_loss` only when `meanflow_aux_weight > 0`; `fm_equivalent` always sets it.
- Naming: the held-out raw MSE keeps its established `val/raw_mse` key (summary key + prior docs depend on it); its train-side twin is `train/raw_mse`. They are the same quantity on different splits.
- Cross-engine caveats from investigation §7.1–7.3 still apply: the a0/raw/aux values are **not** numerically comparable to DPCC runs (ε-space vs u-space); parity here means *coverage*, not comparability.

## Still NOT included (unchanged gaps)

- **imeanflow sampling-quality check (FID analog)** — the §5.5 deferred item (periodic 1-NFE / eval-NFE held-out trajectory MSE). This is the one remaining imeanflow metric family missing; needs a sampling hook in the trainer, out of scope for this pass.

## Verify on cluster

Short training run (`meanflow_jvp`, aux weight > 0 to exercise the aux path) → confirm:
- W&B shows all 9 curves live (incl. `train/lr` rising through warmup then cosine-decaying) and `final_train_loss` in the summary;
- `losses.pkl` contains the three new keys; a resume from an **old** checkpoint (without them) loads cleanly and starts the new series from the resume step;
- with `meanflow_aux_weight=0`: no `train/aux_loss` / `test/aux_loss` curves appear (and no crash).
