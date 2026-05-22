# Fix 12 — Timing Measurement Bug Fix

**Date**: 2026-05-22  
**Branch**: `update_into_FM`  
**Scope**: `fm_visual_aligning_test/eval_fm_visual_aligning.py`

---

## Problem

`avg_inference_time` was identical across `post_processing` and `diffuser` variants even though `post_processing` runs SLSQP projection per replan.

**Root cause**: `t_start = time.time()` was placed **outside** the `if self.action_counter == self.action_seq_size:` replan branch. With `action_seq_size=4`, only 1 in 4 `predict()` calls actually runs inference. The other 3 are cheap cached-action fetches (~0 s). Averaging over all `step_counter` calls diluted the real replan time by 4×, making both variants appear identical.

## Fix

| Location | Before | After |
|----------|--------|-------|
| Top of plan block | `t_start = time.time()` outside `if` | `t_replan = time.time()` as **first line inside** `if` block |
| Bottom of plan block | `self.curr_rollout_time += time.time() - t_start` outside `if`, after action fetch | `self.curr_rollout_time += time.time() - t_replan` as **last line inside** `if` block |
| `avg_time` denominator | `max(1, self.step_counter)` | `max(1, self._replan_count)` |
| Print label | `seconds/step` | `seconds/replan` |
| JSON key | `avg_inference_time_per_step` | `avg_inference_time_per_replan` |
| Aggregate terminal | `Avg inference time/step` | `Avg inference time/replan` |

The metric now correctly measures wall-clock time per planning call (diffusion forward pass + optional SLSQP + trajectory selection), making `post_processing` vs `diffuser` differences clearly visible.
