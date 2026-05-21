# Gen6V4 — Fix 4 Applied

**Date:** 2026-05-20
**Branch:** update_into_FM
**Primary record:** `logs_in_develop/Gen7_FMPCC_Viusal_Aligning/fix_4/FIX4_CHANGELOG.md`

---

## Summary

Fix 4 was designed for Gen7 diagnostics but applies equally to Gen6V4 because both
models share the same `VisualAgentWrapper` eval agent and `Aligning_Sim` sim loop.
All items were applied to `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`
at the same time as the Gen7 file.

---

## Changes applied to Gen6V4 eval

### `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`

| Change | What was added |
|---|---|
| `max_action_delta` param | Config-driven ODE safety clamp threshold (metres) |
| Direction-preserving clamp | `next_action_np * (max_action_delta / raw_mag)` in `predict()` |
| `curr_rollout_clamp_events` | Per-rollout list of `(step, raw_mag)` clamp events |
| `record_step_info()` | Accumulates `mean_distance` per step from `env.step()` info dict |
| PNG 3×3 grid | Expanded from 2×4 to 3×3 (18×15 figsize) with clamp-events scatter at `[2,2]` |
| Wired from config | `max_action_delta=config.get('max_action_delta', None)` in main block |

### `config/visual_aligning_eval.yaml`

```yaml
max_action_delta: 0.01  # metres; null = disabled
```

---

## No retraining required

These are eval-script and config changes only. No model weights or training code
were modified.
