# Gen7 Upgrade 11 — MPC Decision-Point Plot Overhaul

**Date**: 2026-05-22  
**Branch**: `update_into_FM`  
**Scope**: `fm_visual_aligning_test/eval_fm_visual_aligning.py`  
**File changed**: `_export_rollout_realtime()` standalone plot block only

---

## Problem

Previous standalone MPC foresight plot (`_mpc_foresight.png`) was unreadable:
- Green=selected / gray=others distinction added noise rather than clarity
- Every-other-replan sampling (`% 2`) still produced hundreds of overlapping line segments
- Dashed red line made the actual path harder to follow in dense plots
- PNG only — no vector format for inspection at any zoom level

## Changes

### Rendering logic

| Before | After |
|--------|-------|
| green=selected, gray=others | all candidates uniform **green** `lw=0.6 α=0.7` |
| every 2nd replan (`% 2`) | every **6th** replan (`_STRIDE=6`) |
| red **dashed** actual path | red **solid** `lw=1.2` |
| no replan markers | **black dot** `s=30` at anchor position for each shown replan |

### Anchor dot computation
For each shown replan index `step_i`:
```python
spr      = max(1, len(real_pos) // max(1, len(all_cands_list)))  # steps per replan
env_step = min(step_i * spr, len(real_pos) - 1)
anchor   = c_arr[env_step]   # actual robot pos at replan time; fallback: real_pos[env_step]
```
Dot placed on `c_pos` (actual path) — marks exactly **where the robot was** when it generated those 4 candidates.

### Legend
Custom `Line2D` legend entries (no auto-generated duplicates):
- Green line — `MPC candidates (N/step)`
- Black line — `des (commanded)`
- Red line — `actual (c_pos)`
- Black circle — `replan decision point`

### Output files
Both saved to `diagnostics/`:
- `rollout_{idx}_mpc_foresight.png` — 200 DPI (~5200×2200 px)
- `rollout_{idx}_mpc_foresight.svg` — vector, infinite zoom

### What did NOT change
- `figsize=(26, 11)` unchanged
- 9-panel `_report.png` unchanged
- Aggregate PNG (`{variant}.png`) unchanged — thin-line fix from Fix 9.1 still in place

---

## U11.2 — Yaml-settable stride + start/end markers

**Date**: 2026-05-22

### Changes

| | Before | After |
|---|---|---|
| Stride | `_STRIDE = 6` hardcoded | `_STRIDE = self.mpc_foresight_stride` read from yaml |
| Yaml key | — | `mpc_foresight_stride: 6` in `config/visual_aligning_eval.yaml` (default 6) |
| Start marker | none | lime `★` (`s=180`) at `c_arr[0]` (fallback `real_pos[0]`) on XY + XYZ panels |
| End marker | none | red `■` (`s=80`) at `c_arr[-1]` (fallback `real_pos[-1]`) on XY + XYZ panels |
| Legend | 4 entries | 6 entries — added `start` and `end` |

`VisualAgentWrapper.__init__` gains `mpc_foresight_stride=6` param; agent construction passes `config.get('mpc_foresight_stride', 6)`.
