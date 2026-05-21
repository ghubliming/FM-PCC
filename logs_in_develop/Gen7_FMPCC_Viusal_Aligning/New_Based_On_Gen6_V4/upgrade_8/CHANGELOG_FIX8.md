# Fix 8 — Full MPC Logic Recovery Changelog

**Date**: 2026-05-21  
**Branch**: `update_into_FM`  
**Plan Reference**: [PLAN_FIX8_MPC_RECOVERY.md](./PLAN_FIX8_MPC_RECOVERY.md)  
**Research Reference**: [RESEARCH_BATCH_SIZE_TRAJECTORY_SELECTION.md](./RESEARCH_BATCH_SIZE_TRAJECTORY_SELECTION.md)  
**Scope**: `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` · `fm_visual_aligning_test/eval_fm_visual_aligning.py` · `config/visual_aligning_eval.yaml`

---

## P1 — Remove magic `batch_size = 6` override

**File**: both eval scripts (variant loop, just before `agent = VisualAgentWrapper(...)`)

**Removed:**
```python
if 'diffuser' not in variant:
    batch_size = 6
```

**Replaced with:**
```python
if 'diffuser' in variant:
    batch_size = 1
else:
    batch_size = getattr(args, 'batch_size', 4)
```

`diffuser` uses batch=1 (no projection → no candidate diversity needed, 4× cheaper). All projected variants read `args.batch_size` from plan config; fallback is 4 to match DPCC reference.

---

## P2 — Restore DPCC trajectory-selection semantics

**File**: both eval scripts (variant loop)

**Removed** the old flat `trajectory_selection = 'random'` assignment.

**Replaced with:**
```python
if '-t' in variant:
    trajectory_selection = 'temporal_consistency'
elif '-c' in variant or 'dpcc-c' in variant:
    trajectory_selection = 'minimum_projection_cost'
else:
    trajectory_selection = 'first'
```

Suffix convention: no suffix = `first` (index 0, deterministic); `-c` = minimum SLSQP cost; `-t` = temporal consistency vs. previous step.

---

## P3 — Fix `'random'` non-determinism in `VisualAgentWrapper.get_action()`

**File**: both eval scripts (`get_action` method)

**Removed:**
```python
elif self.trajectory_selection == 'random':
    which = np.random.randint(self.batch_size)
    selection_method = 'random'
```

**Replaced with:**
```python
else:
    which = 0   # Fix 8: deterministic first-index (matches DPCC 'random'=0 semantics)
    selection_method = 'first (index 0)'
```

DPCC `Policy.__call__` resolves `'random'` to `which_trajectory = 0` (always the first sample). Our previous `np.random.randint` added stochastic eval variance that diverged from the reference design.

---

## P4 — Add full candidate storage to `VisualAgentWrapper`

**File**: both eval scripts

**New fields in `__init__`:**
```python
self.history_all_candidates      = []   # Fix 8: list of (B,H,3) per replan step per rollout
self.history_selected_idx        = []   # Fix 8: list of which index was chosen per replan step
self.curr_rollout_all_candidates  = []  # Fix 8: per-rollout accumulator cleared on reset()
self.curr_rollout_selected_idx    = []  # Fix 8: per-rollout accumulator cleared on reset()
```

**In `reset()`:**
```python
self.curr_rollout_all_candidates.clear()
self.curr_rollout_selected_idx.clear()
```

**In `update_rollout_info()`:**
```python
'all_candidates': list(self.curr_rollout_all_candidates),
'selected_idx':   list(self.curr_rollout_selected_idx),
...
self.history_all_candidates.append(list(self.curr_rollout_all_candidates))
self.history_selected_idx.append(list(self.curr_rollout_selected_idx))
```

**In `get_action()`, after `which` is determined:**
```python
self.curr_rollout_all_candidates.append(traj_np[:, :, :3].copy())   # (B, H, 3)
self.curr_rollout_selected_idx.append(int(which))
```

---

## P5 — Fix MPC Foresight visualization (per-rollout PNG)

**File**: both eval scripts (`_export_rollout_realtime`, `axes[0,0]` block)

**Before**: stored only the selected trajectory per replan step; all B-1 other candidates discarded after selection.

**After**: reads `data.get('all_candidates')` and `data.get('selected_idx')`; plots every 4th replan step; non-selected candidates plotted as thin lightblue lines (alpha=0.35), selected candidate as bold royalblue (alpha=0.85).

---

## P6 — Fix MPC Foresight visualization (aggregate variant PNG)

**File**: both eval scripts (aggregate `axes[i,5]` block)

Same fix as P5. Replaced flat `plans_list` iteration with candidate-aware loop using `rollout_data.get('all_candidates', [])` and `rollout_data.get('selected_idx', [])`. Title updated to `MPC Foresight — N candidates/step`.

---

## Config — `projection_variants` expanded

**File**: `config/visual_aligning_eval.yaml`

Added `-c` (minimum_projection_cost) and `-t` (temporal_consistency) suffix variants for `post_processing` and `model_free`, matching DPCC `dpcc-r/c/t` ablation structure. Total variants: 13 (was 7).

---

## What Was NOT Changed

- `config/aligning-d3il-visual.py` `plan.batch_size` value (user decision: skip config number changes)
- `max_action_delta`, `constraint_types`, `enlarge_constraints` — independent of MPC logic
- `diffuser` variant: stays batch=1, no trajectory selection needed
- `gradient` variant: stays no suffix (first-index); gradient projection applied independently per sample so batch diversity gives no selection benefit
