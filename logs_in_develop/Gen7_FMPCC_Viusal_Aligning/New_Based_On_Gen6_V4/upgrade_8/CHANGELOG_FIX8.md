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

## P2 — Restore DPCC trajectory-selection semantics (corrected)

**File**: both eval scripts (variant loop)

**Corrected implementation** (exact `dpcc/scripts/eval.py` logic):
```python
trajectory_selection = 'random'
if 'dpcc-t' in variant: trajectory_selection = 'temporal_consistency'
if 'dpcc-c' in variant: trajectory_selection = 'minimum_projection_cost'
```

`'random'` = always index 0 (deterministic). Only `dpcc-t` and `dpcc-c` variants deviate. `post_processing`, `model_free`, `gradient`, `dpcc-r` all stay at `'random'`.

**Prior incorrect version** (initial Fix 8 — wrong substring matching):
```python
if '-t' in variant: trajectory_selection = 'temporal_consistency'
elif '-c' in variant or 'dpcc-c' in variant: trajectory_selection = 'minimum_projection_cost'
else: trajectory_selection = 'first'
```
This was wrong: `-c` would match `dpcc-c-tightened-dt0p25` via a generic `-c` check instead of the specific `dpcc-c` guard; `-t` would have incorrectly fired on any future variant containing `-t`; and `'first'` is not a DPCC-defined string.

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

## Config — `projection_variants` corrected to exact DPCC reference

**File**: `config/visual_aligning_eval.yaml`

**Correction**: initial Fix 8 invented custom `-c`/`-t` suffixes on `post_processing` and `model_free` (not in DPCC reference). These were removed. The YAML now contains the exact same list as `dpcc/config/projection_eval.yaml`:
- `dpcc-r/c/t` + `-tightened` variants (6 variants — the DPCC method)
- `diffuser`, `gradient`, `gradient-tightened`, `post_processing`, `post_processing-tightened`, `model_free`, `model_free-tightened` (7 baselines)
- `dpcc-c-tightened-dt0p25/0p5/2p0/4p0` (4 dt ablations)
Total: 17 variants.

---

## What Was NOT Changed

- `config/aligning-d3il-visual.py` `plan.batch_size` value (user decision: skip config number changes)
- `max_action_delta`, `constraint_types`, `enlarge_constraints` — independent of MPC logic
- `diffuser` variant: stays batch=1, no trajectory selection needed
- `gradient` variant: stays no suffix (first-index); gradient projection applied independently per sample so batch diversity gives no selection benefit

---

## D1 — `clip_denoised` made config-driven

**Files**: `eval_visual_aligning_dpcc.py`, `eval_fm_visual_aligning.py`, `config/aligning-d3il-visual.py`

Removed hardcoded `diffusion_model.clip_denoised = False`. Now reads `getattr(args, 'clip_denoised', False)` so the value comes from `plan_visual_aligning_dpcc.clip_denoised` / `plan_fm_visual_aligning.clip_denoised` in config. Default `False` in both config and code matches reference DPCC. Change to `True` in config to ablate.

---

## D4/B1 — Reverted initial-state row coefficient to original

**File**: `diffuser_visual_aligning/sampling/projection.py`, `fm_visual_aligning/sampling/projection.py`

`[DANGEROUS_FLAG_B1_SCALING]` treatment applied at all 3 sites (`build_matrices`, `project`, `compute_gradient`). Each site now shows:
- Upgrade reason in header comment
- Upgraded code in `# Upgraded code (not in use):` block
- Original code in `# Original code (implemented):` commented block
- Live code is the original (`mat_fix_initial[0, x_idx] = 1`, `b[...] = s_0[x_idx]`)

`_initial_state_x_diffs` field in `DynamicConstraints.__init__` commented out (was only needed for B1 scaling).

---

## D7/A4 — Reverted per-sample initial-state anchor to original

**File**: `diffuser_visual_aligning/sampling/projection.py`, `fm_visual_aligning/sampling/projection.py`

`[DANGEROUS_FLAG_A4_PER_SAMPLE_ANCHOR]` treatment applied at `project()` and `compute_gradient()`. Same comment structure as D4/B1: upgrade reason, upgraded code block, original code block, live code is original (`s_0 = trajectory_reshaped[0, ...]` outside the batch loop).

POSTMORTEM `§7.4` updated: D1, D4, D7 reclassified from `REVIEW` to `LEAVE`.
