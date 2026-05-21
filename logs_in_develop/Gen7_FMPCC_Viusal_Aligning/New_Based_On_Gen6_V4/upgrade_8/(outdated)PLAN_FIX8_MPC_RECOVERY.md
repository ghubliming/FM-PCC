# Fix 8 — Full MPC Logic Recovery for Gen6V4 & Gen7 Visual Eval

**Date:** 2026-05-21  
**Scope:** `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` · `fm_visual_aligning_test/eval_fm_visual_aligning.py` · `config/aligning-d3il-visual.py` · `config/visual_aligning_eval.yaml`  
**Prerequisite reading:** `upgrade_8/RESEARCH_BATCH_SIZE_TRAJECTORY_SELECTION.md`

---

## 1. Problem Summary

Four separate issues were discovered that together constitute a broken MPC inference loop:

| # | Issue | Root Cause |
|:--|:--|:--|
| P1 | `batch_size = 6` hardcoded, ignores config | Commit `7b14333` — ad-hoc magic number, no justification |
| P2 | Config `plan.batch_size: 1` is a dead knob for inference (only stamps directory name) | Overwritten by hardcoded 6 before inference |
| P3 | `trajectory_selection='random'` is genuinely random (`np.random.randint`) — not DPCC semantics | DPCC `'random'` = always index 0 (deterministic); our version adds stochastic variance |
| P4 | MPC foresight diagnostic only stores/plots the **selected** trajectory, not all batch candidates | `history_full_plans.append(action_traj[0])` discards the other B-1 candidates after selection |

---

## 2. Is `plan.batch_size: 1` Dead?

**Partially.** It has two roles:
1. **Path stamping** — `{batch_size}` in the prefix template → produces the `H8_b1_...` directory name. This IS used.
2. **Inference control** — read as `getattr(args, 'batch_size', 1)` → immediately overwritten to 6 for non-diffuser variants. This is DEAD for inference.

Result: the output directory says `b1` but the model runs with 6 candidates. The directory name is a lie.

---

## 3. What DPCC Does (the Reference Design)

```
config/avoiding-d3il.py  plan.batch_size: 4
↓
eval.py  policy(conditions, batch_size=args.batch_size, ...)     ← no override
↓
Policy.__call__  samples all B trajectories  →  trajectory_selection picks one
  - 'random'  →  which_trajectory = 0          (deterministic, always first)
  - 'minimum_projection_cost'  →  argmin(costs) (picks lowest SLSQP cost)
  - 'temporal_consistency'     →  argmin(temporal diff vs prev step)
↓
samples.observations[:, :, :]  stored per replan step  ← FULL BATCH saved
↓
plot: for each replan step, for each b in min(B, 4): plot candidate XY in blue
```

Key: the batch is a **candidate pool**. Every candidate is kept for visualization. The selection is transparent and deterministic.

---

## 4. Changes Required

### 4.1 Config — `config/aligning-d3il-visual.py`

**Two plan sections to update:** `visual_aligning_dpcc plan` and `fm_visual_aligning plan`.

```python
# BEFORE (in both plan sections):
'batch_size': 1,

# AFTER:
'batch_size': 4,      # MPC candidate pool — matches DPCC reference (avoiding-d3il.py plan.batch_size: 4)
                      # Candidates compete under trajectory_selection; diffuser variant always overridden to 1.
```

**Why 4?** Matches DPCC paper default. Documented and principled. Tunable — ablate to 6/8 later if needed. After this change the directory path correctly reads `H8_b4_...` instead of the lying `H8_b1_...`.

---

### 4.2 Config — `config/visual_aligning_eval.yaml`

Expand `projection_variants` to expose trajectory-selection ablation as first-class variants, matching DPCC's `dpcc-r/c/t` naming convention. Each projected method gets three selection variants:

```yaml
projection_variants: [
  # ── Unconstrained baseline ──────────────────────────────────────────────
  'diffuser',                          # no projection, batch=1

  # ── Gradient (in-denoising) ─────────────────────────────────────────────
  'gradient',                          # gradient projection, first-index selection
  'gradient-tightened',

  # ── Post-processing: selection variants ─────────────────────────────────
  'post_processing',                   # selection = first (index 0) — deterministic baseline
  'post_processing-c',                 # selection = minimum_projection_cost
  'post_processing-t',                 # selection = temporal_consistency
  'post_processing-tightened',
  'post_processing-tightened-c',

  # ── Model-free: selection variants ──────────────────────────────────────
  'model_free',                        # selection = first
  'model_free-c',                      # selection = minimum_projection_cost
  'model_free-t',                      # selection = temporal_consistency
  'model_free-tightened',
  'model_free-tightened-c',
]
```

**Note:** `-r` suffix (random) is dropped — DPCC's "random" is always index 0 (deterministic), so we name it the default (no suffix). `-c` = minimum cost. `-t` = temporal consistency. This is cleaner than DPCC's `dpcc-r/c/t` because the projection method (post_processing vs model_free vs gradient) is already in the name.

---

### 4.3 Both Eval Scripts — Batch Size & Selection Logic

**Location:** the block just before `agent = VisualAgentWrapper(...)`, in the variant loop.

**Remove:**
```python
# REMOVE — magic number override:
if 'diffuser' not in variant:
    batch_size = 6
```

**Replace with:**
```python
# diffuser runs a single sample — no projection, no candidate selection, 4x cheaper.
# All projected variants use args.batch_size from plan config (the MPC candidate pool).
if 'diffuser' in variant:
    batch_size = 1
else:
    batch_size = getattr(args, 'batch_size', 4)
```

**Trajectory selection — full recovery:**
```python
# Recover DPCC-style trajectory selection from variant name suffix.
# Default (no suffix) = first index (deterministic, matches DPCC 'random'=0 semantics).
trajectory_selection = 'first'                        # default: deterministic index 0
if '-c' in variant:
    trajectory_selection = 'minimum_projection_cost'  # argmin SLSQP cost over batch
elif '-t' in variant:
    trajectory_selection = 'temporal_consistency'     # closest to previous step
# Existing special cases (kept for backward compat):
# 'dpcc-t' handled by elif above; 'dpcc-c' by -c check.
```

**Fix `'random'` semantics in `VisualAgentWrapper.get_action()`:**
The `elif self.trajectory_selection == 'random':` branch currently does `np.random.randint(self.batch_size)`. Replace with:
```python
# 'first' and legacy 'random' both resolve to index 0 — deterministic, matches DPCC.
# True randomness is never needed: all batch samples are IID from the same conditional.
else:
    which = 0
    selection_method = 'first (index 0)'
```

---

### 4.4 `VisualAgentWrapper` — Candidate Storage

**Problem:** `history_full_plans` stores only the selected trajectory's actions. The other B-1 candidates are discarded after `which` is chosen.

**New storage — add `history_all_candidates`:**

```python
# In __init__, add:
self.curr_rollout_all_candidates = []   # list of (B, H, 3) per replan step
self.curr_rollout_selected_idx   = []   # list of int — which was selected
self.history_all_candidates      = []
self.history_selected_idx        = []

# In reset(), add:
self.curr_rollout_all_candidates.clear()
self.curr_rollout_selected_idx.clear()

# In update_rollout_info(), add to rollout_info dict:
'all_candidates':   [c.copy() for c in self.curr_rollout_all_candidates],
'selected_idx':     list(self.curr_rollout_selected_idx),

# After self.history_full_plans.append(...), add:
self.history_all_candidates.append([c.copy() for c in self.curr_rollout_all_candidates])
self.history_selected_idx.append(list(self.curr_rollout_selected_idx))
```

**In `get_action()`, after `which` is determined:**
```python
# Store all candidates' action trajectories (dims 0:3) in robot space
all_cands_np = traj_np[:, :, :3]   # (B, H, 3) — normalized action space
self.curr_rollout_all_candidates.append(all_cands_np)
self.curr_rollout_selected_idx.append(int(which))
# existing line:
self.history_full_plans.append(action_traj[0].detach().cpu().numpy())
```

---

### 4.5 MPC Foresight Visualization — Both PNG Outputs

There are two PNG generation sites that need the same fix:

**Site A** — `_export_rollout_realtime()` per-rollout report (`rollout_N_report.png`):
- Panel `axes[0,0]` currently: plots every 4th selected plan in blue (thin, low alpha).
- **Fix**: plot all B candidates per replan step as thin light-blue lines; overlay the selected candidate as a bold blue line.

```python
# In _export_rollout_realtime:
all_cands_list = data.get('all_candidates', [])   # list of (B, H, 3) per replan step
sel_idx_list   = data.get('selected_idx',   [])

for step_i, (cands, sel) in enumerate(zip(all_cands_list, sel_idx_list)):
    if step_i % 4 != 0:   # subsample replan steps for readability
        continue
    start = plan_starts[min(step_i, len(plan_starts) - 1)]
    for b in range(cands.shape[0]):
        abs_plan = start + np.cumsum(cands[b, :, :3], axis=0)
        color = 'royalblue' if b == sel else 'lightblue'
        lw    = 1.5         if b == sel else 0.5
        alpha = 0.8         if b == sel else 0.3
        axes[0, 0].plot(abs_plan[:, 0], abs_plan[:, 1],
                        color=color, linewidth=lw, alpha=alpha)

axes[0, 0].set_title(f'XY — MPC foresight (bold=selected, N={len(all_cands_list)} replans)')
```

**Site B** — aggregate variant PNG (`post_processing.png`, axis `[i, 5]`):
- Same fix pattern. Replace the current `for p_idx, plan_deltas in enumerate(plans_list)` loop:

```python
# In the aggregate PNG loop (axes[i, 5]):
all_cands_list = rollout_data.get('all_candidates', [])
sel_idx_list   = rollout_data.get('selected_idx',   [])

axes[i, 4].plot(obs_traj[:, 0], obs_traj[:, 1], 'k-', linewidth=2, label='Real')
axes[i, 5].plot(obs_traj[:, 0], obs_traj[:, 1], 'k-', alpha=0.4)

for step_i, (cands, sel) in enumerate(zip(all_cands_list, sel_idx_list)):
    if step_i % 4 != 0:
        continue
    start = plan_starts[min(step_i, len(plan_starts) - 1)]
    for b in range(cands.shape[0]):
        abs_plan = start + np.cumsum(cands[b, :, :3], axis=0)
        color = 'royalblue' if b == sel else 'lightblue'
        lw    = 1.5 if b == sel else 0.5
        alpha = 0.8 if b == sel else 0.25
        axes[i, 5].plot(abs_plan[:, 0], abs_plan[:, 1],
                        color=color, linewidth=lw, alpha=alpha)

axes[i, 5].set_title(f'MPC Foresight — {cands.shape[0]} candidates/step')
```

---

## 5. What Stays Unchanged

| Item | Status | Reason |
|:--|:--|:--|
| `diffuser` variant batch=1 exception | Keep | No projection → no candidate diversity needed; 4× cheaper |
| `post_processing`/`model_free` → `-c` selection | Becomes explicit `-c` suffix | Now a named variant, not a silent override |
| Diagnostics duplicate path fix (gif/video) | Already applied | Done in FIX-7.2 session |
| `max_action_delta`, `constraint_types`, `enlarge_constraints` | Unchanged | Independent of MPC logic |

---

## 6. File Change Summary

| File | Changes |
|:--|:--|
| `config/aligning-d3il-visual.py` | `plan.batch_size: 1 → 4` (both plan sections) |
| `config/visual_aligning_eval.yaml` | Expand `projection_variants` with `-c`/`-t` suffix variants |
| `eval_visual_aligning_dpcc.py` | Remove `batch_size=6` hardcode; restore selection logic; add candidate storage; fix both PNG sites |
| `fm_visual_aligning_test/eval_fm_visual_aligning.py` | Identical changes |

---

## 7. Verification Checklist (Post-Implementation)

- [ ] Output directory path reads `H8_b4_...` (not `b1`) after config change
- [ ] `diffuser` variant still uses batch=1 (confirmed in agent init log)
- [ ] `dpcc-r` → `which=0` (random), `dpcc-c` → argmin cost, `dpcc-t` → temporal
- [ ] `post_processing`, `model_free`, `gradient` → `which=0` (random, DPCC default)
- [ ] `all_candidates` key present in `master_rollout_history` dict
- [ ] Per-rollout PNG `rollout_0_report.png` shows multiple light-blue candidate lines + one bold selected line in XY panel
- [ ] Aggregate `dpcc-c.png` panel [i,5] shows multiple candidate trajectories, not just selected replans
- [ ] No `batch_size = 6` anywhere in either eval script (`grep "batch_size = 6"` returns empty)

---

## 8. Post-Implementation Correction (Fix 8 Revision)

Sections 4.2 and 4.3 of this plan contained two errors discovered during code review:

**Error 1 — `projection_variants` in YAML (§4.2):**
The plan invented custom suffixes (`post_processing-c`, `post_processing-t`, `model_free-c`, etc.) that do not exist in the reference DPCC repo. The DPCC paper defines `dpcc-r/c/t` as the three selection variants of the full DPCC method; `post_processing` and `model_free` are comparison baselines that always use `'random'` (= index 0) selection. Custom `-c`/`-t` suffixes on baselines are not in the reference and were removed.

**Corrected YAML** (exact copy of `dpcc/config/projection_eval.yaml`):
```yaml
projection_variants: [
  # Figure 2, Table 1 and Figure 3:
  'dpcc-r', 'dpcc-r-tightened',
  'dpcc-c', 'dpcc-c-tightened',
  'dpcc-t', 'dpcc-t-tightened',
  # Table 1:
  'diffuser', 'gradient', 'gradient-tightened',
  'post_processing', 'post_processing-tightened',
  'model_free', 'model_free-tightened',
  # Table 2:
  'dpcc-c-tightened-dt0p25', 'dpcc-c-tightened-dt0p5',
  'dpcc-c-tightened-dt2p0',  'dpcc-c-tightened-dt4p0',
]
```

**Error 2 — trajectory selection logic (§4.3):**
The plan used `-c`/`-t` suffix matching, which would also trigger on `dpcc-c-tightened-dt0p25` via the `-c` check and on `model_free-t` (if it existed) via `-t`. The correct DPCC logic uses exact substring checks:

```python
# WRONG (invented):
if '-t' in variant: trajectory_selection = 'temporal_consistency'
elif '-c' in variant: trajectory_selection = 'minimum_projection_cost'
else: trajectory_selection = 'first'

# CORRECT (exact DPCC eval.py):
trajectory_selection = 'random'
if 'dpcc-t' in variant: trajectory_selection = 'temporal_consistency'
if 'dpcc-c' in variant: trajectory_selection = 'minimum_projection_cost'
```

Both errors corrected in all affected files (YAML + both eval scripts).
