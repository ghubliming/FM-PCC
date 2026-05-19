# Fix 1.3 — Eval Load-Path Mismatch (`max_path_length`) (2026-05-19)

## What is `max_path_length`?

`max_path_length` is a legacy D3IL config key that has **two completely different meanings** in the two Gen6V4 config blocks:

| Block | Value | What it actually does |
|-------|-------|-----------------------|
| `visual_aligning_dpcc` (train) | `1000` | Passed as `max_n_episodes` to `ParityAligningDataset.__init__()`. Controls how many episodes are loaded from the dataset pickle. Has **no step-count semantics** for this model. |
| `plan_visual_aligning_dpcc` (eval) | `512` ← **bug** | Embedded into the `diffusion_loadpath` format string as a literal key fragment. Has **no functional effect on rollout**; rollout length is controlled by `max_episode_length`. |

The semantic reuse is documented in the training block comment:
```python
# max_path_length is reused as max_n_episodes for ParityAligningDataset.
'max_path_length': 1000,
```

---

## The Bug

`args_to_watch_dpcc_train` lists `max_path_length` with abbreviation `'steps'`:
```python
args_to_watch_dpcc_train = [
    ...
    ('max_path_length', 'steps'),
]
```

The training `exp_name` is built from this list → the checkpoint directory name contains `steps{max_path_length}`.

With training `max_path_length=1000`, the checkpoint is saved to:
```
logs/aligning-d3il-visual/visual_aligning_dpcc/H8_K100_D..._aw10_steps1000/<seed>/
```

The eval `diffusion_loadpath` template:
```python
'diffusion_loadpath': 'f:visual_aligning_dpcc/H{horizon}_K{n_diffusion_steps}_D{diffusion}_aw{action_weight}_steps{max_path_length}',
```

With the buggy eval `max_path_length=512`, this resolves to:
```
logs/aligning-d3il-visual/visual_aligning_dpcc/H8_K100_D..._aw10_steps512/<seed>/
```

**That directory does not exist.** `load_diffusion_with_override()` raises `FileNotFoundError` before any rollout begins.

---

## Fix

**File**: `config/aligning-d3il-visual.py`

**Block**: `plan_visual_aligning_dpcc`

**Before**:
```python
'max_episode_length': 1000,
'max_path_length': 512,
```

**After**:
```python
'max_episode_length': 1000,
'max_path_length': 1000,   # MUST match visual_aligning_dpcc.max_path_length (fix_1.3)
```

### Why `max_episode_length` stays at 1000

`max_episode_length` controls the number of simulator steps allowed per rollout in `Aligning_Sim`. It is independent of `max_path_length` and does not need to change.

### Why changing `max_path_length` at eval is safe

`max_path_length` in `plan_visual_aligning_dpcc` appears **only** in the `diffusion_loadpath` and `prefix` format strings — it is never read by any eval execution logic. Changing it from 512 to 1000 only corrects the lookup key; it does not alter rollout behavior.

---

## Invariant Going Forward

> **`plan_visual_aligning_dpcc.max_path_length` MUST equal `visual_aligning_dpcc.max_path_length`.**
>
> If you change the number of training episodes, update **both** config blocks.

---

## Files Changed

| File | Change |
|------|--------|
| `config/aligning-d3il-visual.py` | `plan_visual_aligning_dpcc.max_path_length`: `512` → `1000`; added inline comment |
