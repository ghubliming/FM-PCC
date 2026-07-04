# CHANGELOG — FiLM_V2 (True FiLM) Implementation

**Date**: 2026-06-27
**Branch**: `update_into_FM`
**Plan**: [PLAN_FiLM_V2.md](./PLAN_FiLM_V2.md) · **Feasibility**: [Ideas.md](./Ideas.md)

---

## TL;DR

Added **True FiLM** (per-block γ scale + β shift) as an **OPT-IN** backbone behind a new
config key `film_mode`. **Default is `'v1'` = current "Fake FiLM" = byte-identical to before.**
True FiLM is only built when a config sets `film_mode: 'v2'`.

> [!IMPORTANT]
> **Nothing changes for current runs.** Every existing config, checkpoint, training command,
> and eval command behaves exactly as before, because `film_mode` defaults to `'v1'` and is
> NOT referenced by any v1 checkpoint path. v2 is dormant until explicitly selected.

---

## What was added / changed

### 🟢 NEW files (additive — nothing overwritten)

| File | Purpose |
|---|---|
| `fm_visual_aligning/models/unet1d_temporal_film.py` | `FiLMResidualTemporalBlock` + `UNet1DTemporalFiLMModel` (True FiLM backbone). |
| `diffuser_visual_aligning/models/unet1d_temporal_film.py` | Identical copy for the DDPM visual pipeline. |

Both are self-contained, import only from `.helpers` (same as v1), and are never imported
unless `film_mode == 'v2'`.

### 🟡 MODIFIED files (backward-compatible)

| File | Change | Default behavior |
|---|---|---|
| `fm_visual_aligning/models/visual_unet.py` | `__init__` now reads `film_mode = getattr(config, 'film_mode', 'v1')` and selects backbone: `v2` → `UNet1DTemporalFiLMModel`, else → `UNet1DTemporalCondModel`. `forward()` UNCHANGED. | `v1` → current backbone, identical. |
| `diffuser_visual_aligning/models/visual_unet.py` | Same selection logic with diffuser import path. | `v1` → current backbone, identical. |
| `config/aligning-d3il-visual.py` | Added optional `'film_mode': 'v1'` to 4 blocks: `fm_visual_aligning`, `plan_fm_visual_aligning`, `visual_aligning_dpcc`, `plan_visual_aligning_dpcc`. | All set to `'v1'` → no behavior change. |

### 🔵 UNTOUCHED (guaranteed zero diff)

- `unet1d_temporal_cond.py` (both pipelines) — the v1 backbone is byte-identical, so **every existing checkpoint still loads**.
- `visual_unet.forward()` — both backbones share the same call signature; no branch needed.
- All diffusion/flow engines, datasets, samplers, MPC, DPCC projector, vision encoder.
- All non-visual pipelines (UAV, state-only avoiding, `flow_matcher_v3_*`).
- `imf_visual_aligning` — **NOT modified** (gated on verification; see "Not done" below).

---

## Architecture delta

| | v1 (default, unchanged) | v2 (new, opt-in) |
|---|---|---|
| Per-block formula | `Conv(x) + time_mlp([t ‖ cond])` | `(1 + γ(v))·(Conv(x) + time_mlp(t)) + β(v)` |
| In-block `time_mlp` width | `2·dim` (time+cond concat) | `dim` (time only) |
| Visual delivery | concatenated into time embedding | separate per-block `film_proj` → (γ, β) |
| γ scale/gate | ❌ none | ✅ learned, **zero-init** (identity at step 0) |
| New params | — | ~1.2 M across 16 `film_proj` heads |
| Checkpoint compat | loads all current | **fresh training required** |

---

## How to USE v2 next time (test run)

1. In `config/aligning-d3il-visual.py`, in the `fm_visual_aligning` (and/or `visual_aligning_dpcc`) block:
   - set `'film_mode': 'v2'`
   - change `'prefix'` to a v2 folder, e.g. `'fm_visual_aligning_filmv2/'`
2. In the matching plan block (`plan_fm_visual_aligning`):
   - set `'film_mode': 'v2'`
   - change `'prefix'` and `'diffusion_loadpath'` to the same `fm_visual_aligning_filmv2` folder.
3. **Train from scratch** (v2 weights are architecturally incompatible with v1 — expected).
4. Eval as usual; console prints `[ VisualUNet ] film_mode=v2 — TRUE FiLM backbone ... ACTIVE`.

> [!NOTE]
> The prefix change in step 1–2 keeps v2 checkpoints in a SEPARATE directory so they can
> never overwrite or collide with existing v1 checkpoints. (We deliberately did NOT add
> `film_mode` to `args_to_watch`, because that would have renamed existing v1 checkpoint
> directories and broken loading of current models.)

---

## How to REVERT (fast)

### Option A — disable without deleting (instant, safe)
Do nothing, or ensure all `film_mode` keys are `'v1'`. v2 code is dormant and never executes.
Current behavior is already the default.

### Option B — full clean revert (remove all FiLM_V2 traces)
```bash
# 1. Delete the two NEW backbone files
rm fm_visual_aligning/models/unet1d_temporal_film.py
rm diffuser_visual_aligning/models/unet1d_temporal_film.py

# 2. Revert the 3 modified tracked files to their pre-change state
git checkout -- config/aligning-d3il-visual.py
git checkout -- fm_visual_aligning/models/visual_unet.py
git checkout -- diffuser_visual_aligning/models/visual_unet.py
```
After Option B the tree is exactly as before this change set.

### Option C — revert just one pipeline
Delete that pipeline's `unet1d_temporal_film.py` and `git checkout --` its `visual_unet.py`;
leave the other pipeline intact.

---

## Verification status

- [x] `py_compile` passes on all 5 touched/created files (syntax clean).
- [ ] Runtime forward-shape / identity-at-init / regression checks — **pending on Slurm cluster**
      (Docker dev box has no Python runtime per project env). See PLAN_FiLM_V2.md §10 for the
      exact checks to run after git sync.

---

## NOT done (deferred, by design)

- **`imf_visual_aligning`** — left untouched. Its visual-conditioning path must first be traced
  (it may route through `iMFTrajectoryModel`/`Flow_matcher_U_Net_v2`, which has no `cond_mlp`,
  rather than `UNet1DTemporalCondModel`). See PLAN_FiLM_V2.md §9 Task V0. Implement only after
  confirming the actual injection point.
- **Checkpoint auto-isolation via `args_to_watch`** — intentionally skipped to avoid renaming
  existing v1 checkpoint directories. v2 isolation is handled by the documented manual `prefix`
  override above.

---

## File-by-file change summary (for reviewers)

```
NEW   fm_visual_aligning/models/unet1d_temporal_film.py        (+~290 lines)
NEW   diffuser_visual_aligning/models/unet1d_temporal_film.py  (+~290 lines, identical)
EDIT  fm_visual_aligning/models/visual_unet.py                 (backbone selection in __init__)
EDIT  diffuser_visual_aligning/models/visual_unet.py           (backbone selection in __init__)
EDIT  config/aligning-d3il-visual.py                           (+'film_mode': 'v1' in 4 blocks)
```
