# Fix-18 (Gen7) — Non-Visual One-Shot Run: Code Fixes Applied

**Date**: 2026-05-30
**Branch**: `update_into_FM`
**Related**:
- [`INVESTIGATION_REPORT.md`](INVESTIGATION_REPORT.md) — full evidence + line refs
- [`SEVERITY_AND_RETRAIN_IMPACT.md`](SEVERITY_AND_RETRAIN_IMPACT.md) — what's actually broken vs. what isn't, and what to do with existing checkpoints
- [`STALE_CONFIG_PATCH.md`](STALE_CONFIG_PATCH.md) — side-patch (`utils.Config` always-overwrite) + one-off regen script for pre-Fix-A `model_config.pkl` left over on disk
**Scope**: **Five code-level fixes** (18.1 Fix A / 18.2 Fix B / 18.3 Fix C / 18.4 Fix D / 18.5 Fix E) + one side-patch (`utils.Config`), so the non-visual `K=1` DPCC train + non-visual DPCC eval (all projection variants) + ODE=1 FM eval all run end-to-end. Visual path remains unchanged.
**Source logs**: `temp/one_shot_run/visual_dpcc`, `temp/one_shot_run/visual_fm`, plus the 2026-05-31 console log captured in [`fix_console_logs`](fix_console_logs) (regen-script execution that produced the fresh `model_config.pkl`) and SLURM job `21046` stderr (the UF-13 broadcast crash that motivated Fix C).

---

## Summary

User attempted a "one-shot DGM" experiment: train Visual-DPCC at
`n_diffusion_steps=1` and eval Visual-FM with `flow_steps_v3=1`. Both runs
were launched against the visual variants with a CLI override
`if_vision=False`. Both crashed.

- **DPCC train**: crashed on iteration 0 — `RuntimeError: weight of size
  [32, 9, 5], expected input[64, 23, 8] to have 9 channels`. The visual
  `visual_aligning_dpcc` variant hardcodes `obs_dim=6`; the CLI override
  flipped `if_vision=False` only, so dataset switched to 23-D
  (`StateOnlyAligningDataset`) but the model still built 9 input channels
  (`3+6`).
- **FM eval**: ran one variant (0/5 success — *separate* issue, see below),
  then crashed on the next projector setup —
  `ValueError: operands could not be broadcast together with shapes (23,) (9,)`
  at `Projector.build_matrices`. Root cause:
  `eval_fm_visual_aligning.py:1849` derives `_traj_dim = 23 if not if_vision
  else 9` from `args.if_vision` alone, ignoring that the loaded checkpoint
  and normalizer were visual (9-D).

**Verification by user (post-investigation)**: re-ran the same one-shot
experiment on the visual variants with `if_vision=True` (default). **Both
DPCC train and FM eval completed without crashing.** This confirms the two
bugs are **non-visual-path-only**, triggered exclusively by mis-mixing
visual variants with a CLI `if_vision=False` override.

---

## Code Fixes Applied

Two fixes were applied to source. Both are minimal and additive — they only
affect the non-visual code path; the visual path is bit-for-bit unchanged.

### Fix A — train scripts: override `args.obs_dim` for non-visual

**Files**:
- `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py` (around line 196)
- `fm_visual_aligning_test/train_fm_visual_aligning.py`         (around line 198)

**Symptom prevented**: `RuntimeError: weight of size [32, 9, 5], expected
input[64, 23, 8] to have 9 channels` at first conv on iteration 0.

**Root cause**: `VisualUNet.__init__` (`models/visual_unet.py:73-74`) computes
`transition_dim = action_dim + obs_dim` in the non-visual branch by reading
`args.obs_dim`. The visual variants in `config/aligning-d3il-visual.py`
hardcode `obs_dim=6` (the visual obs anchor). When a user CLI-overrides
`if_vision=False`, the dataset switches to 23-D but `args.obs_dim` stays at
6, so the model builds with 9 input channels.

**Patch**: after the dataset is constructed, if `if_vision=False`, override
`args.obs_dim` to `dataset.obs_normalizer.mins.shape[0]` (= 20 for
`StateOnlyAligningDataset`) **before** building `VisualUNet`. Logs an
explicit `[ train ] FIX-18: overriding args.obs_dim 6 → 20` line when the
override fires.

```python
if not _if_vision:
    _dataset_obs_dim = dataset.obs_normalizer.mins.shape[0]
    if getattr(args, 'obs_dim', None) != _dataset_obs_dim:
        print(f'[ train ] FIX-18: overriding args.obs_dim '
              f'{getattr(args, "obs_dim", None)} → {_dataset_obs_dim} '
              f'(non-visual; from dataset normalizer)')
        args.obs_dim = _dataset_obs_dim
```

### Fix B — eval scripts: derive `_traj_dim` from saved normalizers

**Files**:
- `fm_visual_aligning_test/eval_fm_visual_aligning.py`         (around line 1848)
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (around line 1849)

**Symptom prevented**: `ValueError: operands could not be broadcast together
with shapes (23,) (9,)` at `Projector.build_matrices`.

**Root cause**: `_traj_dim = 9 if args.if_vision else 23` reads the CLI flag
which can be flipped by UF-13's "record-mode auto-enable visual" path. The
checkpoint's saved normalizers are the immutable ground truth for what the
trained model actually produces.

**Patch**: derive `_traj_dim` from `act_normalizer.mins.shape[0] +
obs_normalizer.mins.shape[0]`. Adds a sanity warning if the sum is
unexpected (anything other than 9 or 23).

```python
_act_dim_norm = act_normalizer.mins.shape[0]
_obs_dim_norm = obs_normalizer.mins.shape[0]
_traj_dim = _act_dim_norm + _obs_dim_norm
```

### Fix C (= 18.3) — eval scripts: guard UF-13 record-mode flip on actual checkpoint type

**Added**: 2026-05-31 (after Fix A + Fix B unblocked training but eval still crashed downstream).

**Files**:
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (around lines 1902-1920)
- `fm_visual_aligning_test/eval_fm_visual_aligning.py`           (around lines 1903-1921)

**Symptom prevented**: `ValueError: operands could not be broadcast together
with shapes (1,6) (20,)` at `normalize()` inside `predict()`, when
running eval on a **non-visual** checkpoint with `--record all`.

**Root cause**: UF-13 used to indiscriminately set `if_vision = True`
whenever recording was on, regardless of what the checkpoint was actually
trained for. With a non-visual checkpoint (20-D obs_normalizer), this
forced `Aligning_Sim` into the visual code path, which then called
`agent.predict((bp_image, inhand_image, des_robot_pos, robot_pos),
if_vision=True)`. Inside, the visual branch built a 6-D obs vector and
tried to normalize against the 20-D normalizer → broadcast crash.

The pre-Fix-C UF-13 line:
```python
if not if_vision and args_cli.record != 'none':
    if_vision = True   # ← flips even when there's no image encoder
```

**Patch**: guard the flip on the saved normalizer dim. Only flip when
`obs_normalizer.mins.shape[0] == 6` (i.e. the checkpoint *is* visual).
For non-visual checkpoints, print a NOTE explaining that GIFs/videos
cannot be captured (the model has no image encoder) and proceed with
non-visual rollouts.

```python
_ckpt_is_visual = (obs_normalizer is not None
                   and obs_normalizer.mins.shape[0] == 6)
if not if_vision and args_cli.record != 'none':
    if _ckpt_is_visual:
        if_vision = True
        print('[ eval ] WARNING: ... auto-enabling visual mode ... (UF-13).')
    else:
        print('[ eval ] NOTE: record_mode is active but checkpoint is non-visual '
              f'(obs_normalizer dim = {obs_normalizer.mins.shape[0]}). '
              'Cannot auto-enable visual mode (this model has no image encoder); '
              'proceeding with non-visual rollouts. No GIFs/videos will be captured.')
```

**Consequence**: eval on a non-visual checkpoint with `--record all`
now succeeds and produces metrics + logs, but **no GIFs** (which is
correct: there's no image encoder in the model to render through).
If you want GIFs, you must train a visual checkpoint.

**Out-of-band patch shipped alongside Fix C** (see
[`STALE_CONFIG_PATCH.md`](STALE_CONFIG_PATCH.md)): `utils.Config.save()`
in both DPCC and FM `utils/config.py` previously skipped overwriting
`model_config.pkl` if a stale copy existed on disk. That mismatch caused
eval to instantiate a 9-D model from the stale config and fail to load a
fresh 23-D state dict (the bug surfaced *between* Fix-18 train success
and Fix C; the patch makes future training runs always overwrite, and
the one-off `regen_stale_model_config.py` script repairs existing
broken checkpoints without re-training). Not strictly part of Fix-18's
non-visual fixes but the same investigation thread; documented in its
own MD to keep this changelog focused.

### Fix D (= 18.4) — eval scripts: first-replan DIAG block referenced visual-only var names

**Added**: 2026-05-31 (after Fix C let eval reach the non-visual `predict()` path).

**Files**:
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (around lines 1571-1582)
- `fm_visual_aligning_test/eval_fm_visual_aligning.py`           (around lines 1557-1568)

**Symptom prevented**: `UnboundLocalError: local variable 'obs_6d_np' referenced before assignment`
at the first-replan diagnostic print, only on the non-visual code path.

**Root cause**: the visual branch of `predict()` defines `obs_6d_np` and
`obs_6d_norm` (the 6-D obs anchor + its normalized form). The non-visual
branch defines `obs_20d_np` and `obs_norm` instead — different names.
The one-shot first-replan diagnostic block hardcoded the visual names,
so reaching it from the non-visual path raised UnboundLocalError.

**Patch**: branch the diagnostic on `if_vision` and bind a pair of
local aliases (`_diag_obs_raw`, `_diag_obs_norm`) to whichever pair
exists. Also generalised the print to include the actual obs dim:

```python
if if_vision:
    _diag_obs_raw  = obs_6d_np      # (6,)
    _diag_obs_norm = obs_6d_norm    # (6,)
else:
    _diag_obs_raw  = obs_20d_np     # (20,)
    _diag_obs_norm = obs_norm       # (20,)
diag_lines += [
    f'[ DIAG obs ] des_c_pos={np.round(_diag_obs_raw[:3], 4)}  '
    f'c_pos={np.round(_diag_obs_raw[3:6], 4)}',
    f'[ DIAG obs ] obs_norm (dim={_diag_obs_norm.shape[0]})='
    f'{np.round(_diag_obs_norm, 4)}',
]
```

For non-visual, only the first 6 entries of the 20-D obs are
des_c_pos + c_pos (positions 0-2 and 3-5); the remaining 14 entries
(box pose, target pose) print fully via `obs_norm`. The "image health"
sub-block remains visual-only as before — that's correctly guarded.

**Consequence**: non-visual eval now passes the first-replan diagnostic
and continues into the rollout loop. No effect on visual path.

### Fix E (= 18.5) — eval scripts: `setup_dpcc_projector` slices normalizer to wrong width for 23-D trajectory

**Added**: 2026-05-31 (after Fix D let the first ("diffuser") variant complete a full 5-context rollout; crash moved to the second variant's projector setup).

**Files**:
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (around lines 90-100, inside `setup_dpcc_projector`)
- `fm_visual_aligning_test/eval_fm_visual_aligning.py`           (around lines 90-100, same function name)

**Symptom prevented**: `ValueError: operands could not be broadcast
together with shapes (23,) (9,)` at `Projector.build_matrices` line 401
(`a = bound[0] * (x_max - x_min) / 2`), when the second eval variant
needs a projector and the trajectory is non-visual (23-D). The first
variant (`diffuser`) does NOT instantiate a projector, so it ran clean
through all 5 contexts before the crash on variant 2.

**Root cause**: `setup_dpcc_projector` always sliced `obs_normalizer`
down to its first 6 dims:
```python
proj_obs_normalizer = obs_normalizer
if hasattr(obs_normalizer, 'mins') and len(obs_normalizer.mins) > 6:
    ... = obs_normalizer.mins[:6] ...   # ← hardcoded 6
```
Visual: obs_normalizer is 6-D → no-op. Fine.
Non-visual: obs_normalizer is 20-D → trimmed to 6-D, leaving the
projector's `self.normalizer` with `3 act + 6 obs = 9-D` ranges. But the
halfspace bound vector is built at the full `trajectory_dim = 23` width
by `formulate_halfspace_constraints`. `(23,) * (9,)` → crash.

**Patch**: derive the slice target from `trajectory_dim - action_dim`
(=20 for non-visual, =6 for visual), so the slice only fires when the
normalizer is truly oversized for the trajectory at hand:

```python
_target_obs_dim = trajectory_dim - 3   # action_dim hardcoded 3 throughout
proj_obs_normalizer = obs_normalizer
if hasattr(obs_normalizer, 'mins') and len(obs_normalizer.mins) > _target_obs_dim:
    ...slice to _target_obs_dim...
```

Why this is safe for the trailing 14 trajectory dims (positions 9-22 in
non-visual): they carry **zero bound coefficients** from
`formulate_halfspace_constraints` (which only emits non-zero entries at
the explicit `_DIM` indices, all in 0-8). So `bound[0] * range = 0 *
anything = 0` for those positions, contributing nothing to the
constraint matrix. The slice change keeps PCC's robot-kinematic
constraints (dims 0-8) bit-identical to before; it only enlarges the
normalizer so the shape arithmetic works.

**Consequence**: non-visual eval can now build the projector for any
variant after `diffuser` (`dpcc-r`, `dpcc-c`, `dpcc-t`, post-processing,
gradient, …). Visual eval is unchanged (the slice condition `len > 6`
was already false for visual; new condition `len > 6` is still false).

### What was NOT touched

- `config/aligning-d3il-visual.py` — variants are correct.
- `*/models/visual_unet.py` — the non-visual branch logic is fine; only the
  obs_dim value it reads was wrong.
- `*/datasets/sequence.py` — `StateOnlyAligningDataset` (UF-17) already
  produces 23-D correctly.
- `Aligning_Sim` (`d3il/simulation/aligning_sim.py`) — non-visual branch
  already worked correctly; Fixes C and D just let the eval driver reach
  it and survive its first-replan diagnostic.
- `*/sampling/projection.py` — `Projector` and `build_matrices` are
  correct; the bug was in how the call site sized the normalizer it
  passed in (Fix E).
- `formulate_halfspace_constraints` — emits zero-padded bound vectors
  correctly; the consumer's normalizer just needed to be sized to match.
- The visual path of any of the above scripts.

---

## Files Changed in This Fix

### Code (sources)

| Action | File | Fix |
|---|---|---|
| Modified | `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py` | A (= 18.1) |
| Modified | `fm_visual_aligning_test/train_fm_visual_aligning.py`         | A (= 18.1) |
| Modified | `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`  | B (= 18.2), C (= 18.3), D (= 18.4), E (= 18.5) |
| Modified | `fm_visual_aligning_test/eval_fm_visual_aligning.py`          | B (= 18.2), C (= 18.3), D (= 18.4), E (= 18.5) |
| Modified | `diffuser_visual_aligning/utils/config.py`                    | side-patch (STALE_CONFIG, always-overwrite `model_config.pkl`) |
| Modified | `fm_visual_aligning/utils/config.py`                          | side-patch (STALE_CONFIG, always-overwrite `model_config.pkl`) |

### Docs / one-off scripts

| Action | File |
|---|---|
| Created | `fix_18_nonvisual_step1/INVESTIGATION_REPORT.md` |
| Created | `fix_18_nonvisual_step1/CHANGELOG.md` (this file) |
| Created | `fix_18_nonvisual_step1/SEVERITY_AND_RETRAIN_IMPACT.md` |
| Created | `fix_18_nonvisual_step1/STALE_CONFIG_PATCH.md` (documents the `utils.Config` side-patch + regen script) |
| Created | `fix_18_nonvisual_step1/regen_stale_model_config.py` (one-off cleanup for pre-Fix-A `model_config.pkl` left over on disk) |
| Edited (report) | Status banner added confirming visual-path verification (2026-05-30) |
| Edited (report) | Recommendation #2 and §6 rewritten — vanilla FM train/eval ARE decoupled, so the 1-step FM under-integration is fixed by eval re-run, **not** retraining. Only mean-flow / iMeanFlow requires retraining. |

### Fix numbering recap

- **18.1 (Fix A)** — train scripts override `args.obs_dim` so the model is built 23-D for non-visual.
- **18.2 (Fix B)** — eval scripts derive `_traj_dim` from the saved normalizer (immune to UF-13).
- **18.3 (Fix C)** — eval scripts guard the UF-13 record-mode `if_vision` flip on the saved normalizer dim, so a non-visual checkpoint isn't forced into the visual `predict()` path.
- **18.4 (Fix D)** — eval scripts' first-replan DIAG block aliases `obs_6d_np`/`obs_6d_norm` (visual) vs `obs_20d_np`/`obs_norm` (non-visual) so neither path hits `UnboundLocalError`.
- **18.5 (Fix E)** — `setup_dpcc_projector` now slices the obs normalizer to `trajectory_dim - action_dim` instead of a hardcoded 6, so the 23-D non-visual trajectory gets matching 20-D obs ranges and the projector's bound × range arithmetic stops broadcast-erroring at variant 2+.
- **Side-patch (STALE_CONFIG)** — `utils.Config.save()` always overwrites `model_config.pkl`; sibling regen script repairs pre-existing broken checkpoints without re-training.

---

## Key Findings

1. **Visual path is internally consistent.** `visual_aligning_dpcc` and
   `fm_visual_aligning` both define `obs_dim=6`, `if_vision=True`, model
   spec, dataset spec, and normalizer dims in a single self-consistent
   configuration. There is no plumbing to mismatch.

2. **Non-visual path requires using the matching variant.** The
   `ddpm_encdec_vision_nonvisual` variant (UF-17) is the only correct way to
   run non-visual aligning. CLI-overriding `if_vision=False` on a visual
   variant produces inconsistent state because three components read
   different sources of truth:
   - Dataset reads `config.if_vision` (CLI-mutable).
   - Model reads `config.obs_dim` (frozen at variant definition).
   - Eval projector reads `args.if_vision` (CLI-mutable).

3. **FM 0% success at 1-step Euler is expected**, not a bug. The model
   learned a curved velocity field; integrating it with `Δt=1` lands far
   outside the data manifold. Cure: more eval steps, OR switch to mean-flow
   if you want one-shot validity by construction.

4. **DDPM is fundamentally different from FM here.** DDPM's discrete noise
   schedule means train-time T and eval-time T must match — so testing
   "DDPM at T=1" *does* require retraining. FM does not.

---

## Recommended Next Steps (Per Report §5–§6)

Not applied as code in this fix; tracked here for follow-up:

| Priority | Action |
|---|---|
| High | Re-run DPCC training with `prefix=ddpm_encdec_vision_nonvisual/`, `n_diffusion_steps=1`. Drop the `if_vision=False` CLI override (the variant sets it already). |
| High | Sweep FM eval `flow_steps_v3 ∈ {1, 2, 5, 10, 20}` on the existing checkpoint to characterize curvature of the learned flow. |
| Medium | Add guardrails (§5 of the report) — `_traj_dim` derived from normalizer dims, train-time `obs_dim` assertion, variant/flag coherence check at CLI parse. |
| Low | Only if step-count sweep shows fundamental 1-step failure: train an iMeanFlow variant. |

---

## Sync Note

Documentation-only fix. **Sync to Gen6V4 is parallel**, not a code copy. See
[`Gen6V4_dataset_upgrade_visual_dpcc/Gen7_fix18_applied/CHANGELOG.md`](../../../Gen6_dpcc_Engine_for_visual_aligning/Gen6V4_dataset_upgrade_visual_dpcc/Gen7_fix18_applied/CHANGELOG.md).
