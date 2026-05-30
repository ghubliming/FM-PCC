# Fix-18 (Gen7) — Non-Visual One-Shot Run: Code Fixes Applied

**Date**: 2026-05-30
**Branch**: `update_into_FM`
**Related**:
- [`INVESTIGATION_REPORT.md`](INVESTIGATION_REPORT.md) — full evidence + line refs
- [`SEVERITY_AND_RETRAIN_IMPACT.md`](SEVERITY_AND_RETRAIN_IMPACT.md) — what's actually broken vs. what isn't, and what to do with existing checkpoints
**Scope**: **Two code-level fixes applied** so the non-visual `K=1` DPCC train + `ODE=1` FM eval experiment can run end-to-end. Visual path remains unchanged (verified by user 2026-05-30).
**Source logs**: `temp/one_shot_run/visual_dpcc`, `temp/one_shot_run/visual_fm`

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

### What was NOT touched

- `config/aligning-d3il-visual.py` — variants are correct.
- `*/models/visual_unet.py` — the non-visual branch logic is fine; only the
  obs_dim value it reads was wrong.
- `*/datasets/sequence.py` — `StateOnlyAligningDataset` (UF-17) already
  produces 23-D correctly.
- The visual path of any of the above scripts.

---

## Files Changed in This Fix

| Action | File |
|---|---|
| Modified (code) | `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py` — Fix A |
| Modified (code) | `fm_visual_aligning_test/train_fm_visual_aligning.py` — Fix A |
| Modified (code) | `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` — Fix B |
| Modified (code) | `fm_visual_aligning_test/eval_fm_visual_aligning.py` — Fix B |
| Created | `fix_18_nonvisual_step1/INVESTIGATION_REPORT.md` |
| Created | `fix_18_nonvisual_step1/CHANGELOG.md` (this file) |
| Edited (report) | Status banner added confirming visual-path verification (2026-05-30) |
| Edited (report) | Recommendation #2 and §6 rewritten — vanilla FM train/eval ARE decoupled, so the 1-step FM under-integration is fixed by eval re-run, **not** retraining. Only mean-flow / iMeanFlow requires retraining. |

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
