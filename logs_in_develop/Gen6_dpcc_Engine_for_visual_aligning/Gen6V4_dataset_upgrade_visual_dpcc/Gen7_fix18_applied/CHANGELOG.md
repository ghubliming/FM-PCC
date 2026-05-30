# Fix-18 Applied to Gen6V4 (DPCC) — Non-Visual One-Shot Run

**Date**: 2026-05-30
**Branch**: `update_into_FM`
**Source (Gen7 canonical)**: [`fix_18_nonvisual_step1/CHANGELOG.md`](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_18_nonvisual_step1/CHANGELOG.md)
**Investigation report**: [`fix_18_nonvisual_step1/INVESTIGATION_REPORT.md`](../../../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_18_nonvisual_step1/INVESTIGATION_REPORT.md)
**Scope**: **Two code fixes applied to the DPCC half of the stack**, mirroring the FM fixes in Gen7. Visual path unchanged (verified working by user 2026-05-30).

---

## What Was Diagnosed

Two crashes during a "one-shot DGM" experiment (DPCC `n_diffusion_steps=1`
train + FM `flow_steps_v3=1` eval) launched against the **visual** variants
with a CLI override `if_vision=False`:

| Component | Failure | Root cause |
|---|---|---|
| `diffuser_visual_aligning` train | `RuntimeError: weight of size [32, 9, 5], expected input[64, 23, 8]` at first conv, iteration 0 | Visual variant hardcodes `obs_dim=6`; CLI flipping `if_vision=False` switched the dataset to 23-D but the model still built 9 input channels. |
| `fm_visual_aligning` eval | `ValueError: operands could not be broadcast together with shapes (23,) (9,)` at `Projector.build_matrices` | `eval_fm_visual_aligning.py:1849` derives `_traj_dim` from `args.if_vision` alone; loaded checkpoint + normalizer were visual (9-D). |

User-verified post-investigation: re-running both jobs on the visual
variants with `if_vision=True` (default) completes without crashing. The
crashes are **non-visual-path-only**, caused by selecting a visual variant
and CLI-forcing `if_vision=False` instead of selecting the matching
`ddpm_encdec_vision_nonvisual` variant.

---

## Code Fixes Applied to DPCC

Two fixes mirror what was applied to FM in Gen7. Both are minimal, additive,
and only touch the non-visual code path.

### Fix A — `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py`

After dataset construction, when `if_vision=False`, override
`args.obs_dim` to `dataset.obs_normalizer.mins.shape[0]` (= 20) **before**
building `VisualUNet`. This makes the model's non-visual branch compute
`transition_dim = action_dim + obs_dim = 3 + 20 = 23` instead of inheriting
the visual variant's hardcoded `obs_dim=6` (which produced the 9 vs 23
first-conv crash).

### Fix B — `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`

Replace
```python
_traj_dim = 9 if args.if_vision else 23
```
with derivation from the saved normalizer dims:
```python
_traj_dim = act_normalizer.mins.shape[0] + obs_normalizer.mins.shape[0]
```

This makes projector dimensionality follow the actual checkpoint instead of
the CLI flag (which can be flipped by UF-13's record-mode auto-enable).

### What was NOT touched

- `config/aligning-d3il-visual.py` — `ddpm_encdec_vision_nonvisual` variant
  is correct.
- `diffuser_visual_aligning/models/visual_unet.py` — non-visual branch
  logic is fine; only the `obs_dim` value it received was wrong.
- `diffuser_visual_aligning/datasets/sequence.py` —
  `StateOnlyAligningDataset` already produces 23-D correctly.
- The visual path of any DPCC script.

---

## Files Modified in Gen6V4 (DPCC) by This Fix

| Action | File |
|---|---|
| Modified | `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py` — Fix A |
| Modified | `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` — Fix B |

The FM counterparts of these two fixes are tracked under the Gen7 canonical
entry; both stacks now use the same derivation logic so DPCC and FM stay
behaviourally aligned.

---

## Recommended Next Steps for Gen6V4 (DPCC) Specifically

| Priority | Action |
|---|---|
| High | Re-run DPCC `n_diffusion_steps=1` non-visual training with `prefix=ddpm_encdec_vision_nonvisual/`. Fix A ensures the model now builds a 23-channel input regardless of any stale `obs_dim` in the variant inheritance chain. |
| High | For the visual one-shot DDPM experiment (which user verified runs end-to-end), characterize success-rate-vs-T curves directly. |
| Low | Optional: enforce variant/flag coherence at CLI parse time (refuse `prefix=visual_aligning_dpcc/` + `if_vision=False`). Fix A makes this non-critical but it would still surface user errors earlier. |

---

## Sync Provenance

- Gen7 canonical: `Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_18_nonvisual_step1/`
- This sync:     `Gen6V4_dataset_upgrade_visual_dpcc/Gen7_fix18_applied/`

Both folders contain only documentation. Investigation report lives only at
the Gen7 location; this changelog references it via the link above.
