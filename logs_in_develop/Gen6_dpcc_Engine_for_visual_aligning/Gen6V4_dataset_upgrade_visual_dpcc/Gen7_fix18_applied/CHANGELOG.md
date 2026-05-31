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

---

## Update 2026-05-31 — Fix-18 expanded to 18.6 (Fix-18.1 → 18.6 + STALE_CONFIG side-patch)

The original Fix-18 commit (`606ad1e`, May 30) bundled Fix-18.1 (train obs_dim
override) and Fix-18.2 (eval `_traj_dim` from normalizer). Subsequent crashes
surfaced four more issues on the genuine 23-D non-visual code path, each
fixed in a follow-up commit:

| Fix | Commit | Touches DPCC? | Touches FM? | What it fixes |
|---|---|---|---|---|
| 18.3 UF-13 normalizer-dim guard | `761b2ef` | ✅ | ✅ | UF-13 indiscriminately flipped `if_vision=True` for record mode → visual `predict()` crashes on 23-D model (`(1,6) vs (20,)`). Guarded on `obs_normalizer.mins.shape[0] == 6`. |
| 18.4 DIAG var alias | `20a1895` | ✅ | ✅ | First-replan diagnostic in `predict()` referenced visual-branch-only var names → `UnboundLocalError` on the non-visual path. |
| 18.5 projector slice `_target_obs_dim` | `a361854` | ✅ | ✅ | `setup_dpcc_projector` slicing obs_normalizer to a hardcoded 6 → 9-D vs 23-D mismatch in `Projector.build_matrices` for non-visual. Slice target now `trajectory_dim - action_dim`. |
| 18.6 record_sim_frame env-render hook | (working tree) | ✅ | ✅ | Genuine 23-D non-visual eval can't reach the visual capture buffer (blocked by 18.3 guard) → no GIFs. Added a render-from-sim hook in `aligning_sim.py` (non-visual branch only) and a `Policy.record_sim_frame(env)` method on both eval scripts. |
| STALE_CONFIG | `b125365` | ✅ | ✅ | `utils.Config.save()` skipped overwriting `model_config.pkl` if file existed → stale config across retraining runs → eval loads wrong-shaped model → shape mismatch crash on state_dict load. Always overwrites now. |

**DPCC-side files touched by 18.3-18.6:**
- `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (18.3, 18.4, 18.5, 18.6)
- `d3il/simulation/aligning_sim.py` (18.6 — one `hasattr`-gated hook line in the non-visual branch)
- `diffuser_visual_aligning/utils/config.py` (STALE_CONFIG)

Visual path **unchanged** at every fix. Every fix's guard condition resolves
to either "no-op on visual" (e.g., `if not _if_vision:`) or "same result on
visual" (e.g., Fix-18.5's `_target_obs_dim = 6` when trajectory_dim is 9).

### Critical reminder for future DPCC work

**Dim invariants (do not break):**
- Visual DPCC trajectory: **9-D** `[act(3) | des_c_pos(3) | c_pos(3)]`. Hardcoded
  via `VisualUNet.TRANSITION_DIM = 9` in the visual branch. Documented in
  `Audit_CheckPoint_Fix7/GEN6V4_AUDIT_REPORT.md:314`.
- Non-visual DPCC trajectory: **23-D** `[act(3) | obs(20)]`. Computed via
  `transition_dim = action_dim + obs_dim` in the non-visual branch.
  `obs_dim = 20` is the UF-17 / Fix-18.1 contract (full state including box
  + target poses).
- Obs normalizer dim is the **authoritative checkpoint identity**: 6 = visual,
  20 = non-visual. **`model_config.pkl` was unreliable before STALE_CONFIG was
  patched** — for any pre-Fix-18 checkpoint, prefer reading the state_dict
  tensor shape over the config metadata.

**Audit script** to print the truth for any checkpoint:
```bash
python -c "
import torch, glob, sys
ckpt = sorted(glob.glob('<checkpoint_dir>/state_*.pt'))[-1]
sd = torch.load(ckpt, map_location='cpu')['model']
k = next(k for k in sd if 'downs.0.0.blocks.0.block.0.weight' in k)
n = sd[k].shape[1]
print(f'first-conv input channels: {n}  → {n}-D model  ({\"visual\" if n == 9 else \"non-visual\" if n == 23 else \"unknown\"})')
"
```

### Re-audit at Fix-18.6 commit time

All six post-Fix-18.1 changes (18.2 through 18.6 + STALE_CONFIG) were
re-examined; each maps to a specific crash that occurred during this
session. **None were reverted.** See the Gen7 canonical
`fix_18_nonvisual_step1/CHANGELOG.md` "Final Post-Fix-18 Audit" section for
the full per-fix justification table.
