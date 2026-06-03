# Gen8 — Epoch 1: iMeanFlow Visual Aligning

**Date**: 2026-06-03
**Parent**: Gen7 (FM Visual Aligning) + Gen3v4 (iMF engine, post-fix_3)
**Task**: D3IL Aligning with iMF engine + PCC projection + visual FiLM conditioning
**Principle**: New modules only — Gen7 code is NOT touched.

---

## 1. Motivation

Gen7 (`fm_visual_aligning/`) uses a vanilla Flow Matching ODE engine (`FlowMatchingODE`) for the visual aligning task. Gen3v4 (`flow_matcher_v3_imeanflow/`) implements the iMeanFlow engine (`iMeanFlowODE`) with mean-flow training + dual u/v heads, audited and verified correct against the reference iMF repository (see [AUDIT_REPORT](../../Gen3v4_imf/Gen3v4u2_Major_Upgrade_direct/fix_3/AUDIT_REPORT.md)).

**Gen8 swaps the FM engine → iMF engine** inside the Gen7 visual aligning architecture, preserving:
- 9-D visual trajectory `[act(3) | des_c_pos(3) | c_pos(3)]` (verified correct per [DIM_AUDIT](../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_18_nonvisual_step1/DIM_AUDIT_VS_D3IL.md))
- FiLM-conditioned VisualUNet backbone (bp_cam + inhand_cam → 128-D latent)
- DPCC/SLSQP projection in the sampling loop
- All Fix-18 non-visual path fixes ([CHANGELOG](../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_18_nonvisual_step1/CHANGELOG.md))

### Why iMF over vanilla FM?

| Property | Vanilla FM (Gen7) | iMF (Gen8) |
|----------|-------------------|------------|
| Training target | `v = x_data − noise` (instantaneous velocity) | `u = (x_t − x_r)/h` (mean-flow velocity over `[r, t]`) |
| Theoretical NFE | Needs many steps (curved velocity field) | One-step valid by construction (mean-flow is constant for linear interpolant) |
| h-conditioning | None | Model learns step-size `h`, enabling adaptive-step inference |
| Aux head | None | v-head provides training-time gradient regularization |
| Inference | u-only (post fix_3 audit) | u-only (Deviation A resolved) |

---

## 2. Architecture Overview

```
Gen8 iMF Visual Aligning
├── imf_visual_aligning/              ← NEW core folder (copy-modified from fm_visual_aligning/)
│   ├── __init__.py
│   ├── setup.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── helpers.py                ← from fm_visual_aligning (unchanged)
│   │   ├── visual_unet.py           ← from fm_visual_aligning (unchanged — VisualUNet)
│   │   ├── unet1d_temporal_cond.py   ← from flow_matcher_v3_imeanflow (h-conditioning enabled)
│   │   ├── imf_diffusion.py          ← from flow_matcher_v3_imeanflow (iMeanFlowODE wrapper)
│   │   ├── imf_engine.py             ← from flow_matcher_v3_imeanflow (iMeanFlowEngine)
│   │   ├── imf_trajectory_model.py   ← from flow_matcher_v3_imeanflow (iMFTrajectoryModel)
│   │   ├── imf_losses.py             ← from flow_matcher_v3_imeanflow (unchanged)
│   │   ├── visual_imf_diffusion.py   ← NEW: VisualIMF (extends iMeanFlowODE)
│   │   └── diffusion.py              ← from flow_matcher_v3_imeanflow (base, needed by imf_diffusion)
│   ├── sampling/
│   │   ├── __init__.py               ← from fm_visual_aligning
│   │   └── projection.py             ← from fm_visual_aligning (unchanged — DPCC projector)
│   ├── datasets/                     ← from fm_visual_aligning (unchanged — ParityAligningDataset)
│   └── utils/                        ← from fm_visual_aligning (unchanged — Config, Trainer, etc.)
│
├── imf_visual_aligning_test/         ← NEW entry folder (copy-modified from fm_visual_aligning_test/)
│   ├── train_imf_visual_aligning.py  ← modified: imports from imf_visual_aligning, instantiates iMF
│   └── eval_imf_visual_aligning.py   ← modified: imports from imf_visual_aligning, loads iMF checkpoint
│
└── config/aligning-d3il-visual.py    ← APPEND two new entries (no existing entries modified)
    ├── 'imf_visual_aligning'         ← training config
    └── 'plan_imf_visual_aligning'    ← planning/eval config
```

---

## 3. Implementation Steps

### Step 1: Create `imf_visual_aligning/` (core folder)

Copy entire `fm_visual_aligning/` as the base, then layer iMF-specific files on top.

| Action | Source | Target | Modifications |
|--------|--------|--------|---------------|
| Copy | `fm_visual_aligning/` | `imf_visual_aligning/` | Base scaffold |
| Replace | `fm_visual_aligning/models/diffusion.py` | `imf_visual_aligning/models/diffusion.py` | Copy from `flow_matcher_v3_imeanflow/models/diffusion.py` (base ODE needed by iMF) |
| Add | `flow_matcher_v3_imeanflow/models/imf_diffusion.py` | `imf_visual_aligning/models/imf_diffusion.py` | Fix imports to `imf_visual_aligning.models.*` |
| Add | `flow_matcher_v3_imeanflow/models/imf_engine.py` | `imf_visual_aligning/models/imf_engine.py` | Fix imports |
| Add | `flow_matcher_v3_imeanflow/models/imf_trajectory_model.py` | `imf_visual_aligning/models/imf_trajectory_model.py` | Fix imports |
| Add | `flow_matcher_v3_imeanflow/models/imf_losses.py` | `imf_visual_aligning/models/imf_losses.py` | Fix imports |
| Replace | `flow_matcher_v3_imeanflow/models/unet1d_temporal_cond.py` | `imf_visual_aligning/models/unet1d_temporal_cond.py` | This version has `h_mlp` (h-conditioning); Gen7's does not |
| Create | — | `imf_visual_aligning/models/visual_imf_diffusion.py` | NEW file (see §3.1) |
| Update | — | `imf_visual_aligning/models/__init__.py` | Export new iMF classes |

#### 3.1 `visual_imf_diffusion.py` — The Key New File

This mirrors `visual_gaussian_diffusion.py` (Gen7) but wraps `iMeanFlowODE` instead of `FlowMatchingODE`:

```python
import torch
from imf_visual_aligning.models.imf_diffusion import iMeanFlowODE
from imf_visual_aligning.models.helpers import apply_conditioning


class VisualIMF(iMeanFlowODE):
    """
    iMF engine for Visual-PCC (Gen8).

    Extends iMeanFlowODE with:
    - Visual loss(trajectories, conditions) — FiLM image conditioning
    - Visual forward() for closed-loop inference
    - All iMF features: h-conditioning, u/v dual heads, mean-flow target

    Trajectory: 9D = [act(0:3) | des_c_pos(3:6) | c_pos(6:9)]
    """

    def loss(self, trajectories, conditions):
        """Training entry: unpack visual conditions, delegate to iMF p_losses."""
        if not self.model.if_vision:
            # Non-visual path (Fix-18 compatible)
            x = trajectories
            batch_size = len(x)
            alpha = torch.tensor(self.time_beta_alpha_v3, device=x.device)
            beta = torch.tensor(self.time_beta_beta_v3, device=x.device)
            t = 1.0 - torch.distributions.Beta(alpha, beta).sample((batch_size,))
            return self.p_losses(x, conditions, t)

        # Visual path — extract FiLM inputs
        primary_img = conditions['primary_img'].unsqueeze(1)
        wrist_img   = conditions['wrist_img'].unsqueeze(1)
        obs_0       = conditions[0]
        obs_seq     = trajectories[..., self.action_dim:]

        cond = {
            'visual': (primary_img, wrist_img, obs_seq),
            0: obs_0,
        }

        x = trajectories
        batch_size = len(x)
        alpha = torch.tensor(self.time_beta_alpha_v3, device=x.device)
        beta = torch.tensor(self.time_beta_beta_v3, device=x.device)
        t = 1.0 - torch.distributions.Beta(alpha, beta).sample((batch_size,))
        return self.p_losses(x, cond, t)

    def forward(self, cond, *args, **kwargs):
        """Closed-loop inference: unpack visual tuple → iMF sampling."""
        if isinstance(cond, dict) and 0 in cond and isinstance(cond[0], tuple):
            bp_imgs, inhand_imgs, obs_seq = cond[0]
            snap_obs = obs_seq[:, -1]
            new_cond = {
                0:        snap_obs,
                'visual': (bp_imgs, inhand_imgs, obs_seq),
            }
        else:
            new_cond = cond
        return super().forward(new_cond, *args, **kwargs)
```

The `p_losses` method is inherited from `iMeanFlowODE` and already implements the iMF mean-flow training target `(x_t − x_r)/h` with dual u/v heads. The visual wrapper only handles condition unpacking — identical pattern to Gen7's `VisualFlowMatching` wrapping `FlowMatchingODE`.

#### 3.2 Import Fixups

All copied iMF files use relative imports (`.imf_engine`, `.helpers`, etc.) so no changes are needed when they stay in the same `models/` directory. The only absolute import change is in `visual_imf_diffusion.py` which references `imf_visual_aligning.models.*`.

### Step 2: Create `imf_visual_aligning_test/` (entry folder)

Copy from `fm_visual_aligning_test/`, then modify:

| File | Key Changes |
|------|-------------|
| `train_imf_visual_aligning.py` | Change all `fm_visual_aligning` imports → `imf_visual_aligning`; change diffusion class to `VisualIMF`; change model instantiation to build iMF engine chain |
| `eval_imf_visual_aligning.py` | Same import swap; load iMF checkpoint; `_predict_velocity` uses u-only (inherited from fix_3 `iMeanFlowODE`) |

#### Train script model instantiation diff

```diff
# OLD (Gen7 FM):
-from fm_visual_aligning.models.visual_gaussian_diffusion import VisualFlowMatching
-from fm_visual_aligning.models.visual_unet import VisualUNet
+from imf_visual_aligning.models.visual_imf_diffusion import VisualIMF
+from imf_visual_aligning.models.visual_unet import VisualUNet
+from imf_visual_aligning.models.imf_engine import iMeanFlowEngine
+from imf_visual_aligning.models.imf_trajectory_model import iMFTrajectoryModel
```

**Critical design decision**: Gen7's `VisualUNet` wraps `Flow_matcher_U_Net_v2` with `MultiImageObsEncoder` for FiLM. The iMF's `iMFTrajectoryModel` also wraps `Flow_matcher_U_Net_v2`. For Gen8:

**Option A (Recommended)**: Replace `iMFTrajectoryModel.velocity_net` with `VisualUNet` so FiLM conditioning flows through the iMF dual-head architecture. The aux_head stays on raw `x`. This is ~10 lines in `iMFTrajectoryModel.__init__`.

**Option B**: Create a new `VisualIMFTrajectoryModel` that fuses both. More boilerplate, cleaner separation.

### Step 3: Config Entries in `config/aligning-d3il-visual.py`

Append two new entries at the end of the file (no existing entries modified):

```python
# ─── Gen8 iMF Visual Aligning ────────────────────────────────────────────────

base['imf_visual_aligning'] = {
    **base['fm_visual_aligning'],
    'model': 'imf_visual_aligning.models.visual_unet.VisualUNet',
    'diffusion': 'imf_visual_aligning.models.visual_imf_diffusion.VisualIMF',
    'prefix': 'imf_visual_aligning/',
    'exp_name': watch(args_to_watch_fm_visual_train),
    # iMF-specific params
    'u_loss_weight': 1.0,
    'v_loss_weight': 0.1,
    'loss_schedule': 'balanced',
}

base['plan_imf_visual_aligning'] = {
    **base['plan_fm_visual_aligning'],
    'diffusion': 'imf_visual_aligning.models.visual_imf_diffusion.VisualIMF',
    'prefix': (
        'f:plans/imf_visual_aligning/'
        'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}'
        '_aw{action_weight}_V{if_vision}_steps{max_path_length}_bs{train_batch_size}/'
    ),
    'diffusion_loadpath': (
        'f:imf_visual_aligning/'
        'H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}'
        '_aw{action_weight}_V{if_vision}_steps{max_path_length}_bs{train_batch_size}'
    ),
}
```

### Step 4: `setup.py`

```python
from setuptools import setup, find_packages
setup(name='imf_visual_aligning', packages=find_packages())
```

---

## 4. iMF Audit Compliance (from Gen3v4 fix_3)

All fixes from the [AUDIT_REPORT](../../Gen3v4_imf/Gen3v4u2_Major_Upgrade_direct/fix_3/AUDIT_REPORT.md) are inherited by copying `imf_diffusion.py`:

| Fix | Status in Gen8 |
|-----|----------------|
| fix_1: Training target `(x_t − x_r)/h` | ✅ Inherited from `iMeanFlowODE.p_losses` |
| fix_3 Deviation A: u-only inference | ✅ Inherited from `iMeanFlowODE._predict_velocity` (line 131: `return velocity`) |
| fix_3 Deviation B: frozen `t=0.5` at inference | ✅ Inherited from `iMeanFlowODE.p_sample_loop` (line 187: `T_CONST_INFERENCE = 0.5`) |
| h-conditioning plumbing | ✅ Inherited from `unet1d_temporal_cond.py` (h_mlp) |
| σ=1.0 noise | ✅ Inherited from `iMeanFlowODE.p_sample_loop` (`torch.randn`) |

---

## 5. Fix-18 Compliance (from Gen7)

All non-visual path fixes from [CHANGELOG](../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_18_nonvisual_step1/CHANGELOG.md) are inherited by copying the train/eval scripts:

| Fix | Status in Gen8 |
|-----|----------------|
| 18.1 (obs_dim override for non-visual) | ✅ Copied from train script |
| 18.2 (_traj_dim from normalizer) | ✅ Copied from eval script |
| 18.3 (UF-13 normalizer-dim guard) | ✅ Copied from eval script |
| 18.4 (DIAG var alias) | ✅ Copied from eval script |
| 18.5 (projector slice) | ✅ Copied from eval script |
| 18.6.2 (capture_frame for non-visual GIFs) | ✅ Copied from eval script + aligning_sim |

---

## 6. Dimension Audit (from Gen7 DIM_AUDIT)

Per [DIM_AUDIT_VS_D3IL](../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_18_nonvisual_step1/DIM_AUDIT_VS_D3IL.md):

| Variant | Trajectory Dim | Source |
|---------|---------------|--------|
| Visual | **9-D** `[act(3) \| des_c_pos(3) \| c_pos(3)]` | DPCC architecture (not D3IL) |
| Non-visual | **23-D** `[act(3) \| obs(20)]` | UF-17 extension |
| D3IL vision baseline | 3-D obs + 128-D latent (no joint trajectory) | Different paradigm |

Gen8 inherits these dimensions unchanged. The iMF engine is dimension-agnostic — it operates on `transition_dim = action_dim + obs_dim`.

---

## 7. Key Difference: `unet1d_temporal_cond.py`

Gen7's `unet1d_temporal_cond.py` does NOT have `h_mlp` (h-conditioning). The iMF version from `flow_matcher_v3_imeanflow/` DOES. Gen8 **must** use the iMF version.

```
Gen7 (fm_visual_aligning):     time_mlp(t) only
Gen3v4 iMF:                    time_mlp(t) + h_mlp(h)   ← Gen8 uses this
```

This is the single most critical file replacement — without `h_mlp`, the model cannot learn the mean-flow objective.

---

## 8. Verification Checklist

- [ ] `imf_visual_aligning/` installs cleanly (`pip install -e .`)
- [ ] `python -c "from imf_visual_aligning.models.visual_imf_diffusion import VisualIMF"` succeeds
- [ ] Training launches: `python imf_visual_aligning_test/train_imf_visual_aligning.py --config imf_visual_aligning`
- [ ] 9-D trajectory tensor shape at first forward pass: `(B, 8, 9)`
- [ ] h-conditioning active: verify `h_mlp` contributes to time embedding in forward
- [ ] Loss reports both `diffusion_loss` (u-head) and `aux_loss` (v-head)
- [ ] Eval loads checkpoint and runs 5-context rollout with PCC projection
- [ ] Visual FiLM conditioning produces GIFs (bp_cam + inhand_cam)

---

## 9. File Inventory (Expected)

| New File | Lines (est.) | Source |
|----------|-------------|--------|
| `imf_visual_aligning/models/visual_imf_diffusion.py` | ~60 | **New** |
| `imf_visual_aligning/models/imf_diffusion.py` | ~378 | Copy from Gen3v4 |
| `imf_visual_aligning/models/imf_engine.py` | ~164 | Copy from Gen3v4 |
| `imf_visual_aligning/models/imf_trajectory_model.py` | ~142 | Copy from Gen3v4 |
| `imf_visual_aligning/models/imf_losses.py` | ~80 | Copy from Gen3v4 |
| `imf_visual_aligning/models/unet1d_temporal_cond.py` | ~400 | Copy from Gen3v4 (has h_mlp) |
| `imf_visual_aligning/models/diffusion.py` | ~400 | Copy from Gen3v4 |
| `imf_visual_aligning/models/helpers.py` | ~280 | Copy from Gen7 |
| `imf_visual_aligning/models/visual_unet.py` | ~150 | Copy from Gen7 |
| `imf_visual_aligning_test/train_imf_visual_aligning.py` | ~350 | Modified from Gen7 |
| `imf_visual_aligning_test/eval_imf_visual_aligning.py` | ~3200 | Modified from Gen7 |
| Config additions | ~40 | Appended to existing |

**Total new code**: ~60 lines. **Total copied + import-fixed**: ~5500 lines.

---

## 10. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| VisualUNet ↔ iMFTrajectoryModel wiring conflict | Medium | Option A (§3 Step 2): replace velocity_net inside iMFTrajectoryModel with VisualUNet |
| h_mlp missing from copied U-Net | Low | Explicit verification step (§8 checklist) |
| Stale `model_config.pkl` (STALE_CONFIG bug) | Low | Already fixed in Gen7 Fix-18 side-patch (Config.save always overwrites) |
| Training instability (E4 spike from Gen3v4) | Medium | Apply recommended guardrails: `h_min=1e-3`, `max_grad_norm=1.0`, `lr=2e-4` |

---

## Cross-References

| Document | Content |
|----------|---------|
| [Gen3v4 fix_3 AUDIT_REPORT](../../Gen3v4_imf/Gen3v4u2_Major_Upgrade_direct/fix_3/AUDIT_REPORT.md) | iMF correctness audit vs reference repo |
| [Gen7 DIM_AUDIT_VS_D3IL](../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_18_nonvisual_step1/DIM_AUDIT_VS_D3IL.md) | 9-D/23-D trajectory dimension verification |
| [Gen7 Fix-18 CHANGELOG](../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_18_nonvisual_step1/CHANGELOG.md) | Non-visual path fixes (18.1–18.6.2) |
| [iMF Paper](https://arxiv.org/abs/2502.13129) | Kaiming He et al. — theoretical foundation |
| [Reference iMF repo](/workspaces/imeanflow/) | Official PyTorch implementation |
