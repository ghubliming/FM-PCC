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
├── config/aligning-d3il-visual.py    ← APPEND two new entries (no existing entries modified)
│   ├── 'imf_visual_aligning'         ← training config
│   └── 'plan_imf_visual_aligning'    ← planning/eval config
│
└── Slurm_Codes/sbatch/imf_visual_aligning/   ← NEW (copy-modified from fm_visual_aligning/)
    ├── train_imf_visual_aligning.sh
    ├── eval_imf_visual_aligning.sh
    └── imf_visual_aligning_pipeline.sh       ← submits train → eval (afterok dep)
```

### 2.1 Config strategy — why append to `aligning-d3il-visual.py` (not new file)

**Question raised**: should we mirror Gen9 Ep 2's pattern (new `avoiding-d3il-visual.py` separate from `avoiding-d3il.py`) and create a new `aligning-d3il-visual-imf.py`?

**Answer: No — append to `aligning-d3il-visual.py`**. The two situations differ:

| Situation | What existed before | What was needed | Action |
|---|---|---|---|
| Gen9 Ep 2 (avoiding visual) | `avoiding-d3il.py` — non-visual D3IL only | A *visual* config for a previously *non-visual* task | **New file** `avoiding-d3il-visual.py` (separates visual vs non-visual concerns) |
| **Gen8 Ep 1 (iMF visual aligning)** | `aligning-d3il-visual.py` — already visual-specific, already contains `fm_visual_aligning` + `plan_fm_visual_aligning` | A second engine (iMF) for the *same task, same dimensions, same data path* | **Append** `imf_visual_aligning` + `plan_imf_visual_aligning` |

The shared file is correct because:
- All four entries (`fm_*`, `plan_fm_*`, `imf_*`, `plan_imf_*`) share `obs_dim=6`, `action_dim=3`, `horizon=8`, `if_vision=True`, dataset path, normalizer choice, FiLM shape_meta, projector params.
- Only the diffusion class + a few iMF-specific hyperparams (`u_loss_weight`, `v_loss_weight`, `loss_schedule`, `time_beta_alpha_v3`, `time_beta_beta_v3`) differ.
- `base['imf_visual_aligning'] = { **base['fm_visual_aligning'], ... }` reuses ~95% of the FM entry — keeping them in the same file makes the delta auditable in one view.

The Gen9 split was driven by *task dimension change* (3-D action → 2-D, 6-D obs → 4-D, 9-D traj → 6-D); Gen8 has *no dimension change*, only an engine swap.

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

**Option A (Recommended — chosen)**: Replace `iMFTrajectoryModel.velocity_net` with `VisualUNet` so FiLM conditioning flows through the iMF dual-head architecture. The aux_head stays on raw `x`. This is ~10 lines in `iMFTrajectoryModel.__init__`.

Concrete diff in `imf_visual_aligning/models/imf_trajectory_model.py`:

```python
class iMFTrajectoryModel(nn.Module):
    def __init__(self, transition_dim, dim, dim_mults, h_dim,
                 if_vision=False, shape_meta=None, latent_dim=128,
                 use_aux_head=True, **kwargs):
        super().__init__()
        self.if_vision = if_vision

        if if_vision:
            # Gen8: use VisualUNet (FiLM-conditioned) as velocity net
            from imf_visual_aligning.models.visual_unet import VisualUNet
            self.velocity_net = VisualUNet(
                transition_dim=transition_dim,
                shape_meta=shape_meta,
                latent_dim=latent_dim,
                dim=dim, dim_mults=dim_mults, h_dim=h_dim,
                **kwargs,
            )
        else:
            # Original path (Gen3v4 non-visual)
            self.velocity_net = Flow_matcher_U_Net_v2(
                transition_dim=transition_dim,
                dim=dim, dim_mults=dim_mults, h_dim=h_dim, **kwargs,
            )

        # Aux v-head stays plain (operates on raw x, no FiLM)
        if use_aux_head:
            self.aux_head = Flow_matcher_U_Net_v2(
                transition_dim=transition_dim,
                dim=dim, dim_mults=dim_mults, h_dim=h_dim, **kwargs,
            )
```

Requires `VisualUNet` to accept `h_dim` and forward it into the internal `unet1d_temporal_cond` — Gen7's version already accepts `**kwargs`, so the call site is the only change. Verify by grep'ing `unet1d_temporal_cond` constructor args inside `VisualUNet.__init__`.

**Option B**: Create a new `VisualIMFTrajectoryModel` that fuses both. More boilerplate, cleaner separation. Rejected — duplicates ~120 lines for no semantic gain.

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

### Step 5: Slurm sbatch entries — `Slurm_Codes/sbatch/imf_visual_aligning/`

Copy-modify from `Slurm_Codes/sbatch/fm_visual_aligning/`. Three files:

#### 5.1 `train_imf_visual_aligning.sh` (full-cluster, single-node, 1 GPU, 24h)

Identical to `fm_visual_aligning/train_fm_visual_aligning.sh` except the final python invocation:

```diff
- python fm_visual_aligning_test/train_fm_visual_aligning.py \
+ python imf_visual_aligning_test/train_imf_visual_aligning.py \
      --seeds 6 \
      --use-wandb \
-     --wandb-project FM-PCC-visual-aligning-FM
+     --wandb-project FM-PCC-visual-aligning-iMF
```

Also: `#SBATCH --job-name=train_imf_visual_aligning`.

#### 5.2 `eval_imf_visual_aligning.sh` (eval, 4h)

Same template as `eval_fm_visual_aligning.sh`. Final invocation:

```diff
- python fm_visual_aligning_test/eval_fm_visual_aligning.py $SEED_ARG --record "$RECORD_MODE" --eval-on-train
+ python imf_visual_aligning_test/eval_imf_visual_aligning.py $SEED_ARG --record "$RECORD_MODE" --eval-on-train
```

Plus job-name swap and updated comment block pointing at `logs/aligning-d3il-visual/plans/imf_visual_aligning/...`.

#### 5.3 `imf_visual_aligning_pipeline.sh` (chained train → eval)

Drop-in copy of `fm_visual_aligning_pipeline.sh` with:

```diff
- SBATCH_DIR="Slurm_Codes/sbatch/fm_visual_aligning"
+ SBATCH_DIR="Slurm_Codes/sbatch/imf_visual_aligning"
...
- TRAIN_ID=$(sbatch --parsable $LOG_OPTS "${SBATCH_DIR}/train_fm_visual_aligning.sh")
+ TRAIN_ID=$(sbatch --parsable $LOG_OPTS "${SBATCH_DIR}/train_imf_visual_aligning.sh")
...
- EVAL_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:$TRAIN_ID "${SBATCH_DIR}/eval_fm_visual_aligning.sh")
+ EVAL_ID=$(sbatch --parsable $LOG_OPTS --dependency=afterok:$TRAIN_ID "${SBATCH_DIR}/eval_imf_visual_aligning.sh")
```

And the banner text: `"Launching Visual-iMF (Gen8) Pipeline..."`.

#### 5.4 Slurm-side sanity reuses

All three scripts inherit unchanged:
- `PYTHONPATH` setup including `D3IL_ROOT` and `D3IL_ENV_ROOT`
- `MUJOCO_GL=egl`, `PYOPENGL_PLATFORM=egl`, `MPLBACKEND=agg` for headless render
- W&B key file pickup from `$HOME/FMPCC/.wandb_api_key`
- Latest-log symlink at `Slurm_Codes/logs/latest.log`
- `--partition=gpu-1-student` (override if iMF needs longer than 24h — Gen3v4 trained in ~14h on 5 seeds)

#### 5.5 No changes needed in existing iMF (non-visual) sbatch dir

`Slurm_Codes/sbatch/iMF/` (train_imf.sh, eval_imf.sh, load_results_imf.sh) drives the non-visual `FM_v3_imeanflow_test/` pipeline. Gen8 is visual; these are independent. Do NOT consolidate — the non-visual iMF still has active runs.

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

### 8.1 Docker-side (Phase 0 — pre-cluster, AST + grep only, no Python runtime)

Per the Fix-2 lesson ([Gen9 Ep 2 Fix_2 §8](../../Gen9/Epoch_2_Single_Camera_Avoiding_Pipeline/Fix_2/CHANGELOG.md#8-lesson-for-next-time)): hardcoded integers matching the source task's dimensions are the #1 silent bug source when copy-modifying engine code. Run these BEFORE submitting any cluster job.

- [ ] **AST parses on all new files**:
  ```bash
  python -m py_compile imf_visual_aligning/models/*.py \
      imf_visual_aligning_test/*.py
  ```
- [ ] **Dim-hardcode grep**. Gen8 task dims: `action_dim=3`, `obs_dim=6`, `transition_dim=9`. Hardcodes carried over from Gen3v4 non-visual (`transition_dim=20` or `23`) or from a hypothetical avoiding port (`6` non-visual obs) would surface here:
  ```bash
  grep -nE "= 20 if|= 23 if|transition_dim *= *20|transition_dim *= *23|trajectory_dim - 20|trajectory_dim - 17" \
      imf_visual_aligning imf_visual_aligning_test
  ```
  Expected output: empty. Any hit means a stale Gen3v4 non-visual literal leaked through.
- [ ] **Cross-check `_obs_dim`/`_traj_dim` train-script hardcodes** match config:
  ```bash
  grep -nE "_obs_dim *= *|_traj_dim *= *|transition_dim *= *" \
      imf_visual_aligning_test/train_imf_visual_aligning.py
  ```
  Must show `_obs_dim = 6 if _if_vision else 20` and `transition_dim=9` (or computed `_obs_dim + action_dim`). Confirm `config/aligning-d3il-visual.py['imf_visual_aligning']['obs_dim'] == 6`.
- [ ] **Import chain check** — every new file resolves without ImportError:
  ```bash
  grep -rn "from fm_visual_aligning\|from flow_matcher_v3_imeanflow" \
      imf_visual_aligning imf_visual_aligning_test
  ```
  Expected: empty. Any cross-package import means an import-fixup was missed (Fix-1 of Gen9 Ep 2 was exactly this class of bug).
- [ ] **`datasets/__init__.py`** re-exports are correct names (lesson from Gen9 Ep 2 Fix-1):
  ```bash
  grep -n "ParityAligningDataset\|StateOnlyAligningDataset" \
      imf_visual_aligning/datasets/__init__.py
  ```
  Must match the actual class names exported by `imf_visual_aligning/datasets/sequence.py`.
- [ ] **Slurm sbatch script-name sanity**:
  ```bash
  grep -E "fm_visual_aligning|imf_visual_aligning" \
      Slurm_Codes/sbatch/imf_visual_aligning/*.sh
  ```
  Every line referencing a python path should say `imf_visual_aligning`, not `fm_visual_aligning`. Job-name lines should say `imf_*`.

### 8.2 Cluster-side (Phase 1 — first real submit)

- [ ] `imf_visual_aligning/` installs cleanly (`pip install -e .`)
- [ ] `python -c "from imf_visual_aligning.models.visual_imf_diffusion import VisualIMF"` succeeds
- [ ] Training launches: `sbatch Slurm_Codes/sbatch/imf_visual_aligning/train_imf_visual_aligning.sh`
- [ ] First step does NOT crash with `RuntimeError: The size of tensor a (X) must match the size of tensor b (Y) at non-singleton dimension 2` — if it does, dim-hardcode regression (see Fix-2 of Gen9 Ep 2).
- [ ] 9-D trajectory tensor shape at first forward pass: `(B, 8, 9)`
- [ ] h-conditioning active: verify `h_mlp` contributes to time embedding in forward (printf the embedding shape once)
- [ ] Loss reports both `diffusion_loss` (u-head) and `aux_loss` (v-head)
- [ ] W&B run created under project `FM-PCC-visual-aligning-iMF`

### 8.3 Eval (Phase 2 — afterok or manual)

- [ ] Eval loads checkpoint and runs 5-context rollout with PCC projection
- [ ] Visual FiLM conditioning produces GIFs (bp_cam + inhand_cam)
- [ ] Eval-side dim hardcodes (Gen9 Ep 2 Fix-2 §7 warning class) — verify these are correct for 9-D aligning:
  - `_target_obs_dim = trajectory_dim - 3` → correct for aligning (action_dim=3)
  - `pad = trajectory_dim - 9` → correct for aligning (traj=9)
  - `'x': 6, 'y': 7, 'z': 8` indices → correct for aligning (c_pos at 6:9)

  These were Gen7's hardcodes; copying them over to Gen8 is correct because dimensions match. No Fix-3-equivalent is needed for Gen8.

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
| Config additions in `config/aligning-d3il-visual.py` | ~40 | Appended to existing |
| `Slurm_Codes/sbatch/imf_visual_aligning/train_imf_visual_aligning.sh` | ~60 | Modified from Gen7 |
| `Slurm_Codes/sbatch/imf_visual_aligning/eval_imf_visual_aligning.sh` | ~80 | Modified from Gen7 |
| `Slurm_Codes/sbatch/imf_visual_aligning/imf_visual_aligning_pipeline.sh` | ~55 | Modified from Gen7 |

**Total truly-new code**: ~60 lines (`visual_imf_diffusion.py`) + ~10 lines (`iMFTrajectoryModel` Visual branch) + ~40 lines (config entries) ≈ **~110 lines net new logic**. **Total copied + import-fixed**: ~5700 lines.

---

## 10. Risk Assessment

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| VisualUNet ↔ iMFTrajectoryModel wiring conflict | Medium | Option A (§3 Step 2): replace velocity_net inside iMFTrajectoryModel with VisualUNet — concrete diff provided |
| h_mlp missing from copied U-Net | Low | Explicit verification step (§8.2 checklist) — uses Gen3v4's `unet1d_temporal_cond.py` which already has `h_mlp` |
| Stale `model_config.pkl` (STALE_CONFIG bug) | Low | Already fixed in Gen7 Fix-18 side-patch (Config.save always overwrites) |
| Training instability (E4 spike from Gen3v4) | Medium | Apply recommended guardrails: `h_min=1e-3`, `max_grad_norm=1.0`, `lr=2e-4` |
| **Stale dim hardcodes from source repo** (Gen9 Ep 2 Fix-2 class) | **Medium** | **Phase 0 grep step (§8.1) — run before any cluster submit. Aligning dims match Gen7 so risk is lower than the avoiding port, but `_obs_dim` and `_traj_dim` hardcodes still need explicit cross-check.** |
| Stale `__init__.py` re-exports (Gen9 Ep 2 Fix-1 class) | Low | Phase 0 grep step (§8.1) checks `datasets/__init__.py` matches actual exports |
| Slurm sbatch script-name mismatch (forgot `sed` `fm` → `imf`) | Low | Phase 0 sbatch grep step (§8.1) flags any `fm_visual_aligning` literal still present |
| iMF `h_min` cliff at low-step inference | Low | Inherit Gen3v4 fix_3 default `h_min=1e-3` |

---

## 11. Phase Ordering (executable)

Strict ordering — earlier phases gate later ones. All Docker-side work runs without a Python interpreter (per env policy).

| Phase | Where | What | Gate to next |
|---|---|---|---|
| **0** | Docker | Create all files (§3 Steps 1–4, §3.5). Run §8.1 AST + grep checks. | All §8.1 boxes green. |
| **1** | Docker | Commit (only when user explicitly says so per memory policy) + `git push` to cluster sync. | Push lands on cluster. |
| **2** | Cluster | `pip install -e .` for `imf_visual_aligning`. Submit `train_imf_visual_aligning.sh`. | First 100 steps without dim-broadcast crash (§8.2). |
| **3** | Cluster | Let training run to ~3 epochs. Monitor W&B for E4-style spike. | u-loss and v-loss both trending down. |
| **4** | Cluster | After train completes (or hits time limit), pipeline auto-fires eval (afterok dep). | Eval produces GIFs + DPCC projection logs. |
| **5** | Both | Write CHANGELOG under `logs_in_develop/Gen8/Epoch_1/` summarizing the run. | — |

If Phase 2 crashes: dim-hardcode regression most likely; consult Gen9 Ep 2 Fix-2 CHANGELOG, run §8.1 grep with stricter patterns, then submit Fix-1 of Gen8.

If Phase 3 shows E4 spike (loss jumps by >5× around epoch 4): drop `lr` to `1e-4`, set `max_grad_norm=0.5`, restart from epoch-3 checkpoint. This is the *known* iMF failure mode from Gen3v4 — Gen8 inherits the same engine, so the same fix applies.

---

## Cross-References

| Document | Content |
|----------|---------|
| [Gen3v4 fix_3 AUDIT_REPORT](../../Gen3v4_imf/Gen3v4u2_Major_Upgrade_direct/fix_3/AUDIT_REPORT.md) | iMF correctness audit vs reference repo |
| [Gen7 DIM_AUDIT_VS_D3IL](../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_18_nonvisual_step1/DIM_AUDIT_VS_D3IL.md) | 9-D/23-D trajectory dimension verification |
| [Gen7 Fix-18 CHANGELOG](../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_18_nonvisual_step1/CHANGELOG.md) | Non-visual path fixes (18.1–18.6.2) |
| [Gen9 Ep 2 Fix-1 CHANGELOG](../../Gen9/Epoch_2_Single_Camera_Avoiding_Pipeline/Fix_1/CHANGELOG.md) | Stale `__init__.py` re-export class of bug (referenced by §8.1) |
| [Gen9 Ep 2 Fix-2 CHANGELOG](../../Gen9/Epoch_2_Single_Camera_Avoiding_Pipeline/Fix_2/CHANGELOG.md) | Dim-hardcode class of bug + the grep-recipe lesson reused in §8.1 |
| [iMF Paper](https://arxiv.org/abs/2502.13129) | Kaiming He et al. — theoretical foundation |
| [Reference iMF repo](/workspaces/imeanflow/) | Official PyTorch implementation |
