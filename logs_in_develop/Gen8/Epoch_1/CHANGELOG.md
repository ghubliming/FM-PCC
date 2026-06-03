# Gen8 Epoch 1 — Visual-iMF Aligning: Initial Implementation

**Date**: 2026-06-03
**Status**: ✅ Implementation complete (uncommitted). Cluster smoke pending.
**Parent plan**: [`PLAN.md`](./PLAN.md)
**Lineage**: Gen7 (FM visual aligning) × Gen3v4 fix_3 (iMF engine, audited correct)

---

## 1. What was built

Gen8 wires the Gen3v4 iMeanFlow engine (mean-flow target, h-conditioning, u/v dual heads) into the Gen7 visual aligning architecture (9D trajectory, FiLM dual-cam ResNet, DPCC/SLSQP projector). The task, data path, and all trajectory dimensions are unchanged — only the diffusion engine is swapped.

---

## 2. Files created / modified

### 2.1 New package — `imf_visual_aligning/`

Scaffolded from `fm_visual_aligning/` then layered with iMF-specific files. All `fm_visual_aligning.*` imports rewritten to `imf_visual_aligning.*`.

| File | Status | Notes |
|---|---|---|
| `imf_visual_aligning/setup.py` | modified | `name='imf_visual_aligning'` |
| `imf_visual_aligning/models/__init__.py` | modified | Exports all 8 public classes |
| `imf_visual_aligning/models/visual_imf_diffusion.py` | **new** | `VisualIMF(iMeanFlowODE)` — the key new file (~70 lines) |
| `imf_visual_aligning/models/visual_unet.py` | modified | Added `h=None` to `forward()`, threaded to `self.backbone(..., h=h)` so iMF h-conditioning reaches `h_mlp` |
| `imf_visual_aligning/models/imf_trajectory_model.py` | modified | Added `if_vision=False, vis_config=None`; visual branch uses `VisualUNet(vis_config)` as `velocity_net` |
| `imf_visual_aligning/models/imf_engine.py` | modified | Added `if_vision=False, vis_config=None`; stores `self.if_vision`; threads to `iMFTrajectoryModel` |
| `imf_visual_aligning/models/imf_diffusion.py` | copied | From `flow_matcher_v3_imeanflow/` — iMF p_losses, p_sample_loop (all Gen3v4 fix_3 fixes inherited) |
| `imf_visual_aligning/models/imf_engine.py` | copied + modified | iMeanFlowEngine |
| `imf_visual_aligning/models/imf_losses.py` | copied | Unchanged from Gen3v4 |
| `imf_visual_aligning/models/diffusion.py` | copied | Base ODE from Gen3v4 (not Gen7) |
| `imf_visual_aligning/models/unet1d_temporal_cond.py` | copied | **iMF version** from Gen3v4 — has `h_mlp` (Gen7 did not) |
| `imf_visual_aligning/datasets/` | copied | `ParityAligningDataset`, `StateOnlyAligningDataset`, `LimitsNormalizer` — unchanged |
| `imf_visual_aligning/sampling/` | copied | `Projector` — unchanged |
| `imf_visual_aligning/utils/` | copied | `Config`, `Trainer`, etc — unchanged |

### 2.2 New entry folder — `imf_visual_aligning_test/`

| File | Status | Key changes from Gen7 |
|---|---|---|
| `train_imf_visual_aligning.py` | modified | Header, imports, `experiment='imf_visual_aligning'`, W&B project, §2 builds `iMeanFlowEngine(if_vision=True, vis_config=args)`, §3 builds `VisualIMF` with `u_loss_weight`, `v_loss_weight`, `loss_schedule` |
| `eval_imf_visual_aligning.py` | modified | Header, imports, `experiment='plan_imf_visual_aligning'` |

### 2.3 Config — `config/aligning-d3il-visual.py` (appended)

Two new entries at end of file:

```python
base['imf_visual_aligning']      = { **base['fm_visual_aligning'], model/diffusion/prefix/iMF-params }
base['plan_imf_visual_aligning'] = { **base['plan_fm_visual_aligning'], diffusion/prefix/diffusion_loadpath }
```

No existing entries modified.

### 2.4 Slurm — `Slurm_Codes/sbatch/imf_visual_aligning/` (new dir)

| File | Notes |
|---|---|
| `train_imf_visual_aligning.sh` | 24h, 1 GPU, W&B `FM-PCC-visual-aligning-iMF` |
| `eval_imf_visual_aligning.sh` | 4h, supports `$1=seed $2=record_mode`, afterok-ready |
| `imf_visual_aligning_pipeline.sh` | Chains train→eval with `afterok` dep, unified log timestamp |

---

## 3. Architecture decisions

### 3.1 Option A — `iMFTrajectoryModel.velocity_net = VisualUNet` (chosen)

Per PLAN.md §3 Step 2 Option A:

```
iMeanFlowODE.model     → iMeanFlowEngine
iMeanFlowEngine.model  → iMFTrajectoryModel(if_vision=True, vis_config=args)
iMFTrajectoryModel.velocity_net → VisualUNet(args)    ← FiLM dual-cam ResNet + h-conditioned U-Net
iMFTrajectoryModel.aux_head     → nn.Sequential(Linear, SiLU, Linear) on raw x (no images)
```

The `aux_head` stays plain (9→9 linear over last dim of `(B, H, 9)`) — it receives raw x without image processing, exactly as in Gen3v4 non-visual.

### 3.2 h-conditioning threaded end-to-end

`VisualUNet.forward` now accepts `h=None` and passes it to `self.backbone(..., h=h)`. The backbone is `UNet1DTemporalCondModel` from Gen3v4 (has `h_mlp`). Chain:

```
iMeanFlowODE._predict_velocity(x, cond, t_const, h=h_batch)
  → iMeanFlowEngine.forward_train(x, t, h=h, cond=cond)
  → iMFTrajectoryModel.forward(x, t, h=h, cond=cond)
  → velocity_net(x, cond, t, h=h)          # VisualUNet.forward
  → self.backbone(x, visual_cond, t, h=h)  # Flow_matcher_U_Net_v2.forward
  → t_embed = time_mlp(t) + h_mlp(h)       # h actually reaches the MLP
```

Without this thread, h would silently be ignored by VisualUNet even though `h_mlp` exists in the backbone — the single most important wiring fix.

### 3.3 Config strategy — append, not new file

`aligning-d3il-visual.py` already contains Gen6-Gen7 visual entries. Gen8 reuses ALL shared fields via `**base['fm_visual_aligning']` / `**base['plan_fm_visual_aligning']` — only diffusion class, prefix, and iMF-specific weights differ. Creating a new file would duplicate ~100 lines for no benefit. (Contrast with Gen9 Ep2's `avoiding-d3il-visual.py` which was a new file because task dims changed.)

---

## 4. iMF audit compliance (inherited)

All Gen3v4 fix_3 corrections are inherited via `imf_diffusion.py` copy:

| Fix | Status in Gen8 |
|---|---|
| fix_1: training target `(x_t − x_r)/h` | ✅ Inherited from `iMeanFlowODE.p_losses` |
| fix_3 Deviation A: u-only inference | ✅ `iMeanFlowODE._predict_velocity` returns `velocity` only |
| fix_3 Deviation B: frozen `t=0.5` at inference | ✅ `T_CONST_INFERENCE = 0.5` |
| h-conditioning plumbing | ✅ `h_mlp` in `unet1d_temporal_cond.py` + new `h` thread in `VisualUNet.forward` |
| σ=1.0 noise | ✅ `torch.randn` in `p_sample_loop` |

---

## 5. Fix-18 compliance (inherited)

All Gen7 non-visual path fixes are inherited via train/eval script copy:

| Fix | Status in Gen8 |
|---|---|
| 18.1 `_obs_dim` override for non-visual | ✅ `_obs_dim = 6 if _if_vision else 20` present in train script |
| 18.2 `_traj_dim` from normalizer | ✅ Copied from eval script |
| 18.3 UF-13 normalizer-dim guard | ✅ Copied from eval script |
| 18.4 DIAG var alias | ✅ Copied from eval script |
| 18.5 projector slice | ✅ Copied from eval script |
| 18.6.2 `capture_frame` non-visual GIFs | ✅ Copied from eval script |

---

## 6. Phase-0 smoke check results (Docker, pre-cluster)

| Check | Result |
|---|---|
| AST parse on all 7 new/modified `.py` files | ✅ All pass |
| Stale non-visual dim literals (`= 20 if`, `= 23 if`, `transition_dim=20`) | ✅ None found |
| Cross-package imports (`from fm_visual_aligning.*`, `from flow_matcher_v3_imeanflow.*`) | ✅ None (comments only) |
| `_obs_dim = 6 if _if_vision else 20` correct | ✅ Confirmed |
| `datasets/__init__.py` re-exports match actual class names | ✅ `ParityAligningDataset`, `StateOnlyAligningDataset` |
| Slurm scripts: no `fm_visual_aligning` in python invocation lines | ✅ All `imf_visual_aligning_test/...` |

---

## 7. Known pre-existing eval-side hardcodes (not fixed here)

These are Gen7 hardcodes carried into Gen8 eval script. For aligning (9-D traj, action_dim=3) they are **correct** — unlike Gen9 Ep2's avoiding port which needed Fix-3. No fixing needed.

| File:Line | Hardcode | Aligning-correct? |
|---|---|---|
| `eval_imf_visual_aligning.py` | `_target_obs_dim = trajectory_dim - 3` | ✅ Yes (action_dim=3) |
| `eval_imf_visual_aligning.py` | `pad = trajectory_dim - 9` | ✅ Yes (traj=9) |
| `eval_imf_visual_aligning.py` | `'x': 6, 'y': 7, 'z': 8` (c_pos indices) | ✅ Yes (aligning 9-D layout) |

---

## 8. Next steps

1. **Cluster**: `pip install -e .` in `imf_visual_aligning/`, then `sbatch Slurm_Codes/sbatch/imf_visual_aligning/train_imf_visual_aligning.sh`
2. **First 100 steps**: watch for tensor-broadcast crash — if `tensor a (X) != tensor b (Y)` at dim 2, it's a dim-hardcode regression; consult §4 of Fix-2/CHANGELOG in Gen9 Ep 2.
3. **E4 spike guard**: if loss jumps >5× around epoch 4, drop `learning_rate` to `1e-4`, `max_grad_norm=0.5`, restart from epoch-3 checkpoint (same pattern as Gen3v4).
4. **Eval**: pipeline auto-fires after training (`afterok` dep in pipeline script).
5. **Commit**: only when user explicitly requests (memory policy).

---

## 9. Cross-references

| Document | Content |
|---|---|
| [`PLAN.md`](./PLAN.md) | Full design rationale and phase ordering |
| [`../../Gen3v4_imf/Gen3v4u2_Major_Upgrade_direct/fix_3/AUDIT_REPORT.md`](../../Gen3v4_imf/Gen3v4u2_Major_Upgrade_direct/fix_3/AUDIT_REPORT.md) | iMF correctness audit (Deviations A & B) |
| [`../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_18_nonvisual_step1/CHANGELOG.md`](../../Gen7_FMPCC_Viusal_Aligning/New_Based_On_Gen6_V4/fix_18_nonvisual_step1/CHANGELOG.md) | Non-visual path fixes (18.1–18.6.2) |
| [`../../Gen9/Epoch_2_Single_Camera_Avoiding_Pipeline/Fix_2/CHANGELOG.md`](../../Gen9/Epoch_2_Single_Camera_Avoiding_Pipeline/Fix_2/CHANGELOG.md) | Dim-hardcode lesson and grep recipe (§8.1 source) |
