# Gen6V4 Visual-Aligning DPCC — Plan 2

**Date:** 2026-05-18  
**Status:** Awaiting approval. Supersedes all documents in `plan_1/`.  
**Methodology:** Copy-modify. Pick one source, copy it, modify the copy. Never edit originals.

---

## 1 — Copy-Modify Sources

| Role | Source (read-only, never edit) | Destination (new, we own) |
|------|-------------------------------|--------------------------|
| Core package | `diffuser/` | **`diffuser_visual_aligning/`** (fresh copy) |
| Entry scripts | `FM_v3_ode_selectable_test/` | **`diffuser_visual_aligning_test/`** (fresh copy) |

**Learn-from references** (read-only, never edit):
- `diffuser_visual_aligning(Outdated)/` — previous attempt at the same core copy, use as diff reference
- `ddpm_encdec_vision_test_visual_dpcc(Outdated)/` — old entry scripts, use as diff reference
- `ddpm_encdec_vision/` + `ddpm_encdec_vision_test/` — Gen5/6 visual pipeline, copy any component needed into our destination folders

---

## 2 — Architecture in One Picture

```
[Raw D3IL Pickles (.pkl)]
         │
         ▼
 ParityAligningDataset          ← new class in diffuser_visual_aligning/datasets/sequence.py
 obs   = [des_pos(3D), c_pos(3D)]  ← 6D, from raw pkl (c_pos not in Aligning_Img_Dataset)
 act   = [dx, dy, dz]              ← 3D
 traj  = [act(3) | obs(6)]         ← 9D, shape (H=8, 9)
 images= dual ResNet frames        ← conditioning only, not in trajectory
         │
         ▼ Batch(trajectories[B,H,9], conditions{0, primary_img, wrist_img})
         │
 ┌───────────────────────────────────────────────┐
 │         diffuser_visual_aligning/             │
 │                                               │
 │  VisualUNet (transition_dim=9 hardcoded)      │  ← modified from diffuser/models/
 │      ResNet×2 → 128D → FiLM into 1D U-Net    │
 │                  │                            │
 │  VisualGaussianDiffusion                      │  ← modified from diffuser/models/
 │      loss(self, trajectories, conditions)     │
 │                  │                            │
 │  3D SLSQP Projector (unchanged)               │  ← copied as-is from diffuser/sampling/
 │      bounds on c_pos indices [6, 7, 8]        │
 │      deriv: [6←0, 7←1, 8←2]                  │
 │                  │                            │
 │  RHC Policy: execute a_0 every step           │  ← copied as-is from diffuser/sampling/
 └───────────────────────────────────────────────┘
         │ [dx, dy, dz]
         ▼
 Aligning_Sim.test_agent(VisualAgentWrapper)     ← in diffuser_visual_aligning_test/eval_...py
```

---

## 3 — Trajectory Space (Locked)

```
trajectory x_t ∈ ℝ^(H=8 × 9)
= [ dx   dy   dz  |  des_x  des_y  des_z  |  x    y    z  ]
    idx0  1    2      idx3    4      5       idx6   7    8
    ──── act(3D) ────  ─────── des_pos(3D) ──  ──── c_pos(3D) ────
```

Why 9D, not 6D: DPCC must enforce `x_{t+1} = x_t + dx_t` on the **actual** robot position (`c_pos`). `c_pos` lives only in raw pkl files — `Aligning_Img_Dataset.observations` only exposes `des_c_pos`. Without `c_pos` in the trajectory, the projector is projecting command targets instead of real positions, which violates the physical DPCC contract.

---

## 4 — What Gets Created / Modified

### 4a — New Core: `diffuser_visual_aligning/`

Copy `diffuser/` → `diffuser_visual_aligning/`. Then modify only these files:

| File | Copy base | Modification |
|------|-----------|-------------|
| `datasets/sequence.py` | `diffuser/datasets/sequence.py` | Add `ParityAligningDataset` class: raw pkl loading for `c_pos`, 9D trajectory, dual `LimitsNormalizer`, returns `Batch(trajectories, conditions)` |
| `datasets/normalization.py` | `diffuser/datasets/normalization.py` | Fix import: `from diffuser_visual_aligning.datasets.normalization` (self-contained) — no cross-package imports |
| `models/visual_unet.py` | `ddpm_encdec_vision/models/visual_unet.py` → copy into here | Add `VisualUNet`: `transition_dim=9` hardcoded for visual mode (fix_5 lesson — never read from `config.obs_dim`) |
| `models/visual_gaussian_diffusion.py` | `diffuser_visual_aligning(Outdated)/models/visual_gaussian_diffusion.py` | Rewrite `loss(*args)` → `loss(self, trajectories, conditions)` to match `Batch` namedtuple; 9D-aware conditioning |
| All other files | copied as-is from `diffuser/` | No changes needed |

### 4b — New Test Folder: `diffuser_visual_aligning_test/`

Copy `FM_v3_ode_selectable_test/` → `diffuser_visual_aligning_test/`. Then modify:

| File | Copy base | Modification |
|------|-----------|-------------|
| `train_visual_aligning_dpcc.py` | `FM_v3_ode_selectable_test/train_flow_matching_v3_ode_selectable.py` | Swap: use `ParityAligningDataset`, `VisualUNet`, `VisualGaussianDiffusion`, `diffuser_visual_aligning.utils.Trainer`; save `obs_normalizer.pkl` + `act_normalizer.pkl` |
| `eval_visual_aligning_dpcc.py` | `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` | Swap: wire `VisualAgentWrapper` (learn from `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`); 9D state construction `[des_pos, c_pos]`; load `obs/act_normalizer.pkl`; 9D projector constraints |

### 4c — Config Update

| File | Change |
|------|--------|
| `config/aligning-d3il-visual.py` | Update `visual_aligning_dpcc` block: `obs_dim: 6`, `loader: ParityAligningDataset`, add `max_path_length`; add `diffusion` class pointer |

### 4d — SLURM Scripts (New)

| File | Copy base |
|------|-----------|
| `Slurm_Codes/sbatch/diffuser_visual_aligning/train_visual_aligning_dpcc.sh` | Copy from `Slurm_Codes/sbatch/Visual_Aligning/train_visual_aligning_dpcc.sh`, update paths |
| `Slurm_Codes/sbatch/diffuser_visual_aligning/eval_visual_aligning_dpcc.sh` | New, mirror pattern |
| `Slurm_Codes/sbatch/diffuser_visual_aligning/visual_aligning_dpcc_pipeline.sh` | Chain train → eval |

---

## 5 — Never Touch

| Folder / File | Reason |
|---------------|--------|
| `diffuser/` | Source package — copy-modify principle |
| `ddpm_encdec_vision/` | Source package — copy-modify principle |
| `FM_v3_ode_selectable_test/` | Source for entry scripts — copy-modify principle |
| `fm_encdec_vision/` + `fm_encdec_vision_test/` | Gen7, unrelated |
| `d3il/` | Vendored simulator — always frozen |
| `ddpm_encdec_vision_test_visual_dpcc(Outdated)/` | Outdated attempt — reference only |
| `diffuser_visual_aligning(Outdated)/` | Outdated attempt — reference only |
| All other `flow_matcher_*/`, `FM_*/` | Unrelated generations |

---

## 6 — Implementation Order

1. Copy `diffuser/` → `diffuser_visual_aligning/` (fresh)
2. `diffuser_visual_aligning/datasets/sequence.py` — add `ParityAligningDataset`
3. `diffuser_visual_aligning/models/visual_unet.py` — add `VisualUNet` (copy from `ddpm_encdec_vision`, hardcode `transition_dim=9`)
4. `diffuser_visual_aligning/models/visual_gaussian_diffusion.py` — rewrite `loss()`
5. Copy `FM_v3_ode_selectable_test/` → `diffuser_visual_aligning_test/` (fresh)
6. `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py` — swap imports and dataset
7. `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` — wire `VisualAgentWrapper`, 9D projector
8. `config/aligning-d3il-visual.py` — update config block
9. SLURM scripts

---

## 7 — Key Decisions Locked In

| Decision | Choice | Reason |
|----------|--------|--------|
| Core source | `diffuser/` (DPCC) | Already used in `diffuser_visual_aligning(Outdated)/`; DPCC projector is the goal |
| Entry source | `FM_v3_ode_selectable_test/` | Most recent clean DPCC test template |
| Trajectory | 9D `[act(3) \| des_pos(3) \| c_pos(3)]` | DPCC must project actual `c_pos`, not just desired |
| `transition_dim` | Hardcoded `9` | Never read from `config.obs_dim` — fix_5 lesson |
| `loss()` signature | `loss(self, trajectories, conditions)` | Matches `Batch` namedtuple; no `*args` fragility |
| Normalizer | `LimitsNormalizer` × 2 (obs 6D, act 3D) | Replaces Gen5/6 `Scaler`; self-contained |
| Projector bounds | On `c_pos` indices [6, 7, 8] | Correct physical safety semantics |
| Control | RHC: execute `a_0` every step | Reactive; matches Gen3v2 state-only DPCC |
| Vision always on | No state-only fallback | Reduces branching; fix_5 isolation lesson |
