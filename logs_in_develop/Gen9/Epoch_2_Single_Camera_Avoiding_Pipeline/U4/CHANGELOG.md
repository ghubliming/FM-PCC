# Gen9 E2 U4 — All-Bugs Fix Pass

**Date:** 2026-06-12
**Plan executed:** U3_audit_Fable.md + U3_audit_Fable_ADVERSARIAL_RESPONSE.md — all code-fixable bugs
**Scope:** Coding only. Re-eval and retrains are cluster-side.

---

## Files changed

| Bug | File | Change |
|-----|------|--------|
| B1 | `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | Drop `[:, :, ::-1]` channel flip — eval now passes RGB (matching training) |
| B2 | `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | Render at 96×96 directly (`get_image(width=96, height=96)`) — removes 1024→96 INTER_AREA downsample |
| B3 | `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | Remove dead `trajectory_selection` computation (3 lines); wire `args.mpc_batch_size` into `VisualAgent` |
| B6 | `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | `load_diffusion_with_override` returns `trainer.ema_model.model` / `trainer.ema_model` (was raw `trainer.model`) |
| B1 | `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | Same B1 fix |
| B2 | `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | Same B2 fix |
| B3 | `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | Same B3 fix; wire `args.mpc_batch_size` |
| B6 | `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | Same B6 fix |
| B10 | `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | Add `_warn_pkl_config_mismatch()` (ported from DPCC eval); call it after model load |
| B7 | `diffuser_visual_avoiding/datasets/sequence.py` | Add `episode_split(train_fraction)` method |
| B7 | `fm_visual_avoiding/datasets/sequence.py` | Same |
| B7 | `diffuser_visual_avoiding/utils/training.py` | Use `dataset.episode_split()` when available (no window-leakage); falls back gracefully |
| B7 | `fm_visual_avoiding/utils/training.py` | Same |
| B8 | `diffuser_visual_avoiding/utils/training.py` | Add `self.save(self.step)` at end of `train()` — persists steps 80001–99999 |
| B8 | `fm_visual_avoiding/utils/training.py` | Same |
| B9 | `diffuser_visual_avoiding/utils/training.py` | Add `self.model.train()` at end of `test()` — restores train mode after each validation pass |
| B9 | `fm_visual_avoiding/utils/training.py` | Same |

---

## Bug summary

| # | Severity | Affects | Fix type |
|---|----------|---------|----------|
| B1 | CRITICAL | Existing checkpoints (no retrain) | Eval-only |
| B2 | HIGH | Existing checkpoints (no retrain) | Eval-only |
| B3 | HIGH | Both eval scripts | Eval-only |
| B4 | MEDIUM | Seed lists | **Resolved externally** — manual reset on cluster |
| B5 | MEDIUM | Constraint tier | No code change — `projection_eval.yaml` is authoritative by design; sbatch comment is misleading but cluster-side |
| B6 | MEDIUM | Existing checkpoints (no retrain) | Eval-only |
| B7 | MEDIUM | **Next retrain only** | Training infra |
| B8 | LOW | **Next retrain only** | Training infra |
| B9 | LOW | **Next retrain only** | Training infra |
| B10 | LOW | FM eval missing banner; DPCC mpc_batch_size mismatch | Eval-only |

---

## Details on key fixes

### B1+B2 (eval image pipeline)

Before:
```python
bp_img_raw = env.bp_cam.get_image(depth=False)          # 1024×1024 RGB
bp_img_raw = cv2.resize(bp_img_raw, (96, 96), interpolation=cv2.INTER_AREA)
bp_image   = bp_img_raw[:, :, ::-1].transpose((2,0,1)).copy() / 255.  # BGR ← WRONG
```
After:
```python
bp_img_raw = env.bp_cam.get_image(width=96, height=96, depth=False)   # 96×96 RGB
bp_image   = bp_img_raw.transpose((2,0,1)).copy() / 255.              # RGB ✓
```

### B6 (EMA weights)
`load_diffusion_with_override` now returns `trainer.ema_model` (EMA-smoothed diffusion
wrapper) instead of `trainer.model` (raw weights). Applies to both DPCC and FM evals.

### B7 (episode-level split)
`episode_split(train_fraction)` added to `ParityAvoidingDataset` in both packages.
`Trainer.__init__` uses it when available via `hasattr`. With `horizon=8, stride=1`,
adjacent windows share 7/8 frames — a window-level split puts near-duplicates in both
splits. The episode-level split is a hard boundary: no test episode's frames appear in
training. Takes effect on the **next retrain only** — existing checkpoints are unaffected.

### B8 (final checkpoint)
`n_train_steps=1e5` with `save_freq=20000` gives saves at steps 0/20k/40k/60k/80k.
Steps 80001–99999 were never saved. `self.save(self.step)` appended to `train()` saves
step 99999 (or wherever training actually ends). **Next retrain only.**

### B9 (eval mode leak)
`test()` called `self.model.eval()` but never restored `model.train()`. From step 1000
onward all training ran in eval mode. Currently benign (GroupNorm, no Dropout) but
latent bug. Fixed with `self.model.train()` at end of `test()`. **Next retrain only.**

### B10 (FM pkl banner + mpc_batch_size)
- FM eval: `_warn_pkl_config_mismatch` ported from DPCC eval; checks `horizon`,
  `n_diffusion_steps`, `clip_denoised` against pkl-frozen values.
- Both evals: `plan_batch_size=args.mpc_batch_size` passed to `VisualAgent` — wires the
  config value (DPCC=1, FM=4) instead of using the hardcoded default 4.

---

## What is NOT changed

- `diffuser_visual_avoiding/utils/serialization.py` / `fm_visual_avoiding/utils/serialization.py`:
  `load_diffusion` still returns raw weights — other consumers may rely on this. Only the
  eval scripts' local `load_diffusion_with_override` is updated (B6).
- B5 constraint source-of-truth: `projection_eval.yaml` drives eval by design; the
  `visual_avoiding_eval.yaml` `obstacles_exact` tier is intentionally unused at eval.
  Sbatch comment fix is cluster-side.

---

## Cluster-side next steps

1. Run S5 pkl patch (if not done): `python diffuser_visual_avoiding_test/fix_pkl_clip_denoised.py --find logs/`
2. Re-eval DPCC (B1+B2+B6 now active): `./Slurm_Codes/submit.sh Slurm_Codes/sbatch/diffuser_visual_avoiding/eval_visual_avoiding_dpcc.sh`
3. Re-eval FM: `./Slurm_Codes/submit.sh Slurm_Codes/sbatch/fm_visual_avoiding/eval_fm_visual_avoiding.sh`
4. Interpret results using `RUN_DEBUG_RESULTS.md` decision tree.
5. Next retrain (when ready): B7+B8+B9 are live — new `train_test_split < 1` runs use episode-level split automatically.
