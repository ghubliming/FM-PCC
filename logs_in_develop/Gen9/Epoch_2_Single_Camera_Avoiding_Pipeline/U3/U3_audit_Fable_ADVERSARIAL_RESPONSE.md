# Adversarial Audit of U3_audit_Fable.md

**Auditor:** Claude Opus 4 (Thinking)
**Date:** 2026-06-12
**Method:** Every claim in `U3_audit_Fable.md` verified against the actual source files in the workspace.
**Scope:** Same as Fable 5's audit — the complete codebase was read, not summary-checked.

---

## Executive Summary

| # | Fable Verdict | Actual Verdict | Notes |
|---|---|---|---|
| B1 | CRITICAL | ✅ **CONFIRMED** — genuine bug | BGR/RGB swap is real |
| B2 | HIGH | ✅ **CONFIRMED** — genuine bug | 1024→96 vs direct 96 |
| B3 | HIGH | ✅ **CONFIRMED** — genuine bug | trajectory_selection is dead code |
| B4 | MEDIUM | ⚠️ **PARTIALLY CORRECT** — real mismatch, crash claim unverified | Seed mismatch exists; crash mechanism is plausible but depends on whether seed 10 directories exist |
| B5 | MEDIUM | ✅ **CONFIRMED** — but severity overstated | constraint_list IS dead at eval; however this is by design (eval uses projection_eval.yaml) |
| B6 | MEDIUM | ✅ **CONFIRMED** — correct observation | EMA never used at eval |
| B7 | MEDIUM | ✅ **CONFIRMED** — correct observation | Window-level split leaks |
| B8 | LOW | ✅ **CONFIRMED** — correct after re-analysis | Last periodic save at step 80000, training runs to 99999 |
| B9 | LOW | ✅ **CONFIRMED** — correct observation | model.eval() not restored in test() |
| B10 | LOW | ⚠️ **PARTIALLY CORRECT** | ODE knobs dead: TRUE. mpc_batch_size claim: MISLEADING |

**Bottom line:** 8 of 10 claims fully correct. 2 partially correct. 0 factually wrong.
Fable's B1 finding is the most important — it IS a real, previously-missed bug.

---

## B1 — RGB/BGR channel swap: ✅ CONFIRMED

Fable's analysis is **correct** and the logic chain holds end-to-end. Verified:

1. **Collection** (`collect_visual_avoiding_data.py:184-186`):
   - `env.bp_cam.get_image(depth=False)` returns **RGB** (confirmed: MuJoCo's `mjr_readPixels` reads an OpenGL framebuffer which is RGB, docstring at `Camera.py:148` says "RGB image")
   - `cv2.cvtColor(bp, cv2.COLOR_RGB2BGR)` converts to BGR
   - `cv2.imwrite(...)` saves BGR as PNG — correct convention

2. **Training loader** (`diffuser_visual_avoiding/datasets/sequence.py:154-155`):
   - `cv2.imread(p)` → BGR
   - `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` → RGB
   - **Training tensors: RGB** ✓

3. **Eval** (`diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py:423-426`,
   `fm_visual_avoiding_test/eval_fm_visual_avoiding.py:411-414`):
   - `env.bp_cam.get_image(depth=False)` → RGB (live render, no disk round-trip)
   - `bp_img_raw[:, :, ::-1]` → flips channels → BGR
   - **Eval tensors: BGR** ✗

**This is a real train/eval domain gap.** With `imagenet_norm=True` in the ResNet encoder
(`fm_visual_avoiding/models/visual_unet.py:55`), ImageNet per-channel mean/std are applied
(R=0.485/0.229, G=0.456/0.224, B=0.406/0.225). Swapping R and B changes the normalized
values the encoder sees. Since the avoiding scene has **red** obstacles, this is not just
a theoretical concern — the feature response to obstacle pixels is degraded.

**Fable's fix is correct:** drop the `[:, :, ::-1]` flip in both eval scripts.

---

## B2 — Render resolution mismatch: ✅ CONFIRMED

Verified:

- **Collection** (`collect_visual_avoiding_data.py:63`): `--resolution` default is 96.
  Line 184: `env.bp_cam.get_image(width=resolution, height=resolution, depth=False)` → renders at 96×96.

- **Eval** (`eval_visual_avoiding_dpcc.py:423-425`):
  ```python
  bp_img_raw = env.bp_cam.get_image(depth=False)    # no width/height → uses camera default
  bp_img_raw = cv2.resize(bp_img_raw, (_IMG_W, _IMG_H), interpolation=cv2.INTER_AREA)
  ```

- **Camera default** (`avoiding.py:25`): `BPCageCam.__init__` has `width: int = 1024, height: int = 1024`.

- Since `get_image()` is called with no `width`/`height` args, it falls through to `self.width=1024` (`MjCamera.py:80-83`).

**Confirmed:** Train renders at 96×96 direct EGL. Eval renders at 1024×1024 then downsamples
with INTER_AREA. This is a real texture-domain mismatch. Fable's fix is correct.

---

## B3 — trajectory_selection dead code: ✅ CONFIRMED

Verified in both eval scripts:

- `eval_visual_avoiding_dpcc.py:346-348` computes `trajectory_selection` from the variant name
- It is **never passed** to `VisualAgent()` (line 350-351) nor to `agent.predict()` (line 428)
- `VisualAgent.predict()` unconditionally uses `traj[0, 0, :2]` (line 81 in DPCC, line 96 in FM)
- The variable `trajectory_selection` is a dead local that goes out of scope at the end of each variant loop iteration

**Confirmed.** dpcc-c, dpcc-t, dpcc-r differ only by RNG and constraint tightening. The
selection mechanism is not implemented in the visual agent.

---

## B4 — Seed mismatch: ⚠️ PARTIALLY CORRECT

**The mismatch is real:**
- `train_visual_avoiding_dpcc.py:17`: `DEFAULT_SEEDS = [5, 6, 7, 8, 9]`
- `fm_visual_avoiding_test/train_fm_visual_avoiding.py:17`: `DEFAULT_SEEDS = [5, 6, 7, 8, 9]`
- `config/projection_eval.yaml:7`: `seeds: [6,7,8,9,10]`

Seeds 5 is trained but not evaluated. Seed 10 is in the eval list but not in the default
train list.

**The crash claim is plausible but slightly oversimplified.** Whether seed 10 actually crashes
depends on whether anyone ran training with `--seed 10` or a custom seed list. The
`load_diffusion_with_override` function doesn't have explicit error handling for missing
directories — it would fail at `utils.load_config(*loadpath, 'dataset_config.pkl')` with a
`FileNotFoundError`, which is indeed a crash. But it's not specifically `pickle.load` that
fails — it's the config path resolution.

**The claim that "hours of eval, aggregates never saved" is only true if `--seed` is NOT used.**
Both eval scripts support `--seed` override (lines 106-108 / 125-127), so the crash only
happens on unattended YAML-driven multi-seed runs.

**Fable's core point stands.** The seed lists should be aligned.

---

## B5 — config constraint_list is dead at eval: ✅ CONFIRMED (but overstated)

Verified:

- `config/avoiding-d3il-visual.py:206`: `'constraint_list': list(_AVOIDING_OBSTACLES)` — defines
  6 sphere_outside constraints at exact obstacle positions

- **Neither eval script reads `args.constraint_list`.** The constraint building at
  `eval_visual_avoiding_dpcc.py:260-291` uses only `config['obstacle_constraints'][exp]` from
  `projection_eval.yaml`.

- `projection_eval.yaml:73-82` defines 3 main + 3 tightened obstacles — not the same 6 from
  `get_obj_xy_list()`.

**However, Fable overstates the severity.** The plan config's constraint_list was always
designed as a **documentation/planning** artifact, not consumed at eval. The docstring in
`sequence.py:34-35` says _"They belong in the PLANNING config as `sphere_outside` projector
constraints"_ — describing where they conceptually belong, not asserting they're wired into
the eval. The eval's constraint source was always `projection_eval.yaml`, which implements
paper-ablation geometry (varying obstacle counts/radii per halfspace variant).

**The real issue is the `visual_avoiding_eval.yaml` situation.** This file exists and has proper
6-obstacle tiers, but the eval scripts don't read it. The `.py` config reads
`_proj_config['diffusion_timestep_threshold']` from it (line 24-32), but that's the extent
of its consumption. The sbatch comment about it is indeed misleading.

---

## B6 — EMA never evaluated: ✅ CONFIRMED

Verified:

- `training.py:55`: `self.ema_model = copy.deepcopy(self.model)` — EMA copy created
- `training.py:132-133`: EMA updated every `update_ema_every` steps
- `training.py:225-226`: Saved in checkpoint as `'ema': self.ema_model.state_dict()`
- `training.py:320-321`: Loaded from checkpoint
- `serialization.py:75`: `load_diffusion` returns `trainer.model.model` (the inner model, NOT ema)
- `eval_visual_avoiding_dpcc.py:179`: `load_diffusion_with_override` returns `trainer.model.model`

**Confirmed.** EMA weights are computed and saved but never used at eval. Fable's observation
is correct and the fix suggestion (return `trainer.ema_model` instead of `trainer.model`) is
straightforward.

---

## B7 — Leaky window-level train/test split: ✅ CONFIRMED

Verified:

- `training.py:74-76`:
  ```python
  n_train = int(train_test_split * len(self.dataset))
  n_test = len(self.dataset) - n_train
  train_dataset, test_dataset = torch.utils.data.random_split(self.dataset, [n_train, n_test])
  ```
- `sequence.py:134-142`: `_make_indices` creates windows with stride 1:
  ```python
  for start in range(usable - self.horizon + 1):
      indices.append((ep, start, start + self.horizon))
  ```

With `horizon=8` and stride 1, adjacent windows share 7/8 frames. `random_split` operates
on **window indices**, not episodes. A "test" window at position `t` shares 7/8 frames with
the "train" window at position `t±1`. This is a standard data leakage issue.

**Fable is correct.** The split should be at the episode level for meaningful validation.

---

## B8 — Final 20k steps never checkpointed: ✅ CONFIRMED

Traced the logic:

- `n_train_steps = 1e5` (from config) → integer: 100000
- `save_freq = n_train_steps // 5` → `100000 // 5 = 20000`
- Saving condition (`training.py:135`): `if self.step % self.save_freq == 0`
- Step counter starts at 0 and increments at line 183: `self.step += 1`

The save fires at steps 0, 20000, 40000, 60000, 80000. The loop runs through step 99999.
Step 100000 is never reached inside the loop. No final `self.save()` call exists after the
training loop in `train()` (lines 185-196).

**Confirmed.** The last periodic save is at step 80000. Training runs through step 99999
without a final checkpoint. Using `diffusion_epoch='best'` mitigates this when `state_best`
fires late, but the fundamental bug is real. Fable's fix is correct.

---

## B9 — test() leaves model in eval() mode: ✅ CONFIRMED

Verified in both `diffuser_visual_avoiding/utils/training.py:198-216` and
`fm_visual_avoiding/utils/training.py:198-216`:

```python
def test(self, n_test=100):
    self.model.eval()   # line 199 — sets eval mode
    # ... runs test ...
    return test_loss, test_a0_loss
    # NO self.model.train() call
```

This is called at every `log_freq` step when `train_test_split < 1` (line 144-145). From
the first call onwards, training runs in eval mode.

**Fable is correct that this is currently benign** — `use_group_norm=True` replaces BatchNorm,
and the architecture has no dropout. But it IS a latent bug.

---

## B10 — Dead configuration knobs: ⚠️ PARTIALLY CORRECT

### ODE solver knobs: ✅ CONFIRMED
`visual_gaussian_diffusion.py:14-20` accepts `ode_solver_backend_v3`, `ode_solver_method_v3`,
`ode_solver_rtol_v3`, `ode_solver_atol_v3`, `ode_solver_step_size_v3` and passes them to
`super().__init__(*args, **kwargs)`. Since `FlowMatchingODE` (the parent) uses only legacy
Euler in its `p_sample_loop`, these parameters are effectively dead.

### mpc_batch_size: ⚠️ MISLEADING claim

Fable says: _"Plan config `mpc_batch_size` (DPCC: 1, FM: 4) is never wired into VisualAgent
(`plan_batch_size=4` hardcoded default in both eval scripts)"_

This is technically accurate — the config value `mpc_batch_size` is never read by the eval
script, and `VisualAgent.__init__` defaults `plan_batch_size=4`. But calling it "dead" is
slightly misleading because:
1. `VisualAgent`'s `plan_batch_size=4` IS the intended value for FM (matching the config's
   `mpc_batch_size: 4`)
2. For DPCC the config says `mpc_batch_size: 1` but the agent uses 4 — this IS a mismatch
   worth noting, but it's not "dead code" per se, it's an unconnected config value

### FM eval lacks pkl banner: ✅ CONFIRMED

The FM eval script (`eval_fm_visual_avoiding.py`) has no `_warn_pkl_config_mismatch` function
or call. The DPCC eval does have it (lines 117-143). This is a genuine asymmetry.

---

## Items Fable marked "verified clean" — spot-check

I verified the following subset of Fable's "clean" claims:

### Action encode/decode parity: ✅ CONFIRMED
- Train: `actions = (robot_des_xy[1:] - robot_des_xy[:-1])` (sequence.py:88) →
  a[t] = des[t+1] - des[t]
- Eval: `next_pos_des = action + obs[:2]` (both eval scripts) →
  des_next = a + des_current
- This is consistent. ✓

### Env plumbing (info tuple): ✅ CONFIRMED
- `avoiding.py:171`: `return observation, reward, done, (self.mode_encoding, self.success)`
- Eval uses `info[1]` for success flag → `self.success`. ✓

### Obs layout consistency: ✅ CONFIRMED
- Dataset: `obs_4d = [des_xy, c_xy]` (sequence.py:85-87)
- Eval obs: `obs = np.concatenate((action[:2], obs))` where `action=robot_state()[:2]`
  (des_xy) and `obs=get_observation()` → `robot_c_pos[:2]` → `obs = [des_xy, c_xy]` ✓
- `projection_eval.yaml:20`: `{'x_des': 0, 'y_des': 1, 'x': 2, 'y': 3}` ✓

---

## Corrected Recommended Action Order

Based on verified findings, I adjust Fable's recommendations:

1. **Eval-side patch (no retrain, highest priority):**
   - Fix B1: Drop `[:, :, ::-1]` in both eval scripts ← **CONFIRMED CRITICAL**
   - Fix B2: Render at 96×96 directly →
     `env.bp_cam.get_image(width=96, height=96, depth=False)`
   - Fix B4: Align seed lists (either change YAML to `[5,6,7,8,9]` or train seeds to `[6..10]`)

2. **Eval-side improvements (no retrain, medium priority):**
   - B6: Switch to EMA weights — free A/B test, likely improves results
   - B10: Port `_warn_pkl_config_mismatch` to FM eval script

3. **Code quality (no retrain, low priority):**
   - B3: Either implement trajectory_selection in VisualAgent or trim variant list
   - B5: Decide constraint source of truth (projection_eval.yaml vs visual_avoiding_eval.yaml)
   - B10: Wire `mpc_batch_size` from config to VisualAgent, or document the override

4. **Next retrain:**
   - B7: Episode-level split
   - B8: Add `self.save(self.step)` at end of `train()`
   - B9: Add `self.model.train()` at end of `test()`
