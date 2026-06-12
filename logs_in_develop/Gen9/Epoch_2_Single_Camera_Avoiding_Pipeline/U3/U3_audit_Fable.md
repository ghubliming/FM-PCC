# Gen9 E2 Visual Avoiding — Full Code Audit (U3_audit_Fable)

**Auditor:** Claude Fable 5
**Date:** 2026-06-12
**Rev 2 (2026-06-12):** updated after the adversarial cross-check
(`U3_audit_Fable_ADVERSARIAL_RESPONSE.md`, Opus 4): 8/10 findings confirmed, 0 overturned.
Incorporated its nuances on B4 (crash mechanism precision), B5 (severity framing), and B10
(`mpc_batch_size` wording). B4 additionally marked **resolved externally** — seed lists are
manually reset in the remote codebase.
**Scope:** `diffuser_visual_avoiding/`, `fm_visual_avoiding/`, `diffuser_visual_avoiding_test/`,
`fm_visual_avoiding_test/`, `config/avoiding-d3il-visual.py`, `config/projection_eval.yaml`,
`config/visual_avoiding_eval.yaml`, `collect_visual_avoiding_data/`, and the touched parts of
`d3il` (env, camera, vision encoder). Every model/math/dataset/eval file read in full; constraint
algebra re-derived by hand. **Only verified bugs listed.** Items already known and fixed in U3
(clip_denoised pkl-wins, K=20 banner) are not re-reported.

---

## Summary table

| # | Severity | Area | Bug | Retrain needed? |
|---|---|---|---|---|
| B1 | **CRITICAL** | Eval (both models) | RGB/BGR channel swap at eval — model sees swapped channels | No — eval-side fix |
| B2 | HIGH | Eval (both models) | Render-resolution mismatch: train 96×96 direct, eval 1024×1024 + INTER_AREA | No — eval-side fix |
| B3 | HIGH | Eval (both models) | `trajectory_selection` dropped — dpcc-r/c/t variants mechanically identical | No |
| B4 | ~~MEDIUM~~ resolved externally | Eval ops | Seed mismatch train [5–9] vs eval yaml [6–10] — **handled by manual seed reset on the remote codebase** | No |
| B5 | MEDIUM | Eval validity | 6-obstacle constraint set (`obstacles_exact`) never exercised; eval tests paper-ablation geometry; `visual_avoiding_eval.yaml` unconsumed | No |
| B6 | MEDIUM | ML (repo-wide) | EMA model trained + saved but never evaluated | No (weights already in ckpt) |
| B7 | MEDIUM | ML methodology | Train/test split at window level → leaky validation; "best" checkpoint selection unreliable | Retrain to fix properly |
| B8 | LOW | Training | Final 20k steps never checkpointed (last periodic save at 80k) | — |
| B9 | LOW | Latent | `Trainer.test()` leaves model in eval() mode for rest of training (benign today) | — |
| B10 | LOW | Config hygiene | Dead knobs: `ode_solver_*_v3`, `mpc_batch_size`; FM eval lacks pkl banner | — |

**Headline:** B1 is an S1-family bug that the U3 double-check incorrectly cleared (correction in
§B1). B1+B2+B3 are all eval-side — fixable and re-testable without any retraining.

---

## B1 — CRITICAL: eval feeds channel-swapped (BGR) images to a model trained on RGB

**This corrects the U3 audit's "S1 clean" verdict.** That check compared the dataset loader's
transform against the eval transform and found them "matching" — but it never traced the third
leg: what the **collection script** wrote to disk.

The full chain:

1. **Collection** — `collect_visual_avoiding_data/collect_visual_avoiding_data.py:182-187`:
   ```python
   # get_image returns RGB uint8 (H, W, 3); convert to BGR so cv2.imwrite
   bp = env.bp_cam.get_image(width=resolution, height=resolution, depth=False)
   bp = cv2.cvtColor(bp, cv2.COLOR_RGB2BGR)
   ...
   cv2.imwrite(...)
   ```
   This is the *correct* cv2 convention. PNG on disk → `cv2.imread` returns true-BGR.

2. **Training loader** — `diffuser_visual_avoiding/datasets/sequence.py:154-155`:
   ```python
   img = cv2.imread(p)                                  # true BGR
   img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0   # → true RGB
   ```
   **Training tensors are true RGB.** ✓

3. **Eval** — `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py:423-426` (identical at
   `fm_visual_avoiding_test/eval_fm_visual_avoiding.py:411-414`):
   ```python
   bp_img_raw = env.bp_cam.get_image(depth=False)            # true RGB (Camera.py:141 docstring)
   bp_image = bp_img_raw[:, :, ::-1].transpose((2, 0, 1)) / 255.   # → true BGR  ✗
   ```
   **Eval tensors are true BGR — red and blue swapped relative to training.**

The `[:, :, ::-1]` flip was copied from the aligning eval, where the disk convention differed.
For avoiding it inverts the channels. With `imagenet_norm=True` (per-channel mean/std differ for
R vs B) and a scene whose obstacles are **red**, the ResNet encoder receives an input
distribution it never saw in training. This degrades visual conditioning for **both** DPCC and
FM at eval. It is not the explosion mechanism (that was clip_denoised), but it is a real,
permanent train/eval domain gap.

**Fix (eval-side only, both eval scripts):**
```python
bp_image = bp_img_raw.transpose((2, 0, 1)).copy() / 255.   # keep RGB; drop [:, :, ::-1]
```

---

## B2 — HIGH: render-resolution mismatch (96 direct vs 1024 + downsample)

- Training images: rendered **directly at 96×96** EGL
  (`collect_visual_avoiding_data.py:184`, `--resolution` default 96).
- Eval: `env.bp_cam.get_image(depth=False)` with no size args → the avoiding `BPCageCam`
  default **1024×1024** (`gym_avoiding/envs/avoiding.py:25`), then
  `cv2.resize(..., (96, 96), INTER_AREA)` (`eval_visual_avoiding_dpcc.py:424-425`).

A 1024→96 area-downsample is heavily antialiased; a direct 96×96 MuJoCo render is aliased.
Thin features (obstacle edges, the end-effector) look measurably different. Second train/eval
image-domain gap, stacking with B1.

**Fix:** render at the training resolution and drop the resize:
```python
bp_img_raw = env.bp_cam.get_image(width=96, height=96, depth=False)
```

---

## B3 — HIGH: `trajectory_selection` computed but never used — dpcc-c / dpcc-t results are mislabeled

`eval_visual_avoiding_dpcc.py:346-348` (and FM eval :334-336) computes:
```python
trajectory_selection = 'random'
if 'dpcc-t' in variant: trajectory_selection = 'temporal_consistency'
if 'dpcc-c' in variant: trajectory_selection = 'minimum_projection_cost'
```
…and then never passes it anywhere. `VisualAgent.predict` always executes batch sample 0
(`traj[0, 0, :2]`, line 81). The state-based baseline (`scripts/eval.py:209-216`) passes it to
`Policy(..., trajectory_selection=...)` — the visual rewrite dropped the feature but kept the
full 17-variant list in `projection_eval.yaml`.

**Consequence:** `dpcc-r`, `dpcc-c`, `dpcc-c-tightened`, `dpcc-t`, `dpcc-t-tightened`, and all
`dpcc-c-dt*` rows differ only by RNG — the selection mechanisms they claim to ablate are not
implemented. Any FM-vs-DPCC comparison per variant is fine *within* a variant, but the variant
labels are meaningless.

**Fix:** either implement selection in `VisualAgent.predict` (it already has the per-sample
`projection_costs` available from `projector.project`; temporal consistency needs the previous
plan kept on the agent), or cut the variant list down to `diffuser` + `dpcc-r` until selection
is implemented. Do not publish dpcc-c/dpcc-t numbers from the current code.

---

## B4 — ~~MEDIUM~~ RESOLVED EXTERNALLY: seed mismatch train [5–9] vs eval yaml [6–10]

**Status: no code action — seed lists are manually reset in the remote codebase.** Kept here
only as a record. Original finding, with the adversarial review's precision applied: training
default `DEFAULT_SEEDS = [5,6,7,8,9]` vs `projection_eval.yaml` `seeds: [6,7,8,9,10]`. On an
unattended YAML-driven multi-seed run (no `--seed` override), seed 10's missing run dir fails at
`utils.load_config(...)` path resolution with `FileNotFoundError` after seeds 6–9 complete, and
the post-loop cross-seed aggregate figures are never written; seed 5 is trained but never
evaluated. Per-seed `--seed` fan-out runs (the sbatch pattern) are unaffected.

---

## B5 — MEDIUM: the plan config's 6 exact obstacles are dead; eval measures paper-ablation geometry

`config/avoiding-d3il-visual.py` defines `constraint_list: list(_AVOIDING_OBSTACLES)` — 6
`sphere_outside` constraints at the exact `get_obj_xy_list()` positions, radius 0.04. The
dataset/config docstrings present this as *the* mechanism for obstacle constraints ("They belong
in the PLANNING config as sphere_outside projector constraints").

**The eval never reads `args.constraint_list`.** It builds constraints exclusively from
`projection_eval.yaml` (eval script :260-291): per halfspace-hard variant, **one** obstacle
(radius 0.06/0.08, paper-ablation positions) plus halfspace + bounds + dynamics. Meanwhile
`config/visual_avoiding_eval.yaml` — purpose-built in Fix-4 with `obstacles_exact` tiers — is
consumed **only** for `diffusion_timestep_threshold` by the `.py` config. The sbatch header
(`eval_visual_avoiding_dpcc.sh:59`: "Uses config/visual_avoiding_eval.yaml for seed/variant
configuration") is false.

**Consequence:** the "primary avoiding metric" (projection against the actual 6 obstacles) has
never been run; `collision_free_completed` / violation metrics measure DPCC-paper ablation
geometry, not the env's obstacle field. Success rate (from `info[1]`) is unaffected.

**Severity framing (per adversarial review, accepted):** the `.py` config's `constraint_list`
can be read as a planning/documentation artifact rather than a wiring promise — the eval's
constraint source was always `projection_eval.yaml`. The substantive problem is therefore not
"dead config" per se but that **`visual_avoiding_eval.yaml`'s `obstacles_exact` tier — the
purpose-built primary metric — has never been executed**, and the sbatch header misdocuments
which YAML drives the eval.

**Fix:** decide one source of truth. Cheapest: append `args.constraint_list` entries into the
eval's obstacle-constraint build (they're already in the right `(type, center, radius)` form),
or finish the Fix-4 plan and port the eval to `visual_avoiding_eval.yaml`'s geo variants. Then
fix the sbatch comment.

---

## B6 — MEDIUM: EMA weights trained, saved, and never used at eval (repo-wide)

`Trainer` maintains an EMA copy (`ema_decay=0.995`, `step_start_ema=2000`) and saves it in every
checkpoint (`'ema': self.ema_model.state_dict()`). Upstream diffuser evaluates the EMA model —
the original namedtuple is still visible, commented out, at
`diffuser_visual_avoiding/utils/serialization.py:9`:
```python
# DiffusionExperiment = namedtuple('Diffusion', 'dataset renderer model diffusion ema trainer epoch')
DiffusionExperiment = namedtuple('Diffusion', 'dataset model diffusion trainer epoch losses')
```
Both `load_diffusion` and the eval's `load_diffusion_with_override` return `trainer.model` (raw
online weights). The EMA — the standard, smoother eval model for diffusion — is dead compute.
This is inherited from the repo's DPCC adaptation (state-based `scripts/eval.py` does the same),
so FM-vs-DPCC comparisons are internally consistent, but absolute performance of every model in
the repo is measured on noisier raw weights.

**Fix (one line, no retrain — EMA is already in the checkpoints):** in
`load_diffusion_with_override`, return/use `trainer.ema_model` instead of `trainer.model`.
Worth an A/B re-eval: it is free and often worth several success-rate points.

---

## B7 — MEDIUM: window-level train/test split → leaky validation; "best" checkpoint weakly defined

`Trainer.__init__` (`utils/training.py:74-82`) does `random_split` over **dataset windows**.
Adjacent windows from the same episode share 7 of 8 frames (horizon 8, stride 1 in
`_make_indices`), so nearly every "test" window has near-duplicates in train. Test loss is a
near-copy of train loss; `state_best.pt` (selected on this test loss) and any "best-val step"
reasoning carry little signal about generalization. Additionally the normalizers are fit on all
episodes before the split (minor, same family).

**Fix:** split at the **episode** level (e.g., hold out 10% of episode indices before windowing).
Requires a retrain to produce a meaningful `state_best`. Until then, treat `diffusion_epoch:
'best'` as ≈ "latest with extra steps", not as model selection.

---

## B8 — LOW: last 20% of training is never checkpointed

`save_freq = n_train_steps // 5` and saving happens at `self.step % save_freq == 0`
(`training.py:135`), i.e. at steps 0/20k/40k/60k/80k. The loop ends at step 100k **without a
final save** — `state_80000.pt` is the newest periodic checkpoint; `epoch='latest'` silently
discards the final 20k steps. Mitigated only when `state_best` happens to fire late.
**Fix:** call `self.save(self.step)` at the end of `Trainer.train()`.

---

## B9 — LOW (latent): `Trainer.test()` never restores train mode

`test()` calls `self.model.eval()` and never `self.model.train()` (`training.py:198-216`); from
step 1000 onward training runs in eval() mode. **Currently benign** — verified: the encoder uses
GroupNorm (`use_group_norm=True` swaps all BatchNorm2d), SpatialSoftmax has `noise_std=0.0`, and
the temporal UNet has no dropout/batchnorm. It becomes a silent bug the moment anyone adds
BatchNorm or dropout (e.g., a pretrained ResNet without the GN swap). **Fix:** add
`self.model.train()` at the end of `test()`.

---

## B10 — LOW: dead configuration knobs

- `VisualFlowMatching.__init__` accepts `ode_solver_backend_v3 / method / rtol / atol /
  step_size` and **discards them all** (`fm_visual_avoiding/models/visual_gaussian_diffusion.py:14-21`).
  Only legacy Euler exists; the plan config's solver knobs are decorative.
- Plan config `mpc_batch_size` is an **unconnected config value** (reworded per adversarial
  review): `VisualAgent` hardcodes `plan_batch_size=4`. For FM this coincides with the config's
  `mpc_batch_size: 4`; for DPCC the config says `1` but the agent runs 4 — a silent mismatch,
  not dead code. Wire it through or document the override.
- The FM eval lacks the U3-C2 `_warn_pkl_config_mismatch` banner — the same pkl-wins precedence
  applies to its `flow_steps_v3`/`horizon`. Port the banner over.

---

## Verified clean (checked, no bug found)

- **DDPM math**: cosine β schedule, `q_sample`/`q_posterior`/`predict_start_from_noise`
  coefficients, ε-prediction loss with conditioned-row zeroing, reverse-loop direction. The
  0.5× sampling noise scale is the upstream low-temperature convention (train/infer asymmetry is
  inherited and shared by all baselines).
- **FM math**: linear path `x_t=(1−t)·x₀+t·x₁`, target `v=x₁−x₀`, forward Euler `x+=v/K`,
  time scaling 0→(K−1)/K, Beta(1.5,1) time sampling consistent between `loss()` and visual path.
  (FM's inherited `predict_start_from_noise` returns the noise endpoint, but it is unused in the
  FM sampling path.)
- **Projector algebra**: normalized-space sphere constraint (P, q, v) and Euler-link `deriv`
  rows re-derived by hand — both correct; `skip_initial_state` anchor handling matches reference;
  Fix 9.1/9.2 empty-constraint short-circuits correct. Dynamics coupling (both `des` and `c`
  linked to actions, dt=1) matches the original DPCC avoiding convention.
- **Action encode/decode parity**: train `a[t]=des[t+1]−des[t]` ↔ eval `next_des=a+obs[:2]`;
  obs layout `[des_xy, c_xy]` consistent across dataset, agent, `observation_indices`, and the
  6-D trajectory indices used by constraints.
- **Env plumbing**: `get_observation()` returns c_pos xy; `step` returns
  `(obs, rew, done, (mode_encoding, success))` → `info[1]` is the success flag.
- **Cross-package class override** (Fix_1 importlib path), `apply_conditioning` str-key guard,
  horizon padding (8→8 no-op), normalizer save/load symmetry, `state_best` ↔
  `diffusion_epoch='best'` ↔ `find_latest_checkpoint_step` (ValueError-guarded) interplay.

---

## Recommended action order (all before the next conclusions are drawn)

1. **Eval-side patch, no retrain:** fix B1 (drop `[::-1]`) + B2 (render 96×96) in both eval
   scripts, switch to EMA weights (B6) — then re-run the S5 pkl-patched eval. These together
   change what the model *sees* and *which weights run*; the current eval numbers (FM and DPCC
   alike) do not faithfully measure the trained models. (B4 seeds: handled manually on the
   remote codebase — no code change.)
2. Trim or implement trajectory selection (B3) before quoting per-variant DPCC numbers.
3. Decide the constraint source of truth (B5) — the 6-obstacle `obstacles_exact` tier has never
   actually been tested.
4. On the next retrain only: episode-level split (B7) + final-step save (B8) + `model.train()`
   restore (B9).
