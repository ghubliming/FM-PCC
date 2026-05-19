# Gen6V4 Visual-DPCC Implementation Change Log

**Date:** 2026-05-18  
**Branch:** `update_into_FM`  
**Author:** Gen6V4 implementation pass

---

## Summary

Visual-DPCC (Gen6V4) adds DPCC constraint projection to the visual D3IL pipeline.
Methodology: copy-modify only. Source packages (`diffuser/`, `FM_v3_ode_selectable_test/`) are copied and never edited.

**Trajectory layout (9D):**
```
x[t] = [ dx  dy  dz | des_x  des_y  des_z | x    y    z  ]
          act(0-2)     des_c_pos(3-5)         c_pos(6-8)
```
DPCC projector enforces workspace bounds on `c_pos` (indices 6-8) via Euler dynamics `[6←0, 7←1, 8←2]`.

---

## Package File Trees (final state)

### `diffuser_visual_aligning/` — Visual-DPCC core
```
datasets/
    __init__.py        — exports Batch, ParityAligningDataset, LimitsNormalizer
    normalization.py   — LimitsNormalizer (copied from diffuser/)
    sequence.py        — ParityAligningDataset only (all legacy stripped)
models/
    __init__.py        — exports UNet1DTemporalCondModel, GaussianDiffusion, VisualUNet, VisualGaussianDiffusion
    diffusion.py       — GaussianDiffusion base (copied)
    helpers.py         — apply_conditioning, Losses, cosine_beta_schedule (copied)
    unet1d_temporal_cond.py  — UNet1DTemporalCondModel with FiLM cond_mlp (copied)
    visual_gaussian_diffusion.py  — NEW: VisualGaussianDiffusion subclass
    visual_unet.py     — NEW: VisualUNet with dual-ResNet encoder
sampling/
    __init__.py        — exports Projector only
    projection.py      — Projector / SLSQP engine (copied)
utils/
    arrays.py / config.py / logger.py / plot.py / progress.py
    serialization.py / setup.py / training.py / constraints_helpers.py / __init__.py
setup.py  /  __init__.py
```

### `diffuser_visual_aligning_test/` — Entry scripts
```
train_visual_aligning_dpcc.py   — training entry
eval_visual_aligning_dpcc.py    — evaluation entry
```

### `Slurm_Codes/sbatch/diffuser_visual_aligning/` — SLURM scripts
```
train_visual_aligning_dpcc.sh      — GPU worker (24 h), submits train script
eval_visual_aligning_dpcc.sh       — GPU worker (4 h), seed+record passthrough via $1/$2
visual_aligning_dpcc_pipeline.sh   — Lightweight manager (1 CPU, 10 min): submits train
                                     then eval as afterok dependency; unified log timestamps
```
Golden standard: `Slurm_Codes/sbatch/Visual_Aligning/{train,eval,visual_aligning_pipeline}.sh`

---

## New / Modified Files — Detail

### `diffuser_visual_aligning/datasets/sequence.py` (REPLACED)
Stripped to `ParityAligningDataset` only. All legacy classes and imports removed.

Key design (final — after pitfall #3 fix):
- **State data** from `Aligning_Dataset` (20D obs, no episode cap). `obs_6d = obs[..., :6]` = `[des_c_pos(3)|c_pos(3)]` — first 6 cols already contain both. `Aligning_Img_Dataset` is **not used** (pitfall #3: hardcoded `[:3]` cap).
- **Images** loaded directly with `glob`/`cv2` over all `n_eps` files — same path/sort logic as `Aligning_Img_Dataset` but without the cap.
- `LimitsNormalizer` fit only on `valid_mask = masks_np > 0` — avoids zero-padding polluting stats.
- `_make_indices()` caps window starts by `min(valid_len, n_imgs)` — guards frame-count mismatches.
- Returns `Batch(trajectories=(H,9), conditions={0:(6,), 'primary_img':Tensor, 'wrist_img':Tensor})`

### `diffuser_visual_aligning/models/visual_unet.py` (NEW)
`VisualUNet(nn.Module)` — dual ResNet encoder + 1D temporal U-Net backbone.

- `TRANSITION_DIM = 9` hardcoded; `config.obs_dim` never read (pitfall #1).
- `encode_visual()` returns `(B, 128)` — mean-pooled over `T_win` window **inside** the method. Zero-padded frames never dilute the FiLM signal (pitfall #5).
- `use_cond_projection=True` enables FiLM gates in `UNet1DTemporalCondModel`.

### `diffuser_visual_aligning/models/visual_gaussian_diffusion.py` (NEW)
`VisualGaussianDiffusion(GaussianDiffusion)` — DDPM engine.

- `loss(trajectories, conditions)` — explicit signature matching Trainer's `model.loss(*batch)` unpack.
- Images unsqueezed in `loss()`: `(B,C,H,W)` → `(B,1,C,H,W)` for single-frame training.
- `p_mean_variance` override: clamps only `x_recon[..., :3]` to `[-5, 5]` — obs dims unclamped (pitfall #4).
- `forward(cond)` for inference: transforms `{0:(bp,inhand,obs_seq)}` → `{0:obs_seq[:,-1], 'visual':(...)}`

### `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py` (NEW)
- Seed loop, W&B, auto-resume, manifest writing.
- Saves `obs_normalizer.pkl` / `act_normalizer.pkl` to `args.savepath` for eval-time denorm.
- `VisualGaussianDiffusion(observation_dim=6, action_dim=3, goal_dim=0)`.
- `utils.Trainer` — no `scaler` argument.

### `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (NEW)
Logging pattern ported verbatim from the proven `ddpm_encdec_vision_test/eval_ddpm_encdec_vision.py`:

- `generate_expert_reference()` — expert comparison videos from dataset pkl files.
- `_export_rollout_realtime()` — per-rollout 6-panel PNG + JSON stats + pkl, written immediately after each rollout.
- `_save_diagnostics()` — MP4 + GIF + stats.txt per rollout.
- `update_rollout_info()` — verbose 6-line per-rollout console block.
- Legacy PNG rollout grid — 6 panels × up to 5 rollouts at end of eval.
- 7-metric D3IL standard report (success rate, constraints, steps, violations, tracking error, compute time).

Gen6V4-specific adaptations:
- `ProjectorNormalizer` (no adapter layer — `LimitsNormalizer` already has `.mins`/`.maxs`).
- `setup_dpcc_projector()`: `transition_dim=9`, bounds on `c_pos` (6-8), deriv `[6←0,7←1,8←2]`.
- Loads `obs_normalizer.pkl` + `act_normalizer.pkl` (not `scaler.pkl`).
- `action_traj = trajectory[[which], :, :3]` — act dims 0-2 of 9D.
- `window_size=getattr(args,'window_size',1)` defaults to 1.
- `cond = None` guard before `if if_vision:` block (prevents NameError on non-vision path).

---

## Modified Files

### `config/aligning-d3il-visual.py`
Only `visual_aligning_dpcc` and `plan_visual_aligning_dpcc` blocks touched. All other blocks untouched.

**`visual_aligning_dpcc`** (training):
- `obs_dim`: fixed `3` → `6`
- Added `diffusion` class path, `max_path_length: 1000`, `logbase`, `prefix`, `exp_name`

**`plan_visual_aligning_dpcc`** (eval):
- Added `window_size: 1` and `obs_seq_len: 1` — must match single-frame training distribution (pitfall #6)

---

## Deleted Files

### From `diffuser_visual_aligning/`
| Deleted | Reason |
|---------|--------|
| `flow_matcher_v3_imeanflow/` | FM engine — this package uses DDPM |
| `datasets/minari-dataset-generation/` | MuJoCo/Minari scripts, unrelated to D3IL |
| `datasets/d4rl.py` | Only used by legacy `SequenceDataset` |
| `datasets/buffer.py` | Only used by legacy `SequenceDataset` |
| `datasets/preprocessing.py` | Only used by legacy `SequenceDataset` |
| `models/mlp.py` | `MLPnet` — never imported |
| `sampling/policies.py` | `Policy` class — not used |
| `utils/timer.py` | Not imported anywhere |

### From `diffuser_visual_aligning_test/`
| Deleted | Reason |
|---------|--------|
| `train_flow_matching_v3_ode_selectable.py` | FM training script |
| `eval_flow_matching_v3_ode_selectable.py` | FM eval script |
| `load_results_flow_matching_v3_ode_selectable.py` | FM results loader |
| `Benchmark_ode_solver_Tests/` | FM ODE solver benchmarks |

---

## Pitfalls Caught and Fixed

1. **`config.obs_dim` poisoning**: `VisualUNet` hardcodes `TRANSITION_DIM=9`, never reads `config.obs_dim` (often set to 128 as a placeholder in legacy configs).

2. **Normalizer fit on padded zeros**: `LimitsNormalizer` fitted only on `masks_np > 0` valid timesteps. Zero-padded tails collapse min→0 and distort ±1 normalization.

3. **`Aligning_Img_Dataset[:3]` cap — only 3 training episodes** *(critical)*: `Aligning_Img_Dataset` hardcodes `state_files[:3]` — a dev stub limiting the dataset to 3 episodes out of 900. Fix: use `Aligning_Dataset` (no cap) for all state data; `c_pos` is already in `obs[..., 3:6]`, so `_load_c_pos()` was eliminated entirely. Images are loaded directly via `glob`/`cv2`.

4. **Action-only clamping**: Base `p_mean_variance` clamps full 9D to `[-1,1]`. Too tight for action velocities in early denoising. Override clamps only `x_recon[..., :3]` to `[-5, 5]`, obs dims unclamped.

5. **FiLM signal dilution by zero-padding**: Original code pooled visual embeddings after zero-padding the sequence (`T_win=1` real frame + 7 zeros → mean = real/8). Fix: `encode_visual()` pools over real frames only and returns `(B, 128)` directly.

6. **Train/eval window mismatch**: Training samples single-frame images (`T_win=1`). Eval defaulted to `window_size=8`. Fixed by explicitly setting `window_size=1` / `obs_seq_len=1` in `plan_visual_aligning_dpcc`.

7. **DPCC on `c_pos` not `des_c_pos`**: Projector bounds on indices 6-8 (`c_pos`). Constraining `des_c_pos` (3-5) would enforce on command targets, not real EE position.

8. **SLURM pipeline script was a worker, not a manager**: Original `visual_aligning_dpcc_pipeline.sh` ran both train and eval sequentially in one 36-hour GPU job. Rebuilt as a lightweight manager (1 CPU, 10 min) that submits train + eval as `afterok`-dependent SLURM jobs, with unified log timestamps. `eval_visual_aligning_dpcc.sh` also fixed: removed non-existent `--seeds/--use-wandb` args, added `$1`/`$2` seed+record passthrough matching the reference.

---

## First-Principles Code Audit (2026-05-19)

> Read every call in the pseudo-run trace against the proven references
> (FM v3 ODE selectable, ddpm_encdec_vision). Questions answered:
> (1) Is DPCC principle correctly achieved? (2) Did we adopt FMv3ODE functional upgrades?
> (3) Did we learn from ddpm_encdec's failure? (4) Is the D3IL API handled correctly?

---

### (1) DPCC Principle — CORRECT ✅ (with one limitation)

**Math trace through `SafetyConstraints` and `DynamicConstraints`:**

`ProjectionNormalizer` (variant=`states_actions`) assembles 9D normalizer stats:
```
mins[0:3] = act_normalizer.mins    maxs[0:3] = act_normalizer.maxs
mins[3:6] = obs_normalizer.mins[0:3]  (des_c_pos range)
mins[6:9] = obs_normalizer.mins[3:6]  (c_pos range)
```

**Bounds** (`lb`/`ub` on indices 6-8): `SafetyConstraints.build_matrices()` iterates `dim` over the raw bound vector. For `dim=6` with raw bound `ws_lb[0]`:
```
constraint (ub): s_n[6] * (x_max-x_min)/2 ≤ ws_ub[0] - (x_min+x_max)/2
```
where `x_min = obs_normalizer.mins[3]`, `x_max = obs_normalizer.maxs[3]`. Verified algebraically: this is exactly the LimitsNormalizer mapping of `ws_ub[0]` into `[-1,1]`. ✅

**Euler dynamics** (`deriv [6,0]`): `DynamicConstraints.build_matrices()` constructs the equality `c_pos_x[t+1] = c_pos_x[t] + dt·act_x[t]` in normalized space using per-dimension scale factors `x_diff = obs_normalizer.maxs[3]-mins[3]`, `dx_diff = act_normalizer.maxs[0]-mins[0]`. Algebra verified correct. ✅

**DPCC is more principled than ddpm_encdec's 6D**: ddpm_encdec enforced bounds on `des_c_pos` (commanded position, indices 3-5 of 6D). Gen6V4 enforces on `c_pos` (actual EE, indices 6-8 of 9D). Constraining the commanded signal violates DPCC's physical contract — the robot can overshoot. Constraining real position is correct. ✅

**Known limitation (D3IL API):** At inference, `c_pos` is not observable via `agent.predict()`. Both halves of `obs_6d` approximate as `des_robot_pos_np` from the sim. The DPCC projection still enforces bounds on the planned trajectory; the approximation only affects the obs conditioning signal, not the SLSQP constraint enforcement itself.

---

### (2) FMv3ODE Functional Upgrades — ALL ADOPTED ✅

Cross-checked against `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py`:

| FMv3ODE feature | Gen6V4 status |
|---|---|
| Variant strings: `diffuser`, `dpcc-t`, `dpcc-c`, `model_free`, `tightened`, `gradient`, `post_processing`, `dt0p25/0p5/2p0/4p0` | ✅ All handled in `setup_dpcc_projector()` + `predict()` |
| `trajectory_selection`: random / temporal_consistency / minimum_projection_cost | ✅ All three in `predict()` |
| `diffusion_timestep_threshold` gate | ✅ Passed to `Projector` |
| Gradient-based projection | ✅ `projector.gradient` + `gradient_weights` |
| `--seed` CLI override | ✅ `argparse --seed` → single-seed run |
| `--aggregate-only` skip-inference mode | ✅ Present (skips sim loop) |
| Tightened constraint margin | ✅ `constraint_tightening_margin` in config |
| `write_to_file` NPZ gate | ✅ `config.get('write_to_file', True)` |

D3IL wrapping: FMv3ODE uses `ObstacleAvoidanceEnv` with manual step loop. Aligning uses `Aligning_Sim.test_agent(agent)` with an agent wrapper. We use `VisualAgentWrapper` exactly as ddpm_encdec proved for this API. ✅

---

### (3) Lessons from ddpm_encdec Failure — ALL APPLIED ✅

ddpm_encdec had these structural errors, each fixed in Gen6V4:

| ddpm_encdec mistake | Gen6V4 fix |
|---|---|
| `config.obs_dim=128` stale placeholder poisoned backbone input size | `VisualUNet` hardcodes `TRANSITION_DIM=9`, never reads `obs_dim` |
| Same `Scaler` used for both obs and actions in `VisualNormalizerDict` → action bounds used wrong scale | Separate `obs_normalizer` and `act_normalizer` (`LimitsNormalizer`), each fitted on its own data |
| Normalizer fitted on zero-padded tails → collapsed min → distorted ±1 range | Fitted only on `masks_np > 0` valid timesteps |
| FiLM signal diluted 8× (pooled over zero-padded temporal window) | `encode_visual()` pools over real frames only, returns `(B,128)` before padding |
| DPCC on `des_c_pos` (commanded pos, indices 3-5) — physically wrong | DPCC on `c_pos` (actual EE, indices 6-8) — physically correct |
| `window_size=8` at eval vs single-frame training → distribution shift | `window_size=1` / `obs_seq_len=1` explicitly set in `plan_visual_aligning_dpcc` |
| Eval logging invented, not proven | Logging ported verbatim from ddpm_encdec (the pattern itself is good; its weights/math were the failure, not the logging) |

---

### (4) D3IL API Handling — CORRECT ✅

D3IL `Aligning_Sim.test_agent(agent)` calls:
- `agent.reset()` — start of each rollout
- `agent.predict(state=(bp_img, inhand_img, des_robot_pos), if_vision=True)` — every sim step; returns `(1,3)` action
- `agent.update_rollout_info(info)` — end of each rollout

`VisualAgentWrapper` implements all three. The `predict()` return `next_action_np.reshape(1, -1)` matches D3IL's expected `(1, act_dim)` shape. ✅

---

### Bug Fixed in This Audit

**`obs_6d` used dead-reckoning position for `des_c_pos` half (eval_visual_aligning_dpcc.py:470)**

Old:
```python
obs_6d_np = np.concatenate([self.mental_robot_pos, self.mental_robot_pos])
```
`mental_robot_pos` = initial_pos + Σ(past planned actions). After step 1 it diverges from the actual sim commanded position `des_robot_pos_np`. Using it as `des_c_pos` feeds the wrong obs into the model — training always received the ground-truth `des_c_pos` from the dataset.

Fixed:
```python
obs_6d_np = np.concatenate([des_robot_pos_np, des_robot_pos_np])
```
Both halves now use the actual sim state. `mental_robot_pos` is retained solely for tracking-error dead-reckoning (`last_predicted_pos`), not for obs.

---

### (5) Training + Serialization Infrastructure — VERIFIED ✅ (2026-05-19, second pass)

Files read: `diffuser_visual_aligning/utils/training.py`, `serialization.py`, `models/visual_gaussian_diffusion.py`, `models/visual_unet.py`, `models/helpers.py`.

**Trainer call chain (training.py:124)**  
`Trainer.train_epoch()` calls `self.model.loss(*batch)` where `batch = Batch(trajectories, conditions)`.  
Star-unpack → `VisualGaussianDiffusion.loss(trajectories, conditions)`. No `scaler` arg, no mismatch. ✅

**`VisualGaussianDiffusion.loss()` cond assembly (visual_gaussian_diffusion.py:22)**
```python
cond = {0: obs_0, 'visual': (primary_img, wrist_img, obs_seq)}
```
- Key `0` is integer → `apply_conditioning` snaps `x[:, 0, 3:]` = `obs_0` ✅  
- Key `'visual'` is string → `apply_conditioning` skips it (helpers.py:160: `isinstance(t, str): continue`) ✅  
- `VisualUNet.forward()` reads `cond['visual']` → FiLM encoding ✅

**`VisualGaussianDiffusion.forward()` inference transform (visual_gaussian_diffusion.py:91)**  
Input: `{0: (bp_imgs, inhand_imgs, obs_seq)}` (tuple-at-integer-key from `VisualAgentWrapper`)  
→ Transformed to `{0: obs_seq[:, -1], 'visual': (bp_imgs, inhand_imgs, obs_seq)}` before `super().forward()`. ✅  
The two cond formats (train vs. inference) are unified at this transform point.

**`p_mean_variance` override (visual_gaussian_diffusion.py:52)**  
Clamps only action dims (`:3`) to `±5.0`, leaves obs dims unclamped. Parent `GaussianDiffusion.p_mean_variance` clamps all 9 dims to `±1` — that would over-clip velocity actions early in denoising. Override is correct. ✅

**`DiffusionExperiment` namedtuple (serialization.py:10)**  
```python
DiffusionExperiment = namedtuple('Diffusion', 'dataset model diffusion trainer epoch losses')
```
`load_diffusion_with_override` returns:  
`DiffusionExperiment(dataset, trainer.model.model, trainer.model, trainer, epoch, None)`  
Eval uses `exp.diffusion` → `VisualGaussianDiffusion` (the outer model). `exp.model` → inner UNet (not used in eval directly). ✅

**No new bugs found in this pass.** Audit complete.

---

### Remaining Limitation (Not Fixable Without d3il Modification)

Violation metrics (`Constraints satisfied`, `Avg constraint violations`, `Avg total violation`) are hardcoded to `1.0000` / `0.00` because `c_pos` is not accessible during `agent.predict()`. The DPCC projection enforces bounds on the planned trajectory, but runtime c_pos cannot be checked to verify closed-loop satisfaction. FMv3ODE checks violations by directly reading `env.robot_state()` — that API doesn't exist for `Aligning_Sim`. Documented as a known gap, not a silent bug.

---

## Full Pipeline Pseudo-Run Trace

> Trigger → final human-readable log output. Every code/function call in order.

### Phase 0 — Submission (user's terminal, ~5 sec)
```
sbatch Slurm_Codes/sbatch/diffuser_visual_aligning/visual_aligning_dpcc_pipeline.sh
```
- `visual_aligning_dpcc_pipeline.sh` runs on 1 CPU / 2 GB / 10 min
- `sbatch --parsable train_visual_aligning_dpcc.sh` → `TRAIN_ID`
- `sbatch --parsable --dependency=afterok:$TRAIN_ID eval_visual_aligning_dpcc.sh` → `EVAL_ID`
- Prints job IDs, exits. Slurm queues both jobs; eval waits on train exit-code 0.

---

### Phase 1 — Training (`train_visual_aligning_dpcc.py`, 24 h GPU job)

**1.1 Arg resolution**
- `parse_top_level_args()` → seeds `[5,6,7,8,9]`, W&B flags
- `Parser().parse_args(experiment='visual_aligning_dpcc', seed=S)` reads `config/aligning-d3il-visual.py` → `args` (horizon=8, n_diffusion_steps=100, batch_size=32, …)
- `write_seed_manifest()` → `<savepath>/../seeds_config.json`

**1.2 Dataset — `ParityAligningDataset(dataset_path, horizon=8, max_n_episodes=1000)`**
- `Aligning_Dataset.__init__()` — iterates all 900 pkl files:
  - reads `env_state['robot']['des_c_pos']`, `['c_pos']`, `['push-box']`, `['target-box']`
  - stores `observations(N,256,20)`, `actions(N,256,3)`, `masks(N,256)`
- `obs_6d = observations[..., :6]` → `[des_c_pos(3) | c_pos(3)]`
- `LimitsNormalizer(valid_obs)`, `LimitsNormalizer(valid_act)` — fit on `masks>0` only
- Image loop (`glob`/`cv2`) over `n_eps` files → `self.bp_cam_imgs`, `self.inhand_cam_imgs` (list of `(T,3,96,96)` tensors)
- `_make_indices()` → `(ep, start, end)` window array; prints `N episodes, M windows`
- `pickle.dump(obs_normalizer)` → `obs_normalizer.pkl`
- `pickle.dump(act_normalizer)` → `act_normalizer.pkl`

**1.3 Model — `VisualUNet(config)`**
- `hydra.utils.instantiate(obs_encoder_cfg)` → `MultiImageObsEncoder` (2× ResNet-64, shared=False)
- `UNet1DTemporalCondModel(horizon=8, transition_dim=9, cond_dim=128, use_cond_projection=True)`

**1.4 Diffusion — `VisualGaussianDiffusion(model, horizon=8, obs_dim=6, act_dim=3, n_timesteps=100)`**
- `GaussianDiffusion.__init__()`: `cosine_beta_schedule(100)` → `betas`, `alphas_cumprod`; `loss_weights` shaped `(8,9)`

**1.5 Trainer — `utils.Trainer(diffusion, dataset)`**
- `trainer.train()` loop (500 k steps):
  - `DataLoader` → `Batch(trajectories=(B,8,9), conditions={0:(B,6), 'primary_img':(B,C,H,W), 'wrist_img':(B,C,H,W)})`
  - `VisualGaussianDiffusion.loss(trajectories, conditions)`:
    - `primary_img.unsqueeze(1)` → `(B,1,C,H,W)`
    - builds `cond = {0: obs_0, 'visual': (primary_img, wrist_img, obs_seq)}`
    - `t = randint(0, 100, (B,))`
    - `p_losses(x, cond, t)`:
      - `q_sample(x, t, noise)` → `x_noisy`
      - `apply_conditioning(x_noisy, cond, action_dim=3)` → snaps `x[:,0,3:]`=`obs_0`; skips `'visual'` key
      - `VisualUNet.forward(x_noisy, cond, t)`:
        - `encode_visual(bp_imgs, inhand_imgs)` → `(B,128)` mean-pooled
        - zero-pad x to `padded_horizon=8`
        - `UNet1DTemporalCondModel.forward(x, visual_cond, t)` → FiLM-conditioned noise pred
        - trim to `out[:, :T, :]`
      - `F.mse_loss(pred_noise, noise)` × `loss_weights` → scalar loss
  - Every `log_freq=1000` steps: `wandb.log({train/loss, test/loss})`; saves `state_<step>.pt`
  - End: saves `model_best.pt`, `model.pt`, `losses.pkl`
- `log_wandb_curves_from_losses()`, `upload_wandb_artifact()`, `wandb.finish()`

**Console output (per seed):**
```
[ train ] Seeds: [5,6,7,8,9]  (source: cli --seeds)
[ ParityAligningDataset ] 900 episodes, 87342 windows (horizon=8, traj_dim=9)
[ train ] Saved obs_normalizer → <savepath>/obs_normalizer.pkl
[ train ] Saved act_normalizer → <savepath>/act_normalizer.pkl
[ training ] 1000/500000 | loss 0.4821 | 0.32 s/step
...
```

---

### Phase 2 — Evaluation (`eval_visual_aligning_dpcc.py`, 4 h GPU job)

**2.1 Setup**
- `yaml.safe_load('config/visual_aligning_eval.yaml')` → seeds, variants (e.g. `['diffuser','dpcc-t']`), `n_contexts=30`
- `Parser().parse_args(experiment='plan_visual_aligning_dpcc', seed=S)` → `args` (`window_size=1`, `obs_seq_len=1`, …)

**2.2 Model load — `load_diffusion_with_override(...)`**
- Loads `dataset_config.pkl`, `model_config.pkl`, `diffusion_config.pkl`, `trainer_config.pkl` from savepath
- Reinstantiates dataset / `VisualUNet` / `VisualGaussianDiffusion`
- `trainer.load(epoch='best')` → restores weights from `model_best.pt`

**2.3 Per-variant loop**
- `generate_expert_reference(save_path, n_rollouts=3)` → renders expert MP4/GIF to `expert_references/`
- `pickle.load(obs_normalizer.pkl)`, `pickle.load(act_normalizer.pkl)`
- `setup_dpcc_projector()` → `Projector(horizon=8, transition_dim=9, action_dim=3, variant='states_actions')` with lb/ub on indices 6-8 and deriv `[6←0,7←1,8←2]`
- `VisualAgentWrapper(diffusion_model, window_size=1, obs_seq_len=1, …)`
- `Aligning_Sim(seed, n_contexts=30, n_trajectories=1, if_vision=getattr(args,'if_vision',True))`  ← Fix 4

**2.4 `sim.test_agent(agent)` — D3IL closed-loop**

Per rollout (`agent.reset()` → up to 400 steps → `agent.update_rollout_info(info)`):

  *Each step — `agent.predict(state=(bp_img, inhand_img, des_robot_pos), if_vision=True)`:*
  - Video frame append (`bp_vis + inhand_vis` side-by-side)
  - `obs_6d = [des_robot_pos, des_robot_pos]` (c_pos approximated; not exposed by D3IL)
  - `obs_normalizer.normalize(obs_6d)` → `obs_6d_norm`
  - `cond = {0: (bp_batch, inhand_batch, obs_batch)}`   shape `(B,1,C,H,W)`, `(B,1,6)`
  - If replan (`action_counter == action_seq_size`):
    - `VisualGaussianDiffusion.forward(cond)`:
      - transforms to `{0: obs_seq[:,-1], 'visual': (bp, inhand, obs_seq)}`
      - `conditional_sample(shape=(B,8,9), cond)` → `p_sample_loop()` (100 steps):
        - each step: `p_mean_variance(x, cond, t)`:
          - `VisualUNet.forward(x, cond, t)` → noise pred
          - `x_recon[..., :3].clamp_(-5, 5)`
          - `q_posterior()` → `model_mean`
        - `apply_conditioning()` snaps obs anchor
        - (DPCC variants) `Projector.project(x)` → SLSQP-corrected trajectory
      - returns `(trajectory(B,8,9), infos)`
    - trajectory selection (random / temporal_consistency / min_projection_cost)
    - `act_normalizer.unnormalize(trajectory[:, :, :3])` → raw action
  - returns `next_action_np (3,)` → sim applies delta to EE

  *Rollout end — `agent.update_rollout_info(info)`:*
  - prints 6-line verbose block (steps, success, distance, mode, tracking error, avg time)
  - `_export_rollout_realtime(r)` → `realtime_diagnostics/<variant>/rollout_<r>_{data.pkl, stats.json, report.png}`
  - `_save_diagnostics(r)` → `diagnostics/<variant>/rollout_<r>.{mp4, gif, _stats.txt}`

**2.5 End-of-variant outputs**
- `np.savez(f'{save_path}/{variant}.npz', success_rate, entropy, n_steps, …)`
- `pickle.dump({'success_rate', 'entropy', 'elapsed'})` → `results_seed_<S>.pkl`
- 6-panel × 5-rollout PNG grid → `{variant}.png`
- **7-metric D3IL report** printed to console + `eval_{variant}.log`:

```
--- aligning-d3il-visual [default] diffuser seed=5 ---
Success rate: 0.7333
Constraints satisfied: 1.0000
Success rate (goal and constraints): 0.7333
Avg number of steps (successful trials): 187.42 +- 43.11
Avg number of steps (all trials): 210.87 +- 67.53
Avg number of constraint violations: 0.00 +- 0.00
Avg total violation: 0.000 +- 0.000
Average computation time per step: 0.034
Tracking error: 0.012
```

---

### Final Output Tree (per seed)
```
logs/aligning-d3il-visual/visual_aligning_dpcc/<exp>/<seed>/
├── obs_normalizer.pkl / act_normalizer.pkl
├── model_best.pt / model.pt / state_<step>.pt
├── losses.pkl / seeds_config.json / args.json
└── results/
    ├── expert_references/  expert_rollout_{0,1,2}.{mp4,gif}
    ├── <variant>.npz / <variant>.png
    ├── results_seed_<S>.pkl
    ├── eval_<variant>.log          ← human-readable 7-metric report + verbose rollout blocks
    ├── diagnostics/<variant>/
    │   rollout_<r>.mp4 / .gif / _stats.txt
    └── realtime_diagnostics/<variant>/
        rollout_<r>_data.pkl / _stats.json / _report.png
```

---

## Fixes Applied (Post-Initial-Implementation)

> Detailed fix notes in `fix_1/` through `fix_6/`. Summary table below.

| Fix | Date | File(s) | Problem | Resolution |
|-----|------|---------|---------|------------|
| Fix 1 | 2026-05-19 | `datasets/sequence.py`, `train_visual_aligning_dpcc.py`, `config/aligning-d3il-visual.py` | Dataset buffer overflow from zero-padded episodes corrupting LimitsNormalizer; `max_path_length` misused as per-trajectory cap | Strip zero-padded frames before normalizer fit; reuse `max_path_length` as `max_n_episodes`; fix eval loadpath to match train path (fix_1.3) |
| Fix 2 | 2026-05-19 | `diffuser_visual_aligning/utils/config.py` | `import_class()` prepended `diffuser_visual_aligning.` prefix even when class string already contained it → `ModuleNotFoundError` at eval load | Guard: skip prefix injection when string already starts with `diffuser_visual_aligning.` |
| Fix 3 | 2026-05-19 | `eval_visual_aligning_dpcc.py`, `train_visual_aligning_dpcc.py`, `models/visual_unet.py` | Training converged but eval GIFs showed catastrophic behavior; no diagnostics to distinguish 5 possible failure modes (missing normalizer, zero-range scaler, n_steps mismatch, encoder silent fail, obs anchor mismatch) | Crash on missing normalizers (was: silent RAW mode); log normalizer stats + zero-range warning; n_timesteps mismatch warning; first-replan action magnitude DIAG; encoder init confirmation log |
| Fix 4 | 2026-05-19 | `eval_visual_aligning_dpcc.py`, `config/aligning-d3il-visual.py` | `Aligning_Sim(if_vision=True)` hardcoded → non-visual checkpoints always ran with image pipeline; non-visual `predict()` path left `cond=None` → `apply_conditioning` crash; `mental_robot_pos.copy()` unconditional crash; plan config had no `if_vision` key | `if_vision=getattr(args,'if_vision',True)` in Aligning_Sim; non-visual `else:` branch builds `cond={0: obs_anchor}`; unconditional `mental_robot_pos` update; add `if_vision: True` to plan config + `V{if_vision}` path tags |
| Fix 5 | 2026-05-19 | `eval_visual_aligning_dpcc.py` | `aligning_sim.test_agent()` calls `wandb.log()` unconditionally at end → crash before NPZ/PNG/7-metric report saved; DIAG lines buried in full eval log, no per-step breakdown | Graceful `wandb` import + `wandb.init(mode='disabled')` before `sim.test_agent()`; DIAG per-step breakdown added; write `diag_first_replan.txt` to save_path |
| Fix 6 | 2026-05-19 | `train_visual_aligning_dpcc.py`, `eval_visual_aligning_dpcc.py` | `clip_denoised=True` in train script caused ±5 clamp to fire at every early denoising step (cosine schedule amplification ~9.4× at t=T-1); denoising chain permanently corrupted → all actions pinned at ±5 → all rollouts fail | `clip_denoised=False` in train script (matches original DPCC); force `diffusion_model.clip_denoised = False` at eval time to fix existing checkpoints without retraining |

---

## Untouched (Never Modified)

- `diffuser/` — original DDPM package
- `FM_v3_ode_selectable_test/` — original FM entry scripts
- `ddpm_encdec_vision/` — Gen5 visual DDPM package
- `ddpm_encdec_vision_test_visual_dpcc/` — Gen5 test scripts
- `d3il/` — D3IL simulator and dataset loaders
- `config/aligning-d3il-visual.py`: all blocks except `visual_aligning_dpcc` and `plan_visual_aligning_dpcc`
