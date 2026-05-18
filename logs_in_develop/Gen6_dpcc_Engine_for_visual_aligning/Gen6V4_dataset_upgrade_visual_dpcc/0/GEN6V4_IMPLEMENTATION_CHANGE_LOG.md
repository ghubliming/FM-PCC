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
- `Aligning_Sim(seed, n_contexts=30, n_trajectories=1, if_vision=True)`

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

## Untouched (Never Modified)

- `diffuser/` — original DDPM package
- `FM_v3_ode_selectable_test/` — original FM entry scripts
- `ddpm_encdec_vision/` — Gen5 visual DDPM package
- `ddpm_encdec_vision_test_visual_dpcc/` — Gen5 test scripts
- `d3il/` — D3IL simulator and dataset loaders
- `config/aligning-d3il-visual.py`: all blocks except `visual_aligning_dpcc` and `plan_visual_aligning_dpcc`
