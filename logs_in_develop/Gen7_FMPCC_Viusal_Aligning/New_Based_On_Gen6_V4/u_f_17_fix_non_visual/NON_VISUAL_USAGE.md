# Non-Visual Aligning — Usage Guide (Post UF-17)

**Date**: 2026-05-29  
**Applies to**: Gen7 FM-PCC (`fm_visual_aligning_test/`) and Gen6V4 DPCC (`diffuser_visual_aligning_test/`)

---

## What Non-Visual Mode Is

`if_vision=False` runs the same DPCC/FM pipeline but without cameras. The full task
state (robot EE position, box position/orientation, target position/orientation) lives
inside a **23D trajectory** and is pinned at step 0 via `apply_conditioning`. No ResNet,
no FiLM. Follows original DPCC principle exactly.

```
Trajectory: [dx(0) dy(1) dz(2) | des_x(3) des_y(4) des_z(5) | c_x(6) c_y(7) c_z(8)
             | box_x(9)…box_z(11) | bq_w(12)…bq_z(15)
             | tgt_x(16)…tgt_z(18) | tq_w(19)…tq_z(22)]
= act(3) + obs(20) = 23D
```

Constraint projector is unchanged — c_pos still at dims 6-8, dynamics `[6←0, 7←1, 8←2]`.

---

## Step 1 — Train

### FM (Gen7)

```bash
python fm_visual_aligning_test/train_fm_visual_aligning.py \
    "experiment=fm_visual_aligning" \
    "if_vision=False" \
    "seed=42"
```

Or via SLURM with the existing FM train script — just set `if_vision=False` in the
`fm_visual_aligning` config block, or pass `if_vision=False` as a hydra override.

**What changes at train time** (auto-detected from `args.if_vision`):
- Dataset: `StateOnlyAligningDataset` (23D trajectory, no images)
- `observation_dim=20` passed to `VisualFlowMatching`
- `VisualUNet` uses non-visual branch: `transition_dim = action_dim + obs_dim = 23`
- `VisualFlowMatching.loss()` routes to base `p_losses()` (no image key access)

**Saved checkpoint**:
```
logs/aligning-d3il-visual/fm_visual_aligning/<exp>/seed_<s>/
  state_<step>.pt
  obs_normalizer.pkl     ← 20D LimitsNormalizer
  act_normalizer.pkl     ← 3D LimitsNormalizer
```

### DPCC (Gen6V4)

Same pattern with `diffuser_visual_aligning_test/train_visual_aligning_dpcc.py` and
`experiment=ddpm_encdec_vision_nonvisual`.

---

## Step 2 — Eval

### FM (Gen7)

```bash
python fm_visual_aligning_test/eval_fm_visual_aligning.py \
    "experiment=plan_fm_visual_aligning" \
    "diffusion_loadpath=fm_visual_aligning/<your_exp_path>" \
    "seed=42"
```

The eval script reads `args.if_vision` from the loaded config. When `False`:
- `predict()` receives 20D obs from `aligning_sim.py` (sim prepends `des_c_pos[:3]` to 17D env obs)
- Full 20D obs normalized and passed as `cond={0: obs_anchor_20d}`
- `apply_conditioning` pins obs_20d at trajectory step 0
- Projector built with `trajectory_dim=23`, lb/ub padded to 23D

### DPCC (Gen6V4)

```bash
python diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py \
    "experiment=plan_ddpm_encdec_vision_nonvisual" \
    "diffusion_loadpath=ddpm_encdec_vision_nonvisual/<your_exp_path>" \
    "seed=42"
```

---

## Recording in Non-Visual Mode

By default `if_vision=False` uses `RenderMode.BLIND` — no cameras, no frames.
Passing `--record gif` (or `all`) auto-promotes to visual rendering for the GIF/MP4
output only. The **model still runs in non-visual inference mode** — images are
captured offscreen purely for human review.

```bash
python fm_visual_aligning_test/eval_fm_visual_aligning.py \
    "experiment=plan_fm_visual_aligning" \
    --record gif
# Console warning: "config if_vision=False but record_mode is active → auto-enabling visual mode"
# GIF is saved; model prediction remains state-only
```

---

## How the 20D Obs Reaches the Wrapper

`aligning_sim.py` non-visual branch (no change needed):
```python
pred_action = env.robot_state()   # last commanded pos (3D)
while not done:
    obs = np.concatenate((pred_action[:3], obs))  # prepend des_c_pos → 20D
    pred_action = agent.predict(obs)              # 20D passed to wrapper
    pred_action = pred_action[0] + obs[:3]        # delta + des_c_pos = new abs pos
```

The 20D vector layout at eval time:
```
obs[:3]   = des_c_pos   (last commanded position, prepended by sim)
obs[3:6]  = c_pos       (actual robot EE from env.get_observation)
obs[6:9]  = box_pos
obs[9:13] = box_quat
obs[13:16]= target_pos
obs[16:20]= target_quat
```

This matches `StateOnlyAligningDataset` obs layout exactly — no bridge needed.

---

## Key Differences from Visual Mode

| | Visual | Non-Visual |
|---|---|---|
| `if_vision` | `True` | `False` |
| Trajectory dim | 9D | 23D |
| obs in trajectory | 6D (des_c_pos + c_pos) | 20D (full state) |
| Conditioning | ResNet → 128D FiLM | None (pure apply_conditioning) |
| Camera | EGL offscreen | BLIND (unless `--record` forces visual) |
| Inference cost | ~3× higher (ResNet/step) | Low |
| Dataset | `ParityAligningDataset` | `StateOnlyAligningDataset` |
| Normalizers | obs_normalizer (6D) | obs_normalizer (20D) |
| Projector `trajectory_dim` | 9 | 23 |

---

## Comparing Results

Non-visual eval produces the same output files as visual:
`rollout_N_stats.json`, `results_seed_S.json`, `constraint_metrics.json`.
The `success_rate` and `mean_distance` metrics are directly comparable to:
- Visual FM-PCC / DPCC (same model family, same MPC, same projector)
- D3IL state-only agents (same obs space, same task, same success metric)

Use the `no_constraint` or `dynamics_only` variant for the fairest comparison to D3IL
(D3IL has no constraint projection).
