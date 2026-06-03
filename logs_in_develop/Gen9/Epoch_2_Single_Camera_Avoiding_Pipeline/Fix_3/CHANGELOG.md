# Gen9 Epoch 2 — Fix-3: Eval Script Crashes (ImportError + dim hardcodes + wrong loadpath)

**Date**: 2026-06-03
**Status**: ✅ Fixed (uncommitted)
**Triggered by**: `temp/debug_Gen9E2/outputs` (jobs 21147/21148) + `outputs_2` (job 21156)
**Parent**: [`../CHANGELOG.md`](../CHANGELOG.md), [`../Fix_2/CHANGELOG.md`](../Fix_2/CHANGELOG.md)
**Pre-predicted by**: Fix-2 CHANGELOG §7 flagged this exact class of eval-side hardcodes

---

## 1. Symptom

Both eval jobs crashed at import time (line 58 / 57 respectively):

```
# DPCC eval — job 21147:
File ".../diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py", line 58, in <module>
    from d3il.simulation.avoiding_sim import Aligning_Sim
ImportError: cannot import name 'Aligning_Sim' from 'd3il.simulation.avoiding_sim'

# FM eval — job 21148:
File ".../fm_visual_avoiding_test/eval_fm_visual_avoiding.py", line 57, in <module>
    from d3il.simulation.avoiding_sim import Aligning_Sim
ImportError: cannot import name 'Aligning_Sim' from 'd3il.simulation.avoiding_sim'
```

Both jobs failed identically.

---

## 2. Root causes (layered)

### 2.1 Wrong import name (surface crash)

Both eval scripts were copy-modified from `eval_fm_visual_aligning.py`. The import was changed to point at `avoiding_sim` but the class name `Aligning_Sim` was left unchanged. `avoiding_sim.py` only defines `Avoiding_Sim`.

Additionally, the usage at line 1956 (`sim = Avoiding_Sim(...)`) was correct but referenced an **unimported** name — so even if the import had been silently skipped, a `NameError` on `Avoiding_Sim` would have followed.

### 2.2 `Avoiding_Sim` missing visual interface

The original `Avoiding_Sim` in `avoiding_sim.py` was the bare D3IL non-visual sim:
```python
def __init__(self, seed, device, render, n_cores=1, n_trajectories=30):
    ...
```
It had no `n_contexts`, `n_trajectories_per_context`, `if_vision`, `eval_on_train`, `max_episode_length` — all of which the eval scripts pass. It also returned `(successes, entropy)` from `test_agent()`, not the `(success_rate, mode_encoding, successes, mean_distance)` 4-tuple the eval scripts unpack.

### 2.3 Aligning-era dimension hardcodes in eval scripts

All inherited from the aligning eval copy (predicted in Fix-2 §7):

| Location | Hardcode | Correct for avoiding |
|---|---|---|
| `setup_dpcc_projector` comment + `_DIM` dict | `'x': 6, 'y': 7, 'z': 8`, `dz`, `des_z` in 9D layout | `'x': 4, 'y': 5`, 2D action; no `z` |
| `setup_dpcc_projector` default `trajectory_dim=9` | 9D aligning | 6D avoiding |
| `_target_obs_dim = trajectory_dim - 3` | `action_dim=3` aligning | `trajectory_dim - 2` (action_dim=2) |
| `pad = trajectory_dim - 9` | aligning 9D traj | `trajectory_dim - 6` (avoiding 6D) |
| `np.full(6, -np.inf)` in bounds construction | skip `act(3)+des(3)=6` dims | skip `act(2)+des(2)=4` dims |
| `predict()` unpack: `bp_np, inhand_np, des_robot_pos_np, robot_pos_np = state` | dual cam, 3D pos | single cam, 2D pos |
| `obs_6d_np = concat([des_robot_pos, robot_pos])` | 6D obs | 4D obs `[des_xy(2)\|c_xy(2)]` |
| `cond = {0: (bp_batch, inhand_batch, obs_batch)}` | dual-cam tuple | single-cam tuple `(bp_batch, obs_batch)` |
| `traj_np[:, :, 6:9]` c_pos extraction | aligning layout | `traj_np[:, :, 4:6]` |
| `dummy[:, 3:6]` obs-normalizer slot | obs indices for c_pos | `dummy[:, 2:4]` |
| `trajectory[[which], :, :3]` action extraction | action_dim=3 | `[:2]` (action_dim=2) |
| Multiple `trajectory[..., :3]` diagnostic references | 3D action | `[:2]` |

### 2.4 `ObstacleAvoidanceEnv` differences from `Robot_Push_Env`

| Aspect | Aligning | Avoiding |
|---|---|---|
| `if_vision` param on env | ✅ switches obs format | ❌ not supported — camera always exists, obs always state-only |
| Image source | `(env_state, bp_img, inhand_img) = obs` | `obs = env.reset()` (state); images from `env.bp_cam.get_image()` separately |
| `step()` info return | dict `{'mode', 'success', 'mean_distance'}` | tuple `(mode_enc_array, success_bool)` |
| Context files | `data/aligning/train_contexts.pkl` | ❌ none — random starts only |
| Camera count | dual (bp + inhand) | single (bp only) |
| Action dim | 3D | 2D |
| Mode encoding | 2 modes | binary (success=1, fail=0) |

---

## 3. Fix

### 3.1 `d3il/simulation/avoiding_sim.py` — replace with visual-extended class

Rewrote `Avoiding_Sim` to match `Aligning_Sim`'s interface:

```python
class Avoiding_Sim(BaseSim):
    def __init__(self, seed, device, render, n_cores=1,
                 n_contexts=30, n_trajectories_per_context=1,
                 if_vision=False, eval_on_train=False, max_episode_length=400):
```

Key design decisions:
- `n_contexts` → number of independent random-start episodes (no context files)
- `eval_on_train` accepted but ignored (no train/test split for avoiding; `random=True` always)
- `step()` info tuple `(mode_enc, success)` → converted to `{'mode': int, 'success': bool, 'mean_distance': float}` inline
- `mean_distance` = `|c_xy[1] - env.goal_ypos|` (distance to goal y-line per step, averaged)
- Mode encoding: `1` = success, `0` = failure (2-mode for eval script compatibility)
- `test_agent()` returns `(success_rate, mode_encoding, successes, mean_distance)` — matches eval script unpack
- Visual `eval_agent`: single `bp_cam.get_image()` call; `capture_frame(bp, bp)` passes bp twice since no inhand
- Non-visual `eval_agent`: original loop adapted to context structure and info dict

### 3.2 Both eval scripts — import fix

```python
# Before (broken):
from d3il.simulation.avoiding_sim import Aligning_Sim

# After:
from d3il.simulation.avoiding_sim import Avoiding_Sim
```

### 3.3 Both eval scripts — `setup_dpcc_projector` dims

```python
# Before:
def setup_dpcc_projector(..., trajectory_dim=9):
    _DIM = {'dx': 0, 'dy': 1, 'dz': 2, 'des_x': 3, 'des_y': 4, 'des_z': 5, 'x': 6, 'y': 7, 'z': 8}
    _target_obs_dim = trajectory_dim - 3
    pad = trajectory_dim - 9
    lb = np.concatenate([np.full(6, -np.inf), ws_lb, np.full(pad, -np.inf)])

# After:
def setup_dpcc_projector(..., trajectory_dim=6):
    _DIM = {'dx': 0, 'dy': 1, 'des_x': 2, 'des_y': 3, 'x': 4, 'y': 5}
    _target_obs_dim = trajectory_dim - 2
    pad = trajectory_dim - 6
    lb = np.concatenate([np.full(4, -np.inf), ws_lb, np.full(pad, -np.inf)])
```

### 3.4 Both eval scripts — `predict()` visual path

```python
# Before (aligning, dual-cam, 3D):
bp_np, inhand_np, des_robot_pos_np, robot_pos_np = state
obs_6d_np = np.concatenate([des_robot_pos_np, robot_pos_np])  # (6,)
obs_t = ...  # (1, 6)
cond = {0: (bp_batch, inhand_batch, obs_batch)}

# After (avoiding, single-cam, 2D):
bp_np, des_xy_np, c_xy_np = state
obs_4d_np = np.concatenate([des_xy_np, c_xy_np])  # (4,)
obs_t = ...  # (1, 4)
cond = {0: (bp_batch, obs_batch)}   # no inhand_batch
```

Video capture: single `bp_vis` frame (no side-by-side inhand concatenation).

### 3.5 Both eval scripts — trajectory extraction

```python
# Before (aligning 9D, c_pos at 6:9, action at 0:3):
cpos_norm = traj_np[:, :, 6:9]   # (B, H, 3)
dummy[:, 3:6] = cpos_norm.reshape(-1, 3)
action_traj = trajectory[[which], :, :3]   # (1, H, 3)
norm_a0 = trajectory[[which], 0, :3]
full_norm = trajectory[which, :, :3]

# After (avoiding 6D, c_xy at 4:6, action at 0:2):
cpos_norm = traj_np[:, :, 4:6]   # (B, H, 2)
dummy[:, 2:4] = cpos_norm.reshape(-1, 2)
action_traj = trajectory[[which], :, :2]   # (1, H, 2)
norm_a0 = trajectory[[which], 0, :2]
full_norm = trajectory[which, :, :2]
```

---

## 4. Files touched

```
M  d3il/simulation/avoiding_sim.py                          (full class rewrite)
M  diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py  (import + 5 dim sites)
M  fm_visual_avoiding_test/eval_fm_visual_avoiding.py          (import + 5 dim sites)
```

---

## 5. Verification (Phase-0 Docker)

| Check | Result |
|---|---|
| AST parse on all 3 files | ✅ |
| No `Aligning_Sim` import remaining | ✅ |
| No stale `[:3]` action slices | ✅ |
| No stale `obs_6d_np` / `inhand_batch` in visual path | ✅ |
| `trajectory_dim=6` default in both projector setups | ✅ |
| `_target_obs_dim = trajectory_dim - 2` in both | ✅ |
| `pad = trajectory_dim - 6` in both | ✅ |

**Cluster-side rerun expectation**: both eval scripts should now pass the import stage and reach the first rollout without crashing. The `Avoiding_Sim.eval_agent()` visual branch provides images from `env.bp_cam.get_image()` and calls `agent.predict((bp_image, des_xy, c_xy), if_vision=True)` with the correct 3-item tuple matching the updated `predict()` unpack.

---

---

## Fix-3.5 — `outputs_2` (job 21156): FileNotFoundError after import fixed

### Symptom

After Fix-3 landed (jobs 21147/21148 fixed the ImportError), job 21156 got past import but crashed loading the checkpoint:

```
[ eval loading ] Loading from logs/avoiding-d3il/fm_visual_avoiding/H8_Dfm_visual_avoiding.models.visual_gaussian_diffusion.VisualFlowMatching_a1.5_b1.0_aw1/6
FileNotFoundError: ... 'logs/avoiding-d3il/fm_visual_avoiding/H8_D..._aw1/6/dataset_config.pkl'
```

Two separate mismatches between where training saved and where eval looked.

### Root cause A — wrong log base directory

Both eval scripts had `dataset: str = 'avoiding-d3il'` in their `Parser` class. The `dataset` field drives the log base: `logs/{dataset}/...`. Training used `exp = 'avoiding-d3il-visual'` → saved to `logs/avoiding-d3il-visual/...`. Eval looked in `logs/avoiding-d3il/...`.

### Root cause B — FM `diffusion_loadpath` missing `_V_steps_bs` suffix

The `fm_visual_avoiding` training config uses `args_to_watch_fm_visual_train` which includes `('if_vision', 'V')`, `('max_path_length', 'steps')`, `('batch_size', 'bs')`. Training therefore saved to:
```
fm_visual_avoiding/H8_D..._a1.5_b1.0_aw1_VTrue_steps200_bs64/
```

But `plan_fm_visual_avoiding.diffusion_loadpath` in `avoiding-d3il-visual.py` was:
```python
'f:fm_visual_avoiding/H{horizon}_D{diffusion}_a{time_beta_alpha_v3}_b{time_beta_beta_v3}_aw{action_weight}'
```
— missing `_V{if_vision}_steps{max_path_length}_bs{train_batch_size}`. (DPCC plan already had all three; only FM plan was incomplete.)

### Fix-3.5 changes

| File | Change |
|---|---|
| `fm_visual_avoiding_test/eval_fm_visual_avoiding.py` | `dataset: str = 'avoiding-d3il-visual'` |
| `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py` | same |
| `config/avoiding-d3il-visual.py` `plan_fm_visual_avoiding` | `diffusion_loadpath` += `_V{if_vision}_steps{max_path_length}_bs{train_batch_size}` |

All 3 files AST-clean. DPCC `diffusion_loadpath` already had the full suffix — unchanged.

---

## 6. Known remaining items

- `ObstacleAvoidanceEnv.reset(random=True)` — `random=True` is the default but not tested with the visual model. If the env doesn't randomize starting positions sufficiently, add a context seed based on `context` index.
- `env.goal_ypos` used for `mean_distance` metric — verify this attribute exists on the specific env version on the cluster.
- Non-visual path in the eval scripts (`obs_20d_np[:3]` / `[3:6]`) is still aligning-era and untouched — non-visual avoiding eval is out of scope.

---

## 7. Cross-references

| Document | Content |
|---|---|
| [`../Fix_2/CHANGELOG.md`](../Fix_2/CHANGELOG.md) §7 | Pre-predicted this exact set of eval-side hardcodes |
| [`../CHANGELOG.md`](../CHANGELOG.md) §7 smoke recipe | Dim-hardcode grep step (now confirmed as necessary) |
| `d3il/simulation/aligning_sim.py` | Template for the new `Avoiding_Sim` interface |
| `d3il/environments/d3il/envs/gym_avoiding_env/gym_avoiding/envs/avoiding.py` | `ObstacleAvoidanceEnv` — confirms `max_steps_per_episode`, `bp_cam`, `goal_ypos`, `step()` info tuple format |
