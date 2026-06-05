# U2 — Avoiding Eval Rebuild: Changelog

**Date**: 2026-06-05  
**Status**: ✅ Complete (uncommitted)  
**Plan**: [`PLAN.md`](PLAN.md)

---

## Files changed

### Renames (contents untouched)

| Old path | New path |
|---|---|
| `fm_visual_avoiding_test/` | `fm_visual_avoiding_test (legacy_based_on_visual_aligning)/` |
| `diffuser_visual_avoiding_test/` | `diffuser_visual_avoiding_test (legacy_based_on_visual_aligning)/` |

---

### New `fm_visual_avoiding_test/` (4 files)

| File | Source | Lines |
|---|---|---|
| `eval_fm_visual_avoiding.py` | `FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py` + 3 swaps | 519 |
| `train_fm_visual_avoiding.py` | copied from legacy (unchanged) | — |
| `load_results_fm_visual_avoiding.py` | `scripts/load_results.py` + 3 swaps | 73 |

### New `diffuser_visual_avoiding_test/` (4 files)

| File | Source | Lines |
|---|---|---|
| `eval_visual_avoiding_dpcc.py` | `scripts/eval.py` + 3 swaps | 501 |
| `train_visual_avoiding_dpcc.py` | copied from legacy (unchanged) | — |
| `load_results_visual_avoiding_dpcc.py` | `scripts/load_results.py` + 3 swaps | 73 |

---

## What changed in the eval scripts (the 3 swaps)

### Swap A — Package import

| | FM eval | DPCC eval |
|---|---|---|
| Before | `flow_matcher_v3_ode_selectable.utils` | `diffuser.utils` + `diffuser.sampling.Policy` |
| After | `fm_visual_avoiding.utils` | `fm_visual_avoiding.utils` |
| Projector | `flow_matcher_v3_ode_selectable.sampling.projection` | `diffuser.sampling.Projector` |
| After | `fm_visual_avoiding.sampling.projection` | `fm_visual_avoiding.sampling.projection` |

Model loader stays identical (`load_diffusion_with_override`, 4-pkl pattern). Also added load of `obs_normalizer.pkl` and `act_normalizer.pkl` from checkpoint directory.

### Swap B — Environment

Replaced bare `ObstacleAvoidanceEnv` usage (state only) with direct access to `env.bp_cam` and `env.robot.current_c_pos` — no `Avoiding_Sim` wrapper needed since both attributes are on `ObstacleAvoidanceEnv` itself.

```python
# Per step — added:
bp_img_raw = env.bp_cam.get_image(depth=False)
bp_img_raw = cv2.resize(bp_img_raw, (96, 96), interpolation=cv2.INTER_AREA)
bp_image   = bp_img_raw[:, :, ::-1].transpose((2, 0, 1)).copy() / 255.
c_xy       = env.robot.current_c_pos[:2].copy()
```

### Swap C — Per-step inference

```python
# Before (state-only):
action, samples = policy(conditions={0: obs}, batch_size=args.batch_size, horizon=args.horizon)

# After (visual):
action = agent.predict(bp_image, obs[:2].copy(), c_xy)
# desired_next_pos = next_pos_des[:2]  (replaces samples.observations[0,1,...])
```

---

## Added: `VisualAgent` and `ProjectorNormalizer` classes

Both added at the top of each eval (~55 lines combined). Replace the 700-line legacy `VisualAgentWrapper`.

**`VisualAgent.predict(bp_image, pred_xy, c_xy)`**:
1. Build 4D obs `[des_xy | c_xy]` — matches `ParityAvoidingDataset` obs format
2. Normalise obs with `obs_normalizer`
3. Build `(1,1,C,H,W)` image batch + `(1,1,4)` obs batch
4. Call `VisualFlowMatching.forward({0: (bp_batch, obs_batch)}, projector=...)`
5. Take `traj[0,0,:2]`, unnormalise → 2D delta action

**`ProjectorNormalizer`**: thin wrapper so `Projector` receives `.normalizers['observations']` / `.normalizers['actions']` as expected.

---

## `VisualFlowMatching` trajectory-dim detection

Added to the class-name check so the correct `trajectory_dim=6` / `action_dim=2` / `variant='states_actions'` path is taken:

```python
# FM eval:
if fm_model.__class__.__name__ in ('FlowMatchingODE', 'VisualFlowMatching'):
    ...  # states_actions, action_dim=2

# DPCC eval:
if diffusion.__class__.__name__ in ('GaussianDiffusion', 'VisualFlowMatching'):
    ...  # states_actions, action_dim=2
```

---

## What was dropped vs legacy

| Legacy item | Status |
|---|---|
| 700-line `VisualAgentWrapper` | Replaced by 40-line `VisualAgent` |
| `_export_rollout_realtime` (per-rollout PNG/JSON/pkl) | Dropped |
| `diag_first_replan.txt` | Dropped |
| Expert reference GIFs | Dropped |
| Z-panel / 3D-XYZ panel (aligning-specific) | Dropped |
| WandB eval hooks | Dropped |
| `Avoiding_Sim` wrapper | Not needed — `env.bp_cam` is on `ObstacleAvoidanceEnv` directly |

---

## Post-write logic audit — bugs found and fixed

### Bug 1 (CRASH) — SLURM passes `--record` / `--eval-on-train`; `utils.Parser` rejects unknown args

**Affected**: both eval scripts  
**Root cause**: Both SLURM scripts (`eval_fm_visual_avoiding.sh`, `eval_visual_avoiding_dpcc.sh`) call the eval with:
```bash
python eval_*.py $SEED_ARG --record "$RECORD_MODE" --eval-on-train
```
The new eval's top-level `argparse.ArgumentParser` uses `parse_known_args()` — unknown args land in `remaining_argv`, which is then written back into `sys.argv`. Then `Parser().parse_args()` from `fm_visual_avoiding.utils` calls `super().parse_args()` (strict argparse) and raises `SystemExit` on `--record` and `--eval-on-train`.

**Fix** (both eval scripts): add the two args as explicitly-parsed dummies so they are consumed before `sys.argv` is reset:
```python
parser.add_argument('--record', default='all')
parser.add_argument('--eval-on-train', action='store_true')
```

### Bug 2 (WRONG DIMS) — `VisualGaussianDiffusion` not in DPCC eval's class-name check

**Affected**: `diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py`  
**Root cause**: The trajectory-dim / action-dim detection checks:
```python
if diffusion.__class__.__name__ in ('GaussianDiffusion', 'VisualFlowMatching'):
```
`VisualGaussianDiffusion.__class__.__name__` = `'VisualGaussianDiffusion'` — not in the list → falls to `else` branch → `action_dim=0`, `variant='states'`. This produces wrong projector dims and the wrong trajectory slice for the action.

**Fix**: add `'VisualGaussianDiffusion'` to the match list:
```python
if diffusion.__class__.__name__ in ('GaussianDiffusion', 'VisualGaussianDiffusion', 'VisualFlowMatching'):
```

**Why `VisualGaussianDiffusion` belongs in the `states_actions` branch**: trajectory is 6D `[act(2)|des_xy(2)|c_xy(2)]`, `action_dim=2`, same as `GaussianDiffusion` and `VisualFlowMatching`. Confirmed by reading `VisualGaussianDiffusion` source.

### Confirmed-correct items

| Item | Verified |
|---|---|
| `VisualGaussianDiffusion.forward()` returns `(traj, infos)` tuple | ✅ `p_sample_loop()` → `return x, infos` |
| `VisualAgent.predict()` works for both FM and DDPM models | ✅ same `cond={0:(bp_batch,obs_batch)}` interface |
| `lb`/`ub` only used when `'bounds' in constraint_types` (always defined when used) | ✅ |
| `desired_next_pos` initialized before step loop, updated each step | ✅ |
| `fig_all.savefig()` outside variant loop uses last-iteration `fig_all` (matches baseline) | ✅ |
| Config entries `plan_fm_visual_avoiding` and `plan_visual_avoiding_dpcc` exist | ✅ |

---

## Line count comparison

| Script | Legacy (visual aligning origin) | New (state-only baseline origin) |
|---|---|---|
| FM eval | 2204 | **519** |
| DPCC eval | 2214 | **501** |
| **Reduction** | — | **−78%** |

---

## Acceptance status

| Check | Result |
|---|---|
| Legacy folders renamed, contents intact | ✅ |
| `python3 -m py_compile` on all 4 new Python files | ✅ |
| FM eval ≤ 520 lines | ✅ (519) |
| DPCC eval ≤ 520 lines | ✅ (501) |
| Train scripts copied from legacy | ✅ |
| SLURM scripts unchanged (call by path) | ✅ — no changes needed |
