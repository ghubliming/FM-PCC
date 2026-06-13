# Gen9 Epoch 2 — Fix-5: Rollout Crash (`record_context_info` + expert gen + `_traj_dim`)

**Date**: 2026-06-03
**Status**: ✅ Fixed (uncommitted)
**Triggered by**: `temp/debug_Gen9E2/outputs_4` — Slurm job 21159
**Parent**: [`../Fix_4/CHANGELOG.md`](../Fix_4/CHANGELOG.md)

---

## 1. Symptom

Job 21159 successfully entered the variant loop (`Context 0 Rollout 0` printed) but crashed
immediately after the first rollout started:

```
File "d3il/simulation/avoiding_sim.py", line 82, in eval_agent
    agent.record_context_info({}, int(context))
File "fm_visual_avoiding_test/eval_fm_visual_avoiding.py", line 954, in record_context_info
    pos, quat, target_pos, target_quat = context
ValueError: not enough values to unpack (expected 4, got 0)
```

Two non-blocking warnings also present:
```
[ expert ] WARNING — expert reference generation failed: 'push-box'
[ eval ] WARNING: unexpected _traj_dim=6 (act=2, obs=4); expected 9 (visual) or 23 (non-visual)
```

---

## 2. Root causes

### 2.1 `record_context_info` crash (blocking)

`Avoiding_Sim.eval_agent` called `agent.record_context_info({}, int(context))` with an empty
dict `{}` as the context object.  
`VisualAgentWrapper.record_context_info` is aligning-specific: it unpacks context as a 4-tuple
`(box_pos, box_quat, target_pos, target_quat)` — the position and quaternion of the push-box and
target object that exist in the aligning task but **do not exist in avoiding**.  
Unpacking `{}` as a 4-tuple raises `ValueError: not enough values to unpack`.

### 2.2 Expert generation `'push-box'` KeyError (non-blocking, caught by try/except)

Expert generation code (copy from aligning) read:
```python
box_pos  = expert_data['push-box']['pos'][0]
target_pos = expert_data['target-box']['pos'][0]
context = (box_pos, box_quat, target_pos, target_quat)
env.reset(random=False, context=context)
...
_, bp_img, ih_img = obs   # avoiding obs is state-only; images via bp_cam
```
The avoiding expert pkl has no `push-box` or `target-box` keys. Also the obs unpack assumed
dual-camera aligning format.

### 2.3 `_traj_dim=6` warning (non-blocking)

The guard `if _traj_dim not in (9, 23)` printed a spurious warning for valid avoiding
`_traj_dim = 2 + 4 = 6`.

---

## 3. Fix

### 3.1 `d3il/simulation/avoiding_sim.py`

Removed `record_context_info` call entirely:
```python
# Before (crash):
if hasattr(agent, 'record_context_info'):
    agent.record_context_info({}, int(context))

# After:
# avoiding has no box/target context — record_context_info expects a 4-tuple
# (box_pos, box_quat, target_pos, target_quat) which doesn't exist here; skip.
```

### 3.2 Both eval scripts — expert generation rewrite

```python
# Before (crash):
expert_path = expert_data['robot']['des_c_pos']
box_pos    = expert_data['push-box']['pos'][0]     # KeyError for avoiding
...
env.reset(random=False, context=context)
obs, _, _, _ = env.step(sim_action)
_, bp_img, ih_img = obs                            # wrong format for avoiding
frames.append(np.concatenate([bp_vis, ih_vis], axis=1))

# After (avoiding-compatible):
expert_path = expert_data['robot']['des_c_pos']
env.reset(random=True)          # no fixed context for avoiding
for step in range(len(expert_path)):
    sim_action = np.concatenate((expert_path[step][:3], [0, 1, 0, 0]))
    obs, _, done, _ = env.step(sim_action)
    try:
        bp_img = env.bp_cam.get_image(depth=False)   # single camera
        frames.append(cv2.cvtColor(bp_img, cv2.COLOR_BGR2RGB))
    except Exception:
        pass
    if done:
        break
```

### 3.3 Both eval scripts — `_traj_dim` guard

```python
# Before:
if _traj_dim not in (9, 23):

# After:
if _traj_dim not in (6, 9, 23):  # 6=visual avoiding, 9=visual aligning, 23=non-visual
```

---

## 4. Files touched

```
M  d3il/simulation/avoiding_sim.py                             (remove record_context_info call)
M  fm_visual_avoiding_test/eval_fm_visual_avoiding.py          (expert gen + _traj_dim)
M  diffuser_visual_avoiding_test/eval_visual_avoiding_dpcc.py  (expert gen + _traj_dim)
```

---

## 5. Verification (Phase-0 Docker)

| Check | Result |
|---|---|
| AST parse on all 3 files | ✅ |
| `record_context_info` call removed from `avoiding_sim.py` | ✅ |
| `push-box` / `target-box` keys removed from expert gen | ✅ |
| `_traj_dim not in (6, 9, 23)` in both eval scripts | ✅ |

**Cluster-side expectation**: rollouts run to completion. `update_rollout_info` is safe
(`info.get()` with defaults; `if ci:` skips empty context block). `check_trajectory_constraints`
with `constraint_types: []` (`no_constraint` variant) returns early without accessing
c_pos dim-2 index.

---

## 6. What `update_rollout_info` does with avoiding's info dict

Called with `{'mean_distance': float, 'success': bool, 'mode': 0|1}`:
- `info.get('final_box_pos')` → None → `if _fbp is not None and self.curr_context_info:` → skipped ✓
- `info.get('mean_distance', 0.0)` → distance to goal_ypos ✓
- `info.get('mode', 0)` → 0 or 1 ✓
- `if ci:` where `ci = {}` → falsy → box/target print block skipped ✓

---

## 7. Cross-references

| Document | Content |
|---|---|
| [`../Fix_4/CHANGELOG.md`](../Fix_4/CHANGELOG.md) | Previous fix (yaml redesign) |
| `d3il/simulation/avoiding_sim.py` | `Avoiding_Sim` — eval loop driver |
| `d3il/simulation/aligning_sim.py` | Source of `record_context_info` contract (4-tuple context) |
