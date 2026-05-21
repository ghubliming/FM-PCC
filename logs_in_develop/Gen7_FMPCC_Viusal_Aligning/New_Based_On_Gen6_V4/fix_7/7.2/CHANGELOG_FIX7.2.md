# FIX-7.2 Changelog

**Date**: 2026-05-21  
**Branch**: `update_into_FM`  
**Root cause document**: `fix_7/7.2/PLAN_FIX7.2.md`  
**Status**: Implemented in both Gen7 and Gen6V4 eval scripts.

---

## What Was Fixed

`__RENDER_CTX_MAP` in `mj_render_singleton.py` is a module-level (process-global) dictionary that caches `RenderContextOffscreen` objects by camera name. When `generate_expert_reference()` runs, it creates a `Robot_Push_Env` whose cameras render during rollouts, populating:

```
__RENDER_CTX_MAP["rgbd_cage"]      = RenderContextOffscreen(expert_model, expert_data)
__RENDER_CTX_MAP["rgbd_rb0_rgbd"]  = RenderContextOffscreen(expert_model, expert_data)
```

`env.close()` never calls `reset_singleton()`. FIX-7 reset the body counter to 0, ensuring the variant's cameras use the same names (`rb0`) as the expert gen's cameras. This made things worse: the variant's `get_renderer("rgbd_cage", ...)` hit the cached entry and returned the stale context bound to `expert_model + expert_data`. Every `ctx.render()` call used `mjv_updateScene(expert_model, expert_data, ...)` — rendering the expert gen's robot pose — producing `bp_image std = 0.1978` and the wrong trajectory.

**Fix**: Call `reset_singleton()` (already provided in d3il) after expert gen and after each variant, clearing `__RENDER_CTX_MAP` so each env creates fresh render contexts.

---

## Changes Applied

### `fm_visual_aligning_test/eval_fm_visual_aligning.py` (Gen7)

**1. Pre-loop reset** — inserted after FIX-7 counter reset, before `for variant in projection_variants:`:

```python
# FIX-7.2: Clear the process-global render context cache...
try:
    from environments.d3il.d3il_sim.sims.mj_beta.mj_utils.mj_render_singleton import (
        reset_singleton as _reset_render_singleton,
    )
    _reset_render_singleton()
    print('[ expert ] Render singleton cache cleared (FIX-7.2)')
except Exception as _e:
    print(f'[ expert ] WARNING: Render singleton reset failed: {_e}')
```

**2. Per-variant finally block** — after FIX-7 counter reset in `finally:`:

```python
# FIX-7.2 (per-variant): Clear render context cache so next variant
# creates fresh RenderContextOffscreen objects.
try:
    _reset_render_singleton()
except NameError:
    pass
```

### `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` (Gen6V4)

Identical changes applied at the same structural locations.

---

## Expected Log Markers

```
[ expert ] MjRobot.GLOBAL_MJ_ROBOT_COUNTER reset to 0 (FIX-7)
[ expert ] Render singleton cache cleared (FIX-7.2)
...
[ DIAG img ] bp_image   std=0.2093   ← target
[ DIAG img ] inhand_img std=0.2490   ← target
```

Note: 2 `mju_openResource` warnings will still appear when expert gen ran — this is expected and benign (MuJoCo internal resource tracking, does not affect correctness).
