# Fix 4 — Non-Visual Mode: Eval Ignores `if_vision` Flag (2026-05-19)

## Problem

Codex correctly identified that `plan_visual_aligning_dpcc` had no `if_vision` flag.
The deeper consequence (beyond path naming): **even if you trained with `if_vision: False`,
eval always ran in visual mode** — loading images, running camera pipelines, and then crashing
or producing garbage when the non-visual model tried to process them.

Three bugs caused this:

---

## Bug 1 — `Aligning_Sim(if_vision=True)` hardcoded in eval (PRIMARY)

`eval_visual_aligning_dpcc.py` line 744 was:
```python
sim = Aligning_Sim(..., if_vision=True, ...)
```

`if_vision=True` tells D3IL to:
- Initialize camera sensors and load image streams
- Call `agent.predict((bp_image, inhand_image, des_robot_pos), if_vision=True)` every step

So even a checkpoint trained with `if_vision: False` would receive images at eval.
The `VisualUNet` ignores them (its `if_vision=False` guard skips `encode_visual`), but
the sim still pays the full image loading cost and the `predict()` code built visual cond
unnecessarily.

**Fix:**
```python
# Before
sim = Aligning_Sim(..., if_vision=True, ...)

# After
sim = Aligning_Sim(..., if_vision=getattr(args, 'if_vision', True), ...)
```

---

## Bug 2 — Non-visual `predict()` path left `cond = None` → crash

When `Aligning_Sim(if_vision=False)`, D3IL calls `agent.predict(obs)` (no images in state).
The `predict()` method had no `else:` branch — `cond` was never set, leaving `cond = None`.

Then `self.model(None)` → `VisualGaussianDiffusion.forward(None)` →
`apply_conditioning(x, None, action_dim)` → `AttributeError: 'NoneType'.items()` → **crash**.

**Fix:** Added `else:` branch to `predict()` that:
- Parses `state` as D3IL's concatenated obs vector (`robot_pos` at `[:3]`)
- Builds normalized `obs_6d = [des_robot_pos | des_robot_pos]`
- Builds `cond = {0: obs_anchor}` — plain obs-anchor dict (no `'visual'` key)
  → `VisualGaussianDiffusion.forward()` passes it through unchanged
  → `apply_conditioning(x, {0: obs_anchor}, action_dim)` snaps obs at `t=0` ✅
  → `VisualUNet` ignores the missing `'visual'` key, passes `visual_cond=None` to backbone ✅

---

## Bug 3 — `mental_robot_pos.copy()` crash when non-visual

Line 583 (before fix):
```python
if if_vision:
    self.mental_robot_pos += next_action_np   # only visual path updated it
...
self.last_predicted_pos = self.mental_robot_pos.copy()   # unconditional → None.copy() crash
```

If `if_vision=False`, `mental_robot_pos` was never initialized → `None.copy()` → crash.

**Fix:** Both paths now initialize `mental_robot_pos` from `des_robot_pos_np` inside their
respective branches. The update and copy are unconditional:
```python
self.mental_robot_pos += next_action_np
self.history_desired_actions.append(next_action_np.copy())
self.last_predicted_pos = self.mental_robot_pos.copy()
```

---

## Bug 4 (config) — `plan_visual_aligning_dpcc` had no `if_vision` key

Without `if_vision` in the plan block:
- `getattr(args, 'if_vision', True)` defaults to `True` → visual mode even for non-visual checkpoints
- `V{if_vision}` path tag resolves to empty string → checkpoint directory mismatch

**Fix (in working tree, not yet committed):** Added `'if_vision': True` to
`plan_visual_aligning_dpcc`, added `('if_vision', 'V')` to both `args_to_watch` lists,
updated `prefix` and `diffusion_loadpath` templates to include `V{if_vision}`.

> **Breaking change**: checkpoints trained before this fix have paths without `V` tag.
> Rename them on cluster or set `if_vision` lookup manually:
> ```bash
> mv visual_aligning_dpcc/H8_K100_D..._aw10_steps1000  \
>    visual_aligning_dpcc/H8_K100_D..._aw10_VTrue_steps1000
> ```

---

## Files Changed

| File | Change |
|------|--------|
| `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py` | Bug1: `if_vision=getattr(args,'if_vision',True)` in Aligning_Sim; Bug2: non-visual `else:` branch in `predict()`; Bug3: unconditional `mental_robot_pos` update |
| `config/aligning-d3il-visual.py` | Bug4: add `if_vision: True` to plan block + `V{if_vision}` path tags (in working tree) |

---

## D3IL State Format Reference

| `if_vision` | D3IL call | `state` format |
|-------------|-----------|----------------|
| `True` | `agent.predict((bp_img, inhand_img, des_pos), if_vision=True)` | tuple of 3 |
| `False` | `agent.predict(obs_np)` | flat array, `[:3]` = robot_pos |
