# Changelog — UF-13: Non-Visual Mode GIF Fix

**Date:** 2026-05-26  
**Branch:** update_into_FM  
**Reference:** `UF13_nonvisual_gif_investigation.md` (same directory)  
**Scope:** Auto-enable visual mode when `--record` is active, in both FM and DPCC visual-aligning eval scripts

---

## Summary

Non-visual mode (`if_vision=False` in config) produces no GIFs because frame capture is gated behind `if if_vision:` in `VisualAgentWrapper.predict()`. Investigation (UF-13) concluded no standalone replay script is needed. Fix: when `record_mode != 'none'`, the eval script automatically promotes `if_vision` to `True` so camera frames are always captured — no manual flag required.

No changes to sbatch scripts beyond updating the comment block.

---

## Changed Files

### 1. `fm_visual_aligning_test/eval_fm_visual_aligning.py`

**Updated `Aligning_Sim` instantiation** (inside variant loop):

Before:
```python
sim = Aligning_Sim(..., if_vision=getattr(args, 'if_vision', True), ...)
```

After:
```python
_if_vision_config = getattr(args, 'if_vision', True)
if_vision = _if_vision_config
if not if_vision and args_cli.record != 'none':
    if_vision = True
    print('[ eval ] WARNING: config if_vision=False but record_mode is active → '
          'auto-enabling visual mode so GIFs/videos are captured (UF-13).')

sim = Aligning_Sim(..., if_vision=if_vision, ...)
```

### 2. `diffuser_visual_aligning_test/eval_visual_aligning_dpcc.py`

Identical change to above.

### 3. `Slurm_Codes/sbatch/fm_visual_aligning/eval_fm_visual_aligning.sh`

Updated comment block only — added note that the eval script auto-enables visual mode when `record_mode != 'none'`.

### 4. `Slurm_Codes/sbatch/diffuser_visual_aligning/eval_visual_aligning_dpcc.sh`

Identical comment update to above.

---

## Behaviour Summary

| Config `if_vision` | `--record` | `if_vision` used | GIFs produced |
|---|---|---|---|
| `True` | any | `True` | Yes (unchanged) |
| `False` | `none` | `False` | No (recording off, no change) |
| `False` | `gif`/`video`/`all` | `True` (auto) | Yes + warning printed |

---

## Not Changed

- No new scripts created (UF-13 conclusion)
- No `--if-vision` CLI flag added (user: "makes no sense")
- `VisualAgentWrapper.predict()` logic untouched — auto-enable is purely at the `Aligning_Sim` call site
- sbatch Python invocations unchanged (no new args passed)
