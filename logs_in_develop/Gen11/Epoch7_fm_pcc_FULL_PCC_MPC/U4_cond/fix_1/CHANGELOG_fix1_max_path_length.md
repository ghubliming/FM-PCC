# CHANGELOG — U4 fix_1: per-scene max_path_length

**Date**: 2026-06-28
**Parent**: [../CHANGELOG_U4_cond.md](../CHANGELOG_U4_cond.md)

---

## Problem

`config/uav.py` had a single `max_path_length: 750` for ALL scenes, justified by a comment
that s_curve reaches 22 s × 33 Hz = 726 steps. But each scene has a different max duration,
so the other scenes were massively over-allocating the replay buffer:

| Scene    | Max dur (generator.py) | Max steps @33 Hz | Old buffer | Wasted rows/ep |
|----------|------------------------|------------------|------------|---------------|
| empty    | ≈12.8 s (dist/0.4)     | 421              | 750        | ~329          |
| corridor | 10.0 s                 | 330              | 750        | ~420          |
| s_curve  | 22.0 s                 | 726              | 750        | ~24 ✓         |
| pillars  | 16.0 s                 | 528              | 750        | ~222          |
| all      | bounded by s_curve     | 726              | 750        | ~24 ✓         |

---

## Fix

### `config/uav.py`
Added `MAX_PATH_LENGTH_PER_SCENE` dict (module-level constant, above the `base` block):

```python
MAX_PATH_LENGTH_PER_SCENE = {
    'empty':    450,   # max dur ≈ 12.8 s → 421 steps
    'corridor': 360,   # max dur = 10.0 s → 330 steps
    's_curve':  750,   # max dur = 22.0 s → 726 steps
    'pillars':  560,   # max dur = 16.0 s → 528 steps
    'all':      750,   # pooled, bounded by s_curve
}
```

Updated inline comment on the `max_path_length` key in `base`:
```python
'max_path_length': 750,  # fallback (scene='all', bounded by s_curve); per-scene: see MAX_PATH_LENGTH_PER_SCENE above
```

### `FM_v3_uav_test/train_fm_uav.py`
After `args = parser_obj.parse_args(...)`, before `torch.manual_seed`:

```python
from config.uav import MAX_PATH_LENGTH_PER_SCENE
resolved_max_path = MAX_PATH_LENGTH_PER_SCENE.get(cli_args.scene, args.max_path_length)
if resolved_max_path != args.max_path_length:
    print(f'[ train ] max_path_length: {args.max_path_length} → {resolved_max_path} (scene={cli_args.scene})')
    args.max_path_length = resolved_max_path
```

The scene key comes from `cli_args.scene` (already parsed at top-level). If the scene isn't
in the dict, falls back to the config value (750) — safe for any future scene.

---

## What does NOT change

- Eval code is unaffected (`max_path_length` is a training/dataset concern only).
- Scene `'all'` and `'s_curve'`: behavior byte-identical (both resolve to 750).
- The checkpoint path / `args_to_watch` folder names: unaffected (scene is already in the
  dataset string `uav-<scene>`, not a separate watch key).
- Default `max_path_length: 750` in the config block is still the fallback for unknown scenes.

---

## Verification

- `py_compile` passes on both changed files.
- Buffer sizes cross-checked against `generator.py:_build_traj_and_init` durations and
  `dataset_writer.py:DATASET_HZ=33`.

---

## How to revert

```bash
git checkout -- config/uav.py FM_v3_uav_test/train_fm_uav.py
```
