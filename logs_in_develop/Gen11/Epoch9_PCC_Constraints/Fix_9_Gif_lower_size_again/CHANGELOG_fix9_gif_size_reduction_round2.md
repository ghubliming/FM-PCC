# Epoch 9 Fix_9 — lower UAV GIF size further (round 2)

**Date:** 2026-07-06. Follow-up to Fix_7 — user reported the GIFs still feel too big after the
first pass. Pushed the same two levers (resolution, frame count) further, plus the palette.

## What changed

| Knob | Fix_7 | Fix_9 (now) | Effect |
|---|---|---|---|
| `_make_overhead_renderer` resolution | 200×200 | **140×140** | (140/200)² ≈ 0.49 → another ~2x fewer pixels/frame |
| `rollout_one`'s `frame_stride` | 2 | **3** | 1/3 fewer frames captured → ~1.5x fewer frames |
| `save_rollout_gif`'s `palettesize` | 128 | **64** | smaller GIF color table |
| `fps` | 10 | 10 (unchanged) | playback speed, not a size lever on its own |

Combined pixel+frame reduction vs. Fix_7: roughly `(140/200)² × (2/3) ≈ 0.33` — GIFs should be
about a third the size they were after Fix_7, and roughly **10-15x smaller** than the original
pre-Fix_7 baseline (360px, stride 2, palette 256).

## Why these three and not others

- **Resolution** is still the dominant lever (quadratic in pixel count) and this render is
  still purely a debug/visualization artifact — never fed to a policy (unchanged reasoning
  from Fix_7). 140px lands close to the arm's own 96px vision-input scale while staying
  legible for a multi-obstacle overhead scene (walls, pillars, drone, candidate-fan lines).
- **`frame_stride`** trades smoothness for size linearly — bumped one more notch (2→3) rather
  than more aggressively, to avoid the GIF becoming visibly choppy/hard to follow.
- **`palettesize`** halved again (128→64) — a UAV overhead scene is a handful of flat colors
  (steelblue floor, tomato pillars, colored trajectory lines), not a photo, so 64 colors is
  still ample; diminishing returns below this without visible banding.
- **`fps` left alone** — changing it doesn't shrink the file (frame *count* already controlled
  by `frame_stride`); it only changes perceived playback speed, which isn't what was asked.

## Honest tradeoff
At 140×140 with every 3rd frame, fine detail (individual candidate-fan lines, exact wall
thickness) will read blockier and motion will be visibly less smooth than before Fix_7. This
is an explicit quality-for-size tradeoff, same as Fix_7 — flagging in case it goes further
than wanted; easy to dial back (all three numbers are simple constants) if so.

## Verification
- `py_compile` clean on `eval_fm_uav.py` and `eval_artifacts.py`.
- Confirmed `frame_stride` has no caller override (default flows through unchanged from
  `rollout_one`'s signature — grepped for all `frame_stride` occurrences in the file).
- Could not execute `imageio.mimsave` here (not installed in this Docker shell, cluster-only
  runtime, same caveat as Fix_7) — `palettesize=64` is a plain reduction of the same
  already-defensive (`try`/`except TypeError`-guarded) call from Fix_7, so no new risk
  introduced. **Confirm visually on the next cluster `--record gif` run.**

## Files touched
- `FM_v3_uav_test/eval_fm_uav.py` — `_make_overhead_renderer` default resolution 200→140;
  `rollout_one`'s `frame_stride` default 2→3.
- `FM_v3_uav_test/eval_artifacts.py` — `save_rollout_gif`'s `palettesize` 128→64.
