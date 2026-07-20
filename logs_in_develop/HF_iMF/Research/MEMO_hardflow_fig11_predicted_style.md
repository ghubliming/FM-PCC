# MEMO — HardFlow's paper Fig. 11 styling: what ships, what doesn't, why we didn't use it

**Date:** 2026-07-20 · **Status:** memo / finding only — **not adopted**, no code depends on this.
**Prompted by:** the HardFlow paper's **Fig. 11**, *"Visualization of the generation process from HardFlow in the robotic manipulation task. Since the policy replans in a receding-horizon manner, we show one representative planning instance during execution."* — i.e. does upstream ship code we could reuse for our foresight-fan diagnostic?

---

## The finding

**Upstream ships the visual *vocabulary*, but not the figure *generator*.**

`hardflow/utils/rendering.py` → `AvoidingTrajectoryPlotter.plot_single_trajectory` supports **three** styles:

| style | look | used in repo? |
|---|---|---|
| `"actual"` | black solid, thick; green start / blue end dots | ✅ (our `eval_imf.py` fan) |
| `"multiple_real"` | black solid; green/blue dots; cyan intermediate | ✅ `save_single_trajectory_image` |
| **`"predicted"`** | **magenta, alpha 0.4; start square `ms`, end triangle `m^`, per-point `mx`** | ❌ **never called anywhere** |

Verified by grep: the only two call sites in the entire repo are `"multiple_real"` and (ours) `"actual"`. `"predicted"` appears **only in its own `elif` branch**. `notebooks/collect_results.ipynb` contains **no plotting code at all** (results table only).

**Interpretation:** `"predicted"` is almost certainly the style the authors used to render Fig. 11, with the driver script left out of the release. That is consistent with the rest of the picture — the planned trajectory and terminal prediction *are* computed at every replan but then **discarded** (`run/eval.py:393` unpacks them as `_, _`). See `DISCUSSION_foresight_fan_and_smoothness_paradigms.md` Part 1.

## Why we did NOT adopt it

It was wired up, rendered, and visually inspected — then reverted. Reason:

> **Fig. 11 shows "one representative planning instance." Our fan overlays 5–7 planning instances.**

The `"predicted"` style marks **every intermediate point** with an `mx` and both endpoints with square/triangle glyphs. That reads well for a single highlighted plan (the paper's use case) but becomes visual noise when several horizons are superimposed — the markers dominate and the shape of the fan is lost. Our diagnostic's whole purpose is comparing the *envelope* of many plans across methods, so thin unmarked grey lines communicate better.

**Kept instead:** grey thin lines for planned horizons + orange dashed for the x̂1 terminal predictions (a Gen13-specific object with no paper counterpart — for iMF it is the exact endpoint map `z + (1−τ)·u` versus FM's Euler shot).

## If someone later wants a paper-style Fig. 11 reproduction

The pieces are all present; it is a small job:
1. Pick **one** episode and **one** replan instant (Fig. 11 is a single planning instance).
2. `plotter.plot_single_trajectory(plan[:, action_dim:], ax, style="predicted")`
   — ⚠️ the slice matters: `plot_single_trajectory` reads x,y from **columns 2,3** (observation layout), while planned trajectories are full transitions with x,y at `action_dim+2,+3`. `plan[:, action_dim:]` lines them up.
3. Overlay the executed rollout with `style="actual"`, then `add_environment_elements` / `apply_legend` / `save_figure`.
4. The per-replan planned horizons are already captured by `_save_foresight_fan`'s buffers in `run/eval_imf.py` (enable with `IMF_PLOT_FAN=1`), so no new capture plumbing is needed.

## Takeaway

Nothing was lost by not reusing it — upstream had no generator to reuse, only a style, and that style is optimized for a different figure than ours. Recorded here so the question doesn't get re-investigated later.
