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

## ✅ UPDATE — the style IS used, for the right figure: iMF vs FM comparison (u_8)

Follow-up question: *could* we use this to generate an **iMF vs original FM** comparison? **Yes — and it is the correct use.** Built as **u_8** (`../../Gen13/u_8/`).

**The crucial distinction that resolves the earlier objection:**

| Figure | Plans drawn | `"predicted"` style verdict |
|---|---|---|
| Foresight **fan** (fix_7) | 5–7 overlaid horizons | ❌ per-point markers become clutter |
| **Fig.11 comparison** (u_8) | **ONE** representative planning instance per panel | ✅ **clean — the style's native use case** |

The paper's own words — *"we show one representative planning instance"* — describe exactly the u_8 figure, not the fan. So the style was not "not ideal"; it was being applied to the wrong figure. Both now coexist: fan keeps clean grey lines, the comparison uses upstream `"predicted"`.

**What u_8 produces:** a two-panel side-by-side, **iMF (average velocity `u`)** vs **FM (instantaneous velocity `v`)**, each showing one planning instance in the paper's magenta `"predicted"` style + the executed rollout in `"actual"`, on identical axes/obstacles. Plus an orange dashed **x̂1** overlay — Gen13-specific, no paper counterpart: for iMF that curve is the *exact* endpoint map `z + (1−τ)·u`, for FM the Euler shot `z + (1−τ)·v`. **The seam swap made visible, side by side.**

**Implementation notes** (all confirmed working, rendered and inspected):
- `run/eval_imf.py` now also dumps `{run_id}_fan.npz` (raw planned horizons, x̂1, rollout) whenever fan capture is on — so the comparison is **pure post-processing**: no GPU, no simulator, no model reload.
- `run/make_fig11_comparison.py` assembles the panels via the plotter's `_configure_axis(..., compact=True)` + `add_environment_elements`.
- ⚠️ the index slice still matters: `plot_single_trajectory` reads x,y from **columns 2,3** (observation layout) while plans are full transitions — hence `plan[:, action_dim:]`.
- Representative instance defaults to the **middle** replan (`--plan_idx` to override); `--run_id` picks the episode.

## Takeaway

Upstream shipped a style but no generator. The style is **wrong for the fan** and **right for a single-instance comparison** — so it is now used for exactly the latter (u_8), and the fan keeps its own styling (fix_7). Recorded so neither decision gets re-litigated.
