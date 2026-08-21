# Gen15 Fix_2 — `constraint_overview*.png` framed the wrong axis (corridor y)

**Date:** 2026-08-21
**Severity:** 🟡 cosmetic — **display only. No evaluated number, no projector behaviour, and no DA conclusion is affected.**
**Reported by:** visual inspection of
`logs/UAV_MIX/uav-corridor/plans/mix_uav_mf/.../Emf_K1_.../6/corridor_.../constraint_overview_tightened.png`
— "the objects in the plots are all shifting downwards".
**Files changed:** `mix_uav_test/eval_mix_uav.py` (Gen15) · `FM_v3_uav_test/eval_fm_uav.py` (Gen11 sibling)

---

## 1. Symptom

In the corridor constraint schematics, everything drawn — the two wall halfspaces, the four
wall-end cap balls — sat crushed against the bottom edge of the XY and 3D panels, with the blue
workspace rectangle floating above them over empty space. The lower wall and two of the four
balls were **clipped out of frame entirely**. The XZ panel looked correct.

Both the base and `_tightened` twins were wrong, in `.png` and `.svg` alike — four files per
geo folder.

## 2. Root cause

`plot_geo_constraints`, one line:

```python
lb_d[np.isinf(lb_d)] = _Z_DISP[0]; ub_d[np.isinf(ub_d)] = _Z_DISP[1]
```

`_Z_DISP = (0.0, 2.0)` is the **z** flight-band fallback for a scene that does not constrain
altitude. But this is a boolean mask over the whole 3-vector, so it stamps that z band onto
**any** infinite axis.

`config/uav_projection.yaml:160` declares corridor's y as unbounded *on purpose* —
"x-extent + altitude; y handled by the wall halfspaces":

```yaml
workspace_bounds:
  lb: [-3.2, -.inf, 0.30]
  ub: [ 3.2,  .inf, 1.80]
```

so corridor's y display range became `[0.0, 2.0]`:

| corridor, margin 0.330 | intended | actual (bug) |
|---|---|---|
| displayed y-box | −2.00 … 2.00 | **0.00 … 2.00** |
| `_ylim()` = (lb−0.3, ub+0.3) | (−2.30, 2.30) | **(−0.30, 2.30)** |
| lower wall y = −0.45 | in frame | **clipped** |
| cap balls y = −0.5 | in frame | **clipped** |

The three-line loop immediately below it was written to handle exactly this case:

```python
for _i, _fallback in ((0, (-3.5, 3.5)), (1, (-2.0, 2.0))):
    if np.isinf(lb_d[_i]): lb_d[_i] = _fallback[0]
```

but it was **dead code** — after the mask nothing is infinite any more, so `np.isinf` can never
fire. The bug and its intended guard shipped together.

### 2.1 Blast radius: corridor only, y only

Checked all three UAV scenes in `config/uav_projection.yaml`:

| scene | x | y | z | affected |
|---|---|---|---|---|
| corridor | ±3.2 | **±inf** | 0.30 / 1.80 | 🔴 **yes (y)** |
| pillars | ±3.6 | ±1.5 | 0.30 / 1.80 | no |
| s_curve | ±3.6 | ±1.6 | 0.30 / 1.80 | no |

No scene has an infinite x or z, so `_Z_DISP` and the x fallback never fired for anything else.

### 2.2 Why no evaluated number is affected

`plot_geo_constraints` is a standalone schematic writer. The projector builds its rows in
`setup_dpcc_projector` from the raw config:

```python
lb = np.concatenate([np.full(6, -np.inf), ws_lb + margin, np.full(pad, -np.inf)])
```

`-inf + 0.33 = -inf`, which SLSQP reads as unbounded — correct, and the intended semantics.
The display fallback is never anywhere near it. Every S&C, violation, and timing figure in
`DA_20260820_fm_K_sweep_corridor.md` stands unchanged.

### 2.3 A correct implementation already existed

`mix_uav_test/eval_artifacts.py:293-296` (`draw_projector_geometry`, the per-rollout overlay)
already had the right per-axis form, with all three axes listed:

```python
for _i, _fb in ((0, (-3.6, 3.6)), (1, (-2.0, 2.0)), (2, (0.0, 2.0))):   # clamp ±inf for display
    if np.isinf(lb_d[_i]): lb_d[_i] = _fb[0]
```

So the two `eval_*_uav.py` copies were the stale ones. The fix aligns them to it.

## 3. What changed

Both files, identical edit:

1. **Per-axis clamp, all three axes** — replaces the boolean mask and absorbs the dead loop into
   one live pass. `_Z_DISP` is now bound to index 2 only, where it belongs.
2. **`halfspace_list` / `obstacle_list` moved above the bounds block** — needed by (3).
3. **Unbounded axes are now framed from the real geometry** (new `_geo_extent(axis)` helper):
   the min/max over every drawn halfspace endpoint and inflated obstacle on that axis, padded
   0.35 m. The old constants remain as a last resort for an axis with no geometry at all.
   This is a *legibility* improvement, not part of the bug — a ±2 m frame around corridor's
   ±0.45 walls would have been correct but poor.
   `_geo_extent` deliberately uses raw halfspace endpoints rather than `_wall_xy`: `x_active`
   only shortens a segment, so raw endpoints stay a superset, and it avoids a forward reference
   to a closure defined further down.

Resulting frames (all three scenes, both margins, verified by simulation):

| scene | y-box before | y-box after | all y-features in frame |
|---|---|---|---|
| corridor (base) | `[0.00, 2.00]` 🔴 | `[-1.23, 1.23]` | ✅ |
| corridor (tightened) | `[0.00, 2.00]` 🔴 | `[-1.25, 1.25]` | ✅ |
| pillars | `[-1.17, 1.17]` | `[-1.17, 1.17]` (unchanged) | ✅ |
| s_curve | `[-1.27, 1.27]` | `[-1.27, 1.27]` (unchanged) | ✅ |

## 4. Verification

- `ast.parse` clean on both files.
- No boolean-mask `isinf` assignment remains in live code (only the two explanatory comments).
- `halfspace_list` / `obstacle_list` each assigned exactly once per file — no duplicate left behind.
- Pure-Python simulation of the new clamp against the real `uav_projection.yaml` geometry for all
  three scenes × both margins: every wall endpoint and obstacle centre lands inside `_ylim()`.
- **Not run on the cluster.** Regeneration of the actual PNGs is pending — run on cluster.

## 5. Regenerating the figures

No re-evaluation is needed; the schematic is a pure function of the config. But
`plot_geo_constraints` is **idempotent** —

```python
if os.path.exists(out_png) and os.path.exists(out_svg):
    return
```

— so the stale files must be deleted first or nothing will be rewritten:

```bash
find logs/UAV_MIX -name 'constraint_overview*.png' -delete
find logs/UAV_MIX -name 'constraint_overview*.svg' -delete
```

They are then rewritten by the next eval that touches each geo folder.

## 6. Audit of similar bugs

Swept every `isinf` in live code (excluding `Archived_Codes/` and `*(legacy)` folders).

| pattern | where | verdict |
|---|---|---|
| boolean-mask clamp over all axes | `mix_uav_test/eval_mix_uav.py`, `FM_v3_uav_test/eval_fm_uav.py` | 🔴 **the bug — both fixed** |
| per-axis clamp, 3 axes listed | `mix_uav_test/eval_artifacts.py`, `FM_v3_uav_test/eval_artifacts.py` | ✅ correct (reference impl) |
| `lb_d[2] = _Z_DISP[0] if np.isinf(lb_d[2]) else lb_d[2]` | all 6 visual-aligning/avoiding copies (`fm_`, `imf_`, `mix_`, `diffuser_`, `plot_yaml_constraints.py` twins) | ✅ correct — indexes z explicitly, and those scenes are 2-D with finite x/y |
| `np.where(np.isinf(...), ±1e9, ...)` | visual-aligning projector setup | ✅ correct — deliberate solver-side ±inf → large-finite substitution, whole-vector by design |
| `_z0 = 0.0 if np.isinf(_z0) else _z0` and friends | visual-aligning rollout overlays | ✅ correct — scalar z only |

**No other instance of this defect exists in live code.** The visual-aligning lineage was never
exposed: it indexes `[2]` explicitly, and none of its scenes declare an infinite x or y.

## 7. Follow-ups (not done)

- The x fallback constant differs between the two implementations — `(-3.5, 3.5)` in the old
  `plot_geo_constraints` vs `(-3.6, 3.6)` in `eval_artifacts.py`. Fix_2 adopts `±3.6` in both so
  they agree. Neither is reachable with the current scene set.
- `plot_geo_constraints` is now ~95% duplicated between `eval_mix_uav.py` and `eval_fm_uav.py`,
  and overlaps `eval_artifacts.draw_projector_geometry`. Per the repo's copy-modify sibling
  convention this is left alone deliberately — noted so the next edit is mirrored to both.
