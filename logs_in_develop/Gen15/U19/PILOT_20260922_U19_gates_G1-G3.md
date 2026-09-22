# PILOT — Gen15 U19 · gates G1–G3 · `corridor_v3_tilt` + `corridor_v3_ablation_hump` (2026-09-22)

Jobs 26071 (ablation hump, tag `u19smokecv3ah`) and 26072 (tilt, tag `u19smokecv3t`): mf, seed 6, K = 3, 3 trials
(L, C, R), arms `diffuser` / `dpcc-t-bounds_free-pdes-tightened` / `hardflow_sls-t-bounds_free-pdes-tightened`.
Data: `temp/22-09/FOR_U19/` (npz + results.json + logs; gitignored), pictures copied to [`figs/pilot/`](figs/pilot/).
Gates read with `tools/check_gates_u19.py --geo tilt|hump` (its npz path was fixed on this data: the eval writes
`<geo>/<variant>/<variant>.npz`). Residual violations located with the scorer's own rule, per family, per step.

## 0 · Verdict in one paragraph

**Both scenes work mechanically, and the projector moves z on demand in both — the thing U11–U13 never got.** Under
the tilt every projected flight descends 0.15–0.29 m below the unprojected one at the exit; over the hump every
projected flight climbs 0.30–0.40 m. The violation rate drops 10–15×. **Gate G2 fails on every projected arm** —
collision-free 0/3 — but the residue is one thing only: **6–11 shallow steps (≤ 2–6 cm) at the geometric switch point,
with the setpoint clean at every one of them.** The plan obeyed; the plant, 0.5 m behind its setpoint, clipped the
surface where the constraint hands over (the tilt's window end at x = 2, the hump's apex at x = 0). Same mechanism as the
U16 `-pdes` lag note, now measured on z.

## 1 · Console

Both logs clean: no traceback, the geometry banner reads `hs=3` (tilt) / `hs=4` (hump), scene `scene_corridor_v2.xml`,
`UAV_MIX_GEO_VARIANTS` honoured, 3/18 variants run, `Fix_16` degenerate-channel notice as always, HardFlow's K = 3 thin
warning as always.

## 2 · Gates

| | tilt `diffuser` | tilt `dpcc-t` | tilt `hf-t` | hump `diffuser` | hump `dpcc-t` | hump `hf-t` |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| collision-free | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 | 0/3 |
| success strict / relaxed | 3/3 / 3/3 | 3/3 / 3/3 | 3/3 / 3/3 | 3/3 / 3/3 | 3/3 / 3/3 | 3/3 / 3/3 |
| violating steps per flight | 67 / 115 / 120 | 7 / 8 / 8 | 9 / 8 / 9 | 38 / 31 / 33 | 6 / 7 / 7 | 11 / 11 / 11 |
| violations per step | 0.374 | **0.026** | 0.030 | 0.126 | **0.021** | 0.035 |
| steps | 269 | 295 | 291 | 269 | 312 | 317 |
| goal distance at the end | 0.29–0.30 | 0.29–0.30 | 0.29–0.30 | 0.29–0.30 | 0.29 | 0.28–0.29 |
| **G1** plane binds unprojected | **PASS** 3/3 | | | **PASS** 3/3 | | |
| **G2** collision-free ≥ 2/3 ∧ success ≥ 2/3 | | ❌ 0/3 cf | ❌ 0/3 cf | | ❌ 0/3 cf | ❌ 0/3 cf |
| **G3** Δz projected − unprojected, per trial (L, C, R) | | −0.18 / −0.29 / −0.29 **PASS** | −0.15 / −0.26 / −0.23 (trial L at −0.146, 4 mm short of the −0.15 bar) | | +0.40 / +0.30 / +0.33 **PASS** | +0.40 / +0.31 / +0.35 **PASS** |

Strict success is 3/3 everywhere: the goal distance at the end is 0.28–0.30 m, inside the 0.30 m ball even after the
descent / climb (the drone is back near the route altitude by x = 2.8).

## 3 · Where the residual violations are

Every violating step of every projected flight, family and depth from the scorer's rule (`r_drone` 0.31, raw geometry):

| scene · arm | trial | bad steps | x-range | max depth | family | setpoint depth at the worst step | \|p − p_des\| there |
| :-- | :-- | :-- | :-- | :-- | :-- | :-- | :-- |
| tilt · dpcc-t | L / C / R | 7 / 8 / 8 | **[1.83, 2.00]** | 0.033 / 0.045 / 0.047 | leaned plane only | **0.000** | 0.49–0.52 |
| tilt · hf-t | L / C / R | 9 / 8 / 9 | **[1.80, 1.99]** | 0.059 / 0.051 / 0.055 | leaned plane only | **0.000** | 0.51–0.54 |
| hump · dpcc-t | L / C / R | 6 / 7 / 7 | **[−0.04, 0.07]** | 0.021 / 0.032 / 0.029 | roof only | **0.000** | 0.53–0.59 |
| hump · hf-t | L / C / R | 11 / 11 / 11 | **[−0.04, 0.13]** | 0.032 / 0.032 / 0.028 | roof only | **0.000** | 0.55–0.57 |

Unprojected, for scale: the tilt is violated over x ∈ [−1.5, 2.0] up to **0.40 m** deep, the hump over
x ∈ [−0.5, 0.6] up to **0.34 m**. The projected residue is one segment per flight, 2–6 cm, at the exit / apex.

**Reading.** The geometry is bound to the setpoint (`-pdes`), which the drone trails by ~0.5 m along the route.

- *Tilt, window end.* The plane is live for `x_active` ∈ [−2, 2], gated on the setpoint's x. When the setpoint crosses
  x = 2 the plan is free and starts back toward the route altitude; the drone, still at x ≈ 1.8, follows it up and
  clips the plane by 3–6 cm over its last 0.2 m. Below the surface the drone tracks the centre limit closely
  (`figs/pilot/cv3t_dpcc-t-…png`, side panel). This is the "HardFlow loses 1–2 mm at the corridor exit" of U16, larger
  here because the leaned plane's deepest demand is exactly at the window end.
- *Hump, apex.* The setpoint clears the tightened roof (≥ 1.515 m) and the drone reaches 1.44–1.47 m at x = 0: a
  2–5 cm vertical tracking lag at the top of a 0.4 m climb, while the setpoint is already descending the far side.

Both are **plant lag under a setpoint-bound constraint**, not a projector failure: the setpoint is on the feasible
side at every violating step of every flight.

## 4 · What this means for G4

The pilot's job was to prove the machinery and the mechanism: **done, on both scenes**. The residue is a 2–6 cm
tracking effect that the wave will measure across engines and K (K = 5 tracks better in v2; DPCC vs HardFlow differ
already here: 7–8 vs 9–11 steps). Two ways to run the wave, the author's call:

| option | what | cost | comparability |
| :-- | :-- | :-- | :-- |
| **A · as is** | `submit` both geometries now; the DA reports collision-free, violations per step and depth, as the U16 DA did for fm's 1.5–2.3 residual steps | 2 GPU-days | byte-identical stack to v2 (`margin_base 0.0`) |
| **B · padded planner** | `planning_inflation.margin_base` 0.06 on the two v3 entries (the U7 knob: pads the *planner*, the scorer keeps 0.31) → the 2–6 cm lag is inside the pad; re-pilot first (2 × 20 min), then `submit` | +40 min, then 2 GPU-days | v3 arms share the pad; v2 has none — say so in the table |

**Decision (author, 2026-09-22): A.** *"If it is because the model / projector is not good, i.e. not an env problem, no
need to fix the env for the model / projector."* The residue is the plant trailing a feasible setpoint — the system under
test, not the scene — so the scene stays byte-identical to v2 and the wave measures it as it is. B is not run.

The exact jobs — **superseded 2026-09-23 by the paper driver** (runbook `SLURM_RUNBOOK_20260923_uav_corridor_v3_pillars_v2.md`
§1–§2 variant lists, 68 cells per scene, tags `p23cv3t` / `p23cv3ah`, both scenes, author: "two scenes as piloted"):

```bash
bash Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3_master.sh plan      # what each link submits
bash Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3_master.sh start     # the CHAIN: C1 finishes → C2 starts → … → C5 (author: not the whole grid in squeue)
bash Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3_master.sh status
# or, everything at once:  bash Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3.sh submit all   (36 submissions / 62 eval jobs / 136 cells)
```

(The 22-09 driver's `submit` — U16 layout, tags `u19cv3*` — is kept for reference only; do not run both.)

## 5 · Small things seen

- `check_gates_u19.py` fixed for the real layout (`<geo>/<variant>/<variant>.npz`); nothing else in the tools changed.
- HardFlow trial L under the tilt: Δz −0.146 m against the −0.15 m bar. A bar, not a mechanism; it descends on all three.
- The unprojected tilt flights bind where G0 predicted (L from x ≈ 0.0 flying low at 1.03 m, C from −1.23, R from −1.50) and
  their exit depths (0.23 / 0.38 / 0.40 m) match the G0 estimate (0.28 / 0.34 / 0.39 at the launch altitude).
- The folder also holds unrelated logs of the same day (U18 turbo 26067 / 26073, aligning/avoiding training 26051–26053,
  `da_uav_manual.log`); not read for this note.

---
Claude (Fable 5.1, Claude Code) · 2026-09-22 · analysis local (numpy), nothing executed on the cluster.
