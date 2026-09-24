# TO v3 — UAV-s-curve is complete: Table 6.16's last row, Figure 6.9 rebuilt, and what the finished section says

**2026-09-24 · from the DA side (R44 run chat).** Nothing in `v3/` was touched.
- **Status:** every s-curve cell of Chapter 6 is flown and analysed.
- **Analysis of record:** `Data_Analysis/DA_in_Paper/analysis/DA_20260924_scurve_R44bc_projection_controller.md`, now
  complete. It supersedes the "preliminary" state of my earlier R44b note, whose Table 6.15 and first three Table 6.16 rows
  and whose §3 correction all stand.
- **C1:** job 26204, clean, md5-verified.

## 1 · Table 6.16 `tab:uav-controller`: the complete body (FM, nfe 1, ten flights per cell)

```latex
    none      & cascaded geometric & 9/10 & $23.0 \pm 5.4$   & 0.571 & 1/10 \\
    none      & MuJoCo MPC         & 9/10 & $41.5 \pm 18.4$  & 0.297 & 0/10 \\
    \addlinespace[3pt]
    per-step  & cascaded geometric & 5/10 & $71.3 \pm 66.6$  & 0.572 & 3/10 \\
    per-step  & MuJoCo MPC         & 0/10 & $105.5 \pm 46.0$ & 1.626 & 7/10 \\
```
- **Dataref:** C0 = job 26196, C1 = job 26204, tag `p23scmjpc`, `UAV_MIX_CONTROLLER=mjpc`. The per-step rule is random,
  chosen from Table 6.15: a tie at 5/10, broken by fewer aborted flights, then fewer violating steps.
- The "Every cell is pending" sentence and the `pending` cells go.

## 2 · What the finished table says (storyline: the controller on extreme trajectories)

1. **The limit of the tracking controller is stability at the second turn.** On the unprojected plans, MuJoCo MPC inverts
   on none of ten flights (cascaded geometric: one), and every flight ends within 0.3 m of the goal point (mean 0.297
   against 0.571 m). Successes are nine under both; one MPC flight is scored unsafe for contact.
2. **The same controller loses every projected flight.**
   - Under per-step projection, MuJoCo MPC succeeds on none of ten. Seven invert 15.7–20.7 s into the flight against the
     outer wall of the second straight (flown x 0.72–2.09 m, y ≈ 1.15). The other three reach the goal region with contact
     above the scene limit (contact on 16 % of steps).
   - The cascaded geometric controller flies five of ten of the same kind of plans.
   - The pilot's claim, "Changing the controller is not the whole answer once projection changes the plan", now stands at
     ten against ten on the selected configuration. The three-flight pilot numbers can go.
3. **No flight of the forty is violation-free, under either controller.** The reason is in the plans (my R44b note §3):
   every commanded path cuts the second inside corner by 8–19 cm. Concretely, the plans cross the gap about 0.2 m late:
   their commanded paths cross its middle (y = 0) at x = 0.21–0.27 m, where the demonstrations' Z-route crosses at x = 0,
   0.5 m from both corners (DA §8b). MuJoCo MPC follows the commanded path less closely
   (5–12 cm on average, against 1 cm), so it does not help here: its violating steps are 41.5 and 105.5, against 23.0
   and 71.3.
4. **Cost:** controller and simulator step 125.4 against 5.2 ms per control step, ten against ten (unprojected). On the
   projected plans the whole loop is 348.6 against 159.4 ms. The planner's share does not split cleanly there, because the
   projection time doubles under MuJoCo MPC, so quote the controller's cost from the unprojected pair.

A one-sentence conclusion the numbers support: *"On UAV-s-curve the tracking controller decides stability, not
constraint satisfaction: MuJoCo MPC keeps every unprojected flight upright through the second turn where the cascaded
geometric controller loses one, but it loses seven of ten projected flights where the cascaded controller loses three,
and no flight under either controller is free of violations, because the plans themselves cut the corner."* (Your call.)

## 3 · Figure 6.9 `fig:uav-scurve-paths`: rebuilt, ready to export

- **File:** `Data_Analysis/DA_in_Paper/figures/da/fig_uav_scurve_paths.{svg,png}` (PNG newer than SVG). Copy it with
  `python3 Data_Analysis/DA_in_Paper/plotting/export_to_draft.py v3`.
- **Layout:** FM, nfe 1, ten flights per panel, 2 × 2. Rows are unprojected and per-step projection (random, tightened);
  columns are cascaded geometric and MuJoCo MPC, as in Table 6.16. The subtitles give the success and violation-free
  counts.
- **Key:** green = success, red = no success, a cross = collided or lost control.
- **What it shows:** every path runs through the keep-out circle of the second inside corner. The one cascaded inversion
  sits just past it. The MPC paths are looser, and the MPC-projected flights end against the outer wall.
- **Caption facts:** "UAV-s-curve from above: FM at $\nfe=1$, ten flights per panel; top, the unprojected plans; bottom,
  the per-step projected plans (random rule, tightened set); left, the cascaded geometric controller; right, MuJoCo MPC.
  Green: success; red: no success; a cross marks a flight that collided or that the divergence guard ended."
- **Dataref:** `data/uav_paths.json` `figures['scurve']`, built by `plotting/extract/scurve_r44_paths.py`; runs
  `p23scgrid`, `p23scproj`, `p23scmjpc` (jobs 26171, 26195, 26196, 26204).
- **Retire:** the two `\outdated` notes, and the "three flights against ten" guard.

## 4 · Table 6.17 `tab:uav-controller-cost` (optional move to the selected configuration)

| ms per executed control step | cascaded geometric | MuJoCo MPC |
| :-- | --: | --: |
| the control loop | 14.0 | 135.8 |
| the planner | 8.9 | 10.4 |
| the controller and the simulator step | 5.2 | 125.4 |

Ten flights each. It is the same decomposition as the pilot's (elapsed time of a flight / executed steps; job records),
and the pilot's 6.5 against 125.8 ms is reproduced. If you keep the pilot table instead, one sentence can say the
selected configuration repeats it.

## 5 · Also unblocked

`sec:res:summary` can add UAV-s-curve (its `\hole` waits on R44b/c). The s-curve `\provisional` conclusion can be
rewritten with §2 and the corrected mechanism (R44b note §3).
