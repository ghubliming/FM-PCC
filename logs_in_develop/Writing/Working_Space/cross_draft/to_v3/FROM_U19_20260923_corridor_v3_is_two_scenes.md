# FROM U19 (run side) → v3 · 2026-09-23 · corridor v3 is TWO scenes, not one; pilots done; naming for the DA

**For:** `PENDING_20260923_uav_corridor_v3_pillars_v2_paper_runs.md` §1 and `SLURM_RUNBOOK_20260923_…` §0.2 / §1–§2.
**Source:** `logs_in_develop/Gen15/U19/` (changelog, `PILOT_20260922_U19_gates_G1-G3.md`, `figs/`).

## 1 · The geometry (author, 2026-09-22/23)

The pending file's §1.1 (`corridor_v3` = the v2 slide **plus** the hump in one scene, `hs=5`, suffix `_cv3`, tag `p23cv3`)
is **not** what was built or run. The author redefined v3 on 22-09 and confirmed on 23-09 ("two scenes as piloted"):

| entry (`config/uav_projection.yaml`) | what | folder suffix | paper tag |
| :-- | :-- | :-- | :-- |
| **`corridor_v3_tilt`** — *the* corridor v3 | the v2 slide **leaned over** by −60° about the launch altitude (`z_lean` key): one x–y–z plane that pushes the drone sideways **and down**. Workspace box, walls, caps, scene XML, model, routes, goals byte-identical to v2 (ceiling stays 1.80; the room is below the launch, 0.50 m). Sideways-only is infeasible at launch altitude on every route; the min-norm correction is 87 % z. | `_cv3t` | **`p23cv3t`** |
| `corridor_v3_ablation_hump` — the ablation | no slide; an x–z roof (0 → 1.10 m at x = 0 → 0 over x ∈ [−1.5, 1.5], two `plane: xz` halfspaces), ceiling 1.80 → 2.80 | `_cv3ah` | **`p23cv3ah`** |

There is **no combined slide + hump scene**. Chapter text that says "corridor v3 = slide + roof, a 3-D detour" (ledger §26,
the → v4 row of v3.68) should read: *v3 = the slide leaned into the vertical, so one constraint asks for a lateral and a
vertical move at once; the roof-only scene is an ablation.* The pending file's §5 item 4 (joint G0 for slide + roof) is
moot.

## 2 · Pilots (C0) are done — 22-09, jobs 26071/26072

G0 pass (offline) for both. G1 pass (unprojected violates 3/3), **G3 pass** (projected flights descend 0.15–0.29 m under the
tilt, climb 0.30–0.40 m over the hump — the projector moves z on demand, which U11–U13 never got). **G2 fails** on every
projected arm (collision-free 0/3) by one thing only: 6–11 shallow steps (≤ 2–6 cm) at the geometric switch point
(window end / apex) with the setpoint clean at every one — the plant trailing its setpoint by ~0.5 m. Author's decision:
**run the waves as is; no scene change** ("if it is the model / projector, no need to fix the env"). The runbook's
"stop if G2 fails" is therefore overridden by the author. Full numbers: `Gen15/U19/PILOT_20260922_U19_gates_G1-G3.md`.

## 3 · Driver and two naming facts the DA needs

- Driver: `Slurm_Codes/temp_bash/eval_20260923_p23_corridor_v3.sh` (`plan` / `smoke` / `submit C1..C5|all`), both scenes,
  68 cells each, runbook §1 variant lists, waves C1–C5 in the runbook's order.
- **HardFlow spelling:** the eval accepts only the registered input names `hardflow_new{,-r,-c,-t}-…` and **writes the
  result folders as `hardflow_sls-…`**. The DA reads folders, so `hardflow_sls-…` is right there; a submitter must say
  `hardflow_new`.
- **C3 composition:** the eval refuses a job that keeps HardFlow variants without a `dpcc-*` row (matched-budget guard),
  so the HF jobs carry `dpcc-t` and C2 runs `dpcc-t` only at K = 1, 2. Same 68 cells, each run once.
- Result path: `…/E<engine>_K<k>_mpc4_pid_stopgo_T0.5_p23cv3t/6/corridor_cv3t_bounds+dynamics+geo_bounds+halfspace+obstacles/<variant>/`
  (and `p23cv3ah` / `corridor_cv3ah_…`). `DA_UAV_v1/discovery.py` parses the leading `corridor` token; checked.
- Figure store: the v3.65 altitude builder (`fig_uav_corridor_altitude`) is the right picture for the tilt (z over x); the
  per-variant overview's side panel now also draws the leaned plane's cut along route C and the drone-centre limit.

Claude (Fable 5.1, Claude Code, run side) · 2026-09-23
