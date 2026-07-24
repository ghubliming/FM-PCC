# PLAN — `npz_traj_visualizer` · the full-trajectory MPC analyzer

**Status:** design **v3 — implementation-ready brief**. All open questions resolved by decision (§13).
A separate coding agent implements this; this doc is the single source of truth for that work.
**Date:** v1/v2 2026-07-11 · v3 (decisions locked + implementation brief) 2026-07-11.
**Tool home (per user):** `logs_in_develop/npz_analysis_tool/npz_traj_visualizer/` — code lives HERE,
next to this doc (not under `npz_analysis/`; that earlier idea is superseded).

---

## 0. What this is

One interactive HTML cockpit that **rebuilds the ENTIRE eval from the npz** — every executed step-dot
begin→end, every H-step receding-horizon MPC plan at every step, every candidate, every projection
variant — over the environment as 2D background, with DA_Code-style checkbox layers (default: nothing
shown; every tick adds a layer), and with the *quantitative* analysis built in, not bolted on.

Three pillars — it is a **viewer + inspector + analyzer** in one:

| pillar | question it answers | form |
|---|---|---|
| **SEE** | "what did every method plan & do, everywhere?" | layered canvas/SVG scene, giant-SVG export |
| **INSPECT** | "what exactly happened at step t?" | click a step-dot → per-step drill-down panel |
| **MEASURE** | "which method is quantitatively better, where, why?" | linked metric panels: violation timeline, divergence/drift curves, per-variant stat table |

Relationship to existing tools (this is the superset / endgame):
- `analyze_npz.py` — per-file aggregates → its metrics become the **stat table** panel.
- `compare_horizon_plans.py` — one snapshot, static → becomes **one frame** of this tool's scrubber;
  its violation math, div_ref, env geometry loaders are reused directly.
- ODE-benchmark insight (drift accumulation curves) → becomes the **per-step drift chart**.

## 1. The core scene (SEE)

- **Background = the environment, 2D only.**
  - avoiding-d3il: halfspace triangles + obstacle circles + goal line (from `projection_eval.yaml`,
    reusing `compare_horizon_plans.load_task_constraints` / `_halfspace_third_vertex`).
  - UAV s_curve/pillars: Phase-4 background = axis bounds + altitude reference line only (full pillar
    footprints deferred; see D5 in §13).
  - visual-aligning: its 2D scene (col_map + geometry TBD at Phase 5; see D2 in §13).
  - **3D→2D rule:** never a 3D canvas. 3D scenes get **linked 2D panels** (top-down x–y **+** side x–z,
    like the eval PNGs); hover/scrub highlights the same step t in every panel simultaneously.
- **Main bone:** the executed closed-loop path per variant — highlighted bold line + **a dot at every
  step** (~400–871 steps). This is the skeleton everything hangs off.
- **MPC fan:** at each step-dot, that step's H-step plan(s) as **thin lines** (H8 → 8 waypoints each).
  Candidate selection: ☐0 ☐1 ☐2 ☐3 · ○mean · (no "chosen" — not stored in npz; see D3).
- **Variant overlay:** every projection variant (diffuser, dpcc-c/r/t, gradient, model_free, geo_free,
  bounds_free, post_processing, *-tightened, …) is an independent colour-coded layer group.
- **Default EMPTY.** Ticking adds one layer at a time. Dense combinations are allowed by design
  ("crazy too much lines" is a feature) — usability under density is handled by §7, not by forbidding it.

## 2. Per-step drill-down (INSPECT)

Click (or scrub to) any step-dot:
- **Fan popup:** that step's B×H plan table (physical values) + mini-plot of just that fan, per ticked
  variant side by side.
- **Candidate compare at t:** per candidate — path_len, endpoint, max|coord|, violation flags.
  *Note baked into UI:* candidate k is the **same pre-projection sample across variants only at step 0**
  (shared `torch.manual_seed(i)`; verified empirically — dpcc/model_free/geo_free ≈0.01 apart, `gradient`
  desyncs by ~6.4 and must be excluded from candidate-level equivalence claims).
- **Action & dynamics check:** obs[t], act[t], and the dyn-gap `P[t+1]−P[t]−act[t]` where the schema
  supports it (avoiding executed path: yes; plans: obs-only → N/A, displayed honestly as such).
- **Plan-vs-reality:** overlay of plan(t) against the executed path over [t, t+H] with the per-waypoint
  tracking error.

## 3. Quantitative panels (MEASURE) — what makes it an analyzer

All linked to the same scrubber/selection state as the scene:

1. **Violation timeline strip** (per ticked variant): one row per variant, x = step, colour = plan
   violation fraction at that step (reuses `halfspace_violates`/`obstacle_violates`). A constraint
   breach shows as a coloured band; clicking the band jumps the scrubber there.
   *(avoiding only until UAV/VA constraint geometry lands — emit `null` per step for other envs.)*
2. **Drift / divergence curves** (the ODE-benchmark idea transplanted): x = step, y = per-variant
   `div_ref` (plan divergence vs a reference variant, default diffuser) and/or `max|coord|`
   (explosion detector), per dimension toggleable (x, y, **z** — this is what catches the −587 m
   altitude runaway invisible in top-down).
3. **Tracking-error curve:** ‖plan(t)[k] − executed[t+k]‖ aggregated per step — plan-vs-reality
   divergence over the rollout (the `plan_exec_div` idea, resolved per step instead of one scalar).
4. **Per-variant stat table:** the npz scalar metrics (success_*, phys_*, goal_*, constraint_*,
   n_steps…) + traj-quality aggregates (path_len, straightness, roughness, max_jerk — same definitions
   as `analyze_npz.py`) for the current trial; sortable; p50/p95/max not just mean (benchmark
   `compute_stats` convention).
5. **Candidate-spread panel:** per-step candidate diversity (endpoint spread) per variant — shows where
   the sampler collapses vs stays diverse.

## 4. Architecture — exporter + viewer (decouple heavy npz from the browser)

```
[ Python exporter ]                       [ HTML analyzer ]
npz_traj_export.py      →  scene.json  →  npz_traj_visualizer.html (template)
(venv / cluster; reads      + viewer_<scene>.html (SELF-CONTAINED: template
 every <variant>.npz          with scene.json inlined — just open in browser)
 of one scene; also
 PRECOMPUTES panel data)
```

**Exporter precomputes what the browser shouldn't:** violation masks per step, div_ref/drift curves,
tracking errors, stat table, candidate spread. The HTML stays a renderer + selector — fast and dumb.
- Input handling reuses `find_variant_npz` (flat `<v>.npz` AND nested `<v>/<v>.npz`) and
  `plan_snapshots` (ragged object arrays AND homogeneous 5-D stacks).
- `--env` column conventions shared with the sibling tools (avoiding x,y=2,3 · uav x,y,z=3,4,5 ·
  visual-aligning TBD).
- Per-scene output goes to `<scene_dir>/_traj_viz/`: `scene.json` + `viewer_<scene>.html`.

## 5. `scene.json` data model (draft — implementer may extend, not shrink)

```jsonc
{
  "scene": "s_curve_bounds+dynamics+geo_bounds+halfspace+obstacles",
  "env": "uav",
  "panels": [ {"id":"xy","dims":[0,1]}, {"id":"xz","dims":[0,2]} ],   // indices into pos triple
  "col_map": {"x":3,"y":4,"z":5},
  "plan_every": 3,                    // decimation factor actually used (record it!)
  "background": { "halfspaces":[...], "obstacles":[...], "goal_line":[...], "bounds":[...] },
  "reference": "diffuser",
  "variants": {
    "dpcc-c": {
      "color": "#4363d8",
      "metrics": { "success_strict": 1.0, "goal_dist": 0.03, ... },     // stat table row
      "trials": [ {
        "trial": 0,
        "executed": [[x,y,(z)], ...T],       // FULL resolution, never decimated
        "plan_steps": [0,3,6,...],           // which exec steps have an exported fan
        "plans": [ [[ [x,y,(z)]*H ]*B ], ...],  // per plan_step: B candidates × H waypoints
        "viol_frac_per_step": [...|null],    // precomputed timeline strip (null when no geometry)
        "div_ref_per_step":   [...],         // precomputed drift curve (null for reference itself)
        "track_err_per_step": [...],
        "cand_spread_per_step": [...],
        "explosion_max_per_step": [...]
      } ] } } }
```
Physical units throughout (npz already stores them). Only position dims exported, not all obs cols.
Floats rounded to 4 decimals (JSON size). Executed path always full-res; only plans decimate.

## 6. Rendering engine

- **Canvas 2D** for dense geometry (fans: 10⁵–10⁶ segments) — batched paths, rAF, level-of-detail.
- **SVG overlay** for sparse/interactive: env background, main bones, step marker, hover targets, legend.
- **Panels** (timeline/curves/table) as lightweight inline SVG/canvas charts — same selection state.
- **DA_Code shell reused, its render path NOT:** keep the sidebar checkbox lists, `[ALL]/[NONE]`
  bulk-select, `trigger_plot()`-on-change wiring from `Data_Analysis/Visualizer/index.html`; drop
  PyScript/matplotlib→PNG (0.5–2 s per toggle, flat raster — wrong for layer toggling/zoom/scrub).
- **Playback mode:** ▶ animates the receding horizon — the fan of step t glides along the main bone
  (speed slider, loop range = scrubber window). This is the "receding horizon movie" that no static
  figure can show.
- **Zoom/pan:** wheel-zoom around cursor + drag-pan per panel; double-click = reset to fit mode.
- **Export:** current view → **giant standalone SVG** (vector, publication) or high-DPI PNG (canvas
  `toDataURL` at 2–4× scale); current panel data → CSV; current UI state → shareable URL hash.

## 7. Density & performance (core risk, designed-in)

Worst case: 17 variants × 10 trials × 871 steps × 4 cand × 8 wp ≈ **4.7 M points**.
- Default empty; **one trial at a time** (viewer trial dropdown; exporter `--trials`).
- **Step window / every-Nth scrubber**; "all steps" opt-in with a warning.
- Default **one candidate** (mean); the 4-fan is opt-in.
- Exporter-side decimation with recorded factor (`plan_every`); default auto: keep ≤ ~300 fans per
  variant per trial (avoiding 25 snaps → 1; UAV 871 → 3). Override with `--plan-every 1` for full res.
- Canvas LOD (drop sub-pixel segments when zoomed out); per-variant point budget with auto-decimate
  notice; alpha auto-scaling (more layers ⇒ thinner/fainter fans so density stays readable).

## 8. Scale & explosion handling (learned from the z-blowup)

Executed z explodes (s_curve diffuser: −587 m in npz; −12 000 m in the raw eval PNG) while plans stay
bounded (±3.2). One runaway curve must not flatten the whole scene:
- Per-panel autoscale modes: **fit-executed** (default) · fit-visible-layers · fit-all · manual ·
  **robust** (1–99 percentile clip, clipped points drawn as edge arrows, never silently dropped).
- Optional **symlog** on a panel axis (altitude) so z≈1 and z≈−600 are both legible.
- The explosion curve panel (§3-2) is the honest companion: even when clipped out of the scene, the
  runaway is visible as a curve.

## 9. Sidebar / layer spec (DA_Code pattern)

- **Scene** header (from scene.json) · **Trial** dropdown · **Panels** ☐x–y ☐x–z.
- **Layer types** (global): ☐env bg · ☐main bone · ☐step dots · ☐MPC fan · ☐mean candidate ·
  ☐violation marks · ☐horizon-end markers.
- **Variants:** dynamic checkbox list + [ALL]/[NONE], colour swatch per row.
- **Candidates:** ☐0 ☐1 ☐2 ☐3 · ○mean (labelled "mean — selected idx not stored in npz").
- **Steps:** scrubber, window [a,b], every-N, ▶ playback + speed.
- **Panels toggle:** ☐timeline ☐drift curves ☐tracking error ☐stat table ☐spread.
- **Scale mode** select (§8) · **Export** buttons (SVG / PNG / CSV / copy-state-URL).
- Everything defaults **OFF**; every control re-renders via one `render()`.

## 10. Build phases (ship incrementally; each phase is usable & testable)

1. **MVP pipeline (avoiding):** exporter (flat npz, x–y, trial 0) → `viewer_<scene>.html`: env bg +
   main bone + step dots + one variant + fan + scrubber + zoom/pan. Prove npz→json→canvas end to end.
2. **Layer machinery:** multi-variant overlay, all layer-type toggles, candidates+mean, [ALL]/[NONE],
   giant-SVG export, state-hash.
3. **Analyzer panels:** violation timeline, drift/div + explosion curves, tracking error, stat table,
   spread (exporter precompute lands here). Inspector drill-down v1 (click dot → info panel).
4. **UAV:** nested npz, dual linked panels (x–y, x–z), robust/symlog scale handling, playback mode.
5. **Scale-out & polish:** LOD/budgets, full inspector (fan popup + candidate table), visual-aligning
   env, per-layer opacity, PNG export, multi-trial "trials as layers" mode (D7).

## 11. File layout (FINAL — REVISED per user: code ≠ logs; two parallel trees split by type)

**Rule (user, this session):** `logs_in_develop/` holds ONLY MD/logs — **never code**. `npz_analysis/`
holds ONLY code. The two `npz_*` folders are "the same thing" → keep them as a **parallel pair**:

```
npz_analysis/                              # CODE ONLY
  analyze_npz.py
  compare_horizon_plans.py
  npz_traj_visualizer/
    npz_traj_export.py                 # npz → scene.json + embedded viewer_<scene>.html
    npz_traj_visualizer.html           # viewer TEMPLATE (has /*__SCENE_DATA__*/null marker)

logs_in_develop/npz_analysis_tool/         # DOCS/LOGS ONLY (mirrors the code tree)
  README.md · USAGE_compare_horizon_plans.md · CHANGELOG*.md · CAPABILITY_GAP*.md · MPC_*.md
  npz_traj_visualizer/
    PLAN_npz_traj_visualizer.md        # this doc
    CHANGELOG_npz_traj_visualizer.md   # per-phase changelog

<scene_dir>/_traj_viz/                      # per-scene OUTPUTS (data; not committed)
  scene.json
  viewer_<scene>.html                  # self-contained; open directly in any browser
```
(Supersedes v3's D8, which had put code under logs — corrected this session.)

## 12. Reuse checklist (don't reinvent)

- `npz_analysis/compare_horizon_plans.py`: `find_variant_npz`, `plan_snapshots`,
  `load_task_constraints`, `_halfspace_third_vertex`, `halfspace_violates`, `obstacle_violates`,
  div_ref definition. **Import mechanics:** `npz_analysis/` has no `__init__.py` — add that dir to
  `sys.path` and `import compare_horizon_plans` (its argparse is guarded under `__main__`; import is
  safe and side-effect-free).
- `analyze_npz.py`: traj-quality metric definitions (straightness/roughness/jerk/max_abs), schema-generic
  scalar-metric harvesting (any 1-D numeric array = per-trial metric) for the stat table.
- ODE benchmark (`FM_v3_ode_selectable_test/Benchmark_ode_solver_Tests` v3/v4): drift-accumulation-curve
  concept, `compute_stats` percentile convention (p50/p95/max).
- DA_Code `Data_Analysis/Visualizer/index.html`: sidebar checkbox/bulk-select/trigger wiring (HTML/CSS
  pattern only — NOT its PyScript render path).
- Colour palette: reuse `_PALETTE` from `compare_horizon_plans.py`; assign colours by sorted variant
  name (stable across sessions); extend with more hues past 10 variants (17 exist in UAV scenes).

## 13. DECISIONS — all former open questions resolved (v3, user delegated all choices)

| # | question | **decision** | rationale |
|---|---|---|---|
| D1 | engine | **Vanilla JS + Canvas 2D + SVG overlay. No frameworks, no CDN, fully offline.** | responsive at 10⁵+ segments; PyScript/matplotlib re-render is 0.5–2 s/toggle; offline = works with cluster-downloaded folders |
| D2 | visual-aligning layout | **defer to Phase 5**; implementer inspects a `va_*` npz then (schema keys listed in `analyze_npz.py` header) | don't block Phases 1–4 on an uninspected env |
| D3 | "chosen candidate" | **use candidate-mean, labelled "mean — selected idx not stored"**. Patching evals to persist `selected_idx_all` is OUT OF SCOPE for this tool (separate eval-side task with cross-generation sibling-sync) | honest MVP; eval change is a different blast radius |
| D4 | data transport | **single `scene.json`, floats rounded 4 dp; default deliverable = self-contained `viewer_<scene>.html` with data inlined** (open directly — no http server, no CORS). Template also accepts drag-drop / file-picker of a scene.json. No binary sidecar in v1 | simplest thing that works everywhere incl. `file://`; ~5–20 MB with decimation |
| D5 | UAV background | **Phase 4 = bounds + altitude reference line only; pillar footprints deferred** until the UAV eval's geometry source is confirmed (start at `config/uav_projection.yaml` + the UAV eval's own plotting code) | never draw guessed geometry (hard rule inherited from the `--draw-constraints` work) |
| D6 | hosting | **no server** — embedded single-file viewer is the default artifact (D4) | user just opens the HTML |
| D7 | multi-trial | **one trial at a time (dropdown) through Phase 4**; "trials as layers" overlay = Phase-5 optional mode | density control first |
| D8 | tool location | **`logs_in_develop/npz_analysis_tool/npz_traj_visualizer/`** (user instruction; supersedes v2's `npz_analysis/` placement) | user decides repo layout |
| D9 | exporter deps | **numpy + pyyaml only** (no matplotlib — the exporter never draws) | runs in any venv and on the cluster |
| D10 | trials default | exporter `--trials 0` (one trial); `--trials all` or comma-list to export more | JSON size; rerunning is cheap |

## 14. IMPLEMENTATION BRIEF (for the coding agent — verified facts, do NOT re-derive)

### 14a. Environment & testing constraints
- **This container has NO Python packages installed.** For exporter testing create an isolated venv:
  `python3 -m venv <scratchpad>/npzenv && .../pip install numpy pyyaml` (matplotlib NOT needed).
  Never install into the system env. Real runs also work on the cluster (FMPCC conda env).
- **Node v24 is available** (`node --check`) — syntax-check the viewer's JS by extracting the
  `<script>` body to a temp file. There is no browser in this container; visual QA happens on the
  user's machine — make the viewer self-diagnosing (on-page error banner when data is missing/corrupt,
  console logs for layer counts).
- **Never `git commit`** — the user commits manually. Maintain `CHANGELOG_npz_traj_visualizer.md`
  (this folder) every phase. Never touch `MASTER_TEST_HISTORY.md`. Keep commenting/updating THIS plan
  MD as decisions evolve (append, don't erase history).

### 14b. Real test datasets (already downloaded, verified readable this session)
1. **avoiding (flat layout):** `/workspaces/FM-PCC/temp/npz_imf_debug/halfspace_both-hard-imf-debug/`
   — 13 variants as flat `<v>.npz`; 2 trials; `obs_all[i]=(T≈98–149, 4)`; plans: `diffuser` is a
   homogeneous 5-D object array `(2,25,4,8,4)`, `dpcc-*` are ragged `(2,)`-object of lists of `(B,8,4)`
   (both normalised by `plan_snapshots`); snapshots every `H//2=4` steps; plan dim 4 =
   `[x_des,y_des,x,y]` → **actual pos = cols 2,3**; plans store obs ONLY (no action columns).
   Geometry: halfspace variant `both-hard` = `halfspace_constraints[2],[3]` +
   `obstacle_constraints[5]` from `config/projection_eval.yaml`; `ax_limits=[[0.2,0.8],[-0.3,0.4]]`.
2. **UAV (nested layout):** `/workspaces/FM-PCC/temp/s_curve_bounds+dynamics+geo_bounds+halfspace+obstacles/`
   — 17 variants as `<v>/<v>.npz`; 10 trials × 871 steps; `obs_all=(10,871,6)` `[p_des(0:3)|p(3:6)]`
   → **actual pos = cols 3,4,5**; `act_all=(10,871,3)`; plans `(10,871,4,8,6)` — a fan EVERY step
   (save_every=1). **Known data truth:** executed `p_des_z` (obs col 2) explodes to −587 for
   diffuser/model_free while ALL plan values stay within ±3.2 — this is the acceptance case for §8
   robust scaling, not a bug to "fix".
   ⚠ An earlier copy at `temp/npz_imf_debug/s_curve.../` has ALL npz truncated to exactly 512 KiB
   (BadZipFile) — do not use; the exporter must warn+skip per file on such corruption, never crash.
3. Candidate-comparability facts (surface in INSPECT UI): at snapshot 0, candidates are RNG-shared
   across variants (max-diff ≈0.01) EXCEPT `gradient` (≈6.4 desync). Verified on dataset 2.

### 14c. Exporter CLI (contract)
```
python npz_traj_export.py <scene_dir>
    [--env avoiding|uav|unknown]      # default: infer (plans last-dim 4 → avoiding; 6 → uav)
    [--trials 0|all|0,3,7]            # default 0
    [--plan-every N]                  # default auto (≤~300 fans/variant/trial); recorded in scene.json
    [--variants a,b,c]                # default: all found
    [--reference diffuser]            # div_ref baseline
    [--halfspace-variant both-hard|top-left-hard|top-right-hard]   # avoiding geometry; infer from dir name
    [--config config/projection_eval.yaml]
    [--out DIR]                       # default <scene_dir>/_traj_viz
    [--no-embed]                      # skip viewer_<scene>.html generation
```
Behaviours: per-file load failures warn+skip (never crash the scene); unknown env → export
trajectories with `background:{bounds only}` and `viol_frac_per_step:null`; finish with a summary
table (variants, trials, steps, fans exported, JSON size) **including 3 spot-check analytic values**
(used by the P3 acceptance test).

### 14d. Per-step analytics definitions (exporter computes; alignment = step index)
- `div_ref_per_step[s]` = mean over waypoints of ‖meanplan_variant(s) − meanplan_reference(s)‖ on the
  exported pos dims; only for steps present in BOTH (min length — snapshot counts legitimately differ
  per variant in avoiding; the curve simply ends earlier).
- `viol_frac_per_step[s]` = fraction of ALL candidate waypoints of fan(s) violating any
  halfspace/obstacle (exact inequalities from `compare_horizon_plans`); `null` when no geometry.
- `track_err_per_step[s]` = mean over k∈[0,H) of ‖meanplan(s)[k] − executed[s+k]‖ (clip index at T−1).
- `explosion_max_per_step[s]` = max|coord| over the whole fan(s), all exported dims.
- `cand_spread_per_step[s]` = mean pairwise candidate-endpoint distance (null when B=1).
- Stat table: harvest every 1-D numeric npz key (analyze_npz convention) + computed traj-quality
  numbers (path_len/straightness/roughness/max_jerk/max_abs) for each exported trial.

### 14e. Acceptance criteria (per phase; the user QAs visuals in their own browser)
- **P1:** open `viewer_<scene>.html` (avoiding) directly from disk → empty scene; tick env bg +
  diffuser bone + fan → topology matches the eval PNG; scrubber changes visible fans; zoom/pan works;
  zero console errors; `node --check` passes on extracted JS.
- **P2:** toggling any combination of 13 variants × layer types re-renders <100 ms (avoiding dataset);
  exported SVG opens standalone and contains only the ticked layers.
- **P3:** timeline/curve values match the exporter's 3 printed spot-checks; clicking a curve moves the
  scrubber; stat table sorts.
- **P4:** UAV viewer shows linked x–y + x–z panels; fit-executed on x–z is unusable because of −587 →
  robust/symlog modes make z≈1 and z≈−600 simultaneously legible; playback is smooth at plan_every=3.
- **P5:** visual-aligning scene renders; inspector popup shows the clicked step's B×H table.

### 14f. Explicit non-goals (v1)
- No 3D WebGL. No eval-side changes (D3). No pandas/plotly/d3/other JS deps. No server backend.
- No reconstruction of the *selected* candidate (not stored — labelled mean instead).
- No cross-scene aggregation (that's `Data_Analysis/` territory).

---

## Appendix — version log (audit trail; keep appending, don't erase)
- **v1:** viewer-only concept, tool under `npz_analysis/`. Superseded.
- **v2:** rebuilt as SEE/INSPECT/MEASURE full analyzer: playback, per-step panels, exporter precompute,
  explosion handling, density design; 7 open questions left open.
- **v3:** user delegated all decisions → §13 D1–D10 locked; §14 implementation brief added
  (dataset facts, CLI contract, analytics defs, acceptance criteria).
- **v4 (BUILT this session):** implemented Phases 1–4 core. New decisions logged:
  - **D8 REVISED:** code lives in `npz_analysis/npz_traj_visualizer/`, NOT under logs_in_develop
    (user: logs = MD only; code never in logs). Docs stay in `logs_in_develop/npz_analysis_tool/`.
    The two `npz_*` trees are a parallel pair split by file type (§11).
  - **D11 (UAV z-explosion mapping):** scene panels plot the ACTUAL drone position `p` (cols 3,4,5) —
    physically coherent. The `p_des_z` (col 2) runaway to −587 is NOT put on a panel axis; it is
    surfaced by the exporter's `exec_maxabs_per_step` curve (max|coord| over ALL obs dims) so the
    explosion stays visible (verified: diffuser exec_maxabs peaks 586.95) without wrecking the scene
    frame. Acceptance test P4 wording updated accordingly: robust/fit modes handle the *actual* z
    (bounds_free climbs to ~32), and the explosion CURVE carries the −587 story.
  - **Build status:** exporter `npz_traj_export.py` + viewer template `npz_traj_visualizer.html`
    written & tested — avoiding (6 variants, 0.13 MB) and UAV (4 variants, dual xy/xz panels, 1.11 MB,
    plan_every=3) export cleanly; div_ref matches `compare_horizon_plans` (dpcc-c 0.00915);
    corrupt-npz warn+skip verified; viewer JS passes `node --check`. Browser visual QA = user's side.
  - Deferred to a later pass (documented, not done): symlog axis toggle, edge-arrow clipping markers,
    candidate-mean INSPECT popup table, view-state URL hash, per-layer opacity, UAV pillar geometry
    (D5), visual-aligning env (D2), multi-trial layers (D7). Core scene + all layer toggles + scrubber
    + playback + pan/zoom + 5 analytic charts + stat table + PNG/SVG/CSV export are IN.
