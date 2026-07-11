# fix_1 — MPC fan bug, path-loader 404, Redraw/Recenter controls

**Date:** 2026-07-11 · **File:** `npz_analysis/npz_traj_visualizer/npz_traj_visualizer.html`
(exporter output re-generated for both existing scenes; no exporter code changes this round).

User reported three issues after using the viewer live. All three fixed; both viewers re-exported.

## 1. "MPC fan seems not working" — real bug, now fixed

**Root cause:** the raw-fan draw path was gated on `!S.cands.mean` (`if(!wantC.length && !S.cands.mean)
fan.forEach(...)`), and `S.cands.mean` **defaulted to checked**. So `!S.cands.mean` was always `false`
by default → ticking the "MPC fan" layer checkbox never drew the thin per-candidate lines; only the
bold *mean* line drew (which is easy to miss, often overlapping the main bone).

**Fix — decoupled candidates from layers, no more interaction:**
- **Candidates (0/1/2/3)** now purely *select which candidates* the fan/mean layers use. None ticked
  = use **all** candidates (was previously coupled to a "mean" toggle that lived in the same group).
- **"MPC fan"** and **"mean candidate"** are independent **layer** checkboxes: fan = thin lines for
  the selected candidates; mean = one bold line, the mean of the same selection. Ticking either now
  reliably draws something, with no cross-gating.
- Removed the now-redundant "mean" checkbox from the Candidates group (it lived there confusingly
  since "mean" was never really a *candidate*). Horizon-end markers (`layers.hend`) also updated to
  respect the candidate selection, for consistency.

## 2. Path-loader 404 on an absolute filesystem path

User pasted an **absolute OS path**
(`/workspaces/FM-PCC/temp/npz_imf_debug/halfspace_both-hard-imf-debug/_traj_viz/scene.json`) — this
previously only got ONE retry (bare `/`-prepend, already had a leading `/`, so no retry even ran) and
404'd, because the http server's root (wherever it was started, e.g. `/workspaces/FM-PCC`) is an
*ancestor* of the OS path, not the OS root itself. Python's `http.server` maps its CWD to URL `/`, not
the filesystem `/`.

**Fix — try successively shorter suffixes of the path, longest first, until one resolves:**
`buildPathCandidates()` splits the path into segments and tries `/<all segments>`,
`/<all but first>`, `/<all but first two>`, … down to `/<filename>`, stopping at the first HTTP 200.
Verified with a standalone reproduction of the exact reported path: the 3rd candidate
(`/temp/npz_imf_debug/halfspace_both-hard-imf-debug/_traj_viz/scene.json`) is the one that succeeds
when the server is started at the repo root — the ordinary case. Full URLs (`http://...`) are left
untouched (no mangling). On total failure, the banner now lists every URL actually tried, so a genuine
miss (no server, wrong scene) is easy to diagnose instead of guessing.
- Banner CSS updated (`white-space:pre-wrap`, monospace, scrollable) so the multi-line tried-list
  is actually readable instead of being collapsed onto one line.

## 3. Added Redraw / Recenter controls

New **View** group in the sidebar:
- **Redraw** — forces a re-render (safety net if a view looks stale after some external change).
- **Recenter** — clears ALL panels' pan/zoom offsets and refits to the current scale mode (the same
  thing double-click already did per-panel, now available as one explicit button + documented).

## Verification
- `node --check` on the extracted `<script>` body: **pass**.
- Both scenes re-exported with the updated template (avoiding: 6 variants, 0.26 MB; UAV: all 17
  variants, 4.75 MB) — `div_ref` spot-checks unchanged/consistent with prior runs (exporter untouched).
- `buildPathCandidates()` logic verified standalone against the exact reported failing path (see above).
- Browser interaction (does the fan visibly render, do buttons work) is user-side QA — no browser here.

## Files touched
- `npz_analysis/npz_traj_visualizer/npz_traj_visualizer.html` (viewer template — all 3 fixes)
- Regenerated: `temp/npz_imf_debug/halfspace_both-hard-imf-debug/_traj_viz/*`
- Regenerated: `temp/s_curve_bounds+dynamics+geo_bounds+halfspace+obstacles/_traj_viz/*`
- `USAGE.md` (parent folder) — updated: path-loader section rewritten for the suffix-search behavior
  (any absolute/relative path or URL now works, not just server-root-relative), added a candidates-vs-
  layers clarification and a View/Recenter/Redraw note.

## Not done this round
- No further doc debt from this fix. Still open from earlier (PLAN v4 deferred list): symlog axis,
  edge-arrow clip markers, INSPECT fan popup table, view-state URL hash, per-layer opacity, UAV pillar
  geometry, visual-aligning env, multi-trial layers.
