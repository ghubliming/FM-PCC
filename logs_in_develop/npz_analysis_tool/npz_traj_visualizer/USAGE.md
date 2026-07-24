# USAGE — `npz_traj_visualizer`

Two separate steps, always in this order. Skipping step 1 (or opening the wrong file) is why it
"feels hard to use" — you get a blank template with nothing loaded.

```
STEP 1 (Python, prepares the data)  →  STEP 2 (browser, just look at it)
npz_traj_export.py  reads the .npz     open the GENERATED viewer_<scene>.html
and writes a ready-to-view HTML        (NOT the plain npz_traj_visualizer.html template)
```

---

## Step 1 — prepare the data (run the exporter)

The exporter only needs `numpy` (+ `pyyaml` for avoiding geometry). The on-PATH `python3` has
neither, **but `/usr/local/bin/python3` already has numpy + pyyaml**, so it runs in-container with
no setup:

```bash
/usr/local/bin/python3 npz_analysis/npz_traj_visualizer/npz_traj_export.py <scene_dir>
```

(Heavy torch/MuJoCo/GPU/eval work still belongs on the cluster — this exporter does not.)
If `/usr/local/bin/python3` ever lacks numpy, fall back to a one-time venv:
`python3 -m venv ~/npzenv && ~/npzenv/bin/pip install numpy pyyaml`, then use `~/npzenv/bin/python`.

Examples (note `--env` and `--trials`, the two flags people most often forget):

```bash
# avoiding, ALL trials (default is trial 0 only!) — exact command from the imf halfspace session:
/usr/local/bin/python3 npz_analysis/npz_traj_visualizer/npz_traj_export.py \
  "temp/flow_matching_v3_imeanflow/H8_Dflow_matcher_v3_imeanflow.models.iMeanFlowODE_a1.0_b1.0_aw1_objmeanflow_jvp_bbdit_tslogit_normal/H8_K10_Meuler_T0.5_Dflow_matcher_v3_imeanflow.models.iMeanFlowODE/6/results/halfspace_both-hard" \
  --env avoiding --trials all

# avoiding (flat <variant>.npz layout)
/usr/local/bin/python3 npz_analysis/npz_traj_visualizer/npz_traj_export.py \
  "temp/npz_imf_debug/halfspace_both-hard-imf-debug" --env avoiding

# UAV (nested <variant>/<variant>.npz layout)
/usr/local/bin/python3 npz_analysis/npz_traj_visualizer/npz_traj_export.py \
  "temp/s_curve_bounds+dynamics+geo_bounds+halfspace+obstacles" --env uav
```

This writes into `<scene_dir>/_traj_viz/`:
- `scene.json` — the extracted + precomputed data
- **`viewer_<scene>.html`** ← **this is the file you open**, not the template in
  `npz_analysis/npz_traj_visualizer/npz_traj_visualizer.html`. The template has no data baked in
  (open it and you'll just see an empty sidebar + a file picker — that's expected, see below).

Useful flags:
- `--env avoiding|uav|unknown` — picks the constraint-geometry background (default `unknown` = none).
- `--trials 0` (default, trial 0 only) · `--trials all` · `--trials 0,1,3` — **use `all` if the npz has
  more than one trial**; otherwise the Trial dropdown only offers trial 0 because that's all that got
  exported (not a viewer bug — the other trials simply aren't in `scene.json`). Quick check of how many
  trials an npz holds: the leading dim of `obs_all` / `sampled_trajectories_all`.
- `--variants a,b,c` — subset instead of all.
- `--plan-every N` — thin the MPC fans (auto by default). Cuts file size / load time on big scenes.

Full CLI in `PLAN_npz_traj_visualizer.md` §14c.

Re-run the exporter any time you want a different trial/variant subset — it's cheap (seconds).

## Step 2 — open it in a browser

**No server needed.** The viewer has the data baked directly into the HTML — just open the file:

```bash
xdg-open "temp/npz_imf_debug/halfspace_both-hard-imf-debug/_traj_viz/viewer_halfspace_both-hard-imf-debug.html"
```

or double-click it in a file browser, or drag it into an open browser tab.

**If you're already running `python3 -m http.server 8000` and prefer that** — it also works fine
(just navigate to the file's path under that server).

## Loading data via the port (path/URL box) — the recommended flow if files live in the container

If the browser is on your host machine (e.g. Windows) and the container/WSL filesystem isn't easy to
click through in the native file-picker (it opens Windows Explorer, which doesn't know your container
paths), use the sidebar's **path/URL box** instead of the file picker:

1. In the container, from wherever you want the server ROOT (e.g. repo root): `python3 -m http.server 8000`
2. Open the viewer through that port, e.g. `http://localhost:8000/temp/<scene>/_traj_viz/viewer_<scene>.html`
   (already loads with data inlined — no extra step needed for the SAME scene).
3. To load a **different** scene without re-opening a new tab, paste its `scene.json` path into the
   sidebar's **Load → path/URL box** and click **Load**. **You can paste ANY of these forms** and it
   will find it: a plain absolute filesystem path (e.g.
   `/workspaces/FM-PCC/temp/scene/_traj_viz/scene.json`), a server-root-relative path (e.g.
   `/temp/scene/_traj_viz/scene.json`), or a full `http://host:port/...` URL.

**How it finds the right one automatically:** a browser resolves a plain relative path (no leading
`/`) against the *current page's folder* (`.../_traj_viz/`), not the http-server root — and even with
a leading `/`, the server root might not be the filesystem root you pasted from. So the box doesn't
just try your input once: it **splits the path into segments and tries every suffix**, longest first
(`/workspaces/FM-PCC/temp/.../scene.json`, then `/FM-PCC/temp/.../scene.json`, then
`/temp/.../scene.json`, …, down to `/scene.json`), stopping at the first one that resolves. In the
ordinary case (server started at the repo root) this means **you can just paste the full path shown
anywhere in this chat/terminal** and it works without you doing any translation. If every suffix
fails, the red banner lists every URL it actually tried, so a genuine miss (no server running, or the
file really isn't there) is easy to tell apart from a path-mismatch.

**This box requires the http server** — it uses `fetch()`, which browsers block over `file://`. If
you're not running a server, use the file picker above instead — that one reads local files directly
and always works, no server needed.

## Why it can look empty at first (this is by design, not a bug)

Every layer defaults **OFF** so you build up only what you want to see instead of getting a wall of
lines immediately. On first load, tick boxes in roughly this order:

1. **Variants** (left sidebar, bottom list) — tick at least one, e.g. `diffuser`.
2. **Layers** — tick `main bone` to see the executed path, then `MPC fan` for the plan fan.
3. **Panels** (top of sidebar) — `xy` is on by default; UAV scenes also have `xz` (side/altitude).
4. Drag the **Step** slider, or hit **▶** to play back the receding horizon.

A good first combo to confirm it's working: tick `env background` + `main bone` + one variant — you
should see the constraint triangles (avoiding) and a bold executed path immediately.

**Candidates vs. layers:** the numbered checkboxes under **Candidates** *select which candidates* draw.
They are **built per-scene** (one box per candidate in the fan, so the count matches the batch size, not
a hard-coded 0–3) and **all start checked — checked = shown**. Unticking some narrows the fan/mean to
the selected candidates; unticking **all** hides the fan entirely. (This replaced the old confusing
"none ticked ⇒ use ALL" behaviour, where selecting nothing still drew everything.) `MPC fan` (thin
lines) and `mean candidate` (one bold line, mean of the selected candidates) are independent **Layer**
toggles — tick either on its own and it draws immediately.

**If a view ever looks stuck or off-frame:** use the **View → Recenter** button (resets pan/zoom on
every panel) or **Redraw** (forces a re-render). Double-clicking a single panel resets just that one.

## Quick troubleshooting

| symptom | cause |
|---|---|
| Blank page, only a file-picker, no sidebar data | You opened the **template**, not a generated `viewer_<scene>.html`. Run Step 1. |
| Sidebar populated but the canvas is empty | Nothing is ticked yet — see the checklist above. Layers/variants/candidates all default off. |
| Red banner "No inlined data..." | Same as above — template with no data; either re-open the exported viewer file, or use the file-picker in the sidebar to load a `scene.json` manually. |
| **Trial dropdown shows only trial 0** after you re-exported with `--trials all` | The browser still has the **old** `scene.json` in memory — a page **refresh keeps it**. In the standalone tool, **re-pick the file** (or re-Load the path/URL) to force a fresh read. For an inlined `viewer_<scene>.html`, hard-refresh with **Ctrl+Shift+R** (HTML cache). Confirm you're on the new file: header reads `env=avoiding · 13 variants` and the Trial box lists 0 and 1. |
| Trial dropdown genuinely missing trials | The scene was exported with the default `--trials 0`. Re-run Step 1 with `--trials all`. |
| Console error on load | Check `scene.json` isn't truncated/corrupted — re-run the exporter. |

See `PLAN_npz_traj_visualizer.md` for the full design/architecture and `CHANGELOG_npz_traj_visualizer.md`
for what's built vs. deferred.
