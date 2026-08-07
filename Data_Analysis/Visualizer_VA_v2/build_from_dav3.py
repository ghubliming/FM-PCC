"""Regenerate Visualizer_VA_v2/index.html FROM the DAv3 page.

    python Data_Analysis/Visualizer_VA_v2/build_from_dav3.py

This page IS `Data_Analysis/Visualizer/index.html` plus the edits below, so every
DAv3 fix (U7 no-data messages, U8 value labels, U9 plot legend, U10 result
matrices, U10.1 LaTeX, U11 folder ZIP, seed modes, zoom) is inherited rather than
reimplemented. Re-run this after DAv3 gains something worth having; each edit
asserts its anchor, so a moved anchor fails loudly instead of silently dropping
half the page. Validate the result with test_page_offline.py.

"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / 'Visualizer' / 'index.html'      # the DAv3 page — never modified
DST = HERE / 'index.html'                            # this page — regenerated

# DAv3's file is CRLF; normalise so multi-line anchors match and the derived
# page has consistent LF endings.
html = SRC.read_text().replace('\r\n', '\n')
edits = 0


def sub(old, new, count=1):
    global html, edits
    if old not in html:
        print('ANCHOR NOT FOUND:\n' + old[:400])
        sys.exit(1)
    html = html.replace(old, new, count)
    edits += 1


# ── 1. identity ───────────────────────────────────────────────────────────────
sub('<title>FM-PCC Matrix Explorer</title>',
    '<title>FM-PCC — Visual Aligning Explorer v2</title>')
sub('<h1>FM-PCC EXPLORER</h1>',
    '<h1>FM-PCC VA EXPLORER v2</h1>')
sub('<div>SCIENTIFIC_SUITE_v3.12</div>',
    '<div>DA_VA_v2 &nbsp;|&nbsp; VISUAL ALIGNING SUITE U3</div>')
sub('placeholder="../analysis_results/batch_v3_.../candidates_multidimensional_aggregated.csv"',
    'placeholder="../analysis_results/batch_va2_.../  (folder, not a file)"')

# ── 2. extra CSS ──────────────────────────────────────────────────────────────
sub("""    </style>""",
    """    /* ---- DA_VA_v2 additions ---- */
        .mask-banner { border: 2px solid #000; padding: 9px 13px; font-size: 12px; background: #f1f8e9; margin-bottom: 4px; }
        .mask-banner.unfrozen { background: #e3f2fd; border-color: #1565c0; }
        .roll-tbl { table-layout: auto; font-size: 11px; }
        .roll-tbl th { background: #ececec; position: sticky; top: 0; z-index: 2; }
        tr.frozen-row td { background: #fff3e0; }
        .warn { color: #c62828; font-weight: bold; }
        .muted { color: #888; }
        .view-toggle { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 4px; }
    </style>""")

# ── 3. view tabs above the batch source ───────────────────────────────────────
sub("""    <div class="control-group">
        <label>1. Experiment Batch Source</label>""",
    """    <div class="control-group">
        <label>View</label>
        <div class="view-toggle">
            <div id="btn-agg"  class="mode-btn active" onclick="setViewMode('aggregate')">AGGREGATE</div>
            <div id="btn-roll" class="mode-btn"        onclick="setViewMode('rollout')">PER-ROLLOUT</div>
            <div id="btn-cmp"  class="mode-btn"        onclick="setViewMode('compare')">COMPARE</div>
            <div id="btn-qual" class="mode-btn"        onclick="setViewMode('quality')">QUALITY</div>
        </div>
    </div>
    <div class="control-group">
        <label>1. Experiment Batch Source</label>""")

# ── 4. global mask + split, at the top of the controls panel ──────────────────
sub("""        <div class="control-group"><label>2. Analysis Mode</label>""",
    """        <div class="control-group">
            <label>1.5 Rollout Mask (global)</label>
            <select id="mask-select" onchange="refresh_global()">
                <option value="all">all — every rollout</option>
                <option value="unfrozen">unfrozen — drop D1 box-obstacle conflicts</option>
            </select>
        </div>
        <div class="control-group">
            <label>1.6 Split (global)</label>
            <select id="split-select" onchange="refresh_global()"></select>
        </div>
        <div class="control-group"><label>2. Analysis Mode</label>""")

# ── 5. seeds come from the data, not a hardcoded 6..10 ────────────────────────
sub("""            <div class="checkbox-list" style="display:flex; flex-wrap:wrap; gap:5px;">
                <div class="checkbox-item"><input type="checkbox" class="seed-check" value="6" checked onchange="trigger_plot()"><span>6</span></div>
                <div class="checkbox-item"><input type="checkbox" class="seed-check" value="7" checked onchange="trigger_plot()"><span>7</span></div>
                <div class="checkbox-item"><input type="checkbox" class="seed-check" value="8" checked onchange="trigger_plot()"><span>8</span></div>
                <div class="checkbox-item"><input type="checkbox" class="seed-check" value="9" checked onchange="trigger_plot()"><span>9</span></div>
                <div class="checkbox-item"><input type="checkbox" class="seed-check" value="10" checked onchange="trigger_plot()"><span>10</span></div>
            </div>""",
    """            <div id="seed-list" class="checkbox-list" style="display:flex; flex-wrap:wrap; gap:5px;"></div>""")

# ── 6. geometry wording (this axis is geo here, not a halfspace name) ─────────
sub("""<label>4. Environment Focus</label>""",
    """<label>4. Geometry Focus</label>""")

# ── 7. per-rollout + compare sidebars, after the aggregate controls panel ─────
sub("""        <button class="btn-main" style="background:#444;" py-click="trigger_plot">REFRESH_RE_DRAW</button>
    </div>
</div>""",
    """        <button class="btn-main" style="background:#444;" py-click="trigger_plot">REFRESH_RE_DRAW</button>
    </div>

    <!-- PER-ROLLOUT (VA v1 specialty, rebuilt on the v2 axes) -->
    <div id="roll-controls" style="display:none; flex-direction:column; gap:16px;">
        <div class="control-group"><label>Candidate</label><select id="roll-cand-select" onchange="show_rollout_table()"></select></div>
        <div class="control-group"><label>Variant</label><select id="roll-var-select" onchange="show_rollout_table()"></select></div>
        <div class="control-group"><label>Sort by</label><select id="roll-sort-select" onchange="show_rollout_table()"></select></div>
        <div class="control-group">
            <label>Columns</label>
            <div class="bulk-actions">
                <span class="bulk-link" onclick="bulkSelect('rollcol-check', true)">[ALL]</span>
                <span class="bulk-link" onclick="bulkSelect('rollcol-check', false)">[NONE]</span>
            </div>
            <div id="rollcol-list" class="checkbox-list"></div>
        </div>
        <button class="btn-download" py-click="download_rollout_csv">DOWNLOAD THIS TABLE (CSV)</button>
    </div>

    <!-- COMPARE (VA v1 specialty) -->
    <div id="cmp-controls" style="display:none; flex-direction:column; gap:16px;">
        <div class="control-group"><label>Y metric (per rollout)</label><select id="cmp-y-select"></select></div>
        <div class="control-group"><label>X metric (per rollout)</label><select id="cmp-x-select"></select></div>
        <div class="control-group">
            <label>Chart</label>
            <div class="view-toggle">
                <div id="btn-scatter" class="mode-btn active" onclick="setCmpChart('scatter')">SCATTER</div>
                <div id="btn-bar"     class="mode-btn"        onclick="setCmpChart('bar')">BAR</div>
                <div id="btn-box"     class="mode-btn"        onclick="setCmpChart('box')">BOX</div>
            </div>
        </div>
        <div class="control-group">
            <label>Series</label>
            <select id="cmp-group-select"><option value="variant">Variant</option><option value="Candidate">Candidate</option></select>
        </div>
        <div class="control-group">
            <label>Variants</label>
            <div class="bulk-actions">
                <span class="bulk-link" onclick="bulkSelect('cmp-var-check', true)">[ALL]</span>
                <span class="bulk-link" onclick="bulkSelect('cmp-var-check', false)">[NONE]</span>
            </div>
            <div id="cmp-variant-list" class="checkbox-list"></div>
        </div>
        <div class="control-group">
            <label>Candidates</label>
            <div class="bulk-actions">
                <span class="bulk-link" onclick="bulkSelect('cmp-cand-check', true)">[ALL]</span>
                <span class="bulk-link" onclick="bulkSelect('cmp-cand-check', false)">[NONE]</span>
            </div>
            <div id="cmp-candidate-list" class="checkbox-list"></div>
        </div>
        <button class="btn-main" style="background:#c62828" py-click="trigger_compare">DRAW COMPARE</button>
        <button class="btn-download" id="cmp-download-btn" py-click="download_compare" style="display:none">DOWNLOAD PNG</button>
    </div>
</div>""")

# ── 8. main area: wrap DAv3's content, add the three new views ────────────────
sub("""    <div class="content">
        <div id="scorecard-area" """,
    """    <div class="content">
        <div id="mask-banner" class="mask-banner" style="display:none"></div>
        <div id="aggregate-area">
        <div id="scorecard-area" """)

sub("""        <div class="path-section" id="path-section" style="display:none"><label>Path Audit Map</label><div id="path-map-container"></div></div>
    </div>""",
    """        <div class="path-section" id="path-section" style="display:none"><label>Path Audit Map</label><div id="path-map-container"></div></div>
        </div><!-- /aggregate-area -->

        <div id="rollout-area" style="display:none">
            <div id="rollout-summary"></div>
            <div id="rollout-table-container" style="max-height:72vh;overflow:auto;border:1px solid #000"></div>
        </div>

        <div id="compare-area" style="display:none">
            <div id="compare-plot-area" style="background:#fff;border:1px solid #000;padding:14px;min-height:460px;text-align:center;overflow-x:auto">
                <p style="padding-top:180px;color:#999">SELECT VARIANTS AND DRAW</p>
            </div>
        </div>

        <div id="quality-area" style="display:none">
            <div id="quality-container"></div>
        </div>
    </div>""")

# ── 9. run picker: this page lists batch_va2_* runs ──────────────────────────
sub("""            # Find all links that look like batch_v3_ folders
            batches = re.findall(r'href="(batch_[^/"]+)/?"', html)""",
    """            # DA_VA_v2 runs only — a batch_va2_* folder is the one carrying the
            # native va2_*.csv cube this page reads.
            batches = re.findall(r'href="(batch_va2_[^/"]+)/?"', html)""")
sub("""                data = await resp2.json()
                batches = data.get("batches", [])""",
    """                data = await resp2.json()
                batches = [b for b in data.get("batches", []) if str(b).startswith("batch_va2_")]""")

# ── 10. loader: native CSVs in, DAv3 column names out ────────────────────────
# The DAv3 loader block is located by its boundaries rather than quoted verbatim —
# that file has trailing whitespace a literal anchor cannot survive.
_a = html.index("async def load_data(event=None):")
_b = html.index("def populate_dynamic_filters():")
old_loader_start = html[_a:_b]

new_loader = '''# ---------------------------------------------------------------------------------------
# DA_VA_v2 source frames. The page reads the NATIVE cube and projects it onto the column
# names the DAv3 core below already speaks (halfspace_variant / mean / std / count / value),
# so every inherited feature keeps working while mask + split become live switches.
# ---------------------------------------------------------------------------------------
VA2_FILES = {'agg': 'va2_aggregated_long.csv', 'units': 'va2_units_long.csv',
             'roll': 'per_rollout_detail.csv', 'qual': 'data_quality.csv'}
df_agg_src = None      # va2_aggregated_long.csv   (pooled over seeds)
df_units_src = None    # va2_units_long.csv        (per seed)
df_roll_src = None     # per_rollout_detail.csv    (per rollout)
df_qual_src = None     # data_quality.csv          (per unit)
current_roll_table = None
compare_fig = None

ROLL_DEFAULT_COLUMNS = ['rollout_idx', 'frozen', 'n_success', 'success_relaxed',
                        'mean_dist_per_rollout', 'context_final_xy_dist',
                        'constraint_exec_sat_rate', 'n_violations', 'max_viol_depth_m',
                        'n_steps', 'avg_time_ms', 'max_phys_error_per_rollout']
ROLL_SORT_OPTIONS = [('rollout_idx', 'Rollout index'), ('n_success', 'Success'),
                     ('mean_dist_per_rollout', 'Final distance'),
                     ('context_final_xy_dist', 'Final XY distance'),
                     ('constraint_exec_sat_rate', 'Constraint sat rate'),
                     ('n_violations', 'Violated steps'), ('avg_time_ms', 'Replan time')]


def _norm(df):
    if df is None:
        return None
    for col in ('Candidate', 'variant', 'geo', 'split', 'mask', 'metric', 'FolderName'):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


def current_mask():
    el = document.getElementById("mask-select")
    return el.value if el is not None else "all"


def current_split():
    el = document.getElementById("split-select")
    return el.value if el is not None else "ALL"


def _slice(df, use_mask=True):
    """Global mask + split, on either CSV shape.

    The long CSVs carry BOTH reductions of every metric in a `mask` COLUMN — pick
    one or every row counts twice. per_rollout_detail has no such column, just a
    `frozen` flag per rollout, so the same switch has to drop rows there instead;
    without this the mask silently does nothing in the rollout/compare views.
    """
    if df is None or len(df) == 0:
        return df
    out = df
    mask, split = current_mask(), current_split()
    if use_mask:
        if 'mask' in out.columns:
            out = out[out['mask'] == mask]
        elif mask == 'unfrozen' and 'frozen' in out.columns:
            out = out[out['frozen'].fillna(0) != 1]
    if split != 'ALL' and 'split' in out.columns:
        out = out[out['split'] == split]
    return out


# U3: the relaxed goal+constraint rate. The DA pipeline only derives the STRICT
# product (n_success * zero_violation), and the relaxed one canNOT be recovered
# from the long CSVs after the fact — mean(a*b) != mean(a)*mean(b) once the two
# flags disagree on any single rollout. per_rollout_detail is the only frame that
# still has both side by side, so it is rebuilt here, per rollout, then reduced
# exactly like the pipeline reduces everything else. A future batch that ships
# the column natively wins: the synthesis is skipped (see derive_frames).
DERIVED_METRIC = 'n_success_relaxed_and_constraints'
DERIVED_INPUTS = ('success_relaxed', 'constraint_exec_zero_violation')


def _derived_rows(roll, keys):
    """mean/std/n of DERIVED_METRIC over rollouts, grouped by `keys`. None if unavailable.

    Grouping mirrors the pipeline's own reduction (split included), so with
    split=ALL these rows duplicate per split exactly like the native ones and the
    matrices average them the same way. per_rollout_detail carries FolderName but
    NOT FullPath, so the path column is filled from df_agg afterwards.
    """
    if roll is None or len(roll) == 0:
        return None
    if any(c not in roll.columns for c in DERIVED_INPUTS):
        return None
    keys = [k for k in keys if k in roll.columns]
    if not keys:
        return None
    work = roll[keys].copy()
    work['__v'] = (pd.to_numeric(roll[DERIVED_INPUTS[0]], errors='coerce').values
                   * pd.to_numeric(roll[DERIVED_INPUTS[1]], errors='coerce').values)
    grouped = work.groupby(keys, observed=True, dropna=False)['__v']
    out = pd.DataFrame({'mean': grouped.mean(), 'std': grouped.std(),
                        'n': grouped.count()}).reset_index()
    return out[out['n'] > 0]


def derive_frames():
    """Rebuild df_agg / df_raw for the current mask+split, in DAv3's schema."""
    global df_agg, df_raw, data_version
    agg = _slice(df_agg_src)
    df_agg = pd.DataFrame({
        'Candidate': agg['Candidate'].values,
        'Folder_Name': agg['FolderName'].values,
        'Full_Path': agg['FullPath'].values,
        'variant': agg['variant'].values,
        'halfspace_variant': agg['geo'].values,     # geometry IS this page's environment axis
        'metric': agg['metric'].values,
        'mean': agg['mean'].values,
        'std': agg['std'].values,
        'count': agg['n'].values,                   # pooled rollouts -> SEM is over rollouts
    })
    roll = _slice(df_roll_src)                      # mask here drops the frozen ROWS
    paths = dict(zip(df_agg['Candidate'], df_agg['Full_Path']))
    if DERIVED_METRIC not in set(df_agg['metric'].unique()):
        extra = _derived_rows(roll, ['Candidate', 'FolderName', 'split',
                                     'geo', 'variant'])
        if extra is not None and len(extra):
            df_agg = pd.concat([df_agg, pd.DataFrame({
                'Candidate': extra['Candidate'].values,
                'Folder_Name': extra['FolderName'].values,
                'Full_Path': extra['Candidate'].map(paths).values,
                'variant': extra['variant'].values,
                'halfspace_variant': extra['geo'].values,
                'metric': DERIVED_METRIC,
                'mean': extra['mean'].values,
                'std': extra['std'].values,
                'count': extra['n'].values,
            })], ignore_index=True)
    if df_units_src is not None:
        units = _slice(df_units_src)
        df_raw = pd.DataFrame({
            'Candidate': units['Candidate'].values,
            'Folder_Name': units['FolderName'].values,
            'Full_Path': units['FullPath'].values,
            'seed': units['seed'].values,
            'variant': units['variant'].values,
            'halfspace_variant': units['geo'].values,
            'metric': units['metric'].values,
            'value': units['mean'].values,
        })
        if DERIVED_METRIC not in set(df_raw['metric'].unique()):
            extra = _derived_rows(roll, ['Candidate', 'FolderName', 'split',
                                         'seed', 'geo', 'variant'])
            if extra is not None and len(extra):
                df_raw = pd.concat([df_raw, pd.DataFrame({
                    'Candidate': extra['Candidate'].values,
                    'Folder_Name': extra['FolderName'].values,
                    'Full_Path': extra['Candidate'].map(paths).values,
                    'seed': extra['seed'].values,
                    'variant': extra['variant'].values,
                    'halfspace_variant': extra['geo'].values,
                    'metric': DERIVED_METRIC,
                    'value': extra['mean'].values,
                })], ignore_index=True)
    else:
        df_raw = None
    data_version += 1      # busts the result-matrix cache: mask/split changed the numbers


async def _fetch_csv(base, name, required=False):
    try:
        response = await fetch(f"{base}/{name}")
        if not response.ok:
            if required:
                raise Exception(f"HTTP {response.status} for {name}")
            return None
        return _norm(pd.read_csv(io.StringIO(await response.text()), low_memory=False))
    except Exception as e:
        if required:
            raise
        console.log(f"[ loader ] optional {name} unavailable: {e}")
        return None


async def load_data(event=None):
    global df_agg_src, df_units_src, df_roll_src, df_qual_src, data_version
    mode = window.currentMode
    if mode == "list":
        batch = document.getElementById("batch-select").value
        if not batch:
            document.getElementById("engine-status").innerText = "NO_BATCH_SELECTED"
            return
        base = f"../analysis_results/{batch}"
    else:
        base = document.getElementById("custom-path").value.strip().rstrip("/")
        for name in VA2_FILES.values():          # accept a CSV path or the folder
            if base.endswith(name):
                base = base[: -len(name)].rstrip("/")

    try:
        document.getElementById("engine-status").innerText = "FETCHING..."
        df_agg_src = await _fetch_csv(base, VA2_FILES['agg'], required=True)
        df_units_src = await _fetch_csv(base, VA2_FILES['units'])
        df_roll_src = await _fetch_csv(base, VA2_FILES['roll'])
        df_qual_src = await _fetch_csv(base, VA2_FILES['qual'])

        populate_split_filter()
        derive_frames()
        document.getElementById("controls-panel").style.display = "flex"
        document.getElementById("download-btn").style.display = "block"
        document.getElementById("path-section").style.display = "block"
        document.getElementById("scorecard-area").style.display = "block"
        populate_dynamic_filters()
        render_path_map()
        render_mask_banner()
        window.applyViewMode()          # show the panels for whatever view is active
        await trigger_plot(None)
        document.getElementById("engine-status").innerText = "SYNC_COMPLETE"
    except Exception as e:
        document.getElementById("engine-status").innerText = "LOAD_ERROR"
        console.log(str(e))
        document.getElementById("plot-area").innerHTML = (
            '<div style="padding:40px;color:#b71c1c;font-family:monospace;font-size:13px">'
            f'<b>COULD NOT LOAD {VA2_FILES["agg"]}</b><br><br>{e}<br><br>'
            f'tried <code>{base}/{VA2_FILES["agg"]}</code><br><br>'
            'This page reads the DA_VA_v2 native CSVs — point it at a '
            '<code>batch_va2_*</code> folder.</div>')'''

sub(old_loader_start, new_loader + "\n\n\n")

# ── 11. populate the v2-specific widgets after DAv3's own ────────────────────
sub("""    cands_list = sorted(df_agg['Candidate'].unique(), key=lambda c: (0, int(c)) if c.isdigit() else (1, c))
    document.getElementById("candidate-list").innerHTML = "".join([f'<div class="checkbox-item"><input type="checkbox" class="cand-check" value="{c}" onchange="trigger_plot()"> <span>CAND_{c}</span></div>' for c in cands_list])""",
    """    cands_list = sorted(df_agg['Candidate'].unique(), key=lambda c: (0, int(c)) if c.isdigit() else (1, c))
    document.getElementById("candidate-list").innerHTML = "".join([f'<div class="checkbox-item"><input type="checkbox" class="cand-check" value="{c}" onchange="trigger_plot()"> <span>CAND_{c}</span></div>' for c in cands_list])
    populate_va2_filters(vars_list, cands_list)""")

# ── 12. the v2 view code, inserted just before the wiring ────────────────────
va2_block = '''
# =======================================================================================
# DA_VA_v2 views: seeds/split widgets, mask banner, per-rollout table, rollout compare,
# data quality. Everything above this point is the DAv3 page, unchanged.
# =======================================================================================

def populate_split_filter():
    splits = sorted(df_agg_src['split'].unique()) if 'split' in df_agg_src.columns else []
    options = ['ALL'] + splits
    document.getElementById("split-select").innerHTML = "".join(
        f'<option value="{s}">{s.upper()}</option>' for s in options)


def populate_va2_filters(vars_list, cands_list):
    # Seeds come from the batch, not a hardcoded 6..10 — a visual-aligning run is
    # routinely single-seed, and five dead checkboxes read as a broken filter.
    seeds = []
    if df_units_src is not None and 'seed' in df_units_src.columns:
        seeds = sorted(int(s) for s in pd.to_numeric(df_units_src['seed'], errors='coerce').dropna().unique())
    document.getElementById("seed-list").innerHTML = "".join(
        f'<div class="checkbox-item"><input type="checkbox" class="seed-check" value="{s}" '
        f'checked onchange="trigger_plot()"><span>{s}</span></div>' for s in seeds)

    document.getElementById("roll-cand-select").innerHTML = "".join(
        f'<option value="{c}">CAND_{c}</option>' for c in cands_list)
    document.getElementById("roll-var-select").innerHTML = "".join(
        f'<option value="{v}">{v}</option>' for v in vars_list)
    document.getElementById("roll-sort-select").innerHTML = "".join(
        f'<option value="{k}">{lbl}</option>' for k, lbl in ROLL_SORT_OPTIONS)

    document.getElementById("cmp-variant-list").innerHTML = "".join(
        f'<div class="checkbox-item"><input type="checkbox" class="cmp-var-check" value="{v}"> '
        f'<span>{v}</span></div>' for v in vars_list)
    document.getElementById("cmp-candidate-list").innerHTML = "".join(
        f'<div class="checkbox-item"><input type="checkbox" class="cmp-cand-check" value="{c}"> '
        f'<span>CAND_{c}</span></div>' for c in cands_list)

    if df_roll_src is not None:
        skip = {'Candidate', 'FolderName', 'FullPath', 'seed', 'split', 'geo',
                'geo_base', 'tightened', 'variant', 'variant_raw'}
        metrics = [c for c in df_roll_src.columns if c not in skip]
        def _opts(sel):
            return "".join(f'<option value="{m}"{" selected" if m == sel else ""}>{m}</option>'
                           for m in metrics)
        document.getElementById("cmp-y-select").innerHTML = _opts('mean_dist_per_rollout')
        document.getElementById("cmp-x-select").innerHTML = _opts('context_init_xy_dist')
        shown = [c for c in ROLL_DEFAULT_COLUMNS if c in metrics]
        rest = [c for c in metrics if c not in shown]
        document.getElementById("rollcol-list").innerHTML = "".join(
            f'<div class="checkbox-item"><input type="checkbox" class="rollcol-check" value="{c}" '
            f'{"checked" if c in shown else ""} onchange="show_rollout_table()"> <span>{c}</span></div>'
            for c in shown + rest)


def render_mask_banner():
    total = frozen = None
    if df_qual_src is not None and len(df_qual_src):
        if 'n_rollouts' in df_qual_src.columns:
            total = int(pd.to_numeric(df_qual_src['n_rollouts'], errors='coerce').fillna(0).sum())
        if 'n_frozen' in df_qual_src.columns:
            frozen = int(pd.to_numeric(df_qual_src['n_frozen'], errors='coerce').fillna(0).sum())
    elif df_roll_src is not None and len(df_roll_src):
        total = int(len(df_roll_src))
        frozen = int(pd.to_numeric(df_roll_src.get('frozen', 0), errors='coerce').fillna(0).sum())

    mask, split = current_mask(), current_split()
    if total is None or frozen is None:
        text = "frozen-rollout counts unavailable (data_quality.csv not in this batch)"
    elif mask == 'unfrozen':
        text = (f"mask = UNFROZEN — {frozen} of {total} rollouts excluded (D1 box-obstacle "
                f"conflict: the eval held position and never called the model, so they report "
                f"sat_rate 1.0 and inflate every constraint number)")
    else:
        text = (f"mask = ALL — every rollout kept, of which {frozen} of {total} rollouts are "
                f"D1-FROZEN (model never called; they inflate constraint aggregates). "
                f"Switch to UNFROZEN to exclude them.")
    banner = document.getElementById("mask-banner")
    banner.className = "mask-banner unfrozen" if mask == "unfrozen" else "mask-banner"
    banner.innerHTML = (f"<b>{text}</b><br><span class='muted'>split = {split} &middot; "
                        f"geometry axis = geo &middot; SEM in the matrices is over rollouts</span>")
    banner.style.display = "block"


async def refresh_global(event=None):
    """A global (mask / split) change re-derives every frame and redraws the view."""
    if df_agg_src is None:
        return
    derive_frames()
    render_mask_banner()
    view = window.currentView
    if view == "aggregate":
        await trigger_plot(None)
    elif view == "rollout":
        show_rollout_table()
    elif view == "quality":
        render_quality()


# ── per-rollout table ──────────────────────────────────────────────────────────────────

def show_rollout_table(event=None):
    global current_roll_table
    render_mask_banner()
    container = document.getElementById("rollout-table-container")
    summary = document.getElementById("rollout-summary")
    if df_roll_src is None:
        container.innerHTML = ""
        summary.innerHTML = ('<p class="warn">per_rollout_detail.csv is not in this batch — '
                             'the per-rollout view needs it.</p>')
        return

    data = _slice(df_roll_src)
    geo = document.getElementById("env-select").value
    if geo and 'geo' in data.columns:
        data = data[data['geo'] == geo]
    cand = document.getElementById("roll-cand-select").value
    variant = document.getElementById("roll-var-select").value
    sort_by = document.getElementById("roll-sort-select").value

    subset = data[(data['Candidate'] == cand) & (data['variant'] == variant)]
    if subset.empty:
        container.innerHTML = ""
        summary.innerHTML = (f'<p class="warn">No rollouts for CAND_{cand} / {variant} at '
                             f'geo={geo}, mask={current_mask()}, split={current_split()}.</p>')
        current_roll_table = None
        return

    ascending = sort_by not in ('n_success', 'success_relaxed', 'constraint_exec_sat_rate')
    if sort_by in subset.columns:
        subset = subset.sort_values(sort_by, ascending=ascending, kind='stable')

    columns = [c for c in get_checked("rollcol-check") if c in subset.columns]
    if not columns:
        columns = [c for c in ROLL_DEFAULT_COLUMNS if c in subset.columns]
    # With split on ALL the same rollout_idx appears once per (seed, split) — show
    # whichever identifier actually varies or the rows read as duplicates.
    context = [c for c in ('seed', 'split') if c in subset.columns
               and subset[c].nunique() > 1 and c not in columns]
    columns = context + columns
    current_roll_table = subset[['Candidate', 'seed', 'split', 'geo', 'variant']
                                + [c for c in columns if c not in ('seed', 'split')]]

    n_frozen = int(pd.to_numeric(subset.get('frozen', 0), errors='coerce').fillna(0).sum())
    succ = pd.to_numeric(subset.get('n_success'), errors='coerce').mean() if 'n_success' in subset.columns else float('nan')
    summary.innerHTML = (
        f'<p><b>CAND_{cand} &middot; {variant} &middot; geo {geo}</b> &mdash; {len(subset)} rollouts, '
        f'{n_frozen} frozen, success {"n/a" if succ != succ else f"{succ * 100:.1f}%"} '
        f'&middot; sorted by <code>{sort_by}</code> &middot; frozen rows are tinted</p>')

    head = "".join(f'<th>{c}</th>' for c in columns)
    rows = []
    for _, row in subset.iterrows():
        cells = []
        for c in columns:
            v = row[c]
            if isinstance(v, float):
                cells.append('<td>&mdash;</td>' if v != v else f'<td>{v:.4g}</td>')
            else:
                cells.append(f'<td>{v}</td>')
        klass = ' class="frozen-row"' if float(row.get('frozen', 0) or 0) == 1 else ''
        rows.append(f'<tr{klass}>' + "".join(cells) + '</tr>')
    container.innerHTML = (f'<table class="roll-tbl"><thead><tr>{head}</tr></thead>'
                           f'<tbody>{"".join(rows)}</tbody></table>')


def download_rollout_csv(event=None):
    if current_roll_table is None:
        document.getElementById("engine-status").innerText = "NOTHING_TO_DOWNLOAD"
        return
    _download_bytes(current_roll_table.to_csv(index=False).encode(),
                    f"rollouts_{current_mask()}.csv", "text/csv")
    document.getElementById("engine-status").innerText = "ROLLOUT_CSV_SAVED"


# ── rollout compare ────────────────────────────────────────────────────────────────────

def trigger_compare(event=None):
    global compare_fig
    render_mask_banner()
    area = document.getElementById("compare-plot-area")
    if df_roll_src is None:
        area.innerHTML = ('<p class="warn" style="padding-top:160px">per_rollout_detail.csv is '
                          'not in this batch — the compare view needs it.</p>')
        return

    data = _slice(df_roll_src)
    geo = document.getElementById("env-select").value
    if geo and 'geo' in data.columns:
        data = data[data['geo'] == geo]
    x_metric = document.getElementById("cmp-x-select").value
    y_metric = document.getElementById("cmp-y-select").value
    group_by = document.getElementById("cmp-group-select").value
    variants = get_checked("cmp-var-check")
    cands = get_checked("cmp-cand-check")

    if not variants or not cands:
        area.innerHTML = ('<p style="padding-top:160px;color:#666">Select at least one variant '
                          'and one candidate, then DRAW COMPARE.</p>')
        return
    subset = data[data['variant'].isin(variants) & data['Candidate'].isin(cands)]
    if subset.empty:
        area.innerHTML = (f'<p class="warn" style="padding-top:160px">No rollouts for that '
                          f'selection at geo={geo}, mask={current_mask()}.</p>')
        return
    for col in (x_metric, y_metric):
        if col not in subset.columns:
            area.innerHTML = f'<p class="warn" style="padding-top:160px">Column {col} not in CSV.</p>'
            return
    subset = subset.dropna(subset=[x_metric, y_metric])
    if subset.empty:
        area.innerHTML = (f'<p class="warn" style="padding-top:160px">Every row is NaN for '
                          f'{x_metric} / {y_metric}.</p>')
        return

    chart = window.cmpChartType
    groups = sorted(subset[group_by].astype(str).unique())
    try:
        colours = matplotlib.colormaps['tab20'].colors
    except Exception:
        colours = plt.get_cmap('tab20').colors

    plt.close('all')
    fig, ax = plt.subplots(figsize=(11, 6))
    if chart == 'scatter':
        for i, name in enumerate(groups):
            block = subset[subset[group_by].astype(str) == name]
            ax.scatter(block[x_metric], block[y_metric], s=38, alpha=0.75, edgecolors='k',
                       linewidths=0.3, label=name, color=colours[i % len(colours)])
        ax.set_xlabel(x_metric)
        ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    elif chart == 'bar':
        means = subset.groupby(group_by)[y_metric].mean().reindex(groups)
        stds = subset.groupby(group_by)[y_metric].std().reindex(groups)
        ax.bar(range(len(groups)), means.values, yerr=stds.values, capsize=4, edgecolor='k',
               color=[colours[i % len(colours)] for i in range(len(groups))])
        ax.set_xticks(list(range(len(groups))))
        ax.set_xticklabels(groups, rotation=35, ha='right', fontsize=8)
    else:
        payload, labels = [], []
        for name in groups:
            vals = subset[subset[group_by].astype(str) == name][y_metric].dropna().values
            if len(vals):
                payload.append(vals)
                labels.append(name)
        if not payload:
            area.innerHTML = '<p class="warn" style="padding-top:160px">Nothing to box.</p>'
            return
        # No labels=/tick_labels= kwarg: the two spellings swapped over matplotlib 3.9
        # and pyodide pins an older build than the cluster.
        ax.boxplot(payload)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)

    ax.set_ylabel(y_metric)
    ax.grid(True, alpha=0.2)
    ax.set_title(f"{y_metric} by {group_by}  |  geo={geo}  mask={current_mask()}  "
                 f"split={current_split()}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    compare_fig = fig
    area.innerHTML = ""
    display(fig, target="compare-plot-area")
    document.getElementById("cmp-download-btn").style.display = "block"
    document.getElementById("engine-status").innerText = f"COMPARE_DRAWN ({len(subset)} rollouts)"


def download_compare(event=None):
    if compare_fig is None:
        document.getElementById("engine-status").innerText = "NOTHING_TO_DOWNLOAD"
        return
    buf = io.BytesIO()
    compare_fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    _download_bytes(buf.getvalue(), "compare.png", "image/png")
    document.getElementById("engine-status").innerText = "COMPARE_PNG_SAVED"


# ── data quality ───────────────────────────────────────────────────────────────────────

def render_quality():
    container = document.getElementById("quality-container")
    if df_qual_src is None:
        container.innerHTML = '<p class="warn">data_quality.csv is not in this batch.</p>'
        return
    data = _slice(df_qual_src, use_mask=False)      # quality rows have no mask axis
    flags = None
    for col in ('n_frozen', 'n_cb_tripped'):
        if col in data.columns:
            hit = pd.to_numeric(data[col], errors='coerce').fillna(0) > 0
            flags = hit if flags is None else (flags | hit)
    if 'npz_complete' in data.columns:
        hit = pd.to_numeric(data['npz_complete'], errors='coerce').fillna(1) == 0
        flags = hit if flags is None else (flags | hit)
    flagged = data[flags] if flags is not None else data.iloc[0:0]

    if flagged.empty:
        container.innerHTML = (f'<p style="color:#2e7d32;font-weight:bold">All {len(data)} units '
                               f'clean — no frozen rollouts, no circuit-breaker trips, no partial npz.</p>')
        return
    columns = [c for c in ('Candidate', 'seed', 'split', 'geo', 'variant', 'source', 'n_rollouts',
                           'n_frozen', 'frozen_rate', 'n_cb_tripped', 'cb_skipped_steps',
                           'npz_complete') if c in data.columns]
    head = "".join(f'<th>{c}</th>' for c in columns)
    rows = []
    for _, row in flagged.iterrows():
        cells = []
        for c in columns:
            v = row[c]
            if isinstance(v, float):
                cells.append('<td>&mdash;</td>' if v != v else f'<td>{v:.4g}</td>')
            else:
                cells.append(f'<td>{v}</td>')
        rows.append("<tr>" + "".join(cells) + "</tr>")
    container.innerHTML = (
        f'<p><b>{len(flagged)} of {len(data)} units flagged.</b> <span class="muted">Frozen '
        f'rollouts never called the model; circuit-breaker rollouts ran (partly) unprojected, so '
        f'their constraint metrics do not describe the policy the variant name claims; '
        f'npz_complete=0 is a crash-safety partial.</span></p>'
        f'<table class="roll-tbl"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>')

'''

sub("""document.trigger_plot = create_proxy(trigger_plot)
document.download_folder = create_proxy(download_folder)
asyncio.ensure_future(init_manifest())""",
    va2_block + """
document.trigger_plot = create_proxy(trigger_plot)
document.download_folder = create_proxy(download_folder)
document.refresh_global = create_proxy(refresh_global)
document.show_rollout_table = create_proxy(show_rollout_table)
document.trigger_compare = create_proxy(trigger_compare)
document.render_quality = create_proxy(render_quality)
asyncio.ensure_future(init_manifest())""")

# ── 12b. U3: three more Result Matrices (VA distance + the relaxed pair) ─────
# DAv3 ships four tables, all of them state-only-safe. The visual-aligning eval
# adds a distance-to-goal and a second, position-only success definition, and
# neither has anywhere else to appear in a paper table. Rows/columns/SEM/flag
# machinery is untouched — this is purely three more entries in the list that
# drives BOTH the on-page tables and the LaTeX export.
sub("""SUMMARY_TABLES = [
    ("n_success", "N_SUCCESS -- success rate (goal reached)", "{:.3f}", False),
    ("n_success_and_constraints", "N_SUCCESS + CONSTRAINT -- success rate (goal reached AND constraints satisfied)", "{:.3f}", False),
    ("n_steps", "N_STEPS -- steps taken (episode length)", "{:.1f}", True),
    ("avg_time", "AVG_TIME -- average computation time per planning step [s]", "{:.4f}", True),
]""",
    """SUMMARY_TABLES = [
    ("n_success", "N_SUCCESS -- success rate (goal reached: strict, position AND rotation)", "{:.3f}", False),
    ("n_success_and_constraints", "N_SUCCESS + CONSTRAINT -- success rate (goal reached AND constraints satisfied)", "{:.3f}", False),
    # U3 (visual-aligning only; the state-only avoiding trees leave these columns
    # absent, so their cells render as the honest NULL dash rather than a zero).
    ("success_relaxed", "N_SUCCESS RELAXED -- success rate (position only: final box-target XY distance within pos_min_dist; final angle ignored)", "{:.3f}", False),
    ("n_success_relaxed_and_constraints", "N_SUCCESS RELAXED + CONSTRAINT -- relaxed success AND constraints satisfied", "{:.3f}", False),
    ("mean_dist_per_rollout", "MIN_DIST -- final box-target distance [m], the env's own score (half the position error plus half the rotation error) measured against pos/rot_min_dist -- lower is better", "{:.4f}", True),
    ("n_steps", "N_STEPS -- steps taken (episode length)", "{:.1f}", True),
    ("avg_time", "AVG_TIME -- average computation time per planning step [s]", "{:.4f}", True),
]""")

# the caption block above the tables counts them and names the flagged ones
sub("""              'In the N_STEPS and AVG_TIME tables a trailing <span class="flag">(goal, constraint)</span> marks a cell where '""",
    """              'In the MIN_DIST, N_STEPS and AVG_TIME tables a trailing <span class="flag">(goal, constraint)</span> marks a cell where '""")
sub("""              '<b>EXPORT ZIP</b> writes exactly these four tables to a .tex file inside the archive.'""",
    """              f'<b>EXPORT ZIP</b> writes exactly these {len(SUMMARY_TABLES)} tables to a .tex file inside the archive.'""")
sub("""    # 2. LaTeX — the four Result Matrices, exactly as rendered on the page.""",
    """    # 2. LaTeX — every Result Matrix, exactly as rendered on the page.""")
sub("""# U10.1: LaTeX export of the same four matrices — one standalone, compilable .tex file.""",
    """# U10.1: LaTeX export of the same matrices — one standalone, compilable .tex file.""")

# ── 13. matplotlib module handle for the colormap fallback ───────────────────
sub("""import pandas as pd
import matplotlib.pyplot as plt""",
    """import pandas as pd
import matplotlib
import matplotlib.pyplot as plt""")

# ── 14. JS: view switching (and it must run on load, not only on click) ──────
sub("""    window.currentMode = "list";
    function switchMode(mode) {""",
    """    window.currentMode = "list";
    window.currentView = "aggregate";
    window.cmpChartType = "scatter";

    // Applying the view is a separate function from clicking a tab: load_data calls it
    // so the active view's sidebar panel is visible the moment data arrives. (Without
    // this the aggregate controls stay display:none until the user happens to click a
    // tab, which reads as a completely broken page.)
    function applyViewMode() {
        const view  = window.currentView;
        const btns  = {aggregate: "btn-agg", rollout: "btn-roll", compare: "btn-cmp", quality: "btn-qual"};
        const ctrls = {aggregate: "controls-panel", rollout: "roll-controls", compare: "cmp-controls", quality: null};
        const areas = {aggregate: "aggregate-area", rollout: "rollout-area", compare: "compare-area", quality: "quality-area"};
        const loaded = document.getElementById("path-section").style.display !== "none";
        for (const v of ["aggregate", "rollout", "compare", "quality"]) {
            document.getElementById(btns[v]).classList.toggle("active", v === view);
            if (ctrls[v]) {
                const show = (v === view) && (v !== "aggregate" || loaded);
                document.getElementById(ctrls[v]).style.display = show ? "flex" : "none";
            }
            document.getElementById(areas[v]).style.display = (v === view) ? "block" : "none";
        }
    }
    window.applyViewMode = applyViewMode;

    function setViewMode(view) {
        window.currentView = view;
        applyViewMode();
        if (view === "rollout" && window.document.show_rollout_table) window.document.show_rollout_table();
        if (view === "quality" && window.document.render_quality) window.document.render_quality();
    }

    function setCmpChart(type) {
        window.cmpChartType = type;
        document.getElementById("btn-scatter").classList.toggle("active", type === "scatter");
        document.getElementById("btn-bar").classList.toggle("active", type === "bar");
        document.getElementById("btn-box").classList.toggle("active", type === "box");
    }

    async function refresh_global() {
        if (window.document.refresh_global) {
            try { await window.document.refresh_global(); } catch (e) { console.error(e); }
        }
    }
    function show_rollout_table() {
        if (window.document.show_rollout_table) window.document.show_rollout_table();
    }
    function trigger_compare() {
        if (window.document.trigger_compare) window.document.trigger_compare();
    }

    function switchMode(mode) {""")

# bulkSelect must not fire a plot redraw for the rollout/compare lists
sub("""    function bulkSelect(className, state) {
        const elements = document.getElementsByClassName(className);
        for (let e of elements) { e.checked = state; }
        trigger_plot();
    }""",
    """    function bulkSelect(className, state) {
        const elements = document.getElementsByClassName(className);
        for (let e of elements) { e.checked = state; }
        if (className === "rollcol-check") show_rollout_table();
        else if (className === "cmp-var-check" || className === "cmp-cand-check") { /* wait for DRAW */ }
        else trigger_plot();
    }""")

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(html)
print(f'{edits} edits applied -> {DST}  ({len(html.splitlines())} lines)')
