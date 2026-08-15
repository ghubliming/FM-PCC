"""Regenerate Visualizer_UAV_v1/index.html FROM the DA_VA_v2 page.

    python Data_Analysis/Visualizer_UAV_v1/build_from_va2.py

This page IS `Data_Analysis/Visualizer_VA_v2/index.html` plus the edits below,
which is itself `Data_Analysis/Visualizer/index.html` (DAv3) plus its own — so
every fix in either ancestor (U7 no-data messages, U8 value labels, U9 plot
legend, U10 result matrices, U10.1 LaTeX, U11 folder ZIP, U13 candidate
highlight + legend seed coverage, U14 (G, C) plot flags, U15 distinct variant
colours, U17 audit-map seeds, Last Run stamps, seed modes, zoom, and VA v2's
per-rollout / compare / quality views) is inherited rather than reimplemented.

Re-run this after either ancestor gains something worth having. Each edit
asserts its anchor, so a moved anchor fails loudly instead of silently dropping
half the page.

What this layer changes, and why
--------------------------------

1. **Identity + which CSVs it reads.** `uav_aggregated_long.csv` /
   `uav_units_long.csv`, and a `batch_uav_*` run picker.

2. **The mask means something different.** VA v2's mask drops D1-FROZEN
   rollouts (the eval held position and never called the model). UAV has no D1
   guard; its equivalent hazard is the projection circuit breaker — a rollout
   that ran (partly) UNPROJECTED because SLSQP was too slow, whose constraint
   numbers therefore describe a policy the variant name does not name. So
   `unfrozen` becomes `proj_valid` and the flag column becomes
   `projection_cb_tripped`.

3. **Tightening is back on the VARIANT axis.** UAV enumerates `-tightened`
   variants explicitly in config/uav_projection.yaml instead of generating a
   tightened geometry twin, so all three variant presets have members here and
   the "not offered by this batch" hint VA v2 shows does not apply.

4. **A UAV axis panel (scene / engine / K).** This is the one genuinely new
   control. The Gen15 experiment is a K sweep across engines and scenes (PLAN
   §7.3), which means a batch is a 4-scene x 4-engine x 5-K cross product and
   every plot needs slicing along axes the DAv3 page has no concept of. They are
   real columns in the DA_UAV_v1 CSVs, so the panel is a filter on them, applied
   in `_slice` next to mask and split — i.e. it narrows every view at once,
   including the matrices and the per-rollout table.

5. **UAV metrics everywhere a metric list is hardcoded**: the result-matrix
   tables, the per-rollout default columns and sort options, the compare-view
   default axes, and the reference row (step budget under steps-per-episode,
   because `n_steps` = 396 is meaningless until you know 396 IS the budget).

Validate the result by serving the repo root and opening the page; there is no
offline test harness for this layer (the VA v2 ones live next to that page).
"""
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / 'Visualizer_VA_v2' / 'index.html'   # the VA v2 page — never modified
DST = HERE / 'index.html'                               # this page — regenerated

html = SRC.read_text().replace('\r\n', '\n')
edits = 0


def sub(old, new, count=1):
    global html, edits
    if old not in html:
        print('ANCHOR NOT FOUND:\n' + old[:400])
        sys.exit(1)
    html = html.replace(old, new, count)
    edits += 1


def sub_all(old, new):
    global html, edits
    if old not in html:
        print('ANCHOR NOT FOUND (all):\n' + old[:400])
        sys.exit(1)
    html = html.replace(old, new)
    edits += 1


# ── 1. identity ───────────────────────────────────────────────────────────────
sub('<title>FM-PCC — Visual Aligning Explorer v2</title>',
    '<title>FM-PCC — UAV Mix-ML Explorer</title>')
sub('<h1>FM-PCC VA EXPLORER v2</h1>',
    '<h1>FM-PCC UAV EXPLORER</h1>')
sub('<div>DA_VA_v2 &nbsp;|&nbsp; VISUAL ALIGNING SUITE U3</div>',
    '<div>DA_UAV_v1 &nbsp;|&nbsp; GEN15 UAV MIX-ML SUITE</div>')
sub('placeholder="../analysis_results/batch_va2_.../  (folder, not a file)"',
    'placeholder="../analysis_results/batch_uav_.../  (folder, not a file)"')

# ── 2. extra CSS for the axis panel ──────────────────────────────────────────
sub("""        .cov-tbl td.rowhead { text-align: left; }
    </style>""",
    """        .cov-tbl td.rowhead { text-align: left; }
    /* ---- DA_UAV_v1 additions ---- */
        .axis-row { display: flex; align-items: flex-start; gap: 6px; margin-bottom: 5px; }
        .axis-label { font-size: 10px; font-weight: bold; min-width: 52px; padding-top: 2px; }
        .axis-list { display: flex; flex-wrap: wrap; gap: 4px 8px; flex: 1; }
        .axis-list .checkbox-item { margin-bottom: 0 !important; }
        .axis-hint { font-size: 9px; color: #555; line-height: 1.45; margin-top: 4px; }
    </style>""")

# ── 3. mask wording: the UAV hazard is the projector, not a frozen box ───────
sub("""        .mask-banner.unfrozen { background: #e3f2fd; border-color: #1565c0; }""",
    """        .mask-banner.proj-valid { background: #e3f2fd; border-color: #1565c0; }""")
# The frozen-row tint becomes the circuit-breaker tint. Renamed rather than left
# beside the new rule: a dead `.frozen-row` selector is an invitation to reuse a
# class no code sets any more.
sub("""        tr.frozen-row td { background: #fff3e0; }""",
    """        tr.cb-row td { background: #fff3e0; }""")
sub("""                <option value="unfrozen">unfrozen — drop D1 box-obstacle conflicts</option>""",
    """                <option value="proj_valid">proj_valid — drop circuit-breaker rollouts</option>""")

# ── 4. the UAV axis panel, right under the mask/split globals ────────────────
sub("""        <div class="control-group">
            <label>1.6 Split (global)</label>
            <select id="split-select" onchange="refresh_global()"></select>
        </div>""",
    """        <div class="control-group">
            <label>1.6 Split (global)</label>
            <select id="split-select" onchange="refresh_global()"></select>
        </div>
        <div class="control-group">
            <label>1.7 UAV Axes (global)</label>
            <div class="axis-row"><span class="axis-label">Scene</span>
                 <div id="scene-list" class="axis-list"></div></div>
            <div class="axis-row"><span class="axis-label">Engine</span>
                 <div id="engine-list" class="axis-list"></div></div>
            <div class="axis-row"><span class="axis-label">K (NFE)</span>
                 <div id="kaxis-list" class="axis-list"></div></div>
            <div class="axis-hint">Scene, engine and K come from the run's own path
                (<code>uav-&lt;scene&gt;</code> and the eval tag
                <code>E{engine}_K{k}_mpc{b}_{ctrl}_T{thresh}</code>). Unticking narrows
                <b>every</b> view at once. <b>Matched budget or nothing:</b> two arms at
                different K are two experiments, not two results &mdash; leave one K ticked
                when comparing engines.</div>
        </div>""")

# ── 5. which CSVs this page reads ────────────────────────────────────────────
sub_all('VA2_FILES', 'UAV_FILES')
sub("""UAV_FILES = {'agg': 'va2_aggregated_long.csv', 'units': 'va2_units_long.csv',
             'roll': 'per_rollout_detail.csv', 'qual': 'data_quality.csv'}
df_agg_src = None      # va2_aggregated_long.csv   (pooled over seeds)
df_units_src = None    # va2_units_long.csv        (per seed)""",
    """UAV_FILES = {'agg': 'uav_aggregated_long.csv', 'units': 'uav_units_long.csv',
             'roll': 'per_rollout_detail.csv', 'qual': 'data_quality.csv'}
df_agg_src = None      # uav_aggregated_long.csv   (pooled over seeds)
df_units_src = None    # uav_units_long.csv        (per seed)""")

sub("""            # DA_VA_v2 runs only — a batch_va2_* folder is the one carrying the
            # native va2_*.csv cube this page reads.
            batches = re.findall(r'href="(batch_va2_[^/"]+)/?"', html)""",
    """            # DA_UAV_v1 runs only — a batch_uav_* folder is the one carrying the
            # native uav_*.csv cube this page reads.
            batches = re.findall(r'href="(batch_uav_[^/"]+)/?"', html)""")
sub("""                batches = [b for b in data.get("batches", []) if str(b).startswith("batch_va2_")]""",
    """                batches = [b for b in data.get("batches", []) if str(b).startswith("batch_uav_")]""")
sub("""            'This page reads the DA_VA_v2 native CSVs — point it at a '
            '<code>batch_va2_*</code> folder.</div>')""",
    """            'This page reads the DA_UAV_v1 native CSVs — point it at a '
            '<code>batch_uav_*</code> folder.</div>')""")

# ── 6. per-rollout defaults: UAV metric names ────────────────────────────────
sub("""ROLL_DEFAULT_COLUMNS = ['rollout_idx', 'frozen', 'n_success', 'success_relaxed',
                        'mean_dist_per_rollout', 'context_final_xy_dist',
                        'constraint_exec_sat_rate', 'n_violations', 'max_viol_depth_m',
                        'n_steps', 'avg_time_ms', 'max_phys_error_per_rollout']
ROLL_SORT_OPTIONS = [('rollout_idx', 'Rollout index'), ('n_success', 'Success'),
                     ('mean_dist_per_rollout', 'Final distance'),
                     ('context_final_xy_dist', 'Final XY distance'),
                     ('constraint_exec_sat_rate', 'Constraint sat rate'),
                     ('n_violations', 'Violated steps'), ('avg_time_ms', 'Replan time')]""",
    """ROLL_DEFAULT_COLUMNS = ['rollout_idx', 'projection_cb_tripped', 'homotopy_flown',
                        'n_success', 'success_relaxed', 'n_success_and_constraints',
                        'phys_safe', 'goal_reached', 'goal_dist',
                        'collision_free_completed', 'n_violations',
                        'n_steps', 'steps_to_goal',
                        'avg_time_ms', 'fm_ms', 'proj_ms', 'over_budget_frac',
                        'track_err_mean', 'phys_min_z']
ROLL_SORT_OPTIONS = [('rollout_idx', 'Rollout index'), ('n_success', 'Success'),
                     ('goal_dist', 'Final goal distance'),
                     ('n_steps', 'Steps (episode length)'),
                     ('steps_to_goal', 'Steps to goal'),
                     ('n_violations', 'Violated steps'),
                     ('avg_time_ms', 'Total ms / replan'),
                     ('proj_ms', 'Projection ms / replan'),
                     ('track_err_mean', 'Tracking error')]""")

# ── 7. the mask itself: flag column + axis filters ───────────────────────────
sub("""def _norm(df):
    if df is None:
        return None
    for col in ('Candidate', 'variant', 'geo', 'split', 'mask', 'metric', 'FolderName',
                'LatestSnapshot'):
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df""",
    """MASK_FLAG = 'projection_cb_tripped'      # the UAV analogue of VA v2's `frozen`
AXIS_FILTERS = (('scene', 'scene-check'), ('engine', 'eng-check'), ('K', 'kax-check'))


def _norm(df):
    if df is None:
        return None
    for col in ('Candidate', 'variant', 'geo', 'split', 'mask', 'metric', 'FolderName',
                'LatestSnapshot', 'scene', 'engine', 'controller', 'generation'):
        if col in df.columns:
            df[col] = df[col].astype(str)
    # K is written as an integer but read back as float64 whenever ANY candidate
    # has an unparsable eval tag (its K is blank -> NaN -> the column is float).
    # "4.0" then never matches the "4" on its checkbox, and the filter silently
    # empties the page. Normalise to the integer spelling once, here.
    if 'K' in df.columns:
        df['K'] = df['K'].map(_k_str)
    return df


def _k_str(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return '' if value is None else str(value)
    if number != number:                       # NaN
        return ''
    return str(int(number)) if number.is_integer() else str(number)""")

sub("""def _slice(df, use_mask=True):
    \"\"\"Global mask + split, on either CSV shape.

    The long CSVs carry BOTH reductions of every metric in a `mask` COLUMN — pick
    one or every row counts twice. per_rollout_detail has no such column, just a
    `frozen` flag per rollout, so the same switch has to drop rows there instead;
    without this the mask silently does nothing in the rollout/compare views.
    \"\"\"
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
    return out""",
    """def _slice(df, use_mask=True):
    \"\"\"Global mask + split + UAV axes, on any of the four CSV shapes.

    The long CSVs carry BOTH reductions of every metric in a `mask` COLUMN — pick
    one or every row counts twice. per_rollout_detail has no such column, just a
    `projection_cb_tripped` flag per rollout, so the same switch has to drop rows
    there instead; without this the mask silently does nothing in the
    rollout/compare views.

    The scene / engine / K filters are applied HERE rather than in each view for
    the same reason mask and split are: a Gen15 batch is a cross product over
    those axes, and a filter that reached only the plot would leave the result
    matrices, the audit map and the per-rollout table describing a different
    subset than the chart above them.
    \"\"\"
    if df is None or len(df) == 0:
        return df
    out = df
    mask, split = current_mask(), current_split()
    if use_mask:
        if 'mask' in out.columns:
            out = out[out['mask'] == mask]
        elif mask == 'proj_valid' and MASK_FLAG in out.columns:
            out = out[out[MASK_FLAG].fillna(0) != 1]
    if split != 'ALL' and 'split' in out.columns:
        out = out[out['split'] == split]
    return _axis_slice(out)


def _axis_slice(out):
    \"\"\"Keep only the ticked scenes / engines / K values.

    Nothing ticked in a group means "do not filter on it", not "show nothing" —
    a batch whose candidates have no parsable eval tag has an empty K list, and
    blanking the whole page over a missing axis would hide the very data the user
    came to look at.
    \"\"\"
    for column, css_class in AXIS_FILTERS:
        if column not in out.columns:
            continue
        picked = set(get_checked(css_class))
        if not picked:
            continue
        out = out[out[column].astype(str).isin(picked)]
    return out""")

# ── 8. the derived relaxed pair ──────────────────────────────────────────────
# DA_UAV_v1 writes `n_success_relaxed_and_constraints` natively (the UAV eval
# records all four cells of the {strict, relaxed} x {with, without constraints}
# matrix per rollout), so the synthesis below is normally skipped. The inputs are
# repointed at the UAV column names anyway, for a batch produced before that.
sub("""DERIVED_METRIC = 'n_success_relaxed_and_constraints'
DERIVED_INPUTS = ('success_relaxed', 'constraint_exec_zero_violation')""",
    """DERIVED_METRIC = 'n_success_relaxed_and_constraints'
DERIVED_INPUTS = ('success_relaxed', 'collision_free_completed')""")

# ── 9. mask banner ───────────────────────────────────────────────────────────
sub("""def render_mask_banner():
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
    banner.style.display = "block\"""",
    """def render_mask_banner():
    total = tripped = None
    if df_qual_src is not None and len(df_qual_src):
        if 'n_rollouts' in df_qual_src.columns:
            total = int(pd.to_numeric(df_qual_src['n_rollouts'], errors='coerce').fillna(0).sum())
        if 'n_cb_tripped' in df_qual_src.columns:
            tripped = int(pd.to_numeric(df_qual_src['n_cb_tripped'], errors='coerce').fillna(0).sum())
    elif df_roll_src is not None and len(df_roll_src):
        total = int(len(df_roll_src))
        tripped = int(pd.to_numeric(df_roll_src.get(MASK_FLAG, 0), errors='coerce').fillna(0).sum())

    mask, split = current_mask(), current_split()
    if total is None or tripped is None:
        text = "circuit-breaker counts unavailable (data_quality.csv not in this batch)"
    elif mask == 'proj_valid':
        text = (f"mask = PROJ_VALID — {tripped} of {total} rollouts excluded (the projection "
                f"circuit breaker opened: sustained SLSQP slowness made the eval SKIP "
                f"projection, so those steps flew the UNPROJECTED plan and their constraint "
                f"numbers are not this variant's)")
    else:
        text = (f"mask = ALL — every rollout kept, of which {tripped} of {total} ran "
                f"(partly) UNPROJECTED after the circuit breaker opened. "
                f"Switch to PROJ_VALID to exclude them.")
    axes = _axis_summary()
    banner = document.getElementById("mask-banner")
    banner.className = "mask-banner proj-valid" if mask == "proj_valid" else "mask-banner"
    banner.innerHTML = (f"<b>{text}</b><br><span class='muted'>split = {split} &middot; "
                        f"{axes} &middot; environment axis = geo_tag &middot; "
                        f"SEM in the matrices is over rollouts</span>")
    banner.style.display = "block"


def _axis_summary():
    \"\"\"'scene: corridor · engine: mf, af · K: 4' — what the page is currently showing.

    Worth a line of its own next to the mask: with three global axis filters, a
    plot of two bars can mean "these two arms" or "everything, filtered down to
    two", and only this text distinguishes them.\"\"\"
    bits = []
    for column, css_class in AXIS_FILTERS:
        picked = get_checked(css_class)
        source = df_agg_src if df_agg_src is not None else df_roll_src
        available = []
        if source is not None and column in source.columns:
            available = sorted({str(v) for v in source[column].unique() if str(v)})
        if not available:
            continue
        label = 'K' if column == 'K' else column
        if len(picked) == len(available):
            bits.append(f"{label}: all ({len(available)})")
        else:
            bits.append(f"{label}: {', '.join(picked) if picked else 'all'}")
    return ' &middot; '.join(bits) if bits else 'no UAV axes in this batch'""")

# ── 10. populate the axis panel ──────────────────────────────────────────────
sub("""def populate_va2_filters(vars_list, cands_list):""",
    """def populate_axis_filters():
    \"\"\"Fill the scene / engine / K checkbox groups from the batch itself.

    Every box starts TICKED: the batch as it sits on disk is the honest default,
    and an unticked axis would make the first view a silent subset. A group with
    no values (an axis this batch does not carry) is filled with a short note
    instead of being left as an empty box the user cannot act on.
    \"\"\"
    source = df_agg_src if df_agg_src is not None else df_roll_src
    for column, css_class in AXIS_FILTERS:
        element = document.getElementById(
            {'scene': 'scene-list', 'engine': 'engine-list', 'K': 'kaxis-list'}[column])
        values = []
        if source is not None and column in source.columns:
            values = sorted({str(v) for v in source[column].unique() if str(v)},
                            key=lambda v: (0, int(v)) if v.isdigit() else (1, v))
        if not values:
            element.innerHTML = '<span class="muted" style="font-size:10px">n/a</span>'
            continue
        element.innerHTML = "".join(
            f'<div class="checkbox-item"><input type="checkbox" class="{css_class}" '
            f'value="{v}" checked onchange="refresh_global()"><span>{v}</span></div>'
            for v in values)


def populate_va2_filters(vars_list, cands_list):""")

sub("""        populate_split_filter()
        derive_frames()""",
    """        populate_split_filter()
        populate_axis_filters()      # must precede derive_frames — _slice reads the boxes
        derive_frames()""")

# ── 11. compare-view default axes ────────────────────────────────────────────
sub("""        skip = {'Candidate', 'FolderName', 'FullPath', 'seed', 'split', 'geo',
                'geo_base', 'tightened', 'variant', 'variant_raw'}""",
    """        skip = {'Candidate', 'FolderName', 'FullPath', 'seed', 'split', 'geo',
                'geo_scene', 'tightened', 'variant', 'variant_base', 'variant_raw',
                'scene', 'engine', 'K', 'mpc_batch', 'controller', 'threshold',
                'backbone', 'generation', 'homotopy', 'homotopy_flown', 'scene_json'}""")
sub("""        document.getElementById("cmp-y-select").innerHTML = _opts('mean_dist_per_rollout')
        document.getElementById("cmp-x-select").innerHTML = _opts('context_init_xy_dist')""",
    """        # The Gen15 question in one scatter: what did a plan cost, and did it work.
        document.getElementById("cmp-y-select").innerHTML = _opts('goal_dist')
        document.getElementById("cmp-x-select").innerHTML = _opts('avg_time_ms')""")

# ── 12. per-rollout table: flag column, tint class, wording ──────────────────
sub("""    ascending = sort_by not in ('n_success', 'success_relaxed', 'constraint_exec_sat_rate')""",
    """    ascending = sort_by not in ('n_success', 'success_relaxed', 'phys_safe',
                                'goal_reached', 'collision_free_completed')""")
sub("""    n_frozen = int(pd.to_numeric(subset.get('frozen', 0), errors='coerce').fillna(0).sum())
    succ = pd.to_numeric(subset.get('n_success'), errors='coerce').mean() if 'n_success' in subset.columns else float('nan')
    summary.innerHTML = (
        f'<p><b>CAND_{cand} &middot; {variant} &middot; geo {geo}</b> &mdash; {len(subset)} rollouts, '
        f'{n_frozen} frozen, success {"n/a" if succ != succ else f"{succ * 100:.1f}%"} '
        f'&middot; sorted by <code>{sort_by}</code> &middot; frozen rows are tinted</p>')""",
    """    n_tripped = int(pd.to_numeric(subset.get(MASK_FLAG, 0), errors='coerce').fillna(0).sum())
    succ = pd.to_numeric(subset.get('n_success'), errors='coerce').mean() if 'n_success' in subset.columns else float('nan')
    summary.innerHTML = (
        f'<p><b>CAND_{cand} &middot; {variant} &middot; geo {geo}</b> &mdash; {len(subset)} rollouts, '
        f'{n_tripped} circuit-breaker, success {"n/a" if succ != succ else f"{succ * 100:.1f}%"} '
        f'&middot; sorted by <code>{sort_by}</code> &middot; circuit-breaker rows are tinted '
        f'(they flew an unprojected plan)</p>')""")
sub("""        klass = ' class="frozen-row"' if float(row.get('frozen', 0) or 0) == 1 else ''""",
    """        klass = ' class="cb-row"' if float(row.get(MASK_FLAG, 0) or 0) == 1 else ''""")
sub("""    current_roll_table = subset[['Candidate', 'seed', 'split', 'geo', 'variant']
                                + [c for c in columns if c not in ('seed', 'split')]]""",
    """    lead = [c for c in ('Candidate', 'scene', 'engine', 'K', 'seed', 'split',
                        'geo', 'variant') if c in subset.columns]
    current_roll_table = subset[lead + [c for c in columns if c not in lead]]""")

# ── 13. data-quality view: UAV columns ───────────────────────────────────────
sub("""    for col in ('n_frozen', 'n_cb_tripped'):""",
    """    for col in ('n_cb_tripped', 'cb_sentinel', 'timing_missing'):""")
sub("""        container.innerHTML = (f'<p style="color:#2e7d32;font-weight:bold">All {len(data)} units '
                               f'clean — no frozen rollouts, no circuit-breaker trips, no partial npz.</p>')
        return
    columns = [c for c in ('Candidate', 'seed', 'split', 'geo', 'variant', 'source', 'n_rollouts',
                           'n_frozen', 'frozen_rate', 'n_cb_tripped', 'cb_skipped_steps',
                           'npz_complete') if c in data.columns]""",
    """        container.innerHTML = (f'<p style="color:#2e7d32;font-weight:bold">All {len(data)} units '
                               f'clean — no circuit-breaker trips, no missing timing, no partial npz.</p>')
        return
    columns = [c for c in ('Candidate', 'scene', 'engine', 'K', 'seed', 'geo', 'variant',
                           'source', 'has_projector', 'n_rollouts',
                           'n_cb_tripped', 'cb_tripped_rate', 'cb_skipped_steps',
                           'cb_trips', 'backstop_hits', 'cb_sentinel', 'timing_missing',
                           'npz_complete') if c in data.columns]""")
sub("""        f'<p><b>{len(flagged)} of {len(data)} units flagged.</b> <span class="muted">Frozen '
        f'rollouts never called the model; circuit-breaker rollouts ran (partly) unprojected, so '
        f'their constraint metrics do not describe the policy the variant name claims; '
        f'npz_complete=0 is a crash-safety partial.</span></p>'""",
    """        f'<p><b>{len(flagged)} of {len(data)} units flagged.</b> <span class="muted">'
        f'Circuit-breaker rollouts ran (partly) unprojected, so their constraint metrics do '
        f'not describe the policy the variant name claims &mdash; the eval also drops a '
        f'<code>PROJECTION_CB_TRIPPED.txt</code> beside them (<code>cb_sentinel</code>). '
        f'<code>timing_missing=1</code> means no per-rollout timing was recoverable: the UAV '
        f'npz carries NO timing group, so avg_time / fm_ms / proj_ms come from '
        f'<code>diagnostics/*.json</code> alone and are NaN without it. '
        f'<code>npz_complete=0</code> is a crash-safety partial.</span></p>'""")

# ── 14. result matrices: UAV metrics ─────────────────────────────────────────
sub("""SUMMARY_TABLES = [
    ("n_success", "N_SUCCESS -- success rate (goal reached: strict, position AND rotation)", "{:.3f}", False),
    ("n_success_and_constraints", "N_SUCCESS + CONSTRAINT -- success rate (goal reached AND constraints satisfied)", "{:.3f}", False),
    # U3 (visual-aligning only; the state-only avoiding trees leave these columns
    # absent, so their cells render as the honest NULL dash rather than a zero).
    ("success_relaxed", "N_SUCCESS RELAXED -- success rate (position only: final box-target XY distance within pos_min_dist; final angle ignored)", "{:.3f}", False),
    ("n_success_relaxed_and_constraints", "N_SUCCESS RELAXED + CONSTRAINT -- relaxed success AND constraints satisfied", "{:.3f}", False),
    ("mean_dist_per_rollout", "MIN_DIST -- final box-target distance [m], the env's own score (half the position error plus half the rotation error) measured against pos/rot_min_dist -- lower is better", "{:.4f}", True),
    ("n_steps", "N_STEPS -- steps taken (episode length)", "{:.1f}", True),
    ("avg_time", "AVG_TIME -- average computation time per planning step [s]", "{:.4f}", True),
]""",
    """SUMMARY_TABLES = [
    ("n_success", "N_SUCCESS -- strict success rate (goal reached AND physically safe: contact-free and airborne). On scene=empty the start/goal is random and never given to the state-only policy, so success there is stable flight only", "{:.3f}", False),
    ("n_success_and_constraints", "N_SUCCESS + CONSTRAINT -- strict success AND zero declared-constraint violations", "{:.3f}", False),
    ("success_relaxed", "N_SUCCESS RELAXED -- relaxed success rate (crossed the finish line; the goal-radius test is dropped)", "{:.3f}", False),
    ("n_success_relaxed_and_constraints", "N_SUCCESS RELAXED + CONSTRAINT -- relaxed success AND constraints satisfied", "{:.3f}", False),
    ("phys_safe", "PHYS_SAFE -- physical ground truth (hard MuJoCo contact detection): contact-free for the whole episode and never grounded", "{:.3f}", False),
    ("collision_free_completed", "COLLISION_FREE -- zero-violation rollout rate against the projector's own declared margins (softer than PHYS_SAFE)", "{:.3f}", False),
    ("goal_dist", "GOAL_DIST -- final distance to the goal [m] -- lower is better", "{:.4f}", True),
    ("n_steps", "N_STEPS -- episode length. Early-stops on goal-reach and runs the FULL budget on a miss (U_13), so this measures misses as much as speed -- read it against the STEP BUDGET row below", "{:.1f}", True),
    ("steps_to_goal", "STEPS_TO_GOAL -- episode length over REACHING episodes only: the honest time-to-goal", "{:.1f}", True),
    ("avg_time_ms", "AVG_TIME -- total wall clock per replan [ms]. The real-time budget is 1/control_hz ~ 30 ms; exceeding it is the failure this generation is about", "{:.1f}", True),
    ("fm_ms", "GEN_MS -- generation time per replan [ms], projection subtracted", "{:.1f}", True),
    ("proj_ms", "PROJ_MS -- projection time per replan [ms]. Cheaper generation is supposed to RELEASE budget to this", "{:.1f}", True),
    ("over_budget_frac", "OVER_BUDGET -- fraction of control steps that missed the real-time deadline -- lower is better", "{:.3f}", True),
    ("track_err_mean", "TRACK_ERR -- mean tracking error of the low-level controller against the commanded p_des", "{:.3f}", True),
]""")

sub("""REF_METRIC = 'context_init_xy_dist'
REF_FOR = 'mean_dist_per_rollout'
REF_LABEL = 'INIT XY (ref)'""",
    """# N_STEPS gets one extra row: the episode STEP BUDGET. 396 steps means nothing on
# its own; against a budget of 396 it means every episode ran to the wall without
# reaching the goal. Same role the INIT XY row plays on the visual-aligning page.
REF_METRIC = 'max_episode_length'
REF_FOR = 'n_steps'
REF_LABEL = 'STEP BUDGET (ref)'""")

sub("""FLAG_SKIP = set(FLAG_INPUTS) | {'success_relaxed', 'n_success_relaxed_and_constraints'}""",
    """FLAG_SKIP = (set(FLAG_INPUTS)
             | {'success_relaxed', 'n_success_relaxed_and_constraints',
                'phys_safe', 'collision_free_completed'})""")

# ── 15. variant presets: UAV puts tightening back on the variant axis ────────
sub("""VARIANT_PRESETS = [
    ('dpcc_hf', 'DPCC + HF',
     'diffuser + every dpcc-{r,c,t} and hardflow_new arm, tightened or not. Excludes the '
     'dt-scaled dpcc sweeps and the non-projection baselines (gradient, post_processing, '
     'model_free, geo_free, bounds_free).'),
    ('dpcc_hf_tight', 'DPCC + HF (tightened)',
     'diffuser + only the -tightened dpcc-{r,c,t} and hardflow_new-{r,c,t} arms.'),
    ('dpcc_tight', 'DPCC (tightened)',
     'diffuser + only the -tightened dpcc-{r,c,t} arms. No HardFlow.'),
]""",
    """VARIANT_PRESETS = [
    ('dpcc_hf', 'DPCC + HF',
     'diffuser + every dpcc-{r,c,t} and hardflow_new arm, tightened or not. Excludes the '
     'constraint-family ablations (gradient, post_processing, model_free, geo_free, '
     'bounds_free and their combinations).'),
    ('dpcc_hf_tight', 'DPCC + HF (tightened)',
     'diffuser + only the -tightened dpcc-{r,c,t} and hardflow_new arms.'),
    ('dpcc_tight', 'DPCC (tightened)',
     'diffuser + only the -tightened dpcc-{r,c,t} arms. No HardFlow.'),
    ('ablations', 'Constraint ablations',
     'diffuser + the constraint-family toggles from config/uav_projection.yaml: '
     'model_free (dynamics off), bounds_free (action bounds off), geo_free '
     '(geo_bounds+halfspace+obstacles off) and their combinations. This is the '
     '"which constraint family is actually doing the work" comparison, not the '
     'projection-arm one.'),
]

# The ablation toggles, composable in any combination and optionally tightened.
PRESET_ABLATION_RE = re.compile(
    r'^(?:model_free|bounds_free|geo_free|gradient|post_processing)'
    r'(?:-(?:model_free|bounds_free|geo_free))*(?:-tightened)?$')""")

sub("""        m = PRESET_ARM_RE.match(name)
        if m is None or m.group('extra'):
            continue
        tight = bool(m.group('tight'))
        if key == 'dpcc_hf':
            keep = True
        elif key == 'dpcc_hf_tight':
            keep = tight
        else:                                   # dpcc_tight
            keep = tight and m.group('fam') == 'dpcc'
        if keep:
            out.append(name)""",
    """        if key == 'ablations':
            if PRESET_ABLATION_RE.match(name):
                out.append(name)
            continue
        m = PRESET_ARM_RE.match(name)
        if m is None or m.group('extra'):
            continue
        tight = bool(m.group('tight'))
        if key == 'dpcc_hf':
            keep = True
        elif key == 'dpcc_hf_tight':
            keep = tight
        else:                                   # dpcc_tight
            keep = tight and m.group('fam') == 'dpcc'
        if keep:
            out.append(name)""")

sub("""        if not members:
            # No dead checkbox: a batch without tightened variant names (DA_VA_v2 puts
            # tightening on the GEOMETRY axis instead) simply does not offer that preset.
            empty.append(label)
            continue""",
    """        if not members:
            # No dead checkbox: a batch that ran only a subset of the yaml's
            # projection_variants simply does not offer the presets it cannot fill.
            empty.append(label)
            continue""")

sub("""    if empty:
        hint += ('<br>Not offered by this batch: <b>' + ", ".join(empty) + '</b> '
                 '&mdash; no <code>-tightened</code> variant names here. On a DA_VA_v2 batch '
                 'tightening lives on the geometry axis instead: use <b>4. Geometry Focus</b>.')""",
    """    if empty:
        hint += ('<br>Not offered by this batch: <b>' + ", ".join(empty) + '</b> '
                 '&mdash; none of their variants were run here. UAV enumerates every '
                 '<code>-tightened</code> arm explicitly in <code>config/uav_projection.yaml</code>, '
                 'so a missing preset means a missing RUN, not a different axis.')""")

# ── 16. wording of the geometry axis + the no-data tip ───────────────────────
sub("""<label>4. Geometry Focus</label>""",
    """<label>4. Geo Tag Focus</label>""")
sub("""           f'Tip: hardflow-only metrics (activation_threshold, nlp_solves, nfe, batch_size, '
           f'flow_steps) exist only for <b>hardflow_new*</b> variants. Pick a core metric '
           f'(n_success_and_constraints, avg_time) to compare across all variants.</div>')""",
    """           f'Tips: (1) hardflow-only metrics (nfe_per_plan, nlp_solves_total, '
           f'activation_threshold) exist only for <b>hardflow_new*</b> variants. (2) timing '
           f'metrics (avg_time_ms, fm_ms, proj_ms) need <code>diagnostics/*.json</code> &mdash; '
           f'the UAV npz has no timing group, so a batch analysed with '
           f'<code>--no-diagnostics-scan</code> has none. (3) check <b>1.7 UAV Axes</b>: a '
           f'scene/engine/K combination that was never run is empty by construction. '
           f'Pick a core metric (n_success_and_constraints, n_steps) to compare everything.</div>')""")

# ── 17. the metric-dropdown default ──────────────────────────────────────────
sub("""    default_metric = next((m for m in ["n_success_and_constraints", "n_success", "avg_time"] if m in metrics), metrics[0] if metrics else None)""",
    """    default_metric = next((m for m in ["n_success_and_constraints", "n_success", "avg_time_ms"] if m in metrics), metrics[0] if metrics else None)""")

# ── 18. inherited comments that name the OTHER pipeline's file ───────────────
# Comment-only, but the audit map's seed column is exactly the place someone
# checks when a candidate looks half-run — being sent to a file this pipeline
# never writes wastes the search.
sub("""    # U17: which seeds, from va2_units_long.csv. has_missing is always False on this page,""",
    """    # U17: which seeds, from uav_units_long.csv. has_missing is always False on this page,""")

DST.write_text(html)
print(f'{DST} written from {SRC}  ({edits} edits, {len(html)} bytes)')
