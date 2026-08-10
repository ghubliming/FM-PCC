"""Drive Visualizer_VA_v2's page code against real CSVs with a stubbed browser.

    python Data_Analysis/Visualizer_VA_v2/test_page_offline.py [batch_va2_dir]

Needs pandas + matplotlib, so it runs on the cluster / any machine with the
science stack — not in the AI-coding container.

Not a rendering test — a wiring + data test. The page's whole Python block is
exec'd with fake document/window/pyscript objects, the source frames are set from
real DA_VA_v2 CSVs, and the actual handlers (trigger_plot, show_rollout_table,
trigger_compare, render_quality, refresh_global, download_plot) are called the way
the DOM would call them. Anything that raises, or renders an empty panel, fails.
"""
import asyncio
import pathlib
import re
import sys

import pandas as pd
import matplotlib
matplotlib.use('agg')

HERE = pathlib.Path(__file__).resolve().parent
HTML = HERE / 'index.html'

if len(sys.argv) > 1:
    BATCH = pathlib.Path(sys.argv[1])
else:
    runs = sorted((HERE.parent / 'analysis_results').glob('batch_va2_*'), reverse=True)
    if not runs:
        print('usage: python test_page_offline.py <path/to/batch_va2_.../>')
        sys.exit(2)
    BATCH = runs[0]

def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


failures = []


def check(name, ok, detail=''):
    print(f'  {"PASS" if ok else "FAIL"}  {name}' + (f'  — {detail}' if detail else ''))
    if not ok:
        failures.append(name)


# ── browser stubs ────────────────────────────────────────────────────────────
class Style:
    def __init__(self):
        self.display = 'none'
        self.color = ''
        self.width = ''
        self.height = ''
        self.maxWidth = ''


class ClassList:
    def toggle(self, *a):
        pass


class Element:
    def __init__(self, eid, value=''):
        self.id = eid
        self.value = value
        self.innerHTML = ''
        self.innerText = ''
        self.checked = False
        self.disabled = False
        self.style = Style()
        self.classList = ClassList()
        self.className = ''
        self.href = ''
        self.download = ''
        self.dataset = type('D', (), {})()

    def click(self):
        pass


class Document:
    def __init__(self):
        self.elements = {}
        self.classes = {}

    def getElementById(self, eid):
        return self.elements.setdefault(eid, Element(eid))

    def getElementsByClassName(self, cls):
        return self.classes.get(cls, [])

    def createElement(self, _tag):
        return Element('tmp')

    def querySelectorAll(self, _sel):
        return []

    def set_checks(self, cls, values, checked=True):
        items = []
        for v in values:
            e = Element(f'{cls}:{v}', str(v))
            e.checked = checked
            items.append(e)
        self.classes[cls] = items


class Window:
    currentMode = 'list'
    currentView = 'aggregate'
    cmpChartType = 'scatter'

    def applyZoom(self):
        pass

    def applyViewMode(self):
        pass

    def alert(self, *a):
        pass

    def confirm(self, *a):
        return True


document = Document()
window = Window()
console = type('C', (), {'log': staticmethod(lambda *a: None)})()
displayed = []


def display(fig, target=None):
    displayed.append((fig, target))


# ── load the page's python block ─────────────────────────────────────────────
source = HTML.read_text()
block = source.split('<script type="py">')[1].split('</script>')[0]
block = "\n".join(line for line in block.splitlines()
                  if not line.startswith(('from js import', 'from pyscript import',
                                          'from pyodide.ffi import'))
                  and not line.startswith('asyncio.ensure_future'))
# create_proxy is a no-op outside pyodide; document.X = f then just stores the function.
block = block.replace('create_proxy(', 'identity(')

ns = {'document': document, 'window': window, 'console': console, 'display': display,
      'identity': lambda f: f, 'fetch': None, 'Uint8Array': None, 'File': None, 'URL': None}
exec(compile(block, 'va2_page', 'exec'), ns)
print(f'[harness] page python block: {len(block.splitlines())} lines exec\'d clean')

# ── feed it real data, the way load_data would ───────────────────────────────
def read(name):
    path = BATCH / name
    return ns['_norm'](pd.read_csv(path, low_memory=False)) if path.exists() else None


print(f'\n[harness] batch: {BATCH.name}')
ns['df_agg_src'] = read('va2_aggregated_long.csv')
ns['df_units_src'] = read('va2_units_long.csv')
ns['df_roll_src'] = read('per_rollout_detail.csv')
ns['df_qual_src'] = read('data_quality.csv')
check('all four native CSVs read',
      all(ns[k] is not None for k in ('df_agg_src', 'df_units_src', 'df_roll_src', 'df_qual_src')))

document.getElementById('mask-select').value = 'all'
document.getElementById('seed-mode-select').value = 'standard'
document.getElementById('mode-select').value = 'candidate'
document.getElementById('fig-width').value = '11.0'
document.getElementById('width-zoom').value = '1.0'
document.getElementById('batch-select').value = BATCH.name

print('\n[populate + derive]')
ns['populate_split_filter']()
splits = re.findall(r'value="([^"]+)"', document.getElementById('split-select').innerHTML)
check('split filter populated', 'ALL' in splits, f'options={splits}')
document.getElementById('split-select').value = 'ALL'

ns['derive_frames']()
df_agg = ns['df_agg']
df_raw = ns['df_raw']
check('df_agg in DAv3 schema',
      all(c in df_agg.columns for c in ('Candidate', 'Folder_Name', 'Full_Path', 'variant',
                                        'halfspace_variant', 'metric', 'mean', 'std', 'count')),
      f'{len(df_agg)} rows')
check('df_raw carries seeds', df_raw is not None and 'seed' in df_raw.columns
      and 'value' in df_raw.columns, f'{0 if df_raw is None else len(df_raw)} rows')
# The source CSV holds both masks, so exactly half of it must survive. U3's derived
# relaxed-success+constraint rows are appended on top of that half (one per unit),
# and are counted separately so a double-count would still be caught.
n_derived = int((df_agg['metric'] == ns['DERIVED_METRIC']).sum())
check('mask applied (not double-counted)',
      len(df_agg) - n_derived == len(ns['df_agg_src']) // 2,
      f'{len(df_agg)} of {len(ns["df_agg_src"])} (+{n_derived} derived)')
n_native = int((df_agg['metric'] == 'n_success').sum())
_roll = ns['df_roll_src']
# A state-only avoiding batch has no relaxed-success column at all — then the
# right answer is NO derived rows (and a dash in the table), not a column of zeros.
_has_relaxed = _roll is not None and all(c in _roll.columns for c in ns['DERIVED_INPUTS'])
check('U3 relaxed success + constraint derived per unit',
      (0 < n_derived <= n_native) if _has_relaxed else n_derived == 0,
      f'{n_derived} rows vs {n_native} n_success rows'
      + ('' if _has_relaxed else ' (state-only batch: none expected)'))

# and it must be the mean of the per-rollout PRODUCT, not the product of the means
if _has_relaxed:
    _key = df_agg[df_agg['metric'] == ns['DERIVED_METRIC']].iloc[0]
    _sub = _roll[(_roll['Candidate'] == _key['Candidate'])
                 & (_roll['geo'] == _key['halfspace_variant'])
                 & (_roll['variant'] == _key['variant'])]
    _manual = (pd.to_numeric(_sub['success_relaxed'], errors='coerce')
               * pd.to_numeric(_sub['constraint_exec_zero_violation'], errors='coerce')).mean()
    check('U3 derived value matches the raw rollouts',
          abs(float(_key['mean']) - float(_manual)) < 1e-9,
          f'{_key["variant"]}: page={_key["mean"]:.4f} raw={_manual:.4f} over {len(_sub)} rollouts')
else:
    check('U3 derived value matches the raw rollouts', True, 'SKIP — state-only batch')

ns['populate_dynamic_filters']()
metrics = re.findall(r'value="([^"]+)"', document.getElementById('metric-select').innerHTML)
envs = re.findall(r'value="([^"]+)"', document.getElementById('env-select').innerHTML)
variants = re.findall(r'value="([^"]+)"', document.getElementById('variant-list').innerHTML)
cands = re.findall(r'value="([^"]+)"', document.getElementById('candidate-list').innerHTML)
seeds = re.findall(r'value="([^"]+)"', document.getElementById('seed-list').innerHTML)
check('metrics populated', len(metrics) > 10, f'{len(metrics)}')
check('geometry axis populated', len(envs) >= 1, f'{envs}')
check('variants populated', len(variants) > 5, f'{len(variants)}')
check('candidates populated', len(cands) >= 1, f'{cands}')
check('seeds come from the data (not hardcoded 6..10)', len(seeds) >= 1, f'{seeds}')
roll_cands = re.findall(r'value="([^"]+)"', document.getElementById('roll-cand-select').innerHTML)
roll_cols = re.findall(r'value="([^"]+)"', document.getElementById('rollcol-list').innerHTML)
cmp_y = re.findall(r'value="([^"]+)"', document.getElementById('cmp-y-select').innerHTML)
check('rollout selects populated', roll_cands and roll_cols and cmp_y,
      f'{len(roll_cands)} cands / {len(roll_cols)} cols / {len(cmp_y)} metrics')

ns['render_mask_banner']()
banner = document.getElementById('mask-banner').innerHTML
check('mask banner states the frozen count', 'D1-FROZEN' in banner, banner[:80] + '…')

# ── AGGREGATE view (all of DAv3's machinery) ─────────────────────────────────
print('\n[aggregate view]')
document.getElementById('metric-select').value = 'n_success_and_constraints'
document.getElementById('env-select').value = envs[0]
document.set_checks('var-check', variants[:5])
document.set_checks('cand-check', cands)
document.set_checks('seed-check', seeds)

run(ns['trigger_plot'](None))
check('plot drawn', len(displayed) > 0, f'{len(displayed)} figure(s)')
check('scorecard filled', 'BEST' in document.getElementById('scorecard-container').innerHTML.upper()
      or 'Candidate' in document.getElementById('scorecard-container').innerHTML)
check('U9 plot legend rendered',
      'CAND_' in document.getElementById('selection-map-container').innerHTML)
summary_html = document.getElementById('summary-container').innerHTML
check('U10 result matrices rendered', 'paper-tbl' in summary_html and 'CAND_' in summary_html,
      f'{summary_html.count("<table")} tables')
# Only meaningful when this environment actually HAS a hole — a batch whose every
# candidate ran every variant has no never-run cell to mark, and demanding one is a
# test bug, not a page bug.
_d = df_agg[(df_agg['halfspace_variant'] == document.getElementById('env-select').value)
            & (df_agg['metric'] == 'n_success')]
_grid_full = len(set(zip(_d['Candidate'], _d['variant']))) >= len(cands) * _d['variant'].nunique()
check('matrices mark never-run cells', 'nullcell' in summary_html or _grid_full,
      'SKIP — full grid, nothing to mark' if _grid_full else '')

# U3: the run tally. The whole point is that the candidates are NOT comparable by
# volume, so the table must state a number for every selected candidate.
_cov = re.search(r'<table class="paper-tbl cov-tbl">.*?</table>', summary_html, re.S)
# the optional <span class="hl-name"> is U13's highlight wrapper — match around it so a
# highlighted candidate does not silently drop out of this tally.
_cov_rows = re.findall(r'<td class="rowhead">(?:<span[^>]*>)?CAND_([^<]+?)(?:</span>)?</td>',
                       _cov.group(0)) if _cov else []
check('U3 run-coverage table rendered', sorted(_cov_rows) == sorted(cands),
      f'{len(_cov_rows)} candidates tallied')
check('U3 coverage flags an unbalanced batch',
      ('UNBALANCED' in summary_html) == (len(set(
          int(pd.to_numeric(df_agg[(df_agg['metric'] == 'n_success')
                                   & (df_agg['Candidate'] == c)]['count'],
                            errors='coerce').fillna(0).sum()) for c in cands)) > 1),
      'UNBALANCED' if 'UNBALANCED' in summary_html else 'balanced batch, no warning')

# U3: the INIT XY reference row — only where THIS environment has a start distance.
# (A state-only avoiding geometry has none, and inventing one there would be worse
# than leaving the row out.)
_has_init = bool(((df_agg['metric'] == ns['REF_METRIC'])
                  & (df_agg['halfspace_variant'] == document.getElementById('env-select').value)
                  & (df_agg['Candidate'].isin(cands))
                  & df_agg['mean'].notna()).any())
# Match the row's CSS class, not its label — the label also appears in the caption
# text that explains the row, which would pass this check with no row rendered.
check('U3 INIT XY reference row on the MIN_DIST table',
      ('class="refrow"' in summary_html) == _has_init,
      f'{"present" if _has_init else "absent"} — batch {"has" if _has_init else "lacks"} '
      + ns['REF_METRIC'])

ns['render_path_map']()
check('path audit map rendered', 'CAND_' in document.getElementById('path-map-container').innerHTML)

# "Last Run" (config-snapshot timestamp): the column must appear exactly when the
# batch's CSVs carry a usable LatestSnapshot, and must be absent — not blank — for
# batches produced before DA_VA_v2 wrote that column.
_stamp_map = ns['_stamp_map']()
_path_html = document.getElementById('path-map-container').innerHTML
_legend_html = document.getElementById('selection-map-container').innerHTML
check('Last Run column matches the data', ('Last Run' in _path_html) == bool(_stamp_map),
      f'{len(_stamp_map)} candidate(s) stamped' if _stamp_map
      else 'SKIP-ish — pre-timestamp batch, column correctly omitted')
check('Last Run column in the plot legend too',
      ('Last Run' in _legend_html) == bool(_stamp_map))
if _stamp_map:
    _sample = next(iter(_stamp_map.values()))
    check('Last Run is rendered human-readable',
          bool(re.fullmatch(r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}', _sample))
          and _sample in _path_html, _sample)

# ── U16: the "5. Variants" quick presets ─────────────────────────────────────
print('\n[U16 variant presets]')
_presets = {k: ns['_preset_members'](k, variants) for k, _l, _t in ns['VARIANT_PRESETS']}
_panel = document.getElementById('variant-presets').innerHTML
check('the preset panel renders one box per non-empty preset',
      _panel.count('class="preset-check"') == sum(1 for m in _presets.values() if m),
      ' / '.join(f'{k}:{len(v)}' for k, v in _presets.items()))
# every member has to exist as a real checkbox value, or ticking the preset selects nothing
_orphans = sorted({m for members in _presets.values() for m in members} - set(variants))
check('every preset member is a real variant checkbox', not _orphans, _orphans)
# the exclusions carry the meaning: a dt sweep folded in would put four near-identical
# bars beside the arm being read
_leak = sorted({m for members in _presets.values() for m in members
                if '-dt' in m or m.split('-')[0] in ('gradient', 'post_processing',
                                                     'model_free', 'geo_free', 'bounds_free')})
check('no preset leaks a dt sweep or a non-projection baseline', not _leak, _leak)
if _presets['dpcc_hf']:
    # ticking a preset must produce a selection the plot can actually draw
    document.set_checks('var-check', _presets['dpcc_hf'])
    document.set_checks('cand-check', cands)
    _before_figs = len(displayed)
    run(ns['trigger_plot'](None))
    check('a preset selection draws', len(displayed) > _before_figs
          or 'NO DATA' in document.getElementById('plot-area').innerHTML,
          f'{len(_presets["dpcc_hf"])} variants')
    check('the preset panel survives a redraw',
          'class="preset-check"' in document.getElementById('variant-presets').innerHTML)
else:
    print('  SKIP  this batch has no dpcc/hardflow arms at all')

# ── U15: no two variants may draw in the same colour ─────────────────────────
print('\n[U15 variant colours]')
_all_vars = sorted(df_agg['variant'].astype(str).unique())
_pal = ns['_palette'](len(_all_vars))
check('the real colormaps supply one distinct colour per variant',
      len(_pal) == len(_all_vars)
      and len({tuple(round(c[i], 3) for i in range(3)) for c in _pal}) == len(_all_vars),
      f'{len(_all_vars)} variants in this batch')
# the failure the old colormap='tab10' produced: draw every variant and demand that the
# patches matplotlib actually rendered are pairwise distinct.
document.set_checks('var-check', _all_vars)
document.set_checks('cand-check', cands)
document.getElementById('metric-select').value = 'n_success_and_constraints'
run(ns['trigger_plot'](None))
from matplotlib.container import BarContainer                        # noqa: E402
_containers = [c for c in ns['current_ax'].containers if isinstance(c, BarContainer)]
_face = [tuple(round(v, 3) for v in c.patches[0].get_facecolor()[:3])
         for c in _containers if c.patches]
check('every drawn bar series has its own colour',
      len(set(_face)) == len(_face),
      f'{len(_face)} series, {len(set(_face))} distinct')
# and a variant must keep its colour when the selection changes, or two screenshots of
# the same batch cannot be compared
_before = dict(zip(_all_vars, ns['_variant_colors'](_all_vars)))
document.set_checks('var-check', _all_vars[:3])
run(ns['trigger_plot'](None))
_after = dict(zip(_all_vars[:3], ns['_variant_colors'](_all_vars[:3])))
check('a variant keeps its colour when others are unticked',
      all(_before[v] == _after[v] for v in _all_vars[:3]))
document.set_checks('var-check', variants[:5])

# ── U14: the plot's (G, C) hint must agree with the matrices' own flags ──────
print('\n[U14 plot failure flags]')
_env = document.getElementById('env-select').value
document.getElementById('metric-select').value = 'n_steps'
document.set_checks('var-check', variants[:5])
document.set_checks('cand-check', cands)
run(ns['trigger_plot'](None))
_fmap = ns['_flag_pivot']('Candidate', _env, variants[:5], cands, 'n_steps')
_ctx_f, _ = ns['_summary_context']()
# In candidate mode both sides are keyed by (candidate, variant) and MUST agree — they are
# the same rule on the same numbers, and a disagreement means one of the two is lying.
_mismatch = []
for _c in _ctx_f['cands']:
    for _v in variants[:5]:
        _tbl = ns['_fail_flags'](_ctx_f['rec'], _c, _v)
        _plot = _fmap.get((str(_c), str(_v)), '')
        _want = ('G' if 'goal' in _tbl else '') + ('C' if 'constraint' in _tbl else '')
        if _want != _plot:
            _mismatch.append(f'CAND_{_c}/{_v}: table={_want!r} plot={_plot!r}')
check('plot flags match the Result Matrices exactly', not _mismatch,
      f'{len(_fmap)} flagged bars over {len(_ctx_f["cands"]) * 5} facets'
      if not _mismatch else '; '.join(_mismatch[:4]))
check('flag labels are the compact initials',
      ns['_flag_label']('GC') == '(G, C)' and ns['_flag_label']('') == '')
# a success plot must NOT be flagged — the flag would restate the bar's own height
check('the flag is skipped on its own inputs',
      ns['_flag_pivot']('Candidate', _env, variants[:5], cands, 'n_success') == {}
      and ns['_flag_pivot']('Candidate', _env, variants[:5], cands,
                            'n_success_and_constraints') == {})
# environment mode groups by the environment, so the keys change with the axis
_emap = ns['_flag_pivot']('halfspace_variant', _env, variants[:5], cands, 'n_steps')
check('environment mode flags are keyed by environment',
      all(k[0] in set(envs) for k in _emap), f'{len(_emap)} flagged bars')
# and the axis note has to explain the marks it drew
_xlab = ns['current_ax'].get_xlabel()
# one direction only: a flagged facet whose n_steps bar is itself missing draws no mark,
# so "flags exist" does not oblige a note — but a note with nothing flagged would be a lie.
check('the axes note never explains marks it did not draw',
      ('(G) = goal not always reached' not in _xlab) or bool(_fmap),
      _xlab[:90] or '(no note)')
document.getElementById('metric-select').value = 'n_success_and_constraints'
run(ns['trigger_plot'](None))

# ── U13: seed coverage in the plot legend ────────────────────────────────────
print('\n[U13 legend seed coverage]')
_seed_map = ns['_seed_map']()
check('seed map built from the per-seed frame',
      bool(_seed_map) and set(_seed_map) >= set(cands),
      f'{len(_seed_map)} candidates')
check('Seeds column in the plot legend', ('>Seeds<' in _legend_html) == bool(_seed_map))
# what the page claims a candidate has must be what df_raw actually holds
_probe = cands[0]
_truth = sorted(int(s) for s in pd.to_numeric(
    df_raw[df_raw['Candidate'] == _probe]['seed'], errors='coerce').dropna().unique())
check('Seeds cell matches df_raw', _seed_map.get(_probe, ([], []))[0] == _truth,
      f'CAND_{_probe}: page={_seed_map.get(_probe, ([], []))[0]} raw={_truth}')
# the caution is the point of the column: it must appear exactly when a candidate is
# short of the batch's full seed set, never as decoration.
_short = any(miss for _have, miss in _seed_map.values())
check('NOT FULL caution matches the data', ('NOT FULL' in _legend_html) == _short,
      f'{sum(1 for _h, m in _seed_map.values() if m)} of {len(_seed_map)} candidates short'
      if _short else 'every candidate has every seed — no caution, correctly')

# ── U13: candidate highlight ─────────────────────────────────────────────────
print('\n[U13 candidate highlight]')
document.getElementById('env-select').value = envs[0]
document.set_checks('cand-check', cands)
run(ns['trigger_plot'](None))
check('HL checkbox on every legend row',
      document.getElementById('selection-map-container').innerHTML.count('class="hl-check hl-box"')
      == len(cands), f'{len(cands)} rows')

_pick = cands[0]
_other = cands[1] if len(cands) > 1 else None


async def _toggle(cand, on):
    # exactly what the JS onchange does — set_highlight schedules the redraw
    ns['set_highlight'](cand, on)
    await asyncio.sleep(0.3)


run(_toggle(_pick, True))
_leg = document.getElementById('selection-map-container').innerHTML
_sum = document.getElementById('summary-container').innerHTML
_path = document.getElementById('path-map-container').innerHTML
_marked = f'<span class="hl-name">CAND_{_pick}</span>'
_marked_cell = f'<td class="rowhead">{_marked}</td>'
_plain_cell = f'<td class="rowhead">CAND_{_pick}</td>'
check('highlight reaches the plot legend', _marked in _leg)
# one row head per Result Matrix, plus the run-coverage table — and NOT ONE left plain,
# which is what a half-applied highlight would look like.
check('highlight reaches every result matrix',
      _sum.count(_marked_cell) >= len(ns['SUMMARY_TABLES']) and _plain_cell not in _sum,
      f'{_sum.count(_marked_cell)} row heads marked over {len(ns["SUMMARY_TABLES"])} tables')
check('highlight reaches the path audit map', _marked in _path)
check('highlight leaves other candidates alone',
      _other is None or f'<span class="hl-name">CAND_{_other}</span>' not in _sum,
      'SKIP — single-candidate batch' if _other is None else f'CAND_{_other} untouched')
# the set lives in Python precisely so the box survives the wholesale re-render
check('legend checkbox comes back checked after the re-render',
      f'value="{_pick}" checked' in _leg)
check('[CLEAR HIGHLIGHTS] offered while something is highlighted', 'hl-clear' in _leg)

# the x tick label of the highlighted candidate — the "PLOT naming" half of the feature
_ax = ns['current_ax']
_ticks = {lbl.get_text(): lbl for lbl in _ax.get_xticklabels()}
check('highlighted candidate gets a red bold x tick',
      _pick in _ticks
      and _ticks[_pick].get_color() == ns['HL_COLOR']
      and str(_ticks[_pick].get_fontweight()) == 'bold',
      f'ticks={list(_ticks)[:6]}')
check('other x ticks stay black',
      _other is None or _other not in _ticks or _ticks[_other].get_color() == 'black')

# a .tex / .txt has no colour, so the exports say it in words
_ctx_hl, _ = ns['_summary_context']()
_tex_hl = ns['build_latex'](_ctx_hl, 'batch', 'stamp')
check('LaTeX marks the highlighted candidate', 'highlighted in the viewer' in _tex_hl)

run(_toggle(_pick, False))
check('un-ticking removes the highlight everywhere',
      _marked not in document.getElementById('summary-container').innerHTML
      and _marked not in document.getElementById('path-map-container').innerHTML)

ns['highlighted_cands'].update(cands)
run(ns['_redraw_highlight']())
check('every candidate can be highlighted at once',
      len(ns['highlighted_cands']) == len(cands))


async def _clear():
    ns['clear_highlights']()
    await asyncio.sleep(0.3)


run(_clear())
check('clear_highlights wipes all of them',
      not ns['highlighted_cands']
      and 'hl-name' not in document.getElementById('summary-container').innerHTML)
run(ns['trigger_plot'](None))

# empty selection must explain itself, not go blank
document.set_checks('var-check', [])
run(ns['trigger_plot'](None))
check('U10 empty-selection message', 'NO PLOT' in document.getElementById('plot-area').innerHTML)
document.set_checks('var-check', variants[:5])

# a metric that exists for no selected variant must say so
document.getElementById('metric-select').value = 'frozen_worst_overlap_m' \
    if 'frozen_worst_overlap_m' in metrics else metrics[0]
run(ns['trigger_plot'](None))
area = document.getElementById('plot-area').innerHTML
check('U7 no-data message or a real plot', ('NO DATA' in area) or (len(displayed) > 1))
document.getElementById('metric-select').value = 'n_success_and_constraints'

# per-seed mode
document.getElementById('seed-mode-select').value = 'custom'
run(ns['trigger_plot'](None))
check('per-seed mode draws from va2_units_long', True, 'no exception')
document.getElementById('seed-mode-select').value = 'standard'
run(ns['trigger_plot'](None))

# ── LaTeX export path ────────────────────────────────────────────────────────
print('\n[export]')
ctx, err = ns['_summary_context']()
check('summary context built', ctx is not None, err or f'{len(ctx["cands"])} candidates')
tex = ns['build_latex'](ctx, 'batch', 'stamp')
check('LaTeX compiles-ish', all(t in tex for t in (r'\documentclass', r'\begin{tabular}',
                                                   r'\end{document}')))
check('LaTeX has a table per metric', tex.count(r'\begin{table}') == len(ns['SUMMARY_TABLES']),
      f'{tex.count(chr(92) + "begin{table}")} tables')
check('LaTeX lists candidate source paths', 'CAND_' in tex and 'verbatim' in tex)
check('LaTeX carries the INIT XY reference row',
      (ns['REF_LABEL'] in tex) == _has_init,
      f'{"present" if _has_init else "absent"}, as expected')

# ── mask actually changes the numbers, everywhere ────────────────────────────
print('\n[mask is live]')
key = 'constraint_exec_sat_rate'
tight = [e for e in envs if 'tightened' in e]
if tight:
    document.getElementById('env-select').value = tight[0]
    document.getElementById('mask-select').value = 'all'
    ns['derive_frames']()
    a = ns['df_agg']
    v_all = a[(a['metric'] == key) & (a['halfspace_variant'] == tight[0])]['mean'].mean()
    document.getElementById('mask-select').value = 'unfrozen'
    ns['derive_frames']()
    u = ns['df_agg']
    v_unf = u[(u['metric'] == key) & (u['halfspace_variant'] == tight[0])]['mean'].mean()
    check('aggregate numbers move with the mask', abs(v_all - v_unf) > 1e-6,
          f'all={v_all:.4f} unfrozen={v_unf:.4f}')
    check('data_version bumped so the matrix cache invalidates', ns['data_version'] >= 2,
          f'version={ns["data_version"]}')
    document.getElementById('mask-select').value = 'all'
    ns['derive_frames']()
else:
    print('  SKIP  no tightened geometry in this batch')

# ── PER-ROLLOUT view ─────────────────────────────────────────────────────────
print('\n[per-rollout view]')
roll = ns['df_roll_src']
geo_pick = next((e for e in envs if not roll[roll['geo'] == e].empty), envs[0])
document.getElementById('env-select').value = geo_pick
sub_roll = roll[roll['geo'] == geo_pick]
cand_pick = sorted(sub_roll['Candidate'].unique())[0]
var_pick = sorted(sub_roll[sub_roll['Candidate'] == cand_pick]['variant'].unique())[0]
document.getElementById('roll-cand-select').value = cand_pick
document.getElementById('roll-var-select').value = var_pick
document.getElementById('roll-sort-select').value = 'rollout_idx'
document.set_checks('rollcol-check', roll_cols[:8])

ns['show_rollout_table']()
table = document.getElementById('rollout-table-container').innerHTML
check('rollout table rendered', '<table' in table and '<tr' in table,
      f'{table.count("<tr")} rows')
check('rollout summary line', 'rollouts' in document.getElementById('rollout-summary').innerHTML)
n_all = table.count('<tr')

document.getElementById('mask-select').value = 'unfrozen'
ns['show_rollout_table']()
n_unf = document.getElementById('rollout-table-container').innerHTML.count('<tr')
tight_variant = tight and geo_pick in tight
check('rollout table shrinks under the mask (tightened geo only)',
      (n_unf < n_all) if tight_variant else (n_unf == n_all),
      f'{n_all} -> {n_unf} rows, geo={geo_pick}')
document.getElementById('mask-select').value = 'all'

document.getElementById('roll-sort-select').value = 'n_success'
ns['show_rollout_table']()
check('re-sort does not crash', '<table' in document.getElementById('rollout-table-container').innerHTML)
document.getElementById('roll-cand-select').value = 'no-such-candidate'
ns['show_rollout_table']()
check('missing selection explains itself',
      'No rollouts' in document.getElementById('rollout-summary').innerHTML)
document.getElementById('roll-cand-select').value = cand_pick
ns['show_rollout_table']()

# ── COMPARE view ─────────────────────────────────────────────────────────────
print('\n[compare view]')
# Pick a (geo, candidate) that actually HAS the visual-aligning rollout columns —
# the state-only avoiding candidates carry NaN there by construction.
_ok = roll.dropna(subset=['context_init_xy_dist', 'mean_dist_per_rollout'])
cmp_geo = sorted(_ok['geo'].unique())[0]
cmp_cand = sorted(_ok[_ok['geo'] == cmp_geo]['Candidate'].unique())[0]
cmp_vars = sorted(_ok[(_ok['geo'] == cmp_geo) & (_ok['Candidate'] == cmp_cand)]['variant'].unique())[:4]
document.getElementById('env-select').value = cmp_geo
document.getElementById('cmp-x-select').value = 'context_init_xy_dist'
document.getElementById('cmp-y-select').value = 'mean_dist_per_rollout'
document.getElementById('cmp-group-select').value = 'variant'
document.set_checks('cmp-var-check', cmp_vars)
document.set_checks('cmp-cand-check', [cmp_cand])
print(f'      (compare on geo={cmp_geo}, CAND_{cmp_cand}, {len(cmp_vars)} variants)')
for chart in ('scatter', 'bar', 'box'):
    window.cmpChartType = chart
    before = len(displayed)
    ns['trigger_compare']()
    check(f'{chart} compare drawn', len(displayed) > before,
          document.getElementById('engine-status').innerText
          or document.getElementById('compare-plot-area').innerHTML[:110])
document.set_checks('cmp-var-check', [])
ns['trigger_compare']()
check('compare with no selection explains itself',
      'Select at least one' in document.getElementById('compare-plot-area').innerHTML)

# an all-NaN pairing (state-only candidate x visual-only metric) must say so, not blank
_state_geo = next((e for e in envs if roll[(roll['geo'] == e)
                                           & roll['mean_dist_per_rollout'].isna()].shape[0] > 0), None)
if _state_geo:
    _c = sorted(roll[roll['geo'] == _state_geo]['Candidate'].unique())[0]
    document.getElementById('env-select').value = _state_geo
    document.set_checks('cmp-var-check', sorted(roll[roll['geo'] == _state_geo]['variant'].unique())[:3])
    document.set_checks('cmp-cand-check', [_c])
    window.cmpChartType = 'scatter'
    ns['trigger_compare']()
    check('all-NaN pairing explains itself',
          'NaN' in document.getElementById('compare-plot-area').innerHTML,
          document.getElementById('compare-plot-area').innerHTML[:90])
    document.getElementById('env-select').value = cmp_geo

# ── QUALITY view ─────────────────────────────────────────────────────────────
print('\n[quality view]')
ns['render_quality']()
qhtml = document.getElementById('quality-container').innerHTML
check('quality view rendered', ('flagged' in qhtml) or ('clean' in qhtml), qhtml[:70] + '…')

# ── refresh_global drives whichever view is open ─────────────────────────────
print('\n[global refresh]')
for view in ('aggregate', 'rollout', 'quality'):
    window.currentView = view
    try:
        run(ns['refresh_global'](None))
        ok = True
    except Exception as exc:                                        # noqa: BLE001
        ok = False
        print(f'      raised: {exc}')
    check(f'refresh_global in {view} view', ok)

print('\n' + '=' * 70)
if failures:
    print(f'{len(failures)} FAILURES: {failures}')
    sys.exit(1)
print('ALL CHECKS PASSED')
