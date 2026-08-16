"""A NaN cell must be MARKED, never drawn as a zero — checked on BOTH viewer pages.

    python Data_Analysis/Visualizer_VA_v2/test_nan_not_zero.py <path/to/batch_va2_.../>

Why this exists: pandas' bar plotter runs `fillna(0)` on the frame before handing it
to matplotlib, so a candidate/variant combination that was never run used to arrive
as a real 0-height bar — visually identical to a method that genuinely scored 0.000.
The fix lives in DAv3's `Visualizer/index.html` and is inherited by
`Visualizer_VA_v2/index.html` through build_from_dav3.py, so both are exercised here.

Needs pandas + matplotlib: runs on the cluster / any machine with the science stack,
not in the AI-coding container. The batch must be a DA_VA_v2 output folder (it also
carries the DAv3-shaped compat CSV that the DAv3 page reads).
"""
import asyncio
import pathlib
import sys

import pandas as pd
import matplotlib
matplotlib.use('agg')
from matplotlib.container import BarContainer

HERE = pathlib.Path(__file__).resolve().parent
PAGES = {'DAv3': HERE.parent / 'Visualizer' / 'index.html',
         'VAv2': HERE / 'index.html'}

if len(sys.argv) > 1:
    BATCH = pathlib.Path(sys.argv[1])
else:
    runs = sorted((HERE.parent / 'analysis_results').glob('batch_va2_*'), reverse=True)
    if not runs:
        print('usage: python test_nan_not_zero.py <path/to/batch_va2_.../>')
        sys.exit(2)
    BATCH = runs[0]

failures = []


def check(name, ok, detail=''):
    print(f'  {"PASS" if ok else "FAIL"}  {name}' + (f'  — {detail}' if detail else ''))
    if not ok:
        failures.append(name)


# ── browser stubs (same shape as test_page_offline.py) ───────────────────────
class Style:
    def __init__(self):
        self.display = ''
        self.color = self.width = self.height = self.maxWidth = ''


class ClassList:
    def toggle(self, *a):
        pass


class El:
    def __init__(self, eid, value=''):
        self.id, self.value = eid, value
        self.innerHTML = self.innerText = self.className = self.href = self.download = ''
        self.checked = self.disabled = False
        self.style, self.classList = Style(), ClassList()

    def click(self):
        pass


class Doc:
    def __init__(self):
        self.elements, self.classes = {}, {}

    def getElementById(self, eid):
        return self.elements.setdefault(eid, El(eid))

    def getElementsByClassName(self, cls):
        return self.classes.get(cls, [])

    def createElement(self, _tag):
        return El('tmp')

    def querySelectorAll(self, _sel):
        return []

    def set_checks(self, cls, values):
        self.classes[cls] = []
        for v in values:
            e = El(f'{cls}:{v}', str(v))
            e.checked = True
            self.classes[cls].append(e)


class Win:
    currentMode, currentView, cmpChartType = 'list', 'aggregate', 'scatter'

    def applyZoom(self):
        pass

    def applyViewMode(self):
        pass

    def alert(self, *a):
        pass


def load_page(path, document, window, figures):
    block = path.read_text().replace('\r\n', '\n').split('<script type="py">')[1].split('</script>')[0]
    block = "\n".join(l for l in block.splitlines()
                      if not l.startswith(('from js import', 'from pyscript import',
                                           'from pyodide.ffi import'))
                      and not l.startswith('asyncio.ensure_future'))
    block = block.replace('create_proxy(', 'identity(')
    ns = {'document': document, 'window': window, 'identity': lambda f: f,
          'console': type('C', (), {'log': staticmethod(lambda *a: None)})(),
          'display': lambda fig, target=None: figures.append(fig),
          'fetch': None, 'Uint8Array': None, 'File': None, 'URL': None}
    exec(compile(block, path.name, 'exec'), ns)
    return ns


print(f'[harness] batch: {BATCH.name}')
for name, path in PAGES.items():
    print(f'\n[{name}]')
    document, window, figures = Doc(), Win(), []
    ns = load_page(path, document, window, figures)

    if name == 'VAv2':
        for key, fname in (('df_agg_src', 'va2_aggregated_long.csv'),
                           ('df_units_src', 'va2_units_long.csv'),
                           ('df_roll_src', 'per_rollout_detail.csv'),
                           ('df_qual_src', 'data_quality.csv')):
            ns[key] = ns['_norm'](pd.read_csv(BATCH / fname, low_memory=False))
        document.getElementById('mask-select').value = 'all'
        document.getElementById('split-select').value = 'ALL'
        ns['populate_split_filter']()
        ns['derive_frames']()
    else:
        frame = pd.read_csv(BATCH / 'candidates_multidimensional_aggregated.csv', low_memory=False)
        for col in ('Candidate', 'variant', 'halfspace_variant', 'metric'):
            if col in frame.columns:
                frame[col] = frame[col].astype(str)
        ns['df_agg'], ns['df_raw'] = frame, None

    for eid, value in (('seed-mode-select', 'standard'), ('mode-select', 'candidate'),
                       ('fig-width', '11.0'), ('width-zoom', '1.0'),
                       ('metric-select', 'n_success')):
        document.getElementById(eid).value = value
    ns['populate_dynamic_filters']()

    agg = ns['df_agg']
    env = sorted(agg['halfspace_variant'].unique())[0]
    cands = sorted(agg['Candidate'].unique())
    variants = sorted(agg['variant'].unique())[:6]
    document.getElementById('env-select').value = env
    document.set_checks('cand-check', cands)
    document.set_checks('var-check', variants)
    asyncio.new_event_loop().run_until_complete(ns['trigger_plot'](None))

    ax = figures[-1].axes[0]
    marks = [t.get_text() for t in ax.texts].count('n/a')
    rects = [r for c in ax.containers if isinstance(c, BarContainer) for r in c]
    undrawn = sum(1 for r in rects if r.get_height() != r.get_height())
    zeros = sum(1 for r in rects if r.get_height() == 0)

    # ground truth straight from the frame: a (candidate, variant) pair with no row, or
    # a row whose value is NaN, is missing. Everything else is a real measurement.
    val = 'mean' if 'mean' in agg.columns else 'value'
    block = agg[(agg['metric'] == 'n_success') & (agg['halfspace_variant'] == env)
                & (agg['variant'].isin(variants)) & (agg['Candidate'].isin(cands))]
    have = set(zip(block[block[val].notna()]['Candidate'], block[block[val].notna()]['variant']))
    expected = len(cands) * len(variants) - len(have)

    check('every selected cell gets an x-slot (nothing silently dropped)',
          len(rects) == len(cands) * len(variants),
          f'{len(rects)} bars for {len(cands)}x{len(variants)}')
    check('missing cells are marked n/a', marks == expected,
          f'{marks} marks, {expected} missing cells')
    check('missing cells draw NO bar (pandas fillna(0) undone)', undrawn == expected,
          f'{undrawn} undrawn')
    check('a genuine 0.000 is NOT marked as missing', zeros == 0 or marks == expected,
          f'{zeros} real zero-height bars kept')
    check('the footnote appears iff something is missing',
          ('NO DATA' in ax.get_xlabel()) == (expected > 0), ax.get_xlabel()[:60])

print('\n' + '=' * 70)
if failures:
    print(f'{len(failures)} FAILURES: {failures}')
    sys.exit(1)
print('ALL CHECKS PASSED — NaN is marked, not zeroed, on both pages')
