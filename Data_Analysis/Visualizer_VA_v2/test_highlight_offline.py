"""U13 regression test — candidate highlight + legend seed coverage, stdlib only.

    python3 Data_Analysis/Visualizer_VA_v2/test_highlight_offline.py

Companion to test_page_offline.py, which drives the whole page against real CSVs and
therefore needs pandas + matplotlib (cluster only). This one stubs both libraries out
so the *pure* highlight logic can be checked in the AI-coding container, and — more
importantly — asserts that DAv3 and DA_VA_v2 carry the SAME implementation. The VA
page is generated from the DAv3 one by build_from_dav3.py; a hand-edit to either that
splits them would otherwise only show up as two pages disagreeing about which
candidate is red.
"""
import pathlib
import sys
import types

HERE = pathlib.Path(__file__).resolve().parent
PAGES = {
    'DAv3': HERE.parent / 'Visualizer' / 'index.html',
    'DA_VA_v2': HERE / 'index.html',
}

failures = []


def check(name, ok, detail=''):
    print(f'  {"PASS" if ok else "FAIL"}  {name}' + (f'  — {detail}' if detail else ''))
    if not ok:
        failures.append(name)


# ── stub the science stack: none of the functions under test touch it ────────
for name in ('pandas', 'matplotlib', 'matplotlib.pyplot'):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules['matplotlib'].pyplot = sys.modules['matplotlib.pyplot']


class _Doc:
    """Minimal document: only seed-mode/checkbox reads reach it from these helpers."""

    def __init__(self):
        self.values = {}

    def getElementById(self, eid):
        return types.SimpleNamespace(value=self.values.get(eid, ''), innerHTML='',
                                     style=types.SimpleNamespace(display='block'))

    def getElementsByClassName(self, _cls):
        return []


def load(path):
    src = path.read_text()
    block = src.split('<script type="py">')[1].split('</script>')[0]
    block = "\n".join(line for line in block.splitlines()
                      if not line.startswith(('from js import', 'from pyscript import',
                                              'from pyodide.ffi import'))
                      and not line.startswith('asyncio.ensure_future'))
    block = block.replace('create_proxy(', 'identity(')
    ns = {'document': _Doc(), 'window': types.SimpleNamespace(applyZoom=lambda: None),
          'console': types.SimpleNamespace(log=lambda *a: None),
          'display': lambda *a, **k: None, 'identity': lambda f: f,
          'fetch': None, 'Uint8Array': None, 'File': None, 'URL': None}
    exec(compile(block, path.name, 'exec'), ns)
    return ns, src


class _Label:
    """Stand-in for a matplotlib Text tick label."""

    def __init__(self, text):
        self.text = text
        self.color = None
        self.weight = None

    def get_text(self):
        return self.text

    def set_color(self, c):
        self.color = c

    def set_fontweight(self, w):
        self.weight = w


def run_page(tag, ns):
    print(f'\n[{tag}]')
    hl = ns['highlighted_cands']
    hl.clear()

    # ── the name renderer: one ticked box, one red name ──────────────────────
    check('plain name when not highlighted', ns['_cand_name_html']('65') == 'CAND_65')
    hl.add('65')
    check('highlighted name is wrapped in .hl-name',
          ns['_cand_name_html']('65') == '<span class="hl-name">CAND_65</span>',
          ns['_cand_name_html']('65'))
    check('only the highlighted candidate changes', ns['_cand_name_html']('6') == 'CAND_6')
    # the sidebar hands values over as JS strings; the plot index can be an int
    check('int and str candidate ids agree', ns['_hl'](65) and ns['_hl']('65'))

    # ── the plot x tick labels ───────────────────────────────────────────────
    labels = [_Label(c) for c in ('6', '65', '7')]
    ns['_remember_plot'](types.SimpleNamespace(get_xticklabels=lambda: labels),
                         'Candidate', ['6', '65', '7'])
    ns['_apply_tick_highlight']()
    check('the highlighted tick goes red + bold',
          labels[1].color == ns['HL_COLOR'] and labels[1].weight == 'bold')
    check('the other ticks are explicitly reset to black',
          all(x.color == 'black' and x.weight == 'normal' for x in (labels[0], labels[2])))

    # environment mode has no candidate on the x-axis: mark nothing
    labels2 = [_Label('halfspace_0'), _Label('halfspace_1')]
    ns['_remember_plot'](types.SimpleNamespace(get_xticklabels=lambda: labels2),
                         'halfspace_variant', ['halfspace_0', 'halfspace_1'])
    ns['_apply_tick_highlight']()
    check('environment mode marks no tick', all(x.color is None for x in labels2))

    # a label/category count mismatch must mark NOTHING rather than the wrong bar group
    labels3 = [_Label('6'), _Label('65')]
    ns['_remember_plot'](types.SimpleNamespace(get_xticklabels=lambda: labels3),
                         'Candidate', ['6', '65', '7'])
    ns['_apply_tick_highlight']()
    check('a tick/category mismatch marks nothing', all(x.color is None for x in labels3))

    # ── the Seeds cell ───────────────────────────────────────────────────────
    cell = ns['_seed_cell']({'65': ([6, 7, 8, 9, 10], [])}, '65')
    check('a complete candidate lists its seeds with no caution',
          '6, 7, 8, 9, 10' in cell and 'NOT FULL' not in cell, cell)
    cell = ns['_seed_cell']({'65': ([6, 7], [8, 9, 10])}, '65')
    check('a short candidate is cautioned and the missing seeds named',
          'NOT FULL' in cell and 'missing 8, 9, 10' in cell, cell)
    cell = ns['_seed_cell']({'65': (None, [9])}, '65')
    check('no per-seed CSV says so instead of claiming zero seeds',
          'n/a' in cell and 'none' not in cell and 'missing 9' in cell, cell)
    check('an unknown candidate degrades to a dash',
          '&mdash;' in ns['_seed_cell']({}, '65'))

    # ── U14: the plot's (G, C) failure hint ──────────────────────────────────
    check('no flag renders as nothing at all', ns['_flag_label']("") == "")
    check('goal-only flag', ns['_flag_label']("G") == "(G)")
    check('constraint-only flag', ns['_flag_label']("C") == "(C)")
    check('both flags', ns['_flag_label']("GC") == "(G, C)")
    # the flag must never be drawn on the metrics that define it
    check('the flag inputs are skipped on their own plots',
          set(ns['FLAG_INPUTS']) <= ns['FLAG_SKIP'],
          f'skip={sorted(ns["FLAG_SKIP"])}')

    hl.clear()
    check('clearing puts every name back', ns['_cand_name_html']('65') == 'CAND_65')


loaded = {}
for tag, path in PAGES.items():
    if not path.exists():
        check(f'{tag} page exists', False, str(path))
        continue
    loaded[tag] = load(path)
    run_page(tag, loaded[tag][0])

# ── the two pages must not drift apart ───────────────────────────────────────
print('\n[DAv3 <-> DA_VA_v2 agreement]')
if len(loaded) == 2:
    def def_source(src, name):
        # inspect.getsource cannot reach an exec'd block, so slice the text: from the
        # `def name(` line to the next line that starts in column 0.
        block = src.split('<script type="py">')[1].split('</script>')[0].replace('\r\n', '\n')
        needle = f'\ndef {name}('
        if needle not in block:
            needle = f'\nasync def {name}('
        start = block.index(needle) + 1
        lines = block[start:].splitlines()
        out = [lines[0]]
        for line in lines[1:]:
            if line and not line[0].isspace():
                break
            out.append(line)
        return "\n".join(out).rstrip()

    # _flag_pivot is in the list but FLAG_SKIP deliberately is NOT: the VA page adds its
    # relaxed success pair to the skip set, which is the one intended difference.
    for fn in ('_hl', '_cand_name_html', '_remember_plot', '_apply_tick_highlight',
               '_seed_map', '_seed_cell', 'set_highlight', 'clear_highlights',
               'render_selection_map', '_flag_label', '_flag_pivot'):
        a = def_source(loaded['DAv3'][1], fn)
        b = def_source(loaded['DA_VA_v2'][1], fn)
        check(f'{fn} identical on both pages', a == b,
              '' if a == b else f'{len(a)} vs {len(b)} chars')
    a = def_source(loaded['DAv3'][1], '_redraw_highlight').replace('async def', 'def')
    b = def_source(loaded['DA_VA_v2'][1], '_redraw_highlight').replace('async def', 'def')
    check('_redraw_highlight identical on both pages', a == b)
    # the markup the two pages emit has to match too, not just the python
    for marker in ('class="hl-check hl-box"', 'onchange="toggle_highlight(this)"',
                   'function toggle_highlight(el)', 'function clear_highlights()',
                   'document.set_highlight = identity(set_highlight)'.replace('identity', 'create_proxy'),
                   'NOT FULL', '>Seeds<', '>HL<'):
        check(f'both pages carry {marker!r}',
              all(marker in loaded[t][1] for t in loaded),
              ' / '.join(f'{t}:{loaded[t][1].count(marker)}' for t in loaded))

print('\n' + '=' * 70)
if failures:
    print(f'{len(failures)} FAILURES: {failures}')
    sys.exit(1)
print('ALL CHECKS PASSED')
