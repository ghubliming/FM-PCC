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


# ── stub the science stack ───────────────────────────────────────────────────
# Only _palette actually reaches into it, and it reaches for COLORMAPS — which is exactly
# what wants testing, so they are stubbed with known contents (including a deliberate
# overlap between two maps) rather than left absent.
for name in ('pandas', 'matplotlib', 'matplotlib.pyplot'):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules['matplotlib'].pyplot = sys.modules['matplotlib.pyplot']

FAKE_CMAPS = {
    'tab10': [(i / 10.0, 0.0, 0.0) for i in range(10)],
    # tab20 really does contain every tab10 colour: the generator must not count them twice
    'tab20': [(i / 10.0, 0.0, 0.0) for i in range(10)]
             + [(0.0, i / 10.0, 0.0) for i in range(10)],
    'tab20b': [(0.0, 0.0, i / 10.0) for i in range(10)],
}
FAKE_UNIQUE = 30      # 10 + 10 new + 10 new


def _fake_get_cmap(name):
    if name in FAKE_CMAPS:
        return types.SimpleNamespace(colors=FAKE_CMAPS[name])
    if name in ('turbo', 'hsv'):                      # continuous top-up
        return lambda x: (x, 1.0 - x, (x * 7.0) % 1.0, 1.0)
    raise ValueError(name)                            # the maps this stub does not define


sys.modules['matplotlib.pyplot'].get_cmap = _fake_get_cmap


class _El:
    def __init__(self, eid):
        self.id = eid
        self.value = ''
        self.innerHTML = ''
        self.innerText = ''
        self.checked = False
        self.style = types.SimpleNamespace(display='block')


class _Doc:
    """Minimal document. Elements are CACHED so a renderer's innerHTML can be read back."""

    def __init__(self):
        self.elements = {}

    def getElementById(self, eid):
        return self.elements.setdefault(eid, _El(eid))

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

    # ── U15: no two variants may share a colour ──────────────────────────────
    palette = ns['_palette']
    check('an empty request gives an empty palette', palette(0) == [])
    bad = []
    for want in (1, 2, 9, 10, 11, 20, 25, 30, 31, 48, 96):
        got = palette(want)
        keys = {tuple(round(c[i], 3) for i in range(3)) for c in got}
        if len(got) != want or len(keys) != want:
            bad.append(f'n={want}: {len(got)} colours, {len(keys)} distinct')
    check('every palette size is exactly n distinct colours', not bad, '; '.join(bad))
    # the overlap between the fake tab10 and tab20 must be collapsed, not counted twice —
    # this is the real tab10 ⊂ tab20 relationship, and the reason a naive concat is wrong
    check('duplicate colours across maps are collapsed',
          len({tuple(round(c[i], 3) for i in range(3)) for c in palette(FAKE_UNIQUE)})
          == FAKE_UNIQUE, f'{FAKE_UNIQUE} unique across the qualitative maps')
    # past the qualitative maps the continuous top-up takes over and must still not repeat
    check('the continuous top-up stays distinct',
          len({tuple(round(c[i], 3) for i in range(3))
               for c in palette(FAKE_UNIQUE + 40)}) == FAKE_UNIQUE + 40)
    check('a palette is a prefix of every larger one (stable ordering)',
          palette(5) == palette(40)[:5])
    check('colours are 4-tuple RGBA', all(len(c) == 4 and c[3] == 1.0 for c in palette(12)))

    # ── U16: the "5. Variants" quick presets ─────────────────────────────────
    members = ns['_preset_members']
    # a realistic DAv3 variant list, deliberately including everything a preset must NOT take
    all_vars = sorted([
        'diffuser',
        'dpcc-r', 'dpcc-c', 'dpcc-t',
        'dpcc-r-tightened', 'dpcc-c-tightened', 'dpcc-t-tightened',
        'dpcc-c-tightened-dt0p25', 'dpcc-c-tightened-dt4p0',      # scaling sweep, not an arm
        'hardflow_new', 'hardflow_new-c', 'hardflow_new-c-tightened',
        'gradient', 'gradient-tightened', 'post_processing',      # not projection arms
        'model_free', 'geo_free', 'bounds_free',
    ])
    _full = members('dpcc_hf', all_vars)
    check('DPCC + HF takes diffuser, every dpcc arm and every hardflow arm',
          _full == ['diffuser', 'dpcc-c', 'dpcc-c-tightened', 'dpcc-r', 'dpcc-r-tightened',
                    'dpcc-t', 'dpcc-t-tightened', 'hardflow_new', 'hardflow_new-c',
                    'hardflow_new-c-tightened'], f'{len(_full)}: {_full}')
    _t = members('dpcc_hf_tight', all_vars)
    check('DPCC + HF (tightened) takes only the tightened arms',
          _t == ['diffuser', 'dpcc-c-tightened', 'dpcc-r-tightened', 'dpcc-t-tightened',
                 'hardflow_new-c-tightened'], f'{len(_t)}: {_t}')
    _d = members('dpcc_tight', all_vars)
    check('DPCC (tightened) drops HardFlow too',
          _d == ['diffuser', 'dpcc-c-tightened', 'dpcc-r-tightened', 'dpcc-t-tightened'],
          f'{len(_d)}: {_d}')
    # the exclusions are the point: a silently-included dt sweep puts four near-identical
    # bars next to the one that matters
    _leaked = {v for key, _l, _t2 in ns['VARIANT_PRESETS'] for v in members(key, all_vars)
               if 'dt0p' in v or 'dt4p' in v or v.split('-')[0] in
               ('gradient', 'post_processing', 'model_free', 'geo_free', 'bounds_free')}
    check('no preset leaks a dt sweep or a non-projection baseline', not _leaked, sorted(_leaked))
    check('every preset is a subset of the batch variants',
          all(set(members(k, all_vars)) <= set(all_vars) for k, _l, _t3 in ns['VARIANT_PRESETS']))
    check('presets nest: tightened-dpcc <= tightened-both <= all',
          set(_d) <= set(_t) <= set(_full))
    # a batch with nothing but the baseline must offer NO preset, not a diffuser-only one
    check('diffuser alone is not a preset',
          all(members(k, ['diffuser', 'gradient']) == [] for k, _l, _t4 in ns['VARIANT_PRESETS']))
    check('an empty batch yields no preset',
          all(members(k, []) == [] for k, _l, _t5 in ns['VARIANT_PRESETS']))

    # and the rendered control
    ns['render_variant_presets'](all_vars)
    _panel = ns['document'].getElementById('variant-presets')
    check('all three presets render for a full DAv3 batch',
          _panel.innerHTML.count('class="preset-check"') == 3
          and _panel.style.display == 'block',
          f'{_panel.innerHTML.count(chr(34) + "preset-check" + chr(34))} rendered')
    check('each preset carries its members in data-members',
          f'data-members="{"|".join(_d)}"' in _panel.innerHTML)
    # a visual-aligning batch keeps tightening on the geometry axis, so the two tightened
    # presets have no members — they must be dropped AND explained, not left dead
    ns['render_variant_presets'](['diffuser', 'dpcc-c', 'dpcc-r', 'hardflow_new-c', 'gradient'])
    _panel = ns['document'].getElementById('variant-presets')
    check('a batch without -tightened names offers only the one preset that applies',
          _panel.innerHTML.count('class="preset-check"') == 1
          and 'Not offered by this batch' in _panel.innerHTML
          and '4. Geometry Focus' in _panel.innerHTML)
    ns['render_variant_presets'](['gradient', 'model_free'])
    _panel = ns['document'].getElementById('variant-presets')
    check('a batch with no projection arms hides the panel entirely',
          _panel.innerHTML == '' and _panel.style.display == 'none')

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
               'render_selection_map', '_flag_label', '_flag_pivot',
               '_cmap', '_palette', '_variant_colors',
               '_preset_members', 'render_variant_presets'):
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
                   'NOT FULL', '>Seeds<', '>HL<',
                   'id="variant-presets"', 'onchange="toggle_variant_preset(this)"',
                   'function sync_variant_presets()', 'sync_variant_presets();'):
        check(f'both pages carry {marker!r}',
              all(marker in loaded[t][1] for t in loaded),
              ' / '.join(f'{t}:{loaded[t][1].count(marker)}' for t in loaded))

print('\n' + '=' * 70)
if failures:
    print(f'{len(failures)} FAILURES: {failures}')
    sys.exit(1)
print('ALL CHECKS PASSED')
