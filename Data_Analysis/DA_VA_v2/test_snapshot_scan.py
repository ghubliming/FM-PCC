"""Regression test for the config-snapshot "Last Run" timestamps.

    python Data_Analysis/DA_VA_v2/test_snapshot_scan.py

Stdlib only, on a synthetic tree — so unlike the pandas-based DA / viewer tests
this one runs in the AI-coding container as well as on the cluster.

It covers the four places the feature lives, and in particular that they agree:

  DA_VA_v2/discovery.py                      scan_snapshot_timestamps + formatters
  DA_Code_v3/multi_candidate_discovery.py    the sibling copy (generation pattern)
  Visualizer/index.html                      _fmt_stamp, exec'd standalone
  Visualizer_VA_v2/index.html                the same function, inherited via
                                             build_from_dav3.py

A drift between the pipeline formatter and the page formatter would show up as
two different renderings of the same run in the CSV and in the Path Audit Map,
which is exactly the confusion the column exists to remove.
"""
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # repo root
DA_V3 = os.path.join(ROOT, 'Data_Analysis', 'DA_Code_v3')

# HERE must win: both generations ship a top-level `config` module and DA_VA_v2's
# discovery imports it by bare name. DA_Code_v3's multi_candidate_discovery has no
# such dependency, so it is happy further down the path.
sys.path.insert(0, DA_V3)
sys.path.insert(0, HERE)

import discovery as va2                                              # noqa: E402
import multi_candidate_discovery as dav3                             # noqa: E402

failures = []


def check(name, ok, detail=''):
    print(f'  {"PASS" if ok else "FAIL"}  {name}' + (f'  — {detail}' if detail else ''))
    if not ok:
        failures.append(name)


def eq(name, got, want):
    check(name, got == want, '' if got == want else f'got {got!r}, want {want!r}')


# ── synthetic run tree ───────────────────────────────────────────────────────
def make_tree(root):
    """Mirror what utils/setup.py::snapshot_configs leaves behind.

    cand_multi  seed 6: three eval launches   seed 7: one, and it is the newest
                seed 8: results but no config_snapshot dir at all (older run)
    cand_none   seeds, no snapshot markers anywhere
    cand_wide   seeds 6 and 10 — guards the numeric ordering of the per-seed string
    """
    def stamp(candidate, seed, name, config='avoiding-d3il'):
        directory = os.path.join(root, candidate, str(seed),
                                 f'config_snapshot_{config}')
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, name), 'w') as handle:
            handle.write('Snapshot taken at: (test)\n')

    for seed in (6, 7, 8):
        os.makedirs(os.path.join(root, 'cand_multi', str(seed), 'results'))
    stamp('cand_multi', 6, 'snapshot_20260506_034806')
    stamp('cand_multi', 6, 'snapshot_20260507_101500')
    stamp('cand_multi', 6, 'snapshot_20260504_223649')
    stamp('cand_multi', 7, 'snapshot_20260601_090000')
    # Non-marker files living in the same directory must never be mistaken for one.
    directory = os.path.join(root, 'cand_multi', '6', 'config_snapshot_avoiding-d3il')
    for junk in ('avoiding-d3il.py', 'projection_eval.yaml',
                 'snapshot_not_a_timestamp', 'snapshot_2026_0102',
                 'snapshot_20260506_034806.bak'):
        with open(os.path.join(directory, junk), 'w') as handle:
            handle.write('x')

    for seed in (6, 7):
        os.makedirs(os.path.join(root, 'cand_none', str(seed), 'results'))

    for seed in (6, 10):
        os.makedirs(os.path.join(root, 'cand_wide', str(seed), 'results'))
    stamp('cand_wide', 6, 'snapshot_20260101_000000')
    stamp('cand_wide', 10, 'snapshot_20260102_120000')


root = tempfile.mkdtemp(prefix='snapshot_scan_test_')
try:
    make_tree(root)
    multi = os.path.join(root, 'cand_multi')
    none = os.path.join(root, 'cand_none')
    wide = os.path.join(root, 'cand_wide')

    print('\nscan_snapshot_timestamps')
    got = va2.scan_snapshot_timestamps(multi)
    eq('latest is the newest stamp across seeds', got['latest'], '20260601_090000')
    eq('first is the oldest stamp across seeds', got['first'], '20260504_223649')
    eq('count is every marker, not every seed', got['count'], 4)
    eq('per_seed holds each seed\'s own newest',
       got['per_seed'], {6: '20260507_101500', 7: '20260601_090000'})
    eq('a seed with no snapshot dir is absent, not blank',
       8 in got['per_seed'], False)
    eq('n_seeds_stamped counts stamped seeds only', got['n_seeds_stamped'], 2)

    print('\nempty / missing trees')
    for label, path in (('no markers anywhere', none),
                        ('path does not exist', os.path.join(root, 'nope'))):
        got = va2.scan_snapshot_timestamps(path)
        eq(f'{label}: latest is empty', got['latest'], '')
        eq(f'{label}: count is 0', got['count'], 0)
        eq(f'{label}: per_seed is empty', got['per_seed'], {})

    print('\nseed filtering')
    eq('seeds=None auto-discovers every numeric subdir',
       va2.scan_snapshot_timestamps(multi, None)['n_seeds_stamped'], 2)
    eq('an explicit seed list restricts the scan',
       va2.scan_snapshot_timestamps(multi, [6])['latest'], '20260507_101500')
    eq('a seed list with nothing stamped yields empty',
       va2.scan_snapshot_timestamps(multi, [8])['latest'], '')

    print('\nDA_VA_v2 and DA_Code_v3 copies agree')
    for path in (multi, none, wide):
        name = os.path.basename(path)
        eq(f'{name}: identical scan result',
           dav3.scan_snapshot_timestamps(path), va2.scan_snapshot_timestamps(path))

    print('\nformatters')
    eq('compact stamp -> human stamp',
       va2.format_snapshot_ts('20260506_034806'), '2026-05-06 03:48:06')
    eq('empty in, empty out', va2.format_snapshot_ts(''), '')
    eq('None in, empty out', va2.format_snapshot_ts(None), '')
    eq('unparseable passes through', va2.format_snapshot_ts('whenever'), 'whenever')
    eq('DA_Code_v3 formats identically',
       dav3.format_snapshot_ts('20260506_034806'), va2.format_snapshot_ts('20260506_034806'))
    eq('per-seed string sorts seeds numerically, not lexicographically',
       va2.snapshot_by_seed_str(va2.scan_snapshot_timestamps(wide)['per_seed']),
       '6:20260101_000000 | 10:20260102_120000')
    eq('per-seed string of nothing is empty', va2.snapshot_by_seed_str({}), '')

    print('\nthe HTML pages format stamps the same way')
    pages = [os.path.join(ROOT, 'Data_Analysis', 'Visualizer', 'index.html'),
             os.path.join(ROOT, 'Data_Analysis', 'Visualizer_VA_v2', 'index.html')]
    samples = ['20260506_034806', '', 'nan', 'None', 'whenever', '2026-05-06 03:48:06']
    for page in pages:
        label = os.path.basename(os.path.dirname(page))
        with open(page) as handle:
            source = handle.read()
        match = re.search(r'^def _fmt_stamp\(stamp\):\n(?:[ \t].*\n|\n)*',
                          source, re.M)
        if match is None:
            check(f'{label}: _fmt_stamp found in the page', False)
            continue
        namespace = {}
        exec(match.group(0), namespace)                              # noqa: S102
        page_fmt = namespace['_fmt_stamp']
        # 'nan'/'None' are what a NaN cell becomes after .astype(str); the page must
        # blank those, whereas the pipeline never sees them.
        mismatched = [s for s in samples
                      if s not in ('nan', 'None')
                      and page_fmt(s) != va2.format_snapshot_ts(s)]
        check(f'{label}: agrees with the pipeline formatter', not mismatched,
              f'differs on {mismatched}' if mismatched else '')
        eq(f'{label}: a NaN cell renders blank, not "nan"', page_fmt('nan'), '')
finally:
    shutil.rmtree(root, ignore_errors=True)

print()
if failures:
    print(f'{len(failures)} FAILED: ' + ', '.join(failures))
    sys.exit(1)
print('all snapshot-timestamp checks passed')
