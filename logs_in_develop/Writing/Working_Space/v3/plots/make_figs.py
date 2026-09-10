#!/usr/bin/env python3
"""Regenerate the thesis v3 figures from the batch CSVs.

    python3 make_figs.py                 # every figure, into ../figures/
    python3 make_figs.py k_ladder         # only figures whose name contains this
    python3 make_figs.py --list           # what would be built, and from what
    python3 make_figs.py --out /tmp/x     # somewhere else

Writes ``../figures/MANIFEST.md`` alongside the figures: which corpus each one
came from, at what protocol, on what date. That file is the reason a reader can
tell a figure built from the current data from one left over from a corpus that
has since been superseded -- and the reason a stale figure is a visible fact
rather than a silent one.

Stdlib only, by design: see fmpcc_svg.py.
"""
import argparse
import datetime
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import figures            # noqa: E402
import sources as S       # noqa: E402

DEFAULT_OUT = os.path.normpath(os.path.join(HERE, '..', 'figures'))


def write_manifest(outdir, built, skipped):
    lines = [
        '# Figure manifest — generated, do not edit',
        '',
        f'Built by `plots/make_figs.py` on **{datetime.date.today().isoformat()}**.',
        'Regenerate with `python3 plots/make_figs.py`; to point at new data, edit',
        '`plots/sources.py` and nothing else.',
        '',
        '| figure | corpus | protocol |',
        '| :-- | :-- | :-- |',
    ]
    for name, prov in built:
        corpus, _, protocol = prov.partition(' | ')
        lines.append(f'| `{name}.svg` | `{corpus}` | {protocol} |')
    if skipped:
        lines += ['', '## Not built', '',
                  'Its corpus is not on this machine. `temp/` is a local drop directory and is',
                  'not version-controlled, so this is expected on a fresh checkout — download the',
                  'batch, then re-run.', '',
                  '| figure | missing corpus |', '| :-- | :-- |']
        for name, why in skipped:
            lines.append(f'| `{name}.svg` | {why} |')
    lines += [
        '', '## Corpus registry', '',
        'Mirrors §10 of `data_status/DATASTATUS_20260910_v3_entry_readiness.md`.',
        'A `missing` row is a download away; a `blocked` grade is a *run* away.', '',
        '| key | on disk | grade | protocol |', '| :-- | :-- | :-- | :-- |',
    ]
    for k, c in S.CORPORA.items():
        lines.append(f'| `{k}` | {"yes" if c.available else "**no**"} | {c.status} | {c.protocol} |')
    lines += [
        '', '## Format', '',
        'SVG. `\\includegraphics` needs PDF, so run `tools/svg2pdf.sh` before a LaTeX build;',
        'the container that writes the thesis has no SVG converter, which is why the figures',
        'are committed as SVG and converted where the build happens.', '',
    ]
    with open(os.path.join(outdir, 'MANIFEST.md'), 'w') as f:
        f.write('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('match', nargs='?', help='build only figures whose name contains this')
    ap.add_argument('--out', default=DEFAULT_OUT)
    ap.add_argument('--list', action='store_true')
    a = ap.parse_args()

    if a.list:
        for k, c in S.CORPORA.items():
            print(f'{"ok " if c.available else "MISS"}  {k:<18} {c.rel}')
        print()
        for name, _fn in figures.ALL:
            print(f'      {name}.svg')
        return 0

    todo = [(n, fn) for n, fn in figures.ALL if not a.match or a.match in n]
    if not todo:
        print(f'no figure matches {a.match!r}; try --list')
        return 1

    os.makedirs(a.out, exist_ok=True)
    built, skipped = [], []
    for name, fn in todo:
        try:
            res = fn(a.out)
        except Exception as e:                      # a broken figure must not kill the rest
            print(f'FAIL  {name}: {type(e).__name__}: {e}')
            skipped.append((name, f'builder raised {type(e).__name__}'))
            continue
        if res is None:
            print(f'skip  {name}  (corpus not on this machine)')
            skipped.append((name, 'corpus directory absent'))
            continue
        path, prov = res
        print(f'wrote {os.path.basename(path)}   <- {prov}')
        built.append((name, prov))

    write_manifest(a.out, built, skipped)
    print(f'\n{len(built)} figure(s) in {a.out}; manifest updated.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
