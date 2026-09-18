#!/usr/bin/env python3
"""Copy the figures a draft uses from the store into that draft's figures/ folder.

    python3 export_to_draft.py v3                # by name, under Working_Space/
    python3 export_to_draft.py /path/to/draft    # or by path
    python3 export_to_draft.py v3 --dry-run      # show what would happen
    python3 export_to_draft.py v3 --prune        # also delete draft figures it no longer uses

The rule: **every thesis figure is produced in Data_Analysis/DA_in_Paper/figures/,
and a draft only ever holds copies.** A figure is never drawn or edited inside a
draft. When a figure changes, re-run make_figs.py, then this.

What it does:
  1. reads the draft's .tex sources (the master file, parts/, chapters/) and collects
     every \\includegraphics{name} outside comments;
  2. finds <name>.{svg,pdf,png,jpg,jpeg} in the store, in any group, and copies every
     format present into <draft>/figures/ (a PDF made by svg/svg2pdf.sh travels too);
  3. reports names that are planned (sources.PLANNED) or missing -- missing exits 1;
     and REFUSES to export when a figure's .png is older than its .svg, because the
     PDF renders the PNG and the draft would silently keep the old drawing;
  4. writes <draft>/figures/EXPORTED.md: each figure's group, formats, SHA-256 prefix
     and the file and line that use it.

A figure name must be unique across the store's groups; the export refuses a name
found in two groups rather than guess.
"""
import argparse
import datetime
import glob
import hashlib
import os
import re
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import sources as S                              # noqa: E402
from make_figs import GROUPS, IMAGE_EXT, STORE   # noqa: E402

WORKING_SPACE = os.path.join(S.REPO, 'logs_in_develop', 'Writing', 'Working_Space')
RE_GRAPHIC = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}')


def sha12(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()[:12]


def draft_sources(draft):
    """The .tex files LaTeX reads for this draft -- never bundle/ or handover/ copies."""
    files = sorted(glob.glob(os.path.join(draft, '*.tex')))
    for sub in ('parts', 'chapters'):
        files += sorted(glob.glob(os.path.join(draft, sub, '*.tex')))
    return files


def referenced(draft):
    """-> {name: [(relpath, line), ...]} for every \\includegraphics outside comments."""
    refs = {}
    for f in draft_sources(draft):
        for n, line in enumerate(open(f), 1):
            code = re.sub(r'(?<!\\)%.*$', '', line)
            for name in RE_GRAPHIC.findall(code):
                stem = os.path.splitext(name.strip())[0]
                refs.setdefault(stem, []).append((os.path.relpath(f, draft), n))
    return refs


def store_files(stem):
    """-> (group or None, [paths]) for one figure name; exits on a cross-group clash."""
    hits = []
    for g in GROUPS:
        for ext in IMAGE_EXT:
            p = os.path.join(STORE, g, stem + ext)
            if os.path.isfile(p):
                hits.append((g, p))
    groups = {g for g, _ in hits}
    if len(groups) > 1:
        sys.exit(f'export: "{stem}" exists in several groups ({", ".join(sorted(groups))}); '
                 'delete the stale copy in the store and re-run.')
    return (hits[0][0] if hits else None), [p for _, p in hits]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('draft', help='draft name under Working_Space/ (e.g. v3) or a path')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--prune', action='store_true',
                    help='delete image files in <draft>/figures/ that the draft does not use')
    a = ap.parse_args()

    draft = a.draft if os.path.isdir(a.draft) else os.path.join(WORKING_SPACE, a.draft)
    draft = os.path.abspath(draft)
    if not draft_sources(draft):
        sys.exit(f'export: no .tex sources found in {draft}')
    out = os.path.join(draft, 'figures')
    refs = referenced(draft)
    print(f'draft: {os.path.relpath(draft, S.REPO)} -- {len(refs)} figure(s) referenced')

    # A figure that ships both .svg and .png is RENDERED FROM THE PNG: graphicx picks
    # the raster, not the vector, so a PNG older than its SVG puts a figure in the thesis
    # that no longer matches the builder. That shipped a stale figure for two passes
    # (v3.36 redrew the alignment box to scale; the PDF kept showing the old marker
    # because only the SVG had been rebuilt). It is a hard error now, not a warning.
    stale = []
    for stem in sorted(refs):
        _g, paths = store_files(stem)
        svg = next((q for q in paths if q.endswith('.svg')), None)
        png = next((q for q in paths if q.endswith('.png')), None)
        if svg and png and os.path.getmtime(png) < os.path.getmtime(svg):
            stale.append((stem, os.path.relpath(svg, STORE), os.path.relpath(png, STORE)))
    if stale:
        lines = '\n'.join(f'    {stem}: {png} is older than {svg}' for stem, svg, png in stale)
        cmds = '\n'.join(
            f'    python3.14 plotting/svg/preview_png.py figures/{svg} figures/{png} --scale 3'
            for _stem, svg, png in stale)
        sys.exit(f'export: {len(stale)} figure(s) would ship a PNG older than their SVG, and the\n'
                 f'PDF renders the PNG -- the draft would show the OLD drawing:\n{lines}\n\n'
                 f'Refresh them, then export again:\n{cmds}')

    rows, planned, missing, copied, unchanged = [], [], [], 0, 0
    for stem in sorted(refs):
        group, paths = store_files(stem)
        where = ', '.join(f'{p}:{ln}' for p, ln in refs[stem])
        if not paths:
            (planned if stem in S.PLANNED else missing).append((stem, where))
            continue
        fmts = []
        for src in paths:
            dst = os.path.join(out, os.path.basename(src))
            fmts.append(os.path.splitext(src)[1][1:])
            if os.path.isfile(dst) and sha12(dst) == sha12(src):
                unchanged += 1
                continue
            print(f'  {"would copy" if a.dry_run else "copy"}  {group}/{os.path.basename(src)}')
            if not a.dry_run:
                os.makedirs(out, exist_ok=True)
                shutil.copyfile(src, dst)
            copied += 1
        rows.append((stem, group, '/'.join(fmts), sha12(paths[0]), where))

    pruned = []
    if a.prune and os.path.isdir(out):
        for fn in sorted(os.listdir(out)):
            stem, ext = os.path.splitext(fn)
            if ext.lower() in IMAGE_EXT and stem not in refs:
                pruned.append(fn)
                print(f'  {"would prune" if a.dry_run else "prune"} {fn}')
                if not a.dry_run:
                    os.remove(os.path.join(out, fn))

    if not a.dry_run:
        os.makedirs(out, exist_ok=True)
        L = ['# Exported figures — generated by `Data_Analysis/DA_in_Paper/plotting/export_to_draft.py`', '',
             f'Exported **{datetime.date.today().isoformat()}** from `Data_Analysis/DA_in_Paper/figures/`. '
             'These are copies: change a figure in the store, re-run `make_figs.py`, then export again.', '',
             '| figure | group | formats | sha256 | used in |', '| :-- | :-- | :-- | :-- | :-- |']
        L += [f'| `{s}` | {g} | {f} | `{h}` | {w} |' for s, g, f, h, w in rows]
        if planned:
            L += ['', '## Planned, not yet in the store', '']
            L += [f'- `{s}` — {w}' for s, w in planned]
        if missing:
            L += ['', '## ⚠️ Missing — referenced by the draft, absent from the store', '']
            L += [f'- `{s}` — {w}' for s, w in missing]
        L.append('')
        with open(os.path.join(out, 'EXPORTED.md'), 'w') as f:
            f.write('\n'.join(L))

    print(f'\n{copied} file(s) {"to copy" if a.dry_run else "copied"}, {unchanged} already current'
          + (f', {len(pruned)} pruned' if a.prune else '') + '.')
    for s, w in planned:
        print(f'planned (not in store yet): {s}  <- {w}')
    for s, w in missing:
        print(f'MISSING from the store: {s}  <- {w}')
    return 1 if missing else 0


if __name__ == '__main__':
    sys.exit(main())
