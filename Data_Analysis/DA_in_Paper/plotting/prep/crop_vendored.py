#!/usr/bin/env python3.14
"""Cut the vendored diagnostic dashboards down to the panel the thesis argues from.

    python3.14 plotting/prep/crop_vendored.py [--check]

Reads sources.VENDORED_CROP -- name -> (box, what the box keeps) -- and writes the
cut image to data/prepared/<name><ext>. make_figs.py copies that file instead of
the source and refuses to fall back to the uncut one, so a dashboard cannot reach
a chapter the way it did in the first build of fig:raw-plans.

Needs PIL, hence python3.14: the builders and make_figs.py stay stdlib-only, which
is why this is a separate prepared file rather than a crop inside make_figs.py --
the same arrangement as extract/avoiding_scene.py and data/avoiding_scene.json.

--check reports what is stale (source newer than prepared, or box changed) and
writes nothing; it exits non-zero if anything needs rebuilding.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sources as S                                            # noqa: E402

# The box each prepared file was written with, so a changed box is detected as
# stale even when the file is newer than its source.
STAMP = os.path.join(S.PREPARED, 'CROPS.json')


def stamps():
    try:
        with open(STAMP) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def stale(name):
    """Why `name` needs rebuilding, or None."""
    src, dst = S.vendored_path(name), S.prepared_path(name)
    box = list(S.VENDORED_CROP[name][0])
    if not os.path.isfile(src):
        return f'source absent: {S.VENDORED[name][1]}'
    if not os.path.isfile(dst):
        return 'not built yet'
    if os.path.getmtime(src) > os.path.getmtime(dst):
        return 'source is newer'
    if stamps().get(name) != box:
        return f'box changed to {tuple(box)}'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true', help='report staleness, write nothing')
    a = ap.parse_args()

    todo = {n: stale(n) for n in S.VENDORED_CROP}
    if a.check:
        bad = {n: w for n, w in todo.items() if w}
        for n, w in sorted(bad.items()):
            print(f'STALE  {n}: {w}')
        print(f'{len(S.VENDORED_CROP) - len(bad)} of {len(S.VENDORED_CROP)} prepared files are current')
        return 1 if bad else 0

    from PIL import Image
    os.makedirs(S.PREPARED, exist_ok=True)
    written = stamps()
    rc = 0
    for name, (box, keeps) in S.VENDORED_CROP.items():
        src, dst = S.vendored_path(name), S.prepared_path(name)
        if not os.path.isfile(src):
            print(f'skip   {name}  (source absent)')
            rc = 1
            continue
        im = Image.open(src)
        if box[2] > im.width or box[3] > im.height:
            print(f'FAIL   {name}: box {box} outside the {im.width}x{im.height} source')
            rc = 1
            continue
        cut = im.convert('RGB').crop(box)
        cut.save(dst)
        written[name] = list(box)
        print(f'wrote  data/prepared/{os.path.basename(dst)}  '
              f'{im.width}x{im.height} -> {cut.width}x{cut.height}   ({keeps})')
    with open(STAMP, 'w') as fh:
        json.dump(written, fh, indent=2, sort_keys=True)
        fh.write('\n')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
