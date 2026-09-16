#!/usr/bin/env python3.14
"""Extract reproducible stills from the real simulator GIFs used in Chapter 5.

    python3.14 plotting/prep/extract_env_frames.py [--check]

The output belongs in data/prepared/ because the figure builder is stdlib-only.
The D3IL montage is tracked by git.  UAV rollout GIFs live in the ignored temp/
result tree, so their prepared stills are the portable record; the exact source,
frame and crop remain declared in sources.ENV_RENDER_FRAMES and in the audit note.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sources as S                                            # noqa: E402

STAMP = os.path.join(S.PREPARED, 'ENV_FRAMES.json')


def spec_for(name, spec):
    return {'source': spec['source'], 'frame': spec['frame'],
            'crop': list(spec['crop']) if spec['crop'] else None}


def stamps():
    try:
        with open(STAMP) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def stale(name, spec, old):
    src = os.path.join(S.REPO, spec['source'])
    dst = os.path.join(S.PREPARED, name + '.png')
    if not os.path.isfile(src):
        return f"source absent: {spec['source']}"
    if not os.path.isfile(dst):
        return 'not built yet'
    if old.get(name) != spec_for(name, spec):
        return 'frame or crop declaration changed'
    if os.path.getmtime(src) > os.path.getmtime(dst):
        return 'source is newer'
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--check', action='store_true')
    args = ap.parse_args()
    old = stamps()
    bad = {name: stale(name, spec, old)
           for name, spec in S.ENV_RENDER_FRAMES.items()}
    bad = {name: why for name, why in bad.items() if why}
    if args.check:
        for name, why in sorted(bad.items()):
            print(f'STALE  {name}: {why}')
        print(f'{len(S.ENV_RENDER_FRAMES) - len(bad)} of '
              f'{len(S.ENV_RENDER_FRAMES)} environment frames are current')
        return 1 if bad else 0

    from PIL import Image
    os.makedirs(S.PREPARED, exist_ok=True)
    written = dict(old)
    rc = 0
    for name, spec in S.ENV_RENDER_FRAMES.items():
        src = os.path.join(S.REPO, spec['source'])
        if not os.path.isfile(src):
            print(f'skip   {name}  (source absent: {spec["source"]})')
            rc = 1
            continue
        with Image.open(src) as image:
            if spec['frame'] >= getattr(image, 'n_frames', 1):
                print(f'FAIL   {name}: frame {spec["frame"]} outside source')
                rc = 1
                continue
            image.seek(spec['frame'])
            still = image.convert('RGB')
            if spec['crop']:
                still = still.crop(spec['crop'])
            dst = os.path.join(S.PREPARED, name + '.png')
            still.save(dst)
        written[name] = spec_for(name, spec)
        print(f'wrote  data/prepared/{name}.png  {still.width}x{still.height}  '
              f'({spec["what"]})')
    with open(STAMP, 'w') as fh:
        json.dump(written, fh, indent=2, sort_keys=True)
        fh.write('\n')
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
