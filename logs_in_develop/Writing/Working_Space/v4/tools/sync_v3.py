#!/usr/bin/env python3
"""Detect and absorb changes made to ``v3`` after v4 was branched off it (v3.98, 2026-09-24).

The model is the ordinary three-way merge, with v3 as the upstream branch. v3 is
already split by chapter, so its files are read directly; nothing is re-split:

    base    inherited/v3_base/<file>   what v3 looked like when v4 last synced
    ours    <file>                     the v4 working copy
    theirs  ../v3/<file>               what v3 looks like now

Chapters 1-4 reach v4 THROUGH v3 (v2 -> v3 -> v4, one way): a v2 change is absorbed
by v3 first (v3/tools/sync_v2.py) and arrives here as a v3 change. The stamp records
both versions -- v3's own, and the v2 revision v3's inherited half is at -- so that
"which v2 and which v3 are in this v4?" is one command.

``status``   reports, per file, whether v3 moved (base vs. theirs) and whether v4
             moved (base vs. ours).
``diff``     prints v3's own change (base -> theirs) for one or all files.
``merge``    three-way-merges v3's change into the v4 working copy with
             ``git merge-file`` and advances the baseline on success. Conflicts are
             left in the file with the usual markers and the baseline is *not*
             advanced, so a failed merge can simply be redone.
``stamp``    records the current v3 revision as the baseline's (only when the
             baseline really equals v3 -- checked, never trusted).

Nothing here ever writes into ``../v3`` or ``../v2``. The dependency is one-way by design.

Usage
-----
    python3 tools/sync_v3.py status
    python3 tools/sync_v3.py diff  [chapters/06_results.tex]
    python3 tools/sync_v3.py merge [chapters/06_results.tex] [--dry-run]
    python3 tools/sync_v3.py stamp
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
V4 = os.path.dirname(HERE)
V3 = os.path.normpath(os.path.join(V4, '..', 'v3'))
BASE = os.path.join(V4, 'inherited', 'v3_base')
STATE = os.path.join(V4, 'inherited', 'SYNC_STATE.json')

# Per-file policy and OWNER. See inherited/MANIFEST.md for the rationale.
#   inherit  v4 does not edit this file; a v3 change is taken wholesale
#   merge    v4 edits it too; a v3 change needs a real three-way merge
#   own      v4 rewrote it; the baseline (v3's copy at branch time) exists only so
#            that drift becomes *visible* -- if v3 ever edits its own Ch 7-9 again,
#            `status` says so and a human decides
#   watch    v3's frozen copy of a file v4 no longer carries under that name (v3's
#            07_discussion / 08_conclusion; since v4.1 v4's are 07_conclusion and
#            08_discussion): drift in v3 is reported, nothing is merged, no v4 file
# The owner is the draft that writes the file: v2 (Ch 1-4, preamble, front matter,
# acronyms, bibliography.bib), v3 (Ch 5-6, its preamble additions, bibliography_v3.bib)
# or v4. bundle/make_bundle.py collapses chapters by owner.
POLICY = {
    'parts/00_preamble.tex':               ('inherit', 'v2'),
    'parts/00_preamble_v3.tex':            ('inherit', 'v3'),
    'parts/01_frontmatter.tex':            ('inherit', 'v2'),
    'parts/99_backmatter.tex':             ('merge',   'v2'),
    'chapters/01_introduction.tex':        ('inherit', 'v2'),
    'chapters/02_background.tex':          ('inherit', 'v2'),
    'chapters/03_related_work.tex':        ('inherit', 'v2'),
    'chapters/04_method.tex':              ('inherit', 'v2'),
    'chapters/05_setup.tex':               ('inherit', 'v3'),
    'chapters/06_results.tex':             ('inherit', 'v3'),
    'chapters/07_discussion.tex':          ('watch',   'v3'),
    'chapters/08_conclusion.tex':          ('watch',   'v3'),
    'chapters/07_conclusion.tex':          ('own',     'v4'),
    'chapters/08_discussion.tex':          ('own',     'v4'),
    'chapters/09_appendix.tex':            ('own',     'v4'),
    'chapters/app_ntrial20_feasible.tex':  ('own',     'v4'),
    'bibliography.bib':                    ('inherit', 'v2'),
    'bibliography_v3.bib':                 ('inherit', 'v3'),
}

RE_V3_HEADING = re.compile(r'^## v(\d+)\.(\d+)([a-z]?)\b(.*)$')


def v3_version():
    """The HIGHEST version heading in v3's CHANGELOG, e.g. 'v3.98a — 2026-09-24 20:19 · ...'.

    Highest, not first (v3 learned this the hard way: a stale entry once sat above the
    newest). A letter suffix (v3.98a) sorts after its number. Returns (text, out_of_order).
    """
    best, first = None, None
    try:
        with open(os.path.join(V3, 'CHANGELOG.md')) as f:
            for ln in f:
                m = RE_V3_HEADING.match(ln)
                if not m:
                    continue
                key = (int(m.group(1)), int(m.group(2)), m.group(3))
                text = ln[3:].strip()
                if first is None:
                    first = (key, text)
                if best is None or key > best[0]:
                    best = (key, text)
    except OSError:
        pass
    if best is None:
        return 'unknown', False
    return best[1], first[0] != best[0]


def v2_version_via_v3():
    """The v2 revision v3's inherited half is at, as v3's own sync tool stamped it."""
    try:
        with open(os.path.join(V3, 'inherited', 'SYNC_STATE.json')) as f:
            return json.load(f).get('v2_version', 'unknown')
    except (OSError, ValueError):
        return 'unknown'


def digest(path):
    h = hashlib.sha256()
    try:
        with open(path, 'rb') as f:
            h.update(f.read())
    except OSError:
        return None
    return h.hexdigest()[:16]


def load_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(**kw):
    st = load_state()
    st.update(kw)
    st['synced_at'] = datetime.date.today().isoformat()
    st['v3_version'], _ = v3_version()
    st['v2_version_via_v3'] = v2_version_via_v3()
    st['v3_changelog_sha256_16'] = digest(os.path.join(V3, 'CHANGELOG.md'))
    st['base_sha256_16'] = {rel: digest(os.path.join(BASE, rel)) for rel in POLICY}
    with open(STATE, 'w') as f:
        json.dump(st, f, indent=2, sort_keys=True)
        f.write('\n')
    return st


def read(path):
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        return None


def classify():
    """-> list of (rel, policy, owner, v3_moved, v4_moved)."""
    rows = []
    for rel, (pol, owner) in POLICY.items():
        b, o, t = (read(os.path.join(BASE, rel)),
                   read(os.path.join(V4, rel)),
                   read(os.path.join(V3, rel)))
        rows.append((rel, pol, owner, t != b, (o != b) if pol != 'watch' else False))
    return rows


def cmd_status(_a):
    st = load_state()
    print(f'inherited half is at : {st.get("v3_version", "NOT STAMPED")}')
    print(f'  which carries v2 at: {st.get("v2_version_via_v3", "-")}')
    print(f'last synced          : {st.get("synced_at", "-")}')
    now, out_of_order = v3_version()
    if st.get('v3_version') and now != st['v3_version']:
        print(f'v3 is now at         : {now}   <-- moved')
    if out_of_order:
        print("note                 : v3's CHANGELOG is not in descending version order; "
              'the highest entry is used, not the first.')
    print()
    rows = classify()
    w = max(len(r[0]) for r in rows)
    print(f'{"file".ljust(w)}  policy   owner  v3 moved  v4 moved')
    print('-' * (w + 40))
    for rel, pol, owner, v3m, v4m in rows:
        print(f'{rel.ljust(w)}  {pol:<7}  {owner:<5}  {"YES" if v3m else "  -":>8}  {"YES" if v4m else "  -":>8}')
    stale = [r for r in rows if r[3]]
    print()
    if not stale:
        print('v4 is in sync with v3: no inherited file has moved upstream.')
        return 0
    print(f'{len(stale)} file(s) moved in v3 since the last sync:')
    for rel, pol, _owner, _v3m, v4m in stale:
        how = ('three-way merge needed (v4 edited it too)' if v4m else
               'fast-forward: v3 change can be taken wholesale')
        if pol == 'own':
            how = 'v3 edited a file v4 OWNS -- read v3/CHANGELOG.md and decide by hand'
        if pol == 'watch':
            how = 'v3 edited its FROZEN copy -- read v3/CHANGELOG.md; nothing to merge'
        print(f'  {rel}  [{pol}, {_owner}]  -> {how}')
    print('\nInspect with:  python3 tools/sync_v3.py diff <file>')
    print('Absorb with:   python3 tools/sync_v3.py merge [<file>]')
    return 1


def cmd_diff(a):
    rows = [r for r in classify() if (a.file is None and r[3]) or r[0] == a.file]
    if not rows:
        print('nothing to diff.')
        return 0
    for rel, _pol, _owner, _v3m, _v4m in rows:
        subprocess.run(['diff', '-u',
                        '--label', f'v3_base/{rel}  (baseline)',
                        '--label', f'v3/{rel}  (current v3)',
                        os.path.join(BASE, rel), os.path.join(V3, rel)])
    return 0


def cmd_merge(a):
    rows = [r for r in classify() if r[3] and (a.file is None or r[0] == a.file)]
    if not rows:
        print('nothing to merge: no inherited file has moved in v3.')
        return 0
    conflicts = 0
    for rel, pol, owner, _v3m, v4m in rows:
        ours, base, theirs = (os.path.join(V4, rel), os.path.join(BASE, rel),
                              os.path.join(V3, rel))
        if pol in ('own', 'watch'):
            print(f'skipped  {rel}  [{pol}, {owner}]: v3 edited a v4-owned or frozen file; decide by hand')
            continue
        if a.dry_run:
            print(f'would merge {rel}  [{pol}, {owner}, v4 {"edited" if v4m else "clean"}]')
            continue
        r = subprocess.run(['git', 'merge-file', '-L', 'v4', '-L', 'v3-baseline', '-L', 'v3-current',
                            ours, base, theirs])
        if r.returncode < 0 or r.returncode > 100:
            sys.exit(f'git merge-file failed on {rel}')
        if r.returncode:
            conflicts += 1
            print(f'CONFLICT ({r.returncode} hunk(s)) in {rel} -- resolve, then re-run merge')
        else:
            shutil.copyfile(theirs, base)     # advance the baseline only on a clean merge
            print(f'merged   {rel}  [{pol}, {owner}]')
    if conflicts:
        print(f'\n{conflicts} file(s) left with conflict markers; their baseline was NOT advanced.')
        print('Resolve the markers, then re-run:  python3 tools/sync_v3.py merge')
    elif not a.dry_run:
        st = save_state()
        print(f'\nBaseline advanced to {st["v3_version"]} (stamped in inherited/SYNC_STATE.json).')
        print('Record the absorbed change in CHANGELOG.md.')
    return 1 if conflicts else 0


def cmd_stamp(_a):
    moved = [r[0] for r in classify() if r[3]]
    if moved:
        print('refusing to stamp: these files differ from the baseline, so the stamp would be false:')
        for rel in moved:
            print(f'  {rel}')
        print('\nRun `merge` first, or copy the file into inherited/v3_base/ by hand.')
        return 1
    st = save_state()
    print(f'stamped: inherited half of v4 is at {st["v3_version"]}')
    print(f'         which carries v2 at {st["v2_version_via_v3"]}')
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status')
    sub.add_parser('stamp')
    d = sub.add_parser('diff')
    d.add_argument('file', nargs='?')
    m = sub.add_parser('merge')
    m.add_argument('file', nargs='?')
    m.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    return {'status': cmd_status, 'diff': cmd_diff, 'merge': cmd_merge, 'stamp': cmd_stamp}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
