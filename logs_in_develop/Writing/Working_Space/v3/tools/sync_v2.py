#!/usr/bin/env python3
"""Detect and absorb changes made to ``v2`` after v3 was branched off it.

The model is the ordinary three-way merge, with v2 as the upstream branch:

    base    inherited/v2_base/<file>   what v2 looked like when v3 last synced
    ours    <file>                     the v3 working copy
    theirs  (v2 re-split, in a tempdir) what v2 looks like now

``status``   re-splits v2 and reports, per file, whether v2 moved (base vs.
             theirs) and whether v3 moved (base vs. ours).
``diff``     prints v2's own change (base -> theirs) for one or all files.
``merge``    three-way-merges v2's change into the v3 working copy with
             ``git merge-file`` and advances the baseline on success. Conflicts
             are left in the file with the usual markers and the baseline is
             *not* advanced, so a failed merge can simply be redone.

Nothing here ever writes into ``../v2``. The dependency is one-way by design.

Usage
-----
    python3 tools/sync_v2.py status
    python3 tools/sync_v2.py diff  [chapters/02_background.tex]
    python3 tools/sync_v2.py merge [chapters/02_background.tex] [--dry-run]
"""
import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
V3 = os.path.dirname(HERE)
BASE = os.path.join(V3, 'inherited', 'v2_base')
V2_TEX = os.path.normpath(os.path.join(V3, '..', 'v2', 'thesis_v2.tex'))
V2_BIB = os.path.normpath(os.path.join(V3, '..', 'v2', 'bibliography.bib'))

# Per-file policy. See inherited/MANIFEST.md for the rationale.
#   inherit  v3 does not edit this file; a v2 change is taken wholesale
#   merge    v3 edits it too; a v2 change needs a real three-way merge
#   own      v3 rewrote it; the baseline exists only so drift is *visible*
POLICY = {
    'parts/00_preamble.tex':        'inherit',
    'parts/01_frontmatter.tex':     'inherit',
    'parts/99_backmatter.tex':      'merge',
    'chapters/01_introduction.tex': 'inherit',
    'chapters/02_background.tex':   'merge',
    'chapters/03_related_work.tex': 'inherit',
    'chapters/04_method.tex':       'merge',
    'chapters/05_setup.tex':        'own',
    'chapters/06_results.tex':      'own',
    'chapters/07_discussion.tex':   'own',
    'chapters/08_conclusion.tex':   'own',
    'chapters/09_appendix.tex':     'own',
    # INHERIT, not merge: v3's own entries live in bibliography_v3.bib, so this
    # file only ever moves upstream and always fast-forwards. See the header of
    # bibliography_v3.bib for why both drafts appending to one .bib is the worst
    # possible conflict shape.
    'bibliography.bib':             'inherit',
}

STATE = os.path.join(V3, 'inherited', 'SYNC_STATE.json')


def v2_version():
    """The newest version heading in v2's CHANGELOG, e.g. 'v2.6 - 2026-09-10 - ...'.

    This is the human-readable answer to "which v2 is the inherited half of v3
    at?", and it is what gets stamped into SYNC_STATE.json.
    """
    try:
        with open(os.path.normpath(os.path.join(V3, '..', 'v2', 'CHANGELOG.md'))) as f:
            for ln in f:
                if ln.startswith('## v2'):
                    return ln[3:].strip()
    except OSError:
        pass
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
    st['v2_version'] = v2_version()
    st['thesis_v2_sha256_16'] = digest(V2_TEX)
    st['bibliography_sha256_16'] = digest(V2_BIB)
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


def theirs_dir():
    """Re-split the current v2 into a tempdir and return its path."""
    tmp = tempfile.mkdtemp(prefix='v2split_')
    r = subprocess.run([sys.executable, os.path.join(HERE, 'split_v2.py'), tmp,
                        '--source', V2_TEX, '--quiet'],
                       capture_output=True, text=True)
    if r.returncode:
        shutil.rmtree(tmp, ignore_errors=True)
        sys.exit(r.stdout + r.stderr)
    shutil.copyfile(V2_BIB, os.path.join(tmp, 'bibliography.bib'))
    return tmp


def classify(tmp):
    """-> list of (rel, policy, v2_moved, v3_moved)."""
    rows = []
    for rel, pol in POLICY.items():
        b, o, t = (read(os.path.join(BASE, rel)),
                   read(os.path.join(V3, rel)),
                   read(os.path.join(tmp, rel)))
        rows.append((rel, pol, t != b, o != b))
    return rows


def cmd_status(_a, tmp):
    st = load_state()
    print(f'inherited half is at : {st.get("v2_version", "NOT STAMPED")}')
    print(f'last synced          : {st.get("synced_at", "-")}')
    now = v2_version()
    if st.get('v2_version') and now != st['v2_version']:
        print(f'v2 is now at         : {now}   <-- moved')
    print()
    rows = classify(tmp)
    w = max(len(r[0]) for r in rows)
    print(f'{"file".ljust(w)}  policy   v2 moved  v3 moved')
    print('-' * (w + 30))
    for rel, pol, v2m, v3m in rows:
        print(f'{rel.ljust(w)}  {pol:<7}  {"YES" if v2m else "  -":>8}  {"YES" if v3m else "  -":>8}')
    stale = [r for r in rows if r[2]]
    print()
    if not stale:
        print('v3 is in sync with v2: no inherited file has moved upstream.')
        return 0
    print(f'{len(stale)} file(s) moved in v2 since the last sync:')
    for rel, pol, _v2m, v3m in stale:
        how = ('three-way merge needed (v3 edited it too)' if v3m else
               'fast-forward: v2 change can be taken wholesale')
        print(f'  {rel}  [{pol}]  -> {how}')
    print('\nInspect with:  python3 tools/sync_v2.py diff <file>')
    print('Absorb with:   python3 tools/sync_v2.py merge [<file>]')
    return 1


def cmd_diff(a, tmp):
    rows = [r for r in classify(tmp) if (a.file is None and r[2]) or r[0] == a.file]
    if not rows:
        print('nothing to diff.')
        return 0
    for rel, _pol, _v2m, _v3m in rows:
        subprocess.run(['diff', '-u',
                        '--label', f'v2_base/{rel}  (baseline)',
                        '--label', f'v2/{rel}  (current v2)',
                        os.path.join(BASE, rel), os.path.join(tmp, rel)])
    return 0


def cmd_merge(a, tmp):
    rows = [r for r in classify(tmp) if r[2] and (a.file is None or r[0] == a.file)]
    if not rows:
        print('nothing to merge: no inherited file has moved in v2.')
        return 0
    conflicts = 0
    for rel, pol, _v2m, v3m in rows:
        ours, base, theirs = (os.path.join(V3, rel), os.path.join(BASE, rel),
                              os.path.join(tmp, rel))
        if a.dry_run:
            print(f'would merge {rel}  [{pol}, v3 {"edited" if v3m else "clean"}]')
            continue
        r = subprocess.run(['git', 'merge-file', '-L', 'v3', '-L', 'v2-baseline', '-L', 'v2-current',
                            ours, base, theirs])
        if r.returncode < 0 or r.returncode > 100:
            sys.exit(f'git merge-file failed on {rel}')
        if r.returncode:
            conflicts += 1
            print(f'CONFLICT ({r.returncode} hunk(s)) in {rel} -- resolve, then re-run merge')
        else:
            shutil.copyfile(theirs, base)     # advance the baseline only on a clean merge
            print(f'merged   {rel}  [{pol}]')
    if conflicts:
        print(f'\n{conflicts} file(s) left with conflict markers; their baseline was NOT advanced.')
        print('Resolve the markers, then re-run:  python3 tools/sync_v2.py merge')
    elif not a.dry_run:
        st = save_state()
        print(f'\nBaseline advanced to {st["v2_version"]} (stamped in inherited/SYNC_STATE.json).')
        print('Record the absorbed change in CHANGELOG.md.')
    return 1 if conflicts else 0


def cmd_stamp(_a, tmp):
    """Record the current v2 revision as the baseline's, without merging.

    Only correct when the baseline really does equal v2 -- at branch time, or
    after a merge that was resolved by hand. `status` refuses to lie about it,
    so it is checked here rather than trusted.
    """
    moved = [r[0] for r in classify(tmp) if r[2]]
    if moved:
        print('refusing to stamp: these files differ from the baseline, so the '
              'stamp would be false:')
        for rel in moved:
            print(f'  {rel}')
        print('\nRun `merge` first, or copy the file into inherited/v2_base/ by hand.')
        return 1
    st = save_state()
    print(f'stamped: inherited half of v3 is at {st["v2_version"]}')
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

    tmp = theirs_dir()
    try:
        return {'status': cmd_status, 'diff': cmd_diff, 'merge': cmd_merge,
                'stamp': cmd_stamp}[a.cmd](a, tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    sys.exit(main())
