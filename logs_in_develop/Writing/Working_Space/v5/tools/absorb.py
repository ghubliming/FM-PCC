#!/usr/bin/env python3
"""absorb.py -- carry a change made in a LEGACY draft (v2 / v3 / v4) into v5, the aggregate.

Since 2026-09-25 (author) the thesis lives in v5 and is edited there by the Orchestra. The legacy drafts are
kept and "used less": when the author takes a big job to an owner chat, the owner edits its own files as
before, and this tool brings that change into v5 with an ordinary three-way merge:

    base    inherited/materials/<draft>/<file>    the owner's file as it stood when v5 last absorbed it (at init: v5.0)
    ours    v5/<file>                             v5's working copy, which may have moved on its own
    theirs  ../<draft>/<file>                     the owner's live file now

v2 is one monolith (v2/thesis_v2.tex). It is split the way the release tool and v3/tools/split_v2.py split it
(content markers, never line numbers) and compared part by part against inherited/materials/v2_split/.

    python3 tools/absorb.py status                    has a legacy source moved since v5 last absorbed it? did v5 edit the same file?
    python3 tools/absorb.py diff  [<v5 file>]         the owner's own change (base -> theirs)
    python3 tools/absorb.py merge [<v5 file>] [--dry-run]
                                                      three-way merge into v5 with `git merge-file`; the base is advanced only on a
                                                      clean merge; conflicts stay in the v5 file with the usual markers
    python3 tools/absorb.py versions                  the v2 / v3 / v4 revisions now, against those v5 last absorbed

Nothing here ever writes into ../v2, ../v3 or ../v4 (one-way, as the drafts' own sync tools). A merge is then
recorded by hand: a new `## v5.N` entry in v5/CHANGELOG.md, and the legacy owner's INBOX row marked 🔀 v5.N.
Python 3 stdlib + git.
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
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
V5 = os.path.dirname(HERE)
WS = os.path.dirname(V5)
MATERIALS = os.path.join(V5, 'inherited', 'materials')
STATE = os.path.join(V5, 'inherited', 'ABSORB_STATE.json')
INIT_STATE = os.path.join(V5, 'inherited', 'INIT_STATE.json')

# v5 file -> (draft, live source relative to Working_Space, base relative to inherited/materials)
# 'split' sources are parts of the v2 monolith; their base is the split kept in materials/v2_split/.
MAP = {
    'parts/00_preamble.tex':                    ('v2', 'v2/thesis_v2.tex', 'v2_split/parts/00_preamble.tex'),
    'parts/01_frontmatter.tex':                 ('v2', 'v2/thesis_v2.tex', 'v2_split/parts/01_frontmatter.tex'),
    'chapters/01_introduction.tex':             ('v2', 'v2/thesis_v2.tex', 'v2_split/chapters/01_introduction.tex'),
    'chapters/02_background.tex':               ('v2', 'v2/thesis_v2.tex', 'v2_split/chapters/02_background.tex'),
    'chapters/03_related_work.tex':             ('v2', 'v2/thesis_v2.tex', 'v2_split/chapters/03_related_work.tex'),
    'chapters/04_method.tex':                   ('v2', 'v2/thesis_v2.tex', 'v2_split/chapters/04_method.tex'),
    'parts/99_backmatter.tex':                  ('v2', 'v2/thesis_v2.tex', 'v2_split/parts/99_backmatter.tex'),
    'bibliography.bib':                         ('v2', 'v2/bibliography.bib', 'v2/bibliography.bib'),
    'parts/00_preamble_v3.tex':                 ('v3', 'v3/parts/00_preamble_v3.tex', 'v3/parts/00_preamble_v3.tex'),
    'chapters/05_setup.tex':                    ('v3', 'v3/chapters/05_setup.tex', 'v3/chapters/05_setup.tex'),
    'chapters/06_results.tex':                  ('v3', 'v3/chapters/06_results.tex', 'v3/chapters/06_results.tex'),
    'bibliography_v3.bib':                      ('v3', 'v3/bibliography_v3.bib', 'v3/bibliography_v3.bib'),
    'parts/00_preamble_v4.tex':                 ('v4', 'v4/parts/00_preamble_v4.tex', 'v4/parts/00_preamble_v4.tex'),
    'chapters/07_conclusion.tex':               ('v4', 'v4/chapters/07_conclusion.tex', 'v4/chapters/07_conclusion.tex'),
    'chapters/08_discussion.tex':               ('v4', 'v4/chapters/08_discussion.tex', 'v4/chapters/08_discussion.tex'),
    'chapters/09_appendix.tex':                 ('v4', 'v4/chapters/09_appendix.tex', 'v4/chapters/09_appendix.tex'),
    'chapters/app_long/uav_corridor_rules.tex': ('v4', 'v4/chapters/app_long/uav_corridor_rules.tex', 'v4/chapters/app_long/uav_corridor_rules.tex'),
    'chapters/app_ntrial20_feasible.tex':       ('v4', 'v4/chapters/app_ntrial20_feasible.tex', 'v4/chapters/app_ntrial20_feasible.tex'),
    'bibliography_v4.bib':                      ('v4', 'v4/bibliography_v4.bib', 'v4/bibliography_v4.bib'),
}
V2_CHAPTERS = {'ch:introduction': 'chapters/01_introduction.tex', 'ch:background': 'chapters/02_background.tex',
               'ch:related': 'chapters/03_related_work.tex', 'ch:method': 'chapters/04_method.tex'}
RE_VERSION = re.compile(r'^## v(\d+)\.(\d+)([a-z]?)\b(.*)$')
RE_LABEL = re.compile(r'\\label\{([^}]+)\}')


def read(path):
    try:
        with open(path, encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return None


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16] if text is not None else None


def draft_version(draft):
    """Highest '## vN.M[a]' heading of <draft>/CHANGELOG.md -> 'v3.100b -- date . title' (the owners' own rule)."""
    best = None
    text = read(os.path.join(WS, draft, 'CHANGELOG.md')) or ''
    for ln in text.split('\n'):
        m = RE_VERSION.match(ln)
        if m:
            key = (int(m.group(1)), int(m.group(2)), m.group(3))
            if best is None or key > best[0]:
                best = (key, re.sub(r'\s*→.*$', '', ln[3:].strip()))
    return best[1] if best else 'unknown'


def split_v2(text):
    """The v2 monolith -> {v5 rel: text} for the seven parts v5 takes (the markers of v3/tools/split_v2.py)."""
    lines = text.split('\n')

    def find(pred, what):
        for i, ln in enumerate(lines):
            if pred(ln):
                return i
        sys.exit(f'absorb: v2 marker not found: {what}')
    i_begin = find(lambda l: l.startswith(r'\begin{document}'), r'\begin{document}')
    i_main = find(lambda l: l.startswith(r'\mainmatter{}'), r'\mainmatter{}')
    i_app = find(lambda l: l.startswith(r'\appendix{}'), r'\appendix{}')
    i_abbr = find(lambda l: l.startswith(r'\addchap{Abbreviations}'), r'\addchap{Abbreviations}')
    i_end = find(lambda l: l.startswith(r'\end{document}'), r'\end{document}')
    i_back = i_abbr
    for j in range(i_abbr - 1, max(i_abbr - 6, i_main), -1):
        if lines[j].startswith(r'\microtypesetup{protrusion=false}'):
            i_back = j
            break
    starts = [i for i in range(i_main, i_app) if lines[i].startswith(r'\chapter{')]
    out = {'parts/00_preamble.tex': '\n'.join(lines[:i_begin]) + '\n',
           'parts/01_frontmatter.tex': '\n'.join(lines[i_begin + 1:starts[0]]) + '\n',
           'parts/99_backmatter.tex': '\n'.join(lines[i_back:i_end]) + '\n'}
    bounds = starts + [i_app]
    for k, a in enumerate(starts):
        m = RE_LABEL.search(lines[a])
        label = m.group(1) if m else None
        if label in V2_CHAPTERS:
            out[V2_CHAPTERS[label]] = '\n'.join(lines[a:bounds[k + 1]]) + '\n'
    return out


_v2_split_cache = {}


def theirs_text(rel):
    draft, live, _base = MAP[rel]
    if draft == 'v2' and live.endswith('thesis_v2.tex'):
        if 'split' not in _v2_split_cache:
            t = read(os.path.join(WS, live))
            _v2_split_cache['split'] = split_v2(t) if t is not None else {}
        return _v2_split_cache['split'].get(rel)
    return read(os.path.join(WS, live))


def classify(only=None):
    """-> [(rel, draft, moved_upstream, v5_edited, base_missing)]"""
    rows = []
    for rel, (draft, _live, base) in MAP.items():
        if only and rel != only:
            continue
        b = read(os.path.join(MATERIALS, base))
        o = read(os.path.join(V5, rel))
        t = theirs_text(rel)
        rows.append((rel, draft, t != b, o != b, b is None))
    return rows


def load_state():
    st = read(STATE)
    if st:
        try:
            return json.loads(st)
        except ValueError:
            pass
    init = read(INIT_STATE)
    if init:
        try:
            j = json.loads(init)
            return {'absorbed_at': j.get('initialised_at', '?') + ' (init, v5.0)',
                    **{d: j['built_on'][d]['heading'] for d in ('v2', 'v3', 'v4') if d in j.get('built_on', {})}}
        except (ValueError, KeyError):
            pass
    return {}


def save_state(drafts):
    st = load_state()
    st['absorbed_at'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    for d in drafts:
        st[d] = draft_version(d)
    with open(STATE, 'w', encoding='utf-8') as f:
        json.dump(st, f, indent=2, ensure_ascii=False)
        f.write('\n')
    return st


def cmd_versions(_a):
    st = load_state()
    print(f'v5 last absorbed: {st.get("absorbed_at", "?")}')
    for d in ('v2', 'v3', 'v4'):
        now = draft_version(d)
        was = st.get(d, '?')
        flag = '' if now == was else '   <-- moved'
        print(f'  {d}: absorbed  {was}')
        print(f'      now       {now}{flag}')
    return 0


def cmd_status(_a):
    cmd_versions(_a)
    print()
    rows = classify()
    w = max(len(r[0]) for r in rows)
    print(f'{"v5 file".ljust(w)}  from  legacy moved  v5 edited')
    print('-' * (w + 32))
    for rel, draft, moved, edited, missing in rows:
        print(f'{rel.ljust(w)}  {draft:<4}  {"YES" if moved else "  -":>12}  {"YES" if edited else "  -":>9}'
              + ('   (no base in inherited/materials)' if missing else ''))
    stale = [r for r in rows if r[2]]
    print()
    if not stale:
        print('nothing to absorb: no legacy source has moved since v5 last absorbed it.')
        return 0
    print(f'{len(stale)} file(s) moved in a legacy draft:')
    for rel, draft, _m, edited, _x in stale:
        print(f'  {rel}  [{draft}]  -> ' + ('three-way merge needed (v5 edited it too)' if edited
                                             else 'fast-forward: the legacy change can be taken wholesale'))
    print('\nInspect with:  python3 tools/absorb.py diff <v5 file>')
    print('Absorb with:   python3 tools/absorb.py merge [<v5 file>]   then record a new v5.N in CHANGELOG.md')
    return 1


def cmd_diff(a):
    rows = [r for r in classify(a.file) if (a.file is None and r[2]) or r[0] == a.file]
    if not rows:
        print('nothing to diff.')
        return 0
    for rel, draft, _m, _e, _x in rows:
        base = os.path.join(MATERIALS, MAP[rel][2])
        t = theirs_text(rel)
        with tempfile.NamedTemporaryFile('w', suffix='.tex', delete=False, encoding='utf-8') as tf:
            tf.write(t or '')
            tp = tf.name
        try:
            subprocess.run(['diff', '-u', '--label', f'materials/{MAP[rel][2]}  (base)',
                            '--label', f'{MAP[rel][1]}  (live {draft})', base, tp])
        finally:
            os.unlink(tp)
    return 0


def cmd_merge(a):
    rows = [r for r in classify(a.file) if r[2] and (a.file is None or r[0] == a.file)]
    if not rows:
        print('nothing to merge: no legacy source has moved.')
        return 0
    conflicts, merged, drafts = 0, [], set()
    for rel, draft, _m, edited, missing in rows:
        ours = os.path.join(V5, rel)
        base = os.path.join(MATERIALS, MAP[rel][2])
        t = theirs_text(rel)
        if t is None:
            print(f'skipped  {rel}  [{draft}]: the legacy file is gone (archived?) -- decide by hand')
            continue
        if missing:
            print(f'skipped  {rel}  [{draft}]: no base in inherited/materials -- copy one in by hand first')
            continue
        if a.dry_run:
            print(f'would merge {rel}  [{draft}, v5 {"edited" if edited else "clean"}]')
            continue
        with tempfile.NamedTemporaryFile('w', suffix='.tex', delete=False, encoding='utf-8') as tf:
            tf.write(t)
            tp = tf.name
        try:
            r = subprocess.run(['git', 'merge-file', '-L', 'v5', '-L', 'base', '-L', draft, ours, base, tp])
            if r.returncode < 0 or r.returncode > 100:
                sys.exit(f'git merge-file failed on {rel}')
            if r.returncode:
                conflicts += 1
                print(f'CONFLICT ({r.returncode} hunk(s)) in {rel} -- resolve the markers in v5, then re-run merge')
            else:
                os.makedirs(os.path.dirname(base), exist_ok=True)
                shutil.copyfile(tp, base)          # advance the base only on a clean merge
                merged.append(rel)
                drafts.add(draft)
                print(f'merged   {rel}  [{draft}]')
        finally:
            os.unlink(tp)
    if merged and 'v2' in drafts:                  # keep the monolith copy in step with its split base
        t = read(os.path.join(WS, 'v2', 'thesis_v2.tex'))
        if t is not None:
            with open(os.path.join(MATERIALS, 'v2', 'thesis_v2.tex'), 'w', encoding='utf-8') as f:
                f.write(t)
    if conflicts:
        print(f'\n{conflicts} file(s) left with conflict markers in v5; their base was NOT advanced.')
        print('Resolve them, then re-run:  python3 tools/absorb.py merge')
    elif merged:
        st = save_state(drafts)
        print(f'\nBase advanced for {", ".join(sorted(drafts))}; ABSORB_STATE.json stamped {st["absorbed_at"]}.')
        print('Now: python3 tools/check.py, python3 tools/make_release_v5.py --dry-run, a new v5.N entry in CHANGELOG.md, '
              'and the owner\'s INBOX row marked 🔀 v5.N.')
    return 1 if conflicts else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('status')
    sub.add_parser('versions')
    d = sub.add_parser('diff')
    d.add_argument('file', nargs='?', help='a v5 file, e.g. chapters/06_results.tex')
    m = sub.add_parser('merge')
    m.add_argument('file', nargs='?')
    m.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()
    return {'status': cmd_status, 'versions': cmd_versions, 'diff': cmd_diff, 'merge': cmd_merge}[a.cmd](a)


if __name__ == '__main__':
    sys.exit(main())
