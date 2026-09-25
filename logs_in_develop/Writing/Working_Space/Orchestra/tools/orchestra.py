#!/usr/bin/env python3
"""orchestra.py -- the Orchestra's helper (Working_Space/Orchestra).

The Orchestra handles MINOR and CROSS-LINKED changes to the thesis drafts v2 / v3 / v4, distributes
bigger jobs as TODO lists to the owner chats, and runs a RELEASE when the author asks. Since
2026-09-25 (the Advance Orchestra, job O003) the thesis lives in Working_Space/v5 -- the aggregate the
Orchestra edits directly (runbook v5/README.md); v2 / v3 / v4 are legacy sources, kept and used less.
`status` shows v5 and what it carries, `bump v5` writes the next `## v5.N`, kinds `advance` (an edit
in v5) and `absorb` (a legacy change merged into v5) name the two Advance job types. This tool does
the bookkeeping that is easy to get wrong by hand:

    python3 tools/orchestra.py status [--write] [--full]      the snapshot: draft versions, sync chain, open INBOX rows,
                                                              last release, last job; --write also writes STATE.md
    python3 tools/orchestra.py new-job "<title>" --kind edit|todo|release|sync|check
                                                              jobs/O###_<stamp>_<slug>.md from the template + an open row in CHANGELOG.md
    python3 tools/orchestra.py bump v3 --job O### --title "<what>" [--body-file f] [--dry-run]
                                                              the next revision heading in <draft>/CHANGELOG.md (v3/v4: letter suffix,
                                                              v2: next number -- v3's sync_v2.py cannot read letters), entry inserted newest-first
    python3 tools/orchestra.py note --to v3 --job O### --topic <slug> --item "<INBOX text>" [--body-file f | --from-todo todo/<file>]
                                                              cross_draft/to_v3/FROM_Orchestra_<date>_O###_<topic>.md + the INBOX row (newest first)
    python3 tools/orchestra.py new-todo "<title>" --job O### --to v2,v3,v4[,RELEASE]
                                                              todo/TODO_<stamp>_O###_<slug>.md, one section per target
    python3 tools/orchestra.py close-job O### --summary "<one line>" [--notes v2,v3] [--release <build folder>]
                                                              the CHANGELOG row closed with the versions AFTER the job; job file stamped

Python 3 stdlib only. Never compiles, never commits, never edits a draft's .tex or .bib (that is the
Orchestra's own, hand-made work, recorded through `bump` and `note`). Paths are resolved from this
file, so it runs from any working directory.
"""
import argparse
import datetime as _dt
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ORCH = os.path.dirname(HERE)                      # Working_Space/Orchestra
WS = os.path.dirname(ORCH)                        # Working_Space
WRITING = os.path.dirname(WS)                     # logs_in_develop/Writing
DRAFTS = {d: os.path.join(WS, d) for d in ('v2', 'v3', 'v4', 'v5')}
V5 = os.path.join(WS, 'v5')
RELEASE = os.path.join(WS, 'RELEASE')
RELEASE_OUT = os.path.join(RELEASE, 'output')
CROSS = os.path.join(WS, 'cross_draft')
INBOX = os.path.join(CROSS, 'INBOX.md')
CHANGELOG = os.path.join(ORCH, 'CHANGELOG.md')
STATE = os.path.join(ORCH, 'STATE.md')
JOBS = os.path.join(ORCH, 'jobs')
TODO = os.path.join(ORCH, 'todo')
TEMPLATES = os.path.join(ORCH, 'templates')

RE_VERSION = re.compile(r'^## v(\d+)\.(\d+)([a-z]?)\b(.*)$')
RE_SHORT = re.compile(r'^(v\d+\.\d+[a-z]?)')
RE_JOB = re.compile(r'^O(\d{3})_')
RE_RELEASE_DIR = re.compile(r'^\d{8}_\d{6}_thesis_release_')
TARGETS = ('v2', 'v3', 'v4')                       # the legacy owner chats: cross notes go to them
ALL = ('v2', 'v3', 'v4', 'v5')                     # every draft with a CHANGELOG.md; v5 = the aggregate
INBOX_TARGETS = ('v2', 'v3', 'v4', 'v5')           # sections of cross_draft/INBOX.md
KINDS = ('edit', 'todo', 'release', 'sync', 'check', 'init', 'advance', 'absorb')


# ------------------------------------------------------------------------------------------ io
def read(path):
    with open(path, encoding='utf-8') as f:
        return f.read()


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(text)


def now():
    return _dt.datetime.now()


def stamp(t=None):
    return (t or now()).strftime('%Y%m%d_%H%M')


def human(t=None):
    return (t or now()).strftime('%Y-%m-%d %H:%M')


def today(t=None):
    return (t or now()).strftime('%Y-%m-%d')


def slugify(text, limit=48):
    s = re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_').lower()
    if len(s) > limit:
        s = s[:limit]
        s = s[:s.rfind('_')] if '_' in s else s
    return s.rstrip('_') or 'job'


def fill(template, mapping):
    out = template
    for k, v in mapping.items():
        out = out.replace('{{' + k + '}}', str(v))
    left = [k for k in re.findall(r'\{\{([A-Z0-9_]+)\}\}', out) if k != 'CLOSED']
    if left:
        print(f'  note: unfilled placeholders left for the hand: {sorted(set(left))}')
    return out


def rel(path, start):
    return os.path.relpath(path, start).replace(os.sep, '/')


# ------------------------------------------------------------------------------------ drafts
def draft_version(name):
    """Highest '## vN.M[a]' heading of <draft>/CHANGELOG.md.

    Returns dict(short='v3.100b', key=(3,100,'b'), text='v3.100b -- date . title', letter=True,
    out_of_order=False). Highest, not first: the owners' tools sort the same way."""
    path = os.path.join(DRAFTS[name], 'CHANGELOG.md')
    best, first = None, None
    try:
        for ln in read(path).split('\n'):
            m = RE_VERSION.match(ln)
            if not m:
                continue
            key = (int(m.group(1)), int(m.group(2)), m.group(3))
            text = ln[3:].strip()
            text = re.sub(r'\s*→.*$', '', text)          # drop the link to the individual changelog
            if first is None:
                first = key
            if best is None or key > best[0]:
                best = (key, text)
    except OSError:
        return dict(short='v?', key=None, text='unknown (no CHANGELOG.md)', letter=False, out_of_order=False)
    if best is None:
        return dict(short='v?', key=None, text='unknown', letter=False, out_of_order=False)
    k = best[0]
    return dict(short=f'v{k[0]}.{k[1]}{k[2]}', key=k, text=best[1], letter=bool(k[2]),
                out_of_order=(first != k))


def next_version(name):
    """v3/v4: the next letter of the current revision (v3.100b -> v3.100c, v4.2 -> v4.2a).
    v2: the next number (v2.28 -> v2.29) -- v3/tools/sync_v2.py reads no letter suffix.
    v5: the next number (v5.0 -> v5.1) -- one revision per working pass, the author's scheme."""
    cur = draft_version(name)
    if cur['key'] is None:
        raise SystemExit(f'{name}: no version heading found in its CHANGELOG.md')
    major, minor, letter = cur['key']
    if name in ('v2', 'v5'):
        return f'v{major}.{minor + 1}', cur
    nxt = 'a' if not letter else chr(ord(letter) + 1)
    if nxt > 'z':
        raise SystemExit(f'{name}: letter suffix exhausted at {cur["short"]}; the owner must bump the number')
    return f'v{major}.{minor}{nxt}', cur


def short_of(text):
    m = RE_SHORT.match(text or '')
    return m.group(1) if m else (text or 'unknown')[:12]


def sync_state(name):
    path = os.path.join(DRAFTS[name], 'inherited', 'SYNC_STATE.json')
    try:
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def absorb_state():
    """What v5 carries of v2 / v3 / v4: ABSORB_STATE.json after an absorb, else the INIT_STATE.json of v5.0."""
    for fn in ('ABSORB_STATE.json', 'INIT_STATE.json'):
        try:
            with open(os.path.join(V5, 'inherited', fn), encoding='utf-8') as f:
                j = json.load(f)
        except (OSError, ValueError):
            continue
        if fn == 'INIT_STATE.json':
            b = j.get('built_on', {})
            return dict(absorbed_at=j.get('initialised_at', '?') + ' (init, v5.0)',
                        **{d: b[d]['heading'] for d in TARGETS if d in b})
        return j
    return {}


def v5_builds():
    try:
        dirs = sorted(d for d in os.listdir(RELEASE_OUT)
                      if RE_RELEASE_DIR.match(d) and '_ORCH_' in d and os.path.isdir(os.path.join(RELEASE_OUT, d)))
    except OSError:
        dirs = []
    return (dirs[-1] if dirs else None), len(dirs)


def sync_chain():
    """What v3 carries of v2, what v4 carries of v3 (and of v2 through it), against the live versions."""
    v2, v3, v4 = (draft_version(d) for d in TARGETS)
    s3, s4 = sync_state('v3'), sync_state('v4')
    v2_in_v3 = short_of(s3.get('v2_version'))
    v3_in_v4 = short_of(s4.get('v3_version'))
    v2_in_v4 = short_of(s4.get('v2_version_via_v3'))
    return dict(
        v2=v2, v3=v3, v4=v4, v5=draft_version('v5'),
        v2_in_v3=v2_in_v3, v3_lags=(v2_in_v3 != v2['short']), v3_synced=s3.get('synced_at', '?'),
        v3_in_v4=v3_in_v4, v2_in_v4=v2_in_v4, v4_lags=(v3_in_v4 != v3['short']), v4_synced=s4.get('synced_at', '?'),
    )


# ------------------------------------------------------------------------------------- inbox
def inbox_sections():
    """{'v2': (header_line_index, [row dicts]), ...} -- rows in file order (newest first)."""
    lines = read(INBOX).split('\n')
    out = {}
    cur = None
    for i, ln in enumerate(lines):
        m = re.match(r'^## → (v\d)\s*$', ln)
        if m:
            cur = m.group(1)
            out[cur] = dict(heading=i, header=None, rows=[])
            continue
        if cur is None:
            continue
        if ln.startswith('| :--'):
            out[cur]['header'] = i
            continue
        if ln.startswith('| ') and not ln.startswith('| status') and not ln.startswith('| :--'):
            cells = [c.strip() for c in ln.strip().strip('|').split('|')]
            if len(cells) >= 3:
                out[cur]['rows'].append(dict(line=i, status=cells[0], src=cells[1], item=cells[2],
                                             note=cells[3] if len(cells) > 3 else ''))
    return lines, out


def inbox_open(full=False):
    _, secs = inbox_sections()
    res = {}
    for t in INBOX_TARGETS:
        rows = [r for r in secs.get(t, dict(rows=[]))['rows'] if r['status'].startswith('⏳')]
        res[t] = rows
    return res


def inbox_insert(target, row_line):
    lines, secs = inbox_sections()
    if target not in secs or secs[target]['header'] is None:
        raise SystemExit(f'INBOX.md: no table header under "## → {target}"')
    at = secs[target]['header'] + 1
    lines.insert(at, row_line)
    write(INBOX, '\n'.join(lines))


# ----------------------------------------------------------------------------------- release
def last_release():
    try:
        dirs = sorted(d for d in os.listdir(RELEASE_OUT)
                      if RE_RELEASE_DIR.match(d) and os.path.isdir(os.path.join(RELEASE_OUT, d)))
    except OSError:
        dirs = []
    top = None
    try:
        for ln in read(os.path.join(RELEASE, 'CHANGELOG.md')).split('\n'):
            if ln.startswith('## '):
                top = ln[3:].strip()
                break
    except OSError:
        pass
    return (dirs[-1] if dirs else None), top, len(dirs)


# --------------------------------------------------------------------------------- orchestra
def jobs_list():
    try:
        files = sorted(f for f in os.listdir(JOBS) if RE_JOB.match(f) and f.endswith('.md'))
    except OSError:
        files = []
    return files


def next_job_id():
    n = 0
    for f in jobs_list():
        n = max(n, int(RE_JOB.match(f).group(1)))
    return f'O{n + 1:03d}'


def job_file(job):
    for f in jobs_list():
        if f.startswith(job + '_'):
            return os.path.join(JOBS, f)
    raise SystemExit(f'{job}: no file jobs/{job}_*.md')


def changelog_rows():
    lines = read(CHANGELOG).split('\n')
    header = None
    rows = []
    for i, ln in enumerate(lines):
        if ln.startswith('| :--') and header is None:
            header = i
        elif ln.startswith('| [O'):
            rows.append((i, ln))
    return lines, header, rows


def changelog_insert(row):
    lines, header, _ = changelog_rows()
    if header is None:
        raise SystemExit('Orchestra CHANGELOG.md: table header not found')
    lines.insert(header + 1, row)
    write(CHANGELOG, '\n'.join(lines))


def git_dirty():
    try:
        out = subprocess.run(['git', 'status', '--short', '--', WRITING], capture_output=True, text=True,
                             cwd=WRITING, timeout=20)
        if out.returncode != 0:
            return None
        return len([ln for ln in out.stdout.split('\n') if ln.strip()])
    except Exception:
        return None


def versions_cell():
    return ' · '.join(draft_version(d)['short'] for d in ALL)


# ------------------------------------------------------------------------------------ status
def cmd_status(args):
    t = now()
    ch = sync_chain()
    opened = inbox_open()
    rel_dir, rel_top, n_builds = last_release()
    files = jobs_list()
    _, _, rows = changelog_rows() if os.path.isfile(CHANGELOG) else ([], None, [])
    open_jobs = [ln for _, ln in rows if '⏳' in ln]
    dirty = git_dirty()

    L = []
    L.append(f'# Orchestra STATE · {human(t)}')
    L.append('')
    L.append('Written by `tools/orchestra.py status --write`; a snapshot, not a record. The record is `CHANGELOG.md` and `jobs/`.')
    L.append('')
    L.append('## Drafts (highest `## vN.M` heading of each CHANGELOG.md)')
    L.append('')
    L.append('| draft | revision | heading |')
    L.append('| :-- | :-- | :-- |')
    for d in ALL:
        v = ch[d]
        flag = ''
        if v['out_of_order']:
            flag += ' ⚠ CHANGELOG not in descending order'
        if d == 'v2' and v['letter']:
            flag += ' ⚠ v2 has a letter suffix: v3/tools/sync_v2.py cannot read it'
        if d == 'v5':
            flag += ' — **the thesis** (the aggregate; Advance Orchestra since 2026-09-25)'
        L.append(f'| {d} | **{v["short"]}** | {v["text"]}{flag} |')
    L.append('')
    ab = absorb_state()
    moved = [d for d in TARGETS if short_of(ab.get(d, '')) != ch[d]['short']]
    L.append('## v5 — the aggregate (Advance Orchestra, since 2026-09-25; runbook `v5/README.md`)')
    L.append('')
    L.append(f'- v5 carries (last absorbed {ab.get("absorbed_at", "?")}): '
             + ' · '.join(f'{d} **{short_of(ab.get(d, "?"))}**' for d in TARGETS)
             + f'; legacy drafts moved since: **{", ".join(moved) if moved else "none"}**. '
             f'File view: `cd v5 && python3 tools/absorb.py status`')
    L.append('')
    L.append('## Sync chain (one-way v2 → v3 → v4; a RELEASE reads the LIVE files, so a lag matters only for bundles and inherited copies)')
    L.append('')
    l3 = 'LAGS — v2 moved' if ch['v3_lags'] else 'in step'
    l4 = 'LAGS — v3 moved' if ch['v4_lags'] else 'in step'
    L.append(f'- v3 carries **{ch["v2_in_v3"]}** (synced {ch["v3_synced"]}); v2 is at {ch["v2"]["short"]} → **{l3}**. '
             f'File view: `cd v3 && python3 tools/sync_v2.py status`')
    L.append(f'- v4 carries **{ch["v3_in_v4"]}** / {ch["v2_in_v4"]} (synced {ch["v4_synced"]}); v3 is at {ch["v3"]["short"]} → **{l4}**. '
             f'File view: `cd v4 && python3 tools/sync_v3.py status`')
    L.append('')
    L.append('## cross_draft/INBOX.md — open rows (⏳)')
    L.append('')
    for tgt in INBOX_TARGETS:
        rows_t = opened[tgt]
        L.append(f'- **→ {tgt}: {len(rows_t)} open**')
        for r in rows_t:
            item = re.sub(r'\*\*', '', r['item'])
            if not args.full:
                item = item[:150] + ('…' if len(item) > 150 else '')
            L.append(f'  - [{r["src"]}] {item}')
    L.append('')
    L.append('## RELEASE')
    L.append('')
    L.append(f'- builds kept in `RELEASE/output/`: {n_builds}; newest: `{rel_dir or "none"}`')
    o_dir, o_n = v5_builds()
    L.append(f'- of these built by the Orchestra from v5 (`_ORCH_`): {o_n}; newest: `{o_dir or "none"}` '
             f'(`cd v5 && python3 tools/make_release_v5.py`)')
    L.append(f'- `RELEASE/CHANGELOG.md` top row: {rel_top or "none"}')
    L.append('')
    L.append('## Orchestra')
    L.append('')
    L.append(f'- jobs on file: {len(files)}; open rows in `CHANGELOG.md`: {len(open_jobs)}; next id: **{next_job_id()}**')
    if rows:
        L.append(f'- last row: {rows[0][1]}')
    L.append(f'- git: {dirty if dirty is not None else "?"} modified/untracked path(s) under `logs_in_develop/Writing` '
             f'(the Orchestra never commits)')
    text = '\n'.join(L) + '\n'
    print(text)
    if args.write:
        write(STATE, text)
        print(f'-> written {rel(STATE, os.getcwd())}')


# ----------------------------------------------------------------------------------- new-job
def cmd_new_job(args):
    if args.kind not in KINDS:
        raise SystemExit(f'--kind must be one of {KINDS}')
    t = now()
    job = next_job_id()
    slug = slugify(args.title)
    path = os.path.join(JOBS, f'{job}_{stamp(t)}_{slug}.md')
    if os.path.exists(path):
        raise SystemExit(f'exists: {path}')
    ch = sync_chain()
    rel_dir, _, _ = last_release()
    tpl = read(os.path.join(TEMPLATES, 'JOB_template.md'))
    sync_line = (f'v3 carries {ch["v2_in_v3"]} ({"v2 moved" if ch["v3_lags"] else "in step"}); '
                 f'v4 carries {ch["v3_in_v4"]} / {ch["v2_in_v4"]} ({"v3 moved" if ch["v4_lags"] else "in step"})')
    opened = inbox_open()
    inbox_line = ', '.join(f'→ {tg}: {len(opened[tg])}' for tg in INBOX_TARGETS)
    text = fill(tpl, dict(JOB=job, TITLE=args.title, OPENED=human(t), KIND=args.kind,
                          V2=ch['v2']['short'], V3=ch['v3']['short'], V4=ch['v4']['short'], V5=ch['v5']['short'],
                          RELEASE_LAST=rel_dir or 'none', SYNC=sync_line, INBOX_OPEN=inbox_line))
    write(path, text)
    row = (f'| [{job}](jobs/{os.path.basename(path)}) | {human(t)} → ⏳ | {args.kind} | '
           f'{versions_cell()} | ⏳ open | {args.title.replace("|", "/")} | — | — |')
    changelog_insert(row)
    print(f'{job} opened: {rel(path, os.getcwd())}')
    print(f'CHANGELOG row added (open). Close with:  python3 tools/orchestra.py close-job {job} --summary "..."')


# --------------------------------------------------------------------------------------- bump
DEFAULT_BODY = """- **Changed:** `<file>` (l. <n>–<m> / `\\label{{...}}`): <what, in one sentence per file>.
- **Why:** the author's request (job {job}): "<the author's words>".
- **Checked:** `tools/check.py` → <result>; bundle → <rebuilt / not rebuilt (v2 has none)>; `RELEASE --dry-run` → <result>. **Not compiled.**
- **Left to {draft}:** README / CROSS_STATE / SYNC_STATE untouched — refresh them at your next pass; INBOX row → {draft} (Orchestra {job}).
- Signed: Orchestra (Claude Fable 5.1, Claude Code), {job} · {date}."""

DEFAULT_BODY_V5 = """- **Changed:** `<file>` (l. <n>–<m> / `\\label{{...}}`): <what, in one sentence per file>.
- **Why:** the author's request (job {job}): "<the author's words>".
- **Checked:** `tools/check.py` → <result>; `tools/make_release_v5.py --dry-run` → <result>; `tools/absorb.py status` → <result>. **Not compiled.**
- **INBOX:** <rows resolved here, marked 🔀 {version} — or none>. **Release:** <none / the build folder>.
- Signed: Orchestra (Claude Fable 5.1, Claude Code), {job} · {date}."""


def cmd_bump(args):
    draft = args.draft
    if draft not in DRAFTS:
        raise SystemExit('draft must be v2, v3, v4 or v5')
    nxt, cur = next_version(draft)
    jf = job_file(args.job)
    link = args.link or f'../Orchestra/jobs/{os.path.basename(jf)}'
    heading = f'## {nxt} — {today()} · {args.title} (Orchestra {args.job}) → [`{link}`]({link})'
    if args.body_file:
        body = read(args.body_file)
    elif draft == 'v5':
        body = DEFAULT_BODY_V5.format(job=args.job, version=nxt, date=today())
    else:
        body = DEFAULT_BODY.format(job=args.job, draft=draft, date=today())
    entry = heading + '\n\n' + body.rstrip('\n') + '\n\n'
    print(f'{draft}: {cur["short"]} -> {nxt}')
    print(entry)
    if args.dry_run:
        print('(dry run: nothing written)')
        return
    path = os.path.join(DRAFTS[draft], 'CHANGELOG.md')
    lines = read(path).split('\n')
    at = None
    for i, ln in enumerate(lines):
        if RE_VERSION.match(ln):
            at = i
            break
    if at is None:
        raise SystemExit(f'{path}: no version heading to insert before')
    lines[at:at] = entry.split('\n')
    write(path, '\n'.join(lines))
    print(f'-> inserted before the former top entry in {rel(path, os.getcwd())}')
    if draft == 'v2':
        print('   v2 rule: the next NUMBER was used (v3/tools/sync_v2.py reads no letter suffix).')


# --------------------------------------------------------------------------------------- note
def todo_section(path, target):
    text = read(path)
    m = re.search(rf'^## → {re.escape(target)}\b.*?$(.*?)(?=^## |\Z)', text, re.M | re.S)
    if not m:
        raise SystemExit(f'{path}: no "## → {target}" section')
    body = [ln for ln in m.group(1).split('\n') if not ln.startswith('Owner: the ')]   # the delivery instruction is the Orchestra's, not the owner's
    return '\n'.join(body).strip('\n')


def cmd_note(args):
    if args.to not in TARGETS:
        raise SystemExit('--to must be v2, v3 or v4')
    t = now()
    jf = job_file(args.job)
    topic = slugify(args.topic, 40)
    fname = f'FROM_Orchestra_{t.strftime("%Y%m%d")}_{args.job}_{topic}.md'
    path = os.path.join(CROSS, f'to_{args.to}', fname)
    if os.path.exists(path):
        raise SystemExit(f'exists: {path}')
    if args.body_file:
        body = read(args.body_file).rstrip('\n')
    elif args.from_todo:
        body = todo_section(args.from_todo, args.to)
        body += f'\n\nThe master list (all drafts, cross links, status): `../../Orchestra/{rel(args.from_todo, ORCH)}`.'
    else:
        body = '<what the Orchestra changed in your files, file by file / or the TODO items>'
    tpl = read(os.path.join(TEMPLATES, 'CROSSNOTE_template.md'))
    text = fill(tpl, dict(TO=args.to, DATE=today(t), JOB=args.job, TOPIC=args.topic,
                          JOB_LINK=f'../../Orchestra/jobs/{os.path.basename(jf)}',
                          BODY=body, VERSIONS=versions_cell()))
    write(path, text)
    item = args.item.replace('|', '/').strip()
    row = f'| ⏳ | Orchestra {args.job} · {today(t)} | **{item}** | [`{fname}`](to_{args.to}/{fname}) |'
    inbox_insert(args.to, row)
    print(f'note written: {rel(path, os.getcwd())}')
    print(f'INBOX row inserted under "## → {args.to}" (newest first).')


# ----------------------------------------------------------------------------------- new-todo
def cmd_new_todo(args):
    t = now()
    jf = job_file(args.job)
    targets = [x.strip() for x in args.to.split(',') if x.strip()]
    for x in targets:
        if x not in ALL + ('RELEASE',):
            raise SystemExit(f'--to: unknown target {x}')
    slug = slugify(args.title)
    path = os.path.join(TODO, f'TODO_{stamp(t)}_{args.job}_{slug}.md')
    if os.path.exists(path):
        raise SystemExit(f'exists: {path}')
    tpl = read(os.path.join(TEMPLATES, 'TODO_template.md'))
    sections = []
    for x in targets:
        sections.append(f'## → {x}\n\n'
                        f'Owner: the {x} chat. Delivered as `cross_draft/to_{x}/FROM_Orchestra_<date>_{args.job}_todo.md` + INBOX row '
                        f'(`python3 tools/orchestra.py note --to {x} --job {args.job} --topic todo --from-todo <this file> --item "..."`).\n\n'
                        f'- [ ] <item — file / label / what / why / evidence>\n'
                        f'- [ ] <item>\n'
                        if x != 'RELEASE' else
                        f'## → RELEASE\n\nRun by the Orchestra, only when the author asks.\n\n- [ ] <item>\n')
    text = fill(tpl, dict(TITLE=args.title, JOB=args.job, OPENED=human(t), VERSIONS=versions_cell(),
                          JOB_LINK=f'../jobs/{os.path.basename(jf)}', SECTIONS='\n'.join(sections),
                          TARGETS=', '.join(targets)))
    write(path, text)
    print(f'TODO master written: {rel(path, os.getcwd())}')
    print('Fill the sections, then deliver each with `note --from-todo` (one cross note + INBOX row per target).')


# ---------------------------------------------------------------------------------- close-job
def cmd_close_job(args):
    job = args.job
    lines, header, rows = changelog_rows()
    hit = None
    for i, ln in rows:
        if ln.startswith(f'| [{job}]'):
            hit = i
            break
    if hit is None:
        raise SystemExit(f'{job}: no row in CHANGELOG.md')
    cells = [c.strip() for c in lines[hit].strip().strip('|').split('|')]
    # cells: link, opened → closed, kind, versions, status, what, notes, release
    while len(cells) < 8:
        cells.append('—')
    opened = cells[1].split('→')[0].strip()
    t = now()
    cells[1] = f'{opened} → {human(t)}'
    cells[3] = versions_cell()
    cells[4] = '✅'
    cells[5] = args.summary.replace('|', '/').strip()
    cells[6] = args.notes.replace('|', '/').strip() if args.notes else '—'
    cells[7] = f'`{args.release}`' if args.release else '—'
    lines[hit] = '| ' + ' | '.join(cells) + ' |'
    write(CHANGELOG, '\n'.join(lines))
    jf = job_file(job)
    text = read(jf)
    closed = (f'{human(t)} · versions after: {versions_cell()} · notes left: {cells[6]} · release: {cells[7]} · '
              f'signed: Orchestra (Claude Fable 5.1, Claude Code)')
    if '{{CLOSED}}' in text:
        text = text.replace('{{CLOSED}}', closed)
    else:
        text = text.rstrip('\n') + '\n\n## Closed\n\n' + closed + '\n'
    write(jf, text)
    print(f'{job} closed: row updated in CHANGELOG.md; job file stamped.')


# --------------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)

    p = sub.add_parser('status', help='draft versions, sync chain, open INBOX rows, last release, last job')
    p.add_argument('--write', action='store_true', help='also write STATE.md')
    p.add_argument('--full', action='store_true', help='print the full INBOX items, not the first 150 characters')
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser('new-job', help='open a job: jobs/O###_<stamp>_<slug>.md + an open CHANGELOG row')
    p.add_argument('title')
    p.add_argument('--kind', default='edit', help='|'.join(KINDS))
    p.set_defaults(fn=cmd_new_job)

    p = sub.add_parser('bump', help="insert the next revision entry into a draft's CHANGELOG.md")
    p.add_argument('draft', help='v2 | v3 | v4 | v5 (v5: the next number, one revision per pass)')
    p.add_argument('--job', required=True)
    p.add_argument('--title', required=True, help='the heading text after the date')
    p.add_argument('--body-file', help='markdown body of the entry (default: a skeleton to complete by hand)')
    p.add_argument('--link', help='what the heading links (default: the Orchestra job file; v5: e.g. changelogs/v5.1_<date>_<slug>.md)')
    p.add_argument('--dry-run', action='store_true')
    p.set_defaults(fn=cmd_bump)

    p = sub.add_parser('note', help='write cross_draft/to_<draft>/FROM_Orchestra_... and its INBOX row')
    p.add_argument('--to', required=True)
    p.add_argument('--job', required=True)
    p.add_argument('--topic', required=True, help='short slug for the file name')
    p.add_argument('--item', required=True, help='the INBOX item text (bold, one or two sentences)')
    p.add_argument('--body-file', help='markdown body of the note')
    p.add_argument('--from-todo', help='take the body from the "## → <draft>" section of a todo/ master file')
    p.set_defaults(fn=cmd_note)

    p = sub.add_parser('new-todo', help='a TODO master file with one section per target draft')
    p.add_argument('title')
    p.add_argument('--job', required=True)
    p.add_argument('--to', default='v2,v3,v4', help='comma list of v2,v3,v4,v5,RELEASE')
    p.set_defaults(fn=cmd_new_todo)

    p = sub.add_parser('close-job', help='close the CHANGELOG row with the versions after the job')
    p.add_argument('job')
    p.add_argument('--summary', required=True, help='one line: what was done')
    p.add_argument('--notes', help='which inboxes got a note, e.g. "v3,v4" or "to_v3 ×2"')
    p.add_argument('--release', help='the release build folder name, if one was built')
    p.set_defaults(fn=cmd_close_job)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == '__main__':
    main()
