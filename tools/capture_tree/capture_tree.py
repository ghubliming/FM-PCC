#!/usr/bin/env python3
"""capture_tree.py — dump the file tree of one or more paths into ONE small text file.

Made for the cluster login node ("Slurm lobby"): you want to know what an old run
folder actually contains before deciding whether it is worth downloading, and you
want the answer in a single file you can scp in a second.

The trick that keeps the capture small is **pattern collapsing**: 1080 files named
`rollout_0_stats.json … rollout_1079_stats.json` are printed as one line

    rollout_<N>_stats.json                 x1080   1.4 MiB   [0..1079]

so a tree with 50k files still fits in a few dozen KB. Same for numbered
directories. Nothing is read, opened, deleted or modified — `os.scandir` only.

Stdlib only, no conda env needed (`python3 capture_tree.py …` is enough).

Usage
-----
    # one path
    python3 tools/capture_tree/capture_tree.py logs/d3il_visual_aligning_baseline

    # several paths in one capture file
    python3 tools/capture_tree/capture_tree.py \
        logs/d3il_visual_aligning_baseline \
        logs/aligning-d3il-visual/plans \
        -o tree_capture.txt

    # shallow overview first (cheap on a huge tree)
    python3 tools/capture_tree/capture_tree.py logs -d 3 --dirs-only

    # machine-readable instead of the text tree
    python3 tools/capture_tree/capture_tree.py logs/... --format json -o tree.json
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
import time
from datetime import datetime

# Numbered-sibling detector: `rollout_17_stats.json` → `rollout_<N>_stats.json`,
# `seed_42` → `seed_<N>`, `snapshot_20260812_150355` is left alone (two groups,
# but the second is not a plain counter — collapsing only fires on >= MIN_GROUP
# members anyway, so an accidental match stays harmless).
_NUM_RE = re.compile(r'\d+')

MIN_GROUP = 3          # fewer than this many siblings: print them individually
DEFAULT_SKIP = ('.git', '__pycache__', '.ipynb_checkpoints', '.mypy_cache')


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def human(size):
    value = float(size)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.0f} {unit}' if unit == 'B' else f'{value:.1f} {unit}'
        value /= 1024
    return f'{value:.1f} TiB'


def stamp(epoch):
    if not epoch:
        return ''
    return datetime.fromtimestamp(epoch).strftime('%Y-%m-%d %H:%M')


def pattern_of(name):
    """`rollout_17_stats.json` → (`rollout_<N>_stats.json`, [17]); no digits → None."""
    numbers = _NUM_RE.findall(name)
    if not numbers:
        return None, []
    return _NUM_RE.sub('<N>', name), [int(n) for n in numbers]


def group_by_pattern(entries):
    """Split entries into (collapsed groups, singles).

    `entries` is a list of dicts with at least `name` and `size`. A group is
    every entry sharing one digit-masked name, when there are >= MIN_GROUP of
    them. Order is preserved (sorted by name upstream).
    """
    buckets = {}
    order = []
    for entry in entries:
        key, numbers = pattern_of(entry['name'])
        if key is None:
            key = entry['name']
            numbers = []
        if key not in buckets:
            buckets[key] = {'pattern': key, 'items': [], 'numbers': []}
            order.append(key)
        buckets[key]['items'].append(entry)
        buckets[key]['numbers'].extend(numbers)

    groups, singles = [], []
    for key in order:
        bucket = buckets[key]
        if len(bucket['items']) >= MIN_GROUP and _NUM_RE.search(bucket['items'][0]['name']):
            groups.append(bucket)
        else:
            singles.extend(bucket['items'])
    singles.sort(key=lambda e: e['name'])
    return groups, singles


def natural_key(name):
    """Sort `rollout_2` before `rollout_10`."""
    return [int(part) if part.isdigit() else part
            for part in re.split(r'(\d+)', name)]


def skipped(name, skip_globs):
    return any(fnmatch.fnmatch(name, pattern) for pattern in skip_globs)


# ──────────────────────────────────────────────────────────────────────────────
# scan
# ──────────────────────────────────────────────────────────────────────────────

def ext_of(name):
    """`rollout_3.gif` → `.gif`; `Makefile` → `(no ext)`; dotfiles keep their name."""
    base = os.path.basename(name)
    stem, ext = os.path.splitext(base)
    if not ext or not stem:
        return '(no ext)'
    return ext.lower()


def scan(path, depth, opts, stats, acc):
    """Recursive scan → a node dict. Never follows symlinks.

    `acc` accumulates the per-root extension census ({ext: [count, bytes]}) —
    the "is this worth downloading?" number, since one `.mp4` line can dwarf
    everything else in the tree.
    """
    node = {
        'name': os.path.basename(path.rstrip(os.sep)) or path,
        'path': path,
        'dirs': [],
        'files': [],
        'n_files_here': 0,
        'bytes_here': 0,
        'total_files': 0,
        'total_bytes': 0,
        'newest': 0.0,
        'truncated': False,
        'error': '',
    }
    try:
        entries = list(os.scandir(path))
    except OSError as exc:
        node['error'] = str(exc)
        stats['errors'] += 1
        return node

    sub_dirs, files = [], []
    for entry in entries:
        if skipped(entry.name, opts.skip):
            continue
        try:
            is_dir = entry.is_dir(follow_symlinks=False)
            is_link = entry.is_symlink()
            info = entry.stat(follow_symlinks=False)
        except OSError:
            stats['errors'] += 1
            continue
        if is_link:
            files.append({'name': entry.name + ' -> (symlink)', 'size': 0,
                          'mtime': info.st_mtime})
            continue
        if is_dir:
            sub_dirs.append(entry.name)
        else:
            files.append({'name': entry.name, 'size': info.st_size,
                          'mtime': info.st_mtime})
            bucket = acc.setdefault(ext_of(entry.name), [0, 0])
            bucket[0] += 1
            bucket[1] += info.st_size

    files.sort(key=lambda f: natural_key(f['name']))
    node['files'] = files
    node['n_files_here'] = len(files)
    node['bytes_here'] = sum(f['size'] for f in files)
    node['newest'] = max([f['mtime'] for f in files], default=0.0)
    stats['dirs'] += 1
    stats['files'] += len(files)

    if depth >= opts.max_depth:
        # Still count what is below, so the totals stay honest; just do not print it.
        below_files, below_bytes, below_newest = tally(path, opts, stats, acc)
        node['truncated'] = True
        node['total_files'] = len(files) + below_files
        node['total_bytes'] = node['bytes_here'] + below_bytes
        node['newest'] = max(node['newest'], below_newest)
        node['n_subdirs_hidden'] = len(sub_dirs)
        return node

    for name in sorted(sub_dirs, key=natural_key):
        child = scan(os.path.join(path, name), depth + 1, opts, stats, acc)
        node['dirs'].append(child)

    node['total_files'] = len(files) + sum(d['total_files'] for d in node['dirs'])
    node['total_bytes'] = node['bytes_here'] + sum(d['total_bytes'] for d in node['dirs'])
    node['newest'] = max([node['newest']] + [d['newest'] for d in node['dirs']])
    return node


def tally(path, opts, stats, acc):
    """Count files/bytes/newest below a cut-off point without recording names.

    The extension census still gets filled here, so `--dirs-only` / a shallow
    `-d` gives the full size breakdown even though no filenames are printed.
    """
    n_files, n_bytes, newest = 0, 0, 0.0
    for root, dirnames, filenames in os.walk(path, followlinks=False):
        dirnames[:] = [d for d in dirnames if not skipped(d, opts.skip)]
        for name in filenames:
            if skipped(name, opts.skip):
                continue
            try:
                info = os.lstat(os.path.join(root, name))
            except OSError:
                stats['errors'] += 1
                continue
            n_files += 1
            n_bytes += info.st_size
            newest = max(newest, info.st_mtime)
            bucket = acc.setdefault(ext_of(name), [0, 0])
            bucket[0] += 1
            bucket[1] += info.st_size
    return n_files, n_bytes, newest


# ──────────────────────────────────────────────────────────────────────────────
# render
# ──────────────────────────────────────────────────────────────────────────────

def group_line(group):
    """One collapsed-group line, carrying the whole group's signature.

        rollout_<N>_stats.json   x1080   1.4 MiB   [0..1079]   ea 1.2-1.4 KiB   newest 2026-08-12 15:03

    `ea` (per-file size) is only printed when the members differ in size — that
    is the interesting case: same-name files of wildly different size usually
    mean truncated or half-written rollouts.
    """
    items = group['items']
    sizes = [item['size'] for item in items]
    numbers = group['numbers']
    parts = [f'{group["pattern"]}', f'x{len(items)}', human(sum(sizes))]
    # The index span is only meaningful for a single-counter pattern; with two
    # (`ctx<N>_traj<N>`) the numbers of both axes are pooled and a span would lie.
    if numbers and group['pattern'].count('<N>') == 1:
        parts.append(f'[{min(numbers)}..{max(numbers)}]')
    if min(sizes) != max(sizes):
        parts.append(f'ea {human(min(sizes))}-{human(max(sizes))}')
    else:
        parts.append(f'ea {human(sizes[0])}')
    parts.append(f'newest {stamp(max(item["mtime"] for item in items))}')
    return '   '.join(parts)


def ext_table(acc, limit=12):
    """`[('.mp4', 210, 12884901888), …]` — biggest total bytes first.

    `limit=None` returns every extension (used by the JSON output).
    """
    rows = sorted(acc.items(), key=lambda kv: kv[1][1], reverse=True)
    if limit:
        rows = rows[:limit]
    return [(ext, count, size) for ext, (count, size) in rows]


def render(node, opts, lines, prefix='', is_last=True, is_root=False):
    if is_root:
        lines.append(f'{node["path"]}{os.sep}')
        child_prefix = ''
    else:
        connector = '`-- ' if is_last else '|-- '
        summary = (f'  [{node["total_files"]} files, {human(node["total_bytes"])}'
                   f', newest {stamp(node["newest"])}]'
                   if opts.dir_totals and node['total_files'] else '')
        lines.append(f'{prefix}{connector}{node["name"]}{os.sep}{summary}')
        child_prefix = prefix + ('    ' if is_last else '|   ')

    if node['error']:
        lines.append(f'{child_prefix}    !! unreadable: {node["error"]}')
        return
    if node['truncated']:
        lines.append(f'{child_prefix}    ... depth limit — {node.get("n_subdirs_hidden", 0)} '
                     f'subdir(s), {node["total_files"] - node["n_files_here"]} more file(s) '
                     f'not listed')

    rows = []
    if not opts.dirs_only:
        groups, singles = group_by_pattern(node['files'])
        for group in groups:
            rows.append(('file', group_line(group)))
        for item in singles:
            rows.append(('file', f'{item["name"]}   {human(item["size"])}   '
                                 f'{stamp(item["mtime"])}'))

    n_shown = len(rows)
    hidden = 0
    if opts.max_entries and n_shown > opts.max_entries:
        hidden = n_shown - opts.max_entries
        rows = rows[:opts.max_entries]

    n_dirs = len(node['dirs'])
    for index, (_, text) in enumerate(rows):
        last = (index == len(rows) - 1) and hidden == 0 and n_dirs == 0
        lines.append(f'{child_prefix}{"`-- " if last else "|-- "}{text}')
    if hidden:
        last = n_dirs == 0
        lines.append(f'{child_prefix}{"`-- " if last else "|-- "}... +{hidden} more entr(y/ies) '
                     f'(--max-entries)')

    for index, child in enumerate(node['dirs']):
        render(child, opts, lines, child_prefix, index == n_dirs - 1)


def to_json(node, opts):
    out = {
        'path': node['path'],
        'name': node['name'],
        'total_files': node['total_files'],
        'total_bytes': node['total_bytes'],
        'newest_mtime': stamp(node['newest']),
        'truncated': node['truncated'],
    }
    if node['error']:
        out['error'] = node['error']
    if not opts.dirs_only:
        groups, singles = group_by_pattern(node['files'])
        out['file_groups'] = [
            {'pattern': g['pattern'], 'count': len(g['items']),
             'bytes': sum(i['size'] for i in g['items']),
             'bytes_min': min(i['size'] for i in g['items']),
             'bytes_max': max(i['size'] for i in g['items']),
             'newest': stamp(max(i['mtime'] for i in g['items'])),
             # Single-counter patterns only — see group_line().
             'min': (min(g['numbers'])
                     if g['numbers'] and g['pattern'].count('<N>') == 1 else None),
             'max': (max(g['numbers'])
                     if g['numbers'] and g['pattern'].count('<N>') == 1 else None)}
            for g in groups]
        out['files'] = [{'name': f['name'], 'bytes': f['size'],
                         'mtime': stamp(f['mtime'])} for f in singles]
    out['dirs'] = [to_json(child, opts) for child in node['dirs']]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description='Capture the file tree of one or more paths into a single '
                    'small file (numbered siblings are collapsed).',
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument('paths', nargs='+', help='one or more directories (or files) to capture')
    p.add_argument('-o', '--output', default='tree_capture.txt',
                   help='output file (default: %(default)s). "-" writes to stdout')
    p.add_argument('-d', '--max-depth', type=int, default=8,
                   help='how deep to descend before summarising (default: %(default)s)')
    p.add_argument('-n', '--max-entries', type=int, default=40,
                   help='max file lines printed per directory, AFTER collapsing '
                        '(0 = unlimited, default: %(default)s)')
    p.add_argument('--dirs-only', action='store_true',
                   help='directories only — the cheapest overview of a huge tree')
    p.add_argument('--no-dir-totals', dest='dir_totals', action='store_false',
                   help='omit the per-directory [files, size, newest] summary')
    p.add_argument('--skip', action='append', default=list(DEFAULT_SKIP),
                   metavar='GLOB', help='name glob to skip (repeatable). Default: '
                                        + ', '.join(DEFAULT_SKIP))
    p.add_argument('--format', choices=['text', 'json'], default='text',
                   help='text tree (default) or JSON')
    return p.parse_args(argv)


def main(argv=None):
    opts = parse_args(argv)
    started = time.time()
    stats = {'dirs': 0, 'files': 0, 'errors': 0}

    roots = []
    for path in opts.paths:
        absolute = os.path.abspath(os.path.expanduser(path))
        if not os.path.exists(absolute):
            print(f'[ WARN ] path does not exist, skipped: {absolute}', file=sys.stderr)
            continue
        if not os.path.isdir(absolute):
            print(f'[ WARN ] not a directory, skipped: {absolute}', file=sys.stderr)
            continue
        roots.append(absolute)

    if not roots:
        print('[ FATAL ] no readable directory given.', file=sys.stderr)
        return 1

    nodes, censuses = [], []
    for root in roots:
        acc = {}
        nodes.append(scan(root, 1, opts, stats, acc))
        censuses.append(acc)

    if opts.format == 'json':
        payload = {
            'captured_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'host': os.uname().nodename if hasattr(os, 'uname') else '',
            'options': {'max_depth': opts.max_depth, 'max_entries': opts.max_entries,
                        'dirs_only': opts.dirs_only, 'skip': opts.skip},
            'roots': [
                dict(to_json(node, opts),
                     by_extension=[{'ext': ext, 'count': count, 'bytes': size}
                                   for ext, count, size in ext_table(acc, limit=None)])
                for node, acc in zip(nodes, censuses)
            ],
        }
        text = json.dumps(payload, indent=1)
    else:
        lines = [
            '=' * 78,
            'capture_tree — file-tree capture',
            '=' * 78,
            f'captured : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            + (f'   host {os.uname().nodename}' if hasattr(os, 'uname') else ''),
            f'options  : max_depth={opts.max_depth}  max_entries={opts.max_entries}'
            f'  dirs_only={opts.dirs_only}  skip={opts.skip}',
            '',
            'ROOTS',
            '-' * 78,
        ]
        for node in nodes:
            lines.append(f'  {node["path"]}   {node["total_files"]} files   '
                         f'{human(node["total_bytes"])}   newest {stamp(node["newest"])}')
        lines += ['', 'NOTE  "name_<N>.json  x1080  1.4 MiB  [0..1079]  ea 1.3 KiB  newest ..."',
                  '      = 1080 files collapsed into one line (group total, per-file size,',
                  '        index span, newest mtime).', '']
        for node, acc in zip(nodes, censuses):
            lines += ['=' * 78, '']
            lines.append(f'{node["path"]}  —  BY EXTENSION '
                         f'(what a download would actually cost)')
            lines.append('-' * 78)
            for ext, count, size in ext_table(acc):
                share = (100.0 * size / node['total_bytes']) if node['total_bytes'] else 0.0
                lines.append(f'  {ext:<12} {count:>8} files   {human(size):>10}   {share:5.1f}%')
            lines.append('')
            render(node, opts, lines, is_root=True)
            lines.append('')
        text = '\n'.join(lines)

    if opts.output == '-':
        sys.stdout.write(text + '\n')
    else:
        with open(opts.output, 'w') as f:
            f.write(text + '\n')
        size = os.path.getsize(opts.output)
        print(f'[ ok ] {stats["dirs"]} dirs, {stats["files"]} files scanned in '
              f'{time.time() - started:.1f}s'
              + (f' ({stats["errors"]} unreadable)' if stats['errors'] else ''))
        print(f'[ ok ] capture written: {opts.output}   ({human(size)})')
        if size > 2 * 1024 * 1024:
            print('[ hint ] that is big — rerun with  -d 4 -n 15  or --dirs-only')
    return 0


if __name__ == '__main__':
    sys.exit(main())
