#!/usr/bin/env python3
"""clean_gifs.py — safely delete rollout GIFs (and other render artefacts) under a folder.

Deletes:   every  *.gif  found recursively under the given folder
           (extensions are configurable: --ext gif --ext mp4 ...)
Keeps:     everything else (weights *.pt, *.npz, *.pkl, *.json, PNG diagnostics, ...)

Typical target — one eval/run dir on the cluster:

    logs/aligning-d3il-visual/mix_visual_aligning_diffusion/H8_K100_D..._Ediffusion/6

whose GIFs sit in  .../plans/<variant>/diagnostics/rollout_<i>.gif .

Safety
------
* DRY-RUN by default. Nothing is deleted unless you pass --apply.
* Explicit root: the folder to clean is a required argument — there is no default,
  so the tool can never sweep the whole logs tree by accident. `/` and `$HOME`
  are refused outright.
* Tight match: only regular files whose extension is in --ext (default: gif) are
  ever removed. Symlinks are never followed or deleted.
* --keep-per-dir N keeps the first N GIFs (natural order: rollout_0, rollout_1, ...)
  in every directory, so you can prune while keeping a visual sample.
* --exclude PATH_OR_GLOB protects anything you still need.
* Full audit log ("changelog of what was deleted") written to
  <root>/_clean_gifs_runlogs/clean_gifs_<ts>.log  with BEFORE (sizes + free disk),
  the DELETE / KEEP / EXCLUDE manifest, and AFTER (freed bytes).

Stdlib only — no torch / conda / imageio needed.

Usage
-----
    python clean_gifs.py /path/to/run/6                 # dry-run
    python clean_gifs.py /path/to/run/6 --apply         # actually delete
    python clean_gifs.py /path/to/run/6 --apply --keep-per-dir 2
    python clean_gifs.py /path/to/logs --ext gif --ext mp4 --apply
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

RUNLOGS_DIRNAME = "_clean_gifs_runlogs"
DEFAULT_EXTS = ["gif"]
NUM_RE = re.compile(r"(\d+)")


def human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f}{unit}" if unit != "B" else f"{int(f)}B"
        f /= 1024
    return f"{f:.1f}TB"


def natural_key(name: str):
    """rollout_2.gif sorts before rollout_10.gif."""
    return [int(t) if t.isdigit() else t.lower() for t in NUM_RE.split(name)]


def resolve_excludes(root: Path, patterns: list[str]):
    """Split --exclude values into (absolute ancestor paths, glob patterns).

    A value that resolves to an existing path (absolute, or relative to root) is treated
    as an ancestor filter: anything under it is protected. Everything is ALSO kept as a
    glob so wildcard values like '*expert*' or 'plans/*/diagnostics' work.
    """
    anc: list[Path] = []
    globs: list[str] = list(patterns)
    for p in patterns:
        cand = Path(p)
        if not cand.is_absolute():
            cand = root / p
        try:
            cand = cand.resolve()
        except OSError:
            continue
        if cand.exists():
            anc.append(cand)
    return anc, globs


def is_excluded(fp: Path, root: Path, anc: list[Path], globs: list[str]) -> bool:
    for a in anc:
        if fp == a or a in fp.parents:
            return True
    fp_s = str(fp)
    try:
        rel_s = str(fp.relative_to(root))
    except ValueError:
        rel_s = fp_s
    for g in globs:
        if fnmatch.fnmatch(fp_s, g) or fnmatch.fnmatch(rel_s, g) \
                or fnmatch.fnmatch(fp_s, f"*{g}*"):
            return True
    return False


def scan(root: Path, runlogs_dir: Path, exts: set[str], keep_per_dir: int,
         anc: list[Path], globs: list[str]):
    """Single walk: total size, per-top-level sizes, and the GIF verdicts.

    Returns (total_bytes, per_top {name: bytes},
             to_delete [(path, size, mtime)], kept [(path, size)], excluded [(path, size)]).
    """
    total = 0
    per_top: dict[str, int] = {}
    per_dir: dict[str, list[tuple[Path, int]]] = {}

    for dirpath, dirnames, filenames in os.walk(root):
        # never scan / touch the run-logs folder
        if Path(dirpath) == runlogs_dir:
            dirnames[:] = []
            continue
        if runlogs_dir.name in dirnames and Path(dirpath) == root:
            dirnames.remove(runlogs_dir.name)

        try:
            rel_top = Path(dirpath).relative_to(root).parts[0]
        except (ValueError, IndexError):
            rel_top = "."

        for name in filenames:
            fp = Path(dirpath) / name
            try:
                st = fp.lstat()
            except OSError:
                continue
            if not os.path.isfile(fp) or os.path.islink(fp):
                continue
            total += st.st_size
            per_top[rel_top] = per_top.get(rel_top, 0) + st.st_size
            if name.rsplit(".", 1)[-1].lower() in exts and "." in name:
                per_dir.setdefault(dirpath, []).append((fp, st.st_size))

    to_delete: list[tuple[Path, int, float]] = []
    kept: list[tuple[Path, int]] = []
    excluded: list[tuple[Path, int]] = []

    for dirpath in sorted(per_dir):
        files = sorted(per_dir[dirpath], key=lambda t: natural_key(t[0].name))
        survivors = 0
        for fp, size in files:
            if is_excluded(fp, root, anc, globs):
                excluded.append((fp, size))
            elif survivors < keep_per_dir:
                survivors += 1
                kept.append((fp, size))
            else:
                try:
                    mt = fp.stat().st_mtime
                except OSError:
                    mt = 0.0
                to_delete.append((fp, size, mt))

    return total, per_top, to_delete, kept, excluded


def prune_empty_dirs(root: Path, runlogs_dir: Path, apply: bool):
    """Bottom-up removal of directories that are now empty (never removes root)."""
    removed: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        d = Path(dirpath)
        if d == root or d == runlogs_dir or runlogs_dir in d.parents:
            continue
        try:
            if any(d.iterdir()):
                continue
        except OSError:
            continue
        if apply:
            try:
                d.rmdir()
            except OSError:
                continue
        removed.append(d)
    return removed


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Delete rollout GIFs under a folder (dry-run by default).")
    ap.add_argument("root", type=Path,
                    help="folder to clean, scanned recursively (required — no default)")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default: dry-run, deletes nothing)")
    ap.add_argument("--ext", action="append", default=[], metavar="EXT",
                    help="file extension to delete, without the dot; repeatable "
                         f"(default: {', '.join(DEFAULT_EXTS)}). e.g. --ext gif --ext mp4")
    ap.add_argument("--keep-per-dir", type=int, default=0, metavar="N",
                    help="keep the first N matches per directory in natural order "
                         "(rollout_0, rollout_1, ...) as a visual sample (default: 0)")
    ap.add_argument("--exclude", action="append", default=[], metavar="PATH_OR_GLOB",
                    help="protect a folder/file from deletion; repeatable. Accepts an "
                         "absolute path, a path relative to the root, or a glob "
                         "(e.g. '*expert*'). Anything under an excluded dir is kept.")
    ap.add_argument("--rm-empty-dirs", action="store_true",
                    help="also remove directories left empty after deletion")
    ap.add_argument("--log-dir", type=Path, default=None,
                    help=f"where to write the audit log (default: <root>/{RUNLOGS_DIRNAME})")
    args = ap.parse_args()

    root: Path = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: root does not exist (or is not a dir): {root}", file=sys.stderr)
        return 1
    if root == Path(root.anchor) or root == Path.home():
        print(f"ERROR: refusing to operate on {root}", file=sys.stderr)
        return 1
    if args.keep_per_dir < 0:
        print("ERROR: --keep-per-dir must be >= 0", file=sys.stderr)
        return 1

    exts = {e.lower().lstrip(".") for e in (args.ext or DEFAULT_EXTS)}

    runlogs_dir = (args.log_dir.expanduser().resolve() if args.log_dir
                   else root / RUNLOGS_DIRNAME)
    runlogs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = runlogs_dir / f"clean_gifs_{ts}.log"
    mode = "APPLY (deleting)" if args.apply else "DRY-RUN (no files deleted)"

    print("=" * 63)
    print(" clean_gifs.py")
    print(f" root : {root}")
    print(f" ext  : {', '.join('.' + e for e in sorted(exts))}")
    print(f" mode : {mode}")
    print(f" log  : {logfile}")
    print("=" * 63)

    anc, globs = resolve_excludes(root, args.exclude)
    if args.exclude:
        print(" excl : " + ", ".join(args.exclude))
    if args.keep_per_dir:
        print(f" keep : first {args.keep_per_dir} per directory")

    # ---- BEFORE scan --------------------------------------------------------
    total_before, per_top, to_delete, kept, excluded = scan(
        root, runlogs_dir, exts, args.keep_per_dir, anc, globs)
    free_before = shutil.disk_usage(root).free
    reclaimable = sum(sz for _, sz, _ in to_delete)
    n_dirs = len({fp.parent for fp, _, _ in to_delete})

    log = open(logfile, "w")

    def w(line: str = ""):
        log.write(line + "\n")

    w(f"# clean_gifs run {ts}")
    w(f"# root={root}")
    w(f"# ext={sorted(exts)}")
    w(f"# mode={'APPLY' if args.apply else 'DRY-RUN'}")
    w(f"# keep_per_dir={args.keep_per_dir}")
    if args.exclude:
        w(f"# exclude={args.exclude}")
    w()
    w("== BEFORE ==")
    w(f"total size : {human(total_before)} ({total_before} B)")
    w(f"free disk  : {human(free_before)} ({free_before} B)")
    w("per-top-level folder:")
    for name in sorted(per_top, key=lambda k: -per_top[k]):
        w(f"  {human(per_top[name]):>10}  {name}")
    w()
    w("== DELETE MANIFEST (file | size | mtime) ==")
    for fp, sz, mt in sorted(to_delete):
        mts = datetime.fromtimestamp(mt).strftime("%Y-%m-%d %H:%M:%S")
        w(f"DELETE {human(sz):>10}  {mts}  {fp}")
    if kept:
        w()
        w(f"== KEPT as sample (--keep-per-dir {args.keep_per_dir}) ==")
        for fp, sz in sorted(kept):
            w(f"KEEP {human(sz):>10}  {fp}")
    if excluded:
        w()
        w("== EXCLUDED by --exclude (protected) ==")
        for fp, sz in sorted(excluded):
            w(f"EXCLUDE {human(sz):>10}  {fp}")

    # ---- APPLY --------------------------------------------------------------
    deleted, freed, errors = 0, 0, 0
    if args.apply:
        for fp, sz, _ in to_delete:
            try:
                fp.unlink()
                deleted += 1
                freed += sz
            except OSError as e:
                errors += 1
                w(f"ERROR deleting {fp}: {e}")

    empty_dirs: list[Path] = []
    if args.rm_empty_dirs:
        empty_dirs = prune_empty_dirs(root, runlogs_dir, args.apply)
        if empty_dirs:
            w()
            w("== EMPTY DIRS " + ("REMOVED ==" if args.apply else "(would remove) =="))
            for d in sorted(empty_dirs):
                w(f"RMDIR  {d}")

    # ---- AFTER --------------------------------------------------------------
    if args.apply:
        total_after = total_before - freed
        free_after = shutil.disk_usage(root).free
        w()
        w("== AFTER ==")
        w(f"deleted files : {deleted}  (errors: {errors})")
        w(f"freed         : {human(freed)} ({freed} B)")
        w(f"total size    : {human(total_after)} ({total_after} B)")
        w(f"free disk     : {human(free_after)} ({free_after} B)")
    else:
        w()
        w("== AFTER (projected, dry-run) ==")
        w(f"would delete  : {len(to_delete)} files")
        w(f"would free    : {human(reclaimable)} ({reclaimable} B)")
        w(f"total size -> : {human(total_before - reclaimable)} (projected)")
    log.close()

    # ---- console summary ----------------------------------------------------
    print()
    print(f"BEFORE  total {human(total_before)} | free disk {human(free_before)}")
    print(f"Matched files to delete : {len(to_delete)} in {n_dirs} dir(s), "
          f"{human(reclaimable)}")
    if kept:
        print(f"Kept as sample          : {len(kept)} file(s)")
    if excluded:
        excl_bytes = sum(sz for _, sz in excluded)
        print(f"Protected by --exclude  : {len(excluded)} file(s), {human(excl_bytes)} kept")
    if empty_dirs:
        verb = "removed" if args.apply else "would remove"
        print(f"Empty dirs {verb}   : {len(empty_dirs)}")
    print()
    if args.apply:
        print(f"DONE — deleted {deleted} files, freed {human(freed)} (errors: {errors}).")
    else:
        print(f"DRY-RUN — would free {human(reclaimable)}. "
              f"Re-run with --apply to delete.")
    print(f"Log: {logfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
