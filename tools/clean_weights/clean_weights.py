#!/usr/bin/env python3
"""clean_weights.py — safely prune periodic training checkpoints, keep only the best.

Deletes:   .../<run>/<seed>/state_<epoch>.pt   (periodic snapshots, e.g. state_80000.pt)
Keeps:     .../<run>/<seed>/state_best.pt      (best; each .pt bundles model + ema)
           everything else (state_best.pt, *.pth baselines, *.pkl, gifs, plans, ...)

Safety
------
* DRY-RUN by default. Nothing is deleted unless you pass --apply.
* Best-gated: a directory is pruned ONLY if it contains a state_best.pt. Directories
  with numbered checkpoints but NO state_best.pt (crashed / still-running jobs) are
  SKIPPED and reported, so a run never loses its only weights.
* Tight match: only files matching  state_<digits>.pt  are ever removed.
* Full audit log written to  <root>/_clean_weights_runlogs/clean_weights_<ts>.log
  with BEFORE (sizes + free disk), the DELETE/SKIP manifest, and AFTER (freed bytes).

Stdlib only — no torch / conda needed.

Usage
-----
    python clean_weights.py                       # dry-run on default root
    python clean_weights.py --root /path/to/logs  # dry-run on given root
    python clean_weights.py --apply               # actually delete
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

DEFAULT_ROOT = Path.home() / "FMPCC" / "FM-PCC" / "logs"
RUNLOGS_DIRNAME = "_clean_weights_runlogs"
PERIODIC_RE = re.compile(r"^state_\d+\.pt$")   # state_best.pt does NOT match
BEST_NAME = "state_best.pt"


def human(n: int) -> str:
    f = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if f < 1024 or unit == "TB":
            return f"{f:.1f}{unit}" if unit != "B" else f"{int(f)}B"
        f /= 1024
    return f"{f:.1f}TB"


def resolve_excludes(root: Path, patterns: list[str]):
    """Split --exclude values into (absolute ancestor paths, glob patterns).

    A value that resolves to an existing path (absolute, or relative to root) is treated
    as an ancestor filter: anything under it is protected. Everything is ALSO kept as a
    glob so wildcard values like '*alphaflow*/7' work.
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
    import fnmatch
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


def scan(root: Path, runlogs_dir: Path, anc: list[Path], globs: list[str]):
    """Single walk: total size, per-top-level sizes, and periodic-checkpoint candidates.

    Returns (total_bytes, per_top {name: bytes}, to_delete [(path, size, mtime)],
             nobest_dirs set[Path], excluded [(path, size)]).
    """
    total = 0
    per_top: dict[str, int] = {}
    has_best: set[str] = set()
    candidates: list[tuple[Path, int]] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # never scan / touch the run-logs folder
        if Path(dirpath) == runlogs_dir:
            dirnames[:] = []
            continue
        if runlogs_dir.name in dirnames and Path(dirpath) == root:
            dirnames.remove(runlogs_dir.name)

        if BEST_NAME in filenames:
            has_best.add(dirpath)

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
            if PERIODIC_RE.match(name):
                candidates.append((fp, st.st_size))

    to_delete: list[tuple[Path, int, float]] = []
    nobest_dirs: set[Path] = set()
    excluded: list[tuple[Path, int]] = []
    for fp, size in candidates:
        if is_excluded(fp, root, anc, globs):
            excluded.append((fp, size))
        elif str(fp.parent) in has_best:
            to_delete.append((fp, size, fp.stat().st_mtime))
        else:
            nobest_dirs.add(fp.parent)

    return total, per_top, to_delete, nobest_dirs, excluded


def main() -> int:
    ap = argparse.ArgumentParser(description="Prune periodic checkpoints, keep state_best.pt.")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT,
                    help=f"logs root to scan (default: {DEFAULT_ROOT})")
    ap.add_argument("--apply", action="store_true",
                    help="actually delete (default: dry-run, deletes nothing)")
    ap.add_argument("--exclude", action="append", default=[], metavar="PATH_OR_GLOB",
                    help="protect a folder/file from deletion; repeatable. Accepts an "
                         "absolute path, a path relative to --root, or a glob "
                         "(e.g. '*alphaflow*/7'). Anything under an excluded dir is kept.")
    args = ap.parse_args()

    root: Path = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"ERROR: root does not exist: {root}", file=sys.stderr)
        return 1

    runlogs_dir = root / RUNLOGS_DIRNAME
    runlogs_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    logfile = runlogs_dir / f"clean_weights_{ts}.log"
    mode = "APPLY (deleting)" if args.apply else "DRY-RUN (no files deleted)"

    print("=" * 63)
    print(" clean_weights.py")
    print(f" root : {root}")
    print(f" mode : {mode}")
    print(f" log  : {logfile}")
    print("=" * 63)

    anc, globs = resolve_excludes(root, args.exclude)
    if args.exclude:
        print(" excl : " + ", ".join(args.exclude))

    # ---- BEFORE scan --------------------------------------------------------
    total_before, per_top, to_delete, nobest_dirs, excluded = scan(
        root, runlogs_dir, anc, globs)
    free_before = shutil.disk_usage(root).free
    reclaimable = sum(sz for _, sz, _ in to_delete)

    log = open(logfile, "w")

    def w(line: str = ""):
        log.write(line + "\n")

    w(f"# clean_weights run {ts}")
    w(f"# root={root}")
    w(f"# mode={'APPLY' if args.apply else 'DRY-RUN'}")
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
    if nobest_dirs:
        w()
        w("== SKIPPED: numbered checkpoints but NO state_best.pt ==")
        for d in sorted(nobest_dirs):
            w(f"SKIP-NOBEST  {d}")
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
    print(f"Periodic checkpoints to delete : {len(to_delete)}")
    if excluded:
        excl_bytes = sum(sz for _, sz in excluded)
        print(f"Protected by --exclude         : {len(excluded)} file(s), "
              f"{human(excl_bytes)} kept")
    if nobest_dirs:
        print(f"!! {len(nobest_dirs)} dir(s) with numbered checkpoints but NO "
              f"{BEST_NAME} — SKIPPED (see log)")
    print()
    if args.apply:
        print(f"DONE — deleted {deleted} files, freed {human(freed)} "
              f"(errors: {errors}).")
    else:
        print(f"DRY-RUN — would free {human(reclaimable)}. "
              f"Re-run with --apply to delete.")
    print(f"Log: {logfile}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
