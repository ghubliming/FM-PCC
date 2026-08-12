# capture_tree

Capture the **file tree of one or more paths into a single small file**, so you can decide what an
old run folder actually holds *before* downloading any of it.

- **Runs on the cluster login node** ("Slurm lobby"). Pure Python **stdlib** — no conda env, no
  numpy, no sbatch. Read-only: `os.scandir`/`os.lstat` only, nothing is opened, moved or deleted.
- Sibling of [`../clean_gifs`](../clean_gifs/README.md) and
  [`../clean_weights`](../clean_weights/README.md) — same "cluster file tool" family, but this one
  never writes into the tree it inspects.

## Why it stays small

Eval runs produce thousands of near-identical files (`rollout_0_stats.json … rollout_1079_stats.json`).
`capture_tree` **collapses numbered siblings into one line**:

```
rollout_<N>_stats.json   x1080   1.4 MiB   [0..1079]   ea 1.2 KiB-1.4 KiB   newest 2026-08-12 15:03
```

group total · index span · per-file size (a range only when the members differ — that is how a
truncated/half-written rollout shows itself) · newest mtime. A tree with 50k files still lands in a
few dozen KB.

Every root also gets a **BY EXTENSION** census — the "what would a download actually cost" table:

```
  .mp4              1080 files     11.8 GiB    96.1%
  .json             2161 files      3.1 MiB     0.1%
```

## Usage

```bash
# one path (output: ./tree_capture.txt)
python3 tools/capture_tree/capture_tree.py logs/d3il_visual_aligning_baseline

# several paths, one capture file
python3 tools/capture_tree/capture_tree.py \
    logs/d3il_visual_aligning_baseline \
    logs/aligning-d3il-visual/plans \
    logs/avoiding-d3il/plans \
    -o legacy_runs_tree.txt

# huge/unknown tree: shallow overview first, then zoom in
python3 tools/capture_tree/capture_tree.py logs -d 3 --dirs-only -o overview.txt
python3 tools/capture_tree/capture_tree.py logs/<the interesting one> -d 8 -o detail.txt

# straight to the terminal, or machine-readable
python3 tools/capture_tree/capture_tree.py logs/... -o -
python3 tools/capture_tree/capture_tree.py logs/... --format json -o tree.json
```

Then bring back just that one file:

```bash
scp <cluster>:~/FMPCC/FM-PCC/legacy_runs_tree.txt .
```

## Options

| flag | default | what |
|---|---|---|
| `paths…` | — | one or more directories (required). Non-existent / non-directory args are warned about and skipped |
| `-o, --output` | `tree_capture.txt` | output file; `-` = stdout |
| `-d, --max-depth` | `8` | below this depth the subtree is summarised (file/byte/mtime totals stay correct — only the names are dropped) |
| `-n, --max-entries` | `40` | max file lines per directory **after** collapsing; `0` = unlimited |
| `--dirs-only` | off | directories only — cheapest overview of a huge tree (the extension census is still complete) |
| `--no-dir-totals` | on | drop the per-directory `[N files, size, newest]` summary |
| `--skip GLOB` | `.git`, `__pycache__`, `.ipynb_checkpoints`, `.mypy_cache` | repeatable name glob |
| `--format` | `text` | `text` tree or `json` |

## Reading the output

```
ROOTS                                  ← one line per path: files, total size, newest mtime
BY EXTENSION                           ← per root, biggest bytes first, with % of the root
<tree>
  name/  [N files, size, newest ...]   ← directory line: totals for the whole subtree
  name.json   1.3 KiB   2026-08-12 15:03   ← single file: size + mtime
  name_<N>.json   x1080  …              ← collapsed group (see above)
  ... depth limit — 3 subdir(s), 54 more file(s) not listed
  ... +12 more entr(y/ies) (--max-entries)
  !! unreadable: <errno>                ← permission/IO error; the scan continues
```

Symlinks are listed as `name -> (symlink)` with size 0 and are never followed — so a self-referential
link tree cannot make the scan loop.

## Notes

- Cost is one `stat` per file; a few-hundred-thousand-file tree takes seconds on the login node. If
  a path lives on a slow/archival mount, start with `--dirs-only -d 3`.
- The tool prints a hint when the capture exceeds 2 MiB — that means the collapsing did not catch
  the repetition (unusual file naming); rerun with `-d 4 -n 15`.
- Totals count what is *on disk*, not what a `tar`/`scp` would compress to — for `.json`/`.log`-heavy
  trees the real transfer is much smaller, for `.mp4`/`.gif` it is not.
