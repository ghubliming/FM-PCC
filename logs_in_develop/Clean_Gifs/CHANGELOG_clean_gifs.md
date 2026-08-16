# Changelog — `clean_gifs` tool

**Date:** 2026-08-08
**Author:** Claude (AI-coding), reviewed by user
**Context:** eval run dirs on the cluster are dominated by per-rollout renders, e.g.
`logs/aligning-d3il-visual/mix_visual_aligning_diffusion/H8_K100_D...VisualGaussianDiffusion_aw10_VTrue_steps1000_bs64_filmv1_Ediffusion/6`.
Need the GIF counterpart of `clean_weights`: delete renders under *a given folder*, safely,
with a record of what went.

## Added

- **`tools/clean_gifs/clean_gifs.py`** — Python (stdlib only) tool to delete GIFs recursively
  under one folder. Modelled on `tools/clean_weights/clean_weights.py` (same CLI shape,
  same BEFORE / manifest / AFTER audit log, `human()` / `resolve_excludes()` /
  `is_excluded()` carried over).
  - Deletes files whose extension is in `--ext` (default `gif`; repeatable, so
    `--ext gif --ext mp4` takes the videos too). Target confirmed in
    `mix_visual_aligning_test/eval_mix_visual_aligning.py:1463-1470` — eval writes
    `diagnostics/rollout_<i>.gif` + `.mp4` under `record_mode` `gif`/`video`/`all`,
    and nothing downstream reads them.
  - **Dry-run by default**; `--apply` to delete.
  - **Root is a required positional argument** — no default root, so it cannot sweep the
    whole `logs/` tree by accident. `/` and `$HOME` are refused outright.
  - **`--keep-per-dir N`**: keep the first N matches per directory in *natural* order
    (`rollout_0`, `rollout_1`, …, `rollout_10` — not lexicographic), to prune while keeping
    a visual sample.
  - **`--exclude PATH_OR_GLOB` (repeatable)**: absolute path, path relative to root, or glob
    (e.g. `*expert*` for the expert reference renders). Anything under an excluded dir is kept.
  - **`--rm-empty-dirs`**: bottom-up removal of dirs left empty after deletion; never removes
    the root or the run-logs dir.
  - **`--log-dir`**: put the audit log somewhere other than `<root>/_clean_gifs_runlogs/`.
  - Symlinks are never followed or deleted; only regular files are counted/removed.
  - Single `os.walk` computes BEFORE totals, per-top-level-folder sizes and candidates in one
    pass; its own run-logs folder is skipped.
  - Audit log per run at **`<root>/_clean_gifs_runlogs/clean_gifs_<ts>.log`** (inside the
    training `logs/` tree → gitignored / cluster-side): header (root/ext/mode/keep/exclude),
    **BEFORE** (sizes + free disk), **DELETE** manifest (file, size, mtime) plus `KEEP`,
    `EXCLUDE`, `RMDIR` lines, and **AFTER** (freed bytes, recomputed totals; projected in
    dry-run). Dry-run and apply logs share the format, so they diff cleanly.
- **`tools/clean_gifs/README.md`** — why, keep-vs-delete table, safety, usage examples on the
  real run path, option table, logging, out-of-scope notes.
- **`logs_in_develop/Clean_Gifs/CHANGELOG_clean_gifs.md`** — this file.

## Layout decisions

- Own folder `tools/clean_gifs/` next to `tools/clean_weights/`, not loose at repo root
  (same rule as the 2026-07-29 `clean_weights` entry).
- Dev doc (this changelog) in `logs_in_develop/`; the tool's per-run audit logs go into the
  real training `logs/` tree, never `logs_in_develop/`.

## Verification (local smoke test, AI-coding container)

Ran on a synthetic tree (stdlib file tool only — no pipeline, no GPU):

- Dry-run, `--keep-per-dir 1 --exclude '*expert*'` on 6 GIFs / 1 MP4 / 1 `state_best.pt` /
  1 GIF symlink: reported 4 deletions, 2 KEEP, 1 EXCLUDE, deleted nothing. ✓
- Manifest ordering: `rollout_0` kept, `rollout_1`/`rollout_2`/`rollout_10` deleted →
  natural sort works (lexicographic would have kept `rollout_10`). ✓
- `--apply`: removed exactly those 4; `.mp4`, `state_best.pt`, the symlink and the excluded
  expert GIF untouched; AFTER block recorded freed bytes. ✓
- `--rm-empty-dirs`: emptied `a/diagnostics` and `a` removed bottom-up, root kept. ✓
- Guards: `~` → `ERROR: refusing to operate on /home/vscode`, exit 1; missing root arg →
  argparse usage error. ✓

**Not yet run on the cluster** — real prune is a user action: dry-run → review log → `--apply`.

## Out of scope (manual)

- PNG 9-panel diagnostics (`rollout_*.png`) — pass `--ext png` if you want them gone.
- Checkpoints — that's `tools/clean_weights/`.
- `plans/` npz/pkl rollout data — read by the DA scripts, never touched here.
