# clean_gifs

Delete **rollout GIFs** (and optionally MP4s) under a given folder, recursively, with a
dry-run first and a full audit log of what was removed. Sibling of
[`../clean_weights`](../clean_weights/README.md) — same safety model, same log layout,
different target: renders instead of checkpoints.

- **Runs on the cluster** (where `logs/` lives). Pure Python **stdlib** — no torch/conda/imageio.
- **Never run the pipeline locally** in the AI-coding container; this is just a file tool,
  but it operates on the cluster's `logs/`.

## Why

Eval writes one GIF *and* one MP4 per rollout into
`<run>/<seed>/plans/<variant>/diagnostics/rollout_<i>.gif` (see
`mix_visual_aligning_test/eval_mix_visual_aligning.py`, `record_mode=all`). With hundreds of
rollouts per candidate these dominate a run dir, and they are pure inspection material —
nothing downstream (metrics, npz, DA scripts) reads them.

## What it keeps vs deletes

| File | Action |
|------|--------|
| `*.gif` anywhere under the root | **DELETE** |
| `*.mp4`, `*.png`, `*.npz`, `*.pkl`, `*.json`, `state_*.pt`, … | untouched |
| symlinks (even `*.gif`) | never followed, never deleted |

`--ext` changes the matched extensions (e.g. `--ext gif --ext mp4` to take the videos too).

## Safety

1. **Dry-run by default.** Nothing is deleted unless you pass `--apply`.
2. **Explicit root.** The folder is a required positional argument — there is no default,
   so this can never sweep the whole `logs/` tree by accident. `/` and `$HOME` are refused.
3. **Tight match.** Only regular files whose extension is in `--ext` (default `gif`) are
   removed. Symlinks are skipped.
4. **`--keep-per-dir N`** keeps the first N matches per directory in natural order
   (`rollout_0`, `rollout_1`, …, `rollout_10`) so you can prune but keep a visual sample.
5. **`--exclude`** protects paths/globs (e.g. the expert reference GIFs).
6. **Audit log** for every run — the "changelog of what was deleted" (see below).

## Usage

```bash
cd ~/FMPCC/FM-PCC

RUN=logs/aligning-d3il-visual/mix_visual_aligning_diffusion/H8_K100_Dmix_visual_aligning.models.visual_gaussian_diffusion.VisualGaussianDiffusion_aw10_VTrue_steps1000_bs64_filmv1_Ediffusion/6

# 1) DRY-RUN — reports before-size, what would be deleted, projected reclaim
python tools/clean_gifs/clean_gifs.py "$RUN"

# 2) Review the log
ls -t "$RUN"/_clean_gifs_runlogs/ | head
less "$RUN"/_clean_gifs_runlogs/clean_gifs_*.log

# 3) APPLY — actually delete
python tools/clean_gifs/clean_gifs.py "$RUN" --apply
```

More:

```bash
# keep rollout_0 in every diagnostics dir as a sample
python tools/clean_gifs/clean_gifs.py "$RUN" --apply --keep-per-dir 1

# take the MP4s as well, but protect the expert reference renders
python tools/clean_gifs/clean_gifs.py "$RUN" --apply --ext gif --ext mp4 --exclude '*expert*'

# sweep a whole task tree, tidy up dirs that end up empty
python tools/clean_gifs/clean_gifs.py logs/aligning-d3il-visual --apply --rm-empty-dirs

# keep the audit log outside the folder being cleaned
python tools/clean_gifs/clean_gifs.py "$RUN" --apply --log-dir ~/FMPCC/FM-PCC/logs/_clean_gifs_runlogs
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `ROOT` (positional) | — (**required**) | folder to clean, scanned recursively |
| `--apply` | off (dry-run) | actually delete |
| `--ext EXT` | `gif` | extension to delete, no dot; repeatable |
| `--keep-per-dir N` | `0` | keep the first N matches per directory (natural order) |
| `--exclude PATH_OR_GLOB` | none | protect a folder/file; repeatable. Absolute path, path relative to ROOT, or glob |
| `--rm-empty-dirs` | off | also remove directories left empty after deletion (never ROOT) |
| `--log-dir PATH` | `<ROOT>/_clean_gifs_runlogs` | where to write the audit log |
| `-h/--help` | — | help |

## Logging (the "what was deleted" changelog)

Each invocation writes `<ROOT>/_clean_gifs_runlogs/clean_gifs_<YYYYMMDD_HHMMSS>.log`
(inside the training `logs/` tree → gitignored / cluster-side) with:

- **header** — root, extensions, mode, `keep_per_dir`, excludes.
- **BEFORE** — total size of the root, free disk, per-top-level-folder sizes.
- **DELETE manifest** — every file with size + mtime; `KEEP` lines for the per-dir samples,
  `EXCLUDE` lines for protected files, `RMDIR` lines for empty dirs.
- **AFTER** — deleted count, freed bytes, recomputed total + free disk (in `--apply`;
  projected in dry-run).

The dry-run log and the apply log have the same manifest format, so you can diff them.

## Not handled (do manually if wanted)

- **PNG diagnostics** (`rollout_*.png` 9-panel figures) — small and often the thing you
  actually want to keep; pass `--ext png` explicitly if you disagree.
- **Checkpoints** — that's [`clean_weights`](../clean_weights/README.md).
- **`plans/` npz / pkl rollout data** — the DA scripts read these; never touched here.
