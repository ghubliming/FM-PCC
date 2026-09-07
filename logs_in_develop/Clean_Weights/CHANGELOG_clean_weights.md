# Changelog — `clean_weights` tool

**Date:** 2026-07-29
**Author:** Claude (AI-coding), reviewed by user
**Context:** cluster `logs/` at ~96G and out of space. Need a safe way to delete periodic
training checkpoints and keep only the best weight per run.

## Added

- **`tools/clean_weights/clean_weights.py`** — Python (stdlib only) tool to prune periodic
  checkpoints.
  - Deletes files matching `state_<digits>.pt` (e.g. `state_80000.pt`), keeps `state_best.pt`.
    Verified in the trainers (`*/utils/training.py`): checkpoints are `state_{epoch}.pt` /
    `state_best.pt`, and every `.pt` bundles `{'model', 'ema'}`, so `state_best.pt` is
    self-sufficient for eval/deploy.
  - **Dry-run by default**; `--apply` to delete; `--root` to override (default
    `~/FMPCC/FM-PCC/logs`).
  - **Best-gated safety**: prunes a dir only if it holds a `state_best.pt`; dirs with numbered
    checkpoints but no best (crashed / in-progress) are skipped and reported.
  - **`--exclude PATH_OR_GLOB` (repeatable)**: protect an unfinished run that will resume
    training. Accepts an absolute path, a path relative to `--root`, or a glob
    (e.g. `*alphaflow*/7`); anything under an excluded dir is kept. Protected files are
    reported on console + logged as `EXCLUDE` lines. Added after user flagged that
    `.../AlphaFlowODE_.../7` is mid-training and must not be pruned.
  - Single `os.walk` computes BEFORE totals, per-top-level-folder sizes, and candidates in one
    pass. Skips/ignores its own run-logs folder. Skips symlinks.
  - Writes a timestamped audit log per run to **`logs/_clean_weights_runlogs/`** (inside the
    training `logs/` dir → gitignored / cluster-side) with **BEFORE** (sizes + free disk),
    **DELETE manifest** (file, size, mtime) + `SKIP-NOBEST` lines, and **AFTER** (freed bytes,
    recomputed totals; projected in dry-run).
- **`tools/clean_weights/README.md`** — usage, options, safety, logging, out-of-scope notes.
- **`logs_in_develop/Clean_Weights/PLAN_clean_weights.md`** — the plan (this dir; dev doc).

## Layout decisions (per user)

- Tool + README live in their **own folder** `tools/clean_weights/`, not loose at repo root.
- `logs_in_develop/` is **only** for AI-coding dev docs → only the plan + this changelog go
  there. The tool's run-logs go into the real training `logs/` tree, never `logs_in_develop/`.

## Verification (local smoke test, AI-coding container)

Ran on a synthetic tree (Python file-tool only — no pipeline):
- Dry-run: reported 2 deletions + 1 skipped no-best dir, deleted nothing. ✓
- `--apply`: removed both `state_<n>.pt`, kept `state_best.pt`, left the no-best dir intact,
  log recorded BEFORE/manifest/AFTER with freed bytes. ✓

**Not yet run on the cluster** — real prune of `logs/` is a user action:
`python tools/clean_weights/clean_weights.py` (dry-run) → review log → `--apply`.

## Out of scope (manual)

D3IL `ddpm_*` baselines (`eval_best_*.pth`/`last_*.pth`, no numbered periodics), `plans/`
rollouts, and `gifs*` dirs are left untouched; prune by hand if more space is needed.

---

## 2026-09-03 — Keep highest-numbered checkpoint (de-facto "latest")

**Date:** 2026-09-03
**Author:** Claude (AI-coding), reviewed by user
**Context:** The old tool deleted ALL `state_<digits>.pt` files, keeping only `state_best.pt`.
Investigation revealed the trainers have **no `state_latest.pt` file** — resume is handled by
`find_latest_checkpoint_step()` in each trainer, which scans for the highest-numbered
`state_<digits>.pt`. Deleting all numbered checkpoints made training un-resumable; only
`state_best.pt` survived (the last *validation improvement*, potentially much older than
where training actually was).

### Changed

- **`tools/clean_weights/clean_weights.py`** — now **keeps the highest-numbered
  `state_<epoch>.pt`** (de-facto latest) per directory alongside `state_best.pt`.
  - `scan()` groups candidates by parent directory, identifies the max epoch per dir,
    and excludes it from the deletion list.
  - New `_epoch_from_name()` helper extracts the epoch number from filenames.
  - Audit log gains a **`KEPT LATEST`** section listing every preserved latest file
    (path, size, epoch number), placed before the DELETE manifest.
  - Console output gains a new summary line showing how many latest files were kept
    and their total size.
  - Docstring updated: documents both `state_best.pt` and `state_<max>.pt` as kept,
    with a "Why keep the latest?" explanation block.
  - Safety section gains a new bullet: **Latest-kept**.

- **`tools/clean_weights/README.md`** — updated to document the new behaviour:
  - "What it keeps vs deletes" table now shows `state_<max_epoch>.pt` as **KEEP**.
  - Explanation paragraph documents why: trainers have no `state_latest.pt` file.
  - Safety list gains item 3 (Latest-kept).
  - Logging section documents the new `KEPT LATEST` log section.

### Root cause

The trainers (`mix_visual_avoiding/utils/training.py`, `training_twotime.py`, etc.) save:
- `state_{epoch}.pt` — periodic, via `save(epoch)` at `step % save_freq == 0`
- `state_best.pt` — via `save_best()` when validation loss improves

There is no `state_latest.pt`. Resume logic (`find_latest_checkpoint_step()`) in each
`train_*.py` file scans `state_*.pt`, extracts the integer, and picks `max()`.

### Behaviour summary (after fix)

| File | Action |
|------|--------|
| `state_best.pt` | **KEEP** — best validation, eval/deploy |
| `state_<max>.pt` (highest epoch per dir) | **KEEP** — de-facto latest, training resume |
| `state_<other>.pt` (all other numbered) | **DELETE** |
| everything else | untouched |
