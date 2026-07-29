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
