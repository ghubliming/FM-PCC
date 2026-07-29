# clean_weights

Safely prune **periodic training checkpoints** from the `logs/` tree while keeping the
single best weight per run. Frees the bulk of `logs/` (alphaflow, meanflow, diffusion,
UAV_FM, visual arms, …) without losing anything needed for eval/deployment.

- **Runs on the cluster** (where `logs/` lives). Pure Python **stdlib** — no torch/conda.
- **Never run the pipeline locally** in the AI-coding container; this is just a file tool,
  but it operates on the cluster's `logs/`.

## What it keeps vs deletes

Per `<run>/<seed>/` directory the FM/DPCC/diffusion trainers write:

| File | Action |
|------|--------|
| `state_best.pt` (bundles both `model` + `ema`) | **KEEP** |
| `state_<epoch>.pt`, e.g. `state_80000.pt` (periodic snapshot) | **DELETE** |
| `losses.pkl`, `args.json`, `plans/`, `gifs/`, `*.pth` baselines | untouched |

`state_best.pt` alone is enough for eval/deploy — the numbered snapshots only exist to
resume training from an intermediate step.

## Safety

1. **Dry-run by default.** Nothing is deleted unless you pass `--apply`.
2. **Best-gated.** A directory is pruned **only if** it contains a `state_best.pt`.
   Dirs with numbered checkpoints but **no** `state_best.pt` (crashed / still-running jobs)
   are **skipped and reported** — a run never loses its only weights.
3. **Tight match.** Only files matching `state_<digits>.pt` are ever removed
   (`state_best.pt`, `.pth`, `.pkl`, gifs, plans can't match).
4. **Audit log** for every run (see below).

## Usage

```bash
cd ~/FMPCC/FM-PCC

# 1) DRY-RUN — reports before-size, what would be deleted, projected reclaim
python tools/clean_weights/clean_weights.py
#   or an explicit root:
# python tools/clean_weights/clean_weights.py --root ~/FMPCC/FM-PCC/logs

# 2) Review the log:
ls -t logs/_clean_weights_runlogs/ | head
less logs/_clean_weights_runlogs/clean_weights_*.log

# 3) APPLY — actually delete
python tools/clean_weights/clean_weights.py --apply
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--root PATH` | `~/FMPCC/FM-PCC/logs` | logs root to scan |
| `--apply` | off (dry-run) | actually delete |
| `-h/--help` | — | help |

## Logging

Each invocation writes `logs/_clean_weights_runlogs/clean_weights_<YYYYMMDD_HHMMSS>.log`
(inside the training `logs/` dir, so it's gitignored / cluster-side) with:

- **BEFORE** — total size of the root, free disk, per-top-level-folder sizes.
- **DELETE manifest** — every file with size + mtime; `SKIP-NOBEST` lines for the best-less
  dirs that were left alone.
- **AFTER** — deleted count, freed bytes, recomputed total + free disk (in `--apply`;
  projected in dry-run).

## Not handled (do manually if wanted)

- **D3IL baselines** (`ddpm_*`, `eval_best_*.pth` / `last_*.pth`) — different convention,
  no numbered periodics; left alone.
- **`plans/` rollouts and `gifs*` dirs** — outputs, not weights; delete by hand to reclaim.
- **No-`state_best` dirs** — to prune while keeping the latest snapshot, per dir:
  `ls -v "$d"/state_[0-9]*.pt | head -n -1 | xargs -r rm -f`
