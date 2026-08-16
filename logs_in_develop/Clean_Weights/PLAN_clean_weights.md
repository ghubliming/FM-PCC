# Plan — Safely Prune Training Weights (keep only `state_best.pt`)

**Status:** PLAN ONLY — under review. No tool has been written yet; nothing has run.

**Goal:** `logs/` on the cluster is ~96G and out of space. Reclaim it by deleting the
periodic checkpoint snapshots while keeping the single best weight per run.

**Runs on:** remote cluster (`i6-gpu-1`), where `logs/` actually lives. Never in the
AI-coding container. Implemented in **Python (stdlib only** — `os`, `argparse`, `pathlib`,
`re`, `datetime`, `shutil`); no torch/conda deps, so any Python 3 on the cluster runs it.

---

## Deliverables & layout (to be created after this plan is approved)

`logs_in_develop/` holds **only AI-coding dev docs** — so *this plan* lives there, and
nothing else from this tool does. The tool's run-logs go into the real training `logs/`
tree (cluster-side, already gitignored).

| Path | What |
|------|------|
| `tools/clean_weights/clean_weights.py` | the tool, in **its own folder** under the FM-PCC repo (`~/FMPCC/FM-PCC/tools/clean_weights/` on the cluster). Syncs via git. |
| `tools/clean_weights/README.md` | usage / options / examples, in the same folder. |
| `logs/_clean_weights_runlogs/` | **subfolder inside the training `logs/` dir that collects every run's log** (one timestamped file per invocation). Cluster-side, gitignored with the rest of `logs/`. |
| `logs_in_develop/Clean_Weights/PLAN_clean_weights.md` | this plan (dev doc — the only thing that belongs under `logs_in_develop/`). |

---

## What gets kept vs deleted

The FM / DPCC / diffusion trainers (`flow_matcher_v3_*`, `diffuser`, etc.) save, per
`<run>/<seed>/` directory:

| File | Meaning | Action |
|------|---------|--------|
| `state_best.pt` | best checkpoint, bundles **both** `model` and `ema` weights | **KEEP** |
| `state_<epoch>.pt` (e.g. `state_80000.pt`) | periodic snapshot every N steps | **DELETE** |
| `losses.pkl`, `args.json`, `plans/`, `gifs/` | metrics / config / rollouts | untouched |

Verified in the trainers (`*/utils/training.py`): checkpoints are written as
`state_{epoch}.pt` and the best as `state_best.pt`, and **every** file already contains
`{'model': ..., 'ema': ...}`. So `state_best.pt` is fully self-sufficient — deleting the
numbered snapshots loses nothing needed for eval/deployment, only the ability to resume
training from an intermediate step.

Your example is exactly the intended behavior:
- keep `.../alphaflow/.../7/state_best.pt`
- delete `.../alphaflow/.../7/state_80000.pt` (and all other `state_<n>.pt`)

### Biggest wins (from your `du`)
`flow_matching_v3_meanflow` (13G), `flow_matching_v3_alphaflow` (13G), `UAV_FM` (17G),
`avoiding-d3il/diffusion` (3.6G), the visual arms, and `Gen11E6_init_test` (1.4G). These are
dominated by `state_<n>.pt` files and are the primary target. (`plans/` dirs are rollouts,
not weights — untouched.)

---

## Safety design

1. **Dry-run by default.** Deletes nothing unless `--apply` is passed. First run only reports
   and writes the log.
2. **Best-gated.** A directory is pruned **only if** it contains a `state_best.pt`.
   Directories with numbered checkpoints but **no** `state_best.pt` (crashed / still-running
   jobs) are **skipped and reported** — they never lose their only weights.
3. **Tight match.** Only files matching the regex `state_<digits>.pt` are ever removed.
   `state_best.pt`, `.pth` baselines, `.pkl`, gifs, and plan outputs cannot match.
4. **Full audit log** (see below).

---

## Logging (per requirements)

Every invocation writes one file to `logs/_clean_weights_runlogs/` (inside the training
`logs/` dir, gitignored), named `clean_weights_<YYYYMMDD_HHMMSS>.log`, containing:

- **Header** — run timestamp, `LOGROOT`, mode (`DRY-RUN` / `APPLY`).
- **BEFORE** — total size of `LOGROOT` and free disk space (`df`), plus per-top-level-folder
  sizes so you can see where the space sits.
- **DELETED manifest** — every file, with its size and a timestamp; `SKIP-NOBEST` lines for
  the best-less dirs that were left alone.
- **AFTER** — recomputed total size, free disk, and **reclaimed bytes** (in `--apply` mode;
  in dry-run this is the *projected* reclaimable amount).
- **Summary** — counts (deleted / skipped) and human-readable totals.

Console output mirrors the summary; the full manifest lives in the log file.

---

## Procedure (run on cluster, after approval)

```bash
cd ~/FMPCC/FM-PCC

# 1) DRY-RUN — reports before-size, what would be deleted, projected reclaim
python tools/clean_weights/clean_weights.py                  # default root ~/FMPCC/FM-PCC/logs
#   or explicit root:
# python tools/clean_weights/clean_weights.py --root ~/FMPCC/FM-PCC/logs

# 2) Review the log (before/after/manifest/skip-list):
ls -t logs/_clean_weights_runlogs/ | head
less logs/_clean_weights_runlogs/clean_weights_*.log

# 3) APPLY — actually delete; log records before + after + freed
python tools/clean_weights/clean_weights.py --apply
```

Planned CLI: `--root <path>` (default `~/FMPCC/FM-PCC/logs`), `--apply` (default dry-run),
`-h/--help`.

---

## Scope notes / not touched

- **D3IL visual baselines** (`d3il_visual_aligning_baseline*`, the `ddpm_*` folders) use a
  different convention: `eval_best_*.pth` + `last_*.pth`, no numbered periodics — left alone.
- **`plans/` folders** are evaluation rollouts (`.pkl`/results), not weights — untouched.
  Reclaim those (e.g. `avoiding-d3il/plans` 2.1G, the `uav_expert_data*/gifs*` dirs) by hand
  if wanted; outside this tool's remit.
- **Archived / abandoned run folders** (`(Archive...)`, `(smoke_run)`, `legacy`) are pruned
  like any other *only if* they contain `state_best.pt`. Delete whole dead folders manually
  if you'd rather.

---

## Optional (manual): prune the no-`state_best` dirs too

If, after review, an unfinished run's numbered checkpoints are also disposable but you want to
keep its latest snapshot, do it per dir by hand — kept out of the automated path so the tool
never deletes a run's only weights:

```bash
d=".../<run>/<seed>"
ls -v "$d"/state_[0-9]*.pt | head -n -1 | xargs -r rm -f   # keep highest-epoch, delete rest
```

---

## Open questions for you before I build the tool

1. **Folder name** — `tools/clean_weights/` OK for the tool + its `README.md`?
2. **Run-logs folder name** — `logs/_clean_weights_runlogs/` OK? (inside `logs/`, so it's
   already gitignored / cluster-side; not in `logs_in_develop/`).
3. Anything else to auto-handle vs. leave manual (e.g. the `plans/`/`gifs*` reclaim)?
