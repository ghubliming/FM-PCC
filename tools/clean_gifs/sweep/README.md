# clean_gifs / sweep

Tree-wide GIF thinning for the whole `logs/` tree, with a **percentage** keep policy and a
backup-vs-live split. Companion to [`../clean_gifs.py`](../clean_gifs.py), not a replacement —
see *Which one do I want?* below.

- **Runs on the cluster** (where `logs/` lives). Pure bash + coreutils — no Python, no conda.
- **Never run the pipeline locally** in the AI-coding container; this is a file tool, but it
  operates on the cluster's `logs/`.

## Which one do I want?

| | `../clean_gifs.py` | `sweep/clean_gifs_sweep.sh` (this) |
| :-- | :-- | :-- |
| scope | **one run dir**, given explicitly | the **whole `logs/` tree**, found automatically |
| keep policy | `--keep-per-dir N` — a fixed count, first N | **10% per directory**, evenly spaced (live dirs); 1 per directory (backup dirs) |
| backup vs live | no distinction | **two phases**, run and reported separately |
| re-run safety | re-runs re-thin the survivors | **stamped**, a second `--apply` is a no-op |
| also handles MP4 | yes, `--ext` | gif only |

Rule of thumb: **tidying one run → `clean_gifs.py`. Reclaiming tens of GiB across everything →
this.** The fixed-count policy does not scale across the tree — `--keep-per-dir 2` on a
1080-rollout cell throws away 99.8% of it, while the same 2 on a 3-rollout cell keeps almost
everything. A percentage keeps the sample proportional to how much was recorded.

## Why it exists

`logs/` reached **97.6 GiB**, of which `.gif` was **35.3 GiB — 19,210 files, 36% of the tree**,
larger than every checkpoint combined. On 2026-09-19 the Gen15 U17 pillars wave died of

```
OSError: [Errno 28] No space left on device
```

with 2.9 GiB free: eleven of its twelve child jobs never started at all, because Slurm could not
create their log files. GIFs are rollout **renders** — no metric reads them, `eval_mix_uav.py`
times only the policy call, and they regenerate with `record=gif`. They were the cheapest ~30 GiB
available. Post-mortem: `logs_in_develop/Writing/Working_Space/data_status/SLURM_RUNBOOK_20260919_pillars_enlarged.md` §3c.

## The policy

| phase | target | keep |
| :-- | :-- | :-- |
| **1** | gifs inside `(Bf_*)` / `(Archive*)` / `(smoke*)` / `(legacy*)` / `(Outdated*)` / `(Abandoned*)` dirs | **1 per directory** |
| **2** | gifs in every other (live) directory | **10% per directory**, evenly spaced, never fewer than 1 |

Phase 1 targets the `(Bf_…)` "before fix X" rollback snapshots, which are dead by repo convention
and — grepped on 2026-09-19 — referenced nowhere in `Data_Analysis/DA_in_Paper/` or
`logs_in_develop/Writing/`. Measured on the real tree: **433 directories, 4,934 gifs, 15.07 GiB,
of which 13.34 GiB is reclaimable.**

Selection is deterministic (version-sorted so `rollout_2` precedes `rollout_10`, then evenly
spaced), so the `--apply` run deletes exactly what the check run listed.

### Never touched

| path | why |
| :-- | :-- |
| `uav_expert_data*` | the expert **demonstration** renders (~4 GiB). Author's call 2026-09-19: any of them may still become a thesis figure |
| `*/expert_references/*` | the one gif class a DA figure sources — `DA_in_Paper/plotting/sources.py:300,311` (`fig_aligning_camera_overhead` / `_wrist`) |
| anything outside `logs/` | the script resolves its own root and refuses any path not ending in `FM-PCC/logs` |

## Usage

```bash
cd ~/FMPCC/FM-PCC

# 1) CHECK — the default. Deletes nothing; prints per phase how many files and bytes would go.
bash tools/clean_gifs/sweep/clean_gifs_sweep.sh
bash tools/clean_gifs/sweep/clean_gifs_sweep.sh --phase 1     # one phase only

# 2) APPLY — after reading the check output
bash tools/clean_gifs/sweep/clean_gifs_sweep.sh --phase 1 --apply
bash tools/clean_gifs/sweep/clean_gifs_sweep.sh --phase 2 --apply
```

Do phase 1 first and re-check `df -h`. It is the dead-folder half and usually enough on its own,
which lets you keep every live render.

## Safety

1. **Dry-run by default.** Nothing is deleted without `--apply`.
2. **Root is derived, not given, and is checked.** It walks up for `Slurm_Codes/submit.sh`, then
   refuses any root not matching `*/FM-PCC/logs`.
3. **Idempotent.** Each `--apply` writes a `.gifs_thinned` marker into every directory it touched,
   and stamped directories are skipped forever after. Without this a second `--apply` would thin
   the *survivors* — phase 2 keeping 10% twice leaves 1%. The check run reports
   `already thinned: N`.
4. **Manifest per run** at `logs/_clean_gifs_runlogs/<timestamp>_phase<N>.manifest`, one deleted
   path per line, so "where did that gif go" has an answer later.
5. **Empty directories are left in place** — they cost nothing and preserve the tree shape.

## Known limits

- Deletes `.gif` only. For the MP4 siblings use `../clean_gifs.py --ext mp4`.
- The `.gifs_thinned` marker makes a directory permanently out of scope. If a cell is re-run and
  records fresh gifs that you later want thinned, delete its marker first.
- Phase classification is by path substring. A directory whose name happens to contain `(legacy`
  is treated as a backup dir.
