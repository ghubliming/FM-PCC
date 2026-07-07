# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## ⚠️ CRITICAL ENVIRONMENT RULES — READ FIRST

1. **This Docker container is AI-coding only. NO Python packages are installed.**
   Never attempt to run Python scripts, pytest, or any Python command locally — it will fail.
   All real execution (training, eval, tests) runs on the remote Slurm cluster (**i6-gpu-1**), which has the full FMPCC conda env, GPUs, MuJoCo, and all deps. Code is synced via git.
   → Write code and tests here; when something needs validation, note **"run on cluster"** and let the user run it.

2. **NEVER commit automatically.** The user almost always commits manually. Wait for explicit permission before any `git commit`.
   **Never add a Claude co-author line** (`Co-Authored-By: Claude ...`) to commits.

3. **This repo is under heavy active development** — code is unfinished and buggy in places. Treat everything below as guidance, not fixed truth. Always check the development logs and recent git history for the current state before making assumptions.

## Claude Code persistent memory lives in this repo

Claude's memory files are stored at `.claude/memory/` **in this repo** (so they survive container rebuilds via git). The Claude Code harness reads them from `~/.claude/projects/-workspaces-FM-PCC/memory/`, which is a **symlink** into the repo. After a container rebuild, restore it:

```bash
rm -rf ~/.claude/projects/-workspaces-FM-PCC/memory
ln -s /workspaces/FM-PCC/.claude/memory ~/.claude/projects/-workspaces-FM-PCC/memory
```

If memories seem missing at session start, check this symlink first.

## Where to find the current state (most important)

- **`logs_in_develop/MASTER_TEST_HISTORY.md`** — the master index. Its "Master Trace Map" table maps every research generation (Gen0–Gen11+) to its model folder, test folder, and status (`working on` / `finished` / abandoned). Start here. (Not 100% accurate — cross-check with git history.)
- **`logs_in_develop/`** — the FULL development logs, organized per generation (Gen1…Gen11, Gen3v4_imf, Gen7_FMPCC_Viusal_Aligning, HF_iMF, …). There are many MDs; use MASTER_TEST_HISTORY.md to navigate rather than reading them all.
- **`git log`** — commit messages carry generation tags like `(Gen11 Fix11 & Sync to Gen7/Gen6V4 C4)`.
- **`Slurm_Codes/logs/important_runs/important_runs.md`** — history of important cluster jobs.
- `README.md` is intentionally minimal; don't rely on it alone.

## What this repo is

**FM-PCC (Flow Matching Predictive Control)**: replaces the stochastic diffusion engine of DPCC (Diffusion Predictive Control) with deterministic Flow Matching. Dual-path design:
- **Generative brain**: U-Net predicts a velocity field; an ODE solver generates an unconstrained reference trajectory.
- **Physical brakes (MPC/DPCC projection)**: filters the trajectory to enforce physical/environmental constraints.

The codebase is **majorly based on DPCC at `/workspaces/aux_repo/dpcc`**, mixed with other upstream repos in `/workspaces/aux_repo/` (`d3il`, `diffuser`, `imeanflow`, `mujoco_mpc`, `SafeFlowMPC`, `UAV-Flow`, `drifting_policy`). When behavior is unclear, compare against those upstreams.

## Repo structure convention: generation siblings

Development proceeds by **copy-modify isolation**: each generation/experiment gets its own sibling folder pair at repo root — a model/code folder plus a matching test/eval folder (e.g. `flow_matcher_v3/` ↔ `FM_v3_test/`, `fm_visual_aligning/` ↔ `fm_visual_aligning_test/`, `flow_matcher_v3_uav/` ↔ `FM_v3_uav_test/`). Older generations are kept intact for rollback/A-B comparison; some are archived in `Archived_Codes/` or marked `(Abandoned)`/`legacy`. **Fixes are often mirrored/synced across active sibling generations** (e.g. Gen11 ↔ Gen7 ↔ Gen6V4) — check commit messages for sync patterns before editing only one copy.

Other key areas:
- `config/` — per-task configs (aligning/avoiding D3IL, UAV, eval YAMLs).
- `scripts/` — Gen0 baseline train/eval entry points (`train.py`, `eval.py`); see `TRAINING_CLI_USAGE.md` for CLI options (seeds, etc.). Newer generations have their own scripts inside their folders.
- `Slurm_Codes/` — sbatch scripts, cluster submit pipeline (`submit.sh`), remote log download (`download_remote_logs/export_to_laptop.sh`).
- `Data_Analysis/`, `Results_and_Data_Analysis_Colab_T4/`, `ipynbs_Colab/` — result aggregation, plotting, Colab notebooks.
- `third_party/`, `d3il/` — vendored dependencies.
- `/logs` folders (raw training outputs, weights) are gitignored and exist only on the cluster/local machines.

## Working style

- User prompts may be inaccurate or forget details — verify against `logs_in_develop/` and git history before acting on stale assumptions.
- When adding features/fixes, follow the existing copy-modify sibling pattern rather than refactoring shared code across generations.
- Keep `logs_in_develop/MASTER_TEST_HISTORY.md` in mind when documenting new work (see `logs_in_develop/Prompt_for_auto_update_HISTORY_MD.md` for the update convention).
