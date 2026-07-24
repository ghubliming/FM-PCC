# Fix_11b — force unbuffered stdout so the Fix_11 progress trail actually shows

**Date:** 2026-07-09. Follow-up to [Fix_11](./CHANGELOG_fix11_progress_logging.md).

## Symptom
Running e.g.
```
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh s_curve 6 "" fm_only gif
```
the job log shows **nothing** — no scene/variant/trial breadcrumbs — and it "feels like no
progress." Fix_11 added those `print(...)` lines specifically to make progress visible, so the
absence of output is itself the bug, not (necessarily) a stuck run.

## Root cause: SLURM stdout block-buffering (not a hang)
Under SLURM, python's stdout is a **file, not a TTY**, so CPython uses **block buffering**
(~4–8 KB) instead of line buffering. Every Fix_11 breadcrumb — and the early prints in
`main()` (`seed=`, `n_trials=`) and `eval_scene()` (`cond_mode=`, `variants=`, feasibility
check) — sits in that buffer and only reaches the `.log` when the buffer fills (~50–80 lines)
or the process exits. A slow UAV rollout emits < 4 KB of text for a long time, so the log looks
frozen even while the eval runs fine. If the 24 h `--time` limit later SIGKILLs the job, the
buffered lines are lost entirely — so even the post-mortem "where did it die" trail Fix_11 was
built for never appears.

Verified there was **no** `python -u`, `PYTHONUNBUFFERED`, or `flush=True` anywhere in
`eval_fm_uav.sh` or `eval_fm_uav.py` (nor in any other `Slurm_Codes/sbatch/` script).

## What changed
### `Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh`
- Added `export PYTHONUNBUFFERED=1` in the env-export block (next to `MUJOCO_GL` etc.).
  Equivalent to `python -u`; set as an env var so it also covers any child python the job
  spawns. Makes every `print(...)` appear live in the log — no Python changes needed.

## Not changed
- No change to any Python code, control flow, timing, or output artifacts — this is purely a
  buffering/visibility fix. Fix_11's prints were already correct; they just weren't reaching
  disk in real time.
- Did not touch other sbatch scripts (train, aligning, avoiding). Same one-liner applies if
  their logs ever look "frozen"; left out of this fix to stay scoped to the reported UAV eval.
  - **Follow-up (2026-07-09):** the visual-aligning **eval** launchers (Gen7 FM +
    Gen6V4 diffuser) were later given the same one-liner on request — see
    `logs_in_develop/Gen7_FMPCC_Viusal_Aligning/CHANGELOG_fix11b_unbuffered_stdout_sync.md`.
    Train/pipeline/avoiding/imf scripts remain untouched.

## Verification / next step
- `bash -n Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh` — **run on cluster** (Docker dev env has
  no runtime). Expected: the `[ eval ] seed=…`, `[ eval ] n_trials=…`, then per-variant
  `>>> variant X/N` and per-trial `trial X/N done` lines now stream into the `.log` as the job
  runs, instead of appearing all-at-once (or never).
- If breadcrumbs stream and then genuinely stop advancing, *that* is a real hang and a separate
  investigation — but this fix is what lets you tell the two apart.

## Why corridor "worked" but s_curve looked "gone" (second, INDEPENDENT factor)
The buffering fix explains why you see *nothing*. It does NOT make s_curve as fast as
corridor — s_curve is genuinely far more expensive, for two verified reasons:

1. **Longest episode budget.** `SCENE_MAX_EPISODE_LENGTH`: s_curve = **871** steps vs
   corridor = **396** (~2.2×).
2. **Per-FM-step projector rebuild.** s_curve is the ONLY scene whose halfspaces carry
   `x_active` (config/uav_projection.yaml:206-209, dict form) → `_run_variant` sets
   `_has_x_active=True` → `rebuild_projector` runs a fresh `setup_dpcc_projector` (scipy QP
   build) **every FM step** (~871×/rollout). corridor's halfspaces are plain lists
   (yaml:153-154, no `x_active`) → projector built ONCE per variant.

Multiply by `projection_variants` = **18** (diffuser + 17 DPCC; s_curve keeps the tightened
ones since it has spatial constraints) × `n_trials`=10. So s_curve's DPCC variants do on the
order of 871 × 10 × 17 scipy QP solves — hours to a full day; it can brush/exceed the 24 h
`--time` limit. `diffuser` (variant 1, no projector) is fast and should stream first once
buffering is fixed; the slowdown starts at variant 2 (`gradient`).

**How to tell slow from hung now:** after the fix you should immediately see `seed=`,
`n_trials=`, `cond_mode=`, the feasibility check, `max_episode_length=`, and `>>> variant
1/18: 'diffuser'`, then a `trial X/10 done (…s elapsed, ~…s to go)` line per rollout (Fix_11
ETA). If those appear and advance, it's just slow (the per-step-rebuild path), not hung. If
they appear and freeze on one trial, THAT is a real hang worth a separate look.

**Levers if the ETA blows past 24 h** (user's call, not changed here): trim
`projection_variants`, drop `n_trials`, raise `--time`, or make the s_curve projector rebuild
cheaper/less frequent. Not touched in this fix — flagged for decision.

## Files touched
- `Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh` — one `export PYTHONUNBUFFERED=1`.
