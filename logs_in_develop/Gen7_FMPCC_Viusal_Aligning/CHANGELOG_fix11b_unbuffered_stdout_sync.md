# Fix_11b sync — force unbuffered stdout in the visual-aligning eval sbatch scripts (Gen7 + Gen6V4)

**Date:** 2026-07-09. Sync of
`logs_in_develop/Gen11/Epoch9_PCC_Constraints/Fix_11/CHANGELOG_fix11b_unbuffered_stdout.md`
(UAV origin) into visual-aligning. The original fix deliberately stayed scoped to the UAV eval
sbatch and explicitly left aligning untouched; this extends it now that the same real-time-log
behaviour is wanted for the aligning eval jobs.

## Why aligning needs its own patch (not covered by a shared module)
Each generation launches eval through its **own** sbatch script — there is no shared launcher —
so the `export PYTHONUNBUFFERED=1` from the UAV script does not propagate. Verified: before
this change, `grep -rln 'PYTHONUNBUFFERED\|python -u' Slurm_Codes/sbatch/` matched **only**
`uav_fm/eval_fm_uav.sh`. The two visual-aligning eval launchers had no unbuffering.

## Same root cause as the UAV Fix_11b
Under SLURM python's stdout is a file, not a TTY → CPython block-buffers (~4-8 KB), so the eval
scripts' progress `print(...)` breadcrumbs sit in the buffer and only reach the `.log` when it
fills or the process exits. The log looks frozen while eval runs fine, and a SIGKILL at the
`--time` limit loses the buffered trail entirely.

## What changed (sbatch only — no Python touched)
- `Slurm_Codes/sbatch/fm_visual_aligning/eval_fm_visual_aligning.sh` (**Gen7**, FM) —
  added `export PYTHONUNBUFFERED=1` in the env-export block, right after `MPLBACKEND="agg"`.
- `Slurm_Codes/sbatch/diffuser_visual_aligning/eval_visual_aligning_dpcc.sh` (**Gen6V4**,
  diffuser) — same one-line addition, same location.

Both mirror the UAV placement exactly (equivalent to `python -u`; set as an env var so it also
covers any child python the job spawns). No change to any Python, control flow, timing, or
output artifacts — purely a buffering/visibility fix.

## Deliberately NOT touched (kept scoped, mirroring the original)
- Train / pipeline sbatch scripts for aligning (`train_*`, `*_pipeline*`), and the
  avoiding / imf / d3il-baseline scripts. The same one-liner applies if any of their logs ever
  look "frozen," but this stays scoped to the aligning **eval** jobs the request named. Add on
  request.
- No `-tightened`/variant or eval-code changes here — unrelated to this fix.

## Verification / next step
- `bash -n` both scripts — **run on cluster** (Docker dev env has no runtime). Expected: the
  aligning eval breadcrumbs (per-scene / per-variant / per-trial progress) now stream into the
  `.log` live instead of appearing all-at-once or never.
- If breadcrumbs stream then genuinely stop advancing, that's a real hang and a separate
  investigation — this fix is what lets you tell slow from hung.

## Files touched
- `Slurm_Codes/sbatch/fm_visual_aligning/eval_fm_visual_aligning.sh` — one `export PYTHONUNBUFFERED=1`.
- `Slurm_Codes/sbatch/diffuser_visual_aligning/eval_visual_aligning_dpcc.sh` — one `export PYTHONUNBUFFERED=1`.
