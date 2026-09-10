# `Slurm_Codes/temp_bash/` — one-off submission scripts

Throwaway submit drivers for a specific wave of jobs. **Not** part of the pipeline: the real entry
points stay in `Slurm_Codes/sbatch/` and are invoked through `Slurm_Codes/submit.sh`.

## Why these exist at all

Long `UAV_MIX_VARIANTS` strings pasted into a terminal **break**. The shell wraps around column
~160, and a line break landing inside the quotes splits a variant name — `dpcc-r-\n  geo_free` or
`dpcc-r-  geo_free`. `.strip()` in `mix_uav_test/eval_mix_uav.py` only trims the ENDS of a name, so
the mangled token fails the variant check and the job exits 2. Jobs **25555** and **25582** died
exactly that way, and **25580** died from a related paste accident (the env assignments and
`submit.sh` became two separate commands, so nothing was exported at all).

A script keeps every line short and gets the list right once.

## Convention

* one file per wave, named `<verb>_<YYYYMMDD>_<what>.sh`
* build variant lists as bash **arrays**, one name per line, joined at runtime — never a long literal
* **validate before submitting**: name shape, count, no whitespace, and any rule the eval enforces
  (e.g. HardFlow needs a `dpcc-*` companion at the same K; `hardflow_*` is degenerate at K < 3)
* always set `UAV_EVAL_HOURS=24` — the default is `N_SEEDS × 8` = **8 h** and it silently truncates
  (job 25553 died at 8:00:10 with a variant at trial 7/10)
* locate the repo root by walking up for `Slurm_Codes/submit.sh`, so the script runs from anywhere
* run with `bash Slurm_Codes/temp_bash/<script>.sh`

## Contents

| script | wave |
|---|---|
| `resubmit_20260909_fixes.sh` | af `pillars` K=5 resume + fm `s_curve` K=20 tail (jobs 25586/25587) |
| `eval_20260910_corridor_ball.sh` | U11 `corridor_ball`, 3 engines × 2 tiers (jobs 25599–25604) |
| `submit_20260910_diffusion_baseline_and_scurve_mirror.sh` | pillars diffusion baseline + the af/diffusion rows missing from `s_curve` under `u7hg` |

Safe to delete once a wave has landed and its results are in a DA.
