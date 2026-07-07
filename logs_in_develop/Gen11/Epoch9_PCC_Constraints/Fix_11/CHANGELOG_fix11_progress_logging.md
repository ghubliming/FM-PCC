# Fix_11 — UAV eval progress logging (seed/scene/geo-entry/variant/trial breadcrumbs)

**Date:** 2026-07-06. User complaint: UAV eval logs are too concise to tell how much work is
left or where a run was when it got cut off — relevant because these jobs already sometimes
hit the 24h SLURM time limit (`Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh`), and the old logs
only printed one line per projection variant, AFTER all of its trials finished. If the job
died mid-variant (or mid-trial), the log just stopped with no indication of where.

## Is this feasible / does it make sense? Yes — assessed before implementing

Traced the actual nesting of one UAV eval job (not guessed):
```
for seed in $SEEDS                              (bash loop, eval_fm_uav.sh)
  for scene in scenes                           (main(), only >1 iteration if --scene all)
    for geo_entry in active_geo_variants matches (eval_scene(), usually 1 entry)
      for variant in projection_variants        (eval_scene(), up to 17 variants, e.g. pillars)
        for trial in range(n_trials)             (_run_variant(), e.g. 10-30 rollouts)
          rollout_one(...)
```
Every level already had the data needed to report an index/total — `len(scenes)`,
`len(_entries)`, `len(config['projection_variants'])`, `args.n_trials` were all already local
variables at the right scope. This is pure additive `print(...)` statements layered onto an
existing loop structure — no control-flow or data changes, so risk is low and the "how much /
where" ask is directly answerable without restructuring anything.

## What changed

### `FM_v3_uav_test/eval_fm_uav.py`
- **`main()`**: scene loop now prints `[ eval ] ══ scene X/N: 'name' ══` before each
  `eval_scene(...)` call (only visible when `--scene all`, i.e. `len(scenes) > 1`).
- **`eval_scene()`**:
  - Geo-entry loop: prints `[ eval ] {scene}: geo entry X/N: 'name'` when a scene has more
    than one active `geo_constraint_variants` match (Fix_6's multi-geo-variant case).
  - Variant loop: prints `[ eval ] {scene} [geo_tag=...] >>> variant X/N: 'name'
    (n_trials=...)` right before calling `_run_variant`, for every variant — this is the line
    to `grep` for after a timeout: the last one printed is the variant that was running (or
    had just finished) when the job died.
- **`_run_variant()`**: the trial loop now prints one line per completed rollout:
  `[ eval ] {scene} variant={variant}: trial X/N done  (Ys elapsed this variant, ~Zs to go)`
  — elapsed/ETA computed from a per-variant wall-clock start (`_variant_t0 = time.time()`),
  so from the log alone you can extrapolate whether the remaining variants/trials will fit
  inside the time limit.

### `Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh`
- The `for seed in $SEEDS` loop now prints `[ uav_fm_eval ] seed X/N: scene=... seed=...`
  instead of just `scene=... seed=...` — the outermost level of the breadcrumb trail, useful
  for multi-seed jobs (`SEEDS="6 7 8 9 10"`).

## Verification
- `py_compile` clean on `eval_fm_uav.py`.
- `bash -n` clean on `eval_fm_uav.sh`.
- Dry-ran the elapsed/ETA print-formatting logic standalone (pure Python, no torch/MuJoCo
  needed) against a synthetic 5-trial loop — elapsed increases monotonically, ETA decreases
  monotonically to ~0 at the last trial, no crashes.
- Full SLURM/cluster execution untested here (no torch/MuJoCo runtime in this environment) —
  this change is print-only, touching no control flow, data shapes, or return values, so the
  existing (already-fixed) KeyError-free path is unaffected.

## What did NOT change
- No logic, timing behavior, output artifacts (JSON/NPZ/GIF), or return values — every new
  line is a `print(...)` layered onto an existing loop, nothing was restructured.
- No new SLURM flags or time-limit changes — this only makes it OBSERVABLE where a run is
  when/if it hits the existing 24h limit, it doesn't change the limit itself.

## Files touched
- `FM_v3_uav_test/eval_fm_uav.py` — `main()`, `eval_scene()`, `_run_variant()`.
- `Slurm_Codes/sbatch/uav_fm/eval_fm_uav.sh` — the seed loop.
