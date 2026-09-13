# U16 — corridor_v2 paper evaluation: setup and submission (final U16 stage)

**Date:** 2026-09-13 · **Gen:** 15 · **Bench:** U16 `corridor_v2_slide` + fixes 1/2 ([`CHANGELOG_20260913_u16_fix1_fix2_FULL_REVIEW.md`](CHANGELOG_20260913_u16_fix1_fix2_FULL_REVIEW.md))
**Decision (user):** no further geometry or projector changes. Run the full evaluation on the bench that passed fix 2.
Metrics are identical to pillars / s_curve.

## 1. What changed in the repo

| file | change |
| :-- | :-- |
| `Slurm_Codes/temp_bash/eval_20260913_u16_corridor_v2_paper.sh` | **new** — `plan` (default, submits nothing) / `smoke` / `submit` |
| `Data_Analysis/DA_UAV_v1/config.py` | the paper-run arm names added to `VARIANT_ORDER` and `MAJOR_VARIANTS` (headline table). Discovery needs no list; this sets row order and headline inclusion only |

No eval, projector, config or scene change. `config/uav_projection.yaml` and all U16 fix code are untouched.

## 2. Fixed for every arm

| | |
| :-- | :-- |
| geometry | `corridor_v2_slide` (`scene_corridor_v2.xml`, walls ±1.0, 14° slide, x_active [−2, 2]) |
| normaliser | `FMPCC_SAFE_EPS_FRAC=1.0`, `FMPCC_SAFE_EPS_MODE=scaled` |
| projector | `-bounds_free-pdes-tightened`: geometry on the setpoint, DPCC margin 0.025 m, action cap off. HardFlow gets the **same** stack, so the HF-vs-DPCC comparison is constraint-matched |
| threshold | `diffusion_timestep_threshold: 0.5` (checked in pre-flight) |
| metrics | strict success, S&C, violations on the real drone (unchanged) |
| eval tag | `u16cv2` → `E<engine>_K<k>_mpc4_pid_stopgo_T0.5_u16cv2/`, never pooled with the U14–U16 injections (`u7hg`) |
| trials | 12 per (engine, K, seed, variant) = 4 per route (L, C, R) |
| walls | `UAV_EVAL_HOURS=24` per job |

## 3. Arms and job layout (per seed)

**Name rule:** `-tightened` is always the **last** token. `DA_UAV_v1.discovery.variant_parts` reads tightening with
`endswith`. The eval allow-list, the projector toggles and HardFlow's batch-size rule accept the tokens in any order.
Checked on the extracted logic for every arm: allow-list core, `-pdes` and `-tightened` flags, selection rule,
HardFlow B, result-folder label, DA split.

| engine | K | submission | variants |
| :-- | :-- | :-- | :-- |
| mf / fm / af | 1 | 1 job | diffuser, dpcc-r/c/t-bounds_free-pdes-tightened (HardFlow degenerate at K ≤ 2) |
| mf / fm / af | 3, 5 | part A (2 children) | diffuser, dpcc-r-…, dpcc-c-… |
| mf / fm / af | 3, 5 | part B (2 children) | dpcc-t-…, hardflow_new-bounds_free-pdes-tightened (B = 1), hardflow_new-t-bounds_free-pdes-tightened (B = 4) |
| diffusion | 20 (plan block) | 2 jobs (A: diffuser, dpcc-r · B: dpcc-c, dpcc-t) via `eval_mix_uav.sh` | no HardFlow (no velocity field) |
| diffusion, no checkpoint | 20 | 1 train+eval via `uav_mix_ksweep_pipeline.sh` | the four arms in one eval job |

- **af knobs:** `UAV_MIX_BONE_AF=unet UAV_MIX_AF_ALPHA_END=0.2 UAV_MIX_EPOCH=latest`, as in campaign T2 (the corridor af_unet checkpoint).
- **HardFlow is never submitted without its dpcc row**, per the eval's own HardFlow-only guard.

Dry run of `plan` mode here (fake mf/fm/af checkpoint dirs, removed afterwards): 10 submissions per seed
(9 flow + 1 diffusion train+eval). The K = 3/5 parts each launch 2 children.

## 4. Pre-flight checks inside the script

Geo entry present; scene XML present; `-pdes` toggle, `update_constraint_list` and the composed-toggle allow-list
present in the code; threshold 0.5; DA names registered (warning only). Per (engine, seed) the script looks for a
checkpoint dir `logs/UAV_MIX/uav-corridor/mix_uav_<engine>/*/<seed>`:
- flow engine missing → skipped, with the train command printed;
- diffusion missing → train + eval.

## 5. Known limits, reported as-is (not fixed)

- **Routes C and R cannot meet strict success on this slide.** The drone is pushed to about −0.37 and the corridor
  model never learned to steer back sideways. Expected S&C for projected arms is ≤ ~1/3 of trials (route L).
- **HardFlow may lose 1–2 mm at the corridor exit** (fix 2: one step per route at −1.5 mm), so its collision-free rate may fall below 1.0.
- **Engine ranking is not guaranteed.** The projector repairs constraints for every arm, so engines are expected to
  separate on steps, compute and unprojected-plan violations rather than on constraint satisfaction.
- **Diffusion:** no corridor diffusion checkpoint or eval exists in any local drop or log (0 files). The train+eval
  path is automatic, but its eval runs the four arms in one 24 h job.
- Multi-seed only if checkpoints exist for those seeds; otherwise those seeds are skipped with a message.

## 6. Order of operations

1. `bash …/eval_20260913_u16_corridor_v2_paper.sh` → read the plan and the checkpoint lines.
2. `bash … smoke` → one job: mf, K = 3, 3 trials (L, C, R), GIF on, tag `u16smoke`. Pass if both projected arms are collision-free on all three routes.
3. `bash … submit` → the wave.
4. DA once everything has finished (command printed by `submit`), scanning only the `_u16cv2` result folders.

## 7. Run record

| step | job(s) | status |
| :-- | :-- | :-- |
| smoke | — | not yet submitted |
| submit | — | not yet submitted |
