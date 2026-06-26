# Gen11 E4 U10 — Stress-Test Collection: Changelog

**Date:** 2026-06-12
**Branch:** update_into_FM
**Status:** COMPLETE — code implemented, pending first cluster run

---

## Files changed

| File | Change |
|------|--------|
| `uav_expert_data_collect/stress_trajectories.py` | New — 8 stress case builders + dispatcher |
| `uav_expert_data_collect/collect_stress.py` | New — stress collection driver |
| `Slurm_Codes/sbatch/uav_expert_data/collect_stress.sh` | New — CPU-only sbatch |
| `uav_expert_data_collect/stats_validator.py` | Guard: skip `'stress': True` episodes |

---

## C1 — `stress_trajectories.py` (new)

Exports `build_stress_traj(case, scene, rng) → (traj_fn, init_pos, duration, extras)`.
`extras` always contains `kp_scale` and `kd_scale` (1.0 for all non-gain_extreme cases).

| Case | Scenes | What it does |
|------|--------|-------------|
| `extreme_speed` | pillars, s_curve, corridor | Validated path at T=2–4 s (3–8× overspeed) |
| `tight_fillet` | pillars | LRL blend radius r=0.05 m (deliberate Fix_1 reproduction) |
| `discontinuous` | empty, corridor, pillars | `p_des` teleports 2–3 m at t=3 s and t=6 s |
| `wall_crossing` | corridor, pillars | `traverse_line` through wall/pillar axis |
| `floor_dive` | empty, corridor | `p_des` descends to z=−0.5 (below floor) |
| `ceiling_climb` | empty | Rapid climb to z=4.5 in 2 s |
| `gain_extreme` | corridor, empty | Normal path; kp×5 or kp×0.1 (randomly; stored in extras) |
| `degenerate_hover` | empty | `hover_at` for 2 s — static episode |

## C2 — `collect_stress.py` (new)

- `--cases` required (no default) — the "default off" gate.
- Runs `run_stress_episode()`: same MuJoCo physics loop as `generator.run_trial()` but
  **no rejection** — all episodes saved.
- Computes gate verdicts `{'contact': bool, 'floor': bool}` without acting on them.
- Output root: `logs/uav_expert_data_stress/<scene>/<stress_case>/`
- Episode IDs: `stress_{case}_{scene}_{seed:07d}` — seed recoverable from last token.
- Stress metadata merged into episode after `rollout_to_episode()`:
  - `episode['stress'] = True`
  - `episode['stress_case'] = case`
  - `episode['gate_verdicts'] = {contact, floor}`
  - `episode['metadata']['kp_scale']`, `kd_scale`, `gain_variant_label`
- Writes `stress_summary.json` at root on completion.

## C3 — `collect_stress.sh` (new)

CPU-only (`MUJOCO_GL=disabled`). `$1=cases` (required), `$2=scene`, `$3=n_per_case`,
`$4=seed`. 1-hour time limit (full `--cases all` run is ~50–60 episodes, ~5 min).

## C4 — `stats_validator.py` guard

```python
if ep.get('stress', False):
    continue   # E4 U10: skip stress episodes — not training data
```

Applied inside `load_episodes()` before appending to the list. Belt-and-braces: the
validator is normally pointed at `logs/uav_expert_data/` (not the stress root) so this
guard is a secondary protection against accidental mixing.

---

## Invariants preserved

- `collect.py`, `generator.py`, `trajectories.py`, `verify_blends.py` — **unchanged**.
- Production output path `logs/uav_expert_data/` — **untouched** by any new code.
- Stress root is created only when `collect_stress.py` actually runs.

---

## First-run commands

```bash
# Smoke: wall_crossing + floor_dive only (quickest gate-verdict test)
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect_stress.sh \
    wall_crossing,floor_dive

# Full run
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect_stress.sh all
```

Then check `stress_summary.json` and pass the root to E5 U4 rendering (see pair plan).
