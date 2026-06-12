# Gen11 E5 U4 — Stress-Test Rendering (Plan)

**Date:** 2026-06-12
**Status:** PLAN ONLY — implementation by a separate agent
**Pair unit:** [`../../Epoch4_expert_data/U10_Stress_Tests/PLAN.md`](../../Epoch4_expert_data/U10_Stress_Tests/PLAN.md) (read first — defines the stress dataset)

---

## 1. Goal

When commanded (default OFF), render the E4 U10 stress episodes as:
- **trajectory GIFs** (`generate_trajectory_gifs.py` — state injection, bp + FPV panels)
- **physics GIFs** (`generate_physics_gifs.py` — live re-simulation, CONTACT overlays,
  proximity bar)

so each stress case can be visually inspected. The renderers themselves are under test:
they must survive teleporting states, in-wall positions, floor penetration, and static
episodes.

---

## 2. Why this is mostly already supported

Both scripts already take `--data-dir`. The U10 layout puts `stress_case` in the
homotopy slot (`<scene>/<stress_case>/ep.pkl`), so:

- `discover_episodes()` works unchanged,
- `--per-homotopy 1` gives exactly **1 GIF per (scene, stress case)** — the selective
  inspection pattern from U2 carries over for free.

## 3. The one real gap — physics replay trajectory reconstruction

`generate_physics_gifs.py` re-builds the trajectory via
`_build_traj_and_init(scene, homotopy, rng)`. For a stress pickle, `homotopy` is a
stress-case name → this call would either crash or rebuild the **wrong (normal)**
trajectory.

**Fix:** dispatch on the pickle's `stress_case` field:

```python
if episode.get('stress', False):
    traj_fn, init_pos, dur = build_stress_traj(episode['stress_case'],
                                               episode['scene'], rng)
else:
    traj_fn, init_pos, dur = _build_traj_and_init(scene, homotopy, rng)
```

Seed recovery is unchanged (`int(ep_id.split('_')[-1])` — U10 keeps the `_{seed:07d}`
suffix). `gain_extreme` episodes must also store their gain scales in metadata so the
replay can rebuild the same PID (extend `_make_pid` call accordingly).

## 4. Trajectory-GIF script changes

Minimal. State injection reads `obs`/`q` directly from the pickle — no trajectory
rebuild needed. Required hardening only:

- Skip-list: add `gifs`, `gifs_physics` exclusions relative to the stress root (same
  pattern as production).
- Tolerate out-of-bounds states (drone inside a wall / below floor): `mj_forward` on an
  injected penetrating state is legal in MuJoCo — verify no exception path assumes
  contact-free states.
- `degenerate_hover`: episodes may be very short / static — guard against 0–1 frame GIFs
  (emit the WARN, don't crash the batch).

## 5. CLI / sbatch

No new flags needed beyond what exists — stress rendering is invoked by pointing
`--data-dir` at the stress root:

```bash
# Trajectory GIFs — 1 per (scene, case), stride 5
python uav_expert_data_collect/generate_trajectory_gifs.py \
    --data-dir logs/uav_expert_data_stress --per-homotopy 1 --frame-stride 5

# Physics GIFs — same selection
python uav_expert_data_collect/generate_physics_gifs.py \
    --data-dir logs/uav_expert_data_stress --per-homotopy 1 --frame-stride 5
```

sbatch: add `$6 = data_dir` (optional, default production root) to both
`generate_gifs.sh` and `generate_physics_gifs.sh`:

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_gifs.sh \
    "" "" "" 5 1 logs/uav_expert_data_stress
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/generate_physics_gifs.sh \
    "" "" "" 5 1 logs/uav_expert_data_stress
```

Default behaviour with `$6` omitted is byte-identical to today — "default not" holds.

Outputs land under the stress root (`gifs/`, `gifs_physics/`), never in the production
GIF tree.

## 6. What to look for, per stress case

| Case | Trajectory GIF expectation | Physics GIF expectation |
|------|---------------------------|------------------------|
| `extreme_speed` | path correct, `p` lags `p_des` visibly | heavy clip → unstable / floor crash; red border if obstacle hit |
| `tight_fillet` | sharp fillet visible at corners | reproduction of the Fix_1 failure: saturation → pillar CONTACT |
| `discontinuous` | `p_des` teleports, drone chases | large transient, possible flip/crash |
| `wall_crossing` | injected state passes through wall geometry without renderer crash | **CONTACT border + overlay must fire** — this validates the overlay machinery |
| `floor_dive` | drone rendered below z=0 plane, no crash | floor impact; proximity bar behaviour at d≈0 |
| `ceiling_climb` | steep climb, lag | thrust ceiling visible (climb stalls) |
| `gain_extreme` | normal path | kp×5: oscillation; kp×0.1: huge lag / drift |
| `degenerate_hover` | static frame(s), WARN not crash | stable hover, nothing happens |

## 7. Deliverables checklist (for the implementing agent)

- [ ] `generate_physics_gifs.py`: stress-case dispatch (§3) + gain-scale rebuild
- [ ] `generate_trajectory_gifs.py`: hardening (§4)
- [ ] Both sbatch scripts: optional `$6 = data_dir`
- [ ] Smoke: render `wall_crossing` + `degenerate_hover` first (the two most likely to
      break a renderer), then full `--per-homotopy 1` sweep
- [ ] `CHANGELOG.md` in this folder after implementation

## 8. Acceptance criteria

1. With `$6`/`--data-dir` omitted everywhere, production rendering is unchanged.
2. Full stress sweep (`--per-homotopy 1`, both scripts) completes with **0 crashes**;
   per-episode WARNs are acceptable and logged.
3. `wall_crossing` physics GIFs show CONTACT borders; `floor_dive` shows the floor
   impact; `tight_fillet` visually reproduces the Fix_1 failure mode.
4. No stress GIF is written into `logs/uav_expert_data/gifs*` (production tree clean).
