# Gen11 E4 U10 — Stress-Test Episode Generation (Plan)

**Date:** 2026-06-12
**Status:** PLAN ONLY — implementation by a separate agent
**Pair unit:** [`../../Epoch5_visual_and_validation/U4_Stress_Test_Rendering/PLAN.md`](../../Epoch5_visual_and_validation/U4_Stress_Test_Rendering/PLAN.md)

---

## 1. Goal

Add a **CLI-gated, default-OFF** stress-test mode that generates deliberately *bad*
episodes into a **separate folder** (never mixed with training data), so the E5 GIF /
physics-GIF tools can render them for visual stress inspection of:

- the physics + PID stack under extreme demands,
- the rejection-gate machinery (do the gates actually fire?),
- the renderers themselves (do they survive pathological data — CONTACT overlays,
  proximity bars, teleporting states?).

Nothing in the production collection path changes behaviour unless explicitly commanded.

---

## 2. Key design decisions

### D1 — Standalone driver, not a flag on `collect.py` (recommended)

New script `uav_expert_data_collect/collect_stress.py`. Rationale: `collect.py` is the
validated production path (U9+Fix_1 just closed); a stress mode woven into it risks
regression and accidental invocation. A separate script makes "default not" automatic.
It reuses `generator.py` internals (`_make_pid`, `_is_obstacle_contact`, `SCENE_XMLS`,
`SCENE_OBSTACLES`) — the rollout loop is copied/refactored, **not** the gates.

### D2 — Separate output root

```
logs/uav_expert_data_stress/<scene>/<stress_case>/<episode_id>.pkl
```

- Created only when `collect_stress.py` runs; never by default.
- Layout mirrors the production `<scene>/<homotopy>/` convention with **stress_case in
  the homotopy slot** → all E5 discovery code (`discover_episodes`) and the
  `--per-homotopy` bucketing work unchanged (1 GIF per (scene, case) at N=1).

### D3 — No rejection; record verdicts instead

Stress episodes must NOT be rejected (most would be, by design). The rollout always
saves, and the gate verdicts become metadata:

```python
'stress': True,
'stress_case': '<case_name>',
'gate_verdicts': {'contact': bool, 'floor': bool},   # what production gates WOULD say
'contact_fraction': float, 'min_z': float, 'motor_clip_frac': float,
```

`episode_id` prefix `stress_` + keep the `_{seed:07d}` suffix (E5 physics replay
recovers the seed from the last token — this convention is load-bearing).

### D4 — Stress trajectories live in a new module

`uav_expert_data_collect/stress_trajectories.py` exposes one dispatcher:

```python
build_stress_traj(case, scene, rng) -> (traj_fn, init_pos, duration)
```

Same signature shape as `generator._build_traj_and_init`, so the E5 physics replay can
dispatch on `stress_case` (stored in the pickle) instead of homotopy. **Do not touch
`trajectories.py`** — production geometry stays frozen.

---

## 3. Stress case catalogue (v1)

| Case | What it does | What it stresses |
|------|-------------|------------------|
| `extreme_speed` | pillars/s_curve path at T = 3 s (vs validated 10–22 s) → peak speed ~4–9 m/s | PID saturation, motor clip telemetry, floor gate |
| `tight_fillet` | LRL pillars at blend radius r = 0.05 m → centripetal ≈ 35 m/s² | deliberate re-creation of the Fix_1 failure; contact gate |
| `discontinuous` | p_des teleports mid-episode (`step_to`-style jump of 2–3 m) | PID response to step input; renderer on teleporting `p_des` vs lagging `p` |
| `wall_crossing` | corridor: straight line through a wall (y: −0.3 → +0.3 crossing the wall end); pillars: straight through a pillar axis (y = ±0.6) | contact detection, CONTACT overlay in physics GIF, contact_frac telemetry |
| `floor_dive` | p_des descends to z = −0.3 (below floor) | floor gate (min_z), proximity/crash rendering |
| `ceiling_climb` | z: 1.0 → 4.0 in 2 s | thrust ceiling `u_max = 2·u_hover`, vertical saturation |
| `gain_extreme` | normal trajectory, PID gains kp×5.0 and kp×0.1 variants | oscillation / sluggish tracking; uses `GAIN_VARIANTS`-style scaling outside the production dict |
| `degenerate_hover` | duplicate waypoints / zero-length path | `blended_path` dedup edge case; renderer on a static episode |

Each case: default **N = 3 episodes per applicable scene**, fixed seed base (reproducible),
total ≈ 50–60 episodes, < 2 min collection. Cases declare which scenes they apply to.

---

## 4. CLI

```bash
# default: nothing — script requires explicit case selection
python uav_expert_data_collect/collect_stress.py --cases all
python uav_expert_data_collect/collect_stress.py --cases wall_crossing,floor_dive --scene pillars
python uav_expert_data_collect/collect_stress.py --cases extreme_speed --n-per-case 5 --seed 0
```

- `--cases` required (no default) — this is the "only when commanded" gate.
- `--out-dir` defaults to `logs/uav_expert_data_stress`.
- Writes `stress_summary.json` at the root: per-case episode counts, gate-verdict
  histogram, clip stats.

New sbatch: `Slurm_Codes/sbatch/uav_expert_data/collect_stress.sh`
(`$1 = cases`, `$2 = scene ("" = all)`, `$3 = n_per_case`). CPU-only, `MUJOCO_GL=disabled`,
same repo-root resolution pattern as `collect.sh`.

---

## 5. What must NOT happen

- `collect.py`, `generator.py` gates, `trajectories.py`, `verify_blends.py` behaviour
  unchanged. (Small refactor of `generator.py`'s rollout loop into a reusable helper is
  acceptable if collect.py output is bit-identical.)
- `stats_validator.py` must never count stress episodes → it only ever points at
  `logs/uav_expert_data/`; additionally it should **skip** any pickle with
  `'stress': True` as a belt-and-braces guard.
- Training loaders: out of scope, but the `'stress': True` flag is the contract any
  future loader can filter on.

---

## 6. Deliverables checklist (for the implementing agent)

- [ ] `uav_expert_data_collect/stress_trajectories.py` — 8 case builders + dispatcher
- [ ] `uav_expert_data_collect/collect_stress.py` — driver, no gates, verdict metadata, summary JSON
- [ ] `Slurm_Codes/sbatch/uav_expert_data/collect_stress.sh`
- [ ] Guard in `stats_validator.py` (skip `'stress': True`)
- [ ] E5 U4 rendering support (see pair plan)
- [ ] `CHANGELOG.md` in this folder after implementation

## 7. Acceptance criteria

1. Running `collect.py` (no args changed) produces zero stress artefacts — folder absent.
2. `collect_stress.py --cases all` writes ~50–60 pickles under
   `logs/uav_expert_data_stress/`, none rejected, all carrying `stress: True` +
   `gate_verdicts`.
3. `wall_crossing` episodes show `gate_verdicts.contact = True`; `floor_dive` shows
   `gate_verdicts.floor = True` — proves the gates would have fired.
4. E5 U4 renders both GIF types from the stress root without code crashes (renderer
   robustness is itself a test subject — see pair plan).
