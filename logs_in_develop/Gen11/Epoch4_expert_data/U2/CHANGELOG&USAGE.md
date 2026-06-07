# Gen11 Epoch 4 — U2 Changelog & Usage

**Date**: 2026-06-07  
**Status**: Code done — re-collection pending (SLURM)  
**Parent plan**: [`PLAN.md`](PLAN.md)  
**Predecessor**: Fix_5 final dataset (1769 episodes, 9D FM tensor)

---

## Summary of changes

Two code changes, two files, zero physics re-implementation required.  A full re-collection
SLURM run is needed to produce the new pickles.

---

## Change A — obs 6D → 9D: add `p_des` to observation

**File**: `uav_expert_data_collect/dataset_writer.py`

**Lines changed**: 10 (schema docstring), 52–53 (obs construction)

### Before
```python
obs         : (T, 6)   float32  [p(3), v(3)]
...
obs     = np.array([np.concatenate([s['p'], s['v']]) for s in steps],
                   dtype=np.float32)        # (T, 6)
```

### After
```python
obs         : (T, 9)   float32  [p_des(3), p(3), v(3)]   — U2: p_des prepended
...
obs     = np.array([np.concatenate([s['p_des'], s['p'], s['v']]) for s in steps],
                   dtype=np.float32)        # (T, 9)  [p_des(3) | p(3) | v(3)]
```

### Key implementation detail

`s['p_des']` is read at line 52 **before** the noise offset is applied to `targets`
(lines 65–66).  This means `obs[:, :3]` contains the **unnoisy** commanded position —
the exact setpoint the PID was tracking.  `targets` (debug field) still carries the
noisy version.  No post-processing impurity.

### Schema impact

| Field | Fix_5 (old) | U2 (new) |
|---|---|---|
| `obs` shape | `(T, 6)` | `(T, 9)` |
| `obs` content | `[p(3), v(3)]` | `[p_des(3), p(3), v(3)]` |
| `actions` | `(T-1, 3)` Δp_des | unchanged |
| `targets` | `(T, 3)` noisy p_des | unchanged |
| FM tensor `D` | **9** = act(3)+obs(6) | **12** = act(3)+obs(9) |

**Why**: FM conditioned only on `[p, v]` cannot distinguish homotopy classes at t=0
(all homotopies share the same corridor entry state).  Adding `p_des` mirrors `des_xy`
in D3IL avoiding — giving the FM a goal signal to disambiguate L / C / R at inference.
Full analysis: `DPCC_OBS_DEVIATION.md` §Deviation 2.

---

## Change B — contact thresholds tightened for corridor and s_curve

**File**: `uav_expert_data_collect/generator.py`

**Lines changed**: 85–93 (`SCENE_MAX_CONTACT_FRACTION` dict + comments)

### Before
```python
SCENE_MAX_CONTACT_FRACTION = {
    'empty':    0.02,
    'corridor': 0.02,   # max 4 contact steps in ~200-step episode
    's_curve':  0.08,   # Fix_4: raised for narrow end-face grazes
    'pillars':  0.02,
}
```

### After
```python
SCENE_MAX_CONTACT_FRACTION = {
    'empty':    0.02,   # unchanged
    'corridor': 0.01,   # max 2 contact steps — tightened from 0.02
    's_curve':  0.04,   # halved — still covers end-face grazes
    'pillars':  0.02,   # unchanged
}
```

### Rationale

E5 WS-B GIFs showed that corridor episodes accepted at 2% genuinely include frames where
the drone COM is at the wall surface.  Training the visual FM on these images teaches it
that wall-clip observations are acceptable — biasing inference toward walls.  Tightening
to 1% caps corridor contact at 2 steps per episode.

s_curve 0.08 allowed up to 16 contact steps per episode (wall-sliding during the diagonal
gap crossing).  Halving to 0.04 reduces the worst-case while preserving the Fix_4
rationale (end-face grazes at x=±0.5 on otherwise valid trajectories).

`empty` and `pillars` are unchanged — no visual evidence of problematic contact in those
scenes.

Full analysis: `../Epoch5_visual_and_validation/INVESTIGATION_wall_contact_gifs.md`.

---

## Re-collection command

**One-liner** (submits all 4 scenes as parallel SLURM jobs):
```bash
./Slurm_Codes/sbatch/uav_expert_data/collect_all.sh 500
```

**Per-scene** (submit individually):
```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh empty    500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh corridor 500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh s_curve  500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh pillars  500
```

After collection:

```bash
# Verify obs shape and contact caps
python - <<'EOF'
import pickle, glob, numpy as np
for scene in ['empty', 'corridor', 's_curve', 'pillars']:
    eps = [pickle.load(open(p,'rb'))
           for p in glob.glob(f'logs/uav_expert_data/{scene}/**/*.pkl', recursive=True)]
    if not eps: print(f"{scene}: no pickles"); continue
    obs0 = eps[0]['obs']
    cf   = [ep['metadata']['contact_fraction'] for ep in eps]
    print(f"{scene}: n={len(eps)}  obs={obs0.shape}  max_cf={max(cf):.4f}")
EOF
```

Expected: `obs=(T, 9)` for all scenes; corridor `max_cf ≤ 0.01`; s_curve `max_cf ≤ 0.04`.

---

## Downstream steps after collection

| Component | Action |
|---|---|
| E5 WS-A (camera images) | Re-run with `--skip-existing` — new episode IDs may appear |
| E5 WS-B (GIFs) | Re-run if episode list changes |
| E5 WS-C (mini-FM gate) | Update config: `transition_dim = 12` |
| E6 FM training | Config: `obs_dim = 9`, `action_dim = 3`, `transition_dim = 12` |

---

## Files touched

| File | Change |
|---|---|
| `uav_expert_data_collect/dataset_writer.py` | obs 6D→9D: prepend `s['p_des']`; update schema docstring |
| `uav_expert_data_collect/generator.py` | `SCENE_MAX_CONTACT_FRACTION`: corridor 0.02→0.01, s_curve 0.08→0.04 |
