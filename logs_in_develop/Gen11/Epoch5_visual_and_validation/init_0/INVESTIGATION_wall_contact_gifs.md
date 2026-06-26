# Gen11 E5 Investigation — UAV Hitting Wall in Expert Data GIFs

**Date**: 2026-06-06  
**Status**: Under investigation  
**Run context**: Fix_2 WS-B resubmit (SLURM job — corridor 216 GIFs done; empty/pillars/s_curve pending)  
**Related fix**: [`Fix_2/CHANGELOG.md`](Fix_2/CHANGELOG.md)

---

## Observation

Corridor GIFs (e.g. `logs/uav_expert_data/gifs/corridor/R/corridor_R_pid_default_0000002.gif`)
show the UAV visually clipping or hitting the wall.  These episodes were saved as **expert**
data by the contact-filtered pipeline in `generator.py`, yet visually appear non-expert.

---

## Root cause analysis

### Hypothesis 1 — Contact filter threshold allows brief wall contact (most likely)

`generator.py` defines:

```python
MAX_CONTACT_FRACTION = 0.02                       # global default
SCENE_MAX_CONTACT_FRACTION = {
    'empty':    0.02,
    'corridor': 0.02,
    's_curve':  0.08,   # raised for narrow wall end-faces
    'pillars':  0.02,
}
```

Episodes are **kept** when `contact_fraction ≤ threshold`.  At 2%, a 200-step episode may
have up to **4 contact steps** and still pass.  Those contact steps represent actual MuJoCo
physics where drone geoms touch wall geoms — the positions recorded in `obs[:, :3]` are
at or inside the wall surface.

`generate_trajectory_gifs.py` replays those positions verbatim:

```python
data.qpos[:3] = obs[t, :3]
```

So the GIF faithfully shows the drone at a position that is geometrically inside or touching
the wall during those contact steps.

**Conclusion**: The "wall hit" GIFs are not a GIF artifact — they represent episodes where
brief wall contact genuinely occurred and was accepted by the 2% threshold.

---

### Hypothesis 2 — Identity quaternion makes contact appear worse (contributing factor)

The GIF replay forces a level attitude:

```python
data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]   # identity quaternion — always upright
data.qvel[3:6] = 0.0                      # zero angular velocity
```

In real flight the UAV pitches/rolls during acceleration.  When the drone was flying fast
toward a wall, its actual attitude was tilted — the body may have been further from the wall
than the COM position suggests.  By forcing identity attitude in the GIF, a corner or rotor
of the bounding box may appear to penetrate the wall even when only the COM was near it.

**Conclusion**: Identity quaternion exaggerates the visual severity of near-wall positions.
The contact fraction metric is computed on actual physics (with real attitude), so the 2%
filter is accurate — but the GIF makes it look worse than real flight.

---

### Hypothesis 3 — MuJoCo numerical contact artifacts (minor)

`_is_obstacle_contact` counts **any** non-floor contact:

```python
def _is_obstacle_contact(model, contact):
    n1 = model.geom(contact.geom1).name
    n2 = model.geom(contact.geom2).name
    return n1 != 'floor' and n2 != 'floor'
```

MuJoCo sometimes generates brief penetration contacts at geometry boundaries even when the
body is not physically touching — these transient contacts may inflate `contact_fraction`
slightly or produce saved positions that are technically outside the wall surface by a small
numerical margin.

**Conclusion**: Minor contributor; unlikely to fully explain visible wall clips.

---

## Corridor geometry reference

| Element | Value |
|---------|-------|
| `wall_y_neg` center | y = −0.5 |
| `wall_y_pos` center | y = +0.5 |
| Wall half-extent | 0.05 m |
| Wall faces (inner) | y = ±0.45 |
| Usable corridor width | ~0.90 m |

Homotopy `L`/`C`/`R` = fly left / center / right of corridor.  `R` episodes target y ≈ +0.22
— relatively close to `wall_y_pos` face at y = +0.45.  Brief overshoots at high speed can
clip the wall within the 2% budget.

---

## Data quality question

Is ≤ 2% contact an acceptable quality bar for training the visual FM model?

| Threshold | Max contact steps (200-step ep) | Risk |
|-----------|--------------------------------|------|
| 0.02 (current) | 4 steps | Visible wall clips in GIFs; brief but real contact in training data |
| 0.01 | 2 steps | Reduces contact steps; may reject borderline-valid episodes |
| 0.00 | 0 steps | Zero-tolerance; rejects any numerical MuJoCo contact artifact |

For visual avoidance training, including episodes with brief wall contact teaches the model
that wall-clip observations are acceptable.  This may bias the policy to fly closer to walls
than intended.

**Recommendation**: Consider tightening `corridor` threshold to `0.01` or adding a
post-filter that rejects episodes whose GIF contains frames where `obs[:, 1]` is within
0.05 m of y = ±0.45 (i.e. `|y| > 0.40`).

---

## s_curve note

`s_curve` uses `0.08` threshold (4× the default).  This was intentional (Fix_4 comment in
`generator.py`): narrow wall end-faces at x=±0.5 cause brief grazing on otherwise valid
trajectories.  However, 8% allows up to 16 contact steps in a 200-step episode — visually
this may appear as the drone sliding along the wall during the tight turn.  Worth inspecting
s_curve GIFs when they become available (WS-B resubmit still pending).

---

## Next steps

1. **Inspect contact_fraction distribution** in saved corridor pickles:
   ```python
   import pickle, glob, numpy as np
   fracs = []
   for p in glob.glob('logs/uav_expert_data/corridor/**/*.pkl', recursive=True):
       ep = pickle.load(open(p,'rb'))
       fracs.append(ep['metadata']['contact_fraction'])
   print(np.percentile(fracs, [50, 90, 95, 99, 100]))
   ```
2. **Count episodes with any contact** (`contact_fraction > 0`) vs zero-contact.
3. **Verify homotopy R > L/C** in contact fraction distribution (R flies closer to wall).
4. **Decision**: Tighten corridor threshold to 0.01 and regenerate, or accept current data
   and rely on the visual model learning to avoid walls from context.
5. **s_curve inspection**: Once WS-B GIFs complete, spot-check `s_curve` GIFs for severity
   of wall-sliding under the 0.08 threshold.
