# Gen11 Epoch 4 — U2 Upgrade Plan

**Date**: 2026-06-07  
**Status**: Planning — not yet executed  
**Predecessor**: Fix_5 final dataset (1769 episodes, 9D FM tensor `[Δp_des(3) ‖ p(3), v(3)]`)

---

## Driving principles

This upgrade is motivated by two post-Fix_5 analysis documents:

| Document | Finding | Change it drives |
|---|---|---|
| [`../DPCC_OBS_DEVIATION.md`](../DPCC_OBS_DEVIATION.md) | `p_des` is absent from obs → FM has no goal signal; FM tensor is 9D but controller needs velocity feed-forward that the 9D format cannot supply cleanly | **Change A** — add `p_des` to obs → 9D obs → 12D FM tensor |
| [`../../Epoch5_visual_and_validation/INVESTIGATION_wall_contact_gifs.md`](../../Epoch5_visual_and_validation/INVESTIGATION_wall_contact_gifs.md) | corridor 2% threshold allows up to 4 contact steps per episode; GIFs confirm real wall contact in training data; teaching the visual model that wall clips are acceptable may bias it toward walls | **Change B** — tighten per-scene contact thresholds |

**Combined strategy**: Change B requires a physics re-collection regardless.  Since we are
re-collecting, Change A is done as a code edit (not a post-process pickle hack) — giving
clean, unnoisy `p_des` in obs at zero extra cost.  Both changes ship in one SLURM run.

---

## Change A — Add `p_des` to observation

### What changes

**File**: `uav_expert_data_collect/dataset_writer.py`, line 52.

```python
# BEFORE (6D obs):
obs = np.array([np.concatenate([s['p'], s['v']]) for s in steps], dtype=np.float32)

# AFTER (9D obs):
obs = np.array([np.concatenate([s['p_des'], s['p'], s['v']]) for s in steps], dtype=np.float32)
```

`s['p_des']` is recorded at `generator.py:run_trial()` before the noise offset is applied
to `targets` (noise is applied to `targets` at lines 57–66 of `dataset_writer.py`, after
this line).  So `obs[:, :3]` will contain the **unnoisy** commanded position — the exact
value the PID was tracking.

### Why `p_des` must be in obs (OBS_DEVIATION §Deviation 2)

During training, corridor episodes start at the same entry point under three different
homotopy classes (L/C/R).  Without `p_des` in obs, the FM condition `[p(0), v(0)]` is
nearly identical across all homotopies at t=0 → FM samples a **mixture** of L, C, R at
inference.  Adding `p_des` (which encodes the intended channel) breaks the ambiguity and
mirrors what `des_xy` does in the D3IL avoiding FM.

### Schema impact

| Field | U1 (current) | U2 (new) |
|---|---|---|
| `obs` | `(T, 6)` — `[p(3), v(3)]` | `(T, 9)` — `[p_des(3), p(3), v(3)]` |
| `actions` | `(T-1, 3)` — `Δp_des` | `(T-1, 3)` — unchanged |
| FM tensor `D` | **9D** `[Δp_des(3) ‖ p(3), v(3)]` | **12D** `[Δp_des(3) ‖ p_des(3), p(3), v(3)]` |

`targets` field is unchanged (still noisy p_des, debug only).  `obstacles` and `metadata`
are unchanged.

### Why re-collect rather than post-process pickles

Post-processing would prepend `episode['targets']` (which is noisy — offset by `N(0, 0.02²)` per episode) to obs as a proxy for p_des.  The 2 cm constant offset is the same order of magnitude as the PID tracking error — small but a real impurity.  Since Change B forces re-collection anyway, re-collect with the code edit to get exact unnoisy p_des at zero extra cost.

---

## Change B — Tighten contact filter thresholds

### What changes

**File**: `uav_expert_data_collect/generator.py`, lines 85–89.

```python
# BEFORE:
SCENE_MAX_CONTACT_FRACTION = {
    'empty':    0.02,
    'corridor': 0.02,   # allows 4 contact steps in 200-step episode
    's_curve':  0.08,   # Fix_4: raised for narrow end-face grazes
    'pillars':  0.02,
}

# AFTER (proposed):
SCENE_MAX_CONTACT_FRACTION = {
    'empty':    0.02,   # no change — floor grazes only
    'corridor': 0.01,   # ← tightened: max 2 contact steps in 200-step episode
    's_curve':  0.04,   # ← halved: still allows end-face grazes; reduces wall-sliding
    'pillars':  0.02,   # no change — pillar grazes at 2% are acceptable
}
```

### Why (INVESTIGATION §Data quality question)

The investigation found that corridor GIFs show real wall contact that was accepted at 2%.
For a **visual** FM model, training on images where the drone is at the wall surface teaches
it that wall-clip observations are valid — biasing inference toward walls.

Tightening corridor to 1% reduces worst-case contact from 4 → 2 steps per episode.
Tightening s_curve to 4% halves the wall-sliding allowance while still permitting the
narrow end-face grazes that motivated Fix_4.

## Execution plan

### Step 1 — Code changes (2 files, ~5 lines total)

**`uav_expert_data_collect/dataset_writer.py` line 52** — add `p_des` to obs:
```python
obs = np.array([np.concatenate([s['p_des'], s['p'], s['v']]) for s in steps], dtype=np.float32)
```

Also update the schema comment at the top of the file (lines 10–11):
```python
# obs    : (T, 9)   float32  [p_des(3), p(3), v(3)]   ← was (T, 6)
```

**`uav_expert_data_collect/generator.py` lines 86–89** — update thresholds:
```python
SCENE_MAX_CONTACT_FRACTION = {
    'empty':    0.02,
    'corridor': 0.01,
    's_curve':  0.04,
    'pillars':  0.02,
}
```

---

### Step 2 — Re-collect (SLURM)

```bash
./Slurm_Codes/sbatch/uav_expert_data/collect_all.sh 500
```

Submits 4 independent SLURM jobs in parallel (one per scene).
Script: `Slurm_Codes/sbatch/uav_expert_data/collect_all.sh`.

All four scenes in parallel.  Expected wall time: similar to Fix_5 (~2–3 hours).  The new
episodes will overwrite the Fix_5 pickles in `logs/uav_expert_data/`.

**If WS-A images (camera PNGs) already exist from Epoch 5**: the new pickles have the same
episode_id naming convention → WS-A `--skip-existing` will skip them by folder name.
Re-run WS-A after collection to capture images for any newly generated episodes.

---

### Step 3 — Verify

After collection completes, verify:

```python
import pickle, glob, numpy as np

for scene in ['empty', 'corridor', 's_curve', 'pillars']:
    eps = [pickle.load(open(p,'rb')) for p in glob.glob(f'logs/uav_expert_data/{scene}/**/*.pkl', recursive=True)]
    print(f"\n{scene}: {len(eps)} episodes")
    obs0 = eps[0]['obs']
    print(f"  obs shape: {obs0.shape}   (expect (T, 9))")
    print(f"  obs[:3, :3] (p_des):  {obs0[:3, :3]}")
    print(f"  obs[:3, 3:6] (p):     {obs0[:3, 3:6]}")
    print(f"  max contact_fraction: {max(ep['metadata']['contact_fraction'] for ep in eps):.4f}")
```

**Acceptance criteria**:

| Check | Pass condition |
|---|---|
| `obs` shape | `(T, 9)` for all episodes |
| `obs[:, :3]` content | Matches `targets` (p_des) up to ~2 cm noise tolerance |
| `obs[:, :3]` != `obs[:, 3:6]` | p_des ≠ p (they're different quantities) |
| `corridor` max contact_fraction | ≤ 0.01 |
| `s_curve` max contact_fraction | ≤ 0.04 |
| Episode counts | ≥ 400 per scene (rejection rate within bounds) |
| FM tensor dim check | `actions.shape[-1] + obs.shape[-1] = 3 + 9 = 12` ✅ |

---

## Downstream impact after U2

| Component | Change needed? | What to do |
|---|---|---|
| **E4 pickles** | ✅ Re-collected | Done in Step 2 |
| **E5 WS-A images** | Partial — new episode IDs may appear | Re-run WS-A with `--skip-existing` |
| **E5 WS-B GIFs** | Partial — position data unchanged; GIF content identical | Re-run if episode list changes |
| **E5 WS-C mini-FM gate** | ✅ Yes — tensor dim changes 9D → 12D | Update WS-C config: `transition_dim=12` |
| **E6 FM training** | Config change only | `obs_dim=9`, `action_dim=3`, `transition_dim=12` |
| **E1, E2, E3** | ❌ No change | Physics-only epochs, unaffected |

---

## What U2 does NOT address

- ❌ `s_curve` GIF inspection (WS-B still pending from Fix_2 resubmit) — inspect after WS-B completes before committing to `0.04` threshold
- ❌ Velocity feed-forward gap (DPCC_OBS_DEVIATION §"9D incl a") — 12D with `Δv_des` in action is deferred to U3 or Epoch 6 evaluation; requires architecture change in addition to data change
- ❌ `pid_high_gain` / `pid_low_gain` variants — reserved for Epoch 6
- ❌ Domain randomisation (lighting, texture) — Epoch 6+

---

## Summary

| | Change A (obs) | Change B (contact filter) |
|---|---|---|
| **Code file** | `dataset_writer.py` line 52 | `generator.py` lines 86–89 |
| **Change** | obs 6D → 9D (add `p_des`) | corridor 0.02→0.01, s_curve 0.08→0.04 |
| **Re-collect needed?** | Yes (preferred; also forced by Change B) | Yes |
| **Downstream D** | FM tensor 9D→12D; E6 config change | Slightly fewer corridor episodes |
| **Driving doc** | `DPCC_OBS_DEVIATION.md` §Deviation 2 | `INVESTIGATION_wall_contact_gifs.md` §Data quality |

---

## Cross-references

| Document | Content |
|---|---|
| [`../DPCC_OBS_DEVIATION.md`](../DPCC_OBS_DEVIATION.md) | Full analysis of obs format deviations and fix options |
| [`../../Epoch5_visual_and_validation/INVESTIGATION_wall_contact_gifs.md`](../../Epoch5_visual_and_validation/INVESTIGATION_wall_contact_gifs.md) | Root cause of wall contact in corridor GIFs |
| [`../METHODOLOGY.md`](../METHODOLOGY.md) §4 | Contact filter design rationale (Fix_4 s_curve raise) |
| [`../METHODOLOGY.md`](../METHODOLOGY.md) §8 | Episode schema (obs shape will change from 6D to 9D) |
| [`../Fix_5/CHANGELOG.md`](../Fix_5/CHANGELOG.md) | Final state of data before U2 |
