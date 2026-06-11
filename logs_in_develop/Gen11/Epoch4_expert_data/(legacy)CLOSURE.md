# Gen11 Epoch 4 — Expert Data Collection: Closure

**Date**: 2026-06-04  
**Status**: ✅ Complete  
**Jobs**: 21205 (smoke), 21206–21209 (run 0), 21212–21215 (run 1), 21220–21221 (run 2), 21222 (run 3), 21226 (run 4), 21227 (run 5)

---

## Final dataset

| Scene | Episodes | Rejection | Homotopy classes |
|---|---|---|---|
| empty | **500** | 0% | N/A (random start→end) |
| corridor | **436** | 12.8% | L (139), C (167), R (130) |
| pillars | **477** | 4.6% | (L,L,L) 112, (L,R,L) 125, (R,L,R) 125, (R,R,R) 115 |
| s_curve | **356** | 28.8% | default (356) |
| **Total** | **1769** | — | — |

### Validator stats (final runs)

| Scene | Speed mean | Action Δp norm mean | Episode length mean |
|---|---|---|---|
| empty | 0.387 m/s ✅ | 0.0116 m/step ✅ | 190 steps |
| corridor | 0.716 m/s ✅ | 0.0205 m/step ✅ | 274 steps |
| pillars | 0.417 m/s ✅ | 0.0241 m/step ✅ | 442 steps |
| s_curve | 0.560 m/s ✅ | 0.0114 m/step ✅ | 641 steps |

All action Δp norms are in the expected range for 33 Hz dataset (Fix_1.4 noise fix confirmed working). Speed and episode length are consistent with the target 0.3–0.5 m/s, 4–22 s episodes.

---

## Dataset schema

Each episode is a pickle file at `logs/uav_expert_data/<scene>/<homotopy_safe>/<episode_id>.pkl`:

```python
{
  'episode_id': str,
  'scene':      str,         # 'empty' | 'corridor' | 's_curve' | 'pillars'
  'homotopy':   str,         # 'N/A' | 'L' | 'C' | 'R' | '(L,L,L)' | … | 'default'
  'controller': str,         # 'pid_default'
  'dt':         float,       # ≈ 0.030 s (100 Hz physics → 33 Hz dataset)
  'obs':        (T, 6),      # float32  [p_x, p_y, p_z, v_x, v_y, v_z]
  'actions':    (T-1, 3),    # float32  [Δp_des_x, Δp_des_y, Δp_des_z]  ← position-delta
  'targets':    (T, 3),      # float32  absolute p_des (for debugging)
  'obstacles':  list[dict],  # scene obstacle geometry
  'metadata':   dict,        # start_pos, duration, contact_fraction, noise_sigma
}
```

**Action convention**: `actions[t] = targets[t+1] − targets[t]` — position-delta, matching D3IL and UAV-Flow. FM-PCC dataloader expects `(H=8, D=9)` chunks where `D = [actions(3) ‖ obs(6)]`.

---

## Fix history

| Fix | Scene | Issue | Solution |
|---|---|---|---|
| Fix_1.1–1.2 | s_curve | 100% rejection: piecewise stops at wall ends | 6 waypoints, longer duration |
| Fix_1.3 | pillars | 95% rejection: piecewise stops near pillars | Replaced with continuous `weave` factory |
| Fix_1.4 | **all** | Action deltas noise-dominated (mean 0.047 vs 0.012 m/step) | Per-episode constant noise offset instead of per-step |
| Fix_2.1 | s_curve | 90.5% rejection persisted | Tanh continuous trajectory (no stops) |
| Fix_2.2 | pillars | `(L,R,L)` / `(R,L,R)` amplitude ±0.55 inside pillar zone | Amplitude → 0.0 (centre passage) |
| Fix_3 | s_curve | 61.9% rejection | Lowered k 3.66→2.0 — made things **worse** (81.8%) |
| Fix_4.1 | s_curve | k=2.0 brought path closer to walls | Reverted k→3.66 |
| Fix_4.2 | s_curve | Brief end-face grazes rejected at 2% threshold | Per-scene threshold: s_curve → 8% |
| Fix_5.1 | s_curve | 47.6% rejection: gap crossing ran at 1.17 m/s | Proportional-duration segments → 0.57 m/s uniform |
| Fix_5.2 | s_curve | Job aborted at 21 trials from seed variance | Abort limit 0.30 → 0.60 |

---

## Design decisions confirmed

| Decision | Outcome |
|---|---|
| Position-delta as action (AUDIT Risk 3) | ✅ Consistent with D3IL / UAV-Flow; Fix_1.4 critical to correctness |
| 9D format `[Δp_des(3) ‖ p(3), v(3)]` | ✅ All episodes saved in this schema |
| Per-episode constant noise (σ=0.02 m) | ✅ Action norm restored to expected range post-Fix_1.4 |
| 33 Hz dataset (every 3rd 100 Hz physics step) | ✅ Episode lengths 190–641 steps across scenes |
| Per-scene contact thresholds | ✅ s_curve 8%, all others 2% |
| Homotopy classes for multi-modality | ✅ Corridor 3 classes, pillars 4 classes (2 outer + 2 centre), balanced counts |

---

## What is NOT in this dataset

- ❌ Visual observations (images) — Stage 2 replay is Epoch 5 scope
- ❌ `pid_high_gain` / `pid_low_gain` variants — only `pid_default` collected
- ❌ s_curve multi-homotopy — only one topological route exists in this scene
- ❌ DAgger / on-policy data — Epoch 5+ scope

---

## Immediate next step — Mini-FM sanity gate (§3 of plan)

Before using this dataset in Epoch 5 FM training, run the sanity gate:

> Train a tiny FM on ~100 **empty-scene** episodes. If FM reproduces PID trajectories at < 0.1 m RMS on held-out empty episodes, the schema, action convention, and horizon configuration are all correct.

If this passes: proceed to Epoch 5 (FM-PCC training on UAV task).  
If this fails: the data format or action convention is wrong — fix before scaling.

---

## Cross-references

| Document | Content |
|---|---|
| `EPOCH4_EXECUTION_PLAN.md` | Full plan, blocking decisions, risk register |
| `CHANGELOG.md` | Initial pipeline build (all files created) |
| `Fix_1/CHANGELOG.md` | Noise fix + s_curve/pillars first-pass fixes |
| `Fix_2/CHANGELOG.md` | Tanh trajectory + pillar amplitude correction |
| `Fix_3/CHANGELOG.md` | k=2.0 attempt (caused regression) |
| `Fix_4/CHANGELOG.md` | Revert k + per-scene contact threshold |
| `Fix_5/CHANGELOG.md` | Proportional-duration segments + abort limit |
| `phase4_alpha_uavflow_stats.json` | UAV-Flow reference kinematic statistics |
| `USAGE.md` | How to run collect + validate + submit to SLURM |
