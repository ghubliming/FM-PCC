# Gen11 Epoch 4 — U2 Closure

**Date**: 2026-06-07  
**Status**: ✅ Closed  
**Final dataset**: 1769 episodes, obs=(T, 9), FM tensor 12D

---

## Why U2 was opened

Two concerns emerged after the Fix_5 dataset was collected:

1. **Goal-conditioning gap** (`DPCC_OBS_DEVIATION.md` §Deviation 2) — obs was 6D `[p(3), v(3)]` with no `p_des`. At inference the FM condition is nearly identical across all homotopy classes at t=0, causing it to sample a mixture of L/C/R rather than committing to one. Mirrors the `des_xy` goal signal in D3IL avoiding.

2. **Wall contact in corridor GIFs** (`INVESTIGATION_wall_contact_gifs.md`) — the 2% contact threshold allowed up to 4 contact steps per episode. E5 GIFs confirmed real wall contact in training data, which risks teaching the visual FM that wall clips are acceptable.

---

## What U2 changed (Change A — fully successful)

**`dataset_writer.py` line 52**: obs widened from 6D → 9D by prepending `s['p_des']`:

```
obs: (T, 6)  [p(3), v(3)]          ← Fix_5
obs: (T, 9)  [p_des(3), p(3), v(3)] ← U2
```

`s['p_des']` is read before noise is applied to `targets`, so obs carries the **unnoisy** commanded position — the exact setpoint the PID tracked. FM tensor widens from 9D → 12D.

This change worked correctly from the first run and required no further adjustment.

---

## What U2 changed (Change B — revised by Fix_1)

Original intent: tighten contact thresholds to reduce wall-clip training data.

| Scene | U2 proposed | Fix_1 reverted to | Reason for revert |
|---|---|---|---|
| corridor | 0.02 → 0.01 | 0.02 | L/R homotopies physically always touch walls; 0.01 caused 38.6% rejection and 2.4:1:1 C:L:R imbalance |
| s_curve | 0.08 → 0.04 | 0.08 | 71.4% rejection → ABORT (6 episodes); end-face grazes require 0.08 headroom |
| empty | 0.02 | 0.02 | unchanged |
| pillars | 0.02 | 0.02 | unchanged |

**Key finding**: the corridor contact concern was real but the wrong fix. L and R homotopies fly at y≈±0.22 with walls at y=±0.45 (0.23 m clearance) — brief wall contact is inherent to those trajectories, not a quality defect. Tightening the threshold does not remove the problem; it removes the homotopy diversity. The 0.02 threshold (Fix_5 level) remains the correct balance.

---

## What Fix_1 also fixed (stats_validator code bug)

`stats_validator.py` was reading `obs[:, 3:6]` for velocity. Before U2 (6D obs) that was correct. After U2 (9D obs) those columns became position `p`, causing the validator to report position vector norms (~1.5–2.5) as speed in m/s — a false warning on every run. Fixed to `obs[:, 6:9]`.

---

## Final verified state

Jobs 21324–21327, 2026-06-07, all scenes re-collected under Fix_1 thresholds.

| Scene | Episodes | Rejection | Homotopy balance | Speed | obs |
|---|---|---|---|---|---|
| empty | 500 | 0% | N/A: 500 | 0.387 m/s ✅ | (T, 9) ✅ |
| corridor | 436 | 12.8% | C:167 L:139 R:130 ✅ | 0.716 m/s ✅ | (T, 9) ✅ |
| s_curve | 356 | 28.8% | default: 356 | 0.560 m/s ✅ | (T, 9) ✅ |
| pillars | 477 | 4.6% | balanced across 4 classes ✅ | 0.417 m/s ✅ | (T, 9) ✅ |
| **Total** | **1769** | — | — | — | ✅ |

Rejection rates and total episode count are identical to Fix_5. Only obs format changed.

---

## Why U2 is closed

- Change A (obs 6D→9D) delivered and verified — `p_des` is in obs, unnoisy, correct columns.
- Change B (contact thresholds) attempted, found to be mis-parameterised, correctly reverted to Fix_5 levels. The corridor wall-contact concern is documented but accepted: it is a physical property of L/R homotopies, not a collection defect.
- Stats validator bug fixed — clean output on all future runs.
- 1769 episodes produced, all passing obs shape check `(T, 9)`, all validator checks ✅.
- FM tensor is now 12D `[Δp_des(3) ‖ p_des(3), p(3), v(3)]` — downstream Epoch 6 config needs `obs_dim=9, action_dim=3, transition_dim=12`.

---

## Cross-references

| Document | Content |
|---|---|
| [`PLAN.md`](PLAN.md) | Original U2 design — two changes, rationale |
| [`CHANGELOG&USAGE.md`](CHANGELOG&USAGE.md) | Code changes made, re-collection commands |
| [`Fix_1/CHANGELOG.md`](Fix_1/CHANGELOG.md) | Debug run findings, threshold reverts, validator fix, verified results |
| [`../DPCC_OBS_DEVIATION.md`](../DPCC_OBS_DEVIATION.md) | Goal-conditioning gap that motivated Change A |
| [`../../Epoch5_visual_and_validation/INVESTIGATION_wall_contact_gifs.md`](../../Epoch5_visual_and_validation/INVESTIGATION_wall_contact_gifs.md) | Wall contact concern that motivated Change B |
| [`../CLOSURE.md`](../CLOSURE.md) | Fix_5 final dataset (predecessor state) |
