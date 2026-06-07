# Gen11 Epoch 4 — U2 Fix_1 Changelog

**Date**: 2026-06-07  
**Triggered by**: U2 re-collection debug run (temp/Gen11E4U2_debug)  
**Parent**: [`../CHANGELOG&USAGE.md`](../CHANGELOG&USAGE.md)

---

## What the debug run revealed

| Scene | Result | Issue |
|---|---|---|
| empty | ✅ 500/500, 0% rejected | Good |
| corridor | ⚠️ 307/500, 38.6% rejected | Threshold too tight — homotopy imbalance |
| s_curve | ❌ 6/500, 71.4% rejected → ABORT | Threshold way too tight |
| pillars | ✅ 477/500, 4.6% rejected | Good |

Plus one code bug in `stats_validator.py` unrelated to collection quality.

---

## Fix 1 — s_curve threshold reverted 0.04 → 0.08 (parameter tuning)

**File**: `uav_expert_data_collect/generator.py`

**Problem**: U2 halved s_curve threshold from 0.08 → 0.04 to reduce wall-sliding. In
practice, 71.4% of s_curve trials exceeded 0.04 → collection aborted after only 6
episodes. Fix_4's 0.08 threshold was set specifically because the s_curve scene has
narrow wall end-faces at x=±0.5 that cause brief grazes on otherwise valid diagonal
crossings — those grazes alone push contact_fraction above 0.04.

**Fix**: Revert `s_curve: 0.04 → 0.08`.

---

## Fix 2 — corridor threshold reverted 0.01 → 0.02 (parameter tuning)

**File**: `uav_expert_data_collect/generator.py`

**Problem**: U2 tightened corridor threshold from 0.02 → 0.01. The debug run revealed
that L and R homotopies **always** make brief wall contact (100% of saved L/R episodes
had contact_fraction > 0). This is physically unavoidable — L flies at y≈−0.22 and R
at y≈+0.22 with walls at y=±0.45, leaving only 0.23 m clearance. The 0.01 threshold
caused:

- 38.6% overall rejection (vs 12.8% at 0.02)
- Severe homotopy imbalance: C=167, L=70, R=70 (2.4:1:1 ratio)
- FM trained on this data would overfit to the centre channel

**Fix**: Revert `corridor: 0.01 → 0.02`. The brief wall contact in L/R training data
is not a mislabelled sample — it accurately reflects that those homotopies fly near the
wall. The visual FM must learn to fly near walls for L/R; that is their defining
characteristic.

---

## Fix 3 — stats_validator velocity column corrected (code bug)

**File**: `uav_expert_data_collect/stats_validator.py`, line 49

**Problem**: The validator computed speed from `obs[:, 3:6]`. In the pre-U2 6D obs
format `[p(3), v(3)]`, columns 3:6 were velocity — correct. After U2 widened obs to 9D
`[p_des(3), p(3), v(3)]`, columns 3:6 became position `p` — wrong. The validator was
reporting position vector norms (~1.5–2.5) as if they were speeds in m/s, triggering
a false `⚠️ CHECK` on every run.

**Before**:
```python
obs     = ep['obs']          # (T, 6)
v       = obs[:, 3:6]        # (T, 3) velocity
```

**After**:
```python
obs     = ep['obs']          # (T, 9)  U2: [p_des(3) | p(3) | v(3)]
v       = obs[:, 6:9]        # (T, 3) velocity — U2: shifted from [:, 3:6]
```

This fix does not affect collected data — it only affects the validation report.

---

## Files touched

| File | Change | Type |
|---|---|---|
| `uav_expert_data_collect/generator.py` | `corridor: 0.01→0.02`, `s_curve: 0.04→0.08` | Parameter tuning |
| `uav_expert_data_collect/stats_validator.py` | `obs[:, 3:6]→obs[:, 6:9]` for velocity | Code bug fix |

---

## Re-collection needed

Empty (500 eps) and pillars (477 eps) from the debug run are valid — keep them.  
Only s_curve needs re-collection. Corridor is borderline (307 eps usable but low for
L/R). Recommended re-collection scope:

```bash
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh s_curve  500
./Slurm_Codes/submit.sh Slurm_Codes/sbatch/uav_expert_data/collect.sh corridor 500
```
