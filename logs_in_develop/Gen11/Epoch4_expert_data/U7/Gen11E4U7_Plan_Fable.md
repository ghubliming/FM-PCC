# Gen11E4U7 — Plan

**Date:** 2026-06-10 · **Scope:** s_curve — the last failing scene (100% rejection across F3/F4/F5/F6)
**Inputs:** `U6/{CHANGELOG.md, F6_RESULTS.md}`, F6 reject telemetry (job 21418), wall geometry from `generator.py SCENE_OBSTACLES`, pillars-vs-s_curve physics comparison
**Honest accounting:** F6_RESULTS' recommendation (Seg B 4× weight + low-gain PID) is **withdrawn** — it attacks a non-cause. The numbers below prove the diagonal is geometrically infeasible: no speed or gain change can fix it.

---

## Why "slow it down" is the wrong fix (evidence)

**Speed exoneration — pillars comparison.** Pillars inter-channel diagonals run far harder than Seg B and pass at 90.4%:

| Trajectory | v_peak | a_peak | Result |
|---|---|---|---|
| pillars diagonal (2.43 m / ~3.4 s) | 1.12 m/s | 1.04 m/s² | ✅ 90.4% success |
| s_curve Seg B (1.89 m / ~6.6 s, 2× weight) | 0.45 m/s | 0.21 m/s² | ❌ 0% success |

The same controller flies a 5×-more-aggressive diagonal successfully. Trajectory aggressiveness cannot be the s_curve killer; further slowing (4×) or gain reduction would not have fixed it.

## Actual root cause: the gap diagonal passes INSIDE the rotor-contact zone of both wall corners

Gap-side wall corners (from `SCENE_OBSTACLES['s_curve']`): corner A = (−0.5, −0.25) on `seg1_wall_pos`, corner B = (+0.5, +0.25) on `seg2_wall_neg`.

The nominal Seg B straight line (−0.5,−0.8) → (+0.5,+0.8) approaches:

```
corner A: min clearance 0.291 m at f=0.247 (point −0.253, −0.404)
corner B: min clearance 0.291 m at f=0.753 (symmetric)
rotor reach: 0.31 m   →   0.019 m penetration on the NOMINAL path, zero tracking error needed
```

The U4-era design was infeasible from day one. This explains the entire history:

- **F3 (no hover): 90.5%** — y_jitter (±0.04 m) occasionally rescued the ~2 cm penetration; 2 lucky episodes.
- **F4 timing**: divergence "~2 s into Seg B" = f≈0.25 of t_b — exactly the corner-A passage. The contact impulse caused the attitude divergence U4's P&S attributed to saturation; causality was reversed.
- **F6 smoking gun**: contact rejects #3 (min_z=0.741) and #7 (min_z=0.959) — wall-dragging at healthy altitude, no fall involved. Floor rejects are the corner-kick → tumble → crash variant of the same event.
- Saturation (clip 17–60%) is downstream: this controller saturates at ~1.5–2° attitude error by design (Kp_att=70 vs ~2.5 N motor headroom); it does so in pillars too and recovers. It's the corner impulse it can't recover from.

---

## Changes

### C1 — Replace the Seg B diagonal with a 3-leg Z-route through the gap centerline (`trajectories.py`)

Reuse the `pillar_path` design pattern (piecewise `traverse_line` with v=0 waypoint stops — empirically proven at 90.4% under harsher dynamics):

```
Seg A:  (−3.2, y1, z) → (−0.5, y1, z)    2.7 m  pure-x   (unchanged)
Hov 1:  hover at (−0.5, y1, z)            1.0 s           (unchanged)
Leg B1: (−0.5, y1, z) → (0, y1, z)        0.5 m  pure-x   exit corridor 1
Leg B2: (0, y1, z) → (0, y2, z)           1.6 m  pure-y   cross gap on centerline x=0
Leg B3: (0, y2, z) → (+0.5, y2, z)        0.5 m  pure-x   enter corridor 2
Hov 2:  hover at (+0.5, y2, z)            1.0 s           (unchanged)
Seg C:  (+0.5, y2, z) → (+3.2, y2, z)    2.7 m  pure-x   (unchanged)
```

Verified clearances: leg B2 runs 0.50 m from both corners; legs B1/B3 ≥ 0.55 m; in-corridor margin stays 0.14 m (identical to the corridor scene, which rejects 0%).

**Lag-robust by construction:** at every pinch point the path runs parallel to the nearby wall — tracking lag is along-path and cannot reduce wall clearance. (The old diagonal converted its documented 0.185 m lag directly into corner penetration.)

**Time allocation: plain proportional-to-distance over the 5 traverse legs** (`t_i = T_move · d_i / Σd`), exactly like `pillar_path`. **Remove the 2× Seg-B weight** — speed is exonerated above, and leg B2 lands at v_peak ≈ 0.6–0.8 m/s / a_peak ≈ 0.3–0.6 m/s², well inside the pillars-proven envelope. Update the docstring with the corner-clearance numbers so the constraint is recorded where the next editor will look.

Keep both hovers, `T_HOVER=1.0`, duration range [18,24] s, y_jitter handling, and the U6 controller/telemetry — all unchanged.

### C2 — Log clip stats for ACCEPTED episodes too (`collect.py`, ~5 lines)

F6 could not compare reject-episode clip% against healthy-episode clip% (accepted episodes don't report it). Accumulate `rollout['motor_clip_frac']` for saved episodes and add to `run_summary.json`:

```python
"accepted_clip_mean": ..., "accepted_clip_max": ...
```

Predicted outcome: accepted s_curve episodes show clip% comparable to pillars/corridor baseline — confirming saturation is chronic-but-benign and the corner was the killer. If instead accepted episodes are clean and only s_curve shows high clip, that's a signal worth a follow-up look.

### C3 — Run strategy

1. s_curve 20-trial smoke → gate < 30% (expect near-0: the failure was deterministic geometry, now removed).
2. s_curve full 500.
3. **Optional pillars top-up** for homotopy balance (F6: L_R_L=99, R_L_R=103 vs 125/125): two dedicated runs with `--homotopy "(L,R,L)"` / `"(R,L,R)"`, ~30 trials each, fresh seed offset. Only if downstream training wants balanced classes — 9.6% rejection already passes the gate.
4. empty / corridor / pillars code paths: untouched.

---

## Validation gates

| Run | Expectation | Gate |
|-----|-------------|------|
| s_curve 20-trial smoke | corner contact gone; floor & contact both ≈ 0 | < 30% rejection |
| s_curve full 500 | — | < 30%, then E5 gate |
| sanity on histogram | any remaining rejects: read `reason`+`min_z`+`clip%` before touching anything | — |

If the smoke still rejects > 30%: the histogram decides. `contact` with healthy min_z → re-derive geometry (don't guess); `floor` with clip > 50% → only then revisit gains/u_max (F6's Options B/C become live again, with the corner excluded as confound).

**Definition of done:** all four scenes < 30% on full 500s (empty 0%, corridor 0%, pillars 9.6% already passed), `run_summary.json` carrying reject histograms + accepted-clip stats, E4 dataset complete → proceed to E5 GIF generation.
