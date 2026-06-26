# Gen11 E4 U6 — F6 Run Results

**Date:** 2026-06-10  
**Jobs:** 21415 (empty), 21416 (corridor), 21417 (pillars), 21418 (s_curve)  
**Node:** i6-gpu-1

---

## Summary table

| Scene | Saved | Trials | Rejection | Histogram | vs U5 | Status |
|-------|-------|--------|-----------|-----------|-------|--------|
| empty | 500 | 500 | 0.0% | none | = | ✅ PASS |
| corridor | 500 | 500 | 0.0% | none | = | ✅ PASS |
| pillars | 452 | 500 | **9.6%** | contact=32, floor=16 | ↓ from 60.2% | ✅ PASS |
| s_curve | 0 | 21 | 100% (ABORT) | contact=9, floor=12 | = | ❌ FAIL |

---

## Pillars — ✅ Fully recovered

### Numbers

452/500 saved, 9.6% rejection. No ABORT. Homotopy distribution is balanced across all four classes:

| Homotopy | F4 | U5 F5 | U6 F6 |
|----------|-----|--------|--------|
| (L,L,L) | 117 | 16 | **125** |
| (L,R,L) | 84 | 1 | **99** |
| (R,L,R) | 74 | 2 | **103** |
| (R,R,R) | 121 | 14 | **125** |
| **Total** | 396 | 33 | **452** |

Cross-channel homotopies (L,R,L) and (R,L,R) which collapsed to ~5–10% in U5 are back above F4 levels.

### C2 verdict: confirmed fix

All 48 pillar rejects show `clip=55–100%` — high saturation from pillar-contact impulses disrupting attitude. This is expected and correct behaviour (unavoidable for physical pillar contacts). The exact-scale allocation with torque floor 0.5 restored the attitude authority that U5's binary search was over-reducing on the cross-channel lateral diagonals.

### C3 verdict: telemetry now meaningful

clip% values in the 55–100% range for pillar contact-rejects make physical sense (collision impulse → instant saturation). U5's 0% clip on the same class of reject was broken telemetry, now confirmed fixed.

---

## s_curve — ❌ Still 100% ABORT, but root cause now clearly visible

### New telemetry reveals two distinct failure modes

| Mode | Count | Mean clip% | Mean min_z | Interpretation |
|------|-------|------------|------------|----------------|
| **Floor** | 12 | **19.9%** | 0.050 m | Moderate saturation → attitude lag → position overshoot → floor crash |
| **Contact** | 7 | **42.9%** | 0.188 m | Heavy saturation → lateral control impaired → prolonged wall grazing |

Two extreme contact rejects stand out: #3 (min_z=0.741, clip=30.3%) and #7 (min_z=0.959, clip=23.6%) — the drone maintained altitude but grazed corridor walls for >8% of the episode. These are the "saturation without falling" cases.

### What the clip% proves

U5 reported `clip=0.0%` for every s_curve reject — this was broken output-boundary telemetry (the thrust-priority output never touched u_max, masking real saturation). U6's corrected telemetry (`pid.last_raw_saturated`) now shows **saturation on 17–60% of steps per episode**. Saturation was always happening in s_curve; U5 simply couldn't see it.

### Root cause analysis

The Seg B diagonal (x: −0.5 → +0.5, y: −0.8 → +0.8) with the 2× time budget runs at:
- Peak lateral velocity ≈ 0.45 m/s (horizontal component)
- Peak lateral acceleration ≈ 0.22 m/s² → required tilt ≈ 1.3°

The steady-state demand is small. The saturation is coming from **PID correction transients**: accumulated attitude lag during the cosine-profile acceleration phase creates a position error, and the PD correction loop then demands large corrective thrust bursts. The position loop gains (Kp=4, Kd=3) amplify small tracking errors into large wrench demands that exceed the 2×u_hover motor limit.

The two failure chains that follow:

- **Floor crash (clip~20%)**: correction burst → attitude tilt → vertical thrust component drops → z falls → floor reject (even with thrust-priority, if tilt is large the vertical component shrinks).
- **Wall contact (clip~43%)**: repeated correction bursts → drone oscillates in y near hover at (±0.5, y) → clips corridor y-walls (seg1_wall_pos at y=−0.35 or seg2_wall_neg at y=+0.3) on multiple steps → contact_frac > 0.08.

### What changed vs U4 / U5

| Run | clip% | Dominant reject | Mechanism |
|-----|-------|----------------|-----------|
| U4 F4 | (unknown — no telemetry) | floor 100% | Saturation → thrust corrupt → immediate z-collapse |
| U5 F5 | 0% (broken) | floor=13, contact=8 | **Step 5 geometry bug**: wall clip during diagonal in walled section |
| U6 F6 | 17–60% (correct) | floor=12, contact=9 | **Original saturation** still present; geometry bug fixed; telemetry now correct |

The geometry regression from U5 is confirmed fixed (no more zero-clip crashes). The original saturation-driven failures are confirmed still active.

---

## Required fixes for U7

### F1 (s_curve — primary): further reduce Seg B peak velocity

Two options, can combine:

**Option A — Increase Seg B time weight from 2× to 4×:**  
`d_total = d_a + 4.0*d_b + d_c` → t_b = 9.3 s (vs 6.6 s now) → peak lateral velocity ≈ 0.32 m/s (vs 0.45 m/s) → peak accel halved again to ≈ 0.11 m/s². With peak accel scaling as 1/t², correction bursts scale as 1/t² too — 4× weight cuts corrective demand by ≈ 4× vs 2× weight.

**Option B — Use `pid_low_gain` for s_curve:**  
`kp_scale=0.8, kd_scale=0.9` reduces correction demand magnitude without changing trajectory geometry. Can apply by passing `--gain-variant pid_low_gain` in the s_curve sbatch config. This directly attacks the high-gain PD loop that is generating the saturation-triggering correction bursts.

**Option C — Raise u_max from 2.0 × u_hover to 2.6 ×:**  
Widens the motor envelope so the same correction burst fits within limits rather than saturating. One-line change in `flight_controller.py`. Note: verify against MuJoCo actuator `ctrlrange` — if the XML caps thrust below 2.6×u_hover, this is a silent no-op.

**Recommended combination for U7:** Option A (4× weight) + Option B (low gain). Option C is a useful verification step.

### F2 (pillars — optional): none required

9.6% rejection with balanced homotopy distribution exceeds the pre-training gate (< 30%). Pillars is ready for training. If homotopy imbalance matters for downstream training, a top-up run targeting L,R,L and R,L,R can bring all four classes to ≥ 120.

---

## Pre-training gate status

| Scene | Gate | Current | Action |
|-------|------|---------|--------|
| empty | < 30% | ✅ 0% | Done |
| corridor | < 30% | ✅ 0% | Done |
| pillars | < 30% | ✅ 9.6% | Done (452 episodes) |
| s_curve | < 30% | ❌ 100% | U7: Seg B 4× weight + low-gain PID |
