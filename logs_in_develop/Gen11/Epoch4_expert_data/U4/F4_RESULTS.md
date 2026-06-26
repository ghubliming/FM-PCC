# Gen11 Epoch4 U4 — F4 Run Results

**Date:** 2026-06-10
**SLURM jobs:** 21398 (empty), 21399 (corridor), 21400 (s_curve), 21401 (pillars)
**Node:** i6-gpu-1
**Temp path:** `temp/Gen11E4U3/F4/`
**Fixes deployed:** Fix A (s_curve hover pauses + duration [18,24]s) + Fix B (R_R_R homotopy filter)
**Prior state:** F3 — 2 blockers: s_curve 90.5% rejection ABORT, pillars R_R_R 121 corrupt PKLs

---

## 1. Collection Summary

| Scene    | Target | Saved | Rejected | Rej. Rate | Elapsed | Status |
|----------|--------|-------|----------|-----------|---------|--------|
| empty    | 500    | 500   | 0        | 0.0%      | 77 s    | ✅ Complete |
| corridor | 500    | 500   | 0        | 0.0%      | 97 s    | ✅ Complete |
| pillars  | 500    | 396   | 104      | 20.8%     | 145 s   | ⚠️ 79.2% of target |
| s_curve  | 500    | 0     | 21       | 100.0%    | 9 s     | ❌ ABORT |
| **Total**| 2000   | **1396** | 125   | —         | —       | — |

---

## 2. Fix B — R_R_R Recovery: ✅ SUCCESS

Fix B (homotopy filter `$5` argument in `collect.sh`) worked. Pillars R_R_R PKLs are now
present and intact.

| Homotopy | F3 (corrupt) | F4 (this run) |
|----------|-------------|---------------|
| (L,L,L)  | 114 readable / 3 corrupt | **117** ✅ |
| (L,R,L)  | 84 readable              | **84** ✅ |
| (R,L,R)  | 74 readable              | **74** ✅ |
| (R,R,R)  | **0 readable / 121 corrupt** | **121** ✅ |
| **Total**| 272 readable | **396** ✅ |

Homotopy distribution: L_L_L=117, L_R_L=84, R_L_R=74, R_R_R=121. Slight imbalance on
L_R_L/R_L_R (lower count) but acceptable per FIX_PLAN.md.

### Pillars Speed Spike (Fix C — still deferred)

The max speed in accepted episodes is **6.60 m/s** (p95 = 1.28 m/s, mean = 0.71 m/s).
This is identical to F3 — Fix A/B did not affect pillars dynamics. Speed spikes remain
at the last ~10 steps of each episode (end-of-episode deceleration transient). No floor
crashes. Fix C (max-speed filter) is still deferred — see `FIX_PLAN.md §Fix C`.

---

## 3. Fix A — s_curve Hover Pause: ❌ FAILED

**Result:** 0 saved, 21 rejected, 100% rejection → ABORT (limit 60%).

This is worse than F3 (90.5%) — the hover pause fix did not resolve the s_curve blocker.

### Timing Analysis

| Run | sec/episode (aborted) | Saved | Rejection |
|-----|-----------------------|-------|-----------|
| F3 (no hover) | 8.9 s | 2 | 90.5% |
| F4 (hover, dur=[18,24]s) | 8.9 s | 0 | 100.0% |

Both runs abort episodes at ~8.9 s on average. This means the floor crash is occurring at
the same wall-clock time regardless of the hover pause.

### Why the Hover Fix Didn't Work

With `T=18s`, `T_HOVER=1.0s`, `T_move=16s`:
- Seg A ends (hover 1 begins) at `t_a = 16 × 2.7/7.28 ≈ 5.93 s`
- Hover 1 ends (Seg B begins) at `t ≈ 6.93 s`
- Floor crash occurs at `t ≈ 8.9 s` → **~2.0 s into Seg B diagonal**

The hover pause correctly stabilises the drone at the junction — but the altitude collapse
is happening **during Seg B itself** (the 1.89 m diagonal crossing), not from overshoot
momentum entering the junction. The original FIX_PLAN diagnosis identified the correct
symptom (lag-overshoot at junction) but the altitude drop is caused by the **diagonal
traverse dynamics**, not the junction transition. The hover zeros out accumulated y-velocity
but does not help the drone maintain altitude during the subsequent y-acceleration phase
of Seg B.

### Revised Root Cause Hypothesis for s_curve

The diagonal Seg B requires simultaneous x and y acceleration. On this UAV model, the
combined thrust demand during diagonal acceleration (PID must hold z while accelerating
laterally) consistently drops z below 0.50 m by ~2 s into the crossing. The hover pause
is orthogonal to this problem — it affects entry velocity but not the in-flight thrust
budget during the crossing.

### Candidate Fixes for s_curve (Next Iteration)

| ID | Approach | Rationale |
|----|----------|-----------|
| F5-A | Increase initial z range (`[0.90, 1.30]` from `[0.70, 1.10]`) | More altitude headroom during the 0.30–0.45 m z-drop |
| F5-B | Reduce diagonal speed (longer `t_b` budget for Seg B) | Less lateral acceleration demand → less z-thrust sacrifice |
| F5-C | Split Seg B into step + straight (fly y first, then x) | Eliminates simultaneous x+y demand entirely |
| F5-D | Reduce `Z_FLOOR_MARGIN` to 0.30 m for s_curve only | Accepts crashes down to 0.30 m — risky, changes dataset character |
| F5-E | Increase `T_HOVER` from 1.0 s to 2.0 s | More time for z to recover after Seg A before crossing (doubtful — crash is during Seg B, not at junction) |

**Recommended: F5-A first** (altitude headroom change is a 1-line fix, low risk). If
that brings rejection below 60%, follow with F5-B to further reduce rate.

---

## 4. Dataset Status After F4

| Scene    | Episodes | Homotopy balance | Min z  | Speed p95 | Training-ready |
|----------|----------|-----------------|--------|-----------|----------------|
| empty    | 500      | N/A (500)       | 0.700 m | 0.625 m/s | ✅ Ready |
| corridor | 500      | L=167, C=167, R=166 | 0.701 m | 1.227 m/s | ✅ Ready |
| pillars  | 396      | L_L_L=117, L_R_L=84, R_L_R=74, R_R_R=121 | 0.521 m | 1.280 m/s | ⚠️ Low count (79.2%) + speed spikes |
| s_curve  | 0        | —               | —      | —         | ❌ Blocked |
| **Total**| **1396** | — | — | — | **Not ready** |

**Dataset is NOT ready for training.** s_curve remains a hard blocker.
Pillars is usable at reduced count (396 vs 500 target) but speed spikes are unfiltered.

---

## 5. Comparison Across Runs

| Metric | F3 | F4 | Delta |
|--------|----|----|-------|
| empty saved | 500 | 500 | — |
| corridor saved | 500 | 500 | — |
| pillars readable | 272 | 396 | **+124 (R_R_R recovered)** ✅ |
| pillars R_R_R | 0 | 121 | **+121** ✅ |
| s_curve saved | 2 | 0 | -2 (regression) |
| s_curve rejection | 90.5% | 100.0% | worse ❌ |
| Total readable | 1272 | 1396 | +124 |

---

## 6. Next Steps

| Priority | Action |
|----------|--------|
| P0 | Implement F5-A: raise s_curve initial z range to `[0.90, 1.30]` — 1-line change |
| P0 | Smoke test s_curve (20 trials) after F5-A |
| P1 | If smoke > 60% rejection after F5-A: also apply F5-B (slow down Seg B) |
| P2 | If s_curve < 30% rejection: full 500-trial run |
| P3 | Decide on pillars speed spike filter (Fix C) — deferred, low priority |
| P4 | Once all 4 scenes have ≥ 400 clean episodes: proceed to E5 GIF generation |
