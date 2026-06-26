# Gen11 Epoch4 U4 — Post-Fix_2 Re-collection Evaluation

**Date:** 2026-06-09  
**SLURM jobs:** 21368 (empty), 21369 (corridor), 21370 (s_curve), 21371 (pillars)  
**Temp path:** `temp/Gen11E4U3/F3/`  
**Fix applied:** Fix_2 — Z_FLOOR_MARGIN = 0.50 m in `generator.py`  
**Status:** Partial success — 2/4 scenes clean, 1 aborted, 1 has corrupt PKLs

---

## 1. Collection Summary

| Scene | Target | Saved | Rejected | Rej. Rate | Elapsed | Status |
|---|---|---|---|---|---|---|
| empty    | 500 | 500 | 0   | 0.0%  | 76 s  | ✅ Complete |
| corridor | 500 | 500 | 0   | 0.0%  | 97 s  | ✅ Complete |
| pillars  | 500 | 396 | 104 | 20.8% | 147 s | ⚠️ 79.2% of target |
| s_curve  | 500 | 2   | 19  | 90.5% | 8 s   | ❌ ABORT (limit 60%) |
| **Total**| 2000| **1398** | 123 | — | — | — |

---

## 2. Fix_2 Verification — min_z Check

Ran PKL inspection (snippet 2 from `U3/PKL_INSPECT.md`) on all readable episodes.

| Scene | Readable eps | min_z < 0.50 (floor crash) | Worst min_z | Verdict |
|---|---|---|---|---|
| empty    | 500 | **0** | 0.700 m | ✅ Fix_2 working |
| corridor | 500 | **0** | 0.701 m | ✅ Fix_2 working |
| pillars  | 272 | **0** | 0.521 m | ✅ Fix_2 working |
| s_curve  | 2   | **0** | 0.682 m | ✅ (trivially — 2 saved) |

**Fix_2 verdict: Correct.** Zero floor crashes in any saved episode. The 0.521 m minimum
(pillars, L_R_L) is 0.021 m above the 0.50 m threshold — tight but valid.

Normal hover altitude is 0.7–1.1 m. The 0.50 m floor never triggers on clean
episodes. The contamination from E4 U3 (27%) is now correctly rejected.

---

## 3. Critical Issues Found

### Issue A — s_curve: 90.5% rejection → ABORT ❌

**Root cause:** Fix_2 exposed a fundamental instability. The s_curve PID drops below
z = 0.50 m on >90% of trials. The collection aborted after only 21 attempts (2 saved,
19 rejected) when the rolling rejection rate exceeded the 60% safety limit.

In E4 U3, these same episodes had `contact_fraction=0` (no obstacle contacts, floor
excluded by `_is_obstacle_contact`) and were silently saved. Fix_2 correctly rejects
them, but the result is an effectively unusable scene for data collection.

**What needs investigation (separate fix):**

| Hypothesis | Implication |
|---|---|
| Trajectory too aggressive (large lateral S-moves) | Reduce speed or waypoint amplitude in `s_curve` trajectory generator |
| PID gain too weak for altitude hold during lateral accel | Tune `kp_z` / `ki_z` specifically for s_curve |
| Z_FLOOR_MARGIN too tight for s_curve flight profile | Lower Z_FLOOR_MARGIN to ~0.30 m only for s_curve (dangerous — risk of accepting real crashes) |

Recommendation: investigate trajectory parameters first. The s_curve path likely
demands lateral movement that causes the drone to sacrifice altitude.

### Issue B — Pillars R_R_R: 121/396 PKLs corrupt (truncated) ⚠️

All 121 R_R_R homotopy PKLs are unreadable (`pickle.load` → `EOFError: Ran out of input`).
Three additional L_L_L files are also truncated. Total: 124 corrupt files.

| Homotopy | Files | OK | Corrupt |
|---|---|---|---|
| (L,L,L) | 117 | 114 | 3 |
| (L,R,L) | 84  | 84  | 0 |
| (R,L,R) | 74  | 74  | 0 |
| (R,R,R) | 121 | **0** | **121** |

The R_R_R job ran on node i6-gpu-1 (job 21371, same job as other homotopies). The PKLs
were written on the cluster and transferred to the local temp path. The truncation suggests
the `rsync`/`scp` transfer was interrupted or the R_R_R write was partially buffered
(all 121 files are present but empty/truncated — the file handles exist but data was lost).

**Actions needed:**
1. Re-copy R_R_R PKLs from `logs/uav_expert_data/pillars/R_R_R/` on the cluster.
2. If originals are also corrupt (same node write issue), re-run pillars R_R_R only
   (targeted `--homotopy R_R_R` flag if supported, or full pillars re-run).

### Issue C — Pillars speed spikes in accepted episodes ℹ️

98/272 readable pillars episodes (36.0%) have at least one step with speed > 2.5 m/s.
The worst is 6.287 m/s at t=465/470 (pillars_R_L_R_pid_default_0000210).

| Threshold | Episodes above | % |
|---|---|---|
| > 1.5 m/s | 107 / 272 | 39.3% |
| > 2.0 m/s | 103 / 272 | 37.9% |
| > 2.5 m/s | 98 / 272  | 36.0% |
| > 3.0 m/s | 95 / 272  | 34.9% |
| > 4.0 m/s | 54 / 272  | 19.9% |
| > 5.0 m/s | 26 / 272  |  9.6% |

**Pattern:** Speed spikes occur at the very last ~10 steps of the episode (t=T-10 to T),
as the drone decelerates to the final waypoint. The z-height remains within normal range
(0.71–0.97 m during spike). These are **not floor crashes** — they are end-of-episode
deceleration transients where the PID generates large corrective forces to arrest motion.

**Impact for training:** The model will see brief high-velocity frames at episode end.
These are physically consistent (not corrupted) but may not represent desired behaviour.
Fix_2 does not address this — a separate max-speed rejection filter would be needed if
this is deemed problematic.

---

## 4. Speed and Trajectory Quality (Clean Episodes)

### empty (500 eps)
- Speed: mean=0.387, median=0.421, p95=0.625 m/s ✅ within target 0.30–0.50 m/s
- Min z across all: 0.700 m — drone never goes below hover baseline

### corridor (500 eps)
- Speed: mean=0.695, median=0.750 m/s — slightly above 0.50 target but acceptable
  (corridor requires longer straight runs at higher speed)
- Homotopy balance: L=167, C=167, R=166 — near-perfect ✅

### pillars (272 readable)
- Speed: mean=0.711, median=0.726 m/s — consistent with corridor
- Homotopy balance (readable): (L,L,L)=114, (L,R,L)=84, (R,L,R)=74, (R,R,R)=0
  — heavily imbalanced due to R_R_R corruption; NOT usable for training until fixed
- Worst-case min_z = 0.521 m → still passes Fix_2 with 21 mm margin

---

## 5. Overall Verdict

| Assessment | Detail |
|---|---|
| Fix_2 correctness | ✅ Working — zero floor crashes in saved episodes |
| Dataset cleanliness | ✅ All 1272 readable episodes have min_z > 0.50 m |
| s_curve scene | ❌ Blocked — needs separate PID/trajectory investigation |
| pillars dataset | ❌ Not ready — R_R_R class lost, re-copy needed |
| empty + corridor | ✅ Ready for E6 training |
| Speed spikes (pillars) | ℹ️ Informational — present but not from floor crashes |

**Dataset is NOT ready for E6 training in its current state.**  
Minimum requirements to unblock:
1. Fix s_curve collection (90.5% rejection is a hard blocker)
2. Recover pillars R_R_R (either re-copy or re-run)

---

## 6. Comparison to E4 U3 (Pre-Fix_2)

| Metric | E4 U3 (contaminated) | E4 U4 (Fix_2 applied) |
|---|---|---|
| Floor crashes in dataset | ~492/1829 (27%) | **0/1272 (0%)** |
| s_curve saved | ~329 (85% contaminated) | 2 (90.5% reject rate) |
| Pillars saved | ~500 (65–76% contaminated) | 396 (20.8% reject) |
| Worst min_z | ~0.02–0.10 m (floor crashes) | 0.521 m |
| empty + corridor | Clean (no obstacle z-crash) | Clean |

Fix_2 dramatically improved cleanliness. The high rejection rates confirm the E4 U3
contamination estimates were accurate — the fix is working as intended.

---

## 7. Next Steps

| Priority | Action |
|---|---|
| P0 | Re-copy or re-run `pillars/R_R_R` (121 episodes, ~45 s) |
| P1 | Investigate s_curve PID stability — why does z drop < 0.50 m so often? |
| P2 | After pillars fixed: verify homotopy balance (target ~99 each) |
| P3 | Consider max-speed filter (> 3.0 m/s?) to clean deceleration spikes |
| P4 | Once all 4 scenes clean: proceed to E5 full GIF generation |
