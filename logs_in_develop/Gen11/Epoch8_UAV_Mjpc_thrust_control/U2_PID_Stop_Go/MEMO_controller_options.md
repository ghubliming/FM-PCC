# MEMO — UAV Controller Options (9D vs 12D, PID vs MJPC)

**Context:** Discussion of what happens to `v_des` when `v` is dropped from the FM tensor.
**Conclusion:** Option 4 (9D+PID, v_des=Δp/dt) rejected. Option 2 (this U2) chosen as next test.

---

## The 4 Options

### Option 1 — 12D + PID, `v_des = Δp_des / dt_fm` ✅ Current E7
- FM obs: `[p_des | p | v]` (9D), transition 12D
- FM **sees real `v`** → planner knows momentum → generates smoother waypoints
- PID gets `v_des = action / dt_fm` feedforward → does not brake between waypoints
- **Continuous flight.** FM and PID are implicitly consistent even with timing jitter (FM conditioned on real v, adapts next action accordingly)
- **Status: working baseline**

---

### Option 2 — 9D + PID, `v_des = 0` ⬅ This U2 (next test)
- FM obs: `[p_des | p]` (6D), transition 9D
- FM has no velocity info → cannot account for momentum in planning
- PID gets `v_des = 0` → velocity loop error = `v_real - 0 = v_real` → PID actively brakes to zero at every FM step
- **Strict stop-and-go by construction.** Simpler than MJPC, no new dependencies.
- **Status: implement and test — see PLAN_PID_Stop_Go.md**

---

### Option 3 — 9D + MJPC ⚠ Original E8 plan
- FM obs: `[p_des | p]` (6D), transition 9D
- FM stop-and-go at planning level (no v conditioning) — same root problem as Option 2
- MJPC jointly optimizes velocity within its MPC horizon → may reduce inter-waypoint braking at the controller level, but cannot fix what FM doesn't plan for
- Expensive: ~50× slower than PID, gRPC server, open `task_id` question (U11 §4.1)
- **Status: deprioritised — if Option 2 already stop-and-go, MJPC overhead is not justified**

---

### Option 4 — 9D + PID, `v_des = Δp_des / dt_fm` ❌ Rejected
- Attempts to reconstruct velocity feedforward without having `v` in the FM tensor
- **Timing-sensitive:** `dt_fm` is nominally `1/33s` but FM inference jitter (even 1 ms) shifts the real gap → `v_des` is wrong, PID chases incorrect velocity
- No FM compensation: in Option 1, the FM sees real `v` and adapts; here the FM generated the action blind, so there is no self-correction
- **Conclusion: unreliable by design. Rejected.**

---

## Summary Table

| | FM obs | Transition | v_des to PID | Stop-and-go? | Status |
|---|---|---|---|---|---|
| **Option 1** E7 | `[p_des\|p\|v]` | 12D | `Δp_des/dt_fm` | No — continuous | ✅ Baseline |
| **Option 2** U2 | `[p_des\|p]` | 9D | `0` | Yes — strict | ⬅ Test next |
| **Option 3** MJPC | `[p_des\|p]` | 9D | ignored | Yes at FM level | ⚠ Deprioritised |
| **Option 4** | `[p_des\|p]` | 9D | `Δp_des/dt_fm` | No — but unreliable | ❌ Rejected |

**Key insight:** `v` in the FM tensor does two jobs — FM planning (momentum awareness) and implicit timing self-correction. Dropping it to 9D breaks both; no controller can recover this at the tracker level.
