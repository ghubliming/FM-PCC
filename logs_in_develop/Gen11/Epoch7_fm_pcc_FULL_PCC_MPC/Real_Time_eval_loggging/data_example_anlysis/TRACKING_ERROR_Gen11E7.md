# UAV Gen11E7 — Tracking Error: the measurable test for the `p_des`-drift hypothesis

**Date**: 2026-06-27
**Companion**: [CRITIQUE_three_layer_absurdity.md](./CRITIQUE_three_layer_absurdity.md) (§0–§10) — the *theory*.
This doc is the **empirical** counterpart: what tracking error is, that we already log it, how to read
it to confirm/refute the drift hypothesis, and the decision rule for adding a new `dynamics` constraint.

---

## 1. What "tracking error" is (in DPCC and for us)

The whole `p_des`-vs-`p` story reduces to **one scalar**:

$$ e_t \;=\; \lVert\, p_{\text{des},t} - p_t \,\rVert \quad=\quad \lVert \text{commanded position} - \text{real position} \rVert $$

- In the **DPCC paper** this gap is exactly the **`w_t`** term in the dynamics
  `s_{t+1} = s_t + [a,a]^\top ts + w_t` — the paper's own "**model mismatch** accounting for the
  low-level controller and numerical error" (WHY_FM §13). DPCC openly does **not** model the
  controller; `w_t` is the slack that absorbs the gap. **`w_t` ≈ tracking error.**
- For the **arm**, `e_t ≈ 0` (IK tracks tightly) → `w_t` tiny → the Euler constraint is grounded and works.
- For the **UAV**, the drone **lags** (underactuated, second-order) → `e_t > 0`, possibly growing.

> **Why this one number decides everything (per the CRITIQUE):** the category error (§6.3), the
> "bind `p_des` not `p`" asymmetry (§9), and the drift (§10) are all **invisible when `e_t ≈ 0`** and
> **only bite when `e_t` grows**. So the entire question — *is our `p_des` design actually a problem
> for the UAV?* — is **answered by the distribution of `e_t`**, not by argument.

---

## 2. Your hypothesis, stated precisely

> "The drift = `p_des` losing touch with real `p` = the UAV's PID tracking error."

This is **exactly** `e_t` growing over a rollout. If true:
- `e_t` rises over time (the command runs ahead of the lagging drone), and/or
- `e_t` **spikes before failures** (crash / goal-miss), and/or
- eval `e_t` exceeds the `e_t` seen in **training** (out-of-distribution conditioning, §2.3).

If false (e.g. `e_t` is small and flat and uncorrelated with failure), then the `p_des` design is
**not** the failure cause and the real culprit is elsewhere (FM quality, horizon, termination, etc.).

---

## 3. Do we already log it? **YES** — `track_err` is exactly `e_t`

From `FM_v3_uav_test/eval_fm_uav.py`:

| Granularity | Code | What it is |
|---|---|---|
| per physics sub-step | `track_err.append(‖data.qpos[:3] − p_des‖)` (`:354`) | `e_t` accumulated across the `decim` PID steps |
| per FM control step | `te_step = ‖data.qpos[:3] − p_des‖`; `blog.step(..., track_err=te_step)` (`:357–363`) | `e_t` at each control tick, in the behaviour log |
| per rollout | `track_err_mean = np.mean(track_err)` (`:420`) | rollout-average `e_t` |
| per run | `track_err_mean = mean over rollouts` (`:510`); printed (`:531`) | run-average `e_t` |

So `track_err` **is** `‖p_des − p‖` = the tracking error, and it is persisted per-step (behaviour log)
and aggregated (rollout/run mean). **We do not need new instrumentation to start testing — only to
read what's already there.**

---

## 4. What the logs are MISSING (add these — mean alone can hide the bug)

`track_err_mean` is necessary but **not sufficient**: a rollout that flies cleanly then drifts at the
end has a *low mean* but a *catastrophic tail*. Add:

1. **`track_err_max` and `track_err_final`** per rollout — a growing drift shows here, not in the mean.
2. **The `e_t(t)` time-series** as a saved array (it's in the per-step behaviour log already; just
   surface it for plotting) — to see *bounded vs monotonically growing*.
3. **Training-distribution baseline**: compute the dataset's per-step `‖p_des − p‖` histogram
   (`uav_expert_data_collect` rollouts) once, and overlay. **The decisive comparison is eval-`e_t`
   support vs train-`e_t` support** (§2.3).
4. **Failure correlation**: split `e_t` (mean/max/final) by `result ∈ {SUCCESS, FAIL}` — does `e_t`
   predict failure?
5. **Per-scene breakdown**: `e_t` by scene (`empty / corridor / s_curve / pillars`) and by homotopy —
   aggressive routes (s_curve, pillars) should lag more if the hypothesis holds.

---

## 5. The decision procedure (read the existing logs first)

```
Step 1 — Plot e_t(t) for each rollout (+ max, final).
Step 2 — Overlay the TRAINING ‖p_des − p‖ histogram.
Step 3 — Split e_t by SUCCESS vs FAIL; check per-scene.
```

| Observation | Interpretation | Action |
|---|---|---|
| `e_t` small, flat, ⊆ train support, uncorrelated with FAIL | **Hypothesis FALSE.** `p_des` design is benign for the UAV (like the arm). | Look elsewhere (FM sample quality, horizon, no-termination, goal conditioning). Do **not** touch the constraint. |
| `e_t` grows / spikes before FAIL / exceeds train support | **Hypothesis TRUE.** Drift = PID tracking error; `p_des` runs away from real `p` (OOD). | Add a real `dynamics` treatment (§6) — the command must be tied back to reality. |
| `e_t` borderline (in some scenes only) | Partial — scene-dependent lag. | Start with the cheap fixes (§6.1–6.2); re-measure. |

> **Important:** this is a measurement, not a debate. The CRITIQUE argued the *mechanism*; `track_err`
> gives the *verdict*. Run Steps 1–3 on the Gen11E7 logs before changing any code.

---

## 6. If confirmed: the candidate new `dynamics` constraints (cheapest first)

These mirror [CRITIQUE §10.2](./CRITIQUE_three_layer_absurdity.md). The current `dynamics` constraint
binds `p_des` (a tautology that ignores `e_t`); to make it *respect reality* you need to re-introduce
real `p`:

| # | Treatment | Mechanism | Cost |
|---|---|---|---|
| 1 | **Re-anchor the integrator** | `p_des = (1−α)(p_des + action) + α·p` — bleed the command back toward real `p` each step → bounds `e_t` | software-only, no retrain, no projector change |
| 2 | **Tracking-feasibility band** | add inequality `‖p − p_des‖ ≤ ε` to the projector → it forbids the command running away from real `p` (a *feasible* bind on real `p`, unlike the rigid Euler) | small projector change |
| 3 | **Coarse drone model in projector** | replace the `p_des` tautology with a real command→motion relation so the constraint binds real `p` correctly | medium (needs a rough dynamics model; SafeFlowMPC-style) |
| 4 | **Terminate on `e_t`** (orthogonal safety) | `track_err > τ` for N steps → stop the rollout (the UAV loop currently **never** terminates) | trivial; honest safety net |

**Recommended sequence for Gen11E7:** measure (§5) → if confirmed, ship **#1 + #4** (bound the drift,
stop runaways, no retrain) → measure again → escalate to **#2/#3** only if the projector itself must
enforce reality (e.g. once real obstacle constraints are switched on).

---

## 7. Concrete next steps (Gen11E7)

1. **Read** the Gen11E7 behaviour logs: pull `track_err` time-series per rollout; compute
   `mean / max / final`. (Data already exists — §3.)
2. **Add** `track_err_max` + `track_err_final` to the per-rollout summary and the run summary (§4.1).
3. **Compute** the training `‖p_des − p‖` histogram once and overlay (§4.3).
4. **Split** by SUCCESS/FAIL and by scene (§4.4–4.5).
5. **Decide** via §5's table.
6. If TRUE → implement §6 **#1 + #4**, re-run, compare `track_err` and success.

---

## 8. EVIDENCE — `rollout_corridor_C_10001.log` → verdict: **CONFIRMED**

Read the actual Gen11E7 log (`variant=dpcc-t`, scene=corridor, homotopy=C, 243 steps). It is a
**textbook confirmation** that tracking error is the failure — and it shows the *mechanism*.

**Summary line (verbatim):** `result=FAIL  goal_dist=5.254m  safe=False  min_z=0.085
contact_frac=0.063  contacts=21  max_track_err=2.072m  total_ms mean=145.2 (budget 30.3, over
243/243)`.

### The three phases (with numbers from the log)

| Phase | steps / time | `track_err` | `proj_cost` | what happens |
|---|---|---|---|---|
| **1. Healthy** | 0–10 / t<0.3s | **0.002 → 0.03 m** | ~1 | drone hovers at start, `p_des ≈ p` — **in-distribution** |
| **2. Divergence** | 11–30 / 0.33–0.9s | **0.03 → 0.96 m** (explodes) | **1 → ~13000** (explodes) | FM commands an **aggressive descent** (corridor-C: `p_des.z` 1.19→0.43 fast); drone **can't track** (underactuated), accelerates to `vz≈−4 m/s`, **overshoots → `contact=obstacle` at step 30** |
| **3. Crashed & frozen** | 30–243 / t>0.9s | **0.5 → 2.07 m** (grows unbounded) | **collapses → ~0.7** | drone **stuck at z=0.087** (floor; `min_z=0.085`) for **363 log lines**; FM **keeps commanding** the corridor → the command flies away while the drone sits dead → gap grows monotonically |

### What the log proves

1. **`track_err` IS the diagnostic** — it goes `0.002 → 2.07 m` and tracks the failure exactly
   (`max_track_err=2.072m`). Your hypothesis is **correct**.
2. **The mechanism is an OOD feedback spiral, NOT gradual PID lag.** An aggressive FM command →
   drone can't track → `(p_des, p)` gap goes **out-of-distribution** (training had tight tracking,
   §2.3) → the FM, conditioned on an OOD state, emits worse commands → `proj_cost` explodes → crash.
   This is the §2.3 *pathological case*, observed.
3. **The `dynamics` constraint prevents nothing and costs the most.** `proj_cost` **explodes to
   ~13000 during the divergence** (it flags trouble but cannot stop it — it binds `p_des`, §9.8),
   then **collapses to ~0.7 after the crash** (the drone is dead but the constraint says "fine").
   And `proj_ms ≈ 60 ms/step` is **half** the 145 ms budget blow-out — paying the most to help least.
4. **"FM keeps planning" (the file's title) is literally shown.** Post-crash, the frozen drone gets
   corridor commands forever (363 frozen lines); **there is no termination.**
5. **`proj_cost` as a health signal is misleading** — high during the (recoverable) divergence, low
   after the (unrecoverable) crash. Don't trust it; trust `track_err`.
6. **Real-time violation compounds it.** `total_ms=145` vs `budget=30.3` → running ~7 Hz, not the
   33 Hz it was trained at → each command is held ~5× longer → drone travels further per command →
   bigger gaps. The useless 60 ms projection is the prime thing to cut.

### Verdict & what to do (this rollout decides it)

> **Tracking error is the problem — confirmed.** Not as a slow leak, but as an **OOD spiral**:
> aggressive command → untrackable → gap → OOD conditioning → degradation → crash → unbounded drift.
> The `dynamics` constraint as configured (binds `p_des`) is **part of the cost, not the cure.**

Act on §6, in this order (this log argues for exactly these):
1. **Re-anchor the integrator** `p_des = (1−α)(p_des+act) + α·p` (#1) — caps the gap so it can't go
   OOD, **rate-limiting the descent to what the drone can actually track**; most likely to break the
   spiral *before* the crash. No retrain.
2. **Terminate on `track_err`** (#4) — kills the post-crash "keep planning" garbage immediately.
3. **Fix real-time** — cut the 60 ms projection (e.g. `post_processing` 1× instead of interleaved,
   §10.4 of the critique) or speed up the FM, to restore 33 Hz and shrink per-command travel.
4. Only then consider **#2/#3** (feasibility band / real model in the projector) if needed.

> [!NOTE]
> **Caveat — single rollout, one scene/homotopy.** This is `corridor_C`, which demands a descent
> (the hard maneuver). Before committing, confirm the pattern across scenes (especially `pillars`,
> `s_curve`) and check whether `empty`/easy routes stay in Phase 1. But the **mechanism** here is
> unambiguous, and it matches every prediction in the CRITIQUE.

---

## 9. "Just use `p` instead of `p_des`" — yes, and you spotted the missing piece (change the *action* too)

Your instinct is right, and the crucial addition is the one you made: **"maybe also the action."**
Switching the *constraint binding* to `p` **alone** fails (§9.7 / CRITIQUE §10.2 — the rigid
`p=∫act` fights the FM's lagging-`p` channel). But switching the **action** to `Δp` at the same time
makes it work, because:

> **The Euler relation is a tautology for whatever variable the action is the increment of.**
> - action = `Δp_des` ⇒ `p_des = ∫act` is tautological ⇒ you can only bind `p_des` (today).
> - action = `Δp` ⇒ `p = ∫act` is tautological ⇒ you can bind **real `p`**, *feasibly*, and
>   `skip_initial_state` then pins `p[0] =` the **measured** position → the constraint is **grounded
>   in reality** instead of in the command.

### The full change set ("plan in real position")

| Piece | today (command-centric) | proposed (`p`-centric) |
|---|---|---|
| obs | `[p_des｜p｜v]` | `[p｜v]` (drop the command channel) |
| action | `Δp_des` (command delta) | **`Δp`** (real-position delta) |
| constraint `deriv` | binds `p_des` (tautology, ignores reality) | binds **`p`** (tautology *for the new action* → feasible **and** grounded) |
| eval integration | `p_des += action` (free-running, can run away → the corridor crash) | setpoint `= p + Δp` (**re-anchored to the measured position every step → cannot run away**) |

This structurally **kills the §8 failure**: the setpoint is always "from where I actually am, move a
little," so the command can never fly 0.5 m ahead of a lagging drone, so `(command,real)` can never
go OOD, so the spiral can't start.

### Why WE can do this and DPCC/D3IL could not (and why the paper "can't legalize" `p_des`)

- **We record real `p`** in the dataset (`obs=[p_des｜p｜v]`), so we can **recompute** `action = Δp =
  diff(p)` directly. D3IL's **gamepad** data only ever contained the human's *commands* (`Δdes`) —
  there was no clean "real-Δp" to train on, and the arm tracks so tightly that `Δdes≈Δp` anyway.
- So **the DPCC paper never needed to "legalize" `p_des`** — on the arm it **binds the real position
  `c_pos`** (the Euler holds because the arm tracks). **We** are the ones who bound `p_des`, forced
  by (a) inheriting the `Δdes` action and (b) having no drone model. *Switching to `p` is therefore
  not a hack — it is returning to the paper's own arm design, adapted for a lagging plant by making
  the action `Δp`.*

### The honest catch: lead vs lag (don't expect a free lunch)

Pure replacement (`setpoint = p + Δp`) **grounds** everything but tends to **under-track**: to make
the drone actually *achieve* displacement `Δp`, the PID needs a setpoint that **leads** the drone by
the lag; commanding exactly `p + Δp` does not lead, so the drone moves a bit *less* than intended →
**safe but sluggish** (trades the current *crash* risk for a *goal-miss/slow* risk). The current
scheme has the opposite failure — it **over-leads** (free-running `p_des` → runaway → crash, §8).

**The sweet spot is the re-anchor (§6 #1), which keeps the lead but caps the runaway:**
`p_des = (1−α)(p_des + action) + α·p` — the command still *leads* the drone (so it moves), but is
continuously bled toward the **measured** `p` (so it cannot run away / go OOD). And it needs **no
retrain** (works with the existing FM and action). Full correctness — *lead* **and** *grounded* **and**
*feasible constraint* — is the **real-model** route (SafeFlowMPC §7), which is the most work.

### Recommendation (ordered)

1. **Re-anchor first** (`#1`, no retrain) — likely fixes the §8 spiral immediately; cheap to test.
2. If you retrain anyway, **plan-in-`p`** (action `Δp`, bind real `p`) — the principled grounding;
   add a small **overdrive/lead** (`setpoint = p + k·Δp`, `k>1`) or a coarse PID model to avoid
   under-tracking.
3. **Real model in the projector** (`#3`) — the SafeFlowMPC-grade fix; lead + grounded + feasible.
4. Always: **terminate on `track_err`** (`#4`) so a failure can't turn into 363 frozen-drone steps.

> **One-line answer to "why not `p`?":** you can, and you *should* (it's the paper's own arm design);
> it just isn't a config flip — it needs the **action redefined to `Δp`** (a retrain) to keep the
> constraint feasible, and a bit of **lead** so the grounded command doesn't under-track. The
> no-retrain stand-in that captures 90% of the benefit is the **re-anchor**.

---

## 10. References

| File | Line | Shows |
|---|---|---|
| `FM_v3_uav_test/eval_fm_uav.py` | 301, 354, 357–363 | `track_err = ‖qpos[:3] − p_des‖`, per-step, logged to behaviour log |
| `FM_v3_uav_test/eval_fm_uav.py` | 420, 510, 531 | `track_err_mean` per-rollout / per-run / printed |
| `FM_v3_uav_test/eval_fm_uav.py` | 338, 347 | `p_des += action`; PID `pid.compute(p,q,v,om,p_des,v_des)` — the lag source |
| `CRITIQUE_three_layer_absurdity.md` | §0–§10 | the theory: why `e_t` is the deciding quantity, and the fix menu |
| `WHY_FM_KEEPS_PLANNING.md` | §13 | DPCC paper's `w_t` = "model mismatch (low-level controller + numerical)" = tracking-error slack |
| `uav_expert_data_collect/dataset_writer.py` | — | training obs `[p_des｜p｜v]` → source for the training `‖p_des−p‖` baseline histogram |
