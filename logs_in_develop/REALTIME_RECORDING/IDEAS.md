# Real-Time Behavior Recording — Digital Twin Audit Framework

**Date**: 2026-06-08  
**Scope**: Evaluation framework for FM-PCC class systems — FMv3Ode, Visual Aligning,
DPCC baseline, future Drone FM-PCC. Applies to any system before real-world deployment.

---

## The real problem — timing

We already have GIFs. The problem is not visualisation — it is **timing and real-time
feasibility**.

A real robot closes its control loop at a fixed frequency. For the UAV this is 33 Hz —
each step has a budget of **30 ms**. For Visual Aligning it depends on the task loop rate.
If any single component in the pipeline takes longer than that budget, the system cannot
run in real time. It is not deployable, full stop.

Consider the worst case:

```
Visual Aligning episode = 400 steps
FM inference per step   = 20 s       ← hypothetical slow model

Total inference time    = 400 × 20 s = 8000 s ≈ 2.2 hours
```

A system like that is completely unusable on real hardware, even if the trajectory is
perfect. **The GIF will look fine. The timing log will show it is broken.**

This is the primary question the framework answers:

> **Can this system close its control loop fast enough to run on real hardware?**

The secondary question: **how does it compare to the DPCC baseline?** DPCC has no FM
inference — its per-step cost is only the QP solve. That is the reference. If FM + QP
takes significantly longer than QP alone, we need to know by how much and whether it
fits the hardware budget.

---

## Per-step compute chain

Every step runs this pipeline in sequence. Each box adds latency:

```
observation
    → FM inference (ML model)          [latency: fm_ms]
        → DPCC / QP solve              [latency: qp_ms]
            → PID execution            [latency: pid_ms]   (negligible, ~0.1 ms)
                → physics step         [latency: sim_ms]   (not on real hardware)

Total per-step wall time = fm_ms + qp_ms + pid_ms
Real-time budget         = 1000 / control_hz   (ms)

If total > budget → system cannot close the loop on hardware.
```

DPCC baseline: `fm_ms = 0`, total = `qp_ms` only. This is the minimum possible cost.
FM-PCC: total = `fm_ms + qp_ms`. The question is whether the overhead of FM inference
is acceptable given the performance gain it provides.

---

## What needs to be recorded per step

Timing is the primary field. Everything else provides context for why timing is what it is.

| Component | Primary: timing | Context fields |
|---|---|---|
| **FM inference** | `fm_ms` — wall-clock ms for this step's inference | Predicted horizon `[Δp_des_0…H-1]` |
| **DPCC / QP solve** | `qp_ms` — wall-clock ms for this step's QP | Active/idle, which constraint, override delta |
| **Total step** | `total_ms = fm_ms + qp_ms` ← **the critical number** | vs budget `1000/hz` |
| **Observation** | — | `[p_des, p, v]` fed to FM |
| **State after step** | — | `p, v, q` after physics |
| **Contact** | — | Geom, duration, severity |
| **Tracking error** | — | `|p - p_des|` |

A GIF captures none of this. A structured text log captures all of it.

---

## Per-step log format

Timing is on every step line. One block per timestep:

```
═══ T=2.130s  total_ms=5.3  [BUDGET=30ms ✅] ════════════════════════════
OBS       p_des=(1.820,-0.210,0.901)  p=(1.793,-0.218,0.899)  v=(0.721,0.012,0.003)
          track_err=0.028m

FM        fm_ms=4.2   horizon=[(0.022,-0.001,0.000),(0.022,-0.001,0.000),(0.021,0.000,0.000),(0.020,0.001,0.000)]

DPCC      qp_ms=1.1   status=IDLE   nearest=wall_y_neg  margin=0.282m

STATE     p=(1.815,-0.216,0.901)  v=(0.724,0.009,0.003)  contact=NONE

═══ T=2.160s  total_ms=6.4  [BUDGET=30ms ✅] ════════════════════════════
OBS       p_des=(1.842,-0.211,0.901)  p=(1.815,-0.216,0.901)  v=(0.724,0.009,0.003)
          track_err=0.030m

FM        fm_ms=4.1   horizon=[(0.021,0.002,0.000),...]

DPCC      qp_ms=2.3   status=ACTIVE  constraint=wall_y_neg  margin=0.071m
          fm_cmd=(−0.002,−0.008,0.000)  →  qp_cmd=(0.021,0.003,0.000)   ← FM laterally overridden

STATE     p=(1.836,-0.212,0.901)  v=(0.729,0.003,0.003)  contact=NONE   ← DPCC prevented contact

═══ T=4.500s  total_ms=31.8  [BUDGET=30ms ❌ OVER] ══════════════════════
...
FM        fm_ms=30.2   ← spike — model inference slow on this step
CONTACT   geom=wall_y_neg   fm_cmd_at_contact=(−0.001,−0.009,0.000)
          dpcc_at_contact=ACTIVE   track_err_at_contact=0.044m

# ─────────────────────────────────────────────────────────────────────
# SUMMARY  episode=corridor_L_pid_default_0000056  system=Drone_FMPCC
#          steps=207   duration=6.23s   control_hz=33   budget_ms=30
#
#          TIMING:
#            fm_ms     mean=4.2  max=30.2  p95=5.1  over_budget=1/207 (0.5%)
#            qp_ms     mean=1.4  max=3.1   p95=2.2
#            total_ms  mean=5.6  max=31.8  p95=7.0  over_budget=1/207 (0.5%)
#
#          vs DPCC_baseline (same episode):
#            qp_ms_baseline  mean=1.4  max=3.2   total_ms  mean=1.4
#            fm_overhead     mean=+4.2ms/step (+300%)   acceptable=YES (fits budget)
#
#          BEHAVIOUR:
#            result=SUCCESS  contacts=2  contact_fraction=0.014
#            dpcc_active_steps=12/207 (5.8%)   dpcc_overrides=9
#            max_track_err=0.051m
```

The `# SUMMARY TIMING` block is the first thing to read. If `over_budget` is non-zero or
`total_ms mean` is close to `budget_ms`, the system is not real-time safe.

One-liner to check timing across all episodes:
```bash
grep "total_ms  mean" logs/eval/**/*.log | sort -t= -k2 -n
```

---

## The digital twin purpose — questions to answer before real hardware

The text log is the **digital twin** of a real deployment run. Before flying a real drone
or running on a real robot, these questions must have answers:

### Primary — timing (will it work at all?)

1. **"Does FM inference fit the real-time budget?"**
   — `total_ms mean` vs `budget_ms` in `# SUMMARY TIMING`. If mean total > budget,
   the system cannot close the loop. If p95 > budget, it will miss deadlines 5% of steps.
   A Visual Aligning episode with 400 steps × 20 s/step = 8000 s. Not deployable.

2. **"How much slower is FM-PCC than DPCC baseline?"**
   — `fm_overhead mean` in `# SUMMARY`. DPCC alone costs only `qp_ms`. FM adds `fm_ms`
   on top. If the overhead is +4 ms at 33 Hz (30 ms budget), fine. If it is +25 ms, not fine.

3. **"Are there inference spikes?"**
   — `fm_ms max` and `p95` in `# SUMMARY`. A mean of 5 ms with a max of 200 ms is still
   broken — the spike will cause a missed deadline and a control gap on hardware.

### Secondary — behaviour (does it work correctly?)

4. **"Is DPCC doing real work, or is FM already safe?"**
   — `dpcc_active_steps` fraction. Low = FM is safe. High = FM is unsafe and relying on
   DPCC. Very high = FM is broken, DPCC is compensating for everything.

5. **"When contact happens, was it FM error, DPCC failure, or PID lag?"**
   — `CONTACT` block fields: `fm_cmd_at_contact` shows what FM asked for,
   `dpcc_at_contact` shows whether safety filter was active, `track_err_at_contact` shows
   whether PID could not follow the command. Three different root causes, one text grep.

6. **"Compare FMv3Ode vs DPCC baseline on the same episode."**
   — Run both systems from the same seed, same scene. The `# SUMMARY` line is identical
   in structure — diff or feed to an LLM side by side.

---

## Why GIF alone is not enough for this

| Question | GIF | Text log |
|---|---|---|
| Did FM predict a safe path? | ❌ invisible | ✅ `horizon_0..H` per step |
| Did DPCC override FM? | ❌ invisible | ✅ `DPCC status=ACTIVE` + delta |
| Was contact caused by FM or PID lag? | ❌ cannot distinguish | ✅ `fm_cmd_at_contact` vs `tracking_err_at_contact` |
| Will inference fit real-time budget? | ❌ no timing info | ✅ `inference_ms` per step |
| Compare two systems quantitatively | ❌ subjective | ✅ diff / grep / LLM |

GIF = intuition. Text log = evidence.

---

## Can we load existing outputs, or must we re-eval?

A natural shortcut: we already have eval outputs for past generations (`.npz` result
files, episode `.pkl` trajectories, GIFs). Can we just **load** them and produce this
digital-twin log retroactively, or do we have to **re-run evaluation** with the logger
attached?

The answer splits by field category, and the split is hard:

### Timing fields — ALWAYS require re-eval (cannot ever be loaded)

`fm_ms`, `qp_ms`, `total_ms` are **wall-clock measurements**. They exist only at the
instant the model runs. A saved trajectory records *what* the model produced, never *how
long it took*. There is no field in any `.npz` or `.pkl` from which latency can be
recovered — it was never a number, it was a duration that elapsed and was discarded.

> **Timing is the primary product of this framework, and timing is the one thing that
> can never be reconstructed offline.** Every generation, no exceptions, must be re-run
> with the logger to get its timing block.

This also means timing is **hardware-bound**: a re-eval on the cluster GPU gives cluster
latency, not the latency of the target deployment hardware. To answer "will it fit the
real drone's 30 ms budget" the re-eval must run on representative hardware (or the result
must be explicitly labelled with the machine it was measured on).

### Behaviour / geometry fields — loadable IF the generation saved them

`p`, `p_des`, `v`, `q`, `contacts`, `track_err` are geometric quantities the simulator
already wrote into the episode pickle (`obs=(T,9)`, `q`, `contact_fraction`, …). For any
generation whose eval persisted full trajectories, these can be **loaded directly** and
the `OBS` / `STATE` / `CONTACT` lines reconstructed without re-running physics.

### Decision / intermediate fields — loadable ONLY if explicitly logged at eval time

`fm_horizon` (the full H-step prediction), `dpcc status=ACTIVE/IDLE`, `constraint_margin`,
and the `fm_cmd → qp_cmd` override delta are **intermediate values inside the control
loop**. Standard eval saves the *executed* action, not the full predicted horizon nor the
QP's internal status. Unless the original run dumped them, they are gone — the only way to
recover them is to re-run inference (which is deterministic for a fixed seed, so the
*values* are reproducible even though the *timing* of that re-run is fresh).

### Per-generation reality

| Generation | Timing | Behaviour (p, contacts) | FM horizon / DPCC status | Verdict |
|---|---|---|---|---|
| **Gen3v4** (iMF, state) | re-eval only | loadable from saved trajs | re-eval (not dumped) | **re-eval** for timing+horizon |
| **Gen5/6** (visual DDPM) | re-eval only | loadable | re-eval | **re-eval** |
| **Gen7** (visual FM) | re-eval only | loadable | re-eval | **re-eval** |
| **Gen9** (visual avoiding) | re-eval only | loadable | re-eval | **re-eval** (currently training — wait for stable ckpt) |
| **Gen11** (drone expert data) | re-eval only | already in `.pkl` (`obs`, `q`, contacts) | re-eval (collection logs PID, not FM/QP) | **re-eval** for FM-PCC timing |
| **DPCC baseline** | re-eval only | loadable | `qp_*` re-eval | **re-eval** |

> Per-gen specifics should be confirmed against each generation's actual output schema
> before relying on the "loadable" column — this table reflects the standard eval format,
> not a guarantee that a given run dumped intermediate fields.

### Bottom line

There is **no shortcut**. Because timing is the headline metric and timing can only be
measured live, every generation we want to audit must be **re-evaluated once** with the
`BehaviorLogger` attached. Loading old outputs can at best fill the geometric context
lines — never the numbers the framework exists to produce. The practical cost is one extra
instrumented eval pass per generation; the logger is designed to wrap that pass with near-
zero added latency (timing is captured around existing calls, not by inserting new work).

---

## Implementation approach

The logger wraps any evaluation loop. At each step, each component passes its outputs
to the logger before the next step begins:

```python
logger = BehaviorLogger(episode_id, system_tag, scene, homotopy)

for t, obs in episode:
    fm_out, fm_ms   = fm.infer(obs)           # FM inference
    qp_out, qp_ms, qp_status = dpcc.solve(fm_out, state)  # QP solve
    u               = pid.compute(qp_out, state)           # PID
    state_next      = sim.step(u)             # physics

    logger.step(
        t       = t,
        obs     = obs,
        fm_horizon = fm_out,       # full H-step horizon
        fm_ms   = fm_ms,
        qp_cmd  = qp_out,
        qp_ms   = qp_ms,
        qp_active = qp_status.active,
        nearest_constraint = qp_status.nearest,
        constraint_margin  = qp_status.margin,
        pid_thrust = u,
        state   = state_next,
        contact = sim.contact_info(),
    )

logger.save(f'logs/eval/{episode_id}.log')
```

The logger is **system-agnostic** — for DPCC baseline, `fm_horizon=None` and `qp_*` fields
are always populated. For FMv3Ode, `fm_horizon` is populated. For Visual Aligning, add a
`visual_anchor` field. All share the same grammar.

---

## Priority

| System | When to implement |
|---|---|
| DPCC baseline | First — simplest (no FM fields), validates the logger format |
| FMv3Ode | After baseline — adds `FM` block and `fm_horizon_rmse` to summary |
| Visual Aligning | After FMv3Ode — adds visual anchor field |
| Drone FM-PCC | When Drone FM-PCC evaluation begins |

**GIF + text log should be the standard output of every evaluation run.** The GIF tells
you if it looks right. The text log tells you if it actually is.
