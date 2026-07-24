# Why HardFlow has no foresight fan and never talks about smoothness — while DPCC/FMPCC obsesses over both

**Date:** 2026-07-19
**Context:** questions raised after the Gen13 first run (`logs_in_develop/Gen13/fix_3/INSIGHTS_Gen13_first_run.md`).
**Companions:** `BLEND_HardFlow_iMeanFlow.md`, `THEORY_DeepMix_HF_iMF.md` (this folder); `Gen3v4_imf/U10/K2_train_eval/ANALYSIS_imf_official_K2_train_curve_and_eval.md` (the smoothness/K-sweep worry).

**The two questions:**
1. Why does iMF-in-HardFlow produce no **MPC foresight fan** like DPCC/FMPCC does?
2. Why is **trajectory smoothness** — the thing that dominated Gen3v4's analysis — absent from the HardFlow paradigm *and* from the HardFlow paper?

**Short answers:** (1) is a pure **instrumentation gap** — HardFlow *computes* the fan and then throws it away. (2) is a genuine and important **paradigm difference** — HardFlow converts smoothness from an *emergent property you measure* into a *hard constraint you enforce*, which is exactly why it disappears from the discussion. And this directly explains the Gen13 result.

---

## Part 1 — The foresight fan: computed, then discarded

### What DPCC/FMPCC does
FMPCC explicitly renders the fan. From `scripts/eval.py:222` (and the same block in `FM_v3_test/`, `FM_v3_ode_selectable_test/`, `FM_v3_drifting_test/`):
```python
plot_samples_every = max(1, args.horizon // 2)  # keep PLOT readable — draw foresight fan only every H/2 steps
...
for __ in range(0, len(sampled_trajectories_all[i]), plot_samples_every):   # over timesteps
    for ___ in range(min(args.batch_size, 4)):                              # over batch
        curr_ax.plot(sampled_trajectories_all[i][__][___, :args.horizon, x], ..., 'b')
```
So the fan = **the planned H-step trajectories, for several batch samples, overlaid at multiple replan instants**. It visualizes *what the policy intended*, not only what it did.

### What HardFlow does
The policy returns the same objects — `hardflow_new_forward` returns `(action, trajectories, x_chain, x1_estimation, info)`, where `x_chain` is the full ODE/NLP chain and `x1_estimation` is the per-step terminal prediction. **But the eval loop discards them** (`run/eval.py:393`):
```python
action, samples, _, _, info = policy(conditions, batch_size=cfg.batch_size)
#                     ↑  ↑  x_chain and x1_estimation thrown away
```
And the renderer has no fan capability: `AvoidingTrajectoryPlotter.plot_single_trajectory` supports only `style="actual"`, and the sole call site is
`save_single_trajectory_image(real_trajectory, ...)` → `{run_id}_real.png` — the **executed rollout only**.

### Verdict
**Nothing conceptual is missing.** HardFlow is a receding-horizon MPC exactly like DPCC (H=16, `replan_steps=8`, `controller="rh"`); it plans a full horizon at every replan and could plot the fan tomorrow. The foresight objects are already returned by both `FlowPolicy` and our `ImfFlowPolicy` — including the iMF-exact `x1_estimation` (which for iMF is arguably *more* interesting than FM's, since it's the exact endpoint map rather than an Euler extrapolation).

**This is a cheap, high-value addition for Gen13** — see Part 5.

---

## Part 2 — Smoothness: three different things that keep getting conflated

Much of the confusion dissolves once these are separated:

| # | Notion | Question it answers | Who enforces it |
|---|---|---|---|
| **S1** | **Generative smoothness** | Is a *single sampled* trajectory smooth/non-jittery in state space? | the generative model alone (DPCC/FMPCC) |
| **S2** | **Dynamic feasibility** | Does the plan obey `s_{t+1} = A·s_t + B·a_t + c`? | **HardFlow's NLP, as a hard equality** |
| **S3** | **Cross-replan consistency** | Does the plan at time *t* agree with the plan at *t+8*? | *nobody, by default* (see Part 4) |

Gen3v4 worried almost entirely about **S1**. HardFlow enforces **S2** and thereby *largely obtains* S1 as a by-product. Neither addresses **S3**.

---

## Part 3 — Why HardFlow never discusses smoothness: it made it a constraint, not a metric

This is the core answer. In `_apply_dynamics_constraints` (`flow_policy.py:350+`), for **every** consecutive pair across the horizon (`for i in range(self.horizon - 2)`, plus an `s0→s1` anchor), the NLP imposes a **hard equality**:

```python
opti.subject_to(
    cs.mtimes(A, state) + cs.mtimes(B, action) + c == next_state
)
```

with `A, B, c` the linear dynamics fitted by `run/fit_dynamics.py` (our fit: **R² = 0.9998**, verified against the real env at R² = 0.9989).

**Consequences, and they are large:**

1. **A jittery generative sample cannot survive.** The prox-NLP projects the predicted terminal trajectory onto the intersection of {obstacle-free} ∩ {dynamically consistent} ∩ {action bounds}. A trajectory that zig-zags between timesteps is *dynamically infeasible* and is therefore projected out. Smoothness is not hoped for — it is **structurally imposed**.
2. **So smoothness stops being a metric.** You cannot report "our trajectories are smoother" when smoothness is a feasibility condition every method must satisfy identically. HardFlow's reported quantities are therefore **binary and constraint-shaped**: safety/violation rate, success rate, compute. That is why the paper is silent on smoothness — *not* an oversight, but a consequence of where it put the guarantee.
3. **The generative model's job shrinks.** In DPCC/FMPCC the model must produce a trajectory that is simultaneously (a) task-solving, (b) smooth, (c) obstacle-avoiding. In HardFlow the model only needs to propose something in the right *basin*; the optimizer supplies (b) and (c) with hard guarantees.

### The DPCC/FMPCC contrast
DPCC has a projection step too, but the pipeline's identity is "generative brain + physical brakes" where trajectory quality is *diagnosed* at the generative stage — hence Gen3v4's K-sweep, the smoothness eyeballing, and the "raw `diffuser` variant looks chaotic" analysis. In that framing S1 is a **model-quality readout**. In HardFlow's framing S1 is **not observable as a metric at all**, because the brakes are always on and always hard.

---

## Part 4 — The mechanism that *is* missing from both: S3 (cross-replan consistency)

A real MPC concern the fan would expose: at t=0 the policy plans to go left; at t=8 it replans and goes right. Each plan is individually smooth and feasible; the **executed** trajectory still kinks at the replan seam. Neither S1 nor S2 catches this.

**Notable finding:** HardFlow *has* built this mechanism — and leaves it switched off.

- `ProxyValueModel._compute_consistency_objective` (`run/eval.py:127`) penalizes deviation from the previous plan:
  `-((x - previous_sample)**2).sum()`
- `previous_sample` **is** updated after every planning call — in every guidance method, including our `ImfFlowPolicy` (`imf_flow_policy.py:398`).
- It is selectable via `value_objective="consistency"`.
- **But every avoiding run script uses `value_objective="distance"`** — verified across all 7 FM scripts and our iMF script. `consistency` is never exercised.

So the plumbing for replan coherence is complete, wired, and dormant. Whether it matters is an open, testable question — and it is *more* likely to matter for iMF than for FM, because a coarser field plausibly yields plans that vary more between replans.

---

## Part 5 — Why this explains the Gen13 result

The Gen13 numbers become much less mysterious in this light:

| Run | Success | Reading |
|---|---|---|
| iMF unguided K1/K2 | 0% / 2% | the **raw field is coarse** — `raw_mse_u` ≈ 13 over 96 dims ≈ **0.37/dim** (Gen3v4's easier H8 task reached ≈0.25/dim). Bad S1. |
| iMF + HardFlow K2 | **94%** | the NLP supplies S2 (and obstacle feasibility) **regardless of S1**, rescuing a field that alone solves ~0% of episodes |

**The projection is doing enormous work** — which is precisely the paradigm claim of Part 3, and it is why "the iMF field is rough" (Gen3v4's central worry) turned out *not* to be fatal here. It also reframes the residual gap: E3's 3 violations are far more likely a **projection-count** deficit (K=2 → only 2 projections vs FM's 10; K1→K2 already moved safety 80%→94%) than a smoothness deficit.

**Corollary — a Gen3v4 instinct that does not transfer:** judging iMF by eyeballing raw trajectory smoothness is the *wrong lens inside HardFlow*. The right lens is violation rate at a given NFE/projection budget. A field that looks ugly unguided can be excellent as a HardFlow prior.

---

## Part 6 — Concrete, cheap experiments this suggests

1. **Add the foresight-fan plot to the iMF eval** (Part 1). `x_chain` / `x1_estimation` are already returned and discarded; capturing them costs one line in `_run_env_quiet` plus a plotting helper. It would let us *see* whether iMF's 1-NFE endpoint prediction is qualitatively different from FM's Euler shot — the visual counterpart of the seam swap, and currently our only un-inspected diagnostic. **Additive, Gen13-owned files only.**
2. **Turn on `value_objective="consistency"` for one iMF run** (Part 4). Tests S3, uses machinery that already exists, needs no retraining, and is a plausible differentiator between a coarse-but-fast field and a fine-but-slow one.
3. **Quantify S1 directly instead of eyeballing** — e.g. mean squared second difference of the planned trajectory, before vs after projection. This would *measure* how much smoothness the NLP is manufacturing, turning Part 3's qualitative claim into a number. Cheap to compute from `x_chain` once (1) is in place.
4. (Already the top recommendation from fix_3) **K=4/5 sweep** — the projection-count hypothesis.

---

## Part 7 — What is verified vs. inferred

**Verified in code (this session):** the discarded `_, _` return values; the plotter's single `"actual"` style; FMPCC's explicit fan-plotting block; the hard dynamics equality and its `horizon-2` loop; the fitted-dynamics R² values; the existence of `_compute_consistency_objective` + `previous_sample` updates; and that all avoiding scripts use `value_objective="distance"`.

**Inferred (reasonable, not proven):** that the dynamics equality is *the* dominant smoothing mechanism (it is the most plausible one, but obstacle constraints and the distance objective also shape the projection — experiment 3 above would settle it); that HardFlow's paper omits smoothness *because* it is constrained rather than measured (consistent with the paper's metric set, but the authors' reasoning is not stated); and that iMF's coarser field would benefit more from S3 than FM's.

**Not claimed:** that HardFlow's approach is superior to DPCC's. They place the guarantee in different components — optimizer vs. model — with different costs (HardFlow pays ~41 NFE + 10 NLP solves per plan; DPCC pays model quality). Gen13 is precisely an attempt to lower one of those costs.
