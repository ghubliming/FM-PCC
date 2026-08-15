# CHANGELOG — Gen15 Fix_1: the `projection_ms` infos contract

**Date:** 2026-08-15 · **Type:** bug fix (sampler-only) + new gate
**Found in:** eval job **24583** (`mf`, corridor, seed 6, K=10) — every variant printed `proj_ms=0.0`
**Files:** `mix_uav/models/mf_diffusion.py`, `mix_uav/models/af_diffusion.py`,
`mix_uav_test/gates_mix_uav.py`
**Retraining required:** none. **Existing checkpoints unaffected.** This touches the sampler's
telemetry only — no weights, no objective, no trajectory.

---

## 1. The bug

`FlowMatchingODE.p_sample_loop` ends with

```python
infos['projection_ms'] = proj_ms   # real-time logging: CPU projection wall-time this sample
```

`MeanFlowODE` and `AlphaFlowODE` did not. They were authored in Gen3v6/Gen3v7, where
`sampling/policies.py` has no real-time logging and therefore nothing reads the field. Gen15 put
those engines behind the **UAV** `policies.py`, which does:

```python
self.last_proj_ms = float(infos.get('projection_ms', 0.0))
```

A `.get(..., 0.0)` across an engine boundary. The two-time arms silently reported **zero**.

Gen15's init pass audited the `flow_steps_v3` / `num_steps` difference between these same three
samplers (changelog §5) and never checked the rest of the `infos` contract. One key was checked,
the other was assumed.

## 2. Why it was worse than "a missing column"

`proj_ms` is not merely displayed — the eval **derives** the pure-inference number from it
(`eval_mix_uav.py:1012-1017`):

```python
step_total_ms = (time.perf_counter() - t0) * 1e3        # measured directly — always correct
step_proj_ms  = getattr(policy, 'last_proj_ms', 0.0)    # reported by the ENGINE
step_fm_ms    = max(step_total_ms - step_proj_ms, 0.0)  # DERIVED "PURE inference"
```

So with `proj_ms == 0`:

| field | status on `mf`/`af` before Fix_1 |
|---|---|
| `total_ms` | ✅ correct — measured by the eval, never touched the engine |
| `proj_ms` | ❌ hard 0.0 |
| **`fm_ms`** | ❌ **wrong — silently absorbed the entire projector cost** |

From job 24583 (`mf`, K=10, corridor):

| variant | reported `fm_ms` | reported `proj_ms` | `total_ms` | true `fm_ms` ≈ |
|---|---|---|---|---|
| `diffuser` (no projector) | 88.5 | 0.0 | 88.5 | 88.5 ✅ |
| `dpcc-c` | **269.7** | **0.0** | 269.7 | ~88 (⇒ proj ≈ 181) |
| `dpcc-r-tightened` | **498.7** | **0.0** | 498.7 | ~88 (⇒ proj ≈ 411) |

Independent confirmation that ~88 ms is the true generation cost: gate **G6** measures the
sampler in isolation at K=10 → `mf` 87.21 ms, and the unprojected `diffuser` variant measures
88.5 ms in the live loop. The two agree; everything above that is projector.

**The cross-arm consequence is the dangerous part.** `fm` reports `proj_ms` correctly, so its
`fm_ms` is right. Any table putting the arms side by side would have shown MeanFlow's *pure
generation cost* as ~3× the FM arm's — a fabricated result on **the exact axis MeanFlow exists to
win**, pointing the wrong way, and entirely invisible unless you noticed that a column of zeros
is impossible for an SLSQP projector.

## 3. The fix

Three additive edits per engine, mirroring `FlowMatchingODE` verbatim:

1. `import time`
2. `proj_ms = 0.0` beside `costs = {}` before the sampling loop
3. `_t_proj = time.perf_counter()` / `proj_ms += (time.perf_counter() - _t_proj) * 1e3` around
   **both** projector branches (`compute_gradient` and `project`)
4. `infos['projection_ms'] = proj_ms` before `return x, infos`

No control flow, no numerics, no ordering changed — only a clock read either side of calls that
already happened.

⚠️ **Gen3v6 / Gen3v7 are NOT patched.** `flow_matcher_v3_meanflow/` and
`flow_matcher_v3_alphaflow/` have the same gap, but there it is harmless: their own
`policies.py` never reads the field. Gen15 keeps its isolation — the edit lives only in
`mix_uav/models/`. Syncing upstream is a separate decision.

## 4. New gate: G7 — infos contract + projector timing

The root cause is a class of bug, not an incident: **two engines silently disagreeing on a
contract, papered over by a `.get()` default.** G7 asserts the contract instead.

For every arm it runs a sample with a `_StubProjector(sleep_s=0.002)` and checks:

- `infos` contains `projection_costs` **and** `projection_ms`;
- `projection_ms` is **greater than zero** — a projector that burns 2 ms per call must show up,
  which is what a hard-coded 0.0 can never do;
- all three arms return an **identical key set**.

Added to the gate table and to `gates_mix_uav.sh`'s default run. Cost: seconds.

## 5. What this means for job 24583's data

**It is not wasted, and it does not need re-running to stay usable:**

- every **success / safety / steps / track_err** number is untouched — this fix cannot affect a
  trajectory;
- **`total_ms` is correct as printed** — that is the number the 30.3 ms real-time budget is
  judged against, and the `OVER×3960` verdict stands;
- **`fm_ms` and `proj_ms` from that job must be discarded.** As a serviceable reconstruction,
  `proj_ms ≈ total_ms − 88.5` (the measured unprojected cost at K=10 on this scene). Re-run the
  eval if you want the exact split — it is ~6 h and needs no retraining.

## 6. Follow-ups

1. **Re-run gates** (`gates_mix_uav.sh`) before the next eval — G7 should now report a non-zero
   `projection_ms` on all three arms. This is the cheapest possible confirmation the fix works.
2. The `af` arm has never run an eval, so it was never exposed to this. It is fixed pre-emptively.
3. Open question for the DA: with the split restored, the plan's §7.3 mechanism — *does cheaper
   generation free wall-clock for the projector?* — becomes measurable for the first time on the
   two-time arms. At K=10 nothing is real-time anyway (§5), so the mechanism can only be tested
   at K=1/K=2.
