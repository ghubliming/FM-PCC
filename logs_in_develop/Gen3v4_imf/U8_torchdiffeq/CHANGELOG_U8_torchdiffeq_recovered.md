# U8 — torchdiffeq solver recovered for `iMeanFlowODE`

**Date:** 2026-07-03
**Scope:** `flow_matcher_v3_imeanflow/models/imf_diffusion.py`
**Plan:** `U8_torchdiffeq/PLAN_recover_torchdiffeq_solver.md` (see its §0 for two corrections
made during implementation, summarized below)

## Bug (recap)

`iMeanFlowODE.p_sample_loop` accepted `ode_solver_backend_v3`/`ode_solver_method_v3` (same
config contract as FMv3ODE, even stamped into folder names as `Meuler`/`Mrk4`/…) but never
read either value — it was unconditionally plain Euler. Confirmed live on disk: experiment
folders named `..._Mrk4_..._iMeanFlowODE` had actually run Euler throughout.

## Fix

Ported the torchdiffeq dispatch from the sibling `diffusion.py`'s `FlowMatchingIMF` (which
already had it working) into `iMeanFlowODE.p_sample_loop`:

- Import guard at module top: `from torchdiffeq import odeint as torchdiffeq_odeint` (falls
  back to `None` if not installed, same as the sibling file).
- `p_sample_loop` now branches on `self.ode_solver_backend_v3`:
  - `'legacy_euler'` (default) — **byte-identical to the old behavior**, just moved into an
    `else`. No existing `Meuler` result changes.
  - `'torchdiffeq'` — each macro-step `[t_i, t_i+dt]` is handed to `torchdiffeq.odeint` with
    `method=self.ode_solver_method_v3` (`rk4`, `dopri5`, etc.), `rtol`/`atol`/`step_size` from
    config, matching FMv3ODE's contract exactly.
- `h` is computed **dynamically per sub-stage**, as `h_sub = t1 - t_scalar` (remaining distance
  to the macro step's own declared end) — **not** held fixed at the outer `dt`. This is a
  correction made mid-implementation (see below), not the original design.

**Math grounding for `h_sub`, verified against the actual training code** (not assumed): the
JVP training objective (`_p_losses_meanflow_jvp`, same file) calls
`_jvp(_u_of, (x_r, r, h), ...)` — the primal inputs to `_u_of(z_in, t_in, h_in)` are `(x_r, r, h)`,
i.e. `t_in=r` (the current/anchor time) and `h_in=h=t-r` (forward distance to the later target
time `t`), per the docstring's own identity `(t−r)·u(z_r,r,t) = z_t−z_r` [u = average velocity
over `[r,t]`]. `h_sub = t1 - t_scalar` puts a torchdiffeq sub-stage's own time `t_scalar` in the
`r` role and the macro step's declared end `t1` in the `t` role — same structure the model was
actually trained on, not an invented patch.

**Scope note**: an intermediate version of this fix also added NFE (real network-call count)
tracking, surfaced via new `infos['nfe']`/`infos['flow_steps']` keys. That was unrequested scope
creep beyond "recover the torchdiffeq solver" and has been reverted — `p_sample_loop`'s `infos`
dict has exactly the same keys it had before this change (`diffusion`, `projection_costs`).

## Two corrections made during implementation (both caught by review, not self-caught)

**No step-count gate.** The first version hard-coded `flow_steps_v3 < 10 → raise ValueError`,
justified by "below 10, `h` isn't small enough." That reasoning was never rigorous and was
removed entirely — not softened, removed. `torchdiffeq` now works at any `flow_steps_v3`,
including `K1`/`K2`.

**Why removing it was correct, not just permissive**: the *actual* reason a gate had seemed
necessary was a real bug in the first draft — holding `h` fixed at the outer `dt` for every
RK/adaptive sub-stage call. Classical `rk4` evaluates the RHS at interior points within each
macro step (e.g. `t_i`, `t_i+dt/2`, `t_i+dt/2`, `t_i+dt`); asking `u(x, t_i+dt/2, h=dt)` implies
"average velocity over `[t_i+dt/2, t_i+1.5dt]`" — which overshoots the macro step's own boundary
by `dt/2`, and for the step's own last sub-stage overshoots by a full `dt`, landing past `t=1`
on the final macro step **regardless of how large `flow_steps_v3` is**. The `>=10` threshold was
an attempt to make this overshoot "small in practice" rather than fixing its cause.

The actual fix (now implemented) computes `h` dynamically per sub-stage instead: every
sub-stage's implied interval always ends exactly at the macro boundary, for any solver, any
`flow_steps_v3` — the overshoot is eliminated at its source, not shrunk by restricting `K`.

What's left *is* a genuinely open, purely empirical question with no derivable numeric cutoff:
whether combining `u`-evaluations (which only approximate the instantaneous derivative RK4's
accuracy proof assumes, in the `h→0` limit) via the classical RK4 formula gives a *good*
estimate at small `flow_steps_v3` depends on how much the model's true velocity field curves
over an interval of that size — model- and task-dependent. No threshold is hard-coded for it;
validate empirically (plan §6) at whatever `flow_steps_v3` is actually used.

## What didn't change

- Training (`_p_losses_meanflow_jvp` and friends) — untouched, this is sampling-only.
- Any `legacy_euler` result, at any `flow_steps_v3`.

## Outstanding

- Prior `Mrk4`/`Mmidpoint`/etc. batches (any `flow_steps_v3`) were silently Euler and should be
  re-run to get real (not fabricated) solver-comparison data.
- No local Python runtime here — validate on cluster per the plan's §6 sequence (Euler-via-
  torchdiffeq vs `legacy_euler` regression check first; compare `rk4` vs `euler` at matched
  real network-call count, not matched `flow_steps_v3`; test `K1`/`K2` too, now that it's
  unblocked).
