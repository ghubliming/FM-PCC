# U8 — torchdiffeq solver recovered for `iMeanFlowODE`

**Date:** 2026-07-03
**Scope:** `flow_matcher_v3_imeanflow/models/imf_diffusion.py`
**Plan:** `U8_torchdiffeq/PLAN_recover_torchdiffeq_solver.md`

## Bug

`iMeanFlowODE.p_sample_loop` accepted `ode_solver_backend_v3`/`ode_solver_method_v3` (same
config contract as FMv3ODE, stamped into folder names as `Meuler`/`Mrk4`/…) but never read
either value — always plain Euler. Confirmed on disk: folders named `..._Mrk4_..._iMeanFlowODE`
had actually run Euler throughout.

## Fix

Ported the torchdiffeq dispatch from the sibling `diffusion.py`'s `FlowMatchingIMF` (already
working there) into `iMeanFlowODE.p_sample_loop`:

- Import guard: `from torchdiffeq import odeint as torchdiffeq_odeint` (falls back to `None`).
- `p_sample_loop` branches on `self.ode_solver_backend_v3`:
  - `'legacy_euler'` (default) — byte-identical to the old behavior, moved into an `else`.
  - `'torchdiffeq'` — each macro-step `[t0, t1]` handed to `torchdiffeq.odeint` with
    `method=self.ode_solver_method_v3`, `rtol`/`atol`/`step_size` from config.
- **"Homing missile" `h`**: computed dynamically per sub-stage as `h_sub = t1 - t_scalar`
  (remaining distance to the macro step's own declared end) — not held fixed at the outer `dt`.
  This is the piece that required actual derivation (`FlowMatchingIMF`'s `_predict_velocity` has
  no `h` argument at all, so there was nothing to copy for this part): verified against the
  training code's own JVP primal `(x_r, r, h)` in `_p_losses_meanflow_jvp`, where `u(x,t,h)` is
  trained meaning "average velocity from `t` forward to `t+h`" for `t+h≤1`. `h_sub` keeps every
  sub-stage query at exactly `t_scalar + h_sub = t1 ≤ 1`, in-domain by construction, for any
  solver and any `flow_steps_v3` including `1`.
- No step-count gate. Whether RK4-of-`u` is a *good* estimate at low `flow_steps_v3` is a real,
  open, empirical question (how much the true velocity field curves over a macro interval of
  that size) with no derivable numeric cutoff — not something to hard-code a threshold for.

## What `h_sub` is *not* about

It has nothing to do with the constraint/obstacle projector. That block runs once per outer
macro step, strictly after `torchdiffeq.odeint` returns — it never sees or depends on sub-stage
internals. `h_sub` exists solely to keep the network's own `(t,h)` queries inside its trained
domain; it would be equally necessary with zero constraints in the scene.

## What didn't change

- Training (`_p_losses_meanflow_jvp`) — untouched, sampling-only change.
- Any `legacy_euler` result, at any `flow_steps_v3`.

## Outstanding

- Prior `Mrk4`/`Mmidpoint`/etc. batches (any `flow_steps_v3`) were silently Euler and should be
  re-run for real (not fabricated) solver-comparison data.
- No local Python runtime here — validate on cluster per the plan's validation sequence
  (Euler-via-torchdiffeq vs `legacy_euler` regression check first; compare `rk4` vs `euler` at
  matched real network-call count, not matched `flow_steps_v3`; test `K1`/`K2` too).
