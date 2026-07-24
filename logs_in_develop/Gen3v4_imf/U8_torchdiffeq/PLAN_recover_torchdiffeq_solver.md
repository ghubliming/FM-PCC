# U8 — Recover the torchdiffeq solver for `iMeanFlowODE`

**Date:** 2026-07-03
**Scope:** `flow_matcher_v3_imeanflow/models/imf_diffusion.py` (`iMeanFlowODE.p_sample_loop`)

## The bug

`iMeanFlowODE.__init__` accepts `ode_solver_backend_v3`/`ode_solver_method_v3` (same config
contract as FMv3ODE, even stamped into experiment folder names as `Meuler`/`Mrk4`/…), but
`p_sample_loop` never read either value — it was unconditionally plain forward Euler
(`x = x + velocity * dt`), regardless of what was configured. Confirmed live on disk:
experiment folders named `..._Mrk4_..._iMeanFlowODE` had actually run Euler throughout.

The real torchdiffeq dispatch wasn't deleted from the codebase — it already exists, working,
in a sibling class in the same package: `diffusion.py`'s `FlowMatchingIMF` (a plain, non-mean-flow
flow-matching model). It just never got ported into `iMeanFlowODE`, the class every imeanflow
config actually instantiates.

## What `iMeanFlowODE` needs that a plain copy of `FlowMatchingIMF` doesn't have

`FlowMatchingIMF._predict_velocity(x, cond, t, returns=None)` is a plain instantaneous field —
4 arguments, drops straight into `torchdiffeq`'s `f(t, x)` RHS contract with no adaptation.

`iMeanFlowODE._predict_velocity(x, cond, t, h=None, returns=None, ...)` **requires `h`** — the
mean-flow interval size. This isn't a free/optional knob; it changes what the network computes.
Training conditions on it directly (`_p_losses_meanflow_jvp`, same file): the JVP primal is
`(x_r, r, h)` with `h = t - r`, i.e. **`u(x, t_query, h)` = "average velocity starting at
`t_query`, forward to `t_query + h`"**, trained only for `t_query, h ∈ [0,1]` with
`t_query + h ≤ 1`.

`torchdiffeq.odeint` evaluates its RHS at whatever internal sub-stage times the chosen method
needs (classical `rk4`: 4 sub-stages per macro step, at `t0`, `t0+dt/2`, `t0+dt/2`, `t1`;
adaptive methods like `dopri5` pick their own). A naive port that holds `h` fixed at the outer
macro-step's `dt` for every sub-stage call would ask, e.g., `u(x, t0+dt/2, h=dt)` — "average
velocity over `[t0+dt/2, t0+1.5dt]`" — a `(t,h)` pair with `t+h` past this macro step's own
end `t1`, and on the very last macro step, past `t=1` entirely: outside the domain the model
was ever trained on, regardless of how large `flow_steps_v3` is.

## The fix — "homing missile" `h`: recomputed per sub-stage, not held fixed

Compute `h` **dynamically at every sub-stage**, as the remaining distance to *this macro step's
own declared end* `t1`:

```
h_sub = t1 - t_scalar
```

Every sub-stage's implied query `(t_scalar, h_sub)` then satisfies `t_scalar + h_sub = t1`
**exactly**, by construction — always in-domain, for any solver, any `flow_steps_v3`, including
`flow_steps_v3=1`. At the first sub-stage (`t_scalar=t0`), `h_sub=dt`, matching what Euler would
use. At the step's final sub-stage (`t_scalar=t1`), `h_sub=0` — the valid base case
`u(x,t,h=0)=v(x,t)`, not an out-of-domain query. No approximation or step-count floor is needed
for domain validity — it's satisfied unconditionally by this construction.

**This has nothing to do with the constraint projector.** `p_sample_loop`'s projector block
(obstacle/contact snapping) runs once per **outer** macro step, strictly after the whole
`torchdiffeq.odeint` call returns — it never sees, and is not affected by, anything happening at
the sub-stage level. `h_sub` exists purely to keep the *network's own* queries inside its
trained `(t,h)` domain; it would be equally necessary in a hypothetical scene with zero
constraints and no projector at all.

## What's still an open, genuinely empirical question (not gated in code)

`torchdiffeq`'s accuracy guarantees (e.g. RK4's 4th-order error bound) are derived assuming the
RHS function returns the true *instantaneous* derivative at each sub-stage. `u(x,t,h)` only
equals that derivative in the `h→0` limit. Whether combining `u`-evaluations at different
sub-stage `h` values via the classical RK4 (or any) formula gives a *good* estimate at small
`flow_steps_v3` (where `h` per sub-stage isn't especially close to `0`) depends on how much the
model's true velocity field curves over an interval of that size — model- and task-dependent,
with no derivable numeric cutoff. **No step-count floor is hard-coded.** Validate empirically
(§ below) at whatever `flow_steps_v3` is actually used, including low values.

## Implementation

`p_sample_loop` branches on `self.ode_solver_backend_v3`:

```python
use_torchdiffeq = self.ode_solver_backend_v3 == 'torchdiffeq'
if use_torchdiffeq and torchdiffeq_odeint is None:
    raise RuntimeError("ode_solver_backend_v3='torchdiffeq' but torchdiffeq is not installed.")
...
for i in range(total_steps):
    ...  # unchanged: loop_idx, tau, t_i, step_cfg
    if use_torchdiffeq:
        t0, t1 = float(loop_idx) * dt, float(loop_idx) * dt + dt
        t_span = torch.tensor([t0, t1], device=device, dtype=torch.float32)

        def ode_rhs(t_scalar, state):
            ones = torch.ones(batch_size, device=device, dtype=torch.float32)
            t_batch = ones * t_scalar
            h_sub = ones * (t1 - t_scalar)          # dynamic — the homing-missile computation
            return self._predict_velocity(
                state, cond, t_batch, h=h_sub, returns=returns,
                omega=omega_b, t_min=t_min_b, t_max=t_max_b, cfg_scale=step_cfg,
            )
        odeint_kwargs = {'method': self.ode_solver_method_v3}
        if self.ode_solver_rtol_v3 is not None: odeint_kwargs['rtol'] = float(self.ode_solver_rtol_v3)
        if self.ode_solver_atol_v3 is not None: odeint_kwargs['atol'] = float(self.ode_solver_atol_v3)
        if self.ode_solver_step_size_v3 is not None:
            odeint_kwargs['options'] = {'step_size': float(self.ode_solver_step_size_v3)}  # fixed-step methods only
        x = torchdiffeq_odeint(ode_rhs, x, t_span, **odeint_kwargs)[-1]
    else:
        velocity = self._predict_velocity(x, cond, t_i, h=h_batch, returns=returns,
                                           omega=omega_b, t_min=t_min_b, t_max=t_max_b, cfg_scale=step_cfg)
        x = x + velocity * dt
    x = apply_conditioning(x, cond, self.action_dim, goal_dim=self.goal_dim)
    ... # projector/gradient-guidance block — unchanged, runs after either branch
```

Import guard at the top of `imf_diffusion.py` (mirrors `diffusion.py`):
```python
try:
    from torchdiffeq import odeint as torchdiffeq_odeint
except ImportError:
    torchdiffeq_odeint = None
```

## What this does and doesn't change

- Purely a sampling-time (`p_sample_loop`) change. Training (`_p_losses_meanflow_jvp`) is
  completely untouched — this doesn't affect any checkpoint, only how already-trained models
  are sampled.
- `ode_solver_backend_v3='legacy_euler'` (the default) behaves **exactly as before** — that
  branch is byte-identical to the prior unconditional code, just moved into an `else`.
- `ode_solver_backend_v3='torchdiffeq'` now works at any `flow_steps_v3`, including `K1`/`K2` —
  no gate. Whether that's a *good idea* at very low `K` is the open empirical question above.
- Every past `Mrk4`/`Mmidpoint`/etc. result under `iMeanFlowODE` was silently Euler — those
  folders are mislabeled and should be treated as Euler-only data, not re-derived as their
  claimed method, until re-run under this fix.

## Validation plan (no local Python runtime — cluster-only)

1. **Regression check first**: `ode_solver_backend_v3='torchdiffeq'`, `ode_solver_method_v3='euler'`
   vs the `legacy_euler` path, same seed/checkpoint — should match closely (same update rule,
   only the call path differs). If they diverge, the port has a bug before even reaching `rk4`.
2. Compare `rk4` vs `euler` at matched **real network-call count**, not matched `flow_steps_v3`
   — `rk4` makes ~4 calls per macro-step, `euler` makes 1, so `flow_steps_v3` alone isn't the
   real cost once a multi-stage method is picked. Count calls manually if needed for a fair
   quality-vs-cost comparison (e.g. a temporary counter on `_predict_velocity`).
3. Test at low `flow_steps_v3` (`K1`/`K2`) too, now that nothing blocks it — this is a real,
   open question, not a known-good or known-bad regime.
4. Re-run (or explicitly re-label as Euler) any prior `Mrk4`/`Mmidpoint`/etc. batches before
   citing them as solver-comparison results — they are not what their folder names claim.

## Files touched

- `flow_matcher_v3_imeanflow/models/imf_diffusion.py` — torchdiffeq import guard + dispatch
  branch in `p_sample_loop`. No other files need to change; the config contract already exists
  end-to-end (`config/avoiding-d3il.py`, `args_to_watch` naming, CLI overrides).

## The separate, actually-interesting question this doesn't answer

Does `iMeanFlowODE` at `K1`/`K2` actually beat FMv3ODE at matched low network-call cost — i.e.,
does this codebase's own checkpoint reproduce the paper's headline "few-step beats many-step"
claim? Answerable **today**, with zero further code changes, by comparing existing (or
newly-run) `K1`/`K2` `iMeanFlowODE` results against FMv3ODE results at matched cost on the same
avoiding-task metrics. A training/checkpoint-quality question, not a solver-plumbing one — out
of scope here.
