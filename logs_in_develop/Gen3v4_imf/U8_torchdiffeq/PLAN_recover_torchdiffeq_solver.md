# U8 — Recover the torchdiffeq solver: Gen3v4's `rk4` config is fake

**Date:** 2026-07-03
**Status:** confirmed bug, **re-scoped** — see §0 correction, plan (not yet implemented)
**Scope:** `flow_matcher_v3_imeanflow/models/imf_diffusion.py` (`iMeanFlowODE.p_sample_loop`)

## 0. Two corrections (both superseded an earlier draft of this plan)

**Correction A**: an earlier draft framed the fix as a blanket "Euler → torchdiffeq" swap.
Wrong — the official iMF repo (`/workspaces/imeanflow/imf.py`) has **zero** usage of
torchdiffeq/`odeint`/rk4/any ODE-solver library anywhere. Sampling is `num_steps` applications
of one algebraic line, `z_t - (t-r)*u(...)`, in a loop (`imf.py:42-48`, `sample_one_step` at
`imf.py:90-114`). No integrator is needed there, because the MeanFlow identity makes
`u(z,t,h)` integrate the *true* marginal velocity **exactly** over `[t,t+h]` — that's the
entire point of the paper's speed-up (1-2 network calls instead of 20-100).

But `flow_matcher_v3_imeanflow/models/imf_diffusion.py`'s `p_sample_loop` is **not** that —
it's a fixed-grid loop that queries `u(x, t_i, h=dt)` with `dt=1/flow_steps_v3` at every step
and does `x = x + velocity*dt` (plain Euler), **regardless of `flow_steps_v3`**. When
`flow_steps_v3` is large (`K≥10`, confirmed to exist on disk: `K10`,`K20`,`K40`,`K100`), `dt`
is small, so this genuinely is "treat `u` as a stand-in for `v`, integrate the boring
many-small-step way" — a real, coherent regime where a better integrator (rk4/dopri5) is a
sensible question, and where the solver-config knob silently doing nothing is a real bug.

**Correction B (this one caught a real implementation bug, not just framing)**: the first
implementation of the torchdiffeq branch held `h` **fixed at the outer macro-step's `dt`** for
every internal RK/adaptive sub-stage call, reasoning that this was only sound once `flow_steps`
was "large enough" (an arbitrary `>=10` cutoff was hard-coded). That threshold was never
principled — it was presented as a derived cutoff when it wasn't one, and it was masking a
real, fixable bug rather than a fundamental limitation: holding `h=dt` fixed means a sub-stage
evaluated at, say, `t_i+dt/2` gets asked `u(x, t_i+dt/2, h=dt)`, i.e. "average velocity over
`[t_i+dt/2, t_i+1.5dt]`" — which **overshoots this macro step's own boundary** by `dt/2` (and
by a full `dt` for the last sub-stage of the *last* macro step, landing past `t=1` regardless
of how large `flow_steps_v3` is). **The actual fix**: compute `h` dynamically, per sub-stage,
as `h_sub = t1 - t_scalar` (`t1` = this macro step's declared end). Every sub-stage's implied
query interval `[t_scalar, t1]` then ends exactly at the macro boundary, by construction, for
*any* solver and *any* `flow_steps_v3` — including `flow_steps_v3=1`. This eliminates the
overshoot entirely; there is no longer a domain-validity reason to gate on step count.

What's left is a genuinely different, purely empirical question with no derivable numeric
cutoff: RK4's accuracy guarantee assumes its stage evaluations approximate the *instantaneous*
derivative; `u(x,t,h)` only equals that in the `h→0` limit, so whether combining `u`-evaluations
at different sub-stage `h` values via the classical RK4 formula is actually a *good* estimate
at small `flow_steps_v3` depends on how much the true velocity field curves over an interval of
that size — model- and task-dependent, not something to hard-code a threshold for. **No step
count is gated in the implementation.** Validate per §6 before trusting any result, at any `K`.

## 1. Confirmed: yes, the config is dead code

`config/avoiding-d3il.py:456,840` sets `'diffusion': 'flow_matcher_v3_imeanflow.models.iMeanFlowODE'`
for every imeanflow experiment. `models/__init__.py:7` resolves that name to `imf_diffusion.py`'s
`iMeanFlowODE` class — **not** the sibling `diffusion.py`'s `FlowMatchingIMF`, which is the class
that actually has a working torchdiffeq solver.

`iMeanFlowODE.__init__` (`imf_diffusion.py:66-99`) accepts and stores the exact same solver
config fields FMv3ODE uses — `ode_solver_backend_v3`, `ode_solver_method_v3`,
`ode_solver_rtol_v3`, `ode_solver_atol_v3`, `ode_solver_step_size_v3` — and the experiment-folder
naming tag `('ode_solver_method_v3', 'M')` (`config/avoiding-d3il.py:61`) stamps the configured
method (`Meuler`, `Mrk4`, …) directly into the output path. **But `iMeanFlowODE.p_sample_loop`
(`imf_diffusion.py:196-284`) never reads either attribute.** The sampling loop is:

```python
# imf_diffusion.py:253 — always this, regardless of ode_solver_backend_v3/method_v3
x = x + velocity * dt
```

a bare Python `for` loop doing plain forward Euler, unconditionally.

**Real-world proof this already happened**: experiment folders exist on disk literally named
`H8_K10_Mrk4_T0.5_Dflow_matcher_v3_imeanflow.models.iMeanFlowODE` and
`H8_K20_Mrk4_T0.5_Dflow_matcher_v3_imeanflow.models.iMeanFlowODE` (seen in
`temp/DA_debug/slurm`, candidates 25/28 of the combined batch). Every run behind those folder
names executed plain Euler — the `Mrk4` in the path is fabricated; it has zero relationship to
what actually ran.

## 2. Why this regressed relative to FMv3ODE

`flow_matcher_v3_imeanflow/models/diffusion.py`'s `FlowMatchingIMF.p_sample_loop`
(`diffusion.py:179-268`) **does** have the real dispatch — it's a near-verbatim copy of
`flow_matcher_v3_ode_selectable/models/diffusion.py`'s solver contract, complete and working:

```python
# diffusion.py:212-247 (FlowMatchingIMF — the OTHER, unused-by-config class)
if use_torchdiffeq:
    def ode_rhs(t_scalar, state):
        t_batch = torch.ones(batch_size, ...) * t_scalar
        return self._predict_velocity(state, cond, t_batch, returns=returns)
    x = torchdiffeq_odeint(ode_rhs, x, t_span, method=self.ode_solver_method_v3, ...)[-1]
```

So the torchdiffeq wiring wasn't deleted from the codebase — it exists, tested, in a sibling
file. It's just in the **wrong class**. When `imf_diffusion.py`'s `iMeanFlowODE` was written
(the real mean-flow / JVP-trained model, U4-onward), its constructor signature was copied from
`FlowMatchingIMF`/FMv3ODE (hence it still *accepts* the solver config), but `p_sample_loop`'s
body was written fresh for the mean-flow sampling procedure and the torchdiffeq branch was
never carried over.

## 3. The catch: this isn't a copy-paste — `iMeanFlowODE` needs an extra argument torchdiffeq doesn't have

`FlowMatchingIMF._predict_velocity(x, cond, t, returns=None)` — plain instantaneous field,
4 arguments, drops straight into `torchdiffeq`'s `f(t, x)` RHS contract.

`iMeanFlowODE._predict_velocity(x, cond, t, h=None, returns=None, omega=None, t_min=None,
t_max=None, cfg_scale=0.0)` (`imf_diffusion.py:160-175`) — **requires `h`**, the mean-flow
interval size the model was trained to condition on (`FIX-3/Deviation-A`, same file, quoting
the comment: *"reference iMF's inference uses ONLY the u (mean-velocity) head"*). `h` is not
a free/optional parameter here — it changes what the network computes, not just how it's
integrated.

`torchdiffeq.odeint` calls its RHS at whatever internal stage times the chosen method needs
(classical `rk4` evaluates 4 sub-stages per macro-step at different offsets from `t`; adaptive
methods like `dopri5` pick their own micro-steps entirely). **There is no principled value of
`h` for those sub-stage evaluations** — the model was trained on `(t, h)` pairs meaning "jump
of size `h` starting at `t`," not "one of several RK-tableau stage offsets inside a bigger step
whose size the model never sees."

**The correct choice (implemented)**: compute `h` **dynamically per sub-stage**, as the
remaining distance to *this macro step's own declared end* `t1`: `h_sub = t1 - t_scalar`. Every
sub-stage's implied query interval `[t_scalar, t1]` then ends exactly at the macro boundary —
domain-valid (`t_scalar + h_sub = t1 ≤ 1`) by construction, for any solver, any `flow_steps_v3`.
At the first sub-stage (`t_scalar=t_i`), `h_sub=dt`, matching the Euler path. At the last
sub-stage of a step (`t_scalar=t1`), `h_sub=0`, the valid base case `u(x,t,h=0)=v(x,t)` — not
an out-of-domain query. No approximation is needed for the *domain-validity* question anymore;
what's left (is the *estimate* good at small `flow_steps_v3`) is answered empirically, not by a
threshold (§0 Correction B).

## 4. The fix (implemented)

Ported the dispatch structure from `diffusion.py:190-247` into `imf_diffusion.py`'s
`p_sample_loop`, replacing the unconditional `x = x + velocity * dt` with a branch:

```python
use_torchdiffeq = self.ode_solver_backend_v3 == 'torchdiffeq'
if use_torchdiffeq and torchdiffeq_odeint is None:
    raise RuntimeError("ode_solver_backend_v3='torchdiffeq' but torchdiffeq is not installed.")
...
for i in range(total_steps):
    ...  # unchanged: loop_idx, tau, t_i, step_cfg as today
    if use_torchdiffeq:
        t0, t1 = float(loop_idx) * dt, float(loop_idx) * dt + dt
        t_span = torch.tensor([t0, t1], device=device, dtype=torch.float32)

        def ode_rhs(t_scalar, state):
            ones = torch.ones(batch_size, device=device, dtype=torch.float32)
            t_batch = ones * t_scalar
            h_sub = ones * (t1 - t_scalar)     # dynamic — see §3, not held fixed at dt
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
    ... # projector/gradient-guidance block — unchanged, applies after either branch
```

Import guard at the top of `imf_diffusion.py` (mirror `diffusion.py:6-9`):
```python
try:
    from torchdiffeq import odeint as torchdiffeq_odeint
except ImportError:
    torchdiffeq_odeint = None
```

## 5. What this does and doesn't change

- Purely a sampling-time (`p_sample_loop`) change. Training (`_p_losses_meanflow_jvp`) is
  completely untouched — this doesn't affect any checkpoint, only how already-trained models
  are sampled.
- `ode_solver_backend_v3='legacy_euler'` (the default) behaves **exactly as it does today** —
  the Euler branch is preserved verbatim, just moved into an `else`.
- `ode_solver_backend_v3='torchdiffeq'` now activates at **any** `flow_steps_v3`, including
  `K1`/`K2` — there is no gate. Whether it's a *good idea* to use it at very low `K` is exactly
  the empirical question in §0 Correction B; nothing in the code prevents trying it.
- Every past `Mrk4`/`Mmidpoint`/etc. result under `iMeanFlowODE` was silently Euler — those
  folders are mislabeled and should be treated as Euler-only data, not re-derived as their
  claimed method, until re-run under this fix.

## 6. Validation plan (no local Python runtime — cluster-only)

1. **Regression check first**: run with `ode_solver_backend_v3='torchdiffeq'`,
   `ode_solver_method_v3='euler'` and diff the resulting trajectories against the existing
   `legacy_euler` path on the same seed/checkpoint — should match closely (same step count,
   same update rule, only the call path differs). If they diverge, the port has a bug before
   even reaching `rk4`.
2. Then compare `rk4` vs `euler` at matched `flow_steps_v3` — remember `flow_steps_v3` is
   **not** the real cost once a multi-stage method is picked (rk4 does ≈4× the network calls
   of euler per macro-step; adaptive methods vary). Compare at matched **real network-call
   count**, not matched `flow_steps_v3`, for a fair quality-vs-cost read — count calls manually
   if needed (e.g. a temporary counter on `_predict_velocity`), this plan does not add
   permanent instrumentation for it.
3. Test at low `flow_steps_v3` (`K1`/`K2`) too, now that nothing blocks it — per §0 Correction B
   this is an open empirical question, not a known-good or known-bad regime. Compare against
   plain `legacy_euler` at the same `K` and against FMv3ODE at matched network-call cost.
4. Re-run (or explicitly re-label as Euler) any prior `Mrk4`/`Mmidpoint`/etc. batches before
   citing them as solver-comparison results — they are not what their folder names claim.

## 7. Files touched

- `flow_matcher_v3_imeanflow/models/imf_diffusion.py` — added torchdiffeq import guard +
  dispatch branch in `p_sample_loop` (no step-count gate, §0 Correction B). No other files need
  to change; the config contract already exists end-to-end (`config/avoiding-d3il.py`,
  `args_to_watch` naming, CLI overrides).

## 8. The separate, actually-interesting question this doesn't answer

Does `iMeanFlowODE` at `K1`/`K2` actually beat FMv3ODE at matched low NFE — i.e., does this
codebase's own checkpoint reproduce the paper's headline claim? That's answerable **today**,
with zero code changes, by comparing existing (or newly-run) `K1`/`K2` `iMeanFlowODE` results
against FMv3ODE results at the same step count on the same avoiding-task metrics. Worth doing
as a follow-up, but it's a training/checkpoint-quality question, not a solver-plumbing one —
out of scope for this plan.
