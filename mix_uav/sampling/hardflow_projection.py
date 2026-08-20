"""Gen12 — HardFlow's `hardflow_new` constrained sampler, ported onto FMv3.

Port of `HardFlow/hardflow/models_flow/flow_policy.py::hardflow_new_forward`
(+ `hardflow_formulate`) into the FMPCC codebase.

WHY ONLY `hardflow_new`
-----------------------
HardFlow ships four guidance modes.  Three of them (`projection`,
`projection_relaxed`, `hardflow`) wrap the U-Net with l4casadi and compile the
*network itself* into the NLP, hard-coding the architecture at
`HardFlow/run/eval.py:583-589`.  `hardflow_new` instead calls the network as a
black-box `v = f(x, t)` OUTSIDE the solver, so the NLP is purely algebraic and
any velocity field can be dropped in.  That is the only portable mode, and it
is the only one implemented here.  See PLAN §1.1.

WHAT IT DOES (PLAN §1.2), per ODE step k of K:
  1. reference step   v_k = f(x_k, tau_k);  x_ref = x_k + v_k*dt
  2. terminal predict  x1_ref = x_ref + (1 - tau_{k+1}) * f(x_ref, tau_{k+1})
  3. projection        solve a prox-NLP: keep x1 near x1_ref, satisfy constraints
  4. pull-back         x_{k+1} = x_ref + tau_{k+1} * (x1_proj - x1_ref)
The prox weight carries a tau^2 factor, so early steps are nudged and late
steps are pulled hard onto the feasible set.

DIFFERENCES FROM UPSTREAM, ALL DELIBERATE (see the Gen12 changelog §4):
  * The NLP is built from FMPCC's `constraint_list` (the same list the DPCC
    `Projector` consumes) instead of HardFlow's hard-coded
    `hardflow/utils/avoiding_geometry.py`.  Without this, arm B and arm C would
    be enforcing *different* constraint sets and the comparison would be void.
  * No value-model warm start.  FMPCC has no value model; upstream's warmstart
    only picked a noise seed and the s0 parameter, both of which we get for
    free.  This also keeps the NFE accounting clean (2K here, K+2K upstream).
  * The initial noise matches THE HOST MODEL'S OWN SAMPLER, via the explicit
    `init_noise_scale` argument.  See the fix_4 warning below — this is the one
    thing that does NOT transfer when the port is moved to a new generation.
  * IPOPT and CasADi are silenced by default (PLAN §3.5).

⚠️ fix_4 — INITIAL-NOISE SCALE IS GENERATION-SPECIFIC.  DO NOT HARDCODE.
------------------------------------------------------------------------
Gen12's base (`flow_matcher_v3/models/diffusion.py:164,227`) starts its ODE from
`0.5 * randn`.  Gen3v6's MeanFlow (`mf_diffusion.py:204`) starts from
`randn` — sigma=1.0, chosen to match its `q_sample` training noise
(`mf_diffusion.py:180-183`, `noise = torch.randn_like(x_start)`).  The original
U3 port copied Gen12's 0.5 verbatim, so arm C ran at HALF the trained noise
scale: an out-of-distribution tau=0 state, and arms B/C not sharing a start
distribution — the exact thing this sampler is supposed to guarantee.

`init_noise_scale` is therefore a REQUIRED argument on `HardFlowSampler` (no
default — a wrong scale is silent, so the call site must state it).
`HardFlowPolicy` defaults it to Gen3v6's 1.0 and the eval driver passes it
explicitly anyway.
When porting to another generation, read the scale off that generation's own
`p_sample_loop` (Gen3v7 / alpha-Flow: `af_diffusion.py:260` is sigma=1.0 — do NOT
read it from `flow_matcher_v3_alphaflow/models/diffusion.py`, which is the legacy
FMv3ODE class in the same folder and still uses 0.5).
`gates_hardflow_meanflow.py::gate_h3` pins this numerically.
"""

import time

import numpy as np
import torch

import diffuser.utils as utils
from diffuser.utils.arrays import to_device

from ..models.helpers import apply_conditioning
from .policies import Trajectories

try:
    import casadi as cs
except ImportError:  # pragma: no cover - the cluster env has it, this container does not
    cs = None


# ---------------------------------------------------------------------------#
# ------------------------------- DOF layout --------------------------------#
# ---------------------------------------------------------------------------#

class TrajectoryLayout:
    """Index bookkeeping for the flattened, s0-free trajectory vector.

    The full flat trajectory is
        z = [ a_0 s_0 | a_1 s_1 | ... | a_{H-1} s_{H-1} ],   len = H * T
    with T = action_dim + state_dim.  `s_0` is pinned by the MPC conditioning
    (`apply_conditioning`), so it is NOT a decision variable; removing it gives

        dof = [ a_0 | a_1 s_1 | a_2 s_2 | ... ],   len = H*T - state_dim

    This is byte-identical to HardFlow's `oc_dof` layout — upstream indexes the
    i-th *constrained* step as `2*action_dim + i*(action_dim+state_dim)`, which
    is exactly `state_index(i+1)` below.  `gates_hardflow.py::gate_layout`
    asserts this index-by-index rather than trusting the comment (PLAN §3.2).
    """

    def __init__(self, horizon, action_dim, state_dim):
        self.horizon = int(horizon)
        self.action_dim = int(action_dim)
        self.state_dim = int(state_dim)
        self.transition_dim = self.action_dim + self.state_dim
        self.dof = self.horizon * self.transition_dim - self.state_dim

    def action_index(self, t):
        """Start index of a_t inside the dof vector."""
        if t == 0:
            return 0
        return self.action_dim + (t - 1) * self.transition_dim

    def state_index(self, t):
        """Start index of s_t inside the dof vector. Undefined for t == 0."""
        if t == 0:
            raise IndexError('s_0 is not a decision variable (pinned by conditioning)')
        return 2 * self.action_dim + (t - 1) * self.transition_dim

    def to_dof(self, traj_flat):
        """(..., H*T) full flat trajectory -> (..., dof), dropping s_0."""
        a0 = traj_flat[..., :self.action_dim]
        rest = traj_flat[..., self.transition_dim:]
        return np.concatenate([a0, rest], axis=-1)

    def from_dof(self, dof_vec, s0):
        """(dof,) + (state_dim,) -> (H*T,) full flat trajectory."""
        return np.insert(np.asarray(dof_vec).reshape(-1), self.action_dim, np.asarray(s0).reshape(-1))


# ---------------------------------------------------------------------------#
# --------------------------------- the NLP ---------------------------------#
# ---------------------------------------------------------------------------#

class HardFlowNLP:
    """The per-ODE-step prox-NLP, built once and re-solved with new parameters.

    Decision variable: the normalized dof vector (the *predicted terminal*
    trajectory x1).  Parameters: `s0` (normalized initial state), `tau` (the
    flow time, driving the tau^2 prox weight), and `x1_ref`.

    Constraints come from FMPCC's `constraint_list`, in exactly the semantics
    the DPCC `Projector` implements:
      ('ineq', (C_row, d))            C_row . x_unnorm <= d, steps 1..H-1
      ('eq',   (C_row, d))            equality,             steps 1..H-1
      ['lb'|'ub', vec]                per-dim bound; action dims from step 0,
                                      observation dims from step 1
      ['sphere_outside'|'sphere_inside', [ix, iy], center, radius]  steps 1..H-1
      ('deriv', [x_idx, dx_idx])      x[t+1] = x[t] + dt*dx[t], steps 0..H-2
    `skip_initial_state=True` (DPCC's default) is what drops step 0 above.
    """

    def __init__(self, layout, constraint_list, mins, maxs, dt=1.0,
                 reg_scale=1.0, dynamics_mode='deriv', linear_dynamics=None,
                 print_level=0, print_time=False, solver_opts=None):
        if cs is None:
            raise ImportError(
                'casadi is required for the Gen12 hardflow_new sampler. '
                'It is installed in the FMPCC cluster env; this container has no deps.')

        self.layout = layout
        self.dt = float(dt)
        self.reg_scale = float(reg_scale)
        self.dynamics_mode = dynamics_mode
        self.linear_dynamics = linear_dynamics
        self.mins = np.asarray(mins, dtype=float)
        self.maxs = np.asarray(maxs, dtype=float)
        assert self.mins.shape == (layout.transition_dim,), \
            f'normalizer limits {self.mins.shape} != transition_dim {layout.transition_dim}'

        self.n_solves = 0
        self.n_failures = 0
        # 🔴 Gen15 U2 — cumulative IPOPT wall-time (ms). The UAV frame reports the guidance
        # cost as `proj_ms`; for arm C that cost IS the NLP solve. Without this the HardFlow
        # arm repeats the Fix_1 failure: proj_ms=0 makes the eval derive
        # `fm_ms = total_ms - 0`, so the NLP time is silently booked as pure inference.
        self.solve_ms = 0.0

        self.opti = cs.Opti()
        self.s0_param = self.opti.parameter(layout.state_dim)
        self.tau_param = self.opti.parameter(1)
        self.x1_ref = self.opti.parameter(layout.dof)
        self.x1 = self.opti.variable(layout.dof)

        # tau^2-weighted proximal cost: early ODE steps are nudged, late steps
        # are pulled hard. This IS the hardflow_new schedule (PLAN §1.2).
        self.cost = (0.5 * self.reg_scale
                     * cs.sumsqr(self.x1 - self.x1_ref)
                     * self.tau_param ** 2)

        self._transitions = [self._transition_expr(t) for t in range(layout.horizon)]
        self._apply_constraints(constraint_list)
        self.opti.minimize(self.cost)

        opts = {
            'ipopt.print_level': int(print_level),
            'ipopt.hessian_approximation': 'limited-memory',
            # PLAN §3.5: CasADi's own ~7-line timing table is governed by
            # `print_time`, NOT by ipopt.print_level. Gen13 lost one log to 91%
            # timing spam by silencing only the latter. Silence both.
            'print_time': bool(print_time),
            'ipopt.sb': 'yes',
        }
        if solver_opts:
            opts.update(solver_opts)
        self.opti.solver('ipopt', opts)

    # -- symbolic helpers ----------------------------------------------------

    def _transition_expr(self, t):
        """Symbolic normalized transition vector [a_t | s_t] for step t."""
        a = self.x1[self.layout.action_index(t):
                    self.layout.action_index(t) + self.layout.action_dim]
        if t == 0:
            s = self.s0_param
        else:
            s = self.x1[self.layout.state_index(t):
                        self.layout.state_index(t) + self.layout.state_dim]
        return cs.vertcat(a, s)

    def _unnormalize(self, x_norm):
        """Normalized [-1, 1] transition vector -> physical units."""
        scale = (self.maxs - self.mins) / 2.0
        offset = (self.maxs + self.mins) / 2.0
        return x_norm * cs.DM(scale.reshape(-1, 1)) + cs.DM(offset.reshape(-1, 1))

    # -- constraint assembly -------------------------------------------------

    def _apply_constraints(self, constraint_list):
        H = self.layout.horizon
        unnorm = [self._unnormalize(x) for x in self._transitions]

        for spec in constraint_list:
            kind = spec[0]

            if kind in ('ineq', 'eq'):
                c_row, d = spec[1]
                c_row = np.asarray(c_row, dtype=float).reshape(1, -1)
                for t in range(1, H):                      # skip_initial_state
                    lhs = cs.mtimes(cs.DM(c_row), unnorm[t])
                    if kind == 'ineq':
                        self.opti.subject_to(lhs <= float(d))
                    else:
                        self.opti.subject_to(lhs == float(d))

            elif kind in ('lb', 'ub'):
                bound = np.asarray(spec[1], dtype=float)
                sign = 1.0 if kind == 'ub' else -1.0
                for dim in range(len(bound)):
                    if not np.isfinite(bound[dim]):
                        continue
                    # DPCC bounds action dims from step 0, observation dims
                    # from step 1 (SafetyConstraints.build_matrices).
                    t_start = 0 if dim < self.layout.action_dim else 1
                    for t in range(t_start, H):
                        self.opti.subject_to(
                            sign * unnorm[t][dim] <= sign * float(bound[dim]))

            elif kind in ('sphere_outside', 'sphere_inside'):
                dims, center, radius = spec[1], spec[2], float(spec[3])
                ix, iy = int(dims[0]), int(dims[1])
                cx, cy = float(center[0]), float(center[1])
                for t in range(1, H):
                    sq = ((unnorm[t][ix] - cx) ** 2 + (unnorm[t][iy] - cy) ** 2)
                    if kind == 'sphere_outside':
                        self.opti.subject_to(sq >= radius ** 2)
                    else:
                        self.opti.subject_to(sq <= radius ** 2)

            elif kind == 'deriv':
                if self.dynamics_mode != 'deriv':
                    continue
                x_idx, dx_idx = int(spec[1][0]), int(spec[1][1])
                # DPCC additionally pins x[0] to the measured state. Here s_0 is
                # a *parameter*, so any x_idx >= action_dim is already pinned
                # exactly and that row is redundant. For avoiding-d3il every
                # 'deriv' x_idx is an observation dim — assert it rather than
                # assume it.
                assert x_idx >= self.layout.action_dim, (
                    "'deriv' on an action dim would need an explicit initial-value "
                    'row; not supported (and not used by avoiding-d3il)')
                for t in range(H - 1):
                    self.opti.subject_to(
                        unnorm[t + 1][x_idx]
                        == unnorm[t][x_idx] + self.dt * unnorm[t][dx_idx])

            else:
                raise ValueError(f'Gen12 hardflow NLP: unsupported constraint kind {kind!r}')

        if self.dynamics_mode == 'linear_fit':
            self._apply_linear_dynamics()

    def _apply_linear_dynamics(self):
        """HardFlow-faithful mode: s' = A s + B a + c, in NORMALIZED units.

        PLAN §3.1: A, B, c are expressed in normalized coordinates, so they are
        only valid against the normalizer they were fit with. Use the .npz
        produced by `FM_v3_hardflow_test/fit_dynamics_fmv3.py`, NEVER HardFlow's
        own `logs/avoiding-v0/dynamics/linear_model.npz`.
        """
        assert self.linear_dynamics is not None, \
            "dynamics_mode='linear_fit' requires a fitted linear model"
        A = cs.DM(np.asarray(self.linear_dynamics['A'], dtype=float))
        B = cs.DM(np.asarray(self.linear_dynamics['B'], dtype=float))
        c = cs.DM(np.asarray(self.linear_dynamics['c'], dtype=float).reshape(-1, 1))

        H, ad, sd = self.layout.horizon, self.layout.action_dim, self.layout.state_dim
        for t in range(H - 1):
            a_t = self.x1[self.layout.action_index(t):
                          self.layout.action_index(t) + ad]
            s_t = (self.s0_param if t == 0 else
                   self.x1[self.layout.state_index(t):
                           self.layout.state_index(t) + sd])
            s_next = self.x1[self.layout.state_index(t + 1):
                             self.layout.state_index(t + 1) + sd]
            self.opti.subject_to(
                cs.mtimes(A, cs.reshape(s_t, -1, 1))
                + cs.mtimes(B, cs.reshape(a_t, -1, 1)) + c
                == cs.reshape(s_next, -1, 1))

        # Input saturation in normalized units, as upstream.
        for t in range(H):
            a_t = self.x1[self.layout.action_index(t):
                          self.layout.action_index(t) + ad]
            self.opti.subject_to(a_t <= np.ones(ad))
            self.opti.subject_to(a_t >= -np.ones(ad))

    # -- solve ---------------------------------------------------------------

    def set_s0(self, s0):
        self.opti.set_value(self.s0_param, np.asarray(s0, dtype=float).reshape(-1))

    def solve(self, x1_ref, tau):
        """Project `x1_ref` onto the feasible set at flow time `tau`."""
        x1_ref = np.asarray(x1_ref, dtype=float).reshape(-1)
        self.opti.set_value(self.tau_param, float(tau))
        self.opti.set_value(self.x1_ref, x1_ref)
        self.opti.set_initial(self.x1, x1_ref)
        self.n_solves += 1
        _t0 = time.perf_counter()
        try:
            sol = self.opti.solve_limited()
            self.solve_ms += (time.perf_counter() - _t0) * 1e3
            return np.asarray(sol.value(self.x1), dtype=float).reshape(-1)
        except RuntimeError:
            self.solve_ms += (time.perf_counter() - _t0) * 1e3   # a failed solve still costs
            # Upstream keeps the last iterate rather than aborting the episode.
            # fix_4-style counter: with IPOPT silenced this is the ONLY signal
            # left that a solve did not converge, so it is reported per episode.
            self.n_failures += 1
            return np.asarray(
                self.opti.debug.value(self.x1), dtype=float).reshape(-1)


# ---------------------------------------------------------------------------#
# ------------------------------- the sampler -------------------------------#
# ---------------------------------------------------------------------------#

def resolve_activation_threshold(activation):
    """Map a config value to a float NLP-activation threshold in [0, 1].

    fix_6: DPCC polarity. `threshold` = the fraction of the (late) trajectory over
    which the NLP is active, EXACTLY like DPCC's `diffusion_timestep_threshold`
    (higher = MORE projection). The per-step NLP is solved when τ_{k+1} ≥ (1 −
    threshold), plus always the final step. So:
        1.0 → every step (full)     0.5 → last half     0.0 → terminal-only.
    Accepts a number in [0, 1], or the aliases 'all' → 1.0 / 'late' → 0.5.
    """
    if isinstance(activation, str):
        aliases = {'all': 1.0, 'late': 0.5}
        key = activation.strip().lower()
        if key in aliases:
            return aliases[key]
        # numeric string (e.g. from an env var or CLI, like HFFM_ACT_THRESHOLD=0.5)
        try:
            activation = float(key)
        except ValueError:
            raise ValueError(
                f"activation must be a number in [0,1] or one of {list(aliases)}, "
                f"got {activation!r}")
    thr = float(activation)
    if not (0.0 <= thr <= 1.0):
        raise ValueError(f'activation_threshold must be in [0, 1], got {thr}')
    return thr


# ── B4_PARITY (2026-08-20) — per-variant MPC candidate-fan size for arm C ──────────────
# 🔴 P0. This function exists because `hardflow.batch_size` used to default to 1 while the
# DPCC arms ran `args.batch_size` (4). Both arms loop SERIALLY over candidates around their
# CPU solve (DPCC: `projection.py::Projector.project`, `for i in range(batch_size)`;
# HardFlow: `HardFlowSampler.sample`, `for b in range(batch_size)`), so a mismatched fan
# scales the projection cost almost linearly and makes every arm-B-vs-arm-C wall-clock
# comparison meaningless. It shipped: arm C looked ~25% CHEAPER than DPCC while its
# PER-SOLVE cost is ~1.8-2.2x DPCC's. See
# logs_in_develop/HF_Batch_Parity/CHANGELOG_20260820_HF_batch_parity.md and
# logs_in_develop/Gen3v6_MeanFlow/DA/DA_20260820_HF_lower_avgtime_batchsize_confound.md.
#
# THE RULE (one line): the NAME says the fan.
#   `hardflow_new`            -> 1                  faithful upstream batch-1 control.
#                                                   Upstream asserts batch==1; this arm is
#                                                   what keeps that reading available.
#   `hardflow_new-r|-c|-t`    -> `configured_batch` a selection RULE is only meaningful over
#                                                   a fan, so asking for one asks for the fan.
#                                                   Matches the DPCC arms by construction.
#   `...-tightened`           -> composes          geometry suffix, stripped before parsing.
#
# ⚠️ At B>1 `hardflow_new` and `hardflow_new-r` would be byte-identical (both select index 0),
# so running both at the same fan is duplicated compute. Pinning the bare name to 1 is what
# gives it a distinct meaning instead.
def resolve_hf_batch_size(variant, configured_batch):
    """MPC candidate-fan size for one arm-C variant. See the block comment above.

    `configured_batch` is the run-level fan (yaml `hardflow.batch_size` / `mpc_batch_size`,
    env `HFFM_BATCH`) — i.e. what the SELECTION arms get. The bare `hardflow_new` name is
    pinned to 1 regardless, because that is what the name means.

    Raises on a non-arm-C variant: the DPCC/diffuser arms take `args.batch_size` directly
    and must never be routed through here.
    """
    name = str(variant)
    for _suffix in ('_train_set', '-tightened'):   # bookkeeping/geometry suffixes
        if name.endswith(_suffix):
            name = name[:-len(_suffix)]
    if not name.startswith('hardflow'):
        raise ValueError(
            f'resolve_hf_batch_size is for arm-C (hardflow*) variants only, got {variant!r}')
    if name.endswith(('-r', '-c', '-t')):
        return max(1, int(configured_batch))
    return 1                                        # bare `hardflow_new`: faithful batch-1


class HardFlowSampler:
    """In-loop constrained ODE sampler (`hardflow_new`) for an FMv3 model.

    The model is used ONLY as a black box `v = f(x, t)`.
    """

    def __init__(self, model, layout, nlp, init_noise_scale, two_time, device='cuda',
                 activation_threshold=0.0, verbose=False):
        assert 0.0 <= activation_threshold <= 1.0, \
            f'activation_threshold must be in [0, 1], got {activation_threshold}'
        # fix_4: no default. A wrong scale is silent — it just degrades the field
        # quality — so the call site must state it explicitly (module docstring).
        assert init_noise_scale > 0, \
            f'init_noise_scale must be > 0, got {init_noise_scale}'
        # 🔴 Gen15 U2 — no default either: whether the host engine takes an `h` argument decides
        # which FIELD the NLP is handed (instantaneous vs interval-average). Both mistakes are
        # silent-ish (one TypeErrors, one quietly changes the math), so the call site must say.
        assert isinstance(two_time, bool), (
            'two_time must be an explicit bool from engine_registry (fm=False, mf/af=True); '
            f'got {two_time!r}')
        self.two_time = bool(two_time)
        self.model = model
        self.layout = layout
        self.nlp = nlp
        self.init_noise_scale = float(init_noise_scale)
        self.device = device
        # fix_6 (DPCC polarity): threshold = fraction of the late trajectory projected
        # (higher = MORE projection), matching DPCC's diffusion_timestep_threshold.
        # NLP active when τ_next ≥ (1 − threshold), plus the forced final step.
        # 1.0 = every step (full); 0.5 = last half; 0.0 = terminal-only.
        self.activation_threshold = float(activation_threshold)
        self.verbose = verbose
        self.nfe = 0

    def draw_init_noise(self, batch_size):
        """The tau=0 draw, isolated so `gate_h3` can pin its scale without an NLP."""
        return self.init_noise_scale * torch.randn(
            batch_size, self.layout.horizon, self.layout.transition_dim,
            device=self.device)

    def _velocity_batch(self, X, tau, s0_all, cond, returns):
        """Batched black-box velocity: (B, dof) torch -> (B, dof) torch, on device.

        fix_7: the candidate fan goes through the U-Net in ONE forward pass, exactly
        like DPCC's `p_sample` on a `(B, H, T)` tensor — not a Python per-candidate
        loop. Eval-mode GroupNorm has no cross-batch coupling, so each row's output
        is identical to the old batch-1 call; only the GPU-call *count* changes
        (K+n_active per plan instead of B*(K+n_active)). Stays on the GPU (returns a
        torch tensor) so the ODE integrates without a CPU round-trip per step.
        """
        L = self.layout
        B = X.shape[0]
        full = torch.cat([X[:, :L.action_dim], s0_all, X[:, L.action_dim:]], dim=1)
        traj = full.view(B, L.horizon, L.transition_dim)
        t = torch.full((B,), float(tau), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            # Gen3v6 U3: this is a VERBATIM Gen12 port EXCEPT this one line. Gen12 feeds an
            # instantaneous-velocity FM model, which returns v(x,t) directly. A mean-flow model
            # (MeanFlowODE) returns the interval average u(x,t,r); we query it at h=0 (r=t), where
            # the MeanFlow identity gives u(x,t,0) = v(x,t) EXACTLY (mf_diffusion.py:443-461). So the
            # field handed to the NLP/endpoint is identical to Gen12's, and the projection math is
            # unchanged. h=0 is passed EXPLICITLY (not relying on the h=None→0 backbone default) so a
            # future default change can't silently reintroduce the interval-average field.
            #
            # 🔴 Gen15 U2 — THREE engines share this sampler, and their velocity signatures differ:
            #     FlowMatchingODE._predict_velocity(x, cond, t, returns=None)          ← NO `h`
            #     MeanFlowODE / AlphaFlowODE._predict_velocity(x, cond, t, h=, returns=)
            # Passing `h=` to the `fm` arm is a TypeError; omitting it on a two-time arm silently
            # hands the NLP the interval-average field instead of the instantaneous one. So the
            # host engine declares which it is (`two_time`, from models/engine_registry.py) and
            # this call obeys it. Never infer it from the class name.
            if self.two_time:
                v = self.model._predict_velocity(
                    traj, cond, t, h=torch.zeros_like(t), returns=returns)
            else:
                v = self.model._predict_velocity(traj, cond, t, returns=returns)
        self.nfe += B                                          # count per-sample evals (unchanged metric)
        v_flat = v.reshape(B, -1)
        v_dof = torch.cat([v_flat[:, :L.action_dim],
                           v_flat[:, L.transition_dim:]], dim=1)
        return v_dof

    def sample(self, cond, flow_steps, batch_size=1, returns=None):
        """Run the constrained ODE.

        Returns (x, infos) with x of shape (batch, H, T) — the same contract as
        `GaussianDiffusion.p_sample_loop`, so the eval script can swap samplers.

        fix_7 (DPCC-parity compute): the batch=B candidate fan is integrated as ONE
        GPU tensor, mirroring DPCC's batched `p_sample_loop`. The per-candidate Python
        loop survives ONLY around the CPU NLP solve — which is exactly where DPCC's
        `Projector.project` also loops (`for i in range(batch_size)` scipy). So the
        parallelism structure now matches DPCC operation-for-operation: batched network
        + serial per-candidate solves + a CPU<->GPU transfer only at the NLP boundary
        (active steps), not on every velocity eval. The math is unchanged: each row
        sees the identical velocity evals and the identical per-candidate NLP solve.
        """
        L = self.layout
        K = int(flow_steps)
        dt = 1.0 / K

        # fix_4: the initial-noise law is taken from the HOST model's own sampler
        # via `init_noise_scale` (Gen3v6 MeanFlow: sigma=1.0, mf_diffusion.py:204),
        # NOT hardcoded to Gen12's 0.5. This is what actually makes arms A, B and C
        # start from the same distribution. PLAN §3.3 direction: FMv3 integrates
        # tau = 0 (noise) -> 1 (data), identical to HardFlow.
        x_init = self.draw_init_noise(batch_size)
        x_init = apply_conditioning(x_init, cond, L.action_dim, goal_dim=0)

        # dof view (drop s_0) of every candidate, kept ON the GPU for the whole ODE.
        x_init_flat = x_init.reshape(batch_size, -1)
        X = torch.cat([x_init_flat[:, :L.action_dim],
                       x_init_flat[:, L.transition_dim:]], dim=1)      # (B, dof)
        s0_all = x_init[:, 0, L.action_dim:]                           # (B, state_dim), pinned
        s0_all_np = s0_all.detach().cpu().numpy()

        # Batched conditioning: drop string keys exactly as the old per-candidate
        # path did, but keep the full B batch instead of slicing [b:b+1].
        cond_net = {k: v for k, v in cond.items() if not isinstance(k, str)}
        returns_net = returns

        # U4.2 per-candidate costs (DPCC-style selection).
        #   prox    = Σ_k ‖x1_proj − x1_ref‖²  (total NLP intervention; ranking key)
        #   control = Σ_k ‖u_k‖               (control effort; descriptor)
        # fix_4: `control` previously accumulated ‖(X_next − X_ref)/dt‖, which is the
        # NLP correction, not the field's control effort — i.e. a tau-reweighted copy
        # of `prox`, so candidate_cost='control' silently reproduced 'prox' instead of
        # offering an independent ranking. It now accumulates the velocity magnitude
        # the docstring always claimed.
        cand_prox = np.zeros(batch_size, dtype=np.float64)
        cand_ctrl = torch.zeros(batch_size, device=self.device)
        chains = [X.detach()]                                          # list of (B, dof) on device
        n_solves_before, n_fail_before = self.nlp.n_solves, self.nlp.n_failures
        nfe_before = self.nfe
        solve_ms_before = self.nlp.solve_ms

        for k in range(K):
            tau_k = k * dt
            tau_next = tau_k + dt

            V = self._velocity_batch(X, tau_k, s0_all, cond_net, returns_net)
            X_ref = X + V * dt
            cand_ctrl = cand_ctrl + V.reshape(batch_size, -1).norm(dim=1)   # fix_4: Σ_k ‖u_k‖

            # U4 + fix_6: EXACT DPCC gate. DPCC projects when `loop_idx >= (1 - T)*K`,
            # PLUS the forced final step (the terminal solve carries the safety
            # guarantee). threshold == DPCC's diffusion_timestep_threshold.
            #
            # [Gen12fix8] ROUNDING PARITY: int() added. fix_6 matched DPCC's polarity but
            # not its ROUNDING. DPCC truncates the boundary — `int((1-T)*K)`, i.e. FLOOR
            # (models/diffusion.py:207) — and upstream HardFlow floors too (`k < N//2`,
            # flow_policy.py:868). Comparing against the raw float is CEIL, which made
            # HardFlow do one FEWER projection step than both references whenever (1-T)*K
            # is not an integer. No-op at integer boundaries, so existing runs reproduce
            # bit-identically. Synced from flow_matcher_v3_hardflow (Gen12).
            # See logs_in_develop/Gen12/fix_8/.
            active = (k >= int((1.0 - self.activation_threshold) * K)) or (k == K - 1)
            if active:
                V_next = self._velocity_batch(X_ref, tau_next, s0_all, cond_net, returns_net)
                X1_ref = X_ref + (1.0 - tau_next) * V_next            # (B, dof) GPU
                # --- CPU NLP boundary: one transfer out, serial per-candidate solve
                #     (== DPCC's Projector.project loop), one transfer back.
                X1_ref_np = X1_ref.detach().cpu().numpy()
                X1_proj_np = np.empty_like(X1_ref_np)
                for b in range(batch_size):
                    self.nlp.set_s0(s0_all_np[b])                     # per-candidate s0
                    X1_proj_np[b] = np.asarray(
                        self.nlp.solve(X1_ref_np[b], tau_next),
                        dtype=X1_ref_np.dtype).reshape(-1)
                cand_prox += np.sum(
                    (X1_proj_np.astype(np.float64) - X1_ref_np.astype(np.float64)) ** 2, axis=1)
                X1_proj = torch.as_tensor(X1_proj_np, dtype=X_ref.dtype, device=self.device)
                X_next = X_ref + tau_next * (X1_proj - X1_ref)
            else:
                X_next = X_ref

            X = X_next
            chains.append(X.detach())

        # Reconstruct full trajectories (single CPU transfer of the final dof).
        X_np = X.detach().cpu().numpy()
        out = torch.zeros_like(x_init)
        for b in range(batch_size):
            full = L.from_dof(X_np[b], s0_all_np[b])
            out[b] = torch.as_tensor(full, dtype=torch.float32,
                                     device=self.device).view(L.horizon, L.transition_dim)
        out = apply_conditioning(out, cond, L.action_dim, goal_dim=0)

        # (K+1, B, dof) on device -> (B, K+1, dof) numpy, one transfer.
        dof_chains = torch.stack(chains, dim=1).detach().cpu().numpy()

        infos = {
            'projection_costs': {},
            'candidate_costs': cand_prox,                          # U4.2 ranking key (prox)
            'candidate_costs_control': cand_ctrl.detach().cpu().numpy(),  # U4.2 descriptor
            # fix_4: `nfe` was cumulative-since-construction while `nlp_solves`/
            # `nlp_failures` were per-call deltas, so summing infos across the episode
            # (as the eval driver does for the NLP counters) would have grown
            # quadratically. All three are per-call deltas now; the running total stays
            # available as `nfe_total` and on `policy.nfe`, which is what the eval
            # driver actually reports — so the logged NFE figure is unchanged.
            'nfe': self.nfe - nfe_before,
            'nfe_total': self.nfe,
            'nlp_solves': self.nlp.n_solves - n_solves_before,
            'nlp_failures': self.nlp.n_failures - n_fail_before,
            # 🔴 Gen15 U2 — same key, same units (ms), same meaning as the DPCC engines'
            # `infos['projection_ms']`: the CPU wall-time this plan spent on guidance. Emitting
            # it under the shared name is what lets `proj_ms` / `fm_ms` stay comparable across
            # arm B (project-after) and arm C (NLP-inside). Gate G7 asserts the contract.
            'projection_ms': self.nlp.solve_ms - solve_ms_before,
            'activation_threshold': self.activation_threshold,
            'dof_chains': dof_chains,
        }
        return out, infos


# ---------------------------------------------------------------------------#
# -------------------------------- the policy -------------------------------#
# ---------------------------------------------------------------------------#

class HardFlowPolicy:
    """Drop-in replacement for `sampling.policies.Policy`, arm C of PLAN §5.

    Same call signature and same `(action, Trajectories)` return, so the eval
    loop is shared byte-for-byte between arms B and C.
    """

    # DPCC-parity candidate-selection rules (U4.2). Same names as sampling.Policy.
    SELECTION_RULES = ('random', 'temporal_consistency', 'minimum_projection_cost')

    def __init__(self, model, normalizer, horizon, transition_dim, action_dim,
                 constraint_list, dt=1.0, flow_steps=None, preprocess_fns=[],
                 test_ret=0, reg_scale=1.0, activation_threshold=0.0,
                 trajectory_selection='random', candidate_cost='prox',
                 dynamics_mode='deriv', linear_dynamics=None, print_level=0,
                 print_time=False, device='cuda', goal_dim=0, verbose=False,
                 init_noise_scale=None, two_time=None):
        """🔴 Gen15 U2 — `init_noise_scale` and `two_time` are REQUIRED, no defaults.

        Gen3v6 defaulted `init_noise_scale=1.0` because it hosted exactly one engine. Gen15
        hosts three, and they disagree on BOTH values:

            engine   init_noise_scale                two_time
            fm       0.5   (diffusion.py:184)        False   (_predict_velocity has no `h`)
            mf       1.0   (mf_diffusion.py:205)     True
            af       1.0   (af_diffusion.py:261)     True

        A defaulted `init_noise_scale` is the fix_4 bug re-armed: on the `fm` arm the old 1.0
        default would start the ODE at TWICE its trained noise scale, silently. Both values are
        published per engine by `mix_uav/models/engine_registry.py`; the eval passes them from
        there. Read them off the host engine's own `p_sample_loop`, never off a sibling.
        """
        assert init_noise_scale is not None, (
            'init_noise_scale is required (fm=0.5, mf/af=1.0) — take it from '
            "engine_registry.get(engine)['init_noise_scale']")
        assert two_time is not None, (
            'two_time is required (fm=False, mf/af=True) — take it from '
            "engine_registry.get(engine)['two_time']")
        assert goal_dim == 0, (
            'Gen12 targets avoiding-d3il (goal_dim=0). A goal-conditioned env '
            'would need the goal columns carried through the dof vector.')
        assert trajectory_selection in self.SELECTION_RULES, (
            f'trajectory_selection must be one of {self.SELECTION_RULES}, '
            f'got {trajectory_selection!r}')
        assert candidate_cost in ('prox', 'control'), \
            f"candidate_cost must be 'prox' or 'control', got {candidate_cost!r}"
        from diffuser.datasets.preprocessing import get_policy_preprocess_fn

        self.model = model
        self.normalizer = normalizer
        self.action_dim = action_dim
        self.preprocess_fn = get_policy_preprocess_fn(preprocess_fns)
        self.test_ret = test_ret
        self.device = device
        self.trajectory_selection = trajectory_selection
        self.candidate_cost = candidate_cost
        self.flow_steps = int(flow_steps if flow_steps is not None
                              else getattr(model, 'flow_steps_v3', 10))

        state_dim = transition_dim - action_dim
        self.layout = TrajectoryLayout(horizon, action_dim, state_dim)

        # Limits over the FULL transition vector [actions | observations], in
        # the same order the trajectory uses. Mirrors ProjectionNormalizer.
        mins = np.concatenate([normalizer.normalizers['actions'].mins,
                               normalizer.normalizers['observations'].mins])
        maxs = np.concatenate([normalizer.normalizers['actions'].maxs,
                               normalizer.normalizers['observations'].maxs])

        self.nlp = HardFlowNLP(
            layout=self.layout, constraint_list=constraint_list,
            mins=mins, maxs=maxs, dt=dt, reg_scale=reg_scale,
            dynamics_mode=dynamics_mode, linear_dynamics=linear_dynamics,
            print_level=print_level, print_time=print_time)

        self.sampler = HardFlowSampler(
            model=model, layout=self.layout, nlp=self.nlp,
            init_noise_scale=init_noise_scale, two_time=bool(two_time), device=device,
            activation_threshold=activation_threshold, verbose=verbose)

        # 🔴 Gen15 U2 — the UAV frame's real-time logging contract. `FM_v3_uav_test`'s eval
        # reads these off the policy after every call (`getattr(policy, 'last_proj_ms', 0.0)`
        # etc.) and derives `fm_ms = total_ms - proj_ms` from them. Gen3v6's Policy had none of
        # them, which is exactly what Fix_1 repaired on the mf/af engines — this class came from
        # the same generation and had the same hole. Every `getattr` on the eval side carries a
        # default, so the omission would NOT have crashed: it would have silently booked the
        # entire IPOPT time as pure network inference. Declared here, set in __call__.
        self.last_infos = {}
        self.last_proj_ms = 0.0          # NLP wall-time for the last plan (ms)
        self.last_proj_cost = 0.0        # prox cost of the EXECUTED candidate
        self.last_which_trajectory = 0   # index of the executed candidate in the fan
        # arm C has no DPCC `Projector`; the eval's circuit-breaker probes are all
        # `getattr(policy.projector, ..., 0)`, so None gives the correct "never tripped".
        # NLP convergence health lives in `nlp.n_failures`, reported in the summary instead.
        self.projector = None

        self.prev_observations = None
        self.last_info = {}

    def __call__(self, conditions, batch_size=1, horizon=None, test_ret=None,
                 constraints=None, disable_projection=False):
        conditions = {k: self.preprocess_fn(v) for k, v in conditions.items()}
        conditions = utils.apply_dict(self.normalizer.normalize, conditions, 'observations')
        conditions = utils.to_torch(conditions, dtype=torch.float32, device=self.device)
        conditions = {k: v.unsqueeze(0).repeat(batch_size, 1) for k, v in conditions.items()}

        test_ret = test_ret if test_ret is not None else self.test_ret
        returns = to_device(test_ret * torch.ones(batch_size, 1), self.device)

        start = time.time()
        if disable_projection:
            samples, infos = self.model(conditions, returns=returns, projector=None,
                                        constraints=None, horizon=horizon)
        else:
            samples, infos = self.sampler.sample(conditions, self.flow_steps,
                                                 batch_size=batch_size, returns=returns)
        infos['wall_time'] = time.time() - start
        self.last_info = infos

        trajectories = utils.to_np(samples)
        normed_observations = trajectories[:, :, self.action_dim:]
        observations = self.normalizer.unnormalize(normed_observations, 'observations')

        # U4.2: DPCC-style candidate selection over the batch fan. Mirrors
        # sampling.Policy so arm C and arm B pick candidates by the same rules.
        which_trajectory = self._select(observations, infos, batch_size,
                                        disable_projection)
        self.prev_observations = np.repeat(
            np.expand_dims(observations[which_trajectory], axis=0), batch_size, axis=0)

        actions = trajectories[:, :, :self.action_dim]
        actions = self.normalizer.unnormalize(actions, 'actions')
        action = actions[which_trajectory, 0]

        # Gen15 U2 — publish the UAV frame's real-time diagnostics (see __init__).
        self.last_infos = infos
        self.last_proj_ms = float(infos.get('projection_ms', 0.0))
        self.last_which_trajectory = int(which_trajectory)
        _cand = infos.get('candidate_costs')
        self.last_proj_cost = (float(np.asarray(_cand).reshape(-1)[which_trajectory])
                               if _cand is not None and len(np.asarray(_cand).reshape(-1)) > which_trajectory
                               else 0.0)

        return action, Trajectories(actions, observations)

    def _select(self, observations, infos, batch_size, disable_projection):
        """Pick which candidate to execute (U4.2), mirroring DPCC's Policy."""
        if batch_size == 1 or disable_projection:
            return 0
        sel = self.trajectory_selection
        if sel == 'temporal_consistency' and self.prev_observations is not None:
            # closest to the previously-executed plan (shifted by one step)
            order = np.argsort(np.linalg.norm(
                observations[:, :-1, :] - self.prev_observations[:, 1:, :], axis=(1, 2)))
            return int(order[0])
        if sel == 'minimum_projection_cost':
            key = ('candidate_costs' if self.candidate_cost == 'prox'
                   else 'candidate_costs_control')
            costs = infos.get(key)
            if costs is not None and len(costs) == batch_size:
                return int(np.argmin(costs))
        # fix_4 (naming, not behaviour): 'random' takes slot 0, it does not draw. The
        # randomness lives in the noise, so slot 0 IS an unbiased sample of the fan —
        # and DPCC's Policy does exactly the same (`which_trajectory = 0`). Kept
        # identical so arm B and arm C stay comparable; only the comment was wrong.
        return 0  # 'random' (slot 0), or t/c fallbacks before any history exists

    @property
    def nfe(self):
        return self.sampler.nfe
