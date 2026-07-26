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
  * The initial noise matches FMv3's sampler (`0.5 * randn`), not HardFlow's
    unit `randn`.
  * IPOPT and CasADi are silenced by default (PLAN §3.5).
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
        try:
            sol = self.opti.solve_limited()
            return np.asarray(sol.value(self.x1), dtype=float).reshape(-1)
        except RuntimeError:
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


class HardFlowSampler:
    """In-loop constrained ODE sampler (`hardflow_new`) for an FMv3 model.

    The model is used ONLY as a black box `v = f(x, t)`.
    """

    def __init__(self, model, layout, nlp, device='cuda', activation_threshold=0.0,
                 verbose=False):
        assert 0.0 <= activation_threshold <= 1.0, \
            f'activation_threshold must be in [0, 1], got {activation_threshold}'
        self.model = model
        self.layout = layout
        self.nlp = nlp
        self.device = device
        # fix_6 (DPCC polarity): threshold = fraction of the late trajectory projected
        # (higher = MORE projection), matching DPCC's diffusion_timestep_threshold.
        # NLP active when τ_next ≥ (1 − threshold), plus the forced final step.
        # 1.0 = every step (full); 0.5 = last half; 0.0 = terminal-only.
        self.activation_threshold = float(activation_threshold)
        self.verbose = verbose
        self.nfe = 0

    def _velocity(self, dof_np, tau, s0_torch, cond, returns):
        """Black-box velocity on the dof vector: (dof,) x float -> (dof,)."""
        L = self.layout
        dof = torch.as_tensor(dof_np, dtype=torch.float32, device=self.device).reshape(1, -1)
        full = torch.cat([dof[:, :L.action_dim], s0_torch, dof[:, L.action_dim:]], dim=1)
        traj = full.view(1, L.horizon, L.transition_dim)
        t = torch.full((1,), float(tau), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            v = self.model._predict_velocity(traj, cond, t, returns=returns)
        self.nfe += 1
        v_flat = v.reshape(1, -1)
        v_dof = torch.cat([v_flat[:, :L.action_dim],
                           v_flat[:, L.transition_dim:]], dim=1)
        return v_dof.cpu().numpy().reshape(-1)

    def sample(self, cond, flow_steps, batch_size=1, returns=None):
        """Run the constrained ODE.

        Returns (x, infos) with x of shape (batch, H, T) — the same contract as
        `GaussianDiffusion.p_sample_loop`, so the eval script can swap samplers.
        """
        L = self.layout
        K = int(flow_steps)
        dt = 1.0 / K

        # Same initial-noise law as FMv3's own sampler (0.5 * randn), so arms A,
        # B and C start from the same distribution. PLAN §3.3 direction: FMv3
        # integrates tau = 0 (noise) -> 1 (data), identical to HardFlow.
        x_init = 0.5 * torch.randn(batch_size, L.horizon, L.transition_dim,
                                   device=self.device)
        x_init = apply_conditioning(x_init, cond, L.action_dim, goal_dim=0)

        out = torch.zeros_like(x_init)
        chains = []
        # U4.2: per-candidate cost, for DPCC-style candidate selection.
        #   prox    = Σ_k ‖x1_proj − x1_ref‖²  (total NLP intervention; ranking key)
        #   control = Σ_k ‖u_k‖               (control effort; descriptor)
        # By the minimal-intervention principle the least-intervened candidate is the
        # most faithful to the field — the analog of DPCC's minimum_projection_cost.
        prox_costs, ctrl_costs = [], []
        n_solves_before, n_fail_before = self.nlp.n_solves, self.nlp.n_failures

        for b in range(batch_size):
            s0_torch = x_init[b:b + 1, 0, L.action_dim:]
            s0_np = s0_torch.detach().cpu().numpy().reshape(-1)
            self.nlp.set_s0(s0_np)

            cond_b = {k: v[b:b + 1] for k, v in cond.items() if not isinstance(k, str)}
            returns_b = returns[b:b + 1] if returns is not None else None

            x_flat = x_init[b].reshape(-1).detach().cpu().numpy()
            x_k = L.to_dof(x_flat)
            chain = [x_k.copy()]
            cand_prox, cand_ctrl = 0.0, 0.0

            for k in range(K):
                tau_k = k * dt
                tau_next = tau_k + dt

                v_k = self._velocity(x_k, tau_k, s0_torch, cond_b, returns_b)
                x_ref = x_k + v_k * dt

                # U4 + fix_6: EXACT DPCC gate. DPCC projects when
                # `loop_idx >= (1 - T)*K`; we use the identical `k >= (1 - threshold)*K`
                # (threshold == DPCC's diffusion_timestep_threshold, higher = more
                # projection) PLUS the forced final step. The terminal solve is what the
                # safety guarantee rides on (paper Prop.; PLAN U4 §5.2) — `or k == K-1`
                # is mandatory and makes threshold 0.0 => terminal-only (not truly none).
                active = (k >= (1.0 - self.activation_threshold) * K) or (k == K - 1)
                if active:
                    v_next = self._velocity(x_ref, tau_next, s0_torch, cond_b, returns_b)
                    x1_ref = x_ref + (1.0 - tau_next) * v_next
                    x1_proj = self.nlp.solve(x1_ref, tau_next)
                    cand_prox += float(np.sum((np.asarray(x1_proj) - x1_ref) ** 2))
                    x_next = x_ref + tau_next * (x1_proj - x1_ref)
                else:
                    x_next = x_ref
                x_next = np.asarray(x_next, dtype=float).reshape(-1)
                cand_ctrl += float(np.linalg.norm((x_next - x_ref) / dt))
                x_k = x_next
                chain.append(x_k.copy())

            prox_costs.append(cand_prox)
            ctrl_costs.append(cand_ctrl)
            full = L.from_dof(x_k, s0_np)
            out[b] = torch.as_tensor(full, dtype=torch.float32,
                                     device=self.device).view(L.horizon, L.transition_dim)
            chains.append(np.stack(chain, axis=0))

        out = apply_conditioning(out, cond, L.action_dim, goal_dim=0)

        infos = {
            'projection_costs': {},
            'candidate_costs': np.array(prox_costs),           # U4.2 ranking key (prox)
            'candidate_costs_control': np.array(ctrl_costs),   # U4.2 descriptor
            'nfe': self.nfe,
            'nlp_solves': self.nlp.n_solves - n_solves_before,
            'nlp_failures': self.nlp.n_failures - n_fail_before,
            'activation_threshold': self.activation_threshold,
            'dof_chains': np.stack(chains, axis=0),
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
                 print_time=False, device='cuda', goal_dim=0, verbose=False):
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
            model=model, layout=self.layout, nlp=self.nlp, device=device,
            activation_threshold=activation_threshold, verbose=verbose)

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
        return 0  # 'random' (index 0), or t/c fallbacks before any history exists

    @property
    def nfe(self):
        return self.sampler.nfe
