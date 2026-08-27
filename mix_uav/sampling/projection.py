import os
import time
from collections import deque
import numpy as np
import torch
from scipy.optimize import minimize, Bounds

# ─── Fix_15.2: episode-level "sustained slowness" circuit breaker ─────────────────
# Supersedes Fix_15's per-solve 2 s hard cap, which was the wrong altitude: it killed
# rare-but-legitimate hard solves AND corrupted output (every capped solve silently
# fell back to unprojected). A single slow solve is harmless — one 120 s spike among
# 400 otherwise-0.01 s steps costs nothing. The pathology worth killing is *sustained*
# slowness: when essentially EVERY step of an episode blows the real-time budget
# (JOB 23265 UAV `bounds_free`; JOB 23293 Gen7 `dpcc-r × combined_5` — thousands of
# consecutive solves each hitting the cap). So we judge recent history, not the solve:
#   • each solve keeps a *generous* wall-clock BACKSTOP (default 60 s) that only trips
#     on a genuinely non-terminating solve — normal slow solves finish and keep their
#     real projected result;
#   • a sliding window over the last WINDOW project() calls (= replan steps) counts how
#     many were "slow" (call wall-time > SLOW_MS). When ≥ TRIP_FRAC of the window is
#     slow the breaker OPENS and projection is SKIPPED (returned unprojected, ~0 ms) so
#     the episode stops bleeding time. After COOLDOWN skips it HALF-OPENs and probes one
#     real solve: fast ⇒ close (a new easy episode recovers automatically — the sliding
#     window needs no explicit episode boundary); still slow ⇒ re-open.
# A lone spike never trips it (1 slow / WINDOW); all-steps-slow trips within ~WINDOW
# steps then costs almost nothing. Set FMPCC_PROJ_CB_WINDOW=0 to disable the breaker.
_PROJ_SOLVE_BACKSTOP_S = float(os.environ.get('FMPCC_PROJ_SOLVE_BACKSTOP_S', '60.0'))   # per-solve hang backstop (s); <=0 disables
_PROJ_SLOW_MS          = float(os.environ.get('FMPCC_PROJ_SLOW_MS',          '1000.0')) # a project() call slower than this = one "slow step"
_PROJ_CB_WINDOW        = int(  os.environ.get('FMPCC_PROJ_CB_WINDOW',        '40'))      # steps of recent history to judge; 0 disables
_PROJ_CB_TRIP_FRAC     = float(os.environ.get('FMPCC_PROJ_CB_TRIP_FRAC',     '0.9'))     # fraction of a full window that must be slow to OPEN
_PROJ_CB_COOLDOWN      = int(  os.environ.get('FMPCC_PROJ_CB_COOLDOWN',      '40'))      # OPEN skips before a HALF-OPEN probe


class _SolveBudgetExceeded(Exception):
    """Raised from the SLSQP callback when one solve overruns the generous hang backstop."""
    pass

class Projector:

    def __init__(self, horizon, transition_dim, action_dim=0, goal_dim=0, constraint_list=[], normalizer=None, variant='states', 
                 dt=0.1, cost_dims=None, skip_initial_state=True, diffusion_timestep_threshold=0.5, gradient=False, gradient_weights=None,
                 device='cuda', solver='proxsuite', parallelize=False):
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.dt = torch.tensor(dt, device=device)
        self.skip_initial_state = skip_initial_state
        # self.only_last = only_last
        self.diffusion_timestep_threshold = diffusion_timestep_threshold
        self.gradient = gradient
        self.gradient_weights = gradient_weights
        self.device = device
        self.solver = solver
        self.parallelize = parallelize
        
        # Determine whether to include actions in the projection
        if normalizer is None:
            self.normalizer = None
        elif variant == 'states':
            self.normalizer = ProjectionNormalizer(observation_normalizer=normalizer.normalizers['observations'], goal_dim=goal_dim)
        elif variant == 'states_actions':
        # elif transition_dim != normalizer.normalizers['observations'].maxs.size:
            self.normalizer = ProjectionNormalizer(observation_normalizer=normalizer.normalizers['observations'], 
                                                   action_normalizer=normalizer.normalizers['actions'], goal_dim=goal_dim)
        else:
            KeyError('Invalid variant. Choose either "states" or "states_actions".')            

        # Quadratic cost
        if cost_dims is not None:
            costs = torch.ones(transition_dim, device=self.device)
            for idx in cost_dims:
                costs[idx] = 1
            self.Q = torch.diag(torch.tile(costs, (self.horizon, )))
        else:
            self.Q = torch.eye(transition_dim * horizon, device=self.device)

        self.A = torch.empty((0, self.transition_dim * self.horizon), device=self.device)   # Equality constraints
        self.b = torch.empty(0, device=self.device)
        self.C = torch.empty((0, self.transition_dim * self.horizon), device=self.device)   # Inequality constraints
        self.d = torch.empty(0, device=self.device)

        self.safety_constraints = SafetyConstraints(horizon=horizon, transition_dim=transition_dim, normalizer=self.normalizer, 
                                                 skip_initial_state=self.skip_initial_state, action_dim=action_dim, device=self.device)
        self.dynamic_constraints = DynamicConstraints(horizon=horizon, transition_dim=transition_dim, normalizer=self.normalizer,
                                                      skip_initial_state=self.skip_initial_state, dt=self.dt, device=self.device)
        self.obstacle_constraints = ObstacleConstraints(horizon=horizon, transition_dim=transition_dim, normalizer=self.normalizer,
                                                        skip_initial_state=self.skip_initial_state, dt=self.dt, device=self.device)

        for constraint_spec in constraint_list:
            if constraint_spec[0] == 'deriv':
                self.dynamic_constraints.constraint_list.append(constraint_spec)
            elif constraint_spec[0] == 'lb' or constraint_spec[0] == 'ub' or constraint_spec[0] == 'eq' or constraint_spec[0] == 'ineq':
                self.safety_constraints.constraint_list.append(constraint_spec)
            elif constraint_spec[0] == 'sphere_inside' or constraint_spec[0] == 'sphere_outside':
                self.obstacle_constraints.constraint_list.append(constraint_spec)

        self.safety_constraints.build_matrices()
        self.dynamic_constraints.build_matrices()
        self.obstacle_constraints.build_matrices()
        self.append_linear_constraint(self.safety_constraints)
        self.append_linear_constraint(self.dynamic_constraints)
        self.add_numpy_constraints()     

    def project(self, trajectory, constraints=None):
        """
            trajectory: np.ndarray of shape (batch_size, horizon, transition_dim)
            Solve an optimization problem of the form 
                \hat z =   argmin_z 1/2 z^T Q z + r^T z
                        subject to  Az  = b
                                    Cz <= d
            where z = (o_0, o_1, ..., o_{H-1}) is the trajectory in vector form. The matrices A, b, C, and d are defined by the dynamic and safety constraints.
                                    
        """

        dims = trajectory.shape

        # Fix_15.2: circuit breaker — while OPEN, skip projection entirely and return the
        # unprojected trajectory (fast), so a sustained-slow episode stops bleeding time.
        if not hasattr(self, '_cb_state'):
            self._cb_state = 'closed'
            self._cb_window = deque(maxlen=_PROJ_CB_WINDOW)
            self._cb_skips = 0
            self._cb_trips = 0
        if self._cb_state == 'open':
            self._cb_skips += 1
            if self._cb_skips >= _PROJ_CB_COOLDOWN:
                self._cb_state = 'half_open'          # next call probes one real solve
                self._cb_skips = 0
            self.last_proj_ms = 0.0
            self.last_proj_skipped = True     # Fix_15.3: this step's trajectory is UNPROJECTED
            return trajectory, np.full(dims[0], np.inf, dtype=np.float32)
        _call_t0 = time.perf_counter()
        self.last_proj_skipped = False        # Fix_15.3: real solve ran this step

        # Reshape the trajectory to a batch of vectors (from B x H x T to B x (HT)
        batch_size = trajectory.shape[0]
        trajectory_reshaped = trajectory.reshape(trajectory.shape[0], -1)

        # Cost
        r = - trajectory_reshaped @ self.Q
        r_np = r.cpu().numpy()
        Q = self.Q_np.astype('double')
        trajectory_np = trajectory_reshaped.cpu().numpy()

        # Constraints
        A = self.A_np.astype('double')
        b = self.b_np.astype('double')
        C = self.C_np.astype('double')
        d = self.d_np.astype('double')

        if self.skip_initial_state:
            s_0 = trajectory_reshaped[0, :self.transition_dim]
            if self.solver == 'proxsuite' or self.solver == 'gurobi':
                s_0 = s_0.cpu().numpy()
            counter = 0
            for constraint in self.dynamic_constraints.constraint_list:
                if constraint[0] == 'deriv':
                    x_idx = int(constraint[1][0])
                    b[counter * self.horizon] = s_0[x_idx]
                    counter += 1

        r_np_double = r_np.astype('double')
        trajectory_np_double = trajectory_np.astype('double')
        # Constraints
        constraints = ()
        for constraint_idx in range(len(self.obstacle_constraints.P_list)):
            P = self.obstacle_constraints.P_list[constraint_idx]
            q = self.obstacle_constraints.q_list[constraint_idx]
            v = self.obstacle_constraints.v_list[constraint_idx]
            for t in range(1, self.horizon):                        # Obstacle constraints
                start_idx = t * self.transition_dim
                end_idx = (t + 1) * self.transition_dim
                constraints += ({'type': 'ineq', 'fun': lambda x, start_idx=start_idx, end_idx=end_idx, P=P, q=q, v=v: -x[start_idx: end_idx] @ P @ x[start_idx: end_idx] - q @ x[start_idx: end_idx] + v,
                                    'jac': lambda x, start_idx=start_idx, end_idx=end_idx, P=P, q=q: np.concatenate([np.zeros(start_idx), -2 * P @ x[start_idx: end_idx] - q, np.zeros(len(x) - end_idx)])},)

        if C.size > 0:
            constraints += ({'type': 'ineq', 'fun': lambda x: -C @ x + d, 'jac': lambda x: -C},)
        if A.size > 0:
            constraints += ({'type': 'eq', 'fun': lambda x: A @ x - b, 'jac': lambda x: A},)   
        
        projection_costs = np.ones(batch_size, dtype=np.float32)
        sol_np = np.zeros((batch_size, self.horizon * self.transition_dim), dtype=np.float32)
        # [SolverSwap 2026-08-27] ADD-ON, behaviour-neutral: record per-solve scipy
        # convergence so a caller can COUNT failures. DPCC itself still silently keeps
        # `res.x` on non-convergence — that is unchanged here on purpose, so arm B's
        # numbers do not move. Read by HardFlowNLP._solve_slsqp for `nlp_failures`.
        self.last_solve_success = []
        for i in range(batch_size):
            # Cost
            cost_fun = lambda x: 0.5 * x @ Q @ x + r_np_double[i] @ x # + (A_double @ x - b_double) @ (A_double @ x - b_double)
            jac_cost_fun = lambda x: Q @ x + r_np_double[i]
            # Fix_15.2: generous per-solve BACKSTOP — only catches a genuinely
            # non-terminating solve. SLSQP calls the callback each iteration; raising
            # there unwinds cleanly out of minimize(). Sustained slowness is handled by
            # the circuit breaker (below), not here.
            _t0 = time.perf_counter()
            def _deadline_cb(xk, _t0=_t0):
                if _PROJ_SOLVE_BACKSTOP_S > 0 and time.perf_counter() - _t0 > _PROJ_SOLVE_BACKSTOP_S:
                    raise _SolveBudgetExceeded()
            try:
                res = minimize(fun=cost_fun,
                                x0=trajectory_np_double[i],
                                constraints=constraints,
                                method='SLSQP',
                                jac=jac_cost_fun,
                                bounds=Bounds(-5 * np.ones_like(trajectory_np_double[i]), 5 * np.ones_like(trajectory_np_double[i])),
                                tol=1e-6,
                                callback=_deadline_cb,
                                options={'maxiter': 1000, 'disp': False})
            except _SolveBudgetExceeded:
                # Rare: one solve hit the 60 s hang backstop. Keep it unprojected (inf
                # cost so selection avoids it) and let the circuit breaker judge the trend.
                sol_np[i] = trajectory_np_double[i]
                projection_costs[i] = np.inf
                self._cost_exploded_count = getattr(self, '_cost_exploded_count', 0) + 1
                print(f'[ projector ] solve backstop hit ({_PROJ_SOLVE_BACKSTOP_S:.0f}s) '
                      f'— kept unprojected trajectory (batch {i}).', flush=True)
                continue

            self.last_solve_success.append(bool(res.success))
            sol_np[i] = res.x
            projection_costs[i] = 0.5 * sol_np[i] @ Q @ sol_np[i] + r_np[i] @ sol_np[i] + 0.5 * trajectory_np[i] @ Q @ trajectory_np[i]

            # if np.linalg.norm(A_double @ res.x - b_double) > 1e-3:
            #     print('Equality constraints not satisfied!')
            # if np.any(C_double @ res.x > d_double + 1e-3):
            #     print('Inequality constraints not satisfied!')

        sol = torch.tensor(sol_np, device=self.device).reshape(dims)

        # Fix_15.2: record this step's wall-time and update the circuit breaker.
        _call_ms = (time.perf_counter() - _call_t0) * 1e3
        self.last_proj_ms = _call_ms
        _slow = _call_ms > _PROJ_SLOW_MS
        if self._cb_state == 'half_open':
            if _slow:
                self._cb_state = 'open'; self._cb_skips = 0     # still bad → re-open
            else:
                self._cb_state = 'closed'; self._cb_window.clear()
                print(f'[ projector ] circuit breaker RECOVERED — projection resumed '
                      f'(step {_call_ms:.0f} ms).', flush=True)
        elif self._cb_window.maxlen:                            # closed & breaker enabled
            self._cb_window.append(_slow)
            if len(self._cb_window) == self._cb_window.maxlen and \
               sum(self._cb_window) >= _PROJ_CB_TRIP_FRAC * self._cb_window.maxlen:
                self._cb_state = 'open'; self._cb_skips = 0; self._cb_trips += 1
                print(f'[ projector ] COST EXPLODED (sustained): {sum(self._cb_window)}/'
                      f'{self._cb_window.maxlen} recent steps > {_PROJ_SLOW_MS:.0f} ms — '
                      f'OPENING circuit breaker, skipping projection until it recovers '
                      f'(trip #{self._cb_trips}).', flush=True)

        # print(f'Projection time {self.solver}:', time.time() - start_time)
        return sol, projection_costs    # only implemented for proxsuite and scipy and parallelize=False
    
    def compute_gradient(self, trajectory, constraints=None):
        """
            trajectory: np.ndarray of shape (batch_size, horizon, transition_dim) or (horizon, transition_dim)
            Calculate the (weighted) gradients for the following cost functions:
            c_1 = ||A * tau - b||^2                             --> grad_1 = 2 * A^T (A * tau - b)
            c_2 = max(0, C * tau - d)^2                         --> grad_2 = 2 * C^T max(0, C * tau - d)
            c_3 = sum_{t=1}^{H-1} (s_t^T P s_t + q^T s_t - v)^2 --> grad_3 = 2 * (2 * P s_t + q) * (s_t^T P s_t + q^T s_t - v)                 
        """
        
        trajectory_reshaped = trajectory.reshape(trajectory.shape[0], -1)
        trajectory_np = trajectory_reshaped.cpu().numpy()

        # Constraints
        A, b, C, d = self.A, self.b, self.C, self.d

        if self.skip_initial_state:
            s_0 = trajectory_reshaped[0, :self.transition_dim]
            counter = 0
            for constraint in self.dynamic_constraints.constraint_list:
                if constraint[0] == 'deriv':
                    x_idx = int(constraint[1][0])
                    b[counter * self.horizon] = s_0[x_idx]
                    counter += 1

        # Equality and polytopic constraints
        grad1 = torch.zeros_like(trajectory_reshaped)
        grad2 = torch.zeros_like(trajectory_reshaped)
        for i in range(trajectory.shape[0]):
            grad1[i] = - A.T @ (A @ trajectory_reshaped[i] - b)
            grad2[i] = - C.T @ torch.max(torch.zeros_like(C @ trajectory_reshaped[i] - d), C @ trajectory_reshaped[i] - d)
        grad1 = grad1.reshape(trajectory.shape)
        grad2 = grad2.reshape(trajectory.shape)

        # Obstacle constraints
        grad3 = np.zeros_like(trajectory_np)
        for constraint_idx in range(len(self.obstacle_constraints.P_list)):
            P = self.obstacle_constraints.P_list[constraint_idx]
            q = self.obstacle_constraints.q_list[constraint_idx]
            v = self.obstacle_constraints.v_list[constraint_idx]
            for t in range(1, self.horizon):
                start_idx = t * self.transition_dim
                end_idx = (t + 1) * self.transition_dim
                for i in range(trajectory.shape[0]):
                    if trajectory_np[i, start_idx: end_idx] @ P @ trajectory_np[i, start_idx: end_idx] + q @ trajectory_np[i, start_idx: end_idx] <= v:
                        continue
                    else:
                        grad3[i, start_idx: end_idx] -= 2 * P @ trajectory_np[i, start_idx: end_idx] + q   
        grad3 = torch.tensor(grad3, device=self.device).reshape(trajectory.shape)
        
        if self.gradient_weights is not None:
            grad1 = grad1 * self.gradient_weights[0]
            grad2 = grad2 * self.gradient_weights[1]
            grad3 = grad3 * self.gradient_weights[2]
        
        return grad1 + grad2 + grad3 

    def append_linear_constraint(self, constraint):
        self.C = torch.cat([self.C, constraint.C], dim=0)
        self.d = torch.cat([self.d, constraint.d], dim=0)
        self.A = torch.cat([self.A, constraint.A], dim=0)
        self.b = torch.cat([self.b, constraint.b], dim=0)
        if constraint.__class__.__name__ == 'SafetyConstraints':
            self.A_safe, self.b_safe, self.C_safe, self.d_safe = constraint.A, constraint.b, constraint.C, constraint.d
        elif constraint.__class__.__name__ == 'DynamicConstraints':
            self.A_dyn, self.b_dyn, self.C_dyn, self.d_dyn = constraint.A, constraint.b, constraint.C, constraint.d

    def add_numpy_constraints(self):
        self.A_np = self.A.cpu().numpy()
        self.b_np = self.b.cpu().numpy()     
        self.C_np = self.C.cpu().numpy()
        self.d_np = self.d.cpu().numpy()
        self.Q_np = self.Q.cpu().numpy()


class Constraints:

    def __init__(self, horizon, transition_dim, normalizer=None, device='cuda'):
        self.horizon = horizon
        self.transition_dim = transition_dim
        self.normalizer = normalizer
        self.device = device

        self.A = torch.empty((0, self.transition_dim * self.horizon), device=device)
        self.b = torch.empty(0, device=device)
        self.C = torch.empty((0, self.transition_dim * self.horizon), device=device)
        self.d = torch.empty(0, device=device)

    def build_matrices(self):
        pass

class SafetyConstraints(Constraints):

    def __init__(self, skip_initial_state=True, action_dim=0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial_state = skip_initial_state
        self.action_dim = action_dim
        self.constraint_list = []
        
    def build_matrices(self, constraint_list=None):
        """
            Input:
                constraint_list: list of constraints
                    e.g. [('lb', [-1.0, -inf, 0]), ('ub', [1.0, 2.0, inf]), ('eq', ([0, 1, 1], 1.5)), ('ineq': ([1, 0, 0], 0.5))] -->
                    x_0 in [-1, 1], x_1 in [-inf, 2], x_2 in [0, inf], 0 * x_0 + 1 * x_1 + 1 * x_2 = 1.5, 1 * x_0 + 0 * x_1 + 0 * x_2 <= 0.5,
                    where x_i is the i-th dimension of the state or state-action vectors
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)
                lb: horizon * transition_dim
            C consists of n_bounds blocks of shape (horizon, transition_dim * horizon), where each block corresponds to a 
            constraint (ub or lb) on a specific dimension. The block has a 1 or -1 at the corresponding dimension and time step.
        """

        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)

        for constraint in constraint_list:
            type = constraint[0]
            bound = constraint[1]
            if type == 'lb' or type == 'ub':
                for dim in range(len(bound)):
                    if bound[dim] == -np.inf or bound[dim] == np.inf:
                        continue
                    
                    mat_append = torch.zeros(self.horizon, self.transition_dim * self.horizon, device=self.device)
                    vec_append = torch.zeros(self.horizon, device=self.device)

                    sign = 1 if type == 'ub' else -1
                    for t in range(self.horizon):
                        mat_append[t, t * self.transition_dim + dim] = sign
                        vec_append[t] = sign * bound[dim]
                        
                    if self.normalizer is not None:
                        x_min = self.normalizer.mins[dim]
                        x_max = self.normalizer.maxs[dim]
                        mat_append = mat_append * (x_max - x_min) / 2
                        vec_append = vec_append - sign * (x_min + x_max) / 2

                    if self.skip_initial_state and dim >= self.action_dim:
                        mat_append = mat_append[1:]
                        vec_append = vec_append[1:]

                    self.C = torch.cat((self.C, mat_append), dim=0)
                    self.d = torch.cat((self.d, vec_append), dim=0)
                continue         

            # type == 'eq' or 'ineq'
            mat_append = torch.zeros(self.horizon, self.transition_dim * self.horizon, device=self.device)
            vec_append = torch.zeros(self.horizon, device=self.device)

            for i in range(self.horizon):
                if self.normalizer is not None:
                    x_min = self.normalizer.mins
                    x_max = self.normalizer.maxs

                    # Unnormalize the constraints. TODO: Extend to actions by checking if dim <= action_dim
                    # We have Cs <= d, where s is the unnormalized state vector. This is converted to C's_n <= d', where s_n is the normalized state vector,
                    # by using the fact that s = (s_n + 1) * (s_max - s_min) / 2 + s_min.
                    a = bound[0] * (x_max - x_min) / 2
                    b = bound[1] - bound[0] @ (x_max + x_min) / 2
                else:
                    a = bound[0]
                    b = bound[1]
                
                mat_append[i, i * self.transition_dim: (i + 1) * self.transition_dim] = torch.tensor(a, device=self.device)
                vec_append[i] = torch.tensor(b, device=self.device)

            if self.skip_initial_state:
                mat_append = mat_append[1:]
                vec_append = vec_append[1:]

            if type == 'eq':
                self.A = torch.cat((self.A, mat_append), dim=0)
                self.b = torch.cat((self.b, vec_append), dim=0)
            else:
                self.C = torch.cat((self.C, mat_append), dim=0)
                self.d = torch.cat((self.d, vec_append), dim=0)         

class DynamicConstraints(Constraints):
    def __init__(self, skip_initial_state=True, dt=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial_state = skip_initial_state
        self.dt = dt
        self.constraint_list = []
        
    def build_matrices(self, constraint_list=None):
        """
            Input:
                constraint_list: list of constraints
                    e.g. [('deriv', [0, 2]), ('deriv', [1, 3])] -->
                    x_0[t+1] = x_0[t] + self.dt * x_2[t], x_1[t+1] = x_1[t] + self.dt * x_3[t]                                      (explicit Euler) or
                    x_0[t+1] = x_0[t] + self.dt * (x_2[t] + x_2[t+1]) / 2, x_1[t+1] = x_1[t] + self.dt * (x_3[t] + x_3[t+1]) / 2    (variant of trapezoidal rule)
                    where x_i[t] is the i-th dimension of the state or state-action vectors at time t
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)
        """

        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)
        
        for constraint in constraint_list:
            type = constraint[0]
            vals = constraint[1]
            if 'deriv' in type:
                x_idx = int(vals[0])
                dx_idx = int(vals[1])
            
                mat_append = torch.zeros(self.horizon - 1, self.transition_dim * self.horizon, device=self.device)
                vec_append = torch.zeros(self.horizon - 1, device=self.device)

                # Calculate multiplicative factors needed for normalization
                if self.normalizer is not None: 
                    x_min = self.normalizer.mins[x_idx]
                    x_max = self.normalizer.maxs[x_idx]
                    dx_min = self.normalizer.mins[dx_idx]
                    dx_max = self.normalizer.maxs[dx_idx]
                    x_diff = x_max - x_min
                    dx_diff = dx_max - dx_min
                    dx_sum = dx_max + dx_min

                for i in range(self.horizon - 1):
                    if self.normalizer is not None:
                        mat_append[i, i * self.transition_dim + x_idx] = 1 * x_diff
                        mat_append[i, i * self.transition_dim + dx_idx] = self.dt * dx_diff
                        mat_append[i, (i + 1) * self.transition_dim + x_idx] = -1 * x_diff
                        vec_append[i] = - dx_sum * self.dt
                    else:
                        mat_append[i, i * self.transition_dim + x_idx] = 1
                        mat_append[i, i * self.transition_dim + dx_idx] = self.dt
                        mat_append[i, (i + 1) * self.transition_dim + x_idx] = -1
                        vec_append[i] = 0

                if self.skip_initial_state:     # --> Do that in the projection method because it needs the current state. For that, record the relevant rows
                    mat_fix_initial = torch.zeros(1, self.transition_dim * self.horizon, device=self.device)    # Fix the initial state
                    mat_fix_initial[0, x_idx] = 1
                    mat_append = torch.cat((mat_fix_initial, mat_append), dim=0)
                    vec_append = torch.cat((torch.tensor([0], device=self.device), vec_append), dim=0)          # Must be changed to current state in each iteration!

                self.A = torch.cat((self.A, mat_append), dim=0)
                self.b = torch.cat((self.b, vec_append), dim=0)

class ObstacleConstraints(Constraints):
    def __init__(self, skip_initial_state=True, dt=0.02, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_initial_state = skip_initial_state
        self.dt = dt
        self.constraint_list = []

    def build_matrices(self, constraint_list=None):
        """
            Input:
                constraint_list: list of constraints
                    e.g. [('sphere_inside', [0, 2] [-1, 5], 1), ('sphere_outside', [1, 3], [0, 1], 4)] -->
                    (x_0 + 1)^2 + (x_2 - 5)^2 <= 1, x_1^2 + (x_3 - 1)^2 >= 4
                    where x_i is the i-th dimension of the state or state-action vectors at time t
            Generate the matrix P and the vector q for:
                s_i^T P s_i + q^T s_i <= v
            The matrices have the following shapes:
                C: (horizon * n_bounds, transition_dim * horizon)
                d: (horizon * n_bounds)
        """

        if constraint_list is None:
            constraint_list = self.constraint_list
        else:
            self.constraint_list.extend(constraint_list)

        self.P_list = []
        self.q_list = []
        self.v_list = []
        for constraint in constraint_list:
            type = constraint[0]
            dims = constraint[1]
            center = constraint[2]
            radius = constraint[3]

            P = np.zeros((self.transition_dim, self.transition_dim))
            q = np.zeros(self.transition_dim)
            v = radius ** 2

            dim_counter = 0
            for dim in dims:
                if self.normalizer is not None:
                    delta_s = self.normalizer.maxs[dim] - self.normalizer.mins[dim]
                    s_min = self.normalizer.mins[dim]
                    P[dim, dim] = delta_s ** 2 / 4
                    q[dim] = delta_s**2 / 2 + delta_s * (s_min - center[dim_counter])
                    v -= delta_s**2 / 4 + delta_s * (s_min - center[dim_counter]) + (s_min - center[dim_counter]) ** 2
                else:
                    P[dim, dim] = 1
                    q[dim] = -2 * center[dim_counter]
                    v -= center[dim_counter] ** 2
                dim_counter += 1

            if type == 'sphere_outside':
                P = -P
                q = -q
                v = -v

            self.P_list.append(P)
            self.q_list.append(q)
            self.v_list.append(v)    


class ProjectionNormalizer():
    def __init__(self, observation_normalizer=None, action_normalizer=None, goal_dim=0):
        self.observation_normalizer = observation_normalizer
        self.action_normalizer = action_normalizer
        self.goal_dim = goal_dim
        self.get_limits()

    def get_limits(self):
        if self.observation_normalizer is not None and self.action_normalizer is not None:
            x_max_obs = self.observation_normalizer.maxs[:-self.goal_dim] if self.goal_dim > 0 else self.observation_normalizer.maxs
            x_min_obs = self.observation_normalizer.mins[:-self.goal_dim] if self.goal_dim > 0 else self.observation_normalizer.mins
            x_max = np.concatenate([self.action_normalizer.maxs, x_max_obs])
            x_min = np.concatenate([self.action_normalizer.mins, x_min_obs])
        elif self.observation_normalizer is not None:
            x_max = self.observation_normalizer.maxs[:-self.goal_dim] if self.goal_dim > 0 else self.observation_normalizer.maxs
            x_min = self.observation_normalizer.mins[:-self.goal_dim] if self.goal_dim > 0 else self.observation_normalizer.mins
        elif self.action_normalizer is not None:
            x_max = self.action_normalizer.maxs
            x_min = self.action_normalizer.mins
        
        self.maxs = x_max
        self.mins = x_min

    def normalize(self, x):
        x_normalized = (x - self.mins) / (self.maxs - self.mins) * 2 - 1
        return x_normalized
    
    def unnormalize(self, x_normalized):
        x = (x_normalized + 1) * (self.maxs - self.mins) / 2 + self.mins
        return x
                