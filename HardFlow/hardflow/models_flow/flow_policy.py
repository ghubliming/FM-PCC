from collections import namedtuple
from dataclasses import dataclass
import math
import time
import torch
from torch import nn
from hardflow.config.flow_matching import FlowMatchingEvaluationConfig
from hardflow import utils
from hardflow.models_flow.flow_matcher import (
    apply_conditioning,
    apply_conditioning_from_conditioned_x,
)
from hardflow.utils.arrays import to_torch

import numpy as np
from hardflow.utils.avoiding_geometry import (
    NOVEL_BOUNDARY_LINES,
    NOVEL_HARD_QUADRILATERAL_EDGES,
    NOVEL_OBSTACLE_CENTER,
    NOVEL_OBSTACLE_RADIUS,
    PILLAR_CENTERS,
    PILLAR_RADII,
    is_novel_constraint,
    uses_hard_quadrilateral,
)

try:
    import casadi as cs
except ImportError:
    cs = None

try:
    from qpth.qp import QPFunction, QPSolvers
except ImportError:
    QPFunction = None

Trajectories = namedtuple("Trajectories", "actions observations values")


def to_np(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return x


class ConditionedODESolver(nn.Module):
    def __init__(
        self,
        model,
        conditions,
        action_dim,
        value=None,
        ode_method="euler",
        guidance_lr=1.0,
        guidance_steps=0,
    ):
        super().__init__()
        self.model = model
        self.conditions = conditions
        self.value = value
        self.action_dim = action_dim
        assert ode_method in ["euler"], "Only Euler is supported for now"

        self.guidance_lr = guidance_lr
        self.guidance_steps = guidance_steps

    def forward(self, x, t_span, return_chain=False, *args, **kwargs):
        # x: (batch_size, planning_horizon, transition_dim)
        # t_span: at least 2 elements, should be evenly spaced
        assert len(t_span) > 1, "t_span must have at least 2 elements"

        x0 = x.clone()
        dt = t_span[1] - t_span[0]
        x_chain = [x0]
        for t in t_span:

            if self.value is not None and self.guidance_steps > 0:
                for _ in range(self.guidance_steps):
                    inputs = x.detach().clone().requires_grad_(True)
                    vt = self.model(x, t)
                    vt = apply_conditioning_from_conditioned_x(
                        vt, torch.zeros_like(x), self.conditions, self.action_dim
                    )
                    loss = -self.value(inputs + vt * (1.0 - t))
                    if loss.ndim > 0:
                        loss = loss.mean()
                    (g,) = torch.autograd.grad(loss, inputs, create_graph=False)

                    g = apply_conditioning_from_conditioned_x(
                        g.detach(),
                        torch.zeros_like(x),
                        self.conditions,
                        self.action_dim,
                    )

                    x = x - self.guidance_lr * g

            dx_dt = self.model(x, t)
            dx_dt = apply_conditioning_from_conditioned_x(
                dx_dt, torch.zeros_like(x), self.conditions, self.action_dim
            )
            x = x + dx_dt * dt
            x = x.detach()
            x_chain.append(x)

        if return_chain:
            return x, torch.stack(x_chain, dim=1).detach()
        else:
            return x


class FlowPolicy(nn.Module):
    def __init__(
        self,
        flow_model,
        value_model,
        normalizer,
        action_dim,
        state_dim,
        horizon,
        cfg: FlowMatchingEvaluationConfig,
        constrained_value_fn=None,
        constrained_flow_fn=None,
        dynamics_model=None,
    ):
        super().__init__()
        self.flow_model = flow_model
        self.normalizer = normalizer
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.horizon = horizon

        self.cfg = cfg
        self.value_model = value_model

        self.constrained_value_fn = constrained_value_fn
        self.constrained_flow_fn = constrained_flow_fn
        self.dynamics_model = dynamics_model

        self.obs_norm_mins = self.normalizer.normalizers["observations"].mins
        self.obs_norm_maxs = self.normalizer.normalizers["observations"].maxs
        self.action_norm_mins = self.normalizer.normalizers["actions"].mins
        self.action_norm_maxs = self.normalizer.normalizers["actions"].maxs

    def __call__(self, conditions, batch_size=1, verbose=True):

        if self.cfg.guidance_method in ["original"]:
            pass
        elif self.cfg.guidance_method in ["gradient_guidance"]:
            return self.gradient_guidance_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ["projection"]:
            return self.projection_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ["projection_relaxed"]:
            return self.projection_relaxed_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ["hardflow"]:
            return self.hardflow_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ["hardflow_new"]:
            return self.hardflow_new_forward(conditions, batch_size)
        elif self.cfg.guidance_method in ["oc_flow"]:
            return self.oc_flow_forward(conditions, batch_size)
        else:
            raise ValueError(f"Unsupported guidance method: {self.cfg.guidance_method}")

        assert batch_size == 1, "batch_size must be 1 for no guidance"

        # normalize the observation
        conditions = utils.apply_dict(
            self.normalizer.normalize, conditions, "observations"
        )

        t_start = time.time()

        solver = ConditionedODESolver(
            self.flow_model,
            conditions,
            value=None,
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
        )

        x = torch.randn(
            batch_size,
            self.horizon,
            self.action_dim + self.state_dim,
            device=self.cfg.device,
        )  # (batch_size, planning_horizon, transition_dim)
        x = apply_conditioning(
            x, to_torch(conditions, device=x.device), self.action_dim
        )

        x, x_chain = solver(
            x,
            t_span=torch.linspace(
                *self.cfg.ode_t_span, self.cfg.ode_t_steps + 1, device=x.device
            )[:-1],
            return_chain=True,
        )

        normed_actions = x[:, :, : self.action_dim]
        actions = self.normalizer.unnormalize(to_np(normed_actions), "actions")

        normed_observations = x[:, :, self.action_dim :]
        observations = self.normalizer.unnormalize(
            to_np(normed_observations), "observations"
        )

        values = self.value_model(
            torch.cat((normed_actions, normed_observations), dim=-1)
        )
        values = to_np(values)

        trajectories = Trajectories(actions, observations, values)

        action = actions[0, 0]

        x1_estimation = self.x1_estimate(x_chain, conditions)
        x1_estimation = self.unnormalize_chain(x1_estimation)

        x_chain = self.unnormalize_chain(x_chain)

        t_end = time.time()
        computation_time = t_end - t_start

        info = {"computation_time": computation_time}
        return action, trajectories, x_chain, x1_estimation, info

    def x1_estimate(self, x_chain, conditions):

        batch_size, n_steps, horizon, transition_dim = x_chain.shape

        x1_estimation = torch.zeros_like(x_chain)

        for k in range(n_steps):
            t = to_torch(k / (n_steps - 1), device=x_chain.device)
            current_x = x_chain[:, k, :, :]

            current_v = self.flow_model(current_x, t)
            current_v = apply_conditioning_from_conditioned_x(
                current_v, torch.zeros_like(current_v), conditions, self.action_dim
            )

            predicted_x = current_x + (1.0 - t) * current_v
            x1_estimation[:, k, :, :] = predicted_x

        return x1_estimation

    def unnormalize_chain(self, x_chain):

        normed_x_chain_observation = x_chain[:, :, :, self.action_dim :]
        x_chain_observation = self.normalizer.unnormalize(
            to_np(normed_x_chain_observation), "observations"
        )
        normed_x_chain_action = x_chain[:, :, :, : self.action_dim]
        x_chain_action = self.normalizer.unnormalize(
            to_np(normed_x_chain_action), "actions"
        )
        x_chain = np.concatenate((x_chain_action, x_chain_observation), axis=-1)

        return x_chain

    def _quadrilateral_constraint_expr(self, x_pos, y_pos):
        signed_distances = []
        for vertex, next_vertex in NOVEL_HARD_QUADRILATERAL_EDGES:
            edge = next_vertex - vertex
            edge_norm = float(np.linalg.norm(edge))
            signed_distance = (
                edge[0] * (y_pos - float(vertex[1]))
                - edge[1] * (x_pos - float(vertex[0]))
            ) / edge_norm
            signed_distances.append(signed_distance)

        quadrilateral_constraint = signed_distances[0]
        for signed_distance in signed_distances[1:]:
            quadrilateral_constraint = cs.fmax(
                quadrilateral_constraint, signed_distance
            )

        return quadrilateral_constraint

    def _apply_obstacle_constraints(self, opti, X_var, X_index_selector=None):
        obs_norm_min_x = float(self.obs_norm_mins[2])
        obs_norm_max_x = float(self.obs_norm_maxs[2])
        obs_norm_min_y = float(self.obs_norm_mins[3])
        obs_norm_max_y = float(self.obs_norm_maxs[3])

        # apply constraints for each horizon step (except the first)
        for i in range(self.horizon - 1):
            if X_index_selector is None:
                x_pos = X_var[
                    -1,
                    2 * self.action_dim + i * (self.action_dim + self.state_dim) + 2,
                ]
                y_pos = X_var[
                    -1,
                    2 * self.action_dim + i * (self.action_dim + self.state_dim) + 3,
                ]
            elif X_index_selector == "projection":
                x_pos = X_var[
                    2 * self.action_dim + i * (self.action_dim + self.state_dim) + 2
                ]
                y_pos = X_var[
                    2 * self.action_dim + i * (self.action_dim + self.state_dim) + 3
                ]
            else:
                raise NotImplementedError

            unnorm_x_pos = (x_pos + 1.0) / 2.0 * (
                obs_norm_max_x - obs_norm_min_x
            ) + obs_norm_min_x
            unnorm_y_pos = (y_pos + 1.0) / 2.0 * (
                obs_norm_max_y - obs_norm_min_y
            ) + obs_norm_min_y

            # original constraints that are always applied
            for center, radius in zip(PILLAR_CENTERS, PILLAR_RADII):
                opti.subject_to(
                    np.sqrt(
                        (unnorm_x_pos - center[0]) ** 2
                        + (unnorm_y_pos - center[1]) ** 2
                    )
                    >= radius + self.cfg.obstacle_margin
                )

            if is_novel_constraint(self.cfg.constraint):
                opti.subject_to(
                    np.sqrt(
                        (unnorm_x_pos - NOVEL_OBSTACLE_CENTER[0]) ** 2
                        + (unnorm_y_pos - NOVEL_OBSTACLE_CENTER[1]) ** 2
                    )
                    >= NOVEL_OBSTACLE_RADIUS + self.cfg.obstacle_margin
                )

                for slope, intercept, direction in NOVEL_BOUNDARY_LINES:
                    if direction == "below":
                        opti.subject_to(
                            slope * unnorm_x_pos + intercept - unnorm_y_pos
                            >= self.cfg.obstacle_margin
                        )
                    else:
                        raise ValueError(f"Unsupported boundary direction: {direction}")

                if uses_hard_quadrilateral(self.cfg.constraint):
                    opti.subject_to(
                        self._quadrilateral_constraint_expr(unnorm_x_pos, unnorm_y_pos)
                        >= self.cfg.obstacle_margin
                    )
            else:
                raise ValueError(f"Unsupported constraint: {self.cfg.constraint}")

    def _apply_dynamics_constraints(
        self, opti, X_var, X_index_selector=None, relaxation=0.0
    ):

        assert self.dynamics_model is not None, "dynamics model must be loaded"

        for i in range(self.horizon - 2):
            if X_index_selector is None:
                state_index = 2 * self.action_dim + i * (
                    self.action_dim + self.state_dim
                )
                state = X_var[
                    -1,
                    state_index : state_index + self.state_dim,
                ]
                next_state_index = state_index + (self.action_dim + self.state_dim)
                next_state = X_var[
                    -1,
                    next_state_index : next_state_index + self.state_dim,
                ]
                action_index = state_index - self.action_dim
                action = X_var[
                    -1,
                    action_index : action_index + self.action_dim,
                ]

            elif X_index_selector == "projection":
                state_index = 2 * self.action_dim + i * (
                    self.action_dim + self.state_dim
                )
                state = X_var[state_index : state_index + self.state_dim]
                next_state_index = state_index + (self.action_dim + self.state_dim)
                next_state = X_var[next_state_index : next_state_index + self.state_dim]
                action_index = state_index - self.action_dim
                action = X_var[action_index : action_index + self.action_dim]
            else:
                raise NotImplementedError

            if relaxation == 0.0:
                opti.subject_to(
                    cs.mtimes(self.dynamics_model["A"], cs.reshape(state, -1, 1))
                    + cs.mtimes(self.dynamics_model["B"], cs.reshape(action, -1, 1))
                    + cs.reshape(self.dynamics_model["c"], -1, 1)
                    == cs.reshape(next_state, -1, 1)
                )
            else:
                opti.subject_to(
                    cs.mtimes(self.dynamics_model["A"], cs.reshape(state, -1, 1))
                    + cs.mtimes(self.dynamics_model["B"], cs.reshape(action, -1, 1))
                    + cs.reshape(self.dynamics_model["c"], -1, 1)
                    <= cs.reshape(next_state, -1, 1) + relaxation
                )
                opti.subject_to(
                    cs.mtimes(self.dynamics_model["A"], cs.reshape(state, -1, 1))
                    + cs.mtimes(self.dynamics_model["B"], cs.reshape(action, -1, 1))
                    + cs.reshape(self.dynamics_model["c"], -1, 1)
                    >= cs.reshape(next_state, -1, 1) - relaxation
                )

        if X_index_selector is None:
            s1_index = 2 * self.action_dim
            s1 = X_var[-1, s1_index : s1_index + self.state_dim]
            a0_index = 0
            a0 = X_var[-1, a0_index : a0_index + self.action_dim]
        elif X_index_selector == "projection":
            s1_index = 2 * self.action_dim
            s1 = X_var[s1_index : s1_index + self.state_dim]
            a0_index = 0
            a0 = X_var[a0_index : a0_index + self.action_dim]
        else:
            raise NotImplementedError

        if relaxation == 0.0:
            opti.subject_to(
                cs.mtimes(self.dynamics_model["A"], cs.reshape(self.oc_s0_param, -1, 1))
                + cs.mtimes(self.dynamics_model["B"], cs.reshape(a0, -1, 1))
                + cs.reshape(self.dynamics_model["c"], -1, 1)
                == cs.reshape(s1, -1, 1)
            )
        else:
            opti.subject_to(
                cs.mtimes(self.dynamics_model["A"], cs.reshape(self.oc_s0_param, -1, 1))
                + cs.mtimes(self.dynamics_model["B"], cs.reshape(a0, -1, 1))
                + cs.reshape(self.dynamics_model["c"], -1, 1)
                <= cs.reshape(s1, -1, 1) + relaxation
            )
            opti.subject_to(
                cs.mtimes(self.dynamics_model["A"], cs.reshape(self.oc_s0_param, -1, 1))
                + cs.mtimes(self.dynamics_model["B"], cs.reshape(a0, -1, 1))
                + cs.reshape(self.dynamics_model["c"], -1, 1)
                >= cs.reshape(s1, -1, 1) - relaxation
            )

        # input saturation constraints
        if X_index_selector is None:
            opti.subject_to(
                X_var[-1, 0 : self.action_dim].T >= -np.ones(self.action_dim)
            )
            opti.subject_to(
                X_var[-1, 0 : self.action_dim].T <= np.ones(self.action_dim)
            )
            for i in range(self.horizon - 1):
                action_index = self.action_dim + i * (self.action_dim + self.state_dim)
                opti.subject_to(
                    X_var[-1, action_index : action_index + self.action_dim].T
                    >= -np.ones(self.action_dim)
                )
                opti.subject_to(
                    X_var[-1, action_index : action_index + self.action_dim].T
                    <= np.ones(self.action_dim)
                )
        elif X_index_selector == "projection":
            opti.subject_to(X_var[0 : self.action_dim] >= -np.ones(self.action_dim))
            opti.subject_to(X_var[0 : self.action_dim] <= np.ones(self.action_dim))
            for i in range(self.horizon - 1):
                action_index = self.action_dim + i * (self.action_dim + self.state_dim)
                opti.subject_to(
                    X_var[action_index : action_index + self.action_dim]
                    >= -np.ones(self.action_dim)
                )
                opti.subject_to(
                    X_var[action_index : action_index + self.action_dim]
                    <= np.ones(self.action_dim)
                )
        else:
            raise NotImplementedError

    def _generate_distance_objective(self, opti, X_var, X_index_selector=None):

        obs_norm_min_y = float(self.obs_norm_mins[3])
        obs_norm_max_y = float(self.obs_norm_maxs[3])

        y_pos_goal = 0.35  # self.goal_ypos in avoiding.py

        if X_index_selector is None:
            y_pos_terminal = X_var[
                -1,
                -1,
            ]
        elif X_index_selector == "projection":
            y_pos_terminal = X_var[-1]

        unnorm_y_pos_terminal = (y_pos_terminal + 1.0) / 2.0 * (
            obs_norm_max_y - obs_norm_min_y
        ) + obs_norm_min_y

        self.oc_distance_objective = -cs.sumsqr(unnorm_y_pos_terminal - y_pos_goal)

    def projection_formulate(
        self,
        print_level=2,
        constraint="",
    ):
        """
        Directly project the states to the feasible set.
        """

        assert (
            self.cfg.guidance_method == "projection"
        ), f"guidance_method must be projection, but got {self.cfg.guidance_method}"

        self.oc_cs_opti = cs.Opti()

        self.oc_N_steps = self.cfg.ode_t_steps
        self.oc_dof = self.horizon * (self.state_dim + self.action_dim) - self.state_dim

        self.oc_s0_param = self.oc_cs_opti.parameter(1, self.state_dim)

        self.oc_X_single = self.oc_cs_opti.variable(self.oc_dof)
        self.oc_X_single_ref = self.oc_cs_opti.parameter(self.oc_dof)

        self.oc_control_cost = 0.5 * cs.sumsqr(self.oc_X_single - self.oc_X_single_ref)

        if is_novel_constraint(constraint):
            self._apply_obstacle_constraints(
                self.oc_cs_opti, self.oc_X_single, X_index_selector="projection"
            )
            if self.cfg.dynamics_constraint:
                self._apply_dynamics_constraints(
                    self.oc_cs_opti,
                    self.oc_X_single,
                    X_index_selector="projection",
                )
        else:
            raise ValueError(f"Unsupported constraint: {constraint}")

        self.oc_cs_opti.minimize(self.oc_control_cost)

        solver_opts = {
            "ipopt.print_level": print_level,
            "ipopt.hessian_approximation": "limited-memory",
        }
        self.oc_cs_opti.solver("ipopt", solver_opts)

    def projection_relaxed_formulate(
        self,
        print_level=2,
        constraint="",
    ):
        """
        Inexact projection.
        """

        assert (
            self.cfg.guidance_method == "projection_relaxed"
        ), f"guidance_method must be projection_relaxed, but got {self.cfg.guidance_method}"

        self.oc_dof = self.horizon * (self.state_dim + self.action_dim) - self.state_dim
        self.oc_N_steps = self.cfg.ode_t_steps

        self._build_constraint_funs()

    def _build_constraint_funs(self):

        X = cs.MX.sym("X", self.oc_dof)
        s0 = cs.MX.sym("s0", self.state_dim)

        ineq_exprs = []
        eq_exprs = []

        obs_norm_min_x = float(self.obs_norm_mins[2])
        obs_norm_max_x = float(self.obs_norm_maxs[2])
        obs_norm_min_y = float(self.obs_norm_mins[3])
        obs_norm_max_y = float(self.obs_norm_maxs[3])

        def unnorm_x(x_pos):
            return (x_pos + 1.0) / 2.0 * (
                obs_norm_max_x - obs_norm_min_x
            ) + obs_norm_min_x

        def unnorm_y(y_pos):
            return (y_pos + 1.0) / 2.0 * (
                obs_norm_max_y - obs_norm_min_y
            ) + obs_norm_min_y

        for i in range(self.horizon - 1):
            x_pos = X[2 * self.action_dim + i * (self.action_dim + self.state_dim) + 2]
            y_pos = X[2 * self.action_dim + i * (self.action_dim + self.state_dim) + 3]

            x_u = unnorm_x(x_pos)
            y_u = unnorm_y(y_pos)

            for (cx, cy), r in zip(PILLAR_CENTERS, PILLAR_RADII):
                dist = cs.sqrt((x_u - cx) ** 2 + (y_u - cy) ** 2)
                ineq_exprs.append((r + self.cfg.obstacle_margin) - dist)

            if is_novel_constraint(self.cfg.constraint):
                cx, cy = NOVEL_OBSTACLE_CENTER
                dist = cs.sqrt((x_u - cx) ** 2 + (y_u - cy) ** 2)
                ineq_exprs.append(
                    (NOVEL_OBSTACLE_RADIUS + self.cfg.obstacle_margin) - dist
                )

                for slope, intercept, direction in NOVEL_BOUNDARY_LINES:
                    if direction == "below":
                        ineq_exprs.append(
                            self.cfg.obstacle_margin - (slope * x_u + intercept - y_u)
                        )
                    else:
                        raise ValueError(f"Unsupported boundary direction: {direction}")

                if uses_hard_quadrilateral(self.cfg.constraint):
                    ineq_exprs.append(
                        self.cfg.obstacle_margin
                        - self._quadrilateral_constraint_expr(x_u, y_u)
                    )
            else:
                raise ValueError(f"Unsupported constraint: {self.cfg.constraint}")

        if self.cfg.dynamics_constraint:
            A = self.dynamics_model["A"]
            B = self.dynamics_model["B"]
            c = self.dynamics_model["c"]

            for i in range(self.horizon - 2):
                s_idx = 2 * self.action_dim + i * (self.action_dim + self.state_dim)
                s_i = X[s_idx : s_idx + self.state_dim]
                a_idx = s_idx - self.action_dim
                a_i = X[a_idx : a_idx + self.action_dim]
                sn_idx = s_idx + (self.action_dim + self.state_dim)
                s_ip1 = X[sn_idx : sn_idx + self.state_dim]

                e = (
                    cs.mtimes(A, cs.reshape(s_i, -1, 1))
                    + cs.mtimes(B, cs.reshape(a_i, -1, 1))
                    + cs.reshape(c, -1, 1)
                    - cs.reshape(s_ip1, -1, 1)
                )
                eq_exprs.append(cs.reshape(e, (-1, 1)))

            s1_idx = 2 * self.action_dim
            s1 = X[s1_idx : s1_idx + self.state_dim]
            a0 = X[0 : self.action_dim]
            e0 = (
                cs.mtimes(A, cs.reshape(s0, -1, 1))
                + cs.mtimes(B, cs.reshape(a0, -1, 1))
                + cs.reshape(c, -1, 1)
                - cs.reshape(s1, -1, 1)
            )
            eq_exprs.append(cs.reshape(e0, (-1, 1)))

        if self.cfg.dynamics_constraint:
            a0 = X[0 : self.action_dim]
            ineq_exprs.extend([a0 - 1.0, -a0 - 1.0])
            for i in range(self.horizon - 1):
                a_idx = self.action_dim + i * (self.action_dim + self.state_dim)
                a_i = X[a_idx : a_idx + self.action_dim]
                ineq_exprs.extend([a_i - 1.0, -a_i - 1.0])

        if len(ineq_exprs) > 0:
            c_expr = cs.vertcat(*[cs.reshape(v, (-1, 1)) for v in ineq_exprs])
            Jc_expr = cs.jacobian(c_expr, X)
        else:
            c_expr = cs.MX.zeros(0, 1)
            Jc_expr = cs.MX.zeros(0, X.shape[0])

        if len(eq_exprs) > 0:
            h_expr = cs.vertcat(*eq_exprs)
            Jh_expr = cs.jacobian(h_expr, X)
        else:
            h_expr = cs.MX.zeros(0, 1)
            Jh_expr = cs.MX.zeros(0, X.shape[0])

        self._al_ineq_fun = cs.Function("al_ineq_fun", [X, s0], [c_expr])
        self._al_eq_fun = cs.Function("al_eq_fun", [X, s0], [h_expr])
        self._al_Jc_fun = cs.Function("al_Jc_fun", [X, s0], [Jc_expr])
        self._al_Jh_fun = cs.Function("al_Jh_fun", [X, s0], [Jh_expr])

        z0 = cs.DM.zeros(self.oc_dof, 1)
        s0z = cs.DM.zeros(self.state_dim, 1)
        self._al_m_ineq = int(np.array(self._al_ineq_fun(z0, s0z)).reshape(-1).shape[0])
        self._al_m_eq = int(np.array(self._al_eq_fun(z0, s0z)).reshape(-1).shape[0])

    def hardflow_formulate(
        self,
        print_level=2,
        constraint="",
        objective="",
    ):
        """
        Operate on the predicted terminal state.
        """

        assert self.cfg.guidance_method in (
            "hardflow",
            "hardflow_new",
        ), f"guidance_method must be hardflow or hardflow_new, but got {self.cfg.guidance_method}"

        self.oc_cs_opti = cs.Opti()

        self.oc_N_steps = self.cfg.ode_t_steps
        self.oc_dof = self.horizon * (self.state_dim + self.action_dim) - self.state_dim

        self.oc_s0_param = self.oc_cs_opti.parameter(1, self.state_dim)

        self.oc_t_param = self.oc_cs_opti.parameter(1)

        self.oc_X_terminal_predicted_ref = self.oc_cs_opti.parameter(self.oc_dof)
        self.oc_X_terminal_predicted = self.oc_cs_opti.variable(self.oc_dof)

        self.oc_control_cost = (
            0.5
            * self.cfg.hardflow_reg_scale
            * cs.sumsqr(self.oc_X_terminal_predicted - self.oc_X_terminal_predicted_ref)
            * self.oc_t_param**2
        )

        if is_novel_constraint(constraint):
            self._apply_obstacle_constraints(
                self.oc_cs_opti,
                self.oc_X_terminal_predicted,
                X_index_selector="projection",
            )
            if self.cfg.dynamics_constraint:
                self._apply_dynamics_constraints(
                    self.oc_cs_opti,
                    self.oc_X_terminal_predicted,
                    X_index_selector="projection",
                )
        else:
            raise ValueError(f"Unsupported constraint: {constraint}")

        if objective == "":
            self.oc_cs_opti.minimize(self.oc_control_cost)
        elif objective == "distance":
            self._generate_distance_objective(
                self.oc_cs_opti,
                self.oc_X_terminal_predicted,
                X_index_selector="projection",
            )
            self.oc_cs_opti.minimize(
                self.oc_control_cost
                - self.oc_distance_objective * self.cfg.hardflow_cost_scale
            )
        else:
            raise ValueError(f"Unsupported objective: {objective}")

        solver_opts = {
            "ipopt.print_level": print_level,
            "ipopt.hessian_approximation": "limited-memory",
        }
        self.oc_cs_opti.solver("ipopt", solver_opts)

    def warmstart(self, conditions):

        x = torch.randn(
            self.cfg.warmstart_batch,
            self.horizon,
            self.action_dim + self.state_dim,
            device=self.cfg.device,
        )

        x = apply_conditioning(
            x, to_torch(conditions, device=x.device), self.action_dim
        )
        x_chain = [x.clone()]
        dt = 1.0 / self.oc_N_steps
        for k in range(self.oc_N_steps):
            t = k / self.oc_N_steps
            dx_dt = self.flow_model(x, to_torch(t, device=x.device))
            dx_dt = apply_conditioning_from_conditioned_x(
                dx_dt, torch.zeros_like(x), conditions, self.action_dim
            )
            x = x + dx_dt * dt
            x_chain.append(x)

        values = self.value_model(x_chain[-1])
        best_idx = torch.argmax(values, dim=0)

        best_x_chain = torch.stack(
            [x_step[best_idx].squeeze(0) for x_step in x_chain], dim=0
        )  # (oc_N_steps+1, planning_horizon, transition_dim)
        best_x_chain = best_x_chain.reshape(
            self.oc_N_steps + 1, -1
        )  # (oc_N_steps+1, planning_horizon * transition_dim)
        best_s0 = best_x_chain[0, self.action_dim : self.state_dim + self.action_dim]
        best_dof_chain_part_1 = best_x_chain[:, : self.action_dim]
        best_dof_chain_part_2 = best_x_chain[:, self.action_dim + self.state_dim :]
        best_dof_chain = torch.cat(
            [best_dof_chain_part_1, best_dof_chain_part_2], dim=-1
        )

        best_s0_np = to_np(best_s0)
        best_dof_chain_np = to_np(best_dof_chain)

        return best_s0_np, best_dof_chain_np

    def projection_forward(self, conditions, batch_size=1):
        assert batch_size == 1, "batch_size must be 1 for optimal control"
        assert (
            self.cfg.guidance_method == "projection"
        ), f"guidance_method must be projection, but got {self.cfg.guidance_method}"

        conditions = utils.apply_dict(
            self.normalizer.normalize, conditions, "observations"
        )

        best_s0_np, best_dof_chain_np = self.warmstart(conditions)
        best_final_dof = best_dof_chain_np[-1, :]

        t_start = time.time()

        self.oc_cs_opti.set_value(self.oc_s0_param, best_s0_np)

        X_optimized = [best_dof_chain_np[0, :]]
        U_optimized = []
        dt = 1.0 / self.oc_N_steps
        for k in range(self.oc_N_steps):
            t_k = k * dt
            x_k = X_optimized[k]

            x_k_full = np.concatenate(
                [
                    x_k[: self.action_dim],
                    best_s0_np,
                    x_k[self.action_dim :],
                ],
                axis=0,
            )
            x_k_full = np.reshape(
                x_k_full,
                (1, self.horizon, self.action_dim + self.state_dim),
            )
            x = to_torch(x_k_full, device=self.cfg.device)

            if self.value_model is not None and self.cfg.projection_gradient_steps > 0:
                for _ in range(self.cfg.projection_gradient_steps):
                    inputs = x.detach().clone().requires_grad_(True)
                    vt = self.flow_model(x, to_torch(t_k))
                    vt = apply_conditioning_from_conditioned_x(
                        vt, torch.zeros_like(x), conditions, self.action_dim
                    )
                    loss = -self.value_model(inputs + vt * (1.0 - to_torch(t_k)))
                    if loss.ndim > 0:
                        loss = loss.mean()
                    (g,) = torch.autograd.grad(loss, inputs, create_graph=False)

                    g = apply_conditioning_from_conditioned_x(
                        g.detach(), torch.zeros_like(x), conditions, self.action_dim
                    )

                    x = x - self.cfg.projection_gradient_lr * g

            x_k_full = to_np(x.cpu()).reshape(-1)
            x_k = np.concatenate(
                [
                    x_k_full[: self.action_dim],
                    x_k_full[self.action_dim + self.state_dim :],
                ],
            )

            v_k = self.constrained_flow_fn(x_k, t_k, best_s0_np)
            x_next_ref = x_k + np.array(v_k).squeeze() * dt

            projection_flag = True
            if self.cfg.projection_option == "all":
                pass
            elif self.cfg.projection_option == "late":
                if k < self.oc_N_steps // 2:
                    projection_flag = False
            else:
                raise ValueError(
                    f"Unsupported projection_option: {self.cfg.projection_option}"
                )

            if projection_flag:
                self.oc_cs_opti.set_value(self.oc_X_single_ref, x_next_ref)
                self.oc_cs_opti.set_initial(self.oc_X_single, x_next_ref)
                try:
                    sol = self.oc_cs_opti.solve_limited()
                    x_next = sol.value(self.oc_X_single)
                except RuntimeError as e:
                    print("Solver failed, returning last available value.")
                    x_next = self.oc_cs_opti.debug.value(self.oc_X_single)
            else:
                x_next = x_next_ref

            x_next = np.array(x_next).flatten()

            u_k = (x_next - x_next_ref) / dt
            U_optimized.append(u_k)

            X_optimized.append(x_next)

        optimized_final_dof = X_optimized[-1]

        optimized_dof = np.array(X_optimized)
        optimized_x_chain = np.concatenate(
            [
                optimized_dof[:, : self.action_dim],
                np.tile(best_s0_np, (self.oc_N_steps + 1, 1)),
                optimized_dof[:, self.action_dim :],
            ],
            axis=1,
        )
        optimized_x_chain = np.reshape(
            optimized_x_chain,
            (
                batch_size,
                self.oc_N_steps + 1,
                self.horizon,
                self.state_dim + self.action_dim,
            ),
        )

        print()
        print(
            f"    Norm of Control Inputs: {np.linalg.norm(np.array(U_optimized)):.6f}",
            flush=True,
        )
        print()

        optimized_final_x = np.insert(optimized_final_dof, self.action_dim, best_s0_np)
        optimized_final_x = to_torch(optimized_final_x, device="cpu")
        optimized_final_x = optimized_final_x.reshape(
            1, self.horizon, self.action_dim + self.state_dim
        )

        self.value_model.previous_sample = to_np(optimized_final_x.cpu())

        normed_actions = optimized_final_x[:, :, : self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, "actions")
        actions = to_np(actions)
        normed_observations = optimized_final_x[:, :, self.action_dim :]
        observations = self.normalizer.unnormalize(normed_observations, "observations")
        observations = to_np(observations)
        values = self.value_model(
            torch.cat((normed_actions, normed_observations), dim=-1)
        )
        values = to_np(values)
        trajectories = Trajectories(actions, observations, values)

        action = actions[0, 0]

        x_chain = to_torch(optimized_x_chain, device=self.cfg.device)
        x1_estimation = self.x1_estimate(x_chain, conditions)
        x1_estimation = self.unnormalize_chain(x1_estimation)

        x_chain = self.unnormalize_chain(x_chain)

        t_end = time.time()
        computation_time = t_end - t_start

        info = {
            "computation_time": computation_time,
        }
        return action, trajectories, x_chain, x1_estimation, info

    def projection_relaxed_forward(self, conditions, batch_size=1):
        assert batch_size == 1, "batch_size must be 1 for optimal control"
        assert (
            self.cfg.guidance_method == "projection_relaxed"
        ), f"guidance_method must be projection_relaxed, but got {self.cfg.guidance_method}"

        conditions = utils.apply_dict(
            self.normalizer.normalize, conditions, "observations"
        )

        best_s0_np, best_dof_chain_np = self.warmstart(conditions)
        best_final_dof = best_dof_chain_np[-1, :]

        t_start = time.time()

        X_optimized = [best_dof_chain_np[0, :]]
        U_optimized = []
        dt = 1.0 / self.oc_N_steps

        for k in range(self.oc_N_steps):
            t_k = k * dt
            x_k = X_optimized[k]

            x_k_full = np.concatenate(
                [
                    x_k[: self.action_dim],
                    best_s0_np,
                    x_k[self.action_dim :],
                ],
                axis=0,
            )
            x_k_full = np.reshape(
                x_k_full,
                (1, self.horizon, self.action_dim + self.state_dim),
            )
            x = to_torch(x_k_full, device=self.cfg.device)

            if self.value_model is not None and self.cfg.projection_gradient_steps > 0:
                for _ in range(self.cfg.projection_gradient_steps):
                    inputs = x.detach().clone().requires_grad_(True)
                    vt = self.flow_model(x, to_torch(t_k))
                    vt = apply_conditioning_from_conditioned_x(
                        vt, torch.zeros_like(x), conditions, self.action_dim
                    )
                    loss = -self.value_model(inputs + vt * (1.0 - to_torch(t_k)))
                    if loss.ndim > 0:
                        loss = loss.mean()
                    (g,) = torch.autograd.grad(loss, inputs, create_graph=False)
                    x = x - self.cfg.projection_gradient_lr * g.detach()

            x_k_full = to_np(x.cpu()).reshape(-1)
            x_k = np.concatenate(
                [
                    x_k_full[: self.action_dim],
                    x_k_full[self.action_dim + self.state_dim :],
                ],
            )

            v_k = self.constrained_flow_fn(x_k, t_k, best_s0_np)
            x_next_ref = x_k + np.array(v_k).squeeze() * dt

            # augmented Lagrangian
            al_steps = int(getattr(self.cfg, "oc_inexact_al_steps", 20))
            al_lr = float(getattr(self.cfg, "oc_inexact_al_lr", 0.1))

            x_var = np.array(x_next_ref, dtype=np.float32).reshape(-1)
            s0_val = np.array(best_s0_np, dtype=np.float32).reshape(-1)

            m_ineq = getattr(self, "_al_m_ineq", 0)
            m_eq = getattr(self, "_al_m_eq", 0)

            if k == 0:
                lam = np.zeros(m_ineq, dtype=np.float32)
                nu = np.zeros(m_eq, dtype=np.float32)
                al_rho = float(getattr(self.cfg, "oc_inexact_al_rho", 5.0))

            for _ in range(al_steps):
                c_val = (
                    np.array(self._al_ineq_fun(x_var, s0_val)).reshape(-1)
                    if m_ineq > 0
                    else np.zeros(0)
                )
                h_val = (
                    np.array(self._al_eq_fun(x_var, s0_val)).reshape(-1)
                    if m_eq > 0
                    else np.zeros(0)
                )

                Jc = (
                    np.array(self._al_Jc_fun(x_var, s0_val))
                    if m_ineq > 0
                    else np.zeros((0, x_var.size))
                )
                Jh = (
                    np.array(self._al_Jh_fun(x_var, s0_val))
                    if m_eq > 0
                    else np.zeros((0, x_var.size))
                )

                grad = (x_var - x_next_ref).astype(np.float32)

                if m_eq > 0:
                    grad += Jh.T @ (nu + al_rho * h_val)

                if m_ineq > 0:
                    r = c_val + lam / al_rho
                    r_pos = np.maximum(r, 0.0)
                    grad += al_rho * (Jc.T @ r_pos)

                x_new = x_var - al_lr * grad

                if m_eq > 0:
                    nu = nu + al_rho * h_val
                if m_ineq > 0:
                    lam = np.maximum(0.0, lam + al_rho * c_val)

                x_var = x_new

            x_next = x_var

            u_k = (x_next - x_next_ref) / dt
            U_optimized.append(u_k)

            X_optimized.append(x_next)

        optimized_final_dof = X_optimized[-1]

        optimized_dof = np.array(X_optimized)
        optimized_x_chain = np.concatenate(
            [
                optimized_dof[:, : self.action_dim],
                np.tile(best_s0_np, (self.oc_N_steps + 1, 1)),
                optimized_dof[:, self.action_dim :],
            ],
            axis=1,
        )
        optimized_x_chain = np.reshape(
            optimized_x_chain,
            (
                batch_size,
                self.oc_N_steps + 1,
                self.horizon,
                self.state_dim + self.action_dim,
            ),
        )

        print()
        print(
            f"    Norm of Control Inputs: {np.linalg.norm(np.array(U_optimized)):.6f}",
            flush=True,
        )
        print()

        optimized_final_x = np.insert(optimized_final_dof, self.action_dim, best_s0_np)
        optimized_final_x = to_torch(optimized_final_x, device="cpu")
        optimized_final_x = optimized_final_x.reshape(
            1, self.horizon, self.action_dim + self.state_dim
        )

        self.value_model.previous_sample = to_np(optimized_final_x.cpu())

        normed_actions = optimized_final_x[:, :, : self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, "actions")
        actions = to_np(actions)
        normed_observations = optimized_final_x[:, :, self.action_dim :]
        observations = self.normalizer.unnormalize(normed_observations, "observations")
        observations = to_np(observations)
        values = self.value_model(
            torch.cat((normed_actions, normed_observations), dim=-1)
        )
        values = to_np(values)
        trajectories = Trajectories(actions, observations, values)

        action = actions[0, 0]

        x_chain = to_torch(optimized_x_chain, device=self.cfg.device)
        x1_estimation = self.x1_estimate(x_chain, conditions)
        x1_estimation = self.unnormalize_chain(x1_estimation)

        x_chain = self.unnormalize_chain(x_chain)

        t_end = time.time()
        computation_time = t_end - t_start

        info = {
            "computation_time": computation_time,
        }
        return action, trajectories, x_chain, x1_estimation, info

    def hardflow_forward(self, conditions, batch_size=1):
        assert batch_size == 1, "batch_size must be 1 for optimal control"
        assert (
            self.cfg.guidance_method == "hardflow"
        ), f"guidance_method must be hardflow, but got {self.cfg.guidance_method}"

        conditions = utils.apply_dict(
            self.normalizer.normalize, conditions, "observations"
        )

        best_s0_np, best_dof_chain_np = self.warmstart(conditions)
        best_final_dof = best_dof_chain_np[-1, :]

        t_start = time.time()

        self.oc_cs_opti.set_value(self.oc_s0_param, best_s0_np)

        X_optimized = [best_dof_chain_np[0, :]]
        U_optimized = []
        dt = 1.0 / self.oc_N_steps
        for k in range(self.oc_N_steps):
            t_k = k * dt
            x_k = X_optimized[k]
            v_k = self.constrained_flow_fn(x_k, t_k, best_s0_np)
            x_next_ref = x_k + np.array(v_k).squeeze() * dt

            control_flag = True
            if self.cfg.hardflow_activation == "all":
                pass
            elif self.cfg.hardflow_activation == "late":
                if k < self.oc_N_steps // 2:
                    control_flag = False
            else:
                raise ValueError(
                    f"Unsupported hardflow_activation: {self.cfg.hardflow_activation}"
                )

            if control_flag:
                x_terminal_predicted_ref = (
                    x_next_ref
                    + (1.0 - t_k - dt)
                    * self.constrained_flow_fn(x_next_ref, t_k + dt, best_s0_np).T
                )

                self.oc_cs_opti.set_value(self.oc_t_param, t_k + dt)

                self.oc_cs_opti.set_value(
                    self.oc_X_terminal_predicted_ref, x_terminal_predicted_ref
                )
                self.oc_cs_opti.set_initial(
                    self.oc_X_terminal_predicted, x_terminal_predicted_ref
                )

                try:
                    sol = self.oc_cs_opti.solve_limited()
                    x_terminal_predicted = sol.value(self.oc_X_terminal_predicted)
                except RuntimeError as e:
                    print("Solver failed, returning last available value.")
                    x_terminal_predicted = self.oc_cs_opti.debug.value(
                        self.oc_X_terminal_predicted
                    )

                x_next = x_next_ref + (t_k + dt) * (
                    x_terminal_predicted - x_terminal_predicted_ref
                )
            else:
                x_next = x_next_ref

            x_next = np.array(x_next).flatten()

            u_k = (x_next - x_next_ref) / dt
            U_optimized.append(u_k)

            X_optimized.append(x_next)

        optimized_final_dof = X_optimized[-1]

        optimized_dof = np.array(X_optimized)
        optimized_x_chain = np.concatenate(
            [
                optimized_dof[:, : self.action_dim],
                np.tile(best_s0_np, (self.oc_N_steps + 1, 1)),
                optimized_dof[:, self.action_dim :],
            ],
            axis=1,
        )
        optimized_x_chain = np.reshape(
            optimized_x_chain,
            (
                batch_size,
                self.oc_N_steps + 1,
                self.horizon,
                self.state_dim + self.action_dim,
            ),
        )

        print()
        print(
            f"    Norm of Control Inputs: {np.linalg.norm(np.array(U_optimized)):.6f}",
            flush=True,
        )
        print()

        optimized_final_x = np.insert(optimized_final_dof, self.action_dim, best_s0_np)
        optimized_final_x = to_torch(optimized_final_x, device="cpu")
        optimized_final_x = optimized_final_x.reshape(
            1, self.horizon, self.action_dim + self.state_dim
        )

        self.value_model.previous_sample = to_np(optimized_final_x.cpu())

        normed_actions = optimized_final_x[:, :, : self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, "actions")
        actions = to_np(actions)
        normed_observations = optimized_final_x[:, :, self.action_dim :]
        observations = self.normalizer.unnormalize(normed_observations, "observations")
        observations = to_np(observations)
        values = self.value_model(
            torch.cat((normed_actions, normed_observations), dim=-1)
        )
        values = to_np(values)
        trajectories = Trajectories(actions, observations, values)

        action = actions[0, 0]

        x_chain = to_torch(optimized_x_chain, device=self.cfg.device)
        x1_estimation = self.x1_estimate(x_chain, conditions)
        x1_estimation = self.unnormalize_chain(x1_estimation)

        x_chain = self.unnormalize_chain(x_chain)

        t_end = time.time()
        computation_time = t_end - t_start

        info = {
            "computation_time": computation_time,
        }
        return action, trajectories, x_chain, x1_estimation, info

    def hardflow_new_forward(self, conditions, batch_size=1):
        """
        Variant of hardflow_forward that evaluates the reference flow with
        PyTorch directly instead of through the l4casadi bridge. The CasADi
        NLP solved at every ODE step is purely algebraic, so the optimization
        itself does not depend on the neural network.
        """
        assert batch_size == 1, "batch_size must be 1 for optimal control"
        assert (
            self.cfg.guidance_method == "hardflow_new"
        ), f"guidance_method must be hardflow_new, but got {self.cfg.guidance_method}"

        conditions = utils.apply_dict(
            self.normalizer.normalize, conditions, "observations"
        )

        best_s0_np, best_dof_chain_np = self.warmstart(conditions)
        best_final_dof = best_dof_chain_np[-1, :]

        t_start = time.time()

        device = self.cfg.device
        best_s0_torch = to_torch(best_s0_np, device=device).reshape(1, self.state_dim)

        def flow_eval_np(x_np, t):
            x_torch = to_torch(x_np, device=device).reshape(1, self.oc_dof)
            with torch.no_grad():
                v = self.constrained_flow_fn_torch(x_torch, t, best_s0_torch)
            return to_np(v).reshape(-1)

        self.oc_cs_opti.set_value(self.oc_s0_param, best_s0_np)

        X_optimized = [best_dof_chain_np[0, :]]
        U_optimized = []
        dt = 1.0 / self.oc_N_steps
        # Gen13 U10 (fix: DPCC polarity). threshold = fraction of the late trajectory
        # projected — SAME as DPCC's diffusion_timestep_threshold, higher = MORE
        # projection. NLP active when t_{k+1} >= (1 - threshold). If
        # hardflow_activation_threshold < 0 it is disabled and we fall back to the
        # binary hardflow_activation (all -> 1.0 = every step, late -> 0.5 = last half).
        _thr = getattr(self.cfg, "hardflow_activation_threshold", -1.0)
        if _thr is not None and _thr >= 0.0:
            activation_threshold = float(_thr)
        elif self.cfg.hardflow_activation == "all":
            activation_threshold = 1.0
        elif self.cfg.hardflow_activation == "late":
            activation_threshold = 0.5
        else:
            raise ValueError(
                f"Unsupported hardflow_activation: {self.cfg.hardflow_activation}"
            )

        for k in range(self.oc_N_steps):
            t_k = k * dt
            x_k = X_optimized[k]
            v_k = flow_eval_np(x_k, t_k)
            x_next_ref = x_k + v_k * dt

            # U10 + fix: EXACT DPCC gate. DPCC projects when `loop_idx >= (1 - T)*K`;
            # we use the identical `k >= (1 - threshold)*N` (threshold == DPCC's
            # diffusion_timestep_threshold, higher = more projection) PLUS the forced
            # final step — the terminal solve is what the safety guarantee rides on
            # (paper Prop.). `or k == N-1` makes threshold 0.0 => terminal-only.
            control_flag = (k >= (1.0 - activation_threshold) * self.oc_N_steps) or (k == self.oc_N_steps - 1)

            if control_flag:
                v_next = flow_eval_np(x_next_ref, t_k + dt)
                x_terminal_predicted_ref = x_next_ref + (1.0 - t_k - dt) * v_next

                self.oc_cs_opti.set_value(self.oc_t_param, t_k + dt)

                self.oc_cs_opti.set_value(
                    self.oc_X_terminal_predicted_ref, x_terminal_predicted_ref
                )
                self.oc_cs_opti.set_initial(
                    self.oc_X_terminal_predicted, x_terminal_predicted_ref
                )

                try:
                    sol = self.oc_cs_opti.solve_limited()
                    x_terminal_predicted = sol.value(self.oc_X_terminal_predicted)
                except RuntimeError as e:
                    print("Solver failed, returning last available value.")
                    x_terminal_predicted = self.oc_cs_opti.debug.value(
                        self.oc_X_terminal_predicted
                    )

                x_next = x_next_ref + (t_k + dt) * (
                    x_terminal_predicted - x_terminal_predicted_ref
                )
            else:
                x_next = x_next_ref

            x_next = np.array(x_next).flatten()

            u_k = (x_next - x_next_ref) / dt
            U_optimized.append(u_k)

            X_optimized.append(x_next)

        optimized_final_dof = X_optimized[-1]

        optimized_dof = np.array(X_optimized)
        optimized_x_chain = np.concatenate(
            [
                optimized_dof[:, : self.action_dim],
                np.tile(best_s0_np, (self.oc_N_steps + 1, 1)),
                optimized_dof[:, self.action_dim :],
            ],
            axis=1,
        )
        optimized_x_chain = np.reshape(
            optimized_x_chain,
            (
                batch_size,
                self.oc_N_steps + 1,
                self.horizon,
                self.state_dim + self.action_dim,
            ),
        )

        print()
        print(
            f"    Norm of Control Inputs: {np.linalg.norm(np.array(U_optimized)):.6f}",
            flush=True,
        )
        print()

        optimized_final_x = np.insert(optimized_final_dof, self.action_dim, best_s0_np)
        optimized_final_x = to_torch(optimized_final_x, device="cpu")
        optimized_final_x = optimized_final_x.reshape(
            1, self.horizon, self.action_dim + self.state_dim
        )

        self.value_model.previous_sample = to_np(optimized_final_x.cpu())

        normed_actions = optimized_final_x[:, :, : self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, "actions")
        actions = to_np(actions)
        normed_observations = optimized_final_x[:, :, self.action_dim :]
        observations = self.normalizer.unnormalize(normed_observations, "observations")
        observations = to_np(observations)
        values = self.value_model(
            torch.cat((normed_actions, normed_observations), dim=-1)
        )
        values = to_np(values)
        trajectories = Trajectories(actions, observations, values)

        action = actions[0, 0]

        x_chain = to_torch(optimized_x_chain, device=self.cfg.device)
        x1_estimation = self.x1_estimate(x_chain, conditions)
        x1_estimation = self.unnormalize_chain(x1_estimation)

        x_chain = self.unnormalize_chain(x_chain)

        t_end = time.time()
        computation_time = t_end - t_start

        info = {
            "computation_time": computation_time,
        }
        return action, trajectories, x_chain, x1_estimation, info

    def gradient_guidance_forward(self, conditions, batch_size=1):
        """
        Use gradient guidance to generate actions.
        """
        assert (
            self.cfg.guidance_method == "gradient_guidance"
        ), f"guidance_method must be gradient_guidance, but got {self.cfg.guidance_method}"

        conditions = utils.apply_dict(
            self.normalizer.normalize, conditions, "observations"
        )

        t_start = time.time()

        solver = ConditionedODESolver(
            self.flow_model,
            conditions,
            value=self.value_model,
            ode_method=self.cfg.ode_solver,
            action_dim=self.action_dim,
            guidance_lr=self.cfg.guidance_lr,
            guidance_steps=self.cfg.guidance_steps,
        )

        x = torch.randn(
            batch_size,
            self.horizon,
            self.action_dim + self.state_dim,
            device=self.cfg.device,
        )
        x = apply_conditioning(
            x, to_torch(conditions, device=x.device), self.action_dim
        )

        x, x_chain = solver(
            x,
            t_span=torch.linspace(
                *self.cfg.ode_t_span, self.cfg.ode_t_steps + 1, device=x.device
            )[:-1],
            return_chain=True,
        )

        normed_actions = x[:, :, : self.action_dim]
        actions = self.normalizer.unnormalize(to_np(normed_actions), "actions")

        normed_observations = x[:, :, self.action_dim :]
        observations = self.normalizer.unnormalize(
            to_np(normed_observations), "observations"
        )

        values = self.value_model(
            torch.cat((normed_actions, normed_observations), dim=-1)
        )  # (batch_size, 1)
        values = values.reshape(batch_size, 1)
        values = to_np(values)

        # sort the actions, observations, and values
        sorted_indices = values.argsort(axis=0)[::-1].flatten().copy()
        sorted_actions = actions[sorted_indices]
        sorted_observations = observations[sorted_indices]
        sorted_values = values[sorted_indices]

        trajectories = Trajectories(sorted_actions, sorted_observations, sorted_values)
        action = sorted_actions[0, 0]

        self.previous_sample = np.concatenate(
            [sorted_actions[0:1, :, :], sorted_observations[0:1, :, :]], axis=-1
        )

        x1_estimation = self.x1_estimate(x_chain, conditions)
        x1_estimation = self.unnormalize_chain(x1_estimation)

        x_chain = self.unnormalize_chain(x_chain)

        t_end = time.time()
        computation_time = t_end - t_start

        info = {"computation_time": computation_time}
        return action, trajectories, x_chain, x1_estimation, info

    def oc_flow_forward(
        self,
        conditions,
        batch_size=1,
    ):
        assert batch_size == 1, "batch_size must be 1 for optimal control"
        assert (
            self.cfg.guidance_method == "oc_flow"
        ), f"guidance_method must be oc_flow, but got {self.cfg.guidance_method}"

        device = self.cfg.device

        conditions = utils.apply_dict(
            self.normalizer.normalize, conditions, "observations"
        )

        self.oc_N_steps = self.cfg.ode_t_steps
        self.oc_dof = self.horizon * (self.state_dim + self.action_dim) - self.state_dim

        best_s0_np, best_dof_chain_np = self.warmstart(conditions)

        t_start = time.time()

        best_s0 = torch.tensor(best_s0_np, device=device, dtype=torch.float32)
        if best_s0.dim() == 1:
            best_s0 = best_s0.unsqueeze(0)
        best_dof_chain = torch.tensor(
            best_dof_chain_np, device=device, dtype=torch.float32
        )

        U = torch.zeros(
            self.oc_N_steps, self.oc_dof, device=device, dtype=torch.float32
        )

        def eval_terminal_cost(xN):
            x = torch.cat(
                (xN[: self.action_dim], best_s0.squeeze(), xN[self.action_dim :]),
                dim=-1,
            )
            x = x.reshape(1, self.horizon, self.action_dim + self.state_dim)

            cost = -self.value_model(x)
            return cost

        def rollout(U):
            assert U.shape == (self.oc_N_steps, self.oc_dof)
            X = []
            x = best_dof_chain[0].clone()
            X.append(x)
            dt = 1.0 / self.oc_N_steps

            for k in range(self.oc_N_steps):
                t_k = k * dt
                v_k = self.constrained_flow_fn_torch(
                    x.unsqueeze(0), t_k, best_s0
                ).squeeze(0)
                v_k = v_k + U[k]
                x = x + v_k * dt
                X.append(x)

            return torch.stack(X)

        def gradient_U(U, gamma_u=0.0):
            # forward sweep
            X = rollout(U)
            xN = X[-1].clone().detach().requires_grad_(True)

            # terminal adjoint
            terminal_cost = eval_terminal_cost(xN)
            lam = (
                torch.autograd.grad(terminal_cost, xN, retain_graph=False)[0]
                .detach()
                .clone()
            )

            # backward sweep
            N = self.oc_N_steps
            dt = 1.0 / N
            d = self.oc_dof
            grad = torch.zeros_like(U)

            for j in reversed(range(N)):
                lam_next = lam

                xj = X[j].clone().detach().requires_grad_(True)
                t_j = j * dt
                u_j = U[j].clone().detach()

                def F_j(x_flat):
                    x = x_flat.view_as(xj)
                    f = self.constrained_flow_fn_torch(
                        x.unsqueeze(0), t_j, best_s0
                    ).squeeze(0)
                    return (x + dt * (f + u_j)).view(-1)

                _, vjp = torch.autograd.functional.vjp(
                    F_j, xj.view(-1), v=lam_next.view(-1)
                )
                lam = vjp.view_as(xj).detach()

                g_j = dt * lam_next + (gamma_u * u_j if gamma_u > 0.0 else 0.0)
                grad[j] = g_j

            return grad, X

        for _ in range(self.cfg.oc_flow_steps):
            grad_U, X = gradient_U(U, gamma_u=0.005)
            U = U - self.cfg.oc_flow_lr * grad_U

        U_optimized = U
        X_optimized = rollout(U_optimized)
        U_optimized_np = U_optimized.detach().cpu().numpy()
        X_optimized_np = X_optimized.detach().cpu().numpy()

        optimized_final_dof = X_optimized_np[-1, :]

        optimized_dof = X_optimized_np

        optimized_x_chain = np.concatenate(
            [
                optimized_dof[:, : self.action_dim],
                np.tile(best_s0_np, (self.oc_N_steps + 1, 1)),
                optimized_dof[:, self.action_dim :],
            ],
            axis=1,
        )
        optimized_x_chain = np.reshape(
            optimized_x_chain,
            (
                batch_size,
                self.oc_N_steps + 1,
                self.horizon,
                self.state_dim + self.action_dim,
            ),
        )

        print()
        print(
            f"    Norm of Control Inputs: {np.linalg.norm(U_optimized_np):.6f}",
            flush=True,
        )
        print()

        optimized_final_x = np.insert(optimized_final_dof, self.action_dim, best_s0_np)
        optimized_final_x = to_torch(optimized_final_x, device="cpu")
        optimized_final_x = optimized_final_x.reshape(
            1, self.horizon, self.action_dim + self.state_dim
        )

        normed_actions = optimized_final_x[:, :, : self.action_dim]
        actions = self.normalizer.unnormalize(normed_actions, "actions")
        actions = to_np(actions)

        normed_observations = optimized_final_x[:, :, self.action_dim :]
        observations = self.normalizer.unnormalize(normed_observations, "observations")
        observations = to_np(observations)

        values = self.value_model(
            torch.cat((normed_actions, normed_observations), dim=-1)
        )
        values = to_np(values)

        trajectories = Trajectories(actions, observations, values)
        action = actions[0, 0]

        x_chain = to_torch(optimized_x_chain, device=device)
        x1_estimation = self.x1_estimate(x_chain, conditions)
        x1_estimation = self.unnormalize_chain(x1_estimation)
        x_chain = self.unnormalize_chain(x_chain)

        t_end = time.time()
        computation_time = t_end - t_start

        info = {
            "computation_time": computation_time,
        }
        return action, trajectories, x_chain, x1_estimation, info

    def constrained_flow_fn_torch(self, dof, t, s0):
        device = dof.device
        assert dof.shape == (1, self.oc_dof)

        full_flat = torch.cat(
            [dof[:, : self.action_dim], s0, dof[:, self.action_dim :]], dim=1
        )
        full_traj = full_flat.view(1, self.horizon, self.action_dim + self.state_dim)

        if isinstance(t, (float, int)):
            t = torch.tensor(t, device=device, dtype=full_traj.dtype)
        if t.dim() == 0:
            t = t.unsqueeze(0)

        flow_output = self.flow_model(full_traj, t)
        flow_output_flat = flow_output.reshape(1, -1)

        dof_flow = torch.cat(
            [
                flow_output_flat[:, : self.action_dim],
                flow_output_flat[:, self.action_dim + self.state_dim :],
            ],
            dim=1,
        )
        return dof_flow
