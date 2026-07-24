import os
import csv

import matplotlib

matplotlib.use("Agg")

import hardflow.utils
import tqdm
import tyro
import gym

import numpy as np
import torch
from torch import nn

from hardflow.config.flow_matching import FlowMatchingEvaluationConfig
from hardflow.datasets.sequence import SequenceDataset
from hardflow.models_flow.flow_policy import FlowPolicy
from hardflow.models_flow.unet import TemporalUnet, WrappedFlowUnet
from hardflow.utils.avoiding_geometry import (
    NOVEL_BOUNDARY_LINES,
    NOVEL_HARD_QUADRILATERAL_EDGES,
    NOVEL_OBSTACLE_CENTER,
    NOVEL_OBSTACLE_RADIUS,
    PILLAR_CENTERS,
    PILLAR_RADII,
    is_novel_constraint,
    quadrilateral_constraint_value_numpy,
    uses_hard_quadrilateral,
)
from run.utils import deterministic, save_config, set_cuda_visible_device

import d3il

try:
    import casadi as cs
    import l4casadi as l4c
except ImportError:
    cs = None
    l4c = None


class ProxyValueModel(nn.Module):

    def __init__(
        self,
        horizon: int,
        action_dim: int,
        state_dim: int,
        objective: str = "",
        constraint: str = "",
        obstacle_margin: float = 0.0,
        value_objective_scale: float = 1.0,
        value_constraint_scale: float = 1.0,
        dynamics_constraint: bool = False,
        normalizer=None,
        dynamics_model=None,
    ):

        super().__init__()

        self.horizon = horizon
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.transition_dim = action_dim + state_dim

        self.objective = objective
        self.constraint = constraint
        self.obstacle_margin = obstacle_margin
        self.value_objective_scale = value_objective_scale
        self.value_constraint_scale = value_constraint_scale
        self.dynamics_constraint = dynamics_constraint
        self.normalizer = normalizer
        self.dynamics_model = dynamics_model

        self.previous_sample = None

        if normalizer is not None:
            self.obs_norm_mins = normalizer.normalizers["observations"].mins
            self.obs_norm_maxs = normalizer.normalizers["observations"].maxs

    def forward(
        self,
        x,
    ):
        """
        x: (batch_size, planning_horizon, transition_dim), should be normalized
        value: (batch_size, 1)
        """

        if self.objective == "":
            objective_value = torch.zeros(
                x.shape[0], 1, device=x.device, requires_grad=True
            )
        elif self.objective == "distance":
            objective_value = self._compute_distance_objective(x)
        elif self.objective == "consistency":
            objective_value = self._compute_consistency_objective(x)
        else:
            raise NotImplementedError(f"Objective {self.objective} is not implemented.")

        if self.constraint == "":
            constraint_penalty = torch.zeros(
                x.shape[0], 1, device=x.device, requires_grad=True
            )
        else:
            constraint_values = self._compute_obstacle_constraints(x)
            constraint_violations = torch.clamp(-constraint_values, min=0.0)
            constraint_penalty = torch.sum(
                constraint_violations**2, dim=1, keepdim=True
            )
            if self.dynamics_constraint:
                dynamics_constraint_values = self._compute_dynamics_constraints(x)
                dynamics_penalty = torch.sum(
                    dynamics_constraint_values**2, dim=1, keepdim=True
                )
                constraint_penalty += dynamics_penalty

        value = (
            objective_value * self.value_objective_scale
            - constraint_penalty * self.value_constraint_scale
        )

        return value

    def _compute_consistency_objective(self, x):
        """
        x: (batch_size, planning_horizon, transition_dim)
        previous_sample: (1, planning_horizon, transition_dim)
        objective_value: (batch_size, 1)
        """

        if self.previous_sample is None:
            consistency_objective = torch.zeros(
                x.shape[0], 1, device=x.device, requires_grad=True
            )
        else:
            previous_sample_tensor = torch.tensor(
                self.previous_sample, device=x.device, dtype=x.dtype
            )
            consistency_objective = (
                -((x - previous_sample_tensor) ** 2).sum(dim=(1, 2)).unsqueeze(-1)
            )

        return consistency_objective

    def _compute_distance_objective(self, x):
        """
        x: (batch_size, planning_horizon, transition_dim)
        objective_value: (batch_size, 1)
        """

        obs_norm_min_y = float(self.obs_norm_mins[3])
        obs_norm_max_y = float(self.obs_norm_maxs[3])

        y_pos_goal = 0.35  # self.goal_ypos in avoiding.py

        y_pos_terminal = x[:, -1, -1]
        unnorm_y_pos_terminal = (y_pos_terminal + 1.0) / 2.0 * (
            obs_norm_max_y - obs_norm_min_y
        ) + obs_norm_min_y

        distance_objective = -torch.square(
            unnorm_y_pos_terminal - y_pos_goal
        ).unsqueeze(1)

        return distance_objective

    def _compute_dynamics_constraints(self, x):
        """
        x: (batch_size, planning_horizon, transition_dim)
        constraint_values: (batch_size, num_constraints), = 0 means feasible
        """
        if self.dynamics_model is None:
            return torch.zeros(x.shape[0], 0, device=x.device, requires_grad=True)

        batch_size = x.shape[0]
        actions = x[:, :, : self.action_dim]
        observations = x[:, :, self.action_dim :]

        A_torch = torch.tensor(self.dynamics_model["A"], device=x.device, dtype=x.dtype)
        B_torch = torch.tensor(self.dynamics_model["B"], device=x.device, dtype=x.dtype)
        c_torch = torch.tensor(self.dynamics_model["c"], device=x.device, dtype=x.dtype)

        constraint_values = []

        for i in range(self.horizon - 3):
            current_state = observations[:, i, :]
            next_state = observations[:, i + 1, :]
            current_action = actions[:, i, :]

            predicted_next_state = (
                torch.matmul(current_state, A_torch.T)
                + torch.matmul(current_action, B_torch.T)
                + c_torch.unsqueeze(0)
            )

            dynamics_constraint = predicted_next_state - next_state
            constraint_values.append(dynamics_constraint)

        if len(constraint_values) == 0:
            return torch.zeros(batch_size, 0, device=x.device, requires_grad=True)

        constraint_values = torch.cat(constraint_values, dim=1)
        return constraint_values

    def _compute_obstacle_constraints(self, x):
        """
        x: (batch_size, planning_horizon, transition_dim)
        constraint_values: (batch_size, num_constraints), >= 0 means feasible
        """

        obs_norm_min_x = float(self.obs_norm_mins[2])
        obs_norm_max_x = float(self.obs_norm_maxs[2])
        obs_norm_min_y = float(self.obs_norm_mins[3])
        obs_norm_max_y = float(self.obs_norm_maxs[3])

        batch_size = x.shape[0]
        observations = x[:, :, self.action_dim :]

        x_pos = observations[:, :, 2]
        y_pos = observations[:, :, 3]

        unnorm_x_pos = (x_pos + 1.0) / 2.0 * (
            obs_norm_max_x - obs_norm_min_x
        ) + obs_norm_min_x
        unnorm_y_pos = (y_pos + 1.0) / 2.0 * (
            obs_norm_max_y - obs_norm_min_y
        ) + obs_norm_min_y

        constraint_values_list = []

        obstacle_margin = self.obstacle_margin

        # original constraints that are always applied
        for center, radius in zip(PILLAR_CENTERS, PILLAR_RADII):
            for t in range(self.horizon):
                distance = torch.sqrt(
                    (unnorm_x_pos[:, t] - center[0]) ** 2
                    + (unnorm_y_pos[:, t] - center[1]) ** 2
                )
                pillar_constraint = distance - radius - obstacle_margin
                constraint_values_list.append(pillar_constraint.unsqueeze(1))

        if is_novel_constraint(self.constraint):
            for t in range(self.horizon):
                distance = torch.sqrt(
                    (unnorm_x_pos[:, t] - NOVEL_OBSTACLE_CENTER[0]) ** 2
                    + (unnorm_y_pos[:, t] - NOVEL_OBSTACLE_CENTER[1]) ** 2
                )
                obstacle_constraint = distance - NOVEL_OBSTACLE_RADIUS - obstacle_margin
                constraint_values_list.append(obstacle_constraint.unsqueeze(1))

            for slope, intercept, direction in NOVEL_BOUNDARY_LINES:
                for t in range(self.horizon):
                    if direction == "below":
                        boundary_constraint = (
                            slope * unnorm_x_pos[:, t]
                            + intercept
                            - unnorm_y_pos[:, t]
                            - obstacle_margin
                        )
                    else:
                        raise ValueError(f"Unsupported boundary direction: {direction}")
                    constraint_values_list.append(boundary_constraint.unsqueeze(1))

            if uses_hard_quadrilateral(self.constraint):
                for t in range(self.horizon):
                    quadrilateral_constraint = self._compute_quadrilateral_constraint(
                        unnorm_x_pos[:, t], unnorm_y_pos[:, t], obstacle_margin
                    )
                    constraint_values_list.append(quadrilateral_constraint.unsqueeze(1))
        else:
            raise NotImplementedError(
                f"Constraint {self.constraint} is not implemented."
            )

        if len(constraint_values_list) == 0:
            return torch.zeros(batch_size, 0, device=x.device, requires_grad=True)

        constraint_values = torch.cat(constraint_values_list, dim=1)
        return constraint_values

    def _compute_quadrilateral_constraint(self, x_pos, y_pos, obstacle_margin):
        signed_distances = []
        for vertex, next_vertex in NOVEL_HARD_QUADRILATERAL_EDGES:
            edge = next_vertex - vertex
            edge_norm = float(np.linalg.norm(edge))
            signed_distance = (
                edge[0] * (y_pos - vertex[1]) - edge[1] * (x_pos - vertex[0])
            ) / edge_norm
            signed_distances.append(signed_distance)

        quadrilateral_constraint = torch.max(
            torch.stack(signed_distances, dim=1), dim=1
        )
        return quadrilateral_constraint.values - obstacle_margin


def create_constrained_casadi_functions(
    casadi_flow_fn,
    action_dim: int,
    state_dim: int,
    horizon: int,
):

    print(
        f"create_constrained_casadi_function: horizon={horizon}, action_dim={action_dim}, state_dim={state_dim}"
    )

    transition_dim = action_dim + state_dim
    N_full = horizon * transition_dim
    dof = N_full - state_dim
    print(
        f"Calculated dimensions: transition_dim={transition_dim}, N_full={N_full}, dof={dof}"
    )

    dof_sym = cs.MX.sym("dof_sym", 1, dof)
    s0_sym = cs.MX.sym("s0_sym", 1, state_dim)

    full_traj_sym = cs.horzcat(
        dof_sym[:, :action_dim],
        s0_sym,
        dof_sym[:, action_dim:],
    )
    print(f"full_traj_sym shape: {full_traj_sym.shape}")

    time_sym = cs.MX.sym("t")
    flow_input_sym = cs.horzcat(full_traj_sym, time_sym)

    full_flow_output_sym = casadi_flow_fn(flow_input_sym)

    dof_flow_output_sym = cs.horzcat(
        full_flow_output_sym[:, :action_dim],
        full_flow_output_sym[:, action_dim + state_dim :],
    )

    constrained_flow_fn = cs.Function(
        "constrained_flow_fn",
        [dof_sym, time_sym, s0_sym],
        [dof_flow_output_sym],
    )

    dummy_dof = cs.DM.zeros(1, dof)
    dummy_time = cs.DM(0.0)
    dummy_s0 = cs.DM.zeros(1, state_dim)

    _ = constrained_flow_fn(dummy_dof, dummy_time, dummy_s0)

    return constrained_flow_fn


def run_env(
    env,
    policy,
    cfg: FlowMatchingEvaluationConfig,
    run_id=0,
):
    assert env.name == "avoiding-v0", "This script is designed to run avoiding-v0."

    observation = env.reset()

    conditions = {}

    rollout = [observation.copy()]
    computation_times = []

    save_path = os.path.join(
        cfg.log_folder,
        cfg.env,
        "eval",
        cfg.exp_name,
    )

    success = False
    total_rewards = 0
    total_violations = 0

    assert (
        cfg.replan_steps < cfg.horizon
    ), f"replan steps ({cfg.replan_steps}) must be smaller than horizon ({cfg.horizon})"

    planned_actions = None
    action_index = 0

    pbar = tqdm.tqdm(range(cfg.max_episode_length), desc="Episode")
    for t in pbar:

        if cfg.controller == "rh":  # receding horizon
            if planned_actions is None or action_index >= cfg.replan_steps:
                conditions[0] = observation
                action, samples, _, _, info = policy(
                    conditions, batch_size=cfg.batch_size
                )
                planned_actions = samples.actions[0]
                action_index = 0

                if "computation_time" in info:
                    computation_times.append(info["computation_time"])

            action = planned_actions[action_index]
            action_index += 1

        else:
            raise NotImplementedError

        observation, reward, terminated, info = env.step(action)
        success = info[1]

        total_rewards += reward
        rollout.append(observation.copy())

        # update progress bar
        violation = check_violation(observation, cfg.constraint)
        if violation > 0 or (terminated and not success):
            violation = 1.0
        else:
            violation = 0.0

        total_violations += violation
        pbar.set_postfix(
            {
                "reward": f"{reward:.3f}",
                "total rewards": f"{total_rewards:.3f}",
                "violation": f"{violation:.3f}",
                "total violations": f"{total_violations:.3f}",
            }
        )

        if violation > 0 or terminated or success:
            if success:
                print("\nSucceeded!\n")
                break
            else:
                print("\nTerminated!\n")
                break

    # image of real trajectory
    real_trajectory = np.array(rollout)

    hardflow.utils.save_single_trajectory_image(
        real_trajectory,
        os.path.join(save_path, f"{run_id}_real.png"),
        cfg.constraint,
        cfg.obstacle_margin,
        cfg.draw_obstacle_margin,
    )

    avg_computation_time = np.mean(computation_times) if computation_times else None

    return (
        total_rewards,
        total_violations,
        np.array(rollout),
        success,
        avg_computation_time,
    )


def check_violation(observation, constraint):

    if constraint == "":
        return 0.0

    x = observation[2]
    y = observation[3]

    violation = 0.0

    for center, radius in zip(PILLAR_CENTERS, PILLAR_RADII):
        distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        if distance <= radius:
            violation = 1.0
            return violation

    if is_novel_constraint(constraint):
        distance = np.sqrt(
            (x - NOVEL_OBSTACLE_CENTER[0]) ** 2 + (y - NOVEL_OBSTACLE_CENTER[1]) ** 2
        )
        if distance <= NOVEL_OBSTACLE_RADIUS:
            violation = 1.0

        for slope, intercept, direction in NOVEL_BOUNDARY_LINES:
            if direction == "below" and y > slope * x + intercept:
                violation = 1.0

        if (
            uses_hard_quadrilateral(constraint)
            and quadrilateral_constraint_value_numpy(x, y) < 0
        ):
            violation = 1.0

    else:
        raise NotImplementedError(f"Constraint {constraint} is not implemented.")

    return violation


def evaluate(cfg: FlowMatchingEvaluationConfig):

    # get dataset (only for its normalizer)
    dataset = SequenceDataset(
        env=cfg.env,
        horizon=cfg.horizon,
        normalizer=cfg.normalizer,
        preprocess_fns=cfg.preprocess_fns,
        max_path_length=cfg.max_path_length,
        max_n_episodes=cfg.max_n_episodes,
        termination_penalty=0,
        seed=cfg.seed,
    )
    normalizer = dataset.normalizer

    # load dynamics if available
    dynamics_model = None
    dynamics_path = os.path.join("logs", cfg.env, "dynamics", "linear_model.npz")
    if os.path.exists(dynamics_path) and cfg.dynamics_constraint:
        print(f"Loading fitted dynamics from {dynamics_path}")
        dynamics_data = np.load(dynamics_path, allow_pickle=True)
        dynamics_model = {
            "A": dynamics_data["A"],
            "B": dynamics_data["B"],
            "c": dynamics_data["c"],
            "normalizer": dynamics_data["normalizer"].item(),
        }
        print("Fitted dynamics loaded successfully")
    else:
        print("No fitted dynamics found, proceeding without dynamics model")

    flow_model = TemporalUnet(
        horizon=cfg.horizon,
        transition_dim=cfg.state_dim + cfg.action_dim,
        cond_dim=cfg.state_dim,
        dim=32,
        dim_mults=(1, 4, 8),
        attention=False,
    ).to(cfg.device)
    flow_model.load_state_dict(
        torch.load(
            os.path.join(
                cfg.log_folder,
                cfg.env,
                "flow",
                cfg.flow_exp_name,
                f"model_ema_{cfg.flow_cp}.pth",
            )
        )
    )

    value_model = ProxyValueModel(
        cfg.horizon,
        cfg.action_dim,
        cfg.state_dim,
        objective=cfg.value_objective,
        constraint=cfg.constraint,
        obstacle_margin=cfg.obstacle_margin,
        value_objective_scale=cfg.value_objective_scale,
        value_constraint_scale=cfg.value_constraint_scale,
        dynamics_constraint=cfg.dynamics_constraint,
        normalizer=normalizer,
        dynamics_model=dynamics_model,
    ).to(cfg.device)

    flow_policy = FlowPolicy(
        flow_model=flow_model,
        value_model=value_model,
        normalizer=normalizer,
        action_dim=cfg.action_dim,
        state_dim=cfg.state_dim,
        horizon=cfg.horizon,
        cfg=cfg,
        dynamics_model=dynamics_model,
    )

    # l4casadi
    if (
        cfg.guidance_method == "projection"
        or cfg.guidance_method == "projection_relaxed"
        or cfg.guidance_method == "hardflow"
    ):
        wrapped_flow_model = WrappedFlowUnet(
            horizon=cfg.horizon,
            transition_dim=cfg.state_dim + cfg.action_dim,
            cond_dim=cfg.state_dim,
            dim=32,
            dim_mults=(1, 4, 8),
            attention=False,
        ).to(cfg.device)
        wrapped_flow_model.load_state_dict(
            torch.load(
                os.path.join(
                    cfg.log_folder,
                    cfg.env,
                    "flow",
                    cfg.flow_exp_name,
                    f"model_ema_{cfg.flow_cp}.pth",
                )
            )
        )
        wrapped_flow_model.add_info(
            horizon=cfg.horizon, transition_dim=cfg.state_dim + cfg.action_dim
        )

        l4c_flow_fn = l4c.L4CasADi(wrapped_flow_model, device="cuda", name="flow_model")

        constrained_flow_fn = create_constrained_casadi_functions(
            l4c_flow_fn,
            action_dim=cfg.action_dim,
            state_dim=cfg.state_dim,
            horizon=cfg.horizon,
        )

        flow_policy.constrained_flow_fn = constrained_flow_fn

        if cfg.guidance_method == "projection":
            flow_policy.projection_formulate(
                print_level=cfg.solver_print_level,
                constraint=cfg.constraint,
            )
        elif cfg.guidance_method == "projection_relaxed":
            flow_policy.projection_relaxed_formulate(
                print_level=cfg.solver_print_level,
                constraint=cfg.constraint,
            )
        elif cfg.guidance_method == "hardflow":
            flow_policy.hardflow_formulate(
                print_level=cfg.solver_print_level,
                constraint=cfg.constraint,
                objective=cfg.cost,
            )

    if cfg.guidance_method == "hardflow_new":
        flow_policy.hardflow_formulate(
            print_level=cfg.solver_print_level,
            constraint=cfg.constraint,
            objective=cfg.cost,
        )

    # run env
    env = gym.make("avoiding-v0", render=cfg.render)
    env.set_seed(cfg.seed)
    env.start()

    trajectory_data = []

    for run_id in range(cfg.random_repeat):
        (
            total_rewards,
            total_violations,
            real_trajectory,
            success,
            avg_computation_time,
        ) = run_env(env, flow_policy, cfg, run_id=run_id)

        steps = len(real_trajectory) - 1
        safety = total_violations == 0

        trajectory_data.append(
            {
                "run_id": run_id,
                "steps": steps,
                "total_violations": float(total_violations),
                "total_rewards": float(total_rewards),
                "success": success,
                "safety": safety,
                "average_computation_time": (
                    float(avg_computation_time)
                    if avg_computation_time is not None
                    else None
                ),
            }
        )

    csv_path = os.path.join(
        cfg.log_folder, cfg.env, "eval", cfg.exp_name, "trajectories.csv"
    )
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "run_id",
            "steps",
            "total_violations",
            "total_rewards",
            "success",
            "safety",
            "average_computation_time",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in trajectory_data:
            writer.writerow(data)


if __name__ == "__main__":
    cfg = tyro.cli(FlowMatchingEvaluationConfig)
    set_cuda_visible_device(cfg)
    deterministic(cfg.seed)  # seed everything

    log_subfolder = os.path.join(cfg.log_folder, cfg.env, "eval", cfg.exp_name)
    save_config(cfg, log_subfolder)

    evaluate(cfg)
