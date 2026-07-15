from dataclasses import dataclass, field


@dataclass
class FlowMatchingTrainingConfig:
    seed: int = 0
    device: str = "cuda"
    log_folder: str = "logs"
    exp_name: str = "flow_matching"

    env: str = "avoiding-v0"
    horizon: int = 8
    normalizer: str = "LimitsNormalizer"
    preprocess_fns: list = field(default_factory=lambda: [])
    max_path_length: int = 200
    max_n_episodes: int = 500
    termination_penalty: float = 0

    state_dim: int = 4
    action_dim: int = 2

    flow_matching_type: str = "cfm"

    # training
    n_train_steps: int = 100000
    save_freq: int = 5000
    batch_size: int = 32
    learning_rate: float = 2e-4

    ema_decay: float = 0.995


@dataclass
class FlowMatchingEvaluationConfig:
    seed: int = 0
    random_repeat: int = 5
    device: str = "cuda"
    log_folder: str = "logs"
    exp_name: str = "flow_matching"

    render: bool = False

    controller: str = "rh"
    replan_steps: int = 1

    env: str = "avoiding-v0"
    horizon: int = 8
    normalizer: str = "LimitsNormalizer"
    preprocess_fns: list = field(default_factory=lambda: [])
    max_path_length: int = 200
    max_n_episodes: int = 500
    termination_penalty: float = 0
    max_episode_length: int = 100

    state_dim: int = 4
    action_dim: int = 2

    flow_exp_name: str = "flow_matching"
    flow_cp: str = "0"
    flow_matching_type: str = "cfm"

    batch_size: int = 1
    ode_solver: str = "euler"
    ode_t_span: tuple = (0, 1)
    ode_t_steps: int = 20

    guidance_method: str = "original"

    value_objective_scale: float = 1.0
    value_objective: str = ""
    value_constraint_scale: float = 1.0

    cost: str = ""
    constraint: str = ""
    obstacle_margin: float = 0.0
    draw_obstacle_margin: bool = False
    dynamics_constraint: bool = False
    projection_option: str = "all"
    projection_gradient_lr: float = 1.0
    projection_gradient_steps: int = 5
    hardflow_activation: str = "all"
    warmstart_batch: int = 1
    solver_print_level: int = 5

    hardflow_reg_scale: float = 1.0
    hardflow_cost_scale: float = 1.0
    cost_scale: float = 1.0

    guidance_lr: float = 0.5
    guidance_steps: int = 5

    oc_flow_lr: float = 1.0
    oc_flow_steps: int = 10
