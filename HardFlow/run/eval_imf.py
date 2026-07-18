"""Gen13 — iMF eval entry (additive sibling of run/eval.py, which is untouched).

Reuses run.eval's ProxyValueModel and run_env unchanged via import (run/eval.py
is __main__-guarded, so the import is side-effect free). Only the model build
and policy class differ from run.eval.evaluate():

  * TemporalImfUnet + ImfFlowPolicy instead of TemporalUnet + FlowPolicy
  * guidance methods: original_imf | hardflow_new_imf   (FM methods -> run/eval.py)
  * K (sampler NFE / solver steps) == --ode_t_steps  (K=1/2 is the paper regime)
  * no l4casadi anywhere on this path
  * per-episode mean NFE is appended to the CSV (nfe_* columns)

Everything else — dataset/normalizer, fitted dynamics, avoiding-v0 env loop,
CSV format — mirrors run/eval.py so results are directly comparable with the
frozen FM baselines.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")

import gym
import numpy as np
import torch
import tyro

import d3il  # noqa: F401  (registers avoiding-v0)

from hardflow.datasets.sequence import SequenceDataset
from hardflow.models_flow.imf import (
    ImfEvaluationConfig,
    ImfFlowPolicy,
    TemporalImfUnet,
)
from run.eval import ProxyValueModel, run_env
from run.utils import deterministic, save_config, set_cuda_visible_device


def evaluate_imf(cfg: ImfEvaluationConfig):
    assert cfg.guidance_method in ImfFlowPolicy.IMF_GUIDANCE_METHODS, (
        f"run/eval_imf.py handles {ImfFlowPolicy.IMF_GUIDANCE_METHODS}; "
        f"for FM guidance methods use run/eval.py"
    )

    # dataset (only for its normalizer) — identical to run.eval.evaluate
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

    # fitted dynamics — identical to run.eval.evaluate
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

    # iMF backbone
    flow_model = TemporalImfUnet(
        horizon=cfg.horizon,
        transition_dim=cfg.state_dim + cfg.action_dim,
        cond_dim=cfg.state_dim,
        dim=32,
        dim_mults=(1, 4, 8),
        attention=False,
    ).to(cfg.device)
    ckpt_path = os.path.join(
        cfg.log_folder,
        cfg.env,
        "flow",
        cfg.flow_exp_name,
        f"model_ema_{cfg.flow_cp}.pth",
    )
    print(f"[ eval_imf ] loading iMF checkpoint: {ckpt_path}")
    flow_model.load_state_dict(torch.load(ckpt_path))
    flow_model.eval()

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

    flow_policy = ImfFlowPolicy(
        flow_model=flow_model,
        value_model=value_model,
        normalizer=normalizer,
        action_dim=cfg.action_dim,
        state_dim=cfg.state_dim,
        horizon=cfg.horizon,
        cfg=cfg,
        dynamics_model=dynamics_model,
    )

    if cfg.guidance_method == "hardflow_new_imf":
        flow_policy.hardflow_formulate(
            print_level=cfg.solver_print_level,
            constraint=cfg.constraint,
            objective=cfg.cost,
        )

    # env loop — reuses run.eval.run_env unchanged
    env = gym.make("avoiding-v0", render=cfg.render)
    env.set_seed(cfg.seed)
    env.start()

    trajectory_data = []
    nfe_totals = []

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
        nfe_info = flow_policy._nfe_info()  # last planning call of the episode
        nfe_totals.append(nfe_info["nfe_total"])

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
                "nfe_per_plan": nfe_info["nfe_total"],
                "nfe_sampling": nfe_info["nfe_sampling"],
                "nfe_warmstart": nfe_info["nfe_warmstart"],
                "nfe_diag": nfe_info["nfe_diag"],
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
            "nfe_per_plan",
            "nfe_sampling",
            "nfe_warmstart",
            "nfe_diag",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in trajectory_data:
            writer.writerow(data)

    print(
        f"[ eval_imf ] done. mean NFE/plan: {np.mean(nfe_totals):.1f}  "
        f"csv: {csv_path}"
    )


if __name__ == "__main__":
    cfg = tyro.cli(ImfEvaluationConfig)
    set_cuda_visible_device(cfg)
    deterministic(cfg.seed)

    log_subfolder = os.path.join(cfg.log_folder, cfg.env, "eval", cfg.exp_name)
    save_config(cfg, log_subfolder)

    evaluate_imf(cfg)
