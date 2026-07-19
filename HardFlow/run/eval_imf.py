"""Gen13 — iMF eval entry (additive sibling of run/eval.py, which is untouched).

Reuses run.eval's ProxyValueModel unchanged via import (run/eval.py is
__main__-guarded, so the import is side-effect free). Only the model build
and policy class differ from run.eval.evaluate():

  * TemporalImfUnet + ImfFlowPolicy instead of TemporalUnet + FlowPolicy
  * guidance methods: original_imf | hardflow_new_imf   (FM methods -> run/eval.py)
  * K (sampler NFE / solver steps) == --ode_t_steps  (K=1/2 is the paper regime)
  * no l4casadi anywhere on this path
  * per-episode mean NFE is appended to the CSV (nfe_* columns)

Everything else — dataset/normalizer, fitted dynamics, avoiding-v0 env loop,
CSV format — mirrors run/eval.py so results are directly comparable with the
frozen FM baselines.

NOTE (fix_2, quiet console): run.eval.run_env is FORKED (not imported) as
_run_env_quiet below, purely to fix console-log noise (see its docstring).
run/eval.py itself is left untouched — Gen13's rule is additive-only, and
run.eval.run_env is pre-existing (pre-Gen13) code.
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")

import gym
import numpy as np
import torch
import tqdm
import tyro

import d3il  # noqa: F401  (registers avoiding-v0)
import hardflow.utils
from hardflow.utils.rendering import AvoidingTrajectoryPlotter

from hardflow.datasets.sequence import SequenceDataset
from hardflow.models_flow.imf import (
    ImfEvaluationConfig,
    ImfFlowPolicy,
    TemporalImfUnet,
)
from run.eval import ProxyValueModel, check_violation
from run.utils import deterministic, save_config, set_cuda_visible_device

_IS_TTY = sys.stdout.isatty()


def _save_foresight_fan(planned_fan, x1_fan, real_trajectory, filepath, cfg):
    """u_5(B) — MPC foresight-fan plot (the DPCC/FMPCC-style diagnostic).

    HardFlow computes the planned horizon and the terminal prediction at every
    replan but discards both (`run/eval.py:393` uses `_, _`), so only the executed
    rollout is ever rendered. This draws what the policy *intended*:

      * grey   — planned H-step horizons, one per replan instant (the "fan")
      * orange — the final terminal prediction x̂1 at each replan; for iMF this is
                 the EXACT endpoint map z + (1-tau)*u, vs FM's Euler shot — i.e.
                 the visual counterpart of the Gen13 seam swap
      * black  — the actually executed trajectory (same as the standard *_real.png)

    Index layout: the executed rollout is observations (state_dim=4) with x,y at
    2,3; planned/predicted trajectories are full transitions (action_dim+state_dim)
    with x,y at action_dim+2, action_dim+3.
    """
    px, py = cfg.action_dim + 2, cfg.action_dim + 3

    plotter = AvoidingTrajectoryPlotter(
        constraint=cfg.constraint,
        obstacle_margin=cfg.obstacle_margin,
        draw_obstacle_margin=cfg.draw_obstacle_margin,
    )
    fig, ax = plotter.setup_figure()

    for i, plan in enumerate(planned_fan):
        ax.plot(
            plan[:, px], plan[:, py], "-", color="0.55", linewidth=1.2, zorder=2,
            label="Planned horizon (fan)" if i == 0 else "",
        )
    for i, x1 in enumerate(x1_fan):
        ax.plot(
            x1[:, px], x1[:, py], "--", color="tab:orange", linewidth=1.0, alpha=0.8,
            zorder=3, label="Terminal prediction x̂1" if i == 0 else "",
        )

    plotter.plot_single_trajectory(real_trajectory, ax, style="actual")
    plotter.add_environment_elements(ax)
    plotter.apply_legend(ax)
    plotter.save_figure(fig, filepath)


def _run_env_quiet(env, policy, cfg, run_id=0):
    """Fork of run.eval.run_env with QUIET progress reporting (Gen13 fix_2).

    Functionally IDENTICAL to run.eval.run_env — same env stepping, violation
    counting, image saving, and return values (so CSV numbers stay directly
    comparable to the frozen FM baselines). Forked rather than imported because
    run/eval.py is pre-existing HardFlow code and Gen13's rule is additive-only
    (no edits to existing files).

    WHY: run.eval.run_env calls tqdm's set_postfix() every single timestep.
    Under SLURM, stdout is redirected to a log FILE, not a real terminal, so
    tqdm's carriage-return trick never collapses in place — every one of up to
    100 steps x 50 episodes gets dumped as raw text, producing the
    thousands-of-characters unreadable log lines seen in practice (job 23565).
    This fork uses a live tqdm bar only when stdout IS a real terminal
    (_IS_TTY), and a single compact status line per episode otherwise.
    """
    assert env.name == "avoiding-v0", "This script is designed to run avoiding-v0."

    observation = env.reset()
    conditions = {}
    rollout = [observation.copy()]
    computation_times = []

    save_path = os.path.join(cfg.log_folder, cfg.env, "eval", cfg.exp_name)

    success = False
    total_rewards = 0
    total_violations = 0

    assert (
        cfg.replan_steps < cfg.horizon
    ), f"replan steps ({cfg.replan_steps}) must be smaller than horizon ({cfg.horizon})"

    planned_actions = None
    action_index = 0

    # u_5(B): foresight-fan buffers (stay empty unless cfg.imf_plot_fan)
    planned_fan, x1_fan, n_replan = [], [], 0

    # fix_4: zero the cumulative NLP counters so the summary below is per-episode
    if hasattr(policy, "reset_nlp_stats"):
        policy.reset_nlp_stats()

    steps_iter = (
        tqdm.tqdm(range(cfg.max_episode_length), desc=f"Episode {run_id}")
        if _IS_TTY
        else range(cfg.max_episode_length)
    )

    final_t = 0
    for t in steps_iter:
        final_t = t

        if cfg.controller == "rh":  # receding horizon
            if planned_actions is None or action_index >= cfg.replan_steps:
                conditions[0] = observation
                action, samples, x_chain, x1_est, info = policy(
                    conditions, batch_size=cfg.batch_size
                )
                planned_actions = samples.actions[0]
                action_index = 0
                if "computation_time" in info:
                    computation_times.append(info["computation_time"])
                # u_5(B): foresight-fan capture — OFF by default, so this branch
                # is skipped entirely on the decisive n=200 safety run.
                if getattr(cfg, "imf_plot_fan", False):
                    n_replan += 1
                    if (n_replan - 1) % max(1, cfg.imf_fan_every) == 0:
                        # x_chain: (batch, n_steps+1, horizon, transition_dim),
                        # already unnormalized. [-1] = the FINAL planned horizon.
                        planned_fan.append(np.asarray(x_chain)[0, -1, :, :].copy())
                        x1_fan.append(np.asarray(x1_est)[0, -1, :, :].copy())
            action = planned_actions[action_index]
            action_index += 1
        else:
            raise NotImplementedError

        observation, reward, terminated, info = env.step(action)
        success = info[1]
        total_rewards += reward
        rollout.append(observation.copy())

        violation = check_violation(observation, cfg.constraint)
        violation = 1.0 if (violation > 0 or (terminated and not success)) else 0.0
        total_violations += violation

        if _IS_TTY:
            steps_iter.set_postfix(
                {
                    "reward": f"{reward:.3f}",
                    "total_rewards": f"{total_rewards:.3f}",
                    "violations": f"{total_violations:.0f}",
                }
            )

        if violation > 0 or terminated or success:
            break

    outcome = "SUCCESS" if success else "terminated"
    # fix_4: NLP health is appended here — with IPOPT's own output silenced this
    # is the surviving signal from that block (nlp_fail>0 would flag a projection
    # that never converged, the prime suspect for any constraint violation).
    nlp = policy.nlp_stats() if hasattr(policy, "nlp_stats") else {}
    nlp_str = (
        f"  nlp={nlp['nlp_solves']}"
        + (f" FAILED={nlp['nlp_failures']}" if nlp.get("nlp_failures") else "")
        if nlp
        else ""
    )
    print(
        f"[ eval_imf ] episode {run_id}: {outcome}  steps={final_t + 1}  "
        f"violations={total_violations:.0f}  reward={total_rewards:.3f}{nlp_str}"
    )

    real_trajectory = np.array(rollout)
    hardflow.utils.save_single_trajectory_image(
        real_trajectory,
        os.path.join(save_path, f"{run_id}_real.png"),
        cfg.constraint,
        cfg.obstacle_margin,
        cfg.draw_obstacle_margin,
    )

    # u_5(B): extra fan figure — only when explicitly enabled
    if getattr(cfg, "imf_plot_fan", False) and planned_fan:
        _save_foresight_fan(
            planned_fan,
            x1_fan,
            real_trajectory,
            os.path.join(save_path, f"{run_id}_fan.png"),
            cfg,
        )

    avg_computation_time = np.mean(computation_times) if computation_times else None
    return (
        total_rewards,
        total_violations,
        np.array(rollout),
        success,
        avg_computation_time,
    )


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
    nlp_failures_total = 0

    for run_id in range(cfg.random_repeat):
        (
            total_rewards,
            total_violations,
            real_trajectory,
            success,
            avg_computation_time,
        ) = _run_env_quiet(env, flow_policy, cfg, run_id=run_id)

        steps = len(real_trajectory) - 1
        safety = total_violations == 0
        nfe_info = flow_policy._nfe_info()  # last planning call of the episode
        nfe_totals.append(nfe_info["nfe_total"])
        nlp_info = flow_policy.nlp_stats()  # cumulative over THIS episode (fix_4)
        nlp_failures_total += nlp_info["nlp_failures"]

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
                "nlp_solves": nlp_info["nlp_solves"],
                "nlp_failures": nlp_info["nlp_failures"],
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
            "nlp_solves",
            "nlp_failures",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in trajectory_data:
            writer.writerow(data)

    nlp_note = (
        f"  NLP failures: {nlp_failures_total} (!!)"
        if nlp_failures_total
        else "  all NLP solves OK"
    )
    print(
        f"[ eval_imf ] done. mean NFE/plan: {np.mean(nfe_totals):.1f}{nlp_note}  "
        f"csv: {csv_path}"
    )


if __name__ == "__main__":
    cfg = tyro.cli(ImfEvaluationConfig)
    set_cuda_visible_device(cfg)
    deterministic(cfg.seed)

    log_subfolder = os.path.join(cfg.log_folder, cfg.env, "eval", cfg.exp_name)
    save_config(cfg, log_subfolder)

    evaluate_imf(cfg)
