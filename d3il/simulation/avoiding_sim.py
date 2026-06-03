import logging
import multiprocessing as mp
import os
import random
from envs.gym_avoiding_env.gym_avoiding.envs.avoiding import ObstacleAvoidanceEnv

import cv2
import numpy as np
import torch
import wandb

from simulation.base_sim import BaseSim

log = logging.getLogger(__name__)

# Training images are 96×96 RGB (ParityAvoidingDataset: cv2.cvtColor BGR2RGB + transpose).
# BPCageCam defaults to 1024×1024 — resize every frame before passing to the model.
_IMG_W = _IMG_H = 96


def assign_process_to_cpu(pid, cpus):
    os.sched_setaffinity(pid, cpus)


class Avoiding_Sim(BaseSim):
    """
    Gen9 Ep 2 Fix-3: visual-extended Avoiding_Sim.

    Mirrors Aligning_Sim interface (n_contexts, n_trajectories_per_context, if_vision,
    eval_on_train, max_episode_length) so the visual avoiding eval scripts can call
    test_agent() and get back (success_rate, mode_encoding, successes, mean_distance).

    Key differences from Aligning_Sim:
    - Uses ObstacleAvoidanceEnv (single bp_cam; no inhand_cam; no if_vision kwarg on env)
    - No pre-defined context files — treats each context idx as an independent random-start episode
    - step() returns (obs, reward, done, (mode_enc, success)) tuple; converted to info dict here
    - Visual predict call: agent.predict((bp_image, des_xy, c_xy), if_vision=True)
      — 2D positions, single image (no inhand)
    - Non-visual predict call: agent.predict(np.concat([des_xy, env_obs]))
    - Mode encoding: 1=success, 0=failure (2-mode for eval script compatibility)
    """

    def __init__(
            self,
            seed: int,
            device: str,
            render: bool,
            n_cores: int = 1,
            n_contexts: int = 30,
            n_trajectories_per_context: int = 1,
            if_vision: bool = False,
            eval_on_train: bool = False,
            max_episode_length: int = 400,
    ):
        super().__init__(seed, device, render, n_cores, if_vision)
        self.n_contexts = n_contexts
        self.n_trajectories_per_context = n_trajectories_per_context
        self.eval_on_train = eval_on_train
        self.max_episode_length = max_episode_length

    def eval_agent(self, agent, contexts, n_trajectories,
                   mode_encoding, successes, mean_distance, pid, cpu_set):

        print(os.getpid(), cpu_set)
        if not self.if_vision:
            assign_process_to_cpu(os.getpid(), cpu_set)
        else:
            print(f"Process {os.getpid()} unpinned — visual eval requires all CPU threads.")

        env = ObstacleAvoidanceEnv(render=self.render,
                                   max_steps_per_episode=self.max_episode_length)
        env.start()

        random.seed(self.seed + pid)
        torch.manual_seed(self.seed + pid)
        np.random.seed(self.seed + pid)

        for context in contexts:
            for i in range(n_trajectories):

                agent.reset()
                print(f'Context {context} Rollout {i}')

                obs = env.reset(random=True)  # avoiding has no fixed context files; random start

                # avoiding has no box/target context — record_context_info expects a 4-tuple
                # (box_pos, box_quat, target_pos, target_quat) which doesn't exist here; skip.

                # initial desired 2D position and fixed-z suffix for full 7-D action
                pred_xy = env.robot_state()[:2].copy()
                fixed_z_q = np.concatenate([env.robot_state()[2:], [0, 1, 0, 0]])

                done = False
                rollout_dists = []

                if self.if_vision:
                    while not done:
                        c_xy = env.robot.current_c_pos[:2].copy()

                        # Single camera: env returns BGR 1024×1024; resize to training res 96×96,
                        # then BGR→RGB via [::-1] to match ParityAvoidingDataset._load_images().
                        bp_img_raw = env.bp_cam.get_image(depth=False)
                        bp_img_raw = cv2.resize(bp_img_raw, (_IMG_W, _IMG_H), interpolation=cv2.INTER_AREA)
                        bp_image = bp_img_raw[:, :, ::-1].transpose((2, 0, 1)).copy() / 255.

                        pred_delta = agent.predict(
                            (bp_image, pred_xy.copy(), c_xy.copy()), if_vision=True)
                        pred_xy = pred_delta[0] + pred_xy

                        full_action = np.concatenate([pred_xy, fixed_z_q])
                        obs, reward, done, (mode_enc, success) = env.step(full_action)

                        c_xy = env.robot.current_c_pos[:2].copy()
                        dist = float(abs(c_xy[1] - env.goal_ypos))
                        rollout_dists.append(dist)

                        info_dict = {
                            'mode': 1 if success else 0,
                            'success': bool(success),
                            'mean_distance': dist,
                        }
                        if hasattr(agent, 'record_step_info'):
                            agent.record_step_info(info_dict)

                        # No inhand_cam — pass bp_image twice so capture_frame(bp, ih) signature works
                        if hasattr(agent, 'capture_frame'):
                            try:
                                agent.capture_frame(bp_image, bp_image)
                            except Exception:
                                pass

                else:  # non-visual
                    while not done:
                        obs_concat = np.concatenate([pred_xy, obs])

                        pred_delta = agent.predict(obs_concat)
                        pred_xy = pred_delta[0] + obs_concat[:2]

                        full_action = np.concatenate([pred_xy, fixed_z_q])
                        obs, reward, done, (mode_enc, success) = env.step(full_action)

                        c_xy = env.robot.current_c_pos[:2].copy()
                        dist = float(abs(c_xy[1] - env.goal_ypos))
                        rollout_dists.append(dist)

                        info_dict = {
                            'mode': 1 if success else 0,
                            'success': bool(success),
                            'mean_distance': dist,
                        }
                        if hasattr(agent, 'record_step_info'):
                            agent.record_step_info(info_dict)

                        if hasattr(agent, 'capture_frame'):
                            try:
                                bp_img_raw = env.bp_cam.get_image(depth=False)
                                bp_img_raw = cv2.resize(bp_img_raw, (_IMG_W, _IMG_H), interpolation=cv2.INTER_AREA)
                                bp_np = bp_img_raw[:, :, ::-1].transpose((2, 0, 1)).copy() / 255.
                                agent.capture_frame(bp_np, bp_np)
                            except Exception:
                                pass

                mode_encoding[context, i] = torch.tensor(1 if success else 0, dtype=torch.float32)
                successes[context, i]     = torch.tensor(float(success))
                mean_distance[context, i] = torch.tensor(
                    float(np.mean(rollout_dists)) if rollout_dists else 0.0)

                final_info = {
                    'mean_distance': float(np.mean(rollout_dists)) if rollout_dists else 0.0,
                    'success': bool(success),
                    'mode': 1 if success else 0,
                }
                if hasattr(agent, 'update_rollout_info'):
                    agent.update_rollout_info(final_info)

    ################################
    # n_contexts: number of evaluation episodes (no fixed context files for avoiding)
    # n_trajectories_per_context: trials per episode
    # n_cores: parallel worker count
    ###############################
    def test_agent(self, agent):

        log.info('Starting trained model evaluation')

        mode_encoding = torch.zeros(
            [self.n_contexts, self.n_trajectories_per_context]).share_memory_()
        successes     = torch.zeros(
            (self.n_contexts, self.n_trajectories_per_context)).share_memory_()
        mean_distance = torch.zeros(
            (self.n_contexts, self.n_trajectories_per_context)).share_memory_()

        contexts = np.arange(self.n_contexts)
        workload = self.n_contexts // self.n_cores

        num_cpu = mp.cpu_count()
        cpu_set = list(range(num_cpu))
        print("there are cpus: ", num_cpu)

        ctx = mp.get_context('spawn')

        p_list = []
        if self.n_cores > 1:
            for i in range(self.n_cores):
                p = ctx.Process(
                    target=self.eval_agent,
                    kwargs={
                        "agent":         agent,
                        "contexts":      contexts[i * workload:(i + 1) * workload],
                        "n_trajectories": self.n_trajectories_per_context,
                        "mode_encoding": mode_encoding,
                        "successes":     successes,
                        "mean_distance": mean_distance,
                        "pid":           i,
                        "cpu_set":       set(cpu_set[i:i + 1]),
                    },
                )
                print(f"Start {i}")
                p.start()
                p_list.append(p)
            [p.join() for p in p_list]

        else:
            self.eval_agent(agent, contexts, self.n_trajectories_per_context,
                            mode_encoding, successes, mean_distance, 0, cpu_set=set([0]))

        n_modes = 2
        success_rate = torch.mean(successes).item()

        mode_probs = torch.zeros([self.n_contexts, n_modes])
        for c in range(self.n_contexts):
            mode_probs[c] = torch.tensor([
                (mode_encoding[c] == 0).sum().item() / self.n_trajectories_per_context,
                (mode_encoding[c] == 1).sum().item() / self.n_trajectories_per_context,
            ])
        mode_probs /= (mode_probs.sum(1).reshape(-1, 1) + 1e-12)

        entropy = -(mode_probs * torch.log(mode_probs + 1e-12) /
                    torch.log(torch.tensor(float(n_modes)))).sum(1).mean()

        wandb.log({'score': 0.5 * (success_rate + entropy.item())})
        wandb.log({'Metrics/successes': success_rate})
        wandb.log({'Metrics/entropy': entropy.item()})
        wandb.log({'Metrics/distance': mean_distance.mean().item()})

        print(f'Mean Distance {mean_distance.mean().item()}')
        print(f'Success Rate  {success_rate}')
        print(f'Entropy       {entropy.item()}')

        return success_rate, mode_encoding, successes, mean_distance
