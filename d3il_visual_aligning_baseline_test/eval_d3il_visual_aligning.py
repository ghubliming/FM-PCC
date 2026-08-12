"""
D3IL Visual Aligning Baseline — evaluation script.

Runs any pre-trained D3IL aligning agent (BC, DDPM, BESO…) with rich per-rollout
recording: GIF/video, JSON stats, 2-panel summary PNG.  Output schema matches the
DPCC/FM eval so the same analysis notebooks can compare all three.

No constraint metrics (D3IL has no projector).  No MPC foresight panel.

Typical SLURM invocation:
    python d3il_visual_aligning_baseline_test/eval_d3il_visual_aligning.py \
        --agent-name ddpm_encdec_vision --seed 42 --record all

Output layout:
    logs/d3il_visual_aligning_baseline/{agent_name}/seed_{s}/
        diagnostics/
            rollout_0.gif / rollout_0.mp4
            rollout_0_stats.json
            rollout_0_summary.png
            rollout_0_data.pkl
            …
        results_seed_{s}.json
    logs/d3il_visual_aligning_baseline/{agent_name}/aggregate_results.json

U4.3 — DA export (on by default, `--no-da-export` to skip):
a second, additive copy of the results is written in the Gen14 API shape that
`Data_Analysis/DA_VA_v2` reads natively, so this baseline lands in the same
tables as the FM/MF/AF/diffusion arms without any bridging:

    logs/d3il_visual_aligning_baseline/DA_VA_d3il_baseline/
        d3il_baseline_{agent_name}/{seed}/results[_train_set]/d3il_baseline/
            d3il_baseline.npz  unit_meta.json  diagnostics/rollout_{r}_stats.json

The schema lives in `da_va_export.py` and is shared with
`bridge_d3il_va_to_da_va_v2.py`, which produces the identical layout for runs
that finished before this export existed (those land under a `_`-prefixed root
so DA_VA_v2 knows to apply its legacy reader).
"""

import argparse
import json
import os
import pickle
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
import yaml

try:
    import imageio
    _IMAGEIO = True
except ImportError:
    _IMAGEIO = False
    print('[ WARNING ] imageio not found — video/GIF recording disabled')

# Hydra compose (programmatic — no CWD change, no Hydra output dirs)
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

# D3IL imports — PYTHONPATH set by SLURM: D3IL_ROOT, D3IL_ENV_ROOT
from envs.gym_aligning_env.gym_aligning.envs.aligning import Robot_Push_Env
from agents.utils.sim_path import sim_framework_path
# REAL_TIME_RECORDING_UPDATE — per-step timing/digital-twin recorder (see logs_in_develop/REALTIME_RECORDING).
# D3IL is the DPCC-free BASELINE: fm_ms=total (no projector), proj_ms=0 — the reference every
# FM-PCC fm_overhead is measured against (IDEAS.md §Priority: do the baseline first).
import time as _time
from realtime_recording.behavior_logger import RTRecorder

# U4.3 — DA_VA_v2 export schema, shared with bridge_d3il_va_to_da_va_v2.py.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import da_va_export as da_api
RT_CONTROL_HZ = 30   # REAL_TIME_RECORDING_UPDATE — assumed deployment loop rate (budget=1000/hz ms); tune per target hardware

OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers), replace=True)

_TRAIN_CTX = np.load(
    sim_framework_path("environments/dataset/data/aligning/train_contexts.pkl"),
    allow_pickle=True,
)
_TEST_CTX = np.load(
    sim_framework_path("environments/dataset/data/aligning/test_contexts.pkl"),
    allow_pickle=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# D3ILBaselineWrapper
# ─────────────────────────────────────────────────────────────────────────────

class D3ILBaselineWrapper:
    """
    Wraps any D3IL aligning agent.  Implements the same hook interface as
    VisualAgentWrapper (update_rollout_info, record_context_info) so
    aligning_sim hooks work if ever used, while our own rollout loop calls
    record_step() directly.
    """

    def __init__(self, agent, save_path, record_mode='all'):
        self.agent       = agent
        self.save_path   = save_path
        self.record_mode = record_mode

        # per-rollout buffers
        self.video_frames      = []
        self.ee_traj           = []   # list[np.array(3,)] — des_robot_pos each step
        self.dist_curve        = []   # list[float]        — mean_distance each step
        self.step_times_ms     = []   # U4.3 — agent.predict() wall time per step
        self.step_count        = 0
        self.curr_context_info = {}

        # cross-rollout
        self.history_context_info = []
        self.master_history        = {}
        self._rollout_idx          = 0

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def reset(self):
        self.agent.reset()
        self.video_frames.clear()
        self.ee_traj.clear()
        self.dist_curve.clear()
        self.step_times_ms.clear()
        self.step_count    = 0
        self.curr_context_info = {}

    # ── hooks ─────────────────────────────────────────────────────────────────

    def record_context_info(self, context, context_idx):
        """Called once per rollout after env.reset().
        context = (pos, quat, target_pos, target_quat) from aligning env.
        pos = [box_x, box_y, box_angle_deg]; target_pos = [tgt_x, tgt_y, tgt_angle_deg].
        """
        pos, _quat, target_pos, _tquat = context
        init_xy_dist = float(np.linalg.norm(
            np.array([pos[0], pos[1]]) - np.array([target_pos[0], target_pos[1]])
        ))
        self.curr_context_info = {
            'context_idx':        int(context_idx),
            'box_init_xy':        [float(pos[0]),        float(pos[1])],
            'box_init_angle_deg': float(pos[2]),
            'target_xy':          [float(target_pos[0]), float(target_pos[1])],
            'target_angle_deg':   float(target_pos[2]),
            'init_xy_dist':       init_xy_dist,
        }

    def record_step(self, des_robot_pos, frame_bgr, mean_distance, infer_ms=None):
        """Called from our rollout loop after each env.step().
        frame_bgr: HxWxC uint8 BGR image or None (state-only mode).
        infer_ms:  agent.predict() wall time for this step (U4.3 — the only
                   timing the old runs kept outside the realtime text logs).
        """
        self.ee_traj.append(des_robot_pos[:3].copy())
        self.dist_curve.append(float(mean_distance))
        if infer_ms is not None:
            self.step_times_ms.append(float(infer_ms))
        self.step_count += 1
        if frame_bgr is not None and self.record_mode != 'none':
            # BGR → RGB for video
            self.video_frames.append(frame_bgr[..., ::-1].copy())

    def update_rollout_info(self, info):
        """UF-16.4 hook: called after rollout with final box state.
        Works both from our own loop and from aligning_sim hooks.
        """
        ridx = self._rollout_idx

        # final box state (UF-16.4 formula)
        _fbp = info.get('final_box_pos')
        _fbq = info.get('final_box_quat')
        if _fbp is not None and self.curr_context_info:
            _fx, _fy = float(_fbp[0]), float(_fbp[1])
            _w, _x, _y, _z = (float(_fbq[0]), float(_fbq[1]),
                               float(_fbq[2]), float(_fbq[3]))
            _final_angle = float(np.degrees(
                np.arctan2(2 * (_w * _z + _x * _y), 1 - 2 * (_y ** 2 + _z ** 2))
            ))
            _txy = self.curr_context_info.get('target_xy', [0.0, 0.0])
            _final_xy_dist = float(np.sqrt(
                (_fx - _txy[0]) ** 2 + (_fy - _txy[1]) ** 2
            ))
            self.curr_context_info.update({
                'final_box_xy':        [_fx, _fy],
                'final_box_angle_deg': _final_angle,
                'final_xy_dist':       _final_xy_dist,
            })

        self.history_context_info.append(dict(self.curr_context_info))

        ee_arr = np.array(self.ee_traj) if self.ee_traj else np.zeros((1, 3))
        rec = {
            'success':       bool(info.get('success',       False)),
            'steps':         int(self.step_count),
            'mean_distance': float(info.get('mean_distance', 0.0)),
            'mode':          int(info.get('mode',            0)),
            # U4.1 bug B1 fix: the context index MUST live on the record.
            # compute_behavior_entropy() reads `rec['context']`; it was never
            # stored, so every rollout was skipped and entropy was a hard 0.0 in
            # every results_seed_*.json ever written by this script.
            'context':       int(info.get('context',
                                          self.curr_context_info.get('context_idx', -1))),
            'context_info':  dict(self.curr_context_info),
            'ee_traj':       ee_arr,
            'dist_curve':    list(self.dist_curve),
            # U4.3 — mean seconds per control step (D3IL predicts every step;
            # the action chunk is served from the agent's internal buffer).
            'avg_time':      (float(np.mean(self.step_times_ms)) / 1e3
                              if self.step_times_ms else float('nan')),
        }
        self.master_history[f'rollout_{ridx}'] = rec

        # console summary
        ci = self.curr_context_info
        print(f'  - Success: {rec["success"]}   mode={rec["mode"]}   '
              f'mean_dist={rec["mean_distance"]:.4f}   steps={rec["steps"]}')
        if 'box_init_xy' in ci:
            print(f'  - Box  init XY=({ci["box_init_xy"][0]:.3f}, '
                  f'{ci["box_init_xy"][1]:.3f})  angle={ci["box_init_angle_deg"]:.1f}°')
            print(f'  - Target   XY=({ci["target_xy"][0]:.3f}, '
                  f'{ci["target_xy"][1]:.3f})  angle={ci["target_angle_deg"]:.1f}°')
            print(f'  - Init XY dist (box→target): {ci["init_xy_dist"]:.4f} m')
        if 'final_box_xy' in ci:
            print(f'  - Box final XY=({ci["final_box_xy"][0]:.3f}, '
                  f'{ci["final_box_xy"][1]:.3f})  angle={ci["final_box_angle_deg"]:.1f}°'
                  f'  (dist_to_target: {ci["final_xy_dist"]:.4f} m)')

        if self.save_path is not None:
            self._export_rollout_realtime(ridx)

        self._rollout_idx += 1

    def predict(self, obs, if_vision=False):
        """aligning_sim-compatible interface.
        aligning_sim passes (bp, inhand, des_pos, robot_pos) 4-tuple;
        D3IL agents expect (bp, inhand, des_pos) 3-tuple — we strip the 4th element.
        """
        if if_vision:
            bp, inh, des_pos, _robot_pos = obs
            return self.agent.predict((bp, inh, des_pos), if_vision=True)
        return self.agent.predict(obs)

    # ── per-rollout export ────────────────────────────────────────────────────

    def _export_rollout_realtime(self, ridx):
        diag_path = os.path.join(self.save_path, 'diagnostics')
        os.makedirs(diag_path, exist_ok=True)
        rec = self.master_history[f'rollout_{ridx}']

        # ── video / GIF ────────────────────────────────────────────────────────
        if self.record_mode != 'none' and self.video_frames and _IMAGEIO:
            if self.record_mode in ('video', 'all'):
                try:
                    imageio.mimsave(
                        os.path.join(diag_path, f'rollout_{ridx}.mp4'),
                        self.video_frames, fps=20,
                    )
                except Exception as e:
                    print(f'[ WARNING ] mp4 failed: {e}')
            if self.record_mode in ('gif', 'all'):
                try:
                    imageio.mimsave(
                        os.path.join(diag_path, f'rollout_{ridx}.gif'),
                        self.video_frames, fps=10,
                    )
                except Exception as e:
                    print(f'[ WARNING ] gif failed: {e}')

        # ── stats JSON ─────────────────────────────────────────────────────────
        stats = {
            'rollout_index': int(ridx),
            'success':       rec['success'],
            'steps':         rec['steps'],
            'mean_distance': rec['mean_distance'],
            'mode':          rec['mode'],
            'context_info':  rec['context_info'],
            # U4.3 — additive; DA_VA_v2's flat-schema JSON reader picks this up
            # under exactly this name, so even a hand-copied tree carries timing.
            # NaN → null: json.dump would otherwise emit bare `NaN`, which the
            # HTML viewers' JSON.parse rejects.
            'avg_inference_time_per_replan': da_api.json_safe(rec['avg_time']),
        }
        with open(os.path.join(diag_path, f'rollout_{ridx}_stats.json'), 'w') as f:
            json.dump(stats, f, indent=4)

        # ── pkl (full data) ────────────────────────────────────────────────────
        pkl_rec = {k: v for k, v in rec.items() if k != 'ee_traj'}
        ee = rec['ee_traj']
        pkl_rec['ee_traj'] = ee.tolist() if isinstance(ee, np.ndarray) else ee
        with open(os.path.join(diag_path, f'rollout_{ridx}_data.pkl'), 'wb') as f:
            pickle.dump(pkl_rec, f)

        # ── 2-panel summary PNG ────────────────────────────────────────────────
        try:
            ci  = rec['context_info']
            ee  = rec['ee_traj']   # (T, 3)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            fig.suptitle(
                f'Rollout {ridx}  success={rec["success"]}  '
                f'mean_dist={rec["mean_distance"]:.4f}  steps={rec["steps"]}'
            )

            # left panel: XY trajectory
            ax = axes[0]
            if isinstance(ee, np.ndarray) and ee.shape[0] > 1:
                ax.plot(ee[:, 0], ee[:, 1], 'k-', lw=1.5, label='EE path')
                ax.plot(ee[0,  0], ee[0,  1], 'bs', ms=6, label='EE start')
                ax.plot(ee[-1, 0], ee[-1, 1], 'b^', ms=6, label='EE end')
            if 'box_init_xy' in ci:
                bx, by = ci['box_init_xy']
                tx, ty = ci['target_xy']
                ax.plot(bx, by, 'rx', ms=10, mew=2, label='box init')
                ax.plot(tx, ty, 'g*', ms=12,        label='target')
            if 'final_box_xy' in ci:
                fx, fy = ci['final_box_xy']
                ax.plot(fx, fy, 'r^', ms=8,         label='box final')
            ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)')
            ax.set_title('XY Trajectory')
            ax.legend(fontsize=7)
            ax.set_aspect('equal', 'box')

            # right panel: distance to target over steps
            ax2 = axes[1]
            dc = rec['dist_curve']
            if dc:
                ax2.plot(dc, 'r-', lw=1.5, label='mean_distance')
                ax2.axhline(0.033, color='g', ls='--', lw=1, label='success threshold')
            ax2.set_xlabel('Step'); ax2.set_ylabel('mean_distance')
            ax2.set_title('Distance to Target')
            ax2.legend(fontsize=7)

            plt.tight_layout()
            plt.savefig(os.path.join(diag_path, f'rollout_{ridx}_summary.png'), dpi=100)
            plt.close(fig)
        except Exception as e:
            print(f'[ WARNING ] summary PNG failed: {e}')


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────────────────────

def build_agent(d3il_config_dir, agent_cfg_group, device, seed):
    """Instantiate a D3IL agent via Hydra compose.
    No Hydra main decorator → no CWD change, no output directories created.
    Vision agents use aligning_vision_config (num_hidden_layers=4);
    state agents use aligning_config (num_hidden_layers=6).
    Must match the config used at training time or state_dict keys won't align.
    """
    # Mirror the same auto-select logic as train_d3il_baseline.sh
    base_config = "aligning_vision_config" if "_vision" in agent_cfg_group else "aligning_config"
    GlobalHydra.instance().clear()
    abs_cfg_dir = os.path.abspath(d3il_config_dir)
    with initialize_config_dir(config_dir=abs_cfg_dir):
        cfg = compose(
            config_name=base_config,
            overrides=[
                f"agents={agent_cfg_group}",
                f"seed={seed}",
                f"device={device}",
            ],
        )
    agent = instantiate(cfg.agents)
    return agent


# ─────────────────────────────────────────────────────────────────────────────
# U2: paper-faithful behavior entropy
# ─────────────────────────────────────────────────────────────────────────────

def compute_behavior_entropy(records, n_contexts, n_trajs, n_modes=2):
    """Conditional behavior entropy — the D3IL paper diversity metric (Eq. 2).

    Faithful re-implementation of d3il `simulation/aligning_sim.py:178-194`:
      • per context c, among the SUCCESSFUL rollouts, count how many used each mode;
      • divide by n_trajs (NOT by #successes) → p̃(m|c);
      • row-normalize (+1e-12) → p(m|c);
      • entropy_c = −Σ_m p(m|c)·log(p(m|c)) / log(n_modes)   (base-|B| ⇒ ∈[0,1]);
      • average over the n_contexts initial states (Monte-Carlo estimate of E_{s0}[H]).

    `records` is the list of per-rollout dicts (each has 'context', 'mode', 'success').
    Returns a float in [0, 1]; 1 = even use of all modes, 0 = mode collapse.

    U4.1 bug B1: `update_rollout_info()` never stored 'context', so this loop
    skipped every rollout and returned exactly 0.0 for every run. The record now
    carries it; the `context_info` fallback below keeps this function correct for
    any caller that builds records the older way.
    """
    if n_contexts <= 0 or n_trajs <= 0:
        return 0.0
    # bucket modes of SUCCESSFUL rollouts per context
    counts = np.zeros((n_contexts, n_modes), dtype=np.float64)
    for r in records:
        c = int(r.get('context', r.get('context_info', {}).get('context_idx', -1)))
        if not (0 <= c < n_contexts):
            continue
        if not bool(r.get('success', False)):
            continue                      # success-conditioned (paper: successes[c,:]==1)
        m = int(r.get('mode', -1))
        if 0 <= m < n_modes:
            counts[c, m] += 1.0
    probs = counts / float(n_trajs)                       # p̃(m|c)
    probs = probs / (probs.sum(axis=1, keepdims=True) + 1e-12)   # normalize rows
    ent_per_ctx = -(probs * np.log(probs + 1e-12)
                    / np.log(n_modes)).sum(axis=1)        # ∈[0,1] each
    return float(ent_per_ctx.mean())


# ─────────────────────────────────────────────────────────────────────────────
# Per-seed eval
# ─────────────────────────────────────────────────────────────────────────────

def run_eval_seed(seed, cfg_eval, record_mode):
    agent_name    = cfg_eval.get('agent_name',    'ddpm_encdec_vision')
    agent_cfg_grp = cfg_eval.get('agent_cfg_group', f'{agent_name}_agent')
    ckpt_root     = cfg_eval.get('checkpoint_root', 'logs/d3il_visual_aligning_baseline')
    d3il_cfg_dir  = cfg_eval.get('d3il_config_dir', 'd3il/configs')
    n_contexts    = int(cfg_eval.get('n_contexts', 3))
    n_trajs       = int(cfg_eval.get('n_trajectories_per_context', 1))
    eval_on_train = bool(cfg_eval.get('eval_on_train', False))
    max_ep_len    = int(cfg_eval.get('max_episode_length', 400))
    if_vision     = bool(cfg_eval.get('if_vision', True))
    device        = cfg_eval.get('device', 'cuda')

    ckpt_dir  = os.path.join(ckpt_root, agent_name, f'seed_{seed}', 'weights')
    save_path = os.path.join(ckpt_root, agent_name, f'seed_{seed}')
    os.makedirs(save_path, exist_ok=True)

    print(f'\n{"=" * 72}')
    print(f'[ D3IL Baseline Eval ] agent={agent_name}  seed={seed}')
    print(f'  contexts={n_contexts}  trajs_per_ctx={n_trajs}  '
          f'eval_on_train={eval_on_train}  if_vision={if_vision}')
    print(f'  checkpoint_dir: {ckpt_dir}')
    print(f'  save_path:      {save_path}')
    print(f'{"=" * 72}')

    # ── build agent ────────────────────────────────────────────────────────────
    # base_agent.__init__ calls wandb.log() unconditionally; init disabled so it doesn't crash
    if not wandb.run:
        wandb.init(mode="disabled")
    agent = build_agent(d3il_cfg_dir, agent_cfg_grp, device, seed)

    # load best eval checkpoint; fall back to last
    ckpt_name = getattr(agent, 'eval_model_name', 'eval_best_ddpm.pth')
    ckpt_file = os.path.join(ckpt_dir, ckpt_name)
    if not os.path.exists(ckpt_file):
        last_name = getattr(agent, 'last_model_name', 'last_ddpm.pth')
        ckpt_file = os.path.join(ckpt_dir, last_name)
        print(f'[ WARNING ] eval_best not found — trying last: {ckpt_file}')
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError(f'No checkpoint found in {ckpt_dir}')

    agent.load_pretrained_model(ckpt_dir, sv_name=os.path.basename(ckpt_file))
    print(f'[ checkpoint ] loaded: {ckpt_file}')

    wrapper = D3ILBaselineWrapper(agent, save_path=save_path, record_mode=record_mode)

    ctx_pool = _TRAIN_CTX if eval_on_train else _TEST_CTX
    contexts = list(range(n_contexts))

    # ── environment ────────────────────────────────────────────────────────────
    env = Robot_Push_Env(render=False, if_vision=if_vision,
                         max_steps_per_episode=max_ep_len)
    env.start()

    np.random.seed(seed)
    torch.manual_seed(seed)

    # ── rollout loop ───────────────────────────────────────────────────────────
    for ctx_idx in contexts:
        for traj_i in range(n_trajs):
            print(f'\n[ rollout {wrapper._rollout_idx} ] '
                  f'context={ctx_idx}  traj={traj_i}')
            wrapper.reset()
            wrapper.record_context_info(ctx_pool[ctx_idx], ctx_idx)

            obs  = env.reset(random=False, context=ctx_pool[ctx_idx])
            done = False
            # REAL_TIME_RECORDING_UPDATE — one recorder per rollout (no projector → proj=0).
            rt_rec = RTRecorder(episode_id=f'{agent_name}_ctx{ctx_idx}_traj{traj_i}',
                                variant='baseline', scene='aligning',
                                system='D3IL_VisualAligning_baseline',
                                control_hz=RT_CONTROL_HZ, batch_size=1,
                                horizon=getattr(agent, 'horizon', 1), text_log=True)
            rt_k = 0   # REAL_TIME_RECORDING_UPDATE — per-rollout step counter

            if if_vision:
                env_state, bp_image, inhand_image = obs
                bp_proc  = bp_image.transpose((2, 0, 1)).copy()  / 255.
                inh_proc = inhand_image.transpose((2, 0, 1)).copy() / 255.
                # des_robot_pos tracks the commanded EE position (running delta sum)
                des_robot_pos = env_state[:3].copy()

                while not done:
                    frame_bgr = bp_image   # capture before step

                    _rt_t0 = _time.time()   # REAL_TIME_RECORDING_UPDATE
                    pred_action = agent.predict(
                        (bp_proc, inh_proc, des_robot_pos), if_vision=True
                    )
                    _rt_ms = (_time.time() - _rt_t0) * 1e3   # REAL_TIME_RECORDING_UPDATE — pure agent inference (no projector)
                    pred_xyz = pred_action[0] + des_robot_pos
                    rt_rec.step(t=rt_k / RT_CONTROL_HZ, total_ms=_rt_ms,   # REAL_TIME_RECORDING_UPDATE
                                obs=des_robot_pos, action=np.asarray(pred_action[0]).reshape(-1),
                                pos=np.asarray(des_robot_pos).reshape(-1)[:2],
                                proj_active=False, step_idx=rt_k)
                    rt_k += 1
                    obs, reward, done, info = env.step(
                        np.concatenate((pred_xyz, [0, 1, 0, 0]))
                    )

                    wrapper.record_step(
                        des_robot_pos, frame_bgr,
                        float(info.get('mean_distance', 0.0)),
                        infer_ms=_rt_ms,
                    )

                    des_robot_pos       = pred_xyz[:3]
                    robot_pos, bp_image, inhand_image = obs
                    bp_proc  = bp_image.transpose((2, 0, 1)).copy()  / 255.
                    inh_proc = inhand_image.transpose((2, 0, 1)).copy() / 255.

            else:
                # state-only: obs is the full state vector
                pred_action = env.robot_state()
                while not done:
                    obs_full = np.concatenate((pred_action[:3], obs))
                    _rt_t0 = _time.time()   # REAL_TIME_RECORDING_UPDATE
                    pred_action = agent.predict(obs_full)
                    _rt_ms = (_time.time() - _rt_t0) * 1e3   # REAL_TIME_RECORDING_UPDATE — pure agent inference (no projector)
                    pred_xyz    = pred_action[0] + obs_full[:3]
                    obs, reward, done, info = env.step(
                        np.concatenate((pred_xyz, [0, 1, 0, 0]))
                    )
                    rt_rec.step(t=rt_k / RT_CONTROL_HZ, total_ms=_rt_ms,   # REAL_TIME_RECORDING_UPDATE
                                obs=obs_full[:6], action=np.asarray(pred_action[0]).reshape(-1),
                                pos=np.asarray(pred_xyz).reshape(-1)[:2], proj_active=False,
                                track_err=float(info.get('mean_distance', 0.0)), step_idx=rt_k)
                    rt_k += 1
                    wrapper.record_step(
                        pred_xyz, None,
                        float(info.get('mean_distance', 0.0)),
                        infer_ms=_rt_ms,
                    )

            # REAL_TIME_RECORDING_UPDATE — write per-rollout realtime baseline log + SUMMARY.
            rt_rec.save(os.path.join(save_path, f'realtime_baseline_ctx{ctx_idx}_traj{traj_i}.log'),
                        behaviour={'success': int(bool(info.get('success', False))),
                                   'mean_distance': round(float(info.get('mean_distance', 0.0)), 4)})

            # UF-16.4 hook: read final box state directly from MuJoCo
            _fbox_pos  = env.scene.get_obj_pos(env.push_box)
            _fbox_quat = env.scene.get_obj_quat(env.push_box)
            wrapper.update_rollout_info({
                **info,
                'context':        ctx_idx,
                'final_box_pos':  _fbox_pos,
                'final_box_quat': _fbox_quat,
            })

    # ── per-seed summary ───────────────────────────────────────────────────────
    all_rec      = list(wrapper.master_history.values())
    successes    = [r['success']       for r in all_rec]
    mean_dists   = [r['mean_distance'] for r in all_rec]
    modes        = [r['mode']          for r in all_rec]
    steps        = [r['steps']         for r in all_rec]
    final_xydist = [r['context_info'].get('final_xy_dist',       float('nan')) for r in all_rec]
    final_angles = [r['context_info'].get('final_box_angle_deg', float('nan')) for r in all_rec]

    success_rate = float(np.mean(successes))
    # U2: paper-faithful behavior entropy (matches d3il aligning_sim.py:178-194 / paper Eq. 2).
    entropy = compute_behavior_entropy(all_rec, n_contexts, n_trajs, n_modes=2)
    score   = 0.5 * (success_rate + entropy)   # the paper's headline 'score'
    results = {
        'agent_name':           agent_name,
        'seed':                 int(seed),
        'n_rollouts':           len(all_rec),
        'n_contexts':           int(n_contexts),
        'n_trajectories_per_context': int(n_trajs),
        'success_rate':         success_rate,
        'entropy':              entropy,            # ← U2: the paper diversity metric
        'score':                score,              # ← U2: 0.5*(success_rate + entropy)
        'mean_distance_mean':   float(np.nanmean(mean_dists)),
        'mean_distance_std':    float(np.nanstd(mean_dists)),
        'final_xy_dist_mean':   float(np.nanmean(final_xydist)),
        'final_xy_dist_std':    float(np.nanstd(final_xydist)),
        'final_angle_deg_mean': float(np.nanmean(final_angles)),
        'final_angle_deg_std':  float(np.nanstd(final_angles)),
        'n_steps_mean':         float(np.mean(steps)),
        'n_steps_std':          float(np.std(steps)),
        'mode_0_rate':          float(sum(m == 0 for m in modes) / max(len(modes), 1)),
        # U4.3 — seconds per control step, averaged over rollouts (see the module
        # docstring on timing semantics before comparing this to Gen14 avg_time).
        'avg_time_mean':        float(np.nanmean([r['avg_time'] for r in all_rec])),
        'avg_time_std':         float(np.nanstd([r['avg_time'] for r in all_rec])),
    }

    print(f'\n{"─" * 60}')
    print(f'[ Seed {seed} Summary ]  ({n_contexts} ctx × {n_trajs} traj = {len(all_rec)} rollouts)')
    print(f'  success_rate:       {success_rate:.3f}')
    print(f'  entropy:            {entropy:.3f}')
    print(f'  score (0.5·SR+H):   {score:.3f}')
    print(f'  mean_distance:      {results["mean_distance_mean"]:.4f} ± {results["mean_distance_std"]:.4f}')
    print(f'  final_xy_dist:      {results["final_xy_dist_mean"]:.4f} ± {results["final_xy_dist_std"]:.4f} m')
    print(f'  n_steps:            {results["n_steps_mean"]:.1f} ± {results["n_steps_std"]:.1f}')
    print(f'  mode_0_rate:        {results["mode_0_rate"]:.3f}')
    if n_trajs < 8:
        print(f'  [ note ] entropy needs many trajs/context to be meaningful '
              f'(paper uses 18); current = {n_trajs}.')
    print(f'{"─" * 60}')

    result_file = os.path.join(save_path, f'results_seed_{seed}.json')
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f'[ saved ] {result_file}')

    # ── U4.3: DA_VA_v2-native export ──────────────────────────────────────────
    if cfg_eval.get('da_export', True):
        try:
            export_da_unit(cfg_eval, seed, agent_name, all_rec, results,
                           n_contexts, n_trajs, eval_on_train)
        except Exception as e:                                    # noqa: BLE001
            # The DA copy is a convenience; never let it lose a finished eval.
            print(f'[ WARNING ] DA export failed (results above are unaffected): {e}')

    return results


# ─────────────────────────────────────────────────────────────────────────────
# U4.3 — DA_VA_v2 export
# ─────────────────────────────────────────────────────────────────────────────

def export_da_unit(cfg_eval, seed, agent_name, all_rec, results,
                   n_contexts, n_trajs, eval_on_train):
    """Write this seed's results in the shape `Data_Analysis/DA_VA_v2` reads.

    Same schema module the legacy bridge uses, so a bridged old run and a fresh
    run are byte-compatible on the DA side — the only difference is the root
    folder name (`DA_VA_d3il_baseline/` here, `_DA_VA_BRIDGE_…/` there).
    """
    root = os.path.join(
        cfg_eval.get('save_root',
                     cfg_eval.get('checkpoint_root', 'logs/d3il_visual_aligning_baseline')),
        cfg_eval.get('da_export_root_name', da_api.DA_NATIVE_ROOT_NAME))
    split = 'train' if eval_on_train else 'test'

    records = []
    for ridx, rec in enumerate(all_rec):
        ctx = int(rec.get('context', rec['context_info'].get('context_idx', ridx // max(n_trajs, 1))))
        records.append(da_api.make_record(
            rollout_index=ridx,
            context_idx=ctx,
            traj_idx=ridx % max(n_trajs, 1),
            success=rec['success'],
            steps=rec['steps'],
            mean_distance=rec['mean_distance'],
            mode=rec['mode'],
            context_info=rec['context_info'],
            avg_time_s=rec['avg_time'],
        ))

    scalars = da_api.summarise(records, n_contexts, n_trajs)
    out_dir = da_api.write_unit(
        root, agent_name, seed, records, scalars, split=split,
        args_extra={
            'export_source': 'native_eval',
            'n_contexts': n_contexts,
            'n_trajectories_per_context': n_trajs,
            'eval_on_train': bool(eval_on_train),
            'max_episode_length': int(cfg_eval.get('max_episode_length', 400)),
            'device': cfg_eval.get('device', 'cuda'),
            'agent_cfg_group': cfg_eval.get('agent_cfg_group', ''),
        })
    print(f'[ DA export ] {out_dir}')
    print(f'[ DA export ] read it with: python Data_Analysis/DA_VA_v2/main_da_batch.py '
          f'--parent-path {root}')

    # The scalars here are computed by the shared module; if they disagree with
    # the eval's own numbers something is wrong with one of the two paths.
    if abs(scalars['success_rate'] - results['success_rate']) > 1e-9 or \
            abs(scalars['entropy'] - results['entropy']) > 1e-9:
        print(f'[ WARNING ] DA export scalars differ from the eval summary — '
              f'export(success={scalars["success_rate"]:.4f}, '
              f'entropy={scalars["entropy"]:.4f}) vs '
              f'eval(success={results["success_rate"]:.4f}, '
              f'entropy={results["entropy"]:.4f})')
    return out_dir


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description='D3IL visual aligning baseline eval')
    p.add_argument('--config', default='d3il_visual_aligning_baseline_test/d3il_eval_config.yaml',
                   help='path to eval YAML config')
    p.add_argument('--agent-name',  default=None, help='override agent_name in config')
    p.add_argument('--seed',        type=int, default=None, help='single seed override')
    p.add_argument('--n-contexts',  type=int, default=None, help='override n_contexts')
    p.add_argument('--n-trajectories', type=int, default=None,
                   help='override n_trajectories_per_context (paper uses 18; needed for entropy)')
    p.add_argument('--paper', action='store_true',
                   help='paper-faithful eval preset: n_contexts=60, n_trajectories_per_context=18')
    p.add_argument('--record',      default=None, choices=['all', 'gif', 'video', 'none'],
                   help='recording mode override')
    p.add_argument('--eval-on-train', action='store_true', help='eval on training contexts')
    p.add_argument('--device',      default=None, help='cuda or cpu override')
    p.add_argument('--no-da-export', action='store_true',
                   help='skip the DA_VA_v2-shaped copy of the results (U4.3). '
                        'The copy is additive and cheap; only skip it if the '
                        'DA tool is not wanted for this run.')
    p.add_argument('--da-export-root', default=None,
                   help=f'folder name for the DA export, under save_root '
                        f'(default: {da_api.DA_NATIVE_ROOT_NAME}). Do NOT start it '
                        f'with "_" — that prefix means "legacy/bridged" to DA_VA_v2.')
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # CLI overrides
    if args.agent_name:
        cfg['agent_name']      = args.agent_name
        cfg['agent_cfg_group'] = f'{args.agent_name}_agent'  # derive Hydra group from name
    if args.paper:                                  # U2: paper-faithful eval scale
        cfg['n_contexts'] = 60
        cfg['n_trajectories_per_context'] = 18
        print('[ eval ] --paper preset: n_contexts=60, n_trajectories_per_context=18')
    if args.n_contexts is not None:
        cfg['n_contexts'] = args.n_contexts
    if args.n_trajectories is not None:
        cfg['n_trajectories_per_context'] = args.n_trajectories
    if args.record is not None:
        cfg['record_mode'] = args.record
    if args.eval_on_train:
        cfg['eval_on_train'] = True
    if args.device:
        cfg['device'] = args.device
    if args.no_da_export:
        cfg['da_export'] = False
    if args.da_export_root:
        cfg['da_export_root_name'] = args.da_export_root

    record_mode = cfg.get('record_mode', 'all')

    seeds = [args.seed] if args.seed is not None else cfg.get('seeds', [42])

    all_results = []
    for seed in seeds:
        res = run_eval_seed(seed, cfg, record_mode)
        all_results.append(res)

    # ── cross-seed aggregate ───────────────────────────────────────────────────
    if len(all_results) > 1:
        agg_keys = [
            'success_rate', 'entropy', 'score', 'mean_distance_mean', 'final_xy_dist_mean',
            'final_angle_deg_mean', 'n_steps_mean', 'mode_0_rate',
        ]
        aggregate = {
            'agent_name': all_results[0]['agent_name'],
            'seeds':      [r['seed'] for r in all_results],
            'n_seeds':    len(all_results),
        }
        for k in agg_keys:
            vals = [r[k] for r in all_results]
            aggregate[f'{k}_across_seeds_mean'] = float(np.mean(vals))
            aggregate[f'{k}_across_seeds_std']  = float(np.std(vals))

        agg_dir  = os.path.join(cfg.get('save_root', 'logs/d3il_visual_aligning_baseline'),
                                cfg.get('agent_name', 'ddpm_encdec_vision'))
        agg_file = os.path.join(agg_dir, 'aggregate_results.json')
        os.makedirs(agg_dir, exist_ok=True)
        with open(agg_file, 'w') as f:
            json.dump(aggregate, f, indent=4)
        print(f'\n[ saved aggregate ] {agg_file}')

        print(f'\n{"=" * 72}')
        print(f'[ Aggregate ({len(seeds)} seeds) ]')
        for k in agg_keys:
            m = aggregate[f'{k}_across_seeds_mean']
            s = aggregate[f'{k}_across_seeds_std']
            print(f'  {k:<30} {m:.4f} ± {s:.4f}')
        print(f'{"=" * 72}')


if __name__ == '__main__':
    main()
