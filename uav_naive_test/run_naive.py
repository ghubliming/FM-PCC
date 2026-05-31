"""Epoch-2 driver: load X2 + run a naive task under CascadedPID, log + render.

Usage:
    python uav_naive_test/run_naive.py --task {A|B|C} \
        [--trajectory-format {6D|9D}] [--render] [--seed N]

Output:
    logs_in_develop/Gen11/Epoch2_env/results/<label>/
        log.json     — per-step state, target, control
        metrics.txt  — RMS / max / final position errors
        rollout.gif  — if --render
        controller.txt — PID diagnostic dump

Task → label:
    A → task_A_hover
    B → task_B_step
    C → task_C_circle_{6D|9D}
"""

import argparse
import json
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS_DIR, '..'))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, _REPO)

import mujoco

from flight_controller import CascadedPID, diagnostic_string
import trajectories as trajs

XML_PATH = os.path.join(
    _REPO,
    'd3il/environments/d3il/models/mj/robot/quadrotor/quadrotor_modified.xml',
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task', required=True, choices=['A', 'B', 'C'])
    p.add_argument('--trajectory-format', default='9D', choices=['6D', '9D'])
    p.add_argument('--render', action='store_true', default=False)
    p.add_argument('--render-stride', type=int, default=10,
                   help='Render every Nth physics step (default: every 10).')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out-dir', default=None)
    return p.parse_args()


def build_task(name):
    if name == 'A':
        return {
            'label': 'task_A_hover',
            'init_pos': np.array([0.0, 0.0, 0.3]),
            'duration': 5.0,
            'traj': trajs.hover_at([0.0, 0.0, 0.5]),
        }
    if name == 'B':
        return {
            'label': 'task_B_step',
            'init_pos': np.array([0.0, 0.0, 0.5]),
            'duration': 6.0,
            'traj': trajs.step_to([0, 0, 0.5], [1, 0, 0.5], t_step=2.0),
        }
    if name == 'C':
        return {
            'label': 'task_C_circle',
            'init_pos': np.array([0.5, 0.0, 0.75]),
            'duration': 30.0,
            'traj': trajs.circle([0.0, 0.0], radius=0.5, period=10.0, altitude=0.75),
        }
    raise ValueError(name)


def _resolve_camera(model):
    for i in range(model.ncam):
        if model.camera(i).name == 'track':
            return 'track'
    return -1  # free camera fallback


def run():
    args = parse_args()
    np.random.seed(args.seed)

    if not os.path.exists(XML_PATH):
        print(f'[ run_naive ] ERROR: model XML not found at {XML_PATH}', file=sys.stderr)
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)

    task = build_task(args.task)
    label = task['label']
    if args.task == 'C':
        label = f'{label}_{args.trajectory_format}'

    out_dir = args.out_dir or os.path.join(
        _REPO, 'logs_in_develop/Gen11/Epoch2_env/results', label)
    os.makedirs(out_dir, exist_ok=True)

    # ── initial state ────────────────────────────────────────────────────
    data.qpos[:3] = task['init_pos']
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    controller = CascadedPID(model)
    with open(os.path.join(out_dir, 'controller.txt'), 'w') as f:
        f.write(diagnostic_string(controller) + '\n')

    dt = float(model.opt.timestep)
    n_steps = int(round(task['duration'] / dt))
    use_a = (args.trajectory_format == '9D')

    print(f'[ run_naive ] task={args.task} fmt={args.trajectory_format} '
          f'dt={dt:.5f} steps={n_steps} duration={task["duration"]:.2f}s '
          f'out={out_dir}')

    log = []
    frames = []
    renderer = None
    cam_name_or_id = -1
    if args.render:
        renderer = mujoco.Renderer(model, width=480, height=480)
        cam_name_or_id = _resolve_camera(model)

    for k in range(n_steps):
        t = k * dt
        p_des, v_des, a_des, yaw_des = task['traj'](t)

        p = data.qpos[:3].copy()
        q = data.qpos[3:7].copy()
        v = data.qvel[:3].copy()
        omega_body = data.qvel[3:6].copy()

        u = controller.compute(
            p, q, v, omega_body,
            p_des, v_des,
            a_des if use_a else None,
            yaw_des,
        )
        data.ctrl[:4] = u

        mujoco.mj_step(model, data)

        log.append({
            't': t,
            'p': p.tolist(),
            'q': q.tolist(),
            'v': v.tolist(),
            'omega': omega_body.tolist(),
            'p_des': p_des.tolist(),
            'v_des': v_des.tolist(),
            'a_des': a_des.tolist(),
            'u': u.tolist(),
        })

        if args.render and (k % args.render_stride == 0):
            renderer.update_scene(data, camera=cam_name_or_id)
            frames.append(renderer.render())

    with open(os.path.join(out_dir, 'log.json'), 'w') as f:
        json.dump(log, f)

    # ── metrics ──────────────────────────────────────────────────────────
    pos_err = np.array([
        np.linalg.norm(np.array(s['p']) - np.array(s['p_des'])) for s in log
    ])
    metrics = {
        'task': args.task,
        'trajectory_format': args.trajectory_format if args.task == 'C' else 'N/A',
        'duration_s': task['duration'],
        'n_steps': n_steps,
        'dt': dt,
        'final_pos_err_m': float(pos_err[-1]),
        'mean_pos_err_m': float(pos_err.mean()),
        'rms_pos_err_m': float(np.sqrt((pos_err ** 2).mean())),
        'max_pos_err_m': float(pos_err.max()),
    }
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as f:
        for k, v in metrics.items():
            f.write(f'{k}: {v}\n')

    if args.render and frames:
        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio
        imageio.mimsave(os.path.join(out_dir, 'rollout.gif'), frames, fps=10)
        print(f'[ run_naive ] saved {len(frames)} frames → rollout.gif')

    print(f'[ run_naive ] DONE.  '
          f'final_pos_err={metrics["final_pos_err_m"]:.4f} m  '
          f'rms={metrics["rms_pos_err_m"]:.4f} m  '
          f'max={metrics["max_pos_err_m"]:.4f} m')


if __name__ == '__main__':
    run()
