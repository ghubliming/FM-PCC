"""Epoch-3 driver: load X2 inside a scene, run a task, log + render.

Usage:
    python uav_env_test/run_env.py --scene {empty|corridor|s_curve|pillars} \
        --task {A|B|C|traverse|s_curve|weave} \
        [--render] [--seed N]

Output:
    logs/uav_env/<scene>_<task_label>/
        log.json     — per-step state, target, control, contact-count
        metrics.txt  — RMS / max / final pos err + contact summary
        rollout.gif  — if --render
        controller.txt — PID diagnostic dump

Scenes are wrappers around Epoch 1's quadrotor_modified.xml, adding
floor + lighting + skybox + (per scene) walls or pillars. Controller and
trajectory math are unchanged from Epoch 2 — only the world differs.
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


SCENES_DIR = os.path.join(
    _REPO,
    'd3il/environments/d3il/models/mj/robot/quadrotor/scenes',
)
SCENES = {
    'empty':    'scene_empty.xml',
    'corridor': 'scene_corridor.xml',
    's_curve':  'scene_s_curve.xml',
    'pillars':  'scene_pillars.xml',
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--scene', required=True, choices=list(SCENES))
    p.add_argument('--task', required=True,
                   choices=['A', 'B', 'C', 'traverse', 's_curve', 'weave'])
    p.add_argument('--trajectory-format', default='9D', choices=['6D', '9D'])
    p.add_argument('--render', action='store_true', default=False)
    p.add_argument('--render-stride', type=int, default=10)
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
    if name == 'traverse':
        # Fly through the corridor: enter from x=-2.5, exit at x=+2.5, altitude 0.75.
        return {
            'label': 'task_traverse',
            'init_pos': np.array([-2.5, 0.0, 0.75]),
            'duration': 8.0,
            'traj': trajs.traverse_line([-2.5, 0.0, 0.75], [2.5, 0.0, 0.75], duration=8.0),
        }
    if name == 's_curve':
        # 5-leg Z-route: pure-x into gap, pure-y across centerline, pure-x out.
        # Fix_2: old 3-leg diagonal (-0.5,-0.8)→(+0.5,+0.8) passed 0.019 m inside
        # the rotor-contact zone of both gap-side wall corners — geometrically infeasible.
        # Z-route clearance: 0.50 m from both corners on every leg (rotor reach = 0.31 m).
        return {
            'label': 'task_s_curve',
            'init_pos': np.array([-3.0, -0.8, 0.75]),
            'duration': 15.0,
            'traj': trajs.s_curve_path(
                waypoints=[(-3.0, -0.8, 0.75),
                           (-0.5, -0.8, 0.75),
                           ( 0.0, -0.8, 0.75),
                           ( 0.0,  0.8, 0.75),
                           ( 0.5,  0.8, 0.75),
                           ( 3.0,  0.8, 0.75)],
                segment_duration=3.0,
            ),
        }
    if name == 'weave':
        # Sinusoidal weave through the pillar field (3 pillars per column at x∈{-2,0,2}).
        return {
            'label': 'task_weave',
            'init_pos': np.array([-3.0, 0.0, 1.0]),
            'duration': 10.0,
            'traj': trajs.weave(
                x_range=(-3.0, 3.0),
                y_amplitude=1.0,
                period=4.0,
                altitude=1.0,
                duration=10.0,
            ),
        }
    raise ValueError(name)


def _resolve_camera(model):
    for name in ('fpv', 'track'):
        for i in range(model.ncam):
            if model.camera(i).name == name:
                return name
    return -1


def _is_drone_contact(model, contact):
    """Heuristic: any contact where neither geom is the floor counts as
    a 'drone vs scene obstacle' strike. Floor contact is fine (landing)."""
    g1, g2 = contact.geom1, contact.geom2
    n1 = model.geom(g1).name
    n2 = model.geom(g2).name
    if n1 == 'floor' or n2 == 'floor':
        return False
    return True


def run():
    args = parse_args()
    np.random.seed(args.seed)

    xml = os.path.join(SCENES_DIR, SCENES[args.scene])
    if not os.path.exists(xml):
        print(f'[ run_env ] ERROR: scene XML not found at {xml}', file=sys.stderr)
        sys.exit(1)

    model = mujoco.MjModel.from_xml_path(xml)
    data = mujoco.MjData(model)

    task = build_task(args.task)
    label = f'{args.scene}_{task["label"]}'
    if args.task == 'C':
        label = f'{label}_{args.trajectory_format}'

    out_dir = args.out_dir or os.path.join(_REPO, 'logs/uav_env', label)
    os.makedirs(out_dir, exist_ok=True)

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

    print(f'[ run_env ] scene={args.scene} task={args.task} fmt={args.trajectory_format} '
          f'dt={dt:.5f} steps={n_steps} duration={task["duration"]:.2f}s '
          f'out={out_dir}')

    log = []
    frames = []
    renderer = None
    cam_name_or_id = -1
    if args.render:
        renderer = mujoco.Renderer(model, width=480, height=480)
        cam_name_or_id = _resolve_camera(model)

    obstacle_contact_steps = 0

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

        # Contact accounting (don't abort — log only; floor contacts ignored)
        obstacle_hits = 0
        for ci in range(data.ncon):
            if _is_drone_contact(model, data.contact[ci]):
                obstacle_hits += 1
        if obstacle_hits > 0:
            obstacle_contact_steps += 1

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
            'n_obstacle_contacts': obstacle_hits,
        })

        if args.render and (k % args.render_stride == 0):
            renderer.update_scene(data, camera=cam_name_or_id)
            frames.append(renderer.render())

    with open(os.path.join(out_dir, 'log.json'), 'w') as f:
        json.dump(log, f)

    pos_err = np.array([
        np.linalg.norm(np.array(s['p']) - np.array(s['p_des'])) for s in log
    ])
    metrics = {
        'scene': args.scene,
        'task': args.task,
        'trajectory_format': args.trajectory_format,
        'duration_s': task['duration'],
        'n_steps': n_steps,
        'dt': dt,
        'final_pos_err_m': float(pos_err[-1]),
        'mean_pos_err_m': float(pos_err.mean()),
        'rms_pos_err_m': float(np.sqrt((pos_err ** 2).mean())),
        'max_pos_err_m': float(pos_err.max()),
        'obstacle_contact_steps': obstacle_contact_steps,
        'obstacle_contact_fraction': obstacle_contact_steps / max(n_steps, 1),
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
        print(f'[ run_env ] saved {len(frames)} frames → rollout.gif')

    if obstacle_contact_steps > 0:
        print(f'[ run_env ] WARN: drone touched scene obstacles on '
              f'{obstacle_contact_steps}/{n_steps} steps '
              f'({100*metrics["obstacle_contact_fraction"]:.1f}%) — '
              f'trajectory does not clear obstacles')

    print(f'[ run_env ] DONE.  '
          f'final_pos_err={metrics["final_pos_err_m"]:.4f} m  '
          f'rms={metrics["rms_pos_err_m"]:.4f} m  '
          f'max={metrics["max_pos_err_m"]:.4f} m  '
          f'obstacle_contacts={obstacle_contact_steps}')


if __name__ == '__main__':
    run()
