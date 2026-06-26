"""E4 U10 — Stress-test episode collection driver.

Generates deliberately *bad* episodes into a SEPARATE folder for visual inspection.
Never mixed with production training data.  DEFAULT: does nothing (requires --cases).

Output
------
logs/uav_expert_data_stress/<scene>/<stress_case>/<episode_id>.pkl
logs/uav_expert_data_stress/stress_summary.json

Usage
-----
# All cases, all applicable scenes
python uav_expert_data_collect/collect_stress.py --cases all

# Specific cases
python uav_expert_data_collect/collect_stress.py --cases wall_crossing,floor_dive

# Single scene
python uav_expert_data_collect/collect_stress.py --cases extreme_speed --scene pillars

# More episodes per case
python uav_expert_data_collect/collect_stress.py --cases all --n-per-case 5 --seed 0
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from uav_expert_data_collect.generator import (
    SCENE_XMLS, SCENE_OBSTACLES,
    SCENE_MAX_CONTACT_FRACTION, MAX_CONTACT_FRACTION, Z_FLOOR_MARGIN,
    _is_obstacle_contact,
)
from uav_expert_data_collect.dataset_writer import rollout_to_episode, save_episode
from uav_expert_data_collect.stress_trajectories import (
    build_stress_traj, STRESS_CASES, STRESS_CASE_SCENES,
)
from uav_env_test.flight_controller import CascadedPID

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DEFAULT_OUT = os.path.join(_REPO, 'logs', 'uav_expert_data_stress')
_STRESS_BASE_SEED = 700000  # well-separated from production seeds


def _make_pid(model, kp_scale=1.0, kd_scale=1.0):
    pid = CascadedPID(model)
    pid.Kp_pos = pid.Kp_pos * kp_scale
    pid.Kd_pos = pid.Kd_pos * kd_scale
    return pid


def run_stress_episode(scene, case, seed):
    """Run one stress rollout.  No rejection — always returns a dict."""
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(SCENE_XMLS[scene])
    data  = mujoco.MjData(model)

    traj_fn, init_pos, dur, extras = build_stress_traj(case, scene, rng)

    data.qpos[:3]  = init_pos
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:]   = 0.0
    mujoco.mj_forward(model, data)

    kp_scale = extras.get('kp_scale', 1.0)
    kd_scale = extras.get('kd_scale', 1.0)
    pid = _make_pid(model, kp_scale, kd_scale)

    dt     = float(model.opt.timestep)
    n_step = int(round(dur / dt))
    steps  = []
    n_hit  = 0
    n_clip = 0

    for k in range(n_step):
        t = k * dt
        p_des, v_des, a_des, yaw_des = traj_fn(t)

        p  = data.qpos[:3].copy()
        v  = data.qvel[:3].copy()
        q  = data.qpos[3:7].copy()
        om = data.qvel[3:6].copy()

        u = pid.compute(p, q, v, om, p_des, v_des, a_des, yaw_des)
        n_clip += int(pid.last_raw_saturated)
        data.ctrl[:4] = u
        mujoco.mj_step(model, data)

        hit = any(_is_obstacle_contact(model, data.contact[ci])
                  for ci in range(data.ncon))
        n_hit += int(hit)
        steps.append({'p': p, 'v': v,
                      'p_des': np.asarray(p_des, dtype=float),
                      'q':     q.astype(np.float32)})

    contact_frac    = n_hit  / max(n_step, 1)
    motor_clip_frac = n_clip / max(n_step, 1)
    min_z = min(s['p'][2] for s in steps) if steps else 0.0

    # Compute what production gates WOULD say (do not actually reject).
    contact_limit = SCENE_MAX_CONTACT_FRACTION.get(scene, MAX_CONTACT_FRACTION)
    gate_verdicts = {
        'contact': contact_frac > contact_limit,
        'floor':   min_z < Z_FLOOR_MARGIN,
    }

    return {
        'steps':            steps,
        'scene':            scene,
        'homotopy':         case,   # stress_case in the homotopy slot
        'controller':       'pid_default',
        'dt_physics':       dt,
        'duration':         dur,
        'init_pos':         init_pos.tolist(),
        'obstacles':        SCENE_OBSTACLES[scene],
        'contact_fraction': contact_frac,
        'motor_clip_frac':  motor_clip_frac,
        # stress bookkeeping
        'stress':        True,
        'stress_case':   case,
        'gate_verdicts': gate_verdicts,
        'stress_extras': extras,
    }


def parse_args():
    p = argparse.ArgumentParser(description='E4 U10 stress-test episode collection')
    p.add_argument('--cases', required=True,
                   help='"all" or comma-separated subset of: ' + ', '.join(STRESS_CASES))
    p.add_argument('--scene', default=None,
                   choices=list(SCENE_XMLS),
                   help='Restrict to one scene (default: all applicable per case)')
    p.add_argument('--n-per-case', type=int, default=3,
                   help='Episodes per (case, scene) pair (default: 3)')
    p.add_argument('--seed', type=int, default=0,
                   help='Base seed offset (default: 0)')
    p.add_argument('--out-dir', default=_DEFAULT_OUT,
                   help=f'Output root (default: {_DEFAULT_OUT})')
    return p.parse_args()


def main():
    args = parse_args()

    if args.cases.strip().lower() == 'all':
        cases = list(STRESS_CASES)
    else:
        cases = [c.strip() for c in args.cases.split(',')]
        bad = [c for c in cases if c not in STRESS_CASES]
        if bad:
            print(f'[ stress ] ERROR: unknown cases: {bad}. Valid: {STRESS_CASES}')
            sys.exit(1)

    out_root = args.out_dir
    t0 = time.time()
    summary_by_case = {}
    total_saved = 0

    for case in cases:
        applicable = STRESS_CASE_SCENES[case]
        if args.scene is not None:
            if args.scene not in applicable:
                print(f'[ stress ] SKIP {case}: not applicable to scene={args.scene}')
                continue
            scenes_for_case = [args.scene]
        else:
            scenes_for_case = applicable

        for scene in scenes_for_case:
            key = f'{case}/{scene}'
            saved_here = 0
            print(f'[ stress ] {key} ({args.n_per_case} episodes) …')

            for i in range(args.n_per_case):
                seed = _STRESS_BASE_SEED + args.seed + hash((case, scene, i)) % 100000

                rollout = run_stress_episode(scene, case, seed)

                ep_id = f'stress_{case}_{scene}_{seed:07d}'

                rng2 = np.random.default_rng(seed + 99991)
                episode = rollout_to_episode(rollout, ep_id,
                                             noise_sigma=0.0, rng=rng2)
                # Merge stress fields (not produced by rollout_to_episode)
                episode['stress']        = True
                episode['stress_case']   = case
                episode['gate_verdicts'] = rollout['gate_verdicts']
                episode['metadata']['kp_scale']   = rollout['stress_extras'].get('kp_scale', 1.0)
                episode['metadata']['kd_scale']   = rollout['stress_extras'].get('kd_scale', 1.0)
                episode['metadata']['gain_variant_label'] = \
                    rollout['stress_extras'].get('gain_variant', 'default')

                ep_dir = os.path.join(out_root, scene, case)
                save_episode(episode, ep_dir)
                saved_here += 1

                gv = rollout['gate_verdicts']
                print(f'  → {ep_id}  contact={rollout["contact_fraction"]:.3f}'
                      f'  min_z={min(s["p"][2] for s in rollout["steps"]):.3f}'
                      f'  clip={rollout["motor_clip_frac"]:.2f}'
                      f'  gate[contact={gv["contact"]}, floor={gv["floor"]}]')

            total_saved += saved_here
            summary_by_case[key] = {'saved': saved_here, 'n_per_case': args.n_per_case}

    elapsed = time.time() - t0
    summary = {
        'cases': cases,
        'scene_filter': args.scene,
        'n_per_case': args.n_per_case,
        'total_saved': total_saved,
        'elapsed_s': round(elapsed, 1),
        'by_case_scene': summary_by_case,
    }
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, 'stress_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f'[ stress ] DONE  total_saved={total_saved}  elapsed={elapsed:.0f}s')
    print(f'[ stress ] Output: {out_root}/')
    print(f'[ stress ] Summary: {out_root}/stress_summary.json')


if __name__ == '__main__':
    main()
