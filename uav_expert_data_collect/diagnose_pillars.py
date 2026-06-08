"""Diagnostic script for pillar contact analysis — E4 U3.

Runs 8 pillar episodes (2 per homotopy, seeds 0-7) WITHOUT rejection,
and reports: where contact happens, what geoms, drone y at time of contact.

Usage:
    python uav_expert_data_collect/diagnose_pillars.py
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import mujoco
from uav_expert_data_collect.generator import (
    SCENE_XMLS, _build_traj_and_init, _make_pid, _is_obstacle_contact,
    SCENE_MAX_CONTACT_FRACTION,
)

HOMOTOPIES = ['(L,L,L)', '(L,R,L)', '(R,L,R)', '(R,R,R)']
SEEDS_PER  = 2

def run_diag(scene, homotopy, seed):
    rng   = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(SCENE_XMLS[scene])
    data  = mujoco.MjData(model)

    traj_fn, init_pos, dur = _build_traj_and_init(scene, homotopy, rng)
    data.qpos[:3]  = init_pos
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:]   = 0.0
    mujoco.mj_forward(model, data)

    pid    = _make_pid(model, 'pid_default')
    dt     = float(model.opt.timestep)
    n_step = int(round(dur / dt))

    contacts = []
    for k in range(n_step):
        t = k * dt
        p_des, v_des, a_des, yaw_des = traj_fn(t)
        p  = data.qpos[:3].copy()
        v  = data.qvel[:3].copy()
        q  = data.qpos[3:7].copy()
        om = data.qvel[3:6].copy()

        u = pid.compute(p, q, v, om, p_des, v_des, a_des, yaw_des)
        data.ctrl[:4] = u
        mujoco.mj_step(model, data)

        for ci in range(data.ncon):
            c  = data.contact[ci]
            n1 = model.geom(c.geom1).name
            n2 = model.geom(c.geom2).name
            if n1 != 'floor' and n2 != 'floor':
                p_actual = data.qpos[:3].copy()
                p_des_arr = np.asarray(p_des, dtype=float)
                contacts.append({
                    't':      t,
                    'k':      k,
                    'geom1':  n1,
                    'geom2':  n2,
                    'p':      p_actual.tolist(),
                    'p_des':  p_des_arr.tolist(),
                    'y_err':  float(p_actual[1] - p_des_arr[1]),
                })

    contact_frac = len(set(c['k'] for c in contacts)) / n_step
    return contacts, contact_frac, dur


def main():
    print(f'{"="*70}')
    print(f'Pillar contact diagnostic  (threshold would reject > {SCENE_MAX_CONTACT_FRACTION["pillars"]:.3f})')
    print(f'{"="*70}')

    for homotopy in HOMOTOPIES:
        for seed in range(SEEDS_PER):
            contacts, frac, dur = run_diag('pillars', homotopy, seed)
            rejected = frac > SCENE_MAX_CONTACT_FRACTION['pillars']
            status   = '❌ REJECT' if rejected else '✅ PASS'
            print(f'\n{homotopy}  seed={seed}  dur={dur:.1f}s  '
                  f'contact_frac={frac:.4f}  {status}')

            if not contacts:
                print('  No contacts detected.')
                continue

            # Summarise: group by contact geom pair
            from collections import defaultdict
            by_geom = defaultdict(list)
            for c in contacts:
                key = f"{c['geom1']} ↔ {c['geom2']}"
                by_geom[key].append(c)

            for pair, cs in by_geom.items():
                ys    = [c['p'][1] for c in cs]
                ts    = [c['t'] for c in cs]
                y_err = [c['y_err'] for c in cs]
                print(f'  {pair}: {len(cs)} contact steps  '
                      f't=[{min(ts):.2f}..{max(ts):.2f}]  '
                      f'y=[{min(ys):.3f}..{max(ys):.3f}]  '
                      f'y_err=[{min(y_err):.3f}..{max(y_err):.3f}]')

    print(f'\n{"="*70}')


if __name__ == '__main__':
    main()
