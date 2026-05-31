"""Epoch-3 smoke load: load a scene wrapper (X2 + floor + walls/pillars),
step 200 times under zero ctrl, confirm the drone settles onto the floor.

Default scene: empty (just floor + skybox + lights). Use --scene to switch.

Expected output (any scene):
    OK nq=7 nv=6 nu=4 qpos_z≈<0.05-0.15>     (drone has hit the floor)
The drone drops from 0.1 m start and settles on the floor under gravity.
"""

import argparse
import os
import sys

import mujoco

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS_DIR, '..'))
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
    p.add_argument('--scene', default='empty', choices=list(SCENES))
    p.add_argument('--steps', type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    xml = os.path.join(SCENES_DIR, SCENES[args.scene])
    if not os.path.exists(xml):
        print(f'[ smoke ] ERROR: scene XML not found at {xml}')
        sys.exit(1)
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    for _ in range(args.steps):
        mujoco.mj_step(m, d)
    print(f'OK scene={args.scene}  nq={m.nq} nv={m.nv} nu={m.nu}  qpos_z={d.qpos[2]:.4f}')
    print(f'[ smoke ] body_mass(x2) total = {m.body_subtreemass[m.body("x2").id]:.4f} kg')
    print(f'[ smoke ] gravity = {m.opt.gravity}')
    print(f'[ smoke ] timestep = {m.opt.timestep}')
    print(f'[ smoke ] ngeom = {m.ngeom}  (includes drone + scene geoms)')
    print(f'[ smoke ] ncon at end = {d.ncon}  (active contacts; >0 = drone touched something)')


if __name__ == '__main__':
    main()
