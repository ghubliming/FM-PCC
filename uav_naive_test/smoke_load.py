"""Phase 2-α: load the Epoch 1 X2 model in MuJoCo, step 100 times, print model dims.

This is the runtime check that Epoch 1 §11.4 step 5 deferred to Slurm
(local Docker has no Python runtime per project convention).

Expected output:
    OK nq=7 nv=6 nu=4 qpos_z=<≈0.052>
The drone falls under gravity from 0.1 m start over 0.1 s.
"""

import os
import sys

import mujoco

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_THIS_DIR, '..'))
XML_PATH = os.path.join(
    _REPO,
    'd3il/environments/d3il/models/mj/robot/quadrotor/quadrotor_modified.xml',
)


def main():
    if not os.path.exists(XML_PATH):
        print(f'[ smoke ] ERROR: model not found at {XML_PATH}')
        sys.exit(1)
    m = mujoco.MjModel.from_xml_path(XML_PATH)
    d = mujoco.MjData(m)
    for _ in range(100):
        mujoco.mj_step(m, d)
    print(f'OK nq={m.nq} nv={m.nv} nu={m.nu} qpos_z={d.qpos[2]:.4f}')
    print(f'[ smoke ] body_mass(x2) total = {m.body_subtreemass[m.body("x2").id]:.4f} kg')
    print(f'[ smoke ] gravity = {m.opt.gravity}')
    print(f'[ smoke ] timestep = {m.opt.timestep}')


if __name__ == '__main__':
    main()
