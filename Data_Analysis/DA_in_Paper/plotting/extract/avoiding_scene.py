#!/usr/bin/env python3.14
"""Extract the obstacle-avoidance scene into a plain JSON file for the figure builders.

    python3.14 plotting/extract/avoiding_scene.py

Needs numpy and PyYAML (python3.14 in the AI container has both; plain python3 has
neither), which is why this is a separate step: the SVG builders run on the
standard library and read only ../../data/avoiding_scene.json.

Sources, read at run time, not typed in:
  config/projection_eval.yaml     ax_limits, halfspace_constraints, obstacle_constraints,
                                  enlarge_constraints -- the DPCC benchmark configuration
  scripts/eval.py:93-102          which halfspace and keep-out disk each geometry uses
  diffuser/utils/constraints_helpers.py (plot_environment_constraints)
                                  obstacle centres (radius 0.025) and the goal line y = 0.35
  D3IL avoiding demonstrations    96 episodes; the measured end-effector position c_pos[:, :2]

The per-geometry count of demonstrations that satisfy the constraints repeats the
check of scripts/visualize_data_constraints.py exactly: every step but the last,
halfspace side test, keep-out disk with the obstacle radius 0.025 added.
"""
import json
import os
import pickle
import re

import numpy as np
import yaml

REPO = '/workspaces/FM-PCC'
DATA = '/workspaces/aux_repo/HardFlow/d3il/environments/dataset/data/avoiding/data'
OUT = os.path.join(REPO, 'Data_Analysis', 'DA_in_Paper', 'data', 'avoiding_scene.json')
EXP = 'avoiding-d3il'
OBSTACLE_RADIUS = 0.025                    # constraints_helpers.plot_environment_constraints
OBSTACLE_CENTRES = [[0.5, -0.1], [0.425, 0.08], [0.575, 0.08], [0.35, 0.26], [0.5, 0.26], [0.65, 0.26]]
GOAL_Y = 0.35
# scripts/eval.py:93-102 -- geometry -> (halfspace indices, keep-out disk index)
GEOMETRY_MAP = {
    'top-left-hard':  ([0], 3),
    'top-right-hard': ([1], 4),
    'both-hard':      ([2, 3], 5),
}


def check_source_mapping():
    """Fail loudly if scripts/eval.py no longer maps geometries the way GEOMETRY_MAP says."""
    src = open(os.path.join(REPO, 'scripts', 'eval.py')).read()
    for name, (hs, ob) in GEOMETRY_MAP.items():
        block = re.search(rf"halfspace_variant == '{name}':(.*?)(?:elif|\n\n)", src, flags=re.S).group(1)
        got_hs = [int(i) for i in re.findall(r"\['halfspace_constraints'\]\[exp\]\[(\d+)\]", block)]
        got_ob = [int(i) for i in re.findall(r"\['obstacle_constraints'\]\[exp\]\[(\d+)\]", block)]
        assert got_hs == hs and got_ob == [ob], f'{name}: eval.py says {got_hs}/{got_ob}, expected {hs}/{ob}'


def load_config():
    txt = open(os.path.join(REPO, 'config', 'projection_eval.yaml')).read()
    return yaml.safe_load(txt)


def load_demonstrations():
    demos = []
    for fn in sorted(os.listdir(DATA)):
        s = pickle.load(open(os.path.join(DATA, fn), 'rb'))
        xy = np.asarray(s['robot']['c_pos'])[:, :2]
        demos.append((fn, xy[:-1]))          # visualize_data_constraints uses input_state[:-1]
    return demos


def satisfies(xy, halfspaces, disk):
    for (p0, p1, side) in halfspaces:
        m = (p1[1] - p0[1]) / (p1[0] - p0[0])
        b = p0[1] - m * p0[0]
        line = m * xy[:, 0] + b
        if side == 'below' and np.any(xy[:, 1] > line):
            return False
        if side == 'above' and np.any(xy[:, 1] < line):
            return False
    d = np.linalg.norm(xy - np.asarray(disk['center']), axis=1)
    return not np.any(d < disk['radius'] + OBSTACLE_RADIUS)


def main():
    check_source_mapping()
    cfg = load_config()
    hs_all = cfg['halfspace_constraints'][EXP]
    ob_all = cfg['obstacle_constraints'][EXP]
    delta = float(cfg['enlarge_constraints']['avoiding'])
    demos = load_demonstrations()

    geometries = {}
    for name, (hs_idx, ob_idx) in GEOMETRY_MAP.items():
        hs = [hs_all[i] for i in hs_idx]
        disk = ob_all[ob_idx]
        n_ok = sum(satisfies(xy, hs, disk) for _fn, xy in demos)
        geometries[name] = {
            'halfspaces': [{'p0': h[0], 'p1': h[1], 'feasible_side': h[2]} for h in hs],
            'disk': {'center': disk['center'], 'radius': disk['radius']},
            'demonstrations_satisfying': int(n_ok),
        }

    out = {
        'source': {
            'config': 'config/projection_eval.yaml',
            'geometry_mapping': 'scripts/eval.py:93-102',
            'obstacles_and_goal': 'diffuser/utils/constraints_helpers.py (plot_environment_constraints)',
            'demonstrations': DATA + ' (D3IL obstacle-avoidance dataset)',
        },
        'ax_limits': cfg['ax_limits'][EXP],
        'goal_y': GOAL_Y,
        'obstacles': {'centers': OBSTACLE_CENTRES, 'radius': OBSTACLE_RADIUS},
        'tightening': delta,
        'geometries': geometries,
        'n_demonstrations': len(demos),
        'demonstrations': [[[round(float(x), 4), round(float(y), 4)] for x, y in xy] for _fn, xy in demos],
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'wrote {OUT} ({os.path.getsize(OUT) // 1024} KB): {len(demos)} demonstrations, tightening {delta}')
    for name, g in geometries.items():
        print(f'  {name:15s} satisfied by {g["demonstrations_satisfying"]:2d}/{len(demos)} demonstrations')


if __name__ == '__main__':
    main()
