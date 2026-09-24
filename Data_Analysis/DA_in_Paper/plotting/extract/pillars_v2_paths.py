#!/usr/bin/env python3.14
"""Extract the quadrotor's FLOWN paths on UAV-pillars (pillars_v2, Gen15 U18) into data/uav_paths.json.

    python3.14 plotting/extract/pillars_v2_paths.py

Replaces the `pillars` figure entry (the withdrawn pillars_hg panels of 2026-09-18) with the live pillars_v2 cells
and adds `pillars_cimf`. Source = the committed corpus
  analysis_results_checkpoint/23-09-UAV-Pillars-v2/live_p23uavpv2live   (jobs 26151-26154, tag p23uavpv2live)
whose npz are the avoiding eval's own output with the drone as the environment: obs_all rows are
[x_des, y_des, x, y] in the AVOIDING frame. They are mapped into the arena with uav_avoiding_bridge/frame.py at
SCALE 36 (X = 36 (y_a - 0.035), Y = -36 (x_a - 0.5)), the same map sources._pillars_v2_scene draws the constraints with.

Flags, decided here as for the other UAV figures:
  passed   n_success               -- crossed the finish line
  clean    n_violations == 0       -- never entered the DECLARED constraint set (halfspace / keep-out disk)
  safe     no pillar contact       -- the plant ends a flight on contact; a flight that ended within the radial
                                      rotor reach (0.36 m) of a pillar surface is a contact (all 51 failures are)
  aborted  False                   -- no flight diverged or hit the step cap (analysis/pillars_v2_live.py)
Panels: one geometry x variant cell each, all five seeds x two episodes = 10 flights per panel.
"""
import datetime
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..'))
sys.path.insert(0, REPO)
from uav_avoiding_bridge import frame as F                        # noqa: E402

OUT = os.path.join(REPO, 'Data_Analysis', 'DA_in_Paper', 'data', 'uav_paths.json')
CORPUS = os.path.join(REPO, 'Data_Analysis', 'analysis_results_checkpoint', '23-09-UAV-Pillars-v2', 'live_p23uavpv2live')
SCALE, STRIDE, NDP = 36.0, 2, 3
REACH_A = F.DRONE_REACH_M / SCALE
PILLARS = [(n, np.array([x, y]), r) for n, x, y, r in F.OBSTACLES_A]
GEOS = ('top-right-hard', 'top-left-hard', 'both-hard')
FIGS = {   # key -> (engine folder substring, K, figure title, model label)
    'pillars': ('flow_matching_v3_meanflow', 1, 'UAV-pillars', 'MeanFM, nfe 1'),
    'pillars_cimf': ('flow_matching_v3_alphaflow', 1, 'UAV-pillars', 'CI-MeanFM, nfe 1'),
}
VARIANTS = (('diffuser', 'unprojected'), ('dpcc-t-tightened', 'projected'))   # v3.79: as the s-curve panels


def cell_files(engine, K, geo, variant):
    pat = os.path.join(CORPUS, 'logs', 'avoiding-d3il', 'plans', engine, '*', f'H8_K{K}_*_msgp23uavpv2live', '*',
                       'results', f'halfspace_{geo}', f'{variant}.npz')
    return sorted(glob.glob(pat), key=lambda p: int(p.split('/results/')[0].split('/')[-1]))


def panel(engine, K, geo, variant, sub, label):
    eps, files = [], cell_files(engine, K, geo, variant)
    for f in files:
        d = np.load(f, allow_pickle=True)
        for i, o in enumerate(d['obs_all']):
            o = np.asarray(o, float)
            dr = o[:, 2:4]
            W = np.array([F.to_world_xy(x, y, SCALE) for x, y in dr])
            idx = list(range(0, len(W), STRIDE))
            if idx[-1] != len(W) - 1:
                idx.append(len(W) - 1)
            end_dist = min((np.linalg.norm(dr[-1] - c) - r) for _, c, r in PILLARS)
            passed = bool(d['n_success'][i]); contact = (not passed) and end_dist <= REACH_A + 0.002
            eps.append({'xy': [[round(float(W[j, 0]), NDP), round(float(W[j, 1]), NDP)] for j in idx],
                        'z': [round(F.ALTITUDE, NDP)] * len(idx),
                        'passed': passed, 'clean': bool(d['n_violations'][i] == 0),
                        'homotopy': None, 'homotopy_flown': None, 'min_z': round(F.ALTITUDE, NDP),
                        'aborted': False, 'safe': not contact, 'reason': 'contact' if contact else None})
    rel = os.path.relpath(os.path.dirname(files[0]), REPO) if files else ''
    # v3.79: the geometry is the panel title and the arm the subtitle; the model is the figure's (caption)
    return {'title': f'{geo}', 'sub': f'{sub}', 'variant': variant, 'tag': 'p23uavpv2live',
            'source': rel, 'geometry': geo, 'n': len(eps), 'passed': sum(e['passed'] for e in eps),
            'clean': sum(e['clean'] for e in eps), 'contact': sum(not e['safe'] for e in eps), 'episodes': eps}


def main():
    D = json.load(open(OUT)) if os.path.isfile(OUT) else {'figures': {}}
    for key, (engine, K, title, label) in FIGS.items():
        panels = []
        for var, sub in VARIANTS:                      # rows: before, after projection; columns: the 3 geometries
            for geo in GEOS:
                panels.append(panel(engine, K, geo, var, sub, label))
        D['figures'][key] = {'scene': title, 'panels': panels}
        print(f'{key}: {len(panels)} panels, ' + ', '.join(f"{p['geometry'][:9]}/{p['variant'][:8]} {p['passed']}/{p['n']} pass {p['clean']}/{p['n']} clean {p['contact']} contact" for p in panels))
    D.setdefault('pillars_v2', {})
    D['pillars_v2'] = {'generated': datetime.datetime.now().isoformat(timespec='seconds'), 'corpus': os.path.relpath(CORPUS, REPO),
                       'scale': SCALE, 'stride': STRIDE,
                       'note': 'pillars / pillars_cimf: the live pillars_v2 cells (jobs 26151-26154); flown avoiding-frame positions '
                               'obs_all[:, 2:4] mapped into the arena by uav_avoiding_bridge/frame.py; the pillars_hg panels of '
                               '2026-09-18 were replaced here (U17 abandoned, U18 is the scene of record).'}
    with open(OUT, 'w') as fh:
        json.dump(D, fh, separators=(',', ':'))
    print('wrote', os.path.relpath(OUT, REPO), f'{os.path.getsize(OUT) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
