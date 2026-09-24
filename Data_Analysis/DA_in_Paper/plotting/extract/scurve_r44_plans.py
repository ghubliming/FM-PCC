#!/usr/bin/env python3.14
"""fig:uav-scurve-plans -- the plans the models emit on UAV-s-curve, drawn like fig:raw-plans (Figure 6.1): one
flight per panel, the plan of every twentieth control step with all four candidates (a sample: fig:raw-plans draws
every fourth), each candidate drawn as the measured
planar position of its eight waypoints; the executed path itself is not drawn.

    python3.14 Data_Analysis/DA_in_Paper/plotting/extract/scurve_r44_plans.py
    python3.14 Data_Analysis/DA_in_Paper/plotting/make_figs.py fig_uav_scurve_plans

Author, 2026-09-24 (v3.93): "add the raw MPC traj smooth fig like Figure 6.1 so we can see how is quality ... just
FM K1 vs 2 vs 20 vs mf 1,2 is enough". Writes data/uav_scurve_plans.json only.

Corpus : temp/23-09-FULL/S_CURVE-P2/R44_scurve_20260924_161747 (R44 phase A, tag p23scgrid, the cells of
         tab:uav-scurve), unprojected (`diffuser`), cascaded geometric controller.
Plans  : `sampled_trajectories_all[flight]` holds one plan per control step, shape (4 candidates, 8 waypoints, 6); the
         six are the commanded and the measured position, [p_des, p], as in `obs_all` (checked: waypoint 0 of the plan
         of step t is the measured position at step t, median offset 0). The measured channel, columns 3:5, is drawn,
         as in fig:raw-plans.
Flight : trial 0 of every cell (seed 10000, the same launch pose and route offset in every cell).
"""
import datetime
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from uav_paths import find_cell, REPO        # noqa: E402

ROOT = os.environ.get('SCURVE_RAW', os.path.join(REPO, 'temp/23-09-FULL/S_CURVE-P2/R44_scurve_20260924_161747'))
OUT = os.path.join(REPO, 'Data_Analysis', 'DA_in_Paper', 'data', 'uav_scurve_plans.json')
EVERY = 20       # a sample of the plans, every twentieth control step (author, 24-09: "just some of the mpc traj to show
                 # the quality is enough"); fig:raw-plans draws every fourth
FLIGHT = 0       # trial 0 in every cell
FM = ('mix_uav_fm', 'H8_Dmodels.diffusion.FlowMatchingODE_9D', 'Efm_K{k}_mpc4_pid_stopgo_T0.5_p23scgrid')
MF = ('mix_uav_mf', 'H8_Dmodels.mf_diffusion.MeanFlowODE_9D_dp0.5_bbunet', 'Emf_K{k}_mpc4_pid_stopgo_T0.5_p23scgrid')
PANELS = [('FM', FM, 1), ('MeanFM', MF, 1), ('FM', FM, 2), ('MeanFM', MF, 2), ('FM', FM, 20)]   # 2 columns: FM | MeanFM


def main():
    panels = []
    for label, (folder, model, tag), k in PANELS:
        npz, _res = find_cell(ROOT, 'uav-s_curve', folder, model, tag.format(k=k), 'diffuser')
        if not npz:
            raise SystemExit(f'NOT FOUND: {folder}/{model}/{tag.format(k=k)}')
        d = np.load(npz, allow_pickle=True)
        plans_all = d['sampled_trajectories_all'][FLIGHT]
        obs = np.asarray(d['obs_all'][FLIGHT], float)
        assert len(plans_all) == len(obs), (npz, len(plans_all), len(obs))
        # alignment check: waypoint 0 of the plan of step t is the measured position at step t (median offset of
        # candidate 0 below 1 mm; single steps differ by a few centimetres, which the drawing does not use)
        w0 = np.stack([np.asarray(q, float)[0, 0, 3:5] for q in plans_all])
        assert float(np.median(np.linalg.norm(w0 - obs[:, 3:5], axis=1))) < 1e-3, npz
        plans = []
        for t in range(0, len(plans_all), EVERY):
            p = np.asarray(plans_all[t], float)[:, :, 3:5]              # measured x, y of the 8 waypoints, 4 candidates
            plans.append(np.round(p, 3).tolist())
        success = bool(d['success_relaxed'][FLIGHT])
        aborted = bool(d['divergence_aborted'][FLIGHT])
        panels.append({'label': label, 'k': k, 'tag': tag.format(k=k), 'source': os.path.relpath(npz, REPO),
                       'flight': FLIGHT, 'steps': int(len(obs)), 'success': success, 'aborted': aborted,
                       'safe': bool(d['phys_safe'][FLIGHT]),
                       'cell_success': int(np.sum(d['success_relaxed'])), 'cell_n': int(len(d['success_relaxed'])),
                       'end': np.round(obs[-1, 3:5], 3).tolist(), 'plans': plans})
        print(f'  {label:7s} K{k:<3d} flight {FLIGHT}: {len(obs)} steps, {len(plans)} plans drawn, '
              f'success {success}, aborted {aborted}; cell {panels[-1]["cell_success"]}/{panels[-1]["cell_n"]}')
    out = {'generated': datetime.datetime.now().isoformat(timespec='seconds'), 'corpus': os.path.relpath(ROOT, REPO),
           'every': EVERY, 'flight': FLIGHT, 'channel': 'measured position (columns 3:5 of the 6-D plan)',
           'panels': panels}
    with open(OUT, 'w') as fh:
        json.dump(out, fh, separators=(',', ':'))
    print(f'wrote {os.path.relpath(OUT, REPO)} ({os.path.getsize(OUT) // 1024} KB)')


if __name__ == '__main__':
    main()
