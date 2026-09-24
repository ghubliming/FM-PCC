#!/usr/bin/env python3.14
"""Figure 6.9 (fig:uav-scurve-paths) on the SELECTED configuration: FM, nfe 1, ten flights per panel, the same plans
under the two tracking controllers -- the flown-path companion of tab:uav-controller (R44, 2026-09-24).

    python3.14 Data_Analysis/DA_in_Paper/plotting/extract/scurve_r44_paths.py
    python3.14 Data_Analysis/DA_in_Paper/plotting/make_figs.py fig_uav_scurve_paths

Replaces ONLY `figures['scurve']` in data/uav_paths.json (read-modify-write, as extract/pillars_v2_paths.py does); every
other figure in the file is left as it is. The MeanFM nfe 10 pilot panels that `extract/uav_paths.py::PANELS['scurve']`
still declares are superseded -- note that re-running uav_paths.py rewrites the whole file from its own PANELS (a
pre-existing hazard it shares with the pillars_v2 panels), so run this script after it.

Corpus : temp/23-09-FULL/S_CURVE-P2/R44_scurve_20260924_161747 -- the two R44 fetches of 24-09 (md5-verified, merged):
         tags p23scgrid (A7: unprojected, cascaded geometric), p23scproj (B1: per-step random, cascaded geometric),
         p23scmjpc (C0: unprojected, MuJoCo MPC; C1: per-step random, MuJoCo MPC). Jobs 26171, 26195, 26196, 26204.
Panels : 2 x 2 -- rows unprojected / per-step projection (random, tightened set), columns cascaded geometric / MuJoCo MPC,
         in the row order of tab:uav-controller. Episode schema and down-sampling are extract/uav_paths.py's own.
"""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from uav_paths import episodes, find_cell, OUT, REPO, STRIDE        # noqa: E402  (same schema as every UAV panel)

ROOT = os.environ.get('SCURVE_RAW', os.path.join(REPO, 'temp/23-09-FULL/S_CURVE-P2/R44_scurve_20260924_161747'))
FM = 'H8_Dmodels.diffusion.FlowMatchingODE_9D'
PANELS = [   # (title, subtitle, run tag folder, variant)
    ('Cascaded geometric controller', 'unprojected plan', 'Efm_K1_mpc4_pid_stopgo_T0.5_p23scgrid', 'diffuser'),
    ('MuJoCo MPC', 'unprojected plan', 'Efm_K1_mpc4_mjpc_T0.5_p23scmjpc', 'diffuser'),
    ('Cascaded geometric controller', 'per-step projection', 'Efm_K1_mpc4_pid_stopgo_T0.5_p23scproj', 'dpcc-r-tightened'),
    ('MuJoCo MPC', 'per-step projection', 'Efm_K1_mpc4_mjpc_T0.5_p23scmjpc', 'dpcc-r-tightened'),
]


def main():
    D = json.load(open(OUT)) if os.path.isfile(OUT) else {'figures': {}}
    panels, missing = [], []
    for title, sub, tag, variant in PANELS:
        npz, res = find_cell(ROOT, 'uav-s_curve', 'mix_uav_fm', FM, tag, variant)
        if not npz:
            missing.append(f'{tag} :: {variant}')
            continue
        eps = episodes(npz, res)
        panels.append({'title': title, 'sub': sub, 'variant': variant, 'tag': tag.rsplit('_', 1)[-1],
                       'source': os.path.relpath(npz, REPO), 'episodes': eps, 'n': len(eps),
                       'passed': sum(e['passed'] for e in eps), 'clean': sum(e['clean'] for e in eps)})
        print(f'  {title:30s} {sub:20s} {len(eps):2d} flights  {panels[-1]["passed"]} success, '
              f'{panels[-1]["clean"]} violation-free, {sum(bool(e["aborted"]) for e in eps)} aborted')
    if missing:
        raise SystemExit('NOT FOUND (nothing written): ' + ', '.join(missing))
    D['figures']['scurve'] = {'scene': 'UAV-s-curve', 'panels': panels}
    D['scurve_r44'] = {'generated': datetime.datetime.now().isoformat(timespec='seconds'),
                       'corpus': os.path.relpath(ROOT, REPO), 'stride': STRIDE,
                       'note': ('scurve: FM nfe 1, ten flights per panel, unprojected and per-step random under the cascaded '
                                'geometric controller and MuJoCo MPC (R44: jobs 26171, 26195, 26196, 26204); replaces the '
                                'MeanFM nfe 10 pilot panels (three MPC flights against ten).')}
    with open(OUT, 'w') as fh:
        json.dump(D, fh, separators=(',', ':'))
    print('wrote', os.path.relpath(OUT, REPO), f'{os.path.getsize(OUT) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
