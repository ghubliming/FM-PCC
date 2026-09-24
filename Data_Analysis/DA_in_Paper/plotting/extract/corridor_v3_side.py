#!/usr/bin/env python3.14
"""Extract the UAV-corridor (corridor v3, R33) flights for the side-view figure into data/corridor_v3_side.json.

    python3.14 plotting/extract/corridor_v3_side.py

v3.80. The thesis figure `fig_uav_corridor_side` (builders/altitude.py) draws the two structural results of
analysis/DA_20260924_corridor_v3.md §4 in the chapter's vocabulary: on the tilt, the residue of violating steps in the
last quarter-metre of the leaned plane; on the hump, the stop before the apex at K = 1. It replaces the analysis's own
diagnostic figure (DA_20260923_corridor_v3_preliminary/fig_diag_tilt_residue_hump_stall.png), which stays in the
analysis folder as it is.

Cells are read through analysis/corridor_v3_grid.py -- the loader the DA's tables came from -- so the ten flights are
the ten the chapter's tables count (the first ten of each cell; routes L, C, R cycling). Model: instantaneous-velocity
matching (FM), the one flow model flown at K = 20; per-step projection under rule c, the best per-step rule of
Table tab:uav-corridor at K = 1 and K = 20 on both constraints (DA §3.2).

Quantities, both from the evaluation's own geometry (eval_mix_uav.py `_hs_normal3`, `_exec_constraint_violations`):
  tilt   margin = n3 . (p - P0) - r_drone over the plane's x span [-2, 2]; negative = a violating control step
  hump   altitude z over x; the violation boundary is roof(x) + r_drone * sqrt(1 + s^2) on |x| <= 1.5
         (the perpendicular rotor reach of each roof face, as the scorer tests it)
p is the flown position (obs[:, 3:6]). The top of the plan's altitude range is read from the stored plans
(the largest planned z over the cells drawn), not assumed.
"""
import datetime
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ANALYSIS = os.path.normpath(os.path.join(HERE, '..', '..', 'analysis'))
sys.path.insert(0, ANALYSIS)
import corridor_v3_grid as G                                          # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'corridor_v3_side.json'))
SERIES = (   # key, K, variant, label
    ('raw', 1, G.UNPROJ, 'unprojected, K = 1'),
    ('ps1', 1, G.PER_STEP['c'], 'per-step projection, K = 1'),
    ('ps20', 20, G.PER_STEP['c'], 'per-step projection, K = 20'),
)


def _counts(c):
    n = int(c['n'])
    return dict(n=n, success=int(np.nansum(c['succ'])), violation_free=int(np.nansum(c['cf'])),
                sc=int(np.nansum(c['sc'])), viol_mean=round(float(np.nanmean(c['viol'])), 2))


def tilt():
    cfg = G.geo_config('tilt')
    hs = next(h for h in cfg['halfspace_constraints'] if h.get('z_lean'))
    n3, P0 = G.NS['_hs_normal3'](hs)
    r = float(cfg['inflation']['r_drone'])
    lo, hi = hs['x_active']
    out = []
    for key, K, v, lab in SERIES:
        c = G.load_cell('tilt', 'fm', K, v)
        flights = []
        for o in c['obs']:
            P = o[:, 3:6]
            m = (P - P0) @ n3 - r
            w = (P[:, 0] >= lo) & (P[:, 0] <= hi)
            flights.append([[round(float(x), 4), round(float(y), 4)] for x, y in zip(P[w, 0], m[w])])
        deep = min(min(y for _, y in f) for f in flights)
        xv = [x for f in flights for x, y in f if y < 0]
        out.append(dict(key=key, K=K, variant=v, label=lab, flights=flights, deepest=round(deep, 4),
                        violating_x=[round(min(xv), 3), round(max(xv), 3)] if xv else None,
                        success_flags=[bool(b) for b in c['succ']], **_counts(c)))
    return dict(x_active=[lo, hi], r_drone=r, series=out)


def hump():
    cfg = G.geo_config('hump')
    faces = [h for h in cfg['halfspace_constraints'] if h.get('plane') == 'xz']
    r = float(cfg['inflation']['r_drone'])
    roof = [faces[0]['line'][0], faces[0]['line'][1], faces[1]['line'][1]]
    s = abs((roof[1][1] - roof[0][1]) / (roof[1][0] - roof[0][0]))
    out, ztop = [], -math.inf
    for key, K, v, lab in SERIES:
        c = G.load_cell('hump', 'fm', K, v)
        flights = [[[round(float(x), 4), round(float(z), 4)] for x, z in zip(o[:, 3], o[:, 5])] for o in c['obs']]
        if c.get('plans') is not None:
            ztop = max(ztop, max(float(np.asarray(p, float)[..., [2, 5]].max()) for p in c['plans']))
        ends = [f[-1] for f in flights]
        out.append(dict(key=key, K=K, variant=v, label=lab, flights=flights, **_counts(c),
                        success_flags=[bool(b) for b in c['succ']],
                        end_x=[round(min(e[0] for e in ends), 3), round(max(e[0] for e in ends), 3)],
                        end_z=[round(min(e[1] for e in ends), 3), round(max(e[1] for e in ends), 3)]))
    return dict(roof=roof, slope=s, r_drone=r, limit_offset=round(r * math.hypot(1.0, s), 4),
                plan_z_top=round(ztop, 4), series=out)


def main():
    D = dict(meta=dict(written=datetime.date.today().isoformat(), script='plotting/extract/corridor_v3_side.py',
                       corpus=G.CORPUS, batch=G.BATCH, n_read=G.N_READ, model='fm',
                       tags={g: t for g, (t, _) in G.GEOS.items()},
                       analysis='analysis/DA_20260924_corridor_v3.md'),
             tilt=tilt(), hump=hump())
    with open(OUT, 'w') as f:
        json.dump(D, f, separators=(',', ':'))
    for g in ('tilt', 'hump'):
        for s in D[g]['series']:
            extra = (f" deepest {s['deepest']:+.3f} violating x {s['violating_x']}" if g == 'tilt'
                     else f" end x {s['end_x']} z {s['end_z']}")
            print(f"{g:4} {s['key']:5} n={s['n']} success {s['success']}/{s['n']} violation-free "
                  f"{s['violation_free']}/{s['n']} S&C {s['sc']}/{s['n']} viol {s['viol_mean']}{extra}")
    print('hump limit offset', D['hump']['limit_offset'], 'plan z top', D['hump']['plan_z_top'])
    print(OUT, f'{os.path.getsize(OUT) / 1e3:.0f} kB')


if __name__ == '__main__':
    main()
