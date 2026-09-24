#!/usr/bin/env python3.14
"""UAV-corridor (corridor v3, R33 re-read at the end of the corridor, R45a) -> data/corridor_v3_frontier.json.

    python3.14 plotting/extract/corridor_v3_frontier.py

v3.83. One record per cell (geometry x model x budget x variant), first ten flights, with everything the corridor tables
of Chapter 6 and its appendix print, and the per-configuration points of the frontier figure fig_uav_corridor_tradeoff.

  success   the eval's success at the end of the corridor, x' = 2.0 m (R45a; eval_mix_uav.py SCENE_CLEAR_LINE_X): the
            original latch or the centre past x = 2.0 within the 396 steps, in controlled flight. Read on the recorded
            start-of-step positions, with the one flight that crosses during its last step (hump, diffusion, dpcc-r,
            trial 7) counted as the evaluator counts it (DA_20260924_corridor_v3_R45a_clear_line.md §3).
  sc        success and no violating control step over the flight (strict S&C).
  near      success and at least 95 % of the flight's control steps violation-free (the corridor's frontier bar).
  steps     control steps to the finish line: the first step whose end position is past x = 2.0 (flights that succeed).
  viol, ms  violating control steps and the time to compute one action (batch CSV), per flight.
Spread = sample standard deviation over the flights (Ch 5, single-seed results).
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, '..', '..', 'analysis')))
import corridor_v3_grid as G                                           # noqa: E402

OUT = os.path.normpath(os.path.join(HERE, '..', '..', 'data', 'corridor_v3_frontier.json'))
X_CLEAR, NEAR = 2.0, 0.95
LAST_STEP_CROSSERS = {('hump', 'diffusion', 20, 'ps-r', 7)}      # DA R45a §3


def _ms(a):
    a = np.asarray(a, float)
    return (float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0) if len(a) else (None, None)


def cell_record(geo, lab, eng, K, vn, v):
    c = G.load_cell(geo, eng, K, v)
    n = int(c['n'])
    succ_old = c['succ'].astype(bool)
    vf = c['cf'].astype(bool)
    steps_to = []
    for i, o in enumerate(c['obs']):
        k = int(np.argmax(o[:, 3] >= X_CLEAR))
        steps_to.append(k if o[k, 3] >= X_CLEAR else None)
    succ = np.array([succ_old[i] or steps_to[i] is not None for i in range(n)])
    for (g, e, kk, vv, i) in LAST_STEP_CROSSERS:
        if (g, e, kk, vv) == (geo, eng, K, vn):
            succ[i] = True
            steps_to[i] = int(c['steps'][i])
    viol = np.asarray(c['viol'], float)
    share = 1.0 - viol / np.asarray(c['steps'], float)
    near = succ & (share >= NEAR)
    sc = succ & vf
    st = [steps_to[i] if steps_to[i] is not None else int(c['steps'][i]) for i in range(n) if succ[i]]
    rec = dict(geo=geo, model=lab, engine=eng, K=K, variant=vn,
               projector={'none': 'none', 'ps': 'per-step', 'ep': 'endpoint'}[vn.split('-')[0]],
               rule=vn.split('-', 1)[1] if '-' in vn else '', n=n,
               success=_ms(succ.astype(float)), sc=_ms(sc.astype(float)), near=_ms(near.astype(float)),
               viol=_ms(viol), steps=_ms(st), ms=_ms(c['ms']),
               n_success=int(succ.sum()), n_sc=int(sc.sum()), n_near=int(near.sum()))
    return rec


def best(recs, key):
    """The table's rule: most S&C (or near), then fewest violating steps, then least time."""
    return min(recs, key=lambda r: (-r[key][0], r['viol'][0], r['ms'][0]))


def main():
    cells = []
    for geo in ('tilt', 'hump'):
        for lab, eng, Ks in G.MODELS:
            for K in Ks:
                vs = [('none', G.UNPROJ)] + [('ps-' + k, v) for k, v in G.PER_STEP.items()] + \
                     ([('ep-' + k, v) for k, v in G.ENDPOINT.items()] if (K >= 3 and eng != 'diffusion') else [])
                for vn, v in vs:
                    cells.append(cell_record(geo, lab.split()[0], eng, K, vn, v))
    picks = {'strict': [], 'near': [], 'viol': []}
    for geo in ('tilt', 'hump'):
        for lab, eng, Ks in G.MODELS:
            for K in Ks:
                for proj in ('per-step', 'endpoint'):
                    rs = [r for r in cells if r['geo'] == geo and r['engine'] == eng and r['K'] == K and r['projector'] == proj]
                    if not rs:
                        continue
                    picks['strict'].append(best(rs, 'sc')['variant'] and (geo, eng, K, best(rs, 'sc')['variant']))
                    picks['near'].append((geo, eng, K, best(rs, 'near')['variant']))
                    # v3.84 (author): the corridor is read on violating steps -- most successes (reaching the end
                    # of the corridor), then the fewest violating steps, then the least time
                    picks['viol'].append((geo, eng, K, min(rs, key=lambda r: (-r['success'][0], r['viol'][0], r['ms'][0]))['variant']))
    D = dict(meta=dict(script='plotting/extract/corridor_v3_frontier.py', x_clear=X_CLEAR, near_bar=NEAR,
                       corpus=G.CORPUS, batch=G.BATCH, n_read=G.N_READ,
                       analysis='analysis/DA_20260924_corridor_v3_R45a_clear_line.md'),
             cells=cells, best_strict=picks['strict'], best_near=picks['near'], best_viol=picks['viol'])
    with open(OUT, 'w') as f:
        json.dump(D, f, indent=0)
    print(OUT, len(cells), 'cells')


if __name__ == '__main__':
    main()
