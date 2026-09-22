#!/usr/bin/env python
"""Compare, cell by cell, the Panda source, the turbo replays and the LIVE (Mode L) run (Gen15 U18).

    python uav_avoiding_bridge/compare_live_turbo.py --live <dir containing halfspace_*/ of the live run>
           --turbo <same for the clock turbo> [--turboset <settle turbo>] [--variants diffuser dpcc-r-tightened]

Each dir is a `.../<seed>/results` folder (the one holding halfspace_<geo>/<variant>.npz). Prints the per-cell means
(success, S&C, cf, violations, steps, avg_time) for Panda (src_* of the turbo npz, or the live npz's own values are
the live numbers), and the per-episode paired view live-vs-turbo (same torch trial seeds). Numpy only; runs locally on
downloaded folders.
"""
import argparse
import glob
import os

import numpy as np

KEYS = ('n_success', 'n_success_and_constraints', 'collision_free_completed', 'n_violations', 'n_steps', 'avg_time')


def load(dirpath, variants):
    cells = {}
    for f in sorted(glob.glob(os.path.join(dirpath, 'halfspace_*', '*.npz'))):
        v = os.path.splitext(os.path.basename(f))[0]
        if variants and v not in variants:
            continue
        geo = os.path.basename(os.path.dirname(f))[len('halfspace_'):]
        d = np.load(f, allow_pickle=True)
        cells[(geo, v)] = {k: np.asarray(d[k], float) for k in KEYS if k in d.files} | \
                          {f'src_{k}': np.asarray(d[f'src_{k}'], float) for k in KEYS if f'src_{k}' in d.files}
    return cells


def row(name, c, pre=''):
    g = lambda k: c.get(pre + k)
    if g('n_success') is None:
        return f'  {name:14s}  (missing)'
    return (f"  {name:14s}  succ {g('n_success').mean():.2f}  S&C {g('n_success_and_constraints').mean():.2f}  "
            f"cf {g('collision_free_completed').mean():.2f}  viol {g('n_violations').mean():5.1f}  "
            f"steps {g('n_steps').mean():5.1f}  ms/step {1e3 * g('avg_time').mean():6.1f}  n={len(g('n_success'))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--live', required=True)
    ap.add_argument('--turbo', required=True)
    ap.add_argument('--turboset', default=None)
    ap.add_argument('--variants', nargs='*', default=[])
    a = ap.parse_args()
    live, turbo = load(a.live, a.variants), load(a.turbo, a.variants)
    tset = load(a.turboset, a.variants) if a.turboset else {}
    for key in sorted(set(live) | set(turbo)):
        geo, v = key
        print(f'== {geo} · {v}')
        if key in turbo:
            print(row('Panda (table)', turbo[key], 'src_'))
            print(row('turbo clock', turbo[key]))
        if key in tset:
            print(row('turbo settle', tset[key]))
        if key in live:
            print(row('LIVE (loop)', live[key]))
        if key in live and key in turbo:
            n = min(len(live[key]['n_success']), len(turbo[key]['n_success']))
            ls, ts = live[key]['n_success'][:n], turbo[key]['n_success'][:n]
            lc, tc = live[key]['n_success_and_constraints'][:n], turbo[key]['n_success_and_constraints'][:n]
            print(f'  paired episodes: success agree {int(np.sum(ls == ts))}/{n}   S&C agree {int(np.sum(lc == tc))}/{n}   '
                  f'steps live-turbo mean {np.mean(live[key]["n_steps"][:n] - turbo[key]["n_steps"][:n]):+.1f}')


if __name__ == '__main__':
    main()
