#!/usr/bin/env python3.14
"""Extract the EXECUTED end-effector paths of the two manipulator benchmarks into JSON.

    python3.14 plotting/extract/exec_paths.py [--avoiding-root DIR] [--aligning-root DIR]

The counterpart of ``extract/uav_paths.py``, for the two environments that had no flown-path
figure: D3IL-avoiding and D3IL-aligning. Same division of labour -- this step needs numpy and
reads the gitignored drop; the SVG builders are standard library and read only
``../../data/exec_paths.json``.

Why this exists
---------------
The committed batches carry per-rollout SCALARS. The executed positions live in the rollout
artefacts on the cluster and were staged in wave 3 (2026-09-20, D10a) and in the 09-19 drop
(D10b); see ``Data_Analysis/analysis_results_checkpoint/LEDGER_20260918_v3_figure_artefact_fetch.md``.
Neither drop is committed, so this extract is what persists.

What is read, and the two column conventions
--------------------------------------------
``<run>/<variant>.npz``::

    avoiding   obs_all  ragged, one (T, 4) per episode  = [x_des, y_des, x, y]
    aligning   obs_all  dense  (10, 400, 6)             = [des_xyz | xyz]

Both layouts are des-then-actual, declared in ``config/projection_eval.yaml`` and written by
the two eval scripts; the EXECUTED position is the second half in each case. Only the
projected variants carry ``obs_all`` as paths -- the unprojected ``diffuser.npz`` of an
*avoiding* leaf holds scalars only, which is why the avoiding figure is built from
``dpcc-t-tightened``. On *aligning* the unprojected arm does carry them.

What a path is drawn as, decided here rather than in the builder::

    clean     avoiding  n_violations == 0        aligning  constraint_exec_zero_violation == 1
    reached   avoiding  n_success == 1           aligning  -- the task has no goal test, so
                                                 every aligning path is drawn solid

Naming trap, repeated from the ledger: a folder ``Dmodels.diffusion.GaussianDiffusion`` under a
``flow_matching_v3_*`` family is the FLOW model under its pre-26-May class name. The families
below are keyed on the folder that names the objective, and the diffusion baseline is the one
under ``plans/diffusion/``.
"""
import argparse
import datetime
import glob
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, '..', '..', '..', '..'))
OUT = os.path.join(REPO, 'Data_Analysis', 'DA_in_Paper', 'data', 'exec_paths.json')
AVOIDING_ROOT = os.path.join(REPO, 'temp', '20-09')
ALIGNING_ROOT = os.path.join(REPO, 'temp', '19-09')

STRIDE = 1                  # an avoiding episode is ~60 steps; no thinning needed
ALIGNING_STRIDE = 2         # an aligning episode is 400
NDP = 4                     # metres; 0.1 mm, well below what the figure resolves

# ---------------------------------------------------------------- D3IL-avoiding ---
# The two average-velocity models at the two budgets the benchmark operates them at, under
# the projector and rule of the operating point (author, v3.53: the figure shows the models
# the chapter carries forward, not one panel per model at a different budget each).
AVOIDING_GEOMETRY = 'top-right-hard'
AVOIDING_VARIANT = 'dpcc-t-tightened'
# The DPCC protocol runs n_trials = 2 episodes per seed, and Chapter 6 reports D3IL-avoiding
# at that protocol (v3.55). The staged artefacts are 20-episode runs, but episode i of a run
# is fully determined by i -- `torch.manual_seed(i)` and `env_seed = i`, scripts/eval.py:297-299
# -- so episodes 0 and 1 of a 20-trial run ARE the two a 2-trial run produces. Taking the first
# AVOIDING_EPISODES of each artefact is therefore the 2-episode evaluation, not a subsample of
# a different one.
AVOIDING_EPISODES = 2
# The panel heading names the model and its budget; the subtitle is the panel's own count
# and nothing else. Everything about the protocol is in the caption (v3.49 figure rule).
AVOIDING_PANELS = [
    ('MeanFM, K = 1', 'flow_matching_v3_meanflow', 1),
    ('MeanFM, K = 2', 'flow_matching_v3_meanflow', 2),
    ('CI-MeanFM, K = 1', 'flow_matching_v3_alphaflow', 1),
    ('CI-MeanFM, K = 2', 'flow_matching_v3_alphaflow', 2),
]

# ---------------------------------------------------------------- D3IL-aligning ---
ALIGNING_GEOMETRY = 'combined_5-tightened'
ALIGNING_PANELS = [
    ('no projection', 'diffuser'),
    ('per-step projection', 'dpcc-r'),
    ('endpoint projection', 'hardflow_sls-r'),
]


def _round(xy):
    return [[round(float(x), NDP), round(float(y), NDP)] for x, y in xy]


def _avoiding_cell(root, family, K):
    pat = os.path.join(root, '**', 'plans', family, '*', f'H8_K{K}_*', '*', 'results',
                       f'halfspace_{AVOIDING_GEOMETRY}', f'{AVOIDING_VARIANT}.npz')
    hits = sorted(glob.glob(pat, recursive=True))
    if not hits:
        return None, None
    path = hits[0]
    d = np.load(path, allow_pickle=True)
    obs, ns, nv = d['obs_all'], np.asarray(d['n_success']), np.asarray(d['n_violations'])
    episodes = []
    for i in range(len(ns)):
        a = np.asarray(obs[i], dtype=float)
        if a.ndim != 2 or a.shape[1] < 4:
            continue
        xy = a[::STRIDE, 2:4]
        if len(a) and not np.array_equal(xy[-1], a[-1, 2:4]):
            xy = np.vstack([xy, a[-1:, 2:4]])
        episodes.append({'xy': _round(xy), 'reached': bool(ns[i] == 1), 'clean': bool(nv[i] == 0)})
        if AVOIDING_EPISODES and len(episodes) >= AVOIDING_EPISODES:
            break
    m = re.search(r'/(\d+)/results/', path)
    return episodes, {'run': os.path.relpath(path, root), 'seed': int(m.group(1)) if m else None}


def _context_index(box, target, contexts):
    """Match a rollout to its context by its recorded box and target, so that the figure
    numbers contexts the way sources.ALIGNING_CONTEXTS (and therefore Fig 6.6) does."""
    best, bd = None, 1e9
    for j, c in enumerate(contexts):
        dd = ((box[0] - c['box'][0]) ** 2 + (box[1] - c['box'][1]) ** 2
              + (target[0] - c['target'][0]) ** 2 + (target[1] - c['target'][1]) ** 2)
        if dd < bd:
            best, bd = j, dd
    return best if bd < 1e-4 else None


def _aligning_cell(root, variant, contexts):
    pat = os.path.join(root, '**', ALIGNING_GEOMETRY, f'{variant}_train_set',
                       f'{variant}_train_set.npz')
    hits = sorted(glob.glob(pat, recursive=True))
    if not hits:
        return None, None
    path = hits[0]
    d = np.load(path, allow_pickle=True)
    obs = np.asarray(d['obs_all'], dtype=float)
    clean = np.asarray(d['constraint_exec_zero_violation']).ravel()
    box = np.asarray(d['context_box_init_xy'], dtype=float)
    tgt = np.asarray(d['context_target_xy'], dtype=float)
    episodes = []
    for i in range(obs.shape[0]):
        xy = obs[i, ::ALIGNING_STRIDE, 3:5]
        xy = np.vstack([xy, obs[i, -1:, 3:5]])
        episodes.append({'xy': _round(xy), 'clean': bool(clean[i] == 1), 'reached': True,
                         'context': _context_index(box[i], tgt[i], contexts)})
    return episodes, {'run': os.path.relpath(path, root)}


def _panelise(title, episodes, meta):
    return {'title': title, 'n': len(episodes),
            'clean': sum(e['clean'] for e in episodes),
            'reached': sum(e['reached'] for e in episodes),
            'tag': meta['run'], 'episodes': episodes}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--avoiding-root', default=AVOIDING_ROOT)
    ap.add_argument('--aligning-root', default=ALIGNING_ROOT)
    a = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    import sources as S

    figures = {}

    panels = []
    for title, family, K in AVOIDING_PANELS:
        eps, meta = _avoiding_cell(a.avoiding_root, family, K)
        if eps is None:
            print(f'  MISSING  avoiding {family} K={K}')
            continue
        panels.append(_panelise(title, eps, meta))
        print(f"  {title:18s} {len(eps):2d} episodes, "
              f"{panels[-1]['reached']:2d} reached the goal, {panels[-1]['clean']:2d} violation-free")
    if panels:
        figures['avoiding'] = {'geometry': AVOIDING_GEOMETRY, 'variant': AVOIDING_VARIANT,
                               'panels': panels}

    panels = []
    for title, variant in ALIGNING_PANELS:
        eps, meta = _aligning_cell(a.aligning_root, variant, S.ALIGNING_CONTEXTS)
        if eps is None:
            print(f'  MISSING  aligning {variant}')
            continue
        panels.append(_panelise(title, eps, meta))
        print(f"  {title:21s} {len(eps):2d} contexts, {panels[-1]['clean']:2d} violation-free")
    if panels:
        figures['aligning'] = {'geometry': ALIGNING_GEOMETRY, 'panels': panels}

    out = {'generated': datetime.date.today().isoformat(),
           'source': {
               'avoiding': 'temp/20-09 (wave-3 stage, D10a); obs_all[:, 2:4] is the executed '
                           'end-effector position, config/projection_eval.yaml:20',
               'aligning': 'temp/19-09 (D10b); obs_all[:, :, 3:5] is the executed end-effector '
                           'position; the box pose is NOT logged, see the figure caption',
               'ledger': 'Data_Analysis/analysis_results_checkpoint/'
                         'LEDGER_20260918_v3_figure_artefact_fetch.md',
           },
           'figures': figures}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w') as f:
        json.dump(out, f, separators=(',', ':'))
    print(f'wrote {OUT} ({os.path.getsize(OUT) // 1024} KB)')


if __name__ == '__main__':
    main()
