#!/usr/bin/env python3
"""UAV-s-curve R44a: the raw generative grid (Chapter 6, Table 6.14 `tab:uav-scurve`), from the DA_UAV_v1 batch.

    python3 Data_Analysis/DA_in_Paper/analysis/scurve_r44.py
    SCURVE_BATCH=<batch dir> python3 Data_Analysis/DA_in_Paper/analysis/scurve_r44.py

Corpus    : temp/23-09-FULL/24-09-1000/batch_uav_20260924_081422 (DA_UAV_v1 over the whole UAV_MIX tree, 24-09 08:14).
            Cells: run tag p23scgrid (CI-MeanFM: EPlatest_p23scgrid), geometry s_curve_hg_*, variant diffuser, controller
            pid_stopgo, seed 6, ten flights of the one route. Jobs 26167-26176
            (data_status/SLURM_RUNBOOK_20260923_uav_scurve_R44.md §6).
Reference : analysis_results_checkpoint/15-09/batch_uav_20260915_100816, the tags Table 6.14 printed before
            (u7hg; CI-MeanFM EPlatest_u6unet_ae02), compared flight by flight.
Metrics   : the table's own definitions, per flight from per_rollout_detail.csv:
              success   = success_relaxed = crossed the finish line AND safe (contact within the scene limit, airborne:
                          min z > 0.2 m, not ended by the divergence guard) -- eval_mix_uav.py:1994-2027
              S&C       = n_success_relaxed_and_constraints (success on a violation-free flight)
              distance  = goal_dist, mean over the ten flights
              aborted   = divergence_aborted (the vehicle inverted; never a success)
              ms/step   = avg_time_ms (network time per action; no projection in this grid)
Selection : the chapter's rule -- the configuration whose plans cross the finish line on the most flights; ties ->
            fewer aborted -> smaller distance.
Stdlib only. Author: Claude (Opus 5.5, Claude Code, R44 run chat), 2026-09-24.
"""
import csv
import os
import statistics as st
from collections import defaultdict

REPO = '/workspaces/FM-PCC'
BATCH = os.environ.get('SCURVE_BATCH', os.path.join(REPO, 'temp/23-09-FULL/24-09-1000/batch_uav_20260924_081422'))
OLD = os.path.join(REPO, 'Data_Analysis/analysis_results_checkpoint/15-09/batch_uav_20260915_100816')
TAG = 'p23scgrid'
GRID = [('MeanFM', 'mf', 1), ('MeanFM', 'mf', 2), ('MeanFM', 'mf', 20),
        ('CI-MeanFM, $\\alpha_{\\mathrm{end}}=0.2$', 'af', 1), ('CI-MeanFM, $\\alpha_{\\mathrm{end}}=0.2$', 'af', 2),
        ('CI-MeanFM, $\\alpha_{\\mathrm{end}}=0.2$', 'af', 20),
        ('FM', 'fm', 1), ('FM', 'fm', 2), ('FM', 'fm', 20), ('Diffusion', 'diffusion', 20)]
OLD_TAG = {('mf', 2): 'u7hg', ('af', 1): 'EPlatest_u6unet_ae02', ('af', 2): 'EPlatest_u6unet_ae02',
           ('fm', 2): 'u7hg', ('fm', 20): 'u7hg', ('diffusion', 20): 'u7hg'}
OUTCOME = ['success_relaxed', 'n_success_relaxed_and_constraints', 'goal_dist', 'divergence_aborted', 'divergence_step',
           'n_steps', 'n_violations', 'goal_crossed_line', 'phys_min_z', 'track_err_mean']


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return float('nan')


def tag_of(r):
    return r['FolderName'].rsplit('@', 1)[-1]


def load(path, keep):
    """(engine, K, tag, variant, controller) -> {rollout_idx: row}, s_curve_hg only."""
    out = defaultdict(dict)
    with open(os.path.join(path, 'per_rollout_detail.csv')) as fh:
        for r in csv.DictReader(fh):
            if r['scene'] != 's_curve' or not r['geo'].startswith('s_curve_hg_') or not keep(tag_of(r)):
                continue
            key = (r['engine'], int(f(r['K'])), tag_of(r), r['variant'], r['controller'])
            out[key][int(f(r['rollout_idx']))] = r
    return out


def cell(rows):
    R = [rows[i] for i in sorted(rows)]
    n = len(R)
    cnt = lambda m: sum(f(r[m]) > 0.5 for r in R)
    sd = lambda xs: st.stdev(xs) if len(xs) > 1 else 0.0
    ab = [r for r in R if f(r['divergence_aborted']) > 0.5]
    ok = [r for r in R if f(r['success_relaxed']) > 0.5]
    viol = [f(r['n_violations']) for r in R]
    return dict(
        n=n, succ=cnt('success_relaxed'), sc=cnt('n_success_relaxed_and_constraints'), ab=len(ab),
        cross=cnt('goal_crossed_line'), vfree=cnt('collision_free_completed'),
        crossed_unsafe=sum(1 for r in R if f(r['goal_crossed_line']) > 0.5 and f(r['success_relaxed']) < 0.5),
        dist=st.mean(f(r['goal_dist']) for r in R), dist_sd=sd([f(r['goal_dist']) for r in R]),
        ms=st.mean(f(r['avg_time_ms']) for r in R), ms_sd=sd([f(r['avg_time_ms']) for r in R]),
        viol=st.mean(viol), viol_sd=sd(viol), steps=st.mean(f(r['n_steps']) for r in R),
        steps_ok=(st.mean(f(r['steps_to_goal']) for r in ok) if ok else float('nan')),
        track=st.mean(f(r['track_err_mean']) for r in R),
        div_lo=(min(f(r['divergence_step']) for r in ab) if ab else float('nan')),
        div_hi=(max(f(r['divergence_step']) for r in ab) if ab else float('nan')))


def main():
    new = load(BATCH, lambda t: t.endswith(TAG))
    old = load(OLD, lambda t: t in ('u7hg', 'EPlatest_u6unet_ae02'))
    grid = {}
    for label, e, K in GRID:
        keys = [k for k in new if k[0] == e and k[1] == K and k[3] == 'diffuser' and k[4] == 'pid_stopgo']
        assert len(keys) == 1, (e, K, keys)
        grid[(e, K)] = (keys[0], cell(new[keys[0]]))

    print(f'# UAV-s-curve R44a — tag {TAG}, batch {os.path.basename(BATCH)}\n')
    print('## Coverage\n')
    n_cells = sum(1 for k in new if k[3] == 'diffuser')
    print(f'- cells under the tag: {n_cells} (variant diffuser, controller pid_stopgo); flights per cell: '
          + ', '.join(str(c['n']) for _, c in grid.values()))
    extra = [k for k in new if k[3] != 'diffuser' or k[4] != 'pid_stopgo']
    print(f'- other cells under the tag: {len(extra)}')
    with open(os.path.join(BATCH, 'data_quality.csv')) as fh:
        dq = [r for r in csv.DictReader(fh) if r['FolderName'].endswith(TAG)]
    bad = [r['FolderName'] for r in dq if f(r['n_rollouts']) != 10 or f(r['n_cb_tripped']) != 0
           or f(r['timing_missing']) != 0 or f(r['n_diagnostics_json']) != 10]
    print(f'- data_quality rows: {len(dq)}; failing a gate (n_rollouts 10, no cut-off trip, timing, 10 diagnostics): '
          f'{len(bad)} {bad if bad else ""}')

    print('\n## Table 6.14 `tab:uav-scurve` (ten flights per cell)\n')
    print('| model | nfe | success | S&C | distance [m] | aborted | ms/step |')
    print('| :-- | --: | --: | --: | --: | --: | --: |')
    for label, e, K in GRID:
        c = grid[(e, K)][1]
        print(f'| {label} | {K} | {c["succ"]}/10 | {c["sc"]}/10 | {c["dist"]:.3f} | {c["ab"]}/10 | {c["ms"]:.1f} |')
    print('\nLaTeX rows:\n```latex')
    for label, e, K in GRID:
        c = grid[(e, K)][1]
        print(f'    {label:40s} & {K:<2d} & {c["succ"]}/10 & {c["sc"]}/10 & {c["dist"]:.3f} & {c["ab"]}/10 & {c["ms"]:.1f} \\\\')
    print('```')

    print('\n## Detail per cell\n')
    print('| model | nfe | crossed | crossed, not safe | violation-free | violating steps | steps | steps to line '
          '(successes) | inverted at step | tracking error [m] | distance sd | ms sd |')
    print('| :-- | --: | --: | --: | --: | --: | --: | --: | :-- | --: | --: | --: |')
    for label, e, K in GRID:
        c = grid[(e, K)][1]
        inv = '—' if c['ab'] == 0 else f'{c["div_lo"]:.0f}–{c["div_hi"]:.0f}'
        so = '—' if c['succ'] == 0 else f'{c["steps_ok"]:.0f}'
        print(f'| {label.split(",")[0]} | {K} | {c["cross"]}/10 | {c["crossed_unsafe"]} | {c["vfree"]}/10 | '
              f'{c["viol"]:.1f} ± {c["viol_sd"]:.1f} | {c["steps"]:.0f} | {so} | {inv} | {c["track"]:.3f} | '
              f'{c["dist_sd"]:.3f} | {c["ms_sd"]:.1f} |')

    print('\n## Reproduction of the six cells Table 6.14 printed before (flight by flight)\n')
    print('| model | nfe | old tag | differing outcome values (10 flights × 10 metrics) | ms/step old → new |')
    print('| :-- | --: | :-- | --: | :-- |')
    for (e, K), ot in OLD_TAG.items():
        okey = [k for k in old if k[0] == e and k[1] == K and k[2] == ot and k[3] == 'diffuser' and k[4] == 'pid_stopgo']
        assert len(okey) == 1, (e, K, okey)
        O = old[okey[0]]; N = new[grid[(e, K)][0]]
        assert sorted(O) == sorted(N) == list(range(10))
        diff = 0
        for i in range(10):
            for m in OUTCOME:
                a, b = f(N[i][m]), f(O[i][m])
                if not ((a != a and b != b) or abs(a - b) < 1e-9):
                    diff += 1
        mo = st.mean(f(O[i]['avg_time_ms']) for i in range(10)); mn = grid[(e, K)][1]['ms']
        print(f'| {e} | {K} | {ot} | {diff} | {mo:.1f} → {mn:.1f} |')

    print('\n## Selection (most crossings; ties: fewer aborted, then smaller distance)\n')
    rank = sorted(grid.items(), key=lambda kv: (-kv[1][1]['succ'], kv[1][1]['ab'], kv[1][1]['dist']))
    for i, ((e, K), (_, c)) in enumerate(rank[:4], 1):
        print(f'{i}. {e} nfe={K}: success {c["succ"]}/10, aborted {c["ab"]}/10, distance {c["dist"]:.3f} m')

    print('\n## Existing projected s_curve_hg cells in the same batch (context for R44b; never pooled)\n')
    allc = load(BATCH, lambda t: t in ('u7hg', 'u18sc'))
    print('| tag | model | nfe | variant | success | S&C | aborted | ms/step |')
    print('| :-- | :-- | --: | :-- | --: | --: | --: | --: |')
    for k in sorted(allc):
        e, K, t, v, ctrl = k
        if e != 'fm' or ctrl != 'pid_stopgo' or v == 'diffuser' or 'geo_free' in v:
            continue
        c = cell(allc[k])
        print(f'| {t} | {e} | {K} | {v} | {c["succ"]}/10 | {c["sc"]}/10 | {c["ab"]}/10 | {c["ms"]:.1f} |')


if __name__ == '__main__':
    main()
