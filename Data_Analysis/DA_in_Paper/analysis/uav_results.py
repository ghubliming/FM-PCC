#!/usr/bin/env python3
"""Quadrotor benchmark: the numbers Chapter 6.3 quotes, recomputed from the committed data.

    python3 Data_Analysis/DA_in_Paper/analysis/uav_results.py

Source: Data_Analysis/analysis_results_checkpoint/15-09/batch_uav_20260915_100816/per_rollout_detail.csv.
Stdlib only. Cells are selected by the run tag in FolderName, which is how the analyses of record name
them:
* corridor v2  tag @u17cv2, geometry corridor_cv2s_*   (Gen15/U16/DA_20260914_corridor_v2_paper_full.md)
* pillars      tag @u7hg,   geometry pillars_hg_*      (Campaign CLOSURE_20260910 section 2,
                                                         DA_20260912_pillars_diffusion_baseline_reference.md)
* s-curve      listed from the analyses of record; see the s-curve block below.

Metric columns (per rollout, 0/1 unless stated):
  n_success                          reached within 0.30 m of the route's goal point  ("strict")
  success_relaxed                    crossed the finish line                           ("crossed")
  n_success_and_constraints          strict and collision-free
  n_success_relaxed_and_constraints  crossed and collision-free
  collision_free_completed           no violating control step, drone body radius 0.31 m
  n_violations                       violating control steps (count)
  avg_time_ms                        wall clock per control step

Pillars variants: the 8 `*-geo_free` rows switch the obstacle constraints OFF inside the projection.
They are ablations; the thesis never ranks them against full-constraint rows, so every pillars
aggregate is printed twice, with and without them.
"""
import csv
import math
import os
import statistics as st
from collections import defaultdict

REPO = '/workspaces/FM-PCC'
CSV = os.path.join(REPO, 'Data_Analysis/analysis_results_checkpoint/15-09/'
                         'batch_uav_20260915_100816/per_rollout_detail.csv')
M = ('n_success', 'success_relaxed', 'n_success_and_constraints', 'n_success_relaxed_and_constraints',
     'collision_free_completed', 'n_violations', 'n_steps', 'avg_time_ms', 'goal_dist')


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load():
    cells = defaultdict(list)          # (scene, tag, engine, K, variant) -> rows
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            fn = r['FolderName']
            tag = fn.rsplit('@', 1)[-1] if '@' in fn else ''
            if r['geo'].startswith('corridor_cv2s') and tag.endswith('u17cv2'):
                cells[('corridor_v2', 'u17cv2', r['engine'], int(float(r['K'])), r['variant'])].append(r)
            elif r['geo'].startswith('pillars_hg_') and tag.endswith('u7hg'):
                cells[('pillars', 'u7hg', r['engine'], int(float(r['K'])), r['variant'])].append(r)
    return cells


def agg(rows):
    out = {'n': len(rows)}
    for m in M:
        vals = [f(r.get(m)) for r in rows if f(r.get(m)) is not None]
        out[m] = st.mean(vals) if vals else float('nan')
    return out


def fisher_two_sided(a, n1, b, n2):
    """Exact two-sided Fisher test for a/n1 vs b/n2."""
    tot = a + b
    def p(k):
        return math.comb(n1, k) * math.comb(n2, tot - k) / math.comb(n1 + n2, tot)
    obs = p(a)
    return min(1.0, sum(p(k) for k in range(max(0, tot - n2), min(tot, n1) + 1) if p(k) <= obs + 1e-12))


def row(lab, c):
    return (f'  {lab:40s} n={c["n"]:2d} strict={c["n_success"]:.2f} crossed={c["success_relaxed"]:.2f} '
            f'S&C strict={c["n_success_and_constraints"]:.2f} S&C crossed={c["n_success_relaxed_and_constraints"]:.2f} '
            f'cfree={c["collision_free_completed"]:.2f} viol={c["n_violations"]:.1f} steps={c["n_steps"]:.0f} '
            f'ms={c["avg_time_ms"]:.1f}')


def main():
    C = load()

    print('== corridor v2 (seed 6, 12 flights per cell, routes L/C/R x 4) ==')
    T = '-bounds_free-pdes-tightened'
    for eng, K in (('diffusion', 20), ('mf', 1), ('mf', 3), ('mf', 5), ('af', 1), ('af', 3), ('af', 5),
                   ('fm', 1), ('fm', 3), ('fm', 5)):
        for var in ('diffuser', f'dpcc-r{T}', f'dpcc-c{T}', f'dpcc-t{T}',
                    f'hardflow_sls{T}', f'hardflow_sls-t{T}'):
            rows = C.get(('corridor_v2', 'u17cv2', eng, K, var))
            if rows:
                print(row(f'{eng} K{K} {var.replace(T, "*")}', agg(rows)))
    print('  -- tests (collision-free counts out of 12)')
    def cnt(eng, K, var, m='collision_free_completed'):
        rows = C[('corridor_v2', 'u17cv2', eng, K, var)]
        return int(round(sum(f(r[m]) for r in rows))), len(rows)
    for a, b, m in ((('mf', 3, f'dpcc-t{T}'), ('fm', 3, f'dpcc-t{T}'), 'collision_free_completed'),
                    (('mf', 3, f'dpcc-t{T}'), ('diffusion', 20, f'dpcc-t{T}'), 'n_success_relaxed_and_constraints'),
                    (('mf', 3, f'dpcc-t{T}'), ('mf', 3, 'diffuser'), 'collision_free_completed'),
                    (('mf', 3, f'dpcc-t{T}'), ('mf', 3, f'hardflow_sls-t{T}'), 'collision_free_completed'),
                    (('af', 3, f'dpcc-r{T}'), ('mf', 3, f'dpcc-r{T}'), 'n_success_and_constraints')):
        x, n1 = cnt(*a, m=m)
        y, n2 = cnt(*b, m=m)
        print(f'  {m:34s} {a[0]} K{a[1]} {a[2].replace(T,"*"):18s} {x}/{n1} vs {b[0]} K{b[1]} '
              f'{b[2].replace(T,"*"):18s} {y}/{n2}  Fisher p={fisher_two_sided(x, n1, y, n2):.2g}')

    print('\n== pillars, geometry pillars_hg (seed 6, 10 flights per cell) ==')
    for K in (5, 2, 1):
        for eng in ('mf', 'fm', 'af'):
            vs = {k[4]: v for k, v in C.items() if k[0] == 'pillars' and k[2] == eng and k[3] == K}
            if not vs:
                continue
            for label, keep in (('all variants', lambda v: True),
                                ('full constraint set', lambda v: 'geo_free' not in v)):
                sel = {v: agg(rows) for v, rows in vs.items() if keep(v)}
                sc = [c['n_success_and_constraints'] for c in sel.values()]
                print(f'  {eng} K{K} {label:20s} variants={len(sel):2d} mean S&C strict={st.mean(sc):.3f} '
                      f'ceiling={max(sc):.2f} cells>=0.8={sum(s >= 0.8 - 1e-9 for s in sc)}/{len(sc)}')
            best = max(((agg(r)['n_success_and_constraints'], v) for v, r in vs.items() if 'geo_free' not in v))
            print(f'      best full-constraint cell: {best[1]} S&C strict {best[0]:.2f}')
    print('  -- diffusion K20 (reference)')
    for k, rows in sorted(C.items()):
        if k[0] == 'pillars' and k[2] == 'diffusion':
            print(row(f'diffusion K20 {k[4]}', agg(rows)))
    print('  -- per variant at K5, full constraint set (S&C strict)')
    vars5 = sorted({k[4] for k in C if k[0] == 'pillars' and k[3] == 5 and 'geo_free' not in k[4]})
    for v in vars5:
        line = []
        for eng in ('mf', 'fm', 'af'):
            rows = C.get(('pillars', 'u7hg', eng, 5, v))
            line.append(f'{eng}={agg(rows)["n_success_and_constraints"]:.2f}' if rows else f'{eng}=  - ')
        print(f'  {v:26s} ' + '  '.join(line))
    print('  -- K5, by projection method. phys_safe = no contact, airborne, not diverged (eval_mix_uav.py:1849);')
    print('     it is NOT constraint satisfaction, which is collision_free_completed.')
    for label, keep in (('all variants', lambda v: True), ('full constraint set', lambda v: 'geo_free' not in v)):
        for fam in ('dpcc', 'hardflow'):
            rows = [r for k, v in C.items() if k[0] == 'pillars' and k[3] == 5 and k[4].startswith(fam)
                    and keep(k[4]) for r in v]
            ps = sum(1 for r in rows if f(r.get('phys_safe')) == 0)
            cf = sum(1 for r in rows if f(r.get('collision_free_completed')) == 0)
            ms = st.mean(f(r['avg_time_ms']) for r in rows)
            pm = st.mean(f(r['proj_ms']) for r in rows if f(r.get('proj_ms')) is not None)
            print(f'  {label:20s} {fam:9s} flights={len(rows):3d} physically unsafe={ps:3d} '
                  f'not collision-free={cf:3d}  ms/step={ms:.1f} proj ms={pm:.1f}')

    print('\n== s-curve (listing only; seed 6) ==')
    SC = defaultdict(list)
    with open(CSV) as fh:
        for r in csv.DictReader(fh):
            if r['scene'] == 's_curve' and r['geo'].startswith('s_curve_hg_'):
                SC[(r['engine'], int(float(r['K'])), r.get('controller', ''), r['variant'])].append(r)
    valid = {k: v for k, v in SC.items() if not k[3].startswith('hardflow')}
    sc = [agg(v)['n_success_and_constraints'] for v in valid.values()]
    print(f'  s_curve_hg cells without endpoint projection: {len(valid)}, S&C strict ceiling {max(sc):.2f}, '
          f'cells > 0: {sum(x > 0 for x in sc)}')
    print('  -- unprojected plan, by controller')
    for k in sorted(valid):
        if k[3] == 'diffuser':
            c = agg(valid[k])
            gr = st.mean(f(r['goal_reached']) for r in valid[k] if f(r.get('goal_reached')) is not None)
            mz = min(f(r['phys_min_z']) for r in valid[k] if f(r.get('phys_min_z')) is not None)
            ps = st.mean(f(r['phys_safe']) for r in valid[k] if f(r.get('phys_safe')) is not None)
            print(f'  {k[0]:9s} K{k[1]:<2d} {k[2]:12s} n={c["n"]:2d} goal reached={gr:.2f} goal dist={c["goal_dist"]:.3f} '
                  f'phys safe={ps:.2f} min z={mz:.3f} S&C strict={c["n_success_and_constraints"]:.2f} '
                  f'viol={c["n_violations"]:.1f} ms={c["avg_time_ms"]:.1f}')
    print('  -- per-step projection (dpcc-*), by controller')
    for k in sorted(valid):
        if k[3].startswith('dpcc') and k[2] == 'mjpc':
            c = agg(valid[k])
            gr = st.mean(f(r['goal_reached']) for r in valid[k])
            print(f'  {k[0]:9s} K{k[1]:<2d} {k[2]:12s} {k[3]:22s} n={c["n"]:2d} goal reached={gr:.2f} '
                  f'goal dist={c["goal_dist"]:.3f} S&C={c["n_success_and_constraints"]:.2f} ms={c["avg_time_ms"]:.1f}')
    for k in sorted(valid):
        if k[3].startswith('dpcc') and k[2] != 'mjpc' and k[0] == 'mf' and k[1] == 10:
            c = agg(valid[k])
            gr = st.mean(f(r['goal_reached']) for r in valid[k])
            print(f'  {k[0]:9s} K{k[1]:<2d} {k[2]:12s} {k[3]:22s} n={c["n"]:2d} goal reached={gr:.2f} '
                  f'goal dist={c["goal_dist"]:.3f} S&C={c["n_success_and_constraints"]:.2f} ms={c["avg_time_ms"]:.1f}')


if __name__ == '__main__':
    main()
