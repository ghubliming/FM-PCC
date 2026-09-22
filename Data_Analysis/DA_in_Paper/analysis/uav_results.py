#!/usr/bin/env python3
"""Quadrotor benchmark: the numbers Chapter 6.3 quotes, recomputed from the committed data.

    python3 Data_Analysis/DA_in_Paper/analysis/uav_results.py

Source: Data_Analysis/analysis_results_checkpoint/19-09-UAV-Pillars-Exclude/batch_uav_20260919_111701/
per_rollout_detail.csv. It supersedes the 15-09 batch and adds the 2026-09-18 wave: endpoint projection
under random and minimum-cost selection on corridor v2 (tag u17cv2, K=3 and 5, all three flow models), and
the s-curve endpoint rows re-run after the switched-wall fix (tag u18sc).

🔴 UAV-PILLARS IS EXCLUDED (2026-09-18). `pillars_hg` enforces the constraint the demonstration generator
was built to satisfy -- the demonstrated routes are already 8 cm inside the feasible set -- so its projected
configurations measure how little of an ALREADY-FEASIBLE plan each method disturbs, not whether a method can
make an infeasible plan feasible. Every pillars_hg cell (tags u7hg and u7hga1) is therefore withheld, and
the pillars block below is gated off rather than deleted, so the scene can be restored once the enlarged
geometry (pillars_xl / pillars_xxl, Gen15 U17) has been evaluated. See
logs_in_develop/Writing/Working_Space/data_status/PENDING_20260918_pillars_geometry_redesign.md.
Stdlib only. Cells are selected by the run tag in FolderName, which is how the analyses of record name
them:
* corridor v2  tag @u17cv2, geometry corridor_cv2s_*   (Gen15/U16/DA_20260914_corridor_v2_paper_full.md)
* pillars      tag @u7hg,   geometry pillars_hg_*      (Campaign CLOSURE_20260910 section 2,
                                                         DA_20260912_pillars_diffusion_baseline_reference.md)
* s-curve      listed from the analyses of record; see the s-curve block below.

Success criterion (2026-09-17): the thesis scores a flight by whether it PASSES THE GOAL, i.e. crosses the
finish line of its scene -- the criterion of D3IL-avoiding, transferred to the air. The strict
within-0.30 m-of-the-goal-point columns (`n_success`, `n_success_and_constraints`) are NO LONGER REPORTED:
they measure where a flight stops after passing the goal, not whether it got there, and on scenes whose
constraints push a route off its nominal end point they penalise flights that solved the task. They stay in
the CSV and are printed only in the diagnostic block at the end.

Metric columns (per rollout, 0/1 unless stated):
  success_relaxed                    passed the goal (crossed the finish line)
  n_success_relaxed_and_constraints  passed the goal on a collision-free flight  = S&C, the reported metric
  n_success, n_success_and_constraints   strict goal-point variants, retained for the diagnostic block
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
CSV = os.path.join(REPO, 'Data_Analysis/analysis_results_checkpoint/19-09-UAV-Pillars-Exclude/'
                         'batch_uav_20260919_111701/per_rollout_detail.csv')
# 2026-09-22: flipped back to False by AUTHOR DECISION. The enlarged geometry (pillars_xl, Gen15 U17)
# was evaluated in full and ABANDONED — S&C = 0.00 in all 94 cells; every projector routes into the
# forbidden centre corridor (DA_20260922_pillars_xl_wave.md, Gen15/U17/CLOSURE_20260922_U17_abandoned.md).
# The thesis falls back to pillars_hg WITH the caveat in the module docstring stated in the prose:
# the unprojected flows are already feasible there, so the projected rows measure preservation of a
# feasible plan, not repair of an infeasible one.
PILLARS_EXCLUDED = False
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
            elif r['geo'].startswith('s_curve_hg_') and tag.endswith('u18sc'):
                cells[('s_curve', 'u18sc', r['engine'], int(float(r['K'])), r['variant'])].append(r)
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
    return (f'  {lab:40s} n={c["n"]:2d} passed={c["success_relaxed"]:.2f} '
            f'S&C={c["n_success_relaxed_and_constraints"]:.2f} '
            f'cfree={c["collision_free_completed"]:.2f} viol={c["n_violations"]:.1f} steps={c["n_steps"]:.0f} '
            f'ms={c["avg_time_ms"]:.1f}')


def main():
    C = load()

    print('== corridor v2 (seed 6, 12 flights per cell, routes L/C/R x 4) ==')
    T = '-bounds_free-pdes-tightened'
    for eng, K in (('diffusion', 20), ('mf', 1), ('mf', 3), ('mf', 5), ('af', 1), ('af', 3), ('af', 5),
                   ('fm', 1), ('fm', 3), ('fm', 5)):
        for var in ('diffuser', 'dpcc-c-bounds_free-pdes', f'dpcc-r{T}', f'dpcc-c{T}', f'dpcc-t{T}',
                    f'hardflow_sls{T}', f'hardflow_sls-r{T}', f'hardflow_sls-c{T}', f'hardflow_sls-t{T}'):
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

    print('\n== pillars, geometry pillars_hg ==')
    if PILLARS_EXCLUDED:
        n_cells = len({k for k in C if k[0] == 'pillars'})
        print(f'  EXCLUDED — {n_cells} pillars_hg cells are in the batch and none is reported.')
        print('  The scene enforces the constraint its own demonstrations were generated to satisfy, so its')
        print('  projected configurations do not measure constraint repair. Awaiting pillars_xl / pillars_xxl.')
        print('  PENDING_20260918_pillars_geometry_redesign.md · restore by setting PILLARS_EXCLUDED = False.')
    else:
      for K in (5, 2, 1):
        for eng in ('mf', 'fm', 'af'):
            vs = {k[4]: v for k, v in C.items() if k[0] == 'pillars' and k[2] == eng and k[3] == K}
            if not vs:
                continue
            for label, keep in (('all variants', lambda v: True),
                                ('full constraint set', lambda v: 'geo_free' not in v)):
                sel = {v: agg(rows) for v, rows in vs.items() if keep(v)}
                sc = [c['n_success_relaxed_and_constraints'] for c in sel.values()]
                print(f'  {eng} K{K} {label:20s} variants={len(sel):2d} mean S&C={st.mean(sc):.3f} '
                      f'ceiling={max(sc):.2f} cells>=0.8={sum(s >= 0.8 - 1e-9 for s in sc)}/{len(sc)}')
            best = max(((agg(r)['n_success_relaxed_and_constraints'], v) for v, r in vs.items() if 'geo_free' not in v))
            print(f'      best full-constraint cell: {best[1]} S&C {best[0]:.2f}')
    print('  -- diffusion K20 (reference)')
    for k, rows in sorted(C.items()):
        if k[0] == 'pillars' and k[2] == 'diffusion':
            print(row(f'diffusion K20 {k[4]}', agg(rows)))
    print('  -- per variant at K5, full constraint set (S&C = passed the goal, collision-free)')
    vars5 = sorted({k[4] for k in C if k[0] == 'pillars' and k[3] == 5 and 'geo_free' not in k[4]})
    for v in vars5:
        line = []
        for eng in ('mf', 'fm', 'af'):
            rows = C.get(('pillars', 'u7hg', eng, 5, v))
            line.append(f'{eng}={agg(rows)["n_success_relaxed_and_constraints"]:.2f}' if rows else f'{eng}=  - ')
        print(f'  {v:26s} ' + '  '.join(line))
    # Thesis Table tab:uav-pillars-best and the step-budget paragraph (v3.26).
    print('  -- best PROJECTED configuration per model, full constraint set (tab:uav-pillars-best)')
    for eng, K in (('mf', 5), ('fm', 5), ('af', 5), ('diffusion', 20)):
        vs = {k[4]: v for k, v in C.items() if k[0] == 'pillars' and k[2] == eng and k[3] == K
              and 'geo_free' not in k[4] and k[4] != 'diffuser'}
        # at equal S&C prefer the cheaper configuration (Pareto: same success, less time)
        v, rows = max(vs.items(), key=lambda kv: (agg(kv[1])['n_success_relaxed_and_constraints'],
                                                  -agg(kv[1])['avg_time_ms']))
        c = agg(rows)
        print(f'  {eng:9s} K{K:<2d} {v:20s} S&C={c["n_success_relaxed_and_constraints"]:.2f} '
              f'passed={c["success_relaxed"]:.2f} viol={c["n_violations"]:.1f} '
              f'steps={c["n_steps"]:.0f} ms={c["avg_time_ms"]:.1f} goal_dist={c["goal_dist"]:.3f}')
    SEVEN = ('diffuser', 'dpcc-r', 'dpcc-c', 'dpcc-t', 'dpcc-r-tightened', 'dpcc-c-tightened', 'dpcc-t-tightened')
    print('  -- mean S&C over the seven configurations evaluated at every budget')
    for eng, K in (('mf', 2), ('mf', 5), ('fm', 2), ('fm', 5), ('af', 1), ('af', 2), ('af', 5)):
        cs = [agg(C[('pillars', 'u7hg', eng, K, v)]) for v in SEVEN if ('pillars', 'u7hg', eng, K, v) in C]
        print(f'  {eng} K{K} configs={len(cs)} mean S&C={st.mean(c["n_success_relaxed_and_constraints"] for c in cs):.3f} '
              f'mean ms/step={st.mean(c["avg_time_ms"] for c in cs):.1f}')
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
    sc = [agg(v)['n_success_relaxed_and_constraints'] for v in valid.values()]
    print(f'  s_curve_hg cells without endpoint projection: {len(valid)}, S&C ceiling {max(sc):.2f}, '
          f'cells > 0: {sum(x > 0 for x in sc)}')
    print('  -- unprojected plan, by controller')
    for k in sorted(valid):
        if k[3] == 'diffuser':
            c = agg(valid[k])
            mz = min(f(r['phys_min_z']) for r in valid[k] if f(r.get('phys_min_z')) is not None)
            ps = st.mean(f(r['phys_safe']) for r in valid[k] if f(r.get('phys_safe')) is not None)
            print(f'  {k[0]:9s} K{k[1]:<2d} {k[2]:12s} n={c["n"]:2d} passed={c["success_relaxed"]:.2f} goal dist={c["goal_dist"]:.3f} '
                  f'phys safe={ps:.2f} min z={mz:.3f} S&C={c["n_success_relaxed_and_constraints"]:.2f} '
                  f'viol={c["n_violations"]:.1f} ms={c["avg_time_ms"]:.1f}')
    print('  -- per-step projection (dpcc-*), by controller')
    for k in sorted(valid):
        if k[3].startswith('dpcc') and k[2] == 'mjpc':
            c = agg(valid[k])
            gr = st.mean(f(r['goal_reached']) for r in valid[k])
            print(f'  {k[0]:9s} K{k[1]:<2d} {k[2]:12s} {k[3]:22s} n={c["n"]:2d} goal reached={gr:.2f} '
                  f'goal dist={c["goal_dist"]:.3f} S&C={c["n_success_relaxed_and_constraints"]:.2f} ms={c["avg_time_ms"]:.1f}')
    for k in sorted(valid):
        if k[3].startswith('dpcc') and k[2] != 'mjpc' and k[0] == 'mf' and k[1] == 10:
            c = agg(valid[k])
            gr = st.mean(f(r['goal_reached']) for r in valid[k])
            print(f'  {k[0]:9s} K{k[1]:<2d} {k[2]:12s} {k[3]:22s} n={c["n"]:2d} goal reached={gr:.2f} '
                  f'goal dist={c["goal_dist"]:.3f} S&C={c["n_success_relaxed_and_constraints"]:.2f} ms={c["avg_time_ms"]:.1f}')

    # ── s-curve, the 2026-09-18 re-run after the switched-wall fix (tag u18sc) ────────────
    # These rows REPLACE the pre-fix endpoint rows withheld by the guard in sec:res:uav:projection
    # (ledger R10). They are reportable, but only beside their divergence counts: the drone
    # inverts mid-flight on a large share of flights, INCLUDING on the unprojected plan, so the
    # scores describe the stability of the scene rather than a ranking of the methods.
    print('\n== s-curve, endpoint projection re-run post switched-wall fix (tag u18sc, seed 6) ==')
    print('  aborted = flights ended early by the divergence guard (drone inverted); they count as failures')
    for eng, K in (('fm', 20), ('mf', 10), ('af', 5)):
        cells = {k[4]: v for k, v in C.items() if k[0] == 's_curve' and k[2] == eng and k[3] == K}
        if not cells:
            continue
        tot_ab = tot_n = 0
        for v in ('diffuser', 'dpcc-t', 'hardflow_sls', 'hardflow_sls-r', 'hardflow_sls-c', 'hardflow_sls-t'):
            rows = cells.get(v)
            if not rows:
                continue
            c = agg(rows)
            ab = sum(1 for r in rows if f(r.get('divergence_aborted')) == 1)
            tot_ab += ab
            tot_n += len(rows)
            print(f'  {eng:3s} K{K:<2d} {v:16s} n={c["n"]:2d} passed={c["success_relaxed"]:.2f} '
                  f'S&C={c["n_success_relaxed_and_constraints"]:.2f} cfree={c["collision_free_completed"]:.2f} '
                  f'aborted={ab}/{len(rows)} ms={c["avg_time_ms"]:.1f}')
        print(f'  {eng:3s} K{K:<2d} {"ALL VARIANTS":16s} aborted={tot_ab}/{tot_n} '
              f'({100.0 * tot_ab / tot_n:.0f}% of flights)')


if __name__ == '__main__':
    main()
