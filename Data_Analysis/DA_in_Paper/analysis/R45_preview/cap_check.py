#!/usr/bin/env python3.14
# R45 PREVIEW (v3.81 writing agent, 2026-09-24) -- not the DA of record; the DA agent redoes R45a properly.
# Spec: logs_in_develop/Writing/Working_Space/data_status/PENDING_20260924_uav_corridor_step_cap_R45.md
#   python3.14 Data_Analysis/DA_in_Paper/analysis/R45_preview/cap_check.py
"""Corridor v3 (R33): flights that end on the 396-step cap without reaching the finish line, and whether they
would reach it with more steps if they kept their end-of-flight trend. Read-only; first ten flights per cell."""
import sys, os, numpy as np
sys.path.insert(0, '/workspaces/FM-PCC/Data_Analysis/DA_in_Paper/analysis'); sys.path.insert(0, '/workspaces/FM-PCC')
import corridor_v3_grid as G
import uav_expert_data_collect.generator as gen
CAP, R_GOAL, X_LINE = 396, 0.30, 2.8
HOM = 'LCR'

def goal_for(i):
    rng = np.random.default_rng(10_000 + i)
    traj_fn, init, dur = gen._build_traj_and_init('corridor', HOM[i % 3], rng)
    return np.asarray(traj_fn(dur)[0], float), init

GOALS = [goal_for(i) for i in range(10)]

def steps_needed(o, goal, win):
    """steps beyond the cap to cross, from the mean velocity of the last `win` control steps (None = never)."""
    P = o[:, 3:6]
    v = (P[-1] - P[-1 - win]) / win
    need = []
    if v[0] > 1e-6:
        need.append((X_LINE - P[-1, 0]) / v[0])                       # the finish plane x >= 2.8
    d = lambda p: np.linalg.norm(p - goal)
    dd = (d(P[-1]) - d(P[-1 - win])) / win
    if dd < -1e-6:
        need.append((d(P[-1]) - R_GOAL) / (-dd))                        # the 0.30 m capture sphere
    return (min(need) if need else None), v, d(P[-1])

rows = []
for geo in ('tilt', 'hump'):
    for lab, eng, Ks in G.MODELS:
        for K in Ks:
            vs = [('none', G.UNPROJ)] + [('ps-' + k, v) for k, v in G.PER_STEP.items()] + \
                 ([('ep-' + k, v) for k, v in G.ENDPOINT.items()] if (K >= 3 and eng != 'diffusion') else [])
            for vn, v in vs:
                c = G.load_cell(geo, eng, K, v)
                for i, o in enumerate(c['obs']):
                    succ, steps = c['succ'][i], c['steps'][i]
                    rows.append(dict(geo=geo, model=lab.split()[0], eng=eng, K=K, var=vn, i=i, succ=bool(succ),
                                     steps=int(steps), obs=o, sc=bool(c['sc'][i]), vf=bool(c['cf'][i])))

tot = len(rows)
capped = [r for r in rows if r['steps'] >= CAP and not r['succ']]
other_fail = [r for r in rows if not r['succ'] and r['steps'] < CAP]
print(f'flights read: {tot}  (136 cells x 10)')
print(f'not successful: {sum(not r["succ"] for r in rows)}  = on the {CAP}-step cap without the line: {len(capped)}  + other: {len(other_fail)}')
from collections import Counter
print('capped by group:', Counter((r['geo'], r['eng'], r['K'], r['var']) for r in capped).most_common())

print('\n== extrapolation of the capped flights (constant mean velocity over the last W steps) ==')
for win in (10, 30, 60):
    res = []
    for r in capped:
        g, _ = GOALS[r['i']]
        n, v, dg = steps_needed(r['obs'], g, win)
        res.append((r, n, v, dg))
    for grp_name, pred in (('baseline projected', lambda r: r['eng'] == 'diffusion'),
                           ('flow per-step K<=2 (hump)', lambda r: r['eng'] != 'diffusion')):
        sub = [(r, n, v, dg) for r, n, v, dg in res if pred(r)]
        if not sub: continue
        never = sum(1 for _, n, _, _ in sub if n is None)
        within = {f'+{p}%': sum(1 for _, n, _, _ in sub if n is not None and n <= CAP * p / 100) for p in (10, 25, 50, 100)}
        print(f' W={win:2d}  {grp_name:26s} n={len(sub):3d}  would cross within {within}  never (stalled/away): {never}')

print('\n== the baseline capped flights, one by one (W=30) ==')
for r in capped:
    if r['eng'] != 'diffusion': continue
    g, init = GOALS[r['i']]
    n, v, dg = steps_needed(r['obs'], g, 30)
    P = r['obs'][:, 3:6]
    print(f"  {r['geo']:4} {r['var']:5} trial {r['i']} route {HOM[r['i']%3]}  end x {P[-1,0]:5.2f} y {P[-1,1]:+.2f} z {P[-1,2]:.2f}  "
          f"goal z {g[2]:.2f}  dist {dg:.2f}  vx/step {v[0]*1000:5.1f} mm  vz/step {v[2]*1000:+5.1f} mm  extra steps "
          + (f'{n:6.0f}' if n is not None else ' never') + f"  viol-free {r['vf']}")

print('\n== the flow capped flights: how far they move in their last 60 steps ==')
mv = [np.linalg.norm(r['obs'][-1, 3:6] - r['obs'][-61, 3:6]) for r in capped if r['eng'] != 'diffusion']
if mv: print(f'  n={len(mv)}  displacement over the last 60 steps: median {np.median(mv)*100:.1f} cm, max {max(mv)*100:.1f} cm')

print('\n== steps taken by the flights that DID reach the line (projected) ==')
for grp, pred in (('flow projected', lambda r: r['eng'] != 'diffusion' and r['var'] != 'none'),
                  ('baseline projected', lambda r: r['eng'] == 'diffusion' and r['var'] != 'none'),
                  ('unprojected (all)', lambda r: r['var'] == 'none')):
    st = [r['steps'] for r in rows if pred(r) and r['succ']]
    if st: print(f'  {grp:20s} n={len(st):4d}  steps median {np.median(st):.0f}  p95 {np.percentile(st,95):.0f}  max {max(st)}  (cap {CAP})')

print('\n== baseline: S&C now vs. if every capped flight kept its end trend (budget +25 % / +50 %) ==')
for geo in ('tilt', 'hump'):
    for vn in ('ps-r', 'ps-c', 'ps-t'):
        cell = [r for r in rows if r['geo'] == geo and r['eng'] == 'diffusion' and r['var'] == vn]
        now = sum(r['sc'] for r in cell)
        add = {}
        for p in (25, 50):
            extra = 0
            for r in cell:
                if r['succ'] or not r['vf']: continue
                g, _ = GOALS[r['i']]
                n, _, _ = steps_needed(r['obs'], g, 30)
                if n is not None and n <= CAP * p / 100: extra += 1
            add[p] = extra
        vf_capped = sum(1 for r in cell if not r['succ'] and r['vf'])
        print(f'  {geo} {vn}: S&C now {now}/10; capped & violation-free {vf_capped}; S&C at +25 %: {now+add[25]}/10, at +50 %: {now+add[50]}/10')

print('\n== flow hump K<=2 capped flights: extra steps needed at the end trend (W=30) ==')
ns = []
for r in capped:
    if r['eng'] == 'diffusion': continue
    g, _ = GOALS[r['i']]
    n, v, dg = steps_needed(r['obs'], g, 30)
    ns.append(n if n is not None else np.inf)
ns = np.array(ns)
print(f'  n={len(ns)}  within +100 %: {(ns <= CAP).sum()}  median extra steps {np.median(ns):.0f}  (x the cap: {np.median(ns)/CAP:.0f})')

print('\n== successful flights that ran to the cap (crossed the plane x>=2.8 but never entered the 0.30 m sphere) ==')
for grp, pred in (('flow projected', lambda r: r['eng'] != 'diffusion' and r['var'] != 'none'),
                  ('baseline projected', lambda r: r['eng'] == 'diffusion' and r['var'] != 'none')):
    s = [r for r in rows if pred(r) and r['succ'] and r['steps'] >= CAP]
    after = 0
    for r in s:
        P = r['obs'][:, 3:6]
        k = np.argmax(P[:, 0] >= X_LINE) if (P[:, 0] >= X_LINE).any() else None
        after += (len(P) - k) if k is not None else 0
    fx = [r['obs'][-1, 3] for r in s]
    print(f'  {grp:20s} {len(s)} flights; steps flown after crossing x=2.8: {after} in total; final x {min(fx) if fx else 0:.2f}–{max(fx) if fx else 0:.2f}')
