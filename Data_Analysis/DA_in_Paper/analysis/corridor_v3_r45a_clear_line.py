#!/usr/bin/env python3.14
"""R45a — UAV-corridor v3 re-scored with the finish line at the END OF THE CORRIDOR (x' = 2.0 m), beside the original rule.

    python3.14 Data_Analysis/DA_in_Paper/analysis/corridor_v3_r45a_clear_line.py            # markdown report to stdout
    python3.14 …/corridor_v3_r45a_clear_line.py --json r45a.json                             # + every flight as JSON

Spec : logs_in_develop/Writing/Working_Space/data_status/PENDING_20260924_uav_corridor_step_cap_R45.md §2 (author, 24-09).
Data : the corpus of corridor_v3_grid.py — outcomes from batch_uav_20260924_081422/per_rollout_detail.csv, flown paths from
       the npz in temp/23-09-Corridor-TEMP/plans; the first ten flights of each of the 136 cells (1 360 flights).
Rule :
  success (old)   = success_relaxed: the eval's crossing latch (within 0.30 m of the goal point, or past the plane through it)
                    AND safe (contact ≤ scene limit, lowest altitude > 0.2 m, no divergence abort)
  success_clear   = success (old) OR (the centre reaches x ≥ x' within the 396-step budget AND safe AND not aborted)
  S&C (old / clear) = the success flag AND violation-free, violations counted over the whole flight exactly as before
  "the centre reaches x ≥ x'" is read, like the eval's own crossing latch, on the position AFTER each control step's physics:
  obs_all holds the position at the START of each step, so the positions after the steps are obs_all[1:] plus the final one.
  The final position is not stored as such; it is recovered from the stored final goal distance (`goal_dist`) and final
  altitude (`phys_final_z`) against the goal point rebuilt with the eval's seeding, the final y taken one step ahead of the
  last recorded y. On the 222 capped flights this equals `last recorded x + last step Δx` to within 0.52 mm. The reading on
  recorded positions only is reported beside it (`--recorded-only`); the two differ on exactly one flight (§5).
Checks on every flight the moved line turns into a success (spec §2): (1) past x' before the cap, (2) no abort and safe,
(3) still moving toward the goal at the cap (pace over the last 30 control steps, toward the goal point rebuilt with the
eval's own seeding), (4) no violating step past x'.
Stdlib + numpy + yaml (+ the generator module for the goal points). Claude (Opus 5.5, Claude Code, U19 corridor chat), 2026-09-24.
"""
import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/workspaces/FM-PCC')
import corridor_v3_grid as G                                      # noqa: E402
import uav_expert_data_collect.generator as gen                  # noqa: E402

X_CLEAR = 2.0
SENS_LINES = (2.00, 2.31, 2.36, 2.50)       # 2.31 / 2.36: + rotor reach (+ cap radius); 2.50: aligned flights enter the goal sphere
CAP = 396
PACE_WIN = 30
HOM = 'LCR'


def goal_for(i):
    rng = np.random.default_rng(10_000 + i)
    traj_fn, init, dur = gen._build_traj_and_init('corridor', HOM[i % 3], rng)
    return np.asarray(traj_fn(dur)[0], float)


GOALS = [goal_for(i) for i in range(G.N_READ)]
VARS = lambda eng, K: [('none', G.UNPROJ)] + [('ps-' + k, v) for k, v in G.PER_STEP.items()] + \
    ([('ep-' + k, v) for k, v in G.ENDPOINT.items()] if (K >= 3 and eng != 'diffusion') else [])


def bfloat(row, col):
    v = row.get(col, '')
    return float(v) if v not in ('', 'nan', None) else float('nan')


def flights():
    out = []
    for geo in G.GEOS:
        cfg = G.geo_config(geo)
        for lab, eng, Ks in G.MODELS:
            for K in Ks:
                for vn, v in VARS(eng, K):
                    c = G.load_cell(geo, eng, K, v)
                    brows = G.BATCHROWS.get((geo, eng, K, v), {})
                    for i, o in enumerate(c['obs']):
                        P = o[:, 3:6]
                        x = P[:, 0]
                        br = brows.get(i, {})
                        safe = bfloat(br, 'phys_safe') == 1.0
                        aborted = bfloat(br, 'divergence_aborted') == 1.0
                        viol_steps = [k for k in range(len(o)) if G.NS['_exec_constraint_violations']([o[k].tolist()], cfg)[1]]
                        g = GOALS[i]
                        w = min(PACE_WIN, len(P) - 1)
                        vx = (P[-1, 0] - P[-1 - w, 0]) / w
                        dg = (np.linalg.norm(P[-1] - g) - np.linalg.norm(P[-1 - w] - g)) / w
                        first_clear = int(np.argmax(x >= X_CLEAR)) if (x >= X_CLEAR).any() else None
                        # final position after the last step (not in obs_all): from the stored final goal distance + altitude
                        dist_f, z_f = bfloat(br, 'goal_dist'), bfloat(br, 'phys_final_z')
                        y_f = P[-1, 1] + (P[-1, 1] - P[-2, 1])
                        rad = dist_f ** 2 - (y_f - g[1]) ** 2 - (z_f - g[2]) ** 2
                        x_final = float(g[0] - math.sqrt(rad)) if (rad > 0 and x[-1] < g[0]) else float('nan')
                        out.append(dict(
                            geo=geo, model=lab.replace(' a0.2', ''), eng=eng, K=K, var=vn, variant=v, i=i, route=HOM[i % 3],
                            steps=int(c['steps'][i]), succ=bool(c['succ'][i]), vf=bool(c['cf'][i]), sc=bool(c['sc'][i]),
                            safe=safe, aborted=aborted, min_z=bfloat(br, 'phys_min_z'), contact=bfloat(br, 'phys_contact_frac'),
                            n_viol=int(c['viol'][i]), n_viol_rescored=len(viol_steps),
                            viol_past=[sum(1 for k in viol_steps if x[k] >= L) for L in SENS_LINES],
                            xmax=float(x.max()), x_last=float(x[-1]), first_clear=first_clear, x_final=x_final,
                            x_reach=float(np.nanmax([x.max(), x_final])),
                            last_pace=float(P[-1, 0] - P[-2, 0]), pace_x=float(vx), pace_goal=float(dg),
                            goal_dist_end=float(np.linalg.norm(P[-1] - g)), y_end=float(P[-1, 1]), z_end=float(P[-1, 2])))
    return out


RECORDED_ONLY = False


def rules(f, L=X_CLEAR):
    reach = f['xmax'] if RECORDED_ONLY else f['x_reach']
    s_new = f['succ'] or (reach >= L and f['safe'] and not f['aborted'])
    return s_new, (s_new and f['vf'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--json', default=None)
    ap.add_argument('--recorded-only', action='store_true', help="read x' on the recorded start-of-step positions only")
    a = ap.parse_args()
    global RECORDED_ONLY
    RECORDED_ONLY = a.recorded_only
    F = flights()
    cell = defaultdict(list)
    for f in F:
        cell[(f['geo'], f['eng'], f['K'], f['var'])].append(f)
    print(f'# R45a — corridor v3 with the finish line at the end of the corridor (x\' = {X_CLEAR} m)'
          f' — reach read on {"recorded start-of-step positions only" if RECORDED_ONLY else "the position after every step (eval latch timing)"}\n')
    chk = [f['x_final'] - (f['x_last'] + f['last_pace']) for f in F if f['steps'] >= CAP and not math.isnan(f['x_final'])]
    print(f'final position recovered for {len(chk)} flights on the cap; minus (last recorded x + last step Δx): '
          f'median {np.median(chk) * 1000:+.2f} mm, max |.| {np.max(np.abs(chk)) * 1000:.2f} mm\n')

    # ── corpus facts the spec states (verify) ─────────────────────────────────────────────────────────────────────
    fail = [f for f in F if not f['succ']]
    capped = [f for f in fail if f['steps'] >= CAP]
    print('## 1 · Corpus facts (spec §1–§2, verified)\n')
    print(f'- flights read: {len(F)} ({len(cell)} cells × {G.N_READ}); re-scored violating steps equal the stored ones on '
          f'{sum(f["n_viol"] == f["n_viol_rescored"] for f in F)}/{len(F)}')
    print(f'- not successful under the old rule: {len(fail)}; on the {CAP}-step cap: {len(capped)}; '
          f'aborted {sum(f["aborted"] for f in F)}; unsafe {sum(not f["safe"] for f in F)} (min altitude over all flights '
          f'{min(f["min_z"] for f in F):.2f} m, max contact fraction {max(f["contact"] for f in F):.4f})')
    grp = Counter(('baseline' if f['eng'] == 'diffusion' else 'flow', f['geo'], f['var']) for f in fail)
    print(f'- by group: ' + '; '.join(f'{k[0]} {k[1]} {k[2]} {n}' for k, n in sorted(grp.items())))
    for j, L in enumerate(SENS_LINES):
        vf_ = [f for f in F if f['viol_past'][j] > 0]
        print(f'- violating steps at x ≥ {L:.2f}: {sum(f["viol_past"][j] for f in F)} in {len(vf_)} flight(s)'
              + (': ' + ', '.join(f'{f["geo"]} {f["model"]} K{f["K"]} {f["var"]} trial {f["i"]} ({f["viol_past"][j]})' for f in vf_) if vf_ else ''))

    # ── every cell, both rules ────────────────────────────────────────────────────────────────────────────────────
    print('\n## 2 · Every cell, both rules (success / S&C, of 10)\n')
    print('| block | model | nfe | variant | success old → clear | S&C old → clear | changed |')
    print('| :-- | :-- | --: | :-- | :-- | :-- | :-- |')
    changed_cells = []
    for key in sorted(cell, key=lambda k: (list(G.GEOS).index(k[0]), [m[1] for m in G.MODELS].index(k[1]), k[2], k[3])):
        fs = cell[key]
        so = sum(f['succ'] for f in fs); sco = sum(f['sc'] for f in fs)
        sn = sum(rules(f)[0] for f in fs); scn = sum(rules(f)[1] for f in fs)
        ch = (sn != so) or (scn != sco)
        if ch:
            changed_cells.append(key)
        print(f'| {key[0]} | {fs[0]["model"]} | {key[2]} | {key[3]} | {so} → {sn} | {sco} → {scn} | {"**yes**" if ch else ""} |')
    print(f'\ncells that change: {len(changed_cells)} of {len(cell)}; in S&C: '
          f'{sum(1 for k in changed_cells if sum(rules(f)[1] for f in cell[k]) != sum(f["sc"] for f in cell[k]))}')

    # ── sensitivity to the line's position ───────────────────────────────────────────────────────────────────────
    print('\n## 3 · Sensitivity: S&C of the cells that change, at x\' = ' + ' / '.join(f'{L:.2f}' for L in SENS_LINES) + '\n')
    print('| block | model | nfe | variant | S&C old | ' + ' | '.join(f'x\'={L:.2f}' for L in SENS_LINES) + ' |')
    print('| :-- | :-- | --: | :-- | --: | ' + ' | '.join('--:' for _ in SENS_LINES) + ' |')
    for key in changed_cells:
        fs = cell[key]
        print(f'| {key[0]} | {fs[0]["model"]} | {key[2]} | {key[3]} | {sum(f["sc"] for f in fs)} | '
              + ' | '.join(str(sum(rules(f, L)[1] for f in fs)) for L in SENS_LINES) + ' |')

    # ── the flights the moved line turns into a success: the four checks ─────────────────────────────────────────
    flips = [f for f in F if rules(f)[0] and not f['succ']]
    print(f'\n## 4 · The {len(flips)} flights the moved line turns into a success — the four checks\n')
    print('| block | model | nfe | variant | trial | route | x max recorded | x after the last step | first recorded step at x ≥ 2.0 / steps | safe, no abort | pace at the cap: x (mm/step) · toward goal (mm/step) | violating steps past 2.0 | violation-free → S&C |')
    print('| :-- | :-- | --: | :-- | --: | :-- | --: | --: | :-- | :-- | :-- | --: | :-- |')
    for f in sorted(flips, key=lambda f: (f['geo'], f['eng'], f['var'], f['i'])):
        print(f'| {f["geo"]} | {f["model"]} | {f["K"]} | {f["var"]} | {f["i"]} | {f["route"]} | {f["xmax"]:.4f} | {f["x_final"]:.4f} | '
              f'{f["first_clear"] if f["first_clear"] is not None else "— (after the last step)"} / {f["steps"]} | {"yes" if (f["safe"] and not f["aborted"]) else "**no**"} | '
              f'{f["pace_x"] * 1000:+.1f} · {-f["pace_goal"] * 1000:+.1f} | {f["viol_past"][0]} | '
              f'{"yes → S&C" if f["vf"] else "no (" + str(f["n_viol"]) + " steps) → no S&C"} |')
    c1 = all((f['x_reach'] >= X_CLEAR) if not RECORDED_ONLY else (f['first_clear'] is not None) for f in flips)
    c2 = all(f['safe'] and not f['aborted'] for f in flips)
    c3 = [f for f in flips if not (f['pace_x'] > 0 and f['pace_goal'] < 0)]
    c4 = [f for f in flips if f['viol_past'][0] > 0]
    print(f'\n- (1) past x\' before the cap: {"all" if c1 else "NOT all"}; (2) safe, no abort: {"all" if c2 else "NOT all"}; '
          f'(3) moving toward the goal at the cap: {len(flips) - len(c3)}/{len(flips)}'
          + (f' (not: {[(f["geo"], f["model"], f["K"], f["var"], f["i"]) for f in c3]})' if c3 else '')
          + f'; (4) no violating step past x\': {len(flips) - len(c4)}/{len(flips)}'
          + (f' (not: {[(f["geo"], f["model"], f["K"], f["var"], f["i"], f["viol_past"][0]) for f in c4]})' if c4 else ''))

    # ── near-line flights (the author's question) ────────────────────────────────────────────────────────────────
    print('\n## 5 · Flights that end within 5 cm of x\' without the old success (the line decides them)\n')
    near = [f for f in F if not f['succ'] and (abs(f['xmax'] - X_CLEAR) < 0.05 or abs(f['x_reach'] - X_CLEAR) < 0.05)]
    for f in sorted(near, key=lambda f: f['xmax']):
        side = lambda v: 'past the line' if v >= X_CLEAR else 'short of the line'
        print(f'- {f["geo"]} {f["model"]} K{f["K"]} {f["var"]} trial {f["i"]} route {f["route"]}: furthest recorded x '
              f'(start of step {f["steps"]}) = **{f["xmax"]:.6f} m** → {side(f["xmax"])} by {abs(f["xmax"] - X_CLEAR) * 1000:.1f} mm; '
              f'after its {f["steps"]}th and last step x = **{f["x_final"]:.4f} m** → {side(f["x_final"])} by '
              f'{abs(f["x_final"] - X_CLEAR) * 1000:.1f} mm; pace at the cap {f["pace_x"] * 1000:+.2f} mm/step '
              f'(last recorded step {f["last_pace"] * 1000:+.2f} mm); safe {f["safe"]}; violation-free {f["vf"]}')

    # ── the two projected tables, baseline rows, and the frontier under the clear line ────────────────────────────
    def cnt(fs, L=X_CLEAR, old=False):
        return sum(f['sc'] for f in fs) if old else sum(rules(f, L)[1] for f in fs)

    ms = {}
    for key, fs in cell.items():
        c = G.load_cell(key[0], key[1], key[2], fs[0]['variant'])
        ms[key] = float(np.nanmean(c['ms']))
    print('\n## 6 · The baseline rows of the two projected tables, both rules\n')
    print('| block | table | rule | S&C old | S&C clear | success old → clear | violation-free | ms/step |')
    print('| :-- | :-- | :-- | --: | --: | :-- | --: | --: |')
    for geo in G.GEOS:
        for r in ('ps-r', 'ps-c', 'ps-t'):
            fs = cell[(geo, 'diffusion', 20, r)]
            print(f'| {geo} | projection | {r} | {cnt(fs, old=True)}/10 | **{cnt(fs)}/10** | {sum(f["succ"] for f in fs)} → '
                  f'{sum(rules(f)[0] for f in fs)} | {sum(f["vf"] for f in fs)}/10 | {ms[(geo, "diffusion", 20, r)]:.1f} |')

    def best(geo, eng, K, prefix, old):
        cand = [(k[3], cell[k]) for k in cell if k[0] == geo and k[1] == eng and k[2] == K and k[3].startswith(prefix)]
        if not cand:
            return None
        key = lambda rc: (-cnt(rc[1], old=old), np.mean([f['n_viol'] for f in rc[1]]), ms[(geo, eng, K, rc[0])])
        r, fs = sorted(cand, key=key)[0]
        return r, cnt(fs, old=old), ms[(geo, eng, K, r)]

    print('\n## 7 · `tab:uav-corridor` (best rule per projector) — every row whose entry changes under the clear line\n')
    print('| block | model | nfe | projector | old: S&C · rule · ms | clear: S&C · rule · ms |')
    print('| :-- | :-- | --: | :-- | :-- | :-- |')
    pts = {True: defaultdict(list), False: defaultdict(list)}
    for geo in G.GEOS:
        for lab, eng, Ks in G.MODELS:
            for K in Ks:
                for pj, pre in (('per-step', 'ps-'), ('endpoint', 'ep-')):
                    bo = best(geo, eng, K, pre, True); bn = best(geo, eng, K, pre, False)
                    if bo is None:
                        continue
                    pts[True][geo].append((f'{lab.replace(" a0.2", "")} K{K} {pj} {bo[0]}', bo[1], bo[2]))
                    pts[False][geo].append((f'{lab.replace(" a0.2", "")} K{K} {pj} {bn[0]}', bn[1], bn[2]))
                    if bo != bn:
                        print(f'| {geo} | {lab.replace(" a0.2", "")} | {K} | {pj} | {bo[1]}/10 · {bo[0]} · {bo[2]:.1f} | **{bn[1]}/10 · {bn[0]} · {bn[2]:.1f}** |')

    print('\n## 8 · Frontier (S&C of 10 against ms/step; S&C 0 not eligible; best rule per model × budget × projector)\n')
    for geo in G.GEOS:
        for old in (True, False):
            fr = G.frontier(pts[old][geo]); el = [p for p in pts[old][geo] if p[1] > 0]
            print(f'- **{geo}, {"old rule" if old else "clear line"}**: {len(el)} eligible of {len(pts[old][geo])}; frontier: '
                  + ('none' if not fr else '; '.join(f'{p[0]} ({p[1]}/10, {p[2]:.1f} ms)' for p in fr))
                  + ('' if not el else '  | eligible: ' + '; '.join(f'{p[0]} ({p[1]}/10, {p[2]:.1f} ms)' for p in sorted(el, key=lambda p: p[2]))))

    if a.json:
        json.dump(F, open(a.json, 'w'), indent=1, default=float)
        print(f'\n[json] {a.json}')


if __name__ == '__main__':
    main()
