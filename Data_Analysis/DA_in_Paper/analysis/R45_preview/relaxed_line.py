#!/usr/bin/env python3.14
# R45 PREVIEW (v3.81 writing agent, 2026-09-24) -- not the DA of record; the DA agent redoes R45a properly.
# Spec: logs_in_develop/Writing/Working_Space/data_status/PENDING_20260924_uav_corridor_step_cap_R45.md
#   python3.14 Data_Analysis/DA_in_Paper/analysis/R45_preview/relaxed_line.py
"""Preview only (the DA agent redoes it properly): S&C if the corridor finish line is moved to x' = end of the
constrained section. Original success kept; a flight also succeeds if its centre reaches x >= x' within the 396 steps,
in controlled flight (every corridor flight is: no abort, no unsafe flight). Violations counted over the flight as flown."""
import sys, numpy as np
sys.path.insert(0, '/workspaces/FM-PCC/Data_Analysis/DA_in_Paper/analysis')
import corridor_v3_grid as G
LINES = (2.00, 2.31, 2.36, 2.50)  # 2.00 = the end of the corridor (author's choice, R45a); 2.31/2.36 = + rotor reach (+ cap); 2.50 = where aligned flights enter the 0.30 m goal sphere
changed = []
for geo in ('tilt', 'hump'):
    for lab, eng, Ks in G.MODELS:
        for K in Ks:
            vs = [('none', G.UNPROJ)] + [('ps-' + k, v) for k, v in G.PER_STEP.items()] + \
                 ([('ep-' + k, v) for k, v in G.ENDPOINT.items()] if (K >= 3 and eng != 'diffusion') else [])
            for vn, v in vs:
                c = G.load_cell(geo, eng, K, v)
                succ = c['succ'].astype(bool); vf = c['cf'].astype(bool); sc = c['sc'].astype(bool)
                xmax = np.array([o[:, 3].max() for o in c['obs']])
                row = [int(sc.sum())]
                for L in LINES:
                    s2 = succ | (xmax >= L)
                    row.append(int((s2 & vf).sum()))
                gain = [int(((succ | (xmax >= L)) & ~succ).sum()) for L in LINES]
                if any(r != row[0] for r in row[1:]) or any(gain):
                    changed.append((geo, lab.split()[0], K, vn, row, gain, np.round(np.sort(xmax[~succ]), 2)))
print('cells whose success or S&C changes under a moved line  (S&C now -> at x\'=2.00 / 2.31 / 2.36 / 2.50; successes gained)')
for geo, m, K, vn, row, gain, xs in changed:
    print(f'  {geo:4} {m:10} K{K:<2} {vn:5}  S&C {row[0]}/10 -> ' + ' / '.join(str(r) for r in row[1:]) + f'   +succ {gain}   max-x of the unsuccessful: {[float(x) for x in xs]}')
