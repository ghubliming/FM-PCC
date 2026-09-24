#!/usr/bin/env python3.14
"""Diagnostic figure for DA_20260923_corridor_v3_preliminary.md (not a thesis figure).
(a) tilt, FM K3: body clearance to the leaned plane over x — unprojected vs per-step projected (dpcc-t).
(b) hump: altitude over x — FM K1 per-step (stalls on the roof) vs FM K3 per-step (crosses), unprojected grey.
    python3.14 Data_Analysis/DA_in_Paper/analysis/DA_20260923_corridor_v3_preliminary/make_diag_fig.py
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import corridor_v3_grid as G

GREY, BLUE, PURPLE = '#9aa0a6', '#2a6fb0', '#7b3294'
ROOF, LIMIT, BOX = 'darkorange', 'crimson', 'steelblue'
fig, (a, b) = plt.subplots(1, 2, figsize=(12.5, 4.6), gridspec_kw=dict(wspace=0.22))
for ax in (a, b):
    ax.grid(True, ls=':', lw=0.6, color='0.85')
    for sp in ('top', 'right'): ax.spines[sp].set_visible(False)
    ax.tick_params(labelsize=8)

# (a) tilt clearance
cfg = G.geo_config('tilt')
hs = next(h for h in cfg['halfspace_constraints'] if h.get('z_lean'))
n3, P0 = G.NS['_hs_normal3'](hs); r = cfg['inflation']['r_drone']
for v, col, lw in ((G.UNPROJ, GREY, 0.8), (G.PER_STEP['t'], BLUE, 1.1)):
    c = G.load_cell('tilt', 'fm', 3, v)
    for o in c['obs']:
        P = o[:, 3:6]; x = P[:, 0]
        clr = (P - P0) @ n3 - r
        clr = np.where((x >= -2.0) & (x <= 2.0), clr, np.nan)
        a.plot(x, clr, color=col, lw=lw, alpha=0.8)
a.axhline(0, color=LIMIT, lw=1.2, ls='--')
a.axvline(2.0, color='0.3', lw=0.9, ls=':')
a.text(1.97, 0.55, 'window end\nx = 2.0', fontsize=7, ha='right', va='top', color='0.3')
a.text(-1.95, -0.03, 'below 0 = violating step (body inside the leaned plane)', fontsize=7, color=LIMIT, va='top')
a.annotate('projected: violating only in the last 0.25 m,\nsetpoint already past x = 2 (0.52–0.57 m ahead),\nsetpoint itself clean at every such step',
           xy=(1.9, -0.02), xytext=(-0.2, -0.33), fontsize=7, color=BLUE,
           arrowprops=dict(arrowstyle='->', color=BLUE, lw=0.8))
a.set_xlim(-2.1, 2.1); a.set_ylim(-0.5, 0.6)
a.set_xlabel('drone x (m)', fontsize=8); a.set_ylabel('clearance to the leaned plane − r_drone (m)', fontsize=8)
a.set_title('(a) tilt, FM $K=3$: body clearance along the corridor', fontsize=9, loc='left')
a.legend(handles=[Line2D([0], [0], color=GREY, lw=1, label='unprojected (10 flights)'),
                  Line2D([0], [0], color=BLUE, lw=1.2, label='per-step projected, t (10 flights)')], fontsize=7, loc='lower left')

# (b) hump altitude
xs = np.linspace(-1.5, 1.5, 121); H, s = 1.10, 1.10 / 1.5
raw = H - s * np.abs(xs); lim = raw + 0.31 * np.hypot(1, s)
b.fill_between(xs, 0.3, raw, color=ROOF, alpha=0.12, lw=0)
b.plot(xs, raw, color=ROOF, lw=1.8)
b.plot(xs, lim, color=LIMIT, lw=1.2, ls='--')
b.axhline(1.2992, color=BOX, lw=1.0, ls=':')
b.text(-2.75, 1.31, 'plan altitude box top 1.299 m (the normaliser clips plans here)', fontsize=7, color=BOX, va='bottom')
for v, K, col, lw in ((G.UNPROJ, 1, GREY, 0.8), (G.PER_STEP['t'], 3, BLUE, 1.0), (G.PER_STEP['t'], 1, PURPLE, 1.1)):
    c = G.load_cell('hump', 'fm', K, v)
    for o in c['obs']:
        b.plot(o[:, 3], o[:, 5], color=col, lw=lw, alpha=0.85)
        if col == PURPLE: b.plot(o[-1, 3], o[-1, 5], 'o', color=PURPLE, ms=3)
b.annotate('K = 1: stops on the limit before the apex,\nstep cap reached on 10/10', xy=(-0.12, 1.42), xytext=(-2.7, 1.62),
           fontsize=7, color=PURPLE, arrowprops=dict(arrowstyle='->', color=PURPLE, lw=0.8))
b.set_xlim(-2.9, 2.9); b.set_ylim(0.75, 1.75)
b.set_xlabel('drone x (m)', fontsize=8); b.set_ylabel('altitude z (m)', fontsize=8)
b.set_title('(b) hump, FM: per-step projection at $K=1$ vs $K=3$', fontsize=9, loc='left')
b.legend(handles=[Line2D([0], [0], color=GREY, lw=1, label='unprojected, K=1'),
                  Line2D([0], [0], color=BLUE, lw=1.1, label='per-step t, K=3 (crosses 10/10)'),
                  Line2D([0], [0], color=PURPLE, lw=1.1, label='per-step t, K=1 (crosses 0/10)'),
                  Line2D([0], [0], color=ROOF, lw=1.8, label='roof (raw)'),
                  Line2D([0], [0], color=LIMIT, lw=1.2, ls='--', label='drone-centre limit')], fontsize=7, loc='lower right')
out = os.path.join(HERE, 'fig_diag_tilt_residue_hump_stall.png')
fig.savefig(out, dpi=150, bbox_inches='tight'); print(out)
