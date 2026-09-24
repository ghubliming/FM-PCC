#!/usr/bin/env python3
"""UAV-corridor from the side: the flown altitude of the same flights fig_uav_corridor_paths
draws from above, unprojected and projected, one panel per model.

Why it exists (v3.65, author question of 2026-09-22): every constraint of the corridor --
the two walls, the sloping halfspace, the wall-end caps -- acts in the horizontal plane;
altitude is bounded only by the workspace box, whose limits after inflation lie 0.3-0.5 m
from any flown altitude. The author asked whether the corridor "also slides in z" and
whether the projected output moves in z. This panel is the answer drawn: grey is the
flight without projection, blue the same launch under per-step projection; the dashed
lines are the only altitude limits the projector ever sees. Anything the blue does that
the grey does not is a by-product of the lateral correction, not a constraint response.

Data: `../../data/uav_paths.json`, group `corridor_altitude` (extract/uav_paths.py, F4),
which carries `z` per kept step since v3.65. Standard library only, like every builder.
"""
import os

import sources as S
from svg.fmpcc_svg import Fig, save_grid
from .scenes import FONT_CONSTRAINT
from .paths import _load, _failed, _xmark

RAW = '#8c8c8c'          # unprojected flight
PROJ = '#2471a3'         # per-step projection, tightened (same blue as the launch mark elsewhere)
LIMIT = '#5d6d7e'        # workspace-box altitude limits after inflation
WALLS = '#f2f2f2'        # the x span of the walls and the slide

PANEL_W = 560
PANEL_H = 300
XLIM = (-2.8, 2.8)
ZLIM = (0.45, 1.65)
# config/uav_projection.yaml :: corridor_v2_slide.workspace_bounds z in [0.30, 1.80],
# inflated by planning_inflation.r_drone = 0.31 before it reaches the solver.
Z_BOX = (0.30, 1.80)
R_DRONE = S.UAV_CONSTRAINTS['r_drone']
X_WALLS = (-2.0, 2.0)    # x_active of the walls and the slide


def _panel(title, sub, raw, proj, first):
    fs = FONT_CONSTRAINT
    ml, mr, mt, mb = int(42 * fs), int(10 * fs), int(46 * fs), int(46 * fs)
    f = Fig(PANEL_W, PANEL_H, ml=ml, mr=mr, mt=mt, mb=mb, font=fs)
    f.axes(XLIM, ZLIM)
    f.frame([-2, -1, 0, 1, 2], [0.6, 0.8, 1.0, 1.2, 1.4, 1.6], 'x [m]', 'z [m]' if first else '',
            title, sub, xfmt=lambda v: f'{v:g}', yfmt=lambda v: f'{v:.1f}')
    f.clip_to_box()
    f.vspan(f.X(X_WALLS[0]), f.X(X_WALLS[1]), WALLS)
    lo, hi = Z_BOX[0] + R_DRONE, Z_BOX[1] - R_DRONE
    for z in (lo, hi):
        f.dline([(XLIM[0], z), (XLIM[1], z)], LIMIT, w=1.6, dash='7,5')
    f.text(f.X(XLIM[0]) + 6, f.Y(hi) - 5, f'box limit after inflation, z = {hi:.2f} m', 9.0, LIMIT)
    f.text(f.X(XLIM[0]) + 6, f.Y(lo) + 12, f'box limit after inflation, z = {lo:.2f} m', 9.0, LIMIT)
    for eps, colour, w in ((raw, RAW, 1.3), (proj, PROJ, 1.7)):
        for ep in eps:
            pts = [(f.X(x), f.Y(z)) for (x, _y), z in zip(ep['xy'], ep['z'])]
            if len(pts) < 2:
                continue
            f.poly(pts, colour, w=w)
            if _failed(ep):
                _xmark(f, pts[-1][0], pts[-1][1], colour)
            else:
                f.marker(pts[-1][0], pts[-1][1], 'o', colour, filled=ep['passed'], r=3.6, ew=1.2)
    f.end_clip()
    return f


def _legend(width):
    h = Fig(width, 104, ml=0, mr=0, mt=0, mb=0, font=FONT_CONSTRAINT)
    items = [(RAW, '', 'unprojected plan'),
             (PROJ, '', 'per-step projection, tightened'),
             (LIMIT, '7,5', 'altitude limits the projector sees: workspace box after inflation')]
    row_y, col_x = (30, 76), (16, width // 2 + 16)
    for i, (colour, dash, lab) in enumerate(items):
        x, y = col_x[i % 2], row_y[i // 2]
        h.poly([(x - 14, y), (x + 14, y)], colour, w=3.0, dash=dash)
        h.text(x + 26, y + 7, lab, 18, '#111')
    return h


def fig_uav_corridor_altitude(outdir):
    """Altitude of every corridor flight, unprojected against projected, per model."""
    D = _load()
    if not D or 'corridor_altitude' not in D['figures']:
        return None
    panels = [p for p in D['figures']['corridor_altitude']['panels'] if p['n']]
    if not panels or any('z' not in e for p in panels for e in p['episodes']):
        return None
    pairs = {}
    for p in panels:
        pairs.setdefault(p['title'], {})['raw' if p['variant'] == 'diffuser' else 'proj'] = p
    drawn = []
    for i, (title, pr) in enumerate(pairs.items()):
        if 'raw' not in pr or 'proj' not in pr:
            continue
        k = pr['proj']['sub'].split(' ')[0]
        eps_r, eps_p = pr['raw']['episodes'], pr['proj']['episodes']
        z_r = [z for e in eps_r for z in e['z']]
        z_p = [z for e in eps_p for z in e['z']]
        sub = (f'{k} evaluations · grey z {min(z_r):.2f}–{max(z_r):.2f} m · '
               f'blue z {min(z_p):.2f}–{max(z_p):.2f} m')
        drawn.append(_panel(title, sub, eps_r, eps_p, i % 2 == 0))
    if not drawn:
        return None
    width = 2 * PANEL_W + 8
    path = save_grid(drawn, os.path.join(outdir, 'fig_uav_corridor_altitude.svg'), cols=2, gap=8,
                     header=_legend(width))
    srcs = ', '.join(sorted({p['tag'] for p in panels}))
    return path, (f"data/uav_paths.json (extract/uav_paths.py, group corridor_altitude) | "
                  f"{len(drawn)} models x 2 arms, 12 flights each; runs {srcs}; altitude limits from "
                  f"config/uav_projection.yaml corridor_v2_slide.workspace_bounds z and "
                  f"sources.UAV_CONSTRAINTS r_drone")


ALL = [
    ('fig_uav_corridor_altitude', 'da', fig_uav_corridor_altitude),
]


# ---------------------------------------------------------------------------------------------
# v3.80: UAV-corridor v3 (R33) -- the two constraints read along the corridor, one row each.
#
# The v2 figure above drew the level corridor's altitude (every constraint horizontal). On the
# corridor v3 both constraints act on altitude, and what the chapter reads off the flights is the
# two structural results of analysis/DA_20260924_corridor_v3.md §4: on the tilt, a residue of
# violating steps at the end of the leaned plane; on the hump, the stop before the apex at K = 1.
# Row 1 = tilt: the margin of the flown position to the leaned plane (less the rotor reach), the
# quantity the violation test thresholds, over the plane's x span; right = the plane's last 0.65 m.
# Row 2 = hump: altitude over x with the roof and its violation boundary; right = over the apex.
# Data: data/corridor_v3_side.json (extract/corridor_v3_side.py). Standard library only.
# ---------------------------------------------------------------------------------------------
import json                                                            # noqa: E402

SIDE_COL = {'raw': RAW, 'ps1': '#7fb3d5', 'ps20': '#1b4f72'}
SIDE_W = {'raw': 1.2, 'ps1': 1.6, 'ps20': 1.8}
BOUND = '#c0392b'          # the violation boundary: constraint plus rotor reach
ROOF_FILL = '#f6ddcc'      # the roof itself
ROOF_EDGE = '#ba4a00'
RESIDUE = '#fbeee6'        # the last 0.3 m of the leaned plane
PLAN_TOP = '#5d6d7e'
SIDE_PW, SIDE_PH = 560, 320


def _side_frame(title, sub, xlim, ylim, xt, yt, ylab, xfmt, yfmt):
    fs = FONT_CONSTRAINT
    f = Fig(SIDE_PW, SIDE_PH, ml=int(50 * fs), mr=int(10 * fs), mt=int(46 * fs), mb=int(46 * fs), font=fs)
    f.axes(xlim, ylim)
    f.frame(xt, yt, 'x [m]', ylab, title, sub, xfmt=xfmt, yfmt=yfmt)
    f.clip_to_box()
    return f


def _series(f, block, stop_marks=False):
    for s in block['series']:
        for ep, ok in zip(s['flights'], s['success_flags']):
            pts = [(f.X(x), f.Y(y)) for x, y in ep]
            if len(pts) < 2:
                continue
            f.poly(pts, SIDE_COL[s['key']], w=SIDE_W[s['key']])
            if stop_marks and not ok:
                f.marker(pts[-1][0], pts[-1][1], 'o', SIDE_COL[s['key']], filled=False, r=4.2, ew=1.6)


def _tilt_panel(T, zoom):
    lo, hi = T['x_active']
    if zoom:
        f = _side_frame('(b) tilt, the end of the plane', 'the last 0.65 m of its span',
                        (1.35, 2.05), (-0.14, 0.10), [1.4, 1.6, 1.8, 2.0], [-0.1, -0.05, 0.0, 0.05, 0.1],
                        '', lambda v: f'{v:g}', lambda v: f'{v:g}')
    else:
        f = _side_frame('(a) tilt, margin to the leaned plane', 'below 0: a violating control step',
                        (lo - 0.1, hi + 0.1), (-0.42, 0.45), [-2, -1, 0, 1, 2], [-0.4, -0.2, 0.0, 0.2, 0.4],
                        'margin [m]', lambda v: f'{v:g}', lambda v: f'{v:.1f}')
    f.vspan(f.X(1.70), f.X(hi), RESIDUE)
    f.dline([(lo - 0.2, 0.0), (hi + 0.2, 0.0)], BOUND, w=1.6, dash='7,5')
    f.dline([(hi, f.y0), (hi, f.y1)], PLAN_TOP, w=1.2, dash='2,4')
    _series(f, T)
    f.end_clip()
    if not zoom:
        f.text(f.X(hi) - 6, f.T + 16 * FONT_CONSTRAINT, 'plane ends', 9.0, PLAN_TOP, anchor='end')
    return f


def _hump_panel(H, zoom):
    (x0, z0), (xa, za), (x1, z1) = H['roof']
    off = H['limit_offset']
    xs = [x0 + i * (x1 - x0) / 300 for i in range(301)]
    roof = [(x, za - H['slope'] * abs(x - xa)) for x in xs]
    if zoom:
        f = _side_frame('(d) hump, over the apex', 'K = 1, projected: stopped before it',
                        (-0.6, 0.6), (1.28, 1.56), [-0.4, -0.2, 0.0, 0.2, 0.4], [1.3, 1.35, 1.4, 1.45, 1.5, 1.55],
                        '', lambda v: f'{v:g}', lambda v: f'{v:.2f}')
    else:
        f = _side_frame('(c) hump, altitude along the corridor', 'the vehicle must climb over the roof',
                        (-2.9, 3.0), (0.78, 1.70), [-2, -1, 0, 1, 2], [0.8, 1.0, 1.2, 1.4, 1.6],
                        'z [m]', lambda v: f'{v:g}', lambda v: f'{v:.1f}')
        f.polygon([(x0, f.y0 - 1)] + roof + [(x1, f.y0 - 1)], ROOF_FILL, stroke=ROOF_EDGE, w=1.6)
    # three vertices, not the sampled roof: the PNG preview restarts a dash on every short segment
    f.dline([(x0, z0 + off), (xa, za + off), (x1, z1 + off)], BOUND, w=1.6, dash='7,5')
    f.dline([(f.x0, H['plan_z_top']), (f.x1, H['plan_z_top'])], PLAN_TOP, w=1.4, dash='2,4')
    _series(f, H, stop_marks=True)
    f.end_clip()
    if not zoom:
        f.text(f.X(0.0), f.Y(za) + 26 * FONT_CONSTRAINT, 'roof', 9.0, ROOF_EDGE, anchor='middle')
    return f


def _side_legend(width, T):
    h = Fig(width, 150, ml=0, mr=0, mt=0, mb=0, font=FONT_CONSTRAINT)
    lab = {s['key']: s['label'] for s in T['series']}
    items = [('line', SIDE_COL['raw'], '', lab['raw']),
             ('line', BOUND, '7,5', 'violation boundary'),
             ('line', SIDE_COL['ps1'], '', lab['ps1']),
             ('line', PLAN_TOP, '2,4', 'highest planned altitude (c, d)'),
             ('line', SIDE_COL['ps20'], '', lab['ps20']),
             ('band', RESIDUE, '', 'last 0.3 m of the plane (a, b)'),
             ('stop', SIDE_COL['ps1'], '', 'flight ended short of the finish line')]
    col_x, row_y = (16, width // 2 + 16), (24, 58, 92, 126)
    for i, (kind, colour, dash, text) in enumerate(items):
        x, y = col_x[i % 2], row_y[i // 2]
        if kind == 'line':
            h.poly([(x - 14, y), (x + 14, y)], colour, w=3.2, dash=dash)
        elif kind == 'band':
            h.s.append(f'<rect x="{x - 14}" y="{y - 9}" width="28" height="18" fill="{colour}" '
                       f'stroke="#bbb" stroke-width="0.8"/>')
        else:
            h.marker(x, y, 'o', colour, filled=False, r=6.5, ew=2.2)
        h.text(x + 26, y + 7, text, 16, '#111')
    return h


def fig_uav_corridor_side(outdir):
    """UAV-corridor v3: the tilt margin and the hump altitude along the corridor, FM, K = 1 and 20."""
    if not os.path.isfile(S.CORRIDOR_V3_SIDE):
        return None
    with open(S.CORRIDOR_V3_SIDE) as fh:
        D = json.load(fh)
    T, H = D['tilt'], D['hump']
    panels = [_tilt_panel(T, False), _tilt_panel(T, True), _hump_panel(H, False), _hump_panel(H, True)]
    width = 2 * SIDE_PW + 8
    path = save_grid(panels, os.path.join(outdir, 'fig_uav_corridor_side.svg'), cols=2, gap=8,
                     header=_side_legend(width, T))
    n = {s['key']: s['n'] for s in T['series'] + H['series']}
    m = D['meta']
    return path, (f"data/corridor_v3_side.json (extract/corridor_v3_side.py, {m['written']}) | FM, "
                  f"{'/'.join(str(v) for v in sorted(set(n.values())))} flights per series, first "
                  f"{m['n_read']} of each cell; tags {m['tags']['tilt']} (tilt), {m['tags']['hump']} (hump); "
                  f"per-step rule c = the best per-step rule at K 1 and 20 ({m['analysis']} §3.2); geometry "
                  f"config/uav_projection.yaml corridor_v3_tilt / corridor_v3_ablation_hump, r_drone "
                  f"{T['r_drone']}; planned-altitude top = max planned z of the drawn cells "
                  f"({H['plan_z_top']} m)")


ALL = ALL + [('fig_uav_corridor_side', 'da', fig_uav_corridor_side)]
