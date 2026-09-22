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
