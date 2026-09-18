#!/usr/bin/env python3
"""Flown-path figures: what the vehicle actually did, over the constraint set it flew in.

The three result figures of v3 Chapter 6 that were `\\hole`s until the rollout artefacts
were downloaded. Each is the SAME picture as `scenes.fig_constraints_uav` -- the scene from
above with its inflated and tightened obstacles -- with the executed paths drawn on top and
a mark where each flight ended. Reusing `scenes._uav_constraint_panel` is deliberate: the
constraint set under a path is then guaranteed to be the one the projection saw, because it
is drawn by the same code from the same `sources.UAV_CONSTRAINTS`.

The paths come from `../../data/uav_paths.json`, written by `extract/uav_paths.py` (numpy);
this module, like every builder, is standard library only.

How a flight is drawn -- two independent outcomes, never collapsed into one:

    colour   collision-free (green) or entered an inflated obstacle (red)
    stroke   solid = passed the goal, dashed = did not
    end mark filled = passed the goal, hollow = did not

They are independent on purpose: a flight can pass the goal THROUGH a pillar, and telling
that apart from a clean arrival is the entire point of the projection comparison. Colour is
never the only carrier -- dash and marker fill repeat it, so the figures survive grayscale.
"""
import json
import os

import sources as S
from svg.fmpcc_svg import Fig, save_grid
from .scenes import _uav_constraint_panel, FONT_CONSTRAINT

CLEAN = '#1e8449'        # never entered an inflated obstacle
VIOLATING = '#c0392b'    # entered one
START = '#2471a3'

# Wider than the 470 of fig_constraints_uav: these panels carry an outcome count in the
# subtitle, which is clipped at 470 once a projection variant is named alongside it.
PANEL_W = 560


def _load():
    if not os.path.isfile(S.UAV_PATHS):
        return None
    with open(S.UAV_PATHS) as f:
        return json.load(f)


def _scene(name):
    for scn in S.UAV_CONSTRAINTS['scenes']:
        if scn['name'] == name:
            return scn
    return None


def _panel(scn, panel, first, r_drone, tight):
    """One constraint panel with its flights drawn over it."""
    eps = panel['episodes']
    n, passed, clean = panel['n'], panel['passed'], panel['clean']
    # The panel's own title and counts replace the scene's, so the reader never has to
    # count paths to learn the outcome. The scene dict is copied, not mutated: it is
    # shared with fig_constraints_uav.
    # Kept short on purpose: at FONT_CONSTRAINT a longer subtitle runs off the panel.
    local = dict(scn, title=panel['title'],
                 sub=f"{panel['sub']} · {passed}/{n} passed · {clean}/{n} clean")
    f = _uav_constraint_panel(local, PANEL_W, first, r_drone, tight)

    f.clip_to_box()
    for ep in eps:
        xy = ep['xy']
        if len(xy) < 2:
            continue
        colour = CLEAN if ep['clean'] else VIOLATING
        pts = [(f.X(x), f.Y(y)) for x, y in xy]
        f.poly(pts, colour, w=1.9, dash='' if ep['passed'] else '6,4')
        f.marker(pts[-1][0], pts[-1][1], 'o', colour, filled=ep['passed'], r=4.2, ew=1.3)
    # one start mark for the whole panel: every flight of a cell launches from the same pose
    sx, sy = eps[0]['xy'][0] if eps else (None, None)
    if sx is not None:
        f.marker(f.X(sx), f.Y(sy), 's', START, filled=True, r=4.0, ew=1.2)
    f.end_clip()
    return f


def _legend(width):
    h = Fig(width, 104, ml=0, mr=0, mt=0, mb=0, font=FONT_CONSTRAINT)
    items = [('line', CLEAN, '', 'collision-free flight'),
             ('line', VIOLATING, '', 'entered an obstacle'),
             ('line', '#5d6d7e', '6,4', 'did not pass the goal'),
             ('mark', START, '', 'launch pose · flight end')]
    row_y, col_x, size = (30, 76), (16, width // 2 + 16), 18
    for i, (kind, colour, dash, lab) in enumerate(items):
        x, y = col_x[i % 2], row_y[i // 2]
        if kind == 'line':
            h.poly([(x - 14, y), (x + 14, y)], colour, w=3.0, dash=dash)
        else:
            h.marker(x - 8, y, 's', colour, filled=True, r=4.0, ew=1.2)
            h.marker(x + 8, y, 'o', '#5d6d7e', filled=True, r=4.2, ew=1.3)
        h.text(x + 26, y + 7, lab, size, '#111')
    return h


def _build(key, filename, cols, outdir):
    D = _load()
    if not D or key not in D['figures']:
        return None
    fig = D['figures'][key]
    panels = [p for p in fig['panels'] if p['n']]
    if not panels:
        return None
    scn = _scene(fig['scene'])
    if not scn:
        return None
    C = S.UAV_CONSTRAINTS
    r, t = C['r_drone'], C['tightening']

    drawn = [_panel(scn, p, i % cols == 0, r, t) for i, p in enumerate(panels)]
    width = max(sum(d.w for d in drawn[i:i + cols]) + 8 * (min(cols, len(drawn) - i) - 1)
                for i in range(0, len(drawn), cols))
    path = save_grid(drawn, os.path.join(outdir, filename), cols=cols, gap=8,
                     header=_legend(width))
    flights = sum(p['n'] for p in panels)
    srcs = ', '.join(sorted({p['tag'] for p in panels}))
    return path, (f"data/uav_paths.json (extract/uav_paths.py) | {len(panels)} cells, "
                  f"{flights} flights; runs {srcs}; scene and constraints from "
                  f"sources.UAV_CONSTRAINTS, drawn by scenes._uav_constraint_panel")


def fig_uav_scurve_paths(outdir):
    """The s-curve flown under both controllers, unprojected (companion of tab:uav-controller)."""
    return _build('scurve', 'fig_uav_scurve_paths.svg', 2, outdir)


def fig_uav_pillars_paths(outdir):
    """The pillars flown by each model under its best projection (tab:uav-pillars-best)."""
    return _build('pillars', 'fig_uav_pillars_paths.svg', 2, outdir)


def fig_uav_corridor_paths(outdir):
    """The slide flown across the budget ladder; the baseline is the last row, at its one budget."""
    return _build('corridor', 'fig_uav_corridor_paths.svg', 3, outdir)


ALL = [
    ('fig_uav_scurve_paths', 'da', fig_uav_scurve_paths),
    ('fig_uav_pillars_paths', 'da', fig_uav_pillars_paths),
    ('fig_uav_corridor_paths', 'da', fig_uav_corridor_paths),
]
