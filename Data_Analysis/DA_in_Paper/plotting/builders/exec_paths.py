#!/usr/bin/env python3
"""Executed-path figures for the two manipulator benchmarks.

The counterpart, on D3IL-avoiding and D3IL-aligning, of what `paths.py` draws for the
quadrotor: the scene from above with its constraint set, and the path the end effector
actually took drawn on top of it. Both were `\\hole`s in Chapter 6 until the rollout
artefacts were staged (D10a, D10b); the paths come from `../../data/exec_paths.json`,
written by `extract/exec_paths.py`, so this module is standard library only.

The constraint set under a path is drawn by the SAME code as the environment figures of
Chapter 5 -- `avoiding.geometry_panel` and `scenes.aligning_constraint_panel` -- so it is
guaranteed to be the geometry the projection was given, not a redrawing of it.

Reading rule, the same as the quadrotor figures and carried by two channels so the pictures
survive grayscale:

    colour   green  no violating control step      red  at least one
    stroke   solid  reached the goal               dashed  did not

D3IL-aligning has no goal test -- an episode runs to its step limit and is scored by where
the box ends -- so every aligning path is solid and only colour carries.

WHAT THE ALIGNING FIGURE IS NOT. It draws the END EFFECTOR, which is what the evaluation
logs; the box pose is not in the observation at all. It is therefore not the full counterpart
of the quadrotor figure, where the drawn body is the constrained one. It is the right picture
all the same: on this task the constraint set applies to the planned end-effector position,
so this is exactly the quantity the projector acts on.
"""
import json
import os

import sources as S
from svg.fmpcc_svg import Fig, save_grid
from .avoiding import geometry_panel, _scene, FONT_ENV
from .scenes import aligning_constraint_panel

CLEAN = '#1e8449'
VIOLATING = '#c0392b'
START = '#2471a3'

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    '..', 'data', 'exec_paths.json')
FONT_PATHS = 1.55
ALIGNING_SHOWN = (3, 6)  # one raw violation and one raw violation-free path, both fully in view
AVOIDING_SHOWN = 1  # episode 2, the same initial condition in all four model-budget cells


def _load(key):
    path = os.path.normpath(DATA)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        D = json.load(f)
    return D['figures'].get(key)


def _draw(f, episodes, w=1.7):
    for ep in episodes:
        xy = ep['xy']
        if len(xy) < 2:
            continue
        colour = CLEAN if ep['clean'] else VIOLATING
        pts = [(f.X(x), f.Y(y)) for x, y in xy]
        f.poly(pts, colour, w=w, dash='' if ep.get('reached', True) else '6,4')
        f.marker(pts[-1][0], pts[-1][1], 'o', colour, filled=ep.get('reached', True), r=4.0, ew=1.3)


def _legend(width, font, items):
    h = Fig(width, 52, ml=0, mr=0, mt=0, mb=0, font=font)
    x = 18
    for kind, colour, dash, lab in items:
        if kind == 'line':
            h.poly([(x - 12, 24), (x + 12, 24)], colour, w=2.8, dash=dash)
        else:
            h.marker(x - 8, 24, 's', START, filled=True, r=4.0, ew=1.2)
            h.marker(x + 8, 24, 'o', '#5d6d7e', filled=True, r=4.0, ew=1.3)
        h.text(x + 24, 31, lab, 11, '#111')
        x += 62 + len(lab) * 11.0
    return h


# ═══════════════════════════════════════════════════════════════════════════
#  D3IL-avoiding: matched episode 2, one panel per model-budget cell
# ═══════════════════════════════════════════════════════════════════════════
def fig_avoiding_paths(outdir):
    D = _load('avoiding')
    sc = _scene()
    if not D or sc is None:
        return None
    panels = []
    for i, p in enumerate(D['panels']):
        ep = p['episodes'][AVOIDING_SHOWN]
        f = geometry_panel(sc, D['geometry'], 470, p['title'], '', i == 0, FONT_ENV)
        _draw(f, [ep], w=2.3)
        sx, sy = ep['xy'][0]
        ex, ey = ep['xy'][-1]
        start_x, start_y = f.X(sx), f.Y(sy)
        end_x, end_y = f.X(ex), f.Y(ey)
        f.marker(start_x, start_y, 's', START, filled=True, r=5.2, ew=1.2)
        f.marker(end_x, end_y, 'o', CLEAN, filled=True, r=5.2, ew=1.2)
        f.text(start_x + 10, start_y - 8, 'START', 8.5, '#1b3345', bold=True)
        f.text(end_x + 10, end_y + 19, 'END', 8.5, '#154e2b', bold=True)
        f.end_clip()
        panels.append(f)
    path = save_grid(panels, os.path.join(outdir, 'fig_avoiding_paths.svg'),
                     cols=len(panels), gap=8)
    eps = sum(p['n'] for p in D['panels'])
    return path, (f"data/exec_paths.json (extract/exec_paths.py) | {len(D['panels'])} cells, "
                  f"episode {AVOIDING_SHOWN + 1} of {D['panels'][0]['n']} drawn in each cell "
                  f"({eps} evaluated in total); geometry {D['geometry']}, variant {D['variant']}; "
                  f"constraint set from data/avoiding_scene.json, drawn by avoiding.geometry_panel")


# ═══════════════════════════════════════════════════════════════════════════
#  D3IL-aligning: one panel per projection method, the same ten contexts
# ═══════════════════════════════════════════════════════════════════════════
def fig_aligning_paths(outdir):
    D = _load('aligning')
    if not D:
        return None
    panels = []
    for i, p in enumerate(D['panels']):
        episodes = [p['episodes'][j] for j in ALIGNING_SHOWN]
        sub = f"{sum(ep['clean'] for ep in episodes)} of {len(episodes)} shown violation-free"
        f = aligning_constraint_panel(560, p['title'], sub, i == 0, FONT_PATHS)
        # Show only the box poses corresponding to the displayed paths.
        for ep in episodes:
            c = S.ALIGNING_CONTEXTS[ep['context']]
            f.marker(f.X(c['box'][0]), f.Y(c['box'][1]), 'o', '#8a7a55', filled=True, r=2.6, ew=0.9)
            f.marker(f.X(c['target'][0]), f.Y(c['target'][1]), 'o', '#8a7a55', filled=False, r=3.0, ew=1.1)
        _draw(f, episodes, w=2.1)
        sx, sy = episodes[0]['xy'][0]
        f.marker(f.X(sx), f.Y(sy), 's', START, filled=True, r=4.2, ew=1.2)
        f.end_clip()
        panels.append(f)
    width = sum(q.w for q in panels) + 8 * (len(panels) - 1)
    hdr = _legend(width, FONT_PATHS, [
        ('line', CLEAN, '', 'no violating control step'),
        ('line', VIOLATING, '', 'at least one'),
        ('mark', START, '', 'end effector at the start · where the context ended'),
    ])
    path = save_grid(panels, os.path.join(outdir, 'fig_aligning_paths.svg'),
                     cols=len(panels), gap=8, header=hdr)
    eps = len(ALIGNING_SHOWN) * len(D['panels'])
    return path, (f"data/exec_paths.json (extract/exec_paths.py) | {len(D['panels'])} cells, "
                  f"{eps} drawn paths, episode positions {ALIGNING_SHOWN}, context IDs "
                  f"{tuple(D['panels'][0]['episodes'][j]['context'] for j in ALIGNING_SHOWN)} "
                  f"from ten per cell; geometry {D['geometry']}; EXECUTED END-EFFECTOR xy position "
                  f"(the box pose is not logged); constraint set from sources.ALIGNING_CONSTRAINTS")


ALL = [
    ('fig_avoiding_paths', 'da', fig_avoiding_paths),
    ('fig_aligning_paths', 'da', fig_aligning_paths),
]
