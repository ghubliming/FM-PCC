#!/usr/bin/env python3
"""What the expert data does against the constraint set, for the two environments that
had no such figure.

D3IL-avoiding has had this since v3.9: `fig:constraints-avoiding` draws the 96
demonstrations over the three geometries and `tab:avoiding-geometries` counts how many
satisfy each one (0, 1, 2). It is the figure that makes the benchmark's premise visible --
the demonstrations cross the constraints, so satisfying them can only be the projection's
doing. These two build the same statement for the alignment task and the quadrotor scenes.

Both read `../../data/expert_paths.json` (written by `extract/expert_paths.py`), so this
module stays standard library only, like every builder here.

What is drawn is stated exactly, because neither is a flown demonstration:

* quadrotor -- the demonstration generator's own reference path, one per homotopy class
  the collection cycles through, against the inflated constraint set the projection sees;
* alignment -- the recorded contexts, drawn as the direct push from the box centre to the
  target centre, which is the shortest path a demonstration could take.
"""
import json
import math
import os

import sources as S
from svg.fmpcc_svg import Fig, save_grid
from .scenes import _uav_constraint_panel, FONT_CONSTRAINT

CLEAN = '#1e8449'
VIOLATING = '#c0392b'
START = '#2471a3'
PANEL_W = 560

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    '..', 'data', 'expert_paths.json')


def _load():
    path = os.path.normpath(DATA)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        return json.load(f)


def _scene(name):
    for scn in S.UAV_CONSTRAINTS['scenes']:
        if scn['name'] == name:
            return scn
    return None


def _vehicle(f, centre, r_drone, colour):
    """The X2 at `centre`: its real rotor footprint, inside the disk of radius r_drone the
    projection inflates every obstacle by. Geometry from sources.X2_GEOMETRY, the same
    numbers `fig:x2-dimensions` is drawn from."""
    cx, cy = centre
    g = getattr(S, 'X2_GEOMETRY', None)
    f.circle(cx, cy, r_drone, fill=colour, opacity=0.16, stroke=colour, w=1.4, dash='5,4')
    if not g:
        return
    for dx, dy in g['rotor_xy']:
        f.circle(cx + dx, cy + dy, g['rotor_radius'], fill=colour, opacity=0.55,
                 stroke='#111', w=0.8)
    ax_, ay_ = g['body_semi_axes']
    f.polygon([(cx - ax_, cy - ay_), (cx + ax_, cy - ay_),
               (cx + ax_, cy + ay_), (cx - ax_, cy + ay_)], colour, opacity=0.85,
              stroke='#111', w=0.8)


def _legend(width, items, cols=2):
    rows = (len(items) + cols - 1) // cols
    h = Fig(width, 20 + 40 * rows, ml=0, mr=0, mt=0, mb=0, font=FONT_CONSTRAINT)
    col_x = tuple(16 + (width // cols) * c for c in range(cols))
    for i, (colour, dash, lab) in enumerate(items):
        x, y = col_x[i % cols], 26 + 40 * (i // cols)
        if colour is None:                       # the vehicle key: a disk, not a line
            # the legend Fig has no data axes, so this one is written in pixels
            h.s.append(f'<circle cx="{x}" cy="{y}" r="14" fill="#777" fill-opacity="0.16" '
                       f'stroke="#777" stroke-width="1.4" stroke-dasharray="5,4"/>')
            h.marker(x, y, 's', '#777', filled=True, r=4.5, ew=0.8)
        else:
            h.poly([(x - 14, y), (x + 14, y)], colour, w=3.0, dash=dash)
        h.text(x + 26, y + 7, lab, 18, '#111')
    return h


# ═══════════════════════════════════════════════════════════════════════════
#  Quadrotor: the reference path the demonstrations track, per homotopy class
# ═══════════════════════════════════════════════════════════════════════════
def fig_expert_uav(outdir):
    """The demonstration reference path of each scene against its constraint set."""
    D = _load()
    if not D:
        return None
    C = S.UAV_CONSTRAINTS
    r, t = C['r_drone'], C['tightening']

    drawn = []
    for i, block in enumerate(D['uav']):
        scn = _scene(block['scene'])
        if not scn:
            continue
        n, clean = block['n_routes'], block['n_clean']
        local = dict(scn, title=block['scene'],
                     sub=f"{n} demonstrated route{'s' if n != 1 else ''} · {clean}/{n} clean")
        f = _uav_constraint_panel(local, PANEL_W, i == 0, r, t)
        f.clip_to_box()
        for route in block['routes']:
            colour = CLEAN if route['clean'] else VIOLATING
            pts = [(f.X(x), f.Y(y)) for x, y in route['xy']]
            f.poly(pts, colour, w=2.1)
            f.marker(pts[0][0], pts[0][1], 's', START, filled=True, r=4.0, ew=1.2)
            f.marker(pts[-1][0], pts[-1][1], 'o', colour, filled=True, r=4.2, ew=1.3)
        # The vehicle drawn to scale at the tightest moment of each route. A path is a line
        # and an obstacle is a line, so a line that stays outside another line looks safe;
        # the quadrotor is 0.62 m across and collides with its body, which is the whole
        # reason the constraint set is inflated at all. Drawn last so it sits over the path.
        #
        # De-cluttered: on UAV-corridor all three lanes are tightest at the same x, and three
        # bodies drawn there are one blob. Each vehicle therefore goes to the tightest moment
        # of its own route that is clear of the vehicles already placed (by 3.4 vehicle radii) -- still a real moment
        # of that flight, and on the corridor it spreads them along the stretch where every
        # lane is inside the slide.
        placed = []
        for route in block['routes']:
            spot, overlap = route['worst_xy'], route['worst_overlap']
            for cand in route.get('tight_order') or []:
                cx, cy, ov = cand
                # a violating route must be shown at a moment where it violates, or the
                # vehicle would be drawn green on a red path
                if not route['clean'] and ov <= 0:
                    continue
                if all(math.hypot(cx - px, cy - py) >= 3.4 * r for px, py in placed):
                    spot, overlap = [cx, cy], ov
                    break
            placed.append(spot)
            _vehicle(f, spot, r, VIOLATING if overlap > 0 else CLEAN)
        f.end_clip()
        drawn.append(f)
    if not drawn:
        return None

    width = sum(d.w for d in drawn[:2]) + 8
    legend = _legend(width, [
        (CLEAN, '', 'reference path satisfies the constraints'),
        (VIOLATING, '', 'reference path crosses them'),
        (None, '', f"the vehicle to scale at each route's tightest moment ({r:g} m disk)"),
    ], cols=1)
    path = save_grid(drawn, os.path.join(outdir, 'fig_expert_uav.svg'), cols=2, gap=8,
                     header=legend)
    return path, ('data/expert_paths.json (extract/expert_paths.py) | '
                  'uav_expert_data_collect/trajectories.py reference paths, one per homotopy '
                  'class of generator.py::HOMOTOPY_CLASSES; constraints from '
                  'sources.UAV_CONSTRAINTS, drawn by scenes._uav_constraint_panel')


# ═══════════════════════════════════════════════════════════════════════════
#  Alignment: the recorded contexts against the constraint set
# ═══════════════════════════════════════════════════════════════════════════
def fig_expert_aligning(outdir):
    """Every recorded context, drawn as the direct push, over the alignment constraints."""
    from svg.fmpcc_svg import clip_halfplane
    D = _load()
    C = getattr(S, 'ALIGNING_CONSTRAINTS', None)
    if not D or not C or not D.get('aligning'):
        return None
    fs = 1.35
    t = C['tightening']
    (x0, x1), (y0, y1) = C['extent']['x'], C['extent']['y']
    ml, mr, mt, mb = int(46 * fs), int(14 * fs), int(50 * fs), int(46 * fs)
    w = 620
    pw = w - ml - mr
    ph = pw * (y1 - y0) / (x1 - x0)
    f = Fig(w, int(round(ph + mt + mb)), ml=ml, mr=mr, mt=mt, mb=mb, font=fs)
    f.axes((x0, x1), (y0, y1))

    n = sum(b['n'] for b in D['aligning'])
    hits = sum(b['n_push_hits'] for b in D['aligning'])
    f.frame([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [-0.4, -0.2, 0.0, 0.2, 0.4],
            'x [m]', 'y [m]', 'Vision-conditioned alignment',
            f'{n} recorded contexts · {hits} direct pushes cross the keep-out region',
            xfmt=lambda v: f'{v:g}', yfmt=lambda v: f'{v:g}')
    f.clip_to_box()

    hs = C['halfspace']
    (ax_, ay), (bx, by) = hs['p0'], hs['p1']
    m = (by - ay) / (bx - ax_)
    b = ay - m * ax_
    poly = clip_halfplane([(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                          lambda X, Y: Y - (m * X + b))
    if len(poly) > 2:
        f.polygon(poly, '#5d6d7e', opacity=0.32)
    shift = -t * (1 + m * m) ** 0.5
    xs = (x0 - 0.05, x1 + 0.05)
    f.dline([(x, m * x + b) for x in xs], '#34495e', w=2.4)
    f.dline([(x, m * x + b + shift) for x in xs], '#34495e', w=1.6, dash='8,5')

    dk = C['disk']
    cx, cy = dk['c']
    f.circle(cx, cy, dk['r'], fill='#5d6d7e', opacity=0.32, stroke='#34495e', w=1.8)
    f.circle(cx, cy, dk['r'] + t, stroke='#34495e', w=1.6, dash='8,5')

    for block in D['aligning']:
        for ctx in block['contexts']:
            colour = VIOLATING if ctx['push_hits_disk'] else CLEAN
            f.dline([tuple(ctx['box']), tuple(ctx['target'])], colour, w=1.2, opacity=0.75)
            f.marker(f.X(ctx['box'][0]), f.Y(ctx['box'][1]), 's', colour, filled=True,
                     r=2.8, ew=0.9)
            f.marker(f.X(ctx['target'][0]), f.Y(ctx['target'][1]), 'o', colour, filled=False,
                     r=2.8, ew=0.9)
    f.end_clip()

    lx, ly = f.X(0.215), f.Y(-0.30)
    f.s.append(f'<rect x="{lx - 8}" y="{ly - 9}" width="17" height="17" fill="#5d6d7e" '
               f'fill-opacity="0.32" stroke="#34495e"/>')
    f.text(lx + 17, ly + 4, 'excluded for the planned position', 12, '#111')
    for dy, colour, lab in ((-0.355, VIOLATING, 'push crosses the keep-out region'),
                            (-0.410, CLEAN, 'push clears it')):
        y = f.Y(dy)
        f.poly([(lx - 8, y), (lx + 9, y)], colour, w=2.4)
        f.text(lx + 17, y + 4, lab, 12, '#111')

    path = f.save(os.path.join(outdir, 'fig_expert_aligning.svg'))
    return path, ('data/expert_paths.json (extract/expert_paths.py) | '
                  'd3il/environments/dataset/data/aligning/{train,test}_contexts.pkl, '
                  '60 + 60 contexts; constraints from sources.ALIGNING_CONSTRAINTS')


ALL = [
    ('fig_expert_uav', 'env', fig_expert_uav),
    ('fig_expert_aligning', 'env', fig_expert_aligning),
]
