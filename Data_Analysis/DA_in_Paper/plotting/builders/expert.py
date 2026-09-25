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
# [v3.60] The vehicle silhouette is a SCALE REFERENCE, not a verdict. A violation is read off the
# path line against surfaces that already carry the 0.31 m rotor reach, so colouring the body by
# its own overlap drew the test twice and invited the reading that the wing is what violates.
# Neutral slate; only the path is green or red. (Author, 2026-09-21.)
VEHICLE = '#5d6d7e'
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


LEGEND_LINE = 32      # px between the lines of a wrapped legend key (font 18 x FONT_CONSTRAINT = 28 px)


def _legend(width, items, cols=2):
    """One legend strip above a grid. A key may carry '\n' (v5.3, review A6): its further
    lines are drawn under the first, LEGEND_LINE apart, and the rows below move down -- so a
    long key wraps instead of running past the canvas, which fig_expert_uav's third key did
    (the PNG cut it at "...is in the surfa"). Keys without '\n' render exactly as before."""
    rows = [items[i:i + cols] for i in range(0, len(items), cols)]
    extra = [max(str(lab).count('\n') for _, _, lab in r) for r in rows]
    h = Fig(width, 20 + 40 * len(rows) + LEGEND_LINE * sum(extra), ml=0, mr=0, mt=0, mb=0,
            font=FONT_CONSTRAINT)
    col_x = tuple(16 + (width // cols) * c for c in range(cols))
    y = 26
    for r, ex in zip(rows, extra):
        for c, (colour, dash, lab) in enumerate(r):
            x = col_x[c]
            if colour is None:                   # the vehicle key: a disk, not a line
                # the legend Fig has no data axes, so this one is written in pixels
                h.s.append(f'<circle cx="{x}" cy="{y}" r="14" fill="#777" fill-opacity="0.16" '
                           f'stroke="#777" stroke-width="1.4" stroke-dasharray="5,4"/>')
                h.marker(x, y, 's', '#777', filled=True, r=4.5, ew=0.8)
            else:
                h.poly([(x - 14, y), (x + 14, y)], colour, w=3.0, dash=dash)
            for j, line in enumerate(str(lab).split('\n')):
                h.text(x + 26, y + 7 + LEGEND_LINE * j, line, 18, '#111')
        y += 40 + LEGEND_LINE * ex
    return h


# ═══════════════════════════════════════════════════════════════════════════
#  Quadrotor: the reference path the demonstrations track, per homotopy class
# ═══════════════════════════════════════════════════════════════════════════
def _pillars_panel(r, t, w=PANEL_W):
    """[v3.69b] UAV-pillars: the 96 D3IL-avoiding demonstrations mapped into the arena (scale 36,
    X = 36 (y_a - 0.035), Y = -36 (x_a - 0.5)) over the both-hard constraint set of the scene, coloured
    by whether the demonstration satisfies that geometry -- the same test and count (2 of 96) as
    fig_constraints_avoiding. The scene has no quadrotor demonstrations: this IS its expert data."""
    from builders.avoiding import _demo_satisfies
    scn = _scene('UAV-pillars')
    if scn is None or not os.path.isfile(S.AVOIDING_SCENE):
        return None
    sc = json.load(open(S.AVOIDING_SCENE))
    g = sc['geometries']['both-hard']
    ok = [_demo_satisfies(xy, g, sc['obstacles']['radius']) for xy in sc['demonstrations']]
    assert sum(ok) == g['demonstrations_satisfying']
    n = len(ok)
    local = dict(scn, title='UAV-pillars, both-hard',
                 sub=f'{n} avoiding demonstrations, mapped · {sum(ok)}/{n} clean')
    f = _uav_constraint_panel(local, w, True, r, t)
    f.clip_to_box()
    w = lambda xa, ya: (36.0 * (ya - 0.035), -36.0 * (xa - 0.5))
    # violating ones first and faint, the two satisfying ones on top
    for xy, v in sorted(zip(sc['demonstrations'], ok), key=lambda p: p[1]):
        pts = ' '.join(f'{f.X(X):.1f},{f.Y(Y):.1f}' for X, Y in (w(x, y) for x, y in xy))
        f.s.append(f'<polyline points="{pts}" fill="none" stroke="{CLEAN if v else VIOLATING}" '
                   f'stroke-width="{1.9 if v else 1.0}" stroke-opacity="{0.85 if v else 0.26}" '
                   f'stroke-linejoin="round"/>')
    f.end_clip()
    return f


def _corridor_hump_panel(r, t, w=PANEL_W, xlim=None, zlim=None, equal=False):
    """[v3.69b] UAV-corridor, hump: the roof from the side (x-z, along the corridor) with the band of
    altitudes the demonstrations were flown at, 0.90-1.30 m (tab:uav-demos): every lane passes under
    the roof, which the top view cannot show. Numbers from config/uav_projection.yaml and
    uav_expert_data_collect/generator.py. [v3.70] takes the panel box of fig_constraints_uav."""
    from builders.scenes import _uav_side_panel
    f = _uav_side_panel('hump', w, True, r, t, xlim=xlim, zlim=zlim, equal=equal)
    f.clip_to_box()
    z0, z1 = 0.90, 1.30
    x0, x1 = xlim or (-2.8, 2.8)
    f.polygon([(x0, z0), (x1, z0), (x1, z1), (x0, z1)], VIOLATING, opacity=0.16)
    f.dline([(x0, z0), (x1, z0)], VIOLATING, w=1.6)
    f.dline([(x0, z1), (x1, z1)], VIOLATING, w=1.6)
    f.text(f.X(x1) - 6, f.Y(z0) + 12, 'demonstrated altitudes 0.90-1.30 m', 9.0, VIOLATING, anchor='end')
    f.end_clip()
    return f


def fig_expert_uav(outdir):
    """The demonstrations of each scene against its constraint set, laid out as the matrix of
    fig_constraints_uav (v3.70, author 2026-09-23): one column per scene in the order of the chapter,
    the corridor column carrying the tilt (top view) over the hump (side view), every panel the same
    box. UAV-pillars: the 96 D3IL-avoiding demonstrations mapped into the arena. UAV-corridor and
    UAV-s-curve: the reference path of each passage class."""
    D = _load()
    if not D:
        return None
    C = S.UAV_CONSTRAINTS
    r, t = C['r_drone'], C['tightening']
    W = 470
    RATIO = 24.0 / 27.0                       # the box of fig_constraints_uav
    by_name = {b['scene']: b for b in D['uav']}

    def padded(scn, title, sub):
        (x0, x1) = scn['xlim']
        hy = (x1 - x0) * RATIO / 2
        return dict(scn, title=title, sub=sub, ylim=(-hy, hy))

    def routes_panel(block, title):
        scn = _scene(block['scene'])
        if not scn:
            return None
        n, clean = block['n_routes'], block['n_clean']
        corridor = block['scene'] == 'UAV-corridor'
        local = padded(scn, title, f"{n} route{'s' if n != 1 else ''} · {clean}/{n} clean · top view (x-y)"
                       + (', z = 1.11 m' if corridor else ''))
        f = _uav_constraint_panel(local, W, True, r, t)
        f.clip_to_box()
        for route in block['routes']:
            colour = CLEAN if route['clean'] else VIOLATING
            pts = [(f.X(x), f.Y(y)) for x, y in route['xy']]
            f.poly(pts, colour, w=2.1)
            f.marker(pts[0][0], pts[0][1], 's', START, filled=True, r=4.0, ew=1.2)
            f.marker(pts[-1][0], pts[-1][1], 'o', colour, filled=True, r=4.2, ew=1.3)
        # The vehicle drawn to scale at the tightest moment of each route, so the free channel
        # can be judged against the size of the aircraft. It is a ruler, not a test: the body is
        # already inside the inflated boundary, because every surface here carries r_drone.
        # Drawn last so it sits over the path.
        #
        # De-cluttered: on UAV-corridor all three lanes are tightest at the same x, and three
        # bodies drawn there are one blob. Each vehicle therefore goes to the tightest moment
        # of its own route that is clear of the vehicles already placed (by 3.4 vehicle radii) -- still a real moment
        # of that flight, and on the corridor it spreads them along the stretch where every
        # lane is inside the slide.
        placed = []
        for route in block['routes']:
            spot = route['worst_xy']
            for cand in route.get('tight_order') or []:
                cx, cy, ov = cand
                # a violating route is shown at a moment where it does violate -- that is the
                # moment worth seeing the aircraft at
                if not route['clean'] and ov <= 0:
                    continue
                if all(math.hypot(cx - px, cy - py) >= 3.4 * r for px, py in placed):
                    spot = [cx, cy]
                    break
            placed.append(spot)
            _vehicle(f, spot, r, VEHICLE)
        f.end_clip()
        return f

    # v3.71 (author, 23-09): UAV-pillars is NOT drawn -- its demonstrations are the 96 D3IL-avoiding
    # demonstrations (fig_constraints_avoiding), mapped by the same similarity map as its constraint
    # set; the chapter says so in one sentence instead of replotting them. _pillars_panel stays for
    # any later use. Layout: the corridor's two panels share the first column (tilt over hump), the
    # s-curve the second.
    tilt = routes_panel(by_name['UAV-corridor'], 'UAV-corridor, tilt') if 'UAV-corridor' in by_name else None
    scurve = routes_panel(by_name['UAV-s-curve'], 'UAV-s-curve') if 'UAV-s-curve' in by_name else None
    if tilt is None or scurve is None:
        return None
    side_x, side_z = (-1.9, 1.9), (-0.1, -0.1 + 3.8 * RATIO)
    hump = _corridor_hump_panel(r, t, W, xlim=side_x, zlim=side_z, equal=True)
    blank = lambda: Fig(W, hump.h, ml=0, mr=0, mt=0, mb=0)
    panels = [tilt, scurve, hump, blank()]

    width = 2 * W + 8
    legend = _legend(width, [
        (CLEAN, '', 'demonstration / reference path satisfies the constraints'),
        (VIOLATING, '', 'demonstration / reference path crosses them'),
        (None, '', f"the vehicle to scale, for size only\n(the {r:g} m reach is in the surfaces)"),
    ], cols=1)
    path = save_grid(panels, os.path.join(outdir, 'fig_expert_uav.svg'), cols=2, gap=8,
                     header=legend)
    return path, ('data/expert_paths.json (extract/expert_paths.py) | '
                  'uav_expert_data_collect/trajectories.py reference paths, one per homotopy '
                  'class of generator.py::HOMOTOPY_CLASSES; constraints from '
                  'sources.UAV_CONSTRAINTS, drawn by scenes._uav_constraint_panel; '
                  'UAV-pillars not drawn (v3.71): its demonstrations are the avoiding ones')


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
    # v5.3 (review A6): the subtitle is two lines now (the one-line form ran past the 620 px
    # canvas -- the PNG cut it at "...none the l"), so the top margin holds title + two lines.
    ml, mr, mt, mb = int(46 * fs), int(14 * fs), int(64 * fs), int(46 * fs)
    w = 620
    pw = w - ml - mr
    ph = pw * (y1 - y0) / (x1 - x0)
    f = Fig(w, int(round(ph + mt + mb)), ml=ml, mr=mr, mt=mt, mb=mb, font=fs)
    f.axes((x0, x1), (y0, y1))

    n = sum(b['n'] for b in D['aligning'])
    hits = sum(b['n_push_hits'] for b in D['aligning'])
    hs = sum(b.get('n_push_hits_halfspace', 0) for b in D['aligning'])   # v3.100: 0 of 120
    f.frame([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [-0.4, -0.2, 0.0, 0.2, 0.4],
            'x [m]', 'y [m]', 'Vision-conditioned alignment',
            f'{n} recorded contexts',
            xfmt=lambda v: f'{v:g}', yfmt=lambda v: f'{v:g}')
    f.text(f.L, 49 * fs, f'{hits} direct pushes cross the keep-out region, {hs or "none"} the halfspace',
           10.5, '#555')                                   # the subtitle's second line (frame draws one)
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
