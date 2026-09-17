#!/usr/bin/env python3
"""Environment figures: each simulated scene drawn from its own definition.

These answer "what IS this environment", which the schematic overhead views do
not: they show the third dimension, the real sizes and how much room there is.

Where the geometry comes from, per figure:

* the quadrotor scenes are parsed out of the MJCF the simulator loads,
  ``d3il/environments/d3il/models/mj/robot/quadrotor/scenes/*.xml`` -- so the wall
  positions here cannot drift from the ones flown against;
* the two manipulation scenes are built in Python rather than XML, so their
  primitives are transcribed in ``sources.D3IL_SCENES`` with the file and symbol
  they come from, and checked against it there.

Not a rendering of the simulator: no robot links (those are STL meshes) and no
textures. Every dimension is the real one.
"""
import os
import math
import xml.etree.ElementTree as ET

import sources as S
from svg.fmpcc_svg import Fig
from svg.scene3d import Scene3D

FONT = 1.0   # Fig's numeric text scale

WALL = '#b8bcc0'
PILLAR = '#9aa0a6'
FLOOR = '#eceef0'
OBSTACLE = '#c0392b'
GOAL = '#27ae60'
DRONE = '#2471a3'
PATH = '#e67e22'


def _drone(sc, pos, span=0.62, colour=DRONE):
    """A quadrotor at `pos`, drawn at its real rotor span (2 * r_drone = 0.62 m)."""
    x, y, z = pos
    a = span / 2 * 0.72
    sc.box((x, y, z), (0.05, 0.05, 0.025), colour)
    for dx, dy in ((a, a), (a, -a), (-a, a), (-a, -a)):
        sc.line((x, y, z), (x + dx, y + dy, z), '#154360', 1.6)
        sc.cylinder((x + dx, y + dy, z), span / 2 * 0.34, 0.006, colour, n=16)


# ═══════════════════════════════════════════════════════════════════════════
#  Quadrotor scenes -- geometry parsed from the MJCF the simulator loads
# ═══════════════════════════════════════════════════════════════════════════
def _mjcf_geoms(scene_file):
    path = os.path.join(S.REPO, S.UAV_SCENE_DIR, scene_file)
    if not os.path.isfile(path):
        return None
    out = []
    for g in ET.parse(path).getroot().iter('geom'):
        t = g.get('type')
        if t not in ('box', 'cylinder'):
            continue
        pos = [float(v) for v in (g.get('pos') or '0 0 0').split()]
        size = [float(v) for v in (g.get('size') or '').split()]
        out.append((t, g.get('name') or '', pos, size))
    return out


def _uav_scene(scene_file, cam, path_pts=None, start=None):
    # Figure 5.4 pairs this view with a MuJoCo render that carries the real reference path and
    # the vehicle (prep/render_mujoco_scenes.py), so the builders pass no path here: an
    # illustrative straight line through the pillars contradicted the weaving path above it.
    geoms = _mjcf_geoms(scene_file)
    if not geoms:
        return None
    # Extent of the real geometry, so the floor is sized to the scene instead of
    # being a fixed slab that runs off the page before the walls do.
    xs, ys, zs = [], [], []
    for t, _n, pos, size in geoms:
        r = (size[0], size[0], size[1]) if t == 'cylinder' else tuple(size[:3])
        for i, (c, e) in enumerate(zip(pos, r)):
            (xs, ys, zs)[i].extend([c - e, c + e])
    mx, my = 0.55, 0.55
    x0, x1 = min(xs) - mx, max(xs) + mx
    y0, y1 = min(ys) - my, max(ys) + my
    zmax = max(zs)

    sc = Scene3D(760, 400, font=FONT, **cam)
    sc.fit(Scene3D.bbox_corners((x0, y0, 0.0), (x1, y1, zmax)),
           top=18, bottom=18)
    sc.ground(x0, x1, y0, y1, FLOOR, step=1.0)
    for t, _name, pos, size in geoms:
        if t == 'box':
            # Translucent on purpose: at any angle that shows the corridor is a
            # corridor, the near wall stands between the reader and the flight path.
            sc.box(pos, size, WALL, opacity=0.5)
        else:
            sc.cylinder(pos, size[0], size[1], PILLAR)
    if path_pts:
        sc.path(path_pts, PATH, w=2.4, dash='7,5')
    if start:
        _drone(sc, start)
    sc.draw()
    return sc


def fig_scene_uav_corridor(outdir):
    """Corridor v2: the scene the corridor results are flown in (U16, 2026-09-13)."""
    sc = _uav_scene(
        'scene_corridor_v2.xml',
        dict(azimuth=58, elevation=34))
    if sc is None:
        return None
    path = sc.save(os.path.join(outdir, 'fig_scene_uav_corridor.svg'))
    return path, 'd3il/.../quadrotor/scenes/scene_corridor_v2.xml | MJCF geometry, orthographic'


def fig_scene_uav_pillars(outdir):
    sc = _uav_scene(
        'scene_pillars.xml',
        dict(azimuth=54, elevation=30))
    if sc is None:
        return None
    path = sc.save(os.path.join(outdir, 'fig_scene_uav_pillars.svg'))
    return path, 'd3il/.../quadrotor/scenes/scene_pillars.xml | MJCF geometry, orthographic'


def fig_scene_uav_scurve(outdir):
    sc = _uav_scene(
        'scene_s_curve.xml',
        dict(azimuth=52, elevation=40))
    if sc is None:
        return None
    path = sc.save(os.path.join(outdir, 'fig_scene_uav_scurve.svg'))
    return path, 'd3il/.../quadrotor/scenes/scene_s_curve.xml | MJCF geometry, orthographic'


# ═══════════════════════════════════════════════════════════════════════════
#  Manipulation scenes -- primitives transcribed in sources.D3IL_SCENES
# ═══════════════════════════════════════════════════════════════════════════
def fig_scene_avoiding(outdir):
    """The obstacle-avoidance scene in three dimensions: posts on a table."""
    d = S.D3IL_SCENES['avoiding']
    # The workspace the constraints and the demonstrations live in.
    wx0, wx1, wy0, wy1 = 0.18, 0.82, -0.34, 0.42
    sc = Scene3D(760, 400, azimuth=64, elevation=30, font=FONT)
    sc.fit(Scene3D.bbox_corners((wx0, wy0, 0.0), (wx1, wy1, 0.21)), top=18, bottom=18)
    sc.ground(wx0, wx1, wy0, wy1, FLOOR, step=0.1)
    # The finish line is 1 m long in the simulator and would run well past the
    # table; it is drawn clipped to the workspace, which is the part that is ever
    # crossed. Its y position and thickness are the real ones.
    gx, gy, gz = d['finish_line']['pos']
    hx, hy, hz = d['finish_line']['half']
    sc.box(((wx0 + wx1) / 2, gy, gz + hz), ((wx1 - wx0) / 2, hy, hz), GOAL, opacity=0.9)
    for o in d['obstacles']:
        # MuJoCo places a cylinder by its centre; these sit ON the table, so the
        # drawn centre is the half-height above it.
        x, y, _z = o['pos']
        r, hh = o['size']
        sc.cylinder((x, y, hh), r, hh, OBSTACLE)
    sx, sy, _sz = d['start']
    sc.cylinder((sx, sy, 0.005), 0.018, 0.005, '#1b4f72', n=20)
    sc.text(*sc.project((sx, sy - 0.035, 0.0)), 'start', 10.5, '#1b4f72', anchor='middle', bold=True)
    sc.text(*sc.project((0.30, d['finish_line']['pos'][1] + 0.035, 0.0)), 'finish line',
            10.5, '#1e8449', anchor='middle', bold=True)
    sc.draw()
    path = sc.save(os.path.join(outdir, 'fig_scene_avoiding.svg'))
    return path, 'd3il gym_avoiding_env avoiding_objects.get_obj_list | primitives, orthographic'


def fig_scene_aligning(outdir):
    """The alignment scene: the box with its four coloured sides, and the target pose."""
    d = S.D3IL_SCENES['aligning']
    # The floor has to contain the start pose of the end effector (y = -0.35),
    # or the start marker floats off the table.
    wx0, wx1, wy0, wy1 = 0.30, 0.82, -0.44, 0.50
    sc = Scene3D(760, 400, azimuth=58, elevation=30, font=FONT)
    sc.fit(Scene3D.bbox_corners((wx0, wy0, 0.0), (wx1, wy1, 0.14)), top=18, bottom=18)
    sc.ground(wx0, wx1, wy0, wy1, FLOOR, step=0.1)

    def push_box(cx, cy, yaw_deg, opacity, base_col):
        yaw = math.radians(yaw_deg)
        sc.box((cx, cy, 0.01), (0.05, 0.05, 0.01), base_col,
               opacity=opacity, yaw=yaw)
        for (dx, dy), (hx, hy), col in (((0.05, 0), (0.005, 0.05), '#2e5fa3'),
                                        ((0, 0.05), (0.05, 0.005), '#2e8b57'),
                                        ((-0.05, 0), (0.005, 0.05), '#b03a2e'),
                                        ((0, -0.05), (0.05, 0.005), '#17a2b8')):
            rdx = dx * math.cos(yaw) - dy * math.sin(yaw)
            rdy = dx * math.sin(yaw) + dy * math.cos(yaw)
            sc.box((cx + rdx, cy + rdy, 0.0485), (hx, hy, 0.045), col,
                   opacity=opacity, yaw=yaw)

    tx, ty, _ = d['target_pos']
    push_box(tx, ty, d['target_yaw_deg'], 0.30, '#d7cbb4')
    bx, by, _ = d['box_pos']
    push_box(bx, by, d['box_yaw_deg'], 1.0, '#cba872')
    sx, sy, _sz = d['start']
    sc.cylinder((sx, sy, 0.005), 0.018, 0.005, '#1b4f72', n=20)
    sc.text(*sc.project((sx, sy - 0.05, 0.0)), 'start', 10.5, '#1b4f72', anchor='middle', bold=True)
    sc.text(*sc.project((tx, ty + 0.10, 0.0)), 'target pose', 10.5, '#7d6608',
            anchor='middle', bold=True)
    sc.draw()
    path = sc.save(os.path.join(outdir, 'fig_scene_aligning.svg'))
    return path, 'd3il robot_push_box.xml + aligning_objects.py | primitives, orthographic'


# ═══════════════════════════════════════════════════════════════════════════
#  The constraint set of each quadrotor scene, drawn from above (v3.27).
#  Companion of fig_constraints_avoiding: same vocabulary -- halfspaces, disks,
#  a shaded excluded region and a dashed tightened boundary -- so the reader can
#  compare the aerial constraint sets with the manipulation one directly.
# ═══════════════════════════════════════════════════════════════════════════
def _uav_constraint_panel(scn, w, first, r_drone, tight):
    from svg.fmpcc_svg import clip_halfplane
    (x0, x1), (y0, y1) = scn['xlim'], scn['ylim']
    ml, mr, mt, mb = 48, 12, 46, 46
    pw = w - ml - mr
    ph = pw * (y1 - y0) / (x1 - x0)                  # equal aspect: 1 m is 1 m on both axes
    f = Fig(w, int(round(ph + mt + mb)), ml=ml, mr=mr, mt=mt, mb=mb, font=FONT)
    f.axes((x0, x1), (y0, y1))
    tx = [v for v in range(int(math.ceil(x0)), int(math.floor(x1)) + 1)]
    ty = [v for v in range(int(math.ceil(y0)), int(math.floor(y1)) + 1)]
    f.frame(tx, ty, 'x [m]', 'y [m]' if first else '', scn['title'], scn['sub'],
            xfmt=lambda v: f'{v:g}', yfmt=lambda v: f'{v:g}')
    f.clip_to_box()
    for hs in scn['halfspaces']:
        (ax_, ay), (bx, by) = hs['p0'], hs['p1']
        m = (by - ay) / (bx - ax_)
        b = ay - m * ax_
        below = hs['side'] == 'below'                # the plan must stay below the line
        xa, xb = hs['x_active']
        span = [(xa, y0), (xb, y0), (xb, y1), (xa, y1)]
        # the boundary the solver sees: the wall moved into the free side by the vehicle radius
        shift = -(1 + m * m) ** 0.5 * r_drone if below else (1 + m * m) ** 0.5 * r_drone
        for off, fill in ((0.0, None), (shift, 'band')):
            keep = ((lambda x, y, m=m, b=b, o=off: y - (m * x + b + o)) if below
                    else (lambda x, y, m=m, b=b, o=off: (m * x + b + o) - y))
            poly = clip_halfplane(span, keep)
            if len(poly) > 2:
                f.polygon(poly, '#5d6d7e', opacity=0.34 if fill is None else 0.18)
        xs = (xa, xb)
        f.dline([(x, m * x + b) for x in xs], '#34495e', w=2.2)
        f.dline([(x, m * x + b + shift) for x in xs], '#34495e', w=1.2)
        d = shift + (-tight if below else tight) * (1 + m * m) ** 0.5
        f.dline([(x, m * x + b + d) for x in xs], '#34495e', w=1.4, dash='7,5')
    for dk in scn['disks']:
        cx, cy = dk['c']
        f.circle(cx, cy, dk['r'] + r_drone, fill='#5d6d7e', opacity=0.18, stroke='#34495e', w=1.2)
        f.circle(cx, cy, dk['r'] + r_drone + tight, stroke='#34495e', w=1.4, dash='7,5')
        f.circle(cx, cy, dk['r'], fill='#c0392b', opacity=0.9, stroke='#7b241c', w=1.0)
    f.end_clip()
    return f


def fig_constraints_uav(outdir):
    """The three aerial constraint sets, from above, as the projection sees them."""
    from svg.fmpcc_svg import save_grid
    C = getattr(S, 'UAV_CONSTRAINTS', None)
    if not C:
        return None
    r, t = C['r_drone'], C['tightening']
    panels = [_uav_constraint_panel(scn, 470, i == 0, r, t) for i, scn in enumerate(C['scenes'])]
    width = sum(p.w for p in panels) + 2 * 8
    h = Fig(width, 56, ml=0, mr=0, mt=0, mb=0, font=FONT)
    x = 16
    items = [('wall', 'wall or pillar'),
             ('area', 'excluded: the obstacle itself'),
             ('band', f'excluded: vehicle radius {r:g} m'),
             ('dash', f'tightened by a further {t:g} m')]
    for kind, lab in items:
        if kind == 'wall':
            h.poly([(x - 10, 24), (x + 10, 24)], '#34495e', w=2.6)
        elif kind == 'area':
            h.s.append(f'<rect x="{x - 8}" y="16" width="16" height="16" fill="#5d6d7e" '
                       f'fill-opacity="0.34" stroke="#34495e"/>')
        elif kind == 'band':
            h.s.append(f'<rect x="{x - 8}" y="16" width="16" height="16" fill="#5d6d7e" '
                       f'fill-opacity="0.18" stroke="#34495e"/>')
        else:
            h.poly([(x - 10, 24), (x + 10, 24)], '#34495e', dash='7,5', w=2)
        h.text(x + 20, 31, lab, 11, '#111')
        x += 42 + len(lab) * 10.0
    path = save_grid(panels, os.path.join(outdir, 'fig_constraints_uav.svg'), cols=3, gap=8, header=h)
    return path, ('config/uav_projection.yaml :: corridor_v2_slide, pillars_hg, s_curve_hg '
                  f'| transcribed in sources.UAV_CONSTRAINTS; inflation r_drone {r:g} m, tightening {t:g} m')


ALL = [
    ('fig_scene_avoiding', 'env', fig_scene_avoiding),
    ('fig_scene_aligning', 'env', fig_scene_aligning),
    ('fig_scene_uav_corridor', 'env', fig_scene_uav_corridor),
    ('fig_scene_uav_pillars', 'env', fig_scene_uav_pillars),
    ('fig_scene_uav_scurve', 'env', fig_scene_uav_scurve),
    ('fig_constraints_uav', 'env', fig_constraints_uav),
]
