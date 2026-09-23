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


# v3.69 (author, 2026-09-23): the three orthographic views stand under MuJoCo renders that look
# along +x from behind the start; they used to look back from the finish (camera at +x, +y), which
# read as a left-right mirror of the render above them. Same obliqueness, camera moved 180 deg.
def fig_scene_uav_corridor(outdir):
    """Corridor v2: the scene the corridor results are flown in (U16, 2026-09-13)."""
    sc = _uav_scene(
        'scene_corridor_v2.xml',
        dict(azimuth=196, elevation=34))
    if sc is None:
        return None
    path = sc.save(os.path.join(outdir, 'fig_scene_uav_corridor.svg'))
    return path, 'd3il/.../quadrotor/scenes/scene_corridor_v2.xml | MJCF geometry, orthographic'


def fig_scene_uav_pillars(outdir):
    """Pillars v2 (v3.68b): the D3IL-avoiding field at scale 36, extruded into pillars (Gen15 U18)."""
    sc = _uav_scene(
        'scene_avoiding_pillars_s36.xml',
        dict(azimuth=204, elevation=30))
    if sc is None:
        return None
    path = sc.save(os.path.join(outdir, 'fig_scene_uav_pillars.svg'))
    return path, 'd3il/.../quadrotor/scenes/scene_avoiding_pillars_s36.xml | MJCF geometry, orthographic'


def fig_scene_uav_scurve(outdir):
    sc = _uav_scene(
        'scene_s_curve.xml',
        dict(azimuth=196, elevation=40))
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
# The constraint panels carry small numbers over a wide figure, so they are drawn at a larger
# text scale than the scene renders: at \linewidth across three panels, FONT alone is unreadable.
FONT_CONSTRAINT = 1.55


def _uav_constraint_panel(scn, w, first, r_drone, tight):
    from svg.fmpcc_svg import clip_halfplane
    # v3.68b: a scene may carry its own inflation and tightening (UAV-pillars: the avoiding
    # constraint set mapped as it is, r_drone 0, tightening 0.025 * 36)
    r_drone = scn.get('r_drone', r_drone)
    tight = scn.get('tightening', tight)
    fs = FONT_CONSTRAINT
    (x0, x1), (y0, y1) = scn['xlim'], scn['ylim']
    ml, mr, mt, mb = int(42 * fs), int(10 * fs), int(46 * fs), int(46 * fs)
    pw = w - ml - mr
    ph = pw * (y1 - y0) / (x1 - x0)                  # equal aspect: 1 m is 1 m on both axes
    f = Fig(w, int(round(ph + mt + mb)), ml=ml, mr=mr, mt=mt, mb=mb, font=fs)
    f.axes((x0, x1), (y0, y1))
    step = 5 if (x1 - x0) > 12 else 1
    tx = [v for v in range(int(math.ceil(x0)), int(math.floor(x1)) + 1) if v % step == 0]
    ty = [v for v in range(int(math.ceil(y0)), int(math.floor(y1)) + 1) if v % step == 0]
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
        # v3.37: drawn like the pillar panel, which reads at a glance -- the obstacle in red,
        # ONE light band for the vehicle radius around it, the tightened boundary dashed. The
        # earlier version filled the whole far side dark AND the band light, so a corridor panel
        # was mostly shading and the free channel was the hardest thing in it to find. The far
        # side keeps a faint wash, because a halfspace does exclude everything beyond it and the
        # slide is not a wall one can see the end of.
        shift = -(1 + m * m) ** 0.5 * r_drone if below else (1 + m * m) ** 0.5 * r_drone
        keep_far = ((lambda x, y, m=m, b=b: y - (m * x + b)) if below
                    else (lambda x, y, m=m, b=b: (m * x + b) - y))
        poly = clip_halfplane(span, keep_far)
        if len(poly) > 2:
            f.polygon(poly, '#5d6d7e', opacity=0.10)
        # the band is the strip between the wall and its inflated boundary, on the free side
        if below:                                    # free side is below; shift is negative
            k1 = lambda x, y, m=m, b=b, o=shift: y - (m * x + b + o)
            k2 = lambda x, y, m=m, b=b: (m * x + b) - y
        else:                                        # free side is above; shift is positive
            k1 = lambda x, y, m=m, b=b, o=shift: (m * x + b + o) - y
            k2 = lambda x, y, m=m, b=b: y - (m * x + b)
        band = clip_halfplane(clip_halfplane(span, k1), k2)
        if len(band) > 2:
            f.polygon(band, '#5d6d7e', opacity=0.24)
        xs = (xa, xb)
        # Slate, not red: these panels are also the backdrop of the flown-path figures, where red
        # means a flight that entered an obstacle. A red wall LINE reads as a red path; the small
        # red pillar cores never did, so those keep their colour.
        f.dline([(x, m * x + b) for x in xs], '#34495e', w=2.6)          # the obstacle itself
        f.dline([(x, m * x + b + shift) for x in xs], '#34495e', w=1.2)  # inflated by the radius
        d = shift + (-tight if below else tight) * (1 + m * m) ** 0.5
        f.dline([(x, m * x + b + d) for x in xs], '#34495e', w=1.4, dash='7,5')
    for dk in scn['disks']:
        cx, cy = dk['c']
        if dk.get('physical'):
            # v3.69: a physical obstacle that is NOT a constraint (the pillars of UAV-pillars, whose
            # constraint set is the avoiding one): the solid body, no inflation ring, no tightening.
            f.circle(cx, cy, dk['r'], fill='#c0392b', opacity=0.9, stroke='#7b241c', w=1.0)
            continue
        f.circle(cx, cy, dk['r'] + r_drone, fill='#5d6d7e', opacity=0.18, stroke='#34495e', w=1.2)
        f.circle(cx, cy, dk['r'] + r_drone + tight, stroke='#34495e', w=1.4, dash='7,5')
        if dk.get('keepout'):
            # v3.69: a keep-out disk, the constraint of the avoiding geometry mapped into the arena:
            # translucent, so the pillar it covers stays visible inside it.
            f.circle(cx, cy, dk['r'], fill='#c0392b', opacity=0.28, stroke='#7b241c', w=1.4)
            continue
        # A disk whose ENFORCED radius differs from the physical obstacle (UAV-pillars at
        # `pillars_xl`) is drawn as both: the enforced keep-out in red, and the pillar the
        # simulator contains as a solid core inside it. Anything else would tell the reader the
        # obstacle grew, when what grew is the constraint.
        r_phys = dk.get('r_phys')
        if r_phys is None:
            f.circle(cx, cy, dk['r'], fill='#c0392b', opacity=0.9, stroke='#7b241c', w=1.0)
        else:
            f.circle(cx, cy, dk['r'], fill='#c0392b', opacity=0.28, stroke='#7b241c', w=1.4)
            f.circle(cx, cy, r_phys, fill='#7b241c', opacity=0.95, stroke='#7b241c', w=1.0)
    f.end_clip()
    return f


def _uav_side_panel(kind, w, first, r_drone, tight, xlim=None, zlim=None, equal=False):
    """[v3.68d] UAV-corridor from the side (x-z), route C: the two constraints that leave the
    horizontal plane. `kind` = 'tilt' (the v2 slide leaned -60 deg about z_ref = 1.11: the body
    clears when s_xy(x,0) - tan(60)(z - z_ref) >= r, i.e. below a ceiling that descends along x)
    or 'hump' (the roof 0 -> 1.10 m at x = 0 -> 0 over x in [-1.5, 1.5]; the body clears above
    the roof inflated by r*sqrt(1+m^2)). Numbers from config/uav_projection.yaml
    (corridor_v3_tilt: z_lean deg -60 z_ref 1.11, ub_z 1.80; corridor_v3_ablation_hump: ub_z 2.80).
    """
    fs = FONT_CONSTRAINT
    (x0, x1) = xlim or (-2.8, 2.8)
    (z0, z1) = zlim or ((0.2, 1.9) if kind == 'tilt' else (0.2, 2.95))
    ml, mr, mt, mb = int(42 * fs), int(10 * fs), int(46 * fs), int(46 * fs)
    # v3.69: `equal` gives the panel the same equal-aspect box as _uav_constraint_panel, so the
    # side view sits in the constraint matrix at the size of the top views.
    ph = int(round((w - ml - mr) * (z1 - z0) / (x1 - x0) + mt + mb)) if equal else 300
    f = Fig(w, ph, ml=ml, mr=mr, mt=mt, mb=mb, font=fs)
    f.axes((x0, x1), (z0, z1))
    title = 'UAV-corridor, tilt' if kind == 'tilt' else 'UAV-corridor, hump'
    sub = 'side view (x-z) along the corridor, y = 0'
    zt = [v / 2 for v in range(int(math.ceil(z0 * 2)), int(math.floor(z1 * 2)) + 1) if v > 0]
    f.frame([v for v in (-2, -1, 0, 1, 2) if x0 < v < x1], zt,
            'x [m]', 'z [m]' if first else '', title, sub,
            xfmt=lambda v: f'{v:g}', yfmt=lambda v: f'{v:.1f}')
    f.clip_to_box()
    lb_z, ub_z = 0.30, (1.80 if kind == 'tilt' else 2.80)
    for z in (lb_z + r_drone, ub_z - r_drone):
        f.dline([(x0, z), (x1, z)], '#5d6d7e', w=1.2, dash='3,4')
    f.text(f.X(x1) - 6, f.Y(ub_z - r_drone) + (12 if kind == 'tilt' else -5), 'workspace box after inflation', 9.0, '#5d6d7e', anchor='end')
    xs = [x0 + (x1 - x0) * i / 200 for i in range(201)]
    if kind == 'tilt':
        t = math.tan(math.radians(60.0)); z_ref = 1.11
        m = (-0.05 - 0.95) / 4.0; b = 0.95 - m * (-2.0)
        s_xy = lambda x: (m * x + b) / math.hypot(1.0, m)       # signed distance below the slide at y = 0
        active = [x for x in xs if -2.0 <= x <= 2.0]
        ceil = lambda x, extra: z_ref + (s_xy(x) - r_drone - extra) / t
        raw = lambda x: z_ref + s_xy(x) / t                      # the plane itself (a point clears it)
        f.polygon([(active[0], z1)] + [(x, ceil(x, 0.0)) for x in active] + [(active[-1], z1)],
                  '#5d6d7e', opacity=0.10)
        f.polygon([(x, raw(x)) for x in active] + [(x, ceil(x, 0.0)) for x in reversed(active)],
                  '#5d6d7e', opacity=0.24)
        f.dline([(x, raw(x)) for x in active], '#34495e', w=2.6)
        f.dline([(x, ceil(x, 0.0)) for x in active], '#34495e', w=1.2)
        f.dline([(x, ceil(x, tight)) for x in active], '#34495e', w=1.4, dash='7,5')
        f.text(f.X(x0) + 6, f.Y(z1) + 14, 'excluded: above the leaned plane', 9.0, '#34495e')
    else:
        H = 1.10; m = H / 1.5
        k = math.hypot(1.0, m)
        roof = lambda x: max(0.0, H - abs(x) * m) if abs(x) <= 1.5 else 0.0
        lift = lambda x, extra: roof(x) + (r_drone + extra) * (k if abs(x) <= 1.5 else 1.0)
        span = [x for x in xs if -1.5 <= x <= 1.5]
        f.polygon([(span[0], z0)] + [(x, lift(x, 0.0)) for x in span] + [(span[-1], z0)],
                  '#5d6d7e', opacity=0.10)
        f.polygon([(x, roof(x)) for x in span] + [(x, lift(x, 0.0)) for x in reversed(span)],
                  '#5d6d7e', opacity=0.24)
        f.dline([(x, roof(x)) for x in span], '#34495e', w=2.6)
        f.dline([(x, lift(x, 0.0)) for x in span], '#34495e', w=1.2)
        f.dline([(x, lift(x, tight)) for x in span], '#34495e', w=1.4, dash='7,5')
        f.text(f.X(0.0), f.Y(max(0.42, z0 + 0.2)), 'the roof: excluded below', 9.0, '#34495e', anchor='middle')
    # the launch altitude and the flown band of the level corridor, for scale
    f.dline([(x0, 1.11), (x1, 1.11)], '#2471a3', w=1.2, dash='1,3')
    f.text(f.X(x0) + 6, f.Y(1.11) + 12 if kind == 'tilt' else f.Y(1.11) - 5, 'launch altitude 1.11 m', 9.0, '#2471a3')
    f.end_clip()
    return f


def fig_constraints_uav(outdir):
    """[v3.69, author 2026-09-23] The quadrotor constraint sets as a matrix: one column per scene in
    the order of the chapter (pillars, corridor, s-curve), one panel per constraint set, every panel
    the same size. UAV-pillars: the three avoiding geometries mapped into the arena. UAV-corridor:
    the tilt (top view at the launch altitude, where the leaned plane cuts the corridor) and the hump
    (side view, the roof). UAV-s-curve: its one set. Top views share one aspect ratio (RATIO) so the
    equal-aspect panels come out the same height; the side view is given the same box."""
    from svg.fmpcc_svg import save_grid
    C = getattr(S, 'UAV_CONSTRAINTS', None)
    G = getattr(S, 'UAV_PILLARS_GEOMETRIES', None)
    if not C or not G:
        return None
    r, t = C['r_drone'], C['tightening']
    W = 470
    RATIO = 24.0 / 27.0                       # pillars: x 27 m by y 24 m; every top view is padded to it
    by_name = {scn['name']: scn for scn in C['scenes']}

    def padded(scn, title, sub):
        (x0, x1) = scn['xlim']
        hy = (x1 - x0) * RATIO / 2
        return dict(scn, title=title, sub=sub, ylim=(-hy, hy))

    pillars = [_uav_constraint_panel(dict(G[n], sub=n + ' · top view (x-y)'), W, True, r, t)
               for n in ('top-left-hard', 'top-right-hard', 'both-hard')]
    # v3.69b (author): the corridor column carries the tilt twice -- from above, where the leaned
    # plane cuts the launch altitude, and from the side, where the lean itself is -- and the hump
    # from the side; every subtitle names the plane it is drawn in.
    tilt = _uav_constraint_panel(padded(by_name['UAV-corridor'], 'UAV-corridor, tilt',
                                        'top view (x-y) at the launch altitude, 1.11 m'), W, True, r, t)
    side_x, side_z = (-1.9, 1.9), (-0.1, -0.1 + 3.8 * RATIO)
    tilt_side = _uav_side_panel('tilt', W, True, r, t, xlim=side_x, zlim=side_z, equal=True)
    hump = _uav_side_panel('hump', W, True, r, t, xlim=side_x, zlim=side_z, equal=True)
    scurve = _uav_constraint_panel(padded(by_name['UAV-s-curve'], 'UAV-s-curve', 'top view (x-y)'), W, True, r, t)
    blank = lambda: Fig(W, pillars[0].h, ml=0, mr=0, mt=0, mb=0)
    panels = [pillars[0], tilt, scurve,
              pillars[1], tilt_side, blank(),
              pillars[2], hump, blank()]
    width = 3 * W + 2 * 8
    h = Fig(width, 122, ml=0, mr=0, mt=0, mb=0, font=FONT_CONSTRAINT)
    items = [('wall', 'the constraint boundary: wall, plane or roof'),
             ('band', f'excluded: inflated by the vehicle radius {r:g} m'),
             ('area', 'excluded: beyond the boundary'),
             ('dash', 'tightened boundary'),
             ('keep', 'keep-out disk (a constraint)'),
             ('pill', 'obstacle: wall end, corner or pillar')]
    # three rows of two: at this text size three columns run off the figure
    row_y, col_x, size = (26, 64, 102), (16, width // 2 + 16), 16
    for i, (kind, lab) in enumerate(items):
        x, y = col_x[i % 2], row_y[i // 2]
        if kind == 'wall':
            h.poly([(x - 14, y), (x + 14, y)], '#34495e', w=3.6)
        elif kind == 'area':
            h.s.append(f'<rect x="{x - 11}" y="{y - 11}" width="22" height="22" fill="#5d6d7e" '
                       f'fill-opacity="0.10" stroke="#34495e"/>')
        elif kind == 'band':
            h.s.append(f'<rect x="{x - 11}" y="{y - 11}" width="22" height="22" fill="#5d6d7e" '
                       f'fill-opacity="0.24" stroke="#34495e"/>')
        elif kind == 'dash':
            h.poly([(x - 14, y), (x + 14, y)], '#34495e', dash='9,6', w=3)
        elif kind == 'keep':
            h.s.append(f'<circle cx="{x}" cy="{y}" r="11" fill="#c0392b" fill-opacity="0.28" stroke="#7b241c"/>')
        else:
            h.s.append(f'<circle cx="{x}" cy="{y}" r="9" fill="#c0392b" fill-opacity="0.9" stroke="#7b241c"/>')
        h.text(x + 26, y + 7, lab, size, '#111')
    path = save_grid(panels, os.path.join(outdir, 'fig_constraints_uav.svg'), cols=3, gap=8, header=h)
    return path, ('config/uav_projection.yaml :: corridor_v3_tilt (cut at z_ref), corridor_v3_ablation_hump '
                  '(side view), s_curve_hg; UAV-pillars = data/avoiding_scene.json, the three geometries mapped '
                  'by uav_avoiding_bridge/frame.py at scale 36 | transcribed in sources.UAV_CONSTRAINTS / '
                  f'UAV_PILLARS_GEOMETRIES; inflation r_drone {r:g} m, tightening {t:g} m (pillars: 0 / 0.90 m)')


# ═══════════════════════════════════════════════════════════════════════════
#  The alignment constraint set (v3.31), the companion of fig_constraints_avoiding
#  and fig_constraints_uav: the workspace box the planned end-effector position
#  must stay inside, seen from above and from the side.
# ═══════════════════════════════════════════════════════════════════════════
def aligning_constraint_panel(w, title, sub, ylab, font):
    """An empty panel of the alignment constraint set: the excluded halfspace, its tightened
    boundary and the keep-out disk, on the plane of ALIGNING_CONSTRAINTS.

    Factored out of fig_constraints_aligning (v3.50) so that the executed-path figure of
    Chapter 6 draws the same constraint set from the same declaration, the way the quadrotor
    path figures reuse _uav_constraint_panel. Returns the figure with the clip still OPEN,
    for the caller to draw into.
    """
    from svg.fmpcc_svg import clip_halfplane
    C = S.ALIGNING_CONSTRAINTS
    t = C['tightening']
    (x0, x1), (y0, y1) = C['extent']['x'], C['extent']['y']
    ml, mr, mt, mb = int(46 * font), int(14 * font), int(46 * font), int(46 * font)
    pw = w - ml - mr
    ph = pw * (y1 - y0) / (x1 - x0)
    f = Fig(w, int(round(ph + mt + mb)), ml=ml, mr=mr, mt=mt, mb=mb, font=font)
    f.axes((x0, x1), (y0, y1))
    f.frame([0.2, 0.4, 0.6, 0.8], [-0.4, -0.2, 0.0, 0.2, 0.4],
            'x [m]', 'y [m]' if ylab else '', title, sub,
            xfmt=lambda v: f'{v:g}', yfmt=lambda v: f'{v:g}')
    f.clip_to_box()
    hs = C['halfspace']
    (ax_, ay), (bx, by) = hs['p0'], hs['p1']
    m = (by - ay) / (bx - ax_)
    b = ay - m * ax_
    box = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    poly = clip_halfplane(box, lambda X, Y: Y - (m * X + b))       # excluded: above the line
    if len(poly) > 2:
        f.polygon(poly, '#5d6d7e', opacity=0.32)
    shift = -t * (1 + m * m) ** 0.5
    xs = (x0 - 0.05, x1 + 0.05)
    f.dline([(x, m * x + b) for x in xs], '#34495e', w=2.0)
    f.dline([(x, m * x + b + shift) for x in xs], '#34495e', w=1.5, dash='8,5')
    dk = C['disk']
    f.circle(dk['c'][0], dk['c'][1], dk['r'], fill='#5d6d7e', opacity=0.32, stroke='#34495e', w=1.6)
    f.circle(dk['c'][0], dk['c'][1], dk['r'] + t, stroke='#34495e', w=1.5, dash='8,5')
    return f


def fig_constraints_aligning(outdir):
    """The alignment constraint set: the halfspace and the keep-out disk the plan must respect."""
    from svg.fmpcc_svg import clip_halfplane
    C = getattr(S, 'ALIGNING_CONSTRAINTS', None)
    d = S.D3IL_SCENES.get('aligning')
    if not C or not d:
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
    f.frame([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [-0.4, -0.2, 0.0, 0.2, 0.4],
            'x [m]', 'y [m]', 'Vision-conditioned alignment',
            'the constraint set the planned end-effector position must satisfy',
            xfmt=lambda v: f'{v:g}', yfmt=lambda v: f'{v:g}')
    f.clip_to_box()

    # halfspace: the plan must stay below the line; shade the excluded side and the tightened shift
    hs = C['halfspace']
    (ax_, ay), (bx, by) = hs['p0'], hs['p1']
    m = (by - ay) / (bx - ax_)
    b = ay - m * ax_
    box = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    keep = lambda X, Y: Y - (m * X + b)                     # excluded: above the line
    poly = clip_halfplane(box, keep)
    if len(poly) > 2:
        f.polygon(poly, '#5d6d7e', opacity=0.32)
    shift = -t * (1 + m * m) ** 0.5
    xs = (x0 - 0.05, x1 + 0.05)
    f.dline([(x, m * x + b) for x in xs], '#34495e', w=2.4)
    f.dline([(x, m * x + b + shift) for x in xs], '#34495e', w=1.6, dash='8,5')

    # circular keep-out region
    dk = C['disk']
    cx, cy = dk['c']
    f.circle(cx, cy, dk['r'], fill='#5d6d7e', opacity=0.32, stroke='#34495e', w=1.8)
    f.circle(cx, cy, dk['r'] + t, stroke='#34495e', w=1.6, dash='8,5')

    # The box and its target drawn TO SCALE and at their recorded yaw, not as markers: the
    # footprint is 0.10 m square, larger than the 0.06 m keep-out disk, and a marker would
    # tell the reader the opposite.  Centre of each is marked, because the context, the
    # success metric and the constraint all refer to the centre, not to the footprint.
    hb, hr = d['box_half'], d['box_rim_half']

    def footprint(px, py, yaw_deg, half):
        a_ = math.radians(yaw_deg)
        ca, sa = math.cos(a_), math.sin(a_)
        return [(px + dx * ca - dy * sa, py + dx * sa + dy * ca)
                for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half))]

    def centre_mark(px, py, colour):
        r = 0.011
        f.dline([(px - r, py), (px + r, py)], colour, w=1.6)
        f.dline([(px, py - r), (px, py + r)], colour, w=1.6)
        f.marker(f.X(px), f.Y(py), 'o', colour, filled=True, r=2.6, ew=1.0)

    for kind, pos, yaw, lab in (('target', d['target_pos'], d['target_yaw_deg'], 'target pose'),
                                ('box', d['box_pos'], d['box_yaw_deg'], 'box at the start')):
        px, py = pos[0], pos[1]
        fill = '#d7cbb4' if kind == 'target' else '#cba872'
        edge = '#8a7a55' if kind == 'target' else '#6b5a32'
        f.polygon(footprint(px, py, yaw, hr), fill, opacity=0.35 if kind == 'target' else 0.75,
                  stroke=edge, w=1.4)
        f.polygon(footprint(px, py, yaw, hb), fill, opacity=0.55 if kind == 'target' else 0.95,
                  stroke=edge, w=1.6)
        centre_mark(px, py, edge)
        # below the footprint, centred: beside it the label crosses the box's own corner
        f.text(f.X(px), f.Y(py - hr * 1.41) + 16, lab, 11, '#111', bold=True, anchor='middle')
        f.text(f.X(px), f.Y(py - hr * 1.41) + 30, '0.10 m square', 10, '#555', anchor='middle')

    sx, sy = d['start'][0], d['start'][1]
    f.marker(f.X(sx), f.Y(sy), 'o', '#1b4f72', r=8.0)
    f.text(f.X(sx) + 12, f.Y(sy) + 5, 'end effector', 11, '#111', bold=True)
    f.end_clip()

    # legend inside the panel, in the corner the task leaves empty
    lx, ly = f.X(0.215), f.Y(0.41)
    f.s.append(f'<rect x="{lx - 8}" y="{ly - 9}" width="17" height="17" fill="#5d6d7e" '
               f'fill-opacity="0.32" stroke="#34495e"/>')
    f.text(lx + 17, ly + 4, 'excluded for the planned position', 12, '#111')
    ly2 = f.Y(0.355)
    f.poly([(lx - 8, ly2), (lx + 9, ly2)], '#34495e', dash='8,5', w=2.2)
    f.text(lx + 17, ly2 + 4, f'tightened by {t:g} m', 12, '#111')

    path = f.save(os.path.join(outdir, 'fig_constraints_aligning.svg'))
    return path, ('config/visual_aligning_eval.yaml :: combined_5 halfspace_constraints and '
                  f'obstacle_constraints, enlarge_constraints {t:g} m | transcribed in '
                  'sources.ALIGNING_CONSTRAINTS; box, target and start from sources.D3IL_SCENES[aligning]')


# ═══════════════════════════════════════════════════════════════════════════
#  The ten alignment contexts (v3.42): what "the context is drawn at random"
#  actually looks like. The prose gives the draw ranges as intervals; a reader
#  cannot tell from intervals whether ten draws cover them or cluster. This
#  draws the ten that every section 6.2 number is averaged over.
# ═══════════════════════════════════════════════════════════════════════════
def fig_aligning_contexts(outdir):
    """The ten evaluated contexts: box start and target, as dots, over their draw regions.

    One context is drawn with the real 0.10 m footprints at their recorded yaw, so the
    dots are anchored to the object they stand for -- the box is of the same order as the
    keep-out disk, and a figure of dots alone would say the opposite.
    """
    from svg.fmpcc_svg import clip_halfplane
    C = getattr(S, 'ALIGNING_CONSTRAINTS', None)
    ctxs = getattr(S, 'ALIGNING_CONTEXTS', None)
    draw = getattr(S, 'ALIGNING_CONTEXT_DRAW', None)
    d = S.D3IL_SCENES.get('aligning')
    if not C or not ctxs or not draw or not d:
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
    f.frame([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [-0.4, -0.2, 0.0, 0.2, 0.4],
            'x [m]', 'y [m]', 'Alignment: the ten evaluated contexts',
            'each episode is one draw of (box start, target) from the two shaded regions',
            xfmt=lambda v: f'{v:g}', yfmt=lambda v: f'{v:g}')
    f.clip_to_box()

    # The constraint set, drawn faintly: it is the same plane, and the keep-out disk sits
    # between every start and every target, which is the point of the task.
    hs = C['halfspace']
    (ax_, ay), (bx, by) = hs['p0'], hs['p1']
    m = (by - ay) / (bx - ax_)
    b = ay - m * ax_
    box = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    poly = clip_halfplane(box, lambda X, Y: Y - (m * X + b))
    if len(poly) > 2:
        f.polygon(poly, '#5d6d7e', opacity=0.16)
    f.dline([(x, m * x + b) for x in (x0 - 0.05, x1 + 0.05)], '#34495e', w=1.4)
    dk = C['disk']
    dcx, dcy = dk['c']
    f.circle(dcx, dcy, dk['r'], fill='#5d6d7e', opacity=0.16, stroke='#34495e', w=1.2)

    # The two draw regions, as the environment defines them.
    for kind, colour in (('box', '#6b5a32'), ('target', '#8a7a55')):
        rx, ry = draw[kind]['x'], draw[kind]['y']
        f.polygon([(rx[0], ry[0]), (rx[1], ry[0]), (rx[1], ry[1]), (rx[0], ry[1])],
                  '#cba872' if kind == 'box' else '#d7cbb4', opacity=0.30,
                  stroke=colour, w=1.3, dash='6,4')

    hb, hr = d['box_half'], d['box_rim_half']

    def footprint(px, py, yaw_deg, half):
        a_ = math.radians(yaw_deg)
        ca, sa = math.cos(a_), math.sin(a_)
        return [(px + dx * ca - dy * sa, py + dx * sa + dy * ca)
                for dx, dy in ((-half, -half), (half, -half), (half, half), (-half, half))]

    shown = getattr(S, 'ALIGNING_CONTEXT_SHOWN', 0) % len(ctxs)

    # Every context as a faint start-to-target link, so the reader sees that the ten
    # pushes are ten different pushes and not one route repeated.
    for c in ctxs:
        f.dline([c['box'], c['target']], '#8a7a55', w=1.0, dash='3,4')

    for i, c in enumerate(ctxs):
        for kind, pos, yaw in (('target', c['target'], c['target_yaw']),
                               ('box', c['box'], c['box_yaw'])):
            px, py = pos
            fill = '#d7cbb4' if kind == 'target' else '#cba872'
            edge = '#8a7a55' if kind == 'target' else '#6b5a32'
            if i == shown:
                # the one context drawn as the object itself, at its recorded yaw
                f.polygon(footprint(px, py, yaw, hr), fill,
                          opacity=0.35 if kind == 'target' else 0.75, stroke=edge, w=1.4)
                f.polygon(footprint(px, py, yaw, hb), fill,
                          opacity=0.55 if kind == 'target' else 0.95, stroke=edge, w=1.6)
            f.marker(f.X(px), f.Y(py), 'o', edge, filled=(kind == 'box'), r=3.4, ew=1.4)

    sx, sy = d['start'][0], d['start'][1]
    f.marker(f.X(sx), f.Y(sy), 'o', '#1b4f72', r=7.0)
    f.text(f.X(sx) + 11, f.Y(sy) + 5, 'end effector', 11, '#111', bold=True)

    c = ctxs[shown]
    f.text(f.X(c['box'][0]), f.Y(c['box'][1] - hr * 1.5) + 15,
           'drawn to scale', 10, '#111', bold=True, anchor='middle')
    f.text(f.X(c['box'][0]), f.Y(c['box'][1] - hr * 1.5) + 28,
           '0.10 m square', 10, '#555', anchor='middle')
    f.end_clip()

    lx, ly = f.X(0.215), f.Y(0.41)
    f.marker(lx, ly, 'o', '#6b5a32', filled=True, r=3.4, ew=1.4)
    f.text(lx + 14, ly + 4, 'box at the start, one per context', 12, '#111')
    ly2 = f.Y(0.355)
    f.marker(lx, ly2, 'o', '#8a7a55', filled=False, r=3.4, ew=1.4)
    f.text(lx + 14, ly2 + 4, 'its target', 12, '#111')
    ly3 = f.Y(0.30)
    f.poly([(lx - 8, ly3), (lx + 9, ly3)], '#6b5a32', dash='6,4', w=2.0)
    f.text(lx + 17, ly3 + 4, 'region the pose is drawn from', 12, '#111')

    path = f.save(os.path.join(outdir, 'fig_aligning_contexts.svg'))
    return path, (f'{len(ctxs)} contexts from sources.ALIGNING_CONTEXTS (recovered from '
                  'analysis_results_checkpoint/15-09/batch_va2_20260915_100754, mf_K20 cell; '
                  'mean box-to-target distance 0.4530 m) | draw ranges '
                  'gym_aligning/envs/aligning.py:62-67 | constraint set '
                  'sources.ALIGNING_CONSTRAINTS | footprint sizes sources.D3IL_SCENES[aligning]')


# ═══════════════════════════════════════════════════════════════════════════
#  The quadrotor's size (v3.33): the numbers of tab:platforms drawn to scale,
#  with the inflation radius the projection adds around every obstacle.
# ═══════════════════════════════════════════════════════════════════════════
def fig_platform_x2_dimensions(outdir):
    G = getattr(S, 'X2_GEOMETRY', None)
    C = getattr(S, 'UAV_CONSTRAINTS', None)
    if not G or not C:
        return None
    fs = 1.35
    r_rot = G['rotor_radius']
    xs = [x for x, _ in G['rotor_xy']]
    ys = [y for _, y in G['rotor_xy']]
    half_x, half_y = max(xs) + r_rot, max(ys) + r_rot        # 0.27 and 0.31 m
    r_inf = C['r_drone']
    lim = r_inf + 0.115
    ml, mr, mt, mb = int(40 * fs), int(16 * fs), int(48 * fs), int(40 * fs)
    w = 620
    pw = w - ml - mr
    f = Fig(w, int(round(pw + mt + mb)), ml=ml, mr=mr, mt=mt, mb=mb, font=fs)
    f.axes((-lim, lim), (-lim, lim))
    f.frame([-0.3, -0.15, 0.0, 0.15, 0.3], [-0.3, -0.15, 0.0, 0.15, 0.3],
            'x [m]', 'y [m]', 'Skydio X2, seen from above',
            f"rotor disks, body, and the {r_inf:g} m radius the projection inflates obstacles by",
            xfmt=lambda v: f'{v:g}', yfmt=lambda v: f'{v:g}')
    f.clip_to_box()
    # the radius every obstacle is grown by before the solver sees it
    f.circle(0, 0, r_inf, fill='#5d6d7e', opacity=0.12, stroke='#34495e', w=1.6, dash='8,5')
    # collision footprint
    f.polygon([(-half_x, -half_y), (half_x, -half_y), (half_x, half_y), (-half_x, half_y)],
              'none', stroke='#95a5a6', w=1.4, dash='3,4')
    for cx, cy in G['rotor_xy']:
        f.circle(cx, cy, r_rot, fill='#9aa0a6', opacity=0.5, stroke='#5d6d7e', w=1.4)
    ax, by = G['body_semi_axes']
    f.polygon([(-ax, -by), (ax, -by), (ax, by), (-ax, by)], '#5d6d7e', opacity=0.85,
              stroke='#34495e', w=1.2)
    # one rotor arm, as the distance the table quotes
    f.dline([(0, 0), G['rotor_xy'][2]], '#c0392b', w=2.0)
    f.marker(f.X(0), f.Y(0), 'o', '#c0392b', r=4.0)

    def dim(p0, p1, label, off, vertical=False):
        """A dimension line with end ticks, offset from the shape it measures."""
        (x0_, y0_), (x1_, y1_) = p0, p1
        if vertical:
            x = x0_ + off
            f.dline([(x, y0_), (x, y1_)], '#111', w=1.2)
            for y in (y0_, y1_):
                f.dline([(x - 0.012, y), (x + 0.012, y)], '#111', w=1.2)
            f.text(f.X(x) - 10, f.Y(0.5 * (y0_ + y1_)), label, 12, '#111', anchor='end', bold=True)
        else:
            y = y0_ + off
            f.dline([(x0_, y), (x1_, y)], '#111', w=1.2)
            for x in (x0_, x1_):
                f.dline([(x, y - 0.012), (x, y + 0.012)], '#111', w=1.2)
            f.text(f.X(0.5 * (x0_ + x1_)), f.Y(y) - 8, label, 12, '#111',
                   anchor='middle', bold=True)

    dim((-half_x, half_y), (half_x, half_y), f'{2 * half_x:.2f} m', 0.055)
    dim((half_x, -half_y), (half_x, half_y), f'{2 * half_y:.2f} m', 0.065, vertical=True)
    d = (G['rotor_xy'][2][0] ** 2 + G['rotor_xy'][2][1] ** 2) ** 0.5
    f.text(f.X(0.055), f.Y(0.10), f'{d:.3f} m', 12, '#c0392b', bold=True)
    f.text(f.X(-lim + 0.02), f.Y(-lim + 0.035), f"mass {G['mass_kg']:g} kg", 12, '#111')
    f.end_clip()
    path = f.save(os.path.join(outdir, 'fig_platform_x2_dimensions.svg'))
    return path, ('d3il/.../quadrotor/quadrotor_modified.xml rotor and body geoms | transcribed in '
                  'sources.X2_GEOMETRY; inflation radius from sources.UAV_CONSTRAINTS')


ALL = [
    ('fig_scene_avoiding', 'env', fig_scene_avoiding),
    ('fig_scene_aligning', 'env', fig_scene_aligning),
    ('fig_scene_uav_corridor', 'env', fig_scene_uav_corridor),
    ('fig_scene_uav_pillars', 'env', fig_scene_uav_pillars),
    ('fig_scene_uav_scurve', 'env', fig_scene_uav_scurve),
    ('fig_constraints_uav', 'env', fig_constraints_uav),
    ('fig_constraints_aligning', 'env', fig_constraints_aligning),
    ('fig_aligning_contexts', 'env', fig_aligning_contexts),
    ('fig_platform_x2_dimensions', 'env', fig_platform_x2_dimensions),
]
