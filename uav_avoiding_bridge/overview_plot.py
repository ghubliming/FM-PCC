#!/usr/bin/env python
"""Constraint overview of `pillars_v2` (Gen15 U18) — the static scene, no MuJoCo needed.

For each avoiding geometry (top-left-hard, top-right-hard, both-hard) two panels:
  left   the avoiding frame as the planner sees it: field, six physical cylinders, the geometry's halfspace(s)
         (forbidden side shaded), its ONE keep-out disk (+ tightened ring), start, finish line
  right  the world frame the drone flies: the same objects mapped by frame.to_world_xy at SCALE, physical pillars
         with the rotor-reach ring (+0.31 m), the drone footprint at the start for scale, the arena box

    python uav_avoiding_bridge/overview_plot.py [--out DIR] [--scale S]

`draw_avoiding(ax, geo)` / `draw_world(ax, geo, scale)` are also what turbo.py's world PNG uses.
"""
import argparse
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)
from uav_avoiding_bridge import frame as F                                   # noqa: E402
from uav_avoiding_bridge.scoring import geometry_for, load_projection_cfg   # noqa: E402

DRONE_REACH = F.DRONE_REACH_M     # m, radial rotor reach (0.36)
GEOS = ('top-left-hard', 'top-right-hard', 'both-hard')
C_PILLAR, C_CONS, C_FINISH = '#d98c40', '#1f4fd8', '#2ca02c'


def _clip_rect_by_halfspace(rect, c):
    """Sutherland-Hodgman: keep the FORBIDDEN part of the axis-aligned rect (xlim, ylim) for halfspace c."""
    (x0, x1), (y0, y1) = rect
    p0, p1 = np.asarray(c[0], float), np.asarray(c[1], float)
    m = (p1[1] - p0[1]) / (p1[0] - p0[0])
    sign = 1.0 if c[2] == 'below' else -1.0            # forbidden: sign * (y - line(x)) >= 0
    f = lambda q: sign * (q[1] - (p0[1] + m * (q[0] - p0[0])))
    poly = [np.array(v, float) for v in ((x0, y0), (x1, y0), (x1, y1), (x0, y1))]
    out = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        fa, fb = f(a), f(b)
        if fa >= 0:
            out.append(a)
        if (fa >= 0) != (fb >= 0):
            t = fa / (fa - fb)
            out.append(a + t * (b - a))
    return out


def _halfspace_shade(ax, c, to_xy, xlim, ylim):
    """fill the FORBIDDEN side of halfspace c (avoiding-frame line) inside the field, mapped with to_xy."""
    from matplotlib.patches import Polygon
    poly = _clip_rect_by_halfspace((xlim, ylim), c)
    if len(poly) >= 3:
        ax.add_patch(Polygon(np.array([to_xy(*q) for q in poly]), closed=True, facecolor=C_CONS, edgecolor=C_CONS,
                             alpha=0.14, lw=0, hatch='//', zorder=1))
    p0, p1 = np.asarray(c[0], float), np.asarray(c[1], float)
    m = (p1[1] - p0[1]) / (p1[0] - p0[0])
    xs = np.linspace(xlim[0], xlim[1], 60)
    line = np.array([to_xy(x, p0[1] + m * (x - p0[0])) for x in xs])
    ax.plot(line[:, 0], line[:, 1], color=C_CONS, lw=2, alpha=0.8, zorder=2)


def draw_avoiding(ax, geo, drone_scale=None):
    from matplotlib.patches import Circle, Rectangle
    (x0, x1), (y0, y1) = F.FIELD_X_A, F.FIELD_Y_A
    ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls=':', color='k', lw=1))
    for c in geo['polytopic']:
        _halfspace_shade(ax, c, lambda x, y: (x, y), (x0 - 0.05, x1 + 0.05), (y0 - 0.05, y1 + 0.05))
    for ob in geo['obstacles']:
        ax.add_patch(Circle(ob['center'], ob['radius'], color=C_CONS, alpha=0.25, zorder=2))
        ax.add_patch(Circle(ob['center'], ob['radius'] + geo['enlarge'], fill=False, ls='--', color=C_CONS, alpha=0.7, zorder=2))
    for n, xa, ya, r in F.OBSTACLES_A:
        ax.add_patch(Circle((xa, ya), r, color=C_PILLAR, zorder=3))
        ax.annotate(n.replace('_obs', ''), (xa, ya), (0, 9), textcoords='offset points', ha='center', fontsize=6)
    ax.axhline(F.GOAL_Y_A, color=C_FINISH, lw=2, alpha=0.7)
    ax.plot(*F.START_A, 'ko', ms=6, zorder=5)
    if drone_scale:
        ax.add_patch(Circle(F.START_A, DRONE_REACH / drone_scale, fill=False, color='k', lw=1.2, zorder=5))
    ax.set_xlim(x0 - 0.08, x1 + 0.08); ax.set_ylim(y0 - 0.08, y1 + 0.08); ax.set_aspect('equal')
    ax.set_xlabel('$x_a$'); ax.set_ylabel('$y_a$  (progress ↑)')
    _legend(ax, loc='lower left')


def draw_world(ax, geo, scale=None, drone_at_start=True, arena=True, legend=True):
    from matplotlib.patches import Circle, Rectangle
    s = F.SCALE if scale is None else scale
    to = lambda x, y: F.to_world_xy(x, y, s)
    (x0, x1), (y0, y1) = F.world_field(s)
    if arena:
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ls=':', color='k', lw=1))
    for c in geo['polytopic']:
        _halfspace_shade(ax, c, to, (F.FIELD_X_A[0] - 0.05, F.FIELD_X_A[1] + 0.05), (F.FIELD_Y_A[0] - 0.05, F.FIELD_Y_A[1] + 0.05))
    for ob in geo['obstacles']:
        X, Y = to(*ob['center'])
        ax.add_patch(Circle((X, Y), s * ob['radius'], color=C_CONS, alpha=0.25, zorder=2))
        ax.add_patch(Circle((X, Y), s * (ob['radius'] + geo['enlarge']), fill=False, ls='--', color=C_CONS, alpha=0.7, zorder=2))
    for n, xa, ya, r in F.OBSTACLES_A:
        X, Y = to(xa, ya)
        ax.add_patch(Circle((X, Y), s * r + DRONE_REACH, color=C_PILLAR, alpha=0.18, zorder=2))   # + rotor reach
        ax.add_patch(Circle((X, Y), s * r, color=C_PILLAR, zorder=3))
    fx = to(0.5, F.GOAL_Y_A)[0]
    ax.axvline(fx, color=C_FINISH, lw=2, alpha=0.7)
    sx, sy = to(*F.START_A)
    ax.plot(sx, sy, 'ko', ms=5, zorder=5)
    if drone_at_start:
        ax.add_patch(Circle((sx, sy), DRONE_REACH, fill=False, color='k', lw=1.2, zorder=5))
    ax.set_xlim(x0 - 0.6, x1 + 0.6); ax.set_ylim(y0 - 0.6, y1 + 0.6); ax.set_aspect('equal')
    ax.set_xlabel('X [m]  (= avoiding $+y_a$, progress →)'); ax.set_ylabel('Y [m]  (= avoiding $-x_a$)')
    if legend:
        _legend(ax, world=True)


def _legend(ax, world=False, loc='lower right'):
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h = [Patch(color=C_PILLAR, label='physical pillar (MuJoCo collision)'),
         Patch(color=C_CONS, alpha=0.25, label='planner keep-out disk (this geometry)'),
         Line2D([], [], color=C_CONS, ls='--', label='keep-out tightened (+0.025)'),
         Patch(facecolor=C_CONS, edgecolor=C_CONS, alpha=0.14, hatch='//', label='halfspace: forbidden side (hatched)'),
         Line2D([], [], color=C_FINISH, lw=2, label='finish line (success: cross it)'),
         Line2D([], [], color='k', marker='o', ls='', label='start'),
         Line2D([], [], color='k', marker='o', mfc='none', ls='', label='drone footprint (rotor reach 0.31 m)')]
    if world:
        h.insert(1, Patch(color=C_PILLAR, alpha=0.18, label='pillar + rotor reach (contact if centre enters)'))
    ax.legend(handles=h, loc=loc, fontsize=7, framealpha=0.9)


def _numbers(scale):
    s = scale
    gap = 0.15 - 2 * 0.025
    return (f'scale {s:g}: arena {s * 0.6:.1f} × {s * 0.7:.1f} m · pillars r = {s * 0.025:.2f} / {s * 0.03:.2f} m · '
            f'keep-out disk r = {s * 0.08:.2f} m (tightened +{s * 0.025:.2f}) · drone radial reach {DRONE_REACH} m = rod radius × {DRONE_REACH / s / 0.01:.2f}\n'
            f'row-2/3 opening {s * gap:.2f} m · Panda paths hug obstacles at 0.01–0.02 units → {s * 0.01:.2f}–{s * 0.02:.2f} m in the world · '
            f'action bound {s * 0.01:.2f}–{s * 0.012:.2f} m/step → clock mode tracks a reference rate-limited to 0.6 m/s')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'logs_in_develop', 'Gen15', 'U18', 'figs'))
    ap.add_argument('--scale', type=float, default=None)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = F.SCALE if args.scale is None else args.scale
    cfg = load_projection_cfg()
    os.makedirs(args.out, exist_ok=True)
    tag = F.scene_tag(s)

    # one figure per geometry (avoiding | world)
    for g in GEOS:
        geo = geometry_for(g, cfg)
        fig, (a, b) = plt.subplots(1, 2, figsize=(15, 7), gridspec_kw={'width_ratios': [1, 1.35]})
        draw_avoiding(a, geo, drone_scale=s); a.set_title(f'avoiding frame — {g}\n(what planner, projector and scorer see)', fontsize=10)
        draw_world(b, geo, s); b.set_title(f'world frame — pillars_v2 @ {tag}\n(what the drone flies; orange halo = pillar + rotor reach, '
                                           f'black ring at start = drone footprint)', fontsize=10)
        fig.suptitle(_numbers(s), fontsize=9, y=0.995)
        p = os.path.join(args.out, f'constraint_overview_{tag}_{g}.png')
        fig.savefig(p, dpi=130, bbox_inches='tight'); plt.close(fig); print('wrote', p)

    # all three geometries, world frame, side by side + legend panel
    fig, axes = plt.subplots(1, 3, figsize=(21, 6.5))
    for ax, g in zip(axes, GEOS):
        draw_world(ax, geometry_for(g, cfg), s, legend=(g == GEOS[-1])); ax.set_title(g, fontsize=11)
    fig.suptitle(f'pillars_v2 @ {tag}: the three test-time geometries in the world frame\n' + _numbers(s), fontsize=9)
    p = os.path.join(args.out, f'constraint_overview_{tag}_all_geometries_world.png')
    fig.savefig(p, dpi=120, bbox_inches='tight'); plt.close(fig); print('wrote', p)

    # scale ladder: both-hard at 3, 6, 8, 10, 12 — the "does the drone fit" picture
    ladder = (3, 10, 20, 36, 45)
    fig, axes = plt.subplots(1, len(ladder), figsize=(4.2 * len(ladder), 5))
    geo = geometry_for('both-hard', cfg)
    for ax, sc in zip(axes, ladder):
        draw_world(ax, geo, sc, arena=False, legend=False)
        hug = sc * 0.012 - DRONE_REACH        # a path hugging at 0.012 units (rod radius + 2 mm): slack in the world
        ax.set_title(f'scale {sc}: hugging-path slack {hug:+.2f} m', fontsize=10, color=('r' if hug < 0.0 else 'k'))
        ax.set_xlabel(''); ax.set_ylabel('')
    fig.suptitle('scale ladder (both-hard): orange halo = pillar + 0.36 m radial rotor reach. The Panda paths pass the obstacles at the rod radius '
                 '(0.01-0.02 units), so the faithful scale maps 0.01 onto 0.36: x36', fontsize=10)
    p = os.path.join(args.out, 'constraint_overview_scale_ladder_world.png')
    fig.savefig(p, dpi=110, bbox_inches='tight'); plt.close(fig); print('wrote', p)


if __name__ == '__main__':
    main()
