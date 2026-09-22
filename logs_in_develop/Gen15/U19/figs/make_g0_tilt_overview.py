#!/usr/bin/env python3.14
"""Gen15 U19 · gate G0 (offline, no GPU) — the corridor_v3_tilt constraint: the v2 slide leaned over.

    python3.14 logs_in_develop/Gen15/U19/figs/make_g0_tilt_overview.py

Reads `config/uav_projection.yaml` (corridor_v3_tilt, corridor_v2_slide), no MuJoCo, no torch. Writes
fig_u19_g0_tilt_overview.{png,svg} next to this file and prints the G0 numbers. Same colour convention as the
eval's constraint_overview (steelblue box, darkorange halfspace, crimson drone-centre limit, tomato obstacle).

Geometry: feasible signed distance s(p) = s_xy(x, y) + t·(z − z_ref), t = tan(deg) < 0 → descending gains room.
The body clears the plane when s / √(1+t²) ≥ r_drone.
"""
import os

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpa
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))

FLOWN_LO, FLOWN_HI, LAUNCH_Z = 0.86, 1.24, 1.11         # v2 slide flights (v3.65 changelog §1)
ROUTES = {'L': -0.12, 'C': 0.0, 'R': +0.12}
ROUTE_X = (-2.8, 2.8)
X_EXIT = 2.0

cfg = yaml.safe_load(open(os.path.join(REPO, 'config/uav_projection.yaml')))
geo = {g['name']: g for g in cfg['geo_constraint_variants']}
ent = geo['corridor_v3_tilt']
r = float(ent['planning_inflation']['r_drone'])
tight = float(cfg.get('enlarge_constraints') or 0.0)
z_lb, z_ub = float(ent['workspace_bounds']['lb'][2]), float(ent['workspace_bounds']['ub'][2])
z_floor = z_lb + r
walls = [h for h in ent['halfspace_constraints'] if h.get('z_lean') is None]
slide = next(h for h in ent['halfspace_constraints'] if h.get('z_lean') is not None)
(x1, y1), (x2, y2) = slide['line']
deg, z_ref = float(slide['z_lean']['deg']), float(slide['z_lean']['z_ref'])
t = np.tan(np.radians(deg)); k = np.sqrt(1 + t * t)
d = np.array([x2 - x1, y2 - y1]); d /= np.linalg.norm(d)
n_xy = np.array([-d[1], d[0]]) * (1.0 if slide['side'] == 'above' else -1.0)   # unit, into the feasible side
p0 = np.array([x1, y1])
slope = (y2 - y1) / (x2 - x1)


def s_xy(x, y):
    return n_xy[0] * (x - p0[0]) + n_xy[1] * (y - p0[1])


def y_cut(x, z, margin=0.0):
    """y of the cut of the (margin-offset) plane at altitude z: s_xy + t(z − z_ref) = margin·k."""
    s_need = margin * k - t * (z - z_ref)
    # s_xy(x, y) = s_need  → y = p0y + (s_need − n_x (x − p0x)) / n_y
    return p0[1] + (s_need - n_xy[0] * (x - p0[0])) / n_xy[1]


def z_cut(x, y, margin=0.0):
    """z of the cut of the (margin-offset) plane along y: s_xy + t(z − z_ref) = margin·k."""
    return z_ref + (margin * k - s_xy(x, y)) / t


# ── the demands, per route, at the exit ────────────────────────────────────
rows = []
for name, y in ROUTES.items():
    s = s_xy(X_EXIT, y)
    deficit = r - s / k                                   # perpendicular, planning margin
    dz_only = (r * k - s) / abs(t)                        # descend only
    dy_only = (r * k - s)                                 # lateral only (perpendicular in x-y ≈ y for the 14° line)
    y_lat_only = y - dy_only / abs(n_xy[1])
    lat_feasible = y_lat_only >= -(0.95 - r)
    mn_xy = deficit * (1.0 / k); mn_z = deficit * abs(t) / k
    x_bind = None
    xs = np.linspace(-2.8, X_EXIT, 2000)
    viol = s_xy(xs, y) < r * k
    if viol.any():
        x_bind = float(xs[viol][0])
    rows.append((name, y, s, dz_only, y_lat_only, lat_feasible, mn_xy, mn_z, x_bind))
room = LAUNCH_Z - z_floor
g0 = all(rw[3] <= room for rw in rows) and all(rw[7] >= 0.15 for rw in rows)

# ── figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15.5, 6.6))
gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.2)
ay = fig.add_subplot(gs[0, 0]); ax = fig.add_subplot(gs[0, 1])
for a in (ax, ay):
    a.grid(True, ls=':', lw=0.6, color='0.82'); a.tick_params(labelsize=8)
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)

# — top-down: cuts of the leaned plane at three altitudes —
ay.set_title('top-down (x, y): the leaned slide, cut at three altitudes', fontsize=10, loc='left')
ay.set_aspect('equal')
for h in walls:
    (wx1, wy1), (wx2, wy2) = h['line']
    ay.plot([wx1, wx2], [wy1, wy2], color='darkorange', lw=2.0, zorder=5)
    sg = 1.0 if h['side'] == 'above' else -1.0
    ay.plot([wx1, wx2], [wy1 + sg * r, wy2 + sg * r], color='crimson', lw=0.9, ls='--', zorder=5, alpha=0.7)
for ob in ent['obstacle_constraints']:
    cx, cy = ob['center']
    ay.add_patch(mpa.Circle((cx, cy), ob['radius'] + r, color='tomato', alpha=0.25, lw=0, zorder=4))
xs = np.linspace(x1, x2, 80)
ay.plot(xs, y_cut(xs, z_ref), color='darkorange', lw=2.2, zorder=6)                       # raw at z_ref = the v2 slide
ay.plot(xs, y_cut(xs, z_ref, r), color='crimson', lw=1.5, ls='--', zorder=7)            # centre limit at z_ref
ay.fill_between(xs, y_cut(xs, z_ref), y_cut(xs, z_ref, r), color='crimson', alpha=0.12, lw=0, zorder=2)
v2 = geo['corridor_v2_slide']['halfspace_constraints'][2]
(vx1, vy1), (vx2, vy2) = v2['line']
ay.plot(xs, np.interp(xs, [vx1, vx2], [vy1, vy2]) - r * np.hypot(1.0, slope), color='0.45', lw=1.0, ls=(0, (3, 2)),
        zorder=6)                                                                           # the v2 centre limit, for reference
ay.plot(xs, y_cut(xs, z_floor, r), color='crimson', lw=1.2, ls=':', zorder=7)            # centre limit at the floor
ay.plot(xs, y_cut(xs, (z_ref + z_floor) / 2, r), color='crimson', lw=0.9, ls=':', zorder=7, alpha=0.7)
for name, y in ROUTES.items():
    ay.plot(ROUTE_X, [y, y], color='0.45', lw=0.9, alpha=0.8, zorder=3)
    ay.text(ROUTE_X[0] - 0.08, y, name, fontsize=7, color='0.35', ha='right', va='center')
ay.plot([ROUTE_X[1]] * 2, [-0.95, 0.95], color='seagreen', lw=1.2, ls='-.', zorder=3)
ay.text(1.05, y_cut(1.05, z_ref, r) - 0.05, f'centre limit at z = {z_ref:.2f} (launch):\n{r * k:.2f} m from the line',
        fontsize=6.5, color='crimson', ha='left', va='top')
ay.text(-1.9, y_cut(-1.9, z_floor, r) + 0.04, f'centre limit at z = {z_floor:.2f} (floor)', fontsize=6.5, color='crimson')
ay.text(-0.3, y_cut(-0.3, (z_ref + z_floor) / 2, r) + 0.04, f'at z = {(z_ref + z_floor) / 2:.2f}', fontsize=6, color='crimson', alpha=0.8)
ay.set_xlim(-3.2, 3.2); ay.set_ylim(-1.35, 1.35)
ay.set_xlabel('x (m)', fontsize=8); ay.set_ylabel('y (m)', fontsize=8)
ay.legend(handles=[
    Line2D([0], [0], color='darkorange', lw=2.2, label=f'slide, raw cut at z_ref = {z_ref:.2f} m (= the v2 line)'),
    Line2D([0], [0], color='crimson', lw=1.5, ls='--', label=f'drone-centre limit at z_ref (r·√(1+t²) = {r * k:.2f} m ⊥)'),
    Line2D([0], [0], color='crimson', lw=1.2, ls=':', label='drone-centre limit lower down (room grows)'),
    Line2D([0], [0], color='0.45', lw=1.0, ls=(0, (3, 2)), label=f'v2 slide centre limit (vertical wall, {r:.2f} m)'),
    Line2D([0], [0], color='darkorange', lw=2.0, label='corridor_v2 wall (inner face)'),
], fontsize=6.5, loc='lower left', framealpha=0.95)

# — side view along each route —
ax.set_title(f'side view (x, z): the plane cut along each route, lean {deg:+.0f}° about z_ref', fontsize=10, loc='left')
ax.axhspan(FLOWN_LO, FLOWN_HI, color='0.55', alpha=0.18, lw=0, zorder=1)
ax.text(-1.0, FLOWN_LO + 0.02, f'flown band, v2 (no projection) {FLOWN_LO:.2f}–{FLOWN_HI:.2f} m', fontsize=7, color='0.35', va='bottom')
ax.axhline(LAUNCH_Z, color='0.35', lw=1.0, ls='--', zorder=2)
ax.text(-2.95, LAUNCH_Z + 0.02, f'launch = z_ref {LAUNCH_Z:.2f} m', fontsize=7, color='0.35')
ax.axhline(z_ub, color='steelblue', lw=1.4); ax.axhline(z_ub - r, color='steelblue', lw=1.2, ls='--')
ax.axhline(z_lb, color='steelblue', lw=1.4); ax.axhline(z_floor, color='steelblue', lw=1.2, ls='--')
ax.text(-2.95, z_ub + 0.02, f'ceiling {z_ub:.2f} m (unchanged from v2)', fontsize=7, color='steelblue')
ax.text(-0.6, z_floor + 0.02, f'floor + r_drone = {z_floor:.2f} m (drone centre) → room below launch {room:.2f} m',
        fontsize=7, color='steelblue')
styles = {'L': (0, (4, 2)), 'C': 'solid', 'R': (0, (1, 1.5))}
xs = np.linspace(x1, x2, 120)
for name, y in ROUTES.items():
    zr_ = z_cut(xs, y); ze = z_cut(xs, y, r); zt = z_cut(xs, y, r + tight)
    m = (zr_ > z_lb - 0.05) & (zr_ < z_ub + 0.3)
    ax.plot(xs[m], zr_[m], color='darkorange', lw=2.0 if name == 'C' else 1.2, ls=styles[name], zorder=5)
    me = (ze > z_lb - 0.05) & (ze < z_ub + 0.3)
    ax.plot(xs[me], ze[me], color='crimson', lw=1.4 if name == 'C' else 1.0, ls=styles[name], zorder=6)
    if name == 'C':
        ax.fill_between(xs[me], ze[me], np.minimum(zr_[me], z_ub + 0.3), color='crimson', alpha=0.12, lw=0, zorder=1)
        ax.plot(xs[me], zt[me], color='crimson', lw=0.7, ls=':', zorder=6)
for kk, (name, y, s, dz_only, y_lat, lat_ok, mn_xy, mn_z, x_bind) in enumerate(rows):
    if x_bind is not None:
        ax.plot([x_bind], [LAUNCH_Z], 's', color='crimson', ms=4, zorder=9)
        ax.annotate(f'{name} binds from x = {x_bind:.2f}', xy=(x_bind, LAUNCH_Z), xytext=(-2.6 + 0.0 * kk, 0.78 - 0.07 * kk),
                    fontsize=6.5, color='crimson', arrowprops=dict(arrowstyle='-', color='crimson', lw=0.5, alpha=0.7), zorder=9)
zC = z_cut(X_EXIT, 0.0, r)
ax.annotate('', xy=(X_EXIT, zC), xytext=(X_EXIT, LAUNCH_Z), arrowprops=dict(arrowstyle='<->', color='black', lw=1.2), zorder=8)
ax.text(X_EXIT + 0.06, (zC + LAUNCH_Z) / 2, f'route C, descend only:\n{LAUNCH_Z - zC:.2f} m at the exit', fontsize=7, va='center')
ax.axvline(ROUTE_X[1], color='seagreen', lw=1.2, ls='-.', zorder=2)
ax.text(ROUTE_X[1] - 0.04, z_ub - 0.05, 'route end x = 2.8\n(altitude does not\naffect success)', fontsize=6.5, color='seagreen', ha='right', va='top')
ax.set_xlim(-3.05, 3.05); ax.set_ylim(0.0, z_ub + 0.35)
ax.set_xlabel('x (m)', fontsize=8); ax.set_ylabel('z (m)', fontsize=8)
ax.legend(handles=[
    Line2D([0], [0], color='darkorange', lw=2.0, label='plane, raw cut along route C (y = 0)'),
    Line2D([0], [0], color='crimson', lw=1.4, label=f'drone-centre limit along C (r_drone {r:.2f} m ⊥ to the plane)'),
    Line2D([0], [0], color='crimson', lw=0.7, ls=':', label=f'-tightened (+{tight:.3f} m ⊥)'),
    Line2D([0], [0], color='crimson', lw=1.0, ls=(0, (4, 2)), label='same along route L (y = −0.12)'),
    Line2D([0], [0], color='crimson', lw=1.0, ls=(0, (1, 1.5)), label='same along route R (y = +0.12)'),
    Line2D([0], [0], color='steelblue', lw=1.4, label='workspace box z (raw / − r_drone dashed)'),
    mpa.Patch(color='0.55', alpha=0.18, label='flown band of the v2 slide flights'),
], fontsize=6.5, loc='lower left', bbox_to_anchor=(0.0, 0.02), framealpha=0.95)

# — the demand table —
lines = [f'{"route":6s}{"s_xy":>7s}  {"descend only":>13s}  {"sideways only":>20s}  {"min-norm Δxy / Δz":>19s}',
         f'{"(at the exit x = 2.0)":6s}']
lines = [lines[0]]
for name, y, s, dz_only, y_lat, lat_ok, mn_xy, mn_z, x_bind in rows:
    lat = f'y → {y_lat:+.2f}' + ('' if lat_ok else ' ✗ wall −0.64')
    lines.append(f'{name:6s}{s:+7.3f}  {dz_only:11.2f} m  {lat:>20s}  {mn_xy:9.2f} / {mn_z:.2f} m')
lines.append('demands at the exit, x = 2.0, from the launch altitude')
ax.text(0.99, 0.02, '\n'.join(lines), transform=ax.transAxes, fontsize=6.8, family='monospace', ha='right', va='bottom',
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.7', alpha=0.95), zorder=10)

fig.suptitle(f'Gen15 U19 · gate G0 — corridor_v3_tilt: the v2 slide leaned {deg:+.0f}° (t = {t:.3f}) about z_ref = {z_ref:.2f} m   '
             f'descend-only ≤ room {room:.2f} m on every route: {"yes" if all(rw[3] <= room for rw in rows) else "NO"};   '
             f'min-norm Δz ≥ 0.15 m on every route: {"yes" if all(rw[7] >= 0.15 for rw in rows) else "NO"}   →   G0 {"PASS" if g0 else "FAIL"}',
             fontsize=9.5, fontweight='bold', y=0.995)
fig.text(0.5, 0.005,
         f'corridor_v2_slide plus one key on the slide entry (z_lean). Scene XML, walls, caps, workspace box, model, checkpoint, routes, '
         f'goals and the projector stack are byte-identical to v2. At z_ref the cut is the v2 line, but the body must clear the LEANED '
         f'plane, so the lateral centre limit there is r·√(1+t²) = {r * k:.2f} m; each metre of descent buys |t| = {abs(t):.2f} m of s_xy. '
         f'The min-norm correction moves along the plane normal: {abs(t) / k * 100:.0f} % of it in z, {100 / k:.0f} % in x–y.',
         ha='center', fontsize=7, color='dimgray', style='italic', wrap=True)
fig.subplots_adjust(left=0.05, right=0.99, top=0.91, bottom=0.14)
out = os.path.join(HERE, 'fig_u19_g0_tilt_overview')
fig.savefig(out + '.png', dpi=150, bbox_inches='tight'); fig.savefig(out + '.svg', bbox_inches='tight')
plt.close(fig)

print(f'lean {deg:+.0f}°  t={t:.4f}  k=√(1+t²)={k:.3f}  n_xy={np.round(n_xy, 4).tolist()}  lateral centre limit at z_ref {r * k:.3f} m')
print(f'floor centre limit {z_floor:.2f}  room below launch {room:.2f} m')
for name, y, s, dz_only, y_lat, lat_ok, mn_xy, mn_z, x_bind in rows:
    print(f'route {name} (y {y:+.2f}): s_xy(exit) {s:+.3f}  descend-only {dz_only:.3f} m  sideways-only y {y_lat:+.3f} '
          f'{"ok" if lat_ok else "INFEASIBLE (wall limit -0.64)"}  min-norm Δxy {mn_xy:.3f} / Δz {mn_z:.3f} m  binds from x {x_bind}')
print(f'G0: {"PASS" if g0 else "FAIL"}   -> {out}.png / .svg')
