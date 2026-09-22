#!/usr/bin/env python3.14
"""Gen15 U19 · gate G0 (offline, no GPU) — the corridor_v3_ablation_hump constraint, drawn from the yaml.

    python3.14 logs_in_develop/Gen15/U19/figs/make_g0_ablation_hump_overview.py

Reads `config/uav_projection.yaml` (entries corridor_v3_ablation_hump / corridor_v3_ablation_hump_lo), no MuJoCo, no
torch. Writes fig_u19_g0_ablation_hump_overview.{png,svg} next to this file and prints the G0 numbers.
Colours follow the eval's own constraint_overview convention (steelblue = workspace box,
darkorange = halfspace, crimson = drone-centre limit, tomato = obstacle).
"""
import os
import sys

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpa
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, '..', '..', '..', '..'))

# measured on the v2 slide flights (v3.65 changelog §1; 96 flights of 4 paired cells)
FLOWN_LO, FLOWN_HI, LAUNCH_Z = 0.86, 1.24, 1.11
ROUTES = {'L': -0.12, 'C': 0.0, 'R': +0.12}
ROUTE_X = (-2.8, 2.8)
ENVELOPE_Z_UB, DIV_SLACK = 1.30, 2.0          # SCENE_FLIGHT_ENVELOPE['corridor'], DIV_ENVELOPE_SLACK_M

cfg = yaml.safe_load(open(os.path.join(REPO, 'config/uav_projection.yaml')))
geo = {g['name']: g for g in cfg['geo_constraint_variants']}
hump, hump_lo = geo['corridor_v3_ablation_hump'], geo['corridor_v3_ablation_hump_lo']
r_plan = float(hump['planning_inflation']['r_drone'])
r_score = float(cfg['inflation']['r_drone'])
tight = float(cfg.get('enlarge_constraints') or 0.0)
z_lb, z_ub = float(hump['workspace_bounds']['lb'][2]), float(hump['workspace_bounds']['ub'][2])
z_ub_v2 = float(geo['corridor_v2_slide']['workspace_bounds']['ub'][2])


def roof(entry):
    """(xs, raw z) over the x_active windows of the entry's `plane: xz` halfspaces, plus slope."""
    pts, slopes = [], []
    for hs in entry['halfspace_constraints']:
        if hs.get('plane') != 'xz':
            continue
        (x1, z1), (x2, z2) = hs['line']
        lo, hi = hs['x_active']
        xs = np.linspace(lo, hi, 50)
        s = (z2 - z1) / (x2 - x1)
        pts.append((xs, z1 + s * (xs - x1))); slopes.append(abs(s))
    return pts, slopes


pts, slopes = roof(hump)
pts_lo, _ = roof(hump_lo)
s = slopes[0]
H = max(float(p[1].max()) for p in pts)
H_lo = max(float(p[1].max()) for p in pts_lo)
off_plan = r_plan * np.hypot(1.0, s)               # perpendicular margin, measured vertically
off_tight = (r_plan + tight) * np.hypot(1.0, s)
off_score = r_score * np.hypot(1.0, s)
peak_req, peak_req_t = H + off_plan, H + off_tight
ceil_req = z_ub - r_plan
slot = ceil_req - peak_req_t
slot_v2 = (z_ub_v2 - r_plan) - peak_req_t
climb = peak_req_t - LAUNCH_Z
x_bind = 1.5 * (1 - (LAUNCH_Z - off_plan) / H)       # |x| where the enforced roof passes the launch altitude

# ── figure ──────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(15, 6.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1.0], wspace=0.22)
ax = fig.add_subplot(gs[0, 0]); ay = fig.add_subplot(gs[0, 1])
for a in (ax, ay):
    a.grid(True, ls=':', lw=0.6, color='0.82'); a.tick_params(labelsize=8)
    for sp in ('top', 'right'):
        a.spines[sp].set_visible(False)

# — side view (x, z) —
ax.set_title('corridor_v3_ablation_hump — side view (x, z): the constraint the plan must climb over', fontsize=10, loc='left')
ax.axhspan(FLOWN_LO, FLOWN_HI, color='0.55', alpha=0.18, lw=0, zorder=1)
ax.text(-2.95, (FLOWN_LO + FLOWN_HI) / 2, f'flown band, v2 slide (no projection)\n{FLOWN_LO:.2f}–{FLOWN_HI:.2f} m',
        fontsize=7, color='0.35', va='center')
ax.axhline(LAUNCH_Z, color='0.35', lw=1.0, ls='--', zorder=2)
ax.text(2.95, LAUNCH_Z + 0.02, f'launch {LAUNCH_Z:.2f} m', fontsize=7, color='0.35', ha='right')
# workspace box (raw / inflated)
ax.axhline(z_ub, color='steelblue', lw=1.4, zorder=3)
ax.axhline(ceil_req, color='steelblue', lw=1.2, ls='--', zorder=3)
ax.axhline(z_lb, color='steelblue', lw=1.4, zorder=3)
ax.axhline(z_lb + r_plan, color='steelblue', lw=1.2, ls='--', zorder=3)
ax.text(-2.95, z_ub + 0.03, f'ceiling {z_ub:.2f} m (raw, synthetic: no ceiling geom)', fontsize=7, color='steelblue')
ax.text(-2.95, ceil_req - 0.09, f'ceiling − r_drone = {ceil_req:.2f} m (drone centre)', fontsize=7, color='steelblue')
ax.axhline(z_ub_v2 - r_plan, color='steelblue', lw=0.9, ls=':', zorder=3)
ax.text(2.95, z_ub_v2 - r_plan - 0.03, f'v2 ceiling − r_drone = {z_ub_v2 - r_plan:.2f} m  (slot would be {slot_v2:+.2f} m)',
        fontsize=7, color='steelblue', ha='right', va='top')
# hump: raw, enforced (planning), tightened, second rung
for k, (xs, zr) in enumerate(pts):
    ax.plot(xs, zr, color='darkorange', lw=2.0, zorder=5)
    ax.plot(xs, zr + off_plan, color='crimson', lw=1.4, ls='--', zorder=6)
    ax.plot(xs, zr + off_tight, color='crimson', lw=0.9, ls=':', zorder=6)
    ax.fill_between(xs, zr, zr + off_plan, color='crimson', alpha=0.12, lw=0, zorder=2)
    ax.fill_between(xs, 0, zr, color='darkorange', alpha=0.10, lw=0, zorder=2)
for xs, zr in pts_lo:
    ax.plot(xs, zr, color='darkorange', lw=1.0, ls=(0, (2, 2)), alpha=0.8, zorder=4)
    ax.plot(xs, zr + off_plan, color='crimson', lw=0.8, ls=(0, (2, 2)), alpha=0.7, zorder=4)
ax.text(0.0, H_lo - 0.10, f'2nd rung H={H_lo:.2f} m\n(defined, not scheduled)', fontsize=6.5, color='saddlebrown',
        ha='center', va='top')
# x_active windows + walls span + route end
for hs in hump['halfspace_constraints']:
    if hs.get('plane') == 'xz':
        lo, hi = hs['x_active']
        ax.axvline(lo, color='saddlebrown', lw=0.6, ls=':', zorder=1); ax.axvline(hi, color='saddlebrown', lw=0.6, ls=':', zorder=1)
        ax.text((lo + hi) / 2, 0.06, f'x_active [{lo:.1f}, {hi:.1f}]', fontsize=6.5, color='saddlebrown', ha='center')
for xw in (-2.0, 2.0):
    ax.axvline(xw, color='darkorange', lw=0.8, ls='--', alpha=0.6, zorder=1)
ax.text(-2.0, z_ub + 0.16, 'walls x∈[−2, 2]', fontsize=6.5, color='darkorange', ha='left')
ax.axvline(ROUTE_X[1], color='seagreen', lw=1.2, ls='-.', zorder=2)
ax.text(ROUTE_X[1] - 0.04, 0.35, 'route end / finish plane x=2.8\n(altitude does not affect success)', fontsize=6.5,
        color='seagreen', ha='right')
ax.axvline(ROUTE_X[0], color='seagreen', lw=0.8, ls='-.', alpha=0.6, zorder=2)
# the two demands: climb and slot
ax.annotate('', xy=(0.0, peak_req_t), xytext=(0.0, LAUNCH_Z),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.2), zorder=8)
ax.text(0.08, (peak_req_t + LAUNCH_Z) / 2, f'climb ≈ {climb:.2f} m\n(launch → {peak_req_t:.3f} m)', fontsize=7.5, va='center')
ax.annotate('', xy=(0.9, ceil_req), xytext=(0.9, peak_req_t),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.2), zorder=8)
ax.text(0.98, (ceil_req + peak_req_t) / 2, f'escape slot {slot:.2f} m\nz ∈ [{peak_req_t:.2f}, {ceil_req:.2f}]', fontsize=7.5, va='center')
ax.plot([0.0], [peak_req], 'o', color='crimson', ms=5, zorder=9)
ax.text(-0.12, peak_req + 0.10, f'peak: raw {H:.2f} + {off_plan:.3f} = {peak_req:.3f} m\n(+{tight:.3f} tightened → {peak_req_t:.3f})',
        fontsize=7, color='crimson', ha='right', va='bottom')
ax.plot([-x_bind, x_bind], [LAUNCH_Z, LAUNCH_Z], 's', color='crimson', ms=4, zorder=9)
ax.annotate(f'limit crosses the launch altitude at |x| = {x_bind:.2f}', xy=(-x_bind, LAUNCH_Z), xytext=(-2.2, 0.70),
            fontsize=6.5, color='crimson', arrowprops=dict(arrowstyle='-', color='crimson', lw=0.6), zorder=9)
ax.set_xlim(-3.05, 3.05); ax.set_ylim(0.0, 3.05)
ax.set_xlabel('x (m)', fontsize=8); ax.set_ylabel('z (m)', fontsize=8)
ax.legend(handles=[
    Line2D([0], [0], color='darkorange', lw=2.0, label='hump roof (raw, virtual — scored only)'),
    Line2D([0], [0], color='crimson', lw=1.4, ls='--', label=f'drone-centre limit = roof + {r_plan:.2f}·√(1+s²) = +{off_plan:.3f} m'),
    Line2D([0], [0], color='crimson', lw=0.9, ls=':', label=f'-tightened (+{tight:.3f} m ⊥): +{off_tight:.3f} m'),
    Line2D([0], [0], color='steelblue', lw=1.4, label='workspace box z (raw / − r_drone dashed)'),
    mpa.Patch(color='0.55', alpha=0.18, label='flown band of the v2 slide flights'),
], fontsize=7, loc='upper right', framealpha=0.95)

# — top-down (x, y) —
ay.set_title('top-down (x, y): walls unchanged, no lateral escape', fontsize=10, loc='left')
ay.set_aspect('equal')
for hs in hump['halfspace_constraints']:
    if hs.get('plane', 'xy') != 'xy':
        continue
    (x1, y1), (x2, y2) = hs['line']
    ay.plot([x1, x2], [y1, y2], color='darkorange', lw=2.0, zorder=5)
    sgn = 1.0 if hs['side'] == 'above' else -1.0
    ay.plot([x1, x2], [y1 + sgn * r_plan, y2 + sgn * r_plan], color='crimson', lw=1.2, ls='--', zorder=6)
    ay.fill_between([x1, x2], [y1, y2], [y1 + sgn * r_plan, y2 + sgn * r_plan], color='crimson', alpha=0.12, lw=0)
for ob in hump['obstacle_constraints']:
    cx, cy = ob['center']
    ay.add_patch(mpa.Circle((cx, cy), ob['radius'] + r_plan, color='tomato', alpha=0.25, lw=0, zorder=4))
    ay.plot(cx, cy, '+', color='tomato', ms=6, zorder=5)
ay.add_patch(mpa.Rectangle((-1.5, -0.95), 3.0, 1.9, facecolor='darkorange', alpha=0.10, edgecolor='saddlebrown',
                           ls=':', lw=0.8, zorder=2))
ay.text(0.0, 0.78, 'hump footprint x∈[−1.5, 1.5]\n(z constraint across the full width)', fontsize=7, color='saddlebrown',
        ha='center', va='top')
for name, y in ROUTES.items():
    ay.plot(ROUTE_X, [y, y], color='0.45', lw=0.9, alpha=0.8, zorder=3)
    ay.text(ROUTE_X[0] - 0.08, y, name, fontsize=7, color='0.35', ha='right', va='center')
ay.plot([ROUTE_X[1]] * 2, [-0.95, 0.95], color='seagreen', lw=1.2, ls='-.', zorder=3)
ay.set_xlim(-3.2, 3.2); ay.set_ylim(-1.35, 1.35)
ay.set_xlabel('x (m)', fontsize=8); ay.set_ylabel('y (m)', fontsize=8)
ay.legend(handles=[
    Line2D([0], [0], color='darkorange', lw=2.0, label='corridor_v2 wall (inner face)'),
    Line2D([0], [0], color='crimson', lw=1.2, ls='--', label=f'wall − r_drone ({r_plan:.2f} m)'),
    mpa.Patch(color='tomato', alpha=0.25, label='wall-end cap (+ r_drone)'),
    Line2D([0], [0], color='0.45', lw=0.9, label='routes L / C / R (y = ∓0.12, 0)'),
], fontsize=7, loc='lower right', framealpha=0.95)

g0 = (peak_req > FLOWN_HI) and (slot >= 0.6)
fig.suptitle(f'Gen15 U19 · gate G0 — corridor_v3_ablation_hump (H = {H:.2f} m, slope {s:.3f})   '
             f'enforced peak {peak_req:.3f} m > band top {FLOWN_HI:.2f} m: {"yes" if peak_req > FLOWN_HI else "NO"};   '
             f'slot {slot:.2f} m ≥ 0.6 m: {"yes" if slot >= 0.6 else "NO"}   →   G0 {"PASS" if g0 else "FAIL"}',
             fontsize=10, fontweight='bold', y=0.995)
fig.text(0.5, 0.005,
         f'Scene XML, model, checkpoint, routes, goals and the projector stack (-bounds_free-pdes-tightened) are those of '
         f'corridor_v2_slide; the 14° x–y slide is dropped. The scorer uses the same r_drone ({r_score:.2f} m) as the planner, '
         f'so its limit coincides with the dashed line. Divergence guard: envelope z {ENVELOPE_Z_UB:.2f} + {DIV_SLACK:.1f} m slack = '
         f'{ENVELOPE_Z_UB + DIV_SLACK:.2f} m, above the ceiling.  Every unprojected flight violates at the peak by '
         f'{peak_req - FLOWN_HI:.2f}–{peak_req - FLOWN_LO:.2f} m.',
         ha='center', fontsize=7, color='dimgray', style='italic', wrap=True)
fig.subplots_adjust(left=0.05, right=0.99, top=0.91, bottom=0.14)
out = os.path.join(HERE, 'fig_u19_g0_ablation_hump_overview')
fig.savefig(out + '.png', dpi=150, bbox_inches='tight'); fig.savefig(out + '.svg', bbox_inches='tight')
plt.close(fig)

print(f'H={H:.2f}  slope={s:.4f}  off_plan={off_plan:.4f}  off_tight={off_tight:.4f}  off_score={off_score:.4f}')
print(f'peak required z: {peak_req:.4f} (tightened {peak_req_t:.4f});  flown band {FLOWN_LO}-{FLOWN_HI};  climb {climb:.3f} m')
print(f'ceiling centre limit {ceil_req:.3f};  slot {slot:.3f} m  (v2 ceiling: {slot_v2:+.3f} m);  binds at |x| < {x_bind:.3f}')
print(f'2nd rung H_lo={H_lo:.2f}: peak required {H_lo + off_plan:.4f} (tightened {H_lo + off_tight:.4f})')
print(f'G0: {"PASS" if g0 else "FAIL"}   -> {out}.png / .svg')
