"""U3 — legacy-schema eval artifact writers for UAV FM (Gen11 E6).

Restores the FMv3-ODE-style output set for the UAV closed-loop eval:
per-(scene,seed,projection) `<variant>.npz` + `eval_<variant>.log` + 2-D overview
`<variant>.png` + per-rollout `diagnostics/rollout_<r>_stats.json` +
optional `rollout_<r>.gif`. (Fix_14 dropped the redundant per-variant `all.png` duplicate.)

SCOPE (see U3 PLAN): originally only the `fm_only` / diffuser baseline, with
constraint/PCC fields scaffolded as placeholders so the Epoch-7 DPCC work would be a
drop-in with no schema change. Epoch 7 (`Epoch7_fm_pcc_FULL_PCC_MPC/U1`) has since
filled those placeholders in: `n_success_and_constraints`/`n_violations` are real, and
`write_mpc_foresight` draws the actual per-step MPC candidate fan instead of a stub.

Pure IO + matplotlib; no torch/MuJoCo here (rendering frames are produced by the
caller, which holds the MuJoCo model/data). Importable, syntax-checked in Docker.
"""

import os
import json

import numpy as np

# Heavy per-rollout arrays produced by rollout_one but NOT written to results.json.
HEAVY_KEYS = ('obs_traj', 'act_traj', 'plans', 'frames')

# obs layout (raw): [ p_des(0:3) | p(3:6) | v(6:9) ]  — position is cols 3:6.
P_X, P_Y, P_Z = 3, 4, 5
AIRBORNE_Z = 0.2          # must match the airborne gate in eval_fm_uav.rollout_one

# ── Div_Abort: robust plot windows ───────────────────────────────────────────
# matplotlib autoscales to the DATA. A single runaway excursion — p_des integrating to
# hundreds of metres while the drone tumbles, or one wild candidate in the fan — therefore
# compresses a 7-metre arena into a couple of pixel rows, and the SVG degenerates into two
# near-empty panels with an x/y/z hike through them. The window is instead built from content
# that CANNOT run away (the enforced geometry + a robust percentile band of the flown path)
# and is then allowed to grow for the rest only up to a hard cap. Anything beyond the window
# is still drawn — matplotlib just clips it — and the excursion is called out in a corner
# note so nothing is silently hidden.

VIEW_PCT = (2.0, 98.0)     # percentile band of the flown path that always stays in frame
VIEW_MAX_GROW = 1.0        # extra span (multiples of the core span) the runaway may claim


def _finite_cat(arrays):
    """Flatten `arrays` into one finite 1-D array, or None if nothing usable is left."""
    out = []
    for a in arrays:
        if a is None:
            continue
        a = np.asarray(a, dtype=float).reshape(-1)
        a = a[np.isfinite(a)]
        if a.size:
            out.append(a)
    return np.concatenate(out) if out else None


def view_window(core, extra=(), fixed=(), pad=0.4, pct=VIEW_PCT, max_grow=VIEW_MAX_GROW):
    """(lo, hi) axis limits an excursion cannot destroy.

    core  — content that sets the scale: the flown path (robust `pct` band, so even a flight
            that itself escaped keeps its normal-flight portion in frame).
    fixed — content that must ALWAYS be fully visible: the enforced geometry, obstacles.
    extra — content allowed to widen the window, but only by `max_grow` * core span on each
            side: the commanded p_des path and the MPC candidate fan.
    Returns None when there is nothing finite to plot (caller then leaves autoscale alone).
    """
    core_cat = _finite_cat(core)
    fixed_cat = _finite_cat(fixed)
    if core_cat is not None:
        lo = float(np.percentile(core_cat, pct[0]))
        hi = float(np.percentile(core_cat, pct[1]))
    elif fixed_cat is not None:
        lo, hi = float(fixed_cat.min()), float(fixed_cat.max())
    else:
        return None
    if fixed_cat is not None:
        lo, hi = min(lo, float(fixed_cat.min())), max(hi, float(fixed_cat.max()))
    span = max(hi - lo, 1e-3)
    ex = _finite_cat(extra)
    if ex is not None:
        lo = min(lo, max(float(ex.min()), lo - max_grow * span))
        hi = max(hi, min(float(ex.max()), hi + max_grow * span))
    if hi - lo < 1e-6:
        lo, hi = lo - 0.5, hi + 0.5
    return lo - pad, hi + pad


def _outside_note(ax, series, xlim, ylim):
    """Corner note naming what the clamped window is cutting off (0 → nothing drawn).

    `series` = [(label, xs, ys), ...]. Keeps the clamp honest: the reader is told how many
    samples of which path fell outside and how far the worst one went.
    """
    msgs = []
    for label, xs, ys in series:
        xs = np.asarray(xs, dtype=float).reshape(-1)
        ys = np.asarray(ys, dtype=float).reshape(-1)
        if xs.size == 0 or xs.size != ys.size:
            continue
        bad = (~np.isfinite(xs)) | (~np.isfinite(ys)) | (xs < xlim[0]) | (xs > xlim[1]) | \
              (ys < ylim[0]) | (ys > ylim[1])
        n = int(bad.sum())
        if n:
            far = _finite_cat([np.abs(xs[bad]), np.abs(ys[bad])])
            reach = f', max |coord| {float(far.max()):.1f} m' if far is not None else ''
            msgs.append(f'{n} {label} pt(s) outside view{reach}')
    if msgs:
        ax.text(0.99, 0.01, 'view clamped: ' + '; '.join(msgs), transform=ax.transAxes,
                fontsize=7, color='crimson', ha='right', va='bottom', zorder=14,
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.75, lw=0))


def geometry_anchors(geo_config, obstacles, variant=''):
    """(xs, ys, zs) coordinates that must stay in frame: enforced surfaces + raw obstacles.

    Read-only twin of `draw_projector_geometry`'s geometry resolution (same toggles: geo_free
    drops the geometric families, -tightened widens the margin) — it only collects extents.
    """
    xs, ys, zs = [], [], []
    for obs in (obstacles or []):
        c = obs.get('center', [0.0, 0.0, 0.0])
        r = float(obs.get('radius', 0.0)) if 'radius' in obs else float(
            np.max(obs.get('half_extents', [0.0])))
        xs += [float(c[0]) - r, float(c[0]) + r]
        ys += [float(c[1]) - r, float(c[1]) + r]
        if len(c) > 2:
            zs += [float(c[2]) - r, float(c[2]) + r]
    if not geo_config:
        return xs, ys, zs
    ctypes = list(geo_config.get('constraint_types', []))
    geo_off = 'geo_free' in (variant or '')
    _infl = geo_config.get('inflation') or {}
    margin = float(_infl.get('r_drone', 0.0)) + float(_infl.get('margin_base', 0.0))
    if 'tightened' in (variant or ''):
        margin += float(geo_config.get('enlarge_constraints') or 0.0)
    ws = geo_config.get('workspace_bounds')
    if (not geo_off) and 'geo_bounds' in ctypes and ws is not None:
        lb = np.asarray(ws['lb'], dtype=float) + margin
        ub = np.asarray(ws['ub'], dtype=float) - margin
        for axis, acc in enumerate((xs, ys, zs)):
            if np.isfinite(lb[axis]):
                acc.append(float(lb[axis]))
            if np.isfinite(ub[axis]):
                acc.append(float(ub[axis]))
    if (not geo_off) and 'halfspace' in ctypes:
        for hs in geo_config.get('halfspace_constraints', []):
            (x1, y1), (x2, y2), _side, _xa = _fs_wall_xy(hs)
            xs += [float(x1), float(x2)]
            ys += [float(y1), float(y2)]
    if (not geo_off) and 'obstacles' in ctypes:
        for ob in geo_config.get('obstacle_constraints', []):
            c = ob.get('center', [0.0, 0.0])
            r = float(ob.get('radius', 0.0)) + margin
            xs += [float(c[0]) - r, float(c[0]) + r]
            ys += [float(c[1]) - r, float(c[1]) + r]
    return xs, ys, zs


def rollout_divergence(rollout):
    """The rollout's Div_Abort record, or None when it flew to a normal end."""
    d = (rollout or {}).get('divergence') or {}
    return d if d.get('aborted') else None



def json_safe_rollouts(rollouts):
    """Strip the heavy arrays/frames so the per-rollout metrics stay JSON-serialisable."""
    return [{k: v for k, v in r.items() if k not in HEAVY_KEYS} for r in rollouts]


# ── npz (legacy FMv3-ODE schema) ─────────────────────────────────────────────

def save_npz(out_dir, variant, rollouts, args_dict):
    """Write `<variant>.npz` matching the legacy FMv3ODE schema analysis scripts expect.

    Fix_10 (2/2): array names are group-prefixed to match rollout_one's nested JSON schema
    (`physical`/`constraint`/`goal`/`success` groups), and the physical/goal metrics that were
    always computed but never persisted to NPZ (only the constraint-axis ones were) are now
    saved too — the same "share metrics between the per-rollout JSON and the aggregate-across-
    trials NPZ" mapping, e.g. JSON `physical.safe` <-> NPZ `phys_safe`, JSON
    `constraint.n_violations` <-> NPZ `constraint_n_violations`, JSON `success.strict` <-> NPZ
    `success_strict`. NPZ itself stays flat (np.savez has no native nesting); only the key
    names + the added arrays changed. See
    logs_in_develop/Gen11/Epoch9_PCC_Constraints/Fix_10_json_metrics/PLAN_fix10_2_json_schema_redesign.md
    This is a UAV-only rename/addition — avoiding/visual-aligining's own npz writers untouched.

    Real keys: success_strict, success_relaxed, success_strict_and_constraints,
    success_relaxed_and_constraints, n_steps, phys_safe, phys_contact_frac, phys_min_z,
    phys_final_z, constraint_collision_free, constraint_n_violations,
    constraint_total_violations, goal_reached, goal_dist, goal_crossed_line, obs_all, act_all,
    sampled_trajectories_all, args.
    """
    def _f(r, group, key, default=0.0):
        return float(r.get(group, {}).get(key, default))

    def _b(r, group, key):
        return 1.0 if r.get(group, {}).get(key) else 0.0

    n_steps = np.array([r.get('n_fm_steps', 0) for r in rollouts], dtype=float)
    obs_all = np.array([np.asarray(r.get('obs_traj', [])) for r in rollouts], dtype=object)
    act_all = np.array([np.asarray(r.get('act_traj', [])) for r in rollouts], dtype=object)
    plans_all = np.array([np.asarray(r.get('plans', [])) for r in rollouts], dtype=object)

    # ── success (2x2 matrix, same 4 fields as the JSON `success` group) ──────────
    success_strict = np.array([_b(r, 'success', 'strict') for r in rollouts])
    success_relaxed = np.array([_b(r, 'success', 'relaxed') for r in rollouts])
    success_strict_and_constraints = np.array([_b(r, 'success', 'strict_and_constraints') for r in rollouts])
    success_relaxed_and_constraints = np.array([_b(r, 'success', 'relaxed_and_constraints') for r in rollouts])

    # ── physical (Axis A, hard MuJoCo contact truth) — NEW, was never persisted before ──
    phys_safe = np.array([_b(r, 'physical', 'safe') for r in rollouts])
    phys_contact_frac = np.array([_f(r, 'physical', 'contact_frac') for r in rollouts])
    phys_min_z = np.array([_f(r, 'physical', 'min_z') for r in rollouts])
    phys_final_z = np.array([_f(r, 'physical', 'final_z') for r in rollouts])

    # ── constraint (Axis B, declared-margin truth; dynamics-only → trivially clean) ──
    constraint_collision_free = np.array([_b(r, 'constraint', 'collision_free') for r in rollouts])
    constraint_n_violations = np.array([_f(r, 'constraint', 'n_violations') for r in rollouts])
    constraint_total_violations = np.array([_f(r, 'constraint', 'total_violations') for r in rollouts])

    # ── goal — NEW, was never persisted before ───────────────────────────────────
    goal_reached = np.array([_b(r, 'goal', 'reached') for r in rollouts])
    goal_dist = np.array([_f(r, 'goal', 'dist') for r in rollouts])
    goal_crossed_line = np.array([_b(r, 'goal', 'crossed_line') for r in rollouts])

    # ── projection health (Fix_15.3) — flag rollouts whose projection was ABANDONED ──
    # by the sustained-slowness circuit breaker (projection.py Fix_15.2). A tripped rollout
    # ran (partly) on the UNPROJECTED trajectory, so its constraint metrics are NOT valid —
    # downstream analysis MUST treat projection_cb_tripped==1 rows as "projection broken".
    projection_cb_tripped = np.array([_b(r, 'projection_health', 'cb_tripped') for r in rollouts])
    projection_cb_skipped_steps = np.array([_f(r, 'projection_health', 'cb_skipped_steps') for r in rollouts])

    # ── Div_Abort: which rollouts were STOPPED because the drone lost control ────
    # An aborted row is a miss by construction (physical.safe forced False) and its
    # constraint counts cover only the steps actually flown, so downstream analysis must
    # treat divergence_aborted==1 rows separately rather than averaging them in blind.
    # `divergence_step` is the FM step the abort fired on (-1 when it never did) and
    # `divergence_reason` the greppable tag (nan_state / off_map / off_route /
    # overspeed / inverted; '' when the rollout ended normally).
    divergence_aborted = np.array([_b(r, 'divergence', 'aborted') for r in rollouts])
    divergence_step = np.array([_f(r, 'divergence', 'step', -1.0) for r in rollouts])
    divergence_reason = np.array([(r.get('divergence', {}) or {}).get('reason') or ''
                                  for r in rollouts], dtype=object)

    path = os.path.join(out_dir, f'{variant}.npz')
    np.savez(
        path,
        success_strict=success_strict,
        success_relaxed=success_relaxed,
        success_strict_and_constraints=success_strict_and_constraints,
        success_relaxed_and_constraints=success_relaxed_and_constraints,
        n_steps=n_steps,
        phys_safe=phys_safe,
        phys_contact_frac=phys_contact_frac,
        phys_min_z=phys_min_z,
        phys_final_z=phys_final_z,
        constraint_collision_free=constraint_collision_free,
        constraint_n_violations=constraint_n_violations,
        constraint_total_violations=constraint_total_violations,
        goal_reached=goal_reached,
        goal_dist=goal_dist,
        goal_crossed_line=goal_crossed_line,
        projection_cb_tripped=projection_cb_tripped,               # Fix_15.3
        projection_cb_skipped_steps=projection_cb_skipped_steps,   # Fix_15.3
        divergence_aborted=divergence_aborted,                      # Div_Abort
        divergence_step=divergence_step,                            # Div_Abort
        divergence_reason=divergence_reason,                        # Div_Abort
        obs_all=obs_all,
        act_all=act_all,
        sampled_trajectories_all=plans_all,
        args=np.array(args_dict, dtype=object),
    )
    return path


# ── 2-D overview (top-down x,y + side x,z) — the GIF replacement ──────────────

def plot_overview(out_dir, variant, scene, rollouts):
    """Top-down (x,y) path overview with obstacles + a side (x,z) altitude panel.

    The altitude panel with the airborne-gate line makes the UAV failure mode
    (never leaving the floor) visible at a glance — top-down alone hides it.
    Reuses the scene-aware obstacle drawing from generate_overview_plots.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Reuse the existing UAV-native obstacle drawer + colours; degrade gracefully.
    try:
        from uav_expert_data_collect.generate_overview_plots import (
            _draw_obstacles, _homotopy_color)
        import uav_expert_data_collect.generator as gen
        obstacles = gen.SCENE_OBSTACLES.get(scene, [])
    except Exception as exc:          # pragma: no cover - cluster-only deps
        print(f'[ artifacts ] obstacle/colour helpers unavailable ({exc}); plain plot')
        _draw_obstacles = None
        _homotopy_color = None
        obstacles = []

    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(16, 8))
    palette = {}
    _all_x, _all_y, _all_z = [], [], []      # Div_Abort: view-window bookkeeping
    for r in rollouts:
        obs = np.asarray(r.get('obs_traj', []))
        if obs.ndim != 2 or obs.shape[0] == 0:
            continue
        x, y, z = obs[:, P_X], obs[:, P_Y], obs[:, P_Z]
        _all_x.append(x); _all_y.append(y); _all_z.append(z)
        # Fix_12: color by the class ACTUALLY flown when available (pillars) — the commanded
        # `homotopy` label is only the expert route's tag; the unconditioned FM picks its own.
        _hlabel = r.get('homotopy_flown') or r.get('homotopy', '?')
        color = _homotopy_color(_hlabel, palette) if _homotopy_color else None
        ax_xy.plot(x, y, color=color, lw=1.5, alpha=0.8)
        ax_xy.plot(x[0], y[0], 'o', color='#2ca02c', ms=5, zorder=5)   # start
        ax_xz.plot(x, z, color=color, lw=1.5, alpha=0.8)

        # Div_Abort: ✖ where this trial lost control (the trace stops there).
        _dv = rollout_divergence(r)
        if _dv and _dv.get('p') is not None:
            _ap = np.asarray(_dv['p'], dtype=float)
            ax_xy.scatter([_ap[0]], [_ap[1]], marker='X', s=150, color='darkred',
                          edgecolors='white', linewidths=1.0, zorder=8)
            ax_xz.scatter([_ap[0]], [_ap[2]], marker='X', s=150, color='darkred',
                          edgecolors='white', linewidths=1.0, zorder=8)

    if _draw_obstacles is not None:
        _draw_obstacles(ax_xy, obstacles)
    ax_xy.set_title(f'{scene} — top-down (x, y)')
    ax_xy.set_xlabel('x [m]'); ax_xy.set_ylabel('y [m]')
    ax_xy.set_aspect('equal', 'box'); ax_xy.grid(True, alpha=0.3)

    ax_xz.axhline(AIRBORNE_Z, ls='--', color='r', lw=1.2,
                  label=f'airborne gate z={AIRBORNE_Z} m')
    ax_xz.set_title(f'{scene} — side (x, z): altitude')
    ax_xz.set_xlabel('x [m]'); ax_xz.set_ylabel('z [m]')
    ax_xz.grid(True, alpha=0.3); ax_xz.legend(loc='best', fontsize=8)

    # Div_Abort: clamp the overview to a window one runaway trial cannot destroy (same rule as
    # the foresight SVG — flown paths set the scale, obstacles stay in frame). Aborted traces
    # stop at their ✖, so the surviving excursion is at most the last step or two.
    _gx, _gy, _gz = geometry_anchors(None, obstacles)
    _xl = view_window(_all_x, fixed=_gx)
    _yl = view_window(_all_y, fixed=_gy)
    _zl = view_window(_all_z, fixed=list(_gz) + [0.0, AIRBORNE_Z], pad=0.15)
    if _xl:
        ax_xy.set_xlim(*_xl); ax_xz.set_xlim(*_xl)
    if _yl:
        ax_xy.set_ylim(*_yl)
    if _zl:
        ax_xz.set_ylim(*_zl)

    fig.suptitle(f'UAV FM eval — {scene} — variant={variant} — {len(rollouts)} trials',
                 fontsize=13)
    # Div_Abort: flag trials stopped early because the drone lost control.
    _n_div = sum(1 for r in rollouts if (r.get('divergence') or {}).get('aborted'))
    if _n_div:
        _reasons = sorted({r['divergence']['reason'] for r in rollouts
                           if (r.get('divergence') or {}).get('aborted')})
        fig.text(0.5, 0.915,
                 f'✖ DIVERGENCE ABORT on {_n_div}/{len(rollouts)} trials '
                 f'({", ".join(_reasons)}) — those traces stop at their ✖ marker',
                 color='white', backgroundcolor='darkred', fontsize=11, fontweight='bold',
                 ha='center', va='center')
    # Fix_15.3: flag if the projection circuit breaker tripped on ANY trial of this variant.
    _n_tripped = sum(1 for r in rollouts if (r.get('projection_health') or {}).get('cb_tripped'))
    if _n_tripped:
        fig.text(0.5, 0.95,
                 f'⚠ PROJECTION CIRCUIT-BREAKER TRIPPED on {_n_tripped}/{len(rollouts)} trials '
                 f'— those paths are UNPROJECTED (sustained SLSQP slowness)',
                 color='white', backgroundcolor='crimson', fontsize=11, fontweight='bold',
                 ha='center', va='center')
    fig.tight_layout()
    main = os.path.join(out_dir, f'{variant}.png')
    fig.savefig(main, dpi=130)
    # Fix_14: dropped the `all.png` alias — `out_dir` is the PER-VARIANT folder
    # (results/<geo_tag>/<variant>/), so `all.png` was a byte-identical duplicate of
    # `<variant>.png`, not the "all variants, this seed" aggregate its name promised (that would
    # have to live one level up at <geo_tag>/). Nothing consumes it; removed to stop the confusion.
    plt.close(fig)
    return main


# ── per-rollout diagnostics ──────────────────────────────────────────────────

def save_rollout_stats(diag_dir, idx, rollout):
    """One JSON of metrics per rollout (heavy arrays stripped)."""
    os.makedirs(diag_dir, exist_ok=True)
    stats = {k: v for k, v in rollout.items() if k not in HEAVY_KEYS}
    path = os.path.join(diag_dir, f'rollout_{idx}_stats.json')
    with open(path, 'w') as f:
        json.dump(stats, f, indent=2)
    return path


def save_rollout_gif(diag_dir, idx, frames, fps=10):
    """Write an overhead GIF from caller-rendered RGB frames (opt-in).

    Fix_7: encode with a reduced/optimized palette + delta-frame compression when the
    installed imageio's GIF writer supports it — background (scene/obstacles) is static
    across frames, only the drone moves, so `subrectangles` alone typically shrinks file
    size substantially with no visible quality loss. Falls back to the plain call if the
    active imageio backend/version rejects these kwargs (never fails the save outright).
    Fix_9: palettesize lowered 128→64 (still ample for a steelblue/tomato/drone overhead
    scene — a handful of flat colors, not a photo) for a further ~2x on the color-table side.
    """
    if not frames:
        return None
    import imageio
    os.makedirs(diag_dir, exist_ok=True)
    path = os.path.join(diag_dir, f'rollout_{idx}.gif')
    try:
        imageio.mimsave(path, frames, fps=fps, subrectangles=True, palettesize=64)
    except TypeError:
        imageio.mimsave(path, frames, fps=fps)
    return path


# ── projector constraint-geometry overlay (Fix_14) ───────────────────────────

def _fs_normalize_halfspace(hs):
    """Foresight-local copy of eval_fm_uav._normalize_halfspace (kept here to avoid a
    circular import back into the __main__ eval module). Returns (triple, x_active)."""
    if isinstance(hs, dict):
        line = hs['line']
        return [line[0], line[1], hs['side']], hs.get('x_active')
    return [hs[0], hs[1], hs[2]], None


def _fs_wall_xy(hs):
    """Resolve a halfspace to (p1, p2, side, x_active), clipped to its live x-range so the
    s_curve per-segment walls are drawn only where they are actually enforced."""
    triple, x_active = _fs_normalize_halfspace(hs)
    (x1, y1), (x2, y2), side = triple
    if x_active is not None and abs(x2 - x1) > 1e-9:
        lo, hi = x_active
        t0 = (lo - x1) / (x2 - x1); t1 = (hi - x1) / (x2 - x1)
        t0, t1 = sorted((max(0.0, min(1.0, t0)), max(0.0, min(1.0, t1))))
        x1n = x1 + t0 * (x2 - x1); x2n = x1 + t1 * (x2 - x1)
        y1n = y1 + t0 * (y2 - y1); y2n = y1 + t1 * (y2 - y1)
        return (x1n, y1n), (x2n, y2n), side, (lo, hi)
    return (x1, y1), (x2, y2), side, x_active


def draw_projector_geometry(ax_xy, ax_xz, geo_config, variant=''):
    """Fix_14: overlay the ENFORCED projector constraint surfaces onto the foresight panels.

    Without this the foresight SVG only showed the raw scene obstacles (SCENE_OBSTACLES), so on
    e.g. s_curve/dpcc-r there was nothing to read the green candidate fan against — you couldn't
    tell whether the projector actually solved the corner halfspaces and re-routed the plan.

    Mirrors eval_fm_uav.plot_geo_constraints (same colours, same TRUE enforced margin
    r_drone+margin_base [+enlarge if -tightened]) but paints onto the caller's existing axes.
    Respects the per-variant toggles so the overlay is what THIS variant's QP saw:
      • `geo_free`  → geometric families removed (drawn as an explicit 'NOT enforced' note)
      • `-tightened`→ margin += enlarge_constraints
    Returns matplotlib legend handles for the caller to merge into its legend.
    """
    import matplotlib.patches as _mpa
    from matplotlib.lines import Line2D as _Line2D

    if not geo_config:
        return []
    ctypes  = list(geo_config.get('constraint_types', []))
    geo_off = 'geo_free' in (variant or '')
    _infl   = geo_config.get('inflation') or {}
    inflation_base = float(_infl.get('r_drone', 0.0)) + float(_infl.get('margin_base', 0.0))
    enlarge = float(geo_config.get('enlarge_constraints') or 0.0) if 'tightened' in (variant or '') else 0.0
    margin  = inflation_base + enlarge

    show_box = (not geo_off) and 'geo_bounds' in ctypes and geo_config.get('workspace_bounds') is not None
    hs_list  = geo_config.get('halfspace_constraints', []) if ((not geo_off) and 'halfspace' in ctypes) else []
    obs_list = geo_config.get('obstacle_constraints', [])  if ((not geo_off) and 'obstacles' in ctypes) else []

    lb_d = ub_d = None
    if show_box:
        lb_d = np.array(geo_config['workspace_bounds']['lb'], dtype=float) + margin
        ub_d = np.array(geo_config['workspace_bounds']['ub'], dtype=float) - margin
        for _i, _fb in ((0, (-3.6, 3.6)), (1, (-2.0, 2.0)), (2, (0.0, 2.0))):   # clamp ±inf for display
            if np.isinf(lb_d[_i]): lb_d[_i] = _fb[0]
            if np.isinf(ub_d[_i]): ub_d[_i] = _fb[1]
        ax_xy.add_patch(_mpa.Rectangle((lb_d[0], lb_d[1]), ub_d[0]-lb_d[0], ub_d[1]-lb_d[1],
                                       lw=1.4, edgecolor='steelblue', facecolor='none',
                                       ls='--', alpha=0.85, zorder=2))
        ax_xz.axhline(lb_d[2], color='steelblue', ls='--', lw=1.0, alpha=0.7, zorder=2)
        ax_xz.axhline(ub_d[2], color='steelblue', ls='--', lw=1.0, alpha=0.7, zorder=2)

    cz_mid = (lb_d[2] + ub_d[2]) / 2 if lb_d is not None else 0.9
    for hs in hs_list:
        (hx1, hy1), (hx2, hy2), side, x_active = _fs_wall_xy(hs)
        ax_xy.plot([hx1, hx2], [hy1, hy2], color='darkorange', lw=2.2, zorder=6)
        dx, dy = hx2 - hx1, hy2 - hy1; nrm = np.hypot(dx, dy) or 1.0
        nx, ny = (-dy/nrm, dx/nrm) if side == 'above' else (dy/nrm, -dx/nrm)  # arrow → feasible side
        mx, my = (hx1 + hx2) / 2, (hy1 + hy2) / 2
        ax_xy.annotate('', xy=(mx + nx*0.25, my + ny*0.25), xytext=(mx, my),
                       arrowprops=dict(arrowstyle='->', color='darkorange', lw=1.4), zorder=6)
        if x_active is not None:
            ax_xy.text(mx, my, f'x∈[{x_active[0]:.1f},{x_active[1]:.1f}]', fontsize=6,
                       color='saddlebrown', ha='center', va='bottom', zorder=7)
        xb_lo, xb_hi = sorted((hx1, hx2))
        ax_xz.axvspan(xb_lo, xb_hi, color='darkorange', alpha=0.08, zorder=1)
    for obs in obs_list:
        cx, cy = float(obs['center'][0]), float(obs['center'][1])
        ax_xy.add_patch(_mpa.Circle((cx, cy), float(obs['radius']) + margin, lw=1.5,
                                    edgecolor='tomato', facecolor='tomato', alpha=0.18, zorder=5))
        ax_xy.plot(cx, cy, '+', color='tomato', ms=7, zorder=6)
        ax_xz.add_patch(_mpa.Circle((cx, cz_mid), float(obs['radius']) + margin, lw=1.0,
                                    edgecolor='tomato', facecolor='none', ls='--', alpha=0.6, zorder=2))

    handles = []
    if show_box:
        handles.append(_Line2D([0], [0], color='steelblue', ls='--', lw=1.4, label='workspace box (enforced)'))
    if hs_list:
        handles.append(_Line2D([0], [0], color='darkorange', lw=2.2, label='halfspace wall (enforced)'))
    if obs_list:
        handles.append(_Line2D([0], [0], marker='o', color='w', markerfacecolor='tomato',
                               markersize=9, alpha=0.6, label='obstacle+margin (enforced)'))
    if geo_off and any(t in ctypes for t in ('geo_bounds', 'halfspace', 'obstacles')):
        ax_xy.text(0.02, 0.98, 'geo_free: geometry NOT enforced\n(surfaces hidden)',
                   transform=ax_xy.transAxes, fontsize=8, color='crimson', va='top', ha='left', zorder=13)
    return handles


def write_mpc_foresight(diag_dir, idx, rollout, scene, stride=6, geo_config=None, variant=''):
    """MPC candidate-fan foresight SVG — UAV-specific two-panel design.

    Panel layout (UAV is a 3D flight task; altitude deserves its own axis):
      LEFT  — XY top-down:  horizontal navigation, obstacle avoidance
      RIGHT — XZ altitude:  Z profile, airborne gate, p_des-z explosion visible

    Both panels share the Gen7 dual-path convention:
      green       = MPC candidate p_des fan (cols 0,1,2), every `stride` FM steps
      black solid = commanded p_des path  (obs_traj cols 0:3)
      red solid   = actual drone position (obs_traj cols 3:6)
      black dot   = replan anchor at actual p position
      lime ★ / red ■ = start / end

    Fix_14: when `geo_config` (+ `variant`) is supplied the ENFORCED projector surfaces are
    overlaid via `draw_projector_geometry` — steelblue workspace box, darkorange halfspace walls
    (s_curve segments clipped to their live x-range, with a feasible-side arrow), tomato
    obstacle balls, all at the TRUE margin r_drone+margin_base [+enlarge if -tightened]. This is
    what lets you see whether the candidate fan actually solved the corner (dpcc-r on s_curve).
    Raw SCENE_OBSTACLES are still drawn underneath (physical collision truth vs planning surface).

    Why NOT the Gen7 3D panel: matplotlib 3D SVG is a static projection — can't
    rotate, altitude (Z) is compressed into perspective, candidate fan becomes a
    clutter of overlapping green lines.  For UAV the dedicated XZ panel reads the
    altitude story clearly (did p_des_z explode? is the drone airborne?) at a glance.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D as _Line2D

    os.makedirs(diag_dir, exist_ok=True)
    path = os.path.join(diag_dir, f'rollout_{idx}_mpc_foresight.svg')

    # Fix_12: `success` became a nested group in the Fix_10 schema — bool(dict) was always
    # True, so every foresight title read success=1. Read the strict field, tolerate both.
    _succ = rollout.get('success')
    _succ = _succ.get('strict') if isinstance(_succ, dict) else _succ

    plans    = rollout.get('plans', [])
    obs_traj = np.asarray(rollout.get('obs_traj', []))
    if not plans or obs_traj.ndim != 2 or obs_traj.shape[0] == 0:
        fig, ax = plt.subplots(figsize=(4, 1.2))
        ax.axis('off')
        ax.text(0.02, 0.5, 'no candidate-fan data for this rollout',
                fontsize=10, color='#888888')
        fig.savefig(path)
        plt.close(fig)
        return path

    # UAV obs = [p_des(0:3) | p(3:6) | v(6:9)]
    des = obs_traj[:, 0:3]   # commanded p_des — black
    act = obs_traj[:, 3:6]   # actual drone p  — red
    n_cands = np.asarray(plans[0]).shape[0] if np.asarray(plans[0]).ndim == 3 else 1
    n_steps = len(des)

    # ── obstacle geometry (reuse proven helpers from generate_overview_plots) ──
    try:
        from uav_expert_data_collect.generate_overview_plots import _draw_obstacles
        import uav_expert_data_collect.generator as _gen
        obstacles = _gen.SCENE_OBSTACLES.get(scene, [])
    except Exception:
        obstacles = []
        _draw_obstacles = None

    _div = rollout_divergence(rollout)
    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle(
        f'Rollout {idx} — MPC Decision Points  '
        f'(success={int(bool(_succ))},  {n_cands} cands/step,  '
        f'every {stride} FM steps shown)'
        + (f'   ✖ ABORTED at step {_div["step"]} — {_div["reason"]}' if _div else ''),
        fontsize=13)
    # Div_Abort: loud banner naming WHEN / WHERE / WHY the flight was declared lost. The panels
    # below therefore show a TRUNCATED episode — everything after this step never happened.
    if _div:
        fig.text(0.5, 0.925,
                 f'✖ DIVERGENCE ABORT — {_div["reason"]} at FM step {_div["step"]} '
                 f'(t={_div["time_s"]:.2f}s, physics step {_div["physics_step"]}):  {_div["detail"]}',
                 color='white', backgroundcolor='darkred', fontsize=11, fontweight='bold',
                 ha='center', va='center')
    # Fix_15.3: loud banner when the projection circuit breaker tripped for this rollout —
    # the green candidate fan below is (partly) UNPROJECTED, so it does NOT reflect the
    # enforced constraints. See projection.py Fix_15.2 (sustained SLSQP slowness).
    _ph = rollout.get('projection_health', {}) or {}
    if _ph.get('cb_tripped'):
        fig.text(0.5, 0.955,
                 f'⚠ PROJECTION CIRCUIT-BREAKER TRIPPED — {int(_ph.get("cb_skipped_steps", 0))} step(s) '
                 f'UNPROJECTED (sustained SLSQP slowness). Candidate fan is NOT constraint-valid.',
                 color='white', backgroundcolor='crimson', fontsize=12, fontweight='bold',
                 ha='center', va='center')

    # ── candidate fan ─────────────────────────────────────────────────────────
    _cand_pts = []          # Div_Abort: every drawn candidate point, for the view window
    for step_i, plan in enumerate(plans):
        if step_i % stride != 0:
            continue
        cand = np.asarray(plan)
        if cand.ndim != 3:
            continue
        _cand_pts.append(cand[:, :, :3].reshape(-1, 3))   # Div_Abort: view-window bookkeeping
        anchor = act[min(step_i, n_steps - 1)]
        for b in range(cand.shape[0]):
            ax_xy.plot(cand[b, :, 0], cand[b, :, 1],
                       color='green', lw=0.6, alpha=0.7, zorder=4)
            ax_xz.plot(cand[b, :, 0], cand[b, :, 2],
                       color='green', lw=0.6, alpha=0.7, zorder=4)
        ax_xy.scatter([anchor[0]], [anchor[1]], color='black', s=30, zorder=8, linewidths=0)
        ax_xz.scatter([anchor[0]], [anchor[2]], color='black', s=30, zorder=8, linewidths=0)

    # ── XY panel — obstacles + paths ─────────────────────────────────────────
    if _draw_obstacles is not None:
        _draw_obstacles(ax_xy, obstacles)          # proven: circles for cylinders, rects for boxes
    # Fix_14: overlay the ENFORCED projector surfaces (walls/balls/box at the true margin) so the
    # green candidate fan can be read against what the QP constrained — not just raw geometry.
    _geo_handles = draw_projector_geometry(ax_xy, ax_xz, geo_config, variant)

    ax_xy.plot(act[:, 0], act[:, 1], color='red',   lw=1.2, zorder=9)
    ax_xy.plot(des[:, 0], des[:, 1], color='black', lw=1.2, zorder=7)
    ax_xy.scatter([act[0, 0]],  [act[0, 1]],  color='lime', marker='*', s=180, zorder=12, linewidths=0)
    ax_xy.scatter([act[-1, 0]], [act[-1, 1]], color='red',  marker='s', s=80,  zorder=12, linewidths=0)
    _lgd = [
        _Line2D([0],[0], color='green', lw=0.8, label=f'MPC candidates ({n_cands}/step)'),
        _Line2D([0],[0], color='black', lw=1.2, label='des (p_des)'),
        _Line2D([0],[0], color='red',   lw=1.2, label='actual (p)'),
        _Line2D([0],[0], marker='o', color='w', markerfacecolor='black',
                markersize=7, label='replan anchor'),
        _Line2D([0],[0], marker='*', color='w', markerfacecolor='lime',
                markersize=10, label='start'),
        _Line2D([0],[0], marker='s', color='w', markerfacecolor='red',
                markersize=7,  label='end'),
    ]
    ax_xy.legend(handles=_lgd + _geo_handles, fontsize=9)
    ax_xy.set_title(f'XY top-down + enforced constraints  (every {stride} steps)', fontsize=12)
    ax_xy.set_xlabel('X (m)', fontsize=11); ax_xy.set_ylabel('Y (m)', fontsize=11)
    ax_xy.set_aspect('equal', adjustable='box')   # Div_Abort: 'box' honours the clamped limits
    ax_xy.grid(True, alpha=0.3)

    # ── XZ altitude panel — obstacle silhouettes + paths ─────────────────────
    import matplotlib.patches as _mpa
    for obs in obstacles:
        cx = float(obs['center'][0])
        cz = float(obs['center'][2]) if len(obs['center']) > 2 else 0.75
        otype = obs.get('type', '')
        if otype == 'cylinder':
            r  = float(obs['radius'])
            hh = float(obs.get('half_height', 1.0))
            ax_xz.add_patch(_mpa.Rectangle(
                (cx - r, 0.0), 2 * r, cz + hh,
                facecolor='#555555', edgecolor='#222222',
                lw=1.2, alpha=0.65, zorder=3))
        elif otype == 'box':
            hx = float(obs['half_extents'][0])
            hz = float(obs['half_extents'][2])
            ax_xz.add_patch(_mpa.Rectangle(
                (cx - hx, cz - hz), 2 * hx, 2 * hz,
                facecolor='#555555', edgecolor='#222222',
                lw=1.2, alpha=0.65, zorder=3))

    ax_xz.plot(act[:, 0], act[:, 2], color='red',   lw=1.2, zorder=9, label='actual (p)')
    ax_xz.plot(des[:, 0], des[:, 2], color='black', lw=1.2, zorder=7, label='des (p_des)')
    ax_xz.axhline(AIRBORNE_Z, ls='--', color='orange', lw=1.2,
                  label=f'airborne gate z={AIRBORNE_Z} m', zorder=5)
    ax_xz.scatter([act[0, 0]],  [act[0, 2]],  color='lime', marker='*', s=180, zorder=12, linewidths=0)
    ax_xz.scatter([act[-1, 0]], [act[-1, 2]], color='red',  marker='s', s=80,  zorder=12, linewidths=0)
    ax_xz.legend(fontsize=9)
    ax_xz.set_title('XZ altitude + obstacle silhouettes', fontsize=12)
    ax_xz.set_xlabel('X (m)', fontsize=11); ax_xz.set_ylabel('Z (m)', fontsize=11)
    ax_xz.grid(True, alpha=0.3)

    # ── Div_Abort: clamp both panels to a window a runaway cannot destroy ─────
    # Scale comes from the flown path (robust band) and the enforced geometry; p_des and the
    # candidate fan may widen it only up to VIEW_MAX_GROW spans. Excursions are still drawn
    # (matplotlib clips them) and counted in the corner note, so nothing is hidden silently.
    _cands = np.concatenate(_cand_pts, axis=0) if _cand_pts else np.zeros((0, 3))
    _gx, _gy, _gz = geometry_anchors(geo_config, obstacles, variant)
    _xlim = view_window([act[:, 0]], extra=[des[:, 0], _cands[:, 0]], fixed=_gx)
    _ylim = view_window([act[:, 1]], extra=[des[:, 1], _cands[:, 1]], fixed=_gy)
    _zlim = view_window([act[:, 2]], extra=[des[:, 2], _cands[:, 2]],
                        fixed=list(_gz) + [0.0, AIRBORNE_Z], pad=0.15)
    if _xlim:
        ax_xy.set_xlim(*_xlim); ax_xz.set_xlim(*_xlim)
    if _ylim:
        ax_xy.set_ylim(*_ylim)
    if _zlim:
        ax_xz.set_ylim(*_zlim)
    if _xlim and _ylim:
        _outside_note(ax_xy, [('p_des', des[:, 0], des[:, 1]), ('actual', act[:, 0], act[:, 1]),
                              ('candidate', _cands[:, 0], _cands[:, 1])], _xlim, _ylim)
    if _xlim and _zlim:
        _outside_note(ax_xz, [('p_des', des[:, 0], des[:, 2]), ('actual', act[:, 0], act[:, 2]),
                              ('candidate', _cands[:, 0], _cands[:, 2])], _xlim, _zlim)

    # ── Div_Abort: mark WHERE the abort fired, on both panels ────────────────
    # ✖ = the drone's last physical position; the dotted leader points at the commanded p_des
    # it was chasing (usually far outside the window — that IS the failure).
    if _div and _div.get('p') is not None:
        _ap = np.asarray(_div['p'], dtype=float)
        _ad = np.asarray(_div.get('p_des') or _div['p'], dtype=float)
        for _ax, (_i, _j) in ((ax_xy, (0, 1)), (ax_xz, (0, 2))):
            _ax.plot([_ap[_i], _ad[_i]], [_ap[_j], _ad[_j]], color='darkred', ls=':', lw=1.4,
                     alpha=0.9, zorder=13)
            _ax.scatter([_ap[_i]], [_ap[_j]], marker='X', s=260, color='darkred',
                        edgecolors='white', linewidths=1.2, zorder=14)
            _ax.annotate(f'ABORT step {_div["step"]}\n{_div["reason"]}',
                         xy=(_ap[_i], _ap[_j]), xytext=(6, 8), textcoords='offset points',
                         fontsize=8, color='darkred', fontweight='bold', zorder=14,
                         bbox=dict(boxstyle='round,pad=0.25', facecolor='white', alpha=0.8, lw=0))

    fig.tight_layout()
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


# ── eval log ─────────────────────────────────────────────────────────────────

def write_eval_log(out_dir, variant, summary, rollouts):
    """Plain-text eval log (per-rollout lines + summary), mirroring eval_<variant>.log."""
    path = os.path.join(out_dir, f'eval_{variant}.log')
    with open(path, 'w') as f:
        f.write('=' * 70 + '\n')
        f.write(f"UAV FM eval  |  scene={summary['scene']}  seed={summary['seed']}  "
                f"variant={variant}\n")
        f.write('=' * 70 + '\n')
        # Fix_15.3: loud banner if the projection circuit breaker tripped on any trial.
        _tripped = [i for i, r in enumerate(rollouts) if (r.get('projection_health') or {}).get('cb_tripped')]
        if _tripped:
            f.write('!' * 70 + '\n')
            f.write(f"  !!! PROJECTION CIRCUIT-BREAKER TRIPPED on {len(_tripped)}/{len(rollouts)} "
                    f"trials: {_tripped}\n")
            f.write("  !!! Those trials ran (partly) UNPROJECTED (sustained SLSQP slowness,\n")
            f.write("  !!! projection.py Fix_15.2) — their constraint metrics are NOT valid.\n")
            f.write('!' * 70 + '\n')
        # Div_Abort: loud banner if any trial was cut short by a lost-control abort.
        _diverged = [i for i, r in enumerate(rollouts) if (r.get('divergence') or {}).get('aborted')]
        if _diverged:
            f.write('!' * 70 + '\n')
            f.write(f"  !!! DIVERGENCE ABORT on {len(_diverged)}/{len(rollouts)} trials: {_diverged}\n")
            for _i in _diverged:
                _d = rollouts[_i]['divergence']
                f.write(f"  !!!   trial {_i}: {_d['reason']} @ step {_d['step']} "
                        f"(t={_d['time_s']:.3f}s)  p={np.round(_d['p'], 2).tolist()}  "
                        f"p_des={np.round(_d['p_des'], 2).tolist()}  |v|={_d['speed']:.2f} m/s\n")
                f.write(f"  !!!     why: {_d['detail']}\n")
            f.write("  !!! Those flights were STOPPED early and are scored as misses.\n")
            f.write('!' * 70 + '\n')
        for i, r in enumerate(rollouts):
            phys = r.get('physical', {}); goal = r.get('goal', {}); succ = r.get('success', {})
            # Fix_12: show the flown class next to the commanded label (pillars only).
            _flown = r.get('homotopy_flown')
            _flown_str = f"flown={_flown:<10}  " if _flown else ''
            # Fix_15.3: per-rollout circuit-breaker marker (skipped-step count when tripped).
            _ph = r.get('projection_health', {}) or {}
            _cb_str = f"  cb=TRIPPED({int(_ph.get('cb_skipped_steps', 0))})" if _ph.get('cb_tripped') else ''
            # Div_Abort: per-rollout abort marker (reason + the step it fired on).
            _dv = r.get('divergence') or {}
            _dv_str = f"  ABORT({_dv.get('reason')}@{_dv.get('step')})" if _dv.get('aborted') else ''
            f.write(
                f"  rollout {i:2d}  homotopy={r.get('homotopy','?'):<10}  {_flown_str}"
                f"success={int(bool(succ.get('strict')))}  "
                f"success_relaxed={int(bool(succ.get('relaxed')))}  "
                f"contact={phys.get('contact_frac', float('nan')):.3f}  "
                f"min_z={phys.get('min_z', float('nan')):.3f}  "
                f"goal_dist={goal.get('dist', float('nan')):.3f}  "
                f"track_err={r.get('track_err_mean', float('nan')):.2f}{_cb_str}{_dv_str}\n")
        f.write('-' * 70 + '\n')
        _s = summary.get('success', {}); _p = summary.get('physical', {})
        _c = summary.get('constraint', {}); _g = summary.get('goal', {}); _t = summary.get('timing', {})
        f.write(f"  success_rate (goal+safe): {_s.get('strict_rate', float('nan')):.3f}\n")
        f.write(f"  success_relaxed_rate (crossed finish line): "
                f"{_s.get('relaxed_rate', float('nan')):.3f}\n")
        f.write(f"  success_and_constraints : {_s.get('strict_and_constraints_rate', float('nan')):.3f}\n")
        f.write(f"  safe_rate (contact-free+airborne): {_p.get('safe_rate', float('nan')):.3f}\n")
        f.write(f"  collision_free_rate     : {_c.get('collision_free_rate', float('nan')):.3f}  "
                f"(violations mean: {_c.get('n_violations_mean', 0):.2f})\n")
        f.write(f"  contact_frac_mean     : {_p.get('contact_frac_mean', float('nan')):.3f}\n")
        f.write(f"  goal_dist_mean        : {_g.get('dist_mean', float('nan')):.3f}\n")
        f.write(f"  goal_reached_rate     : {_g.get('reached_rate', float('nan')):.3f}\n")
        # U_13: DPCC-style step accounting (early-stop on goal-reach; miss = full budget).
        _st = summary.get('steps', {})
        f.write(f"  steps_mean            : {_st.get('mean', float('nan')):.1f}  "
                f"(to_goal {_st.get('to_goal_mean', float('nan')):.1f} / "
                f"budget {_st.get('max_episode_length', '?')})\n")
        f.write(f"  track_err_mean        : {summary['track_err_mean']:.3f}\n")
        # Div_Abort: how many trials never finished because the drone lost control.
        _n_div = sum(1 for r in rollouts if (r.get('divergence') or {}).get('aborted'))
        f.write(f"  divergence_aborts     : {_n_div}/{len(rollouts)}"
                + (f"  reasons={sorted({r['divergence']['reason'] for r in rollouts if (r.get('divergence') or {}).get('aborted')})}"
                   if _n_div else '') + '\n')
        f.write(f"  fm_ms mean/p95        : {_t.get('fm_ms_mean', float('nan')):.1f}/{_t.get('fm_ms_p95', float('nan')):.1f}\n")
        f.write('  [ PCC constraint metrics: placeholder — Epoch 7 ]\n')
    return path
