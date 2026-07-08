"""U3 — legacy-schema eval artifact writers for UAV FM (Gen11 E6).

Restores the FMv3-ODE-style output set for the UAV closed-loop eval:
per-(scene,seed,projection) `<variant>.npz` + `eval_<variant>.log` + 2-D overview
`<variant>.png`/`all.png` + per-rollout `diagnostics/rollout_<r>_stats.json` +
optional `rollout_<r>.gif`.

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
    for r in rollouts:
        obs = np.asarray(r.get('obs_traj', []))
        if obs.ndim != 2 or obs.shape[0] == 0:
            continue
        x, y, z = obs[:, P_X], obs[:, P_Y], obs[:, P_Z]
        # Fix_12: color by the class ACTUALLY flown when available (pillars) — the commanded
        # `homotopy` label is only the expert route's tag; the unconditioned FM picks its own.
        _hlabel = r.get('homotopy_flown') or r.get('homotopy', '?')
        color = _homotopy_color(_hlabel, palette) if _homotopy_color else None
        ax_xy.plot(x, y, color=color, lw=1.5, alpha=0.8)
        ax_xy.plot(x[0], y[0], 'o', color='#2ca02c', ms=5, zorder=5)   # start
        ax_xz.plot(x, z, color=color, lw=1.5, alpha=0.8)

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

    fig.suptitle(f'UAV FM eval — {scene} — variant={variant} — {len(rollouts)} trials',
                 fontsize=13)
    fig.tight_layout()
    main = os.path.join(out_dir, f'{variant}.png')
    fig.savefig(main, dpi=130)
    fig.savefig(os.path.join(out_dir, 'all.png'), dpi=130)   # aggregate alias (single-seed)
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


def write_mpc_foresight(diag_dir, idx, rollout, scene, stride=6):
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

    fig, (ax_xy, ax_xz) = plt.subplots(1, 2, figsize=(22, 9))
    fig.suptitle(
        f'Rollout {idx} — MPC Decision Points  '
        f'(success={int(bool(_succ))},  {n_cands} cands/step,  '
        f'every {stride} FM steps shown)',
        fontsize=13)

    # ── candidate fan ─────────────────────────────────────────────────────────
    for step_i, plan in enumerate(plans):
        if step_i % stride != 0:
            continue
        cand = np.asarray(plan)
        if cand.ndim != 3:
            continue
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
    ax_xy.legend(handles=_lgd, fontsize=9)
    ax_xy.set_title(f'XY top-down + obstacles  (every {stride} steps)', fontsize=12)
    ax_xy.set_xlabel('X (m)', fontsize=11); ax_xy.set_ylabel('Y (m)', fontsize=11)
    ax_xy.set_aspect('equal', adjustable='datalim')
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
        for i, r in enumerate(rollouts):
            phys = r.get('physical', {}); goal = r.get('goal', {}); succ = r.get('success', {})
            # Fix_12: show the flown class next to the commanded label (pillars only).
            _flown = r.get('homotopy_flown')
            _flown_str = f"flown={_flown:<10}  " if _flown else ''
            f.write(
                f"  rollout {i:2d}  homotopy={r.get('homotopy','?'):<10}  {_flown_str}"
                f"success={int(bool(succ.get('strict')))}  "
                f"success_relaxed={int(bool(succ.get('relaxed')))}  "
                f"contact={phys.get('contact_frac', float('nan')):.3f}  "
                f"min_z={phys.get('min_z', float('nan')):.3f}  "
                f"goal_dist={goal.get('dist', float('nan')):.3f}  "
                f"track_err={r.get('track_err_mean', float('nan')):.2f}\n")
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
        f.write(f"  track_err_mean        : {summary['track_err_mean']:.3f}\n")
        f.write(f"  fm_ms mean/p95        : {_t.get('fm_ms_mean', float('nan')):.1f}/{_t.get('fm_ms_p95', float('nan')):.1f}\n")
        f.write('  [ PCC constraint metrics: placeholder — Epoch 7 ]\n')
    return path
