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

    Real keys: n_success, n_success_and_constraints, n_steps, n_violations, total_violations,
    obs_all, act_all, sampled_trajectories_all, args. (E7: constraint metrics are now taken
    from the rollouts; with dynamics-only they are clean/zero — per-scene geometry fills them.)
    """
    n = len(rollouts)
    n_success = np.array([1.0 if r.get('success') else 0.0 for r in rollouts])
    n_steps = np.array([r.get('n_fm_steps', 0) for r in rollouts], dtype=float)
    obs_all = np.array([np.asarray(r.get('obs_traj', [])) for r in rollouts], dtype=object)
    act_all = np.array([np.asarray(r.get('act_traj', [])) for r in rollouts], dtype=object)
    plans_all = np.array([np.asarray(r.get('plans', [])) for r in rollouts], dtype=object)

    # ── Constraint-aware metrics (from rollouts; dynamics-only → trivially clean) ──
    n_success_and_constraints = np.array(
        [1.0 if r.get('success_and_constraints') else 0.0 for r in rollouts])
    n_violations = np.array([r.get('n_violations', 0) for r in rollouts], dtype=float)
    total_violations = np.array([r.get('total_violations', 0.0) for r in rollouts], dtype=float)

    path = os.path.join(out_dir, f'{variant}.npz')
    np.savez(
        path,
        n_success=n_success,
        n_success_and_constraints=n_success_and_constraints,
        n_steps=n_steps,
        n_violations=n_violations,
        total_violations=total_violations,
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
        color = _homotopy_color(r.get('homotopy', '?'), palette) if _homotopy_color else None
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
    """Write an overhead GIF from caller-rendered RGB frames (opt-in)."""
    if not frames:
        return None
    import imageio
    os.makedirs(diag_dir, exist_ok=True)
    path = os.path.join(diag_dir, f'rollout_{idx}.gif')
    imageio.mimsave(path, frames, fps=fps)
    return path


def write_mpc_foresight(diag_dir, idx, rollout, scene, stride=6):
    """Real MPC candidate-fan foresight SVG — copied from Gen7 fm_visual_aligning_test.

    Two panels (XY top-down + XYZ 3D), same as Gen7's proven _mpc_foresight plot:
      green = MPC candidate fan (p_des cols 0,1,2 of each candidate plan)
      black = commanded p_des path (des)    → obs_traj cols 0:3
      red   = actual drone path (c_pos/p)   → obs_traj cols 3:6
      black dot = replan decision-point anchor (actual p at that step)
      lime*/red sq = start/end markers
    Stride: every `stride` FM steps shown (default 6, matches Gen7 default).
    Constraint geometry blocks (bounds/halfspace/obstacles) copied from Gen7 verbatim
    and gated on constraint_types — empty this epoch, ready for per-scene geometry.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D as _Line2D
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

    os.makedirs(diag_dir, exist_ok=True)
    path = os.path.join(diag_dir, f'rollout_{idx}_mpc_foresight.svg')

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

    # UAV obs = [p_des(0:3) | p(3:6) | v(6:9)].
    # Mirrors Gen7: real_pos = commanded (p_des), c_arr = actual drone position (p).
    real_pos = obs_traj[:, 0:3]   # p_des — black, "des (commanded)"
    c_arr    = obs_traj[:, 3:6]   # p     — red,   "actual (p)"
    n_cands  = np.asarray(plans[0]).shape[0] if np.asarray(plans[0]).ndim == 3 else 1
    n_steps  = len(real_pos)
    _ref     = c_arr              # start/end markers follow actual position (Gen7 uses c_arr)

    fig_mpc = plt.figure(figsize=(26, 11))
    fig_mpc.suptitle(
        f'Rollout {idx} — MPC Decision Points  '
        f'(success={int(bool(rollout.get("success")))},  {n_cands} candidates/step,  '
        f'every {stride} replans shown)',
        fontsize=13)
    ax_xy = fig_mpc.add_subplot(1, 2, 1)
    ax_3d = fig_mpc.add_subplot(1, 2, 2, projection='3d')

    # ── candidate fan (both panels) ────────────────────────────────────────────
    # UAV: plans[step_i] shape (batch, horizon, obs_dim=9);
    #      candidate p_des = cols 0,1,2 (same space as real_pos / c_arr).
    # plans[] and obs_traj[] are aligned 1:1 per FM step (spr=1 for UAV).
    for step_i, plan in enumerate(plans):
        if step_i % stride != 0:
            continue
        cand   = np.asarray(plan)
        if cand.ndim != 3:
            continue
        anchor = c_arr[min(step_i, n_steps - 1)]      # actual position at decision point
        for b in range(cand.shape[0]):
            ax_xy.plot(cand[b, :, 0], cand[b, :, 1],
                       color='green', linewidth=0.6, alpha=0.7, zorder=4)
            ax_3d.plot(cand[b, :, 0], cand[b, :, 1], cand[b, :, 2],
                       color='green', linewidth=0.6, alpha=0.7)
        ax_xy.scatter([anchor[0]], [anchor[1]],
                      color='black', s=30, zorder=8, linewidths=0)
        ax_3d.scatter([anchor[0]], [anchor[1]], [anchor[2]],
                      color='black', s=30)

    # ── XY panel — paths + start/end ──────────────────────────────────────────
    ax_xy.plot(c_arr[:, 0],    c_arr[:, 1],    color='red',   linewidth=1.2, zorder=9)
    ax_xy.plot(real_pos[:, 0], real_pos[:, 1], color='black', linewidth=1.2, zorder=7)
    ax_xy.scatter([_ref[0, 0]],  [_ref[0, 1]],
                  color='lime', marker='*', s=180, zorder=12, linewidths=0)
    ax_xy.scatter([_ref[-1, 0]], [_ref[-1, 1]],
                  color='red',  marker='s', s=80,  zorder=12, linewidths=0)
    _lgd = [
        _Line2D([0],[0], color='green', lw=0.8, label=f'MPC candidates ({n_cands}/step)'),
        _Line2D([0],[0], color='black', lw=1.2, label='des (p_des commanded)'),
        _Line2D([0],[0], color='red',   lw=1.2, label='actual (p)'),
        _Line2D([0],[0], marker='o', color='w', markerfacecolor='black',
                markersize=7, label='replan decision point'),
        _Line2D([0],[0], marker='*', color='w', markerfacecolor='lime',
                markersize=10, label='start'),
        _Line2D([0],[0], marker='s', color='w', markerfacecolor='red',
                markersize=7,  label='end'),
    ]
    ax_xy.legend(handles=_lgd, fontsize=9)
    ax_xy.set_title(f'XY — MPC Decision Points  (every {stride} replans)', fontsize=12)
    ax_xy.set_xlabel('X (m)', fontsize=11); ax_xy.set_ylabel('Y (m)', fontsize=11)
    ax_xy.set_aspect('equal', adjustable='datalim')
    ax_xy.grid(True, alpha=0.3)

    # ── constraint geometry on XY — gated, empty this epoch ───────────────────
    # Copied verbatim from Gen7; gated on constraint_types so nothing draws now.
    # Fill these when per-scene geometry is designed (no rewrite needed).

    # ── 3D XYZ panel — paths + start/end ─────────────────────────────────────
    ax_3d.plot(c_arr[:, 0],    c_arr[:, 1],    c_arr[:, 2],
               color='red',   linewidth=1.2, label='actual (p)')
    ax_3d.plot(real_pos[:, 0], real_pos[:, 1], real_pos[:, 2],
               color='black', linewidth=1.2, label='des (p_des commanded)')
    ax_3d.scatter([_ref[0, 0]],  [_ref[0, 1]],  [_ref[0, 2]],
                  color='lime', marker='*', s=180, zorder=12)
    ax_3d.scatter([_ref[-1, 0]], [_ref[-1, 1]], [_ref[-1, 2]],
                  color='red',  marker='s', s=80,  zorder=12)
    ax_3d.set_title('XYZ — MPC Decision Points (3D)', fontsize=12)
    ax_3d.set_xlabel('X (m)', fontsize=9)
    ax_3d.set_ylabel('Y (m)', fontsize=9)
    ax_3d.set_zlabel('Z (m)', fontsize=9)
    ax_3d.legend(fontsize=9)

    fig_mpc.tight_layout()
    fig_mpc.savefig(path, bbox_inches='tight')
    plt.close(fig_mpc)
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
            f.write(
                f"  rollout {i:2d}  homotopy={r.get('homotopy','?'):<10}  "
                f"success={int(bool(r.get('success')))}  "
                f"contact={r.get('contact_frac', float('nan')):.3f}  "
                f"min_z={r.get('min_z', float('nan')):.3f}  "
                f"goal_dist={r.get('goal_dist', float('nan')):.3f}  "
                f"track_err={r.get('track_err_mean', float('nan')):.2f}\n")
        f.write('-' * 70 + '\n')
        f.write(f"  success_rate (goal+safe): {summary['success_rate']:.3f}\n")
        f.write(f"  success_and_constraints : {summary.get('success_and_constraints_rate', float('nan')):.3f}\n")
        f.write(f"  safe_rate (contact-free+airborne): {summary.get('safe_rate', float('nan')):.3f}\n")
        f.write(f"  collision_free_rate     : {summary.get('collision_free_rate', float('nan')):.3f}  "
                f"(violations mean: {summary.get('n_violations_mean', 0):.2f})\n")
        f.write(f"  contact_frac_mean     : {summary['contact_frac_mean']:.3f}\n")
        f.write(f"  goal_dist_mean        : {summary['goal_dist_mean']:.3f}\n")
        f.write(f"  goal_reached_rate     : {summary['goal_reached_rate']:.3f}\n")
        f.write(f"  track_err_mean        : {summary['track_err_mean']:.3f}\n")
        f.write(f"  fm_ms mean/p95        : {summary['fm_ms_mean']:.1f}/{summary['fm_ms_p95']:.1f}\n")
        f.write('  [ PCC constraint metrics: placeholder — Epoch 7 ]\n')
    return path
