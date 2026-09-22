"""Standard per-cell artifacts of a Mode T (turbo) cell — the avoiding suite, drawn for the drone (Gen15 U18).

    <variant>.png                          avoiding-style per-cell figure (rows = episodes, cols = x/y/x_des/y_des over
                                           steps, path against the constraints, path + stored MPC foresight fans)
    diagnostics/rollout_<i>_mpc_foresight.svg   per episode: the UAV suite's own writer (eval_artifacts.write_mpc_foresight),
                                           LEFT XY top-down (green = stored candidate fans, black = p_des, red = drone),
                                           RIGHT XZ altitude, with the avoiding geometry mapped into the world
    eval_<variant>.log                     summary text: Panda vs drone, plant stats, per-episode table

No MuJoCo, no GL, no torch: matplotlib only. The foresight fans are the planner's stored `sampled_trajectories_all`
(they were produced on the table; the replay does not plan), drawn exactly as the avoiding eval draws them
(obs cols x=2, y=3, first `horizon` rows, up to 4 candidates, every horizon//2 steps).
"""
import os

import numpy as np

from uav_avoiding_bridge import frame as F
from uav_avoiding_bridge.overview_plot import draw_avoiding, draw_world

OBS_X, OBS_Y = 2, 3


def _fans(fans_ep):
    """list of per-step arrays (n_cand, H, obs_dim) or [] if the source has none."""
    if fans_ep is None:
        return []
    out = []
    for f in fans_ep:
        a = np.asarray(f, float)
        if a.ndim == 3:
            out.append(a)
    return out


def save_cell_png(path, cell, geo, rows_list, panda_list, fans_list, per, plot_how_many=10):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    n = min(len(rows_list), plot_how_many)
    if n == 0:
        return None
    H = next((f[0].shape[1] for f in fans_list[:n] if f), 8)
    every = max(1, H // 2)
    fig, ax = plt.subplots(n, 6, figsize=(30, 5 * n), squeeze=False)
    fig.suptitle(f"{cell['engine']} · {cell['eval']} · seed {cell['seed']} · {cell['geo']} · {cell['variant']} — DRONE replay "
                 f"(black = drone, grey = Panda source)", fontsize=12)
    names = ['x', 'y', 'x_des', 'y_des']
    cols = [OBS_X, OBS_Y, 0, 1]
    for i in range(n):
        R = np.asarray(rows_list[i], float)
        Pn = np.asarray(panda_list[i], float)
        sc = per[i] or {}
        for j in range(4):
            if len(R): ax[i, j].plot(R[:, cols[j]], 'k')
            if len(Pn): ax[i, j].plot(Pn[:, cols[j]], color='0.6', ls='--', lw=0.8)
            ax[i, j].set_title(names[j])
        for a_ in (ax[i, 4], ax[i, 5]):
            draw_avoiding(a_, geo)
            if len(Pn): a_.plot(Pn[:, OBS_X], Pn[:, OBS_Y], color='0.6', ls='--', lw=0.8, zorder=6)
            if len(R):
                a_.plot(R[:, OBS_X], R[:, OBS_Y], 'k', zorder=7)
                a_.plot(R[0, OBS_X], R[0, OBS_Y], 'go', label='Start', zorder=8)
            a_.get_legend().remove() if a_.get_legend() else None
        fans = fans_list[i] if i < len(fans_list) else []
        for k in range(0, len(fans), every):
            f = fans[k]
            for c in range(min(f.shape[0], 4)):
                ax[i, 5].plot(f[c, :H, OBS_X], f[c, :H, OBS_Y], 'b', lw=0.6, alpha=0.7, zorder=6)
                ax[i, 5].plot(f[c, 0, OBS_X], f[c, 0, OBS_Y], 'go', ms=2, zorder=6)
        ax[i, 4].set_title(f"drone path · ended {sc.get('ended', '?')} · S&C {int(sc.get('n_success_and_constraints', 0))} "
                           f"viol {int(sc.get('n_violations', 0))} steps {int(sc.get('n_steps', 0))}")
        ax[i, 5].set_title(f'+ stored MPC foresight (every {every} steps, ≤ 4 candidates)')
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(path, dpi=70)
    plt.close(fig)
    return path


def uav_geo_config(geo, scale):
    """The avoiding geometry as a UAV-yaml-style geo_config in WORLD coordinates, for eval_artifacts'
    draw_projector_geometry / _draw_obstacles. Side labels are re-derived after the rotation by testing the
    (feasible) start point against the mapped line — no algebra on the slope sign."""
    to = lambda x, y: list(F.to_world_xy(x, y, scale))
    S = to(*F.START_A)
    hs = []
    for c in geo['polytopic']:
        p0, p1 = to(*c[0]), to(*c[1])
        # feasible side in the avoiding frame is c[2]; find where the start lies w.r.t. the WORLD line
        if abs(p1[0] - p0[0]) < 1e-9:
            hs.append([p0, p1, 'below']); continue
        y_line = p0[1] + (p1[1] - p0[1]) / (p1[0] - p0[0]) * (S[0] - p0[0])
        hs.append([p0, p1, 'below' if S[1] <= y_line else 'above'])
    obs = [{'type': 'sphere_outside', 'dimensions': ['x', 'y'], 'center': to(*ob['center']), 'radius': scale * ob['radius']}
           for ob in geo['obstacles']]
    scene_obs = [{'type': 'cylinder', 'name': f'pillar_{n}', 'center': [*to(xa, ya), F.PILLAR_HEIGHT / 2],
                  'radius': scale * r, 'half_height': F.PILLAR_HEIGHT / 2} for n, xa, ya, r in F.OBSTACLES_A]
    return {'constraint_types': ['halfspace', 'obstacles'], 'halfspace_constraints': hs, 'obstacle_constraints': obs,
            'inflation': {'r_drone': 0.0, 'margin_base': 0.0}, 'enlarge_constraints': scale * geo['enlarge'],
            'scene_obstacles': scene_obs}


def uav_rollout(rows, fans, ep_rec, sc, scale, altitude):
    """Pack one replayed episode into the UAV rollout schema write_mpc_foresight reads:
    plans = per-step (n_cand, H, 6) [p_des_w | p_w], obs_traj = (T, 6) [p_des_w | p_w]."""
    R = np.asarray(rows, float)
    T = len(R)
    Pd = np.array([[*F.to_world_xy(r[0], r[1], scale), altitude] for r in R]) if T else np.zeros((0, 3))
    pw = np.asarray(ep_rec['path_w'], float) if ep_rec and ep_rec.get('path_w') else np.zeros((0, 3))
    if len(pw) != T:                                   # fall back to the mapped drone xy at the altitude
        pw = np.array([[*F.to_world_xy(r[OBS_X], r[OBS_Y], scale), altitude] for r in R]) if T else np.zeros((0, 3))
    obs_traj = np.hstack([Pd, pw]) if T else np.zeros((0, 6))
    plans = []
    for f in fans[:T]:
        a = np.asarray(f, float)                       # (n_cand, H, 4) [x_des, y_des, x, y]
        n_c, H = a.shape[0], a.shape[1]
        w = np.zeros((n_c, H, 6))
        for b in range(n_c):
            for h in range(H):
                w[b, h, 0:2] = F.to_world_xy(a[b, h, 0], a[b, h, 1], scale); w[b, h, 2] = altitude
                w[b, h, 3:5] = F.to_world_xy(a[b, h, OBS_X], a[b, h, OBS_Y], scale); w[b, h, 5] = altitude
        plans.append(w)
    div = None
    ended = (sc or {}).get('ended', '')
    if ended and ended not in ('success', 'contact', 'exhausted', 'cap'):
        div = {'aborted': True, 'reason': ended.split(' ')[0], 'detail': ended, 'step': int(T - 1),
               'time_s': float('nan'), 'physics_step': -1}
    return {'plans': plans, 'obs_traj': obs_traj, 'success': {'strict': bool((sc or {}).get('n_success', 0))},
            'divergence': div or {'aborted': False}}


def write_foresight_svg(diag_dir, cell, i, rows, fans, geo, scale, sc, ep_rec=None, stride=6, altitude=None):
    """The UAV suite's own MPC-foresight SVG (mix_uav_test/eval_artifacts.write_mpc_foresight), unchanged: LEFT XY
    top-down with the green candidate fan / black p_des / red drone path, RIGHT XZ altitude — fed with the replay
    packed into its rollout schema and the avoiding geometry mapped into the world."""
    from mix_uav_test.eval_artifacts import write_mpc_foresight
    alt = F.ALTITUDE if altitude is None else altitude
    rollout = uav_rollout(rows, fans, ep_rec, sc, scale, alt)
    return write_mpc_foresight(diag_dir, i, rollout, f"pillars_v2@{F.scene_tag(scale)}", stride=stride,
                               geo_config=uav_geo_config(geo, scale), variant=cell['variant'])


def write_eval_log(path, cell, summ, per, settings):
    p, d, pl = summ['panda'], summ['drone'], summ['plant']
    L = [f"------------------------ turbo replay {cell['engine']} | {cell['eval']} | seed {cell['seed']} | {cell['geo']} | {cell['variant']} ----------------------------",
         f"source npz : {summ.get('src_npz', '')}",
         f"plant      : {settings}",
         f"n episodes : {summ['n']}   wall {summ['wall_s']:.1f} s",
         '',
         f"PANDA  Success rate: {p['success']:.2f}   Constraints satisfied: {p['cf']:.2f}   Success (goal and constraints): {p['sc']:.2f}   "
         f"Avg violations: {p['n_viol']:.2f}   Avg steps: {p['n_steps']:.1f}",
         f"DRONE  Success rate: {d['success']:.2f}   Constraints satisfied: {d['cf']:.2f}   Success (goal and constraints): {d['sc']:.2f}   "
         f"Avg violations: {d['n_viol']:.2f}   Avg steps: {d['n_steps']:.1f}",
         f"agree (success, cf) on {summ['agree_episodes']}/{summ['n']} episodes   ended {summ['ended']}",
         f"plant  track_err mean {pl['track_err_mean_m']:.3f} m  p95 {pl['track_err_p95_m']:.3f} m   gap_a p95 {pl['gap_a_p95']:.4f} "
         f"(panda {pl['panda_gap_a_p95']:.4f})   contact episodes {pl['contact_episodes']}   |v|max {pl['speed_max_ms']:.2f} m/s",
         '', 'ep  ended      succ  S&C  cf  viol  total_viol  steps  src_len  grace']
    for i, sc in enumerate(per):
        if sc is None:
            L.append(f'{i:2d}  (empty source episode)'); continue
        L.append(f"{i:2d}  {sc['ended']:9s}  {int(sc['n_success'])}     {int(sc['n_success_and_constraints'])}    {int(sc['collision_free_completed'])}   "
                 f"{int(sc['n_violations']):3d}   {sc['total_violations']:8.4f}   {int(sc['n_steps']):4d}   {sc['src_len']:4d}     {sc['grace_used']}")
    with open(path, 'w') as fh:
        fh.write('\n'.join(L) + '\n')
    return path
