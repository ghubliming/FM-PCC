#!/usr/bin/env python
"""Mode T — bridging-turbo (Gen15 U18): replay STORED avoiding executions through the quadrotor plant.

Every avoiding results npz (all five arms) stores `obs_all`: per episode the executed [x_des, y_des, x, y]
sequence, i.e. the setpoint path the planner + projector produced and the Panda tracked. This script flies that
setpoint sequence with the drone in the scaled pillars_v2 scene, rescores the DRONE's path with the avoiding
scorer (scoring.py) and writes a mirrored results tree the existing loaders / DA pick up unchanged:

    logs/avoiding-d3il/plans/<engine>/<train>/<eval>/<seed>/results/halfspace_<geo>/<variant>.npz   (source, read only)
    <out-root>/<engine>/<train>/<eval>_msg<tag>/<seed>/results/halfspace_<geo>/<variant>.npz        (output, same keys)
                                                                        + <variant>.png              (avoiding-style cell figure)
                                                                        + eval_<variant>.log         (summary + per-episode table)
                                                                        + <variant>_uav_plant.json   (sidecar, plant stats)
                                                                        + <variant>_uav_world.png    (world-frame paths)
                                                                        + diagnostics/<variant>/rollout_<i>_mpc_foresight.svg
                                                                        (+ rollout_<i>.gif only with --gif on a GPU node)
    <out-root> defaults to logs/UAV_MIX/uav-pillars/plans/avoiding_bridge  (fix2: it IS the UAV pillars scene, so the
    results live with the other UAV-pillars runs; the avoiding-style relative layout below it is kept so the avoiding
    loaders / batch reporter work when pointed at <out-root>). --in-place restores the beside-the-source layout.

No network, no NLP, no GPU. avg_time is COPIED from the source (planner time; the plant's time is not a metric).

    python uav_avoiding_bridge/turbo.py --dry-run                                    # list what would be replayed
    python uav_avoiding_bridge/turbo.py --eval-glob '*K20*msg20trials' --engine flow_matching_v3_ode_selectable \
        --seeds 6 --geos both-hard --variants diffuser dpcc-r-tightened               # the G1 pilot cell
    python uav_avoiding_bridge/turbo.py                                              # the whole corpus

Replay semantics: one stored setpoint per control period (clock) or per settle (settle); stop at finish crossed,
drone-pillar contact, divergence, or the env cap. When the stored sequence is exhausted before the finish, the last
setpoint is HELD for --grace-s seconds (the Panda arrived inside the same step; a lagging drone gets that long).
n_steps = flown periods - 1, the source's convention.
"""
import argparse
import fnmatch
import glob
import json
import os
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from uav_avoiding_bridge import frame as F                       # noqa: E402
from uav_avoiding_bridge.scoring import EpisodeScorer, geometry_for, load_projection_cfg   # noqa: E402

RESCORED = ('n_success', 'n_success_and_constraints', 'n_steps', 'n_violations', 'total_violations',
            'collision_free_completed', 'obs_all')


# ── discovery ─────────────────────────────────────────────────────────────────────────────────────
def discover(root, engines, train_glob, eval_glob, seeds, geos, variants, tag, out_root=None):
    cells = []
    for npz in sorted(glob.glob(os.path.join(root, '**', 'results', 'halfspace_*', '*.npz'), recursive=True)):
        hv_dir = os.path.dirname(npz)
        seed_dir = os.path.dirname(os.path.dirname(hv_dir))
        eval_dir = os.path.dirname(seed_dir)
        train_dir = os.path.dirname(eval_dir)
        rel = os.path.relpath(train_dir, root)
        engine = rel.split(os.sep)[0]
        ev = os.path.basename(eval_dir)
        if 'uav' in ev.lower() or ev.endswith(tag):            # never read our own or a live-UAV output as a source
            continue
        variant = os.path.splitext(os.path.basename(npz))[0]
        geo = os.path.basename(hv_dir)[len('halfspace_'):]
        try:
            seed = int(os.path.basename(seed_dir))
        except ValueError:
            continue
        if engines and not any(fnmatch.fnmatch(engine, e) or e in engine for e in engines):
            continue
        if train_glob and not fnmatch.fnmatch(os.path.basename(train_dir), train_glob):
            continue
        if eval_glob and not any(fnmatch.fnmatch(ev, g) for g in eval_glob):
            continue
        if seeds and seed not in seeds:
            continue
        if geos and geo not in geos:
            continue
        if variants and variant not in variants:
            continue
        out_eval = f'{ev}-{tag}' if '_msg' in ev else f'{ev}_msg{tag}'
        out_train = train_dir if out_root is None else os.path.join(out_root, rel)     # fix2: mirror under out_root
        out_npz = os.path.join(out_train, out_eval, str(seed), 'results', f'halfspace_{geo}', f'{variant}.npz')
        cells.append({'npz': npz, 'out': out_npz, 'engine': engine, 'train': os.path.basename(train_dir),
                      'eval': ev, 'seed': seed, 'geo': geo, 'variant': variant})
    return cells


# ── replay of one episode ─────────────────────────────────────────────────────────────────────────
def replay_episode(plant, setpoints, actions, scorer, grace_steps, grace_mode='extend'):
    plant.reset()
    obs_reset = plant.obs4().astype(float)          # [start, start]
    rows, acts, success, ended, grace_used = [], [], False, 'exhausted', 0
    mode = np.zeros(9)
    n_src = len(setpoints)
    k = 0
    while True:
        if k < n_src:
            sp, a = setpoints[k], (actions[k] if k < len(actions) else np.zeros(2))
        elif grace_used < grace_steps:
            # fix4: 'extend' continues the LAST COMMANDED INCREMENT (the Panda arm crosses the finish line with the
            # momentum of its last step; a drone parked on a setpoint 5 mm short never does). 'hold' parks.
            if grace_mode == 'extend' and n_src >= 2:
                d_last = np.asarray(setpoints[-1], float) - np.asarray(setpoints[-2], float)
                sp = np.asarray(setpoints[-1], float) + d_last * (grace_used + 1)
                a = d_last
            else:
                sp, a = setpoints[-1], np.zeros(2)
            grace_used += 1
        else:
            break
        obs, _, term, (mode, succ) = plant.step(np.concatenate((sp, [0.12, 0, 1, 0, 0])))
        rows.append([float(sp[0]), float(sp[1]), float(obs[0]), float(obs[1])])
        acts.append(np.asarray(a, float)[:2])
        k += 1
        if succ:
            success, ended = True, 'success'
            break
        if term:
            ended = plant.last_reason
            break
    if not rows:                                     # empty source episode — nothing flown
        rows.append(obs_reset.tolist())
    checked = np.vstack([obs_reset[None, :], np.asarray(rows[:-1], float)]) if len(rows) > 1 else obs_reset[None, :]
    sc = scorer.score(checked, acts, success)
    sc.update({'n_success': float(success), 'n_steps': float(len(rows) - 1), 'ended': ended,
               'grace_used': grace_used, 'flown': len(rows), 'src_len': n_src, 'mode': mode.tolist()})
    return np.asarray(rows, np.float32), sc


# ── one cell ──────────────────────────────────────────────────────────────────────────────────────
def run_cell(cell, plant, cfg, args):
    src = np.load(cell['npz'], allow_pickle=True)
    if 'obs_all' not in src.files:
        return None, f'no obs_all in {cell["npz"]}'
    obs_all, act_all = src['obs_all'], src['act_all'] if 'act_all' in src.files else None
    n = len(obs_all)
    if args.limit_episodes:
        n = min(n, args.limit_episodes)
    scorer = EpisodeScorer(geometry_for(cell['geo'], cfg))
    grace_steps = int(round(args.grace_s * plant.control_hz)) if plant.replay == 'clock' else int(round(args.grace_s))
    plant.records = []
    plant.episode = -1                      # fix4: sidecar episode ids are per cell (were running across cells)
    per, new_obs = [], []
    t0 = time.perf_counter()
    for i in range(n):
        o = np.asarray(obs_all[i], float)
        if o.ndim != 2 or o.shape[0] == 0:
            per.append(None); new_obs.append(np.zeros((0, 4), np.float32)); continue
        a = np.asarray(act_all[i], float) if act_all is not None and len(act_all) > i else np.zeros((0, 2))
        plant.gif_on = bool(args.gif and i < args.gif and plant._renderer is not None)
        rows, sc = replay_episode(plant, o[:, :2], a, scorer, grace_steps, args.grace_mode)
        per.append(sc); new_obs.append(rows)
        if plant.gif_on:
            from mix_uav_test.eval_artifacts import save_rollout_gif      # the UAV evals' writer (imageio, no torch)
            gif_dir = os.path.join(os.path.dirname(cell['out']), 'diagnostics', cell['variant'])
            gp = save_rollout_gif(gif_dir, i, plant.pop_frames(), fps=args.gif_fps)
            if gp:
                print(f'[ turbo ]   gif ep{i}: {gp}')
    plant.close()
    wall = time.perf_counter() - t0

    # ── assemble the output npz: every source key copied, the rescored ones replaced ──
    # when --limit-episodes truncates, the output carries ONLY the replayed episodes (never zero-padding)
    out = {k: src[k] for k in src.files}
    def col(key, default=0.0):
        return np.array([(p[key] if p is not None else default) for p in per], float)
    out['n_success'] = col('n_success')
    out['n_success_and_constraints'] = col('n_success_and_constraints')
    out['n_steps'] = col('n_steps')
    out['n_violations'] = col('n_violations')
    out['total_violations'] = col('total_violations')
    out['collision_free_completed'] = col('collision_free_completed')
    out['obs_all'] = np.array(new_obs, dtype=object)
    if act_all is not None:
        out['act_all'] = np.array([act_all[i] for i in range(n)], dtype=object)
    if 'sampled_trajectories_all' in src.files and n < len(obs_all):
        out['sampled_trajectories_all'] = np.array([src['sampled_trajectories_all'][i] for i in range(n)], dtype=object)
    for k in ('n_success', 'n_success_and_constraints', 'n_steps', 'n_violations', 'total_violations',
              'collision_free_completed', 'avg_time'):
        if k in src.files:
            out[f'src_{k}'] = np.asarray(src[k], float)[:n]
    out['avg_time'] = np.asarray(src['avg_time'], float)[:n]          # planner time, copied: the plant is not timed
    out['src_npz'] = cell['npz']
    out['uav_bridge'] = json.dumps({'mode': 'turbo', 'tag': args.tag, 'grace_s': args.grace_s, 'grace_mode': args.grace_mode, 'limit_episodes': args.limit_episodes,
                                    'settings': plant.settings(), 'wall_s': wall, 'n_replayed': n})
    os.makedirs(os.path.dirname(cell['out']), exist_ok=True)
    np.savez(cell['out'], **out)

    # ── sidecar + summary ──
    src_s, src_sc = np.asarray(src['n_success'], float)[:n], np.asarray(src['n_success_and_constraints'], float)[:n]
    drone_s, drone_sc = out['n_success'][:n], out['n_success_and_constraints'][:n]
    agree = int(np.sum((src_s == drone_s) & (src_sc == drone_sc)))
    recs = plant.records
    summ = {
        'cell': {k: cell[k] for k in ('engine', 'train', 'eval', 'seed', 'geo', 'variant')},
        'n': n, 'wall_s': wall,
        'panda': {'success': float(src_s.mean()), 'sc': float(src_sc.mean()),
                  'cf': float(np.asarray(src['collision_free_completed'], float)[:n].mean()),
                  'n_viol': float(np.asarray(src['n_violations'], float)[:n].mean()),
                  'n_steps': float(np.asarray(src['n_steps'], float)[:n].mean())},
        'drone': {'success': float(drone_s.mean()), 'sc': float(drone_sc.mean()),
                  'cf': float(out['collision_free_completed'][:n].mean()),
                  'n_viol': float(out['n_violations'][:n].mean()), 'n_steps': float(out['n_steps'][:n].mean())},
        'agree_episodes': agree,
        'ended': dict({e: sum(1 for p in per if p and p['ended'] == e) for e in ('success', 'contact', 'exhausted', 'cap')},
                      diverged=sum(1 for p in per if p and p['ended'] not in ('success', 'contact', 'exhausted', 'cap'))),
        'plant': {'track_err_mean_m': float(np.mean([r['summary']['track_err_mean_m'] for r in recs])) if recs else float('nan'),
                  'track_err_p95_m': float(np.max([r['summary']['track_err_p95_m'] for r in recs])) if recs else float('nan'),
                  'gap_a_p95': float(np.max([r['summary']['gap_a_p95'] for r in recs])) if recs else float('nan'),
                  'gap_a_mean': float(np.mean([r['summary']['gap_a_mean'] for r in recs])) if recs else float('nan'),
                  'panda_gap_a_p95': float(np.percentile(np.concatenate(
                      [np.linalg.norm(np.asarray(obs_all[i], float)[:, :2] - np.asarray(obs_all[i], float)[:, 2:4], axis=1)
                       for i in range(n) if np.asarray(obs_all[i]).ndim == 2 and len(obs_all[i])]), 95)) if n else float('nan'),
                  'contact_episodes': sum(1 for r in recs if r['summary']['contact_steps'] > 0),
                  'speed_max_ms': float(max(r['summary']['speed_max_ms'] for r in recs)) if recs else float('nan')},
        'per_episode': per,
    }
    summ['src_npz'] = cell['npz']
    side = cell['out'][:-4] + '_uav_plant.json'
    plant.save_records(side, extra={'summary': {k: v for k, v in summ.items() if k != 'per_episode'},
                                    'per_episode': per})
    # ── the standard suite (no GL): <variant>.png, eval_<variant>.log, world png, foresight SVGs ──
    from uav_avoiding_bridge import artifacts as A
    out_dir = os.path.dirname(cell['out'])
    fans_all = src['sampled_trajectories_all'] if 'sampled_trajectories_all' in src.files else None
    fans_list = [A._fans(fans_all[i]) if fans_all is not None and i < len(fans_all) else [] for i in range(n)]
    panda_list = [np.asarray(obs_all[i], float) if np.asarray(obs_all[i]).ndim == 2 else np.zeros((0, 4)) for i in range(n)]
    try:
        A.write_eval_log(os.path.join(out_dir, f"eval_{cell['variant']}.log"), cell, summ, per, plant.settings())
    except Exception as exc:                                         # pragma: no cover
        print(f'[ turbo ] eval log failed: {exc}')
    if not args.no_png:
        try:
            A.save_cell_png(cell['out'][:-4] + '.png', cell, scorer.geo, new_obs[:n], panda_list, fans_list, per[:n])
            world_png(cell, new_obs[:n], per[:n], plant, scorer.geo, cell['out'][:-4] + '_uav_world.png')
        except Exception as exc:                                     # pragma: no cover
            print(f'[ turbo ] png failed: {exc}')
    if args.foresight > 0:
        diag = os.path.join(out_dir, 'diagnostics', cell['variant'])
        os.makedirs(diag, exist_ok=True)
        recs_by_ep = {r['episode']: r for r in plant.records}
        for i in range(min(n, args.foresight)):
            if per[i] is None:
                continue
            try:
                A.write_foresight_svg(diag, cell, i, new_obs[i], fans_list[i], scorer.geo, plant.scale, per[i],
                                      recs_by_ep.get(i), stride=args.foresight_stride, altitude=plant.altitude)
            except Exception as exc:                                 # pragma: no cover
                print(f'[ turbo ] foresight svg ep{i} failed: {exc}')
    return summ, None


# ── world-frame picture: physical pillars + mapped constraints + flown paths ──────────────────────
def world_png(cell, obs_list, per, plant, geo, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle
    s = plant.scale
    fig, ax = plt.subplots(figsize=(9, 7))
    (x0, x1), (y0, y1) = F.world_field(s)
    for n_, xa, ya, r in F.OBSTACLES_A:
        X, Y = F.to_world_xy(xa, ya, s)
        ax.add_patch(Circle((X, Y), s * r, color='#d98c40', zorder=3))
        ax.add_patch(Circle((X, Y), s * r + F.DRONE_REACH_M, color='#d98c40', alpha=0.15, zorder=2))     # + rotor reach
    for ob in geo['obstacles']:
        X, Y = F.to_world_xy(*ob['center'], s)
        ax.add_patch(Circle((X, Y), s * ob['radius'], color='b', alpha=0.15, zorder=2))
        ax.add_patch(Circle((X, Y), s * (ob['radius'] + geo['enlarge']), fill=False, ls='--', color='b', alpha=0.5, zorder=2))
    for c in geo['polytopic']:
        p0, p1 = c[0], c[1]
        m = (p1[1] - p0[1]) / (p1[0] - p0[0])
        xs = np.linspace(F.FIELD_X_A[0] - 0.1, F.FIELD_X_A[1] + 0.1, 50)
        ys = p0[1] + m * (xs - p0[0])
        W = np.array([F.to_world_xy(x, y, s) for x, y in zip(xs, ys)])
        ax.plot(W[:, 0], W[:, 1], 'b-', lw=2, alpha=0.6, zorder=2)
        # shade the forbidden side lightly: sample field points, keep those violating
        gx, gy = np.meshgrid(np.linspace(*F.FIELD_X_A, 60), np.linspace(*F.FIELD_Y_A, 70))
        val = gy - (p0[1] + m * (gx - p0[0]))
        bad = (val >= 0) if c[2] == 'below' else (val <= 0)
        Wg = np.array([F.to_world_xy(x, y, s) for x, y in zip(gx[bad], gy[bad])])
        if len(Wg):
            ax.scatter(Wg[:, 0], Wg[:, 1], s=4, color='b', alpha=0.06, zorder=1)
    fx = F.to_world_xy(0.5, F.GOAL_Y_A, s)[0]
    ax.axvline(fx, color='g', lw=2, alpha=0.6)
    sx, sy = F.to_world_xy(*F.START_A, s)
    ax.plot(sx, sy, 'ko', ms=6)
    for rows, sc in zip(obs_list, per):
        if sc is None or len(rows) == 0:
            continue
        W = np.array([F.to_world_xy(r[2], r[3], s) for r in rows])
        Wd = np.array([F.to_world_xy(r[0], r[1], s) for r in rows])
        col = 'g' if sc['n_success_and_constraints'] else ('orange' if sc['n_success'] else 'r')
        ax.plot(Wd[:, 0], Wd[:, 1], color='gray', lw=0.6, alpha=0.4, zorder=4)
        ax.plot(W[:, 0], W[:, 1], color=col, lw=1.2, alpha=0.9, zorder=5)
    ax.set_xlim(x0 - 0.5, x1 + 0.5); ax.set_ylim(y0 - 0.5, y1 + 0.5); ax.set_aspect('equal')
    ax.set_title(f"{cell['engine']} | {cell['eval']}\nseed {cell['seed']} · {cell['geo']} · {cell['variant']} · "
                 f"drone paths (green S&C, orange success only, red fail); gray = stored setpoints", fontsize=9)
    ax.set_xlabel('X [m]  (avoiding +y)'); ax.set_ylabel('Y [m]  (avoiding -x)')
    fig.savefig(path, dpi=110, bbox_inches='tight'); plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--root', default=os.path.join(REPO, 'logs', 'avoiding-d3il', 'plans'), help='where the avoiding sources are')
    ap.add_argument('--out-root', default=os.path.join(REPO, 'logs', 'UAV_MIX', 'uav-pillars', 'plans', 'avoiding_bridge'),
                    help='fix2: outputs mirror the source layout below this root (the UAV-pillars scene folder)')
    ap.add_argument('--in-place', action='store_true', help='write beside the sources instead (pre-fix2 layout)')
    ap.add_argument('--engine', action='append', default=[], help='engine folder under plans/ (substring or glob), repeatable')
    ap.add_argument('--train-glob', default=None, help='fnmatch on the TRAIN folder name')
    ap.add_argument('--eval-glob', action='append', default=[], help='fnmatch on the EVAL folder name, repeatable (OR)')
    ap.add_argument('--seeds', type=int, nargs='*', default=[])
    ap.add_argument('--geos', nargs='*', default=[])
    ap.add_argument('--variants', nargs='*', default=[])
    ap.add_argument('--tag', default=f'uavpv2{F.scene_tag()}turbo', help='output eval-folder tag (must contain "uav")')
    ap.add_argument('--replay', choices=('clock', 'settle'), default='clock')
    ap.add_argument('--hz', type=float, default=1.0, help='clock mode: setpoints per second (fix3 default 1 at scale 36)')
    ap.add_argument('--vmax', type=float, default=1.0, help='clock mode: rate limit of the tracked reference [m/s]')
    ap.add_argument('--ff', action='store_true', help='clock mode: enable velocity feed-forward (unstable above ~0.5 m/s with this PID; off by default since fix3)')
    ap.add_argument('--gain', default='pid_default')
    ap.add_argument('--no-contact-stop', action='store_true', help='do not end the episode on drone-pillar contact')
    ap.add_argument('--grace-s', type=float, default=2.0, help='after the stored sequence ends: keep flying this long (clock: seconds; settle: steps)')
    ap.add_argument('--grace-mode', choices=('extend', 'hold'), default='extend',
                    help="extend = continue the last commanded increment (Panda momentum), hold = park on the last setpoint")
    ap.add_argument('--limit-episodes', type=int, default=0, help='replay only the first N episodes of each cell (pilot)')
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--force', action='store_true', help='re-run cells whose output npz exists')
    ap.add_argument('--no-png', action='store_true')
    ap.add_argument('--max-cells', type=int, default=0)
    ap.add_argument('--foresight', type=int, default=3, help='write diagnostics/rollout_<i>_mpc_foresight.svg for the first N episodes per cell (0 = off)')
    ap.add_argument('--foresight-stride', type=int, default=6, help='draw the stored candidate fan every N steps')
    ap.add_argument('--gif', type=int, default=0, help='record an overhead MuJoCo GIF for the first N episodes of each cell '
                    '(needs MUJOCO_GL=egl on a GPU node; writes <cell>/diagnostics/<variant>/rollout_<i>.gif)')
    ap.add_argument('--gif-fps', type=int, default=10)
    ap.add_argument('--gif-res', type=int, default=160)
    ap.add_argument('--gif-cam', type=float, default=8.0, help='camera height above the drone [m]')
    ap.add_argument('--gif-stride', type=int, default=20, help='physics steps per frame (20 = 5 sim-fps at dt 0.01)')
    args = ap.parse_args()
    if 'uav' not in args.tag.lower():
        sys.exit('--tag must contain "uav" (it keeps the outputs apart from the Panda results)')

    out_root = None if args.in_place else args.out_root
    cells = discover(args.root, args.engine, args.train_glob, args.eval_glob, set(args.seeds), set(args.geos),
                     set(args.variants), args.tag, out_root)
    todo = [c for c in cells if args.force or not os.path.exists(c['out'])]
    if args.max_cells:
        todo = todo[:args.max_cells]
    print(f'[ turbo ] sources={args.root}\n[ turbo ] outputs={out_root or "(in place, beside the sources)"}\n'
          f'[ turbo ] {len(cells)} matching cells, {len(todo)} to run '
          f'({len(cells) - len(todo)} already have {args.tag} output)  tag={args.tag} replay={args.replay} '
          f'hz={args.hz:g} vmax={args.vmax:g} ff={args.ff} grace={args.grace_s:g}s limit={args.limit_episodes or "all"}')
    print(F.describe())
    for c in todo[:60]:
        print(f"   {c['engine']:38s} {c['eval'][:60]:60s} s{c['seed']} {c['geo']:15s} {c['variant']}")
    if len(todo) > 60:
        print(f'   … {len(todo) - 60} more')
    if args.dry_run or not todo:
        return

    from uav_avoiding_bridge.plant import UavAvoidingPlant
    plant = UavAvoidingPlant(control_hz=args.hz, feedforward=args.ff, gain=args.gain, replay=args.replay,
                             terminate_on_contact=not args.no_contact_stop, v_max=args.vmax)
    if args.gif > 0:
        ok = plant.enable_gif(res=args.gif_res, cam_distance=args.gif_cam, frame_stride=args.gif_stride)
        print(f'[ turbo ] GIF: first {args.gif} episodes per cell, {args.gif_res}px, cam {args.gif_cam:g} m, '
              f'stride {args.gif_stride} -> {"on" if ok else "UNAVAILABLE (no GL context)"}')
    cfg = load_projection_cfg()
    summaries, t_all = [], time.perf_counter()
    for j, c in enumerate(todo, 1):
        summ, err = run_cell(c, plant, cfg, args)
        if err:
            print(f'[ turbo ] {j}/{len(todo)} SKIP {err}')
            continue
        p, d, pl = summ['panda'], summ['drone'], summ['plant']
        print(f"[ turbo ] {j}/{len(todo)} {c['engine']} | {c['eval'][:48]} | s{c['seed']} {c['geo']} {c['variant']}  "
              f"n={summ['n']} {summ['wall_s']:.0f}s\n"
              f"           panda  succ {p['success']:.2f}  S&C {p['sc']:.2f}  cf {p['cf']:.2f}  viol {p['n_viol']:.1f}  steps {p['n_steps']:.1f}\n"
              f"           drone  succ {d['success']:.2f}  S&C {d['sc']:.2f}  cf {d['cf']:.2f}  viol {d['n_viol']:.1f}  steps {d['n_steps']:.1f}"
              f"   agree {summ['agree_episodes']}/{summ['n']}  ended {summ['ended']}\n"
              f"           plant  track_err mean {pl['track_err_mean_m']:.3f} m  p95 {pl['track_err_p95_m']:.3f} m  "
              f"gap_a p95 {pl['gap_a_p95']:.4f} (panda {pl['panda_gap_a_p95']:.4f})  contact eps {pl['contact_episodes']}  "
              f"vmax {pl['speed_max_ms']:.2f} m/s", flush=True)
        summaries.append({k: v for k, v in summ.items() if k != 'per_episode'})
    run_dir = os.path.join(out_root or args.root, '_uav_turbo_runs')
    os.makedirs(run_dir, exist_ok=True)
    stamp = time.strftime('%Y%m%d_%H%M%S')
    with open(os.path.join(run_dir, f'turbo_{args.tag}_{stamp}.json'), 'w') as fh:
        json.dump({'args': vars(args), 'settings': plant.settings(), 'cells': summaries,
                   'wall_s': time.perf_counter() - t_all}, fh, indent=1)
    print(f'[ turbo ] done: {len(summaries)} cells in {time.perf_counter() - t_all:.0f}s -> {run_dir}')


if __name__ == '__main__':
    main()
