#!/usr/bin/env python3
"""
npz_traj_export.py — rebuild a whole eval scene from its .npz files into a compact scene.json
and a self-contained interactive viewer HTML (npz_traj_visualizer).

See PLAN_npz_traj_visualizer.md (this folder) for the full design. In short: for one scene
(a results dir holding <variant>.npz flat, or <variant>/<variant>.npz nested), it extracts —
per variant, per trial — the executed closed-loop path (obs_all), the H-step receding-horizon
MPC plan fan at (decimated) steps, and PRECOMPUTES the analytics the browser shouldn't
(violation fraction, divergence-from-reference, tracking error, explosion max, candidate spread,
plus the npz scalar-metric stat row). Output: <scene_dir>/_traj_viz/{scene.json, viewer_<scene>.html}.

Reuses npz_analysis/compare_horizon_plans.py (find_variant_npz, plan_snapshots,
load_task_constraints, halfspace/obstacle violation maths). numpy + pyyaml only — never draws.

Usage:
    python npz_traj_export.py <scene_dir> [--env avoiding|uav|unknown] [--trials 0|all|0,3,7]
        [--plan-every N] [--variants a,b,c] [--reference diffuser]
        [--halfspace-variant both-hard|top-left-hard|top-right-hard]
        [--config config/projection_eval.yaml] [--out DIR] [--no-embed]

Column conventions (col_map = ACTUAL position; see D11 in the plan):
  avoiding: obs=[x_des,y_des,x,y]        → x,y = cols 2,3          (2D)
  uav:      obs=[p_des(0:3)|p(3:6)]      → x,y,z = cols 3,4,5      (3D → x-y + x-z panels)
            (the p_des_z runaway in col 2 is caught by the executed-explosion curve, not the scene)
"""

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

# ── reuse the proven helpers from the sibling tool ───────────────────────────
# Layout: code lives in npz_analysis/npz_traj_visualizer/ (logs_in_develop is MD/logs ONLY).
_HERE = os.path.dirname(os.path.abspath(__file__))          # .../npz_analysis/npz_traj_visualizer
_NPZ_ANALYSIS = os.path.abspath(os.path.join(_HERE, '..'))  # .../npz_analysis
_REPO = os.path.abspath(os.path.join(_HERE, '..', '..'))    # repo root
if _NPZ_ANALYSIS not in sys.path:
    sys.path.insert(0, _NPZ_ANALYSIS)
import compare_horizon_plans as chp  # noqa: E402  (argparse guarded under __main__ → safe import)

_TEMPLATE = os.path.join(_HERE, 'npz_traj_visualizer.html')

# Extended, stable palette (sorted-name assignment → stable colours across sessions).
_PALETTE = ['#000000', '#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4', '#42d4f4',
            '#f032e6', '#9A6324', '#808000', '#469990', '#800000', '#000075', '#a9a9a9',
            '#e6beff', '#fabed4', '#aaffc3', '#ffd8b1', '#dcbeff', '#fffac8']


def r4(a):
    """Round to 4 dp and make JSON-safe (lists, no numpy types, NaN→None)."""
    arr = np.asarray(a, dtype=float)
    out = np.where(np.isfinite(arr), np.round(arr, 4), np.nan)
    return _nan_to_none(out.tolist())


def _nan_to_none(x):
    if isinstance(x, list):
        return [_nan_to_none(v) for v in x]
    if isinstance(x, float) and (x != x):
        return None
    return x


def parse_trials(spec, n_trials):
    if spec == 'all':
        return list(range(n_trials))
    out = []
    for tok in str(spec).split(','):
        tok = tok.strip()
        if tok.isdigit() and int(tok) < n_trials:
            out.append(int(tok))
    return out or [0]


def infer_env(plans_last_dim, obs_dim):
    if plans_last_dim == 6 or obs_dim == 6:
        return 'uav'
    if plans_last_dim == 4 or obs_dim == 4:
        return 'avoiding'
    return 'unknown'


def env_col_map(env, obs_dim):
    if env == 'uav':
        return {'x': 3, 'y': 4, 'z': 5}, [('xy', 0, 1), ('xz', 0, 2)], 3
    if env == 'avoiding':
        return {'x': 2, 'y': 3}, [('xy', 0, 1)], 2
    # unknown: best-effort first two dims
    return {'x': 0, 'y': 1}, [('xy', 0, 1)], 2


def load_variant(path, trials):
    """Return dict: {metrics:{}, trials:{t:{obs:(T,D), plans:[snaps [B,H,d]]}}} or None on failure."""
    try:
        data = np.load(path, allow_pickle=True)
    except Exception as exc:
        print(f'  ! {os.path.basename(path)}: load failed ({type(exc).__name__}) — skipped')
        return None
    out = {'metrics': {}, 'trials': {}}
    files = set(data.files)
    obs_all = data['obs_all'] if 'obs_all' in files else None
    plans_all = data['sampled_trajectories_all'] if 'sampled_trajectories_all' in files else None
    # scalar-metric harvest (analyze_npz convention: any 1-D numeric array = per-trial metric)
    for k in files:
        if k in ('obs_all', 'act_all', 'sampled_trajectories_all', 'args'):
            continue
        v = data[k]
        if isinstance(v, np.ndarray) and v.dtype != object and np.issubdtype(v.dtype, np.number):
            flat = np.atleast_1d(v).ravel().astype(float)
            if flat.size:
                out['metrics'][k] = flat  # keep full per-trial array; row-select later
    for t in trials:
        rec = {}
        if obs_all is not None and t < len(obs_all):
            try:
                rec['obs'] = np.asarray(obs_all[t], dtype=float)
            except Exception:
                rec['obs'] = None
        if plans_all is not None and t < len(plans_all):
            rec['plans'] = chp.plan_snapshots(plans_all[t])
        out['trials'][t] = rec
    data.close()
    return out


def mean_plan(snap, dims):
    """Mean over candidates of the exported position dims → [H, len(dims)]."""
    return snap[:, :, dims].mean(axis=0)


_CLEAN_TOL = 1e-2   # ‖h0 - executed‖ below this ⇒ the fan's h=0 lies ON the executed path


def _nearest_executed(m, ex, guess):
    """Nearest executed sample to the plan's h=0 within ±2 of `guess`; returns (index, residual).

    Ties (to 6dp — data is stored at 4dp) break toward `guess` so a stalled robot (many executed
    samples equal to h=0) does not drift off the nominal step.
    """
    T = ex.shape[0]
    h0 = m[0]
    lo, hi = max(0, guess - 2), min(T - 1, guess + 2)
    best = min(range(lo, hi + 1),
               key=lambda j: (round(float(np.linalg.norm(ex[j] - h0)), 6), abs(j - guess)))
    return best, float(np.linalg.norm(ex[best] - h0))


def _recording_offset(guesses, executed, mlist, dims):
    """The single structural step-offset between the plan's conditioning and the executed buffer.

    Every plan is conditioned on a specific obs (``conditions={0: obs}`` → h=0 === that obs), but the
    eval loops record the executed buffer with DIFFERENT phase relative to that conditioning:
      • avoiding/imf (eval_flow_matching_v3_imeanflow.py): appends obs_buffer AFTER env.step advances
        it, and never stores the initial reset obs → executed is shifted +1 control step (offset 1).
      • uav (eval_fm_uav.py): appends obs_traj BEFORE the state update → already aligned (offset 0).
    We DETECT this rather than hard-code it: for the fans whose h=0 lands cleanly on the executed path
    (residual < _CLEAN_TOL — verified dist == 0 for projected fans), take the MODE of nominal-minus-
    matched-index. The offset is a fixed property of the loop, so applying that one value uniformly
    keeps later divergence (an unprojected raw plan drifting OFF the path) VISIBLE instead of snapping
    each fan onto whatever dot is nearest. Returns (offset, off_path_fraction) for a caller sanity log.
    """
    if executed is None or executed.ndim != 2 or max(dims) >= executed.shape[1]:
        return 0, 0.0
    ex = executed[:, dims]
    clean = []
    off_path = 0
    for g, m in zip(guesses, mlist):
        idx, rd = _nearest_executed(m, ex, g)
        if rd <= _CLEAN_TOL:
            clean.append(g - idx)
        else:
            off_path += 1
    frac = off_path / max(1, len(guesses))
    if not clean:
        return 0, frac                    # nothing landed on the path (see caller's WARN)
    vals, cnts = np.unique(clean, return_counts=True)
    return int(vals[int(np.argmax(cnts))]), frac


def snapshot_analytics(snaps, ref_means, executed, dims, all_dims, polys, obsts, xcol_local):
    """Per-snapshot arrays aligned to the executed step where each fan's h=0 truly starts.

    Nominal step is i*save_every; a single detected recording offset (see ``_recording_offset``) then
    shifts every fan onto its executed dot. ``offset``/``off_path_frac`` are returned for a sanity log.
    """
    n = len(snaps)
    save_every = 1
    if executed is not None and executed.ndim == 2 and n > 1:
        save_every = max(1, int(round((executed.shape[0] - 1) / (n - 1))))
    guesses = [i * save_every for i in range(n)]
    mlist = [mean_plan(s, dims) for s in snaps]                  # [H, d] per snapshot
    offset, off_path_frac = _recording_offset(guesses, executed, mlist, dims)
    div_ref, viol, track, spread, expl, steps = [], [], [], [], [], []
    have_geo = bool(polys or obsts)
    for i, s in enumerate(snaps):
        m = mlist[i]
        step = max(0, guesses[i] - offset)                      # executed idx of the fan's h=0
        steps.append(step)
        # divergence from reference (same ordinal)
        if ref_means is not None and i < len(ref_means) and ref_means[i].shape == m.shape:
            div_ref.append(float(np.linalg.norm(m - ref_means[i], axis=1).mean()))
        else:
            div_ref.append(None)
        # explosion: max|coord| over the whole fan, all plan dims
        expl.append(float(np.nanmax(np.abs(s))))
        # candidate spread: mean pairwise endpoint distance on exported dims
        if s.shape[0] > 1:
            ends = s[:, -1, dims]
            pd = [np.linalg.norm(ends[a] - ends[b]) for a in range(len(ends)) for b in range(a + 1, len(ends))]
            spread.append(float(np.mean(pd)) if pd else None)
        else:
            spread.append(None)
        # tracking error: mean_k ||mean_plan[k] - executed[anchor+k]|| on exported dims
        track.append(_track_err(m, executed, dims, step))
        # violation fraction over all candidate waypoints (avoiding geometry)
        if have_geo:
            xs = s[:, :, dims[0]]
            ys = s[:, :, dims[1]]
            mask = np.zeros(xs.shape, dtype=bool)
            for con in polys:
                mask |= chp.halfspace_violates(xs, ys, con)
            for ob in obsts:
                mask |= chp.obstacle_violates(xs, ys, ob)
            viol.append(float(mask.mean()))
        else:
            viol.append(None)
    return dict(steps=steps, div_ref=div_ref, viol=viol, track=track, spread=spread,
                explosion=expl, save_every=save_every, offset=offset, off_path_frac=off_path_frac)


def _track_err(m, executed, dims, step):
    """mean_k ||plan_mean[k] - executed[step+k]|| on the exported position dims."""
    if executed is None or executed.ndim != 2:
        return None
    T = executed.shape[0]
    H = m.shape[0]
    idx = np.clip(np.arange(step, step + H), 0, T - 1)
    ex = executed[idx][:, dims] if max(dims) < executed.shape[1] else None
    if ex is None:
        return None
    return float(np.linalg.norm(m - ex, axis=1).mean())


def build_scene(scene_dir, args):
    variant_map = chp.find_variant_npz(scene_dir)
    if not variant_map:
        print(f'[export] no .npz under {scene_dir}', file=sys.stderr)
        sys.exit(1)
    if args.variants:
        want = [v.strip() for v in args.variants.split(',') if v.strip()]
        variant_map = {v: variant_map[v] for v in want if v in variant_map} or variant_map

    # peek the first loadable variant to infer env / dims / n_trials
    env, obs_dim, plan_dim, n_trials = args.env, None, None, None
    peek = None
    for v, p in variant_map.items():
        try:
            z = np.load(p, allow_pickle=True)
        except Exception:
            continue
        if 'obs_all' in z.files:
            n_trials = len(z['obs_all'])
            obs_dim = np.asarray(z['obs_all'][0], dtype=float).shape[-1]
        if 'sampled_trajectories_all' in z.files:
            sn = chp.plan_snapshots(z['sampled_trajectories_all'][0])
            if sn:
                plan_dim = sn[0].shape[-1]
        z.close()
        peek = v
        break
    if peek is None:
        print('[export] every variant failed to load (corrupt/truncated npz?)', file=sys.stderr)
        sys.exit(1)
    if env == 'unknown' or env is None:
        env = infer_env(plan_dim or 0, obs_dim or 0)
    col_map, panels_def, ndim = env_col_map(env, obs_dim or 0)
    pos_cols = [col_map['x'], col_map['y']] + ([col_map['z']] if ndim == 3 else [])
    # position lives at pos_cols in BOTH obs_all and the plan tensor (same column layout), so all
    # analytics index snapshots/executed with pos_cols. Exported arrays are then in [x,y(,z)] order.
    dims = pos_cols
    trials = parse_trials(args.trials, n_trials or 1)

    # geometry (avoiding only for now — D5)
    background = {'bounds': None, 'halfspaces': [], 'obstacles': [], 'goal_line': None}
    if env == 'avoiding':
        hv = args.halfspace_variant or chp.infer_halfspace_variant(scene_dir) or 'both-hard'
        try:
            polys, obsts, ax_limits = chp.load_task_constraints(args.config, args.exp, hv)
            background['bounds'] = ax_limits
            for con in polys:
                background['halfspaces'].append({
                    'tri': [list(map(float, con[0])), list(map(float, con[1])),
                            list(map(float, chp._halfspace_third_vertex(con, ax_limits)))],
                    'raw': [list(map(float, con[0])), list(map(float, con[1])), con[2]]})
            for ob in obsts:
                background['obstacles'].append({'center': list(map(float, ob['center'])),
                                                'radius': float(ob['radius'])})
            print(f'[export] avoiding geometry: variant={hv} halfspaces={len(polys)} obstacles={len(obsts)}')
        except Exception as exc:
            polys, obsts = [], []
            print(f'[export] WARNING geometry load failed ({type(exc).__name__}: {exc})')
    else:
        polys, obsts = [], []

    # reference variant's per-snapshot mean plans (for div_ref), per trial
    ref_means = {t: None for t in trials}
    ref = variant_map.get(args.reference)
    if ref:
        rv = load_variant(ref, trials)
        if rv:
            for t in trials:
                snaps = rv['trials'].get(t, {}).get('plans')
                if snaps:
                    ref_means[t] = [mean_plan(s, dims) for s in snaps]

    # decide plan_every (auto: keep ≤ ~300 fans/variant/trial) from the peek variant's snapshot count
    plan_every = args.plan_every
    if not plan_every:
        max_snaps = 0
        rv0 = load_variant(variant_map[peek], trials[:1])
        for t in trials[:1]:
            snaps = rv0['trials'].get(t, {}).get('plans') or []
            max_snaps = max(max_snaps, len(snaps))
        plan_every = max(1, int(np.ceil(max_snaps / 300.0)))
    print(f'[export] env={env} col_map={col_map} panels={[p[0] for p in panels_def]} '
          f'trials={trials} plan_every={plan_every}')

    scene = {
        'scene': os.path.basename(os.path.normpath(scene_dir)),
        'env': env, 'col_map': col_map,
        'panels': [{'id': pid, 'dims': [a, b]} for pid, a, b in panels_def],
        'ndim': ndim, 'plan_every': plan_every, 'reference': args.reference,
        'background': background, 'variants': {},
        'generated': datetime.now().isoformat(timespec='seconds'),
    }

    names = sorted(variant_map)
    color_of = {v: _PALETTE[i % len(_PALETTE)] for i, v in enumerate(names)}
    spot = []  # 3 spot-check values for the P3 acceptance test
    for v in names:
        rv = load_variant(variant_map[v], trials)
        if rv is None:
            continue
        vd = {'color': color_of[v], 'metrics': {}, 'trials': []}
        for k, arr in rv['metrics'].items():
            # store the first exported trial's value (or the scalar) for the stat table
            t0 = trials[0]
            vd['metrics'][k] = float(arr[t0]) if t0 < arr.size else float(arr.mean())
        for t in trials:
            rec = rv['trials'].get(t, {})
            obs = rec.get('obs')
            snaps = rec.get('plans') or []
            if obs is None:
                continue
            # executed path on exported pos cols + full-dim explosion curve (catches col-2 p_des_z)
            exec_pos = obs[:, pos_cols]
            exec_maxabs = np.nanmax(np.abs(obs), axis=1)
            an = snapshot_analytics(snaps, ref_means.get(t), obs, dims, list(range(obs.shape[1])),
                                    polys, obsts, 0)
            # sanity log: detected recording offset + fraction of fans whose h=0 does NOT lie on the
            # executed path. Expect offset 1 for avoiding/imf (post-step obs), 0 for uav (pre-step obs).
            # A high off-path fraction is EXPECTED and correct to show — the raw plan of an unprojected
            # variant, or a projected plan once its solver gives up (s_curve circuit-breaker), really
            # does diverge from the executed path. The WARN fires only when NO fan's h=0 lands on the
            # path (off_path_frac ≈ 1): then the offset is undetectable, pointing at a real coord/
            # conditioning bug rather than mere divergence.
            off_frac = an.get('off_path_frac', 0.0)
            flag = '  <-- WARN: no fan h0 on executed path (offset undetectable — coord bug?)' \
                if off_frac > 0.999 else ''
            print(f'[export] {v}[t{t}] recording offset={an.get("offset")} '
                  f'off_path_frac={off_frac:.2f}{flag}')
            # decimate plans + per-snapshot analytics by plan_every
            keep = list(range(0, len(snaps), plan_every))
            plans_out, psteps = [], []
            for i in keep:
                s = snaps[i]
                plans_out.append(r4(s[:, :, pos_cols]))          # [B,H,ndim]
                psteps.append(an['steps'][i])
            def dec(a):
                return [a[i] for i in keep]
            trial_out = {
                'trial': int(t),
                'executed': r4(exec_pos),
                'exec_maxabs': r4(exec_maxabs),
                'plan_steps': psteps,
                'plans': plans_out,
                'div_ref_per_step': dec(an['div_ref']),
                'viol_frac_per_step': dec(an['viol']),
                'track_err_per_step': dec(an['track']),
                'cand_spread_per_step': dec(an['spread']),
                'explosion_max_per_step': dec(an['explosion']),
                'save_every': an['save_every'],
                'n_steps': int(obs.shape[0]),
            }
            vd['trials'].append(trial_out)
            if len(spot) < 3 and any(x is not None for x in trial_out['div_ref_per_step']):
                sv = next((x for x in trial_out['div_ref_per_step'] if x is not None), None)
                spot.append(f'{v}[t{t}] div_ref[0]={sv}')
        scene['variants'][v] = vd

    return scene, spot


def write_outputs(scene, out_dir, embed):
    os.makedirs(out_dir, exist_ok=True)
    scene_path = os.path.join(out_dir, 'scene.json')
    payload = json.dumps(scene, separators=(',', ':'))
    with open(scene_path, 'w') as f:
        f.write(payload)
    size_mb = len(payload) / 1e6
    viewer_path = None
    if embed:
        if not os.path.isfile(_TEMPLATE):
            print(f'[export] WARNING template not found at {_TEMPLATE}; wrote scene.json only.')
        else:
            with open(_TEMPLATE) as f:
                html = f.read()
            marker = '/*__SCENE_DATA__*/null'
            if marker not in html:
                print('[export] WARNING template missing /*__SCENE_DATA__*/null marker; scene.json only.')
            else:
                html = html.replace(marker, 'window.SCENE_DATA=' + payload + ';/*__END__*/')
                viewer_path = os.path.join(out_dir, f'viewer_{scene["scene"]}.html')
                with open(viewer_path, 'w') as f:
                    f.write(html)
    return scene_path, viewer_path, size_mb


def main():
    ap = argparse.ArgumentParser(description='Export an eval scene to scene.json + viewer HTML.')
    ap.add_argument('scene_dir')
    ap.add_argument('--env', choices=['avoiding', 'uav', 'unknown'], default='unknown')
    ap.add_argument('--trials', default='0')
    ap.add_argument('--plan-every', type=int, default=0)
    ap.add_argument('--variants', default=None)
    ap.add_argument('--reference', default='diffuser')
    ap.add_argument('--halfspace-variant', default=None)
    ap.add_argument('--exp', default='avoiding-d3il')
    ap.add_argument('--config', default=os.path.join(_REPO, 'config', 'projection_eval.yaml'))
    ap.add_argument('--out', default=None)
    ap.add_argument('--no-embed', action='store_true')
    args = ap.parse_args()

    scene_dir = os.path.abspath(args.scene_dir)
    scene, spot = build_scene(scene_dir, args)
    out_dir = args.out or os.path.join(scene_dir, '_traj_viz')
    scene_path, viewer_path, size_mb = write_outputs(scene, out_dir, not args.no_embed)

    # summary table
    print('\n[export] SUMMARY')
    print(f'  scene       : {scene["scene"]}  (env={scene["env"]})')
    nv = len(scene['variants'])
    print(f'  variants    : {nv} exported')
    for v, vd in scene['variants'].items():
        for tr in vd['trials']:
            print(f'    {v:22s} trial {tr["trial"]}: steps={tr["n_steps"]:4d} '
                  f'fans={len(tr["plans"]):4d} (every {scene["plan_every"]})')
            break
    print(f'  scene.json  : {size_mb:.2f} MB -> {scene_path}')
    if viewer_path:
        print(f'  viewer      : {viewer_path}')
        print('               (open directly in a browser — data is inlined)')
    if spot:
        print('  spot-checks : ' + ' | '.join(spot))


if __name__ == '__main__':
    main()
