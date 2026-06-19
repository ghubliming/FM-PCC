#!/usr/bin/env python3
"""
npz analyzer — summarize eval `.npz` result files into human-readable / DA-ready CSVs.

Works on the eval outputs of BOTH schemas in this repo:
  • avoiding  (eval_flow_matching_v3_imeanflow.py): n_success, n_success_and_constraints,
    n_steps, n_violations, total_violations, avg_time, collision_free_completed, obs_all, act_all, args
  • visual-aligning / fm-visual (`va_*`): success_rate, entropy, mode_encoding, n_success, n_steps,
    avg_time, mean_distance, physical_tracking_errors, context_*, obs_all, act_all,
    sampled_trajectories_all, args
…and is **schema-generic**: any 1-D numeric array is treated as a per-trial metric and summarized,
so new/renamed keys are picked up automatically.

What it produces (in --out):
  • files_summary.csv  — ONE row per .npz: scalar/per-trial metric means + executed-trajectory
                          quality aggregates + key args (objective/backbone/NFE/ω/…).
  • per_trial.csv      — ONE row per (file, trial): per-trial metric + per-trajectory quality numbers.
  • a compact table printed to stdout.

Trajectory quality is computed on the EXECUTED closed-loop path (`obs_all`) — the same path the
metrics are built from. It exposes the "smooth vs exploded/chaotic" quality the success/violation
numbers are blind to (path length, straightness, step spikes, jerk). See
logs_in_develop/Gen3v4_imf/U6/DEBUG_DiT_Eval_Trajectory_Explosion.md for why this matters.

Usage:
    python npz_analysis/analyze_npz.py <path-to-dir-or-file> [--out DIR] [--xy-cols 0 1] [--no-recursive]
"""

import argparse
import csv
import os
import sys
from datetime import datetime
from glob import glob

import numpy as np

# Metric keys we know are "higher-level is better/worse"; printed first when present.
HEADLINE_KEYS = [
    'n_success', 'success_rate', 'n_success_and_constraints', 'collision_free_completed',
    'n_steps', 'n_violations', 'total_violations', 'avg_time', 'mean_distance', 'entropy',
]
# args fields worth surfacing as columns (only those present are emitted).
ARG_KEYS = [
    'dataset', 'horizon', 'diffusion', 'imf_objective', 'imf_backbone',
    'flow_steps_v3', 'ode_inference_steps_v3', 'time_beta_alpha_v3', 'time_beta_beta_v3',
    'meanflow_cfg_omega', 'meanflow_cfg_t_min', 'meanflow_cfg_t_max',
    'dual_head', 'interval_cfg', 'action_weight', 'seed', 'diffusion_loadpath',
]
# object-array keys that are trajectory payloads, not per-trial scalars.
TRAJ_KEYS = {'obs_all', 'act_all', 'sampled_trajectories_all', 'plans_all',
             'physical_tracking_errors', 'mode_encoding'}


def find_npz(path, recursive=True):
    if os.path.isfile(path) and path.endswith('.npz'):
        return [path]
    pat = '**/*.npz' if recursive else '*.npz'
    return sorted(glob(os.path.join(path, pat), recursive=recursive))


def load_args_meta(data):
    """Best-effort extract a flat dict of args metadata (Namespace or dict, maybe 0-d object array)."""
    if 'args' not in data.files:
        return {}
    try:
        a = data['args']
        if isinstance(a, np.ndarray) and a.dtype == object and a.ndim == 0:
            a = a.item()
        if isinstance(a, dict):
            meta = a
        else:
            meta = vars(a)  # argparse.Namespace
        return {k: meta[k] for k in ARG_KEYS if k in meta}
    except Exception as exc:  # custom unpicklable class, etc. — don't crash the run
        return {'_args_error': type(exc).__name__}


def per_trial_metrics(data):
    """Return {metric_name: 1-D float array} for every numeric per-trial array in the npz."""
    out = {}
    for k in data.files:
        if k in TRAJ_KEYS or k == 'args':
            continue
        v = data[k]
        if not isinstance(v, np.ndarray) or v.dtype == object:
            continue
        if not np.issubdtype(v.dtype, np.number):
            continue
        flat = np.atleast_1d(v).ravel()
        if flat.size == 0:
            continue
        out[k] = flat.astype(float)
    return out


def analyze_traj(traj, cols):
    """Quality metrics for one executed trajectory [T, D] on the given xy columns."""
    nan = float('nan')
    blank = dict(points=0, path_len=nan, net_disp=nan, straightness=nan,
                 mean_step=nan, max_step=nan, std_step=nan,
                 mean_jerk=nan, max_jerk=nan, roughness=nan)
    try:
        a = np.asarray(traj, dtype=float)
    except Exception:
        return blank
    if a.ndim != 2 or a.shape[0] < 2:
        return dict(blank, points=int(a.shape[0]) if a.ndim >= 1 else 0)
    cols = [c for c in cols if c < a.shape[1]] or list(range(min(2, a.shape[1])))
    xy = a[:, cols]
    diffs = np.diff(xy, axis=0)
    step = np.linalg.norm(diffs, axis=1)
    path_len = float(step.sum())
    net = float(np.linalg.norm(xy[-1] - xy[0]))
    med = float(np.median(step)) if step.size else nan
    # 2nd difference = acceleration/jerk proxy (curvature of the path)
    if len(diffs) >= 2:
        acc = np.diff(diffs, axis=0)
        jerk = np.linalg.norm(acc, axis=1)
        mean_jerk, max_jerk = float(jerk.mean()), float(jerk.max())
    else:
        mean_jerk = max_jerk = nan
    return dict(
        points=int(xy.shape[0]),
        path_len=path_len,
        net_disp=net,
        straightness=(net / path_len) if path_len > 1e-9 else nan,  # 1=straight, →0=chaotic
        mean_step=float(step.mean()), max_step=float(step.max()), std_step=float(step.std()),
        mean_jerk=mean_jerk, max_jerk=max_jerk,
        roughness=(float(step.max()) / med) if med and med > 1e-9 else nan,  # spike index
    )


def mean_ignore_nan(vals):
    arr = np.array([v for v in vals if v == v], dtype=float)  # drop NaN
    return float(arr.mean()) if arr.size else float('nan')


def process_file(npz_path, root, cols):
    rel = os.path.relpath(npz_path, root) if os.path.isdir(root) else os.path.basename(npz_path)
    variant = os.path.splitext(os.path.basename(npz_path))[0]
    parent = os.path.basename(os.path.dirname(npz_path))
    file_row = {'file': rel, 'variant': variant, 'parent_dir': parent}
    trial_rows = []
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:
        file_row['_load_error'] = f'{type(exc).__name__}: {exc}'
        return file_row, trial_rows

    metrics = per_trial_metrics(data)
    file_row['n_trials'] = int(max((m.size for m in metrics.values()), default=0))
    # scalar means for every per-trial metric (success-type means == rate)
    for k, m in metrics.items():
        file_row[f'{k}__mean'] = float(np.nanmean(m))
        if m.size > 1:
            file_row[f'{k}__std'] = float(np.nanstd(m))

    # executed-trajectory quality (obs_all)
    traj_metric_names = ['path_len', 'net_disp', 'straightness', 'mean_step', 'max_step',
                         'std_step', 'mean_jerk', 'max_jerk', 'roughness', 'points']
    per_traj = []
    if 'obs_all' in data.files:
        obs_all = data['obs_all']
        try:
            n = len(obs_all)
        except TypeError:
            n = 0
        for i in range(n):
            per_traj.append(analyze_traj(obs_all[i], cols))
        for name in traj_metric_names:
            file_row[f'traj_{name}__mean'] = mean_ignore_nan([t[name] for t in per_traj])
        file_row['n_traj'] = len(per_traj)

    # args metadata
    file_row.update({f'arg_{k}': v for k, v in load_args_meta(data).items()})

    # per-trial rows (join scalar metrics + traj metrics by index)
    n_rows = max([file_row.get('n_trials', 0), len(per_traj)])
    for i in range(n_rows):
        row = {'file': rel, 'variant': variant, 'trial': i}
        for k, m in metrics.items():
            if i < m.size:
                row[k] = m[i]
        if i < len(per_traj):
            for name in traj_metric_names:
                row[f'traj_{name}'] = per_traj[i][name]
        trial_rows.append(row)

    data.close()
    return file_row, trial_rows


def replot_trajectories(npz_path, out_dir, cols, rel):
    """Regenerate the executed (x,y) trajectory plot from obs_all — the SAME data drawn as the
    black path in the eval's <variant>.png. Returns the saved png path, or None."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None
    if 'obs_all' not in data.files:
        data.close(); return None
    obs_all = data['obs_all']
    try:
        n = len(obs_all)
    except TypeError:
        data.close(); return None
    fig, ax = plt.subplots(figsize=(7, 7))
    drawn = 0
    for i in range(n):
        try:
            a = np.asarray(obs_all[i], dtype=float)
        except Exception:
            continue
        if a.ndim != 2 or a.shape[0] < 1:
            continue
        c = [x for x in cols if x < a.shape[1]] or [0, min(1, a.shape[1] - 1)]
        ax.plot(a[:, c[0]], a[:, c[1]], lw=1.2, alpha=0.8, label=f'trial {i}')
        ax.plot(a[0, c[0]], a[0, c[1]], 'go', ms=5)  # start dot
        drawn += 1
    if drawn == 0:
        plt.close(fig); data.close(); return None
    ax.set_title(f'{rel}\nexecuted path · obs cols (x={cols[0]}, y={cols[1]})')
    ax.set_xlabel(f'obs[{cols[0]}]'); ax.set_ylabel(f'obs[{cols[1]}]')
    ax.set_aspect('equal', 'datalim')
    name = rel.replace(os.sep, '__').replace('.npz', '') + '_replot.png'
    out_png = os.path.join(out_dir, name)
    fig.savefig(out_png, dpi=120, bbox_inches='tight')
    plt.close(fig); data.close()
    return out_png


def write_csv(path, rows):
    if not rows:
        return 0
    # union of all keys, headline/known first for readability
    keys = []
    for k in (['file', 'variant', 'parent_dir', 'trial', 'n_trials', 'n_traj']
              + [f'{h}__mean' for h in HEADLINE_KEYS] + HEADLINE_KEYS):
        if any(k in r for r in rows) and k not in keys:
            keys.append(k)
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction='ignore')
        w.writeheader()
        for r in rows:
            w.writerow(r)
    return len(rows)


def print_table(file_rows):
    """Compact stdout summary of headline numbers + trajectory roughness."""
    cols = [('variant', 22), ('n_trials', 8), ('n_success__mean', 11), ('success_rate__mean', 12),
            ('collision_free_completed__mean', 9), ('n_steps__mean', 9),
            ('traj_straightness__mean', 11), ('traj_max_jerk__mean', 11), ('traj_roughness__mean', 10)]
    hdr = {'n_success__mean': 'success', 'success_rate__mean': 'succ_rate',
           'collision_free_completed__mean': 'collfree', 'n_steps__mean': 'steps',
           'traj_straightness__mean': 'straight', 'traj_max_jerk__mean': 'maxjerk',
           'traj_roughness__mean': 'rough'}
    line = '  '.join(f'{hdr.get(k, k):>{w}.{w}}' for k, w in cols)
    print('\n' + line)
    print('-' * len(line))
    for r in sorted(file_rows, key=lambda x: x.get('file', '')):
        cells = []
        for k, w in cols:
            v = r.get(k, '')
            if isinstance(v, float):
                v = f'{v:.3f}' if v == v else 'nan'
            cells.append(f'{str(v):>{w}.{w}}')
        print('  '.join(cells))
    print()


def main():
    ap = argparse.ArgumentParser(description='Summarize eval .npz files into CSVs.')
    ap.add_argument('path', help='Directory (scanned recursively) or a single .npz file.')
    ap.add_argument('--out', default=None, help='Output dir (default: <path>/_npz_analysis).')
    ap.add_argument('--xy-cols', type=int, nargs=2, default=[0, 1],
                    help='Observation columns treated as (x, y). AVOIDING executed path = "2 3" '
                         '(cols 0,1 are x_des,y_des); default 0 1.')
    ap.add_argument('--replot', action='store_true',
                    help='Regenerate the executed (x,y) trajectory plot from obs_all into a PNG per npz '
                         '(the same path drawn as the black line in the eval figure).')
    ap.add_argument('--no-recursive', action='store_true', help='Do not recurse into subdirs.')
    args = ap.parse_args()

    root = os.path.abspath(args.path)
    files = find_npz(root, recursive=not args.no_recursive)
    if not files:
        print(f'[npz-analyze] no .npz found under {root}', file=sys.stderr)
        sys.exit(1)

    base = root if os.path.isdir(root) else os.path.dirname(root)
    out = args.out or os.path.join(base, '_npz_analysis')
    os.makedirs(out, exist_ok=True)

    print(f'[npz-analyze] scanning {len(files)} file(s) under {root}')
    file_rows, trial_rows = [], []
    replots = []
    for p in files:
        fr, tr = process_file(p, base, args.xy_cols)
        file_rows.append(fr)
        trial_rows.extend(tr)
        if args.replot:
            png = replot_trajectories(p, out, args.xy_cols, fr.get('file', os.path.basename(p)))
            if png:
                replots.append(png)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fsum = os.path.join(out, f'files_summary_{stamp}.csv')
    ftri = os.path.join(out, f'per_trial_{stamp}.csv')
    n1 = write_csv(fsum, file_rows)
    n2 = write_csv(ftri, trial_rows)
    print_table(file_rows)
    print(f'[npz-analyze] files_summary: {n1} rows -> {fsum}')
    print(f'[npz-analyze] per_trial:     {n2} rows -> {ftri}')
    if args.replot:
        print(f'[npz-analyze] replot:        {len(replots)} png(s) -> {out}')
    errs = [r['file'] for r in file_rows if '_load_error' in r]
    if errs:
        print(f'[npz-analyze] WARNING: {len(errs)} file(s) failed to load (see _load_error col).')


if __name__ == '__main__':
    main()
