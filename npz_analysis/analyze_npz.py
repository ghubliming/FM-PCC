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

MPC plan-fan analysis (`sampled_trajectories_all`, the candidate foresight trajectories), when present:
  • plan_*  columns — quality of the OPEN-LOOP plans (not the executed path): plan_path_len,
    plan_straightness, plan_roughness, plan_max_jerk, plan_max_step.
  • plan_max_abs — largest |coordinate| over all plan waypoints → a direct EXPLOSION detector
    (e.g. UAV altitude plans diving to −227 m show up as a huge plan_max_abs while the executed
    path looks tame).
  • plan_cand_spread — candidate diversity per snapshot (mean pairwise endpoint distance; NaN when
    batch=1, e.g. UAV).
  • plan_exec_div / plan_exec_div_best — how far the foresight plans are from what was actually
    executed (candidate-0 and best-candidate) → quantifies plan-vs-reality divergence (the runaway).
  • --replot-plans — per trial, overlay the candidate fan (blue) on the executed path (black).

Usage:
    python npz_analysis/analyze_npz.py <path-to-dir-or-file> [--out DIR] [--xy-cols 0 1]
        [--replot] [--replot-plans] [--dump-xy] [--no-recursive]
    # UAV: position x,y live in obs cols 3,4 (obs = [p_des(0:3) | p(3:6) | v(6:9)]):
    python npz_analysis/analyze_npz.py logs/UAV_FM/uav-pillars/plans --xy-cols 3 4 --replot-plans
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
# Additional headline keys added for dynamics checking
HEADLINE_KEYS += ['traj_dyn_gap_max', 'plan_dyn_gap_max']
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
                 mean_jerk=nan, max_jerk=nan, roughness=nan, max_abs=nan)
    try:
        a = np.asarray(traj, dtype=float)
    except Exception:
        return blank
    if a.ndim != 2 or a.shape[0] < 2:
        return dict(blank, points=int(a.shape[0]) if a.ndim >= 1 else 0)
    # explosion detector: largest |value| over ALL dims (not just xy) — catches a blow-up
    # on any axis (e.g. UAV altitude p_des_z → −227 while the xy plane looks tame).
    max_abs = float(np.nanmax(np.abs(a))) if a.size else nan
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
        max_abs=max_abs,                                                     # explosion (all dims)
    )

def analyze_dynamics_gap_traj(traj, acts, p_cols):
    """Check P[t+1] - P[t] = Act[t] on the executed trajectory."""
    try:
        a = np.asarray(traj, dtype=float)
        act_a = np.asarray(acts, dtype=float)
    except Exception:
        return float('nan')
    if a.ndim != 2 or a.shape[0] < 2 or act_a.ndim != 2:
        return float('nan')
        
    c = [x for x in p_cols if x < a.shape[1]] or list(range(min(2, a.shape[1])))
    # Compare up to the minimum length (act_a might be 1 step shorter in some schemas)
    n_steps = min(a.shape[0] - 1, act_a.shape[0])
    if n_steps <= 0:
        return float('nan')
        
    p_t = a[:n_steps, c]
    p_next = a[1:n_steps+1, c]
    act_t = act_a[:n_steps, :len(c)]
    
    gap = np.abs((p_next - p_t) - act_t)
    return float(np.nanmax(gap))



def mean_ignore_nan(vals):
    arr = np.array([v for v in vals if v == v], dtype=float)  # drop NaN
    return float(arr.mean()) if arr.size else float('nan')


# ── MPC plan-fan (candidate trajectory) analysis ─────────────────────────────
# `sampled_trajectories_all[trial]` = a sequence of plan SNAPSHOTS; each snapshot is
# the model's open-loop foresight as [batch, horizon, dim] (batch = # candidates).
# Schemas vary: avoiding snapshots every horizon//2 steps with batch>1; UAV snapshots
# every step with batch=1. We normalise both to a list of [batch, horizon, dim].
PLAN_METRIC_NAMES = ['plan_n_snap', 'plan_batch', 'plan_path_len', 'plan_straightness',
                     'plan_roughness', 'plan_max_jerk', 'plan_max_step', 'plan_max_abs',
                     'plan_cand_spread', 'plan_exec_div', 'plan_exec_div_best', 'plan_dyn_gap_max']


def _plan_snapshots(entry):
    """Normalise one trial's plan payload to a list of [batch, horizon, dim] float arrays."""
    snaps = []
    try:
        seq = list(entry)
    except TypeError:
        return snaps
    for s in seq:
        try:
            a = np.asarray(s, dtype=float)
        except Exception:
            continue
        if a.ndim == 2:        # [horizon, dim] → single candidate
            a = a[None, ...]
        if a.ndim == 3 and a.shape[1] >= 1:   # [batch, horizon, dim]
            snaps.append(a)
    return snaps


def _plan_exec_divergence(snaps, executed, cols):
    """How far each foresight plan is from what was actually executed (the runaway lens).

    Returns (mean over candidate-0, mean over best candidate). Aligns snapshot s to
    executed step round(s·(T-1)/(n_snap-1)); plan waypoint k → executed step start+k.
    """
    nan = float('nan')
    try:
        ex = np.asarray(executed, dtype=float)
    except Exception:
        return nan, nan
    if ex.ndim != 2 or ex.shape[0] < 1 or not snaps:
        return nan, nan
    T = ex.shape[0]
    nsnap = len(snaps)
    ecols = [c for c in cols if c < ex.shape[1]] or list(range(min(2, ex.shape[1])))
    exy = ex[:, ecols]
    first, best = [], []
    for si, s in enumerate(snaps):
        start = int(round(si * (T - 1) / max(nsnap - 1, 1)))
        pcols = [c for c in cols if c < s.shape[2]] or list(range(min(2, s.shape[2])))
        cand = []
        for b in range(s.shape[0]):
            pxy = s[b][:, pcols]
            idx = np.clip(np.arange(start, start + pxy.shape[0]), 0, T - 1)
            cand.append(float(np.nanmean(np.linalg.norm(pxy - exy[idx], axis=1))))
        if cand:
            first.append(cand[0])
            best.append(min(cand))
    return mean_ignore_nan(first), mean_ignore_nan(best)


def analyze_plans(plan_entry, cols, executed=None):
    """Quality + explosion + candidate-spread + executed-divergence for one trial's plan fan."""
    nan = float('nan')
    out = {k: nan for k in PLAN_METRIC_NAMES}
    out['plan_n_snap'], out['plan_batch'] = 0, 0
    snaps = _plan_snapshots(plan_entry)
    if not snaps:
        return out
    qual = {m: [] for m in ('path_len', 'straightness', 'roughness', 'max_jerk', 'max_step')}
    spreads, max_abs = [], 0.0
    batch = max(s.shape[0] for s in snaps)
    for s in snaps:                                   # s = [batch, horizon, dim]
        pcols = [c for c in cols if c < s.shape[2]] or list(range(min(2, s.shape[2])))
        ends = []
        for b in range(s.shape[0]):
            q = analyze_traj(s[b], cols)              # reuse executed-path quality metrics
            for m in qual:
                qual[m].append(q[m])
            if q['max_abs'] == q['max_abs']:          # all-dim explosion (not NaN)
                max_abs = max(max_abs, q['max_abs'])
            xy = s[b][:, pcols]
            if xy.size:
                ends.append(xy[-1])
        if len(ends) > 1:                             # candidate diversity (batch>1 only)
            e = np.asarray(ends)
            pair = [np.linalg.norm(e[a] - e[b]) for a in range(len(e)) for b in range(a + 1, len(e))]
            spreads.append(float(np.mean(pair)))
    out['plan_n_snap'] = len(snaps)
    out['plan_batch'] = int(batch)
    out['plan_path_len'] = mean_ignore_nan(qual['path_len'])
    out['plan_straightness'] = mean_ignore_nan(qual['straightness'])
    out['plan_roughness'] = mean_ignore_nan(qual['roughness'])
    out['plan_max_jerk'] = mean_ignore_nan(qual['max_jerk'])
    out['plan_max_step'] = mean_ignore_nan(qual['max_step'])
    out['plan_max_abs'] = max_abs                     # ← explosion detector (e.g. UAV p_des→ −227)
    out['plan_cand_spread'] = mean_ignore_nan(spreads) if spreads else nan
    if executed is not None:
        out['plan_exec_div'], out['plan_exec_div_best'] = _plan_exec_divergence(snaps, executed, cols)
    return out

def analyze_dynamics_gap_plan(plan_entry, p_cols, act_cols):
    """Check P[h+1] - P[h] = Act[h] inside the MPC candidate plan fan."""
    snaps = _plan_snapshots(plan_entry)
    if not snaps:
        return float('nan')

    max_gap_overall = 0.0
    valid = False
    for s in snaps:
        # s is [batch, horizon, dim]
        if s.shape[1] < 2:
            continue
        dim = s.shape[2]
        if any(c >= dim for c in p_cols) or any(c >= dim for c in act_cols):
            # columns out of range for this plan schema — skip rather than crash
            continue
        p = s[:, :, p_cols]
        act = s[:, :, act_cols]
        # P[h+1] - P[h] vs Act[h]
        p_t = p[:, :-1, :]
        p_next = p[:, 1:, :]
        act_t = act[:, :-1, :]
        gap = np.abs((p_next - p_t) - act_t)
        max_gap_overall = max(max_gap_overall, float(np.nanmax(gap)))
        valid = True
        
    return max_gap_overall if valid else float('nan')



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
                         'std_step', 'mean_jerk', 'max_jerk', 'roughness', 'max_abs', 'points']
    per_traj = []
    traj_dyn_gaps = []
    
    # Determine the columns based on the environment (avoiding vs uav)
    env = getattr(sys, '_npz_env', 'unknown')
    if env == 'uav':
        # obs_all layout: [p_des(0:3) | p(3:6) | v(6:9)] (9D, always stored in full).
        # Actual drone x,y are at cols 3,4 (p_des_x,p_des_y are 0,1 — the setpoint, not the path).
        obs_p_cols = [3, 4]
        # Plans store traj.observations (obs-only): [p_des(0:3) | p(3:6)] = 6 cols.
        # Actual x,y in plans also at cols 3,4.
        plan_p_cols = [3, 4]
        # Plans store NO action columns (same as avoiding: obs-only).
        # plan_act_cols=[0,1] would pick p_des_x/y — not Δp_des — giving garbage gap values.
        plan_act_cols = None
    elif env == 'avoiding':
        # obs_all layout: [x_des(0), y_des(1), x(2), y(3)]
        # traj_dyn_gap_max uses x_des cols [0,1] + act_all (Δx_des):
        #   gap = (x_des[t+1]-x_des[t]) - act[t] = 0 always (simulator is exact).
        #   Trivially 0 for ALL variants — not informative, but not wrong.
        obs_p_cols = [0, 1]
        # plan schema (sampled_trajectories_all, from eval_flow_matching_v3_imeanflow.py):
        #   stores samples.observations only → shape (B, H, 4) = [x_des, y_des, x, y].
        #   NO action columns are stored in the plan.  plan_act_cols = [0,1] would pick
        #   x_des (~-3.2) as the "action", giving gap = 0.02 - (-3.2) ≈ 3.22 — nonsense.
        #   plan_dyn_gap_max cannot be computed from this schema; mark it NaN.
        plan_p_cols = [2, 3]
        plan_act_cols = None   # None = actions not stored in plans → NaN for plan_dyn_gap_max
    else:
        # Fallback to the provided CLI cols.
        # plan_act_cols=None: without knowing the schema we cannot identify which columns
        # are actions vs observations. Guessing [0,1] would silently pick whatever is at
        # those columns (e.g. x_des ≈ ±3.2 in avoiding) and produce nonsense gap values.
        obs_p_cols = cols
        plan_p_cols = [c + 2 for c in cols]  # best guess: action_dim=2 prepended
        plan_act_cols = None

    if 'obs_all' in data.files:
        obs_all = data['obs_all']
        act_all = data['act_all'] if 'act_all' in data.files else None
        try:
            n = len(obs_all)
        except TypeError:
            n = 0
        for i in range(n):
            per_traj.append(analyze_traj(obs_all[i], obs_p_cols))
            if act_all is not None and i < len(act_all):
                traj_dyn_gaps.append(analyze_dynamics_gap_traj(obs_all[i], act_all[i], obs_p_cols))
        for name in traj_metric_names:
            file_row[f'traj_{name}__mean'] = mean_ignore_nan([t[name] for t in per_traj])
        if traj_dyn_gaps:
            file_row['traj_dyn_gap_max'] = float(np.nanmax(traj_dyn_gaps))
        file_row['n_traj'] = len(per_traj)

    # MPC plan-fan quality (sampled_trajectories_all) — the candidate foresight trajectories.
    # plan_max_abs / plan_exec_div expose a plan explosion the executed-path metrics miss.
    per_plan = []
    if 'sampled_trajectories_all' in data.files:
        plans_all = data['sampled_trajectories_all']
        try:
            npn = len(plans_all)
        except TypeError:
            npn = 0
        obs_for_div = data['obs_all'] if 'obs_all' in data.files else None
        for i in range(npn):
            ex_i = obs_for_div[i] if (obs_for_div is not None and i < len(obs_for_div)) else None
            plan_metrics = analyze_plans(plans_all[i], plan_p_cols, executed=ex_i)
            # plan_dyn_gap_max: requires action columns inside the plan tensor.
            # plan_act_cols=None means actions are not stored in this schema (e.g. avoiding
            # plans store obs-only); computing the gap would index the wrong columns and
            # give nonsense (e.g. ~3.2 from treating x_des as the action). Use NaN instead.
            if plan_act_cols is not None:
                plan_metrics['plan_dyn_gap_max'] = analyze_dynamics_gap_plan(
                    plans_all[i], plan_p_cols, plan_act_cols)
            else:
                plan_metrics['plan_dyn_gap_max'] = float('nan')
            per_plan.append(plan_metrics)
            
        for name in PLAN_METRIC_NAMES:
            file_row[f'{name}__mean'] = mean_ignore_nan([p[name] for p in per_plan])
        if per_plan:
            _gaps = [p['plan_dyn_gap_max'] for p in per_plan if not np.isnan(p['plan_dyn_gap_max'])]
            file_row['plan_dyn_gap_max'] = float(max(_gaps)) if _gaps else float('nan')
        file_row['n_plan'] = len(per_plan)

    # args metadata
    file_row.update({f'arg_{k}': v for k, v in load_args_meta(data).items()})

    # per-trial rows (join scalar metrics + traj + plan metrics by index)
    n_rows = max([file_row.get('n_trials', 0), len(per_traj), len(per_plan)])
    for i in range(n_rows):
        row = {'file': rel, 'variant': variant, 'trial': i}
        for k, m in metrics.items():
            if i < m.size:
                row[k] = m[i]
        if i < len(per_traj):
            for name in traj_metric_names:
                row[f'traj_{name}'] = per_traj[i][name]
        if i < len(per_plan):
            for name in PLAN_METRIC_NAMES:
                row[name] = per_plan[i][name]
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


def replot_with_plans(npz_path, out_dir, cols, rel, max_snaps=60, max_cand=4):
    """Per trial: executed path (black) + the MPC candidate plan fan (blue) + plan starts.

    This is the figure that makes a plan explosion visible — the foresight plans peel
    away from the executed path (e.g. UAV altitude plans diving to −227 m). Returns the
    list of saved PNGs (one per trial), or None if the npz lacks plan data."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None
    if 'sampled_trajectories_all' not in data.files or 'obs_all' not in data.files:
        data.close(); return None
    obs_all, plans_all = data['obs_all'], data['sampled_trajectories_all']
    try:
        n = min(len(obs_all), len(plans_all))
    except TypeError:
        data.close(); return None
    saved = []
    for i in range(n):
        try:
            ex = np.asarray(obs_all[i], dtype=float)
        except Exception:
            continue
        if ex.ndim != 2 or ex.shape[0] < 1:
            continue
        ecols = [c for c in cols if c < ex.shape[1]] or [0, min(1, ex.shape[1] - 1)]
        snaps = _plan_snapshots(plans_all[i])
        if not snaps:
            continue
        fig, ax = plt.subplots(figsize=(7, 7))
        stride = max(1, len(snaps) // max_snaps)
        for si in range(0, len(snaps), stride):
            s = snaps[si]
            pcols = [c for c in cols if c < s.shape[2]] or [0, min(1, s.shape[2] - 1)]
            for b in range(min(s.shape[0], max_cand)):
                pxy = s[b][:, pcols]
                ax.plot(pxy[:, 0], pxy[:, 1], 'b', lw=0.6, alpha=0.35)
                ax.plot(pxy[0, 0], pxy[0, 1], 'g.', ms=3)
        ax.plot(ex[:, ecols[0]], ex[:, ecols[1]], 'k', lw=1.6, label='executed')
        ax.plot(ex[0, ecols[0]], ex[0, ecols[1]], 'go', ms=6)
        ax.set_title(f'{rel} · trial {i}\nexecuted (black) + plan fan (blue) · cols (x={cols[0]}, y={cols[1]})')
        ax.set_xlabel(f'obs[{cols[0]}]'); ax.set_ylabel(f'obs[{cols[1]}]')
        ax.set_aspect('equal', 'datalim')
        name = rel.replace(os.sep, '__').replace('.npz', '') + f'_trial{i}_plans.png'
        out_png = os.path.join(out_dir, name)
        fig.savefig(out_png, dpi=120, bbox_inches='tight')
        plt.close(fig)
        saved.append(out_png)
    data.close()
    return saved


def dump_xy_rows(npz_path, cols, rel, variant, env='unknown'):
    """Raw per-step (x,y) of every executed trajectory AND predicted plan → list of row dicts."""
    rows = []
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception:
        return rows
        
    if env == 'uav':
        # obs_all: [p_des(0:3) | p(3:6) | v(6:9)] (9D). Actual drone x,y at cols 3,4.
        obs_p_cols = [3, 4]
        # Plans: obs-only [p_des(0:3) | p(3:6)] (6D). Actual x,y at cols 3,4.
        plan_p_cols = [3, 4]
        plan_act_cols = None  # UAV plans store obs-only; actions (Δp_des) not in plan tensor
    elif env == 'avoiding':
        obs_p_cols = [0, 1]
        plan_p_cols = [2, 3]
        plan_act_cols = None  # avoiding plans store obs-only; [0,1] would be x_des, not actions
    else:
        obs_p_cols = cols
        plan_p_cols = [c + 2 for c in cols]
        plan_act_cols = None  # unknown schema: don't guess action cols, avoid nonsense gaps

    # Dump executed path
    if 'obs_all' in data.files:
        obs = data['obs_all']
        acts = data['act_all'] if 'act_all' in data.files else None
        n = len(obs) if isinstance(obs, (list, tuple, np.ndarray)) else 0
        for i in range(n):
            try:
                a = np.asarray(obs[i], dtype=float)
                act_a = np.asarray(acts[i], dtype=float) if (acts is not None and i < len(acts)) else None
            except Exception:
                continue
            if a.ndim != 2 or a.shape[0] < 1:
                continue
            c = [x for x in obs_p_cols if x < a.shape[1]] or [0, min(1, a.shape[1] - 1)]
            for s in range(a.shape[0]):
                row = {'file': rel, 'variant': variant, 'source': 'executed', 'trial': i, 'step': s,
                       'x': float(a[s, c[0]]), 'y': float(a[s, c[1]])}
                if act_a is not None and s < act_a.shape[0]:
                    act_x = float(act_a[s, 0]) if act_a.shape[1] > 0 else float('nan')
                    act_y = float(act_a[s, 1]) if act_a.shape[1] > 1 else float('nan')
                    row['act_x'] = act_x
                    row['act_y'] = act_y
                    
                    # Gap calc for executed
                    if s + 1 < a.shape[0]:
                        x_next = float(a[s+1, c[0]])
                        y_next = float(a[s+1, c[1]])
                        row['gap_x'] = (x_next - row['x']) - act_x
                        row['gap_y'] = (y_next - row['y']) - act_y
                        status_x = 'Correct' if abs(row['gap_x']) < 1e-4 else f"Gap: {row['gap_x']:.5f}"
                        status_y = 'Correct' if abs(row['gap_y']) < 1e-4 else f"Gap: {row['gap_y']:.5f}"
                        row['status'] = 'Correct' if status_x == 'Correct' and status_y == 'Correct' else 'Gap'
                rows.append(row)

    # Dump predicted plans
    if 'sampled_trajectories_all' in data.files:
        plans = data['sampled_trajectories_all']
        n = len(plans) if isinstance(plans, (list, tuple, np.ndarray)) else 0
        for i in range(n):
            snaps = _plan_snapshots(plans[i])
            for snap_idx, snap in enumerate(snaps):
                for b in range(snap.shape[0]):
                    plan = snap[b]
                    for h in range(plan.shape[0]):
                        try:
                            x = float(plan[h, plan_p_cols[0]])
                            y = float(plan[h, plan_p_cols[1]])
                        except IndexError:
                            continue
                        # plan_act_cols=None: actions not stored in plans (e.g. avoiding obs-only)
                        if plan_act_cols is not None:
                            try:
                                act_x = float(plan[h, plan_act_cols[0]])
                                act_y = float(plan[h, plan_act_cols[1]])
                            except IndexError:
                                act_x = act_y = float('nan')
                        else:
                            act_x = act_y = float('nan')

                        row = {'file': rel, 'variant': variant, 'source': 'plan', 'trial': i,
                               'snapshot': snap_idx, 'candidate': b, 'step': h,
                               'x': x, 'y': y, 'act_x': act_x, 'act_y': act_y}
                               
                        if h + 1 < plan.shape[0]:
                            x_next = float(plan[h+1, plan_p_cols[0]])
                            y_next = float(plan[h+1, plan_p_cols[1]])
                            row['gap_x'] = (x_next - x) - act_x
                            row['gap_y'] = (y_next - y) - act_y
                            status_x = 'Correct' if abs(row['gap_x']) < 1e-4 else f"Gap: {row['gap_x']:.5f}"
                            status_y = 'Correct' if abs(row['gap_y']) < 1e-4 else f"Gap: {row['gap_y']:.5f}"
                            row['status'] = 'Correct' if status_x == 'Correct' and status_y == 'Correct' else 'Gap'
                        rows.append(row)
                        
    data.close()
    return rows


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
    """Compact stdout summary of headline numbers + trajectory + plan-fan quality."""
    cols = [('variant', 22), ('n_trials', 8), ('success_rate__mean', 12), ('n_steps__mean', 9),
            ('traj_straightness__mean', 11), ('traj_dyn_gap_max', 11)]
    hdr = {'success_rate__mean': 'succ_rate', 'n_steps__mean': 'steps',
           'traj_straightness__mean': 'straight', 'traj_dyn_gap_max': 'exec_dyngap'}
    # only show plan columns if at least one file has plan data
    if any('plan_max_abs__mean' in r for r in file_rows):
        cols += [('plan_max_abs__mean', 12), ('plan_dyn_gap_max', 12), ('plan_exec_div__mean', 11)]
        hdr.update({'plan_max_abs__mean': 'plan_maxabs', 'plan_dyn_gap_max': 'plan_dyngap', 'plan_exec_div__mean': 'plan_exdiv'})
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
    ap.add_argument('--replot-plans', action='store_true',
                    help='Per trial, overlay the MPC candidate plan fan (blue) on the executed path '
                         '(black) from sampled_trajectories_all — makes a plan explosion visible.')
    ap.add_argument('--dump-xy', action='store_true',
                    help='Write the RAW per-step (x,y) points of every trajectory to points_<ts>.csv '
                         '(columns: file, variant, trial, step, x, y).')
    ap.add_argument('--env', choices=['avoiding', 'uav', 'unknown'], default='unknown',
                    help='Environment layout to use for mapping action/position columns in plans.')
    ap.add_argument('--no-recursive', action='store_true', help='Do not recurse into subdirs.')
    args = ap.parse_args()
    
    # Store env on sys so it can be accessed in process_file without altering signature
    sys._npz_env = args.env

    root = os.path.abspath(args.path)
    files = find_npz(root, recursive=not args.no_recursive)
    if not files:
        print(f'[npz-analyze] no .npz found under {root}', file=sys.stderr)
        sys.exit(1)

    base = root if os.path.isdir(root) else os.path.dirname(root)
    out = args.out or os.path.join(base, '_npz_analysis')
    os.makedirs(out, exist_ok=True)

    print(f'[npz-analyze] scanning {len(files)} file(s) under {root}')
    file_rows, trial_rows, point_rows = [], [], []
    replots = []
    for p in files:
        fr, tr = process_file(p, base, args.xy_cols)
        file_rows.append(fr)
        trial_rows.extend(tr)
        rel = fr.get('file', os.path.basename(p))
        if args.replot:
            png = replot_trajectories(p, out, args.xy_cols, rel)
            if png:
                replots.append(png)
        if args.replot_plans:
            pngs = replot_with_plans(p, out, args.xy_cols, rel)
            if pngs:
                replots.extend(pngs)
        if args.dump_xy:
            point_rows.extend(dump_xy_rows(p, args.xy_cols, rel, fr.get('variant', ''), env=args.env))

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    fsum = os.path.join(out, f'files_summary_{stamp}.csv')
    ftri = os.path.join(out, f'per_trial_{stamp}.csv')
    n1 = write_csv(fsum, file_rows)
    n2 = write_csv(ftri, trial_rows)
    print_table(file_rows)
    print(f'[npz-analyze] files_summary: {n1} rows -> {fsum}')
    print(f'[npz-analyze] per_trial:     {n2} rows -> {ftri}')
    if args.replot or args.replot_plans:
        print(f'[npz-analyze] replot:        {len(replots)} png(s) -> {out}')
    if args.dump_xy:
        fpts = os.path.join(out, f'points_{stamp}.csv')
        n3 = write_csv(fpts, point_rows)
        print(f'[npz-analyze] points (xy):   {n3} rows -> {fpts}')
    errs = [r['file'] for r in file_rows if '_load_error' in r]
    if errs:
        print(f'[npz-analyze] WARNING: {len(errs)} file(s) failed to load (see _load_error col).')


if __name__ == '__main__':
    main()
