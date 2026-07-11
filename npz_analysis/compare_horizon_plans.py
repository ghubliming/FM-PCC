#!/usr/bin/env python3
"""
compare_horizon_plans.py — HORIZONTAL (across-method) comparison of the H-step MPC plan.

Sibling/standalone to analyze_npz.py (which does per-file, vertical aggregates). This script
answers a different question:

    "For the SAME trial/seed, at the SAME MPC decision (snapshot), how does each projection
     method bend the H-step foresight plan?"

Each projection variant of one eval batch writes its own <variant>.npz into a shared results
folder (e.g. halfspace_both-hard/{diffuser,dpcc-c,dpcc-r,dpcc-t,gradient,model_free,
post_processing}.npz). Every variant is driven from the SAME seed, so:

  • snapshot 0 starts from an IDENTICAL state across all variants → a clean apples-to-apples
    comparison of what each projection does to the same nominal plan.
  • snapshot k>0 does NOT share state: the executed paths diverge (episodes even differ in
    length → differing snapshot counts), so the same index is a DIFFERENT physical decision
    point per method. The script prints this warning and each variant's snapshot count.

Schema (avoiding / D3IL, produced by eval_flow_matching_v3_imeanflow.py):
  sampled_trajectories_all[trial] = list/array of snapshots, each [batch, H, obs_dim];
  obs_dim=4 = [x_des, y_des, x, y] → actual position at cols 2,3 (obs_indices x=2,y=3).
  Snapshots are saved every horizon//2 steps, so snapshot k ≈ executed step k*(H//2).
  The npz stores ALL batch candidates in original order; the executed/selected candidate
  index is NOT recorded (trajectory_selection = random | temporal_consistency |
  minimum_projection_cost), so this tool treats the whole candidate fan, not "candidate 0".

Outputs (into --out, default <path>/_horizon_compare):
  • horizon_compare_t<trial>_s<snap>.png — one overlay: each variant a colour, its candidate
    fan thin + candidate mean bold, start dot + horizon-end marker.
  • horizon_compare_t<trial>_s<snap>.csv — long/tidy: variant,trial,snapshot,candidate,step,x,y.
  • a stdout table: per variant the plan endpoint, path length, max|coord| (explosion),
    and divergence-from-reference (mean per-waypoint distance of the candidate-mean plan to the
    reference variant's, default `diffuser`) → how far each projection pulls the plan.

Usage:
  python npz_analysis/compare_horizon_plans.py <results-dir> [--trial 0] [--snapshot 0]
      [--env avoiding|uav|unknown] [--xy-cols 2 3] [--variants diffuser,dpcc-c,...]
      [--reference diffuser] [--align index|step] [--out DIR]
"""

import argparse
import csv
import os
import sys
from glob import glob

import numpy as np

# distinct colours for up to ~10 variants (theme-neutral, colour-blind-friendly-ish)
_PALETTE = ['#000000', '#e6194B', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
            '#42d4f4', '#f032e6', '#9A6324', '#808000']


def find_variant_npz(path):
    """Return {variant_name: npz_path}, variant = npz basename.

    Supports BOTH result layouts in this repo:
      • flat (avoiding): <dir>/<variant>.npz
      • nested (UAV):     <dir>/<variant>/<variant>.npz
    Flat wins on a name clash. Variant name = npz basename (== subdir name in the nested layout).
    """
    if os.path.isfile(path) and path.endswith('.npz'):
        return {os.path.splitext(os.path.basename(path))[0]: path}
    files = sorted(glob(os.path.join(path, '*.npz')))          # flat
    files += sorted(glob(os.path.join(path, '*', '*.npz')))    # nested <variant>/<variant>.npz
    out = {}
    for p in files:
        out.setdefault(os.path.splitext(os.path.basename(p))[0], p)
    return out


def plan_snapshots(entry):
    """Normalise one trial's plan payload to a list of [batch, H, dim] float arrays.

    Handles BOTH storage shapes seen in this repo:
      • ragged  (dpcc-*): object array (n_trials,), entry = list of [batch,H,dim]
      • stacked (imf/diffuser when every trial has equal snapshot count): homogeneous
        (n_trials, n_snap, batch, H, dim) array, entry = (n_snap,batch,H,dim).
    """
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
        if a.ndim == 2:                       # [H, dim] → single candidate
            a = a[None, ...]
        if a.ndim == 3 and a.shape[1] >= 1:   # [batch, H, dim]
            snaps.append(a)
    return snaps


def load_variant(npz_path, trial):
    """Return (snaps, executed_obs) for one variant at `trial`, or (None, None)."""
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as exc:
        print(f'  ! {os.path.basename(npz_path)}: load failed ({type(exc).__name__})', file=sys.stderr)
        return None, None
    snaps, executed = None, None
    if 'sampled_trajectories_all' in data.files:
        st = data['sampled_trajectories_all']
        if trial < len(st):
            snaps = plan_snapshots(st[trial])
    if 'obs_all' in data.files:
        obs = data['obs_all']
        if trial < len(obs):
            try:
                executed = np.asarray(obs[trial], dtype=float)
            except Exception:
                executed = None
    data.close()
    return snaps, executed


def pick_snapshot(snaps, requested, align, save_every):
    """Resolve the snapshot index for this variant.

    align='index' → use `requested` directly (same index across methods; only snapshot 0
                    shares state — see module docstring).
    align='step'  → interpret `requested` as a target EXECUTED step and pick the snapshot whose
                    start step (k*save_every) is nearest to it, per variant (compensates for the
                    differing snapshot counts, though states still diverge for step>0).
    Returns (snap_idx, approx_exec_step) or (None, None) if unavailable.
    """
    if not snaps:
        return None, None
    n = len(snaps)
    if align == 'step':
        target = requested
        idx = int(round(target / max(save_every, 1)))
        idx = max(0, min(idx, n - 1))
    else:
        idx = requested
        if idx < 0:
            idx += n
        if idx < 0 or idx >= n:
            return None, None
    return idx, idx * save_every


def candidate_mean(snap, cols):
    """Mean plan over candidates → [H, 2] on the xy cols (for a clean bold overlay + divergence)."""
    c = [x for x in cols if x < snap.shape[2]] or list(range(min(2, snap.shape[2])))
    return snap[:, :, c].mean(axis=0)


# ── task constraints (avoiding halfspace tasks) ──────────────────────────────
# Reproduces eval_flow_matching_v3_imeanflow.py's variant→constraint selection (lines ~59-67)
# and utils.constraints_helpers' draw/inequality maths, so the overlay + violation flag match
# what the projection actually enforced. Kept self-contained (no import of the eval package).
_HALFSPACE_VARIANTS = ['both-hard', 'top-left-hard', 'top-right-hard']


def infer_halfspace_variant(path):
    base = os.path.basename(os.path.normpath(path if os.path.isdir(path) else os.path.dirname(path)))
    for v in _HALFSPACE_VARIANTS:
        if v in base:
            return v
    return None


def load_task_constraints(config_path, exp, halfspace_variant):
    """Return (polytopic_list, obstacle_list, ax_limits) for one avoiding halfspace variant."""
    import yaml  # lazy: only needed under --draw-constraints
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    hs = cfg['halfspace_constraints'][exp]
    ob = cfg['obstacle_constraints'][exp]
    sel = {'top-left-hard': ([hs[0]], [ob[3]]),
           'top-right-hard': ([hs[1]], [ob[4]]),
           'both-hard': ([hs[2], hs[3]], [ob[5]])}
    poly, obst = sel.get(halfspace_variant, ([], []))
    return poly, obst, cfg['ax_limits'][exp]


def _halfspace_third_vertex(constraint, ax_limits):
    """The triangle's 3rd vertex, matching utils.plot_halfspace_constraints' avoiding branch."""
    p0, p1, side = constraint[0], constraint[1], constraint[2]
    slope = (p1[1] - p0[1]) / (p1[0] - p0[0])
    if slope > 0:
        return [ax_limits[0][1], ax_limits[1][0]] if side == 'above' else [ax_limits[0][0], ax_limits[1][1]]
    return [ax_limits[0][0], ax_limits[1][0]] if side == 'above' else [ax_limits[0][1], ax_limits[1][1]]


def halfspace_violates(x, y, constraint):
    """True where (x,y) is on the FORBIDDEN side, per formulate_halfspace_constraints' inequality.

    'below' feasible ⇔ y < m*x + d (d = y0 - m*x0) ⇒ violate when (-m*x + y) >= d. 'above' mirrored.
    Vectorised over numpy arrays.
    """
    p0, p1, side = constraint[0], constraint[1], constraint[2]
    m = (p1[1] - p0[1]) / (p1[0] - p0[0])
    d = p0[1] - m * p0[0]
    val = -m * np.asarray(x) + np.asarray(y)
    return val >= d if side == 'below' else val <= d


def obstacle_violates(x, y, obst):
    """True where (x,y) is INSIDE a sphere_outside obstacle (distance < radius)."""
    c = obst['center']
    return np.hypot(np.asarray(x) - c[0], np.asarray(y) - c[1]) < obst['radius']


def plan_violation_mask(snap, cols, polys, obsts):
    """Boolean [batch, H]: waypoint violates ANY halfspace or obstacle constraint."""
    c = [x for x in cols if x < snap.shape[2]] or list(range(min(2, snap.shape[2])))
    xs, ys = snap[:, :, c[0]], snap[:, :, c[1]]
    mask = np.zeros(xs.shape, dtype=bool)
    for con in polys:
        mask |= halfspace_violates(xs, ys, con)
    for ob in obsts:
        mask |= obstacle_violates(xs, ys, ob)
    return mask


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('path', help='Results dir holding <variant>.npz files (or a single .npz).')
    ap.add_argument('--trial', type=int, default=0, help='Trial/seed index (default 0).')
    ap.add_argument('--snapshot', type=int, default=0,
                    help='Snapshot index (align=index) or target executed step (align=step). '
                         'Default 0 = the only truly state-aligned decision across methods.')
    ap.add_argument('--align', choices=['index', 'step'], default='index',
                    help='index: same snapshot index per method; step: nearest snapshot to a '
                         'target executed step (compensates for differing snapshot counts).')
    ap.add_argument('--env', choices=['avoiding', 'uav', 'unknown'], default='avoiding',
                    help='Column preset for xy position (default avoiding: cols 2,3).')
    ap.add_argument('--xy-cols', type=int, nargs=2, default=None,
                    help='Override the (x,y) plan columns. Default by --env.')
    ap.add_argument('--variants', default=None,
                    help='Comma list to restrict/order variants (default: all npz in dir).')
    ap.add_argument('--reference', default='diffuser',
                    help='Variant used as the unprojected reference for divergence (default diffuser).')
    ap.add_argument('--candidate', default='mean',
                    help='Which plan to draw/compare per variant: "mean" (candidate-mean, default) or '
                         'an int candidate index k. With a shared RNG seed the SAME candidate k is the '
                         'same pre-projection sample across variants AT SNAPSHOT 0 → picking one k gives '
                         'a clean per-candidate projection comparison (thin fan hidden). Only valid at '
                         'snapshot 0; later snapshots have diverged/desynced candidates.')
    ap.add_argument('--horizon-div', type=int, default=None,
                    help='Save-every override (H//2). Default inferred as 4 (avoiding H=8).')
    ap.add_argument('--out', default=None, help='Output dir (default <path>/_horizon_compare).')
    ap.add_argument('--no-plot', action='store_true', help='Skip the PNG (CSV + table only).')
    ap.add_argument('--draw-constraints', action='store_true',
                    help='Overlay the avoiding halfspace triangles + obstacle circle the projection '
                         'enforced, flag plan waypoints that VIOLATE them (plan_viol column + red x '
                         'on the plot), and lock the frame to the config ax_limits.')
    ap.add_argument('--config', default='config/projection_eval.yaml',
                    help='Config with halfspace/obstacle/ax_limits (default config/projection_eval.yaml).')
    ap.add_argument('--halfspace-variant', default=None,
                    help='both-hard | top-left-hard | top-right-hard (default: inferred from folder name).')
    ap.add_argument('--exp', default='avoiding-d3il', help='Task key in the config (default avoiding-d3il).')
    ap.add_argument('--show-executed', action='store_true',
                    help='Also draw each variant\'s executed closed-loop path (obs_all) faintly, for '
                         'plan-vs-reality context.')
    ap.add_argument('--full-frame', action='store_true',
                    help='Lock the view to the full environment (config ax_limits). Off by default: '
                         'the view auto-zooms to the plans (the env frame makes the small H-step plans '
                         'too tiny to see).')
    ap.add_argument('--show-env', action='store_true',
                    help='Also draw the avoiding env itself (blue filled halfspace funnel + obstacle '
                         'circle). OFF by default — --draw-constraints still flags violations (red x + '
                         'plan_viol) without cluttering the plot with the env background.')
    args = ap.parse_args()

    # xy columns: avoiding actual position = cols 2,3; uav = 3,4; else 0,1 unless overridden.
    if args.xy_cols is not None:
        cols = list(args.xy_cols)
    elif args.env == 'avoiding':
        cols = [2, 3]
    elif args.env == 'uav':
        cols = [3, 4]
    else:
        cols = [0, 1]
    # last-resort fallback only; per-variant save_every is inferred from data below (env-agnostic).
    save_every = args.horizon_div if args.horizon_div else 4

    # --candidate: 'mean' → candidate-mean (cand_idx=None); else a specific candidate index.
    cand_idx = None
    if str(args.candidate).lower() != 'mean':
        try:
            cand_idx = int(args.candidate)
        except ValueError:
            print(f'[compare] bad --candidate {args.candidate!r} (want "mean" or an int); using mean.')
    if cand_idx is not None and args.snapshot != 0:
        print('[compare] WARNING: --candidate k is only apples-to-apples at snapshot 0 (shared RNG '
              'seed). At snapshot>0 candidate k has diverged/desynced across variants.')

    variant_map = find_variant_npz(args.path)
    if not variant_map:
        print(f'[compare] no .npz under {args.path}', file=sys.stderr)
        sys.exit(1)
    if args.variants:
        want = [v.strip() for v in args.variants.split(',') if v.strip()]
        variant_map = {v: variant_map[v] for v in want if v in variant_map} or variant_map

    base = args.path if os.path.isdir(args.path) else os.path.dirname(args.path)
    out = args.out or os.path.join(base, '_horizon_compare')
    os.makedirs(out, exist_ok=True)

    # optional: load the enforced constraints (for drawing + violation flagging)
    polys, obsts, cfg_ax = [], [], None
    if args.draw_constraints:
        hv = args.halfspace_variant or infer_halfspace_variant(args.path)
        if hv is None:
            print('[compare] WARNING: could not infer --halfspace-variant from folder name; pass it '
                  'explicitly (both-hard | top-left-hard | top-right-hard). Skipping constraints.')
        else:
            try:
                polys, obsts, cfg_ax = load_task_constraints(args.config, args.exp, hv)
                print(f'[compare] constraints: variant={hv}  halfspaces={len(polys)}  obstacles={len(obsts)}  '
                      f'(from {args.config})')
            except Exception as exc:
                print(f'[compare] WARNING: constraint load failed ({type(exc).__name__}: {exc}); '
                      'continuing without them.')

    print(f'[compare] trial={args.trial}  snapshot={args.snapshot} (align={args.align})  '
          f'xy-cols={cols}  variants={list(variant_map)}')
    if args.snapshot != 0 and args.align == 'index':
        print('[compare] WARNING: snapshot>0 with align=index — executed paths have diverged, so '
              'this index is a DIFFERENT physical state per method (not apples-to-apples). '
              'Use --align step, or --snapshot 0, for a state-aligned comparison.')

    # ── gather each variant's chosen snapshot ────────────────────────────────
    rows, per_variant, ref_mean = [], {}, None
    have_con = bool(polys or obsts)
    for v, p in variant_map.items():
        snaps, executed = load_variant(p, args.trial)
        if not snaps:
            print(f'  {v:16s}: no plan snapshots for trial {args.trial} — skipped')
            continue
        # save_every (snapshot k → executed step k*save_every) is ENV-SPECIFIC: avoiding saves
        # every H//2 steps, UAV saves every step. Rather than hardcode a constant (footgun across
        # envs), infer it per-variant from the data: executed_len / n_snapshots. Falls back to the
        # CLI --horizon-div, then to `save_every`, only when executed length is unavailable.
        if args.horizon_div:
            save_every_v = args.horizon_div
        elif executed is not None and getattr(executed, 'ndim', 0) == 2 and len(snaps) > 1:
            save_every_v = max(1, int(round((executed.shape[0] - 1) / (len(snaps) - 1))))
        else:
            save_every_v = save_every
        idx, approx_step = pick_snapshot(snaps, args.snapshot, args.align, save_every_v)
        if idx is None:
            print(f'  {v:16s}: snapshot {args.snapshot} out of range (has {len(snaps)}) — skipped')
            continue
        snap = snaps[idx]                                   # [batch, H, dim]
        c = [x for x in cols if x < snap.shape[2]] or list(range(min(2, snap.shape[2])))
        if any(x >= snap.shape[2] for x in cols):           # requested xy cols exceed plan dim → warn
            print(f'  {v:16s}: WARNING xy-cols {cols} exceed plan dim {snap.shape[2]}; '
                  f'falling back to {c}. Pass --xy-cols for this env/schema.')
        if cand_idx is None:                                # representative plan: mean or candidate k
            mean_xy = candidate_mean(snap, cols)            # [H,2]
        else:
            k = min(cand_idx, snap.shape[0] - 1)
            if cand_idx >= snap.shape[0]:
                print(f'  {v:16s}: candidate {cand_idx} >= batch {snap.shape[0]}; using {k}.')
            mean_xy = snap[k][:, c]
        viol_mask = plan_violation_mask(snap, cols, polys, obsts) if have_con else None
        viol_frac = float(viol_mask.mean()) if viol_mask is not None else float('nan')
        per_variant[v] = dict(snap=snap, cols=c, idx=idx, approx_step=approx_step,
                              n_snap=len(snaps), mean_xy=mean_xy, executed=executed,
                              viol_frac=viol_frac)
        if v == args.reference:
            ref_mean = mean_xy
        # tidy long rows (all candidates, all steps)
        for b in range(snap.shape[0]):
            for h in range(snap.shape[1]):
                rows.append({'variant': v, 'trial': args.trial, 'snapshot': idx,
                             'approx_exec_step': approx_step, 'candidate': b, 'step': h,
                             'x': float(snap[b, h, c[0]]), 'y': float(snap[b, h, c[1]])})

    if not per_variant:
        print('[compare] nothing to compare (no variant had the requested snapshot).')
        sys.exit(1)

    # ── stdout table: geometry + divergence from reference (+ violation if drawn) ──
    vcol = f' {"plan_viol":>9}' if have_con else ''
    print(f'\n{"variant":16s} {"snap":>4} {"n_snap":>6} {"~step":>6} {"batch":>5} '
          f'{"path_len":>8} {"max|c|":>7} {"end_xy":>16} {"div_ref":>8}{vcol}')
    print('-' * (90 + (10 if have_con else 0)))
    for v, d in per_variant.items():
        m = d['mean_xy']
        seg = np.diff(m, axis=0)
        path_len = float(np.linalg.norm(seg, axis=1).sum())
        max_abs = float(np.nanmax(np.abs(d['snap'])))
        end = m[-1]
        if ref_mean is not None and m.shape == ref_mean.shape:
            div = float(np.linalg.norm(m - ref_mean, axis=1).mean())
            div_s = f'{div:8.4f}' if v != args.reference else '     ref'
        else:
            div_s = '     n/a'
        vstr = f' {d["viol_frac"]*100:8.1f}%' if have_con else ''
        print(f'{v:16s} {d["idx"]:>4} {d["n_snap"]:>6} {d["approx_step"]:>6} '
              f'{d["snap"].shape[0]:>5} {path_len:8.4f} {max_abs:7.3f} '
              f'({end[0]:6.3f},{end[1]:6.3f}) {div_s}{vstr}')
    if have_con:
        print('  plan_viol = % of plan waypoints (all candidates) inside a forbidden region.')
    print()

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = os.path.join(out, f'horizon_compare_t{args.trial}_s{args.snapshot}.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['variant', 'trial', 'snapshot', 'approx_exec_step',
                                          'candidate', 'step', 'x', 'y'])
        w.writeheader()
        w.writerows(rows)
    print(f'[compare] csv: {len(rows)} rows -> {csv_path}')

    # ── overlay plot ──────────────────────────────────────────────────────────
    if not args.no_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon, Circle
        fig, ax = plt.subplots(figsize=(8, 8))
        # constraint overlay first (behind the plans). Frame: auto-zoom to the plans by default so
        # they stay visible; only lock to the full-env ax_limits under --full-frame.
        frame = cfg_ax if (args.full_frame and cfg_ax is not None) else None
        if have_con and cfg_ax is not None and args.show_env:   # env background is opt-in
            for con in polys:
                tri = np.array([con[0], con[1], _halfspace_third_vertex(con, cfg_ax)])
                ax.add_patch(Polygon(tri, color='b', alpha=0.15, zorder=0))
            for ob in obsts:
                ax.add_patch(Circle(ob['center'], ob['radius'], color='b', alpha=0.20, zorder=0))
        for i, (v, d) in enumerate(per_variant.items()):
            col = _PALETTE[i % len(_PALETTE)]
            snap, c = d['snap'], d['cols']
            if args.show_executed and d.get('executed') is not None:      # plan-vs-reality context
                ex = np.asarray(d['executed'], dtype=float)
                ec = [x for x in cols if x < ex.shape[1]] or c
                ax.plot(ex[:, ec[0]], ex[:, ec[1]], color=col, lw=0.8, alpha=0.25, ls='--')
            if cand_idx is None:                            # candidate fan (thin) — only in mean mode
                for b in range(snap.shape[0]):
                    xy = snap[b][:, c]
                    ax.plot(xy[:, 0], xy[:, 1], color=col, lw=0.7, alpha=0.30)
            m = d['mean_xy']                                # representative plan (bold): mean or cand k
            ax.plot(m[:, 0], m[:, 1], color=col, lw=2.2, alpha=0.95, label=v)
            ax.plot(m[0, 0], m[0, 1], 'o', color=col, ms=7, mec='k', mew=0.6)   # start
            ax.plot(m[-1, 0], m[-1, 1], 's', color=col, ms=7, mec='k', mew=0.6)  # H-end
            if have_con:                                     # red x on mean-plan waypoints that violate
                vm = np.zeros(m.shape[0], dtype=bool)
                for con in polys:
                    vm |= halfspace_violates(m[:, 0], m[:, 1], con)
                for ob in obsts:
                    vm |= obstacle_violates(m[:, 0], m[:, 1], ob)
                if vm.any():
                    ax.plot(m[vm, 0], m[vm, 1], 'x', color='red', ms=9, mew=2, zorder=5)
        rep = 'candidate mean' if cand_idx is None else f'candidate {cand_idx}'
        fantxt = 'candidate fan (thin) + candidate mean (bold)' if cand_idx is None else f'{rep} (bold)'
        title = (f'H-step plan · trial {args.trial} · snapshot {args.snapshot} (align={args.align}) · {rep}\n'
                 f'{fantxt}; ●=start ■=horizon-end · xy cols {cols}')
        if have_con:
            title += ('\nblue = forbidden region · red × = violating plan waypoint'
                      if args.show_env else '\nred × = violating plan waypoint')
        ax.set_title(title)
        ax.set_xlabel(f'x (obs col {cols[0]})'); ax.set_ylabel(f'y (obs col {cols[1]})')
        if frame is not None:                               # --full-frame: whole environment
            ax.set_xlim(frame[0]); ax.set_ylim(frame[1])
        else:
            # auto-zoom to the PLANS only (drawn constraint patches would otherwise blow the view
            # out to full-env size, and the executed paths can span the whole map — both excluded).
            pts = np.vstack([d['snap'][:, :, d['cols']].reshape(-1, 2) for d in per_variant.values()])
            lo, hi = pts.min(0), pts.max(0)
            pad = 0.08 * np.maximum(hi - lo, 1e-3)
            ax.set_xlim(lo[0] - pad[0], hi[0] + pad[0])
            ax.set_ylim(lo[1] - pad[1], hi[1] + pad[1])
        ax.set_aspect('equal')
        ax.legend(fontsize=8, loc='best')
        png = os.path.join(out, f'horizon_compare_t{args.trial}_s{args.snapshot}.png')
        fig.savefig(png, dpi=130, bbox_inches='tight')
        plt.close(fig)
        print(f'[compare] png: {png}')


if __name__ == '__main__':
    main()
