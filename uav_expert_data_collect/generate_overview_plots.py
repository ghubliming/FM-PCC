"""E5 U5 — 2-D overhead trajectory overview plots.

Generates top-down (XY-plane) plots of UAV trajectories with obstacle geometry
overlay.  No MuJoCo or GPU required — pure matplotlib / numpy.

Two modes
---------
summary      One PNG per scene: all episodes overlaid, coloured by homotopy.
             Good for dataset coverage and balance inspection.
per-episode  One PNG per episode: commanded p_des (dashed blue) vs actual p
             (solid red), start/end markers, metadata in title.
both         Both modes.

Output layout
-------------
<out_dir>/<scene>_summary.png                           (summary)
<out_dir>/<scene>/<homotopy_safe>/<ep_id>_plot.png      (per-episode)

Usage
-----
# Scene summaries — all episodes in production data
python uav_expert_data_collect/generate_overview_plots.py --mode summary

# Selective: 1 per-episode plot per homotopy bucket (9 plots total)
python uav_expert_data_collect/generate_overview_plots.py --mode per-episode --per-homotopy 1

# Stress root — see what stress trajectories look like
python uav_expert_data_collect/generate_overview_plots.py \\
    --data-dir logs/uav_expert_data_stress --mode both --per-homotopy 1

# Single scene summary
python uav_expert_data_collect/generate_overview_plots.py --scene pillars
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')   # headless — no display needed
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DEFAULT_DATA_DIR = os.path.join(_REPO, 'logs', 'uav_expert_data')

# Consistent homotopy colours across scenes (extras fall through to tab10 cycle).
_FIXED_COLORS = {
    'L':       '#e41a1c',
    'C':       '#377eb8',
    'R':       '#4daf4a',
    '(L,L,L)': '#e41a1c',
    '(L,R,L)': '#ff7f00',
    '(R,L,R)': '#984ea3',
    '(R,R,R)': '#377eb8',
    'default': '#377eb8',
    'N/A':     '#888888',
    # stress cases
    'extreme_speed':    '#e41a1c',
    'tight_fillet':     '#ff7f00',
    'discontinuous':    '#984ea3',
    'wall_crossing':    '#a65628',
    'floor_dive':       '#4daf4a',
    'ceiling_climb':    '#f781bf',
    'gain_extreme':     '#999999',
    'degenerate_hover': '#377eb8',
}
_TAB10 = plt.get_cmap('tab10')


def _homotopy_color(homotopy, dynamic_palette):
    if homotopy in _FIXED_COLORS:
        return _FIXED_COLORS[homotopy]
    if homotopy not in dynamic_palette:
        dynamic_palette[homotopy] = _TAB10(len(dynamic_palette) % 10)
    return dynamic_palette[homotopy]


def _safe_dir(homotopy):
    return (homotopy.replace('(', '').replace(')', '')
                    .replace(',', '_').replace('/', '_').replace(' ', ''))


# ── Obstacle drawing ──────────────────────────────────────────────────────────

def _draw_obstacles(ax, obstacles, alpha=0.65, zorder=3):
    """Draw XY projections of all obstacle geometry onto ax."""
    for obs in obstacles:
        cx = float(obs['center'][0])
        cy = float(obs['center'][1])
        otype = obs.get('type', '')

        if otype == 'cylinder':
            r = float(obs['radius'])
            # Body
            patch = mpatches.Circle(
                (cx, cy), r,
                facecolor='#555555', edgecolor='#222222',
                linewidth=1.2, alpha=alpha, zorder=zorder)
            ax.add_patch(patch)
            # Safety margin ring (rotor reach 0.31 m + pillar radius)
            safety_r = r + 0.31
            ring = mpatches.Circle(
                (cx, cy), safety_r,
                facecolor='none', edgecolor='#cc4400',
                linewidth=0.6, linestyle='--', alpha=0.5, zorder=zorder)
            ax.add_patch(ring)

        elif otype == 'box':
            hx = float(obs['half_extents'][0])
            hy = float(obs['half_extents'][1])
            patch = mpatches.Rectangle(
                (cx - hx, cy - hy), 2 * hx, 2 * hy,
                facecolor='#555555', edgecolor='#222222',
                linewidth=1.2, alpha=alpha, zorder=zorder)
            ax.add_patch(patch)


# ── Episode discovery (same pattern as GIF scripts) ──────────────────────────

def discover_episodes(data_dir, scene_filter=None):
    """Walk data_dir and return list of (scene, homotopy_safe, ep_id, pkl_path)."""
    episodes = []
    for scene in sorted(os.listdir(data_dir)):
        scene_path = os.path.join(data_dir, scene)
        if not os.path.isdir(scene_path):
            continue
        if scene in ('images', 'gifs', 'gifs_physics', 'plots'):
            continue
        if scene_filter is not None and scene != scene_filter:
            continue
        for homotopy in sorted(os.listdir(scene_path)):
            homo_path = os.path.join(scene_path, homotopy)
            if not os.path.isdir(homo_path):
                continue
            for fn in sorted(os.listdir(homo_path)):
                if fn.endswith('.pkl') and not fn.startswith('run_summary'):
                    ep_id = os.path.splitext(fn)[0]
                    episodes.append((scene, homotopy, ep_id,
                                     os.path.join(homo_path, fn)))
    return episodes


def _load_episode(pkl_path):
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


# ── Summary plot — one PNG per scene ─────────────────────────────────────────

def plot_scene_summary(episodes_for_scene, scene, out_dir, dpi=150,
                       skip_existing=True):
    """All episodes in one figure, coloured by homotopy."""
    out_path = os.path.join(out_dir, f'{scene}_summary.png')
    if skip_existing and os.path.exists(out_path):
        return out_path, True   # (path, skipped)

    # Load all episodes
    records = []
    for (_, homotopy, ep_id, pkl_path) in episodes_for_scene:
        try:
            records.append((homotopy, _load_episode(pkl_path)))
        except Exception as exc:
            print(f'  [ plot ] WARN: could not load {ep_id}: {exc}')

    if not records:
        return None, False

    dynamic_palette = {}
    homotopy_counts = defaultdict(int)
    for h, _ in records:
        homotopy_counts[h] += 1

    n_eps = len(records)
    # Alpha: dense datasets fade individual lines; sparse datasets stay bold.
    alpha = float(np.clip(8.0 / max(n_eps / len(homotopy_counts), 1), 0.07, 0.55))

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_xlabel('x (m)', fontsize=10)
    ax.set_ylabel('y (m)', fontsize=10)

    # Draw obstacles from first episode
    _draw_obstacles(ax, records[0][1].get('obstacles', []))

    # Plot actual trajectories coloured by homotopy
    for homotopy, ep in records:
        obs = ep['obs']          # (T, 9)
        p_xy = obs[:, 3:5]       # actual position XY
        color = _homotopy_color(homotopy, dynamic_palette)
        ax.plot(p_xy[:, 0], p_xy[:, 1],
                color=color, linewidth=0.8, alpha=alpha, zorder=2)

    # Start markers (one per homotopy, representative)
    plotted_starts = set()
    for homotopy, ep in records:
        if homotopy in plotted_starts:
            continue
        obs = ep['obs']
        color = _homotopy_color(homotopy, dynamic_palette)
        ax.plot(obs[0, 3], obs[0, 4], 'o', color=color,
                markersize=5, zorder=5, markeredgecolor='white', markeredgewidth=0.5)
        plotted_starts.add(homotopy)

    # Legend
    legend_handles = [
        Line2D([0], [0], color=_homotopy_color(h, dynamic_palette),
               linewidth=2.0, label=f'{h}  (n={homotopy_counts[h]})')
        for h in sorted(homotopy_counts)
    ]
    # Obstacle legend entries
    legend_handles += [
        mpatches.Patch(facecolor='#555555', edgecolor='#222222',
                       alpha=0.65, label='obstacle'),
        Line2D([0], [0], color='#cc4400', linewidth=0.8, linestyle='--',
               alpha=0.6, label='safety margin (rotor)'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7,
              framealpha=0.85)

    title_parts = [f'Scene: {scene}', f'{n_eps} episodes',
                   f'{len(homotopy_counts)} homotopies']
    ax.set_title('  |  '.join(title_parts), fontsize=10, pad=8)

    os.makedirs(out_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_path, False


# ── Per-episode plot ──────────────────────────────────────────────────────────

def plot_episode(scene, homotopy_safe, ep_id, pkl_path,
                 out_dir, dpi=150, skip_existing=True):
    """Commanded p_des (dashed) vs actual p (solid), start/end markers."""
    ep_dir  = os.path.join(out_dir, scene, homotopy_safe)
    out_path = os.path.join(ep_dir, f'{ep_id}_plot.png')
    if skip_existing and os.path.exists(out_path):
        return out_path, True

    try:
        ep = _load_episode(pkl_path)
    except Exception as exc:
        print(f'  [ plot ] WARN: could not load {ep_id}: {exc}')
        return None, False

    obs = ep['obs']       # (T, 9): [p_des(3) | p(3) | v(3)]
    p_des_xy = obs[:, 0:2]
    p_act_xy = obs[:, 3:5]
    T = len(obs)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.set_xlabel('x (m)', fontsize=9)
    ax.set_ylabel('y (m)', fontsize=9)

    _draw_obstacles(ax, ep.get('obstacles', []))

    # Colour trajectory by time (viridis) for commanded, flat red for actual
    n = len(p_des_xy)
    cmap = plt.get_cmap('Blues')
    for i in range(n - 1):
        frac = i / max(n - 2, 1)
        ax.plot(p_des_xy[i:i+2, 0], p_des_xy[i:i+2, 1],
                color=cmap(0.35 + 0.55 * frac), linewidth=1.0,
                alpha=0.7, zorder=2)

    ax.plot(p_act_xy[:, 0], p_act_xy[:, 1],
            color='#cc2222', linewidth=1.2, alpha=0.85, zorder=3,
            label='actual p')

    # Start / end markers
    ax.plot(p_act_xy[0, 0],  p_act_xy[0, 1],  'o', color='#00aa00',
            markersize=7, zorder=6, label='start', markeredgecolor='white')
    ax.plot(p_act_xy[-1, 0], p_act_xy[-1, 1], '*', color='#aa0000',
            markersize=9, zorder=6, label='end',   markeredgecolor='white')

    # Legend
    legend_handles = [
        Line2D([0], [0], color='#4477aa', linewidth=1.5, label='p_des (commanded)'),
        Line2D([0], [0], color='#cc2222', linewidth=1.5, label='p (actual)'),
        Line2D([0], [0], marker='o', color='#00aa00', linewidth=0,
               markersize=6, label='start'),
        Line2D([0], [0], marker='*', color='#aa0000', linewidth=0,
               markersize=8, label='end'),
    ]
    ax.legend(handles=legend_handles, loc='upper right', fontsize=7)

    # Title: metadata
    meta   = ep.get('metadata', {})
    cf     = meta.get('contact_fraction', ep.get('contact_fraction', '?'))
    dur    = meta.get('total_time', '?')
    ctrl   = ep.get('controller', '?')
    stress_tag = ''
    if ep.get('stress', False):
        gv = ep.get('gate_verdicts', {})
        stress_tag = (f'  |  STRESS={ep["stress_case"]}'
                      f'  gate[C={gv.get("contact","?")}, F={gv.get("floor","?")}]')
    ax.set_title(
        f'{ep_id}\n'
        f'scene={scene}  homotopy={ep.get("homotopy","?")}  '
        f'T={dur:.1f}s  contact={cf:.4f}  ctrl={ctrl}'
        f'{stress_tag}',
        fontsize=6.5, pad=6)

    os.makedirs(ep_dir, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    return out_path, False


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='E5 U5: 2-D overhead trajectory overview plots.')
    p.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                   help='Root of episode pickles (default: logs/uav_expert_data)')
    p.add_argument('--out-dir', default=None,
                   help='Output root (default: <data-dir>/plots)')
    p.add_argument('--scene', default=None,
                   help='Process only this scene (default: all)')
    p.add_argument('--mode', default='summary',
                   choices=['summary', 'per-episode', 'both'],
                   help='Plot mode (default: summary)')
    p.add_argument('--max-episodes', type=int, default=None,
                   help='Cap total episodes processed')
    p.add_argument('--per-homotopy', type=int, default=None,
                   help='Keep at most N episodes per (scene, homotopy) bucket')
    p.add_argument('--dpi', type=int, default=150,
                   help='Output PNG DPI (default: 150)')
    p.add_argument('--no-skip', action='store_true',
                   help='Regenerate existing plots (default: skip)')
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args    = parse_args()
    out_dir = args.out_dir or os.path.join(args.data_dir, 'plots')
    skip    = not args.no_skip

    episodes = discover_episodes(args.data_dir, scene_filter=args.scene)
    if not episodes:
        print(f'[ plot ] No episodes found in {args.data_dir}')
        sys.exit(1)

    # Selective per-homotopy bucketing (same logic as GIF scripts)
    if args.per_homotopy is not None:
        from collections import defaultdict as _dd
        buckets = _dd(list)
        for ep in episodes:
            buckets[(ep[0], ep[1])].append(ep)
        episodes = [ep for eps in buckets.values()
                    for ep in eps[:args.per_homotopy]]

    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]

    print(f'[ plot ] {len(episodes)} episodes  mode={args.mode}  '
          f'dpi={args.dpi}  out={out_dir}')

    # Group by scene
    by_scene = defaultdict(list)
    for ep_info in episodes:
        by_scene[ep_info[0]].append(ep_info)

    do_summary    = args.mode in ('summary', 'both')
    do_per_episode = args.mode in ('per-episode', 'both')

    gen_summary  = 0
    gen_episode  = 0
    skip_summary = 0
    skip_episode = 0
    errors       = []

    for scene, scene_eps in sorted(by_scene.items()):
        print(f'\n[ plot ] Scene: {scene}  ({len(scene_eps)} episodes)')

        if do_summary:
            try:
                path, was_skip = plot_scene_summary(
                    scene_eps, scene, out_dir,
                    dpi=args.dpi, skip_existing=skip)
                if was_skip:
                    skip_summary += 1
                    print(f'  [ plot ] summary: SKIP (exists)')
                elif path:
                    gen_summary += 1
                    print(f'  [ plot ] summary → {os.path.relpath(path, _REPO)}')
            except Exception as exc:
                errors.append((f'{scene}_summary', str(exc)))
                print(f'  [ plot ] summary ERROR: {exc}')

        if do_per_episode:
            for scene_name, homotopy_safe, ep_id, pkl_path in scene_eps:
                try:
                    path, was_skip = plot_episode(
                        scene_name, homotopy_safe, ep_id, pkl_path,
                        out_dir, dpi=args.dpi, skip_existing=skip)
                    if was_skip:
                        skip_episode += 1
                    elif path:
                        gen_episode += 1
                        if gen_episode <= 3 or gen_episode % 50 == 0:
                            print(f'  [ plot ] {ep_id} → {os.path.relpath(path, _REPO)}')
                except Exception as exc:
                    errors.append((ep_id, str(exc)))
                    print(f'  [ plot ] ERROR {ep_id}: {exc}')

    print()
    print('=' * 60)
    print('[ plot ] Done.')
    if do_summary:
        print(f'  Summaries generated : {gen_summary}')
        print(f'  Summaries skipped   : {skip_summary}')
    if do_per_episode:
        print(f'  Episodes generated  : {gen_episode}')
        print(f'  Episodes skipped    : {skip_episode}')
    if errors:
        print(f'  Errors              : {len(errors)}')
        for name, err in errors[:10]:
            print(f'    {name}: {err}')
    print(f'  Output              : {out_dir}/')
    print('=' * 60)


if __name__ == '__main__':
    main()
