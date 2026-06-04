"""Validate generated dataset statistics against Phase 4-α UAV-Flow targets.

Usage
-----
python uav_expert_data_collect/stats_validator.py --data-dir logs/uav_expert_data/corridor
python uav_expert_data_collect/stats_validator.py --data-dir logs/uav_expert_data  --all-scenes
"""

import argparse
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

_REPO       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_STATS_JSON = os.path.join(
    _REPO,
    'logs_in_develop/Gen11/Epoch4_expert_data/phase4_alpha_uavflow_stats.json',
)


def load_episodes(data_dir):
    episodes = []
    for root, _, files in os.walk(data_dir):
        for fn in sorted(files):
            if fn.endswith('.pkl') and not fn.startswith('run_summary'):
                path = os.path.join(root, fn)
                try:
                    with open(path, 'rb') as f:
                        episodes.append(pickle.load(f))
                except Exception as e:
                    print(f'  [ warn ] Could not load {path}: {e}')
    return episodes


def _pct(arr, p):
    return float(np.percentile(arr, p))


def compute_dataset_stats(episodes):
    speeds_mps, ep_lens, act_norms, homotopy_counts = [], [], [], {}
    for ep in episodes:
        obs     = ep['obs']          # (T, 6)
        actions = ep['actions']      # (T-1, 3)
        v       = obs[:, 3:6]        # (T, 3) velocity
        speed   = np.linalg.norm(v, axis=1)
        speeds_mps.extend(speed.tolist())
        ep_lens.append(len(obs))
        act_norms.extend(np.linalg.norm(actions, axis=1).tolist())
        h = ep.get('homotopy', 'unknown')
        homotopy_counts[h] = homotopy_counts.get(h, 0) + 1

    s = np.array(speeds_mps)
    e = np.array(ep_lens)
    a = np.array(act_norms)

    return {
        'n_episodes':       len(episodes),
        'homotopy_counts':  homotopy_counts,
        'speed_mps': {
            'mean': round(float(s.mean()), 4), 'median': round(float(np.median(s)), 4),
            'p5':   round(_pct(s, 5),  4),     'p95':    round(_pct(s, 95), 4),
            'max':  round(float(s.max()), 4),
        },
        'episode_length_steps': {
            'mean': round(float(e.mean()), 1), 'median': float(np.median(e)),
            'min':  int(e.min()),               'max':    int(e.max()),
        },
        'action_delta_norm_m': {
            'mean': round(float(a.mean()), 5), 'p95': round(_pct(a, 95), 5),
        },
    }


def print_comparison(gen_stats, ref_stats):
    print('\n' + '=' * 60)
    print('  Generated dataset vs UAV-Flow Phase 4-α targets')
    print('=' * 60)
    print(f'  Episodes generated  : {gen_stats["n_episodes"]}')
    print(f'  Homotopy breakdown  : {gen_stats["homotopy_counts"]}')
    print()
    print('  Speed (m/s):')
    gs = gen_stats['speed_mps']
    print(f'    Generated  mean={gs["mean"]:.3f}  median={gs["median"]:.3f}  '
          f'p95={gs["p95"]:.3f}')
    print(f'    UAV-Flow   velocity unknown (no timestamps in dataset)')
    print(f'    Our target 0.30–0.50 m/s  (RESEARCH §3.3)')
    ok_speed = 0.15 <= gs['mean'] <= 0.80
    print(f'    STATUS: {"✅ OK" if ok_speed else "⚠️  CHECK — mean speed outside 0.15–0.80 m/s"}')
    print()
    print('  Episode length (steps at ~33 Hz):')
    el = gen_stats['episode_length_steps']
    ref_ep = ref_stats.get('episode_length_waypoints', {}) if ref_stats else {}
    print(f'    Generated  mean={el["mean"]:.0f}  median={el["median"]:.0f}  '
          f'[{el["min"]}, {el["max"]}]')
    print(f'    UAV-Flow   median={ref_ep.get("median","?")} waypoints '
          f'(different Hz — not directly comparable)')
    print()
    print('  Action Δp_des norm (m per step):')
    an = gen_stats['action_delta_norm_m']
    print(f'    Generated  mean={an["mean"]:.4f}  p95={an["p95"]:.4f}')
    # At 33 Hz, 0.3 m/s → 0.009 m/step; 0.5 m/s → 0.015 m/step
    ok_action = 0.002 <= an['mean'] <= 0.05
    print(f'    Expected   0.009–0.015 m/step (0.3–0.5 m/s at 33 Hz)')
    print(f'    STATUS: {"✅ OK" if ok_action else "⚠️  CHECK — action magnitude unexpected"}')
    print('=' * 60 + '\n')


def parse_args():
    p = argparse.ArgumentParser(description='UAV dataset stats validator')
    p.add_argument('--data-dir', required=True,
                   help='Root of uav_expert_data (single scene or all-scenes root)')
    p.add_argument('--all-scenes', action='store_true',
                   help='Recurse through all scene subdirs under --data-dir')
    p.add_argument('--stats-json', default=_STATS_JSON)
    return p.parse_args()


def main():
    args = parse_args()

    ref_stats = None
    if os.path.exists(args.stats_json):
        with open(args.stats_json) as f:
            ref_stats = json.load(f)
    else:
        print(f'[ validate ] Phase 4-α stats not found at {args.stats_json}')

    print(f'[ validate ] Loading episodes from {args.data_dir} …')
    episodes = load_episodes(args.data_dir)
    if not episodes:
        print('[ validate ] No episodes found. Run collect.py first.')
        return

    gen_stats = compute_dataset_stats(episodes)
    print_comparison(gen_stats, ref_stats)

    # Also dump raw stats as JSON alongside the data
    out_json = os.path.join(args.data_dir, 'dataset_stats.json')
    with open(out_json, 'w') as f:
        json.dump(gen_stats, f, indent=2)
    print(f'[ validate ] Stats saved → {out_json}')


if __name__ == '__main__':
    main()
