"""Expert data collection driver — Epoch 4.

Usage
-----
python uav_expert_data_collect/collect.py \\
    --scene {empty|corridor|s_curve|pillars} \\
    --n-trials N \\
    [--seed BASE_SEED] \\
    [--gain-variant {pid_default|pid_high_gain|pid_low_gain}] \\
    [--homotopy {all|<specific_label>}] \\
    [--noise-sigma FLOAT] \\
    [--out-dir PATH] \\
    [--reject-limit FLOAT]

Output
------
logs/uav_expert_data/<scene>/<homotopy_safe>/<episode_id>.pkl
logs/uav_expert_data/<scene>/run_summary.json
"""

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from uav_expert_data_collect.generator import (
    run_trial, HOMOTOPY_CLASSES, GAIN_VARIANTS, SCENE_XMLS,
)
from uav_expert_data_collect.dataset_writer import rollout_to_episode, save_episode

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _safe_dir(homotopy):
    """Convert homotopy label to a filesystem-safe directory name."""
    return (homotopy
            .replace('(', '').replace(')', '')
            .replace(',', '_')
            .replace('/', '_')
            .replace(' ', ''))


def parse_args():
    p = argparse.ArgumentParser(description='UAV expert data collector (Epoch 4)')
    p.add_argument('--scene',        required=True, choices=list(SCENE_XMLS))
    p.add_argument('--n-trials',     type=int, default=200)
    p.add_argument('--seed',         type=int, default=0)
    p.add_argument('--gain-variant', default='pid_default', choices=list(GAIN_VARIANTS))
    p.add_argument('--homotopy',     default='all',
                   help='"all" cycles all homotopy classes; pass a specific label to use one')
    p.add_argument('--noise-sigma',  type=float, default=0.02,
                   help='Std of Gaussian noise added to targets (0 = off)')
    p.add_argument('--out-dir',      default=None,
                   help='Root output dir (default: logs/uav_expert_data/<scene>)')
    p.add_argument('--reject-limit', type=float, default=0.60,
                   help='Abort if rolling rejection rate exceeds this (default: 0.60)')
    return p.parse_args()


def main():
    args = parse_args()

    out_root = args.out_dir or os.path.join(_REPO, 'logs', 'uav_expert_data', args.scene)

    homotopy_pool = (HOMOTOPY_CLASSES[args.scene]
                     if args.homotopy == 'all'
                     else [args.homotopy])

    saved          = 0
    rejected       = 0
    reject_counter = Counter()   # U5 Step 1: reason → count
    t0             = time.time()

    print(f'[ collect ] scene={args.scene}  n_trials={args.n_trials}  '
          f'homotopy_pool={homotopy_pool}  gain={args.gain_variant}  '
          f'seed={args.seed}  noise_sigma={args.noise_sigma}')

    for i in range(args.n_trials):
        homotopy   = homotopy_pool[i % len(homotopy_pool)]
        trial_seed = args.seed + i

        rollout = run_trial(
            scene=args.scene,
            homotopy=homotopy,
            gain_variant=args.gain_variant,
            seed=trial_seed,
        )

        # U5 Step 1: run_trial now returns a reject dict instead of None
        if rollout is None or rollout.get('rejected'):
            rejected += 1
            reason = rollout.get('reason', 'unknown') if rollout else 'unknown'
            reject_counter[reason] += 1
            min_z_str = (f'  min_z={rollout["min_z"]:.3f}'
                         if rollout and 'min_z' in rollout else '')
            clip_str  = (f'  clip={rollout["motor_clip_frac"]:.1%}'
                         if rollout and 'motor_clip_frac' in rollout else '')
            print(f'[ collect ] REJECT #{rejected}  reason={reason}{min_z_str}{clip_str}')
            total = rejected + saved
            if total > 20 and rejected / total > args.reject_limit:
                hist = '  '.join(f'{k}={v}' for k, v in reject_counter.most_common())
                print(f'[ collect ] ABORT: rejection rate '
                      f'{rejected/total:.1%} > limit {args.reject_limit:.1%}. '
                      f'HISTOGRAM: {hist}')
                break
            continue

        ep_id = (f'{args.scene}'
                 f'_{_safe_dir(homotopy)}'
                 f'_{args.gain_variant}'
                 f'_{trial_seed:07d}')

        rng     = np.random.default_rng(trial_seed + 99991)
        episode = rollout_to_episode(rollout, episode_id=ep_id,
                                     noise_sigma=args.noise_sigma, rng=rng)
        ep_dir  = os.path.join(out_root, _safe_dir(homotopy))
        path    = save_episode(episode, ep_dir)
        saved  += 1

        if saved % 50 == 0 or saved == 1:
            elapsed = time.time() - t0
            rate    = elapsed / max(saved, 1)
            eta     = rate * max(args.n_trials - (saved + rejected), 0)
            print(f'[ collect ] saved={saved:4d}  rejected={rejected:3d}  '
                  f't={elapsed:.0f}s  ~{rate:.2f}s/ep  '
                  f'ETA≈{eta:.0f}s  last→{os.path.relpath(path, _REPO)}')

    elapsed = time.time() - t0
    total   = saved + rejected
    summary = {
        'scene':            args.scene,
        'gain_variant':     args.gain_variant,
        'seed':             args.seed,
        'n_trials':         args.n_trials,
        'saved':            saved,
        'rejected':         rejected,
        'rejection_rate':   round(rejected / max(total, 1), 4),
        'reject_histogram': dict(reject_counter),
        'elapsed_s':        round(elapsed, 1),
        'sec_per_episode':  round(elapsed / max(saved, 1), 3),
    }
    os.makedirs(out_root, exist_ok=True)
    with open(os.path.join(out_root, 'run_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    hist = '  '.join(f'{k}={v}' for k, v in reject_counter.most_common()) or 'none'
    print(f'[ collect ] DONE  saved={saved}  rejected={rejected}  '
          f'({summary["rejection_rate"]:.1%} rejected)  '
          f'elapsed={elapsed:.0f}s  HISTOGRAM: {hist}')


if __name__ == '__main__':
    main()
