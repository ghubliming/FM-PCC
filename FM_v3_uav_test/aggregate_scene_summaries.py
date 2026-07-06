"""Aggregate per-(scene,seed) UAV FM eval results into per-scene + cross-scene summaries.

Reads every  logs/UAV_FM/uav-<scene>/plans/<exp_name>/<seed>/<projection>/results.json  produced
by eval_fm_uav.py, rolls up mean±std across seeds per scene → logs/UAV_FM/uav-<scene>/plans/SCENE_SUMMARY.json,
and a top-level logs/UAV_FM/fm_uav_ALL_SCENES_SUMMARY.json across the 4 scenes.

Pure stdlib (no numpy/torch) — runs anywhere, including the cluster login node.

    python FM_v3_uav_test/aggregate_scene_summaries.py
    python FM_v3_uav_test/aggregate_scene_summaries.py --projection fm_only --scenes empty pillars
"""

import argparse
import glob
import json
import os

SCENES = ['empty', 'corridor', 's_curve', 'pillars']
# Fix_10 (2/2): (output_name, group, key) — `results.json`'s `summary` is now grouped into
# physical/constraint/goal/success/timing (see eval_fm_uav.py::rollout_one); `group=None`
# means the field stayed top-level (track_err_mean). output_name follows the same
# group-prefixed leaf-name convention as the source schema, so this rollup's own output
# keys stay consistent with what produced them.
METRICS = [
    ('success_strict_rate', 'success', 'strict_rate'),
    ('phys_contact_frac_mean', 'physical', 'contact_frac_mean'),
    ('goal_dist_mean', 'goal', 'dist_mean'),
    ('goal_reached_rate', 'goal', 'reached_rate'),
    ('track_err_mean', None, 'track_err_mean'),
    ('fm_ms_mean', 'timing', 'fm_ms_mean'),
    ('fm_ms_p95', 'timing', 'fm_ms_p95'),
]


def _extract(summ, group, key):
    return summ.get(group, {}).get(key) if group else summ.get(key)


def _mean_std(xs):
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    var = sum((x - m) ** 2 for x in xs) / len(xs)
    return m, var ** 0.5


def aggregate_scene(scene, projection, logbase):
    # Eval results live under the plans/ tree (any depth): <logbase>/uav-<scene>/plans/**/<proj>/results.json
    pat = os.path.join(logbase, f'uav-{scene}', 'plans', '**', projection, 'results.json')
    files = sorted(glob.glob(pat, recursive=True))
    per_seed = []
    for fp in files:
        try:
            summ = json.load(open(fp)).get('summary', {})
        except (json.JSONDecodeError, OSError):
            continue
        # seed = the <seed> path component (…/<seed>/eval/<proj>/results.json)
        seed = summ.get('seed', os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(fp)))))
        per_seed.append({'seed': seed, **{name: _extract(summ, group, key) for name, group, key in METRICS}})
    if not per_seed:
        return None
    agg = {'scene': scene, 'projection': projection, 'n_seeds': len(per_seed),
           'seeds': [s['seed'] for s in per_seed]}
    for name, group, key in METRICS:
        vals = [s[name] for s in per_seed if isinstance(s.get(name), (int, float))]
        mean, std = _mean_std(vals)
        agg[f'{name}_mean'] = mean
        agg[f'{name}_std'] = std
    agg['per_seed'] = per_seed
    out = os.path.join(logbase, f'uav-{scene}', 'plans', 'SCENE_SUMMARY.json')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, 'w') as f:
        json.dump(agg, f, indent=2)
    print(f'[ agg ] {scene}: {len(per_seed)} seed(s)  '
          f'success={agg.get("success_strict_rate_mean")}  → {out}')
    return agg


def main():
    p = argparse.ArgumentParser(description='Roll up per-scene + cross-scene UAV FM eval summaries.')
    p.add_argument('--scenes', nargs='+', default=SCENES, choices=SCENES)
    p.add_argument('--projection', default='fm_only')
    p.add_argument('--logbase', default='logs/UAV_FM')   # matches config/uav.py logbase
    args = p.parse_args()

    all_scenes = {}
    for s in args.scenes:
        agg = aggregate_scene(s, args.projection, args.logbase)
        if agg is not None:
            all_scenes[s] = {k: agg[k] for k in agg if k != 'per_seed'}
        else:
            print(f'[ agg ] {s}: no eval results found (skipped)')

    top = os.path.join(args.logbase, 'fm_uav_ALL_SCENES_SUMMARY.json')
    with open(top, 'w') as f:
        json.dump({'projection': args.projection, 'scenes': all_scenes}, f, indent=2)
    print(f'[ agg ] cross-scene roll-up ({len(all_scenes)} scene(s)) → {top}')


if __name__ == '__main__':
    main()
