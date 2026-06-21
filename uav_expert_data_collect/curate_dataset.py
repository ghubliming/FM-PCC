"""Phase-0 prep helper — curate the raw E4 collection into a training-only dataset.

Copies ONLY accepted episode pickles for the chosen scenes into a versioned, manifest-tracked
tree that the FM trainer reads (`data/uav_fm/v1/<scene>/*.pkl`). Skips `run_summary*`,
the `_stress` tree, and any old/rejected pillars — so the trainer never ingests debug clutter
(see EPOCH6_PLAN.md §0b). Walks homotopy subdirs of each raw scene dir.

Usage:
    python uav_expert_data_collect/curate_dataset.py \
        --scenes empty corridor s_curve pillars \
        --pillars-src logs/uav_expert_data/pillars_v2 \
        --out data/uav_fm/v1
"""

import argparse
import datetime
import glob
import json
import os
import shutil
import subprocess

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SCENES = ['empty', 'corridor', 's_curve', 'pillars']


def _git_rev():
    try:
        return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'],
                                       cwd=_REPO).decode().strip()
    except Exception:
        return 'unknown'


def _episode_pkls(scene_root):
    """All accepted episode pkls under a raw scene dir (recurses homotopy subdirs)."""
    out = []
    for path in glob.glob(os.path.join(scene_root, '**', '*.pkl'), recursive=True):
        base = os.path.basename(path)
        if base.startswith('run_summary') or base.startswith('manifest'):
            continue
        out.append(path)
    return sorted(out)


def main():
    p = argparse.ArgumentParser(description='Curate raw E4 UAV data into a training dataset.')
    p.add_argument('--scenes', nargs='+', default=SCENES, choices=SCENES)
    p.add_argument('--raw-root', default=os.path.join(_REPO, 'logs', 'uav_expert_data'),
                   help='Raw E4 collection root (default: logs/uav_expert_data).')
    p.add_argument('--pillars-src', default=None,
                   help='Override raw dir for pillars (e.g. the recollected logs/uav_expert_data/pillars_v2).')
    p.add_argument('--out', default=os.path.join(_REPO, 'data', 'uav_fm', 'v1'),
                   help='Curated output root (default: data/uav_fm/v1).')
    args = p.parse_args()

    manifest = {
        'generated': datetime.datetime.now().isoformat(timespec='seconds'),
        'git_rev': _git_rev(),
        'schema': {'horizon': 8, 'transition_dim': 12, 'obs_dim': 9, 'action_dim': 3,
                   'obs_layout': '[p_des(3) | p(3) | v(3)]', 'action': 'Δp_des(3)'},
        'scenes': {},
    }

    for scene in args.scenes:
        src = args.pillars_src if (scene == 'pillars' and args.pillars_src) \
            else os.path.join(args.raw_root, scene)
        if not os.path.isdir(src):
            raise FileNotFoundError(f'Raw scene dir not found: {src}')
        dst = os.path.join(args.out, scene)
        os.makedirs(dst, exist_ok=True)

        pkls = _episode_pkls(src)
        for fp in pkls:
            shutil.copy2(fp, os.path.join(dst, os.path.basename(fp)))
        manifest['scenes'][scene] = {'source': os.path.relpath(src, _REPO), 'n_episodes': len(pkls)}
        print(f'[ curate ] {scene}: {len(pkls)} episodes  ←  {src}')

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    total = sum(s['n_episodes'] for s in manifest['scenes'].values())
    print(f'[ curate ] wrote {total} episodes + manifest.json → {args.out}')


if __name__ == '__main__':
    main()
