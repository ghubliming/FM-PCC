"""
One-time patch: set clip_denoised=True in diffusion_config.pkl.

Background
----------
Training saves the full Config object to diffusion_config.pkl. The eval script
loads that pkl — it never reads config/avoiding-d3il-visual.py at runtime.
The checkpoint was trained when clip_denoised=False was set in the config file,
so the pkl carries False. Changing the .py config file has no effect on eval
until the pkl is updated.

Usage (run on cluster from repo root)
--------------------------------------
    # Find the checkpoint directory
    find logs/ -name diffusion_config.pkl

    # Patch it (pass the directory, not the file)
    python diffuser_visual_avoiding_test/fix_pkl_clip_denoised.py \
        logs/avoiding-d3il-visual/visual_avoiding_dpcc/<run_dir>/<seed>

    # Or patch all matching checkpoints at once
    python diffuser_visual_avoiding_test/fix_pkl_clip_denoised.py --find logs/
"""

import argparse
import glob
import json
import os
import pickle
import shutil


def patch_one(ckpt_dir: str, dry_run: bool = False) -> bool:
    pkl_path = os.path.join(ckpt_dir, 'diffusion_config.pkl')
    if not os.path.exists(pkl_path):
        print(f'  SKIP  {pkl_path} — not found')
        return False

    with open(pkl_path, 'rb') as f:
        cfg = pickle.load(f)

    before = cfg._dict.get('clip_denoised', '<missing>')
    if before is True:
        print(f'  OK    {pkl_path} — already True, nothing to do')
        return False

    print(f'  PATCH {pkl_path}  clip_denoised: {before!r} → True')
    if dry_run:
        return False

    # Backup original before overwriting
    bak = pkl_path + '.bak'
    shutil.copy2(pkl_path, bak)

    cfg._dict['clip_denoised'] = True

    with open(pkl_path, 'wb') as f:
        pickle.dump(cfg, f)

    # Also update the JSON sidecar if present (kept for human readability)
    json_path = pkl_path.replace('.pkl', '.json')
    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
        data['clip_denoised'] = True
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f'         updated sidecar: {json_path}')

    # Verify round-trip
    with open(pkl_path, 'rb') as f:
        verify = pickle.load(f)
    assert verify._dict['clip_denoised'] is True, 'round-trip verify failed'
    print(f'         verified ✓  (backup at {bak})')
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_dir', nargs='?',
                        help='Path to checkpoint directory containing diffusion_config.pkl')
    parser.add_argument('--find', metavar='ROOT',
                        help='Recursively find and patch all diffusion_config.pkl under ROOT')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print what would be changed without writing')
    args = parser.parse_args()

    if args.find:
        pkls = sorted(glob.glob(os.path.join(args.find, '**/diffusion_config.pkl'), recursive=True))
        if not pkls:
            print(f'No diffusion_config.pkl found under {args.find}')
            return
        print(f'Found {len(pkls)} pkl file(s) under {args.find}:\n')
        patched = 0
        for pkl in pkls:
            patched += int(patch_one(os.path.dirname(pkl), dry_run=args.dry_run))
        print(f'\nDone — {patched} file(s) patched.')
    elif args.ckpt_dir:
        patch_one(args.ckpt_dir, dry_run=args.dry_run)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
