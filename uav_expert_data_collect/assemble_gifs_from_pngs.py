"""Epoch 5 WS-B (Option B2) — Assemble GIFs from pre-rendered PNG frames.

Lightweight CPU-only script: reads WS-A's saved PNGs (bp-cam + track-cam),
stitches them side-by-side, and saves as GIF/MP4. This is an alternative to
generate_trajectory_gifs.py (Option B1) when WS-A has already rendered images.

Output layout
-------------
logs/uav_expert_data/gifs/{scene}/{homotopy_safe}/{ep_id}.gif

Usage
-----
# Full assembly (after WS-A has run)
python uav_expert_data_collect/assemble_gifs_from_pngs.py

# Single scene
python uav_expert_data_collect/assemble_gifs_from_pngs.py --scene empty

# Custom fps + mp4
python uav_expert_data_collect/assemble_gifs_from_pngs.py --fps 15 --mp4

Environment
-----------
CPU-only. Requires imageio and cv2.

See: logs_in_develop/Gen11/Epoch5_visual_and_validation/EPOCH5_PLAN.md §3.2
"""

import argparse
import os
import sys

import cv2
import imageio
import numpy as np
from tqdm import tqdm

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_DEFAULT_DATA_DIR = os.path.join(_REPO, 'logs', 'uav_expert_data')


def parse_args():
    p = argparse.ArgumentParser(
        description='Epoch 5 WS-B (B2): assemble GIFs from pre-rendered PNGs.')
    p.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                   help='Root dir of uav_expert_data (contains images/ subdir)')
    p.add_argument('--scene', default=None,
                   help='Process only this scene (default: all)')
    p.add_argument('--fps', type=int, default=10,
                   help='GIF framerate. Default: 10')
    p.add_argument('--frame-stride', type=int, default=1,
                   help='Keep every Nth frame (1=all). Default: 1')
    p.add_argument('--mp4', action='store_true',
                   help='Also save MP4.')
    p.add_argument('--max-episodes', type=int, default=None,
                   help='Cap number of episodes.')
    p.add_argument('--skip-existing', action='store_true', default=True,
                   help='Skip if GIF already exists.')
    p.add_argument('--no-skip', action='store_true',
                   help='Regenerate all.')
    return p.parse_args()


def discover_image_episodes(data_dir, scene_filter=None):
    """Find episodes that have both bp-cam and track-cam PNG directories.

    Returns list of (scene, homotopy, ep_id, bp_dir, track_dir).
    """
    bp_root = os.path.join(data_dir, 'images', 'bp-cam')
    track_root = os.path.join(data_dir, 'images', 'track-cam')

    if not os.path.isdir(bp_root):
        return []

    episodes = []
    for scene in sorted(os.listdir(bp_root)):
        if scene_filter is not None and scene != scene_filter:
            continue
        scene_bp = os.path.join(bp_root, scene)
        if not os.path.isdir(scene_bp):
            continue
        for homotopy in sorted(os.listdir(scene_bp)):
            homo_bp = os.path.join(scene_bp, homotopy)
            if not os.path.isdir(homo_bp):
                continue
            for ep_id in sorted(os.listdir(homo_bp)):
                bp_dir = os.path.join(homo_bp, ep_id)
                track_dir = os.path.join(track_root, scene, homotopy, ep_id)
                if os.path.isdir(bp_dir) and os.path.isdir(track_dir):
                    episodes.append((scene, homotopy, ep_id, bp_dir, track_dir))
    return episodes


def _load_frames(bp_dir, track_dir, stride=1):
    """Load and stitch bp + track PNGs. Returns list of RGB arrays."""
    bp_files = sorted(os.listdir(bp_dir),
                      key=lambda f: int(os.path.splitext(f)[0]))
    track_files = sorted(os.listdir(track_dir),
                         key=lambda f: int(os.path.splitext(f)[0]))
    n = min(len(bp_files), len(track_files))

    frames = []
    for i in range(0, n, stride):
        bp_img = cv2.imread(os.path.join(bp_dir, bp_files[i]))
        track_img = cv2.imread(os.path.join(track_dir, track_files[i]))
        if bp_img is None or track_img is None:
            continue
        # cv2 reads BGR; convert to RGB for imageio
        bp_rgb = cv2.cvtColor(bp_img, cv2.COLOR_BGR2RGB)
        track_rgb = cv2.cvtColor(track_img, cv2.COLOR_BGR2RGB)
        stitched = np.concatenate([bp_rgb, track_rgb], axis=1)
        frames.append(stitched)
    return frames


def main():
    args = parse_args()
    skip = args.skip_existing and not args.no_skip

    episodes = discover_image_episodes(args.data_dir, scene_filter=args.scene)
    if not episodes:
        print(f'[ assemble ] No image episodes found in {args.data_dir}/images/')
        print('  Run collect_camera_images.py (WS-A) first, or use '
              'generate_trajectory_gifs.py (B1) for direct rendering.')
        sys.exit(1)

    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]

    print(f'[ assemble ] {len(episodes)} episodes  fps={args.fps}  '
          f'stride={args.frame_stride}  mp4={args.mp4}')

    assembled = 0
    skipped = 0
    errors = []

    for scene, homotopy, ep_id, bp_dir, track_dir in tqdm(
            episodes, desc='Assembling'):

        gif_dir = os.path.join(args.data_dir, 'gifs', scene, homotopy)
        gif_path = os.path.join(gif_dir, f'{ep_id}.gif')

        if skip and os.path.exists(gif_path):
            skipped += 1
            continue

        try:
            frames = _load_frames(bp_dir, track_dir, stride=args.frame_stride)
            if not frames:
                tqdm.write(f'[ assemble ] WARN: {ep_id} produced 0 frames')
                continue

            os.makedirs(gif_dir, exist_ok=True)
            imageio.mimsave(gif_path, frames, fps=args.fps, loop=0)

            if args.mp4:
                mp4_path = os.path.join(gif_dir, f'{ep_id}.mp4')
                imageio.mimsave(mp4_path, frames, fps=args.fps)

            assembled += 1

        except Exception as exc:
            errors.append((ep_id, str(exc)))
            tqdm.write(f'[ assemble ] WARN: {ep_id} failed — {exc}')

    print()
    print('=' * 60)
    print(f'[ assemble ] Done.')
    print(f'  Assembled:  {assembled} GIFs')
    print(f'  Skipped:    {skipped} (already existed)')
    print(f'  Errors:     {len(errors)}')
    print(f'  Output dir: {args.data_dir}/gifs/')
    print('=' * 60)


if __name__ == '__main__':
    main()
