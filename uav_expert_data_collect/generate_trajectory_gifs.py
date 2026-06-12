"""Epoch 5 WS-B (Option B1) — Generate GIF/MP4 from Epoch 4 state pickles.

Replays each episode in MuJoCo (EGL offscreen), renders both camera streams,
stitches them side-by-side per frame, and assembles into a GIF (and optionally
MP4). This is for human visual inspection only — not consumed by training.

Output layout
-------------
logs/uav_expert_data/gifs/{scene}/{homotopy_safe}/{ep_id}.gif
logs/uav_expert_data/gifs/{scene}/{homotopy_safe}/{ep_id}.mp4   (if --mp4)

Usage
-----
# Smoke: one episode
python uav_expert_data_collect/generate_trajectory_gifs.py --max-episodes 1

# Full collection
python uav_expert_data_collect/generate_trajectory_gifs.py

# Single scene + mp4
python uav_expert_data_collect/generate_trajectory_gifs.py --scene corridor --mp4

# Subsample frames (every Nth frame)
python uav_expert_data_collect/generate_trajectory_gifs.py --frame-stride 3

Environment
-----------
Requires MUJOCO_GL=egl and GPU for offscreen rendering.
Requires imageio[ffmpeg] for MP4 output.

See: logs_in_develop/Gen11/Epoch5_visual_and_validation/EPOCH5_PLAN.md §3
"""

import argparse
import os
import pickle
import sys

import cv2
import imageio
import mujoco
import numpy as np
from tqdm import tqdm

# ── repo root on sys.path ────────────────────────────────────────────────────
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO)

from uav_expert_data_collect.generator import SCENE_XMLS

_DEFAULT_DATA_DIR = os.path.join(_REPO, 'logs', 'uav_expert_data')
_TRACK_CAM_NAME = 'fpv'


def parse_args():
    p = argparse.ArgumentParser(
        description='Epoch 5 WS-B: generate trajectory GIFs from UAV state pickles.')
    p.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                   help='Root dir of Epoch 4 state pickles')
    p.add_argument('--resolution', type=int, default=96,
                   help='Render resolution (square). Default: 96')
    p.add_argument('--scene', default=None, choices=list(SCENE_XMLS),
                   help='Process only this scene (default: all)')
    p.add_argument('--max-episodes', type=int, default=None,
                   help='Cap number of episodes (for smoke tests).')
    p.add_argument('--per-homotopy', type=int, default=None,
                   help='Keep at most N episodes per (scene, homotopy) bucket. '
                        'Use 1 for minimum-coverage inspection (9 GIFs total across all scenes).')
    p.add_argument('--fps', type=int, default=10,
                   help='GIF/MP4 framerate. Default: 10')
    p.add_argument('--frame-stride', type=int, default=1,
                   help='Keep every Nth frame (1=all, 3=every third). Default: 1')
    p.add_argument('--mp4', action='store_true',
                   help='Also save MP4 (requires imageio-ffmpeg).')
    p.add_argument('--overlay', action='store_true', default=True,
                   help='Burn episode info text into frames (default: on).')
    p.add_argument('--no-overlay', action='store_true',
                   help='Disable text overlay on frames.')
    p.add_argument('--skip-existing', action='store_true', default=True,
                   help='Skip episodes whose GIF already exists (default: on).')
    p.add_argument('--no-skip', action='store_true',
                   help='Regenerate even if GIF exists.')
    return p.parse_args()


# ── episode discovery (shared pattern with collect_camera_images.py) ──────────

def discover_episodes(data_dir, scene_filter=None):
    """Walk data_dir and return list of (scene, homotopy_safe, ep_id, pkl_path)."""
    episodes = []
    for scene in sorted(os.listdir(data_dir)):
        scene_path = os.path.join(data_dir, scene)
        if not os.path.isdir(scene_path):
            continue
        if scene in ('images', 'gifs'):
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


# ── rendering ─────────────────────────────────────────────────────────────────

def _render_overhead(model, data, renderer):
    """Bird's-eye view via free camera."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = data.qpos[:3]
    cam.distance = 5.0
    cam.azimuth = 0.0
    cam.elevation = -90.0
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


def _render_track(model, data, renderer, cam_id):
    """Body-mounted FPV via named camera."""
    renderer.update_scene(data, camera=cam_id)
    return renderer.render().copy()


def _burn_overlay(frame, text, position=(4, 12)):
    """Burn small white text onto a frame (BGR)."""
    cv2.putText(frame, text, position,
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1,
                cv2.LINE_AA)
    return frame


def replay_to_frames(model, data, renderer, episode, resolution,
                     frame_stride=1, overlay=True):
    """Replay episode and return list of stitched (bp | track) RGB frames.

    Returns list of (H, W*2, 3) uint8 RGB arrays (ready for imageio).
    """
    obs      = episode['obs']          # (T, 9)  U2: [p_des(3) | p(3) | v(3)]
    q_stored = episode.get('q', None)  # (T, 4)  D-prep: actual quaternion (E4 U3+); None until re-collect
    T = len(obs)
    scene = episode.get('scene', '?')
    homotopy = episode.get('homotopy', '?')
    ep_id = episode.get('episode_id', '?')

    track_cam_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, _TRACK_CAM_NAME)
    if track_cam_id < 0:
        raise RuntimeError(f'Camera "{_TRACK_CAM_NAME}" not found in model.')

    frames = []
    for t in range(T):
        if t % frame_stride != 0:
            continue

        # Inject state
        data.qpos[:3] = obs[t, 3:6]   # U2: p at columns 3:6 (was :3 in 6D format)
        # D-prep: use stored quaternion for attitude-aware rendering; fall back to level
        data.qpos[3:7] = q_stored[t] if q_stored is not None else [1.0, 0.0, 0.0, 0.0]
        data.qvel[:3] = obs[t, 6:9]   # U2: v at columns 6:9 (was 3:6 in 6D format)
        data.qvel[3:6] = 0.0
        mujoco.mj_forward(model, data)

        bp_rgb = _render_overhead(model, data, renderer)
        track_rgb = _render_track(model, data, renderer, track_cam_id)

        if overlay:
            # Work in BGR for cv2.putText, convert back
            bp_bgr = cv2.cvtColor(bp_rgb, cv2.COLOR_RGB2BGR)
            track_bgr = cv2.cvtColor(track_rgb, cv2.COLOR_RGB2BGR)
            label = f'{scene}/{homotopy} t={t}/{T}'
            _burn_overlay(bp_bgr, label)
            _burn_overlay(track_bgr, 'FPV-onboard')
            bp_rgb = cv2.cvtColor(bp_bgr, cv2.COLOR_BGR2RGB)
            track_rgb = cv2.cvtColor(track_bgr, cv2.COLOR_BGR2RGB)

        # Stitch side by side: [bp | track]
        stitched = np.concatenate([bp_rgb, track_rgb], axis=1)
        frames.append(stitched)

    return frames


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    skip = args.skip_existing and not args.no_skip
    overlay = args.overlay and not args.no_overlay

    episodes = discover_episodes(args.data_dir, scene_filter=args.scene)
    if not episodes:
        print(f'[ gif-gen ] No episodes found in {args.data_dir}')
        sys.exit(1)

    if args.per_homotopy is not None:
        from collections import defaultdict
        buckets = defaultdict(list)
        for ep in episodes:
            buckets[(ep[0], ep[1])].append(ep)
        episodes = [ep for eps in buckets.values() for ep in eps[:args.per_homotopy]]

    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]

    print(f'[ gif-gen ] {len(episodes)} episodes  '
          f'res={args.resolution}  fps={args.fps}  stride={args.frame_stride}  '
          f'mp4={args.mp4}  overlay={overlay}')

    # Group by scene
    from collections import defaultdict
    by_scene = defaultdict(list)
    for ep_info in episodes:
        by_scene[ep_info[0]].append(ep_info)

    generated = 0
    skipped = 0
    errors = []

    for scene, scene_eps in sorted(by_scene.items()):
        print(f'\n[ gif-gen ] Scene: {scene} ({len(scene_eps)} episodes)')

        xml_path = SCENE_XMLS.get(scene)
        if xml_path is None or not os.path.exists(xml_path):
            print(f'[ gif-gen ] WARN: scene XML not found for "{scene}", skipping.')
            continue

        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=args.resolution,
                                    width=args.resolution)

        for scene_name, homotopy, ep_id, pkl_path in tqdm(
                scene_eps, desc=f'  {scene}'):

            gif_dir = os.path.join(
                args.data_dir, 'gifs', scene_name, homotopy)
            gif_path = os.path.join(gif_dir, f'{ep_id}.gif')

            if skip and os.path.exists(gif_path):
                skipped += 1
                continue

            try:
                with open(pkl_path, 'rb') as f:
                    episode = pickle.load(f)

                frames = replay_to_frames(
                    model, data, renderer, episode, args.resolution,
                    frame_stride=args.frame_stride, overlay=overlay)

                if not frames:
                    tqdm.write(f'[ gif-gen ] WARN: {ep_id} produced 0 frames')
                    continue

                os.makedirs(gif_dir, exist_ok=True)
                imageio.mimsave(gif_path, frames, fps=args.fps, loop=0)

                if args.mp4:
                    mp4_path = os.path.join(gif_dir, f'{ep_id}.mp4')
                    imageio.mimsave(mp4_path, frames, fps=args.fps)

                generated += 1

            except Exception as exc:
                errors.append((ep_id, str(exc)))
                tqdm.write(f'[ gif-gen ] WARN: {ep_id} failed — {exc}')

        if hasattr(renderer, 'close'):
            renderer.close()

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print(f'[ gif-gen ] Done.')
    print(f'  Generated:   {generated} GIFs')
    print(f'  Skipped:     {skipped} (already existed)')
    print(f'  Errors:      {len(errors)}')
    print(f'  Output dir:  {args.data_dir}/gifs/')
    if errors:
        print('  Failed episodes:')
        for ep_id, err in errors[:20]:
            print(f'    {ep_id}: {err}')
    print('=' * 60)


if __name__ == '__main__':
    main()
