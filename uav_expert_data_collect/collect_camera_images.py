"""Epoch 5 WS-A — Collect camera images from Epoch 4 state pickles.

Replays each Epoch 4 episode in MuJoCo (EGL offscreen), injecting stored
qpos/qvel state at each timestep and rendering two camera streams:

  bp-cam    : fixed overhead bird's-eye camera (added per-scene)
  track-cam : body-mounted FPV camera (from quadrotor_modified.xml "track")

Uses direct state injection (qpos/qvel + mj_forward) rather than action
replay to avoid PID drift and guarantee pixel-perfect correspondence
between stored obs and rendered frame.

Output layout
-------------
logs/uav_expert_data/images/bp-cam/{scene}/{homotopy_safe}/{ep_id}/
    0.png, 1.png, …
logs/uav_expert_data/images/track-cam/{scene}/{homotopy_safe}/{ep_id}/
    0.png, 1.png, …

Usage
-----
# Smoke: one episode
python uav_expert_data_collect/collect_camera_images.py --max-episodes 1

# Full collection
python uav_expert_data_collect/collect_camera_images.py

# Specific scene
python uav_expert_data_collect/collect_camera_images.py --scene corridor

Environment
-----------
Requires MUJOCO_GL=egl and GPU for offscreen rendering.

See: logs_in_develop/Gen11/Epoch5_visual_and_validation/EPOCH5_PLAN.md §2
"""

import argparse
import os
import pickle
import sys

import cv2
import mujoco
import numpy as np
from tqdm import tqdm

# ── repo root on sys.path ────────────────────────────────────────────────────
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO)

from uav_expert_data_collect.generator import SCENE_XMLS

_DEFAULT_DATA_DIR = os.path.join(_REPO, 'logs', 'uav_expert_data')

# Camera names in the MuJoCo model.
# "fpv" is nose-mounted, looks along +x body axis (Fix_1: replaces old "track" chase cam).
# "bp_overhead" is a fixed world-frame camera added programmatically at runtime.
_TRACK_CAM_NAME = 'fpv'


def parse_args():
    p = argparse.ArgumentParser(
        description='Epoch 5 WS-A: collect camera images from UAV state pickles.')
    p.add_argument('--data-dir', default=_DEFAULT_DATA_DIR,
                   help='Root dir of Epoch 4 state pickles (default: logs/uav_expert_data)')
    p.add_argument('--resolution', type=int, default=96,
                   help='Output image resolution (square). Default: 96')
    p.add_argument('--scene', default=None, choices=list(SCENE_XMLS),
                   help='Process only this scene (default: all)')
    p.add_argument('--max-episodes', type=int, default=None,
                   help='Cap number of episodes (for smoke tests).')
    p.add_argument('--skip-existing', action='store_true', default=True,
                   help='Skip episodes whose image folders already exist (default: on).')
    p.add_argument('--no-skip', action='store_true',
                   help='Re-collect even if image folders exist.')
    return p.parse_args()


# ── episode discovery ─────────────────────────────────────────────────────────

def discover_episodes(data_dir, scene_filter=None):
    """Walk data_dir and return list of (scene, homotopy_safe, ep_id, pkl_path).

    Expected layout: data_dir/{scene}/{homotopy_safe}/{ep_id}.pkl
    """
    episodes = []
    for scene in sorted(os.listdir(data_dir)):
        scene_path = os.path.join(data_dir, scene)
        if not os.path.isdir(scene_path):
            continue
        if scene == 'images':
            continue  # skip our own output dir
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

def _add_overhead_camera(model):
    """Check if an overhead camera already exists; if not, we'll use a spec."""
    # MuJoCo doesn't allow adding cameras to a compiled model, so we render
    # the overhead view using a mujoco.MjvCamera spec at render time.
    pass  # Handled inline during rendering


def render_frame(model, data, renderer, cam_id, resolution):
    """Render a single frame from the given camera id."""
    renderer.update_scene(data, camera=cam_id)
    img = renderer.render()  # RGB uint8 (H, W, 3)
    return img.copy()


def render_frame_overhead(model, data, renderer, resolution):
    """Render a bird's-eye overhead view using a free-camera spec."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    # Centre on drone position, look straight down
    drone_pos = data.qpos[:3]
    cam.lookat[:] = drone_pos
    cam.distance = 5.0
    cam.azimuth = 0.0
    cam.elevation = -90.0  # straight down
    renderer.update_scene(data, camera=cam)
    img = renderer.render()
    return img.copy()


def replay_and_capture(model, data, renderer, episode, resolution):
    """Inject states from episode and capture frames at each timestep.

    Args:
        model, data : MuJoCo model and data (already loaded for correct scene)
        renderer    : mujoco.Renderer
        episode     : dict from Epoch 4 pickle
        resolution  : int

    Returns:
        (bp_frames, track_frames) — parallel lists of (H, W, 3) uint8 BGR arrays
    """
    obs      = episode['obs']          # (T, 9)  U2: [p_des(3) | p(3) | v(3)]
    q_stored = episode.get('q', None)  # (T, 4)  D-prep: actual quaternion (E4 U3+); None until re-collect
    T = len(obs)

    # Find the "track" camera id
    track_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA,
                                      _TRACK_CAM_NAME)
    if track_cam_id < 0:
        raise RuntimeError(
            f'Camera "{_TRACK_CAM_NAME}" not found in model. '
            f'Check quadrotor_modified.xml has <camera name="track" .../>.')

    bp_frames = []
    track_frames = []

    for t in range(T):
        # Inject position and velocity from stored obs
        p = obs[t, 3:6]   # U2: p at columns 3:6 (was :3 in 6D format)
        v = obs[t, 6:9]   # U2: v at columns 6:9 (was 3:6 in 6D format)
        data.qpos[:3] = p
        # D-prep: use stored quaternion for attitude-aware rendering; fall back to level
        data.qpos[3:7] = q_stored[t] if q_stored is not None else [1.0, 0.0, 0.0, 0.0]
        data.qvel[:3] = v
        data.qvel[3:6] = 0.0

        # Forward kinematics (no physics step — just update positions for render)
        mujoco.mj_forward(model, data)

        # Render both views
        bp_img = render_frame_overhead(model, data, renderer, resolution)
        track_img = render_frame(model, data, renderer, track_cam_id, resolution)

        # MuJoCo renderer returns RGB; convert to BGR for cv2.imwrite
        # (downstream loaders do BGR→RGB on read, matching D3IL convention)
        bp_bgr = cv2.cvtColor(bp_img, cv2.COLOR_RGB2BGR)
        track_bgr = cv2.cvtColor(track_img, cv2.COLOR_RGB2BGR)

        bp_frames.append(bp_bgr)
        track_frames.append(track_bgr)

    return bp_frames, track_frames


# ── saving ────────────────────────────────────────────────────────────────────

def save_frames(frames, out_dir):
    """Save list of BGR frames as numbered PNGs."""
    os.makedirs(out_dir, exist_ok=True)
    for t, frame in enumerate(frames):
        cv2.imwrite(os.path.join(out_dir, f'{t}.png'), frame)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    skip = args.skip_existing and not args.no_skip
    resolution = args.resolution
    data_dir = args.data_dir

    episodes = discover_episodes(data_dir, scene_filter=args.scene)
    if not episodes:
        print(f'[ cam-collect ] No episodes found in {data_dir}')
        sys.exit(1)

    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]

    print(f'[ cam-collect ] {len(episodes)} episodes  '
          f'resolution={resolution}×{resolution}  skip_existing={skip}')

    # Group episodes by scene to reuse model/renderer
    from collections import defaultdict
    by_scene = defaultdict(list)
    for ep_info in episodes:
        by_scene[ep_info[0]].append(ep_info)

    collected = 0
    skipped = 0
    errors = []

    for scene, scene_eps in sorted(by_scene.items()):
        print(f'\n[ cam-collect ] Scene: {scene} ({len(scene_eps)} episodes)')

        xml_path = SCENE_XMLS.get(scene)
        if xml_path is None or not os.path.exists(xml_path):
            print(f'[ cam-collect ] WARN: scene XML not found for "{scene}", skipping.')
            continue

        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        renderer = mujoco.Renderer(model, height=resolution, width=resolution)

        for scene_name, homotopy, ep_id, pkl_path in tqdm(
                scene_eps, desc=f'  {scene}'):

            bp_dir = os.path.join(
                data_dir, 'images', 'bp-cam', scene_name, homotopy, ep_id)
            track_dir = os.path.join(
                data_dir, 'images', 'track-cam', scene_name, homotopy, ep_id)

            # Skip if both dirs are populated
            if (skip
                    and os.path.isdir(bp_dir) and len(os.listdir(bp_dir)) > 0
                    and os.path.isdir(track_dir) and len(os.listdir(track_dir)) > 0):
                skipped += 1
                continue

            try:
                with open(pkl_path, 'rb') as f:
                    episode = pickle.load(f)

                bp_frames, track_frames = replay_and_capture(
                    model, data, renderer, episode, resolution)
                save_frames(bp_frames, bp_dir)
                save_frames(track_frames, track_dir)
                collected += 1

            except Exception as exc:
                errors.append((ep_id, str(exc)))
                tqdm.write(f'[ cam-collect ] WARN: {ep_id} failed — {exc}')

        if hasattr(renderer, 'close'):
            renderer.close()

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print(f'[ cam-collect ] Done.')
    print(f'  Collected:   {collected} episodes')
    print(f'  Skipped:     {skipped} episodes (already existed)')
    print(f'  Errors:      {len(errors)}')
    print(f'  Output dir:  {data_dir}/images/')
    if errors:
        print('  Failed episodes:')
        for ep_id, err in errors[:20]:
            print(f'    {ep_id}: {err}')
        if len(errors) > 20:
            print(f'    ... and {len(errors) - 20} more')
    print('=' * 60)


if __name__ == '__main__':
    main()
