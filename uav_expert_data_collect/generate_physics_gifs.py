"""Epoch 5 WS-D (U3) — Physics replay GIFs from Epoch 4 state pickles.

Unlike generate_trajectory_gifs.py (state injection via mj_forward), this
script RE-RUNS the exact original physics simulation (mj_step) and captures
frames live. Contact events appear as red-border / CONTACT overlays; a
proximity bar shows nearest-obstacle distance each frame.

Seed recovery: episode_id always ends with _{trial_seed:07d} (see collect.py:103).
int(ep_id.split('_')[-1]) recovers the original trial seed reliably.

Output layout
-------------
logs/uav_expert_data/gifs_physics/{scene}/{homotopy_safe}/{ep_id}_physics.gif

Usage
-----
# Smoke test — 3 pillar episodes (most likely to show contact events)
python uav_expert_data_collect/generate_physics_gifs.py --max-episodes 3 --scene pillars

# Full run, all scenes
python uav_expert_data_collect/generate_physics_gifs.py

# Single scene with frame stride 3 (smaller GIFs, faster)
python uav_expert_data_collect/generate_physics_gifs.py --scene corridor --frame-stride 3

Environment
-----------
Requires MUJOCO_GL=egl and a GPU for offscreen rendering (mujoco.Renderer).
See generate_physics_gifs.sh for the correct SBATCH setup.
"""

import argparse
import os
import pickle
import sys
from collections import defaultdict

import cv2
import imageio
import mujoco
import numpy as np
from tqdm import tqdm

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, _REPO)

from uav_expert_data_collect.generator import (
    SCENE_XMLS,
    SCENE_OBSTACLES,
    _build_traj_and_init,
    _is_obstacle_contact,
    _make_pid,
)

_DEFAULT_DATA_DIR = os.path.join(_REPO, 'logs', 'uav_expert_data')
_TRACK_CAM_NAME   = 'track'


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description='Epoch 5 WS-D: physics replay GIFs from UAV state pickles.')
    p.add_argument('--data-dir',     default=_DEFAULT_DATA_DIR,
                   help='Root dir of Epoch 4 state pickles (default: logs/uav_expert_data)')
    p.add_argument('--out-dir',      default=None,
                   help='Output root (default: <data-dir>/gifs_physics)')
    p.add_argument('--resolution',   type=int, default=96,
                   help='Render resolution (square). Default: 96')
    p.add_argument('--scene',        default=None, choices=list(SCENE_XMLS),
                   help='Process only this scene (default: all)')
    p.add_argument('--max-episodes', type=int, default=None,
                   help='Cap total episodes (for smoke tests)')
    p.add_argument('--fps',          type=int, default=10,
                   help='GIF framerate. Default: 10')
    p.add_argument('--frame-stride', type=int, default=3,
                   help='Keep every Nth dataset-rate frame (1=33 Hz, 3≈11 Hz). Default: 3')
    p.add_argument('--mp4',          action='store_true',
                   help='Also save MP4 (requires imageio-ffmpeg)')
    p.add_argument('--skip-existing', action='store_true', default=True,
                   help='Skip episodes whose GIF already exists (default: on)')
    p.add_argument('--no-skip',      action='store_true',
                   help='Regenerate even if GIF exists')
    return p.parse_args()


# ── Episode discovery ─────────────────────────────────────────────────────────

def discover_episodes(data_dir, scene_filter=None):
    """Walk data_dir and return list of (scene, homotopy_safe, ep_id, pkl_path)."""
    episodes = []
    for scene in sorted(os.listdir(data_dir)):
        scene_path = os.path.join(data_dir, scene)
        if not os.path.isdir(scene_path):
            continue
        if scene in ('images', 'gifs', 'gifs_physics'):
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


# ── Obstacle distance ─────────────────────────────────────────────────────────

def _nearest_obstacle_dist(pos, obstacles):
    """Signed distance from drone body pos to nearest obstacle surface (metres).

    Uses standard SDF for each obstacle type:
      cylinder : 2-D radial distance to vertical axis minus radius
      box      : standard box SDF (positive outside, negative inside)

    Returns 99.0 when there are no obstacles (empty scene).
    """
    if not obstacles:
        return 99.0

    px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
    min_dist = 99.0

    for obs in obstacles:
        cx, cy, cz = obs['center']

        if obs['type'] == 'cylinder':
            d = np.sqrt((px - cx) ** 2 + (py - cy) ** 2) - obs['radius']

        elif obs['type'] == 'box':
            hx, hy, hz = obs['half_extents']
            qx = abs(px - cx) - hx
            qy = abs(py - cy) - hy
            qz = abs(pz - cz) - hz
            # Standard box SDF: outside = length of clamped-to-positive q
            #                   inside  = largest (least negative) component
            outside = np.sqrt(max(qx, 0.0) ** 2 +
                              max(qy, 0.0) ** 2 +
                              max(qz, 0.0) ** 2)
            inside  = min(max(qx, qy, qz), 0.0)
            d = outside + inside

        else:
            continue

        if d < min_dist:
            min_dist = d

    return float(min_dist)


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _render_overhead(model, data, renderer):
    cam           = mujoco.MjvCamera()
    cam.type      = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = data.qpos[:3]
    cam.distance  = 5.0
    cam.azimuth   = 0.0
    cam.elevation = -90.0
    renderer.update_scene(data, camera=cam)
    return renderer.render().copy()


def _render_track(model, data, renderer, cam_id):
    renderer.update_scene(data, camera=cam_id)
    return renderer.render().copy()


def _burn_text(frame, text, pos, color=(255, 255, 255), scale=0.35, thickness=1):
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _apply_contact_border(frame, width=4):
    frame[:width, :]  = [0, 0, 255]
    frame[-width:, :] = [0, 0, 255]
    frame[:, :width]  = [0, 0, 255]
    frame[:, -width:] = [0, 0, 255]


def _build_frame(renderer, model, data, track_cam_id,
                 is_contact, dist, t, scene, homotopy, res):
    """Render overhead + track, apply overlays, return stitched RGB frame."""
    bp_rgb = _render_overhead(model, data, renderer)
    tr_rgb = _render_track(model, data, renderer, track_cam_id)

    bp_bgr = cv2.cvtColor(bp_rgb, cv2.COLOR_RGB2BGR)
    tr_bgr = cv2.cvtColor(tr_rgb, cv2.COLOR_RGB2BGR)

    # Base text
    _burn_text(bp_bgr, f'{scene}/{homotopy} t={t:.2f}s', (4, 12))
    _burn_text(tr_bgr, 'FPV-physics', (4, 12))

    if is_contact:
        _apply_contact_border(bp_bgr)
        _apply_contact_border(tr_bgr)
        _burn_text(bp_bgr, 'CONTACT', (4, 26),
                   color=(0, 0, 255), scale=0.45, thickness=2)

    # Proximity bar on bp — bottom strip, green→orange→red over 0–0.5 m
    bar_frac  = float(np.clip(dist / 0.5, 0.0, 1.0))
    bar_w     = int(bar_frac * res)
    bar_color = ((0, 255, 0) if dist > 0.30
                 else (0, 165, 255) if dist > 0.15
                 else (0, 0, 255))
    if res > 8:
        bp_bgr[-8:-4, :bar_w] = bar_color
    _burn_text(bp_bgr, f'd={dist:.2f}m',
               pos=(max(0, res - 58), res - 4), scale=0.30)

    bp_rgb = cv2.cvtColor(bp_bgr, cv2.COLOR_BGR2RGB)
    tr_rgb = cv2.cvtColor(tr_bgr, cv2.COLOR_BGR2RGB)
    return np.concatenate([bp_rgb, tr_rgb], axis=1)


# ── Physics replay ────────────────────────────────────────────────────────────

def physics_replay_frames(model, data, renderer, episode, res, frame_stride=3):
    """Re-run original physics from trial_seed and return (frames, n_contact_steps).

    Recovers trial_seed from the last token of episode_id (7-digit zero-padded).
    Re-runs _build_traj_and_init + _make_pid with the same seed, then steps
    mj_step for the full duration, capturing frames at dataset rate / frame_stride.
    """
    ep_id      = episode['episode_id']
    scene      = episode.get('scene',      'unknown')
    homotopy   = episode.get('homotopy',   'unknown')
    controller = episode.get('controller', 'pid_default')
    obstacles  = episode.get('obstacles',  SCENE_OBSTACLES.get(scene, []))

    # Seed is always the last _-separated token of episode_id (7-digit zero-padded)
    trial_seed = int(ep_id.split('_')[-1])

    rng = np.random.default_rng(trial_seed)
    traj_fn, init_pos, dur = _build_traj_and_init(scene, homotopy, rng)

    data.qpos[:3]  = init_pos
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:]   = 0.0
    mujoco.mj_forward(model, data)

    pid    = _make_pid(model, controller)
    dt     = float(model.opt.timestep)
    n_step = int(round(dur / dt))

    track_cam_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, _TRACK_CAM_NAME)
    if track_cam_id < 0:
        raise RuntimeError(f'Camera "{_TRACK_CAM_NAME}" not found in model.')

    # Capture every (dataset_stride * frame_stride) physics steps
    # dataset_stride ≈ 1 / (dt * 33 Hz) — aligns with the 33 Hz collect rate
    dataset_stride = max(1, round(1.0 / (dt * 33.0)))
    capture_every  = dataset_stride * frame_stride

    frames     = []
    n_contact  = 0

    for k in range(n_step):
        t = k * dt
        p_des, v_des, a_des, yaw_des = traj_fn(t)

        p  = data.qpos[:3].copy()
        v  = data.qvel[:3].copy()
        q  = data.qpos[3:7].copy()
        om = data.qvel[3:6].copy()

        u = pid.compute(p, q, v, om, p_des, v_des, a_des, yaw_des)
        data.ctrl[:4] = u
        mujoco.mj_step(model, data)

        is_contact = any(
            _is_obstacle_contact(model, data.contact[ci])
            for ci in range(data.ncon))
        if is_contact:
            n_contact += 1

        if k % capture_every == 0:
            dist  = _nearest_obstacle_dist(data.qpos[:3], obstacles)
            frame = _build_frame(renderer, model, data, track_cam_id,
                                 is_contact, dist, t, scene, homotopy, res)
            frames.append(frame)

    return frames, n_contact


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args     = parse_args()
    skip     = args.skip_existing and not args.no_skip
    out_root = args.out_dir or os.path.join(args.data_dir, 'gifs_physics')

    episodes = discover_episodes(args.data_dir, scene_filter=args.scene)
    if not episodes:
        print(f'[ phys-gif ] No episodes found in {args.data_dir}')
        sys.exit(1)

    if args.max_episodes is not None:
        episodes = episodes[:args.max_episodes]

    print(f'[ phys-gif ] {len(episodes)} episodes  '
          f'res={args.resolution}  fps={args.fps}  stride={args.frame_stride}  '
          f'mp4={args.mp4}  skip={skip}')

    by_scene = defaultdict(list)
    for ep_info in episodes:
        by_scene[ep_info[0]].append(ep_info)

    generated = 0
    skipped   = 0
    errors    = []

    for scene, scene_eps in sorted(by_scene.items()):
        print(f'\n[ phys-gif ] Scene: {scene} ({len(scene_eps)} episodes)')

        xml_path = SCENE_XMLS.get(scene)
        if xml_path is None or not os.path.exists(xml_path):
            print(f'[ phys-gif ] WARN: XML not found for "{scene}", skipping.')
            continue

        model    = mujoco.MjModel.from_xml_path(xml_path)
        data     = mujoco.MjData(model)
        renderer = mujoco.Renderer(model,
                                   height=args.resolution,
                                   width=args.resolution)

        for scene_name, homotopy, ep_id, pkl_path in tqdm(
                scene_eps, desc=f'  {scene}'):

            gif_dir  = os.path.join(out_root, scene_name, homotopy)
            gif_path = os.path.join(gif_dir, f'{ep_id}_physics.gif')

            if skip and os.path.exists(gif_path):
                skipped += 1
                continue

            try:
                with open(pkl_path, 'rb') as f:
                    episode = pickle.load(f)

                frames, n_contact = physics_replay_frames(
                    model, data, renderer, episode, args.resolution,
                    frame_stride=args.frame_stride)

                if not frames:
                    tqdm.write(f'[ phys-gif ] WARN: {ep_id} — 0 frames')
                    continue

                os.makedirs(gif_dir, exist_ok=True)
                imageio.mimsave(gif_path, frames, fps=args.fps, loop=0)

                if args.mp4:
                    mp4_path = os.path.join(gif_dir, f'{ep_id}_physics.mp4')
                    imageio.mimsave(mp4_path, frames, fps=args.fps)

                generated += 1
                if n_contact > 0:
                    tqdm.write(
                        f'[ phys-gif ] CONTACT: {ep_id}  steps={n_contact}')

            except Exception as exc:
                errors.append((ep_id, str(exc)))
                tqdm.write(f'[ phys-gif ] ERROR: {ep_id} — {exc}')

        if hasattr(renderer, 'close'):
            renderer.close()

    print()
    print('=' * 60)
    print('[ phys-gif ] Done.')
    print(f'  Generated:  {generated} GIFs')
    print(f'  Skipped:    {skipped} (already existed)')
    print(f'  Errors:     {len(errors)}')
    print(f'  Output:     {out_root}/')
    if errors:
        print('  Failed episodes:')
        for ep_id, err in errors[:20]:
            print(f'    {ep_id}: {err}')
    print('=' * 60)


if __name__ == '__main__':
    main()
