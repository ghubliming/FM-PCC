"""
Collect camera image data for D3IL avoiding task by replaying state expert demos.

Replays each recorded expert trajectory in MuJoCo (EGL offscreen), captures bp-cam
frames at every timestep, and saves them as 96×96 BGR PNG files in the same directory
structure as the visual aligning dataset.

Output layout:
    d3il/environments/dataset/data/avoiding/all_data/
        state/               ← symlink → ../data  (existing state files)
        images/bp-cam/
            env_0/0.png, 1.png, ...
            env_1/0.png, 1.png, ...
        images/inhand-cam/   ← duplicate of bp-cam (Option B); replace with robot_cam later
            env_0/0.png, ...
        train_files.pkl      ← list of filenames for training
        eval_files.pkl

Usage:
    # Smoke: 5 episodes
    python collect_visual_avoiding_data/collect_visual_avoiding_data.py --max-episodes 5

    # Full collection
    python collect_visual_avoiding_data/collect_visual_avoiding_data.py

Environment:
    Must be run from repo root with conda env FMPCC active.
    Requires EGL for offscreen rendering on headless cluster:
        export MUJOCO_GL=egl
        export EGL_DEVICE_ID=0

See:
    logs_in_develop/Gen9/VISUAL_AVOIDING_DATA_COLLECTION.md — full design doc
    logs_in_develop/Gen9/camera_image_from_state/CHANGELOG.md — change record
"""

import argparse
import os
import pickle
import sys

import cv2
import numpy as np
from tqdm import tqdm

# ── repo root on sys.path ────────────────────────────────────────────────────
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_D3IL_ROOT = os.path.join(_REPO_ROOT, 'd3il')
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, _D3IL_ROOT)
os.environ.setdefault('D3IL_DIR', os.path.join(_D3IL_ROOT, 'environments', 'd3il'))


def parse_args():
    p = argparse.ArgumentParser(
        description='Collect bp-cam images for D3IL avoiding task from state demos.')
    p.add_argument('--resolution',   type=int,   default=96,
                   help='Output image resolution (square). Default: 96')
    p.add_argument('--train-ratio',  type=float, default=0.9,
                   help='Fraction of episodes used for training. Default: 0.9')
    p.add_argument('--max-episodes', type=int,   default=None,
                   help='Cap number of episodes (useful for smoke tests).')
    p.add_argument('--skip-existing', action='store_true', default=True,
                   help='Skip episodes whose image folder already exists (default: on).')
    p.add_argument('--no-skip',      action='store_true',
                   help='Re-collect even if image folder exists.')
    return p.parse_args()


# ── path helpers ─────────────────────────────────────────────────────────────

def _fw_path(*parts):
    """Build path rooted at d3il/ (mirrors sim_framework_path without import)."""
    return os.path.join(_D3IL_ROOT, *parts)


def _state_dir():
    return _fw_path('environments', 'dataset', 'data', 'avoiding', 'data')


def _all_data_dir():
    return _fw_path('environments', 'dataset', 'data', 'avoiding', 'all_data')


# ── directory setup ───────────────────────────────────────────────────────────

def setup_dirs(all_data):
    """Create output tree and symlink state/ → ../data."""
    os.makedirs(os.path.join(all_data, 'images', 'bp-cam'),     exist_ok=True)
    os.makedirs(os.path.join(all_data, 'images', 'inhand-cam'), exist_ok=True)

    state_link = os.path.join(all_data, 'state')
    if not os.path.exists(state_link):
        state_src = _state_dir()
        os.symlink(state_src, state_link)
        print(f'[ collect ] symlink: {state_link} → {state_src}')


# ── env initialisation ────────────────────────────────────────────────────────

def build_env():
    """
    Instantiate ObstacleAvoidanceEnv in EGL offscreen mode and start the sim.
    The interactive viewer is disabled (render=False); camera capture uses
    MuJoCo's offscreen EGL path directly via env.bp_cam.get_image().
    """
    from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import \
        ObstacleAvoidanceEnv

    print('[ collect ] Initialising ObstacleAvoidanceEnv (EGL offscreen)...')
    env = ObstacleAvoidanceEnv(render=False)
    env.start()
    print('[ collect ] Env ready.')
    return env


# ── replay + capture ──────────────────────────────────────────────────────────

def replay_and_capture(env, des_c_pos, resolution):
    """
    Replay one expert episode and capture bp-cam at each step.

    Args:
        env:        ObstacleAvoidanceEnv (already started)
        des_c_pos:  (T+1, 3) desired TCP positions from state pickle
        resolution: int — output image size (square)

    Returns:
        frames: list of (resolution, resolution, 3) uint8 BGR arrays, length T
    """
    T = len(des_c_pos) - 1

    env.reset()

    # fixed_z — actual z height from the sim after reset (≈ 0.12 m)
    fixed_z = env.robot_state()[2:]   # shape (1,) or scalar

    frames = []
    for t in range(T):
        # Absolute desired TCP position at step t+1 (x, y from demo; z from sim)
        next_xy  = des_c_pos[t + 1, :2]
        cmd_pos  = np.concatenate([next_xy, fixed_z])       # (3,)
        cmd_7d   = np.concatenate([cmd_pos, [0, 1, 0, 0]])  # (7,) pos + quat

        env.step(cmd_7d)

        # Capture bp-cam at current sim state
        frame = env.bp_cam.get_image(width=resolution, height=resolution)
        # get_image already applies vertical_flip; returns BGR uint8 (H, W, 3)
        frames.append(frame.astype(np.uint8))

    return frames


# ── image saving ──────────────────────────────────────────────────────────────

def save_frames(frames, bp_dir, inhand_dir):
    """
    Save frames to bp-cam and inhand-cam directories.
    inhand-cam is a duplicate of bp-cam (Option B — placeholder).
    """
    os.makedirs(bp_dir,     exist_ok=True)
    os.makedirs(inhand_dir, exist_ok=True)

    for t, frame in enumerate(frames):
        path_bp = os.path.join(bp_dir,     f'{t}.png')
        path_ih = os.path.join(inhand_dir, f'{t}.png')
        cv2.imwrite(path_bp, frame)
        cv2.imwrite(path_ih, frame)   # duplicate for now (Option B)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    skip = args.skip_existing and not args.no_skip

    state_dir = _state_dir()
    all_data  = _all_data_dir()

    if not os.path.isdir(state_dir):
        print(f'[ collect ] ERROR: state directory not found: {state_dir}')
        print('  Ensure the D3IL avoiding dataset has been downloaded to that path.')
        sys.exit(1)

    setup_dirs(all_data)

    # ── gather state file list ────────────────────────────────────────────────
    state_files = sorted(f for f in os.listdir(state_dir) if f.endswith('.pkl'))
    if not state_files:
        print(f'[ collect ] ERROR: no .pkl files found in {state_dir}')
        sys.exit(1)

    if args.max_episodes is not None:
        state_files = state_files[:args.max_episodes]

    print(f'[ collect ] {len(state_files)} episodes  '
          f'(resolution={args.resolution}×{args.resolution}, skip_existing={skip})')

    # ── build env once ────────────────────────────────────────────────────────
    env = build_env()

    # ── episode loop ──────────────────────────────────────────────────────────
    collected = 0
    skipped   = 0
    errors    = []

    for fname in tqdm(state_files, desc='Collecting'):
        ep_name    = os.path.splitext(fname)[0]   # e.g. "env_0"
        bp_dir     = os.path.join(all_data, 'images', 'bp-cam',     ep_name)
        inhand_dir = os.path.join(all_data, 'images', 'inhand-cam', ep_name)

        if skip and os.path.isdir(bp_dir) and len(os.listdir(bp_dir)) > 0:
            skipped += 1
            continue

        try:
            with open(os.path.join(state_dir, fname), 'rb') as f:
                env_state = pickle.load(f)

            des_c_pos = env_state['robot']['des_c_pos']   # (T+1, 3)

            frames = replay_and_capture(env, des_c_pos, args.resolution)
            save_frames(frames, bp_dir, inhand_dir)
            collected += 1

        except Exception as exc:
            errors.append((fname, str(exc)))
            tqdm.write(f'[ collect ] WARN: {fname} failed — {exc}')

    # ── train / eval split ────────────────────────────────────────────────────
    n_train = int(len(state_files) * args.train_ratio)
    train_files = state_files[:n_train]
    eval_files  = state_files[n_train:]

    with open(os.path.join(all_data, 'train_files.pkl'), 'wb') as f:
        pickle.dump(train_files, f)
    with open(os.path.join(all_data, 'eval_files.pkl'), 'wb') as f:
        pickle.dump(eval_files, f)

    # ── summary ───────────────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print(f'[ collect ] Done.')
    print(f'  Collected:   {collected} episodes')
    print(f'  Skipped:     {skipped} episodes (already existed)')
    print(f'  Errors:      {len(errors)}')
    print(f'  Train split: {len(train_files)} / Eval split: {len(eval_files)}')
    print(f'  Output dir:  {all_data}')
    if errors:
        print('  Failed episodes:')
        for fn, err in errors:
            print(f'    {fn}: {err}')
    print('=' * 60)


if __name__ == '__main__':
    main()
