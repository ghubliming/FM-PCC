"""
Collect camera image data for D3IL avoiding task by replaying state expert demos.

Replays each recorded expert trajectory in MuJoCo (EGL offscreen), captures
BOTH the cage cam (env.bp_cam) and the wrist cam (env.robot.inhand_cam) at every
timestep, and saves them as 96×96 BGR PNG files in the same directory structure
as the visual aligning dataset.

Output layout:
    d3il/environments/dataset/data/avoiding/all_data/
        state/               ← symlink → ../data  (existing state files)
        images/bp-cam/       ← env.bp_cam (cage / third-person)
            env_0/0.png, 1.png, ...
            env_1/0.png, 1.png, ...
        images/inhand-cam/   ← env.robot.inhand_cam (wrist / first-person)
            env_0/0.png, ...
        train_files.pkl      ← list of filenames for training
        eval_files.pkl

Option A vs B (history): earlier versions duplicated bp-cam into inhand-cam
because the avoiding env doesn't re-export `self.inhand_cam`. We now read the
wrist cam directly off the robot (env.robot.inhand_cam) — same MjInhandCamera
instance the aligning env exposes. Zero D3IL files were modified.

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
    MuJoCo's offscreen EGL path directly via env.bp_cam.get_image() and
    env.robot.inhand_cam.get_image().
    """
    from d3il.environments.d3il.envs.gym_avoiding_env.gym_avoiding.envs.avoiding import \
        ObstacleAvoidanceEnv

    print('[ collect ] Initialising ObstacleAvoidanceEnv (EGL offscreen)...')
    env = ObstacleAvoidanceEnv(render=False)
    env.start()

    # Sanity check: the avoiding env doesn't expose self.inhand_cam, but the
    # robot owns one. Fail fast if that assumption breaks (e.g. robot subclass
    # change), so we don't silently fall back to single-camera output.
    if not hasattr(env, 'robot') or getattr(env.robot, 'inhand_cam', None) is None:
        raise RuntimeError(
            '[ collect ] env.robot.inhand_cam not found. The avoiding env\'s '
            'robot must own an MjInhandCamera (MjRobot.py:62) for two-stream '
            'capture. Aborting rather than emit duplicate bp-cam data.'
        )
    print('[ collect ] Env ready (bp_cam + robot.inhand_cam available).')
    return env


def _preflight_cameras(env, resolution):
    """One-shot render from each cam to surface backend errors before the loop."""
    env.reset()
    bp     = env.bp_cam.get_image(           width=resolution, height=resolution, depth=False)
    inhand = env.robot.inhand_cam.get_image( width=resolution, height=resolution, depth=False)
    for name, arr in (('bp_cam', bp), ('inhand_cam', inhand)):
        if arr is None or arr.ndim != 3 or arr.shape[2] != 3:
            raise RuntimeError(f'[ collect ] preflight: {name} returned bad shape '
                               f'{None if arr is None else arr.shape}')
        if arr.std() < 1.0:
            print(f'[ collect ] WARN: {name} preflight image looks ~uniform '
                  f'(std={arr.std():.2f}). Render context may be misconfigured.')
    print(f'[ collect ] preflight ok — bp_cam {bp.shape}, inhand_cam {inhand.shape}')


# ── replay + capture ──────────────────────────────────────────────────────────

def replay_and_capture(env, des_c_pos, resolution):
    """
    Replay one expert episode and capture both cameras at each step.

    Args:
        env:        ObstacleAvoidanceEnv (already started)
        des_c_pos:  (T+1, 3) desired TCP positions from state pickle
        resolution: int — output image size (square)

    Returns:
        (bp_frames, inhand_frames): two parallel lists of length T, each
        element a (resolution, resolution, 3) uint8 BGR array.
    """
    T = len(des_c_pos) - 1

    env.reset()

    # fixed_z — actual z height from the sim after reset (≈ 0.12 m)
    fixed_z = env.robot_state()[2:]   # shape (1,) or scalar

    bp_frames, inhand_frames = [], []
    for t in range(T):
        # Absolute desired TCP position at step t+1 (x, y from demo; z from sim)
        next_xy  = des_c_pos[t + 1, :2]
        cmd_pos  = np.concatenate([next_xy, fixed_z])       # (3,)
        cmd_7d   = np.concatenate([cmd_pos, [0, 1, 0, 0]])  # (7,) pos + quat

        env.step(cmd_7d)

        # Capture both cameras at the current sim state.
        # depth=False is REQUIRED — default depth=True returns (rgb, depth) tuple.
        # get_image returns RGB uint8 (H, W, 3); convert to BGR so cv2.imwrite
        # stores it correctly (the visual aligning loader does BGR→RGB on read).
        bp     = env.bp_cam.get_image(           width=resolution, height=resolution, depth=False)
        inhand = env.robot.inhand_cam.get_image( width=resolution, height=resolution, depth=False)
        bp     = cv2.cvtColor(bp,     cv2.COLOR_RGB2BGR)
        inhand = cv2.cvtColor(inhand, cv2.COLOR_RGB2BGR)
        bp_frames.append(bp.astype(np.uint8))
        inhand_frames.append(inhand.astype(np.uint8))

    return bp_frames, inhand_frames


# ── image saving ──────────────────────────────────────────────────────────────

def save_frames(bp_frames, inhand_frames, bp_dir, inhand_dir):
    """
    Save the two streams to their respective directories.
    Caller guarantees len(bp_frames) == len(inhand_frames).
    """
    os.makedirs(bp_dir,     exist_ok=True)
    os.makedirs(inhand_dir, exist_ok=True)

    for t, (bp, ih) in enumerate(zip(bp_frames, inhand_frames)):
        cv2.imwrite(os.path.join(bp_dir,     f'{t}.png'), bp)
        cv2.imwrite(os.path.join(inhand_dir, f'{t}.png'), ih)


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

    # ── build env once + preflight both cameras ──────────────────────────────
    env = build_env()
    _preflight_cameras(env, args.resolution)

    # ── episode loop ──────────────────────────────────────────────────────────
    collected = 0
    skipped   = 0
    errors    = []

    for fname in tqdm(state_files, desc='Collecting'):
        ep_name    = os.path.splitext(fname)[0]   # e.g. "env_0"
        bp_dir     = os.path.join(all_data, 'images', 'bp-cam',     ep_name)
        inhand_dir = os.path.join(all_data, 'images', 'inhand-cam', ep_name)

        # Skip only if BOTH dirs are populated — guards against partial Option-B
        # runs where bp-cam exists but inhand-cam is missing or duplicated.
        if (skip
                and os.path.isdir(bp_dir)     and len(os.listdir(bp_dir))     > 0
                and os.path.isdir(inhand_dir) and len(os.listdir(inhand_dir)) > 0):
            skipped += 1
            continue

        try:
            with open(os.path.join(state_dir, fname), 'rb') as f:
                env_state = pickle.load(f)

            des_c_pos = env_state['robot']['des_c_pos']   # (T+1, 3)

            bp_frames, inhand_frames = replay_and_capture(env, des_c_pos, args.resolution)
            save_frames(bp_frames, inhand_frames, bp_dir, inhand_dir)
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
