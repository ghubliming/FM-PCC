"""UAV fork of the FMv3-ODE-selectable data loader.

Stripped to the single ``uav-<scene>`` branch — no D4RL / Minari / D3IL dependencies
(the source ``avoiding-d3il`` branch imported ``d3il.agents.utils.sim_path``; the UAV
fork must not depend on D3IL). The episode contract handed to ``ReplayBuffer.add_path``
is identical to the source avoiding branch, so the generic ``SequenceDataset`` is reused
unchanged.

env string convention (mirrors d4rl's ``hopper-medium-replay`` style):
    'uav-all'      → pool all 4 scenes
    'uav-empty'    → single scene
    'uav-corridor' | 'uav-s_curve' | 'uav-pillars'

Curated dataset root (Phase-0 prep output, the ONLY tree the trainer reads):
    data/uav_fm/v1/<scene>/*.pkl       (+ manifest.json, which is skipped)
Override with the UAV_FM_DATA_ROOT env var if the curated set lives elsewhere.

Per-episode pickle schema (authoritative, see mini_fm_sanity.py / dataset_writer.py):
    ep['obs']     : (T, 9)   [ p_des(3) | p(3) | v(3) ]
    ep['actions'] : (T-1, 3) Δp_des  (= np.diff(targets); POSITION-DELTA convention)
"""

import os
import pickle

import numpy as np

SCENES = ['empty', 'corridor', 's_curve', 'pillars']

# repo_root/flow_matcher_v3_uav/datasets/d4rl.py  →  repo_root
_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_DEFAULT_DATA_ROOT = os.path.join(_REPO, 'data', 'uav_fm', 'v1')


def _data_root():
    return os.environ.get('UAV_FM_DATA_ROOT', _DEFAULT_DATA_ROOT)


def _scenes_for(env):
    """Parse the scene selector out of an 'uav-<scene>' env string."""
    if not env.startswith('uav'):
        raise NotImplementedError(f"UAV loader only handles 'uav-*' envs, got {env!r}")
    parts = env.split('-', 1)
    scene = parts[1] if len(parts) == 2 else 'all'
    if scene == 'all':
        return list(SCENES)
    if scene in SCENES:
        return [scene]
    raise ValueError(f"Unknown UAV scene {scene!r}; expected 'all' or one of {SCENES}")


def _iter_episode_files(scenes):
    root = _data_root()
    for scene in scenes:
        scene_dir = os.path.join(root, scene)
        if not os.path.isdir(scene_dir):
            raise FileNotFoundError(
                f"Curated UAV scene dir not found: {scene_dir}. "
                f"Run Phase-0 curation (curate_dataset.py) or set UAV_FM_DATA_ROOT.")
        for fn in sorted(os.listdir(scene_dir)):
            if not fn.endswith('.pkl'):
                continue
            if fn.startswith('run_summary') or fn.startswith('manifest'):
                continue
            yield os.path.join(scene_dir, fn)


def sequence_dataset(env, preprocess_fn):
    """Yield per-episode dicts {observations, actions, rewards, terminals}.

    Mirrors the source avoiding-d3il branch exactly: obs/actions aligned to length
    T-1, dummy rewards, terminal flag on the last step. preprocess_fn kept in the
    signature for API parity (identity for UAV, preprocess_fns=[]).
    """
    scenes = _scenes_for(env)
    n = 0
    for path in _iter_episode_files(scenes):
        with open(path, 'rb') as f:
            ep = pickle.load(f)

        obs = np.asarray(ep['obs'], dtype=np.float32)          # (T, 9) [p_des|p|v]
        actions = np.asarray(ep['actions'], dtype=np.float32)  # (T-1, 3) Δp_des
        valid_len = len(actions)
        if valid_len < 2:
            continue

        # Align obs to actions (both T-1), exactly like the avoiding branch pairs
        # input_state[:-1] with the velocity deltas.
        episode_data = {
            'observations': obs[:valid_len],
            'actions': actions[:valid_len],
            'rewards': np.zeros(valid_len, dtype=np.float32),
            'terminals': np.concatenate((np.zeros(valid_len - 1), np.array([1]))).astype(np.float32),
        }
        n += 1
        yield episode_data

    if n == 0:
        raise RuntimeError(
            f"No UAV episodes loaded for env={env!r} under {_data_root()}. "
            f"Is the curated dataset present?")
