"""Convert raw PID rollouts to FM-PCC-format episode pickles.

Schema (Decision 3 of EPOCH4_EXECUTION_PLAN.md)
------------------------------------------------
episode_id  : str
scene       : str    'empty'|'corridor'|'s_curve'|'pillars'
homotopy    : str    '(L,R,L)'|'N/A'|…
controller  : str    'pid_default'|'pid_high_gain'|…
dt          : float  dataset timestep (s)  ≈ 0.030 (100 Hz physics → 33 Hz dataset)
obs         : (T, 9)   float32  [p_des(3), p(3), v(3)]   — U2: p_des prepended (DPCC_OBS_DEVIATION §Deviation 2)
actions     : (T-1, 3) float32  [Δp_des(3)]   — position-DELTA convention (AUDIT Risk 3)
targets     : (T, 3)   float32  absolute p_des (kept for debugging; not fed to FM)
q           : (T, 4)   float32  actual body quaternion [w,x,y,z] — D-prep: for attitude-aware WS-A/B rendering (E4 U3+)
obstacles   : list[dict]
metadata    : dict   {start_pos, total_time, dt_physics, contact_fraction,
                      controller_gains, noise_sigma}

Action convention
-----------------
actions[t] = targets[t+1] - targets[t]   (forward difference of commanded positions)
Matches D3IL: vel_state = des_c_pos[1:] - des_c_pos[:-1]
Matches UAV-Flow: _transform_to_local_frame → body-frame Δpose
"""

import os
import pickle

import numpy as np

# Physics dt = 0.01 s (100 Hz).  Dataset target: 33 Hz → keep every 3rd step.
DATASET_HZ     = 33
NOISE_SIGMA    = 0.02    # Gaussian noise on targets to thicken data manifold (m)


def _downsample(steps, dt_physics, dataset_hz=DATASET_HZ):
    stride = max(1, round(1.0 / (dt_physics * dataset_hz)))
    return steps[::stride]


def rollout_to_episode(rollout, episode_id, noise_sigma=NOISE_SIGMA, rng=None):
    """Convert a run_trial() dict to a schema-locked episode dict.

    Parameters
    ----------
    rollout    : dict returned by generator.run_trial()
    episode_id : unique string identifier
    noise_sigma: std of Gaussian noise added to targets (0 = no noise)
    rng        : np.random.Generator or None (creates one from episode_id hash)
    """
    steps = _downsample(rollout['steps'], rollout['dt_physics'])
    T = len(steps)

    obs     = np.array([np.concatenate([s['p_des'], s['p'], s['v']]) for s in steps],
                       dtype=np.float32)        # (T, 9)  [p_des(3) | p(3) | v(3)]
    q       = np.array([s['q'] for s in steps],
                       dtype=np.float32)        # (T, 4)  [w,x,y,z]  D-prep: attitude for rendering
    targets = np.array([s['p_des'] for s in steps],
                       dtype=np.float32)        # (T, 3)  absolute

    if noise_sigma > 0.0:
        if rng is None:
            rng = np.random.default_rng(abs(hash(episode_id)) % (2**32))
        # Fix_1: one constant offset per episode, not per-step independent noise.
        # Per-step noise makes actions = np.diff(noisy_targets) noise-dominated
        # (std ≈ √2·σ = 0.028 m/step >> signal ≈ 0.012 m/step at 0.4 m/s 33 Hz).
        # A constant offset shifts the whole trajectory rigidly: Δ of a constant
        # is zero, so action deltas are unaffected.
        offset = rng.normal(0.0, noise_sigma, (1, 3)).astype(np.float32)
        targets = targets + offset

    actions = np.diff(targets, axis=0).astype(np.float32)  # (T-1, 3)  Δp_des

    dt_dataset = rollout['dt_physics'] * max(1, round(1.0 / (rollout['dt_physics'] * DATASET_HZ)))

    return {
        'episode_id': episode_id,
        'scene':      rollout['scene'],
        'homotopy':   rollout['homotopy'],
        'controller': rollout['controller'],
        'dt':         float(dt_dataset),
        'obs':        obs,
        'actions':    actions,
        'targets':    targets,
        'q':          q,
        'obstacles':  rollout['obstacles'],
        'metadata': {
            'start_pos':        rollout['init_pos'],
            'total_time':       rollout['duration'],
            'dt_physics':       rollout['dt_physics'],
            'contact_fraction': rollout['contact_fraction'],
            'controller_gains': rollout['controller'],
            'noise_sigma':      float(noise_sigma),
        },
    }


def save_episode(episode, out_dir):
    """Pickle episode to out_dir/<episode_id>.pkl.  Returns the saved path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f'{episode["episode_id"]}.pkl')
    with open(path, 'wb') as f:
        pickle.dump(episode, f, protocol=4)
    return path


def load_episode(path):
    with open(path, 'rb') as f:
        return pickle.load(f)
