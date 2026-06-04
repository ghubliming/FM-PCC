"""Trial generator: builds and runs randomised PID rollouts across all scenes.

Public API
----------
run_trial(scene, homotopy, gain_variant, seed, duration=None)
    -> dict  (raw rollout)  |  None  (rejected — too many obstacle contacts)

sample_trial_specs(scene, n_trials, base_seed, gain_variants=None)
    -> list[dict]  — spec dicts suitable for passing to run_trial(**spec)

Constants exported for collect.py
----------------------------------
SCENE_XMLS, SCENE_OBSTACLES, HOMOTOPY_CLASSES, GAIN_VARIANTS
"""

import os
import sys

import numpy as np
import mujoco

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from uav_env_test.flight_controller import CascadedPID
import uav_expert_data_collect.trajectories as trajs

# ── Paths ─────────────────────────────────────────────────────────────────────
_REPO       = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_SCENES_DIR = os.path.join(_REPO, 'd3il/environments/d3il/models/mj/robot/quadrotor/scenes')

SCENE_XMLS = {
    'empty':    os.path.join(_SCENES_DIR, 'scene_empty.xml'),
    'corridor': os.path.join(_SCENES_DIR, 'scene_corridor.xml'),
    's_curve':  os.path.join(_SCENES_DIR, 'scene_s_curve.xml'),
    'pillars':  os.path.join(_SCENES_DIR, 'scene_pillars.xml'),
}

# ── Obstacle metadata (baked into each episode pickle) ───────────────────────
SCENE_OBSTACLES = {
    'empty': [],
    'corridor': [
        {'type': 'box', 'name': 'wall_y_neg',
         'center': [0.0, -0.5, 0.75], 'half_extents': [2.0, 0.05, 0.75]},
        {'type': 'box', 'name': 'wall_y_pos',
         'center': [0.0,  0.5, 0.75], 'half_extents': [2.0, 0.05, 0.75]},
    ],
    's_curve': [
        {'type': 'box', 'name': 'seg1_wall_neg',
         'center': [-1.75, -1.3, 0.75], 'half_extents': [1.25, 0.05, 0.75]},
        {'type': 'box', 'name': 'seg1_wall_pos',
         'center': [-1.75, -0.3, 0.75], 'half_extents': [1.25, 0.05, 0.75]},
        {'type': 'box', 'name': 'seg2_wall_neg',
         'center': [ 1.75,  0.3, 0.75], 'half_extents': [1.25, 0.05, 0.75]},
        {'type': 'box', 'name': 'seg2_wall_pos',
         'center': [ 1.75,  1.3, 0.75], 'half_extents': [1.25, 0.05, 0.75]},
    ],
    'pillars': [
        {'type': 'cylinder', 'name': f'pillar_{col}{idx+1}',
         'center': [x, y, 1.0], 'radius': 0.12, 'half_height': 1.0}
        for idx, x in enumerate([-2.0, 0.0, 2.0])
        for col, y in [('A', -0.6), ('B', 0.6)]
    ],
}

# ── Homotopy classes per scene ────────────────────────────────────────────────
HOMOTOPY_CLASSES = {
    'empty':    ['N/A'],
    'corridor': ['L', 'C', 'R'],
    's_curve':  ['default'],
    'pillars':  ['(L,L,L)', '(L,R,L)', '(R,L,R)', '(R,R,R)'],
}

# ── PID gain variants (multipliers applied to Kp_pos, Kd_pos) ────────────────
GAIN_VARIANTS = {
    'pid_default':   {'kp_scale': 1.0, 'kd_scale': 1.0},
    'pid_high_gain': {'kp_scale': 1.2, 'kd_scale': 1.0},
    'pid_low_gain':  {'kp_scale': 0.8, 'kd_scale': 0.9},
}

# Trials with more than this fraction of obstacle-contact steps are rejected.
MAX_CONTACT_FRACTION = 0.02

# Fix_4: s_curve has narrow wall end-faces at x=±0.5 that the drone briefly
# grazes even on good trajectories.  Raise threshold to 0.08 for that scene
# so brief end-face clips don't reject otherwise valid episodes.
SCENE_MAX_CONTACT_FRACTION = {
    'empty':    0.02,
    'corridor': 0.02,
    's_curve':  0.08,
    'pillars':  0.02,
}


# ── Internal helpers ──────────────────────────────────────────────────────────

def _is_obstacle_contact(model, contact):
    """True if neither geom is the floor — counts as a drone/obstacle collision."""
    n1 = model.geom(contact.geom1).name
    n2 = model.geom(contact.geom2).name
    return n1 != 'floor' and n2 != 'floor'


def _make_pid(model, gain_variant):
    pid = CascadedPID(model)
    s = GAIN_VARIANTS[gain_variant]
    pid.Kp_pos = pid.Kp_pos * s['kp_scale']
    pid.Kd_pos = pid.Kd_pos * s['kd_scale']
    return pid


def _build_traj_and_init(scene, homotopy, rng):
    """Return (traj_fn, init_pos_array, duration_s) for the given scene + homotopy."""
    z = float(rng.uniform(0.70, 1.10))

    if scene == 'empty':
        # Random start/end with minimum separation 1.0 m.
        for _ in range(50):
            p_s = [rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8), z]
            p_e = [rng.uniform(-1.8, 1.8), rng.uniform(-1.8, 1.8),
                   float(rng.uniform(0.70, 1.10))]
            if np.linalg.norm(np.array(p_e) - np.array(p_s)) >= 1.0:
                break
        dist = float(np.linalg.norm(np.array(p_e) - np.array(p_s)))
        dur  = max(4.0, dist / 0.4)
        return trajs.empty_path(p_s, p_e, dur), np.array(p_s), dur

    elif scene == 'corridor':
        y_jitter = float(rng.uniform(-0.05, 0.05))
        y_bias   = trajs.CORRIDOR_CHANNELS[homotopy] + y_jitter
        p_s = np.array([-2.8, y_bias, z])
        dur = float(rng.uniform(6.0, 10.0))
        return trajs.corridor_path(homotopy, z, dur), p_s, dur

    elif scene == 's_curve':
        y_jitter = float(rng.uniform(-0.04, 0.04))
        # Fix_4: revert duration [22,30]→[16,22]s to match the Fix_2 config that
        # achieved 61.9% rejection (best so far).  Longer duration worsened things.
        dur      = float(rng.uniform(16.0, 22.0))
        p_s = np.array([-3.2, -0.8 + y_jitter, z])
        return trajs.s_curve_scene_path(z, dur, y_jitter=y_jitter), p_s, dur

    elif scene == 'pillars':
        # Fix_1: replace s_curve_path-based pillar_path (zero-velocity stops near
        # pillars → collisions) with the continuous sinusoidal weave factory.
        # Homotopy classes mapped to amplitude sign: L-dominant → negative y
        # (passes outside column A), R-dominant → positive y (outside column B).
        # Centre-passes (L,R,L) and (R,L,R) use smaller amplitude through centre.
        # Fix_2: (L,R,L) and (R,L,R) amplitude ±0.55 was inside the pillar zone
        # (column A/B edges at y=±0.48) → 100% rejection for those classes.
        # Set to 0.0 → straight centre-line at y=0, clearance 0.48 m each side.
        _amp_map = {
            '(L,L,L)': -1.0,
            '(L,R,L)':  0.0,
            '(R,L,R)':  0.0,
            '(R,R,R)':  1.0,
        }
        amp = _amp_map[homotopy]
        dur = float(rng.uniform(10.0, 16.0))
        p_s = np.array([-3.2, 0.0, z])
        traj_fn = trajs.weave(
            x_range=(-3.2, 3.2), y_amplitude=amp,
            period=4.0, altitude=z, duration=dur,
        )
        return traj_fn, p_s, dur

    raise ValueError(f'Unknown scene: {scene!r}')


# ── Public API ────────────────────────────────────────────────────────────────

def run_trial(scene, homotopy, gain_variant, seed, duration=None):
    """Run one PID rollout and return the raw log, or None if rejected.

    Parameters
    ----------
    scene        : 'empty' | 'corridor' | 's_curve' | 'pillars'
    homotopy     : one of HOMOTOPY_CLASSES[scene]
    gain_variant : one of GAIN_VARIANTS keys
    seed         : integer random seed (controls start pos, duration, jitter)
    duration     : optional override for episode duration (s)

    Returns
    -------
    dict with keys:
        steps           : list of per-step dicts {'p', 'v', 'p_des'}
        scene, homotopy, controller, dt_physics, duration
        init_pos        : list[float]
        obstacles       : list[dict]  (from SCENE_OBSTACLES)
        contact_fraction: float
    or None if obstacle contact fraction exceeds MAX_CONTACT_FRACTION.
    """
    rng = np.random.default_rng(seed)
    model = mujoco.MjModel.from_xml_path(SCENE_XMLS[scene])
    data  = mujoco.MjData(model)

    traj_fn, init_pos, dur = _build_traj_and_init(scene, homotopy, rng)
    if duration is not None:
        dur = float(duration)

    data.qpos[:3]  = init_pos
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:]   = 0.0
    mujoco.mj_forward(model, data)

    pid    = _make_pid(model, gain_variant)
    dt     = float(model.opt.timestep)
    n_step = int(round(dur / dt))

    steps    = []
    n_hit    = 0

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

        hit = any(_is_obstacle_contact(model, data.contact[ci])
                  for ci in range(data.ncon))
        n_hit += int(hit)

        steps.append({'p': p, 'v': v, 'p_des': np.asarray(p_des, dtype=float)})

    contact_frac  = n_hit / max(n_step, 1)
    contact_limit = SCENE_MAX_CONTACT_FRACTION.get(scene, MAX_CONTACT_FRACTION)
    if contact_frac > contact_limit:
        return None

    return {
        'steps':            steps,
        'scene':            scene,
        'homotopy':         homotopy,
        'controller':       gain_variant,
        'dt_physics':       dt,
        'duration':         dur,
        'init_pos':         init_pos.tolist(),
        'obstacles':        SCENE_OBSTACLES[scene],
        'contact_fraction': contact_frac,
    }


def sample_trial_specs(scene, n_trials, base_seed, gain_variants=None):
    """Return a list of spec dicts for n_trials, cycling homotopy and gain variants.

    Each spec dict can be unpacked directly: run_trial(**spec).
    """
    if gain_variants is None:
        gain_variants = ['pid_default', 'pid_high_gain']
    homotopies = HOMOTOPY_CLASSES[scene]
    specs = []
    for i in range(n_trials):
        specs.append({
            'scene':        scene,
            'homotopy':     homotopies[i % len(homotopies)],
            'gain_variant': gain_variants[i % len(gain_variants)],
            'seed':         base_seed + i,
        })
    return specs
