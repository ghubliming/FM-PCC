"""The avoiding violation arithmetic, copied from the eval loops so Mode T scores exactly as Mode L (Gen15 U18).

Source of truth: FM_v3_ode_selectable_test/eval_flow_matching_v3_ode_selectable.py (loop body, lines ~330-385) and
flow_matcher_v3_ode_selectable/utils/constraints_helpers.py (formulate_halfspace_constraints /
formulate_bounds_constraints). Re-implemented here for a 4-D obs = [x_des, y_des, x, y] and a 2-D action
[vx, vy] so that no model package (torch) has to be imported by a CPU replay job.

Conventions reproduced 1:1
  * the check runs on the obs BEFORE each executed step: obs_0 = the reset obs [start, start], then the obs
    returned by step k-1 for k >= 1; the obs returned by the LAST executed step is never checked (the loop breaks)
  * halfspace / obstacle checks use the ACTUAL position (obs[2], obs[3]) and the UNTIGHTENED geometry
  * bounds add their overshoot to total_violations for k > 0 only, with the action of step k-1
  * n_violations counts steps with a halfspace-or-obstacle violation (bounds never count as a violating step)
  * an episode that ends without success has collision_free_completed = 0, whatever its violations
  * n_steps = (executed steps) - 1  [the loop stores the 0-based index of the breaking iteration]
"""
import os

import numpy as np
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(_HERE)
PROJECTION_YAML = os.path.join(REPO, 'config', 'projection_eval.yaml')
EXP = 'avoiding-d3il'
OBS_IDX = {'x_des': 0, 'y_des': 1, 'x': 2, 'y': 3}          # config observation_indices['avoiding']
ACT_OBS_IDX = {'vx': 0, 'vy': 1, 'x_des': 2, 'y_des': 3, 'x': 4, 'y': 5}   # action (2) + obs (4)
ACTION_DIM = 2


def load_projection_cfg(path=PROJECTION_YAML):
    with open(path) as fh:
        return yaml.safe_load(fh)


def geometry_for(halfspace_variant, cfg=None):
    """Exactly the selection the eval loop makes for a geometry name."""
    cfg = cfg or load_projection_cfg()
    hs, ob = cfg['halfspace_constraints'][EXP], cfg['obstacle_constraints'][EXP]
    if halfspace_variant == 'top-left-hard':
        poly, obst = [hs[0]], [ob[3]]
    elif halfspace_variant == 'top-right-hard':
        poly, obst = [hs[1]], [ob[4]]
    elif halfspace_variant == 'both-hard':
        poly, obst = [hs[2], hs[3]], [ob[5]]
    else:
        raise ValueError(f'unknown halfspace variant {halfspace_variant!r}')
    return {
        'name': halfspace_variant,
        'polytopic': poly,
        'obstacles': obst,
        'bounds': cfg['bounds'][EXP],
        'enlarge': float(cfg['enlarge_constraints']['avoiding']),
        'constraint_types': list(cfg['constraint_types']),
        'ax_limits': cfg['ax_limits'][EXP],
    }


def halfspace_row(constraint, enlarge=0.0, trajectory_dim=4, x_idx=OBS_IDX['x'], y_idx=OBS_IDX['y']):
    """formulate_halfspace_constraints for the obs part only: returns (c, d) with violation iff obs @ c >= d."""
    p0, p1, side = np.asarray(constraint[0], float), np.asarray(constraint[1], float), constraint[2]
    m = (p1[1] - p0[1]) / (p1[0] - p0[0])
    n = np.array([-1.0, 1.0 / m]) / np.linalg.norm([-1.0, 1.0 / m])
    if (m > 0 and side == 'below') or (m < 0 and side == 'above'):
        n = -n
    pe = p0 + enlarge * n
    d = pe[1] - m * pe[0]
    c = np.zeros(trajectory_dim)
    if side == 'below':
        c[x_idx], c[y_idx] = -m, 1.0
    elif side == 'above':
        c[x_idx], c[y_idx] = m, -1.0
        d = -d
    return c, d


def bounds_vectors(bounds, trajectory_dim=6, idx=ACT_OBS_IDX):
    lb, ub = -np.inf * np.ones(trajectory_dim), np.inf * np.ones(trajectory_dim)
    for b in bounds:
        for j, dim in enumerate(b['dimensions']):
            if b['type'] == 'lower' and dim in idx:
                lb[idx[dim]] = b['values'][j]
            elif b['type'] == 'upper' and dim in idx:
                ub[idx[dim]] = b['values'][j]
    return lb, ub


class EpisodeScorer:
    """One geometry, many episodes."""

    def __init__(self, geo):
        self.geo = geo
        self.rows = [halfspace_row(c, 0.0) for c in geo['polytopic']] if 'halfspace' in geo['constraint_types'] else []
        self.obstacles = list(geo['obstacles']) if 'obstacles' in geo['constraint_types'] else []
        self.use_bounds = 'bounds' in geo['constraint_types']
        self.lb, self.ub = bounds_vectors(geo['bounds'])

    def score(self, obs_checked, actions, success):
        """obs_checked: (n, 4) obs before each executed step; actions: (>= n-1, 2) action of step k-1 for k>=1."""
        obs_checked = np.asarray(obs_checked, float)
        n_viol, total, cf = 0, 0.0, 1
        for k, obs in enumerate(obs_checked):
            violated = 0
            for c, d in self.rows:
                val = float(obs @ c)
                if val >= d:
                    violated, total, cf = 1, total + (val - d), 0
            for ob in self.obstacles:
                dist = float(np.linalg.norm(obs[[OBS_IDX['x'], OBS_IDX['y']]] - np.asarray(ob['center'], float)))
                if dist < ob['radius']:
                    violated, total, cf = 1, total + (ob['radius'] - dist), 0
            if k > 0 and self.use_bounds:
                a = np.asarray(actions[k - 1], float)[:ACTION_DIM] if k - 1 < len(actions) else np.zeros(ACTION_DIM)
                act_obs = np.concatenate((a, obs))
                total += float(np.sum(np.maximum(0.0, act_obs - self.ub)) + np.sum(np.maximum(0.0, self.lb - act_obs)))
            n_viol += violated
        if not success:
            cf = 0
        return {'n_violations': float(n_viol), 'total_violations': float(total),
                'collision_free_completed': float(cf),
                'n_success_and_constraints': float(1 if (success and cf) else 0)}
