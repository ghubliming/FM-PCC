import numpy as np
import matplotlib

def formulate_halfspace_constraints(constraint, enlarge_constraints, trajectory_dim, act_obs_indices):
    # E9 robustness fix: the original slope-based formulation divides by the slope
    # (`n = [-1, 1/m]`), so it blows up for a horizontal wall (m = 0, e.g. the UAV corridor
    # walls that run along x) and is undefined for a vertical wall (dx = 0, m = inf).
    # The SLOPED branch below is byte-identical to the original for every non-degenerate line
    # (dx != 0 and dy != 0) — i.e. all avoiding-task / arm inputs are unaffected. Two extra
    # branches handle the axis-aligned degenerate walls the UAV scenes introduce. The
    # horizontal branch equals the m->0 limit of the sloped formula (verified by hand); the
    # vertical branch defines 'above' = larger-x feasible, 'below' = smaller-x feasible.
    p0 = np.asarray(constraint[0], dtype=float)
    p1 = np.asarray(constraint[1], dtype=float)
    side = constraint[2]
    ix, iy = act_obs_indices['x'], act_obs_indices['y']
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    C_row = np.zeros(trajectory_dim)

    if dx != 0.0 and dy != 0.0:
        # ── ORIGINAL sloped path — unchanged output for all existing (arm) inputs ──
        m = dy / dx
        n = np.array([-1.0, 1.0 / m]); n = n / np.linalg.norm(n)
        if (m > 0 and side == 'below') or (m < 0 and side == 'above'):
            n = -n
        p0e = p0 + enlarge_constraints * n
        d = p0e[1] - m * p0e[0]
        if side == 'below':
            C_row[ix] = -m; C_row[iy] = 1.0
        elif side == 'above':
            C_row[ix] = m;  C_row[iy] = -1.0; d = -d
        return C_row, d

    if dy == 0.0 and dx != 0.0:
        # ── horizontal wall (m = 0): feasible +y ('above') / -y ('below'); tighten shrinks ──
        if side == 'above':
            C_row[iy] = -1.0; d = -(p0[1] + enlarge_constraints)
        else:                                    # 'below'
            C_row[iy] =  1.0; d =  (p0[1] - enlarge_constraints)
        return C_row, d

    if dx == 0.0 and dy != 0.0:
        # ── vertical wall (m = inf): feasible +x ('above') / -x ('below'); tighten shrinks ──
        if side == 'above':
            C_row[ix] = -1.0; d = -(p0[0] + enlarge_constraints)
        else:                                    # 'below'
            C_row[ix] =  1.0; d =  (p0[0] - enlarge_constraints)
        return C_row, d

    raise ValueError(f'degenerate halfspace constraint: p0 == p1 ({p0.tolist()})')

def formulate_bounds_constraints(constraint_types, bounds, trajectory_dim, act_obs_indices):
    lower_bound = -np.inf * np.ones(trajectory_dim)
    upper_bound = np.inf * np.ones(trajectory_dim)
    if 'bounds' in constraint_types:
        for bound in bounds:
            for dim_idx, dim in enumerate(bound['dimensions']):
                if bound['type'] == 'lower' and dim in act_obs_indices:
                    lower_bound[act_obs_indices[dim]] = bound['values'][dim_idx]
                elif bound['type'] == 'upper' and dim in act_obs_indices:
                    upper_bound[act_obs_indices[dim]] = bound['values'][dim_idx]
    return lower_bound, upper_bound

def formulate_dynamics_constraints(exp, act_obs_indices, action_dim):
    dynamic_constraints = []
    if 'pointmaze' in exp:
        dynamic_constraints = [
            ('deriv', np.array([act_obs_indices['x'], act_obs_indices['vx']])),
            ('deriv', np.array([act_obs_indices['y'], act_obs_indices['vy']])),
        ]
    if 'antmaze' in exp:
        dynamic_constraints = [
            ('deriv', np.array([act_obs_indices['x'], act_obs_indices['vx']])),
            ('deriv', np.array([act_obs_indices['y'], act_obs_indices['vy']])),
            ('deriv', np.array([act_obs_indices['z'], act_obs_indices['vz']])),
        ]
    if 'avoiding' in exp and action_dim > 0:
        dynamic_constraints = [
            ('deriv', np.array([act_obs_indices['x'], act_obs_indices['vx']])),
            ('deriv', np.array([act_obs_indices['y'], act_obs_indices['vy']])),
            ('deriv', np.array([act_obs_indices['x_des'], act_obs_indices['vx']])),
            ('deriv', np.array([act_obs_indices['y_des'], act_obs_indices['vy']])),
        ]
    return dynamic_constraints

# Plotting
def plot_environment_constraints(exp, ax, flip_xy=False):
    if exp == 'pointmaze-umaze-dense-v2':
        ax.add_patch(matplotlib.patches.Rectangle((-1.5, -0.5), 2, 1, color='k', alpha=0.2))
    if exp == 'pointmaze-medium-dense-v2':
        bottom_left_corners = [[-1, 2], [0, 2], [-1, 1], [-3, 0], [1, 0], [2, 0], [-1, -1], [-2, -2], [1, -2], [0, -3]]
        for corner in bottom_left_corners:
            ax.add_patch(matplotlib.patches.Rectangle((corner[0], corner[1]), 1, 1, color='k', alpha=0.2))
    elif exp == 'antmaze-umaze-v1':
        ax.add_patch(matplotlib.patches.Rectangle((-6, -2), 8, 4, color='k', alpha=0.2))
    elif exp == 'avoiding-d3il':
        centers = [[0.5, -0.1], [0.425, 0.08], [0.575, 0.08], [0.35, 0.26], [0.5, 0.26], [0.65, 0.26]]
        for center in centers:
            if flip_xy:
                ax.add_patch(matplotlib.patches.Circle(center[::-1], 0.025, color='r'))
            ax.add_patch(matplotlib.patches.Circle(center, 0.025, color='r'))
        ax.plot([0.2, 0.8], [0.35, 0.35], color=[0.4, 1, 0.4], linewidth=5)

def plot_halfspace_constraints(exp, polytopic_constraints, ax, ax_limits, flip_xy=False, enlarge_constraints=0):
    for constraint in polytopic_constraints:
        mat = np.vstack((constraint[:2], np.zeros(2)))
        mat_enlarged = np.vstack((constraint[:2], np.zeros(2)))
        if 'pointmaze' in exp:
            mat[2] = np.array([1.5, -1.5]) if constraint[2] == 'above' else np.array([1.5, 1.5])
        elif 'antmaze' in exp:
            mat[2] = np.array([6, -6]) if constraint[2] == 'above' else np.array([6, 6])
        elif 'avoiding' in exp:
            # Works for triangles with two vertices on the negative y-axis
            slope = (constraint[1][1] - constraint[0][1]) / (constraint[1][0] - constraint[0][0])
            if slope > 0 and constraint[2] == 'above':
                mat[2] = np.array([ax_limits[0][1], ax_limits[1][0]])
                mat_enlarged[2] = np.array([ax_limits[0][1], ax_limits[1][0]])
            elif slope > 0 and constraint[2] == 'below':
                mat[2] = np.array([ax_limits[0][0], ax_limits[1][1]])
                mat_enlarged[2] = np.array([ax_limits[0][0], ax_limits[1][1]])
                mat_enlarged[0, 1] -= enlarge_constraints / np.cos(np.arctan(slope))
                mat_enlarged[1, 0] += enlarge_constraints * np.sin(np.arctan(slope))
            elif slope < 0 and constraint[2] == 'above':
                mat[2] = np.array([ax_limits[0][0], ax_limits[1][0]])
                mat_enlarged[2] = np.array([ax_limits[0][0], ax_limits[1][0]])
            elif slope < 0 and constraint[2] == 'below':
                mat[2] = np.array([ax_limits[0][1], ax_limits[1][1]])
                mat_enlarged[2] = np.array([ax_limits[0][1], ax_limits[1][1]])
                mat_enlarged[0, 1] -= enlarge_constraints / np.cos(np.arctan(slope))
                mat_enlarged[1, 0] += enlarge_constraints * np.sin(np.arctan(slope))
        if flip_xy:
            mat = mat[:, ::-1]
        ax.add_patch(matplotlib.patches.Polygon(mat, color='b', alpha=0.2))
        ax.add_patch(matplotlib.patches.Polygon(mat_enlarged, color='b', alpha=0.1, linestyle='--'))
