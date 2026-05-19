"""
B1 Unit Test — DynamicConstraints initial-state scale row.

Fix 8 B1 changed the initial-state row in DynamicConstraints.build_matrices() from:
    mat_fix_initial[0, x_idx] = 1          # pre-B1: coefficient 1
to:
    mat_fix_initial[0, x_idx] = x_diff     # post-B1: same scale as dynamics rows

And in project() / compute_gradient():
    b[counter * self.horizon] = x_diff * s_0[x_idx]   # post-B1 (was s_0[x_idx])

Without B1 the initial-state constraint was proportionally weaker than the dynamics rows
by a factor of x_diff (~0.4 for typical Franka x-axis). SLSQP would relax the initial-state
anchor more than intended.

This file contains two tests:
  1. Structural test: A[0, x_idx] == x_diff  (coefficient in the constraint matrix)
  2. Functional test: after projection, the equality constraints are satisfied to < 1e-3
"""

import sys
import os
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from diffuser_visual_aligning.sampling.projection import Projector, ProjectionNormalizer


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

class _MockNorm:
    def __init__(self, mins, maxs):
        self.mins = np.array(mins, dtype=np.float32)
        self.maxs = np.array(maxs, dtype=np.float32)


def _make_normalizer():
    """
    2D transition: [action (dim 0), position (dim 1)]
      action   : mins=[-0.01], maxs=[0.01]  → dx_diff = 0.02
      position : mins=[0.30],  maxs=[0.70]  → x_diff  = 0.40  ← B1 scale factor
    """
    act_norm = _MockNorm(mins=[-0.01], maxs=[0.01])
    obs_norm = _MockNorm(mins=[0.30],  maxs=[0.70])
    return ProjectionNormalizer(
        observation_normalizer=obs_norm,
        action_normalizer=act_norm,
        goal_dim=0,
    )


def _make_projector(horizon=4):
    """Projector with one Euler dynamics constraint: pos[t+1] = pos[t] + act[t]*dt."""
    return Projector(
        horizon=horizon,
        transition_dim=2,           # [act, pos]
        action_dim=1,
        constraint_list=[('deriv', [1, 0])],   # pos (dim 1) driven by act (dim 0)
        normalizer=_make_normalizer(),
        variant='states_actions',
        dt=0.1,
        skip_initial_state=True,
        gradient=False,
        device='cpu',
    )


# ---------------------------------------------------------------------------
# Test 1 — Structural: A[0, x_idx] == x_diff
# ---------------------------------------------------------------------------

def test_b1_initial_row_coefficient():
    """
    The first row of A (the initial-state row) must have its non-zero coefficient
    equal to x_diff = 0.40, not 1.  A coefficient of 1 would indicate B1 was not applied.
    """
    projector = _make_projector()
    x_diff_expected = 0.70 - 0.30   # = 0.40

    A = projector.A_np   # shape (H, transition_dim * H)  — H rows for 1 deriv constraint

    # Row 0 is the initial-state row (added first by skip_initial_state logic).
    initial_row = A[0]
    nonzero_idx = np.nonzero(initial_row)[0]

    assert len(nonzero_idx) == 1, (
        f'Expected exactly 1 non-zero in initial-state row, got {len(nonzero_idx)}: {nonzero_idx}'
    )
    coeff = initial_row[nonzero_idx[0]]
    assert abs(abs(coeff) - x_diff_expected) < 1e-6, (
        f'B1 coefficient mismatch: A[0, x_idx]={coeff:.6f}, expected ±{x_diff_expected:.6f}. '
        f'B1 fix may not be applied — coefficient should be x_diff, not 1.'
    )

    # Verify dynamics rows also use x_diff (consistency check)
    x_diff_dynamics = []
    for row_idx in range(1, 4):   # rows 1..H-1 are dynamics rows
        row = A[row_idx]
        pos_coeffs = [row[t * 2 + 1] for t in range(4) if row[t * 2 + 1] != 0]
        for c in pos_coeffs:
            x_diff_dynamics.append(abs(c))
    assert all(abs(v - x_diff_expected) < 1e-6 for v in x_diff_dynamics), (
        f'Dynamics row position coefficients inconsistent with x_diff: {x_diff_dynamics}'
    )

    print(f'  [PASS] A[0, x_idx] = {coeff:.4f} == x_diff = {x_diff_expected:.4f}')
    print(f'  [PASS] Dynamics rows also use x_diff consistently.')


# ---------------------------------------------------------------------------
# Test 2 — Functional: equality constraints satisfied after projection
# ---------------------------------------------------------------------------

def test_b1_projection_satisfies_constraints():
    """
    Build a trajectory whose initial position intentionally differs from the anchor.
    After projection, all equality constraints (including the initial-state row) must
    be satisfied: || A @ sol - b || < 1e-3 for every batch element.

    If B1 were NOT applied (coefficient 1 vs x_diff = 0.4), SLSQP would weight the
    initial-state row ~2.5× less than the dynamics rows, potentially leaving a larger
    residual on the initial-state constraint.
    """
    H = 4
    projector = _make_projector(horizon=H)

    rng = np.random.default_rng(0)
    batch_size = 3
    traj_np = rng.uniform(-0.8, 0.8, size=(batch_size, H, 2)).astype(np.float32)

    traj_t = torch.tensor(traj_np, device='cpu')
    sol, costs = projector.project(traj_t)
    sol_np = sol.cpu().numpy()

    A = projector.A_np.astype('double')

    for i in range(batch_size):
        # Reconstruct b for this sample (b[0] = x_diff * s_0[x_idx])
        s_0 = traj_np[i, 0]   # initial state of sample i (un-projected)
        x_diff = projector.dynamic_constraints._initial_state_x_diffs[0]
        x_idx = 1   # position dimension

        b = projector.b_np.astype('double').copy()
        b[0] = x_diff * s_0[x_idx]

        sol_flat = sol_np[i].flatten().astype('double')
        residual = np.abs(A @ sol_flat - b)
        max_res = residual.max()

        assert max_res < 1e-3, (
            f'Sample {i}: equality constraints violated after projection. '
            f'max |A@sol - b| = {max_res:.6f}. '
            f'Initial-state residual = {residual[0]:.6f}.'
        )
        print(f'  [PASS] Sample {i}: max constraint residual = {max_res:.2e}  '
              f'(initial-state row residual = {residual[0]:.2e})')


# ---------------------------------------------------------------------------
# Test 3 — Functional: initial state is preserved in projected solution
# ---------------------------------------------------------------------------

def test_b1_initial_state_preserved():
    """
    The projected solution must keep x[0, pos] equal to s_0[pos] (within SLSQP tolerance).
    This verifies the initial-state constraint is actively enforced, not just present.
    """
    projector = _make_projector(horizon=6)

    rng = np.random.default_rng(7)
    traj_np = rng.uniform(-1.0, 1.0, size=(1, 6, 2)).astype(np.float32)
    traj_t = torch.tensor(traj_np, device='cpu')

    sol, _ = projector.project(traj_t)
    sol_np = sol.cpu().numpy()

    s_0_pos = traj_np[0, 0, 1]      # initial position (normalized) from input
    projected_pos = sol_np[0, 0, 1] # position at t=0 in projected solution

    err = abs(projected_pos - s_0_pos)
    assert err < 5e-3, (
        f'Initial state not preserved after projection. '
        f's_0[pos]={s_0_pos:.4f}, projected={projected_pos:.4f}, err={err:.6f}.'
    )
    print(f'  [PASS] s_0[pos]={s_0_pos:.4f}, projected[0,pos]={projected_pos:.4f}, '
          f'err={err:.2e}')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    print('=' * 60)
    print('B1 Unit Test — DynamicConstraints initial-state scale row')
    print('=' * 60)
    print()

    print('Test 1 — Structural: A[0, x_idx] == x_diff')
    test_b1_initial_row_coefficient()
    print()

    print('Test 2 — Functional: equality constraints satisfied after projection')
    test_b1_projection_satisfies_constraints()
    print()

    print('Test 3 — Functional: initial state is preserved in projected solution')
    test_b1_initial_state_preserved()
    print()

    print('All B1 tests passed.')
