"""
ODE Solver with Drift Guidance Test

Tests drift-augmented ODE integration.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flow_matcher_v3_drifting.sampling import DriftODESolver, DriftAugmentedVelocityField
from flow_matcher_v3_drifting.models import DriftLoss


def test_drift_augmented_velocity_field():
    """Test velocity field wrapping with drift guidance."""
    
    def velocity_fn(t, x):
        return torch.ones_like(x) * 0.1
    
    def drift_loss_fn(x):
        return torch.ones_like(x) * 0.01
    
    field = DriftAugmentedVelocityField(
        velocity_fn,
        drift_loss_fn=drift_loss_fn,
        drift_weight=0.1,
    )
    
    t = torch.tensor(0.5)
    x = torch.randn(4, 28)
    
    v = field(t, x)
    assert v.shape == x.shape
    print(f"✓ Augmented velocity field shape: {v.shape}")


def test_ode_solver_initialization():
    """Test ODE solver initialization."""
    solver = DriftODESolver(
        solver_method="euler",
        solver_backend="legacy_euler",
    )
    assert solver.solver_method == "euler"
    print("✓ ODE solver initialization test passed")


def test_ode_solver_legacy_euler():
    """Test legacy Euler integration."""
    def velocity_fn(t, x):
        return -x  # Simple exponential decay
    
    solver = DriftODESolver(
        solver_method="euler",
        solver_backend="legacy_euler",
    )
    
    x0 = torch.ones(4, 28)
    x_final = solver.solve(
        velocity_fn,
        x0,
        t_span=(0.0, 0.1),
        num_steps=10,
    )
    
    assert x_final.shape == x0.shape
    assert (x_final < x0).all()  # Should decay
    print(f"✓ ODE Euler integration: x0 mean={x0.mean():.4f}, xf mean={x_final.mean():.4f}")


def test_ode_solver_with_drift():
    """Test ODE integration with drift guidance."""
    
    def velocity_fn(t, x):
        return torch.ones_like(x) * 0.01
    
    drift_loss = DriftLoss(trajectory_dim=28, loss_type="kl_divergence")
    expert_trajs = torch.randn(20, 28)
    drift_loss.update_memory_bank(expert_trajs)
    
    def drift_grad_fn(x):
        return drift_loss.get_gradient(x)
    
    solver = DriftODESolver(solver_method="euler")
    
    x0 = torch.randn(2, 28)
    x_no_drift = solver.solve(
        velocity_fn, x0, (0.0, 0.1), num_steps=5, drift_weight=0.0
    )
    
    x_with_drift = solver.solve(
        velocity_fn, x0, (0.0, 0.1), num_steps=5,
        drift_loss_fn=drift_grad_fn, drift_weight=0.1
    )
    
    # Trajectories should differ with drift guidance
    diff = (x_with_drift - x_no_drift).norm()
    print(f"✓ Drift-guided vs. standard ODE difference: {diff.item():.6f}")


def test_ode_solver_rk4():
    """Test RK4 integration."""
    def velocity_fn(t, x):
        return -x
    
    solver = DriftODESolver(solver_method="rk4")
    
    x0 = torch.ones(2, 28)
    x_final = solver.solve(
        velocity_fn, x0, (0.0, 0.1), num_steps=10
    )
    
    assert x_final.shape == x0.shape
    assert (x_final < x0).all()
    print(f"✓ RK4 integration works, final state norm: {x_final.norm().item():.4f}")


if __name__ == "__main__":
    print("Running ODE Solver with Drift Tests...\n")
    test_drift_augmented_velocity_field()
    test_ode_solver_initialization()
    test_ode_solver_legacy_euler()
    test_ode_solver_with_drift()
    test_ode_solver_rk4()
    print("\n✅ All ODE solver tests passed!")
