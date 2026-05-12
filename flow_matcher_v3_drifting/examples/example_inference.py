"""
Example: FM-D Inference with Drift Guidance

Demonstrates sampling trajectories with drift-guided ODE integration.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flow_matcher_v3_drifting.sampling import DriftODESolver, sample_trajectory_with_drift
from flow_matcher_v3_drifting.models import DriftLoss


def create_dummy_diffusion_model():
    """Create a dummy FM model for demonstration."""
    class DummyFMModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(28 + 16, 28)
        
        def forward(self, x, cond, t, returns=None, use_dropout=False):
            # Simple linear velocity field
            return self.fc(torch.cat([x, cond], dim=-1)) * 0.1
    
    return DummyFMModel()


def example_fm_d_inference():
    """
    Example FM-D inference with drift-guided ODE integration.
    """
    
    print("=" * 60)
    print("FM-D Inference Example")
    print("=" * 60)
    
    # Initialize models
    fm_model = create_dummy_diffusion_model()
    drift_loss = DriftLoss(
        trajectory_dim=28,
        loss_type='kl_divergence',
        memory_bank_size=100,
    )
    
    # Populate memory bank with expert trajectories
    expert_trajectories = torch.randn(50, 28)
    drift_loss.update_memory_bank(expert_trajectories)
    
    print("\nInitialized:")
    print(f"  - FM model: DummyFMModel")
    print(f"  - Drift loss: KL divergence")
    print(f"  - Memory bank size: 50 expert trajectories\n")
    
    # Initialize ODE solver
    solver = DriftODESolver(
        solver_method='euler',
        solver_backend='legacy_euler',
    )
    
    # Sampling configurations
    configs = [
        {"drift_weight": 0.0, "label": "Pure FM (no drift)"},
        {"drift_weight": 0.1, "label": "FM with drift (λ=0.1)"},
        {"drift_weight": 0.2, "label": "FM with drift (λ=0.2)"},
    ]
    
    batch_size = 4
    num_steps = 10
    
    print("Sampling trajectories...\n")
    
    for config in configs:
        drift_weight = config['drift_weight']
        label = config['label']
        
        # Initial state (noise)
        x0 = torch.randn(batch_size, 28)
        
        # Goal condition
        goal = torch.randn(batch_size, 16)
        
        # Define velocity function
        def velocity_fn(t, x):
            t_tensor = torch.tensor(t, dtype=x.dtype)
            return fm_model(x, goal, t_tensor)
        
        # Define drift gradient function (only used if drift_weight > 0)
        def drift_grad_fn(x):
            return drift_loss.get_gradient(x)
        
        # Solve ODE
        if drift_weight > 0:
            trajectory = solver.solve(
                velocity_fn,
                x0,
                t_span=(0.0, 1.0),
                num_steps=num_steps,
                drift_loss_fn=drift_grad_fn,
                drift_weight=drift_weight,
            )
        else:
            trajectory = solver.solve(
                velocity_fn,
                x0,
                t_span=(0.0, 1.0),
                num_steps=num_steps,
                drift_weight=0.0,
            )
        
        # Compute trajectory statistics
        trajectory_norm = trajectory.norm(dim=-1).mean()
        velocity_norm = (trajectory - x0).norm(dim=-1).mean()
        
        print(f"{label}:")
        print(f"  - Output trajectory norm: {trajectory_norm:.4f}")
        print(f"  - Total displacement: {velocity_norm:.4f}")
        print(f"  - Shape: {trajectory.shape}\n")
    
    # Alternative: Use convenience function
    print("Using sample_trajectory_with_drift() convenience function:\n")
    
    trajectory = sample_trajectory_with_drift(
        model=fm_model,
        x0=torch.randn(1, 28),
        cond=torch.randn(1, 16),
        t_span=(0.0, 1.0),
        num_steps=10,
        drift_loss_fn=drift_loss,
        drift_weight=0.15,
        solver_method='euler',
        solver_backend='legacy_euler',
    )
    
    print(f"Trajectory shape: {trajectory.shape}")
    print(f"Trajectory statistics:")
    print(f"  - Mean: {trajectory.mean().item():.4f}")
    print(f"  - Std: {trajectory.std().item():.4f}")
    print(f"  - L2 norm: {trajectory.norm().item():.4f}")
    
    print("\n" + "=" * 60)
    print("Inference Complete!")
    print("=" * 60)


if __name__ == "__main__":
    example_fm_d_inference()
