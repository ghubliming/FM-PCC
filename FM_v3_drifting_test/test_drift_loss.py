"""
Drift Loss Integration Test

Tests drift loss computation, backward pass, and gradient flow.
"""

import torch
import sys
import os

# Add parent to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flow_matcher_v3_drifting.models import DriftLoss


def test_drift_loss_initialization():
    """Test DriftLoss module initialization."""
    drift_loss = DriftLoss(
        trajectory_dim=28,
        loss_type="kl_divergence",
        memory_bank_size=100,
    )
    assert drift_loss.trajectory_dim == 28
    assert drift_loss.loss_type == "kl_divergence"
    print("✓ DriftLoss initialization test passed")


def test_drift_loss_kl_divergence():
    """Test KL divergence loss computation."""
    drift_loss = DriftLoss(trajectory_dim=28, loss_type="kl_divergence")
    
    # Create synthetic trajectories
    expert_trajs = torch.randn(10, 28)
    sampled_traj = torch.randn(5, 28, requires_grad=True)
    
    # Update memory bank
    drift_loss.update_memory_bank(expert_trajs)
    
    # Compute loss
    loss_dict = drift_loss.forward(sampled_traj)
    assert 'loss' in loss_dict
    assert loss_dict['loss'].shape == torch.Size([])
    assert loss_dict['loss'].requires_grad
    print(f"✓ KL divergence loss: {loss_dict['loss'].item():.4f}")
    
    # Test backward pass
    loss = loss_dict['loss']
    loss.backward()
    assert sampled_traj.grad is not None
    print("✓ KL divergence backward pass test passed")


def test_drift_loss_mmd():
    """Test MMD loss computation."""
    drift_loss = DriftLoss(trajectory_dim=28, loss_type="mmd")
    
    expert_trajs = torch.randn(50, 28)
    sampled_traj = torch.randn(8, 28, requires_grad=True)
    
    drift_loss.update_memory_bank(expert_trajs)
    loss_dict = drift_loss.forward(sampled_traj)
    
    assert 'loss' in loss_dict
    assert loss_dict['loss'] >= 0  # MMD is non-negative
    print(f"✓ MMD loss: {loss_dict['loss'].item():.4f}")


def test_drift_loss_adversarial():
    """Test adversarial loss computation."""
    drift_loss = DriftLoss(trajectory_dim=28, loss_type="adversarial")
    
    expert_trajs = torch.randn(50, 28)
    sampled_traj = torch.randn(8, 28, requires_grad=True)
    
    drift_loss.update_memory_bank(expert_trajs)
    loss_dict = drift_loss.forward(sampled_traj)
    
    assert 'gen_loss' in loss_dict
    assert 'dis_loss' in loss_dict
    print(f"✓ Adversarial gen_loss: {loss_dict['gen_loss'].item():.4f}")
    print(f"✓ Adversarial dis_loss: {loss_dict['dis_loss'].item():.4f}")


def test_drift_loss_gradient():
    """Test gradient computation for ODE guidance."""
    drift_loss = DriftLoss(trajectory_dim=28, loss_type="kl_divergence")
    
    expert_trajs = torch.randn(20, 28)
    drift_loss.update_memory_bank(expert_trajs)
    
    traj = torch.randn(1, 28)
    grad = drift_loss.get_gradient(traj)
    
    assert grad.shape == traj.shape
    assert not grad.requires_grad  # Gradient should be detached
    print(f"✓ Gradient shape: {grad.shape}, norm: {grad.norm().item():.4f}")


def test_memory_bank_circular():
    """Test circular buffer behavior of memory bank."""
    drift_loss = DriftLoss(trajectory_dim=28, memory_bank_size=10)
    
    # Add batches sequentially
    batch1 = torch.randn(5, 28)
    batch2 = torch.randn(5, 28)
    batch3 = torch.randn(5, 28)
    
    drift_loss.update_memory_bank(batch1)
    assert not drift_loss.memory_bank_full
    
    drift_loss.update_memory_bank(batch2)
    assert drift_loss.memory_bank_full
    
    # Add beyond capacity (should wrap)
    drift_loss.update_memory_bank(batch3)
    assert drift_loss.memory_bank_full
    print("✓ Memory bank circular buffer test passed")


if __name__ == "__main__":
    print("Running Drift Loss Integration Tests...\n")
    test_drift_loss_initialization()
    test_drift_loss_kl_divergence()
    test_drift_loss_mmd()
    test_drift_loss_adversarial()
    test_drift_loss_gradient()
    test_memory_bank_circular()
    print("\n✅ All drift loss tests passed!")
