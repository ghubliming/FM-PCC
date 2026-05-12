"""
FM-D Training Utilities Test

Tests drift loss scheduling, memory bank, and training wrappers.
"""

import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flow_matcher_v3_drifting.utils.drift_training import (
    DriftLossScheduler,
    DriftMemoryBank,
    DriftTrainingWrapper,
    compute_combined_loss,
)
from flow_matcher_v3_drifting.models import DriftLoss


def test_drift_loss_scheduler_warmup():
    """Test warmup schedule."""
    scheduler = DriftLossScheduler(
        mode="warmup",
        start_weight=0.0,
        target_weight=0.1,
        warmup_steps=100,
    )
    
    weights = []
    for _ in range(150):
        weights.append(scheduler.get_weight())
        scheduler.step()
    
    # Check warmup phase
    assert weights[0] < weights[50] < weights[100]
    # Check plateau
    assert weights[100] == weights[149]
    assert abs(weights[149] - 0.1) < 1e-6
    print(f"✓ Warmup schedule: {weights[0]:.4f} → {weights[50]:.4f} → {weights[100]:.4f}")


def test_drift_loss_scheduler_constant():
    """Test constant schedule."""
    scheduler = DriftLossScheduler(
        mode="constant",
        target_weight=0.2,
    )
    
    for _ in range(1000):
        assert abs(scheduler.get_weight() - 0.2) < 1e-6
        scheduler.step()
    
    print("✓ Constant schedule test passed")


def test_drift_loss_scheduler_exponential():
    """Test exponential decay schedule."""
    scheduler = DriftLossScheduler(
        mode="exponential_decay",
        target_weight=0.1,
        decay_rate=0.99,
    )
    
    weights = []
    for _ in range(50):
        weights.append(scheduler.get_weight())
        scheduler.step()
    
    # Check monotonic decrease
    for i in range(len(weights) - 1):
        assert weights[i] >= weights[i + 1]
    
    print(f"✓ Exponential decay: {weights[0]:.4f} → {weights[-1]:.4f}")


def test_drift_memory_bank():
    """Test circular buffer memory bank."""
    bank = DriftMemoryBank(max_size=100, trajectory_dim=28)
    
    # Add trajectories
    batch1 = torch.randn(30, 28)
    batch2 = torch.randn(40, 28)
    batch3 = torch.randn(50, 28)
    
    bank.push(batch1)
    assert bank.ptr == 30
    assert not bank.full
    
    bank.push(batch2)
    assert bank.ptr == 70
    assert not bank.full
    
    bank.push(batch3)
    assert bank.full
    
    # Sample from bank
    sample = bank.sample(20)
    assert sample.shape == (20, 28)
    
    # Get all
    all_trajs = bank.get_all()
    assert all_trajs.shape[0] == 100
    
    print("✓ Memory bank circular buffer test passed")


def test_combined_loss():
    """Test combined FM + drift loss."""
    fm_loss = torch.tensor(0.5)
    drift_loss = torch.tensor(0.1)
    
    total, loss_dict = compute_combined_loss(fm_loss, drift_loss, drift_weight=0.2)
    
    expected = 0.5 + 0.2 * 0.1
    assert abs(total.item() - expected) < 1e-6
    assert loss_dict['loss_fm'] == 0.5
    assert loss_dict['loss_drift'] == 0.1
    print(f"✓ Combined loss: {total.item():.4f} = {loss_dict['loss_fm']:.4f} + 0.2*{loss_dict['loss_drift']:.4f}")


def test_drift_training_wrapper():
    """Test complete training wrapper."""
    drift_loss = DriftLoss(trajectory_dim=28, loss_type="kl_divergence")
    memory_bank = DriftMemoryBank(max_size=200)
    scheduler = DriftLossScheduler(
        mode="warmup",
        target_weight=0.1,
        warmup_steps=100,
    )
    
    wrapper = DriftTrainingWrapper(
        drift_loss_fn=drift_loss,
        memory_bank=memory_bank,
        drift_scheduler=scheduler,
    )
    
    # Add expert trajectories
    expert = torch.randn(20, 28)
    wrapper.update_memory_bank_from_batch(expert)
    
    # Compute training loss
    sampled = torch.randn(8, 28, requires_grad=True)
    fm_loss = torch.tensor(0.5, requires_grad=True)
    
    total_loss, loss_dict = wrapper.compute_training_loss(sampled, fm_loss)
    
    assert 'loss_fm' in loss_dict
    assert 'loss_drift' in loss_dict
    assert 'drift_weight' in loss_dict
    
    # Step scheduler
    wrapper.step()
    
    print(f"✓ Training wrapper: fm_loss={loss_dict['loss_fm']:.4f}, drift_loss={loss_dict['loss_drift']:.4f}")


if __name__ == "__main__":
    print("Running Training Utilities Tests...\n")
    test_drift_loss_scheduler_warmup()
    test_drift_loss_scheduler_constant()
    test_drift_loss_scheduler_exponential()
    test_drift_memory_bank()
    test_combined_loss()
    test_drift_training_wrapper()
    print("\n✅ All training utilities tests passed!")
