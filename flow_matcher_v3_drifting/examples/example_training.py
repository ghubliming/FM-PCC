"""
Example: FM-D Training Script

Demonstrates full training loop with drift loss augmentation.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from flow_matcher_v3_drifting.models import GaussianDiffusion, DriftLoss
from flow_matcher_v3_drifting.utils.drift_training import (
    DriftTrainingWrapper,
    DriftLossScheduler,
    DriftMemoryBank,
)
from flow_matcher_v3_drifting.utils.drift_metrics import DriftMetricsTracker


def create_dummy_dataset(num_samples=100, trajectory_dim=28, horizon=8):
    """Create synthetic training data."""
    # Simulated expert trajectories
    trajectories = torch.randn(num_samples, trajectory_dim)
    conditions = torch.randn(num_samples, 16)  # goal/context dimension
    
    dataset = TensorDataset(trajectories, conditions)
    return dataset


def example_fm_d_training():
    """
    Example FM-D training loop.
    
    Note: This uses dummy components for illustration.
    In real usage, load actual FM-ODE model and dataset.
    """
    
    print("=" * 60)
    print("FM-D Training Example")
    print("=" * 60)
    
    # Hyperparameters
    num_epochs = 5
    batch_size = 8
    learning_rate = 1e-4
    trajectory_dim = 28
    
    # Create dummy dataset
    dataset = create_dummy_dataset(num_samples=100, trajectory_dim=trajectory_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize FM components
    # (In real usage, load pre-trained FM-ODE model)
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(trajectory_dim + 16, trajectory_dim)
        
        def forward(self, x, cond, t, returns=None, use_dropout=False):
            return self.fc(torch.cat([x, cond], dim=-1))
    
    model = DummyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize drift components
    drift_loss_fn = DriftLoss(
        trajectory_dim=trajectory_dim,
        loss_type='kl_divergence',
        memory_bank_size=200,
    )
    
    drift_scheduler = DriftLossScheduler(
        mode='warmup',
        start_weight=0.0,
        target_weight=0.1,
        warmup_steps=100,
    )
    
    memory_bank = DriftMemoryBank(max_size=200, trajectory_dim=trajectory_dim)
    
    trainer = DriftTrainingWrapper(
        drift_loss_fn=drift_loss_fn,
        memory_bank=memory_bank,
        drift_scheduler=drift_scheduler,
    )
    
    metrics = DriftMetricsTracker()
    
    print(f"\nTraining Configuration:")
    print(f"  - Epochs: {num_epochs}")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Trajectory dim: {trajectory_dim}")
    print(f"  - Drift loss type: kl_divergence")
    print(f"  - Drift warmup: 100 steps\n")
    
    # Training loop
    total_steps = 0
    for epoch in range(num_epochs):
        epoch_losses = {'fm': [], 'drift': [], 'total': []}
        
        for batch_idx, (trajectories, conditions) in enumerate(dataloader):
            # Update memory bank with expert trajectories
            trainer.update_memory_bank_from_batch(trajectories)
            
            # Forward pass: predict velocity field
            sampled = model(trajectories, conditions, torch.tensor(0.5))
            
            # Dummy FM loss (MSE for example)
            target = torch.randn_like(trajectories)
            fm_loss = nn.functional.mse_loss(sampled, target)
            
            # Compute total loss with drift
            total_loss, loss_dict = trainer.compute_training_loss(
                sampled.detach(), fm_loss
            )
            
            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Update scheduler
            trainer.step()
            total_steps += 1
            
            # Track metrics
            metrics.update(
                fm_loss=loss_dict['loss_fm'],
                drift_loss=loss_dict['loss_drift'],
                total_loss=loss_dict['loss_total'],
                drift_weight=loss_dict['drift_weight'],
            )
            
            epoch_losses['fm'].append(loss_dict['loss_fm'])
            epoch_losses['drift'].append(loss_dict['loss_drift'])
            epoch_losses['total'].append(loss_dict['loss_total'])
            
            # Log progress
            if (batch_idx + 1) % 5 == 0:
                print(
                    f"Epoch {epoch+1}/{num_epochs} | Step {total_steps:3d} | "
                    f"FM Loss: {loss_dict['loss_fm']:.4f} | "
                    f"Drift Loss: {loss_dict['loss_drift']:.4f} | "
                    f"λ: {loss_dict['drift_weight']:.4f}"
                )
        
        # Epoch summary
        avg_fm = sum(epoch_losses['fm']) / len(epoch_losses['fm'])
        avg_drift = sum(epoch_losses['drift']) / len(epoch_losses['drift'])
        avg_total = sum(epoch_losses['total']) / len(epoch_losses['total'])
        
        print(f"\n[Epoch {epoch+1}] Avg FM Loss: {avg_fm:.4f}, "
              f"Avg Drift Loss: {avg_drift:.4f}, "
              f"Avg Total: {avg_total:.4f}\n")
    
    # Training summary
    print("=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total steps: {total_steps}")
    print(f"Final metrics:")
    for name, value in metrics.get_all_means().items():
        print(f"  - {name}: {value:.4f}")


if __name__ == "__main__":
    example_fm_d_training()
