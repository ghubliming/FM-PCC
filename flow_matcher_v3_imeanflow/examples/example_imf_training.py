#!/usr/bin/env python3
"""
Example: Training Improved Mean Flows on Generated Trajectory Data

Demonstrates:
- Loading synthetic trajectory datasets
- Creating dual-velocity models
- Computing u/v training targets
- Training with dual-loss and scheduler
- Monitoring metrics
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.imf_velocity import TimeConditionedDualVelocity
from utils.imf_training import (
    ImfTrainingWrapper,
    DualVelocityScheduler,
    compute_trajectory_targets,
)
from utils.imf_metrics import ImfMetricsTracker


def create_synthetic_trajectories(
    num_trajectories: int = 100,
    seq_length: int = 20,
    state_dim: int = 28,
    device: str = 'cpu',
) -> torch.Tensor:
    """
    Generate synthetic trajectory data.
    
    Simulates smooth, curved paths with:
    - Random initial position
    - Lazy Brownian motion dynamics
    - Gaussian smoothing
    """
    trajectories = []
    
    for _ in range(num_trajectories):
        # Random starting position
        pos = torch.randn(state_dim, device=device) * 0.5
        
        # Generate trajectory with smooth random walk
        traj = [pos.clone()]
        
        # Velocity
        vel = torch.randn(state_dim, device=device) * 0.1
        
        for t in range(seq_length - 1):
            # Update velocity (momentum + noise)
            vel = 0.8 * vel + torch.randn(state_dim, device=device) * 0.05
            
            # Update position
            pos = pos + vel * 0.1
            
            traj.append(pos.clone())
        
        traj = torch.stack(traj)  # (T, D)
        trajectories.append(traj)
    
    return torch.stack(trajectories)  # (B, T, D)


def create_dataloader(
    trajectories: torch.Tensor,
    batch_size: int = 16,
    shuffle: bool = True,
) -> DataLoader:
    """Create PyTorch DataLoader from trajectories."""
    dataset = TensorDataset(trajectories)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return loader


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    trainer: ImfTrainingWrapper,
    metrics: ImfMetricsTracker,
    device: str,
) -> dict:
    """
    Train for one epoch.
    
    Returns:
        stats: Dictionary with loss and metric summaries
    """
    model.train()
    
    epoch_losses = []
    epoch_metrics = []
    
    for batch_idx, (traj_batch,) in enumerate(dataloader):
        traj_batch = traj_batch.to(device)
        B, T, D = traj_batch.shape
        
        # Compute velocity targets
        u_target, v_target = compute_trajectory_targets(traj_batch)
        
        # Time samples (uniform across trajectory)
        t = torch.linspace(0, 1, T, device=device).unsqueeze(0).expand(B, T)
        
        # Model forward pass
        u_pred, v_pred = model(traj_batch, t)
        
        # Compute loss
        loss, loss_dict = trainer.compute_training_loss(
            u_pred=u_pred,
            u_target=u_target,
            v_pred=v_pred,
            v_target=v_target,
        )
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        # Metrics
        metrics.compute_u_error(u_pred, u_target)
        metrics.compute_v_error(v_pred, v_target)
        metrics.compute_smoothness(traj_batch)
        u_contrib, v_contrib = metrics.compute_decomposition(u_pred, v_pred)
        
        epoch_losses.append(loss.item())
        epoch_metrics.append(loss_dict)
        
        trainer.step()
        
        if batch_idx % 5 == 0:
            print(
                f"  Batch {batch_idx:3d}: "
                f"loss={loss.item():.4f}, "
                f"L_u={loss_dict['loss_u']:.4f}, "
                f"L_v={loss_dict['loss_v']:.4f}, "
                f"w_u={loss_dict['weight_u']:.2f}, "
                f"w_v={loss_dict['weight_v']:.2f}, "
                f"u_contrib={u_contrib:.2f}"
            )
    
    # Summary statistics
    stats = {
        'loss': sum(epoch_losses) / len(epoch_losses),
        'metrics': metrics.get_summary(),
    }
    
    return stats


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    trainer: ImfTrainingWrapper,
    device: str,
) -> dict:
    """Validate on held-out data."""
    model.eval()
    
    val_losses = []
    
    with torch.no_grad():
        for traj_batch, in dataloader:
            traj_batch = traj_batch.to(device)
            B, T, D = traj_batch.shape
            
            u_target, v_target = compute_trajectory_targets(traj_batch)
            t = torch.linspace(0, 1, T, device=device).unsqueeze(0).expand(B, T)
            
            u_pred, v_pred = model(traj_batch, t)
            loss, _ = trainer.compute_training_loss(
                u_pred=u_pred,
                u_target=u_target,
                v_pred=v_pred,
                v_target=v_target,
            )
            
            val_losses.append(loss.item())
    
    return {'val_loss': sum(val_losses) / len(val_losses)}


def main():
    """Main training script."""
    # Configuration
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    STATE_DIM = 28
    BATCH_SIZE = 16
    NUM_EPOCHS = 20
    LEARNING_RATE = 1e-3
    
    print("=" * 80)
    print("iMeanFlow Training Example")
    print("=" * 80)
    print(f"Device: {DEVICE}")
    print(f"State dimension: {STATE_DIM}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Epochs: {NUM_EPOCHS}")
    print()
    
    # ============================================================================
    # Data
    # ============================================================================
    print("Creating synthetic trajectory data...")
    trajectories = create_synthetic_trajectories(
        num_trajectories=500,
        seq_length=20,
        state_dim=STATE_DIM,
        device=DEVICE,
    )
    print(f"  Trajectories shape: {trajectories.shape}")
    
    # Split train/val
    split_idx = int(0.8 * len(trajectories))
    train_traj = trajectories[:split_idx]
    val_traj = trajectories[split_idx:]
    
    train_loader = create_dataloader(train_traj, BATCH_SIZE, shuffle=True)
    val_loader = create_dataloader(val_traj, BATCH_SIZE, shuffle=False)
    print(f"  Train: {len(train_traj)} trajectories")
    print(f"  Val: {len(val_traj)} trajectories")
    print()
    
    # ============================================================================
    # Model
    # ============================================================================
    print("Creating dual-velocity model...")
    model = TimeConditionedDualVelocity(
        state_dim=STATE_DIM,
        hidden_dim=128,
        time_dim=64,
        use_jvp=False,  # Disable JVP for example (can enable for safety)
    ).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")
    print()
    
    # ============================================================================
    # Training
    # ============================================================================
    print("Setting up training...")
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler_obj = DualVelocityScheduler(
        mode='u_first',
        total_steps=NUM_EPOCHS * len(train_loader),
        weight_u_start=0.8,
        weight_v_start=0.0,
        weight_u_end=0.5,
        weight_v_end=0.5,
    )
    
    trainer = ImfTrainingWrapper(
        loss_weights={'u': 0.5, 'v': 0.5, 'jvp': 0.0},
        scheduler=scheduler_obj,
        loss_type='mse',
    )
    
    metrics = ImfMetricsTracker(window_size=50)
    print(f"  Loss schedule: u_first")
    print(f"  Optimizer: Adam (lr={LEARNING_RATE})")
    print()
    
    # ============================================================================
    # Train Loop
    # ============================================================================
    print("Starting training...")
    print()
    
    best_val_loss = float('inf')
    
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        
        # Train
        metrics.reset()
        train_stats = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            trainer=trainer,
            metrics=metrics,
            device=DEVICE,
        )
        
        # Validate
        val_stats = validate(
            model=model,
            dataloader=val_loader,
            trainer=trainer,
            device=DEVICE,
        )
        
        # Log
        print(
            f"  Train loss: {train_stats['loss']:.4f}, "
            f"Val loss: {val_stats['val_loss']:.4f}"
        )
        
        metrics_summary = train_stats['metrics']
        if metrics_summary:
            print(f"  Metrics: ", end="")
            if 'u_error' in metrics_summary:
                print(f"u_err={metrics_summary['u_error']:.4f}, ", end="")
            if 'v_error' in metrics_summary:
                print(f"v_err={metrics_summary['v_error']:.4f}, ", end="")
            if 'smoothness' in metrics_summary:
                print(f"smooth={metrics_summary['smoothness']:.4f}", end="")
            print()
        
        # Save best model
        if val_stats['val_loss'] < best_val_loss:
            best_val_loss = val_stats['val_loss']
            print(f"  ✓ New best model (val_loss={best_val_loss:.4f})")
        
        print()
    
    print("=" * 80)
    print("Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
