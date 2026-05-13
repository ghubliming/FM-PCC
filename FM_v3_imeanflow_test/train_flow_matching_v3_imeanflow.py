#!/usr/bin/env python3
"""
iMeanFlow (Improved Mean Flows) Training Script

Trains dual-velocity trajectory models on D3IL demonstration data.
Supports multi-seed training with Weights & Biases logging.

Usage:
    python train_flow_matching_v3_imeanflow.py --seeds 6 7 8 9 10 --use-wandb
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import numpy as np
import argparse
import json
from datetime import datetime
from tqdm import tqdm

# Setup paths
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from flow_matcher_v3_imeanflow.models.imf_velocity import TimeConditionedDualVelocity
from flow_matcher_v3_imeanflow.utils.imf_training import (
    ImfTrainingWrapper,
    DualVelocityScheduler,
    compute_trajectory_targets,
)
from flow_matcher_v3_imeanflow.utils.imf_metrics import ImfMetricsTracker


# Try to import W&B
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


class TrajectorySynthesizer:
    """Generate synthetic trajectory data (placeholder for real D3IL data)."""
    
    @staticmethod
    def create_synthetic_trajectories(
        num_trajectories: int = 500,
        seq_length: int = 20,
        state_dim: int = 28,
        device: str = 'cpu',
    ) -> torch.Tensor:
        """Create smooth synthetic trajectories."""
        trajectories = []
        
        for _ in range(num_trajectories):
            pos = torch.randn(state_dim, device=device) * 0.5
            traj = [pos.clone()]
            vel = torch.randn(state_dim, device=device) * 0.1
            
            for t in range(seq_length - 1):
                vel = 0.8 * vel + torch.randn(state_dim, device=device) * 0.05
                pos = pos + vel * 0.1
                traj.append(pos.clone())
            
            traj = torch.stack(traj)
            trajectories.append(traj)
        
        return torch.stack(trajectories)


class ImfTrainer:
    """End-to-end trainer for Improved Mean Flows."""
    
    def __init__(
        self,
        device: str = 'cuda',
        state_dim: int = 28,
        batch_size: int = 32,
        learning_rate: float = 5e-4,
        num_epochs: int = 100,
        use_wandb: bool = False,
        wandb_project: str = 'FMPCC-iMF',
        run_name: str = None,
        seed: int = 42,
    ):
        self.device = device
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.use_wandb = use_wandb and HAS_WANDB
        self.wandb_project = wandb_project
        self.run_name = run_name or f'iMF-seed{seed}'
        self.seed = seed
        
        # Set seed for reproducibility
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Initialize model
        self.model = TimeConditionedDualVelocity(
            state_dim=state_dim,
            hidden_dim=256,
            time_dim=128,
            include_jvp=False,  # Can enable for safety-critical tasks
        ).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = DualVelocityScheduler(
            mode='u_first',
            total_steps=num_epochs * 1000,  # Assuming ~1000 steps per epoch
        )
        self.trainer = ImfTrainingWrapper(scheduler=self.scheduler)
        self.metrics = ImfMetricsTracker()
        
        # W&B logging
        if self.use_wandb:
            wandb.init(
                project=wandb_project,
                name=self.run_name,
                config={
                    'state_dim': state_dim,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'num_epochs': num_epochs,
                    'seed': seed,
                },
            )
    
    def train_epoch(self, train_loader: DataLoader) -> dict:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = []
        
        for batch_idx, (traj_batch,) in enumerate(tqdm(train_loader, desc='Training', leave=False)):
            traj_batch = traj_batch.to(self.device)
            B, T, D = traj_batch.shape
            
            # Compute targets
            u_target, v_target = compute_trajectory_targets(traj_batch)
            
            # Time samples
            t = torch.linspace(0, 1, T, device=self.device).unsqueeze(0).expand(B, T)
            
            # Forward pass
            u_pred, v_pred = self.model(traj_batch, t)
            
            # Loss
            loss, loss_dict = self.trainer.compute_training_loss(
                u_pred=u_pred,
                u_target=u_target,
                v_pred=v_pred,
                v_target=v_target,
            )
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Metrics
            self.metrics.compute_u_error(u_pred.detach(), u_target)
            self.metrics.compute_v_error(v_pred.detach(), v_target)
            
            epoch_losses.append(loss.item())
            self.trainer.step()
        
        avg_loss = np.mean(epoch_losses)
        metrics_dict = self.metrics.get_summary()
        
        return {
            'train_loss': avg_loss,
            'metrics': metrics_dict,
        }
    
    def validate(self, val_loader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        val_losses = []
        
        with torch.no_grad():
            for traj_batch, in val_loader:
                traj_batch = traj_batch.to(self.device)
                B, T, D = traj_batch.shape
                
                u_target, v_target = compute_trajectory_targets(traj_batch)
                t = torch.linspace(0, 1, T, device=self.device).unsqueeze(0).expand(B, T)
                
                u_pred, v_pred = self.model(traj_batch, t)
                loss, _ = self.trainer.compute_training_loss(
                    u_pred, u_target, v_pred, v_target
                )
                val_losses.append(loss.item())
        
        return np.mean(val_losses)
    
    def save_checkpoint(self, path: Path, is_best: bool = False):
        """Save model checkpoint."""
        path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'seed': self.seed,
            'epoch': getattr(self, 'current_epoch', 0),
        }
        
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = path.parent / 'state_best.pt'
            torch.save(checkpoint, best_path)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        """Run full training loop."""
        best_val_loss = float('inf')
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train
            self.metrics.reset()
            train_stats = self.train_epoch(train_loader)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Logging
            print(f"Epoch {epoch+1}/{self.num_epochs}")
            print(f"  Train loss: {train_stats['train_loss']:.4f}, Val loss: {val_loss:.4f}")
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_stats['train_loss'],
                    'val_loss': val_loss,
                })
            
            # Save checkpoint
            if (epoch + 1) % 5 == 0:
                ckpt_path = Path(f'checkpoints/epoch_{epoch+1}.pt')
                self.save_checkpoint(ckpt_path)
            
            # Save best
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(Path('checkpoints/state_best.pt'), is_best=True)
                print(f"  ✓ New best model (val_loss={val_loss:.4f})")
        
        print("Training complete!")
        return best_val_loss
    
    def finalize(self):
        """Finalize training and close W&B."""
        if self.use_wandb:
            wandb.finish()


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description='Train iMF models')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42],
                       help='Random seeds for reproducibility')
    parser.add_argument('--use-wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--wandb-project', default='FMPCC-iMF',
                       help='W&B project name')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=5e-4)
    parser.add_argument('--num-epochs', type=int, default=100)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("iMeanFlow Training")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"Seeds: {args.seeds}")
    print(f"Use W&B: {args.use_wandb}")
    print()
    
    # Training loop for each seed
    for seed in args.seeds:
        print(f"\n{'='*80}")
        print(f"Training Seed {seed}")
        print(f"{'='*80}\n")
        
        # Create synthetic data (placeholder)
        print("Generating synthetic trajectory data...")
        trajectories = TrajectorySynthesizer.create_synthetic_trajectories(
            num_trajectories=500,
            seq_length=20,
            state_dim=28,
            device=args.device,
        )
        
        split_idx = int(0.8 * len(trajectories))
        train_traj = trajectories[:split_idx]
        val_traj = trajectories[split_idx:]
        
        train_loader = DataLoader(
            TensorDataset(train_traj),
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(val_traj),
            batch_size=args.batch_size,
            shuffle=False,
        )
        
        # Initialize trainer
        trainer = ImfTrainer(
            device=args.device,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            run_name=f'iMF-seed{seed}',
            seed=seed,
        )
        
        # Train
        best_val_loss = trainer.train(train_loader, val_loader)
        
        # Cleanup
        trainer.finalize()
        
        print(f"✓ Seed {seed} complete (best_val_loss={best_val_loss:.4f})")
    
    print("\n" + "=" * 80)
    print("All training jobs complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
