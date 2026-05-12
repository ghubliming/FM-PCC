"""
FM-D Training Utilities

Functions for:
- Drift loss computation during training
- Warmup schedule for drift activation
- Combined FM + drift loss optimization
- Memory bank management
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


class DriftLossScheduler:
    """
    Manages drift loss weight scheduling during training.
    
    Useful patterns:
    - Warmup: Start with λ=0, linearly increase to target
    - Constant: Keep λ fixed
    - Decay: Exponentially decay λ over training
    """
    
    def __init__(
        self,
        mode: str = "warmup",
        start_weight: float = 0.0,
        target_weight: float = 0.1,
        warmup_steps: int = 1000,
        decay_rate: float = 0.999,
    ):
        """
        Args:
            mode: "warmup" | "constant" | "exponential_decay"
            start_weight: Initial drift loss weight (for warmup)
            target_weight: Target / constant drift loss weight
            warmup_steps: Number of steps for warmup phase
            decay_rate: Decay rate per step (exponential mode)
        """
        self.mode = mode
        self.start_weight = start_weight
        self.target_weight = target_weight
        self.warmup_steps = warmup_steps
        self.decay_rate = decay_rate
        self.step = 0

    def step(self) -> None:
        """Advance scheduler by one step."""
        self.step += 1

    def get_weight(self) -> float:
        """Get current drift loss weight."""
        if self.mode == "warmup":
            if self.step >= self.warmup_steps:
                return self.target_weight
            else:
                # Linear warmup
                progress = self.step / self.warmup_steps
                return (
                    self.start_weight +
                    (self.target_weight - self.start_weight) * progress
                )
        
        elif self.mode == "constant":
            return self.target_weight
        
        elif self.mode == "exponential_decay":
            return self.target_weight * (self.decay_rate ** self.step)
        
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def reset(self) -> None:
        """Reset scheduler."""
        self.step = 0


class DriftMemoryBank:
    """
    Circular buffer for storing expert trajectory samples.
    Used to compute drift loss reference distribution.
    """
    
    def __init__(
        self,
        max_size: int = 5000,
        trajectory_dim: int = 28,
    ):
        """
        Args:
            max_size: Maximum number of trajectories to store
            trajectory_dim: Dimension of flattened trajectory
        """
        self.max_size = max_size
        self.trajectory_dim = trajectory_dim
        self.buffer = torch.zeros(max_size, trajectory_dim, dtype=torch.float32)
        self.ptr = 0
        self.full = False

    def push(self, trajectories: torch.Tensor) -> None:
        """
        Add trajectories to buffer (circular).
        
        Args:
            trajectories: (B, T, state_dim) or (B, T*state_dim) tensor
        """
        if trajectories.dim() == 3:
            B, T, state_dim = trajectories.shape
            trajectories = trajectories.reshape(B, -1)
        
        B = trajectories.shape[0]
        
        if self.ptr + B <= self.max_size:
            self.buffer[self.ptr:self.ptr + B] = trajectories
            self.ptr += B
        else:
            # Wrap around
            remaining = self.max_size - self.ptr
            self.buffer[self.ptr:] = trajectories[:remaining]
            self.buffer[:B - remaining] = trajectories[remaining:]
            self.full = True
        
        if self.ptr == self.max_size:
            self.ptr = 0
            self.full = True

    def sample(self, batch_size: int) -> torch.Tensor:
        """
        Sample random batch from buffer.
        
        Args:
            batch_size: Number of trajectories to sample
            
        Returns:
            (batch_size, trajectory_dim) tensor
        """
        if not self.full and self.ptr == 0:
            return torch.zeros(batch_size, self.trajectory_dim)
        
        max_idx = self.max_size if self.full else self.ptr
        indices = torch.randint(0, max_idx, (batch_size,))
        return self.buffer[indices].clone()

    def get_all(self) -> torch.Tensor:
        """
        Get all stored trajectories.
        
        Returns:
            (N, trajectory_dim) tensor where N ≤ max_size
        """
        if self.full:
            return self.buffer.clone()
        else:
            return self.buffer[:self.ptr].clone()


def compute_combined_loss(
    fm_loss: torch.Tensor,
    drift_loss: torch.Tensor,
    drift_weight: float = 0.1,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute combined FM + drift loss for training.
    
    Args:
        fm_loss: Flow matching regression loss
        drift_loss: Drift loss (KL / MMD / adversarial)
        drift_weight: Weight of drift term (λ)
        
    Returns:
        (total_loss, loss_dict)
    """
    total_loss = fm_loss + drift_weight * drift_loss
    
    return total_loss, {
        'loss_fm': fm_loss.detach().cpu().item(),
        'loss_drift': drift_loss.detach().cpu().item(),
        'loss_total': total_loss.detach().cpu().item(),
        'drift_weight': drift_weight,
    }


class DriftTrainingWrapper:
    """
    Wrapper for drift-aware training loop modifications.
    Handles:
    - Memory bank updates
    - Drift loss computation
    - Warmup scheduling
    - Metric logging
    """
    
    def __init__(
        self,
        drift_loss_fn,
        memory_bank: Optional[DriftMemoryBank] = None,
        drift_scheduler: Optional[DriftLossScheduler] = None,
    ):
        """
        Args:
            drift_loss_fn: DriftLoss module
            memory_bank: DriftMemoryBank for reference trajectories
            drift_scheduler: DriftLossScheduler for weight scheduling
        """
        self.drift_loss_fn = drift_loss_fn
        self.memory_bank = memory_bank or DriftMemoryBank()
        self.drift_scheduler = drift_scheduler or DriftLossScheduler(
            mode="warmup",
            target_weight=0.1,
            warmup_steps=1000,
        )

    def update_memory_bank_from_batch(
        self,
        expert_trajectories: torch.Tensor,
    ) -> None:
        """
        Update memory bank with expert trajectories from training batch.
        
        Args:
            expert_trajectories: (B, T, state_dim) or (B, T*state_dim)
        """
        if expert_trajectories.dim() == 3:
            B, T, state_dim = expert_trajectories.shape
            expert_trajectories = expert_trajectories.reshape(B, -1)
        
        self.memory_bank.push(expert_trajectories)
        self.drift_loss_fn.update_memory_bank(expert_trajectories)

    def compute_training_loss(
        self,
        sampled_trajectory: torch.Tensor,
        fm_loss: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute combined FM + drift loss for a training step.
        
        Args:
            sampled_trajectory: (B, T, state_dim) generated trajectory
            fm_loss: Flow matching loss from main training loop
            
        Returns:
            (total_loss, loss_dict)
        """
        # Get current drift weight from scheduler
        drift_weight = self.drift_scheduler.get_weight()
        
        # Compute drift loss
        drift_loss_dict = self.drift_loss_fn.forward(sampled_trajectory)
        drift_loss = drift_loss_dict['loss']
        
        # Combine losses
        total_loss, loss_dict = compute_combined_loss(
            fm_loss,
            drift_loss,
            drift_weight=drift_weight,
        )
        
        loss_dict['drift_loss_dict'] = drift_loss_dict
        
        return total_loss, loss_dict

    def step(self) -> None:
        """Advance scheduler by one step."""
        self.drift_scheduler.step()


def create_drift_training_config(
    base_config: Dict,
    drift_enabled: bool = True,
    drift_loss_weight: float = 0.1,
    drift_warmup_epochs: int = 10,
) -> Dict:
    """
    Create FM-D training configuration from base config.
    
    Args:
        base_config: Base FM-v3 configuration
        drift_enabled: Enable drift augmentation
        drift_loss_weight: Target drift loss weight
        drift_warmup_epochs: Epochs for drift warmup
        
    Returns:
        Updated config dict with drift settings
    """
    config = base_config.copy()
    
    config['use_drift_augmentation'] = drift_enabled
    config['drift_loss_weight'] = drift_loss_weight
    config['drift_loss_type'] = 'kl_divergence'
    config['drift_warmup_epochs'] = drift_warmup_epochs
    
    # Derived settings
    if 'n_steps_per_epoch' in config:
        config['drift_warmup_steps'] = (
            config['n_steps_per_epoch'] * drift_warmup_epochs
        )
    
    return config
