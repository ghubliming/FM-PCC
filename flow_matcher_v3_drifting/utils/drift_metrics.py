"""
Drift Metrics & Logging for FM-D Training and Inference

Computes and tracks:
- Drift loss values (KL, MMD, adversarial)
- Trajectory quality metrics (smoothness, constraint violations)
- Memory bank statistics
- ODE solver performance (steps, time)
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, List
from collections import defaultdict


class DriftMetricsTracker:
    """
    Tracks drift-related metrics during training and inference.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: Number of steps for rolling average
        """
        self.window_size = window_size
        self.metrics = defaultdict(list)
        self.step_count = 0

    def update(self, **kwargs) -> None:
        """
        Record metric values.
        
        Args:
            **kwargs: Named metrics (e.g., drift_loss=0.5, ode_steps=10)
        """
        for name, value in kwargs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            self.metrics[name].append(value)
        self.step_count += 1

    def get_mean(self, name: str) -> float:
        """Get mean of metric over window."""
        if name not in self.metrics or len(self.metrics[name]) == 0:
            return 0.0
        return sum(self.metrics[name][-self.window_size:]) / min(
            self.window_size, len(self.metrics[name])
        )

    def get_all_means(self) -> Dict[str, float]:
        """Get means of all tracked metrics."""
        return {name: self.get_mean(name) for name in self.metrics.keys()}

    def reset(self) -> None:
        """Clear all metrics."""
        self.metrics.clear()
        self.step_count = 0


def compute_trajectory_smoothness(trajectory: torch.Tensor) -> torch.Tensor:
    """
    Compute trajectory smoothness as negative acceleration magnitude.
    Smoother trajectories have lower acceleration.
    
    Args:
        trajectory: (B, T, state_dim) or (T, state_dim)
        
    Returns:
        smoothness metric (scalar)
    """
    if trajectory.dim() == 2:
        trajectory = trajectory.unsqueeze(0)
    
    # Compute first derivative (velocity)
    vel = torch.diff(trajectory, dim=1)  # (B, T-1, state_dim)
    
    # Compute second derivative (acceleration)
    accel = torch.diff(vel, dim=1)  # (B, T-2, state_dim)
    
    # L2 norm of acceleration
    accel_norm = torch.norm(accel, p=2, dim=-1)  # (B, T-2)
    
    # Mean acceleration magnitude (lower = smoother)
    smoothness = accel_norm.mean()
    
    return smoothness


def compute_constraint_satisfaction(
    trajectory: torch.Tensor,
    constraint_fn,
    threshold: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Evaluate constraint satisfaction along trajectory.
    
    Args:
        trajectory: (B, T, state_dim) or (T, state_dim)
        constraint_fn: Function returning constraint values (lower = better)
        threshold: Constraint threshold (violations if > threshold)
        
    Returns:
        dict with keys:
            'violation_rate': fraction of steps violating constraints
            'mean_violation': mean constraint violation magnitude
            'max_violation': maximum violation in trajectory
    """
    if trajectory.dim() == 2:
        trajectory = trajectory.unsqueeze(0)
    
    B, T, state_dim = trajectory.shape
    violations = []
    
    for t in range(T):
        state_t = trajectory[:, t, :]  # (B, state_dim)
        constraint_vals = constraint_fn(state_t)  # (B,)
        violations.append(constraint_vals)
    
    violations = torch.stack(violations, dim=1)  # (B, T)
    
    # Count violations
    violated = (violations > threshold).float()
    violation_rate = violated.mean()
    
    # Magnitude of violations
    violation_mag = F.relu(violations - threshold)
    mean_violation = violation_mag.mean()
    max_violation = violation_mag.max()
    
    return {
        'violation_rate': violation_rate,
        'mean_violation': mean_violation,
        'max_violation': max_violation,
    }


def compute_trajectory_fidelity(
    sampled_trajectory: torch.Tensor,
    reference_trajectories: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Measure fidelity of sampled trajectory to reference distribution.
    
    Args:
        sampled_trajectory: (B, T, state_dim) generated trajectory
        reference_trajectories: (N, T, state_dim) reference expert trajectories
        
    Returns:
        dict with keys:
            'min_distance': minimum L2 distance to any reference
            'mean_distance': mean L2 distance to all references
            'coverage': fraction of reference trajectories within threshold
    """
    if sampled_trajectory.dim() == 2:
        sampled_trajectory = sampled_trajectory.unsqueeze(0)
    
    B = sampled_trajectory.shape[0]
    N = reference_trajectories.shape[0]
    
    # Flatten trajectories
    sampled_flat = sampled_trajectory.reshape(B, -1)  # (B, T*state_dim)
    ref_flat = reference_trajectories.reshape(N, -1)  # (N, T*state_dim)
    
    # Compute pairwise distances
    # (B, N)
    distances = torch.cdist(sampled_flat, ref_flat, p=2)
    
    min_distance = distances.min(dim=1)[0].mean()
    mean_distance = distances.mean()
    
    # Coverage: samples within 1 std of at least one reference
    threshold = distances.std()
    coverage = (distances.min(dim=1)[0] < threshold).float().mean()
    
    return {
        'min_distance': min_distance,
        'mean_distance': mean_distance,
        'coverage': coverage,
    }


def compute_ode_efficiency(
    num_steps_taken: int,
    max_steps: int = 50,
) -> Dict[str, float]:
    """
    Compute ODE solver efficiency metrics.
    
    Args:
        num_steps_taken: Actual steps used by adaptive solver
        max_steps: Maximum allowed steps
        
    Returns:
        dict with efficiency metrics
    """
    efficiency = num_steps_taken / max_steps
    
    return {
        'steps_taken': float(num_steps_taken),
        'step_efficiency': efficiency,
        'wasted_budget': 1.0 - efficiency,
    }


class DriftLogger:
    """
    Comprehensive logger for FM-D metrics.
    """
    
    def __init__(self, log_file: Optional[str] = None):
        """
        Args:
            log_file: Optional file path for saving logs
        """
        self.log_file = log_file
        self.tracker = DriftMetricsTracker()
        self.logs = []

    def log_training_step(
        self,
        step: int,
        loss_fm: float,
        loss_drift: float,
        loss_total: float,
        lr: float,
    ) -> None:
        """Log training step metrics."""
        entry = {
            'step': step,
            'loss_fm': loss_fm,
            'loss_drift': loss_drift,
            'loss_total': loss_total,
            'learning_rate': lr,
        }
        self.logs.append(entry)
        self.tracker.update(
            loss_fm=loss_fm,
            loss_drift=loss_drift,
            loss_total=loss_total,
        )

    def log_validation_step(
        self,
        step: int,
        trajectory: torch.Tensor,
        reference_trajectories: Optional[torch.Tensor] = None,
        constraint_fn=None,
    ) -> Dict[str, float]:
        """Log validation metrics."""
        metrics = {}
        
        # Trajectory smoothness
        smoothness = compute_trajectory_smoothness(trajectory)
        metrics['smoothness'] = float(smoothness)
        self.tracker.update(smoothness=smoothness)
        
        # Constraint satisfaction
        if constraint_fn is not None:
            constraints = compute_constraint_satisfaction(
                trajectory, constraint_fn
            )
            metrics.update({k: float(v) for k, v in constraints.items()})
            self.tracker.update(**constraints)
        
        # Fidelity to expert distribution
        if reference_trajectories is not None:
            fidelity = compute_trajectory_fidelity(
                trajectory, reference_trajectories
            )
            metrics.update({k: float(v) for k, v in fidelity.items()})
            self.tracker.update(**fidelity)
        
        return metrics

    def get_summary(self) -> Dict[str, float]:
        """Get summary of all tracked metrics."""
        return self.tracker.get_all_means()

    def save(self, path: str) -> None:
        """Save logs to file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.logs, f, indent=2)


def log_memory_bank_stats(memory_bank: torch.Tensor) -> Dict[str, float]:
    """
    Compute statistics of drift loss memory bank.
    
    Args:
        memory_bank: (N, trajectory_dim) tensor of stored trajectories
        
    Returns:
        Statistics dict
    """
    if memory_bank.shape[0] == 0:
        return {'occupancy': 0.0}
    
    # Mean distance between trajectories
    pairwise_dist = torch.cdist(memory_bank, memory_bank, p=2)
    mean_dist = pairwise_dist[pairwise_dist > 0].mean()
    
    return {
        'bank_size': float(memory_bank.shape[0]),
        'bank_dim': float(memory_bank.shape[1]),
        'mean_pairwise_distance': float(mean_dist),
    }
