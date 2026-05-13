"""
Improved Mean Flows Metrics and Analysis

Implements trajectory metrics, decomposition analysis, and performance tracking.
"""

import torch
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ImfMetricsTracker:
    """
    Tracks metrics for dual-velocity trajectory generation.
    
    Computes:
    - u_error: Average velocity prediction error
    - v_error: Instantaneous velocity prediction error
    - combined_error: Total velocity prediction error (u + v vs target)
    - smoothness: Trajectory smoothness (acceleration magnitude)
    - decomposition: How much u vs v contributes to final trajectory
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics tracker.
        
        Args:
            window_size: Exponential moving average window
        """
        self.window_size = window_size
        self.reset()
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.u_errors = []
        self.v_errors = []
        self.combined_errors = []
        self.smoothness_values = []
        self.u_contrib = []
        self.v_contrib = []
    
    def compute_u_error(
        self,
        u_pred: torch.Tensor,
        u_target: torch.Tensor,
    ) -> float:
        """
        Compute average velocity prediction error (MSE).
        
        Args:
            u_pred: Predicted average velocity (B, T, D) or (B, D)
            u_target: Target average velocity
        
        Returns:
            error: MSE loss value
        """
        error = F.mse_loss(u_pred, u_target).item()
        self.u_errors.append(error)
        return error
    
    def compute_v_error(
        self,
        v_pred: torch.Tensor,
        v_target: torch.Tensor,
    ) -> float:
        """
        Compute instantaneous velocity prediction error (MSE).
        
        Args:
            v_pred: Predicted instantaneous velocity
            v_target: Target instantaneous velocity
        
        Returns:
            error: MSE loss value
        """
        error = F.mse_loss(v_pred, v_target).item()
        self.v_errors.append(error)
        return error
    
    def compute_combined_error(
        self,
        u_pred: torch.Tensor,
        v_pred: torch.Tensor,
        target_velocity: torch.Tensor,
    ) -> float:
        """
        Compute combined velocity error (u + v vs target).
        
        Args:
            u_pred: Predicted average velocity
            v_pred: Predicted instantaneous velocity
            target_velocity: Target total velocity (du/dt from trajectory)
        
        Returns:
            error: MSE loss value
        """
        combined_pred = u_pred + v_pred
        error = F.mse_loss(combined_pred, target_velocity).item()
        self.combined_errors.append(error)
        return error
    
    def compute_smoothness(
        self,
        trajectory: torch.Tensor,
        normalize: bool = True,
    ) -> float:
        """
        Compute trajectory smoothness via acceleration magnitude.
        
        Smoothness = 1 / (1 + mean(||acceleration||))
        High values → smooth trajectory
        Low values → jerky trajectory
        
        Args:
            trajectory: (B, T, D) trajectory sequence
            normalize: Whether to normalize by max acceleration
        
        Returns:
            smoothness: Smoothness score in [0, 1]
        """
        if trajectory.dim() != 3:
            raise ValueError(f"Expected (B, T, D), got {trajectory.shape}")
        
        B, T, D = trajectory.shape
        
        if T < 3:
            return 1.0  # Can't compute acceleration
        
        # Compute velocities (central difference)
        v = torch.zeros(B, T, D, device=trajectory.device)
        v[:, 1:-1] = (trajectory[:, 2:] - trajectory[:, :-2]) / 2.0
        v[:, 0] = trajectory[:, 1] - trajectory[:, 0]
        v[:, -1] = trajectory[:, -1] - trajectory[:, -2]
        
        # Compute accelerations
        a = torch.zeros(B, T, D, device=trajectory.device)
        a[:, 1:-1] = (v[:, 2:] - v[:, :-2]) / 2.0
        a[:, 0] = v[:, 1] - v[:, 0]
        a[:, -1] = v[:, -1] - v[:, -2]
        
        # Magnitude of acceleration
        accel_mag = torch.norm(a, dim=-1)  # (B, T)
        mean_accel = accel_mag.mean()
        
        # Smoothness score (higher = smoother)
        smoothness = 1.0 / (1.0 + mean_accel.item())
        self.smoothness_values.append(smoothness)
        
        return smoothness
    
    def compute_decomposition(
        self,
        u_pred: torch.Tensor,
        v_pred: torch.Tensor,
    ) -> Tuple[float, float]:
        """
        Analyze contribution of u vs v to combined velocity.
        
        Returns:
            u_contribution: Fraction of total motion explained by u
            v_contribution: Fraction of total motion explained by v
        """
        combined = u_pred + v_pred
        
        # L2 norms along state dimension
        u_norm = torch.norm(u_pred, p=2)
        v_norm = torch.norm(v_pred, p=2)
        combined_norm = torch.norm(combined, p=2)
        
        if combined_norm.item() < 1e-6:
            return 0.5, 0.5
        
        # Contribution as fraction of combined motion
        u_contrib = (u_norm / combined_norm).item()
        v_contrib = (v_norm / combined_norm).item()
        
        # Normalize to sum to 1 (in case they both contribute < 1)
        total = u_contrib + v_contrib
        if total > 0:
            u_contrib /= total
            v_contrib /= total
        
        self.u_contrib.append(u_contrib)
        self.v_contrib.append(v_contrib)
        
        return u_contrib, v_contrib
    
    def compute_velocity_alignment(
        self,
        u_pred: torch.Tensor,
        v_pred: torch.Tensor,
    ) -> float:
        """
        Compute alignment between u and v vectors.
        
        Alignment in [0, 1]: 1 = perfectly aligned, 0 = orthogonal
        
        Args:
            u_pred: Average velocity
            v_pred: Instantaneous velocity
        
        Returns:
            alignment: Cosine similarity score
        """
        # Flatten to vectors
        u_flat = u_pred.reshape(-1)
        v_flat = v_pred.reshape(-1)
        
        # Cosine similarity
        cos_sim = F.cosine_similarity(u_flat.unsqueeze(0), v_flat.unsqueeze(0))
        
        # Convert from [-1, 1] to [0, 1]
        alignment = (cos_sim.item() + 1.0) / 2.0
        
        return alignment
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics across all tracked metrics."""
        summary = {}
        
        if self.u_errors:
            summary['u_error'] = sum(self.u_errors[-self.window_size:]) / len(self.u_errors[-self.window_size:])
        
        if self.v_errors:
            summary['v_error'] = sum(self.v_errors[-self.window_size:]) / len(self.v_errors[-self.window_size:])
        
        if self.combined_errors:
            summary['combined_error'] = sum(self.combined_errors[-self.window_size:]) / len(self.combined_errors[-self.window_size:])
        
        if self.smoothness_values:
            summary['smoothness'] = sum(self.smoothness_values[-self.window_size:]) / len(self.smoothness_values[-self.window_size:])
        
        if self.u_contrib:
            summary['u_contribution'] = sum(self.u_contrib[-self.window_size:]) / len(self.u_contrib[-self.window_size:])
        
        if self.v_contrib:
            summary['v_contribution'] = sum(self.v_contrib[-self.window_size:]) / len(self.v_contrib[-self.window_size:])
        
        return summary


class TrajectoryQualityMetrics:
    """
    Comprehensive trajectory quality analysis.
    
    Evaluates:
    - Feasibility: Acceleration, velocity limits
    - Optimality: Path length, energy efficiency
    - Safety: Distance to obstacles (if provided)
    """
    
    @staticmethod
    def compute_path_length(trajectory: torch.Tensor) -> float:
        """
        Compute total path length of trajectory.
        
        Args:
            trajectory: (B, T, D) or (T, D) trajectory
        
        Returns:
            total_length: Sum of step distances
        """
        if trajectory.dim() == 3:
            trajectory = trajectory[0]  # Take first batch
        
        diffs = torch.diff(trajectory, dim=0)
        distances = torch.norm(diffs, dim=-1)
        return distances.sum().item()
    
    @staticmethod
    def compute_max_velocity(trajectory: torch.Tensor) -> float:
        """
        Compute maximum velocity magnitude in trajectory.
        
        Args:
            trajectory: (B, T, D) or (T, D) trajectory
        
        Returns:
            max_vel: Maximum velocity magnitude
        """
        if trajectory.dim() == 3:
            trajectory = trajectory[0]
        
        diffs = torch.diff(trajectory, dim=0)
        velocities = torch.norm(diffs, dim=-1)
        return velocities.max().item()
    
    @staticmethod
    def compute_max_acceleration(trajectory: torch.Tensor) -> float:
        """
        Compute maximum acceleration magnitude in trajectory.
        
        Args:
            trajectory: (B, T, D) or (T, D) trajectory
        
        Returns:
            max_accel: Maximum acceleration magnitude
        """
        if trajectory.dim() == 3:
            trajectory = trajectory[0]
        
        T, D = trajectory.shape
        if T < 3:
            return 0.0
        
        # Compute velocities
        vel = torch.diff(trajectory, dim=0)
        
        # Compute accelerations
        accel = torch.diff(vel, dim=0)
        accel_mag = torch.norm(accel, dim=-1)
        
        return accel_mag.max().item()
    
    @staticmethod
    def compute_min_distance_to_obstacles(
        trajectory: torch.Tensor,
        obstacle_centers: torch.Tensor,
        obstacle_radii: torch.Tensor,
    ) -> float:
        """
        Compute minimum distance to any obstacle.
        
        Args:
            trajectory: (B, T, D) or (T, D) trajectory
            obstacle_centers: (N, D) obstacle centers
            obstacle_radii: (N,) obstacle radii
        
        Returns:
            min_dist: Minimum distance to nearest obstacle surface
                     (negative = collision)
        """
        if trajectory.dim() == 3:
            trajectory = trajectory[0]  # (T, D)
        
        T, D = trajectory.shape
        N = obstacle_centers.shape[0]
        
        min_dist = float('inf')
        
        for t in range(T):
            pos = trajectory[t]  # (D,)
            
            # Distance to each obstacle
            for n in range(N):
                center = obstacle_centers[n]
                radius = obstacle_radii[n].item()
                
                dist_to_center = torch.norm(pos - center).item()
                dist_to_surface = dist_to_center - radius
                
                min_dist = min(min_dist, dist_to_surface)
        
        return min_dist if min_dist < float('inf') else 0.0


def aggregate_batch_metrics(
    metrics_list,
) -> Dict[str, float]:
    """
    Aggregate metrics across a batch of trajectories.
    
    Args:
        metrics_list: List of metric dictionaries
    
    Returns:
        aggregated: Dictionary with mean and std for each metric
    """
    if not metrics_list:
        return {}
    
    # Group by metric name
    metric_names = metrics_list[0].keys()
    aggregated = {}
    
    for name in metric_names:
        values = [m[name] for m in metrics_list]
        values_tensor = torch.tensor(values)
        
        aggregated[f'{name}_mean'] = values_tensor.mean().item()
        aggregated[f'{name}_std'] = values_tensor.std().item()
        aggregated[f'{name}_min'] = values_tensor.min().item()
        aggregated[f'{name}_max'] = values_tensor.max().item()
    
    return aggregated


if __name__ == '__main__':
    # Test ImfMetricsTracker
    batch_size = 4
    seq_len = 10
    state_dim = 28
    
    u_pred = torch.randn(batch_size, seq_len, state_dim)
    u_target = torch.randn(batch_size, seq_len, state_dim)
    v_pred = torch.randn(batch_size, seq_len, state_dim)
    v_target = torch.randn(batch_size, seq_len, state_dim)
    target_vel = torch.randn(batch_size, seq_len, state_dim)
    
    tracker = ImfMetricsTracker()
    
    u_err = tracker.compute_u_error(u_pred, u_target)
    v_err = tracker.compute_v_error(v_pred, v_target)
    comb_err = tracker.compute_combined_error(u_pred, v_pred, target_vel)
    
    print(f"u_error: {u_err:.4f}")
    print(f"v_error: {v_err:.4f}")
    print(f"combined_error: {comb_err:.4f}")
    
    # Test smoothness
    traj = torch.randn(batch_size, seq_len, state_dim).cumsum(dim=1)
    smoothness = tracker.compute_smoothness(traj)
    print(f"smoothness: {smoothness:.4f}")
    
    # Test decomposition
    u_contrib, v_contrib = tracker.compute_decomposition(u_pred, v_pred)
    print(f"u_contribution: {u_contrib:.4f}, v_contribution: {v_contrib:.4f}")
    
    # Test alignment
    align = tracker.compute_velocity_alignment(u_pred, v_pred)
    print(f"alignment: {align:.4f}")
    
    # Test summary
    summary = tracker.get_summary()
    print(f"Summary: {summary}")
    
    # Test trajectory quality metrics
    traj_qual = TrajectoryQualityMetrics()
    path_len = traj_qual.compute_path_length(traj)
    max_vel = traj_qual.compute_max_velocity(traj)
    max_accel = traj_qual.compute_max_acceleration(traj)
    print(f"path_length: {path_len:.4f}, max_vel: {max_vel:.4f}, max_accel: {max_accel:.4f}")
