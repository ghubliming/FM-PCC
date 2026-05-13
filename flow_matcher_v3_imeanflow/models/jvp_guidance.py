"""
Jacobian-Vector Product (JVP) Guidance Module for Constraint-Aware Trajectory Generation

Computes gradient-based guidance for satisfying trajectory constraints.
Can incorporate collision avoidance, smoothness penalties, and task-specific constraints.

The JVP provides second-order derivative information to refine velocity predictions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Tuple


class JVPGuidance(nn.Module):
    """
    Jacobian-Vector Product guidance for constraint-aware trajectory refinement.
    
    Computes the product of constraint gradient (Jacobian) with velocity prediction (vector)
    to incorporate constraint satisfaction into the velocity field.
    
    Args:
        state_dim: Dimension of trajectory state
        constraint_types: List of constraints to include ('collision', 'smoothness', 'task')
        constraint_weight: Scale factor for JVP guidance (default: 1.0)
        enable_grad: Whether to compute gradients (default: True)
    """
    
    def __init__(
        self,
        state_dim: int,
        constraint_types: Optional[list] = None,
        constraint_weight: float = 1.0,
        enable_grad: bool = True,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.constraint_types = constraint_types or []
        self.constraint_weight = constraint_weight
        self.enable_grad = enable_grad
    
    def compute_jacobian(
        self,
        constraint_fn: Callable,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Jacobian matrix of constraint function w.r.t. state.
        
        Args:
            constraint_fn: Callable that maps state to scalar constraint value
            x: State tensor (B, state_dim)
        
        Returns:
            jacobian: Jacobian matrix (B, 1, state_dim)
        """
        x_clone = x.clone().detach().requires_grad_(True)
        
        # Compute constraint
        c = constraint_fn(x_clone)  # (B,) or (B, 1)
        
        # Compute jacobian
        jacobian = torch.autograd.grad(
            outputs=c.sum() if c.dim() > 1 else c,
            inputs=x_clone,
            create_graph=True,
            retain_graph=True,
        )[0]  # (B, state_dim)
        
        if jacobian.dim() == 1:
            jacobian = jacobian.unsqueeze(0)  # (1, state_dim)
        
        return jacobian
    
    def compute_jvp(
        self,
        jacobian: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute Jacobian-Vector Product: J @ v
        
        Args:
            jacobian: Jacobian matrix (B, state_dim) or (B, 1, state_dim)
            v: Velocity vector (B, state_dim)
        
        Returns:
            jvp: JVP result (B, state_dim)
        """
        if jacobian.dim() == 3:
            jacobian = jacobian.squeeze(1)  # (B, state_dim)
        
        # JVP: (B, state_dim) @ (B, state_dim) -> (B, state_dim)
        # Using element-wise multiplication then sum along state dim
        jvp = jacobian * v  # Element-wise
        
        return jvp
    
    def collision_free_constraint(
        self,
        x: torch.Tensor,
        obstacle_centers: torch.Tensor,
        obstacle_radii: torch.Tensor,
    ) -> torch.Tensor:
        """
        Collision avoidance constraint: distance to obstacles.
        
        Formula: constraint = min_i ||x - o_i|| - r_i
        Negative values indicate collision.
        
        Args:
            x: State (B, state_dim)
            obstacle_centers: Obstacle positions (num_obs, state_dim)
            obstacle_radii: Obstacle radii (num_obs,)
        
        Returns:
            constraint: Constraint value (B,)
        """
        # Compute distance to all obstacles
        # x: (B, state_dim) -> (B, 1, state_dim)
        # obstacles: (num_obs, state_dim)
        distances = torch.cdist(x, obstacle_centers)  # (B, num_obs)
        
        # Radii: (num_obs,) -> (1, num_obs)
        radii = obstacle_radii.unsqueeze(0)
        
        # Minimum signed distance
        min_dist = (distances - radii).min(dim=1)[0]  # (B,)
        
        return min_dist
    
    def smoothness_constraint(
        self,
        v: torch.Tensor,
        max_accel: float = 10.0,
    ) -> torch.Tensor:
        """
        Smoothness constraint: limit acceleration magnitude.
        
        Args:
            v: Velocity (B, state_dim) or (B, T, state_dim)
            max_accel: Maximum allowed acceleration
        
        Returns:
            constraint: Acceleration constraint (B,)
        """
        if v.dim() == 3:
            # Compute acceleration from velocity sequence
            a = torch.diff(v, dim=1)  # (B, T-1, state_dim)
            a_norm = torch.linalg.norm(a, dim=-1).max(dim=1)[0]  # (B,)
        else:
            # For single velocity, constraint is just norm
            a_norm = torch.linalg.norm(v, dim=-1)  # (B,)
        
        # Squared constraint: a² - max_a²
        constraint = max_accel ** 2 - a_norm ** 2  # (B,)
        
        return constraint
    
    def forward(
        self,
        x: torch.Tensor,
        v_base: torch.Tensor,
        constraint_fn: Optional[Callable] = None,
        constraint_weight: Optional[float] = None,
    ) -> torch.Tensor:
        """
        Compute JVP-guided refinement to velocity.
        
        Args:
            x: State (B, state_dim)
            v_base: Base velocity (B, state_dim)
            constraint_fn: Callable mapping x -> constraint value
            constraint_weight: Weight for JVP contribution (default: self.constraint_weight)
        
        Returns:
            v_jvp: JVP guidance term (B, state_dim)
        """
        if constraint_fn is None:
            # No constraint: return zero refinement
            return torch.zeros_like(v_base)
        
        if constraint_weight is None:
            constraint_weight = self.constraint_weight
        
        # Compute Jacobian of constraint w.r.t. state
        jacobian = self.compute_jacobian(constraint_fn, x)  # (B, state_dim)
        
        # Compute JVP
        jvp = self.compute_jvp(jacobian, v_base)  # (B, state_dim)
        
        # Scale and return
        v_jvp = constraint_weight * jvp
        
        return v_jvp
    
    def forward_with_explicit_constraint(
        self,
        x: torch.Tensor,
        v_base: torch.Tensor,
        constraint_type: str = 'smoothness',
        **constraint_kwargs,
    ) -> torch.Tensor:
        """
        Compute JVP with built-in constraint function.
        
        Args:
            x: State (B, state_dim)
            v_base: Base velocity (B, state_dim)
            constraint_type: 'smoothness', 'collision_free', or 'task'
            **constraint_kwargs: Additional arguments for constraint function
        
        Returns:
            v_jvp: JVP guidance term (B, state_dim)
        """
        if constraint_type == 'smoothness':
            max_accel = constraint_kwargs.get('max_accel', 10.0)
            constraint_fn = lambda v: self.smoothness_constraint(v, max_accel)
            return self.forward(x, v_base, constraint_fn)
        
        elif constraint_type == 'collision_free':
            obstacle_centers = constraint_kwargs.get('obstacle_centers', None)
            obstacle_radii = constraint_kwargs.get('obstacle_radii', None)
            
            if obstacle_centers is None or obstacle_radii is None:
                return torch.zeros_like(v_base)
            
            constraint_fn = lambda state: self.collision_free_constraint(
                state, obstacle_centers, obstacle_radii
            )
            return self.forward(x, v_base, constraint_fn)
        
        else:
            # Unknown constraint type
            return torch.zeros_like(v_base)


class SoftConstraintModule(nn.Module):
    """
    Soft constraint module that learns to weight multiple constraints.
    
    Uses a neural network to adaptively weight different constraint contributions.
    """
    
    def __init__(
        self,
        state_dim: int,
        num_constraints: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.num_constraints = num_constraints
        
        # Neural network to predict constraint weights
        self.weight_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_constraints),
            nn.Softmax(dim=-1),  # Normalized weights
        )
    
    def forward(
        self,
        x: torch.Tensor,
        constraint_values: list,  # List of (B, state_dim) constraint guidance
    ) -> torch.Tensor:
        """
        Combine multiple constraint guidance terms with learned weights.
        
        Args:
            x: State (B, state_dim)
            constraint_values: List of constraint guidance terms
        
        Returns:
            combined: Weighted combination of constraints
        """
        # Predict weights
        weights = self.weight_network(x)  # (B, num_constraints)
        
        # Stack constraint values
        stacked = torch.stack(constraint_values, dim=-1)  # (B, state_dim, num_constraints)
        
        # Weighted combination
        combined = torch.einsum('bsd,bc->bsd', stacked, weights)  # (B, state_dim)
        
        return combined


if __name__ == '__main__':
    # Test JVP guidance
    batch_size = 4
    state_dim = 28
    
    jvp_module = JVPGuidance(state_dim=state_dim)
    
    x = torch.randn(batch_size, state_dim)
    v_base = torch.randn(batch_size, state_dim)
    
    # Define a simple quadratic constraint
    def constraint_fn(state):
        return (state ** 2).sum(dim=-1)  # Sum of squares
    
    # Compute JVP guidance
    v_jvp = jvp_module(x, v_base, constraint_fn)
    print(f"JVP guidance shape: {v_jvp.shape}")
    
    # Test smoothness constraint
    v_guidance = jvp_module.forward_with_explicit_constraint(
        x, v_base, constraint_type='smoothness', max_accel=10.0
    )
    print(f"Smoothness guidance shape: {v_guidance.shape}")
    
    # Test soft constraint module
    soft_constraint = SoftConstraintModule(state_dim=state_dim, num_constraints=3)
    guidance_list = [torch.randn(batch_size, state_dim) for _ in range(3)]
    combined = soft_constraint(x, guidance_list)
    print(f"Soft constraint combined shape: {combined.shape}")
