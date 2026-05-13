"""
Improved Mean Flows Trajectory Sampler

High-level inference API for trajectory generation with dual-velocity fields.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from .models.imf_velocity import DualVelocityField, TimeConditionedDualVelocity
from .sampling.imf_ode_solvers import ImfODESolver, sample_trajectory_imf


class ImfTrajectorySampler(nn.Module):
    """
    High-level trajectory sampler combining dual-velocity prediction and ODE integration.
    
    Provides simplified interface for trajectory generation:
    - sample_single_step(): Fast inference (NFE=1)
    - sample_dual_step(): Higher quality (NFE=2)
    - sample_with_guidance(): Constraint-guided sampling
    
    Args:
        velocity_model: DualVelocityField or TimeConditionedDualVelocity
        num_steps: Number of ODE integration steps
        solver_type: 'euler', 'rk4', or 'dopri5' (default: 'dopri5')
    """
    
    def __init__(
        self,
        velocity_model: nn.Module,
        num_steps: int = 10,
        solver_type: str = 'dopri5',
        state_dim: int = 28,
    ):
        super().__init__()
        self.velocity_model = velocity_model
        self.num_steps = num_steps
        self.solver_type = solver_type
        self.state_dim = state_dim
        
        # Create ODE solver
        self.ode_solver = ImfODESolver(
            num_steps=num_steps,
            solver_type=solver_type,
        )
    
    def sample_single_step(
        self,
        z_init: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fast single-step sampling (NFE=1).
        
        Directly integrates combined velocity: z_final = z_0 + ∫(u + v) dt
        
        Args:
            z_init: (B,) or (B, state_dim) initial state
            t: (B,) optional time conditioning
        
        Returns:
            z_final: (B, state_dim) final trajectory point
        """
        if z_init.dim() == 1:
            z_init = z_init.unsqueeze(-1)
        
        B = z_init.shape[0]
        device = z_init.device
        
        if t is None:
            t = torch.zeros(B, device=device)
        
        # Velocity prediction
        with torch.no_grad():
            u, v = self.velocity_model(z_init, t)
        
        v_combined = u + v
        
        # ODE integration
        z_final = self.ode_solver.sample_single_step(
            z_init=z_init,
            velocity_func=lambda _, x: v_combined,
            t_span=(0.0, 1.0),
        )
        
        return z_final
    
    def sample_dual_step(
        self,
        z_init: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        t_split: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Higher-quality dual-step sampling (NFE=2).
        
        Phase 1: Integrate u (global direction)
        Phase 2: Integrate v (local refinement)
        
        Args:
            z_init: (B, state_dim) initial state
            t: (B,) optional time
            t_split: When to switch from u to v phase (default: 0.5)
        
        Returns:
            z_u: Intermediate point after u integration
            z_v: Final point after v integration
            z_combined: For reference, direct combined integration
        """
        if z_init.dim() == 1:
            z_init = z_init.unsqueeze(-1)
        
        B = z_init.shape[0]
        device = z_init.device
        
        if t is None:
            t = torch.zeros(B, device=device)
        
        with torch.no_grad():
            u, _ = self.velocity_model(z_init, t)
        
        # Phase 1: Integrate u only (0 → t_split)
        z_u = self.ode_solver.solve_manual(
            z_init=z_init,
            velocity_func=lambda _, x: u,
            t0=0.0,
            t1=t_split,
            method='rk4',
        )
        
        # Phase 2: Integrate v (t_split → 1)
        with torch.no_grad():
            _, v = self.velocity_model(z_u, t)
        
        z_v = self.ode_solver.solve_manual(
            z_init=z_u,
            velocity_func=lambda _, x: v,
            t0=t_split,
            t1=1.0,
            method='rk4',
        )
        
        # Reference: combined integration
        with torch.no_grad():
            u_ref, v_ref = self.velocity_model(z_init, t)
        
        z_combined = self.ode_solver.solve_manual(
            z_init=z_init,
            velocity_func=lambda _, x: u_ref + v_ref,
            t0=0.0,
            t1=1.0,
            method='rk4',
        )
        
        return z_u, z_v, z_combined
    
    def sample_multi_step(
        self,
        z_init: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        num_phases: int = 4,
    ) -> Dict[str, torch.Tensor]:
        """
        Multi-step sampling with alternating u/v phases.
        
        Useful for analyzing contribution of each phase.
        
        Args:
            z_init: (B, state_dim) initial state
            t: (B,) optional time
            num_phases: Number of alternating phases (default: 4)
        
        Returns:
            results: Dictionary with 'phase_0', 'phase_1', ..., 'final'
        """
        results = {'phase_0': z_init}
        z_current = z_init
        
        device = z_init.device
        B = z_init.shape[0]
        
        if t is None:
            t = torch.zeros(B, device=device)
        
        phase_duration = 1.0 / num_phases
        
        for phase_idx in range(num_phases):
            t_start = phase_idx * phase_duration
            t_end = (phase_idx + 1) * phase_duration
            
            with torch.no_grad():
                u, v = self.velocity_model(z_current, t)
            
            # Alternate between u and v
            vel = u if phase_idx % 2 == 0 else v
            
            z_next = self.ode_solver.solve_manual(
                z_init=z_current,
                velocity_func=lambda _, x: vel,
                t0=t_start,
                t1=t_end,
                method='rk4',
            )
            
            results[f'phase_{phase_idx + 1}'] = z_next
            z_current = z_next
        
        return results


class ConditionalImfSampler(ImfTrajectorySampler):
    """
    Trajectory sampler with conditional inputs (goal, constraints, context).
    
    Extends ImfTrajectorySampler with:
    - Goal conditioning: Steer toward target state
    - Constraint guidance: Avoid collisions, enforce smoothness
    - Context modulation: Adapt generation based on environment
    
    Args:
        velocity_model: Dual-velocity model
        num_steps: ODE integration steps
        solver_type: Solver type
        goal_weight: Strength of goal conditioning
    """
    
    def __init__(
        self,
        velocity_model: nn.Module,
        num_steps: int = 10,
        solver_type: str = 'dopri5',
        state_dim: int = 28,
        goal_weight: float = 0.1,
    ):
        super().__init__(velocity_model, num_steps, solver_type, state_dim)
        self.goal_weight = goal_weight
    
    def sample_toward_goal(
        self,
        z_init: torch.Tensor,
        z_goal: torch.Tensor,
        t: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Sample trajectory steered toward goal state.
        
        Args:
            z_init: (B, state_dim) initial state
            z_goal: (B, state_dim) goal state
            t: (B,) optional time
        
        Returns:
            z_final: (B, state_dim) trajectory point
        """
        if z_init.dim() == 1:
            z_init = z_init.unsqueeze(-1)
        if z_goal.dim() == 1:
            z_goal = z_goal.unsqueeze(-1)
        
        B = z_init.shape[0]
        device = z_init.device
        
        if t is None:
            t = torch.zeros(B, device=device)
        
        def combined_velocity(_, z):
            """Velocity with goal guidance."""
            with torch.no_grad():
                u, v = self.velocity_model(z, t)
            
            # Goal direction
            goal_dir = (z_goal - z) / (torch.norm(z_goal - z, dim=-1, keepdim=True) + 1e-6)
            
            # Blend with goal direction
            v_combined = u + v + self.goal_weight * goal_dir
            
            return v_combined
        
        # ODE integration with goal guidance
        z_final = self.ode_solver.solve_manual(
            z_init=z_init,
            velocity_func=combined_velocity,
            t0=0.0,
            t1=1.0,
            method='rk4',
        )
        
        return z_final
    
    def sample_avoiding_obstacles(
        self,
        z_init: torch.Tensor,
        obstacle_centers: torch.Tensor,
        obstacle_radii: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        avoidance_weight: float = 0.05,
    ) -> torch.Tensor:
        """
        Sample trajectory avoiding obstacles.
        
        Args:
            z_init: (B, state_dim) initial state
            obstacle_centers: (N, state_dim) obstacle centers
            obstacle_radii: (N,) obstacle radii
            t: (B,) optional time
            avoidance_weight: Strength of avoidance (higher = stronger)
        
        Returns:
            z_final: (B, state_dim) trajectory point avoiding obstacles
        """
        if z_init.dim() == 1:
            z_init = z_init.unsqueeze(-1)
        
        B = z_init.shape[0]
        device = z_init.device
        
        if t is None:
            t = torch.zeros(B, device=device)
        
        def repulsive_velocity(_, z):
            """Velocity with obstacle repulsion."""
            with torch.no_grad():
                u, v = self.velocity_model(z, t)
            
            # Compute repulsion from obstacles
            repulsion = torch.zeros_like(z)
            
            for i in range(obstacle_centers.shape[0]):
                center = obstacle_centers[i]
                radius = obstacle_radii[i]
                
                # Vector from obstacle to position
                dist_vec = z - center  # (B, state_dim)
                dist = torch.norm(dist_vec, dim=-1, keepdim=True) + 1e-6
                
                # Repulsion magnitude (increases as we approach)
                repulsion_mag = torch.relu(radius - dist) / (dist + 1e-6)
                
                # Repulsion direction
                repulsion += repulsion_mag * (dist_vec / dist)
            
            v_combined = u + v + avoidance_weight * repulsion
            
            return v_combined
        
        # ODE integration
        z_final = self.ode_solver.solve_manual(
            z_init=z_init,
            velocity_func=repulsive_velocity,
            t0=0.0,
            t1=1.0,
            method='rk4',
        )
        
        return z_final


if __name__ == '__main__':
    # Test ImfTrajectorySampler
    state_dim = 28
    batch_size = 4
    
    # Simple dual-velocity model for testing
    velocity_model = DualVelocityField(
        state_dim=state_dim,
        hidden_dim=64,
        use_jvp=False,
    )
    
    sampler = ImfTrajectorySampler(
        velocity_model=velocity_model,
        num_steps=5,
        solver_type='euler',
        state_dim=state_dim,
    )
    
    # Test single-step sampling
    z_init = torch.randn(batch_size, state_dim)
    z_final = sampler.sample_single_step(z_init)
    print(f"Single-step: {z_init.shape} → {z_final.shape}")
    
    # Test dual-step sampling
    z_u, z_v, z_combined = sampler.sample_dual_step(z_init)
    print(f"Dual-step: u={z_u.shape}, v={z_v.shape}, combined={z_combined.shape}")
    
    # Test multi-step sampling
    results = sampler.sample_multi_step(z_init, num_phases=4)
    print(f"Multi-step phases: {list(results.keys())}")
    
    # Test conditional sampler
    cond_sampler = ConditionalImfSampler(
        velocity_model=velocity_model,
        num_steps=5,
        state_dim=state_dim,
        goal_weight=0.1,
    )
    
    z_goal = torch.randn(batch_size, state_dim)
    z_guided = cond_sampler.sample_toward_goal(z_init, z_goal)
    print(f"Goal-guided: {z_guided.shape}")
    
    # Test obstacle avoidance
    obstacle_centers = torch.randn(3, state_dim)
    obstacle_radii = torch.tensor([0.5, 0.5, 0.5])
    
    z_safe = cond_sampler.sample_avoiding_obstacles(
        z_init, obstacle_centers, obstacle_radii
    )
    print(f"Obstacle-avoiding: {z_safe.shape}")
