"""
Improved Mean Flows ODE Solvers

Implements single-step (NFE=1) and dual-step (NFE=2) ODE integration
for fast trajectory generation with improved quality.

Based on "Improved Mean Flows: On the Challenges of Fastforward Generative Models"
"""

import torch
import torch.nn as nn
from typing import Callable, Optional, Tuple, List
from scipy.integrate import odeint
import numpy as np

try:
    from torchdiffeq import odeint as torch_odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


class ImfODESolver(nn.Module):
    """
    ODE solver for Improved Mean Flows dual-velocity trajectory generation.
    
    Supports single-step (NFE=1) and dual-step (NFE=2) integration,
    with multiple backend options (euler, rk4, dopri5).
    
    Args:
        solver_backend: 'manual' (Euler/RK4), 'torchdiffeq' (adaptive)
        solver_method: 'euler', 'rk4' (manual) or 'dopri5', 'adams' (torchdiffeq)
        rtol: Relative tolerance for adaptive solvers
        atol: Absolute tolerance for adaptive solvers
        max_steps: Maximum steps for adaptive solvers
    """
    
    def __init__(
        self,
        solver_backend: str = 'manual',
        solver_method: str = 'euler',
        rtol: float = 1e-5,
        atol: float = 1e-6,
        max_steps: int = 100,
    ):
        super().__init__()
        self.solver_backend = solver_backend
        self.solver_method = solver_method
        self.rtol = rtol
        self.atol = atol
        self.max_steps = max_steps
        
        if solver_backend == 'torchdiffeq' and not HAS_TORCHDIFFEQ:
            print("Warning: torchdiffeq not available, falling back to manual solver")
            self.solver_backend = 'manual'
    
    def euler_step(
        self,
        velocity_fn: Callable,
        x: torch.Tensor,
        t: float,
        dt: float,
    ) -> torch.Tensor:
        """Single Euler step."""
        v = velocity_fn(t, x)
        return x + dt * v
    
    def rk4_step(
        self,
        velocity_fn: Callable,
        x: torch.Tensor,
        t: float,
        dt: float,
    ) -> torch.Tensor:
        """Single RK4 step."""
        k1 = velocity_fn(t, x)
        k2 = velocity_fn(t + dt/2, x + (dt/2) * k1)
        k3 = velocity_fn(t + dt/2, x + (dt/2) * k2)
        k4 = velocity_fn(t + dt, x + dt * k3)
        
        return x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def solve_manual(
        self,
        velocity_fn: Callable,
        x0: torch.Tensor,
        t_span: Tuple[float, float],
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Manual ODE integration using Euler or RK4.
        
        Args:
            velocity_fn: Callable (t, x) -> dx/dt
            x0: Initial state (B, state_dim)
            t_span: (t_start, t_end)
            num_steps: Number of integration steps
        
        Returns:
            x_final: Final state after integration
        """
        t_start, t_end = t_span
        dt = (t_end - t_start) / num_steps
        
        x = x0.clone()
        t = t_start
        
        for _ in range(num_steps):
            if self.solver_method == 'euler':
                x = self.euler_step(velocity_fn, x, t, dt)
            elif self.solver_method == 'rk4':
                x = self.rk4_step(velocity_fn, x, t, dt)
            else:
                raise ValueError(f"Unknown solver method: {self.solver_method}")
            
            t += dt
        
        return x
    
    def solve_torchdiffeq(
        self,
        velocity_fn: Callable,
        x0: torch.Tensor,
        t_span: Tuple[float, float],
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Adaptive ODE integration using torchdiffeq.
        
        Args:
            velocity_fn: Callable (t, x) -> dx/dt
            x0: Initial state
            t_span: (t_start, t_end)
            num_steps: Number of steps (used for fixed grid if method is fixed)
        
        Returns:
            x_final: Final state after integration
        """
        t_start, t_end = t_span
        t_eval = torch.linspace(t_start, t_end, num_steps + 1, device=x0.device)
        
        # Wrapper for torchdiffeq (expects (t, x) but processes batch)
        def velocity_wrapper(t, y):
            # y is (B, state_dim), t is scalar
            return velocity_fn(float(t), y)
        
        try:
            solution = torch_odeint(
                velocity_wrapper,
                x0,
                t_eval,
                method=self.solver_method,
                rtol=self.rtol,
                atol=self.atol,
            )
            # solution shape: (num_steps+1, B, state_dim)
            return solution[-1]
        except Exception as e:
            print(f"torchdiffeq failed: {e}, falling back to manual")
            return self.solve_manual(velocity_fn, x0, t_span, num_steps)
    
    def solve(
        self,
        velocity_fn: Callable,
        x0: torch.Tensor,
        t_span: Tuple[float, float],
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Solve ODE with selected backend.
        
        Args:
            velocity_fn: Callable (t, x) -> dx/dt
            x0: Initial state
            t_span: (t_start, t_end)
            num_steps: Number of integration steps
        
        Returns:
            x_final: Final state
        """
        if self.solver_backend == 'torchdiffeq':
            return self.solve_torchdiffeq(velocity_fn, x0, t_span, num_steps)
        else:
            return self.solve_manual(velocity_fn, x0, t_span, num_steps)
    
    def sample_single_step(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        cond: torch.Tensor,
        t_span: Tuple[float, float] = (0.0, 1.0),
        num_steps: int = 10,
    ) -> torch.Tensor:
        """
        Single-step sampling: integrate u + v in one ODE solve.
        
        NFE=1 (one function evaluation): z_final = z_0 + ∫₀¹ (u + v) dt
        
        Args:
            model: Dual-velocity network that returns (u, v)
            x0: Initial noise (B, state_dim)
            cond: Conditioning information (B, cond_dim)
            t_span: Integration time span
            num_steps: Steps for ODE integration
        
        Returns:
            z_final: Final trajectory
        """
        def combined_velocity(t, x):
            with torch.no_grad():
                u, v = model(x)  # Both (B, state_dim)
                return u + v
        
        x_final = self.solve(combined_velocity, x0, t_span, num_steps)
        return x_final
    
    def sample_dual_step(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        cond: torch.Tensor,
        t_span: Tuple[float, float] = (0.0, 1.0),
        num_steps: int = 10,
        t_split: float = 0.5,
    ) -> torch.Tensor:
        """
        Dual-step sampling: refine trajectory in two phases.
        
        NFE=2: 
        - Step 1: z_mid = z_0 + ∫₀^t_split u dt  (global direction)
        - Step 2: z_final = z_mid + ∫_{t_split}^1 v dt  (local refinement)
        
        Args:
            model: Dual-velocity network
            x0: Initial noise
            cond: Conditioning information
            t_span: Overall time span
            num_steps: Steps per phase
            t_split: Time to switch from u to v
        
        Returns:
            z_final: Final trajectory
        """
        t_start, t_end = t_span
        
        # Phase 1: Integrate u (average velocity) to midpoint
        def u_velocity(t, x):
            with torch.no_grad():
                u, _ = model(x)
                return u
        
        x_mid = self.solve(u_velocity, x0, (t_start, t_split), num_steps)
        
        # Phase 2: Integrate v (instantaneous velocity) from midpoint to end
        def v_velocity(t, x):
            with torch.no_grad():
                _, v = model(x)
                return v
        
        x_final = self.solve(v_velocity, x_mid, (t_split, t_end), num_steps)
        
        return x_final
    
    def sample_multi_step(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        cond: torch.Tensor,
        t_span: Tuple[float, float] = (0.0, 1.0),
        num_steps: int = 10,
        num_phases: int = 4,
    ) -> torch.Tensor:
        """
        Multi-step sampling: alternate between u and v.
        
        Args:
            model: Dual-velocity network
            x0: Initial noise
            cond: Conditioning
            t_span: Overall time span
            num_steps: Steps per phase
            num_phases: Number of alternating phases
        
        Returns:
            z_final: Final trajectory
        """
        t_start, t_end = t_span
        t_split = np.linspace(t_start, t_end, num_phases + 1)
        
        x = x0.clone()
        
        for i in range(num_phases):
            if i % 2 == 0:
                # Even phase: use u
                def velocity(t, state):
                    with torch.no_grad():
                        u, _ = model(state)
                        return u
            else:
                # Odd phase: use v
                def velocity(t, state):
                    with torch.no_grad():
                        _, v = model(state)
                        return v
            
            x = self.solve(velocity, x, (t_split[i], t_split[i+1]), num_steps)
        
        return x


def sample_trajectory_imf(
    model: nn.Module,
    x0: torch.Tensor,
    cond: torch.Tensor,
    nfe: int = 1,
    solver: str = 'euler',
    num_steps: int = 10,
    t_span: Tuple[float, float] = (0.0, 1.0),
    t_split: float = 0.5,
) -> torch.Tensor:
    """
    High-level API for iMF trajectory sampling.
    
    Args:
        model: Dual-velocity network
        x0: Initial noise (B, state_dim)
        cond: Conditioning (B, cond_dim)
        nfe: Number of function evaluations (1 or 2)
        solver: 'euler', 'rk4', 'dopri5'
        num_steps: Steps for ODE integration
        t_span: Time integration span
        t_split: Split time for dual-step sampling
    
    Returns:
        trajectory: Generated trajectory
    """
    # Select backend and method
    if solver == 'dopri5':
        backend = 'torchdiffeq'
        method = 'dopri5'
    else:
        backend = 'manual'
        method = solver
    
    ode_solver = ImfODESolver(solver_backend=backend, solver_method=method)
    
    if nfe == 1:
        return ode_solver.sample_single_step(model, x0, cond, t_span, num_steps)
    elif nfe == 2:
        return ode_solver.sample_dual_step(model, x0, cond, t_span, num_steps, t_split)
    else:
        raise ValueError(f"nfe must be 1 or 2, got {nfe}")


if __name__ == '__main__':
    # Test ODE solvers
    batch_size = 4
    state_dim = 28
    
    # Create dummy dual-velocity model
    class DummyDualVelocity(nn.Module):
        def forward(self, x):
            u = -x * 0.1  # Decay
            v = torch.randn_like(x) * 0.01  # Small noise
            return u, v
    
    model = DummyDualVelocity()
    x0 = torch.randn(batch_size, state_dim)
    cond = torch.randn(batch_size, 1)
    
    # Test single-step
    solver = ImfODESolver(solver_backend='manual', solver_method='euler')
    result1 = solver.sample_single_step(model, x0, cond, num_steps=5)
    print(f"Single-step result: {result1.shape}")
    
    # Test dual-step
    result2 = solver.sample_dual_step(model, x0, cond, num_steps=5)
    print(f"Dual-step result: {result2.shape}")
    
    # Test high-level API
    result3 = sample_trajectory_imf(model, x0, cond, nfe=1, num_steps=5)
    print(f"API single-step: {result3.shape}")
    
    result4 = sample_trajectory_imf(model, x0, cond, nfe=2, num_steps=5)
    print(f"API dual-step: {result4.shape}")
