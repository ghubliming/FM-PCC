"""
ODE Solvers with Drift Guidance for FM-D

Wraps standard ODE integrators (RK45, Dopri5, Euler) with optional drift loss guidance.
Drift guidance modulates the velocity field during integration to steer trajectories
toward the learned expert distribution.
"""

import torch
from typing import Callable, Optional, Tuple, List
try:
    from torchdiffeq import odeint as torchdiffeq_odeint
except ImportError:
    torchdiffeq_odeint = None


class DriftAugmentedVelocityField:
    """
    Wraps a velocity field function to include drift loss guidance.
    
    Standard: v(x, t) = model_velocity(x, t)
    With drift: v(x, t) = model_velocity(x, t) + lambda * grad_drift_loss(x)
    """
    
    def __init__(
        self,
        velocity_fn: Callable,
        drift_loss_fn: Optional[Callable] = None,
        drift_weight: float = 0.1,
        drift_clip: float = 1.0,
    ):
        """
        Args:
            velocity_fn: Function returning velocity field v(x, t) or v(x, cond, t)
            drift_loss_fn: Function returning drift loss gradient grad_loss(x)
            drift_weight: Weight of drift guidance lambda (0 = no drift)
            drift_clip: Clip drift gradient norm to prevent instability
        """
        self.velocity_fn = velocity_fn
        self.drift_loss_fn = drift_loss_fn
        self.drift_weight = float(drift_weight)
        self.drift_clip = float(drift_clip)

    def __call__(self, t: torch.Tensor, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Compute augmented velocity field at state x, time t.
        
        Args:
            t: Time step (scalar tensor)
            x: State (B, trajectory_dim)
            **kwargs: Additional arguments for velocity_fn (e.g., cond, returns)
            
        Returns:
            v(t, x) = velocity + drift_guidance
        """
        # Compute base velocity
        velocity = self.velocity_fn(t, x, **kwargs)
        
        # Add drift guidance if enabled
        if self.drift_loss_fn is not None and self.drift_weight > 0:
            drift_grad = self.drift_loss_fn(x)
            
            # Clip drift gradient to prevent divergence
            drift_norm = torch.norm(drift_grad, p=2, dim=-1, keepdim=True).clamp(min=1e-8)
            drift_grad_clipped = drift_grad * torch.clamp(drift_norm, max=self.drift_clip) / drift_norm
            
            velocity = velocity + self.drift_weight * drift_grad_clipped
        
        return velocity


class DriftODESolver:
    """
    ODE solver with drift guidance for FM-D trajectories.
    Supports multiple backend solvers with unified interface.
    """
    
    def __init__(
        self,
        solver_method: str = "euler",
        solver_backend: str = "legacy_euler",
        rtol: float = 1e-5,
        atol: float = 1e-6,
        step_size: Optional[float] = None,
    ):
        """
        Args:
            solver_method: "euler" | "rk4" | "rk45" | "dopri5" | "adams"
            solver_backend: "legacy_euler" | "torchdiffeq"
            rtol: Relative tolerance for adaptive solvers
            atol: Absolute tolerance for adaptive solvers
            step_size: Fixed step size for non-adaptive solvers (if None, auto)
        """
        self.solver_method = solver_method
        self.solver_backend = solver_backend
        self.rtol = rtol
        self.atol = atol
        self.step_size = step_size
        
        if solver_backend == "torchdiffeq" and torchdiffeq_odeint is None:
            print("Warning: torchdiffeq not available, falling back to legacy_euler")
            self.solver_backend = "legacy_euler"

    def solve(
        self,
        velocity_fn: Callable,
        x0: torch.Tensor,
        t_span: Tuple[float, float],
        num_steps: Optional[int] = None,
        drift_loss_fn: Optional[Callable] = None,
        drift_weight: float = 0.0,
        drift_clip: float = 1.0,
        **kwargs,
    ) -> torch.Tensor:
        """
        Solve ODE with optional drift guidance.
        
        Args:
            velocity_fn: Velocity field function v(t, x, **kwargs)
            x0: Initial state (B, state_dim)
            t_span: (t_start, t_end) integration interval
            num_steps: Number of integration steps (for fixed solvers)
            drift_loss_fn: Function returning drift loss gradient
            drift_weight: Weight of drift guidance
            drift_clip: Clipping norm for stable integration
            **kwargs: Additional arguments passed to velocity_fn
            
        Returns:
            State tensor at t_end, shape (B, state_dim)
        """
        
        # Wrap velocity function with drift guidance if enabled
        if drift_weight > 0 and drift_loss_fn is not None:
            augmented_fn = DriftAugmentedVelocityField(
                velocity_fn,
                drift_loss_fn=drift_loss_fn,
                drift_weight=drift_weight,
                drift_clip=drift_clip,
            )
        else:
            def augmented_fn(t, x):
                return velocity_fn(t, x, **kwargs)
        
        t_start, t_end = t_span
        
        if self.solver_backend == "torchdiffeq" and torchdiffeq_odeint is not None:
            return self._solve_torchdiffeq(
                augmented_fn, x0, t_start, t_end, **kwargs
            )
        else:
            return self._solve_legacy(
                augmented_fn, x0, t_start, t_end, num_steps=num_steps
            )

    def _solve_legacy(
        self,
        velocity_fn: Callable,
        x0: torch.Tensor,
        t_start: float,
        t_end: float,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Legacy explicit Euler or RK4 integration (no external dependencies).
        """
        if num_steps is None:
            num_steps = 10  # Default number of steps
        
        dt = (t_end - t_start) / num_steps
        x = x0.clone()
        
        for step in range(num_steps):
            t = t_start + step * dt
            t_tensor = torch.tensor(t, dtype=x.dtype, device=x.device)
            
            if self.solver_method == "euler":
                k1 = velocity_fn(t_tensor, x)
                x = x + dt * k1
                
            elif self.solver_method == "rk4":
                k1 = velocity_fn(t_tensor, x)
                k2 = velocity_fn(t_tensor + dt/2, x + (dt/2) * k1)
                k3 = velocity_fn(t_tensor + dt/2, x + (dt/2) * k2)
                k4 = velocity_fn(t_tensor + dt, x + dt * k3)
                x = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            else:
                raise ValueError(f"Legacy solver doesn't support {self.solver_method}")
        
        return x

    def _solve_torchdiffeq(
        self,
        velocity_fn: Callable,
        x0: torch.Tensor,
        t_start: float,
        t_end: float,
        **kwargs,
    ) -> torch.Tensor:
        """
        Use torchdiffeq backend for adaptive stepping.
        """
        # Create time points for integration
        t = torch.linspace(t_start, t_end, 2, dtype=x0.dtype, device=x0.device)
        
        # Wrap for torchdiffeq (which expects (t, y) signature)
        def ode_func(t, y):
            return velocity_fn(t, y)
        
        # Solve with torchdiffeq
        solution = torchdiffeq_odeint(
            ode_func,
            x0,
            t,
            method=self.solver_method,
            rtol=self.rtol,
            atol=self.atol,
            step_size=self.step_size,
        )
        
        # Return final state
        return solution[-1]


def sample_trajectory_with_drift(
    model,
    x0: torch.Tensor,
    cond: Optional[torch.Tensor] = None,
    returns: Optional[torch.Tensor] = None,
    t_span: Tuple[float, float] = (0.0, 1.0),
    num_steps: int = 10,
    drift_loss_fn: Optional[Callable] = None,
    drift_weight: float = 0.1,
    solver_method: str = "euler",
    solver_backend: str = "legacy_euler",
) -> torch.Tensor:
    """
    Convenience function: sample trajectory with FM+drift guidance.
    
    Args:
        model: FM network (assumes model(x, cond, t) signature)
        x0: Initial state (B, state_dim)
        cond: Conditioning info (B, cond_dim)
        returns: Optional return-to-go info (B, 1)
        t_span: (t_start, t_end) for ODE integration
        num_steps: Number of integration steps
        drift_loss_fn: Function to compute drift guidance
        drift_weight: Drift weight lambda
        solver_method: "euler" | "rk4" | "rk45" | etc.
        solver_backend: "legacy_euler" | "torchdiffeq"
        
    Returns:
        Final trajectory state (B, state_dim)
    """
    
    def velocity_fn(t, x, cond=None, returns=None):
        """Velocity field from FM model."""
        return model(x, cond, t, returns=returns)
    
    def drift_grad_fn(x):
        """Gradient of drift loss."""
        if drift_loss_fn is None:
            return torch.zeros_like(x)
        return drift_loss_fn.get_gradient(x)
    
    solver = DriftODESolver(
        solver_method=solver_method,
        solver_backend=solver_backend,
    )
    
    return solver.solve(
        velocity_fn,
        x0,
        t_span,
        num_steps=num_steps,
        drift_loss_fn=drift_grad_fn if drift_weight > 0 else None,
        drift_weight=drift_weight,
        cond=cond,
        returns=returns,
    )
