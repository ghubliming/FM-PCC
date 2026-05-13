"""
Unit Tests for Improved Mean Flows Modules

Comprehensive test suite covering:
- Velocity field models (u/v decomposition)
- JVP guidance and constraint computation
- ODE solvers (single/dual/multi-step)
- Training utilities (loss, scheduler)
- Metrics tracking
- DiT transformer backbone
- Trajectory sampling
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.imf_velocity import (
    MLP,
    DualVelocityField,
    TimeConditionedDualVelocity,
)
from models.jvp_guidance import JVPGuidance, SoftConstraintModule
from sampling.imf_ode_solvers import ImfODESolver
from utils.imf_training import (
    DualVelocityLoss,
    DualVelocityScheduler,
    ImfTrainingWrapper,
    compute_trajectory_targets,
)
from utils.imf_metrics import ImfMetricsTracker, TrajectoryQualityMetrics
from models.imf_dit_trajectory import (
    TimeEmbedding,
    MultiHeadAttention,
    ImfDiTTrajectory,
)
from sampling.imf_trajectory_sampler import ImfTrajectorySampler, ConditionalImfSampler


# ============================================================================
# Test Parameters
# ============================================================================

DEVICE = 'cpu'
STATE_DIM = 28
BATCH_SIZE = 4
SEQ_LEN = 10


# ============================================================================
# Tests: Velocity Field Models
# ============================================================================

class TestMLP:
    """Test simple MLP network."""
    
    def test_mlp_forward_shape(self):
        mlp = MLP(input_dim=28, output_dim=28, hidden_dim=64)
        x = torch.randn(4, 28)
        y = mlp(x)
        assert y.shape == (4, 28)
    
    def test_mlp_with_different_dims(self):
        mlp = MLP(input_dim=50, output_dim=20, hidden_dim=100)
        x = torch.randn(2, 50)
        y = mlp(x)
        assert y.shape == (2, 20)


class TestDualVelocityField:
    """Test dual-velocity decomposition."""
    
    def test_dual_velocity_forward_shape(self):
        model = DualVelocityField(state_dim=STATE_DIM, hidden_dim=64, use_jvp=False)
        x = torch.randn(BATCH_SIZE, STATE_DIM)
        u, v = model(x)
        
        assert u.shape == (BATCH_SIZE, STATE_DIM)
        assert v.shape == (BATCH_SIZE, STATE_DIM)
    
    def test_dual_velocity_with_jvp(self):
        model = DualVelocityField(state_dim=STATE_DIM, hidden_dim=64, use_jvp=True)
        x = torch.randn(BATCH_SIZE, STATE_DIM)
        u, v = model(x)
        
        assert u.shape == (BATCH_SIZE, STATE_DIM)
        assert v.shape == (BATCH_SIZE, STATE_DIM)
    
    def test_compute_u_target(self):
        model = DualVelocityField(state_dim=STATE_DIM, hidden_dim=64)
        traj = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM)
        u_target = model.compute_u_target(traj)
        
        assert u_target.shape == traj.shape
    
    def test_compute_v_target(self):
        model = DualVelocityField(state_dim=STATE_DIM, hidden_dim=64)
        traj = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM)
        v_target = model.compute_v_target(traj)
        
        assert v_target.shape == traj.shape


class TestTimeConditionedDualVelocity:
    """Test time-conditioned velocity model."""
    
    def test_time_conditioned_forward(self):
        model = TimeConditionedDualVelocity(
            state_dim=STATE_DIM,
            hidden_dim=64,
            time_dim=32,
        )
        x = torch.randn(BATCH_SIZE, STATE_DIM)
        t = torch.rand(BATCH_SIZE)
        
        u, v = model(x, t)
        
        assert u.shape == (BATCH_SIZE, STATE_DIM)
        assert v.shape == (BATCH_SIZE, STATE_DIM)
    
    def test_time_conditioned_sequence(self):
        model = TimeConditionedDualVelocity(
            state_dim=STATE_DIM,
            hidden_dim=64,
            time_dim=32,
        )
        x = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM)
        t = torch.linspace(0, 1, SEQ_LEN).unsqueeze(0).expand(BATCH_SIZE, SEQ_LEN)
        
        u, v = model(x, t)
        
        assert u.shape == x.shape
        assert v.shape == x.shape


# ============================================================================
# Tests: JVP Guidance
# ============================================================================

class TestJVPGuidance:
    """Test Jacobian-Vector Product guidance."""
    
    def test_jvp_guidance_initialization(self):
        jvp = JVPGuidance(state_dim=STATE_DIM)
        assert jvp.state_dim == STATE_DIM
    
    def test_collision_free_constraint(self):
        jvp = JVPGuidance(state_dim=STATE_DIM)
        x = torch.randn(BATCH_SIZE, STATE_DIM)
        obstacle_centers = torch.randn(3, STATE_DIM)
        obstacle_radii = torch.full((3,), 0.5)
        
        constraint = jvp.collision_free_constraint(x, obstacle_centers, obstacle_radii)
        assert constraint.shape == (BATCH_SIZE,)
    
    def test_smoothness_constraint(self):
        jvp = JVPGuidance(state_dim=STATE_DIM)
        v = torch.randn(BATCH_SIZE, STATE_DIM)
        
        constraint = jvp.smoothness_constraint(v, max_accel=1.0)
        assert constraint.shape == (BATCH_SIZE,)
    
    def test_jvp_forward(self):
        jvp = JVPGuidance(state_dim=STATE_DIM)
        x = torch.randn(BATCH_SIZE, STATE_DIM, requires_grad=True)
        v = torch.randn(BATCH_SIZE, STATE_DIM)
        
        def constraint_fn(z):
            return torch.norm(z, dim=-1)
        
        jvp_product = jvp.compute_jvp(x, v, constraint_fn)
        assert jvp_product.shape == (BATCH_SIZE,)


class TestSoftConstraintModule:
    """Test learned constraint weighting."""
    
    def test_soft_constraint_forward(self):
        module = SoftConstraintModule(state_dim=STATE_DIM, num_constraints=3)
        x = torch.randn(BATCH_SIZE, STATE_DIM)
        
        weights = module(x)
        assert weights.shape == (BATCH_SIZE, 3)
        assert torch.allclose(weights.sum(dim=-1), torch.ones(BATCH_SIZE))


# ============================================================================
# Tests: ODE Solvers
# ============================================================================

class TestImfODESolver:
    """Test ODE integration."""
    
    def test_solver_initialization(self):
        solver = ImfODESolver(num_steps=10, solver_type='rk4')
        assert solver.num_steps == 10
    
    def test_euler_step(self):
        solver = ImfODESolver(num_steps=5)
        x = torch.randn(BATCH_SIZE, STATE_DIM)
        v = torch.randn(BATCH_SIZE, STATE_DIM)
        dt = 0.1
        
        x_next = solver.euler_step(x, v, dt)
        assert x_next.shape == x.shape
        assert not torch.allclose(x_next, x)  # Should change
    
    def test_rk4_step(self):
        solver = ImfODESolver(num_steps=5)
        x = torch.randn(BATCH_SIZE, STATE_DIM)
        
        def velocity_func(t, x):
            return torch.randn_like(x) * 0.1
        
        dt = 0.1
        x_next = solver.rk4_step(x, velocity_func, 0, dt)
        assert x_next.shape == x.shape
    
    def test_solve_manual_euler(self):
        solver = ImfODESolver(num_steps=5, solver_type='euler')
        x0 = torch.randn(BATCH_SIZE, STATE_DIM)
        
        def velocity_func(t, x):
            return -x * 0.1  # Decay
        
        x_final = solver.solve_manual(x0, velocity_func, 0.0, 1.0, method='euler')
        assert x_final.shape == x0.shape
        assert torch.norm(x_final) < torch.norm(x0)  # Should decay


# ============================================================================
# Tests: Training
# ============================================================================

class TestDualVelocityLoss:
    """Test dual-velocity loss computation."""
    
    def test_loss_computation(self):
        loss_fn = DualVelocityLoss(weight_u=0.5, weight_v=0.5)
        
        u_pred = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM)
        u_target = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM)
        v_pred = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM)
        v_target = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM)
        
        loss, loss_dict = loss_fn(u_pred, u_target, v_pred, v_target)
        
        assert loss.item() > 0
        assert 'loss_u' in loss_dict
        assert 'loss_v' in loss_dict
    
    def test_loss_with_jvp(self):
        loss_fn = DualVelocityLoss(weight_u=0.5, weight_v=0.5, weight_jvp=0.2)
        
        u_pred = torch.randn(BATCH_SIZE, STATE_DIM)
        u_target = torch.randn(BATCH_SIZE, STATE_DIM)
        v_pred = torch.randn(BATCH_SIZE, STATE_DIM)
        v_target = torch.randn(BATCH_SIZE, STATE_DIM)
        jvp_penalty = torch.rand(BATCH_SIZE) * 0.1
        
        loss, loss_dict = loss_fn(u_pred, u_target, v_pred, v_target, jvp_penalty)
        assert loss.item() > 0


class TestDualVelocityScheduler:
    """Test loss weight scheduling."""
    
    def test_balanced_schedule(self):
        scheduler = DualVelocityScheduler(mode='balanced', total_steps=100)
        
        w_u, w_v = scheduler.get_weights()
        assert w_u > 0 and w_v > 0
    
    def test_u_first_schedule(self):
        scheduler = DualVelocityScheduler(
            mode='u_first',
            total_steps=100,
            weight_u_start=0.9,
            weight_v_start=0.0,
        )
        
        # Early steps: u dominant
        scheduler.step = 0
        w_u, w_v = scheduler.get_weights()
        assert w_u > w_v
        
        # Late steps: balanced
        scheduler.step = 100
        w_u, w_v = scheduler.get_weights()
        assert abs(w_u - w_v) < 0.1


class TestImfTrainingWrapper:
    """Test training wrapper."""
    
    def test_training_wrapper_initialization(self):
        wrapper = ImfTrainingWrapper(
            loss_weights={'u': 0.5, 'v': 0.5},
        )
        assert wrapper.loss_fn is not None
    
    def test_compute_training_loss(self):
        wrapper = ImfTrainingWrapper()
        
        u_pred = torch.randn(BATCH_SIZE, STATE_DIM)
        u_target = torch.randn(BATCH_SIZE, STATE_DIM)
        v_pred = torch.randn(BATCH_SIZE, STATE_DIM)
        v_target = torch.randn(BATCH_SIZE, STATE_DIM)
        
        loss, loss_dict = wrapper.compute_training_loss(
            u_pred, u_target, v_pred, v_target
        )
        
        assert loss.item() > 0
        assert 'weight_u' in loss_dict


class TestVelocityTargetExtraction:
    """Test target velocity computation."""
    
    def test_compute_trajectory_targets(self):
        traj = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM)
        u_target, v_target = compute_trajectory_targets(traj)
        
        assert u_target.shape == traj.shape
        assert v_target.shape == traj.shape


# ============================================================================
# Tests: Metrics
# ============================================================================

class TestImfMetricsTracker:
    """Test metrics tracking."""
    
    def test_metrics_initialization(self):
        tracker = ImfMetricsTracker()
        assert len(tracker.u_errors) == 0
    
    def test_compute_u_error(self):
        tracker = ImfMetricsTracker()
        
        u_pred = torch.randn(BATCH_SIZE, STATE_DIM)
        u_target = torch.randn(BATCH_SIZE, STATE_DIM)
        
        error = tracker.compute_u_error(u_pred, u_target)
        assert error > 0
        assert len(tracker.u_errors) == 1
    
    def test_compute_smoothness(self):
        tracker = ImfMetricsTracker()
        
        traj = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM).cumsum(dim=1)
        smoothness = tracker.compute_smoothness(traj)
        
        assert 0 <= smoothness <= 1
    
    def test_get_summary(self):
        tracker = ImfMetricsTracker()
        
        for _ in range(5):
            u_pred = torch.randn(BATCH_SIZE, STATE_DIM)
            u_target = torch.randn(BATCH_SIZE, STATE_DIM)
            tracker.compute_u_error(u_pred, u_target)
        
        summary = tracker.get_summary()
        assert 'u_error' in summary


class TestTrajectoryQualityMetrics:
    """Test trajectory quality analysis."""
    
    def test_path_length(self):
        metrics = TrajectoryQualityMetrics()
        traj = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM).cumsum(dim=1)
        
        length = metrics.compute_path_length(traj)
        assert length > 0
    
    def test_max_velocity(self):
        metrics = TrajectoryQualityMetrics()
        traj = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM).cumsum(dim=1)
        
        max_vel = metrics.compute_max_velocity(traj)
        assert max_vel > 0
    
    def test_max_acceleration(self):
        metrics = TrajectoryQualityMetrics()
        traj = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM).cumsum(dim=1).cumsum(dim=1)
        
        max_accel = metrics.compute_max_acceleration(traj)
        assert max_accel >= 0


# ============================================================================
# Tests: DiT
# ============================================================================

class TestTimeEmbedding:
    """Test sinusoidal time embedding."""
    
    def test_time_embedding_shape(self):
        embed = TimeEmbedding(dim=64)
        t = torch.rand(BATCH_SIZE)
        
        emb = embed(t)
        assert emb.shape == (BATCH_SIZE, 64)
    
    def test_time_embedding_range(self):
        embed = TimeEmbedding(dim=64)
        t = torch.linspace(0, 1, BATCH_SIZE)
        
        emb = embed(t)
        assert torch.isfinite(emb).all()


class TestMultiHeadAttention:
    """Test multi-head attention."""
    
    def test_attention_shape(self):
        attn = MultiHeadAttention(dim=64, heads=4)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, 64)
        
        out = attn(x)
        assert out.shape == x.shape
    
    def test_attention_with_mask(self):
        attn = MultiHeadAttention(dim=64, heads=4)
        x = torch.randn(BATCH_SIZE, SEQ_LEN, 64)
        mask = torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.bool)
        
        out = attn(x, mask)
        assert out.shape == x.shape


class TestImfDiTTrajectory:
    """Test DiT trajectory model."""
    
    def test_dit_forward(self):
        model = ImfDiTTrajectory(
            state_dim=STATE_DIM,
            latent_dim=64,
            num_blocks=2,
        )
        
        x = torch.randn(BATCH_SIZE, SEQ_LEN, STATE_DIM)
        t = torch.rand(BATCH_SIZE)
        
        u, v = model(x, t)
        
        assert u.shape == (BATCH_SIZE, SEQ_LEN, STATE_DIM)
        assert v.shape == (BATCH_SIZE, SEQ_LEN, STATE_DIM)


# ============================================================================
# Tests: Sampling
# ============================================================================

class TestImfTrajectorySampler:
    """Test trajectory sampler."""
    
    def test_sampler_initialization(self):
        model = DualVelocityField(state_dim=STATE_DIM, hidden_dim=64)
        sampler = ImfTrajectorySampler(model, num_steps=5, state_dim=STATE_DIM)
        
        assert sampler.velocity_model is not None
    
    def test_single_step_sampling(self):
        model = DualVelocityField(state_dim=STATE_DIM, hidden_dim=64)
        sampler = ImfTrajectorySampler(model, num_steps=5, solver_type='euler', state_dim=STATE_DIM)
        
        z_init = torch.randn(BATCH_SIZE, STATE_DIM)
        z_final = sampler.sample_single_step(z_init)
        
        assert z_final.shape == z_init.shape
    
    def test_dual_step_sampling(self):
        model = DualVelocityField(state_dim=STATE_DIM, hidden_dim=64)
        sampler = ImfTrajectorySampler(model, num_steps=5, solver_type='euler', state_dim=STATE_DIM)
        
        z_init = torch.randn(BATCH_SIZE, STATE_DIM)
        z_u, z_v, z_combined = sampler.sample_dual_step(z_init)
        
        assert z_u.shape == z_init.shape
        assert z_v.shape == z_init.shape
        assert z_combined.shape == z_init.shape
    
    def test_multi_step_sampling(self):
        model = DualVelocityField(state_dim=STATE_DIM, hidden_dim=64)
        sampler = ImfTrajectorySampler(model, num_steps=5, solver_type='euler', state_dim=STATE_DIM)
        
        z_init = torch.randn(BATCH_SIZE, STATE_DIM)
        results = sampler.sample_multi_step(z_init, num_phases=4)
        
        assert 'phase_0' in results
        assert 'phase_4' in results


class TestConditionalImfSampler:
    """Test conditional sampler."""
    
    def test_goal_guided_sampling(self):
        model = DualVelocityField(state_dim=STATE_DIM, hidden_dim=64)
        sampler = ConditionalImfSampler(model, num_steps=5, solver_type='euler', state_dim=STATE_DIM)
        
        z_init = torch.randn(BATCH_SIZE, STATE_DIM)
        z_goal = torch.randn(BATCH_SIZE, STATE_DIM)
        
        z_guided = sampler.sample_toward_goal(z_init, z_goal)
        assert z_guided.shape == z_init.shape
    
    def test_obstacle_avoidance_sampling(self):
        model = DualVelocityField(state_dim=STATE_DIM, hidden_dim=64)
        sampler = ConditionalImfSampler(model, num_steps=5, solver_type='euler', state_dim=STATE_DIM)
        
        z_init = torch.randn(BATCH_SIZE, STATE_DIM)
        obstacle_centers = torch.randn(3, STATE_DIM)
        obstacle_radii = torch.full((3,), 0.5)
        
        z_safe = sampler.sample_avoiding_obstacles(z_init, obstacle_centers, obstacle_radii)
        assert z_safe.shape == z_init.shape


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
