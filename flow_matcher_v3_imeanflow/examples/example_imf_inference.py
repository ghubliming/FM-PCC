#!/usr/bin/env python3
"""
Example: Inference with Improved Mean Flows

Demonstrates:
- Loading a trained dual-velocity model
- Single-step and dual-step sampling
- Goal-guided and constraint-aware sampling
- Visualization of decomposition
"""

import torch
import sys
from pathlib import Path

# Add project to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.imf_velocity import TimeConditionedDualVelocity
from sampling.imf_trajectory_sampler import ImfTrajectorySampler, ConditionalImfSampler


def demo_basic_sampling():
    """Demo 1: Basic single-step and dual-step sampling."""
    print("=" * 80)
    print("Demo 1: Basic Sampling (Single-step vs Dual-step)")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dim = 28
    batch_size = 4
    
    # Create model
    model = TimeConditionedDualVelocity(
        state_dim=state_dim,
        hidden_dim=128,
        time_dim=64,
    ).to(device)
    
    # Create sampler
    sampler = ImfTrajectorySampler(
        velocity_model=model,
        num_steps=10,
        solver_type='rk4',
        state_dim=state_dim,
    )
    
    # Initial state
    z_init = torch.randn(batch_size, state_dim, device=device)
    
    # Single-step sampling (fast, NFE=1)
    print("\nSingle-step sampling (NFE=1):")
    z_fast = sampler.sample_single_step(z_init)
    print(f"  Input shape: {z_init.shape}")
    print(f"  Output shape: {z_fast.shape}")
    print(f"  Change magnitude: {(z_fast - z_init).norm(dim=-1).mean().item():.4f}")
    
    # Dual-step sampling (higher quality, NFE=2)
    print("\nDual-step sampling (NFE=2):")
    z_u, z_v, z_combined = sampler.sample_dual_step(z_init, t_split=0.5)
    print(f"  After u phase: {z_u.shape}")
    print(f"  After v phase: {z_v.shape}")
    print(f"  Direct combined: {z_combined.shape}")
    
    # Compare u vs v contributions
    u_change = (z_u - z_init).norm(dim=-1).mean().item()
    v_change = (z_v - z_u).norm(dim=-1).mean().item()
    print(f"  u phase change: {u_change:.4f}")
    print(f"  v phase change: {v_change:.4f}")
    print(f"  Ratio (v/u): {v_change / (u_change + 1e-6):.4f}")
    
    print()


def demo_multi_phase_sampling():
    """Demo 2: Multi-phase sampling analysis."""
    print("=" * 80)
    print("Demo 2: Multi-phase Sampling (Understanding u vs v)")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dim = 28
    batch_size = 1
    
    model = TimeConditionedDualVelocity(
        state_dim=state_dim,
        hidden_dim=128,
        time_dim=64,
    ).to(device)
    
    sampler = ImfTrajectorySampler(
        velocity_model=model,
        num_steps=5,
        solver_type='rk4',
        state_dim=state_dim,
    )
    
    z_init = torch.randn(batch_size, state_dim, device=device)
    
    # 4-phase sampling
    print("\nPhase-by-phase trajectory:")
    results = sampler.sample_multi_step(z_init, num_phases=4)
    
    for phase_name, z_phase in results.items():
        if phase_name == 'phase_0':
            print(f"  {phase_name}: Initial state (baseline)")
        else:
            prev_phase = f"phase_{int(phase_name.split('_')[1]) - 1}"
            z_prev = results[prev_phase]
            delta = (z_phase - z_prev).norm().item()
            print(f"  {phase_name}: Change={delta:.4f}")
    
    print()


def demo_goal_guided_sampling():
    """Demo 3: Goal-guided sampling."""
    print("=" * 80)
    print("Demo 3: Goal-Guided Sampling")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dim = 28
    batch_size = 4
    
    model = TimeConditionedDualVelocity(
        state_dim=state_dim,
        hidden_dim=128,
        time_dim=64,
    ).to(device)
    
    sampler = ConditionalImfSampler(
        velocity_model=model,
        num_steps=10,
        solver_type='rk4',
        state_dim=state_dim,
        goal_weight=0.1,
    )
    
    z_init = torch.randn(batch_size, state_dim, device=device)
    z_goal = torch.randn(batch_size, state_dim, device=device)
    
    # Sample without goal
    z_unconditioned = sampler.sample_single_step(z_init)
    
    # Sample with goal
    z_guided = sampler.sample_toward_goal(z_init, z_goal)
    
    # Analyze goal alignment
    print(f"\nGoal-guided sampling:")
    print(f"  Initial state L2 norm: {z_init.norm(dim=-1).mean().item():.4f}")
    print(f"  Goal state L2 norm: {z_goal.norm(dim=-1).mean().item():.4f}")
    
    # Distance to goal
    dist_uncond = (z_unconditioned - z_goal).norm(dim=-1).mean().item()
    dist_guided = (z_guided - z_goal).norm(dim=-1).mean().item()
    dist_init = (z_init - z_goal).norm(dim=-1).mean().item()
    
    print(f"\n  Initial distance to goal: {dist_init:.4f}")
    print(f"  Distance after unconditioned sampling: {dist_uncond:.4f}")
    print(f"  Distance after goal-guided sampling: {dist_guided:.4f}")
    print(f"  Goal guidance improvement: {(dist_uncond - dist_guided) / (dist_init + 1e-6) * 100:.1f}%")
    
    print()


def demo_obstacle_avoidance():
    """Demo 4: Obstacle-aware sampling."""
    print("=" * 80)
    print("Demo 4: Obstacle Avoidance")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dim = 28
    batch_size = 4
    
    model = TimeConditionedDualVelocity(
        state_dim=state_dim,
        hidden_dim=128,
        time_dim=64,
    ).to(device)
    
    sampler = ConditionalImfSampler(
        velocity_model=model,
        num_steps=10,
        solver_type='rk4',
        state_dim=state_dim,
    )
    
    z_init = torch.randn(batch_size, state_dim, device=device)
    
    # Define obstacles
    num_obstacles = 3
    obstacle_centers = torch.randn(num_obstacles, state_dim, device=device)
    obstacle_radii = torch.full((num_obstacles,), 0.3, device=device)
    
    # Sample without obstacle avoidance
    z_unconstrained = sampler.sample_single_step(z_init)
    
    # Sample with obstacle avoidance
    z_safe = sampler.sample_avoiding_obstacles(
        z_init,
        obstacle_centers=obstacle_centers,
        obstacle_radii=obstacle_radii,
        avoidance_weight=0.05,
    )
    
    # Check distances
    print(f"\nObstacle avoidance:")
    print(f"  Number of obstacles: {num_obstacles}")
    print(f"  Obstacle radius: {obstacle_radii[0].item():.3f}")
    
    def min_obstacle_distance(z, centers, radii):
        """Compute minimum distance to obstacle surfaces."""
        distances = []
        for i in range(centers.shape[0]):
            dist_to_surface = (z - centers[i]).norm(dim=-1) - radii[i]
            distances.append(dist_to_surface)
        return torch.min(torch.stack(distances), dim=0)[0]
    
    min_dist_unconstrained = min_obstacle_distance(z_unconstrained, obstacle_centers, obstacle_radii).mean().item()
    min_dist_safe = min_obstacle_distance(z_safe, obstacle_centers, obstacle_radii).mean().item()
    
    print(f"  Min distance (unconstrained): {min_dist_unconstrained:.4f}")
    print(f"  Min distance (with avoidance): {min_dist_safe:.4f}")
    
    is_collision_unconstrained = min_dist_unconstrained < 0
    is_collision_safe = min_dist_safe < 0
    
    if is_collision_unconstrained:
        print(f"  ⚠ Unconstrained trajectory has collision!")
    else:
        print(f"  ✓ Unconstrained trajectory is collision-free")
    
    if is_collision_safe:
        print(f"  ⚠ Safe trajectory still has collision (increase avoidance_weight)")
    else:
        print(f"  ✓ Safe trajectory is collision-free")
    
    print()


def demo_velocity_decomposition():
    """Demo 5: Analyze u vs v decomposition."""
    print("=" * 80)
    print("Demo 5: Velocity Field Decomposition (u vs v)")
    print("=" * 80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    state_dim = 28
    batch_size = 8
    
    model = TimeConditionedDualVelocity(
        state_dim=state_dim,
        hidden_dim=128,
        time_dim=64,
    ).to(device)
    
    # Sample random states and time
    x = torch.randn(batch_size, state_dim, device=device)
    t = torch.rand(batch_size, device=device)
    
    # Get velocity decomposition
    with torch.no_grad():
        u, v = model(x, t)
    
    # Analyze magnitudes
    u_norm = u.norm(dim=-1)  # (B,)
    v_norm = v.norm(dim=-1)  # (B,)
    combined_norm = (u + v).norm(dim=-1)  # (B,)
    
    print(f"\nVelocity decomposition analysis:")
    print(f"  Batch size: {batch_size}")
    print(f"  State dimension: {state_dim}")
    print()
    
    print(f"  Average u (global) magnitude: {u_norm.mean().item():.4f} ± {u_norm.std().item():.4f}")
    print(f"  Average v (local) magnitude: {v_norm.mean().item():.4f} ± {v_norm.std().item():.4f}")
    print(f"  Average combined magnitude: {combined_norm.mean().item():.4f} ± {combined_norm.std().item():.4f}")
    print()
    
    # Decomposition ratios
    u_contrib = 100 * u_norm / (combined_norm + 1e-6)
    v_contrib = 100 * v_norm / (combined_norm + 1e-6)
    
    print(f"  u contribution: {u_contrib.mean().item():.1f}% (range: {u_contrib.min():.1f}%-{u_contrib.max():.1f}%)")
    print(f"  v contribution: {v_contrib.mean().item():.1f}% (range: {v_contrib.min():.1f}%-{v_contrib.max():.1f}%)")
    print()
    
    # Cosine similarity
    u_flat = u.reshape(batch_size, -1)
    v_flat = v.reshape(batch_size, -1)
    cos_sim = torch.nn.functional.cosine_similarity(u_flat, v_flat)
    cos_sim = (cos_sim + 1) / 2  # Convert to [0, 1]
    
    print(f"  u-v alignment (cosine similarity): {cos_sim.mean().item():.4f}")
    print(f"    (1.0 = perfectly aligned, 0.0 = orthogonal)")
    print()


def main():
    """Run all demonstrations."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 78 + "║")
    print("║" + "iMeanFlow Inference Demonstration".center(78) + "║")
    print("║" + " " * 78 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    demo_basic_sampling()
    demo_multi_phase_sampling()
    demo_goal_guided_sampling()
    demo_obstacle_avoidance()
    demo_velocity_decomposition()
    
    print("=" * 80)
    print("All demonstrations complete!")
    print("=" * 80)


if __name__ == '__main__':
    main()
