"""
Drift Loss Module for FM-D (Flow Matcher-Drifting)

Adapted from /workspaces/drifting/drift_loss.py for trajectory control domain.
Measures deviation of sampled trajectories from learned distribution.

Key differences from original drifting:
- Operates on trajectory tensors (T, state_dim) instead of image tensors
- KL divergence computed via learned encoder on sequence embeddings
- Gradient-based drift guidance for ODE integration
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict


class DriftLoss(nn.Module):
    """
    Drift loss for trajectory distribution matching.
    
    Computes distance between sampled trajectory and reference (expert) distribution
    to guide ODE integration via gradient ascent on trajectory quality.
    """
    
    def __init__(
        self,
        trajectory_dim: int,
        loss_type: str = "kl_divergence",
        memory_bank_size: int = 5000,
        temperature: float = 0.1,
    ):
        """
        Args:
            trajectory_dim: Dimensionality of trajectory (T * state_dim)
            loss_type: "kl_divergence" | "adversarial" | "mmd"
            memory_bank_size: Size of reference distribution buffer
            temperature: Softmax temperature for probability scaling
        """
        super().__init__()
        self.trajectory_dim = trajectory_dim
        self.loss_type = loss_type
        self.memory_bank_size = memory_bank_size
        self.temperature = temperature
        
        # Initialize reference trajectory bank (circular buffer)
        self.register_buffer(
            'memory_bank',
            torch.zeros(memory_bank_size, trajectory_dim, dtype=torch.float32)
        )
        self.register_buffer('memory_bank_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('memory_bank_full', torch.tensor(False, dtype=torch.bool))
        
        if loss_type == "kl_divergence":
            self.encoder = self._build_encoder(trajectory_dim, output_dim=128)
            self.discriminator = None
        elif loss_type == "adversarial":
            self.encoder = self._build_encoder(trajectory_dim, output_dim=128)
            self.discriminator = self._build_discriminator(input_dim=128)
        elif loss_type == "mmd":
            self.encoder = self._build_encoder(trajectory_dim, output_dim=128)
            self.discriminator = None
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def _build_encoder(self, input_dim: int, output_dim: int = 128) -> nn.Module:
        """Build trajectory encoder (simple MLP)."""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.LayerNorm(output_dim),
        )

    def _build_discriminator(self, input_dim: int = 128) -> nn.Module:
        """Build simple discriminator for adversarial loss."""
        return nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def update_memory_bank(self, trajectories: torch.Tensor) -> None:
        """
        Update circular buffer with new reference trajectories (expert demos).
        
        Args:
            trajectories: (B, T*state_dim) tensor of reference trajectories
        """
        B = trajectories.shape[0]
        ptr = int(self.memory_bank_ptr)
        
        # Wrap around if necessary
        if ptr + B > self.memory_bank_size:
            self.memory_bank[ptr:] = trajectories[:self.memory_bank_size - ptr]
            remaining = B - (self.memory_bank_size - ptr)
            self.memory_bank[:remaining] = trajectories[self.memory_bank_size - ptr:]
            self.memory_bank_full.fill_(True)
        else:
            self.memory_bank[ptr:ptr + B] = trajectories
            if ptr + B == self.memory_bank_size:
                self.memory_bank_full.fill_(True)
        
        self.memory_bank_ptr.fill_((ptr + B) % self.memory_bank_size)

    def compute_kl_divergence(
        self,
        sampled_trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        KL divergence loss: D_KL(Q_sampled || P_expert)
        
        Args:
            sampled_trajectory: (T*state_dim,) or (B, T*state_dim) trajectory
            
        Returns:
            loss scalar or (B,) tensor
        """
        if sampled_trajectory.dim() == 1:
            sampled_trajectory = sampled_trajectory.unsqueeze(0)
        
        B = sampled_trajectory.shape[0]
        
        # Encode sampled trajectory
        q_z = self.encoder(sampled_trajectory)  # (B, 128)
        
        # Get current memory bank (expert trajectories)
        if self.memory_bank_full:
            ref_trajs = self.memory_bank  # (memory_bank_size, T*state_dim)
        else:
            ptr = int(self.memory_bank_ptr)
            ref_trajs = self.memory_bank[:ptr]
        
        if ref_trajs.shape[0] == 0:
            # Memory bank not yet populated; return zero loss with gradient
            return torch.zeros(B, device=sampled_trajectory.device)
        
        # Encode reference trajectories
        with torch.no_grad():
            p_z = self.encoder(ref_trajs)  # (N_ref, 128)
        
        # Compute pairwise distances (L2 norm)
        # (B, 128) vs (N_ref, 128) -> (B, N_ref)
        dist = torch.cdist(q_z, p_z, p=2)  # Euclidean distance
        
        # Softmax over reference distribution
        probs = torch.softmax(-dist / self.temperature, dim=1)  # (B, N_ref)
        
        # KL as negative log probability of closest match
        # (Higher prob → lower loss)
        kl = -torch.log(probs.max(dim=1)[0] + 1e-8)  # (B,)
        
        return kl.mean()

    def compute_mmd_loss(
        self,
        sampled_trajectory: torch.Tensor,
        sigma: float = 1.0,
    ) -> torch.Tensor:
        """
        Maximum Mean Discrepancy loss between sampled and reference distributions.
        
        Args:
            sampled_trajectory: (B, T*state_dim) trajectory
            sigma: RBF kernel bandwidth parameter
            
        Returns:
            mmd loss scalar
        """
        if sampled_trajectory.dim() == 1:
            sampled_trajectory = sampled_trajectory.unsqueeze(0)
        
        # Get reference trajectories
        if self.memory_bank_full:
            ref_trajs = self.memory_bank
        else:
            ptr = int(self.memory_bank_ptr)
            ref_trajs = self.memory_bank[:ptr]
        
        if ref_trajs.shape[0] == 0:
            return torch.zeros(1, device=sampled_trajectory.device)[0]
        
        # Encode both
        q_z = self.encoder(sampled_trajectory)  # (B, 128)
        with torch.no_grad():
            p_z = self.encoder(ref_trajs)  # (N, 128)
        
        # RBF kernel: k(x, y) = exp(-||x-y||^2 / sigma)
        def rbf_kernel(x, y, sigma=1.0):
            """Compute RBF kernel matrix."""
            dist_sq = torch.cdist(x, y, p=2) ** 2
            return torch.exp(-dist_sq / (2 * sigma ** 2))
        
        # Compute kernel matrices
        K_qq = rbf_kernel(q_z, q_z, sigma)  # (B, B)
        K_pp = rbf_kernel(p_z, p_z, sigma)  # (N, N)
        K_qp = rbf_kernel(q_z, p_z, sigma)  # (B, N)
        
        # MMD^2 = E[K_qq] - 2*E[K_qp] + E[K_pp]
        mmd_sq = (K_qq.mean() - 2 * K_qp.mean() + K_pp.mean()).clamp(min=0)
        
        return torch.sqrt(mmd_sq + 1e-8)

    def compute_adversarial_loss(
        self,
        sampled_trajectory: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Adversarial loss: discriminator tries to distinguish sampled from expert.
        
        Args:
            sampled_trajectory: (B, T*state_dim) trajectory
            
        Returns:
            (gen_loss, dis_loss): Generator and discriminator losses
        """
        if sampled_trajectory.dim() == 1:
            sampled_trajectory = sampled_trajectory.unsqueeze(0)
        
        # Get reference trajectories
        if self.memory_bank_full:
            ref_trajs = self.memory_bank
        else:
            ptr = int(self.memory_bank_ptr)
            ref_trajs = self.memory_bank[:ptr]
        
        if ref_trajs.shape[0] == 0:
            return (
                torch.zeros(1, device=sampled_trajectory.device)[0],
                torch.zeros(1, device=sampled_trajectory.device)[0],
            )
        
        # Downsample reference to batch size for stability
        if ref_trajs.shape[0] > sampled_trajectory.shape[0]:
            idx = torch.randperm(ref_trajs.shape[0])[:sampled_trajectory.shape[0]]
            ref_trajs = ref_trajs[idx]
        
        q_z = self.encoder(sampled_trajectory)  # (B, 128)
        p_z = self.encoder(ref_trajs).detach()  # (B, 128)
        
        # Discriminator predictions
        logits_q = self.discriminator(q_z)  # (B, 1)
        logits_p = self.discriminator(p_z)  # (B, 1)
        
        # Generator loss (fool discriminator)
        gen_loss = torch.nn.functional.softplus(-logits_q).mean()
        
        # Discriminator loss (classify real vs fake)
        dis_loss = (
            torch.nn.functional.softplus(-logits_p).mean() +
            torch.nn.functional.softplus(logits_q).mean()
        )
        
        return gen_loss, dis_loss

    def forward(
        self,
        sampled_trajectory: torch.Tensor,
        requires_grad: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute drift loss for trajectory.
        
        Args:
            sampled_trajectory: (B, T*state_dim) or (T*state_dim,) tensor
            requires_grad: Whether loss should track gradients (for ODE guidance)
            
        Returns:
            dict with keys:
                'loss': scalar loss value
                'loss_raw': raw (unweighted) loss
                'gen_loss': (adversarial only) generator loss
                'dis_loss': (adversarial only) discriminator loss
        """
        if sampled_trajectory.dim() == 1:
            sampled_trajectory = sampled_trajectory.unsqueeze(0)
        
        result = {}
        
        if self.loss_type == "kl_divergence":
            loss = self.compute_kl_divergence(sampled_trajectory)
            result['loss'] = loss
            result['loss_raw'] = loss.detach()
            
        elif self.loss_type == "mmd":
            loss = self.compute_mmd_loss(sampled_trajectory)
            result['loss'] = loss
            result['loss_raw'] = loss.detach()
            
        elif self.loss_type == "adversarial":
            gen_loss, dis_loss = self.compute_adversarial_loss(sampled_trajectory)
            result['loss'] = gen_loss
            result['gen_loss'] = gen_loss
            result['dis_loss'] = dis_loss
            result['loss_raw'] = gen_loss.detach()
        
        return result

    def get_gradient(
        self,
        trajectory: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradient of drift loss with respect to trajectory.
        Used to guide ODE integration: dx/dt += lambda * grad_loss.
        
        Args:
            trajectory: (T*state_dim,) or (B, T*state_dim) tensor (requires grad)
            
        Returns:
            Gradient tensor same shape as input
        """
        trajectory_req = trajectory.clone().detach().requires_grad_(True)
        loss_dict = self.forward(trajectory_req, requires_grad=True)
        loss = loss_dict['loss']
        
        # Backprop to get gradient
        loss.backward()
        
        grad = trajectory_req.grad
        return grad if grad is not None else torch.zeros_like(trajectory)
