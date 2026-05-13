"""
Diffusion Transformer (DiT) for iMeanFlow Trajectory Modeling

Optional Transformer-based backbone for improved trajectory sequence modeling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Encode time step into embedding.
        
        Args:
            t: (B,) or () time values in [0, 1]
        
        Returns:
            emb: (B, dim) embedding
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        
        B = t.shape[0]
        emb = torch.zeros(B, self.dim, device=t.device, dtype=t.dtype)
        
        # Sinusoidal embedding
        freq_bands = torch.arange(self.dim // 2, dtype=t.dtype, device=t.device)
        freq_bands = 2.0 ** (freq_bands / (self.dim // 2))
        
        t_scaled = t.unsqueeze(1) * freq_bands.unsqueeze(0)  # (B, D//2)
        
        emb[:, 0::2] = torch.sin(t_scaled)
        emb[:, 1::2] = torch.cos(t_scaled)
        
        return emb


class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention for trajectory sequences.
    
    Args:
        dim: Feature dimension
        heads: Number of attention heads
        head_dim: Dimension per head (default: dim // heads)
    """
    
    def __init__(self, dim: int, heads: int = 8, head_dim: Optional[int] = None):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.head_dim = head_dim or (dim // heads)
        self.scale = self.head_dim ** -0.5
        
        assert dim == heads * self.head_dim, "dim must be divisible by heads"
        
        self.to_qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            x: (B, T, D) sequence
            mask: (B, T) or (B, 1, T, T) attention mask
        
        Returns:
            out: (B, T, D) attended sequence
        """
        B, T, D = x.shape
        
        # Project to Q, K, V
        qkv = self.to_qkv(x)  # (B, T, 3D)
        qkv = qkv.reshape(B, T, 3, self.heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, T, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, T, T)
        
        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T)
            scores = scores.masked_fill(~mask, float('-inf'))
        
        attn = F.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=0.1, training=self.training)
        
        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, heads, T, head_dim)
        out = out.permute(0, 2, 1, 3).reshape(B, T, D)
        
        # Output projection
        out = self.out_proj(out)
        
        return out


class FeedForward(nn.Module):
    """Feed-forward network with residual connection."""
    
    def __init__(self, dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or 4 * dim
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Transformer block: LayerNorm → Attention → Residual → LayerNorm → FFN → Residual
    """
    
    def __init__(self, dim: int, heads: int = 8, ff_dim: Optional[int] = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, heads)
        
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, ff_dim)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply transformer block with residual connections.
        
        Args:
            x: (B, T, D) input
            mask: (B, T) attention mask
        
        Returns:
            out: (B, T, D) output
        """
        # Attention with residual
        x = x + self.attn(self.norm1(x), mask)
        
        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))
        
        return x


class ImfDiTTrajectory(nn.Module):
    """
    Diffusion Transformer for iMeanFlow trajectory generation.
    
    Architecture:
    1. Input embedding: state → latent
    2. Time embedding: time step → latent
    3. Transformer blocks: contextualize sequence
    4. Dual output heads: predict u and v velocities separately
    
    Args:
        state_dim: Dimensionality of state space
        latent_dim: Transformer latent dimension
        num_blocks: Number of transformer blocks
        num_heads: Number of attention heads
        time_dim: Time embedding dimension
        output_dim: Velocity field output dimension (usually = state_dim)
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 256,
        num_blocks: int = 4,
        num_heads: int = 8,
        time_dim: int = 128,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.time_dim = time_dim
        self.output_dim = output_dim or state_dim
        
        # Input embedding
        self.state_embed = nn.Linear(state_dim, latent_dim)
        
        # Time embedding
        self.time_embed = TimeEmbedding(time_dim)
        self.time_proj = nn.Sequential(
            nn.Linear(time_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(latent_dim, heads=num_heads)
            for _ in range(num_blocks)
        ])
        
        # Output layer norm
        self.norm_out = nn.LayerNorm(latent_dim)
        
        # Dual velocity heads
        self.u_head = nn.Linear(latent_dim, self.output_dim)
        self.v_head = nn.Linear(latent_dim, self.output_dim)
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through DiT.
        
        Args:
            x: (B, T, state_dim) trajectory sequence
            t: (B,) time steps in [0, 1]
            mask: (B, T) optional padding mask
        
        Returns:
            u: (B, T, output_dim) average velocity
            v: (B, T, output_dim) instantaneous velocity
        """
        B, T, D = x.shape
        
        # Embed state
        h = self.state_embed(x)  # (B, T, latent_dim)
        
        # Embed and project time
        t_emb = self.time_embed(t)  # (B, time_dim)
        t_proj = self.time_proj(t_emb)  # (B, latent_dim)
        
        # Add time conditioning to all positions
        t_cond = t_proj.unsqueeze(1)  # (B, 1, latent_dim)
        h = h + t_cond
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            h = block(h, mask)
        
        # Output normalization
        h = self.norm_out(h)
        
        # Dual velocity heads
        u = self.u_head(h)  # (B, T, output_dim)
        v = self.v_head(h)  # (B, T, output_dim)
        
        return u, v


class ImfDiTTrajectoryWithContext(nn.Module):
    """
    Enhanced DiT with additional context inputs (e.g., goal state, constraints).
    
    Args:
        state_dim: State dimension
        latent_dim: Transformer latent dimension
        num_blocks: Number of transformer blocks
        context_dim: Context feature dimension (default: state_dim)
    """
    
    def __init__(
        self,
        state_dim: int,
        latent_dim: int = 256,
        num_blocks: int = 4,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.context_dim = context_dim or state_dim
        
        # Base DiT
        self.dit = ImfDiTTrajectory(
            state_dim=state_dim,
            latent_dim=latent_dim,
            num_blocks=num_blocks,
        )
        
        # Context encoder
        if self.context_dim > 0:
            self.context_embed = nn.Sequential(
                nn.Linear(self.context_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim),
            )
        else:
            self.context_embed = None
    
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with optional context.
        
        Args:
            x: (B, T, state_dim) trajectory
            t: (B,) time
            context: (B, context_dim) optional context (e.g., goal)
            mask: (B, T) optional mask
        
        Returns:
            u: (B, T, state_dim) average velocity
            v: (B, T, state_dim) instantaneous velocity
        """
        # Get base predictions
        u, v = self.dit(x, t, mask)
        
        # Integrate context if provided
        if context is not None and self.context_embed is not None:
            context_encoded = self.context_embed(context)  # (B, latent_dim)
            context_scaled = context_encoded.unsqueeze(1) * 0.1  # (B, 1, latent_dim)
            
            # Modulate outputs by context
            u = u + context_scaled
            v = v + context_scaled
        
        return u, v


if __name__ == '__main__':
    # Test ImfDiTTrajectory
    batch_size = 4
    seq_len = 10
    state_dim = 28
    latent_dim = 128
    
    model = ImfDiTTrajectory(
        state_dim=state_dim,
        latent_dim=latent_dim,
        num_blocks=2,
    )
    
    x = torch.randn(batch_size, seq_len, state_dim)
    t = torch.rand(batch_size)
    
    u, v = model(x, t)
    print(f"Input: {x.shape}")
    print(f"u output: {u.shape}")
    print(f"v output: {v.shape}")
    
    # Test with mask
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    mask[:, -2:] = False  # Mask last 2 positions
    
    u_masked, v_masked = model(x, t, mask)
    print(f"u with mask: {u_masked.shape}")
    print(f"v with mask: {v_masked.shape}")
    
    # Test DiT with context
    model_ctx = ImfDiTTrajectoryWithContext(
        state_dim=state_dim,
        context_dim=state_dim,
    )
    
    context = torch.randn(batch_size, state_dim)
    u_ctx, v_ctx = model_ctx(x, t, context, mask)
    print(f"u with context: {u_ctx.shape}")
    print(f"v with context: {v_ctx.shape}")
