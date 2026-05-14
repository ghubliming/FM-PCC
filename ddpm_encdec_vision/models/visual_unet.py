import torch
import torch.nn as nn
from omegaconf import OmegaConf
import hydra
import os
import sys

# Ensure d3il is in path
sys.path.append(os.path.abspath('d3il'))

class VisualUNet(nn.Module):
    """
    Wraps a vision encoder and a temporal backbone.
    This acts as the 'model' inside the FMv3 engine.
    """
    def __init__(self, config):
        super().__init__()
        self.device = getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. Instantiate Vision Encoder
        shape_meta = {
            "obs": {
                "agentview_image": {"shape": [3, 96, 96], "type": "rgb"},
                "in_hand_image": {"shape": [3, 96, 96], "type": "rgb"}
            }
        }
        
        obs_encoder_cfg = OmegaConf.create({
            "_target_": "agents.models.vision.multi_image_obs_encoder.MultiImageObsEncoder",
            "shape_meta": shape_meta,
            "rgb_model": {
                "_target_": "agents.models.vision.model_getter.get_resnet",
                "input_shape": [3, 96, 96],
                "output_size": 64
            },
            "resize_shape": None,
            "random_crop": False,
            "use_group_norm": True,
            "share_rgb_model": False,
            "imagenet_norm": True
        })
        self.obs_encoder = hydra.utils.instantiate(obs_encoder_cfg).to(self.device)
        
        # 2. Instantiate Backbone (Standard DDPM UNet)
        from diffuser.models.unet1d_temporal_cond import UNet1DTemporalCondModel
        backbone_class = UNet1DTemporalCondModel
        
        # Calculate latent dim: encoder outputs 64*2 = 128 (default)
        latent_dim = 128
        
        self.backbone = backbone_class(
            horizon=config.horizon,
            transition_dim=config.action_dim + 3, # action + robot_pos
            cond_dim=latent_dim,
            dim=getattr(config, "dim", 128),
            dim_mults=getattr(config, "dim_mults", (1, 2, 4, 8)),
            returns_condition=getattr(config, "returns_condition", False),
            condition_dropout=getattr(config, "condition_dropout", 0.1),
            use_cond_projection=True,  # Enable FiLM conditioning for visual embeddings
        ).to(self.device)

    def encode_visual(self, bp_imgs, inhand_imgs, state=None):
        B, T, C, H, W = bp_imgs.size()
        bp_imgs = bp_imgs.view(B * T, C, H, W)
        inhand_imgs = inhand_imgs.view(B * T, C, H, W)
        
        obs_dict = {
            "agentview_image": bp_imgs,
            "in_hand_image": inhand_imgs,
        }
        if state is not None:
            # state is [B, T, 3]
            obs_dict["robot_ee_pos"] = state.view(B * T, -1)
            
        features = self.obs_encoder(obs_dict)
        features = features.view(B, T, -1)
        return features

    def forward(self, x, cond, t, returns=None, use_dropout=True, force_dropout=False):
        """
        x: [B, T, action_dim + state_dim]
        cond: dict containing 'visual' -> (bp_imgs, inhand_imgs, state)
        """
        if isinstance(cond, dict) and 'visual' in cond:
            bp_imgs, inhand_imgs, state = cond['visual']
        else:
            # Fallback for old code paths or if cond is already the tuple
            bp_imgs, inhand_imgs, state = cond
            
        visual_emb = self.encode_visual(bp_imgs, inhand_imgs, state=state)
        
        return self.backbone(x, visual_emb, t, returns=returns, use_dropout=use_dropout, force_dropout=force_dropout)
