import torch
import torch.nn as nn
from ddpm_encdec_vision.models.visual_unet import VisualUNet as ParentVisualUNet
from diffuser_visual_aligning.models.unet1d_temporal_cond import UNet1DTemporalCondModel

class VisualUNet(ParentVisualUNet):
    """
    Subclass of ddpm_encdec_vision VisualUNet that swaps out the backbone 
    with diffuser_visual_aligning.models.UNet1DTemporalCondModel.
    """
    def __init__(self, config):
        nn.Module.__init__(self)
        self.device = getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")
        self.if_vision = getattr(config, "if_vision", True)
        
        # 1. Instantiate multi-modal observation encoder from parent config
        if self.if_vision:
            from omegaconf import OmegaConf
            import hydra
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
            latent_dim = 128
        else:
            self.obs_encoder = None
            latent_dim = 0
        
        # 2. Setup horizon auto-padding constraints
        self.target_horizon = config.horizon
        self.padded_horizon = ((self.target_horizon + 7) // 8) * 8

        # 3. Dynamic transition dimension supporting both visual (3D) and non-visual (20D) states
        obs_dim = getattr(config, 'obs_dim', 3 if self.if_vision else 20)
        transition_dim = config.action_dim + obs_dim

        # 4. Instantiate local backbone from diffuser_visual_aligning package
        self.backbone = UNet1DTemporalCondModel(
            horizon=self.padded_horizon,
            transition_dim=transition_dim,
            cond_dim=latent_dim,
            dim=getattr(config, "dim", 128),
            dim_mults=getattr(config, "dim_mults", (1, 2, 4, 8)),
            returns_condition=getattr(config, "returns_condition", False),
            condition_dropout=getattr(config, "condition_dropout", 0.1),
            use_cond_projection=self.if_vision,
        ).to(self.device)
