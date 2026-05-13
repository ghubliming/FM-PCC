import torch
import torch.nn as nn
from omegaconf import OmegaConf
import hydra
import sys
import os

# Ensure d3il is in path if not already
sys.path.append(os.path.abspath('d3il'))

from agents.utils.scaler import Scaler
from environments.dataset.aligning_dataset import Aligning_Dataset

class VisualDiffusionBridge(nn.Module):
    """Bridges D3IL's visual DDPM into FM-PCC's engine structure."""
    
    def __init__(self, config):
        super().__init__()
        
        self.device = getattr(config, "device", "cuda" if torch.cuda.is_available() else "cpu")
        self.window_size = getattr(config, "window_size", 8)
        
        # Construct the shape meta for the obs_encoder
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
        
        # Diffusion core
        model_cfg = OmegaConf.create({
            "_target_": "agents.models.diffusion.diffusion_policy.Diffusion",
            "_recursive_": False,
            "state_dim": 128,
            "action_dim": 3,
            "beta_schedule": 'cosine',
            "n_timesteps": 16,
            "loss_type": 'l2',
            "clip_denoised": True,
            "predict_epsilon": True,
            "device": str(self.device),
            "diffusion_x": False,
            "diffusion_x_M": 10,
            "model": {
                "_target_": "agents.models.diffusion.diffusion_models.DiffusionEncDec",
                "_recursive_": False,
                "state_dim": 128,
                "action_dim": 3,
                "goal_conditioned": False,
                "goal_seq_len": 10,
                "obs_seq_len": 5,
                "action_seq_len": 4,
                "embed_pdrob": 0,
                "embed_dim": 64,
                "device": str(self.device),
                "linear_output": True,
                "encoder": {
                    "_target_": "agents.models.act.act_vae.TransformerEncoder",
                    "embed_dim": 64,
                    "n_heads": 4,
                    "n_layers": 2,
                    "attn_pdrop": 0.1,
                    "resid_pdrop": 0.1,
                    "bias": False,
                    "block_size": self.window_size + 1
                },
                "decoder": {
                    "_target_": "agents.models.act.act_vae.TransformerDecoder",
                    "embed_dim": 64,
                    "cross_embed": 64,
                    "n_heads": 4,
                    "n_layers": 4,
                    "attn_pdrop": 0.1,
                    "resid_pdrop": 0.1,
                    "bias": False,
                    "block_size": self.window_size + 1
                }
            }
        })
        
        self.obs_encoder = hydra.utils.instantiate(obs_encoder_cfg).to(self.device)
        self.diffusion_model = hydra.utils.instantiate(model_cfg).to(self.device)

        self._set_action_bounds(config)

    def _set_action_bounds(self, config):
        """Initialize diffusion clamp bounds from the aligning training dataset."""
        try:
            train_data_path = getattr(config, "train_data_path", None)
            if train_data_path is None:
                raise ValueError("Missing train_data_path in visual config")

            dataset = Aligning_Dataset(
                data_directory=train_data_path,
                device="cpu",
                obs_dim=20,
                action_dim=self.diffusion_model.action_dim,
                max_len_data=getattr(config, "max_len_data", 512),
                window_size=getattr(config, "window_size", 8),
            )
            scaler = Scaler(
                dataset.get_all_observations(),
                dataset.get_all_actions(),
                getattr(config, "scale_data", True),
                self.device,
            )
            self.diffusion_model.min_action = torch.from_numpy(scaler.y_bounds[0, :]).to(self.device)
            self.diffusion_model.max_action = torch.from_numpy(scaler.y_bounds[1, :]).to(self.device)
        except Exception:
            # Keep inference alive if the dataset-derived bounds cannot be built.
            default_bounds = torch.tensor([-0.01, -0.01, -0.01], device=self.device)
            self.diffusion_model.min_action = default_bounds
            self.diffusion_model.max_action = -default_bounds
        
    def encode_visual(self, bp_imgs, inhand_imgs, state=None):
        """[B,T,3,96,96] x 2 -> [B,T,128]"""
        B, T, C, H, W = bp_imgs.size()
        
        bp_imgs = bp_imgs.view(B * T, C, H, W)
        inhand_imgs = inhand_imgs.view(B * T, C, H, W)
        
        obs_dict = {
            "agentview_image": bp_imgs,
            "in_hand_image": inhand_imgs,
        }
        if state is not None:
            obs_dict["robot_ee_pos"] = state.view(B * T, -1)
            
        features = self.obs_encoder(obs_dict)
        features = features.view(B, T, -1)
        return features

    def loss(self, bp_imgs, inhand_imgs, obs, act, mask):
        """Training loss using D3IL's DDPM."""
        visual_emb = self.encode_visual(bp_imgs, inhand_imgs, state=None)
        # diffusion_model.loss expects (action, state, goal, weights)
        # However, D3IL's Diffusion.loss expects (x, state, goal, weights)
        # We pass visual_emb as state.
        loss_val = self.diffusion_model.loss(act, visual_emb, goal=None)
        # Return loss and an empty info dict (Trainer expects loss, info)
        return loss_val, {}

    def predict(self, visual_emb):
        """Inference using D3IL's DDPM."""
        return self.diffusion_model(visual_emb, None)
