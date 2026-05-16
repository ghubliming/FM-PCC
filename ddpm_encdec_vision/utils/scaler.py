import logging
import numpy as np
import torch
import einops

log = logging.getLogger(__name__)

class Scaler:
    """
    Standardizes inputs/outputs for Diffusion models.
    Essential for matching the Signal-to-Noise Ratio (SNR) expected by the U-Net.
    """
    def __init__(self, x_data: np.ndarray, y_data: np.ndarray, scale_data: bool, device: str):
        self.scale_data = scale_data
        self.device = device
        
        if isinstance(x_data, torch.Tensor):
            x_data = x_data.detach().cpu().numpy()
            y_data = y_data.detach().cpu().numpy()
            
        # check the length and rearrange if required
        if len(x_data.shape) == 2:
            pass
        elif len(x_data.shape) == 3:
            x_data = einops.rearrange(x_data, "s t x -> (s t) x")
            y_data = einops.rearrange(y_data, "s t x -> (s t) x")
        
        # Calculate stats
        self.x_mean = torch.from_numpy(x_data.mean(0)).float().to(device)
        self.x_std = torch.from_numpy(x_data.std(0)).float().to(device)
        self.y_mean = torch.from_numpy(y_data.mean(0)).float().to(device)
        self.y_std = torch.from_numpy(y_data.std(0)).float().to(device)
        
        # --- ZERO VARIANCE SAFETY (FIX #29) ---
        # If a dimension is constant, std is 0. Dividing by 0 causes 10^10 drift.
        x_const = self.x_std < 1e-4
        y_const = self.y_std < 1e-4
        if x_const.any() or y_const.any():
            log.warning(f'[ Scaler ] Detected constant dimensions! Fixing std=1.0 for indices: X:{torch.where(x_const)[0].cpu().numpy()}, Y:{torch.where(y_const)[0].cpu().numpy()}')
        
        self.x_std[x_const] = 1.0
        self.y_std[y_const] = 1.0
        # --------------------------------------
        
        # Bounds for clipping
        self.y_min = torch.from_numpy(y_data.min(0)).float().to(device)
        self.y_max = torch.from_numpy(y_data.max(0)).float().to(device)
        
        log.info(f'[ Scaler ] Initialized with scale_data={scale_data}')
        log.info(f'[ Scaler ] x_mean: {self.x_mean.cpu().numpy()}')
        log.info(f'[ Scaler ] y_mean: {self.y_mean.cpu().numpy()}')
        log.info(f'[ Scaler ] x_std: {self.x_std.cpu().numpy()}')
        log.info(f'[ Scaler ] y_std: {self.y_std.cpu().numpy()}')

    @torch.no_grad()
    def scale_input(self, x):
        if not self.scale_data:
            return x.to(self.device).float()
        
        x = x.to(self.device).float()
        # Handle sequence padding or varying shapes by broadcasting mean/std
        out = (x - self.x_mean) / (self.x_std + 1e-12)
        return out

    @torch.no_grad()
    def scale_output(self, y):
        if not self.scale_data:
            return y.to(self.device).float()
            
        y = y.to(self.device).float()
        out = (y - self.y_mean) / (self.y_std + 1e-12)
        return out

    @torch.no_grad()
    def inverse_scale_output(self, y):
        if not self.scale_data:
            return y.to(self.device).float()
            
        y = y.to(self.device).float()
        out = y * (self.y_std + 1e-12) + self.y_mean
        return out

    @torch.no_grad()
    def clip_action(self, y):
        # Optional safety clipping based on training data bounds
        return torch.clamp(y, self.y_min, self.y_max).float()
