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
        
        # Define safe standard deviations using the legacy stability floor
        self.x_std_safe = torch.clamp(self.x_std, min=1e-2)
        self.y_std_safe = torch.clamp(self.y_std, min=1e-2)
        
        # Bounds for clipping (Raw)
        self.y_min = torch.from_numpy(y_data.min(0)).float().to(device)
        self.y_max = torch.from_numpy(y_data.max(0)).float().to(device)
        
        # --- API COMPATIBILITY LAYER (D3IL Parity) ---
        # D3IL models expect self.y_bounds to be the [min, max] of the SCALED data.
        self.y_bounds = np.zeros((2, y_data.shape[-1]))
        if self.scale_data:
            # We use the raw min/max but scale them using our Safe Std
            y_min_np = y_data.min(0)
            y_max_np = y_data.max(0)
            y_mean_np = self.y_mean.cpu().numpy()
            y_std_safe_np = self.y_std_safe.cpu().numpy()
            
            self.y_bounds[0, :] = (y_min_np - y_mean_np) / y_std_safe_np
            self.y_bounds[1, :] = (y_max_np - y_mean_np) / y_std_safe_np
        else:
            self.y_bounds[0, :] = y_data.min(0)
            self.y_bounds[1, :] = y_data.max(0)
        # ---------------------------------------------

        log.info(f'[ Scaler ] Initialized with scale_data={scale_data}')
        log.info(f'[ Scaler ] x_mean: {self.x_mean.cpu().numpy()}')
        log.info(f'[ Scaler ] y_mean: {self.y_mean.cpu().numpy()}')
        log.info(f'[ Scaler ] x_std: {self.x_std.cpu().numpy()}')
        log.info(f'[ Scaler ] y_std: {self.y_std.cpu().numpy()}')
        log.info(f'[ Scaler ] x_std_safe min: {self.x_std_safe.min().item():.4f}')
        log.info(f'[ Scaler ] y_std_safe min: {self.y_std_safe.min().item():.4f}')

    @torch.no_grad()
    def scale_input(self, x):
        if not self.scale_data:
            return x.to(self.device).float()
        return (x.to(self.device).float() - self.x_mean) / self.x_std_safe

    @torch.no_grad()
    def scale_output(self, y):
        if not self.scale_data:
            return y.to(self.device).float()
        return (y.to(self.device).float() - self.y_mean) / self.y_std_safe

    @torch.no_grad()
    def inverse_scale_input(self, x):
        if not self.scale_data:
            return x.to(self.device).float()
        return x.to(self.device).float() * self.x_std_safe + self.x_mean

    @torch.no_grad()
    def inverse_scale_output(self, y):
        if not self.scale_data:
            return y.to(self.device).float()
        return y.to(self.device).float() * self.y_std_safe + self.y_mean

    @torch.no_grad()
    def clip_action(self, y):
        # Optional safety clipping based on training data bounds
        return torch.clamp(y, self.y_min, self.y_max).float()
