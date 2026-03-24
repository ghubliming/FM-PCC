# Example script to train Flow Matching model using existing project Trainer

from flow_matcher.models.diffusion import GaussianDiffusion
from flow_matcher.models.unet1d_temporal_cond import UNet1DTemporalCondModel
from flow_matcher.utils.training import Trainer
# Import your dataset and config as needed

# Example: minimal dummy dataset (replace with your real dataset)
import torch
from torch.utils.data import Dataset

class DummyDataset(Dataset):
    def __init__(self, n=100, obs_dim=4, act_dim=2, horizon=8):
        self.data = torch.randn(n, horizon, obs_dim + act_dim)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        x = self.data[idx]
        cond = {0: x[0, -2:]}  # Example: condition on first action
        return x, cond, None  # (x, cond, returns)

# Model and training setup
obs_dim = 4
act_dim = 2
horizon = 8
model = UNet1DTemporalCondModel(
    input_dim=obs_dim + act_dim,
    horizon=horizon,
    cond_dim=obs_dim,
    dim=64,
    dim_mults=(1, 2, 4),
)
diffusion = GaussianDiffusion(
    model=model,
    horizon=horizon,
    observation_dim=obs_dim,
    action_dim=act_dim,
    n_timesteps=100,
)
dataset = DummyDataset(n=100, obs_dim=obs_dim, act_dim=act_dim, horizon=horizon)
trainer = Trainer(
    diffusion_model=diffusion,
    dataset=dataset,
    n_train_steps=100,
    train_batch_size=8,
    train_lr=1e-3,
    train_device='cpu',
)

if __name__ == "__main__":
    trainer.train_epoch(n_train_steps=10)
    print("FM training test completed.")
