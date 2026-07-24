from itertools import cycle
import os
import torch
import tyro

from hardflow.config.flow_matching import FlowMatchingTrainingConfig
from hardflow.datasets.sequence import SequenceDataset
from hardflow.models_flow.flow_matcher import FlowMatcher
from hardflow.models_flow.unet import TemporalUnet
from hardflow.utils.arrays import batch_to_device
from run.utils import deterministic, set_cuda_visible_device, save_config
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
import tqdm


def train(cfg: FlowMatchingTrainingConfig, log_subfolder: str):
    dataset = SequenceDataset(
        env=cfg.env,
        horizon=cfg.horizon,
        normalizer=cfg.normalizer,
        preprocess_fns=cfg.preprocess_fns,
        max_path_length=cfg.max_path_length,
        max_n_episodes=cfg.max_n_episodes,
        termination_penalty=0,
        seed=cfg.seed,
    )

    unet = TemporalUnet(
        horizon=cfg.horizon,
        transition_dim=cfg.state_dim + cfg.action_dim,
        cond_dim=cfg.state_dim,
        dim=32,
        dim_mults=(1, 4, 8),
    ).to(cfg.device)

    flow_matcher = FlowMatcher(
        action_dim=cfg.action_dim,
        model=unet,
        flow_matching_type=cfg.flow_matching_type,
    )

    train_loader = cycle(DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True))

    optimizer = torch.optim.Adam(unet.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.save_freq * 2
    )
    ema = torch.optim.swa_utils.AveragedModel(
        unet,
        avg_fn=lambda avg, new, num: cfg.ema_decay * avg + (1 - cfg.ema_decay) * new,
    )

    writer = SummaryWriter(log_dir=os.path.join(log_subfolder, "tensorboard_logs"))

    for i in tqdm.tqdm(range(cfg.n_train_steps)):
        batch = batch_to_device(
            next(train_loader), cfg.device
        )  # (batch_size, planning_horizon, transition_dim)
        loss, infos = flow_matcher.loss(*batch)
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        ema.update_parameters(unet)

        if i % cfg.save_freq == 0:
            torch.save(
                unet.state_dict(),
                os.path.join(log_subfolder, f"model_{i // cfg.save_freq}.pth"),
            )
            torch.save(
                ema.module.state_dict(),
                os.path.join(log_subfolder, f"model_ema_{i // cfg.save_freq}.pth"),
            )
        writer.add_scalar("loss", loss, i)


if __name__ == "__main__":
    cfg = tyro.cli(FlowMatchingTrainingConfig)

    set_cuda_visible_device(cfg)
    deterministic(cfg.seed)  # seed everything

    log_subfolder = os.path.join(cfg.log_folder, cfg.env, "flow", cfg.exp_name)
    save_config(cfg, log_subfolder)

    train(cfg, log_subfolder)
