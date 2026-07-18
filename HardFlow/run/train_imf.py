"""Gen13 — iMF training entry (additive sibling of run/train.py, which is untouched).

Differences vs run/train.py:
  * TemporalImfUnet + ImfMatcher instead of TemporalUnet + FlowMatcher(cfm)
  * tensorboard is OPTIONAL (try-import; fix_2 lesson) — metrics always go to
    a metrics.csv next to the checkpoints regardless
  * cosine LR anneals over the FULL n_train_steps (Gen3v4 lesson: don't freeze
    on a noisy plateau), instead of T_max = save_freq * 2
  * a FINAL checkpoint is saved at the end (cp index n_train_steps//save_freq)
  * raw_mse_u / raw_mse_v / a0_mse are logged — judge convergence on these,
    NEVER on the adaptive `loss` (flat by construction).
"""

import csv
import os
from itertools import cycle

import torch
import tqdm
import tyro

from hardflow.datasets.sequence import SequenceDataset
from hardflow.models_flow.imf import ImfMatcher, ImfTrainingConfig, TemporalImfUnet
from hardflow.utils.arrays import batch_to_device
from run.utils import deterministic, save_config, set_cuda_visible_device

from torch.utils.data import DataLoader

try:  # optional (only training uses it; keep the eval path dependency-free)
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None


def train(cfg: ImfTrainingConfig, log_subfolder: str):
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

    unet = TemporalImfUnet(
        horizon=cfg.horizon,
        transition_dim=cfg.state_dim + cfg.action_dim,
        cond_dim=cfg.state_dim,
        dim=32,
        dim_mults=(1, 4, 8),
    ).to(cfg.device)

    matcher = ImfMatcher(
        model=unet,
        action_dim=cfg.action_dim,
        p_mean=cfg.imf_p_mean,
        p_std=cfg.imf_p_std,
        data_proportion=cfg.imf_data_proportion,
        adp_p=cfg.imf_adp_p,
        adp_eps=cfg.imf_adp_eps,
    )

    train_loader = cycle(DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True))

    optimizer = torch.optim.Adam(unet.parameters(), lr=cfg.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.n_train_steps
    )
    ema = torch.optim.swa_utils.AveragedModel(
        unet,
        avg_fn=lambda avg, new, num: cfg.ema_decay * avg + (1 - cfg.ema_decay) * new,
    )

    writer = None
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=os.path.join(log_subfolder, "tensorboard_logs"))
    else:
        print("[ train_imf ] tensorboard not installed -> metrics.csv only")

    metrics_path = os.path.join(log_subfolder, "metrics.csv")
    metric_keys = ["loss", "raw_mse_u", "raw_mse_v", "a0_mse", "fm_frac", "h_mean"]
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["step"] + metric_keys)

    for i in tqdm.tqdm(range(cfg.n_train_steps)):
        batch = batch_to_device(next(train_loader), cfg.device)
        loss, infos = matcher.loss(*batch)
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

        if i % cfg.log_freq == 0:
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([i] + [infos[k] for k in metric_keys])
            if writer is not None:
                for k in metric_keys:
                    writer.add_scalar(k, infos[k], i)
            tqdm.tqdm.write(
                f"[ train_imf ] step {i}  raw_mse_u {infos['raw_mse_u']:.4f}  "
                f"raw_mse_v {infos['raw_mse_v']:.4f}  a0 {infos['a0_mse']:.5f}  "
                f"(adaptive loss {infos['loss']:.4f} — flat by design)"
            )

    # final checkpoint (cp index = n_train_steps // save_freq)
    final_cp = cfg.n_train_steps // cfg.save_freq
    torch.save(
        unet.state_dict(), os.path.join(log_subfolder, f"model_{final_cp}.pth")
    )
    torch.save(
        ema.module.state_dict(),
        os.path.join(log_subfolder, f"model_ema_{final_cp}.pth"),
    )
    print(f"[ train_imf ] done. final checkpoints saved with cp index {final_cp}")


if __name__ == "__main__":
    cfg = tyro.cli(ImfTrainingConfig)

    set_cuda_visible_device(cfg)
    deterministic(cfg.seed)

    log_subfolder = os.path.join(cfg.log_folder, cfg.env, "flow", cfg.exp_name)
    save_config(cfg, log_subfolder)

    train(cfg, log_subfolder)
