"""Gen13 U9 — FM (HardFlow) training entry with W&B + CSV logging.

ADDITIVE sibling of `run/train.py`, which is pre-existing and stays untouched.
The TRAINING MATH IS IDENTICAL — same TemporalUnet, same FlowMatcher('cfm'),
same optimiser/scheduler/EMA/checkpoint cadence. **Only logging differs**, so a
checkpoint produced here is a faithful reproduction of `run/train.py`'s.

Why this file exists (two problems with the original, both logging-related):

 1. `run/train.py:12` imports `SummaryWriter` at MODULE level, un-guarded. In
    `hardflow_clone` (no tensorboard) FM training therefore **crashes after ~4 s**
    — exactly what killed pipeline job 23559 and forced the replication onto the
    downloaded checkpoint. Here tensorboard is a try-import.
 2. It logs a single scalar to TensorBoard only — no CSV, no W&B. So even a
    successful run left nothing comparable with Gen13's `metrics.csv`.

W&B pattern copied from FMPCC Gen3v4 iMF
(`FM_v3_imeanflow_test/train_flow_matching_v3_imeanflow.py`): `sanitize_wandb_env`,
best-effort `wandb.init` in try/except, and the `$HOME/FMPCC/.wandb_api_key`
convention set by the sbatch. Same wandb version (0.17.5) inherited from the
FMPCC clone.
"""

import csv
import os
import sys
from dataclasses import dataclass
from itertools import cycle

import torch
import tqdm
import tyro

from hardflow.config.flow_matching import FlowMatchingTrainingConfig
from hardflow.datasets.sequence import SequenceDataset
from hardflow.models_flow.flow_matcher import FlowMatcher
from hardflow.models_flow.unet import TemporalUnet
from hardflow.utils.arrays import batch_to_device
from run.utils import deterministic, save_config, set_cuda_visible_device

from torch.utils.data import DataLoader

try:  # U9 fix: was an un-guarded module-level import in run/train.py
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

_IS_TTY = sys.stdout.isatty()   # fix_4: no tqdm spam in batch logs


@dataclass
class FmTrainingConfig(FlowMatchingTrainingConfig):
    """FM training config + logging knobs (training params unchanged)."""

    log_freq: int = 200
    use_wandb: bool = False
    wandb_project: str = "FMPCC-HF-iMF"
    wandb_entity: str = ""
    wandb_group: str = ""


def sanitize_wandb_env():
    """Clear malformed W&B service tokens (copied from FMPCC Gen3v4 iMF)."""
    for env_key in ("WANDB_SERVICE", "WANDB__SERVICE"):
        token = os.environ.get(env_key)
        if token and len(token.split("-")) != 5:
            print(f"[ train_fm ] clearing malformed {env_key}")
            os.environ.pop(env_key, None)


def init_wandb(cfg, log_subfolder):
    """Best-effort W&B init — never let logging kill a long training run."""
    if not getattr(cfg, "use_wandb", False):
        return None
    sanitize_wandb_env()
    try:
        import wandb
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            group=cfg.wandb_group or None,
            name=f"HF-FM-{cfg.exp_name}-seed{cfg.seed}",
            config=vars(cfg),
            dir=log_subfolder,
            reinit=True,
        )
        print(f"[ train_fm ] W&B online: project={cfg.wandb_project} run={run.name}")
        return run
    except Exception as e:                                    # noqa: BLE001
        print(f"[ train_fm ] W&B init FAILED ({e}) — continuing with CSV only")
        return None


def train(cfg: FmTrainingConfig, log_subfolder: str):
    # ---- identical to run/train.py from here ---------------------------------
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
    # ---- end identical block -------------------------------------------------

    writer = None
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=os.path.join(log_subfolder, "tensorboard_logs"))
    else:
        print("[ train_fm ] tensorboard not installed -> metrics.csv only")

    wandb_run = init_wandb(cfg, log_subfolder)

    metrics_path = os.path.join(log_subfolder, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "loss"])

    step_iter = (
        tqdm.tqdm(range(cfg.n_train_steps))
        if _IS_TTY
        else range(cfg.n_train_steps)
    )
    if not _IS_TTY:
        print(
            f"[ train_fm ] non-tty (batch) mode: progress bar disabled, "
            f"logging every {cfg.log_freq} steps of {cfg.n_train_steps}",
            flush=True,
        )

    for i in step_iter:
        batch = batch_to_device(next(train_loader), cfg.device)
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

        if i % cfg.log_freq == 0:
            lv = float(loss.detach().cpu())
            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([i, lv])
            if writer is not None:
                writer.add_scalar("loss", lv, i)
            if wandb_run is not None:
                try:
                    wandb_run.log({"loss": lv}, step=i)
                except Exception as e:                        # noqa: BLE001
                    print(f"[ train_fm ] W&B log failed ({e}); disabling")
                    wandb_run = None
            msg = f"[ train_fm ] step {i}  loss {lv:.5f}"
            if _IS_TTY:
                tqdm.tqdm.write(msg)
            else:
                print(msg, flush=True)

    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:                                     # noqa: BLE001
            pass
    print("[ train_fm ] done.")


if __name__ == "__main__":
    cfg = tyro.cli(FmTrainingConfig)

    set_cuda_visible_device(cfg)
    deterministic(cfg.seed)

    log_subfolder = os.path.join(cfg.log_folder, cfg.env, "flow", cfg.exp_name)
    save_config(cfg, log_subfolder)

    train(cfg, log_subfolder)
