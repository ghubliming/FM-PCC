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
  * (fix_4) NO tqdm progress bar under SLURM — see _IS_TTY below.
"""

import csv
import os
import sys
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

# fix_4: under `Slurm_Codes/submit.sh` stdout is a redirected FILE, not a tty, so
# tqdm's in-place carriage-return update never collapses — a 100k-step bar dumps
# every single update as raw text. First run (job 23579) produced a 4.6 MB log
# whose three longest lines were 791k/865k/742k chars (98% of the file). The
# periodic `[ train_imf ] step ...` lines below already carry all real signal,
# so the bar is simply disabled off-tty.
_IS_TTY = sys.stdout.isatty()



def sanitize_wandb_env():
    """U9 — clear malformed W&B service tokens (copied from FMPCC Gen3v4 iMF).

    A stale/partial WANDB_SERVICE token inherited from the environment makes
    wandb.init() hang or fail on the cluster. Gen3v4 hit this; same env here.
    """
    for env_key in ("WANDB_SERVICE", "WANDB__SERVICE"):
        token = os.environ.get(env_key)
        if token and len(token.split("-")) != 5:
            print(f"[ train_imf ] clearing malformed {env_key}")
            os.environ.pop(env_key, None)


def init_wandb(cfg, log_subfolder):
    """U9 — best-effort W&B init. NEVER let logging kill a 4-hour training run."""
    if not getattr(cfg, "use_wandb", False):
        return None
    sanitize_wandb_env()
    try:
        import wandb
        # Tag the run with the Slurm job id (empty off-cluster)
        slurm_suffix = f"-slurm-{os.environ['SLURM_JOB_ID']}" if os.environ.get('SLURM_JOB_ID') else ''
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            group=cfg.wandb_group or None,
            name=f"HF-iMF-{cfg.exp_name}-seed{cfg.seed}{slurm_suffix}",
            config=vars(cfg),
            dir=log_subfolder,
            reinit=True,
        )
        print(f"[ train_imf ] W&B online: project={cfg.wandb_project} run={run.name}")
        return run
    except Exception as e:                                   # noqa: BLE001
        print(f"[ train_imf ] W&B init FAILED ({e}) — continuing with CSV only")
        return None


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

    wandb_run = init_wandb(cfg, log_subfolder)

    metrics_path = os.path.join(log_subfolder, "metrics.csv")
    metric_keys = ["loss", "raw_mse_u", "raw_mse_v", "a0_mse", "fm_frac", "h_mean", "grad_norm"]
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["step"] + metric_keys)

    # fix_4: live bar ONLY on an interactive terminal; plain range() under SLURM.
    step_iter = (
        tqdm.tqdm(range(cfg.n_train_steps))
        if _IS_TTY
        else range(cfg.n_train_steps)
    )
    if not _IS_TTY:
        print(
            f"[ train_imf ] non-tty (batch) mode: progress bar disabled, "
            f"logging every {cfg.log_freq} steps of {cfg.n_train_steps}",
            flush=True,
        )

    for i in step_iter:
        batch = batch_to_device(next(train_loader), cfg.device)
        loss, infos = matcher.loss(*batch)
        loss.backward()
        # U9.2: clip + record the pre-clip grad norm (the instability metric)
        gn = torch.nn.utils.clip_grad_norm_(
            unet.parameters(),
            cfg.grad_clip if cfg.grad_clip > 0 else float("inf"),
        )
        infos["grad_norm"] = float(gn)
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
            if wandb_run is not None:                      # U9
                try:
                    wandb_run.log({k: infos[k] for k in metric_keys}, step=i)
                except Exception as e:                     # noqa: BLE001
                    print(f"[ train_imf ] W&B log failed ({e}); disabling")
                    wandb_run = None
            msg = (
                f"[ train_imf ] step {i}  raw_mse_u {infos['raw_mse_u']:.4f}  "
                f"raw_mse_v {infos['raw_mse_v']:.4f}  a0 {infos['a0_mse']:.5f}  "
                f"gnorm {infos['grad_norm']:.2f}  "
                f"(adaptive loss {infos['loss']:.4f} — flat by design)"
            )
            if _IS_TTY:
                tqdm.tqdm.write(msg)
            else:
                print(msg, flush=True)

    # final checkpoint (cp index = n_train_steps // save_freq)
    final_cp = cfg.n_train_steps // cfg.save_freq
    torch.save(
        unet.state_dict(), os.path.join(log_subfolder, f"model_{final_cp}.pth")
    )
    torch.save(
        ema.module.state_dict(),
        os.path.join(log_subfolder, f"model_ema_{final_cp}.pth"),
    )
    if wandb_run is not None:                              # U9
        try:
            wandb_run.finish()
        except Exception:                                  # noqa: BLE001
            pass
    print(f"[ train_imf ] done. final checkpoints saved with cp index {final_cp}")


if __name__ == "__main__":
    cfg = tyro.cli(ImfTrainingConfig)

    set_cuda_visible_device(cfg)
    deterministic(cfg.seed)

    log_subfolder = os.path.join(cfg.log_folder, cfg.env, "flow", cfg.exp_name)
    save_config(cfg, log_subfolder)

    train(cfg, log_subfolder)
