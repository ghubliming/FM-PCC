"""Gen13 U11 — Mix-ML training entry (additive sibling of run/train_imf.py).

ONE trainer for all three MLbones. Identical to run/train_imf.py in every respect
(TemporalImfUnet backbone, cosine LR over the full budget, EMA, grad-clip + pre-clip
grad_norm logging, final checkpoint, CSV/W&B) EXCEPT the matcher is chosen by
`--ml_type` via ml.build_matcher:

    imf -> ImfMatcher (frozen; byte-identical to run/train_imf.py — gate G0)
    mf  -> MfMatcher  (Gen3v6 MeanFlow: analytic-v JVP tangent)
    af  -> AfMatcher  (Gen3v7 α-Flow: bootstrapped α:1->0 anneal)

run/train_imf.py stays the untouched canonical iMF entry; this file never edits it.

EVAL is objective-agnostic — MF/AF checkpoints are evaluated by the existing
run/eval_imf.py (it loads this same TemporalImfUnet + ImfFlowPolicy). No eval_ml.py.
"""

import csv
import os
import sys
from itertools import cycle

import torch
import tqdm
import tyro

from hardflow.datasets.sequence import SequenceDataset
from hardflow.models_flow.imf import TemporalImfUnet
from hardflow.models_flow.ml import MlTrainingConfig, build_matcher
from hardflow.utils.arrays import batch_to_device
from run.utils import deterministic, save_config, set_cuda_visible_device

from torch.utils.data import DataLoader

try:  # optional (only training uses it; keep the eval path dependency-free)
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

# fix_4 (inherited): off-tty (SLURM) => no tqdm bar; periodic prints carry the signal.
_IS_TTY = sys.stdout.isatty()


def sanitize_wandb_env():
    """U9 — clear malformed W&B service tokens (copied from run/train_imf.py)."""
    for env_key in ("WANDB_SERVICE", "WANDB__SERVICE"):
        token = os.environ.get(env_key)
        if token and len(token.split("-")) != 5:
            print(f"[ train_ml ] clearing malformed {env_key}")
            os.environ.pop(env_key, None)


def init_wandb(cfg, log_subfolder):
    """U9 — best-effort W&B init. NEVER let logging kill a training run."""
    if not getattr(cfg, "use_wandb", False):
        return None
    sanitize_wandb_env()
    try:
        import wandb
        slurm_suffix = f"-slurm-{os.environ['SLURM_JOB_ID']}" if os.environ.get('SLURM_JOB_ID') else ''
        run = wandb.init(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity or None,
            group=cfg.wandb_group or None,
            name=f"HF-ML-{cfg.ml_type}-{cfg.exp_name}-seed{cfg.seed}{slurm_suffix}",
            config=vars(cfg),
            dir=log_subfolder,
            reinit=True,
        )
        print(f"[ train_ml ] W&B online: project={cfg.wandb_project} run={run.name}")
        return run
    except Exception as e:                                   # noqa: BLE001
        print(f"[ train_ml ] W&B init FAILED ({e}) — continuing with CSV only")
        return None


def train(cfg: MlTrainingConfig, log_subfolder: str):
    print(f"[ train_ml ] ml_type={cfg.ml_type}  exp_name={cfg.exp_name}")

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

    # Shared backbone across all MLbones (U11 holds architecture constant).
    unet = TemporalImfUnet(
        horizon=cfg.horizon,
        transition_dim=cfg.state_dim + cfg.action_dim,
        cond_dim=cfg.state_dim,
        dim=32,
        dim_mults=(1, 4, 8),
    ).to(cfg.device)

    matcher = build_matcher(cfg, unet)

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
        print("[ train_ml ] tensorboard not installed -> metrics.csv only")

    wandb_run = init_wandb(cfg, log_subfolder)

    # family-aware metrics: AF adds α-schedule telemetry (gate G4). grad_norm last.
    metric_keys = ["loss", "raw_mse_u", "raw_mse_v", "a0_mse", "fm_frac", "h_mean"]
    if cfg.ml_type == "af":
        metric_keys += ["alpha", "discrete_frac", "clamp_frac"]
    metric_keys += ["grad_norm"]

    metrics_path = os.path.join(log_subfolder, "metrics.csv")
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["step"] + metric_keys)

    step_iter = (
        tqdm.tqdm(range(cfg.n_train_steps))
        if _IS_TTY
        else range(cfg.n_train_steps)
    )
    if not _IS_TTY:
        print(
            f"[ train_ml ] non-tty (batch) mode: progress bar disabled, "
            f"logging every {cfg.log_freq} steps of {cfg.n_train_steps}",
            flush=True,
        )

    # AF only: the α anneal is driven by the optimizer step. No-op for imf/mf.
    has_step = hasattr(matcher, "set_step")

    for i in step_iter:
        if has_step:
            matcher.set_step(i)
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
            if wandb_run is not None:
                try:
                    wandb_run.log({k: infos[k] for k in metric_keys}, step=i)
                except Exception as e:                     # noqa: BLE001
                    print(f"[ train_ml ] W&B log failed ({e}); disabling")
                    wandb_run = None
            alpha_str = f"  alpha {infos['alpha']:.3f}" if cfg.ml_type == "af" else ""
            msg = (
                f"[ train_ml:{cfg.ml_type} ] step {i}  raw_mse_u {infos['raw_mse_u']:.4f}  "
                f"raw_mse_v {infos['raw_mse_v']:.4f}  a0 {infos['a0_mse']:.5f}  "
                f"gnorm {infos['grad_norm']:.2f}{alpha_str}  "
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
    if wandb_run is not None:
        try:
            wandb_run.finish()
        except Exception:                                  # noqa: BLE001
            pass
    print(f"[ train_ml ] done. final checkpoints saved with cp index {final_cp}")


if __name__ == "__main__":
    cfg = tyro.cli(MlTrainingConfig)

    assert cfg.ml_type in ("imf", "mf", "af"), (
        f"--ml_type must be imf|mf|af, got {cfg.ml_type!r}"
    )

    set_cuda_visible_device(cfg)
    deterministic(cfg.seed)

    log_subfolder = os.path.join(cfg.log_folder, cfg.env, "flow", cfg.exp_name)
    save_config(cfg, log_subfolder)

    train(cfg, log_subfolder)
