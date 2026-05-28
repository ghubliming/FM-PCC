"""
D3IL Visual Aligning — training wrapper.

Mirrors d3il/run.py for the aligning task but:
  - config_path points directly at d3il/configs/ (relative to this file)
  - trains only; no end-of-train env_sim.test_agent() call
  - hydra.run.dir is overridden from the SLURM script to a predictable path under
    logs/d3il_visual_aligning_baseline/{agent_name}/seed_{s}/weights/
    so weights land in the gitignored logs/ tree, never inside d3il/

Typical SLURM invocation:
    python d3il_visual_aligning_baseline_test/train_d3il_visual_aligning.py \
        "agents=ddpm_encdec_vision_agent" \
        "agent_name=ddpm_encdec_vision" \
        "seed=42" \
        "hydra.run.dir=logs/d3il_visual_aligning_baseline/ddpm_encdec_vision/seed_42/weights"
"""

import os
import random

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf

OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
torch.cuda.empty_cache()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


@hydra.main(config_path="../d3il/configs", config_name="aligning_vision_config")
def main(cfg: DictConfig) -> None:
    _set_seed(cfg.seed)

    wandb_mode = "online" if cfg.get("use_wandb", False) else "disabled"
    wandb.config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group=cfg.group,
        mode=wandb_mode,
        config=wandb.config,
    )

    print(f"[ train ] agent={cfg.agent_name}  seed={cfg.seed}  "
          f"epochs={cfg.epoch}  save_dir={os.getcwd()}")

    agent = hydra.utils.instantiate(cfg.agents)
    agent.train_agent()

    print(f"[ train ] done — weights in {agent.working_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
