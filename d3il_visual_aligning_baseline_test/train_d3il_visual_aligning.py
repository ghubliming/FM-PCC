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

    use_wandb = cfg.get("use_wandb", True)
    # throw_on_missing=False so ??? placeholders in vision config don't crash
    cfg_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=False)
    _wb = cfg_dict.get('wandb', {})
    entity  = _wb.get('entity')  if _wb.get('entity')  not in (None, '???') else None
    project = _wb.get('project') if _wb.get('project') not in (None, '???') else 'd3il-baseline'
    wandb.init(
        project=project,
        entity=entity,
        group=cfg_dict.get('group', f'aligning_{cfg_dict.get("agent_name", "")}'),
        mode="online" if use_wandb else "disabled",
        config=cfg_dict,
    )

    print(f"[ train ] agent={cfg.agent_name}  seed={cfg.seed}  "
          f"epochs={cfg.epoch}  save_dir={os.getcwd()}")

    agent = hydra.utils.instantiate(cfg.agents)

    # Vision agents have train_vision_agent(); state agents use train_agent().
    # Detect via the policy's visual_input flag so no external flag is needed.
    _is_visual = getattr(getattr(agent, 'model', None), 'visual_input', False)
    if _is_visual and hasattr(agent, 'train_vision_agent'):
        print(f'[ train ] visual path → train_vision_agent()')
        agent.train_vision_agent()
    else:
        print(f'[ train ] state path → train_agent()')
        agent.train_agent()

    print(f"[ train ] done — weights in {agent.working_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
