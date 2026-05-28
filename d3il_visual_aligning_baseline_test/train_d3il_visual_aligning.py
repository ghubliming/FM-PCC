"""
D3IL Visual Aligning — training wrapper.

Mirrors d3il/run_vision.py for the aligning task but:
  - config_path points directly at d3il/configs/ (relative to this file)
  - vision agents: outer epoch loop with val-loss checkpoint saving
    (run_vision.py uses simulation success rate; we use val loss instead
     to avoid spinning up MuJoCo during training)
  - state agents: delegates to agent.train_agent() which has its own epoch loop
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

import logging
import os
import random

import hydra
import numpy as np
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("add", lambda *numbers: sum(numbers))
torch.cuda.empty_cache()


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def _train_vision(agent) -> None:
    """
    Epoch loop for vision agents, mirroring d3il/run_vision.py.
    Uses validation loss for checkpoint selection instead of simulation
    (avoids MuJoCo during training; state agents use the same approach via train_agent()).
    """
    best_val_loss = float("inf")

    for num_epoch in tqdm(range(agent.epoch)):
        agent.train_vision_agent()

        if not (num_epoch + 1) % agent.eval_every_n_epochs:
            # Validation pass — iterate val dataloader like state agents do
            val_losses = []
            for data in agent.test_dataloader:
                bp_imgs, inhand_imgs, obs, action, mask = data
                bp_imgs   = bp_imgs.to(agent.device)
                inhand_imgs = inhand_imgs.to(agent.device)
                obs    = agent.scaler.scale_input(obs)
                action = agent.scaler.scale_output(action)
                # Slice obs/action the same way train_vision_agent does if needed
                if hasattr(agent, 'obs_seq_len'):
                    action  = action[:, agent.obs_seq_len - 1:, :].contiguous()
                    obs     = obs[:, :agent.obs_seq_len].contiguous()
                    bp_imgs = bp_imgs[:, :agent.obs_seq_len].contiguous()
                    inhand_imgs = inhand_imgs[:, :agent.obs_seq_len].contiguous()
                state = (bp_imgs, inhand_imgs, obs)
                val_loss = agent.evaluate(state, action)
                val_losses.append(val_loss)

            avg_val = sum(val_losses) / len(val_losses)
            log.info(f"Epoch {num_epoch}: val loss = {avg_val:.6f}")
            wandb.log({"val_loss": avg_val, "epoch": num_epoch})

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                agent.store_model_weights(agent.working_dir, sv_name=agent.eval_model_name)
                log.info("New best val loss — checkpoint saved.")
                wandb.log({"best_model_epoch": num_epoch})

    agent.store_model_weights(agent.working_dir, sv_name=agent.last_model_name)
    log.info("Training done!")


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

    # Vision agents need the outer epoch loop (train_vision_agent = 1 epoch only).
    # State agents use train_agent() which has its own complete epoch loop.
    _is_visual = getattr(getattr(agent, 'model', None), 'visual_input', False)
    if _is_visual and hasattr(agent, 'train_vision_agent'):
        print(f'[ train ] visual path → _train_vision() loop x{cfg.epoch} epochs')
        _train_vision(agent)
    else:
        print(f'[ train ] state path → train_agent()')
        agent.train_agent()

    print(f"[ train ] done — weights in {agent.working_dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
