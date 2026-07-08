---
name: config-folder-convention
description: config/ convention (from DPCC) — .py files are train+eval setup entries, .yaml files are constraint-projection configs
metadata:
  type: project
---

`/workspaces/FM-PCC/config/` is normally where experiment setup lives (DPCC convention):

- **`.py` files** (e.g. `avoiding-d3il.py`, `aligning-d3il-visual.py`, `uav.py`) = train + eval configuration entries. Structured as a `base` dict with one block per experiment/generation (e.g. `'flow_matching_v3_imeanflow'`) plus `plan*` blocks for eval; scripts load a block via `Parser().parse_args(experiment=<block_name>)`.
- **`.yaml` files** (e.g. `projection_eval.yaml`, `uav_projection.yaml`, `visual_aligning_eval.yaml`, `visual_avoiding_eval.yaml`) = constraint-projection configs (DPCC/MPC safety projection: geometric constraints, bounds, obstacle definitions).

**Why:** Knowing this split avoids hunting for parameters in the wrong place — model/training hyperparameters go in the .py block, constraint geometry goes in the .yaml.

**How to apply:** When adding or tuning an experiment, edit its block in the matching config .py; when changing constraints/projection behavior, edit the matching .yaml. Note the eval `plan*` block's watched args must match the training block's checkpoint path fields. See also [[fmpcc-dev-logs-navigation]] and [[slurm-sbatch-is-real-entrypoint]].
