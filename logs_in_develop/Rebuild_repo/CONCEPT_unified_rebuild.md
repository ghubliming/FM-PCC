# Concept: Unified FM-PCC Repository Rebuild

> **Status**: CONCEPT / IDEAS — not a finalized plan  
> **Last Updated**: 2026-09-07 (v2)  
> **Location**: `logs_in_develop/Rebuild_repo/`  
> **Changelog**: See [`CHANGELOG_unified_rebuild.md`](file:///workspaces/FM-PCC/logs_in_develop/Rebuild_repo/CHANGELOG_unified_rebuild.md)

---

## 1 — The Experiment Matrix

The paper's claims rest on a **4-axis Cartesian product**:

```
Experiment = ML_Model × Projector × Environment × Seed
```

### Axis 1: ML Model (Generative Engine)

| Key | Full Name | Origin Gen | Backbone Options |
|-----|-----------|------------|------------------|
| `fm` | Flow Matching ODE | Gen11 (FMv3) | U-Net |
| `mf` | MeanFlow ODE | Gen3v6 | U-Net, DiT, MF-DiT |
| `af` | α-Flow ODE | Gen3v7 | U-Net, DiT, SiT |
| `ddpm` | Gaussian Diffusion (baseline) | upstream | U-Net |

### Axis 2: Projector (Control Layer)

| Key | Full Name | Description |
|-----|-----------|-------------|
| `pcc` | PCC Projection | Model-Predictive Control filter enforcing physical constraints |
| `hf` | HardFlow Projection | In-loop constrained sampling via Neural Lyapunov–Projection barrier |

### Axis 3: Environment (Task Domain)

| Key | Full Name | Observation Mode |
|-----|-----------|-----------------|
| `avoiding` | D3IL Obstacle Avoiding | State-only |
| `visual_avoiding` | D3IL Visual Avoiding | Image + State |
| `visual_aligning` | D3IL Visual Aligning | Image + State |
| `uav_circle` | UAV Circle Tracking | State-only |
| `uav_lemniscate` | UAV Lemniscate (∞) Tracking | State-only |
| `uav_helix` | UAV Helix Tracking | State-only |
| `uav_random` | UAV Random Waypoint | State-only |

### Axis 4: Seeds

Default: `[5, 6, 7, 8, 9]` — configurable via CLI `--seeds`.

### Total: `4 × 2 × 7 × 5 = 280 runs` (not all valid — invalid combos rejected by registry)

---

## 2 — Target Architecture

Single `fmpcc/` Python package. Registry-driven dispatch. One `train.py`, one `eval.py`.

```
FM-PCC/
├── fmpcc/                       # THE single package
│   ├── models/
│   │   ├── registry.py          # universal dispatch table
│   │   ├── backbones/           # unet1d.py, dit.py, sit.py, ...
│   │   ├── engines/             # flow_matching_ode.py, meanflow_ode.py, alphaflow_ode.py, gaussian_diffusion.py
│   │   ├── conditioning/        # vision_trajectory_unet.py, vision_trajectory_unet_twotime.py, ...
│   │   └── film/                # concat_cond.py, affine_film.py
│   ├── projectors/              # pcc_projection.py, hardflow_projection.py
│   ├── envs/                    # avoiding.py, visual_avoiding.py, visual_aligning.py, uav.py
│   ├── datasets/                # d3il.py, uav.py, normalization.py
│   ├── sampling/                # policy.py, guides.py
│   ├── utils/                   # config.py, config_resolver.py, training.py, training_twotime.py
│   └── configs/                 # defaults.py, per-env py + yaml
├── scripts/                     # train.py, eval.py, load_results.py
├── tests/
├── config/                      # eval YAML (backward compat)
├── Data_Analysis/               # unchanged
└── Archived_Codes/              # old generation folders
```

### Registry-Driven Dispatch

```python
# fmpcc/models/registry.py  (conceptual sketch)

ENGINES = {
    'fm':   { model: ..., diffusion: ..., trainer: ..., ... },
    'mf':   { model: ..., diffusion: ..., trainer: ..., ... },
    'af':   { model: ..., diffusion: ..., trainer: ..., ... },
    'ddpm': { model: ..., diffusion: ..., trainer: ..., ... },
}

PROJECTORS = {
    'pcc':      { class: ..., supports: ['fm','mf','af','ddpm'] },
    'hardflow': { class: ..., supports: ['fm','mf','af'] },  # not ddpm
}

ENVS = {
    'avoiding':        { dataset: ..., config: ..., visual: False },
    'visual_avoiding': { dataset: ..., config: ..., visual: True  },
    'visual_aligning': { dataset: ..., config: ..., visual: True  },
    'uav':             { dataset: ..., config: ..., visual: False, sub_cases: [...] },
}

ENGINE_ALIASES = { 'diffusion': 'ddpm', 'diffuser': 'ddpm' }
PROJECTOR_ALIASES = { 'dpcc': 'pcc', 'dpcc-r': 'pcc-r', 'dpcc-c': 'pcc-c', 'dpcc-t': 'pcc-t' }
CONDITIONING_ALIASES = { 'v1': 'concat', 'v2': 'affine', 'film_v1': 'concat', 'film_v2': 'affine' }
```

---

## 3 — Unified CLI

```bash
# Training — ONE command, ANY combo
python scripts/train.py --engine fm --projector pcc --env visual_avoiding --seeds 5 6 7 8 9

# Eval — same pattern, full CLI override on BOTH py and yaml params
python scripts/eval.py --engine af --projector pcc --env visual_aligning --seeds 5 6 7 8 9 \
    --nfe 5 10 20 --proj-threshold 0.1 --epoch latest
```

---

## 4 — Unified Config Resolution

Both `.py` (training) and `.yaml` (eval) configs go through a single `ConfigResolver`.

**Precedence**: `CLI flags > env vars > YAML file > .py defaults`

Every key in the YAML automatically becomes a CLI flag (`--key-name`), with types inferred from defaults. No per-param wiring needed.

```python
# fmpcc/utils/config_resolver.py  (conceptual sketch)
class ConfigResolver:
    def __init__(self, py_defaults: dict, yaml_path: str = None):
        # Merge py defaults + yaml
        ...
    
    def add_cli_overrides(self, parser: argparse.ArgumentParser):
        # Auto-register every config key as a --flag
        for key, default in self.base.items():
            cli_key = f'--{key.replace("_", "-")}'
            # type inferred from default value
            ...
    
    def resolve(self, cli_args) -> dict:
        # CLI > env > yaml > py
        ...
```

---

## 5 — Naming Conventions

### 5.1 — Engine Module Naming

| Current | Proposed | Class |
|---------|----------|-------|
| `diffuser/`, `diffuser_visual_*` | **Archived** → `fmpcc/` | — |
| `fm_ode.py` | `flow_matching_ode.py` | `FlowMatchingODE` |
| `mf_ode.py` | `meanflow_ode.py` | `MeanFlowODE`, `MeanFlowEngine` |
| `af_ode.py` | `alphaflow_ode.py` | `AlphaFlowODE`, `AlphaFlowEngine` |
| `ddpm.py` | `gaussian_diffusion.py` | `GaussianDiffusion` |
| `diffusion_config.pkl` | `engine_config.pkl` | — |
| `'diffuser'` (Data_Analysis key) | `'ddpm'` | — |

### 5.2 — Projection Variant Naming

| Current | Proposed | Meaning |
|---------|----------|---------|
| `dpcc-r` | **`pcc-r`** | PCC with **r**eference trajectory selection |
| `dpcc-c` | **`pcc-c`** | PCC with minimum projection **c**ost selection |
| `dpcc-t` | **`pcc-t`** | PCC with **t**emporal consistency selection |
| `dpcc-c-dt*` | **`pcc-c-dt*`** | PCC-c dt ablations |

### 5.3 — Backbone / Conditioning Naming

| Current | Proposed | What It Is |
|---------|----------|------------|
| `VisualUNet` | **`VisionTrajectoryUNet`** | ResNet vision encoder → conditioning → 1D temporal U-Net. Single-time engines (FM, DDPM). |
| `VisualUNetTwoTime` | **`VisionTrajectoryUNetTwoTime`** | Same + two-time (h_mlp) U-Net. JVP-based engines (MF, AF). |
| `visual/` (folder) | **`conditioning/`** | Handles vision encoding + backbone composition |

### 5.4 — FiLM Conditioning Mode Naming

| Current Tag | Proposed Code Name | Proposed Paper Name | Architecture |
|-------------|--------------------|---------------------|-------------|
| `film_mode='v1'` | **`ConcatCond`** | **Concat-Conditioned U-Net** | Visual latent concatenated with time embedding → single additive bias per block |
| `film_mode='v2'` | **`AffineFiLM`** | **Affine-FiLM U-Net** | Visual latent → per-block γ scale + β shift (`γ·h + β`) |

Config key: `film_mode` → `conditioning_mode`, values `'concat'` / `'affine'`.

> **Note**: ConcatCond (v1) empirically outperforms AffineFiLM (v2). Likely ships as the main method in the paper. User's choice which framing.

### 5.5 — Experiment Path Template

```
logs/{env}/{engine}_{projector}/[bb_{backbone}]/[cc_{conditioning}]/H{horizon}_D{nfe}/seed_{seed}/
```

Examples:
```
logs/visual_avoiding/fm_pcc/H64_D10/seed_5/
logs/visual_aligning/mf_pcc/cc_affine/H8_K2/seed_6/
logs/uav_circle/mf_hardflow/bb_dit/H64_D10/seed_7/
```

### 5.6 — Full Name Mapping (Quick Reference)

| Category | Current | Proposed | Alias? |
|----------|---------|----------|--------|
| Package | `diffuser/`, `diffuser_visual_*` | `fmpcc/` | N/A (archived) |
| Engine key | `'diffusion'`, `'diffuser'` | `'ddpm'` | ✅ `ENGINE_ALIASES` |
| Projector key | `'dpcc'` | `'pcc'` | ✅ `PROJECTOR_ALIASES` |
| Projector variants | `'dpcc-r/c/t'` | `'pcc-r/c/t'` | ✅ `PROJECTOR_ALIASES` |
| Conditioning mode | `film_mode='v1'` | `conditioning_mode='concat'` | ✅ `CONDITIONING_ALIASES` |
| Conditioning mode | `film_mode='v2'` | `conditioning_mode='affine'` | ✅ `CONDITIONING_ALIASES` |
| Backbone class | `VisualUNet` | `VisionTrajectoryUNet` | N/A |
| Backbone class | `VisualUNetTwoTime` | `VisionTrajectoryUNetTwoTime` | N/A |
| Config file | `diffusion_config.pkl` | `engine_config.pkl` | Loader reads both |
| Projector file | `dpcc_projection.py` | `pcc_projection.py` | N/A |
| Checkpoint | *(none)* | `state_latest.pt` | N/A (new) |
| Manifest | *(none)* | `checkpoint_manifest.json` | N/A (new) |

---

## 6 — Checkpoint Tracking

### Current Status

| What | How It Works Now | Problem |
|------|-----------------|---------|
| `state_best.pt` | Saved when `test_loss < best_test_loss`. Contains `'step'` key inside dict. | Must `torch.load()` to read the step — needs GPU memory. |
| `state_{step}.pt` | Periodic saves. "Latest" inferred by scanning filenames. | After `clean_weights.py` pruning, surviving highest may be much older than training frontier. |
| `state_latest.pt` | **Does not exist.** | `clean_weights.py`: *"The trainers have no state_latest.pt file."* |

### Proposed

#### A. `state_latest.pt` — hard-saved every `log_freq` steps

```python
def save_latest(self):
    savepath = os.path.join(self.logdir, 'state_latest.pt')
    _atomic_torch_save(self._checkpoint_payload(), savepath)
```

- Survives `clean_weights.py` pruning (not matched by `state_\d+.pt` regex)
- Resume logic checks `state_latest.pt` first, falls back to scanning numbered files

#### B. `checkpoint_manifest.json` — lightweight metadata, no GPU needed

```json
{
  "latest_step": 95000,
  "latest_file": "state_latest.pt",
  "latest_timestamp": "2026-09-07T12:34:56Z",
  "best_step": 82000,
  "best_file": "state_best.pt",
  "best_test_loss": 0.00342,
  "best_timestamp": "2026-09-07T11:15:23Z",
  "periodic_checkpoints": [
    {"step": 10000, "file": "state_10000.pt"},
    {"step": 20000, "file": "state_20000.pt"}
  ],
  "total_train_steps": 100000,
  "engine": "mf",
  "seed": 6
}
```

---

## 7 — Backward Compatibility

- **Output files**: same names, same internal structure (`state_best.pt`, `losses.pkl`, NPZ arrays)
- **Alias tables**: all renames backed by `ENGINE_ALIASES`, `PROJECTOR_ALIASES`, `CONDITIONING_ALIASES`
- **Old checkpoints**: loader inspects `engine` key, accepts missing key with warning
- **Data_Analysis**: accepts both old and new variant spellings

---

## 8 — Open Questions

| # | Question | Impact |
|---|----------|--------|
| Q1 | ConcatCond as **main** + AffineFiLM as **ablation**, or both equal? | Paper framing |
| Q2 | Build on `mix_visual_aligning` or fresh scaffold + cherry-pick? | Implementation |
| Q3 | UAV sub-cases: 4 env keys or `--uav-sub` flag? | CLI design |
| Q4 | Include D3IL baselines in unified framework? | Scope |
| Q5 | Branch naming — `rebuild/unified-api`? | Git |
