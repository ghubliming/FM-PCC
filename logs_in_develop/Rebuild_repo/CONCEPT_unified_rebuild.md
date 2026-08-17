# Concept: Unified FM-PCC Repository Rebuild

> **Status**: DRAFT — awaiting review  
> **Branch**: `rebuild/unified-api` (proposed)  
> **Date**: 2026-08-17  

---

## 1 — Motivation: Why Rebuild

The current repo has grown through **copy-modify isolation** across 15+ generations.
This delivered velocity during research but left behind:

| Pain Point | Evidence |
|---|---|
| **~20 sibling model folders** that duplicate 80-90 % of code | `diffuser/`, `flow_matcher_v3/`, `fm_visual_avoiding/`, `fm_visual_aligning/`, `flow_matcher_v3_uav/`, `mix_uav/`, `mix_visual_aligning/`, `imf_visual_aligning/`, … |
| **Inconsistent naming** | `flow_matcher_v3`, `fm_visual_avoiding`, `diffuser_visual_aligning`, `mix_uav` — 4 different naming schemes for the same conceptual layer |
| **Per-folder test scripts** with divergent CLI APIs | `scripts/train.py`, `FM_v3_test/train_FM_v3.py`, `mix_uav_test/train_mix_uav.py` — each invented its own arg parser |
| **Cross-generation sync burden** | Fixes must be mirrored across 3-5 active siblings (commit messages: "Sync to Gen7/Gen6V4 C4") |
| **No single entry point** for the full experiment matrix | Reviewers / collaborators cannot reproduce all results from one command |

The goal is a **single, clean codebase** that can express the full experiment matrix of the submitted paper while keeping **output format backward-compatible** with existing analysis pipelines (`Data_Analysis/`, Colab notebooks).

---

## 2 — The Experiment Matrix

The paper's claims rest on a **4-axis Cartesian product**:

```
Experiment = ML_Model × Projector × Environment × Seed
```

### 2.1 — Axis 1: ML Model (Generative Engine)

| Key | Full Name | Origin Gen | Backbone Options |
|-----|-----------|------------|------------------|
| `fm` | Flow Matching ODE | Gen11 (FMv3) | U-Net |
| `mf` | MeanFlow ODE | Gen3v6 | U-Net, DiT, MF-DiT |
| `af` | α-Flow ODE | Gen3v7 | U-Net, DiT, SiT |
| `ddpm` | Gaussian Diffusion (DPCC baseline) | DPCC upstream | U-Net |

> **Note**: `mix_uav/models/engine_registry.py` already implements this dispatch for UAV. The rebuild **generalises** the registry to all environments.

### 2.2 — Axis 2: Projector (Control Layer)

| Key | Full Name | Description |
|-----|-----------|-------------|
| `dpcc` | MPC / DPCC Projection | Model-Predictive Control filter that enforces physical constraints on the generated trajectory |
| `hf` | HardFlow Projection | In-loop constrained sampling via the Neural Lyapunov–Projection (NLP) barrier function |

### 2.3 — Axis 3: Environment (Task Domain)

| Key | Full Name | Observation Mode | Sub-cases |
|-----|-----------|-----------------|-----------|
| `avoiding` | D3IL Obstacle Avoiding | State-only | — |
| `visual_avoiding` | D3IL Visual Avoiding | Image + State (FiLM) | — |
| `visual_aligning` | D3IL Visual Aligning | Image + State (FiLM) | — |
| `uav` | UAV Trajectory Tracking | State-only | **4 sub-cases** (see below) |

#### UAV Sub-Cases

| Sub | Environment ID | Description |
|-----|---------------|-------------|
| `uav_circle` | Circle tracking | Circular reference trajectory |
| `uav_lemniscate` | Lemniscate (∞) tracking | Figure-8 reference trajectory |
| `uav_helix` | Helix tracking | 3D helical reference trajectory |
| `uav_random` | Random waypoint | Stochastic waypoint sequences |

### 2.4 — Axis 4: Seeds

Default seed set: `[5, 6, 7, 8, 9]` (5 seeds, matching existing runs).
Configurable via CLI `--seeds` or JSON config file (preserve existing mechanism from `TRAINING_CLI_USAGE.md`).

### 2.5 — Total Experiment Count

```
4 models × 2 projectors × 7 envs × 5 seeds = 280 runs
```

> Not all cells are valid (e.g., DDPM + HardFlow is unsupported — DDPM lacks a velocity field).
> The registry will encode validity constraints explicitly and skip/error on invalid combos.

---

## 3 — Target Architecture

### 3.1 — Directory Layout (proposed)

```
FM-PCC/                          # repo root
├── fmpcc/                       # ← THE single Python package (replaces all siblings)
│   ├── __init__.py
│   ├── models/                  # ALL model code, ONE copy
│   │   ├── __init__.py
│   │   ├── registry.py          # ← generalised engine_registry (from mix_uav)
│   │   ├── backbones/           # network architectures
│   │   │   ├── unet1d.py        # temporal U-Net (from Gen11)
│   │   │   ├── unet1d_twotime.py   # two-time U-Net (from Gen3v6)
│   │   │   ├── unet1d_ddpm.py   # DPCC baseline U-Net
│   │   │   ├── dit.py           # DiT transformer
│   │   │   ├── sit.py           # SiT transformer
│   │   │   └── mlp.py           # value / simple MLP
│   │   ├── engines/             # generative objective wrappers
│   │   │   ├── fm_ode.py        # FlowMatchingODE
│   │   │   ├── mf_ode.py        # MeanFlowODE + MeanFlowEngine
│   │   │   ├── af_ode.py        # AlphaFlowODE + AlphaFlowEngine
│   │   │   └── ddpm.py          # GaussianDiffusion
│   │   └── visual/              # visual conditioning wrappers
│   │       ├── visual_unet.py   # FiLM-conditioned visual U-Net
│   │       ├── visual_unet_twotime.py
│   │       └── visual_diffusion.py   # thin wrappers per engine
│   ├── projectors/              # constraint-enforcement layers
│   │   ├── __init__.py
│   │   ├── dpcc_projection.py   # MPC / DPCC projection
│   │   └── hardflow_projection.py   # HardFlow NLP projection
│   ├── envs/                    # environment-specific adapters
│   │   ├── __init__.py
│   │   ├── avoiding.py          # D3IL avoiding (state)
│   │   ├── visual_avoiding.py   # D3IL visual avoiding
│   │   ├── visual_aligning.py   # D3IL visual aligning
│   │   └── uav.py               # UAV (4 sub-cases parametric)
│   ├── datasets/                # data loading (unified)
│   │   ├── __init__.py
│   │   ├── d3il.py              # D3IL dataset loader
│   │   ├── uav.py               # UAV dataset loader
│   │   └── normalization.py
│   ├── sampling/                # trajectory sampling logic
│   │   ├── __init__.py
│   │   ├── policy.py            # unified Policy class
│   │   └── guides.py            # guidance wrappers
│   ├── utils/                   # shared utilities
│   │   ├── __init__.py
│   │   ├── config.py            # Config class (from diffuser/utils)
│   │   ├── training.py          # base Trainer
│   │   ├── training_twotime.py  # two-time Trainer extension
│   │   ├── serialization.py     # checkpoint save/load
│   │   └── timer.py
│   └── configs/                 # YAML/Python config definitions
│       ├── defaults.py          # shared default hyperparameters
│       ├── avoiding.py
│       ├── visual_avoiding.py
│       ├── visual_aligning.py
│       └── uav.py               # UAV configs (4 sub-cases)
├── scripts/                     # CLI entry points (unified)
│   ├── train.py                 # ONE train script for ALL combos
│   ├── eval.py                  # ONE eval script for ALL combos
│   └── load_results.py          # result loader (backward-compat output)
├── tests/                       # unit / smoke tests
│   ├── test_registry.py
│   ├── test_config_resolution.py
│   └── test_output_compat.py    # assert output format matches legacy
├── config/                      # ← keep for eval YAML backward compat
├── Slurm_Codes/                 # SLURM scripts (updated paths)
├── Data_Analysis/               # unchanged
├── Archived_Codes/              # old generation folders moved here
└── requirements.txt
```

### 3.2 — Key Design Principle: Registry-Driven Dispatch

The core insight from `mix_uav/models/engine_registry.py` is the right pattern — **extend it to be the universal dispatch table** across ALL four axes:

```python
# fmpcc/models/registry.py  (conceptual sketch)

ENGINES = {
    'fm':   { model: ..., diffusion: ..., trainer: ..., ... },
    'mf':   { model: ..., diffusion: ..., trainer: ..., ... },
    'af':   { model: ..., diffusion: ..., trainer: ..., ... },
    'ddpm': { model: ..., diffusion: ..., trainer: ..., ... },
}

PROJECTORS = {
    'dpcc':     { class: ..., supports: ['fm','mf','af','ddpm'] },
    'hardflow': { class: ..., supports: ['fm','mf','af'] },  # not ddpm
}

ENVS = {
    'avoiding':        { dataset: ..., config: ..., visual: False },
    'visual_avoiding': { dataset: ..., config: ..., visual: True  },
    'visual_aligning': { dataset: ..., config: ..., visual: True  },
    'uav':             { dataset: ..., config: ..., visual: False, sub_cases: [...] },
}

def build_experiment(engine, projector, env, seed, **overrides):
    """Single function to construct the full train/eval pipeline."""
    ...
```

### 3.3 — Unified CLI

```bash
# Training — ONE command, ANY combo
python scripts/train.py \
    --engine fm \
    --projector dpcc \
    --env visual_avoiding \
    --seeds 5 6 7 8 9 \
    --use-wandb

# UAV with sub-case
python scripts/train.py \
    --engine mf \
    --projector hardflow \
    --env uav \
    --uav-sub circle \
    --backbone dit \
    --seeds 5 6 7

# Eval — same pattern
python scripts/eval.py \
    --engine af \
    --projector dpcc \
    --env visual_aligning \
    --seeds 5 6 7 8 9 \
    --nfe 5 10 20
```

---

## 4 — Naming Conventions

### 4.1 — Python Package & Module Names

| Rule | Convention | Example |
|------|-----------|---------|
| Package name | `fmpcc` (single, flat) | `import fmpcc` |
| Module names | `snake_case`, descriptive | `fmpcc.models.engines.fm_ode` |
| Class names | `PascalCase` | `FlowMatchingODE`, `MeanFlowEngine` |
| Registry keys | Short `snake_case` strings | `'fm'`, `'mf'`, `'af'`, `'ddpm'` |
| Config keys | `snake_case` | `engine`, `projector`, `env`, `uav_sub` |

### 4.2 — Experiment Path Naming (for logs / checkpoints)

Maintain a **deterministic, human-readable path template**:

```
logs/{env}/{engine}_{projector}/[backbone_bb]/[extra_tokens]/H{horizon}_D{nfe}/seed_{seed}/
```

Examples:
```
logs/visual_avoiding/fm_dpcc/H64_D10/seed_5/
logs/uav_circle/mf_hardflow/bb_dit/dp_0.5/H64_D10/seed_7/
logs/visual_aligning/af_dpcc/bb_sit/H64_D20/seed_9/
logs/avoiding/ddpm_dpcc/K_20/H64/seed_5/
```

### 4.3 — Output File Naming (backward compatible)

These files MUST keep their existing names for analysis pipeline compatibility:

| File | Content | Format |
|------|---------|--------|
| `state_best.pt` | Best checkpoint | PyTorch state dict |
| `state_{step}.pt` | Step checkpoint | PyTorch state dict |
| `losses.pkl` | Training loss history | Pickle |
| `args.json` | Run arguments snapshot | JSON |
| `model_config.pkl` | Model constructor kwargs | Pickle |
| `diffusion_config.pkl` | Diffusion constructor kwargs | Pickle |
| `seeds_config.json` | Seed manifest | JSON |

---

## 5 — Backward Compatibility Strategy

### 5.1 — Output Format Preservation

> **Hard constraint**: existing `Data_Analysis/`, `Results_and_Data_Analysis_Colab_T4/`, and Colab notebooks must work without modification.

Strategy:
- **Checkpoint files**: same names, same internal structure (`state_best.pt`, `losses.pkl`, etc.)
- **NPZ result files**: same array keys and shapes
- **Config pickle files**: same constructor kwargs (but now generated from unified registry)
- **Eval output structure**: `eval_results/` folder with same per-seed, per-K layout

### 5.2 — Migration Path

```
Phase 1: Build `fmpcc/` alongside old folders (both coexist)
Phase 2: Validate output parity — run same configs, diff outputs
Phase 3: Move old folders to `Archived_Codes/`
Phase 4: Update Slurm scripts to point to new `scripts/train.py`
```

---

## 6 — What Gets Merged vs. Deduplicated

### 6.1 — Code That Is Currently Copy-Pasted Across Folders

| Component | Current Copies | Action |
|-----------|---------------|--------|
| `helpers.py` (sinusoidal embeddings, norms) | 8+ copies | → `fmpcc/models/backbones/helpers.py` |
| `unet1d_temporal_cond.py` | 8+ copies | → `fmpcc/models/backbones/unet1d.py` |
| `diffusion.py` (FlowMatchingODE) | 6+ copies | → `fmpcc/models/engines/fm_ode.py` |
| `utils/config.py` (Config class) | 8+ copies | → `fmpcc/utils/config.py` |
| `utils/training.py` (Trainer) | 8+ copies | → `fmpcc/utils/training.py` |
| `sampling/` (Policy, guides) | 8+ copies | → `fmpcc/sampling/` |
| `datasets/` (sequence dataset) | 8+ copies | → `fmpcc/datasets/` |

### 6.2 — Code That Is Genuinely Different Per-Variant

| Component | Unique Per | How Handled |
|-----------|-----------|-------------|
| Visual conditioning wrappers | Env (visual vs state) | Conditional composition in registry |
| Two-time trainer vs one-time | Engine (mf/af vs fm/ddpm) | Trainer subclass selected by registry |
| FiLM U-Net vs plain U-Net | Env (visual vs state) | Separate backbone files, registry selects |
| MPC constraint parameters | Env (avoiding vs UAV physics) | Per-env config files in `fmpcc/configs/` |

---

## 7 — Implementation Approach

### Phase 1: Scaffold & Registry (Week 1)
- [ ] Create `rebuild/unified-api` branch
- [ ] Build `fmpcc/` package skeleton
- [ ] Port `engine_registry.py` → generalised `registry.py` (engines + projectors + envs)
- [ ] Port `utils/config.py` (single canonical copy)
- [ ] Implement unified CLI arg parser in `scripts/train.py`

### Phase 2: Port Models & Engines (Week 2)
- [ ] Port all backbone architectures (deduplicate)
- [ ] Port all 4 engine implementations
- [ ] Port visual conditioning layer
- [ ] Port projector implementations (DPCC, HardFlow)

### Phase 3: Port Environments & Datasets (Week 3)
- [ ] Port dataset loaders (D3IL, UAV)
- [ ] Port environment-specific configs
- [ ] Port sampling / policy code

### Phase 4: Port Training & Eval (Week 4)
- [ ] Unify `train.py` — one script, registry-driven
- [ ] Unify `eval.py` — one script, registry-driven
- [ ] Port `load_results.py` with backward-compat output

### Phase 5: Validation & Migration (Week 5)
- [ ] Run parity tests: old code vs new code, same config → same output
- [ ] Update Slurm scripts
- [ ] Archive old generation folders
- [ ] Update documentation

---

## 8 — Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| **Subtle behavioral drift** during deduplication | Silent metric regression | Phase 5 parity tests: load same checkpoint, run same eval, diff NPZ outputs bit-for-bit |
| **model_config.pkl format** differs between engine arms (U-Net kwargs vs Engine kwargs) | Checkpoint loading breaks | Registry-aware checkpoint loader that inspects `engine` key before unpickling |
| **DDPM K is train-time** | Cannot sweep K at eval time for DDPM | Preserve existing behavior: K in folder name, separate training runs per K |
| **FiLM visual conditioning** wiring differs between envs | Wrong image encoder selected | Registry encodes `visual: True/False` per env, auto-selects wrapper |
| **HardFlow requires velocity field** | DDPM arm crash | `supports_hardflow=False` in DDPM registry row, validation gate at build time |

---

## 9 — Open Questions for Discussion

**Q1**: Should we keep the `config/*.py` Python config files (current approach) or migrate to pure YAML configs? YAML is more declarative but Python configs allow computed defaults (e.g., `af_alpha_end_step = n_train_steps`).

**Q2**: The `mix_uav` and `mix_visual_aligning` folders already implement a partial version of this registry pattern. Should we **build on top of `mix_uav`** as the starting point, or start fresh and cherry-pick?

**Q3**: Should the UAV 4 sub-cases be modeled as 4 separate `env` keys (`uav_circle`, `uav_lemniscate`, `uav_helix`, `uav_random`) or as one `env=uav` with a `--uav-sub` flag? The latter is cleaner but the former keeps the path structure flatter.

**Q4**: Should we also unify the D3IL baseline models (from `d3il_visual_aligning_baseline_test/`) into this same framework, or keep them as a separate external comparison?

**Q5**: Branch naming — `rebuild/unified-api` is proposed. Any preference?

---

## 10 — Success Criteria

1. **Single `train.py`** can launch any cell in the 4-axis experiment matrix
2. **Single `eval.py`** can evaluate any trained checkpoint
3. **Output files** are format-identical to current pipeline outputs
4. **`Data_Analysis/`** scripts work without modification on new outputs
5. **All existing checkpoints** can be loaded by the new code (backward-compat loader)
6. **Zero code duplication** across model/engine/env variants
7. **Comprehensive registry validation** — invalid combos fail fast with clear error messages
