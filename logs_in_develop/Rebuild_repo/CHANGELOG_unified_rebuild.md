# Changelog: Unified FM-PCC Repository Rebuild

> Tracks user requests, version history, and what changed vs. the current codebase.  
> Main document: [`CONCEPT_unified_rebuild.md`](file:///workspaces/FM-PCC/logs_in_develop/Rebuild_repo/CONCEPT_unified_rebuild.md)

---

## Version History

### v1 — 2026-08-17 (Initial Concept)

#### User Request

> Different ML Models × Different Projectors (DPCC + HardFlow) × Different Environments (Avoiding / Visual Avoiding / Visual Aligning / UAV with 4 sub-cases) × Seeds — as the final agreed-upon submitted repo. Unify API and naming. Output format should stay the same as the old version.

#### What Was Proposed

- Defined the 4-axis experiment matrix: `ML_Model × Projector × Environment × Seed`
- Proposed `fmpcc/` unified Python package replacing ~20 sibling folders
- Registry-driven dispatch pattern (generalised from `mix_uav/models/engine_registry.py`)
- Single `train.py` + `eval.py` entry points
- Backward-compatible output format for `Data_Analysis/` and Colab notebooks
- 5-phase implementation plan (scaffold → port models → port envs → port training → validate)

#### Current Codebase vs. v1 Proposal

| Aspect | Current Codebase | v1 Proposal |
|--------|-----------------|-------------|
| Package structure | ~20 sibling folders (`diffuser/`, `flow_matcher_v3/`, `fm_visual_avoiding/`, ...) | Single `fmpcc/` package |
| Code duplication | 8+ copies of `helpers.py`, `unet1d_temporal_cond.py`, `config.py`, `training.py` | One canonical copy each |
| Train script | Per-folder (`train_FM_v3.py`, `train_mix_uav.py`, ...) | One `scripts/train.py` |
| Eval script | Per-folder | One `scripts/eval.py` |
| Engine dispatch | `mix_visual_aligning/models/engine_registry.py` (partial, one env only) | Universal registry across all 4 axes |
| Output format | Per-generation naming | Same format, deterministic paths |

---

### v2 — 2026-09-07 (Naming & Infra Overhaul)

#### User Request

> 1. The `diffuser` naming should change — into `flow_matcher`, etc.
> 2. The `dpcc-c` needs to change into pure `pcc-c`, etc.
> 3. Aggregate `.py` + `.yaml` configs (maybe already done) and ensure **both** have CLI override — unlike now, where only the `.py` has override.
> 4. `Visual_Unet` name sucks — maybe a real name with detail, a good name.
> 5. The `film_v1` also sucks, and it's weird as fuck that v1 works better than v2 — so we maybe use v1 as the final in the paper. Need fancy names for **both** v1 and v2; I will choose which one. (This might be a duplicate with point 4 — can merge into one "Backbone Naming" section, or keep separate, both fine.)
> 6. Training: check if we already record somewhere what the exact step of `state_latest` and `state_best` are. Maybe hard-save both files. Update this doc.

#### What Changed (v1 → v2)

| KP | What Changed | Current Codebase Status | v2 Proposed |
|----|-------------|------------------------|-------------|
| **KP1**: `diffuser` purge | `diffuser` appears in 50+ files — package names, class names, Data_Analysis keys, config pkl filenames. Came from upstream DPCC, not our paper. | Purged. `flow_matching_ode.py`, `gaussian_diffusion.py`, etc. `ENGINE_ALIASES` maps `'diffuser' → 'ddpm'`. |
| **KP2**: `dpcc-*` → `pcc-*` | Eval YAML uses `dpcc-r`, `dpcc-c`, `dpcc-t`. The "D" = "Diffusion" — misleading when projector works with FM/MF/AF too. | Renamed: `pcc-r`, `pcc-c`, `pcc-t`. `PROJECTOR_ALIASES` for backward compat. |
| **KP3**: Unified config CLI override | `.py` training configs: full CLI override via argparse. `.yaml` eval configs: ad-hoc env vars (`MIX_PROJ_T`, `MIX_EPOCH`) + bespoke per-flag wiring. No systematic mechanism. | `ConfigResolver` merges both with `CLI > env > yaml > py`. Every YAML key auto-becomes a CLI flag. |
| **KP4**: `VisualUNet` rename | `VisualUNet`, `VisualUNetTwoTime` — opaque names. Actually: dual-ResNet encoder → FiLM → 1D temporal U-Net on trajectory tensors `[B, H, 9]`. | `VisionTrajectoryUNet`, `VisionTrajectoryUNetTwoTime`. Folder `visual/` → `conditioning/`. |
| **KP5**: `film_v1`/`film_v2` rename | `film_mode='v1'` (concat bias, "fake FiLM") and `film_mode='v2'` (per-block γ·x+β, "true FiLM"). Meaningless version tags. v1 empirically better than v2. | **ConcatCond** (v1) and **AffineFiLM** (v2). Config key `conditioning_mode` with values `'concat'`/`'affine'`. |
| **KP6**: Checkpoint tracking | No `state_latest.pt` (trainers only write `state_{step}.pt` + `state_best.pt`). No manifest — must `torch.load()` to read what step best was at. `clean_weights.py` confirms the gap. | Hard-save `state_latest.pt` every `log_freq`. Write `checkpoint_manifest.json` (step, loss, timestamp — no GPU needed). |

#### Codebase Evidence Found During v2 Research

| Finding | Location | Relevance |
|---------|----------|-----------|
| `state_latest.pt` does not exist | [`clean_weights.py:11`](file:///workspaces/FM-PCC/tools/clean_weights/clean_weights.py#L11): *"The trainers have no state_latest.pt file"* | Confirms KP6 gap |
| `state_best.pt` contains `'step'` key but requires full load | [`training_twotime.py:496`](file:///workspaces/FM-PCC/mix_visual_aligning/utils/training_twotime.py#L496): `'step': self.step` in `_checkpoint_payload()` | Step IS recorded but not externally readable |
| `find_latest_checkpoint_step()` scans filenames | Copied across 15+ train scripts, e.g. [`train_mix_visual_aligning.py:231`](file:///workspaces/FM-PCC/mix_visual_aligning_test/train_mix_visual_aligning.py#L231) | No `state_latest.pt` to check |
| `film_mode` dispatch in `VisualUNet` | [`visual_unet.py:61-66`](file:///workspaces/FM-PCC/mix_visual_aligning/models/visual_unet.py#L61): `'v1' → UNet1DTemporalCondModel`, `'v2' → UNet1DTemporalFiLMModel` | Confirms v1/v2 architecture difference |
| YAML eval override is ad-hoc | [`eval_mix_visual_aligning.py:2823-2862`](file:///workspaces/FM-PCC/mix_visual_aligning_test/eval_mix_visual_aligning.py#L2823): `--flow-steps`, `--proj-threshold`, `--epoch` each wired separately | Confirms KP3 gap |
| `ENGINE_ALIASES` already partially exists | [`engine_registry.py:95-97`](file:///workspaces/FM-PCC/mix_visual_aligning/models/engine_registry.py#L95): `'ddpm' → 'diffusion'` | v2 inverts this: `'diffusion' → 'ddpm'` as canonical |
| `dpcc-r/c/t` in eval YAML | [`visual_aligning_eval.yaml:67-69`](file:///workspaces/FM-PCC/config/visual_aligning_eval.yaml#L67) | Confirms KP2 rename scope |

---

## KP Summary: Proposed vs. Current Codebase

| KP | Current Codebase | Proposed (Concept) | Status |
|----|-----------------|-------------------|--------|
| **KP1** | `diffuser` naming from upstream in 50+ files | Purged. Proper names + `ENGINE_ALIASES`. | 💡 Concept |
| **KP2** | `dpcc-c/r/t` — "D" prefix misleading for non-diffusion engines | `pcc-c/r/t` + `PROJECTOR_ALIASES`. | 💡 Concept |
| **KP3** | `.py` has CLI override; `.yaml` needs bespoke wiring per param | `ConfigResolver` — both get systematic CLI override. | 💡 Concept |
| **KP4** | `VisualUNet` — opaque, says nothing about architecture | `VisionTrajectoryUNet` — describes dual-ResNet + FiLM + temporal U-Net. | 💡 Concept |
| **KP5** | `film_v1`/`film_v2` — meaningless tags, v1 beats v2 | **ConcatCond** / **AffineFiLM** — proper paper-worthy names. | 💡 Concept |
| **KP6** | No `state_latest.pt`, no manifest, step buried in .pt files | `state_latest.pt` + `checkpoint_manifest.json`. | 💡 Concept |
